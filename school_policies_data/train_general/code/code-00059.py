                    "Dropped cmap subtable from font '%s':\t"
                    "format %2s, platformID %2s, platEncID %2s",
                    fontIdx,
                    subtable.format,
                    subtable.platformID,
                    subtable.platEncID,
                )
        if format12 is not None:
            chosenCmapTables.append((format12, fontIdx))
        elif format4 is not None:
            chosenCmapTables.append((format4, fontIdx))

        if format14 is not None:
            chosenUvsTables.append(format14)

    # Build the unicode mapping
    merger.cmap = cmap = {}
    fontIndexForGlyph = {}
    glyphSets = [None for f in merger.fonts] if hasattr(merger, "fonts") else None

    for table, fontIdx in chosenCmapTables:
        # handle duplicates
        for uni, gid in table.cmap.items():
            oldgid = cmap.get(uni, None)
            if oldgid is None:
                cmap[uni] = gid
                fontIndexForGlyph[gid] = fontIdx
            elif is_Default_Ignorable(uni) or uni in (0x25CC,):  # U+25CC DOTTED CIRCLE
                continue
            elif oldgid != gid:
                # Char previously mapped to oldgid, now to gid.
                # Record, to fix up in GSUB 'locl' later.
                if merger.duplicateGlyphsPerFont[fontIdx].get(oldgid) is None:
                    if glyphSets is not None:
                        oldFontIdx = fontIndexForGlyph[oldgid]
                        for idx in (fontIdx, oldFontIdx):
                            if glyphSets[idx] is None:
                                glyphSets[idx] = merger.fonts[idx].getGlyphSet()
                        # if _glyphsAreSame(glyphSets[oldFontIdx], glyphSets[fontIdx], oldgid, gid):
                        # 	continue
                    merger.duplicateGlyphsPerFont[fontIdx][oldgid] = gid
                elif merger.duplicateGlyphsPerFont[fontIdx][oldgid] != gid:
                    # Char previously mapped to oldgid but oldgid is already remapped to a different
                    # gid, because of another Unicode character.
                    # TODO: Try harder to do something about these.
                    log.warning(
                        "Dropped mapping from codepoint %#06X to glyphId '%s'", uni, gid
                    )

    merger.uvsDict = computeMegaUvs(merger, chosenUvsTables)


def renameCFFCharStrings(merger, glyphOrder, cffTable):
    """Rename topDictIndex charStrings based on glyphOrder."""
    td = cffTable.cff.topDictIndex[0]

    charStrings = {}
    for i, v in enumerate(td.CharStrings.charStrings.values()):
        glyphName = glyphOrder[i]
        charStrings[glyphName] = v
    td.CharStrings.charStrings = charStrings

    td.charset = list(glyphOrder)
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod, Roozbeh Pournader

from fontTools import ttLib
import fontTools.merge.base
from fontTools.merge.cmap import (
    computeMegaGlyphOrder,
    computeMegaCmap,
    renameCFFCharStrings,
)
from fontTools.merge.layout import layoutPreMerge, layoutPostMerge
from fontTools.merge.options import Options
import fontTools.merge.tables
from fontTools.misc.loggingTools import Timer
from functools import reduce
import sys
import logging

log = logging.getLogger("fontTools.merge")
timer = Timer(logger=logging.getLogger(__name__ + ".timer"), level=logging.INFO)


class Merger(object):
    """Font merger.

    This class merges multiple files into a single OpenType font, taking into
    account complexities such as OpenType layout (``GSUB``/``GPOS``) tables and
    cross-font metrics (for example ``hhea.ascent`` is set to the maximum value
    across all the fonts).

    If multiple glyphs map to the same Unicode value, and the glyphs are considered
    sufficiently different (that is, they differ in any of paths, widths, or
    height), then subsequent glyphs are renamed and a lookup in the ``locl``
    feature will be created to disambiguate them. For example, if the arguments
    are an Arabic font and a Latin font and both contain a set of parentheses,
    the Latin glyphs will be renamed to ``parenleft.1`` and ``parenright.1``,
    and a lookup will be inserted into the to ``locl`` feature (creating it if
    necessary) under the ``latn`` script to substitute ``parenleft`` with
    ``parenleft.1`` etc.

    Restrictions:

    - All fonts must have the same units per em.
    - If duplicate glyph disambiguation takes place as described above then the
      fonts must have a ``GSUB`` table.

    Attributes:
            options: Currently unused.
    """

    def __init__(self, options=None):
        if not options:
            options = Options()

        self.options = options

    def _openFonts(self, fontfiles):
        fonts = [ttLib.TTFont(fontfile) for fontfile in fontfiles]
        for font, fontfile in zip(fonts, fontfiles):
            font._merger__fontfile = fontfile
            font._merger__name = font["name"].getDebugName(4)
        return fonts

    def merge(self, fontfiles):
        """Merges fonts together.

        Args:
                fontfiles: A list of file names to be merged

        Returns:
                A :class:`fontTools.ttLib.TTFont` object. Call the ``save`` method on
                this to write it out to an OTF file.
        """
        #
        # Settle on a mega glyph order.
        #
        fonts = self._openFonts(fontfiles)
        glyphOrders = [list(font.getGlyphOrder()) for font in fonts]
        computeMegaGlyphOrder(self, glyphOrders)

        # Take first input file sfntVersion
        sfntVersion = fonts[0].sfntVersion

        # Reload fonts and set new glyph names on them.
        fonts = self._openFonts(fontfiles)
        for font, glyphOrder in zip(fonts, glyphOrders):
            font.setGlyphOrder(glyphOrder)
            if "CFF " in font:
                renameCFFCharStrings(self, glyphOrder, font["CFF "])

        cmaps = [font["cmap"] for font in fonts]
        self.duplicateGlyphsPerFont = [{} for _ in fonts]
        computeMegaCmap(self, cmaps)

        mega = ttLib.TTFont(sfntVersion=sfntVersion)
        mega.setGlyphOrder(self.glyphOrder)

        for font in fonts:
            self._preMerge(font)

        self.fonts = fonts

        allTags = reduce(set.union, (list(font.keys()) for font in fonts), set())
        allTags.remove("GlyphOrder")

        for tag in sorted(allTags):
            if tag in self.options.drop_tables:
                continue

            with timer("merge '%s'" % tag):
                tables = [font.get(tag, NotImplemented) for font in fonts]

                log.info("Merging '%s'.", tag)
                clazz = ttLib.getTableClass(tag)
                table = clazz(tag).merge(self, tables)
                # XXX Clean this up and use:  table = mergeObjects(tables)

                if table is not NotImplemented and table is not False:
                    mega[tag] = table
                    log.info("Merged '%s'.", tag)
                else:
                    log.info("Dropped '%s'.", tag)

        del self.duplicateGlyphsPerFont
        del self.fonts

        self._postMerge(mega)

        return mega

    def mergeObjects(self, returnTable, logic, tables):
        # Right now we don't use self at all.  Will use in the future
        # for options and logging.

        allKeys = set.union(
            set(),
            *(vars(table).keys() for table in tables if table is not NotImplemented),
        )
        for key in allKeys:
            log.info(" %s", key)
            try:
                mergeLogic = logic[key]
            except KeyError:
                try:
                    mergeLogic = logic["*"]
                except KeyError:
                    raise Exception(
                        "Don't know how to merge key %s of class %s"
                        % (key, returnTable.__class__.__name__)
                    )
            if mergeLogic is NotImplemented:
                continue
            value = mergeLogic(getattr(table, key, NotImplemented) for table in tables)
            if value is not NotImplemented:
                setattr(returnTable, key, value)

        return returnTable

    def _preMerge(self, font):
        layoutPreMerge(font)

    def _postMerge(self, font):
        layoutPostMerge(font)

        if "OS/2" in font:
            # https://github.com/fonttools/fonttools/issues/2538
            # TODO: Add an option to disable this?
            font["OS/2"].recalcAvgCharWidth(font)


__all__ = ["Options", "Merger", "main"]


@timer("make one with everything (TOTAL TIME)")
def main(args=None):
    """Merge multiple fonts into one"""
    from fontTools import configLogger

    if args is None:
        args = sys.argv[1:]

    options = Options()
    args = options.parse_opts(args)
    fontfiles = []
    if options.input_file:
        with open(options.input_file) as inputfile:
            fontfiles = [
                line.strip()
                for line in inputfile.readlines()
                if not line.lstrip().startswith("#")
            ]
    for g in args:
        fontfiles.append(g)

    if len(fontfiles) < 1:
        print(
            "usage: fonttools merge [font1 ... fontN] [--input-file=filelist.txt] [--output-file=merged.ttf] [--import-file=tables.ttx]",
            file=sys.stderr,
        )
        print(
            "                                   [--drop-tables=tags] [--verbose] [--timing]",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print(" font1 ... fontN              Files to merge.", file=sys.stderr)
        print(
            " --input-file=<filename>      Read files to merge from a text file, each path new line. # Comment lines allowed.",
            file=sys.stderr,
        )
        print(
            " --output-file=<filename>     Specify output file name (default: merged.ttf).",
            file=sys.stderr,
        )
        print(
            " --import-file=<filename>     TTX file to import after merging. This can be used to set metadata.",
            file=sys.stderr,
        )
        print(
            " --drop-tables=<table tags>   Comma separated list of table tags to skip, case sensitive.",
            file=sys.stderr,
        )
        print(
            " --verbose                    Output progress information.",
            file=sys.stderr,
        )
        print(" --timing                     Output progress timing.", file=sys.stderr)
        return 1

    configLogger(level=logging.INFO if options.verbose else logging.WARNING)
    if options.timing:
        timer.logger.setLevel(logging.DEBUG)
    else:
        timer.logger.disabled = True

    merger = Merger(options=options)
    font = merger.merge(fontfiles)

    if options.import_file:
        font.importXML(options.import_file)

    with timer("compile and save font"):
        font.save(options.output_file)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod, Roozbeh Pournader

from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging

log = logging.getLogger("fontTools.merge")


def mergeLookupLists(lst):
    # TODO Do smarter merge.
    return sumLists(lst)


def mergeFeatures(lst):
    assert lst
    self = otTables.Feature()
    self.FeatureParams = None
    self.LookupListIndex = mergeLookupLists(
        [l.LookupListIndex for l in lst if l.LookupListIndex]
    )
    self.LookupCount = len(self.LookupListIndex)
    return self


def mergeFeatureLists(lst):
    d = {}
    for l in lst:
        for f in l:
            tag = f.FeatureTag
            if tag not in d:
                d[tag] = []
            d[tag].append(f.Feature)
    ret = []
    for tag in sorted(d.keys()):
        rec = otTables.FeatureRecord()
        rec.FeatureTag = tag
        rec.Feature = mergeFeatures(d[tag])
        ret.append(rec)
    return ret


def mergeLangSyses(lst):
    assert lst

    # TODO Support merging ReqFeatureIndex
    assert all(l.ReqFeatureIndex == 0xFFFF for l in lst)

    self = otTables.LangSys()
    self.LookupOrder = None
    self.ReqFeatureIndex = 0xFFFF
    self.FeatureIndex = mergeFeatureLists(
        [l.FeatureIndex for l in lst if l.FeatureIndex]
    )
    self.FeatureCount = len(self.FeatureIndex)
    return self


def mergeScripts(lst):
    assert lst

    if len(lst) == 1:
        return lst[0]
    langSyses = {}
    for sr in lst:
        for lsr in sr.LangSysRecord:
            if lsr.LangSysTag not in langSyses:
                langSyses[lsr.LangSysTag] = []
            langSyses[lsr.LangSysTag].append(lsr.LangSys)
    lsrecords = []
    for tag, langSys_list in sorted(langSyses.items()):
        lsr = otTables.LangSysRecord()
        lsr.LangSys = mergeLangSyses(langSys_list)
        lsr.LangSysTag = tag
        lsrecords.append(lsr)

    self = otTables.Script()
    self.LangSysRecord = lsrecords
    self.LangSysCount = len(lsrecords)
    dfltLangSyses = [s.DefaultLangSys for s in lst if s.DefaultLangSys]
    if dfltLangSyses:
        self.DefaultLangSys = mergeLangSyses(dfltLangSyses)
    else:
        self.DefaultLangSys = None
    return self


def mergeScriptRecords(lst):
    d = {}
    for l in lst:
        for s in l:
            tag = s.ScriptTag
            if tag not in d:
                d[tag] = []
            d[tag].append(s.Script)
    ret = []
    for tag in sorted(d.keys()):
        rec = otTables.ScriptRecord()
        rec.ScriptTag = tag
        rec.Script = mergeScripts(d[tag])
        ret.append(rec)
    return ret


otTables.ScriptList.mergeMap = {
    "ScriptCount": lambda lst: None,  # TODO
    "ScriptRecord": mergeScriptRecords,
}
otTables.BaseScriptList.mergeMap = {
    "BaseScriptCount": lambda lst: None,  # TODO
    # TODO: Merge duplicate entries
    "BaseScriptRecord": lambda lst: sorted(
        sumLists(lst), key=lambda s: s.BaseScriptTag
    ),
}

otTables.FeatureList.mergeMap = {
    "FeatureCount": sum,
    "FeatureRecord": lambda lst: sorted(sumLists(lst), key=lambda s: s.FeatureTag),
}

otTables.LookupList.mergeMap = {
    "LookupCount": sum,
    "Lookup": sumLists,
}

otTables.Coverage.mergeMap = {
    "Format": min,
    "glyphs": sumLists,
}

otTables.ClassDef.mergeMap = {
    "Format": min,
    "classDefs": sumDicts,
}

otTables.LigCaretList.mergeMap = {
    "Coverage": mergeObjects,
    "LigGlyphCount": sum,
    "LigGlyph": sumLists,
}

otTables.AttachList.mergeMap = {
    "Coverage": mergeObjects,
    "GlyphCount": sum,
    "AttachPoint": sumLists,
}

# XXX Renumber MarkFilterSets of lookups
otTables.MarkGlyphSetsDef.mergeMap = {
    "MarkSetTableFormat": equal,
    "MarkSetCount": sum,
    "Coverage": sumLists,
}

otTables.Axis.mergeMap = {
    "*": mergeObjects,
}

# XXX Fix BASE table merging
otTables.BaseTagList.mergeMap = {
    "BaseTagCount": sum,
    "BaselineTag": sumLists,
}

otTables.GDEF.mergeMap = otTables.GSUB.mergeMap = otTables.GPOS.mergeMap = (
    otTables.BASE.mergeMap
) = otTables.JSTF.mergeMap = otTables.MATH.mergeMap = {
    "*": mergeObjects,
    "Version": max,
}

ttLib.getTableClass("GDEF").mergeMap = ttLib.getTableClass("GSUB").mergeMap = (
    ttLib.getTableClass("GPOS").mergeMap
) = ttLib.getTableClass("BASE").mergeMap = ttLib.getTableClass(
    "JSTF"
).mergeMap = ttLib.getTableClass(
    "MATH"
).mergeMap = {
    "tableTag": onlyExisting(equal),  # XXX clean me up
    "table": mergeObjects,
}


@add_method(ttLib.getTableClass("GSUB"))
def merge(self, m, tables):
    assert len(tables) == len(m.duplicateGlyphsPerFont)
    for i, (table, dups) in enumerate(zip(tables, m.duplicateGlyphsPerFont)):
        if not dups:
            continue
        if table is None or table is NotImplemented:
            log.warning(
                "Have non-identical duplicates to resolve for '%s' but no GSUB. Are duplicates intended?: %s",
                m.fonts[i]._merger__name,
                dups,
            )
            continue

        synthFeature = None
        synthLookup = None
        for script in table.table.ScriptList.ScriptRecord:
            if script.ScriptTag == "DFLT":
                continue  # XXX
            for langsys in [script.Script.DefaultLangSys] + [
                l.LangSys for l in script.Script.LangSysRecord
            ]:
                if langsys is None:
                    continue  # XXX Create!
                feature = [v for v in langsys.FeatureIndex if v.FeatureTag == "locl"]
                assert len(feature) <= 1
                if feature:
                    feature = feature[0]
                else:
                    if not synthFeature:
                        synthFeature = otTables.FeatureRecord()
                        synthFeature.FeatureTag = "locl"
                        f = synthFeature.Feature = otTables.Feature()
                        f.FeatureParams = None
                        f.LookupCount = 0
                        f.LookupListIndex = []
                        table.table.FeatureList.FeatureRecord.append(synthFeature)
                        table.table.FeatureList.FeatureCount += 1
                    feature = synthFeature
                    langsys.FeatureIndex.append(feature)
                    langsys.FeatureIndex.sort(key=lambda v: v.FeatureTag)

                if not synthLookup:
                    subtable = otTables.SingleSubst()
                    subtable.mapping = dups
                    synthLookup = otTables.Lookup()
                    synthLookup.LookupFlag = 0
                    synthLookup.LookupType = 1
                    synthLookup.SubTableCount = 1
                    synthLookup.SubTable = [subtable]
                    if table.table.LookupList is None:
                        # mtiLib uses None as default value for LookupList,
                        # while feaLib points to an empty array with count 0
                        # TODO: make them do the same
                        table.table.LookupList = otTables.LookupList()
                        table.table.LookupList.Lookup = []
                        table.table.LookupList.LookupCount = 0
                    table.table.LookupList.Lookup.append(synthLookup)
                    table.table.LookupList.LookupCount += 1

                if feature.Feature.LookupListIndex[:1] != [synthLookup]:
                    feature.Feature.LookupListIndex[:0] = [synthLookup]
                    feature.Feature.LookupCount += 1

    DefaultTable.merge(self, m, tables)
    return self


@add_method(
    otTables.SingleSubst,
    otTables.MultipleSubst,
    otTables.AlternateSubst,
    otTables.LigatureSubst,
    otTables.ReverseChainSingleSubst,
    otTables.SinglePos,
    otTables.PairPos,
    otTables.CursivePos,
    otTables.MarkBasePos,
    otTables.MarkLigPos,
    otTables.MarkMarkPos,
)
def mapLookups(self, lookupMap):
    pass


# Copied and trimmed down from subset.py
@add_method(
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def __merge_classify_context(self):
    class ContextHelper(object):
        def __init__(self, klass, Format):
            if klass.__name__.endswith("Subst"):
                Typ = "Sub"
                Type = "Subst"
            else:
                Typ = "Pos"
                Type = "Pos"
            if klass.__name__.startswith("Chain"):
                Chain = "Chain"
            else:
                Chain = ""
            ChainTyp = Chain + Typ

            self.Typ = Typ
            self.Type = Type
            self.Chain = Chain
            self.ChainTyp = ChainTyp

            self.LookupRecord = Type + "LookupRecord"

            if Format == 1:
                self.Rule = ChainTyp + "Rule"
                self.RuleSet = ChainTyp + "RuleSet"
            elif Format == 2:
                self.Rule = ChainTyp + "ClassRule"
                self.RuleSet = ChainTyp + "ClassSet"

    if self.Format not in [1, 2, 3]:
        return None  # Don't shoot the messenger; let it go
    if not hasattr(self.__class__, "_merge__ContextHelpers"):
        self.__class__._merge__ContextHelpers = {}
    if self.Format not in self.__class__._merge__ContextHelpers:
        helper = ContextHelper(self.__class__, self.Format)
        self.__class__._merge__ContextHelpers[self.Format] = helper
    return self.__class__._merge__ContextHelpers[self.Format]


@add_method(
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def mapLookups(self, lookupMap):
    c = self.__merge_classify_context()

    if self.Format in [1, 2]:
        for rs in getattr(self, c.RuleSet):
            if not rs:
                continue
            for r in getattr(rs, c.Rule):
                if not r:
                    continue
                for ll in getattr(r, c.LookupRecord):
                    if not ll:
                        continue
                    ll.LookupListIndex = lookupMap[ll.LookupListIndex]
    elif self.Format == 3:
        for ll in getattr(self, c.LookupRecord):
            if not ll:
                continue
            ll.LookupListIndex = lookupMap[ll.LookupListIndex]
    else:
        assert 0, "unknown format: %s" % self.Format


@add_method(otTables.ExtensionSubst, otTables.ExtensionPos)
def mapLookups(self, lookupMap):
    if self.Format == 1:
        self.ExtSubTable.mapLookups(lookupMap)
    else:
        assert 0, "unknown format: %s" % self.Format


@add_method(otTables.Lookup)
def mapLookups(self, lookupMap):
    for st in self.SubTable:
        if not st:
            continue
        st.mapLookups(lookupMap)


@add_method(otTables.LookupList)
def mapLookups(self, lookupMap):
    for l in self.Lookup:
        if not l:
            continue
        l.mapLookups(lookupMap)


@add_method(otTables.Lookup)
def mapMarkFilteringSets(self, markFilteringSetMap):
    if self.LookupFlag & 0x0010:
        self.MarkFilteringSet = markFilteringSetMap[self.MarkFilteringSet]


@add_method(otTables.LookupList)
def mapMarkFilteringSets(self, markFilteringSetMap):
    for l in self.Lookup:
        if not l:
            continue
        l.mapMarkFilteringSets(markFilteringSetMap)


@add_method(otTables.Feature)
def mapLookups(self, lookupMap):
    self.LookupListIndex = [lookupMap[i] for i in self.LookupListIndex]


@add_method(otTables.FeatureList)
def mapLookups(self, lookupMap):
    for f in self.FeatureRecord:
        if not f or not f.Feature:
            continue
        f.Feature.mapLookups(lookupMap)


@add_method(otTables.DefaultLangSys, otTables.LangSys)
def mapFeatures(self, featureMap):
    self.FeatureIndex = [featureMap[i] for i in self.FeatureIndex]
    if self.ReqFeatureIndex != 65535:
        self.ReqFeatureIndex = featureMap[self.ReqFeatureIndex]


@add_method(otTables.Script)
def mapFeatures(self, featureMap):
    if self.DefaultLangSys:
        self.DefaultLangSys.mapFeatures(featureMap)
    for l in self.LangSysRecord:
        if not l or not l.LangSys:
            continue
        l.LangSys.mapFeatures(featureMap)


@add_method(otTables.ScriptList)
def mapFeatures(self, featureMap):
    for s in self.ScriptRecord:
        if not s or not s.Script:
            continue
        s.Script.mapFeatures(featureMap)


def layoutPreMerge(font):
    # Map indices to references

    GDEF = font.get("GDEF")
    GSUB = font.get("GSUB")
    GPOS = font.get("GPOS")

    for t in [GSUB, GPOS]:
        if not t:
            continue

        if t.table.LookupList:
            lookupMap = {i: v for i, v in enumerate(t.table.LookupList.Lookup)}
            t.table.LookupList.mapLookups(lookupMap)
            t.table.FeatureList.mapLookups(lookupMap)

            if (
                GDEF
                and GDEF.table.Version >= 0x00010002
                and GDEF.table.MarkGlyphSetsDef
            ):
                markFilteringSetMap = {
                    i: v for i, v in enumerate(GDEF.table.MarkGlyphSetsDef.Coverage)
                }
                t.table.LookupList.mapMarkFilteringSets(markFilteringSetMap)

        if t.table.FeatureList and t.table.ScriptList:
            featureMap = {i: v for i, v in enumerate(t.table.FeatureList.FeatureRecord)}
            t.table.ScriptList.mapFeatures(featureMap)

    # TODO FeatureParams nameIDs


def layoutPostMerge(font):
    # Map references back to indices

    GDEF = font.get("GDEF")
    GSUB = font.get("GSUB")
    GPOS = font.get("GPOS")

    for t in [GSUB, GPOS]:
        if not t:
            continue

        if t.table.FeatureList and t.table.ScriptList:
            # Collect unregistered (new) features.
            featureMap = GregariousIdentityDict(t.table.FeatureList.FeatureRecord)
            t.table.ScriptList.mapFeatures(featureMap)

            # Record used features.
            featureMap = AttendanceRecordingIdentityDict(
                t.table.FeatureList.FeatureRecord
            )
            t.table.ScriptList.mapFeatures(featureMap)
            usedIndices = featureMap.s

            # Remove unused features
            t.table.FeatureList.FeatureRecord = [
                f
                for i, f in enumerate(t.table.FeatureList.FeatureRecord)
                if i in usedIndices
            ]

            # Map back to indices.
            featureMap = NonhashableDict(t.table.FeatureList.FeatureRecord)
            t.table.ScriptList.mapFeatures(featureMap)

            t.table.FeatureList.FeatureCount = len(t.table.FeatureList.FeatureRecord)

        if t.table.LookupList:
            # Collect unregistered (new) lookups.
            lookupMap = GregariousIdentityDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)

            # Record used lookups.
            lookupMap = AttendanceRecordingIdentityDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)
            usedIndices = lookupMap.s

            # Remove unused lookups
            t.table.LookupList.Lookup = [
                l for i, l in enumerate(t.table.LookupList.Lookup) if i in usedIndices
            ]

            # Map back to indices.
            lookupMap = NonhashableDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)

            t.table.LookupList.LookupCount = len(t.table.LookupList.Lookup)

            if GDEF and GDEF.table.Version >= 0x00010002:
                markFilteringSetMap = NonhashableDict(
                    GDEF.table.MarkGlyphSetsDef.Coverage
                )
                t.table.LookupList.mapMarkFilteringSets(markFilteringSetMap)

    # TODO FeatureParams nameIDs
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod, Roozbeh Pournader

from fontTools import ttLib, cffLib
from fontTools.misc.psCharStrings import T2WidthExtractor
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.cmap import computeMegaCmap
from fontTools.merge.util import *
import logging

log = logging.getLogger("fontTools.merge")


ttLib.getTableClass("maxp").mergeMap = {
    "*": max,
    "tableTag": equal,
    "tableVersion": equal,
    "numGlyphs": sum,
    "maxStorage": first,
    "maxFunctionDefs": first,
    "maxInstructionDefs": first,
    # TODO When we correctly merge hinting data, update these values:
    # maxFunctionDefs, maxInstructionDefs, maxSizeOfInstructions
}

headFlagsMergeBitMap = {
    "size": 16,
    "*": bitwise_or,
    1: bitwise_and,  # Baseline at y = 0
    2: bitwise_and,  # lsb at x = 0
    3: bitwise_and,  # Force ppem to integer values. FIXME?
    5: bitwise_and,  # Font is vertical
    6: lambda bit: 0,  # Always set to zero
    11: bitwise_and,  # Font data is 'lossless'
    13: bitwise_and,  # Optimized for ClearType
    14: bitwise_and,  # Last resort font. FIXME? equal or first may be better
    15: lambda bit: 0,  # Always set to zero
}

ttLib.getTableClass("head").mergeMap = {
    "tableTag": equal,
    "tableVersion": max,
    "fontRevision": max,
    "checkSumAdjustment": lambda lst: 0,  # We need *something* here
    "magicNumber": equal,
    "flags": mergeBits(headFlagsMergeBitMap),
    "unitsPerEm": equal,
    "created": current_time,
    "modified": current_time,
    "xMin": min,
    "yMin": min,
    "xMax": max,
    "yMax": max,
    "macStyle": first,
    "lowestRecPPEM": max,
    "fontDirectionHint": lambda lst: 2,
    "indexToLocFormat": first,
    "glyphDataFormat": equal,
}

ttLib.getTableClass("hhea").mergeMap = {
    "*": equal,
    "tableTag": equal,
    "tableVersion": max,
    "ascent": max,
    "descent": min,
    "lineGap": max,
    "advanceWidthMax": max,
    "minLeftSideBearing": min,
    "minRightSideBearing": min,
    "xMaxExtent": max,
    "caretSlopeRise": first,
    "caretSlopeRun": first,
    "caretOffset": first,
    "numberOfHMetrics": recalculate,
}

ttLib.getTableClass("vhea").mergeMap = {
    "*": equal,
    "tableTag": equal,
    "tableVersion": max,
    "ascent": max,
    "descent": min,
    "lineGap": max,
    "advanceHeightMax": max,
    "minTopSideBearing": min,
    "minBottomSideBearing": min,
    "yMaxExtent": max,
    "caretSlopeRise": first,
    "caretSlopeRun": first,
    "caretOffset": first,
    "numberOfVMetrics": recalculate,
}

os2FsTypeMergeBitMap = {
    "size": 16,
    "*": lambda bit: 0,
    1: bitwise_or,  # no embedding permitted
    2: bitwise_and,  # allow previewing and printing documents
    3: bitwise_and,  # allow editing documents
    8: bitwise_or,  # no subsetting permitted
    9: bitwise_or,  # no embedding of outlines permitted
}


def mergeOs2FsType(lst):
    lst = list(lst)
    if all(item == 0 for item in lst):
        return 0

    # Compute least restrictive logic for each fsType value
    for i in range(len(lst)):
        # unset bit 1 (no embedding permitted) if either bit 2 or 3 is set
        if lst[i] & 0x000C:
            lst[i] &= ~0x0002
        # set bit 2 (allow previewing) if bit 3 is set (allow editing)
        elif lst[i] & 0x0008:
            lst[i] |= 0x0004
        # set bits 2 and 3 if everything is allowed
        elif lst[i] == 0:
            lst[i] = 0x000C

    fsType = mergeBits(os2FsTypeMergeBitMap)(lst)
    # unset bits 2 and 3 if bit 1 is set (some font is "no embedding")
    if fsType & 0x0002:
        fsType &= ~0x000C
    return fsType


ttLib.getTableClass("OS/2").mergeMap = {
    "*": first,
    "tableTag": equal,
    "version": max,
    "xAvgCharWidth": first,  # Will be recalculated at the end on the merged font
    "fsType": mergeOs2FsType,  # Will be overwritten
    "panose": first,  # FIXME: should really be the first Latin font
    "ulUnicodeRange1": bitwise_or,
    "ulUnicodeRange2": bitwise_or,
    "ulUnicodeRange3": bitwise_or,
    "ulUnicodeRange4": bitwise_or,
    "fsFirstCharIndex": min,
    "fsLastCharIndex": max,
    "sTypoAscender": max,
    "sTypoDescender": min,
    "sTypoLineGap": max,
    "usWinAscent": max,
    "usWinDescent": max,
    # Version 1
    "ulCodePageRange1": onlyExisting(bitwise_or),
    "ulCodePageRange2": onlyExisting(bitwise_or),
    # Version 2, 3, 4
    "sxHeight": onlyExisting(max),
    "sCapHeight": onlyExisting(max),
    "usDefaultChar": onlyExisting(first),
    "usBreakChar": onlyExisting(first),
    "usMaxContext": onlyExisting(max),
    # version 5
    "usLowerOpticalPointSize": onlyExisting(min),
    "usUpperOpticalPointSize": onlyExisting(max),
}


@add_method(ttLib.getTableClass("OS/2"))
def merge(self, m, tables):
    DefaultTable.merge(self, m, tables)
    if self.version < 2:
        # bits 8 and 9 are reserved and should be set to zero
        self.fsType &= ~0x0300
    if self.version >= 3:
        # Only one of bits 1, 2, and 3 may be set. We already take
        # care of bit 1 implications in mergeOs2FsType. So unset
        # bit 2 if bit 3 is already set.
        if self.fsType & 0x0008:
            self.fsType &= ~0x0004
    return self


ttLib.getTableClass("post").mergeMap = {
    "*": first,
    "tableTag": equal,
    "formatType": max,
    "isFixedPitch": min,
    "minMemType42": max,
    "maxMemType42": lambda lst: 0,
    "minMemType1": max,
    "maxMemType1": lambda lst: 0,
    "mapping": onlyExisting(sumDicts),
    "extraNames": lambda lst: [],
}

ttLib.getTableClass("vmtx").mergeMap = ttLib.getTableClass("hmtx").mergeMap = {
    "tableTag": equal,
    "metrics": sumDicts,
}

ttLib.getTableClass("name").mergeMap = {
    "tableTag": equal,
    "names": first,  # FIXME? Does mixing name records make sense?
}

ttLib.getTableClass("loca").mergeMap = {
    "*": recalculate,
    "tableTag": equal,
}

ttLib.getTableClass("glyf").mergeMap = {
    "tableTag": equal,
    "glyphs": sumDicts,
    "glyphOrder": sumLists,
    "_reverseGlyphOrder": recalculate,
    "axisTags": equal,
}


@add_method(ttLib.getTableClass("glyf"))
def merge(self, m, tables):
    for i, table in enumerate(tables):
        for g in table.glyphs.values():
            if i:
                # Drop hints for all but first font, since
                # we don't map functions / CVT values.
                g.removeHinting()
            # Expand composite glyphs to load their
            # composite glyph names.
            if g.isComposite():
                g.expand(table)
    return DefaultTable.merge(self, m, tables)


ttLib.getTableClass("prep").mergeMap = lambda self, lst: first(lst)
ttLib.getTableClass("fpgm").mergeMap = lambda self, lst: first(lst)
ttLib.getTableClass("cvt ").mergeMap = lambda self, lst: first(lst)
ttLib.getTableClass("gasp").mergeMap = lambda self, lst: first(
    lst
)  # FIXME? Appears irreconcilable


@add_method(ttLib.getTableClass("CFF "))
def merge(self, m, tables):
    if any(hasattr(table.cff[0], "FDSelect") for table in tables):
        raise NotImplementedError("Merging CID-keyed CFF tables is not supported yet")

    for table in tables:
        table.cff.desubroutinize()

    newcff = tables[0]
    newfont = newcff.cff[0]
    private = newfont.Private
    newDefaultWidthX, newNominalWidthX = private.defaultWidthX, private.nominalWidthX
    storedNamesStrings = []
    glyphOrderStrings = []
    glyphOrder = set(newfont.getGlyphOrder())

    for name in newfont.strings.strings:
        if name not in glyphOrder:
            storedNamesStrings.append(name)
        else:
            glyphOrderStrings.append(name)

    chrset = list(newfont.charset)
    newcs = newfont.CharStrings
    log.debug("FONT 0 CharStrings: %d.", len(newcs))

    for i, table in enumerate(tables[1:], start=1):
        font = table.cff[0]
        defaultWidthX, nominalWidthX = (
            font.Private.defaultWidthX,
            font.Private.nominalWidthX,
        )
        widthsDiffer = (
            defaultWidthX != newDefaultWidthX or nominalWidthX != newNominalWidthX
        )
        font.Private = private
        fontGlyphOrder = set(font.getGlyphOrder())
        for name in font.strings.strings:
            if name in fontGlyphOrder:
                glyphOrderStrings.append(name)
        cs = font.CharStrings
        gs = table.cff.GlobalSubrs
        log.debug("Font %d CharStrings: %d.", i, len(cs))
        chrset.extend(font.charset)
        if newcs.charStringsAreIndexed:
            for i, name in enumerate(cs.charStrings, start=len(newcs)):
                newcs.charStrings[name] = i
                newcs.charStringsIndex.items.append(None)
        for name in cs.charStrings:
            if widthsDiffer:
                c = cs[name]
                defaultWidthXToken = object()
                extractor = T2WidthExtractor([], [], nominalWidthX, defaultWidthXToken)
                extractor.execute(c)
                width = extractor.width
                if width is not defaultWidthXToken:
                    # The following will be wrong if the width is added
                    # by a subroutine. Ouch!
                    c.program.pop(0)
                else:
                    width = defaultWidthX
                if width != newDefaultWidthX:
                    c.program.insert(0, width - newNominalWidthX)
            newcs[name] = cs[name]

    newfont.charset = chrset
    newfont.numGlyphs = len(chrset)
    newfont.strings.strings = glyphOrderStrings + storedNamesStrings

    return newcff


@add_method(ttLib.getTableClass("cmap"))
def merge(self, m, tables):
    if not hasattr(m, "cmap"):
        computeMegaCmap(m, tables)
    cmap = m.cmap

    cmapBmpOnly = {uni: gid for uni, gid in cmap.items() if uni <= 0xFFFF}
    self.tables = []
    module = ttLib.getTableModule("cmap")
    if len(cmapBmpOnly) != len(cmap):
        # format-12 required.
        cmapTable = module.cmap_classes[12](12)
        cmapTable.platformID = 3
        cmapTable.platEncID = 10
        cmapTable.language = 0
        cmapTable.cmap = cmap
        self.tables.append(cmapTable)
    # always create format-4
    cmapTable = module.cmap_classes[4](4)
    cmapTable.platformID = 3
    cmapTable.platEncID = 1
    cmapTable.language = 0
    cmapTable.cmap = cmapBmpOnly
    # ordered by platform then encoding
    self.tables.insert(0, cmapTable)

    uvsDict = m.uvsDict
    if uvsDict:
        # format-14
        uvsTable = module.cmap_classes[14](14)
        uvsTable.platformID = 0
        uvsTable.platEncID = 5
        uvsTable.language = 0
        uvsTable.cmap = {}
        uvsTable.uvsDict = uvsDict
        # ordered by platform then encoding
        self.tables.insert(0, uvsTable)
    self.tableVersion = 0
    self.numSubTables = len(self.tables)
    return self
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod, Roozbeh Pournader

from fontTools.ttLib.tables.DefaultTable import DefaultTable
import logging

log = logging.getLogger("fontTools.merge")


def add_method(*clazzes, **kwargs):
    """Returns a decorator function that adds a new method to one or
    more classes."""
    allowDefault = kwargs.get("allowDefaultTable", False)

    def wrapper(method):
        done = []
        for clazz in clazzes:
            if clazz in done:
                continue  # Support multiple names of a clazz
            done.append(clazz)
            assert allowDefault or clazz != DefaultTable, "Oops, table class not found."
            assert (
                method.__name__ not in clazz.__dict__
            ), "Oops, class '%s' has method '%s'." % (clazz.__name__, method.__name__)
            setattr(clazz, method.__name__, method)
        return None

    return wrapper


def mergeObjects(lst):
    lst = [item for item in lst if item is not NotImplemented]
    if not lst:
        return NotImplemented
    lst = [item for item in lst if item is not None]
    if not lst:
        return None

    clazz = lst[0].__class__
    assert all(type(item) == clazz for item in lst), lst

    logic = clazz.mergeMap
    returnTable = clazz()
    returnDict = {}

    allKeys = set.union(set(), *(vars(table).keys() for table in lst))
    for key in allKeys:
        try:
            mergeLogic = logic[key]
        except KeyError:
            try:
                mergeLogic = logic["*"]
            except KeyError:
                raise Exception(
                    "Don't know how to merge key %s of class %s" % (key, clazz.__name__)
                )
        if mergeLogic is NotImplemented:
            continue
        value = mergeLogic(getattr(table, key, NotImplemented) for table in lst)
        if value is not NotImplemented:
            returnDict[key] = value

    returnTable.__dict__ = returnDict

    return returnTable


@add_method(DefaultTable, allowDefaultTable=True)
def merge(self, m, tables):
    if not hasattr(self, "mergeMap"):
        log.info("Don't know how to merge '%s'.", self.tableTag)
        return NotImplemented

    logic = self.mergeMap

    if isinstance(logic, dict):
        return m.mergeObjects(self, self.mergeMap, tables)
    else:
        return logic(tables)
<|endoftext|># FontDame-to-FontTools for OpenType Layout tables
#
# Source language spec is available at:
# http://monotype.github.io/OpenType_Table_Source/otl_source.html
# https://github.com/Monotype/OpenType_Table_Source/

from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging


class MtiLibError(Exception):
    pass


class ReferenceNotFoundError(MtiLibError):
    pass


class FeatureNotFoundError(ReferenceNotFoundError):
    pass


class LookupNotFoundError(ReferenceNotFoundError):
    pass


log = logging.getLogger("fontTools.mtiLib")


def makeGlyph(s):
    if s[:2] in ["U ", "u "]:
        return ttLib.TTFont._makeGlyphName(int(s[2:], 16))
    elif s[:2] == "# ":
        return "glyph%.5d" % int(s[2:])
    assert s.find(" ") < 0, "Space found in glyph name: %s" % s
    assert s, "Glyph name is empty"
    return s


def makeGlyphs(l):
    return [makeGlyph(g) for g in l]


def mapLookup(sym, mapping):
    # Lookups are addressed by name.  So resolved them using a map if available.
    # Fallback to parsing as lookup index if a map isn't provided.
    if mapping is not None:
        try:
            idx = mapping[sym]
        except KeyError:
            raise LookupNotFoundError(sym)
    else:
        idx = int(sym)
    return idx


def mapFeature(sym, mapping):
    # Features are referenced by index according the spec.  So, if symbol is an
    # integer, use it directly.  Otherwise look up in the map if provided.
    try:
        idx = int(sym)
    except ValueError:
        try:
            idx = mapping[sym]
        except KeyError:
            raise FeatureNotFoundError(sym)
    return idx


def setReference(mapper, mapping, sym, setter, collection, key):
    try:
        mapped = mapper(sym, mapping)
    except ReferenceNotFoundError as e:
        try:
            if mapping is not None:
                mapping.addDeferredMapping(
                    lambda ref: setter(collection, key, ref), sym, e
                )
                return
        except AttributeError:
            pass
        raise
    setter(collection, key, mapped)


class DeferredMapping(dict):
    def __init__(self):
        self._deferredMappings = []

    def addDeferredMapping(self, setter, sym, e):
        log.debug("Adding deferred mapping for symbol '%s' %s", sym, type(e).__name__)
        self._deferredMappings.append((setter, sym, e))

    def applyDeferredMappings(self):
        for setter, sym, e in self._deferredMappings:
            log.debug(
                "Applying deferred mapping for symbol '%s' %s", sym, type(e).__name__
            )
            try:
                mapped = self[sym]
            except KeyError:
                raise e
            setter(mapped)
            log.debug("Set to %s", mapped)
        self._deferredMappings = []


def parseScriptList(lines, featureMap=None):
    self = ot.ScriptList()
    records = []
    with lines.between("script table"):
        for line in lines:
            while len(line) < 4:
                line.append("")
            scriptTag, langSysTag, defaultFeature, features = line
            log.debug("Adding script %s language-system %s", scriptTag, langSysTag)

            langSys = ot.LangSys()
            langSys.LookupOrder = None
            if defaultFeature:
                setReference(
                    mapFeature,
                    featureMap,
                    defaultFeature,
                    setattr,
                    langSys,
                    "ReqFeatureIndex",
                )
            else:
                langSys.ReqFeatureIndex = 0xFFFF
            syms = stripSplitComma(features)
            langSys.FeatureIndex = theList = [3] * len(syms)
            for i, sym in enumerate(syms):
                setReference(mapFeature, featureMap, sym, setitem, theList, i)
            langSys.FeatureCount = len(langSys.FeatureIndex)

            script = [s for s in records if s.ScriptTag == scriptTag]
            if script:
                script = script[0].Script
            else:
                scriptRec = ot.ScriptRecord()
                scriptRec.ScriptTag = scriptTag + " " * (4 - len(scriptTag))
                scriptRec.Script = ot.Script()
                records.append(scriptRec)
                script = scriptRec.Script
                script.DefaultLangSys = None
                script.LangSysRecord = []
                script.LangSysCount = 0

            if langSysTag == "default":
                script.DefaultLangSys = langSys
            else:
                langSysRec = ot.LangSysRecord()
                langSysRec.LangSysTag = langSysTag + " " * (4 - len(langSysTag))
                langSysRec.LangSys = langSys
                script.LangSysRecord.append(langSysRec)
                script.LangSysCount = len(script.LangSysRecord)

    for script in records:
        script.Script.LangSysRecord = sorted(
            script.Script.LangSysRecord, key=lambda rec: rec.LangSysTag
        )
    self.ScriptRecord = sorted(records, key=lambda rec: rec.ScriptTag)
    self.ScriptCount = len(self.ScriptRecord)
    return self


def parseFeatureList(lines, lookupMap=None, featureMap=None):
    self = ot.FeatureList()
    self.FeatureRecord = []
    with lines.between("feature table"):
        for line in lines:
            name, featureTag, lookups = line
            if featureMap is not None:
                assert name not in featureMap, "Duplicate feature name: %s" % name
                featureMap[name] = len(self.FeatureRecord)
            # If feature name is integer, make sure it matches its index.
            try:
                assert int(name) == len(self.FeatureRecord), "%d %d" % (
                    name,
                    len(self.FeatureRecord),
                )
            except ValueError:
                pass
            featureRec = ot.FeatureRecord()
            featureRec.FeatureTag = featureTag
            featureRec.Feature = ot.Feature()
            self.FeatureRecord.append(featureRec)
            feature = featureRec.Feature
            feature.FeatureParams = None
            syms = stripSplitComma(lookups)
            feature.LookupListIndex = theList = [None] * len(syms)
            for i, sym in enumerate(syms):
                setReference(mapLookup, lookupMap, sym, setitem, theList, i)
            feature.LookupCount = len(feature.LookupListIndex)

    self.FeatureCount = len(self.FeatureRecord)
    return self


def parseLookupFlags(lines):
    flags = 0
    filterset = None
    allFlags = [
        "righttoleft",
        "ignorebaseglyphs",
        "ignoreligatures",
        "ignoremarks",
        "markattachmenttype",
        "markfiltertype",
    ]
    while lines.peeks()[0].lower() in allFlags:
        line = next(lines)
        flag = {
            "righttoleft": 0x0001,
            "ignorebaseglyphs": 0x0002,
            "ignoreligatures": 0x0004,
            "ignoremarks": 0x0008,
        }.get(line[0].lower())
        if flag:
            assert line[1].lower() in ["yes", "no"], line[1]
            if line[1].lower() == "yes":
                flags |= flag
            continue
        if line[0].lower() == "markattachmenttype":
            flags |= int(line[1]) << 8
            continue
        if line[0].lower() == "markfiltertype":
            flags |= 0x10
            filterset = int(line[1])
    return flags, filterset


def parseSingleSubst(lines, font, _lookupMap=None):
    mapping = {}
    for line in lines:
        assert len(line) == 2, line
        line = makeGlyphs(line)
        mapping[line[0]] = line[1]
    return otl.buildSingleSubstSubtable(mapping)


def parseMultiple(lines, font, _lookupMap=None):
    mapping = {}
    for line in lines:
        line = makeGlyphs(line)
        mapping[line[0]] = line[1:]
    return otl.buildMultipleSubstSubtable(mapping)


def parseAlternate(lines, font, _lookupMap=None):
    mapping = {}
    for line in lines:
        line = makeGlyphs(line)
        mapping[line[0]] = line[1:]
    return otl.buildAlternateSubstSubtable(mapping)


def parseLigature(lines, font, _lookupMap=None):
    mapping = {}
    for line in lines:
        assert len(line) >= 2, line
        line = makeGlyphs(line)
        mapping[tuple(line[1:])] = line[0]
    return otl.buildLigatureSubstSubtable(mapping)


def parseSinglePos(lines, font, _lookupMap=None):
    values = {}
    for line in lines:
        assert len(line) == 3, line
        w = line[0].title().replace(" ", "")
        assert w in valueRecordFormatDict
        g = makeGlyph(line[1])
        v = int(line[2])
        if g not in values:
            values[g] = ValueRecord()
        assert not hasattr(values[g], w), (g, w)
        setattr(values[g], w, v)
    return otl.buildSinglePosSubtable(values, font.getReverseGlyphMap())


def parsePair(lines, font, _lookupMap=None):
    self = ot.PairPos()
    self.ValueFormat1 = self.ValueFormat2 = 0
    typ = lines.peeks()[0].split()[0].lower()
    if typ in ("left", "right"):
        self.Format = 1
        values = {}
        for line in lines:
            assert len(line) == 4, line
            side = line[0].split()[0].lower()
            assert side in ("left", "right"), side
            what = line[0][len(side) :].title().replace(" ", "")
            mask = valueRecordFormatDict[what][0]
            glyph1, glyph2 = makeGlyphs(line[1:3])
            value = int(line[3])
            if not glyph1 in values:
                values[glyph1] = {}
            if not glyph2 in values[glyph1]:
                values[glyph1][glyph2] = (ValueRecord(), ValueRecord())
            rec2 = values[glyph1][glyph2]
            if side == "left":
                self.ValueFormat1 |= mask
                vr = rec2[0]
            else:
                self.ValueFormat2 |= mask
                vr = rec2[1]
            assert not hasattr(vr, what), (vr, what)
            setattr(vr, what, value)
        self.Coverage = makeCoverage(set(values.keys()), font)
        self.PairSet = []
        for glyph1 in self.Coverage.glyphs:
            values1 = values[glyph1]
            pairset = ot.PairSet()
            records = pairset.PairValueRecord = []
            for glyph2 in sorted(values1.keys(), key=font.getGlyphID):
                values2 = values1[glyph2]
                pair = ot.PairValueRecord()
                pair.SecondGlyph = glyph2
                pair.Value1 = values2[0]
                pair.Value2 = values2[1] if self.ValueFormat2 else None
                records.append(pair)
            pairset.PairValueCount = len(pairset.PairValueRecord)
            self.PairSet.append(pairset)
        self.PairSetCount = len(self.PairSet)
    elif typ.endswith("class"):
        self.Format = 2
        classDefs = [None, None]
        while lines.peeks()[0].endswith("class definition begin"):
            typ = lines.peek()[0][: -len("class definition begin")].lower()
            idx, klass = {
                "first": (0, ot.ClassDef1),
                "second": (1, ot.ClassDef2),
            }[typ]
            assert classDefs[idx] is None
            classDefs[idx] = parseClassDef(lines, font, klass=klass)
        self.ClassDef1, self.ClassDef2 = classDefs
        self.Class1Count, self.Class2Count = (
            1 + max(c.classDefs.values()) for c in classDefs
        )
        self.Class1Record = [ot.Class1Record() for i in range(self.Class1Count)]
        for rec1 in self.Class1Record:
            rec1.Class2Record = [ot.Class2Record() for j in range(self.Class2Count)]
            for rec2 in rec1.Class2Record:
                rec2.Value1 = ValueRecord()
                rec2.Value2 = ValueRecord()
        for line in lines:
            assert len(line) == 4, line
            side = line[0].split()[0].lower()
            assert side in ("left", "right"), side
            what = line[0][len(side) :].title().replace(" ", "")
            mask = valueRecordFormatDict[what][0]
            class1, class2, value = (int(x) for x in line[1:4])
            rec2 = self.Class1Record[class1].Class2Record[class2]
            if side == "left":
                self.ValueFormat1 |= mask
                vr = rec2.Value1
            else:
                self.ValueFormat2 |= mask
                vr = rec2.Value2
            assert not hasattr(vr, what), (vr, what)
            setattr(vr, what, value)
        for rec1 in self.Class1Record:
            for rec2 in rec1.Class2Record:
                rec2.Value1 = ValueRecord(self.ValueFormat1, rec2.Value1)
                rec2.Value2 = (
                    ValueRecord(self.ValueFormat2, rec2.Value2)
                    if self.ValueFormat2
                    else None
                )

        self.Coverage = makeCoverage(set(self.ClassDef1.classDefs.keys()), font)
    else:
        assert 0, typ
    return self


def parseKernset(lines, font, _lookupMap=None):
    typ = lines.peeks()[0].split()[0].lower()
    if typ in ("left", "right"):
        with lines.until(
            ("firstclass definition begin", "secondclass definition begin")
        ):
            return parsePair(lines, font)
    return parsePair(lines, font)


def makeAnchor(data, klass=ot.Anchor):
    assert len(data) <= 2
    anchor = klass()
    anchor.Format = 1
    anchor.XCoordinate, anchor.YCoordinate = intSplitComma(data[0])
    if len(data) > 1 and data[1] != "":
        anchor.Format = 2
        anchor.AnchorPoint = int(data[1])
    return anchor


def parseCursive(lines, font, _lookupMap=None):
    records = {}
    for line in lines:
        assert len(line) in [3, 4], line
        idx, klass = {
            "entry": (0, ot.EntryAnchor),
            "exit": (1, ot.ExitAnchor),
        }[line[0]]
        glyph = makeGlyph(line[1])
        if glyph not in records:
            records[glyph] = [None, None]
        assert records[glyph][idx] is None, (glyph, idx)
        records[glyph][idx] = makeAnchor(line[2:], klass)
    return otl.buildCursivePosSubtable(records, font.getReverseGlyphMap())


def makeMarkRecords(data, coverage, c):
    records = []
    for glyph in coverage.glyphs:
        klass, anchor = data[glyph]
        record = c.MarkRecordClass()
        record.Class = klass
        setattr(record, c.MarkAnchor, anchor)
        records.append(record)
    return records


def makeBaseRecords(data, coverage, c, classCount):
    records = []
    idx = {}
    for glyph in coverage.glyphs:
        idx[glyph] = len(records)
        record = c.BaseRecordClass()
        anchors = [None] * classCount
        setattr(record, c.BaseAnchor, anchors)
        records.append(record)
    for (glyph, klass), anchor in data.items():
        record = records[idx[glyph]]
        anchors = getattr(record, c.BaseAnchor)
        assert anchors[klass] is None, (glyph, klass)
        anchors[klass] = anchor
    return records


def makeLigatureRecords(data, coverage, c, classCount):
    records = [None] * len(coverage.glyphs)
    idx = {g: i for i, g in enumerate(coverage.glyphs)}

    for (glyph, klass, compIdx, compCount), anchor in data.items():
        record = records[idx[glyph]]
        if record is None:
            record = records[idx[glyph]] = ot.LigatureAttach()
            record.ComponentCount = compCount
            record.ComponentRecord = [ot.ComponentRecord() for i in range(compCount)]
            for compRec in record.ComponentRecord:
                compRec.LigatureAnchor = [None] * classCount
        assert record.ComponentCount == compCount, (
            glyph,
            record.ComponentCount,
            compCount,
        )

        anchors = record.ComponentRecord[compIdx - 1].LigatureAnchor
        assert anchors[klass] is None, (glyph, compIdx, klass)
        anchors[klass] = anchor
    return records


def parseMarkToSomething(lines, font, c):
    self = c.Type()
    self.Format = 1
    markData = {}
    baseData = {}
    Data = {
        "mark": (markData, c.MarkAnchorClass),
        "base": (baseData, c.BaseAnchorClass),
        "ligature": (baseData, c.BaseAnchorClass),
    }
    maxKlass = 0
    for line in lines:
        typ = line[0]
        assert typ in ("mark", "base", "ligature")
        glyph = makeGlyph(line[1])
        data, anchorClass = Data[typ]
        extraItems = 2 if typ == "ligature" else 0
        extras = tuple(int(i) for i in line[2 : 2 + extraItems])
        klass = int(line[2 + extraItems])
        anchor = makeAnchor(line[3 + extraItems :], anchorClass)
        if typ == "mark":
            key, value = glyph, (klass, anchor)
        else:
            key, value = ((glyph, klass) + extras), anchor
        assert key not in data, key
        data[key] = value
        maxKlass = max(maxKlass, klass)

    # Mark
    markCoverage = makeCoverage(set(markData.keys()), font, c.MarkCoverageClass)
    markArray = c.MarkArrayClass()
    markRecords = makeMarkRecords(markData, markCoverage, c)
    setattr(markArray, c.MarkRecord, markRecords)
    setattr(markArray, c.MarkCount, len(markRecords))
    setattr(self, c.MarkCoverage, markCoverage)
    setattr(self, c.MarkArray, markArray)
    self.ClassCount = maxKlass + 1

    # Base
    self.classCount = 0 if not baseData else 1 + max(k[1] for k, v in baseData.items())
    baseCoverage = makeCoverage(
        set([k[0] for k in baseData.keys()]), font, c.BaseCoverageClass
    )
    baseArray = c.BaseArrayClass()
    if c.Base == "Ligature":
        baseRecords = makeLigatureRecords(baseData, baseCoverage, c, self.classCount)
    else:
        baseRecords = makeBaseRecords(baseData, baseCoverage, c, self.classCount)
    setattr(baseArray, c.BaseRecord, baseRecords)
    setattr(baseArray, c.BaseCount, len(baseRecords))
    setattr(self, c.BaseCoverage, baseCoverage)
    setattr(self, c.BaseArray, baseArray)

    return self


class MarkHelper(object):
    def __init__(self):
        for Which in ("Mark", "Base"):
            for What in ("Coverage", "Array", "Count", "Record", "Anchor"):
                key = Which + What
                if Which == "Mark" and What in ("Count", "Record", "Anchor"):
                    value = key
                else:
                    value = getattr(self, Which) + What
                if value == "LigatureRecord":
                    value = "LigatureAttach"
                setattr(self, key, value)
                if What != "Count":
                    klass = getattr(ot, value)
                    setattr(self, key + "Class", klass)


class MarkToBaseHelper(MarkHelper):
    Mark = "Mark"
    Base = "Base"
    Type = ot.MarkBasePos


class MarkToMarkHelper(MarkHelper):
    Mark = "Mark1"
    Base = "Mark2"
    Type = ot.MarkMarkPos


class MarkToLigatureHelper(MarkHelper):
    Mark = "Mark"
    Base = "Ligature"
    Type = ot.MarkLigPos


def parseMarkToBase(lines, font, _lookupMap=None):
    return parseMarkToSomething(lines, font, MarkToBaseHelper())


def parseMarkToMark(lines, font, _lookupMap=None):
    return parseMarkToSomething(lines, font, MarkToMarkHelper())


def parseMarkToLigature(lines, font, _lookupMap=None):
    return parseMarkToSomething(lines, font, MarkToLigatureHelper())


def stripSplitComma(line):
    return [s.strip() for s in line.split(",")] if line else []


def intSplitComma(line):
    return [int(i) for i in line.split(",")] if line else []


# Copied from fontTools.subset
class ContextHelper(object):
    def __init__(self, klassName, Format):
        if klassName.endswith("Subst"):
            Typ = "Sub"
            Type = "Subst"
        else:
            Typ = "Pos"
            Type = "Pos"
        if klassName.startswith("Chain"):
            Chain = "Chain"
            InputIdx = 1
            DataLen = 3
        else:
            Chain = ""
            InputIdx = 0
            DataLen = 1
        ChainTyp = Chain + Typ

        self.Typ = Typ
        self.Type = Type
        self.Chain = Chain
        self.ChainTyp = ChainTyp
        self.InputIdx = InputIdx
        self.DataLen = DataLen

        self.LookupRecord = Type + "LookupRecord"

        if Format == 1:
            Coverage = lambda r: r.Coverage
            ChainCoverage = lambda r: r.Coverage
            ContextData = lambda r: (None,)
            ChainContextData = lambda r: (None, None, None)
            SetContextData = None
            SetChainContextData = None
            RuleData = lambda r: (r.Input,)
            ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

            def SetRuleData(r, d):
                (r.Input,) = d
                (r.GlyphCount,) = (len(x) + 1 for x in d)

            def ChainSetRuleData(r, d):
                r.Backtrack, r.Input, r.LookAhead = d
                (
                    r.BacktrackGlyphCount,
                    r.InputGlyphCount,
                    r.LookAheadGlyphCount,
                ) = (len(d[0]), len(d[1]) + 1, len(d[2]))

        elif Format == 2:
            Coverage = lambda r: r.Coverage
            ChainCoverage = lambda r: r.Coverage
            ContextData = lambda r: (r.ClassDef,)
            ChainContextData = lambda r: (
                r.BacktrackClassDef,
                r.InputClassDef,
                r.LookAheadClassDef,
            )

            def SetContextData(r, d):
                (r.ClassDef,) = d

            def SetChainContextData(r, d):
                r.BacktrackClassDef, r.InputClassDef, r.LookAheadClassDef = d

            RuleData = lambda r: (r.Class,)
            ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

            def SetRuleData(r, d):
                (r.Class,) = d
                (r.GlyphCount,) = (len(x) + 1 for x in d)

            def ChainSetRuleData(r, d):
                r.Backtrack, r.Input, r.LookAhead = d
                (
                    r.BacktrackGlyphCount,
                    r.InputGlyphCount,
                    r.LookAheadGlyphCount,
                ) = (len(d[0]), len(d[1]) + 1, len(d[2]))

        elif Format == 3:
            Coverage = lambda r: r.Coverage[0]
            ChainCoverage = lambda r: r.InputCoverage[0]
            ContextData = None
            ChainContextData = None
            SetContextData = None
            SetChainContextData = None
            RuleData = lambda r: r.Coverage
            ChainRuleData = lambda r: (
                r.BacktrackCoverage + r.InputCoverage + r.LookAheadCoverage
            )

            def SetRuleData(r, d):
                (r.Coverage,) = d
                (r.GlyphCount,) = (len(x) for x in d)

            def ChainSetRuleData(r, d):
                r.BacktrackCoverage, r.InputCoverage, r.LookAheadCoverage = d
                (
                    r.BacktrackGlyphCount,
                    r.InputGlyphCount,
                    r.LookAheadGlyphCount,
                ) = (len(x) for x in d)

        else:
            assert 0, "unknown format: %s" % Format

        if Chain:
            self.Coverage = ChainCoverage
            self.ContextData = ChainContextData
            self.SetContextData = SetChainContextData
            self.RuleData = ChainRuleData
            self.SetRuleData = ChainSetRuleData
        else:
            self.Coverage = Coverage
            self.ContextData = ContextData
            self.SetContextData = SetContextData
            self.RuleData = RuleData
            self.SetRuleData = SetRuleData

        if Format == 1:
            self.Rule = ChainTyp + "Rule"
            self.RuleCount = ChainTyp + "RuleCount"
            self.RuleSet = ChainTyp + "RuleSet"
            self.RuleSetCount = ChainTyp + "RuleSetCount"
            self.Intersect = lambda glyphs, c, r: [r] if r in glyphs else []
        elif Format == 2:
            self.Rule = ChainTyp + "ClassRule"
            self.RuleCount = ChainTyp + "ClassRuleCount"
            self.RuleSet = ChainTyp + "ClassSet"
            self.RuleSetCount = ChainTyp + "ClassSetCount"
            self.Intersect = lambda glyphs, c, r: (
                c.intersect_class(glyphs, r)
                if c
                else (set(glyphs) if r == 0 else set())
            )

            self.ClassDef = "InputClassDef" if Chain else "ClassDef"
            self.ClassDefIndex = 1 if Chain else 0
            self.Input = "Input" if Chain else "Class"


def parseLookupRecords(items, klassName, lookupMap=None):
    klass = getattr(ot, klassName)
    lst = []
    for item in items:
        rec = klass()
        item = stripSplitComma(item)
        assert len(item) == 2, item
        idx = int(item[0])
        assert idx > 0, idx
        rec.SequenceIndex = idx - 1
        setReference(mapLookup, lookupMap, item[1], setattr, rec, "LookupListIndex")
        lst.append(rec)
    return lst


def makeClassDef(classDefs, font, klass=ot.Coverage):
    if not classDefs:
        return None
    self = klass()
    self.classDefs = dict(classDefs)
    return self


def parseClassDef(lines, font, klass=ot.ClassDef):
    classDefs = {}
    with lines.between("class definition"):
        for line in lines:
            glyph = makeGlyph(line[0])
            assert glyph not in classDefs, glyph
            classDefs[glyph] = int(line[1])
    return makeClassDef(classDefs, font, klass)


def makeCoverage(glyphs, font, klass=ot.Coverage):
    if not glyphs:
        return None
    if isinstance(glyphs, set):
        glyphs = sorted(glyphs)
    coverage = klass()
    coverage.glyphs = sorted(set(glyphs), key=font.getGlyphID)
    return coverage


def parseCoverage(lines, font, klass=ot.Coverage):
    glyphs = []
    with lines.between("coverage definition"):
        for line in lines:
            glyphs.append(makeGlyph(line[0]))
    return makeCoverage(glyphs, font, klass)


def bucketizeRules(self, c, rules, bucketKeys):
    buckets = {}
    for seq, recs in rules:
        buckets.setdefault(seq[c.InputIdx][0], []).append(
            (tuple(s[1 if i == c.InputIdx else 0 :] for i, s in enumerate(seq)), recs)
        )

    rulesets = []
    for firstGlyph in bucketKeys:
        if firstGlyph not in buckets:
            rulesets.append(None)
            continue
        thisRules = []
        for seq, recs in buckets[firstGlyph]:
            rule = getattr(ot, c.Rule)()
            c.SetRuleData(rule, seq)
            setattr(rule, c.Type + "Count", len(recs))
            setattr(rule, c.LookupRecord, recs)
            thisRules.append(rule)

        ruleset = getattr(ot, c.RuleSet)()
        setattr(ruleset, c.Rule, thisRules)
        setattr(ruleset, c.RuleCount, len(thisRules))
        rulesets.append(ruleset)

    setattr(self, c.RuleSet, rulesets)
    setattr(self, c.RuleSetCount, len(rulesets))


def parseContext(lines, font, Type, lookupMap=None):
    self = getattr(ot, Type)()
    typ = lines.peeks()[0].split()[0].lower()
    if typ == "glyph":
        self.Format = 1
        log.debug("Parsing %s format %s", Type, self.Format)
        c = ContextHelper(Type, self.Format)
        rules = []
        for line in lines:
            assert line[0].lower() == "glyph", line[0]
            while len(line) < 1 + c.DataLen:
                line.append("")
            seq = tuple(makeGlyphs(stripSplitComma(i)) for i in line[1 : 1 + c.DataLen])
            recs = parseLookupRecords(line[1 + c.DataLen :], c.LookupRecord, lookupMap)
            rules.append((seq, recs))

        firstGlyphs = set(seq[c.InputIdx][0] for seq, recs in rules)
        self.Coverage = makeCoverage(firstGlyphs, font)
        bucketizeRules(self, c, rules, self.Coverage.glyphs)
    elif typ.endswith("class"):
        self.Format = 2
        log.debug("Parsing %s format %s", Type, self.Format)
        c = ContextHelper(Type, self.Format)
        classDefs = [None] * c.DataLen
        while lines.peeks()[0].endswith("class definition begin"):
            typ = lines.peek()[0][: -len("class definition begin")].lower()
            idx, klass = {
                1: {
                    "": (0, ot.ClassDef),
                },
                3: {
                    "backtrack": (0, ot.BacktrackClassDef),
                    "": (1, ot.InputClassDef),
                    "lookahead": (2, ot.LookAheadClassDef),
                },
            }[c.DataLen][typ]
            assert classDefs[idx] is None, idx
            classDefs[idx] = parseClassDef(lines, font, klass=klass)
        c.SetContextData(self, classDefs)
        rules = []
        for line in lines:
            assert line[0].lower().startswith("class"), line[0]
            while len(line) < 1 + c.DataLen:
                line.append("")
            seq = tuple(intSplitComma(i) for i in line[1 : 1 + c.DataLen])
            recs = parseLookupRecords(line[1 + c.DataLen :], c.LookupRecord, lookupMap)
            rules.append((seq, recs))
        firstClasses = set(seq[c.InputIdx][0] for seq, recs in rules)
        firstGlyphs = set(
            g for g, c in classDefs[c.InputIdx].classDefs.items() if c in firstClasses
        )
        self.Coverage = makeCoverage(firstGlyphs, font)
        bucketizeRules(self, c, rules, range(max(firstClasses) + 1))
    elif typ.endswith("coverage"):
        self.Format = 3
        log.debug("Parsing %s format %s", Type, self.Format)
        c = ContextHelper(Type, self.Format)
        coverages = tuple([] for i in range(c.DataLen))
        while lines.peeks()[0].endswith("coverage definition begin"):
            typ = lines.peek()[0][: -len("coverage definition begin")].lower()
            idx, klass = {
                1: {
                    "": (0, ot.Coverage),
                },
                3: {
                    "backtrack": (0, ot.BacktrackCoverage),
                    "input": (1, ot.InputCoverage),
                    "lookahead": (2, ot.LookAheadCoverage),
                },
            }[c.DataLen][typ]
            coverages[idx].append(parseCoverage(lines, font, klass=klass))
        c.SetRuleData(self, coverages)
        lines = list(lines)
        assert len(lines) == 1
        line = lines[0]
        assert line[0].lower() == "coverage", line[0]
        recs = parseLookupRecords(line[1:], c.LookupRecord, lookupMap)
        setattr(self, c.Type + "Count", len(recs))
        setattr(self, c.LookupRecord, recs)
    else:
        assert 0, typ
    return self


def parseContextSubst(lines, font, lookupMap=None):
    return parseContext(lines, font, "ContextSubst", lookupMap=lookupMap)


def parseContextPos(lines, font, lookupMap=None):
    return parseContext(lines, font, "ContextPos", lookupMap=lookupMap)


def parseChainedSubst(lines, font, lookupMap=None):
    return parseContext(lines, font, "ChainContextSubst", lookupMap=lookupMap)


def parseChainedPos(lines, font, lookupMap=None):
    return parseContext(lines, font, "ChainContextPos", lookupMap=lookupMap)


def parseReverseChainedSubst(lines, font, _lookupMap=None):
    self = ot.ReverseChainSingleSubst()
    self.Format = 1
    coverages = ([], [])
    while lines.peeks()[0].endswith("coverage definition begin"):
        typ = lines.peek()[0][: -len("coverage definition begin")].lower()
        idx, klass = {
            "backtrack": (0, ot.BacktrackCoverage),
            "lookahead": (1, ot.LookAheadCoverage),
        }[typ]
        coverages[idx].append(parseCoverage(lines, font, klass=klass))
    self.BacktrackCoverage = coverages[0]
    self.BacktrackGlyphCount = len(self.BacktrackCoverage)
    self.LookAheadCoverage = coverages[1]
    self.LookAheadGlyphCount = len(self.LookAheadCoverage)
    mapping = {}
    for line in lines:
        assert len(line) == 2, line
        line = makeGlyphs(line)
        mapping[line[0]] = line[1]
    self.Coverage = makeCoverage(set(mapping.keys()), font)
    self.Substitute = [mapping[k] for k in self.Coverage.glyphs]
    self.GlyphCount = len(self.Substitute)
    return self


def parseLookup(lines, tableTag, font, lookupMap=None):
    line = lines.expect("lookup")
    _, name, typ = line
    log.debug("Parsing lookup type %s %s", typ, name)
    lookup = ot.Lookup()
    lookup.LookupFlag, filterset = parseLookupFlags(lines)
    if filterset is not None:
        lookup.MarkFilteringSet = filterset
    lookup.LookupType, parseLookupSubTable = {
        "GSUB": {
            "single": (1, parseSingleSubst),
            "multiple": (2, parseMultiple),
            "alternate": (3, parseAlternate),
            "ligature": (4, parseLigature),
            "context": (5, parseContextSubst),
            "chained": (6, parseChainedSubst),
            "reversechained": (8, parseReverseChainedSubst),
        },
        "GPOS": {
            "single": (1, parseSinglePos),
            "pair": (2, parsePair),
            "kernset": (2, parseKernset),
            "cursive": (3, parseCursive),
            "mark to base": (4, parseMarkToBase),
            "mark to ligature": (5, parseMarkToLigature),
            "mark to mark": (6, parseMarkToMark),
            "context": (7, parseContextPos),
            "chained": (8, parseChainedPos),
        },
    }[tableTag][typ]

    with lines.until("lookup end"):
        subtables = []

        while lines.peek():
            with lines.until(("% subtable", "subtable end")):
                while lines.peek():
                    subtable = parseLookupSubTable(lines, font, lookupMap)
                    assert lookup.LookupType == subtable.LookupType
                    subtables.append(subtable)
            if lines.peeks()[0] in ("% subtable", "subtable end"):
                next(lines)
    lines.expect("lookup end")

    lookup.SubTable = subtables
    lookup.SubTableCount = len(lookup.SubTable)
    if lookup.SubTableCount == 0:
        # Remove this return when following is fixed:
        # https://github.com/fonttools/fonttools/issues/789
        return None
    return lookup


def parseGSUBGPOS(lines, font, tableTag):
    container = ttLib.getTableClass(tableTag)()
    lookupMap = DeferredMapping()
    featureMap = DeferredMapping()
    assert tableTag in ("GSUB", "GPOS")
    log.debug("Parsing %s", tableTag)
    self = getattr(ot, tableTag)()
    self.Version = 0x00010000
    fields = {
        "script table begin": (
            "ScriptList",
            lambda lines: parseScriptList(lines, featureMap),
        ),
        "feature table begin": (
            "FeatureList",
            lambda lines: parseFeatureList(lines, lookupMap, featureMap),
        ),
        "lookup": ("LookupList", None),
    }
    for attr, parser in fields.values():
        setattr(self, attr, None)
    while lines.peek() is not None:
        typ = lines.peek()[0].lower()
        if typ not in fields:
            log.debug("Skipping %s", lines.peek())
            next(lines)
            continue
        attr, parser = fields[typ]
        if typ == "lookup":
            if self.LookupList is None:
                self.LookupList = ot.LookupList()
                self.LookupList.Lookup = []
            _, name, _ = lines.peek()
            lookup = parseLookup(lines, tableTag, font, lookupMap)
            if lookupMap is not None:
                assert name not in lookupMap, "Duplicate lookup name: %s" % name
                lookupMap[name] = len(self.LookupList.Lookup)
            else:
                assert int(name) == len(self.LookupList.Lookup), "%d %d" % (
                    name,
                    len(self.Lookup),
                )
            self.LookupList.Lookup.append(lookup)
        else:
            assert getattr(self, attr) is None, attr
            setattr(self, attr, parser(lines))
    if self.LookupList:
        self.LookupList.LookupCount = len(self.LookupList.Lookup)
    if lookupMap is not None:
        lookupMap.applyDeferredMappings()
        if os.environ.get(LOOKUP_DEBUG_ENV_VAR):
            if "Debg" not in font:
                font["Debg"] = newTable("Debg")
                font["Debg"].data = {}
            debug = (
                font["Debg"]
                .data.setdefault(LOOKUP_DEBUG_INFO_KEY, {})
                .setdefault(tableTag, {})
            )
            for name, lookup in lookupMap.items():
                debug[str(lookup)] = ["", name, ""]

        featureMap.applyDeferredMappings()
    container.table = self
    return container


def parseGSUB(lines, font):
    return parseGSUBGPOS(lines, font, "GSUB")


def parseGPOS(lines, font):
    return parseGSUBGPOS(lines, font, "GPOS")


def parseAttachList(lines, font):
    points = {}
    with lines.between("attachment list"):
        for line in lines:
            glyph = makeGlyph(line[0])
            assert glyph not in points, glyph
            points[glyph] = [int(i) for i in line[1:]]
    return otl.buildAttachList(points, font.getReverseGlyphMap())


def parseCaretList(lines, font):
    carets = {}
    with lines.between("carets"):
        for line in lines:
            glyph = makeGlyph(line[0])
            assert glyph not in carets, glyph
            num = int(line[1])
            thisCarets = [int(i) for i in line[2:]]
            assert num == len(thisCarets), line
            carets[glyph] = thisCarets
    return otl.buildLigCaretList(carets, {}, font.getReverseGlyphMap())


def makeMarkFilteringSets(sets, font):
    self = ot.MarkGlyphSetsDef()
    self.MarkSetTableFormat = 1
    self.MarkSetCount = 1 + max(sets.keys())
    self.Coverage = [None] * self.MarkSetCount
    for k, v in sorted(sets.items()):
        self.Coverage[k] = makeCoverage(set(v), font)
    return self


def parseMarkFilteringSets(lines, font):
    sets = {}
    with lines.between("set definition"):
        for line in lines:
            assert len(line) == 2, line
            glyph = makeGlyph(line[0])
            # TODO accept set names
            st = int(line[1])
            if st not in sets:
                sets[st] = []
            sets[st].append(glyph)
    return makeMarkFilteringSets(sets, font)


def parseGDEF(lines, font):
    container = ttLib.getTableClass("GDEF")()
    log.debug("Parsing GDEF")
    self = ot.GDEF()
    fields = {
        "class definition begin": (
            "GlyphClassDef",
            lambda lines, font: parseClassDef(lines, font, klass=ot.GlyphClassDef),
        ),
        "attachment list begin": ("AttachList", parseAttachList),
        "carets begin": ("LigCaretList", parseCaretList),
        "mark attachment class definition begin": (
            "MarkAttachClassDef",
            lambda lines, font: parseClassDef(lines, font, klass=ot.MarkAttachClassDef),
        ),
        "markfilter set definition begin": ("MarkGlyphSetsDef", parseMarkFilteringSets),
    }
    for attr, parser in fields.values():
        setattr(self, attr, None)
    while lines.peek() is not None:
        typ = lines.peek()[0].lower()
        if typ not in fields:
            log.debug("Skipping %s", typ)
            next(lines)
            continue
        attr, parser = fields[typ]
        assert getattr(self, attr) is None, attr
        setattr(self, attr, parser(lines, font))
    self.Version = 0x00010000 if self.MarkGlyphSetsDef is None else 0x00010002
    container.table = self
    return container


def parseCmap(lines, font):
    container = ttLib.getTableClass("cmap")()
    log.debug("Parsing cmap")
    tables = []
    while lines.peek() is not None:
        lines.expect("cmap subtable %d" % len(tables))
        platId, encId, fmt, lang = [
            parseCmapId(lines, field)
            for field in ("platformID", "encodingID", "format", "language")
        ]
        table = cmap_classes[fmt](fmt)
        table.platformID = platId
        table.platEncID = encId
        table.language = lang
        table.cmap = {}
        line = next(lines)
        while line[0] != "end subtable":
            table.cmap[int(line[0], 16)] = line[1]
            line = next(lines)
        tables.append(table)
    container.tableVersion = 0
    container.tables = tables
    return container


def parseCmapId(lines, field):
    line = next(lines)
    assert field == line[0]
    return int(line[1])


def parseTable(lines, font, tableTag=None):
    log.debug("Parsing table")
    line = lines.peeks()
    tag = None
    if line[0].split()[0] == "FontDame":
        tag = line[0].split()[1]
    elif " ".join(line[0].split()[:3]) == "Font Chef Table":
        tag = line[0].split()[3]
    if tag is not None:
        next(lines)
        tag = tag.ljust(4)
        if tableTag is None:
            tableTag = tag
        else:
            assert tableTag == tag, (tableTag, tag)

    assert (
        tableTag is not None
    ), "Don't know what table to parse and data doesn't specify"

    return {
        "GSUB": parseGSUB,
        "GPOS": parseGPOS,
        "GDEF": parseGDEF,
        "cmap": parseCmap,
    }[tableTag](lines, font)


class Tokenizer(object):
    def __init__(self, f):
        # TODO BytesIO / StringIO as needed?  also, figure out whether we work on bytes or unicode
        lines = iter(f)
        try:
            self.filename = f.name
        except:
            self.filename = None
        self.lines = iter(lines)
        self.line = ""
        self.lineno = 0
        self.stoppers = []
        self.buffer = None

    def __iter__(self):
        return self

    def _next_line(self):
        self.lineno += 1
        line = self.line = next(self.lines)
        line = [s.strip() for s in line.split("\t")]
        if len(line) == 1 and not line[0]:
            del line[0]
        if line and not line[-1]:
            log.warning("trailing tab found on line %d: %s" % (self.lineno, self.line))
            while line and not line[-1]:
                del line[-1]
        return line

    def _next_nonempty(self):
        while True:
            line = self._next_line()
            # Skip comments and empty lines
            if line and line[0] and (line[0][0] != "%" or line[0] == "% subtable"):
                return line

    def _next_buffered(self):
        if self.buffer:
            ret = self.buffer
            self.buffer = None
            return ret
        else:
            return self._next_nonempty()

    def __next__(self):
        line = self._next_buffered()
        if line[0].lower() in self.stoppers:
            self.buffer = line
            raise StopIteration
        return line

    def next(self):
        return self.__next__()

    def peek(self):
        if not self.buffer:
            try:
                self.buffer = self._next_nonempty()
            except StopIteration:
                return None
        if self.buffer[0].lower() in self.stoppers:
            return None
        return self.buffer

    def peeks(self):
        ret = self.peek()
        return ret if ret is not None else ("",)

    @contextmanager
    def between(self, tag):
        start = tag + " begin"
        end = tag + " end"
        self.expectendswith(start)
        self.stoppers.append(end)
        yield
        del self.stoppers[-1]
        self.expect(tag + " end")

    @contextmanager
    def until(self, tags):
        if type(tags) is not tuple:
            tags = (tags,)
        self.stoppers.extend(tags)
        yield
        del self.stoppers[-len(tags) :]

    def expect(self, s):
        line = next(self)
        tag = line[0].lower()
        assert tag == s, "Expected '%s', got '%s'" % (s, tag)
        return line

    def expectendswith(self, s):
        line = next(self)
        tag = line[0].lower()
        assert tag.endswith(s), "Expected '*%s', got '%s'" % (s, tag)
        return line


def build(f, font, tableTag=None):
    """Convert a Monotype font layout file to an OpenType layout object

    A font object must be passed, but this may be a "dummy" font; it is only
    used for sorting glyph sets when making coverage tables and to hold the
    OpenType layout table while it is being built.

    Args:
            f: A file object.
            font (TTFont): A font object.
            tableTag (string): If provided, asserts that the file contains data for the
                    given OpenType table.

    Returns:
            An object representing the table. (e.g. ``table_G_S_U_B_``)
    """
    lines = Tokenizer(f)
    return parseTable(lines, font, tableTag=tableTag)


def main(args=None, font=None):
    """Convert a FontDame OTL file to TTX XML

    Writes XML output to stdout.

    Args:
            args: Command line arguments (``--font``, ``--table``, input files).
    """
    import sys
    from fontTools import configLogger
    from fontTools.misc.testTools import MockFont

    if args is None:
        args = sys.argv[1:]

    # configure the library logger (for >= WARNING)
    configLogger()
    # comment this out to enable debug messages from mtiLib's logger
    # log.setLevel(logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools mtiLib",
        description=main.__doc__,
    )

    parser.add_argument(
        "--font",
        "-f",
        metavar="FILE",
        dest="font",
        help="Input TTF files (used for glyph classes and sorting coverage tables)",
    )
    parser.add_argument(
        "--table",
        "-t",
        metavar="TABLE",
        dest="tableTag",
        help="Table to fill (sniffed from input file if not provided)",
    )
    parser.add_argument(
        "inputs", metavar="FILE", type=str, nargs="+", help="Input FontDame .txt files"
    )

    args = parser.parse_args(args)

    if font is None:
        if args.font:
            font = ttLib.TTFont(args.font)
        else:
            font = MockFont()

    for f in args.inputs:
        log.debug("Processing %s", f)
        with open(f, "rt", encoding="utf-8-sig") as f:
            table = build(f, font, tableTag=args.tableTag)
        blob = table.compile(font)  # Make sure it compiles
        decompiled = table.__class__()
        decompiled.decompile(blob, font)  # Make sure it decompiles!

        # continue
        from fontTools.misc import xmlWriter

        tag = table.tableTag
        writer = xmlWriter.XMLWriter(sys.stdout)
        writer.begintag(tag)
        writer.newline()
        # table.toXML(writer, font)
        decompiled.toXML(writer, font)
        writer.endtag(tag)
        writer.newline()


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.pens.filterPen import FilterPen, FilterPointPen

__all__ = ["TransformPen", "TransformPointPen"]


class TransformPen(FilterPen):
    """Pen that transforms all coordinates using a Affine transformation,
    and passes them to another pen.
    """

    def __init__(self, outPen, transformation):
        """The 'outPen' argument is another pen object. It will receive the
        transformed coordinates. The 'transformation' argument can either
        be a six-tuple, or a fontTools.misc.transform.Transform object.
        """
        super(TransformPen, self).__init__(outPen)
        if not hasattr(transformation, "transformPoint"):
            from fontTools.misc.transform import Transform

            transformation = Transform(*transformation)
        self._transformation = transformation
        self._transformPoint = transformation.transformPoint
        self._stack = []

    def moveTo(self, pt):
        self._outPen.moveTo(self._transformPoint(pt))

    def lineTo(self, pt):
        self._outPen.lineTo(self._transformPoint(pt))

    def curveTo(self, *points):
        self._outPen.curveTo(*self._transformPoints(points))

    def qCurveTo(self, *points):
        if points[-1] is None:
            points = self._transformPoints(points[:-1]) + [None]
        else:
            points = self._transformPoints(points)
        self._outPen.qCurveTo(*points)

    def _transformPoints(self, points):
        transformPoint = self._transformPoint
        return [transformPoint(pt) for pt in points]

    def closePath(self):
        self._outPen.closePath()

    def endPath(self):
        self._outPen.endPath()

    def addComponent(self, glyphName, transformation):
        transformation = self._transformation.transform(transformation)
        self._outPen.addComponent(glyphName, transformation)


class TransformPointPen(FilterPointPen):
    """PointPen that transforms all coordinates using a Affine transformation,
    and passes them to another PointPen.

    For example::

        >>> from fontTools.pens.recordingPen import RecordingPointPen
        >>> rec = RecordingPointPen()
        >>> pen = TransformPointPen(rec, (2, 0, 0, 2, -10, 5))
        >>> v = iter(rec.value)
        >>> pen.beginPath(identifier="contour-0")
        >>> next(v)
        ('beginPath', (), {'identifier': 'contour-0'})

        >>> pen.addPoint((100, 100), "line")
        >>> next(v)
        ('addPoint', ((190, 205), 'line', False, None), {})

        >>> pen.endPath()
        >>> next(v)
        ('endPath', (), {})

        >>> pen.addComponent("a", (1, 0, 0, 1, -10, 5), identifier="component-0")
        >>> next(v)
        ('addComponent', ('a', <Transform [2 0 0 2 -30 15]>), {'identifier': 'component-0'})
    """

    def __init__(self, outPointPen, transformation):
        """The 'outPointPen' argument is another point pen object.
        It will receive the transformed coordinates.
        The 'transformation' argument can either be a six-tuple, or a
        fontTools.misc.transform.Transform object.
        """
        super().__init__(outPointPen)
        if not hasattr(transformation, "transformPoint"):
            from fontTools.misc.transform import Transform

            transformation = Transform(*transformation)
        self._transformation = transformation
        self._transformPoint = transformation.transformPoint

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, **kwargs):
        self._outPen.addPoint(
            self._transformPoint(pt), segmentType, smooth, name, **kwargs
        )

    def addComponent(self, baseGlyphName, transformation, **kwargs):
        transformation = self._transformation.transform(transformation)
        self._outPen.addComponent(baseGlyphName, transformation, **kwargs)


if __name__ == "__main__":
    from fontTools.pens.basePen import _TestPen

    pen = TransformPen(_TestPen(None), (2, 0, 0.5, 2, -10, 0))
    pen.moveTo((0, 0))
    pen.lineTo((0, 100))
    pen.curveTo((50, 75), (60, 50), (50, 25), (0, 0))
    pen.closePath()
<|endoftext|># Copyright 2016 Google Inc. All Rights Reserved.
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

import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen


class Cu2QuPen(FilterPen):
    """A filter pen to convert cubic bezier curves to quadratic b-splines
    using the FontTools SegmentPen protocol.

    Args:

        other_pen: another SegmentPen used to draw the transformed outline.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: flip the contours' direction but keep starting point.
        stats: a dictionary counting the point numbers of quadratic segments.
        all_quadratic: if True (default), only quadratic b-splines are generated.
            if False, quadratic curves or cubic curves are generated depending
            on which one is more economical.
    """

    def __init__(
        self,
        other_pen,
        max_err,
        reverse_direction=False,
        stats=None,
        all_quadratic=True,
    ):
        if reverse_direction:
            other_pen = ReverseContourPen(other_pen)
        super().__init__(other_pen)
        self.max_err = max_err
        self.stats = stats
        self.all_quadratic = all_quadratic

    def _convert_curve(self, pt1, pt2, pt3):
        curve = (self.current_pt, pt1, pt2, pt3)
        result = curve_to_quadratic(curve, self.max_err, self.all_quadratic)
        if self.stats is not None:
            n = str(len(result) - 2)
            self.stats[n] = self.stats.get(n, 0) + 1
        if self.all_quadratic:
            self.qCurveTo(*result[1:])
        else:
            if len(result) == 3:
                self.qCurveTo(*result[1:])
            else:
                assert len(result) == 4
                super().curveTo(*result[1:])

    def curveTo(self, *points):
        n = len(points)
        if n == 3:
            # this is the most common case, so we special-case it
            self._convert_curve(*points)
        elif n > 3:
            for segment in decomposeSuperBezierSegment(points):
                self._convert_curve(*segment)
        else:
            self.qCurveTo(*points)


class Cu2QuPointPen(BasePointToSegmentPen):
    """A filter pen to convert cubic bezier curves to quadratic b-splines
    using the FontTools PointPen protocol.

    Args:
        other_point_pen: another PointPen used to draw the transformed outline.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: reverse the winding direction of all contours.
        stats: a dictionary counting the point numbers of quadratic segments.
        all_quadratic: if True (default), only quadratic b-splines are generated.
            if False, quadratic curves or cubic curves are generated depending
            on which one is more economical.
    """

    __points_required = {
        "move": (1, operator.eq),
        "line": (1, operator.eq),
        "qcurve": (2, operator.ge),
        "curve": (3, operator.eq),
    }

    def __init__(
        self,
        other_point_pen,
        max_err,
        reverse_direction=False,
        stats=None,
        all_quadratic=True,
    ):
        BasePointToSegmentPen.__init__(self)
        if reverse_direction:
            self.pen = ReverseContourPointPen(other_point_pen)
        else:
            self.pen = other_point_pen
        self.max_err = max_err
        self.stats = stats
        self.all_quadratic = all_quadratic

    def _flushContour(self, segments):
        assert len(segments) >= 1
        closed = segments[0][0] != "move"
        new_segments = []
        prev_points = segments[-1][1]
        prev_on_curve = prev_points[-1][0]
        for segment_type, points in segments:
            if segment_type == "curve":
                for sub_points in self._split_super_bezier_segments(points):
                    on_curve, smooth, name, kwargs = sub_points[-1]
                    bcp1, bcp2 = sub_points[0][0], sub_points[1][0]
                    cubic = [prev_on_curve, bcp1, bcp2, on_curve]
                    quad = curve_to_quadratic(cubic, self.max_err, self.all_quadratic)
                    if self.stats is not None:
                        n = str(len(quad) - 2)
                        self.stats[n] = self.stats.get(n, 0) + 1
                    new_points = [(pt, False, None, {}) for pt in quad[1:-1]]
                    new_points.append((on_curve, smooth, name, kwargs))
                    if self.all_quadratic or len(new_points) == 2:
                        new_segments.append(["qcurve", new_points])
                    else:
                        new_segments.append(["curve", new_points])
                    prev_on_curve = sub_points[-1][0]
            else:
                new_segments.append([segment_type, points])
                prev_on_curve = points[-1][0]
        if closed:
            # the BasePointToSegmentPen.endPath method that calls _flushContour
            # rotates the point list of closed contours so that they end with
            # the first on-curve point. We restore the original starting point.
            new_segments = new_segments[-1:] + new_segments[:-1]
        self._drawPoints(new_segments)

    def _split_super_bezier_segments(self, points):
        sub_segments = []
        # n is the number of control points
        n = len(points) - 1
        if n == 2:
            # a simple bezier curve segment
            sub_segments.append(points)
        elif n > 2:
            # a "super" bezier; decompose it
            on_curve, smooth, name, kwargs = points[-1]
            num_sub_segments = n - 1
            for i, sub_points in enumerate(
                decomposeSuperBezierSegment([pt for pt, _, _, _ in points])
            ):
                new_segment = []
                for point in sub_points[:-1]:
                    new_segment.append((point, False, None, {}))
                if i == (num_sub_segments - 1):
                    # the last on-curve keeps its original attributes
                    new_segment.append((on_curve, smooth, name, kwargs))
                else:
                    # on-curves of sub-segments are always "smooth"
                    new_segment.append((sub_points[-1], True, None, {}))
                sub_segments.append(new_segment)
        else:
            raise AssertionError("expected 2 control points, found: %d" % n)
        return sub_segments

    def _drawPoints(self, segments):
        pen = self.pen
        pen.beginPath()
        last_offcurves = []
        points_required = self.__points_required
        for i, (segment_type, points) in enumerate(segments):
            if segment_type in points_required:
                n, op = points_required[segment_type]
                assert op(len(points), n), (
                    f"illegal {segment_type!r} segment point count: "
                    f"expected {n}, got {len(points)}"
                )
                offcurves = points[:-1]
                if i == 0:
                    # any off-curve points preceding the first on-curve
                    # will be appended at the end of the contour
                    last_offcurves = offcurves
                else:
                    for pt, smooth, name, kwargs in offcurves:
                        pen.addPoint(pt, None, smooth, name, **kwargs)
                pt, smooth, name, kwargs = points[-1]
                if pt is None:
                    assert segment_type == "qcurve"
                    # special quadratic contour with no on-curve points:
                    # we need to skip the "None" point. See also the Pen
                    # protocol's qCurveTo() method and fontTools.pens.basePen
                    pass
                else:
                    pen.addPoint(pt, segment_type, smooth, name, **kwargs)
            else:
                raise AssertionError("unexpected segment type: %r" % segment_type)
        for pt, smooth, name, kwargs in last_offcurves:
            pen.addPoint(pt, None, smooth, name, **kwargs)
        pen.endPath()

    def addComponent(self, baseGlyphName, transformation):
        assert self.currentPath is None
        self.pen.addComponent(baseGlyphName, transformation)


class Cu2QuMultiPen:
    """A filter multi-pen to convert cubic bezier curves to quadratic b-splines
    in a interpolation-compatible manner, using the FontTools SegmentPen protocol.

    Args:

        other_pens: list of SegmentPens used to draw the transformed outlines.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: flip the contours' direction but keep starting point.

    This pen does not follow the normal SegmentPen protocol. Instead, its
    moveTo/lineTo/qCurveTo/curveTo methods take a list of tuples that are
    arguments that would normally be passed to a SegmentPen, one item for
    each of the pens in other_pens.
    """

    # TODO Simplify like 3e8ebcdce592fe8a59ca4c3a294cc9724351e1ce
    # Remove start_pts and _add_moveTO

    def __init__(self, other_pens, max_err, reverse_direction=False):
        if reverse_direction:
            other_pens = [
                ReverseContourPen(pen, outputImpliedClosingLine=True)
                for pen in other_pens
            ]
        self.pens = other_pens
        self.max_err = max_err
        self.start_pts = None
        self.current_pts = None

    def _check_contour_is_open(self):
        if self.current_pts is None:
            raise AssertionError("moveTo is required")

    def _check_contour_is_closed(self):
        if self.current_pts is not None:
            raise AssertionError("closePath or endPath is required")

    def _add_moveTo(self):
        if self.start_pts is not None:
            for pt, pen in zip(self.start_pts, self.pens):
                pen.moveTo(*pt)
            self.start_pts = None

    def moveTo(self, pts):
        self._check_contour_is_closed()
        self.start_pts = self.current_pts = pts
        self._add_moveTo()

    def lineTo(self, pts):
        self._check_contour_is_open()
        self._add_moveTo()
        for pt, pen in zip(pts, self.pens):
            pen.lineTo(*pt)
        self.current_pts = pts

    def qCurveTo(self, pointsList):
        self._check_contour_is_open()
        if len(pointsList[0]) == 1:
            self.lineTo([(points[0],) for points in pointsList])
            return
        self._add_moveTo()
        current_pts = []
        for points, pen in zip(pointsList, self.pens):
            pen.qCurveTo(*points)
            current_pts.append((points[-1],))
        self.current_pts = current_pts

    def _curves_to_quadratic(self, pointsList):
        curves = []
        for current_pt, points in zip(self.current_pts, pointsList):
            curves.append(current_pt + points)
        quadratics = curves_to_quadratic(curves, [self.max_err] * len(curves))
        pointsList = []
        for quadratic in quadratics:
            pointsList.append(quadratic[1:])
        self.qCurveTo(pointsList)

    def curveTo(self, pointsList):
        self._check_contour_is_open()
        self._curves_to_quadratic(pointsList)

    def closePath(self):
        self._check_contour_is_open()
        if self.start_pts is None:
            for pen in self.pens:
                pen.closePath()
        self.current_pts = self.start_pts = None

    def endPath(self):
        self._check_contour_is_open()
        if self.start_pts is None:
            for pen in self.pens:
                pen.endPath()
        self.current_pts = self.start_pts = None

    def addComponent(self, glyphName, transformations):
        self._check_contour_is_closed()
        for trans, pen in zip(transformations, self.pens):
            pen.addComponent(glyphName, trans)
<|endoftext|># Copyright (c) 2009 Type Supply LLC
# Author: Tal Leming

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fontTools.cffLib.specializer import commandsToProgram, specializeCommands
from fontTools.misc.psCharStrings import T2CharString
from fontTools.misc.roundTools import otRound, roundFunc
from fontTools.pens.basePen import BasePen


class T2CharStringPen(BasePen):
    """Pen to draw Type 2 CharStrings.

    The 'roundTolerance' argument controls the rounding of point coordinates.
    It is defined as the maximum absolute difference between the original
    float and the rounded integer value.
    The default tolerance of 0.5 means that all floats are rounded to integer;
    a value of 0 disables rounding; values in between will only round floats
    which are close to their integral part within the tolerated range.
    """

    def __init__(
        self,
        width: float | None,
        glyphSet: Dict[str, Any] | None,
        roundTolerance: float = 0.5,
        CFF2: bool = False,
    ) -> None:
        super(T2CharStringPen, self).__init__(glyphSet)
        self.round = roundFunc(roundTolerance)
        self._CFF2 = CFF2
        self._width = width
        self._commands: List[Tuple[str | bytes, List[float]]] = []
        self._p0 = (0, 0)

    def _p(self, pt: Tuple[float, float]) -> List[float]:
        p0 = self._p0
        pt = self._p0 = (self.round(pt[0]), self.round(pt[1]))
        return [pt[0] - p0[0], pt[1] - p0[1]]

    def _moveTo(self, pt: Tuple[float, float]) -> None:
        self._commands.append(("rmoveto", self._p(pt)))

    def _lineTo(self, pt: Tuple[float, float]) -> None:
        self._commands.append(("rlineto", self._p(pt)))

    def _curveToOne(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        pt3: Tuple[float, float],
    ) -> None:
        _p = self._p
        self._commands.append(("rrcurveto", _p(pt1) + _p(pt2) + _p(pt3)))

    def _closePath(self) -> None:
        pass

    def _endPath(self) -> None:
        pass

    def getCharString(
        self,
        private: Dict | None = None,
        globalSubrs: List | None = None,
        optimize: bool = True,
    ) -> T2CharString:
        commands = self._commands
        if optimize:
            maxstack = 48 if not self._CFF2 else 513
            commands = specializeCommands(
                commands, generalizeFirst=False, maxstack=maxstack
            )
        program = commandsToProgram(commands)
        if self._width is not None:
            assert (
                not self._CFF2
            ), "CFF2 does not allow encoding glyph width in CharString."
            program.insert(0, otRound(self._width))
        if not self._CFF2:
            program.append("endchar")
        charString = T2CharString(
            program=program, private=private, globalSubrs=globalSubrs
        )
        return charString
<|endoftext|>from __future__ import annotations

from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import (
    AbstractPointPen,
    DecomposingPointPen,
    ReverseFlipped,
)
from fontTools.pens.recordingPen import RecordingPen


class _PassThruComponentsMixin(object):
    def addComponent(self, glyphName, transformation, **kwargs):
        self._outPen.addComponent(glyphName, transformation, **kwargs)


class FilterPen(_PassThruComponentsMixin, AbstractPen):
    """Base class for pens that apply some transformation to the coordinates
    they receive and pass them to another pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPen
    >>> rec = RecordingPen()
    >>> pen = FilterPen(rec)
    >>> v = iter(rec.value)

    >>> pen.moveTo((0, 0))
    >>> next(v)
    ('moveTo', ((0, 0),))

    >>> pen.lineTo((1, 1))
    >>> next(v)
    ('lineTo', ((1, 1),))

    >>> pen.curveTo((2, 2), (3, 3), (4, 4))
    >>> next(v)
    ('curveTo', ((2, 2), (3, 3), (4, 4)))

    >>> pen.qCurveTo((5, 5), (6, 6), (7, 7), (8, 8))
    >>> next(v)
    ('qCurveTo', ((5, 5), (6, 6), (7, 7), (8, 8)))

    >>> pen.closePath()
    >>> next(v)
    ('closePath', ())

    >>> pen.moveTo((9, 9))
    >>> next(v)
    ('moveTo', ((9, 9),))

    >>> pen.endPath()
    >>> next(v)
    ('endPath', ())

    >>> pen.addComponent('foo', (1, 0, 0, 1, 0, 0))
    >>> next(v)
    ('addComponent', ('foo', (1, 0, 0, 1, 0, 0)))
    """

    def __init__(self, outPen):
        self._outPen = outPen
        self.current_pt = None

    def moveTo(self, pt):
        self._outPen.moveTo(pt)
        self.current_pt = pt

    def lineTo(self, pt):
        self._outPen.lineTo(pt)
        self.current_pt = pt

    def curveTo(self, *points):
        self._outPen.curveTo(*points)
        self.current_pt = points[-1]

    def qCurveTo(self, *points):
        self._outPen.qCurveTo(*points)
        self.current_pt = points[-1]

    def closePath(self):
        self._outPen.closePath()
        self.current_pt = None

    def endPath(self):
        self._outPen.endPath()
        self.current_pt = None


class ContourFilterPen(_PassThruComponentsMixin, RecordingPen):
    """A "buffered" filter pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output pen.

    Components are passed through unchanged.
    """

    def __init__(self, outPen):
        super(ContourFilterPen, self).__init__()
        self._outPen = outPen

    def closePath(self):
        super(ContourFilterPen, self).closePath()
        self._flushContour()

    def endPath(self):
        super(ContourFilterPen, self).endPath()
        self._flushContour()

    def _flushContour(self):
        result = self.filterContour(self.value)
        if result is not None:
            self.value = result
        self.replay(self._outPen)
        self.value = []

    def filterContour(self, contour):
        """Subclasses must override this to perform the filtering.

        The contour is a list of pen (operator, operands) tuples.
        Operators are strings corresponding to the AbstractPen methods:
        "moveTo", "lineTo", "curveTo", "qCurveTo", "closePath" and
        "endPath". The operands are the positional arguments that are
        passed to each method.

        If the method doesn't return a value (i.e. returns None), it's
        assumed that the argument was modified in-place.
        Otherwise, the return value is drawn with the output pen.
        """
        return  # or return contour


class FilterPointPen(_PassThruComponentsMixin, AbstractPointPen):
    """Baseclass for point pens that apply some transformation to the
    coordinates they receive and pass them to another point pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = FilterPointPen(rec)
    >>> v = iter(rec.value)
    >>> pen.beginPath(identifier="abc")
    >>> next(v)
    ('beginPath', (), {'identifier': 'abc'})
    >>> pen.addPoint((1, 2), "line", False)
    >>> next(v)
    ('addPoint', ((1, 2), 'line', False, None), {})
    >>> pen.addComponent("a", (2, 0, 0, 2, 10, -10), identifier="0001")
    >>> next(v)
    ('addComponent', ('a', (2, 0, 0, 2, 10, -10)), {'identifier': '0001'})
    >>> pen.endPath()
    >>> next(v)
    ('endPath', (), {})
    """

    def __init__(self, outPen):
        self._outPen = outPen

    def beginPath(self, identifier=None, **kwargs):
        kwargs = dict(kwargs)
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._outPen.beginPath(**kwargs)

    def endPath(self):
        self._outPen.endPath()

    def addPoint(
        self,
        pt,
        segmentType=None,
        smooth=False,
        name=None,
        identifier=None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._outPen.addPoint(pt, segmentType, smooth, name, **kwargs)


class _DecomposingFilterMixinBase:
    """Base mixin class with common `addComponent` logic for decomposing filter pens."""

    def addComponent(self, baseGlyphName, transformation, **kwargs):
        # only decompose the component if it's included in the set
        if self.include is None or baseGlyphName in self.include:
            # if we're decomposing nested components, temporarily set include to None
            include_bak = self.include
            if self.decomposeNested and self.include:
                self.include = None
            try:
                super().addComponent(baseGlyphName, transformation, **kwargs)
            finally:
                if self.include != include_bak:
                    self.include = include_bak
        else:
            _PassThruComponentsMixin.addComponent(
                self, baseGlyphName, transformation, **kwargs
            )


class _DecomposingFilterPenMixin(_DecomposingFilterMixinBase):
    """Mixin class that decomposes components as regular contours for segment pens.

    Used by DecomposingFilterPen.

    Takes two required parameters, another segment pen 'outPen' to draw
    with, and a 'glyphSet' dict of drawable glyph objects to draw components from.

    The 'skipMissingComponents' and 'reverseFlipped' optional arguments work the
    same as in the DecomposingPen. reverseFlipped is bool only (True/False).

    In addition, the decomposing filter pens also take the following two options:

    'include' is an optional set of component base glyph names to consider for
    decomposition; the default include=None means decompose all components no matter
    the base glyph name).

    'decomposeNested' (bool) controls whether to recurse decomposition into nested
    components of components (this only matters when 'include' was also provided);
    if False, only decompose top-level components included in the set, but not
    also their children.
    """

    # raises MissingComponentError if base glyph is not found in glyphSet
    skipMissingComponents = False

    def __init__(
        self,
        outPen,
        glyphSet,
        skipMissingComponents=None,
        reverseFlipped: bool = False,
        include: set[str] | None = None,
        decomposeNested: bool = True,
        **kwargs,
    ):
        assert isinstance(
            reverseFlipped, bool
        ), f"Expected bool, got {type(reverseFlipped).__name__}"
        super().__init__(
            outPen=outPen,
            glyphSet=glyphSet,
            skipMissingComponents=skipMissingComponents,
            reverseFlipped=reverseFlipped,
            **kwargs,
        )
        self.include = include
        self.decomposeNested = decomposeNested


class _DecomposingFilterPointPenMixin(_DecomposingFilterMixinBase):
    """Mixin class that decomposes components as regular contours for point pens.

    Takes two required parameters, another point pen 'outPen' to draw
    with, and a 'glyphSet' dict of drawable glyph objects to draw components from.

    The 'skipMissingComponents' and 'reverseFlipped' optional arguments work the
    same as in the DecomposingPointPen. reverseFlipped accepts bool | ReverseFlipped
    (see DecomposingPointPen).

    In addition, the decomposing filter pens also take the following two options:

    'include' is an optional set of component base glyph names to consider for
    decomposition; the default include=None means decompose all components no matter
    the base glyph name).

    'decomposeNested' (bool) controls whether to recurse decomposition into nested
    components of components (this only matters when 'include' was also provided);
    if False, only decompose top-level components included in the set, but not
    also their children.
    """

    # raises MissingComponentError if base glyph is not found in glyphSet
    skipMissingComponents = False

    def __init__(
        self,
        outPen,
        glyphSet,
        skipMissingComponents=None,
        reverseFlipped: bool | ReverseFlipped = False,
        include: set[str] | None = None,
        decomposeNested: bool = True,
        **kwargs,
    ):
        super().__init__(
            outPen=outPen,
            glyphSet=glyphSet,
            skipMissingComponents=skipMissingComponents,
            reverseFlipped=reverseFlipped,
            **kwargs,
        )
        self.include = include
        self.decomposeNested = decomposeNested


class DecomposingFilterPen(_DecomposingFilterPenMixin, DecomposingPen, FilterPen):
    """Filter pen that draws components as regular contours."""

    pass


class DecomposingFilterPointPen(
    _DecomposingFilterPointPenMixin, DecomposingPointPen, FilterPointPen
):
    """Filter point pen that draws components as regular contours."""

    pass


class ContourFilterPointPen(_PassThruComponentsMixin, AbstractPointPen):
    """A "buffered" filter point pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output point pen.

    Components are passed through unchanged.

    The ``filterContour`` method can modify the contour in-place (return None)
    or return a new contour to replace it.
    """

    def __init__(self, outPen):
        self._outPen = outPen
        self.currentContour = None
        self.currentContourKwargs = None

    def beginPath(self, identifier=None, **kwargs):
        if self.currentContour is not None:
            raise ValueError("Path already begun")
        kwargs = dict(kwargs)
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.currentContour = []
        self.currentContourKwargs = kwargs

    def endPath(self):
        if self.currentContour is None:
            raise ValueError("Path not begun")
        self._flushContour()
        self.currentContour = None
        self.currentContourKwargs = None

    def _flushContour(self):
        """Flush the current contour to the output pen."""
        result = self.filterContour(self.currentContour)
        if result is not None:
            self.currentContour = result

        # Draw the filtered contour
        self._outPen.beginPath(**self.currentContourKwargs)
        for pt, segmentType, smooth, name, kwargs in self.currentContour:
            self._outPen.addPoint(pt, segmentType, smooth, name, **kwargs)
        self._outPen.endPath()

    def filterContour(self, contour):
        """Subclasses must override this to perform the filtering.

        The contour is a list of (pt, segmentType, smooth, name, kwargs) tuples.
        If the method doesn't return a value (i.e. returns None), it's
        assumed that the contour was modified in-place.
        Otherwise, the return value replaces the original contour.
        """
        return  # or return contour

    def addPoint(
        self,
        pt,
        segmentType=None,
        smooth=False,
        name=None,
        identifier=None,
        **kwargs,
    ):
        if self.currentContour is None:
            raise ValueError("Path not begun")
        kwargs = dict(kwargs)
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.currentContour.append((pt, segmentType, smooth, name, kwargs))


class OnCurveFirstPointPen(ContourFilterPointPen):
    """Filter point pen that ensures closed contours start with an on-curve point.

    If a closed contour starts with an off-curve point (segmentType=None), it rotates
    the points list so that the first on-curve point (segmentType != None) becomes
    the start point. Open contours and contours already starting with on-curve points
    are passed through unchanged.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = OnCurveFirstPointPen(rec)
    >>> # Closed contour starting with off-curve - will be rotated
    >>> pen.beginPath()
    >>> pen.addPoint((0, 0), None)  # off-curve
    >>> pen.addPoint((100, 100), "line")  # on-curve - will become start
    >>> pen.addPoint((200, 0), None)  # off-curve
    >>> pen.addPoint((300, 100), "curve")  # on-curve
    >>> pen.endPath()
    >>> # The contour should now start with (100, 100) "line"
    >>> rec.value[0]
    ('beginPath', (), {})
    >>> rec.value[1]
    ('addPoint', ((100, 100), 'line', False, None), {})
    >>> rec.value[2]
    ('addPoint', ((200, 0), None, False, None), {})
    >>> rec.value[3]
    ('addPoint', ((300, 100), 'curve', False, None), {})
    >>> rec.value[4]
    ('addPoint', ((0, 0), None, False, None), {})
    """

    def filterContour(self, contour):
        """Rotate closed contour to start with first on-curve point if needed."""
        if not contour:
            return

        # Check if it's a closed contour (no "move" segmentType)
        is_closed = contour[0][1] != "move"

        if is_closed and contour[0][1] is None:
            # Closed contour starting with off-curve - need to rotate
            # Find the first on-curve point
            for i, (pt, segmentType, smooth, name, kwargs) in enumerate(contour):
                if segmentType is not None:
                    # Rotate the points list so it starts with the first on-curve point
                    return contour[i:] + contour[:i]
<|endoftext|>from fontTools.pens.basePen import BasePen, OpenContourError

try:
    import cython
except (AttributeError, ImportError):
    # if cython not installed, use mock module with no-op decorators and types
    from fontTools.misc import cython
COMPILED = cython.compiled


__all__ = ["MomentsPen"]


class MomentsPen(BasePen):

    def __init__(self, glyphset=None):
        BasePen.__init__(self, glyphset)

        self.area = 0
        self.momentX = 0
        self.momentY = 0
        self.momentXX = 0
        self.momentXY = 0
        self.momentYY = 0

    def _moveTo(self, p0):
        self._startPoint = p0

    def _closePath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            self._lineTo(self._startPoint)

    def _endPath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            raise OpenContourError("Glyph statistics is not defined on open contours.")

    @cython.locals(r0=cython.double)
    @cython.locals(r1=cython.double)
    @cython.locals(r2=cython.double)
    @cython.locals(r3=cython.double)
    @cython.locals(r4=cython.double)
    @cython.locals(r5=cython.double)
    @cython.locals(r6=cython.double)
    @cython.locals(r7=cython.double)
    @cython.locals(r8=cython.double)
    @cython.locals(r9=cython.double)
    @cython.locals(r10=cython.double)
    @cython.locals(r11=cython.double)
    @cython.locals(r12=cython.double)
    @cython.locals(x0=cython.double, y0=cython.double)
    @cython.locals(x1=cython.double, y1=cython.double)
    def _lineTo(self, p1):
        x0, y0 = self._getCurrentPoint()
        x1, y1 = p1

        r0 = x1 * y0
        r1 = x1 * y1
        r2 = x1**2
        r3 = r2 * y1
        r4 = y0 - y1
        r5 = r4 * x0
        r6 = x0**2
        r7 = 2 * y0
        r8 = y0**2
        r9 = y1**2
        r10 = x1**3
        r11 = y0**3
        r12 = y1**3

        self.area += -r0 / 2 - r1 / 2 + x0 * (y0 + y1) / 2
        self.momentX += -r2 * y0 / 6 - r3 / 3 - r5 * x1 / 6 + r6 * (r7 + y1) / 6
        self.momentY += (
            -r0 * y1 / 6 - r8 * x1 / 6 - r9 * x1 / 6 + x0 * (r8 + r9 + y0 * y1) / 6
        )
        self.momentXX += (
            -r10 * y0 / 12
            - r10 * y1 / 4
            - r2 * r5 / 12
            - r4 * r6 * x1 / 12
            + x0**3 * (3 * y0 + y1) / 12
        )
        self.momentXY += (
            -r2 * r8 / 24
            - r2 * r9 / 8
            - r3 * r7 / 24
            + r6 * (r7 * y1 + 3 * r8 + r9) / 24
            - x0 * x1 * (r8 - r9) / 12
        )
        self.momentYY += (
            -r0 * r9 / 12
            - r1 * r8 / 12
            - r11 * x1 / 12
            - r12 * x1 / 12
            + x0 * (r11 + r12 + r8 * y1 + r9 * y0) / 12
        )

    @cython.locals(r0=cython.double)
    @cython.locals(r1=cython.double)
    @cython.locals(r2=cython.double)
    @cython.locals(r3=cython.double)
    @cython.locals(r4=cython.double)
    @cython.locals(r5=cython.double)
    @cython.locals(r6=cython.double)
    @cython.locals(r7=cython.double)
    @cython.locals(r8=cython.double)
    @cython.locals(r9=cython.double)
    @cython.locals(r10=cython.double)
    @cython.locals(r11=cython.double)
    @cython.locals(r12=cython.double)
    @cython.locals(r13=cython.double)
    @cython.locals(r14=cython.double)
    @cython.locals(r15=cython.double)
    @cython.locals(r16=cython.double)
    @cython.locals(r17=cython.double)
    @cython.locals(r18=cython.double)
    @cython.locals(r19=cython.double)
    @cython.locals(r20=cython.double)
    @cython.locals(r21=cython.double)
    @cython.locals(r22=cython.double)
    @cython.locals(r23=cython.double)
    @cython.locals(r24=cython.double)
    @cython.locals(r25=cython.double)
    @cython.locals(r26=cython.double)
    @cython.locals(r27=cython.double)
    @cython.locals(r28=cython.double)
    @cython.locals(r29=cython.double)
    @cython.locals(r30=cython.double)
    @cython.locals(r31=cython.double)
    @cython.locals(r32=cython.double)
    @cython.locals(r33=cython.double)
    @cython.locals(r34=cython.double)
    @cython.locals(r35=cython.double)
    @cython.locals(r36=cython.double)
    @cython.locals(r37=cython.double)
    @cython.locals(r38=cython.double)
    @cython.locals(r39=cython.double)
    @cython.locals(r40=cython.double)
    @cython.locals(r41=cython.double)
    @cython.locals(r42=cython.double)
    @cython.locals(r43=cython.double)
    @cython.locals(r44=cython.double)
    @cython.locals(r45=cython.double)
    @cython.locals(r46=cython.double)
    @cython.locals(r47=cython.double)
    @cython.locals(r48=cython.double)
    @cython.locals(r49=cython.double)
    @cython.locals(r50=cython.double)
    @cython.locals(r51=cython.double)
    @cython.locals(r52=cython.double)
    @cython.locals(r53=cython.double)
    @cython.locals(x0=cython.double, y0=cython.double)
    @cython.locals(x1=cython.double, y1=cython.double)
    @cython.locals(x2=cython.double, y2=cython.double)
    def _qCurveToOne(self, p1, p2):
        x0, y0 = self._getCurrentPoint()
        x1, y1 = p1
        x2, y2 = p2

        r0 = 2 * y1
        r1 = r0 * x2
        r2 = x2 * y2
        r3 = 3 * r2
        r4 = 2 * x1
        r5 = 3 * y0
        r6 = x1**2
        r7 = x2**2
        r8 = 4 * y1
        r9 = 10 * y2
        r10 = 2 * y2
        r11 = r4 * x2
        r12 = x0**2
        r13 = 10 * y0
        r14 = r4 * y2
        r15 = x2 * y0
        r16 = 4 * x1
        r17 = r0 * x1 + r2
        r18 = r2 * r8
        r19 = y1**2
        r20 = 2 * r19
        r21 = y2**2
        r22 = r21 * x2
        r23 = 5 * r22
        r24 = y0**2
        r25 = y0 * y2
        r26 = 5 * r24
        r27 = x1**3
        r28 = x2**3
        r29 = 30 * y1
        r30 = 6 * y1
        r31 = 10 * r7 * x1
        r32 = 5 * y2
        r33 = 12 * r6
        r34 = 30 * x1
        r35 = x1 * y1
        r36 = r3 + 20 * r35
        r37 = 12 * x1
        r38 = 20 * r6
        r39 = 8 * r6 * y1
        r40 = r32 * r7
        r41 = 60 * y1
        r42 = 20 * r19
        r43 = 4 * r19
        r44 = 15 * r21
        r45 = 12 * x2
        r46 = 12 * y2
        r47 = 6 * x1
        r48 = 8 * r19 * x1 + r23
        r49 = 8 * y1**3
        r50 = y2**3
        r51 = y0**3
        r52 = 10 * y1
        r53 = 12 * y1

        self.area += (
            -r1 / 6
            - r3 / 6
            + x0 * (r0 + r5 + y2) / 6
            + x1 * y2 / 3
            - y0 * (r4 + x2) / 6
        )
        self.momentX += (
            -r11 * (-r10 + y1) / 30
            + r12 * (r13 + r8 + y2) / 30
            + r6 * y2 / 15
            - r7 * r8 / 30
            - r7 * r9 / 30
            + x0 * (r14 - r15 - r16 * y0 + r17) / 30
            - y0 * (r11 + 2 * r6 + r7) / 30
        )
        self.momentY += (
            -r18 / 30
            - r20 * x2 / 30
            - r23 / 30
            - r24 * (r16 + x2) / 30
            + x0 * (r0 * y2 + r20 + r21 + r25 + r26 + r8 * y0) / 30
            + x1 * y2 * (r10 + y1) / 15
            - y0 * (r1 + r17) / 30
        )
        self.momentXX += (
            r12 * (r1 - 5 * r15 - r34 * y0 + r36 + r9 * x1) / 420
            + 2 * r27 * y2 / 105
            - r28 * r29 / 420
            - r28 * y2 / 4
            - r31 * (r0 - 3 * y2) / 420
            - r6 * x2 * (r0 - r32) / 105
            + x0**3 * (r30 + 21 * y0 + y2) / 84
            - x0
            * (
                r0 * r7
                + r15 * r37
                - r2 * r37
                - r33 * y2
                + r38 * y0
                - r39
                - r40
                + r5 * r7
            )
            / 420
            - y0 * (8 * r27 + 5 * r28 + r31 + r33 * x2) / 420
        )
        self.momentXY += (
            r12 * (r13 * y2 + 3 * r21 + 105 * r24 + r41 * y0 + r42 + r46 * y1) / 840
            - r16 * x2 * (r43 - r44) / 840
            - r21 * r7 / 8
            - r24 * (r38 + r45 * x1 + 3 * r7) / 840
            - r41 * r7 * y2 / 840
            - r42 * r7 / 840
            + r6 * y2 * (r32 + r8) / 210
            + x0
            * (
                -r15 * r8
                + r16 * r25
                + r18
                + r21 * r47
                - r24 * r34
                - r26 * x2
                + r35 * r46
                + r48
            )
            / 420
            - y0 * (r16 * r2 + r30 * r7 + r35 * r45 + r39 + r40) / 420
        )
        self.momentYY += (
            -r2 * r42 / 420
            - r22 * r29 / 420
            - r24 * (r14 + r36 + r52 * x2) / 420
            - r49 * x2 / 420
            - r50 * x2 / 12
            - r51 * (r47 + x2) / 84
            + x0
            * (
                r19 * r46
                + r21 * r5
                + r21 * r52
                + r24 * r29
                + r25 * r53
                + r26 * y2
                + r42 * y0
                + r49
                + 5 * r50
                + 35 * r51
            )
            / 420
            + x1 * y2 * (r43 + r44 + r9 * y1) / 210
            - y0 * (r19 * r45 + r2 * r53 - r21 * r4 + r48) / 420
        )

    @cython.locals(r0=cython.double)
    @cython.locals(r1=cython.double)
    @cython.locals(r2=cython.double)
    @cython.locals(r3=cython.double)
    @cython.locals(r4=cython.double)
    @cython.locals(r5=cython.double)
    @cython.locals(r6=cython.double)
    @cython.locals(r7=cython.double)
    @cython.locals(r8=cython.double)
    @cython.locals(r9=cython.double)
    @cython.locals(r10=cython.double)
    @cython.locals(r11=cython.double)
    @cython.locals(r12=cython.double)
    @cython.locals(r13=cython.double)
    @cython.locals(r14=cython.double)
    @cython.locals(r15=cython.double)
    @cython.locals(r16=cython.double)
    @cython.locals(r17=cython.double)
    @cython.locals(r18=cython.double)
    @cython.locals(r19=cython.double)
    @cython.locals(r20=cython.double)
    @cython.locals(r21=cython.double)
    @cython.locals(r22=cython.double)
    @cython.locals(r23=cython.double)
    @cython.locals(r24=cython.double)
    @cython.locals(r25=cython.double)
    @cython.locals(r26=cython.double)
    @cython.locals(r27=cython.double)
    @cython.locals(r28=cython.double)
    @cython.locals(r29=cython.double)
    @cython.locals(r30=cython.double)
    @cython.locals(r31=cython.double)
    @cython.locals(r32=cython.double)
    @cython.locals(r33=cython.double)
    @cython.locals(r34=cython.double)
    @cython.locals(r35=cython.double)
    @cython.locals(r36=cython.double)
    @cython.locals(r37=cython.double)
    @cython.locals(r38=cython.double)
    @cython.locals(r39=cython.double)
    @cython.locals(r40=cython.double)
    @cython.locals(r41=cython.double)
    @cython.locals(r42=cython.double)
    @cython.locals(r43=cython.double)
    @cython.locals(r44=cython.double)
    @cython.locals(r45=cython.double)
    @cython.locals(r46=cython.double)
    @cython.locals(r47=cython.double)
    @cython.locals(r48=cython.double)
    @cython.locals(r49=cython.double)
    @cython.locals(r50=cython.double)
    @cython.locals(r51=cython.double)
    @cython.locals(r52=cython.double)
    @cython.locals(r53=cython.double)
    @cython.locals(r54=cython.double)
    @cython.locals(r55=cython.double)
    @cython.locals(r56=cython.double)
    @cython.locals(r57=cython.double)
    @cython.locals(r58=cython.double)
    @cython.locals(r59=cython.double)
    @cython.locals(r60=cython.double)
    @cython.locals(r61=cython.double)
    @cython.locals(r62=cython.double)
    @cython.locals(r63=cython.double)
    @cython.locals(r64=cython.double)
    @cython.locals(r65=cython.double)
    @cython.locals(r66=cython.double)
    @cython.locals(r67=cython.double)
    @cython.locals(r68=cython.double)
    @cython.locals(r69=cython.double)
    @cython.locals(r70=cython.double)
    @cython.locals(r71=cython.double)
    @cython.locals(r72=cython.double)
    @cython.locals(r73=cython.double)
    @cython.locals(r74=cython.double)
    @cython.locals(r75=cython.double)
    @cython.locals(r76=cython.double)
    @cython.locals(r77=cython.double)
    @cython.locals(r78=cython.double)
    @cython.locals(r79=cython.double)
    @cython.locals(r80=cython.double)
    @cython.locals(r81=cython.double)
    @cython.locals(r82=cython.double)
    @cython.locals(r83=cython.double)
    @cython.locals(r84=cython.double)
    @cython.locals(r85=cython.double)
    @cython.locals(r86=cython.double)
    @cython.locals(r87=cython.double)
    @cython.locals(r88=cython.double)
    @cython.locals(r89=cython.double)
    @cython.locals(r90=cython.double)
    @cython.locals(r91=cython.double)
    @cython.locals(r92=cython.double)
    @cython.locals(r93=cython.double)
    @cython.locals(r94=cython.double)
    @cython.locals(r95=cython.double)
    @cython.locals(r96=cython.double)
    @cython.locals(r97=cython.double)
    @cython.locals(r98=cython.double)
    @cython.locals(r99=cython.double)
    @cython.locals(r100=cython.double)
    @cython.locals(r101=cython.double)
    @cython.locals(r102=cython.double)
    @cython.locals(r103=cython.double)
    @cython.locals(r104=cython.double)
    @cython.locals(r105=cython.double)
    @cython.locals(r106=cython.double)
    @cython.locals(r107=cython.double)
    @cython.locals(r108=cython.double)
    @cython.locals(r109=cython.double)
    @cython.locals(r110=cython.double)
    @cython.locals(r111=cython.double)
    @cython.locals(r112=cython.double)
    @cython.locals(r113=cython.double)
    @cython.locals(r114=cython.double)
    @cython.locals(r115=cython.double)
    @cython.locals(r116=cython.double)
    @cython.locals(r117=cython.double)
    @cython.locals(r118=cython.double)
    @cython.locals(r119=cython.double)
    @cython.locals(r120=cython.double)
    @cython.locals(r121=cython.double)
    @cython.locals(r122=cython.double)
    @cython.locals(r123=cython.double)
    @cython.locals(r124=cython.double)
    @cython.locals(r125=cython.double)
    @cython.locals(r126=cython.double)
    @cython.locals(r127=cython.double)
    @cython.locals(r128=cython.double)
    @cython.locals(r129=cython.double)
    @cython.locals(r130=cython.double)
    @cython.locals(r131=cython.double)
    @cython.locals(r132=cython.double)
    @cython.locals(x0=cython.double, y0=cython.double)
    @cython.locals(x1=cython.double, y1=cython.double)
    @cython.locals(x2=cython.double, y2=cython.double)
    @cython.locals(x3=cython.double, y3=cython.double)
    def _curveToOne(self, p1, p2, p3):
        x0, y0 = self._getCurrentPoint()
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        r0 = 6 * y2
        r1 = r0 * x3
        r2 = 10 * y3
        r3 = r2 * x3
        r4 = 3 * y1
        r5 = 6 * x1
        r6 = 3 * x2
        r7 = 6 * y1
        r8 = 3 * y2
        r9 = x2**2
        r10 = 45 * r9
        r11 = r10 * y3
        r12 = x3**2
        r13 = r12 * y2
        r14 = r12 * y3
        r15 = 7 * y3
        r16 = 15 * x3
        r17 = r16 * x2
        r18 = x1**2
        r19 = 9 * r18
        r20 = x0**2
        r21 = 21 * y1
        r22 = 9 * r9
        r23 = r7 * x3
        r24 = 9 * y2
        r25 = r24 * x2 + r3
        r26 = 9 * x2
        r27 = x2 * y3
        r28 = -r26 * y1 + 15 * r27
        r29 = 3 * x1
        r30 = 45 * x1
        r31 = 12 * x3
        r32 = 45 * r18
        r33 = 5 * r12
        r34 = r8 * x3
        r35 = 105 * y0
        r36 = 30 * y0
        r37 = r36 * x2
        r38 = 5 * x3
        r39 = 15 * y3
        r40 = 5 * y3
        r41 = r40 * x3
        r42 = x2 * y2
        r43 = 18 * r42
        r44 = 45 * y1
        r45 = r41 + r43 + r44 * x1
        r46 = y2 * y3
        r47 = r46 * x3
        r48 = y2**2
        r49 = 45 * r48
        r50 = r49 * x3
        r51 = y3**2
        r52 = r51 * x3
        r53 = y1**2
        r54 = 9 * r53
        r55 = y0**2
        r56 = 21 * x1
        r57 = 6 * x2
        r58 = r16 * y2
        r59 = r39 * y2
        r60 = 9 * r48
        r61 = r6 * y3
        r62 = 3 * y3
        r63 = r36 * y2
        r64 = y1 * y3
        r65 = 45 * r53
        r66 = 5 * r51
        r67 = x2**3
        r68 = x3**3
        r69 = 630 * y2
        r70 = 126 * x3
        r71 = x1**3
        r72 = 126 * x2
        r73 = 63 * r9
        r74 = r73 * x3
        r75 = r15 * x3 + 15 * r42
        r76 = 630 * x1
        r77 = 14 * x3
        r78 = 21 * r27
        r79 = 42 * x1
        r80 = 42 * x2
        r81 = x1 * y2
        r82 = 63 * r42
        r83 = x1 * y1
        r84 = r41 + r82 + 378 * r83
        r85 = x2 * x3
        r86 = r85 * y1
        r87 = r27 * x3
        r88 = 27 * r9
        r89 = r88 * y2
        r90 = 42 * r14
        r91 = 90 * x1
        r92 = 189 * r18
        r93 = 378 * r18
        r94 = r12 * y1
        r95 = 252 * x1 * x2
        r96 = r79 * x3
        r97 = 30 * r85
        r98 = r83 * x3
        r99 = 30 * x3
        r100 = 42 * x3
        r101 = r42 * x1
        r102 = r10 * y2 + 14 * r14 + 126 * r18 * y1 + r81 * r99
        r103 = 378 * r48
        r104 = 18 * y1
        r105 = r104 * y2
        r106 = y0 * y1
        r107 = 252 * y2
        r108 = r107 * y0
        r109 = y0 * y3
        r110 = 42 * r64
        r111 = 378 * r53
        r112 = 63 * r48
        r113 = 27 * x2
        r114 = r27 * y2
        r115 = r113 * r48 + 42 * r52
        r116 = x3 * y3
        r117 = 54 * r42
        r118 = r51 * x1
        r119 = r51 * x2
        r120 = r48 * x1
        r121 = 21 * x3
        r122 = r64 * x1
        r123 = r81 * y3
        r124 = 30 * r27 * y1 + r49 * x2 + 14 * r52 + 126 * r53 * x1
        r125 = y2**3
        r126 = y3**3
        r127 = y1**3
        r128 = y0**3
        r129 = r51 * y2
        r130 = r112 * y3 + r21 * r51
        r131 = 189 * r53
        r132 = 90 * y2

        self.area += (
            -r1 / 20
            - r3 / 20
            - r4 * (x2 + x3) / 20
            + x0 * (r7 + r8 + 10 * y0 + y3) / 20
            + 3 * x1 * (y2 + y3) / 20
            + 3 * x2 * y3 / 10
            - y0 * (r5 + r6 + x3) / 20
        )
        self.momentX += (
            r11 / 840
            - r13 / 8
            - r14 / 3
            - r17 * (-r15 + r8) / 840
            + r19 * (r8 + 2 * y3) / 840
            + r20 * (r0 + r21 + 56 * y0 + y3) / 168
            + r29 * (-r23 + r25 + r28) / 840
            - r4 * (10 * r12 + r17 + r22) / 840
            + x0
            * (
                12 * r27
                + r30 * y2
                + r34
                - r35 * x1
                - r37
                - r38 * y0
                + r39 * x1
                - r4 * x3
                + r45
            )
            / 840
            - y0 * (r17 + r30 * x2 + r31 * x1 + r32 + r33 + 18 * r9) / 840
        )
        self.momentY += (
            -r4 * (r25 + r58) / 840
            - r47 / 8
            - r50 / 840
            - r52 / 6
            - r54 * (r6 + 2 * x3) / 840
            - r55 * (r56 + r57 + x3) / 168
            + x0
            * (
                r35 * y1
                + r40 * y0
                + r44 * y2
                + 18 * r48
                + 140 * r55
                + r59
                + r63
                + 12 * r64
                + r65
                + r66
            )
            / 840
            + x1 * (r24 * y1 + 10 * r51 + r59 + r60 + r7 * y3) / 280
            + x2 * y3 * (r15 + r8) / 56
            - y0 * (r16 * y1 + r31 * y2 + r44 * x2 + r45 + r61 - r62 * x1) / 840
        )
        self.momentXX += (
            -r12 * r72 * (-r40 + r8) / 9240
            + 3 * r18 * (r28 + r34 - r38 * y1 + r75) / 3080
            + r20
            * (
                r24 * x3
                - r72 * y0
                - r76 * y0
                - r77 * y0
                + r78
                + r79 * y3
                + r80 * y1
                + 210 * r81
                + r84
            )
            / 9240
            - r29
            * (
                r12 * r21
                + 14 * r13
                + r44 * r9
                - r73 * y3
                + 54 * r86
                - 84 * r87
                - r89
                - r90
            )
            / 9240
            - r4 * (70 * r12 * x2 + 27 * r67 + 42 * r68 + r74) / 9240
            + 3 * r67 * y3 / 220
            - r68 * r69 / 9240
            - r68 * y3 / 4
            - r70 * r9 * (-r62 + y2) / 9240
            + 3 * r71 * (r24 + r40) / 3080
            + x0**3 * (r24 + r44 + 165 * y0 + y3) / 660
            + x0
            * (
                r100 * r27
                + 162 * r101
                + r102
                + r11
                + 63 * r18 * y3
                + r27 * r91
                - r33 * y0
                - r37 * x3
                + r43 * x3
                - r73 * y0
                - r88 * y1
                + r92 * y2
                - r93 * y0
                - 9 * r94
                - r95 * y0
                - r96 * y0
                - r97 * y1
                - 18 * r98
                + r99 * x1 * y3
            )
            / 9240
            - y0
            * (
                r12 * r56
                + r12 * r80
                + r32 * x3
                + 45 * r67
                + 14 * r68
                + 126 * r71
                + r74
                + r85 * r91
                + 135 * r9 * x1
                + r92 * x2
            )
            / 9240
        )
        self.momentXY += (
            -r103 * r12 / 18480
            - r12 * r51 / 8
            - 3 * r14 * y2 / 44
            + 3 * r18 * (r105 + r2 * y1 + 18 * r46 + 15 * r48 + 7 * r51) / 6160
            + r20
            * (
                1260 * r106
                + r107 * y1
                + r108
                + 28 * r109
                + r110
                + r111
                + r112
                + 30 * r46
                + 2310 * r55
                + r66
            )
            / 18480
            - r54 * (7 * r12 + 18 * r85 + 15 * r9) / 18480
            - r55 * (r33 + r73 + r93 + r95 + r96 + r97) / 18480
            - r7 * (42 * r13 + r82 * x3 + 28 * r87 + r89 + r90) / 18480
            - 3 * r85 * (r48 - r66) / 220
            + 3 * r9 * y3 * (r62 + 2 * y2) / 440
            + x0
            * (
                -r1 * y0
                - 84 * r106 * x2
                + r109 * r56
                + 54 * r114
                + r117 * y1
                + 15 * r118
                + 21 * r119
                + 81 * r120
                + r121 * r46
                + 54 * r122
                + 60 * r123
                + r124
                - r21 * x3 * y0
                + r23 * y3
                - r54 * x3
                - r55 * r72
                - r55 * r76
                - r55 * r77
                + r57 * y0 * y3
                + r60 * x3
                + 84 * r81 * y0
                + 189 * r81 * y1
            )
            / 9240
            + x1
            * (
                r104 * r27
                - r105 * x3
                - r113 * r53
                + 63 * r114
                + r115
                - r16 * r53
                + 28 * r47
                + r51 * r80
            )
            / 3080
            - y0
            * (
                54 * r101
                + r102
                + r116 * r5
                + r117 * x3
                + 21 * r13
                - r19 * y3
                + r22 * y3
                + r78 * x3
                + 189 * r83 * x2
                + 60 * r86
                + 81 * r9 * y1
                + 15 * r94
                + 54 * r98
            )
            / 9240
        )
        self.momentYY += (
            -r103 * r116 / 9240
            - r125 * r70 / 9240
            - r126 * x3 / 12
            - 3 * r127 * (r26 + r38) / 3080
            - r128 * (r26 + r30 + x3) / 660
            - r4 * (r112 * x3 + r115 - 14 * r119 + 84 * r47) / 9240
            - r52 * r69 / 9240
            - r54 * (r58 + r61 + r75) / 9240
            - r55
            * (r100 * y1 + r121 * y2 + r26 * y3 + r79 * y2 + r84 + 210 * x2 * y1)
            / 9240
            + x0
            * (
                r108 * y1
                + r110 * y0
                + r111 * y0
                + r112 * y0
                + 45 * r125
                + 14 * r126
                + 126 * r127
                + 770 * r128
                + 42 * r129
                + r130
                + r131 * y2
                + r132 * r64
                + 135 * r48 * y1
                + 630 * r55 * y1
                + 126 * r55 * y2
                + 14 * r55 * y3
                + r63 * y3
                + r65 * y3
                + r66 * y0
            )
            / 9240
            + x1
            * (
                27 * r125
                + 42 * r126
                + 70 * r129
                + r130
                + r39 * r53
                + r44 * r48
                + 27 * r53 * y2
                + 54 * r64 * y2
            )
            / 3080
            + 3 * x2 * y3 * (r48 + r66 + r8 * y3) / 220
            - y0
            * (
                r100 * r46
                + 18 * r114
                - 9 * r118
                - 27 * r120
                - 18 * r122
                - 30 * r123
                + r124
                + r131 * x2
                + r132 * x3 * y1
                + 162 * r42 * y1
                + r50
                + 63 * r53 * x3
                + r64 * r99
            )
            / 9240
        )


if __name__ == "__main__":
    from fontTools.misc.symfont import x, y, printGreenPen

    printGreenPen(
        "MomentsPen",
        [
            ("area", 1),
            ("momentX", x),
            ("momentY", y),
            ("momentXX", x**2),
            ("momentXY", x * y),
            ("momentYY", y**2),
        ],
    )
<|endoftext|>"""Calculate the area of a glyph."""

from fontTools.pens.basePen import BasePen

__all__ = ["AreaPen"]


class AreaPen(BasePen):
    def __init__(self, glyphset=None):
        BasePen.__init__(self, glyphset)
        self.value = 0

    def _moveTo(self, p0):
        self._p0 = self._startPoint = p0

    def _lineTo(self, p1):
        x0, y0 = self._p0
        x1, y1 = p1
        self.value -= (x1 - x0) * (y1 + y0) * 0.5
        self._p0 = p1

    def _qCurveToOne(self, p1, p2):
        # https://github.com/Pomax/bezierinfo/issues/44
        p0 = self._p0
        x0, y0 = p0[0], p0[1]
        x1, y1 = p1[0] - x0, p1[1] - y0
        x2, y2 = p2[0] - x0, p2[1] - y0
        self.value -= (x2 * y1 - x1 * y2) / 3
        self._lineTo(p2)
        self._p0 = p2

    def _curveToOne(self, p1, p2, p3):
        # https://github.com/Pomax/bezierinfo/issues/44
        p0 = self._p0
        x0, y0 = p0[0], p0[1]
        x1, y1 = p1[0] - x0, p1[1] - y0
        x2, y2 = p2[0] - x0, p2[1] - y0
        x3, y3 = p3[0] - x0, p3[1] - y0
        self.value -= (x1 * (-y2 - y3) + x2 * (y1 - 2 * y3) + x3 * (y1 + 2 * y2)) * 0.15
        self._lineTo(p3)
        self._p0 = p3

    def _closePath(self):
        self._lineTo(self._startPoint)
        del self._p0, self._startPoint

    def _endPath(self):
        if self._p0 != self._startPoint:
            # Area is not defined for open contours.
            raise NotImplementedError
        del self._p0, self._startPoint
<|endoftext|># Modified from https://github.com/adobe-type-tools/psautohint/blob/08b346865710ed3c172f1eb581d6ef243b203f99/python/psautohint/ufoFont.py#L800-L838
import hashlib

from fontTools.pens.basePen import MissingComponentError
from fontTools.pens.pointPen import AbstractPointPen


class HashPointPen(AbstractPointPen):
    """
    This pen can be used to check if a glyph's contents (outlines plus
    components) have changed.

    Components are added as the original outline plus each composite's
    transformation.

    Example: You have some TrueType hinting code for a glyph which you want to
    compile. The hinting code specifies a hash value computed with HashPointPen
    that was valid for the glyph's outlines at the time the hinting code was
    written. Now you can calculate the hash for the glyph's current outlines to
    check if the outlines have changed, which would probably make the hinting
    code invalid.

    > glyph = ufo[name]
    > hash_pen = HashPointPen(glyph.width, ufo)
    > glyph.drawPoints(hash_pen)
    > ttdata = glyph.lib.get("public.truetype.instructions", None)
    > stored_hash = ttdata.get("id", None)  # The hash is stored in the "id" key
    > if stored_hash is None or stored_hash != hash_pen.hash:
    >    logger.error(f"Glyph hash mismatch, glyph '{name}' will have no instructions in font.")
    > else:
    >    # The hash values are identical, the outline has not changed.
    >    # Compile the hinting code ...
    >    pass

    If you want to compare a glyph from a source format which supports floating point
    coordinates and transformations against a glyph from a format which has restrictions
    on the precision of floats, e.g. UFO vs. TTF, you must use an appropriate rounding
    function to make the values comparable. For TTF fonts with composites, this
    construct can be used to make the transform values conform to F2Dot14:

    > ttf_hash_pen = HashPointPen(ttf_glyph_width, ttFont.getGlyphSet())
    > ttf_round_pen = RoundingPointPen(ttf_hash_pen, transformRoundFunc=partial(floatToFixedToFloat, precisionBits=14))
    > ufo_hash_pen = HashPointPen(ufo_glyph.width, ufo)
    > ttf_glyph.drawPoints(ttf_round_pen, ttFont["glyf"])
    > ufo_round_pen = RoundingPointPen(ufo_hash_pen, transformRoundFunc=partial(floatToFixedToFloat, precisionBits=14))
    > ufo_glyph.drawPoints(ufo_round_pen)
    > assert ttf_hash_pen.hash == ufo_hash_pen.hash
    """

    def __init__(self, glyphWidth=0, glyphSet=None):
        self.glyphset = glyphSet
        self.data = ["w%s" % round(glyphWidth, 9)]

    @property
    def hash(self):
        data = "".join(self.data)
        if len(data) >= 128:
            data = hashlib.sha512(data.encode("ascii")).hexdigest()
        return data

    def beginPath(self, identifier=None, **kwargs):
        pass

    def endPath(self):
        self.data.append("|")

    def addPoint(
        self,
        pt,
        segmentType=None,
        smooth=False,
        name=None,
        identifier=None,
        **kwargs,
    ):
        if segmentType is None:
            pt_type = "o"  # offcurve
        else:
            pt_type = segmentType[0]
        self.data.append(f"{pt_type}{pt[0]:g}{pt[1]:+g}")

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        tr = "".join([f"{t:+}" for t in transformation])
        self.data.append("[")
        try:
            self.glyphset[baseGlyphName].drawPoints(self)
        except KeyError:
            raise MissingComponentError(baseGlyphName)
        self.data.append(f"({tr})]")
<|endoftext|>"""fontTools.pens.pointInsidePen -- Pen implementing "point inside" testing
for shapes.
"""

from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import solveQuadratic, solveCubic

__all__ = ["PointInsidePen"]


class PointInsidePen(BasePen):
    """This pen implements "point inside" testing: to test whether
    a given point lies inside the shape (black) or outside (white).
    Instances of this class can be recycled, as long as the
    setTestPoint() method is used to set the new point to test.

    :Example:
        .. code-block::

            pen = PointInsidePen(glyphSet, (100, 200))
            outline.draw(pen)
            isInside = pen.getResult()

    Both the even-odd algorithm and the non-zero-winding-rule
    algorithm are implemented. The latter is the default, specify
    True for the evenOdd argument of __init__ or setTestPoint
    to use the even-odd algorithm.
    """

    # This class implements the classical "shoot a ray from the test point
    # to infinity and count how many times it intersects the outline" (as well
    # as the non-zero variant, where the counter is incremented if the outline
    # intersects the ray in one direction and decremented if it intersects in
    # the other direction).
    # I found an amazingly clear explanation of the subtleties involved in
    # implementing this correctly for polygons here:
    #   http://graphics.cs.ucdavis.edu/~okreylos/TAship/Spring2000/PointInPolygon.html
    # I extended the principles outlined on that page to curves.

    def __init__(self, glyphSet, testPoint, evenOdd=False):
        BasePen.__init__(self, glyphSet)
        self.setTestPoint(testPoint, evenOdd)

    def setTestPoint(self, testPoint, evenOdd=False):
        """Set the point to test. Call this _before_ the outline gets drawn."""
        self.testPoint = testPoint
        self.evenOdd = evenOdd
        self.firstPoint = None
        self.intersectionCount = 0

    def getWinding(self):
        if self.firstPoint is not None:
            # always make sure the sub paths are closed; the algorithm only works
            # for closed paths.
            self.closePath()
        return self.intersectionCount

    def getResult(self):
        """After the shape has been drawn, getResult() returns True if the test
        point lies within the (black) shape, and False if it doesn't.
        """
        winding = self.getWinding()
        if self.evenOdd:
            result = winding % 2
        else:  # non-zero
            result = self.intersectionCount != 0
        return not not result

    def _addIntersection(self, goingUp):
        if self.evenOdd or goingUp:
            self.intersectionCount += 1
        else:
            self.intersectionCount -= 1

    def _moveTo(self, point):
        if self.firstPoint is not None:
            # always make sure the sub paths are closed; the algorithm only works
            # for closed paths.
            self.closePath()
        self.firstPoint = point

    def _lineTo(self, point):
        x, y = self.testPoint
        x1, y1 = self._getCurrentPoint()
        x2, y2 = point

        if x1 < x and x2 < x:
            return
        if y1 < y and y2 < y:
            return
        if y1 >= y and y2 >= y:
            return

        dx = x2 - x1
        dy = y2 - y1
        t = (y - y1) / dy
        ix = dx * t + x1
        if ix < x:
            return
        self._addIntersection(y2 > y1)

    def _curveToOne(self, bcp1, bcp2, point):
        x, y = self.testPoint
        x1, y1 = self._getCurrentPoint()
        x2, y2 = bcp1
        x3, y3 = bcp2
        x4, y4 = point

        if x1 < x and x2 < x and x3 < x and x4 < x:
            return
        if y1 < y and y2 < y and y3 < y and y4 < y:
            return
        if y1 >= y and y2 >= y and y3 >= y and y4 >= y:
            return

        dy = y1
        cy = (y2 - dy) * 3.0
        by = (y3 - y2) * 3.0 - cy
        ay = y4 - dy - cy - by
        solutions = sorted(solveCubic(ay, by, cy, dy - y))
        solutions = [t for t in solutions if -0.0 <= t <= 1.0]
        if not solutions:
            return

        dx = x1
        cx = (x2 - dx) * 3.0
        bx = (x3 - x2) * 3.0 - cx
        ax = x4 - dx - cx - bx

        above = y1 >= y
        lastT = None
        for t in solutions:
            if t == lastT:
                continue
            lastT = t
            t2 = t * t
            t3 = t2 * t

            direction = 3 * ay * t2 + 2 * by * t + cy
            incomingGoingUp = outgoingGoingUp = direction > 0.0
            if direction == 0.0:
                direction = 6 * ay * t + 2 * by
                outgoingGoingUp = direction > 0.0
                incomingGoingUp = not outgoingGoingUp
                if direction == 0.0:
                    direction = ay
                    incomingGoingUp = outgoingGoingUp = direction > 0.0

            xt = ax * t3 + bx * t2 + cx * t + dx
            if xt < x:
                continue

            if t in (0.0, -0.0):
                if not outgoingGoingUp:
                    self._addIntersection(outgoingGoingUp)
            elif t == 1.0:
                if incomingGoingUp:
                    self._addIntersection(incomingGoingUp)
            else:
                if incomingGoingUp == outgoingGoingUp:
                    self._addIntersection(outgoingGoingUp)
                # else:
                #   we're not really intersecting, merely touching

    def _qCurveToOne_unfinished(self, bcp, point):
        # XXX need to finish this, for now doing it through a cubic
        # (BasePen implements _qCurveTo in terms of a cubic) will
        # have to do.
        x, y = self.testPoint
        x1, y1 = self._getCurrentPoint()
        x2, y2 = bcp
        x3, y3 = point
        c = y1
        b = (y2 - c) * 2.0
        a = y3 - c - b
        solutions = sorted(solveQuadratic(a, b, c - y))
        solutions = [
            t for t in solutions if ZERO_MINUS_EPSILON <= t <= ONE_PLUS_EPSILON
        ]
        if not solutions:
            return
        # XXX

    def _closePath(self):
        if self._getCurrentPoint() != self.firstPoint:
            self.lineTo(self.firstPoint)
        self.firstPoint = None

    def _endPath(self):
        """Insideness is not defined for open contours."""
        raise NotImplementedError
<|endoftext|>"""Pen recording operations that can be accessed or replayed."""

from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen

__all__ = [
    "replayRecording",
    "RecordingPen",
    "DecomposingRecordingPen",
    "DecomposingRecordingPointPen",
    "RecordingPointPen",
    "lerpRecordings",
]


def replayRecording(recording, pen):
    """Replay a recording, as produced by RecordingPen or DecomposingRecordingPen,
    to a pen.

    Note that recording does not have to be produced by those pens.
    It can be any iterable of tuples of method name and tuple-of-arguments.
    Likewise, pen can be any objects receiving those method calls.
    """
    for operator, operands in recording:
        getattr(pen, operator)(*operands)


class RecordingPen(AbstractPen):
    """Pen recording operations that can be accessed or replayed.

    The recording can be accessed as pen.value; or replayed using
    pen.replay(otherPen).

    :Example:
        .. code-block::

            from fontTools.ttLib import TTFont
            from fontTools.pens.recordingPen import RecordingPen

            glyph_name = 'dollar'
            font_path = 'MyFont.otf'

            font = TTFont(font_path)
            glyphset = font.getGlyphSet()
            glyph = glyphset[glyph_name]

            pen = RecordingPen()
            glyph.draw(pen)
            print(pen.value)
    """

    def __init__(self):
        self.value = []

    def moveTo(self, p0):
        self.value.append(("moveTo", (p0,)))

    def lineTo(self, p1):
        self.value.append(("lineTo", (p1,)))

    def qCurveTo(self, *points):
        self.value.append(("qCurveTo", points))

    def curveTo(self, *points):
        self.value.append(("curveTo", points))

    def closePath(self):
        self.value.append(("closePath", ()))

    def endPath(self):
        self.value.append(("endPath", ()))

    def addComponent(self, glyphName, transformation):
        self.value.append(("addComponent", (glyphName, transformation)))

    def addVarComponent(self, glyphName, transformation, location):
        self.value.append(("addVarComponent", (glyphName, transformation, location)))

    def replay(self, pen):
        replayRecording(self.value, pen)

    draw = replay


class DecomposingRecordingPen(DecomposingPen, RecordingPen):
    """Same as RecordingPen, except that it doesn't keep components
    as references, but draws them decomposed as regular contours.

    The constructor takes a required 'glyphSet' positional argument,
    a dictionary of glyph objects (i.e. with a 'draw' method) keyed
    by thir name; other arguments are forwarded to the DecomposingPen's
    constructor::

        >>> class SimpleGlyph(object):
        ...     def draw(self, pen):
        ...         pen.moveTo((0, 0))
        ...         pen.curveTo((1, 1), (2, 2), (3, 3))
        ...         pen.closePath()
        >>> class CompositeGlyph(object):
        ...     def draw(self, pen):
        ...         pen.addComponent('a', (1, 0, 0, 1, -1, 1))
        >>> class MissingComponent(object):
        ...     def draw(self, pen):
        ...         pen.addComponent('foobar', (1, 0, 0, 1, 0, 0))
        >>> class FlippedComponent(object):
        ...     def draw(self, pen):
        ...         pen.addComponent('a', (-1, 0, 0, 1, 0, 0))
        >>> glyphSet = {
        ...    'a': SimpleGlyph(),
        ...    'b': CompositeGlyph(),
        ...    'c': MissingComponent(),
        ...    'd': FlippedComponent(),
        ... }
        >>> for name, glyph in sorted(glyphSet.items()):
        ...     pen = DecomposingRecordingPen(glyphSet)
        ...     try:
        ...         glyph.draw(pen)
        ...     except pen.MissingComponentError:
        ...         pass
        ...     print("{}: {}".format(name, pen.value))
        a: [('moveTo', ((0, 0),)), ('curveTo', ((1, 1), (2, 2), (3, 3))), ('closePath', ())]
        b: [('moveTo', ((-1, 1),)), ('curveTo', ((0, 2), (1, 3), (2, 4))), ('closePath', ())]
        c: []
        d: [('moveTo', ((0, 0),)), ('curveTo', ((-1, 1), (-2, 2), (-3, 3))), ('closePath', ())]

        >>> for name, glyph in sorted(glyphSet.items()):
        ...     pen = DecomposingRecordingPen(
        ...         glyphSet, skipMissingComponents=True, reverseFlipped=True,
        ...     )
        ...     glyph.draw(pen)
        ...     print("{}: {}".format(name, pen.value))
        a: [('moveTo', ((0, 0),)), ('curveTo', ((1, 1), (2, 2), (3, 3))), ('closePath', ())]
        b: [('moveTo', ((-1, 1),)), ('curveTo', ((0, 2), (1, 3), (2, 4))), ('closePath', ())]
        c: []
        d: [('moveTo', ((0, 0),)), ('lineTo', ((-3, 3),)), ('curveTo', ((-2, 2), (-1, 1), (0, 0))), ('closePath', ())]
    """

    # raises MissingComponentError(KeyError) if base glyph is not found in glyphSet
    skipMissingComponents = False


class RecordingPointPen(AbstractPointPen):
    """PointPen recording operations that can be accessed or replayed.

    The recording can be accessed as pen.value; or replayed using
    pointPen.replay(otherPointPen).

    :Example:
        .. code-block::

            from defcon import Font
            from fontTools.pens.recordingPen import RecordingPointPen

            glyph_name = 'a'
            font_path = 'MyFont.ufo'

            font = Font(font_path)
            glyph = font[glyph_name]

            pen = RecordingPointPen()
            glyph.drawPoints(pen)
            print(pen.value)

            new_glyph = font.newGlyph('b')
            pen.replay(new_glyph.getPointPen())
    """

    def __init__(self):
        self.value = []

    def beginPath(self, identifier=None, **kwargs):
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.value.append(("beginPath", (), kwargs))

    def endPath(self):
        self.value.append(("endPath", (), {}))

    def addPoint(
        self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs
    ):
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.value.append(("addPoint", (pt, segmentType, smooth, name), kwargs))

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.value.append(("addComponent", (baseGlyphName, transformation), kwargs))

    def addVarComponent(
        self, baseGlyphName, transformation, location, identifier=None, **kwargs
    ):
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.value.append(
            ("addVarComponent", (baseGlyphName, transformation, location), kwargs)
        )

    def replay(self, pointPen):
        for operator, args, kwargs in self.value:
            getattr(pointPen, operator)(*args, **kwargs)

    drawPoints = replay


class DecomposingRecordingPointPen(DecomposingPointPen, RecordingPointPen):
    """Same as RecordingPointPen, except that it doesn't keep components
    as references, but draws them decomposed as regular contours.

    The constructor takes a required 'glyphSet' positional argument,
    a dictionary of pointPen-drawable glyph objects (i.e. with a 'drawPoints' method)
    keyed by thir name; other arguments are forwarded to the DecomposingPointPen's
    constructor::

        >>> from pprint import pprint
        >>> class SimpleGlyph(object):
        ...     def drawPoints(self, pen):
        ...         pen.beginPath()
        ...         pen.addPoint((0, 0), "line")
        ...         pen.addPoint((1, 1))
        ...         pen.addPoint((2, 2))
        ...         pen.addPoint((3, 3), "curve")
        ...         pen.endPath()
        >>> class CompositeGlyph(object):
        ...     def drawPoints(self, pen):
        ...         pen.addComponent('a', (1, 0, 0, 1, -1, 1))
        >>> class MissingComponent(object):
        ...     def drawPoints(self, pen):
        ...         pen.addComponent('foobar', (1, 0, 0, 1, 0, 0))
        >>> class FlippedComponent(object):
        ...     def drawPoints(self, pen):
        ...         pen.addComponent('a', (-1, 0, 0, 1, 0, 0))
        >>> glyphSet = {
        ...    'a': SimpleGlyph(),
        ...    'b': CompositeGlyph(),
        ...    'c': MissingComponent(),
        ...    'd': FlippedComponent(),
        ... }
        >>> for name, glyph in sorted(glyphSet.items()):
        ...     pen = DecomposingRecordingPointPen(glyphSet)
        ...     try:
        ...         glyph.drawPoints(pen)
        ...     except pen.MissingComponentError:
        ...         pass
        ...     pprint({name: pen.value})
        {'a': [('beginPath', (), {}),
               ('addPoint', ((0, 0), 'line', False, None), {}),
               ('addPoint', ((1, 1), None, False, None), {}),
               ('addPoint', ((2, 2), None, False, None), {}),
               ('addPoint', ((3, 3), 'curve', False, None), {}),
               ('endPath', (), {})]}
        {'b': [('beginPath', (), {}),
               ('addPoint', ((-1, 1), 'line', False, None), {}),
               ('addPoint', ((0, 2), None, False, None), {}),
               ('addPoint', ((1, 3), None, False, None), {}),
               ('addPoint', ((2, 4), 'curve', False, None), {}),
               ('endPath', (), {})]}
        {'c': []}
        {'d': [('beginPath', (), {}),
               ('addPoint', ((0, 0), 'line', False, None), {}),
               ('addPoint', ((-1, 1), None, False, None), {}),
               ('addPoint', ((-2, 2), None, False, None), {}),
               ('addPoint', ((-3, 3), 'curve', False, None), {}),
               ('endPath', (), {})]}

        >>> for name, glyph in sorted(glyphSet.items()):
        ...     pen = DecomposingRecordingPointPen(
        ...         glyphSet, skipMissingComponents=True, reverseFlipped=True,
        ...     )
        ...     glyph.drawPoints(pen)
        ...     pprint({name: pen.value})
        {'a': [('beginPath', (), {}),
               ('addPoint', ((0, 0), 'line', False, None), {}),
               ('addPoint', ((1, 1), None, False, None), {}),
               ('addPoint', ((2, 2), None, False, None), {}),
               ('addPoint', ((3, 3), 'curve', False, None), {}),
               ('endPath', (), {})]}
        {'b': [('beginPath', (), {}),
               ('addPoint', ((-1, 1), 'line', False, None), {}),
               ('addPoint', ((0, 2), None, False, None), {}),
               ('addPoint', ((1, 3), None, False, None), {}),
               ('addPoint', ((2, 4), 'curve', False, None), {}),
               ('endPath', (), {})]}
        {'c': []}
        {'d': [('beginPath', (), {}),
               ('addPoint', ((0, 0), 'curve', False, None), {}),
               ('addPoint', ((-3, 3), 'line', False, None), {}),
               ('addPoint', ((-2, 2), None, False, None), {}),
               ('addPoint', ((-1, 1), None, False, None), {}),
               ('endPath', (), {})]}
    """

    # raises MissingComponentError(KeyError) if base glyph is not found in glyphSet
    skipMissingComponents = False


def lerpRecordings(recording1, recording2, factor=0.5):
    """Linearly interpolate between two recordings. The recordings
    must be decomposed, i.e. they must not contain any components.

    Factor is typically between 0 and 1. 0 means the first recording,
    1 means the second recording, and 0.5 means the average of the
    two recordings. Other values are possible, and can be useful to
    extrapolate. Defaults to 0.5.

    Returns a generator with the new recording.
    """
    if len(recording1) != len(recording2):
        raise ValueError(
            "Mismatched lengths: %d and %d" % (len(recording1), len(recording2))
        )
    for (op1, args1), (op2, args2) in zip(recording1, recording2):
        if op1 != op2:
            raise ValueError("Mismatched operations: %s, %s" % (op1, op2))
        if op1 == "addComponent":
            raise ValueError("Cannot interpolate components")
        else:
            mid_args = [
                (x1 + (x2 - x1) * factor, y1 + (y2 - y1) * factor)
                for (x1, y1), (x2, y2) in zip(args1, args2)
            ]
        yield (op1, mid_args)


if __name__ == "__main__":
    pen = RecordingPen()
    pen.moveTo((0, 0))
    pen.lineTo((0, 100))
    pen.curveTo((50, 75), (60, 50), (50, 25))
    pen.closePath()
    from pprint import pprint

    pprint(pen.value)
<|endoftext|># Copyright 2016 Google Inc. All Rights Reserved.
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

from fontTools.qu2cu import quadratic_to_curves
from fontTools.pens.filterPen import ContourFilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
import math


class Qu2CuPen(ContourFilterPen):
    """A filter pen to convert quadratic bezier splines to cubic curves
    using the FontTools SegmentPen protocol.

    Args:

        other_pen: another SegmentPen used to draw the transformed outline.
        max_err: maximum approximation error in font units. For optimal results,
            if you know the UPEM of the font, we recommend setting this to a
            value equal, or close to UPEM / 1000.
        reverse_direction: flip the contours' direction but keep starting point.
        stats: a dictionary counting the point numbers of cubic segments.
    """

    def __init__(
        self,
        other_pen,
        max_err,
        all_cubic=False,
        reverse_direction=False,
        stats=None,
    ):
        if reverse_direction:
            other_pen = ReverseContourPen(other_pen)
        super().__init__(other_pen)
        self.all_cubic = all_cubic
        self.max_err = max_err
        self.stats = stats

    def _quadratics_to_curve(self, q):
        curves = quadratic_to_curves(q, self.max_err, all_cubic=self.all_cubic)
        if self.stats is not None:
            for curve in curves:
                n = str(len(curve) - 2)
                self.stats[n] = self.stats.get(n, 0) + 1
        for curve in curves:
            if len(curve) == 4:
                yield ("curveTo", curve[1:])
            else:
                yield ("qCurveTo", curve[1:])

    def filterContour(self, contour):
        quadratics = []
        currentPt = None
        newContour = []
        for op, args in contour:
            if op == "qCurveTo" and (
                self.all_cubic or (len(args) > 2 and args[-1] is not None)
            ):
                if args[-1] is None:
                    raise NotImplementedError(
                        "oncurve-less contours with all_cubic not implemented"
                    )
                quadratics.append((currentPt,) + args)
            else:
                if quadratics:
                    newContour.extend(self._quadratics_to_curve(quadratics))
                    quadratics = []
                newContour.append((op, args))
            currentPt = args[-1] if args else None
        if quadratics:
            newContour.extend(self._quadratics_to_curve(quadratics))

        if not self.all_cubic:
            # Add back implicit oncurve points
            contour = newContour
            newContour = []
            for op, args in contour:
                if op == "qCurveTo" and newContour and newContour[-1][0] == "qCurveTo":
                    pt0 = newContour[-1][1][-2]
                    pt1 = newContour[-1][1][-1]
                    pt2 = args[0]
                    if (
                        pt1 is not None
                        and math.isclose(pt2[0] - pt1[0], pt1[0] - pt0[0])
                        and math.isclose(pt2[1] - pt1[1], pt1[1] - pt0[1])
                    ):
                        newArgs = newContour[-1][1][:-1] + args
                        newContour[-1] = (op, newArgs)
                        continue

                newContour.append((op, args))

        return newContour
<|endoftext|>from typing import Callable
from fontTools.pens.basePen import BasePen


def pointToString(pt, ntos=str):
    return " ".join(ntos(i) for i in pt)


class SVGPathPen(BasePen):
    """Pen to draw SVG path d commands.

    Args:
        glyphSet: a dictionary of drawable glyph objects keyed by name
            used to resolve component references in composite glyphs.
        ntos: a callable that takes a number and returns a string, to
            customize how numbers are formatted (default: str).

    :Example:
        .. code-block::

            >>> pen = SVGPathPen(None)
            >>> pen.moveTo((0, 0))
            >>> pen.lineTo((1, 1))
            >>> pen.curveTo((2, 2), (3, 3), (4, 4))
            >>> pen.closePath()
            >>> pen.getCommands()
            'M0 0 1 1C2 2 3 3 4 4Z'

    Note:
        Fonts have a coordinate system where Y grows up, whereas in SVG,
        Y grows down.  As such, rendering path data from this pen in
        SVG typically results in upside-down glyphs.  You can fix this
        by wrapping the data from this pen in an SVG group element with
        transform, or wrap this pen in a transform pen.  For example:
        .. code-block:: python

            spen = svgPathPen.SVGPathPen(glyphset)
            pen= TransformPen(spen , (1, 0, 0, -1, 0, 0))
            glyphset[glyphname].draw(pen)
            print(tpen.getCommands())
    """

    def __init__(self, glyphSet, ntos: Callable[[float], str] = str):
        BasePen.__init__(self, glyphSet)
        self._commands = []
        self._lastCommand = None
        self._lastX = None
        self._lastY = None
        self._ntos = ntos

    def _handleAnchor(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.moveTo((10, 10))
        >>> pen._commands
        ['M10 10']
        """
        if self._lastCommand == "M":
            self._commands.pop(-1)

    def _moveTo(self, pt):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen._commands
        ['M0 0']

        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 0))
        >>> pen._commands
        ['M10 0']

        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 10))
        >>> pen._commands
        ['M0 10']
        """
        self._handleAnchor()
        t = "M%s" % (pointToString(pt, self._ntos))
        self._commands.append(t)
        self._lastCommand = "M"
        self._lastX, self._lastY = pt

    def _lineTo(self, pt):
        """
        # duplicate point
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((10, 10))
        >>> pen._commands
        ['M10 10']

        # vertical line
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((10, 0))
        >>> pen._commands
        ['M10 10', 'V0']

        # horizontal line
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((0, 10))
        >>> pen._commands
        ['M10 10', 'H0']

        # basic
        >>> pen = SVGPathPen(None)
        >>> pen.lineTo((70, 80))
        >>> pen._commands
        ['L70 80']

        # basic following a moveto
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.lineTo((10, 10))
        >>> pen._commands
        ['M0 0', ' 10 10']
        """
        x, y = pt
        # duplicate point
        if x == self._lastX and y == self._lastY:
            return
        # vertical line
        elif x == self._lastX:
            cmd = "V"
            pts = self._ntos(y)
        # horizontal line
        elif y == self._lastY:
            cmd = "H"
            pts = self._ntos(x)
        # previous was a moveto
        elif self._lastCommand == "M":
            cmd = None
            pts = " " + pointToString(pt, self._ntos)
        # basic
        else:
            cmd = "L"
            pts = pointToString(pt, self._ntos)
        # write the string
        t = ""
        if cmd:
            t += cmd
            self._lastCommand = cmd
        t += pts
        self._commands.append(t)
        # store for future reference
        self._lastX, self._lastY = pt

    def _curveToOne(self, pt1, pt2, pt3):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.curveTo((10, 20), (30, 40), (50, 60))
        >>> pen._commands
        ['C10 20 30 40 50 60']
        """
        t = "C"
        t += pointToString(pt1, self._ntos) + " "
        t += pointToString(pt2, self._ntos) + " "
        t += pointToString(pt3, self._ntos)
        self._commands.append(t)
        self._lastCommand = "C"
        self._lastX, self._lastY = pt3

    def _qCurveToOne(self, pt1, pt2):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.qCurveTo((10, 20), (30, 40))
        >>> pen._commands
        ['Q10 20 30 40']
        >>> from fontTools.misc.roundTools import otRound
        >>> pen = SVGPathPen(None, ntos=lambda v: str(otRound(v)))
        >>> pen.qCurveTo((3, 3), (7, 5), (11, 4))
        >>> pen._commands
        ['Q3 3 5 4', 'Q7 5 11 4']
        """
        assert pt2 is not None
        t = "Q"
        t += pointToString(pt1, self._ntos) + " "
        t += pointToString(pt2, self._ntos)
        self._commands.append(t)
        self._lastCommand = "Q"
        self._lastX, self._lastY = pt2

    def _closePath(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.closePath()
        >>> pen._commands
        ['Z']
        """
        self._commands.append("Z")
        self._lastCommand = "Z"
        self._lastX = self._lastY = None

    def _endPath(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.endPath()
        >>> pen._commands
        []
        """
        self._lastCommand = None
        self._lastX = self._lastY = None

    def getCommands(self):
        return "".join(self._commands)


def main(args=None):
    """Generate per-character SVG from font and text"""

    if args is None:
        import sys

        args = sys.argv[1:]

    from fontTools.ttLib import TTFont
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools pens.svgPathPen", description="Generate SVG from text"
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("text", metavar="text", nargs="?", help="Text string.")
    parser.add_argument(
        "-y",
        metavar="<number>",
        help="Face index into a collection to open. Zero based.",
    )
    parser.add_argument(
        "--glyphs",
        metavar="whitespace-separated list of glyph names",
        type=str,
        help="Glyphs to show. Exclusive with text option",
    )
    parser.add_argument(
        "--variations",
        metavar="AXIS=LOC",
        default="",
        help="List of space separated locations. A location consist in "
        "the name of a variation axis, followed by '=' and a number. E.g.: "
        "wght=700 wdth=80. The default is the location of the base master.",
    )

    options = parser.parse_args(args)

    fontNumber = int(options.y) if options.y is not None else 0

    font = TTFont(options.font, fontNumber=fontNumber)
    text = options.text
    glyphs = options.glyphs

    location = {}
    for tag_v in options.variations.split():
        fields = tag_v.split("=")
        tag = fields[0].strip()
        v = float(fields[1])
        location[tag] = v

    hhea = font["hhea"]
    ascent, descent = hhea.ascent, hhea.descent

    glyphset = font.getGlyphSet(location=location)
    cmap = font["cmap"].getBestCmap()

    if glyphs is not None and text is not None:
        raise ValueError("Options --glyphs and --text are exclusive")

    if glyphs is None:
        glyphs = " ".join(cmap[ord(u)] for u in text)

    glyphs = glyphs.split()

    s = ""
    width = 0
    for g in glyphs:
        glyph = glyphset[g]

        pen = SVGPathPen(glyphset)
        glyph.draw(pen)
        commands = pen.getCommands()

        s += '<g transform="translate(%d %d) scale(1 -1)"><path d="%s"/></g>\n' % (
            width,
            ascent,
            commands,
        )

        width += glyph.width

    print('<?xml version="1.0" encoding="UTF-8"?>')
    print(
        '<svg width="%d" height="%d" xmlns="http://www.w3.org/2000/svg">'
        % (width, ascent - descent)
    )
    print(s, end="")
    print("</svg>")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        import doctest

        sys.exit(doctest.testmod().failed)

    sys.exit(main())
<|endoftext|>"""Pen multiplexing drawing to one or more pens."""

from fontTools.pens.basePen import AbstractPen

__all__ = ["TeePen"]


class TeePen(AbstractPen):
    """Pen multiplexing drawing to one or more pens.

    Use either as TeePen(pen1, pen2, ...) or TeePen(iterableOfPens)."""

    def __init__(self, *pens):
        if len(pens) == 1:
            pens = pens[0]
        self.pens = pens

    def moveTo(self, p0):
        for pen in self.pens:
            pen.moveTo(p0)

    def lineTo(self, p1):
        for pen in self.pens:
            pen.lineTo(p1)

    def qCurveTo(self, *points):
        for pen in self.pens:
            pen.qCurveTo(*points)

    def curveTo(self, *points):
        for pen in self.pens:
            pen.curveTo(*points)

    def closePath(self):
        for pen in self.pens:
            pen.closePath()

    def endPath(self):
        for pen in self.pens:
            pen.endPath()

    def addComponent(self, glyphName, transformation):
        for pen in self.pens:
            pen.addComponent(glyphName, transformation)


if __name__ == "__main__":
    from fontTools.pens.basePen import _TestPen

    pen = TeePen(_TestPen(), _TestPen())
    pen.moveTo((0, 0))
    pen.lineTo((0, 100))
    pen.curveTo((50, 75), (60, 50), (50, 25))
    pen.closePath()
<|endoftext|>from fontTools.misc.arrayTools import pairwise
from fontTools.pens.filterPen import ContourFilterPen

__all__ = ["reversedContour", "ReverseContourPen"]


class ReverseContourPen(ContourFilterPen):
    """Filter pen that passes outline data to another pen, but reversing
    the winding direction of all contours. Components are simply passed
    through unchanged.

    Closed contours are reversed in such a way that the first point remains
    the first point.
    """

    def __init__(self, outPen, outputImpliedClosingLine=False):
        super().__init__(outPen)
        self.outputImpliedClosingLine = outputImpliedClosingLine

    def filterContour(self, contour):
        return reversedContour(contour, self.outputImpliedClosingLine)


def reversedContour(contour, outputImpliedClosingLine=False):
    """Generator that takes a list of pen's (operator, operands) tuples,
    and yields them with the winding direction reversed.
    """
    if not contour:
        return  # nothing to do, stop iteration

    # valid contours must have at least a starting and ending command,
    # can't have one without the other
    assert len(contour) > 1, "invalid contour"

    # the type of the last command determines if the contour is closed
    contourType = contour.pop()[0]
    assert contourType in ("endPath", "closePath")
    closed = contourType == "closePath"

    firstType, firstPts = contour.pop(0)
    assert firstType in ("moveTo", "qCurveTo"), (
        "invalid initial segment type: %r" % firstType
    )
    firstOnCurve = firstPts[-1]
    if firstType == "qCurveTo":
        # special case for TrueType paths contaning only off-curve points
        assert firstOnCurve is None, "off-curve only paths must end with 'None'"
        assert not contour, "only one qCurveTo allowed per off-curve path"
        firstPts = (firstPts[0],) + tuple(reversed(firstPts[1:-1])) + (None,)

    if not contour:
        # contour contains only one segment, nothing to reverse
        if firstType == "moveTo":
            closed = False  # single-point paths can't be closed
        else:
            closed = True  # off-curve paths are closed by definition
        yield firstType, firstPts
    else:
        lastType, lastPts = contour[-1]
        lastOnCurve = lastPts[-1]
        if closed:
            # for closed paths, we keep the starting point
            yield firstType, firstPts
            if firstOnCurve != lastOnCurve:
                # emit an implied line between the last and first points
                yield "lineTo", (lastOnCurve,)
                contour[-1] = (lastType, tuple(lastPts[:-1]) + (firstOnCurve,))

            if len(contour) > 1:
                secondType, secondPts = contour[0]
            else:
                # contour has only two points, the second and last are the same
                secondType, secondPts = lastType, lastPts

            if not outputImpliedClosingLine:
                # if a lineTo follows the initial moveTo, after reversing it
                # will be implied by the closePath, so we don't emit one;
                # unless the lineTo and moveTo overlap, in which case we keep the
                # duplicate points
                if secondType == "lineTo" and firstPts != secondPts:
                    del contour[0]
                    if contour:
                        contour[-1] = (lastType, tuple(lastPts[:-1]) + secondPts)
        else:
            # for open paths, the last point will become the first
            yield firstType, (lastOnCurve,)
            contour[-1] = (lastType, tuple(lastPts[:-1]) + (firstOnCurve,))

        # we iterate over all segment pairs in reverse order, and yield
        # each one with the off-curve points reversed (if any), and
        # with the on-curve point of the following segment
        for (curType, curPts), (_, nextPts) in pairwise(contour, reverse=True):
            yield curType, tuple(reversed(curPts[:-1])) + (nextPts[-1],)

    yield "closePath" if closed else "endPath", ()
<|endoftext|>from fontTools.pens.basePen import BasePen

__all__ = ["QtPen"]


class QtPen(BasePen):
    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            from PyQt5.QtGui import QPainterPath

            path = QPainterPath()
        self.path = path

    def _moveTo(self, p):
        self.path.moveTo(*p)

    def _lineTo(self, p):
        self.path.lineTo(*p)

    def _curveToOne(self, p1, p2, p3):
        self.path.cubicTo(*p1, *p2, *p3)

    def _qCurveToOne(self, p1, p2):
        self.path.quadTo(*p1, *p2)

    def _closePath(self):
        self.path.closeSubpath()
<|endoftext|>"""Pen to draw to a Cairo graphics library context."""

from fontTools.pens.basePen import BasePen

__all__ = ["CairoPen"]


class CairoPen(BasePen):
    """Pen to draw to a Cairo graphics library context."""

    def __init__(self, glyphSet, context):
        BasePen.__init__(self, glyphSet)
        self.context = context

    def _moveTo(self, p):
        self.context.move_to(*p)

    def _lineTo(self, p):
        self.context.line_to(*p)

    def _curveToOne(self, p1, p2, p3):
        self.context.curve_to(*p1, *p2, *p3)

    def _closePath(self):
        self.context.close_path()
<|endoftext|>from fontTools.pens.basePen import BasePen
from reportlab.graphics.shapes import Path

__all__ = ["ReportLabPen"]


class ReportLabPen(BasePen):
    """A pen for drawing onto a ``reportlab.graphics.shapes.Path`` object."""

    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            path = Path()
        self.path = path

    def _moveTo(self, p):
        x, y = p
        self.path.moveTo(x, y)

    def _lineTo(self, p):
        x, y = p
        self.path.lineTo(x, y)

    def _curveToOne(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        self.path.curveTo(x1, y1, x2, y2, x3, y3)

    def _closePath(self):
        self.path.closePath()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: reportLabPen.py <OTF/TTF font> <glyphname> [<image file to create>]"
        )
        print(
            "  If no image file name is created, by default <glyphname>.png is created."
        )
        print("  example: reportLabPen.py Arial.TTF R test.png")
        print(
            "  (The file format will be PNG, regardless of the image file name supplied)"
        )
        sys.exit(0)

    from fontTools.ttLib import TTFont
    from reportlab.lib import colors

    path = sys.argv[1]
    glyphName = sys.argv[2]
    if len(sys.argv) > 3:
        imageFile = sys.argv[3]
    else:
        imageFile = "%s.png" % glyphName

    font = TTFont(path)  # it would work just as well with fontTools.t1Lib.T1Font
    gs = font.getGlyphSet()
    pen = ReportLabPen(gs, Path(fillColor=colors.red, strokeWidth=5))
    g = gs[glyphName]
    g.draw(pen)

    w, h = g.width, 1000
    from reportlab.graphics import renderPM
    from reportlab.graphics.shapes import Group, Drawing, scale

    # Everything is wrapped in a group to allow transformations.
    g = Group(pen.path)
    g.translate(0, 200)
    g.scale(0.3, 0.3)

    d = Drawing(w, h)
    d.add(g)

    renderPM.drawToFile(d, imageFile, fmt="PNG")
<|endoftext|>from fontTools.pens.filterPen import ContourFilterPen


class ExplicitClosingLinePen(ContourFilterPen):
    """A filter pen that adds an explicit lineTo to the first point of each closed
    contour if the end point of the last segment is not already the same as the first point.
    Otherwise, it passes the contour through unchanged.

    >>> from pprint import pprint
    >>> from fontTools.pens.recordingPen import RecordingPen
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.lineTo((100, 0))
    >>> pen.lineTo((100, 100))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('lineTo', ((100, 0),)),
     ('lineTo', ((100, 100),)),
     ('lineTo', ((0, 0),)),
     ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.lineTo((100, 0))
    >>> pen.lineTo((100, 100))
    >>> pen.lineTo((0, 0))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('lineTo', ((100, 0),)),
     ('lineTo', ((100, 100),)),
     ('lineTo', ((0, 0),)),
     ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.curveTo((100, 0), (0, 100), (100, 100))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('curveTo', ((100, 0), (0, 100), (100, 100))),
     ('lineTo', ((0, 0),)),
     ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.curveTo((100, 0), (0, 100), (100, 100))
    >>> pen.lineTo((0, 0))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('curveTo', ((100, 0), (0, 100), (100, 100))),
     ('lineTo', ((0, 0),)),
     ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.curveTo((100, 0), (0, 100), (0, 0))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('curveTo', ((100, 0), (0, 100), (0, 0))),
     ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)), ('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.closePath()
    >>> pprint(rec.value)
    [('closePath', ())]
    >>> rec = RecordingPen()
    >>> pen = ExplicitClosingLinePen(rec)
    >>> pen.moveTo((0, 0))
    >>> pen.lineTo((100, 0))
    >>> pen.lineTo((100, 100))
    >>> pen.endPath()
    >>> pprint(rec.value)
    [('moveTo', ((0, 0),)),
     ('lineTo', ((100, 0),)),
     ('lineTo', ((100, 100),)),
     ('endPath', ())]
    """

    def filterContour(self, contour):
        if (
            not contour
            or contour[0][0] != "moveTo"
            or contour[-1][0] != "closePath"
            or len(contour) < 3
        ):
            return
        movePt = contour[0][1][0]
        lastSeg = contour[-2][1]
        if lastSeg and movePt != lastSeg[-1]:
            contour[-1:] = [("lineTo", (movePt,)), ("closePath", ())]
<|endoftext|>from fontTools.pens.basePen import BasePen

from Quartz.CoreGraphics import CGPathCreateMutable, CGPathMoveToPoint
from Quartz.CoreGraphics import CGPathAddLineToPoint, CGPathAddCurveToPoint
from Quartz.CoreGraphics import CGPathAddQuadCurveToPoint, CGPathCloseSubpath

__all__ = ["QuartzPen"]


class QuartzPen(BasePen):
    """A pen that creates a CGPath

    Parameters
    - path: an optional CGPath to add to
    - xform: an optional CGAffineTransform to apply to the path
    """

    def __init__(self, glyphSet, path=None, xform=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            path = CGPathCreateMutable()
        self.path = path
        self.xform = xform

    def _moveTo(self, pt):
        x, y = pt
        CGPathMoveToPoint(self.path, self.xform, x, y)

    def _lineTo(self, pt):
        x, y = pt
        CGPathAddLineToPoint(self.path, self.xform, x, y)

    def _curveToOne(self, p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        CGPathAddCurveToPoint(self.path, self.xform, x1, y1, x2, y2, x3, y3)

    def _qCurveToOne(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        CGPathAddQuadCurveToPoint(self.path, self.xform, x1, y1, x2, y2)

    def _closePath(self):
        CGPathCloseSubpath(self.path)
<|endoftext|>"""fontTools.pens.basePen.py -- Tools and base classes to build pen objects.

The Pen Protocol

A Pen is a kind of object that standardizes the way how to "draw" outlines:
it is a middle man between an outline and a drawing. In other words:
it is an abstraction for drawing outlines, making sure that outline objects
don't need to know the details about how and where they're being drawn, and
that drawings don't need to know the details of how outlines are stored.

The most basic pattern is this::

        outline.draw(pen)  # 'outline' draws itself onto 'pen'

Pens can be used to render outlines to the screen, but also to construct
new outlines. Eg. an outline object can be both a drawable object (it has a
draw() method) as well as a pen itself: you *build* an outline using pen
methods.

The AbstractPen class defines the Pen protocol. It implements almost
nothing (only no-op closePath() and endPath() methods), but is useful
for documentation purposes. Subclassing it basically tells the reader:
"this class implements the Pen protocol.". An examples of an AbstractPen
subclass is :py:class:`fontTools.pens.transformPen.TransformPen`.

The BasePen class is a base implementation useful for pens that actually
draw (for example a pen renders outlines using a native graphics engine).
BasePen contains a lot of base functionality, making it very easy to build
a pen that fully conforms to the pen protocol. Note that if you subclass
BasePen, you *don't* override moveTo(), lineTo(), etc., but _moveTo(),
_lineTo(), etc. See the BasePen doc string for details. Examples of
BasePen subclasses are fontTools.pens.boundsPen.BoundsPen and
fontTools.pens.cocoaPen.CocoaPen.

Coordinates are usually expressed as (x, y) tuples, but generally any
sequence of length 2 will do.
"""

from typing import Tuple, Dict

from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity

__all__ = [
    "AbstractPen",
    "NullPen",
    "BasePen",
    "PenError",
    "decomposeSuperBezierSegment",
    "decomposeQuadraticSegment",
]


class PenError(Exception):
    """Represents an error during penning."""


class OpenContourError(PenError):
    pass


class AbstractPen:
    def moveTo(self, pt: Tuple[float, float]) -> None:
        """Begin a new sub path, set the current point to 'pt'. You must
        end each sub path with a call to pen.closePath() or pen.endPath().
        """
        raise NotImplementedError

    def lineTo(self, pt: Tuple[float, float]) -> None:
        """Draw a straight line from the current point to 'pt'."""
        raise NotImplementedError

    def curveTo(self, *points: Tuple[float, float]) -> None:
        """Draw a cubic bezier with an arbitrary number of control points.

