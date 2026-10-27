        if len(color) == 3:
            color = color + (1,)
        cr.set_source_rgba(*color)
        cr.translate(self.start_arrow_length + x, y)
        cr.move_to(0, 0)
        cr.line_to(
            -self.start_arrow_length,
            -self.start_arrow_length * 0.4,
        )
        cr.line_to(
            -self.start_arrow_length,
            self.start_arrow_length * 0.4,
        )
        cr.close_path()
        cr.fill()
        cr.restore()

    def draw_text(self, text, *, x=0, y=0, color=(0, 0, 0), width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        text = text.splitlines()
        cr = cairo.Context(self.surface)
        cr.set_source_rgb(*color)
        cr.set_font_size(self.font_size)
        cr.select_font_face(
            "@cairo:monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )
        text_width = 0
        text_height = 0
        font_extents = cr.font_extents()
        font_font_size = font_extents[2]
        font_ascent = font_extents[0]
        for line in text:
            extents = cr.text_extents(line)
            text_width = max(text_width, extents.x_advance)
            text_height += font_font_size
        if not text_width:
            return
        cr.translate(x, y)
        scale = min(width / text_width, height / text_height)
        # center
        cr.translate(
            (width - text_width * scale) / 2, (height - text_height * scale) / 2
        )
        cr.scale(scale, scale)

        cr.translate(0, font_ascent)
        for line in text:
            cr.move_to(0, 0)
            cr.show_text(line)
            cr.translate(0, font_font_size)

    def draw_cupcake(self):
        self.draw_label(
            self.no_issues_label,
            x=self.pad,
            y=self.pad,
            color=self.no_issues_label_color,
            width=self.width - 2 * self.pad,
            align=0.5,
            bold=True,
            font_size=self.title_font_size,
        )

        self.draw_text(
            self.cupcake,
            x=self.pad,
            y=self.pad + self.font_size,
            width=self.width - 2 * self.pad,
            height=self.height - 2 * self.pad - self.font_size,
            color=self.cupcake_color,
        )

    def draw_emoticon(self, emoticon, x=0, y=0):
        self.draw_text(
            emoticon,
            x=x,
            y=y,
            color=self.emoticon_color,
            width=self.panel_width,
            height=self.panel_height,
        )


class InterpolatablePostscriptLike(InterpolatablePlot):
    def __exit__(self, type, value, traceback):
        self.surface.finish()

    def show_page(self):
        super().show_page()
        self.surface.show_page()


class InterpolatablePS(InterpolatablePostscriptLike):
    def __enter__(self):
        self.surface = cairo.PSSurface(self.out, self.width, self.height)
        return self


class InterpolatablePDF(InterpolatablePostscriptLike):
    def __enter__(self):
        self.surface = cairo.PDFSurface(self.out, self.width, self.height)
        self.surface.set_metadata(
            cairo.PDF_METADATA_CREATOR, "fonttools varLib.interpolatable"
        )
        self.surface.set_metadata(cairo.PDF_METADATA_CREATE_DATE, "")
        return self


class InterpolatableSVG(InterpolatablePlot):
    def __enter__(self):
        self.sink = BytesIO()
        self.surface = cairo.SVGSurface(self.sink, self.width, self.height)
        return self

    def __exit__(self, type, value, traceback):
        if self.surface is not None:
            self.show_page()

    def show_page(self):
        super().show_page()
        self.surface.finish()
        self.out.append(self.sink.getvalue())
        self.sink = BytesIO()
        self.surface = cairo.SVGSurface(self.sink, self.width, self.height)
<|endoftext|>"""
Merge OpenType Layout tables (GDEF / GPOS / GSUB).
"""

import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
    _compression_level_from_env,
    compact_pair_pos,
)

log = logging.getLogger("fontTools.varLib.merger")

from .errors import (
    ShouldBeConstant,
    FoundANone,
    MismatchedTypes,
    NotANone,
    LengthsDiffer,
    KeysDiffer,
    InconsistentGlyphOrder,
    InconsistentExtensions,
    InconsistentFormats,
    UnsupportedFormat,
    VarLibMergeError,
)


class Merger(object):
    def __init__(self, font=None):
        self.font = font
        # mergeTables populates this from the parent's master ttfs
        self.ttfs = None

    @classmethod
    def merger(celf, clazzes, attrs=(None,)):
        assert celf != Merger, "Subclass Merger instead."
        if "mergers" not in celf.__dict__:
            celf.mergers = {}
        if type(clazzes) in (type, enum.EnumMeta):
            clazzes = (clazzes,)
        if type(attrs) == str:
            attrs = (attrs,)

        def wrapper(method):
            assert method.__name__ == "merge"
            done = []
            for clazz in clazzes:
                if clazz in done:
                    continue  # Support multiple names of a clazz
                done.append(clazz)
                mergers = celf.mergers.setdefault(clazz, {})
                for attr in attrs:
                    assert attr not in mergers, (
                        "Oops, class '%s' has merge function for '%s' defined already."
                        % (clazz.__name__, attr)
                    )
                    mergers[attr] = method
            return None

        return wrapper

    @classmethod
    def mergersFor(celf, thing, _default={}):
        typ = type(thing)

        for celf in celf.mro():
            mergers = getattr(celf, "mergers", None)
            if mergers is None:
                break

            m = celf.mergers.get(typ, None)
            if m is not None:
                return m

        return _default

    def mergeObjects(self, out, lst, exclude=()):
        if hasattr(out, "ensureDecompiled"):
            out.ensureDecompiled(recurse=False)
        for item in lst:
            if hasattr(item, "ensureDecompiled"):
                item.ensureDecompiled(recurse=False)
        keys = sorted(vars(out).keys())
        if not all(keys == sorted(vars(v).keys()) for v in lst):
            raise KeysDiffer(
                self, expected=keys, got=[sorted(vars(v).keys()) for v in lst]
            )
        mergers = self.mergersFor(out)
        defaultMerger = mergers.get("*", self.__class__.mergeThings)
        try:
            for key in keys:
                if key in exclude:
                    continue
                value = getattr(out, key)
                values = [getattr(table, key) for table in lst]
                mergerFunc = mergers.get(key, defaultMerger)
                mergerFunc(self, value, values)
        except VarLibMergeError as e:
            e.stack.append("." + key)
            raise

    def mergeLists(self, out, lst):
        if not allEqualTo(out, lst, len):
            raise LengthsDiffer(self, expected=len(out), got=[len(x) for x in lst])
        for i, (value, values) in enumerate(zip(out, zip(*lst))):
            try:
                self.mergeThings(value, values)
            except VarLibMergeError as e:
                e.stack.append("[%d]" % i)
                raise

    def mergeThings(self, out, lst):
        if not allEqualTo(out, lst, type):
            raise MismatchedTypes(
                self, expected=type(out).__name__, got=[type(x).__name__ for x in lst]
            )
        mergerFunc = self.mergersFor(out).get(None, None)
        if mergerFunc is not None:
            mergerFunc(self, out, lst)
        elif isinstance(out, enum.Enum):
            # need to special-case Enums as have __dict__ but are not regular 'objects',
            # otherwise mergeObjects/mergeThings get trapped in a RecursionError
            if not allEqualTo(out, lst):
                raise ShouldBeConstant(self, expected=out, got=lst)
        elif hasattr(out, "__dict__"):
            self.mergeObjects(out, lst)
        elif isinstance(out, list):
            self.mergeLists(out, lst)
        else:
            if not allEqualTo(out, lst):
                raise ShouldBeConstant(self, expected=out, got=lst)

    def mergeTables(self, font, master_ttfs, tableTags):
        for tag in tableTags:
            if tag not in font:
                continue
            try:
                self.ttfs = master_ttfs
                self.mergeThings(font[tag], [m.get(tag) for m in master_ttfs])
            except VarLibMergeError as e:
                e.stack.append(tag)
                raise


#
# Aligning merger
#
class AligningMerger(Merger):
    pass


@AligningMerger.merger(ot.GDEF, "GlyphClassDef")
def merge(merger, self, lst):
    if self is None:
        if not allNone(lst):
            raise NotANone(merger, expected=None, got=lst)
        return

    lst = [l.classDefs for l in lst]
    self.classDefs = {}
    # We only care about the .classDefs
    self = self.classDefs

    allKeys = set()
    allKeys.update(*[l.keys() for l in lst])
    for k in allKeys:
        allValues = nonNone(l.get(k) for l in lst)
        if not allEqual(allValues):
            raise ShouldBeConstant(
                merger, expected=allValues[0], got=lst, stack=["." + k]
            )
        if not allValues:
            self[k] = None
        else:
            self[k] = allValues[0]


def _SinglePosUpgradeToFormat2(self):
    if self.Format == 2:
        return self

    ret = ot.SinglePos()
    ret.Format = 2
    ret.Coverage = self.Coverage
    ret.ValueFormat = self.ValueFormat
    ret.Value = [self.Value for _ in ret.Coverage.glyphs]
    ret.ValueCount = len(ret.Value)

    return ret


def _merge_GlyphOrders(font, lst, values_lst=None, default=None):
    """Takes font and list of glyph lists (must be sorted by glyph id), and returns
    two things:
    - Combined glyph list,
    - If values_lst is None, return input glyph lists, but padded with None when a glyph
      was missing in a list.  Otherwise, return values_lst list-of-list, padded with None
      to match combined glyph lists.
    """
    if values_lst is None:
        dict_sets = [set(l) for l in lst]
    else:
        dict_sets = [{g: v for g, v in zip(l, vs)} for l, vs in zip(lst, values_lst)]
    combined = set()
    combined.update(*dict_sets)

    sortKey = font.getReverseGlyphMap().__getitem__
    order = sorted(combined, key=sortKey)
    # Make sure all input glyphsets were in proper order
    if not all(sorted(vs, key=sortKey) == vs for vs in lst):
        raise InconsistentGlyphOrder()
    del combined

    paddedValues = None
    if values_lst is None:
        padded = [
            [glyph if glyph in dict_set else default for glyph in order]
            for dict_set in dict_sets
        ]
    else:
        assert len(lst) == len(values_lst)
        padded = [
            [dict_set[glyph] if glyph in dict_set else default for glyph in order]
            for dict_set in dict_sets
        ]
    return order, padded


@AligningMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
    # Code below sometimes calls us with self being
    # a new object. Copy it from lst and recurse.
    self.__dict__ = lst[0].__dict__.copy()
    merger.mergeObjects(self, lst)


@AligningMerger.merger(ot.Anchor)
def merge(merger, self, lst):
    # Code below sometimes calls us with self being
    # a new object. Copy it from lst and recurse.
    self.__dict__ = lst[0].__dict__.copy()
    merger.mergeObjects(self, lst)


def _Lookup_SinglePos_get_effective_value(merger, subtables, glyph):
    for self in subtables:
        if (
            self is None
            or type(self) != ot.SinglePos
            or self.Coverage is None
            or glyph not in self.Coverage.glyphs
        ):
            continue
        if self.Format == 1:
            return self.Value
        elif self.Format == 2:
            return self.Value[self.Coverage.glyphs.index(glyph)]
        else:
            raise UnsupportedFormat(merger, subtable="single positioning lookup")
    return None


def _Lookup_PairPos_get_effective_value_pair(
    merger, subtables, firstGlyph, secondGlyph
):
    for self in subtables:
        if (
            self is None
            or type(self) != ot.PairPos
            or self.Coverage is None
            or firstGlyph not in self.Coverage.glyphs
        ):
            continue
        if self.Format == 1:
            ps = self.PairSet[self.Coverage.glyphs.index(firstGlyph)]
            pvr = ps.PairValueRecord
            for rec in pvr:  # TODO Speed up
                if rec.SecondGlyph == secondGlyph:
                    return rec
            continue
        elif self.Format == 2:
            klass1 = self.ClassDef1.classDefs.get(firstGlyph, 0)
            klass2 = self.ClassDef2.classDefs.get(secondGlyph, 0)
            return self.Class1Record[klass1].Class2Record[klass2]
        else:
            raise UnsupportedFormat(merger, subtable="pair positioning lookup")
    return None


@AligningMerger.merger(ot.SinglePos)
def merge(merger, self, lst):
    self.ValueFormat = valueFormat = reduce(int.__or__, [l.ValueFormat for l in lst], 0)
    if not (len(lst) == 1 or (valueFormat & ~0xF == 0)):
        raise UnsupportedFormat(merger, subtable="single positioning lookup")

    # If all have same coverage table and all are format 1,
    coverageGlyphs = self.Coverage.glyphs
    if all(v.Format == 1 for v in lst) and all(
        coverageGlyphs == v.Coverage.glyphs for v in lst
    ):
        self.Value = otBase.ValueRecord(valueFormat, self.Value)
        if valueFormat != 0:
            # If v.Value is None, it means a kerning of 0; we want
            # it to participate in the model still.
            # https://github.com/fonttools/fonttools/issues/3111
            merger.mergeThings(
                self.Value,
                [v.Value if v.Value is not None else otBase.ValueRecord() for v in lst],
            )
        self.ValueFormat = self.Value.getFormat()
        return

    # Upgrade everything to Format=2
    self.Format = 2
    lst = [_SinglePosUpgradeToFormat2(v) for v in lst]

    # Align them
    glyphs, padded = _merge_GlyphOrders(
        merger.font, [v.Coverage.glyphs for v in lst], [v.Value for v in lst]
    )

    self.Coverage.glyphs = glyphs
    self.Value = [otBase.ValueRecord(valueFormat) for _ in glyphs]
    self.ValueCount = len(self.Value)

    for i, values in enumerate(padded):
        for j, glyph in enumerate(glyphs):
            if values[j] is not None:
                continue
            # Fill in value from other subtables
            # Note!!! This *might* result in behavior change if ValueFormat2-zeroedness
            # is different between used subtable and current subtable!
            # TODO(behdad) Check and warn if that happens?
            v = _Lookup_SinglePos_get_effective_value(
                merger, merger.lookup_subtables[i], glyph
            )
            if v is None:
                v = otBase.ValueRecord(valueFormat)
            values[j] = v

    merger.mergeLists(self.Value, padded)

    # Merge everything else; though, there shouldn't be anything else. :)
    merger.mergeObjects(
        self, lst, exclude=("Format", "Coverage", "Value", "ValueCount", "ValueFormat")
    )
    self.ValueFormat = reduce(
        int.__or__, [v.getEffectiveFormat() for v in self.Value], 0
    )


@AligningMerger.merger(ot.PairSet)
def merge(merger, self, lst):
    # Align them
    glyphs, padded = _merge_GlyphOrders(
        merger.font,
        [[v.SecondGlyph for v in vs.PairValueRecord] for vs in lst],
        [vs.PairValueRecord for vs in lst],
    )

    self.PairValueRecord = pvrs = []
    for glyph in glyphs:
        pvr = ot.PairValueRecord()
        pvr.SecondGlyph = glyph
        pvr.Value1 = (
            otBase.ValueRecord(merger.valueFormat1) if merger.valueFormat1 else None
        )
        pvr.Value2 = (
            otBase.ValueRecord(merger.valueFormat2) if merger.valueFormat2 else None
        )
        pvrs.append(pvr)
    self.PairValueCount = len(self.PairValueRecord)

    for i, values in enumerate(padded):
        for j, glyph in enumerate(glyphs):
            # Fill in value from other subtables
            v = ot.PairValueRecord()
            v.SecondGlyph = glyph
            if values[j] is not None:
                vpair = values[j]
            else:
                vpair = _Lookup_PairPos_get_effective_value_pair(
                    merger, merger.lookup_subtables[i], self._firstGlyph, glyph
                )
            if vpair is None:
                v1, v2 = None, None
            else:
                v1 = getattr(vpair, "Value1", None)
                v2 = getattr(vpair, "Value2", None)
            v.Value1 = (
                otBase.ValueRecord(merger.valueFormat1, src=v1)
                if merger.valueFormat1
                else None
            )
            v.Value2 = (
                otBase.ValueRecord(merger.valueFormat2, src=v2)
                if merger.valueFormat2
                else None
            )
            values[j] = v
    del self._firstGlyph

    merger.mergeLists(self.PairValueRecord, padded)


def _PairPosFormat1_merge(self, lst, merger):
    assert allEqual(
        [l.ValueFormat2 == 0 for l in lst if l.PairSet]
    ), "Report bug against fonttools."

    # Merge everything else; makes sure Format is the same.
    merger.mergeObjects(
        self,
        lst,
        exclude=("Coverage", "PairSet", "PairSetCount", "ValueFormat1", "ValueFormat2"),
    )

    empty = ot.PairSet()
    empty.PairValueRecord = []
    empty.PairValueCount = 0

    # Align them
    glyphs, padded = _merge_GlyphOrders(
        merger.font,
        [v.Coverage.glyphs for v in lst],
        [v.PairSet for v in lst],
        default=empty,
    )

    self.Coverage.glyphs = glyphs
    self.PairSet = [ot.PairSet() for _ in glyphs]
    self.PairSetCount = len(self.PairSet)
    for glyph, ps in zip(glyphs, self.PairSet):
        ps._firstGlyph = glyph

    merger.mergeLists(self.PairSet, padded)


def _ClassDef_invert(self, allGlyphs=None):
    if isinstance(self, dict):
        classDefs = self
    else:
        classDefs = self.classDefs if self and self.classDefs else {}
    m = max(classDefs.values()) if classDefs else 0

    ret = []
    for _ in range(m + 1):
        ret.append(set())

    for k, v in classDefs.items():
        ret[v].add(k)

    # Class-0 is special.  It's "everything else".
    if allGlyphs is None:
        ret[0] = None
    else:
        # Limit all classes to glyphs in allGlyphs.
        # Collect anything without a non-zero class into class=zero.
        ret[0] = class0 = set(allGlyphs)
        for s in ret[1:]:
            s.intersection_update(class0)
            class0.difference_update(s)

    return ret


def _ClassDef_merge_classify(lst, allGlyphses=None):
    self = ot.ClassDef()
    self.classDefs = classDefs = {}
    allGlyphsesWasNone = allGlyphses is None
    if allGlyphsesWasNone:
        allGlyphses = [None] * len(lst)

    classifier = classifyTools.Classifier()
    for classDef, allGlyphs in zip(lst, allGlyphses):
        sets = _ClassDef_invert(classDef, allGlyphs)
        if allGlyphs is None:
            sets = sets[1:]
        classifier.update(sets)
    classes = classifier.getClasses()

    if allGlyphsesWasNone:
        classes.insert(0, set())

    for i, classSet in enumerate(classes):
        if i == 0:
            continue
        for g in classSet:
            classDefs[g] = i

    return self, classes


def _PairPosFormat2_align_matrices(self, lst, font, transparent=False):
    matrices = [l.Class1Record for l in lst]

    # Align first classes
    self.ClassDef1, classes = _ClassDef_merge_classify(
        [l.ClassDef1 for l in lst], [l.Coverage.glyphs for l in lst]
    )
    self.Class1Count = len(classes)
    new_matrices = []
    for l, matrix in zip(lst, matrices):
        nullRow = None
        coverage = set(l.Coverage.glyphs)
        classDef1 = l.ClassDef1.classDefs
        class1Records = []
        for classSet in classes:
            exemplarGlyph = next(iter(classSet))
            if exemplarGlyph not in coverage:
                # Follow-up to e6125b353e1f54a0280ded5434b8e40d042de69f,
                # Fixes https://github.com/googlei18n/fontmake/issues/470
                # Again, revert 8d441779e5afc664960d848f62c7acdbfc71d7b9
                # when merger becomes selfless.
                nullRow = None
                if nullRow is None:
                    nullRow = ot.Class1Record()
                    class2records = nullRow.Class2Record = []
                    # TODO: When merger becomes selfless, revert e6125b353e1f54a0280ded5434b8e40d042de69f
                    for _ in range(l.Class2Count):
                        if transparent:
                            rec2 = None
                        else:
                            rec2 = ot.Class2Record()
                            rec2.Value1 = (
                                otBase.ValueRecord(self.ValueFormat1)
                                if self.ValueFormat1
                                else None
                            )
                            rec2.Value2 = (
                                otBase.ValueRecord(self.ValueFormat2)
                                if self.ValueFormat2
                                else None
                            )
                        class2records.append(rec2)
                rec1 = nullRow
            else:
                klass = classDef1.get(exemplarGlyph, 0)
                rec1 = matrix[klass]  # TODO handle out-of-range?
            class1Records.append(rec1)
        new_matrices.append(class1Records)
    matrices = new_matrices
    del new_matrices

    # Align second classes
    self.ClassDef2, classes = _ClassDef_merge_classify([l.ClassDef2 for l in lst])
    self.Class2Count = len(classes)
    new_matrices = []
    for l, matrix in zip(lst, matrices):
        classDef2 = l.ClassDef2.classDefs
        class1Records = []
        for rec1old in matrix:
            oldClass2Records = rec1old.Class2Record
            rec1new = ot.Class1Record()
            class2Records = rec1new.Class2Record = []
            for classSet in classes:
                if not classSet:  # class=0
                    rec2 = oldClass2Records[0]
                else:
                    exemplarGlyph = next(iter(classSet))
                    klass = classDef2.get(exemplarGlyph, 0)
                    rec2 = oldClass2Records[klass]
                class2Records.append(copy.deepcopy(rec2))
            class1Records.append(rec1new)
        new_matrices.append(class1Records)
    matrices = new_matrices
    del new_matrices

    return matrices


def _PairPosFormat2_merge(self, lst, merger):
    assert allEqual(
        [l.ValueFormat2 == 0 for l in lst if l.Class1Record]
    ), "Report bug against fonttools."

    merger.mergeObjects(
        self,
        lst,
        exclude=(
            "Coverage",
            "ClassDef1",
            "Class1Count",
            "ClassDef2",
            "Class2Count",
            "Class1Record",
            "ValueFormat1",
            "ValueFormat2",
        ),
    )

    # Align coverages
    glyphs, _ = _merge_GlyphOrders(merger.font, [v.Coverage.glyphs for v in lst])
    self.Coverage.glyphs = glyphs

    # Currently, if the coverage of PairPosFormat2 subtables are different,
    # we do NOT bother walking down the subtable list when filling in new
    # rows for alignment.  As such, this is only correct if current subtable
    # is the last subtable in the lookup.  Ensure that.
    #
    # Note that our canonicalization process merges trailing PairPosFormat2's,
    # so in reality this is rare.
    for l, subtables in zip(lst, merger.lookup_subtables):
        if l.Coverage.glyphs != glyphs:
            assert l == subtables[-1]

    matrices = _PairPosFormat2_align_matrices(self, lst, merger.font)

    self.Class1Record = list(matrices[0])  # TODO move merger to be selfless
    merger.mergeLists(self.Class1Record, matrices)


@AligningMerger.merger(ot.PairPos)
def merge(merger, self, lst):
    merger.valueFormat1 = self.ValueFormat1 = reduce(
        int.__or__, [l.ValueFormat1 for l in lst], 0
    )
    merger.valueFormat2 = self.ValueFormat2 = reduce(
        int.__or__, [l.ValueFormat2 for l in lst], 0
    )

    if self.Format == 1:
        _PairPosFormat1_merge(self, lst, merger)
    elif self.Format == 2:
        _PairPosFormat2_merge(self, lst, merger)
    else:
        raise UnsupportedFormat(merger, subtable="pair positioning lookup")

    del merger.valueFormat1, merger.valueFormat2

    # Now examine the list of value records, and update to the union of format values,
    # as merge might have created new values.
    vf1 = 0
    vf2 = 0
    if self.Format == 1:
        for pairSet in self.PairSet:
            for pairValueRecord in pairSet.PairValueRecord:
                pv1 = getattr(pairValueRecord, "Value1", None)
                if pv1 is not None:
                    vf1 |= pv1.getFormat()
                pv2 = getattr(pairValueRecord, "Value2", None)
                if pv2 is not None:
                    vf2 |= pv2.getFormat()
    elif self.Format == 2:
        for class1Record in self.Class1Record:
            for class2Record in class1Record.Class2Record:
                pv1 = getattr(class2Record, "Value1", None)
                if pv1 is not None:
                    vf1 |= pv1.getFormat()
                pv2 = getattr(class2Record, "Value2", None)
                if pv2 is not None:
                    vf2 |= pv2.getFormat()
    self.ValueFormat1 = vf1
    self.ValueFormat2 = vf2


def _MarkBasePosFormat1_merge(self, lst, merger, Mark="Mark", Base="Base"):
    self.ClassCount = max(l.ClassCount for l in lst)

    MarkCoverageGlyphs, MarkRecords = _merge_GlyphOrders(
        merger.font,
        [getattr(l, Mark + "Coverage").glyphs for l in lst],
        [getattr(l, Mark + "Array").MarkRecord for l in lst],
    )
    getattr(self, Mark + "Coverage").glyphs = MarkCoverageGlyphs

    BaseCoverageGlyphs, BaseRecords = _merge_GlyphOrders(
        merger.font,
        [getattr(l, Base + "Coverage").glyphs for l in lst],
        [getattr(getattr(l, Base + "Array"), Base + "Record") for l in lst],
    )
    getattr(self, Base + "Coverage").glyphs = BaseCoverageGlyphs

    # MarkArray
    records = []
    for g, glyphRecords in zip(MarkCoverageGlyphs, zip(*MarkRecords)):
        allClasses = [r.Class for r in glyphRecords if r is not None]

        # TODO Right now we require that all marks have same class in
        # all masters that cover them.  This is not required.
        #
        # We can relax that by just requiring that all marks that have
        # the same class in a master, have the same class in every other
        # master.  Indeed, if, say, a sparse master only covers one mark,
        # that mark probably will get class 0, which would possibly be
        # different from its class in other masters.
        #
        # We can even go further and reclassify marks to support any
        # input.  But, since, it's unlikely that two marks being both,
        # say, "top" in one master, and one being "top" and other being
        # "top-right" in another master, we shouldn't do that, as any
        # failures in that case will probably signify mistakes in the
        # input masters.

        if not allEqual(allClasses):
            raise ShouldBeConstant(merger, expected=allClasses[0], got=allClasses)
        else:
            rec = ot.MarkRecord()
            rec.Class = allClasses[0]
            allAnchors = [None if r is None else r.MarkAnchor for r in glyphRecords]
            if allNone(allAnchors):
                anchor = None
            else:
                anchor = ot.Anchor()
                anchor.Format = 1
                merger.mergeThings(anchor, allAnchors)
            rec.MarkAnchor = anchor
        records.append(rec)
    array = ot.MarkArray()
    array.MarkRecord = records
    array.MarkCount = len(records)
    setattr(self, Mark + "Array", array)

    # BaseArray
    records = []
    for g, glyphRecords in zip(BaseCoverageGlyphs, zip(*BaseRecords)):
        if allNone(glyphRecords):
            rec = None
        else:
            rec = getattr(ot, Base + "Record")()
            anchors = []
            setattr(rec, Base + "Anchor", anchors)
            glyphAnchors = [
                [] if r is None else getattr(r, Base + "Anchor") for r in glyphRecords
            ]
            for l in glyphAnchors:
                l.extend([None] * (self.ClassCount - len(l)))
            for allAnchors in zip(*glyphAnchors):
                if allNone(allAnchors):
                    anchor = None
                else:
                    anchor = ot.Anchor()
                    anchor.Format = 1
                    merger.mergeThings(anchor, allAnchors)
                anchors.append(anchor)
        records.append(rec)
    array = getattr(ot, Base + "Array")()
    setattr(array, Base + "Record", records)
    setattr(array, Base + "Count", len(records))
    setattr(self, Base + "Array", array)


@AligningMerger.merger(ot.MarkBasePos)
def merge(merger, self, lst):
    if not allEqualTo(self.Format, (l.Format for l in lst)):
        raise InconsistentFormats(
            merger,
            subtable="mark-to-base positioning lookup",
            expected=self.Format,
            got=[l.Format for l in lst],
        )
    if self.Format == 1:
        _MarkBasePosFormat1_merge(self, lst, merger)
    else:
        raise UnsupportedFormat(merger, subtable="mark-to-base positioning lookup")


@AligningMerger.merger(ot.MarkMarkPos)
def merge(merger, self, lst):
    if not allEqualTo(self.Format, (l.Format for l in lst)):
        raise InconsistentFormats(
            merger,
            subtable="mark-to-mark positioning lookup",
            expected=self.Format,
            got=[l.Format for l in lst],
        )
    if self.Format == 1:
        _MarkBasePosFormat1_merge(self, lst, merger, "Mark1", "Mark2")
    else:
        raise UnsupportedFormat(merger, subtable="mark-to-mark positioning lookup")


def _PairSet_flatten(lst, font):
    self = ot.PairSet()
    self.Coverage = ot.Coverage()

    # Align them
    glyphs, padded = _merge_GlyphOrders(
        font,
        [[v.SecondGlyph for v in vs.PairValueRecord] for vs in lst],
        [vs.PairValueRecord for vs in lst],
    )

    self.Coverage.glyphs = glyphs
    self.PairValueRecord = pvrs = []
    for values in zip(*padded):
        for v in values:
            if v is not None:
                pvrs.append(v)
                break
        else:
            assert False
    self.PairValueCount = len(self.PairValueRecord)

    return self


def _Lookup_PairPosFormat1_subtables_flatten(lst, font):
    assert allEqual(
        [l.ValueFormat2 == 0 for l in lst if l.PairSet]
    ), "Report bug against fonttools."

    self = ot.PairPos()
    self.Format = 1
    self.Coverage = ot.Coverage()
    self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
    self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)

    # Align them
    glyphs, padded = _merge_GlyphOrders(
        font, [v.Coverage.glyphs for v in lst], [v.PairSet for v in lst]
    )

    self.Coverage.glyphs = glyphs
    self.PairSet = [
        _PairSet_flatten([v for v in values if v is not None], font)
        for values in zip(*padded)
    ]
    self.PairSetCount = len(self.PairSet)
    return self


def _Lookup_PairPosFormat2_subtables_flatten(lst, font):
    assert allEqual(
        [l.ValueFormat2 == 0 for l in lst if l.Class1Record]
    ), "Report bug against fonttools."

    self = ot.PairPos()
    self.Format = 2
    self.Coverage = ot.Coverage()
    self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
    self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)

    # Align them
    glyphs, _ = _merge_GlyphOrders(font, [v.Coverage.glyphs for v in lst])
    self.Coverage.glyphs = glyphs

    matrices = _PairPosFormat2_align_matrices(self, lst, font, transparent=True)

    matrix = self.Class1Record = []
    for rows in zip(*matrices):
        row = ot.Class1Record()
        matrix.append(row)
        row.Class2Record = []
        row = row.Class2Record
        for cols in zip(*list(r.Class2Record for r in rows)):
            col = next(iter(c for c in cols if c is not None))
            row.append(col)

    return self


def _Lookup_PairPos_subtables_canonicalize(lst, font):
    """Merge multiple Format1 subtables at the beginning of lst,
    and merge multiple consecutive Format2 subtables that have the same
    Class2 (ie. were split because of offset overflows).  Returns new list."""
    lst = list(lst)

    l = len(lst)
    i = 0
    while i < l and lst[i].Format == 1:
        i += 1
    lst[:i] = [_Lookup_PairPosFormat1_subtables_flatten(lst[:i], font)]

    l = len(lst)
    i = l
    while i > 0 and lst[i - 1].Format == 2:
        i -= 1
    lst[i:] = [_Lookup_PairPosFormat2_subtables_flatten(lst[i:], font)]

    return lst


def _Lookup_SinglePos_subtables_flatten(lst, font, min_inclusive_rec_format):
    glyphs, _ = _merge_GlyphOrders(font, [v.Coverage.glyphs for v in lst], None)
    num_glyphs = len(glyphs)
    new = ot.SinglePos()
    new.Format = 2
    new.ValueFormat = min_inclusive_rec_format
    new.Coverage = ot.Coverage()
    new.Coverage.glyphs = glyphs
    new.ValueCount = num_glyphs
    new.Value = [None] * num_glyphs
    for singlePos in lst:
        if singlePos.Format == 1:
            val_rec = singlePos.Value
            for gname in singlePos.Coverage.glyphs:
                i = glyphs.index(gname)
                new.Value[i] = copy.deepcopy(val_rec)
        elif singlePos.Format == 2:
            for j, gname in enumerate(singlePos.Coverage.glyphs):
                val_rec = singlePos.Value[j]
                i = glyphs.index(gname)
                new.Value[i] = copy.deepcopy(val_rec)
    return [new]


@AligningMerger.merger(ot.CursivePos)
def merge(merger, self, lst):
    # Align them
    glyphs, padded = _merge_GlyphOrders(
        merger.font,
        [l.Coverage.glyphs for l in lst],
        [l.EntryExitRecord for l in lst],
    )

    self.Format = 1
    self.Coverage = ot.Coverage()
    self.Coverage.glyphs = glyphs
    self.EntryExitRecord = []
    for _ in glyphs:
        rec = ot.EntryExitRecord()
        rec.EntryAnchor = ot.Anchor()
        rec.EntryAnchor.Format = 1
        rec.ExitAnchor = ot.Anchor()
        rec.ExitAnchor.Format = 1
        self.EntryExitRecord.append(rec)
    merger.mergeLists(self.EntryExitRecord, padded)
    self.EntryExitCount = len(self.EntryExitRecord)


@AligningMerger.merger(ot.EntryExitRecord)
def merge(merger, self, lst):
    if all(master.EntryAnchor is None for master in lst):
        self.EntryAnchor = None
    if all(master.ExitAnchor is None for master in lst):
        self.ExitAnchor = None
    merger.mergeObjects(self, lst)


@AligningMerger.merger(ot.Lookup)
def merge(merger, self, lst):
    subtables = merger.lookup_subtables = [l.SubTable for l in lst]

    # Remove Extension subtables
    for l, sts in list(zip(lst, subtables)) + [(self, self.SubTable)]:
        if not sts:
            continue
        if sts[0].__class__.__name__.startswith("Extension"):
            if not allEqual([st.__class__ for st in sts]):
                raise InconsistentExtensions(
                    merger,
                    expected="Extension",
                    got=[st.__class__.__name__ for st in sts],
                )
            if not allEqual([st.ExtensionLookupType for st in sts]):
                raise InconsistentExtensions(merger)
            l.LookupType = sts[0].ExtensionLookupType
            new_sts = [st.ExtSubTable for st in sts]
            del sts[:]
            sts.extend(new_sts)

    isPairPos = self.SubTable and isinstance(self.SubTable[0], ot.PairPos)

    if isPairPos:
        # AFDKO and feaLib sometimes generate two Format1 subtables instead of one.
        # Merge those before continuing.
        # https://github.com/fonttools/fonttools/issues/719
        self.SubTable = _Lookup_PairPos_subtables_canonicalize(
            self.SubTable, merger.font
        )
        subtables = merger.lookup_subtables = [
            _Lookup_PairPos_subtables_canonicalize(st, merger.font) for st in subtables
        ]
    else:
        isSinglePos = self.SubTable and isinstance(self.SubTable[0], ot.SinglePos)
        if isSinglePos:
            numSubtables = [len(st) for st in subtables]
            if not all([nums == numSubtables[0] for nums in numSubtables]):
                # Flatten list of SinglePos subtables to single Format 2 subtable,
                # with all value records set to the rec format type.
                # We use buildSinglePos() to optimize the lookup after merging.
                valueFormatList = [t.ValueFormat for st in subtables for t in st]
                # Find the minimum value record that can accomodate all the singlePos subtables.
                mirf = reduce(ior, valueFormatList)
                self.SubTable = _Lookup_SinglePos_subtables_flatten(
                    self.SubTable, merger.font, mirf
                )
                subtables = merger.lookup_subtables = [
                    _Lookup_SinglePos_subtables_flatten(st, merger.font, mirf)
                    for st in subtables
                ]
                flattened = True
            else:
                flattened = False

    merger.mergeLists(self.SubTable, subtables)
    self.SubTableCount = len(self.SubTable)

    if isPairPos:
        # If format-1 subtable created during canonicalization is empty, remove it.
        assert len(self.SubTable) >= 1 and self.SubTable[0].Format == 1
        if not self.SubTable[0].Coverage.glyphs:
            self.SubTable.pop(0)
            self.SubTableCount -= 1

        # If format-2 subtable created during canonicalization is empty, remove it.
        assert len(self.SubTable) >= 1 and self.SubTable[-1].Format == 2
        if not self.SubTable[-1].Coverage.glyphs:
            self.SubTable.pop(-1)
            self.SubTableCount -= 1

        # Compact the merged subtables
        # This is a good moment to do it because the compaction should create
        # smaller subtables, which may prevent overflows from happening.
        # Keep reading the value from the ENV until ufo2ft switches to the config system
        level = merger.font.cfg.get(
            "fontTools.otlLib.optimize.gpos:COMPRESSION_LEVEL",
            default=_compression_level_from_env(),
        )
        if level != 0:
            log.info("Compacting GPOS...")
            self.SubTable = compact_pair_pos(merger.font, level, self.SubTable)
            self.SubTableCount = len(self.SubTable)

    elif isSinglePos and flattened:
        singlePosTable = self.SubTable[0]
        glyphs = singlePosTable.Coverage.glyphs
        # We know that singlePosTable is Format 2, as this is set
        # in _Lookup_SinglePos_subtables_flatten.
        singlePosMapping = {
            gname: valRecord for gname, valRecord in zip(glyphs, singlePosTable.Value)
        }
        self.SubTable = buildSinglePos(
            singlePosMapping, merger.font.getReverseGlyphMap()
        )
    merger.mergeObjects(self, lst, exclude=["SubTable", "SubTableCount"])

    del merger.lookup_subtables


#
# InstancerMerger
#


class InstancerMerger(AligningMerger):
    """A merger that takes multiple master fonts, and instantiates
    an instance."""

    def __init__(self, font, model, location):
        Merger.__init__(self, font)
        self.model = model
        self.location = location
        self.masterScalars = model.getMasterScalars(location)


@InstancerMerger.merger(ot.CaretValue)
def merge(merger, self, lst):
    assert self.Format == 1
    Coords = [a.Coordinate for a in lst]
    model = merger.model
    masterScalars = merger.masterScalars
    self.Coordinate = otRound(
        model.interpolateFromValuesAndScalars(Coords, masterScalars)
    )


@InstancerMerger.merger(ot.Anchor)
def merge(merger, self, lst):
    assert self.Format == 1
    XCoords = [a.XCoordinate for a in lst]
    YCoords = [a.YCoordinate for a in lst]
    model = merger.model
    masterScalars = merger.masterScalars
    self.XCoordinate = otRound(
        model.interpolateFromValuesAndScalars(XCoords, masterScalars)
    )
    self.YCoordinate = otRound(
        model.interpolateFromValuesAndScalars(YCoords, masterScalars)
    )


@InstancerMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
    model = merger.model
    masterScalars = merger.masterScalars
    # TODO Handle differing valueformats
    for name, tableName in [
        ("XAdvance", "XAdvDevice"),
        ("YAdvance", "YAdvDevice"),
        ("XPlacement", "XPlaDevice"),
        ("YPlacement", "YPlaDevice"),
    ]:
        assert not hasattr(self, tableName)

        if hasattr(self, name):
            values = [getattr(a, name, 0) for a in lst]
            value = otRound(
                model.interpolateFromValuesAndScalars(values, masterScalars)
            )
            setattr(self, name, value)


#
# MutatorMerger
#


class MutatorMerger(AligningMerger):
    """A merger that takes a variable font, and instantiates
    an instance.  While there's no "merging" to be done per se,
    the operation can benefit from many operations that the
    aligning merger does."""

    def __init__(self, font, instancer, deleteVariations=True):
        Merger.__init__(self, font)
        self.instancer = instancer
        self.deleteVariations = deleteVariations


@MutatorMerger.merger((ot.CaretValue, ot.BaseCoord))
def merge(merger, self, lst):
    # Hack till we become selfless.
    self.__dict__ = lst[0].__dict__.copy()

    if self.Format != 3:
        return

    instancer = merger.instancer
    dev = self.DeviceTable
    if merger.deleteVariations:
        del self.DeviceTable
    if dev:
        assert dev.DeltaFormat == 0x8000
        varidx = (dev.StartSize << 16) + dev.EndSize
        delta = otRound(instancer[varidx])
        self.Coordinate += delta

    if merger.deleteVariations:
        self.Format = 1


@MutatorMerger.merger(ot.Anchor)
def merge(merger, self, lst):
    # Hack till we become selfless.
    self.__dict__ = lst[0].__dict__.copy()

    if self.Format != 3:
        return

    instancer = merger.instancer
    for v in "XY":
        tableName = v + "DeviceTable"
        if not hasattr(self, tableName):
            continue
        dev = getattr(self, tableName)
        if merger.deleteVariations:
            delattr(self, tableName)
        if dev is None:
            continue

        assert dev.DeltaFormat == 0x8000
        varidx = (dev.StartSize << 16) + dev.EndSize
        delta = otRound(instancer[varidx])

        attr = v + "Coordinate"
        setattr(self, attr, getattr(self, attr) + delta)

    if merger.deleteVariations:
        self.Format = 1


@MutatorMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
    # Hack till we become selfless.
    self.__dict__ = lst[0].__dict__.copy()

    instancer = merger.instancer
    for name, tableName in [
        ("XAdvance", "XAdvDevice"),
        ("YAdvance", "YAdvDevice"),
        ("XPlacement", "XPlaDevice"),
        ("YPlacement", "YPlaDevice"),
    ]:
        if not hasattr(self, tableName):
            continue
        dev = getattr(self, tableName)
        if merger.deleteVariations:
            delattr(self, tableName)
        if dev is None:
            continue

        assert dev.DeltaFormat == 0x8000
        varidx = (dev.StartSize << 16) + dev.EndSize
        delta = otRound(instancer[varidx])

        setattr(self, name, getattr(self, name, 0) + delta)


#
# VariationMerger
#


class VariationMerger(AligningMerger):
    """A merger that takes multiple master fonts, and builds a
    variable font."""

    def __init__(self, model, axisTags, font):
        Merger.__init__(self, font)
        self.store_builder = varStore.OnlineVarStoreBuilder(axisTags)
        self.setModel(model)

    def setModel(self, model):
        self.model = model
        self.store_builder.setModel(model)

    def mergeThings(self, out, lst):
        masterModel = None
        origTTFs = None
        if None in lst:
            if allNone(lst):
                if out is not None:
                    raise FoundANone(self, got=lst)
                return

            # temporarily subset the list of master ttfs to the ones for which
            # master values are not None
            origTTFs = self.ttfs
            if self.ttfs:
                self.ttfs = subList([v is not None for v in lst], self.ttfs)

            masterModel = self.model
            model, lst = masterModel.getSubModel(lst)
            self.setModel(model)

        super(VariationMerger, self).mergeThings(out, lst)

        if masterModel:
            self.setModel(masterModel)
        if origTTFs:
            self.ttfs = origTTFs


def buildVarDevTable(store_builder, master_values):
    if allEqual(master_values):
        return master_values[0], None
    base, varIdx = store_builder.storeMasters(master_values)
    return base, builder.buildVarDevTable(varIdx)


@VariationMerger.merger(ot.BaseCoord)
def merge(merger, self, lst):
    if self.Format != 1:
        raise UnsupportedFormat(merger, subtable="a baseline coordinate")
    self.Coordinate, DeviceTable = buildVarDevTable(
        merger.store_builder, [a.Coordinate for a in lst]
    )
    if DeviceTable:
        self.Format = 3
        self.DeviceTable = DeviceTable


@VariationMerger.merger(ot.CaretValue)
def merge(merger, self, lst):
    if self.Format != 1:
        raise UnsupportedFormat(merger, subtable="a caret")
    self.Coordinate, DeviceTable = buildVarDevTable(
        merger.store_builder, [a.Coordinate for a in lst]
    )
    if DeviceTable:
        self.Format = 3
        self.DeviceTable = DeviceTable


@VariationMerger.merger(ot.Anchor)
def merge(merger, self, lst):
    if self.Format != 1:
        raise UnsupportedFormat(merger, subtable="an anchor")
    self.XCoordinate, XDeviceTable = buildVarDevTable(
        merger.store_builder, [a.XCoordinate for a in lst]
    )
    self.YCoordinate, YDeviceTable = buildVarDevTable(
        merger.store_builder, [a.YCoordinate for a in lst]
    )
    if XDeviceTable or YDeviceTable:
        self.Format = 3
        self.XDeviceTable = XDeviceTable
        self.YDeviceTable = YDeviceTable


@VariationMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
    for name, tableName in [
        ("XAdvance", "XAdvDevice"),
        ("YAdvance", "YAdvDevice"),
        ("XPlacement", "XPlaDevice"),
        ("YPlacement", "YPlaDevice"),
    ]:
        if hasattr(self, name):
            value, deviceTable = buildVarDevTable(
                merger.store_builder, [getattr(a, name, 0) for a in lst]
            )
            setattr(self, name, value)
            if deviceTable:
                setattr(self, tableName, deviceTable)


class COLRVariationMerger(VariationMerger):
    """A specialized VariationMerger that takes multiple master fonts containing
    COLRv1 tables, and builds a variable COLR font.

    COLR tables are special in that variable subtables can be associated with
    multiple delta-set indices (via VarIndexBase).
    They also contain tables that must change their type (not simply the Format)
    as they become variable (e.g. Affine2x3 -> VarAffine2x3) so this merger takes
    care of that too.
    """

    def __init__(self, model, axisTags, font, allowLayerReuse=True):
        VariationMerger.__init__(self, model, axisTags, font)
        # maps {tuple(varIdxes): VarIndexBase} to facilitate reuse of VarIndexBase
        # between variable tables with same varIdxes.
        self.varIndexCache = {}
        # flat list of all the varIdxes generated while merging
        self.varIdxes = []
        # set of id()s of the subtables that contain variations after merging
        # and need to be upgraded to the associated VarType.
        self.varTableIds = set()
        # we keep these around for rebuilding a LayerList while merging PaintColrLayers
        self.layers = []
        self.layerReuseCache = None
        if allowLayerReuse:
            self.layerReuseCache = LayerReuseCache()
        # flag to ensure BaseGlyphList is fully merged before LayerList gets processed
        self._doneBaseGlyphs = False

    def mergeTables(self, font, master_ttfs, tableTags=("COLR",)):
        if "COLR" in tableTags and "COLR" in font:
            # The merger modifies the destination COLR table in-place. If this contains
            # multiple PaintColrLayers referencing the same layers from LayerList, it's
            # a problem because we may risk modifying the same paint more than once, or
            # worse, fail while attempting to do that.
            # We don't know whether the master COLR table was built with layer reuse
            # disabled, thus to be safe we rebuild its LayerList so that it contains only
            # unique layers referenced from non-overlapping PaintColrLayers throughout
            # the base paint graphs.
            self.expandPaintColrLayers(font["COLR"].table)
        VariationMerger.mergeTables(self, font, master_ttfs, tableTags)

    def checkFormatEnum(self, out, lst, validate=lambda _: True):
        fmt = out.Format
        formatEnum = out.formatEnum
        ok = False
        try:
            fmt = formatEnum(fmt)
        except ValueError:
            pass
        else:
            ok = validate(fmt)
        if not ok:
            raise UnsupportedFormat(self, subtable=type(out).__name__, value=fmt)
        expected = fmt
        got = []
        for v in lst:
            fmt = getattr(v, "Format", None)
            try:
                fmt = formatEnum(fmt)
            except ValueError:
                pass
            got.append(fmt)
        if not allEqualTo(expected, got):
            raise InconsistentFormats(
                self,
                subtable=type(out).__name__,
                expected=expected,
                got=got,
            )
        return expected

    def mergeSparseDict(self, out, lst):
        for k in out.keys():
            try:
                self.mergeThings(out[k], [v.get(k) for v in lst])
            except VarLibMergeError as e:
                e.stack.append(f"[{k!r}]")
                raise

    def mergeAttrs(self, out, lst, attrs):
        for attr in attrs:
            value = getattr(out, attr)
            values = [getattr(item, attr) for item in lst]
            try:
                self.mergeThings(value, values)
            except VarLibMergeError as e:
                e.stack.append(f".{attr}")
                raise

    def storeMastersForAttr(self, out, lst, attr):
        master_values = [getattr(item, attr) for item in lst]

        # VarStore treats deltas for fixed-size floats as integers, so we
        # must convert master values to int before storing them in the builder
        # then back to float.
        is_fixed_size_float = False
        conv = out.getConverterByName(attr)
        if isinstance(conv, BaseFixedValue):
            is_fixed_size_float = True
            master_values = [conv.toInt(v) for v in master_values]

        baseValue = master_values[0]
        varIdx = ot.NO_VARIATION_INDEX
        if not allEqual(master_values):
            baseValue, varIdx = self.store_builder.storeMasters(master_values)

        if is_fixed_size_float:
            baseValue = conv.fromInt(baseValue)

        return baseValue, varIdx

    def storeVariationIndices(self, varIdxes) -> int:
        # try to reuse an existing VarIndexBase for the same varIdxes, or else
        # create a new one
        key = tuple(varIdxes)
        varIndexBase = self.varIndexCache.get(key)

        if varIndexBase is None:
            # scan for a full match anywhere in the self.varIdxes
            for i in range(len(self.varIdxes) - len(varIdxes) + 1):
                if self.varIdxes[i : i + len(varIdxes)] == varIdxes:
                    self.varIndexCache[key] = varIndexBase = i
                    break

        if varIndexBase is None:
            # try find a partial match at the end of the self.varIdxes
            for n in range(len(varIdxes) - 1, 0, -1):
                if self.varIdxes[-n:] == varIdxes[:n]:
                    varIndexBase = len(self.varIdxes) - n
                    self.varIndexCache[key] = varIndexBase
                    self.varIdxes.extend(varIdxes[n:])
                    break

        if varIndexBase is None:
            # no match found, append at the end
            self.varIndexCache[key] = varIndexBase = len(self.varIdxes)
            self.varIdxes.extend(varIdxes)

        return varIndexBase

    def mergeVariableAttrs(self, out, lst, attrs) -> int:
        varIndexBase = ot.NO_VARIATION_INDEX
        varIdxes = []
        for attr in attrs:
            baseValue, varIdx = self.storeMastersForAttr(out, lst, attr)
            setattr(out, attr, baseValue)
            varIdxes.append(varIdx)

        if any(v != ot.NO_VARIATION_INDEX for v in varIdxes):
            varIndexBase = self.storeVariationIndices(varIdxes)

        return varIndexBase

    @classmethod
    def convertSubTablesToVarType(cls, table):
        for path in dfs_base_table(
            table,
            skip_root=True,
            predicate=lambda path: (
                getattr(type(path[-1].value), "VarType", None) is not None
            ),
        ):
            st = path[-1]
            subTable = st.value
            varType = type(subTable).VarType
            newSubTable = varType()
            newSubTable.__dict__.update(subTable.__dict__)
            newSubTable.populateDefaults()
            parent = path[-2].value
            if st.index is not None:
                getattr(parent, st.name)[st.index] = newSubTable
            else:
                setattr(parent, st.name, newSubTable)

    @staticmethod
    def expandPaintColrLayers(colr):
        """Rebuild LayerList without PaintColrLayers reuse.

        Each base paint graph is fully DFS-traversed (with exception of PaintColrGlyph
        which are irrelevant for this); any layers referenced via PaintColrLayers are
        collected into a new LayerList and duplicated when reuse is detected, to ensure
        that all paints are distinct objects at the end of the process.
        PaintColrLayers's FirstLayerIndex/NumLayers are updated so that no overlap
        is left. Also, any consecutively nested PaintColrLayers are flattened.
        The COLR table's LayerList is replaced with the new unique layers.
        A side effect is also that any layer from the old LayerList which is not
        referenced by any PaintColrLayers is dropped.
        """
        if not colr.LayerList:
            # if no LayerList, there's nothing to expand
            return
        uniqueLayerIDs = set()
        newLayerList = []
        for rec in colr.BaseGlyphList.BaseGlyphPaintRecord:
            frontier = [rec.Paint]
            while frontier:
                paint = frontier.pop()
                if paint.Format == ot.PaintFormat.PaintColrGlyph:
                    # don't traverse these, we treat them as constant for merging
                    continue
                elif paint.Format == ot.PaintFormat.PaintColrLayers:
                    # de-treeify any nested PaintColrLayers, append unique copies to
                    # the new layer list and update PaintColrLayers index/count
                    children = list(_flatten_layers(paint, colr))
                    first_layer_index = len(newLayerList)
                    for layer in children:
                        if id(layer) in uniqueLayerIDs:
                            layer = copy.deepcopy(layer)
                            assert id(layer) not in uniqueLayerIDs
                        newLayerList.append(layer)
                        uniqueLayerIDs.add(id(layer))
                    paint.FirstLayerIndex = first_layer_index
                    paint.NumLayers = len(children)
                else:
                    children = paint.getChildren(colr)
                frontier.extend(reversed(children))
        # sanity check all the new layers are distinct objects
        assert len(newLayerList) == len(uniqueLayerIDs)
        colr.LayerList.Paint = newLayerList
        colr.LayerList.LayerCount = len(newLayerList)


@COLRVariationMerger.merger(ot.BaseGlyphList)
def merge(merger, self, lst):
    # ignore BaseGlyphCount, allow sparse glyph sets across masters
    out = {rec.BaseGlyph: rec for rec in self.BaseGlyphPaintRecord}
    masters = [{rec.BaseGlyph: rec for rec in m.BaseGlyphPaintRecord} for m in lst]

    for i, g in enumerate(out.keys()):
        try:
            # missing base glyphs don't participate in the merge
            merger.mergeThings(out[g], [v.get(g) for v in masters])
        except VarLibMergeError as e:
            e.stack.append(f".BaseGlyphPaintRecord[{i}]")
            e.cause["location"] = f"base glyph {g!r}"
            raise

    merger._doneBaseGlyphs = True


@COLRVariationMerger.merger(ot.LayerList)
def merge(merger, self, lst):
    # nothing to merge for LayerList, assuming we have already merged all PaintColrLayers
    # found while traversing the paint graphs rooted at BaseGlyphPaintRecords.
    assert merger._doneBaseGlyphs, "BaseGlyphList must be merged before LayerList"
    # Simply flush the final list of layers and go home.
    self.LayerCount = len(merger.layers)
    self.Paint = merger.layers


def _flatten_layers(root, colr):
    assert root.Format == ot.PaintFormat.PaintColrLayers
    for paint in root.getChildren(colr):
        if paint.Format == ot.PaintFormat.PaintColrLayers:
            yield from _flatten_layers(paint, colr)
        else:
            yield paint


def _merge_PaintColrLayers(self, out, lst):
    # we only enforce that the (flat) number of layers is the same across all masters
    # but we allow FirstLayerIndex to differ to acommodate for sparse glyph sets.

    out_layers = list(_flatten_layers(out, self.font["COLR"].table))

    # sanity check ttfs are subset to current values (see VariationMerger.mergeThings)
    # before matching each master PaintColrLayers to its respective COLR by position
    assert len(self.ttfs) == len(lst)
    master_layerses = [
        list(_flatten_layers(lst[i], self.ttfs[i]["COLR"].table))
        for i in range(len(lst))
    ]

    try:
        self.mergeLists(out_layers, master_layerses)
    except VarLibMergeError as e:
        # NOTE: This attribute doesn't actually exist in PaintColrLayers but it's
        # handy to have it in the stack trace for debugging.
        e.stack.append(".Layers")
        raise

    # following block is very similar to LayerListBuilder._beforeBuildPaintColrLayers
    # but I couldn't find a nice way to share the code between the two...

    if self.layerReuseCache is not None:
        # successful reuse can make the list smaller
        out_layers = self.layerReuseCache.try_reuse(out_layers)

    # if the list is still too big we need to tree-fy it
    is_tree = len(out_layers) > MAX_PAINT_COLR_LAYER_COUNT
    out_layers = build_n_ary_tree(out_layers, n=MAX_PAINT_COLR_LAYER_COUNT)

    # We now have a tree of sequences with Paint leaves.
    # Convert the sequences into PaintColrLayers.
    def listToColrLayers(paint):
        if isinstance(paint, list):
            layers = [listToColrLayers(l) for l in paint]
            paint = ot.Paint()
            paint.Format = int(ot.PaintFormat.PaintColrLayers)
            paint.NumLayers = len(layers)
            paint.FirstLayerIndex = len(self.layers)
            self.layers.extend(layers)
            if self.layerReuseCache is not None:
                self.layerReuseCache.add(layers, paint.FirstLayerIndex)
        return paint

    out_layers = [listToColrLayers(l) for l in out_layers]

    if len(out_layers) == 1 and out_layers[0].Format == ot.PaintFormat.PaintColrLayers:
        # special case when the reuse cache finds a single perfect PaintColrLayers match
        # (it can only come from a successful reuse, _flatten_layers has gotten rid of
        # all nested PaintColrLayers already); we assign it directly and avoid creating
        # an extra table
        out.NumLayers = out_layers[0].NumLayers
        out.FirstLayerIndex = out_layers[0].FirstLayerIndex
    else:
        out.NumLayers = len(out_layers)
        out.FirstLayerIndex = len(self.layers)

        self.layers.extend(out_layers)

        # Register our parts for reuse provided we aren't a tree
        # If we are a tree the leaves registered for reuse and that will suffice
        if self.layerReuseCache is not None and not is_tree:
            self.layerReuseCache.add(out_layers, out.FirstLayerIndex)


@COLRVariationMerger.merger((ot.Paint, ot.ClipBox))
def merge(merger, self, lst):
    fmt = merger.checkFormatEnum(self, lst, lambda fmt: not fmt.is_variable())

    if fmt is ot.PaintFormat.PaintColrLayers:
        _merge_PaintColrLayers(merger, self, lst)
        return

    varFormat = fmt.as_variable()

    varAttrs = ()
    if varFormat is not None:
        varAttrs = otBase.getVariableAttrs(type(self), varFormat)
    staticAttrs = (c.name for c in self.getConverters() if c.name not in varAttrs)

    merger.mergeAttrs(self, lst, staticAttrs)

    varIndexBase = merger.mergeVariableAttrs(self, lst, varAttrs)

    subTables = [st.value for st in self.iterSubTables()]

    # Convert table to variable if itself has variations or any subtables have
    isVariable = varIndexBase != ot.NO_VARIATION_INDEX or any(
        id(table) in merger.varTableIds for table in subTables
    )

    if isVariable:
        if varAttrs:
            # Some PaintVar* don't have any scalar attributes that can vary,
            # only indirect offsets to other variable subtables, thus have
            # no VarIndexBase of their own (e.g. PaintVarTransform)
            self.VarIndexBase = varIndexBase

        if subTables:
            # Convert Affine2x3 -> VarAffine2x3, ColorLine -> VarColorLine, etc.
            merger.convertSubTablesToVarType(self)

        assert varFormat is not None
        self.Format = int(varFormat)


@COLRVariationMerger.merger((ot.Affine2x3, ot.ColorStop))
def merge(merger, self, lst):
    varType = type(self).VarType

    varAttrs = otBase.getVariableAttrs(varType)
    staticAttrs = (c.name for c in self.getConverters() if c.name not in varAttrs)

    merger.mergeAttrs(self, lst, staticAttrs)

    varIndexBase = merger.mergeVariableAttrs(self, lst, varAttrs)

    if varIndexBase != ot.NO_VARIATION_INDEX:
        self.VarIndexBase = varIndexBase
        # mark as having variations so the parent table will convert to Var{Type}
        merger.varTableIds.add(id(self))


@COLRVariationMerger.merger(ot.ColorLine)
def merge(merger, self, lst):
    merger.mergeAttrs(self, lst, (c.name for c in self.getConverters()))

    if any(id(stop) in merger.varTableIds for stop in self.ColorStop):
        merger.convertSubTablesToVarType(self)
        merger.varTableIds.add(id(self))


@COLRVariationMerger.merger(ot.ClipList, "clips")
def merge(merger, self, lst):
    # 'sparse' in that we allow non-default masters to omit ClipBox entries
    # for some/all glyphs (i.e. they don't participate)
    merger.mergeSparseDict(self, lst)
<|endoftext|>from .interpolatableHelpers import *


def test_starting_point(glyph0, glyph1, ix, tolerance, matching):
    if matching is None:
        matching = list(range(len(glyph0.isomorphisms)))
    contour0 = glyph0.isomorphisms[ix]
    contour1 = glyph1.isomorphisms[matching[ix]]
    m0Vectors = glyph0.greenVectors
    m1Vectors = [glyph1.greenVectors[i] for i in matching]

    c0 = contour0[0]
    # Next few lines duplicated below.
    costs = [vdiff_hypot2_complex(c0[0], c1[0]) for c1 in contour1]
    min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
    first_cost = costs[0]
    proposed_point = contour1[min_cost_idx][1]
    reverse = contour1[min_cost_idx][2]

    if min_cost < first_cost * tolerance:
        # c0 is the first isomorphism of the m0 master
        # contour1 is list of all isomorphisms of the m1 master
        #
        # If the two shapes are both circle-ish and slightly
        # rotated, we detect wrong start point. This is for
        # example the case hundreds of times in
        # RobotoSerif-Italic[GRAD,opsz,wdth,wght].ttf
        #
        # If the proposed point is only one off from the first
        # point (and not reversed), try harder:
        #
        # Find the major eigenvector of the covariance matrix,
        # and rotate the contours by that angle. Then find the
        # closest point again.  If it matches this time, let it
        # pass.

        num_points = len(glyph1.points[ix])
        leeway = 3
        if not reverse and (
            proposed_point <= leeway or proposed_point >= num_points - leeway
        ):
            # Try harder

            # Recover the covariance matrix from the GreenVectors.
            # This is a 2x2 matrix.
            transforms = []
            for vector in (m0Vectors[ix], m1Vectors[ix]):
                # meanX = vector[1]
                # meanY = vector[2]
                stddevX = vector[3] * 0.5
                stddevY = vector[4] * 0.5
                correlation = vector[5]
                if correlation:
                    correlation /= abs(vector[0])

                # https://cookierobotics.com/007/
                a = stddevX * stddevX  # VarianceX
                c = stddevY * stddevY  # VarianceY
                b = correlation * stddevX * stddevY  # Covariance

                delta = (((a - c) * 0.5) ** 2 + b * b) ** 0.5
                lambda1 = (a + c) * 0.5 + delta  # Major eigenvalue
                lambda2 = (a + c) * 0.5 - delta  # Minor eigenvalue
                theta = atan2(lambda1 - a, b) if b != 0 else (pi * 0.5 if a < c else 0)
                trans = Transform()
                # Don't translate here. We are working on the complex-vector
                # that includes more than just the points. It's horrible what
                # we are doing anyway...
                # trans = trans.translate(meanX, meanY)
                trans = trans.rotate(theta)
                if lambda2 < 0:
                    lambda2 = 0
                trans = trans.scale(sqrt(lambda1), sqrt(lambda2))
                transforms.append(trans)

            trans = transforms[0]
            new_c0 = (
                [complex(*trans.transformPoint((pt.real, pt.imag))) for pt in c0[0]],
            ) + c0[1:]
            trans = transforms[1]
            new_contour1 = []
            for c1 in contour1:
                new_c1 = (
                    [
                        complex(*trans.transformPoint((pt.real, pt.imag)))
                        for pt in c1[0]
                    ],
                ) + c1[1:]
                new_contour1.append(new_c1)

            # Next few lines duplicate from above.
            costs = [
                vdiff_hypot2_complex(new_c0[0], new_c1[0]) for new_c1 in new_contour1
            ]
            min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
            first_cost = costs[0]
            if min_cost < first_cost * tolerance:
                # Don't report this
                # min_cost = first_cost
                # reverse = False
                # proposed_point = 0  # new_contour1[min_cost_idx][1]
                pass

    this_tolerance = min_cost / first_cost if first_cost else 1
    log.debug(
        "test-starting-point: tolerance %g",
        this_tolerance,
    )
    return this_tolerance, proposed_point, reverse
<|endoftext|>"""Module to build FeatureVariation tables:
https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#featurevariations-table

NOTE: The API is experimental and subject to change.
"""

from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict

from .errors import VarLibError, VarLibValidationError


def addFeatureVariations(font, conditionalSubstitutions, featureTag="rvrn"):
    """Add conditional substitutions to a Variable Font.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Example:

    # >>> f = TTFont(srcPath)
    # >>> condSubst = [
    # ...     # A list of (Region, Substitution) tuples.
    # ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
    # ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
    # ... ]
    # >>> addFeatureVariations(f, condSubst)
    # >>> f.save(dstPath)

    The `featureTag` parameter takes either a str or a iterable of str (the single str
    is kept for backwards compatibility), and defines which feature(s) will be
    associated with the feature variations.
    Note, if this is "rvrn", then the substitution lookup will be inserted at the
    beginning of the lookup list so that it is processed before others, otherwise
    for any other feature tags it will be appended last.
    """

    # process first when "rvrn" is the only listed tag
    featureTags = [featureTag] if isinstance(featureTag, str) else sorted(featureTag)
    processLast = "rvrn" not in featureTags or len(featureTags) > 1

    _checkSubstitutionGlyphsExist(
        glyphNames=set(font.getGlyphOrder()),
        substitutions=conditionalSubstitutions,
    )

    substitutions = overlayFeatureVariations(conditionalSubstitutions)

    # turn substitution dicts into tuples of tuples, so they are hashable
    conditionalSubstitutions, allSubstitutions = makeSubstitutionsHashable(
        substitutions
    )
    if "GSUB" not in font:
        font["GSUB"] = buildGSUB()
    else:
        existingTags = _existingVariableFeatures(font["GSUB"].table).intersection(
            featureTags
        )
        if existingTags:
            raise VarLibError(
                f"FeatureVariations already exist for feature tag(s): {existingTags}"
            )

    # setup lookups
    lookupMap = buildSubstitutionLookups(
        font["GSUB"].table, allSubstitutions, processLast
    )

    # addFeatureVariationsRaw takes a list of
    #  ( {condition}, [ lookup indices ] )
    # so rearrange our lookups to match
    conditionsAndLookups = []
    for conditionSet, substitutions in conditionalSubstitutions:
        conditionsAndLookups.append(
            (conditionSet, [lookupMap[s] for s in substitutions])
        )

    addFeatureVariationsRaw(font, font["GSUB"].table, conditionsAndLookups, featureTags)

    # Update OS/2.usMaxContext in case the font didn't have features before, but
    # does now, if the OS/2 table exists. The table may be required, but
    # fontTools needs to be able to deal with non-standard fonts. Since feature
    # variations are always 1:1 mappings, we can set the value to at least 1
    # instead of recomputing it with `otlLib.maxContextCalc.maxCtxFont()`.
    if (os2 := font.get("OS/2")) is not None:
        os2.usMaxContext = max(1, os2.usMaxContext)


def _existingVariableFeatures(table):
    existingFeatureVarsTags = set()
    if hasattr(table, "FeatureVariations") and table.FeatureVariations is not None:
        features = table.FeatureList.FeatureRecord
        for fvr in table.FeatureVariations.FeatureVariationRecord:
            for ftsr in fvr.FeatureTableSubstitution.SubstitutionRecord:
                existingFeatureVarsTags.add(features[ftsr.FeatureIndex].FeatureTag)
    return existingFeatureVarsTags


def _checkSubstitutionGlyphsExist(glyphNames, substitutions):
    referencedGlyphNames = set()
    for _, substitution in substitutions:
        referencedGlyphNames |= substitution.keys()
        referencedGlyphNames |= set(substitution.values())
    missing = referencedGlyphNames - glyphNames
    if missing:
        raise VarLibValidationError(
            "Missing glyphs are referenced in conditional substitution rules:"
            f" {', '.join(missing)}"
        )


def overlayFeatureVariations(conditionalSubstitutions):
    """Compute overlaps between all conditional substitutions.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Returns data is in similar but different format.  Overlaps of distinct
    substitution Boxes (*not* Regions) are explicitly listed as distinct rules,
    and rules with the same Box merged.  The more specific rules appear earlier
    in the resulting list.  Moreover, instead of just a dictionary of substitutions,
    a list of dictionaries is returned for substitutions corresponding to each
    unique space, with each dictionary being identical to one of the input
    substitution dictionaries.  These dictionaries are not merged to allow data
    sharing when they are converted into font tables.

    Example::

        >>> condSubst = [
        ...     # A list of (Region, Substitution) tuples.
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
        ...     ([{"wght": (0.5, 1.0), "wdth": (-1, 1.0)}], {"dollar": "dollar.rvrn"}),
        ... ]
        >>> from pprint import pprint
        >>> pprint(overlayFeatureVariations(condSubst))
        [({'wdth': (0.5, 1.0), 'wght': (0.5, 1.0)},
          [{'dollar': 'dollar.rvrn'}, {'cent': 'cent.rvrn'}]),
         ({'wdth': (0.5, 1.0)}, [{'cent': 'cent.rvrn'}]),
         ({'wght': (0.5, 1.0)}, [{'dollar': 'dollar.rvrn'}])]

    """

    # Merge same-substitutions rules, as this creates fewer number oflookups.
    merged = OrderedDict()
    for value, key in conditionalSubstitutions:
        key = hashdict(key)
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = value
    conditionalSubstitutions = [(v, dict(k)) for k, v in merged.items()]
    del merged

    # Merge same-region rules, as this is cheaper.
    # Also convert boxes to hashdict()
    #
    # Reversing is such that earlier entries win in case of conflicting substitution
    # rules for the same region.
    merged = OrderedDict()
    for key, value in reversed(conditionalSubstitutions):
        key = tuple(
            sorted(
                (hashdict(cleanupBox(k)) for k in key),
                key=lambda d: tuple(sorted(d.items())),
            )
        )
        if key in merged:
            merged[key].update(value)
        else:
            merged[key] = dict(value)
    conditionalSubstitutions = list(reversed(merged.items()))
    del merged

    # Overlay
    #
    # Rank is the bit-set of the index of all contributing layers.
    initMapInit = ((hashdict(), 0),)  # Initializer representing the entire space
    boxMap = OrderedDict(initMapInit)  # Map from Box to Rank
    for i, (currRegion, _) in enumerate(conditionalSubstitutions):
        newMap = OrderedDict(initMapInit)
        currRank = 1 << i
        for box, rank in boxMap.items():
            for currBox in currRegion:
                intersection, remainder = overlayBox(currBox, box)
                if intersection is not None:
                    intersection = hashdict(intersection)
                    newMap[intersection] = newMap.get(intersection, 0) | rank | currRank
                if remainder is not None:
                    remainder = hashdict(remainder)
                    newMap[remainder] = newMap.get(remainder, 0) | rank
        boxMap = newMap

    # Generate output
    items = []
    for box, rank in sorted(
        boxMap.items(), key=(lambda BoxAndRank: -bit_count(BoxAndRank[1]))
    ):
        # Skip any box that doesn't have any substitution.
        if rank == 0:
            continue
        substsList = []
        i = 0
        while rank:
            if rank & 1:
                substsList.append(conditionalSubstitutions[i][1])
            rank >>= 1
            i += 1
        items.append((dict(box), substsList))
    return items


#
# Terminology:
#
# A 'Box' is a dict representing an orthogonal "rectangular" bit of N-dimensional space.
# The keys in the dict are axis tags, the values are (minValue, maxValue) tuples.
# Missing dimensions (keys) are substituted by the default min and max values
# from the corresponding axes.
#


def overlayBox(top, bot):
    """Overlays ``top`` box on top of ``bot`` box.

    Returns two items:

    * Box for intersection of ``top`` and ``bot``, or None if they don't intersect.
    * Box for remainder of ``bot``.  Remainder box might not be exact (since the
      remainder might not be a simple box), but is inclusive of the exact
      remainder.
    """

    # Intersection
    intersection = {}
    intersection.update(top)
    intersection.update(bot)
    for axisTag in set(top) & set(bot):
        min1, max1 = top[axisTag]
        min2, max2 = bot[axisTag]
        minimum = max(min1, min2)
        maximum = min(max1, max2)
        if not minimum < maximum:
            return None, bot  # Do not intersect
        intersection[axisTag] = minimum, maximum

    # Remainder
    #
    # Remainder is empty if bot's each axis range lies within that of intersection.
    #
    # Remainder is shrank if bot's each, except for exactly one, axis range lies
    # within that of intersection, and that one axis, it extrudes out of the
    # intersection only on one side.
    #
    # Bot is returned in full as remainder otherwise, as true remainder is not
    # representable as a single box.

    remainder = dict(bot)
    extruding = False
    fullyInside = True
    for axisTag in top:
        if axisTag in bot:
            continue
        extruding = True
        fullyInside = False
        break
    for axisTag in bot:
        if axisTag not in top:
            continue  # Axis range lies fully within
        min1, max1 = intersection[axisTag]
        min2, max2 = bot[axisTag]
        if min1 <= min2 and max2 <= max1:
            continue  # Axis range lies fully within

        # Bot's range doesn't fully lie within that of top's for this axis.
        # We know they intersect, so it cannot lie fully without either; so they
        # overlap.

        # If we have had an overlapping axis before, remainder is not
        # representable as a box, so return full bottom and go home.
        if extruding:
            return intersection, bot
        extruding = True
        fullyInside = False

        # Otherwise, cut remainder on this axis and continue.
        if min1 <= min2:
            # Right side survives.
            minimum = max(max1, min2)
            maximum = max2
        elif max2 <= max1:
            # Left side survives.
            minimum = min2
            maximum = min(min1, max2)
        else:
            # Remainder leaks out from both sides.  Can't cut either.
            return intersection, bot

        remainder[axisTag] = minimum, maximum

    if fullyInside:
        # bot is fully within intersection.  Remainder is empty.
        return intersection, None

    return intersection, remainder


def cleanupBox(box):
    """Return a sparse copy of `box`, without redundant (default) values.

    >>> cleanupBox({})
    {}
    >>> cleanupBox({'wdth': (0.0, 1.0)})
    {'wdth': (0.0, 1.0)}
    >>> cleanupBox({'wdth': (-1.0, 1.0)})
    {}

    """
    return {tag: limit for tag, limit in box.items() if limit != (-1.0, 1.0)}


#
# Low level implementation
#


def addFeatureVariationsRaw(font, table, conditionalSubstitutions, featureTag="rvrn"):
    """Low level implementation of addFeatureVariations that directly
    models the possibilities of the FeatureVariations table."""

    featureTags = [featureTag] if isinstance(featureTag, str) else sorted(featureTag)
    processLast = "rvrn" not in featureTags or len(featureTags) > 1

    #
    # if a <featureTag> feature is not present:
    #     make empty <featureTag> feature
    #     sort features, get <featureTag> feature index
    #     add <featureTag> feature to all scripts
    # if a <featureTag> feature is present:
    #     reuse <featureTag> feature index
    # make lookups
    # add feature variations
    #
    if table.Version < 0x00010001:
        table.Version = 0x00010001  # allow table.FeatureVariations

    varFeatureIndices = set()

    existingTags = {
        feature.FeatureTag
        for feature in table.FeatureList.FeatureRecord
        if feature.FeatureTag in featureTags
    }

    newTags = set(featureTags) - existingTags
    if newTags:
        varFeatures = []
        for featureTag in sorted(newTags):
            varFeature = buildFeatureRecord(featureTag, [])
            table.FeatureList.FeatureRecord.append(varFeature)
            varFeatures.append(varFeature)
        table.FeatureList.FeatureCount = len(table.FeatureList.FeatureRecord)

        sortFeatureList(table)

        for varFeature in varFeatures:
            varFeatureIndex = table.FeatureList.FeatureRecord.index(varFeature)

            for scriptRecord in table.ScriptList.ScriptRecord:
                if scriptRecord.Script.DefaultLangSys is None:
                    # We need to have a default LangSys to attach variations to.
                    langSys = ot.LangSys()
                    langSys.LookupOrder = None
                    langSys.ReqFeatureIndex = 0xFFFF
                    langSys.FeatureIndex = []
                    langSys.FeatureCount = 0
                    scriptRecord.Script.DefaultLangSys = langSys
                langSystems = [lsr.LangSys for lsr in scriptRecord.Script.LangSysRecord]
                for langSys in [scriptRecord.Script.DefaultLangSys] + langSystems:
                    langSys.FeatureIndex.append(varFeatureIndex)
                    langSys.FeatureCount = len(langSys.FeatureIndex)
            varFeatureIndices.add(varFeatureIndex)

    if existingTags:
        # indices may have changed if we inserted new features and sorted feature list
        # so we must do this after the above
        varFeatureIndices.update(
            index
            for index, feature in enumerate(table.FeatureList.FeatureRecord)
            if feature.FeatureTag in existingTags
        )

    axisIndices = {
        axis.axisTag: axisIndex for axisIndex, axis in enumerate(font["fvar"].axes)
    }

    hasFeatureVariations = (
        hasattr(table, "FeatureVariations") and table.FeatureVariations is not None
    )

    featureVariationRecords = []
    for conditionSet, lookupIndices in conditionalSubstitutions:
        conditionTable = []
        for axisTag, (minValue, maxValue) in sorted(conditionSet.items()):
            if minValue > maxValue:
                raise VarLibValidationError(
                    "A condition set has a minimum value above the maximum value."
                )
            ct = buildConditionTable(axisIndices[axisTag], minValue, maxValue)
            conditionTable.append(ct)
        records = []
        for varFeatureIndex in sorted(varFeatureIndices):
            existingLookupIndices = table.FeatureList.FeatureRecord[
                varFeatureIndex
            ].Feature.LookupListIndex
            combinedLookupIndices = (
                existingLookupIndices + lookupIndices
                if processLast
                else lookupIndices + existingLookupIndices
            )

            records.append(
                buildFeatureTableSubstitutionRecord(
                    varFeatureIndex, combinedLookupIndices
                )
            )
        if hasFeatureVariations and (
            fvr := findFeatureVariationRecord(table.FeatureVariations, conditionTable)
        ):
            fvr.FeatureTableSubstitution.SubstitutionRecord.extend(records)
            fvr.FeatureTableSubstitution.SubstitutionCount = len(
                fvr.FeatureTableSubstitution.SubstitutionRecord
            )
        else:
            featureVariationRecords.append(
                buildFeatureVariationRecord(conditionTable, records)
            )

    if hasFeatureVariations:
        if table.FeatureVariations.Version != 0x00010000:
            raise VarLibError(
                "Unsupported FeatureVariations table version: "
                f"0x{table.FeatureVariations.Version:08x} (expected 0x00010000)."
            )
        table.FeatureVariations.FeatureVariationRecord.extend(featureVariationRecords)
        table.FeatureVariations.FeatureVariationCount = len(
            table.FeatureVariations.FeatureVariationRecord
        )
    else:
        table.FeatureVariations = buildFeatureVariations(featureVariationRecords)


#
# Building GSUB/FeatureVariations internals
#


def buildGSUB():
    """Build a GSUB table from scratch."""
    fontTable = newTable("GSUB")
    gsub = fontTable.table = ot.GSUB()
    gsub.Version = 0x00010001  # allow gsub.FeatureVariations

    gsub.ScriptList = ot.ScriptList()
    gsub.ScriptList.ScriptRecord = []
    gsub.FeatureList = ot.FeatureList()
    gsub.FeatureList.FeatureRecord = []
    gsub.LookupList = ot.LookupList()
    gsub.LookupList.Lookup = []

    srec = ot.ScriptRecord()
    srec.ScriptTag = "DFLT"
    srec.Script = ot.Script()
    srec.Script.DefaultLangSys = None
    srec.Script.LangSysRecord = []
    srec.Script.LangSysCount = 0

    langrec = ot.LangSysRecord()
    langrec.LangSys = ot.LangSys()
    langrec.LangSys.ReqFeatureIndex = 0xFFFF
    langrec.LangSys.FeatureIndex = []
    srec.Script.DefaultLangSys = langrec.LangSys

    gsub.ScriptList.ScriptRecord.append(srec)
    gsub.ScriptList.ScriptCount = 1
    gsub.FeatureVariations = None

    return fontTable


def makeSubstitutionsHashable(conditionalSubstitutions):
    """Turn all the substitution dictionaries in sorted tuples of tuples so
    they are hashable, to detect duplicates so we don't write out redundant
    data."""
    allSubstitutions = set()
    condSubst = []
    for conditionSet, substitutionMaps in conditionalSubstitutions:
        substitutions = []
        for substitutionMap in substitutionMaps:
            subst = tuple(sorted(substitutionMap.items()))
            substitutions.append(subst)
            allSubstitutions.add(subst)
        condSubst.append((conditionSet, substitutions))
    return condSubst, sorted(allSubstitutions)


class ShifterVisitor(TTVisitor):
    def __init__(self, shift):
        self.shift = shift


@ShifterVisitor.register_attr(ot.Feature, "LookupListIndex")  # GSUB/GPOS
def visit(visitor, obj, attr, value):
    shift = visitor.shift
    value = [l + shift for l in value]
    setattr(obj, attr, value)


@ShifterVisitor.register_attr(
    (ot.SubstLookupRecord, ot.PosLookupRecord), "LookupListIndex"
)
def visit(visitor, obj, attr, value):
    setattr(obj, attr, visitor.shift + value)


def buildSubstitutionLookups(gsub, allSubstitutions, processLast=False):
    """Build the lookups for the glyph substitutions, return a dict mapping
    the substitution to lookup indices."""

    # Insert lookups at the beginning of the lookup vector
    # https://github.com/googlefonts/fontmake/issues/950

    firstIndex = len(gsub.LookupList.Lookup) if processLast else 0
    lookupMap = {}
    for i, substitutionMap in enumerate(allSubstitutions):
        lookupMap[substitutionMap] = firstIndex + i

    if not processLast:
        # Shift all lookup indices in gsub by len(allSubstitutions)
        shift = len(allSubstitutions)
        visitor = ShifterVisitor(shift)
        visitor.visit(gsub.FeatureList.FeatureRecord)
        visitor.visit(gsub.LookupList.Lookup)

    for i, subst in enumerate(allSubstitutions):
        substMap = dict(subst)
        lookup = buildLookup([buildSingleSubstSubtable(substMap)])
        if processLast:
            gsub.LookupList.Lookup.append(lookup)
        else:
            gsub.LookupList.Lookup.insert(i, lookup)
        assert gsub.LookupList.Lookup[lookupMap[subst]] is lookup
    gsub.LookupList.LookupCount = len(gsub.LookupList.Lookup)
    return lookupMap


def buildFeatureVariations(featureVariationRecords):
    """Build the FeatureVariations subtable."""
    fv = ot.FeatureVariations()
    fv.Version = 0x00010000
    fv.FeatureVariationRecord = featureVariationRecords
    fv.FeatureVariationCount = len(featureVariationRecords)
    return fv


def buildFeatureRecord(featureTag, lookupListIndices):
    """Build a FeatureRecord."""
    fr = ot.FeatureRecord()
    fr.FeatureTag = featureTag
    fr.Feature = ot.Feature()
    fr.Feature.LookupListIndex = lookupListIndices
    fr.Feature.populateDefaults()
    return fr


def buildFeatureVariationRecord(conditionTable, substitutionRecords):
    """Build a FeatureVariationRecord."""
    fvr = ot.FeatureVariationRecord()
    if len(conditionTable) != 0:
        fvr.ConditionSet = ot.ConditionSet()
        fvr.ConditionSet.ConditionTable = conditionTable
        fvr.ConditionSet.ConditionCount = len(conditionTable)
    else:
        fvr.ConditionSet = None
    fvr.FeatureTableSubstitution = ot.FeatureTableSubstitution()
    fvr.FeatureTableSubstitution.Version = 0x00010000
    fvr.FeatureTableSubstitution.SubstitutionRecord = substitutionRecords
    fvr.FeatureTableSubstitution.SubstitutionCount = len(substitutionRecords)
    return fvr


def buildFeatureTableSubstitutionRecord(featureIndex, lookupListIndices):
    """Build a FeatureTableSubstitutionRecord."""
    ftsr = ot.FeatureTableSubstitutionRecord()
    ftsr.FeatureIndex = featureIndex
    ftsr.Feature = ot.Feature()
    ftsr.Feature.LookupListIndex = lookupListIndices
    ftsr.Feature.LookupCount = len(lookupListIndices)
    return ftsr


def buildConditionTable(axisIndex, filterRangeMinValue, filterRangeMaxValue):
    """Build a ConditionTable."""
    ct = ot.ConditionTable()
    ct.Format = 1
    ct.AxisIndex = axisIndex
    ct.FilterRangeMinValue = filterRangeMinValue
    ct.FilterRangeMaxValue = filterRangeMaxValue
    return ct


def findFeatureVariationRecord(featureVariations, conditionTable):
    """Find a FeatureVariationRecord that has the same conditionTable."""
    if featureVariations.Version != 0x00010000:
        raise VarLibError(
            "Unsupported FeatureVariations table version: "
            f"0x{featureVariations.Version:08x} (expected 0x00010000)."
        )

    for fvr in featureVariations.FeatureVariationRecord:
        if conditionTable == fvr.ConditionSet.ConditionTable:
            return fvr

    return None


def sortFeatureList(table):
    """Sort the feature list by feature tag, and remap the feature indices
    elsewhere. This is needed after the feature list has been modified.
    """
    # decorate, sort, undecorate, because we need to make an index remapping table
    tagIndexFea = [
        (fea.FeatureTag, index, fea)
        for index, fea in enumerate(table.FeatureList.FeatureRecord)
    ]
    tagIndexFea.sort()
    table.FeatureList.FeatureRecord = [fea for tag, index, fea in tagIndexFea]
    featureRemap = dict(
        zip([index for tag, index, fea in tagIndexFea], range(len(tagIndexFea)))
    )

    # Remap the feature indices
    remapFeatures(table, featureRemap)


def remapFeatures(table, featureRemap):
    """Go through the scripts list, and remap feature indices."""
    for scriptIndex, script in enumerate(table.ScriptList.ScriptRecord):
        defaultLangSys = script.Script.DefaultLangSys
        if defaultLangSys is not None:
            _remapLangSys(defaultLangSys, featureRemap)
        for langSysRecordIndex, langSysRec in enumerate(script.Script.LangSysRecord):
            langSys = langSysRec.LangSys
            _remapLangSys(langSys, featureRemap)

    if hasattr(table, "FeatureVariations") and table.FeatureVariations is not None:
        for fvr in table.FeatureVariations.FeatureVariationRecord:
            for ftsr in fvr.FeatureTableSubstitution.SubstitutionRecord:
                ftsr.FeatureIndex = featureRemap[ftsr.FeatureIndex]


def _remapLangSys(langSys, featureRemap):
    if langSys.ReqFeatureIndex != 0xFFFF:
        langSys.ReqFeatureIndex = featureRemap[langSys.ReqFeatureIndex]
    langSys.FeatureIndex = [featureRemap[index] for index in langSys.FeatureIndex]


if __name__ == "__main__":
    import doctest, sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""
Tool to find wrong contour order between different masters, and
other interpolatability (or lack thereof) issues.

Call as:
$ fonttools varLib.interpolatable font1 font2 ...
"""

from .interpolatableHelpers import *
from .interpolatableTestContourOrder import test_contour_order
from .interpolatableTestStartingPoint import test_starting_point
from fontTools.pens.recordingPen import (
    RecordingPen,
    DecomposingRecordingPen,
    lerpRecordings,
)
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.statisticsPen import StatisticsPen, StatisticsControlPen
from fontTools.pens.momentsPen import OpenContourError
from fontTools.varLib.models import piecewiseLinearMap, normalizeLocation
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.transform import Transform
from collections import defaultdict
from types import SimpleNamespace
from functools import wraps
from pprint import pformat
from math import sqrt, atan2, pi
import logging
import os

log = logging.getLogger("fontTools.varLib.interpolatable")

DEFAULT_TOLERANCE = 0.95
DEFAULT_KINKINESS = 0.5
DEFAULT_KINKINESS_LENGTH = 0.002  # ratio of UPEM
DEFAULT_UPEM = 1000


class Glyph:
    ITEMS = (
        "recordings",
        "greenStats",
        "controlStats",
        "greenVectors",
        "controlVectors",
        "nodeTypes",
        "isomorphisms",
        "points",
        "openContours",
    )

    def __init__(self, glyphname, glyphset):
        self.name = glyphname
        for item in self.ITEMS:
            setattr(self, item, [])
        self._populate(glyphset)

    def _fill_in(self, ix):
        for item in self.ITEMS:
            if len(getattr(self, item)) == ix:
                getattr(self, item).append(None)

    def _populate(self, glyphset):
        glyph = glyphset[self.name]
        self.doesnt_exist = glyph is None
        if self.doesnt_exist:
            return

        perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
        try:
            glyph.draw(perContourPen, outputImpliedClosingLine=True)
        except TypeError:
            glyph.draw(perContourPen)
        self.recordings = perContourPen.value
        del perContourPen

        for ix, contour in enumerate(self.recordings):
            nodeTypes = [op for op, arg in contour.value]
            self.nodeTypes.append(nodeTypes)

            greenStats = StatisticsPen(glyphset=glyphset)
            controlStats = StatisticsControlPen(glyphset=glyphset)
            try:
                contour.replay(greenStats)
                contour.replay(controlStats)
                self.openContours.append(False)
            except OpenContourError as e:
                self.openContours.append(True)
                self._fill_in(ix)
                continue
            self.greenStats.append(greenStats)
            self.controlStats.append(controlStats)
            self.greenVectors.append(contour_vector_from_stats(greenStats))
            self.controlVectors.append(contour_vector_from_stats(controlStats))

            # Check starting point
            if nodeTypes[0] == "addComponent":
                self._fill_in(ix)
                continue

            assert nodeTypes[0] == "moveTo"
            assert nodeTypes[-1] in ("closePath", "endPath")
            points = SimpleRecordingPointPen()
            converter = SegmentToPointPen(points, False)
            contour.replay(converter)
            # points.value is a list of pt,bool where bool is true if on-curve and false if off-curve;
            # now check all rotations and mirror-rotations of the contour and build list of isomorphic
            # possible starting points.
            self.points.append(points.value)

            isomorphisms = []
            self.isomorphisms.append(isomorphisms)

            # Add rotations
            add_isomorphisms(points.value, isomorphisms, False)
            # Add mirrored rotations
            add_isomorphisms(points.value, isomorphisms, True)

    def draw(self, pen, contour_idx=None):
        if contour_idx is None:
            for contour in self.recordings:
                contour.draw(pen)
        else:
            self.recordings[contour_idx].draw(pen)


def test_gen(
    glyphsets,
    glyphs=None,
    names=None,
    ignore_missing=False,
    *,
    locations=None,
    tolerance=DEFAULT_TOLERANCE,
    kinkiness=DEFAULT_KINKINESS,
    upem=DEFAULT_UPEM,
    show_all=False,
    discrete_axes=[],
):
    if tolerance >= 10:
        tolerance *= 0.01
    assert 0 <= tolerance <= 1
    if kinkiness >= 10:
        kinkiness *= 0.01
    assert 0 <= kinkiness

    names = names or [repr(g) for g in glyphsets]

    if glyphs is None:
        # `glyphs = glyphsets[0].keys()` is faster, certainly, but doesn't allow for sparse TTFs/OTFs given out of order
        # ... risks the sparse master being the first one, and only processing a subset of the glyphs
        glyphs = {g for glyphset in glyphsets for g in glyphset.keys()}

    parents, order = find_parents_and_order(
        glyphsets, locations, discrete_axes=discrete_axes
    )

    def grand_parent(i, glyphname):
        if i is None:
            return None
        i = parents[i]
        if i is None:
            return None
        while parents[i] is not None and glyphsets[i][glyphname] is None:
            i = parents[i]
        return i

    for glyph_name in glyphs:
        log.info("Testing glyph %s", glyph_name)
        allGlyphs = [Glyph(glyph_name, glyphset) for glyphset in glyphsets]
        if len([1 for glyph in allGlyphs if not glyph.doesnt_exist]) <= 1:
            continue
        for master_idx, (glyph, glyphset, name) in enumerate(
            zip(allGlyphs, glyphsets, names)
        ):
            if glyph.doesnt_exist:
                if not ignore_missing:
                    yield (
                        glyph_name,
                        {
                            "type": InterpolatableProblem.MISSING,
                            "master": name,
                            "master_idx": master_idx,
                        },
                    )
                continue

            has_open = False
            for ix, open in enumerate(glyph.openContours):
                if not open:
                    continue
                has_open = True
                yield (
                    glyph_name,
                    {
                        "type": InterpolatableProblem.OPEN_PATH,
                        "master": name,
                        "master_idx": master_idx,
                        "contour": ix,
                    },
                )
            if has_open:
                continue

        matchings = [None] * len(glyphsets)

        for m1idx in order:
            glyph1 = allGlyphs[m1idx]
            if glyph1 is None or not glyph1.nodeTypes:
                continue
            m0idx = grand_parent(m1idx, glyph_name)
            if m0idx is None:
                continue
            glyph0 = allGlyphs[m0idx]
            if glyph0 is None or not glyph0.nodeTypes:
                continue

            #
            # Basic compatibility checks
            #

            m0 = glyph0.nodeTypes
            m1 = glyph1.nodeTypes
            if len(m0) != len(m1):
                yield (
                    glyph_name,
                    {
                        "type": InterpolatableProblem.PATH_COUNT,
                        "master_1": names[m0idx],
                        "master_2": names[m1idx],
                        "master_1_idx": m0idx,
                        "master_2_idx": m1idx,
                        "value_1": len(m0),
                        "value_2": len(m1),
                    },
                )
                continue

            if m0 != m1:
                for pathIx, (nodes1, nodes2) in enumerate(zip(m0, m1)):
                    if nodes1 == nodes2:
                        continue
                    if len(nodes1) != len(nodes2):
                        yield (
                            glyph_name,
                            {
                                "type": InterpolatableProblem.NODE_COUNT,
                                "path": pathIx,
                                "master_1": names[m0idx],
                                "master_2": names[m1idx],
                                "master_1_idx": m0idx,
                                "master_2_idx": m1idx,
                                "value_1": len(nodes1),
                                "value_2": len(nodes2),
                            },
                        )
                        continue
                    for nodeIx, (n1, n2) in enumerate(zip(nodes1, nodes2)):
                        if n1 != n2:
                            yield (
                                glyph_name,
                                {
                                    "type": InterpolatableProblem.NODE_INCOMPATIBILITY,
                                    "path": pathIx,
                                    "node": nodeIx,
                                    "master_1": names[m0idx],
                                    "master_2": names[m1idx],
                                    "master_1_idx": m0idx,
                                    "master_2_idx": m1idx,
                                    "value_1": n1,
                                    "value_2": n2,
                                },
                            )
                            continue

            #
            # InterpolatableProblem.CONTOUR_ORDER check
            #

            this_tolerance, matching = test_contour_order(glyph0, glyph1)
            if this_tolerance < tolerance:
                yield (
                    glyph_name,
                    {
                        "type": InterpolatableProblem.CONTOUR_ORDER,
                        "master_1": names[m0idx],
                        "master_2": names[m1idx],
                        "master_1_idx": m0idx,
                        "master_2_idx": m1idx,
                        "value_1": list(range(len(matching))),
                        "value_2": matching,
                        "tolerance": this_tolerance,
                    },
                )
                matchings[m1idx] = matching

            #
            # wrong-start-point / weight check
            #

            m0Isomorphisms = glyph0.isomorphisms
            m1Isomorphisms = glyph1.isomorphisms
            m0Vectors = glyph0.greenVectors
            m1Vectors = glyph1.greenVectors
            recording0 = glyph0.recordings
            recording1 = glyph1.recordings

            # If contour-order is wrong, adjust it
            matching = matchings[m1idx]
            if (
                matching is not None and m1Isomorphisms
            ):  # m1 is empty for composite glyphs
                m1Isomorphisms = [m1Isomorphisms[i] for i in matching]
                m1Vectors = [m1Vectors[i] for i in matching]
                recording1 = [recording1[i] for i in matching]

            midRecording = []
            for c0, c1 in zip(recording0, recording1):
                try:
                    r = RecordingPen()
                    r.value = list(lerpRecordings(c0.value, c1.value))
                    midRecording.append(r)
                except ValueError:
                    # Mismatch because of the reordering above
                    midRecording.append(None)

            for ix, (contour0, contour1) in enumerate(
                zip(m0Isomorphisms, m1Isomorphisms)
            ):
                if (
                    contour0 is None
                    or contour1 is None
                    or len(contour0) == 0
                    or len(contour0) != len(contour1)
                ):
                    # We already reported this; or nothing to do; or not compatible
                    # after reordering above.
                    continue

                this_tolerance, proposed_point, reverse = test_starting_point(
                    glyph0, glyph1, ix, tolerance, matching
                )

                if this_tolerance < tolerance:
                    yield (
                        glyph_name,
                        {
                            "type": InterpolatableProblem.WRONG_START_POINT,
                            "contour": ix,
                            "master_1": names[m0idx],
                            "master_2": names[m1idx],
                            "master_1_idx": m0idx,
                            "master_2_idx": m1idx,
                            "value_1": 0,
                            "value_2": proposed_point,
                            "reversed": reverse,
                            "tolerance": this_tolerance,
                        },
                    )

                # Weight check.
                #
                # If contour could be mid-interpolated, and the two
                # contours have the same area sign, proceeed.
                #
                # The sign difference can happen if it's a weirdo
                # self-intersecting contour; ignore it.
                contour = midRecording[ix]

                if contour and (m0Vectors[ix][0] < 0) == (m1Vectors[ix][0] < 0):
                    midStats = StatisticsPen(glyphset=None)
                    contour.replay(midStats)

                    midVector = contour_vector_from_stats(midStats)

                    m0Vec = m0Vectors[ix]
                    m1Vec = m1Vectors[ix]
                    size0 = m0Vec[0] * m0Vec[0]
                    size1 = m1Vec[0] * m1Vec[0]
                    midSize = midVector[0] * midVector[0]

                    for overweight, problem_type in enumerate(
                        (
                            InterpolatableProblem.UNDERWEIGHT,
                            InterpolatableProblem.OVERWEIGHT,
                        )
                    ):
                        if overweight:
                            expectedSize = max(size0, size1)
                            continue  # disabled
                        else:
                            expectedSize = sqrt(size0 * size1)

                        log.debug(
                            "%s: actual size %g; threshold size %g, master sizes: %g, %g",
                            problem_type,
                            midSize,
                            expectedSize,
                            size0,
                            size1,
                        )

                        if (
                            not overweight and expectedSize * tolerance > midSize + 1e-5
                        ) or (overweight and 1e-5 + expectedSize / tolerance < midSize):
                            try:
                                if overweight:
                                    this_tolerance = expectedSize / midSize
                                else:
                                    this_tolerance = midSize / expectedSize
                            except ZeroDivisionError:
                                this_tolerance = 0
                            log.debug("tolerance %g", this_tolerance)
                            yield (
                                glyph_name,
                                {
                                    "type": problem_type,
                                    "contour": ix,
                                    "master_1": names[m0idx],
                                    "master_2": names[m1idx],
                                    "master_1_idx": m0idx,
                                    "master_2_idx": m1idx,
                                    "tolerance": this_tolerance,
                                },
                            )

            #
            # "kink" detector
            #
            m0 = glyph0.points
            m1 = glyph1.points

            # If contour-order is wrong, adjust it
            if matchings[m1idx] is not None and m1:  # m1 is empty for composite glyphs
                m1 = [m1[i] for i in matchings[m1idx]]

            t = 0.1  # ~sin(radian(6)) for tolerance 0.95
            deviation_threshold = (
                upem * DEFAULT_KINKINESS_LENGTH * DEFAULT_KINKINESS / kinkiness
            )

            for ix, (contour0, contour1) in enumerate(zip(m0, m1)):
                if (
                    contour0 is None
                    or contour1 is None
                    or len(contour0) == 0
                    or len(contour0) != len(contour1)
                ):
                    # We already reported this; or nothing to do; or not compatible
                    # after reordering above.
                    continue

                # Walk the contour, keeping track of three consecutive points, with
                # middle one being an on-curve. If the three are co-linear then
                # check for kinky-ness.
                for i in range(len(contour0)):
                    pt0 = contour0[i]
                    pt1 = contour1[i]
                    if not pt0[1] or not pt1[1]:
                        # Skip off-curves
                        continue
                    pt0_prev = contour0[i - 1]
                    pt1_prev = contour1[i - 1]
                    pt0_next = contour0[(i + 1) % len(contour0)]
                    pt1_next = contour1[(i + 1) % len(contour1)]

                    if pt0_prev[1] and pt1_prev[1]:
                        # At least one off-curve is required
                        continue
                    if pt0_next[1] and pt1_next[1]:
                        # At least one off-curve is required
                        continue

                    pt0 = complex(*pt0[0])
                    pt1 = complex(*pt1[0])
                    pt0_prev = complex(*pt0_prev[0])
                    pt1_prev = complex(*pt1_prev[0])
                    pt0_next = complex(*pt0_next[0])
                    pt1_next = complex(*pt1_next[0])

                    # We have three consecutive points. Check whether
                    # they are colinear.
                    d0_prev = pt0 - pt0_prev
                    d0_next = pt0_next - pt0
                    d1_prev = pt1 - pt1_prev
                    d1_next = pt1_next - pt1

                    sin0 = d0_prev.real * d0_next.imag - d0_prev.imag * d0_next.real
                    sin1 = d1_prev.real * d1_next.imag - d1_prev.imag * d1_next.real
                    try:
                        sin0 /= abs(d0_prev) * abs(d0_next)
                        sin1 /= abs(d1_prev) * abs(d1_next)
                    except ZeroDivisionError:
                        continue

                    if abs(sin0) > t or abs(sin1) > t:
                        # Not colinear / not smooth.
                        continue

                    # Check the mid-point is actually, well, in the middle.
                    dot0 = d0_prev.real * d0_next.real + d0_prev.imag * d0_next.imag
                    dot1 = d1_prev.real * d1_next.real + d1_prev.imag * d1_next.imag
                    if dot0 < 0 or dot1 < 0:
                        # Sharp corner.
                        continue

                    # Fine, if handle ratios are similar...
                    r0 = abs(d0_prev) / (abs(d0_prev) + abs(d0_next))
                    r1 = abs(d1_prev) / (abs(d1_prev) + abs(d1_next))
                    r_diff = abs(r0 - r1)
                    if abs(r_diff) < t:
                        # Smooth enough.
                        continue

                    mid = (pt0 + pt1) / 2
                    mid_prev = (pt0_prev + pt1_prev) / 2
                    mid_next = (pt0_next + pt1_next) / 2

                    mid_d0 = mid - mid_prev
                    mid_d1 = mid_next - mid

                    sin_mid = mid_d0.real * mid_d1.imag - mid_d0.imag * mid_d1.real
                    try:
                        sin_mid /= abs(mid_d0) * abs(mid_d1)
                    except ZeroDivisionError:
                        continue

                    # ...or if the angles are similar.
                    if abs(sin_mid) * (tolerance * kinkiness) <= t:
                        # Smooth enough.
                        continue

                    # How visible is the kink?

                    cross = sin_mid * abs(mid_d0) * abs(mid_d1)
                    arc_len = abs(mid_d0 + mid_d1)
                    deviation = abs(cross / arc_len)
                    if deviation < deviation_threshold:
                        continue
                    deviation_ratio = deviation / arc_len
                    if deviation_ratio > t:
                        continue

                    this_tolerance = t / (abs(sin_mid) * kinkiness)

                    log.debug(
                        "kink: deviation %g; deviation_ratio %g; sin_mid %g; r_diff %g",
                        deviation,
                        deviation_ratio,
                        sin_mid,
                        r_diff,
                    )
                    log.debug("tolerance %g", this_tolerance)
                    yield (
                        glyph_name,
                        {
                            "type": InterpolatableProblem.KINK,
                            "contour": ix,
                            "master_1": names[m0idx],
                            "master_2": names[m1idx],
                            "master_1_idx": m0idx,
                            "master_2_idx": m1idx,
                            "value": i,
                            "tolerance": this_tolerance,
                        },
                    )

            #
            # --show-all
            #

            if show_all:
                yield (
                    glyph_name,
                    {
                        "type": InterpolatableProblem.NOTHING,
                        "master_1": names[m0idx],
                        "master_2": names[m1idx],
                        "master_1_idx": m0idx,
                        "master_2_idx": m1idx,
                    },
                )


@wraps(test_gen)
def test(*args, **kwargs):
    problems = defaultdict(list)
    for glyphname, problem in test_gen(*args, **kwargs):
        problems[glyphname].append(problem)
    return problems


def recursivelyAddGlyph(glyphname, glyphset, ttGlyphSet, glyf):
    if glyphname in glyphset:
        return
    glyphset[glyphname] = ttGlyphSet[glyphname]

    for component in getattr(glyf[glyphname], "components", []):
        recursivelyAddGlyph(component.glyphName, glyphset, ttGlyphSet, glyf)


def ensure_parent_dir(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return path


def main(args=None):
    """Test for interpolatability issues between fonts"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        "fonttools varLib.interpolatable",
        description=main.__doc__,
    )
    parser.add_argument(
        "--glyphs",
        action="store",
        help="Space-separate name of glyphs to check",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all glyph pairs, even if no problems are found",
    )
    parser.add_argument(
        "--tolerance",
        action="store",
        type=float,
        help="Error tolerance. Between 0 and 1. Default %s" % DEFAULT_TOLERANCE,
    )
    parser.add_argument(
        "--kinkiness",
        action="store",
        type=float,
        help="How aggressively report kinks. Default %s" % DEFAULT_KINKINESS,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format",
    )
    parser.add_argument(
        "--pdf",
        action="store",
        help="Output report in PDF format",
    )
    parser.add_argument(
        "--ps",
        action="store",
        help="Output report in PostScript format",
    )
    parser.add_argument(
        "--html",
        action="store",
        help="Output report in HTML format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only exit with code 1 or 0, no output",
    )
    parser.add_argument(
        "--output",
        action="store",
        help="Output file for the problem report; Default: stdout",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Will not report glyphs missing from sparse masters as errors",
    )
    parser.add_argument(
        "inputs",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Input a single variable font / DesignSpace / Glyphs file, or multiple TTF/UFO files",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        type=str,
        action="append",
        help="Name of the master to use in the report. If not provided, all are used.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Run verbosely.")
    parser.add_argument("--debug", action="store_true", help="Run with debug output.")

    args = parser.parse_args(args)

    from fontTools import configLogger

    configLogger(level=("INFO" if args.verbose else "WARNING"))
    if args.debug:
        configLogger(level="DEBUG")

    glyphs = args.glyphs.split() if args.glyphs else None

    from os.path import basename

    fonts = []
    names = []
    locations = []
    discrete_axes = set()
    upem = DEFAULT_UPEM

    original_args_inputs = tuple(args.inputs)

    if len(args.inputs) == 1:
        designspace = None
        if args.inputs[0].endswith(".designspace"):
            from fontTools.designspaceLib import DesignSpaceDocument

            designspace = DesignSpaceDocument.fromfile(args.inputs[0])
            args.inputs = [master.path for master in designspace.sources]
            locations = [master.location for master in designspace.sources]
            discrete_axes = {
                a.name for a in designspace.axes if not hasattr(a, "minimum")
            }
            axis_triples = {
                a.name: (a.minimum, a.default, a.maximum)
                for a in designspace.axes
                if a.name not in discrete_axes
            }
            axis_mappings = {a.name: a.map for a in designspace.axes}
            axis_triples = {
                k: tuple(piecewiseLinearMap(v, dict(axis_mappings[k])) for v in vv)
                for k, vv in axis_triples.items()
            }

        elif args.inputs[0].endswith((".glyphs", ".glyphspackage")):
            from glyphsLib import GSFont, to_designspace

            gsfont = GSFont(args.inputs[0])
            upem = gsfont.upm
            designspace = to_designspace(gsfont)
            fonts = [source.font for source in designspace.sources]
            names = ["%s-%s" % (f.info.familyName, f.info.styleName) for f in fonts]
            args.inputs = []
            locations = [master.location for master in designspace.sources]
            axis_triples = {
                a.name: (a.minimum, a.default, a.maximum) for a in designspace.axes
            }
            axis_mappings = {a.name: a.map for a in designspace.axes}
            axis_triples = {
                k: tuple(piecewiseLinearMap(v, dict(axis_mappings[k])) for v in vv)
                for k, vv in axis_triples.items()
            }

        elif args.inputs[0].endswith(".ttf") or args.inputs[0].endswith(".otf"):
            from fontTools.ttLib import TTFont

            # Is variable font?

            font = TTFont(args.inputs[0])
            upem = font["head"].unitsPerEm

            fvar = font["fvar"]
            axisMapping = {}
            for axis in fvar.axes:
                axisMapping[axis.axisTag] = {
                    -1: axis.minValue,
                    0: axis.defaultValue,
                    1: axis.maxValue,
                }
            normalized = False
            if "avar" in font:
                avar = font["avar"]
                if getattr(avar.table, "VarStore", None):
                    axisMapping = {tag: {-1: -1, 0: 0, 1: 1} for tag in axisMapping}
                    normalized = True
                else:
                    for axisTag, segments in avar.segments.items():
                        fvarMapping = axisMapping[axisTag].copy()
                        for location, value in segments.items():
                            axisMapping[axisTag][value] = piecewiseLinearMap(
                                location, fvarMapping
                            )

            # Gather all glyphs at their "master" locations
            ttGlyphSets = {}
            glyphsets = defaultdict(dict)

            if "gvar" in font:
                gvar = font["gvar"]
                glyf = font["glyf"]

                if glyphs is None:
                    glyphs = sorted(gvar.variations.keys())
                for glyphname in glyphs:
                    for var in gvar.variations[glyphname]:
                        locDict = {}
                        loc = []
                        for tag, val in sorted(var.axes.items()):
                            locDict[tag] = val[1]
                            loc.append((tag, val[1]))

                        locTuple = tuple(loc)
                        if locTuple not in ttGlyphSets:
                            ttGlyphSets[locTuple] = font.getGlyphSet(
                                location=locDict, normalized=True, recalcBounds=False
                            )

                        recursivelyAddGlyph(
                            glyphname, glyphsets[locTuple], ttGlyphSets[locTuple], glyf
                        )

            elif "CFF2" in font:
                fvarAxes = font["fvar"].axes
                cff2 = font["CFF2"].cff.topDictIndex[0]
                charstrings = cff2.CharStrings

                if glyphs is None:
                    glyphs = sorted(charstrings.keys())
                for glyphname in glyphs:
                    cs = charstrings[glyphname]
                    private = cs.private

                    # Extract vsindex for the glyph
                    vsindices = {getattr(private, "vsindex", 0)}
                    vsindex = getattr(private, "vsindex", 0)
                    last_op = 0
                    # The spec says vsindex can only appear once and must be the first
                    # operator in the charstring, but we support multiple.
                    # https://github.com/harfbuzz/boring-expansion-spec/issues/158
                    for op in cs.program:
                        if op == "blend":
                            vsindices.add(vsindex)
                        elif op == "vsindex":
                            assert isinstance(last_op, int)
                            vsindex = last_op
                        last_op = op

                    if not hasattr(private, "vstore"):
                        continue

                    varStore = private.vstore.otVarStore
                    for vsindex in vsindices:
                        varData = varStore.VarData[vsindex]
                        for regionIndex in varData.VarRegionIndex:
                            region = varStore.VarRegionList.Region[regionIndex]

                            locDict = {}
                            loc = []
                            for axisIndex, axis in enumerate(region.VarRegionAxis):
                                tag = fvarAxes[axisIndex].axisTag
                                val = axis.PeakCoord
                                locDict[tag] = val
                                loc.append((tag, val))

                            locTuple = tuple(loc)
                            if locTuple not in ttGlyphSets:
                                ttGlyphSets[locTuple] = font.getGlyphSet(
                                    location=locDict,
                                    normalized=True,
                                    recalcBounds=False,
                                )

                            glyphset = glyphsets[locTuple]
                            glyphset[glyphname] = ttGlyphSets[locTuple][glyphname]

            names = ["''"]
            fonts = [font.getGlyphSet()]
            locations = [{}]
            axis_triples = {a: (-1, 0, +1) for a in sorted(axisMapping.keys())}
            for locTuple in sorted(glyphsets.keys(), key=lambda v: (len(v), v)):
                name = (
                    "'"
                    + " ".join(
                        "%s=%s"
                        % (
                            k,
                            floatToFixedToStr(
                                piecewiseLinearMap(v, axisMapping[k]), 14
                            ),
                        )
                        for k, v in locTuple
                    )
                    + "'"
                )
                if normalized:
                    name += " (normalized)"
                names.append(name)
                fonts.append(glyphsets[locTuple])
                locations.append(dict(locTuple))

            args.ignore_missing = True
            args.inputs = []

    if not locations:
        locations = [{} for _ in fonts]

    for filename in args.inputs:
        if filename.endswith(".ufo"):
            from fontTools.ufoLib import UFOReader

            font = UFOReader(filename)
            info = SimpleNamespace()
            font.readInfo(info)
            upem = info.unitsPerEm
            fonts.append(font)
        else:
            from fontTools.ttLib import TTFont

            font = TTFont(filename)
            upem = font["head"].unitsPerEm
            fonts.append(font)

        names.append(basename(filename).rsplit(".", 1)[0])

    if len(fonts) < 2:
        log.warning("Font file does not seem to be variable. Nothing to check.")
        return

    glyphsets = []
    for font in fonts:
        if hasattr(font, "getGlyphSet"):
            glyphset = font.getGlyphSet()
        else:
            glyphset = font
        glyphsets.append({k: glyphset[k] for k in glyphset.keys()})

    if args.name:
        accepted_names = set(args.name)
        glyphsets = [
            glyphset
            for name, glyphset in zip(names, glyphsets)
            if name in accepted_names
        ]
        locations = [
            location
            for name, location in zip(names, locations)
            if name in accepted_names
        ]
        names = [name for name in names if name in accepted_names]

    if not glyphs:
        glyphs = sorted(set([gn for glyphset in glyphsets for gn in glyphset.keys()]))

    glyphsSet = set(glyphs)
    for glyphset in glyphsets:
        glyphSetGlyphNames = set(glyphset.keys())
        diff = glyphsSet - glyphSetGlyphNames
        if diff:
            for gn in diff:
                glyphset[gn] = None

    # Normalize locations
    locations = [
        {
            **normalizeLocation(loc, axis_triples),
            **{k: v for k, v in loc.items() if k in discrete_axes},
        }
        for loc in locations
    ]
    tolerance = args.tolerance or DEFAULT_TOLERANCE
    kinkiness = args.kinkiness if args.kinkiness is not None else DEFAULT_KINKINESS

    try:
        log.info("Running on %d glyphsets", len(glyphsets))
        log.info("Locations: %s", pformat(locations))
        problems_gen = test_gen(
            glyphsets,
            glyphs=glyphs,
            names=names,
            locations=locations,
            upem=upem,
            ignore_missing=args.ignore_missing,
            tolerance=tolerance,
            kinkiness=kinkiness,
            show_all=args.show_all,
            discrete_axes=discrete_axes,
        )
        problems = defaultdict(list)

        f = (
            sys.stdout
            if args.output is None
            else open(ensure_parent_dir(args.output), "w")
        )

        if not args.quiet:
            if args.json:
                import json

                for glyphname, problem in problems_gen:
                    problems[glyphname].append(problem)

                print(json.dumps(problems), file=f)
            else:
                last_glyphname = None
                for glyphname, p in problems_gen:
                    problems[glyphname].append(p)

                    if glyphname != last_glyphname:
                        print(f"Glyph {glyphname} was not compatible:", file=f)
                        last_glyphname = glyphname
                        last_master_idxs = None

                    master_idxs = (
                        (p["master_idx"],)
                        if "master_idx" in p
                        else (p["master_1_idx"], p["master_2_idx"])
                    )
                    if master_idxs != last_master_idxs:
                        master_names = (
                            (p["master"],)
                            if "master" in p
                            else (p["master_1"], p["master_2"])
                        )
                        print(f"  Masters: %s:" % ", ".join(master_names), file=f)
                        last_master_idxs = master_idxs

                    if p["type"] == InterpolatableProblem.MISSING:
                        print(
                            "    Glyph was missing in master %s" % p["master"], file=f
                        )
                    elif p["type"] == InterpolatableProblem.OPEN_PATH:
                        print(
                            "    Glyph has an open path in master %s" % p["master"],
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.PATH_COUNT:
                        print(
                            "    Path count differs: %i in %s, %i in %s"
                            % (
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.NODE_COUNT:
                        print(
                            "    Node count differs in path %i: %i in %s, %i in %s"
                            % (
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.NODE_INCOMPATIBILITY:
                        print(
                            "    Node %d incompatible in path %i: %s in %s, %s in %s"
                            % (
                                p["node"],
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.CONTOUR_ORDER:
                        print(
                            "    Contour order differs: %s in %s, %s in %s"
                            % (
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.WRONG_START_POINT:
                        print(
                            "    Contour %d start point differs: %s in %s, %s in %s; reversed: %s"
                            % (
                                p["contour"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                                p["reversed"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.UNDERWEIGHT:
                        print(
                            "    Contour %d interpolation is underweight: %s, %s"
                            % (
                                p["contour"],
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.OVERWEIGHT:
                        print(
                            "    Contour %d interpolation is overweight: %s, %s"
                            % (
                                p["contour"],
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.KINK:
                        print(
                            "    Contour %d has a kink at %s: %s, %s"
                            % (
                                p["contour"],
                                p["value"],
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == InterpolatableProblem.NOTHING:
                        print(
                            "    Showing %s and %s"
                            % (
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
        else:
            for glyphname, problem in problems_gen:
                problems[glyphname].append(problem)

        problems = sort_problems(problems)

        for p in "ps", "pdf":
            arg = getattr(args, p)
            if arg is None:
                continue
            log.info("Writing %s to %s", p.upper(), arg)
            from .interpolatablePlot import InterpolatablePS, InterpolatablePDF

            PlotterClass = InterpolatablePS if p == "ps" else InterpolatablePDF

            with PlotterClass(
                ensure_parent_dir(arg), glyphsets=glyphsets, names=names
            ) as doc:
                doc.add_title_page(
                    original_args_inputs, tolerance=tolerance, kinkiness=kinkiness
                )
                if problems:
                    doc.add_summary(problems)
                doc.add_problems(problems)
                if not problems and not args.quiet:
                    doc.draw_cupcake()
                if problems:
                    doc.add_index()
                    doc.add_table_of_contents()

        if args.html:
            log.info("Writing HTML to %s", args.html)
            from .interpolatablePlot import InterpolatableSVG

            svgs = []
            glyph_starts = {}
            with InterpolatableSVG(svgs, glyphsets=glyphsets, names=names) as svg:
                svg.add_title_page(
                    original_args_inputs,
                    show_tolerance=False,
                    tolerance=tolerance,
                    kinkiness=kinkiness,
                )
                for glyph, glyph_problems in problems.items():
                    glyph_starts[len(svgs)] = glyph
                    svg.add_problems(
                        {glyph: glyph_problems},
                        show_tolerance=False,
                        show_page_number=False,
                    )
                if not problems and not args.quiet:
                    svg.draw_cupcake()

            import base64
            import html

            with open(ensure_parent_dir(args.html), "wb") as f:
                f.write(b"<!DOCTYPE html>\n")
                f.write(
                    b'<html><body align="center" style="font-family: sans-serif; text-color: #222">\n'
                )
                f.write(b"<title>fonttools varLib.interpolatable report</title>\n")
                for i, svg in enumerate(svgs):
                    if i in glyph_starts:
                        f.write(
                            f"<h1>Glyph {html.escape(glyph_starts[i])}</h1>\n".encode(
                                "utf-8"
                            )
                        )
                    f.write("<img src='data:image/svg+xml;base64,".encode("utf-8"))
                    f.write(base64.b64encode(svg))
                    f.write(b"' />\n")
                    f.write(b"<hr>\n")
                f.write(b"</body></html>\n")

    except Exception as e:
        e.args += original_args_inputs
        log.error(e)
        raise

    if problems:
        return problems


if __name__ == "__main__":
    import sys

    problems = main()
    sys.exit(int(bool(problems)))
<|endoftext|>try:
    import cython
except (AttributeError, ImportError):
    # if cython not installed, use mock module with no-op decorators and types
    from fontTools.misc import cython
COMPILED = cython.compiled

from typing import (
    Sequence,
    Tuple,
    Union,
)
from numbers import Integral, Real

_Point = Tuple[Real, Real]
_Delta = Tuple[Real, Real]
_PointSegment = Sequence[_Point]
_DeltaSegment = Sequence[_Delta]
_DeltaOrNone = Union[_Delta, None]
_DeltaOrNoneSegment = Sequence[_DeltaOrNone]
_Endpoints = Sequence[Integral]


MAX_LOOKBACK = 8


@cython.cfunc
@cython.locals(
    j=cython.int,
    n=cython.int,
    x1=cython.double,
    x2=cython.double,
    d1=cython.double,
    d2=cython.double,
    scale=cython.double,
    x=cython.double,
    d=cython.double,
)
def iup_segment(
    coords: _PointSegment, rc1: _Point, rd1: _Delta, rc2: _Point, rd2: _Delta
):  # -> _DeltaSegment:
    """Given two reference coordinates `rc1` & `rc2` and their respective
    delta vectors `rd1` & `rd2`, returns interpolated deltas for the set of
    coordinates `coords`."""

    # rc1 = reference coord 1
    # rd1 = reference delta 1
    out_arrays = [None, None]
    for j in 0, 1:
        out_arrays[j] = out = []
        x1, x2, d1, d2 = rc1[j], rc2[j], rd1[j], rd2[j]

        if x1 == x2:
            n = len(coords)
            if d1 == d2:
                out.extend([d1] * n)
            else:
                out.extend([0] * n)
            continue

        if x1 > x2:
            x1, x2 = x2, x1
            d1, d2 = d2, d1

        # x1 < x2
        scale = (d2 - d1) / (x2 - x1)
        for pair in coords:
            x = pair[j]

            if x <= x1:
                d = d1
            elif x >= x2:
                d = d2
            else:
                # Interpolate
                #
                # NOTE: we assign an explicit intermediate variable here in
                # order to disable a fused mul-add optimization. See:
                #
                # - https://godbolt.org/z/YsP4T3TqK,
                # - https://github.com/fonttools/fonttools/issues/3703
                nudge = (x - x1) * scale
                d = d1 + nudge

            out.append(d)

    return zip(*out_arrays)


def iup_contour(deltas: _DeltaOrNoneSegment, coords: _PointSegment) -> _DeltaSegment:
    """For the contour given in `coords`, interpolate any missing
    delta values in delta vector `deltas`.

    Returns fully filled-out delta vector."""

    assert len(deltas) == len(coords)
    if None not in deltas:
        return deltas

    n = len(deltas)
    # indices of points with explicit deltas
    indices = [i for i, v in enumerate(deltas) if v is not None]
    if not indices:
        # All deltas are None.  Return 0,0 for all.
        return [(0, 0)] * n

    out = []
    it = iter(indices)
    start = next(it)
    if start != 0:
        # Initial segment that wraps around
        i1, i2, ri1, ri2 = 0, start, start, indices[-1]
        out.extend(
            iup_segment(
                coords[i1:i2], coords[ri1], deltas[ri1], coords[ri2], deltas[ri2]
            )
        )
    out.append(deltas[start])
    for end in it:
        if end - start > 1:
            i1, i2, ri1, ri2 = start + 1, end, start, end
            out.extend(
                iup_segment(
                    coords[i1:i2], coords[ri1], deltas[ri1], coords[ri2], deltas[ri2]
                )
            )
        out.append(deltas[end])
        start = end
    if start != n - 1:
        # Final segment that wraps around
        i1, i2, ri1, ri2 = start + 1, n, start, indices[0]
        out.extend(
            iup_segment(
                coords[i1:i2], coords[ri1], deltas[ri1], coords[ri2], deltas[ri2]
            )
        )

    assert len(deltas) == len(out), (len(deltas), len(out))
    return out


def iup_delta(
    deltas: _DeltaOrNoneSegment, coords: _PointSegment, ends: _Endpoints
) -> _DeltaSegment:
    """For the outline given in `coords`, with contour endpoints given
    in sorted increasing order in `ends`, interpolate any missing
    delta values in delta vector `deltas`.

    Returns fully filled-out delta vector."""

    assert sorted(ends) == ends and len(coords) == (ends[-1] + 1 if ends else 0) + 4
    n = len(coords)
    ends = ends + [n - 4, n - 3, n - 2, n - 1]
    out = []
    start = 0
    for end in ends:
        end += 1
        contour = iup_contour(deltas[start:end], coords[start:end])
        out.extend(contour)
        start = end

    return out


# Optimizer


@cython.cfunc
@cython.inline
@cython.locals(
    i=cython.int,
    j=cython.int,
    # tolerance=cython.double, # https://github.com/fonttools/fonttools/issues/3282
    x=cython.double,
    y=cython.double,
    p=cython.double,
    q=cython.double,
)
@cython.returns(int)
def can_iup_in_between(
    deltas: _DeltaSegment,
    coords: _PointSegment,
    i: Integral,
    j: Integral,
    tolerance: Real,
):  # -> bool:
    """Return true if the deltas for points at `i` and `j` (`i < j`) can be
    successfully used to interpolate deltas for points in between them within
    provided error tolerance."""

    assert j - i >= 2
    interp = iup_segment(coords[i + 1 : j], coords[i], deltas[i], coords[j], deltas[j])
    deltas = deltas[i + 1 : j]

    return all(
        abs(complex(x - p, y - q)) <= tolerance
        for (x, y), (p, q) in zip(deltas, interp)
    )


@cython.locals(
    cj=cython.double,
    dj=cython.double,
    lcj=cython.double,
    ldj=cython.double,
    ncj=cython.double,
    ndj=cython.double,
    force=cython.int,
    forced=set,
)
def _iup_contour_bound_forced_set(
    deltas: _DeltaSegment, coords: _PointSegment, tolerance: Real = 0
) -> set:
    """The forced set is a conservative set of points on the contour that must be encoded
    explicitly (ie. cannot be interpolated).  Calculating this set allows for significantly
    speeding up the dynamic-programming, as well as resolve circularity in DP.

    The set is precise; that is, if an index is in the returned set, then there is no way
    that IUP can generate delta for that point, given `coords` and `deltas`.
    """
    assert len(deltas) == len(coords)

    n = len(deltas)
    forced = set()
    # Track "last" and "next" points on the contour as we sweep.
    for i in range(len(deltas) - 1, -1, -1):
        ld, lc = deltas[i - 1], coords[i - 1]
        d, c = deltas[i], coords[i]
        nd, nc = deltas[i - n + 1], coords[i - n + 1]

        for j in (0, 1):  # For X and for Y
            cj = c[j]
            dj = d[j]
            lcj = lc[j]
            ldj = ld[j]
            ncj = nc[j]
            ndj = nd[j]

            if lcj <= ncj:
                c1, c2 = lcj, ncj
                d1, d2 = ldj, ndj
            else:
                c1, c2 = ncj, lcj
                d1, d2 = ndj, ldj

            force = False

            # If the two coordinates are the same, then the interpolation
            # algorithm produces the same delta if both deltas are equal,
            # and zero if they differ.
            #
            # This test has to be before the next one.
            if c1 == c2:
                if abs(d1 - d2) > tolerance and abs(dj) > tolerance:
                    force = True

            # If coordinate for current point is between coordinate of adjacent
            # points on the two sides, but the delta for current point is NOT
            # between delta for those adjacent points (considering tolerance
            # allowance), then there is no way that current point can be IUP-ed.
            # Mark it forced.
            elif c1 <= cj <= c2:  # and c1 != c2
                if not (min(d1, d2) - tolerance <= dj <= max(d1, d2) + tolerance):
                    force = True

            # Otherwise, the delta should either match the closest, or have the
            # same sign as the interpolation of the two deltas.
            else:  # cj < c1 or c2 < cj
                if d1 != d2:
                    if cj < c1:
                        if (
                            abs(dj) > tolerance
                            and abs(dj - d1) > tolerance
                            and ((dj - tolerance < d1) != (d1 < d2))
                        ):
                            force = True
                    else:  # c2 < cj
                        if (
                            abs(dj) > tolerance
                            and abs(dj - d2) > tolerance
                            and ((d2 < dj + tolerance) != (d1 < d2))
                        ):
                            force = True

            if force:
                forced.add(i)
                break

    return forced


@cython.locals(
    i=cython.int,
    j=cython.int,
    best_cost=cython.double,
    best_j=cython.int,
    cost=cython.double,
    forced=set,
    tolerance=cython.double,
)
def _iup_contour_optimize_dp(
    deltas: _DeltaSegment,
    coords: _PointSegment,
    forced=set(),
    tolerance: Real = 0,
    lookback: Integral = None,
):
    """Straightforward Dynamic-Programming.  For each index i, find least-costly encoding of
    points 0 to i where i is explicitly encoded.  We find this by considering all previous
    explicit points j and check whether interpolation can fill points between j and i.

    Note that solution always encodes last point explicitly.  Higher-level is responsible
    for removing that restriction.

    As major speedup, we stop looking further whenever we see a "forced" point."""

    n = len(deltas)
    if lookback is None:
        lookback = n
    lookback = min(lookback, MAX_LOOKBACK)
    costs = {-1: 0}
    chain = {-1: None}
    for i in range(0, n):
        best_cost = costs[i - 1] + 1

        costs[i] = best_cost
        chain[i] = i - 1

        if i - 1 in forced:
            continue

        for j in range(i - 2, max(i - lookback, -2), -1):
            cost = costs[j] + 1

            if cost < best_cost and can_iup_in_between(deltas, coords, j, i, tolerance):
                costs[i] = best_cost = cost
                chain[i] = j

            if j in forced:
                break

    return chain, costs


def _rot_list(l: list, k: int):
    """Rotate list by k items forward.  Ie. item at position 0 will be
    at position k in returned list.  Negative k is allowed."""
    n = len(l)
    k %= n
    if not k:
        return l
    return l[n - k :] + l[: n - k]


def _rot_set(s: set, k: int, n: int):
    k %= n
    if not k:
        return s
    return {(v + k) % n for v in s}


def iup_contour_optimize(
    deltas: _DeltaSegment, coords: _PointSegment, tolerance: Real = 0.0
) -> _DeltaOrNoneSegment:
    """For contour with coordinates `coords`, optimize a set of delta
    values `deltas` within error `tolerance`.

    Returns delta vector that has most number of None items instead of
    the input delta.
    """

    n = len(deltas)

    # Get the easy cases out of the way:

    # If all are within tolerance distance of 0, encode nothing:
    if all(abs(complex(*p)) <= tolerance for p in deltas):
        return [None] * n

    # If there's exactly one point, return it:
    if n == 1:
        return deltas

    # If all deltas are exactly the same, return just one (the first one):
    d0 = deltas[0]
    if all(d0 == d for d in deltas):
        return [d0] + [None] * (n - 1)

    # Else, solve the general problem using Dynamic Programming.

    forced = _iup_contour_bound_forced_set(deltas, coords, tolerance)
    # The _iup_contour_optimize_dp() routine returns the optimal encoding
    # solution given the constraint that the last point is always encoded.
    # To remove this constraint, we use two different methods, depending on
    # whether forced set is non-empty or not:

    # Debugging: Make the next if always take the second branch and observe
    # if the font size changes (reduced); that would mean the forced-set
    # has members it should not have.
    if forced:
        # Forced set is non-empty: rotate the contour start point
        # such that the last point in the list is a forced point.
        k = (n - 1) - max(forced)
        assert k >= 0

        deltas = _rot_list(deltas, k)
        coords = _rot_list(coords, k)
        forced = _rot_set(forced, k, n)

        # Debugging: Pass a set() instead of forced variable to the next call
        # to exercise forced-set computation for under-counting.
        chain, costs = _iup_contour_optimize_dp(deltas, coords, forced, tolerance)

        # Assemble solution.
        solution = set()
        i = n - 1
        while i is not None:
            solution.add(i)
            i = chain[i]
        solution.remove(-1)

        # if not forced <= solution:
        # 	print("coord", coords)
        # 	print("deltas", deltas)
        # 	print("len", len(deltas))
        assert forced <= solution, (forced, solution)

        deltas = [deltas[i] if i in solution else None for i in range(n)]

        deltas = _rot_list(deltas, -k)
    else:
        # Repeat the contour an extra time, solve the new case, then look for solutions of the
        # circular n-length problem in the solution for new linear case.  I cannot prove that
        # this always produces the optimal solution...
        chain, costs = _iup_contour_optimize_dp(
            deltas + deltas, coords + coords, forced, tolerance, n
        )
        best_sol, best_cost = None, n + 1

        for start in range(n - 1, len(costs) - 1):
            # Assemble solution.
            solution = set()
            i = start
            while i > start - n:
                solution.add(i % n)
                i = chain[i]
            if i == start - n:
                cost = costs[start] - costs[start - n]
                if cost <= best_cost:
                    best_sol, best_cost = solution, cost

        # if not forced <= best_sol:
        # 	print("coord", coords)
        # 	print("deltas", deltas)
        # 	print("len", len(deltas))
        assert forced <= best_sol, (forced, best_sol)

        deltas = [deltas[i] if i in best_sol else None for i in range(n)]

    return deltas


def iup_delta_optimize(
    deltas: _DeltaSegment,
    coords: _PointSegment,
    ends: _Endpoints,
    tolerance: Real = 0.0,
) -> _DeltaOrNoneSegment:
    """For the outline given in `coords`, with contour endpoints given
    in sorted increasing order in `ends`, optimize a set of delta
    values `deltas` within error `tolerance`.

    Returns delta vector that has most number of None items instead of
    the input delta.
    """
    assert sorted(ends) == ends and len(coords) == (ends[-1] + 1 if ends else 0) + 4
    n = len(coords)
    ends = ends + [n - 4, n - 3, n - 2, n - 1]
    out = []
    start = 0
    for end in ends:
        contour = iup_contour_optimize(
            deltas[start : end + 1], coords[start : end + 1], tolerance
        )
        assert len(contour) == end - start + 1
        out.extend(contour)
        start = end + 1

    return out
<|endoftext|>from fontTools.varLib.models import normalizeValue


def _denormalize(v, triplet):
    if v >= 0:
        return triplet[1] + v * (triplet[2] - triplet[1])
    else:
        return triplet[1] + v * (triplet[1] - triplet[0])


def map(
    font, location, *, inputNormalized=False, outputNormalized=False, dropZeroes=False
):
    if "fvar" not in font:
        return None

    fvar = font["fvar"]
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes}
    unknownAxes = sorted(tag for tag in location if tag not in axes)
    if unknownAxes:
        raise ValueError(f"Unknown axis tag(s): {', '.join(unknownAxes)}")

    if not inputNormalized:
        location = {
            tag: normalizeValue(value, axes[tag]) for tag, value in location.items()
        }

    if "avar" in font:
        location = font["avar"].renormalizeLocation(location, font, dropZeroes)

    if not outputNormalized:
        location = {
            tag: _denormalize(value, axes[tag]) for tag, value in location.items()
        }

    return location


def main(args=None):
    """Map variation coordinates through the `avar` table."""

    from fontTools.ttLib import TTFont
    import argparse

    if args is None:
        import sys

        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        "fonttools varLib.avar.map",
        description="Map variation coordinates through the `avar` table.",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    parser.add_argument(
        "coords",
        metavar="[AXIS=value...]",
        help="Coordinates to map, e.g. 'wght=700 wdth=75'.",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "-f", action="store_true", help="Do not omit axes at default location."
    )
    parser.add_argument(
        "-i", action="store_true", help="Input coordinates are normalized (-1..1)."
    )
    parser.add_argument(
        "-o", action="store_true", help="Output coordinates as normalized (-1..1)."
    )

    options = parser.parse_args(args)

    if not options.coords:
        parser.error(
            "No coordinates provided. Please specify at least one axis coordinate (e.g., wght=500)"
        )

    if options.font.endswith(".designspace"):
        from .build import build

        font = TTFont()
        build(font, options.font)
    else:
        font = TTFont(options.font)
        if "fvar" not in font:
            parser.error(f"Font '{options.font}' does not contain an 'fvar' table.")

    location = {}
    for item in options.coords:
        tag, sep, value = item.partition("=")
        if not sep or not tag or not value:
            parser.error(
                f"Invalid coordinate {item!r}. Expected AXIS=value, e.g. wght=500"
            )
        try:
            location[tag] = float(value)
        except ValueError:
            parser.error(
                f"Invalid coordinate value in {item!r}. Expected a number after '='"
            )

    try:
        mapped = map(
            font,
            location,
            inputNormalized=options.i,
            outputNormalized=options.o,
            dropZeroes=not options.f,
        )
    except ValueError as e:
        parser.error(str(e))
    assert mapped is not None

    for tag in mapped:
        v = mapped[tag]
        v = int(v) if v == int(v) else v
        print(f"{tag}={v:g}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.varLib import _add_fvar, _add_avar, load_designspace
from fontTools.ttLib import newTable
import logging

log = logging.getLogger("fontTools.varLib.avar")


def build(font, designspace_file):
    ds = load_designspace(designspace_file, require_sources=False)

    if not "fvar" in font:
        if "name" not in font:
            font["name"] = newTable("name")
        _add_fvar(font, ds.axes, ds.instances)

    axisTags = [a.axisTag for a in font["fvar"].axes]

    if "avar" in font:
        log.warning("avar table already present, overwriting.")
        del font["avar"]

    _add_avar(font, ds.axes, ds.axisMappings, axisTags)


def main(args=None):
    """Add `avar` table from designspace file to variable font."""

    from fontTools.ttLib import TTFont
    from fontTools.misc.cliTools import makeOutputFileName
    from fontTools import configLogger
    import argparse

    if args is None:
        import sys

        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        "fonttools varLib.avar.build",
        description="Add `avar` table from designspace file to variable font.",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    parser.add_argument(
        "designspace",
        metavar="family.designspace",
        help="Designspace file.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output font file name.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )

    options = parser.parse_args(args)

    configLogger(level=("INFO" if options.verbose else "WARNING"))

    font = TTFont(options.font)

    build(font, options.designspace)

    if options.output_file is None:
        outfile = makeOutputFileName(options.font, overWrite=True, suffix=".avar")
    else:
        outfile = options.output_file
    if outfile:
        log.info("Saving %s", outfile)
        font.save(outfile)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.ttLib import newTable
from fontTools.ttLib.tables._f_v_a_r import Axis as fvarAxis
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.basePen import NullPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.varLib.models import piecewiseLinearMap, normalizeValue
from fontTools.misc.cliTools import makeOutputFileName
import math
import logging
from pprint import pformat

__all__ = [
    "planWeightAxis",
    "planWidthAxis",
    "planSlantAxis",
    "planOpticalSizeAxis",
    "planAxis",
    "sanitizeWeight",
    "sanitizeWidth",
    "sanitizeSlant",
    "measureWeight",
    "measureWidth",
    "measureSlant",
    "normalizeLinear",
    "normalizeLog",
    "normalizeDegrees",
    "interpolateLinear",
    "interpolateLog",
    "processAxis",
    "makeDesignspaceSnippet",
    "addEmptyAvar",
    "main",
]

log = logging.getLogger("fontTools.varLib.avar.plan")

WEIGHTS = [
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
]

WIDTHS = [
    25.0,
    37.5,
    50.0,
    62.5,
    75.0,
    87.5,
    100.0,
    112.5,
    125.0,
    137.5,
    150.0,
    162.5,
    175.0,
    187.5,
    200.0,
]

SLANTS = list(math.degrees(math.atan(d / 20.0)) for d in range(-20, 21))

SIZES = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    18,
    24,
    30,
    36,
    48,
    60,
    72,
    96,
    120,
    144,
    192,
    240,
    288,
]


SAMPLES = 8


def normalizeLinear(value, rangeMin, rangeMax):
    """Linearly normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation."""
    return (value - rangeMin) / (rangeMax - rangeMin)


def interpolateLinear(t, a, b):
    """Linear interpolation between a and b, with t typically in [0, 1]."""
    return a + t * (b - a)


def normalizeLog(value, rangeMin, rangeMax):
    """Logarithmically normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation."""
    logMin = math.log(rangeMin)
    logMax = math.log(rangeMax)
    return (math.log(value) - logMin) / (logMax - logMin)


def interpolateLog(t, a, b):
    """Logarithmic interpolation between a and b, with t typically in [0, 1]."""
    logA = math.log(a)
    logB = math.log(b)
    return math.exp(logA + t * (logB - logA))


def normalizeDegrees(value, rangeMin, rangeMax):
    """Angularly normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation."""
    tanMin = math.tan(math.radians(rangeMin))
    tanMax = math.tan(math.radians(rangeMax))
    return (math.tan(math.radians(value)) - tanMin) / (tanMax - tanMin)


def measureWeight(glyphset, glyphs=None):
    """Measure the perceptual average weight of the given glyphs."""
    if isinstance(glyphs, dict):
        frequencies = glyphs
    else:
        frequencies = {g: 1 for g in glyphs}

    wght_sum = wdth_sum = 0
    for glyph_name in glyphs:
        if frequencies is not None:
            frequency = frequencies.get(glyph_name, 0)
            if frequency == 0:
                continue
        else:
            frequency = 1

        glyph = glyphset[glyph_name]

        pen = AreaPen(glyphset=glyphset)
        glyph.draw(pen)

        mult = glyph.width * frequency
        wght_sum += mult * abs(pen.value)
        wdth_sum += mult

    return wght_sum / wdth_sum


def measureWidth(glyphset, glyphs=None):
    """Measure the average width of the given glyphs."""
    if isinstance(glyphs, dict):
        frequencies = glyphs
    else:
        frequencies = {g: 1 for g in glyphs}

    wdth_sum = 0
    freq_sum = 0
    for glyph_name in glyphs:
        if frequencies is not None:
            frequency = frequencies.get(glyph_name, 0)
            if frequency == 0:
                continue
        else:
            frequency = 1

        glyph = glyphset[glyph_name]

        pen = NullPen()
        glyph.draw(pen)

        wdth_sum += glyph.width * frequency
        freq_sum += frequency

    return wdth_sum / freq_sum


def measureSlant(glyphset, glyphs=None):
    """Measure the perceptual average slant angle of the given glyphs."""
    if isinstance(glyphs, dict):
        frequencies = glyphs
    else:
        frequencies = {g: 1 for g in glyphs}

    slnt_sum = 0
    freq_sum = 0
    for glyph_name in glyphs:
        if frequencies is not None:
            frequency = frequencies.get(glyph_name, 0)
            if frequency == 0:
                continue
        else:
            frequency = 1

        glyph = glyphset[glyph_name]

        pen = StatisticsPen(glyphset=glyphset)
        glyph.draw(pen)

        mult = glyph.width * frequency
        slnt_sum += mult * pen.slant
        freq_sum += mult

    return -math.degrees(math.atan(slnt_sum / freq_sum))


def sanitizeWidth(userTriple, designTriple, pins, measurements):
    """Sanitize the width axis limits."""

    minVal, defaultVal, maxVal = (
        measurements[designTriple[0]],
        measurements[designTriple[1]],
        measurements[designTriple[2]],
    )

    calculatedMinVal = userTriple[1] * (minVal / defaultVal)
    calculatedMaxVal = userTriple[1] * (maxVal / defaultVal)

    log.info("Original width axis limits: %g:%g:%g", *userTriple)
    log.info(
        "Calculated width axis limits: %g:%g:%g",
        calculatedMinVal,
        userTriple[1],
        calculatedMaxVal,
    )

    if (
        abs(calculatedMinVal - userTriple[0]) / userTriple[1] > 0.05
        or abs(calculatedMaxVal - userTriple[2]) / userTriple[1] > 0.05
    ):
        log.warning("Calculated width axis min/max do not match user input.")
        log.warning(
            "  Current width axis limits: %g:%g:%g",
            *userTriple,
        )
        log.warning(
            "  Suggested width axis limits: %g:%g:%g",
            calculatedMinVal,
            userTriple[1],
            calculatedMaxVal,
        )

        return False

    return True


def sanitizeWeight(userTriple, designTriple, pins, measurements):
    """Sanitize the weight axis limits."""

    if len(set(userTriple)) < 3:
        return True

    minVal, defaultVal, maxVal = (
        measurements[designTriple[0]],
        measurements[designTriple[1]],
        measurements[designTriple[2]],
    )

    logMin = math.log(minVal)
    logDefault = math.log(defaultVal)
    logMax = math.log(maxVal)

    t = (userTriple[1] - userTriple[0]) / (userTriple[2] - userTriple[0])
    y = math.exp(logMin + t * (logMax - logMin))
    t = (y - minVal) / (maxVal - minVal)
    calculatedDefaultVal = userTriple[0] + t * (userTriple[2] - userTriple[0])

    log.info("Original weight axis limits: %g:%g:%g", *userTriple)
    log.info(
        "Calculated weight axis limits: %g:%g:%g",
        userTriple[0],
        calculatedDefaultVal,
        userTriple[2],
    )

    if abs(calculatedDefaultVal - userTriple[1]) / userTriple[1] > 0.05:
        log.warning("Calculated weight axis default does not match user input.")

        log.warning(
            "  Current weight axis limits: %g:%g:%g",
            *userTriple,
        )

        log.warning(
            "  Suggested weight axis limits, changing default: %g:%g:%g",
            userTriple[0],
            calculatedDefaultVal,
            userTriple[2],
        )

        t = (userTriple[2] - userTriple[0]) / (userTriple[1] - userTriple[0])
        y = math.exp(logMin + t * (logDefault - logMin))
        t = (y - minVal) / (defaultVal - minVal)
        calculatedMaxVal = userTriple[0] + t * (userTriple[1] - userTriple[0])
        log.warning(
            "  Suggested weight axis limits, changing maximum: %g:%g:%g",
            userTriple[0],
            userTriple[1],
            calculatedMaxVal,
        )

        t = (userTriple[0] - userTriple[2]) / (userTriple[1] - userTriple[2])
        y = math.exp(logMax + t * (logDefault - logMax))
        t = (y - maxVal) / (defaultVal - maxVal)
        calculatedMinVal = userTriple[2] + t * (userTriple[1] - userTriple[2])
        log.warning(
            "  Suggested weight axis limits, changing minimum: %g:%g:%g",
            calculatedMinVal,
            userTriple[1],
            userTriple[2],
        )

        return False

    return True


def sanitizeSlant(userTriple, designTriple, pins, measurements):
    """Sanitize the slant axis limits."""

    log.info("Original slant axis limits: %g:%g:%g", *userTriple)
    log.info(
        "Calculated slant axis limits: %g:%g:%g",
        measurements[designTriple[0]],
        measurements[designTriple[1]],
        measurements[designTriple[2]],
    )

    if (
        abs(measurements[designTriple[0]] - userTriple[0]) > 1
        or abs(measurements[designTriple[1]] - userTriple[1]) > 1
        or abs(measurements[designTriple[2]] - userTriple[2]) > 1
    ):
        log.warning("Calculated slant axis min/default/max do not match user input.")
        log.warning(
            "  Current slant axis limits: %g:%g:%g",
            *userTriple,
        )
        log.warning(
            "  Suggested slant axis limits: %g:%g:%g",
            measurements[designTriple[0]],
            measurements[designTriple[1]],
            measurements[designTriple[2]],
        )

        return False

    return True


def planAxis(
    measureFunc,
    normalizeFunc,
    interpolateFunc,
    glyphSetFunc,
    axisTag,
    axisLimits,
    values,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitizeFunc=None,
):
    """Plan an axis.

    measureFunc: callable that takes a glyphset and an optional
    list of glyphnames, and returns the glyphset-wide measurement
    to be used for the axis.

    normalizeFunc: callable that takes a measurement and a minimum
    and maximum, and normalizes the measurement into the range 0..1,
    possibly extrapolating too.

    interpolateFunc: callable that takes a normalized t value, and a
    minimum and maximum, and returns the interpolated value,
    possibly extrapolating too.

    glyphSetFunc: callable that takes a variations "location" dictionary,
    and returns a glyphset.

    axisTag: the axis tag string.

    axisLimits: a triple of minimum, default, and maximum values for
    the axis. Or an `fvar` Axis object.

    values: a list of output values to map for this axis.

    samples: the number of samples to use when sampling. Default 8.

    glyphs: a list of glyph names to use when sampling. Defaults to None,
    which will process all glyphs.

    designLimits: an optional triple of minimum, default, and maximum values
    represenging the "design" limits for the axis. If not provided, the
    axisLimits will be used.

    pins: an optional dictionary of before/after mapping entries to pin in
    the output.

    sanitizeFunc: an optional callable to call to sanitize the axis limits.
    """

    if isinstance(axisLimits, fvarAxis):
        axisLimits = (axisLimits.minValue, axisLimits.defaultValue, axisLimits.maxValue)
    minValue, defaultValue, maxValue = axisLimits

    if samples is None:
        samples = SAMPLES
    if glyphs is None:
        glyphs = glyphSetFunc({}).keys()
    if pins is None:
        pins = {}
    else:
        pins = pins.copy()

    log.info(
        "Axis limits min %g / default %g / max %g", minValue, defaultValue, maxValue
    )
    triple = (minValue, defaultValue, maxValue)

    if designLimits is not None:
        log.info("Axis design-limits min %g / default %g / max %g", *designLimits)
    else:
        designLimits = triple

    if pins:
        log.info("Pins %s", sorted(pins.items()))
    pins.update(
        {
            minValue: designLimits[0],
            defaultValue: designLimits[1],
            maxValue: designLimits[2],
        }
    )

    out = {}
    outNormalized = {}

    axisMeasurements = {}
    for value in sorted({minValue, defaultValue, maxValue} | set(pins.keys())):
        glyphset = glyphSetFunc(location={axisTag: value})
        designValue = pins[value]
        axisMeasurements[designValue] = measureFunc(glyphset, glyphs)

    if sanitizeFunc is not None:
        log.info("Sanitizing axis limit values for the `%s` axis.", axisTag)
        sanitizeFunc(triple, designLimits, pins, axisMeasurements)

    log.debug("Calculated average value:\n%s", pformat(axisMeasurements))

    for (rangeMin, targetMin), (rangeMax, targetMax) in zip(
        list(sorted(pins.items()))[:-1],
        list(sorted(pins.items()))[1:],
    ):
        targetValues = {w for w in values if rangeMin < w < rangeMax}
        if not targetValues:
            continue

        normalizedMin = normalizeValue(rangeMin, triple)
        normalizedMax = normalizeValue(rangeMax, triple)
        normalizedTargetMin = normalizeValue(targetMin, designLimits)
        normalizedTargetMax = normalizeValue(targetMax, designLimits)

        log.info("Planning target values %s.", sorted(targetValues))
        log.info("Sampling %u points in range %g,%g.", samples, rangeMin, rangeMax)
        valueMeasurements = axisMeasurements.copy()
        for sample in range(1, samples + 1):
            value = rangeMin + (rangeMax - rangeMin) * sample / (samples + 1)
            log.debug("Sampling value %g.", value)
            glyphset = glyphSetFunc(location={axisTag: value})
            designValue = piecewiseLinearMap(value, pins)
            valueMeasurements[designValue] = measureFunc(glyphset, glyphs)
        log.debug("Sampled average value:\n%s", pformat(valueMeasurements))

        measurementValue = {}
        for value in sorted(valueMeasurements):
            measurementValue[valueMeasurements[value]] = value

        out[rangeMin] = targetMin
        outNormalized[normalizedMin] = normalizedTargetMin
        for value in sorted(targetValues):
            t = normalizeFunc(value, rangeMin, rangeMax)
            targetMeasurement = interpolateFunc(
                t, valueMeasurements[targetMin], valueMeasurements[targetMax]
            )
            targetValue = piecewiseLinearMap(targetMeasurement, measurementValue)
            log.debug("Planned mapping value %g to %g." % (value, targetValue))
            out[value] = targetValue
            valueNormalized = normalizedMin + (value - rangeMin) / (
                rangeMax - rangeMin
            ) * (normalizedMax - normalizedMin)
            outNormalized[valueNormalized] = normalizedTargetMin + (
                targetValue - targetMin
            ) / (targetMax - targetMin) * (normalizedTargetMax - normalizedTargetMin)
        out[rangeMax] = targetMax
        outNormalized[normalizedMax] = normalizedTargetMax

    log.info("Planned mapping for the `%s` axis:\n%s", axisTag, pformat(out))
    log.info(
        "Planned normalized mapping for the `%s` axis:\n%s",
        axisTag,
        pformat(outNormalized),
    )

    if all(abs(k - v) < 0.01 for k, v in outNormalized.items()):
        log.info("Detected identity mapping for the `%s` axis. Dropping.", axisTag)
        out = {}
        outNormalized = {}

    return out, outNormalized


def planWeightAxis(
    glyphSetFunc,
    axisLimits,
    weights=None,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitize=False,
):
    """Plan a weight (`wght`) axis.

    weights: A list of weight values to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=weights, and the appropriate
    arguments. See documenation for planAxis for more information.
    """

    if weights is None:
        weights = WEIGHTS

    return planAxis(
        measureWeight,
        normalizeLinear,
        interpolateLog,
        glyphSetFunc,
        "wght",
        axisLimits,
        values=weights,
        samples=samples,
        glyphs=glyphs,
        designLimits=designLimits,
        pins=pins,
        sanitizeFunc=sanitizeWeight if sanitize else None,
    )


def planWidthAxis(
    glyphSetFunc,
    axisLimits,
    widths=None,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitize=False,
):
    """Plan a width (`wdth`) axis.

    widths: A list of width values (percentages) to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=widths, and the appropriate
    arguments. See documenation for planAxis for more information.
    """

    if widths is None:
        widths = WIDTHS

    return planAxis(
        measureWidth,
        normalizeLinear,
        interpolateLinear,
        glyphSetFunc,
        "wdth",
        axisLimits,
        values=widths,
        samples=samples,
        glyphs=glyphs,
        designLimits=designLimits,
        pins=pins,
        sanitizeFunc=sanitizeWidth if sanitize else None,
    )


def planSlantAxis(
    glyphSetFunc,
    axisLimits,
    slants=None,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitize=False,
):
    """Plan a slant (`slnt`) axis.

    slants: A list slant angles to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=slants, and the appropriate
    arguments. See documenation for planAxis for more information.
    """

    if slants is None:
        slants = SLANTS

    return planAxis(
        measureSlant,
        normalizeDegrees,
        interpolateLinear,
        glyphSetFunc,
        "slnt",
        axisLimits,
        values=slants,
        samples=samples,
        glyphs=glyphs,
        designLimits=designLimits,
        pins=pins,
        sanitizeFunc=sanitizeSlant if sanitize else None,
    )


def planOpticalSizeAxis(
    glyphSetFunc,
    axisLimits,
    sizes=None,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitize=False,
):
    """Plan a optical-size (`opsz`) axis.

    sizes: A list of optical size values to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=sizes, and the appropriate
    arguments. See documenation for planAxis for more information.
    """

    if sizes is None:
        sizes = SIZES

    return planAxis(
        measureWeight,
        normalizeLog,
        interpolateLog,
        glyphSetFunc,
        "opsz",
        axisLimits,
        values=sizes,
        samples=samples,
        glyphs=glyphs,
        designLimits=designLimits,
        pins=pins,
    )


def makeDesignspaceSnippet(axisTag, axisName, axisLimit, mapping):
    """Make a designspace snippet for a single axis."""

    designspaceSnippet = (
        '    <axis tag="%s" name="%s" minimum="%g" default="%g" maximum="%g"'
        % ((axisTag, axisName) + axisLimit)
    )
    if mapping:
        designspaceSnippet += ">\n"
    else:
        designspaceSnippet += "/>"

    for key, value in mapping.items():
        designspaceSnippet += '      <map input="%g" output="%g"/>\n' % (key, value)

    if mapping:
        designspaceSnippet += "    </axis>"

    return designspaceSnippet


def addEmptyAvar(font):
    """Add an empty `avar` table to the font."""
    font["avar"] = avar = newTable("avar")
    for axis in font["fvar"].axes:
        avar.segments[axis.axisTag] = {}


def processAxis(
    font,
    planFunc,
    axisTag,
    axisName,
    values,
    samples=None,
    glyphs=None,
    designLimits=None,
    pins=None,
    sanitize=False,
    plot=False,
):
    """Process a single axis."""

    axisLimits = None
    for axis in font["fvar"].axes:
        if axis.axisTag == axisTag:
            axisLimits = axis
            break
    if axisLimits is None:
        return ""
    axisLimits = (axisLimits.minValue, axisLimits.defaultValue, axisLimits.maxValue)

    log.info("Planning %s axis.", axisName)

    if "avar" in font:
        existingMapping = font["avar"].segments[axisTag]
        font["avar"].segments[axisTag] = {}
    else:
        existingMapping = None

    if values is not None and isinstance(values, str):
        values = [float(w) for w in values.split()]

    if designLimits is not None and isinstance(designLimits, str):
        designLimits = [float(d) for d in designLimits.split(":")]
        assert (
            len(designLimits) == 3
            and designLimits[0] <= designLimits[1] <= designLimits[2]
        )
    else:
        designLimits = None

    if pins is not None and isinstance(pins, str):
        newPins = {}
        for pin in pins.split():
            before, after = pin.split(":")
            newPins[float(before)] = float(after)
        pins = newPins
        del newPins

    mapping, mappingNormalized = planFunc(
        font.getGlyphSet,
        axisLimits,
        values,
        samples=samples,
        glyphs=glyphs,
        designLimits=designLimits,
        pins=pins,
        sanitize=sanitize,
    )

    if plot:
        from matplotlib import pyplot

        pyplot.plot(
            sorted(mappingNormalized),
            [mappingNormalized[k] for k in sorted(mappingNormalized)],
        )
        pyplot.show()

    if existingMapping is not None:
        log.info("Existing %s mapping:\n%s", axisName, pformat(existingMapping))

    if mapping:
        if "avar" not in font:
            addEmptyAvar(font)
        font["avar"].segments[axisTag] = mappingNormalized
    else:
        if "avar" in font:
            font["avar"].segments[axisTag] = {}

    designspaceSnippet = makeDesignspaceSnippet(
        axisTag,
        axisName,
        axisLimits,
        mapping,
    )
    return designspaceSnippet


def main(args=None):
    """Plan the standard axis mappings for a variable font"""

    if args is None:
        import sys

        args = sys.argv[1:]

    from fontTools import configLogger
    from fontTools.ttLib import TTFont
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.avar.plan",
        description="Plan `avar` table for variable font",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output font file name.",
    )
    parser.add_argument(
        "--weights", type=str, help="Space-separate list of weights to generate."
    )
    parser.add_argument(
        "--widths", type=str, help="Space-separate list of widths to generate."
    )
    parser.add_argument(
        "--slants", type=str, help="Space-separate list of slants to generate."
    )
    parser.add_argument(
        "--sizes", type=str, help="Space-separate list of optical-sizes to generate."
    )
    parser.add_argument("--samples", type=int, help="Number of samples.")
    parser.add_argument(
        "-s", "--sanitize", action="store_true", help="Sanitize axis limits"
    )
    parser.add_argument(
        "-g",
        "--glyphs",
        type=str,
        help="Space-separate list of glyphs to use for sampling.",
    )
    parser.add_argument(
        "--weight-design-limits",
        type=str,
        help="min:default:max in design units for the `wght` axis.",
    )
    parser.add_argument(
        "--width-design-limits",
        type=str,
        help="min:default:max in design units for the `wdth` axis.",
    )
    parser.add_argument(
        "--slant-design-limits",
        type=str,
        help="min:default:max in design units for the `slnt` axis.",
    )
    parser.add_argument(
        "--optical-size-design-limits",
        type=str,
        help="min:default:max in design units for the `opsz` axis.",
    )
    parser.add_argument(
        "--weight-pins",
        type=str,
        help="Space-separate list of before:after pins for the `wght` axis.",
    )
    parser.add_argument(
        "--width-pins",
        type=str,
        help="Space-separate list of before:after pins for the `wdth` axis.",
    )
    parser.add_argument(
        "--slant-pins",
        type=str,
        help="Space-separate list of before:after pins for the `slnt` axis.",
    )
    parser.add_argument(
        "--optical-size-pins",
        type=str,
        help="Space-separate list of before:after pins for the `opsz` axis.",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot the resulting mapping."
    )

    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )
    logging_group.add_argument(
        "-q", "--quiet", action="store_true", help="Turn verbosity off."
    )

    options = parser.parse_args(args)

    configLogger(
        level=("DEBUG" if options.verbose else "WARNING" if options.quiet else "INFO")
    )

    font = TTFont(options.font)
    if not "fvar" in font:
        log.error("Not a variable font.")
        return 1

    if options.glyphs is not None:
        glyphs = options.glyphs.split()
        if ":" in options.glyphs:
            glyphs = {}
            for g in options.glyphs.split():
                if ":" in g:
                    glyph, frequency = g.split(":")
                    glyphs[glyph] = float(frequency)
                else:
                    glyphs[g] = 1.0
    else:
        glyphs = None

    designspaceSnippets = []

    designspaceSnippets.append(
        processAxis(
            font,
            planWeightAxis,
            "wght",
            "Weight",
            values=options.weights,
            samples=options.samples,
            glyphs=glyphs,
            designLimits=options.weight_design_limits,
            pins=options.weight_pins,
            sanitize=options.sanitize,
            plot=options.plot,
        )
    )
    designspaceSnippets.append(
        processAxis(
            font,
            planWidthAxis,
            "wdth",
            "Width",
            values=options.widths,
            samples=options.samples,
            glyphs=glyphs,
            designLimits=options.width_design_limits,
            pins=options.width_pins,
            sanitize=options.sanitize,
            plot=options.plot,
        )
    )
    designspaceSnippets.append(
        processAxis(
            font,
            planSlantAxis,
            "slnt",
            "Slant",
            values=options.slants,
            samples=options.samples,
            glyphs=glyphs,
            designLimits=options.slant_design_limits,
            pins=options.slant_pins,
            sanitize=options.sanitize,
            plot=options.plot,
        )
    )
    designspaceSnippets.append(
        processAxis(
            font,
            planOpticalSizeAxis,
            "opsz",
            "OpticalSize",
            values=options.sizes,
            samples=options.samples,
            glyphs=glyphs,
            designLimits=options.optical_size_design_limits,
            pins=options.optical_size_pins,
            sanitize=options.sanitize,
            plot=options.plot,
        )
    )

    log.info("Designspace snippet:")
    for snippet in designspaceSnippets:
        if snippet:
            print(snippet)

    if options.output_file is None:
        outfile = makeOutputFileName(options.font, overWrite=True, suffix=".avar")
    else:
        outfile = options.output_file
    if outfile:
        log.info("Saving %s", outfile)
        font.save(outfile)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>import logging

log = logging.getLogger("fontTools.varLib.avar")


def main(args=None):
    from fontTools.ttLib import TTFont
    from fontTools.misc.cliTools import makeOutputFileName
    from fontTools import configLogger
    import argparse
    import sys

    print(
        "WARNING: This script is deprecated. Use `fonttools varLib.avar.build` "
        "or `fonttools varLib.avar.unbuild` instead.\n",
        file=sys.stderr,
    )

    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        "fonttools varLib.avar",
        description="Add `avar` table from designspace file to variable font.",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    parser.add_argument(
        "designspace",
        metavar="family.designspace",
        help="Designspace file.",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output font file name.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )

    options = parser.parse_args(args)

    configLogger(level=("INFO" if options.verbose else "WARNING"))

    font = TTFont(options.font)

    if options.designspace is None:
        from .unbuild import unbuild

        unbuild(font)
        return 0

    from .build import build

    build(font, options.designspace)

    if options.output_file is None:
        outfile = makeOutputFileName(options.font, overWrite=True, suffix=".avar")
    else:
        outfile = options.output_file
    if outfile:
        log.info("Saving %s", outfile)
        font.save(outfile)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.varLib.models import VariationModel
from fontTools.varLib.varStore import VarStoreInstancer
from fontTools.misc.fixedTools import fixedToFloat as fi2fl
from itertools import product
import sys


def _denormalize(v, axis):
    if v >= 0:
        return axis.defaultValue + v * (axis.maxValue - axis.defaultValue)
    else:
        return axis.defaultValue + v * (axis.defaultValue - axis.minValue)


def _pruneLocations(locations, poles, axisTags):
    # Now we have all the input locations, find which ones are
    # not needed and remove them.

    # Note: This algorithm is heavily tied to how VariationModel
    # is implemented.  It assumes that input was extracted from
    # VariationModel-generated object, like an ItemVariationStore
    # created by fontmake using varLib.models.VariationModel.
    # Some CoPilot blabbering:
    # I *think* I can prove that this algorithm is correct, but
    # I'm not 100% sure.  It's possible that there are edge cases
    # where this algorithm will fail.  I'm not sure how to prove
    # that it's correct, but I'm also not sure how to prove that
    # it's incorrect.  I'm not sure how to write a test case that
    # would prove that it's incorrect.  I'm not sure how to write
    # a test case that would prove that it's correct.

    model = VariationModel(locations, axisTags)
    modelMapping = model.mapping
    modelSupports = model.supports
    pins = {tuple(k.items()): None for k in poles}
    for location in poles:
        i = locations.index(location)
        i = modelMapping[i]
        support = modelSupports[i]
        supportAxes = set(support.keys())
        for axisTag, (minV, _, maxV) in support.items():
            for v in (minV, maxV):
                if v in (-1, 0, 1):
                    continue
                for pin in pins.keys():
                    pinLocation = dict(pin)
                    pinAxes = set(pinLocation.keys())
                    if pinAxes != supportAxes:
                        continue
                    if axisTag not in pinAxes:
                        continue
                    if pinLocation[axisTag] == v:
                        break
                else:
                    # No pin found. Go through the previous masters
                    # and find a suitable pin.  Going backwards is
                    # better because it can find a pin that is close
                    # to the pole in more dimensions, and reducing
                    # the total number of pins needed.
                    for candidateIdx in range(i - 1, -1, -1):
                        candidate = modelSupports[candidateIdx]
                        candidateAxes = set(candidate.keys())
                        if candidateAxes != supportAxes:
                            continue
                        if axisTag not in candidateAxes:
                            continue
                        candidate = {
                            k: defaultV for k, (_, defaultV, _) in candidate.items()
                        }
                        if candidate[axisTag] == v:
                            pins[tuple(candidate.items())] = None
                            break
                    else:
                        assert False, "No pin found"
    return [dict(t) for t in pins.keys()]


def mappings_from_avar(font, denormalize=True):
    fvarAxes = font["fvar"].axes
    axisMap = {a.axisTag: a for a in fvarAxes}
    axisTags = [a.axisTag for a in fvarAxes]
    axisIndexes = {a.axisTag: i for i, a in enumerate(fvarAxes)}
    if "avar" not in font:
        return {}, []
    avar = font["avar"]
    axisMaps = {
        tag: seg
        for tag, seg in avar.segments.items()
        if seg and seg != {-1: -1, 0: 0, 1: 1}
    }
    mappings = []

    if getattr(avar, "majorVersion", 1) == 2:
        varStore = avar.table.VarStore
        regions = varStore.VarRegionList.Region

        # Find all the input locations; this finds "poles", that are
        # locations of the peaks, and "corners", that are locations
        # of the corners of the regions.  These two sets of locations
        # together constitute inputLocations to consider.

        poles = {(): None}  # Just using it as an ordered set
        inputLocations = set({()})
        for varData in varStore.VarData:
            regionIndices = varData.VarRegionIndex
            for regionIndex in regionIndices:
                peakLocation = []
                corners = []
                region = regions[regionIndex]
                for axisIndex, axis in enumerate(region.VarRegionAxis):
                    if axis.PeakCoord == 0:
                        continue
                    axisTag = axisTags[axisIndex]
                    peakLocation.append((axisTag, axis.PeakCoord))
                    corner = []
                    if axis.StartCoord != 0:
                        corner.append((axisTag, axis.StartCoord))
                    if axis.EndCoord != 0:
                        corner.append((axisTag, axis.EndCoord))
                    corners.append(corner)
                corners = set(product(*corners))
                peakLocation = tuple(peakLocation)
                poles[peakLocation] = None
                inputLocations.add(peakLocation)
                inputLocations.update(corners)

        # Sort them by number of axes, then by axis order
        inputLocations = [
            dict(t)
            for t in sorted(
                inputLocations,
                key=lambda t: (len(t), tuple(axisIndexes[tag] for tag, _ in t)),
            )
        ]
        poles = [dict(t) for t in poles.keys()]
        inputLocations = _pruneLocations(inputLocations, list(poles), axisTags)

        # Find the output locations, at input locations
        varIdxMap = avar.table.VarIdxMap
        instancer = VarStoreInstancer(varStore, fvarAxes)
        for location in inputLocations:
            instancer.setLocation(location)
            outputLocation = {}
            for axisIndex, axisTag in enumerate(axisTags):
                varIdx = axisIndex
                if varIdxMap is not None:
                    varIdx = varIdxMap[varIdx]
                delta = instancer[varIdx]
                if delta != 0:
                    v = location.get(axisTag, 0)
                    v = v + fi2fl(delta, 14)
                    # See https://github.com/fonttools/fonttools/pull/3598#issuecomment-2266082009
                    # v = max(-1, min(1, v))
                    outputLocation[axisTag] = v
            mappings.append((location, outputLocation))

        # Remove base master we added, if it maps to the default location
        assert mappings[0][0] == {}
        if mappings[0][1] == {}:
            mappings.pop(0)

    if denormalize:
        for tag, seg in axisMaps.items():
            if tag not in axisMap:
                raise ValueError(f"Unknown axis tag {tag}")
            denorm = lambda v: _denormalize(v, axisMap[tag])
            axisMaps[tag] = {denorm(k): denorm(v) for k, v in seg.items()}

        for i, (inputLoc, outputLoc) in enumerate(mappings):
            inputLoc = {
                tag: _denormalize(val, axisMap[tag]) for tag, val in inputLoc.items()
            }
            outputLoc = {
                tag: _denormalize(val, axisMap[tag]) for tag, val in outputLoc.items()
            }
            mappings[i] = (inputLoc, outputLoc)

    return axisMaps, mappings


def unbuild(font, f=sys.stdout):
    fvar = font["fvar"]
    axes = fvar.axes
    segments, mappings = mappings_from_avar(font)

    if "name" in font:
        name = font["name"]
        axisNames = {
            axis.axisTag: name.getDebugName(axis.axisNameID) or axis.axisTag
            for axis in axes
        }
    else:
        axisNames = {a.axisTag: a.axisTag for a in axes}

    print("<?xml version='1.0' encoding='UTF-8'?>", file=f)
    print('<designspace format="5.1">', file=f)
    print("  <axes>", file=f)
    for axis in axes:

        axisName = axisNames[axis.axisTag]

        triplet = (axis.minValue, axis.defaultValue, axis.maxValue)
        triplet = [int(v) if v == int(v) else v for v in triplet]

        axisMap = segments.get(axis.axisTag)
        closing = "/>" if axisMap is None else ">"

        print(
            f'    <axis tag="{axis.axisTag}" name="{axisName}" minimum="{triplet[0]}" maximum="{triplet[2]}" default="{triplet[1]}"{closing}',
            file=f,
        )
        if axisMap is not None:
            for k in sorted(axisMap.keys()):
                v = axisMap[k]
                k = int(k) if k == int(k) else k
                v = int(v) if v == int(v) else v
                print(f'      <map input="{k}" output="{v}"/>', file=f)
            print("    </axis>", file=f)
    if mappings:
        print("    <mappings>", file=f)
        for inputLoc, outputLoc in mappings:
            print("      <mapping>", file=f)
            print("        <input>", file=f)
            for tag in sorted(inputLoc.keys()):
                v = inputLoc[tag]
                v = int(v) if v == int(v) else v
                print(
                    f'          <dimension name="{axisNames[tag]}" xvalue="{v}"/>',
                    file=f,
                )
            print("        </input>", file=f)
            print("        <output>", file=f)
            for tag in sorted(outputLoc.keys()):
                v = outputLoc[tag]
                v = int(v) if v == int(v) else v
                print(
