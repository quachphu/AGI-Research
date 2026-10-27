        if self.baseMarks or self.marks:
            subtables_.append((self.marks, self.baseMarks))
        return subtables_

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.get_subtables_() == other.get_subtables_()
        )

    def inferGlyphClasses(self):
        result = {}
        for marks, baseMarks in self.get_subtables_():
            result.update({glyph: 3 for glyph in baseMarks})
            result.update({glyph: 3 for glyph in marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-mark
            positioning lookup.
        """
        subtables = []
        for subtable in self.get_subtables_():
            markClasses = self.buildMarkClasses_(subtable[0])
            markClassList = sorted(markClasses.keys(), key=markClasses.get)
            marks = {
                mark: (markClasses[mc], anchor)
                for mark, (mc, anchor) in subtable[0].items()
            }

            st = ot.MarkMarkPos()
            st.Format = 1
            st.ClassCount = len(markClasses)
            st.Mark1Coverage = buildCoverage(marks, self.glyphMap)
            st.Mark2Coverage = buildCoverage(subtable[1], self.glyphMap)
            st.Mark1Array = buildMarkArray(marks, self.glyphMap)
            st.Mark2Array = ot.Mark2Array()
            st.Mark2Array.Mark2Count = len(st.Mark2Coverage.glyphs)
            st.Mark2Array.Mark2Record = []
            for base in st.Mark2Coverage.glyphs:
                anchors = [subtable[1][base].get(mc) for mc in markClassList]
                st.Mark2Array.Mark2Record.append(buildMark2Record(anchors))
            subtables.append(st)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.subtables_.append((self.marks, self.baseMarks))
        self.marks = {}
        self.baseMarks = {}


class ReverseChainSingleSubstBuilder(LookupBuilder):
    """Builds a Reverse Chaining Contextual Single Substitution (GSUB8) lookup.

    Users are expected to manually add substitutions to the ``substitutions``
    attribute after the object has been initialized, e.g.::

        # reversesub [a e n] d' by d.alt;
        prefix = [ ["a", "e", "n"] ]
        suffix = []
        mapping = { "d": "d.alt" }
        builder.substitutions.append( (prefix, suffix, mapping) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        substitutions: A three-element tuple consisting of a prefix sequence,
            a suffix sequence, and a dictionary of single substitutions.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 8)
        self.rules = []  # (prefix, suffix, mapping)

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual substitution lookup.
        """
        subtables = []
        for prefix, suffix, mapping in self.rules:
            st = ot.ReverseChainSingleSubst()
            st.Format = 1
            self.setBacktrackCoverage_(prefix, st)
            self.setLookAheadCoverage_(suffix, st)
            st.Coverage = buildCoverage(mapping.keys(), self.glyphMap)
            st.GlyphCount = len(mapping)
            st.Substitute = [mapping[g] for g in st.Coverage.glyphs]
            subtables.append(st)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        # Nothing to do here, each substitution is in its own subtable.
        pass


class AnySubstBuilder(LookupBuilder):
    """A temporary builder for Single, Multiple, or Ligature substitution lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub x by y;
        builder.mapping[("x",)] = ("y",)
        # sub a by b c;
        builder.mapping[("a",)] = ("b", "c")
        # sub f i by f_i;
        builder.mapping[("f", "i")] = ("f_i",)

    Then call `promote_lookup_type()` to convert this builder into the
    appropriate type of substitution lookup builder. This would promote single
    substitutions to either multiple or ligature substitutions, depending on the
    rest of the rules in the mapping.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: An ordered dictionary mapping a tuple of glyph names to another
            tuple of glyph names.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 0)
        self.mapping = OrderedDict()

    def _add_to_single_subst(self, builder, key, value):
        if key[0] != self.SUBTABLE_BREAK_:
            key = key[0]
        builder.mapping[key] = value[0]

    def _add_to_multiple_subst(self, builder, key, value):
        if key[0] != self.SUBTABLE_BREAK_:
            key = key[0]
        builder.mapping[key] = value

    def _add_to_ligature_subst(self, builder, key, value):
        builder.ligatures[key] = value[0]

    def can_add_mapping(self, mapping) -> bool:
        if mapping is None:
            return True
        # single sub rules can be treated as (degenerate) liga-or-multi sub
        # rules, but multi and liga sub rules themselves have incompatible
        # representations. It is uncommon that these are in the same set of
        # rules, but it happens.
        is_multi = any(len(v) > 1 for v in mapping.values())
        is_liga = any(len(k) > 1 for k in mapping.keys())

        has_existing_multi = False
        has_existing_liga = False

        for k, v in self.mapping.items():
            if k[0] == self.SUBTABLE_BREAK_:
                continue
            if len(k) > 1:
                has_existing_liga = True
            if len(v) > 1:
                has_existing_multi = True

        can_reuse = not (has_existing_multi and is_liga) and not (
            has_existing_liga and is_multi
        )
        return can_reuse

    def promote_lookup_type(self, is_named_lookup):
        # https://github.com/fonttools/fonttools/issues/612
        # A multiple substitution may have a single destination, in which case
        # it will look just like a single substitution. So if there are both
        # multiple and single substitutions, upgrade all the single ones to
        # multiple substitutions. Similarly, a ligature substitution may have a
        # single source glyph, so if there are both ligature and single
        # substitutions, upgrade all the single ones to ligature substitutions.
        builder_classes = []
        for key, value in self.mapping.items():
            if key[0] == self.SUBTABLE_BREAK_:
                builder_classes.append(None)
            elif len(key) == 1 and len(value) == 1:
                builder_classes.append(SingleSubstBuilder)
            elif len(key) == 1 and len(value) != 1:
                builder_classes.append(MultipleSubstBuilder)
            elif len(key) > 1 and len(value) == 1:
                builder_classes.append(LigatureSubstBuilder)
            else:
                assert False, "Should not happen"

        has_multiple = any(b is MultipleSubstBuilder for b in builder_classes)
        has_ligature = any(b is LigatureSubstBuilder for b in builder_classes)

        # If we have mixed single and multiple substitutions,
        # upgrade all single substitutions to multiple substitutions.
        to_multiple = has_multiple and not has_ligature

        # If we have mixed single and ligature substitutions,
        # upgrade all single substitutions to ligature substitutions.
        to_ligature = has_ligature and not has_multiple

        # If we have only single substitutions, we can keep them as is.
        to_single = not has_ligature and not has_multiple

        ret = []
        if to_single:
            builder = SingleSubstBuilder(self.font, self.location)
            for key, value in self.mapping.items():
                self._add_to_single_subst(builder, key, value)
            ret = [builder]
        elif to_multiple:
            builder = MultipleSubstBuilder(self.font, self.location)
            for key, value in self.mapping.items():
                self._add_to_multiple_subst(builder, key, value)
            ret = [builder]
        elif to_ligature:
            builder = LigatureSubstBuilder(self.font, self.location)
            for key, value in self.mapping.items():
                self._add_to_ligature_subst(builder, key, value)
            ret = [builder]
        elif is_named_lookup:
            # This is a named lookup with mixed substitutions that can’t be promoted,
            # since we can’t split it into multiple lookups, we return None here to
            # signal that to the caller
            return None
        else:
            curr_builder = None
            for builder_class, (key, value) in zip(
                builder_classes, self.mapping.items()
            ):
                if curr_builder is None or type(curr_builder) is not builder_class:
                    curr_builder = builder_class(self.font, self.location)
                    ret.append(curr_builder)
                if builder_class is SingleSubstBuilder:
                    self._add_to_single_subst(curr_builder, key, value)
                elif builder_class is MultipleSubstBuilder:
                    self._add_to_multiple_subst(curr_builder, key, value)
                elif builder_class is LigatureSubstBuilder:
                    self._add_to_ligature_subst(curr_builder, key, value)
                else:
                    assert False, "Should not happen"

        for builder in ret:
            builder.extension = self.extension
            builder.lookupflag = self.lookupflag
            builder.markFilterSet = self.markFilterSet
        return ret

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        assert False

    def getAlternateGlyphs(self):
        return {
            key[0]: value
            for key, value in self.mapping.items()
            if len(key) == 1 and len(value) == 1
        }

    def add_subtable_break(self, location):
        self.mapping[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class SingleSubstBuilder(LookupBuilder):
    """Builds a Single Substitution (GSUB1) lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub x by y;
        builder.mapping["x"] = "y"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: A dictionary mapping a single glyph name to another glyph name.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 1)
        self.mapping = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the multiple
            substitution lookup.
        """
        subtables = self.build_subst_subtables(self.mapping, buildSingleSubstSubtable)
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        return {glyph: [repl] for glyph, repl in self.mapping.items()}

    def add_subtable_break(self, location):
        self.mapping[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class ClassPairPosSubtableBuilder(object):
    """Builds class-based Pair Positioning (GPOS2 format 2) subtables.

    Note that this does *not* build a GPOS2 ``otTables.Lookup`` directly,
    but builds a list of ``otTables.PairPos`` subtables. It is used by the
    :class:`PairPosBuilder` below.

    Attributes:
        builder (PairPosBuilder): A pair positioning lookup builder.
    """

    def __init__(self, builder):
        self.builder_ = builder
        self.classDef1_, self.classDef2_ = None, None
        self.values_ = {}  # (glyphclass1, glyphclass2) --> (value1, value2)
        self.forceSubtableBreak_ = False
        self.subtables_ = []

    def addPair(self, gc1, value1, gc2, value2):
        """Add a pair positioning rule.

        Args:
            gc1: A set of glyph names for the "left" glyph
            value1: An ``otTables.ValueRecord`` object for the left glyph's
                positioning.
            gc2: A set of glyph names for the "right" glyph
            value2: An ``otTables.ValueRecord`` object for the right glyph's
                positioning.
        """
        mergeable = (
            not self.forceSubtableBreak_
            and self.classDef1_ is not None
            and self.classDef1_.canAdd(gc1)
            and self.classDef2_ is not None
            and self.classDef2_.canAdd(gc2)
        )
        if not mergeable:
            self.flush_()
            self.classDef1_ = ClassDefBuilder(useClass0=True)
            self.classDef2_ = ClassDefBuilder(useClass0=False)
            self.values_ = {}
        self.classDef1_.add(gc1)
        self.classDef2_.add(gc2)
        self.values_[(gc1, gc2)] = (value1, value2)

    def addSubtableBreak(self):
        """Add an explicit subtable break at this point."""
        self.forceSubtableBreak_ = True

    def subtables(self):
        """Return the list of ``otTables.PairPos`` subtables constructed."""
        self.flush_()
        return self.subtables_

    def flush_(self):
        if self.classDef1_ is None or self.classDef2_ is None:
            return
        st = buildPairPosClassesSubtable(self.values_, self.builder_.glyphMap)
        if st.Coverage is None:
            return
        self.subtables_.append(st)
        self.forceSubtableBreak_ = False


class PairPosBuilder(LookupBuilder):
    """Builds a Pair Positioning (GPOS2) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        pairs: An array of class-based pair positioning tuples. Usually
            manipulated with the :meth:`addClassPair` method below.
        glyphPairs: A dictionary mapping a tuple of glyph names to a tuple
            of ``otTables.ValueRecord`` objects. Usually manipulated with the
            :meth:`addGlyphPair` method below.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 2)
        self.pairs = []  # [(gc1, value1, gc2, value2)*]
        self.glyphPairs = {}  # (glyph1, glyph2) --> (value1, value2)
        self.locations = {}  # (gc1, gc2) --> (filepath, line, column)

    def addClassPair(self, location, glyphclass1, value1, glyphclass2, value2):
        """Add a class pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule. Unused.
            glyphclass1: A set of glyph names for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyphclass2: A set of glyph names for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        self.pairs.append((glyphclass1, value1, glyphclass2, value2))

    def addGlyphPair(self, location, glyph1, value1, glyph2, value2):
        """Add a glyph pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule.
            glyph1: A glyph name for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyph2: A glyph name for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        key = (glyph1, glyph2)
        oldValue = self.glyphPairs.get(key, None)
        if oldValue is not None:
            # the Feature File spec explicitly allows specific pairs generated
            # by an 'enum' rule to be overridden by preceding single pairs
            otherLoc = self.locations[key]
            log.debug(
                "Already defined position for pair %s %s at %s; "
                "choosing the first value",
                glyph1,
                glyph2,
                otherLoc,
            )
        else:
            self.glyphPairs[key] = (value1, value2)
            self.locations[key] = location

    def add_subtable_break(self, location):
        self.pairs.append(
            (
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
            )
        )

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.glyphPairs == other.glyphPairs
            and self.pairs == other.pairs
        )

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the pair positioning
            lookup.
        """
        builders = {}
        builder = ClassPairPosSubtableBuilder(self)
        for glyphclass1, value1, glyphclass2, value2 in self.pairs:
            if glyphclass1 is self.SUBTABLE_BREAK_:
                builder.addSubtableBreak()
                continue
            builder.addPair(glyphclass1, value1, glyphclass2, value2)
        subtables = []
        if self.glyphPairs:
            subtables.extend(buildPairPosGlyphs(self.glyphPairs, self.glyphMap))
        subtables.extend(builder.subtables())
        lookup = self.buildLookup_(subtables)

        # Compact the lookup
        # This is a good moment to do it because the compaction should create
        # smaller subtables, which may prevent overflows from happening.
        # Keep reading the value from the ENV until ufo2ft switches to the config system
        level = self.font.cfg.get(
            "fontTools.otlLib.optimize.gpos:COMPRESSION_LEVEL",
            default=_compression_level_from_env(),
        )
        if level != 0:
            log.info("Compacting GPOS...")
            compact_lookup(self.font, level, lookup)

        return lookup


class SinglePosBuilder(LookupBuilder):
    """Builds a Single Positioning (GPOS1) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: A dictionary mapping a glyph name to a ``otTables.ValueRecord``
            objects. Usually manipulated with the :meth:`add_pos` method below.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 1)
        self.locations = {}  # glyph -> (filename, line, column)
        self.mapping = {}  # glyph -> ot.ValueRecord

    def add_pos(self, location, glyph, otValueRecord):
        """Add a single positioning rule.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup.
            glyph: A glyph name.
            otValueRection: A ``otTables.ValueRecord`` used to position the
                glyph.
        """
        if otValueRecord is None:
            otValueRecord = ValueRecord()
        if not self.can_add(glyph, otValueRecord):
            otherLoc = self.locations[glyph]
            raise OpenTypeLibError(
                'Already defined different position for glyph "%s" at %s'
                % (glyph, otherLoc),
                location,
            )
        if otValueRecord:
            self.mapping[glyph] = otValueRecord
        self.locations[glyph] = location

    def can_add(self, glyph, value):
        assert isinstance(value, ValueRecord)
        curValue = self.mapping.get(glyph)
        return curValue is None or curValue == value

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the single positioning
            lookup.
        """
        subtables = buildSinglePos(self.mapping, self.glyphMap)
        return self.buildLookup_(subtables)


# GSUB


def buildSingleSubstSubtable(mapping):
    """Builds a single substitution (GSUB1) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SingleSubstBuilder` instead.

    Args:
        mapping: A dictionary mapping input glyph names to output glyph names.

    Returns:
        An ``otTables.SingleSubst`` object, or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.SingleSubst()
    self.mapping = dict(mapping)
    return self


def buildMultipleSubstSubtable(mapping):
    """Builds a multiple substitution (GSUB2) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MultipleSubstBuilder` instead.

    Example::

        # sub uni06C0 by uni06D5.fina hamza.above
        # sub uni06C2 by uni06C1.fina hamza.above;

        subtable = buildMultipleSubstSubtable({
            "uni06C0": [ "uni06D5.fina", "hamza.above"],
            "uni06C2": [ "uni06D1.fina", "hamza.above"]
        })

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.MultipleSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.MultipleSubst()
    self.mapping = dict(mapping)
    return self


def buildAlternateSubstSubtable(mapping):
    """Builds an alternate substitution (GSUB3) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.AlternateSubstBuilder` instead.

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.AlternateSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.AlternateSubst()
    self.alternates = dict(mapping)
    return self


def buildLigatureSubstSubtable(mapping):
    """Builds a ligature substitution (GSUB4) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.LigatureSubstBuilder` instead.

    Example::

        # sub f f i by f_f_i;
        # sub f i by f_i;

        subtable = buildLigatureSubstSubtable({
            ("f", "f", "i"): "f_f_i",
            ("f", "i"): "f_i",
        })

    Args:
        mapping: A dictionary mapping tuples of glyph names to output
            glyph names.

    Returns:
        An ``otTables.LigatureSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """

    if not mapping:
        return None
    self = ot.LigatureSubst()
    # The following single line can replace the rest of this function
    # with fontTools >= 3.1:
    # self.ligatures = dict(mapping)
    self.ligatures = {}
    for components in sorted(mapping.keys(), key=self._getLigatureSortKey):
        ligature = ot.Ligature()
        ligature.Component = components[1:]
        ligature.CompCount = len(ligature.Component) + 1
        ligature.LigGlyph = mapping[components]
        firstGlyph = components[0]
        self.ligatures.setdefault(firstGlyph, []).append(ligature)
    return self


# GPOS


def buildAnchor(x, y, point=None, deviceX=None, deviceY=None):
    """Builds an Anchor table.

    This determines the appropriate anchor format based on the passed parameters.

    Args:
        x (int): X coordinate.
        y (int): Y coordinate.
        point (int): Index of glyph contour point, if provided.
        deviceX (``otTables.Device``): X coordinate device table, if provided.
        deviceY (``otTables.Device``): Y coordinate device table, if provided.

    Returns:
        An ``otTables.Anchor`` object.
    """
    self = ot.Anchor()
    self.XCoordinate, self.YCoordinate = x, y
    self.Format = 1
    if point is not None:
        self.AnchorPoint = point
        self.Format = 2
    if deviceX is not None or deviceY is not None:
        assert (
            self.Format == 1
        ), "Either point, or both of deviceX/deviceY, must be None."
        self.XDeviceTable = deviceX
        self.YDeviceTable = deviceY
        self.Format = 3
    return self


def buildBaseArray(bases, numMarkClasses, glyphMap):
    """Builds a base array record.

    As part of building mark-to-base positioning rules, you will need to define
    a ``BaseArray`` record, which "defines for each base glyph an array of
    anchors, one for each mark class." This function builds the base array
    subtable.

    Example::

        bases = {"a": {0: a3, 1: a5}, "b": {0: a4, 1: a5}}
        basearray = buildBaseArray(bases, 2, font.getReverseGlyphMap())

    Args:
        bases (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being dictionaries mapping mark class ID
            to the appropriate ``otTables.Anchor`` object used for attaching marks
            of that class.
        numMarkClasses (int): The total number of mark classes for which anchors
            are defined.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.BaseArray`` object.
    """
    self = ot.BaseArray()
    self.BaseRecord = []
    for base in sorted(bases, key=glyphMap.__getitem__):
        b = bases[base]
        anchors = [b.get(markClass) for markClass in range(numMarkClasses)]
        self.BaseRecord.append(buildBaseRecord(anchors))
    self.BaseCount = len(self.BaseRecord)
    return self


def buildBaseRecord(anchors):
    # [otTables.Anchor, otTables.Anchor, ...] --> otTables.BaseRecord
    self = ot.BaseRecord()
    self.BaseAnchor = anchors
    return self


def buildComponentRecord(anchors):
    """Builds a component record.

    As part of building mark-to-ligature positioning rules, you will need to
    define ``ComponentRecord`` objects, which contain "an array of offsets...
    to the Anchor tables that define all the attachment points used to attach
    marks to the component." This function builds the component record.

    Args:
        anchors: A list of ``otTables.Anchor`` objects or ``None``.

    Returns:
        A ``otTables.ComponentRecord`` object or ``None`` if no anchors are
        supplied.
    """
    if not anchors:
        return None
    self = ot.ComponentRecord()
    self.LigatureAnchor = anchors
    return self


def buildCursivePosSubtable(attach, glyphMap):
    """Builds a cursive positioning (GPOS3) subtable.

    Cursive positioning lookups are made up of a coverage table of glyphs,
    and a set of ``EntryExitRecord`` records containing the anchors for
    each glyph. This function builds the cursive positioning subtable.

    Example::

        subtable = buildCursivePosSubtable({
            "AlifIni": (None, buildAnchor(0, 50)),
            "BehMed": (buildAnchor(500,250), buildAnchor(0,50)),
            # ...
        }, font.getReverseGlyphMap())

    Args:
        attach (dict): A mapping between glyph names and a tuple of two
            ``otTables.Anchor`` objects representing entry and exit anchors.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.CursivePos`` object, or ``None`` if the attachment
        dictionary was empty.
    """
    if not attach:
        return None
    self = ot.CursivePos()
    self.Format = 1
    self.Coverage = buildCoverage(attach.keys(), glyphMap)
    self.EntryExitRecord = []
    for glyph in self.Coverage.glyphs:
        entryAnchor, exitAnchor = attach[glyph]
        rec = ot.EntryExitRecord()
        rec.EntryAnchor = entryAnchor
        rec.ExitAnchor = exitAnchor
        self.EntryExitRecord.append(rec)
    self.EntryExitCount = len(self.EntryExitRecord)
    return self


def buildDevice(deltas):
    """Builds a Device record as part of a ValueRecord or Anchor.

    Device tables specify size-specific adjustments to value records
    and anchors to reflect changes based on the resolution of the output.
    For example, one could specify that an anchor's Y position should be
    increased by 1 pixel when displayed at 8 pixels per em. This routine
    builds device records.

    Args:
        deltas: A dictionary mapping pixels-per-em sizes to the delta
            adjustment in pixels when the font is displayed at that size.

    Returns:
        An ``otTables.Device`` object if any deltas were supplied, or
        ``None`` otherwise.
    """
    if not deltas:
        return None
    self = ot.Device()
    keys = deltas.keys()
    self.StartSize = startSize = min(keys)
    self.EndSize = endSize = max(keys)
    assert 0 <= startSize <= endSize
    self.DeltaValue = deltaValues = [
        deltas.get(size, 0) for size in range(startSize, endSize + 1)
    ]
    maxDelta = max(deltaValues)
    minDelta = min(deltaValues)
    assert minDelta > -129 and maxDelta < 128
    if minDelta > -3 and maxDelta < 2:
        self.DeltaFormat = 1
    elif minDelta > -9 and maxDelta < 8:
        self.DeltaFormat = 2
    else:
        self.DeltaFormat = 3
    return self


def buildLigatureArray(ligs, numMarkClasses, glyphMap):
    """Builds a LigatureArray subtable.

    As part of building a mark-to-ligature lookup, you will need to define
    the set of anchors (for each mark class) on each component of the ligature
    where marks can be attached. For example, for an Arabic divine name ligature
    (lam lam heh), you may want to specify mark attachment positioning for
    superior marks (fatha, etc.) and inferior marks (kasra, etc.) on each glyph
    of the ligature. This routine builds the ligature array record.

    Example::

        buildLigatureArray({
            "lam-lam-heh": [
                { 0: superiorAnchor1, 1: inferiorAnchor1 }, # attach points for lam1
                { 0: superiorAnchor2, 1: inferiorAnchor2 }, # attach points for lam2
                { 0: superiorAnchor3, 1: inferiorAnchor3 }, # attach points for heh
            ]
        }, 2, font.getReverseGlyphMap())

    Args:
        ligs (dict): A mapping of ligature names to an array of dictionaries:
            for each component glyph in the ligature, an dictionary mapping
            mark class IDs to anchors.
        numMarkClasses (int): The number of mark classes.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.LigatureArray`` object if deltas were supplied.
    """
    self = ot.LigatureArray()
    self.LigatureAttach = []
    for lig in sorted(ligs, key=glyphMap.__getitem__):
        anchors = []
        for component in ligs[lig]:
            anchors.append([component.get(mc) for mc in range(numMarkClasses)])
        self.LigatureAttach.append(buildLigatureAttach(anchors))
    self.LigatureCount = len(self.LigatureAttach)
    return self


def buildLigatureAttach(components):
    # [[Anchor, Anchor], [Anchor, Anchor, Anchor]] --> LigatureAttach
    self = ot.LigatureAttach()
    self.ComponentRecord = [buildComponentRecord(c) for c in components]
    self.ComponentCount = len(self.ComponentRecord)
    return self


def buildMarkArray(marks, glyphMap):
    """Builds a mark array subtable.

    As part of building mark-to-* positioning rules, you will need to define
    a MarkArray subtable, which "defines the class and the anchor point
    for a mark glyph." This function builds the mark array subtable.

    Example::

        mark = {
            "acute": (0, buildAnchor(300,712)),
            # ...
        }
        markarray = buildMarkArray(marks, font.getReverseGlyphMap())

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.MarkArray`` object.
    """
    self = ot.MarkArray()
    self.MarkRecord = []
    for mark in sorted(marks.keys(), key=glyphMap.__getitem__):
        markClass, anchor = marks[mark]
        markrec = buildMarkRecord(markClass, anchor)
        self.MarkRecord.append(markrec)
    self.MarkCount = len(self.MarkRecord)
    return self


@deprecateFunction(
    "use buildMarkBasePosSubtable() instead", category=DeprecationWarning
)
def buildMarkBasePos(marks, bases, glyphMap):
    """Build a list of MarkBasePos (GPOS4) subtables.

    .. deprecated:: 4.58.0
           Use :func:`buildMarkBasePosSubtable` instead.
    """
    return [buildMarkBasePosSubtable(marks, bases, glyphMap)]


def buildMarkBasePosSubtable(marks, bases, glyphMap):
    """Build a single MarkBasePos (GPOS4) subtable.

    This builds a mark-to-base lookup subtable containing all of the referenced
    marks and bases.

    Example::

        # a1, a2, a3, a4, a5 = buildAnchor(500, 100), ...

        marks = {"acute": (0, a1), "grave": (0, a1), "cedilla": (1, a2)}
        bases = {"a": {0: a3, 1: a5}, "b": {0: a4, 1: a5}}
        markbaseposes = [buildMarkBasePosSubtable(marks, bases, font.getReverseGlyphMap())]

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        bases (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being dictionaries mapping mark class ID
            to the appropriate ``otTables.Anchor`` object used for attaching marks
            of that class. (See :func:`buildBaseArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.MarkBasePos`` object.
    """
    self = ot.MarkBasePos()
    self.Format = 1
    self.MarkCoverage = buildCoverage(marks, glyphMap)
    self.MarkArray = buildMarkArray(marks, glyphMap)
    self.ClassCount = max([mc for mc, _ in marks.values()]) + 1
    self.BaseCoverage = buildCoverage(bases, glyphMap)
    self.BaseArray = buildBaseArray(bases, self.ClassCount, glyphMap)
    return self


@deprecateFunction("use buildMarkLigPosSubtable() instead", category=DeprecationWarning)
def buildMarkLigPos(marks, ligs, glyphMap):
    """Build a list of MarkLigPos (GPOS5) subtables.

    .. deprecated:: 4.58.0
       Use :func:`buildMarkLigPosSubtable` instead.
    """
    return [buildMarkLigPosSubtable(marks, ligs, glyphMap)]


def buildMarkLigPosSubtable(marks, ligs, glyphMap):
    """Build a single MarkLigPos (GPOS5) subtable.

    This builds a mark-to-base lookup subtable containing all of the referenced
    marks and bases.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MarkLigPosBuilder` instead.

    Example::

        # a1, a2, a3, a4, a5 = buildAnchor(500, 100), ...
        marks = {
            "acute": (0, a1),
            "grave": (0, a1),
            "cedilla": (1, a2)
        }
        ligs = {
            "f_i": [
                { 0: a3, 1: a5 }, # f
                { 0: a4, 1: a5 }  # i
                ],
        #   "c_t": [{...}, {...}]
        }
        markligpose = buildMarkLigPosSubtable(marks, ligs,
            font.getReverseGlyphMap())

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        ligs (dict): A mapping of ligature names to an array of dictionaries:
            for each component glyph in the ligature, an dictionary mapping
            mark class IDs to anchors. (See :func:`buildLigatureArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.MarkLigPos`` object.
    """
    self = ot.MarkLigPos()
    self.Format = 1
    self.MarkCoverage = buildCoverage(marks, glyphMap)
    self.MarkArray = buildMarkArray(marks, glyphMap)
    self.ClassCount = max([mc for mc, _ in marks.values()]) + 1
    self.LigatureCoverage = buildCoverage(ligs, glyphMap)
    self.LigatureArray = buildLigatureArray(ligs, self.ClassCount, glyphMap)
    return self


def buildMarkRecord(classID, anchor):
    assert isinstance(classID, int)
    assert isinstance(anchor, ot.Anchor)
    self = ot.MarkRecord()
    self.Class = classID
    self.MarkAnchor = anchor
    return self


def buildMark2Record(anchors):
    # [otTables.Anchor, otTables.Anchor, ...] --> otTables.Mark2Record
    self = ot.Mark2Record()
    self.Mark2Anchor = anchors
    return self


def _getValueFormat(f, values, i):
    # Helper for buildPairPos{Glyphs|Classes}Subtable.
    if f is not None:
        return f
    mask = 0
    for value in values:
        if value is not None and value[i] is not None:
            mask |= value[i].getFormat()
    return mask


def buildPairPosClassesSubtable(pairs, glyphMap, valueFormat1=None, valueFormat2=None):
    """Builds a class pair adjustment (GPOS2 format 2) subtable.

    Kerning tables are generally expressed as pair positioning tables using
    class-based pair adjustments. This routine builds format 2 PairPos
    subtables.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.ClassPairPosSubtableBuilder`
    instead, as this takes care of ensuring that the supplied pairs can be
    formed into non-overlapping classes and emitting individual subtables
    whenever the non-overlapping requirement means that a new subtable is
    required.

    Example::

        pairs = {}

        pairs[(
            [ "K", "X" ],
            [ "W", "V" ]
        )] = ( buildValue(xAdvance=+5), buildValue() )
        # pairs[(... , ...)] = (..., ...)

        pairpos = buildPairPosClassesSubtable(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of lists of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.
        valueFormat1: Force the "left" value records to the given format.
        valueFormat2: Force the "right" value records to the given format.

    Returns:
        A ``otTables.PairPos`` object.
    """
    coverage = set()
    classDef1 = ClassDefBuilder(useClass0=True)
    classDef2 = ClassDefBuilder(useClass0=False)
    for gc1, gc2 in sorted(pairs):
        coverage.update(gc1)
        classDef1.add(gc1)
        classDef2.add(gc2)
    self = ot.PairPos()
    self.Format = 2
    valueFormat1 = self.ValueFormat1 = _getValueFormat(valueFormat1, pairs.values(), 0)
    valueFormat2 = self.ValueFormat2 = _getValueFormat(valueFormat2, pairs.values(), 1)
    self.Coverage = buildCoverage(coverage, glyphMap)
    self.ClassDef1 = classDef1.build()
    self.ClassDef2 = classDef2.build()
    classes1 = classDef1.classes()
    classes2 = classDef2.classes()
    self.Class1Record = []
    for c1 in classes1:
        rec1 = ot.Class1Record()
        rec1.Class2Record = []
        self.Class1Record.append(rec1)
        for c2 in classes2:
            rec2 = ot.Class2Record()
            val1, val2 = pairs.get((c1, c2), (None, None))
            rec2.Value1 = (
                ValueRecord(src=val1, valueFormat=valueFormat1)
                if valueFormat1
                else None
            )
            rec2.Value2 = (
                ValueRecord(src=val2, valueFormat=valueFormat2)
                if valueFormat2
                else None
            )
            rec1.Class2Record.append(rec2)
    self.Class1Count = len(self.Class1Record)
    self.Class2Count = len(classes2)
    return self


def buildPairPosGlyphs(pairs, glyphMap):
    """Builds a list of glyph-based pair adjustment (GPOS2 format 1) subtables.

    This organises a list of pair positioning adjustments into subtables based
    on common value record formats.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.PairPosBuilder`
    instead.

    Example::

        pairs = {
            ("K", "W"): ( buildValue(xAdvance=+5), buildValue() ),
            ("K", "V"): ( buildValue(xAdvance=+5), buildValue() ),
            # ...
        }

        subtables = buildPairPosGlyphs(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.PairPos`` objects.
    """

    p = {}  # (formatA, formatB) --> {(glyphA, glyphB): (valA, valB)}
    for (glyphA, glyphB), (valA, valB) in pairs.items():
        formatA = valA.getFormat() if valA is not None else 0
        formatB = valB.getFormat() if valB is not None else 0
        pos = p.setdefault((formatA, formatB), {})
        pos[(glyphA, glyphB)] = (valA, valB)
    return [
        buildPairPosGlyphsSubtable(pos, glyphMap, formatA, formatB)
        for ((formatA, formatB), pos) in sorted(p.items())
    ]


def buildPairPosGlyphsSubtable(pairs, glyphMap, valueFormat1=None, valueFormat2=None):
    """Builds a single glyph-based pair adjustment (GPOS2 format 1) subtable.

    This builds a PairPos subtable from a dictionary of glyph pairs and
    their positioning adjustments. See also :func:`buildPairPosGlyphs`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.PairPosBuilder` instead.

    Example::

        pairs = {
            ("K", "W"): ( buildValue(xAdvance=+5), buildValue() ),
            ("K", "V"): ( buildValue(xAdvance=+5), buildValue() ),
            # ...
        }

        pairpos = buildPairPosGlyphsSubtable(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.
        valueFormat1: Force the "left" value records to the given format.
        valueFormat2: Force the "right" value records to the given format.

    Returns:
        A ``otTables.PairPos`` object.
    """
    self = ot.PairPos()
    self.Format = 1
    valueFormat1 = self.ValueFormat1 = _getValueFormat(valueFormat1, pairs.values(), 0)
    valueFormat2 = self.ValueFormat2 = _getValueFormat(valueFormat2, pairs.values(), 1)
    p = {}
    for (glyphA, glyphB), (valA, valB) in pairs.items():
        p.setdefault(glyphA, []).append((glyphB, valA, valB))
    self.Coverage = buildCoverage({g for g, _ in pairs.keys()}, glyphMap)
    self.PairSet = []
    for glyph in self.Coverage.glyphs:
        ps = ot.PairSet()
        ps.PairValueRecord = []
        self.PairSet.append(ps)
        for glyph2, val1, val2 in sorted(p[glyph], key=lambda x: glyphMap[x[0]]):
            pvr = ot.PairValueRecord()
            pvr.SecondGlyph = glyph2
            pvr.Value1 = (
                ValueRecord(src=val1, valueFormat=valueFormat1)
                if valueFormat1
                else None
            )
            pvr.Value2 = (
                ValueRecord(src=val2, valueFormat=valueFormat2)
                if valueFormat2
                else None
            )
            ps.PairValueRecord.append(pvr)
        ps.PairValueCount = len(ps.PairValueRecord)
    self.PairSetCount = len(self.PairSet)
    return self


def buildSinglePos(mapping, glyphMap):
    """Builds a list of single adjustment (GPOS1) subtables.

    This builds a list of SinglePos subtables from a dictionary of glyph
    names and their positioning adjustments. The format of the subtables are
    determined to optimize the size of the resulting subtables.
    See also :func:`buildSinglePosSubtable`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SinglePosBuilder` instead.

    Example::

        mapping = {
            "V": buildValue({ "xAdvance" : +5 }),
            # ...
        }

        subtables = buildSinglePos(pairs, font.getReverseGlyphMap())

    Args:
        mapping (dict): A mapping between glyphnames and
            ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.SinglePos`` objects.
    """
    result, handled = [], set()
    # In SinglePos format 1, the covered glyphs all share the same ValueRecord.
    # In format 2, each glyph has its own ValueRecord, but these records
    # all have the same properties (eg., all have an X but no Y placement).
    coverages, masks, values = {}, {}, {}
    for glyph, value in mapping.items():
        key = _getSinglePosValueKey(value)
        coverages.setdefault(key, []).append(glyph)
        masks.setdefault(key[0], []).append(key)
        values[key] = value

    # If a ValueRecord is shared between multiple glyphs, we generate
    # a SinglePos format 1 subtable; that is the most compact form.
    for key, glyphs in coverages.items():
        # 5 ushorts is the length of introducing another sublookup
        if len(glyphs) * _getSinglePosValueSize(key) > 5:
            format1Mapping = {g: values[key] for g in glyphs}
            result.append(buildSinglePosSubtable(format1Mapping, glyphMap))
            handled.add(key)

    # In the remaining ValueRecords, look for those whose valueFormat
    # (the set of used properties) is shared between multiple records.
    # These will get encoded in format 2.
    for valueFormat, keys in masks.items():
        f2 = [k for k in keys if k not in handled]
        if len(f2) > 1:
            format2Mapping = {}
            for k in f2:
                format2Mapping.update((g, values[k]) for g in coverages[k])
            result.append(buildSinglePosSubtable(format2Mapping, glyphMap))
            handled.update(f2)

    # The remaining ValueRecords are only used by a few glyphs, normally
    # one. We encode these in format 1 again.
    for key, glyphs in coverages.items():
        if key not in handled:
            for g in glyphs:
                st = buildSinglePosSubtable({g: values[key]}, glyphMap)
            result.append(st)

    # When the OpenType layout engine traverses the subtables, it will
    # stop after the first matching subtable.  Therefore, we sort the
    # resulting subtables by decreasing coverage size; this increases
    # the chance that the layout engine can do an early exit. (Of course,
    # this would only be true if all glyphs were equally frequent, which
    # is not really the case; but we do not know their distribution).
    # If two subtables cover the same number of glyphs, we sort them
    # by glyph ID so that our output is deterministic.
    result.sort(key=lambda t: _getSinglePosTableKey(t, glyphMap))
    return result


def buildSinglePosSubtable(values, glyphMap):
    """Builds a single adjustment (GPOS1) subtable.

    This builds a list of SinglePos subtables from a dictionary of glyph
    names and their positioning adjustments. The format of the subtable is
    determined to optimize the size of the output.
    See also :func:`buildSinglePos`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SinglePosBuilder` instead.

    Example::

        mapping = {
            "V": buildValue({ "xAdvance" : +5 }),
            # ...
        }

        subtable = buildSinglePos(pairs, font.getReverseGlyphMap())

    Args:
        mapping (dict): A mapping between glyphnames and
            ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.SinglePos`` object.
    """
    self = ot.SinglePos()
    self.Coverage = buildCoverage(values.keys(), glyphMap)
    valueFormat = self.ValueFormat = reduce(
        int.__or__, [v.getFormat() for v in values.values()], 0
    )
    valueRecords = [
        ValueRecord(src=values[g], valueFormat=valueFormat)
        for g in self.Coverage.glyphs
    ]
    if all(v == valueRecords[0] for v in valueRecords):
        self.Format = 1
        if self.ValueFormat != 0:
            self.Value = valueRecords[0]
        else:
            self.Value = None
    else:
        self.Format = 2
        self.Value = valueRecords
        self.ValueCount = len(self.Value)
    return self


def _getSinglePosTableKey(subtable, glyphMap):
    assert isinstance(subtable, ot.SinglePos), subtable
    glyphs = subtable.Coverage.glyphs
    return (-len(glyphs), glyphMap[glyphs[0]])


def _getSinglePosValueKey(valueRecord):
    # otBase.ValueRecord --> (2, ("YPlacement": 12))
    assert isinstance(valueRecord, ValueRecord), valueRecord
    valueFormat, result = 0, []
    for name, value in valueRecord.__dict__.items():
        if isinstance(value, ot.Device):
            result.append((name, _makeDeviceTuple(value)))
        else:
            result.append((name, value))
        valueFormat |= valueRecordFormatDict[name][0]
    result.sort()
    result.insert(0, valueFormat)
    return tuple(result)


_DeviceTuple = namedtuple("_DeviceTuple", "DeltaFormat StartSize EndSize DeltaValue")


def _makeDeviceTuple(device):
    # otTables.Device --> tuple, for making device tables unique
    return _DeviceTuple(
        device.DeltaFormat,
        device.StartSize,
        device.EndSize,
        () if device.DeltaFormat & 0x8000 else tuple(device.DeltaValue),
    )


def _getSinglePosValueSize(valueKey):
    # Returns how many ushorts this valueKey (short form of ValueRecord) takes up
    count = 0
    for _, v in valueKey[1:]:
        if isinstance(v, _DeviceTuple):
            count += len(v.DeltaValue) + 3
        else:
            count += 1
    return count


def buildValue(value):
    """Builds a positioning value record.

    Value records are used to specify coordinates and adjustments for
    positioning and attaching glyphs. Many of the positioning functions
    in this library take ``otTables.ValueRecord`` objects as arguments.
    This function builds value records from dictionaries.

    Args:
        value (dict): A dictionary with zero or more of the following keys:
            - ``xPlacement``
            - ``yPlacement``
            - ``xAdvance``
            - ``yAdvance``
            - ``xPlaDevice``
            - ``yPlaDevice``
            - ``xAdvDevice``
            - ``yAdvDevice``

    Returns:
        An ``otTables.ValueRecord`` object.
    """
    self = ValueRecord()
    for k, v in value.items():
        setattr(self, k, v)
    return self


# GDEF


def buildAttachList(attachPoints, glyphMap):
    """Builds an AttachList subtable.

    A GDEF table may contain an Attachment Point List table (AttachList)
    which stores the contour indices of attachment points for glyphs with
    attachment points. This routine builds AttachList subtables.

    Args:
        attachPoints (dict): A mapping between glyph names and a list of
            contour indices.

    Returns:
        An ``otTables.AttachList`` object if attachment points are supplied,
            or ``None`` otherwise.
    """
    if not attachPoints:
        return None
    self = ot.AttachList()
    self.Coverage = buildCoverage(attachPoints.keys(), glyphMap)
    self.AttachPoint = [buildAttachPoint(attachPoints[g]) for g in self.Coverage.glyphs]
    self.GlyphCount = len(self.AttachPoint)
    return self


def buildAttachPoint(points):
    # [4, 23, 41] --> otTables.AttachPoint
    # Only used by above.
    if not points:
        return None
    self = ot.AttachPoint()
    self.PointIndex = sorted(set(points))
    self.PointCount = len(self.PointIndex)
    return self


def buildCaretValueForCoord(coord):
    # 500 --> otTables.CaretValue, format 1
    # (500, DeviceTable) --> otTables.CaretValue, format 3
    self = ot.CaretValue()
    if isinstance(coord, tuple):
        self.Format = 3
        self.Coordinate, self.DeviceTable = coord
    else:
        self.Format = 1
        self.Coordinate = coord
    return self


def buildCaretValueForPoint(point):
    # 4 --> otTables.CaretValue, format 2
    self = ot.CaretValue()
    self.Format = 2
    self.CaretValuePoint = point
    return self


def buildLigCaretList(coords, points, glyphMap):
    """Builds a ligature caret list table.

    Ligatures appear as a single glyph representing multiple characters; however
    when, for example, editing text containing a ``f_i`` ligature, the user may
    want to place the cursor between the ``f`` and the ``i``. The ligature caret
    list in the GDEF table specifies the position to display the "caret" (the
    character insertion indicator, typically a flashing vertical bar) "inside"
    the ligature to represent an insertion point. The insertion positions may
    be specified either by coordinate or by contour point.

    Example::

        coords = {
            "f_f_i": [300, 600] # f|fi cursor at 300 units, ff|i cursor at 600.
        }
        points = {
            "c_t": [28] # c|t cursor appears at coordinate of contour point 28.
        }
        ligcaretlist = buildLigCaretList(coords, points, font.getReverseGlyphMap())

    Args:
        coords: A mapping between glyph names and a list of coordinates for
            the insertion point of each ligature component after the first one.
        points: A mapping between glyph names and a list of contour points for
            the insertion point of each ligature component after the first one.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.LigCaretList`` object if any carets are present, or
            ``None`` otherwise."""
    glyphs = set(coords.keys()) if coords else set()
    if points:
        glyphs.update(points.keys())
    carets = {g: buildLigGlyph(coords.get(g), points.get(g)) for g in glyphs}
    carets = {g: c for g, c in carets.items() if c is not None}
    if not carets:
        return None
    self = ot.LigCaretList()
    self.Coverage = buildCoverage(carets.keys(), glyphMap)
    self.LigGlyph = [carets[g] for g in self.Coverage.glyphs]
    self.LigGlyphCount = len(self.LigGlyph)
    return self


def buildLigGlyph(coords, points):
    # ([500], [4]) --> otTables.LigGlyph; None for empty coords/points
    carets = []
    if coords:
        coords = sorted(coords, key=lambda c: c[0] if isinstance(c, tuple) else c)
        carets.extend([buildCaretValueForCoord(c) for c in coords])
    if points:
        carets.extend([buildCaretValueForPoint(p) for p in sorted(points)])
    if not carets:
        return None
    self = ot.LigGlyph()
    self.CaretValue = carets
    self.CaretCount = len(self.CaretValue)
    return self


def buildMarkGlyphSetsDef(markSets, glyphMap):
    """Builds a mark glyph sets definition table.

    OpenType Layout lookups may choose to use mark filtering sets to consider
    or ignore particular combinations of marks. These sets are specified by
    setting a flag on the lookup, but the mark filtering sets are defined in
    the ``GDEF`` table. This routine builds the subtable containing the mark
    glyph set definitions.

    Example::

        set0 = set("acute", "grave")
        set1 = set("caron", "grave")

        markglyphsets = buildMarkGlyphSetsDef([set0, set1], font.getReverseGlyphMap())

    Args:

        markSets: A list of sets of glyphnames.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns
        An ``otTables.MarkGlyphSetsDef`` object.
    """
    if not markSets:
        return None
    self = ot.MarkGlyphSetsDef()
    self.MarkSetTableFormat = 1
    self.Coverage = [buildCoverage(m, glyphMap) for m in markSets]
    self.MarkSetCount = len(self.Coverage)
    return self


class ClassDefBuilder(object):
    """Helper for building ClassDef tables."""

    def __init__(self, useClass0):
        self.classes_ = set()
        self.glyphs_ = {}
        self.useClass0_ = useClass0

    def canAdd(self, glyphs):
        if isinstance(glyphs, (set, frozenset)):
            glyphs = sorted(glyphs)
        glyphs = tuple(glyphs)
        if glyphs in self.classes_:
            return True
        for glyph in glyphs:
            if glyph in self.glyphs_:
                return False
        return True

    def add(self, glyphs):
        if isinstance(glyphs, (set, frozenset)):
            glyphs = sorted(glyphs)
        glyphs = tuple(glyphs)
        if glyphs in self.classes_:
            return
        self.classes_.add(glyphs)
        for glyph in glyphs:
            if glyph in self.glyphs_:
                raise OpenTypeLibError(
                    f"Glyph {glyph} is already present in class.", None
                )
            self.glyphs_[glyph] = glyphs

    def classes(self):
        # In ClassDef1 tables, class id #0 does not need to be encoded
        # because zero is the default. Therefore, we use id #0 for the
        # glyph class that has the largest number of members. However,
        # in other tables than ClassDef1, 0 means "every other glyph"
        # so we should not use that ID for any real glyph classes;
        # we implement this by inserting an empty set at position 0.
        #
        # TODO: Instead of counting the number of glyphs in each class,
        # we should determine the encoded size. If the glyphs in a large
        # class form a contiguous range, the encoding is actually quite
        # compact, whereas a non-contiguous set might need a lot of bytes
        # in the output file. We don't get this right with the key below.
        result = sorted(self.classes_, key=lambda s: (-len(s), s))
        if not self.useClass0_:
            result.insert(0, frozenset())
        return result

    def build(self):
        glyphClasses = {}
        for classID, glyphs in enumerate(self.classes()):
            if classID == 0:
                continue
            for glyph in glyphs:
                glyphClasses[glyph] = classID
        classDef = ot.ClassDef()
        classDef.classDefs = glyphClasses
        return classDef


AXIS_VALUE_NEGATIVE_INFINITY = fixedToFloat(-0x80000000, 16)
AXIS_VALUE_POSITIVE_INFINITY = fixedToFloat(0x7FFFFFFF, 16)

STATName = Union[int, str, Dict[str, str]]
"""A raw name ID, English name, or multilingual name."""


def buildStatTable(
    ttFont: TTFont,
    axes,
    locations=None,
    elidedFallbackName: Union[STATName, STATNameStatement] = 2,
    windowsNames: bool = True,
    macNames: bool = True,
) -> None:
    """Add a 'STAT' table to 'ttFont'.

    'axes' is a list of dictionaries describing axes and their
    values.

    Example::

        axes = [
            dict(
                tag="wght",
                name="Weight",
                ordering=0,  # optional
                values=[
                    dict(value=100, name='Thin'),
                    dict(value=300, name='Light'),
                    dict(value=400, name='Regular', flags=0x2),
                    dict(value=900, name='Black'),
                ],
            )
        ]

    Each axis dict must have 'tag' and 'name' items. 'tag' maps
    to the 'AxisTag' field. 'name' can be a name ID (int), a string,
    or a dictionary containing multilingual names (see the
    addMultilingualName() name table method), and will translate to
    the AxisNameID field.

    An axis dict may contain an 'ordering' item that maps to the
    AxisOrdering field. If omitted, the order of the axes list is
    used to calculate AxisOrdering fields.

    The axis dict may contain a 'values' item, which is a list of
    dictionaries describing AxisValue records belonging to this axis.

    Each value dict must have a 'name' item, which can be a name ID
    (int), a string, or a dictionary containing multilingual names,
    like the axis name. It translates to the ValueNameID field.

    Optionally the value dict can contain a 'flags' item. It maps to
    the AxisValue Flags field, and will be 0 when omitted.

    The format of the AxisValue is determined by the remaining contents
    of the value dictionary:

    If the value dict contains a 'value' item, an AxisValue record
    Format 1 is created. If in addition to the 'value' item it contains
    a 'linkedValue' item, an AxisValue record Format 3 is built.

    If the value dict contains a 'nominalValue' item, an AxisValue
    record Format 2 is built. Optionally it may contain 'rangeMinValue'
    and 'rangeMaxValue' items. These map to -Infinity and +Infinity
    respectively if omitted.

    You cannot specify Format 4 AxisValue tables this way, as they are
    not tied to a single axis, and specify a name for a location that
    is defined by multiple axes values. Instead, you need to supply the
    'locations' argument.

    The optional 'locations' argument specifies AxisValue Format 4
    tables. It should be a list of dicts, where each dict has a 'name'
    item, which works just like the value dicts above, an optional
    'flags' item (defaulting to 0x0), and a 'location' dict. A
    location dict key is an axis tag, and the associated value is the
    location on the specified axis. They map to the AxisIndex and Value
    fields of the AxisValueRecord.

    Example::

        locations = [
            dict(name='Regular ABCD', location=dict(wght=300, ABCD=100)),
            dict(name='Bold ABCD XYZ', location=dict(wght=600, ABCD=200)),
        ]

    The optional 'elidedFallbackName' argument can be a name ID (int),
    a string, a dictionary containing multilingual names, or a list of
    STATNameStatements. It translates to the ElidedFallbackNameID field.

    The 'ttFont' argument must be a TTFont instance that already has a
    'name' table. If a 'STAT' table already exists, it will be
    overwritten by the newly created one.
    """
    ttFont["STAT"] = ttLib.newTable("STAT")
    statTable = ttFont["STAT"].table = ot.STAT()
    statTable.ElidedFallbackNameID = _addName(
        ttFont, elidedFallbackName, windows=windowsNames, mac=macNames
    )

    # 'locations' contains data for AxisValue Format 4
    axisRecords, axisValues = _buildAxisRecords(
        axes, ttFont, windowsNames=windowsNames, macNames=macNames
    )
    if not locations:
        statTable.Version = 0x00010001
    else:
        # We'll be adding Format 4 AxisValue records, which
        # requires a higher table version
        statTable.Version = 0x00010002
        multiAxisValues = _buildAxisValuesFormat4(
            locations, axes, ttFont, windowsNames=windowsNames, macNames=macNames
        )
        axisValues = multiAxisValues + axisValues
    ttFont["name"].names.sort()

    # Store AxisRecords
    axisRecordArray = ot.AxisRecordArray()
    axisRecordArray.Axis = axisRecords
    # XXX these should not be hard-coded but computed automatically
    statTable.DesignAxisRecordSize = 8
    statTable.DesignAxisRecord = axisRecordArray
    statTable.DesignAxisCount = len(axisRecords)

    statTable.AxisValueCount = 0
    statTable.AxisValueArray = None
    if axisValues:
        # Store AxisValueRecords
        axisValueArray = ot.AxisValueArray()
        axisValueArray.AxisValue = axisValues
        statTable.AxisValueArray = axisValueArray
        statTable.AxisValueCount = len(axisValues)


def _buildAxisRecords(axes, ttFont, windowsNames=True, macNames=True):
    axisRecords = []
    axisValues = []
    for axisRecordIndex, axisDict in enumerate(axes):
        axis = ot.AxisRecord()
        axis.AxisTag = axisDict["tag"]
        axis.AxisNameID = _addName(
            ttFont, axisDict["name"], 256, windows=windowsNames, mac=macNames
        )
        axis.AxisOrdering = axisDict.get("ordering", axisRecordIndex)
        axisRecords.append(axis)

        for axisVal in axisDict.get("values", ()):
            axisValRec = ot.AxisValue()
            axisValRec.AxisIndex = axisRecordIndex
            axisValRec.Flags = axisVal.get("flags", 0)
            axisValRec.ValueNameID = _addName(
                ttFont, axisVal["name"], windows=windowsNames, mac=macNames
            )

            if "value" in axisVal:
                axisValRec.Value = axisVal["value"]
                if "linkedValue" in axisVal:
                    axisValRec.Format = 3
                    axisValRec.LinkedValue = axisVal["linkedValue"]
                else:
                    axisValRec.Format = 1
            elif "nominalValue" in axisVal:
                axisValRec.Format = 2
                axisValRec.NominalValue = axisVal["nominalValue"]
                axisValRec.RangeMinValue = axisVal.get(
                    "rangeMinValue", AXIS_VALUE_NEGATIVE_INFINITY
                )
                axisValRec.RangeMaxValue = axisVal.get(
                    "rangeMaxValue", AXIS_VALUE_POSITIVE_INFINITY
                )
            else:
                raise ValueError("Can't determine format for AxisValue")

            axisValues.append(axisValRec)
    return axisRecords, axisValues


def _buildAxisValuesFormat4(locations, axes, ttFont, windowsNames=True, macNames=True):
    axisTagToIndex = {}
    for axisRecordIndex, axisDict in enumerate(axes):
        axisTagToIndex[axisDict["tag"]] = axisRecordIndex

    axisValues = []
    for axisLocationDict in locations:
        axisValRec = ot.AxisValue()
        axisValRec.Format = 4
        axisValRec.ValueNameID = _addName(
            ttFont, axisLocationDict["name"], windows=windowsNames, mac=macNames
        )
        axisValRec.Flags = axisLocationDict.get("flags", 0)
        axisValueRecords = []
        for tag, value in axisLocationDict["location"].items():
            avr = ot.AxisValueRecord()
            avr.AxisIndex = axisTagToIndex[tag]
            avr.Value = value
            axisValueRecords.append(avr)
        axisValueRecords.sort(key=lambda avr: avr.AxisIndex)
        axisValRec.AxisCount = len(axisValueRecords)
        axisValRec.AxisValueRecord = axisValueRecords
        axisValues.append(axisValRec)
    return axisValues


def _addName(
    ttFont: TTFont,
    value: Union[STATName, STATNameStatement],
    minNameID: int = 0,
    windows: bool = True,
    mac: bool = True,
) -> int:
    nameTable = ttFont["name"]
    if isinstance(value, int):
        # Already a nameID
        return value
    if isinstance(value, str):
        names = dict(en=value)
    elif isinstance(value, dict):
        names = value
    elif isinstance(value, list):
        nameID = nameTable._findUnusedNameID()
        for nameRecord in value:
            if isinstance(nameRecord, STATNameStatement):
                nameTable.setName(
                    nameRecord.string,
                    nameID,
                    nameRecord.platformID,
                    nameRecord.platEncID,
                    nameRecord.langID,
                )
            else:
                raise TypeError("value must be a list of STATNameStatements")
        return nameID
    else:
        raise TypeError("value must be int, str, dict or list")
    return nameTable.addMultilingualName(
        names, ttFont=ttFont, windows=windows, mac=mac, minNameID=minNameID
    )


def buildMathTable(
    ttFont,
    constants=None,
    italicsCorrections=None,
    topAccentAttachments=None,
    extendedShapes=None,
    mathKerns=None,
    minConnectorOverlap=0,
    vertGlyphVariants=None,
    horizGlyphVariants=None,
    vertGlyphAssembly=None,
    horizGlyphAssembly=None,
):
    """
    Add a 'MATH' table to 'ttFont'.

    'constants' is a dictionary of math constants. The keys are the constant
    names from the MATH table specification (with capital first letter), and the
    values are the constant values as numbers.

    'italicsCorrections' is a dictionary of italic corrections. The keys are the
    glyph names, and the values are the italic corrections as numbers.

    'topAccentAttachments' is a dictionary of top accent attachments. The keys
    are the glyph names, and the values are the top accent horizontal positions
    as numbers.

    'extendedShapes' is a set of extended shape glyphs.

    'mathKerns' is a dictionary of math kerns. The keys are the glyph names, and
    the values are dictionaries. The keys of these dictionaries are the side
    names ('TopRight', 'TopLeft', 'BottomRight', 'BottomLeft'), and the values
    are tuples of two lists. The first list contains the correction heights as
    numbers, and the second list contains the kern values as numbers.

    'minConnectorOverlap' is the minimum connector overlap as a number.

    'vertGlyphVariants' is a dictionary of vertical glyph variants. The keys are
    the glyph names, and the values are tuples of glyph name and full advance height.

    'horizGlyphVariants' is a dictionary of horizontal glyph variants. The keys
    are the glyph names, and the values are tuples of glyph name and full
    advance width.

    'vertGlyphAssembly' is a dictionary of vertical glyph assemblies. The keys
    are the glyph names, and the values are tuples of assembly parts and italics
    correction. The assembly parts are tuples of glyph name, flags, start
    connector length, end connector length, and full advance height.

    'horizGlyphAssembly' is a dictionary of horizontal glyph assemblies. The
    keys are the glyph names, and the values are tuples of assembly parts
    and italics correction. The assembly parts are tuples of glyph name, flags,
    start connector length, end connector length, and full advance width.

    Where a number is expected, an integer or a float can be used. The floats
    will be rounded.

    Example::

        constants = {
            "ScriptPercentScaleDown": 70,
            "ScriptScriptPercentScaleDown": 50,
            "DelimitedSubFormulaMinHeight": 24,
            "DisplayOperatorMinHeight": 60,
            ...
        }
        italicsCorrections = {
            "fitalic-math": 100,
            "fbolditalic-math": 120,
            ...
        }
        topAccentAttachments = {
            "circumflexcomb": 500,
            "acutecomb": 400,
            "A": 300,
            "B": 340,
            ...
        }
        extendedShapes = {"parenleft", "parenright", ...}
        mathKerns = {
            "A": {
                "TopRight": ([-50, -100], [10, 20, 30]),
                "TopLeft": ([50, 100], [10, 20, 30]),
                ...
            },
            ...
        }
        vertGlyphVariants = {
            "parenleft": [("parenleft", 700), ("parenleft.size1", 1000), ...],
            "parenright": [("parenright", 700), ("parenright.size1", 1000), ...],
            ...
        }
        vertGlyphAssembly = {
            "braceleft": [
                (
                    ("braceleft.bottom", 0, 0, 200, 500),
                    ("braceleft.extender", 1, 200, 200, 200)),
                    ("braceleft.middle", 0, 100, 100, 700),
                    ("braceleft.extender", 1, 200, 200, 200),
                    ("braceleft.top", 0, 200, 0, 500),
                ),
                100,
            ],
            ...
        }
    """
    glyphMap = ttFont.getReverseGlyphMap()

    ttFont["MATH"] = math = ttLib.newTable("MATH")
    math.table = table = ot.MATH()
    table.Version = 0x00010000
    table.populateDefaults()

    table.MathConstants = _buildMathConstants(constants)
    table.MathGlyphInfo = _buildMathGlyphInfo(
        glyphMap,
        italicsCorrections,
        topAccentAttachments,
        extendedShapes,
        mathKerns,
    )
    table.MathVariants = _buildMathVariants(
        glyphMap,
        minConnectorOverlap,
        vertGlyphVariants,
        horizGlyphVariants,
        vertGlyphAssembly,
        horizGlyphAssembly,
    )


def _buildMathConstants(constants):
    if not constants:
        return None

    mathConstants = ot.MathConstants()
    for conv in mathConstants.getConverters():
        value = otRound(constants.get(conv.name, 0))
        if conv.tableClass:
            assert issubclass(conv.tableClass, ot.MathValueRecord)
            value = _mathValueRecord(value)
        setattr(mathConstants, conv.name, value)
    return mathConstants


def _buildMathGlyphInfo(
    glyphMap,
    italicsCorrections,
    topAccentAttachments,
    extendedShapes,
    mathKerns,
):
    if not any([extendedShapes, italicsCorrections, topAccentAttachments, mathKerns]):
        return None

    info = ot.MathGlyphInfo()
    info.populateDefaults()

    if italicsCorrections:
        coverage = buildCoverage(italicsCorrections.keys(), glyphMap)
        info.MathItalicsCorrectionInfo = ot.MathItalicsCorrectionInfo()
        info.MathItalicsCorrectionInfo.Coverage = coverage
        info.MathItalicsCorrectionInfo.ItalicsCorrectionCount = len(coverage.glyphs)
        info.MathItalicsCorrectionInfo.ItalicsCorrection = [
            _mathValueRecord(italicsCorrections[n]) for n in coverage.glyphs
        ]

    if topAccentAttachments:
        coverage = buildCoverage(topAccentAttachments.keys(), glyphMap)
        info.MathTopAccentAttachment = ot.MathTopAccentAttachment()
        info.MathTopAccentAttachment.TopAccentCoverage = coverage
        info.MathTopAccentAttachment.TopAccentAttachmentCount = len(coverage.glyphs)
        info.MathTopAccentAttachment.TopAccentAttachment = [
            _mathValueRecord(topAccentAttachments[n]) for n in coverage.glyphs
        ]

    if extendedShapes:
        info.ExtendedShapeCoverage = buildCoverage(extendedShapes, glyphMap)

    if mathKerns:
        coverage = buildCoverage(mathKerns.keys(), glyphMap)
        info.MathKernInfo = ot.MathKernInfo()
        info.MathKernInfo.MathKernCoverage = coverage
        info.MathKernInfo.MathKernCount = len(coverage.glyphs)
        info.MathKernInfo.MathKernInfoRecords = []
        for glyph in coverage.glyphs:
            record = ot.MathKernInfoRecord()
            for side in {"TopRight", "TopLeft", "BottomRight", "BottomLeft"}:
                if side in mathKerns[glyph]:
                    correctionHeights, kernValues = mathKerns[glyph][side]
                    assert len(correctionHeights) == len(kernValues) - 1
                    kern = ot.MathKern()
                    kern.HeightCount = len(correctionHeights)
                    kern.CorrectionHeight = [
                        _mathValueRecord(h) for h in correctionHeights
                    ]
                    kern.KernValue = [_mathValueRecord(v) for v in kernValues]
                    setattr(record, f"{side}MathKern", kern)
            info.MathKernInfo.MathKernInfoRecords.append(record)

    return info


def _buildMathVariants(
    glyphMap,
    minConnectorOverlap,
    vertGlyphVariants,
    horizGlyphVariants,
    vertGlyphAssembly,
    horizGlyphAssembly,
):
    if not any(
        [vertGlyphVariants, horizGlyphVariants, vertGlyphAssembly, horizGlyphAssembly]
    ):
        return None

    variants = ot.MathVariants()
    variants.populateDefaults()

    variants.MinConnectorOverlap = minConnectorOverlap

    if vertGlyphVariants or vertGlyphAssembly:
        variants.VertGlyphCoverage, variants.VertGlyphConstruction = (
            _buildMathGlyphConstruction(
                glyphMap,
                vertGlyphVariants,
                vertGlyphAssembly,
            )
        )

    if horizGlyphVariants or horizGlyphAssembly:
        variants.HorizGlyphCoverage, variants.HorizGlyphConstruction = (
            _buildMathGlyphConstruction(
                glyphMap,
                horizGlyphVariants,
                horizGlyphAssembly,
            )
        )

    return variants


def _buildMathGlyphConstruction(glyphMap, variants, assemblies):
    glyphs = set()
    if variants:
        glyphs.update(variants.keys())
    if assemblies:
        glyphs.update(assemblies.keys())
    coverage = buildCoverage(glyphs, glyphMap)
    constructions = []

    for glyphName in coverage.glyphs:
        construction = ot.MathGlyphConstruction()
        construction.populateDefaults()

        if variants and glyphName in variants:
            construction.VariantCount = len(variants[glyphName])
            construction.MathGlyphVariantRecord = []
            for variantName, advance in variants[glyphName]:
                record = ot.MathGlyphVariantRecord()
                record.VariantGlyph = variantName
                record.AdvanceMeasurement = otRound(advance)
                construction.MathGlyphVariantRecord.append(record)

        if assemblies and glyphName in assemblies:
            parts, ic = assemblies[glyphName]
            construction.GlyphAssembly = ot.GlyphAssembly()
            construction.GlyphAssembly.ItalicsCorrection = _mathValueRecord(ic)
            construction.GlyphAssembly.PartCount = len(parts)
            construction.GlyphAssembly.PartRecords = []
            for part in parts:
                part_name, flags, start, end, advance = part
                record = ot.GlyphPartRecord()
                record.glyph = part_name
                record.PartFlags = int(flags)
                record.StartConnectorLength = otRound(start)
                record.EndConnectorLength = otRound(end)
                record.FullAdvance = otRound(advance)
                construction.GlyphAssembly.PartRecords.append(record)

        constructions.append(construction)

    return coverage, constructions


def _mathValueRecord(value):
    value_record = ot.MathValueRecord()
    value_record.Value = otRound(value)
    return value_record
<|endoftext|>class OpenTypeLibError(Exception):
    def __init__(self, message, location):
        Exception.__init__(self, message)
        self.location = location

    def __str__(self):
        message = Exception.__str__(self)
        if self.location:
            return f"{self.location}: {message}"
        else:
            return message
<|endoftext|>__all__ = ["maxCtxFont"]


def maxCtxFont(font):
    """Calculate the usMaxContext value for an entire font."""

    maxCtx = 0
    for tag in ("GSUB", "GPOS"):
        if tag not in font:
            continue
        table = font[tag].table
        if not table.LookupList:
            continue
        for lookup in table.LookupList.Lookup:
            for st in lookup.SubTable:
                maxCtx = maxCtxSubtable(maxCtx, tag, lookup.LookupType, st)
    return maxCtx


def maxCtxSubtable(maxCtx, tag, lookupType, st):
    """Calculate usMaxContext based on a single lookup table (and an existing
    max value).
    """

    # single positioning, single / multiple substitution
    if (tag == "GPOS" and lookupType == 1) or (
        tag == "GSUB" and lookupType in (1, 2, 3)
    ):
        maxCtx = max(maxCtx, 1)

    # pair positioning
    elif tag == "GPOS" and lookupType == 2:
        maxCtx = max(maxCtx, 2)

    # ligatures
    elif tag == "GSUB" and lookupType == 4:
        for ligatures in st.ligatures.values():
            for ligature in ligatures:
                maxCtx = max(maxCtx, ligature.CompCount)

    # context
    elif (tag == "GPOS" and lookupType == 7) or (tag == "GSUB" and lookupType == 5):
        maxCtx = maxCtxContextualSubtable(maxCtx, st, "Pos" if tag == "GPOS" else "Sub")

    # chained context
    elif (tag == "GPOS" and lookupType == 8) or (tag == "GSUB" and lookupType == 6):
        maxCtx = maxCtxContextualSubtable(
            maxCtx, st, "Pos" if tag == "GPOS" else "Sub", "Chain"
        )

    # extensions
    elif (tag == "GPOS" and lookupType == 9) or (tag == "GSUB" and lookupType == 7):
        maxCtx = maxCtxSubtable(maxCtx, tag, st.ExtensionLookupType, st.ExtSubTable)

    # reverse-chained context
    elif tag == "GSUB" and lookupType == 8:
        maxCtx = maxCtxContextualRule(maxCtx, st, "Reverse")

    return maxCtx


def maxCtxContextualSubtable(maxCtx, st, ruleType, chain=""):
    """Calculate usMaxContext based on a contextual feature subtable."""

    if st.Format == 1:
        for ruleset in getattr(st, "%s%sRuleSet" % (chain, ruleType)):
            if ruleset is None:
                continue
            for rule in getattr(ruleset, "%s%sRule" % (chain, ruleType)):
                if rule is None:
                    continue
                maxCtx = maxCtxContextualRule(maxCtx, rule, chain)

    elif st.Format == 2:
        for ruleset in getattr(st, "%s%sClassSet" % (chain, ruleType)):
            if ruleset is None:
                continue
            for rule in getattr(ruleset, "%s%sClassRule" % (chain, ruleType)):
                if rule is None:
                    continue
                maxCtx = maxCtxContextualRule(maxCtx, rule, chain)

    elif st.Format == 3:
        maxCtx = maxCtxContextualRule(maxCtx, st, chain)

    return maxCtx


def maxCtxContextualRule(maxCtx, st, chain):
    """Calculate usMaxContext based on a contextual feature rule."""

    if not chain:
        return max(maxCtx, st.GlyphCount)
    elif chain == "Reverse":
        return max(maxCtx, 1 + st.LookAheadGlyphCount)
    return max(maxCtx, st.InputGlyphCount + st.LookAheadGlyphCount)
<|endoftext|>from argparse import RawTextHelpFormatter
from fontTools.otlLib.optimize.gpos import COMPRESSION_LEVEL, compact
from fontTools.ttLib import TTFont


def main(args=None):
    """Optimize the layout tables of an existing font"""
    from argparse import ArgumentParser

    from fontTools import configLogger

    parser = ArgumentParser(
        prog="otlLib.optimize",
        description=main.__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("font")
    parser.add_argument(
        "-o", metavar="OUTPUTFILE", dest="outfile", default=None, help="output file"
    )
    parser.add_argument(
        "--gpos-compression-level",
        help=COMPRESSION_LEVEL.help,
        default=COMPRESSION_LEVEL.default,
        choices=list(range(10)),
        type=int,
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
        level=("DEBUG" if options.verbose else "ERROR" if options.quiet else "INFO")
    )

    font = TTFont(options.font)
    compact(font, options.gpos_compression_level)
    font.save(options.outfile or options.font)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>import logging
import os
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import cached_property, reduce
from itertools import chain
from math import log2
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from fontTools.config import OPTIONS
from fontTools.misc.intTools import bit_count, bit_indices
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otBase, otTables

log = logging.getLogger(__name__)

COMPRESSION_LEVEL = OPTIONS[f"{__name__}:COMPRESSION_LEVEL"]

# Kept because ufo2ft depends on it, to be removed once ufo2ft uses the config instead
# https://github.com/fonttools/fonttools/issues/2592
GPOS_COMPACT_MODE_ENV_KEY = "FONTTOOLS_GPOS_COMPACT_MODE"
GPOS_COMPACT_MODE_DEFAULT = str(COMPRESSION_LEVEL.default)


def _compression_level_from_env() -> int:
    env_level = GPOS_COMPACT_MODE_DEFAULT
    if GPOS_COMPACT_MODE_ENV_KEY in os.environ:
        import warnings

        warnings.warn(
            f"'{GPOS_COMPACT_MODE_ENV_KEY}' environment variable is deprecated. "
            "Please set the 'fontTools.otlLib.optimize.gpos:COMPRESSION_LEVEL' option "
            "in TTFont.cfg.",
            DeprecationWarning,
        )

        env_level = os.environ[GPOS_COMPACT_MODE_ENV_KEY]
    if len(env_level) == 1 and env_level in "0123456789":
        return int(env_level)
    raise ValueError(f"Bad {GPOS_COMPACT_MODE_ENV_KEY}={env_level}")


def compact(font: TTFont, level: int) -> TTFont:
    # Ideal plan:
    #  1. Find lookups of Lookup Type 2: Pair Adjustment Positioning Subtable
    #     https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#lookup-type-2-pair-adjustment-positioning-subtable
    #  2. Extract glyph-glyph kerning and class-kerning from all present subtables
    #  3. Regroup into different subtable arrangements
    #  4. Put back into the lookup
    #
    # Actual implementation:
    #  2. Only class kerning is optimized currently
    #  3. If the input kerning is already in several subtables, the subtables
    #     are not grouped together first; instead each subtable is treated
    #     independently, so currently this step is:
    #     Split existing subtables into more smaller subtables
    gpos = font.get("GPOS")

    # If the font does not contain a GPOS table, there is nothing to do.
    if gpos is None:
        return font

    for lookup in gpos.table.LookupList.Lookup:
        if lookup.LookupType == 2:
            compact_lookup(font, level, lookup)
        elif lookup.LookupType == 9 and lookup.SubTable[0].ExtensionLookupType == 2:
            compact_ext_lookup(font, level, lookup)

    return font


def compact_lookup(font: TTFont, level: int, lookup: otTables.Lookup) -> None:
    new_subtables = compact_pair_pos(font, level, lookup.SubTable)
    lookup.SubTable = new_subtables
    lookup.SubTableCount = len(new_subtables)


def compact_ext_lookup(font: TTFont, level: int, lookup: otTables.Lookup) -> None:
    new_subtables = compact_pair_pos(
        font, level, [ext_subtable.ExtSubTable for ext_subtable in lookup.SubTable]
    )
    new_ext_subtables = []
    for subtable in new_subtables:
        ext_subtable = otTables.ExtensionPos()
        ext_subtable.Format = 1
        ext_subtable.ExtSubTable = subtable
        new_ext_subtables.append(ext_subtable)
    lookup.SubTable = new_ext_subtables
    lookup.SubTableCount = len(new_ext_subtables)


def compact_pair_pos(
    font: TTFont, level: int, subtables: Sequence[otTables.PairPos]
) -> Sequence[otTables.PairPos]:
    new_subtables = []
    for subtable in subtables:
        if subtable.Format == 1:
            # Not doing anything to Format 1 (yet?)
            new_subtables.append(subtable)
        elif subtable.Format == 2:
            new_subtables.extend(compact_class_pairs(font, level, subtable))
    return new_subtables


def compact_class_pairs(
    font: TTFont, level: int, subtable: otTables.PairPos
) -> List[otTables.PairPos]:
    from fontTools.otlLib.builder import buildPairPosClassesSubtable

    subtables = []
    classes1: DefaultDict[int, List[str]] = defaultdict(list)
    for g in subtable.Coverage.glyphs:
        classes1[subtable.ClassDef1.classDefs.get(g, 0)].append(g)
    classes2: DefaultDict[int, List[str]] = defaultdict(list)
    for g, i in subtable.ClassDef2.classDefs.items():
        classes2[i].append(g)
    all_pairs = {}
    for i, class1 in enumerate(subtable.Class1Record):
        for j, class2 in enumerate(class1.Class2Record):
            if is_really_zero(class2):
                continue
            all_pairs[(tuple(sorted(classes1[i])), tuple(sorted(classes2[j])))] = (
                getattr(class2, "Value1", None),
                getattr(class2, "Value2", None),
            )
    grouped_pairs = cluster_pairs_by_class2_coverage_custom_cost(font, all_pairs, level)
    for pairs in grouped_pairs:
        subtables.append(buildPairPosClassesSubtable(pairs, font.getReverseGlyphMap()))
    return subtables


def is_really_zero(class2: otTables.Class2Record) -> bool:
    v1 = getattr(class2, "Value1", None)
    v2 = getattr(class2, "Value2", None)
    return (v1 is None or v1.getEffectiveFormat() == 0) and (
        v2 is None or v2.getEffectiveFormat() == 0
    )


Pairs = Dict[
    Tuple[Tuple[str, ...], Tuple[str, ...]],
    Tuple[otBase.ValueRecord, otBase.ValueRecord],
]


# Adapted from https://github.com/fonttools/fonttools/blob/f64f0b42f2d1163b2d85194e0979def539f5dca3/Lib/fontTools/ttLib/tables/otTables.py#L935-L958
def _getClassRanges(glyphIDs: Iterable[int]):
    glyphIDs = sorted(glyphIDs)
    last = glyphIDs[0]
    ranges = [[last]]
    for glyphID in glyphIDs[1:]:
        if glyphID != last + 1:
            ranges[-1].append(last)
            ranges.append([glyphID])
        last = glyphID
    ranges[-1].append(last)
    return ranges, glyphIDs[0], glyphIDs[-1]


# Adapted from https://github.com/fonttools/fonttools/blob/f64f0b42f2d1163b2d85194e0979def539f5dca3/Lib/fontTools/ttLib/tables/otTables.py#L960-L989
def _classDef_bytes(
    class_data: List[Tuple[List[Tuple[int, int]], int, int]],
    class_ids: List[int],
    coverage=False,
):
    if not class_ids:
        return 0
    first_ranges, min_glyph_id, max_glyph_id = class_data[class_ids[0]]
    range_count = len(first_ranges)
    for i in class_ids[1:]:
        data = class_data[i]
        range_count += len(data[0])
        min_glyph_id = min(min_glyph_id, data[1])
        max_glyph_id = max(max_glyph_id, data[2])
    glyphCount = max_glyph_id - min_glyph_id + 1
    # https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#class-definition-table-format-1
    format1_bytes = 6 + glyphCount * 2
    # https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#class-definition-table-format-2
    format2_bytes = 4 + range_count * 6
    return min(format1_bytes, format2_bytes)


ClusteringContext = namedtuple(
    "ClusteringContext",
    [
        "lines",
        "all_class1",
        "all_class1_data",
        "all_class2_data",
        "valueFormat1_bytes",
        "valueFormat2_bytes",
    ],
)


@dataclass
class Cluster:
    ctx: ClusteringContext
    indices_bitmask: int

    @cached_property
    def indices(self):
        return bit_indices(self.indices_bitmask)

    @cached_property
    def column_indices(self):
        # Indices of columns that have a 1 in at least 1 line
        #   => binary OR all the lines
        bitmask = reduce(int.__or__, (self.ctx.lines[i] for i in self.indices))
        return bit_indices(bitmask)

    @property
    def width(self):
        # Add 1 because Class2=0 cannot be used but needs to be encoded.
        return len(self.column_indices) + 1

    @cached_property
    def cost(self):
        return (
            # 2 bytes to store the offset to this subtable in the Lookup table above
            2
            # Contents of the subtable
            # From: https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#pair-adjustment-positioning-format-2-class-pair-adjustment
            # uint16	posFormat	Format identifier: format = 2
            + 2
            # Offset16	coverageOffset	Offset to Coverage table, from beginning of PairPos subtable.
            + 2
            + self.coverage_bytes
            # uint16	valueFormat1	ValueRecord definition — for the first glyph of the pair (may be zero).
            + 2
            # uint16	valueFormat2	ValueRecord definition — for the second glyph of the pair (may be zero).
            + 2
            # Offset16	classDef1Offset	Offset to ClassDef table, from beginning of PairPos subtable — for the first glyph of the pair.
            + 2
            + self.classDef1_bytes
            # Offset16	classDef2Offset	Offset to ClassDef table, from beginning of PairPos subtable — for the second glyph of the pair.
            + 2
            + self.classDef2_bytes
            # uint16	class1Count	Number of classes in classDef1 table — includes Class 0.
            + 2
            # uint16	class2Count	Number of classes in classDef2 table — includes Class 0.
            + 2
            # Class1Record	class1Records[class1Count]	Array of Class1 records, ordered by classes in classDef1.
            + (self.ctx.valueFormat1_bytes + self.ctx.valueFormat2_bytes)
            * len(self.indices)
            * self.width
        )

    @property
    def coverage_bytes(self):
        format1_bytes = (
            # From https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#coverage-format-1
            # uint16	coverageFormat	Format identifier — format = 1
            # uint16	glyphCount	Number of glyphs in the glyph array
            4
            # uint16	glyphArray[glyphCount]	Array of glyph IDs — in numerical order
            + sum(len(self.ctx.all_class1[i]) for i in self.indices) * 2
        )
        ranges = sorted(
            chain.from_iterable(self.ctx.all_class1_data[i][0] for i in self.indices)
        )
        merged_range_count = 0
        last = None
        for start, end in ranges:
            if last is not None and start != last + 1:
                merged_range_count += 1
            last = end
        format2_bytes = (
            # From https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#coverage-format-2
            # uint16	coverageFormat	Format identifier — format = 2
            # uint16	rangeCount	Number of RangeRecords
            4
            # RangeRecord	rangeRecords[rangeCount]	Array of glyph ranges — ordered by startGlyphID.
            # uint16	startGlyphID	First glyph ID in the range
            # uint16	endGlyphID	Last glyph ID in the range
            # uint16	startCoverageIndex	Coverage Index of first glyph ID in range
            + merged_range_count * 6
        )
        return min(format1_bytes, format2_bytes)

    @property
    def classDef1_bytes(self):
        # We can skip encoding one of the Class1 definitions, and use
        # Class1=0 to represent it instead, because Class1 is gated by the
        # Coverage definition. Use Class1=0 for the highest byte savings.
        # Going through all options takes too long, pick the biggest class
        # = what happens in otlLib.builder.ClassDefBuilder.classes()
        biggest_index = max(self.indices, key=lambda i: len(self.ctx.all_class1[i]))
        return _classDef_bytes(
            self.ctx.all_class1_data, [i for i in self.indices if i != biggest_index]
        )

    @property
    def classDef2_bytes(self):
        # All Class2 need to be encoded because we can't use Class2=0
        return _classDef_bytes(self.ctx.all_class2_data, self.column_indices)


def cluster_pairs_by_class2_coverage_custom_cost(
    font: TTFont,
    pairs: Pairs,
    compression: int = 5,
) -> List[Pairs]:
    if not pairs:
        # The subtable was actually empty?
        return [pairs]

    # Sorted for reproducibility/determinism
    all_class1 = sorted(set(pair[0] for pair in pairs))
    all_class2 = sorted(set(pair[1] for pair in pairs))

    # Use Python's big ints for binary vectors representing each line
    lines = [
        sum(
            1 << i if (class1, class2) in pairs else 0
            for i, class2 in enumerate(all_class2)
        )
        for class1 in all_class1
    ]

    # Map glyph names to ids and work with ints throughout for ClassDef formats
    name_to_id = font.getReverseGlyphMap()
    # Each entry in the arrays below is (range_count, min_glyph_id, max_glyph_id)
    all_class1_data = [
        _getClassRanges(name_to_id[name] for name in cls) for cls in all_class1
    ]
    all_class2_data = [
        _getClassRanges(name_to_id[name] for name in cls) for cls in all_class2
    ]

    format1 = 0
    format2 = 0
    for pair, value in pairs.items():
        format1 |= value[0].getEffectiveFormat() if value[0] else 0
        format2 |= value[1].getEffectiveFormat() if value[1] else 0
    valueFormat1_bytes = bit_count(format1) * 2
    valueFormat2_bytes = bit_count(format2) * 2

    ctx = ClusteringContext(
        lines,
        all_class1,
        all_class1_data,
        all_class2_data,
        valueFormat1_bytes,
        valueFormat2_bytes,
    )

    cluster_cache: Dict[int, Cluster] = {}

    def make_cluster(indices: int) -> Cluster:
        cluster = cluster_cache.get(indices, None)
        if cluster is not None:
            return cluster
        cluster = Cluster(ctx, indices)
        cluster_cache[indices] = cluster
        return cluster

    def merge(cluster: Cluster, other: Cluster) -> Cluster:
        return make_cluster(cluster.indices_bitmask | other.indices_bitmask)

    # Agglomerative clustering by hand, checking the cost gain of the new
    # cluster against the previously separate clusters
    # Start with 1 cluster per line
    # cluster = set of lines = new subtable
    clusters = [make_cluster(1 << i) for i in range(len(lines))]

    # Cost of 1 cluster with everything
    # `(1 << len) - 1` gives a bitmask full of 1's of length `len`
    cost_before_splitting = make_cluster((1 << len(lines)) - 1).cost
    log.debug(f"        len(clusters) = {len(clusters)}")

    while len(clusters) > 1:
        lowest_cost_change = None
        best_cluster_index = None
        best_other_index = None
        best_merged = None
        for i, cluster in enumerate(clusters):
            for j, other in enumerate(clusters[i + 1 :]):
                merged = merge(cluster, other)
                cost_change = merged.cost - cluster.cost - other.cost
                if lowest_cost_change is None or cost_change < lowest_cost_change:
                    lowest_cost_change = cost_change
                    best_cluster_index = i
                    best_other_index = i + 1 + j
                    best_merged = merged
        assert lowest_cost_change is not None
        assert best_cluster_index is not None
        assert best_other_index is not None
        assert best_merged is not None

        # If the best merge we found is still taking down the file size, then
        # there's no question: we must do it, because it's beneficial in both
        # ways (lower file size and lower number of subtables).  However, if the
        # best merge we found is not reducing file size anymore, then we need to
        # look at the other stop criteria = the compression factor.
        if lowest_cost_change > 0:
            # Stop critera: check whether we should keep merging.
            # Compute size reduction brought by splitting
            cost_after_splitting = sum(c.cost for c in clusters)
            # size_reduction so that after = before * (1 - size_reduction)
            # E.g. before = 1000, after = 800, 1 - 800/1000 = 0.2
            size_reduction = 1 - cost_after_splitting / cost_before_splitting

            # Force more merging by taking into account the compression number.
            # Target behaviour: compression number = 1 to 9, default 5 like gzip
            #   - 1 = accept to add 1 subtable to reduce size by 50%
            #   - 5 = accept to add 5 subtables to reduce size by 50%
            # See https://github.com/harfbuzz/packtab/blob/master/Lib/packTab/__init__.py#L690-L691
            # Given the size reduction we have achieved so far, compute how many
            # new subtables are acceptable.
            max_new_subtables = -log2(1 - size_reduction) * compression
            log.debug(
                f"            len(clusters) = {len(clusters):3d}    size_reduction={size_reduction:5.2f}    max_new_subtables={max_new_subtables}",
            )
            if compression == 9:
                # Override level 9 to mean: create any number of subtables
                max_new_subtables = len(clusters)

            # If we have managed to take the number of new subtables below the
            # threshold, then we can stop.
            if len(clusters) <= max_new_subtables + 1:
                break

        # No reason to stop yet, do the merge and move on to the next.
        del clusters[best_other_index]
        clusters[best_cluster_index] = best_merged

    # All clusters are final; turn bitmasks back into the "Pairs" format
    pairs_by_class1: Dict[Tuple[str, ...], Pairs] = defaultdict(dict)
    for pair, values in pairs.items():
        pairs_by_class1[pair[0]][pair] = values
    pairs_groups: List[Pairs] = []
    for cluster in clusters:
        pairs_group: Pairs = dict()
        for i in cluster.indices:
            class1 = all_class1[i]
            pairs_group.update(pairs_by_class1[class1])
        pairs_groups.append(pairs_group)
    return pairs_groups
<|endoftext|>"""
This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.
"""

# See doc/source/reference/routines.dtypes.rst for module-level docs

__all__ = []


def _add_dtype_helper(DType, alias):
    # Function to add DTypes a bit more conveniently without channeling them
    # through `numpy._core._multiarray_umath` namespace or similar.
    from numpy import dtypes

    setattr(dtypes, DType.__name__, DType)
    __all__.append(DType.__name__)

    if alias:
        alias = alias.removeprefix("numpy.dtypes.")
        setattr(dtypes, alias, DType)
        __all__.append(alias)
<|endoftext|># This file is generated by numpy's build process
# It contains system_info results at the time of building this package.
from enum import Enum
from numpy._core._multiarray_umath import (
    __cpu_features__,
    __cpu_baseline__,
    __cpu_dispatch__,
)

__all__ = ["show_config"]
_built_with_meson = True


class DisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def _cleanup(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return {k: _cleanup(v) for k, v in d.items() if v and _cleanup(v)}
    else:
        return d


CONFIG = _cleanup(
    {
        "Compilers": {
            "c": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.2.1",
                "commands": r"cc",
                "args": r"",
                "linker args": r"",
            },
            "cython": {
                "name": "cython",
                "linker": r"cython",
                "version": "3.1.0",
                "commands": r"cython",
                "args": r"",
                "linker args": r"",
            },
            "c++": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.2.1",
                "commands": r"c++",
                "args": r"",
                "linker args": r"",
            },
        },
        "Machine Information": {
            "host": {
                "cpu": "x86_64",
                "family": "x86_64",
                "endian": "little",
                "system": "linux",
            },
            "build": {
                "cpu": "x86_64",
                "family": "x86_64",
                "endian": "little",
                "system": "linux",
            },
            "cross-compiled": bool("False".lower().replace("false", "")),
        },
        "Build Dependencies": {
            "blas": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace("false", "")),
                "version": "0.3.29",
                "detection method": "pkgconfig",
                "include directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
                "lib directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
                "openblas configuration": r"OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
                "pc file directory": r"/project/.openblas",
            },
            "lapack": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace("false", "")),
                "version": "0.3.29",
                "detection method": "pkgconfig",
                "include directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
                "lib directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
                "openblas configuration": r"OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
                "pc file directory": r"/project/.openblas",
            },
        },
        "Python Information": {
            "path": r"/tmp/build-env-a8ncef9o/bin/python",
            "version": "3.10",
        },
        "SIMD Extensions": {
            "baseline": __cpu_baseline__,
            "found": [
                feature for feature in __cpu_dispatch__ if __cpu_features__[feature]
            ],
            "not found": [
                feature for feature in __cpu_dispatch__ if not __cpu_features__[feature]
            ],
        },
    }
)


def _check_pyyaml():
    import yaml

    return yaml


def show(mode=DisplayModes.stdout.value):
    """
    Show libraries and system information on which NumPy was built
    and is being used

    Parameters
    ----------
    mode : {`'stdout'`, `'dicts'`}, optional.
        Indicates how to display the config information.
        `'stdout'` prints to console, `'dicts'` returns a dictionary
        of the configuration.

    Returns
    -------
    out : {`dict`, `None`}
        If mode is `'dicts'`, a dict is returned, else None

    See Also
    --------
    get_include : Returns the directory containing NumPy C
                  header files.

    Notes
    -----
    1. The `'stdout'` mode will give more readable
       output if ``pyyaml`` is installed

    """
    if mode == DisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = _check_pyyaml()

            print(yaml.dump(CONFIG))
        except ModuleNotFoundError:
            import warnings
            import json

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(CONFIG, indent=2))
    elif mode == DisplayModes.dicts.value:
        return CONFIG
    else:
        raise AttributeError(
            f"Invalid `mode`, use one of: {', '.join([e.value for e in DisplayModes])}"
        )


def show_config(mode=DisplayModes.stdout.value):
    return show(mode)


show_config.__doc__ = show.__doc__
show_config.__module__ = "numpy"
<|endoftext|>import argparse
from pathlib import Path
import sys

from .version import __version__
from .lib._utils_impl import get_include


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--cflags",
        action="store_true",
        help="Compile flag needed when using the NumPy headers.",
    )
    parser.add_argument(
        "--pkgconfigdir",
        action="store_true",
        help=("Print the pkgconfig directory in which `numpy.pc` is stored "
              "(useful for setting $PKG_CONFIG_PATH)."),
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cflags:
        print("-I" + get_include())
    if args.pkgconfigdir:
        _path = Path(get_include()) / '..' / 'lib' / 'pkgconfig'
        print(_path.resolve())


if __name__ == "__main__":
    main()
<|endoftext|>import warnings

# 2018-05-29, PendingDeprecationWarning added to matrix.__new__
# 2020-01-23, numpy 1.19.0 PendingDeprecatonWarning
warnings.warn("Importing from numpy.matlib is deprecated since 1.19.0. "
              "The matrix subclass is not the recommended way to represent "
              "matrices or deal with linear algebra (see "
              "https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). "
              "Please adjust your code to use regular ndarray. ",
              PendingDeprecationWarning, stacklevel=2)

import numpy as np
from numpy.matrixlib.defmatrix import matrix, asmatrix
# Matlib.py contains all functions in the numpy namespace with a few
# replacements. See doc/source/reference/routines.matlib.rst for details.
# Need * as we're copying the numpy namespace.
from numpy import *  # noqa: F403

__version__ = np.__version__

__all__ = np.__all__[:] # copy numpy namespace
__all__ += ['rand', 'randn', 'repmat']

def empty(shape, dtype=None, order='C'):
    """Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    See Also
    --------
    numpy.empty : Equivalent array function.
    matlib.zeros : Return a matrix of zeros.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    Unlike other matrix creation functions (e.g. `matlib.zeros`,
    `matlib.ones`), `matlib.empty` does not initialize the values of the
    matrix, and may therefore be marginally faster. However, the values
    stored in the newly allocated matrix are arbitrary. For reproducible
    behavior, be sure to set each element of the matrix before reading.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307], # random
            [  7.39337286e-309,   3.22135945e-309]])
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0], # random
            [ 6586976, 22740995]])

    """
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    order : {'C', 'F'}, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.
    matlib.zeros : Zero matrix.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.ones((2,3))
    matrix([[1.,  1.,  1.],
            [1.,  1.,  1.]])

    >>> np.matlib.ones(2)
    matrix([[1.,  1.]])

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a

def zeros(shape, dtype=None, order='C'):
    """
    Return a matrix of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is float.
    order : {'C', 'F'}, optional
        Whether to store the result in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Zero matrix of given shape, dtype, and order.

    See Also
    --------
    numpy.zeros : Equivalent array function.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.zeros((2, 3))
    matrix([[0.,  0.,  0.],
            [0.,  0.,  0.]])

    >>> np.matlib.zeros(2)
    matrix([[0.,  0.]])

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a

def identity(n,dtype=None):
    """
    Returns the square identity matrix of given size.

    Parameters
    ----------
    n : int
        Size of the returned identity matrix.
    dtype : data-type, optional
        Data-type of the output. Defaults to ``float``.

    Returns
    -------
    out : matrix
        `n` x `n` matrix with its main diagonal set to one,
        and all other elements zero.

    See Also
    --------
    numpy.identity : Equivalent array function.
    matlib.eye : More general matrix identity function.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.identity(3, dtype=int)
    matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

    """
    a = array([1]+n*[0], dtype=dtype)
    b = empty((n, n), dtype=dtype)
    b.flat = a
    return b

def eye(n,M=None, k=0, dtype=float, order='C'):
    """
    Return a matrix with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output, defaults to `n`.
    k : int, optional
        Index of the diagonal: 0 refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype : dtype, optional
        Data-type of the returned matrix.
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    I : matrix
        A `n` x `M` matrix where all elements are equal to zero,
        except for the `k`-th diagonal, whose values are equal to one.

    See Also
    --------
    numpy.eye : Equivalent array function.
    identity : Square identity matrix.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.eye(3, k=1, dtype=float)
    matrix([[0.,  1.,  0.],
            [0.,  0.,  1.],
            [0.,  0.,  0.]])

    """
    return asmatrix(np.eye(n, M=M, k=k, dtype=dtype, order=order))

def rand(*args):
    """
    Return a matrix of random values with given shape.

    Create a matrix of the given shape and propagate it with
    random samples from a uniform distribution over ``[0, 1)``.

    Parameters
    ----------
    \\*args : Arguments
        Shape of the output.
        If given as N integers, each integer specifies the size of one
        dimension.
        If given as a tuple, this tuple gives the complete shape.

    Returns
    -------
    out : ndarray
        The matrix of random values with shape given by `\\*args`.

    See Also
    --------
    randn, numpy.random.RandomState.rand

    Examples
    --------
    >>> np.random.seed(123)
    >>> import numpy.matlib
    >>> np.matlib.rand(2, 3)
    matrix([[0.69646919, 0.28613933, 0.22685145],
            [0.55131477, 0.71946897, 0.42310646]])
    >>> np.matlib.rand((2, 3))
    matrix([[0.9807642 , 0.68482974, 0.4809319 ],
            [0.39211752, 0.34317802, 0.72904971]])

    If the first argument is a tuple, other arguments are ignored:

    >>> np.matlib.rand((2, 3), 4)
    matrix([[0.43857224, 0.0596779 , 0.39804426],
            [0.73799541, 0.18249173, 0.17545176]])

    """
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.rand(*args))

def randn(*args):
    """
    Return a random matrix with data from the "standard normal" distribution.

    `randn` generates a matrix filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

    Parameters
    ----------
    \\*args : Arguments
        Shape of the output.
        If given as N integers, each integer specifies the size of one
        dimension. If given as a tuple, this tuple gives the complete shape.

    Returns
    -------
    Z : matrix of floats
        A matrix of floating-point samples drawn from the standard normal
        distribution.

    See Also
    --------
    rand, numpy.random.RandomState.randn

    Notes
    -----
    For random samples from the normal distribution with mean ``mu`` and
    standard deviation ``sigma``, use::

        sigma * np.matlib.randn(...) + mu

    Examples
    --------
    >>> np.random.seed(123)
    >>> import numpy.matlib
    >>> np.matlib.randn(1)
    matrix([[-1.0856306]])
    >>> np.matlib.randn(1, 2, 3)
    matrix([[ 0.99734545,  0.2829785 , -1.50629471],
            [-0.57860025,  1.65143654, -2.42667924]])

    Two-by-four matrix of samples from the normal distribution with
    mean 3 and standard deviation 2.5:

    >>> 2.5 * np.matlib.randn((2, 4)) + 3
    matrix([[1.92771843, 6.16484065, 0.83314899, 1.30278462],
            [2.76322758, 6.72847407, 1.40274501, 1.8900451 ]])

    """
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.randn(*args))

def repmat(a, m, n):
    """
    Repeat a 0-D to 2-D array or matrix MxN times.

    Parameters
    ----------
    a : array_like
        The array or matrix to be repeated.
    m, n : int
        The number of times `a` is repeated along the first and second axes.

    Returns
    -------
    out : ndarray
        The result of repeating `a`.

    Examples
    --------
    >>> import numpy.matlib
    >>> a0 = np.array(1)
    >>> np.matlib.repmat(a0, 2, 3)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> a1 = np.arange(4)
    >>> np.matlib.repmat(a1, 2, 2)
    array([[0, 1, 2, 3, 0, 1, 2, 3],
           [0, 1, 2, 3, 0, 1, 2, 3]])

    >>> a2 = np.asmatrix(np.arange(6).reshape(2, 3))
    >>> np.matlib.repmat(a2, 2, 3)
    matrix([[0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5]])

    """
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)
<|endoftext|>"""
============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""
__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array',
           'as_ctypes_type']

import os
import numpy as np
from numpy._core.multiarray import _flagdict, flagsobj

try:
    import ctypes
except ImportError:
    ctypes = None

if ctypes is None:
    def _dummy(*args, **kwds):
        """
        Dummy object that raises an ImportError if ctypes is not available.

        Raises
        ------
        ImportError
            If ctypes is not available.

        """
        raise ImportError("ctypes is not available.")
    load_library = _dummy
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp
    _ndptr_base = object
else:
    import numpy._core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic
    _ndptr_base = ctypes.c_void_p

    # Adapted from Albert Strasheim
    def load_library(libname, loader_path):
        """
        It is possible to load a library using

        >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

        But there are cross-platform considerations, such as library file extensions,
        plus the fact Windows will just load the first library it finds with that name.
        NumPy supplies the load_library function as a convenience.

        .. versionchanged:: 1.20.0
            Allow libname and loader_path to take any
            :term:`python:path-like object`.

        Parameters
        ----------
        libname : path-like
            Name of the library, which can have 'lib' as a prefix,
            but without an extension.
        loader_path : path-like
            Where the library can be found.

        Returns
        -------
        ctypes.cdll[libpath] : library object
           A ctypes library object

        Raises
        ------
        OSError
            If there is no library with the expected extension, or the
            library is defective and cannot be loaded.
        """
        # Convert path-like objects into strings
        libname = os.fsdecode(libname)
        loader_path = os.fsdecode(loader_path)

        ext = os.path.splitext(libname)[1]
        if not ext:
            import sys
            import sysconfig
            # Try to load library with platform-specific name, otherwise
            # default to libname.[so|dll|dylib].  Sometimes, these files are
            # built erroneously on non-linux platforms.
            base_ext = ".so"
            if sys.platform.startswith("darwin"):
                base_ext = ".dylib"
            elif sys.platform.startswith("win"):
                base_ext = ".dll"
            libname_ext = [libname + base_ext]
            so_ext = sysconfig.get_config_var("EXT_SUFFIX")
            if not so_ext == base_ext:
                libname_ext.insert(0, libname + so_ext)
        else:
            libname_ext = [libname]

        loader_path = os.path.abspath(loader_path)
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)
        else:
            libdir = loader_path

        for ln in libname_ext:
            libpath = os.path.join(libdir, ln)
            if os.path.exists(libpath):
                try:
                    return ctypes.cdll[libpath]
                except OSError:
                    ## defective lib file
                    raise
        ## if no successful return in the libname_ext loop:
        raise OSError("no file with expected extension")


def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num

_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
              'OWNDATA', 'WRITEBACKIFCOPY']
def _flags_fromnum(num):
    res = []
    for key in _flagnames:
        value = _flagdict[key]
        if (num & value):
            res.append(key)
    return res


class _ndptr(_ndptr_base):
    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, np.ndarray):
            raise TypeError("argument must be an ndarray")
        if cls._dtype_ is not None \
               and obj.dtype != cls._dtype_:
            raise TypeError("array must have data type %s" % cls._dtype_)
        if cls._ndim_ is not None \
               and obj.ndim != cls._ndim_:
            raise TypeError("array must have %d dimension(s)" % cls._ndim_)
        if cls._shape_ is not None \
               and obj.shape != cls._shape_:
            raise TypeError("array must have shape %s" % str(cls._shape_))
        if cls._flags_ is not None \
               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError("array must have flags %s" %
                    _flags_fromnum(cls._flags_))
        return obj.ctypes


class _concrete_ndptr(_ndptr):
    """
    Like _ndptr, but with `_shape_` and `_dtype_` specified.

    Notably, this means the pointer has enough information to reconstruct
    the array, which is not generally true.
    """
    def _check_retval_(self):
        """
        This method is called when this class is used as the .restype
        attribute for a shared-library function, to automatically wrap the
        pointer into an array.
        """
        return self.contents

    @property
    def contents(self):
        """
        Get an ndarray viewing the data pointed to by this pointer.

        This mirrors the `contents` attribute of a normal ctypes pointer
        """
        full_dtype = np.dtype((self._dtype_, self._shape_))
        full_ctype = ctypes.c_char * full_dtype.itemsize
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        return np.frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)


# Factory for an array-checking class with from_param defined for
#  use with ctypes argtypes mechanism
_pointer_type_cache = {}
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    """
    Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example, ``POINTER(c_double)``, since several restrictions
    can be specified, which are verified upon calling the ctypes function.
    These include data type, number of dimensions, shape and flags.  If a
    given array does not satisfy the specified restrictions,
    a ``TypeError`` is raised.

    Parameters
    ----------
    dtype : data-type, optional
        Array data-type.
    ndim : int, optional
        Number of array dimensions.
    shape : tuple of ints, optional
        Array shape.
    flags : str or tuple of str
        Array flags; may be one or more of:

        - C_CONTIGUOUS / C / CONTIGUOUS
        - F_CONTIGUOUS / F / FORTRAN
        - OWNDATA / O
        - WRITEABLE / W
        - ALIGNED / A
        - WRITEBACKIFCOPY / X

    Returns
    -------
    klass : ndpointer type object
        A type object, which is an ``_ndtpr`` instance containing
        dtype, ndim, shape and flags information.

    Raises
    ------
    TypeError
        If a given array does not satisfy the specified restrictions.

    Examples
    --------
    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
    ...                                                  ndim=1,
    ...                                                  flags='C_CONTIGUOUS')]
    ... #doctest: +SKIP
    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
    ... #doctest: +SKIP

    """

    # normalize dtype to dtype | None
    if dtype is not None:
        dtype = np.dtype(dtype)

    # normalize flags to int | None
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, np.integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError("invalid flags specification") from e
            num = _num_fromflags(flags)

    # normalize shape to tuple | None
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            # single integer -> 1-tuple
            shape = (shape,)

    cache_key = (dtype, ndim, shape, num)

    try:
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass

    # produce a name for the new type
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        name += "_"+"x".join(str(x) for x in shape)
    if flags is not None:
        name += "_"+"_".join(flags)

    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr

    klass = type("ndpointer_%s"%name, (base,),
                 {"_dtype_": dtype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    _pointer_type_cache[cache_key] = klass
    return klass


if ctypes is not None:
    def _ctype_ndarray(element_type, shape):
        """ Create an ndarray of the given element type and shape """
        for dim in shape[::-1]:
            element_type = dim * element_type
            # prevent the type name include np.ctypeslib
            element_type.__module__ = None
        return element_type


    def _get_scalar_type_map():
        """
        Return a dictionary mapping native endian scalar dtype to ctypes types
        """
        ct = ctypes
        simple_types = [
            ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
            ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
            ct.c_float, ct.c_double,
            ct.c_bool,
        ]
        return {np.dtype(ctype): ctype for ctype in simple_types}


    _scalar_type_map = _get_scalar_type_map()


    def _ctype_from_dtype_scalar(dtype):
        # swapping twice ensure that `=` is promoted to <, >, or |
        dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
        dtype_native = dtype.newbyteorder('=')
        try:
            ctype = _scalar_type_map[dtype_native]
        except KeyError as e:
            raise NotImplementedError(
                "Converting {!r} to a ctypes type".format(dtype)
            ) from None

        if dtype_with_endian.byteorder == '>':
            ctype = ctype.__ctype_be__
        elif dtype_with_endian.byteorder == '<':
            ctype = ctype.__ctype_le__

        return ctype


    def _ctype_from_dtype_subarray(dtype):
        element_dtype, shape = dtype.subdtype
        ctype = _ctype_from_dtype(element_dtype)
        return _ctype_ndarray(ctype, shape)


    def _ctype_from_dtype_structured(dtype):
        # extract offsets of each field
        field_data = []
        for name in dtype.names:
            field_dtype, offset = dtype.fields[name][:2]
            field_data.append((offset, name, _ctype_from_dtype(field_dtype)))

        # ctypes doesn't care about field order
        field_data = sorted(field_data, key=lambda f: f[0])

        if len(field_data) > 1 and all(offset == 0 for offset, name, ctype in field_data):
            # union, if multiple fields all at address 0
            size = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                _fields_.append((name, ctype))
                size = max(size, ctypes.sizeof(ctype))

            # pad to the right size
            if dtype.itemsize != size:
                _fields_.append(('', ctypes.c_char * dtype.itemsize))

            # we inserted manual padding, so always `_pack_`
            return type('union', (ctypes.Union,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))
        else:
            last_offset = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                padding = offset - last_offset
                if padding < 0:
                    raise NotImplementedError("Overlapping fields")
                if padding > 0:
                    _fields_.append(('', ctypes.c_char * padding))

                _fields_.append((name, ctype))
                last_offset = offset + ctypes.sizeof(ctype)


            padding = dtype.itemsize - last_offset
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))

            # we inserted manual padding, so always `_pack_`
            return type('struct', (ctypes.Structure,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))


    def _ctype_from_dtype(dtype):
        if dtype.fields is not None:
            return _ctype_from_dtype_structured(dtype)
        elif dtype.subdtype is not None:
            return _ctype_from_dtype_subarray(dtype)
        else:
            return _ctype_from_dtype_scalar(dtype)


    def as_ctypes_type(dtype):
        r"""
        Convert a dtype into a ctypes type.

        Parameters
        ----------
        dtype : dtype
            The dtype to convert

        Returns
        -------
        ctype
            A ctype scalar, union, array, or struct

        Raises
        ------
        NotImplementedError
            If the conversion is not possible

        Notes
        -----
        This function does not losslessly round-trip in either direction.

        ``np.dtype(as_ctypes_type(dt))`` will:

        - insert padding fields
        - reorder fields to be sorted by offset
        - discard field titles

        ``as_ctypes_type(np.dtype(ctype))`` will:

        - discard the class names of `ctypes.Structure`\ s and
          `ctypes.Union`\ s
        - convert single-element `ctypes.Union`\ s into single-element
          `ctypes.Structure`\ s
        - insert padding fields

        Examples
        --------
        Converting a simple dtype:

        >>> dt = np.dtype('int8')
        >>> ctype = np.ctypeslib.as_ctypes_type(dt)
        >>> ctype
        <class 'ctypes.c_byte'>

        Converting a structured dtype:

        >>> dt = np.dtype([('x', 'i4'), ('y', 'f4')])
        >>> ctype = np.ctypeslib.as_ctypes_type(dt)
        >>> ctype
        <class 'struct'>

        """
        return _ctype_from_dtype(np.dtype(dtype))


    def as_array(obj, shape=None):
        """
        Create a numpy array from a ctypes array or POINTER.

        The numpy array shares the memory with the ctypes object.

        The shape parameter must be given if converting from a ctypes POINTER.
        The shape parameter is ignored if converting from a ctypes array

        Examples
        --------
        Converting a ctypes integer array:

        >>> import ctypes
        >>> ctypes_array = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
        >>> np_array = np.ctypeslib.as_array(ctypes_array)
        >>> np_array
        array([0, 1, 2, 3, 4], dtype=int32)

        Converting a ctypes POINTER:

        >>> import ctypes
        >>> buffer = (ctypes.c_int * 5)(0, 1, 2, 3, 4)
        >>> pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int))
        >>> np_array = np.ctypeslib.as_array(pointer, (5,))
        >>> np_array
        array([0, 1, 2, 3, 4], dtype=int32)

        """
        if isinstance(obj, ctypes._Pointer):
            # convert pointers to an array of the desired shape
            if shape is None:
                raise TypeError(
                    'as_array() requires a shape argument when called on a '
                    'pointer')
            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
            obj = ctypes.cast(obj, p_arr_type).contents

        return np.asarray(obj)


    def as_ctypes(obj):
        """
        Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted.

        Examples
        --------
        Create ctypes object from inferred int ``np.array``:

        >>> inferred_int_array = np.array([1, 2, 3])
        >>> c_int_array = np.ctypeslib.as_ctypes(inferred_int_array)
        >>> type(c_int_array)
        <class 'c_long_Array_3'>
        >>> c_int_array[:]
        [1, 2, 3]

        Create ctypes object from explicit 8 bit unsigned int ``np.array`` :

        >>> exp_int_array = np.array([1, 2, 3], dtype=np.uint8)
        >>> c_int_array = np.ctypeslib.as_ctypes(exp_int_array)
        >>> type(c_int_array)
        <class 'c_ubyte_Array_3'>
        >>> c_int_array[:]
        [1, 2, 3]

        """
        ai = obj.__array_interface__
        if ai["strides"]:
            raise TypeError("strided arrays not supported")
        if ai["version"] != 3:
            raise TypeError("only __array_interface__ version 3 supported")
        addr, readonly = ai["data"]
        if readonly:
            raise TypeError("readonly arrays unsupported")

        # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
        # dtype.itemsize (gh-14214)
        ctype_scalar = as_ctypes_type(ai["typestr"])
        result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
        result = result_type.from_address(addr)
        result.__keep = obj
        return result
<|endoftext|>"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more (for Python <= 3.11)

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
import os
import sys
import warnings

from ._globals import _NoValue, _CopyMode
from ._expired_attrs_2_0 import __expired_attributes__


# If a version with git hash was stored, use that instead
from . import version
from .version import __version__

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    # Allow distributors to run custom init code before importing numpy._core
    from . import _distributor_init

    try:
        from numpy.__config__ import show_config
    except ImportError as e:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg) from e

    from . import _core
    from ._core import (
        False_, ScalarType, True_,
        abs, absolute, acos, acosh, add, all, allclose,
        amax, amin, any, arange, arccos, arccosh, arcsin, arcsinh,
        arctan, arctan2, arctanh, argmax, argmin, argpartition, argsort,
        argwhere, around, array, array2string, array_equal, array_equiv,
        array_repr, array_str, asanyarray, asarray, ascontiguousarray,
        asfortranarray, asin, asinh, atan, atanh, atan2, astype, atleast_1d,
        atleast_2d, atleast_3d, base_repr, binary_repr, bitwise_and,
        bitwise_count, bitwise_invert, bitwise_left_shift, bitwise_not,
        bitwise_or, bitwise_right_shift, bitwise_xor, block, bool, bool_,
        broadcast, busday_count, busday_offset, busdaycalendar, byte, bytes_,
        can_cast, cbrt, cdouble, ceil, character, choose, clip, clongdouble,
        complex128, complex64, complexfloating, compress, concat, concatenate,
        conj, conjugate, convolve, copysign, copyto, correlate, cos, cosh,
        count_nonzero, cross, csingle, cumprod, cumsum, cumulative_prod,
        cumulative_sum, datetime64, datetime_as_string, datetime_data,
        deg2rad, degrees, diagonal, divide, divmod, dot, double, dtype, e,
        einsum, einsum_path, empty, empty_like, equal, errstate, euler_gamma,
        exp, exp2, expm1, fabs, finfo, flatiter, flatnonzero, flexible,
        float16, float32, float64, float_power, floating, floor, floor_divide,
        fmax, fmin, fmod, format_float_positional, format_float_scientific,
        frexp, from_dlpack, frombuffer, fromfile, fromfunction, fromiter,
        frompyfunc, fromstring, full, full_like, gcd, generic, geomspace,
        get_printoptions, getbufsize, geterr, geterrcall, greater,
        greater_equal, half, heaviside, hstack, hypot, identity, iinfo,
        indices, inexact, inf, inner, int16, int32, int64, int8, int_, intc,
        integer, intp, invert, is_busday, isclose, isdtype, isfinite,
        isfortran, isinf, isnan, isnat, isscalar, issubdtype, lcm, ldexp,
        left_shift, less, less_equal, lexsort, linspace, little_endian, log,
        log10, log1p, log2, logaddexp, logaddexp2, logical_and, logical_not,
        logical_or, logical_xor, logspace, long, longdouble, longlong, matmul,
        matvec, matrix_transpose, max, maximum, may_share_memory, mean, memmap,
        min, min_scalar_type, minimum, mod, modf, moveaxis, multiply, nan,
        ndarray, ndim, nditer, negative, nested_iters, newaxis, nextafter,
        nonzero, not_equal, number, object_, ones, ones_like, outer, partition,
        permute_dims, pi, positive, pow, power, printoptions, prod,
        promote_types, ptp, put, putmask, rad2deg, radians, ravel, recarray,
        reciprocal, record, remainder, repeat, require, reshape, resize,
        result_type, right_shift, rint, roll, rollaxis, round, sctypeDict,
        searchsorted, set_printoptions, setbufsize, seterr, seterrcall, shape,
        shares_memory, short, sign, signbit, signedinteger, sin, single, sinh,
        size, sort, spacing, sqrt, square, squeeze, stack, std,
        str_, subtract, sum, swapaxes, take, tan, tanh, tensordot,
        timedelta64, trace, transpose, true_divide, trunc, typecodes, ubyte,
        ufunc, uint, uint16, uint32, uint64, uint8, uintc, uintp, ulong,
        ulonglong, unsignedinteger, unstack, ushort, var, vdot, vecdot,
        vecmat, void, vstack, where, zeros, zeros_like
    )

    # NOTE: It's still under discussion whether these aliases
    # should be removed.
    for ta in ["float96", "float128", "complex192", "complex256"]:
        try:
            globals()[ta] = getattr(_core, ta)
        except AttributeError:
            pass
    del ta

    from . import lib
    from .lib import scimath as emath
    from .lib._histograms_impl import (
        histogram, histogram_bin_edges, histogramdd
    )
    from .lib._nanfunctions_impl import (
        nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean,
        nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd,
        nansum, nanvar
    )
    from .lib._function_base_impl import (
        select, piecewise, trim_zeros, copy, iterable, percentile, diff,
        gradient, angle, unwrap, sort_complex, flip, rot90, extract, place,
        vectorize, asarray_chkfinite, average, bincount, digitize, cov,
        corrcoef, median, sinc, hamming, hanning, bartlett, blackman,
        kaiser, trapezoid, trapz, i0, meshgrid, delete, insert, append,
        interp, quantile
    )
    from .lib._twodim_base_impl import (
        diag, diagflat, eye, fliplr, flipud, tri, triu, tril, vander,
        histogram2d, mask_indices, tril_indices, tril_indices_from,
        triu_indices, triu_indices_from
    )
    from .lib._shape_base_impl import (
        apply_over_axes, apply_along_axis, array_split, column_stack, dsplit,
        dstack, expand_dims, hsplit, kron, put_along_axis, row_stack, split,
        take_along_axis, tile, vsplit
    )
    from .lib._type_check_impl import (
        iscomplexobj, isrealobj, imag, iscomplex, isreal, nan_to_num, real,
        real_if_close, typename, mintypecode, common_type
    )
    from .lib._arraysetops_impl import (
        ediff1d, in1d, intersect1d, isin, setdiff1d, setxor1d, union1d,
        unique, unique_all, unique_counts, unique_inverse, unique_values
    )
    from .lib._ufunclike_impl import fix, isneginf, isposinf
    from .lib._arraypad_impl import pad
    from .lib._utils_impl import (
        show_runtime, get_include, info
    )
    from .lib._stride_tricks_impl import (
        broadcast_arrays, broadcast_shapes, broadcast_to
    )
    from .lib._polynomial_impl import (
        poly, polyint, polyder, polyadd, polysub, polymul, polydiv, polyval,
        polyfit, poly1d, roots
    )
    from .lib._npyio_impl import (
        savetxt, loadtxt, genfromtxt, load, save, savez, packbits,
        savez_compressed, unpackbits, fromregex
    )
    from .lib._index_tricks_impl import (
        diag_indices_from, diag_indices, fill_diagonal, ndindex, ndenumerate,
        ix_, c_, r_, s_, ogrid, mgrid, unravel_index, ravel_multi_index,
        index_exp
    )

    from . import matrixlib as _mat
    from .matrixlib import (
        asmatrix, bmat, matrix
    )

    # public submodules are imported lazily, therefore are accessible from
    # __getattr__. Note that `distutils` (deprecated) and `array_api`
    # (experimental label) are not added here, because `from numpy import *`
    # must not raise any warnings - that's too disruptive.
    __numpy_submodules__ = {
        "linalg", "fft", "dtypes", "random", "polynomial", "ma",
        "exceptions", "lib", "ctypeslib", "testing", "typing",
        "f2py", "test", "rec", "char", "core", "strings",
    }

    # We build warning messages for former attributes
    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")

    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")

    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")

    _type_info = [
        ("object", ""),  # The NumPy scalar only exists by name.
        ("float", _specific_msg.format("float64")),
        ("complex", _specific_msg.format("complex128")),
        ("str", _specific_msg.format("str_")),
        ("int", _int_extended_msg.format("int"))]

    __former_attrs__ = {
         n: _msg.format(n=n, extended_msg=extended_msg)
         for n, extended_msg in _type_info
     }


    # Some of these could be defined right away, but most were aliases to
    # the Python objects and only removed in NumPy 1.24.  Defining them should
    # probably wait for NumPy 1.26 or 2.0.
    # When defined, these should possibly not be added to `__all__` to avoid
    # import with `from numpy import *`.
    __future_scalars__ = {"str", "bytes", "object"}

    __array_api_version__ = "2023.12"

    from ._array_api_info import __array_namespace_info__

    # now that numpy core module is imported, can initialize limits
    _core.getlimits._register_known_types()

    __all__ = list(
        __numpy_submodules__ |
        set(_core.__all__) |
        set(_mat.__all__) |
        set(lib._histograms_impl.__all__) |
        set(lib._nanfunctions_impl.__all__) |
        set(lib._function_base_impl.__all__) |
        set(lib._twodim_base_impl.__all__) |
        set(lib._shape_base_impl.__all__) |
        set(lib._type_check_impl.__all__) |
        set(lib._arraysetops_impl.__all__) |
        set(lib._ufunclike_impl.__all__) |
        set(lib._arraypad_impl.__all__) |
        set(lib._utils_impl.__all__) |
        set(lib._stride_tricks_impl.__all__) |
        set(lib._polynomial_impl.__all__) |
        set(lib._npyio_impl.__all__) |
        set(lib._index_tricks_impl.__all__) |
        {"emath", "show_config", "__version__", "__array_namespace_info__"}
    )

    # Filter out Cython harmless warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    def __getattr__(attr):
        # Warn for expired attributes
        import warnings

        if attr == "linalg":
            import numpy.linalg as linalg
            return linalg
        elif attr == "fft":
            import numpy.fft as fft
            return fft
        elif attr == "dtypes":
            import numpy.dtypes as dtypes
            return dtypes
        elif attr == "random":
            import numpy.random as random
            return random
        elif attr == "polynomial":
            import numpy.polynomial as polynomial
            return polynomial
        elif attr == "ma":
            import numpy.ma as ma
            return ma
        elif attr == "ctypeslib":
            import numpy.ctypeslib as ctypeslib
            return ctypeslib
        elif attr == "exceptions":
            import numpy.exceptions as exceptions
            return exceptions
        elif attr == "testing":
            import numpy.testing as testing
            return testing
        elif attr == "matlib":
            import numpy.matlib as matlib
            return matlib
        elif attr == "f2py":
            import numpy.f2py as f2py
            return f2py
        elif attr == "typing":
            import numpy.typing as typing
            return typing
        elif attr == "rec":
            import numpy.rec as rec
            return rec
        elif attr == "char":
            import numpy.char as char
            return char
        elif attr == "array_api":
            raise AttributeError("`numpy.array_api` is not available from "
                                 "numpy 2.0 onwards", name=None)
        elif attr == "core":
            import numpy.core as core
            return core
        elif attr == "strings":
            import numpy.strings as strings
            return strings
        elif attr == "distutils":
            if 'distutils' in __numpy_submodules__:
                import numpy.distutils as distutils
                return distutils
            else:
                raise AttributeError("`numpy.distutils` is not available from "
                                     "Python 3.12 onwards", name=None)

        if attr in __future_scalars__:
            # And future warnings for those that will change, but also give
            # the AttributeError
            warnings.warn(
                f"In the future `np.{attr}` will be defined as the "
                "corresponding NumPy scalar.", FutureWarning, stacklevel=2)

        if attr in __former_attrs__:
            raise AttributeError(__former_attrs__[attr], name=None)

        if attr in __expired_attributes__:
            raise AttributeError(
                f"`np.{attr}` was removed in the NumPy 2.0 release. "
                f"{__expired_attributes__[attr]}",
                name=None
            )

        if attr == "chararray":
            warnings.warn(
                "`np.chararray` is deprecated and will be removed from "
                "the main namespace in the future. Use an array with a string "
                "or bytes dtype instead.", DeprecationWarning, stacklevel=2)
            import numpy.char as char
            return char.chararray

        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))

    def __dir__():
        public_symbols = (
            globals().keys() | __numpy_submodules__
        )
        public_symbols -= {
            "matrixlib", "matlib", "tests", "conftest", "version",
            "compat", "distutils", "array_api"
        }
        return list(public_symbols)

    # Pytest testing
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester

    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-5:
                raise AssertionError
        except AssertionError:
            msg = ("The current Numpy installation ({!r}) fails to "
                   "pass simple sanity checks. This can be caused for example "
                   "by incorrect BLAS library being linked in, or by mixing "
                   "package managers (pip, conda, apt, ...). Search closed "
                   "numpy issues for similar problems.")
            raise RuntimeError(msg.format(__file__)) from None

    _sanity_check()
    del _sanity_check

    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3., 2., 1.])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    if sys.platform == "darwin":
        from . import exceptions
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            # Throw runtime error, if the test failed Check for warning and error_message
            if len(w) > 0:
                for _wn in w:
                    if _wn.category is exceptions.RankWarning:
                        # Ignore other warnings, they may not be relevant (see gh-25433).
                        error_message = (
                            f"{_wn.category.__name__}: {_wn.message}"
                        )
                        msg = (
                            "Polyfit sanity test emitted a warning, most likely due "
                            "to using a buggy Accelerate backend."
                            "\nIf you compiled yourself, more information is available at:"
                            "\nhttps://numpy.org/devdocs/building/index.html"
                            "\nOtherwise report this to the vendor "
                            "that provided NumPy.\n\n{}\n".format(error_message))
                        raise RuntimeError(msg)
                del _wn
            del w
    del _mac_os_check

    def hugepage_setup():
        """
        We usually use madvise hugepages support, but on some old kernels it
        is slow and thus better avoided. Specifically kernel version 4.6
        had a bug fix which probably fixed this:
        https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
        """
        use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
        if sys.platform == "linux" and use_hugepage is None:
            # If there is an issue with parsing the kernel version,
            # set use_hugepage to 0. Usage of LooseVersion will handle
            # the kernel version parsing better, but avoided since it
            # will increase the import time.
            # See: #16679 for related discussion.
            try:
                use_hugepage = 1
                kernel_version = os.uname().release.split(".")[:2]
                kernel_version = tuple(int(v) for v in kernel_version)
                if kernel_version < (4, 6):
                    use_hugepage = 0
            except ValueError:
                use_hugepage = 0
        elif use_hugepage is None:
            # This is not Linux, so it should not matter, just enable anyway
            use_hugepage = 1
        else:
            use_hugepage = int(use_hugepage)
        return use_hugepage

    # Note that this will currently only make a difference on Linux
    _core.multiarray._set_madvise_hugepage(hugepage_setup())
    del hugepage_setup

    # Give a warning if NumPy is reloaded or imported on a sub-interpreter
    # We do this from python, since the C-module may not be reloaded and
    # it is tidier organized.
    _core.multiarray._multiarray_umath._reload_guard()

    # TODO: Remove the environment variable entirely now that it is "weak"
    if (os.environ.get("NPY_PROMOTION_STATE", "weak") != "weak"):
        warnings.warn(
            "NPY_PROMOTION_STATE was a temporary feature for NumPy 2.0 "
            "transition and is ignored after NumPy 2.2.",
            UserWarning, stacklevel=2)

    # Tell PyInstaller where to find hook-numpy.py
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        return [str(Path(__file__).with_name("_pyinstaller").resolve())]


# Remove symbols imported for internal use
del os, sys, warnings
<|endoftext|>"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from numpy._core import (
    dtype,
    bool,
    intp,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)


class __array_namespace_info__:
    """
    Get the array API inspection namespace for NumPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for NumPy.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """

    __module__ = 'numpy'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for NumPy.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          NumPy.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new NumPy arrays.

        For NumPy, this always returns ``'cpu'``.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new NumPy arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return "cpu"

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new NumPy arrays.

        For NumPy, this always returns the following dictionary:

        - **"real floating"**: ``numpy.float64``
        - **"complex floating"**: ``numpy.complex128``
        - **"integral"**: ``numpy.intp``
        - **"indexing"**: ``numpy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for. For NumPy, only
            ``'cpu'`` is allowed.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new NumPy
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': numpy.float64,
         'complex floating': numpy.complex128,
         'integral': numpy.int64,
         'indexing': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by NumPy.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for. For NumPy, only ``'cpu'`` is
            allowed.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            NumPy data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': numpy.int8,
         'int16': numpy.int16,
         'int32': numpy.int32,
         'int64': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        """
        The devices supported by NumPy.

        For NumPy, this always returns ``['cpu']``.

        Returns
        -------
        devices : list of str
            The devices supported by NumPy.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        ['cpu']

        """
        return ["cpu"]
<|endoftext|>"""
Pytest test running.

This module implements the ``test()`` function for NumPy modules. The usual
boiler plate for doing that is to put the following in the module
``__init__.py`` file::

    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester


Warnings filtering and other runtime settings should be dealt with in the
``pytest.ini`` file in the numpy repo root. The behavior of the test depends on
whether or not that file is found as follows:

* ``pytest.ini`` is present (develop mode)
    All warnings except those explicitly filtered out are raised as error.
* ``pytest.ini`` is absent (release mode)
    DeprecationWarnings and PendingDeprecationWarnings are ignored, other
    warnings are passed through.

In practice, tests run from the numpy repo are run in development mode with
``spin``, through the standard ``spin test`` invocation or from an inplace
build with ``pytest numpy``.

This module is imported by every numpy subpackage, so lies at the top level to
simplify circular import issues. For the same reason, it contains no numpy
imports at module scope, instead importing numpy within function calls.
"""
import sys
import os

__all__ = ['PytestTester']


def _show_numpy_info():
    import numpy as np

    print("NumPy version %s" % np.__version__)
    info = np.lib._utils_impl._opt_info()
    print("NumPy CPU features: ", (info if info else 'nothing enabled'))


class PytestTester:
    """
    Pytest test runner.

    A test function is typically added to a package's __init__.py like so::

      from numpy._pytesttester import PytestTester
      test = PytestTester(__name__).test
      del PytestTester

    Calling this test function finds and runs all tests associated with the
    module and all its sub-modules.

    Attributes
    ----------
    module_name : str
        Full path to the package to test.

    Parameters
    ----------
    module_name : module name
        The name of the module to test.

    Notes
    -----
    Unlike the previous ``nose``-based implementation, this class is not
    publicly exposed as it performs some ``numpy``-specific warning
    suppression.

    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.__module__ = module_name

    def __call__(self, label='fast', verbose=1, extra_argv=None,
                 doctests=False, coverage=False, durations=-1, tests=None):
        """
        Run tests for module using pytest.

        Parameters
        ----------
        label : {'fast', 'full'}, optional
            Identifies the tests to run. When set to 'fast', tests decorated
            with `pytest.mark.slow` are skipped, when 'full', the slow marker
            is ignored.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-3. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to pytests.
        doctests : bool, optional
            .. note:: Not supported
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            Requires installation of (pip) pytest-cov.
        durations : int, optional
            If < 0, do nothing, If 0, report time of all tests, if > 0,
            report the time of the slowest `timer` tests. Default is -1.
        tests : test or list of tests
            Tests to be executed with pytest '--pyargs'

        Returns
        -------
        result : bool
            Return True on success, false otherwise.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for
        it. For example, to run all tests for numpy.lib:

        >>> np.lib.test() #doctest: +SKIP

        Examples
        --------
        >>> result = np.lib.test() #doctest: +SKIP
        ...
        1023 passed, 2 skipped, 6 deselected, 1 xfailed in 10.39 seconds
        >>> result
        True

        """
        import pytest
        import warnings

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        # setup the pytest arguments
        pytest_args = ["-l"]

        # offset verbosity. The "-q" cancels a "-v".
        pytest_args += ["-q"]

        if sys.version_info < (3, 12):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                # Filter out distutils cpu warnings (could be localized to
                # distutils tests). ASV has problems with top level import,
                # so fetch module for suppression here.
                from numpy.distutils import cpuinfo

        # Filter out annoying import messages. Want these in both develop and
        # release mode.
        pytest_args += [
            "-W ignore:Not importing directory",
            "-W ignore:numpy.dtype size changed",
            "-W ignore:numpy.ufunc size changed",
            "-W ignore::UserWarning:cpuinfo",
            ]

        # When testing matrices, ignore their PendingDeprecationWarnings
        pytest_args += [
            "-W ignore:the matrix subclass is not",
            "-W ignore:Importing from numpy.matlib is",
            ]

        if doctests:
            pytest_args += ["--doctest-modules"]

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose > 1:
            pytest_args += ["-" + "v"*(verbose - 1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            # not importing at the top level to avoid circular import of module
            from numpy.testing import IS_PYPY
            if IS_PYPY:
                pytest_args += ["-m", "not slow and not slow_pypy"]
            else:
                pytest_args += ["-m", "not slow"]

        elif label != "full":
            pytest_args += ["-m", label]

        if durations >= 0:
            pytest_args += ["--durations=%s" % durations]

        if tests is None:
            tests = [self.module_name]

        pytest_args += ["--pyargs"] + list(tests)

        # run tests.
        _show_numpy_info()

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0
<|endoftext|>"""
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""
import enum

from ._utils import set_module as _set_module

__all__ = ['_NoValue', '_CopyMode']


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


class _NoValueType:
    """Special keyword value.

    The instance of this class may be used as the default value assigned to a
    keyword if no other obvious default (e.g., `None`) is suitable,

    Common reasons for using this keyword are:

    - A new keyword is added to a function, and that function forwards its
      inputs to another function or method which can be defined outside of
      NumPy. For example, ``np.std(x)`` calls ``x.std``, so when a ``keepdims``
      keyword was added that could only be forwarded if the user explicitly
      specified ``keepdims``; downstream array libraries may not have added
      the same keyword, so adding ``x.std(..., keepdims=keepdims)``
      unconditionally could have broken previously working code.
    - A keyword is being deprecated, and a deprecation warning must only be
      emitted when the keyword is used.

    """
    __instance = None
    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


@_set_module("numpy")
class _CopyMode(enum.Enum):
    """
    An enumeration for the copy modes supported
    by numpy.copy() and numpy.array(). The following three modes are supported,

    - ALWAYS: This means that a deep copy of the input
              array will always be taken.
    - IF_NEEDED: This means that a deep copy of the input
                 array will be taken only if necessary.
    - NEVER: This means that the deep copy will never be taken.
             If a copy cannot be avoided then a `ValueError` will be
             raised.

    Note that the buffer-protocol could in theory do copies.  NumPy currently
    assumes an object exporting the buffer protocol will never do this.
    """

    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self):
        # For backwards compatibility
        if self == _CopyMode.ALWAYS:
            return True

        if self == _CopyMode.NEVER:
            return False

        raise ValueError(f"{self} is neither True nor False.")
<|endoftext|>"""
Exceptions and Warnings (:mod:`numpy.exceptions`)
=================================================

General exceptions used by NumPy.  Note that some exceptions may be module
specific, such as linear algebra errors.

.. versionadded:: NumPy 1.25

    The exceptions module is new in NumPy 1.25.  Older exceptions remain
    available through the main NumPy namespace for compatibility.

.. currentmodule:: numpy.exceptions

Warnings
--------
.. autosummary::
   :toctree: generated/

   ComplexWarning             Given when converting complex to real.
   VisibleDeprecationWarning  Same as a DeprecationWarning, but more visible.
   RankWarning                Issued when the design matrix is rank deficient.

Exceptions
----------
.. autosummary::
   :toctree: generated/

    AxisError          Given when an axis was invalid.
    DTypePromotionError   Given when no common dtype could be found.
    TooHardError       Error specific to `numpy.shares_memory`.

"""


__all__ = [
    "ComplexWarning", "VisibleDeprecationWarning", "ModuleDeprecationWarning",
    "TooHardError", "AxisError", "DTypePromotionError"]


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


class ComplexWarning(RuntimeWarning):
    """
    The warning raised when casting a complex dtype to a real dtype.

    As implemented, casting a complex number to a real discards its imaginary
    part, but this behavior may not be what the user actually wants.

    """
    pass


class ModuleDeprecationWarning(DeprecationWarning):
    """Module deprecation warning.

    .. warning::

        This warning should not be used, since nose testing is not relevant
        anymore.

    The nose tester turns ordinary Deprecation warnings into test failures.
    That makes it hard to deprecate whole modules, because they get
    imported by default. So this is a special Deprecation warning that the
    nose tester will let pass without making tests fail.

    """
    pass


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """
    pass


class RankWarning(RuntimeWarning):
    """Matrix rank warning.

    Issued by polynomial functions when the design matrix is rank deficient.

    """
    pass


# Exception used in shares_memory()
class TooHardError(RuntimeError):
    """max_work was exceeded.

    This is raised whenever the maximum number of candidate solutions
    to consider specified by the ``max_work`` parameter is exceeded.
    Assigning a finite number to max_work may have caused the operation
    to fail.

    """
    pass


class AxisError(ValueError, IndexError):
    """Axis supplied was invalid.

    This is raised whenever an ``axis`` parameter is specified that is larger
    than the number of array dimensions.
    For compatibility with code written against older numpy versions, which
    raised a mixture of :exc:`ValueError` and :exc:`IndexError` for this
    situation, this exception subclasses both to ensure that
    ``except ValueError`` and ``except IndexError`` statements continue
    to catch ``AxisError``.

    Parameters
    ----------
    axis : int or str
        The out of bounds axis or a custom exception message.
        If an axis is provided, then `ndim` should be specified as well.
    ndim : int, optional
        The number of array dimensions.
    msg_prefix : str, optional
        A prefix for the exception message.

    Attributes
    ----------
    axis : int, optional
        The out of bounds axis or ``None`` if a custom exception
        message was provided. This should be the axis as passed by
        the user, before any normalization to resolve negative indices.

        .. versionadded:: 1.22
    ndim : int, optional
        The number of array dimensions or ``None`` if a custom exception
        message was provided.

        .. versionadded:: 1.22


    Examples
    --------
    >>> import numpy as np
    >>> array_1d = np.arange(10)
    >>> np.cumsum(array_1d, axis=1)
    Traceback (most recent call last):
      ...
    numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1

    Negative axes are preserved:

    >>> np.cumsum(array_1d, axis=-2)
    Traceback (most recent call last):
      ...
    numpy.exceptions.AxisError: axis -2 is out of bounds for array of dimension 1

    The class constructor generally takes the axis and arrays'
    dimensionality as arguments:

    >>> print(np.exceptions.AxisError(2, 1, msg_prefix='error'))
    error: axis 2 is out of bounds for array of dimension 1

    Alternatively, a custom exception message can be passed:

    >>> print(np.exceptions.AxisError('Custom error message'))
    Custom error message

    """

    __slots__ = ("axis", "ndim", "_msg")

    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is msg_prefix is None:
            # single-argument form: directly set the error message
            self._msg = axis
            self.axis = None
            self.ndim = None
        else:
            self._msg = msg_prefix
            self.axis = axis
            self.ndim = ndim

    def __str__(self):
        axis = self.axis
        ndim = self.ndim

        if axis is ndim is None:
            return self._msg
        else:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if self._msg is not None:
                msg = f"{self._msg}: {msg}"
            return msg


class DTypePromotionError(TypeError):
    """Multiple DTypes could not be converted to a common one.

    This exception derives from ``TypeError`` and is raised whenever dtypes
    cannot be converted to a single common one.  This can be because they
    are of a different category/class or incompatible instances of the same
    one (see Examples).

    Notes
    -----
    Many functions will use promotion to find the correct result and
    implementation.  For these functions the error will typically be chained
    with a more specific error indicating that no implementation was found
    for the input dtypes.

    Typically promotion should be considered "invalid" between the dtypes of
    two arrays when `arr1 == arr2` can safely return all ``False`` because the
    dtypes are fundamentally different.

    Examples
    --------
    Datetimes and complex numbers are incompatible classes and cannot be
    promoted:

    >>> import numpy as np
    >>> np.result_type(np.dtype("M8[s]"), np.complex128)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
     ...
    DTypePromotionError: The DType <class 'numpy.dtype[datetime64]'> could not
    be promoted by <class 'numpy.dtype[complex128]'>. This means that no common
    DType exists for the given inputs. For example they cannot be stored in a
    single array unless the dtype is `object`. The full list of DTypes is:
    (<class 'numpy.dtype[datetime64]'>, <class 'numpy.dtype[complex128]'>)

    For example for structured dtypes, the structure can mismatch and the
    same ``DTypePromotionError`` is given when two structured dtypes with
    a mismatch in their number of fields is given:

    >>> dtype1 = np.dtype([("field1", np.float64), ("field2", np.int64)])
    >>> dtype2 = np.dtype([("field1", np.float64)])
    >>> np.promote_types(dtype1, dtype2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
     ...
    DTypePromotionError: field names `('field1', 'field2')` and `('field1',)`
    mismatch.

    """  # NOQA
    pass
<|endoftext|>"""
============================
Typing (:mod:`numpy.typing`)
============================

.. versionadded:: 1.20

Large parts of the NumPy API have :pep:`484`-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Mypy plugin
-----------

.. versionadded:: 1.21

.. automodule:: numpy.typing.mypy_plugin

.. currentmodule:: numpy.typing

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should instead use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as a invariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: "np.floating[T]", b: "np.floating[T]") -> "np.floating[T]":
    ...     ...

Consequently, the likes of `~numpy.float16`, `~numpy.float32` and
`~numpy.float64` are still sub-types of `~numpy.floating`, but, contrary to
runtime, they're not necessarily considered as sub-classes.

Timedelta64
~~~~~~~~~~~

The `~numpy.timedelta64` class is not considered a subclass of
`~numpy.signedinteger`, the former only inheriting from `~numpy.generic`
while static type checking.

0D arrays
~~~~~~~~~

During runtime numpy aggressively casts any passed 0D arrays into their
corresponding `~numpy.generic` instance. Until the introduction of shape
typing (see :pep:`646`) it is unfortunately not possible to make the
necessary distinction between 0D and >0D arrays. While thus not strictly
correct, all operations are that can potentially perform a 0D-array -> scalar
cast are currently annotated as exclusively returning an `~numpy.ndarray`.

If it is known in advance that an operation *will* perform a
0D-array -> scalar cast, then one can consider manually remedying the
situation with either `typing.cast` or a ``# type: ignore`` comment.

Record array dtypes
~~~~~~~~~~~~~~~~~~~

The dtype of `numpy.recarray`, and the :ref:`routines.array-creation.rec`
functions in general, can be specified in one of two ways:

* Directly via the ``dtype`` argument.
* With up to five helper arguments that operate via `numpy.rec.format_parser`:
  ``formats``, ``names``, ``titles``, ``aligned`` and ``byteorder``.

These two approaches are currently typed as being mutually exclusive,
*i.e.* if ``dtype`` is specified than one may not specify ``formats``.
While this mutual exclusivity is not (strictly) enforced during runtime,
combining both dtype specifiers can lead to unexpected or even downright
buggy behavior.

API
---

"""
# NOTE: The API section will be appended with additional entries
# further down in this file

from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
)

__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

if __doc__ is not None:
    from numpy._typing._add_docstring import _docstrings
    __doc__ += _docstrings
    __doc__ += '\n.. autoclass:: numpy.typing.NBitBase\n'
    del _docstrings

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""A mypy_ plugin for managing a number of platform-specific annotations.
Its functionality can be split into three distinct parts:

* Assigning the (platform-dependent) precisions of certain `~numpy.number`
  subclasses, including the likes of `~numpy.int_`, `~numpy.intp` and
  `~numpy.longlong`. See the documentation on
  :ref:`scalar types <arrays.scalars.built-in>` for a comprehensive overview
  of the affected classes. Without the plugin the precision of all relevant
  classes will be inferred as `~typing.Any`.
* Removing all extended-precision `~numpy.number` subclasses that are
  unavailable for the platform in question. Most notably this includes the
  likes of `~numpy.float128` and `~numpy.complex256`. Without the plugin *all*
  extended-precision types will, as far as mypy is concerned, be available
  to all platforms.
* Assigning the (platform-dependent) precision of `~numpy.ctypeslib.c_intp`.
  Without the plugin the type will default to `ctypes.c_int64`.

  .. versionadded:: 1.22

Examples
--------
To enable the plugin, one must add it to their mypy `configuration file`_:

.. code-block:: ini

    [mypy]
    plugins = numpy.typing.mypy_plugin

.. _mypy: https://mypy-lang.org/
.. _configuration file: https://mypy.readthedocs.io/en/stable/config_file.html

"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    import mypy.types
    from mypy.types import Type
    from mypy.plugin import Plugin, AnalyzeTypeContext
    from mypy.nodes import MypyFile, ImportFrom, Statement
    from mypy.build import PRI_MED

    _HookFunc = Callable[[AnalyzeTypeContext], Type]
    MYPY_EX: None | ModuleNotFoundError = None
except ModuleNotFoundError as ex:
    MYPY_EX = ex

__all__: list[str] = []


def _get_precision_dict() -> dict[str, str]:
    names = [
        ("_NBitByte", np.byte),
        ("_NBitShort", np.short),
        ("_NBitIntC", np.intc),
        ("_NBitIntP", np.intp),
        ("_NBitInt", np.int_),
        ("_NBitLong", np.long),
        ("_NBitLongLong", np.longlong),

        ("_NBitHalf", np.half),
        ("_NBitSingle", np.single),
        ("_NBitDouble", np.double),
        ("_NBitLongDouble", np.longdouble),
    ]
    ret = {}
    module = "numpy._typing"
    for name, typ in names:
        n: int = 8 * typ().dtype.itemsize
        ret[f'{module}._nbit.{name}'] = f"{module}._nbit_base._{n}Bit"
    return ret


def _get_extended_precision_list() -> list[str]:
    extended_names = [
        "uint128",
        "uint256",
        "int128",
        "int256",
        "float80",
        "float96",
        "float128",
        "float256",
        "complex160",
        "complex192",
        "complex256",
        "complex512",
    ]
    return [i for i in extended_names if hasattr(np, i)]

def _get_c_intp_name() -> str:
    # Adapted from `np.core._internal._getintp_ctype`
    char = np.dtype('n').char
