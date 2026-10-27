        GlyphCount = table.AttachList.GlyphCount
        table.AttachList.AttachPoint = [
            table.AttachList.AttachPoint[i] for i in indices if i < GlyphCount
        ]
        table.AttachList.GlyphCount = len(table.AttachList.AttachPoint)
    if hasattr(table, "MarkGlyphSetsDef") and table.MarkGlyphSetsDef:
        markGlyphSets = table.MarkGlyphSetsDef
        for coverage in markGlyphSets.Coverage:
            if coverage:
                coverage.subset(glyphs)

        s.used_mark_sets = [i for i, c in enumerate(markGlyphSets.Coverage) if c.glyphs]
        markGlyphSets.Coverage = [c for c in markGlyphSets.Coverage if c.glyphs]

    return True


def _pruneGDEF(font):
    if "GDEF" not in font:
        return
    gdef = font["GDEF"]
    table = gdef.table
    if not hasattr(table, "VarStore"):
        return

    store = table.VarStore

    usedVarIdxes = set()

    # Collect.
    table.collect_device_varidxes(usedVarIdxes)
    if "GPOS" in font:
        font["GPOS"].table.collect_device_varidxes(usedVarIdxes)

    # Subset.
    varidx_map = store.subset_varidxes(usedVarIdxes)

    # Map.
    table.remap_device_varidxes(varidx_map)
    if "GPOS" in font:
        font["GPOS"].table.remap_device_varidxes(varidx_map)


@_add_method(ttLib.getTableClass("GDEF"))
def prune_post_subset(self, font, options):
    table = self.table
    # XXX check these against OTS
    if table.LigCaretList and not table.LigCaretList.LigGlyphCount:
        table.LigCaretList = None
    if table.MarkAttachClassDef and not table.MarkAttachClassDef.classDefs:
        table.MarkAttachClassDef = None
    if table.GlyphClassDef and not table.GlyphClassDef.classDefs:
        table.GlyphClassDef = None
    if table.AttachList and not table.AttachList.GlyphCount:
        table.AttachList = None
    if hasattr(table, "VarStore"):
        _pruneGDEF(font)
        if table.VarStore.VarDataCount == 0:
            if table.Version == 0x00010003:
                table.Version = 0x00010002
    if (
        not hasattr(table, "MarkGlyphSetsDef")
        or not table.MarkGlyphSetsDef
        or not table.MarkGlyphSetsDef.Coverage
    ):
        table.MarkGlyphSetsDef = None
        if table.Version == 0x00010002:
            table.Version = 0x00010000
    return bool(
        table.LigCaretList
        or table.MarkAttachClassDef
        or table.GlyphClassDef
        or table.AttachList
        or (table.Version >= 0x00010002 and table.MarkGlyphSetsDef)
        or (table.Version >= 0x00010003 and table.VarStore)
    )


@_add_method(ttLib.getTableClass("kern"))
def prune_pre_subset(self, font, options):
    # Prune unknown kern table types
    self.kernTables = [t for t in self.kernTables if hasattr(t, "kernTable")]
    return bool(self.kernTables)


@_add_method(ttLib.getTableClass("kern"))
def subset_glyphs(self, s):
    glyphs = s.glyphs_gsubed
    for t in self.kernTables:
        t.kernTable = {
            (a, b): v
            for (a, b), v in t.kernTable.items()
            if a in glyphs and b in glyphs
        }
    self.kernTables = [t for t in self.kernTables if t.kernTable]
    return bool(self.kernTables)


@_add_method(ttLib.getTableClass("vmtx"))
def subset_glyphs(self, s):
    self.metrics = _dict_subset(self.metrics, s.glyphs)
    for g in s.glyphs_emptied:
        self.metrics[g] = (0, 0)
    return bool(self.metrics)


@_add_method(ttLib.getTableClass("hmtx"))
def subset_glyphs(self, s):
    self.metrics = _dict_subset(self.metrics, s.glyphs)
    for g in s.glyphs_emptied:
        self.metrics[g] = (0, 0)
    return True  # Required table


@_add_method(ttLib.getTableClass("hdmx"))
def subset_glyphs(self, s):
    self.hdmx = {sz: _dict_subset(l, s.glyphs) for sz, l in self.hdmx.items()}
    for sz in self.hdmx:
        for g in s.glyphs_emptied:
            self.hdmx[sz][g] = 0
    return bool(self.hdmx)


@_add_method(ttLib.getTableClass("ankr"))
def subset_glyphs(self, s):
    table = self.table.AnchorPoints
    assert table.Format == 0, "unknown 'ankr' format %s" % table.Format
    table.Anchors = {
        glyph: table.Anchors[glyph] for glyph in s.glyphs if glyph in table.Anchors
    }
    return len(table.Anchors) > 0


@_add_method(ttLib.getTableClass("bsln"))
def closure_glyphs(self, s):
    table = self.table.Baseline
    if table.Format in (2, 3):
        s.glyphs.add(table.StandardGlyph)


@_add_method(ttLib.getTableClass("bsln"))
def subset_glyphs(self, s):
    table = self.table.Baseline
    if table.Format in (1, 3):
        baselines = {
            glyph: table.BaselineValues.get(glyph, table.DefaultBaseline)
            for glyph in s.glyphs
        }
        if len(baselines) > 0:
            mostCommon, _cnt = Counter(baselines.values()).most_common(1)[0]
            table.DefaultBaseline = mostCommon
            baselines = {glyph: b for glyph, b in baselines.items() if b != mostCommon}
        if len(baselines) > 0:
            table.BaselineValues = baselines
        else:
            table.Format = {1: 0, 3: 2}[table.Format]
            del table.BaselineValues
    return True


@_add_method(ttLib.getTableClass("lcar"))
def subset_glyphs(self, s):
    table = self.table.LigatureCarets
    if table.Format in (0, 1):
        table.Carets = {
            glyph: table.Carets[glyph] for glyph in s.glyphs if glyph in table.Carets
        }
        return len(table.Carets) > 0
    else:
        assert False, "unknown 'lcar' format %s" % table.Format


@_add_method(ttLib.getTableClass("gvar"))
def prune_pre_subset(self, font, options):
    if options.notdef_glyph and not options.notdef_outline:
        self.variations[font.glyphOrder[0]] = []
    return True


@_add_method(ttLib.getTableClass("gvar"))
def subset_glyphs(self, s):
    self.variations = _dict_subset(self.variations, s.glyphs)
    self.glyphCount = len(self.variations)
    return bool(self.variations)


def _remap_index_map(s, varidx_map, table_map):
    map_ = {k: varidx_map[v] for k, v in table_map.mapping.items()}
    # Emptied glyphs are remapped to:
    # if GID <= last retained GID, 0/0: delta set for 0/0 is expected to exist & zeros compress well
    # if GID > last retained GID, major/minor of the last retained glyph: will be optimized out by table compiler
    last_idx = varidx_map[table_map.mapping[s.last_retained_glyph]]
    for g, i in s.reverseEmptiedGlyphMap.items():
        map_[g] = last_idx if i > s.last_retained_order else 0
    return map_


@_add_method(ttLib.getTableClass("HVAR"))
def subset_glyphs(self, s):
    table = self.table

    used = set()
    advIdxes_ = set()
    retainAdvMap = False

    if table.AdvWidthMap:
        table.AdvWidthMap.mapping = _dict_subset(table.AdvWidthMap.mapping, s.glyphs)
        used.update(table.AdvWidthMap.mapping.values())
    else:
        used.update(s.reverseOrigGlyphMap.values())
        advIdxes_ = used.copy()
        retainAdvMap = s.options.retain_gids

    if table.LsbMap:
        table.LsbMap.mapping = _dict_subset(table.LsbMap.mapping, s.glyphs)
        used.update(table.LsbMap.mapping.values())
    if table.RsbMap:
        table.RsbMap.mapping = _dict_subset(table.RsbMap.mapping, s.glyphs)
        used.update(table.RsbMap.mapping.values())

    varidx_map = table.VarStore.subset_varidxes(
        used, retainFirstMap=retainAdvMap, advIdxes=advIdxes_
    )

    if table.AdvWidthMap:
        table.AdvWidthMap.mapping = _remap_index_map(s, varidx_map, table.AdvWidthMap)
    if table.LsbMap:
        table.LsbMap.mapping = _remap_index_map(s, varidx_map, table.LsbMap)
    if table.RsbMap:
        table.RsbMap.mapping = _remap_index_map(s, varidx_map, table.RsbMap)

    # TODO Return emptiness...
    return True


@_add_method(ttLib.getTableClass("VVAR"))
def subset_glyphs(self, s):
    table = self.table

    used = set()
    advIdxes_ = set()
    retainAdvMap = False

    if table.AdvHeightMap:
        table.AdvHeightMap.mapping = _dict_subset(table.AdvHeightMap.mapping, s.glyphs)
        used.update(table.AdvHeightMap.mapping.values())
    else:
        used.update(s.reverseOrigGlyphMap.values())
        advIdxes_ = used.copy()
        retainAdvMap = s.options.retain_gids

    if table.TsbMap:
        table.TsbMap.mapping = _dict_subset(table.TsbMap.mapping, s.glyphs)
        used.update(table.TsbMap.mapping.values())
    if table.BsbMap:
        table.BsbMap.mapping = _dict_subset(table.BsbMap.mapping, s.glyphs)
        used.update(table.BsbMap.mapping.values())
    if table.VOrgMap:
        table.VOrgMap.mapping = _dict_subset(table.VOrgMap.mapping, s.glyphs)
        used.update(table.VOrgMap.mapping.values())

    varidx_map = table.VarStore.subset_varidxes(
        used, retainFirstMap=retainAdvMap, advIdxes=advIdxes_
    )

    if table.AdvHeightMap:
        table.AdvHeightMap.mapping = _remap_index_map(s, varidx_map, table.AdvHeightMap)
    if table.TsbMap:
        table.TsbMap.mapping = _remap_index_map(s, varidx_map, table.TsbMap)
    if table.BsbMap:
        table.BsbMap.mapping = _remap_index_map(s, varidx_map, table.BsbMap)
    if table.VOrgMap:
        table.VOrgMap.mapping = _remap_index_map(s, varidx_map, table.VOrgMap)

    # TODO Return emptiness...
    return True


@_add_method(ttLib.getTableClass("VORG"))
def subset_glyphs(self, s):
    self.VOriginRecords = {
        g: v for g, v in self.VOriginRecords.items() if g in s.glyphs
    }
    self.numVertOriginYMetrics = len(self.VOriginRecords)
    return True  # Never drop; has default metrics


@_add_method(ttLib.getTableClass("opbd"))
def subset_glyphs(self, s):
    table = self.table.OpticalBounds
    if table.Format == 0:
        table.OpticalBoundsDeltas = {
            glyph: table.OpticalBoundsDeltas[glyph]
            for glyph in s.glyphs
            if glyph in table.OpticalBoundsDeltas
        }
        return len(table.OpticalBoundsDeltas) > 0
    elif table.Format == 1:
        table.OpticalBoundsPoints = {
            glyph: table.OpticalBoundsPoints[glyph]
            for glyph in s.glyphs
            if glyph in table.OpticalBoundsPoints
        }
        return len(table.OpticalBoundsPoints) > 0
    else:
        assert False, "unknown 'opbd' format %s" % table.Format


@_add_method(ttLib.getTableClass("post"))
def prune_pre_subset(self, font, options):
    if not options.glyph_names:
        self.formatType = 3.0
    return True  # Required table


@_add_method(ttLib.getTableClass("post"))
def subset_glyphs(self, s):
    self.extraNames = []  # This seems to do it
    return True  # Required table


@_add_method(ttLib.getTableClass("prop"))
def subset_glyphs(self, s):
    prop = self.table.GlyphProperties
    if prop.Format == 0:
        return prop.DefaultProperties != 0
    elif prop.Format == 1:
        prop.Properties = {
            g: prop.Properties.get(g, prop.DefaultProperties) for g in s.glyphs
        }
        mostCommon, _cnt = Counter(prop.Properties.values()).most_common(1)[0]
        prop.DefaultProperties = mostCommon
        prop.Properties = {
            g: prop for g, prop in prop.Properties.items() if prop != mostCommon
        }
        if len(prop.Properties) == 0:
            del prop.Properties
            prop.Format = 0
            return prop.DefaultProperties != 0
        return True
    else:
        assert False, "unknown 'prop' format %s" % prop.Format


def _paint_glyph_names(paint, colr):
    result = set()

    def callback(paint):
        if paint.Format in {
            otTables.PaintFormat.PaintGlyph,
            otTables.PaintFormat.PaintColrGlyph,
        }:
            result.add(paint.Glyph)

    paint.traverse(colr, callback)
    return result


@_add_method(ttLib.getTableClass("COLR"))
def closure_glyphs(self, s):
    if self.version > 0:
        # on decompiling COLRv1, we only keep around the raw otTables
        # but for subsetting we need dicts with fully decompiled layers;
        # we store them temporarily in the C_O_L_R_ instance and delete
        # them after we have finished subsetting.
        self.ColorLayers = self._decompileColorLayersV0(self.table)
        self.ColorLayersV1 = {
            rec.BaseGlyph: rec.Paint
            for rec in self.table.BaseGlyphList.BaseGlyphPaintRecord
        }

    decompose = s.glyphs
    while decompose:
        layers = set()
        for g in decompose:
            for layer in self.ColorLayers.get(g, []):
                layers.add(layer.name)

            if self.version > 0:
                paint = self.ColorLayersV1.get(g)
                if paint is not None:
                    layers.update(_paint_glyph_names(paint, self.table))

        layers -= s.glyphs
        s.glyphs.update(layers)
        decompose = layers


@_add_method(ttLib.getTableClass("COLR"))
def subset_glyphs(self, s):
    from fontTools.colorLib.unbuilder import unbuildColrV1
    from fontTools.colorLib.builder import buildColrV1, populateCOLRv0

    # only include glyphs after COLR closure, which in turn comes after cmap and GSUB
    # closure, but importantly before glyf/CFF closures. COLR layers can refer to
    # composite glyphs, and that's ok, since glyf/CFF closures happen after COLR closure
    # and take care of those. If we also included glyphs resulting from glyf/CFF closures
    # when deciding which COLR base glyphs to retain, then we may end up with a situation
    # whereby a COLR base glyph is kept, not because directly requested (cmap)
    # or substituted (GSUB) or referenced by another COLRv1 PaintColrGlyph, but because
    # it corresponds to (has same GID as) a non-COLR glyph that happens to be used as a
    # component in glyf or CFF table. Best case scenario we retain more glyphs than
    # required; worst case we retain incomplete COLR records that try to reference
    # glyphs that are no longer in the final subset font.
    # https://github.com/fonttools/fonttools/issues/2461
    s.glyphs = s.glyphs_colred

    self.ColorLayers = {
        g: self.ColorLayers[g] for g in s.glyphs if g in self.ColorLayers
    }
    if self.version == 0:
        return bool(self.ColorLayers)

    colorGlyphsV1 = unbuildColrV1(self.table.LayerList, self.table.BaseGlyphList)
    self.table.LayerList, self.table.BaseGlyphList = buildColrV1(
        {g: colorGlyphsV1[g] for g in colorGlyphsV1 if g in s.glyphs}
    )
    del self.ColorLayersV1

    if self.table.ClipList is not None:
        clips = self.table.ClipList.clips
        self.table.ClipList.clips = {g: clips[g] for g in clips if g in s.glyphs}

    layersV0 = self.ColorLayers
    if not self.table.BaseGlyphList.BaseGlyphPaintRecord:
        # no more COLRv1 glyphs: downgrade to version 0
        self.version = 0
        del self.table
        return bool(layersV0)

    populateCOLRv0(
        self.table,
        {g: [(layer.name, layer.colorID) for layer in layersV0[g]] for g in layersV0},
    )
    del self.ColorLayers

    # TODO: also prune ununsed varIndices in COLR.VarStore
    return True


@_add_method(ttLib.getTableClass("CPAL"))
def prune_post_subset(self, font, options):
    # Keep whole "CPAL" if "SVG " is present as it may be referenced by the latter
    # via 'var(--color{palette_entry_index}, ...)' CSS color variables.
    # For now we just assume this is the case by the mere presence of "SVG " table,
    # for parsing SVG to collect all the used indices is too much work...
    # TODO(anthrotype): Do The Right Thing (TM).
    if "SVG " in font:
        return True

    colr = font.get("COLR")
    if not colr:  # drop CPAL if COLR was subsetted to empty
        return False

    colors_by_index = defaultdict(list)

    def collect_colors_by_index(paint):
        if hasattr(paint, "PaletteIndex"):  # either solid colors...
            colors_by_index[paint.PaletteIndex].append(paint)
        elif hasattr(paint, "ColorLine"):  # ... or gradient color stops
            for stop in paint.ColorLine.ColorStop:
                colors_by_index[stop.PaletteIndex].append(stop)

    if colr.version == 0:
        for layers in colr.ColorLayers.values():
            for layer in layers:
                colors_by_index[layer.colorID].append(layer)
    else:
        if colr.table.LayerRecordArray:
            for layer in colr.table.LayerRecordArray.LayerRecord:
                colors_by_index[layer.PaletteIndex].append(layer)
        for record in colr.table.BaseGlyphList.BaseGlyphPaintRecord:
            record.Paint.traverse(colr.table, collect_colors_by_index)

    # don't remap palette entry index 0xFFFF, this is always the foreground color
    # https://github.com/fonttools/fonttools/issues/2257
    retained_palette_indices = set(colors_by_index.keys()) - {0xFFFF}
    for palette in self.palettes:
        palette[:] = [c for i, c in enumerate(palette) if i in retained_palette_indices]
        assert len(palette) == len(retained_palette_indices)

    for new_index, old_index in enumerate(sorted(retained_palette_indices)):
        for record in colors_by_index[old_index]:
            if hasattr(record, "colorID"):  # v0
                record.colorID = new_index
            elif hasattr(record, "PaletteIndex"):  # v1
                record.PaletteIndex = new_index
            else:
                raise AssertionError(record)

    self.numPaletteEntries = len(self.palettes[0])

    if self.version == 1:
        kept_labels = []
        for i, label in enumerate(self.paletteEntryLabels):
            if i in retained_palette_indices:
                kept_labels.append(label)
        self.paletteEntryLabels = kept_labels
    return bool(self.numPaletteEntries)


@_add_method(otTables.MathGlyphConstruction)
def closure_glyphs(self, glyphs):
    variants = set()
    for v in self.MathGlyphVariantRecord:
        variants.add(v.VariantGlyph)
    if self.GlyphAssembly:
        for p in self.GlyphAssembly.PartRecords:
            variants.add(p.glyph)
    return variants


@_add_method(otTables.MathVariants)
def closure_glyphs(self, s):
    glyphs = frozenset(s.glyphs)
    variants = set()

    if self.VertGlyphCoverage:
        indices = self.VertGlyphCoverage.intersect(glyphs)
        for i in indices:
            variants.update(self.VertGlyphConstruction[i].closure_glyphs(glyphs))

    if self.HorizGlyphCoverage:
        indices = self.HorizGlyphCoverage.intersect(glyphs)
        for i in indices:
            variants.update(self.HorizGlyphConstruction[i].closure_glyphs(glyphs))

    s.glyphs.update(variants)


@_add_method(ttLib.getTableClass("VARC"))
def subset_glyphs(self, s):
    indices = self.table.Coverage.subset(s.glyphs)
    self.table.VarCompositeGlyphs.VarCompositeGlyph = _list_subset(
        self.table.VarCompositeGlyphs.VarCompositeGlyph, indices
    )
    return bool(self.table.VarCompositeGlyphs.VarCompositeGlyph)


@_add_method(ttLib.getTableClass("VARC"))
def closure_glyphs(self, s):
    if self.table.VarCompositeGlyphs is None:
        return

    glyphMap = {glyphName: i for i, glyphName in enumerate(self.table.Coverage.glyphs)}
    glyphRecords = self.table.VarCompositeGlyphs.VarCompositeGlyph

    glyphs = s.glyphs
    covered = set()
    new = set(glyphs)
    while new:
        oldNew = new
        new = set()
        for glyphName in oldNew:
            if glyphName in covered:
                continue
            idx = glyphMap.get(glyphName)
            if idx is None:
                continue
            glyph = glyphRecords[idx]
            for comp in glyph.components:
                name = comp.glyphName
                glyphs.add(name)
                if name not in covered:
                    new.add(name)


@_add_method(ttLib.getTableClass("VARC"))
def prune_post_subset(self, font, options):
    table = self.table

    axisIndicesList = table.AxisIndicesList
    if axisIndicesList is not None:
        items = axisIndicesList.Item
        usedIndices = set()
        for glyph in table.VarCompositeGlyphs.VarCompositeGlyph:
            for comp in glyph.components:
                if comp.axisIndicesIndex is not None:
                    usedIndices.add(comp.axisIndicesIndex)
        usedIndices = sorted(usedIndices)
        if usedIndices:
            axisIndicesList.Item = _list_subset(items, usedIndices)
            mapping = {old: new for new, old in enumerate(usedIndices)}
            for glyph in table.VarCompositeGlyphs.VarCompositeGlyph:
                for comp in glyph.components:
                    if comp.axisIndicesIndex is not None:
                        comp.axisIndicesIndex = mapping[comp.axisIndicesIndex]
        else:
            table.AxisIndicesList = None

    conditionList = table.ConditionList
    if conditionList is not None:
        conditionTables = conditionList.ConditionTable
        usedIndices = set()
        for glyph in table.VarCompositeGlyphs.VarCompositeGlyph:
            for comp in glyph.components:
                if comp.conditionIndex is not None:
                    usedIndices.add(comp.conditionIndex)
        usedIndices = sorted(usedIndices)
        if usedIndices:
            conditionList.ConditionTable = _list_subset(conditionTables, usedIndices)
            mapping = {old: new for new, old in enumerate(usedIndices)}
            for glyph in table.VarCompositeGlyphs.VarCompositeGlyph:
                for comp in glyph.components:
                    if comp.conditionIndex is not None:
                        comp.conditionIndex = mapping[comp.conditionIndex]
        else:
            table.ConditionList = None

    # Conditions can reference the variation store, so subset them first.
    store = table.MultiVarStore
    if store is not None:
        usedVarIdxes = set()
        table.collect_varidxes(usedVarIdxes)
        varidx_map = store.subset_varidxes(usedVarIdxes)
        table.remap_varidxes(varidx_map)
        if not store:
            table.MultiVarStore = None

    return True


@_add_method(ttLib.getTableClass("MATH"))
def closure_glyphs(self, s):
    if self.table.MathVariants:
        self.table.MathVariants.closure_glyphs(s)


@_add_method(otTables.MathItalicsCorrectionInfo)
def subset_glyphs(self, s):
    indices = self.Coverage.subset(s.glyphs)
    self.ItalicsCorrection = _list_subset(self.ItalicsCorrection, indices)
    self.ItalicsCorrectionCount = len(self.ItalicsCorrection)
    return bool(self.ItalicsCorrectionCount)


@_add_method(otTables.MathTopAccentAttachment)
def subset_glyphs(self, s):
    indices = self.TopAccentCoverage.subset(s.glyphs)
    self.TopAccentAttachment = _list_subset(self.TopAccentAttachment, indices)
    self.TopAccentAttachmentCount = len(self.TopAccentAttachment)
    return bool(self.TopAccentAttachmentCount)


@_add_method(otTables.MathKernInfo)
def subset_glyphs(self, s):
    indices = self.MathKernCoverage.subset(s.glyphs)
    self.MathKernInfoRecords = _list_subset(self.MathKernInfoRecords, indices)
    self.MathKernCount = len(self.MathKernInfoRecords)
    return bool(self.MathKernCount)


@_add_method(otTables.MathGlyphInfo)
def subset_glyphs(self, s):
    if self.MathItalicsCorrectionInfo:
        self.MathItalicsCorrectionInfo.subset_glyphs(s)
    if self.MathTopAccentAttachment:
        self.MathTopAccentAttachment.subset_glyphs(s)
    if self.MathKernInfo:
        self.MathKernInfo.subset_glyphs(s)
    if self.ExtendedShapeCoverage:
        self.ExtendedShapeCoverage.subset(s.glyphs)
    return True


@_add_method(otTables.MathVariants)
def subset_glyphs(self, s):
    if self.VertGlyphCoverage:
        indices = self.VertGlyphCoverage.subset(s.glyphs_wo_math_closure)
        self.VertGlyphConstruction = _list_subset(self.VertGlyphConstruction, indices)
        self.VertGlyphCount = len(self.VertGlyphConstruction)

    if self.HorizGlyphCoverage:
        indices = self.HorizGlyphCoverage.subset(s.glyphs_wo_math_closure)
        self.HorizGlyphConstruction = _list_subset(self.HorizGlyphConstruction, indices)
        self.HorizGlyphCount = len(self.HorizGlyphConstruction)

    return True


@_add_method(ttLib.getTableClass("MATH"))
def subset_glyphs(self, s):
    s.glyphs = s.glyphs_mathed
    if self.table.MathGlyphInfo:
        self.table.MathGlyphInfo.subset_glyphs(s)
    if self.table.MathVariants:
        self.table.MathVariants.subset_glyphs(s)
    return True


@_add_method(ttLib.getTableModule("glyf").Glyph)
def remapComponentsFast(self, glyphidmap):
    if not self.data or struct.unpack(">h", self.data[:2])[0] >= 0:
        return  # Not composite
    data = self.data = bytearray(self.data)
    i = 10
    more = 1
    while more:
        flags = (data[i] << 8) | data[i + 1]
        glyphID = (data[i + 2] << 8) | data[i + 3]
        # Remap
        glyphID = glyphidmap[glyphID]
        data[i + 2] = glyphID >> 8
        data[i + 3] = glyphID & 0xFF
        i += 4
        flags = int(flags)

        if flags & 0x0001:
            i += 4  # ARG_1_AND_2_ARE_WORDS
        else:
            i += 2
        if flags & 0x0008:
            i += 2  # WE_HAVE_A_SCALE
        elif flags & 0x0040:
            i += 4  # WE_HAVE_AN_X_AND_Y_SCALE
        elif flags & 0x0080:
            i += 8  # WE_HAVE_A_TWO_BY_TWO
        more = flags & 0x0020  # MORE_COMPONENTS


@_add_method(ttLib.getTableClass("glyf"))
def closure_glyphs(self, s):
    glyphSet = self.glyphs
    decompose = s.glyphs
    while decompose:
        components = set()
        for g in decompose:
            if g not in glyphSet:
                continue
            gl = glyphSet[g]
            for c in gl.getComponentNames(self):
                components.add(c)
        components -= s.glyphs
        s.glyphs.update(components)
        decompose = components


@_add_method(ttLib.getTableClass("glyf"))
def prune_pre_subset(self, font, options):
    if options.notdef_glyph and not options.notdef_outline:
        g = self[self.glyphOrder[0]]
        # Yay, easy!
        g.__dict__.clear()
        g.data = b""
    return True


@_add_method(ttLib.getTableClass("glyf"))
def subset_glyphs(self, s):
    self.glyphs = _dict_subset(self.glyphs, s.glyphs)
    if not s.options.retain_gids:
        indices = [i for i, g in enumerate(self.glyphOrder) if g in s.glyphs]
        glyphmap = {o: n for n, o in enumerate(indices)}
        for v in self.glyphs.values():
            if hasattr(v, "data"):
                v.remapComponentsFast(glyphmap)
    Glyph = ttLib.getTableModule("glyf").Glyph
    for g in s.glyphs_emptied:
        self.glyphs[g] = Glyph()
        self.glyphs[g].data = b""
    self.glyphOrder = [
        g for g in self.glyphOrder if g in s.glyphs or g in s.glyphs_emptied
    ]
    # Don't drop empty 'glyf' tables, otherwise 'loca' doesn't get subset.
    return True


@_add_method(ttLib.getTableClass("glyf"))
def prune_post_subset(self, font, options):
    remove_hinting = not options.hinting
    for v in self.glyphs.values():
        v.trim(remove_hinting=remove_hinting)
    return True


@_add_method(ttLib.getTableClass("cmap"))
def closure_glyphs(self, s):
    tables = [t for t in self.tables if t.isUnicode()]

    # Closure unicodes, which for now is pulling in bidi mirrored variants
    if s.options.bidi_closure:
        additional_unicodes = set()
        for u in s.unicodes_requested:
            mirror_u = mirrored(u)
            if mirror_u is not None:
                additional_unicodes.add(mirror_u)
        s.unicodes_requested.update(additional_unicodes)

    # Close glyphs
    for table in tables:
        if table.format == 14:
            for varSelector, cmap in table.uvsDict.items():
                if varSelector not in s.unicodes_requested:
                    continue
                glyphs = {g for u, g in cmap if u in s.unicodes_requested}
                if None in glyphs:
                    glyphs.remove(None)
                s.glyphs.update(glyphs)
        else:
            cmap = table.cmap
            intersection = s.unicodes_requested.intersection(cmap.keys())
            s.glyphs.update(cmap[u] for u in intersection)

    # Calculate unicodes_missing
    s.unicodes_missing = s.unicodes_requested.copy()
    for table in tables:
        s.unicodes_missing.difference_update(table.cmap)


@_add_method(ttLib.getTableClass("cmap"))
def prune_pre_subset(self, font, options):
    if not options.legacy_cmap:
        # Drop non-Unicode / non-Symbol cmaps
        self.tables = [t for t in self.tables if t.isUnicode() or t.isSymbol()]
    if not options.symbol_cmap:
        self.tables = [t for t in self.tables if not t.isSymbol()]
    # TODO(behdad) Only keep one subtable?
    # For now, drop format=0 which can't be subset_glyphs easily?
    self.tables = [t for t in self.tables if t.format != 0]
    self.numSubTables = len(self.tables)
    return True  # Required table


@_add_method(ttLib.getTableClass("cmap"))
def subset_glyphs(self, s):
    s.glyphs = None  # We use s.glyphs_requested and s.unicodes_requested only

    tables_format12_bmp = []
    table_plat0_enc3 = {}  # Unicode platform, Unicode BMP only, keyed by language
    table_plat3_enc1 = {}  # Windows platform, Unicode BMP, keyed by language

    for t in self.tables:
        if t.platformID == 0 and t.platEncID == 3:
            table_plat0_enc3[t.language] = t
        if t.platformID == 3 and t.platEncID == 1:
            table_plat3_enc1[t.language] = t

        if t.format == 14:
            # TODO(behdad) We drop all the default-UVS mappings
            # for glyphs_requested.  So it's the caller's responsibility to make
            # sure those are included.
            t.uvsDict = {
                v: [
                    (u, g)
                    for u, g in l
                    if g in s.glyphs_requested or u in s.unicodes_requested
                ]
                for v, l in t.uvsDict.items()
                if v in s.unicodes_requested
            }
            t.uvsDict = {v: l for v, l in t.uvsDict.items() if l}
        elif t.isUnicode():
            t.cmap = {
                u: g
                for u, g in t.cmap.items()
                if g in s.glyphs_requested or u in s.unicodes_requested
            }
            # Collect format 12 tables that hold only basic multilingual plane
            # codepoints.
            if t.format == 12 and t.cmap and max(t.cmap.keys()) < 0x10000:
                tables_format12_bmp.append(t)
        else:
            t.cmap = {u: g for u, g in t.cmap.items() if g in s.glyphs_requested}

    # Fomat 12 tables are redundant if they contain just the same BMP codepoints
    # their little BMP-only encoding siblings contain.
    for t in tables_format12_bmp:
        if (
            t.platformID == 0  # Unicode platform
            and t.platEncID == 4  # Unicode full repertoire
            and t.language in table_plat0_enc3  # Have a BMP-only sibling?
            and table_plat0_enc3[t.language].cmap == t.cmap
        ):
            t.cmap.clear()
        elif (
            t.platformID == 3  # Windows platform
            and t.platEncID == 10  # Unicode full repertoire
            and t.language in table_plat3_enc1  # Have a BMP-only sibling?
            and table_plat3_enc1[t.language].cmap == t.cmap
        ):
            t.cmap.clear()

    self.tables = [t for t in self.tables if (t.cmap if t.format != 14 else t.uvsDict)]
    self.numSubTables = len(self.tables)
    # TODO(behdad) Convert formats when needed.
    # In particular, if we have a format=12 without non-BMP
    # characters, convert it to format=4 if there's not one.
    return True  # Required table


@_add_method(ttLib.getTableClass("DSIG"))
def prune_pre_subset(self, font, options):
    # Drop all signatures since they will be invalid
    self.usNumSigs = 0
    self.signatureRecords = []
    return True


@_add_method(ttLib.getTableClass("maxp"))
def prune_pre_subset(self, font, options):
    if not options.hinting:
        if self.tableVersion == 0x00010000:
            self.maxZones = 1
            self.maxTwilightPoints = 0
            self.maxStorage = 0
            self.maxFunctionDefs = 0
            self.maxInstructionDefs = 0
            self.maxStackElements = 0
            self.maxSizeOfInstructions = 0
    return True


NAME_IDS_TO_OBFUSCATE = {1, 2, 3, 4, 6, 16, 17, 18}


@_add_method(ttLib.getTableClass("name"))
def prune_post_subset(self, font, options):
    visitor = NameRecordVisitor()
    visitor.visit(font)
    nameIDs = set(options.name_IDs) | visitor.seen
    if "*" in options.name_IDs:
        nameIDs |= {n.nameID for n in self.names if n.nameID < 256}
    self.names = [n for n in self.names if n.nameID in nameIDs]
    if not options.name_legacy:
        # TODO(behdad) Sometimes (eg Apple Color Emoji) there's only a macroman
        # entry for Latin and no Unicode names.
        self.names = [n for n in self.names if n.isUnicode()]
    # TODO(behdad) Option to keep only one platform's
    if "*" not in options.name_languages:
        # TODO(behdad) This is Windows-platform specific!
        self.names = [n for n in self.names if n.langID in options.name_languages]
    if options.obfuscate_names:
        namerecs = []
        # Preserve names to be scrambled or dropped elsewhere so that other
        # parts of the font don't break.
        needRemapping = visitor.seen.intersection(NAME_IDS_TO_OBFUSCATE)
        if needRemapping:
            _remap_select_name_ids(font, needRemapping)
        for n in self.names:
            if n.nameID in [1, 4]:
                n.string = ".\x7f".encode("utf_16_be") if n.isUnicode() else ".\x7f"
            elif n.nameID in [2, 6]:
                n.string = "\x7f".encode("utf_16_be") if n.isUnicode() else "\x7f"
            elif n.nameID == 3:
                n.string = ""
            elif n.nameID in [16, 17, 18]:
                continue
            namerecs.append(n)
        self.names = namerecs
    return True  # Required table


def _remap_select_name_ids(font: ttLib.TTFont, needRemapping: set[int]) -> None:
    """Remap a set of IDs so that the originals can be safely scrambled or
    dropped.

    For each name record whose name id is in the `needRemapping` set, make a copy
    and allocate a new unused name id in the font-specific range (> 255).

    Finally update references to these in the `fvar` and `STAT` tables.
    """

    if "fvar" not in font and "STAT" not in font:
        return

    name = font["name"]

    # 1. Assign new IDs for names to be preserved.
    existingIds = {record.nameID for record in name.names}
    remapping = {}
    nextId = name._findUnusedNameID() - 1  # Should skip gaps in name IDs.
    for nameId in needRemapping:
        nextId += 1  # We should have complete freedom until 32767.
        assert nextId not in existingIds, "_findUnusedNameID did not skip gaps"
        if nextId > 32767:
            raise ValueError("Ran out of name IDs while trying to remap existing ones.")
        remapping[nameId] = nextId

    # 2. Copy records to use the new ID. We can't rewrite them in place, because
    #    that could make IDs 1 to 6 "disappear" from code that follows. Some
    #    tools that produce EOT fonts expect them to exist, even when they're
    #    scrambled. See https://github.com/fonttools/fonttools/issues/165.
    copiedRecords = []
    for record in name.names:
        if record.nameID not in needRemapping:
            continue
        recordCopy = makeName(
            record.string,
            remapping[record.nameID],
            record.platformID,
            record.platEncID,
            record.langID,
        )
        copiedRecords.append(recordCopy)
    name.names.extend(copiedRecords)

    # 3. Rewrite the corresponding IDs in other tables. For now, care only about
    #    STAT and fvar. If more tables need to be changed, consider adapting
    #    NameRecordVisitor to rewrite IDs wherever it finds them.
    fvar = font.get("fvar")
    if fvar is not None:
        for axis in fvar.axes:
            axis.axisNameID = remapping.get(axis.axisNameID, axis.axisNameID)
        for instance in fvar.instances:
            nameID = instance.subfamilyNameID
            instance.subfamilyNameID = remapping.get(nameID, nameID)
            nameID = instance.postscriptNameID
            instance.postscriptNameID = remapping.get(nameID, nameID)

    stat = font.get("STAT")
    if stat is None:
        return
    elidedNameID = stat.table.ElidedFallbackNameID
    stat.table.ElidedFallbackNameID = remapping.get(elidedNameID, elidedNameID)
    if stat.table.DesignAxisRecord:
        for axis in stat.table.DesignAxisRecord.Axis:
            axis.AxisNameID = remapping.get(axis.AxisNameID, axis.AxisNameID)
    if stat.table.AxisValueArray:
        for value in stat.table.AxisValueArray.AxisValue:
            value.ValueNameID = remapping.get(value.ValueNameID, value.ValueNameID)


@_add_method(ttLib.getTableClass("head"))
def prune_post_subset(self, font, options):
    # Force re-compiling head table, to update any recalculated values.
    return True


# TODO(behdad) OS/2 ulCodePageRange?
# TODO(behdad) Drop AAT tables.
# TODO(behdad) Drop unneeded GSUB/GPOS Script/LangSys entries.
# TODO(behdad) Drop empty GSUB/GPOS, and GDEF if no GSUB/GPOS left
# TODO(behdad) Drop GDEF subitems if unused by lookups
# TODO(behdad) Avoid recursing too much (in GSUB/GPOS and in CFF)
# TODO(behdad) Text direction considerations.
# TODO(behdad) Text script / language considerations.
# TODO(behdad) Optionally drop 'kern' table if GPOS available
# TODO(behdad) Implement --unicode='*' to choose all cmap'ed
# TODO(behdad) Drop old-spec Indic scripts


class Options(object):
    class OptionError(Exception):
        pass

    class UnknownOptionError(OptionError):
        pass

    # spaces in tag names (e.g. "SVG ", "cvt ") are stripped by the argument parser
    _drop_tables_default = [
        "JSTF",
        "DSIG",
        "EBDT",
        "EBLC",
        "EBSC",
        "PCLT",
        "LTSH",
    ]
    _drop_tables_default += ["Feat", "Glat", "Gloc", "Silf", "Sill"]  # Graphite
    _no_subset_tables_default = [
        "avar",
        "BASE",
        "fvar",
        "gasp",
        "head",
        "hhea",
        "maxp",
        "vhea",
        "OS/2",
        "loca",
        "name",
        "cvt",
        "fpgm",
        "prep",
        "VDMX",
        "DSIG",
        "CPAL",
        "MVAR",
        "cvar",
        "STAT",
    ]
    _hinting_tables_default = ["cvt", "cvar", "fpgm", "prep", "hdmx", "VDMX"]

    # Based on HarfBuzz shapers
    _layout_features_groups = {
        # Default shaper
        "common": ["rvrn", "ccmp", "liga", "locl", "mark", "mkmk", "rlig"],
        "fractions": ["frac", "numr", "dnom"],
        "horizontal": ["calt", "clig", "curs", "kern", "rclt"],
        "vertical": ["valt", "vert", "vkrn", "vpal", "vrt2"],
        "ltr": ["ltra", "ltrm"],
        "rtl": ["rtla", "rtlm"],
        "rand": ["rand"],
        "justify": ["jalt"],
        "private": ["Harf", "HARF", "Buzz", "BUZZ"],
        "east_asian_spacing": ["chws", "vchw", "halt", "vhal", "palt"],
        # Complex shapers
        "arabic": [
            "init",
            "medi",
            "fina",
            "isol",
            "med2",
            "fin2",
            "fin3",
            "cswh",
            "mset",
            "stch",
        ],
        "hangul": ["ljmo", "vjmo", "tjmo"],
        "tibetan": ["abvs", "blws", "abvm", "blwm"],
        "indic": [
            "nukt",
            "akhn",
            "rphf",
            "rkrf",
            "pref",
            "blwf",
            "half",
            "abvf",
            "pstf",
            "cfar",
            "vatu",
            "cjct",
            "init",
            "pres",
            "abvs",
            "blws",
            "psts",
            "haln",
            "dist",
            "abvm",
            "blwm",
        ],
    }
    _layout_features_default = _uniq_sort(
        sum(iter(_layout_features_groups.values()), [])
    )

    def __init__(self, **kwargs):
        self.drop_tables = self._drop_tables_default[:]
        self.no_subset_tables = self._no_subset_tables_default[:]
        self.passthrough_tables = False  # keep/drop tables we can't subset
        self.hinting_tables = self._hinting_tables_default[:]
        self.legacy_kern = False  # drop 'kern' table if GPOS available
        self.layout_closure = True
        self.layout_features = self._layout_features_default[:]
        self.layout_scripts = ["*"]
        self.ignore_missing_glyphs = False
        self.ignore_missing_unicodes = True
        self.hinting = True
        self.glyph_names = False
        self.legacy_cmap = False
        self.symbol_cmap = False
        self.name_IDs = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ]  # https://github.com/fonttools/fonttools/issues/1170#issuecomment-364631225
        self.name_legacy = False
        self.name_languages = [0x0409]  # English
        self.obfuscate_names = False  # to make webfont unusable as a system font
        self.retain_gids = False
        self.notdef_glyph = True  # gid0 for TrueType / .notdef for CFF
        self.notdef_outline = False  # No need for notdef to have an outline really
        self.recommended_glyphs = False  # gid1, gid2, gid3 for TrueType
        self.recalc_bounds = False  # Recalculate font bounding boxes
        self.recalc_timestamp = False  # Recalculate font modified timestamp
        self.prune_unicode_ranges = True  # Clear unused 'ulUnicodeRange' bits
        self.prune_codepage_ranges = True  # Clear unused 'ulCodePageRange' bits
        self.recalc_average_width = False  # update 'xAvgCharWidth'
        self.recalc_max_context = False  # update 'usMaxContext'
        self.canonical_order = None  # Order tables as recommended
        self.flavor = None  # May be 'woff' or 'woff2'
        self.with_zopfli = False  # use zopfli instead of zlib for WOFF 1.0
        self.desubroutinize = False  # Desubroutinize CFF CharStrings
        self.harfbuzz_repacker = USE_HARFBUZZ_REPACKER.default
        self.verbose = False
        self.timing = False
        self.xml = False
        self.font_number = -1
        self.pretty_svg = False
        self.lazy = True
        self.bidi_closure = True

        self.set(**kwargs)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise self.UnknownOptionError("Unknown option '%s'" % k)
            setattr(self, k, v)

    def parse_opts(self, argv, ignore_unknown=[]):
        posargs = []
        passthru_options = []
        for a in argv:
            orig_a = a
            if not a.startswith("--"):
                posargs.append(a)
                continue
            a = a[2:]
            i = a.find("=")
            op = "="
            if i == -1:
                if a.startswith("no-"):
                    k = a[3:]
                    if k == "canonical-order":
                        # reorderTables=None is faster than False (the latter
                        # still reorders to "keep" the original table order)
                        v = None
                    else:
                        v = False
                else:
                    k = a
                    v = True
                if k.endswith("?"):
                    k = k[:-1]
                    v = "?"
            else:
                k = a[:i]
                if k[-1] in "-+":
                    op = k[-1] + "="  # Op is '-=' or '+=' now.
                    k = k[:-1]
                v = a[i + 1 :]
            ok = k
            k = k.replace("-", "_")
            if not hasattr(self, k):
                if ignore_unknown is True or ok in ignore_unknown:
                    passthru_options.append(orig_a)
                    continue
                else:
                    raise self.UnknownOptionError("Unknown option '%s'" % a)

            ov = getattr(self, k)
            if v == "?":
                print("Current setting for '%s' is: %s" % (ok, ov))
                continue
            if isinstance(ov, bool):
                v = bool(v)
            elif isinstance(ov, int):
                v = int(v)
            elif isinstance(ov, str):
                v = str(v)  # redundant
            elif isinstance(ov, list):
                if isinstance(v, bool):
                    raise self.OptionError(
                        "Option '%s' requires values to be specified using '='" % a
                    )
                vv = v.replace(",", " ").split()
                if vv == [""]:
                    vv = []
                vv = [int(x, 0) if len(x) and x[0] in "0123456789" else x for x in vv]
                if op == "=":
                    v = vv
                elif op == "+=":
                    v = ov
                    v.extend(vv)
                elif op == "-=":
                    v = ov
                    for x in vv:
                        if x in v:
                            v.remove(x)
                else:
                    assert False

            setattr(self, k, v)

        return posargs + passthru_options


class Subsetter(object):
    class SubsettingError(Exception):
        pass

    class MissingGlyphsSubsettingError(SubsettingError):
        pass

    class MissingUnicodesSubsettingError(SubsettingError):
        pass

    def __init__(self, options=None):
        if not options:
            options = Options()

        self.options = options
        self.unicodes_requested = set()
        self.glyph_names_requested = set()
        self.glyph_ids_requested = set()

    def populate(self, glyphs=[], gids=[], unicodes=[], text=""):
        self.unicodes_requested.update(unicodes)
        if isinstance(text, bytes):
            text = text.decode("utf_8")
        text_utf32 = text.encode("utf-32-be")
        nchars = len(text_utf32) // 4
        for u in struct.unpack(">%dL" % nchars, text_utf32):
            self.unicodes_requested.add(u)
        self.glyph_names_requested.update(glyphs)
        self.glyph_ids_requested.update(gids)

    def _prune_pre_subset(self, font):
        for tag in self._sort_tables(font):
            if (
                tag.strip() in self.options.drop_tables
                or (
                    tag.strip() in self.options.hinting_tables
                    and not self.options.hinting
                )
                or (tag == "kern" and (not self.options.legacy_kern and "GPOS" in font))
            ):
                log.info("%s dropped", tag)
                del font[tag]
                continue

            clazz = ttLib.getTableClass(tag)

            if hasattr(clazz, "prune_pre_subset"):
                with timer("load '%s'" % tag):
                    table = font[tag]
                with timer("prune '%s'" % tag):
                    retain = table.prune_pre_subset(font, self.options)
                if not retain:
                    log.info("%s pruned to empty; dropped", tag)
                    del font[tag]
                    continue
                else:
                    log.info("%s pruned", tag)

    def _closure_glyphs(self, font):
        realGlyphs = set(font.getGlyphOrder())
        self.orig_glyph_order = glyph_order = font.getGlyphOrder()

        self.glyphs_requested = set()
        self.glyphs_requested.update(self.glyph_names_requested)
        self.glyphs_requested.update(
            glyph_order[i] for i in self.glyph_ids_requested if i < len(glyph_order)
        )

        self.glyphs_missing = set()
        self.glyphs_missing.update(self.glyphs_requested.difference(realGlyphs))
        self.glyphs_missing.update(
            i for i in self.glyph_ids_requested if i >= len(glyph_order)
        )
        if self.glyphs_missing:
            log.info("Missing requested glyphs: %s", self.glyphs_missing)
            if not self.options.ignore_missing_glyphs:
                raise self.MissingGlyphsSubsettingError(self.glyphs_missing)

        self.glyphs = self.glyphs_requested.copy()

        self.unicodes_missing = set()
        if "cmap" in font:
            with timer("close glyph list over 'cmap'"):
                font["cmap"].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
        self.glyphs_cmaped = frozenset(self.glyphs)
        if self.unicodes_missing:
            missing = ["U+%04X" % u for u in self.unicodes_missing]
            log.info("Missing glyphs for requested Unicodes: %s", missing)
            if not self.options.ignore_missing_unicodes:
                raise self.MissingUnicodesSubsettingError(missing)
            del missing

        if self.options.notdef_glyph:
            if "glyf" in font:
                self.glyphs.add(font.getGlyphName(0))
                log.info("Added gid0 to subset")
            else:
                self.glyphs.add(".notdef")
                log.info("Added .notdef to subset")
        if self.options.recommended_glyphs:
            if "glyf" in font:
                for i in range(min(4, len(font.getGlyphOrder()))):
                    self.glyphs.add(font.getGlyphName(i))
                log.info("Added first four glyphs to subset")

        self.glyphs_wo_math_closure = frozenset(self.glyphs)
        if "MATH" in font:
            with timer("close glyph list over 'MATH'"):
                log.info(
                    "Closing glyph list over 'MATH': %d glyphs before", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
                font["MATH"].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info(
                    "Closed glyph list over 'MATH': %d glyphs after", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
        self.glyphs_mathed = frozenset(self.glyphs)

        if self.options.layout_closure and "GSUB" in font:
            with timer("close glyph list over 'GSUB'"):
                log.info(
                    "Closing glyph list over 'GSUB': %d glyphs before", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
                font["GSUB"].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info(
                    "Closed glyph list over 'GSUB': %d glyphs after", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
        self.glyphs_gsubed = frozenset(self.glyphs)

        for table in ("COLR", "bsln"):
            if table in font:
                with timer("close glyph list over '%s'" % table):
                    log.info(
                        "Closing glyph list over '%s': %d glyphs before",
                        table,
                        len(self.glyphs),
                    )
                    log.glyphs(self.glyphs, font=font)
                    font[table].closure_glyphs(self)
                    self.glyphs.intersection_update(realGlyphs)
                    log.info(
                        "Closed glyph list over '%s': %d glyphs after",
                        table,
                        len(self.glyphs),
                    )
                    log.glyphs(self.glyphs, font=font)
            setattr(self, f"glyphs_{table.lower()}ed", frozenset(self.glyphs))

        if "VARC" in font:
            with timer("close glyph list over 'VARC'"):
                log.info(
                    "Closing glyph list over 'VARC': %d glyphs before", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
                font["VARC"].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info(
                    "Closed glyph list over 'VARC': %d glyphs after", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
        self.glyphs_glyfed = frozenset(self.glyphs)

        if "glyf" in font:
            with timer("close glyph list over 'glyf'"):
                log.info(
                    "Closing glyph list over 'glyf': %d glyphs before", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
                font["glyf"].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info(
                    "Closed glyph list over 'glyf': %d glyphs after", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
        self.glyphs_glyfed = frozenset(self.glyphs)

        if "CFF " in font:
            with timer("close glyph list over 'CFF '"):
                log.info(
                    "Closing glyph list over 'CFF ': %d glyphs before", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
                font["CFF "].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info(
                    "Closed glyph list over 'CFF ': %d glyphs after", len(self.glyphs)
                )
                log.glyphs(self.glyphs, font=font)
        self.glyphs_cffed = frozenset(self.glyphs)

        self.glyphs_retained = frozenset(self.glyphs)

        order = font.getReverseGlyphMap()
        self.reverseOrigGlyphMap = {g: order[g] for g in self.glyphs_retained}

        self.last_retained_order = max(self.reverseOrigGlyphMap.values())
        self.last_retained_glyph = font.getGlyphOrder()[self.last_retained_order]

        self.glyphs_emptied = frozenset()
        if self.options.retain_gids:
            self.glyphs_emptied = {
                g
                for g in realGlyphs - self.glyphs_retained
                if order[g] <= self.last_retained_order
            }

        self.reverseEmptiedGlyphMap = {g: order[g] for g in self.glyphs_emptied}

        if not self.options.retain_gids:
            new_glyph_order = [g for g in glyph_order if g in self.glyphs_retained]
        else:
            new_glyph_order = [
                g for g in glyph_order if font.getGlyphID(g) <= self.last_retained_order
            ]
        # We'll call font.setGlyphOrder() at the end of _subset_glyphs when all
        # tables have been subsetted. Below, we use the new glyph order to get
        # a map from old to new glyph indices, which can be useful when
        # subsetting individual tables (e.g. SVG) that refer to GIDs.
        self.new_glyph_order = new_glyph_order
        self.glyph_index_map = {
            order[new_glyph_order[i]]: i for i in range(len(new_glyph_order))
        }

        log.info("Retaining %d glyphs", len(self.glyphs_retained))

        del self.glyphs

    def _subset_glyphs(self, font):
        self.used_mark_sets = []
        for tag in self._sort_tables(font):
            clazz = ttLib.getTableClass(tag)

            if tag.strip() in self.options.no_subset_tables:
                log.info("%s subsetting not needed", tag)
            elif hasattr(clazz, "subset_glyphs"):
                with timer("subset '%s'" % tag):
                    table = font[tag]
                    self.glyphs = self.glyphs_retained
                    retain = table.subset_glyphs(self)
                    del self.glyphs
                if not retain:
                    log.info("%s subsetted to empty; dropped", tag)
                    del font[tag]
                else:
                    log.info("%s subsetted", tag)
            elif self.options.passthrough_tables:
                log.info("%s NOT subset; don't know how to subset", tag)
            else:
                log.warning("%s NOT subset; don't know how to subset; dropped", tag)
                del font[tag]

        with timer("subset GlyphOrder"):
            font.setGlyphOrder(self.new_glyph_order)

    def _prune_post_subset(self, font):
        tableTags = font.keys()
        # Prune the name table last because when we're pruning the name table,
        # we visit each table in the font to see what name table records are
        # still in use.
        if "name" in tableTags:
            tableTags.remove("name")
            tableTags.append("name")
        for tag in tableTags:
            if tag == "GlyphOrder":
                continue
            if tag == "OS/2":
                if self.options.prune_unicode_ranges:
                    old_uniranges = font[tag].getUnicodeRanges()
                    new_uniranges = font[tag].recalcUnicodeRanges(font, pruneOnly=True)
                    if old_uniranges != new_uniranges:
                        log.info(
                            "%s Unicode ranges pruned: %s", tag, sorted(new_uniranges)
                        )
                if self.options.prune_codepage_ranges and font[tag].version >= 1:
                    # codepage range fields were added with OS/2 format 1
                    # https://learn.microsoft.com/en-us/typography/opentype/spec/os2#version-1
                    old_codepages = font[tag].getCodePageRanges()
                    new_codepages = font[tag].recalcCodePageRanges(font, pruneOnly=True)
                    if old_codepages != new_codepages:
                        log.info(
                            "%s CodePage ranges pruned: %s",
                            tag,
                            sorted(new_codepages),
                        )
                if self.options.recalc_average_width:
                    old_avg_width = font[tag].xAvgCharWidth
                    new_avg_width = font[tag].recalcAvgCharWidth(font)
                    if old_avg_width != new_avg_width:
                        log.info("%s xAvgCharWidth updated: %d", tag, new_avg_width)
                if self.options.recalc_max_context:
                    max_context = maxCtxFont(font)
                    if max_context != font[tag].usMaxContext:
                        font[tag].usMaxContext = max_context
                        log.info("%s usMaxContext updated: %d", tag, max_context)
            clazz = ttLib.getTableClass(tag)
            if hasattr(clazz, "prune_post_subset"):
                with timer("prune '%s'" % tag):
                    table = font[tag]
                    retain = table.prune_post_subset(font, self.options)
                if not retain:
                    log.info("%s pruned to empty; dropped", tag)
                    del font[tag]
                else:
                    log.info("%s pruned", tag)

    def _sort_tables(self, font):
        tagOrder = ["GDEF", "GPOS", "GSUB", "fvar", "avar", "gvar", "name", "glyf"]
        tagOrder = {t: i + 1 for i, t in enumerate(tagOrder)}
        tags = sorted(font.keys(), key=lambda tag: tagOrder.get(tag, 0))
        return [t for t in tags if t != "GlyphOrder"]

    def subset(self, font):
        self._prune_pre_subset(font)
        self._closure_glyphs(font)
        self._subset_glyphs(font)
        self._prune_post_subset(font)


@timer("load font")
def load_font(fontFile, options, checkChecksums=0, dontLoadGlyphNames=False, lazy=True):
    font = ttLib.TTFont(
        fontFile,
        checkChecksums=checkChecksums,
        recalcBBoxes=options.recalc_bounds,
        recalcTimestamp=options.recalc_timestamp,
        lazy=lazy,
        fontNumber=options.font_number,
    )

    # Hack:
    #
    # If we don't need glyph names, change 'post' class to not try to
    # load them.	It avoid lots of headache with broken fonts as well
    # as loading time.
    #
    # Ideally ttLib should provide a way to ask it to skip loading
    # glyph names.	But it currently doesn't provide such a thing.
    #
    if dontLoadGlyphNames:
        post = ttLib.getTableClass("post")
        saved = post.decode_format_2_0
        post.decode_format_2_0 = post.decode_format_3_0
        f = font["post"]
        if f.formatType == 2.0:
            f.formatType = 3.0
        post.decode_format_2_0 = saved

    return font


@timer("compile and save font")
def save_font(font, outfile, options):
    if options.with_zopfli and options.flavor == "woff":
        from fontTools.ttLib import sfnt

        sfnt.USE_ZOPFLI = True
    font.flavor = options.flavor
    font.cfg[USE_HARFBUZZ_REPACKER] = options.harfbuzz_repacker
    font.save(outfile, reorderTables=options.canonical_order)


def parse_unicodes(s):
    import re

    s = re.sub(r"0[xX]", " ", s)
    s = re.sub(r"[<+>,;&#\\xXuU\n	]", " ", s)
    l = []
    for item in s.split():
        fields = item.split("-")
        if len(fields) == 1:
            l.append(int(item, 16))
        else:
            start, end = fields
            l.extend(range(int(start, 16), int(end, 16) + 1))
    return l


def parse_gids(s):
    l = []
    for item in s.replace(",", " ").split():
        fields = item.split("-")
        if len(fields) == 1:
            l.append(int(fields[0]))
        else:
            l.extend(range(int(fields[0]), int(fields[1]) + 1))
    return l


def parse_glyphs(s):
    return s.replace(",", " ").split()


def usage():
    print("usage:", __usage__, file=sys.stderr)
    print("Try fonttools subset --help for more information.\n", file=sys.stderr)


@timer("make one with everything (TOTAL TIME)")
def main(args=None):
    """OpenType font subsetter and optimizer"""
    from os.path import splitext
    from fontTools import configLogger

    if args is None:
        args = sys.argv[1:]

    if "--help" in args:
        print(__doc__)
        return 0

    options = Options()
    try:
        args = options.parse_opts(
            args,
            ignore_unknown=[
                "gids",
                "gids-file",
                "glyphs",
                "glyphs-file",
                "text",
                "text-file",
                "unicodes",
                "unicodes-file",
                "output-file",
            ],
        )
    except options.OptionError as e:
        usage()
        print("ERROR:", e, file=sys.stderr)
        return 2

    if len(args) < 2:
        usage()
        return 1

    configLogger(level=logging.INFO if options.verbose else logging.WARNING)
    if options.timing:
        timer.logger.setLevel(logging.DEBUG)
    else:
        timer.logger.disabled = True

    fontfile = args[0]
    args = args[1:]

    subsetter = Subsetter(options=options)
    outfile = None
    glyphs = []
    gids = []
    unicodes = []
    wildcard_glyphs = False
    wildcard_unicodes = False
    text = ""
    for g in args:
        if g == "*":
            wildcard_glyphs = True
            continue
        if g.startswith("--output-file="):
            outfile = g[14:]
            continue
        if g.startswith("--text="):
            text += g[7:]
            continue
        if g.startswith("--text-file="):
            with open(g[12:], encoding="utf-8-sig") as f:
                text += f.read().replace("\n", "")
            continue
        if g.startswith("--unicodes="):
            if g[11:] == "*":
                wildcard_unicodes = True
            else:
                unicodes.extend(parse_unicodes(g[11:]))
            continue
        if g.startswith("--unicodes-file="):
            with open(g[16:]) as f:
                for line in f.readlines():
                    unicodes.extend(parse_unicodes(line.split("#")[0]))
            continue
        if g.startswith("--gids="):
            gids.extend(parse_gids(g[7:]))
            continue
        if g.startswith("--gids-file="):
            with open(g[12:]) as f:
                for line in f.readlines():
                    gids.extend(parse_gids(line.split("#")[0]))
            continue
        if g.startswith("--glyphs="):
            if g[9:] == "*":
                wildcard_glyphs = True
            else:
                glyphs.extend(parse_glyphs(g[9:]))
            continue
        if g.startswith("--glyphs-file="):
            with open(g[14:]) as f:
                for line in f.readlines():
                    glyphs.extend(parse_glyphs(line.split("#")[0]))
            continue
        glyphs.append(g)

    dontLoadGlyphNames = not options.glyph_names and not glyphs
    lazy = options.lazy
    font = load_font(
        fontfile, options, dontLoadGlyphNames=dontLoadGlyphNames, lazy=lazy
    )

    if outfile is None:
        ext = "." + options.flavor.lower() if options.flavor is not None else None
        outfile = makeOutputFileName(
            fontfile, extension=ext, overWrite=True, suffix=".subset"
        )

    with timer("compile glyph list"):
        if wildcard_glyphs:
            glyphs.extend(font.getGlyphOrder())
        if wildcard_unicodes:
            for t in font["cmap"].tables:
                if t.isUnicode():
                    unicodes.extend(t.cmap.keys())
                    if t.format == 14:
                        unicodes.extend(t.uvsDict.keys())
        assert "" not in glyphs

    log.info("Text: '%s'" % text)
    log.info("Unicodes: %s", unicodes)
    log.info("Glyphs: %s", glyphs)
    log.info("Gids: %s", gids)

    subsetter.populate(glyphs=glyphs, gids=gids, unicodes=unicodes, text=text)
    subsetter.subset(font)

    save_font(font, outfile, options)

    if options.verbose:
        import os

        log.info("Input font:% 7d bytes: %s" % (os.path.getsize(fontfile), fontfile))
        log.info("Subset font:% 7d bytes: %s" % (os.path.getsize(outfile), outfile))

    if options.xml:
        font.saveXML(sys.stdout)

    font.close()


__all__ = [
    "Options",
    "Subsetter",
    "load_font",
    "save_font",
    "parse_gids",
    "parse_glyphs",
    "parse_unicodes",
    "main",
]
<|endoftext|>from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort


class _ClosureGlyphsT2Decompiler(psCharStrings.SimpleT2Decompiler):
    def __init__(self, components, localSubrs, globalSubrs):
        psCharStrings.SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs)
        self.components = components

    def op_endchar(self, index):
        args = self.popall()
        if len(args) >= 4:
            from fontTools.encodings.StandardEncoding import StandardEncoding

            # endchar can do seac accent bulding; The T2 spec says it's deprecated,
            # but recent software that shall remain nameless does output it.
            adx, ady, bchar, achar = args[-4:]
            baseGlyph = StandardEncoding[bchar]
            accentGlyph = StandardEncoding[achar]
            self.components.add(baseGlyph)
            self.components.add(accentGlyph)


@_add_method(ttLib.getTableClass("CFF "))
def closure_glyphs(self, s):
    cff = self.cff
    assert len(cff) == 1
    font = cff[cff.keys()[0]]
    glyphSet = font.CharStrings

    decompose = s.glyphs
    while decompose:
        components = set()
        for g in decompose:
            if g not in glyphSet:
                continue
            gl = glyphSet[g]

            subrs = getattr(gl.private, "Subrs", [])
            decompiler = _ClosureGlyphsT2Decompiler(components, subrs, gl.globalSubrs)
            decompiler.execute(gl)
        components -= s.glyphs
        s.glyphs.update(components)
        decompose = components


def _empty_charstring(font, glyphName, isCFF2, ignoreWidth=False):
    c, fdSelectIndex = font.CharStrings.getItemAndSelector(glyphName)
    if isCFF2 or ignoreWidth:
        # CFF2 charstrings have no widths nor 'endchar' operators
        c.setProgram([] if isCFF2 else ["endchar"])
    else:
        if hasattr(font, "FDArray") and font.FDArray is not None:
            private = font.FDArray[fdSelectIndex].Private
        else:
            private = font.Private
        dfltWdX = private.defaultWidthX
        nmnlWdX = private.nominalWidthX
        pen = NullPen()
        c.draw(pen)  # this will set the charstring's width
        if c.width != dfltWdX:
            c.program = [c.width - nmnlWdX, "endchar"]
        else:
            c.program = ["endchar"]


@_add_method(ttLib.getTableClass("CFF "))
def prune_pre_subset(self, font, options):
    cff = self.cff
    # CFF table must have one font only
    cff.fontNames = cff.fontNames[:1]

    if options.notdef_glyph and not options.notdef_outline:
        isCFF2 = cff.major > 1
        for fontname in cff.keys():
            font = cff[fontname]
            _empty_charstring(font, ".notdef", isCFF2=isCFF2)

    # Clear useless Encoding
    for fontname in cff.keys():
        font = cff[fontname]
        # https://github.com/fonttools/fonttools/issues/620
        font.Encoding = "StandardEncoding"

    return True  # bool(cff.fontNames)


@_add_method(ttLib.getTableClass("CFF "))
def subset_glyphs(self, s):
    cff = self.cff
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings

        glyphs = s.glyphs.union(s.glyphs_emptied)

        # Load all glyphs
        for g in font.charset:
            if g not in glyphs:
                continue
            c, _ = cs.getItemAndSelector(g)

        if cs.charStringsAreIndexed:
            indices = [i for i, g in enumerate(font.charset) if g in glyphs]
            csi = cs.charStringsIndex
            csi.items = [csi.items[i] for i in indices]
            del csi.file, csi.offsets
            if hasattr(font, "FDSelect"):
                sel = font.FDSelect
                sel.format = None
                sel.gidArray = [sel.gidArray[i] for i in indices]
            newCharStrings = {}
            for indicesIdx, charsetIdx in enumerate(indices):
                g = font.charset[charsetIdx]
                if g in cs.charStrings:
                    newCharStrings[g] = indicesIdx
            cs.charStrings = newCharStrings
        else:
            cs.charStrings = {g: v for g, v in cs.charStrings.items() if g in glyphs}
        font.charset = [g for g in font.charset if g in glyphs]
        font.numGlyphs = len(font.charset)

        if s.options.retain_gids:
            isCFF2 = cff.major > 1
            for g in s.glyphs_emptied:
                _empty_charstring(font, g, isCFF2=isCFF2, ignoreWidth=True)

    return True  # any(cff[fontname].numGlyphs for fontname in cff.keys())


@_add_method(ttLib.getTableClass("CFF "))
def prune_post_subset(self, ttfFont, options):
    cff = self.cff
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings

        # Drop unused FontDictionaries
        if hasattr(font, "FDSelect"):
            sel = font.FDSelect
            indices = _uniq_sort(sel.gidArray)
            sel.gidArray = [indices.index(ss) for ss in sel.gidArray]
            arr = font.FDArray
            arr.items = [arr[i] for i in indices]
            del arr.file, arr.offsets

    # Desubroutinize if asked for
    if options.desubroutinize:
        cff.desubroutinize()

    # Drop hints if not needed
    if not options.hinting:
        self.remove_hints()
    elif not options.desubroutinize:
        self.remove_unused_subroutines()
    return True


@deprecateFunction(
    "use 'CFFFontSet.desubroutinize()' instead", category=DeprecationWarning
)
@_add_method(ttLib.getTableClass("CFF "))
def desubroutinize(self):
    self.cff.desubroutinize()


@deprecateFunction(
    "use 'CFFFontSet.remove_hints()' instead", category=DeprecationWarning
)
@_add_method(ttLib.getTableClass("CFF "))
def remove_hints(self):
    self.cff.remove_hints()


@deprecateFunction(
    "use 'CFFFontSet.remove_unused_subroutines' instead", category=DeprecationWarning
)
@_add_method(ttLib.getTableClass("CFF "))
def remove_unused_subroutines(self):
    self.cff.remove_unused_subroutines()
<|endoftext|>from __future__ import annotations

import re
from functools import cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:
    from lxml import etree
except ImportError:
    # lxml is required for subsetting SVG, but we prefer to delay the import error
    # until subset_glyphs() is called (i.e. if font to subset has an 'SVG ' table)
    etree = None

from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument

__all__ = ["subset_glyphs"]


GID_RE = re.compile(r"^glyph(\d+)$")

NAMESPACES = {
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
}
XLINK_HREF = f'{{{NAMESPACES["xlink"]}}}href'


@cache
def xpath(path):
    # compile XPath upfront, caching result to reuse on multiple elements
    return etree.XPath(path, namespaces=NAMESPACES)


def group_elements_by_id(tree: etree.Element) -> Dict[str, etree.Element]:
    # select all svg elements with 'id' attribute no matter where they are
    # including the root element itself:
    # https://github.com/fonttools/fonttools/issues/2548
    return {el.attrib["id"]: el for el in xpath("//svg:*[@id]")(tree)}


def parse_css_declarations(style_attr: str) -> Dict[str, str]:
    # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/style
    # https://developer.mozilla.org/en-US/docs/Web/CSS/Syntax#css_declarations
    result = {}
    for declaration in style_attr.split(";"):
        if declaration.count(":") == 1:
            property_name, value = declaration.split(":")
            property_name = property_name.strip()
            result[property_name] = value.strip()
        elif declaration.strip():
            raise ValueError(f"Invalid CSS declaration syntax: {declaration}")
    return result


def iter_referenced_ids(tree: etree.Element) -> Iterator[str]:
    # Yield all the ids that can be reached via references from this element tree.
    # We currently support xlink:href (as used by <use> and gradient templates),
    # and local url(#...) links found in fill or clip-path attributes
    # TODO(anthrotype): Check we aren't missing other supported kinds of reference
    find_svg_elements_with_references = xpath(
        ".//svg:*[ "
        "starts-with(@xlink:href, '#') "
        "or starts-with(@fill, 'url(#') "
        "or starts-with(@clip-path, 'url(#') "
        "or contains(@style, ':url(#') "
        "]",
    )
    for el in chain([tree], find_svg_elements_with_references(tree)):
        ref_id = href_local_target(el)
        if ref_id is not None:
            yield ref_id

        attrs = el.attrib
        if "style" in attrs:
            attrs = {**dict(attrs), **parse_css_declarations(el.attrib["style"])}
        for attr in ("fill", "clip-path"):
            if attr in attrs:
                value = attrs[attr]
                if value.startswith("url(#") and value.endswith(")"):
                    ref_id = value[5:-1]
                    assert ref_id
                    yield ref_id


def closure_element_ids(
    elements: Dict[str, etree.Element], element_ids: Set[str]
) -> None:
    # Expand the initial subset of element ids to include ids that can be reached
    # via references from the initial set.
    unvisited = element_ids
    while unvisited:
        referenced: Set[str] = set()
        for el_id in unvisited:
            if el_id not in elements:
                # ignore dangling reference; not our job to validate svg
                continue
            referenced.update(iter_referenced_ids(elements[el_id]))
        referenced -= element_ids
        element_ids.update(referenced)
        unvisited = referenced


def subset_elements(el: etree.Element, retained_ids: Set[str]) -> bool:
    # Keep elements if their id is in the subset, or any of their children's id is.
    # Drop elements whose id is not in the subset, and either have no children,
    # or all their children are being dropped.
    if el.attrib.get("id") in retained_ids:
        # if id is in the set, don't recurse; keep whole subtree
        return True
    # recursively subset all the children; we use a list comprehension instead
    # of a parentheses-less generator expression because we don't want any() to
    # short-circuit, as our function has a side effect of dropping empty elements.
    if any([subset_elements(e, retained_ids) for e in el]):
        return True
    assert len(el) == 0
    parent = el.getparent()
    if parent is not None:
        parent.remove(el)
    return False


def remap_glyph_ids(
    svg: etree.Element, glyph_index_map: Dict[int, int]
) -> Dict[str, str]:
    # Given {old_gid: new_gid} map, rename all elements containing id="glyph{gid}"
    # special attributes
    elements = group_elements_by_id(svg)
    id_map = {}
    for el_id, el in elements.items():
        m = GID_RE.match(el_id)
        if not m:
            continue
        old_index = int(m.group(1))
        new_index = glyph_index_map.get(old_index)
        if new_index is not None:
            if old_index == new_index:
                continue
            new_id = f"glyph{new_index}"
        else:
            # If the old index is missing, the element correspond to a glyph that was
            # excluded from the font's subset.
            # We rename it to avoid clashes with the new GIDs or other element ids.
            new_id = f".{el_id}"
            n = count(1)
            while new_id in elements:
                new_id = f"{new_id}.{next(n)}"

        id_map[el_id] = new_id
        el.attrib["id"] = new_id

    return id_map


def href_local_target(el: etree.Element) -> Optional[str]:
    if XLINK_HREF in el.attrib:
        href = el.attrib[XLINK_HREF]
        if href.startswith("#") and len(href) > 1:
            return href[1:]  # drop the leading #
    return None


def update_glyph_href_links(svg: etree.Element, id_map: Dict[str, str]) -> None:
    # update all xlink:href="#glyph..." attributes to point to the new glyph ids
    for el in xpath(".//svg:*[starts-with(@xlink:href, '#glyph')]")(svg):
        old_id = href_local_target(el)
        assert old_id is not None
        if old_id in id_map:
            new_id = id_map[old_id]
            el.attrib[XLINK_HREF] = f"#{new_id}"


def ranges(ints: Iterable[int]) -> Iterator[Tuple[int, int]]:
    # Yield sorted, non-overlapping (min, max) ranges of consecutive integers
    sorted_ints = iter(sorted(set(ints)))
    try:
        start = end = next(sorted_ints)
    except StopIteration:
        return
    for v in sorted_ints:
        if v - 1 == end:
            end = v
        else:
            yield (start, end)
            start = end = v
    yield (start, end)


@_add_method(ttLib.getTableClass("SVG "))
def subset_glyphs(self, s) -> bool:
    if etree is None:
        raise ImportError("No module named 'lxml', required to subset SVG")

    # glyph names (before subsetting)
    glyph_order: List[str] = s.orig_glyph_order
    # map from glyph names to original glyph indices
    rev_orig_glyph_map: Dict[str, int] = s.reverseOrigGlyphMap
    # map from original to new glyph indices (after subsetting)
    glyph_index_map: Dict[int, int] = s.glyph_index_map

    new_docs: List[SVGDocument] = []
    for doc in self.docList:
        glyphs = {
            glyph_order[i] for i in range(doc.startGlyphID, doc.endGlyphID + 1)
        }.intersection(s.glyphs)
        if not glyphs:
            # no intersection: we can drop the whole record
            continue

        svg = etree.fromstring(
            # encode because fromstring dislikes xml encoding decl if input is str.
            # SVG xml encoding must be utf-8 as per OT spec.
            doc.data.encode("utf-8"),
            parser=etree.XMLParser(
                # Disable libxml2 security restrictions to support very deep trees.
                # Without this we would get an error like this:
                # `lxml.etree.XMLSyntaxError: internal error: Huge input lookup`
                # when parsing big fonts e.g. noto-emoji-picosvg.ttf.
                huge_tree=True,
                # ignore blank text as it's not meaningful in OT-SVG; it also prevents
                # dangling tail text after removing an element when pretty_print=True
                remove_blank_text=True,
                # don't replace entities; we don't expect any in OT-SVG and they may
                # be abused for XXE attacks
                resolve_entities=False,
            ),
        )

        elements = group_elements_by_id(svg)
        gids = {rev_orig_glyph_map[g] for g in glyphs}
        element_ids = {f"glyph{i}" for i in gids}
        closure_element_ids(elements, element_ids)

        if not subset_elements(svg, element_ids):
            continue

        if not s.options.retain_gids:
            id_map = remap_glyph_ids(svg, glyph_index_map)
            update_glyph_href_links(svg, id_map)

        new_doc = etree.tostring(svg, pretty_print=s.options.pretty_svg).decode("utf-8")

        new_gids = (glyph_index_map[i] for i in gids)
        for start, end in ranges(new_gids):
            new_docs.append(SVGDocument(new_doc, start, end, doc.compressed))

    self.docList = new_docs

    return bool(self.docList)
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


class Error(Exception):
    """Base Cu2Qu exception class for all other errors."""


class ApproxNotFoundError(Error):
    def __init__(self, curve):
        message = "no approximation found: %s" % curve
        super().__init__(message)
        self.curve = curve


class UnequalZipLengthsError(Error):
    pass


class IncompatibleGlyphsError(Error):
    def __init__(self, glyphs):
        assert len(glyphs) > 1
        self.glyphs = glyphs
        names = set(repr(g.name) for g in glyphs)
        if len(names) > 1:
            self.combined_name = "{%s}" % ", ".join(sorted(names))
        else:
            self.combined_name = names.pop()

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.combined_name)


class IncompatibleSegmentNumberError(IncompatibleGlyphsError):
    def __str__(self):
        return "Glyphs named %s have different number of segments" % (
            self.combined_name
        )


class IncompatibleSegmentTypesError(IncompatibleGlyphsError):
    def __init__(self, glyphs, segments):
        IncompatibleGlyphsError.__init__(self, glyphs)
        self.segments = segments

    def __str__(self):
        lines = []
        ndigits = len(str(max(self.segments)))
        for i, tags in sorted(self.segments.items()):
            lines.append(
                "%s: (%s)" % (str(i).rjust(ndigits), ", ".join(repr(t) for t in tags))
            )
        return "Glyphs named %s have incompatible segment types:\n  %s" % (
            self.combined_name,
            "\n  ".join(lines),
        )


class IncompatibleFontsError(Error):
    def __init__(self, glyph_errors):
        self.glyph_errors = glyph_errors

    def __str__(self):
        return "fonts contains incompatible glyphs: %s" % (
            ", ".join(repr(g) for g in sorted(self.glyph_errors.keys()))
        )
<|endoftext|># cython: language_level=3
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

# Copyright 2015 Google Inc. All Rights Reserved.
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

import math

from .errors import Error as Cu2QuError, ApproxNotFoundError

__all__ = ["curve_to_quadratic", "curves_to_quadratic"]

MAX_N = 100

NAN = float("NaN")


@cython.cfunc
@cython.inline
@cython.returns(cython.double)
@cython.locals(v1=cython.complex, v2=cython.complex, result=cython.double)
def dot(v1, v2):
    """Return the dot product of two vectors.

    Args:
        v1 (complex): First vector.
        v2 (complex): Second vector.

    Returns:
        double: Dot product.
    """
    result = (v1 * v2.conjugate()).real
    # When vectors are perpendicular (i.e. dot product is 0), the above expression may
    # yield slightly different results when running in pure Python vs C/Cython,
    # both of which are correct within IEEE-754 floating-point precision.
    # It's probably due to the different order of operations and roundings in each
    # implementation. Because we are using the result in a denominator and catching
    # ZeroDivisionError (see `calc_intersect`), it's best to normalize the result here.
    if abs(result) < 1e-15:
        result = 0.0
    return result


@cython.cfunc
@cython.locals(z=cython.complex, den=cython.double)
@cython.locals(zr=cython.double, zi=cython.double)
def _complex_div_by_real(z, den):
    """Divide complex by real using Python's method (two separate divisions).

    This ensures bit-exact compatibility with Python's complex division,
    avoiding C's multiply-by-reciprocal optimization that can cause 1 ULP differences
    on some platforms/compilers (e.g. clang on macOS arm64).

    https://github.com/fonttools/fonttools/issues/3928
    """
    zr = z.real
    zi = z.imag
    return complex(zr / den, zi / den)


@cython.cfunc
@cython.inline
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
@cython.locals(
    _1=cython.complex, _2=cython.complex, _3=cython.complex, _4=cython.complex
)
def calc_cubic_points(a, b, c, d):
    _1 = d
    _2 = _complex_div_by_real(c, 3.0) + d
    _3 = _complex_div_by_real(b + c, 3.0) + _2
    _4 = a + d + c + b
    return _1, _2, _3, _4


@cython.cfunc
@cython.inline
@cython.locals(
    p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex
)
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
def calc_cubic_parameters(p0, p1, p2, p3):
    c = (p1 - p0) * 3.0
    b = (p2 - p1) * 3.0 - c
    d = p0
    a = p3 - d - c - b
    return a, b, c, d


@cython.cfunc
@cython.inline
@cython.locals(
    p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex
)
def split_cubic_into_n_iter(p0, p1, p2, p3, n):
    """Split a cubic Bezier into n equal parts.

    Splits the curve into `n` equal parts by curve time.
    (t=0..1/n, t=1/n..2/n, ...)

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        An iterator yielding the control points (four complex values) of the
        subcurves.
    """
    # Hand-coded special-cases
    if n == 2:
        return iter(split_cubic_into_two(p0, p1, p2, p3))
    if n == 3:
        return iter(split_cubic_into_three(p0, p1, p2, p3))
    if n == 4:
        a, b = split_cubic_into_two(p0, p1, p2, p3)
        return iter(
            split_cubic_into_two(a[0], a[1], a[2], a[3])
            + split_cubic_into_two(b[0], b[1], b[2], b[3])
        )
    if n == 6:
        a, b = split_cubic_into_two(p0, p1, p2, p3)
        return iter(
            split_cubic_into_three(a[0], a[1], a[2], a[3])
            + split_cubic_into_three(b[0], b[1], b[2], b[3])
        )

    return _split_cubic_into_n_gen(p0, p1, p2, p3, n)


@cython.locals(
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
    n=cython.int,
)
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
@cython.locals(
    dt=cython.double, delta_2=cython.double, delta_3=cython.double, i=cython.int
)
@cython.locals(
    a1=cython.complex, b1=cython.complex, c1=cython.complex, d1=cython.complex
)
def _split_cubic_into_n_gen(p0, p1, p2, p3, n):
    a, b, c, d = calc_cubic_parameters(p0, p1, p2, p3)
    dt = 1 / n
    delta_2 = dt * dt
    delta_3 = dt * delta_2
    for i in range(n):
        t1 = i * dt
        t1_2 = t1 * t1
        # calc new a, b, c and d
        a1 = a * delta_3
        b1 = (3 * a * t1 + b) * delta_2
        c1 = (2 * b * t1 + c + 3 * a * t1_2) * dt
        d1 = a * t1 * t1_2 + b * t1_2 + c * t1 + d
        yield calc_cubic_points(a1, b1, c1, d1)


@cython.cfunc
@cython.inline
@cython.locals(
    p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex
)
@cython.locals(mid=cython.complex, deriv3=cython.complex)
def split_cubic_into_two(p0, p1, p2, p3):
    """Split a cubic Bezier into two equal parts.

    Splits the curve into two equal parts at t = 0.5

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        tuple: Two cubic Beziers (each expressed as a tuple of four complex
        values).
    """
    mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
    deriv3 = (p3 + p2 - p1 - p0) * 0.125
    return (
        (p0, (p0 + p1) * 0.5, mid - deriv3, mid),
        (mid, mid + deriv3, (p2 + p3) * 0.5, p3),
    )


@cython.cfunc
@cython.inline
@cython.locals(
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
)
@cython.locals(
    mid1=cython.complex,
    deriv1=cython.complex,
    mid2=cython.complex,
    deriv2=cython.complex,
)
def split_cubic_into_three(p0, p1, p2, p3):
    """Split a cubic Bezier into three equal parts.

    Splits the curve into three equal parts at t = 1/3 and t = 2/3

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        tuple: Three cubic Beziers (each expressed as a tuple of four complex
        values).
    """
    mid1 = (8 * p0 + 12 * p1 + 6 * p2 + p3) * (1 / 27)
    deriv1 = (p3 + 3 * p2 - 4 * p0) * (1 / 27)
    mid2 = (p0 + 6 * p1 + 12 * p2 + 8 * p3) * (1 / 27)
    deriv2 = (4 * p3 - 3 * p1 - p0) * (1 / 27)
    return (
        (p0, _complex_div_by_real(2 * p0 + p1, 3.0), mid1 - deriv1, mid1),
        (mid1, mid1 + deriv1, mid2 - deriv2, mid2),
        (mid2, mid2 + deriv2, _complex_div_by_real(p2 + 2 * p3, 3.0), p3),
    )


@cython.cfunc
@cython.inline
@cython.returns(cython.complex)
@cython.locals(
    t=cython.double,
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
)
@cython.locals(_p1=cython.complex, _p2=cython.complex)
def cubic_approx_control(t, p0, p1, p2, p3):
    """Approximate a cubic Bezier using a quadratic one.

    Args:
        t (double): Position of control point.
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        complex: Location of candidate control point on quadratic curve.
    """
    _p1 = p0 + (p1 - p0) * 1.5
    _p2 = p3 + (p2 - p3) * 1.5
    return _p1 + (_p2 - _p1) * t


@cython.cfunc
@cython.inline
@cython.returns(cython.complex)
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
@cython.locals(ab=cython.complex, cd=cython.complex, p=cython.complex, h=cython.double)
def calc_intersect(a, b, c, d):
    """Calculate the intersection of two lines.

    Args:
        a (complex): Start point of first line.
        b (complex): End point of first line.
        c (complex): Start point of second line.
        d (complex): End point of second line.

    Returns:
        complex: Location of intersection if one present, ``complex(NaN,NaN)``
        if no intersection was found.
    """
    ab = b - a
    cd = d - c
    p = ab * 1j
    try:
        h = dot(p, a - c) / dot(p, cd)
    except ZeroDivisionError:
        # if 3 or 4 points are equal, we do have an intersection despite the zero-div:
        # return one of the off-curves so that the algorithm can attempt a one-curve
        # solution if it's within tolerance:
        # https://github.com/linebender/kurbo/pull/484
        if b == c and (a == b or c == d):
            return b
        return complex(NAN, NAN)
    return c + cd * h


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


@cython.cfunc
@cython.inline
@cython.locals(tolerance=cython.double)
@cython.locals(
    q1=cython.complex,
    c0=cython.complex,
    c1=cython.complex,
    c2=cython.complex,
    c3=cython.complex,
)
def cubic_approx_quadratic(cubic, tolerance):
    """Approximate a cubic Bezier with a single quadratic within a given tolerance.

    Args:
        cubic (sequence): Four complex numbers representing control points of
            the cubic Bezier curve.
        tolerance (double): Permitted deviation from the original curve.

    Returns:
        Three complex numbers representing control points of the quadratic
        curve if it fits within the given tolerance, or ``None`` if no suitable
        curve could be calculated.
    """

    q1 = calc_intersect(cubic[0], cubic[1], cubic[2], cubic[3])
    if math.isnan(q1.imag):
        return None
    c0 = cubic[0]
    c3 = cubic[3]
    c1 = c0 + (q1 - c0) * (2 / 3)
    c2 = c3 + (q1 - c3) * (2 / 3)
    if not cubic_farthest_fit_inside(0, c1 - cubic[1], c2 - cubic[2], 0, tolerance):
        return None
    return c0, q1, c3


@cython.cfunc
@cython.locals(n=cython.int, tolerance=cython.double)
@cython.locals(i=cython.int)
@cython.locals(all_quadratic=cython.int)
@cython.locals(
    c0=cython.complex, c1=cython.complex, c2=cython.complex, c3=cython.complex
)
@cython.locals(
    q0=cython.complex,
    q1=cython.complex,
    next_q1=cython.complex,
    q2=cython.complex,
    d1=cython.complex,
)
def cubic_approx_spline(cubic, n, tolerance, all_quadratic):
    """Approximate a cubic Bezier curve with a spline of n quadratics.

    Args:
        cubic (sequence): Four complex numbers representing control points of
            the cubic Bezier curve.
        n (int): Number of quadratic Bezier curves in the spline.
        tolerance (double): Permitted deviation from the original curve.

    Returns:
        A list of ``n+2`` complex numbers, representing control points of the
        quadratic spline if it fits within the given tolerance, or ``None`` if
        no suitable spline could be calculated.
    """

    if n == 1:
        return cubic_approx_quadratic(cubic, tolerance)
    if n == 2 and all_quadratic == False:
        return cubic

    cubics = split_cubic_into_n_iter(cubic[0], cubic[1], cubic[2], cubic[3], n)

    # calculate the spline of quadratics and check errors at the same time.
    next_cubic = next(cubics)
    next_q1 = cubic_approx_control(
        0, next_cubic[0], next_cubic[1], next_cubic[2], next_cubic[3]
    )
    q2 = cubic[0]
    d1 = 0j
    spline = [cubic[0], next_q1]
    for i in range(1, n + 1):
        # Current cubic to convert
        c0, c1, c2, c3 = next_cubic

        # Current quadratic approximation of current cubic
        q0 = q2
        q1 = next_q1
        if i < n:
            next_cubic = next(cubics)
            next_q1 = cubic_approx_control(
                i / (n - 1), next_cubic[0], next_cubic[1], next_cubic[2], next_cubic[3]
            )
            spline.append(next_q1)
            q2 = (q1 + next_q1) * 0.5
        else:
            q2 = c3

        # End-point deltas
        d0 = d1
        d1 = q2 - c3

        if abs(d1) > tolerance or not cubic_farthest_fit_inside(
            d0,
            q0 + (q1 - q0) * (2 / 3) - c1,
            q2 + (q1 - q2) * (2 / 3) - c2,
            d1,
            tolerance,
        ):
            return None
    spline.append(cubic[3])

    return spline


@cython.locals(max_err=cython.double)
@cython.locals(n=cython.int)
@cython.locals(all_quadratic=cython.int)
def curve_to_quadratic(curve, max_err, all_quadratic=True):
    """Approximate a cubic Bezier curve with a spline of n quadratics.

    Args:
        cubic (sequence): Four 2D tuples representing control points of
            the cubic Bezier curve.
        max_err (double): Permitted deviation from the original curve.
        all_quadratic (bool): If True (default) returned value is a
            quadratic spline. If False, it's either a single quadratic
            curve or a single cubic curve.

    Returns:
        If all_quadratic is True: A list of 2D tuples representing
        control points of the quadratic spline.

        If all_quadratic is False: Either a quadratic curve (if length
        of output is 3), or a cubic curve (if length of output is 4).

    Raises:
        fontTools.cu2qu.errors.ApproxNotFoundError: if no suitable
        approximation can be found with the given parameters.
    """

    curve = [complex(*p) for p in curve]

    for n in range(1, MAX_N + 1):
        spline = cubic_approx_spline(curve, n, max_err, all_quadratic)
        if spline is not None:
            # done. go home
            return [(s.real, s.imag) for s in spline]

    raise ApproxNotFoundError(curve)


@cython.locals(l=cython.int, last_i=cython.int, i=cython.int)
@cython.locals(all_quadratic=cython.int)
def curves_to_quadratic(curves, max_errors, all_quadratic=True):
    """Return quadratic Bezier splines approximating the input cubic Beziers.

    Args:
        curves: A sequence of *n* curves, each curve being a sequence of four
            2D tuples.
        max_errors: A sequence of *n* floats representing the maximum permissible
            deviation from each of the cubic Bezier curves.
        all_quadratic (bool): If True (default) returned values are a
            quadratic spline. If False, they are either a single quadratic
            curve or a single cubic curve.

    Example::

        >>> curves_to_quadratic( [
        ...   [ (50,50), (100,100), (150,100), (200,50) ],
        ...   [ (75,50), (120,100), (150,75),  (200,60) ]
        ... ], [1,1] )
        [[(50.0, 50.0), (75.0, 75.0), (125.0, 91.66666666666666), (175.0, 75.0), (200.0, 50.0)], [(75.0, 50.0), (97.5, 75.0), (135.41666666666666, 82.08333333333333), (175.0, 67.5), (200.0, 60.0)]]

    The returned splines have "implied oncurve points" suitable for use in
    TrueType ``glif`` outlines - i.e. in the first spline returned above,
    the first quadratic segment runs from (50,50) to
    ( (75 + 125)/2 , (120 + 91.666..)/2 ) = (100, 83.333...).

    Returns:
        If all_quadratic is True, a list of splines, each spline being a list
        of 2D tuples. If ``curves`` is empty, returns an empty list.

        If all_quadratic is False, a list of curves, each curve being a quadratic
        (length 3), or cubic (length 4).

    Raises:
        ValueError: if ``max_errors`` does not match the number of curves.
        fontTools.cu2qu.errors.ApproxNotFoundError: if no suitable approximation
        can be found for all curves with the given parameters.
    """

    curves = [[complex(*p) for p in curve] for curve in curves]
    if len(max_errors) != len(curves):
        raise ValueError("max_errors must match the number of curves")
    if not curves:
        return []

    l = len(curves)
    splines = [None] * l
    last_i = i = 0
    n = 1
    while True:
        spline = cubic_approx_spline(curves[i], n, max_errors[i], all_quadratic)
        if spline is None:
            if n == MAX_N:
                break
            n += 1
            last_i = i
            continue
        splines[i] = spline
        i = (i + 1) % l
        if i == last_i:
            # done. go home
            return [[(s.real, s.imag) for s in spline] for spline in splines]

    raise ApproxNotFoundError(curves)
<|endoftext|># Copyright 2015 Google Inc. All Rights Reserved.
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


"""Converts cubic bezier curves to quadratic splines.

Conversion is performed such that the quadratic splines keep the same end-curve
tangents as the original cubics. The approach is iterative, increasing the
number of segments for a spline until the error gets below a bound.

Respective curves from multiple fonts will be converted at once to ensure that
the resulting splines are interpolation-compatible.
"""

import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen

from . import curves_to_quadratic
from .errors import (
    UnequalZipLengthsError,
    IncompatibleSegmentNumberError,
    IncompatibleSegmentTypesError,
    IncompatibleGlyphsError,
    IncompatibleFontsError,
)

__all__ = ["fonts_to_quadratic", "font_to_quadratic"]

# The default approximation error below is a relative value (1/1000 of the EM square).
# Later on, we convert it to absolute font units by multiplying it by a font's UPEM
# (see fonts_to_quadratic).
DEFAULT_MAX_ERR = 0.001
CURVE_TYPE_LIB_KEY = "com.github.googlei18n.cu2qu.curve_type"

logger = logging.getLogger(__name__)


_zip = zip


def zip(*args):
    """Ensure each argument to zip has the same length. Also make sure a list is
    returned for python 2/3 compatibility.
    """

    if len(set(len(a) for a in args)) != 1:
        raise UnequalZipLengthsError(*args)
    return list(_zip(*args))


def _validate_positive_tolerance(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_positive_tolerances(values, name):
    for value in values:
        _validate_positive_tolerance(value, name)


def _validate_length(values, expected, name):
    if len(values) != expected:
        raise ValueError(f"{name} must match the number of inputs")


class GetSegmentsPen(AbstractPen):
    """Pen to collect segments into lists of points for conversion.

    Curves always include their initial on-curve point, so some points are
    duplicated between segments.
    """

    def __init__(self):
        self._last_pt = None
        self.segments = []

    def _add_segment(self, tag, *args):
        if tag in ["move", "line", "qcurve", "curve"]:
            self._last_pt = args[-1]
        self.segments.append((tag, args))

    def moveTo(self, pt):
        self._add_segment("move", pt)

    def lineTo(self, pt):
        self._add_segment("line", pt)

    def qCurveTo(self, *points):
        self._add_segment("qcurve", self._last_pt, *points)

    def curveTo(self, *points):
        self._add_segment("curve", self._last_pt, *points)

    def closePath(self):
        self._add_segment("close")

    def endPath(self):
        self._add_segment("end")

    def addComponent(self, glyphName, transformation):
        pass


def _get_segments(glyph):
    """Get a glyph's segments as extracted by GetSegmentsPen."""

    pen = GetSegmentsPen()
    # glyph.draw(pen)
    # We can't simply draw the glyph with the pen, but we must initialize the
    # PointToSegmentPen explicitly with outputImpliedClosingLine=True.
    # By default PointToSegmentPen does not outputImpliedClosingLine -- unless
    # last and first point on closed contour are duplicated. Because we are
    # converting multiple glyphs at the same time, we want to make sure
    # this function returns the same number of segments, whether or not
    # the last and first point overlap.
    # https://github.com/googlefonts/fontmake/issues/572
    # https://github.com/fonttools/fonttools/pull/1720
    pointPen = PointToSegmentPen(pen, outputImpliedClosingLine=True)
    glyph.drawPoints(pointPen)
    return pen.segments


def _set_segments(glyph, segments, reverse_direction):
    """Draw segments as extracted by GetSegmentsPen back to a glyph."""

    glyph.clearContours()
    pen = glyph.getPen()
    if reverse_direction:
        pen = ReverseContourPen(pen)
    for tag, args in segments:
        if tag == "move":
            pen.moveTo(*args)
        elif tag == "line":
            pen.lineTo(*args)
        elif tag == "curve":
            pen.curveTo(*args[1:])
        elif tag == "qcurve":
            pen.qCurveTo(*args[1:])
        elif tag == "close":
            pen.closePath()
        elif tag == "end":
            pen.endPath()
        else:
            raise AssertionError('Unhandled segment type "%s"' % tag)


def _segments_to_quadratic(segments, max_err, stats, all_quadratic=True):
    """Return quadratic approximations of cubic segments."""

    assert all(s[0] == "curve" for s in segments), "Non-cubic given to convert"

    new_points = curves_to_quadratic([s[1] for s in segments], max_err, all_quadratic)
    n = len(new_points[0])
    assert all(len(s) == n for s in new_points[1:]), "Converted incompatibly"

    spline_length = str(n - 2)
    stats[spline_length] = stats.get(spline_length, 0) + 1

    if all_quadratic or n == 3:
        return [("qcurve", p) for p in new_points]
    else:
        return [("curve", p) for p in new_points]


def _glyphs_to_quadratic(glyphs, max_err, reverse_direction, stats, all_quadratic=True):
    """Do the actual conversion of a set of compatible glyphs, after arguments
    have been set up.

    Empty glyphs (without contours) are ignored and passed through unchanged.

    Return True if the glyphs were modified, else return False.
    """

    # Skip empty glyphs (with zero contours)
    non_empty_indices = [i for i, g in enumerate(glyphs) if len(g) > 0]
    if not non_empty_indices:
        return False

    glyphs = [glyphs[i] for i in non_empty_indices]
    max_err = [max_err[i] for i in non_empty_indices]

    try:
        segments_by_location = zip(*[_get_segments(g) for g in glyphs])
    except UnequalZipLengthsError:
        raise IncompatibleSegmentNumberError(glyphs)
    if not any(segments_by_location):
        return False

    # always modify input glyphs if reverse_direction is True
    glyphs_modified = reverse_direction

    new_segments_by_location = []
    incompatible = {}
    for i, segments in enumerate(segments_by_location):
        tag = segments[0][0]
        if not all(s[0] == tag for s in segments[1:]):
            incompatible[i] = [s[0] for s in segments]
        elif tag == "curve":
            new_segments = _segments_to_quadratic(
                segments, max_err, stats, all_quadratic
            )
            if all_quadratic or new_segments != segments:
                glyphs_modified = True
            segments = new_segments
        new_segments_by_location.append(segments)

    if glyphs_modified:
        new_segments_by_glyph = zip(*new_segments_by_location)
        for glyph, new_segments in zip(glyphs, new_segments_by_glyph):
            _set_segments(glyph, new_segments, reverse_direction)

    if incompatible:
        raise IncompatibleSegmentTypesError(glyphs, segments=incompatible)
    return glyphs_modified


def glyphs_to_quadratic(
    glyphs, max_err=None, reverse_direction=False, stats=None, all_quadratic=True
):
    """Convert the curves of a set of compatible of glyphs to quadratic.

    All curves will be converted to quadratic at once, ensuring interpolation
    compatibility. If this is not required, calling glyphs_to_quadratic with one
    glyph at a time may yield slightly more optimized results.

    Empty glyphs (without contours) are ignored and passed through unchanged.

    Return True if glyphs were modified, else return False.

    Raises IncompatibleGlyphsError if glyphs have non-interpolatable outlines.
    """
    if stats is None:
        stats = {}

    if max_err is None:
        # assume 1000 is the default UPEM
        max_err = DEFAULT_MAX_ERR * 1000

    if isinstance(max_err, (list, tuple)):
        max_errors = max_err
    else:
        max_errors = [max_err] * len(glyphs)
    _validate_length(max_errors, len(glyphs), "max_err")
    _validate_positive_tolerances(max_errors, "max_err")

    return _glyphs_to_quadratic(
        glyphs, max_errors, reverse_direction, stats, all_quadratic
    )


def fonts_to_quadratic(
    fonts,
    max_err_em=None,
    max_err=None,
    reverse_direction=False,
    stats=None,
    dump_stats=False,
    remember_curve_type=True,
    all_quadratic=True,
):
    """Convert the curves of a collection of fonts to quadratic.

    All curves will be converted to quadratic at once, ensuring interpolation
    compatibility. If this is not required, calling fonts_to_quadratic with one
    font at a time may yield slightly more optimized results.

    Empty glyphs (without contours) are ignored and passed through unchanged.

    Return the set of modified glyph names if any, else return an empty set.

    By default, cu2qu stores the curve type in the fonts' lib, under a private
    key "com.github.googlei18n.cu2qu.curve_type", and will not try to convert
    them again if the curve type is already set to "quadratic".
    Setting 'remember_curve_type' to False disables this optimization.

    Raises IncompatibleFontsError if same-named glyphs from different fonts
    have non-interpolatable outlines.
    """

    if remember_curve_type:
        curve_types = {f.lib.get(CURVE_TYPE_LIB_KEY, "cubic") for f in fonts}
        if len(curve_types) == 1:
            curve_type = next(iter(curve_types))
            if curve_type in ("quadratic", "mixed"):
                logger.info("Curves already converted to quadratic")
                return False
            elif curve_type == "cubic":
                pass  # keep converting
            else:
                raise NotImplementedError(curve_type)
        elif len(curve_types) > 1:
            # going to crash later if they do differ
            logger.warning("fonts may contain different curve types")

    if stats is None:
        stats = {}

    if max_err_em is not None and max_err is not None:
        raise TypeError("Only one of max_err and max_err_em can be specified.")
    if max_err_em is None and max_err is None:
        max_err_em = DEFAULT_MAX_ERR

    if isinstance(max_err, (list, tuple)):
        _validate_length(max_err, len(fonts), "max_err")
        max_errors = max_err
        _validate_positive_tolerances(max_errors, "max_err")
    elif max_err is not None:
        _validate_positive_tolerance(max_err, "max_err")
        max_errors = [max_err] * len(fonts)

    if isinstance(max_err_em, (list, tuple)):
        _validate_length(max_err_em, len(fonts), "max_err_em")
        _validate_positive_tolerances(max_err_em, "max_err_em")
        max_errors = [f.info.unitsPerEm * e for f, e in zip(fonts, max_err_em)]
    elif max_err_em is not None:
        _validate_positive_tolerance(max_err_em, "max_err_em")
        max_errors = [f.info.unitsPerEm * max_err_em for f in fonts]

    modified = set()
    glyph_errors = {}
    for name in set().union(*(f.keys() for f in fonts)):
        glyphs = []
        cur_max_errors = []
        for font, error in zip(fonts, max_errors):
            if name in font:
                glyphs.append(font[name])
                cur_max_errors.append(error)
        try:
            if _glyphs_to_quadratic(
                glyphs, cur_max_errors, reverse_direction, stats, all_quadratic
            ):
                modified.add(name)
        except IncompatibleGlyphsError as exc:
            logger.error(exc)
            glyph_errors[name] = exc

    if glyph_errors:
        raise IncompatibleFontsError(glyph_errors)

    if modified and dump_stats:
        spline_lengths = sorted(stats.keys())
        logger.info(
            "New spline lengths: %s"
            % (", ".join("%s: %d" % (l, stats[l]) for l in spline_lengths))
        )

    if remember_curve_type:
        for font in fonts:
            curve_type = font.lib.get(CURVE_TYPE_LIB_KEY, "cubic")
            new_curve_type = "quadratic" if all_quadratic else "mixed"
            if curve_type != new_curve_type:
                font.lib[CURVE_TYPE_LIB_KEY] = new_curve_type
    return modified


def glyph_to_quadratic(glyph, **kwargs):
    """Convenience wrapper around glyphs_to_quadratic, for just one glyph.
    Return True if the glyph was modified, else return False.
    """

    return glyphs_to_quadratic([glyph], **kwargs)


def font_to_quadratic(font, **kwargs):
    """Convenience wrapper around fonts_to_quadratic, for just one font.
    Return the set of modified glyph names if any, else return empty set.
    """

    return fonts_to_quadratic([font], **kwargs)
<|endoftext|>"""Benchmark the cu2qu algorithm performance."""

from .cu2qu import *
import random
import timeit

MAX_ERR = 0.05


def generate_curve():
    return [
        tuple(float(random.randint(0, 2048)) for coord in range(2))
        for point in range(4)
    ]


def setup_curve_to_quadratic():
    return generate_curve(), MAX_ERR


def setup_curves_to_quadratic():
    num_curves = 3
    return ([generate_curve() for curve in range(num_curves)], [MAX_ERR] * num_curves)


def run_benchmark(module, function, setup_suffix="", repeat=5, number=1000):
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
    run_benchmark("cu2qu", "curve_to_quadratic")
    run_benchmark("cu2qu", "curves_to_quadratic")


if __name__ == "__main__":
    random.seed(1)
    main()
<|endoftext|>import os
import argparse
import logging
import shutil
import multiprocessing as mp
from contextlib import closing
from functools import partial

import fontTools
from .ufo import font_to_quadratic, fonts_to_quadratic

ufo_module = None
try:
    import ufoLib2 as ufo_module
except ImportError:
    try:
        import defcon as ufo_module
    except ImportError as e:
        pass


logger = logging.getLogger("fontTools.cu2qu")


def _cpu_count():
    try:
        return mp.cpu_count()
    except NotImplementedError:  # pragma: no cover
        return 1


def open_ufo(path):
    if hasattr(ufo_module.Font, "open"):  # ufoLib2
        return ufo_module.Font.open(path)
    return ufo_module.Font(path)  # defcon


def _font_to_quadratic(input_path, output_path=None, **kwargs):
    ufo = open_ufo(input_path)
    logger.info("Converting curves for %s", input_path)
    if font_to_quadratic(ufo, **kwargs):
        logger.info("Saving %s", output_path)
        if output_path:
            ufo.save(output_path)
        else:
            ufo.save()  # save in-place
    elif output_path:
        _copytree(input_path, output_path)


def _samepath(path1, path2):
    # TODO on python3+, there's os.path.samefile
    path1 = os.path.normcase(os.path.abspath(os.path.realpath(path1)))
    path2 = os.path.normcase(os.path.abspath(os.path.realpath(path2)))
    return path1 == path2


def _copytree(input_path, output_path):
    if _samepath(input_path, output_path):
        logger.debug("input and output paths are the same file; skipped copy")
        return
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(input_path, output_path)


def _main(args=None):
    """Convert a UFO font from cubic to quadratic curves"""
    parser = argparse.ArgumentParser(prog="cu2qu")
    parser.add_argument("--version", action="version", version=fontTools.__version__)
    parser.add_argument(
        "infiles",
        nargs="+",
        metavar="INPUT",
        help="one or more input UFO source file(s).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-e",
        "--conversion-error",
        type=float,
        metavar="ERROR",
        default=None,
        help="maximum approximation error measured in EM (default: 0.001)",
    )
    parser.add_argument(
        "-m",
        "--mixed",
        default=False,
        action="store_true",
        help="whether to used mixed quadratic and cubic curves",
    )
    parser.add_argument(
        "--keep-direction",
        dest="reverse_direction",
        action="store_false",
        help="do not reverse the contour direction",
    )

    mode_parser = parser.add_mutually_exclusive_group()
    mode_parser.add_argument(
        "-i",
        "--interpolatable",
        action="store_true",
        help="whether curve conversion should keep interpolation compatibility",
    )
    mode_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        nargs="?",
        default=1,
        const=_cpu_count(),
        metavar="N",
        help="convert using N multiple processes (default: %(default)s)",
    )

    output_parser = parser.add_mutually_exclusive_group()
    output_parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        metavar="OUTPUT",
        help=(
            "output filename for the converted UFO. By default fonts are "
            "modified in place. This only works with a single input."
        ),
    )
    output_parser.add_argument(
        "-d",
        "--output-dir",
        default=None,
        metavar="DIRECTORY",
        help="output directory where to save converted UFOs",
    )

    options = parser.parse_args(args)

    if ufo_module is None:
        parser.error("Either ufoLib2 or defcon are required to run this script.")

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
        # save in-place
        output_paths = [None] * len(options.infiles)

    kwargs = dict(
        dump_stats=options.verbose > 0,
        max_err_em=options.conversion_error,
        reverse_direction=options.reverse_direction,
        all_quadratic=False if options.mixed else True,
    )

    if options.interpolatable:
        logger.info("Converting curves compatibly")
        ufos = [open_ufo(infile) for infile in options.infiles]
        if fonts_to_quadratic(ufos, **kwargs):
            for ufo, output_path in zip(ufos, output_paths):
                logger.info("Saving %s", output_path)
                if output_path:
                    ufo.save(output_path)
                else:
                    ufo.save()
        else:
            for input_path, output_path in zip(options.infiles, output_paths):
                if output_path:
                    _copytree(input_path, output_path)
    else:
        jobs = min(len(options.infiles), options.jobs) if options.jobs > 1 else 1
        if jobs > 1:
            func = partial(_font_to_quadratic, **kwargs)
            logger.info("Running %d parallel processes", jobs)
            with closing(mp.Pool(jobs)) as pool:
                pool.starmap(func, zip(options.infiles, output_paths))
        else:
            for input_path, output_path in zip(options.infiles, output_paths):
                _font_to_quadratic(input_path, output_path, **kwargs)
<|endoftext|>"""Simplify TrueType glyphs by merging overlapping contours/components.

Requires https://github.com/fonttools/skia-pathops
"""

import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping

from fontTools.cffLib import CFFFontSet
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.misc.psCharStrings import T2CharString
from fontTools.misc.roundTools import otRound, noRound
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.t2CharStringPen import T2CharStringPen

import pathops

__all__ = ["removeOverlaps"]


class RemoveOverlapsError(Exception):
    pass


log = logging.getLogger("fontTools.ttLib.removeOverlaps")

_TTGlyphMapping = Mapping[str, ttFont._TTGlyph]


def skPathFromGlyph(glyphName: str, glyphSet: _TTGlyphMapping) -> pathops.Path:
    path = pathops.Path()
    pathPen = path.getPen(glyphSet=glyphSet)
    glyphSet[glyphName].draw(pathPen)
    return path


def skPathFromGlyphComponent(
    component: _g_l_y_f.GlyphComponent, glyphSet: _TTGlyphMapping
):
    baseGlyphName, transformation = component.getComponentInfo()
    path = skPathFromGlyph(baseGlyphName, glyphSet)
    return path.transform(*transformation)


def componentsOverlap(glyph: _g_l_y_f.Glyph, glyphSet: _TTGlyphMapping) -> bool:
    if not glyph.isComposite():
        raise ValueError("This method only works with TrueType composite glyphs")
    if len(glyph.components) < 2:
        return False  # single component, no overlaps

    component_paths = {}

    def _get_nth_component_path(index: int) -> pathops.Path:
        if index not in component_paths:
            component_paths[index] = skPathFromGlyphComponent(
                glyph.components[index], glyphSet
            )
        return component_paths[index]

    return any(
        pathops.op(
            _get_nth_component_path(i),
            _get_nth_component_path(j),
            pathops.PathOp.INTERSECTION,
            fix_winding=False,
            keep_starting_points=False,
        )
        for i, j in itertools.combinations(range(len(glyph.components)), 2)
    )


def ttfGlyphFromSkPath(path: pathops.Path) -> _g_l_y_f.Glyph:
    # Skia paths have no 'components', no need for glyphSet
    ttPen = TTGlyphPen(glyphSet=None)
    path.draw(ttPen)
    glyph = ttPen.glyph()
    assert not glyph.isComposite()
    # compute glyph.xMin (glyfTable parameter unused for non composites)
    glyph.recalcBounds(glyfTable=None)
    return glyph


def _charString_from_SkPath(
    path: pathops.Path, charString: T2CharString
) -> T2CharString:
    if charString.width == charString.private.defaultWidthX:
        width = None
    else:
        width = charString.width - charString.private.nominalWidthX
    t2Pen = T2CharStringPen(width=width, glyphSet=None)
    path.draw(t2Pen)
    return t2Pen.getCharString(charString.private, charString.globalSubrs)


def _round_path(
    path: pathops.Path, round: Callable[[float], float] = otRound
) -> pathops.Path:
    rounded_path = pathops.Path()
    for verb, points in path:
        rounded_path.add(verb, *((round(p[0]), round(p[1])) for p in points))
    return rounded_path


def _simplify(
    path: pathops.Path,
    debugGlyphName: str,
    *,
    round: Callable[[float], float] = otRound,
) -> pathops.Path:
    # skia-pathops has a bug where it sometimes fails to simplify paths when there
    # are float coordinates and control points are very close to one another.
    # Rounding coordinates to integers works around the bug.
    # Since we are going to round glyf coordinates later on anyway, here it is
    # ok(-ish) to also round before simplify. Better than failing the whole process
    # for the entire font.
    # https://bugs.chromium.org/p/skia/issues/detail?id=11958
    # https://github.com/google/fonts/issues/3365
    # TODO(anthrotype): remove once this Skia bug is fixed
    try:
        return pathops.simplify(path, clockwise=path.clockwise)
    except pathops.PathOpsError:
        pass

    path = _round_path(path, round=round)
    try:
        path = pathops.simplify(path, clockwise=path.clockwise)
        log.debug(
            "skia-pathops failed to simplify '%s' with float coordinates, "
            "but succeded using rounded integer coordinates",
            debugGlyphName,
        )
        return path
    except pathops.PathOpsError as e:
        if log.isEnabledFor(logging.DEBUG):
            path.dump()
        raise RemoveOverlapsError(
            f"Failed to remove overlaps from glyph {debugGlyphName!r}"
        ) from e

    raise AssertionError("Unreachable")


def _same_path(path1: pathops.Path, path2: pathops.Path) -> bool:
    return {tuple(c) for c in path1.contours} == {tuple(c) for c in path2.contours}


def removeTTGlyphOverlaps(
    glyphName: str,
    glyphSet: _TTGlyphMapping,
    glyfTable: _g_l_y_f.table__g_l_y_f,
    hmtxTable: _h_m_t_x.table__h_m_t_x,
    removeHinting: bool = True,
) -> bool:
    glyph = glyfTable[glyphName]
    # decompose composite glyphs only if components overlap each other
    if (
        glyph.numberOfContours > 0
        or glyph.isComposite()
        and componentsOverlap(glyph, glyphSet)
    ):
        path = skPathFromGlyph(glyphName, glyphSet)

        # remove overlaps
        path2 = _simplify(path, glyphName)

        # replace TTGlyph if simplified path is different (ignoring contour order)
        if not _same_path(path, path2):
            glyfTable[glyphName] = glyph = ttfGlyphFromSkPath(path2)
            # simplified glyph is always unhinted
            assert not glyph.program
            # also ensure hmtx LSB == glyph.xMin so glyph origin is at x=0
            width, lsb = hmtxTable[glyphName]
            if lsb != glyph.xMin:
                hmtxTable[glyphName] = (width, glyph.xMin)
            return True

    if removeHinting:
        glyph.removeHinting()
    return False


def _remove_glyf_overlaps(
    *,
    font: ttFont.TTFont,
    glyphNames: Iterable[str],
    glyphSet: _TTGlyphMapping,
    removeHinting: bool,
    ignoreErrors: bool,
) -> None:
    glyfTable = font["glyf"]
    hmtxTable = font["hmtx"]

    # process all simple glyphs first, then composites with increasing component depth,
    # so that by the time we test for component intersections the respective base glyphs
    # have already been simplified
    glyphNames = sorted(
        glyphNames,
        key=lambda name: (
            (
                glyfTable[name].getCompositeMaxpValues(glyfTable).maxComponentDepth
                if glyfTable[name].isComposite()
                else 0
            ),
            name,
        ),
    )
    modified = set()
    for glyphName in glyphNames:
        try:
            if removeTTGlyphOverlaps(
                glyphName, glyphSet, glyfTable, hmtxTable, removeHinting
            ):
                modified.add(glyphName)
        except RemoveOverlapsError:
            if not ignoreErrors:
                raise
            log.error("Failed to remove overlaps for '%s'", glyphName)

    log.debug("Removed overlaps for %s glyphs:\n%s", len(modified), " ".join(modified))


def _remove_charstring_overlaps(
    *,
    glyphName: str,
    glyphSet: _TTGlyphMapping,
    cffFontSet: CFFFontSet,
) -> bool:
    path = skPathFromGlyph(glyphName, glyphSet)

    # remove overlaps
    path2 = _simplify(path, glyphName, round=noRound)

    # replace TTGlyph if simplified path is different (ignoring contour order)
    if not _same_path(path, path2):
        charStrings = cffFontSet[0].CharStrings
        charStrings[glyphName] = _charString_from_SkPath(path2, charStrings[glyphName])
        return True

    return False


def _remove_cff_overlaps(
    *,
    font: ttFont.TTFont,
    glyphNames: Iterable[str],
    glyphSet: _TTGlyphMapping,
    removeHinting: bool,
    ignoreErrors: bool,
    table_tag: str,
    removeUnusedSubroutines: bool = True,
) -> None:
    cffFontSet = font[table_tag].cff
    modified = set()
    for glyphName in glyphNames:
        try:
            if _remove_charstring_overlaps(
                glyphName=glyphName,
                glyphSet=glyphSet,
                cffFontSet=cffFontSet,
            ):
                modified.add(glyphName)
        except RemoveOverlapsError:
            if not ignoreErrors:
                raise
            log.error("Failed to remove overlaps for '%s'", glyphName)

    if not modified:
        log.debug("No overlaps found in the specified CFF glyphs")
        return

    if removeHinting:
        cffFontSet.remove_hints()

    if removeUnusedSubroutines:
        cffFontSet.remove_unused_subroutines()

    log.debug("Removed overlaps for %s glyphs:\n%s", len(modified), " ".join(modified))


def removeOverlaps(
    font: ttFont.TTFont,
    glyphNames: Optional[Iterable[str]] = None,
    removeHinting: bool = True,
    ignoreErrors: bool = False,
    *,
    removeUnusedSubroutines: bool = True,
) -> None:
    """Simplify glyphs in TTFont by merging overlapping contours.

    Overlapping components are first decomposed to simple contours, then merged.

    Currently this only works for fonts with 'glyf' or 'CFF ' tables.
    Raises NotImplementedError if 'glyf' or 'CFF ' tables are absent.

    Note that removing overlaps invalidates the hinting. By default we drop hinting
    from all glyphs whether or not overlaps are removed from a given one, as it would
    look weird if only some glyphs are left (un)hinted.

    Args:
        font: input TTFont object, modified in place.
        glyphNames: optional iterable of glyph names (str) to remove overlaps from.
            By default, all glyphs in the font are processed.
        removeHinting (bool): set to False to keep hinting for unmodified glyphs.
        ignoreErrors (bool): set to True to ignore errors while removing overlaps,
            thus keeping the tricky glyphs unchanged (fonttools/fonttools#2363).
        removeUnusedSubroutines (bool): set to False to keep unused subroutines
            in CFF table after removing overlaps. Default is to remove them if
            any glyphs are modified.
    """

    if "glyf" not in font and "CFF " not in font and "CFF2" not in font:
        raise NotImplementedError(
            "No outline data found in the font: missing 'glyf', 'CFF ', or 'CFF2' table"
        )

    if glyphNames is None:
        glyphNames = font.getGlyphOrder()

    # Wraps the underlying glyphs, takes care of interfacing with drawing pens
    glyphSet = font.getGlyphSet()

    if "glyf" in font:
        _remove_glyf_overlaps(
            font=font,
            glyphNames=glyphNames,
            glyphSet=glyphSet,
            removeHinting=removeHinting,
            ignoreErrors=ignoreErrors,
        )

    if "CFF " in font or "CFF2" in font:
        _remove_cff_overlaps(
            font=font,
            glyphNames=glyphNames,
            glyphSet=glyphSet,
            removeHinting=removeHinting,
            ignoreErrors=ignoreErrors,
            table_tag="CFF " if "CFF " in font else "CFF2",
            removeUnusedSubroutines=removeUnusedSubroutines,
        )


def main(args=None):
    """Simplify glyphs in TTFont by merging overlapping contours."""

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools ttLib.removeOverlaps", description=__doc__
    )

    parser.add_argument("input", metavar="INPUT.ttf", help="Input font file")
    parser.add_argument("output", metavar="OUTPUT.ttf", help="Output font file")
    parser.add_argument(
        "glyphs",
        metavar="GLYPHS",
        nargs="*",
        help="Optional list of glyph names to remove overlaps from",
    )
    parser.add_argument(
        "--keep-hinting",
        action="store_true",
        help="Keep hinting for unmodified glyphs, default is to drop hinting",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="ignore errors while removing overlaps, "
        "thus keeping the tricky glyphs unchanged",
    )
    parser.add_argument(
        "--keep-unused-subroutines",
        action="store_true",
        help="Keep unused subroutines in CFF table after removing overlaps, "
        "default is to remove them if any glyphs are modified",
    )
    args = parser.parse_args(args)

    with ttFont.TTFont(args.input) as font:
        removeOverlaps(
            font=font,
            glyphNames=args.glyphs or None,
            removeHinting=not args.keep_hinting,
            ignoreErrors=args.ignore_errors,
            removeUnusedSubroutines=not args.keep_unused_subroutines,
        )
        font.save(args.output)


if __name__ == "__main__":
    main()
<|endoftext|>"""ttLib/sfnt.py -- low-level module to deal with the sfnt file format.

Defines two public classes:

- SFNTReader
- SFNTWriter

(Normally you don't have to use these classes explicitly; they are
used automatically by ttLib.TTFont.)

The reading and writing of sfnt files is separated in two distinct
classes, since whenever the number of tables changes or whenever
a table's length changes you need to rewrite the whole file anyway.
"""

from __future__ import annotations

from collections.abc import KeysView
from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging

log = logging.getLogger(__name__)


class SFNTReader(object):
    def __new__(cls, *args, **kwargs):
        """Return an instance of the SFNTReader sub-class which is compatible
        with the input file type.
        """
        if args and cls is SFNTReader:
            infile = args[0]
            infile.seek(0)
            sfntVersion = Tag(infile.read(4))
            infile.seek(0)
            if sfntVersion == "wOF2":
                # return new WOFF2Reader object
                from fontTools.ttLib.woff2 import WOFF2Reader

                return object.__new__(WOFF2Reader)
        # return default object
        return object.__new__(cls)

    def __init__(self, file, checkChecksums=0, fontNumber=-1):
        self.file = file
        self.checkChecksums = checkChecksums

        self.flavor = None
        self.flavorData = None
        self.DirectoryEntry = SFNTDirectoryEntry
        self.file.seek(0)
        self.sfntVersion = self.file.read(4)
        self.file.seek(0)
        if self.sfntVersion == b"ttcf":
            header = readTTCHeader(self.file)
            numFonts = header.numFonts
            if not 0 <= fontNumber < numFonts:
                raise TTLibFileIsCollectionError(
                    "specify a font number between 0 and %d (inclusive)"
                    % (numFonts - 1)
                )
            self.numFonts = numFonts
            self.file.seek(header.offsetTable[fontNumber])
            data = self.file.read(sfntDirectorySize)
            if len(data) != sfntDirectorySize:
                raise TTLibError("Not a Font Collection (not enough data)")
            sstruct.unpack(sfntDirectoryFormat, data, self)
        elif self.sfntVersion == b"wOFF":
            self.flavor = "woff"
            self.DirectoryEntry = WOFFDirectoryEntry
            data = self.file.read(woffDirectorySize)
            if len(data) != woffDirectorySize:
                raise TTLibError("Not a WOFF font (not enough data)")
            sstruct.unpack(woffDirectoryFormat, data, self)
        else:
            data = self.file.read(sfntDirectorySize)
            if len(data) != sfntDirectorySize:
                raise TTLibError("Not a TrueType or OpenType font (not enough data)")
            sstruct.unpack(sfntDirectoryFormat, data, self)
        self.sfntVersion = Tag(self.sfntVersion)

        if self.sfntVersion not in ("\x00\x01\x00\x00", "OTTO", "true"):
            raise TTLibError("Not a TrueType or OpenType font (bad sfntVersion)")
        tables: dict[Tag, DirectoryEntry] = {}
        for i in range(self.numTables):
            entry = self.DirectoryEntry()
            entry.fromFile(self.file)
            tag = Tag(entry.tag)
            tables[tag] = entry
        self.tables = OrderedDict(sorted(tables.items(), key=lambda i: i[1].offset))

        # Load flavor data if any
        if self.flavor == "woff":
            self.flavorData = WOFFFlavorData(self)

    def has_key(self, tag: str | bytes) -> bool:
        return tag in self.tables

    __contains__ = has_key

    def keys(self) -> KeysView[Tag]:
        return self.tables.keys()

    def __getitem__(self, tag: str | bytes) -> bytes:
        """Fetch the raw table data."""
        entry = self.tables[Tag(tag)]
        data = entry.loadData(self.file)
        if self.checkChecksums:
            if tag == "head":
                # Beh: we have to special-case the 'head' table.
                checksum = calcChecksum(data[:8] + b"\0\0\0\0" + data[12:])
            else:
                checksum = calcChecksum(data)
            if self.checkChecksums > 1:
                # Be obnoxious, and barf when it's wrong
                assert checksum == entry.checkSum, "bad checksum for '%s' table" % tag
            elif checksum != entry.checkSum:
                # Be friendly, and just log a warning.
                log.warning("bad checksum for '%s' table", tag)
        return data

    def __delitem__(self, tag: str | bytes) -> None:
        del self.tables[Tag(tag)]

    def close(self) -> None:
        self.file.close()

    # We define custom __getstate__ and __setstate__ to make SFNTReader pickle-able
    # and deepcopy-able. When a TTFont is loaded as lazy=True, SFNTReader holds a
    # reference to an external file object which is not pickleable. So in __getstate__
    # we store the file name and current position, and in __setstate__ we reopen the
    # same named file after unpickling.

    def __getstate__(self):
        if isinstance(self.file, BytesIO):
            # BytesIO is already pickleable, return the state unmodified
            return self.__dict__

        # remove unpickleable file attribute, and only store its name and pos
        state = self.__dict__.copy()
        del state["file"]
        state["_filename"] = self.file.name
        state["_filepos"] = self.file.tell()
        return state

    def __setstate__(self, state):
        if "file" not in state:
            self.file = open(state.pop("_filename"), "rb")
            self.file.seek(state.pop("_filepos"))
        self.__dict__.update(state)


# default compression level for WOFF 1.0 tables and metadata
ZLIB_COMPRESSION_LEVEL = 6

# if set to True, use zopfli instead of zlib for compressing WOFF 1.0.
# The Python bindings are available at https://pypi.python.org/pypi/zopfli
USE_ZOPFLI = False

# mapping between zlib's compression levels and zopfli's 'numiterations'.
# Use lower values for files over several MB in size or it will be too slow
ZOPFLI_LEVELS = {
    # 0: 0,  # can't do 0 iterations...
    1: 1,
    2: 3,
    3: 5,
    4: 8,
    5: 10,
    6: 15,
    7: 25,
    8: 50,
    9: 100,
}


def compress(data, level=ZLIB_COMPRESSION_LEVEL):
    """Compress 'data' to Zlib format. If 'USE_ZOPFLI' variable is True,
    zopfli is used instead of the zlib module.
    The compression 'level' must be between 0 and 9. 1 gives best speed,
    9 gives best compression (0 gives no compression at all).
    The default value is a compromise between speed and compression (6).
    """
    if not (0 <= level <= 9):
        raise ValueError("Bad compression level: %s" % level)
    if not USE_ZOPFLI or level == 0:
        from zlib import compress

        return compress(data, level)
    else:
        from zopfli.zlib import compress

        return compress(data, numiterations=ZOPFLI_LEVELS[level])


class SFNTWriter(object):
    def __new__(cls, *args, **kwargs):
        """Return an instance of the SFNTWriter sub-class which is compatible
        with the specified 'flavor'.
        """
        flavor = None
        if kwargs and "flavor" in kwargs:
            flavor = kwargs["flavor"]
        elif args and len(args) > 3:
            flavor = args[3]
        if cls is SFNTWriter:
            if flavor == "woff2":
                # return new WOFF2Writer object
                from fontTools.ttLib.woff2 import WOFF2Writer

                return object.__new__(WOFF2Writer)
        # return default object
        return object.__new__(cls)

    def __init__(
        self,
        file,
        numTables,
        sfntVersion="\000\001\000\000",
        flavor=None,
        flavorData=None,
    ):
        self.file = file
        self.numTables = numTables
        self.sfntVersion = Tag(sfntVersion)
        self.flavor = flavor
        self.flavorData = flavorData

        if self.flavor == "woff":
            self.directoryFormat = woffDirectoryFormat
            self.directorySize = woffDirectorySize
            self.DirectoryEntry = WOFFDirectoryEntry

            self.signature = "wOFF"

            # to calculate WOFF checksum adjustment, we also need the original SFNT offsets
            self.origNextTableOffset = (
                sfntDirectorySize + numTables * sfntDirectoryEntrySize
            )
        else:
            assert not self.flavor, "Unknown flavor '%s'" % self.flavor
            self.directoryFormat = sfntDirectoryFormat
            self.directorySize = sfntDirectorySize
            self.DirectoryEntry = SFNTDirectoryEntry

            from fontTools.ttLib import getSearchRange

            self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(
                numTables, 16
            )

        self.directoryOffset = self.file.tell()
        self.nextTableOffset = (
            self.directoryOffset
            + self.directorySize
            + numTables * self.DirectoryEntry.formatSize
        )
        # clear out directory area
        self.file.seek(self.nextTableOffset)
        # make sure we're actually where we want to be. (old cStringIO bug)
        self.file.write(b"\0" * (self.nextTableOffset - self.file.tell()))
        self.tables = OrderedDict()

    def setEntry(self, tag, entry):
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)

        self.tables[tag] = entry

    def __setitem__(self, tag, data):
        """Write raw table data to disk."""
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)

        entry = self.DirectoryEntry()
        entry.tag = tag
        entry.offset = self.nextTableOffset
        if tag == "head":
            entry.checkSum = calcChecksum(data[:8] + b"\0\0\0\0" + data[12:])
            self.headTable = data
            entry.uncompressed = True
        else:
            entry.checkSum = calcChecksum(data)
        entry.saveData(self.file, data)

        if self.flavor == "woff":
            entry.origOffset = self.origNextTableOffset
            self.origNextTableOffset += (entry.origLength + 3) & ~3

        self.nextTableOffset = self.nextTableOffset + ((entry.length + 3) & ~3)
        # Add NUL bytes to pad the table data to a 4-byte boundary.
        # Don't depend on f.seek() as we need to add the padding even if no
        # subsequent write follows (seek is lazy), ie. after the final table
        # in the font.
        self.file.write(b"\0" * (self.nextTableOffset - self.file.tell()))
        assert self.nextTableOffset == self.file.tell()

        self.setEntry(tag, entry)

    def __getitem__(self, tag):
        return self.tables[tag]

    def close(self):
        """All tables must have been written to disk. Now write the
        directory.
        """
        tables = sorted(self.tables.items())
        if len(tables) != self.numTables:
            raise TTLibError(
                "wrong number of tables; expected %d, found %d"
                % (self.numTables, len(tables))
            )

        if self.flavor == "woff":
            self.signature = b"wOFF"
            self.reserved = 0

            self.totalSfntSize = 12
            self.totalSfntSize += 16 * len(tables)
            for tag, entry in tables:
                self.totalSfntSize += (entry.origLength + 3) & ~3

            data = self.flavorData if self.flavorData else WOFFFlavorData()
            if data.majorVersion is not None and data.minorVersion is not None:
                self.majorVersion = data.majorVersion
                self.minorVersion = data.minorVersion
            else:
                if hasattr(self, "headTable"):
                    self.majorVersion, self.minorVersion = struct.unpack(
                        ">HH", self.headTable[4:8]
                    )
                else:
                    self.majorVersion = self.minorVersion = 0
            if data.metaData:
                self.metaOrigLength = len(data.metaData)
                self.file.seek(0, 2)
                self.metaOffset = self.file.tell()
                compressedMetaData = compress(data.metaData)
                self.metaLength = len(compressedMetaData)
                self.file.write(compressedMetaData)
            else:
                self.metaOffset = self.metaLength = self.metaOrigLength = 0
            if data.privData:
                self.file.seek(0, 2)
                off = self.file.tell()
                paddedOff = (off + 3) & ~3
                self.file.write(b"\0" * (paddedOff - off))
                self.privOffset = self.file.tell()
                self.privLength = len(data.privData)
                self.file.write(data.privData)
            else:
                self.privOffset = self.privLength = 0

            self.file.seek(0, 2)
            self.length = self.file.tell()

        else:
            assert not self.flavor, "Unknown flavor '%s'" % self.flavor
            pass

        directory = sstruct.pack(self.directoryFormat, self)

        self.file.seek(self.directoryOffset + self.directorySize)
        seenHead = 0
        for tag, entry in tables:
            if tag == "head":
                seenHead = 1
            directory = directory + entry.toString()
        if seenHead:
            self.writeMasterChecksum(directory)
        self.file.seek(self.directoryOffset)
        self.file.write(directory)

    def _calcMasterChecksum(self, directory):
        # calculate checkSumAdjustment
        checksums = []
        for tag in self.tables.keys():
            checksums.append(self.tables[tag].checkSum)

        if self.DirectoryEntry != SFNTDirectoryEntry:
            # Create a SFNT directory for checksum calculation purposes
            from fontTools.ttLib import getSearchRange

            self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(
                self.numTables, 16
            )
            directory = sstruct.pack(sfntDirectoryFormat, self)
            tables = sorted(self.tables.items())
            for tag, entry in tables:
                sfntEntry = SFNTDirectoryEntry()
                sfntEntry.tag = entry.tag
                sfntEntry.checkSum = entry.checkSum
                sfntEntry.offset = entry.origOffset
                sfntEntry.length = entry.origLength
                directory = directory + sfntEntry.toString()

        directory_end = sfntDirectorySize + len(self.tables) * sfntDirectoryEntrySize
        assert directory_end == len(directory)

        checksums.append(calcChecksum(directory))
        checksum = sum(checksums) & 0xFFFFFFFF
        # BiboAfba!
        checksumadjustment = (0xB1B0AFBA - checksum) & 0xFFFFFFFF
        return checksumadjustment

    def writeMasterChecksum(self, directory):
        checksumadjustment = self._calcMasterChecksum(directory)
        # write the checksum to the file
        self.file.seek(self.tables["head"].offset + 8)
        self.file.write(struct.pack(">L", checksumadjustment))

    def reordersTables(self):
        return False


# -- sfnt directory helpers and cruft

ttcHeaderFormat = """
		> # big endian
		TTCTag:                  4s # "ttcf"
		Version:                 L  # 0x00010000 or 0x00020000
		numFonts:                L  # number of fonts
		# OffsetTable[numFonts]: L  # array with offsets from beginning of file
		# ulDsigTag:             L  # version 2.0 only
		# ulDsigLength:          L  # version 2.0 only
		# ulDsigOffset:          L  # version 2.0 only
"""

ttcHeaderSize = sstruct.calcsize(ttcHeaderFormat)

ttcTailFormatV2 = """
		> # big endian
		ulDsigTag:               L  # version 2.0 only
		ulDsigLength:            L  # version 2.0 only
		ulDsigOffset:            L  # version 2.0 only
"""

ttcTailSizeV2 = sstruct.calcsize(ttcTailFormatV2)

TTC_V1 = 0x00010000
TTC_V2 = 0x00020000

sfntDirectoryFormat = """
		> # big endian
		sfntVersion:    4s
		numTables:      H    # number of tables
		searchRange:    H    # (max2 <= numTables)*16
		entrySelector:  H    # log2(max2 <= numTables)
		rangeShift:     H    # numTables*16-searchRange
"""

sfntDirectorySize = sstruct.calcsize(sfntDirectoryFormat)

sfntDirectoryEntryFormat = """
		> # big endian
		tag:            4s
		checkSum:       L
		offset:         L
		length:         L
"""

sfntDirectoryEntrySize = sstruct.calcsize(sfntDirectoryEntryFormat)

woffDirectoryFormat = """
		> # big endian
		signature:      4s   # "wOFF"
		sfntVersion:    4s
		length:         L    # total woff file size
		numTables:      H    # number of tables
		reserved:       H    # set to 0
		totalSfntSize:  L    # uncompressed size
		majorVersion:   H    # major version of WOFF file
		minorVersion:   H    # minor version of WOFF file
		metaOffset:     L    # offset to metadata block
		metaLength:     L    # length of compressed metadata
		metaOrigLength: L    # length of uncompressed metadata
		privOffset:     L    # offset to private data block
		privLength:     L    # length of private data block
"""

woffDirectorySize = sstruct.calcsize(woffDirectoryFormat)

woffDirectoryEntryFormat = """
		> # big endian
		tag:            4s
		offset:         L
		length:         L    # compressed length
		origLength:     L    # original length
		checkSum:       L    # original checksum
"""

woffDirectoryEntrySize = sstruct.calcsize(woffDirectoryEntryFormat)


class DirectoryEntry(object):
    def __init__(self):
        self.uncompressed = False  # if True, always embed entry raw

    def fromFile(self, file):
        data = file.read(self.formatSize)
        if len(data) != self.formatSize:
            # A file truncated inside the table directory, or one whose
            # numTables is corrupt, would otherwise fail with a struct.error.
            raise TTLibError(
                "unexpected end of table directory: expected %d bytes but got %d"
                % (self.formatSize, len(data))
            )
        sstruct.unpack(self.format, data, self)

    def fromString(self, str):
        sstruct.unpack(self.format, str, self)

    def toString(self):
        return sstruct.pack(self.format, self)

    def __repr__(self):
        if hasattr(self, "tag"):
            return "<%s '%s' at %x>" % (self.__class__.__name__, self.tag, id(self))
        else:
            return "<%s at %x>" % (self.__class__.__name__, id(self))

    def loadData(self, file):
        file.seek(self.offset)
        data = file.read(self.length)
        if len(data) != self.length:
            # A corrupt or truncated table directory entry can point past the
            # end of the file; raise a TTLibError instead of a bare assertion
            # (which is also silently skipped under `python -O`).
            tag = getattr(self, "tag", None)
            raise TTLibError(
                "unexpected end of '%s' table data: expected %d bytes but got "
                "%d at offset %d"
                % (
                    Tag(tag) if tag is not None else "????",
                    self.length,
                    len(data),
                    self.offset,
                )
            )
        if hasattr(self.__class__, "decodeData"):
            data = self.decodeData(data)
        return data

    def saveData(self, file, data):
        if hasattr(self.__class__, "encodeData"):
            data = self.encodeData(data)
        self.length = len(data)
        file.seek(self.offset)
        file.write(data)

    def decodeData(self, rawData):
        return rawData

    def encodeData(self, data):
        return data


class SFNTDirectoryEntry(DirectoryEntry):
    format = sfntDirectoryEntryFormat
    formatSize = sfntDirectoryEntrySize


class WOFFDirectoryEntry(DirectoryEntry):
    format = woffDirectoryEntryFormat
    formatSize = woffDirectoryEntrySize

    def __init__(self):
        super(WOFFDirectoryEntry, self).__init__()
        # With fonttools<=3.1.2, the only way to set a different zlib
        # compression level for WOFF directory entries was to set the class
        # attribute 'zlibCompressionLevel'. This is now replaced by a globally
        # defined `ZLIB_COMPRESSION_LEVEL`, which is also applied when
        # compressing the metadata. For backward compatibility, we still
        # use the class attribute if it was already set.
        if not hasattr(WOFFDirectoryEntry, "zlibCompressionLevel"):
            self.zlibCompressionLevel = ZLIB_COMPRESSION_LEVEL

    def decodeData(self, rawData):
        import zlib

        if self.length == self.origLength:
            data = rawData
        else:
            assert self.length < self.origLength
            data = zlib.decompress(rawData)
            assert len(data) == self.origLength
        return data

    def encodeData(self, data):
        self.origLength = len(data)
        if not self.uncompressed:
            compressedData = compress(data, self.zlibCompressionLevel)
        if self.uncompressed or len(compressedData) >= self.origLength:
            # Encode uncompressed
            rawData = data
            self.length = self.origLength
        else:
            rawData = compressedData
            self.length = len(rawData)
        return rawData


class WOFFFlavorData:
    Flavor = "woff"

    def __init__(self, reader=None):
        self.majorVersion = None
        self.minorVersion = None
        self.metaData = None
        self.privData = None
        if reader:
            self.majorVersion = reader.majorVersion
            self.minorVersion = reader.minorVersion
            if reader.metaLength:
                reader.file.seek(reader.metaOffset)
                rawData = reader.file.read(reader.metaLength)
                assert len(rawData) == reader.metaLength
                data = self._decompress(rawData)
                assert len(data) == reader.metaOrigLength
                self.metaData = data
            if reader.privLength:
                reader.file.seek(reader.privOffset)
                data = reader.file.read(reader.privLength)
                assert len(data) == reader.privLength
                self.privData = data

    def _decompress(self, rawData):
        import zlib

        return zlib.decompress(rawData)


def calcChecksum(data):
    """Calculate the checksum for an arbitrary block of data.

    If the data length is not a multiple of four, it assumes
    it is to be padded with null byte.

            >>> print(calcChecksum(b"abcd"))
            1633837924
            >>> print(calcChecksum(b"abcdxyz"))
            3655064932
    """
    remainder = len(data) % 4
    if remainder:
        data += b"\0" * (4 - remainder)
    value = 0
    blockSize = 4096
    assert blockSize % 4 == 0
    for i in range(0, len(data), blockSize):
        block = data[i : i + blockSize]
        longs = struct.unpack(">%dL" % (len(block) // 4), block)
        value = (value + sum(longs)) & 0xFFFFFFFF
    return value


def readTTCHeader(file):
    file.seek(0)
    data = file.read(ttcHeaderSize)
    if len(data) != ttcHeaderSize:
        raise TTLibError("Not a Font Collection (not enough data)")
    self = SimpleNamespace()
    sstruct.unpack(ttcHeaderFormat, data, self)
    if self.TTCTag != "ttcf":
        raise TTLibError("Not a Font Collection")
    assert self.Version in (TTC_V1, TTC_V2), (
        "unrecognized TTC version 0x%08x" % self.Version
    )
    self.offsetTable = struct.unpack(
        ">%dL" % self.numFonts, file.read(self.numFonts * 4)
    )
    if self.Version == TTC_V2:
        # Unpack additional DSIG fields
        data = file.read(ttcTailSizeV2)
        if len(data) != ttcTailSizeV2:
            raise TTLibError("Not a Font Collection (not enough data)")
        sstruct.unpack(ttcTailFormatV2, data, self)
    return self


def writeTTCHeader(file, numFonts, version=TTC_V1):
    self = SimpleNamespace()
    self.TTCTag = "ttcf"
    self.Version = version
    assert self.Version in (TTC_V1, TTC_V2), (
        "unrecognized TTC version 0x%08x" % self.Version
    )
    self.numFonts = numFonts
    file.seek(0)
    file.write(sstruct.pack(ttcHeaderFormat, self))
    offset = file.tell()
    file.write(struct.pack(">%dL" % self.numFonts, *([0] * self.numFonts)))
    if version == TTC_V2:
        # write empty ulDsigTag, ulDsigLength, ulDsigOffset
        # Actual values are be written in TTCollection.save()
        file.write(struct.pack(">3L", 0, 0, 0))
    return offset


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>from fontTools.ttLib.ttFont import TTFont
from io import BytesIO
import struct
import logging
from contextlib import contextmanager
from fontTools.ttLib.sfnt import TTC_V1, TTC_V2, readTTCHeader, writeTTCHeader
from fontTools.ttLib.tables.D_S_I_G_ import table_D_S_I_G_
from fontTools.misc.timeTools import timestampNow

log = logging.getLogger(__name__)


@contextmanager
def _sharedModifiedTimestamp(fonts):
    """Pin a single ``head.modified`` timestamp across an iterable of TTFonts.

    When several fonts are saved together with ``recalcTimestamp=True``, each
    one restamps its own ``head`` table via ``timestampNow()``, whose
    granularity is one second. If the wall clock ticks between two fonts'
    ``head.compile()`` calls they get different ``modified`` values, which
    defeats byte-identical table sharing (``shareTables=True``) and makes the
    saved size non-deterministic.

    This reads ``timestampNow()`` once (so ``SOURCE_DATE_EPOCH`` is still
    honored) and applies it to every font that would have recalculated its own
    timestamp, suppressing the per-font restamp for the duration. Each font's
    ``recalcTimestamp`` flag is restored on exit, even if the body raises; the
    pinned ``modified`` value is left in place, matching a normal recalc save.
    Fonts with ``recalcTimestamp`` falsy, or without a ``head`` table, are left
    untouched (their ``head`` is not even loaded).

    See https://github.com/fonttools/fonttools/issues/4110.
    """
    now = timestampNow()
    restore = []
    try:
        for font in fonts:
            if font.recalcTimestamp and "head" in font:
                # record the original flag *before* mutating, so the finally
                # block restores it even if loading 'head' below raises
                restore.append((font, font.recalcTimestamp))
                font["head"].modified = now
                font.recalcTimestamp = False
        yield now
    finally:
        for font, recalcTimestamp in restore:
            font.recalcTimestamp = recalcTimestamp


class TTCollection(object):
    """Object representing a TrueType Collection / OpenType Collection.
    The main API is self.fonts being a list of TTFont instances.

    If shareTables is True, then different fonts in the collection
    might point to the same table object if the data for the table was
    the same in the font file.  Note, however, that this might result
    in suprises and incorrect behavior if the different fonts involved
    have different GlyphOrder.  Use only if you know what you are doing.
    """

    def __init__(self, file=None, shareTables=False, **kwargs):
        fonts = self.fonts = []
        if file is None:
            return

        assert "fontNumber" not in kwargs, kwargs

        closeStream = False
        if not hasattr(file, "read"):
            file = open(file, "rb")
            closeStream = True

        tableCache = {} if shareTables else None

        header = readTTCHeader(file)
        for i in range(header.numFonts):
            font = TTFont(file, fontNumber=i, _tableCache=tableCache, **kwargs)
            fonts.append(font)

        if header.Version == TTC_V2:
            self.dsig: table_D_S_I_G_ | None = None
            if header.ulDsigOffset != 0:
                self.dsig = table_D_S_I_G_("DSIG")
                file.seek(header.ulDsigOffset, 0)
                self.dsig.data = file.read(header.ulDsigLength)

        # don't close file if lazy=True, as the TTFont hold a reference to the original
        # file; the file will be closed once the TTFonts are closed in the
        # TTCollection.close(). We still want to close the file if lazy is None or
        # False, because in that case the TTFont no longer need the original file
        # and we want to avoid 'ResourceWarning: unclosed file'.
        if not kwargs.get("lazy") and closeStream:
            file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        for font in self.fonts:
            font.close()

    def save(self, file, shareTables=True):
        """Save the font to disk. Similarly to the constructor,
        the 'file' argument can be either a pathname or a writable
        file object.
        """
        if not hasattr(file, "write"):
            final = None
            file = open(file, "wb")
        else:
            # assume "file" is a writable file object
            # write to a temporary stream to allow saving to unseekable streams
            final = file
            file = BytesIO()

        tableCache = {} if shareTables else None

        # A V2 TTC will be saved if self.dsig is present, even if it is None
        version = TTC_V2 if hasattr(self, "dsig") else TTC_V1

        # Pin one 'modified' timestamp so the fonts' 'head' tables can be shared.
        with _sharedModifiedTimestamp(self.fonts):
            offsets_offset = writeTTCHeader(file, len(self.fonts), version=version)
            offsets = []
            for font in self.fonts:
                offsets.append(file.tell())
                font._save(file, tableCache=tableCache)
                file.seek(0, 2)

        file.seek(offsets_offset)
        file.write(struct.pack(">%dL" % len(self.fonts), *offsets))

        if version == TTC_V2 and self.dsig is not None:
            # Compile the DSIG if necessary
            if hasattr(self.dsig, "data"):
                data = self.dsig.data
            else:
                data = self.dsig.compile(None)
            # Write the DSIG tag, length, and offset
            # We are at the offset where the DSIG header starts
            dsig_header_fields_offset = file.tell()
            # The DSIG data will be written to the end of the file, go there
            file.seek(0, 2)
            dsig_offset = file.tell()
            # Write the actual data
            file.write(data)
            # Go back to the DSIG header
            file.seek(dsig_header_fields_offset)
            # Write the tag
            file.write(b"DSIG")
            # Write the length and offset
            file.write(struct.pack(">2L", len(data), dsig_offset))

        if final:
            final.write(file.getvalue())
        file.close()

    def saveXML(self, fileOrPath, newlinestr="\n", writeVersion=True, **kwargs):
        from fontTools.misc import xmlWriter

        writer = xmlWriter.XMLWriter(fileOrPath, newlinestr=newlinestr)

        if writeVersion:
            from fontTools import version

            version = ".".join(version.split(".")[:2])
            writer.begintag("ttCollection", ttLibVersion=version)
        else:
            writer.begintag("ttCollection")
        writer.newline()
        writer.newline()

        for font in self.fonts:
            font._saveXML(writer, writeVersion=False, **kwargs)
            writer.newline()

        writer.endtag("ttCollection")
        writer.newline()

        writer.close()

    def __getitem__(self, item):
        return self.fonts[item]

    def __setitem__(self, item, value):
        self.fonts[item] = value

    def __delitem__(self, item):
        return self.fonts[item]

    def __len__(self):
        return len(self.fonts)

    def __iter__(self):
        return iter(self.fonts)
<|endoftext|>"""Reorder glyphs in a font."""

__author__ = "Rod Sheeter"

# See https://docs.google.com/document/d/1h9O-C_ndods87uY0QeIIcgAMiX2gDTpvO_IhMJsKAqs/
# for details.


from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from fontTools.ttLib.tables import otTables as ot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from typing import (
    Optional,
    Any,
    Callable,
    Deque,
    Iterable,
    List,
    Tuple,
)

_COVERAGE_ATTR = "Coverage"  # tables that have one coverage use this name


def _sort_by_gid(
    get_glyph_id: Callable[[str], int],
    glyphs: List[str],
    parallel_list: Optional[List[Any]],
):
    if parallel_list:
        reordered = sorted(
            ((g, e) for g, e in zip(glyphs, parallel_list)),
            key=lambda t: get_glyph_id(t[0]),
        )
        sorted_glyphs, sorted_parallel_list = map(list, zip(*reordered))
        parallel_list[:] = sorted_parallel_list
    else:
        sorted_glyphs = sorted(glyphs, key=get_glyph_id)

    glyphs[:] = sorted_glyphs


def _get_dotted_attr(value: Any, dotted_attr: str) -> Any:
    attr_names = dotted_attr.split(".")
    assert attr_names

    while attr_names:
        attr_name = attr_names.pop(0)
        value = getattr(value, attr_name)
    return value


class ReorderRule(ABC):
    """A rule to reorder something in a font to match the fonts glyph order."""

    @abstractmethod
    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None: ...


@dataclass(frozen=True)
class ReorderCoverage(ReorderRule):
    """Reorder a Coverage table, and optionally a list that is sorted parallel to it."""

    # A list that is parallel to Coverage
    parallel_list_attr: Optional[str] = None
    coverage_attr: str = _COVERAGE_ATTR

    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None:
        coverage = _get_dotted_attr(value, self.coverage_attr)

        if type(coverage) is not list:
            # Normal path, process one coverage that might have a parallel list
            parallel_list = None
            if self.parallel_list_attr:
                parallel_list = _get_dotted_attr(value, self.parallel_list_attr)
                assert (
                    type(parallel_list) is list
                ), f"{self.parallel_list_attr} should be a list"
                assert len(parallel_list) == len(coverage.glyphs), "Nothing makes sense"

            _sort_by_gid(font.getGlyphID, coverage.glyphs, parallel_list)

        else:
            # A few tables have a list of coverage. No parallel list can exist.
            assert (
                not self.parallel_list_attr
            ), f"Can't have multiple coverage AND a parallel list; {self}"
            for coverage_entry in coverage:
                _sort_by_gid(font.getGlyphID, coverage_entry.glyphs, None)


@dataclass(frozen=True)
class ReorderList(ReorderRule):
    """Reorder the items within a list to match the updated glyph order.

    Useful when a list ordered by coverage itself contains something ordered by a gid.
    For example, the PairSet table of https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#lookup-type-2-pair-adjustment-positioning-subtable.
    """

    list_attr: str
    key: str

    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None:
        lst = _get_dotted_attr(value, self.list_attr)
        assert isinstance(lst, list), f"{self.list_attr} should be a list"
        lst.sort(key=lambda v: font.getGlyphID(getattr(v, self.key)))


# (Type, Optional Format) => List[ReorderRule]
# Encodes the relationships Cosimo identified
_REORDER_RULES = {
    # GPOS
    (ot.SinglePos, 1): [ReorderCoverage()],
    (ot.SinglePos, 2): [ReorderCoverage(parallel_list_attr="Value")],
    (ot.PairPos, 1): [ReorderCoverage(parallel_list_attr="PairSet")],
    (ot.PairSet, None): [ReorderList("PairValueRecord", key="SecondGlyph")],
    (ot.PairPos, 2): [ReorderCoverage()],
    (ot.CursivePos, 1): [ReorderCoverage(parallel_list_attr="EntryExitRecord")],
    (ot.MarkBasePos, 1): [
        ReorderCoverage(
            coverage_attr="MarkCoverage", parallel_list_attr="MarkArray.MarkRecord"
        ),
        ReorderCoverage(
            coverage_attr="BaseCoverage", parallel_list_attr="BaseArray.BaseRecord"
        ),
    ],
    (ot.MarkLigPos, 1): [
        ReorderCoverage(
            coverage_attr="MarkCoverage", parallel_list_attr="MarkArray.MarkRecord"
        ),
        ReorderCoverage(
            coverage_attr="LigatureCoverage",
            parallel_list_attr="LigatureArray.LigatureAttach",
        ),
    ],
    (ot.MarkMarkPos, 1): [
        ReorderCoverage(
            coverage_attr="Mark1Coverage", parallel_list_attr="Mark1Array.MarkRecord"
        ),
        ReorderCoverage(
            coverage_attr="Mark2Coverage", parallel_list_attr="Mark2Array.Mark2Record"
        ),
    ],
    (ot.ContextPos, 1): [ReorderCoverage(parallel_list_attr="PosRuleSet")],
    (ot.ContextPos, 2): [ReorderCoverage()],
    (ot.ContextPos, 3): [ReorderCoverage()],
    (ot.ChainContextPos, 1): [ReorderCoverage(parallel_list_attr="ChainPosRuleSet")],
    (ot.ChainContextPos, 2): [ReorderCoverage()],
    (ot.ChainContextPos, 3): [
        ReorderCoverage(coverage_attr="BacktrackCoverage"),
        ReorderCoverage(coverage_attr="InputCoverage"),
        ReorderCoverage(coverage_attr="LookAheadCoverage"),
    ],
    # GSUB
    (ot.ContextSubst, 1): [ReorderCoverage(parallel_list_attr="SubRuleSet")],
    (ot.ContextSubst, 2): [ReorderCoverage()],
    (ot.ContextSubst, 3): [ReorderCoverage()],
    (ot.ChainContextSubst, 1): [ReorderCoverage(parallel_list_attr="ChainSubRuleSet")],
    (ot.ChainContextSubst, 2): [ReorderCoverage()],
    (ot.ChainContextSubst, 3): [
        ReorderCoverage(coverage_attr="BacktrackCoverage"),
        ReorderCoverage(coverage_attr="InputCoverage"),
        ReorderCoverage(coverage_attr="LookAheadCoverage"),
    ],
    (ot.ReverseChainSingleSubst, 1): [
        ReorderCoverage(parallel_list_attr="Substitute"),
        ReorderCoverage(coverage_attr="BacktrackCoverage"),
        ReorderCoverage(coverage_attr="LookAheadCoverage"),
    ],
    # GDEF
    (ot.AttachList, None): [ReorderCoverage(parallel_list_attr="AttachPoint")],
    (ot.LigCaretList, None): [ReorderCoverage(parallel_list_attr="LigGlyph")],
    (ot.MarkGlyphSetsDef, None): [ReorderCoverage()],
    # MATH
    (ot.MathGlyphInfo, None): [ReorderCoverage(coverage_attr="ExtendedShapeCoverage")],
    (ot.MathItalicsCorrectionInfo, None): [
        ReorderCoverage(parallel_list_attr="ItalicsCorrection")
    ],
    (ot.MathTopAccentAttachment, None): [
        ReorderCoverage(
            coverage_attr="TopAccentCoverage", parallel_list_attr="TopAccentAttachment"
        )
    ],
    (ot.MathKernInfo, None): [
        ReorderCoverage(
            coverage_attr="MathKernCoverage", parallel_list_attr="MathKernInfoRecords"
        )
    ],
    (ot.MathVariants, None): [
        ReorderCoverage(
            coverage_attr="VertGlyphCoverage",
            parallel_list_attr="VertGlyphConstruction",
        ),
        ReorderCoverage(
            coverage_attr="HorizGlyphCoverage",
            parallel_list_attr="HorizGlyphConstruction",
        ),
    ],
}


# TODO Port to otTraverse

SubTablePath = Tuple[otBase.BaseTable.SubTableEntry, ...]


def _bfs_base_table(
    root: otBase.BaseTable, root_accessor: str
) -> Iterable[SubTablePath]:
    yield from _traverse_ot_data(
        root, root_accessor, lambda frontier, new: frontier.extend(new)
    )


# Given f(current frontier, new entries) add new entries to frontier
AddToFrontierFn = Callable[[Deque[SubTablePath], List[SubTablePath]], None]


def _traverse_ot_data(
    root: otBase.BaseTable, root_accessor: str, add_to_frontier_fn: AddToFrontierFn
) -> Iterable[SubTablePath]:
    # no visited because general otData is forward-offset only and thus cannot cycle

    frontier: Deque[SubTablePath] = deque()
    frontier.append((otBase.BaseTable.SubTableEntry(root_accessor, root),))
    while frontier:
        # path is (value, attr_name) tuples. attr_name is attr of parent to get value
        path = frontier.popleft()
        current = path[-1].value

        yield path

        new_entries = []
        for subtable_entry in current.iterSubTables():
            new_entries.append(path + (subtable_entry,))

        add_to_frontier_fn(frontier, new_entries)


def reorderGlyphs(font: ttLib.TTFont, new_glyph_order: List[str]):
    old_glyph_order = font.getGlyphOrder()
    if len(new_glyph_order) != len(old_glyph_order):
        raise ValueError(
            f"New glyph order contains {len(new_glyph_order)} glyphs, "
            f"but font has {len(old_glyph_order)} glyphs"
        )

    if set(old_glyph_order) != set(new_glyph_order):
        raise ValueError(
            "New glyph order does not contain the same set of glyphs as the font:\n"
            f"* only in new: {set(new_glyph_order) - set(old_glyph_order)}\n"
            f"* only in old: {set(old_glyph_order) - set(new_glyph_order)}"
        )

    # Changing the order of glyphs in a TTFont requires that all tables that use
    # glyph indexes have been fully.
    # Cf. https://github.com/fonttools/fonttools/issues/2060
    font.ensureDecompiled()
    not_loaded = sorted(t for t in font.keys() if not font.isLoaded(t))
    if not_loaded:
        raise ValueError(f"Everything should be loaded, following aren't: {not_loaded}")

    font.setGlyphOrder(new_glyph_order)

    coverage_containers = {"GDEF", "GPOS", "GSUB", "MATH"}
    for tag in coverage_containers:
        if tag in font.keys():
            for path in _bfs_base_table(font[tag].table, f'font["{tag}"]'):
                value = path[-1].value
                reorder_key = (type(value), getattr(value, "Format", None))
                for reorder in _REORDER_RULES.get(reorder_key, []):
                    reorder.apply(font, value)

    for tag in ["CFF ", "CFF2"]:
        if tag in font:
            cff_table = font[tag]
            charstrings = cff_table.cff.topDictIndex[0].CharStrings.charStrings
            cff_table.cff.topDictIndex[0].charset = new_glyph_order
            cff_table.cff.topDictIndex[0].CharStrings.charStrings = {
                k: charstrings.get(k) for k in new_glyph_order
            }
<|endoftext|>"""ttLib.macUtils.py -- Various Mac-specific stuff."""

from io import BytesIO
from fontTools.misc.macRes import ResourceReader, ResourceError


def getSFNTResIndices(path):
    """Determine whether a file has a 'sfnt' resource fork or not."""
    try:
        reader = ResourceReader(path)
        indices = reader.getIndices("sfnt")
        reader.close()
        return indices
    except ResourceError:
        return []


def openTTFonts(path):
    """Given a pathname, return a list of TTFont objects. In the case
    of a flat TTF/OTF file, the list will contain just one font object;
    but in the case of a Mac font suitcase it will contain as many
    font objects as there are sfnt resources in the file.
    """
    from fontTools import ttLib

    fonts = []
    sfnts = getSFNTResIndices(path)
    if not sfnts:
        fonts.append(ttLib.TTFont(path))
    else:
        for index in sfnts:
            fonts.append(ttLib.TTFont(path, index))
        if not fonts:
            raise ttLib.TTLibError("no fonts found in file '%s'" % path)
    return fonts


class SFNTResourceReader(BytesIO):
    """Simple read-only file wrapper for 'sfnt' resources."""

    def __init__(self, path, res_name_or_index):
        from fontTools import ttLib

        reader = ResourceReader(path)
        if isinstance(res_name_or_index, str):
            rsrc = reader.getNamedResource("sfnt", res_name_or_index)
        else:
            rsrc = reader.getIndResource("sfnt", res_name_or_index)
        if rsrc is None:
            raise ttLib.TTLibError("sfnt resource not found: %s" % res_name_or_index)
        reader.close()
        self.rsrc = rsrc
        super(SFNTResourceReader, self).__init__(rsrc.data)
        self.name = path
<|endoftext|>from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
    TTFont,
    TTLibError,
    getTableModule,
    getTableClass,
    getSearchRange,
)
from fontTools.ttLib.sfnt import (
    SFNTReader,
    SFNTWriter,
    DirectoryEntry,
    WOFFFlavorData,
    sfntDirectoryFormat,
    sfntDirectorySize,
    SFNTDirectoryEntry,
    sfntDirectoryEntrySize,
    calcChecksum,
)
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging

log = logging.getLogger("fontTools.ttLib.woff2")

haveBrotli = False
try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli
    haveBrotli = True
except ImportError:
    pass


class WOFF2Reader(SFNTReader):
    flavor = "woff2"

    def __init__(self, file, checkChecksums=0, fontNumber=-1):
        if not haveBrotli:
            log.error(
                "The WOFF2 decoder requires the Brotli Python extension, available at: "
                "https://github.com/google/brotli"
            )
            raise ImportError("No module named brotli")

        self.file = file

        signature = Tag(self.file.read(4))
        if signature != b"wOF2":
            raise TTLibError("Not a WOFF2 font (bad signature)")

        self.file.seek(0)
        self.DirectoryEntry = WOFF2DirectoryEntry
        data = self.file.read(woff2DirectorySize)
        if len(data) != woff2DirectorySize:
            raise TTLibError("Not a WOFF2 font (not enough data)")
        sstruct.unpack(woff2DirectoryFormat, data, self)

        self.tables = OrderedDict()
        offset = 0
        for i in range(self.numTables):
            entry = self.DirectoryEntry()
            entry.fromFile(self.file)
            tag = Tag(entry.tag)
            self.tables[tag] = entry
            entry.offset = offset
            offset += entry.length

        totalUncompressedSize = offset
        compressedData = self.file.read(self.totalCompressedSize)
        decompressedData = brotli.decompress(compressedData)
        if len(decompressedData) != totalUncompressedSize:
            raise TTLibError(
                "unexpected size for decompressed font data: expected %d, found %d"
                % (totalUncompressedSize, len(decompressedData))
            )
        self.transformBuffer = BytesIO(decompressedData)

        self.file.seek(0, 2)
        if self.length != self.file.tell():
            raise TTLibError("reported 'length' doesn't match the actual file size")

        self.flavorData = WOFF2FlavorData(self)

        # make empty TTFont to store data while reconstructing tables
        self.ttFont = TTFont(recalcBBoxes=False, recalcTimestamp=False)

    def __getitem__(self, tag):
        """Fetch the raw table data. Reconstruct transformed tables."""
        entry = self.tables[Tag(tag)]
        if not hasattr(entry, "data"):
            if entry.transformed:
                entry.data = self.reconstructTable(tag)
            else:
                entry.data = entry.loadData(self.transformBuffer)
        return entry.data

    def reconstructTable(self, tag):
        """Reconstruct table named 'tag' from transformed data."""
        entry = self.tables[Tag(tag)]
        rawData = entry.loadData(self.transformBuffer)
        if tag == "glyf":
            # no need to pad glyph data when reconstructing
            padding = self.padding if hasattr(self, "padding") else None
            data = self._reconstructGlyf(rawData, padding)
        elif tag == "loca":
            data = self._reconstructLoca()
        elif tag == "hmtx":
            data = self._reconstructHmtx(rawData)
        else:
            raise TTLibError("transform for table '%s' is unknown" % tag)
        return data

    def _reconstructGlyf(self, data, padding=None):
        """Return recostructed glyf table data, and set the corresponding loca's
        locations. Optionally pad glyph offsets to the specified number of bytes.
        """
        self.ttFont["loca"] = WOFF2LocaTable()
        glyfTable = self.ttFont["glyf"] = WOFF2GlyfTable()
        glyfTable.reconstruct(data, self.ttFont)
        if padding:
            glyfTable.padding = padding
        data = glyfTable.compile(self.ttFont)
        return data

    def _reconstructLoca(self):
        """Return reconstructed loca table data."""
        if "loca" not in self.ttFont:
            # make sure glyf is reconstructed first
            self.tables["glyf"].data = self.reconstructTable("glyf")
        locaTable = self.ttFont["loca"]
        data = locaTable.compile(self.ttFont)
        if len(data) != self.tables["loca"].origLength:
            raise TTLibError(
                "reconstructed 'loca' table doesn't match original size: "
                "expected %d, found %d" % (self.tables["loca"].origLength, len(data))
            )
        return data

    def _reconstructHmtx(self, data):
        """Return reconstructed hmtx table data."""
        # Before reconstructing 'hmtx' table we need to parse other tables:
        # 'glyf' is required for reconstructing the sidebearings from the glyphs'
        # bounding box; 'hhea' is needed for the numberOfHMetrics field.
        if "glyf" in self.flavorData.transformedTables:
            # transformed 'glyf' table is self-contained, thus 'loca' not needed
            tableDependencies = ("maxp", "hhea", "glyf")
        else:
            # decompiling untransformed 'glyf' requires 'loca', which requires 'head'
            tableDependencies = ("maxp", "head", "hhea", "loca", "glyf")
        for tag in tableDependencies:
            self._decompileTable(tag)
        hmtxTable = self.ttFont["hmtx"] = WOFF2HmtxTable()
        hmtxTable.reconstruct(data, self.ttFont)
        data = hmtxTable.compile(self.ttFont)
        return data

    def _decompileTable(self, tag):
        """Decompile table data and store it inside self.ttFont."""
        data = self[tag]
        if self.ttFont.isLoaded(tag):
            return self.ttFont[tag]
        tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.ttFont.tables[tag] = table
        table.decompile(data, self.ttFont)


class WOFF2Writer(SFNTWriter):
    flavor = "woff2"

    def __init__(
        self,
        file,
        numTables,
        sfntVersion="\000\001\000\000",
        flavor=None,
        flavorData=None,
    ):
        if not haveBrotli:
            log.error(
                "The WOFF2 encoder requires the Brotli Python extension, available at: "
                "https://github.com/google/brotli"
            )
            raise ImportError("No module named brotli")

        self.file = file
        self.numTables = numTables
        self.sfntVersion = Tag(sfntVersion)
        self.flavorData = WOFF2FlavorData(data=flavorData)

        self.directoryFormat = woff2DirectoryFormat
        self.directorySize = woff2DirectorySize
        self.DirectoryEntry = WOFF2DirectoryEntry

        self.signature = Tag("wOF2")

        self.nextTableOffset = 0
        self.transformBuffer = BytesIO()

        self.tables = OrderedDict()

        # make empty TTFont to store data while normalising and transforming tables
        self.ttFont = TTFont(recalcBBoxes=False, recalcTimestamp=False)

    def __setitem__(self, tag, data):
        """Associate new entry named 'tag' with raw table data."""
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)
        if tag == "DSIG":
            # always drop DSIG table, since the encoding process can invalidate it
            self.numTables -= 1
            return

        entry = self.DirectoryEntry()
        entry.tag = Tag(tag)
        entry.flags = getKnownTagIndex(entry.tag)
        # WOFF2 table data are written to disk only on close(), after all tags
        # have been specified
        entry.data = data

        self.tables[tag] = entry

    def close(self):
        """All tags must have been specified. Now write the table data and directory."""
        if len(self.tables) != self.numTables:
            raise TTLibError(
                "wrong number of tables; expected %d, found %d"
                % (self.numTables, len(self.tables))
            )

        if self.sfntVersion in ("\x00\x01\x00\x00", "true"):
            isTrueType = True
        elif self.sfntVersion == "OTTO":
            isTrueType = False
        else:
            raise TTLibError("Not a TrueType or OpenType font (bad sfntVersion)")

        # The WOFF2 spec no longer requires the glyph offsets to be 4-byte aligned.
        # However, the reference WOFF2 implementation still fails to reconstruct
        # 'unpadded' glyf tables, therefore we need to 'normalise' them.
        # See:
        # https://github.com/khaledhosny/ots/issues/60
        # https://github.com/google/woff2/issues/15
        if (
            isTrueType
            and "glyf" in self.flavorData.transformedTables
            and "glyf" in self.tables
        ):
            self._normaliseGlyfAndLoca(padding=4)
        self._setHeadTransformFlag()

        # To pass the legacy OpenType Sanitiser currently included in browsers,
        # we must sort the table directory and data alphabetically by tag.
        # See:
        # https://github.com/google/woff2/pull/3
        # https://lists.w3.org/Archives/Public/public-webfonts-wg/2015Mar/0000.html
        #
        # 2023: We rely on this in _transformTables where we expect that
        # "loca" comes after "glyf" table.
        self.tables = OrderedDict(sorted(self.tables.items()))

        self.totalSfntSize = self._calcSFNTChecksumsLengthsAndOffsets()

        fontData = self._transformTables()
        compressedFont = brotli.compress(fontData, mode=brotli.MODE_FONT)

        self.totalCompressedSize = len(compressedFont)
        self.length = self._calcTotalSize()
        self.majorVersion, self.minorVersion = self._getVersion()
        self.reserved = 0

        directory = self._packTableDirectory()
        self.file.seek(0)
        self.file.write(pad(directory + compressedFont, size=4))
        self._writeFlavorData()

    def _normaliseGlyfAndLoca(self, padding=4):
        """Recompile glyf and loca tables, aligning glyph offsets to multiples of
        'padding' size. Update the head table's 'indexToLocFormat' accordingly while
        compiling loca.
        """
        if self.sfntVersion == "OTTO":
            return

        for tag in ("maxp", "head", "loca", "glyf", "fvar"):
            if tag in self.tables:
                self._decompileTable(tag)
        self.ttFont["glyf"].padding = padding
        for tag in ("glyf", "loca"):
            self._compileTable(tag)

    def _setHeadTransformFlag(self):
        """Set bit 11 of 'head' table flags to indicate that the font has undergone
        a lossless modifying transform. Re-compile head table data."""
        self._decompileTable("head")
        self.ttFont["head"].flags |= 1 << 11
        self._compileTable("head")

    def _decompileTable(self, tag):
        """Fetch table data, decompile it, and store it inside self.ttFont."""
        tag = Tag(tag)
        if tag not in self.tables:
            raise TTLibError("missing required table: %s" % tag)
        if self.ttFont.isLoaded(tag):
            return
        data = self.tables[tag].data
        if tag == "loca":
            tableClass = WOFF2LocaTable
        elif tag == "glyf":
            tableClass = WOFF2GlyfTable
        elif tag == "hmtx":
            tableClass = WOFF2HmtxTable
        else:
            tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.ttFont.tables[tag] = table
        table.decompile(data, self.ttFont)

    def _compileTable(self, tag):
        """Compile table and store it in its 'data' attribute."""
        self.tables[tag].data = self.ttFont[tag].compile(self.ttFont)

    def _calcSFNTChecksumsLengthsAndOffsets(self):
        """Compute the 'original' SFNT checksums, lengths and offsets for checksum
        adjustment calculation. Return the total size of the uncompressed font.
        """
        offset = sfntDirectorySize + sfntDirectoryEntrySize * len(self.tables)
        for tag, entry in self.tables.items():
            data = entry.data
            entry.origOffset = offset
            entry.origLength = len(data)
            if tag == "head":
                entry.checkSum = calcChecksum(data[:8] + b"\0\0\0\0" + data[12:])
            else:
                entry.checkSum = calcChecksum(data)
            offset += (entry.origLength + 3) & ~3
        return offset

    def _transformTables(self):
        """Return transformed font data."""
        transformedTables = self.flavorData.transformedTables
        for tag, entry in self.tables.items():
            data = None
            if tag in transformedTables:
                data = self.transformTable(tag)
                if data is not None:
                    entry.transformed = True
            if data is None:
                if tag == "glyf":
                    # Currently we always sort table tags so
                    # 'loca' comes after 'glyf'.
                    transformedTables.discard("loca")
                # pass-through the table data without transformation
                data = entry.data
                entry.transformed = False
            entry.offset = self.nextTableOffset
            entry.saveData(self.transformBuffer, data)
            self.nextTableOffset += entry.length
        self.writeMasterChecksum()
        fontData = self.transformBuffer.getvalue()
        return fontData

    def transformTable(self, tag):
        """Return transformed table data, or None if some pre-conditions aren't
        met -- in which case, the non-transformed table data will be used.
        """
        if tag == "loca":
            data = b""
        elif tag == "glyf":
            for tag in ("maxp", "head", "loca", "glyf"):
                self._decompileTable(tag)
            glyfTable = self.ttFont["glyf"]
            data = glyfTable.transform(self.ttFont)
        elif tag == "hmtx":
            if "glyf" not in self.tables:
                return
            for tag in ("maxp", "head", "hhea", "loca", "glyf", "hmtx"):
                self._decompileTable(tag)
            hmtxTable = self.ttFont["hmtx"]
            data = hmtxTable.transform(self.ttFont)  # can be None
        else:
            raise TTLibError("Transform for table '%s' is unknown" % tag)
        return data

    def _calcMasterChecksum(self):
        """Calculate checkSumAdjustment."""
        checksums = []
        for tag in self.tables.keys():
            checksums.append(self.tables[tag].checkSum)

        # Create a SFNT directory for checksum calculation purposes
        self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(
            self.numTables, 16
        )
        directory = sstruct.pack(sfntDirectoryFormat, self)
        tables = sorted(self.tables.items())
        for tag, entry in tables:
            sfntEntry = SFNTDirectoryEntry()
            sfntEntry.tag = entry.tag
            sfntEntry.checkSum = entry.checkSum
            sfntEntry.offset = entry.origOffset
            sfntEntry.length = entry.origLength
            directory = directory + sfntEntry.toString()

        directory_end = sfntDirectorySize + len(self.tables) * sfntDirectoryEntrySize
        assert directory_end == len(directory)

        checksums.append(calcChecksum(directory))
        checksum = sum(checksums) & 0xFFFFFFFF
        # BiboAfba!
        checksumadjustment = (0xB1B0AFBA - checksum) & 0xFFFFFFFF
        return checksumadjustment

    def writeMasterChecksum(self):
        """Write checkSumAdjustment to the transformBuffer."""
        checksumadjustment = self._calcMasterChecksum()
        self.transformBuffer.seek(self.tables["head"].offset + 8)
        self.transformBuffer.write(struct.pack(">L", checksumadjustment))

    def _calcTotalSize(self):
        """Calculate total size of WOFF2 font, including any meta- and/or private data."""
        offset = self.directorySize
        for entry in self.tables.values():
            offset += len(entry.toString())
        offset += self.totalCompressedSize
        offset = (offset + 3) & ~3
        offset = self._calcFlavorDataOffsetsAndSize(offset)
        return offset

    def _calcFlavorDataOffsetsAndSize(self, start):
        """Calculate offsets and lengths for any meta- and/or private data."""
        offset = start
        data = self.flavorData
        if data.metaData:
            self.metaOrigLength = len(data.metaData)
            self.metaOffset = offset
            self.compressedMetaData = brotli.compress(
                data.metaData, mode=brotli.MODE_TEXT
            )
            self.metaLength = len(self.compressedMetaData)
            offset += self.metaLength
        else:
            self.metaOffset = self.metaLength = self.metaOrigLength = 0
            self.compressedMetaData = b""
        if data.privData:
            # make sure private data is padded to 4-byte boundary
            offset = (offset + 3) & ~3
            self.privOffset = offset
            self.privLength = len(data.privData)
            offset += self.privLength
        else:
            self.privOffset = self.privLength = 0
        return offset

    def _getVersion(self):
        """Return the WOFF2 font's (majorVersion, minorVersion) tuple."""
        data = self.flavorData
        if data.majorVersion is not None and data.minorVersion is not None:
            return data.majorVersion, data.minorVersion
        else:
            # if None, return 'fontRevision' from 'head' table
            if "head" in self.tables:
                return struct.unpack(">HH", self.tables["head"].data[4:8])
            else:
                return 0, 0

    def _packTableDirectory(self):
        """Return WOFF2 table directory data."""
        directory = sstruct.pack(self.directoryFormat, self)
        for entry in self.tables.values():
            directory = directory + entry.toString()
        return directory

    def _writeFlavorData(self):
        """Write metadata and/or private data using appropiate padding."""
        compressedMetaData = self.compressedMetaData
        privData = self.flavorData.privData
        if compressedMetaData and privData:
            compressedMetaData = pad(compressedMetaData, size=4)
        if compressedMetaData:
            self.file.seek(self.metaOffset)
            assert self.file.tell() == self.metaOffset
            self.file.write(compressedMetaData)
        if privData:
            self.file.seek(self.privOffset)
            assert self.file.tell() == self.privOffset
            self.file.write(privData)

    def reordersTables(self):
        return True


# -- woff2 directory helpers and cruft

woff2DirectoryFormat = """
		> # big endian
		signature:           4s   # "wOF2"
		sfntVersion:         4s
		length:              L    # total woff2 file size
		numTables:           H    # number of tables
		reserved:            H    # set to 0
		totalSfntSize:       L    # uncompressed size
		totalCompressedSize: L    # compressed size
		majorVersion:        H    # major version of WOFF file
		minorVersion:        H    # minor version of WOFF file
		metaOffset:          L    # offset to metadata block
		metaLength:          L    # length of compressed metadata
		metaOrigLength:      L    # length of uncompressed metadata
		privOffset:          L    # offset to private data block
		privLength:          L    # length of private data block
"""

woff2DirectorySize = sstruct.calcsize(woff2DirectoryFormat)

woff2KnownTags = (
    "cmap",
    "head",
    "hhea",
    "hmtx",
    "maxp",
    "name",
    "OS/2",
    "post",
    "cvt ",
    "fpgm",
    "glyf",
    "loca",
    "prep",
    "CFF ",
    "VORG",
    "EBDT",
    "EBLC",
    "gasp",
    "hdmx",
    "kern",
    "LTSH",
    "PCLT",
    "VDMX",
    "vhea",
    "vmtx",
    "BASE",
    "GDEF",
    "GPOS",
    "GSUB",
    "EBSC",
    "JSTF",
    "MATH",
    "CBDT",
    "CBLC",
    "COLR",
    "CPAL",
    "SVG ",
    "sbix",
    "acnt",
    "avar",
    "bdat",
    "bloc",
    "bsln",
    "cvar",
    "fdsc",
    "feat",
    "fmtx",
    "fvar",
    "gvar",
    "hsty",
    "just",
    "lcar",
    "mort",
    "morx",
    "opbd",
    "prop",
    "trak",
    "Zapf",
    "Silf",
    "Glat",
    "Gloc",
    "Feat",
    "Sill",
)

woff2FlagsFormat = """
		> # big endian
