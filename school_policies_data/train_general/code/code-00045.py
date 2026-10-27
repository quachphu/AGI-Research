            elif isinstance(item, VAst.GroupName):
                items.append(self._groupName(item))
            elif isinstance(item, VAst.Enum):
                item = self._coverage(item.enum, flatten=True)
                if flatten:
                    items.extend(item)
                else:
                    items.append(ast.GlyphClass(item))
            elif isinstance(item, VAst.Range):
                item = self._glyphSet(item)
                if flatten:
                    items.extend(item)
                else:
                    items.append(ast.GlyphClass(item))
            else:
                raise NotImplementedError(item)
        return items

    def _context(self, context):
        out = []
        for item in context:
            coverage = self._coverage(item, flatten=True)
            if len(coverage) > 1:
                coverage = ast.GlyphClass(coverage)
            else:
                coverage = coverage[0]
            out.append(coverage)
        return out

    def _groupDefinition(self, group):
        name = self._className(group.name)
        glyphs = self._coverage(group.enum.enum, flatten=True)
        glyphclass = ast.GlyphClass(glyphs)
        classdef = ast.GlyphClassDefinition(name, glyphclass)
        self._glyphclasses[group.name.lower()] = classdef

    def _glyphDefinition(self, glyph):
        try:
            self._glyph_map[glyph.name] = self._glyph_order[glyph.id]
        except TypeError:
            pass

        if glyph.type in ("BASE", "MARK", "LIGATURE", "COMPONENT"):
            if glyph.type not in self._gdef:
                self._gdef[glyph.type] = ast.GlyphClass()
            self._gdef[glyph.type].glyphs.append(self._glyphName(glyph.name))

        if glyph.type == "MARK":
            self._marks.add(glyph.name)
        elif glyph.type == "LIGATURE":
            self._ligatures[glyph.name] = glyph.components

    def _scriptDefinition(self, script):
        stag = script.tag
        for lang in script.langs:
            ltag = lang.tag
            for feature in lang.features:
                lookups = {l.split("\\")[0]: True for l in feature.lookups}
                ftag = feature.tag
                if ftag not in self._features:
                    self._features[ftag] = {}
                if stag not in self._features[ftag]:
                    self._features[ftag][stag] = {}
                assert ltag not in self._features[ftag][stag]
                self._features[ftag][stag][ltag] = lookups.keys()

    def _settingDefinition(self, setting, ignore_unsupported=False):
        if setting.name.startswith("COMPILER_"):
            self._settings[setting.name] = setting.value
        elif not ignore_unsupported:
            log.warning(f"Unsupported setting ignored: {setting.name}")

    def _adjustment(self, adjustment):
        adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by = adjustment

        adv_device = adv_adjust_by and adv_adjust_by.items() or None
        dx_device = dx_adjust_by and dx_adjust_by.items() or None
        dy_device = dy_adjust_by and dy_adjust_by.items() or None

        return ast.ValueRecord(
            xPlacement=dx,
            yPlacement=dy,
            xAdvance=adv,
            xPlaDevice=dx_device,
            yPlaDevice=dy_device,
            xAdvDevice=adv_device,
        )

    def _anchor(self, adjustment):
        adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by = adjustment

        assert not adv_adjust_by
        dx_device = dx_adjust_by and dx_adjust_by.items() or None
        dy_device = dy_adjust_by and dy_adjust_by.items() or None

        return ast.Anchor(
            dx or 0,
            dy or 0,
            xDeviceTable=dx_device or None,
            yDeviceTable=dy_device or None,
        )

    def _anchorDefinition(self, anchordef):
        anchorname = anchordef.name
        glyphname = anchordef.glyph_name
        anchor = self._anchor(anchordef.pos)

        if glyphname not in self._anchors:
            self._anchors[glyphname] = {}
        if anchorname.startswith("MARK_"):
            anchorname = anchorname[:5] + anchorname[5:].lower()
        else:
            anchorname = anchorname.lower()
        if anchorname not in self._anchors[glyphname]:
            self._anchors[glyphname][anchorname] = {}
        self._anchors[glyphname][anchorname][anchordef.component] = anchor

    def _gposLookup(self, lookup, fealookup):
        statements = fealookup.statements

        pos = lookup.pos
        if isinstance(pos, VAst.PositionAdjustPairDefinition):
            for (idx1, idx2), (pos1, pos2) in pos.adjust_pair.items():
                coverage_1 = pos.coverages_1[idx1 - 1]
                coverage_2 = pos.coverages_2[idx2 - 1]

                # If not both are groups, use “enum pos” otherwise makeotf will
                # fail.
                enumerated = False
                for item in coverage_1 + coverage_2:
                    if not isinstance(item, VAst.GroupName):
                        enumerated = True

                glyphs1 = self._coverage(coverage_1)
                glyphs2 = self._coverage(coverage_2)
                record1 = self._adjustment(pos1)
                record2 = self._adjustment(pos2)
                assert len(glyphs1) == 1
                assert len(glyphs2) == 1
                statements.append(
                    ast.PairPosStatement(
                        glyphs1[0], record1, glyphs2[0], record2, enumerated=enumerated
                    )
                )
        elif isinstance(pos, VAst.PositionAdjustSingleDefinition):
            for a, b in pos.adjust_single:
                glyphs = self._coverage(a)
                record = self._adjustment(b)
                assert len(glyphs) == 1
                statements.append(
                    ast.SinglePosStatement([(glyphs[0], record)], [], [], False)
                )
        elif isinstance(pos, VAst.PositionAttachDefinition):
            anchors = {}
            allmarks = set()
            for coverage, anchorname in pos.coverage_to:
                # In feature files mark classes are global, but in VOLT they
                # are defined per-lookup. If we output mark class definitions
                # for all marks that use a given anchor, we might end up with a
                # mark used in two different classes in the same lookup, which
                # is causes feature file compilation error.
                # At the expense of uglier feature code, we make the mark class
                # name by appending the current lookup name not the anchor
                # name, and output mark class definitions only for marks used
                # in this lookup.
                classname = self._className(f"{anchorname}.{lookup.name}")
                markclass = ast.MarkClass(classname)

                # Anchor names are case-insensitive in VOLT
                anchorname = anchorname.lower()

                # We might still end in marks used in two different anchor
                # classes, so we filter out already used marks.
                marks = set()
                for mark in coverage:
                    marks.update(mark.glyphSet())
                if not marks.isdisjoint(allmarks):
                    marks.difference_update(allmarks)
                    if not marks:
                        continue
                allmarks.update(marks)

                for glyphname in marks:
                    glyph = self._glyphName(glyphname)
                    anchor = self._anchors[glyphname][f"MARK_{anchorname}"][1]
                    markdef = ast.MarkClassDefinition(markclass, anchor, glyph)
                    self._markclasses[(glyphname, classname)] = markdef

                for base in pos.coverage:
                    for name in base.glyphSet():
                        if name not in anchors:
                            anchors[name] = []
                        if (anchorname, classname) not in anchors[name]:
                            anchors[name].append((anchorname, classname))

            is_ligature = all(n in self._ligatures for n in anchors)
            is_mark = all(n in self._marks for n in anchors)
            for name in anchors:
                components = 1
                if is_ligature:
                    components = self._ligatures[name]

                marks = [[] for _ in range(components)]
                for mark, classname in anchors[name]:
                    markclass = ast.MarkClass(classname)
                    for component in range(1, components + 1):
                        if component in self._anchors[name][mark]:
                            anchor = self._anchors[name][mark][component]
                            marks[component - 1].append((anchor, markclass))

                base = self._glyphName(name)
                if is_mark:
                    mark = ast.MarkMarkPosStatement(base, marks[0])
                elif is_ligature:
                    mark = ast.MarkLigPosStatement(base, marks)
                else:
                    mark = ast.MarkBasePosStatement(base, marks[0])
                statements.append(mark)
        elif isinstance(pos, VAst.PositionAttachCursiveDefinition):
            # Collect enter and exit glyphs
            enter_coverage = []
            for coverage in pos.coverages_enter:
                for base in coverage:
                    for name in base.glyphSet():
                        enter_coverage.append(name)
            exit_coverage = []
            for coverage in pos.coverages_exit:
                for base in coverage:
                    for name in base.glyphSet():
                        exit_coverage.append(name)

            # Write enter anchors, also check if the glyph has exit anchor and
            # write it, too.
            for name in enter_coverage:
                glyph = self._glyphName(name)
                entry = self._anchors[name]["entry"][1]
                exit = None
                if name in exit_coverage:
                    exit = self._anchors[name]["exit"][1]
                    exit_coverage.pop(exit_coverage.index(name))
                statements.append(ast.CursivePosStatement(glyph, entry, exit))

            # Write any remaining exit anchors.
            for name in exit_coverage:
                glyph = self._glyphName(name)
                exit = self._anchors[name]["exit"][1]
                statements.append(ast.CursivePosStatement(glyph, None, exit))
        else:
            raise NotImplementedError(pos)

    def _gposContextLookup(self, lookup, prefix, suffix, ignore, fealookup, chained):
        statements = fealookup.statements

        pos = lookup.pos
        if isinstance(pos, VAst.PositionAdjustPairDefinition):
            for (idx1, idx2), (pos1, pos2) in pos.adjust_pair.items():
                glyphs1 = self._coverage(pos.coverages_1[idx1 - 1])
                glyphs2 = self._coverage(pos.coverages_2[idx2 - 1])
                assert len(glyphs1) == 1
                assert len(glyphs2) == 1
                glyphs = (glyphs1[0], glyphs2[0])

                if ignore:
                    statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
                else:
                    statement = ast.ChainContextPosStatement(
                        prefix, glyphs, suffix, [chained, chained]
                    )
                statements.append(statement)
        elif isinstance(pos, VAst.PositionAdjustSingleDefinition):
            glyphs = [ast.GlyphClass()]
            for a, _ in pos.adjust_single:
                glyphs[0].extend(self._coverage(a, flatten=True))

            if ignore:
                statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
            else:
                statement = ast.ChainContextPosStatement(
                    prefix, glyphs, suffix, [chained]
                )
            statements.append(statement)
        elif isinstance(pos, VAst.PositionAttachDefinition):
            glyphs = [ast.GlyphClass()]
            for coverage, _ in pos.coverage_to:
                glyphs[0].extend(self._coverage(coverage, flatten=True))

            if ignore:
                statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
            else:
                statement = ast.ChainContextPosStatement(
                    prefix, glyphs, suffix, [chained]
                )
            statements.append(statement)
        else:
            raise NotImplementedError(pos)

    def _gsubLookup(self, lookup, fealookup):
        statements = fealookup.statements

        sub = lookup.sub

        # Alternate substitutions are represented by adding multiple
        # substitutions for the same glyph, so we need to collect them into one
        # to many mapping.
        if isinstance(sub, VAst.SubstitutionAlternateDefinition):
            alternates = {}
            for key, val in sub.mapping.items():
                if not key or not val:
                    path, line, column = sub.location
                    log.warning(f"{path}:{line}:{column}: Ignoring empty substitution")
                    continue
                glyphs = self._coverage(key)
                replacements = self._coverage(val)
                assert len(glyphs) == 1
                for src_glyph, repl_glyph in zip(
                    glyphs[0].glyphSet(), replacements[0].glyphSet()
                ):
                    alternates.setdefault(str(self._glyphName(src_glyph)), []).append(
                        str(self._glyphName(repl_glyph))
                    )

            for glyph, replacements in alternates.items():
                statement = ast.AlternateSubstStatement(
                    [], glyph, [], ast.GlyphClass(replacements)
                )
                statements.append(statement)
            return

        for key, val in sub.mapping.items():
            if not key or not val:
                path, line, column = sub.location
                log.warning(f"{path}:{line}:{column}: Ignoring empty substitution")
                continue
            glyphs = self._coverage(key)
            replacements = self._coverage(val)
            if isinstance(sub, VAst.SubstitutionSingleDefinition):
                assert len(glyphs) == 1
                assert len(replacements) == 1
                statements.append(
                    ast.SingleSubstStatement(glyphs, replacements, [], [], False)
                )
            elif isinstance(sub, VAst.SubstitutionReverseChainingSingleDefinition):
                # This is handled in gsubContextLookup()
                pass
            elif isinstance(sub, VAst.SubstitutionMultipleDefinition):
                assert len(glyphs) == 1
                statements.append(
                    ast.MultipleSubstStatement([], glyphs[0], [], replacements)
                )
            elif isinstance(sub, VAst.SubstitutionLigatureDefinition):
                assert len(replacements) == 1
                statement = ast.LigatureSubstStatement(
                    [], glyphs, [], replacements[0], False
                )

                # If any of the input glyphs is a group, we need to
                # explode the substitution into multiple ligature substitutions
                # since feature file syntax does not support classes in
                # ligature substitutions.
                n = max(len(x.glyphSet()) for x in glyphs)
                if n > 1:
                    # All input should either be groups of the same length or single glyphs
                    assert all(len(x.glyphSet()) in (n, 1) for x in glyphs)
                    glyphs = [x.glyphSet() for x in glyphs]
                    glyphs = [([x[0]] * n if len(x) == 1 else x) for x in glyphs]

                    # In this case ligature replacements must be a group of the same length
                    # as the input groups, or a single glyph. VOLT
                    # allows the replacement glyphs to be longer and truncates them.
                    # So well allow that and zip() below will do the truncation
                    # for us.
                    replacement = replacements[0].glyphSet()
                    if len(replacement) == 1:
                        replacement = [replacement[0]] * n
                    assert len(replacement) >= n

                    # Add the unexploded statement commented out for reference.
                    statements.append(ast.Comment(f"# {statement}"))

                    for zipped in zip(*glyphs, replacement):
                        zipped = [self._glyphName(x) for x in zipped]
                        statements.append(
                            ast.LigatureSubstStatement(
                                [], zipped[:-1], [], zipped[-1], False
                            )
                        )
                else:
                    statements.append(statement)
            else:
                raise NotImplementedError(sub)

    def _gsubContextLookup(self, lookup, prefix, suffix, ignore, fealookup, chained):
        statements = fealookup.statements

        sub = lookup.sub

        if isinstance(sub, VAst.SubstitutionReverseChainingSingleDefinition):
            # Reverse substitutions is a special case, it can’t use chained lookups.
            for key, val in sub.mapping.items():
                if not key or not val:
                    path, line, column = sub.location
                    log.warning(f"{path}:{line}:{column}: Ignoring empty substitution")
                    continue
                glyphs = self._coverage(key)
                replacements = self._coverage(val)
                statements.append(
                    ast.ReverseChainSingleSubstStatement(
                        prefix, suffix, glyphs, replacements
                    )
                )
            fealookup.chained = []
            return

        if not isinstance(
            sub,
            (
                VAst.SubstitutionSingleDefinition,
                VAst.SubstitutionMultipleDefinition,
                VAst.SubstitutionLigatureDefinition,
                VAst.SubstitutionAlternateDefinition,
            ),
        ):
            raise NotImplementedError(type(sub))

        glyphs = []
        for key, val in sub.mapping.items():
            if not key or not val:
                path, line, column = sub.location
                log.warning(f"{path}:{line}:{column}: Ignoring empty substitution")
                continue
            glyphs.extend(self._coverage(key, flatten=True))

        if len(glyphs) > 1:
            glyphs = [ast.GlyphClass(glyphs)]
        if ignore:
            statements.append(ast.IgnoreSubstStatement([(prefix, glyphs, suffix)]))
        else:
            statements.append(
                ast.ChainContextSubstStatement(prefix, glyphs, suffix, [chained])
            )

    def _lookupDefinition(self, lookup):
        mark_attachement = None
        mark_filtering = None

        flags = 0
        if lookup.direction == "RTL":
            flags |= 1
        if not lookup.process_base:
            flags |= 2
        # FIXME: Does VOLT support this?
        # if not lookup.process_ligatures:
        #     flags |= 4
        if not lookup.process_marks:
            flags |= 8
        elif isinstance(lookup.process_marks, str):
            mark_attachement = self._groupName(lookup.process_marks)
        elif lookup.mark_glyph_set is not None:
            mark_filtering = self._groupName(lookup.mark_glyph_set)

        lookupflags = None
        if flags or mark_attachement is not None or mark_filtering is not None:
            lookupflags = ast.LookupFlagStatement(
                flags, mark_attachement, mark_filtering
            )

        use_extension = False
        if self._settings.get("COMPILER_USEEXTENSIONLOOKUPS"):
            use_extension = True

        if "\\" in lookup.name:
            # Merge sub lookups as subtables (lookups named “base\sub”),
            # makeotf/feaLib will issue a warning and ignore the subtable
            # statement if it is not a pairpos lookup, though.
            name = lookup.name.split("\\")[0]
            if name.lower() not in self._lookups:
                fealookup = Lookup(
                    self._lookupName(name),
                    use_extension=use_extension,
                )
                if lookupflags is not None:
                    fealookup.statements.append(lookupflags)
                fealookup.statements.append(ast.Comment("# " + lookup.name))
            else:
                fealookup = self._lookups[name.lower()]
                fealookup.statements.append(ast.SubtableStatement())
                fealookup.statements.append(ast.Comment("# " + lookup.name))
            self._lookups[name.lower()] = fealookup
        else:
            fealookup = Lookup(
                self._lookupName(lookup.name),
                use_extension=use_extension,
            )
            if lookupflags is not None:
                fealookup.statements.append(lookupflags)
            self._lookups[lookup.name.lower()] = fealookup

        if lookup.comments is not None:
            fealookup.statements.append(ast.Comment("# " + lookup.comments))

        contexts = []
        for context in lookup.context:
            prefix = self._context(context.left)
            suffix = self._context(context.right)
            ignore = context.ex_or_in == "EXCEPT_CONTEXT"
            contexts.append([prefix, suffix, ignore])
            # It seems that VOLT will create contextual substitution using
            # only the input if there is no other contexts in this lookup.
            if ignore and len(lookup.context) == 1:
                contexts.append([[], [], False])

        if contexts:
            chained = ast.LookupBlock(
                self._lookupName(lookup.name + " chained"),
                use_extension=use_extension,
            )
            fealookup.chained.append(chained)
            if lookup.sub is not None:
                self._gsubLookup(lookup, chained)
            elif lookup.pos is not None:
                self._gposLookup(lookup, chained)
            for prefix, suffix, ignore in contexts:
                if lookup.sub is not None:
                    self._gsubContextLookup(
                        lookup, prefix, suffix, ignore, fealookup, chained
                    )
                elif lookup.pos is not None:
                    self._gposContextLookup(
                        lookup, prefix, suffix, ignore, fealookup, chained
                    )
        else:
            if lookup.sub is not None:
                self._gsubLookup(lookup, fealookup)
            elif lookup.pos is not None:
                self._gposLookup(lookup, fealookup)


def main(args=None):
    """Convert MS VOLT to AFDKO feature files."""

    import argparse
    from pathlib import Path

    from fontTools import configLogger

    parser = argparse.ArgumentParser(
        "fonttools voltLib.voltToFea", description=main.__doc__
    )
    parser.add_argument(
        "input", metavar="INPUT", type=Path, help="input font/VTP file to process"
    )
    parser.add_argument(
        "featurefile", metavar="OUTPUT", type=Path, help="output feature file"
    )
    parser.add_argument(
        "-t",
        "--table",
        action="append",
        choices=TABLES,
        dest="tables",
        help="List of tables to write, by default all tables are written",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error messages"
    )
    parser.add_argument(
        "--traceback", action="store_true", help="Don’t catch exceptions"
    )

    options = parser.parse_args(args)

    configLogger(level=("ERROR" if options.quiet else "INFO"))

    file_or_path = options.input
    font = None
    try:
        font = TTFont(file_or_path)
        if "TSIV" in font:
            file_or_path = StringIO(font["TSIV"].data.decode("utf-8"))
        else:
            log.error('"TSIV" table is missing, font was not saved from VOLT?')
            return 1
    except TTLibError:
        pass

    converter = VoltToFea(file_or_path, font)
    try:
        fea = converter.convert(options.tables)
    except NotImplementedError as e:
        if options.traceback:
            raise
        location = getattr(e.args[0], "location", None)
        message = f'"{e}" is not supported'
        if location:
            path, line, column = location
            log.error(f"{path}:{line}:{column}: {message}")
        else:
            log.error(message)
        return 1
    with open(options.featurefile, "w") as feafile:
        feafile.write(fea)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple


class Pos(NamedTuple):
    adv: int
    dx: int
    dy: int
    adv_adjust_by: dict
    dx_adjust_by: dict
    dy_adjust_by: dict

    def __str__(self):
        res = " POS"
        for attr in ("adv", "dx", "dy"):
            value = getattr(self, attr)
            if value is not None:
                res += f" {attr.upper()} {value}"
                adjust_by = getattr(self, f"{attr}_adjust_by", {})
                for size, adjustment in adjust_by.items():
                    res += f" ADJUST_BY {adjustment} AT {size}"
        res += " END_POS"
        return res


class Element(object):
    def __init__(self, location=None):
        self.location = location

    def build(self, builder):
        pass

    def __str__(self):
        raise NotImplementedError


class Statement(Element):
    pass


class Expression(Element):
    pass


class VoltFile(Statement):
    def __init__(self):
        Statement.__init__(self, location=None)
        self.statements = []

    def build(self, builder):
        for s in self.statements:
            s.build(builder)

    def __str__(self):
        return "\n" + "\n".join(str(s) for s in self.statements) + " END\n"


class GlyphDefinition(Statement):
    def __init__(self, name, gid, gunicode, gtype, components, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.id = gid
        self.unicode = gunicode
        self.type = gtype
        self.components = components

    def __str__(self):
        res = f'DEF_GLYPH "{self.name}" ID {self.id}'
        if self.unicode is not None:
            if len(self.unicode) > 1:
                unicodes = ",".join(f"U+{u:04X}" for u in self.unicode)
                res += f' UNICODEVALUES "{unicodes}"'
            else:
                res += f" UNICODE {self.unicode[0]}"
        if self.type is not None:
            res += f" TYPE {self.type}"
        if self.components is not None:
            res += f" COMPONENTS {self.components}"
        res += " END_GLYPH"
        return res


class GroupDefinition(Statement):
    def __init__(self, name, enum, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.enum = enum
        self.glyphs_ = None

    def glyphSet(self, groups=None):
        if groups is not None and self.name in groups:
            raise VoltLibError(
                'Group "%s" contains itself.' % (self.name), self.location
            )
        if self.glyphs_ is None:
            if groups is None:
                groups = set({self.name})
            else:
                groups.add(self.name)
            self.glyphs_ = self.enum.glyphSet(groups)
        return self.glyphs_

    def __str__(self):
        enum = self.enum and str(self.enum) or ""
        return f'DEF_GROUP "{self.name}"\n{enum}\nEND_GROUP'


class GlyphName(Expression):
    """A single glyph name, such as cedilla."""

    def __init__(self, glyph, location=None):
        Expression.__init__(self, location)
        self.glyph = glyph

    def glyphSet(self):
        return (self.glyph,)

    def __str__(self):
        return f' GLYPH "{self.glyph}"'


class Enum(Expression):
    """An enum"""

    def __init__(self, enum, location=None):
        Expression.__init__(self, location)
        self.enum = enum

    def __iter__(self):
        for e in self.glyphSet():
            yield e

    def glyphSet(self, groups=None):
        glyphs = []
        for element in self.enum:
            if isinstance(element, (GroupName, Enum)):
                glyphs.extend(element.glyphSet(groups))
            else:
                glyphs.extend(element.glyphSet())
        return tuple(glyphs)

    def __str__(self):
        enum = "".join(str(e) for e in self.enum)
        return f" ENUM{enum} END_ENUM"


class GroupName(Expression):
    """A glyph group"""

    def __init__(self, group, parser, location=None):
        Expression.__init__(self, location)
        self.group = group
        self.parser_ = parser

    def glyphSet(self, groups=None):
        group = self.parser_.resolve_group(self.group)
        if group is not None:
            self.glyphs_ = group.glyphSet(groups)
            return self.glyphs_
        else:
            raise VoltLibError(
                'Group "%s" is used but undefined.' % (self.group), self.location
            )

    def __str__(self):
        return f' GROUP "{self.group}"'


class Range(Expression):
    """A glyph range"""

    def __init__(self, start, end, parser, location=None):
        Expression.__init__(self, location)
        self.start = start
        self.end = end
        self.parser = parser

    def glyphSet(self):
        return tuple(self.parser.glyph_range(self.start, self.end))

    def __str__(self):
        return f' RANGE "{self.start}" TO "{self.end}"'


class ScriptDefinition(Statement):
    def __init__(self, name, tag, langs, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.langs = langs

    def __str__(self):
        res = "DEF_SCRIPT"
        if self.name is not None:
            res += f' NAME "{self.name}"'
        res += f' TAG "{self.tag}"\n\n'
        for lang in self.langs:
            res += f"{lang}"
        res += "END_SCRIPT"
        return res


class LangSysDefinition(Statement):
    def __init__(self, name, tag, features, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.features = features

    def __str__(self):
        res = "DEF_LANGSYS"
        if self.name is not None:
            res += f' NAME "{self.name}"'
        res += f' TAG "{self.tag}"\n\n'
        for feature in self.features:
            res += f"{feature}"
        res += "END_LANGSYS\n"
        return res


class FeatureDefinition(Statement):
    def __init__(self, name, tag, lookups, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.lookups = lookups

    def __str__(self):
        res = f'DEF_FEATURE NAME "{self.name}" TAG "{self.tag}"\n'
        res += " " + " ".join(f'LOOKUP "{l}"' for l in self.lookups) + "\n"
        res += "END_FEATURE\n"
        return res


class LookupDefinition(Statement):
    def __init__(
        self,
        name,
        process_base,
        process_marks,
        mark_glyph_set,
        direction,
        reversal,
        comments,
        context,
        sub,
        pos,
        location=None,
    ):
        Statement.__init__(self, location)
        self.name = name
        self.process_base = process_base
        self.process_marks = process_marks
        self.mark_glyph_set = mark_glyph_set
        self.direction = direction
        self.reversal = reversal
        self.comments = comments
        self.context = context
        self.sub = sub
        self.pos = pos

    def __str__(self):
        res = f'DEF_LOOKUP "{self.name}"'
        res += f' {self.process_base and "PROCESS_BASE" or "SKIP_BASE"}'
        if self.process_marks:
            res += " PROCESS_MARKS "
            if self.mark_glyph_set:
                res += f'MARK_GLYPH_SET "{self.mark_glyph_set}"'
            elif isinstance(self.process_marks, str):
                res += f'"{self.process_marks}"'
            else:
                res += "ALL"
        else:
            res += " SKIP_MARKS"
        if self.direction is not None:
            res += f" DIRECTION {self.direction}"
        if self.reversal:
            res += " REVERSAL"
        if self.comments is not None:
            comments = self.comments.replace("\n", r"\n")
            res += f'\nCOMMENTS "{comments}"'
        if self.context:
            res += "\n" + "\n".join(str(c) for c in self.context)
        else:
            res += "\nIN_CONTEXT\nEND_CONTEXT"
        if self.sub:
            res += f"\n{self.sub}"
        if self.pos:
            res += f"\n{self.pos}"
        return res


class SubstitutionDefinition(Statement):
    def __init__(self, mapping, location=None):
        Statement.__init__(self, location)
        self.mapping = mapping

    def __str__(self):
        res = "AS_SUBSTITUTION\n"
        for src, dst in self.mapping.items():
            src = "".join(str(s) for s in src)
            dst = "".join(str(d) for d in dst)
            res += f"SUB{src}\nWITH{dst}\nEND_SUB\n"
        res += "END_SUBSTITUTION"
        return res


class SubstitutionSingleDefinition(SubstitutionDefinition):
    pass


class SubstitutionMultipleDefinition(SubstitutionDefinition):
    pass


class SubstitutionLigatureDefinition(SubstitutionDefinition):
    pass


class SubstitutionAlternateDefinition(SubstitutionDefinition):
    pass


class SubstitutionReverseChainingSingleDefinition(SubstitutionDefinition):
    pass


class PositionAttachDefinition(Statement):
    def __init__(self, coverage, coverage_to, location=None):
        Statement.__init__(self, location)
        self.coverage = coverage
        self.coverage_to = coverage_to

    def __str__(self):
        coverage = "".join(str(c) for c in self.coverage)
        res = f"AS_POSITION\nATTACH{coverage}\nTO"
        for coverage, anchor in self.coverage_to:
            coverage = "".join(str(c) for c in coverage)
            res += f'{coverage} AT ANCHOR "{anchor}"'
        res += "\nEND_ATTACH\nEND_POSITION"
        return res


class PositionAttachCursiveDefinition(Statement):
    def __init__(self, coverages_exit, coverages_enter, location=None):
        Statement.__init__(self, location)
        self.coverages_exit = coverages_exit
        self.coverages_enter = coverages_enter

    def __str__(self):
        res = "AS_POSITION\nATTACH_CURSIVE"
        for coverage in self.coverages_exit:
            coverage = "".join(str(c) for c in coverage)
            res += f"\nEXIT {coverage}"
        for coverage in self.coverages_enter:
            coverage = "".join(str(c) for c in coverage)
            res += f"\nENTER {coverage}"
        res += "\nEND_ATTACH\nEND_POSITION"
        return res


class PositionAdjustPairDefinition(Statement):
    def __init__(self, coverages_1, coverages_2, adjust_pair, location=None):
        Statement.__init__(self, location)
        self.coverages_1 = coverages_1
        self.coverages_2 = coverages_2
        self.adjust_pair = adjust_pair

    def __str__(self):
        res = "AS_POSITION\nADJUST_PAIR\n"
        for coverage in self.coverages_1:
            coverage = " ".join(str(c) for c in coverage)
            res += f" FIRST {coverage}"
        res += "\n"
        for coverage in self.coverages_2:
            coverage = " ".join(str(c) for c in coverage)
            res += f" SECOND {coverage}"
        res += "\n"
        for (id_1, id_2), (pos_1, pos_2) in self.adjust_pair.items():
            res += f" {id_1} {id_2} BY{pos_1}{pos_2}\n"
        res += "\nEND_ADJUST\nEND_POSITION"
        return res


class PositionAdjustSingleDefinition(Statement):
    def __init__(self, adjust_single, location=None):
        Statement.__init__(self, location)
        self.adjust_single = adjust_single

    def __str__(self):
        res = "AS_POSITION\nADJUST_SINGLE"
        for coverage, pos in self.adjust_single:
            coverage = "".join(str(c) for c in coverage)
            res += f"{coverage} BY{pos}"
        res += "\nEND_ADJUST\nEND_POSITION"
        return res


class ContextDefinition(Statement):
    def __init__(self, ex_or_in, left=None, right=None, location=None):
        Statement.__init__(self, location)
        self.ex_or_in = ex_or_in
        self.left = left if left is not None else []
        self.right = right if right is not None else []

    def __str__(self):
        res = self.ex_or_in + "\n"
        for coverage in self.left:
            coverage = "".join(str(c) for c in coverage)
            res += f" LEFT{coverage}\n"
        for coverage in self.right:
            coverage = "".join(str(c) for c in coverage)
            res += f" RIGHT{coverage}\n"
        res += "END_CONTEXT"
        return res


class AnchorDefinition(Statement):
    def __init__(self, name, gid, glyph_name, component, locked, pos, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.gid = gid
        self.glyph_name = glyph_name
        self.component = component
        self.locked = locked
        self.pos = pos

    def __str__(self):
        locked = self.locked and " LOCKED" or ""
        return (
            f'DEF_ANCHOR "{self.name}"'
            f" ON {self.gid}"
            f" GLYPH {self.glyph_name}"
            f" COMPONENT {self.component}"
            f"{locked}"
            f" AT {self.pos} END_ANCHOR"
        )


class SettingDefinition(Statement):
    def __init__(self, name, value, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.value = value

    def __str__(self):
        if self.value is True:
            return f"{self.name}"
        if isinstance(self.value, (tuple, list)):
            value = " ".join(str(v) for v in self.value)
            return f"{self.name} {value}"
        return f"{self.name} {self.value}"
<|endoftext|>from fontTools.voltLib.error import VoltLibError


class Lexer(object):
    NUMBER = "NUMBER"
    STRING = "STRING"
    NAME = "NAME"
    NEWLINE = "NEWLINE"

    CHAR_WHITESPACE_ = " \t"
    CHAR_NEWLINE_ = "\r\n"
    CHAR_DIGIT_ = "0123456789"
    CHAR_UC_LETTER_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CHAR_LC_LETTER_ = "abcdefghijklmnopqrstuvwxyz"
    CHAR_UNDERSCORE_ = "_"
    CHAR_PERIOD_ = "."
    CHAR_NAME_START_ = (
        CHAR_UC_LETTER_ + CHAR_LC_LETTER_ + CHAR_PERIOD_ + CHAR_UNDERSCORE_
    )
    CHAR_NAME_CONTINUATION_ = CHAR_NAME_START_ + CHAR_DIGIT_

    def __init__(self, text, filename):
        self.filename_ = filename
        self.line_ = 1
        self.pos_ = 0
        self.line_start_ = 0
        self.text_ = text
        self.text_length_ = len(text)

    def __iter__(self):
        return self

    def next(self):  # Python 2
        return self.__next__()

    def __next__(self):  # Python 3
        while True:
            token_type, token, location = self.next_()
            if token_type not in {Lexer.NEWLINE}:
                return (token_type, token, location)

    def location_(self):
        column = self.pos_ - self.line_start_ + 1
        return (self.filename_ or "<volt>", self.line_, column)

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
        if cur_char == '"':
            self.pos_ += 1
            self.scan_until_('"\r\n')
            if self.pos_ < self.text_length_ and self.text_[self.pos_] == '"':
                self.pos_ += 1
                return (Lexer.STRING, text[start + 1 : self.pos_ - 1], location)
            else:
                raise VoltLibError("Expected '\"' to terminate string", location)
        if cur_char in Lexer.CHAR_NAME_START_:
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_NAME_CONTINUATION_)
            token = text[start : self.pos_]
            return (Lexer.NAME, token, location)
        if cur_char in Lexer.CHAR_DIGIT_:
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.NUMBER, int(text[start : self.pos_], 10), location)
        if cur_char == "-" and next_char in Lexer.CHAR_DIGIT_:
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.NUMBER, int(text[start : self.pos_], 10), location)
        raise VoltLibError("Unexpected character: '%s'" % cur_char, location)

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
<|endoftext|>class VoltLibError(Exception):
    def __init__(self, message, location):
        Exception.__init__(self, message)
        self.location = location

    def __str__(self):
        message = Exception.__str__(self)
        if self.location:
            path, line, column = self.location
            return "%s:%d:%d: %s" % (path, line, column, message)
        else:
            return message
<|endoftext|>import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open

PARSE_FUNCS = {
    "DEF_GLYPH": "parse_def_glyph_",
    "DEF_GROUP": "parse_def_group_",
    "DEF_SCRIPT": "parse_def_script_",
    "DEF_LOOKUP": "parse_def_lookup_",
    "DEF_ANCHOR": "parse_def_anchor_",
    "GRID_PPEM": "parse_ppem_",
    "PRESENTATION_PPEM": "parse_ppem_",
    "PPOSITIONING_PPEM": "parse_ppem_",
    "COMPILER_USEEXTENSIONLOOKUPS": "parse_noarg_option_",
    "COMPILER_USEPAIRPOSFORMAT2": "parse_noarg_option_",
    "CMAP_FORMAT": "parse_cmap_format",
    "DO_NOT_TOUCH_CMAP": "parse_noarg_option_",
}


class Parser(object):
    def __init__(self, path):
        self.doc_ = ast.VoltFile()
        self.glyphs_ = OrderedSymbolTable()
        self.groups_ = SymbolTable()
        self.anchors_ = {}  # dictionary of SymbolTable() keyed by glyph
        self.scripts_ = SymbolTable()
        self.langs_ = SymbolTable()
        self.lookups_ = SymbolTable()
        self.next_token_type_, self.next_token_ = (None, None)
        self.next_token_location_ = None
        self.make_lexer_(path)
        self.advance_lexer_()

    def make_lexer_(self, file_or_path):
        if hasattr(file_or_path, "read"):
            filename = getattr(file_or_path, "name", None)
            data = file_or_path.read()
        else:
            filename = file_or_path
            with open(file_or_path, "r") as f:
                data = f.read()
        self.lexer_ = Lexer(data, filename)

    def parse(self):
        statements = self.doc_.statements
        while self.next_token_type_ is not None:
            self.advance_lexer_()
            if self.cur_token_ in PARSE_FUNCS.keys():
                func = getattr(self, PARSE_FUNCS[self.cur_token_])
                statements.append(func())
            elif self.is_cur_keyword_("END"):
                break
            else:
                raise VoltLibError(
                    "Expected " + ", ".join(sorted(PARSE_FUNCS.keys())),
                    self.cur_token_location_,
                )
        return self.doc_

    def parse_def_glyph_(self):
        assert self.is_cur_keyword_("DEF_GLYPH")
        location = self.cur_token_location_
        name = self.expect_string_()
        self.expect_keyword_("ID")
        gid = self.expect_number_()
        if gid < 0:
            raise VoltLibError("Invalid glyph ID", self.cur_token_location_)
        gunicode = None
        if self.next_token_ == "UNICODE":
            self.expect_keyword_("UNICODE")
            gunicode = [self.expect_number_()]
            if gunicode[0] < 0:
                raise VoltLibError("Invalid glyph UNICODE", self.cur_token_location_)
        elif self.next_token_ == "UNICODEVALUES":
            self.expect_keyword_("UNICODEVALUES")
            gunicode = self.parse_unicode_values_()
        gtype = None
        if self.next_token_ == "TYPE":
            self.expect_keyword_("TYPE")
            gtype = self.expect_name_()
            assert gtype in ("BASE", "LIGATURE", "MARK", "COMPONENT")
        components = None
        if self.next_token_ == "COMPONENTS":
            self.expect_keyword_("COMPONENTS")
            components = self.expect_number_()
        self.expect_keyword_("END_GLYPH")
        if self.glyphs_.resolve(name) is not None:
            raise VoltLibError(
                'Glyph "%s" (gid %i) already defined' % (name, gid), location
            )
        def_glyph = ast.GlyphDefinition(
            name, gid, gunicode, gtype, components, location=location
        )
        self.glyphs_.define(name, def_glyph)
        return def_glyph

    def parse_def_group_(self):
        assert self.is_cur_keyword_("DEF_GROUP")
        location = self.cur_token_location_
        name = self.expect_string_()
        enum = None
        if self.next_token_ == "ENUM":
            enum = self.parse_enum_()
        self.expect_keyword_("END_GROUP")
        if self.groups_.resolve(name) is not None:
            raise VoltLibError(
                'Glyph group "%s" already defined, '
                "group names are case insensitive" % name,
                location,
            )
        def_group = ast.GroupDefinition(name, enum, location=location)
        self.groups_.define(name, def_group)
        return def_group

    def parse_def_script_(self):
        assert self.is_cur_keyword_("DEF_SCRIPT")
        location = self.cur_token_location_
        name = None
        if self.next_token_ == "NAME":
            self.expect_keyword_("NAME")
            name = self.expect_string_()
        self.expect_keyword_("TAG")
        tag = self.expect_string_()
        if self.scripts_.resolve(tag) is not None:
            raise VoltLibError(
                'Script "%s" already defined, '
                "script tags are case insensitive" % tag,
                location,
            )
        self.langs_.enter_scope()
        langs = []
        while self.next_token_ != "END_SCRIPT":
            self.advance_lexer_()
            lang = self.parse_langsys_()
            self.expect_keyword_("END_LANGSYS")
            if self.langs_.resolve(lang.tag) is not None:
                raise VoltLibError(
                    'Language "%s" already defined in script "%s", '
                    "language tags are case insensitive" % (lang.tag, tag),
                    location,
                )
            self.langs_.define(lang.tag, lang)
            langs.append(lang)
        self.expect_keyword_("END_SCRIPT")
        self.langs_.exit_scope()
        def_script = ast.ScriptDefinition(name, tag, langs, location=location)
        self.scripts_.define(tag, def_script)
        return def_script

    def parse_langsys_(self):
        assert self.is_cur_keyword_("DEF_LANGSYS")
        location = self.cur_token_location_
        name = None
        if self.next_token_ == "NAME":
            self.expect_keyword_("NAME")
            name = self.expect_string_()
        self.expect_keyword_("TAG")
        tag = self.expect_string_()
        features = []
        while self.next_token_ != "END_LANGSYS":
            self.advance_lexer_()
            feature = self.parse_feature_()
            self.expect_keyword_("END_FEATURE")
            features.append(feature)
        def_langsys = ast.LangSysDefinition(name, tag, features, location=location)
        return def_langsys

    def parse_feature_(self):
        assert self.is_cur_keyword_("DEF_FEATURE")
        location = self.cur_token_location_
        self.expect_keyword_("NAME")
        name = self.expect_string_()
        self.expect_keyword_("TAG")
        tag = self.expect_string_()
        lookups = []
        while self.next_token_ != "END_FEATURE":
            # self.advance_lexer_()
            self.expect_keyword_("LOOKUP")
            lookup = self.expect_string_()
            lookups.append(lookup)
        feature = ast.FeatureDefinition(name, tag, lookups, location=location)
        return feature

    def parse_def_lookup_(self):
        assert self.is_cur_keyword_("DEF_LOOKUP")
        location = self.cur_token_location_
        name = self.expect_string_()
        if not name[0].isalpha():
            raise VoltLibError(
                'Lookup name "%s" must start with a letter' % name, location
            )
        if self.lookups_.resolve(name) is not None:
            raise VoltLibError(
                'Lookup "%s" already defined, '
                "lookup names are case insensitive" % name,
                location,
            )
        process_base = True
        if self.next_token_ == "PROCESS_BASE":
            self.advance_lexer_()
        elif self.next_token_ == "SKIP_BASE":
            self.advance_lexer_()
            process_base = False
        process_marks = True
        mark_glyph_set = None
        if self.next_token_ == "PROCESS_MARKS":
            self.advance_lexer_()
            if self.next_token_ == "MARK_GLYPH_SET":
                self.advance_lexer_()
                mark_glyph_set = self.expect_string_()
            elif self.next_token_ == "ALL":
                self.advance_lexer_()
            elif self.next_token_ == "NONE":
                self.advance_lexer_()
                process_marks = False
            elif self.next_token_type_ == Lexer.STRING:
                process_marks = self.expect_string_()
            else:
                raise VoltLibError(
                    "Expected ALL, NONE, MARK_GLYPH_SET or an ID. "
                    "Got %s" % (self.next_token_type_),
                    location,
                )
        elif self.next_token_ == "SKIP_MARKS":
            self.advance_lexer_()
            process_marks = False
        direction = None
        if self.next_token_ == "DIRECTION":
            self.expect_keyword_("DIRECTION")
            direction = self.expect_name_()
            assert direction in ("LTR", "RTL")
        reversal = None
        if self.next_token_ == "REVERSAL":
            self.expect_keyword_("REVERSAL")
            reversal = True
        comments = None
        if self.next_token_ == "COMMENTS":
            self.expect_keyword_("COMMENTS")
            comments = self.expect_string_().replace(r"\n", "\n")
        context = []
        while self.next_token_ in ("EXCEPT_CONTEXT", "IN_CONTEXT"):
            context = self.parse_context_()
        as_pos_or_sub = self.expect_name_()
        sub = None
        pos = None
        if as_pos_or_sub == "AS_SUBSTITUTION":
            sub = self.parse_substitution_(reversal)
        elif as_pos_or_sub == "AS_POSITION":
            pos = self.parse_position_()
        else:
            raise VoltLibError(
                "Expected AS_SUBSTITUTION or AS_POSITION. " "Got %s" % (as_pos_or_sub),
                location,
            )
        def_lookup = ast.LookupDefinition(
            name,
            process_base,
            process_marks,
            mark_glyph_set,
            direction,
            reversal,
            comments,
            context,
            sub,
            pos,
            location=location,
        )
        self.lookups_.define(name, def_lookup)
        return def_lookup

    def parse_context_(self):
        location = self.cur_token_location_
        contexts = []
        while self.next_token_ in ("EXCEPT_CONTEXT", "IN_CONTEXT"):
            side = None
            coverage = None
            ex_or_in = self.expect_name_()
            # side_contexts = [] # XXX
            if self.next_token_ != "END_CONTEXT":
                left = []
                right = []
                while self.next_token_ in ("LEFT", "RIGHT"):
                    side = self.expect_name_()
                    coverage = self.parse_coverage_()
                    if side == "LEFT":
                        left.append(coverage)
                    else:
                        right.append(coverage)
                self.expect_keyword_("END_CONTEXT")
                context = ast.ContextDefinition(
                    ex_or_in, left, right, location=location
                )
                contexts.append(context)
            else:
                self.expect_keyword_("END_CONTEXT")
        return contexts

    def parse_substitution_(self, reversal):
        assert self.is_cur_keyword_("AS_SUBSTITUTION")
        location = self.cur_token_location_
        src = []
        dest = []
        if self.next_token_ != "SUB":
            raise VoltLibError("Expected SUB", location)
        while self.next_token_ == "SUB":
            self.expect_keyword_("SUB")
            src.append(self.parse_coverage_())
            self.expect_keyword_("WITH")
            dest.append(self.parse_coverage_())
            self.expect_keyword_("END_SUB")
        self.expect_keyword_("END_SUBSTITUTION")
        max_src = max([len(cov) for cov in src])
        max_dest = max([len(cov) for cov in dest])

        # many to many or mixed is invalid
        if max_src > 1 and max_dest > 1:
            raise VoltLibError("Invalid substitution type", location)

        mapping = dict(zip(tuple(src), tuple(dest)))
        if max_src == 1 and max_dest == 1:
            # Alternate substitutions are represented by adding multiple
            # substitutions for the same glyph, so we detect that here
            glyphs = [x.glyphSet() for cov in src for x in cov]  # flatten src
            if len(set(glyphs)) != len(glyphs):  # src has duplicates
                sub = ast.SubstitutionAlternateDefinition(mapping, location=location)
            else:
                if reversal:
                    # Reversal is valid only for single glyph substitutions
                    # and VOLT ignores it otherwise.
                    sub = ast.SubstitutionReverseChainingSingleDefinition(
                        mapping, location=location
                    )
                else:
                    sub = ast.SubstitutionSingleDefinition(mapping, location=location)
        elif max_src == 1 and max_dest > 1:
            sub = ast.SubstitutionMultipleDefinition(mapping, location=location)
        elif max_src > 1 and max_dest == 1:
            sub = ast.SubstitutionLigatureDefinition(mapping, location=location)
        return sub

    def parse_position_(self):
        assert self.is_cur_keyword_("AS_POSITION")
        location = self.cur_token_location_
        pos_type = self.expect_name_()
        if pos_type not in ("ATTACH", "ATTACH_CURSIVE", "ADJUST_PAIR", "ADJUST_SINGLE"):
            raise VoltLibError(
                "Expected ATTACH, ATTACH_CURSIVE, ADJUST_PAIR, ADJUST_SINGLE", location
            )
        if pos_type == "ATTACH":
            position = self.parse_attach_()
        elif pos_type == "ATTACH_CURSIVE":
            position = self.parse_attach_cursive_()
        elif pos_type == "ADJUST_PAIR":
            position = self.parse_adjust_pair_()
        elif pos_type == "ADJUST_SINGLE":
            position = self.parse_adjust_single_()
        self.expect_keyword_("END_POSITION")
        return position

    def parse_attach_(self):
        assert self.is_cur_keyword_("ATTACH")
        location = self.cur_token_location_
        coverage = self.parse_coverage_()
        coverage_to = []
        self.expect_keyword_("TO")
        while self.next_token_ != "END_ATTACH":
            cov = self.parse_coverage_()
            self.expect_keyword_("AT")
            self.expect_keyword_("ANCHOR")
            anchor_name = self.expect_string_()
            coverage_to.append((cov, anchor_name))
        self.expect_keyword_("END_ATTACH")
        position = ast.PositionAttachDefinition(
            coverage, coverage_to, location=location
        )
        return position

    def parse_attach_cursive_(self):
        assert self.is_cur_keyword_("ATTACH_CURSIVE")
        location = self.cur_token_location_
        coverages_exit = []
        coverages_enter = []
        while self.next_token_ != "ENTER":
            self.expect_keyword_("EXIT")
            coverages_exit.append(self.parse_coverage_())
        while self.next_token_ != "END_ATTACH":
            self.expect_keyword_("ENTER")
            coverages_enter.append(self.parse_coverage_())
        self.expect_keyword_("END_ATTACH")
        position = ast.PositionAttachCursiveDefinition(
            coverages_exit, coverages_enter, location=location
        )
        return position

    def parse_adjust_pair_(self):
        assert self.is_cur_keyword_("ADJUST_PAIR")
        location = self.cur_token_location_
        coverages_1 = []
        coverages_2 = []
        adjust_pair = {}
        while self.next_token_ == "FIRST":
            self.advance_lexer_()
            coverage_1 = self.parse_coverage_()
            coverages_1.append(coverage_1)
        while self.next_token_ == "SECOND":
            self.advance_lexer_()
            coverage_2 = self.parse_coverage_()
            coverages_2.append(coverage_2)
        while self.next_token_ != "END_ADJUST":
            id_1 = self.expect_number_()
            id_2 = self.expect_number_()
            self.expect_keyword_("BY")
            pos_1 = self.parse_pos_()
            pos_2 = self.parse_pos_()
            adjust_pair[(id_1, id_2)] = (pos_1, pos_2)
        self.expect_keyword_("END_ADJUST")
        position = ast.PositionAdjustPairDefinition(
            coverages_1, coverages_2, adjust_pair, location=location
        )
        return position

    def parse_adjust_single_(self):
        assert self.is_cur_keyword_("ADJUST_SINGLE")
        location = self.cur_token_location_
        adjust_single = []
        while self.next_token_ != "END_ADJUST":
            coverages = self.parse_coverage_()
            self.expect_keyword_("BY")
            pos = self.parse_pos_()
            adjust_single.append((coverages, pos))
        self.expect_keyword_("END_ADJUST")
        position = ast.PositionAdjustSingleDefinition(adjust_single, location=location)
        return position

    def parse_def_anchor_(self):
        assert self.is_cur_keyword_("DEF_ANCHOR")
        location = self.cur_token_location_
        name = self.expect_string_()
        self.expect_keyword_("ON")
        gid = self.expect_number_()
        self.expect_keyword_("GLYPH")
        glyph_name = self.expect_name_()
        self.expect_keyword_("COMPONENT")
        component = self.expect_number_()
        # check for duplicate anchor names on this glyph
        if glyph_name in self.anchors_:
            anchor = self.anchors_[glyph_name].resolve(name)
            if anchor is not None and anchor.component == component:
                raise VoltLibError(
                    'Anchor "%s" already defined, '
                    "anchor names are case insensitive" % name,
                    location,
                )
        if self.next_token_ == "LOCKED":
            locked = True
            self.advance_lexer_()
        else:
            locked = False
        self.expect_keyword_("AT")
        pos = self.parse_pos_()
        self.expect_keyword_("END_ANCHOR")
        anchor = ast.AnchorDefinition(
            name, gid, glyph_name, component, locked, pos, location=location
        )
        if glyph_name not in self.anchors_:
            self.anchors_[glyph_name] = SymbolTable()
        self.anchors_[glyph_name].define(name, anchor)
        return anchor

    def parse_adjust_by_(self):
        self.advance_lexer_()
        assert self.is_cur_keyword_("ADJUST_BY")
        adjustment = self.expect_number_()
        self.expect_keyword_("AT")
        size = self.expect_number_()
        return adjustment, size

    def parse_pos_(self):
        # VOLT syntax doesn't seem to take device Y advance
        self.advance_lexer_()
        location = self.cur_token_location_
        assert self.is_cur_keyword_("POS"), location
        adv = None
        dx = None
        dy = None
        adv_adjust_by = {}
        dx_adjust_by = {}
        dy_adjust_by = {}
        if self.next_token_ == "ADV":
            self.advance_lexer_()
            adv = self.expect_number_()
            while self.next_token_ == "ADJUST_BY":
                adjustment, size = self.parse_adjust_by_()
                adv_adjust_by[size] = adjustment
        if self.next_token_ == "DX":
            self.advance_lexer_()
            dx = self.expect_number_()
            while self.next_token_ == "ADJUST_BY":
                adjustment, size = self.parse_adjust_by_()
                dx_adjust_by[size] = adjustment
        if self.next_token_ == "DY":
            self.advance_lexer_()
            dy = self.expect_number_()
            while self.next_token_ == "ADJUST_BY":
                adjustment, size = self.parse_adjust_by_()
                dy_adjust_by[size] = adjustment
        self.expect_keyword_("END_POS")
        return ast.Pos(adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by)

    def parse_unicode_values_(self):
        location = self.cur_token_location_
        try:
            unicode_values = self.expect_string_().split(",")
            unicode_values = [int(uni[2:], 16) for uni in unicode_values if uni != ""]
        except ValueError as err:
            raise VoltLibError(str(err), location)
        return unicode_values if unicode_values != [] else None

    def parse_enum_(self):
        self.expect_keyword_("ENUM")
        location = self.cur_token_location_
        enum = ast.Enum(self.parse_coverage_(), location=location)
        self.expect_keyword_("END_ENUM")
        return enum

    def parse_coverage_(self):
        coverage = []
        location = self.cur_token_location_
        while self.next_token_ in ("GLYPH", "GROUP", "RANGE", "ENUM"):
            if self.next_token_ == "ENUM":
                enum = self.parse_enum_()
                coverage.append(enum)
            elif self.next_token_ == "GLYPH":
                self.expect_keyword_("GLYPH")
                name = self.expect_string_()
                coverage.append(ast.GlyphName(name, location=location))
            elif self.next_token_ == "GROUP":
                self.expect_keyword_("GROUP")
                name = self.expect_string_()
                coverage.append(ast.GroupName(name, self, location=location))
            elif self.next_token_ == "RANGE":
                self.expect_keyword_("RANGE")
                start = self.expect_string_()
                self.expect_keyword_("TO")
                end = self.expect_string_()
                coverage.append(ast.Range(start, end, self, location=location))
        return tuple(coverage)

    def resolve_group(self, group_name):
        return self.groups_.resolve(group_name)

    def glyph_range(self, start, end):
        return self.glyphs_.range(start, end)

    def parse_ppem_(self):
        location = self.cur_token_location_
        ppem_name = self.cur_token_
        value = self.expect_number_()
        setting = ast.SettingDefinition(ppem_name, value, location=location)
        return setting

    def parse_noarg_option_(self):
        location = self.cur_token_location_
        name = self.cur_token_
        value = True
        setting = ast.SettingDefinition(name, value, location=location)
        return setting

    def parse_cmap_format(self):
        location = self.cur_token_location_
        name = self.cur_token_
        value = (self.expect_number_(), self.expect_number_(), self.expect_number_())
        setting = ast.SettingDefinition(name, value, location=location)
        return setting

    def is_cur_keyword_(self, k):
        return (self.cur_token_type_ is Lexer.NAME) and (self.cur_token_ == k)

    def expect_string_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.STRING:
            raise VoltLibError("Expected a string", self.cur_token_location_)
        return self.cur_token_

    def expect_keyword_(self, keyword):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NAME and self.cur_token_ == keyword:
            return self.cur_token_
        raise VoltLibError('Expected "%s"' % keyword, self.cur_token_location_)

    def expect_name_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NAME:
            return self.cur_token_
        raise VoltLibError("Expected a name", self.cur_token_location_)

    def expect_number_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.NUMBER:
            raise VoltLibError("Expected a number", self.cur_token_location_)
        return self.cur_token_

    def advance_lexer_(self):
        self.cur_token_type_, self.cur_token_, self.cur_token_location_ = (
            self.next_token_type_,
            self.next_token_,
            self.next_token_location_,
        )
        try:
            if self.is_cur_keyword_("END"):
                raise StopIteration
            (
                self.next_token_type_,
                self.next_token_,
                self.next_token_location_,
            ) = self.lexer_.next()
        except StopIteration:
            self.next_token_type_, self.next_token_ = (None, None)


class SymbolTable(object):
    def __init__(self):
        self.scopes_ = [{}]

    def enter_scope(self):
        self.scopes_.append({})

    def exit_scope(self):
        self.scopes_.pop()

    def define(self, name, item):
        self.scopes_[-1][name] = item

    def resolve(self, name, case_insensitive=True):
        for scope in reversed(self.scopes_):
            item = scope.get(name)
            if item:
                return item
        if case_insensitive:
            for key in scope:
                if key.lower() == name.lower():
                    return scope[key]
        return None


class OrderedSymbolTable(SymbolTable):
    def __init__(self):
        self.scopes_ = [{}]

    def enter_scope(self):
        self.scopes_.append({})

    def resolve(self, name, case_insensitive=False):
        SymbolTable.resolve(self, name, case_insensitive=case_insensitive)

    def range(self, start, end):
        for scope in reversed(self.scopes_):
            if start in scope and end in scope:
                start_idx = list(scope.keys()).index(start)
                end_idx = list(scope.keys()).index(end)
                return list(scope.keys())[start_idx : end_idx + 1]
        return None
<|endoftext|>import argparse
import logging
import sys
from io import StringIO
from pathlib import Path

from fontTools import configLogger
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib.parser import Parser
from fontTools.voltLib.voltToFea import TABLES, VoltToFea

log = logging.getLogger("fontTools.feaLib")

SUPPORTED_TABLES = TABLES + ["cmap"]


def invalid_fea_glyph_name(name):
    """Check if the glyph name is valid according to FEA syntax."""
    if name[0] not in Lexer.CHAR_NAME_START_:
        return True
    if any(c not in Lexer.CHAR_NAME_CONTINUATION_ for c in name[1:]):
        return True
    return False


def sanitize_glyph_name(name):
    """Sanitize the glyph name to ensure it is valid according to FEA syntax."""
    sanitized = ""
    for i, c in enumerate(name):
        if i == 0 and c not in Lexer.CHAR_NAME_START_:
            sanitized += "a" + c
        elif c not in Lexer.CHAR_NAME_CONTINUATION_:
            sanitized += "_"
        else:
            sanitized += c

    return sanitized


def main(args=None):
    """Build tables from a MS VOLT project into an OTF font"""
    parser = argparse.ArgumentParser(
        description="Use fontTools to compile MS VOLT projects."
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to the input font/VTP file to process",
        type=Path,
    )
    parser.add_argument(
        "-f",
        "--font",
        metavar="INPUT_FONT",
        help="Path to the input font (if INPUT is a VTP file)",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        metavar="OUTPUT",
        help="Path to the output font.",
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--tables",
        metavar="TABLE_TAG",
        choices=SUPPORTED_TABLES,
        nargs="+",
        help="Specify the table(s) to be built.",
    )
    parser.add_argument(
        "-F",
        "--debug-feature-file",
        help="Write the generated feature file to disk.",
        action="store_true",
    )
    parser.add_argument(
        "--ship",
        help="Remove source VOLT tables from output font.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase the logger verbosity. Multiple -v options are allowed.",
        action="count",
        default=0,
    )
    parser.add_argument(
        "-T",
        "--traceback",
        help="show traceback for exceptions.",
        action="store_true",
    )
    options = parser.parse_args(args)

    levels = ["WARNING", "INFO", "DEBUG"]
    configLogger(level=levels[min(len(levels) - 1, options.verbose)])

    output_font = options.output or Path(
        makeOutputFileName(options.font or options.input)
    )
    log.info(f"Compiling MS VOLT to '{output_font}'")

    file_or_path = options.input
    font = None

    # If the input is a font file, extract the VOLT data from the "TSIV" table
    try:
        font = TTFont(file_or_path)
        if "TSIV" in font:
            file_or_path = StringIO(font["TSIV"].data.decode("utf-8"))
        else:
            log.error('"TSIV" table is missing')
            return 1
    except TTLibError:
        pass

    # If input is not a font file, the font must be provided
    if font is None:
        if not options.font:
            log.error("Please provide an input font")
            return 1
        font = TTFont(options.font)

    # FEA syntax does not allow some glyph names that VOLT accepts, so if we
    # found such glyph name we will temporarily rename such glyphs.
    glyphOrder = font.getGlyphOrder()
    tempGlyphOrder = None
    if any(invalid_fea_glyph_name(n) for n in glyphOrder):
        tempGlyphOrder = []
        for n in glyphOrder:
            if invalid_fea_glyph_name(n):
                n = sanitize_glyph_name(n)
                existing = set(tempGlyphOrder) | set(glyphOrder)
                while n in existing:
                    n = "a" + n
            tempGlyphOrder.append(n)
        font.setGlyphOrder(tempGlyphOrder)

    doc = Parser(file_or_path).parse()

    log.info("Converting VTP data to FEA")
    converter = VoltToFea(doc, font)
    try:
        fea = converter.convert(options.tables, ignore_unsupported_settings=True)
    except NotImplementedError as e:
        if options.traceback:
            raise
        location = getattr(e.args[0], "location", None)
        message = f'"{e}" is not supported'
        if location:
            path, line, column = location
            log.error(f"{path}:{line}:{column}: {message}")
        else:
            log.error(message)
        return 1

    fea_filename = options.input
    if options.debug_feature_file:
        fea_filename = output_font.with_suffix(".fea")
        log.info(f"Writing FEA to '{fea_filename}'")
        with open(fea_filename, "w") as fp:
            fp.write(fea)

    log.info("Compiling FEA to OpenType tables")
    try:
        addOpenTypeFeaturesFromString(
            font,
            fea,
            filename=fea_filename,
            tables=options.tables,
        )
    except FeatureLibError as e:
        if options.traceback:
            raise
        log.error(e)
        return 1

    if options.ship:
        for tag in ["TSIV", "TSIS", "TSIP", "TSID"]:
            if tag in font:
                del font[tag]

    # Restore original glyph names.
    if tempGlyphOrder:
        import io

        f = io.BytesIO()
        font.save(f)
        font = TTFont(f)
        font.setGlyphOrder(glyphOrder)
        font["post"].extraNames = []

    font.save(output_font)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>from __future__ import annotations

from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
    LookupDebugInfo,
    LOOKUP_DEBUG_INFO_KEY,
    LOOKUP_DEBUG_ENV_VAR,
)
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar, VariableScalarBuilder
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
    AlternateSubstBuilder,
    ChainContextPosBuilder,
    ChainContextSubstBuilder,
    LigatureSubstBuilder,
    MultipleSubstBuilder,
    CursivePosBuilder,
    MarkBasePosBuilder,
    MarkLigPosBuilder,
    MarkMarkPosBuilder,
    ReverseChainSingleSubstBuilder,
    SingleSubstBuilder,
    ClassPairPosSubtableBuilder,
    PairPosBuilder,
    SinglePosBuilder,
    ChainContextualRule,
    AnySubstBuilder,
)
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.errors import VarLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os

log = logging.getLogger(__name__)


def addOpenTypeFeatures(font, featurefile, tables=None, debug=False):
    """Add features from a file to a font. Note that this replaces any features
    currently present.

    Args:
        font (feaLib.ttLib.TTFont): The font object.
        featurefile: Either a path or file object (in which case we
            parse it into an AST), or a pre-parsed AST instance.
        tables: If passed, restrict the set of affected tables to those in the
            list.
        debug: Whether to add source debugging information to the font in the
            ``Debg`` table

    """
    builder = Builder(font, featurefile)
    builder.build(tables=tables, debug=debug)


def addOpenTypeFeaturesFromString(
    font, features, filename=None, tables=None, debug=False
):
    """Add features from a string to a font. Note that this replaces any
    features currently present.

    Args:
        font (feaLib.ttLib.TTFont): The font object.
        features: A string containing feature code.
        filename: The directory containing ``filename`` is used as the root of
            relative ``include()`` paths; if ``None`` is provided, the current
            directory is assumed.
        tables: If passed, restrict the set of affected tables to those in the
            list.
        debug: Whether to add source debugging information to the font in the
            ``Debg`` table

    """

    featurefile = StringIO(tostr(features))
    if filename:
        featurefile.name = filename
    addOpenTypeFeatures(font, featurefile, tables=tables, debug=debug)


class Builder(object):
    supportedTables = frozenset(
        Tag(tag)
        for tag in [
            "BASE",
            "GDEF",
            "GPOS",
            "GSUB",
            "OS/2",
            "head",
            "hhea",
            "name",
            "vhea",
            "STAT",
        ]
    )

    def __init__(self, font, featurefile):
        self.font = font
        # 'featurefile' can be either a path or file object (in which case we
        # parse it into an AST), or a pre-parsed AST instance
        if isinstance(featurefile, FeatureFile):
            self.parseTree, self.file = featurefile, None
        else:
            self.parseTree, self.file = None, featurefile
        self.glyphMap = font.getReverseGlyphMap()
        self.varstorebuilder = None
        if "fvar" in font:
            self.axes = font["fvar"].axes
            self.varstorebuilder = OnlineVarStoreBuilder(
                [ax.axisTag for ax in self.axes]
            )
            self.scalar_builder = VariableScalarBuilder.from_ttf(font)
        self.default_language_systems_ = set()
        self.script_ = None
        self.language_ = None
        self.lookupflag_ = 0
        self.lookupflag_markFilterSet_ = None
        self.use_extension_ = False
        self.language_systems = set()
        self.seen_non_DFLT_script_ = False
        self.named_lookups_ = {}
        self.cur_lookup_ = None
        self.cur_lookup_name_ = None
        self.cur_feature_name_ = None
        self.lookups_ = []
        self.lookup_locations = {"GSUB": {}, "GPOS": {}}
        self.features_ = {}  # ('latn', 'DEU ', 'smcp') --> [LookupBuilder*]
        self.required_features_ = {}  # ('latn', 'DEU ') --> 'scmp'
        self.feature_variations_ = {}
        # for feature 'aalt'
        self.aalt_features_ = []  # [(location, featureName)*], for 'aalt'
        self.aalt_location_ = None
        self.aalt_alternates_ = {}
        self.aalt_use_extension_ = False
        # for 'featureNames'
        self.featureNames_ = set()
        self.featureNames_ids_ = {}
        # for 'cvParameters'
        self.cv_parameters_ = set()
        self.cv_parameters_ids_ = {}
        self.cv_num_named_params_ = {}
        self.cv_characters_ = defaultdict(list)
        # for feature 'size'
        self.size_parameters_ = None
        # for table 'head'
        self.fontRevision_ = None  # 2.71
        # for table 'name'
        self.names_ = []
        # for table 'BASE'
        self.base_horiz_axis_ = None
        self.base_vert_axis_ = None
        # for table 'GDEF'
        self.attachPoints_ = {}  # "a" --> {3, 7}
        self.ligCaretCoords_ = {}  # "f_f_i" --> {300, 600}
        self.ligCaretPoints_ = {}  # "f_f_i" --> {3, 7}
        self.glyphClassDefs_ = {}  # "fi" --> (2, (file, line, column))
        self.markAttach_ = {}  # "acute" --> (4, (file, line, column))
        self.markAttachClassID_ = {}  # frozenset({"acute", "grave"}) --> 4
        self.markFilterSets_ = {}  # frozenset({"acute", "grave"}) --> 4
        # for table 'OS/2'
        self.os2_ = {}
        # for table 'hhea'
        self.hhea_ = {}
        # for table 'vhea'
        self.vhea_ = {}
        # for table 'STAT'
        self.stat_ = {}
        # for conditionsets
        self.conditionsets_ = {}

    def build(self, tables=None, debug=False):
        if self.parseTree is None:
            self.parseTree = Parser(self.file, self.glyphMap).parse()
        self.parseTree.build(self)
        # by default, build all the supported tables
        if tables is None:
            tables = self.supportedTables
        else:
            tables = frozenset(tables)
            unsupported = tables - self.supportedTables
            if unsupported:
                unsupported_string = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    "The following tables were requested but are unsupported: "
                    f"{unsupported_string}."
                )
        if "GSUB" in tables:
            self.build_feature_aalt_()
        if "head" in tables:
            self.build_head()
        if "hhea" in tables:
            self.build_hhea()
        if "vhea" in tables:
            self.build_vhea()
        if "name" in tables:
            self.build_name()
        if "OS/2" in tables:
            self.build_OS_2()
        if "STAT" in tables:
            self.build_STAT()
        for tag in ("GPOS", "GSUB"):
            if tag not in tables:
                continue
            table = self.makeTable(tag)
            if self.feature_variations_:
                self.makeFeatureVariations(table, tag)
            if (
                table.ScriptList.ScriptCount > 0
                or table.FeatureList.FeatureCount > 0
                or table.LookupList.LookupCount > 0
            ):
                fontTable = self.font[tag] = newTable(tag)
                fontTable.table = table
            elif tag in self.font:
                del self.font[tag]
        if any(tag in self.font for tag in ("GPOS", "GSUB")) and "OS/2" in self.font:
            self.font["OS/2"].usMaxContext = maxCtxFont(self.font)
        if "GDEF" in tables:
            gdef = self.buildGDEF()
            if gdef:
                self.font["GDEF"] = gdef
            elif "GDEF" in self.font:
                del self.font["GDEF"]
        if "BASE" in tables:
            base = self.buildBASE()
            if base:
                self.font["BASE"] = base
            elif "BASE" in self.font:
                del self.font["BASE"]
        if debug or os.environ.get(LOOKUP_DEBUG_ENV_VAR):
            self.buildDebg()

    def get_chained_lookup_(self, location, builder_class):
        result = builder_class(self.font, location)
        result.lookupflag = self.lookupflag_
        result.markFilterSet = self.lookupflag_markFilterSet_
        result.extension = self.use_extension_
        self.lookups_.append(result)
        return result

    def add_lookup_to_feature_(self, lookup, feature_name):
        for script, lang in self.language_systems:
            key = (script, lang, feature_name)
            self.features_.setdefault(key, []).append(lookup)

    def get_lookup_(self, location, builder_class, mapping=None):
        if (
            self.cur_lookup_
            and type(self.cur_lookup_) == builder_class
            and self.cur_lookup_.lookupflag == self.lookupflag_
            and self.cur_lookup_.markFilterSet == self.lookupflag_markFilterSet_
            and self.cur_lookup_.can_add_mapping(mapping)
        ):
            return self.cur_lookup_
        if self.cur_lookup_name_ and self.cur_lookup_:
            raise FeatureLibError(
                "Within a named lookup block, all rules must be of "
                "the same lookup type and flag",
                location,
            )
        self.cur_lookup_ = builder_class(self.font, location)
        self.cur_lookup_.lookupflag = self.lookupflag_
        self.cur_lookup_.markFilterSet = self.lookupflag_markFilterSet_
        self.cur_lookup_.extension = self.use_extension_
        self.lookups_.append(self.cur_lookup_)
        if self.cur_lookup_name_:
            # We are starting a lookup rule inside a named lookup block.
            self.named_lookups_[self.cur_lookup_name_] = self.cur_lookup_
        if self.cur_feature_name_:
            # We are starting a lookup rule inside a feature. This includes
            # lookup rules inside named lookups inside features.
            self.add_lookup_to_feature_(self.cur_lookup_, self.cur_feature_name_)
        return self.cur_lookup_

    def build_feature_aalt_(self):
        if not self.aalt_features_ and not self.aalt_alternates_:
            return
        # > alternate glyphs will be sorted in the order that the source features
        # > are named in the aalt definition, not the order of the feature definitions
        # > in the file. Alternates defined explicitly ... will precede all others.
        # https://github.com/fonttools/fonttools/issues/836
        alternates = {g: list(a) for g, a in self.aalt_alternates_.items()}
        for location, name in self.aalt_features_ + [(None, "aalt")]:
            feature = [
                (script, lang, feature, lookups)
                for (script, lang, feature), lookups in self.features_.items()
                if feature == name
            ]
            # "aalt" does not have to specify its own lookups, but it might.
            if not feature and name != "aalt":
                warnings.warn("%s: Feature %s has not been defined" % (location, name))
                continue
            for script, lang, feature, lookups in feature:
                for lookuplist in lookups:
                    if not isinstance(lookuplist, list):
                        lookuplist = [lookuplist]
                    for lookup in lookuplist:
                        for glyph, alts in lookup.getAlternateGlyphs().items():
                            alts_for_glyph = alternates.setdefault(glyph, [])
                            alts_for_glyph.extend(
                                g for g in alts if g not in alts_for_glyph
                            )
        single = {
            glyph: repl[0] for glyph, repl in alternates.items() if len(repl) == 1
        }
        multi = {glyph: repl for glyph, repl in alternates.items() if len(repl) > 1}
        if not single and not multi:
            return
        self.features_ = {
            (script, lang, feature): lookups
            for (script, lang, feature), lookups in self.features_.items()
            if feature != "aalt"
        }
        old_lookups = self.lookups_
        self.lookups_ = []
        self.start_feature(self.aalt_location_, "aalt", self.aalt_use_extension_)
        if single:
            single_lookup = self.get_lookup_(location, SingleSubstBuilder)
            single_lookup.mapping = single
        if multi:
            multi_lookup = self.get_lookup_(location, AlternateSubstBuilder)
            multi_lookup.alternates = multi
        self.end_feature()
        self.lookups_.extend(old_lookups)

    def build_head(self):
        if not self.fontRevision_:
            return
        table = self.font.get("head")
        if not table:  # this only happens for unit tests
            table = self.font["head"] = newTable("head")
            table.decompile(b"\0" * 54, self.font)
            table.tableVersion = 1.0
            table.magicNumber = 0x5F0F3CF5
            table.created = table.modified = 3406620153  # 2011-12-13 11:22:33
        table.fontRevision = self.fontRevision_

    def build_hhea(self):
        if not self.hhea_:
            return
        table = self.font.get("hhea")
        if not table:  # this only happens for unit tests
            table = self.font["hhea"] = newTable("hhea")
            table.decompile(b"\0" * 36, self.font)
            table.tableVersion = 0x00010000
        if "caretoffset" in self.hhea_:
            table.caretOffset = self.hhea_["caretoffset"]
        if "ascender" in self.hhea_:
            table.ascent = self.hhea_["ascender"]
        if "descender" in self.hhea_:
            table.descent = self.hhea_["descender"]
        if "linegap" in self.hhea_:
            table.lineGap = self.hhea_["linegap"]

    def build_vhea(self):
        if not self.vhea_:
            return
        table = self.font.get("vhea")
        if not table:  # this only happens for unit tests
            table = self.font["vhea"] = newTable("vhea")
            table.decompile(b"\0" * 36, self.font)
            table.tableVersion = 0x00011000
        if "verttypoascender" in self.vhea_:
            table.ascent = self.vhea_["verttypoascender"]
        if "verttypodescender" in self.vhea_:
            table.descent = self.vhea_["verttypodescender"]
        if "verttypolinegap" in self.vhea_:
            table.lineGap = self.vhea_["verttypolinegap"]

    def get_user_name_id(self, table):
        # Try to find first unused font-specific name id
        nameIDs = [name.nameID for name in table.names]
        for user_name_id in range(256, 32767):
            if user_name_id not in nameIDs:
                return user_name_id

    def buildFeatureParams(self, tag):
        # by convention, a missing name ID is represented by 0xffff.
        # the spec says that these fields can be 'NULL', but 'NULL' is not
        # well defined for the purpose of nameIDs?
        NO_NAME_ID = 0xFFFF
        params = None
        if tag == "size":
            params = otTables.FeatureParamsSize()
            (
                params.DesignSize,
                params.SubfamilyID,
                params.RangeStart,
                params.RangeEnd,
            ) = self.size_parameters_
            if tag in self.featureNames_ids_:
                params.SubfamilyNameID = self.featureNames_ids_[tag]
            else:
                params.SubfamilyNameID = 0
        elif tag in self.featureNames_:
            if not self.featureNames_ids_:
                # name table wasn't selected among the tables to build; skip
                pass
            else:
                assert tag in self.featureNames_ids_
                params = otTables.FeatureParamsStylisticSet()
                params.Version = 0
                params.UINameID = self.featureNames_ids_[tag]
        elif tag in self.cv_parameters_:
            params = otTables.FeatureParamsCharacterVariants()
            params.Format = 0
            params.FeatUILabelNameID = self.cv_parameters_ids_.get(
                (tag, "FeatUILabelNameID"), NO_NAME_ID
            )
            params.FeatUITooltipTextNameID = self.cv_parameters_ids_.get(
                (tag, "FeatUITooltipTextNameID"), NO_NAME_ID
            )
            params.SampleTextNameID = self.cv_parameters_ids_.get(
                (tag, "SampleTextNameID"), NO_NAME_ID
            )
            params.NumNamedParameters = self.cv_num_named_params_.get(tag, 0)
            params.FirstParamUILabelNameID = self.cv_parameters_ids_.get(
                (tag, "ParamUILabelNameID_0"), NO_NAME_ID
            )
            params.CharCount = len(self.cv_characters_[tag])
            params.Character = self.cv_characters_[tag]
        return params

    def build_name(self):
        if not self.names_:
            return
        table = self.font.get("name")
        if not table:  # this only happens for unit tests
            table = self.font["name"] = newTable("name")
            table.names = []
        for name in self.names_:
            nameID, platformID, platEncID, langID, string = name
            # For featureNames block, nameID is 'feature tag'
            # For cvParameters blocks, nameID is ('feature tag', 'block name')
            if not isinstance(nameID, int):
                tag = nameID
                if tag in self.featureNames_:
                    if tag not in self.featureNames_ids_:
                        self.featureNames_ids_[tag] = self.get_user_name_id(table)
                        assert self.featureNames_ids_[tag] is not None
                    nameID = self.featureNames_ids_[tag]
                elif tag[0] in self.cv_parameters_:
                    if tag not in self.cv_parameters_ids_:
                        self.cv_parameters_ids_[tag] = self.get_user_name_id(table)
                        assert self.cv_parameters_ids_[tag] is not None
                    nameID = self.cv_parameters_ids_[tag]
            table.setName(string, nameID, platformID, platEncID, langID)
        table.names.sort()

    def build_OS_2(self):
        if not self.os2_:
            return
        table = self.font.get("OS/2")
        if not table:  # this only happens for unit tests
            table = self.font["OS/2"] = newTable("OS/2")
            data = b"\0" * sstruct.calcsize(getTableModule("OS/2").OS2_format_0)
            table.decompile(data, self.font)
        version = 0
        if "fstype" in self.os2_:
            table.fsType = self.os2_["fstype"]
        if "panose" in self.os2_:
            panose = getTableModule("OS/2").Panose()
            (
                panose.bFamilyType,
                panose.bSerifStyle,
                panose.bWeight,
                panose.bProportion,
                panose.bContrast,
                panose.bStrokeVariation,
                panose.bArmStyle,
                panose.bLetterForm,
                panose.bMidline,
                panose.bXHeight,
            ) = self.os2_["panose"]
            table.panose = panose
        if "typoascender" in self.os2_:
            table.sTypoAscender = self.os2_["typoascender"]
        if "typodescender" in self.os2_:
            table.sTypoDescender = self.os2_["typodescender"]
        if "typolinegap" in self.os2_:
            table.sTypoLineGap = self.os2_["typolinegap"]
        if "winascent" in self.os2_:
            table.usWinAscent = self.os2_["winascent"]
        if "windescent" in self.os2_:
            table.usWinDescent = self.os2_["windescent"]
        if "vendor" in self.os2_:
            table.achVendID = safeEval("'''" + self.os2_["vendor"] + "'''")
        if "weightclass" in self.os2_:
            table.usWeightClass = self.os2_["weightclass"]
        if "widthclass" in self.os2_:
            table.usWidthClass = self.os2_["widthclass"]
        if "unicoderange" in self.os2_:
            table.setUnicodeRanges(self.os2_["unicoderange"])
        if "codepagerange" in self.os2_:
            pages = self.build_codepages_(self.os2_["codepagerange"])
            table.ulCodePageRange1, table.ulCodePageRange2 = pages
            version = 1
        if "xheight" in self.os2_:
            table.sxHeight = self.os2_["xheight"]
            version = 2
        if "capheight" in self.os2_:
            table.sCapHeight = self.os2_["capheight"]
            version = 2
        if "loweropsize" in self.os2_:
            table.usLowerOpticalPointSize = self.os2_["loweropsize"]
            version = 5
        if "upperopsize" in self.os2_:
            table.usUpperOpticalPointSize = self.os2_["upperopsize"]
            version = 5

        def checkattr(table, attrs):
            for attr in attrs:
                if not hasattr(table, attr):
                    setattr(table, attr, 0)

        table.version = max(version, table.version)
        # this only happens for unit tests
        if version >= 1:
            checkattr(table, ("ulCodePageRange1", "ulCodePageRange2"))
        if version >= 2:
            checkattr(
                table,
                (
                    "sxHeight",
                    "sCapHeight",
                    "usDefaultChar",
                    "usBreakChar",
                    "usMaxContext",
                ),
            )
        if version >= 5:
            checkattr(table, ("usLowerOpticalPointSize", "usUpperOpticalPointSize"))

    def setElidedFallbackName(self, value, location):
        # ElidedFallbackName is a convenience method for setting
        # ElidedFallbackNameID so only one can be allowed
        for token in ("ElidedFallbackName", "ElidedFallbackNameID"):
            if token in self.stat_:
                raise FeatureLibError(
                    f"{token} is already set.",
                    location,
                )
        if isinstance(value, int):
            self.stat_["ElidedFallbackNameID"] = value
        elif isinstance(value, list):
            self.stat_["ElidedFallbackName"] = value
        else:
            raise AssertionError(value)

    def addDesignAxis(self, designAxis, location):
        if "DesignAxes" not in self.stat_:
            self.stat_["DesignAxes"] = []
        if designAxis.tag in (r.tag for r in self.stat_["DesignAxes"]):
            raise FeatureLibError(
                f'DesignAxis already defined for tag "{designAxis.tag}".',
                location,
            )
        if designAxis.axisOrder in (r.axisOrder for r in self.stat_["DesignAxes"]):
            raise FeatureLibError(
                f"DesignAxis already defined for axis number {designAxis.axisOrder}.",
                location,
            )
        self.stat_["DesignAxes"].append(designAxis)

    def addAxisValueRecord(self, axisValueRecord, location):
        if "AxisValueRecords" not in self.stat_:
            self.stat_["AxisValueRecords"] = []
        # Check for duplicate AxisValueRecords
        for record_ in self.stat_["AxisValueRecords"]:
            if (
                {n.asFea() for n in record_.names}
                == {n.asFea() for n in axisValueRecord.names}
                and {n.asFea() for n in record_.locations}
                == {n.asFea() for n in axisValueRecord.locations}
                and record_.flags == axisValueRecord.flags
            ):
                raise FeatureLibError(
                    "An AxisValueRecord with these values is already defined.",
                    location,
                )
        self.stat_["AxisValueRecords"].append(axisValueRecord)

    def build_STAT(self):
        if not self.stat_:
            return

        axes = self.stat_.get("DesignAxes")
        if not axes:
            raise FeatureLibError("DesignAxes not defined", None)
        axisValueRecords = self.stat_.get("AxisValueRecords")
        axisValues = {}
        format4_locations = []
        for tag in axes:
            axisValues[tag.tag] = []
        if axisValueRecords is not None:
            for avr in axisValueRecords:
                valuesDict = {}
                if avr.flags > 0:
                    valuesDict["flags"] = avr.flags
                if len(avr.locations) == 1:
                    location = avr.locations[0]
                    values = location.values
                    if len(values) == 1:  # format1
                        valuesDict.update({"value": values[0], "name": avr.names})
                    if len(values) == 2:  # format3
                        valuesDict.update(
                            {
                                "value": values[0],
                                "linkedValue": values[1],
                                "name": avr.names,
                            }
                        )
                    if len(values) == 3:  # format2
                        nominal, minVal, maxVal = values
                        valuesDict.update(
                            {
                                "nominalValue": nominal,
                                "rangeMinValue": minVal,
                                "rangeMaxValue": maxVal,
                                "name": avr.names,
                            }
                        )
                    axisValues[location.tag].append(valuesDict)
                else:
                    valuesDict.update(
                        {
                            "location": {i.tag: i.values[0] for i in avr.locations},
                            "name": avr.names,
                        }
                    )
                    format4_locations.append(valuesDict)

        designAxes = [
            {
                "ordering": a.axisOrder,
                "tag": a.tag,
                "name": a.names,
                "values": axisValues[a.tag],
            }
            for a in axes
        ]

        nameTable = self.font.get("name")
        if not nameTable:  # this only happens for unit tests
            nameTable = self.font["name"] = newTable("name")
            nameTable.names = []

        if "ElidedFallbackNameID" in self.stat_:
            nameID = self.stat_["ElidedFallbackNameID"]
            name = nameTable.getDebugName(nameID)
            if not name:
                raise FeatureLibError(
                    f"ElidedFallbackNameID {nameID} points "
                    "to a nameID that does not exist in the "
                    '"name" table',
                    None,
                )
        elif "ElidedFallbackName" in self.stat_:
            nameID = self.stat_["ElidedFallbackName"]

        otl.buildStatTable(
            self.font,
            designAxes,
            locations=format4_locations,
            elidedFallbackName=nameID,
        )

    def build_codepages_(self, pages):
        pages2bits = {
            1252: 0,
            1250: 1,
            1251: 2,
            1253: 3,
            1254: 4,
            1255: 5,
            1256: 6,
            1257: 7,
            1258: 8,
            874: 16,
            932: 17,
            936: 18,
            949: 19,
            950: 20,
            1361: 21,
            869: 48,
            866: 49,
            865: 50,
            864: 51,
            863: 52,
            862: 53,
            861: 54,
            860: 55,
            857: 56,
            855: 57,
            852: 58,
            775: 59,
            737: 60,
            708: 61,
            850: 62,
            437: 63,
        }
        bits = [pages2bits[p] for p in pages if p in pages2bits]
        pages = []
        for i in range(2):
            pages.append("")
            for j in range(i * 32, (i + 1) * 32):
                if j in bits:
                    pages[i] += "1"
                else:
                    pages[i] += "0"
        return [binary2num(p[::-1]) for p in pages]

    def buildBASE(self):
        if not self.base_horiz_axis_ and not self.base_vert_axis_:
            return None
        base = otTables.BASE()
        base.Version = 0x00010000
        base.HorizAxis = self.buildBASEAxis(self.base_horiz_axis_)
        base.VertAxis = self.buildBASEAxis(self.base_vert_axis_)

        result = newTable("BASE")
        result.table = base
        return result

    def buildBASECoord(self, c):
        coord = otTables.BaseCoord()
        coord.Format = 1
        coord.Coordinate = c
        return coord

    def buildBASEAxis(self, axis):
        if not axis:
            return
        bases, scripts, minmax = axis
        axis = otTables.Axis()
        axis.BaseTagList = otTables.BaseTagList()
        axis.BaseTagList.BaselineTag = bases
        axis.BaseTagList.BaseTagCount = len(bases)
        axis.BaseScriptList = otTables.BaseScriptList()
        axis.BaseScriptList.BaseScriptRecord = []
        axis.BaseScriptList.BaseScriptCount = len(scripts)
        for script in sorted(scripts):
            minmax_for_script = [
                record[1:] for record in minmax if record[0] == script[0]
            ]
            record = otTables.BaseScriptRecord()
            record.BaseScriptTag = script[0]
            record.BaseScript = otTables.BaseScript()
            record.BaseScript.BaseValues = otTables.BaseValues()
            record.BaseScript.BaseValues.DefaultIndex = bases.index(script[1])
            record.BaseScript.BaseValues.BaseCoord = []
            record.BaseScript.BaseValues.BaseCoordCount = len(script[2])
            record.BaseScript.BaseLangSysRecord = []

            for c in script[2]:
                record.BaseScript.BaseValues.BaseCoord.append(self.buildBASECoord(c))
            for language, min_coord, max_coord in sorted(minmax_for_script):
                minmax_record = otTables.MinMax()
                minmax_record.MinCoord = self.buildBASECoord(min_coord)
                minmax_record.MaxCoord = self.buildBASECoord(max_coord)
                minmax_record.FeatMinMaxCount = 0
                if language == "dflt":
                    record.BaseScript.DefaultMinMax = minmax_record
                else:
                    lang_record = otTables.BaseLangSysRecord()
                    lang_record.BaseLangSysTag = language
                    lang_record.MinMax = minmax_record
                    record.BaseScript.BaseLangSysRecord.append(lang_record)
            record.BaseScript.BaseLangSysCount = len(
                record.BaseScript.BaseLangSysRecord
            )
            axis.BaseScriptList.BaseScriptRecord.append(record)
        return axis

    def buildGDEF(self):
        gdef = otTables.GDEF()
        gdef.GlyphClassDef = self.buildGDEFGlyphClassDef_()
        gdef.AttachList = otl.buildAttachList(self.attachPoints_, self.glyphMap)
        gdef.LigCaretList = otl.buildLigCaretList(
            self.ligCaretCoords_, self.ligCaretPoints_, self.glyphMap
        )
        gdef.MarkAttachClassDef = self.buildGDEFMarkAttachClassDef_()
        gdef.MarkGlyphSetsDef = self.buildGDEFMarkGlyphSetsDef_()
        gdef.Version = 0x00010002 if gdef.MarkGlyphSetsDef else 0x00010000
        if self.varstorebuilder:
            store = self.varstorebuilder.finish()
            if store:
                gdef.Version = 0x00010003
                gdef.VarStore = store
                varidx_map = store.optimize()

                gdef.remap_device_varidxes(varidx_map)
                if "GPOS" in self.font:
                    self.font["GPOS"].table.remap_device_varidxes(varidx_map)
        if any(
            (
                gdef.GlyphClassDef,
                gdef.AttachList,
                gdef.LigCaretList,
                gdef.MarkAttachClassDef,
                gdef.MarkGlyphSetsDef,
            )
        ) or hasattr(gdef, "VarStore"):
            result = newTable("GDEF")
            result.table = gdef
            return result
        else:
            return None

    def buildGDEFGlyphClassDef_(self):
        if self.glyphClassDefs_:
            classes = {g: c for (g, (c, _)) in self.glyphClassDefs_.items()}
        else:
            classes = {}
            for lookup in self.lookups_:
                classes.update(lookup.inferGlyphClasses())
            for markClass in self.parseTree.markClasses.values():
                for markClassDef in markClass.definitions:
                    for glyph in markClassDef.glyphSet():
                        classes[glyph] = 3
        if classes:
            result = otTables.GlyphClassDef()
            result.classDefs = classes
            return result
        else:
            return None

    def buildGDEFMarkAttachClassDef_(self):
        classDefs = {g: c for g, (c, _) in self.markAttach_.items()}
        if not classDefs:
            return None
        result = otTables.MarkAttachClassDef()
        result.classDefs = classDefs
        return result

    def buildGDEFMarkGlyphSetsDef_(self):
        sets = []
        for glyphs, id_ in sorted(
            self.markFilterSets_.items(), key=lambda item: item[1]
        ):
            sets.append(glyphs)
        return otl.buildMarkGlyphSetsDef(sets, self.glyphMap)

    def buildDebg(self):
        if "Debg" not in self.font:
            self.font["Debg"] = newTable("Debg")
            self.font["Debg"].data = {}
        self.font["Debg"].data[LOOKUP_DEBUG_INFO_KEY] = self.lookup_locations

    def buildLookups_(self, tag):
        assert tag in ("GPOS", "GSUB"), tag
        for lookup in self.lookups_:
            lookup.lookup_index = None
        lookups = []
        for lookup in self.lookups_:
            if lookup.table != tag:
                continue
            name = self.get_lookup_name_(lookup)
            resolved = lookup.promote_lookup_type(is_named_lookup=name is not None)
            if resolved is None:
                raise FeatureLibError(
                    "Within a named lookup block, all rules must be of "
                    "the same lookup type and flag",
                    lookup.location,
                )
            for l in resolved:
                lookup.lookup_index = len(lookups)
                self.lookup_locations[tag][str(lookup.lookup_index)] = LookupDebugInfo(
                    location=str(lookup.location),
                    name=name,
                    feature=None,
                )
                lookups.append(l)
        otLookups = []
        for l in lookups:
            try:
                otLookups.append(l.build())
            except OpenTypeLibError as e:
                raise FeatureLibError(str(e), e.location) from e
            except Exception as e:
                location = self.lookup_locations[tag][str(l.lookup_index)].location
                raise FeatureLibError(str(e), location) from e
        return otLookups

    def makeTable(self, tag):
        table = getattr(otTables, tag, None)()
        table.Version = 0x00010000
        table.ScriptList = otTables.ScriptList()
        table.ScriptList.ScriptRecord = []
        table.FeatureList = otTables.FeatureList()
        table.FeatureList.FeatureRecord = []
        table.LookupList = otTables.LookupList()
        table.LookupList.Lookup = self.buildLookups_(tag)

        # Build a table for mapping (tag, lookup_indices) to feature_index.
        # For example, ('liga', (2,3,7)) --> 23.
        feature_indices = {}
        required_feature_indices = {}  # ('latn', 'DEU') --> 23
        scripts = {}  # 'latn' --> {'DEU': [23, 24]} for feature #23,24
        # Sort the feature table by feature tag:
        # https://github.com/fonttools/fonttools/issues/568
        sortFeatureTag = lambda f: (f[0][2], f[0][1], f[0][0], f[1])
        for key, lookups in sorted(self.features_.items(), key=sortFeatureTag):
            script, lang, feature_tag = key
            # l.lookup_index will be None when a lookup is not needed
            # for the table under construction. For example, substitution
            # rules will have no lookup_index while building GPOS tables.
            # We also deduplicate lookup indices, as they only get applied once
            # within a given feature:
            # https://github.com/fonttools/fonttools/issues/2946
            lookup_indices = tuple(
                dict.fromkeys(
                    l.lookup_index for l in lookups if l.lookup_index is not None
                )
            )
            # order doesn't matter, but lookup_indices preserves it.
            # We want to combine identical sets of lookups (order doesn't matter)
            # but also respect the order provided by the user (although there's
            # a reasonable argument to just sort and dedupe, which fontc does)
            lookup_key = frozenset(lookup_indices)

            size_feature = tag == "GPOS" and feature_tag == "size"
            force_feature = self.any_feature_variations(feature_tag, tag)
            if len(lookup_indices) == 0 and not size_feature and not force_feature:
                continue

            for ix in lookup_indices:
                try:
                    self.lookup_locations[tag][str(ix)] = self.lookup_locations[tag][
                        str(ix)
                    ]._replace(feature=key)
                except KeyError:
                    warnings.warn(
                        "feaLib.Builder subclass needs upgrading to "
                        "stash debug information. See fonttools#2065."
                    )

            feature_key = (feature_tag, lookup_key)
            feature_index = feature_indices.get(feature_key)
            if feature_index is None:
                feature_index = len(table.FeatureList.FeatureRecord)
                frec = otTables.FeatureRecord()
                frec.FeatureTag = feature_tag
                frec.Feature = otTables.Feature()
                frec.Feature.FeatureParams = self.buildFeatureParams(feature_tag)
                frec.Feature.LookupListIndex = list(lookup_indices)
                frec.Feature.LookupCount = len(lookup_indices)
                table.FeatureList.FeatureRecord.append(frec)
                feature_indices[feature_key] = feature_index
            scripts.setdefault(script, {}).setdefault(lang, []).append(feature_index)
            if self.required_features_.get((script, lang)) == feature_tag:
                required_feature_indices[(script, lang)] = feature_index

        # Build ScriptList.
        for script, lang_features in sorted(scripts.items()):
            srec = otTables.ScriptRecord()
            srec.ScriptTag = script
            srec.Script = otTables.Script()
            srec.Script.DefaultLangSys = None
            srec.Script.LangSysRecord = []
            for lang, feature_indices in sorted(lang_features.items()):
                langrec = otTables.LangSysRecord()
                langrec.LangSys = otTables.LangSys()
                langrec.LangSys.LookupOrder = None

                req_feature_index = required_feature_indices.get((script, lang))
                if req_feature_index is None:
                    langrec.LangSys.ReqFeatureIndex = 0xFFFF
                else:
                    langrec.LangSys.ReqFeatureIndex = req_feature_index

                langrec.LangSys.FeatureIndex = [
                    i for i in feature_indices if i != req_feature_index
                ]
                langrec.LangSys.FeatureCount = len(langrec.LangSys.FeatureIndex)

                if lang == "dflt":
                    srec.Script.DefaultLangSys = langrec.LangSys
                else:
                    langrec.LangSysTag = lang
                    srec.Script.LangSysRecord.append(langrec)
            srec.Script.LangSysCount = len(srec.Script.LangSysRecord)
            table.ScriptList.ScriptRecord.append(srec)

        table.ScriptList.ScriptCount = len(table.ScriptList.ScriptRecord)
        table.FeatureList.FeatureCount = len(table.FeatureList.FeatureRecord)
        table.LookupList.LookupCount = len(table.LookupList.Lookup)
        return table

    def makeFeatureVariations(self, table, table_tag):
        feature_vars = {}
        has_any_variations = False
        # Sort out which lookups to build, gather their indices
        for (_, _, feature_tag), variations in self.feature_variations_.items():
            feature_vars[feature_tag] = []
            for conditionset, builders in variations.items():
                raw_conditionset = self.conditionsets_[conditionset]
                indices = []
                for b in builders:
                    if b.table != table_tag:
                        continue
                    assert b.lookup_index is not None
                    indices.append(b.lookup_index)
                    has_any_variations = True
                feature_vars[feature_tag].append((raw_conditionset, indices))

        if has_any_variations:
            for feature_tag, conditions_and_lookups in feature_vars.items():
                addFeatureVariationsRaw(
                    self.font, table, conditions_and_lookups, feature_tag
                )

    def any_feature_variations(self, feature_tag, table_tag):
        for (_, _, feature), variations in self.feature_variations_.items():
            if feature != feature_tag:
                continue
            for conditionset, builders in variations.items():
                if any(b.table == table_tag for b in builders):
                    return True
        return False

    def get_lookup_name_(self, lookup):
        rev = {v: k for k, v in self.named_lookups_.items()}
        if lookup in rev:
            return rev[lookup]
        return None

    def add_language_system(self, location, script, language):
        # OpenType Feature File Specification, section 4.b.i
        if script == "DFLT" and language == "dflt" and self.default_language_systems_:
            raise FeatureLibError(
                'If "languagesystem DFLT dflt" is present, it must be '
                "the first of the languagesystem statements",
                location,
            )
        if script == "DFLT":
            if self.seen_non_DFLT_script_:
                raise FeatureLibError(
                    'languagesystems using the "DFLT" script tag must '
                    "precede all other languagesystems",
                    location,
                )
        else:
            self.seen_non_DFLT_script_ = True
        if (script, language) in self.default_language_systems_:
            raise FeatureLibError(
                '"languagesystem %s %s" has already been specified'
                % (script.strip(), language.strip()),
                location,
            )
        self.default_language_systems_.add((script, language))

    def get_default_language_systems_(self):
        # OpenType Feature File specification, 4.b.i. languagesystem:
        # If no "languagesystem" statement is present, then the
        # implementation must behave exactly as though the following
        # statement were present at the beginning of the feature file:
        # languagesystem DFLT dflt;
        if self.default_language_systems_:
            return frozenset(self.default_language_systems_)
        else:
            return frozenset({("DFLT", "dflt")})

    def start_feature(self, location, name, use_extension=False):
        if use_extension and name != "aalt":
            raise FeatureLibError(
                "'useExtension' keyword for feature blocks is allowed only for 'aalt' feature",
                location,
            )
        self.language_systems = self.get_default_language_systems_()
        # The current script and language start out as the first declared
        # language system, sorted by (script, language) tag. Kind of odd,
        # but matches makeotf (and fea-rs matches us)
        self.script_, self.language_ = min(self.language_systems)
        self.cur_lookup_ = None
        self.cur_feature_name_ = name
        self.lookupflag_ = 0
        self.lookupflag_markFilterSet_ = None
        self.use_extension_ = use_extension
        if name == "aalt":
            self.aalt_location_ = location
            self.aalt_use_extension_ = use_extension

    def end_feature(self):
        assert self.cur_feature_name_ is not None
        self.cur_feature_name_ = None
        self.language_systems = None
        self.script_ = None
        self.language_ = None
        self.cur_lookup_ = None
        self.lookupflag_ = 0
        self.lookupflag_markFilterSet_ = None
        self.use_extension_ = False

    def start_lookup_block(self, location, name, use_extension=False):
        if name in self.named_lookups_:
            raise FeatureLibError(
                'Lookup "%s" has already been defined' % name, location
            )
        if self.cur_feature_name_ == "aalt":
            raise FeatureLibError(
                "Lookup blocks cannot be placed inside 'aalt' features; "
                "move it out, and then refer to it with a lookup statement",
                location,
            )
        self.cur_lookup_name_ = name
        self.named_lookups_[name] = None
        self.cur_lookup_ = None
        self.use_extension_ = use_extension
        if self.cur_feature_name_ is None:
            self.lookupflag_ = 0
            self.lookupflag_markFilterSet_ = None

    def end_lookup_block(self):
        assert self.cur_lookup_name_ is not None
        self.cur_lookup_name_ = None
        self.cur_lookup_ = None
        self.use_extension_ = False
        if self.cur_feature_name_ is None:
            self.lookupflag_ = 0
            self.lookupflag_markFilterSet_ = None

    def add_lookup_call(self, lookup_name):
        assert lookup_name in self.named_lookups_, lookup_name
        self.cur_lookup_ = None
        lookup = self.named_lookups_[lookup_name]
        if lookup is not None:  # skip empty named lookup
            self.add_lookup_to_feature_(lookup, self.cur_feature_name_)

    def set_font_revision(self, location, revision):
        self.fontRevision_ = revision

    def set_language(self, location, language, include_default, required):
        assert len(language) == 4
        if self.cur_feature_name_ in ("aalt", "size"):
            raise FeatureLibError(
                "Language statements are not allowed "
                'within "feature %s"' % self.cur_feature_name_,
                location,
            )
        if self.cur_feature_name_ is None:
            raise FeatureLibError(
                "Language statements are not allowed "
                "within standalone lookup blocks",
                location,
            )
        self.cur_lookup_ = None

        key = (self.script_, language, self.cur_feature_name_)
        lookups = self.features_.get((key[0], "dflt", key[2]))
        if (language == "dflt" or include_default) and lookups:
            self.features_[key] = lookups[:]
        else:
            # if we aren't including default we need to manually remove the
            # default lookups, which were added to all declared langsystems
            # as they were encountered (we don't remove all lookups because
            # we want to allow duplicate script/lang statements;
            # see https://github.com/fonttools/fonttools/issues/3748
            cur_lookups = self.features_.get(key, [])
            self.features_[key] = [x for x in cur_lookups if x not in lookups]
        self.language_systems = frozenset([(self.script_, language)])
        self.language_ = language

        if required:
            key = (self.script_, language)
            if key in self.required_features_:
                raise FeatureLibError(
                    "Language %s (script %s) has already "
                    "specified feature %s as its required feature"
                    % (
                        language.strip(),
                        self.script_.strip(),
                        self.required_features_[key].strip(),
                    ),
                    location,
                )
            self.required_features_[key] = self.cur_feature_name_

    def getMarkAttachClass_(self, location, glyphs):
        glyphs = frozenset(glyphs)
        id_ = self.markAttachClassID_.get(glyphs)
        if id_ is not None:
            return id_
        id_ = len(self.markAttachClassID_) + 1
        self.markAttachClassID_[glyphs] = id_
        for glyph in glyphs:
            if glyph in self.markAttach_:
                _, loc = self.markAttach_[glyph]
                raise FeatureLibError(
                    "Glyph %s already has been assigned "
                    "a MarkAttachmentType at %s" % (glyph, loc),
                    location,
                )
            self.markAttach_[glyph] = (id_, location)
        return id_

    def getMarkFilterSet_(self, location, glyphs):
        glyphs = frozenset(glyphs)
        id_ = self.markFilterSets_.get(glyphs)
        if id_ is not None:
            return id_
        id_ = len(self.markFilterSets_)
        self.markFilterSets_[glyphs] = id_
        return id_

    def set_lookup_flag(self, location, value, markAttach, markFilter):
        value = value & 0xFF
        if markAttach is not None:
            markAttachClass = self.getMarkAttachClass_(location, markAttach)
            value = value | (markAttachClass << 8)
        if markFilter is not None:
            markFilterSet = self.getMarkFilterSet_(location, markFilter)
            value = value | 0x10
            self.lookupflag_markFilterSet_ = markFilterSet
        else:
            self.lookupflag_markFilterSet_ = None
        self.lookupflag_ = value

    def set_script(self, location, script):
        if self.cur_feature_name_ in ("aalt", "size"):
            raise FeatureLibError(
                "Script statements are not allowed "
                'within "feature %s"' % self.cur_feature_name_,
                location,
            )
        if self.cur_feature_name_ is None:
            raise FeatureLibError(
                "Script statements are not allowed " "within standalone lookup blocks",
                location,
            )
        if script == self.script_ and self.language_ == "dflt":
            # A script statement naming the already-current script still narrows
            # the language systems that the rules after it are registered under,
            # and still terminates the current lookup; it differs from the normal
            # path only in leaving script_ and the lookupflag alone, see
            # https://github.com/fonttools/fonttools/issues/1824.
            self.set_language(location, "dflt", include_default=True, required=False)
            return
        self.cur_lookup_ = None
        self.script_ = script
        self.lookupflag_ = 0
        self.lookupflag_markFilterSet_ = None
        self.set_language(location, "dflt", include_default=True, required=False)

    def find_lookup_builders_(self, lookups):
        """Helper for building chain contextual substitutions

        Given a list of lookup names, finds the LookupBuilder for each name.
        If an input name is None, it gets mapped to a None LookupBuilder.
        """
        lookup_builders = []
        for lookuplist in lookups:
            if lookuplist is not None:
                lookup_builders.append(
                    [self.named_lookups_.get(l.name) for l in lookuplist]
                )
            else:
                lookup_builders.append(None)
        return lookup_builders

    def add_attach_points(self, location, glyphs, contourPoints):
        for glyph in glyphs:
            self.attachPoints_.setdefault(glyph, set()).update(contourPoints)

    def add_feature_reference(self, location, featureName):
        if self.cur_feature_name_ != "aalt":
            raise FeatureLibError(
                'Feature references are only allowed inside "feature aalt"', location
            )
        self.aalt_features_.append((location, featureName))

    def add_featureName(self, tag):
        self.featureNames_.add(tag)

    def add_cv_parameter(self, tag):
        self.cv_parameters_.add(tag)

    def add_to_cv_num_named_params(self, tag):
        """Adds new items to ``self.cv_num_named_params_``
        or increments the count of existing items."""
        if tag in self.cv_num_named_params_:
            self.cv_num_named_params_[tag] += 1
        else:
            self.cv_num_named_params_[tag] = 1

    def add_cv_character(self, character, tag):
        self.cv_characters_[tag].append(character)

    def set_base_axis(self, bases, scripts, vertical, minmax=[]):
        if vertical:
            self.base_vert_axis_ = (bases, scripts, minmax)
        else:
            self.base_horiz_axis_ = (bases, scripts, minmax)

    def set_size_parameters(
        self, location, DesignSize, SubfamilyID, RangeStart, RangeEnd
    ):
        if self.cur_feature_name_ != "size":
            raise FeatureLibError(
                "Parameters statements are not allowed "
                'within "feature %s"' % self.cur_feature_name_,
                location,
            )
        self.size_parameters_ = [DesignSize, SubfamilyID, RangeStart, RangeEnd]
        for script, lang in self.language_systems:
            key = (script, lang, self.cur_feature_name_)
            self.features_.setdefault(key, [])

    # GSUB rules

    def add_any_subst_(self, location, mapping):
        lookup = self.get_lookup_(location, AnySubstBuilder, mapping=mapping)
        for key, value in mapping.items():
            if key in lookup.mapping:
                if value == lookup.mapping[key]:
                    log.info(
                        'Removing duplicate substitution from "%s" to "%s" at %s',
                        ", ".join(key),
                        ", ".join(value),
                        location,
                    )
                else:
                    raise FeatureLibError(
                        'Already defined substitution for "%s"' % ", ".join(key),
                        location,
                    )
            lookup.mapping[key] = value

    # GSUB 1
    def add_single_subst(self, location, prefix, suffix, mapping, forceChain):
        if self.cur_feature_name_ == "aalt":
            for from_glyph, to_glyph in mapping.items():
                alts = self.aalt_alternates_.setdefault(from_glyph, [])
                if to_glyph not in alts:
                    alts.append(to_glyph)
            return
        if prefix or suffix or forceChain:
            self.add_single_subst_chained_(location, prefix, suffix, mapping)
            return

        self.add_any_subst_(
            location,
            {(key,): (value,) for key, value in mapping.items()},
        )

    # GSUB 2
    def add_multiple_subst(
        self, location, prefix, glyph, suffix, replacements, forceChain=False
    ):
        if prefix or suffix or forceChain:
            self.add_multi_subst_chained_(location, prefix, glyph, suffix, replacements)
            return
        self.add_any_subst_(
            location,
            {(glyph,): tuple(replacements)},
        )

    # GSUB 3
    def add_alternate_subst(self, location, prefix, glyph, suffix, replacement):
        if self.cur_feature_name_ == "aalt":
            alts = self.aalt_alternates_.setdefault(glyph, [])
            alts.extend(g for g in replacement if g not in alts)
            return
        if prefix or suffix:
            chain = self.get_lookup_(location, ChainContextSubstBuilder)
            lookup = chain.find_chainable_alternate_subst(glyph)
            if lookup is None:
                lookup = self.get_chained_lookup_(location, AlternateSubstBuilder)
            self._add_contextual_rule(chain, prefix, [{glyph}], suffix, [lookup])
        else:
            lookup = self.get_lookup_(location, AlternateSubstBuilder)
        if glyph in lookup.alternates:
            raise FeatureLibError(
                'Already defined alternates for glyph "%s"' % glyph, location
            )
        # We allow empty replacement glyphs here.
        lookup.alternates[glyph] = replacement

    # GSUB 4
    def add_ligature_subst(
        self, location, prefix, glyphs, suffix, replacement, forceChain
    ):
        if prefix or suffix or forceChain:
            self.add_ligature_subst_chained_(
                location, prefix, glyphs, suffix, replacement
            )
            return
        if not all(glyphs):
            raise FeatureLibError("Empty glyph class in substitution", location)

        # OpenType feature file syntax, section 5.d, "Ligature substitution":
        # "Since the OpenType specification does not allow ligature
        # substitutions to be specified on target sequences that contain
        # glyph classes, the implementation software will enumerate
        # all specific glyph sequences if glyph classes are detected"
        self.add_any_subst_(
            location,
            {g: (replacement,) for g in itertools.product(*glyphs)},
        )

    @staticmethod
    def _add_contextual_rule(chain, prefix, glyphs, suffix, lookups):
        """Add a contextual rule, merging with the last rule if possible.

        Consecutive rules that share the same prefix, suffix, lookups, and have
        a single input position can be merged into one rule with broader input
        coverage. This produces more compact binary tables (often Format 3).
        """
        if len(glyphs) == 1 and chain.rules and not chain.rules[-1].is_subtable_break:
            last = chain.rules[-1]
            if (
                len(last.glyphs) == 1
                and last.prefix == prefix
                and last.suffix == suffix
                and last.lookups == lookups
            ):
                if not isinstance(last.glyphs[0], set):
                    last.glyphs[0] = set(last.glyphs[0])
                last.glyphs[0].update(glyphs[0])
                return
        chain.rules.append(ChainContextualRule(prefix, glyphs, suffix, lookups))

    # GSUB 5/6
    def add_chain_context_subst(self, location, prefix, glyphs, suffix, lookups):
        if not all(glyphs) or not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual substitution", location
            )
        lookup = self.get_lookup_(location, ChainContextSubstBuilder)
        resolved = self.find_lookup_builders_(lookups)
        self._add_contextual_rule(lookup, prefix, glyphs, suffix, resolved)

    def add_single_subst_chained_(self, location, prefix, suffix, mapping):
        if not mapping or not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual substitution", location
            )
        # https://github.com/fonttools/fonttools/issues/512
        # https://github.com/fonttools/fonttools/issues/2150
        chain = self.get_lookup_(location, ChainContextSubstBuilder)
        sub = chain.find_chainable_subst(mapping, SingleSubstBuilder)
        if sub is None:
            sub = self.get_chained_lookup_(location, SingleSubstBuilder)
        sub.mapping.update(mapping)
        keys = set(mapping.keys())
        self._add_contextual_rule(chain, prefix, [keys], suffix, [sub])

    def add_multi_subst_chained_(self, location, prefix, glyph, suffix, replacements):
        if not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual substitution", location
            )
        # https://github.com/fonttools/fonttools/issues/3551
        chain = self.get_lookup_(location, ChainContextSubstBuilder)
        sub = chain.find_chainable_subst({glyph: replacements}, MultipleSubstBuilder)
        if sub is None:
            sub = self.get_chained_lookup_(location, MultipleSubstBuilder)
        sub.mapping[glyph] = replacements
        # https://github.com/fonttools/fonttools/issues/4016
        self._add_contextual_rule(chain, prefix, [{glyph}], suffix, [sub])

    def add_ligature_subst_chained_(
        self, location, prefix, glyphs, suffix, replacement
    ):
        # https://github.com/fonttools/fonttools/issues/3701
        if not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual substitution", location
            )
        chain = self.get_lookup_(location, ChainContextSubstBuilder)
        sub = chain.find_chainable_ligature_subst(glyphs, replacement)
        if sub is None:
            sub = self.get_chained_lookup_(location, LigatureSubstBuilder)

        for g in itertools.product(*glyphs):
            existing = sub.ligatures.get(g, replacement)
            if existing != replacement:
                raise FeatureLibError(
                    f"Conflicting ligature sub rules: '{g}' maps to '{existing}' and '{replacement}'",
                    location,
                )

            sub.ligatures[g] = replacement

        chain.rules.append(ChainContextualRule(prefix, glyphs, suffix, [sub]))

    # GSUB 8
    def add_reverse_chain_single_subst(self, location, old_prefix, old_suffix, mapping):
        if not mapping:
            raise FeatureLibError("Empty glyph class in substitution", location)
        lookup = self.get_lookup_(location, ReverseChainSingleSubstBuilder)
        lookup.rules.append((old_prefix, old_suffix, mapping))

    # GPOS rules

    # GPOS 1
    def add_single_pos(self, location, prefix, suffix, pos, forceChain):
        if prefix or suffix or forceChain:
            self.add_single_pos_chained_(location, prefix, suffix, pos)
        else:
            lookup = self.get_lookup_(location, SinglePosBuilder)
            for glyphs, value in pos:
                if not glyphs:
                    raise FeatureLibError(
                        "Empty glyph class in positioning rule", location
                    )
                otValueRecord = self.makeOpenTypeValueRecord(
                    location, value, pairPosContext=False
                )
                for glyph in glyphs:
                    try:
                        lookup.add_pos(location, glyph, otValueRecord)
                    except OpenTypeLibError as e:
                        raise FeatureLibError(str(e), e.location) from e

    # GPOS 2
    def add_class_pair_pos(self, location, glyphclass1, value1, glyphclass2, value2):
        if not glyphclass1 or not glyphclass2:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        lookup = self.get_lookup_(location, PairPosBuilder)
        v1 = self.makeOpenTypeValueRecord(location, value1, pairPosContext=True)
        v2 = self.makeOpenTypeValueRecord(location, value2, pairPosContext=True)
        cls1 = tuple(sorted(set(glyphclass1)))
        cls2 = tuple(sorted(set(glyphclass2)))
        lookup.addClassPair(location, cls1, v1, cls2, v2)

    def add_specific_pair_pos(self, location, glyph1, value1, glyph2, value2):
        if not glyph1 or not glyph2:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        lookup = self.get_lookup_(location, PairPosBuilder)
        v1 = self.makeOpenTypeValueRecord(location, value1, pairPosContext=True)
        v2 = self.makeOpenTypeValueRecord(location, value2, pairPosContext=True)
        lookup.addGlyphPair(location, glyph1, v1, glyph2, v2)

    # GPOS 3
    def add_cursive_pos(self, location, glyphclass, entryAnchor, exitAnchor):
        if not glyphclass:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        lookup = self.get_lookup_(location, CursivePosBuilder)
        lookup.add_attachment(
            location,
            glyphclass,
            self.makeOpenTypeAnchor(location, entryAnchor),
            self.makeOpenTypeAnchor(location, exitAnchor),
        )

    # GPOS 4
    def add_mark_base_pos(self, location, bases, marks):
        builder = self.get_lookup_(location, MarkBasePosBuilder)
        self.add_marks_(location, builder, marks)
        if not bases:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        for baseAnchor, markClass in marks:
            otBaseAnchor = self.makeOpenTypeAnchor(location, baseAnchor)
            for base in bases:
                builder.bases.setdefault(base, {})[markClass.name] = otBaseAnchor

    # GPOS 5
    def add_mark_lig_pos(self, location, ligatures, components):
        builder = self.get_lookup_(location, MarkLigPosBuilder)
        componentAnchors = []
        if not ligatures:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        for marks in components:
            anchors = {}
            self.add_marks_(location, builder, marks)
            for ligAnchor, markClass in marks:
                anchors[markClass.name] = self.makeOpenTypeAnchor(location, ligAnchor)
            componentAnchors.append(anchors)
        for glyph in ligatures:
            builder.ligatures[glyph] = componentAnchors

    # GPOS 6
    def add_mark_mark_pos(self, location, baseMarks, marks):
        builder = self.get_lookup_(location, MarkMarkPosBuilder)
        self.add_marks_(location, builder, marks)
        if not baseMarks:
            raise FeatureLibError("Empty glyph class in positioning rule", location)
        for baseAnchor, markClass in marks:
            otBaseAnchor = self.makeOpenTypeAnchor(location, baseAnchor)
            for baseMark in baseMarks:
                builder.baseMarks.setdefault(baseMark, {})[
                    markClass.name
                ] = otBaseAnchor

    # GPOS 7/8
    def add_chain_context_pos(self, location, prefix, glyphs, suffix, lookups):
        if not all(glyphs) or not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual positioning rule", location
            )
        lookup = self.get_lookup_(location, ChainContextPosBuilder)
        resolved = self.find_lookup_builders_(lookups)
        self._add_contextual_rule(lookup, prefix, glyphs, suffix, resolved)

    def add_single_pos_chained_(self, location, prefix, suffix, pos):
        if not pos or not all(prefix) or not all(suffix):
            raise FeatureLibError(
                "Empty glyph class in contextual positioning rule", location
            )
        # https://github.com/fonttools/fonttools/issues/514
        chain = self.get_lookup_(location, ChainContextPosBuilder)
        targets = []
        for _, _, _, lookups in chain.rules:
            targets.extend(lookups)
        subs = []
        for glyphs, value in pos:
            if value is None:
                subs.append(None)
                continue
            otValue = self.makeOpenTypeValueRecord(
                location, value, pairPosContext=False
            )
            sub = chain.find_chainable_single_pos(targets, glyphs, otValue)
            if sub is None:
                sub = self.get_chained_lookup_(location, SinglePosBuilder)
                targets.append(sub)
            for glyph in glyphs:
                sub.add_pos(location, glyph, otValue)
            subs.append(sub)
        assert len(pos) == len(subs), (pos, subs)
        glyphs = [g for g, v in pos]
        self._add_contextual_rule(chain, prefix, glyphs, suffix, subs)

    def add_marks_(self, location, lookupBuilder, marks):
        """Helper for add_mark_{base,liga,mark}_pos."""
        for _, markClass in marks:
            for markClassDef in markClass.definitions:
                for mark in markClassDef.glyphs.glyphSet():
                    if mark not in lookupBuilder.marks:
                        otMarkAnchor = self.makeOpenTypeAnchor(
                            location, copy.deepcopy(markClassDef.anchor)
                        )
                        lookupBuilder.marks[mark] = (markClass.name, otMarkAnchor)
                    else:
                        existingMarkClass = lookupBuilder.marks[mark][0]
                        if markClass.name != existingMarkClass:
                            raise FeatureLibError(
                                "Glyph %s cannot be in both @%s and @%s"
                                % (mark, existingMarkClass, markClass.name),
                                location,
                            )

    def add_subtable_break(self, location):
        self.cur_lookup_.add_subtable_break(location)

    def setGlyphClass_(self, location, glyph, glyphClass):
        oldClass, oldLocation = self.glyphClassDefs_.get(glyph, (None, None))
        if oldClass and oldClass != glyphClass:
            raise FeatureLibError(
                "Glyph %s was assigned to a different class at %s"
                % (glyph, oldLocation),
                location,
            )
        self.glyphClassDefs_[glyph] = (glyphClass, location)

    def add_glyphClassDef(
        self, location, baseGlyphs, ligatureGlyphs, markGlyphs, componentGlyphs
    ):
        for glyph in baseGlyphs:
            self.setGlyphClass_(location, glyph, 1)
        for glyph in ligatureGlyphs:
            self.setGlyphClass_(location, glyph, 2)
        for glyph in markGlyphs:
            self.setGlyphClass_(location, glyph, 3)
        for glyph in componentGlyphs:
            self.setGlyphClass_(location, glyph, 4)

    def add_ligatureCaretByIndex_(self, location, glyphs, carets):
        for glyph in glyphs:
            if glyph not in self.ligCaretPoints_:
                self.ligCaretPoints_[glyph] = carets

    def makeLigCaret(self, location, caret):
        if not isinstance(caret, VariableScalar):
            return caret
        default, device = self.makeVariablePos(location, caret)
        if device is not None:
            return (default, device)
        return default

    def add_ligatureCaretByPos_(self, location, glyphs, carets):
        carets = [self.makeLigCaret(location, caret) for caret in carets]
        for glyph in glyphs:
            if glyph not in self.ligCaretCoords_:
                self.ligCaretCoords_[glyph] = carets

    def add_name_record(self, location, nameID, platformID, platEncID, langID, string):
        self.names_.append([nameID, platformID, platEncID, langID, string])

    def add_os2_field(self, key, value):
        self.os2_[key] = value

    def add_hhea_field(self, key, value):
        self.hhea_[key] = value

    def add_vhea_field(self, key, value):
        self.vhea_[key] = value

    def add_conditionset(self, location, key, value):
        if "fvar" not in self.font:
            raise FeatureLibError(
                "Cannot add feature variations to a font without an 'fvar' table",
                location,
            )

        if key in self.conditionsets_:
            raise FeatureLibError(
                f"Condition set '{key}' has the same name as a previous condition set",
                location,
            )

        # Normalize
        axisMap = {
            axis.axisTag: (axis.minValue, axis.defaultValue, axis.maxValue)
            for axis in self.axes
        }

        value = {
            tag: (
                normalizeValue(bottom, axisMap[tag]),
                normalizeValue(top, axisMap[tag]),
            )
            for tag, (bottom, top) in value.items()
        }

        # NOTE: This might result in rounding errors (off-by-ones) compared to
        # rules in Designspace files, since we're working with what's in the
        # `avar` table rather than the original values.
        if "avar" in self.font:
            mapping = self.font["avar"].segments
            value = {
                axis: tuple(
                    piecewiseLinearMap(v, mapping[axis]) if axis in mapping else v
                    for v in condition_range
                )
                for axis, condition_range in value.items()
            }

        self.conditionsets_[key] = value

    def makeVariablePos(
        self, location, varscalar: VariableScalar
    ) -> tuple[int, int | None]:
        """Make a pos statement from a VariableScalar, returning the default
        value, and optionally the variation index if the scalar genuinely
        requires variation too."""

        if self.varstorebuilder is None or self.scalar_builder is None:
            raise FeatureLibError(
                "Can't define a variable scalar in a non-variable font", location
            )

        if not varscalar.does_vary:
            return self.scalar_builder.default_value(varscalar), None

        try:
            default, index = self.scalar_builder.add_to_variation_store(
                varscalar, self.varstorebuilder
            )
        except VarLibError as e:
            raise FeatureLibError(
                "Failed to compute deltas for variable scalar", location
            ) from e

        device = None
        if index is not None and index != 0xFFFFFFFF:
            device = buildVarDevTable(index)

        return default, device

    def makeAnchorPos(self, varscalar, deviceTable, location):
        device = None
        if not isinstance(varscalar, VariableScalar):
            if deviceTable is not None:
                device = otl.buildDevice(dict(deviceTable))
            return varscalar, device
        default, device = self.makeVariablePos(location, varscalar)
        if device is not None and deviceTable is not None:
            raise FeatureLibError(
                "Can't define a device coordinate and variable scalar", location
            )
        return default, device

    def makeOpenTypeAnchor(self, location, anchor):
        """ast.Anchor --> otTables.Anchor"""
        if anchor is None:
            return None
        deviceX, deviceY = None, None
        if anchor.xDeviceTable is not None:
            deviceX = otl.buildDevice(dict(anchor.xDeviceTable))
        if anchor.yDeviceTable is not None:
            deviceY = otl.buildDevice(dict(anchor.yDeviceTable))
        x, deviceX = self.makeAnchorPos(anchor.x, anchor.xDeviceTable, location)
        y, deviceY = self.makeAnchorPos(anchor.y, anchor.yDeviceTable, location)
        otlanchor = otl.buildAnchor(x, y, anchor.contourpoint, deviceX, deviceY)
        return otlanchor

    _VALUEREC_ATTRS = {
        name[0].lower() + name[1:]: (name, isDevice)
        for _, name, isDevice, _ in otBase.valueRecordFormat
        if not name.startswith("Reserved")
    }

    def makeOpenTypeValueRecord(self, location, v, pairPosContext):
        """ast.ValueRecord --> otBase.ValueRecord"""
        if not v:
            return None

        vr = {}
        for astName, (otName, isDevice) in self._VALUEREC_ATTRS.items():
            val = getattr(v, astName, None)
            if not val:
                continue
            if isDevice:
                vr[otName] = otl.buildDevice(dict(val))
            elif isinstance(val, VariableScalar):
                otDeviceName = otName[0:4] + "Device"
                feaDeviceName = otDeviceName[0].lower() + otDeviceName[1:]
                if getattr(v, feaDeviceName):
                    raise FeatureLibError(
                        "Can't define a device coordinate and variable scalar", location
                    )
                vr[otName], device = self.makeVariablePos(location, val)
                if device is not None:
                    vr[otDeviceName] = device
            else:
                vr[otName] = val

        if pairPosContext and not vr:
            vr = {"YAdvance": 0} if v.vertical else {"XAdvance": 0}
        valRec = otl.buildValue(vr)
        return valRec
<|endoftext|>from typing import NamedTuple


class FeatureLibLocation(NamedTuple):
    """A location in a feature file"""

    file: str
    line: int
    column: int

    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"
<|endoftext|>import weakref
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools

SHIFT = " " * 4

__all__ = [
    "Element",
    "FeatureFile",
    "Comment",
    "GlyphName",
    "GlyphClass",
    "GlyphClassName",
    "MarkClassName",
    "AnonymousBlock",
    "Block",
    "FeatureBlock",
    "NestedBlock",
    "LookupBlock",
    "GlyphClassDefinition",
    "GlyphClassDefStatement",
    "MarkClass",
    "MarkClassDefinition",
    "AlternateSubstStatement",
    "Anchor",
    "AnchorDefinition",
    "AttachStatement",
    "AxisValueLocationStatement",
    "BaseAxis",
    "CVParametersNameStatement",
    "ChainContextPosStatement",
    "ChainContextSubstStatement",
    "CharacterStatement",
    "ConditionsetStatement",
    "CursivePosStatement",
    "ElidedFallbackName",
    "ElidedFallbackNameID",
    "Expression",
    "FeatureNameStatement",
    "FeatureReferenceStatement",
    "FontRevisionStatement",
    "HheaField",
    "IgnorePosStatement",
    "IgnoreSubstStatement",
    "IncludeStatement",
    "LanguageStatement",
    "LanguageSystemStatement",
    "LigatureCaretByIndexStatement",
    "LigatureCaretByPosStatement",
    "LigatureSubstStatement",
    "LookupFlagStatement",
    "LookupReferenceStatement",
    "MarkBasePosStatement",
    "MarkLigPosStatement",
    "MarkMarkPosStatement",
    "MultipleSubstStatement",
    "NameRecord",
    "OS2Field",
    "PairPosStatement",
    "ReverseChainSingleSubstStatement",
    "ScriptStatement",
    "SinglePosStatement",
    "SingleSubstStatement",
    "SizeParameters",
    "Statement",
    "STATAxisValueStatement",
    "STATDesignAxisStatement",
    "STATNameStatement",
    "SubtableStatement",
    "TableBlock",
    "ValueRecord",
    "ValueRecordDefinition",
    "VheaField",
]


def deviceToString(device):
    if device is None:
        return "<device NULL>"
    else:
        return "<device %s>" % ", ".join("%d %d" % t for t in device)


fea_keywords = set(
    [
        "anchor",
        "anchordef",
        "anon",
        "anonymous",
        "by",
        "contour",
        "cursive",
        "device",
        "enum",
        "enumerate",
        "excludedflt",
        "exclude_dflt",
        "feature",
        "from",
        "ignore",
        "ignorebaseglyphs",
        "ignoreligatures",
        "ignoremarks",
        "include",
        "includedflt",
        "include_dflt",
        "language",
        "languagesystem",
        "lookup",
        "lookupflag",
        "mark",
        "markattachmenttype",
        "markclass",
        "nameid",
        "null",
        "parameters",
        "pos",
        "position",
        "required",
        "righttoleft",
        "reversesub",
        "rsub",
        "script",
        "sub",
        "substitute",
        "subtable",
        "table",
        "usemarkfilteringset",
        "useextension",
        "valuerecorddef",
        "base",
        "gdef",
        "head",
        "hhea",
        "name",
        "vhea",
        "vmtx",
    ]
)


def asFea(g):
    if hasattr(g, "asFea"):
        return g.asFea()
    elif isinstance(g, tuple) and len(g) == 2:
        return asFea(g[0]) + " - " + asFea(g[1])  # a range
    elif g.lower() in fea_keywords:
        return "\\" + g
    else:
        return g


class Element(object):
    """A base class representing "something" in a feature file."""

    def __init__(self, location=None):
        #: location of this element as a `FeatureLibLocation` object.
        if location and not isinstance(location, FeatureLibLocation):
            location = FeatureLibLocation(*location)
        self.location = location

    def build(self, builder):
        pass

    def asFea(self, indent=""):
        """Returns this element as a string of feature code. For block-type
        elements (such as :class:`FeatureBlock`), the `indent` string is
        added to the start of each line in the output."""
        raise NotImplementedError

    def __str__(self):
        return self.asFea()


class Statement(Element):
    pass


class Expression(Element):
    pass


class Comment(Element):
    """A comment in a feature file."""

    def __init__(self, text, location=None):
        super(Comment, self).__init__(location)
        #: Text of the comment
        self.text = text

    def asFea(self, indent=""):
        return self.text


class NullGlyph(Expression):
    """The NULL glyph, used in glyph deletion substitutions."""

    def __init__(self, location=None):
        Expression.__init__(self, location)
        #: The name itself as a string

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return ()

    def asFea(self, indent=""):
        return "NULL"


class GlyphName(Expression):
    """A single glyph name, such as ``cedilla``."""

    def __init__(self, glyph, location=None):
        Expression.__init__(self, location)
        #: The name itself as a string
        self.glyph = glyph

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return (self.glyph,)

    def asFea(self, indent=""):
        return asFea(self.glyph)


class GlyphClass(Expression):
    """A glyph class, such as ``[acute cedilla grave]``."""

    def __init__(self, glyphs=None, location=None):
        Expression.__init__(self, location)
        #: The list of glyphs in this class, as :class:`GlyphName` objects.
        self.glyphs = glyphs if glyphs is not None else []
        self.original = []
        self.curr = 0

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphs)

    def asFea(self, indent=""):
        if len(self.original):
            if self.curr < len(self.glyphs):
                self.original.extend(self.glyphs[self.curr :])
                self.curr = len(self.glyphs)
            return "[" + " ".join(map(asFea, self.original)) + "]"
        else:
            return "[" + " ".join(map(asFea, self.glyphs)) + "]"

    def extend(self, glyphs):
        """Add a list of :class:`GlyphName` objects to the class."""
        self.glyphs.extend(glyphs)

    def append(self, glyph):
        """Add a single :class:`GlyphName` object to the class."""
        self.glyphs.append(glyph)

    def add_range(self, start, end, glyphs):
        """Add a range (e.g. ``A-Z``) to the class. ``start`` and ``end``
        are either :class:`GlyphName` objects or strings representing the
        start and end glyphs in the class, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr :])
        self.original.append((start, end))
        self.glyphs.extend(glyphs)
        self.curr = len(self.glyphs)

    def add_cid_range(self, start, end, glyphs):
        """Add a range to the class by glyph ID. ``start`` and ``end`` are the
        initial and final IDs, and ``glyphs`` is the full list of
        :class:`GlyphName` objects in the range."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr :])
        self.original.append(("\\{}".format(start), "\\{}".format(end)))
        self.glyphs.extend(glyphs)
        self.curr = len(self.glyphs)

    def add_class(self, gc):
        """Add glyphs from the given :class:`GlyphClassName` object to the
        class."""
        if self.curr < len(self.glyphs):
            self.original.extend(self.glyphs[self.curr :])
        self.original.append(gc)
        self.glyphs.extend(gc.glyphSet())
        self.curr = len(self.glyphs)


class GlyphClassName(Expression):
    """A glyph class name, such as ``@FRENCH_MARKS``. This must be instantiated
    with a :class:`GlyphClassDefinition` object."""

    def __init__(self, glyphclass, location=None):
        Expression.__init__(self, location)
        assert isinstance(glyphclass, GlyphClassDefinition)
        self.glyphclass = glyphclass

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphclass.glyphSet())

    def asFea(self, indent=""):
        return "@" + self.glyphclass.name


class MarkClassName(Expression):
    """A mark class name, such as ``@FRENCH_MARKS`` defined with ``markClass``.
    This must be instantiated with a :class:`MarkClass` object."""

    def __init__(self, markClass, location=None):
        Expression.__init__(self, location)
        assert isinstance(markClass, MarkClass)
        self.markClass = markClass

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return self.markClass.glyphSet()

    def asFea(self, indent=""):
        return "@" + self.markClass.name


class AnonymousBlock(Statement):
    """An anonymous data block."""

    def __init__(self, tag, content, location=None):
        Statement.__init__(self, location)
        self.tag = tag  #: string containing the block's "tag"
        self.content = content  #: block data as string

    def asFea(self, indent=""):
        res = "anon {} {{\n".format(self.tag)
        res += self.content
        res += "}} {};\n\n".format(self.tag)
        return res


class Block(Statement):
    """A block of statements: feature, lookup, etc."""

    def __init__(self, location=None):
        Statement.__init__(self, location)
        self.statements = []  #: Statements contained in the block

    def build(self, builder):
        """When handed a 'builder' object of comparable interface to
        :class:`fontTools.feaLib.builder`, walks the statements in this
        block, calling the builder callbacks."""
        for s in self.statements:
            s.build(builder)

    def asFea(self, indent=""):
        indent += SHIFT
        return (
            indent
            + ("\n" + indent).join([s.asFea(indent=indent) for s in self.statements])
            + "\n"
        )


class FeatureFile(Block):
    """The top-level element of the syntax tree, containing the whole feature
    file in its ``statements`` attribute."""

    def __init__(self):
        Block.__init__(self, location=None)
        self.markClasses = {}  # name --> ast.MarkClass

    def asFea(self, indent=""):
        return "\n".join(s.asFea(indent=indent) for s in self.statements)


class FeatureBlock(Block):
    """A named feature block."""

    def __init__(self, name, use_extension=False, location=None):
        Block.__init__(self, location)
        self.name, self.use_extension = name, use_extension

    def build(self, builder):
        """Call the ``start_feature`` callback on the builder object, visit
        all the statements in this feature, and then call ``end_feature``."""
        builder.start_feature(self.location, self.name, self.use_extension)
        # language exclude_dflt statements modify builder.features_
        # limit them to this block with temporary builder.features_
        features = builder.features_
        builder.features_ = {}
        Block.build(self, builder)
        for key, value in builder.features_.items():
            features.setdefault(key, []).extend(value)
        builder.features_ = features
        builder.end_feature()

    def asFea(self, indent=""):
        res = indent + "feature %s " % self.name.strip()
        if self.use_extension:
            res += "useExtension "
        res += "{\n"
        res += Block.asFea(self, indent=indent)
        res += indent + "} %s;\n" % self.name.strip()
        return res


class NestedBlock(Block):
    """A block inside another block, for example when found inside a
    ``cvParameters`` block."""

    def __init__(self, tag, block_name, location=None):
        Block.__init__(self, location)
        self.tag = tag
        self.block_name = block_name

    def build(self, builder):
        Block.build(self, builder)
        if self.block_name == "ParamUILabelNameID":
            builder.add_to_cv_num_named_params(self.tag)

    def asFea(self, indent=""):
        res = "{}{} {{\n".format(indent, self.block_name)
        res += Block.asFea(self, indent=indent)
        res += "{}}};\n".format(indent)
        return res


class LookupBlock(Block):
    """A named lookup, containing ``statements``."""

    def __init__(self, name, use_extension=False, location=None):
        Block.__init__(self, location)
        self.name, self.use_extension = name, use_extension

    def build(self, builder):
        builder.start_lookup_block(self.location, self.name, self.use_extension)
        Block.build(self, builder)
        builder.end_lookup_block()

    def asFea(self, indent=""):
        res = "lookup {} ".format(self.name)
        if self.use_extension:
            res += "useExtension "
        res += "{\n"
        res += Block.asFea(self, indent=indent)
        res += "{}}} {};\n".format(indent, self.name)
        return res


class TableBlock(Block):
    """A ``table ... { }`` block."""

    def __init__(self, name, location=None):
        Block.__init__(self, location)
        self.name = name

    def asFea(self, indent=""):
        res = "table {} {{\n".format(self.name.strip())
        res += super(TableBlock, self).asFea(indent=indent)
        res += "}} {};\n".format(self.name.strip())
        return res


class GlyphClassDefinition(Statement):
    """Example: ``@UPPERCASE = [A-Z];``."""

    def __init__(self, name, glyphs, location=None):
        Statement.__init__(self, location)
        self.name = name  #: class name as a string, without initial ``@``
        self.glyphs = glyphs  #: a :class:`GlyphClass` object

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphs.glyphSet())

    def asFea(self, indent=""):
        return "@" + self.name + " = " + self.glyphs.asFea() + ";"


class GlyphClassDefStatement(Statement):
    """Example: ``GlyphClassDef @UPPERCASE, [B], [C], [D];``. The parameters
    must be either :class:`GlyphClass` or :class:`GlyphClassName` objects, or
    ``None``."""

    def __init__(
        self, baseGlyphs, markGlyphs, ligatureGlyphs, componentGlyphs, location=None
    ):
        Statement.__init__(self, location)
        self.baseGlyphs, self.markGlyphs = (baseGlyphs, markGlyphs)
        self.ligatureGlyphs = ligatureGlyphs
        self.componentGlyphs = componentGlyphs

    def build(self, builder):
        """Calls the builder's ``add_glyphClassDef`` callback."""
        base = self.baseGlyphs.glyphSet() if self.baseGlyphs else tuple()
        liga = self.ligatureGlyphs.glyphSet() if self.ligatureGlyphs else tuple()
        mark = self.markGlyphs.glyphSet() if self.markGlyphs else tuple()
        comp = self.componentGlyphs.glyphSet() if self.componentGlyphs else tuple()
        builder.add_glyphClassDef(self.location, base, liga, mark, comp)

    def asFea(self, indent=""):
        return "GlyphClassDef {}, {}, {}, {};".format(
            self.baseGlyphs.asFea() if self.baseGlyphs else "",
            self.ligatureGlyphs.asFea() if self.ligatureGlyphs else "",
            self.markGlyphs.asFea() if self.markGlyphs else "",
            self.componentGlyphs.asFea() if self.componentGlyphs else "",
        )


class MarkClass(object):
    """One `or more` ``markClass`` statements for the same mark class.

    While glyph classes can be defined only once, the feature file format
    allows expanding mark classes with multiple definitions, each using
    different glyphs and anchors. The following are two ``MarkClassDefinitions``
    for the same ``MarkClass``::

        markClass [acute grave] <anchor 350 800> @FRENCH_ACCENTS;
        markClass [cedilla] <anchor 350 -200> @FRENCH_ACCENTS;

    The ``MarkClass`` object is therefore just a container for a list of
    :class:`MarkClassDefinition` statements.
    """

    def __init__(self, name):
        self.name = name
        self.definitions = []
        self.glyphs = OrderedDict()  # glyph --> ast.MarkClassDefinitions

    def addDefinition(self, definition):
        """Add a :class:`MarkClassDefinition` statement to this mark class."""
        assert isinstance(definition, MarkClassDefinition)
        self.definitions.append(weakref.proxy(definition))
        for glyph in definition.glyphSet():
            if glyph in self.glyphs:
                otherLoc = self.glyphs[glyph].location
                if otherLoc is None:
                    end = ""
                else:
                    end = f" at {otherLoc}"
                raise FeatureLibError(
                    "Glyph %s already defined%s" % (glyph, end), definition.location
                )
            self.glyphs[glyph] = definition

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return tuple(self.glyphs.keys())

    def asFea(self, indent=""):
        res = "\n".join(d.asFea() for d in self.definitions)
        return res


class MarkClassDefinition(Statement):
    """A single ``markClass`` statement. The ``markClass`` should be a
    :class:`MarkClass` object, the ``anchor`` an :class:`Anchor` object,
    and the ``glyphs`` parameter should be a `glyph-containing object`_ .

    Example:

        .. code:: python

            mc = MarkClass("FRENCH_ACCENTS")
            mc.addDefinition( MarkClassDefinition(mc, Anchor(350, 800),
                GlyphClass([ GlyphName("acute"), GlyphName("grave") ])
            ) )
            mc.addDefinition( MarkClassDefinition(mc, Anchor(350, -200),
                GlyphClass([ GlyphName("cedilla") ])
            ) )

            mc.asFea()
            # markClass [acute grave] <anchor 350 800> @FRENCH_ACCENTS;
            # markClass [cedilla] <anchor 350 -200> @FRENCH_ACCENTS;

    """

    def __init__(self, markClass, anchor, glyphs, location=None):
        Statement.__init__(self, location)
        assert isinstance(markClass, MarkClass)
        assert isinstance(anchor, Anchor) and isinstance(glyphs, Expression)
        self.markClass, self.anchor, self.glyphs = markClass, anchor, glyphs

    def glyphSet(self):
        """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
        return self.glyphs.glyphSet()

    def asFea(self, indent=""):
        return "markClass {} {} @{};".format(
            self.glyphs.asFea(), self.anchor.asFea(), self.markClass.name
        )


class AlternateSubstStatement(Statement):
    """A ``sub ... from ...`` statement.

    ``glyph`` and ``replacement`` should be `glyph-containing objects`_.
    ``prefix`` and ``suffix`` should be lists of `glyph-containing objects`_."""

    def __init__(self, prefix, glyph, suffix, replacement, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyph, self.suffix = (prefix, glyph, suffix)
        self.replacement = replacement

    def build(self, builder):
        """Calls the builder's ``add_alternate_subst`` callback."""
        glyph = self.glyph.glyphSet()
        assert len(glyph) == 1, glyph
        glyph = list(glyph)[0]
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        replacement = self.replacement.glyphSet()
        builder.add_alternate_subst(self.location, prefix, glyph, suffix, replacement)

    def asFea(self, indent=""):
        res = "sub "
        if len(self.prefix) or len(self.suffix):
            if len(self.prefix):
                res += " ".join(map(asFea, self.prefix)) + " "
            res += asFea(self.glyph) + "'"  # even though we really only use 1
            if len(self.suffix):
                res += " " + " ".join(map(asFea, self.suffix))
        else:
            res += asFea(self.glyph)
        res += " from "
        res += asFea(self.replacement)
        res += ";"
        return res


class Anchor(Expression):
    """An ``Anchor`` element, used inside a ``pos`` rule.

    If a ``name`` is given, this will be used in preference to the coordinates.
    Other values should be integer.
    """

    def __init__(
        self,
        x,
        y,
        name=None,
        contourpoint=None,
        xDeviceTable=None,
        yDeviceTable=None,
        location=None,
    ):
        Expression.__init__(self, location)
        self.name = name
        self.x, self.y, self.contourpoint = x, y, contourpoint
        self.xDeviceTable, self.yDeviceTable = xDeviceTable, yDeviceTable

    def asFea(self, indent=""):
        if self.name is not None:
            return "<anchor {}>".format(self.name)
        res = "<anchor {} {}".format(self.x, self.y)
        if self.contourpoint:
            res += " contourpoint {}".format(self.contourpoint)
        if self.xDeviceTable or self.yDeviceTable:
            res += " "
            res += deviceToString(self.xDeviceTable)
            res += " "
            res += deviceToString(self.yDeviceTable)
        res += ">"
        return res


class AnchorDefinition(Statement):
    """A named anchor definition. (2.e.viii). ``name`` should be a string."""

    def __init__(self, name, x, y, contourpoint=None, location=None):
        Statement.__init__(self, location)
        self.name, self.x, self.y, self.contourpoint = name, x, y, contourpoint

    def asFea(self, indent=""):
        res = "anchorDef {} {}".format(self.x, self.y)
        if self.contourpoint:
            res += " contourpoint {}".format(self.contourpoint)
        res += " {};".format(self.name)
        return res


class AttachStatement(Statement):
    """A ``GDEF`` table ``Attach`` statement."""

    def __init__(self, glyphs, contourPoints, location=None):
        Statement.__init__(self, location)
        self.glyphs = glyphs  #: A `glyph-containing object`_
        self.contourPoints = contourPoints  #: A list of integer contour points

    def build(self, builder):
        """Calls the builder's ``add_attach_points`` callback."""
        glyphs = self.glyphs.glyphSet()
        builder.add_attach_points(self.location, glyphs, self.contourPoints)

    def asFea(self, indent=""):
        return "Attach {} {};".format(
            self.glyphs.asFea(), " ".join(str(c) for c in self.contourPoints)
        )


class ChainContextPosStatement(Statement):
    r"""A chained contextual positioning statement.

    ``prefix``, ``glyphs``, and ``suffix`` should be lists of
    `glyph-containing objects`_ .

    ``lookups`` should be a list of elements representing what lookups
    to apply at each glyph position. Each element should be a
    :class:`LookupBlock` to apply a single chaining lookup at the given
    position, a list of :class:`LookupBlock`\ s to apply multiple
    lookups, or ``None`` to apply no lookup. The length of the outer
    list should equal the length of ``glyphs``; the inner lists can be
    of variable length."""

    def __init__(self, prefix, glyphs, suffix, lookups, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyphs, self.suffix = prefix, glyphs, suffix
        self.lookups = list(lookups)
        for i, lookup in enumerate(lookups):
            if lookup:
                try:
                    iter(lookup)
                except TypeError:
                    self.lookups[i] = [lookup]

    def build(self, builder):
        """Calls the builder's ``add_chain_context_pos`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        glyphs = [g.glyphSet() for g in self.glyphs]
        suffix = [s.glyphSet() for s in self.suffix]
        builder.add_chain_context_pos(
            self.location, prefix, glyphs, suffix, self.lookups
        )

    def asFea(self, indent=""):
        res = "pos "
        if (
            len(self.prefix)
            or len(self.suffix)
            or any([x is not None for x in self.lookups])
        ):
            if len(self.prefix):
                res += " ".join(g.asFea() for g in self.prefix) + " "
            for i, g in enumerate(self.glyphs):
                res += g.asFea() + "'"
                if self.lookups[i]:
                    for lu in self.lookups[i]:
                        res += " lookup " + lu.name
                if i < len(self.glyphs) - 1:
                    res += " "
            if len(self.suffix):
                res += " " + " ".join(map(asFea, self.suffix))
        else:
            res += " ".join(map(asFea, self.glyphs))
        res += ";"
        return res


class ChainContextSubstStatement(Statement):
    r"""A chained contextual substitution statement.

    ``prefix``, ``glyphs``, and ``suffix`` should be lists of
    `glyph-containing objects`_ .

    ``lookups`` should be a list of elements representing what lookups
    to apply at each glyph position. Each element should be a
    :class:`LookupBlock` to apply a single chaining lookup at the given
    position, a list of :class:`LookupBlock`\ s to apply multiple
    lookups, or ``None`` to apply no lookup. The length of the outer
    list should equal the length of ``glyphs``; the inner lists can be
    of variable length."""

    def __init__(self, prefix, glyphs, suffix, lookups, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyphs, self.suffix = prefix, glyphs, suffix
        self.lookups = list(lookups)
        for i, lookup in enumerate(lookups):
            if lookup:
                try:
                    iter(lookup)
                except TypeError:
                    self.lookups[i] = [lookup]

    def build(self, builder):
        """Calls the builder's ``add_chain_context_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        glyphs = [g.glyphSet() for g in self.glyphs]
        suffix = [s.glyphSet() for s in self.suffix]
        builder.add_chain_context_subst(
            self.location, prefix, glyphs, suffix, self.lookups
        )

    def asFea(self, indent=""):
        res = "sub "
        if (
            len(self.prefix)
            or len(self.suffix)
            or any([x is not None for x in self.lookups])
        ):
            if len(self.prefix):
                res += " ".join(g.asFea() for g in self.prefix) + " "
            for i, g in enumerate(self.glyphs):
                res += g.asFea() + "'"
                if self.lookups[i]:
                    for lu in self.lookups[i]:
                        res += " lookup " + lu.name
                if i < len(self.glyphs) - 1:
                    res += " "
            if len(self.suffix):
                res += " " + " ".join(map(asFea, self.suffix))
        else:
            res += " ".join(map(asFea, self.glyphs))
        res += ";"
        return res


class CursivePosStatement(Statement):
    """A cursive positioning statement. Entry and exit anchors can either
    be :class:`Anchor` objects or ``None``."""

    def __init__(self, glyphclass, entryAnchor, exitAnchor, location=None):
        Statement.__init__(self, location)
        self.glyphclass = glyphclass
        self.entryAnchor, self.exitAnchor = entryAnchor, exitAnchor

    def build(self, builder):
        """Calls the builder object's ``add_cursive_pos`` callback."""
        builder.add_cursive_pos(
            self.location, self.glyphclass.glyphSet(), self.entryAnchor, self.exitAnchor
        )

    def asFea(self, indent=""):
        entry = self.entryAnchor.asFea() if self.entryAnchor else "<anchor NULL>"
        exit = self.exitAnchor.asFea() if self.exitAnchor else "<anchor NULL>"
        return "pos cursive {} {} {};".format(self.glyphclass.asFea(), entry, exit)


class FeatureReferenceStatement(Statement):
    """Example: ``feature salt;``"""

    def __init__(self, featureName, location=None):
        Statement.__init__(self, location)
        self.location, self.featureName = (location, featureName)

    def build(self, builder):
        """Calls the builder object's ``add_feature_reference`` callback."""
        builder.add_feature_reference(self.location, self.featureName)

    def asFea(self, indent=""):
        return "feature {};".format(self.featureName)


class IgnorePosStatement(Statement):
    """An ``ignore pos`` statement, containing `one or more` contexts to ignore.

    ``chainContexts`` should be a list of ``(prefix, glyphs, suffix)`` tuples,
    with each of ``prefix``, ``glyphs`` and ``suffix`` being
    `glyph-containing objects`_ ."""

    def __init__(self, chainContexts, location=None):
        Statement.__init__(self, location)
        self.chainContexts = chainContexts

    def build(self, builder):
        """Calls the builder object's ``add_chain_context_pos`` callback on each
        rule context."""
        for prefix, glyphs, suffix in self.chainContexts:
            prefix = [p.glyphSet() for p in prefix]
            glyphs = [g.glyphSet() for g in glyphs]
            suffix = [s.glyphSet() for s in suffix]
            builder.add_chain_context_pos(self.location, prefix, glyphs, suffix, [])

    def asFea(self, indent=""):
        contexts = []
        for prefix, glyphs, suffix in self.chainContexts:
            res = ""
            if len(prefix) or len(suffix):
                if len(prefix):
                    res += " ".join(map(asFea, prefix)) + " "
                res += " ".join(g.asFea() + "'" for g in glyphs)
                if len(suffix):
                    res += " " + " ".join(map(asFea, suffix))
            else:
                res += " ".join(map(asFea, glyphs))
            contexts.append(res)
        return "ignore pos " + ", ".join(contexts) + ";"


class IgnoreSubstStatement(Statement):
    """An ``ignore sub`` statement, containing `one or more` contexts to ignore.

    ``chainContexts`` should be a list of ``(prefix, glyphs, suffix)`` tuples,
    with each of ``prefix``, ``glyphs`` and ``suffix`` being
    `glyph-containing objects`_ ."""

    def __init__(self, chainContexts, location=None):
        Statement.__init__(self, location)
        self.chainContexts = chainContexts

    def build(self, builder):
        """Calls the builder object's ``add_chain_context_subst`` callback on
        each rule context."""
        for prefix, glyphs, suffix in self.chainContexts:
            prefix = [p.glyphSet() for p in prefix]
            glyphs = [g.glyphSet() for g in glyphs]
            suffix = [s.glyphSet() for s in suffix]
            builder.add_chain_context_subst(self.location, prefix, glyphs, suffix, [])

    def asFea(self, indent=""):
        contexts = []
        for prefix, glyphs, suffix in self.chainContexts:
            res = ""
            if len(prefix):
                res += " ".join(map(asFea, prefix)) + " "
            res += " ".join(g.asFea() + "'" for g in glyphs)
            if len(suffix):
                res += " " + " ".join(map(asFea, suffix))
            contexts.append(res)
        return "ignore sub " + ", ".join(contexts) + ";"


class IncludeStatement(Statement):
    """An ``include()`` statement."""

    def __init__(self, filename, location=None):
        super(IncludeStatement, self).__init__(location)
        self.filename = filename  #: String containing name of file to include

    def build(self):
        # TODO: consider lazy-loading the including parser/lexer?
        raise FeatureLibError(
            "Building an include statement is not implemented yet. "
            "Instead, use Parser(..., followIncludes=True) for building.",
            self.location,
        )

    def asFea(self, indent=""):
        return indent + "include(%s);" % self.filename


class LanguageStatement(Statement):
    """A ``language`` statement within a feature."""

    def __init__(self, language, include_default=True, required=False, location=None):
        Statement.__init__(self, location)
        assert len(language) == 4
        self.language = language  #: A four-character language tag
        self.include_default = include_default  #: If false, "exclude_dflt"
        self.required = required

    def build(self, builder):
        """Call the builder object's ``set_language`` callback."""
        builder.set_language(
            location=self.location,
            language=self.language,
            include_default=self.include_default,
            required=self.required,
        )

    def asFea(self, indent=""):
        res = "language {}".format(self.language.strip())
        if not self.include_default:
            res += " exclude_dflt"
        if self.required:
            res += " required"
        res += ";"
        return res


class LanguageSystemStatement(Statement):
    """A top-level ``languagesystem`` statement."""

    def __init__(self, script, language, location=None):
        Statement.__init__(self, location)
        self.script, self.language = (script, language)

    def build(self, builder):
        """Calls the builder object's ``add_language_system`` callback."""
        builder.add_language_system(self.location, self.script, self.language)

    def asFea(self, indent=""):
        return "languagesystem {} {};".format(self.script, self.language.strip())


class FontRevisionStatement(Statement):
    """A ``head`` table ``FontRevision`` statement. ``revision`` should be a
    number, and will be formatted to three significant decimal places."""

    def __init__(self, revision, location=None):
        Statement.__init__(self, location)
        self.revision = revision

    def build(self, builder):
        builder.set_font_revision(self.location, self.revision)

    def asFea(self, indent=""):
        return "FontRevision {:.3f};".format(self.revision)


class LigatureCaretByIndexStatement(Statement):
    """A ``GDEF`` table ``LigatureCaretByIndex`` statement. ``glyphs`` should be
    a `glyph-containing object`_, and ``carets`` should be a list of integers."""

    def __init__(self, glyphs, carets, location=None):
        Statement.__init__(self, location)
        self.glyphs, self.carets = (glyphs, carets)

    def build(self, builder):
        """Calls the builder object's ``add_ligatureCaretByIndex_`` callback."""
        glyphs = self.glyphs.glyphSet()
        builder.add_ligatureCaretByIndex_(self.location, glyphs, set(self.carets))

    def asFea(self, indent=""):
        return "LigatureCaretByIndex {} {};".format(
            self.glyphs.asFea(), " ".join(str(x) for x in self.carets)
        )


class LigatureCaretByPosStatement(Statement):
    """A ``GDEF`` table ``LigatureCaretByPos`` statement. ``glyphs`` should be
    a `glyph-containing object`_, and ``carets`` should be a list of integers."""

    def __init__(self, glyphs, carets, location=None):
        Statement.__init__(self, location)
        self.glyphs, self.carets = (glyphs, carets)

    def build(self, builder):
        """Calls the builder object's ``add_ligatureCaretByPos_`` callback."""
        glyphs = self.glyphs.glyphSet()
        builder.add_ligatureCaretByPos_(self.location, glyphs, set(self.carets))

    def asFea(self, indent=""):
        return "LigatureCaretByPos {} {};".format(
            self.glyphs.asFea(), " ".join(str(x) for x in self.carets)
        )


class LigatureSubstStatement(Statement):
    """A chained contextual substitution statement.

    ``prefix``, ``glyphs``, and ``suffix`` should be lists of
    `glyph-containing objects`_; ``replacement`` should be a single
    `glyph-containing object`_.

    If ``forceChain`` is True, this is expressed as a chaining rule
    (e.g. ``sub f' i' by f_i``) even when no context is given."""

    def __init__(self, prefix, glyphs, suffix, replacement, forceChain, location=None):
        Statement.__init__(self, location)
        self.prefix, self.glyphs, self.suffix = (prefix, glyphs, suffix)
        self.replacement, self.forceChain = replacement, forceChain

    def build(self, builder):
        prefix = [p.glyphSet() for p in self.prefix]
        glyphs = [g.glyphSet() for g in self.glyphs]
        suffix = [s.glyphSet() for s in self.suffix]
        builder.add_ligature_subst(
            self.location, prefix, glyphs, suffix, self.replacement, self.forceChain
        )

    def asFea(self, indent=""):
        res = "sub "
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += " ".join(g.asFea() for g in self.prefix) + " "
            res += " ".join(g.asFea() + "'" for g in self.glyphs)
            if len(self.suffix):
                res += " " + " ".join(g.asFea() for g in self.suffix)
        else:
            res += " ".join(g.asFea() for g in self.glyphs)
        res += " by "
        res += asFea(self.replacement)
        res += ";"
        return res


class LookupFlagStatement(Statement):
    """A ``lookupflag`` statement. The ``value`` should be an integer value
    representing the flags in use, but not including the ``markAttachment``
    class and ``markFilteringSet`` values, which must be specified as
    glyph-containing objects."""

    def __init__(
        self, value=0, markAttachment=None, markFilteringSet=None, location=None
    ):
        Statement.__init__(self, location)
        self.value = value
        self.markAttachment = markAttachment
        self.markFilteringSet = markFilteringSet

    def build(self, builder):
        """Calls the builder object's ``set_lookup_flag`` callback."""
        markAttach = None
        if self.markAttachment is not None:
            markAttach = self.markAttachment.glyphSet()
        markFilter = None
        if self.markFilteringSet is not None:
            markFilter = self.markFilteringSet.glyphSet()
        builder.set_lookup_flag(self.location, self.value, markAttach, markFilter)

    def asFea(self, indent=""):
        res = []
        flags = ["RightToLeft", "IgnoreBaseGlyphs", "IgnoreLigatures", "IgnoreMarks"]
        curr = 1
        for i in range(len(flags)):
            if self.value & curr != 0:
                res.append(flags[i])
            curr = curr << 1
        if self.markAttachment is not None:
            res.append("MarkAttachmentType {}".format(self.markAttachment.asFea()))
        if self.markFilteringSet is not None:
            res.append("UseMarkFilteringSet {}".format(self.markFilteringSet.asFea()))
        if not res:
            res = ["0"]
        return "lookupflag {};".format(" ".join(res))


class LookupReferenceStatement(Statement):
    """Represents a ``lookup ...;`` statement to include a lookup in a feature.

    The ``lookup`` should be a :class:`LookupBlock` object."""

    def __init__(self, lookup, location=None):
        Statement.__init__(self, location)
        self.location, self.lookup = (location, lookup)

    def build(self, builder):
        """Calls the builder object's ``add_lookup_call`` callback."""
        builder.add_lookup_call(self.lookup.name)

    def asFea(self, indent=""):
        return "lookup {};".format(self.lookup.name)


class MarkBasePosStatement(Statement):
    """A mark-to-base positioning rule. The ``base`` should be a
    `glyph-containing object`_. The ``marks`` should be a list of
    (:class:`Anchor`, :class:`MarkClass`) tuples."""

    def __init__(self, base, marks, location=None):
        Statement.__init__(self, location)
        self.base, self.marks = base, marks

    def build(self, builder):
        """Calls the builder object's ``add_mark_base_pos`` callback."""
        builder.add_mark_base_pos(self.location, self.base.glyphSet(), self.marks)

    def asFea(self, indent=""):
        res = "pos base {}".format(self.base.asFea())
        for a, m in self.marks:
            res += "\n" + indent + SHIFT + "{} mark @{}".format(a.asFea(), m.name)
        res += ";"
        return res


class MarkLigPosStatement(Statement):
    """A mark-to-ligature positioning rule. The ``ligatures`` must be a
    `glyph-containing object`_. The ``marks`` should be a list of lists: each
    element in the top-level list represents a component glyph, and is made
    up of a list of (:class:`Anchor`, :class:`MarkClass`) tuples representing
    mark attachment points for that position.

    Example::

        m1 = MarkClass("TOP_MARKS")
        m2 = MarkClass("BOTTOM_MARKS")
        # ... add definitions to mark classes...

        glyph = GlyphName("lam_meem_jeem")
        marks = [
            [ (Anchor(625,1800), m1) ], # Attachments on 1st component (lam)
            [ (Anchor(376,-378), m2) ], # Attachments on 2nd component (meem)
            [ ]                         # No attachments on the jeem
        ]
        mlp = MarkLigPosStatement(glyph, marks)

        mlp.asFea()
        # pos ligature lam_meem_jeem <anchor 625 1800> mark @TOP_MARKS
        # ligComponent <anchor 376 -378> mark @BOTTOM_MARKS;

    """

    def __init__(self, ligatures, marks, location=None):
        Statement.__init__(self, location)
        self.ligatures, self.marks = ligatures, marks

    def build(self, builder):
        """Calls the builder object's ``add_mark_lig_pos`` callback."""
        builder.add_mark_lig_pos(self.location, self.ligatures.glyphSet(), self.marks)

    def asFea(self, indent=""):
        res = "pos ligature {}".format(self.ligatures.asFea())
        ligs = []
        for l in self.marks:
            temp = ""
            if l is None or not len(l):
                temp = "\n" + indent + SHIFT * 2 + "<anchor NULL>"
            else:
                for a, m in l:
                    temp += (
                        "\n"
                        + indent
                        + SHIFT * 2
                        + "{} mark @{}".format(a.asFea(), m.name)
                    )
            ligs.append(temp)
        res += ("\n" + indent + SHIFT + "ligComponent").join(ligs)
        res += ";"
        return res


class MarkMarkPosStatement(Statement):
    """A mark-to-mark positioning rule. The ``baseMarks`` must be a
    `glyph-containing object`_. The ``marks`` should be a list of
    (:class:`Anchor`, :class:`MarkClass`) tuples."""

    def __init__(self, baseMarks, marks, location=None):
        Statement.__init__(self, location)
        self.baseMarks, self.marks = baseMarks, marks

    def build(self, builder):
        """Calls the builder object's ``add_mark_mark_pos`` callback."""
        builder.add_mark_mark_pos(self.location, self.baseMarks.glyphSet(), self.marks)

    def asFea(self, indent=""):
        res = "pos mark {}".format(self.baseMarks.asFea())
        for a, m in self.marks:
            res += "\n" + indent + SHIFT + "{} mark @{}".format(a.asFea(), m.name)
        res += ";"
        return res


class MultipleSubstStatement(Statement):
    """A multiple substitution statement.

    Args:
        prefix: a list of `glyph-containing objects`_.
        glyph: a single glyph-containing object.
        suffix: a list of glyph-containing objects.
        replacement: a list of glyph-containing objects.
        forceChain: If true, the statement is expressed as a chaining rule
            (e.g. ``sub f' i' by f_i``) even when no context is given.
    """

    def __init__(
        self, prefix, glyph, suffix, replacement, forceChain=False, location=None
    ):
        Statement.__init__(self, location)
        self.prefix, self.glyph, self.suffix = prefix, glyph, suffix
        self.replacement = replacement
        self.forceChain = forceChain

    def build(self, builder):
        """Calls the builder object's ``add_multiple_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        if hasattr(self.glyph, "glyphSet"):
            originals = self.glyph.glyphSet()
        else:
            originals = [self.glyph]
        count = len(originals)
        replaces = []
        for r in self.replacement:
            if hasattr(r, "glyphSet"):
                replace = r.glyphSet()
            else:
                replace = [r]
            if len(replace) == 1 and len(replace) != count:
                replace = replace * count
            replaces.append(replace)
        replaces = list(zip(*replaces))

        seen_originals = set()
        for i, original in enumerate(originals):
            if original not in seen_originals:
                seen_originals.add(original)
                builder.add_multiple_subst(
                    self.location,
                    prefix,
                    original,
                    suffix,
                    replaces and replaces[i] or (),
                    self.forceChain,
                )

    def asFea(self, indent=""):
        res = "sub "
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += " ".join(map(asFea, self.prefix)) + " "
            res += asFea(self.glyph) + "'"
            if len(self.suffix):
                res += " " + " ".join(map(asFea, self.suffix))
        else:
            res += asFea(self.glyph)
        replacement = self.replacement or [NullGlyph()]
        res += " by "
        res += " ".join(map(asFea, replacement))
        res += ";"
        return res


class PairPosStatement(Statement):
    """A pair positioning statement.

    ``glyphs1`` and ``glyphs2`` should be `glyph-containing objects`_.
    ``valuerecord1`` should be a :class:`ValueRecord` object;
    ``valuerecord2`` should be either a :class:`ValueRecord` object or ``None``.
    If ``enumerated`` is true, then this is expressed as an
    `enumerated pair <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#6.b.ii>`_.
    """

    def __init__(
        self,
        glyphs1,
        valuerecord1,
        glyphs2,
        valuerecord2,
        enumerated=False,
        location=None,
    ):
        Statement.__init__(self, location)
        self.enumerated = enumerated
        self.glyphs1, self.valuerecord1 = glyphs1, valuerecord1
        self.glyphs2, self.valuerecord2 = glyphs2, valuerecord2

    def build(self, builder):
        """Calls a callback on the builder object:

        * If the rule is enumerated, calls ``add_specific_pair_pos`` on each
          combination of first and second glyphs.
        * If the glyphs are both single :class:`GlyphName` objects, calls
          ``add_specific_pair_pos``.
        * Else, calls ``add_class_pair_pos``.
        """
        if self.enumerated:
            g = [self.glyphs1.glyphSet(), self.glyphs2.glyphSet()]
            seen_pair = False
            for glyph1, glyph2 in itertools.product(*g):
                seen_pair = True
                builder.add_specific_pair_pos(
                    self.location, glyph1, self.valuerecord1, glyph2, self.valuerecord2
                )
            if not seen_pair:
                raise FeatureLibError(
                    "Empty glyph class in positioning rule", self.location
                )
            return

        is_specific = isinstance(self.glyphs1, GlyphName) and isinstance(
            self.glyphs2, GlyphName
        )
        if is_specific:
            builder.add_specific_pair_pos(
                self.location,
                self.glyphs1.glyph,
                self.valuerecord1,
                self.glyphs2.glyph,
                self.valuerecord2,
            )
        else:
            builder.add_class_pair_pos(
                self.location,
                self.glyphs1.glyphSet(),
                self.valuerecord1,
                self.glyphs2.glyphSet(),
                self.valuerecord2,
            )

    def asFea(self, indent=""):
        res = "enum " if self.enumerated else ""
        if self.valuerecord2:
            res += "pos {} {} {} {};".format(
                self.glyphs1.asFea(),
                self.valuerecord1.asFea(),
                self.glyphs2.asFea(),
                self.valuerecord2.asFea(),
            )
        else:
            res += "pos {} {} {};".format(
                self.glyphs1.asFea(), self.glyphs2.asFea(), self.valuerecord1.asFea()
            )
        return res


class ReverseChainSingleSubstStatement(Statement):
    """A reverse chaining substitution statement. You don't see those every day.

    Note the unusual argument order: ``suffix`` comes `before` ``glyphs``.
    ``old_prefix``, ``old_suffix``, ``glyphs`` and ``replacements`` should be
    lists of `glyph-containing objects`_. ``glyphs`` and ``replacements`` should
    be one-item lists.
    """

    def __init__(self, old_prefix, old_suffix, glyphs, replacements, location=None):
        Statement.__init__(self, location)
        self.old_prefix, self.old_suffix = old_prefix, old_suffix
        self.glyphs = glyphs
        self.replacements = replacements

    def build(self, builder):
        prefix = [p.glyphSet() for p in self.old_prefix]
        suffix = [s.glyphSet() for s in self.old_suffix]
        originals = self.glyphs[0].glyphSet()
        replaces = self.replacements[0].glyphSet()
        if len(replaces) == 1:
            replaces = replaces * len(originals)
        builder.add_reverse_chain_single_subst(
            self.location, prefix, suffix, dict(zip(originals, replaces))
        )

    def asFea(self, indent=""):
        res = "rsub "
        if len(self.old_prefix) or len(self.old_suffix):
            if len(self.old_prefix):
                res += " ".join(asFea(g) for g in self.old_prefix) + " "
            res += " ".join(asFea(g) + "'" for g in self.glyphs)
            if len(self.old_suffix):
                res += " " + " ".join(asFea(g) for g in self.old_suffix)
        else:
            res += " ".join(map(asFea, self.glyphs))
        res += " by {};".format(" ".join(asFea(g) for g in self.replacements))
        return res


class SingleSubstStatement(Statement):
    """A single substitution statement.

    Note the unusual argument order: ``prefix`` and suffix come `after`
