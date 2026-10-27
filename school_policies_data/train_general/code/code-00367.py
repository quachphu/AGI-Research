      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim
      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan
      coth inf max tanh""".split())

    _ambi_delims = set(r"""
      | \| / \backslash \uparrow \downarrow \updownarrow \Uparrow
      \Downarrow \Updownarrow . \vert \Vert""".split())
    _left_delims = set(r"""
      ( [ \{ < \lfloor \langle \lceil \lbrace \leftbrace \lbrack \leftparen \lgroup
      """.split())
    _right_delims = set(r"""
      ) ] \} > \rfloor \rangle \rceil \rbrace \rightbrace \rbrack \rightparen \rgroup
      """.split())
    _delims = _left_delims | _right_delims | _ambi_delims

    _small_greek = set([unicodedata.name(chr(i)).split()[-1].lower() for i in
                       range(ord('\N{GREEK SMALL LETTER ALPHA}'),
                             ord('\N{GREEK SMALL LETTER OMEGA}') + 1)])
    _latin_alphabets = set(string.ascii_letters)

    def __init__(self) -> None:
        p = types.SimpleNamespace()

        def set_names_and_parse_actions() -> None:
            for key, val in vars(p).items():
                if not key.startswith('_'):
                    # Set names on (almost) everything -- very useful for debugging
                    # token, placeable, and auto_delim are forward references which
                    # are left without names to ensure useful error messages
                    if key not in ("token", "placeable", "auto_delim"):
                        val.set_name(key)
                    # Set actions
                    if hasattr(self, key):
                        val.set_parse_action(getattr(self, key))

        # Root definitions.

        # In TeX parlance, a csname is a control sequence name (a "\foo").
        def csnames(group: str, names: Iterable[str]) -> Regex:
            ends_with_alpha = []
            ends_with_nonalpha = []
            for name in names:
                if name[-1].isalpha():
                    ends_with_alpha.append(name)
                else:
                    ends_with_nonalpha.append(name)
            return Regex(
                r"\\(?P<{group}>(?:{alpha})(?![A-Za-z]){additional}{nonalpha})".format(
                    group=group,
                    alpha="|".join(map(re.escape, ends_with_alpha)),
                    additional="|" if ends_with_nonalpha else "",
                    nonalpha="|".join(map(re.escape, ends_with_nonalpha)),
                )
            )

        p.float_literal  = Regex(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)")
        p.space          = one_of(self._space_widths)("space")

        p.style_literal  = one_of(
            [str(e.value) for e in self._MathStyle])("style_literal")

        p.symbol         = Regex(
            r"[a-zA-Z0-9 +\-*/<>=:,.;!\?&'@()\[\]|\U00000080-\U0001ffff]"
            r"|\\[%${}\[\]_|]"
            + r"|\\(?:{})(?![A-Za-z])".format(
                "|".join(map(re.escape, tex2uni)))
        )("sym").leave_whitespace()
        p.unknown_symbol = Regex(r"\\[A-Za-z]+")("name")

        p.font           = csnames("font", self._fontnames)
        p.start_group    = Optional(r"\math" + one_of(self._fontnames)("font")) + "{"
        p.end_group      = Literal("}")

        p.delim          = one_of(self._delims)

        # Mutually recursive definitions.  (Minimizing the number of Forward
        # elements is important for speed.)
        p.auto_delim       = Forward()
        p.placeable        = Forward()
        p.named_placeable  = Forward()
        p.required_group   = Forward()
        p.optional_group   = Forward()
        p.token            = Forward()

        # Workaround for placable being part of a cycle of definitions
        # calling `p.placeable("name")` results in a copy, so not guaranteed
        # to get the definition added after it is used.
        # ref https://github.com/matplotlib/matplotlib/issues/25204
        # xref https://github.com/pyparsing/pyparsing/issues/95
        p.named_placeable <<= p.placeable

        set_names_and_parse_actions()  # for mutually recursive definitions.

        p.optional_group <<= "{" + ZeroOrMore(p.token)("group") + "}"
        p.required_group <<= "{" + OneOrMore(p.token)("group") + "}"

        p.customspace = cmd(r"\hspace", "{" + p.float_literal("space") + "}")

        p.accent = (
            csnames("accent", [*self._accent_map, *self._wide_accents])
            - p.named_placeable("sym"))

        p.function = csnames("name", self._function_names)

        p.group = p.start_group + ZeroOrMore(p.token)("group") + p.end_group
        p.unclosed_group = (p.start_group + ZeroOrMore(p.token)("group") + StringEnd())

        p.frac  = cmd(r"\frac", p.required_group("num") + p.required_group("den"))
        p.dfrac = cmd(r"\dfrac", p.required_group("num") + p.required_group("den"))
        p.binom = cmd(r"\binom", p.required_group("num") + p.required_group("den"))

        p.genfrac = cmd(
            r"\genfrac",
            "{" + Optional(p.delim)("ldelim") + "}"
            + "{" + Optional(p.delim)("rdelim") + "}"
            + "{" + p.float_literal("rulesize") + "}"
            + "{" + Optional(p.style_literal)("style") + "}"
            + p.required_group("num")
            + p.required_group("den"))

        p.sqrt = cmd(
            r"\sqrt{value}",
            Optional("[" + OneOrMore(NotAny("]") + p.token)("root") + "]")
            + p.required_group("value"))

        p.overline = cmd(r"\overline", p.required_group("body"))

        p.overset  = cmd(
            r"\overset",
            p.optional_group("annotation") + p.optional_group("body"))
        p.underset = cmd(
            r"\underset",
            p.optional_group("annotation") + p.optional_group("body"))

        p.text = cmd(r"\text", QuotedString('{', '\\', end_quote_char="}"))

        p.substack = cmd(r"\substack",
                           nested_expr(opener="{", closer="}",
                                       content=Group(OneOrMore(p.token)) +
                                       ZeroOrMore(Literal("\\\\").suppress()))("parts"))

        p.subsuper = (
            (Optional(p.placeable)("nucleus")
             + OneOrMore(one_of(["_", "^"]) - p.placeable)("subsuper")
             + Regex("'*")("apostrophes"))
            | Regex("'+")("apostrophes")
            | (p.named_placeable("nucleus") + Regex("'*")("apostrophes"))
        )

        p.simple = p.space | p.customspace | p.font | p.subsuper

        p.token <<= (
            p.simple
            | p.auto_delim
            | p.unclosed_group
            | p.unknown_symbol  # Must be last
        )

        p.operatorname = cmd(r"\operatorname", "{" + ZeroOrMore(p.simple)("name") + "}")

        p.boldsymbol = cmd(
            r"\boldsymbol", "{" + ZeroOrMore(p.simple)("value") + "}")

        p.placeable     <<= (
            p.accent     # Must be before symbol as all accents are symbols
            | p.symbol   # Must be second to catch all named symbols and single
                         # chars not in a group
            | p.function
            | p.operatorname
            | p.group
            | p.frac
            | p.dfrac
            | p.binom
            | p.genfrac
            | p.overset
            | p.underset
            | p.sqrt
            | p.overline
            | p.text
            | p.boldsymbol
            | p.substack
        )

        mdelim = r"\middle" - (p.delim("mdelim") | Error("Expected a delimiter"))
        p.auto_delim    <<= (
            r"\left" - (p.delim("left") | Error("Expected a delimiter"))
            + ZeroOrMore(p.simple | p.auto_delim | mdelim)("mid")
            + r"\right" - (p.delim("right") | Error("Expected a delimiter"))
        )

        # Leaf definitions.
        p.math          = OneOrMore(p.token)
        p.math_string   = QuotedString('$', '\\', unquote_results=False)
        p.non_math      = Regex(r"(?:(?:\\[$])|[^$])*").leave_whitespace()
        p.main          = (
            p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        )
        set_names_and_parse_actions()  # for leaf definitions.

        self._expression = p.main
        self._math_expression = p.math

        # To add space to nucleus operators after sub/superscripts
        self._in_subscript_or_superscript = False

    def parse(self, s: str, fonts_object: Fonts, fontsize: float, dpi: float) -> Hlist:
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
        self._state_stack = [
            ParserState(fonts_object, 'default', 'rm', fontsize, dpi)]
        self._em_width_cache: dict[tuple[str, float, float], float] = {}
        try:
            result = self._expression.parse_string(s)
        except ParseBaseException as err:
            # explain becomes a plain method on pyparsing 3 (err.explain(0)).
            raise ValueError("\n" + ParseException.explain(err, 0)) from None
        self._state_stack = []
        self._in_subscript_or_superscript = False
        # prevent operator spacing from leaking into a new expression
        self._em_width_cache = {}
        ParserElement.reset_cache()
        return T.cast(Hlist, result[0])  # Known return type from main.

    def get_state(self) -> ParserState:
        """Get the current `State` of the parser."""
        return self._state_stack[-1]

    def pop_state(self) -> None:
        """Pop a `State` off of the stack."""
        self._state_stack.pop()

    def push_state(self) -> None:
        """Push a new `State` onto the stack, copying the current state."""
        self._state_stack.append(self.get_state().copy())

    def main(self, toks: ParseResults) -> list[Hlist]:
        return [Hlist(toks.as_list())]

    def math_string(self, toks: ParseResults) -> ParseResults:
        return self._math_expression.parse_string(toks[0][1:-1], parse_all=True)

    def math(self, toks: ParseResults) -> T.Any:
        hlist = Hlist(toks.as_list())
        self.pop_state()
        return [hlist]

    def non_math(self, toks: ParseResults) -> T.Any:
        s = toks[0].replace(r'\$', '$')
        symbols = [Char(c, self.get_state()) for c in s]
        hlist = Hlist(symbols)
        # We're going into math now, so set font to 'it'
        self.push_state()
        self.get_state().font = mpl.rcParams['mathtext.default']
        return [hlist]

    float_literal = staticmethod(pyparsing_common.convert_to_float)

    def text(self, toks: ParseResults) -> T.Any:
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist = Hlist([Char(c, state) for c in toks[1]])
        self.pop_state()
        return [hlist]

    def _make_space(self, percentage: float) -> Kern:
        # In TeX, an em (the unit usually used to measure horizontal lengths)
        # is not the width of the character 'm'; it is the same in different
        # font styles (e.g. roman or italic). Mathtext, however, uses 'm' in
        # the italic style so that horizontal spaces don't depend on the
        # current font style.
        state = self.get_state()
        key = (state.font, state.fontsize, state.dpi)
        width = self._em_width_cache.get(key)
        if width is None:
            metrics = state.fontset.get_metrics(
                'it', mpl.rcParams['mathtext.default'], 'm',
                state.fontsize, state.dpi)
            width = metrics.advance
            self._em_width_cache[key] = width
        return Kern(width * percentage)

    _space_widths = {
        r'\,':         0.16667,   # 3/18 em = 3 mu
        r'\thinspace': 0.16667,   # 3/18 em = 3 mu
        r'\/':         0.16667,   # 3/18 em = 3 mu
        r'\>':         0.22222,   # 4/18 em = 4 mu
        r'\:':         0.22222,   # 4/18 em = 4 mu
        r'\;':         0.27778,   # 5/18 em = 5 mu
        r'\ ':         0.33333,   # 6/18 em = 6 mu
        r'~':          0.33333,   # 6/18 em = 6 mu, nonbreakable
        r'\enspace':   0.5,       # 9/18 em = 9 mu
        r'\quad':      1,         # 1 em = 18 mu
        r'\qquad':     2,         # 2 em = 36 mu
        r'\!':         -0.16667,  # -3/18 em = -3 mu
    }

    def space(self, toks: ParseResults) -> T.Any:
        num = self._space_widths[toks["space"]]
        box = self._make_space(num)
        return [box]

    def customspace(self, toks: ParseResults) -> T.Any:
        return [self._make_space(toks["space"])]

    def symbol(self, s: str, loc: int,
               toks: ParseResults | dict[str, str]) -> T.Any:
        c = toks["sym"]
        if c == "-":
            # "U+2212 minus sign is the preferred representation of the unary
            # and binary minus sign rather than the ASCII-derived U+002D
            # hyphen-minus, because minus sign is unambiguous and because it
            # is rendered with a more desirable length, usually longer than a
            # hyphen." (https://www.unicode.org/reports/tr25/)
            c = "\N{MINUS SIGN}"
        try:
            char = Char(c, self.get_state())
        except ValueError as err:
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # iterate until we find previous character, needed for cases
            # such as $=-2$, ${ -2}$, $ -2$, or $   -2$.
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # Binary operators at start of string should not be spaced
            # Also, operators in sub- or superscripts should not be spaced
            if (self._in_subscript_or_superscript or (
                    c in self._binary_operators and (
                    len(s[:loc].split()) == 0 or prev_char in {
                        '{', *self._left_delims, *self._relation_symbols}))):
                return [char]
            else:
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            next_char = next((c for c in s[loc + 1:] if c != ' '), '')

            # Do not space commas between brackets
            if c == ',':
                if prev_char == '{' and next_char == '}':
                    return [char]

            # Do not space dots as decimal separators
            if c == '.' and prev_char.isdigit() and next_char.isdigit():
                return [char]
            else:
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        return [char]

    def unknown_symbol(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        raise ParseFatalException(s, loc, f"Unknown symbol: {toks['name']}")

    _accent_map = {
        r'hat':            r'\circumflexaccent',
        r'breve':          r'\combiningbreve',
        r'bar':            r'\combiningoverline',
        r'grave':          r'\combininggraveaccent',
        r'acute':          r'\combiningacuteaccent',
        r'tilde':          r'\combiningtilde',
        r'dot':            r'\combiningdotabove',
        r'ddot':           r'\combiningdiaeresis',
        r'dddot':          r'\combiningthreedotsabove',
        r'ddddot':         r'\combiningfourdotsabove',
        r'vec':            r'\combiningrightarrowabove',
        r'"':              r'\combiningdiaeresis',
        r"`":              r'\combininggraveaccent',
        r"'":              r'\combiningacuteaccent',
        r'~':              r'\combiningtilde',
        r'.':              r'\combiningdotabove',
        r'^':              r'\circumflexaccent',
        r'overrightarrow': r'\rightarrow',
        r'overleftarrow':  r'\leftarrow',
        r'mathring':       r'\circ',
    }

    _wide_accents = set(r"widehat widetilde widebar".split())

    def accent(self, toks: ParseResults) -> T.Any:
        state = self.get_state()
        thickness = state.get_current_underline_thickness()
        accent = toks["accent"]
        sym = toks["sym"]
        accent_box: Node
        if accent in self._wide_accents:
            accent_box = AutoWidthChar(
                '\\' + accent, sym.width, state, char_class=Accent)
        else:
            accent_box = Accent(self._accent_map[accent], state)
        if accent == 'mathring':
            accent_box.shrink()
            accent_box.shrink()
        centered = HCentered([Hbox(sym.width / 4.0), accent_box])
        centered.hpack(sym.width, 'exactly')
        return Vlist([
                centered,
                Vbox(0., thickness * 2.0),
                Hlist([sym])
                ])

    def function(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        hlist = self.operatorname(s, loc, toks)
        hlist.function_name = toks["name"]
        return hlist

    def operatorname(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist_list: list[Node] = []
        # Change the font of Chars, but leave Kerns alone
        name = toks["name"]
        for c in name:
            if isinstance(c, Char):
                c.font = 'rm'
                c._update_metrics()
                hlist_list.append(c)
            elif isinstance(c, str):
                hlist_list.append(Char(c, state))
            else:
                hlist_list.append(c)
        next_char_loc = loc + len(name) + 1
        if isinstance(name, ParseResults):
            next_char_loc += len('operatorname{}')
        next_char = next((c for c in s[next_char_loc:] if c != ' '), '')
        delimiters = self._delims | {'^', '_'}
        if (next_char not in delimiters and
                name not in self._overunder_functions):
            # Add thin space except when followed by parenthesis, bracket, etc.
            hlist_list += [self._make_space(self._space_widths[r'\,'])]
        self.pop_state()
        # if followed by a super/subscript, set flag to true
        # This flag tells subsuper to add space after this operator
        if next_char in {'^', '_'}:
            self._in_subscript_or_superscript = True
        else:
            self._in_subscript_or_superscript = False

        return Hlist(hlist_list)

    def start_group(self, toks: ParseResults) -> T.Any:
        self.push_state()
        # Deal with LaTeX-style font tokens
        if toks.get("font"):
            self.get_state().font = toks.get("font")
        return []

    def group(self, toks: ParseResults) -> T.Any:
        grp = Hlist(toks.get("group", []))
        return [grp]

    def required_group(self, toks: ParseResults) -> T.Any:
        return Hlist(toks.get("group", []))

    optional_group = required_group

    def end_group(self) -> T.Any:
        self.pop_state()
        return []

    def unclosed_group(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        raise ParseFatalException(s, len(s), "Expected '}'")

    def font(self, toks: ParseResults) -> T.Any:
        self.get_state().font = toks["font"]
        return []

    def is_overunder(self, nucleus: Node) -> bool:
        if isinstance(nucleus, Char):
            return nucleus.c in self._overunder_symbols
        elif isinstance(nucleus, Hlist) and hasattr(nucleus, 'function_name'):
            return nucleus.function_name in self._overunder_functions
        return False

    def is_dropsub(self, nucleus: Node) -> bool:
        if isinstance(nucleus, Char):
            return nucleus.c in self._dropsub_symbols
        return False

    def is_slanted(self, nucleus: Node) -> bool:
        if isinstance(nucleus, Char):
            return nucleus.is_slanted()
        return False

    def subsuper(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        nucleus = toks.get("nucleus", Hbox(0))
        subsuper = toks.get("subsuper", [])
        napostrophes = len(toks.get("apostrophes", []))

        if not subsuper and not napostrophes:
            return nucleus

        sub = super = None
        while subsuper:
            op, arg, *subsuper = subsuper
            if op == '_':
                if sub is not None:
                    raise ParseFatalException("Double subscript")
                sub = arg
            else:
                if super is not None:
                    raise ParseFatalException("Double superscript")
                super = arg

        state = self.get_state()
        rule_thickness = state.fontset.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        xHeight = state.fontset.get_xheight(
            state.font, state.fontsize, state.dpi)

        if napostrophes:
            if super is None:
                super = Hlist([])
            for i in range(napostrophes):
                super.children.extend(self.symbol(s, loc, {"sym": "\\prime"}))
            # kern() and hpack() needed to get the metrics right after
            # extending
            super.kern()
            super.hpack()

        # Handle over/under symbols, such as sum or prod
        if self.is_overunder(nucleus):
            vlist = []
            shift = 0.
            width = nucleus.width
            if super is not None:
                super.shrink()
                width = max(width, super.width)
            if sub is not None:
                sub.shrink()
                width = max(width, sub.width)

            vgap = rule_thickness * 3.0
            if super is not None:
                hlist = HCentered([super])
                hlist.hpack(width, 'exactly')
                vlist.extend([hlist, Vbox(0, vgap)])
            hlist = HCentered([nucleus])
            hlist.hpack(width, 'exactly')
            vlist.append(hlist)
            if sub is not None:
                hlist = HCentered([sub])
                hlist.hpack(width, 'exactly')
                vlist.extend([Vbox(0, vgap), hlist])
                shift = hlist.height + vgap + nucleus.depth
            vlt = Vlist(vlist)
            vlt.shift_amount = shift
            result = Hlist([vlt])
            return [result]

        # We remove kerning on the last character for consistency (otherwise
        # it will compute kerning based on non-shrunk characters and may put
        # them too close together when superscripted)
        # We change the width of the last character to match the advance to
        # consider some fonts with weird metrics: e.g. stix's f has a width of
        # 7.75 and a kerning of -4.0 for an advance of 3.72, and we want to put
        # the superscript at the advance
        last_char = nucleus
        if isinstance(nucleus, Hlist):
            new_children = nucleus.children
            if len(new_children):
                # remove last kern
                if (isinstance(new_children[-1], Kern) and
                        isinstance(new_children[-2], Char)):
                    new_children = new_children[:-1]
                last_char = new_children[-1]
                if isinstance(last_char, Char):
                    last_char.width = last_char._metrics.advance
            # create new Hlist without kerning
            nucleus = Hlist(new_children, do_kern=False)
        else:
            if isinstance(nucleus, Char):
                last_char.width = last_char._metrics.advance
            nucleus = Hlist([nucleus])

        # Handle regular sub/superscripts
        constants = _get_font_constant_set(state)
        lc_height   = last_char.height
        lc_baseline = 0.0
        if self.is_dropsub(last_char):
            lc_baseline = last_char.depth

        # Compute kerning for sub and super
        superkern = constants.delta * xHeight
        subkern = constants.delta * xHeight
        if self.is_slanted(last_char):
            superkern += constants.delta * xHeight
            superkern += (constants.delta_slanted *
                          (lc_height - xHeight * 2. / 3.))
            if self.is_dropsub(last_char):
                subkern = (3 * constants.delta -
                           constants.delta_integral) * lc_height
                superkern = (3 * constants.delta +
                             constants.delta_integral) * lc_height
            else:
                subkern = 0

        x: List
        if super is None:
            # node757
            # Note: One of super or sub must be a Node if we're in this function, but
            # mypy can't know this, since it can't interpret pyparsing expressions,
            # hence the cast.
            x = Hlist([Kern(subkern), T.cast(Node, sub)])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_down = lc_baseline + constants.subdrop * xHeight
            else:
                shift_down = constants.sub1 * xHeight
            x.shift_amount = shift_down
        else:
            x = Hlist([Kern(superkern), super])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_up = lc_height - constants.subdrop * xHeight
            else:
                shift_up = constants.sup1 * xHeight
            if sub is None:
                x.shift_amount = -shift_up
            else:  # Both sub and superscript
                y = Hlist([Kern(subkern), sub])
                y.shrink()
                if self.is_dropsub(last_char):
                    shift_down = lc_baseline + constants.subdrop * xHeight
                else:
                    shift_down = constants.sub2 * xHeight
                # If sub and superscript collide, move super up
                clr = (2.0 * rule_thickness -
                       ((shift_up - x.depth) - (y.height - shift_down)))
                if clr > 0.:
                    shift_up += clr
                x = Vlist([
                    x,
                    Kern((shift_up - x.depth) - (y.height - shift_down)),
                    y])
                x.shift_amount = shift_down

        if not self.is_dropsub(last_char):
            x.width += constants.script_space * xHeight

        # Do we need to add a space after the nucleus?
        # To find out, check the flag set by operatorname
        spaced_nucleus: list[Node] = [nucleus, x]
        if self._in_subscript_or_superscript:
            spaced_nucleus += [self._make_space(self._space_widths[r'\,'])]
            self._in_subscript_or_superscript = False

        result = Hlist(spaced_nucleus)
        return [result]

    def _genfrac(self, ldelim: str, rdelim: str, rule: float | None, style: _MathStyle,
                 num: Hlist, den: Hlist) -> T.Any:
        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        for _ in range(style.value):
            num.shrink()
            den.shrink()
        cnum = HCentered([num])
        cden = HCentered([den])
        width = max(num.width, den.width)
        cnum.hpack(width, 'exactly')
        cden.hpack(width, 'exactly')
        vlist = Vlist([cnum,                      # numerator
                       Vbox(0, thickness * 2.0),  # space
                       Hrule(state, rule),        # rule
                       Vbox(0, thickness * 2.0),  # space
                       cden                       # denominator
                       ])

        # Shift so the fraction line sits in the middle of the
        # equals sign
        metrics = state.fontset.get_metrics(
            state.font, mpl.rcParams['mathtext.default'],
            '=', state.fontsize, state.dpi)
        shift = (cden.height -
                 ((metrics.ymax + metrics.ymin) / 2 -
                  thickness * 3.0))
        vlist.shift_amount = shift

        result = [Hlist([vlist, Hbox(thickness * 2.)])]
        if ldelim or rdelim:
            if ldelim == '':
                ldelim = '.'
            if rdelim == '':
                rdelim = '.'
            return self._auto_sized_delimiter(ldelim,
                                              T.cast(list[Box | Char | str],
                                                     result),
                                              rdelim)
        return result

    def style_literal(self, toks: ParseResults) -> T.Any:
        return self._MathStyle(int(toks["style_literal"]))

    def genfrac(self, toks: ParseResults) -> T.Any:
        return self._genfrac(
            toks.get("ldelim", ""), toks.get("rdelim", ""),
            toks["rulesize"], toks.get("style", self._MathStyle.TEXTSTYLE),
            toks["num"], toks["den"])

    def frac(self, toks: ParseResults) -> T.Any:
        return self._genfrac(
            "", "", self.get_state().get_current_underline_thickness(),
            self._MathStyle.TEXTSTYLE, toks["num"], toks["den"])

    def dfrac(self, toks: ParseResults) -> T.Any:
        return self._genfrac(
            "", "", self.get_state().get_current_underline_thickness(),
            self._MathStyle.DISPLAYSTYLE, toks["num"], toks["den"])

    def binom(self, toks: ParseResults) -> T.Any:
        return self._genfrac(
            "(", ")", 0,
            self._MathStyle.TEXTSTYLE, toks["num"], toks["den"])

    def _genset(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        annotation = toks["annotation"]
        body = toks["body"]
        thickness = self.get_state().get_current_underline_thickness()

        annotation.shrink()
        centered_annotation = HCentered([annotation])
        centered_body = HCentered([body])
        width = max(centered_annotation.width, centered_body.width)
        centered_annotation.hpack(width, 'exactly')
        centered_body.hpack(width, 'exactly')

        vgap = thickness * 3
        if s[loc + 1] == "u":  # \underset
            vlist = Vlist([
                centered_body,               # body
                Vbox(0, vgap),               # space
                centered_annotation          # annotation
            ])
            # Shift so the body sits in the same vertical position
            vlist.shift_amount = centered_body.depth + centered_annotation.height + vgap
        else:  # \overset
            vlist = Vlist([
                centered_annotation,         # annotation
                Vbox(0, vgap),               # space
                centered_body                # body
            ])

        # To add horizontal gap between symbols: wrap the Vlist into
        # an Hlist and extend it with an Hbox(0, horizontal_gap)
        return vlist

    overset = underset = _genset

    def sqrt(self, toks: ParseResults) -> T.Any:
        root = toks.get("root")
        body = toks["value"]
        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        # Determine the height of the body, and add a little extra to
        # the height so it doesn't seem cramped
        height = body.height - body.shift_amount + thickness * 5.0
        depth = body.depth + body.shift_amount
        check = AutoHeightChar(r'\__sqrt__', height, depth, state, always=True)
        height = check.height - check.shift_amount
        depth = check.depth + check.shift_amount

        # Put a little extra space to the left and right of the body
        padded_body = Hlist([Hbox(2 * thickness), body, Hbox(2 * thickness)])
        rightside = Vlist([Hrule(state), Glue('fill'), padded_body])
        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # Add the root and shift it upward so it is above the tick.
        # The value of 0.6 is a hard-coded hack ;)
        if not root:
            root = Box(check.width * 0.5, 0., 0.)
        else:
            root = Hlist(root)
            root.shrink()
            root.shrink()

        root_vlist = Vlist([Hlist([root])])
        root_vlist.shift_amount = -height * 0.6

        hlist = Hlist([root_vlist,               # Root
                       # Negative kerning to put root over tick
                       Kern(-check.width * 0.5),
                       check,                    # Check
                       rightside])               # Body
        return [hlist]

    def overline(self, toks: ParseResults) -> T.Any:
        body = toks["body"]

        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        height = body.height - body.shift_amount + thickness * 3.0
        depth = body.depth + body.shift_amount

        # Place overline above body
        rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])

        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        hlist = Hlist([rightside])
        return [hlist]

    def _auto_sized_delimiter(self, front: str,
                              middle: list[Box | Char | str],
                              back: str) -> T.Any:
        state = self.get_state()
        if len(middle):
            height = max([x.height for x in middle if not isinstance(x, str)])
            depth = max([x.depth for x in middle if not isinstance(x, str)])
            factor = None
            for idx, el in enumerate(middle):
                if el == r'\middle':
                    c = T.cast(str, middle[idx + 1])  # Should be one of p.delims.
                    if c != '.':
                        middle[idx + 1] = AutoHeightChar(
                                c, height, depth, state, factor=factor)
                    else:
                        middle.remove(c)
                    del middle[idx]
            # There should only be \middle and its delimiter as str, which have
            # just been removed.
            middle_part = T.cast(list[Box | Char], middle)
        else:
            height = 0
            depth = 0
            factor = 1.0
            middle_part = []

        parts: list[Node] = []
        # \left. and \right. aren't supposed to produce any symbols
        if front != '.':
            parts.append(
                AutoHeightChar(front, height, depth, state, factor=factor))
        parts.extend(middle_part)
        if back != '.':
            parts.append(
                AutoHeightChar(back, height, depth, state, factor=factor))
        hlist = Hlist(parts)
        return hlist

    def auto_delim(self, toks: ParseResults) -> T.Any:
        return self._auto_sized_delimiter(
            toks["left"],
            # if "mid" in toks ... can be removed when requiring pyparsing 3.
            toks["mid"].as_list() if "mid" in toks else [],
            toks["right"])

    def boldsymbol(self, toks: ParseResults) -> T.Any:
        self.push_state()
        state = self.get_state()
        hlist: list[Node] = []
        name = toks["value"]
        for c in name:
            if isinstance(c, Hlist):
                k = c.children[1]
                if isinstance(k, Char):
                    k.font = "bf"
                    k._update_metrics()
                hlist.append(c)
            elif isinstance(c, Char):
                c.font = "bf"
                if (c.c in self._latin_alphabets or
                   c.c[1:] in self._small_greek):
                    c.font = "bfit"
                    c._update_metrics()
                c._update_metrics()
                hlist.append(c)
            else:
                hlist.append(c)
        self.pop_state()

        return Hlist(hlist)

    def substack(self, toks: ParseResults) -> T.Any:
        parts = toks["parts"]
        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        hlist = [Hlist(k) for k in parts[0]]
        max_width = max(map(lambda c: c.width, hlist))

        vlist = []
        for sub in hlist:
            cp = HCentered([sub])
            cp.hpack(max_width, 'exactly')
            vlist.append(cp)

        stack = [val
                 for pair in zip(vlist, [Vbox(0, thickness * 2)] * len(vlist))
                 for val in pair]
        del stack[-1]
        vlt = Vlist(stack)
        result = [Hlist([vlt])]
        return result
<|endoftext|>"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make
  colormaps.

  :ref:`colormaps` an in-depth discussion of choosing
  colormaps.

  :ref:`colormapnorms` for more details about data normalization.
"""

from collections.abc import Mapping

import matplotlib as mpl
from matplotlib import _api, colors
# TODO make this warn on access
from matplotlib.colorizer import _ScalarMappable as ScalarMappable  # noqa
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
from matplotlib._cm_multivar import cmap_families as multivar_cmaps
from matplotlib._cm_bivar import cmaps as bivar_cmaps


_LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    cmap_d = {**cmaps_listed}
    for name, spec in datad.items():
        cmap_d[name] = (  # Precache the cmaps at a fixed lutsize..
            colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
            if 'red' in spec else
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE))

    # Register colormap aliases for gray and grey.
    aliases = {
        # alias -> original name
        'grey': 'gray',
        'gist_grey': 'gist_gray',
        'gist_yerg': 'gist_yarg',
        'Grays': 'Greys',
    }
    for alias, original_name in aliases.items():
        cmap = cmap_d[original_name].copy()
        cmap.name = alias
        cmap_d[alias] = cmap

    # Generate reversed cmaps.
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()
        cmap_d[rmap.name] = rmap
    return cmap_d


class ColormapRegistry(Mapping):
    r"""
    Container for colormaps that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.colormaps`. There should be
    no need for users to instantiate `.ColormapRegistry` themselves.

    Read access uses a dict-like interface mapping names to `.Colormap`\s::

        import matplotlib as mpl
        cmap = mpl.colormaps['viridis']

    Returned `.Colormap`\s are copies, so that their modification does not
    change the global definition of the colormap.

    Additional colormaps can be added via `.ColormapRegistry.register`::

        mpl.colormaps.register(my_colormap)

    To get a list of all registered colormaps, you can do::

        from matplotlib import colormaps
        list(colormaps)
    """
    def __init__(self, cmaps):
        self._cmaps = cmaps
        self._builtin_cmaps = tuple(cmaps)

    def __getitem__(self, item):
        try:
            return self._cmaps[item].copy()
        except KeyError:
            raise KeyError(f"{item!r} is not a known colormap name") from None

    def __iter__(self):
        return iter(self._cmaps)

    def __len__(self):
        return len(self._cmaps)

    def __str__(self):
        return ('ColormapRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def __call__(self):
        """
        Return a list of the registered colormap names.

        This exists only for backward-compatibility in `.pyplot` which had a
        ``plt.colormaps()`` method. The recommended way to get this list is
        now ``list(colormaps)``.
        """
        return list(self)

    def register(self, cmap, *, name=None, force=False):
        """
        Register a new colormap.

        The colormap name can then be used as a string argument to any ``cmap``
        parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

        The colormap registry stores a copy of the given colormap, so that
        future changes to the original colormap instance do not affect the
        registered colormap. Think of this as the registry taking a snapshot
        of the colormap at registration.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            The colormap to register.

        name : str, optional
            The name for the colormap. If not given, ``cmap.name`` is used.

        force : bool, default: False
            If False, a ValueError is raised if trying to overwrite an already
            registered name. True supports overwriting registered colormaps
            other than the builtin colormaps.
        """
        _api.check_isinstance(colors.Colormap, cmap=cmap)

        name = name or cmap.name
        if name in self:
            if not force:
                # don't allow registering an already existing cmap
                # unless explicitly asked to
                raise ValueError(
                    f'A colormap named "{name}" is already registered.')
            elif name in self._builtin_cmaps:
                # We don't allow overriding a builtin.
                raise ValueError("Re-registering the builtin cmap "
                                 f"{name!r} is not allowed.")

            # Warn that we are updating an already existing colormap
            _api.warn_external(f"Overwriting the cmap {name!r} "
                               "that was already in the registry.")

        self._cmaps[name] = cmap.copy()
        # Someone may set the extremes of a builtin colormap and want to register it
        # with a different name for future lookups. The object would still have the
        # builtin name, so we should update it to the registered name
        if self._cmaps[name].name != name:
            self._cmaps[name].name = name

    def unregister(self, name):
        """
        Remove a colormap from the registry.

        You cannot remove built-in colormaps.

        If the named colormap is not registered, returns with no error, raises
        if you try to de-register a default colormap.

        .. warning::

            Colormap names are currently a shared namespace that may be used
            by multiple packages. Use `unregister` only if you know you
            have registered that name before. In particular, do not
            unregister just in case to clean the name before registering a
            new colormap.

        Parameters
        ----------
        name : str
            The name of the colormap to be removed.

        Raises
        ------
        ValueError
            If you try to remove a default built-in colormap.
        """
        if name in self._builtin_cmaps:
            raise ValueError(f"cannot unregister {name!r} which is a builtin "
                             "colormap.")
        self._cmaps.pop(name, None)

    def get_cmap(self, cmap):
        """
        Return a color map specified through *cmap*.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap` or None

            - if a `.Colormap`, return it
            - if a string, look it up in ``mpl.colormaps``
            - if None, return the Colormap defined in :rc:`image.cmap`

        Returns
        -------
        Colormap
        """
        # get the default color map
        if cmap is None:
            return self[mpl.rcParams["image.cmap"]]

        # if the user passed in a Colormap, simply return it
        if isinstance(cmap, colors.Colormap):
            return cmap
        if isinstance(cmap, str):
            _api.check_in_list(sorted(_colormaps), cmap=cmap)
            # otherwise, it must be a string so look it up
            return self[cmap]
        raise TypeError(
            'get_cmap expects None or an instance of a str or Colormap . ' +
            f'you passed {cmap!r} of type {type(cmap)}'
        )


# public access to the colormaps should be via `matplotlib.colormaps`. For now,
# we still create the registry here, but that should stay an implementation
# detail.
_colormaps = ColormapRegistry(_gen_cmap_registry())
globals().update(_colormaps)

_multivar_colormaps = ColormapRegistry(multivar_cmaps)

_bivar_colormaps = ColormapRegistry(bivar_cmaps)


# This is an exact copy of pyplot.get_cmap(). It was removed in 3.9, but apparently
# caused more user trouble than expected. Re-added for 3.9.1 and extended the
# deprecation period for two additional minor releases.
@_api.deprecated(
    '3.7',
    removal='3.11',
    alternative="``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()``"
                " or ``pyplot.get_cmap()``"
    )
def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.

    Returns
    -------
    Colormap
    """
    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    _api.check_in_list(sorted(_colormaps), name=name)
    if lut is None:
        return _colormaps[name]
    else:
        return _colormaps[name].resampled(lut)


def _ensure_cmap(cmap):
    """
    Ensure that we have a `.Colormap` object.

    For internal use to preserve type stability of errors.

    Parameters
    ----------
    cmap : None, str, Colormap

        - if a `Colormap`, return it
        - if a string, look it up in mpl.colormaps
        - if None, look up the default color map in mpl.colormaps

    Returns
    -------
    Colormap

    """
    if isinstance(cmap, colors.Colormap):
        return cmap
    cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
    # use check_in_list to ensure type stability of the exception raised by
    # the internal usage of this (ValueError vs KeyError)
    if cmap_name not in _colormaps:
        _api.check_in_list(sorted(_colormaps), cmap=cmap_name)
    return mpl.colormaps[cmap_name]
<|endoftext|>"""
Streamline plotting for 2D vector fields.

"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines


__all__ = ['streamplot']


def streamplot(axes, x, y, u, v, density=1, linewidth=None, color=None,
               cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
               minlength=0.1, transform=None, zorder=None, start_points=None,
               maxlength=4.0, integration_direction='both',
               broken_streamlines=True):
    """
    Draw streamlines of a vector flow.

    Parameters
    ----------
    x, y : 1D/2D arrays
        Evenly spaced strictly increasing arrays to make a grid.  If 2D, all
        rows of *x* must be equal and all columns of *y* must be equal; i.e.,
        they must be as if generated by ``np.meshgrid(x_1d, y_1d)``.
    u, v : 2D arrays
        *x* and *y*-velocities. The number of rows and columns must match
        the length of *y* and *x*, respectively.
    density : float or (float, float)
        Controls the closeness of streamlines. When ``density = 1``, the domain
        is divided into a 30x30 grid. *density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use a tuple
        (density_x, density_y).
    linewidth : float or 2D array
        The width of the streamlines. With a 2D array the line width can be
        varied across the grid. The array must have the same shape as *u*
        and *v*.
    color : :mpltype:`color` or 2D array
        The streamline color. If given an array, its values are converted to
        colors using *cmap* and *norm*.  The array must have the same shape
        as *u* and *v*.
    cmap, norm
        Data normalization and colormapping parameters for *color*; only used
        if *color* is an array of floats. See `~.Axes.imshow` for a detailed
        description.
    arrowsize : float
        Scaling factor for the arrow size.
    arrowstyle : str
        Arrow style specification.
        See `~matplotlib.patches.FancyArrowPatch`.
    minlength : float
        Minimum length of streamline in axes coordinates.
    start_points : (N, 2) array
        Coordinates of starting points for the streamlines in data coordinates
        (the same coordinates as the *x* and *y* arrays).
    zorder : float
        The zorder of the streamlines and arrows.
        Artists with lower zorder values are drawn first.
    maxlength : float
        Maximum length of streamline in axes coordinates.
    integration_direction : {'forward', 'backward', 'both'}, default: 'both'
        Integrate the streamline in forward, backward or both directions.
    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER
    broken_streamlines : boolean, default: True
        If False, forces streamlines to continue until they
        leave the plot domain.  If True, they may be terminated if they
        come too close to another streamline.

    Returns
    -------
    StreamplotSet
        Container object with attributes

        - ``lines``: `.LineCollection` of streamlines

        - ``arrows``: `.PatchCollection` containing `.FancyArrowPatch`
          objects representing the arrows half-way along streamlines.

        This container will probably change in the future to allow changes
        to the colormap, alpha, etc. for both lines and arrows, but these
        changes should be backward compatible.
    """
    grid = Grid(x, y)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = mpl.rcParams['lines.linewidth']

    line_kw = {}
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    _api.check_in_list(['both', 'forward', 'backward'],
                       integration_direction=integration_direction)

    if integration_direction == 'both':
        maxlength /= 2.

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError("If 'color' is given, it must match the shape of "
                             "the (x, y) grid")
        line_colors = [[]]  # Empty entry allows concatenation of zero arrays.
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError("If 'linewidth' is given, it must match the "
                             "shape of the (x, y) grid")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    # Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must match the shape of the (x, y) grid")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = _get_integrator(u, v, dmap, minlength, maxlength,
                                integration_direction)

    trajectories = []
    if start_points is None:
        for xm, ym in _gen_starting_points(mask.shape):
            if mask[ym, xm] == 0:
                xg, yg = dmap.mask2grid(xm, ym)
                t = integrate(xg, yg, broken_streamlines)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width and
                    grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError(f"Starting point ({xs}, {ys}) outside of "
                                 "data boundaries")

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            # Floating point issues can cause xg, yg to be slightly out of
            # bounds for xs, ys on the upper boundaries. Because we have
            # already checked that the starting points are within the original
            # grid, clip the xg, yg to the grid to work around this issue
            xg = np.clip(xg, 0, grid.nx - 1)
            yg = np.clip(yg, 0, grid.ny - 1)

            t = integrate(xg, yg, broken_streamlines)
            if t is not None:
                trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        cmap = cm._ensure_cmap(cmap)

    streamlines = []
    arrows = []
    for t in trajectories:
        tgx, tgy = t.T
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty = dmap.grid2data(tgx, tgy)
        tx += grid.x_origin
        ty += grid.y_origin

        # Create multiple tiny segments if varying width or color is given
        if isinstance(linewidth, np.ndarray) or use_multicolor_lines:
            points = np.transpose([tx, ty]).reshape(-1, 1, 2)
            streamlines.extend(np.hstack([points[:-1], points[1:]]))
        else:
            points = np.transpose([tx, ty])
            streamlines.append(points)

        # Add arrows halfway along each trajectory.
        s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
        n = np.searchsorted(s, s[-1] / 2.)
        arrow_tail = (tx[n], ty[n])
        arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]))

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))

        p = patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, **arrow_kw)
        arrows.append(p)

    lc = mcollections.LineCollection(
        streamlines, transform=transform, **line_kw)
    lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))
        lc.set_cmap(cmap)
        lc.set_norm(norm)
    axes.add_collection(lc)

    ac = mcollections.PatchCollection(arrows)
    # Adding the collection itself is broken; see #2341.
    for p in arrows:
        axes.add_patch(p)

    axes.autoscale_view()
    stream_container = StreamplotSet(lc, ac)
    return stream_container


class StreamplotSet:

    def __init__(self, lines, arrows):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================

class DomainMap:
    """
    Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / (grid.nx - 1)
        self.y_grid2mask = (mask.ny - 1) / (grid.ny - 1)

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        return round(xi * self.x_grid2mask), round(yi * self.y_grid2mask)

    def mask2grid(self, xm, ym):
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg, broken_streamlines=True):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym, broken_streamlines)

    def reset_start_point(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg, broken_streamlines=True):
        if not self.grid.within_grid(xg, yg):
            raise InvalidIndexError
        xm, ym = self.grid2mask(xg, yg)
        self.mask._update_trajectory(xm, ym, broken_streamlines)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid:
    """Grid of data."""
    def __init__(self, x, y):

        if np.ndim(x) == 1:
            pass
        elif np.ndim(x) == 2:
            x_row = x[0]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if np.ndim(y) == 1:
            pass
        elif np.ndim(y) == 2:
            yt = np.transpose(y)  # Also works for nested lists.
            y_col = yt[0]
            if not np.allclose(y_col, yt):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        if not (np.diff(x) > 0).all():
            raise ValueError("'x' must be strictly increasing")
        if not (np.diff(y) > 0).all():
            raise ValueError("'y' must be strictly increasing")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

        if not np.allclose(np.diff(x), self.width / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.height / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return whether (*xi*, *yi*) is a valid index of the grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since *xi* can be `self.nx - 1 < xi < self.nx`
        return 0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1


class StreamMask:
    """
    Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        try:
            self.nx, self.ny = (30 * np.broadcast_to(density, 2)).astype(int)
        except ValueError as err:
            raise ValueError("'density' must be a scalar or be of length "
                             "2") from err
        if self.nx < 0 or self.ny < 0:
            raise ValueError("'density' must be positive")
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, args):
        return self._mask[args]

    def _start_trajectory(self, xm, ym, broken_streamlines=True):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym, broken_streamlines)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask[t] = 0

    def _update_trajectory(self, xm, ym, broken_streamlines=True):
        """
        Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xy != (xm, ym):
            if self[ym, xm] == 0:
                self._traj.append((ym, xm))
                self._mask[ym, xm] = 1
                self._current_xy = (xm, ym)
            else:
                if broken_streamlines:
                    raise InvalidIndexError
                else:
                    pass


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# =======================

def _get_integrator(u, v, dmap, minlength, maxlength, integration_direction):

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(xi, yi):
        if not dmap.grid.within_grid(xi, yi):
            raise OutOfBounds
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi)
        vi = interpgrid(v, xi, yi)
        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def integrate(x0, y0, broken_streamlines=True):
        """
        Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, xy_traj = 0., []

        try:
            dmap.start_trajectory(x0, y0, broken_streamlines)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'backward']:
            s, xyt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength,
                                     broken_streamlines)
            stotal += s
            xy_traj += xyt[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            s, xyt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength,
                                     broken_streamlines)
            stotal += s
            xy_traj += xyt[1:]

        if stotal > minlength:
            return np.broadcast_arrays(xy_traj, np.empty((1, 2)))[0]
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


class OutOfBounds(IndexError):
    pass


def _integrate_rk12(x0, y0, dmap, f, maxlength, broken_streamlines=True):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as RK4 and RK45 solvers (using
    similar Python implementations) in most setups.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xyf_traj = []

    while True:
        try:
            if dmap.grid.within_grid(xi, yi):
                xyf_traj.append((xi, yi))
            else:
                raise OutOfBounds

            # Compute the two intermediate gradients.
            # f should raise OutOfBounds if the locations given are
            # outside the grid.
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x, yi + ds * k1y)

        except OutOfBounds:
            # Out of the domain during this step.
            # Take an Euler step to the boundary to improve neatness
            # unless the trajectory is currently empty.
            if xyf_traj:
                ds, xyf_traj = _euler_step(xyf_traj, dmap, f)
                stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        ny, nx = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            try:
                dmap.update_trajectory(xi, yi, broken_streamlines)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xyf_traj


def _euler_step(xyf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi, yi = xyf_traj[-1]
    cx, cy = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xyf_traj.append((xi + cx * ds, yi + cy * ds))
    return ds, xyf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi):
    """Fast 2D, linear interpolation on an integer grid"""

    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    ai = a0 * (1 - yt) + a1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai


def _gen_starting_points(shape):
    """
    Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    direction = 'right'
    for i in range(nx * ny):
        yield x, y

        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'
<|endoftext|>from matplotlib import cbook
from matplotlib.artist import Artist


class Container(tuple):
    """
    Base class for containers.

    Containers are classes that collect semantically related Artists such as
    the bars of a bar plot.
    """

    def __repr__(self):
        return f"<{type(self).__name__} object of {len(self)} artists>"

    def __new__(cls, *args, **kwargs):
        return tuple.__new__(cls, args[0])

    def __init__(self, kl, label=None):
        self._callbacks = cbook.CallbackRegistry(signals=["pchanged"])
        self._remove_method = None
        self._label = str(label) if label is not None else None

    def remove(self):
        for c in cbook.flatten(
                self, scalarp=lambda x: isinstance(x, Artist)):
            if c is not None:
                c.remove()
        if self._remove_method:
            self._remove_method(self)

    def get_children(self):
        return [child for child in cbook.flatten(self) if child is not None]

    get_label = Artist.get_label
    set_label = Artist.set_label
    add_callback = Artist.add_callback
    remove_callback = Artist.remove_callback
    pchanged = Artist.pchanged


class BarContainer(Container):
    """
    Container for the artists of bar plots (e.g. created by `.Axes.bar`).

    The container can be treated as a tuple of the *patches* themselves.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    patches : list of :class:`~matplotlib.patches.Rectangle`
        The artists of the bars.

    errorbar : None or :class:`~matplotlib.container.ErrorbarContainer`
        A container for the error bar artists if error bars are present.
        *None* otherwise.

    datavalues : None or array-like
        The underlying data values corresponding to the bars.

    orientation : {'vertical', 'horizontal'}, default: None
        If 'vertical', the bars are assumed to be vertical.
        If 'horizontal', the bars are assumed to be horizontal.

    """

    def __init__(self, patches, errorbar=None, *, datavalues=None,
                 orientation=None, **kwargs):
        self.patches = patches
        self.errorbar = errorbar
        self.datavalues = datavalues
        self.orientation = orientation
        super().__init__(patches, **kwargs)


class ErrorbarContainer(Container):
    """
    Container for the artists of error bars (e.g. created by `.Axes.errorbar`).

    The container can be treated as the *lines* tuple itself.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    lines : tuple
        Tuple of ``(data_line, caplines, barlinecols)``.

        - data_line : A `~matplotlib.lines.Line2D` instance of x, y plot markers
          and/or line.
        - caplines : A tuple of `~matplotlib.lines.Line2D` instances of the error
          bar caps.
        - barlinecols : A tuple of `~matplotlib.collections.LineCollection` with the
          horizontal and vertical error ranges.

    has_xerr, has_yerr : bool
        ``True`` if the errorbar has x/y errors.

    """

    def __init__(self, lines, has_xerr=False, has_yerr=False, **kwargs):
        self.lines = lines
        self.has_xerr = has_xerr
        self.has_yerr = has_yerr
        super().__init__(lines, **kwargs)


class StemContainer(Container):
    """
    Container for the artists created in a :meth:`.Axes.stem` plot.

    The container can be treated like a namedtuple ``(markerline, stemlines,
    baseline)``.

    Attributes
    ----------
    markerline : `~matplotlib.lines.Line2D`
        The artist of the markers at the stem heads.

    stemlines : `~matplotlib.collections.LineCollection`
        The artists of the vertical lines for all stems.

    baseline : `~matplotlib.lines.Line2D`
        The artist of the horizontal baseline.
    """
    def __init__(self, markerline_stemlines_baseline, **kwargs):
        """
        Parameters
        ----------
        markerline_stemlines_baseline : tuple
            Tuple of ``(markerline, stemlines, baseline)``.
            ``markerline`` contains the `.Line2D` of the markers,
            ``stemlines`` is a `.LineCollection` of the main lines,
            ``baseline`` is the `.Line2D` of the baseline.
        """
        markerline, stemlines, baseline = markerline_stemlines_baseline
        self.markerline = markerline
        self.stemlines = stemlines
        self.baseline = baseline
        super().__init__(markerline_stemlines_baseline, **kwargs)
<|endoftext|>r"""
Functions to handle markers; used by the marker functionality of
`~matplotlib.axes.Axes.plot`, `~matplotlib.axes.Axes.scatter`, and
`~matplotlib.axes.Axes.errorbar`.

All possible markers are defined here:

============================== ====== =========================================
marker                         symbol description
============================== ====== =========================================
``"."``                        |m00|  point
``","``                        |m01|  pixel
``"o"``                        |m02|  circle
``"v"``                        |m03|  triangle_down
``"^"``                        |m04|  triangle_up
``"<"``                        |m05|  triangle_left
``">"``                        |m06|  triangle_right
``"1"``                        |m07|  tri_down
``"2"``                        |m08|  tri_up
``"3"``                        |m09|  tri_left
``"4"``                        |m10|  tri_right
``"8"``                        |m11|  octagon
``"s"``                        |m12|  square
``"p"``                        |m13|  pentagon
``"P"``                        |m23|  plus (filled)
``"*"``                        |m14|  star
``"h"``                        |m15|  hexagon1
``"H"``                        |m16|  hexagon2
``"+"``                        |m17|  plus
``"x"``                        |m18|  x
``"X"``                        |m24|  x (filled)
``"D"``                        |m19|  diamond
``"d"``                        |m20|  thin_diamond
``"|"``                        |m21|  vline
``"_"``                        |m22|  hline
``0`` (``TICKLEFT``)           |m25|  tickleft
``1`` (``TICKRIGHT``)          |m26|  tickright
``2`` (``TICKUP``)             |m27|  tickup
``3`` (``TICKDOWN``)           |m28|  tickdown
``4`` (``CARETLEFT``)          |m29|  caretleft
``5`` (``CARETRIGHT``)         |m30|  caretright
``6`` (``CARETUP``)            |m31|  caretup
``7`` (``CARETDOWN``)          |m32|  caretdown
``8`` (``CARETLEFTBASE``)      |m33|  caretleft (centered at base)
``9`` (``CARETRIGHTBASE``)     |m34|  caretright (centered at base)
``10`` (``CARETUPBASE``)       |m35|  caretup (centered at base)
``11`` (``CARETDOWNBASE``)     |m36|  caretdown (centered at base)
``"none"`` or ``"None"``              nothing
``" "`` or  ``""``                    nothing
``"$...$"``                    |m37|  Render the string using mathtext.
                                      E.g ``"$f$"`` for marker showing the
                                      letter ``f``.
``verts``                             A list of (x, y) pairs used for Path
                                      vertices. The center of the marker is
                                      located at (0, 0) and the size is
                                      normalized, such that the created path
                                      is encapsulated inside the unit cell.
``path``                              A `~matplotlib.path.Path` instance.
``(numsides, 0, angle)``              A regular polygon with ``numsides``
                                      sides, rotated by ``angle``.
``(numsides, 1, angle)``              A star-like symbol with ``numsides``
                                      sides, rotated by ``angle``.
``(numsides, 2, angle)``              An asterisk with ``numsides`` sides,
                                      rotated by ``angle``.
============================== ====== =========================================

Note that special symbols can be defined via the
:ref:`STIX math font <mathtext>`,
e.g. ``"$\u266B$"``. For an overview over the STIX font symbols refer to the
`STIX font table <http://www.stixfonts.org/allGlyphs.html>`_.
Also see the :doc:`/gallery/text_labels_and_annotations/stix_fonts_demo`.

Integer numbers from ``0`` to ``11`` create lines and triangles. Those are
equally accessible via capitalized variables, like ``CARETDOWNBASE``.
Hence the following are equivalent::

    plt.plot([1, 2, 3], marker=11)
    plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)

Markers join and cap styles can be customized by creating a new instance of
MarkerStyle.
A MarkerStyle can also have a custom `~matplotlib.transforms.Transform`
allowing it to be arbitrarily rotated or offset.

Examples showing the use of markers:

* :doc:`/gallery/lines_bars_and_markers/marker_reference`
* :doc:`/gallery/lines_bars_and_markers/scatter_star_poly`
* :doc:`/gallery/lines_bars_and_markers/multivariate_marker_plot`

.. |m00| image:: /_static/markers/m00.png
.. |m01| image:: /_static/markers/m01.png
.. |m02| image:: /_static/markers/m02.png
.. |m03| image:: /_static/markers/m03.png
.. |m04| image:: /_static/markers/m04.png
.. |m05| image:: /_static/markers/m05.png
.. |m06| image:: /_static/markers/m06.png
.. |m07| image:: /_static/markers/m07.png
.. |m08| image:: /_static/markers/m08.png
.. |m09| image:: /_static/markers/m09.png
.. |m10| image:: /_static/markers/m10.png
.. |m11| image:: /_static/markers/m11.png
.. |m12| image:: /_static/markers/m12.png
.. |m13| image:: /_static/markers/m13.png
.. |m14| image:: /_static/markers/m14.png
.. |m15| image:: /_static/markers/m15.png
.. |m16| image:: /_static/markers/m16.png
.. |m17| image:: /_static/markers/m17.png
.. |m18| image:: /_static/markers/m18.png
.. |m19| image:: /_static/markers/m19.png
.. |m20| image:: /_static/markers/m20.png
.. |m21| image:: /_static/markers/m21.png
.. |m22| image:: /_static/markers/m22.png
.. |m23| image:: /_static/markers/m23.png
.. |m24| image:: /_static/markers/m24.png
.. |m25| image:: /_static/markers/m25.png
.. |m26| image:: /_static/markers/m26.png
.. |m27| image:: /_static/markers/m27.png
.. |m28| image:: /_static/markers/m28.png
.. |m29| image:: /_static/markers/m29.png
.. |m30| image:: /_static/markers/m30.png
.. |m31| image:: /_static/markers/m31.png
.. |m32| image:: /_static/markers/m32.png
.. |m33| image:: /_static/markers/m33.png
.. |m34| image:: /_static/markers/m34.png
.. |m35| image:: /_static/markers/m35.png
.. |m36| image:: /_static/markers/m36.png
.. |m37| image:: /_static/markers/m37.png
"""
import copy

from collections.abc import Sized

import numpy as np

import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle

# special-purpose marker identifiers:
(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN,
 CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
 CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE) = range(12)

_empty_path = Path(np.empty((0, 2)))


class MarkerStyle:
    """
    A class representing marker types.

    Instances are immutable. If you need to change anything, create a new
    instance.

    Attributes
    ----------
    markers : dict
        All known markers.
    filled_markers : tuple
        All known filled markers. This is a subset of *markers*.
    fillstyles : tuple
        The supported fillstyles.
    """

    markers = {
        '.': 'point',
        ',': 'pixel',
        'o': 'circle',
        'v': 'triangle_down',
        '^': 'triangle_up',
        '<': 'triangle_left',
        '>': 'triangle_right',
        '1': 'tri_down',
        '2': 'tri_up',
        '3': 'tri_left',
        '4': 'tri_right',
        '8': 'octagon',
        's': 'square',
        'p': 'pentagon',
        '*': 'star',
        'h': 'hexagon1',
        'H': 'hexagon2',
        '+': 'plus',
        'x': 'x',
        'D': 'diamond',
        'd': 'thin_diamond',
        '|': 'vline',
        '_': 'hline',
        'P': 'plus_filled',
        'X': 'x_filled',
        TICKLEFT: 'tickleft',
        TICKRIGHT: 'tickright',
        TICKUP: 'tickup',
        TICKDOWN: 'tickdown',
        CARETLEFT: 'caretleft',
        CARETRIGHT: 'caretright',
        CARETUP: 'caretup',
        CARETDOWN: 'caretdown',
        CARETLEFTBASE: 'caretleftbase',
        CARETRIGHTBASE: 'caretrightbase',
        CARETUPBASE: 'caretupbase',
        CARETDOWNBASE: 'caretdownbase',
        "None": 'nothing',
        "none": 'nothing',
        ' ': 'nothing',
        '': 'nothing'
    }

    # Just used for informational purposes.  is_filled()
    # is calculated in the _set_* functions.
    filled_markers = (
        '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
        'P', 'X')

    fillstyles = ('full', 'left', 'right', 'bottom', 'top', 'none')
    _half_fillstyles = ('left', 'right', 'bottom', 'top')

    def __init__(self, marker,
                 fillstyle=None, transform=None, capstyle=None, joinstyle=None):
        """
        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values, see the module docstring
              `matplotlib.markers`.

        fillstyle : str, default: :rc:`markers.fillstyle`
            One of 'full', 'left', 'right', 'bottom', 'top', 'none'.

        transform : `~matplotlib.transforms.Transform`, optional
            Transform that will be combined with the native transform of the
            marker.

        capstyle : `.CapStyle` or %(CapStyle)s, optional
            Cap style that will override the default cap style of the marker.

        joinstyle : `.JoinStyle` or %(JoinStyle)s, optional
            Join style that will override the default join style of the marker.
        """
        self._marker_function = None
        self._user_transform = transform
        self._user_capstyle = CapStyle(capstyle) if capstyle is not None else None
        self._user_joinstyle = JoinStyle(joinstyle) if joinstyle is not None else None
        self._set_fillstyle(fillstyle)
        self._set_marker(marker)

    def _recache(self):
        if self._marker_function is None:
            return
        self._path = _empty_path
        self._transform = IdentityTransform()
        self._alt_path = None
        self._alt_transform = None
        self._snap_threshold = None
        self._joinstyle = JoinStyle.round
        self._capstyle = self._user_capstyle or CapStyle.butt
        # Initial guess: Assume the marker is filled unless the fillstyle is
        # set to 'none'. The marker function will override this for unfilled
        # markers.
        self._filled = self._fillstyle != 'none'
        self._marker_function()

    def __bool__(self):
        return bool(len(self._path.vertices))

    def is_filled(self):
        return self._filled

    def get_fillstyle(self):
        return self._fillstyle

    def _set_fillstyle(self, fillstyle):
        """
        Set the fillstyle.

        Parameters
        ----------
        fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            The part of the marker surface that is colored with
            markerfacecolor.
        """
        if fillstyle is None:
            fillstyle = mpl.rcParams['markers.fillstyle']
        _api.check_in_list(self.fillstyles, fillstyle=fillstyle)
        self._fillstyle = fillstyle

    def get_joinstyle(self):
        return self._joinstyle.name

    def get_capstyle(self):
        return self._capstyle.name

    def get_marker(self):
        return self._marker

    def _set_marker(self, marker):
        """
        Set the marker.

        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values see the module docstring
              `matplotlib.markers`.
        """
        if isinstance(marker, str) and cbook.is_math_text(marker):
            self._marker_function = self._set_mathtext_path
        elif isinstance(marker, (int, str)) and marker in self.markers:
            self._marker_function = getattr(self, '_set_' + self.markers[marker])
        elif (isinstance(marker, np.ndarray) and marker.ndim == 2 and
                marker.shape[1] == 2):
            self._marker_function = self._set_vertices
        elif isinstance(marker, Path):
            self._marker_function = self._set_path_marker
        elif (isinstance(marker, Sized) and len(marker) in (2, 3) and
                marker[1] in (0, 1, 2)):
            self._marker_function = self._set_tuple_marker
        elif isinstance(marker, MarkerStyle):
            self.__dict__ = copy.deepcopy(marker.__dict__)
        else:
            try:
                Path(marker)
                self._marker_function = self._set_vertices
            except ValueError as err:
                raise ValueError(
                    f'Unrecognized marker style {marker!r}') from err

        if not isinstance(marker, MarkerStyle):
            self._marker = marker
            self._recache()

    def get_path(self):
        """
        Return a `.Path` for the primary part of the marker.

        For unfilled markers this is the whole marker, for filled markers,
        this is the area to be drawn with *markerfacecolor*.
        """
        return self._path

    def get_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_path()`.
        """
        if self._user_transform is None:
            return self._transform.frozen()
        else:
            return (self._transform + self._user_transform).frozen()

    def get_alt_path(self):
        """
        Return a `.Path` for the alternate part of the marker.

        For unfilled markers, this is *None*; for filled markers, this is the
        area to be drawn with *markerfacecoloralt*.
        """
        return self._alt_path

    def get_alt_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_alt_path()`.
        """
        if self._user_transform is None:
            return self._alt_transform.frozen()
        else:
            return (self._alt_transform + self._user_transform).frozen()

    def get_snap_threshold(self):
        return self._snap_threshold

    def get_user_transform(self):
        """Return user supplied part of marker transform."""
        if self._user_transform is not None:
            return self._user_transform.frozen()

    def transformed(self, transform):
        """
        Return a new version of this marker with the transform applied.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Affine2D`
            Transform will be combined with current user supplied transform.
        """
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is not None:
            new_marker._user_transform += transform
        else:
            new_marker._user_transform = transform
        return new_marker

    def rotated(self, *, deg=None, rad=None):
        """
        Return a new version of this marker rotated by specified angle.

        Parameters
        ----------
        deg : float, optional
            Rotation angle in degrees.

        rad : float, optional
            Rotation angle in radians.

        .. note:: You must specify exactly one of deg or rad.
        """
        if deg is None and rad is None:
            raise ValueError('One of deg or rad is required')
        if deg is not None and rad is not None:
            raise ValueError('Only one of deg and rad can be supplied')
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is None:
            new_marker._user_transform = Affine2D()

        if deg is not None:
            new_marker._user_transform.rotate_deg(deg)
        if rad is not None:
            new_marker._user_transform.rotate(rad)

        return new_marker

    def scaled(self, sx, sy=None):
        """
        Return new marker scaled by specified scale factors.

        If *sy* is not given, the same scale is applied in both the *x*- and
        *y*-directions.

        Parameters
        ----------
        sx : float
            *X*-direction scaling factor.
        sy : float, optional
            *Y*-direction scaling factor.
        """
        if sy is None:
            sy = sx

        new_marker = MarkerStyle(self)
        _transform = new_marker._user_transform or Affine2D()
        new_marker._user_transform = _transform.scale(sx, sy)
        return new_marker

    def _set_nothing(self):
        self._filled = False

    def _set_custom_marker(self, path):
        rescale = np.max(np.abs(path.vertices))  # max of x's and y's.
        self._transform = Affine2D().scale(0.5 / rescale)
        self._path = path

    def _set_path_marker(self):
        self._set_custom_marker(self._marker)

    def _set_vertices(self):
        self._set_custom_marker(Path(self._marker))

    def _set_tuple_marker(self):
        marker = self._marker
        if len(marker) == 2:
            numsides, rotation = marker[0], 0.0
        elif len(marker) == 3:
            numsides, rotation = marker[0], marker[2]
        symstyle = marker[1]
        if symstyle == 0:
            self._path = Path.unit_regular_polygon(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.miter
        elif symstyle == 1:
            self._path = Path.unit_regular_star(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        elif symstyle == 2:
            self._path = Path.unit_regular_asterisk(numsides)
            self._filled = False
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        else:
            raise ValueError(f"Unexpected tuple marker: {marker}")
        self._transform = Affine2D().scale(0.5).rotate_deg(rotation)

    def _set_mathtext_path(self):
        """
        Draw mathtext markers '$...$' using `.TextPath` object.

        Submitted by tcb
        """
        from matplotlib.text import TextPath

        # again, the properties could be initialised just once outside
        # this function
        text = TextPath(xy=(0, 0), s=self.get_marker(),
                        usetex=mpl.rcParams['text.usetex'])
        if len(text.vertices) == 0:
            return

        bbox = text.get_extents()
        max_dim = max(bbox.width, bbox.height)
        self._transform = (
            Affine2D()
            .translate(-bbox.xmin + 0.5 * -bbox.width, -bbox.ymin + 0.5 * -bbox.height)
            .scale(1.0 / max_dim))
        self._path = text
        self._snap = False

    def _half_fill(self):
        return self.get_fillstyle() in self._half_fillstyles

    def _set_circle(self, size=1.0):
        self._transform = Affine2D().scale(0.5 * size)
        self._snap_threshold = np.inf
        if not self._half_fill():
            self._path = Path.unit_circle()
        else:
            self._path = self._alt_path = Path.unit_circle_righthalf()
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180.)

    def _set_point(self):
        self._set_circle(size=0.5)

    def _set_pixel(self):
        self._path = Path.unit_rectangle()
        # Ideally, you'd want -0.5, -0.5 here, but then the snapping
        # algorithm in the Agg backend will round this to a 2x2
        # rectangle from (-1, -1) to (1, 1).  By offsetting it
        # slightly, we can force it to be (0, 0) to (1, 1), which both
        # makes it only be a single pixel and places it correctly
        # aligned to 1-width stroking (i.e. the ticks).  This hack is
        # the best of a number of bad alternatives, mainly because the
        # backends are not aware of what marker is actually being used
        # beyond just its path data.
        self._transform = Affine2D().translate(-0.49999, -0.49999)
        self._snap_threshold = None

    _triangle_path = Path._create_closed([[0, 1], [-1, -1], [1, -1]])
    # Going down halfway looks to small.  Golden ratio is too far.
    _triangle_path_u = Path._create_closed([[0, 1], [-3/5, -1/5], [3/5, -1/5]])
    _triangle_path_d = Path._create_closed(
        [[-3/5, -1/5], [3/5, -1/5], [1, -1], [-1, -1]])
    _triangle_path_l = Path._create_closed([[0, 1], [0, -1], [-1, -1]])
    _triangle_path_r = Path._create_closed([[0, 1], [0, -1], [1, -1]])

    def _set_triangle(self, rot, skip):
        self._transform = Affine2D().scale(0.5).rotate_deg(rot)
        self._snap_threshold = 5.0

        if not self._half_fill():
            self._path = self._triangle_path
        else:
            mpaths = [self._triangle_path_u,
                      self._triangle_path_l,
                      self._triangle_path_d,
                      self._triangle_path_r]

            fs = self.get_fillstyle()
            if fs == 'top':
                self._path = mpaths[(0 + skip) % 4]
                self._alt_path = mpaths[(2 + skip) % 4]
            elif fs == 'bottom':
                self._path = mpaths[(2 + skip) % 4]
                self._alt_path = mpaths[(0 + skip) % 4]
            elif fs == 'left':
                self._path = mpaths[(1 + skip) % 4]
                self._alt_path = mpaths[(3 + skip) % 4]
            else:
                self._path = mpaths[(3 + skip) % 4]
                self._alt_path = mpaths[(1 + skip) % 4]

            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_triangle_up(self):
        return self._set_triangle(0.0, 0)

    def _set_triangle_down(self):
        return self._set_triangle(180.0, 2)

    def _set_triangle_left(self):
        return self._set_triangle(90.0, 3)

    def _set_triangle_right(self):
        return self._set_triangle(270.0, 1)

    def _set_square(self):
        self._transform = Affine2D().translate(-0.5, -0.5)
        self._snap_threshold = 2.0
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            # Build a bottom filled square out of two rectangles, one filled.
            self._path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
                               [0.0, 0.5], [0.0, 0.0]])
            self._alt_path = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0],
                                   [0.0, 1.0], [0.0, 0.5]])
            fs = self.get_fillstyle()
            rotate = {'bottom': 0, 'right': 90, 'top': 180, 'left': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_diamond(self):
        self._transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
        self._snap_threshold = 5.0
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            self._path = Path([[0, 0], [1, 0], [1, 1], [0, 0]])
            self._alt_path = Path([[0, 0], [0, 1], [1, 1], [0, 0]])
            fs = self.get_fillstyle()
            rotate = {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_thin_diamond(self):
        self._set_diamond()
        self._transform.scale(0.6, 1.0)

    def _set_pentagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        polypath = Path.unit_regular_polygon(5)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            y = (1 + np.sqrt(5)) / 4.
            top = Path(verts[[0, 1, 4, 0]])
            bottom = Path(verts[[1, 2, 3, 4, 1]])
            left = Path([verts[0], verts[1], verts[2], [0, -y], verts[0]])
            right = Path([verts[0], verts[4], verts[3], [0, -y], verts[0]])
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]
            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_star(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        polypath = Path.unit_regular_star(5, innerCircle=0.381966)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            top = Path(np.concatenate([verts[0:4], verts[7:10], verts[0:1]]))
            bottom = Path(np.concatenate([verts[3:8], verts[3:4]]))
            left = Path(np.concatenate([verts[0:6], verts[0:1]]))
            right = Path(np.concatenate([verts[0:1], verts[5:10], verts[0:1]]))
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]
            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.bevel

    def _set_hexagon1(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = None

        polypath = Path.unit_regular_polygon(6)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            # not drawing inside lines
            x = np.abs(np.cos(5 * np.pi / 6.))
            top = Path(np.concatenate([[(-x, 0)], verts[[1, 0, 5]], [(x, 0)]]))
            bottom = Path(np.concatenate([[(-x, 0)], verts[2:5], [(x, 0)]]))
            left = Path(verts[0:4])
            right = Path(verts[[0, 5, 4, 3]])
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]
            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_hexagon2(self):
        self._transform = Affine2D().scale(0.5).rotate_deg(30)
        self._snap_threshold = None

        polypath = Path.unit_regular_polygon(6)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices
            # not drawing inside lines
            x, y = np.sqrt(3) / 4, 3 / 4.
            top = Path(verts[[1, 0, 5, 4, 1]])
            bottom = Path(verts[1:5])
            left = Path(np.concatenate([
                [(x, y)], verts[:3], [(-x, -y), (x, y)]]))
            right = Path(np.concatenate([
                [(x, y)], verts[5:2:-1], [(-x, -y)]]))
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]
            self._alt_transform = self._transform

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_octagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        polypath = Path.unit_regular_polygon(8)

        if not self._half_fill():
            self._transform.rotate_deg(22.5)
            self._path = polypath
        else:
            x = np.sqrt(2.) / 4.
            self._path = self._alt_path = Path(
                [[0, -1], [0, 1], [-x, 1], [-1, x],
                 [-1, -x], [-x, -1], [0, -1]])
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'left': 0, 'bottom': 90, 'right': 180, 'top': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180.0)

        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])

    def _set_vline(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._line_marker_path

    def _set_hline(self):
        self._set_vline()
        self._transform = self._transform.rotate_deg(90)

    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])

    def _set_tickleft(self):
        self._transform = Affine2D().scale(-1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path

    def _set_tickright(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path

    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])

    def _set_tickup(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path

    def _set_tickdown(self):
        self._transform = Affine2D().scale(1.0, -1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path

    _tri_path = Path([[0.0, 0.0], [0.0, -1.0],
                      [0.0, 0.0], [0.8, 0.5],
                      [0.0, 0.0], [-0.8, 0.5]],
                     [Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO])

    def _set_tri_down(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        self._filled = False
        self._path = self._tri_path

    def _set_tri_up(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(180)

    def _set_tri_left(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(270)

    def _set_tri_right(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(90)

    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])

    def _set_caretdown(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._caret_path
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    def _set_caretup(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleft(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretright(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(90)

    _caret_path_base = Path([[-1.0, 0.0], [0.0, -1.5], [1.0, 0]])

    def _set_caretdownbase(self):
        self._set_caretdown()
        self._path = self._caret_path_base

    def _set_caretupbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleftbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretrightbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(90)

    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0],
                       [0.0, -1.0], [0.0, 1.0]],
                      [Path.MOVETO, Path.LINETO,
                       Path.MOVETO, Path.LINETO])

    def _set_plus(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._plus_path

    _x_path = Path([[-1.0, -1.0], [1.0, 1.0],
                    [-1.0, 1.0], [1.0, -1.0]],
                   [Path.MOVETO, Path.LINETO,
                    Path.MOVETO, Path.LINETO])

    def _set_x(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._x_path

    _plus_filled_path = Path._create_closed(np.array([
        (-1, -3), (+1, -3), (+1, -1), (+3, -1), (+3, +1), (+1, +1),
        (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, -1), (-1, -1)]) / 6)
    _plus_filled_path_t = Path._create_closed(np.array([
        (+3, 0), (+3, +1), (+1, +1), (+1, +3),
        (-1, +3), (-1, +1), (-3, +1), (-3, 0)]) / 6)

    def _set_plus_filled(self):
        self._transform = Affine2D()
        self._snap_threshold = 5.0
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        if not self._half_fill():
            self._path = self._plus_filled_path
        else:
            # Rotate top half path to support all partitions
            self._path = self._alt_path = self._plus_filled_path_t
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180)

    _x_filled_path = Path._create_closed(np.array([
        (-1, -2), (0, -1), (+1, -2), (+2, -1), (+1, 0), (+2, +1),
        (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0), (-2, -1)]) / 4)
    _x_filled_path_t = Path._create_closed(np.array([
        (+1, 0), (+2, +1), (+1, +2), (0, +1),
        (-1, +2), (-2, +1), (-1, 0)]) / 4)

    def _set_x_filled(self):
        self._transform = Affine2D()
        self._snap_threshold = 5.0
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        if not self._half_fill():
            self._path = self._x_filled_path
        else:
            # Rotate top half path to support all partitions
            self._path = self._alt_path = self._x_filled_path_t
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180)
<|endoftext|>"""
A module for converting numbers or color arguments to *RGB* or *RGBA*.

*RGB* and *RGBA* are sequences of, respectively, 3 or 4 floats in the
range 0-1.

This module includes functions and classes for color specification conversions,
and for mapping numbers to colors in a 1-D array of colors called a colormap.

Mapping data onto colors using a colormap typically involves two steps: a data
array is first mapped onto the range 0-1 using a subclass of `Normalize`,
then this number is mapped to a color using a subclass of `Colormap`.  Two
subclasses of `Colormap` provided here:  `LinearSegmentedColormap`, which uses
piecewise-linear interpolation to define colormaps, and `ListedColormap`, which
makes a colormap from a list of colors.

.. seealso::

  :ref:`colormap-manipulation` for examples of how to
  make colormaps and

  :ref:`colormaps` for a list of built-in colormaps.

  :ref:`colormapnorms` for more details about data
  normalization

  More colormaps are available at palettable_.

The module also provides functions for checking whether an object can be
interpreted as a color (`is_color_like`), for converting such an object
to an RGBA tuple (`to_rgba`) or to an HTML-like hex string in the
"#rrggbb" format (`to_hex`), and a sequence of colors to an (n, 4)
RGBA array (`to_rgba_array`).  Caching is used for efficiency.

Colors that Matplotlib recognizes are listed at
:ref:`colors_def`.

.. _palettable: https://jiffyclub.github.io/palettable/
.. _xkcd color survey: https://xkcd.com/color/rgb/
"""

import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale, _image
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS


class _ColorMapping(dict):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.cache = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.cache.clear()

    def __delitem__(self, key):
        super().__delitem__(key)
        self.cache.clear()


_colors_full_map = {}
# Set by reverse priority order.
_colors_full_map.update(XKCD_COLORS)
_colors_full_map.update({k.replace('grey', 'gray'): v
                         for k, v in XKCD_COLORS.items()
                         if 'grey' in k})
_colors_full_map.update(CSS4_COLORS)
_colors_full_map.update(TABLEAU_COLORS)
_colors_full_map.update({k.replace('gray', 'grey'): v
                         for k, v in TABLEAU_COLORS.items()
                         if 'gray' in k})
_colors_full_map.update(BASE_COLORS)
_colors_full_map = _ColorMapping(_colors_full_map)

_REPR_PNG_SIZE = (512, 64)
_BIVAR_REPR_PNG_SIZE = 256


def get_named_colors_mapping():
    """Return the global mapping of names to named colors."""
    return _colors_full_map


class ColorSequenceRegistry(Mapping):
    r"""
    Container for sequences of colors that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.color_sequences`. There
    should be no need for users to instantiate `.ColorSequenceRegistry`
    themselves.

    Read access uses a dict-like interface mapping names to lists of colors::

        import matplotlib as mpl
        colors = mpl.color_sequences['tab10']

    For a list of built in color sequences, see :doc:`/gallery/color/color_sequences`.
    The returned lists are copies, so that their modification does not change
    the global definition of the color sequence.

    Additional color sequences can be added via
    `.ColorSequenceRegistry.register`::

        mpl.color_sequences.register('rgb', ['r', 'g', 'b'])
    """

    _BUILTIN_COLOR_SEQUENCES = {
        'tab10': _cm._tab10_data,
        'tab20': _cm._tab20_data,
        'tab20b': _cm._tab20b_data,
        'tab20c': _cm._tab20c_data,
        'Pastel1': _cm._Pastel1_data,
        'Pastel2': _cm._Pastel2_data,
        'Paired': _cm._Paired_data,
        'Accent': _cm._Accent_data,
        'Dark2': _cm._Dark2_data,
        'Set1': _cm._Set1_data,
        'Set2': _cm._Set2_data,
        'Set3': _cm._Set3_data,
        'petroff10': _cm._petroff10_data,
    }

    def __init__(self):
        self._color_sequences = {**self._BUILTIN_COLOR_SEQUENCES}

    def __getitem__(self, item):
        try:
            return list(self._color_sequences[item])
        except KeyError:
            raise KeyError(f"{item!r} is not a known color sequence name")

    def __iter__(self):
        return iter(self._color_sequences)

    def __len__(self):
        return len(self._color_sequences)

    def __str__(self):
        return ('ColorSequenceRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def register(self, name, color_list):
        """
        Register a new color sequence.

        The color sequence registry stores a copy of the given *color_list*, so
        that future changes to the original list do not affect the registered
        color sequence. Think of this as the registry taking a snapshot
        of *color_list* at registration.

        Parameters
        ----------
        name : str
            The name for the color sequence.

        color_list : list of :mpltype:`color`
            An iterable returning valid Matplotlib colors when iterating over.
            Note however that the returned color sequence will always be a
            list regardless of the input type.

        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(f"{name!r} is a reserved name for a builtin "
                             "color sequence")

        color_list = list(color_list)  # force copy and coerce type to list
        for color in color_list:
            try:
                to_rgba(color)
            except ValueError:
                raise ValueError(
                    f"{color!r} is not a valid color specification")

        self._color_sequences[name] = color_list

    def unregister(self, name):
        """
        Remove a sequence from the registry.

        You cannot remove built-in color sequences.

        If the name is not registered, returns with no error.
        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(
                f"Cannot unregister builtin color sequence {name!r}")
        self._color_sequences.pop(name, None)


_color_sequences = ColorSequenceRegistry()


def _sanitize_extrema(ex):
    if ex is None:
        return ex
    try:
        ret = ex.item()
    except AttributeError:
        ret = float(ex)
    return ret

_nth_color_re = re.compile(r"\AC[0-9]+\Z")


def _is_nth_color(c):
    """Return whether *c* can be interpreted as an item in the color cycle."""
    return isinstance(c, str) and _nth_color_re.match(c)


def is_color_like(c):
    """Return whether *c* can be interpreted as an RGB(A) color."""
    # Special-case nth color syntax because it cannot be parsed during setup.
    if _is_nth_color(c):
        return True
    try:
        to_rgba(c)
    except (TypeError, ValueError):
        return False
    else:
        return True


def _has_alpha_channel(c):
    """Return whether *c* is a color with an alpha channel."""
    # 4-element sequences are interpreted as r, g, b, a
    return not isinstance(c, str) and len(c) == 4


def _check_color_like(**kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is color-like.
    """
    for k, v in kwargs.items():
        if not is_color_like(v):
            raise ValueError(
                f"{v!r} is not a valid value for {k}: supported inputs are "
                f"(r, g, b) and (r, g, b, a) 0-1 float tuples; "
                f"'#rrggbb', '#rrggbbaa', '#rgb', '#rgba' strings; "
                f"named color strings; "
                f"string reprs of 0-1 floats for grayscale values; "
                f"'C0', 'C1', ... strings for colors of the color cycle; "
                f"and pairs combining one of the above with an alpha value")


def same_color(c1, c2):
    """
    Return whether the colors *c1* and *c2* are the same.

    *c1*, *c2* can be single colors or lists/arrays of colors.
    """
    c1 = to_rgba_array(c1)
    c2 = to_rgba_array(c2)
    n1 = max(c1.shape[0], 1)  # 'none' results in shape (0, 4), but is 1-elem
    n2 = max(c2.shape[0], 1)  # 'none' results in shape (0, 4), but is 1-elem

    if n1 != n2:
        raise ValueError('Different number of elements passed.')
    # The following shape test is needed to correctly handle comparisons with
    # 'none', which results in a shape (0, 4) array and thus cannot be tested
    # via value comparison.
    return c1.shape == c2.shape and (c1 == c2).all()


def to_rgba(c, alpha=None):
    """
    Convert *c* to an RGBA color.

    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``

    alpha : float, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

    Returns
    -------
    tuple
        Tuple of floats ``(r, g, b, a)``, where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    if isinstance(c, tuple) and len(c) == 2:
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    # Special-case nth color syntax because it should not be cached.
    if _is_nth_color(c):
        prop_cycler = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycler.by_key().get('color', ['k'])
        c = colors[int(c[1:]) % len(colors)]
    try:
        rgba = _colors_full_map.cache[c, alpha]
    except (KeyError, TypeError):  # Not in cache, or unhashable.
        rgba = None
    if rgba is None:  # Suppress exception chaining of cache lookup failure.
        rgba = _to_rgba_no_colorcycle(c, alpha)
        try:
            _colors_full_map.cache[c, alpha] = rgba
        except TypeError:
            pass
    return rgba


def _to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    orig_c = c
    if c is np.ma.masked:
        return (0., 0., 0., 0.)
    if isinstance(c, str):
        if c.lower() == "none":
            return (0., 0., 0., 0.)
        # Named color.
        try:
            # This may turn c into a non-string, so we check again below.
            c = _colors_full_map[c]
        except KeyError:
            if len(orig_c) != 1:
                try:
                    c = _colors_full_map[c.lower()]
                except KeyError:
                    pass
    if isinstance(c, str):
        # hex color in #rrggbb format.
        match = re.match(r"\A#[a-fA-F0-9]{6}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1:3], c[3:5], c[5:7]])
                    + (alpha if alpha is not None else 1.,))
        # hex color in #rgb format, shorthand for #rrggbb.
        match = re.match(r"\A#[a-fA-F0-9]{3}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1]*2, c[2]*2, c[3]*2])
                    + (alpha if alpha is not None else 1.,))
        # hex color with alpha in #rrggbbaa format.
        match = re.match(r"\A#[a-fA-F0-9]{8}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1:3], c[3:5], c[5:7], c[7:9]]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # hex color with alpha in #rgba format, shorthand for #rrggbbaa.
        match = re.match(r"\A#[a-fA-F0-9]{4}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1]*2, c[2]*2, c[3]*2, c[4]*2]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # string gray.
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(
                    f"Invalid string grayscale value {orig_c!r}. "
                    f"Value must be within 0-1 range")
            return c, c, c, alpha if alpha is not None else 1.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # turn 2-D array into 1-D array
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    # tuple color.
    if not np.iterable(c):
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    if len(c) not in [3, 4]:
        raise ValueError("RGBA sequence should have length 3 or 4")
    if not all(isinstance(x, Real) for x in c):
        # Checks that don't work: `map(float, ...)`, `np.array(..., float)` and
        # `np.array(...).astype(float)` would all convert "0.5" to 0.5.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # Return a tuple to prevent the cached value from being modified.
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError("RGBA values should be within 0-1 range")
    return c


def to_rgba_array(c, alpha=None):
    """
    Convert *c* to a (n, 4) array of RGBA colors.

    Parameters
    ----------
    c : Matplotlib color or array of colors
        If *c* is a masked array, an `~numpy.ndarray` is returned with a
        (0, 0, 0, 0) row for each masked value or row in *c*.

    alpha : float or sequence of floats, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

        If *alpha* is a sequence and *c* is a single color, *c* will be
        repeated to match the length of *alpha*.

    Returns
    -------
    array
        (n, 4) array of RGBA colors,  where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], Real):
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    # Special-case inputs that are already arrays, for performance.  (If the
    # array has the wrong kind or shape, raise the error during one-at-a-time
    # conversion.)
    if np.iterable(alpha):
        alpha = np.asarray(alpha).ravel()
    if (isinstance(c, np.ndarray) and c.dtype.kind in "if"
            and c.ndim == 2 and c.shape[1] in [3, 4]):
        mask = c.mask.any(axis=1) if np.ma.is_masked(c) else None
        c = np.ma.getdata(c)
        if np.iterable(alpha):
            if c.shape[0] == 1 and alpha.shape[0] > 1:
                c = np.tile(c, (alpha.shape[0], 1))
            elif c.shape[0] != alpha.shape[0]:
                raise ValueError("The number of colors must match the number"
                                 " of alpha values if there are more than one"
                                 " of each.")
        if c.shape[1] == 3:
            result = np.column_stack([c, np.zeros(len(c))])
            result[:, -1] = alpha if alpha is not None else 1.
        elif c.shape[1] == 4:
            result = c.copy()
            if alpha is not None:
                result[:, -1] = alpha
        if mask is not None:
            result[mask] = 0
        if np.any((result < 0) | (result > 1)):
            raise ValueError("RGBA values should be within 0-1 range")
        return result
    # Handle single values.
    # Note that this occurs *after* handling inputs that are already arrays, as
    # `to_rgba(c, alpha)` (below) is expensive for such inputs, due to the need
    # to format the array in the ValueError message(!).
    if cbook._str_lower_equal(c, "none"):
        return np.zeros((0, 4), float)
    try:
        if np.iterable(alpha):
            return np.array([to_rgba(c, a) for a in alpha], float)
        else:
            return np.array([to_rgba(c, alpha)], float)
    except TypeError:
        pass
    except ValueError as e:
        if e.args == ("'alpha' must be between 0 and 1, inclusive", ):
            # ValueError is from _to_rgba_no_colorcycle().
            raise e
    if isinstance(c, str):
        raise ValueError(f"{c!r} is not a valid color value.")

    if len(c) == 0:
        return np.zeros((0, 4), float)

    # Quick path if the whole sequence can be directly converted to a numpy
    # array in one shot.
    if isinstance(c, Sequence):
        lens = {len(cc) if isinstance(cc, (list, tuple)) else -1 for cc in c}
        if lens == {3}:
            rgba = np.column_stack([c, np.ones(len(c))])
        elif lens == {4}:
            rgba = np.array(c)
        else:
            rgba = np.array([to_rgba(cc) for cc in c])
    else:
        rgba = np.array([to_rgba(cc) for cc in c])

    if alpha is not None:
        rgba[:, 3] = alpha
        if isinstance(c, Sequence):
            # ensure that an explicit alpha does not overwrite full transparency
            # for "none"
            none_mask = [cbook._str_equal(cc, "none") for cc in c]
            rgba[:, 3][none_mask] = 0
    return rgba


def to_rgb(c):
    """Convert *c* to an RGB color, silently dropping the alpha channel."""
    return to_rgba(c)[:3]


def to_hex(c, keep_alpha=False):
    """
    Convert *c* to a hex color.

    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`

    keep_alpha : bool, default: False
      If False, use the ``#rrggbb`` format, otherwise use ``#rrggbbaa``.

    Returns
    -------
    str
      ``#rrggbb`` or ``#rrggbbaa`` hex color string
    """
    c = to_rgba(c)
    if not keep_alpha:
        c = c[:3]
    return "#" + "".join(format(round(val * 255), "02x") for val in c)


### Backwards-compatible color-conversion API


cnames = CSS4_COLORS
hexColorPattern = re.compile(r"\A#[a-fA-F0-9]{6}\Z")
rgb2hex = to_hex
hex2color = to_rgb


class ColorConverter:
    """
    A class only kept for backwards compatibility.

    Its functionality is entirely provided by module-level functions.
    """
    colors = _colors_full_map
    cache = _colors_full_map.cache
    to_rgb = staticmethod(to_rgb)
    to_rgba = staticmethod(to_rgba)
    to_rgba_array = staticmethod(to_rgba_array)


colorConverter = ColorConverter()


### End of backwards-compatible color-conversion API


def _create_lookup_table(N, data, gamma=1.0):
    r"""
    Create an *N* -element 1D lookup table.

    This assumes a mapping :math:`f : [0, 1] \rightarrow [0, 1]`. The returned
    data is an array of N values :math:`y = f(x)` where x is sampled from
    [0, 1].

    By default (*gamma* = 1) x is equidistantly sampled from [0, 1]. The
    *gamma* correction factor :math:`\gamma` distorts this equidistant
    sampling by :math:`x \rightarrow x^\gamma`.

    Parameters
    ----------
    N : int
        The number of elements of the created lookup table; at least 1.

    data : (M, 3) array-like or callable
        Defines the mapping :math:`f`.

        If a (M, 3) array-like, the rows define values (x, y0, y1).  The x
        values must start with x=0, end with x=1, and all x values be in
        increasing order.

        A value between :math:`x_i` and :math:`x_{i+1}` is mapped to the range
        :math:`y^1_{i-1} \ldots y^0_i` by linear interpolation.

        For the simple case of a y-continuous mapping, y0 and y1 are identical.

        The two values of y are to allow for discontinuous mapping functions.
        E.g. a sawtooth with a period of 0.2 and an amplitude of 1 would be::

            [(0, 1, 0), (0.2, 1, 0), (0.4, 1, 0), ..., [(1, 1, 0)]

        In the special case of ``N == 1``, by convention the returned value
        is y0 for x == 1.

        If *data* is a callable, it must accept and return numpy arrays::

           data(x : ndarray) -> ndarray

        and map values between 0 - 1 to 0 - 1.

    gamma : float
        Gamma correction factor for input distribution x of the mapping.

        See also https://en.wikipedia.org/wiki/Gamma_correction.

    Returns
    -------
    array
        The lookup table where ``lut[x * (N-1)]`` gives the closest value
        for values of x between 0 and 1.

    Notes
    -----
    This function is internally used for `.LinearSegmentedColormap`.
    """

    if callable(data):
        xind = np.linspace(0, 1, N) ** gamma
        lut = np.clip(np.array(data(xind), dtype=float), 0, 1)
        return lut

    try:
        adata = np.array(data)
    except Exception as err:
        raise TypeError("data must be convertible to an array") from err
    _api.check_shape((None, 3), data=adata)

    x = adata[:, 0]
    y0 = adata[:, 1]
    y1 = adata[:, 2]

    if x[0] != 0. or x[-1] != 1.0:
        raise ValueError(
            "data mapping points must start with x=0 and end with x=1")
    if (np.diff(x) < 0).any():
        raise ValueError("data mapping points must have x in increasing order")
    # begin generation of lookup table
    if N == 1:
        # convention: use the y = f(x=1) value for a 1-element lookup table
        lut = np.array(y0[-1])
    else:
        x = x * (N - 1)
        xind = (N - 1) * np.linspace(0, 1, N) ** gamma
        ind = np.searchsorted(x, xind)[1:-1]

        distance = (xind[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])
        lut = np.concatenate([
            [y1[0]],
            distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1],
            [y0[-1]],
        ])
    # ensure that the lut is confined to values between 0 and 1 by clipping it
    return np.clip(lut, 0.0, 1.0)


class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """

    def __init__(self, name, N=256):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        """
        self.name = name
        self.N = int(N)  # ensure that N is always int
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.
        self._rgba_under = None
        self._rgba_over = None
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False
        self.n_variates = 1
        #: When this colormap exists on a scalar mappable and colorbar_extend
        #: is not False, colorbar creation will pick up ``colorbar_extend`` as
        #: the default value for the ``extend`` keyword in the
        #: `matplotlib.colorbar.Colorbar` constructor.
        self.colorbar_extend = False

    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        rgba, mask = self._get_rgba_and_mask(X, alpha=alpha, bytes=bytes)
        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba

    def _get_rgba_and_mask(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        colors : np.ndarray
            Array of RGBA values with a shape of ``X.shape + (4, )``.
        mask : np.ndarray
            Boolean array with True where the input is ``np.nan`` or masked.
        """
        if not self._isinit:
            self._init()

        xa = np.array(X, copy=True)
        if not xa.dtype.isnative:
            # Native byteorder is faster.
            xa = xa.byteswap().view(xa.dtype.newbyteorder())
        if xa.dtype.kind == "f":
            xa *= self.N
            # xa == 1 (== N after multiplication) is not out of range.
            xa[xa == self.N] = self.N - 1
        # Pre-compute the masks before casting to int (which can truncate
        # negative values to zero or wrap large floats to negative ints).
        mask_under = xa < 0
        mask_over = xa >= self.N
        # If input was masked, get the bad mask from it; else mask out nans.
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid="ignore"):
            # We need this cast for unsigned ints as well as floats
            xa = xa.astype(int)
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad

        lut = self._lut
        if bytes:
            lut = (lut * 255).astype(np.uint8)

        rgba = lut.take(xa, axis=0, mode='clip')

        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # Will be cast to uint8 upon assignment.
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(
                    f"alpha is array-like but its shape {alpha.shape} does "
                    f"not match that of X {xa.shape}")
            rgba[..., -1] = alpha
            # If the "bad" color is all zeros, then ignore alpha input.
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)

        return rgba, mask_bad

    def __copy__(self):
        cls = self.__class__
        cmapobject = cls.__new__(cls)
        cmapobject.__dict__.update(self.__dict__)
        if self._isinit:
            cmapobject._lut = np.copy(self._lut)
        return cmapobject

    def __eq__(self, other):
        if (not isinstance(other, Colormap) or
                self.colorbar_extend != other.colorbar_extend):
            return False
        # To compare lookup tables the Colormaps have to be initialized
        if not self._isinit:
            self._init()
        if not other._isinit:
            other._init()
        return np.array_equal(self._lut, other._lut)

    def get_bad(self):
        """Get the color for masked values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_bad])

    def set_bad(self, color='k', alpha=None):
        """Set the color for masked values."""
        self._rgba_bad = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_under(self):
        """Get the color for low out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_under])

    def set_under(self, color='k', alpha=None):
        """Set the color for low out-of-range values."""
        self._rgba_under = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_over(self):
        """Get the color for high out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_over])

    def set_over(self, color='k', alpha=None):
        """Set the color for high out-of-range values."""
        self._rgba_over = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def set_extremes(self, *, bad=None, under=None, over=None):
        """
        Set the colors for masked (*bad*) values and, when ``norm.clip =
        False``, low (*under*) and high (*over*) out-of-range values.
        """
        if bad is not None:
            self.set_bad(bad)
        if under is not None:
            self.set_under(under)
        if over is not None:
            self.set_over(over)

    def with_extremes(self, *, bad=None, under=None, over=None):
        """
        Return a copy of the colormap, for which the colors for masked (*bad*)
        values and, when ``norm.clip = False``, low (*under*) and high (*over*)
        out-of-range values, have been set accordingly.
        """
        new_cm = self.copy()
        new_cm.set_extremes(bad=bad, under=under, over=over)
        return new_cm

    def _set_extremes(self):
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N - 1]
        self._lut[self._i_bad] = self._rgba_bad

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        """Return whether the colormap is grayscale."""
        if not self._isinit:
            self._init()
        return (np.all(self._lut[:, 0] == self._lut[:, 1]) and
                np.all(self._lut[:, 0] == self._lut[:, 2]))

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        if hasattr(self, '_resample'):
            _api.warn_external(
                "The ability to resample a color map is now public API "
                f"However the class {type(self)} still only implements "
                "the previous private _resample method.  Please update "
                "your class."
            )
            return self._resample(lutsize)

        raise NotImplementedError()

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        .. note:: This function is not implemented for the base class.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        See Also
        --------
        LinearSegmentedColormap.reversed
        ListedColormap.reversed
        """
        raise NotImplementedError()

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]),
                    (_REPR_PNG_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        title = self.name + ' colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        def color_block(color):
            hex_color = to_hex(color, keep_alpha=True)
            return (f'<div title="{hex_color}" '
                    'style="display: inline-block; '
                    'width: 1em; height: 1em; '
                    'margin: 0; '
                    'vertical-align: middle; '
                    'border: 1px solid #555; '
                    f'background-color: {hex_color};"></div>')

        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.name}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.name} colormap" '
                f'title="{self.name}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {_REPR_PNG_SIZE[0]+2}px; '
                'display: flex; justify-content: space-between;">'
                '<div style="float: left;">'
                f'{color_block(self.get_under())} under'
                '</div>'
                '<div style="margin: 0 auto; display: inline-block;">'
                f'bad {color_block(self.get_bad())}'
                '</div>'
                '<div style="float: right;">'
                f'over {color_block(self.get_over())}'
                '</div>'
                '</div>')

    def copy(self):
        """Return a copy of the colormap."""
        return self.__copy__()


class LinearSegmentedColormap(Colormap):
    """
    Colormap objects based on lookup tables using linear segments.

    The lookup table is generated using linear interpolation for each
    primary color, with the 0-1 domain divided into any number of
    segments.
    """

    def __init__(self, name, segmentdata, N=256, gamma=1.0):
        """
        Create colormap from linear mapping segments

        segmentdata argument is a dictionary with a red, green and blue
        entries. Each entry should be a list of *x*, *y0*, *y1* tuples,
        forming rows in a table. Entries for alpha are optional.

        Example: suppose you want red to increase from 0 to 1 over
        the bottom half, green to do the same over the middle half,
        and blue over the top half.  Then you would use::

            cdict = {'red':   [(0.0,  0.0, 0.0),
                               (0.5,  1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'green': [(0.0,  0.0, 0.0),
                               (0.25, 0.0, 0.0),
                               (0.75, 1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'blue':  [(0.0,  0.0, 0.0),
                               (0.5,  0.0, 0.0),
                               (1.0,  1.0, 1.0)]}

        Each row in the table for a given color is a sequence of
        *x*, *y0*, *y1* tuples.  In each sequence, *x* must increase
        monotonically from 0 to 1.  For any input value *z* falling
        between *x[i]* and *x[i+1]*, the output value of a given color
        will be linearly interpolated between *y1[i]* and *y0[i+1]*::

            row i:   x  y0  y1
                           /
                          /
            row i+1: x  y0  y1

        Hence y0 in the first row and y1 in the last row are never used.

        See Also
        --------
        LinearSegmentedColormap.from_list
            Static method; factory function for generating a smoothly-varying
            LinearSegmentedColormap.
        """
        # True only if all colors in map are identical; needed for contouring.
        self.monochrome = False
        super().__init__(name, N)
        self._segmentdata = segmentdata
        self._gamma = gamma

    def _init(self):
        self._lut = np.ones((self.N + 3, 4), float)
        self._lut[:-3, 0] = _create_lookup_table(
            self.N, self._segmentdata['red'], self._gamma)
        self._lut[:-3, 1] = _create_lookup_table(
            self.N, self._segmentdata['green'], self._gamma)
        self._lut[:-3, 2] = _create_lookup_table(
            self.N, self._segmentdata['blue'], self._gamma)
        if 'alpha' in self._segmentdata:
            self._lut[:-3, 3] = _create_lookup_table(
                self.N, self._segmentdata['alpha'], 1)
        self._isinit = True
        self._set_extremes()

    def set_gamma(self, gamma):
        """Set a new gamma value and regenerate colormap."""
        self._gamma = gamma
        self._init()

    @staticmethod
    def from_list(name, colors, N=256, gamma=1.0):
        """
        Create a `LinearSegmentedColormap` from a list of colors.

        Parameters
        ----------
        name : str
            The name of the colormap.
        colors : list of :mpltype:`color` or list of (value, color)
            If only colors are given, they are equidistantly mapped from the
            range :math:`[0, 1]`; i.e. 0 maps to ``colors[0]`` and 1 maps to
            ``colors[-1]``.
            If (value, color) pairs are given, the mapping is from *value*
            to *color*. This can be used to divide the range unevenly.
        N : int
            The number of RGB quantization levels.
        gamma : float
        """
        if not np.iterable(colors):
            raise ValueError('colors must be iterable')

        if (isinstance(colors[0], Sized) and len(colors[0]) == 2
                and not isinstance(colors[0], str)):
            # List of value, color pairs
            vals, colors = zip(*colors)
        else:
            vals = np.linspace(0, 1, len(colors))

        r, g, b, a = to_rgba_array(colors).T
        cdict = {
            "red": np.column_stack([vals, r, r]),
            "green": np.column_stack([vals, g, g]),
            "blue": np.column_stack([vals, b, b]),
            "alpha": np.column_stack([vals, a, a]),
        }

        return LinearSegmentedColormap(name, cdict, N, gamma)

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        new_cmap = LinearSegmentedColormap(self.name, self._segmentdata,
                                           lutsize)
        new_cmap._rgba_over = self._rgba_over
        new_cmap._rgba_under = self._rgba_under
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap

    # Helper ensuring picklability of the reversed cmap.
    @staticmethod
    def _reverser(func, x):
        return func(1 - x)

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        LinearSegmentedColormap
            The reversed colormap.
        """
        if name is None:
            name = self.name + "_r"

        # Using a partial object keeps the cmap picklable.
        data_r = {key: (functools.partial(self._reverser, data)
                        if callable(data) else
                        [(1.0 - x, y1, y0) for x, y0, y1 in reversed(data)])
                  for key, data in self._segmentdata.items()}

        new_cmap = LinearSegmentedColormap(name, data_r, self.N, self._gamma)
        # Reverse the over/under values too
        new_cmap._rgba_over = self._rgba_under
        new_cmap._rgba_under = self._rgba_over
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap


class ListedColormap(Colormap):
    """
    Colormap object generated from a list of colors.

    This may be most useful when indexing directly into a colormap,
    but it can also be used to generate special colormaps for ordinary
    mapping.

    Parameters
    ----------
    colors : list, array
        Sequence of Matplotlib color specifications (color names or RGB(A)
        values).
    name : str, optional
        String to identify the colormap.
    N : int, optional
        Number of entries in the map. The default is *None*, in which case
        there is one colormap entry for each element in the list of colors.
        If ::

            N < len(colors)

        the list will be truncated at *N*. If ::

            N > len(colors)

        the list will be extended by repetition.
    """
    def __init__(self, colors, name='from_list', N=None):
        self.monochrome = False  # Are all colors identical? (for contour.py)
        if N is None:
            self.colors = colors
            N = len(colors)
        else:
            if isinstance(colors, str):
                self.colors = [colors] * N
                self.monochrome = True
            elif np.iterable(colors):
                if len(colors) == 1:
                    self.monochrome = True
                self.colors = list(
                    itertools.islice(itertools.cycle(colors), N))
            else:
                try:
                    gray = float(colors)
                except TypeError:
                    pass
                else:
                    self.colors = [gray] * N
                self.monochrome = True
        super().__init__(name, N)

    def _init(self):
        self._lut = np.zeros((self.N + 3, 4), float)
        self._lut[:-3] = to_rgba_array(self.colors)
        self._isinit = True
        self._set_extremes()

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        colors = self(np.linspace(0, 1, lutsize))
        new_cmap = ListedColormap(colors, name=self.name)
        # Keep the over/under values too
        new_cmap._rgba_over = self._rgba_over
        new_cmap._rgba_under = self._rgba_under
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        ListedColormap
            A reversed instance of the colormap.
        """
        if name is None:
            name = self.name + "_r"

        colors_r = list(reversed(self.colors))
        new_cmap = ListedColormap(colors_r, name=name, N=self.N)
        # Reverse the over/under values too
        new_cmap._rgba_over = self._rgba_under
        new_cmap._rgba_under = self._rgba_over
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap


class MultivarColormap:
    """
    Class for holding multiple `~matplotlib.colors.Colormap` for use in a
    `~matplotlib.cm.ScalarMappable` object
    """
    def __init__(self, colormaps, combination_mode, name='multivariate colormap'):
        """
        Parameters
        ----------
        colormaps: list or tuple of `~matplotlib.colors.Colormap` objects
            The individual colormaps that are combined
        combination_mode: str, 'sRGB_add' or 'sRGB_sub'
            Describe how colormaps are combined in sRGB space

            - If 'sRGB_add' -> Mixing produces brighter colors
              `sRGB = sum(colors)`
            - If 'sRGB_sub' -> Mixing produces darker colors
              `sRGB = 1 - sum(1 - colors)`
        name : str, optional
            The name of the colormap family.
        """
        self.name = name

        if not np.iterable(colormaps) \
           or len(colormaps) == 1 \
           or isinstance(colormaps, str):
            raise ValueError("A MultivarColormap must have more than one colormap.")
        colormaps = list(colormaps)  # ensure cmaps is a list, i.e. not a tuple
        for i, cmap in enumerate(colormaps):
            if isinstance(cmap, str):
                colormaps[i] = mpl.colormaps[cmap]
            elif not isinstance(cmap, Colormap):
                raise ValueError("colormaps must be a list of objects that subclass"
                                 " Colormap or a name found in the colormap registry.")

        self._colormaps = colormaps
        _api.check_in_list(['sRGB_add', 'sRGB_sub'], combination_mode=combination_mode)
        self._combination_mode = combination_mode
        self.n_variates = len(colormaps)
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.

    def __call__(self, X, alpha=None, bytes=False, clip=True):
        r"""
        Parameters
        ----------
        X : tuple (X0, X1, ...) of length equal to the number of colormaps
            X0, X1 ...:
            float or int, `~numpy.ndarray` or scalar
            The data value(s) to convert to RGBA.
            For floats, *Xi...* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *Xi...*  should be in the interval ``[0, self[i].N)`` to
            return RGBA values *indexed* from colormap [i] with index ``Xi``, where
            self[i] is colormap i.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching *Xi*, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.
        clip : bool, default: True
            If True, clip output to 0 to 1

        Returns
        -------
        Tuple of RGBA values if X[0] is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """

        if len(X) != len(self):
            raise ValueError(
                f'For the selected colormap the data must have a first dimension '
                f'{len(self)}, not {len(X)}')
        rgba, mask_bad = self[0]._get_rgba_and_mask(X[0], bytes=False)
        for c, xx in zip(self[1:], X[1:]):
            sub_rgba, sub_mask_bad = c._get_rgba_and_mask(xx, bytes=False)
            rgba[..., :3] += sub_rgba[..., :3]  # add colors
            rgba[..., 3] *= sub_rgba[..., 3]  # multiply alpha
            mask_bad |= sub_mask_bad

        if self.combination_mode == 'sRGB_sub':
            rgba[..., :3] -= len(self) - 1

        rgba[mask_bad] = self.get_bad()

        if clip:
            rgba = np.clip(rgba, 0, 1)

        if alpha is not None:
            if clip:
                alpha = np.clip(alpha, 0, 1)
            if np.shape(alpha) not in [(), np.shape(X[0])]:
                raise ValueError(
                    f"alpha is array-like but its shape {np.shape(alpha)} does "
                    f"not match that of X[0] {np.shape(X[0])}")
            rgba[..., -1] *= alpha

        if bytes:
            if not clip:
                raise ValueError(
                    "clip cannot be false while bytes is true"
                    " as uint8 does not support values below 0"
                    " or above 255.")
            rgba = (rgba * 255).astype('uint8')

        if not np.iterable(X[0]):
            rgba = tuple(rgba)

        return rgba

    def copy(self):
        """Return a copy of the multivarcolormap."""
        return self.__copy__()

    def __copy__(self):
        cls = self.__class__
        cmapobject = cls.__new__(cls)
        cmapobject.__dict__.update(self.__dict__)
        cmapobject._colormaps = [cm.copy() for cm in self._colormaps]
        cmapobject._rgba_bad = np.copy(self._rgba_bad)
        return cmapobject

    def __eq__(self, other):
        if not isinstance(other, MultivarColormap):
            return False
        if len(self) != len(other):
            return False
        for c0, c1 in zip(self, other):
            if c0 != c1:
                return False
        if not all(self._rgba_bad == other._rgba_bad):
            return False
        if self.combination_mode != other.combination_mode:
            return False
        return True

    def __getitem__(self, item):
        return self._colormaps[item]

    def __iter__(self):
        for c in self._colormaps:
            yield c

    def __len__(self):
        return len(self._colormaps)

    def __str__(self):
        return self.name

    def get_bad(self):
        """Get the color for masked values."""
        return np.array(self._rgba_bad)

    def resampled(self, lutshape):
        """
        Return a new colormap with *lutshape* entries.

        Parameters
        ----------
        lutshape : tuple of (`int`, `None`)
            The tuple must have a length matching the number of variates.
            For each element in the tuple, if `int`, the corresponding colorbar
            is resampled, if `None`, the corresponding colorbar is not resampled.

        Returns
        -------
        MultivarColormap
        """

        if not np.iterable(lutshape) or len(lutshape) != len(self):
            raise ValueError(f"lutshape must be of length {len(self)}")
        new_cmap = self.copy()
        for i, s in enumerate(lutshape):
            if s is not None:
                new_cmap._colormaps[i] = self[i].resampled(s)
        return new_cmap

    def with_extremes(self, *, bad=None, under=None, over=None):
        """
        Return a copy of the `MultivarColormap` with modified out-of-range attributes.

        The *bad* keyword modifies the copied `MultivarColormap` while *under* and
        *over* modifies the attributes of the copied component colormaps.
        Note that *under* and *over* colors are subject to the mixing rules determined
        by the *combination_mode*.

        Parameters
        ----------
        bad: :mpltype:`color`, default: None
            If Matplotlib color, the bad value is set accordingly in the copy

        under tuple of :mpltype:`color`, default: None
            If tuple, the `under` value of each component is set with the values
            from the tuple.

        over tuple of :mpltype:`color`, default: None
            If tuple, the `over` value of each component is set with the values
            from the tuple.

        Returns
        -------
        MultivarColormap
            copy of self with attributes set

        """
        new_cm = self.copy()
        if bad is not None:
            new_cm._rgba_bad = to_rgba(bad)
        if under is not None:
            if not np.iterable(under) or len(under) != len(new_cm):
                raise ValueError("*under* must contain a color for each scalar colormap"
                                 f" i.e. be of length {len(new_cm)}.")
            else:
                for c, b in zip(new_cm, under):
                    c.set_under(b)
        if over is not None:
            if not np.iterable(over) or len(over) != len(new_cm):
                raise ValueError("*over* must contain a color for each scalar colormap"
                                 f" i.e. be of length {len(new_cm)}.")
            else:
                for c, b in zip(new_cm, over):
                    c.set_over(b)
        return new_cm

    @property
    def combination_mode(self):
        return self._combination_mode

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]),
                                (_REPR_PNG_SIZE[1], 1))
        pixels = np.zeros((_REPR_PNG_SIZE[1]*len(self), _REPR_PNG_SIZE[0], 4),
                          dtype=np.uint8)
        for i, c in enumerate(self):
            pixels[i*_REPR_PNG_SIZE[1]:(i+1)*_REPR_PNG_SIZE[1], :] = c(X, bytes=True)
        png_bytes = io.BytesIO()
        title = self.name + ' multivariate colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the MultivarColormap."""
        return ''.join([c._repr_html_() for c in self._colormaps])


class BivarColormap:
    """
    Base class for all bivariate to RGBA mappings.

    Designed as a drop-in replacement for Colormap when using a 2D
    lookup table. To be used with `~matplotlib.cm.ScalarMappable`.
    """

    def __init__(self, N=256, M=256, shape='square', origin=(0, 0),
                 name='bivariate colormap'):
        """
        Parameters
        ----------
        N : int, default: 256
            The number of RGB quantization levels along the first axis.
        M : int, default: 256
            The number of RGB quantization levels along the second axis.
        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - 'square' each variate is clipped to [0,1] independently
            - 'circle' the variates are clipped radially to the center
              of the colormap, and a circular mask is applied when the colormap
              is displayed
            - 'ignore' the variates are not clipped, but instead assigned the
              'outside' color
            - 'circleignore' a circular mask is applied, but the data is not
              clipped and instead assigned the 'outside' color

        origin : (float, float), default: (0,0)
            The relative origin of the colormap. Typically (0, 0), for colormaps
            that are linear on both axis, and (.5, .5) for circular colormaps.
            Used when getting 1D colormaps from 2D colormaps.
        name : str, optional
            The name of the colormap.
        """

        self.name = name
        self.N = int(N)  # ensure that N is always int
        self.M = int(M)
        _api.check_in_list(['square', 'circle', 'ignore', 'circleignore'], shape=shape)
        self._shape = shape
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.
        self._rgba_outside = (1.0, 0.0, 1.0, 1.0)
        self._isinit = False
        self.n_variates = 2
        self._origin = (float(origin[0]), float(origin[1]))
        '''#: When this colormap exists on a scalar mappable and colorbar_extend
        #: is not False, colorbar creation will pick up ``colorbar_extend`` as
        #: the default value for the ``extend`` keyword in the
        #: `matplotlib.colorbar.Colorbar` constructor.
        self.colorbar_extend = False'''

    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : tuple (X0, X1), X0 and X1: float or int or array-like
            The data value(s) to convert to RGBA.

            - For floats, *X* should be in the interval ``[0.0, 1.0]`` to
              return the RGBA values ``X*100`` percent along the Colormap.
            - For integers, *X* should be in the interval ``[0, Colormap.N)`` to
              return RGBA values *indexed* from the Colormap with index ``X``.

        alpha : float or array-like or None, default: None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X0, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """

        if len(X) != 2:
            raise ValueError(
                f'For a `BivarColormap` the data must have a first dimension '
                f'2, not {len(X)}')

        if not self._isinit:
            self._init()

        X0 = np.ma.array(X[0], copy=True)
        X1 = np.ma.array(X[1], copy=True)
        # clip to shape of colormap, circle square, etc.
        self._clip((X0, X1))

        # Native byteorder is faster.
        if not X0.dtype.isnative:
            X0 = X0.byteswap().view(X0.dtype.newbyteorder())
        if not X1.dtype.isnative:
            X1 = X1.byteswap().view(X1.dtype.newbyteorder())

        if X0.dtype.kind == "f":
            X0 *= self.N
            # xa == 1 (== N after multiplication) is not out of range.
            X0[X0 == self.N] = self.N - 1

        if X1.dtype.kind == "f":
            X1 *= self.M
            # xa == 1 (== N after multiplication) is not out of range.
            X1[X1 == self.M] = self.M - 1

        # Pre-compute the masks before casting to int (which can truncate)
        mask_outside = (X0 < 0) | (X1 < 0) | (X0 >= self.N) | (X1 >= self.M)
        # If input was masked, get the bad mask from it; else mask out nans.
        mask_bad_0 = X0.mask if np.ma.is_masked(X0) else np.isnan(X0)
        mask_bad_1 = X1.mask if np.ma.is_masked(X1) else np.isnan(X1)
        mask_bad = mask_bad_0 | mask_bad_1

        with np.errstate(invalid="ignore"):
            # We need this cast for unsigned ints as well as floats
            X0 = X0.astype(int)
            X1 = X1.astype(int)

        # Set masked values to zero
        # The corresponding rgb values will be replaced later
        for X_part in [X0, X1]:
            X_part[mask_outside] = 0
            X_part[mask_bad] = 0

        rgba = self._lut[X0, X1]
        if np.isscalar(X[0]):
            rgba = np.copy(rgba)
        rgba[mask_outside] = self._rgba_outside
        rgba[mask_bad] = self._rgba_bad
        if bytes:
            rgba = (rgba * 255).astype(np.uint8)
        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # Will be cast to uint8 upon assignment.
            if np.shape(alpha) not in [(), np.shape(X0)]:
                raise ValueError(
                    f"alpha is array-like but its shape {np.shape(alpha)} does "
                    f"not match that of X[0] {np.shape(X0)}")
            rgba[..., -1] = alpha
            # If the "bad" color is all zeros, then ignore alpha input.
            if (np.array(self._rgba_bad) == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)

        if not np.iterable(X[0]):
            rgba = tuple(rgba)
        return rgba

    @property
    def lut(self):
        """
        For external access to the lut, i.e. for displaying the cmap.
        For circular colormaps this returns a lut with a circular mask.

        Internal functions (such as to_rgb()) should use _lut
        which stores the lut without a circular mask
        A lut without the circular mask is needed in to_rgb() because the
        conversion from floats to ints results in some some pixel-requests
        just outside of the circular mask

        """
        if not self._isinit:
            self._init()
        lut = np.copy(self._lut)
        if self.shape == 'circle' or self.shape == 'circleignore':
            n = np.linspace(-1, 1, self.N)
            m = np.linspace(-1, 1, self.M)
            radii_sqr = (n**2)[:, np.newaxis] + (m**2)[np.newaxis, :]
            mask_outside = radii_sqr > 1
            lut[mask_outside, 3] = 0
        return lut

    def __copy__(self):
        cls = self.__class__
        cmapobject = cls.__new__(cls)
        cmapobject.__dict__.update(self.__dict__)

        cmapobject._rgba_outside = np.copy(self._rgba_outside)
        cmapobject._rgba_bad = np.copy(self._rgba_bad)
        cmapobject._shape = self.shape
        if self._isinit:
            cmapobject._lut = np.copy(self._lut)
        return cmapobject

    def __eq__(self, other):
        if not isinstance(other, BivarColormap):
            return False
        # To compare lookup tables the Colormaps have to be initialized
        if not self._isinit:
            self._init()
        if not other._isinit:
            other._init()
        if not np.array_equal(self._lut, other._lut):
            return False
        if not np.array_equal(self._rgba_bad, other._rgba_bad):
            return False
        if not np.array_equal(self._rgba_outside, other._rgba_outside):
            return False
        if self.shape != other.shape:
            return False
        return True

    def get_bad(self):
        """Get the color for masked values."""
        return self._rgba_bad

    def get_outside(self):
        """Get the color for out-of-range values."""
        return self._rgba_outside

    def resampled(self, lutshape, transposed=False):
        """
        Return a new colormap with *lutshape* entries.

        Note that this function does not move the origin.

        Parameters
        ----------
        lutshape : tuple of ints or None
            The tuple must be of length 2, and each entry is either an int or None.

            - If an int, the corresponding axis is resampled.
            - If negative the corresponding axis is resampled in reverse
            - If -1, the axis is inverted
            - If 1 or None, the corresponding axis is not resampled.

        transposed : bool, default: False
            if True, the axes are swapped after resampling

        Returns
        -------
        BivarColormap
        """

        if not np.iterable(lutshape) or len(lutshape) != 2:
            raise ValueError("lutshape must be of length 2")
        lutshape = [lutshape[0], lutshape[1]]
        if lutshape[0] is None or lutshape[0] == 1:
            lutshape[0] = self.N
        if lutshape[1] is None or lutshape[1] == 1:
            lutshape[1] = self.M

        inverted = [False, False]
        if lutshape[0] < 0:
            inverted[0] = True
            lutshape[0] = -lutshape[0]
            if lutshape[0] == 1:
                lutshape[0] = self.N
        if lutshape[1] < 0:
            inverted[1] = True
            lutshape[1] = -lutshape[1]
            if lutshape[1] == 1:
                lutshape[1] = self.M
        x_0, x_1 = np.mgrid[0:1:(lutshape[0] * 1j), 0:1:(lutshape[1] * 1j)]
        if inverted[0]:
            x_0 = x_0[::-1, :]
        if inverted[1]:
            x_1 = x_1[:, ::-1]

        # we need to use shape = 'square' while resampling the colormap.
        # if the colormap has shape = 'circle' we would otherwise get *outside* in the
        # resampled colormap
        shape_memory = self._shape
        self._shape = 'square'
        if transposed:
            new_lut = self((x_1, x_0))
            new_cmap = BivarColormapFromImage(new_lut, name=self.name,
                                              shape=shape_memory,
                                              origin=self.origin[::-1])
        else:
            new_lut = self((x_0, x_1))
            new_cmap = BivarColormapFromImage(new_lut, name=self.name,
                                              shape=shape_memory,
                                              origin=self.origin)
        self._shape = shape_memory

        new_cmap._rgba_bad = self._rgba_bad
        new_cmap._rgba_outside = self._rgba_outside
        return new_cmap

    def reversed(self, axis_0=True, axis_1=True):
        """
        Reverses both or one of the axis.
        """
        r_0 = -1 if axis_0 else 1
        r_1 = -1 if axis_1 else 1
        return self.resampled((r_0, r_1))

    def transposed(self):
        """
        Transposes the colormap by swapping the order of the axis
        """
        return self.resampled((None, None), transposed=True)

    def with_extremes(self, *, bad=None, outside=None, shape=None, origin=None):
        """
        Return a copy of the `BivarColormap` with modified attributes.

        Note that the *outside* color is only relevant if `shape` = 'ignore'
        or 'circleignore'.

        Parameters
        ----------
        bad : None or :mpltype:`color`
            If Matplotlib color, the *bad* value is set accordingly in the copy

        outside : None or :mpltype:`color`
            If Matplotlib color and shape is 'ignore' or 'circleignore', values
            *outside* the colormap are colored accordingly in the copy

        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - If 'square' each variate is clipped to [0,1] independently
            - If 'circle' the variates are clipped radially to the center
              of the colormap, and a circular mask is applied when the colormap
              is displayed
            - If 'ignore' the variates are not clipped, but instead assigned the
              *outside* color
            - If 'circleignore' a circular mask is applied, but the data is not
              clipped and instead assigned the *outside* color

        origin : (float, float)
            The relative origin of the colormap. Typically (0, 0), for colormaps
            that are linear on both axis, and (.5, .5) for circular colormaps.
            Used when getting 1D colormaps from 2D colormaps.

        Returns
        -------
        BivarColormap
            copy of self with attributes set
        """
        new_cm = self.copy()
        if bad is not None:
            new_cm._rgba_bad = to_rgba(bad)
        if outside is not None:
            new_cm._rgba_outside = to_rgba(outside)
        if shape is not None:
            _api.check_in_list(['square', 'circle', 'ignore', 'circleignore'],
                               shape=shape)
            new_cm._shape = shape
        if origin is not None:
            new_cm._origin = (float(origin[0]), float(origin[1]))

        return new_cm

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError("Abstract class only")

    @property
    def shape(self):
        return self._shape

    @property
    def origin(self):
        return self._origin

    def _clip(self, X):
        """
        For internal use when applying a BivarColormap to data.
        i.e. cm.ScalarMappable().to_rgba()
        Clips X[0] and X[1] according to 'self.shape'.
        X is modified in-place.

        Parameters
        ----------
        X: np.array
            array of floats or ints to be clipped
        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - If 'square' each variate is clipped to [0,1] independently
            - If 'circle' the variates are clipped radially to the center
              of the colormap.
              It is assumed that a circular mask is applied when the colormap
              is displayed
            - If 'ignore' the variates are not clipped, but instead assigned the
              'outside' color
            - If 'circleignore' a circular mask is applied, but the data is not clipped
              and instead assigned the 'outside' color

        """
        if self.shape == 'square':
            for X_part, mx in zip(X, (self.N, self.M)):
                X_part[X_part < 0] = 0
                if X_part.dtype.kind == "f":
                    X_part[X_part > 1] = 1
                else:
                    X_part[X_part >= mx] = mx - 1

        elif self.shape == 'ignore':
            for X_part, mx in zip(X, (self.N, self.M)):
                X_part[X_part < 0] = -1
                if X_part.dtype.kind == "f":
                    X_part[X_part > 1] = -1
                else:
                    X_part[X_part >= mx] = -1

        elif self.shape == 'circle' or self.shape == 'circleignore':
            for X_part in X:
                if X_part.dtype.kind != "f":
                    raise NotImplementedError(
                        "Circular bivariate colormaps are only"
                        " implemented for use with with floats")
            radii_sqr = (X[0] - 0.5)**2 + (X[1] - 0.5)**2
            mask_outside = radii_sqr > 0.25
            if self.shape == 'circle':
                overextend = 2 * np.sqrt(radii_sqr[mask_outside])
                X[0][mask_outside] = (X[0][mask_outside] - 0.5) / overextend + 0.5
                X[1][mask_outside] = (X[1][mask_outside] - 0.5) / overextend + 0.5
            else:
                X[0][mask_outside] = -1
                X[1][mask_outside] = -1

    def __getitem__(self, item):
        """Creates and returns a colorbar along the selected axis"""
        if not self._isinit:
            self._init()
        if item == 0:
            origin_1_as_int = int(self._origin[1]*self.M)
            if origin_1_as_int > self.M-1:
                origin_1_as_int = self.M-1
            one_d_lut = self._lut[:, origin_1_as_int]
            new_cmap = ListedColormap(one_d_lut, name=f'{self.name}_0', N=self.N)

        elif item == 1:
            origin_0_as_int = int(self._origin[0]*self.N)
            if origin_0_as_int > self.N-1:
                origin_0_as_int = self.N-1
            one_d_lut = self._lut[origin_0_as_int, :]
            new_cmap = ListedColormap(one_d_lut, name=f'{self.name}_1', N=self.M)
        else:
            raise KeyError(f"only 0 or 1 are"
                           f" valid keys for BivarColormap, not {item!r}")
        new_cmap._rgba_bad = self._rgba_bad
        if self.shape in ['ignore', 'circleignore']:
            new_cmap.set_over(self._rgba_outside)
            new_cmap.set_under(self._rgba_outside)
        return new_cmap

    def _repr_png_(self):
        """Generate a PNG representation of the BivarColormap."""
        if not self._isinit:
            self._init()
        pixels = self.lut
        if pixels.shape[0] < _BIVAR_REPR_PNG_SIZE:
            pixels = np.repeat(pixels,
                               repeats=_BIVAR_REPR_PNG_SIZE//pixels.shape[0],
                               axis=0)[:256, :]
        if pixels.shape[1] < _BIVAR_REPR_PNG_SIZE:
            pixels = np.repeat(pixels,
                               repeats=_BIVAR_REPR_PNG_SIZE//pixels.shape[1],
                               axis=1)[:, :256]
        pixels = (pixels[::-1, :, :] * 255).astype(np.uint8)
        png_bytes = io.BytesIO()
        title = self.name + ' BivarColormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        def color_block(color):
            hex_color = to_hex(color, keep_alpha=True)
            return (f'<div title="{hex_color}" '
                    'style="display: inline-block; '
                    'width: 1em; height: 1em; '
                    'margin: 0; '
                    'vertical-align: middle; '
                    'border: 1px solid #555; '
                    f'background-color: {hex_color};"></div>')

        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.name}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.name} BivarColormap" '
                f'title="{self.name}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {_BIVAR_REPR_PNG_SIZE+2}px; '
                'display: flex; justify-content: space-between;">'
                '<div style="float: left;">'
                f'{color_block(self.get_outside())} outside'
                '</div>'
                '<div style="float: right;">'
                f'bad {color_block(self.get_bad())}'
                '</div></div>')

    def copy(self):
        """Return a copy of the colormap."""
        return self.__copy__()


class SegmentedBivarColormap(BivarColormap):
    """
    BivarColormap object generated by supersampling a regular grid.

    Parameters
    ----------
    patch : np.array
        Patch is required to have a shape (k, l, 3), and will get supersampled
        to a lut of shape (N, N, 4).
    N : int
        The number of RGB quantization levels along each axis.
    shape : {'square', 'circle', 'ignore', 'circleignore'}

        - If 'square' each variate is clipped to [0,1] independently
        - If 'circle' the variates are clipped radially to the center
          of the colormap, and a circular mask is applied when the colormap
          is displayed
        - If 'ignore' the variates are not clipped, but instead assigned the
          'outside' color
        - If 'circleignore' a circular mask is applied, but the data is not clipped

    origin : (float, float)
        The relative origin of the colormap. Typically (0, 0), for colormaps
        that are linear on both axis, and (.5, .5) for circular colormaps.
        Used when getting 1D colormaps from 2D colormaps.

    name : str, optional
        The name of the colormap.
    """

    def __init__(self, patch, N=256, shape='square', origin=(0, 0),
                 name='segmented bivariate colormap'):
        _api.check_shape((None, None, 3), patch=patch)
        self.patch = patch
        super().__init__(N, N, shape, origin, name=name)

    def _init(self):
        s = self.patch.shape
        _patch = np.empty((s[0], s[1], 4))
        _patch[:, :, :3] = self.patch
        _patch[:, :, 3] = 1
        transform = mpl.transforms.Affine2D().translate(-0.5, -0.5)\
                                .scale(self.N / (s[1] - 1), self.N / (s[0] - 1))
        self._lut = np.empty((self.N, self.N, 4))

        _image.resample(_patch, self._lut, transform, _image.BILINEAR,
                        resample=False, alpha=1)
        self._isinit = True


class BivarColormapFromImage(BivarColormap):
    """
    BivarColormap object generated by supersampling a regular grid.

    Parameters
    ----------
    lut : nparray of shape (N, M, 3) or (N, M, 4)
        The look-up-table
    shape: {'square', 'circle', 'ignore', 'circleignore'}

        - If 'square' each variate is clipped to [0,1] independently
        - If 'circle' the variates are clipped radially to the center
          of the colormap, and a circular mask is applied when the colormap
          is displayed
        - If 'ignore' the variates are not clipped, but instead assigned the
          'outside' color
        - If 'circleignore' a circular mask is applied, but the data is not clipped

    origin: (float, float)
        The relative origin of the colormap. Typically (0, 0), for colormaps
        that are linear on both axis, and (.5, .5) for circular colormaps.
        Used when getting 1D colormaps from 2D colormaps.
    name : str, optional
        The name of the colormap.

    """

    def __init__(self, lut, shape='square', origin=(0, 0), name='from image'):
        # We can allow for a PIL.Image as input in the following way, but importing
        # matplotlib.image.pil_to_array() results in a circular import
        # For now, this function only accepts numpy arrays.
        # i.e.:
        # if isinstance(Image, lut):
        #    lut = image.pil_to_array(lut)
        lut = np.array(lut, copy=True)
        if lut.ndim != 3 or lut.shape[2] not in (3, 4):
            raise ValueError("The lut must be an array of shape (n, m, 3) or (n, m, 4)",
                             " or a PIL.image encoded as RGB or RGBA")

        if lut.dtype == np.uint8:
            lut = lut.astype(np.float32)/255
        if lut.shape[2] == 3:
            new_lut = np.empty((lut.shape[0], lut.shape[1], 4), dtype=lut.dtype)
            new_lut[:, :, :3] = lut
            new_lut[:, :, 3] = 1.
            lut = new_lut
        self._lut = lut
        super().__init__(lut.shape[0], lut.shape[1], shape, origin, name=name)

    def _init(self):
        self._isinit = True


class Normalize:
    """
    A class which, when called, maps values within the interval
    ``[vmin, vmax]`` linearly to the interval ``[0.0, 1.0]``. The mapping of
    values outside ``[vmin, vmax]`` depends on *clip*.

    Examples
    --------
    ::

        x = [-2, -1, 0, 1, 2]

        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        norm(x)  # [-0.5, 0., 0.5, 1., 1.5]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=True)
        norm(x)  # [0., 0., 0.5, 1., 1.]

    See Also
    --------
    :ref:`colormapnorms`
    """

    def __init__(self, vmin=None, vmax=None, clip=False):
        """
        Parameters
        ----------
        vmin, vmax : float or None
            Values within the range ``[vmin, vmax]`` from the input data will be
            linearly mapped to ``[0, 1]``. If either *vmin* or *vmax* is not
            provided, they default to the minimum and maximum values of the input,
            respectively.

        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Notes
        -----
        If ``vmin == vmax``, input data will be mapped to 0.
        """
        self._vmin = _sanitize_extrema(vmin)
        self._vmax = _sanitize_extrema(vmax)
        self._clip = clip
        self._scale = None
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmin:
            self._vmin = value
            self._changed()

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmax:
            self._vmax = value
            self._changed()

    @property
    def clip(self):
        return self._clip

    @clip.setter
    def clip(self, value):
        if value != self._clip:
            self._clip = value
            self._changed()

    def _changed(self):
        """
        Call this whenever the norm is changed to notify all the
        callback listeners to the 'changed' signal.
        """
        self.callbacks.process('changed')

    @staticmethod
    def process_value(value):
        """
        Homogenize the input *value* for easy and efficient normalization.

        *value* can be a scalar or sequence.

        Parameters
        ----------
        value
            Data to normalize.

        Returns
        -------
        result : masked array
            Masked array with the same shape as *value*.
        is_scalar : bool
            Whether *value* is a scalar.

        Notes
        -----
        Float dtypes are preserved; integer types with two bytes or smaller are
        converted to np.float32, and larger types are converted to np.float64.
        Preserving float32 when possible, and using in-place operations,
        greatly improves speed for large arrays.
        """
        is_scalar = not np.iterable(value)
        if is_scalar:
            value = [value]
        dtype = np.min_scalar_type(value)
        if np.issubdtype(dtype, np.integer) or dtype.type is np.bool_:
            # bool_/int8/int16 -> float32; int32/int64 -> float64
            dtype = np.promote_types(dtype, np.float32)
        # ensure data passed in as an ndarray subclass are interpreted as
        # an ndarray. See issue #6622.
        mask = np.ma.getmask(value)
        data = np.asarray(value)
        result = np.ma.array(data, mask=mask, dtype=dtype, copy=True)
        return result, is_scalar

    def __call__(self, value, clip=None):
        """
        Normalize the data and return the normalized data.

        Parameters
        ----------
        value
            Data to normalize.
        clip : bool, optional
            See the description of the parameter *clip* in `.Normalize`.

            If ``None``, defaults to ``self.clip`` (which defaults to
            ``False``).

        Notes
        -----
        If not already initialized, ``self.vmin`` and ``self.vmax`` are
        initialized using ``self.autoscale_None(value)``.
        """
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        if self.vmin is None or self.vmax is None:
            self.autoscale_None(result)
        # Convert at least to float, without losing precision.
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        if vmin == vmax:
            result.fill(0)  # Or should it be all masked?  Or 0.5?
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            # ma division is very slow; we can take a shortcut
            resdat = result.data
            resdat -= vmin
            resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        """
        Maps the normalized value (i.e., index in the colormap) back to image
        data value.

        Parameters
        ----------
        value
            Normalized value.
        """
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)

        if np.iterable(value):
            val = np.ma.asarray(value)
            return vmin + val * (vmax - vmin)
        else:
            return vmin + value * (vmax - vmin)

    def autoscale(self, A):
        """Set *vmin*, *vmax* to min, max of *A*."""
        with self.callbacks.blocked():
            # Pause callbacks while we are updating so we only get
            # a single update signal at the end
            self.vmin = self.vmax = None
            self.autoscale_None(A)
        self._changed()

    def autoscale_None(self, A):
        """If *vmin* or *vmax* are not set, use the min/max of *A* to set them."""
        A = np.asanyarray(A)

        if isinstance(A, np.ma.MaskedArray):
            # we need to make the distinction between an array, False, np.bool_(False)
            if A.mask is False or not A.mask.shape:
                A = A.data

        if self.vmin is None and A.size:
            self.vmin = A.min()
        if self.vmax is None and A.size:
            self.vmax = A.max()

    def scaled(self):
        """Return whether *vmin* and *vmax* are both set."""
        return self.vmin is not None and self.vmax is not None


class TwoSlopeNorm(Normalize):
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
            ...                               vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """

        super().__init__(vmin=vmin, vmax=vmax)
        self._vcenter = vcenter
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, value):
        if value != self._vcenter:
            self._vcenter = value
            self._changed()

    def autoscale_None(self, A):
        """
        Get vmin and vmax.

        If vcenter isn't in the range [vmin, vmax], either vmin or vmax
        is expanded so that vcenter lies in the middle of the modified range
        [vmin, vmax].
        """
        super().autoscale_None(A)
        if self.vmin >= self.vcenter:
            self.vmin = self.vcenter - (self.vmax - self.vcenter)
        if self.vmax <= self.vcenter:
            self.vmax = self.vcenter + (self.vcenter - self.vmin)

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The *clip* argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        # note that we must extrapolate for tick locators:
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1], left=-np.inf, right=np.inf),
            mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)
        result = np.interp(value, [0, 0.5, 1], [vmin, vcenter, vmax],
                           left=-np.inf, right=np.inf)
        return result


class CenteredNorm(Normalize):
    def __init__(self, vcenter=0, halfrange=None, clip=False):
        """
        Normalize symmetrical data around a center (0 by default).

        Unlike `TwoSlopeNorm`, `CenteredNorm` applies an equal rate of change
        around the center.

        Useful when mapping symmetrical data around a conceptual center
        e.g., data that range from -2 to 4, with 0 as the midpoint, and
        with equal rates of change around that midpoint.

        Parameters
        ----------
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Examples
        --------
        This maps data values -2 to 0.25, 0 to 0.5, and 4 to 1.0
        (assuming equal rates of change above and below 0.0):

            >>> import matplotlib.colors as mcolors
            >>> norm = mcolors.CenteredNorm(halfrange=4.0)
            >>> data = [-2., 0., 4.]
            >>> norm(data)
            array([0.25, 0.5 , 1.  ])
        """
        super().__init__(vmin=None, vmax=None, clip=clip)
        self._vcenter = vcenter
        # calling the halfrange setter to set vmin and vmax
        self.halfrange = halfrange

    def autoscale(self, A):
        """
