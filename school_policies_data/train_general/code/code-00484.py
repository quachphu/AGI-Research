       div_expr = div_any_type + SkipTo(div | div_end)("body")
       for div_header in div_expr.search_string(html):
           print(div_header.body)

    prints:

    .. testoutput::

       1 4 0 1 0
       1,3 2,3 1,1
    """
    classattr = f"{namespace}:class" if namespace else "class"
    return with_attribute(**{classattr: classname})


# Compatibility synonyms
# fmt: off
replaceWith = replaced_by_pep8("replaceWith", replace_with)
removeQuotes = replaced_by_pep8("removeQuotes", remove_quotes)
withAttribute = replaced_by_pep8("withAttribute", with_attribute)
withClass = replaced_by_pep8("withClass", with_class)
matchOnlyAtCol = replaced_by_pep8("matchOnlyAtCol", match_only_at_col)
# fmt: on
<|endoftext|># helpers.py
from enum import Enum, auto
import html.entities
import operator
import re
import sys
import typing

from . import __diag__
from .core import *
from .util import (
    _bslash,
    _flatten,
    _escape_regex_range_chars,
    make_compressed_re,
    replaced_by_pep8,
)


def _suppression(expr: Union[ParserElement, str]) -> ParserElement:
    # internal helper to avoid wrapping Suppress inside another Suppress
    if isinstance(expr, Suppress):
        return expr
    return Suppress(expr)


#
# global helpers
#
def counted_array(
    expr: ParserElement, int_expr: typing.Optional[ParserElement] = None, **kwargs
) -> ParserElement:
    """Helper to define a counted list of expressions.

    This helper defines a pattern of the form::

        integer expr expr expr...

    where the leading integer tells how many expr expressions follow.
    The matched tokens returns the array of expr tokens as a list - the
    leading count token is suppressed.

    If ``int_expr`` is specified, it should be a pyparsing expression
    that produces an integer value.

    Examples:

    .. doctest::

        >>> counted_array(Word(alphas)).parse_string('2 ab cd ef')
        ParseResults(['ab', 'cd'], {})

    - In this parser, the leading integer value is given in binary,
      '10' indicating that 2 values are in the array:

      .. doctest::

        >>> binary_constant = Word('01').set_parse_action(lambda t: int(t[0], 2))
        >>> counted_array(Word(alphas), int_expr=binary_constant
        ...     ).parse_string('10 ab cd ef')
        ParseResults(['ab', 'cd'], {})

    - If other fields must be parsed after the count but before the
      list items, give the fields results names and they will
      be preserved in the returned ParseResults:

      .. doctest::

         >>> ppc = pyparsing.common
         >>> count_with_metadata = ppc.integer + Word(alphas)("type")
         >>> typed_array = counted_array(Word(alphanums),
         ...     int_expr=count_with_metadata)("items")
         >>> result = typed_array.parse_string("3 bool True True False")
         >>> print(result.dump())
         ['True', 'True', 'False']
         - items: ['True', 'True', 'False']
         - type: 'bool'
    """
    intExpr: typing.Optional[ParserElement] = deprecate_argument(
        kwargs, "intExpr", None
    )

    intExpr = intExpr or int_expr
    array_expr = Forward()

    def count_field_parse_action(s, l, t):
        nonlocal array_expr
        n = t[0]
        array_expr <<= (expr * n) if n else Empty()
        # clear list contents, but keep any named results
        del t[:]

    if intExpr is None:
        intExpr = Word(nums).set_parse_action(lambda t: int(t[0]))
    else:
        intExpr = intExpr.copy()
    intExpr.set_name("arrayLen")
    intExpr.add_parse_action(count_field_parse_action, call_during_try=True)
    return (intExpr + array_expr).set_name(f"(len) {expr}...")


def match_previous_literal(expr: ParserElement) -> ParserElement:
    """Helper to define an expression that is indirectly defined from
    the tokens matched in a previous expression, that is, it looks for
    a 'repeat' of a previous expression.  For example::

    .. testcode::

       first = Word(nums)
       second = match_previous_literal(first)
       match_expr = first + ":" + second

    will match ``"1:1"``, but not ``"1:2"``.  Because this
    matches a previous literal, will also match the leading
    ``"1:1"`` in ``"1:10"``. If this is not desired, use
    :class:`match_previous_expr`. Do *not* use with packrat parsing
    enabled.
    """
    rep = Forward()

    def copy_token_to_repeater(s, l, t):
        if not t:
            rep << Empty()
            return

        if len(t) == 1:
            rep << t[0]
            return

        # flatten t tokens
        tflat = _flatten(t.as_list())
        rep << And(Literal(tt) for tt in tflat)

    expr.add_parse_action(copy_token_to_repeater, call_during_try=True)
    rep.set_name(f"(prev) {expr}")
    return rep


def match_previous_expr(expr: ParserElement) -> ParserElement:
    """Helper to define an expression that is indirectly defined from
    the tokens matched in a previous expression, that is, it looks for
    a 'repeat' of a previous expression.  For example:

    .. testcode::

       first = Word(nums)
       second = match_previous_expr(first)
       match_expr = first + ":" + second

    will match ``"1:1"``, but not ``"1:2"``.  Because this
    matches by expressions, will *not* match the leading ``"1:1"``
    in ``"1:10"``; the expressions are evaluated first, and then
    compared, so ``"1"`` is compared with ``"10"``. Do *not* use
    with packrat parsing enabled.
    """
    rep = Forward()
    e2 = expr.copy()
    rep <<= e2

    def copy_token_to_repeater(s, l, t):
        matchTokens = _flatten(t.as_list())

        def must_match_these_tokens(s, l, t):
            theseTokens = _flatten(t.as_list())
            if theseTokens != matchTokens:
                raise ParseException(
                    s, l, f"Expected {matchTokens}, found{theseTokens}"
                )

        rep.set_parse_action(must_match_these_tokens, call_during_try=True)

    expr.add_parse_action(copy_token_to_repeater, call_during_try=True)
    rep.set_name(f"(prev) {expr}")
    return rep


def one_of(
    strs: Union[typing.Iterable[str], str],
    caseless: bool = False,
    use_regex: bool = True,
    as_keyword: bool = False,
    **kwargs,
) -> ParserElement:
    """Helper to quickly define a set of alternative :class:`Literal` s,
    and makes sure to do longest-first testing when there is a conflict,
    regardless of the input order, but returns
    a :class:`MatchFirst` for best performance.

    :param strs: a string of space-delimited literals, or a collection of
       string literals
    :param caseless: treat all literals as caseless
    :param use_regex: bool - as an optimization, will
       generate a :class:`Regex` object; otherwise, will generate
       a :class:`MatchFirst` object (if ``caseless=True`` or
       ``as_keyword=True``, or if creating a :class:`Regex` raises an exception)
    :param as_keyword: bool - enforce :class:`Keyword`-style matching on the
       generated expressions

    Parameters ``asKeyword`` and ``useRegex`` are retained for pre-PEP8
    compatibility, but will be removed in a future release.

    Example:

    .. testcode::

       comp_oper = one_of("< = > <= >= !=")
       var = Word(alphas)
       number = Word(nums)
       term = var | number
       comparison_expr = term + comp_oper + term
       print(comparison_expr.search_string("B = 12  AA=23 B<=AA AA>12"))

    prints:

    .. testoutput::

       [['B', '=', '12'], ['AA', '=', '23'], ['B', '<=', 'AA'], ['AA', '>', '12']]
    """
    useRegex: bool = deprecate_argument(kwargs, "useRegex", True)
    asKeyword: bool = deprecate_argument(kwargs, "asKeyword", False)

    asKeyword = asKeyword or as_keyword
    useRegex = useRegex and use_regex

    if (
        isinstance(caseless, str_type)
        and __diag__.warn_on_multiple_string_args_to_oneof
    ):
        warnings.warn(
            "warn_on_multiple_string_args_to_oneof:"
            " More than one string argument passed to one_of, pass"
            " choices as a list or space-delimited string",
            PyparsingDiagnosticWarning,
            stacklevel=2,
        )

    if caseless:
        is_equal = lambda a, b: a.upper() == b.upper()
        masks = lambda a, b: b.upper().startswith(a.upper())
    else:
        is_equal = operator.eq
        masks = lambda a, b: b.startswith(a)

    symbols: list[str]
    if isinstance(strs, str_type):
        strs = typing.cast(str, strs)
        symbols = strs.split()
    elif isinstance(strs, Iterable):
        symbols = list(strs)
    else:
        raise TypeError("Invalid argument to one_of, expected string or iterable")
    if not symbols:
        return NoMatch()

    # reorder given symbols to take care to avoid masking longer choices with shorter ones
    # (but only if the given symbols are not just single characters)
    i = 0
    while i < len(symbols) - 1:
        cur = symbols[i]
        for j, other in enumerate(symbols[i + 1 :]):
            if is_equal(other, cur):
                del symbols[i + j + 1]
                break
            if len(other) > len(cur) and masks(cur, other):
                del symbols[i + j + 1]
                symbols.insert(i, other)
                break
        else:
            i += 1

    if useRegex:
        re_flags: int = re.IGNORECASE if caseless else 0

        try:
            if all(len(sym) == 1 for sym in symbols):
                # symbols are just single characters, create range regex pattern
                patt = f"[{''.join(_escape_regex_range_chars(sym) for sym in symbols)}]"
            else:
                patt = "|".join(re.escape(sym) for sym in symbols)

            # wrap with \b word break markers if defining as keywords
            if asKeyword:
                patt = rf"\b(?:{patt})\b"

            ret = Regex(patt, flags=re_flags)
            ret.set_name(" | ".join(repr(s) for s in symbols))

            if caseless:
                # add parse action to return symbols as specified, not in random
                # casing as found in input string
                symbol_map = {sym.lower(): sym for sym in symbols}
                ret.add_parse_action(lambda s, l, t: symbol_map[t[0].lower()])

            return ret

        except re.error:
            warnings.warn(
                "Exception creating Regex for one_of, building MatchFirst",
                PyparsingDiagnosticWarning,
                stacklevel=2,
            )

    # last resort, just use MatchFirst of Token class corresponding to caseless
    # and asKeyword settings
    CASELESS = KEYWORD = True
    parse_element_class = {
        (CASELESS, KEYWORD): CaselessKeyword,
        (CASELESS, not KEYWORD): CaselessLiteral,
        (not CASELESS, KEYWORD): Keyword,
        (not CASELESS, not KEYWORD): Literal,
    }[(caseless, asKeyword)]
    return MatchFirst(parse_element_class(sym) for sym in symbols).set_name(
        " | ".join(symbols)
    )


def dict_of(key: ParserElement, value: ParserElement) -> Dict:
    """Helper to easily and clearly define a dictionary by specifying
    the respective patterns for the key and value.  Takes care of
    defining the :class:`Dict`, :class:`ZeroOrMore`, and
    :class:`Group` tokens in the proper order.  The key pattern
    can include delimiting markers or punctuation, as long as they are
    suppressed, thereby leaving the significant key text.  The value
    pattern can include named results, so that the :class:`Dict` results
    can include named token fields.

    Example:

    .. doctest::

       >>> text = "shape: SQUARE posn: upper left color: light blue texture: burlap"

       >>> data_word = Word(alphas)
       >>> label = data_word + FollowedBy(':')
       >>> attr_expr = (
       ...    label
       ...    + Suppress(':')
       ...    + OneOrMore(data_word, stop_on=label)
       ...    .set_parse_action(' '.join))
       >>> print(attr_expr[1, ...].parse_string(text).dump())
       ['shape', 'SQUARE', 'posn', 'upper left', 'color', 'light blue', 'texture', 'burlap']

       >>> attr_label = label
       >>> attr_value = Suppress(':') + OneOrMore(data_word, stop_on=label
       ...   ).set_parse_action(' '.join)

       # similar to Dict, but simpler call format
       >>> result = dict_of(attr_label, attr_value).parse_string(text)
       >>> print(result.dump())
       [['shape', 'SQUARE'], ['posn', 'upper left'], ['color', 'light blue'], ['texture', 'burlap']]
       - color: 'light blue'
       - posn: 'upper left'
       - shape: 'SQUARE'
       - texture: 'burlap'
       [0]:
         ['shape', 'SQUARE']
       [1]:
         ['posn', 'upper left']
       [2]:
         ['color', 'light blue']
       [3]:
         ['texture', 'burlap']

       >>> print(result['shape'])
       SQUARE
       >>> print(result.shape)  # object attribute access works too
       SQUARE
       >>> print(result.as_dict())
       {'shape': 'SQUARE', 'posn': 'upper left', 'color': 'light blue', 'texture': 'burlap'}
    """
    return Dict(OneOrMore(Group(key + value)))


def original_text_for(
    expr: ParserElement, as_string: bool = True, **kwargs
) -> ParserElement:
    """Helper to return the original, untokenized text for a given
    expression.  Useful to restore the parsed fields of an HTML start
    tag into the raw tag text itself, or to revert separate tokens with
    intervening whitespace back to the original matching input text. By
    default, returns a string containing the original parsed text.

    If the optional ``as_string`` argument is passed as
    ``False``, then the return value is
    a :class:`ParseResults` containing any results names that
    were originally matched, and a single token containing the original
    matched text from the input string.  So if the expression passed to
    :class:`original_text_for` contains expressions with defined
    results names, you must set ``as_string`` to ``False`` if you
    want to preserve those results name values.

    The ``asString`` pre-PEP8 argument is retained for compatibility,
    but will be removed in a future release.

    Example:

    .. testcode::

       src = "this is test <b> bold <i>text</i> </b> normal text "
       for tag in ("b", "i"):
           opener, closer = make_html_tags(tag)
           patt = original_text_for(opener + ... + closer)
           print(patt.search_string(src)[0])

    prints:

    .. testoutput::

       ['<b> bold <i>text</i> </b>']
       ['<i>text</i>']
    """
    asString: bool = deprecate_argument(kwargs, "asString", True)

    asString = asString and as_string

    locMarker = Empty().set_parse_action(lambda s, loc, t: loc)
    endlocMarker = locMarker.copy()
    endlocMarker.callPreparse = False
    matchExpr = locMarker("_original_start") + expr + endlocMarker("_original_end")
    if asString:
        extractText = lambda s, l, t: s[t._original_start : t._original_end]
    else:

        def extractText(s, l, t):
            t[:] = [s[t.pop("_original_start") : t.pop("_original_end")]]

    matchExpr.set_parse_action(extractText)
    matchExpr.ignoreExprs = expr.ignoreExprs
    matchExpr.suppress_warning(Diagnostics.warn_ungrouped_named_tokens_in_collection)
    return matchExpr


def ungroup(expr: ParserElement) -> ParserElement:
    """Helper to undo pyparsing's default grouping of And expressions,
    even if all but one are non-empty.
    """
    return TokenConverter(expr).add_parse_action(lambda t: t[0])


def locatedExpr(expr: ParserElement) -> ParserElement:
    """
    .. deprecated:: 3.0.0
       Use the :class:`Located` class instead. Note that `Located`
       returns results with one less grouping level.

    Helper to decorate a returned token with its starting and ending
    locations in the input string.

    This helper adds the following results names:

    - ``locn_start`` - location where matched expression begins
    - ``locn_end`` - location where matched expression ends
    - ``value`` - the actual parsed results

    Be careful if the input text contains ``<TAB>`` characters, you
    may want to call :meth:`ParserElement.parse_with_tabs`
    """
    warnings.warn(
        f"{'locatedExpr'!r} deprecated - use {'Located'!r}",
        PyparsingDeprecationWarning,
        stacklevel=2,
    )

    locator = Empty().set_parse_action(lambda ss, ll, tt: ll)
    return Group(
        locator("locn_start")
        + expr("value")
        + locator.copy().leave_whitespace()("locn_end")
    )


# define special default value to permit None as a significant value for
# ignore_expr
_NO_IGNORE_EXPR_GIVEN = NoMatch()


class _NestedExpr(ParseElementEnhance):
    """Helper method for defining nested lists enclosed in opening and
    closing delimiters (``"("`` and ``")"`` are the default).

    :param opener: str - opening character for a nested list
       (default= ``"("``); can also be a pyparsing expression

    :param closer: str - closing character for a nested list
       (default= ``")"``); can also be a pyparsing expression

    :param content: expression for items within the nested lists

    :param ignore_expr: expression for ignoring opening and closing delimiters
       (default = :class:`quoted_string`)

    Parameter ``ignoreExpr`` is retained for compatibility
    but will be removed in a future release.

    If an expression is not provided for the content argument, the
    nested expression will capture all whitespace-delimited content
    between delimiters as a list of separate values.

    Use the ``ignore_expr`` argument to define expressions that may
    contain opening or closing characters that should not be treated as
    opening or closing characters for nesting, such as quoted_string or
    a comment expression.  Specify multiple expressions using an
    :class:`Or` or :class:`MatchFirst`. The default is
    :class:`quoted_string`, but if no expressions are to be ignored, then
    pass ``None`` for this argument.

    Example:

    .. testcode::

       data_type = one_of("void int short long char float double")
       decl_data_type = Combine(data_type + Opt(Word('*')))
       ident = Word(alphas+'_', alphanums+'_')
       number = pyparsing_common.number
       arg = Group(decl_data_type + ident)
       LPAR, RPAR = map(Suppress, "()")

       code_body = nested_expr('{', '}', ignore_expr=(quoted_string | c_style_comment))

       c_function = (decl_data_type("type")
                     + ident("name")
                     + LPAR + Opt(DelimitedList(arg), [])("args") + RPAR
                     + code_body("body"))
       c_function.ignore(c_style_comment)

       source_code = '''
           int is_odd(int x) {
               return (x%2);
           }

           int dec_to_hex(char hchar) {
               if (hchar >= '0' && hchar <= '9') {
                   return (ord(hchar)-ord('0'));
               } else {
                   return (10+ord(hchar)-ord('A'));
               }
           }
       '''
       for func in c_function.search_string(source_code):
           print(f"{func.name} ({func.type}) args: {func.args}")


    prints:

    .. testoutput::

       is_odd (int) args: [['int', 'x']]
       dec_to_hex (int) args: [['char', 'hchar']]
    """

    def __init__(
        self,
        opener: Union[str, ParserElement] = "(",
        closer: Union[str, ParserElement] = ")",
        content: typing.Optional[ParserElement] = None,
        ignore_expr: typing.Optional[ParserElement] = _NO_IGNORE_EXPR_GIVEN,
        **kwargs,
    ):
        ignoreExpr = deprecate_argument(kwargs, "ignoreExpr", _NO_IGNORE_EXPR_GIVEN)
        if ignoreExpr != ignore_expr:
            ignoreExpr = (
                ignore_expr if ignoreExpr is _NO_IGNORE_EXPR_GIVEN else ignoreExpr
            )
        if ignoreExpr is _NO_IGNORE_EXPR_GIVEN:
            ignoreExpr = quoted_string()

        if opener == closer:
            raise ValueError("opening and closing strings cannot be the same")

        original_content = content
        if content is None:
            if isinstance(opener, str_type) and isinstance(closer, str_type):
                opener_str = str(opener)
                closer_str = str(closer)
                if len(opener_str) == 1 and len(closer_str) == 1:
                    if ignoreExpr is not None:
                        content = Combine(
                            OneOrMore(
                                ~ignoreExpr
                                + CharsNotIn(
                                    opener_str
                                    + closer_str
                                    + ParserElement.DEFAULT_WHITE_CHARS,
                                    exact=1,
                                )
                            )
                        )
                    else:
                        content = Combine(
                            Empty()
                            + CharsNotIn(
                                opener_str
                                + closer_str
                                + ParserElement.DEFAULT_WHITE_CHARS
                            )
                        )
                else:
                    if ignoreExpr is not None:
                        content = Combine(
                            OneOrMore(
                                ~ignoreExpr
                                + ~Literal(opener_str)
                                + ~Literal(closer_str)
                                + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1)
                            )
                        )
                    else:
                        content = Combine(
                            OneOrMore(
                                ~Literal(opener_str)
                                + ~Literal(closer_str)
                                + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1)
                            )
                        )
            else:
                raise ValueError(
                    "opening and closing arguments must be strings if no content expression is given"
                )

            if ParserElement.DEFAULT_WHITE_CHARS:
                content.set_parse_action(
                    lambda t: t[0].strip(ParserElement.DEFAULT_WHITE_CHARS)
                )

        super().__init__(content, savelist=True)
        self.opener = _suppression(opener)
        self.closer = _suppression(closer)
        self.opener_raw = opener
        self.closer_raw = closer
        self.content = content
        self.ignore_expr = ignoreExpr
        self.saveAsList = True
        self.errmsg = None
        self.original_content = original_content

    def _generateDefaultName(self) -> str:
        if self.original_content is None:
            return f"nested {self.opener_raw}{self.closer_raw} expression"
        else:
            return f"nested {self.opener_raw}{self.original_content}{self.closer_raw} expression"

    def parseImpl(self, instring, loc, do_actions=True):
        loc, _ = self.opener._parse(instring, loc, do_actions=do_actions)
        stack = [[]]
        while stack:
            # 1. try ignore_expr
            if self.ignore_expr is not None:
                try:
                    loc, toks = self.ignore_expr._parse(
                        instring, loc, do_actions=do_actions
                    )
                    stack[-1].extend(toks)
                    continue
                except ParseException:
                    pass
            # 2. try opener
            try:
                loc, _ = self.opener._parse(instring, loc, do_actions=do_actions)
                stack.append([])
                continue
            except ParseException:
                pass
            # 3. try content
            try:
                next_loc, toks = self.content._parse(
                    instring, loc, do_actions=do_actions
                )
                if next_loc > loc:
                    loc = next_loc
                    stack[-1].extend(toks)
                    continue
            except ParseException:
                pass
            # 4. try closer
            try:
                loc, _ = self.closer._parse(instring, loc, do_actions=do_actions)
                top = ParseResults(stack.pop())
                if stack:
                    stack[-1].append(top)
                else:
                    return loc, ParseResults([top])
                continue
            except ParseException:
                pass

            raise ParseException(instring, loc, f"Expected {self.closer_raw!r}")


def nested_expr(
    opener: Union[str, ParserElement] = "(",
    closer: Union[str, ParserElement] = ")",
    content: typing.Optional[ParserElement] = None,
    ignore_expr: typing.Optional[ParserElement] = _NO_IGNORE_EXPR_GIVEN,
    **kwargs,
) -> ParserElement:
    """Helper method for defining nested lists enclosed in opening and
    closing delimiters (``"("`` and ``")"`` are the default).

    :param opener: str - opening character for a nested list
       (default= ``"("``); can also be a pyparsing expression

    :param closer: str - closing character for a nested list
       (default= ``")"``); can also be a pyparsing expression

    :param content: expression for items within the nested lists

    :param ignore_expr: expression for ignoring opening and closing delimiters
       (default = :class:`quoted_string`)

    Parameter ``ignoreExpr`` is retained for compatibility
    but will be removed in a future release.

    If an expression is not provided for the content argument, the
    nested expression will capture all whitespace-delimited content
    between delimiters as a list of separate values.

    Use the ``ignore_expr`` argument to define expressions that may
    contain opening or closing characters that should not be treated as
    opening or closing characters for nesting, such as quoted_string or
    a comment expression.  Specify multiple expressions using an
    :class:`Or` or :class:`MatchFirst`. The default is
    :class:`quoted_string`, but if no expressions are to be ignored, then
    pass ``None`` for this argument.

    Example:

    .. testcode::

       data_type = one_of("void int short long char float double")
       decl_data_type = Combine(data_type + Opt(Word('*')))
       ident = Word(alphas+'_', alphanums+'_')
       number = pyparsing_common.number
       arg = Group(decl_data_type + ident)
       LPAR, RPAR = map(Suppress, "()")

       code_body = nested_expr('{', '}', ignore_expr=(quoted_string | c_style_comment))

       c_function = (decl_data_type("type")
                     + ident("name")
                     + LPAR + Opt(DelimitedList(arg), [])("args") + RPAR
                     + code_body("body"))
       c_function.ignore(c_style_comment)

       source_code = '''
           int is_odd(int x) {
               return (x%2);
           }

           int dec_to_hex(char hchar) {
               if (hchar >= '0' && hchar <= '9') {
                   return (ord(hchar)-ord('0'));
               } else {
                   return (10+ord(hchar)-ord('A'));
               }
           }
       '''
       for func in c_function.search_string(source_code):
           print(f"{func.name} ({func.type}) args: {func.args}")


    prints:

    .. testoutput::

       is_odd (int) args: [['int', 'x']]
       dec_to_hex (int) args: [['char', 'hchar']]
    """
    return _NestedExpr(
        opener, closer, content=content, ignore_expr=ignore_expr, **kwargs
    )


def _makeTags(tagStr, xml, suppress_LT=Suppress("<"), suppress_GT=Suppress(">")):
    """Internal helper to construct opening and closing tag expressions,
    given a tag name"""
    if isinstance(tagStr, str_type):
        resname = tagStr
        tagStr = Keyword(tagStr, caseless=not xml)
    else:
        resname = tagStr.name

    tagAttrName = Word(alphas, alphanums + "_-:")
    if xml:
        tagAttrValue = dbl_quoted_string.copy().set_parse_action(remove_quotes)
        openTag = (
            suppress_LT
            + tagStr("tag")
            + Dict(ZeroOrMore(Group(tagAttrName + Suppress("=") + tagAttrValue)))
            + Opt("/", default=[False])("empty").set_parse_action(
                lambda s, l, t: t[0] == "/"
            )
            + suppress_GT
        )
    else:
        tagAttrValue = quoted_string.copy().set_parse_action(remove_quotes) | Word(
            printables, exclude_chars=">"
        )
        openTag = (
            suppress_LT
            + tagStr("tag")
            + Dict(
                ZeroOrMore(
                    Group(
                        tagAttrName.set_parse_action(lambda t: t[0].lower())
                        + Opt(Suppress("=") + tagAttrValue)
                    )
                )
            )
            + Opt("/", default=[False])("empty").set_parse_action(
                lambda s, l, t: t[0] == "/"
            )
            + suppress_GT
        )
    closeTag = Combine(Literal("</") + tagStr + ">", adjacent=False)

    openTag.set_name(f"<{resname}>")
    # add start<tagname> results name in parse action now that ungrouped names are not reported at two levels
    openTag.add_parse_action(
        lambda t: t.__setitem__(
            "start" + "".join(resname.replace(":", " ").title().split()), t.copy()
        )
    )
    closeTag = closeTag(
        "end" + "".join(resname.replace(":", " ").title().split())
    ).set_name(f"</{resname}>")
    openTag.tag = resname
    closeTag.tag = resname
    openTag.tag_body = SkipTo(closeTag())
    return openTag, closeTag


def make_html_tags(
    tag_str: Union[str, ParserElement],
) -> tuple[ParserElement, ParserElement]:
    """Helper to construct opening and closing tag expressions for HTML,
    given a tag name. Matches tags in either upper or lower case,
    attributes with namespaces and with quoted or unquoted values.

    Example:

    .. testcode::

       text = '<td>More info at the <a href="https://github.com/pyparsing/pyparsing/wiki">pyparsing</a> wiki page</td>'
       # make_html_tags returns pyparsing expressions for the opening and
       # closing tags as a 2-tuple
       a, a_end = make_html_tags("A")
       link_expr = a + SkipTo(a_end)("link_text") + a_end

       for link in link_expr.search_string(text):
           # attributes in the <A> tag (like "href" shown here) are
           # also accessible as named results
           print(link.link_text, '->', link.href)

    prints:

    .. testoutput::

       pyparsing -> https://github.com/pyparsing/pyparsing/wiki
    """
    return _makeTags(tag_str, False)


def make_xml_tags(
    tag_str: Union[str, ParserElement],
) -> tuple[ParserElement, ParserElement]:
    """Helper to construct opening and closing tag expressions for XML,
    given a tag name. Matches tags only in the given upper/lower case.

    Example: similar to :class:`make_html_tags`
    """
    return _makeTags(tag_str, True)


any_open_tag: ParserElement
any_close_tag: ParserElement
any_open_tag, any_close_tag = make_html_tags(
    Word(alphas, alphanums + "_:").set_name("any tag")
)

_htmlEntityMap = {k.rstrip(";"): v for k, v in html.entities.html5.items()}
_most_common_entities = "nbsp lt gt amp quot apos cent pound euro copy".replace(
    " ", "|"
)
common_html_entity = Regex(
    lambda: f"&(?P<entity>{_most_common_entities}|{make_compressed_re(_htmlEntityMap)});"
).set_name("common HTML entity")


def replace_html_entity(s, l, t):
    """Helper parser action to replace common HTML entities with their special characters"""
    return _htmlEntityMap.get(t.entity)


class OpAssoc(Enum):
    """Enumeration of operator associativity
    - used in constructing InfixNotationOperatorSpec for :class:`infix_notation`"""

    LEFT = 1
    RIGHT = 2


InfixNotationOperatorArgType = Union[
    ParserElement, str, tuple[Union[ParserElement, str], Union[ParserElement, str]]
]
InfixNotationOperatorSpec = Union[
    tuple[
        InfixNotationOperatorArgType,
        int,
        OpAssoc,
        typing.Optional[ParseAction],
    ],
    tuple[
        InfixNotationOperatorArgType,
        int,
        OpAssoc,
    ],
]


class _ParserState(Enum):
    EXPECT_OPERAND = auto()
    EXPECT_OPERATOR = auto()


class _OpType(Enum):
    PREFIX = auto()
    POSTFIX = auto()
    INFIX2 = auto()
    INFIX2_RIGHT = auto()
    TERNARY_LEFT_STAGE1 = auto()
    TERNARY_LEFT_STAGE2 = auto()
    TERNARY_RIGHT_STAGE1 = auto()
    TERNARY_RIGHT_STAGE2 = auto()
    LPAR = auto()


class _InfixNotationOperatorSpec(NamedTuple):
    op: InfixNotationOperatorArgType
    arity: int
    op_type: _OpType
    parse_action: typing.Optional[list[Callable]]


class _InfixNotation(ParseElementEnhance):
    """
    An iterative / stack-based implementation of infix_notation (using operator-precedence /
    shunting-yard style evaluation with an explicit stack).
    Prevents Python recursion limits from being exceeded on deeply nested expressions or long chains.
    """
    def __init__(
        self,
        base_expr: ParserElement,
        op_list: list[_InfixNotationOperatorSpec],
        lpar: Union[str, ParserElement] = Suppress("("),
        rpar: Union[str, ParserElement] = Suppress(")"),
    ):
        from .core import _trim_arity

        while isinstance(base_expr, _InfixNotation):
            op_list[:0] = base_expr.op_list[:]  # type: ignore[has-type]
            base_expr = base_expr.base_expr  # type: ignore[has-type]

        super().__init__(base_expr, savelist=True)
        self.base_expr = base_expr
        self.op_list = op_list
        self.lpar = Suppress(lpar) if isinstance(lpar, str) else lpar
        self.rpar = Suppress(rpar) if isinstance(rpar, str) else rpar
        self.keep_parens = not (isinstance(self.lpar, Suppress) and isinstance(self.rpar, Suppress))
        self.errmsg = None

        # Classify operators by arity and associativity
        # Precedence is defined by order in op_list (higher index in op_list = lower precedence)
        # We assign higher numeric precedence to earlier entries in op_list
        self.prefix_ops: list[_InfixNotationOperatorSpec] = []   # (expr, prec, assoc, pa_list)
        self.postfix_ops: list[_InfixNotationOperatorSpec] = []  # (expr, prec, assoc, pa_list)
        self.infix_ops = []  # type: ignore[var-annotated]

        total_ops = len(op_list)
        for idx, oper_def in enumerate(op_list):
            op_expr, arity, assoc, *pa_opt = (oper_def + (None,))[:4]
            pa = pa_opt[0] if pa_opt else None
            pa_list = (
                [_trim_arity(f) for f in pa]
                if isinstance(pa, (tuple, list))
                else ([_trim_arity(pa)] if pa is not None else [])
            )

            if not 1 <= arity <= 3:
                raise ValueError("operator must be unary (1), binary (2), or ternary (3)")

            if assoc not in (OpAssoc.LEFT, OpAssoc.RIGHT):
                raise ValueError("operator must indicate right or left associativity")

            # Precedence: top of list has highest precedence
            prec = (total_ops - idx) * 10

            if arity == 1:
                if isinstance(op_expr, str_type):
                    op_expr = Literal(op_expr)
                if assoc is OpAssoc.RIGHT:
                    self.prefix_ops.append((op_expr, prec, assoc, pa_list))
                else:
                    self.postfix_ops.append((op_expr, prec, assoc, pa_list))
            elif arity == 2:
                if op_expr is not None and isinstance(op_expr, str_type):
                    op_expr = Literal(op_expr)
                self.infix_ops.append((op_expr, prec, assoc, pa_list, 2, None))
            elif arity == 3:
                if not isinstance(op_expr, (tuple, list)) or len(op_expr) != 2:
                    raise ValueError(
                        "if numterms=3, opExpr must be a tuple or list of two expressions"
                    )
                op1, op2 = op_expr
                if isinstance(op1, str_type):
                    op1 = Literal(op1)
                if isinstance(op2, str_type):
                    op2 = Literal(op2)
                self.infix_ops.append((op1, prec, assoc, pa_list, 3, op2))

    def _generateDefaultName(self) -> str:
        return f"{self.base_expr} infix expression"

    def _run_parse_actions(self, pa_list, instring, loc, tokens):
        ret_tokens = tokens
        for fn in pa_list:
            res = fn(instring, loc, ret_tokens)
            if res is not None and res is not ret_tokens:
                if isinstance(res, (ParseResults, list, tuple)):
                    ret_tokens = ParseResults(res, aslist=True)
                else:
                    return res
        return ret_tokens

    def _apply_operator(self, op_info, operand_stack, instring):
        op_type = op_info["type"]
        pa_list = op_info.get("pa", [])
        start_loc = op_info.get("loc", 0)

        if op_type is _OpType.PREFIX:
            op_tok = op_info["op"]
            arg = operand_stack.pop()
            tokens = []
            if isinstance(op_tok, (list, ParseResults)):
                tokens.extend(op_tok)
            elif op_tok is not None:
                tokens.append(op_tok)
            tokens.append(arg)
            tokens_pr = ParseResults(tokens)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([tokens_pr]))
            else:
                res = tokens_pr
            operand_stack.append(res)

        elif op_type is _OpType.POSTFIX:
            arg = operand_stack.pop()
            if isinstance(arg, ParseResults):
                res_pr = arg.copy()
            elif isinstance(arg, list):
                res_pr = ParseResults(arg)
            else:
                res_pr = ParseResults([arg])
            for op_tok in op_info["ops"]:
                if isinstance(op_tok, ParseResults):
                    res_pr += op_tok
                elif isinstance(op_tok, list):
                    res_pr.extend(op_tok)
                elif op_tok is not None:
                    res_pr.append(op_tok)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([res_pr]))
            else:
                res = res_pr
            operand_stack.append(res)

        elif op_type is _OpType.INFIX2:
            ops = op_info["ops"]
            num_ops = len(ops)
            args = [operand_stack.pop() for _ in range(num_ops + 1)]
            args.reverse()
            tokens = [args[0]]
            for i in range(num_ops):
                op_tok = ops[i]
                if isinstance(op_tok, (list, ParseResults)):
                    tokens.extend(op_tok)
                elif op_tok is not None:
                    tokens.append(op_tok)
                tokens.append(args[i + 1])
            tokens_pr = ParseResults(tokens)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([tokens_pr]))
            else:
                res = tokens_pr
            operand_stack.append(res)

        elif op_type is _OpType.INFIX2_RIGHT:
            op_tok = op_info["op"]
            right = operand_stack.pop()
            left = operand_stack.pop()
            tokens = [left]
            if isinstance(op_tok, (list, ParseResults)):
                tokens.extend(op_tok)
            elif op_tok is not None:
                tokens.append(op_tok)
            tokens.append(right)
            tokens_pr = ParseResults(tokens)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([tokens_pr]))
            else:
                res = tokens_pr
            operand_stack.append(res)

        elif op_type is _OpType.TERNARY_LEFT_STAGE2:
            ops_list = op_info["ops"]
            num_ops = len(ops_list)
            num_operands = 2 * num_ops + 1
            args = [operand_stack.pop() for _ in range(num_operands)]
            args.reverse()
            tokens = [args[0]]
            for i in range(num_ops):
                op1_tok, op2_tok = ops_list[i]
                if isinstance(op1_tok, (list, ParseResults)):
                    tokens.extend(op1_tok)
                elif op1_tok is not None:
                    tokens.append(op1_tok)
                tokens.append(args[2 * i + 1])
                if isinstance(op2_tok, (list, ParseResults)):
                    tokens.extend(op2_tok)
                elif op2_tok is not None:
                    tokens.append(op2_tok)
                tokens.append(args[2 * i + 2])
            tokens_pr = ParseResults(tokens)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([tokens_pr]))
            else:
                res = tokens_pr
            operand_stack.append(res)

        elif op_type is _OpType.TERNARY_RIGHT_STAGE2:
            op1_tok = op_info["op1"]
            op2_tok = op_info["op2"]
            arg3 = operand_stack.pop()
            arg2 = operand_stack.pop()
            arg1 = operand_stack.pop()
            tokens = [arg1]
            if isinstance(op1_tok, (list, ParseResults)):
                tokens.extend(op1_tok)
            elif op1_tok is not None:
                tokens.append(op1_tok)
            tokens.append(arg2)
            if isinstance(op2_tok, (list, ParseResults)):
                tokens.extend(op2_tok)
            elif op2_tok is not None:
                tokens.append(op2_tok)
            tokens.append(arg3)
            tokens_pr = ParseResults(tokens)
            if pa_list:
                res = self._run_parse_actions(pa_list, instring, start_loc, ParseResults([tokens_pr]))
            else:
                res = tokens_pr
            operand_stack.append(res)

        return bool(pa_list)

    def parseImpl(self, instring, loc, do_actions=True):
        operand_stack = []
        operator_stack = []
        last_pa_applied = [False]

        def reduce_operators(min_prec):
            while operator_stack:
                top = operator_stack[-1]
                if top["type"] in (_OpType.LPAR, _OpType.TERNARY_LEFT_STAGE1, _OpType.TERNARY_RIGHT_STAGE1):
                    break
                if top["prec"] > min_prec:
                    op_info = operator_stack.pop()
                    last_pa_applied[0] = self._apply_operator(op_info, operand_stack, instring)
                else:
                    break

        state = _ParserState.EXPECT_OPERAND
        paren_depth = 0

        while True:
            try:
                loc = self.preParse(instring, loc)
            except ParseException:
                pass

            if state is _ParserState.EXPECT_OPERAND:
                # 1. Try prefix operators
                matched_prefix = False
                for op_expr, prec, assoc, pa_list in self.prefix_ops:
                    try:
                        next_loc, toks = op_expr._parse(instring, loc, do_actions=do_actions)
                        op_tok = toks.as_list()[0] if len(toks) == 1 else toks.as_list()
                        operator_stack.append({
                            "type": _OpType.PREFIX,
                            "op": op_tok,
                            "prec": prec,
                            "assoc": assoc,
                            "pa": pa_list if do_actions else [],
                            "loc": loc,
                        })
                        loc = next_loc
                        matched_prefix = True
                        break
                    except ParseException:
                        pass
                if matched_prefix:
                    continue

                # 2. Try base_expr
                try:
                    next_loc, base_toks = self.base_expr._parse(instring, loc, do_actions=do_actions)
                    if not base_toks and not base_toks._tokdict:
                        loc = next_loc
                        continue
                    operand = base_toks[0] if len(base_toks) == 1 else base_toks
                    operand_stack.append(operand)
                    loc = next_loc
                    state = _ParserState.EXPECT_OPERATOR
                    continue
                except ParseException:
                    pass

                # 3. Try lpar
                try:
                    next_loc, lpar_toks = self.lpar._parse(instring, loc, do_actions=do_actions)
                    operator_stack.append({
                        "type": _OpType.LPAR,
                        "paren_toks": lpar_toks.as_list(),
                        "loc": loc,
                    })
                    paren_depth += 1
                    loc = next_loc
                    continue
                except ParseException:
                    raise ParseException(instring, loc, f"Expected {self.base_expr}")

            elif state is _ParserState.EXPECT_OPERATOR:
                # 1. Try postfix operators
                matched_postfix = False
                for op_expr, prec, assoc, pa_list in self.postfix_ops:
                    try:
                        next_loc, toks = op_expr._parse(instring, loc, do_actions=do_actions)
                        op_tok = toks
                        if (
                            operator_stack
                            and operator_stack[-1]["type"] is _OpType.POSTFIX
                            and operator_stack[-1]["prec"] == prec
                            and operator_stack[-1]["assoc"] is OpAssoc.LEFT
                        ):
                            operator_stack[-1]["ops"].append(op_tok)
                        else:
                            reduce_operators(prec)
                            operator_stack.append({
                                "type": _OpType.POSTFIX,
                                "ops": [op_tok],
                                "prec": prec,
                                "assoc": assoc,
                                "pa": pa_list if do_actions else [],
                                "loc": loc,
                            })
                        loc = next_loc
                        matched_postfix = True
                        break
                    except ParseException:
                        pass
                if matched_postfix:
                    continue

                # 2. Try rpar if inside parens
                if paren_depth > 0:
                    try:
                        next_loc, rpar_toks = self.rpar._parse(instring, loc, do_actions=do_actions)
                        reduce_operators(-1)  # reduce all operators inside this paren
                        if operator_stack and operator_stack[-1]["type"] is _OpType.LPAR:
                            lpar_info = operator_stack.pop()
                            if self.keep_parens:
                                top_val = operand_stack.pop()
                                operand_stack.append(ParseResults([*lpar_info["paren_toks"], top_val, *rpar_toks.as_list()]))
                        paren_depth -= 1
                        loc = next_loc
                        continue
                    except ParseException:
                        pass

                # 3. Try matching op2 for an open ternary operator (STAGE1)
                matched_op2 = False
                for i in range(len(operator_stack) - 1, -1, -1):
                    item = operator_stack[i]
                    if item["type"] is _OpType.LPAR:
                        break
                    if item["type"] in (_OpType.TERNARY_LEFT_STAGE1, _OpType.TERNARY_RIGHT_STAGE1):
                        try:
                            next_loc, toks = item["op2_expr"]._parse(instring, loc, do_actions=do_actions)
                            op2_tok = toks.as_list()[0] if len(toks) == 1 else toks.as_list()
                            while operator_stack and operator_stack[-1] is not item:
                                op_info = operator_stack.pop()
                                self._apply_operator(op_info, operand_stack, instring)
                            if item["type"] is _OpType.TERNARY_LEFT_STAGE1:
                                item["type"] = _OpType.TERNARY_LEFT_STAGE2
                                item["ops"][-1].append(op2_tok)
                            else:
                                item["type"] = _OpType.TERNARY_RIGHT_STAGE2
                                item["op2"] = op2_tok
                            loc = next_loc
                            matched_op2 = True
                            state = _ParserState.EXPECT_OPERAND
                            break
                        except ParseException:
                            pass
                if matched_op2:
                    continue

                def _check_operand_follows(check_loc):
                    for p_op, _, _, _ in self.prefix_ops:
                        try:
                            p_op._parse(instring, check_loc, do_actions=False)
                            return True
                        except ParseException:
                            pass
                    try:
                        self.base_expr._parse(instring, check_loc, do_actions=False)
                        return True
                    except ParseException:
                        pass
                    try:
                        self.lpar._parse(instring, check_loc, do_actions=False)
                        return True
                    except ParseException:
                        pass
                    return False

                # 4. Try infix binary and ternary operators
                matched_infix = False
                for op_expr, prec, assoc, pa_list, arity, op2_expr in self.infix_ops:
                    if arity == 2:
                        if op_expr is None:
                            if not _check_operand_follows(loc):
                                continue
                            next_loc = loc
                            op_tok = None
                        else:
                            try:
                                next_loc, toks = op_expr._parse(instring, loc, do_actions=do_actions)
                            except ParseException:
                                continue
                            if next_loc == loc:
                                if not _check_operand_follows(loc):
                                    continue
                                op_tok = (toks.as_list()[0] if len(toks) == 1 else toks.as_list()) if toks else None
                            else:
                                op_tok = toks.as_list()[0] if len(toks) == 1 else toks.as_list()

                        if assoc is OpAssoc.LEFT:
                            reduce_operators(prec)
                            if (
                                operator_stack
                                and operator_stack[-1]["type"] is _OpType.INFIX2
                                and operator_stack[-1]["prec"] == prec
                            ):
                                operator_stack[-1]["ops"].append(op_tok)
                            else:
                                operator_stack.append({
                                    "type": _OpType.INFIX2,
                                    "ops": [op_tok],
                                    "prec": prec,
                                    "assoc": assoc,
                                    "pa": pa_list if do_actions else [],
                                    "loc": loc,
                                })
                        else:
                            reduce_operators(prec)
                            operator_stack.append({
                                "type": _OpType.INFIX2_RIGHT,
                                "op": op_tok,
                                "prec": prec,
                                "assoc": assoc,
                                "pa": pa_list if do_actions else [],
                                "loc": loc,
                            })
                        loc = next_loc
                        matched_infix = True
                        state = _ParserState.EXPECT_OPERAND
                        break
                    elif arity == 3:
                        try:
                            next_loc, toks = op_expr._parse(instring, loc, do_actions=do_actions)
                            if next_loc == loc and not _check_operand_follows(loc):
                                continue
                            op_tok = toks.as_list()[0] if len(toks) == 1 else toks.as_list()
                            if assoc is OpAssoc.LEFT:
                                if (
                                    operator_stack
                                    and operator_stack[-1]["type"] is _OpType.TERNARY_LEFT_STAGE2
                                    and operator_stack[-1]["prec"] == prec
                                ):
                                    operator_stack[-1]["type"] = _OpType.TERNARY_LEFT_STAGE1
                                    operator_stack[-1]["ops"].append([op_tok])
                                else:
                                    reduce_operators(prec)
                                    operator_stack.append({
                                        "type": _OpType.TERNARY_LEFT_STAGE1,
                                        "ops": [[op_tok]],
                                        "op2_expr": op2_expr,
                                        "prec": prec,
                                        "assoc": assoc,
                                        "pa": pa_list if do_actions else [],
                                        "loc": loc,
                                    })
                            else:
                                reduce_operators(prec)
                                operator_stack.append({
                                    "type": _OpType.TERNARY_RIGHT_STAGE1,
                                    "op1": op_tok,
                                    "op2_expr": op2_expr,
                                    "prec": prec,
                                    "assoc": assoc,
                                    "pa": pa_list if do_actions else [],
                                    "loc": loc,
                                })
                            loc = next_loc
                            matched_infix = True
                            state = _ParserState.EXPECT_OPERAND
                            break
                        except ParseException:
                            pass
                if matched_infix:
                    continue

                # No more operators can be consumed at this level
                break

        # Reduce remaining operators
        reduce_operators(-1)

        if paren_depth != 0 or len(operand_stack) != 1 or operator_stack:
            raise ParseException(instring, loc, "Unbalanced parentheses or expression syntax error")

        final_result = operand_stack.pop()
        if last_pa_applied[0] and isinstance(final_result, ParseResults):
            return loc, final_result
        else:
            return loc, ParseResults([final_result])


def infix_notation(base_expr, op_list, lpar="(", rpar=")"):
    """Helper method for constructing grammars of expressions made up of
    operators working in a precedence hierarchy.  Operators may be unary
    or binary, left- or right-associative.  Parse actions can also be
    attached to operator expressions. The generated parser will also
    recognize the use of parentheses to override operator precedences
    (see example below).

    Note: if you define a deep operator list, you may see performance
    issues when using infix_notation. See
    :class:`ParserElement.enable_packrat` for a mechanism to potentially
    improve your parser performance.

    Parameters:

    :param base_expr: expression representing the most basic operand to
       be used in the expression
    :param op_list: list of tuples, one for each operator precedence level
       in the expression grammar; each tuple is of the form ``(op_expr,
       num_operands, right_left_assoc, (optional)parse_action)``, where:

       - ``op_expr`` is the pyparsing expression for the operator; may also
         be a string, which will be converted to a Literal; if ``num_operands``
         is 3, ``op_expr`` is a tuple of two expressions, for the two
         operators separating the 3 terms
       - ``num_operands`` is the number of terms for this operator (must be 1,
         2, or 3)
       - ``right_left_assoc`` is the indicator whether the operator is right
         or left associative, using the pyparsing-defined constants
         ``OpAssoc.RIGHT`` and ``OpAssoc.LEFT``.
       - ``parse_action`` is the parse action to be associated with
         expressions matching this operator expression (the parse action
         tuple member may be omitted); if the parse action is passed
         a tuple or list of functions, this is equivalent to calling
         ``set_parse_action(*fn)``
         (:class:`ParserElement.set_parse_action`)

    :param lpar: expression for matching left-parentheses; if passed as a
       str, then will be parsed as ``Suppress(lpar)``. If lpar is passed as
       an expression (such as ``Literal('(')``), then it will be kept in
       the parsed results, and grouped with them. (default= ``Suppress('(')``)
    :param rpar: expression for matching right-parentheses; if passed as a
       str, then will be parsed as ``Suppress(rpar)``. If rpar is passed as
       an expression (such as ``Literal(')')``), then it will be kept in
       the parsed results, and grouped with them. (default= ``Suppress(')')``)

    Example:

    .. testcode::

       # simple example of four-function arithmetic with ints and
       # variable names
       integer = pyparsing_common.signed_integer
       varname = pyparsing_common.identifier

       arith_expr = infix_notation(integer | varname,
           [
           ('-', 1, OpAssoc.RIGHT),
           (one_of('* /'), 2, OpAssoc.LEFT),
           (one_of('+ -'), 2, OpAssoc.LEFT),
           ])

       arith_expr.run_tests('''
           5+3*6
           (5+3)*6
           (5+x)*y
           -2--11
           ''', full_dump=False)

    prints:

    .. testoutput::
       :options: +NORMALIZE_WHITESPACE


       5+3*6
       [[5, '+', [3, '*', 6]]]

       (5+3)*6
       [[[5, '+', 3], '*', 6]]

       (5+x)*y
       [[[5, '+', 'x'], '*', 'y']]

       -2--11
       [[['-', 2], '-', ['-', 11]]]
    """
    return _InfixNotation(base_expr, op_list, lpar=lpar, rpar=rpar)


def indentedBlock(blockStatementExpr, indentStack, indent=True, backup_stacks=[]):
    """
    .. deprecated:: 3.0.0
       Use the :class:`IndentedBlock` class instead. Note that `IndentedBlock`
       has a difference method signature.

    Helper method for defining space-delimited indentation blocks,
    such as those used to define block statements in Python source code.

    :param blockStatementExpr: expression defining syntax of statement that
      is repeated within the indented block

    :param indentStack: list created by caller to manage indentation stack
      (multiple ``statementWithIndentedBlock`` expressions within a single
      grammar should share a common ``indentStack``)

    :param indent: boolean indicating whether block must be indented beyond
      the current level; set to ``False`` for block of left-most statements

    A valid block must contain at least one ``blockStatement``.

    (Note that indentedBlock uses internal parse actions which make it
    incompatible with packrat parsing.)

    Example:

    .. testcode::

       data = '''
       def A(z):
         A1
         B = 100
         G = A2
         A2
         A3
       B
       def BB(a,b,c):
         BB1
         def BBA():
           bba1
           bba2
           bba3
       C
       D
       def spam(x,y):
            def eggs(z):
                pass
       '''

       indentStack = [1]
       stmt = Forward()

       identifier = Word(alphas, alphanums)
       funcDecl = ("def" + identifier + Group("(" + Opt(delimitedList(identifier)) + ")") + ":")
       func_body = indentedBlock(stmt, indentStack)
       funcDef = Group(funcDecl + func_body)

       rvalue = Forward()
       funcCall = Group(identifier + "(" + Opt(delimitedList(rvalue)) + ")")
       rvalue << (funcCall | identifier | Word(nums))
       assignment = Group(identifier + "=" + rvalue)
       stmt << (funcDef | assignment | identifier)

       module_body = stmt[1, ...]

       parseTree = module_body.parseString(data)
       parseTree.pprint()

    prints:

    .. testoutput::

       [['def',
         'A',
         ['(', 'z', ')'],
         ':',
         [['A1'], [['B', '=', '100']], [['G', '=', 'A2']], ['A2'], ['A3']]],
        'B',
        ['def',
         'BB',
         ['(', 'a', 'b', 'c', ')'],
         ':',
         [['BB1'], [['def', 'BBA', ['(', ')'], ':', [['bba1'], ['bba2'], ['bba3']]]]]],
        'C',
        'D',
        ['def',
         'spam',
         ['(', 'x', 'y', ')'],
         ':',
         [[['def', 'eggs', ['(', 'z', ')'], ':', [['pass']]]]]]]
    """
    warnings.warn(
        f"{'indentedBlock'!r} deprecated - use {'IndentedBlock'!r}",
        PyparsingDeprecationWarning,
        stacklevel=2,
    )

    backup_stacks.append(indentStack[:])

    def reset_stack():
        indentStack[:] = backup_stacks[-1]

    def checkPeerIndent(s, l, t):
        if l >= len(s):
            return
        curCol = col(l, s)
        if curCol != indentStack[-1]:
            if curCol > indentStack[-1]:
                raise ParseException(s, l, "illegal nesting")
            raise ParseException(s, l, "not a peer entry")

    def checkSubIndent(s, l, t):
        curCol = col(l, s)
        if curCol > indentStack[-1]:
            indentStack.append(curCol)
        else:
            raise ParseException(s, l, "not a subentry")

    def checkUnindent(s, l, t):
        if l >= len(s):
            return
        curCol = col(l, s)
        if not (indentStack and curCol in indentStack):
            raise ParseException(s, l, "not an unindent")
        if curCol < indentStack[-1]:
            indentStack.pop()

    NL = OneOrMore(LineEnd().set_whitespace_chars("\t ").suppress())
    INDENT = (Empty() + Empty().set_parse_action(checkSubIndent)).set_name("INDENT")
    PEER = Empty().set_parse_action(checkPeerIndent).set_name("")
    UNDENT = Empty().set_parse_action(checkUnindent).set_name("UNINDENT")
    if indent:
        smExpr = Group(
            Opt(NL)
            + INDENT
            + OneOrMore(PEER + Group(blockStatementExpr) + Opt(NL))
            + UNDENT
        )
    else:
        smExpr = Group(
            Opt(NL)
            + OneOrMore(PEER + Group(blockStatementExpr) + Opt(NL))
            + Opt(UNDENT)
        )

    # add a parse action to remove backup_stack from list of backups
    smExpr.add_parse_action(
        lambda: backup_stacks.pop(-1) and None if backup_stacks else None
    )
    smExpr.set_fail_action(lambda a, b, c, d: reset_stack())
    blockStatementExpr.ignore(_bslash + LineEnd())
    return smExpr.set_name("indented block")


# it's easy to get these comment structures wrong - they're very common,
# so may as well make them available
c_style_comment = Regex(r"/\*(?:[^*]|\*(?!/))*\*\/").set_name("C style comment")
"Comment of the form ``/* ... */``"

html_comment = Regex(r"<!--[\s\S]*?-->").set_name("HTML comment")
"Comment of the form ``<!-- ... -->``"

rest_of_line = Regex(r".*").leave_whitespace().set_name("rest of line")
dbl_slash_comment = Regex(r"//(?:\\\n|[^\n])*").set_name("// comment")
"Comment of the form ``// ... (to end of line)``"

cpp_style_comment = Regex(
    r"(?:/\*(?:[^*]|\*(?!/))*\*\/)|(?://(?:\\\n|[^\n])*)"
).set_name("C++ style comment")
"Comment of either form :class:`c_style_comment` or :class:`dbl_slash_comment`"

java_style_comment = cpp_style_comment
"Same as :class:`cpp_style_comment`"

python_style_comment = Regex(r"#.*").set_name("Python style comment")
"Comment of the form ``# ... (to end of line)``"


# build list of built-in expressions, for future reference if a global default value
# gets updated
_builtin_exprs: list[ParserElement] = [
    v for v in vars().values() if isinstance(v, ParserElement)
]


# compatibility function, superseded by DelimitedList class
def delimited_list(
    expr: Union[str, ParserElement],
    delim: Union[str, ParserElement] = ",",
    combine: bool = False,
    min: typing.Optional[int] = None,
    max: typing.Optional[int] = None,
    *,
    allow_trailing_delim: bool = False,
) -> ParserElement:
    """
    .. deprecated:: 3.1.0
       Use the :class:`DelimitedList` class instead.
    """
    return DelimitedList(
        expr, delim, combine, min, max, allow_trailing_delim=allow_trailing_delim
    )


# Compatibility synonyms
# fmt: off
opAssoc = OpAssoc
anyOpenTag = any_open_tag
anyCloseTag = any_close_tag
commonHTMLEntity = common_html_entity
cStyleComment = c_style_comment
htmlComment = html_comment
restOfLine = rest_of_line
dblSlashComment = dbl_slash_comment
cppStyleComment = cpp_style_comment
javaStyleComment = java_style_comment
pythonStyleComment = python_style_comment
delimitedList = replaced_by_pep8("delimitedList", DelimitedList)
delimited_list = replaced_by_pep8("delimited_list", DelimitedList)
countedArray = replaced_by_pep8("countedArray", counted_array)
matchPreviousLiteral = replaced_by_pep8("matchPreviousLiteral", match_previous_literal)
matchPreviousExpr = replaced_by_pep8("matchPreviousExpr", match_previous_expr)
oneOf = replaced_by_pep8("oneOf", one_of)
dictOf = replaced_by_pep8("dictOf", dict_of)
originalTextFor = replaced_by_pep8("originalTextFor", original_text_for)
nestedExpr = replaced_by_pep8("nestedExpr", nested_expr)
makeHTMLTags = replaced_by_pep8("makeHTMLTags", make_html_tags)
makeXMLTags = replaced_by_pep8("makeXMLTags", make_xml_tags)
replaceHTMLEntity = replaced_by_pep8("replaceHTMLEntity", replace_html_entity)
infixNotation = replaced_by_pep8("infixNotation", infix_notation)
# fmt: on
<|endoftext|># exceptions.py
from __future__ import annotations

import copy
import re
import sys
import typing
import warnings
from functools import cached_property

from .warnings import PyparsingDeprecationWarning
from .unicode import pyparsing_unicode as ppu
from .util import (
    _collapse_string_to_ranges,
    col,
    deprecate_argument,
    line,
    lineno,
    replaced_by_pep8,
)


class _ExceptionWordUnicodeSet(
    ppu.Latin1, ppu.LatinA, ppu.LatinB, ppu.Greek, ppu.Cyrillic
):
    pass


_extract_alphanums = _collapse_string_to_ranges(_ExceptionWordUnicodeSet.alphanums)
_exception_word_extractor = re.compile(fr"([{_extract_alphanums}]{{1,16}})|.")


class ParseBaseException(Exception):
    """base exception class for all parsing runtime exceptions"""

    loc: int
    msg: str
    pstr: str
    parser_element: typing.Any  # "ParserElement"
    args: tuple[str, int, typing.Optional[str]]

    __slots__ = (
        "loc",
        "msg",
        "pstr",
        "parser_element",
        "args",
    )

    # Performance tuning: we construct a *lot* of these, so keep this
    # constructor as small and fast as possible
    def __init__(
        self,
        pstr: str,
        loc: int = 0,
        msg: typing.Optional[str] = None,
        elem=None,
    ) -> None:
        if msg is None:
            msg, pstr = pstr, ""

        self.loc = loc
        self.msg = msg
        self.pstr = pstr
        self.parser_element = elem
        self.args = (pstr, loc, msg)

    @staticmethod
    def explain_exception(exc: Exception, depth: int = 16) -> str:
        """
        Method to take an exception and translate the Python internal traceback into a list
        of the pyparsing expressions that caused the exception to be raised.

        Parameters:

        - exc - exception raised during parsing (need not be a ParseException, in support
          of Python exceptions that might be raised in a parse action)
        - depth (default=16) - number of levels back in the stack trace to list expression
          and function names; if None, the full stack trace names will be listed; if 0, only
          the failing input line, marker, and exception string will be shown

        Returns a multi-line string listing the ParserElements and/or function names in the
        exception's stack trace.
        """
        import inspect
        from .core import ParserElement

        if depth is None:
            depth = sys.getrecursionlimit()
        ret: list[str] = []
        if isinstance(exc, ParseBaseException):
            ret.append(exc.line)
            ret.append(f"{'^':>{exc.column}}")
        ret.append(f"{type(exc).__name__}: {exc}")

        if depth <= 0 or exc.__traceback__ is None:
            return "\n".join(ret)

        callers = inspect.getinnerframes(exc.__traceback__, context=depth)
        seen: set[int] = set()
        for ff in callers[-depth:]:
            frm = ff[0]

            f_self = frm.f_locals.get("self", None)
            if isinstance(f_self, ParserElement):
                if not frm.f_code.co_name.startswith(("parseImpl", "_parseNoCache")):
                    continue
                if id(f_self) in seen:
                    continue
                seen.add(id(f_self))

                self_type = type(f_self)
                ret.append(f"{self_type.__module__}.{self_type.__name__} - {f_self}")

            elif f_self is not None:
                self_type = type(f_self)
                ret.append(f"{self_type.__module__}.{self_type.__name__}")

            else:
                code = frm.f_code
                if code.co_name in ("wrapper", "<module>"):
                    continue

                ret.append(code.co_name)

            depth -= 1
            if not depth:
                break

        return "\n".join(ret)

    @classmethod
    def _from_exception(cls, pe) -> ParseBaseException:
        """
        internal factory method to simplify creating one type of ParseException
        from another - avoids having __init__ signature conflicts among subclasses
        """
        return cls(pe.pstr, pe.loc, pe.msg, pe.parser_element)

    @cached_property
    def line(self) -> str:
        """
        Return the line of text where the exception occurred.
        """
        return line(self.loc, self.pstr)

    @cached_property
    def lineno(self) -> int:
        """
        Return the 1-based line number of text where the exception occurred.
        """
        return lineno(self.loc, self.pstr)

    @cached_property
    def col(self) -> int:
        """
        Return the 1-based column on the line of text where the exception occurred.
        """
        return col(self.loc, self.pstr)

    @cached_property
    def column(self) -> int:
        """
        Return the 1-based column on the line of text where the exception occurred.
        """
        return col(self.loc, self.pstr)

    @cached_property
    def found(self) -> str:
        if not self.pstr:
            return ""

        if self.loc >= len(self.pstr):
            return "end of text"

        # pull out next word at error location
        found_match = _exception_word_extractor.match(self.pstr, self.loc)
        if found_match is not None:
            found_text = found_match[0]
        else:
            found_text = self.pstr[self.loc : self.loc + 1]

        return repr(found_text).replace(r"\\", "\\")

    # pre-PEP8 compatibility
    @property
    def parserElement(self):
        warnings.warn(
            "parserElement is deprecated, use parser_element",
            PyparsingDeprecationWarning,
            stacklevel=2,
        )
        return self.parser_element

    @parserElement.setter
    def parserElement(self, elem):
        warnings.warn(
            "parserElement is deprecated, use parser_element",
            PyparsingDeprecationWarning,
            stacklevel=2,
        )
        self.parser_element = elem

    def copy(self):
        return copy.copy(self)

    def formatted_message(self) -> str:
        """
        Output the formatted exception message.
        Can be overridden to customize the message formatting or contents.

        .. versionadded:: 3.2.0
        """
        found_phrase = f", found {self.found}" if self.found else ""
        return f"{self.msg}{found_phrase}  (at char {self.loc}), (line:{self.lineno}, col:{self.column})"

    def __str__(self) -> str:
        """
        .. versionchanged:: 3.2.0
           Now uses :meth:`formatted_message` to format message.
        """
        try:
            return self.formatted_message()
        except Exception as ex:
            return (
                f"{type(self).__name__}: {self.msg}"
                f" ({type(ex).__name__}: {ex} while formatting message)"
            )

    def __repr__(self):
        return str(self)

    def mark_input_line(
        self, marker_string: typing.Optional[str] = None, **kwargs
    ) -> str:
        """
        Extracts the exception line from the input string, and marks
        the location of the exception with a special symbol.
        """
        markerString: str = deprecate_argument(kwargs, "markerString", ">!<")

        markerString = marker_string if marker_string is not None else markerString
        line_str = self.line
        line_column = self.column - 1
        if markerString:
            line_str = f"{line_str[:line_column]}{markerString}{line_str[line_column:]}"
        return line_str.strip()

    def explain(self, depth: int = 16) -> str:
        """
        Method to translate the Python internal traceback into a list
        of the pyparsing expressions that caused the exception to be raised.

        Parameters:

        - depth (default=16) - number of levels back in the stack trace to list expression
          and function names; if None, the full stack trace names will be listed; if 0, only
          the failing input line, marker, and exception string will be shown

        Returns a multi-line string listing the ParserElements and/or function names in the
        exception's stack trace.

        Example:

        .. testcode::

            # an expression to parse 3 integers
            expr = pp.Word(pp.nums) * 3
            try:
                # a failing parse - the third integer is prefixed with "A"
                expr.parse_string("123 456 A789")
            except pp.ParseException as pe:
                print(pe.explain(depth=0))

        prints:

        .. testoutput::

            123 456 A789
                    ^
            ParseException: Expected W:(0-9), found 'A789'  (at char 8), (line:1, col:9)

        Note: the diagnostic output will include string representations of the expressions
        that failed to parse. These representations will be more helpful if you use `set_name` to
        give identifiable names to your expressions. Otherwise they will use the default string
        forms, which may be cryptic to read.

        Note: pyparsing's default truncation of exception tracebacks may also truncate the
        stack of expressions that are displayed in the ``explain`` output. To get the full listing
        of parser expressions, you may have to set ``ParserElement.verbose_stacktrace = True``
        """
        return self.explain_exception(self, depth)

    # Compatibility synonyms
    # fmt: off
    markInputline = replaced_by_pep8("markInputline", mark_input_line)
    # fmt: on


class ParseException(ParseBaseException):
    """
    Exception thrown when a parse expression doesn't match the input string

    Example:

    .. testcode::

        integer = Word(nums).set_name("integer")
        try:
            integer.parse_string("ABC")
        except ParseException as pe:
            print(pe, f"column: {pe.column}")

    prints:

    .. testoutput::

       Expected integer, found 'ABC'  (at char 0), (line:1, col:1) column: 1

    """


class ParseFatalException(ParseBaseException):
    """
    User-throwable exception thrown when inconsistent parse content
    is found; stops all parsing immediately
    """


class ParseSyntaxException(ParseFatalException):
    """
    Just like :class:`ParseFatalException`, but thrown internally
    when an :class:`ErrorStop<And._ErrorStop>` ('-' operator) indicates
    that parsing is to stop immediately because an unbacktrackable
    syntax error has been found.
    """


class RecursiveGrammarException(Exception):
    """
    .. deprecated:: 3.0.0
       Only used by the deprecated :meth:`ParserElement.validate`.

    Exception thrown by :class:`ParserElement.validate` if the
    grammar could be left-recursive; parser may need to enable
    left recursion using :class:`ParserElement.enable_left_recursion<ParserElement.enable_left_recursion>`
    """

    def __init__(self, parseElementList) -> None:
        self.parseElementTrace = parseElementList

    def __str__(self) -> str:
        return f"RecursiveGrammarException: {self.parseElementTrace}"
<|endoftext|># mypy: ignore-errors
from __future__ import annotations

import itertools
import railroad
import pyparsing
import dataclasses
import typing
from typing import (
    Generic,
    TypeVar,
    Callable,
    Iterable,
)
from jinja2 import Template
from io import StringIO
import inspect
import re


jinja2_template_source = """\
{% if not embed %}
<!DOCTYPE html>
<html>
<head>
{% endif %}
    {% if not head %}
        <style>
            .railroad-heading {
                font-family: monospace;
            }
        </style>
    {% else %}
        {{ head | safe }}
    {% endif %}
{% if not embed %}
</head>
<body>
{% endif %}
<meta charset="UTF-8"/>
{{ body | safe }}
{% for diagram in diagrams %}
    <div class="railroad-group">
        <h1 class="railroad-heading" id="{{ diagram.bookmark }}">{{ diagram.title }}</h1>
        <div class="railroad-description">{{ diagram.text }}</div>
        <div class="railroad-svg">
            {{ diagram.svg }}
        </div>
    </div>
{% endfor %}
{% if not embed %}
</body>
</html>
{% endif %}
"""

template = Template(jinja2_template_source)


_bookmark_lookup = {}
_bookmark_ids = itertools.count(start=1)

def _make_bookmark(s: str) -> str:
    """
    Converts a string into a valid HTML bookmark (ID or anchor name).
    """
    if s in _bookmark_lookup:
        return _bookmark_lookup[s]

    # Replace invalid characters with hyphens and ensure only valid characters
    bookmark = re.sub(r'[^a-zA-Z0-9-]+', '-', s)

    # Ensure it starts with a letter by adding 'z' if necessary
    if not bookmark[:1].isalpha():
        bookmark = f"z{bookmark}"

    # Convert to lowercase and strip hyphens
    bookmark = bookmark.lower().strip('-')

    _bookmark_lookup[s] = bookmark = f"{bookmark}-{next(_bookmark_ids):04d}"

    return bookmark


def _collapse_verbose_regex(regex_str: str) -> str:
    if "\n" not in regex_str:
        return regex_str
    collapsed = pyparsing.Regex(r"#.*$").suppress().transform_string(regex_str)
    collapsed = re.sub(r"\s*\n\s*", "", collapsed)
    return collapsed


@dataclasses.dataclass
class NamedDiagram:
    """
    A simple structure for associating a name with a railroad diagram
    """

    name: str
    index: int
    diagram: railroad.DiagramItem = None

    @property
    def bookmark(self):
        bookmark = _make_bookmark(self.name)
        return bookmark


T = TypeVar("T")


class EachItem(railroad.Group):
    """
    Custom railroad item to compose a:

    - :class:`railroad.Group` containing a

      - :class:`railroad.OneOrMore` containing a

        - :class:`railroad.Choice` of the elements in the
          :class:`railroad.Each`

    with the group label indicating that all must be matched
    """

    all_label = "[ALL]"

    def __init__(self, *items) -> None:
        choice_item = railroad.Choice(len(items) - 1, *items)
        one_or_more_item = railroad.OneOrMore(item=choice_item)
        super().__init__(one_or_more_item, label=self.all_label)


class AnnotatedItem(railroad.Group):
    """
    Simple subclass of Group that creates an annotation label
    """

    def __init__(self, label: str, item) -> None:
        super().__init__(item=item, label=f"[{label}]" if label else "")


class EditablePartial(Generic[T]):
    """
    Acts like a functools.partial, but can be edited. In other words, it represents a type that hasn't yet been
    constructed.
    """

    # We need this here because the railroad constructors actually transform the data, so can't be called until the
    # entire tree is assembled

    def __init__(self, func: Callable[..., T], args: list, kwargs: dict) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_call(cls, func: Callable[..., T], *args, **kwargs) -> EditablePartial[T]:
        """
        If you call this function in the same way that you would call the constructor,
        it will store the arguments as you expect. For example
        ``EditablePartial.from_call(Fraction, 1, 3)() == Fraction(1, 3)``
        """
        return EditablePartial(func=func, args=list(args), kwargs=kwargs)

    @property
    def name(self):
        return self.kwargs["name"]

    def __call__(self) -> T:
        """
        Evaluate the partial and return the result
        """
        args = self.args.copy()
        kwargs = self.kwargs.copy()

        # This is a helpful hack to allow you to specify varargs parameters (e.g. *args) as keyword args (e.g.
        # args=['list', 'of', 'things'])
        arg_spec = inspect.getfullargspec(self.func)
        if arg_spec.varargs in self.kwargs:
            args += kwargs.pop(arg_spec.varargs)

        return self.func(*args, **kwargs)


def railroad_to_html(diagrams: list[NamedDiagram], embed=False, **kwargs) -> str:
    """
    Given a list of :class:`NamedDiagram`, produce a single HTML string
    that visualises those diagrams.

    :params kwargs: kwargs to be passed in to the template
    """
    data = []
    for diagram in diagrams:
        if diagram.diagram is None:
            continue
        io = StringIO()
        try:
            css = kwargs.get("css")
            diagram.diagram.writeStandalone(io.write, css=css)
        except AttributeError:
            diagram.diagram.writeSvg(io.write)
        title = diagram.name
        if diagram.index == 0:
            title += " (root)"
        data.append(
            {
                "title": title, "text": "", "svg": io.getvalue(), "bookmark": diagram.bookmark
            }
        )

    return template.render(diagrams=data, embed=embed, **kwargs)


def resolve_partial(partial: EditablePartial[T]) -> T:
    """
    Recursively resolves a collection of Partials into whatever type they are
    """
    if isinstance(partial, EditablePartial):
        partial.args = resolve_partial(partial.args)
        partial.kwargs = resolve_partial(partial.kwargs)
        return partial()
    elif isinstance(partial, list):
        return [resolve_partial(x) for x in partial]
    elif isinstance(partial, dict):
        return {key: resolve_partial(x) for key, x in partial.items()}
    else:
        return partial


def to_railroad(
    element: pyparsing.ParserElement,
    diagram_kwargs: typing.Optional[dict] = None,
    vertical: int = 3,
    show_results_names: bool = False,
    show_groups: bool = False,
    show_hidden: bool = False,
) -> list[NamedDiagram]:
    """
    Convert a pyparsing element tree into a list of diagrams. This is the recommended entrypoint to diagram
    creation if you want to access the Railroad tree before it is converted to HTML

    :param element: base element of the parser being diagrammed

    :param diagram_kwargs: kwargs to pass to the :meth:`Diagram` constructor

    :param vertical: (optional) int - limit at which number of alternatives
        should be shown vertically instead of horizontally

    :param show_results_names: bool to indicate whether results name
        annotations should be included in the diagram

    :param show_groups: bool to indicate whether groups should be highlighted
        with an unlabeled surrounding box

    :param show_hidden: bool to indicate whether internal elements that are
        typically hidden should be shown
    """
    # Convert the whole tree underneath the root
    lookup = ConverterState(diagram_kwargs=diagram_kwargs or {})
    _to_diagram_element(
        element,
        lookup=lookup,
        parent=None,
        vertical=vertical,
        show_results_names=show_results_names,
        show_groups=show_groups,
        show_hidden=show_hidden,
    )

    root_id = id(element)
    # Convert the root if it hasn't been already
    if root_id in lookup:
        if not element.customName:
            lookup[root_id].name = ""
        lookup[root_id].mark_for_extraction(root_id, lookup, force=True)

    # Now that we're finished, we can convert from intermediate structures into Railroad elements
    diags = list(lookup.diagrams.values())
    if len(diags) > 1:
        # collapse out duplicate diags with the same name
        seen = set()
        deduped_diags = []
        for d in diags:
            # don't extract SkipTo elements, they are uninformative as subdiagrams
            if d.name == "...":
                continue
            if d.name is not None and d.name not in seen:
                seen.add(d.name)
                deduped_diags.append(d)
        resolved = [resolve_partial(partial) for partial in deduped_diags]
    else:
        # special case - if just one diagram, always display it, even if
        # it has no name
        resolved = [resolve_partial(partial) for partial in diags]
    return sorted(resolved, key=lambda diag: diag.index)


def _should_vertical(
    specification: int, exprs: Iterable[pyparsing.ParserElement]
) -> bool:
    """
    Returns true if we should return a vertical list of elements
    """
    if specification is None:
        return False
    else:
        return len(_visible_exprs(exprs)) >= specification


@dataclasses.dataclass
class ElementState:
    """
    State recorded for an individual pyparsing Element
    """

    #: The pyparsing element that this represents
    element: pyparsing.ParserElement
    #: The output Railroad element in an unconverted state
    converted: EditablePartial
    #: The parent Railroad element, which we store so that we can extract this if it's duplicated
    parent: EditablePartial
    #: The order in which we found this element, used for sorting diagrams if this is extracted into a diagram
    number: int
    #: The name of the element
    name: str = None
    #: The index of this inside its parent
    parent_index: typing.Optional[int] = None
    #: If true, we should extract this out into a subdiagram
    extract: bool = False
    #: If true, all of this element's children have been filled out
    complete: bool = False

    def mark_for_extraction(
        self, el_id: int, state: ConverterState, name: str = None, force: bool = False
    ):
        """
        Called when this instance has been seen twice, and thus should eventually be extracted into a sub-diagram
        :param el_id: id of the element
        :param state: element/diagram state tracker
        :param name: name to use for this element's text
        :param force: If true, force extraction now, regardless of the state of this. Only useful for extracting the
        root element when we know we're finished
        """
        self.extract = True

        # Set the name
        if not self.name:
            if name:
                # Allow forcing a custom name
                self.name = name
            elif self.element.customName:
                self.name = self.element.customName
            else:
                self.name = ""

        # Just because this is marked for extraction doesn't mean we can do it yet. We may have to wait for children
        # to be added
        # Also, if this is just a string literal etc, don't bother extracting it
        if force or (self.complete and _worth_extracting(self.element)):
            state.extract_into_diagram(el_id)


class ConverterState:
    """
    Stores some state that persists between recursions into the element tree
    """
    index_generator = itertools.count(start=1)

    def __init__(self, diagram_kwargs: typing.Optional[dict] = None) -> None:
        #: A dictionary mapping ParserElements to state relating to them
        self._element_diagram_states: dict[int, ElementState] = {}
        #: A dictionary mapping ParserElement IDs to subdiagrams generated from them
        self.diagrams: dict[int, EditablePartial[NamedDiagram]] = {}
        #: The index of the next element. This is used for sorting
        self.index: int = 0
        #: Shared kwargs that are used to customize the construction of diagrams
        self.diagram_kwargs: dict = diagram_kwargs or {}
        self.extracted_diagram_names: set[str] = set()

    def __setitem__(self, key: int, value: ElementState):
        self._element_diagram_states[key] = value

    def __getitem__(self, key: int) -> ElementState:
        return self._element_diagram_states[key]

    def __delitem__(self, key: int):
        del self._element_diagram_states[key]

    def __contains__(self, key: int):
        return key in self._element_diagram_states

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def generate_index(self) -> int:
        """
        Generate a number used to index a diagram
        """
        return next(self.index_generator)

    def extract_into_diagram(self, el_id: int):
        """
        Used when we encounter the same token twice in the same tree. When this
        happens, we replace all instances of that token with a terminal, and
        create a new subdiagram for the token
        """
        position = self[el_id]

        # Replace the original definition of this element with a regular block
        if position.parent:
            href = f"#{_make_bookmark(position.name)}"
            ret = EditablePartial.from_call(railroad.NonTerminal, text=position.name, href=href)
            if "item" in position.parent.kwargs:
                position.parent.kwargs["item"] = ret
            elif "items" in position.parent.kwargs:
                position.parent.kwargs["items"][position.parent_index] = ret

        # If the element we're extracting is a group, skip to its content but keep the title
        if position.converted.func == railroad.Group:
            content = position.converted.kwargs["item"]
        else:
            content = position.converted

        self.diagrams[el_id] = EditablePartial.from_call(
            NamedDiagram,
            name=position.name,
            diagram=EditablePartial.from_call(
                railroad.Diagram, content, **self.diagram_kwargs
            ),
            index=position.number,
        )

        del self[el_id]


def _worth_extracting(element: pyparsing.ParserElement) -> bool:
    """
    Returns true if this element is worth having its own sub-diagram. Simply, if any of its children
    themselves have children, then its complex enough to extract
    """
    children = element.recurse()
    return any(child.recurse() for child in children)


def _apply_diagram_item_enhancements(fn):
    """
    decorator to ensure enhancements to a diagram item (such as results name annotations)
    get applied on return from _to_diagram_element (we do this since there are several
    returns in _to_diagram_element)
    """

    def _inner(
        element: pyparsing.ParserElement,
        parent: typing.Optional[EditablePartial],
        lookup: ConverterState = None,
        vertical: int = None,
        index: int = 0,
        name_hint: str = None,
        show_results_names: bool = False,
        show_groups: bool = False,
        show_hidden: bool = False,
    ) -> typing.Optional[EditablePartial]:
        ret = fn(
            element,
            parent,
            lookup,
            vertical,
            index,
            name_hint,
            show_results_names,
            show_groups,
            show_hidden,
        )

        # apply annotation for results name, if present
        if show_results_names and ret is not None:
            element_results_name = element.resultsName
            if element_results_name:
                # add "*" to indicate if this is a "list all results" name
                modal_tag = "" if element.modalResults else "*"
                ret = EditablePartial.from_call(
                    railroad.Group,
                    item=ret,
                    label=f"{repr(element_results_name)}{modal_tag}",
                )

        return ret

    return _inner


def _visible_exprs(exprs: Iterable[pyparsing.ParserElement]):
    non_diagramming_exprs = (
        pyparsing.ParseElementEnhance,
        pyparsing.PositionToken,
        pyparsing.And._ErrorStop,
    )
    return [
        e
        for e in exprs
        if not isinstance(e, non_diagramming_exprs)
    ]


@_apply_diagram_item_enhancements
def _to_diagram_element(
    element: pyparsing.ParserElement,
    parent: typing.Optional[EditablePartial],
    lookup: ConverterState = None,
    vertical: int = None,
    index: int = 0,
    name_hint: str = None,
    show_results_names: bool = False,
    show_groups: bool = False,
    show_hidden: bool = False,
) -> typing.Optional[EditablePartial]:
    """
    Recursively converts a PyParsing Element to a railroad Element
    :param lookup: The shared converter state that keeps track of useful things
    :param index: The index of this element within the parent
    :param parent: The parent of this element in the output tree
    :param vertical: Controls at what point we make a list of elements vertical. If this is an integer (the default),
    it sets the threshold of the number of items before we go vertical. If True, always go vertical, if False, never
    do so
    :param name_hint: If provided, this will override the generated name
    :param show_results_names: bool flag indicating whether to add annotations for results names
    :param show_groups: bool flag indicating whether to show groups using bounding box
    :param show_hidden: bool flag indicating whether to show elements that are typically hidden
    :returns: The converted version of the input element, but as a Partial that hasn't yet been constructed
    """
    exprs = element.recurse()
    name = name_hint or element.customName or type(element).__name__

    # Python's id() is used to provide a unique identifier for elements
    el_id = id(element)

    element_results_name = element.resultsName

    # Here we basically bypass processing certain wrapper elements if they contribute nothing to the diagram
    if not element.customName:
        if isinstance(
            element,
            (
                # pyparsing.TokenConverter,
                pyparsing.Forward,
                pyparsing.Located,
                pyparsing.AtStringStart,
                pyparsing.AtLineStart,
            ),
        ):
            # However, if this element has a useful custom name, and its child does not, we can pass it on to the child
            if exprs:
                if not exprs[0].customName:
                    propagated_name = name
                else:
                    propagated_name = None

                return _to_diagram_element(
                    element.expr,
                    parent=parent,
                    lookup=lookup,
                    vertical=vertical,
                    index=index,
                    name_hint=propagated_name,
                    show_results_names=show_results_names,
                    show_groups=show_groups,
                    show_hidden=show_hidden,
                )

    # If the element isn't worth extracting, we always treat it as the first time we say it
    if _worth_extracting(element):
        looked_up = lookup.get(el_id)
        if looked_up and looked_up.name is not None:
            # If we've seen this element exactly once before, we are only just now finding out that it's a duplicate,
            # so we have to extract it into a new diagram.
            looked_up.mark_for_extraction(el_id, lookup, name=name_hint)
            href = f"#{_make_bookmark(looked_up.name)}"
            ret = EditablePartial.from_call(railroad.NonTerminal, text=looked_up.name, href=href)
            return ret

        elif el_id in lookup.diagrams:
            # If we have seen the element at least twice before, and have already extracted it into a subdiagram, we
            # just put in a marker element that refers to the sub-diagram
            text = lookup.diagrams[el_id].kwargs["name"]
            ret = EditablePartial.from_call(
                railroad.NonTerminal, text=text, href=f"#{_make_bookmark(text)}"
            )
            return ret

    # Recursively convert child elements
    # Here we find the most relevant Railroad element for matching pyparsing Element
    # We use ``items=[]`` here to hold the place for where the child elements will go once created

    # see if this element is normally hidden, and whether hidden elements are desired
    # if not, just return None
    if not element.show_in_diagram and not show_hidden:
        return None

    if isinstance(element, pyparsing.And):
        # detect And's created with ``expr*N`` notation - for these use a OneOrMore with a repeat
        # (all will have the same name, and resultsName)
        if not exprs:
            return None
        if len(set((e.name, e.resultsName) for e in exprs)) == 1 and len(exprs) > 2:
            ret = EditablePartial.from_call(
                railroad.OneOrMore, item="", repeat=str(len(exprs))
            )
        elif _should_vertical(vertical, exprs):
            ret = EditablePartial.from_call(railroad.Stack, items=[])
        else:
            ret = EditablePartial.from_call(railroad.Sequence, items=[])
    elif isinstance(element, (pyparsing.Or, pyparsing.MatchFirst)):
        if not exprs:
            return None
        if _should_vertical(vertical, exprs):
            ret = EditablePartial.from_call(railroad.Choice, 0, items=[])
        else:
            ret = EditablePartial.from_call(railroad.HorizontalChoice, items=[])
    elif isinstance(element, pyparsing.Each):
        if not exprs:
            return None
        ret = EditablePartial.from_call(EachItem, items=[])
    elif isinstance(element, pyparsing.NotAny):
        ret = EditablePartial.from_call(AnnotatedItem, label="NOT", item="")
    elif isinstance(element, pyparsing.FollowedBy):
        ret = EditablePartial.from_call(AnnotatedItem, label="LOOKAHEAD", item="")
    elif isinstance(element, pyparsing.PrecededBy):
        ret = EditablePartial.from_call(AnnotatedItem, label="LOOKBEHIND", item="")
    elif isinstance(element, pyparsing.Group):
        if show_groups:
            ret = EditablePartial.from_call(AnnotatedItem, label="", item="")
        else:
            ret = EditablePartial.from_call(
                railroad.Group, item=None, label=element_results_name
            )
    elif isinstance(element, pyparsing.TokenConverter):
        label = type(element).__name__.lower()
        if label == "tokenconverter":
            ret = EditablePartial.from_call(railroad.Sequence, items=[])
        else:
            ret = EditablePartial.from_call(AnnotatedItem, label=label, item="")
    elif isinstance(element, pyparsing.Opt):
        ret = EditablePartial.from_call(railroad.Optional, item="")
    elif isinstance(element, pyparsing.OneOrMore):
        if element.not_ender is not None:
            args = [
                parent,
                lookup,
                vertical,
                index,
                name_hint,
                show_results_names,
                show_groups,
                show_hidden,
            ]
            return _to_diagram_element(
                (~element.not_ender.expr + element.expr)[1, ...].set_name(element.name),
                *args,
            )
        ret = EditablePartial.from_call(railroad.OneOrMore, item=None)
    elif isinstance(element, pyparsing.ZeroOrMore):
        if element.not_ender is not None:
            args = [
                parent,
                lookup,
                vertical,
                index,
                name_hint,
                show_results_names,
                show_groups,
                show_hidden,
            ]
            return _to_diagram_element(
                (~element.not_ender.expr + element.expr)[...].set_name(element.name),
                *args,
            )
        ret = EditablePartial.from_call(railroad.ZeroOrMore, item="")
    elif isinstance(element, pyparsing.Empty) and not element.customName:
        # Skip unnamed "Empty" elements
        ret = None
    elif isinstance(element, pyparsing.ParseElementEnhance):
        ret = EditablePartial.from_call(railroad.Sequence, items=[])
    elif len(exprs) > 0 and not element_results_name:
        ret = EditablePartial.from_call(railroad.Group, item="", label=name)
    elif isinstance(element, pyparsing.Regex):
        collapsed_patt = _collapse_verbose_regex(element.pattern)
        ret = EditablePartial.from_call(railroad.Terminal, collapsed_patt)
    elif len(exprs) > 0:
        ret = EditablePartial.from_call(railroad.Sequence, items=[])
    else:
        terminal = EditablePartial.from_call(railroad.Terminal, element.defaultName)
        ret = terminal

    if ret is None:
        return

    # Indicate this element's position in the tree so we can extract it if necessary
    lookup[el_id] = ElementState(
        element=element,
        converted=ret,
        parent=parent,
        parent_index=index,
        number=lookup.generate_index(),
    )
    if element.customName:
        lookup[el_id].mark_for_extraction(el_id, lookup, element.customName)

    i = 0
    for expr in exprs:
        # Add a placeholder index in case we have to extract the child before we even add it to the parent
        if "items" in ret.kwargs:
            ret.kwargs["items"].insert(i, None)

        item = _to_diagram_element(
            expr,
            parent=ret,
            lookup=lookup,
            vertical=vertical,
            index=i,
            show_results_names=show_results_names,
            show_groups=show_groups,
            show_hidden=show_hidden,
        )

        # Some elements don't need to be shown in the diagram
        if item is not None:
            if "item" in ret.kwargs:
                ret.kwargs["item"] = item
            elif "items" in ret.kwargs:
                # If we've already extracted the child, don't touch this index, since it's occupied by a nonterminal
                ret.kwargs["items"][i] = item
                i += 1
        elif "items" in ret.kwargs:
            # If we're supposed to skip this element, remove it from the parent
            del ret.kwargs["items"][i]

    # If all this items children are none, skip this item
    if ret and (
        ("items" in ret.kwargs and len(ret.kwargs["items"]) == 0)
        or ("item" in ret.kwargs and ret.kwargs["item"] is None)
    ):
        ret = EditablePartial.from_call(railroad.Terminal, name)

    # Mark this element as "complete", ie it has all of its children
    if el_id in lookup:
        lookup[el_id].complete = True

    if el_id in lookup and lookup[el_id].extract and lookup[el_id].complete:
        lookup.extract_into_diagram(el_id)
        if ret is not None:
            text = lookup.diagrams[el_id].kwargs["name"]
            href = f"#{_make_bookmark(text)}"
            ret = EditablePartial.from_call(
                railroad.NonTerminal, text=text, href=href
            )

    return ret
<|endoftext|>import warnings
from functools import lru_cache
import pyparsing as pp


@lru_cache(maxsize=None)
def camel_to_snake(s: str) -> str:
    """
    Convert CamelCase to snake_case.
    """
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in s).lstrip("_")


pre_pep8_method_names = """
addCondition addParseAction anyCloseTag anyOpenTag asDict asList cStyleComment canParseNext conditionAsParseAction 
convertToDate convertToDatetime convertToFloat convertToInteger countedArray cppStyleComment dblQuotedString 
dblSlashComment defaultName dictOf disableMemoization downcaseTokens enableLeftRecursion enablePackrat getName 
htmlComment ignoreWhitespace infixNotation inlineLiteralsUsing javaStyleComment leaveWhitespace 
lineEnd lineStart matchOnlyAtCol matchPreviousExpr matchPreviousLiteral nestedExpr nullDebugAction oneOf 
originalTextFor parseFile parseString parseWithTabs pythonStyleComment quotedString removeQuotes replaceWith 
resetCache restOfLine runTests scanString searchString setBreak setDebug setDebugActions setDefaultWhitespaceChars 
setFailAction setName setParseAction setResultsName setWhitespaceChars sglQuotedString stringEnd stringStart tokenMap 
traceParseAction transformString tryParse unicodeString upcaseTokens withAttribute withClass
""".split()

special_changes = {
    "opAssoc": "OpAssoc",
    "delimitedList": "DelimitedList",
    "delimited_list": "DelimitedList",
    "replaceHTMLEntity": "replace_html_entity",
    "makeHTMLTags": "make_html_tags",
    "makeXMLTags": "make_xml_tags",
    "commonHTMLEntity": "common_html_entity",
    "stripHTMLTags": "strip_html_tags",
    "indentedBlock": "IndentedBlock",
    "locatedExpr": "Located",
}

pre_pep8_arg_names = """parseAll maxMatches listAllMatches callDuringTry includeSeparators fullDump printResults 
failureTests postParse matchString identChars maxMismatches initChars bodyChars asKeyword excludeChars asGroupList 
asMatch quoteChar escChar escQuote unquoteResults endQuoteChar convertWhitespaceEscapes notChars wordChars stopOn 
failOn joinString markerString intExpr useRegex asString ignoreExpr""".split()

special_changes_arg_names = {
    "asList": "aslist",
}

pre_pep8_method_name = pp.one_of(pre_pep8_method_names, as_keyword=True)
pre_pep8_method_name.set_parse_action(lambda t: camel_to_snake(t[0]))
special_pre_pep8_name = pp.one_of(special_changes, as_keyword=True)
def update_special_changes(s, l, t):
    if t[0] == "indentedBlock":
        warnings.warn(
            "Conversion of 'indentedBlock' to new 'IndentedBlock'"
            " requires added code changes to remove 'indentStack' argument\n"
            f"  {pp.lineno(l, s)}: {pp.line(l, s)}",
            stacklevel=2,
        )
    elif t[0] == "locatedExpr":
        warnings.warn(
            "Conversion of 'locatedExpr' to new 'Located'"
            " may require added code changes - Located does not automatically"
            " group parsed elements\n"
            f"  {pp.lineno(l, s)}: {pp.line(l, s)}",
            stacklevel=2,
        )
    return special_changes[t[0]]
special_pre_pep8_name.set_parse_action(update_special_changes)
# only replace arg names if part of an arg list
pre_pep8_arg_name = pp.Regex(
    rf"{pp.util.make_compressed_re(pre_pep8_arg_names)}"
) + pp.FollowedBy("=")
pre_pep8_arg_name.set_parse_action(lambda t: camel_to_snake(t[0]))
special_pre_pep8_arg_name = pp.one_of(special_changes_arg_names, as_keyword=True) + pp.FollowedBy("=")
special_pre_pep8_arg_name.set_parse_action(lambda t: special_changes_arg_names[t[0]])

pep8_converter = special_pre_pep8_arg_name | pre_pep8_method_name | special_pre_pep8_name | pre_pep8_arg_name

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    argparser = argparse.ArgumentParser(
        description = (
            "Utility to convert Python pyparsing scripts using legacy"
            " camelCase names to use PEP8 snake_case names."
            "\nBy default, this script will only show whether this script would make any changes."
        )
    )
    argparser.add_argument("--verbose", "-v", action="store_true", help="Show unified diff for each source file")
    argparser.add_argument("-vv", action="store_true", dest="verbose2", help="Show unified diff for each source file, plus names of scanned files with no changes")
    argparser.add_argument("--update", "-u", action="store_true", help="Update source files in-place")
    argparser.add_argument("--encoding", type=str, default="utf-8", help="Encoding of source files (default: utf-8)")
    argparser.add_argument("--exit-zero-even-if-changed", "-exit0", action="store_true", help="Exit with status code 0 even if changes were made")
    argparser.add_argument("source_filename", nargs="+", help="Source filenames or filename patterns of Python files to be converted")
    args = argparser.parse_args()


    def show_diffs(original, modified):
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(), modified.splitlines(), lineterm=""
        )
        sys.stdout.writelines(f"{diff_line}\n" for diff_line in diff)

    exit_status = 0

    for filename_pattern in args.source_filename:

        for filename in Path().glob(filename_pattern):
            if not Path(filename).is_file():
                continue

            try:
                original_contents = Path(filename).read_text(encoding=args.encoding)
                modified_contents = pep8_converter.transform_string(
                    original_contents
                )

                if modified_contents != original_contents:
                    if args.update:
                        Path(filename).write_text(modified_contents, encoding=args.encoding)
                        print(f"Converted {filename}")
                    else:
                        print(f"Found required changes in {filename}")

                    if args.verbose:
                        show_diffs(original_contents, modified_contents)
                        print()

                    exit_status = 1

                else:
                    if args.verbose2:
                        print(f"No required changes in {filename}")

            except Exception as e:
                print(f"Failed to convert {filename}: {type(e).__name__}: {e}")

    sys.exit(exit_status if not args.exit_zero_even_if_changed else 0)
<|endoftext|>import functools
import itertools
import operator

from jaraco.functools import pass_none
from jaraco.text import yield_lines
from more_itertools import consume

from ._importlib import metadata
from ._itertools import ensure_unique
from .errors import OptionError


def ensure_valid(ep):
    """
    Exercise one of the dynamic properties to trigger
    the pattern match.

    This function is deprecated in favor of importlib_metadata 8.7 and
    Python 3.14 importlib.metadata, which validates entry points on
    construction.
    """
    try:
        ep.extras  # noqa: B018 # evaluated to trigger validation/side effect
    except (AttributeError, AssertionError) as ex:
        # Why both? See https://github.com/python/importlib_metadata/issues/488
        msg = (
            f"Problems to parse {ep}.\nPlease ensure entry-point follows the spec: "
            "https://packaging.python.org/en/latest/specifications/entry-points/"
        )
        raise OptionError(msg) from ex


def load_group(value, group):
    """
    Given a value of an entry point or series of entry points,
    return each as an EntryPoint.
    """
    # normalize to a single sequence of lines
    lines = yield_lines(value)
    text = f'[{group}]\n' + '\n'.join(lines)
    return metadata.EntryPoints._from_text(text)


def by_group_and_name(ep):
    return ep.group, ep.name


def validate(eps: metadata.EntryPoints):
    """
    Ensure entry points are unique by group and name and validate each.
    """
    consume(map(ensure_valid, ensure_unique(eps, key=by_group_and_name)))
    return eps


@functools.singledispatch
def load(eps):
    """
    Given a Distribution.entry_points, produce EntryPoints.
    """
    groups = itertools.chain.from_iterable(
        load_group(value, group) for group, value in eps.items()
    )
    return validate(metadata.EntryPoints(groups))


@load.register(str)
def _(eps):
    r"""
    >>> ep, = load('[console_scripts]\nfoo=bar')
    >>> ep.group
    'console_scripts'
    >>> ep.name
    'foo'
    >>> ep.value
    'bar'
    """
    return validate(metadata.EntryPoints(metadata.EntryPoints._from_text(eps)))


load.register(type(None), lambda x: x)


@pass_none
def render(eps: metadata.EntryPoints):
    by_group = operator.attrgetter('group')
    groups = itertools.groupby(sorted(eps, key=by_group), by_group)

    return '\n'.join(f'[{group}]\n{render_items(items)}\n' for group, items in groups)


def render_items(eps):
    return '\n'.join(f'{ep.name} = {ep.value}' for ep in sorted(eps))
<|endoftext|>import itertools
import os

from .compat import py312

from distutils import log

flatten = itertools.chain.from_iterable


class Installer:
    nspkg_ext = '-nspkg.pth'

    def install_namespaces(self) -> None:
        nsp = self._get_all_ns_packages()
        if not nsp:
            return
        filename = self._get_nspkg_file()
        self.outputs.append(filename)
        log.info("Installing %s", filename)
        lines = map(self._gen_nspkg_line, nsp)

        with open(filename, 'wt', encoding=py312.PTH_ENCODING) as f:
            # Python<3.13 requires encoding="locale" instead of "utf-8"
            # See: python/cpython#77102
            f.writelines(lines)

    def uninstall_namespaces(self) -> None:
        filename = self._get_nspkg_file()
        if not os.path.exists(filename):
            return
        log.info("Removing %s", filename)
        os.remove(filename)

    def _get_nspkg_file(self):
        filename, _ = os.path.splitext(self._get_target())
        return filename + self.nspkg_ext

    def _get_target(self):
        return self.target

    _nspkg_tmpl = (
        "import sys, types, os",
        "p = os.path.join(%(root)s, *%(pth)r)",
        "importlib = __import__('importlib.util')",
        "__import__('importlib.machinery')",
        (
            "m = "
            "sys.modules.setdefault(%(pkg)r, "
            "importlib.util.module_from_spec("
            "importlib.machinery.PathFinder.find_spec(%(pkg)r, "
            "[os.path.dirname(p)])))"
        ),
        ("m = m or sys.modules.setdefault(%(pkg)r, types.ModuleType(%(pkg)r))"),
        "mp = (m or []) and m.__dict__.setdefault('__path__',[])",
        "(p not in mp) and mp.append(p)",
    )
    "lines for the namespace installer"

    _nspkg_tmpl_multi = ('m and setattr(sys.modules[%(parent)r], %(child)r, m)',)
    "additional line(s) when a parent package is indicated"

    def _get_root(self):
        return "sys._getframe(1).f_locals['sitedir']"

    def _gen_nspkg_line(self, pkg):
        pth = tuple(pkg.split('.'))
        root = self._get_root()
        tmpl_lines = self._nspkg_tmpl
        parent, sep, child = pkg.rpartition('.')
        if parent:
            tmpl_lines += self._nspkg_tmpl_multi
        return ';'.join(tmpl_lines) % locals() + '\n'

    def _get_all_ns_packages(self):
        """Return sorted list of all package namespaces"""
        pkgs = self.distribution.namespace_packages or []
        return sorted(set(flatten(map(self._pkg_names, pkgs))))

    @staticmethod
    def _pkg_names(pkg):
        """
        Given a namespace package, yield the components of that
        package.

        >>> names = Installer._pkg_names('a.b.c')
        >>> set(names) == set(['a', 'a.b', 'a.b.c'])
        True
        """
        parts = pkg.split('.')
        while parts:
            yield '.'.join(parts)
            parts.pop()


class DevelopInstaller(Installer):
    def _get_root(self):
        return repr(str(self.egg_path))

    def _get_target(self):
        return self.egg_link
<|endoftext|>"""A PEP 517 interface to setuptools

Previously, when a user or a command line tool (let's call it a "frontend")
needed to make a request of setuptools to take a certain action, for
example, generating a list of installation requirements, the frontend
would call "setup.py egg_info" or "setup.py bdist_wheel" on the command line.

PEP 517 defines a different method of interfacing with setuptools. Rather
than calling "setup.py" directly, the frontend should:

  1. Set the current directory to the directory with a setup.py file
  2. Import this module into a safe python interpreter (one in which
     setuptools can potentially set global variables or crash hard).
  3. Call one of the functions defined in PEP 517.

What each function does is defined in PEP 517. However, here is a "casual"
definition of the functions (this definition should not be relied on for
bug reports or API stability):

  - `build_wheel`: build a wheel in the folder and return the basename
  - `get_requires_for_build_wheel`: get the `setup_requires` to build
  - `prepare_metadata_for_build_wheel`: get the `install_requires`
  - `build_sdist`: build an sdist in the folder and return the basename
  - `get_requires_for_build_sdist`: get the `setup_requires` to build

Again, this is not a formal definition! Just a "taste" of the module.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import tokenize
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import setuptools

from . import errors
from ._path import StrPath, same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning

import distutils
from distutils.util import strtobool

if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = [
    'SetupRequirementsError',
    '__legacy__',
    'build_editable',
    'build_sdist',
    'build_wheel',
    'get_requires_for_build_editable',
    'get_requires_for_build_sdist',
    'get_requires_for_build_wheel',
    'prepare_metadata_for_build_editable',
    'prepare_metadata_for_build_wheel',
]


class SetupRequirementsError(BaseException):
    def __init__(self, specifiers) -> None:
        self.specifiers = specifiers


class Distribution(setuptools.dist.Distribution):
    def fetch_build_eggs(self, specifiers) -> NoReturn:
        specifier_list = list(parse_strings(specifiers))

        raise SetupRequirementsError(specifier_list)

    @classmethod
    @contextlib.contextmanager
    def patch(cls) -> Iterator[None]:
        """
        Replace
        distutils.dist.Distribution with this class
        for the duration of this context.
        """
        orig = distutils.core.Distribution
        distutils.core.Distribution = cls  # type: ignore[misc] # monkeypatching
        try:
            yield
        finally:
            distutils.core.Distribution = orig  # type: ignore[misc] # monkeypatching


@contextlib.contextmanager
def no_install_setup_requires():
    """Temporarily disable installing setup_requires

    Under PEP 517, the backend reports build dependencies to the frontend,
    and the frontend is responsible for ensuring they're installed.
    So setuptools (acting as a backend) should not try to install them.
    """
    orig = setuptools._install_setup_requires
    setuptools._install_setup_requires = lambda attrs: None
    try:
        yield
    finally:
        setuptools._install_setup_requires = orig


def _get_immediate_subdirectories(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def _file_with_extension(directory: StrPath, extension: str | tuple[str, ...]):
    matching = (f for f in os.listdir(directory) if f.endswith(extension))
    try:
        (file,) = matching
    except ValueError:
        raise ValueError(
            'No distribution was found. Ensure that `setup.py` '
            'is not empty and that it calls `setup()`.'
        ) from None
    return file


def _open_setup_script(setup_script):
    if not os.path.exists(setup_script):
        # Supply a default setup.py
        return io.StringIO("from setuptools import setup; setup()")

    return tokenize.open(setup_script)


@contextlib.contextmanager
def suppress_known_deprecation():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'setup.py install is deprecated')
        yield


_ConfigSettings: TypeAlias = Mapping[str, str | list[str] | None] | None
"""
Currently the user can run::

    pip install -e . --config-settings key=value
    python -m build -C--key=value -C key=value

- pip will pass both key and value as strings and overwriting repeated keys
  (pypa/pip#11059).
- build will accumulate values associated with repeated keys in a list.
  It will also accept keys with no associated value.
  This means that an option passed by build can be ``str | list[str] | None``.
- PEP 517 specifies that ``config_settings`` is an optional dict.
"""


class _ConfigSettingsTranslator:
    """Translate ``config_settings`` into distutils-style command arguments.
    Only a limited number of options is currently supported.
    """

    # See pypa/setuptools#1928 pypa/setuptools#2491

    def _get_config(self, key: str, config_settings: _ConfigSettings) -> list[str]:
        """
        Get the value of a specific key in ``config_settings`` as a list of strings.

        >>> fn = _ConfigSettingsTranslator()._get_config
        >>> fn("--global-option", None)
        []
        >>> fn("--global-option", {})
        []
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': ['foo']})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo bar'})
        ['foo', 'bar']
        """
        cfg = config_settings or {}
        opts = cfg.get(key) or []
        return shlex.split(opts) if isinstance(opts, str) else opts

    def _global_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Let the user specify ``verbose`` or ``quiet`` + escape hatch via
        ``--global-option``.
        Note: ``-v``, ``-vv``, ``-vvv`` have similar effects in setuptools,
        so we just have to cover the basic scenario ``-v``.

        >>> fn = _ConfigSettingsTranslator()._global_args
        >>> list(fn(None))
        []
        >>> list(fn({"verbose": "False"}))
        ['-q']
        >>> list(fn({"verbose": "1"}))
        ['-v']
        >>> list(fn({"--verbose": None}))
        ['-v']
        >>> list(fn({"verbose": "true", "--global-option": "-q --no-user-cfg"}))
        ['-v', '-q', '--no-user-cfg']
        >>> list(fn({"--quiet": None}))
        ['-q']
        """
        cfg = config_settings or {}
        falsey = {"false", "no", "0", "off"}
        if "verbose" in cfg or "--verbose" in cfg:
            level = str(cfg.get("verbose") or cfg.get("--verbose") or "1")
            yield ("-q" if level.lower() in falsey else "-v")
        if "quiet" in cfg or "--quiet" in cfg:
            level = str(cfg.get("quiet") or cfg.get("--quiet") or "1")
            yield ("-v" if level.lower() in falsey else "-q")

        yield from self._get_config("--global-option", config_settings)

    def __dist_info_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``dist_info`` command accepts ``tag-date`` and ``tag-build``.

        .. warning::
           We cannot use this yet as it requires the ``sdist`` and ``bdist_wheel``
           commands run in ``build_sdist`` and ``build_wheel`` to reuse the egg-info
           directory created in ``prepare_metadata_for_build_wheel``.

        >>> fn = _ConfigSettingsTranslator()._ConfigSettingsTranslator__dist_info_args
        >>> list(fn(None))
        []
        >>> list(fn({"tag-date": "False"}))
        ['--no-date']
        >>> list(fn({"tag-date": None}))
        ['--no-date']
        >>> list(fn({"tag-date": "true", "tag-build": ".a"}))
        ['--tag-date', '--tag-build', '.a']
        """
        cfg = config_settings or {}
        if "tag-date" in cfg:
            val = strtobool(str(cfg["tag-date"] or "false"))
            yield ("--tag-date" if val else "--no-date")
        if "tag-build" in cfg:
            yield from ["--tag-build", str(cfg["tag-build"])]

    def _editable_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``editable_wheel`` command accepts ``editable-mode=strict``.

        >>> fn = _ConfigSettingsTranslator()._editable_args
        >>> list(fn(None))
        []
        >>> list(fn({"editable-mode": "strict"}))
        ['--mode', 'strict']
        """
        cfg = config_settings or {}
        mode = cfg.get("editable-mode") or cfg.get("editable_mode")
        if not mode:
            return
        yield from ["--mode", str(mode)]

    def _arbitrary_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Users may expect to pass arbitrary lists of arguments to a command
        via "--global-option" (example provided in PEP 517 of a "escape hatch").

        >>> fn = _ConfigSettingsTranslator()._arbitrary_args
        >>> list(fn(None))
        []
        >>> list(fn({}))
        []
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': ['foo']}))
        ['foo']
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': 'foo bar'}))
        ['foo', 'bar']
        >>> list(fn({'--global-option': 'foo'}))
        []
        """
        yield from self._get_config("--build-option", config_settings)


class _BuildMetaBackend(_ConfigSettingsTranslator):
    def _get_build_requires(
        self, config_settings: _ConfigSettings, requirements: list[str]
    ):
        sys.argv = [
            *sys.argv[:1],
            *self._global_args(config_settings),
            "egg_info",
        ]
        try:
            with Distribution.patch():
                self.run_setup()
        except SetupRequirementsError as e:
            requirements += e.specifiers

        return requirements

    def run_setup(self, setup_script: str = 'setup.py') -> None:
        # Note that we can reuse our build directory between calls
        # Correctness comes first, then optimization later
        __file__ = os.path.abspath(setup_script)
        __name__ = '__main__'

        with _open_setup_script(__file__) as f:
            code = f.read().replace(r'\r\n', r'\n')

        try:
            exec(code, locals())  # noqa: S102 # exec is intentional here
        except SystemExit as e:
            if e.code:
                raise
            # We ignore exit code indicating success
            SetuptoolsDeprecationWarning.emit(
                "Running `setup.py` directly as CLI tool is deprecated.",
                "Please avoid using `sys.exit(0)` or similar statements "
                "that don't fit in the paradigm of a configuration file.",
                see_url="https://blog.ganssle.io/articles/2021/10/"
                "setup-py-deprecated.html",
            )

    def get_requires_for_build_wheel(
        self, config_settings: _ConfigSettings = None
    ) -> list[str]:
        return self._get_build_requires(config_settings, requirements=[])

    def get_requires_for_build_sdist(
        self, config_settings: _ConfigSettings = None
    ) -> list[str]:
        return self._get_build_requires(config_settings, requirements=[])

    def _bubble_up_info_directory(
        self, metadata_directory: StrPath, suffix: str
    ) -> str:
        """
        PEP 517 requires that the .dist-info directory be placed in the
        metadata_directory. To comply, we MUST copy the directory to the root.

        Returns the basename of the info directory, e.g. `proj-0.0.0.dist-info`.
        """
        info_dir = self._find_info_directory(metadata_directory, suffix)
        if not same_path(info_dir.parent, metadata_directory):
            shutil.move(str(info_dir), metadata_directory)
            # PEP 517 allow other files and dirs to exist in metadata_directory
        return info_dir.name

    def _find_info_directory(self, metadata_directory: StrPath, suffix: str) -> Path:
        for parent, dirs, _ in os.walk(metadata_directory):
            candidates = [f for f in dirs if f.endswith(suffix)]

            if len(candidates) != 0 or len(dirs) != 1:
                assert len(candidates) == 1, (
                    f"Exactly one {suffix} should have been produced, but found {len(candidates)}: {candidates}"
                )
                return Path(parent, candidates[0])

        msg = f"No {suffix} directory found in {metadata_directory}"
        raise errors.InternalError(msg)

    def prepare_metadata_for_build_wheel(
        self, metadata_directory: StrPath, config_settings: _ConfigSettings = None
    ) -> str:
        sys.argv = [
            *sys.argv[:1],
            *self._global_args(config_settings),
            "dist_info",
            "--output-dir",
            str(metadata_directory),
            "--keep-egg-info",
        ]
        with no_install_setup_requires():
            self.run_setup()

        self._bubble_up_info_directory(metadata_directory, ".egg-info")
        return self._bubble_up_info_directory(metadata_directory, ".dist-info")

    def _build_with_temp_dir(
        self,
        setup_command: Iterable[str],
        result_extension: str | tuple[str, ...],
        result_directory: StrPath,
        config_settings: _ConfigSettings,
        arbitrary_args: Iterable[str] = (),
    ):
        result_directory = os.path.abspath(result_directory)

        # Build in a temporary directory, then copy to the target.
        os.makedirs(result_directory, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".tmp-", dir=result_directory
        ) as tmp_dist_dir:
            sys.argv = [
                *sys.argv[:1],
                *self._global_args(config_settings),
                *setup_command,
                "--dist-dir",
                tmp_dist_dir,
                *arbitrary_args,
            ]
            with no_install_setup_requires():
                self.run_setup()

            result_basename = _file_with_extension(tmp_dist_dir, result_extension)
            result_path = os.path.join(result_directory, result_basename)
            if os.path.exists(result_path):
                # os.rename will fail overwriting on non-Unix.
                os.remove(result_path)
            os.rename(os.path.join(tmp_dist_dir, result_basename), result_path)

        return result_basename

    def build_wheel(
        self,
        wheel_directory: StrPath,
        config_settings: _ConfigSettings = None,
        metadata_directory: StrPath | None = None,
    ) -> str:
        def _build(cmd: list[str]):
            with suppress_known_deprecation():
                return self._build_with_temp_dir(
                    cmd,
                    '.whl',
                    wheel_directory,
                    config_settings,
                    self._arbitrary_args(config_settings),
                )

        if metadata_directory is None:
            return _build(['bdist_wheel'])

        try:
            return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
        except SystemExit as ex:  # pragma: nocover
            # pypa/setuptools#4683
            if "--dist-info-dir not recognized" not in str(ex):
                raise
            _IncompatibleBdistWheel.emit()
            return _build(['bdist_wheel'])

    def build_sdist(
        self, sdist_directory: StrPath, config_settings: _ConfigSettings = None
    ) -> str:
        return self._build_with_temp_dir(
            ['sdist', '--formats', 'gztar'], '.tar.gz', sdist_directory, config_settings
        )

    def _get_dist_info_dir(self, metadata_directory: StrPath | None) -> str | None:
        if not metadata_directory:
            return None
        dist_info_candidates = list(Path(metadata_directory).glob("*.dist-info"))
        assert len(dist_info_candidates) <= 1
        return str(dist_info_candidates[0]) if dist_info_candidates else None

    def build_editable(
        self,
        wheel_directory: StrPath,
        config_settings: _ConfigSettings = None,
        metadata_directory: StrPath | None = None,
    ) -> str:
        # XXX can or should we hide our editable_wheel command normally?
        info_dir = self._get_dist_info_dir(metadata_directory)
        opts = ["--dist-info-dir", info_dir] if info_dir else []
        cmd = ["editable_wheel", *opts, *self._editable_args(config_settings)]
        with suppress_known_deprecation():
            return self._build_with_temp_dir(
                cmd, ".whl", wheel_directory, config_settings
            )

    def get_requires_for_build_editable(
        self, config_settings: _ConfigSettings = None
    ) -> list[str]:
        return self.get_requires_for_build_wheel(config_settings)

    def prepare_metadata_for_build_editable(
        self, metadata_directory: StrPath, config_settings: _ConfigSettings = None
    ) -> str:
        return self.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


class _BuildMetaLegacyBackend(_BuildMetaBackend):
    """Compatibility backend for setuptools

    This is a version of setuptools.build_meta that endeavors
    to maintain backwards
    compatibility with pre-PEP 517 modes of invocation. It
    exists as a temporary
    bridge between the old packaging mechanism and the new
    packaging mechanism,
    and will eventually be removed.
    """

    def run_setup(self, setup_script: str = 'setup.py') -> None:
        # In order to maintain compatibility with scripts assuming that
        # the setup.py script is in a directory on the PYTHONPATH, inject
        # '' into sys.path. (pypa/setuptools#1642)
        sys_path = list(sys.path)  # Save the original path

        script_dir = os.path.dirname(os.path.abspath(setup_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Some setup.py scripts (e.g. in pygame and numpy) use sys.argv[0] to
        # get the directory of the source code. They expect it to refer to the
        # setup.py script.
        sys_argv_0 = sys.argv[0]
        sys.argv[0] = setup_script

        try:
            super().run_setup(setup_script=setup_script)
        finally:
            # While PEP 517 frontends should be calling each hook in a fresh
            # subprocess according to the standard (and thus it should not be
            # strictly necessary to restore the old sys.path), we'll restore
            # the original path so that the path manipulation does not persist
            # within the hook after run_setup is called.
            sys.path[:] = sys_path
            sys.argv[0] = sys_argv_0


class _IncompatibleBdistWheel(SetuptoolsDeprecationWarning):
    _SUMMARY = "wheel.bdist_wheel is deprecated, please import it from setuptools"
    _DETAILS = """
    Ensure that any custom bdist_wheel implementation is a subclass of
    setuptools.command.bdist_wheel.bdist_wheel.
    """
    _DUE_DATE = (2025, 10, 15)
    # Initially introduced in 2024/10/15, but maybe too disruptive to be enforced?
    _SEE_URL = "https://github.com/pypa/wheel/pull/631"


# The primary backend
_BACKEND = _BuildMetaBackend()

get_requires_for_build_wheel = _BACKEND.get_requires_for_build_wheel
get_requires_for_build_sdist = _BACKEND.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _BACKEND.prepare_metadata_for_build_wheel
build_wheel = _BACKEND.build_wheel
build_sdist = _BACKEND.build_sdist
get_requires_for_build_editable = _BACKEND.get_requires_for_build_editable
prepare_metadata_for_build_editable = _BACKEND.prepare_metadata_for_build_editable
build_editable = _BACKEND.build_editable


# The legacy backend
__legacy__ = _BuildMetaLegacyBackend()
<|endoftext|>"""
Handling of Core Metadata for Python packages (including reading and writing).

See: https://packaging.python.org/en/latest/specifications/core-metadata/
"""

from __future__ import annotations

import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, canonicalize_version
from packaging.version import Version

from . import _normalization, _reqs
from ._static import is_static
from .warnings import SetuptoolsDeprecationWarning

from distutils.util import rfc822_escape


def get_metadata_version(self):
    mv = getattr(self, 'metadata_version', None)
    if mv is None:
        mv = Version('2.4')
        self.metadata_version = mv
    return mv


def rfc822_unescape(content: str) -> str:
    """Reverse RFC-822 escaping by removing leading whitespaces from content."""
    lines = content.splitlines()
    if len(lines) == 1:
        return lines[0].lstrip()
    return '\n'.join((lines[0].lstrip(), textwrap.dedent('\n'.join(lines[1:]))))


def _read_field_from_msg(msg: Message, field: str) -> str | None:
    """Read Message header field."""
    value = msg[field]
    if value == 'UNKNOWN':
        return None
    return value


def _read_field_unescaped_from_msg(msg: Message, field: str) -> str | None:
    """Read Message header field and apply rfc822_unescape."""
    value = _read_field_from_msg(msg, field)
    if value is None:
        return value
    return rfc822_unescape(value)


def _read_list_from_msg(msg: Message, field: str) -> list[str] | None:
    """Read Message header field and return all results as list."""
    values = msg.get_all(field, None)
    if values == []:
        return None
    return values


def _read_payload_from_msg(msg: Message) -> str | None:
    value = str(msg.get_payload()).strip()
    if value == 'UNKNOWN' or not value:
        return None
    return value


def read_pkg_file(self, file):
    """Reads the metadata values from a file object."""
    msg = message_from_file(file)

    self.metadata_version = Version(msg['metadata-version'])
    self.name = _read_field_from_msg(msg, 'name')
    self.version = _read_field_from_msg(msg, 'version')
    self.description = _read_field_from_msg(msg, 'summary')
    # we are filling author only.
    self.author = _read_field_from_msg(msg, 'author')
    self.maintainer = None
    self.author_email = _read_field_from_msg(msg, 'author-email')
    self.maintainer_email = None
    self.url = _read_field_from_msg(msg, 'home-page')
    self.download_url = _read_field_from_msg(msg, 'download-url')
    self.license = _read_field_unescaped_from_msg(msg, 'license')
    self.license_expression = _read_field_unescaped_from_msg(msg, 'license-expression')

    self.long_description = _read_field_unescaped_from_msg(msg, 'description')
    if self.long_description is None and self.metadata_version >= Version('2.1'):
        self.long_description = _read_payload_from_msg(msg)
    self.description = _read_field_from_msg(msg, 'summary')

    if 'keywords' in msg:
        self.keywords = _read_field_from_msg(msg, 'keywords').split(',')

    self.platforms = _read_list_from_msg(msg, 'platform')
    self.classifiers = _read_list_from_msg(msg, 'classifier')

    # PEP 314 - these fields only exist in 1.1
    if self.metadata_version == Version('1.1'):
        self.requires = _read_list_from_msg(msg, 'requires')
        self.provides = _read_list_from_msg(msg, 'provides')
        self.obsoletes = _read_list_from_msg(msg, 'obsoletes')
    else:
        self.requires = None
        self.provides = None
        self.obsoletes = None

    self.license_files = _read_list_from_msg(msg, 'license-file')


def single_line(val):
    """
    Quick and dirty validation for Summary pypa/setuptools#1390.
    """
    if '\n' in val:
        # TODO: Replace with `raise ValueError("newlines not allowed")`
        # after reviewing #2893.
        msg = "newlines are not allowed in `summary` and will break in the future"
        SetuptoolsDeprecationWarning.emit("Invalid config.", msg)
        # due_date is undefined. Controversial change, there was a lot of push back.
        val = val.strip().split('\n')[0]
    return val


def write_pkg_info(self, base_dir):
    """Write the PKG-INFO file into the release tree."""
    temp = ""
    final = os.path.join(base_dir, 'PKG-INFO')
    try:
        # Use a temporary file while writing to avoid race conditions
        # (e.g. `importlib.metadata` reading `.egg-info/PKG-INFO`):
        with NamedTemporaryFile("w", encoding="utf-8", dir=base_dir, delete=False) as f:
            temp = f.name
            self.write_pkg_file(f)
        permissions = stat.S_IMODE(os.lstat(temp).st_mode)
        os.chmod(temp, permissions | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temp, final)  # atomic operation.
    finally:
        if temp and os.path.exists(temp):
            os.remove(temp)


# Based on Python 3.5 version
def write_pkg_file(self, file):  # noqa: C901  # is too complex (14)  # FIXME
    """Write the PKG-INFO format data to a file object."""
    version = self.get_metadata_version()

    def write_field(key, value):
        file.write(f"{key}: {value}\n")

    write_field('Metadata-Version', str(version))
    write_field('Name', self.get_name())
    write_field('Version', self.get_version())

    summary = self.get_description()
    if summary:
        write_field('Summary', single_line(summary))

    optional_fields = (
        ('Home-page', 'url'),
        ('Download-URL', 'download_url'),
        ('Author', 'author'),
        ('Author-email', 'author_email'),
        ('Maintainer', 'maintainer'),
        ('Maintainer-email', 'maintainer_email'),
    )

    for field, attr in optional_fields:
        attr_val = getattr(self, attr, None)
        if attr_val is not None:
            write_field(field, attr_val)

    if license_expression := self.license_expression:
        write_field('License-Expression', license_expression)
    elif license := self.get_license():
        write_field('License', rfc822_escape(license))

    for label, url in self.project_urls.items():
        write_field('Project-URL', f'{label}, {url}')

    keywords = ','.join(self.get_keywords())
    if keywords:
        write_field('Keywords', keywords)

    platforms = self.get_platforms() or []
    for platform in platforms:
        write_field('Platform', platform)

    self._write_list(file, 'Classifier', self.get_classifiers())

    # PEP 314
    self._write_list(file, 'Requires', self.get_requires())
    self._write_list(file, 'Provides', self.get_provides())
    self._write_list(file, 'Obsoletes', self.get_obsoletes())

    # Setuptools specific for PEP 345
    if hasattr(self, 'python_requires'):
        write_field('Requires-Python', self.python_requires)

    # PEP 566
    if self.long_description_content_type:
        write_field('Description-Content-Type', self.long_description_content_type)

    safe_license_files = map(_safe_license_file, self.license_files or [])
    self._write_list(file, 'License-File', safe_license_files)
    _write_requirements(self, file)

    for field, attr in _POSSIBLE_DYNAMIC_FIELDS.items():
        if (val := getattr(self, attr, None)) and not is_static(val):
            write_field('Dynamic', field)

    long_description = self.get_long_description()
    if long_description:
        file.write(f"\n{long_description}")
        if not long_description.endswith("\n"):
            file.write("\n")


def _write_requirements(self, file):
    for req in _reqs.parse(self.install_requires):
        file.write(f"Requires-Dist: {req}\n")

    processed_extras = {}
    for augmented_extra, reqs in self.extras_require.items():
        # Historically, setuptools allows "augmented extras": `<extra>:<condition>`
        unsafe_extra, _, condition = augmented_extra.partition(":")
        unsafe_extra = unsafe_extra.strip()
        extra = _normalization.safe_extra(unsafe_extra)

        if extra:
            _write_provides_extra(file, processed_extras, extra, unsafe_extra)
        for req in _reqs.parse_strings(reqs):
            r = _include_extra(req, extra, condition.strip())
            file.write(f"Requires-Dist: {r}\n")

    return processed_extras


def _include_extra(req: str, extra: str, condition: str) -> Requirement:
    r = Requirement(req)  # create a fresh object that can be modified
    parts = (
        f"({r.marker})" if r.marker else None,
        f"({condition})" if condition else None,
        f"extra == {extra!r}" if extra else None,
    )
    r.marker = Marker(" and ".join(x for x in parts if x))
    return r


def _write_provides_extra(file, processed_extras, safe, unsafe):
    previous = processed_extras.get(safe)
    if previous == unsafe:
        SetuptoolsDeprecationWarning.emit(
            'Ambiguity during "extra" normalization for dependencies.',
            f"""
            {previous!r} and {unsafe!r} normalize to the same value:\n
                {safe!r}\n
            In future versions, setuptools might halt the build process.
            """,
            see_url="https://peps.python.org/pep-0685/",
        )
    else:
        processed_extras[safe] = unsafe
        file.write(f"Provides-Extra: {safe}\n")


# from pypa/distutils#244; needed only until that logic is always available
def get_fullname(self):
    return _distribution_fullname(self.get_name(), self.get_version())


def _distribution_fullname(name: str, version: str) -> str:
    """
    >>> _distribution_fullname('setup.tools', '1.0-2')
    'setup_tools-1.0.post2'
    >>> _distribution_fullname('setup-tools', '1.2post2')
    'setup_tools-1.2.post2'
    >>> _distribution_fullname('setup-tools', '1.0-r2')
    'setup_tools-1.0.post2'
    >>> _distribution_fullname('setup.tools', '1.0.post')
    'setup_tools-1.0.post0'
    >>> _distribution_fullname('setup.tools', '1.0+ubuntu-1')
    'setup_tools-1.0+ubuntu.1'
    """
    return "{}-{}".format(
        canonicalize_name(name).replace('-', '_'),
        canonicalize_version(version, strip_trailing_zero=False),
    )


def _safe_license_file(file):
    # XXX: Do we need this after the deprecation discussed in #4892, #4896??
    normalized = os.path.normpath(file).replace(os.sep, "/")
    if "../" in normalized:
        return os.path.basename(normalized)  # Temporarily restore pre PEP639 behaviour
    return normalized


_POSSIBLE_DYNAMIC_FIELDS = {
    # Core Metadata Field x related Distribution attribute
    "author": "author",
    "author-email": "author_email",
    "classifier": "classifiers",
    "description": "long_description",
    "description-content-type": "long_description_content_type",
    "download-url": "download_url",
    "home-page": "url",
    "keywords": "keywords",
    "license": "license",
    # XXX: License-File is complicated because the user gives globs that are expanded
    #      during the build. Without special handling it is likely always
    #      marked as Dynamic, which is an acceptable outcome according to:
    #      https://github.com/pypa/setuptools/issues/4629#issuecomment-2331233677
    "license-file": "license_files",
    "license-expression": "license_expression",  # PEP 639
    "maintainer": "maintainer",
    "maintainer-email": "maintainer_email",
    "obsoletes": "obsoletes",
    # "obsoletes-dist": "obsoletes_dist",  # NOT USED
    "platform": "platforms",
    "project-url": "project_urls",
    "provides": "provides",
    # "provides-dist": "provides_dist",  # NOT USED
    "provides-extra": "extras_require",
    "requires": "requires",
    "requires-dist": "install_requires",
    # "requires-external": "requires_external",  # NOT USED
    "requires-python": "python_requires",
    "summary": "description",
    # "supported-platform": "supported_platforms",  # NOT USED
}
<|endoftext|>"""
Monkey patching of distutils.
"""

from __future__ import annotations

import inspect
import platform
import sys
import types
from typing import TypeVar, cast, overload

import distutils.filelist

_T = TypeVar("_T")
_UnpatchT = TypeVar("_UnpatchT", type, types.FunctionType)


__all__: list[str] = []
"""
Everything is private. Contact the project team
if you think you need this functionality.
"""


def _get_mro(cls):
    """
    Returns the bases classes for cls sorted by the MRO.

    Works around an issue on Jython where inspect.getmro will not return all
    base classes if multiple classes share the same name. Instead, this
    function will return a tuple containing the class itself, and the contents
    of cls.__bases__. See https://github.com/pypa/setuptools/issues/1024.
    """
    if platform.python_implementation() == "Jython":
        return (cls,) + cls.__bases__
    return inspect.getmro(cls)


@overload
def get_unpatched(item: _UnpatchT) -> _UnpatchT: ...
@overload
def get_unpatched(item: object) -> None: ...
def get_unpatched(
    item: type | types.FunctionType | object,
) -> type | types.FunctionType | None:
    if isinstance(item, type):
        return get_unpatched_class(item)
    if isinstance(item, types.FunctionType):
        return get_unpatched_function(item)
    return None


def get_unpatched_class(cls: type[_T]) -> type[_T]:
    """Protect against re-patching the distutils if reloaded

    Also ensures that no other distutils extension monkeypatched the distutils
    first.
    """
    external_bases = (
        cast(type[_T], cls)
        for cls in _get_mro(cls)
        if not cls.__module__.startswith('setuptools')
    )
    base = next(external_bases)
    if not base.__module__.startswith('distutils'):
        msg = f"distutils has already been patched by {cls!r}"
        raise AssertionError(msg)
    return base


def patch_all() -> None:
    import setuptools

    # we can't patch distutils.cmd, alas
    distutils.core.Command = setuptools.Command  # type: ignore[misc,assignment] # monkeypatching

    _patch_distribution_metadata()

    # Install Distribution throughout the distutils
    for module in distutils.dist, distutils.core, distutils.cmd:
        module.Distribution = setuptools.dist.Distribution

    # Install the patched Extension
    distutils.core.Extension = setuptools.extension.Extension  # type: ignore[misc,assignment] # monkeypatching
    distutils.extension.Extension = setuptools.extension.Extension  # type: ignore[misc,assignment] # monkeypatching
    if 'distutils.command.build_ext' in sys.modules:
        sys.modules[
            'distutils.command.build_ext'
        ].Extension = setuptools.extension.Extension


def _patch_distribution_metadata():
    from . import _core_metadata

    """Patch write_pkg_file and read_pkg_file for higher metadata standards"""
    for attr in (
        'write_pkg_info',
        'write_pkg_file',
        'read_pkg_file',
        'get_metadata_version',
        'get_fullname',
    ):
        new_val = getattr(_core_metadata, attr)
        setattr(distutils.dist.DistributionMetadata, attr, new_val)


def patch_func(replacement, target_mod, func_name) -> None:
    """
    Patch func_name in target_mod with replacement

    Important - original must be resolved by name to avoid
    patching an already patched function.
    """
    original = getattr(target_mod, func_name)

    # set the 'unpatched' attribute on the replacement to
    # point to the original.
    vars(replacement).setdefault('unpatched', original)

    # replace the function in the original module
    setattr(target_mod, func_name, replacement)


def get_unpatched_function(candidate):
    return candidate.unpatched
<|endoftext|>from functools import wraps
from typing import TypeVar

import packaging.specifiers

from .warnings import SetuptoolsDeprecationWarning


class Static:
    """
    Wrapper for built-in object types that are allow setuptools to identify
    static core metadata (in opposition to ``Dynamic``, as defined :pep:`643`).

    The trick is to mark values with :class:`Static` when they come from
    ``pyproject.toml`` or ``setup.cfg``, so if any plugin overwrite the value
    with a built-in, setuptools will be able to recognise the change.

    We inherit from built-in classes, so that we don't need to change the existing
    code base to deal with the new types.
    We also should strive for immutability objects to avoid changes after the
    initial parsing.
    """

    _mutated_: bool = False  # TODO: Remove after deprecation warning is solved


def _prevent_modification(target: type, method: str, copying: str) -> None:
    """
    Because setuptools is very flexible we cannot fully prevent
    plugins and user customizations from modifying static values that were
    parsed from config files.
    But we can attempt to block "in-place" mutations and identify when they
    were done.
    """
    fn = getattr(target, method, None)
    if fn is None:
        return

    @wraps(fn)
    def _replacement(self: Static, *args, **kwargs):
        # TODO: After deprecation period raise NotImplementedError instead of warning
        #       which obviated the existence and checks of the `_mutated_` attribute.
        self._mutated_ = True
        SetuptoolsDeprecationWarning.emit(
            "Direct modification of value will be disallowed",
            f"""
            In an effort to implement PEP 643, direct/in-place changes of static values
            that come from configuration files are deprecated.
            If you need to modify this value, please first create a copy with {copying}
            and make sure conform to all relevant standards when overriding setuptools
            functionality (https://packaging.python.org/en/latest/specifications/).
            """,
            due_date=(2025, 10, 10),  # Initially introduced in 2024-09-06
        )
        return fn(self, *args, **kwargs)

    _replacement.__doc__ = ""  # otherwise doctest may fail.
    setattr(target, method, _replacement)


class Str(str, Static):
    pass


class Tuple(tuple, Static):
    pass


class List(list, Static):
    """
    :meta private:
    >>> x = List([1, 2, 3])
    >>> is_static(x)
    True
    >>> x += [0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    SetuptoolsDeprecationWarning: Direct modification ...
    >>> is_static(x)  # no longer static after modification
    False
    >>> y = list(x)
    >>> y.clear()
    >>> y
    []
    >>> y == x
    False
    >>> is_static(List(y))
    True
    """


# Make `List` immutable-ish
# (certain places of setuptools/distutils issue a warn if we use tuple instead of list)
for _method in (
    '__delitem__',
    '__iadd__',
    '__setitem__',
    'append',
    'clear',
    'extend',
    'insert',
    'remove',
    'reverse',
    'pop',
):
    _prevent_modification(List, _method, "`list(value)`")


class Dict(dict, Static):
    """
    :meta private:
    >>> x = Dict({'a': 1, 'b': 2})
    >>> is_static(x)
    True
    >>> x['c'] = 0  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    SetuptoolsDeprecationWarning: Direct modification ...
    >>> x._mutated_
    True
    >>> is_static(x)  # no longer static after modification
    False
    >>> y = dict(x)
    >>> y.popitem()
    ('b', 2)
    >>> y == x
    False
    >>> is_static(Dict(y))
    True
    """


# Make `Dict` immutable-ish (we cannot inherit from types.MappingProxyType):
for _method in (
    '__delitem__',
    '__ior__',
    '__setitem__',
    'clear',
    'pop',
    'popitem',
    'setdefault',
    'update',
):
    _prevent_modification(Dict, _method, "`dict(value)`")


class SpecifierSet(packaging.specifiers.SpecifierSet, Static):
    """Not exactly a built-in type but useful for ``requires-python``"""


T = TypeVar("T")


def noop(value: T) -> T:
    """
    >>> noop(42)
    42
    """
    return value


_CONVERSIONS = {str: Str, tuple: Tuple, list: List, dict: Dict}


def attempt_conversion(value: T) -> T:
    """
    >>> is_static(attempt_conversion("hello"))
    True
    >>> is_static(object())
    False
    """
    return _CONVERSIONS.get(type(value), noop)(value)  # type: ignore[call-overload]


def is_static(value: object) -> bool:
    """
    >>> is_static(a := Dict({'a': 1}))
    True
    >>> is_static(dict(a))
    False
    >>> is_static(b := List([1, 2, 3]))
    True
    >>> is_static(list(b))
    False
    """
    return isinstance(value, Static) and not value._mutated_


EMPTY_LIST = List()
EMPTY_DICT = Dict()
<|endoftext|>from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from setuptools._path import StrPath

from .monkey import get_unpatched

import distutils.core
import distutils.errors
import distutils.extension


def _have_cython() -> bool:
    """
    Return True if Cython can be imported.
    """
    cython_impl = 'Cython.Distutils.build_ext'
    try:
        # from (cython_impl) import build_ext
        __import__(cython_impl, fromlist=['build_ext']).build_ext  # noqa: B018 # evaluated to trigger validation/side effect
    except Exception:  # noqa: BLE001 # intentional broad fallback
        return False
    return True


# for compatibility
have_pyrex = _have_cython
if TYPE_CHECKING:
    # Work around a mypy issue where type[T] can't be used as a base: https://github.com/python/mypy/issues/10962
    from distutils.core import Extension as _Extension
else:
    _Extension = get_unpatched(distutils.core.Extension)


class Extension(_Extension):
    """
    Describes a single extension module.

    This means that all source files will be compiled into a single binary file
    ``<module path>.<suffix>`` (with ``<module path>`` derived from ``name`` and
    ``<suffix>`` defined by one of the values in
    ``importlib.machinery.EXTENSION_SUFFIXES``).

    In the case ``.pyx`` files are passed as ``sources and`` ``Cython`` is **not**
    installed in the build environment, ``setuptools`` may also try to look for the
    equivalent ``.cpp`` or ``.c`` files.

    :arg str name:
      the full name of the extension, including any packages -- ie.
      *not* a filename or pathname, but Python dotted name

    :arg Iterable[str | os.PathLike[str]] sources:
      iterable of source filenames, (except strings, which could be misinterpreted
      as a single filename), relative to the distribution root
      (where the setup script lives), in Unix form (slash-separated)
      for portability.  Source files may be C, C++, SWIG (.i),
      platform-specific resource files, or whatever else is recognized
      by the "build_ext" command as source for a Python extension.

    :keyword list[str] include_dirs:
      list of directories to search for C/C++ header files (in Unix
      form for portability)

    :keyword list[tuple[str, str|None]] define_macros:
      list of macros to define; each macro is defined using a 2-tuple:
      the first item corresponding to the name of the macro and the second
      item either a string with its value or None to
      define it without a particular value (equivalent of "#define
      FOO" in source or -DFOO on Unix C compiler command line)

    :keyword list[str] undef_macros:
      list of macros to undefine explicitly

    :keyword list[str] library_dirs:
      list of directories to search for C/C++ libraries at link time

    :keyword list[str] libraries:
      list of library names (not filenames or paths) to link against

    :keyword list[str] runtime_library_dirs:
      list of directories to search for C/C++ libraries at run time
      (for shared extensions, this is when the extension is loaded).
      Setting this will cause an exception during build on Windows
      platforms.

    :keyword list[str] extra_objects:
      list of extra files to link with (eg. object files not implied
      by 'sources', static library that must be explicitly specified,
      binary resource files, etc.)

    :keyword list[str] extra_compile_args:
      any extra platform- and compiler-specific information to use
      when compiling the source files in 'sources'.  For platforms and
      compilers where "command line" makes sense, this is typically a
      list of command-line arguments, but for other platforms it could
      be anything.

    :keyword list[str] extra_link_args:
      any extra platform- and compiler-specific information to use
      when linking object files together to create the extension (or
      to create a new static Python interpreter).  Similar
      interpretation as for 'extra_compile_args'.

    :keyword list[str] export_symbols:
      list of symbols to be exported from a shared extension.  Not
      used on all platforms, and not generally necessary for Python
      extensions, which typically export exactly one symbol: "init" +
      extension_name.

    :keyword list[str] swig_opts:
      any extra options to pass to SWIG if a source file has the .i
      extension.

    :keyword list[str] depends:
      list of files that the extension depends on

    :keyword str language:
      extension language (i.e. "c", "c++", "objc"). Will be detected
      from the source extensions if not provided.

    :keyword bool optional:
      specifies that a build failure in the extension should not abort the
      build process, but simply not install the failing extension.

    :keyword bool py_limited_api:
      opt-in flag for the usage of :doc:`Python's limited API <python:c-api/stable>`.

    :raises setuptools.errors.PlatformError: if ``runtime_library_dirs`` is
      specified on Windows. (since v63)
    """

    # These 4 are set and used in setuptools/command/build_ext.py
    # The lack of a default value and risk of `AttributeError` is purposeful
    # to avoid people forgetting to call finalize_options if they modify the extension list.
    # See example/rationale in https://github.com/pypa/setuptools/issues/4529.
    _full_name: str  #: Private API, internal use only.
    _links_to_dynamic: bool  #: Private API, internal use only.
    _needs_stub: bool  #: Private API, internal use only.
    _file_name: str  #: Private API, internal use only.

    def __init__(
        self,
        name: str,
        sources: Iterable[StrPath],
        *args,
        py_limited_api: bool = False,
        **kw,
    ) -> None:
        # The *args is needed for compatibility as calls may use positional
        # arguments. py_limited_api may be set only via keyword.
        self.py_limited_api = py_limited_api
        super().__init__(
            name,
            sources,  # type: ignore[arg-type] # Vendored version of setuptools supports PathLike
            *args,
            **kw,
        )

    def _convert_pyx_sources_to_lang(self):
        """
        Replace sources with .pyx extensions to sources with the target
        language extension. This mechanism allows language authors to supply
        pre-converted sources but to prefer the .pyx sources.
        """
        if _have_cython():
            # the build has Cython, so allow it to compile the .pyx files
            return
        lang = self.language or ''
        target_ext = '.cpp' if lang.lower() == 'c++' else '.c'
        sub = functools.partial(re.sub, '.pyx$', target_ext)
        self.sources = list(map(sub, self.sources))


class Library(Extension):
    """Just like a regular Extension, but built as a library instead"""
<|endoftext|>"""setuptools.errors

Provides exceptions used by setuptools modules.
"""

from __future__ import annotations

from distutils import errors as _distutils_errors

# Re-export errors from distutils to facilitate the migration to PEP632

ByteCompileError = _distutils_errors.DistutilsByteCompileError
CCompilerError = _distutils_errors.CCompilerError
ClassError = _distutils_errors.DistutilsClassError
CompileError = _distutils_errors.CompileError
ExecError = _distutils_errors.DistutilsExecError
FileError = _distutils_errors.DistutilsFileError
InternalError = _distutils_errors.DistutilsInternalError
LibError = _distutils_errors.LibError
LinkError = _distutils_errors.LinkError
ModuleError = _distutils_errors.DistutilsModuleError
OptionError = _distutils_errors.DistutilsOptionError
PlatformError = _distutils_errors.DistutilsPlatformError
PreprocessError = _distutils_errors.PreprocessError
SetupError = _distutils_errors.DistutilsSetupError
TemplateError = _distutils_errors.DistutilsTemplateError
UnknownFileError = _distutils_errors.UnknownFileError

# The root error class in the hierarchy
BaseError = _distutils_errors.DistutilsError


class InvalidConfigError(OptionError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for invalid configurations."""


class RemovedConfigError(OptionError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for configurations that were deprecated and removed."""


class RemovedCommandError(BaseError, RuntimeError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for commands that have been removed in setuptools.

    Since ``setuptools`` is built on ``distutils``, simply removing a command
    from ``setuptools`` will make the behavior fall back to ``distutils``; this
    error is raised if a command exists in ``distutils`` but has been actively
    removed in ``setuptools``.
    """


class PackageDiscoveryError(BaseError, RuntimeError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Impossible to perform automatic discovery of packages and/or modules.

    The current project layout or given discovery options can lead to problems when
    scanning the project directory.

    Setuptools might also refuse to complete auto-discovery if an error prone condition
    is detected (e.g. when a project is organised as a flat-layout but contains
    multiple directories that can be taken as top-level packages inside a single
    distribution [*]_). In these situations the users are encouraged to be explicit
    about which packages to include or to make the discovery parameters more specific.

    .. [*] Since multi-package distributions are uncommon it is very likely that the
       developers did not intend for all the directories to be packaged, and are just
       leaving auxiliary code in the repository top-level, such as maintenance-related
       scripts.
    """
<|endoftext|>from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING, TypeVar

from more_itertools import unique_everseen

if TYPE_CHECKING:
    from typing import TypeAlias

StrPath: TypeAlias = str | os.PathLike[str]  #  Same as _typeshed.StrPath
StrPathT = TypeVar("StrPathT", bound=str | os.PathLike[str])


def ensure_directory(path):
    """Ensure that the parent directory of `path` exists"""
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def same_path(p1: StrPath, p2: StrPath) -> bool:
    """Differs from os.path.samefile because it does not require paths to exist.
    Purely string based (no comparison between i-nodes).
    >>> same_path("a/b", "./a/b")
    True
    >>> same_path("a/b", "a/./b")
    True
    >>> same_path("a/b", "././a/b")
    True
    >>> same_path("a/b", "./a/b/c/..")
    True
    >>> same_path("a/b", "../a/b/c")
    False
    >>> same_path("a", "a/b")
    False
    """
    return normpath(p1) == normpath(p2)


def _cygwin_patch(filename: StrPath):  # pragma: nocover
    """
    Contrary to POSIX 2008, on Cygwin, getcwd (3) contains
    symlink components. Using
    os.path.abspath() works around this limitation. A fix in os.getcwd()
    would probably better, in Cygwin even more so, except
    that this seems to be by design...
    """
    return os.path.abspath(filename) if sys.platform == 'cygwin' else filename


def normpath(filename: StrPath) -> str:
    """Normalize a file/dir name for comparison purposes."""
    return os.path.normcase(os.path.realpath(os.path.normpath(_cygwin_patch(filename))))


@contextlib.contextmanager
def paths_on_pythonpath(paths):
    """
    Add the indicated paths to the head of the PYTHONPATH environment
    variable so that subprocesses will also see the packages at
    these paths.

    Do this in a context that restores the value on exit.

    >>> getfixture('monkeypatch').setenv('PYTHONPATH', 'anything')
    >>> with paths_on_pythonpath(['foo', 'bar']):
    ...     assert 'foo' in os.environ['PYTHONPATH']
    ...     assert 'anything' in os.environ['PYTHONPATH']
    >>> os.environ['PYTHONPATH']
    'anything'

    >>> getfixture('monkeypatch').delenv('PYTHONPATH')
    >>> with paths_on_pythonpath(['foo', 'bar']):
    ...     assert 'foo' in os.environ['PYTHONPATH']
    >>> os.environ.get('PYTHONPATH')
    """
    nothing = object()
    orig_pythonpath = os.environ.get('PYTHONPATH', nothing)
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    try:
        prefix = os.pathsep.join(unique_everseen(paths))
        to_join = filter(None, [prefix, current_pythonpath])
        new_path = os.pathsep.join(to_join)
        if new_path:
            os.environ['PYTHONPATH'] = new_path
        yield
    finally:
        if orig_pythonpath is nothing:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = orig_pythonpath
<|endoftext|>"""
Helpers for normalization as expected in wheel/sdist/module file names
and core metadata
"""

import re
from typing import TYPE_CHECKING

import packaging

# https://packaging.python.org/en/latest/specifications/core-metadata/#name
_VALID_NAME = re.compile(r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", re.IGNORECASE)
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Z0-9._-]+", re.IGNORECASE)
_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]+", re.IGNORECASE)
_PEP440_FALLBACK = re.compile(
    r"^v?(?P<safe>(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*)", re.IGNORECASE
)


def safe_identifier(name: str) -> str:
    """Make a string safe to be used as Python identifier.
    >>> safe_identifier("12abc")
    '_12abc'
    >>> safe_identifier("__editable__.myns.pkg-78.9.3_local")
    '__editable___myns_pkg_78_9_3_local'
    """
    safe = re.sub(r'\W|^(?=\d)', '_', name)
    assert safe.isidentifier()
    return safe


def safe_name(component: str) -> str:
    """Escape a component used as a project name according to Core Metadata.
    >>> safe_name("hello world")
    'hello-world'
    >>> safe_name("hello?world")
    'hello-world'
    >>> safe_name("hello_world")
    'hello_world'
    """
    return _UNSAFE_NAME_CHARS.sub("-", component)


def safe_version(version: str) -> str:
    """Convert an arbitrary string into a valid version string.
    Can still raise an ``InvalidVersion`` exception.
    To avoid exceptions use ``best_effort_version``.
    >>> safe_version("1988 12 25")
    '1988.12.25'
    >>> safe_version("v0.2.1")
    '0.2.1'
    >>> safe_version("v0.2?beta")
    '0.2b0'
    >>> safe_version("v0.2 beta")
    '0.2b0'
    >>> safe_version("ubuntu lts")
    Traceback (most recent call last):
    ...
    packaging.version.InvalidVersion: Invalid version: 'ubuntu.lts'
    """
    v = version.replace(' ', '.')
    try:
        return str(packaging.version.Version(v))
    except packaging.version.InvalidVersion:
        attempt = _UNSAFE_NAME_CHARS.sub("-", v)
        return str(packaging.version.Version(attempt))


def best_effort_version(version: str) -> str:
    """Convert an arbitrary string into a version-like string.
    Fallback when ``safe_version`` is not safe enough.
    >>> best_effort_version("v0.2 beta")
    '0.2b0'
    >>> best_effort_version("ubuntu lts")
    '0.dev0+sanitized.ubuntu.lts'
    >>> best_effort_version("0.23ubuntu1")
    '0.23.dev0+sanitized.ubuntu1'
    >>> best_effort_version("0.23-")
    '0.23.dev0+sanitized'
    >>> best_effort_version("0.-_")
    '0.dev0+sanitized'
    >>> best_effort_version("42.+?1")
    '42.dev0+sanitized.1'
    """
    try:
        return safe_version(version)
    except packaging.version.InvalidVersion:
        v = version.replace(' ', '.')
        match = _PEP440_FALLBACK.search(v)
        if match:
            safe = match["safe"]
            rest = v[len(safe) :]
        else:
            safe = "0"
            rest = version
        safe_rest = _NON_ALPHANUMERIC.sub(".", rest).strip(".")
        local = f"sanitized.{safe_rest}".strip(".")
        return safe_version(f"{safe}.dev0+{local}")


def safe_extra(extra: str) -> str:
    """Normalize extra name according to PEP 685
    >>> safe_extra("_FrIeNdLy-._.-bArD")
    'friendly-bard'
    >>> safe_extra("FrIeNdLy-._.-bArD__._-")
    'friendly-bard'
    """
    return _NON_ALPHANUMERIC.sub("-", extra).strip("-").lower()


def filename_component(value: str) -> str:
    """Normalize each component of a filename (e.g. distribution/version part of wheel)
    Note: ``value`` needs to be already normalized.
    >>> filename_component("my-pkg")
    'my_pkg'
    """
    return value.replace("-", "_").strip("_")


def filename_component_broken(value: str) -> str:
    """
    Produce the incorrect filename component for compatibility.

    See pypa/setuptools#4167 for detailed analysis.

    TODO: replace this with filename_component after pip 24 is
    nearly-ubiquitous.

    >>> filename_component_broken('foo_bar-baz')
    'foo-bar-baz'
    """
    return value.replace('_', '-')


def safer_name(value: str) -> str:
    """Like ``safe_name`` but can be used as filename component for wheel"""
    # See bdist_wheel.safer_name
    return (
        # Per https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization
        re
        .sub(r"[-_.]+", "-", safe_name(value))
        .lower()
        # Per https://packaging.python.org/en/latest/specifications/binary-distribution-format/#escaping-and-unicode
        .replace("-", "_")
    )


def safer_best_effort_version(value: str) -> str:
    """Like ``best_effort_version`` but can be used as filename component for wheel"""
    # See bdist_wheel.safer_verion
    # TODO: Replace with only safe_version in the future (no need for best effort)
    return filename_component(best_effort_version(value))


def _missing_canonicalize_license_expression(expression: str) -> str:
    """
    Defer import error to affect only users that actually use it
    https://github.com/pypa/setuptools/issues/4894
    >>> _missing_canonicalize_license_expression("a OR b")
    Traceback (most recent call last):
    ...
    ImportError: ...Cannot import `packaging.licenses`...
    """
    raise ImportError(
        "Cannot import `packaging.licenses`."
        """
        Setuptools>=77.0.0 requires "packaging>=24.2" to work properly.
        Please make sure you have a suitable version installed.
        """
    )


try:
    from packaging.licenses import (
        canonicalize_license_expression as _canonicalize_license_expression,
    )
except ImportError:  # pragma: nocover
    if not TYPE_CHECKING:
        # XXX: pyright is still upset even with # pyright: ignore[reportAssignmentType]
        _canonicalize_license_expression = _missing_canonicalize_license_expression
<|endoftext|>import platform


def windows_only(func):
    if platform.system() != 'Windows':
        return lambda *args, **kwargs: None
    return func


@windows_only
def hide_file(path: str) -> None:
    """
    Set the hidden attribute on a file or directory.

    From https://stackoverflow.com/questions/19622133/

    `path` must be text.
    """
    import ctypes
    import ctypes.wintypes

    SetFileAttributes = ctypes.windll.kernel32.SetFileAttributesW
    SetFileAttributes.argtypes = ctypes.wintypes.LPWSTR, ctypes.wintypes.DWORD
    SetFileAttributes.restype = ctypes.wintypes.BOOL

    FILE_ATTRIBUTE_HIDDEN = 0x02

    ret = SetFileAttributes(path, FILE_ATTRIBUTE_HIDDEN)
    if not ret:
        raise ctypes.WinError()
<|endoftext|>"""
Filename globbing utility. Mostly a copy of `glob` from Python 3.5.

Changes include:
 * `yield from` and PEP3102 `*` removed.
 * Hidden files are not ignored.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    from _typeshed import BytesPath, StrOrBytesPath, StrPath

    _StrOrBytesT = TypeVar("_StrOrBytesT", str, bytes)

__all__ = ["escape", "glob", "iglob"]


def glob(pathname: _StrOrBytesT, recursive: bool = False) -> list[_StrOrBytesT]:
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, recursive=recursive))


def iglob(pathname: _StrOrBytesT, recursive: bool = False) -> Iterator[_StrOrBytesT]:
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    it = _iglob(pathname, recursive)
    if recursive and _isrecursive(pathname):
        s = next(it)  # skip empty string
        assert not s
    return it


def _iglob(pathname: _StrOrBytesT, recursive: bool) -> Iterator[_StrOrBytesT]:
    dirname, basename = os.path.split(pathname)
    glob_in_dir = glob2 if recursive and _isrecursive(basename) else glob1

    if not has_magic(pathname):
        if basename:
            if os.path.lexists(pathname):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if os.path.isdir(dirname):
                yield pathname
        return

    if not dirname:
        yield from glob_in_dir(dirname, basename)
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs: Iterable[_StrOrBytesT] = _iglob(dirname, recursive)
    else:
        dirs = [dirname]
    if not has_magic(basename):
        glob_in_dir = glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename):
            yield os.path.join(dirname, name)


# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames. `glob1` accepts a pattern while `glob0`
# takes a literal basename (so it only has to check for its existence).


@overload
def glob1(dirname: StrPath, pattern: str) -> list[str]: ...
@overload
def glob1(dirname: BytesPath, pattern: bytes) -> list[bytes]: ...
def glob1(dirname: StrOrBytesPath, pattern: str | bytes) -> list[str] | list[bytes]:
    if not dirname:
        if isinstance(pattern, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except OSError:
        return []
    # mypy false-positives: str or bytes type possibility is always kept in sync
    return fnmatch.filter(names, pattern)  # type: ignore[type-var, return-value]


def glob0(dirname, basename):
    if not basename:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if os.path.isdir(dirname):
            return [basename]
    else:
        if os.path.lexists(os.path.join(dirname, basename)):
            return [basename]
    return []


# This helper function recursively yields relative pathnames inside a literal
# directory.


@overload
def glob2(dirname: StrPath, pattern: str) -> Iterator[str]: ...
@overload
def glob2(dirname: BytesPath, pattern: bytes) -> Iterator[bytes]: ...
def glob2(dirname: StrOrBytesPath, pattern: str | bytes) -> Iterator[str | bytes]:
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname)


# Recursively yields relative pathnames inside a literal directory.
@overload
def _rlistdir(dirname: StrPath) -> Iterator[str]: ...
@overload
def _rlistdir(dirname: BytesPath) -> Iterator[bytes]: ...
def _rlistdir(dirname: StrOrBytesPath) -> Iterator[str | bytes]:
    if not dirname:
        if isinstance(dirname, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except OSError:
        return
    for x in names:
        yield x
        # mypy false-positives: str or bytes type possibility is always kept in sync
        path = os.path.join(dirname, x) if dirname else x  # type: ignore[arg-type]
        for y in _rlistdir(path):
            yield os.path.join(x, y)  # type: ignore[arg-type]


magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')


def has_magic(s: str | bytes) -> bool:
    if isinstance(s, bytes):
        return magic_check_bytes.search(s) is not None
    else:
        return magic_check.search(s) is not None


def _isrecursive(pattern: str | bytes) -> bool:
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'


def escape(pathname):
    """Escape all special characters."""
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(rb'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname
<|endoftext|>"""
Launch the Python script on the command line after
setuptools is bootstrapped via import.
"""

# Note that setuptools gets imported implicitly by the
# invocation of this script using python -m setuptools.launch

import sys
import tokenize


def run() -> None:
    """
    Run the script in sys.argv[1] as if it had
    been invoked naturally.
    """
    __builtins__  # noqa: B018 # evaluated to trigger validation/side effect
    script_name = sys.argv[1]
    namespace = dict(
        __file__=script_name,
        __name__='__main__',
        __doc__=None,
    )
    sys.argv[:] = sys.argv[1:]

    open_ = getattr(tokenize, 'open', open)
    with open_(script_name) as fid:
        script = fid.read()
    norm_script = script.replace('\\r\\n', '\\n')
    code = compile(norm_script, script_name, 'exec')
    exec(code, namespace)  # noqa: S102 # exec is intentional here


if __name__ == '__main__':
    run()
<|endoftext|>from __future__ import annotations

import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from ._importlib import metadata, resources

if TYPE_CHECKING:
    from typing_extensions import Self

from .warnings import SetuptoolsWarning

from distutils.command.build_scripts import first_line_re
from distutils.util import get_platform


class _SplitArgs(TypedDict, total=False):
    comments: bool
    posix: bool


class CommandSpec(list):
    """
    A command spec for a #! header, specified as a list of arguments akin to
    those passed to Popen.
    """

    options: list[str] = []  # noqa: RUF012 # reassigned per-instance in CommandSpec.best
    split_args = _SplitArgs()

    @classmethod
    def best(cls):
        """
        Choose the best CommandSpec class based on environmental conditions.
        """
        return cls

    @classmethod
    def _sys_executable(cls):
        _default = os.path.normpath(sys.executable)
        return os.environ.get('__PYVENV_LAUNCHER__', _default)

    @classmethod
    def from_param(cls, param: Self | str | Iterable[str] | None) -> Self:
        """
        Construct a CommandSpec from a parameter to build_scripts, which may
        be None.
        """
        if isinstance(param, cls):
            return param
        if isinstance(param, str):
            return cls.from_string(param)
        if isinstance(param, Iterable):
            return cls(param)
        if param is None:
            return cls.from_environment()
        raise TypeError(f"Argument has an unsupported type {type(param)}")

    @classmethod
    def from_environment(cls):
        return cls([cls._sys_executable()])

    @classmethod
    def from_string(cls, string: str) -> Self:
        """
        Construct a command spec from a simple string representing a command
        line parseable by shlex.split.
        """
        items = shlex.split(string, **cls.split_args)
        return cls(items)

    def install_options(self, script_text: str):
        self.options = shlex.split(self._extract_options(script_text))
        cmdline = subprocess.list2cmdline(self)
        if not isascii(cmdline):
            self.options[:0] = ['-x']

    @staticmethod
    def _extract_options(orig_script):
        """
        Extract any options from the first line of the script.
        """
        first = (orig_script + '\n').splitlines()[0]
        match = _first_line_re().match(first)
        options = match.group(1) or '' if match else ''
        return options.strip()

    def as_header(self):
        return self._render(self + list(self.options))

    @staticmethod
    def _strip_quotes(item):
        _QUOTES = '"\''
        for q in _QUOTES:
            if item.startswith(q) and item.endswith(q):
                return item[1:-1]
        return item

    @staticmethod
    def _render(items):
        cmdline = subprocess.list2cmdline(
            CommandSpec._strip_quotes(item.strip()) for item in items
        )
        return '#!' + cmdline + '\n'


class WindowsCommandSpec(CommandSpec):
    split_args = _SplitArgs(posix=False)


class ScriptWriter:
    """
    Encapsulates behavior around writing entry point scripts for console and
    gui apps.
    """

    template = textwrap.dedent(
        r"""
        # EASY-INSTALL-ENTRY-SCRIPT: %(spec)r,%(group)r,%(name)r
        import re
        import sys

        # for compatibility with easy_install; see #2198
        __requires__ = %(spec)r

        try:
            from importlib.metadata import distribution
        except ImportError:
            try:
                from importlib_metadata import distribution
            except ImportError:
                from pkg_resources import load_entry_point


        def importlib_load_entry_point(spec, group, name):
            dist_name, _, _ = spec.partition('==')
            matches = (
                entry_point
                for entry_point in distribution(dist_name).entry_points
                if entry_point.group == group and entry_point.name == name
            )
            return next(matches).load()


        globals().setdefault('load_entry_point', importlib_load_entry_point)


        if __name__ == '__main__':
            sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
            sys.exit(load_entry_point(%(spec)r, %(group)r, %(name)r)())
        """
    ).lstrip()

    command_spec_class = CommandSpec

    @classmethod
    def get_args(cls, dist, header=None):
        """
        Yield write_script() argument tuples for a distribution's
        console_scripts and gui_scripts entry points.
        """

        # If distribution is not an importlib.metadata.Distribution, assume
        # it's a pkg_resources.Distribution and transform it.
        if not hasattr(dist, 'entry_points'):
            SetuptoolsWarning.emit("Unsupported distribution encountered.")
            dist = metadata.Distribution.at(dist.egg_info)

        if header is None:
            header = cls.get_header()
        spec = f'{dist.name}=={dist.version}'
        for type_ in 'console', 'gui':
            group = f'{type_}_scripts'
            for ep in dist.entry_points.select(group=group):
                name = ep.name
                cls._ensure_safe_name(ep.name)
                script_text = cls.template % locals()
                args = cls._get_script_args(type_, ep.name, header, script_text)
                yield from args

    @staticmethod
    def _ensure_safe_name(name):
        """
        Prevent paths in *_scripts entry point names.
        """
        has_path_sep = re.search(r'[\\/]', name)
        if has_path_sep:
            raise ValueError("Path separators not allowed in script names")

    @classmethod
    def best(cls):
        """
        Select the best ScriptWriter for this environment.
        """
        if sys.platform == 'win32' or (os.name == 'java' and os._name == 'nt'):
            return WindowsScriptWriter.best()
        else:
            return cls

    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        # Simply write the stub with no extension.
        yield (name, header + script_text)

    @classmethod
    def get_header(
        cls,
        script_text: str = "",
        executable: str | CommandSpec | Iterable[str] | None = None,
    ) -> str:
        """Create a #! line, getting options (if any) from script_text"""
        cmd = cls.command_spec_class.best().from_param(executable)
        cmd.install_options(script_text)
        return cmd.as_header()


class WindowsScriptWriter(ScriptWriter):
    command_spec_class = WindowsCommandSpec

    @classmethod
    def best(cls):
        """
        Select the best ScriptWriter suitable for Windows
        """
        writer_lookup = dict(
            executable=WindowsExecutableLauncherWriter,
            natural=cls,
        )
        # for compatibility, use the executable launcher by default
        launcher = os.environ.get('SETUPTOOLS_LAUNCHER', 'executable')
        return writer_lookup[launcher]

    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        "For Windows, add a .py extension"
        ext = dict(console='.pya', gui='.pyw')[type_]
        if ext not in os.environ['PATHEXT'].lower().split(';'):
            msg = (
                "{ext} not listed in PATHEXT; scripts will not be "
                "recognized as executables."
            ).format(**locals())
            SetuptoolsWarning.emit(msg)
        old = ['.pya', '.py', '-script.py', '.pyc', '.pyo', '.pyw', '.exe']
        old.remove(ext)
        header = cls._adjust_header(type_, header)
        blockers = [name + x for x in old]
        yield name + ext, header + script_text, 't', blockers

    @classmethod
    def _adjust_header(cls, type_, orig_header):
        """
        Make sure 'pythonw' is used for gui and 'python' is used for
        console (regardless of what sys.executable is).
        """
        pattern = 'pythonw.exe'
        repl = 'python.exe'
        if type_ == 'gui':
            pattern, repl = repl, pattern
        pattern_ob = re.compile(re.escape(pattern), re.IGNORECASE)
        new_header = pattern_ob.sub(string=orig_header, repl=repl)
        return new_header if cls._use_header(new_header) else orig_header

    @staticmethod
    def _use_header(new_header):
        """
        Should _adjust_header use the replaced header?

        On non-windows systems, always use. On
        Windows systems, only use the replaced header if it resolves
        to an executable on the system.
        """
        clean_header = new_header[2:-1].strip('"')
        return sys.platform != 'win32' or shutil.which(clean_header)


class WindowsExecutableLauncherWriter(WindowsScriptWriter):
    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        """
        For Windows, add a .py extension and an .exe launcher
        """
        if type_ == 'gui':
            launcher_type = 'gui'
            ext = '-script.pyw'
            old = ['.pyw']
        else:
            launcher_type = 'cli'
            ext = '-script.py'
            old = ['.py', '.pyc', '.pyo']
        hdr = cls._adjust_header(type_, header)
        blockers = [name + x for x in old]
        yield (name + ext, hdr + script_text, 't', blockers)
        yield (
            name + '.exe',
            get_win_launcher(launcher_type),
            'b',  # write in binary mode
        )
        if not is_64bit():
            # install a manifest for the launcher to prevent Windows
            # from detecting it as an installer (which it will for
            #  launchers like easy_install.exe). Consider only
            #  adding a manifest for launchers detected as installers.
            #  See Distribute #143 for details.
            m_name = name + '.exe.manifest'
            yield (m_name, load_launcher_manifest(name), 't')


def get_win_launcher(type):
    """
    Load the Windows launcher (executable) suitable for launching a script.

    `type` should be either 'cli' or 'gui'

    Returns the executable as a byte string.
    """
    launcher_fn = f'{type}.exe'
    if is_64bit():
        if get_platform() == "win-arm64":
            launcher_fn = launcher_fn.replace(".", "-arm64.")
        else:
            launcher_fn = launcher_fn.replace(".", "-64.")
    else:
        launcher_fn = launcher_fn.replace(".", "-32.")
    return resources.files('setuptools').joinpath(launcher_fn).read_bytes()


def load_launcher_manifest(name):
    res = resources.files(__name__).joinpath('launcher manifest.xml')
    return res.read_text(encoding='utf-8') % vars()


def _first_line_re():
    """
    Return a regular expression based on first_line_re suitable for matching
    strings.
    """
    if isinstance(first_line_re.pattern, str):
        return first_line_re

    # first_line_re in Python >=3.1.4 and >=3.2.1 is a bytes pattern.
    return re.compile(first_line_re.pattern.decode())


def is_64bit():
    return struct.calcsize("P") == 8


def isascii(s):
    try:
        s.encode('ascii')
    except UnicodeError:
        return False
    return True
<|endoftext|>import inspect
import logging
import sys

from . import monkey

import distutils.log


def _not_warning(record):
    return record.levelno < logging.WARNING


def configure() -> None:
    """
    Configure logging to emit warning and above to stderr
    and everything else to stdout. This behavior is provided
    for compatibility with distutils.log but may change in
    the future.
    """
    err_handler = logging.StreamHandler()
    err_handler.setLevel(logging.WARNING)
    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.addFilter(_not_warning)
    handlers = err_handler, out_handler
    logging.basicConfig(
        format="{message}", style='{', handlers=handlers, level=logging.DEBUG
    )
    if inspect.ismodule(distutils.dist.log):
        monkey.patch_func(set_threshold, distutils.log, 'set_threshold')
        # For some reason `distutils.log` module is getting cached in `distutils.dist`
        # and then loaded again when patched,
        # implying: id(distutils.log) != id(distutils.dist.log).
        # Make sure the same module object is used everywhere:
        distutils.dist.log = distutils.log


def set_threshold(level: int) -> int:
    logging.root.setLevel(level * 10)
    return set_threshold.unpatched(level)
<|endoftext|>"""Automatic discovery of Python modules and packages (for inclusion in the
distribution) and other config values.

For the purposes of this module, the following nomenclature is used:

- "src-layout": a directory representing a Python project that contains a "src"
  folder. Everything under the "src" folder is meant to be included in the
  distribution when packaging the project. Example::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── src/
        └── mypkg/
            ├── __init__.py
            ├── mymodule.py
            └── my_data_file.txt

- "flat-layout": a Python project that does not use "src-layout" but instead
  have a directory under the project root for each package::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── mypkg/
        ├── __init__.py
        ├── mymodule.py
        └── my_data_file.txt

- "single-module": a project that contains a single Python script direct under
  the project root (no directory used)::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── mymodule.py

"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Iterator, Mapping
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import _distutils_hack.override  # noqa: F401

from ._path import StrPath

from distutils import log
from distutils.util import convert_path

if TYPE_CHECKING:
    from setuptools import Distribution

chain_iter = itertools.chain.from_iterable


def _valid_name(path: StrPath) -> bool:
    # Ignore invalid names that cannot be imported directly
    return os.path.basename(path).isidentifier()


class _Filter:
    """
    Given a list of patterns, create a callable that will be true only if
    the input matches at least one of the patterns.
    """

    def __init__(self, *patterns: str) -> None:
        self._patterns = dict.fromkeys(patterns)

    def __call__(self, item: str) -> bool:
        return any(fnmatchcase(item, pat) for pat in self._patterns)

    def __contains__(self, item: str) -> bool:
        return item in self._patterns


class _Finder:
    """Base class that exposes functionality for module/package finders"""

    ALWAYS_EXCLUDE: ClassVar[tuple[str, ...]] = ()
    DEFAULT_EXCLUDE: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def find(
        cls,
        where: StrPath = '.',
        exclude: Iterable[str] = (),
        include: Iterable[str] = ('*',),
    ) -> list[str]:
        """Return a list of all Python items (packages or modules, depending on
        the finder implementation) found within directory ``where``.

        ``where`` is the root directory which will be searched.
        It should be supplied as a "cross-platform" (i.e. URL-style) path;
        it will be converted to the appropriate local path syntax.

        ``exclude`` is a sequence of names to exclude; ``*`` can be used
        as a wildcard in the names.
        When finding packages, ``foo.*`` will exclude all subpackages of ``foo``
        (but not ``foo`` itself).

        ``include`` is a sequence of names to include.
        If it's specified, only the named items will be included.
        If it's not specified, all found items will be included.
        ``include`` can contain shell style wildcard patterns just like
        ``exclude``.
        """

        exclude = exclude or cls.DEFAULT_EXCLUDE
        return list(
            cls._find_iter(
                convert_path(str(where)),
