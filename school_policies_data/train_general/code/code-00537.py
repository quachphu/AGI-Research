                return x.domain

    @property
    def route(self):
        for x in self:
            if x.token_type == 'obs-route':
                return x.domains

    @property
    def addr_spec(self):
        for x in self:
            if x.token_type == 'addr-spec':
                if x.local_part:
                    return x.addr_spec
                else:
                    return quote_string(x.local_part) + x.addr_spec
        else:
            return '<>'


class ObsRoute(TokenList):

    token_type = 'obs-route'

    @property
    def domains(self):
        return [x.domain for x in self if x.token_type == 'domain']


class Mailbox(TokenList):

    token_type = 'mailbox'

    @property
    def display_name(self):
        if self[0].token_type == 'name-addr':
            return self[0].display_name

    @property
    def local_part(self):
        return self[0].local_part

    @property
    def domain(self):
        return self[0].domain

    @property
    def route(self):
        if self[0].token_type == 'name-addr':
            return self[0].route

    @property
    def addr_spec(self):
        return self[0].addr_spec


class InvalidMailbox(TokenList):

    token_type = 'invalid-mailbox'

    @property
    def display_name(self):
        return None

    local_part = domain = route = addr_spec = display_name


class Domain(TokenList):

    token_type = 'domain'
    as_ew_allowed = False

    @property
    def domain(self):
        return ''.join(super().value.split())


class DotAtom(TokenList):
    token_type = 'dot-atom'


class DotAtomText(TokenList):
    token_type = 'dot-atom-text'
    as_ew_allowed = True


class NoFoldLiteral(TokenList):
    token_type = 'no-fold-literal'
    as_ew_allowed = False


class AddrSpec(TokenList):

    token_type = 'addr-spec'
    as_ew_allowed = False

    @property
    def local_part(self):
        return self[0].local_part

    @property
    def domain(self):
        if len(self) < 3:
            return None
        return self[-1].domain

    @property
    def value(self):
        if len(self) < 3:
            return self[0].value
        return self[0].value.rstrip()+self[1].value+self[2].value.lstrip()

    @property
    def addr_spec(self):
        nameset = set(self.local_part)
        if len(nameset) > len(nameset-DOT_ATOM_ENDS):
            lp = quote_string(self.local_part)
        else:
            lp = self.local_part
        if self.domain is not None:
            return lp + '@' + self.domain
        return lp


class ObsLocalPart(TokenList):

    token_type = 'obs-local-part'
    as_ew_allowed = False


class DisplayName(Phrase):

    token_type = 'display-name'
    ew_combine_allowed = False

    @property
    def display_name(self):
        res = TokenList(self)
        if len(res) == 0:
            return res.value
        if res[0].token_type == 'cfws':
            res.pop(0)
        else:
            if res[0][0].token_type == 'cfws':
                res[0] = TokenList(res[0][1:])
        if res[-1].token_type == 'cfws':
            res.pop()
        else:
            if res[-1][-1].token_type == 'cfws':
                res[-1] = TokenList(res[-1][:-1])
        return res.value

    @property
    def value(self):
        quote = False
        if self.defects:
            quote = True
        else:
            for x in self:
                if x.token_type == 'quoted-string':
                    quote = True
        if len(self) != 0 and quote:
            pre = post = ''
            if self[0].token_type=='cfws' or self[0][0].token_type=='cfws':
                pre = ' '
            if self[-1].token_type=='cfws' or self[-1][-1].token_type=='cfws':
                post = ' '
            return pre+quote_string(self.display_name)+post
        else:
            return super().value


class LocalPart(TokenList):

    token_type = 'local-part'
    as_ew_allowed = False

    @property
    def value(self):
        if self[0].token_type == "quoted-string":
            return self[0].quoted_value
        else:
            return self[0].value

    @property
    def local_part(self):
        # Strip whitespace from front, back, and around dots.
        res = [DOT]
        last = DOT
        last_is_tl = False
        for tok in self[0] + [DOT]:
            if tok.token_type == 'cfws':
                continue
            if (last_is_tl and tok.token_type == 'dot' and
                    last[-1].token_type == 'cfws'):
                res[-1] = TokenList(last[:-1])
            is_tl = isinstance(tok, TokenList)
            if (is_tl and last.token_type == 'dot' and
                    tok[0].token_type == 'cfws'):
                res.append(TokenList(tok[1:]))
            else:
                res.append(tok)
            last = res[-1]
            last_is_tl = is_tl
        res = TokenList(res[1:-1])
        return res.value


class DomainLiteral(TokenList):

    token_type = 'domain-literal'
    as_ew_allowed = False

    @property
    def domain(self):
        return ''.join(super().value.split())

    @property
    def ip(self):
        for x in self:
            if x.token_type == 'ptext':
                return x.value


class MIMEVersion(TokenList):

    token_type = 'mime-version'
    major = None
    minor = None


class Parameter(TokenList):

    token_type = 'parameter'
    sectioned = False
    extended = False
    charset = 'us-ascii'

    @property
    def section_number(self):
        # Because the first token, the attribute (name) eats CFWS, the second
        # token is always the section if there is one.
        return self[1].number if self.sectioned else 0

    @property
    def param_value(self):
        # This is part of the "handle quoted extended parameters" hack.
        for token in self:
            if token.token_type == 'value':
                return token.stripped_value
            if token.token_type == 'quoted-string':
                for token in token:
                    if token.token_type == 'bare-quoted-string':
                        for token in token:
                            if token.token_type == 'value':
                                return token.stripped_value
        return ''


class InvalidParameter(Parameter):

    token_type = 'invalid-parameter'


class Attribute(TokenList):

    token_type = 'attribute'

    @property
    def stripped_value(self):
        for token in self:
            if token.token_type.endswith('attrtext'):
                return token.value

class Section(TokenList):

    token_type = 'section'
    number = None


class Value(TokenList):

    token_type = 'value'

    @property
    def stripped_value(self):
        token = self[0]
        if token.token_type == 'cfws':
            token = self[1]
        if token.token_type.endswith(
                ('quoted-string', 'attribute', 'extended-attribute')):
            return token.stripped_value
        return self.value


class MimeParameters(TokenList):

    token_type = 'mime-parameters'
    syntactic_break = False

    @property
    def params(self):
        # The RFC specifically states that the ordering of parameters is not
        # guaranteed and may be reordered by the transport layer.  So we have
        # to assume the RFC 2231 pieces can come in any order.  However, we
        # output them in the order that we first see a given name, which gives
        # us a stable __str__.
        params = {}  # Using order preserving dict from Python 3.7+
        for token in self:
            if not token.token_type.endswith('parameter'):
                continue
            if token[0].token_type != 'attribute':
                continue
            name = token[0].value.strip()
            if name not in params:
                params[name] = []
            params[name].append((token.section_number, token))
        for name, parts in params.items():
            parts = sorted(parts, key=itemgetter(0))
            first_param = parts[0][1]
            charset = first_param.charset
            # Our arbitrary error recovery is to ignore duplicate parameters,
            # to use appearance order if there are duplicate rfc 2231 parts,
            # and to ignore gaps.  This mimics the error recovery of get_param.
            if not first_param.extended and len(parts) > 1:
                if parts[1][0] == 0:
                    parts[1][1].defects.append(errors.InvalidHeaderDefect(
                        'duplicate parameter name; duplicate(s) ignored'))
                    parts = parts[:1]
                # Else assume the *0* was missing...note that this is different
                # from get_param, but we registered a defect for this earlier.
            value_parts = []
            i = 0
            for section_number, param in parts:
                if section_number != i:
                    # We could get fancier here and look for a complete
                    # duplicate extended parameter and ignore the second one
                    # seen.  But we're not doing that.  The old code didn't.
                    if not param.extended:
                        param.defects.append(errors.InvalidHeaderDefect(
                            'duplicate parameter name; duplicate ignored'))
                        continue
                    else:
                        param.defects.append(errors.InvalidHeaderDefect(
                            "inconsistent RFC2231 parameter numbering"))
                i += 1
                value = param.param_value
                if param.extended:
                    try:
                        value = urllib.parse.unquote_to_bytes(value)
                    except UnicodeEncodeError:
                        # source had surrogate escaped bytes.  What we do now
                        # is a bit of an open question.  I'm not sure this is
                        # the best choice, but it is what the old algorithm did
                        value = urllib.parse.unquote(value, encoding='latin-1')
                    else:
                        try:
                            value = value.decode(charset, 'surrogateescape')
                        except (LookupError, UnicodeEncodeError):
                            # XXX: there should really be a custom defect for
                            # unknown character set to make it easy to find,
                            # because otherwise unknown charset is a silent
                            # failure.
                            value = value.decode('us-ascii', 'surrogateescape')
                        if utils._has_surrogates(value):
                            param.defects.append(errors.UndecodableBytesDefect())
                value_parts.append(value)
            value = ''.join(value_parts)
            yield name, value

    def __str__(self):
        params = []
        for name, value in self.params:
            if value:
                params.append('{}={}'.format(name, quote_string(value)))
            else:
                params.append(name)
        params = '; '.join(params)
        return ' ' + params if params else ''


class ParameterizedHeaderValue(TokenList):

    # Set this false so that the value doesn't wind up on a new line even
    # if it and the parameters would fit there but not on the first line.
    syntactic_break = False

    @property
    def params(self):
        for token in reversed(self):
            if token.token_type == 'mime-parameters':
                return token.params
        return {}


class ContentType(ParameterizedHeaderValue):
    token_type = 'content-type'
    as_ew_allowed = False
    maintype = 'text'
    subtype = 'plain'


class ContentDisposition(ParameterizedHeaderValue):
    token_type = 'content-disposition'
    as_ew_allowed = False
    content_disposition = None


class ContentTransferEncoding(TokenList):
    token_type = 'content-transfer-encoding'
    as_ew_allowed = False
    cte = '7bit'


class HeaderLabel(TokenList):
    token_type = 'header-label'
    as_ew_allowed = False


class MsgID(TokenList):
    token_type = 'msg-id'
    as_ew_allowed = False

    def fold(self, policy):
        # message-id tokens may not be folded.
        return str(self) + policy.linesep


class MessageID(MsgID):
    token_type = 'message-id'


class InvalidMessageID(MessageID):
    token_type = 'invalid-message-id'


class Header(TokenList):
    token_type = 'header'


#
# Terminal classes and instances
#

class Terminal(str):

    as_ew_allowed = True
    ew_combine_allowed = True
    syntactic_break = True

    def __new__(cls, value, token_type):
        self = super().__new__(cls, value)
        self.token_type = token_type
        self.defects = []
        return self

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super().__repr__())

    def pprint(self):
        print(self.__class__.__name__ + '/' + self.token_type)

    @property
    def all_defects(self):
        return list(self.defects)

    def _pp(self, indent=''):
        return ["{}{}/{}({}){}".format(
            indent,
            self.__class__.__name__,
            self.token_type,
            super().__repr__(),
            '' if not self.defects else ' {}'.format(self.defects),
            )]

    def pop_trailing_ws(self):
        # This terminates the recursion.
        return None

    @property
    def comments(self):
        return []

    def __getnewargs__(self):
        return(str(self), self.token_type)


class WhiteSpaceTerminal(Terminal):

    @property
    def value(self):
        return ' '

    def startswith_fws(self):
        return self and self[0] in WSP


class ValueTerminal(Terminal):

    @property
    def value(self):
        return self

    def startswith_fws(self):
        return False


class EWWhiteSpaceTerminal(WhiteSpaceTerminal):

    @property
    def value(self):
        return ''

    def __str__(self):
        return ''


class _InvalidEwError(errors.HeaderParseError):
    """Invalid encoded word found while parsing headers."""


# XXX these need to become classes and used as instances so
# that a program can't change them in a parse tree and screw
# up other parse trees.  Maybe should have  tests for that, too.
DOT = ValueTerminal('.', 'dot')
ListSeparator = ValueTerminal(',', 'list-separator')
ListSeparator.as_ew_allowed = False
ListSeparator.syntactic_break = False
RouteComponentMarker = ValueTerminal('@', 'route-component-marker')

#
# Parser
#

# Parse strings according to RFC822/2047/2822/5322 rules.
#
# This is a stateless parser.  Each get_XXX function accepts a string and
# returns either a Terminal or a TokenList representing the RFC object named
# by the method and a string containing the remaining unparsed characters
# from the input.  Thus a parser method consumes the next syntactic construct
# of a given type and returns a token representing the construct plus the
# unparsed remainder of the input string.
#
# For example, if the first element of a structured header is a 'phrase',
# then:
#
#     phrase, value = get_phrase(value)
#
# returns the complete phrase from the start of the string value, plus any
# characters left in the string after the phrase is removed.

_wsp_splitter = re.compile(r'([{}]+)'.format(''.join(WSP))).split
_non_atom_end_matcher = re.compile(r"[^{}]+".format(
    re.escape(''.join(ATOM_ENDS)))).match
_non_printable_finder = re.compile(r"[\x00-\x20\x7F]").findall
_non_token_end_matcher = re.compile(r"[^{}]+".format(
    re.escape(''.join(TOKEN_ENDS)))).match
_non_attribute_end_matcher = re.compile(r"[^{}]+".format(
    re.escape(''.join(ATTRIBUTE_ENDS)))).match
_non_extended_attribute_end_matcher = re.compile(r"[^{}]+".format(
    re.escape(''.join(EXTENDED_ATTRIBUTE_ENDS)))).match

def _validate_xtext(xtext):
    """If input token contains ASCII non-printables, register a defect."""

    non_printables = _non_printable_finder(xtext)
    if non_printables:
        xtext.defects.append(errors.NonPrintableDefect(non_printables))
    if utils._has_surrogates(xtext):
        xtext.defects.append(errors.UndecodableBytesDefect(
            "Non-ASCII characters found in header token"))

def _get_ptext_to_endchars(value, endchars):
    """Scan printables/quoted-pairs until endchars and return unquoted ptext.

    This function turns a run of qcontent, ccontent-without-comments, or
    dtext-with-quoted-printables into a single string by unquoting any
    quoted printables.  It returns the string, the remaining value, and
    a flag that is True iff there were any quoted printables decoded.

    """
    fragment, *remainder = _wsp_splitter(value, 1)
    vchars = []
    escape = False
    had_qp = False
    for pos in range(len(fragment)):
        if fragment[pos] == '\\':
            if escape:
                escape = False
                had_qp = True
            else:
                escape = True
                continue
        if escape:
            escape = False
        elif fragment[pos] in endchars:
            break
        vchars.append(fragment[pos])
    else:
        pos = pos + 1
    return ''.join(vchars), ''.join([fragment[pos:]] + remainder), had_qp

def get_fws(value):
    """FWS = 1*WSP

    This isn't the RFC definition.  We're using fws to represent tokens where
    folding can be done, but when we are parsing the *un*folding has already
    been done so we don't need to watch out for CRLF.

    """
    newvalue = value.lstrip()
    fws = WhiteSpaceTerminal(value[:len(value)-len(newvalue)], 'fws')
    return fws, newvalue

def get_encoded_word(value):
    """ encoded-word = "=?" charset "?" encoding "?" encoded-text "?="

    """
    ew = EncodedWord()
    if not value.startswith('=?'):
        raise errors.HeaderParseError(
            "expected encoded word but found {}".format(value))
    tok, *remainder = value[2:].split('?=', 1)
    if tok == value[2:]:
        raise errors.HeaderParseError(
            "expected encoded word but found {}".format(value))
    remstr = ''.join(remainder)
    if (len(remstr) > 1 and
        remstr[0] in hexdigits and
        remstr[1] in hexdigits and
        tok.count('?') < 2):
        # The ? after the CTE was followed by an encoded word escape (=XX).
        rest, *remainder = remstr.split('?=', 1)
        tok = tok + '?=' + rest
    if len(tok.split()) > 1:
        ew.defects.append(errors.InvalidHeaderDefect(
            "whitespace inside encoded word"))
    ew.cte = value
    value = ''.join(remainder)
    try:
        text, charset, lang, defects = _ew.decode('=?' + tok + '?=')
    except (ValueError, KeyError):
        raise _InvalidEwError(
            "encoded word format invalid: '{}'".format(ew.cte))
    ew.charset = charset
    ew.lang = lang
    ew.defects.extend(defects)
    while text:
        if text[0] in WSP:
            token, text = get_fws(text)
            ew.append(token)
            continue
        chars, *remainder = _wsp_splitter(text, 1)
        vtext = ValueTerminal(chars, 'vtext')
        _validate_xtext(vtext)
        ew.append(vtext)
        text = ''.join(remainder)
    # Encoded words should be followed by a WS
    if value and value[0] not in WSP:
        ew.defects.append(errors.InvalidHeaderDefect(
            "missing trailing whitespace after encoded-word"))
    return ew, value

def get_unstructured(value):
    """unstructured = (*([FWS] vchar) *WSP) / obs-unstruct
       obs-unstruct = *((*LF *CR *(obs-utext) *LF *CR)) / FWS)
       obs-utext = %d0 / obs-NO-WS-CTL / LF / CR

       obs-NO-WS-CTL is control characters except WSP/CR/LF.

    So, basically, we have printable runs, plus control characters or nulls in
    the obsolete syntax, separated by whitespace.  Since RFC 2047 uses the
    obsolete syntax in its specification, but requires whitespace on either
    side of the encoded words, I can see no reason to need to separate the
    non-printable-non-whitespace from the printable runs if they occur, so we
    parse this into xtext tokens separated by WSP tokens.

    Because an 'unstructured' value must by definition constitute the entire
    value, this 'get' routine does not return a remaining value, only the
    parsed TokenList.

    """
    # XXX: but what about bare CR and LF?  They might signal the start or
    # end of an encoded word.  YAGNI for now, since our current parsers
    # will never send us strings with bare CR or LF.

    unstructured = UnstructuredTokenList()
    while value:
        if value[0] in WSP:
            token, value = get_fws(value)
            unstructured.append(token)
            continue
        valid_ew = True
        if value.startswith('=?'):
            try:
                token, value = get_encoded_word(value)
            except _InvalidEwError:
                valid_ew = False
            except errors.HeaderParseError:
                # XXX: Need to figure out how to register defects when
                # appropriate here.
                pass
            else:
                have_ws = True
                if len(unstructured) > 0:
                    if unstructured[-1].token_type != 'fws':
                        unstructured.defects.append(errors.InvalidHeaderDefect(
                            "missing whitespace before encoded word"))
                        have_ws = False
                if have_ws and len(unstructured) > 1:
                    if unstructured[-2].token_type == 'encoded-word':
                        unstructured[-1] = EWWhiteSpaceTerminal(
                            unstructured[-1], 'fws')
                unstructured.append(token)
                continue
        tok, *remainder = _wsp_splitter(value, 1)
        # Split in the middle of an atom if there is a rfc2047 encoded word
        # which does not have WSP on both sides. The defect will be registered
        # the next time through the loop.
        # This needs to only be performed when the encoded word is valid;
        # otherwise, performing it on an invalid encoded word can cause
        # the parser to go in an infinite loop.
        if valid_ew and rfc2047_matcher.search(tok):
            tok, *remainder = value.partition('=?')
        vtext = ValueTerminal(tok, 'vtext')
        _validate_xtext(vtext)
        unstructured.append(vtext)
        value = ''.join(remainder)
    return unstructured

def get_qp_ctext(value):
    r"""ctext = <printable ascii except \ ( )>

    This is not the RFC ctext, since we are handling nested comments in comment
    and unquoting quoted-pairs here.  We allow anything except the '()'
    characters, but if we find any ASCII other than the RFC defined printable
    ASCII, a NonPrintableDefect is added to the token's defects list.  Since
    quoted pairs are converted to their unquoted values, what is returned is
    a 'ptext' token.  In this case it is a WhiteSpaceTerminal, so it's value
    is ' '.

    """
    ptext, value, _ = _get_ptext_to_endchars(value, '()')
    ptext = WhiteSpaceTerminal(ptext, 'ptext')
    _validate_xtext(ptext)
    return ptext, value

def get_qcontent(value):
    """qcontent = qtext / quoted-pair

    We allow anything except the DQUOTE character, but if we find any ASCII
    other than the RFC defined printable ASCII, a NonPrintableDefect is
    added to the token's defects list.  Any quoted pairs are converted to their
    unquoted values, so what is returned is a 'ptext' token.  In this case it
    is a ValueTerminal.

    """
    ptext, value, _ = _get_ptext_to_endchars(value, '"')
    ptext = ValueTerminal(ptext, 'ptext')
    _validate_xtext(ptext)
    return ptext, value

def get_atext(value):
    """atext = <matches _atext_matcher>

    We allow any non-ATOM_ENDS in atext, but add an InvalidATextDefect to
    the token's defects list if we find non-atext characters.
    """
    m = _non_atom_end_matcher(value)
    if not m:
        raise errors.HeaderParseError(
            "expected atext but found '{}'".format(value))
    atext = m.group()
    value = value[len(atext):]
    atext = ValueTerminal(atext, 'atext')
    _validate_xtext(atext)
    return atext, value

def get_bare_quoted_string(value):
    """bare-quoted-string = DQUOTE *([FWS] qcontent) [FWS] DQUOTE

    A quoted-string without the leading or trailing white space.  Its
    value is the text between the quote marks, with whitespace
    preserved and quoted pairs decoded.
    """
    if value[0] != '"':
        raise errors.HeaderParseError(
            "expected '\"' but found '{}'".format(value))
    bare_quoted_string = BareQuotedString()
    value = value[1:]
    if value and value[0] == '"':
        token, value = get_qcontent(value)
        bare_quoted_string.append(token)
    while value and value[0] != '"':
        if value[0] in WSP:
            token, value = get_fws(value)
        elif value[:2] == '=?':
            valid_ew = False
            try:
                token, value = get_encoded_word(value)
                bare_quoted_string.defects.append(errors.InvalidHeaderDefect(
                    "encoded word inside quoted string"))
                valid_ew = True
            except errors.HeaderParseError:
                token, value = get_qcontent(value)
            # Collapse the whitespace between two encoded words that occur in a
            # bare-quoted-string.
            if valid_ew and len(bare_quoted_string) > 1:
                if (bare_quoted_string[-1].token_type == 'fws' and
                        bare_quoted_string[-2].token_type == 'encoded-word'):
                    bare_quoted_string[-1] = EWWhiteSpaceTerminal(
                        bare_quoted_string[-1], 'fws')
        else:
            token, value = get_qcontent(value)
        bare_quoted_string.append(token)
    if not value:
        bare_quoted_string.defects.append(errors.InvalidHeaderDefect(
            "end of header inside quoted string"))
        return bare_quoted_string, value
    return bare_quoted_string, value[1:]

def get_comment(value):
    """comment = "(" *([FWS] ccontent) [FWS] ")"
       ccontent = ctext / quoted-pair / comment

    We handle nested comments here, and quoted-pair in our qp-ctext routine.
    """
    if value and value[0] != '(':
        raise errors.HeaderParseError(
            "expected '(' but found '{}'".format(value))
    comment = Comment()
    value = value[1:]
    while value and value[0] != ")":
        if value[0] in WSP:
            token, value = get_fws(value)
        elif value[0] == '(':
            token, value = get_comment(value)
        else:
            token, value = get_qp_ctext(value)
        comment.append(token)
    if not value:
        comment.defects.append(errors.InvalidHeaderDefect(
            "end of header inside comment"))
        return comment, value
    return comment, value[1:]

def get_cfws(value):
    """CFWS = (1*([FWS] comment) [FWS]) / FWS

    """
    cfws = CFWSList()
    while value and value[0] in CFWS_LEADER:
        if value[0] in WSP:
            token, value = get_fws(value)
        else:
            token, value = get_comment(value)
        cfws.append(token)
    return cfws, value

def get_quoted_string(value):
    """quoted-string = [CFWS] <bare-quoted-string> [CFWS]

    'bare-quoted-string' is an intermediate class defined by this
    parser and not by the RFC grammar.  It is the quoted string
    without any attached CFWS.
    """
    quoted_string = QuotedString()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        quoted_string.append(token)
    token, value = get_bare_quoted_string(value)
    quoted_string.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        quoted_string.append(token)
    return quoted_string, value

def get_atom(value):
    """atom = [CFWS] 1*atext [CFWS]

    An atom could be an rfc2047 encoded word.
    """
    atom = Atom()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        atom.append(token)
    if value and value[0] in ATOM_ENDS:
        raise errors.HeaderParseError(
            "expected atom but found '{}'".format(value))
    if value.startswith('=?'):
        try:
            token, value = get_encoded_word(value)
        except errors.HeaderParseError:
            # XXX: need to figure out how to register defects when
            # appropriate here.
            token, value = get_atext(value)
    else:
        token, value = get_atext(value)
    atom.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        atom.append(token)
    return atom, value

def get_dot_atom_text(value):
    """ dot-text = 1*atext *("." 1*atext)

    """
    dot_atom_text = DotAtomText()
    if not value or value[0] in ATOM_ENDS:
        raise errors.HeaderParseError("expected atom at a start of "
            "dot-atom-text but found '{}'".format(value))
    while value and value[0] not in ATOM_ENDS:
        token, value = get_atext(value)
        dot_atom_text.append(token)
        if value and value[0] == '.':
            dot_atom_text.append(DOT)
            value = value[1:]
    if dot_atom_text[-1] is DOT:
        raise errors.HeaderParseError("expected atom at end of dot-atom-text "
            "but found '{}'".format('.'+value))
    return dot_atom_text, value

def get_dot_atom(value):
    """ dot-atom = [CFWS] dot-atom-text [CFWS]

    Any place we can have a dot atom, we could instead have an rfc2047 encoded
    word.
    """
    dot_atom = DotAtom()
    if value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        dot_atom.append(token)
    if value.startswith('=?'):
        try:
            token, value = get_encoded_word(value)
        except errors.HeaderParseError:
            # XXX: need to figure out how to register defects when
            # appropriate here.
            token, value = get_dot_atom_text(value)
    else:
        token, value = get_dot_atom_text(value)
    dot_atom.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        dot_atom.append(token)
    return dot_atom, value

def get_word(value):
    """word = atom / quoted-string

    Either atom or quoted-string may start with CFWS.  We have to peel off this
    CFWS first to determine which type of word to parse.  Afterward we splice
    the leading CFWS, if any, into the parsed sub-token.

    If neither an atom or a quoted-string is found before the next special, a
    HeaderParseError is raised.

    The token returned is either an Atom or a QuotedString, as appropriate.
    This means the 'word' level of the formal grammar is not represented in the
    parse tree; this is because having that extra layer when manipulating the
    parse tree is more confusing than it is helpful.

    """
    if value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
    else:
        leader = None
    if not value:
        raise errors.HeaderParseError(
            "Expected 'atom' or 'quoted-string' but found nothing.")
    if value[0]=='"':
        token, value = get_quoted_string(value)
    elif value[0] in SPECIALS:
        raise errors.HeaderParseError("Expected 'atom' or 'quoted-string' "
                                      "but found '{}'".format(value))
    else:
        token, value = get_atom(value)
    if leader is not None:
        token[:0] = [leader]
    return token, value

def get_phrase(value):
    """ phrase = 1*word / obs-phrase
        obs-phrase = word *(word / "." / CFWS)

    This means a phrase can be a sequence of words, periods, and CFWS in any
    order as long as it starts with at least one word.  If anything other than
    words is detected, an ObsoleteHeaderDefect is added to the token's defect
    list.  We also accept a phrase that starts with CFWS followed by a dot;
    this is registered as an InvalidHeaderDefect, since it is not supported by
    even the obsolete grammar.

    """
    phrase = Phrase()
    try:
        token, value = get_word(value)
        phrase.append(token)
    except errors.HeaderParseError:
        phrase.defects.append(errors.InvalidHeaderDefect(
            "phrase does not start with word"))
    while value and value[0] not in PHRASE_ENDS:
        if value[0]=='.':
            phrase.append(DOT)
            phrase.defects.append(errors.ObsoleteHeaderDefect(
                "period in 'phrase'"))
            value = value[1:]
        else:
            try:
                token, value = get_word(value)
            except errors.HeaderParseError:
                if value[0] in CFWS_LEADER:
                    token, value = get_cfws(value)
                    phrase.defects.append(errors.ObsoleteHeaderDefect(
                        "comment found without atom"))
                else:
                    raise
            phrase.append(token)
    return phrase, value

def get_local_part(value):
    """ local-part = dot-atom / quoted-string / obs-local-part

    """
    local_part = LocalPart()
    leader = None
    if value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
    if not value:
        raise errors.HeaderParseError(
            "expected local-part but found '{}'".format(value))
    try:
        token, value = get_dot_atom(value)
    except errors.HeaderParseError:
        try:
            token, value = get_word(value)
        except errors.HeaderParseError:
            if value[0] != '\\' and value[0] in PHRASE_ENDS:
                raise
            token = TokenList()
    if leader is not None:
        token[:0] = [leader]
    local_part.append(token)
    if value and (value[0]=='\\' or value[0] not in PHRASE_ENDS):
        obs_local_part, value = get_obs_local_part(str(local_part) + value)
        if obs_local_part.token_type == 'invalid-obs-local-part':
            local_part.defects.append(errors.InvalidHeaderDefect(
                "local-part is not dot-atom, quoted-string, or obs-local-part"))
        else:
            local_part.defects.append(errors.ObsoleteHeaderDefect(
                "local-part is not a dot-atom (contains CFWS)"))
        local_part[0] = obs_local_part
    try:
        local_part.value.encode('ascii')
    except UnicodeEncodeError:
        local_part.defects.append(errors.NonASCIILocalPartDefect(
                "local-part contains non-ASCII characters)"))
    return local_part, value

def get_obs_local_part(value):
    """ obs-local-part = word *("." word)
    """
    obs_local_part = ObsLocalPart()
    last_non_ws_was_dot = False
    while value and (value[0]=='\\' or value[0] not in PHRASE_ENDS):
        if value[0] == '.':
            if last_non_ws_was_dot:
                obs_local_part.defects.append(errors.InvalidHeaderDefect(
                    "invalid repeated '.'"))
            obs_local_part.append(DOT)
            last_non_ws_was_dot = True
            value = value[1:]
            continue
        elif value[0]=='\\':
            obs_local_part.append(ValueTerminal(value[0],
                                                'misplaced-special'))
            value = value[1:]
            obs_local_part.defects.append(errors.InvalidHeaderDefect(
                "'\\' character outside of quoted-string/ccontent"))
            last_non_ws_was_dot = False
            continue
        if obs_local_part and obs_local_part[-1].token_type != 'dot':
            obs_local_part.defects.append(errors.InvalidHeaderDefect(
                "missing '.' between words"))
        try:
            token, value = get_word(value)
            last_non_ws_was_dot = False
        except errors.HeaderParseError:
            if value[0] not in CFWS_LEADER:
                raise
            token, value = get_cfws(value)
        obs_local_part.append(token)
    if (obs_local_part[0].token_type == 'dot' or
            obs_local_part[0].token_type=='cfws' and
            obs_local_part[1].token_type=='dot'):
        obs_local_part.defects.append(errors.InvalidHeaderDefect(
            "Invalid leading '.' in local part"))
    if (obs_local_part[-1].token_type == 'dot' or
            obs_local_part[-1].token_type=='cfws' and
            obs_local_part[-2].token_type=='dot'):
        obs_local_part.defects.append(errors.InvalidHeaderDefect(
            "Invalid trailing '.' in local part"))
    if obs_local_part.defects:
        obs_local_part.token_type = 'invalid-obs-local-part'
    return obs_local_part, value

def get_dtext(value):
    r""" dtext = <printable ascii except \ [ ]> / obs-dtext
        obs-dtext = obs-NO-WS-CTL / quoted-pair

    We allow anything except the excluded characters, but if we find any
    ASCII other than the RFC defined printable ASCII, a NonPrintableDefect is
    added to the token's defects list.  Quoted pairs are converted to their
    unquoted values, so what is returned is a ptext token, in this case a
    ValueTerminal.  If there were quoted-printables, an ObsoleteHeaderDefect is
    added to the returned token's defect list.

    """
    ptext, value, had_qp = _get_ptext_to_endchars(value, '[]')
    ptext = ValueTerminal(ptext, 'ptext')
    if had_qp:
        ptext.defects.append(errors.ObsoleteHeaderDefect(
            "quoted printable found in domain-literal"))
    _validate_xtext(ptext)
    return ptext, value

def _check_for_early_dl_end(value, domain_literal):
    if value:
        return False
    domain_literal.append(errors.InvalidHeaderDefect(
        "end of input inside domain-literal"))
    domain_literal.append(ValueTerminal(']', 'domain-literal-end'))
    return True

def get_domain_literal(value):
    """ domain-literal = [CFWS] "[" *([FWS] dtext) [FWS] "]" [CFWS]

    """
    domain_literal = DomainLiteral()
    if value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        domain_literal.append(token)
    if not value:
        raise errors.HeaderParseError("expected domain-literal")
    if value[0] != '[':
        raise errors.HeaderParseError("expected '[' at start of domain-literal "
                "but found '{}'".format(value))
    value = value[1:]
    if _check_for_early_dl_end(value, domain_literal):
        return domain_literal, value
    domain_literal.append(ValueTerminal('[', 'domain-literal-start'))
    if value[0] in WSP:
        token, value = get_fws(value)
        domain_literal.append(token)
    token, value = get_dtext(value)
    domain_literal.append(token)
    if _check_for_early_dl_end(value, domain_literal):
        return domain_literal, value
    if value[0] in WSP:
        token, value = get_fws(value)
        domain_literal.append(token)
    if _check_for_early_dl_end(value, domain_literal):
        return domain_literal, value
    if value[0] != ']':
        raise errors.HeaderParseError("expected ']' at end of domain-literal "
                "but found '{}'".format(value))
    domain_literal.append(ValueTerminal(']', 'domain-literal-end'))
    value = value[1:]
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        domain_literal.append(token)
    return domain_literal, value

def get_domain(value):
    """ domain = dot-atom / domain-literal / obs-domain
        obs-domain = atom *("." atom))

    """
    domain = Domain()
    leader = None
    if value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
    if not value:
        raise errors.HeaderParseError(
            "expected domain but found '{}'".format(value))
    if value[0] == '[':
        token, value = get_domain_literal(value)
        if leader is not None:
            token[:0] = [leader]
        domain.append(token)
        return domain, value
    try:
        token, value = get_dot_atom(value)
    except errors.HeaderParseError:
        token, value = get_atom(value)
    if value and value[0] == '@':
        raise errors.HeaderParseError('Invalid Domain')
    if leader is not None:
        token[:0] = [leader]
    domain.append(token)
    if value and value[0] == '.':
        domain.defects.append(errors.ObsoleteHeaderDefect(
            "domain is not a dot-atom (contains CFWS)"))
        if domain[0].token_type == 'dot-atom':
            domain[:] = domain[0]
        while value and value[0] == '.':
            domain.append(DOT)
            token, value = get_atom(value[1:])
            domain.append(token)
    return domain, value

def get_addr_spec(value):
    """ addr-spec = local-part "@" domain

    """
    addr_spec = AddrSpec()
    token, value = get_local_part(value)
    addr_spec.append(token)
    if not value or value[0] != '@':
        addr_spec.defects.append(errors.InvalidHeaderDefect(
            "addr-spec local part with no domain"))
        return addr_spec, value
    addr_spec.append(ValueTerminal('@', 'address-at-symbol'))
    token, value = get_domain(value[1:])
    addr_spec.append(token)
    return addr_spec, value

def get_obs_route(value):
    """ obs-route = obs-domain-list ":"
        obs-domain-list = *(CFWS / ",") "@" domain *("," [CFWS] ["@" domain])

        Returns an obs-route token with the appropriate sub-tokens (that is,
        there is no obs-domain-list in the parse tree).
    """
    obs_route = ObsRoute()
    while value and (value[0]==',' or value[0] in CFWS_LEADER):
        if value[0] in CFWS_LEADER:
            token, value = get_cfws(value)
            obs_route.append(token)
        elif value[0] == ',':
            obs_route.append(ListSeparator)
            value = value[1:]
    if not value or value[0] != '@':
        raise errors.HeaderParseError(
            "expected obs-route domain but found '{}'".format(value))
    obs_route.append(RouteComponentMarker)
    token, value = get_domain(value[1:])
    obs_route.append(token)
    while value and value[0]==',':
        obs_route.append(ListSeparator)
        value = value[1:]
        if not value:
            break
        if value[0] in CFWS_LEADER:
            token, value = get_cfws(value)
            obs_route.append(token)
        if value[0] == '@':
            obs_route.append(RouteComponentMarker)
            token, value = get_domain(value[1:])
            obs_route.append(token)
    if not value:
        raise errors.HeaderParseError("end of header while parsing obs-route")
    if value[0] != ':':
        raise errors.HeaderParseError( "expected ':' marking end of "
            "obs-route but found '{}'".format(value))
    obs_route.append(ValueTerminal(':', 'end-of-obs-route-marker'))
    return obs_route, value[1:]

def get_angle_addr(value):
    """ angle-addr = [CFWS] "<" addr-spec ">" [CFWS] / obs-angle-addr
        obs-angle-addr = [CFWS] "<" obs-route addr-spec ">" [CFWS]

    """
    angle_addr = AngleAddr()
    if value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        angle_addr.append(token)
    if not value or value[0] != '<':
        raise errors.HeaderParseError(
            "expected angle-addr but found '{}'".format(value))
    angle_addr.append(ValueTerminal('<', 'angle-addr-start'))
    value = value[1:]
    # Although it is not legal per RFC5322, SMTP uses '<>' in certain
    # circumstances.
    if value[0] == '>':
        angle_addr.append(ValueTerminal('>', 'angle-addr-end'))
        angle_addr.defects.append(errors.InvalidHeaderDefect(
            "null addr-spec in angle-addr"))
        value = value[1:]
        return angle_addr, value
    try:
        token, value = get_addr_spec(value)
    except errors.HeaderParseError:
        try:
            token, value = get_obs_route(value)
            angle_addr.defects.append(errors.ObsoleteHeaderDefect(
                "obsolete route specification in angle-addr"))
        except errors.HeaderParseError:
            raise errors.HeaderParseError(
                "expected addr-spec or obs-route but found '{}'".format(value))
        angle_addr.append(token)
        token, value = get_addr_spec(value)
    angle_addr.append(token)
    if value and value[0] == '>':
        value = value[1:]
    else:
        angle_addr.defects.append(errors.InvalidHeaderDefect(
            "missing trailing '>' on angle-addr"))
    angle_addr.append(ValueTerminal('>', 'angle-addr-end'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        angle_addr.append(token)
    return angle_addr, value

def get_display_name(value):
    """ display-name = phrase

    Because this is simply a name-rule, we don't return a display-name
    token containing a phrase, but rather a display-name token with
    the content of the phrase.

    """
    display_name = DisplayName()
    token, value = get_phrase(value)
    display_name.extend(token[:])
    display_name.defects = token.defects[:]
    return display_name, value


def get_name_addr(value):
    """ name-addr = [display-name] angle-addr

    """
    name_addr = NameAddr()
    # Both the optional display name and the angle-addr can start with cfws.
    leader = None
    if value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
        if not value:
            raise errors.HeaderParseError(
                "expected name-addr but found '{}'".format(leader))
    if value[0] != '<':
        if value[0] in PHRASE_ENDS:
            raise errors.HeaderParseError(
                "expected name-addr but found '{}'".format(value))
        token, value = get_display_name(value)
        if not value:
            raise errors.HeaderParseError(
                "expected name-addr but found '{}'".format(token))
        if leader is not None:
            token[0][:0] = [leader]
            leader = None
        name_addr.append(token)
    token, value = get_angle_addr(value)
    if leader is not None:
        token[:0] = [leader]
    name_addr.append(token)
    return name_addr, value

def get_mailbox(value):
    """ mailbox = name-addr / addr-spec

    """
    # The only way to figure out if we are dealing with a name-addr or an
    # addr-spec is to try parsing each one.
    mailbox = Mailbox()
    try:
        token, value = get_name_addr(value)
    except errors.HeaderParseError:
        try:
            token, value = get_addr_spec(value)
        except errors.HeaderParseError:
            raise errors.HeaderParseError(
                "expected mailbox but found '{}'".format(value))
    if any(isinstance(x, errors.InvalidHeaderDefect)
                       for x in token.all_defects):
        mailbox.token_type = 'invalid-mailbox'
    mailbox.append(token)
    return mailbox, value

def get_invalid_mailbox(value, endchars):
    """ Read everything up to one of the chars in endchars.

    This is outside the formal grammar.  The InvalidMailbox TokenList that is
    returned acts like a Mailbox, but the data attributes are None.

    """
    invalid_mailbox = InvalidMailbox()
    while value and value[0] not in endchars:
        if value[0] in PHRASE_ENDS:
            invalid_mailbox.append(ValueTerminal(value[0],
                                                 'misplaced-special'))
            value = value[1:]
        else:
            token, value = get_phrase(value)
            invalid_mailbox.append(token)
    return invalid_mailbox, value

def get_mailbox_list(value):
    """ mailbox-list = (mailbox *("," mailbox)) / obs-mbox-list
        obs-mbox-list = *([CFWS] ",") mailbox *("," [mailbox / CFWS])

    For this routine we go outside the formal grammar in order to improve error
    handling.  We recognize the end of the mailbox list only at the end of the
    value or at a ';' (the group terminator).  This is so that we can turn
    invalid mailboxes into InvalidMailbox tokens and continue parsing any
    remaining valid mailboxes.  We also allow all mailbox entries to be null,
    and this condition is handled appropriately at a higher level.

    """
    mailbox_list = MailboxList()
    while value and value[0] != ';':
        try:
            token, value = get_mailbox(value)
            mailbox_list.append(token)
        except errors.HeaderParseError:
            leader = None
            if value[0] in CFWS_LEADER:
                leader, value = get_cfws(value)
                if not value or value[0] in ',;':
                    mailbox_list.append(leader)
                    mailbox_list.defects.append(errors.ObsoleteHeaderDefect(
                        "empty element in mailbox-list"))
                else:
                    token, value = get_invalid_mailbox(value, ',;')
                    if leader is not None:
                        token[:0] = [leader]
                    mailbox_list.append(token)
                    mailbox_list.defects.append(errors.InvalidHeaderDefect(
                        "invalid mailbox in mailbox-list"))
            elif value[0] == ',':
                mailbox_list.defects.append(errors.ObsoleteHeaderDefect(
                    "empty element in mailbox-list"))
            else:
                token, value = get_invalid_mailbox(value, ',;')
                if leader is not None:
                    token[:0] = [leader]
                mailbox_list.append(token)
                mailbox_list.defects.append(errors.InvalidHeaderDefect(
                    "invalid mailbox in mailbox-list"))
        if value and value[0] not in ',;':
            # Crap after mailbox; treat it as an invalid mailbox.
            # The mailbox info will still be available.
            mailbox = mailbox_list[-1]
            mailbox.token_type = 'invalid-mailbox'
            token, value = get_invalid_mailbox(value, ',;')
            mailbox.extend(token)
            mailbox_list.defects.append(errors.InvalidHeaderDefect(
                "invalid mailbox in mailbox-list"))
        if value and value[0] == ',':
            mailbox_list.append(ListSeparator)
            value = value[1:]
    return mailbox_list, value


def get_group_list(value):
    """ group-list = mailbox-list / CFWS / obs-group-list
        obs-group-list = 1*([CFWS] ",") [CFWS]

    """
    group_list = GroupList()
    if not value:
        group_list.defects.append(errors.InvalidHeaderDefect(
            "end of header before group-list"))
        return group_list, value
    leader = None
    if value and value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
        if not value:
            # This should never happen in email parsing, since CFWS-only is a
            # legal alternative to group-list in a group, which is the only
            # place group-list appears.
            group_list.defects.append(errors.InvalidHeaderDefect(
                "end of header in group-list"))
            group_list.append(leader)
            return group_list, value
        if value[0] == ';':
            group_list.append(leader)
            return group_list, value
    token, value = get_mailbox_list(value)
    if len(token.all_mailboxes)==0:
        if leader is not None:
            group_list.append(leader)
        group_list.extend(token)
        group_list.defects.append(errors.ObsoleteHeaderDefect(
            "group-list with empty entries"))
        return group_list, value
    if leader is not None:
        token[:0] = [leader]
    group_list.append(token)
    return group_list, value

def get_group(value):
    """ group = display-name ":" [group-list] ";" [CFWS]

    """
    group = Group()
    token, value = get_display_name(value)
    if not value or value[0] != ':':
        raise errors.HeaderParseError("expected ':' at end of group "
            "display name but found '{}'".format(value))
    group.append(token)
    group.append(ValueTerminal(':', 'group-display-name-terminator'))
    value = value[1:]
    if value and value[0] == ';':
        group.append(ValueTerminal(';', 'group-terminator'))
        return group, value[1:]
    token, value = get_group_list(value)
    group.append(token)
    if not value:
        group.defects.append(errors.InvalidHeaderDefect(
            "end of header in group"))
    elif value[0] != ';':
        raise errors.HeaderParseError(
            "expected ';' at end of group but found {}".format(value))
    group.append(ValueTerminal(';', 'group-terminator'))
    value = value[1:]
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        group.append(token)
    return group, value

def get_address(value):
    """ address = mailbox / group

    Note that counter-intuitively, an address can be either a single address or
    a list of addresses (a group).  This is why the returned Address object has
    a 'mailboxes' attribute which treats a single address as a list of length
    one.  When you need to differentiate between to two cases, extract the single
    element, which is either a mailbox or a group token.

    """
    # The formal grammar isn't very helpful when parsing an address.  mailbox
    # and group, especially when allowing for obsolete forms, start off very
    # similarly.  It is only when you reach one of @, <, or : that you know
    # what you've got.  So, we try each one in turn, starting with the more
    # likely of the two.  We could perhaps make this more efficient by looking
    # for a phrase and then branching based on the next character, but that
    # would be a premature optimization.
    address = Address()
    try:
        token, value = get_group(value)
    except errors.HeaderParseError:
        try:
            token, value = get_mailbox(value)
        except errors.HeaderParseError:
            raise errors.HeaderParseError(
                "expected address but found '{}'".format(value))
    address.append(token)
    return address, value

def get_address_list(value):
    """ address_list = (address *("," address)) / obs-addr-list
        obs-addr-list = *([CFWS] ",") address *("," [address / CFWS])

    We depart from the formal grammar here by continuing to parse until the end
    of the input, assuming the input to be entirely composed of an
    address-list.  This is always true in email parsing, and allows us
    to skip invalid addresses to parse additional valid ones.

    """
    address_list = AddressList()
    while value:
        try:
            token, value = get_address(value)
            address_list.append(token)
        except errors.HeaderParseError as err:
            leader = None
            if value[0] in CFWS_LEADER:
                leader, value = get_cfws(value)
                if not value or value[0] == ',':
                    address_list.append(leader)
                    address_list.defects.append(errors.ObsoleteHeaderDefect(
                        "address-list entry with no content"))
                else:
                    token, value = get_invalid_mailbox(value, ',')
                    if leader is not None:
                        token[:0] = [leader]
                    address_list.append(Address([token]))
                    address_list.defects.append(errors.InvalidHeaderDefect(
                        "invalid address in address-list"))
            elif value[0] == ',':
                address_list.defects.append(errors.ObsoleteHeaderDefect(
                    "empty element in address-list"))
            else:
                token, value = get_invalid_mailbox(value, ',')
                if leader is not None:
                    token[:0] = [leader]
                address_list.append(Address([token]))
                address_list.defects.append(errors.InvalidHeaderDefect(
                    "invalid address in address-list"))
        if value and value[0] != ',':
            # Crap after address; treat it as an invalid mailbox.
            # The mailbox info will still be available.
            mailbox = address_list[-1][0]
            mailbox.token_type = 'invalid-mailbox'
            token, value = get_invalid_mailbox(value, ',')
            mailbox.extend(token)
            address_list.defects.append(errors.InvalidHeaderDefect(
                "invalid address in address-list"))
        if value:  # Must be a , at this point.
            address_list.append(ListSeparator)
            value = value[1:]
    return address_list, value


def get_no_fold_literal(value):
    """ no-fold-literal = "[" *dtext "]"
    """
    no_fold_literal = NoFoldLiteral()
    if not value:
        raise errors.HeaderParseError(
            "expected no-fold-literal but found '{}'".format(value))
    if value[0] != '[':
        raise errors.HeaderParseError(
            "expected '[' at the start of no-fold-literal "
            "but found '{}'".format(value))
    no_fold_literal.append(ValueTerminal('[', 'no-fold-literal-start'))
    value = value[1:]
    token, value = get_dtext(value)
    no_fold_literal.append(token)
    if not value or value[0] != ']':
        raise errors.HeaderParseError(
            "expected ']' at the end of no-fold-literal "
            "but found '{}'".format(value))
    no_fold_literal.append(ValueTerminal(']', 'no-fold-literal-end'))
    return no_fold_literal, value[1:]

def get_msg_id(value):
    """msg-id = [CFWS] "<" id-left '@' id-right  ">" [CFWS]
       id-left = dot-atom-text / obs-id-left
       id-right = dot-atom-text / no-fold-literal / obs-id-right
       no-fold-literal = "[" *dtext "]"
    """
    msg_id = MsgID()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        msg_id.append(token)
    if not value or value[0] != '<':
        raise errors.HeaderParseError(
            "expected msg-id but found '{}'".format(value))
    msg_id.append(ValueTerminal('<', 'msg-id-start'))
    value = value[1:]
    # Parse id-left.
    try:
        token, value = get_dot_atom_text(value)
    except errors.HeaderParseError:
        try:
            # obs-id-left is same as local-part of add-spec.
            token, value = get_obs_local_part(value)
            msg_id.defects.append(errors.ObsoleteHeaderDefect(
                "obsolete id-left in msg-id"))
        except errors.HeaderParseError:
            raise errors.HeaderParseError(
                "expected dot-atom-text or obs-id-left"
                " but found '{}'".format(value))
    msg_id.append(token)
    if not value or value[0] != '@':
        msg_id.defects.append(errors.InvalidHeaderDefect(
            "msg-id with no id-right"))
        # Even though there is no id-right, if the local part
        # ends with `>` let's just parse it too and return
        # along with the defect.
        if value and value[0] == '>':
            msg_id.append(ValueTerminal('>', 'msg-id-end'))
            value = value[1:]
        return msg_id, value
    msg_id.append(ValueTerminal('@', 'address-at-symbol'))
    value = value[1:]
    # Parse id-right.
    try:
        token, value = get_dot_atom_text(value)
    except errors.HeaderParseError:
        try:
            token, value = get_no_fold_literal(value)
        except errors.HeaderParseError as e:
            try:
                token, value = get_domain(value)
                msg_id.defects.append(errors.ObsoleteHeaderDefect(
                    "obsolete id-right in msg-id"))
            except errors.HeaderParseError:
                raise errors.HeaderParseError(
                    "expected dot-atom-text, no-fold-literal or obs-id-right"
                    " but found '{}'".format(value))
    msg_id.append(token)
    if value and value[0] == '>':
        value = value[1:]
    else:
        msg_id.defects.append(errors.InvalidHeaderDefect(
            "missing trailing '>' on msg-id"))
    msg_id.append(ValueTerminal('>', 'msg-id-end'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        msg_id.append(token)
    return msg_id, value


def parse_message_id(value):
    """message-id      =   "Message-ID:" msg-id CRLF
    """
    message_id = MessageID()
    try:
        token, value = get_msg_id(value)
        message_id.append(token)
    except errors.HeaderParseError as ex:
        token = get_unstructured(value)
        message_id = InvalidMessageID(token)
        message_id.defects.append(
            errors.InvalidHeaderDefect("Invalid msg-id: {!r}".format(ex)))
    else:
        # Value after parsing a valid msg_id should be None.
        if value:
            message_id.defects.append(errors.InvalidHeaderDefect(
                "Unexpected {!r}".format(value)))

    return message_id

#
# XXX: As I begin to add additional header parsers, I'm realizing we probably
# have two level of parser routines: the get_XXX methods that get a token in
# the grammar, and parse_XXX methods that parse an entire field value.  So
# get_address_list above should really be a parse_ method, as probably should
# be get_unstructured.
#

def parse_mime_version(value):
    """ mime-version = [CFWS] 1*digit [CFWS] "." [CFWS] 1*digit [CFWS]

    """
    # The [CFWS] is implicit in the RFC 2045 BNF.
    # XXX: This routine is a bit verbose, should factor out a get_int method.
    mime_version = MIMEVersion()
    if not value:
        mime_version.defects.append(errors.HeaderMissingRequiredValue(
            "Missing MIME version number (eg: 1.0)"))
        return mime_version
    if value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
        if not value:
            mime_version.defects.append(errors.HeaderMissingRequiredValue(
                "Expected MIME version number but found only CFWS"))
    digits = ''
    while value and value[0] != '.' and value[0] not in CFWS_LEADER:
        digits += value[0]
        value = value[1:]
    if not digits.isdigit():
        mime_version.defects.append(errors.InvalidHeaderDefect(
            "Expected MIME major version number but found {!r}".format(digits)))
        mime_version.append(ValueTerminal(digits, 'xtext'))
    else:
        mime_version.major = int(digits)
        mime_version.append(ValueTerminal(digits, 'digits'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if not value or value[0] != '.':
        if mime_version.major is not None:
            mime_version.defects.append(errors.InvalidHeaderDefect(
                "Incomplete MIME version; found only major number"))
        if value:
            mime_version.append(ValueTerminal(value, 'xtext'))
        return mime_version
    mime_version.append(ValueTerminal('.', 'version-separator'))
    value = value[1:]
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if not value:
        if mime_version.major is not None:
            mime_version.defects.append(errors.InvalidHeaderDefect(
                "Incomplete MIME version; found only major number"))
        return mime_version
    digits = ''
    while value and value[0] not in CFWS_LEADER:
        digits += value[0]
        value = value[1:]
    if not digits.isdigit():
        mime_version.defects.append(errors.InvalidHeaderDefect(
            "Expected MIME minor version number but found {!r}".format(digits)))
        mime_version.append(ValueTerminal(digits, 'xtext'))
    else:
        mime_version.minor = int(digits)
        mime_version.append(ValueTerminal(digits, 'digits'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mime_version.append(token)
    if value:
        mime_version.defects.append(errors.InvalidHeaderDefect(
            "Excess non-CFWS text after MIME version"))
        mime_version.append(ValueTerminal(value, 'xtext'))
    return mime_version

def get_invalid_parameter(value):
    """ Read everything up to the next ';'.

    This is outside the formal grammar.  The InvalidParameter TokenList that is
    returned acts like a Parameter, but the data attributes are None.

    """
    invalid_parameter = InvalidParameter()
    while value and value[0] != ';':
        if value[0] in PHRASE_ENDS:
            invalid_parameter.append(ValueTerminal(value[0],
                                                   'misplaced-special'))
            value = value[1:]
        else:
            token, value = get_phrase(value)
            invalid_parameter.append(token)
    return invalid_parameter, value

def get_ttext(value):
    """ttext = <matches _ttext_matcher>

    We allow any non-TOKEN_ENDS in ttext, but add defects to the token's
    defects list if we find non-ttext characters.  We also register defects for
    *any* non-printables even though the RFC doesn't exclude all of them,
    because we follow the spirit of RFC 5322.

    """
    m = _non_token_end_matcher(value)
    if not m:
        raise errors.HeaderParseError(
            "expected ttext but found '{}'".format(value))
    ttext = m.group()
    value = value[len(ttext):]
    ttext = ValueTerminal(ttext, 'ttext')
    _validate_xtext(ttext)
    return ttext, value

def get_token(value):
    """token = [CFWS] 1*ttext [CFWS]

    The RFC equivalent of ttext is any US-ASCII chars except space, ctls, or
    tspecials.  We also exclude tabs even though the RFC doesn't.

    The RFC implies the CFWS but is not explicit about it in the BNF.

    """
    mtoken = Token()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mtoken.append(token)
    if value and value[0] in TOKEN_ENDS:
        raise errors.HeaderParseError(
            "expected token but found '{}'".format(value))
    token, value = get_ttext(value)
    mtoken.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        mtoken.append(token)
    return mtoken, value

def get_attrtext(value):
    """attrtext = 1*(any non-ATTRIBUTE_ENDS character)

    We allow any non-ATTRIBUTE_ENDS in attrtext, but add defects to the
    token's defects list if we find non-attrtext characters.  We also register
    defects for *any* non-printables even though the RFC doesn't exclude all of
    them, because we follow the spirit of RFC 5322.

    """
    m = _non_attribute_end_matcher(value)
    if not m:
        raise errors.HeaderParseError(
            "expected attrtext but found {!r}".format(value))
    attrtext = m.group()
    value = value[len(attrtext):]
    attrtext = ValueTerminal(attrtext, 'attrtext')
    _validate_xtext(attrtext)
    return attrtext, value

def get_attribute(value):
    """ [CFWS] 1*attrtext [CFWS]

    This version of the BNF makes the CFWS explicit, and as usual we use a
    value terminal for the actual run of characters.  The RFC equivalent of
    attrtext is the token characters, with the subtraction of '*', "'", and '%'.
    We include tab in the excluded set just as we do for token.

    """
    attribute = Attribute()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        attribute.append(token)
    if value and value[0] in ATTRIBUTE_ENDS:
        raise errors.HeaderParseError(
            "expected token but found '{}'".format(value))
    token, value = get_attrtext(value)
    attribute.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        attribute.append(token)
    return attribute, value

def get_extended_attrtext(value):
    """attrtext = 1*(any non-ATTRIBUTE_ENDS character plus '%')

    This is a special parsing routine so that we get a value that
    includes % escapes as a single string (which we decode as a single
    string later).

    """
    m = _non_extended_attribute_end_matcher(value)
    if not m:
        raise errors.HeaderParseError(
            "expected extended attrtext but found {!r}".format(value))
    attrtext = m.group()
    value = value[len(attrtext):]
    attrtext = ValueTerminal(attrtext, 'extended-attrtext')
    _validate_xtext(attrtext)
    return attrtext, value

def get_extended_attribute(value):
    """ [CFWS] 1*extended_attrtext [CFWS]

    This is like the non-extended version except we allow % characters, so that
    we can pick up an encoded value as a single string.

    """
    # XXX: should we have an ExtendedAttribute TokenList?
    attribute = Attribute()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        attribute.append(token)
    if value and value[0] in EXTENDED_ATTRIBUTE_ENDS:
        raise errors.HeaderParseError(
            "expected token but found '{}'".format(value))
    token, value = get_extended_attrtext(value)
    attribute.append(token)
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        attribute.append(token)
    return attribute, value

def get_section(value):
    """ '*' digits

    The formal BNF is more complicated because leading 0s are not allowed.  We
    check for that and add a defect.  We also assume no CFWS is allowed between
    the '*' and the digits, though the RFC is not crystal clear on that.
    The caller should already have dealt with leading CFWS.

    """
    section = Section()
    if not value or value[0] != '*':
        raise errors.HeaderParseError("Expected section but found {}".format(
                                        value))
    section.append(ValueTerminal('*', 'section-marker'))
    value = value[1:]
    if not value or not value[0].isdigit():
        raise errors.HeaderParseError("Expected section number but "
                                      "found {}".format(value))
    digits = ''
    while value and value[0].isdigit():
        digits += value[0]
        value = value[1:]
    if digits[0] == '0' and digits != '0':
        section.defects.append(errors.InvalidHeaderDefect(
                "section number has an invalid leading 0"))
    section.number = int(digits)
    section.append(ValueTerminal(digits, 'digits'))
    return section, value


def get_value(value):
    """ quoted-string / attribute

    """
    v = Value()
    if not value:
        raise errors.HeaderParseError("Expected value but found end of string")
    leader = None
    if value[0] in CFWS_LEADER:
        leader, value = get_cfws(value)
    if not value:
        raise errors.HeaderParseError("Expected value but found "
                                      "only {}".format(leader))
    if value[0] == '"':
        token, value = get_quoted_string(value)
    else:
        token, value = get_extended_attribute(value)
    if leader is not None:
        token[:0] = [leader]
    v.append(token)
    return v, value

def get_parameter(value):
    """ attribute [section] ["*"] [CFWS] "=" value

    The CFWS is implied by the RFC but not made explicit in the BNF.  This
    simplified form of the BNF from the RFC is made to conform with the RFC BNF
    through some extra checks.  We do it this way because it makes both error
    recovery and working with the resulting parse tree easier.
    """
    # It is possible CFWS would also be implicitly allowed between the section
    # and the 'extended-attribute' marker (the '*') , but we've never seen that
    # in the wild and we will therefore ignore the possibility.
    param = Parameter()
    token, value = get_attribute(value)
    param.append(token)
    if not value or value[0] == ';':
        param.defects.append(errors.InvalidHeaderDefect("Parameter contains "
            "name ({}) but no value".format(token)))
        return param, value
    if value[0] == '*':
        try:
            token, value = get_section(value)
            param.sectioned = True
            param.append(token)
        except errors.HeaderParseError:
            pass
        if not value:
            raise errors.HeaderParseError("Incomplete parameter")
        if value[0] == '*':
            param.append(ValueTerminal('*', 'extended-parameter-marker'))
            value = value[1:]
            param.extended = True
    if value[0] != '=':
        raise errors.HeaderParseError("Parameter not followed by '='")
    param.append(ValueTerminal('=', 'parameter-separator'))
    value = value[1:]
    leader = None
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        param.append(token)
    remainder = None
    appendto = param
    if param.extended and value and value[0] == '"':
        # Now for some serious hackery to handle the common invalid case of
        # double quotes around an extended value.  We also accept (with defect)
        # a value marked as encoded that isn't really.
        qstring, remainder = get_quoted_string(value)
        inner_value = qstring.stripped_value
        semi_valid = False
        if param.section_number == 0:
            if inner_value and inner_value[0] == "'":
                semi_valid = True
            else:
                token, rest = get_attrtext(inner_value)
                if rest and rest[0] == "'":
                    semi_valid = True
        else:
            try:
                token, rest = get_extended_attrtext(inner_value)
            except:
                pass
            else:
                if not rest:
                    semi_valid = True
        if semi_valid:
            param.defects.append(errors.InvalidHeaderDefect(
                "Quoted string value for extended parameter is invalid"))
            param.append(qstring)
            for t in qstring:
                if t.token_type == 'bare-quoted-string':
                    t[:] = []
                    appendto = t
                    break
            value = inner_value
        else:
            remainder = None
            param.defects.append(errors.InvalidHeaderDefect(
                "Parameter marked as extended but appears to have a "
                "quoted string value that is non-encoded"))
    if value and value[0] == "'":
        token = None
    else:
        token, value = get_value(value)
    if not param.extended or param.section_number > 0:
        if not value or value[0] != "'":
            appendto.append(token)
            if remainder is not None:
                assert not value, value
                value = remainder
            return param, value
        param.defects.append(errors.InvalidHeaderDefect(
            "Apparent initial-extended-value but attribute "
            "was not marked as extended or was not initial section"))
    if not value:
        # Assume the charset/lang is missing and the token is the value.
        param.defects.append(errors.InvalidHeaderDefect(
            "Missing required charset/lang delimiters"))
        appendto.append(token)
        if remainder is None:
            return param, value
    else:
        if token is not None:
            for t in token:
                if t.token_type == 'extended-attrtext':
                    break
            t.token_type == 'attrtext'
            appendto.append(t)
            param.charset = t.value
        if value[0] != "'":
            raise errors.HeaderParseError("Expected RFC2231 char/lang encoding "
                                          "delimiter, but found {!r}".format(value))
        appendto.append(ValueTerminal("'", 'RFC2231-delimiter'))
        value = value[1:]
        if value and value[0] != "'":
            token, value = get_attrtext(value)
            appendto.append(token)
            param.lang = token.value
            if not value or value[0] != "'":
                raise errors.HeaderParseError("Expected RFC2231 char/lang encoding "
                                  "delimiter, but found {}".format(value))
        appendto.append(ValueTerminal("'", 'RFC2231-delimiter'))
        value = value[1:]
    if remainder is not None:
        # Treat the rest of value as bare quoted string content.
        v = Value()
        while value:
            if value[0] in WSP:
                token, value = get_fws(value)
            elif value[0] == '"':
                token = ValueTerminal('"', 'DQUOTE')
                value = value[1:]
            else:
                token, value = get_qcontent(value)
            v.append(token)
        token = v
    else:
        token, value = get_value(value)
    appendto.append(token)
    if remainder is not None:
        assert not value, value
        value = remainder
    return param, value

def parse_mime_parameters(value):
    """ parameter *( ";" parameter )

    That BNF is meant to indicate this routine should only be called after
    finding and handling the leading ';'.  There is no corresponding rule in
    the formal RFC grammar, but it is more convenient for us for the set of
    parameters to be treated as its own TokenList.

    This is 'parse' routine because it consumes the remaining value, but it
    would never be called to parse a full header.  Instead it is called to
    parse everything after the non-parameter value of a specific MIME header.

    """
    mime_parameters = MimeParameters()
    while value:
        try:
            token, value = get_parameter(value)
            mime_parameters.append(token)
        except errors.HeaderParseError as err:
            leader = None
            if value[0] in CFWS_LEADER:
                leader, value = get_cfws(value)
            if not value:
                mime_parameters.append(leader)
                return mime_parameters
            if value[0] == ';':
                if leader is not None:
                    mime_parameters.append(leader)
                mime_parameters.defects.append(errors.InvalidHeaderDefect(
                    "parameter entry with no content"))
            else:
                token, value = get_invalid_parameter(value)
                if leader:
                    token[:0] = [leader]
                mime_parameters.append(token)
                mime_parameters.defects.append(errors.InvalidHeaderDefect(
                    "invalid parameter {!r}".format(token)))
        if value and value[0] != ';':
            # Junk after the otherwise valid parameter.  Mark it as
            # invalid, but it will have a value.
            param = mime_parameters[-1]
            param.token_type = 'invalid-parameter'
            token, value = get_invalid_parameter(value)
            param.extend(token)
            mime_parameters.defects.append(errors.InvalidHeaderDefect(
                "parameter with invalid trailing text {!r}".format(token)))
        if value:
            # Must be a ';' at this point.
            mime_parameters.append(ValueTerminal(';', 'parameter-separator'))
            value = value[1:]
    return mime_parameters

def _find_mime_parameters(tokenlist, value):
    """Do our best to find the parameters in an invalid MIME header

    """
    while value and value[0] != ';':
        if value[0] in PHRASE_ENDS:
            tokenlist.append(ValueTerminal(value[0], 'misplaced-special'))
            value = value[1:]
        else:
            token, value = get_phrase(value)
            tokenlist.append(token)
    if not value:
        return
    tokenlist.append(ValueTerminal(';', 'parameter-separator'))
    tokenlist.append(parse_mime_parameters(value[1:]))

def parse_content_type_header(value):
    """ maintype "/" subtype *( ";" parameter )

    The maintype and substype are tokens.  Theoretically they could
    be checked against the official IANA list + x-token, but we
    don't do that.
    """
    ctype = ContentType()
    recover = False
    if not value:
        ctype.defects.append(errors.HeaderMissingRequiredValue(
            "Missing content type specification"))
        return ctype
    try:
        token, value = get_token(value)
    except errors.HeaderParseError:
        ctype.defects.append(errors.InvalidHeaderDefect(
            "Expected content maintype but found {!r}".format(value)))
        _find_mime_parameters(ctype, value)
        return ctype
    ctype.append(token)
    # XXX: If we really want to follow the formal grammar we should make
    # mantype and subtype specialized TokenLists here.  Probably not worth it.
    if not value or value[0] != '/':
        ctype.defects.append(errors.InvalidHeaderDefect(
            "Invalid content type"))
        if value:
            _find_mime_parameters(ctype, value)
        return ctype
    ctype.maintype = token.value.strip().lower()
    ctype.append(ValueTerminal('/', 'content-type-separator'))
    value = value[1:]
    try:
        token, value = get_token(value)
    except errors.HeaderParseError:
        ctype.defects.append(errors.InvalidHeaderDefect(
            "Expected content subtype but found {!r}".format(value)))
        _find_mime_parameters(ctype, value)
        return ctype
    ctype.append(token)
    ctype.subtype = token.value.strip().lower()
    if not value:
        return ctype
    if value[0] != ';':
        ctype.defects.append(errors.InvalidHeaderDefect(
            "Only parameters are valid after content type, but "
            "found {!r}".format(value)))
        # The RFC requires that a syntactically invalid content-type be treated
        # as text/plain.  Perhaps we should postel this, but we should probably
        # only do that if we were checking the subtype value against IANA.
        del ctype.maintype, ctype.subtype
        _find_mime_parameters(ctype, value)
        return ctype
    ctype.append(ValueTerminal(';', 'parameter-separator'))
    ctype.append(parse_mime_parameters(value[1:]))
    return ctype

def parse_content_disposition_header(value):
    """ disposition-type *( ";" parameter )

    """
    disp_header = ContentDisposition()
    if not value:
        disp_header.defects.append(errors.HeaderMissingRequiredValue(
            "Missing content disposition"))
        return disp_header
    try:
        token, value = get_token(value)
    except errors.HeaderParseError:
        disp_header.defects.append(errors.InvalidHeaderDefect(
            "Expected content disposition but found {!r}".format(value)))
        _find_mime_parameters(disp_header, value)
        return disp_header
    disp_header.append(token)
    disp_header.content_disposition = token.value.strip().lower()
    if not value:
        return disp_header
    if value[0] != ';':
        disp_header.defects.append(errors.InvalidHeaderDefect(
            "Only parameters are valid after content disposition, but "
            "found {!r}".format(value)))
        _find_mime_parameters(disp_header, value)
        return disp_header
    disp_header.append(ValueTerminal(';', 'parameter-separator'))
    disp_header.append(parse_mime_parameters(value[1:]))
    return disp_header

def parse_content_transfer_encoding_header(value):
    """ mechanism

    """
    # We should probably validate the values, since the list is fixed.
    cte_header = ContentTransferEncoding()
    if not value:
        cte_header.defects.append(errors.HeaderMissingRequiredValue(
            "Missing content transfer encoding"))
        return cte_header
    try:
        token, value = get_token(value)
    except errors.HeaderParseError:
        cte_header.defects.append(errors.InvalidHeaderDefect(
            "Expected content transfer encoding but found {!r}".format(value)))
    else:
        cte_header.append(token)
        cte_header.cte = token.value.strip().lower()
    if not value:
        return cte_header
    while value:
        cte_header.defects.append(errors.InvalidHeaderDefect(
            "Extra text after content transfer encoding"))
        if value[0] in PHRASE_ENDS:
            cte_header.append(ValueTerminal(value[0], 'misplaced-special'))
            value = value[1:]
        else:
            token, value = get_phrase(value)
            cte_header.append(token)
    return cte_header


#
# Header folding
#
# Header folding is complex, with lots of rules and corner cases.  The
# following code does its best to obey the rules and handle the corner
# cases, but you can be sure there are few bugs:)
#
# This folder generally canonicalizes as it goes, preferring the stringified
# version of each token.  The tokens contain information that supports the
# folder, including which tokens can be encoded in which ways.
#
# Folded text is accumulated in a simple list of strings ('lines'), each
# one of which should be less than policy.max_line_length ('maxlen').
#

def _steal_trailing_WSP_if_exists(lines):
    wsp = ''
    if lines and lines[-1] and lines[-1][-1] in WSP:
        wsp = lines[-1][-1]
        lines[-1] = lines[-1][:-1]
    return wsp

def _refold_parse_tree(parse_tree, *, policy):
    """Return string of contents of parse_tree folded according to RFC rules.

    """
    # max_line_length 0/None means no limit, ie: infinitely long.
    maxlen = policy.max_line_length or sys.maxsize
    encoding = 'utf-8' if policy.utf8 else 'us-ascii'
    lines = ['']
    last_ew = None
    wrap_as_ew_blocked = 0
    want_encoding = False
    end_ew_not_allowed = Terminal('', 'wrap_as_ew_blocked')
    parts = list(parse_tree)
    while parts:
        part = parts.pop(0)
        if part is end_ew_not_allowed:
            wrap_as_ew_blocked -= 1
            continue
        tstr = str(part)
        if not want_encoding:
            if part.token_type == 'ptext':
                # Encode if tstr contains special characters.
                want_encoding = not SPECIALSNL.isdisjoint(tstr)
            else:
                # Encode if tstr contains newlines.
                want_encoding = not NLSET.isdisjoint(tstr)
        try:
            tstr.encode(encoding)
            charset = encoding
        except UnicodeEncodeError:
            if any(isinstance(x, errors.UndecodableBytesDefect)
                   for x in part.all_defects):
                charset = 'unknown-8bit'
            else:
                # If policy.utf8 is false this should really be taken from a
                # 'charset' property on the policy.
                charset = 'utf-8'
            want_encoding = True
        if part.token_type == 'mime-parameters':
            # Mime parameter folding (using RFC2231) is extra special.
            _fold_mime_parameters(part, lines, maxlen, encoding)
            continue
        if want_encoding and not wrap_as_ew_blocked:
            if not part.as_ew_allowed:
                want_encoding = False
                last_ew = None
                if part.syntactic_break:
                    encoded_part = part.fold(policy=policy)[:-len(policy.linesep)]
                    if policy.linesep not in encoded_part:
                        # It fits on a single line
                        if len(encoded_part) > maxlen - len(lines[-1]):
                            # But not on this one, so start a new one.
                            newline = _steal_trailing_WSP_if_exists(lines)
                            # XXX what if encoded_part has no leading FWS?
                            lines.append(newline)
                        lines[-1] += encoded_part
                        continue
                # Either this is not a major syntactic break, so we don't
                # want it on a line by itself even if it fits, or it
                # doesn't fit on a line by itself.  Either way, fall through
                # to unpacking the subparts and wrapping them.
            if not hasattr(part, 'encode'):
                # It's not a Terminal, do each piece individually.
                parts = list(part) + parts
                want_encoding = False
                continue
            elif part.as_ew_allowed:
                # It's a terminal, wrap it as an encoded word, possibly
                # combining it with previously encoded words if allowed.
                last_ew = _fold_as_ew(tstr, lines, maxlen, last_ew,
                                      part.ew_combine_allowed, charset)
                want_encoding = False
                continue
            else:
                # It's a terminal which should be kept non-encoded
                # (e.g. a ListSeparator).
                last_ew = None
                want_encoding = False
                # fall through

        if len(tstr) <= maxlen - len(lines[-1]):
            lines[-1] += tstr
            continue
        # This part is too long to fit.  The RFC wants us to break at
        # "major syntactic breaks", so unless we don't consider this
        # to be one, check if it will fit on the next line by itself.
        if (part.syntactic_break and
                len(tstr) + 1 <= maxlen):
            newline = _steal_trailing_WSP_if_exists(lines)
            if newline or part.startswith_fws():
                lines.append(newline + tstr)
                last_ew = None
                continue
        if not hasattr(part, 'encode'):
            # It's not a terminal, try folding the subparts.
            newparts = list(part)
            if part.token_type == 'comment':
                newparts = (
                    [ValueTerminal('(', 'ptext')] +
                    [ValueTerminal(make_parenthesis_pairs(p), 'ptext')
                     if p.token_type == 'ptext' else p
                     for p in newparts] +
                    [ValueTerminal(')', 'ptext')])
            if not part.as_ew_allowed:
                wrap_as_ew_blocked += 1
                newparts.append(end_ew_not_allowed)
            parts = newparts + parts
            continue
        if part.as_ew_allowed and not wrap_as_ew_blocked:
            # It doesn't need CTE encoding, but encode it anyway so we can
            # wrap it.
            parts.insert(0, part)
            want_encoding = True
            continue
        # We can't figure out how to wrap, it, so give up.
        newline = _steal_trailing_WSP_if_exists(lines)
        if newline or part.startswith_fws():
            lines.append(newline + tstr)
        else:
            # We can't fold it onto the next line either...
            lines[-1] += tstr
    return policy.linesep.join(lines) + policy.linesep

def _fold_as_ew(to_encode, lines, maxlen, last_ew, ew_combine_allowed, charset):
    """Fold string to_encode into lines as encoded word, combining if allowed.
    Return the new value for last_ew, or None if ew_combine_allowed is False.

    If there is already an encoded word in the last line of lines (indicated by
    a non-None value for last_ew) and ew_combine_allowed is true, decode the
    existing ew, combine it with to_encode, and re-encode.  Otherwise, encode
    to_encode.  In either case, split to_encode as necessary so that the
    encoded segments fit within maxlen.

    """
    if last_ew is not None and ew_combine_allowed:
        to_encode = str(
            get_unstructured(lines[-1][last_ew:] + to_encode))
        lines[-1] = lines[-1][:last_ew]
    if to_encode[0] in WSP:
        # We're joining this to non-encoded text, so don't encode
        # the leading blank.
        leading_wsp = to_encode[0]
        to_encode = to_encode[1:]
        if (len(lines[-1]) == maxlen):
            lines.append(_steal_trailing_WSP_if_exists(lines))
        lines[-1] += leading_wsp
    trailing_wsp = ''
    if to_encode[-1] in WSP:
        # Likewise for the trailing space.
        trailing_wsp = to_encode[-1]
        to_encode = to_encode[:-1]
    new_last_ew = len(lines[-1]) if last_ew is None else last_ew

    encode_as = 'utf-8' if charset == 'us-ascii' else charset

    # The RFC2047 chrome takes up 7 characters plus the length
    # of the charset name.
    chrome_len = len(encode_as) + 7

    if (chrome_len + 1) >= maxlen:
        raise errors.HeaderParseError(
            "max_line_length is too small to fit an encoded word")

    while to_encode:
        remaining_space = maxlen - len(lines[-1])
        text_space = remaining_space - chrome_len
        if text_space <= 0:
            lines.append(' ')
            continue

        to_encode_word = to_encode[:text_space]
        encoded_word = _ew.encode(to_encode_word, charset=encode_as)
        excess = len(encoded_word) - remaining_space
        while excess > 0:
            # Since the chunk to encode is guaranteed to fit into less than 100 characters,
            # shrinking it by one at a time shouldn't take long.
            to_encode_word = to_encode_word[:-1]
            encoded_word = _ew.encode(to_encode_word, charset=encode_as)
            excess = len(encoded_word) - remaining_space
        lines[-1] += encoded_word
        to_encode = to_encode[len(to_encode_word):]

        if to_encode:
            lines.append(' ')
            new_last_ew = len(lines[-1])
    lines[-1] += trailing_wsp
    return new_last_ew if ew_combine_allowed else None

def _fold_mime_parameters(part, lines, maxlen, encoding):
    """Fold TokenList 'part' into the 'lines' list as mime parameters.

    Using the decoded list of parameters and values, format them according to
    the RFC rules, including using RFC2231 encoding if the value cannot be
    expressed in 'encoding' and/or the parameter+value is too long to fit
    within 'maxlen'.

    """
    # Special case for RFC2231 encoding: start from decoded values and use
    # RFC2231 encoding iff needed.
    #
    # Note that the 1 and 2s being added to the length calculations are
    # accounting for the possibly-needed spaces and semicolons we'll be adding.
    #
    for name, value in part.params:
        # XXX What if this ';' puts us over maxlen the first time through the
        # loop?  We should split the header value onto a newline in that case,
        # but to do that we need to recognize the need earlier or reparse the
        # header, so I'm going to ignore that bug for now.  It'll only put us
        # one character over.
        if not lines[-1].rstrip().endswith(';'):
            lines[-1] += ';'
        charset = encoding
        error_handler = 'strict'
        try:
            value.encode(encoding)
            encoding_required = False
        except UnicodeEncodeError:
            encoding_required = True
            if utils._has_surrogates(value):
                charset = 'unknown-8bit'
                error_handler = 'surrogateescape'
            else:
                charset = 'utf-8'
        if encoding_required:
            encoded_value = urllib.parse.quote(
                value, safe='', errors=error_handler)
            tstr = "{}*={}''{}".format(name, charset, encoded_value)
        else:
            tstr = '{}={}'.format(name, quote_string(value))
        if len(lines[-1]) + len(tstr) + 1 < maxlen:
            lines[-1] = lines[-1] + ' ' + tstr
            continue
        elif len(tstr) + 2 <= maxlen:
            lines.append(' ' + tstr)
            continue
        # We need multiple sections.  We are allowed to mix encoded and
        # non-encoded sections, but we aren't going to.  We'll encode them all.
        section = 0
        extra_chrome = charset + "''"
        while value:
            chrome_len = len(name) + len(str(section)) + 3 + len(extra_chrome)
            if maxlen <= chrome_len + 3:
                # We need room for the leading blank, the trailing semicolon,
                # and at least one character of the value.  If we don't
                # have that, we'd be stuck, so in that case fall back to
                # the RFC standard width.
                maxlen = 78
            splitpoint = maxchars = maxlen - chrome_len - 2
            while True:
                partial = value[:splitpoint]
                encoded_value = urllib.parse.quote(
                    partial, safe='', errors=error_handler)
                if len(encoded_value) <= maxchars:
                    break
                splitpoint -= 1
            lines.append(" {}*{}*={}{}".format(
                name, section, extra_chrome, encoded_value))
            extra_chrome = ''
            section += 1
            value = value[splitpoint:]
            if value:
                lines[-1] += ';'
<|endoftext|># Copyright (C) 2001-2010 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Miscellaneous utilities."""

__all__ = [
    'collapse_rfc2231_value',
    'decode_params',
    'decode_rfc2231',
    'encode_rfc2231',
    'formataddr',
    'formatdate',
    'format_datetime',
    'getaddresses',
    'make_msgid',
    'mktime_tz',
    'parseaddr',
    'parsedate',
    'parsedate_tz',
    'parsedate_to_datetime',
    'unquote',
    ]

import os
import re
import time
import random
import socket
import datetime
import urllib.parse

from email._parseaddr import quote
from email._parseaddr import AddressList as _AddressList
from email._parseaddr import mktime_tz

from email._parseaddr import parsedate, parsedate_tz, _parsedate_tz

# Intrapackage imports
from email.charset import Charset

COMMASPACE = ', '
EMPTYSTRING = ''
UEMPTYSTRING = ''
CRLF = '\r\n'
TICK = "'"

specialsre = re.compile(r'[][\\()<>@,:;".]')
escapesre = re.compile(r'[\\"]')


def _has_surrogates(s):
    """Return True if s contains surrogate-escaped binary data."""
    # This check is based on the fact that unless there are surrogates, utf8
    # (Python's default encoding) can encode any string.  This is the fastest
    # way to check for surrogates, see issue 11454 for timings.
    try:
        s.encode()
        return False
    except UnicodeEncodeError:
        return True

# How to deal with a string containing bytes before handing it to the
# application through the 'normal' interface.
def _sanitize(string):
    # Turn any escaped bytes into unicode 'unknown' char.  If the escaped
    # bytes happen to be utf-8 they will instead get decoded, even if they
    # were invalid in the charset the source was supposed to be in.  This
    # seems like it is not a bad thing; a defect was still registered.
    original_bytes = string.encode('utf-8', 'surrogateescape')
    return original_bytes.decode('utf-8', 'replace')



# Helpers

def formataddr(pair, charset='utf-8'):
    """The inverse of parseaddr(), this takes a 2-tuple of the form
    (realname, email_address) and returns the string value suitable
    for an RFC 2822 From, To or Cc header.

    If the first element of pair is false, then the second element is
    returned unmodified.

    The optional charset is the character set that is used to encode
    realname in case realname is not ASCII safe.  Can be an instance of str or
    a Charset-like object which has a header_encode method.  Default is
    'utf-8'.
    """
    name, address = pair
    # The address MUST (per RFC) be ascii, so raise a UnicodeError if it isn't.
    address.encode('ascii')
    if name:
        try:
            name.encode('ascii')
        except UnicodeEncodeError:
            if isinstance(charset, str):
                charset = Charset(charset)
            encoded_name = charset.header_encode(name)
            return "%s <%s>" % (encoded_name, address)
        else:
            quotes = ''
            if specialsre.search(name):
                quotes = '"'
            name = escapesre.sub(r'\\\g<0>', name)
            return '%s%s%s <%s>' % (quotes, name, quotes, address)
    return address


def _iter_escaped_chars(addr):
    pos = 0
    escape = False
    for pos, ch in enumerate(addr):
        if escape:
            yield (pos, '\\' + ch)
            escape = False
        elif ch == '\\':
            escape = True
        else:
            yield (pos, ch)
    if escape:
        yield (pos, '\\')


def _strip_quoted_realnames(addr):
    """Strip real names between quotes."""
    if '"' not in addr:
        # Fast path
        return addr

    start = 0
    open_pos = None
    result = []
    for pos, ch in _iter_escaped_chars(addr):
        if ch == '"':
            if open_pos is None:
                open_pos = pos
            else:
                if start != open_pos:
                    result.append(addr[start:open_pos])
                start = pos + 1
                open_pos = None

    if start < len(addr):
        result.append(addr[start:])

    return ''.join(result)


supports_strict_parsing = True

def getaddresses(fieldvalues, *, strict=True):
    """Return a list of (REALNAME, EMAIL) or ('','') for each fieldvalue.

    When parsing fails for a fieldvalue, a 2-tuple of ('', '') is returned in
    its place.

    If strict is true, use a strict parser which rejects malformed inputs.
    """

    # If strict is true, if the resulting list of parsed addresses is greater
    # than the number of fieldvalues in the input list, a parsing error has
    # occurred and consequently a list containing a single empty 2-tuple [('',
    # '')] is returned in its place. This is done to avoid invalid output.
    #
    # Malformed input: getaddresses(['alice@example.com <bob@example.com>'])
    # Invalid output: [('', 'alice@example.com'), ('', 'bob@example.com')]
    # Safe output: [('', '')]

    if not strict:
        all = COMMASPACE.join(str(v) for v in fieldvalues)
        a = _AddressList(all)
        return a.addresslist

    fieldvalues = [str(v) for v in fieldvalues]
    fieldvalues = _pre_parse_validation(fieldvalues)
    addr = COMMASPACE.join(fieldvalues)
    a = _AddressList(addr)
    result = _post_parse_validation(a.addresslist)

    # Treat output as invalid if the number of addresses is not equal to the
    # expected number of addresses.
    n = 0
    for v in fieldvalues:
        # When a comma is used in the Real Name part it is not a deliminator.
        # So strip those out before counting the commas.
        v = _strip_quoted_realnames(v)
        # Expected number of addresses: 1 + number of commas
        n += 1 + v.count(',')
    if len(result) != n:
        return [('', '')]

    return result


def _check_parenthesis(addr):
    # Ignore parenthesis in quoted real names.
    addr = _strip_quoted_realnames(addr)

    opens = 0
    for pos, ch in _iter_escaped_chars(addr):
        if ch == '(':
            opens += 1
        elif ch == ')':
            opens -= 1
            if opens < 0:
                return False
    return (opens == 0)


def _pre_parse_validation(email_header_fields):
    accepted_values = []
    for v in email_header_fields:
        if not _check_parenthesis(v):
            v = "('', '')"
        accepted_values.append(v)

    return accepted_values


def _post_parse_validation(parsed_email_header_tuples):
    accepted_values = []
    # The parser would have parsed a correctly formatted domain-literal
    # The existence of an [ after parsing indicates a parsing failure
    for v in parsed_email_header_tuples:
        if '[' in v[1]:
            v = ('', '')
        accepted_values.append(v)

    return accepted_values


def _format_timetuple_and_zone(timetuple, zone):
    return '%s, %02d %s %04d %02d:%02d:%02d %s' % (
        ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][timetuple[6]],
        timetuple[2],
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][timetuple[1] - 1],
        timetuple[0], timetuple[3], timetuple[4], timetuple[5],
        zone)

def formatdate(timeval=None, localtime=False, usegmt=False):
    """Returns a date string as specified by RFC 2822, e.g.:

    Fri, 09 Nov 2001 01:08:47 -0000

    Optional timeval if given is a floating point time value as accepted by
    gmtime() and localtime(), otherwise the current time is used.

    Optional localtime is a flag that when True, interprets timeval, and
    returns a date relative to the local timezone instead of UTC, properly
    taking daylight savings time into account.

    Optional argument usegmt means that the timezone is written out as
    an ascii string, not numeric one (so "GMT" instead of "+0000"). This
    is needed for HTTP, and is only used when localtime==False.
    """
    # Note: we cannot use strftime() because that honors the locale and RFC
    # 2822 requires that day and month names be the English abbreviations.
    if timeval is None:
        timeval = time.time()
    if localtime or usegmt:
        dt = datetime.datetime.fromtimestamp(timeval, datetime.timezone.utc)
    else:
        dt = datetime.datetime.utcfromtimestamp(timeval)
    if localtime:
        dt = dt.astimezone()
        usegmt = False
    return format_datetime(dt, usegmt)

def format_datetime(dt, usegmt=False):
    """Turn a datetime into a date string as specified in RFC 2822.

    If usegmt is True, dt must be an aware datetime with an offset of zero.  In
    this case 'GMT' will be rendered instead of the normal +0000 required by
    RFC2822.  This is to support HTTP headers involving date stamps.
    """
    now = dt.timetuple()
    if usegmt:
        if dt.tzinfo is None or dt.tzinfo != datetime.timezone.utc:
            raise ValueError("usegmt option requires a UTC datetime")
        zone = 'GMT'
    elif dt.tzinfo is None:
        zone = '-0000'
    else:
        zone = dt.strftime("%z")
    return _format_timetuple_and_zone(now, zone)


def make_msgid(idstring=None, domain=None):
    """Returns a string suitable for RFC 2822 compliant Message-ID, e.g:

    <142480216486.20800.16526388040877946887@nightshade.la.mastaler.com>

    Optional idstring if given is a string used to strengthen the
    uniqueness of the message id.  Optional domain if given provides the
    portion of the message id after the '@'.  It defaults to the locally
    defined hostname.
    """
    timeval = int(time.time()*100)
    pid = os.getpid()
    randint = random.getrandbits(64)
    if idstring is None:
        idstring = ''
    else:
        idstring = '.' + idstring
    if domain is None:
        domain = socket.getfqdn()
    msgid = '<%d.%d.%d%s@%s>' % (timeval, pid, randint, idstring, domain)
    return msgid


def parsedate_to_datetime(data):
    parsed_date_tz = _parsedate_tz(data)
    if parsed_date_tz is None:
        raise ValueError('Invalid date value or format "%s"' % str(data))
    *dtuple, tz = parsed_date_tz
    if tz is None:
        return datetime.datetime(*dtuple[:6])
    return datetime.datetime(*dtuple[:6],
            tzinfo=datetime.timezone(datetime.timedelta(seconds=tz)))


def parseaddr(addr, *, strict=True):
    """
    Parse addr into its constituent realname and email address parts.

    Return a tuple of realname and email address, unless the parse fails, in
    which case return a 2-tuple of ('', '').

    If strict is True, use a strict parser which rejects malformed inputs.
    """
    if not strict:
        addrs = _AddressList(addr).addresslist
        if not addrs:
            return ('', '')
        return addrs[0]

    if isinstance(addr, list):
        addr = addr[0]

    if not isinstance(addr, str):
        return ('', '')

    addr = _pre_parse_validation([addr])[0]
    addrs = _post_parse_validation(_AddressList(addr).addresslist)

    if not addrs or len(addrs) > 1:
        return ('', '')

    return addrs[0]


# rfc822.unquote() doesn't properly de-backslash-ify in Python pre-2.3.
def unquote(str):
    """Remove quotes from a string."""
    if len(str) > 1:
        if str.startswith('"') and str.endswith('"'):
            return str[1:-1].replace('\\\\', '\\').replace('\\"', '"')
        if str.startswith('<') and str.endswith('>'):
            return str[1:-1]
    return str



# RFC2231-related functions - parameter encoding and decoding
def decode_rfc2231(s):
    """Decode string according to RFC 2231"""
    parts = s.split(TICK, 2)
    if len(parts) <= 2:
        return None, None, s
    return parts


def encode_rfc2231(s, charset=None, language=None):
    """Encode string according to RFC 2231.

    If neither charset nor language is given, then s is returned as-is.  If
    charset is given but not language, the string is encoded using the empty
    string for language.
    """
    s = urllib.parse.quote(s, safe='', encoding=charset or 'ascii')
    if charset is None and language is None:
        return s
    if language is None:
        language = ''
    return "%s'%s'%s" % (charset, language, s)


rfc2231_continuation = re.compile(r'^(?P<name>\w+)\*((?P<num>[0-9]+)\*?)?$',
    re.ASCII)

def decode_params(params):
    """Decode parameters list according to RFC 2231.

    params is a sequence of 2-tuples containing (param name, string value).
    """
    new_params = [params[0]]
    # Map parameter's name to a list of continuations.  The values are a
    # 3-tuple of the continuation number, the string value, and a flag
    # specifying whether a particular segment is %-encoded.
    rfc2231_params = {}
    for name, value in params[1:]:
        encoded = name.endswith('*')
        value = unquote(value)
        mo = rfc2231_continuation.match(name)
        if mo:
            name, num = mo.group('name', 'num')
            if num is not None:
                num = int(num)
            rfc2231_params.setdefault(name, []).append((num, value, encoded))
        else:
            new_params.append((name, '"%s"' % quote(value)))
    if rfc2231_params:
        for name, continuations in rfc2231_params.items():
            value = []
            extended = False
            # Sort by number
            continuations.sort()
            # And now append all values in numerical order, converting
            # %-encodings for the encoded segments.  If any of the
            # continuation names ends in a *, then the entire string, after
            # decoding segments and concatenating, must have the charset and
            # language specifiers at the beginning of the string.
            for num, s, encoded in continuations:
                if encoded:
                    # Decode as "latin-1", so the characters in s directly
                    # represent the percent-encoded octet values.
                    # collapse_rfc2231_value treats this as an octet sequence.
                    s = urllib.parse.unquote(s, encoding="latin-1")
                    extended = True
                value.append(s)
            value = quote(EMPTYSTRING.join(value))
            if extended:
                charset, language, value = decode_rfc2231(value)
                new_params.append((name, (charset, language, '"%s"' % value)))
            else:
                new_params.append((name, '"%s"' % value))
    return new_params

def collapse_rfc2231_value(value, errors='replace',
                           fallback_charset='us-ascii'):
    if not isinstance(value, tuple) or len(value) != 3:
        return unquote(value)
    # While value comes to us as a unicode string, we need it to be a bytes
    # object.  We do not want bytes() normal utf-8 decoder, we want a straight
    # interpretation of the string as character bytes.
    charset, language, text = value
    if charset is None:
        # Issue 17369: if charset/lang is None, decode_rfc2231 couldn't parse
        # the value, so use the fallback_charset.
        charset = fallback_charset
    rawbytes = bytes(text, 'raw-unicode-escape')
    try:
        return str(rawbytes, charset, errors)
    except LookupError:
        # charset is not a known codec.
        return unquote(text)


#
# datetime doesn't provide a localtime function yet, so provide one.  Code
# adapted from the patch in issue 9527.  This may not be perfect, but it is
# better than not having it.
#

def localtime(dt=None, isdst=-1):
    """Return local time as an aware datetime object.

    If called without arguments, return current time.  Otherwise *dt*
    argument should be a datetime instance, and it is converted to the
    local time zone according to the system time zone database.  If *dt* is
    naive (that is, dt.tzinfo is None), it is assumed to be in local time.
    In this case, a positive or zero value for *isdst* causes localtime to
    presume initially that summer time (for example, Daylight Saving Time)
    is or is not (respectively) in effect for the specified time.  A
    negative value for *isdst* causes the localtime() function to attempt
    to divine whether summer time is in effect for the specified time.

    """
    if dt is None:
        return datetime.datetime.now(datetime.timezone.utc).astimezone()
    if dt.tzinfo is not None:
        return dt.astimezone()
    # We have a naive datetime.  Convert to a (localtime) timetuple and pass to
    # system mktime together with the isdst hint.  System mktime will return
    # seconds since epoch.
    tm = dt.timetuple()[:-1] + (isdst,)
    seconds = time.mktime(tm)
    localtm = time.localtime(seconds)
    try:
        delta = datetime.timedelta(seconds=localtm.tm_gmtoff)
        tz = datetime.timezone(delta, localtm.tm_zone)
    except AttributeError:
        # Compute UTC offset and compare with the value implied by tm_isdst.
        # If the values match, use the zone name implied by tm_isdst.
        delta = dt - datetime.datetime(*time.gmtime(seconds)[:6])
        dst = time.daylight and localtm.tm_isdst > 0
        gmtoff = -(time.altzone if dst else time.timezone)
        if delta == datetime.timedelta(seconds=gmtoff):
            tz = datetime.timezone(delta, time.tzname[dst])
        else:
            tz = datetime.timezone(delta)
    return dt.replace(tzinfo=tz)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Ben Gertzfield
# Contact: email-sig@python.org

"""Quoted-printable content transfer encoding per RFCs 2045-2047.

This module handles the content transfer encoding method defined in RFC 2045
to encode US ASCII-like 8-bit data called `quoted-printable'.  It is used to
safely encode text that is in a character set similar to the 7-bit US ASCII
character set, but that includes some 8-bit characters that are normally not
allowed in email bodies or headers.

Quoted-printable is very space-inefficient for encoding binary files; use the
email.base64mime module for that instead.

This module provides an interface to encode and decode both headers and bodies
with quoted-printable encoding.

RFC 2045 defines a method for including character set information in an
`encoded-word' in a header.  This method is commonly used for 8-bit real names
in To:/From:/Cc: etc. fields, as well as Subject: lines.

This module does not do the line wrapping or end-of-line character
conversion necessary for proper internationalized headers; it only
does dumb encoding and decoding.  To deal with the various line
wrapping issues, use the email.header module.
"""

__all__ = [
    'body_decode',
    'body_encode',
    'body_length',
    'decode',
    'decodestring',
    'header_decode',
    'header_encode',
    'header_length',
    'quote',
    'unquote',
    ]

import re

from string import ascii_letters, digits, hexdigits

CRLF = '\r\n'
NL = '\n'
EMPTYSTRING = ''

# Build a mapping of octets to the expansion of that octet.  Since we're only
# going to have 256 of these things, this isn't terribly inefficient
# space-wise.  Remember that headers and bodies have different sets of safe
# characters.  Initialize both maps with the full expansion, and then override
# the safe bytes with the more compact form.
_QUOPRI_MAP = ['=%02X' % c for c in range(256)]
_QUOPRI_HEADER_MAP = _QUOPRI_MAP[:]
_QUOPRI_BODY_MAP = _QUOPRI_MAP[:]

# Safe header bytes which need no encoding.
for c in b'-!*+/' + ascii_letters.encode('ascii') + digits.encode('ascii'):
    _QUOPRI_HEADER_MAP[c] = chr(c)
# Headers have one other special encoding; spaces become underscores.
_QUOPRI_HEADER_MAP[ord(' ')] = '_'

# Safe body bytes which need no encoding.
for c in (b' !"#$%&\'()*+,-./0123456789:;<>'
          b'?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
          b'abcdefghijklmnopqrstuvwxyz{|}~\t'):
    _QUOPRI_BODY_MAP[c] = chr(c)



# Helpers
def header_check(octet):
    """Return True if the octet should be escaped with header quopri."""
    return chr(octet) != _QUOPRI_HEADER_MAP[octet]


def body_check(octet):
    """Return True if the octet should be escaped with body quopri."""
    return chr(octet) != _QUOPRI_BODY_MAP[octet]


def header_length(bytearray):
    """Return a header quoted-printable encoding length.

    Note that this does not include any RFC 2047 chrome added by
    `header_encode()`.

    :param bytearray: An array of bytes (a.k.a. octets).
    :return: The length in bytes of the byte array when it is encoded with
        quoted-printable for headers.
    """
    return sum(len(_QUOPRI_HEADER_MAP[octet]) for octet in bytearray)


def body_length(bytearray):
    """Return a body quoted-printable encoding length.

    :param bytearray: An array of bytes (a.k.a. octets).
    :return: The length in bytes of the byte array when it is encoded with
        quoted-printable for bodies.
    """
    return sum(len(_QUOPRI_BODY_MAP[octet]) for octet in bytearray)


def _max_append(L, s, maxlen, extra=''):
    if not isinstance(s, str):
        s = chr(s)
    if not L:
        L.append(s.lstrip())
    elif len(L[-1]) + len(s) <= maxlen:
        L[-1] += extra + s
    else:
        L.append(s.lstrip())


def unquote(s):
    """Turn a string in the form =AB to the ASCII character with value 0xab"""
    return chr(int(s[1:3], 16))


def quote(c):
    return _QUOPRI_MAP[ord(c)]


def header_encode(header_bytes, charset='iso-8859-1'):
    """Encode a single header line with quoted-printable (like) encoding.

    Defined in RFC 2045, this `Q' encoding is similar to quoted-printable, but
    used specifically for email header fields to allow charsets with mostly 7
    bit characters (and some 8 bit) to remain more or less readable in non-RFC
    2045 aware mail clients.

    charset names the character set to use in the RFC 2046 header.  It
    defaults to iso-8859-1.
    """
    # Return empty headers as an empty string.
    if not header_bytes:
        return ''
    # Iterate over every byte, encoding if necessary.
    encoded = header_bytes.decode('latin1').translate(_QUOPRI_HEADER_MAP)
    # Now add the RFC chrome to each encoded chunk and glue the chunks
    # together.
    return '=?%s?q?%s?=' % (charset, encoded)


_QUOPRI_BODY_ENCODE_MAP = _QUOPRI_BODY_MAP[:]
for c in b'\r\n':
    _QUOPRI_BODY_ENCODE_MAP[c] = chr(c)

def body_encode(body, maxlinelen=76, eol=NL):
    """Encode with quoted-printable, wrapping at maxlinelen characters.

    Each line of encoded text will end with eol, which defaults to "\\n".  Set
    this to "\\r\\n" if you will be using the result of this function directly
    in an email.

    Each line will be wrapped at, at most, maxlinelen characters before the
    eol string (maxlinelen defaults to 76 characters, the maximum value
    permitted by RFC 2045).  Long lines will have the 'soft line break'
    quoted-printable character "=" appended to them, so the decoded text will
    be identical to the original text.

    The minimum maxlinelen is 4 to have room for a quoted character ("=XX")
    followed by a soft line break.  Smaller values will generate a
    ValueError.

    """

    if maxlinelen < 4:
        raise ValueError("maxlinelen must be at least 4")
    if not body:
        return body

    # quote special characters
    body = body.translate(_QUOPRI_BODY_ENCODE_MAP)

    soft_break = '=' + eol
    # leave space for the '=' at the end of a line
    maxlinelen1 = maxlinelen - 1

    encoded_body = []
    append = encoded_body.append

    for line in body.splitlines():
        # break up the line into pieces no longer than maxlinelen - 1
        start = 0
        laststart = len(line) - 1 - maxlinelen
        while start <= laststart:
            stop = start + maxlinelen1
            # make sure we don't break up an escape sequence
            if line[stop - 2] == '=':
                append(line[start:stop - 1])
                start = stop - 2
            elif line[stop - 1] == '=':
                append(line[start:stop])
                start = stop - 1
            else:
                append(line[start:stop] + '=')
                start = stop

        # handle rest of line, special case if line ends in whitespace
        if line and line[-1] in ' \t':
            room = start - laststart
            if room >= 3:
                # It's a whitespace character at end-of-line, and we have room
                # for the three-character quoted encoding.
                q = quote(line[-1])
            elif room == 2:
                # There's room for the whitespace character and a soft break.
                q = line[-1] + soft_break
            else:
                # There's room only for a soft break.  The quoted whitespace
                # will be the only content on the subsequent line.
                q = soft_break + quote(line[-1])
            append(line[start:-1] + q)
        else:
            append(line[start:])

    # add back final newline if present
    if body[-1] in CRLF:
        append('')

    return eol.join(encoded_body)



# BAW: I'm not sure if the intent was for the signature of this function to be
# the same as base64MIME.decode() or not...
def decode(encoded, eol=NL):
    """Decode a quoted-printable string.

    Lines are separated with eol, which defaults to \\n.
    """
    if not encoded:
        return encoded
    # BAW: see comment in encode() above.  Again, we're building up the
    # decoded string with string concatenation, which could be done much more
    # efficiently.
    decoded = ''

    for line in encoded.splitlines():
        line = line.rstrip()
        if not line:
            decoded += eol
            continue

        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if c != '=':
                decoded += c
                i += 1
            # Otherwise, c == "=".  Are we at the end of the line?  If so, add
            # a soft line break.
            elif i+1 == n:
                i += 1
                continue
            # Decode if in form =AB
            elif i+2 < n and line[i+1] in hexdigits and line[i+2] in hexdigits:
                decoded += unquote(line[i:i+3])
                i += 3
            # Otherwise, not in form =AB, pass literally
            else:
                decoded += c
                i += 1

            if i == n:
                decoded += eol
    # Special case if original string did not end with eol
    if encoded[-1] not in '\r\n' and decoded.endswith(eol):
        decoded = decoded[:-1]
    return decoded


# For convenience and backwards compatibility w/ standard base64 module
body_decode = decode
decodestring = decode



def _unquote_match(match):
    """Turn a match in the form =AB to the ASCII character with value 0xab"""
    s = match.group(0)
    return unquote(s)


# Header decoding is done a bit differently
def header_decode(s):
    """Decode a string encoded with RFC 2045 MIME header `Q' encoding.

    This function does not parse a full MIME header value encoded with
    quoted-printable (like =?iso-8859-1?q?Hello_World?=) -- please use
    the high level email.header class for that functionality.
    """
    s = s.replace('_', ' ')
    return re.sub(r'=[a-fA-F0-9]{2}', _unquote_match, s, flags=re.ASCII)
<|endoftext|># Copyright (C) 2001-2007 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""A package for parsing, handling, and generating email messages."""

__all__ = [
    'base64mime',
    'charset',
    'encoders',
    'errors',
    'feedparser',
    'generator',
    'header',
    'iterators',
    'message',
    'message_from_file',
    'message_from_binary_file',
    'message_from_string',
    'message_from_bytes',
    'mime',
    'parser',
    'quoprimime',
    'utils',
    ]



# Some convenience routines.  Don't import Parser and Message as side-effects
# of importing email since those cascadingly import most of the rest of the
# email package.
def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def message_from_bytes(s, *args, **kws):
    """Parse a bytes string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import BytesParser
    return BytesParser(*args, **kws).parsebytes(s)

def message_from_file(fp, *args, **kws):
    """Read a file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import Parser
    return Parser(*args, **kws).parse(fp)

def message_from_binary_file(fp, *args, **kws):
    """Read a binary file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import BytesParser
    return BytesParser(*args, **kws).parse(fp)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Encodings and related functions."""

__all__ = [
    'encode_7or8bit',
    'encode_base64',
    'encode_noop',
    'encode_quopri',
    ]


from base64 import encodebytes as _bencode
from quopri import encodestring as _encodestring



def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    # Must encode spaces, which quopri.encodestring() doesn't do
    return enc.replace(b' ', b'=20')


def encode_base64(msg):
    """Encode the message's payload in Base64.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload(decode=True)
    encdata = str(_bencode(orig), 'ascii')
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'base64'



def encode_quopri(msg):
    """Encode the message's payload in quoted-printable.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload(decode=True)
    encdata = _qencode(orig)
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'quoted-printable'



def encode_7or8bit(msg):
    """Set the Content-Transfer-Encoding header to 7bit or 8bit."""
    orig = msg.get_payload(decode=True)
    if orig is None:
        # There's no payload.  For backwards compatibility we use 7bit
        msg['Content-Transfer-Encoding'] = '7bit'
        return
    # We play a trick to make this go fast.  If decoding from ASCII succeeds,
    # we know the data must be 7bit, otherwise treat it as 8bit.
    try:
        orig.decode('ascii')
    except UnicodeError:
        msg['Content-Transfer-Encoding'] = '8bit'
    else:
        msg['Content-Transfer-Encoding'] = '7bit'



def encode_noop(msg):
    """Do nothing."""
<|endoftext|># Copyright (C) 2002-2007 Python Software Foundation
# Contact: email-sig@python.org

"""Email address parsing code.

Lifted directly from rfc822.py.  This should eventually be rewritten.
"""

__all__ = [
    'mktime_tz',
    'parsedate',
    'parsedate_tz',
    'quote',
    ]

import time, calendar

SPACE = ' '
EMPTYSTRING = ''
COMMASPACE = ', '

# Parse a date field
_monthnames = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
               'aug', 'sep', 'oct', 'nov', 'dec',
               'january', 'february', 'march', 'april', 'may', 'june', 'july',
               'august', 'september', 'october', 'november', 'december']

_daynames = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

# The timezone table does not include the military time zones defined
# in RFC822, other than Z.  According to RFC1123, the description in
# RFC822 gets the signs wrong, so we can't rely on any such time
# zones.  RFC1123 recommends that numeric timezone indicators be used
# instead of timezone names.

_timezones = {'UT':0, 'UTC':0, 'GMT':0, 'Z':0,
              'AST': -400, 'ADT': -300,  # Atlantic (used in Canada)
              'EST': -500, 'EDT': -400,  # Eastern
              'CST': -600, 'CDT': -500,  # Central
              'MST': -700, 'MDT': -600,  # Mountain
              'PST': -800, 'PDT': -700   # Pacific
              }


def parsedate_tz(data):
    """Convert a date string to a time tuple.

    Accounts for military timezones.
    """
    res = _parsedate_tz(data)
    if not res:
        return
    if res[9] is None:
        res[9] = 0
    return tuple(res)

def _parsedate_tz(data):
    """Convert date to extended time tuple.

    The last (additional) element is the time zone offset in seconds, except if
    the timezone was specified as -0000.  In that case the last element is
    None.  This indicates a UTC timestamp that explicitly declaims knowledge of
    the source timezone, as opposed to a +0000 timestamp that indicates the
    source timezone really was UTC.

    """
    if not data:
        return None
    data = data.split()
    if not data:  # This happens for whitespace-only input.
        return None
    # The FWS after the comma after the day-of-week is optional, so search and
    # adjust for this.
    if data[0].endswith(',') or data[0].lower() in _daynames:
        # There's a dayname here. Skip it
        del data[0]
    else:
        i = data[0].rfind(',')
        if i >= 0:
            data[0] = data[0][i+1:]
    if len(data) == 3: # RFC 850 date, deprecated
        stuff = data[0].split('-')
        if len(stuff) == 3:
            data = stuff + data[1:]
    if len(data) == 4:
        s = data[3]
        i = s.find('+')
        if i == -1:
            i = s.find('-')
        if i > 0:
            data[3:] = [s[:i], s[i:]]
        else:
            data.append('') # Dummy tz
    if len(data) < 5:
        return None
    data = data[:5]
    [dd, mm, yy, tm, tz] = data
    if not (dd and mm and yy):
        return None
    mm = mm.lower()
    if mm not in _monthnames:
        dd, mm = mm, dd.lower()
        if mm not in _monthnames:
            return None
    mm = _monthnames.index(mm) + 1
    if mm > 12:
        mm -= 12
    if dd[-1] == ',':
        dd = dd[:-1]
    i = yy.find(':')
    if i > 0:
        yy, tm = tm, yy
    if yy[-1] == ',':
        yy = yy[:-1]
        if not yy:
            return None
    if not yy[0].isdigit():
        yy, tz = tz, yy
    if tm[-1] == ',':
        tm = tm[:-1]
    tm = tm.split(':')
    if len(tm) == 2:
        [thh, tmm] = tm
        tss = '0'
    elif len(tm) == 3:
        [thh, tmm, tss] = tm
    elif len(tm) == 1 and '.' in tm[0]:
        # Some non-compliant MUAs use '.' to separate time elements.
        tm = tm[0].split('.')
        if len(tm) == 2:
            [thh, tmm] = tm
            tss = 0
        elif len(tm) == 3:
            [thh, tmm, tss] = tm
        else:
            return None
    else:
        return None
    try:
        yy = int(yy)
        dd = int(dd)
        thh = int(thh)
        tmm = int(tmm)
        tss = int(tss)
    except ValueError:
        return None
    # Check for a yy specified in two-digit format, then convert it to the
    # appropriate four-digit format, according to the POSIX standard. RFC 822
    # calls for a two-digit yy, but RFC 2822 (which obsoletes RFC 822)
    # mandates a 4-digit yy. For more information, see the documentation for
    # the time module.
    if yy < 100:
        # The year is between 1969 and 1999 (inclusive).
        if yy > 68:
            yy += 1900
        # The year is between 2000 and 2068 (inclusive).
        else:
            yy += 2000
    tzoffset = None
    tz = tz.upper()
    if tz in _timezones:
        tzoffset = _timezones[tz]
    else:
        try:
            tzoffset = int(tz)
        except ValueError:
            pass
        if tzoffset==0 and tz.startswith('-'):
            tzoffset = None
    # Convert a timezone offset into seconds ; -0500 -> -18000
    if tzoffset:
        if tzoffset < 0:
            tzsign = -1
            tzoffset = -tzoffset
        else:
            tzsign = 1
        tzoffset = tzsign * ( (tzoffset//100)*3600 + (tzoffset % 100)*60)
    # Daylight Saving Time flag is set to -1, since DST is unknown.
    return [yy, mm, dd, thh, tmm, tss, 0, 1, -1, tzoffset]


def parsedate(data):
    """Convert a time string to a time tuple."""
    t = parsedate_tz(data)
    if isinstance(t, tuple):
        return t[:9]
    else:
        return t


def mktime_tz(data):
    """Turn a 10-tuple as returned by parsedate_tz() into a POSIX timestamp."""
    if data[9] is None:
        # No zone info, so localtime is better assumption than GMT
        return time.mktime(data[:8] + (-1,))
    else:
        t = calendar.timegm(data)
        return t - data[9]


def quote(str):
    """Prepare string to be used in a quoted string.

    Turns backslash and double quote characters into quoted pairs.  These
    are the only characters that need to be quoted inside a quoted string.
    Does not add the surrounding double quotes.
    """
    return str.replace('\\', '\\\\').replace('"', '\\"')


class AddrlistClass:
    """Address parser class by Ben Escoto.

    To understand what this class does, it helps to have a copy of RFC 2822 in
    front of you.

    Note: this class interface is deprecated and may be removed in the future.
    Use email.utils.AddressList instead.
    """

    def __init__(self, field):
        """Initialize a new instance.

        `field' is an unparsed address header field, containing
        one or more addresses.
        """
        self.specials = '()<>@,:;.\"[]'
        self.pos = 0
        self.LWS = ' \t'
        self.CR = '\r\n'
        self.FWS = self.LWS + self.CR
        self.atomends = self.specials + self.LWS + self.CR
        # Note that RFC 2822 now specifies `.' as obs-phrase, meaning that it
        # is obsolete syntax.  RFC 2822 requires that we recognize obsolete
        # syntax, so allow dots in phrases.
        self.phraseends = self.atomends.replace('.', '')
        self.field = field
        self.commentlist = []

    def gotonext(self):
        """Skip white space and extract comments."""
        wslist = []
        while self.pos < len(self.field):
            if self.field[self.pos] in self.LWS + '\n\r':
                if self.field[self.pos] not in '\n\r':
                    wslist.append(self.field[self.pos])
                self.pos += 1
            elif self.field[self.pos] == '(':
                self.commentlist.append(self.getcomment())
            else:
                break
        return EMPTYSTRING.join(wslist)

    def getaddrlist(self):
        """Parse all addresses.

        Returns a list containing all of the addresses.
        """
        result = []
        while self.pos < len(self.field):
            ad = self.getaddress()
            if ad:
                result += ad
            else:
                result.append(('', ''))
        return result

    def getaddress(self):
        """Parse the next address."""
        self.commentlist = []
        self.gotonext()

        oldpos = self.pos
        oldcl = self.commentlist
        plist = self.getphraselist()

        self.gotonext()
        returnlist = []

        if self.pos >= len(self.field):
            # Bad email address technically, no domain.
            if plist:
                returnlist = [(SPACE.join(self.commentlist), plist[0])]

        elif self.field[self.pos] in '.@':
            # email address is just an addrspec
            # this isn't very efficient since we start over
            self.pos = oldpos
            self.commentlist = oldcl
            addrspec = self.getaddrspec()
            returnlist = [(SPACE.join(self.commentlist), addrspec)]

        elif self.field[self.pos] == ':':
            # address is a group
            returnlist = []

            fieldlen = len(self.field)
            self.pos += 1
            while self.pos < len(self.field):
                self.gotonext()
                if self.pos < fieldlen and self.field[self.pos] == ';':
                    self.pos += 1
                    break
                returnlist = returnlist + self.getaddress()

        elif self.field[self.pos] == '<':
            # Address is a phrase then a route addr
            routeaddr = self.getrouteaddr()

            if self.commentlist:
                returnlist = [(SPACE.join(plist) + ' (' +
                               ' '.join(self.commentlist) + ')', routeaddr)]
            else:
                returnlist = [(SPACE.join(plist), routeaddr)]

        else:
            if plist:
                returnlist = [(SPACE.join(self.commentlist), plist[0])]
            elif self.field[self.pos] in self.specials:
                self.pos += 1

        self.gotonext()
        if self.pos < len(self.field) and self.field[self.pos] == ',':
            self.pos += 1
        return returnlist

    def getrouteaddr(self):
        """Parse a route address (Return-path value).

        This method just skips all the route stuff and returns the addrspec.
        """
        if self.field[self.pos] != '<':
            return

        expectroute = False
        self.pos += 1
        self.gotonext()
        adlist = ''
        while self.pos < len(self.field):
            if expectroute:
                self.getdomain()
                expectroute = False
            elif self.field[self.pos] == '>':
                self.pos += 1
                break
            elif self.field[self.pos] == '@':
                self.pos += 1
                expectroute = True
            elif self.field[self.pos] == ':':
                self.pos += 1
            else:
                adlist = self.getaddrspec()
                self.pos += 1
                break
            self.gotonext()

        return adlist

    def getaddrspec(self):
        """Parse an RFC 2822 addr-spec."""
        aslist = []

        self.gotonext()
        while self.pos < len(self.field):
            preserve_ws = True
            if self.field[self.pos] == '.':
                if aslist and not aslist[-1].strip():
                    aslist.pop()
                aslist.append('.')
                self.pos += 1
                preserve_ws = False
            elif self.field[self.pos] == '"':
                aslist.append('"%s"' % quote(self.getquote()))
            elif self.field[self.pos] in self.atomends:
                if aslist and not aslist[-1].strip():
                    aslist.pop()
                break
            else:
                aslist.append(self.getatom())
            ws = self.gotonext()
            if preserve_ws and ws:
                aslist.append(ws)

        if self.pos >= len(self.field) or self.field[self.pos] != '@':
            return EMPTYSTRING.join(aslist)

        aslist.append('@')
        self.pos += 1
        self.gotonext()
        domain = self.getdomain()
        if not domain:
            # Invalid domain, return an empty address instead of returning a
            # local part to denote failed parsing.
            return EMPTYSTRING
        return EMPTYSTRING.join(aslist) + domain

    def getdomain(self):
        """Get the complete domain name from an address."""
        sdlist = []
        while self.pos < len(self.field):
            if self.field[self.pos] in self.LWS:
                self.pos += 1
            elif self.field[self.pos] == '(':
                self.commentlist.append(self.getcomment())
            elif self.field[self.pos] == '[':
                sdlist.append(self.getdomainliteral())
            elif self.field[self.pos] == '.':
                self.pos += 1
                sdlist.append('.')
            elif self.field[self.pos] == '@':
                # bpo-34155: Don't parse domains with two `@` like
                # `a@malicious.org@important.com`.
                return EMPTYSTRING
            elif self.field[self.pos] in self.atomends:
                break
            else:
                sdlist.append(self.getatom())
        return EMPTYSTRING.join(sdlist)

    def getdelimited(self, beginchar, endchars, allowcomments=True):
        """Parse a header fragment delimited by special characters.

        `beginchar' is the start character for the fragment.
        If self is not looking at an instance of `beginchar' then
        getdelimited returns the empty string.

        `endchars' is a sequence of allowable end-delimiting characters.
        Parsing stops when one of these is encountered.

        If `allowcomments' is non-zero, embedded RFC 2822 comments are allowed
        within the parsed fragment.
        """
        if self.field[self.pos] != beginchar:
            return ''

        slist = ['']
        quote = False
        self.pos += 1
        while self.pos < len(self.field):
            if quote:
                slist.append(self.field[self.pos])
                quote = False
            elif self.field[self.pos] in endchars:
                self.pos += 1
                break
            elif allowcomments and self.field[self.pos] == '(':
                slist.append(self.getcomment())
                continue        # have already advanced pos from getcomment
            elif self.field[self.pos] == '\\':
                quote = True
            else:
                slist.append(self.field[self.pos])
            self.pos += 1

        return EMPTYSTRING.join(slist)

    def getquote(self):
        """Get a quote-delimited fragment from self's field."""
        return self.getdelimited('"', '"\r', False)

    def getcomment(self):
        """Get a parenthesis-delimited fragment from self's field."""
        return self.getdelimited('(', ')\r', True)

    def getdomainliteral(self):
        """Parse an RFC 2822 domain-literal."""
        return '[%s]' % self.getdelimited('[', ']\r', False)

    def getatom(self, atomends=None):
        """Parse an RFC 2822 atom.

        Optional atomends specifies a different set of end token delimiters
        (the default is to use self.atomends).  This is used e.g. in
        getphraselist() since phrase endings must not include the `.' (which
        is legal in phrases)."""
        atomlist = ['']
        if atomends is None:
            atomends = self.atomends

        while self.pos < len(self.field):
            if self.field[self.pos] in atomends:
                break
            else:
                atomlist.append(self.field[self.pos])
            self.pos += 1

        return EMPTYSTRING.join(atomlist)

    def getphraselist(self):
        """Parse a sequence of RFC 2822 phrases.

        A phrase is a sequence of words, which are in turn either RFC 2822
        atoms or quoted-strings.  Phrases are canonicalized by squeezing all
        runs of continuous whitespace into one space.
        """
        plist = []

        while self.pos < len(self.field):
            if self.field[self.pos] in self.FWS:
                self.pos += 1
            elif self.field[self.pos] == '"':
                plist.append(self.getquote())
            elif self.field[self.pos] == '(':
                self.commentlist.append(self.getcomment())
            elif self.field[self.pos] in self.phraseends:
                break
            else:
                plist.append(self.getatom(self.phraseends))

        return plist

class AddressList(AddrlistClass):
    """An AddressList encapsulates a list of parsed RFC 2822 addresses."""
    def __init__(self, field):
        AddrlistClass.__init__(self, field)
        if field:
            self.addresslist = self.getaddrlist()
        else:
            self.addresslist = []

    def __len__(self):
        return len(self.addresslist)

    def __add__(self, other):
        # Set union
        newaddr = AddressList(None)
        newaddr.addresslist = self.addresslist[:]
        for x in other.addresslist:
            if not x in self.addresslist:
                newaddr.addresslist.append(x)
        return newaddr

    def __iadd__(self, other):
        # Set union, in-place
        for x in other.addresslist:
            if not x in self.addresslist:
                self.addresslist.append(x)
        return self

    def __sub__(self, other):
        # Set difference
        newaddr = AddressList(None)
        for x in self.addresslist:
            if not x in other.addresslist:
                newaddr.addresslist.append(x)
        return newaddr

    def __isub__(self, other):
        # Set difference, in-place
        for x in other.addresslist:
            if x in self.addresslist:
                self.addresslist.remove(x)
        return self

    def __getitem__(self, index):
        # Make indexing, slices, and 'in' work
        return self.addresslist[index]
<|endoftext|>"""Policy framework for the email package.

Allows fine grained feature control of how the package parses and emits data.
"""

import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates

__all__ = [
    'Policy',
    'Compat32',
    'compat32',
    ]


class _PolicyBase:

    """Policy Object basic framework.

    This class is useless unless subclassed.  A subclass should define
    class attributes with defaults for any values that are to be
    managed by the Policy object.  The constructor will then allow
    non-default values to be set for these attributes at instance
    creation time.  The instance will be callable, taking these same
    attributes keyword arguments, and returning a new instance
    identical to the called instance except for those values changed
    by the keyword arguments.  Instances may be added, yielding new
    instances with any non-default values from the right hand
    operand overriding those in the left hand operand.  That is,

        A + B == A(<non-default values of B>)

    The repr of an instance can be used to reconstruct the object
    if and only if the repr of the values can be used to reconstruct
    those values.

    """

    def __init__(self, **kw):
        """Create new Policy, possibly overriding some defaults.

        See class docstring for a list of overridable attributes.

        """
        for name, value in kw.items():
            if hasattr(self, name):
                super(_PolicyBase,self).__setattr__(name, value)
            else:
                raise TypeError(
                    "{!r} is an invalid keyword argument for {}".format(
                        name, self.__class__.__name__))

    def __repr__(self):
        args = [ "{}={!r}".format(name, value)
                 for name, value in self.__dict__.items() ]
        return "{}({})".format(self.__class__.__name__, ', '.join(args))

    def clone(self, **kw):
        """Return a new instance with specified attributes changed.

        The new instance has the same attribute values as the current object,
        except for the changes passed in as keyword arguments.

        """
        newpolicy = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            object.__setattr__(newpolicy, attr, value)
        for attr, value in kw.items():
            if not hasattr(self, attr):
                raise TypeError(
                    "{!r} is an invalid keyword argument for {}".format(
                        attr, self.__class__.__name__))
            object.__setattr__(newpolicy, attr, value)
        return newpolicy

    def __setattr__(self, name, value):
        if hasattr(self, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(self.__class__.__name__, name))

    def __add__(self, other):
        """Non-default values from right operand override those from left.

        The object returned is a new instance of the subclass.

        """
        return self.clone(**other.__dict__)


def _append_doc(doc, added_doc):
    doc = doc.rsplit('\n', 1)[0]
    added_doc = added_doc.split('\n', 1)[1]
    return doc + '\n' + added_doc

def _extend_docstrings(cls):
    if cls.__doc__ and cls.__doc__.startswith('+'):
        cls.__doc__ = _append_doc(cls.__bases__[0].__doc__, cls.__doc__)
    for name, attr in cls.__dict__.items():
        if attr.__doc__ and attr.__doc__.startswith('+'):
            for c in (c for base in cls.__bases__ for c in base.mro()):
                doc = getattr(getattr(c, name), '__doc__')
                if doc:
                    attr.__doc__ = _append_doc(doc, attr.__doc__)
                    break
    return cls


class Policy(_PolicyBase, metaclass=abc.ABCMeta):

    r"""Controls for how messages are interpreted and formatted.

    Most of the classes and many of the methods in the email package accept
    Policy objects as parameters.  A Policy object contains a set of values and
    functions that control how input is interpreted and how output is rendered.
    For example, the parameter 'raise_on_defect' controls whether or not an RFC
    violation results in an error being raised or not, while 'max_line_length'
    controls the maximum length of output lines when a Message is serialized.

    Any valid attribute may be overridden when a Policy is created by passing
    it as a keyword argument to the constructor.  Policy objects are immutable,
    but a new Policy object can be created with only certain values changed by
    calling the Policy instance with keyword arguments.  Policy objects can
    also be added, producing a new Policy object in which the non-default
    attributes set in the right hand operand overwrite those specified in the
    left operand.

    Settable attributes:

    raise_on_defect     -- If true, then defects should be raised as errors.
                           Default: False.

    linesep             -- string containing the value to use as separation
                           between output lines.  Default '\n'.

    cte_type            -- Type of allowed content transfer encodings

                           7bit  -- ASCII only
                           8bit  -- Content-Transfer-Encoding: 8bit is allowed

                           Default: 8bit.  Also controls the disposition of
                           (RFC invalid) binary data in headers; see the
                           documentation of the binary_fold method.

    max_line_length     -- maximum length of lines, excluding 'linesep',
                           during serialization.  None or 0 means no line
                           wrapping is done.  Default is 78.

    mangle_from_        -- a flag that, when True escapes From_ lines in the
                           body of the message by putting a `>' in front of
                           them. This is used when the message is being
                           serialized by a generator. Default: True.

    message_factory     -- the class to use to create new message objects.
                           If the value is None, the default is Message.

    verify_generated_headers
                        -- if true, the generator verifies that each header
                           they are properly folded, so that a parser won't
                           treat it as multiple headers, start-of-body, or
                           part of another header.
                           This is a check against custom Header & fold()
                           implementations.
    """

    raise_on_defect = False
    linesep = '\n'
    cte_type = '8bit'
    max_line_length = 78
    mangle_from_ = False
    message_factory = None
    verify_generated_headers = True

    def handle_defect(self, obj, defect):
        """Based on policy, either raise defect or call register_defect.

            handle_defect(obj, defect)

        defect should be a Defect subclass, but in any case must be an
        Exception subclass.  obj is the object on which the defect should be
        registered if it is not raised.  If the raise_on_defect is True, the
        defect is raised as an error, otherwise the object and the defect are
        passed to register_defect.

        This method is intended to be called by parsers that discover defects.
        The email package parsers always call it with Defect instances.

        """
        if self.raise_on_defect:
            raise defect
        self.register_defect(obj, defect)

    def register_defect(self, obj, defect):
        """Record 'defect' on 'obj'.

        Called by handle_defect if raise_on_defect is False.  This method is
        part of the Policy API so that Policy subclasses can implement custom
        defect handling.  The default implementation calls the append method of
        the defects attribute of obj.  The objects used by the email package by
        default that get passed to this method will always have a defects
        attribute with an append method.

        """
        obj.defects.append(defect)

    def header_max_count(self, name):
        """Return the maximum allowed number of headers named 'name'.

        Called when a header is added to a Message object.  If the returned
        value is not 0 or None, and there are already a number of headers with
        the name 'name' equal to the value returned, a ValueError is raised.

        Because the default behavior of Message's __setitem__ is to append the
        value to the list of headers, it is easy to create duplicate headers
        without realizing it.  This method allows certain headers to be limited
        in the number of instances of that header that may be added to a
        Message programmatically.  (The limit is not observed by the parser,
        which will faithfully produce as many headers as exist in the message
        being parsed.)

        The default implementation returns None for all header names.
        """
        return None

    @abc.abstractmethod
    def header_source_parse(self, sourcelines):
        """Given a list of linesep terminated strings constituting the lines of
        a single header, return the (name, value) tuple that should be stored
        in the model.  The input lines should retain their terminating linesep
        characters.  The lines passed in by the email package may contain
        surrogateescaped binary data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def header_store_parse(self, name, value):
        """Given the header name and the value provided by the application
        program, return the (name, value) that should be stored in the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def header_fetch_parse(self, name, value):
        """Given the header name and the value from the model, return the value
        to be returned to the application program that is requesting that
        header.  The value passed in by the email package may contain
        surrogateescaped binary data if the lines were parsed by a BytesParser.
        The returned value should not contain any surrogateescaped data.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def fold(self, name, value):
        """Given the header name and the value from the model, return a string
        containing linesep characters that implement the folding of the header
        according to the policy controls.  The value passed in by the email
        package may contain surrogateescaped binary data if the lines were
        parsed by a BytesParser.  The returned value should not contain any
        surrogateescaped data.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def fold_binary(self, name, value):
        """Given the header name and the value from the model, return binary
        data containing linesep characters that implement the folding of the
        header according to the policy controls.  The value passed in by the
        email package may contain surrogateescaped binary data.

        """
        raise NotImplementedError


@_extend_docstrings
class Compat32(Policy):

    """+
    This particular policy is the backward compatibility Policy.  It
    replicates the behavior of the email package version 5.1.
    """

    mangle_from_ = True

    def _sanitize_header(self, name, value):
        # If the header value contains surrogates, return a Header using
        # the unknown-8bit charset to encode the bytes as encoded words.
        if not isinstance(value, str):
            # Assume it is already a header object
            return value
        if _has_surrogates(value):
            return header.Header(value, charset=_charset.UNKNOWN8BIT,
                                 header_name=name)
        else:
            return value

    def header_source_parse(self, sourcelines):
        """+
        The name is parsed as everything up to the ':' and returned unmodified.
        The value is determined by stripping leading whitespace off the
        remainder of the first line, joining all subsequent lines together, and
        stripping any trailing carriage return or linefeed characters.

        """
        name, value = sourcelines[0].split(':', 1)
        value = value.lstrip(' \t') + ''.join(sourcelines[1:])
        return (name, value.rstrip('\r\n'))

    def header_store_parse(self, name, value):
        """+
        The name and value are returned unmodified.
        """
        return (name, value)

    def header_fetch_parse(self, name, value):
        """+
        If the value contains binary data, it is converted into a Header object
        using the unknown-8bit charset.  Otherwise it is returned unmodified.
        """
        return self._sanitize_header(name, value)

    def fold(self, name, value):
        """+
        Headers are folded using the Header folding algorithm, which preserves
        existing line breaks in the value, and wraps each resulting line to the
        max_line_length.  Non-ASCII binary data are CTE encoded using the
        unknown-8bit charset.

        """
        return self._fold(name, value, sanitize=True)

    def fold_binary(self, name, value):
        """+
        Headers are folded using the Header folding algorithm, which preserves
        existing line breaks in the value, and wraps each resulting line to the
        max_line_length.  If cte_type is 7bit, non-ascii binary data is CTE
        encoded using the unknown-8bit charset.  Otherwise the original source
        header is used, with its existing line breaks and/or binary data.

        """
        folded = self._fold(name, value, sanitize=self.cte_type=='7bit')
        return folded.encode('ascii', 'surrogateescape')

    def _fold(self, name, value, sanitize):
        parts = []
        parts.append('%s: ' % name)
        if isinstance(value, str):
            if _has_surrogates(value):
                if sanitize:
                    h = header.Header(value,
                                      charset=_charset.UNKNOWN8BIT,
                                      header_name=name)
                else:
                    # If we have raw 8bit data in a byte string, we have no idea
                    # what the encoding is.  There is no safe way to split this
                    # string.  If it's ascii-subset, then we could do a normal
                    # ascii split, but if it's multibyte then we could break the
                    # string.  There's no way to know so the least harm seems to
                    # be to not split the string and risk it being too long.
                    parts.append(value)
                    h = None
            else:
                h = header.Header(value, header_name=name)
        else:
            # Assume it is a Header-like object.
            h = value
        if h is not None:
            # The Header class interprets a value of None for maxlinelen as the
            # default value of 78, as recommended by RFC 2822.
            maxlinelen = 0
            if self.max_line_length is not None:
                maxlinelen = self.max_line_length
            parts.append(h.encode(linesep=self.linesep, maxlinelen=maxlinelen))
        parts.append(self.linesep)
        return ''.join(parts)


compat32 = Compat32()
<|endoftext|>"""Representing and manipulating email headers via custom objects.

This module provides an implementation of the HeaderRegistry API.
The implementation is designed to flexibly follow RFC5322 rules.
"""
from types import MappingProxyType

from email import utils
from email import errors
from email import _header_value_parser as parser

class Address:

    def __init__(self, display_name='', username='', domain='', addr_spec=None):
        """Create an object representing a full email address.

        An address can have a 'display_name', a 'username', and a 'domain'.  In
        addition to specifying the username and domain separately, they may be
        specified together by using the addr_spec keyword *instead of* the
        username and domain keywords.  If an addr_spec string is specified it
        must be properly quoted according to RFC 5322 rules; an error will be
        raised if it is not.

        An Address object has display_name, username, domain, and addr_spec
        attributes, all of which are read-only.  The addr_spec and the string
        value of the object are both quoted according to RFC5322 rules, but
        without any Content Transfer Encoding.

        """

        inputs = ''.join(filter(None, (display_name, username, domain, addr_spec)))
        if '\r' in inputs or '\n' in inputs:
            raise ValueError("invalid arguments; address parts cannot contain CR or LF")

        # This clause with its potential 'raise' may only happen when an
        # application program creates an Address object using an addr_spec
        # keyword.  The email library code itself must always supply username
        # and domain.
        if addr_spec is not None:
            if username or domain:
                raise TypeError("addrspec specified when username and/or "
                                "domain also specified")
            a_s, rest = parser.get_addr_spec(addr_spec)
            if rest:
                raise ValueError("Invalid addr_spec; only '{}' "
                                 "could be parsed from '{}'".format(
                                    a_s, addr_spec))
            if a_s.all_defects:
                raise a_s.all_defects[0]
            username = a_s.local_part
            domain = a_s.domain
        self._display_name = display_name
        self._username = username
        self._domain = domain

    @property
    def display_name(self):
        return self._display_name

    @property
    def username(self):
        return self._username

    @property
    def domain(self):
        return self._domain

    @property
    def addr_spec(self):
        """The addr_spec (username@domain) portion of the address, quoted
        according to RFC 5322 rules, but with no Content Transfer Encoding.
        """
        lp = self.username
        if not parser.DOT_ATOM_ENDS.isdisjoint(lp):
            lp = parser.quote_string(lp)
        if self.domain:
            return lp + '@' + self.domain
        if not lp:
            return '<>'
        return lp

    def __repr__(self):
        return "{}(display_name={!r}, username={!r}, domain={!r})".format(
                        self.__class__.__name__,
                        self.display_name, self.username, self.domain)

    def __str__(self):
        disp = self.display_name
        if not parser.SPECIALS.isdisjoint(disp):
            disp = parser.quote_string(disp)
        if disp:
            addr_spec = '' if self.addr_spec=='<>' else self.addr_spec
            return "{} <{}>".format(disp, addr_spec)
        return self.addr_spec

    def __eq__(self, other):
        if not isinstance(other, Address):
            return NotImplemented
        return (self.display_name == other.display_name and
                self.username == other.username and
                self.domain == other.domain)


class Group:

    def __init__(self, display_name=None, addresses=None):
        """Create an object representing an address group.

        An address group consists of a display_name followed by colon and a
        list of addresses (see Address) terminated by a semi-colon.  The Group
        is created by specifying a display_name and a possibly empty list of
        Address objects.  A Group can also be used to represent a single
        address that is not in a group, which is convenient when manipulating
        lists that are a combination of Groups and individual Addresses.  In
        this case the display_name should be set to None.  In particular, the
        string representation of a Group whose display_name is None is the same
        as the Address object, if there is one and only one Address object in
        the addresses list.

        """
        self._display_name = display_name
        self._addresses = tuple(addresses) if addresses else tuple()

    @property
    def display_name(self):
        return self._display_name

    @property
    def addresses(self):
        return self._addresses

    def __repr__(self):
        return "{}(display_name={!r}, addresses={!r}".format(
                 self.__class__.__name__,
                 self.display_name, self.addresses)

    def __str__(self):
        if self.display_name is None and len(self.addresses)==1:
            return str(self.addresses[0])
        disp = self.display_name
        if disp is not None and not parser.SPECIALS.isdisjoint(disp):
            disp = parser.quote_string(disp)
        adrstr = ", ".join(str(x) for x in self.addresses)
        adrstr = ' ' + adrstr if adrstr else adrstr
        return "{}:{};".format(disp, adrstr)

    def __eq__(self, other):
        if not isinstance(other, Group):
            return NotImplemented
        return (self.display_name == other.display_name and
                self.addresses == other.addresses)


# Header Classes #

class BaseHeader(str):

    """Base class for message headers.

    Implements generic behavior and provides tools for subclasses.

    A subclass must define a classmethod named 'parse' that takes an unfolded
    value string and a dictionary as its arguments.  The dictionary will
    contain one key, 'defects', initialized to an empty list.  After the call
    the dictionary must contain two additional keys: parse_tree, set to the
    parse tree obtained from parsing the header, and 'decoded', set to the
    string value of the idealized representation of the data from the value.
    (That is, encoded words are decoded, and values that have canonical
    representations are so represented.)

    The defects key is intended to collect parsing defects, which the message
    parser will subsequently dispose of as appropriate.  The parser should not,
    insofar as practical, raise any errors.  Defects should be added to the
    list instead.  The standard header parsers register defects for RFC
    compliance issues, for obsolete RFC syntax, and for unrecoverable parsing
    errors.

    The parse method may add additional keys to the dictionary.  In this case
    the subclass must define an 'init' method, which will be passed the
    dictionary as its keyword arguments.  The method should use (usually by
    setting them as the value of similarly named attributes) and remove all the
    extra keys added by its parse method, and then use super to call its parent
    class with the remaining arguments and keywords.

    The subclass should also make sure that a 'max_count' attribute is defined
    that is either None or 1. XXX: need to better define this API.

    """

    def __new__(cls, name, value):
        kwds = {'defects': []}
        cls.parse(value, kwds)
        if utils._has_surrogates(kwds['decoded']):
            kwds['decoded'] = utils._sanitize(kwds['decoded'])
        self = str.__new__(cls, kwds['decoded'])
        del kwds['decoded']
        self.init(name, **kwds)
        return self

    def init(self, name, *, parse_tree, defects):
        self._name = name
        self._parse_tree = parse_tree
        self._defects = defects

    @property
    def name(self):
        return self._name

    @property
    def defects(self):
        return tuple(self._defects)

    def __reduce__(self):
        return (
            _reconstruct_header,
            (
                self.__class__.__name__,
                self.__class__.__bases__,
                str(self),
            ),
            self.__dict__)

    @classmethod
    def _reconstruct(cls, value):
        return str.__new__(cls, value)

    def fold(self, *, policy):
        """Fold header according to policy.

        The parsed representation of the header is folded according to
        RFC5322 rules, as modified by the policy.  If the parse tree
        contains surrogateescaped bytes, the bytes are CTE encoded using
        the charset 'unknown-8bit".

        Any non-ASCII characters in the parse tree are CTE encoded using
        charset utf-8. XXX: make this a policy setting.

        The returned value is an ASCII-only string possibly containing linesep
        characters, and ending with a linesep character.  The string includes
        the header name and the ': ' separator.

        """
        # At some point we need to put fws here if it was in the source.
        header = parser.Header([
            parser.HeaderLabel([
                parser.ValueTerminal(self.name, 'header-name'),
                parser.ValueTerminal(':', 'header-sep')]),
            ])
        if self._parse_tree:
            header.append(
                parser.CFWSList([parser.WhiteSpaceTerminal(' ', 'fws')]))
        header.append(self._parse_tree)
        return header.fold(policy=policy)


def _reconstruct_header(cls_name, bases, value):
    return type(cls_name, bases, {})._reconstruct(value)


class UnstructuredHeader:

    max_count = None
    value_parser = staticmethod(parser.get_unstructured)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = cls.value_parser(value)
        kwds['decoded'] = str(kwds['parse_tree'])


class UniqueUnstructuredHeader(UnstructuredHeader):

    max_count = 1


class DateHeader:

    """Header whose value consists of a single timestamp.

    Provides an additional attribute, datetime, which is either an aware
    datetime using a timezone, or a naive datetime if the timezone
    in the input string is -0000.  Also accepts a datetime as input.
    The 'value' attribute is the normalized form of the timestamp,
    which means it is the output of format_datetime on the datetime.
    """

    max_count = None

    # This is used only for folding, not for creating 'decoded'.
    value_parser = staticmethod(parser.get_unstructured)

    @classmethod
    def parse(cls, value, kwds):
        if not value:
            kwds['defects'].append(errors.HeaderMissingRequiredValue())
            kwds['datetime'] = None
            kwds['decoded'] = ''
            kwds['parse_tree'] = parser.TokenList()
            return
        if isinstance(value, str):
            kwds['decoded'] = value
            try:
                value = utils.parsedate_to_datetime(value)
            except ValueError:
                kwds['defects'].append(errors.InvalidDateDefect('Invalid date value or format'))
                kwds['datetime'] = None
                kwds['parse_tree'] = parser.TokenList()
                return
        kwds['datetime'] = value
        kwds['decoded'] = utils.format_datetime(kwds['datetime'])
        kwds['parse_tree'] = cls.value_parser(kwds['decoded'])

    def init(self, *args, **kw):
        self._datetime = kw.pop('datetime')
        super().init(*args, **kw)

    @property
    def datetime(self):
        return self._datetime


class UniqueDateHeader(DateHeader):

    max_count = 1


class AddressHeader:

    max_count = None

    @staticmethod
    def value_parser(value):
        address_list, value = parser.get_address_list(value)
        assert not value, 'this should not happen'
        return address_list

    @classmethod
    def parse(cls, value, kwds):
        if isinstance(value, str):
            # We are translating here from the RFC language (address/mailbox)
            # to our API language (group/address).
            kwds['parse_tree'] = address_list = cls.value_parser(value)
            groups = []
            for addr in address_list.addresses:
                groups.append(Group(addr.display_name,
                                    [Address(mb.display_name or '',
                                             mb.local_part or '',
                                             mb.domain or '')
                                     for mb in addr.all_mailboxes]))
            defects = list(address_list.all_defects)
        else:
            # Assume it is Address/Group stuff
            if not hasattr(value, '__iter__'):
                value = [value]
            groups = [Group(None, [item]) if not hasattr(item, 'addresses')
                                          else item
                                    for item in value]
            defects = []
        kwds['groups'] = groups
        kwds['defects'] = defects
        kwds['decoded'] = ', '.join([str(item) for item in groups])
        if 'parse_tree' not in kwds:
            kwds['parse_tree'] = cls.value_parser(kwds['decoded'])

    def init(self, *args, **kw):
        self._groups = tuple(kw.pop('groups'))
        self._addresses = None
        super().init(*args, **kw)

    @property
    def groups(self):
        return self._groups

    @property
    def addresses(self):
        if self._addresses is None:
            self._addresses = tuple(address for group in self._groups
                                            for address in group.addresses)
        return self._addresses


class UniqueAddressHeader(AddressHeader):

    max_count = 1


class SingleAddressHeader(AddressHeader):

    @property
    def address(self):
        if len(self.addresses)!=1:
            raise ValueError(("value of single address header {} is not "
                "a single address").format(self.name))
        return self.addresses[0]


class UniqueSingleAddressHeader(SingleAddressHeader):

    max_count = 1


class MIMEVersionHeader:

    max_count = 1

    value_parser = staticmethod(parser.parse_mime_version)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)
        kwds['major'] = None if parse_tree.minor is None else parse_tree.major
        kwds['minor'] = parse_tree.minor
        if parse_tree.minor is not None:
            kwds['version'] = '{}.{}'.format(kwds['major'], kwds['minor'])
        else:
            kwds['version'] = None

    def init(self, *args, **kw):
        self._version = kw.pop('version')
        self._major = kw.pop('major')
        self._minor = kw.pop('minor')
        super().init(*args, **kw)

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def version(self):
        return self._version


class ParameterizedMIMEHeader:

    # Mixin that handles the params dict.  Must be subclassed and
    # a property value_parser for the specific header provided.

    max_count = 1

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)
        if parse_tree.params is None:
            kwds['params'] = {}
        else:
            # The MIME RFCs specify that parameter ordering is arbitrary.
            kwds['params'] = {utils._sanitize(name).lower():
                                    utils._sanitize(value)
                               for name, value in parse_tree.params}

    def init(self, *args, **kw):
        self._params = kw.pop('params')
        super().init(*args, **kw)

    @property
    def params(self):
        return MappingProxyType(self._params)


class ContentTypeHeader(ParameterizedMIMEHeader):

    value_parser = staticmethod(parser.parse_content_type_header)

    def init(self, *args, **kw):
        super().init(*args, **kw)
        self._maintype = utils._sanitize(self._parse_tree.maintype)
        self._subtype = utils._sanitize(self._parse_tree.subtype)

    @property
    def maintype(self):
        return self._maintype

    @property
    def subtype(self):
        return self._subtype

    @property
    def content_type(self):
        return self.maintype + '/' + self.subtype


class ContentDispositionHeader(ParameterizedMIMEHeader):

    value_parser = staticmethod(parser.parse_content_disposition_header)

    def init(self, *args, **kw):
        super().init(*args, **kw)
        cd = self._parse_tree.content_disposition
        self._content_disposition = cd if cd is None else utils._sanitize(cd)

    @property
    def content_disposition(self):
        return self._content_disposition


class ContentTransferEncodingHeader:

    max_count = 1

    value_parser = staticmethod(parser.parse_content_transfer_encoding_header)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)

    def init(self, *args, **kw):
        super().init(*args, **kw)
        self._cte = utils._sanitize(self._parse_tree.cte)

    @property
    def cte(self):
        return self._cte


class MessageIDHeader:

    max_count = 1
    value_parser = staticmethod(parser.parse_message_id)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)


# The header factory #

_default_header_map = {
    'subject':                      UniqueUnstructuredHeader,
    'date':                         UniqueDateHeader,
    'resent-date':                  DateHeader,
    'orig-date':                    UniqueDateHeader,
    'sender':                       UniqueSingleAddressHeader,
    'resent-sender':                SingleAddressHeader,
    'to':                           UniqueAddressHeader,
    'resent-to':                    AddressHeader,
    'cc':                           UniqueAddressHeader,
    'resent-cc':                    AddressHeader,
    'bcc':                          UniqueAddressHeader,
    'resent-bcc':                   AddressHeader,
    'from':                         UniqueAddressHeader,
    'resent-from':                  AddressHeader,
    'reply-to':                     UniqueAddressHeader,
    'mime-version':                 MIMEVersionHeader,
    'content-type':                 ContentTypeHeader,
    'content-disposition':          ContentDispositionHeader,
    'content-transfer-encoding':    ContentTransferEncodingHeader,
    'message-id':                   MessageIDHeader,
    }

class HeaderRegistry:

    """A header_factory and header registry."""

    def __init__(self, base_class=BaseHeader, default_class=UnstructuredHeader,
                       use_default_map=True):
        """Create a header_factory that works with the Policy API.

        base_class is the class that will be the last class in the created
        header class's __bases__ list.  default_class is the class that will be
        used if "name" (see __call__) does not appear in the registry.
        use_default_map controls whether or not the default mapping of names to
        specialized classes is copied in to the registry when the factory is
        created.  The default is True.

        """
        self.registry = {}
        self.base_class = base_class
        self.default_class = default_class
        if use_default_map:
            self.registry.update(_default_header_map)

    def map_to_type(self, name, cls):
        """Register cls as the specialized class for handling "name" headers.

        """
        self.registry[name.lower()] = cls

    def __getitem__(self, name):
        cls = self.registry.get(name.lower(), self.default_class)
        return type('_'+cls.__name__, (cls, self.base_class), {})

    def __call__(self, name, value):
        """Create a header instance for header 'name' from 'value'.

        Creates a header instance by creating a specialized class for parsing
        and representing the specified header by combining the factory
        base_class with a specialized class from the registry or the
        default_class, and passing the name and value to the constructed
        class's constructor.

        """
        return self[name](name, value)
<|endoftext|># Copyright (C) 2001-2010 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Classes to generate plain text from a message object tree."""

__all__ = ['Generator', 'DecodedGenerator', 'BytesGenerator']

import re
import sys
import time
import random

from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
from email.errors import HeaderWriteError

UNDERSCORE = '_'
NL = '\n'  # XXX: no longer used by the code below.

NLCRE = re.compile(r'\r\n|\r|\n')
fcre = re.compile(r'^From ', re.MULTILINE)
NEWLINE_WITHOUT_FWSP = re.compile(r'\r\n[^ \t]|\r[^ \n\t]|\n[^ \t]')
NEWLINE_WITHOUT_FWSP_BYTES = re.compile(br'\r\n[^ \t]|\r[^ \n\t]|\n[^ \t]')



class Generator:
    """Generates output from a Message object tree.

    This basic generator writes the message to the given file object as plain
    text.
    """
    #
    # Public interface
    #

    def __init__(self, outfp, mangle_from_=None, maxheaderlen=None, *,
                 policy=None):
        """Create the generator for message flattening.

        outfp is the output file-like object for writing the message to.  It
        must have a write() method.

        Optional mangle_from_ is a flag that, when True (the default if policy
        is not set), escapes From_ lines in the body of the message by putting
        a `>' in front of them.

        Optional maxheaderlen specifies the longest length for a non-continued
        header.  When a header line is longer (in characters, with tabs
        expanded to 8 spaces) than maxheaderlen, the header will split as
        defined in the Header class.  Set maxheaderlen to zero to disable
        header wrapping.  The default is 78, as recommended (but not required)
        by RFC 2822.

        The policy keyword specifies a policy object that controls a number of
        aspects of the generator's operation.  If no policy is specified,
        the policy associated with the Message object passed to the
        flatten method is used.

        """

        if mangle_from_ is None:
            mangle_from_ = True if policy is None else policy.mangle_from_
        self._fp = outfp
        self._mangle_from_ = mangle_from_
        self.maxheaderlen = maxheaderlen
        self.policy = policy

    def write(self, s):
        # Just delegate to the file object
        self._fp.write(s)

    def flatten(self, msg, unixfrom=False, linesep=None):
        r"""Print the message object tree rooted at msg to the output file
        specified when the Generator instance was created.

        unixfrom is a flag that forces the printing of a Unix From_ delimiter
        before the first object in the message tree.  If the original message
        has no From_ delimiter, a `standard' one is crafted.  By default, this
        is False to inhibit the printing of any From_ delimiter.

        Note that for subobjects, no From_ line is printed.

        linesep specifies the characters used to indicate a new line in
        the output.  The default value is determined by the policy specified
        when the Generator instance was created or, if none was specified,
        from the policy associated with the msg.

        """
        # We use the _XXX constants for operating on data that comes directly
        # from the msg, and _encoded_XXX constants for operating on data that
        # has already been converted (to bytes in the BytesGenerator) and
        # inserted into a temporary buffer.
        policy = msg.policy if self.policy is None else self.policy
        if linesep is not None:
            policy = policy.clone(linesep=linesep)
        if self.maxheaderlen is not None:
            policy = policy.clone(max_line_length=self.maxheaderlen)
        self._NL = policy.linesep
        self._encoded_NL = self._encode(self._NL)
        self._EMPTY = ''
        self._encoded_EMPTY = self._encode(self._EMPTY)
        # Because we use clone (below) when we recursively process message
        # subparts, and because clone uses the computed policy (not None),
        # submessages will automatically get set to the computed policy when
        # they are processed by this code.
        old_gen_policy = self.policy
        old_msg_policy = msg.policy
        try:
            self.policy = policy
            msg.policy = policy
            if unixfrom:
                ufrom = msg.get_unixfrom()
                if not ufrom:
                    ufrom = 'From nobody ' + time.ctime(time.time())
                self.write(ufrom + self._NL)
            self._write(msg)
        finally:
            self.policy = old_gen_policy
            msg.policy = old_msg_policy

    def clone(self, fp):
        """Clone this generator with the exact same options."""
        return self.__class__(fp,
                              self._mangle_from_,
                              None, # Use policy setting, which we've adjusted
                              policy=self.policy)

    #
    # Protected interface - undocumented ;/
    #

    # Note that we use 'self.write' when what we are writing is coming from
    # the source, and self._fp.write when what we are writing is coming from a
    # buffer (because the Bytes subclass has already had a chance to transform
    # the data in its write method in that case).  This is an entirely
    # pragmatic split determined by experiment; we could be more general by
    # always using write and having the Bytes subclass write method detect when
    # it has already transformed the input; but, since this whole thing is a
    # hack anyway this seems good enough.

    def _new_buffer(self):
        # BytesGenerator overrides this to return BytesIO.
        return StringIO()

    def _encode(self, s):
        # BytesGenerator overrides this to encode strings to bytes.
        return s

    def _write_lines(self, lines):
        # We have to transform the line endings.
        if not lines:
            return
        lines = NLCRE.split(lines)
        for line in lines[:-1]:
            self.write(line)
            self.write(self._NL)
        if lines[-1]:
            self.write(lines[-1])
        # XXX logic tells me this else should be needed, but the tests fail
        # with it and pass without it.  (NLCRE.split ends with a blank element
        # if and only if there was a trailing newline.)
        #else:
        #    self.write(self._NL)

    def _write(self, msg):
        # We can't write the headers yet because of the following scenario:
        # say a multipart message includes the boundary string somewhere in
        # its body.  We'd have to calculate the new boundary /before/ we write
        # the headers so that we can write the correct Content-Type:
        # parameter.
        #
        # The way we do this, so as to make the _handle_*() methods simpler,
        # is to cache any subpart writes into a buffer.  The we write the
        # headers and the buffer contents.  That way, subpart handlers can
        # Do The Right Thing, and can still modify the Content-Type: header if
        # necessary.
        oldfp = self._fp
        try:
            self._munge_cte = None
            self._fp = sfp = self._new_buffer()
            self._dispatch(msg)
        finally:
            self._fp = oldfp
            munge_cte = self._munge_cte
            del self._munge_cte
        # If we munged the cte, copy the message again and re-fix the CTE.
        if munge_cte:
            msg = deepcopy(msg)
            # Preserve the header order if the CTE header already exists.
            if msg.get('content-transfer-encoding') is None:
                msg['Content-Transfer-Encoding'] = munge_cte[0]
            else:
                msg.replace_header('content-transfer-encoding', munge_cte[0])
            msg.replace_header('content-type', munge_cte[1])
        # Write the headers.  First we see if the message object wants to
        # handle that itself.  If not, we'll do it generically.
        meth = getattr(msg, '_write_headers', None)
        if meth is None:
            self._write_headers(msg)
        else:
            meth(self)
        self._fp.write(sfp.getvalue())

    def _dispatch(self, msg):
        # Get the Content-Type: for the message, then try to dispatch to
        # self._handle_<maintype>_<subtype>().  If there's no handler for the
        # full MIME type, then dispatch to self._handle_<maintype>().  If
        # that's missing too, then dispatch to self._writeBody().
        main = msg.get_content_maintype()
        sub = msg.get_content_subtype()
        specific = UNDERSCORE.join((main, sub)).replace('-', '_')
        meth = getattr(self, '_handle_' + specific, None)
        if meth is None:
            generic = main.replace('-', '_')
            meth = getattr(self, '_handle_' + generic, None)
            if meth is None:
                meth = self._writeBody
        meth(msg)

    #
    # Default handlers
    #

    def _write_headers(self, msg):
        for h, v in msg.raw_items():
            folded = self.policy.fold(h, v)
            if self.policy.verify_generated_headers:
                linesep = self.policy.linesep
                if not folded.endswith(self.policy.linesep):
                    raise HeaderWriteError(
                        f'folded header does not end with {linesep!r}: {folded!r}')
                if NEWLINE_WITHOUT_FWSP.search(folded.removesuffix(linesep)):
                    raise HeaderWriteError(
                        f'folded header contains newline: {folded!r}')
            self.write(folded)
        # A blank line always separates headers from body
        self.write(self._NL)

    #
    # Handlers for writing types and subtypes
    #

    def _handle_text(self, msg):
        payload = msg.get_payload()
        if payload is None:
            return
        if not isinstance(payload, str):
            raise TypeError('string payload expected: %s' % type(payload))
        if _has_surrogates(msg._payload):
            charset = msg.get_param('charset')
            if charset is not None:
                # XXX: This copy stuff is an ugly hack to avoid modifying the
                # existing message.
                msg = deepcopy(msg)
                del msg['content-transfer-encoding']
                msg.set_payload(payload, charset)
                payload = msg.get_payload()
                self._munge_cte = (msg['content-transfer-encoding'],
                                   msg['content-type'])
        if self._mangle_from_:
            payload = fcre.sub('>From ', payload)
        self._write_lines(payload)

    # Default body handler
    _writeBody = _handle_text

    def _handle_multipart(self, msg):
        # The trick here is to write out each part separately, merge them all
        # together, and then make sure that the boundary we've chosen isn't
        # present in the payload.
        msgtexts = []
        subparts = msg.get_payload()
        if subparts is None:
            subparts = []
        elif isinstance(subparts, str):
            # e.g. a non-strict parse of a message with no starting boundary.
            self.write(subparts)
            return
        elif not isinstance(subparts, list):
            # Scalar payload
            subparts = [subparts]
        for part in subparts:
            s = self._new_buffer()
            g = self.clone(s)
            g.flatten(part, unixfrom=False, linesep=self._NL)
            msgtexts.append(s.getvalue())
        # BAW: What about boundaries that are wrapped in double-quotes?
        boundary = msg.get_boundary()
        if not boundary:
            # Create a boundary that doesn't appear in any of the
            # message texts.
            alltext = self._encoded_NL.join(msgtexts)
            boundary = self._make_boundary(alltext)
            msg.set_boundary(boundary)
        # If there's a preamble, write it out, with a trailing CRLF
        if msg.preamble is not None:
            if self._mangle_from_:
                preamble = fcre.sub('>From ', msg.preamble)
            else:
                preamble = msg.preamble
            self._write_lines(preamble)
            self.write(self._NL)
        # dash-boundary transport-padding CRLF
        self.write('--' + boundary + self._NL)
        # body-part
        if msgtexts:
            self._fp.write(msgtexts.pop(0))
        # *encapsulation
        # --> delimiter transport-padding
        # --> CRLF body-part
        for body_part in msgtexts:
            # delimiter transport-padding CRLF
            self.write(self._NL + '--' + boundary + self._NL)
            # body-part
            self._fp.write(body_part)
        # close-delimiter transport-padding
        self.write(self._NL + '--' + boundary + '--' + self._NL)
        if msg.epilogue is not None:
            if self._mangle_from_:
                epilogue = fcre.sub('>From ', msg.epilogue)
            else:
                epilogue = msg.epilogue
            self._write_lines(epilogue)

    def _handle_multipart_signed(self, msg):
        # The contents of signed parts has to stay unmodified in order to keep
        # the signature intact per RFC1847 2.1, so we disable header wrapping.
        # RDM: This isn't enough to completely preserve the part, but it helps.
        p = self.policy
        self.policy = p.clone(max_line_length=0)
        try:
            self._handle_multipart(msg)
        finally:
            self.policy = p

    def _handle_message_delivery_status(self, msg):
        # We can't just write the headers directly to self's file object
        # because this will leave an extra newline between the last header
        # block and the boundary.  Sigh.
        blocks = []
        for part in msg.get_payload():
            s = self._new_buffer()
            g = self.clone(s)
            g.flatten(part, unixfrom=False, linesep=self._NL)
            text = s.getvalue()
            lines = text.split(self._encoded_NL)
            # Strip off the unnecessary trailing empty line
            if lines and lines[-1] == self._encoded_EMPTY:
                blocks.append(self._encoded_NL.join(lines[:-1]))
            else:
                blocks.append(text)
        # Now join all the blocks with an empty line.  This has the lovely
        # effect of separating each block with an empty line, but not adding
        # an extra one after the last one.
        self._fp.write(self._encoded_NL.join(blocks))

    def _handle_message(self, msg):
        s = self._new_buffer()
        g = self.clone(s)
        # The payload of a message/rfc822 part should be a multipart sequence
        # of length 1.  The zeroth element of the list should be the Message
        # object for the subpart.  Extract that object, stringify it, and
        # write it out.
        # Except, it turns out, when it's a string instead, which happens when
        # and only when HeaderParser is used on a message of mime type
        # message/rfc822.  Such messages are generated by, for example,
        # Groupwise when forwarding unadorned messages.  (Issue 7970.)  So
        # in that case we just emit the string body.
        payload = msg._payload
        if isinstance(payload, list):
            g.flatten(msg.get_payload(0), unixfrom=False, linesep=self._NL)
            payload = s.getvalue()
        else:
            payload = self._encode(payload)
        self._fp.write(payload)

    # This used to be a module level function; we use a classmethod for this
    # and _compile_re so we can continue to provide the module level function
    # for backward compatibility by doing
    #   _make_boundary = Generator._make_boundary
    # at the end of the module.  It *is* internal, so we could drop that...
    @classmethod
    def _make_boundary(cls, text=None):
        # Craft a random boundary.  If text is given, ensure that the chosen
        # boundary doesn't appear in the text.
        token = random.randrange(sys.maxsize)
        boundary = ('=' * 15) + (_fmt % token) + '=='
        if text is None:
            return boundary
        b = boundary
        counter = 0
        while True:
            cre = cls._compile_re('^--' + re.escape(b) + '(--)?$', re.MULTILINE)
            if not cre.search(text):
                break
            b = boundary + '.' + str(counter)
            counter += 1
        return b

    @classmethod
    def _compile_re(cls, s, flags):
        return re.compile(s, flags)


class BytesGenerator(Generator):
    """Generates a bytes version of a Message object tree.

    Functionally identical to the base Generator except that the output is
    bytes and not string.  When surrogates were used in the input to encode
    bytes, these are decoded back to bytes for output.  If the policy has
    cte_type set to 7bit, then the message is transformed such that the
    non-ASCII bytes are properly content transfer encoded, using the charset
    unknown-8bit.

    The outfp object must accept bytes in its write method.
    """

    def write(self, s):
        self._fp.write(s.encode('ascii', 'surrogateescape'))

    def _new_buffer(self):
        return BytesIO()

    def _encode(self, s):
        return s.encode('ascii')

    def _write_headers(self, msg):
        # This is almost the same as the string version, except for handling
        # strings with 8bit bytes.
        for h, v in msg.raw_items():
            folded = self.policy.fold_binary(h, v)
            if self.policy.verify_generated_headers:
                linesep = self.policy.linesep.encode()
                if not folded.endswith(linesep):
                    raise HeaderWriteError(
                        f'folded header does not end with {linesep!r}: {folded!r}')
                if NEWLINE_WITHOUT_FWSP_BYTES.search(folded.removesuffix(linesep)):
                    raise HeaderWriteError(
                        f'folded header contains newline: {folded!r}')
            self._fp.write(folded)
        # A blank line always separates headers from body
        self.write(self._NL)

    def _handle_text(self, msg):
        # If the string has surrogates the original source was bytes, so
        # just write it back out.
        if msg._payload is None:
            return
        if _has_surrogates(msg._payload) and not self.policy.cte_type=='7bit':
            if self._mangle_from_:
                msg._payload = fcre.sub(">From ", msg._payload)
            self._write_lines(msg._payload)
        else:
            super(BytesGenerator,self)._handle_text(msg)

    # Default body handler
    _writeBody = _handle_text

    @classmethod
    def _compile_re(cls, s, flags):
        return re.compile(s.encode('ascii'), flags)



_FMT = '[Non-text (%(type)s) part of message omitted, filename %(filename)s]'

class DecodedGenerator(Generator):
    """Generates a text representation of a message.

    Like the Generator base class, except that non-text parts are substituted
    with a format string representing the part.
    """
    def __init__(self, outfp, mangle_from_=None, maxheaderlen=None, fmt=None, *,
                 policy=None):
        """Like Generator.__init__() except that an additional optional
        argument is allowed.

        Walks through all subparts of a message.  If the subpart is of main
        type `text', then it prints the decoded payload of the subpart.

