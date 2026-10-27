        Otherwise, fmt is a format string that is used instead of the message
        payload.  fmt is expanded with the following keywords (in
        %(keyword)s format):

        type       : Full MIME type of the non-text part
        maintype   : Main MIME type of the non-text part
        subtype    : Sub-MIME type of the non-text part
        filename   : Filename of the non-text part
        description: Description associated with the non-text part
        encoding   : Content transfer encoding of the non-text part

        The default value for fmt is None, meaning

        [Non-text (%(type)s) part of message omitted, filename %(filename)s]
        """
        Generator.__init__(self, outfp, mangle_from_, maxheaderlen,
                           policy=policy)
        if fmt is None:
            self._fmt = _FMT
        else:
            self._fmt = fmt

    def _dispatch(self, msg):
        for part in msg.walk():
            maintype = part.get_content_maintype()
            if maintype == 'text':
                print(part.get_payload(decode=False), file=self)
            elif maintype == 'multipart':
                # Just skip this
                pass
            else:
                print(self._fmt % {
                    'type'       : part.get_content_type(),
                    'maintype'   : part.get_content_maintype(),
                    'subtype'    : part.get_content_subtype(),
                    'filename'   : part.get_filename('[no filename]'),
                    'description': part.get('Content-Description',
                                            '[no description]'),
                    'encoding'   : part.get('Content-Transfer-Encoding',
                                            '[no encoding]'),
                    }, file=self)



# Helper used by Generator._make_boundary
_width = len(repr(sys.maxsize-1))
_fmt = '%%0%dd' % _width

# Backward compatibility
_make_boundary = Generator._make_boundary
<|endoftext|># Copyright (C) 2001-2007 Python Software Foundation
# Author: Barry Warsaw, Thomas Wouters, Anthony Baxter
# Contact: email-sig@python.org

"""A parser of RFC 2822 and MIME email messages."""

__all__ = ['Parser', 'HeaderParser', 'BytesParser', 'BytesHeaderParser',
           'FeedParser', 'BytesFeedParser']

from io import StringIO, TextIOWrapper

from email.feedparser import FeedParser, BytesFeedParser
from email._policybase import compat32


class Parser:
    def __init__(self, _class=None, *, policy=compat32):
        """Parser of RFC 2822 and MIME email messages.

        Creates an in-memory object tree representing the email message, which
        can then be manipulated and turned over to a Generator to return the
        textual representation of the message.

        The string must be formatted as a block of RFC 2822 headers and header
        continuation lines, optionally preceded by a `Unix-from' header.  The
        header block is terminated either by the end of the string or by a
        blank line.

        _class is the class to instantiate for new message objects when they
        must be created.  This class must have a constructor that can take
        zero arguments.  Default is Message.Message.

        The policy keyword specifies a policy object that controls a number of
        aspects of the parser's operation.  The default policy maintains
        backward compatibility.

        """
        self._class = _class
        self.policy = policy

    def parse(self, fp, headersonly=False):
        """Create a message structure from the data in a file.

        Reads all the data from the file and returns the root of the message
        structure.  Optional headersonly is a flag specifying whether to stop
        parsing after reading the headers or not.  The default is False,
        meaning it parses the entire contents of the file.
        """
        feedparser = FeedParser(self._class, policy=self.policy)
        if headersonly:
            feedparser._set_headersonly()
        while True:
            data = fp.read(8192)
            if not data:
                break
            feedparser.feed(data)
        return feedparser.close()

    def parsestr(self, text, headersonly=False):
        """Create a message structure from a string.

        Returns the root of the message structure.  Optional headersonly is a
        flag specifying whether to stop parsing after reading the headers or
        not.  The default is False, meaning it parses the entire contents of
        the file.
        """
        return self.parse(StringIO(text), headersonly=headersonly)



class HeaderParser(Parser):
    def parse(self, fp, headersonly=True):
        return Parser.parse(self, fp, True)

    def parsestr(self, text, headersonly=True):
        return Parser.parsestr(self, text, True)


class BytesParser:

    def __init__(self, *args, **kw):
        """Parser of binary RFC 2822 and MIME email messages.

        Creates an in-memory object tree representing the email message, which
        can then be manipulated and turned over to a Generator to return the
        textual representation of the message.

        The input must be formatted as a block of RFC 2822 headers and header
        continuation lines, optionally preceded by a `Unix-from' header.  The
        header block is terminated either by the end of the input or by a
        blank line.

        _class is the class to instantiate for new message objects when they
        must be created.  This class must have a constructor that can take
        zero arguments.  Default is Message.Message.
        """
        self.parser = Parser(*args, **kw)

    def parse(self, fp, headersonly=False):
        """Create a message structure from the data in a binary file.

        Reads all the data from the file and returns the root of the message
        structure.  Optional headersonly is a flag specifying whether to stop
        parsing after reading the headers or not.  The default is False,
        meaning it parses the entire contents of the file.
        """
        fp = TextIOWrapper(fp, encoding='ascii', errors='surrogateescape')
        try:
            return self.parser.parse(fp, headersonly)
        finally:
            fp.detach()


    def parsebytes(self, text, headersonly=False):
        """Create a message structure from a byte string.

        Returns the root of the message structure.  Optional headersonly is a
        flag specifying whether to stop parsing after reading the headers or
        not.  The default is False, meaning it parses the entire contents of
        the file.
        """
        text = text.decode('ASCII', errors='surrogateescape')
        return self.parser.parsestr(text, headersonly)


class BytesHeaderParser(BytesParser):
    def parse(self, fp, headersonly=True):
        return BytesParser.parse(self, fp, headersonly=True)

    def parsebytes(self, text, headersonly=True):
        return BytesParser.parsebytes(self, text, headersonly=True)
<|endoftext|># Copyright (C) 2001-2007 Python Software Foundation
# Author: Ben Gertzfield, Barry Warsaw
# Contact: email-sig@python.org

__all__ = [
    'Charset',
    'add_alias',
    'add_charset',
    'add_codec',
    ]

from functools import partial

import email.base64mime
import email.quoprimime

from email import errors
from email.encoders import encode_7or8bit



# Flags for types of header encodings
QP          = 1 # Quoted-Printable
BASE64      = 2 # Base64
SHORTEST    = 3 # the shorter of QP and base64, but only for headers

# In "=?charset?q?hello_world?=", the =?, ?q?, and ?= add up to 7
RFC2047_CHROME_LEN = 7

DEFAULT_CHARSET = 'us-ascii'
UNKNOWN8BIT = 'unknown-8bit'
EMPTYSTRING = ''



# Defaults
CHARSETS = {
    # input        header enc  body enc output conv
    'iso-8859-1':  (QP,        QP,      None),
    'iso-8859-2':  (QP,        QP,      None),
    'iso-8859-3':  (QP,        QP,      None),
    'iso-8859-4':  (QP,        QP,      None),
    # iso-8859-5 is Cyrillic, and not especially used
    # iso-8859-6 is Arabic, also not particularly used
    # iso-8859-7 is Greek, QP will not make it readable
    # iso-8859-8 is Hebrew, QP will not make it readable
    'iso-8859-9':  (QP,        QP,      None),
    'iso-8859-10': (QP,        QP,      None),
    # iso-8859-11 is Thai, QP will not make it readable
    'iso-8859-13': (QP,        QP,      None),
    'iso-8859-14': (QP,        QP,      None),
    'iso-8859-15': (QP,        QP,      None),
    'iso-8859-16': (QP,        QP,      None),
    'windows-1252':(QP,        QP,      None),
    'viscii':      (QP,        QP,      None),
    'us-ascii':    (None,      None,    None),
    'big5':        (BASE64,    BASE64,  None),
    'gb2312':      (BASE64,    BASE64,  None),
    'euc-jp':      (BASE64,    None,    'iso-2022-jp'),
    'shift_jis':   (BASE64,    None,    'iso-2022-jp'),
    'iso-2022-jp': (BASE64,    None,    None),
    'koi8-r':      (BASE64,    BASE64,  None),
    'utf-8':       (SHORTEST,  BASE64, 'utf-8'),
    }

# Aliases for other commonly-used names for character sets.  Map
# them to the real ones used in email.
ALIASES = {
    'latin_1': 'iso-8859-1',
    'latin-1': 'iso-8859-1',
    'latin_2': 'iso-8859-2',
    'latin-2': 'iso-8859-2',
    'latin_3': 'iso-8859-3',
    'latin-3': 'iso-8859-3',
    'latin_4': 'iso-8859-4',
    'latin-4': 'iso-8859-4',
    'latin_5': 'iso-8859-9',
    'latin-5': 'iso-8859-9',
    'latin_6': 'iso-8859-10',
    'latin-6': 'iso-8859-10',
    'latin_7': 'iso-8859-13',
    'latin-7': 'iso-8859-13',
    'latin_8': 'iso-8859-14',
    'latin-8': 'iso-8859-14',
    'latin_9': 'iso-8859-15',
    'latin-9': 'iso-8859-15',
    'latin_10':'iso-8859-16',
    'latin-10':'iso-8859-16',
    'cp949':   'ks_c_5601-1987',
    'euc_jp':  'euc-jp',
    'euc_kr':  'euc-kr',
    'ascii':   'us-ascii',
    }


# Map charsets to their Unicode codec strings.
CODEC_MAP = {
    'gb2312':      'eucgb2312_cn',
    'big5':        'big5_tw',
    # Hack: We don't want *any* conversion for stuff marked us-ascii, as all
    # sorts of garbage might be sent to us in the guise of 7-bit us-ascii.
    # Let that stuff pass through without conversion to/from Unicode.
    'us-ascii':    None,
    }



# Convenience functions for extending the above mappings
def add_charset(charset, header_enc=None, body_enc=None, output_charset=None):
    """Add character set properties to the global registry.

    charset is the input character set, and must be the canonical name of a
    character set.

    Optional header_enc and body_enc is either charset.QP for
    quoted-printable, charset.BASE64 for base64 encoding, charset.SHORTEST for
    the shortest of qp or base64 encoding, or None for no encoding.  SHORTEST
    is only valid for header_enc.  It describes how message headers and
    message bodies in the input charset are to be encoded.  Default is no
    encoding.

    Optional output_charset is the character set that the output should be
    in.  Conversions will proceed from input charset, to Unicode, to the
    output charset when the method Charset.convert() is called.  The default
    is to output in the same character set as the input.

    Both input_charset and output_charset must have Unicode codec entries in
    the module's charset-to-codec mapping; use add_codec(charset, codecname)
    to add codecs the module does not know about.  See the codecs module's
    documentation for more information.
    """
    if body_enc == SHORTEST:
        raise ValueError('SHORTEST not allowed for body_enc')
    CHARSETS[charset] = (header_enc, body_enc, output_charset)


def add_alias(alias, canonical):
    """Add a character set alias.

    alias is the alias name, e.g. latin-1
    canonical is the character set's canonical name, e.g. iso-8859-1
    """
    ALIASES[alias] = canonical


def add_codec(charset, codecname):
    """Add a codec that map characters in the given charset to/from Unicode.

    charset is the canonical name of a character set.  codecname is the name
    of a Python codec, as appropriate for the second argument to the unicode()
    built-in, or to the encode() method of a Unicode string.
    """
    CODEC_MAP[charset] = codecname



# Convenience function for encoding strings, taking into account
# that they might be unknown-8bit (ie: have surrogate-escaped bytes)
def _encode(string, codec):
    if codec == UNKNOWN8BIT:
        return string.encode('ascii', 'surrogateescape')
    else:
        return string.encode(codec)



class Charset:
    """Map character sets to their email properties.

    This class provides information about the requirements imposed on email
    for a specific character set.  It also provides convenience routines for
    converting between character sets, given the availability of the
    applicable codecs.  Given a character set, it will do its best to provide
    information on how to use that character set in an email in an
    RFC-compliant way.

    Certain character sets must be encoded with quoted-printable or base64
    when used in email headers or bodies.  Certain character sets must be
    converted outright, and are not allowed in email.  Instances of this
    module expose the following information about a character set:

    input_charset: The initial character set specified.  Common aliases
                   are converted to their `official' email names (e.g. latin_1
                   is converted to iso-8859-1).  Defaults to 7-bit us-ascii.

    header_encoding: If the character set must be encoded before it can be
                     used in an email header, this attribute will be set to
                     charset.QP (for quoted-printable), charset.BASE64 (for
                     base64 encoding), or charset.SHORTEST for the shortest of
                     QP or BASE64 encoding.  Otherwise, it will be None.

    body_encoding: Same as header_encoding, but describes the encoding for the
                   mail message's body, which indeed may be different than the
                   header encoding.  charset.SHORTEST is not allowed for
                   body_encoding.

    output_charset: Some character sets must be converted before they can be
                    used in email headers or bodies.  If the input_charset is
                    one of them, this attribute will contain the name of the
                    charset output will be converted to.  Otherwise, it will
                    be None.

    input_codec: The name of the Python codec used to convert the
                 input_charset to Unicode.  If no conversion codec is
                 necessary, this attribute will be None.

    output_codec: The name of the Python codec used to convert Unicode
                  to the output_charset.  If no conversion codec is necessary,
                  this attribute will have the same value as the input_codec.
    """
    def __init__(self, input_charset=DEFAULT_CHARSET):
        # RFC 2046, $4.1.2 says charsets are not case sensitive.  We coerce to
        # unicode because its .lower() is locale insensitive.  If the argument
        # is already a unicode, we leave it at that, but ensure that the
        # charset is ASCII, as the standard (RFC XXX) requires.
        try:
            if isinstance(input_charset, str):
                input_charset.encode('ascii')
            else:
                input_charset = str(input_charset, 'ascii')
        except UnicodeError:
            raise errors.CharsetError(input_charset)
        input_charset = input_charset.lower()
        # Set the input charset after filtering through the aliases
        self.input_charset = ALIASES.get(input_charset, input_charset)
        # We can try to guess which encoding and conversion to use by the
        # charset_map dictionary.  Try that first, but let the user override
        # it.
        henc, benc, conv = CHARSETS.get(self.input_charset,
                                        (SHORTEST, BASE64, None))
        if not conv:
            conv = self.input_charset
        # Set the attributes, allowing the arguments to override the default.
        self.header_encoding = henc
        self.body_encoding = benc
        self.output_charset = ALIASES.get(conv, conv)
        # Now set the codecs.  If one isn't defined for input_charset,
        # guess and try a Unicode codec with the same name as input_codec.
        self.input_codec = CODEC_MAP.get(self.input_charset,
                                         self.input_charset)
        self.output_codec = CODEC_MAP.get(self.output_charset,
                                          self.output_charset)

    def __repr__(self):
        return self.input_charset.lower()

    def __eq__(self, other):
        return str(self) == str(other).lower()

    def get_body_encoding(self):
        """Return the content-transfer-encoding used for body encoding.

        This is either the string `quoted-printable' or `base64' depending on
        the encoding used, or it is a function in which case you should call
        the function with a single argument, the Message object being
        encoded.  The function should then set the Content-Transfer-Encoding
        header itself to whatever is appropriate.

        Returns "quoted-printable" if self.body_encoding is QP.
        Returns "base64" if self.body_encoding is BASE64.
        Returns conversion function otherwise.
        """
        assert self.body_encoding != SHORTEST
        if self.body_encoding == QP:
            return 'quoted-printable'
        elif self.body_encoding == BASE64:
            return 'base64'
        else:
            return encode_7or8bit

    def get_output_charset(self):
        """Return the output character set.

        This is self.output_charset if that is not None, otherwise it is
        self.input_charset.
        """
        return self.output_charset or self.input_charset

    def header_encode(self, string):
        """Header-encode a string by converting it first to bytes.

        The type of encoding (base64 or quoted-printable) will be based on
        this charset's `header_encoding`.

        :param string: A unicode string for the header.  It must be possible
            to encode this string to bytes using the character set's
            output codec.
        :return: The encoded string, with RFC 2047 chrome.
        """
        codec = self.output_codec or 'us-ascii'
        header_bytes = _encode(string, codec)
        # 7bit/8bit encodings return the string unchanged (modulo conversions)
        encoder_module = self._get_encoder(header_bytes)
        if encoder_module is None:
            return string
        return encoder_module.header_encode(header_bytes, codec)

    def header_encode_lines(self, string, maxlengths):
        """Header-encode a string by converting it first to bytes.

        This is similar to `header_encode()` except that the string is fit
        into maximum line lengths as given by the argument.

        :param string: A unicode string for the header.  It must be possible
            to encode this string to bytes using the character set's
            output codec.
        :param maxlengths: Maximum line length iterator.  Each element
            returned from this iterator will provide the next maximum line
            length.  This parameter is used as an argument to built-in next()
            and should never be exhausted.  The maximum line lengths should
            not count the RFC 2047 chrome.  These line lengths are only a
            hint; the splitter does the best it can.
        :return: Lines of encoded strings, each with RFC 2047 chrome.
        """
        # See which encoding we should use.
        codec = self.output_codec or 'us-ascii'
        header_bytes = _encode(string, codec)
        encoder_module = self._get_encoder(header_bytes)
        encoder = partial(encoder_module.header_encode, charset=codec)
        # Calculate the number of characters that the RFC 2047 chrome will
        # contribute to each line.
        charset = self.get_output_charset()
        extra = len(charset) + RFC2047_CHROME_LEN
        # Now comes the hard part.  We must encode bytes but we can't split on
        # bytes because some character sets are variable length and each
        # encoded word must stand on its own.  So the problem is you have to
        # encode to bytes to figure out this word's length, but you must split
        # on characters.  This causes two problems: first, we don't know how
        # many octets a specific substring of unicode characters will get
        # encoded to, and second, we don't know how many ASCII characters
        # those octets will get encoded to.  Unless we try it.  Which seems
        # inefficient.  In the interest of being correct rather than fast (and
        # in the hope that there will be few encoded headers in any such
        # message), brute force it. :(
        lines = []
        current_line = []
        maxlen = next(maxlengths) - extra
        for character in string:
            current_line.append(character)
            this_line = EMPTYSTRING.join(current_line)
            length = encoder_module.header_length(_encode(this_line, charset))
            if length > maxlen:
                # This last character doesn't fit so pop it off.
                current_line.pop()
                # Does nothing fit on the first line?
                if not lines and not current_line:
                    lines.append(None)
                else:
                    separator = (' ' if lines else '')
                    joined_line = EMPTYSTRING.join(current_line)
                    header_bytes = _encode(joined_line, codec)
                    lines.append(encoder(header_bytes))
                current_line = [character]
                maxlen = next(maxlengths) - extra
        joined_line = EMPTYSTRING.join(current_line)
        header_bytes = _encode(joined_line, codec)
        lines.append(encoder(header_bytes))
        return lines

    def _get_encoder(self, header_bytes):
        if self.header_encoding == BASE64:
            return email.base64mime
        elif self.header_encoding == QP:
            return email.quoprimime
        elif self.header_encoding == SHORTEST:
            len64 = email.base64mime.header_length(header_bytes)
            lenqp = email.quoprimime.header_length(header_bytes)
            if len64 < lenqp:
                return email.base64mime
            else:
                return email.quoprimime
        else:
            return None

    def body_encode(self, string):
        """Body-encode a string by converting it first to bytes.

        The type of encoding (base64 or quoted-printable) will be based on
        self.body_encoding.  If body_encoding is None, we assume the
        output charset is a 7bit encoding, so re-encoding the decoded
        string using the ascii codec produces the correct string version
        of the content.
        """
        if not string:
            return string
        if self.body_encoding is BASE64:
            if isinstance(string, str):
                string = string.encode(self.output_charset)
            return email.base64mime.body_encode(string)
        elif self.body_encoding is QP:
            # quopromime.body_encode takes a string, but operates on it as if
            # it were a list of byte codes.  For a (minimal) history on why
            # this is so, see changeset 0cf700464177.  To correctly encode a
            # character set, then, we must turn it into pseudo bytes via the
            # latin1 charset, which will encode any byte as a single code point
            # between 0 and 255, which is what body_encode is expecting.
            if isinstance(string, str):
                string = string.encode(self.output_charset)
            string = string.decode('latin1')
            return email.quoprimime.body_encode(string)
        else:
            if isinstance(string, str):
                string = string.encode(self.output_charset).decode('ascii')
            return string
<|endoftext|># Copyright (C) 2004-2006 Python Software Foundation
# Authors: Baxter, Wouters and Warsaw
# Contact: email-sig@python.org

"""FeedParser - An email feed parser.

The feed parser implements an interface for incrementally parsing an email
message, line by line.  This has advantages for certain applications, such as
those reading email messages off a socket.

FeedParser.feed() is the primary interface for pushing new data into the
parser.  It returns when there's nothing more it can do with the available
data.  When you have no more data to push into the parser, call .close().
This completes the parsing and returns the root message object.

The other advantage of this parser is that it will never raise a parsing
exception.  Instead, when it finds something unexpected, it adds a 'defect' to
the current message.  Defects are just instances that live on the message
object's .defects attribute.
"""

__all__ = ['FeedParser', 'BytesFeedParser']

import re

from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO

NLCRE = re.compile(r'\r\n|\r|\n')
NLCRE_bol = re.compile(r'(\r\n|\r|\n)')
NLCRE_eol = re.compile(r'(\r\n|\r|\n)\Z')
NLCRE_crack = re.compile(r'(\r\n|\r|\n)')
# RFC 2822 $3.6.8 Optional fields.  ftext is %d33-57 / %d59-126, Any character
# except controls, SP, and ":".
headerRE = re.compile(r'^(From |[\041-\071\073-\176]*:|[\t ])')
EMPTYSTRING = ''
NL = '\n'

NeedMoreData = object()



class BufferedSubFile(object):
    """A file-ish object that can have new data loaded into it.

    You can also push and pop line-matching predicates onto a stack.  When the
    current predicate matches the current line, a false EOF response
    (i.e. empty string) is returned instead.  This lets the parser adhere to a
    simple abstraction -- it parses until EOF closes the current message.
    """
    def __init__(self):
        # Text stream of the last partial line pushed into this object.
        # See issue 22233 for why this is a text stream and not a list.
        self._partial = StringIO(newline='')
        # A deque of full, pushed lines
        self._lines = deque()
        # The stack of false-EOF checking predicates.
        self._eofstack = []
        # A flag indicating whether the file has been closed or not.
        self._closed = False

    def push_eof_matcher(self, pred):
        self._eofstack.append(pred)

    def pop_eof_matcher(self):
        return self._eofstack.pop()

    def close(self):
        # Don't forget any trailing partial line.
        self._partial.seek(0)
        self.pushlines(self._partial.readlines())
        self._partial.seek(0)
        self._partial.truncate()
        self._closed = True

    def readline(self):
        if not self._lines:
            if self._closed:
                return ''
            return NeedMoreData
        # Pop the line off the stack and see if it matches the current
        # false-EOF predicate.
        line = self._lines.popleft()
        # RFC 2046, section 5.1.2 requires us to recognize outer level
        # boundaries at any level of inner nesting.  Do this, but be sure it's
        # in the order of most to least nested.
        for ateof in reversed(self._eofstack):
            if ateof(line):
                # We're at the false EOF.  But push the last line back first.
                self._lines.appendleft(line)
                return ''
        return line

    def unreadline(self, line):
        # Let the consumer push a line back into the buffer.
        assert line is not NeedMoreData
        self._lines.appendleft(line)

    def push(self, data):
        """Push some new data into this object."""
        self._partial.write(data)
        if '\n' not in data and '\r' not in data:
            # No new complete lines, wait for more.
            return

        # Crack into lines, preserving the linesep characters.
        self._partial.seek(0)
        parts = self._partial.readlines()
        self._partial.seek(0)
        self._partial.truncate()

        # If the last element of the list does not end in a newline, then treat
        # it as a partial line.  We only check for '\n' here because a line
        # ending with '\r' might be a line that was split in the middle of a
        # '\r\n' sequence (see bugs 1555570 and 1721862).
        if not parts[-1].endswith('\n'):
            self._partial.write(parts.pop())
        self.pushlines(parts)

    def pushlines(self, lines):
        self._lines.extend(lines)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line == '':
            raise StopIteration
        return line



class FeedParser:
    """A feed-style parser of email."""

    def __init__(self, _factory=None, *, policy=compat32):
        """_factory is called with no arguments to create a new message obj

        The policy keyword specifies a policy object that controls a number of
        aspects of the parser's operation.  The default policy maintains
        backward compatibility.

        """
        self.policy = policy
        self._old_style_factory = False
        if _factory is None:
            if policy.message_factory is None:
                from email.message import Message
                self._factory = Message
            else:
                self._factory = policy.message_factory
        else:
            self._factory = _factory
            try:
                _factory(policy=self.policy)
            except TypeError:
                # Assume this is an old-style factory
                self._old_style_factory = True
        self._input = BufferedSubFile()
        self._msgstack = []
        self._parse = self._parsegen().__next__
        self._cur = None
        self._last = None
        self._headersonly = False

    # Non-public interface for supporting Parser's headersonly flag
    def _set_headersonly(self):
        self._headersonly = True

    def feed(self, data):
        """Push more data into the parser."""
        self._input.push(data)
        self._call_parse()

    def _call_parse(self):
        try:
            self._parse()
        except StopIteration:
            pass

    def close(self):
        """Parse all remaining data and return the root message object."""
        self._input.close()
        self._call_parse()
        root = self._pop_message()
        assert not self._msgstack
        # Look for final set of defects
        if root.get_content_maintype() == 'multipart' \
               and not root.is_multipart():
            defect = errors.MultipartInvariantViolationDefect()
            self.policy.handle_defect(root, defect)
        return root

    def _new_message(self):
        if self._old_style_factory:
            msg = self._factory()
        else:
            msg = self._factory(policy=self.policy)
        if self._cur and self._cur.get_content_type() == 'multipart/digest':
            msg.set_default_type('message/rfc822')
        if self._msgstack:
            self._msgstack[-1].attach(msg)
        self._msgstack.append(msg)
        self._cur = msg
        self._last = msg

    def _pop_message(self):
        retval = self._msgstack.pop()
        if self._msgstack:
            self._cur = self._msgstack[-1]
        else:
            self._cur = None
        return retval

    def _parsegen(self):
        # Create a new message and start by parsing headers.
        self._new_message()
        headers = []
        # Collect the headers, searching for a line that doesn't match the RFC
        # 2822 header or continuation pattern (including an empty line).
        for line in self._input:
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            if not headerRE.match(line):
                # If we saw the RFC defined header/body separator
                # (i.e. newline), just throw it away. Otherwise the line is
                # part of the body so push it back.
                if not NLCRE.match(line):
                    defect = errors.MissingHeaderBodySeparatorDefect()
                    self.policy.handle_defect(self._cur, defect)
                    self._input.unreadline(line)
                break
            headers.append(line)
        # Done with the headers, so parse them and figure out what we're
        # supposed to see in the body of the message.
        self._parse_headers(headers)
        # Headers-only parsing is a backwards compatibility hack, which was
        # necessary in the older parser, which could raise errors.  All
        # remaining lines in the input are thrown into the message body.
        if self._headersonly:
            lines = []
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                if line == '':
                    break
                lines.append(line)
            self._cur.set_payload(EMPTYSTRING.join(lines))
            return
        if self._cur.get_content_type() == 'message/delivery-status':
            # message/delivery-status contains blocks of headers separated by
            # a blank line.  We'll represent each header block as a separate
            # nested message object, but the processing is a bit different
            # than standard message/* types because there is no body for the
            # nested messages.  A blank line separates the subparts.
            while True:
                self._input.push_eof_matcher(NLCRE.match)
                for retval in self._parsegen():
                    if retval is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                msg = self._pop_message()
                # We need to pop the EOF matcher in order to tell if we're at
                # the end of the current file, not the end of the last block
                # of message headers.
                self._input.pop_eof_matcher()
                # The input stream must be sitting at the newline or at the
                # EOF.  We want to see if we're at the end of this subpart, so
                # first consume the blank line, then test the next line to see
                # if we're at this subpart's EOF.
                while True:
                    line = self._input.readline()
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                while True:
                    line = self._input.readline()
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                if line == '':
                    break
                # Not at EOF so this is a line we're going to need.
                self._input.unreadline(line)
            return
        if self._cur.get_content_maintype() == 'message':
            # The message claims to be a message/* type, then what follows is
            # another RFC 2822 message.
            for retval in self._parsegen():
                if retval is NeedMoreData:
                    yield NeedMoreData
                    continue
                break
            self._pop_message()
            return
        if self._cur.get_content_maintype() == 'multipart':
            boundary = self._cur.get_boundary()
            if boundary is None:
                # The message /claims/ to be a multipart but it has not
                # defined a boundary.  That's a problem which we'll handle by
                # reading everything until the EOF and marking the message as
                # defective.
                defect = errors.NoBoundaryInMultipartDefect()
                self.policy.handle_defect(self._cur, defect)
                lines = []
                for line in self._input:
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    lines.append(line)
                self._cur.set_payload(EMPTYSTRING.join(lines))
                return
            # Make sure a valid content type was specified per RFC 2045:6.4.
            if (str(self._cur.get('content-transfer-encoding', '8bit')).lower()
                    not in ('7bit', '8bit', 'binary')):
                defect = errors.InvalidMultipartContentTransferEncodingDefect()
                self.policy.handle_defect(self._cur, defect)
            # Create a line match predicate which matches the inter-part
            # boundary as well as the end-of-multipart boundary.  Don't push
            # this onto the input stream until we've scanned past the
            # preamble.
            separator = '--' + boundary
            boundaryre = re.compile(
                '(?P<sep>' + re.escape(separator) +
                r')(?P<end>--)?(?P<ws>[ \t]*)(?P<linesep>\r\n|\r|\n)?$')
            capturing_preamble = True
            preamble = []
            linesep = False
            close_boundary_seen = False
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                if line == '':
                    break
                mo = boundaryre.match(line)
                if mo:
                    # If we're looking at the end boundary, we're done with
                    # this multipart.  If there was a newline at the end of
                    # the closing boundary, then we need to initialize the
                    # epilogue with the empty string (see below).
                    if mo.group('end'):
                        close_boundary_seen = True
                        linesep = mo.group('linesep')
                        break
                    # We saw an inter-part boundary.  Were we in the preamble?
                    if capturing_preamble:
                        if preamble:
                            # According to RFC 2046, the last newline belongs
                            # to the boundary.
                            lastline = preamble[-1]
                            eolmo = NLCRE_eol.search(lastline)
                            if eolmo:
                                preamble[-1] = lastline[:-len(eolmo.group(0))]
                            self._cur.preamble = EMPTYSTRING.join(preamble)
                        capturing_preamble = False
                        self._input.unreadline(line)
                        continue
                    # We saw a boundary separating two parts.  Consume any
                    # multiple boundary lines that may be following.  Our
                    # interpretation of RFC 2046 BNF grammar does not produce
                    # body parts within such double boundaries.
                    while True:
                        line = self._input.readline()
                        if line is NeedMoreData:
                            yield NeedMoreData
                            continue
                        mo = boundaryre.match(line)
                        if not mo:
                            self._input.unreadline(line)
                            break
                    # Recurse to parse this subpart; the input stream points
                    # at the subpart's first line.
                    self._input.push_eof_matcher(boundaryre.match)
                    for retval in self._parsegen():
                        if retval is NeedMoreData:
                            yield NeedMoreData
                            continue
                        break
                    # Because of RFC 2046, the newline preceding the boundary
                    # separator actually belongs to the boundary, not the
                    # previous subpart's payload (or epilogue if the previous
                    # part is a multipart).
                    if self._last.get_content_maintype() == 'multipart':
                        epilogue = self._last.epilogue
                        if epilogue == '':
                            self._last.epilogue = None
                        elif epilogue is not None:
                            mo = NLCRE_eol.search(epilogue)
                            if mo:
                                end = len(mo.group(0))
                                self._last.epilogue = epilogue[:-end]
                    else:
                        payload = self._last._payload
                        if isinstance(payload, str):
                            mo = NLCRE_eol.search(payload)
                            if mo:
                                payload = payload[:-len(mo.group(0))]
                                self._last._payload = payload
                    self._input.pop_eof_matcher()
                    self._pop_message()
                    # Set the multipart up for newline cleansing, which will
                    # happen if we're in a nested multipart.
                    self._last = self._cur
                else:
                    # I think we must be in the preamble
                    assert capturing_preamble
                    preamble.append(line)
            # We've seen either the EOF or the end boundary.  If we're still
            # capturing the preamble, we never saw the start boundary.  Note
            # that as a defect and store the captured text as the payload.
            if capturing_preamble:
                defect = errors.StartBoundaryNotFoundDefect()
                self.policy.handle_defect(self._cur, defect)
                self._cur.set_payload(EMPTYSTRING.join(preamble))
                epilogue = []
                for line in self._input:
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                self._cur.epilogue = EMPTYSTRING.join(epilogue)
                return
            # If we're not processing the preamble, then we might have seen
            # EOF without seeing that end boundary...that is also a defect.
            if not close_boundary_seen:
                defect = errors.CloseBoundaryNotFoundDefect()
                self.policy.handle_defect(self._cur, defect)
                return
            # Everything from here to the EOF is epilogue.  If the end boundary
            # ended in a newline, we'll need to make sure the epilogue isn't
            # None
            if linesep:
                epilogue = ['']
            else:
                epilogue = []
            for line in self._input:
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                epilogue.append(line)
            # Any CRLF at the front of the epilogue is not technically part of
            # the epilogue.  Also, watch out for an empty string epilogue,
            # which means a single newline.
            if epilogue:
                firstline = epilogue[0]
                bolmo = NLCRE_bol.match(firstline)
                if bolmo:
                    epilogue[0] = firstline[len(bolmo.group(0)):]
            self._cur.epilogue = EMPTYSTRING.join(epilogue)
            return
        # Otherwise, it's some non-multipart type, so the entire rest of the
        # file contents becomes the payload.
        lines = []
        for line in self._input:
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            lines.append(line)
        self._cur.set_payload(EMPTYSTRING.join(lines))

    def _parse_headers(self, lines):
        # Passed a list of lines that make up the headers for the current msg
        lastheader = ''
        lastvalue = []
        for lineno, line in enumerate(lines):
            # Check for continuation
            if line[0] in ' \t':
                if not lastheader:
                    # The first line of the headers was a continuation.  This
                    # is illegal, so let's note the defect, store the illegal
                    # line, and ignore it for purposes of headers.
                    defect = errors.FirstHeaderLineIsContinuationDefect(line)
                    self.policy.handle_defect(self._cur, defect)
                    continue
                lastvalue.append(line)
                continue
            if lastheader:
                self._cur.set_raw(*self.policy.header_source_parse(lastvalue))
                lastheader, lastvalue = '', []
            # Check for envelope header, i.e. unix-from
            if line.startswith('From '):
                if lineno == 0:
                    # Strip off the trailing newline
                    mo = NLCRE_eol.search(line)
                    if mo:
                        line = line[:-len(mo.group(0))]
                    self._cur.set_unixfrom(line)
                    continue
                elif lineno == len(lines) - 1:
                    # Something looking like a unix-from at the end - it's
                    # probably the first line of the body, so push back the
                    # line and stop.
                    self._input.unreadline(line)
                    return
                else:
                    # Weirdly placed unix-from line.  Note this as a defect
                    # and ignore it.
                    defect = errors.MisplacedEnvelopeHeaderDefect(line)
                    self._cur.defects.append(defect)
                    continue
            # Split the line on the colon separating field name from value.
            # There will always be a colon, because if there wasn't the part of
            # the parser that calls us would have started parsing the body.
            i = line.find(':')

            # If the colon is on the start of the line the header is clearly
            # malformed, but we might be able to salvage the rest of the
            # message. Track the error but keep going.
            if i == 0:
                defect = errors.InvalidHeaderDefect("Missing header name.")
                self._cur.defects.append(defect)
                continue

            assert i>0, "_parse_headers fed line with no : and no leading WS"
            lastheader = line[:i]
            lastvalue = [line]
        # Done with all the lines, so handle the last header.
        if lastheader:
            self._cur.set_raw(*self.policy.header_source_parse(lastvalue))


class BytesFeedParser(FeedParser):
    """Like FeedParser, but feed accepts bytes."""

    def feed(self, data):
        super().feed(data.decode('ascii', 'surrogateescape'))
<|endoftext|># Copyright (C) 2002-2007 Python Software Foundation
# Author: Ben Gertzfield
# Contact: email-sig@python.org

"""Base64 content transfer encoding per RFCs 2045-2047.

This module handles the content transfer encoding method defined in RFC 2045
to encode arbitrary 8-bit data using the three 8-bit bytes in four 7-bit
characters encoding known as Base64.

It is used in the MIME standards for email to attach images, audio, and text
using some 8-bit character sets to messages.

This module provides an interface to encode and decode both headers and bodies
with Base64 encoding.

RFC 2045 defines a method for including character set information in an
`encoded-word' in a header.  This method is commonly used for 8-bit real names
in To:, From:, Cc:, etc. fields, as well as Subject: lines.

This module does not do the line wrapping or end-of-line character conversion
necessary for proper internationalized headers; it only does dumb encoding and
decoding.  To deal with the various line wrapping issues, use the email.header
module.
"""

__all__ = [
    'body_decode',
    'body_encode',
    'decode',
    'decodestring',
    'header_encode',
    'header_length',
    ]


from base64 import b64encode
from binascii import b2a_base64, a2b_base64

CRLF = '\r\n'
NL = '\n'
EMPTYSTRING = ''

# See also Charset.py
MISC_LEN = 7



# Helpers
def header_length(bytearray):
    """Return the length of s when it is encoded with base64."""
    groups_of_3, leftover = divmod(len(bytearray), 3)
    # 4 bytes out for each 3 bytes (or nonzero fraction thereof) in.
    n = groups_of_3 * 4
    if leftover:
        n += 4
    return n



def header_encode(header_bytes, charset='iso-8859-1'):
    """Encode a single header line with Base64 encoding in a given charset.

    charset names the character set to use to encode the header.  It defaults
    to iso-8859-1.  Base64 encoding is defined in RFC 2045.
    """
    if not header_bytes:
        return ""
    if isinstance(header_bytes, str):
        header_bytes = header_bytes.encode(charset)
    encoded = b64encode(header_bytes).decode("ascii")
    return '=?%s?b?%s?=' % (charset, encoded)



def body_encode(s, maxlinelen=76, eol=NL):
    r"""Encode a string with base64.

    Each line will be wrapped at, at most, maxlinelen characters (defaults to
    76 characters).

    Each line of encoded text will end with eol, which defaults to "\n".  Set
    this to "\r\n" if you will be using the result of this function directly
    in an email.
    """
    if not s:
        return ""

    encvec = []
    max_unencoded = maxlinelen * 3 // 4
    for i in range(0, len(s), max_unencoded):
        # BAW: should encode() inherit b2a_base64()'s dubious behavior in
        # adding a newline to the encoded string?
        enc = b2a_base64(s[i:i + max_unencoded]).decode("ascii")
        if enc.endswith(NL) and eol != NL:
            enc = enc[:-1] + eol
        encvec.append(enc)
    return EMPTYSTRING.join(encvec)



def decode(string):
    """Decode a raw base64 string, returning a bytes object.

    This function does not parse a full MIME header value encoded with
    base64 (like =?iso-8859-1?b?bmloISBuaWgh?=) -- please use the high
    level email.header class for that functionality.
    """
    if not string:
        return bytes()
    elif isinstance(string, str):
        return a2b_base64(string.encode('raw-unicode-escape'))
    else:
        return a2b_base64(string)


# For convenience and backwards compatibility w/ standard base64 module
body_decode = decode
decodestring = decode
<|endoftext|>import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime

class ContentManager:

    def __init__(self):
        self.get_handlers = {}
        self.set_handlers = {}

    def add_get_handler(self, key, handler):
        self.get_handlers[key] = handler

    def get_content(self, msg, *args, **kw):
        content_type = msg.get_content_type()
        if content_type in self.get_handlers:
            return self.get_handlers[content_type](msg, *args, **kw)
        maintype = msg.get_content_maintype()
        if maintype in self.get_handlers:
            return self.get_handlers[maintype](msg, *args, **kw)
        if '' in self.get_handlers:
            return self.get_handlers[''](msg, *args, **kw)
        raise KeyError(content_type)

    def add_set_handler(self, typekey, handler):
        self.set_handlers[typekey] = handler

    def set_content(self, msg, obj, *args, **kw):
        if msg.get_content_maintype() == 'multipart':
            # XXX: is this error a good idea or not?  We can remove it later,
            # but we can't add it later, so do it for now.
            raise TypeError("set_content not valid on multipart")
        handler = self._find_set_handler(msg, obj)
        msg.clear_content()
        handler(msg, obj, *args, **kw)

    def _find_set_handler(self, msg, obj):
        full_path_for_error = None
        for typ in type(obj).__mro__:
            if typ in self.set_handlers:
                return self.set_handlers[typ]
            qname = typ.__qualname__
            modname = getattr(typ, '__module__', '')
            full_path = '.'.join((modname, qname)) if modname else qname
            if full_path_for_error is None:
                full_path_for_error = full_path
            if full_path in self.set_handlers:
                return self.set_handlers[full_path]
            if qname in self.set_handlers:
                return self.set_handlers[qname]
            name = typ.__name__
            if name in self.set_handlers:
                return self.set_handlers[name]
        if None in self.set_handlers:
            return self.set_handlers[None]
        raise KeyError(full_path_for_error)


raw_data_manager = ContentManager()


def get_text_content(msg, errors='replace'):
    content = msg.get_payload(decode=True)
    charset = msg.get_param('charset', 'ASCII')
    return content.decode(charset, errors=errors)
raw_data_manager.add_get_handler('text', get_text_content)


def get_non_text_content(msg):
    return msg.get_payload(decode=True)
for maintype in 'audio image video application'.split():
    raw_data_manager.add_get_handler(maintype, get_non_text_content)


def get_message_content(msg):
    return msg.get_payload(0)
for subtype in 'rfc822 external-body'.split():
    raw_data_manager.add_get_handler('message/'+subtype, get_message_content)


def get_and_fixup_unknown_message_content(msg):
    # If we don't understand a message subtype, we are supposed to treat it as
    # if it were application/octet-stream, per
    # tools.ietf.org/html/rfc2046#section-5.2.4.  Feedparser doesn't do that,
    # so do our best to fix things up.  Note that it is *not* appropriate to
    # model message/partial content as Message objects, so they are handled
    # here as well.  (How to reassemble them is out of scope for this comment :)
    return bytes(msg.get_payload(0))
raw_data_manager.add_get_handler('message',
                                 get_and_fixup_unknown_message_content)


def _prepare_set(msg, maintype, subtype, headers):
    msg['Content-Type'] = '/'.join((maintype, subtype))
    if headers:
        if not hasattr(headers[0], 'name'):
            mp = msg.policy
            headers = [mp.header_factory(*mp.header_source_parse([header]))
                       for header in headers]
        try:
            for header in headers:
                if header.defects:
                    raise header.defects[0]
                msg[header.name] = header
        except email.errors.HeaderDefect as exc:
            raise ValueError("Invalid header: {}".format(
                                header.fold(policy=msg.policy))) from exc


def _finalize_set(msg, disposition, filename, cid, params):
    if disposition is None and filename is not None:
        disposition = 'attachment'
    if disposition is not None:
        msg['Content-Disposition'] = disposition
    if filename is not None:
        msg.set_param('filename',
                      filename,
                      header='Content-Disposition',
                      replace=True)
    if cid is not None:
        msg['Content-ID'] = cid
    if params is not None:
        for key, value in params.items():
            msg.set_param(key, value)


# XXX: This is a cleaned-up version of base64mime.body_encode (including a bug
# fix in the calculation of unencoded_bytes_per_line).  It would be nice to
# drop both this and quoprimime.body_encode in favor of enhanced binascii
# routines that accepted a max_line_length parameter.
def _encode_base64(data, max_line_length):
    encoded_lines = []
    unencoded_bytes_per_line = max_line_length // 4 * 3
    for i in range(0, len(data), unencoded_bytes_per_line):
        thisline = data[i:i+unencoded_bytes_per_line]
        encoded_lines.append(binascii.b2a_base64(thisline).decode('ascii'))
    return ''.join(encoded_lines)


def _encode_text(string, charset, cte, policy):
    lines = string.encode(charset).splitlines()
    linesep = policy.linesep.encode('ascii')
    def embedded_body(lines): return linesep.join(lines) + linesep
    def normal_body(lines): return b'\n'.join(lines) + b'\n'
    if cte is None:
        # Use heuristics to decide on the "best" encoding.
        if max((len(x) for x in lines), default=0) <= policy.max_line_length:
            try:
                return '7bit', normal_body(lines).decode('ascii')
            except UnicodeDecodeError:
                pass
            if policy.cte_type == '8bit':
                return '8bit', normal_body(lines).decode('ascii', 'surrogateescape')
        sniff = embedded_body(lines[:10])
        sniff_qp = quoprimime.body_encode(sniff.decode('latin-1'),
                                          policy.max_line_length)
        sniff_base64 = binascii.b2a_base64(sniff)
        # This is a little unfair to qp; it includes lineseps, base64 doesn't.
        if len(sniff_qp) > len(sniff_base64):
            cte = 'base64'
        else:
            cte = 'quoted-printable'
            if len(lines) <= 10:
                return cte, sniff_qp
    if cte == '7bit':
        data = normal_body(lines).decode('ascii')
    elif cte == '8bit':
        data = normal_body(lines).decode('ascii', 'surrogateescape')
    elif cte == 'quoted-printable':
        data = quoprimime.body_encode(normal_body(lines).decode('latin-1'),
                                      policy.max_line_length)
    elif cte == 'base64':
        data = _encode_base64(embedded_body(lines), policy.max_line_length)
    else:
        raise ValueError("Unknown content transfer encoding {}".format(cte))
    return cte, data


def set_text_content(msg, string, subtype="plain", charset='utf-8', cte=None,
                     disposition=None, filename=None, cid=None,
                     params=None, headers=None):
    _prepare_set(msg, 'text', subtype, headers)
    cte, payload = _encode_text(string, charset, cte, msg.policy)
    msg.set_payload(payload)
    msg.set_param('charset',
                  email.charset.ALIASES.get(charset, charset),
                  replace=True)
    msg['Content-Transfer-Encoding'] = cte
    _finalize_set(msg, disposition, filename, cid, params)
raw_data_manager.add_set_handler(str, set_text_content)


def set_message_content(msg, message, subtype="rfc822", cte=None,
                       disposition=None, filename=None, cid=None,
                       params=None, headers=None):
    if subtype == 'partial':
        raise ValueError("message/partial is not supported for Message objects")
    if subtype == 'rfc822':
        if cte not in (None, '7bit', '8bit', 'binary'):
            # http://tools.ietf.org/html/rfc2046#section-5.2.1 mandate.
            raise ValueError(
                "message/rfc822 parts do not support cte={}".format(cte))
        # 8bit will get coerced on serialization if policy.cte_type='7bit'.  We
        # may end up claiming 8bit when it isn't needed, but the only negative
        # result of that should be a gateway that needs to coerce to 7bit
        # having to look through the whole embedded message to discover whether
        # or not it actually has to do anything.
        cte = '8bit' if cte is None else cte
    elif subtype == 'external-body':
        if cte not in (None, '7bit'):
            # http://tools.ietf.org/html/rfc2046#section-5.2.3 mandate.
            raise ValueError(
                "message/external-body parts do not support cte={}".format(cte))
        cte = '7bit'
    elif cte is None:
        # http://tools.ietf.org/html/rfc2046#section-5.2.4 says all future
        # subtypes should be restricted to 7bit, so assume that.
        cte = '7bit'
    _prepare_set(msg, 'message', subtype, headers)
    msg.set_payload([message])
    msg['Content-Transfer-Encoding'] = cte
    _finalize_set(msg, disposition, filename, cid, params)
raw_data_manager.add_set_handler(email.message.Message, set_message_content)


def set_bytes_content(msg, data, maintype, subtype, cte='base64',
                     disposition=None, filename=None, cid=None,
                     params=None, headers=None):
    _prepare_set(msg, maintype, subtype, headers)
    if cte == 'base64':
        data = _encode_base64(data, max_line_length=msg.policy.max_line_length)
    elif cte == 'quoted-printable':
        # XXX: quoprimime.body_encode won't encode newline characters in data,
        # so we can't use it.  This means max_line_length is ignored.  Another
        # bug to fix later.  (Note: encoders.quopri is broken on line ends.)
        data = binascii.b2a_qp(data, istext=False, header=False, quotetabs=True)
        data = data.decode('ascii')
    elif cte == '7bit':
        data = data.decode('ascii')
    elif cte in ('8bit', 'binary'):
        data = data.decode('ascii', 'surrogateescape')
    msg.set_payload(data)
    msg['Content-Transfer-Encoding'] = cte
    _finalize_set(msg, disposition, filename, cid, params)
for typ in (bytes, bytearray, memoryview):
    raw_data_manager.add_set_handler(typ, set_bytes_content)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Various types of useful iterators and generators."""

__all__ = [
    'body_line_iterator',
    'typed_subpart_iterator',
    'walk',
    # Do not include _structure() since it's part of the debugging API.
    ]

import sys
from io import StringIO



# This function will become a method of the Message class
def walk(self):
    """Walk over the message tree, yielding each subpart.

    The walk is performed in depth-first order.  This method is a
    generator.
    """
    yield self
    if self.is_multipart():
        for subpart in self.get_payload():
            yield from subpart.walk()



# These two functions are imported into the Iterators.py interface module.
def body_line_iterator(msg, decode=False):
    """Iterate over the parts, returning string payloads line-by-line.

    Optional decode (default False) is passed through to .get_payload().
    """
    for subpart in msg.walk():
        payload = subpart.get_payload(decode=decode)
        if isinstance(payload, str):
            yield from StringIO(payload)


def typed_subpart_iterator(msg, maintype='text', subtype=None):
    """Iterate over the subparts with a given MIME type.

    Use `maintype' as the main MIME type to match against; this defaults to
    "text".  Optional `subtype' is the MIME subtype to match against; if
    omitted, only the main type is matched.
    """
    for subpart in msg.walk():
        if subpart.get_content_maintype() == maintype:
            if subtype is None or subpart.get_content_subtype() == subtype:
                yield subpart



def _structure(msg, fp=None, level=0, include_default=False):
    """A handy debugging aid"""
    if fp is None:
        fp = sys.stdout
    tab = ' ' * (level * 4)
    print(tab + msg.get_content_type(), end='', file=fp)
    if include_default:
        print(' [%s]' % msg.get_default_type(), file=fp)
    else:
        print(file=fp)
    if msg.is_multipart():
        for subpart in msg.get_payload():
            _structure(subpart, fp, level+1, include_default)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Class representing text/* type MIME documents."""

__all__ = ['MIMEText']

from email.charset import Charset
from email.mime.nonmultipart import MIMENonMultipart



class MIMEText(MIMENonMultipart):
    """Class for generating text/* type MIME documents."""

    def __init__(self, _text, _subtype='plain', _charset=None, *, policy=None):
        """Create a text/* type MIME document.

        _text is the string for this message object.

        _subtype is the MIME sub content type, defaulting to "plain".

        _charset is the character set parameter added to the Content-Type
        header.  This defaults to "us-ascii".  Note that as a side-effect, the
        Content-Transfer-Encoding header will also be set.
        """

        # If no _charset was specified, check to see if there are non-ascii
        # characters present. If not, use 'us-ascii', otherwise use utf-8.
        # XXX: This can be removed once #7304 is fixed.
        if _charset is None:
            try:
                _text.encode('us-ascii')
                _charset = 'us-ascii'
            except UnicodeEncodeError:
                _charset = 'utf-8'

        MIMENonMultipart.__init__(self, 'text', _subtype, policy=policy,
                                  **{'charset': str(_charset)})

        self.set_payload(_text, _charset)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Class representing image/* type MIME documents."""

__all__ = ['MIMEImage']

import imghdr

from email import encoders
from email.mime.nonmultipart import MIMENonMultipart



class MIMEImage(MIMENonMultipart):
    """Class for generating image/* type MIME documents."""

    def __init__(self, _imagedata, _subtype=None,
                 _encoder=encoders.encode_base64, *, policy=None, **_params):
        """Create an image/* type MIME document.

        _imagedata contains the bytes for the raw image data.  If the data
        can be decoded by the standard Python `imghdr' module, then the
        subtype will be automatically included in the Content-Type header.
        Otherwise, you can specify the specific image subtype via the _subtype
        parameter.

        _encoder is a function which will perform the actual encoding for
        transport of the image data.  It takes one argument, which is this
        Image instance.  It should use get_payload() and set_payload() to
        change the payload to the encoded form.  It should also add any
        Content-Transfer-Encoding or other headers to the message as
        necessary.  The default encoding is Base64.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        if _subtype is None:
            _subtype = imghdr.what(None, _imagedata)
        if _subtype is None:
            raise TypeError('Could not guess image MIME subtype')
        MIMENonMultipart.__init__(self, 'image', _subtype, policy=policy,
                                  **_params)
        self.set_payload(_imagedata)
        _encoder(self)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Keith Dart
# Contact: email-sig@python.org

"""Class representing application/* type MIME documents."""

__all__ = ["MIMEApplication"]

from email import encoders
from email.mime.nonmultipart import MIMENonMultipart


class MIMEApplication(MIMENonMultipart):
    """Class for generating application/* MIME documents."""

    def __init__(self, _data, _subtype='octet-stream',
                 _encoder=encoders.encode_base64, *, policy=None, **_params):
        """Create an application/* type MIME document.

        _data contains the bytes for the raw application data.

        _subtype is the MIME content type subtype, defaulting to
        'octet-stream'.

        _encoder is a function which will perform the actual encoding for
        transport of the application data, defaulting to base64 encoding.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        if _subtype is None:
            raise TypeError('Invalid application MIME subtype')
        MIMENonMultipart.__init__(self, 'application', _subtype, policy=policy,
                                  **_params)
        self.set_payload(_data)
        _encoder(self)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Class representing message/* MIME documents."""

__all__ = ['MIMEMessage']

from email import message
from email.mime.nonmultipart import MIMENonMultipart



class MIMEMessage(MIMENonMultipart):
    """Class representing message/* MIME documents."""

    def __init__(self, _msg, _subtype='rfc822', *, policy=None):
        """Create a message/* type MIME document.

        _msg is a message object and must be an instance of Message, or a
        derived class of Message, otherwise a TypeError is raised.

        Optional _subtype defines the subtype of the contained message.  The
        default is "rfc822" (this is defined by the MIME standard, even though
        the term "rfc822" is technically outdated by RFC 2822).
        """
        MIMENonMultipart.__init__(self, 'message', _subtype, policy=policy)
        if not isinstance(_msg, message.Message):
            raise TypeError('Argument is not an instance of Message')
        # It's convenient to use this base class method.  We need to do it
        # this way or we'll get an exception
        message.Message.attach(self, _msg)
        # And be sure our default type is set correctly
        self.set_default_type('message/rfc822')
<|endoftext|># Copyright (C) 2002-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Base class for MIME type messages that are not multipart."""

__all__ = ['MIMENonMultipart']

from email import errors
from email.mime.base import MIMEBase



class MIMENonMultipart(MIMEBase):
    """Base class for MIME non-multipart type messages."""

    def attach(self, payload):
        # The public API prohibits attaching multiple subparts to MIMEBase
        # derived subtypes since none of them are, by definition, of content
        # type multipart/*
        raise errors.MultipartConversionError(
            'Cannot attach additional subparts to non-multipart/*')
<|endoftext|># Copyright (C) 2001-2007 Python Software Foundation
# Author: Anthony Baxter
# Contact: email-sig@python.org

"""Class representing audio/* type MIME documents."""

__all__ = ['MIMEAudio']

import sndhdr

from io import BytesIO
from email import encoders
from email.mime.nonmultipart import MIMENonMultipart



_sndhdr_MIMEmap = {'au'  : 'basic',
                   'wav' :'x-wav',
                   'aiff':'x-aiff',
                   'aifc':'x-aiff',
                   }

# There are others in sndhdr that don't have MIME types. :(
# Additional ones to be added to sndhdr? midi, mp3, realaudio, wma??
def _whatsnd(data):
    """Try to identify a sound file type.

    sndhdr.what() has a pretty cruddy interface, unfortunately.  This is why
    we re-do it here.  It would be easier to reverse engineer the Unix 'file'
    command and use the standard 'magic' file, as shipped with a modern Unix.
    """
    hdr = data[:512]
    fakefile = BytesIO(hdr)
    for testfn in sndhdr.tests:
        res = testfn(hdr, fakefile)
        if res is not None:
            return _sndhdr_MIMEmap.get(res[0])
    return None



class MIMEAudio(MIMENonMultipart):
    """Class for generating audio/* MIME documents."""

    def __init__(self, _audiodata, _subtype=None,
                 _encoder=encoders.encode_base64, *, policy=None, **_params):
        """Create an audio/* type MIME document.

        _audiodata contains the bytes for the raw audio data.  If this data
        can be decoded by the standard Python `sndhdr' module, then the
        subtype will be automatically included in the Content-Type header.
        Otherwise, you can specify  the specific audio subtype via the
        _subtype parameter.  If _subtype is not given, and no subtype can be
        guessed, a TypeError is raised.

        _encoder is a function which will perform the actual encoding for
        transport of the image data.  It takes one argument, which is this
        Image instance.  It should use get_payload() and set_payload() to
        change the payload to the encoded form.  It should also add any
        Content-Transfer-Encoding or other headers to the message as
        necessary.  The default encoding is Base64.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        if _subtype is None:
            _subtype = _whatsnd(_audiodata)
        if _subtype is None:
            raise TypeError('Could not find audio MIME subtype')
        MIMENonMultipart.__init__(self, 'audio', _subtype, policy=policy,
                                  **_params)
        self.set_payload(_audiodata)
        _encoder(self)
<|endoftext|># Copyright (C) 2002-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Base class for MIME multipart/* type messages."""

__all__ = ['MIMEMultipart']

from email.mime.base import MIMEBase



class MIMEMultipart(MIMEBase):
    """Base class for MIME multipart/* type messages."""

    def __init__(self, _subtype='mixed', boundary=None, _subparts=None,
                 *, policy=None,
                 **_params):
        """Creates a multipart/* type message.

        By default, creates a multipart/mixed message, with proper
        Content-Type and MIME-Version headers.

        _subtype is the subtype of the multipart content type, defaulting to
        `mixed'.

        boundary is the multipart boundary string.  By default it is
        calculated as needed.

        _subparts is a sequence of initial subparts for the payload.  It
        must be an iterable object, such as a list.  You can always
        attach new subparts to the message by using the attach() method.

        Additional parameters for the Content-Type header are taken from the
        keyword arguments (or passed into the _params argument).
        """
        MIMEBase.__init__(self, 'multipart', _subtype, policy=policy, **_params)

        # Initialise _payload to an empty list as the Message superclass's
        # implementation of is_multipart assumes that _payload is a list for
        # multipart messages.
        self._payload = []

        if _subparts:
            for p in _subparts:
                self.attach(p)
        if boundary:
            self.set_boundary(boundary)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Base class for MIME specializations."""

__all__ = ['MIMEBase']

import email.policy

from email import message



class MIMEBase(message.Message):
    """Base class for MIME specializations."""

    def __init__(self, _maintype, _subtype, *, policy=None, **_params):
        """This constructor adds a Content-Type: and a MIME-Version: header.

        The Content-Type: header is taken from the _maintype and _subtype
        arguments.  Additional parameters for this header are taken from the
        keyword arguments.
        """
        if policy is None:
            policy = email.policy.compat32
        message.Message.__init__(self, policy=policy)
        ctype = '%s/%s' % (_maintype, _subtype)
        self.add_header('Content-Type', ctype, **_params)
        self['MIME-Version'] = '1.0'
<|endoftext|># Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Execute computations asynchronously using threads or processes."""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

from concurrent.futures._base import (FIRST_COMPLETED,
                                      FIRST_EXCEPTION,
                                      ALL_COMPLETED,
                                      CancelledError,
                                      TimeoutError,
                                      InvalidStateError,
                                      BrokenExecutor,
                                      Future,
                                      Executor,
                                      wait,
                                      as_completed)

__all__ = (
    'FIRST_COMPLETED',
    'FIRST_EXCEPTION',
    'ALL_COMPLETED',
    'CancelledError',
    'TimeoutError',
    'BrokenExecutor',
    'Future',
    'Executor',
    'wait',
    'as_completed',
    'ProcessPoolExecutor',
    'ThreadPoolExecutor',
)


def __dir__():
    return __all__ + ('__author__', '__doc__')


def __getattr__(name):
    global ProcessPoolExecutor, ThreadPoolExecutor

    if name == 'ProcessPoolExecutor':
        from .process import ProcessPoolExecutor as pe
        ProcessPoolExecutor = pe
        return pe

    if name == 'ThreadPoolExecutor':
        from .thread import ThreadPoolExecutor as te
        ThreadPoolExecutor = te
        return te

    raise AttributeError(f"module {__name__} has no attribute {name}")
<|endoftext|># Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The following diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     |        |     | 4, result |    |         |
|          |     | ...        |     |        |     | 3, except |    |         |
+----------+     +------------+     +--------+     +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
import traceback


_threads_wakeups = weakref.WeakKeyDictionary()
_global_shutdown = False


class _ThreadWakeup:
    def __init__(self):
        self._closed = False
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        if not self._closed:
            self._closed = True
            self._writer.close()
            self._reader.close()

    def wakeup(self):
        if not self._closed:
            self._writer.send_bytes(b"")

    def clear(self):
        if not self._closed:
            while self._reader.poll():
                self._reader.recv_bytes()


def _python_exit():
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeups.items())
    for _, thread_wakeup in items:
        # call not protected by ProcessPoolExecutor._shutdown_lock
        thread_wakeup.wakeup()
    for t, _ in items:
        t.join()

# Register for `_python_exit()` to be called just before joining all
# non-daemon threads. This is used instead of `atexit.register()` for
# compatibility with subinterpreters, which no longer support daemon threads.
# See bpo-39812 for context.
threading._register_atexit(_python_exit)

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1


# On Windows, WaitForMultipleObjects is used to wait for processes to finish.
# It can wait on, at most, 63 objects. There is an overhead of two objects:
# - the result queue reader
# - the thread wakeup reader
_MAX_WINDOWS_WORKERS = 63 - 2

# Hack to embed stringification of remote traceback in local traceback

class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb
    def __str__(self):
        return self.tb

class _ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        # Traceback object needs to be garbage-collected as its frames
        # contain references to all the objects in the exception scope
        self.exc.__traceback__ = None
        self.tb = '\n"""\n%s"""' % tb
    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)

def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc

class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

class _ResultItem(object):
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result

class _CallItem(object):
    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""
    def __init__(self, max_size=0, *, ctx, pending_work_items, shutdown_lock,
                 thread_wakeup):
        self.pending_work_items = pending_work_items
        self.shutdown_lock = shutdown_lock
        self.thread_wakeup = thread_wakeup
        super().__init__(max_size, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format(''.join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            with self.shutdown_lock:
                self.thread_wakeup.wakeup()
            # work_item can be None if another process terminated. In this
            # case, the executor_manager_thread fails all work_items
            # with BrokenProcessPool
            if work_item is not None:
                work_item.future.set_exception(e)
        else:
            super()._on_queue_feeder_error(e, obj)


def _get_chunks(*iterables, chunksize):
    """ Iterates over zip()ed iterables in chunks. """
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result,
                                     exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(_ResultItem(work_id, exception=exc))


def _process_worker(call_queue, result_queue, initializer, initargs):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item


class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """

    def __init__(self, executor):
        # Store references to necessary internals of the executor.

        # A _ThreadWakeup to allow waking up the queue_manager_thread from the
        # main Thread and avoid deadlocks caused by permanently locked queues.
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock

        # A weakref.ref to the ProcessPoolExecutor that owns this thread. Used
        # to determine if the ProcessPoolExecutor has been garbage collected
        # and that the manager can exit.
        # When the executor gets garbage collected, the weakref callback
        # will wake up the queue management thread so that it can terminate
        # if there is no pending work item.
        def weakref_cb(_,
                       thread_wakeup=self.thread_wakeup,
                       shutdown_lock=self.shutdown_lock):
            mp.util.debug('Executor collected: triggering callback for'
                          ' QueueManager wakeup')
            with shutdown_lock:
                thread_wakeup.wakeup()

        self.executor_reference = weakref.ref(executor, weakref_cb)

        # A list of the ctx.Process instances used as workers.
        self.processes = executor._processes

        # A ctx.Queue that will be filled with _CallItems derived from
        # _WorkItems for processing by the process workers.
        self.call_queue = executor._call_queue

        # A ctx.SimpleQueue of _ResultItems generated by the process workers.
        self.result_queue = executor._result_queue

        # A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        self.work_ids_queue = executor._work_ids

        # A dict mapping work ids to _WorkItems e.g.
        #     {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        self.pending_work_items = executor._pending_work_items

        super().__init__()

    def run(self):
        # Main loop for the executor manager thread.

        while True:
            self.add_call_item_to_queue()

            result_item, is_broken, cause = self.wait_result_broken_or_wakeup()

            if is_broken:
                self.terminate_broken(cause)
                return
            if result_item is not None:
                self.process_result_item(result_item)
                # Delete reference to result_item to avoid keeping references
                # while waiting on new results.
                del result_item

                # attempt to increment idle process count
                executor = self.executor_reference()
                if executor is not None:
                    executor._idle_worker_semaphore.release()
                del executor

            if self.is_shutting_down():
                self.flag_executor_shutting_down()

                # When only canceled futures remain in pending_work_items, our
                # next call to wait_result_broken_or_wakeup would hang forever.
                # This makes sure we have some running futures or none at all.
                self.add_call_item_to_queue()

                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not self.pending_work_items:
                    self.join_executor_internals()
                    return

    def add_call_item_to_queue(self):
        # Fills call_queue with _WorkItems from pending_work_items.
        # This function never blocks.
        while True:
            if self.call_queue.full():
                return
            try:
                work_id = self.work_ids_queue.get(block=False)
            except queue.Empty:
                return
            else:
                work_item = self.pending_work_items[work_id]

                if work_item.future.set_running_or_notify_cancel():
                    self.call_queue.put(_CallItem(work_id,
                                                  work_item.fn,
                                                  work_item.args,
                                                  work_item.kwargs),
                                        block=True)
                else:
                    del self.pending_work_items[work_id]
                    continue

    def wait_result_broken_or_wakeup(self):
        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        result_reader = self.result_queue._reader
        assert not self.thread_wakeup._closed
        wakeup_reader = self.thread_wakeup._reader
        readers = [result_reader, wakeup_reader]
        worker_sentinels = [p.sentinel for p in list(self.processes.values())]
        ready = mp.connection.wait(readers + worker_sentinels)

        cause = None
        is_broken = True
        result_item = None
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                is_broken = False
            except BaseException as e:
                cause = traceback.format_exception(type(e), e, e.__traceback__)

        elif wakeup_reader in ready:
            is_broken = False

        with self.shutdown_lock:
            self.thread_wakeup.clear()

        return result_item, is_broken, cause

    def process_result_item(self, result_item):
        # Process the received a result_item. This can be either the PID of a
        # worker that exited gracefully or a _ResultItem

        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert self.is_shutting_down()
            p = self.processes.pop(result_item)
            p.join()
            if not self.processes:
                self.join_executor_internals()
                return
        else:
            # Received a _ResultItem so mark the future as completed.
            work_item = self.pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)

    def is_shutting_down(self):
        # Check whether we should start shutting down the executor.
        executor = self.executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        return (_global_shutdown or executor is None
                or executor._shutdown_thread)

    def terminate_broken(self, cause):
        # Terminate the executor because it is in a broken state. The cause
        # argument can be used to display more information on the error that
        # lead the executor into becoming broken.

        # Mark the process pool broken so that submits fail right now.
        executor = self.executor_reference()
        if executor is not None:
            executor._broken = ('A child process terminated '
                                'abruptly, the process pool is not '
                                'usable anymore')
            executor._shutdown_thread = True
            executor = None

        # All pending tasks are to be marked failed with the following
        # BrokenProcessPool error
        bpe = BrokenProcessPool("A process in the process pool was "
                                "terminated abruptly while the future was "
                                "running or pending.")
        if cause is not None:
            bpe.__cause__ = _RemoteTraceback(
                f"\n'''\n{''.join(cause)}'''")

        # Mark pending tasks as failed.
        for work_id, work_item in self.pending_work_items.items():
            work_item.future.set_exception(bpe)
            # Delete references to object. See issue16284
            del work_item
        self.pending_work_items.clear()

        # Terminate remaining workers forcibly: the queues or their
        # locks may be in a dirty state and block forever.
        for p in self.processes.values():
            p.terminate()

        # clean up resources
        self.join_executor_internals()

    def flag_executor_shutting_down(self):
        # Flag the executor as shutting down and cancel remaining tasks if
        # requested as early as possible if it is not gc-ed yet.
        executor = self.executor_reference()
        if executor is not None:
            executor._shutdown_thread = True
            # Cancel pending work items if requested.
            if executor._cancel_pending_futures:
                # Cancel all pending futures and update pending_work_items
                # to only have futures that are currently running.
                new_pending_work_items = {}
                for work_id, work_item in self.pending_work_items.items():
                    if not work_item.future.cancel():
                        new_pending_work_items[work_id] = work_item
                self.pending_work_items = new_pending_work_items
                # Drain work_ids_queue since we no longer need to
                # add items to the call queue.
                while True:
                    try:
                        self.work_ids_queue.get_nowait()
                    except queue.Empty:
                        break
                # Make sure we do this only once to not waste time looping
                # on running processes over and over.
                executor._cancel_pending_futures = False

    def shutdown_workers(self):
        n_children_to_stop = self.get_n_children_alive()
        n_sentinels_sent = 0
        # Send the right number of sentinels, to make sure all children are
        # properly terminated.
        while (n_sentinels_sent < n_children_to_stop
                and self.get_n_children_alive() > 0):
            for i in range(n_children_to_stop - n_sentinels_sent):
                try:
                    self.call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except queue.Full:
                    break

    def join_executor_internals(self):
        self.shutdown_workers()
        # Release the queue's resources as soon as possible.
        self.call_queue.close()
        self.call_queue.join_thread()
        with self.shutdown_lock:
            self.thread_wakeup.close()
        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on Mac OS X.
        for p in self.processes.values():
            p.join()

    def get_n_children_alive(self):
        # This is an upper bound on the number of children alive.
        return sum(p.is_alive() for p in self.processes.values())


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        import multiprocessing.synchronize
    except ImportError:
        _system_limited = (
            "This Python build lacks multiprocessing.synchronize, usually due "
            "to named semaphores being unavailable on this platform."
        )
        raise NotImplementedError(_system_limited)
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # indetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = ("system provides too few semaphores (%d"
                       " available, 256 necessary)" % nsems_max)
    raise NotImplementedError(_system_limited)


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


class BrokenProcessPool(_base.BrokenExecutor):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


class ProcessPoolExecutor(_base.Executor):
    def __init__(self, max_workers=None, mp_context=None,
                 initializer=None, initargs=()):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
            mp_context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: A callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                self._max_workers = min(_MAX_WINDOWS_WORKERS,
                                        self._max_workers)
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            elif (sys.platform == 'win32' and
                max_workers > _MAX_WINDOWS_WORKERS):
                raise ValueError(
                    f"max_workers must be <= {_MAX_WINDOWS_WORKERS}")

            self._max_workers = max_workers

        if mp_context is None:
            mp_context = mp.get_context()
        self._mp_context = mp_context

        # https://github.com/python/cpython/issues/90622
        self._safe_to_dynamically_spawn_children = (
                self._mp_context.get_start_method(allow_none=False) != "fork")

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        self._initializer = initializer
        self._initargs = initargs

        # Management thread
        self._executor_manager_thread = None

        # Map of pids to processes
        self._processes = {}

        # Shutdown is a two-step process.
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._idle_worker_semaphore = threading.Semaphore(0)
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}
        self._cancel_pending_futures = False

        # _ThreadWakeup is a communication channel used to interrupt the wait
        # of the main loop of executor_manager_thread from another thread (e.g.
        # when calling executor.submit or executor.shutdown). We do not use the
        # _result_queue to send wakeup signals to the executor_manager_thread
        # as it could result in a deadlock if a worker process dies with the
        # _result_queue write lock still acquired.
        #
        # _shutdown_lock must be locked to access _ThreadWakeup.
        self._executor_manager_thread_wakeup = _ThreadWakeup()

        # Create communication channels for the executor
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        queue_size = self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(
            max_size=queue_size, ctx=self._mp_context,
            pending_work_items=self._pending_work_items,
            shutdown_lock=self._shutdown_lock,
            thread_wakeup=self._executor_manager_thread_wakeup)
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True
        self._result_queue = mp_context.SimpleQueue()
        self._work_ids = queue.Queue()

    def _start_executor_manager_thread(self):
        if self._executor_manager_thread is None:
            # Start the processes so that their sentinels are known.
            if not self._safe_to_dynamically_spawn_children:  # ie, using fork.
                self._launch_processes()
            self._executor_manager_thread = _ExecutorManagerThread(self)
            self._executor_manager_thread.start()
            _threads_wakeups[self._executor_manager_thread] = \
                self._executor_manager_thread_wakeup

    def _adjust_process_count(self):
        # if there's an idle process, we don't need to spawn a new one.
        if self._idle_worker_semaphore.acquire(blocking=False):
            return

        process_count = len(self._processes)
        if process_count < self._max_workers:
            # Assertion disabled as this codepath is also used to replace a
            # worker that unexpectedly dies, even when using the 'fork' start
            # method. That means there is still a potential deadlock bug. If a
            # 'fork' mp_context worker dies, we'll be forking a new one when
            # we know a thread is running (self._executor_manager_thread).
            #assert self._safe_to_dynamically_spawn_children or not self._executor_manager_thread, 'https://github.com/python/cpython/issues/90622'
            self._spawn_process()

    def _launch_processes(self):
        # https://github.com/python/cpython/issues/90622
        assert not self._executor_manager_thread, (
                'Processes cannot be fork()ed after the thread has started, '
                'deadlock in the child processes could result.')
        for _ in range(len(self._processes), self._max_workers):
            self._spawn_process()

    def _spawn_process(self):
        p = self._mp_context.Process(
            target=_process_worker,
            args=(self._call_queue,
                  self._result_queue,
                  self._initializer,
                  self._initargs))
        p.start()
        self._processes[p.pid] = p

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool(self._broken)
            if self._shutdown_thread:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after '
                                   'interpreter shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._executor_manager_thread_wakeup.wakeup()

            if self._safe_to_dynamically_spawn_children:
                self._adjust_process_count()
            self._start_executor_manager_thread()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(partial(_process_chunk, fn),
                              _get_chunks(*iterables, chunksize=chunksize),
                              timeout=timeout)
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            self._cancel_pending_futures = cancel_futures
            self._shutdown_thread = True
            if self._executor_manager_thread_wakeup is not None:
                # Wake up queue management thread
                self._executor_manager_thread_wakeup.wakeup()

        if self._executor_manager_thread is not None and wait:
            self._executor_manager_thread.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._executor_manager_thread = None
        self._call_queue = None
        if self._result_queue is not None and wait:
            self._result_queue.close()
        self._result_queue = None
        self._processes = None
        self._executor_manager_thread_wakeup = None

    shutdown.__doc__ = _base.Executor.shutdown.__doc__
<|endoftext|># Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

import collections
import logging
import threading
import time
import types

FIRST_COMPLETED = 'FIRST_COMPLETED'
FIRST_EXCEPTION = 'FIRST_EXCEPTION'
ALL_COMPLETED = 'ALL_COMPLETED'
_AS_COMPLETED = '_AS_COMPLETED'

# Possible future states (for internal use by the futures package).
PENDING = 'PENDING'
RUNNING = 'RUNNING'
# The future was cancelled by the user...
CANCELLED = 'CANCELLED'
# ...and _Waiter.add_cancelled() was called by a worker.
CANCELLED_AND_NOTIFIED = 'CANCELLED_AND_NOTIFIED'
FINISHED = 'FINISHED'

_FUTURE_STATES = [
    PENDING,
    RUNNING,
    CANCELLED,
    CANCELLED_AND_NOTIFIED,
    FINISHED
]

_STATE_TO_DESCRIPTION_MAP = {
    PENDING: "pending",
    RUNNING: "running",
    CANCELLED: "cancelled",
    CANCELLED_AND_NOTIFIED: "cancelled",
    FINISHED: "finished"
}

# Logger for internal use by the futures package.
LOGGER = logging.getLogger("concurrent.futures")

class Error(Exception):
    """Base class for all future-related exceptions."""
    pass

class CancelledError(Error):
    """The Future was cancelled."""
    pass

class TimeoutError(Error):
    """The operation exceeded the given deadline."""
    pass

class InvalidStateError(Error):
    """The operation is not allowed in this state."""
    pass

class _Waiter(object):
    """Provides the event that wait() and as_completed() block on."""
    def __init__(self):
        self.event = threading.Event()
        self.finished_futures = []

    def add_result(self, future):
        self.finished_futures.append(future)

    def add_exception(self, future):
        self.finished_futures.append(future)

    def add_cancelled(self, future):
        self.finished_futures.append(future)

class _AsCompletedWaiter(_Waiter):
    """Used by as_completed()."""

    def __init__(self):
        super(_AsCompletedWaiter, self).__init__()
        self.lock = threading.Lock()

    def add_result(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_result(future)
            self.event.set()

    def add_exception(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_exception(future)
            self.event.set()

    def add_cancelled(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_cancelled(future)
            self.event.set()

class _FirstCompletedWaiter(_Waiter):
    """Used by wait(return_when=FIRST_COMPLETED)."""

    def add_result(self, future):
        super().add_result(future)
        self.event.set()

    def add_exception(self, future):
        super().add_exception(future)
        self.event.set()

    def add_cancelled(self, future):
        super().add_cancelled(future)
        self.event.set()

class _AllCompletedWaiter(_Waiter):
    """Used by wait(return_when=FIRST_EXCEPTION and ALL_COMPLETED)."""

    def __init__(self, num_pending_calls, stop_on_exception):
        self.num_pending_calls = num_pending_calls
        self.stop_on_exception = stop_on_exception
        self.lock = threading.Lock()
        super().__init__()

    def _decrement_pending_calls(self):
        with self.lock:
            self.num_pending_calls -= 1
            if not self.num_pending_calls:
                self.event.set()

    def add_result(self, future):
        super().add_result(future)
        self._decrement_pending_calls()

    def add_exception(self, future):
        super().add_exception(future)
        if self.stop_on_exception:
            self.event.set()
        else:
            self._decrement_pending_calls()

    def add_cancelled(self, future):
        super().add_cancelled(future)
        self._decrement_pending_calls()

class _AcquireFutures(object):
    """A context manager that does an ordered acquire of Future conditions."""

    def __init__(self, futures):
        self.futures = sorted(futures, key=id)

    def __enter__(self):
        for future in self.futures:
            future._condition.acquire()

    def __exit__(self, *args):
        for future in self.futures:
            future._condition.release()

def _create_and_install_waiters(fs, return_when):
    if return_when == _AS_COMPLETED:
        waiter = _AsCompletedWaiter()
    elif return_when == FIRST_COMPLETED:
        waiter = _FirstCompletedWaiter()
    else:
        pending_count = sum(
                f._state not in [CANCELLED_AND_NOTIFIED, FINISHED] for f in fs)

        if return_when == FIRST_EXCEPTION:
            waiter = _AllCompletedWaiter(pending_count, stop_on_exception=True)
        elif return_when == ALL_COMPLETED:
            waiter = _AllCompletedWaiter(pending_count, stop_on_exception=False)
        else:
            raise ValueError("Invalid return condition: %r" % return_when)

    for f in fs:
        f._waiters.append(waiter)

    return waiter


def _yield_finished_futures(fs, waiter, ref_collect):
    """
    Iterate on the list *fs*, yielding finished futures one by one in
    reverse order.
    Before yielding a future, *waiter* is removed from its waiters
    and the future is removed from each set in the collection of sets
    *ref_collect*.

    The aim of this function is to avoid keeping stale references after
    the future is yielded and before the iterator resumes.
    """
    while fs:
        f = fs[-1]
        for futures_set in ref_collect:
            futures_set.remove(f)
        with f._condition:
            f._waiters.remove(waiter)
        del f
        # Careful not to keep a reference to the popped value
        yield fs.pop()


def as_completed(fs, timeout=None):
    """An iterator over the given futures that yields each as it completes.

    Args:
        fs: The sequence of Futures (possibly created by different Executors) to
            iterate over.
        timeout: The maximum number of seconds to wait. If None, then there
            is no limit on the wait time.

    Returns:
        An iterator that yields the given Futures as they complete (finished or
        cancelled). If any given Futures are duplicated, they will be returned
        once.

    Raises:
        TimeoutError: If the entire result iterator could not be generated
            before the given timeout.
    """
    if timeout is not None:
        end_time = timeout + time.monotonic()

    fs = set(fs)
    total_futures = len(fs)
    with _AcquireFutures(fs):
        finished = set(
                f for f in fs
                if f._state in [CANCELLED_AND_NOTIFIED, FINISHED])
        pending = fs - finished
        waiter = _create_and_install_waiters(fs, _AS_COMPLETED)
    finished = list(finished)
    try:
        yield from _yield_finished_futures(finished, waiter,
                                           ref_collect=(fs,))

        while pending:
            if timeout is None:
                wait_timeout = None
            else:
                wait_timeout = end_time - time.monotonic()
                if wait_timeout < 0:
                    raise TimeoutError(
                            '%d (of %d) futures unfinished' % (
                            len(pending), total_futures))

            waiter.event.wait(wait_timeout)

            with waiter.lock:
                finished = waiter.finished_futures
                waiter.finished_futures = []
                waiter.event.clear()

            # reverse to keep finishing order
            finished.reverse()
            yield from _yield_finished_futures(finished, waiter,
                                               ref_collect=(fs, pending))

    finally:
        # Remove waiter from unfinished futures
        for f in fs:
            with f._condition:
                f._waiters.remove(waiter)

DoneAndNotDoneFutures = collections.namedtuple(
        'DoneAndNotDoneFutures', 'done not_done')
def wait(fs, timeout=None, return_when=ALL_COMPLETED):
    """Wait for the futures in the given sequence to complete.

    Args:
        fs: The sequence of Futures (possibly created by different Executors) to
            wait upon.
        timeout: The maximum number of seconds to wait. If None, then there
            is no limit on the wait time.
        return_when: Indicates when this function should return. The options
            are:

            FIRST_COMPLETED - Return when any future finishes or is
                              cancelled.
            FIRST_EXCEPTION - Return when any future finishes by raising an
                              exception. If no future raises an exception
                              then it is equivalent to ALL_COMPLETED.
            ALL_COMPLETED -   Return when all futures finish or are cancelled.

    Returns:
        A named 2-tuple of sets. The first set, named 'done', contains the
        futures that completed (is finished or cancelled) before the wait
        completed. The second set, named 'not_done', contains uncompleted
        futures. Duplicate futures given to *fs* are removed and will be 
        returned only once.
    """
    fs = set(fs)
    with _AcquireFutures(fs):
        done = {f for f in fs
                   if f._state in [CANCELLED_AND_NOTIFIED, FINISHED]}
        not_done = fs - done
        if (return_when == FIRST_COMPLETED) and done:
            return DoneAndNotDoneFutures(done, not_done)
        elif (return_when == FIRST_EXCEPTION) and done:
            if any(f for f in done
                   if not f.cancelled() and f.exception() is not None):
                return DoneAndNotDoneFutures(done, not_done)

        if len(done) == len(fs):
            return DoneAndNotDoneFutures(done, not_done)

        waiter = _create_and_install_waiters(fs, return_when)

    waiter.event.wait(timeout)
    for f in fs:
        with f._condition:
            f._waiters.remove(waiter)

    done.update(waiter.finished_futures)
    return DoneAndNotDoneFutures(done, fs - done)


def _result_or_cancel(fut, timeout=None):
    try:
        try:
            return fut.result(timeout)
        finally:
            fut.cancel()
    finally:
        # Break a reference cycle with the exception in self._exception
        del fut


class Future(object):
    """Represents the result of an asynchronous computation."""

    def __init__(self):
        """Initializes the future. Should not be called by clients."""
        self._condition = threading.Condition()
        self._state = PENDING
        self._result = None
        self._exception = None
        self._waiters = []
        self._done_callbacks = []

    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except Exception:
                LOGGER.exception('exception calling callback for %r', self)

    def __repr__(self):
        with self._condition:
            if self._state == FINISHED:
                if self._exception:
                    return '<%s at %#x state=%s raised %s>' % (
                        self.__class__.__name__,
                        id(self),
                        _STATE_TO_DESCRIPTION_MAP[self._state],
                        self._exception.__class__.__name__)
                else:
                    return '<%s at %#x state=%s returned %s>' % (
                        self.__class__.__name__,
                        id(self),
                        _STATE_TO_DESCRIPTION_MAP[self._state],
                        self._result.__class__.__name__)
            return '<%s at %#x state=%s>' % (
                    self.__class__.__name__,
                    id(self),
                   _STATE_TO_DESCRIPTION_MAP[self._state])

    def cancel(self):
        """Cancel the future if possible.

        Returns True if the future was cancelled, False otherwise. A future
        cannot be cancelled if it is running or has already completed.
        """
        with self._condition:
            if self._state in [RUNNING, FINISHED]:
                return False

            if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                return True

            self._state = CANCELLED
            self._condition.notify_all()

        self._invoke_callbacks()
        return True

    def cancelled(self):
        """Return True if the future was cancelled."""
        with self._condition:
            return self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]

    def running(self):
        """Return True if the future is currently executing."""
        with self._condition:
            return self._state == RUNNING

    def done(self):
        """Return True if the future was cancelled or finished executing."""
        with self._condition:
            return self._state in [CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED]

    def __get_result(self):
        if self._exception:
            try:
                raise self._exception
            finally:
                # Break a reference cycle with the exception in self._exception
                self = None
        else:
            return self._result

    def add_done_callback(self, fn):
        """Attaches a callable that will be called when the future finishes.

        Args:
            fn: A callable that will be called with this future as its only
                argument when the future completes or is cancelled. The callable
                will always be called by a thread in the same process in which
                it was added. If the future has already completed or been
                cancelled then the callable will be called immediately. These
                callables are called in the order that they were added.
        """
        with self._condition:
            if self._state not in [CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED]:
                self._done_callbacks.append(fn)
                return
        try:
            fn(self)
        except Exception:
            LOGGER.exception('exception calling callback for %r', self)

    def result(self, timeout=None):
        """Return the result of the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the result if the future
                isn't done. If None, then there is no limit on the wait time.

        Returns:
            The result of the call that the future represents.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
            Exception: If the call raised then that exception will be raised.
        """
        try:
            with self._condition:
                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()

                self._condition.wait(timeout)

                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()
                else:
                    raise TimeoutError()
        finally:
            # Break a reference cycle with the exception in self._exception
            self = None

    def exception(self, timeout=None):
        """Return the exception raised by the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the exception if the
                future isn't done. If None, then there is no limit on the wait
                time.

        Returns:
            The exception raised by the call that the future represents or None
            if the call completed without raising.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
        """

        with self._condition:
            if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                raise CancelledError()
            elif self._state == FINISHED:
                return self._exception

            self._condition.wait(timeout)

            if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                raise CancelledError()
            elif self._state == FINISHED:
                return self._exception
            else:
                raise TimeoutError()

    # The following methods should only be used by Executors and in tests.
    def set_running_or_notify_cancel(self):
        """Mark the future as running or process any cancel notifications.

        Should only be used by Executor implementations and unit tests.

        If the future has been cancelled (cancel() was called and returned
        True) then any threads waiting on the future completing (though calls
        to as_completed() or wait()) are notified and False is returned.

        If the future was not cancelled then it is put in the running state
        (future calls to running() will return True) and True is returned.

        This method should be called by Executor implementations before
        executing the work associated with this future. If this method returns
        False then the work should not be executed.

        Returns:
            False if the Future was cancelled, True otherwise.

        Raises:
            RuntimeError: if this method was already called or if set_result()
                or set_exception() was called.
        """
        with self._condition:
            if self._state == CANCELLED:
                self._state = CANCELLED_AND_NOTIFIED
                for waiter in self._waiters:
                    waiter.add_cancelled(self)
                # self._condition.notify_all() is not necessary because
                # self.cancel() triggers a notification.
                return False
            elif self._state == PENDING:
                self._state = RUNNING
                return True
            else:
                LOGGER.critical('Future %s in unexpected state: %s',
                                id(self),
                                self._state)
                raise RuntimeError('Future in unexpected state')

    def set_result(self, result):
        """Sets the return value of work associated with the future.

        Should only be used by Executor implementations and unit tests.
        """
        with self._condition:
            if self._state in {CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED}:
                raise InvalidStateError('{}: {!r}'.format(self._state, self))
            self._result = result
            self._state = FINISHED
            for waiter in self._waiters:
                waiter.add_result(self)
            self._condition.notify_all()
        self._invoke_callbacks()

    def set_exception(self, exception):
        """Sets the result of the future as being the given exception.

        Should only be used by Executor implementations and unit tests.
        """
        with self._condition:
            if self._state in {CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED}:
                raise InvalidStateError('{}: {!r}'.format(self._state, self))
            self._exception = exception
            self._state = FINISHED
            for waiter in self._waiters:
                waiter.add_exception(self)
            self._condition.notify_all()
        self._invoke_callbacks()

    __class_getitem__ = classmethod(types.GenericAlias)

class Executor(object):
    """This is an abstract base class for concrete asynchronous executors."""

    def submit(self, fn, /, *args, **kwargs):
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        raise NotImplementedError()

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if timeout is not None:
            end_time = timeout + time.monotonic()

        fs = [self.submit(fn, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield _result_or_cancel(fs.pop())
                    else:
                        yield _result_or_cancel(fs.pop(), end_time - time.monotonic())
            finally:
                for future in fs:
                    future.cancel()
        return result_iterator()

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class BrokenExecutor(RuntimeError):
    """
    Raised when a executor has become non-functional after a severe failure.
    """
<|endoftext|># Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ThreadPoolExecutor."""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os


_threads_queues = weakref.WeakKeyDictionary()
_shutdown = False
# Lock that ensures that new workers are not created while the interpreter is
# shutting down. Must be held while mutating _threads_queues and _shutdown.
_global_shutdown_lock = threading.Lock()

def _python_exit():
    global _shutdown
    with _global_shutdown_lock:
        _shutdown = True
    items = list(_threads_queues.items())
    for t, q in items:
        q.put(None)
    for t, q in items:
        t.join()

# Register for `_python_exit()` to be called just before joining all
# non-daemon threads. This is used instead of `atexit.register()` for
# compatibility with subinterpreters, which no longer support daemon threads.
# See bpo-39812 for context.
threading._register_atexit(_python_exit)

# At fork, reinitialize the `_global_shutdown_lock` lock in the child process
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_global_shutdown_lock.acquire,
                        after_in_child=_global_shutdown_lock._at_fork_reinit,
                        after_in_parent=_global_shutdown_lock.release)


class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)

    __class_getitem__ = classmethod(types.GenericAlias)


def _worker(executor_reference, work_queue, initializer, initargs):
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return
    try:
        while True:
            work_item = work_queue.get(block=True)
            if work_item is not None:
                work_item.run()
                # Delete references to object. See issue16284
                del work_item

                # attempt to increment idle count
                executor = executor_reference()
                if executor is not None:
                    executor._idle_semaphore.release()
                del executor
                continue

            executor = executor_reference()
            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown.
            if _shutdown or executor is None or executor._shutdown:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown = True
                # Notice other workers
                work_queue.put(None)
                return
            del executor
    except BaseException:
        _base.LOGGER.critical('Exception in worker', exc_info=True)


class BrokenThreadPool(_base.BrokenExecutor):
    """
    Raised when a worker thread in a ThreadPoolExecutor failed initializing.
    """


class ThreadPoolExecutor(_base.Executor):

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__

    def __init__(self, max_workers=None, thread_name_prefix='',
                 initializer=None, initargs=()):
        """Initializes a new ThreadPoolExecutor instance.

        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
            initializer: A callable used to initialize worker threads.
            initargs: A tuple of arguments to pass to the initializer.
        """
        if max_workers is None:
            # ThreadPoolExecutor is often used to:
            # * CPU bound task which releases GIL
            # * I/O bound task (which releases GIL, of course)
            #
            # We use cpu_count + 4 for both types of tasks.
            # But we limit it to 32 to avoid consuming surprisingly large resource
            # on many core machine.
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._max_workers = max_workers
        self._work_queue = queue.SimpleQueue()
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = set()
        self._broken = False
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread_name_prefix = (thread_name_prefix or
                                    ("ThreadPoolExecutor-%d" % self._counter()))
        self._initializer = initializer
        self._initargs = initargs

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock, _global_shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _shutdown:
                raise RuntimeError('cannot schedule new futures after '
                                   'interpreter shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._work_queue.put(w)
            self._adjust_thread_count()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=_worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs))
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue

    def _initializer_failed(self):
        with self._shutdown_lock:
            self._broken = ('A thread initializer failed, the thread pool '
                            'is not usable anymore')
            # Drain work queue and mark pending futures failed
            while True:
                try:
                    work_item = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if work_item is not None:
                    work_item.future.set_exception(BrokenThreadPool(self._broken))

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            self._shutdown = True
            if cancel_futures:
                # Drain all work items from the queue, and then cancel their
                # associated futures.
                while True:
                    try:
                        work_item = self._work_queue.get_nowait()
                    except queue.Empty:
                        break
                    if work_item is not None:
                        work_item.future.cancel()

            # Send a wake-up to prevent threads calling
            # _work_queue.get(block=True) from permanently blocking.
            self._work_queue.put(None)
        if wait:
            for t in self._threads:
                t.join()
    shutdown.__doc__ = _base.Executor.shutdown.__doc__
<|endoftext|># Copyright 2001-2019 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Configuration functions for the logging package for Python. The core package
is based on PEP 282 and comments thereto in comp.lang.python, and influenced
by Apache's log4j system.

Copyright (C) 2001-2019 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!
"""

import errno
import io
import logging
import logging.handlers
import re
import struct
import sys
import threading
import traceback

from socketserver import ThreadingTCPServer, StreamRequestHandler


DEFAULT_LOGGING_CONFIG_PORT = 9030

RESET_ERROR = errno.ECONNRESET

#
#   The following code implements a socket listener for on-the-fly
#   reconfiguration of logging.
#
#   _listener holds the server object doing the listening
_listener = None

def fileConfig(fname, defaults=None, disable_existing_loggers=True, encoding=None):
    """
    Read the logging configuration from a ConfigParser-format file.

    This can be called several times from an application, allowing an end user
    the ability to select from various pre-canned configurations (if the
    developer provides a mechanism to present the choices and load the chosen
    configuration).
    """
    import configparser

    if isinstance(fname, configparser.RawConfigParser):
        cp = fname
    else:
        cp = configparser.ConfigParser(defaults)
        if hasattr(fname, 'readline'):
            cp.read_file(fname)
        else:
            encoding = io.text_encoding(encoding)
            cp.read(fname, encoding=encoding)

    formatters = _create_formatters(cp)

    # critical section
    logging._acquireLock()
    try:
        _clearExistingHandlers()

        # Handlers add themselves to logging._handlers
        handlers = _install_handlers(cp, formatters)
        _install_loggers(cp, handlers, disable_existing_loggers)
    finally:
        logging._releaseLock()


def _resolve(name):
    """Resolve a dotted name to a global object."""
    name = name.split('.')
    used = name.pop(0)
    found = __import__(used)
    for n in name:
        used = used + '.' + n
        try:
            found = getattr(found, n)
        except AttributeError:
            __import__(used)
            found = getattr(found, n)
    return found

def _strip_spaces(alist):
    return map(str.strip, alist)

def _create_formatters(cp):
    """Create and return formatters"""
    flist = cp["formatters"]["keys"]
    if not len(flist):
        return {}
    flist = flist.split(",")
    flist = _strip_spaces(flist)
    formatters = {}
    for form in flist:
        sectname = "formatter_%s" % form
        fs = cp.get(sectname, "format", raw=True, fallback=None)
        dfs = cp.get(sectname, "datefmt", raw=True, fallback=None)
        stl = cp.get(sectname, "style", raw=True, fallback='%')
        c = logging.Formatter
        class_name = cp[sectname].get("class")
        if class_name:
            c = _resolve(class_name)
        f = c(fs, dfs, stl)
        formatters[form] = f
    return formatters


def _install_handlers(cp, formatters):
    """Install and return handlers"""
    hlist = cp["handlers"]["keys"]
    if not len(hlist):
        return {}
    hlist = hlist.split(",")
    hlist = _strip_spaces(hlist)
    handlers = {}
    fixups = [] #for inter-handler references
    for hand in hlist:
        section = cp["handler_%s" % hand]
        klass = section["class"]
        fmt = section.get("formatter", "")
        try:
            klass = eval(klass, vars(logging))
        except (AttributeError, NameError):
            klass = _resolve(klass)
        args = section.get("args", '()')
        args = eval(args, vars(logging))
        kwargs = section.get("kwargs", '{}')
        kwargs = eval(kwargs, vars(logging))
        h = klass(*args, **kwargs)
        h.name = hand
        if "level" in section:
            level = section["level"]
            h.setLevel(level)
        if len(fmt):
            h.setFormatter(formatters[fmt])
        if issubclass(klass, logging.handlers.MemoryHandler):
            target = section.get("target", "")
            if len(target): #the target handler may not be loaded yet, so keep for later...
                fixups.append((h, target))
        handlers[hand] = h
    #now all handlers are loaded, fixup inter-handler references...
    for h, t in fixups:
        h.setTarget(handlers[t])
    return handlers

def _handle_existing_loggers(existing, child_loggers, disable_existing):
    """
    When (re)configuring logging, handle loggers which were in the previous
    configuration but are not in the new configuration. There's no point
    deleting them as other threads may continue to hold references to them;
    and by disabling them, you stop them doing any logging.

    However, don't disable children of named loggers, as that's probably not
    what was intended by the user. Also, allow existing loggers to NOT be
    disabled if disable_existing is false.
    """
    root = logging.root
    for log in existing:
        logger = root.manager.loggerDict[log]
        if log in child_loggers:
            if not isinstance(logger, logging.PlaceHolder):
                logger.setLevel(logging.NOTSET)
                logger.handlers = []
                logger.propagate = True
        else:
            logger.disabled = disable_existing

def _install_loggers(cp, handlers, disable_existing):
    """Create and install loggers"""

    # configure the root first
    llist = cp["loggers"]["keys"]
    llist = llist.split(",")
    llist = list(_strip_spaces(llist))
    llist.remove("root")
    section = cp["logger_root"]
    root = logging.root
    log = root
    if "level" in section:
        level = section["level"]
        log.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    hlist = section["handlers"]
    if len(hlist):
        hlist = hlist.split(",")
        hlist = _strip_spaces(hlist)
        for hand in hlist:
            log.addHandler(handlers[hand])

    #and now the others...
    #we don't want to lose the existing loggers,
    #since other threads may have pointers to them.
    #existing is set to contain all existing loggers,
    #and as we go through the new configuration we
    #remove any which are configured. At the end,
    #what's left in existing is the set of loggers
    #which were in the previous configuration but
    #which are not in the new configuration.
    existing = list(root.manager.loggerDict.keys())
    #The list needs to be sorted so that we can
    #avoid disabling child loggers of explicitly
    #named loggers. With a sorted list it is easier
    #to find the child loggers.
    existing.sort()
    #We'll keep the list of existing loggers
    #which are children of named loggers here...
    child_loggers = []
    #now set up the new ones...
    for log in llist:
        section = cp["logger_%s" % log]
        qn = section["qualname"]
        propagate = section.getint("propagate", fallback=1)
        logger = logging.getLogger(qn)
        if qn in existing:
            i = existing.index(qn) + 1 # start with the entry after qn
            prefixed = qn + "."
            pflen = len(prefixed)
            num_existing = len(existing)
            while i < num_existing:
                if existing[i][:pflen] == prefixed:
                    child_loggers.append(existing[i])
                i += 1
            existing.remove(qn)
        if "level" in section:
            level = section["level"]
            logger.setLevel(level)
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.propagate = propagate
        logger.disabled = 0
        hlist = section["handlers"]
        if len(hlist):
            hlist = hlist.split(",")
            hlist = _strip_spaces(hlist)
            for hand in hlist:
                logger.addHandler(handlers[hand])

    #Disable any old loggers. There's no point deleting
    #them as other threads may continue to hold references
    #and by disabling them, you stop them doing any logging.
    #However, don't disable children of named loggers, as that's
    #probably not what was intended by the user.
    #for log in existing:
    #    logger = root.manager.loggerDict[log]
    #    if log in child_loggers:
    #        logger.level = logging.NOTSET
    #        logger.handlers = []
    #        logger.propagate = 1
    #    elif disable_existing_loggers:
    #        logger.disabled = 1
    _handle_existing_loggers(existing, child_loggers, disable_existing)


def _clearExistingHandlers():
    """Clear and close existing handlers"""
    logging._handlers.clear()
    logging.shutdown(logging._handlerList[:])
    del logging._handlerList[:]


IDENTIFIER = re.compile('^[a-z_][a-z0-9_]*$', re.I)


def valid_ident(s):
    m = IDENTIFIER.match(s)
    if not m:
        raise ValueError('Not a valid Python identifier: %r' % s)
    return True


class ConvertingMixin(object):
    """For ConvertingXXX's, this mixin class provides common functions"""

    def convert_with_key(self, key, value, replace=True):
        result = self.configurator.convert(value)
        #If the converted value is different, save for next time
        if value is not result:
            if replace:
                self[key] = result
            if type(result) in (ConvertingDict, ConvertingList,
                               ConvertingTuple):
                result.parent = self
                result.key = key
        return result

    def convert(self, value):
        result = self.configurator.convert(value)
        if value is not result:
            if type(result) in (ConvertingDict, ConvertingList,
                               ConvertingTuple):
                result.parent = self
        return result


# The ConvertingXXX classes are wrappers around standard Python containers,
# and they serve to convert any suitable values in the container. The
# conversion converts base dicts, lists and tuples to their wrapped
# equivalents, whereas strings which match a conversion format are converted
# appropriately.
#
# Each wrapper should have a configurator attribute holding the actual
# configurator to use for conversion.

class ConvertingDict(dict, ConvertingMixin):
    """A converting dictionary wrapper."""

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        return self.convert_with_key(key, value)

    def get(self, key, default=None):
        value = dict.get(self, key, default)
        return self.convert_with_key(key, value)

    def pop(self, key, default=None):
        value = dict.pop(self, key, default)
        return self.convert_with_key(key, value, replace=False)

class ConvertingList(list, ConvertingMixin):
    """A converting list wrapper."""
    def __getitem__(self, key):
        value = list.__getitem__(self, key)
        return self.convert_with_key(key, value)

    def pop(self, idx=-1):
        value = list.pop(self, idx)
        return self.convert(value)

class ConvertingTuple(tuple, ConvertingMixin):
    """A converting tuple wrapper."""
    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        # Can't replace a tuple entry.
        return self.convert_with_key(key, value, replace=False)

class BaseConfigurator(object):
    """
    The configurator base class which defines some useful defaults.
    """

    CONVERT_PATTERN = re.compile(r'^(?P<prefix>[a-z]+)://(?P<suffix>.*)$')

    WORD_PATTERN = re.compile(r'^\s*(\w+)\s*')
    DOT_PATTERN = re.compile(r'^\.\s*(\w+)\s*')
    INDEX_PATTERN = re.compile(r'^\[\s*(\w+)\s*\]\s*')
    DIGIT_PATTERN = re.compile(r'^\d+$')

    value_converters = {
        'ext' : 'ext_convert',
        'cfg' : 'cfg_convert',
    }

    # We might want to use a different one, e.g. importlib
    importer = staticmethod(__import__)

    def __init__(self, config):
        self.config = ConvertingDict(config)
        self.config.configurator = self

    def resolve(self, s):
        """
        Resolve strings to objects using standard import and attribute
        syntax.
        """
        name = s.split('.')
        used = name.pop(0)
        try:
            found = self.importer(used)
            for frag in name:
                used += '.' + frag
                try:
                    found = getattr(found, frag)
                except AttributeError:
                    self.importer(used)
                    found = getattr(found, frag)
            return found
        except ImportError:
            e, tb = sys.exc_info()[1:]
            v = ValueError('Cannot resolve %r: %s' % (s, e))
            v.__cause__, v.__traceback__ = e, tb
            raise v

    def ext_convert(self, value):
        """Default converter for the ext:// protocol."""
        return self.resolve(value)

    def cfg_convert(self, value):
        """Default converter for the cfg:// protocol."""
        rest = value
        m = self.WORD_PATTERN.match(rest)
        if m is None:
            raise ValueError("Unable to convert %r" % value)
        else:
            rest = rest[m.end():]
            d = self.config[m.groups()[0]]
            #print d, rest
            while rest:
                m = self.DOT_PATTERN.match(rest)
                if m:
                    d = d[m.groups()[0]]
                else:
                    m = self.INDEX_PATTERN.match(rest)
                    if m:
                        idx = m.groups()[0]
                        if not self.DIGIT_PATTERN.match(idx):
                            d = d[idx]
                        else:
                            try:
                                n = int(idx) # try as number first (most likely)
                                d = d[n]
                            except TypeError:
                                d = d[idx]
                if m:
                    rest = rest[m.end():]
                else:
                    raise ValueError('Unable to convert '
                                     '%r at %r' % (value, rest))
        #rest should be empty
        return d

    def convert(self, value):
        """
        Convert values to an appropriate type. dicts, lists and tuples are
        replaced by their converting alternatives. Strings are checked to
        see if they have a conversion format and are converted if they do.
        """
        if not isinstance(value, ConvertingDict) and isinstance(value, dict):
            value = ConvertingDict(value)
            value.configurator = self
        elif not isinstance(value, ConvertingList) and isinstance(value, list):
            value = ConvertingList(value)
            value.configurator = self
        elif not isinstance(value, ConvertingTuple) and\
                 isinstance(value, tuple) and not hasattr(value, '_fields'):
            value = ConvertingTuple(value)
            value.configurator = self
        elif isinstance(value, str): # str for py3k
            m = self.CONVERT_PATTERN.match(value)
            if m:
                d = m.groupdict()
                prefix = d['prefix']
                converter = self.value_converters.get(prefix, None)
                if converter:
                    suffix = d['suffix']
                    converter = getattr(self, converter)
                    value = converter(suffix)
        return value

    def configure_custom(self, config):
        """Configure an object with a user-supplied factory."""
        c = config.pop('()')
        if not callable(c):
            c = self.resolve(c)
        props = config.pop('.', None)
        # Check for valid identifiers
        kwargs = {k: config[k] for k in config if valid_ident(k)}
        result = c(**kwargs)
        if props:
            for name, value in props.items():
                setattr(result, name, value)
        return result

    def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value

class DictConfigurator(BaseConfigurator):
    """
    Configure logging using a dictionary-like object to describe the
    configuration.
    """

    def configure(self):
        """Do the configuration."""

        config = self.config
        if 'version' not in config:
            raise ValueError("dictionary doesn't specify a version")
        if config['version'] != 1:
            raise ValueError("Unsupported version: %s" % config['version'])
        incremental = config.pop('incremental', False)
        EMPTY_DICT = {}
        logging._acquireLock()
        try:
            if incremental:
                handlers = config.get('handlers', EMPTY_DICT)
                for name in handlers:
                    if name not in logging._handlers:
                        raise ValueError('No handler found with '
                                         'name %r'  % name)
                    else:
                        try:
                            handler = logging._handlers[name]
                            handler_config = handlers[name]
                            level = handler_config.get('level', None)
                            if level:
                                handler.setLevel(logging._checkLevel(level))
                        except Exception as e:
                            raise ValueError('Unable to configure handler '
                                             '%r' % name) from e
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    try:
                        self.configure_logger(name, loggers[name], True)
                    except Exception as e:
                        raise ValueError('Unable to configure logger '
                                         '%r' % name) from e
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root, True)
                    except Exception as e:
                        raise ValueError('Unable to configure root '
                                         'logger') from e
            else:
                disable_existing = config.pop('disable_existing_loggers', True)

                _clearExistingHandlers()

                # Do formatters first - they don't refer to anything else
                formatters = config.get('formatters', EMPTY_DICT)
                for name in formatters:
                    try:
                        formatters[name] = self.configure_formatter(
                                                            formatters[name])
                    except Exception as e:
                        raise ValueError('Unable to configure '
                                         'formatter %r' % name) from e
                # Next, do filters - they don't refer to anything else, either
                filters = config.get('filters', EMPTY_DICT)
                for name in filters:
                    try:
                        filters[name] = self.configure_filter(filters[name])
                    except Exception as e:
                        raise ValueError('Unable to configure '
                                         'filter %r' % name) from e

                # Next, do handlers - they refer to formatters and filters
                # As handlers can refer to other handlers, sort the keys
                # to allow a deterministic order of configuration
                handlers = config.get('handlers', EMPTY_DICT)
                deferred = []
                for name in sorted(handlers):
                    try:
                        handler = self.configure_handler(handlers[name])
                        handler.name = name
                        handlers[name] = handler
                    except Exception as e:
                        if 'target not configured yet' in str(e.__cause__):
                            deferred.append(name)
                        else:
                            raise ValueError('Unable to configure handler '
                                             '%r' % name) from e

                # Now do any that were deferred
                for name in deferred:
                    try:
                        handler = self.configure_handler(handlers[name])
                        handler.name = name
                        handlers[name] = handler
                    except Exception as e:
                        raise ValueError('Unable to configure handler '
                                         '%r' % name) from e

                # Next, do loggers - they refer to handlers and filters

                #we don't want to lose the existing loggers,
                #since other threads may have pointers to them.
                #existing is set to contain all existing loggers,
                #and as we go through the new configuration we
                #remove any which are configured. At the end,
                #what's left in existing is the set of loggers
                #which were in the previous configuration but
                #which are not in the new configuration.
                root = logging.root
                existing = list(root.manager.loggerDict.keys())
                #The list needs to be sorted so that we can
                #avoid disabling child loggers of explicitly
                #named loggers. With a sorted list it is easier
                #to find the child loggers.
                existing.sort()
                #We'll keep the list of existing loggers
                #which are children of named loggers here...
                child_loggers = []
                #now set up the new ones...
                loggers = config.get('loggers', EMPTY_DICT)
                for name in loggers:
                    if name in existing:
                        i = existing.index(name) + 1 # look after name
                        prefixed = name + "."
                        pflen = len(prefixed)
                        num_existing = len(existing)
                        while i < num_existing:
                            if existing[i][:pflen] == prefixed:
                                child_loggers.append(existing[i])
                            i += 1
                        existing.remove(name)
                    try:
                        self.configure_logger(name, loggers[name])
                    except Exception as e:
                        raise ValueError('Unable to configure logger '
                                         '%r' % name) from e

                #Disable any old loggers. There's no point deleting
                #them as other threads may continue to hold references
                #and by disabling them, you stop them doing any logging.
                #However, don't disable children of named loggers, as that's
                #probably not what was intended by the user.
                #for log in existing:
                #    logger = root.manager.loggerDict[log]
                #    if log in child_loggers:
                #        logger.level = logging.NOTSET
                #        logger.handlers = []
                #        logger.propagate = True
                #    elif disable_existing:
                #        logger.disabled = True
                _handle_existing_loggers(existing, child_loggers,
                                         disable_existing)

                # And finally, do the root logger
                root = config.get('root', None)
                if root:
                    try:
                        self.configure_root(root)
                    except Exception as e:
                        raise ValueError('Unable to configure root '
                                         'logger') from e
        finally:
            logging._releaseLock()

    def configure_formatter(self, config):
        """Configure a formatter from a dictionary."""
        if '()' in config:
            factory = config['()'] # for use in exception handler
            try:
                result = self.configure_custom(config)
            except TypeError as te:
                if "'format'" not in str(te):
                    raise
                #Name of parameter changed from fmt to format.
                #Retry with old name.
                #This is so that code can be used with older Python versions
                #(e.g. by Django)
                config['fmt'] = config.pop('format')
                config['()'] = factory
                result = self.configure_custom(config)
        else:
            fmt = config.get('format', None)
            dfmt = config.get('datefmt', None)
            style = config.get('style', '%')
            cname = config.get('class', None)

            if not cname:
                c = logging.Formatter
            else:
                c = _resolve(cname)

            # A TypeError would be raised if "validate" key is passed in with a formatter callable
            # that does not accept "validate" as a parameter
            if 'validate' in config:  # if user hasn't mentioned it, the default will be fine
                result = c(fmt, dfmt, style, config['validate'])
            else:
                result = c(fmt, dfmt, style)

        return result

    def configure_filter(self, config):
        """Configure a filter from a dictionary."""
        if '()' in config:
            result = self.configure_custom(config)
        else:
            name = config.get('name', '')
            result = logging.Filter(name)
        return result

    def add_filters(self, filterer, filters):
        """Add filters to a filterer from a list of names."""
        for f in filters:
            try:
                filterer.addFilter(self.config['filters'][f])
            except Exception as e:
                raise ValueError('Unable to add filter %r' % f) from e

    def configure_handler(self, config):
        """Configure a handler from a dictionary."""
        config_copy = dict(config)  # for restoring in case of error
        formatter = config.pop('formatter', None)
        if formatter:
            try:
                formatter = self.config['formatters'][formatter]
            except Exception as e:
                raise ValueError('Unable to set formatter '
                                 '%r' % formatter) from e
        level = config.pop('level', None)
        filters = config.pop('filters', None)
        if '()' in config:
            c = config.pop('()')
            if not callable(c):
                c = self.resolve(c)
            factory = c
        else:
            cname = config.pop('class')
            klass = self.resolve(cname)
            #Special case for handler which refers to another handler
            if issubclass(klass, logging.handlers.MemoryHandler) and\
                'target' in config:
                try:
                    th = self.config['handlers'][config['target']]
                    if not isinstance(th, logging.Handler):
                        config.update(config_copy)  # restore for deferred cfg
                        raise TypeError('target not configured yet')
                    config['target'] = th
                except Exception as e:
                    raise ValueError('Unable to set target handler '
                                     '%r' % config['target']) from e
            elif issubclass(klass, logging.handlers.SMTPHandler) and\
                'mailhost' in config:
                config['mailhost'] = self.as_tuple(config['mailhost'])
            elif issubclass(klass, logging.handlers.SysLogHandler) and\
                'address' in config:
                config['address'] = self.as_tuple(config['address'])
            factory = klass
        props = config.pop('.', None)
        kwargs = {k: config[k] for k in config if valid_ident(k)}
        try:
            result = factory(**kwargs)
        except TypeError as te:
            if "'stream'" not in str(te):
                raise
            #The argument name changed from strm to stream
            #Retry with old name.
            #This is so that code can be used with older Python versions
            #(e.g. by Django)
            kwargs['strm'] = kwargs.pop('stream')
            result = factory(**kwargs)
        if formatter:
            result.setFormatter(formatter)
        if level is not None:
            result.setLevel(logging._checkLevel(level))
        if filters:
            self.add_filters(result, filters)
        if props:
            for name, value in props.items():
                setattr(result, name, value)
        return result

    def add_handlers(self, logger, handlers):
        """Add handlers to a logger from a list of names."""
        for h in handlers:
            try:
                logger.addHandler(self.config['handlers'][h])
            except Exception as e:
                raise ValueError('Unable to add handler %r' % h) from e

    def common_logger_config(self, logger, config, incremental=False):
        """
        Perform configuration which is common to root and non-root loggers.
        """
        level = config.get('level', None)
        if level is not None:
            logger.setLevel(logging._checkLevel(level))
        if not incremental:
            #Remove any existing handlers
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            handlers = config.get('handlers', None)
            if handlers:
                self.add_handlers(logger, handlers)
            filters = config.get('filters', None)
            if filters:
                self.add_filters(logger, filters)

    def configure_logger(self, name, config, incremental=False):
        """Configure a non-root logger from a dictionary."""
        logger = logging.getLogger(name)
        self.common_logger_config(logger, config, incremental)
        logger.disabled = False
        propagate = config.get('propagate', None)
        if propagate is not None:
            logger.propagate = propagate

    def configure_root(self, config, incremental=False):
        """Configure a root logger from a dictionary."""
        root = logging.getLogger()
        self.common_logger_config(root, config, incremental)

dictConfigClass = DictConfigurator

def dictConfig(config):
    """Configure logging using a dictionary."""
    dictConfigClass(config).configure()


def listen(port=DEFAULT_LOGGING_CONFIG_PORT, verify=None):
    """
    Start up a socket server on the specified port, and listen for new
    configurations.

    These will be sent as a file suitable for processing by fileConfig().
    Returns a Thread object on which you can call start() to start the server,
    and which you can join() when appropriate. To stop the server, call
    stopListening().

    Use the ``verify`` argument to verify any bytes received across the wire
    from a client. If specified, it should be a callable which receives a
    single argument - the bytes of configuration data received across the
    network - and it should return either ``None``, to indicate that the
    passed in bytes could not be verified and should be discarded, or a
    byte string which is then passed to the configuration machinery as
    normal. Note that you can return transformed bytes, e.g. by decrypting
    the bytes passed in.
    """

    class ConfigStreamHandler(StreamRequestHandler):
        """
        Handler for a logging configuration request.

        It expects a completely new logging configuration and uses fileConfig
        to install it.
        """
        def handle(self):
            """
            Handle a request.

            Each request is expected to be a 4-byte length, packed using
            struct.pack(">L", n), followed by the config file.
            Uses fileConfig() to do the grunt work.
            """
            try:
                conn = self.connection
                chunk = conn.recv(4)
                if len(chunk) == 4:
                    slen = struct.unpack(">L", chunk)[0]
                    chunk = self.connection.recv(slen)
                    while len(chunk) < slen:
                        chunk = chunk + conn.recv(slen - len(chunk))
                    if self.server.verify is not None:
                        chunk = self.server.verify(chunk)
                    if chunk is not None:   # verified, can process
                        chunk = chunk.decode("utf-8")
                        try:
                            import json
                            d =json.loads(chunk)
                            assert isinstance(d, dict)
                            dictConfig(d)
                        except Exception:
                            #Apply new configuration.

                            file = io.StringIO(chunk)
                            try:
                                fileConfig(file)
                            except Exception:
                                traceback.print_exc()
                    if self.server.ready:
                        self.server.ready.set()
            except OSError as e:
                if e.errno != RESET_ERROR:
                    raise

    class ConfigSocketReceiver(ThreadingTCPServer):
        """
        A simple TCP socket-based logging config receiver.
        """

        allow_reuse_address = 1

        def __init__(self, host='localhost', port=DEFAULT_LOGGING_CONFIG_PORT,
                     handler=None, ready=None, verify=None):
            ThreadingTCPServer.__init__(self, (host, port), handler)
            logging._acquireLock()
            self.abort = 0
            logging._releaseLock()
            self.timeout = 1
            self.ready = ready
            self.verify = verify

        def serve_until_stopped(self):
            import select
            abort = 0
            while not abort:
                rd, wr, ex = select.select([self.socket.fileno()],
                                           [], [],
                                           self.timeout)
                if rd:
                    self.handle_request()
                logging._acquireLock()
                abort = self.abort
                logging._releaseLock()
            self.server_close()

    class Server(threading.Thread):

        def __init__(self, rcvr, hdlr, port, verify):
            super(Server, self).__init__()
            self.rcvr = rcvr
            self.hdlr = hdlr
            self.port = port
            self.verify = verify
            self.ready = threading.Event()

        def run(self):
            server = self.rcvr(port=self.port, handler=self.hdlr,
                               ready=self.ready,
                               verify=self.verify)
            if self.port == 0:
                self.port = server.server_address[1]
            self.ready.set()
            global _listener
            logging._acquireLock()
            _listener = server
            logging._releaseLock()
            server.serve_until_stopped()

    return Server(ConfigSocketReceiver, ConfigStreamHandler, port, verify)

def stopListening():
    """
    Stop the listening server which was created with a call to listen().
    """
    global _listener
    logging._acquireLock()
    try:
        if _listener:
            _listener.abort = 1
            _listener = None
    finally:
        logging._releaseLock()
<|endoftext|># Copyright 2001-2019 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Logging package for Python. Based on PEP 282 and comments thereto in
comp.lang.python.

Copyright (C) 2001-2019 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!
"""

import sys, os, time, io, re, traceback, warnings, weakref, collections.abc

from string import Template
from string import Formatter as StrFormatter


__all__ = ['BASIC_FORMAT', 'BufferingFormatter', 'CRITICAL', 'DEBUG', 'ERROR',
           'FATAL', 'FileHandler', 'Filter', 'Formatter', 'Handler', 'INFO',
           'LogRecord', 'Logger', 'LoggerAdapter', 'NOTSET', 'NullHandler',
           'StreamHandler', 'WARN', 'WARNING', 'addLevelName', 'basicConfig',
           'captureWarnings', 'critical', 'debug', 'disable', 'error',
           'exception', 'fatal', 'getLevelName', 'getLogger', 'getLoggerClass',
           'info', 'log', 'makeLogRecord', 'setLoggerClass', 'shutdown',
           'warn', 'warning', 'getLogRecordFactory', 'setLogRecordFactory',
           'lastResort', 'raiseExceptions']

import threading

__author__  = "Vinay Sajip <vinay_sajip@red-dove.com>"
__status__  = "production"
# The following module attributes are no longer updated.
__version__ = "0.5.1.2"
__date__    = "07 February 2010"

#---------------------------------------------------------------------------
#   Miscellaneous module data
#---------------------------------------------------------------------------

#
#_startTime is used as the base when calculating the relative time of events
#
_startTime = time.time()

#
#raiseExceptions is used to see if exceptions during handling should be
#propagated
#
raiseExceptions = True

#
# If you don't want threading information in the log, set this to zero
#
logThreads = True

#
# If you don't want multiprocessing information in the log, set this to zero
#
logMultiprocessing = True

#
# If you don't want process information in the log, set this to zero
#
logProcesses = True

#---------------------------------------------------------------------------
#   Level related stuff
#---------------------------------------------------------------------------
#
# Default levels and level names, these can be replaced with any positive set
# of values having corresponding names. There is a pseudo-level, NOTSET, which
# is only really there as a lower limit for user-defined levels. Handlers and
# loggers are initialized with NOTSET so that they will log all messages, even
# at user-defined levels.
#

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_levelToName = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}
_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}

def getLevelName(level):
    """
    Return the textual or numeric representation of logging level 'level'.

    If the level is one of the predefined levels (CRITICAL, ERROR, WARNING,
    INFO, DEBUG) then you get the corresponding string. If you have
    associated levels with names using addLevelName then the name you have
    associated with 'level' is returned.

    If a numeric value corresponding to one of the defined levels is passed
    in, the corresponding string representation is returned.

    If a string representation of the level is passed in, the corresponding
    numeric value is returned.

    If no matching numeric or string value is passed in, the string
    'Level %s' % level is returned.
    """
    # See Issues #22386, #27937 and #29220 for why it's this way
    result = _levelToName.get(level)
    if result is not None:
        return result
    result = _nameToLevel.get(level)
    if result is not None:
        return result
    return "Level %s" % level

def addLevelName(level, levelName):
    """
    Associate 'levelName' with 'level'.

    This is used when converting levels to text during message formatting.
    """
    _acquireLock()
    try:    #unlikely to cause an exception, but you never know...
        _levelToName[level] = levelName
        _nameToLevel[levelName] = level
    finally:
        _releaseLock()

if hasattr(sys, '_getframe'):
    currentframe = lambda: sys._getframe(3)
else: #pragma: no cover
    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception
        except Exception:
            return sys.exc_info()[2].tb_frame.f_back

#
# _srcfile is used when walking the stack to check when we've got the first
# caller stack frame, by skipping frames whose filename is that of this
# module's source. It therefore should contain the filename of this module's
# source file.
#
# Ordinarily we would use __file__ for this, but frozen modules don't always
# have __file__ set, for some reason (see Issue #21736). Thus, we get the
# filename from a handy code object from a function defined in this module.
# (There's no particular reason for picking addLevelName.)
#

_srcfile = os.path.normcase(addLevelName.__code__.co_filename)

# _srcfile is only used in conjunction with sys._getframe().
# To provide compatibility with older versions of Python, set _srcfile
# to None if _getframe() is not available; this value will prevent
# findCaller() from being called. You can also do this if you want to avoid
# the overhead of fetching caller information, even when _getframe() is
# available.
#if not hasattr(sys, '_getframe'):
#    _srcfile = None


def _checkLevel(level):
    if isinstance(level, int):
        rv = level
    elif str(level) == level:
        if level not in _nameToLevel:
            raise ValueError("Unknown level: %r" % level)
        rv = _nameToLevel[level]
    else:
        raise TypeError("Level not an integer or a valid string: %r"
                        % (level,))
    return rv

#---------------------------------------------------------------------------
#   Thread-related stuff
#---------------------------------------------------------------------------

#
#_lock is used to serialize access to shared data structures in this module.
#This needs to be an RLock because fileConfig() creates and configures
#Handlers, and so might arbitrary user threads. Since Handler code updates the
#shared dictionary _handlers, it needs to acquire the lock. But if configuring,
#the lock would already have been acquired - so we need an RLock.
#The same argument applies to Loggers and Manager.loggerDict.
#
_lock = threading.RLock()

def _acquireLock():
    """
    Acquire the module-level lock for serializing access to shared data.

    This should be released with _releaseLock().
    """
    if _lock:
        _lock.acquire()

def _releaseLock():
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()


# Prevent a held logging lock from blocking a child from logging.

if not hasattr(os, 'register_at_fork'):  # Windows and friends.
    def _register_at_fork_reinit_lock(instance):
        pass  # no-op when os.register_at_fork does not exist.
else:
    # A collection of instances with a _at_fork_reinit method (logging.Handler)
    # to be called in the child after forking.  The weakref avoids us keeping
    # discarded Handler instances alive.
    _at_fork_reinit_lock_weakset = weakref.WeakSet()

    def _register_at_fork_reinit_lock(instance):
        _acquireLock()
        try:
            _at_fork_reinit_lock_weakset.add(instance)
        finally:
            _releaseLock()

    def _after_at_fork_child_reinit_locks():
        for handler in _at_fork_reinit_lock_weakset:
            handler._at_fork_reinit()

        # _acquireLock() was called in the parent before forking.
        # The lock is reinitialized to unlocked state.
        _lock._at_fork_reinit()

    os.register_at_fork(before=_acquireLock,
                        after_in_child=_after_at_fork_child_reinit_locks,
                        after_in_parent=_releaseLock)


#---------------------------------------------------------------------------
#   The logging record
#---------------------------------------------------------------------------

class LogRecord(object):
    """
    A LogRecord instance represents an event being logged.

    LogRecord instances are created every time something is logged. They
    contain all the information pertinent to the event being logged. The
    main information passed in is in msg and args, which are combined
    using str(msg) % args to create the message field of the record. The
    record also includes information such as when the record was created,
    the source line where the logging call was made, and any exception
    information to be logged.
    """
    def __init__(self, name, level, pathname, lineno,
                 msg, args, exc_info, func=None, sinfo=None, **kwargs):
        """
        Initialize a logging record with interesting information.
        """
        ct = time.time()
        self.name = name
        self.msg = msg
        #
        # The following statement allows passing of a dictionary as a sole
        # argument, so that you can do something like
        #  logging.debug("a %(a)d b %(b)s", {'a':1, 'b':2})
        # Suggested by Stefan Behnel.
        # Note that without the test for args[0], we get a problem because
        # during formatting, we test to see if the arg is present using
        # 'if self.args:'. If the event being logged is e.g. 'Value is %d'
        # and if the passed arg fails 'if self.args:' then no formatting
        # is done. For example, logger.warning('Value is %d', 0) would log
        # 'Value is %d' instead of 'Value is 0'.
        # For the use case of passing a dictionary, this should not be a
        # problem.
        # Issue #21172: a request was made to relax the isinstance check
        # to hasattr(args[0], '__getitem__'). However, the docs on string
        # formatting still seem to suggest a mapping object is required.
        # Thus, while not removing the isinstance check, it does now look
        # for collections.abc.Mapping rather than, as before, dict.
        if (args and len(args) == 1 and isinstance(args[0], collections.abc.Mapping)
            and args[0]):
            args = args[0]
        self.args = args
        self.levelname = getLevelName(level)
        self.levelno = level
        self.pathname = pathname
        try:
            self.filename = os.path.basename(pathname)
            self.module = os.path.splitext(self.filename)[0]
        except (TypeError, ValueError, AttributeError):
            self.filename = pathname
            self.module = "Unknown module"
        self.exc_info = exc_info
        self.exc_text = None      # used to cache the traceback text
        self.stack_info = sinfo
        self.lineno = lineno
        self.funcName = func
        self.created = ct
        self.msecs = int((ct - int(ct)) * 1000) + 0.0  # see gh-89047
        self.relativeCreated = (self.created - _startTime) * 1000
        if logThreads:
            self.thread = threading.get_ident()
            self.threadName = threading.current_thread().name
        else: # pragma: no cover
            self.thread = None
            self.threadName = None
        if not logMultiprocessing: # pragma: no cover
            self.processName = None
        else:
            self.processName = 'MainProcess'
            mp = sys.modules.get('multiprocessing')
            if mp is not None:
                # Errors may occur if multiprocessing has not finished loading
                # yet - e.g. if a custom import hook causes third-party code
                # to run when multiprocessing calls import. See issue 8200
                # for an example
                try:
                    self.processName = mp.current_process().name
                except Exception: #pragma: no cover
                    pass
        if logProcesses and hasattr(os, 'getpid'):
            self.process = os.getpid()
        else:
            self.process = None

    def __repr__(self):
        return '<LogRecord: %s, %s, %s, %s, "%s">'%(self.name, self.levelno,
            self.pathname, self.lineno, self.msg)

    def getMessage(self):
        """
        Return the message for this LogRecord.

        Return the message for this LogRecord after merging any user-supplied
        arguments with the message.
        """
        msg = str(self.msg)
        if self.args:
            msg = msg % self.args
        return msg

#
#   Determine which class to use when instantiating log records.
#
_logRecordFactory = LogRecord

def setLogRecordFactory(factory):
    """
    Set the factory to be used when instantiating a log record.

    :param factory: A callable which will be called to instantiate
    a log record.
    """
    global _logRecordFactory
    _logRecordFactory = factory

def getLogRecordFactory():
    """
    Return the factory to be used when instantiating a log record.
    """

    return _logRecordFactory

def makeLogRecord(dict):
    """
    Make a LogRecord whose attributes are defined by the specified dictionary,
    This function is useful for converting a logging event received over
    a socket connection (which is sent as a dictionary) into a LogRecord
    instance.
    """
    rv = _logRecordFactory(None, None, "", 0, "", (), None, None)
    rv.__dict__.update(dict)
    return rv


#---------------------------------------------------------------------------
#   Formatter classes and functions
#---------------------------------------------------------------------------
_str_formatter = StrFormatter()
del StrFormatter


class PercentStyle(object):

    default_format = '%(message)s'
    asctime_format = '%(asctime)s'
    asctime_search = '%(asctime)'
    validation_pattern = re.compile(r'%\(\w+\)[#0+ -]*(\*|\d+)?(\.(\*|\d+))?[diouxefgcrsa%]', re.I)

    def __init__(self, fmt, *, defaults=None):
        self._fmt = fmt or self.default_format
        self._defaults = defaults

    def usesTime(self):
        return self._fmt.find(self.asctime_search) >= 0

    def validate(self):
        """Validate the input format, ensure it matches the correct style"""
        if not self.validation_pattern.search(self._fmt):
            raise ValueError("Invalid format '%s' for '%s' style" % (self._fmt, self.default_format[0]))

    def _format(self, record):
        if defaults := self._defaults:
            values = defaults | record.__dict__
        else:
            values = record.__dict__
        return self._fmt % values

    def format(self, record):
        try:
            return self._format(record)
        except KeyError as e:
            raise ValueError('Formatting field not found in record: %s' % e)


class StrFormatStyle(PercentStyle):
    default_format = '{message}'
    asctime_format = '{asctime}'
    asctime_search = '{asctime'

    fmt_spec = re.compile(r'^(.?[<>=^])?[+ -]?#?0?(\d+|{\w+})?[,_]?(\.(\d+|{\w+}))?[bcdefgnosx%]?$', re.I)
    field_spec = re.compile(r'^(\d+|\w+)(\.\w+|\[[^]]+\])*$')

    def _format(self, record):
        if defaults := self._defaults:
            values = defaults | record.__dict__
        else:
            values = record.__dict__
        return self._fmt.format(**values)

    def validate(self):
        """Validate the input format, ensure it is the correct string formatting style"""
        fields = set()
        try:
            for _, fieldname, spec, conversion in _str_formatter.parse(self._fmt):
                if fieldname:
                    if not self.field_spec.match(fieldname):
                        raise ValueError('invalid field name/expression: %r' % fieldname)
                    fields.add(fieldname)
                if conversion and conversion not in 'rsa':
                    raise ValueError('invalid conversion: %r' % conversion)
                if spec and not self.fmt_spec.match(spec):
                    raise ValueError('bad specifier: %r' % spec)
        except ValueError as e:
            raise ValueError('invalid format: %s' % e)
        if not fields:
            raise ValueError('invalid format: no fields')


class StringTemplateStyle(PercentStyle):
    default_format = '${message}'
    asctime_format = '${asctime}'
    asctime_search = '${asctime}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tpl = Template(self._fmt)

    def usesTime(self):
        fmt = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_search) >= 0

    def validate(self):
        pattern = Template.pattern
        fields = set()
        for m in pattern.finditer(self._fmt):
            d = m.groupdict()
            if d['named']:
                fields.add(d['named'])
            elif d['braced']:
                fields.add(d['braced'])
            elif m.group(0) == '$':
                raise ValueError('invalid format: bare \'$\' not allowed')
        if not fields:
            raise ValueError('invalid format: no fields')

    def _format(self, record):
        if defaults := self._defaults:
            values = defaults | record.__dict__
        else:
            values = record.__dict__
        return self._tpl.substitute(**values)


BASIC_FORMAT = "%(levelname)s:%(name)s:%(message)s"

_STYLES = {
    '%': (PercentStyle, BASIC_FORMAT),
    '{': (StrFormatStyle, '{levelname}:{name}:{message}'),
    '$': (StringTemplateStyle, '${levelname}:${name}:${message}'),
}

class Formatter(object):
    """
    Formatter instances are used to convert a LogRecord to text.

    Formatters need to know how a LogRecord is constructed. They are
    responsible for converting a LogRecord to (usually) a string which can
    be interpreted by either a human or an external system. The base Formatter
    allows a formatting string to be specified. If none is supplied, the
    style-dependent default value, "%(message)s", "{message}", or
    "${message}", is used.

    The Formatter can be initialized with a format string which makes use of
    knowledge of the LogRecord attributes - e.g. the default value mentioned
    above makes use of the fact that the user's message and arguments are pre-
    formatted into a LogRecord's message attribute. Currently, the useful
    attributes in a LogRecord are described by:

    %(name)s            Name of the logger (logging channel)
    %(levelno)s         Numeric logging level for the message (DEBUG, INFO,
                        WARNING, ERROR, CRITICAL)
    %(levelname)s       Text logging level for the message ("DEBUG", "INFO",
                        "WARNING", "ERROR", "CRITICAL")
    %(pathname)s        Full pathname of the source file where the logging
                        call was issued (if available)
    %(filename)s        Filename portion of pathname
    %(module)s          Module (name portion of filename)
    %(lineno)d          Source line number where the logging call was issued
                        (if available)
    %(funcName)s        Function name
    %(created)f         Time when the LogRecord was created (time.time()
                        return value)
    %(asctime)s         Textual time when the LogRecord was created
    %(msecs)d           Millisecond portion of the creation time
    %(relativeCreated)d Time in milliseconds when the LogRecord was created,
                        relative to the time the logging module was loaded
                        (typically at application startup time)
    %(thread)d          Thread ID (if available)
    %(threadName)s      Thread name (if available)
    %(process)d         Process ID (if available)
    %(message)s         The result of record.getMessage(), computed just as
                        the record is emitted
    """

    converter = time.localtime

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *,
                 defaults=None):
        """
        Initialize the formatter with specified format strings.

        Initialize the formatter either with the specified format string, or a
        default as described above. Allow for specialized date formatting with
        the optional datefmt argument. If datefmt is omitted, you get an
        ISO8601-like (or RFC 3339-like) format.

        Use a style parameter of '%', '{' or '$' to specify that you want to
        use one of %-formatting, :meth:`str.format` (``{}``) formatting or
        :class:`string.Template` formatting in your format string.

        .. versionchanged:: 3.2
           Added the ``style`` parameter.
        """
        if style not in _STYLES:
            raise ValueError('Style must be one of: %s' % ','.join(
                             _STYLES.keys()))
        self._style = _STYLES[style][0](fmt, defaults=defaults)
        if validate:
            self._style.validate()

        self._fmt = self._style._fmt
        self.datefmt = datefmt

    default_time_format = '%Y-%m-%d %H:%M:%S'
    default_msec_format = '%s,%03d'

    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as formatted text.

        This method should be called from format() by a formatter which
        wants to make use of a formatted time. This method can be overridden
        in formatters to provide for any specific requirement, but the
        basic behaviour is as follows: if datefmt (a string) is specified,
        it is used with time.strftime() to format the creation time of the
        record. Otherwise, an ISO8601-like (or RFC 3339-like) format is used.
        The resulting string is returned. This function uses a user-configurable
        function to convert the creation time to a tuple. By default,
        time.localtime() is used; to change this for a particular formatter
        instance, set the 'converter' attribute to a function with the same
