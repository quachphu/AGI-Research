    '\u2264'   #  0x00f3 -> LESS-THAN OR EQUAL TO
    '\u2320'   #  0x00f4 -> TOP HALF INTEGRAL
    '\u2321'   #  0x00f5 -> BOTTOM HALF INTEGRAL
    '\xf7'     #  0x00f6 -> DIVISION SIGN
    '\u2248'   #  0x00f7 -> ALMOST EQUAL TO
    '\xb0'     #  0x00f8 -> DEGREE SIGN
    '\u2219'   #  0x00f9 -> BULLET OPERATOR
    '\xb7'     #  0x00fa -> MIDDLE DOT
    '\u221a'   #  0x00fb -> SQUARE ROOT
    '\u207f'   #  0x00fc -> SUPERSCRIPT LATIN SMALL LETTER N
    '\xb2'     #  0x00fd -> SUPERSCRIPT TWO
    '\u25a0'   #  0x00fe -> BLACK SQUARE
    '\xa0'     #  0x00ff -> NO-BREAK SPACE
)

### Encoding Map

encoding_map = {
    0x0000: 0x0000,     #  NULL
    0x0001: 0x0001,     #  START OF HEADING
    0x0002: 0x0002,     #  START OF TEXT
    0x0003: 0x0003,     #  END OF TEXT
    0x0004: 0x0004,     #  END OF TRANSMISSION
    0x0005: 0x0005,     #  ENQUIRY
    0x0006: 0x0006,     #  ACKNOWLEDGE
    0x0007: 0x0007,     #  BELL
    0x0008: 0x0008,     #  BACKSPACE
    0x0009: 0x0009,     #  HORIZONTAL TABULATION
    0x000a: 0x000a,     #  LINE FEED
    0x000b: 0x000b,     #  VERTICAL TABULATION
    0x000c: 0x000c,     #  FORM FEED
    0x000d: 0x000d,     #  CARRIAGE RETURN
    0x000e: 0x000e,     #  SHIFT OUT
    0x000f: 0x000f,     #  SHIFT IN
    0x0010: 0x0010,     #  DATA LINK ESCAPE
    0x0011: 0x0011,     #  DEVICE CONTROL ONE
    0x0012: 0x0012,     #  DEVICE CONTROL TWO
    0x0013: 0x0013,     #  DEVICE CONTROL THREE
    0x0014: 0x0014,     #  DEVICE CONTROL FOUR
    0x0015: 0x0015,     #  NEGATIVE ACKNOWLEDGE
    0x0016: 0x0016,     #  SYNCHRONOUS IDLE
    0x0017: 0x0017,     #  END OF TRANSMISSION BLOCK
    0x0018: 0x0018,     #  CANCEL
    0x0019: 0x0019,     #  END OF MEDIUM
    0x001a: 0x001a,     #  SUBSTITUTE
    0x001b: 0x001b,     #  ESCAPE
    0x001c: 0x001c,     #  FILE SEPARATOR
    0x001d: 0x001d,     #  GROUP SEPARATOR
    0x001e: 0x001e,     #  RECORD SEPARATOR
    0x001f: 0x001f,     #  UNIT SEPARATOR
    0x0020: 0x0020,     #  SPACE
    0x0021: 0x0021,     #  EXCLAMATION MARK
    0x0022: 0x0022,     #  QUOTATION MARK
    0x0023: 0x0023,     #  NUMBER SIGN
    0x0024: 0x0024,     #  DOLLAR SIGN
    0x0025: 0x0025,     #  PERCENT SIGN
    0x0026: 0x0026,     #  AMPERSAND
    0x0027: 0x0027,     #  APOSTROPHE
    0x0028: 0x0028,     #  LEFT PARENTHESIS
    0x0029: 0x0029,     #  RIGHT PARENTHESIS
    0x002a: 0x002a,     #  ASTERISK
    0x002b: 0x002b,     #  PLUS SIGN
    0x002c: 0x002c,     #  COMMA
    0x002d: 0x002d,     #  HYPHEN-MINUS
    0x002e: 0x002e,     #  FULL STOP
    0x002f: 0x002f,     #  SOLIDUS
    0x0030: 0x0030,     #  DIGIT ZERO
    0x0031: 0x0031,     #  DIGIT ONE
    0x0032: 0x0032,     #  DIGIT TWO
    0x0033: 0x0033,     #  DIGIT THREE
    0x0034: 0x0034,     #  DIGIT FOUR
    0x0035: 0x0035,     #  DIGIT FIVE
    0x0036: 0x0036,     #  DIGIT SIX
    0x0037: 0x0037,     #  DIGIT SEVEN
    0x0038: 0x0038,     #  DIGIT EIGHT
    0x0039: 0x0039,     #  DIGIT NINE
    0x003a: 0x003a,     #  COLON
    0x003b: 0x003b,     #  SEMICOLON
    0x003c: 0x003c,     #  LESS-THAN SIGN
    0x003d: 0x003d,     #  EQUALS SIGN
    0x003e: 0x003e,     #  GREATER-THAN SIGN
    0x003f: 0x003f,     #  QUESTION MARK
    0x0040: 0x0040,     #  COMMERCIAL AT
    0x0041: 0x0041,     #  LATIN CAPITAL LETTER A
    0x0042: 0x0042,     #  LATIN CAPITAL LETTER B
    0x0043: 0x0043,     #  LATIN CAPITAL LETTER C
    0x0044: 0x0044,     #  LATIN CAPITAL LETTER D
    0x0045: 0x0045,     #  LATIN CAPITAL LETTER E
    0x0046: 0x0046,     #  LATIN CAPITAL LETTER F
    0x0047: 0x0047,     #  LATIN CAPITAL LETTER G
    0x0048: 0x0048,     #  LATIN CAPITAL LETTER H
    0x0049: 0x0049,     #  LATIN CAPITAL LETTER I
    0x004a: 0x004a,     #  LATIN CAPITAL LETTER J
    0x004b: 0x004b,     #  LATIN CAPITAL LETTER K
    0x004c: 0x004c,     #  LATIN CAPITAL LETTER L
    0x004d: 0x004d,     #  LATIN CAPITAL LETTER M
    0x004e: 0x004e,     #  LATIN CAPITAL LETTER N
    0x004f: 0x004f,     #  LATIN CAPITAL LETTER O
    0x0050: 0x0050,     #  LATIN CAPITAL LETTER P
    0x0051: 0x0051,     #  LATIN CAPITAL LETTER Q
    0x0052: 0x0052,     #  LATIN CAPITAL LETTER R
    0x0053: 0x0053,     #  LATIN CAPITAL LETTER S
    0x0054: 0x0054,     #  LATIN CAPITAL LETTER T
    0x0055: 0x0055,     #  LATIN CAPITAL LETTER U
    0x0056: 0x0056,     #  LATIN CAPITAL LETTER V
    0x0057: 0x0057,     #  LATIN CAPITAL LETTER W
    0x0058: 0x0058,     #  LATIN CAPITAL LETTER X
    0x0059: 0x0059,     #  LATIN CAPITAL LETTER Y
    0x005a: 0x005a,     #  LATIN CAPITAL LETTER Z
    0x005b: 0x005b,     #  LEFT SQUARE BRACKET
    0x005c: 0x005c,     #  REVERSE SOLIDUS
    0x005d: 0x005d,     #  RIGHT SQUARE BRACKET
    0x005e: 0x005e,     #  CIRCUMFLEX ACCENT
    0x005f: 0x005f,     #  LOW LINE
    0x0060: 0x0060,     #  GRAVE ACCENT
    0x0061: 0x0061,     #  LATIN SMALL LETTER A
    0x0062: 0x0062,     #  LATIN SMALL LETTER B
    0x0063: 0x0063,     #  LATIN SMALL LETTER C
    0x0064: 0x0064,     #  LATIN SMALL LETTER D
    0x0065: 0x0065,     #  LATIN SMALL LETTER E
    0x0066: 0x0066,     #  LATIN SMALL LETTER F
    0x0067: 0x0067,     #  LATIN SMALL LETTER G
    0x0068: 0x0068,     #  LATIN SMALL LETTER H
    0x0069: 0x0069,     #  LATIN SMALL LETTER I
    0x006a: 0x006a,     #  LATIN SMALL LETTER J
    0x006b: 0x006b,     #  LATIN SMALL LETTER K
    0x006c: 0x006c,     #  LATIN SMALL LETTER L
    0x006d: 0x006d,     #  LATIN SMALL LETTER M
    0x006e: 0x006e,     #  LATIN SMALL LETTER N
    0x006f: 0x006f,     #  LATIN SMALL LETTER O
    0x0070: 0x0070,     #  LATIN SMALL LETTER P
    0x0071: 0x0071,     #  LATIN SMALL LETTER Q
    0x0072: 0x0072,     #  LATIN SMALL LETTER R
    0x0073: 0x0073,     #  LATIN SMALL LETTER S
    0x0074: 0x0074,     #  LATIN SMALL LETTER T
    0x0075: 0x0075,     #  LATIN SMALL LETTER U
    0x0076: 0x0076,     #  LATIN SMALL LETTER V
    0x0077: 0x0077,     #  LATIN SMALL LETTER W
    0x0078: 0x0078,     #  LATIN SMALL LETTER X
    0x0079: 0x0079,     #  LATIN SMALL LETTER Y
    0x007a: 0x007a,     #  LATIN SMALL LETTER Z
    0x007b: 0x007b,     #  LEFT CURLY BRACKET
    0x007c: 0x007c,     #  VERTICAL LINE
    0x007d: 0x007d,     #  RIGHT CURLY BRACKET
    0x007e: 0x007e,     #  TILDE
    0x007f: 0x007f,     #  DELETE
    0x00a0: 0x00ff,     #  NO-BREAK SPACE
    0x00a1: 0x00ad,     #  INVERTED EXCLAMATION MARK
    0x00a2: 0x009b,     #  CENT SIGN
    0x00a3: 0x009c,     #  POUND SIGN
    0x00a5: 0x009d,     #  YEN SIGN
    0x00aa: 0x00a6,     #  FEMININE ORDINAL INDICATOR
    0x00ab: 0x00ae,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00ac: 0x00aa,     #  NOT SIGN
    0x00b0: 0x00f8,     #  DEGREE SIGN
    0x00b1: 0x00f1,     #  PLUS-MINUS SIGN
    0x00b2: 0x00fd,     #  SUPERSCRIPT TWO
    0x00b5: 0x00e6,     #  MICRO SIGN
    0x00b7: 0x00fa,     #  MIDDLE DOT
    0x00ba: 0x00a7,     #  MASCULINE ORDINAL INDICATOR
    0x00bb: 0x00af,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00bc: 0x00ac,     #  VULGAR FRACTION ONE QUARTER
    0x00bd: 0x00ab,     #  VULGAR FRACTION ONE HALF
    0x00bf: 0x00a8,     #  INVERTED QUESTION MARK
    0x00c4: 0x008e,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x00c5: 0x008f,     #  LATIN CAPITAL LETTER A WITH RING ABOVE
    0x00c6: 0x0092,     #  LATIN CAPITAL LIGATURE AE
    0x00c7: 0x0080,     #  LATIN CAPITAL LETTER C WITH CEDILLA
    0x00c9: 0x0090,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x00d1: 0x00a5,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x00d6: 0x0099,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x00dc: 0x009a,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x00df: 0x00e1,     #  LATIN SMALL LETTER SHARP S
    0x00e0: 0x0085,     #  LATIN SMALL LETTER A WITH GRAVE
    0x00e1: 0x00a0,     #  LATIN SMALL LETTER A WITH ACUTE
    0x00e2: 0x0083,     #  LATIN SMALL LETTER A WITH CIRCUMFLEX
    0x00e4: 0x0084,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x00e5: 0x0086,     #  LATIN SMALL LETTER A WITH RING ABOVE
    0x00e6: 0x0091,     #  LATIN SMALL LIGATURE AE
    0x00e7: 0x0087,     #  LATIN SMALL LETTER C WITH CEDILLA
    0x00e8: 0x008a,     #  LATIN SMALL LETTER E WITH GRAVE
    0x00e9: 0x0082,     #  LATIN SMALL LETTER E WITH ACUTE
    0x00ea: 0x0088,     #  LATIN SMALL LETTER E WITH CIRCUMFLEX
    0x00eb: 0x0089,     #  LATIN SMALL LETTER E WITH DIAERESIS
    0x00ec: 0x008d,     #  LATIN SMALL LETTER I WITH GRAVE
    0x00ed: 0x00a1,     #  LATIN SMALL LETTER I WITH ACUTE
    0x00ee: 0x008c,     #  LATIN SMALL LETTER I WITH CIRCUMFLEX
    0x00ef: 0x008b,     #  LATIN SMALL LETTER I WITH DIAERESIS
    0x00f1: 0x00a4,     #  LATIN SMALL LETTER N WITH TILDE
    0x00f2: 0x0095,     #  LATIN SMALL LETTER O WITH GRAVE
    0x00f3: 0x00a2,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00f4: 0x0093,     #  LATIN SMALL LETTER O WITH CIRCUMFLEX
    0x00f6: 0x0094,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x00f7: 0x00f6,     #  DIVISION SIGN
    0x00f9: 0x0097,     #  LATIN SMALL LETTER U WITH GRAVE
    0x00fa: 0x00a3,     #  LATIN SMALL LETTER U WITH ACUTE
    0x00fb: 0x0096,     #  LATIN SMALL LETTER U WITH CIRCUMFLEX
    0x00fc: 0x0081,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x00ff: 0x0098,     #  LATIN SMALL LETTER Y WITH DIAERESIS
    0x0192: 0x009f,     #  LATIN SMALL LETTER F WITH HOOK
    0x0393: 0x00e2,     #  GREEK CAPITAL LETTER GAMMA
    0x0398: 0x00e9,     #  GREEK CAPITAL LETTER THETA
    0x03a3: 0x00e4,     #  GREEK CAPITAL LETTER SIGMA
    0x03a6: 0x00e8,     #  GREEK CAPITAL LETTER PHI
    0x03a9: 0x00ea,     #  GREEK CAPITAL LETTER OMEGA
    0x03b1: 0x00e0,     #  GREEK SMALL LETTER ALPHA
    0x03b4: 0x00eb,     #  GREEK SMALL LETTER DELTA
    0x03b5: 0x00ee,     #  GREEK SMALL LETTER EPSILON
    0x03c0: 0x00e3,     #  GREEK SMALL LETTER PI
    0x03c3: 0x00e5,     #  GREEK SMALL LETTER SIGMA
    0x03c4: 0x00e7,     #  GREEK SMALL LETTER TAU
    0x03c6: 0x00ed,     #  GREEK SMALL LETTER PHI
    0x207f: 0x00fc,     #  SUPERSCRIPT LATIN SMALL LETTER N
    0x20a7: 0x009e,     #  PESETA SIGN
    0x2219: 0x00f9,     #  BULLET OPERATOR
    0x221a: 0x00fb,     #  SQUARE ROOT
    0x221e: 0x00ec,     #  INFINITY
    0x2229: 0x00ef,     #  INTERSECTION
    0x2248: 0x00f7,     #  ALMOST EQUAL TO
    0x2261: 0x00f0,     #  IDENTICAL TO
    0x2264: 0x00f3,     #  LESS-THAN OR EQUAL TO
    0x2265: 0x00f2,     #  GREATER-THAN OR EQUAL TO
    0x2310: 0x00a9,     #  REVERSED NOT SIGN
    0x2320: 0x00f4,     #  TOP HALF INTEGRAL
    0x2321: 0x00f5,     #  BOTTOM HALF INTEGRAL
    0x2500: 0x00c4,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x2502: 0x00b3,     #  BOX DRAWINGS LIGHT VERTICAL
    0x250c: 0x00da,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x2510: 0x00bf,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x2514: 0x00c0,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x2518: 0x00d9,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x251c: 0x00c3,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x2524: 0x00b4,     #  BOX DRAWINGS LIGHT VERTICAL AND LEFT
    0x252c: 0x00c2,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x2534: 0x00c1,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x253c: 0x00c5,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x2550: 0x00cd,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x2551: 0x00ba,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x2552: 0x00d5,     #  BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    0x2553: 0x00d6,     #  BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    0x2554: 0x00c9,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x2555: 0x00b8,     #  BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    0x2556: 0x00b7,     #  BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    0x2557: 0x00bb,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x2558: 0x00d4,     #  BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    0x2559: 0x00d3,     #  BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    0x255a: 0x00c8,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x255b: 0x00be,     #  BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    0x255c: 0x00bd,     #  BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    0x255d: 0x00bc,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x255e: 0x00c6,     #  BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    0x255f: 0x00c7,     #  BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    0x2560: 0x00cc,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x2561: 0x00b5,     #  BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    0x2562: 0x00b6,     #  BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    0x2563: 0x00b9,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x2564: 0x00d1,     #  BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    0x2565: 0x00d2,     #  BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    0x2566: 0x00cb,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x2567: 0x00cf,     #  BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    0x2568: 0x00d0,     #  BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    0x2569: 0x00ca,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x256a: 0x00d8,     #  BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    0x256b: 0x00d7,     #  BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    0x256c: 0x00ce,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x2580: 0x00df,     #  UPPER HALF BLOCK
    0x2584: 0x00dc,     #  LOWER HALF BLOCK
    0x2588: 0x00db,     #  FULL BLOCK
    0x258c: 0x00dd,     #  LEFT HALF BLOCK
    0x2590: 0x00de,     #  RIGHT HALF BLOCK
    0x2591: 0x00b0,     #  LIGHT SHADE
    0x2592: 0x00b1,     #  MEDIUM SHADE
    0x2593: 0x00b2,     #  DARK SHADE
    0x25a0: 0x00fe,     #  BLACK SQUARE
}
<|endoftext|>#
# euc_jis_2004.py: Python Unicode Codec for EUC_JIS_2004
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_jp, codecs
import _multibytecodec as mbc

codec = _codecs_jp.getcodec('euc_jis_2004')

class Codec(codecs.Codec):
    encode = codec.encode
    decode = codec.decode

class IncrementalEncoder(mbc.MultibyteIncrementalEncoder,
                         codecs.IncrementalEncoder):
    codec = codec

class IncrementalDecoder(mbc.MultibyteIncrementalDecoder,
                         codecs.IncrementalDecoder):
    codec = codec

class StreamReader(Codec, mbc.MultibyteStreamReader, codecs.StreamReader):
    codec = codec

class StreamWriter(Codec, mbc.MultibyteStreamWriter, codecs.StreamWriter):
    codec = codec

def getregentry():
    return codecs.CodecInfo(
        name='euc_jis_2004',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec mac_farsi generated from 'MAPPINGS/VENDORS/APPLE/FARSI.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='mac-farsi',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> CONTROL CHARACTER
    '\x01'     #  0x01 -> CONTROL CHARACTER
    '\x02'     #  0x02 -> CONTROL CHARACTER
    '\x03'     #  0x03 -> CONTROL CHARACTER
    '\x04'     #  0x04 -> CONTROL CHARACTER
    '\x05'     #  0x05 -> CONTROL CHARACTER
    '\x06'     #  0x06 -> CONTROL CHARACTER
    '\x07'     #  0x07 -> CONTROL CHARACTER
    '\x08'     #  0x08 -> CONTROL CHARACTER
    '\t'       #  0x09 -> CONTROL CHARACTER
    '\n'       #  0x0A -> CONTROL CHARACTER
    '\x0b'     #  0x0B -> CONTROL CHARACTER
    '\x0c'     #  0x0C -> CONTROL CHARACTER
    '\r'       #  0x0D -> CONTROL CHARACTER
    '\x0e'     #  0x0E -> CONTROL CHARACTER
    '\x0f'     #  0x0F -> CONTROL CHARACTER
    '\x10'     #  0x10 -> CONTROL CHARACTER
    '\x11'     #  0x11 -> CONTROL CHARACTER
    '\x12'     #  0x12 -> CONTROL CHARACTER
    '\x13'     #  0x13 -> CONTROL CHARACTER
    '\x14'     #  0x14 -> CONTROL CHARACTER
    '\x15'     #  0x15 -> CONTROL CHARACTER
    '\x16'     #  0x16 -> CONTROL CHARACTER
    '\x17'     #  0x17 -> CONTROL CHARACTER
    '\x18'     #  0x18 -> CONTROL CHARACTER
    '\x19'     #  0x19 -> CONTROL CHARACTER
    '\x1a'     #  0x1A -> CONTROL CHARACTER
    '\x1b'     #  0x1B -> CONTROL CHARACTER
    '\x1c'     #  0x1C -> CONTROL CHARACTER
    '\x1d'     #  0x1D -> CONTROL CHARACTER
    '\x1e'     #  0x1E -> CONTROL CHARACTER
    '\x1f'     #  0x1F -> CONTROL CHARACTER
    ' '        #  0x20 -> SPACE, left-right
    '!'        #  0x21 -> EXCLAMATION MARK, left-right
    '"'        #  0x22 -> QUOTATION MARK, left-right
    '#'        #  0x23 -> NUMBER SIGN, left-right
    '$'        #  0x24 -> DOLLAR SIGN, left-right
    '%'        #  0x25 -> PERCENT SIGN, left-right
    '&'        #  0x26 -> AMPERSAND, left-right
    "'"        #  0x27 -> APOSTROPHE, left-right
    '('        #  0x28 -> LEFT PARENTHESIS, left-right
    ')'        #  0x29 -> RIGHT PARENTHESIS, left-right
    '*'        #  0x2A -> ASTERISK, left-right
    '+'        #  0x2B -> PLUS SIGN, left-right
    ','        #  0x2C -> COMMA, left-right; in Arabic-script context, displayed as 0x066C ARABIC THOUSANDS SEPARATOR
    '-'        #  0x2D -> HYPHEN-MINUS, left-right
    '.'        #  0x2E -> FULL STOP, left-right; in Arabic-script context, displayed as 0x066B ARABIC DECIMAL SEPARATOR
    '/'        #  0x2F -> SOLIDUS, left-right
    '0'        #  0x30 -> DIGIT ZERO;  in Arabic-script context, displayed as 0x06F0 EXTENDED ARABIC-INDIC DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE;   in Arabic-script context, displayed as 0x06F1 EXTENDED ARABIC-INDIC DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO;   in Arabic-script context, displayed as 0x06F2 EXTENDED ARABIC-INDIC DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE; in Arabic-script context, displayed as 0x06F3 EXTENDED ARABIC-INDIC DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR;  in Arabic-script context, displayed as 0x06F4 EXTENDED ARABIC-INDIC DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE;  in Arabic-script context, displayed as 0x06F5 EXTENDED ARABIC-INDIC DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX;   in Arabic-script context, displayed as 0x06F6 EXTENDED ARABIC-INDIC DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN; in Arabic-script context, displayed as 0x06F7 EXTENDED ARABIC-INDIC DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT; in Arabic-script context, displayed as 0x06F8 EXTENDED ARABIC-INDIC DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE;  in Arabic-script context, displayed as 0x06F9 EXTENDED ARABIC-INDIC DIGIT NINE
    ':'        #  0x3A -> COLON, left-right
    ';'        #  0x3B -> SEMICOLON, left-right
    '<'        #  0x3C -> LESS-THAN SIGN, left-right
    '='        #  0x3D -> EQUALS SIGN, left-right
    '>'        #  0x3E -> GREATER-THAN SIGN, left-right
    '?'        #  0x3F -> QUESTION MARK, left-right
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET, left-right
    '\\'       #  0x5C -> REVERSE SOLIDUS, left-right
    ']'        #  0x5D -> RIGHT SQUARE BRACKET, left-right
    '^'        #  0x5E -> CIRCUMFLEX ACCENT, left-right
    '_'        #  0x5F -> LOW LINE, left-right
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET, left-right
    '|'        #  0x7C -> VERTICAL LINE, left-right
    '}'        #  0x7D -> RIGHT CURLY BRACKET, left-right
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> CONTROL CHARACTER
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xa0'     #  0x81 -> NO-BREAK SPACE, right-left
    '\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u06ba'   #  0x8B -> ARABIC LETTER NOON GHUNNA
    '\xab'     #  0x8C -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\u2026'   #  0x93 -> HORIZONTAL ELLIPSIS, right-left
    '\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\xbb'     #  0x98 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0x9B -> DIVISION SIGN, right-left
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    ' '        #  0xA0 -> SPACE, right-left
    '!'        #  0xA1 -> EXCLAMATION MARK, right-left
    '"'        #  0xA2 -> QUOTATION MARK, right-left
    '#'        #  0xA3 -> NUMBER SIGN, right-left
    '$'        #  0xA4 -> DOLLAR SIGN, right-left
    '\u066a'   #  0xA5 -> ARABIC PERCENT SIGN
    '&'        #  0xA6 -> AMPERSAND, right-left
    "'"        #  0xA7 -> APOSTROPHE, right-left
    '('        #  0xA8 -> LEFT PARENTHESIS, right-left
    ')'        #  0xA9 -> RIGHT PARENTHESIS, right-left
    '*'        #  0xAA -> ASTERISK, right-left
    '+'        #  0xAB -> PLUS SIGN, right-left
    '\u060c'   #  0xAC -> ARABIC COMMA
    '-'        #  0xAD -> HYPHEN-MINUS, right-left
    '.'        #  0xAE -> FULL STOP, right-left
    '/'        #  0xAF -> SOLIDUS, right-left
    '\u06f0'   #  0xB0 -> EXTENDED ARABIC-INDIC DIGIT ZERO, right-left (need override)
    '\u06f1'   #  0xB1 -> EXTENDED ARABIC-INDIC DIGIT ONE, right-left (need override)
    '\u06f2'   #  0xB2 -> EXTENDED ARABIC-INDIC DIGIT TWO, right-left (need override)
    '\u06f3'   #  0xB3 -> EXTENDED ARABIC-INDIC DIGIT THREE, right-left (need override)
    '\u06f4'   #  0xB4 -> EXTENDED ARABIC-INDIC DIGIT FOUR, right-left (need override)
    '\u06f5'   #  0xB5 -> EXTENDED ARABIC-INDIC DIGIT FIVE, right-left (need override)
    '\u06f6'   #  0xB6 -> EXTENDED ARABIC-INDIC DIGIT SIX, right-left (need override)
    '\u06f7'   #  0xB7 -> EXTENDED ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    '\u06f8'   #  0xB8 -> EXTENDED ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    '\u06f9'   #  0xB9 -> EXTENDED ARABIC-INDIC DIGIT NINE, right-left (need override)
    ':'        #  0xBA -> COLON, right-left
    '\u061b'   #  0xBB -> ARABIC SEMICOLON
    '<'        #  0xBC -> LESS-THAN SIGN, right-left
    '='        #  0xBD -> EQUALS SIGN, right-left
    '>'        #  0xBE -> GREATER-THAN SIGN, right-left
    '\u061f'   #  0xBF -> ARABIC QUESTION MARK
    '\u274a'   #  0xC0 -> EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
    '\u0621'   #  0xC1 -> ARABIC LETTER HAMZA
    '\u0622'   #  0xC2 -> ARABIC LETTER ALEF WITH MADDA ABOVE
    '\u0623'   #  0xC3 -> ARABIC LETTER ALEF WITH HAMZA ABOVE
    '\u0624'   #  0xC4 -> ARABIC LETTER WAW WITH HAMZA ABOVE
    '\u0625'   #  0xC5 -> ARABIC LETTER ALEF WITH HAMZA BELOW
    '\u0626'   #  0xC6 -> ARABIC LETTER YEH WITH HAMZA ABOVE
    '\u0627'   #  0xC7 -> ARABIC LETTER ALEF
    '\u0628'   #  0xC8 -> ARABIC LETTER BEH
    '\u0629'   #  0xC9 -> ARABIC LETTER TEH MARBUTA
    '\u062a'   #  0xCA -> ARABIC LETTER TEH
    '\u062b'   #  0xCB -> ARABIC LETTER THEH
    '\u062c'   #  0xCC -> ARABIC LETTER JEEM
    '\u062d'   #  0xCD -> ARABIC LETTER HAH
    '\u062e'   #  0xCE -> ARABIC LETTER KHAH
    '\u062f'   #  0xCF -> ARABIC LETTER DAL
    '\u0630'   #  0xD0 -> ARABIC LETTER THAL
    '\u0631'   #  0xD1 -> ARABIC LETTER REH
    '\u0632'   #  0xD2 -> ARABIC LETTER ZAIN
    '\u0633'   #  0xD3 -> ARABIC LETTER SEEN
    '\u0634'   #  0xD4 -> ARABIC LETTER SHEEN
    '\u0635'   #  0xD5 -> ARABIC LETTER SAD
    '\u0636'   #  0xD6 -> ARABIC LETTER DAD
    '\u0637'   #  0xD7 -> ARABIC LETTER TAH
    '\u0638'   #  0xD8 -> ARABIC LETTER ZAH
    '\u0639'   #  0xD9 -> ARABIC LETTER AIN
    '\u063a'   #  0xDA -> ARABIC LETTER GHAIN
    '['        #  0xDB -> LEFT SQUARE BRACKET, right-left
    '\\'       #  0xDC -> REVERSE SOLIDUS, right-left
    ']'        #  0xDD -> RIGHT SQUARE BRACKET, right-left
    '^'        #  0xDE -> CIRCUMFLEX ACCENT, right-left
    '_'        #  0xDF -> LOW LINE, right-left
    '\u0640'   #  0xE0 -> ARABIC TATWEEL
    '\u0641'   #  0xE1 -> ARABIC LETTER FEH
    '\u0642'   #  0xE2 -> ARABIC LETTER QAF
    '\u0643'   #  0xE3 -> ARABIC LETTER KAF
    '\u0644'   #  0xE4 -> ARABIC LETTER LAM
    '\u0645'   #  0xE5 -> ARABIC LETTER MEEM
    '\u0646'   #  0xE6 -> ARABIC LETTER NOON
    '\u0647'   #  0xE7 -> ARABIC LETTER HEH
    '\u0648'   #  0xE8 -> ARABIC LETTER WAW
    '\u0649'   #  0xE9 -> ARABIC LETTER ALEF MAKSURA
    '\u064a'   #  0xEA -> ARABIC LETTER YEH
    '\u064b'   #  0xEB -> ARABIC FATHATAN
    '\u064c'   #  0xEC -> ARABIC DAMMATAN
    '\u064d'   #  0xED -> ARABIC KASRATAN
    '\u064e'   #  0xEE -> ARABIC FATHA
    '\u064f'   #  0xEF -> ARABIC DAMMA
    '\u0650'   #  0xF0 -> ARABIC KASRA
    '\u0651'   #  0xF1 -> ARABIC SHADDA
    '\u0652'   #  0xF2 -> ARABIC SUKUN
    '\u067e'   #  0xF3 -> ARABIC LETTER PEH
    '\u0679'   #  0xF4 -> ARABIC LETTER TTEH
    '\u0686'   #  0xF5 -> ARABIC LETTER TCHEH
    '\u06d5'   #  0xF6 -> ARABIC LETTER AE
    '\u06a4'   #  0xF7 -> ARABIC LETTER VEH
    '\u06af'   #  0xF8 -> ARABIC LETTER GAF
    '\u0688'   #  0xF9 -> ARABIC LETTER DDAL
    '\u0691'   #  0xFA -> ARABIC LETTER RREH
    '{'        #  0xFB -> LEFT CURLY BRACKET, right-left
    '|'        #  0xFC -> VERTICAL LINE, right-left
    '}'        #  0xFD -> RIGHT CURLY BRACKET, right-left
    '\u0698'   #  0xFE -> ARABIC LETTER JEH
    '\u06d2'   #  0xFF -> ARABIC LETTER YEH BARREE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>"""
Python 'utf-32-be' Codec
"""
import codecs

### Codec APIs

encode = codecs.utf_32_be_encode

def decode(input, errors='strict'):
    return codecs.utf_32_be_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.utf_32_be_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = codecs.utf_32_be_decode

class StreamWriter(codecs.StreamWriter):
    encode = codecs.utf_32_be_encode

class StreamReader(codecs.StreamReader):
    decode = codecs.utf_32_be_decode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='utf-32-be',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec mac_iceland generated from 'MAPPINGS/VENDORS/APPLE/ICELAND.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='mac-iceland',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> CONTROL CHARACTER
    '\x01'     #  0x01 -> CONTROL CHARACTER
    '\x02'     #  0x02 -> CONTROL CHARACTER
    '\x03'     #  0x03 -> CONTROL CHARACTER
    '\x04'     #  0x04 -> CONTROL CHARACTER
    '\x05'     #  0x05 -> CONTROL CHARACTER
    '\x06'     #  0x06 -> CONTROL CHARACTER
    '\x07'     #  0x07 -> CONTROL CHARACTER
    '\x08'     #  0x08 -> CONTROL CHARACTER
    '\t'       #  0x09 -> CONTROL CHARACTER
    '\n'       #  0x0A -> CONTROL CHARACTER
    '\x0b'     #  0x0B -> CONTROL CHARACTER
    '\x0c'     #  0x0C -> CONTROL CHARACTER
    '\r'       #  0x0D -> CONTROL CHARACTER
    '\x0e'     #  0x0E -> CONTROL CHARACTER
    '\x0f'     #  0x0F -> CONTROL CHARACTER
    '\x10'     #  0x10 -> CONTROL CHARACTER
    '\x11'     #  0x11 -> CONTROL CHARACTER
    '\x12'     #  0x12 -> CONTROL CHARACTER
    '\x13'     #  0x13 -> CONTROL CHARACTER
    '\x14'     #  0x14 -> CONTROL CHARACTER
    '\x15'     #  0x15 -> CONTROL CHARACTER
    '\x16'     #  0x16 -> CONTROL CHARACTER
    '\x17'     #  0x17 -> CONTROL CHARACTER
    '\x18'     #  0x18 -> CONTROL CHARACTER
    '\x19'     #  0x19 -> CONTROL CHARACTER
    '\x1a'     #  0x1A -> CONTROL CHARACTER
    '\x1b'     #  0x1B -> CONTROL CHARACTER
    '\x1c'     #  0x1C -> CONTROL CHARACTER
    '\x1d'     #  0x1D -> CONTROL CHARACTER
    '\x1e'     #  0x1E -> CONTROL CHARACTER
    '\x1f'     #  0x1F -> CONTROL CHARACTER
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> CONTROL CHARACTER
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0x81 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe3'     #  0x8B -> LATIN SMALL LETTER A WITH TILDE
    '\xe5'     #  0x8C -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\xec'     #  0x93 -> LATIN SMALL LETTER I WITH GRAVE
    '\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf2'     #  0x98 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf5'     #  0x9B -> LATIN SMALL LETTER O WITH TILDE
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xdd'     #  0xA0 -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u2122'   #  0xAA -> TRADE MARK SIGN
    '\xb4'     #  0xAB -> ACUTE ACCENT
    '\xa8'     #  0xAC -> DIAERESIS
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\xc6'     #  0xAE -> LATIN CAPITAL LETTER AE
    '\xd8'     #  0xAF -> LATIN CAPITAL LETTER O WITH STROKE
    '\u221e'   #  0xB0 -> INFINITY
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\xa5'     #  0xB4 -> YEN SIGN
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\u2202'   #  0xB6 -> PARTIAL DIFFERENTIAL
    '\u2211'   #  0xB7 -> N-ARY SUMMATION
    '\u220f'   #  0xB8 -> N-ARY PRODUCT
    '\u03c0'   #  0xB9 -> GREEK SMALL LETTER PI
    '\u222b'   #  0xBA -> INTEGRAL
    '\xaa'     #  0xBB -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0xBC -> MASCULINE ORDINAL INDICATOR
    '\u03a9'   #  0xBD -> GREEK CAPITAL LETTER OMEGA
    '\xe6'     #  0xBE -> LATIN SMALL LETTER AE
    '\xf8'     #  0xBF -> LATIN SMALL LETTER O WITH STROKE
    '\xbf'     #  0xC0 -> INVERTED QUESTION MARK
    '\xa1'     #  0xC1 -> INVERTED EXCLAMATION MARK
    '\xac'     #  0xC2 -> NOT SIGN
    '\u221a'   #  0xC3 -> SQUARE ROOT
    '\u0192'   #  0xC4 -> LATIN SMALL LETTER F WITH HOOK
    '\u2248'   #  0xC5 -> ALMOST EQUAL TO
    '\u2206'   #  0xC6 -> INCREMENT
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xC8 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\xc0'     #  0xCB -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc3'     #  0xCC -> LATIN CAPITAL LETTER A WITH TILDE
    '\xd5'     #  0xCD -> LATIN CAPITAL LETTER O WITH TILDE
    '\u0152'   #  0xCE -> LATIN CAPITAL LIGATURE OE
    '\u0153'   #  0xCF -> LATIN SMALL LIGATURE OE
    '\u2013'   #  0xD0 -> EN DASH
    '\u2014'   #  0xD1 -> EM DASH
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u25ca'   #  0xD7 -> LOZENGE
    '\xff'     #  0xD8 -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\u0178'   #  0xD9 -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\u2044'   #  0xDA -> FRACTION SLASH
    '\u20ac'   #  0xDB -> EURO SIGN
    '\xd0'     #  0xDC -> LATIN CAPITAL LETTER ETH
    '\xf0'     #  0xDD -> LATIN SMALL LETTER ETH
    '\xde'     #  0xDE -> LATIN CAPITAL LETTER THORN
    '\xfe'     #  0xDF -> LATIN SMALL LETTER THORN
    '\xfd'     #  0xE0 -> LATIN SMALL LETTER Y WITH ACUTE
    '\xb7'     #  0xE1 -> MIDDLE DOT
    '\u201a'   #  0xE2 -> SINGLE LOW-9 QUOTATION MARK
    '\u201e'   #  0xE3 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2030'   #  0xE4 -> PER MILLE SIGN
    '\xc2'     #  0xE5 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xca'     #  0xE6 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xc1'     #  0xE7 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xcb'     #  0xE8 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xc8'     #  0xE9 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xcd'     #  0xEA -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xEB -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xEC -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xcc'     #  0xED -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xEF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\uf8ff'   #  0xF0 -> Apple logo
    '\xd2'     #  0xF1 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xda'     #  0xF2 -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xF3 -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xd9'     #  0xF4 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\u0131'   #  0xF5 -> LATIN SMALL LETTER DOTLESS I
    '\u02c6'   #  0xF6 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u02dc'   #  0xF7 -> SMALL TILDE
    '\xaf'     #  0xF8 -> MACRON
    '\u02d8'   #  0xF9 -> BREVE
    '\u02d9'   #  0xFA -> DOT ABOVE
    '\u02da'   #  0xFB -> RING ABOVE
    '\xb8'     #  0xFC -> CEDILLA
    '\u02dd'   #  0xFD -> DOUBLE ACUTE ACCENT
    '\u02db'   #  0xFE -> OGONEK
    '\u02c7'   #  0xFF -> CARON
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python 'raw-unicode-escape' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs

### Codec APIs

class Codec(codecs.Codec):

    # Note: Binding these as C functions will result in the class not
    # converting them to methods. This is intended.
    encode = codecs.raw_unicode_escape_encode
    decode = codecs.raw_unicode_escape_decode

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.raw_unicode_escape_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input, errors, final):
        return codecs.raw_unicode_escape_decode(input, errors, final)

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    def decode(self, input, errors='strict'):
        return codecs.raw_unicode_escape_decode(input, errors, False)

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='raw-unicode-escape',
        encode=Codec.encode,
        decode=Codec.decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )
<|endoftext|>""" Python 'latin-1' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs

### Codec APIs

class Codec(codecs.Codec):

    # Note: Binding these as C functions will result in the class not
    # converting them to methods. This is intended.
    encode = codecs.latin_1_encode
    decode = codecs.latin_1_decode

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.latin_1_encode(input,self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.latin_1_decode(input,self.errors)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

class StreamConverter(StreamWriter,StreamReader):

    encode = codecs.latin_1_decode
    decode = codecs.latin_1_encode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='iso8859-1',
        encode=Codec.encode,
        decode=Codec.decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python 'ascii' Codec


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs

### Codec APIs

class Codec(codecs.Codec):

    # Note: Binding these as C functions will result in the class not
    # converting them to methods. This is intended.
    encode = codecs.ascii_encode
    decode = codecs.ascii_decode

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.ascii_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.ascii_decode(input, self.errors)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

class StreamConverter(StreamWriter,StreamReader):

    encode = codecs.ascii_decode
    decode = codecs.ascii_encode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='ascii',
        encode=Codec.encode,
        decode=Codec.decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )
<|endoftext|>""" Python Character Mapping Codec cp874 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP874.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp874',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\u20ac'   #  0x80 -> EURO SIGN
    '\ufffe'   #  0x81 -> UNDEFINED
    '\ufffe'   #  0x82 -> UNDEFINED
    '\ufffe'   #  0x83 -> UNDEFINED
    '\ufffe'   #  0x84 -> UNDEFINED
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\ufffe'   #  0x86 -> UNDEFINED
    '\ufffe'   #  0x87 -> UNDEFINED
    '\ufffe'   #  0x88 -> UNDEFINED
    '\ufffe'   #  0x89 -> UNDEFINED
    '\ufffe'   #  0x8A -> UNDEFINED
    '\ufffe'   #  0x8B -> UNDEFINED
    '\ufffe'   #  0x8C -> UNDEFINED
    '\ufffe'   #  0x8D -> UNDEFINED
    '\ufffe'   #  0x8E -> UNDEFINED
    '\ufffe'   #  0x8F -> UNDEFINED
    '\ufffe'   #  0x90 -> UNDEFINED
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\ufffe'   #  0x99 -> UNDEFINED
    '\ufffe'   #  0x9A -> UNDEFINED
    '\ufffe'   #  0x9B -> UNDEFINED
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0e01'   #  0xA1 -> THAI CHARACTER KO KAI
    '\u0e02'   #  0xA2 -> THAI CHARACTER KHO KHAI
    '\u0e03'   #  0xA3 -> THAI CHARACTER KHO KHUAT
    '\u0e04'   #  0xA4 -> THAI CHARACTER KHO KHWAI
    '\u0e05'   #  0xA5 -> THAI CHARACTER KHO KHON
    '\u0e06'   #  0xA6 -> THAI CHARACTER KHO RAKHANG
    '\u0e07'   #  0xA7 -> THAI CHARACTER NGO NGU
    '\u0e08'   #  0xA8 -> THAI CHARACTER CHO CHAN
    '\u0e09'   #  0xA9 -> THAI CHARACTER CHO CHING
    '\u0e0a'   #  0xAA -> THAI CHARACTER CHO CHANG
    '\u0e0b'   #  0xAB -> THAI CHARACTER SO SO
    '\u0e0c'   #  0xAC -> THAI CHARACTER CHO CHOE
    '\u0e0d'   #  0xAD -> THAI CHARACTER YO YING
    '\u0e0e'   #  0xAE -> THAI CHARACTER DO CHADA
    '\u0e0f'   #  0xAF -> THAI CHARACTER TO PATAK
    '\u0e10'   #  0xB0 -> THAI CHARACTER THO THAN
    '\u0e11'   #  0xB1 -> THAI CHARACTER THO NANGMONTHO
    '\u0e12'   #  0xB2 -> THAI CHARACTER THO PHUTHAO
    '\u0e13'   #  0xB3 -> THAI CHARACTER NO NEN
    '\u0e14'   #  0xB4 -> THAI CHARACTER DO DEK
    '\u0e15'   #  0xB5 -> THAI CHARACTER TO TAO
    '\u0e16'   #  0xB6 -> THAI CHARACTER THO THUNG
    '\u0e17'   #  0xB7 -> THAI CHARACTER THO THAHAN
    '\u0e18'   #  0xB8 -> THAI CHARACTER THO THONG
    '\u0e19'   #  0xB9 -> THAI CHARACTER NO NU
    '\u0e1a'   #  0xBA -> THAI CHARACTER BO BAIMAI
    '\u0e1b'   #  0xBB -> THAI CHARACTER PO PLA
    '\u0e1c'   #  0xBC -> THAI CHARACTER PHO PHUNG
    '\u0e1d'   #  0xBD -> THAI CHARACTER FO FA
    '\u0e1e'   #  0xBE -> THAI CHARACTER PHO PHAN
    '\u0e1f'   #  0xBF -> THAI CHARACTER FO FAN
    '\u0e20'   #  0xC0 -> THAI CHARACTER PHO SAMPHAO
    '\u0e21'   #  0xC1 -> THAI CHARACTER MO MA
    '\u0e22'   #  0xC2 -> THAI CHARACTER YO YAK
    '\u0e23'   #  0xC3 -> THAI CHARACTER RO RUA
    '\u0e24'   #  0xC4 -> THAI CHARACTER RU
    '\u0e25'   #  0xC5 -> THAI CHARACTER LO LING
    '\u0e26'   #  0xC6 -> THAI CHARACTER LU
    '\u0e27'   #  0xC7 -> THAI CHARACTER WO WAEN
    '\u0e28'   #  0xC8 -> THAI CHARACTER SO SALA
    '\u0e29'   #  0xC9 -> THAI CHARACTER SO RUSI
    '\u0e2a'   #  0xCA -> THAI CHARACTER SO SUA
    '\u0e2b'   #  0xCB -> THAI CHARACTER HO HIP
    '\u0e2c'   #  0xCC -> THAI CHARACTER LO CHULA
    '\u0e2d'   #  0xCD -> THAI CHARACTER O ANG
    '\u0e2e'   #  0xCE -> THAI CHARACTER HO NOKHUK
    '\u0e2f'   #  0xCF -> THAI CHARACTER PAIYANNOI
    '\u0e30'   #  0xD0 -> THAI CHARACTER SARA A
    '\u0e31'   #  0xD1 -> THAI CHARACTER MAI HAN-AKAT
    '\u0e32'   #  0xD2 -> THAI CHARACTER SARA AA
    '\u0e33'   #  0xD3 -> THAI CHARACTER SARA AM
    '\u0e34'   #  0xD4 -> THAI CHARACTER SARA I
    '\u0e35'   #  0xD5 -> THAI CHARACTER SARA II
    '\u0e36'   #  0xD6 -> THAI CHARACTER SARA UE
    '\u0e37'   #  0xD7 -> THAI CHARACTER SARA UEE
    '\u0e38'   #  0xD8 -> THAI CHARACTER SARA U
    '\u0e39'   #  0xD9 -> THAI CHARACTER SARA UU
    '\u0e3a'   #  0xDA -> THAI CHARACTER PHINTHU
    '\ufffe'   #  0xDB -> UNDEFINED
    '\ufffe'   #  0xDC -> UNDEFINED
    '\ufffe'   #  0xDD -> UNDEFINED
    '\ufffe'   #  0xDE -> UNDEFINED
    '\u0e3f'   #  0xDF -> THAI CURRENCY SYMBOL BAHT
    '\u0e40'   #  0xE0 -> THAI CHARACTER SARA E
    '\u0e41'   #  0xE1 -> THAI CHARACTER SARA AE
    '\u0e42'   #  0xE2 -> THAI CHARACTER SARA O
    '\u0e43'   #  0xE3 -> THAI CHARACTER SARA AI MAIMUAN
    '\u0e44'   #  0xE4 -> THAI CHARACTER SARA AI MAIMALAI
    '\u0e45'   #  0xE5 -> THAI CHARACTER LAKKHANGYAO
    '\u0e46'   #  0xE6 -> THAI CHARACTER MAIYAMOK
    '\u0e47'   #  0xE7 -> THAI CHARACTER MAITAIKHU
    '\u0e48'   #  0xE8 -> THAI CHARACTER MAI EK
    '\u0e49'   #  0xE9 -> THAI CHARACTER MAI THO
    '\u0e4a'   #  0xEA -> THAI CHARACTER MAI TRI
    '\u0e4b'   #  0xEB -> THAI CHARACTER MAI CHATTAWA
    '\u0e4c'   #  0xEC -> THAI CHARACTER THANTHAKHAT
    '\u0e4d'   #  0xED -> THAI CHARACTER NIKHAHIT
    '\u0e4e'   #  0xEE -> THAI CHARACTER YAMAKKAN
    '\u0e4f'   #  0xEF -> THAI CHARACTER FONGMAN
    '\u0e50'   #  0xF0 -> THAI DIGIT ZERO
    '\u0e51'   #  0xF1 -> THAI DIGIT ONE
    '\u0e52'   #  0xF2 -> THAI DIGIT TWO
    '\u0e53'   #  0xF3 -> THAI DIGIT THREE
    '\u0e54'   #  0xF4 -> THAI DIGIT FOUR
    '\u0e55'   #  0xF5 -> THAI DIGIT FIVE
    '\u0e56'   #  0xF6 -> THAI DIGIT SIX
    '\u0e57'   #  0xF7 -> THAI DIGIT SEVEN
    '\u0e58'   #  0xF8 -> THAI DIGIT EIGHT
    '\u0e59'   #  0xF9 -> THAI DIGIT NINE
    '\u0e5a'   #  0xFA -> THAI CHARACTER ANGKHANKHU
    '\u0e5b'   #  0xFB -> THAI CHARACTER KHOMUT
    '\ufffe'   #  0xFC -> UNDEFINED
    '\ufffe'   #  0xFD -> UNDEFINED
    '\ufffe'   #  0xFE -> UNDEFINED
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec for CP1125

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_map)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_map)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp1125',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )

### Decoding Map

decoding_map = codecs.make_identity_dict(range(256))
decoding_map.update({
    0x0080: 0x0410,     #  CYRILLIC CAPITAL LETTER A
    0x0081: 0x0411,     #  CYRILLIC CAPITAL LETTER BE
    0x0082: 0x0412,     #  CYRILLIC CAPITAL LETTER VE
    0x0083: 0x0413,     #  CYRILLIC CAPITAL LETTER GHE
    0x0084: 0x0414,     #  CYRILLIC CAPITAL LETTER DE
    0x0085: 0x0415,     #  CYRILLIC CAPITAL LETTER IE
    0x0086: 0x0416,     #  CYRILLIC CAPITAL LETTER ZHE
    0x0087: 0x0417,     #  CYRILLIC CAPITAL LETTER ZE
    0x0088: 0x0418,     #  CYRILLIC CAPITAL LETTER I
    0x0089: 0x0419,     #  CYRILLIC CAPITAL LETTER SHORT I
    0x008a: 0x041a,     #  CYRILLIC CAPITAL LETTER KA
    0x008b: 0x041b,     #  CYRILLIC CAPITAL LETTER EL
    0x008c: 0x041c,     #  CYRILLIC CAPITAL LETTER EM
    0x008d: 0x041d,     #  CYRILLIC CAPITAL LETTER EN
    0x008e: 0x041e,     #  CYRILLIC CAPITAL LETTER O
    0x008f: 0x041f,     #  CYRILLIC CAPITAL LETTER PE
    0x0090: 0x0420,     #  CYRILLIC CAPITAL LETTER ER
    0x0091: 0x0421,     #  CYRILLIC CAPITAL LETTER ES
    0x0092: 0x0422,     #  CYRILLIC CAPITAL LETTER TE
    0x0093: 0x0423,     #  CYRILLIC CAPITAL LETTER U
    0x0094: 0x0424,     #  CYRILLIC CAPITAL LETTER EF
    0x0095: 0x0425,     #  CYRILLIC CAPITAL LETTER HA
    0x0096: 0x0426,     #  CYRILLIC CAPITAL LETTER TSE
    0x0097: 0x0427,     #  CYRILLIC CAPITAL LETTER CHE
    0x0098: 0x0428,     #  CYRILLIC CAPITAL LETTER SHA
    0x0099: 0x0429,     #  CYRILLIC CAPITAL LETTER SHCHA
    0x009a: 0x042a,     #  CYRILLIC CAPITAL LETTER HARD SIGN
    0x009b: 0x042b,     #  CYRILLIC CAPITAL LETTER YERU
    0x009c: 0x042c,     #  CYRILLIC CAPITAL LETTER SOFT SIGN
    0x009d: 0x042d,     #  CYRILLIC CAPITAL LETTER E
    0x009e: 0x042e,     #  CYRILLIC CAPITAL LETTER YU
    0x009f: 0x042f,     #  CYRILLIC CAPITAL LETTER YA
    0x00a0: 0x0430,     #  CYRILLIC SMALL LETTER A
    0x00a1: 0x0431,     #  CYRILLIC SMALL LETTER BE
    0x00a2: 0x0432,     #  CYRILLIC SMALL LETTER VE
    0x00a3: 0x0433,     #  CYRILLIC SMALL LETTER GHE
    0x00a4: 0x0434,     #  CYRILLIC SMALL LETTER DE
    0x00a5: 0x0435,     #  CYRILLIC SMALL LETTER IE
    0x00a6: 0x0436,     #  CYRILLIC SMALL LETTER ZHE
    0x00a7: 0x0437,     #  CYRILLIC SMALL LETTER ZE
    0x00a8: 0x0438,     #  CYRILLIC SMALL LETTER I
    0x00a9: 0x0439,     #  CYRILLIC SMALL LETTER SHORT I
    0x00aa: 0x043a,     #  CYRILLIC SMALL LETTER KA
    0x00ab: 0x043b,     #  CYRILLIC SMALL LETTER EL
    0x00ac: 0x043c,     #  CYRILLIC SMALL LETTER EM
    0x00ad: 0x043d,     #  CYRILLIC SMALL LETTER EN
    0x00ae: 0x043e,     #  CYRILLIC SMALL LETTER O
    0x00af: 0x043f,     #  CYRILLIC SMALL LETTER PE
    0x00b0: 0x2591,     #  LIGHT SHADE
    0x00b1: 0x2592,     #  MEDIUM SHADE
    0x00b2: 0x2593,     #  DARK SHADE
    0x00b3: 0x2502,     #  BOX DRAWINGS LIGHT VERTICAL
    0x00b4: 0x2524,     #  BOX DRAWINGS LIGHT VERTICAL AND LEFT
    0x00b5: 0x2561,     #  BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    0x00b6: 0x2562,     #  BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    0x00b7: 0x2556,     #  BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    0x00b8: 0x2555,     #  BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    0x00b9: 0x2563,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x00ba: 0x2551,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x00bb: 0x2557,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x00bc: 0x255d,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x00bd: 0x255c,     #  BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    0x00be: 0x255b,     #  BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    0x00bf: 0x2510,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x00c0: 0x2514,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x00c1: 0x2534,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x00c2: 0x252c,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x00c3: 0x251c,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x00c4: 0x2500,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x00c5: 0x253c,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x00c6: 0x255e,     #  BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    0x00c7: 0x255f,     #  BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    0x00c8: 0x255a,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x00c9: 0x2554,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x00ca: 0x2569,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x00cb: 0x2566,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x00cc: 0x2560,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x00cd: 0x2550,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x00ce: 0x256c,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x00cf: 0x2567,     #  BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    0x00d0: 0x2568,     #  BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    0x00d1: 0x2564,     #  BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    0x00d2: 0x2565,     #  BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    0x00d3: 0x2559,     #  BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    0x00d4: 0x2558,     #  BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    0x00d5: 0x2552,     #  BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    0x00d6: 0x2553,     #  BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    0x00d7: 0x256b,     #  BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    0x00d8: 0x256a,     #  BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    0x00d9: 0x2518,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x00da: 0x250c,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x00db: 0x2588,     #  FULL BLOCK
    0x00dc: 0x2584,     #  LOWER HALF BLOCK
    0x00dd: 0x258c,     #  LEFT HALF BLOCK
    0x00de: 0x2590,     #  RIGHT HALF BLOCK
    0x00df: 0x2580,     #  UPPER HALF BLOCK
    0x00e0: 0x0440,     #  CYRILLIC SMALL LETTER ER
    0x00e1: 0x0441,     #  CYRILLIC SMALL LETTER ES
    0x00e2: 0x0442,     #  CYRILLIC SMALL LETTER TE
    0x00e3: 0x0443,     #  CYRILLIC SMALL LETTER U
    0x00e4: 0x0444,     #  CYRILLIC SMALL LETTER EF
    0x00e5: 0x0445,     #  CYRILLIC SMALL LETTER HA
    0x00e6: 0x0446,     #  CYRILLIC SMALL LETTER TSE
    0x00e7: 0x0447,     #  CYRILLIC SMALL LETTER CHE
    0x00e8: 0x0448,     #  CYRILLIC SMALL LETTER SHA
    0x00e9: 0x0449,     #  CYRILLIC SMALL LETTER SHCHA
    0x00ea: 0x044a,     #  CYRILLIC SMALL LETTER HARD SIGN
    0x00eb: 0x044b,     #  CYRILLIC SMALL LETTER YERU
    0x00ec: 0x044c,     #  CYRILLIC SMALL LETTER SOFT SIGN
    0x00ed: 0x044d,     #  CYRILLIC SMALL LETTER E
    0x00ee: 0x044e,     #  CYRILLIC SMALL LETTER YU
    0x00ef: 0x044f,     #  CYRILLIC SMALL LETTER YA
    0x00f0: 0x0401,     #  CYRILLIC CAPITAL LETTER IO
    0x00f1: 0x0451,     #  CYRILLIC SMALL LETTER IO
    0x00f2: 0x0490,     #  CYRILLIC CAPITAL LETTER GHE WITH UPTURN
    0x00f3: 0x0491,     #  CYRILLIC SMALL LETTER GHE WITH UPTURN
    0x00f4: 0x0404,     #  CYRILLIC CAPITAL LETTER UKRAINIAN IE
    0x00f5: 0x0454,     #  CYRILLIC SMALL LETTER UKRAINIAN IE
    0x00f6: 0x0406,     #  CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    0x00f7: 0x0456,     #  CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    0x00f8: 0x0407,     #  CYRILLIC CAPITAL LETTER YI
    0x00f9: 0x0457,     #  CYRILLIC SMALL LETTER YI
    0x00fa: 0x00b7,     #  MIDDLE DOT
    0x00fb: 0x221a,     #  SQUARE ROOT
    0x00fc: 0x2116,     #  NUMERO SIGN
    0x00fd: 0x00a4,     #  CURRENCY SIGN
    0x00fe: 0x25a0,     #  BLACK SQUARE
    0x00ff: 0x00a0,     #  NO-BREAK SPACE
})

### Decoding Table

decoding_table = (
    '\x00'     #  0x0000 -> NULL
    '\x01'     #  0x0001 -> START OF HEADING
    '\x02'     #  0x0002 -> START OF TEXT
    '\x03'     #  0x0003 -> END OF TEXT
    '\x04'     #  0x0004 -> END OF TRANSMISSION
    '\x05'     #  0x0005 -> ENQUIRY
    '\x06'     #  0x0006 -> ACKNOWLEDGE
    '\x07'     #  0x0007 -> BELL
    '\x08'     #  0x0008 -> BACKSPACE
    '\t'       #  0x0009 -> HORIZONTAL TABULATION
    '\n'       #  0x000a -> LINE FEED
    '\x0b'     #  0x000b -> VERTICAL TABULATION
    '\x0c'     #  0x000c -> FORM FEED
    '\r'       #  0x000d -> CARRIAGE RETURN
    '\x0e'     #  0x000e -> SHIFT OUT
    '\x0f'     #  0x000f -> SHIFT IN
    '\x10'     #  0x0010 -> DATA LINK ESCAPE
    '\x11'     #  0x0011 -> DEVICE CONTROL ONE
    '\x12'     #  0x0012 -> DEVICE CONTROL TWO
    '\x13'     #  0x0013 -> DEVICE CONTROL THREE
    '\x14'     #  0x0014 -> DEVICE CONTROL FOUR
    '\x15'     #  0x0015 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x0016 -> SYNCHRONOUS IDLE
    '\x17'     #  0x0017 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x0018 -> CANCEL
    '\x19'     #  0x0019 -> END OF MEDIUM
    '\x1a'     #  0x001a -> SUBSTITUTE
    '\x1b'     #  0x001b -> ESCAPE
    '\x1c'     #  0x001c -> FILE SEPARATOR
    '\x1d'     #  0x001d -> GROUP SEPARATOR
    '\x1e'     #  0x001e -> RECORD SEPARATOR
    '\x1f'     #  0x001f -> UNIT SEPARATOR
    ' '        #  0x0020 -> SPACE
    '!'        #  0x0021 -> EXCLAMATION MARK
    '"'        #  0x0022 -> QUOTATION MARK
    '#'        #  0x0023 -> NUMBER SIGN
    '$'        #  0x0024 -> DOLLAR SIGN
    '%'        #  0x0025 -> PERCENT SIGN
    '&'        #  0x0026 -> AMPERSAND
    "'"        #  0x0027 -> APOSTROPHE
    '('        #  0x0028 -> LEFT PARENTHESIS
    ')'        #  0x0029 -> RIGHT PARENTHESIS
    '*'        #  0x002a -> ASTERISK
    '+'        #  0x002b -> PLUS SIGN
    ','        #  0x002c -> COMMA
    '-'        #  0x002d -> HYPHEN-MINUS
    '.'        #  0x002e -> FULL STOP
    '/'        #  0x002f -> SOLIDUS
    '0'        #  0x0030 -> DIGIT ZERO
    '1'        #  0x0031 -> DIGIT ONE
    '2'        #  0x0032 -> DIGIT TWO
    '3'        #  0x0033 -> DIGIT THREE
    '4'        #  0x0034 -> DIGIT FOUR
    '5'        #  0x0035 -> DIGIT FIVE
    '6'        #  0x0036 -> DIGIT SIX
    '7'        #  0x0037 -> DIGIT SEVEN
    '8'        #  0x0038 -> DIGIT EIGHT
    '9'        #  0x0039 -> DIGIT NINE
    ':'        #  0x003a -> COLON
    ';'        #  0x003b -> SEMICOLON
    '<'        #  0x003c -> LESS-THAN SIGN
    '='        #  0x003d -> EQUALS SIGN
    '>'        #  0x003e -> GREATER-THAN SIGN
    '?'        #  0x003f -> QUESTION MARK
    '@'        #  0x0040 -> COMMERCIAL AT
    'A'        #  0x0041 -> LATIN CAPITAL LETTER A
    'B'        #  0x0042 -> LATIN CAPITAL LETTER B
    'C'        #  0x0043 -> LATIN CAPITAL LETTER C
    'D'        #  0x0044 -> LATIN CAPITAL LETTER D
    'E'        #  0x0045 -> LATIN CAPITAL LETTER E
    'F'        #  0x0046 -> LATIN CAPITAL LETTER F
    'G'        #  0x0047 -> LATIN CAPITAL LETTER G
    'H'        #  0x0048 -> LATIN CAPITAL LETTER H
    'I'        #  0x0049 -> LATIN CAPITAL LETTER I
    'J'        #  0x004a -> LATIN CAPITAL LETTER J
    'K'        #  0x004b -> LATIN CAPITAL LETTER K
    'L'        #  0x004c -> LATIN CAPITAL LETTER L
    'M'        #  0x004d -> LATIN CAPITAL LETTER M
    'N'        #  0x004e -> LATIN CAPITAL LETTER N
    'O'        #  0x004f -> LATIN CAPITAL LETTER O
    'P'        #  0x0050 -> LATIN CAPITAL LETTER P
    'Q'        #  0x0051 -> LATIN CAPITAL LETTER Q
    'R'        #  0x0052 -> LATIN CAPITAL LETTER R
    'S'        #  0x0053 -> LATIN CAPITAL LETTER S
    'T'        #  0x0054 -> LATIN CAPITAL LETTER T
    'U'        #  0x0055 -> LATIN CAPITAL LETTER U
    'V'        #  0x0056 -> LATIN CAPITAL LETTER V
    'W'        #  0x0057 -> LATIN CAPITAL LETTER W
    'X'        #  0x0058 -> LATIN CAPITAL LETTER X
    'Y'        #  0x0059 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x005a -> LATIN CAPITAL LETTER Z
    '['        #  0x005b -> LEFT SQUARE BRACKET
    '\\'       #  0x005c -> REVERSE SOLIDUS
    ']'        #  0x005d -> RIGHT SQUARE BRACKET
    '^'        #  0x005e -> CIRCUMFLEX ACCENT
    '_'        #  0x005f -> LOW LINE
    '`'        #  0x0060 -> GRAVE ACCENT
    'a'        #  0x0061 -> LATIN SMALL LETTER A
    'b'        #  0x0062 -> LATIN SMALL LETTER B
    'c'        #  0x0063 -> LATIN SMALL LETTER C
    'd'        #  0x0064 -> LATIN SMALL LETTER D
    'e'        #  0x0065 -> LATIN SMALL LETTER E
    'f'        #  0x0066 -> LATIN SMALL LETTER F
    'g'        #  0x0067 -> LATIN SMALL LETTER G
    'h'        #  0x0068 -> LATIN SMALL LETTER H
    'i'        #  0x0069 -> LATIN SMALL LETTER I
    'j'        #  0x006a -> LATIN SMALL LETTER J
    'k'        #  0x006b -> LATIN SMALL LETTER K
    'l'        #  0x006c -> LATIN SMALL LETTER L
    'm'        #  0x006d -> LATIN SMALL LETTER M
    'n'        #  0x006e -> LATIN SMALL LETTER N
    'o'        #  0x006f -> LATIN SMALL LETTER O
    'p'        #  0x0070 -> LATIN SMALL LETTER P
    'q'        #  0x0071 -> LATIN SMALL LETTER Q
    'r'        #  0x0072 -> LATIN SMALL LETTER R
    's'        #  0x0073 -> LATIN SMALL LETTER S
    't'        #  0x0074 -> LATIN SMALL LETTER T
    'u'        #  0x0075 -> LATIN SMALL LETTER U
    'v'        #  0x0076 -> LATIN SMALL LETTER V
    'w'        #  0x0077 -> LATIN SMALL LETTER W
    'x'        #  0x0078 -> LATIN SMALL LETTER X
    'y'        #  0x0079 -> LATIN SMALL LETTER Y
    'z'        #  0x007a -> LATIN SMALL LETTER Z
    '{'        #  0x007b -> LEFT CURLY BRACKET
    '|'        #  0x007c -> VERTICAL LINE
    '}'        #  0x007d -> RIGHT CURLY BRACKET
    '~'        #  0x007e -> TILDE
    '\x7f'     #  0x007f -> DELETE
    '\u0410'   #  0x0080 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0x0081 -> CYRILLIC CAPITAL LETTER BE
    '\u0412'   #  0x0082 -> CYRILLIC CAPITAL LETTER VE
    '\u0413'   #  0x0083 -> CYRILLIC CAPITAL LETTER GHE
    '\u0414'   #  0x0084 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0x0085 -> CYRILLIC CAPITAL LETTER IE
    '\u0416'   #  0x0086 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0417'   #  0x0087 -> CYRILLIC CAPITAL LETTER ZE
    '\u0418'   #  0x0088 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0x0089 -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0x008a -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0x008b -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0x008c -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0x008d -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0x008e -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0x008f -> CYRILLIC CAPITAL LETTER PE
    '\u0420'   #  0x0090 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0x0091 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0x0092 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0x0093 -> CYRILLIC CAPITAL LETTER U
    '\u0424'   #  0x0094 -> CYRILLIC CAPITAL LETTER EF
    '\u0425'   #  0x0095 -> CYRILLIC CAPITAL LETTER HA
    '\u0426'   #  0x0096 -> CYRILLIC CAPITAL LETTER TSE
    '\u0427'   #  0x0097 -> CYRILLIC CAPITAL LETTER CHE
    '\u0428'   #  0x0098 -> CYRILLIC CAPITAL LETTER SHA
    '\u0429'   #  0x0099 -> CYRILLIC CAPITAL LETTER SHCHA
    '\u042a'   #  0x009a -> CYRILLIC CAPITAL LETTER HARD SIGN
    '\u042b'   #  0x009b -> CYRILLIC CAPITAL LETTER YERU
    '\u042c'   #  0x009c -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042d'   #  0x009d -> CYRILLIC CAPITAL LETTER E
    '\u042e'   #  0x009e -> CYRILLIC CAPITAL LETTER YU
    '\u042f'   #  0x009f -> CYRILLIC CAPITAL LETTER YA
    '\u0430'   #  0x00a0 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0x00a1 -> CYRILLIC SMALL LETTER BE
    '\u0432'   #  0x00a2 -> CYRILLIC SMALL LETTER VE
    '\u0433'   #  0x00a3 -> CYRILLIC SMALL LETTER GHE
    '\u0434'   #  0x00a4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0x00a5 -> CYRILLIC SMALL LETTER IE
    '\u0436'   #  0x00a6 -> CYRILLIC SMALL LETTER ZHE
    '\u0437'   #  0x00a7 -> CYRILLIC SMALL LETTER ZE
    '\u0438'   #  0x00a8 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0x00a9 -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0x00aa -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0x00ab -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0x00ac -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0x00ad -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0x00ae -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0x00af -> CYRILLIC SMALL LETTER PE
    '\u2591'   #  0x00b0 -> LIGHT SHADE
    '\u2592'   #  0x00b1 -> MEDIUM SHADE
    '\u2593'   #  0x00b2 -> DARK SHADE
    '\u2502'   #  0x00b3 -> BOX DRAWINGS LIGHT VERTICAL
    '\u2524'   #  0x00b4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u2561'   #  0x00b5 -> BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    '\u2562'   #  0x00b6 -> BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    '\u2556'   #  0x00b7 -> BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    '\u2555'   #  0x00b8 -> BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    '\u2563'   #  0x00b9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'   #  0x00ba -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'   #  0x00bb -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'   #  0x00bc -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u255c'   #  0x00bd -> BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    '\u255b'   #  0x00be -> BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    '\u2510'   #  0x00bf -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x00c0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'   #  0x00c1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'   #  0x00c2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0x00c3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'   #  0x00c4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'   #  0x00c5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u255e'   #  0x00c6 -> BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    '\u255f'   #  0x00c7 -> BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    '\u255a'   #  0x00c8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'   #  0x00c9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'   #  0x00ca -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'   #  0x00cb -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'   #  0x00cc -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'   #  0x00cd -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'   #  0x00ce -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\u2567'   #  0x00cf -> BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    '\u2568'   #  0x00d0 -> BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    '\u2564'   #  0x00d1 -> BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    '\u2565'   #  0x00d2 -> BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    '\u2559'   #  0x00d3 -> BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    '\u2558'   #  0x00d4 -> BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    '\u2552'   #  0x00d5 -> BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    '\u2553'   #  0x00d6 -> BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    '\u256b'   #  0x00d7 -> BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    '\u256a'   #  0x00d8 -> BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    '\u2518'   #  0x00d9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'   #  0x00da -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'   #  0x00db -> FULL BLOCK
    '\u2584'   #  0x00dc -> LOWER HALF BLOCK
    '\u258c'   #  0x00dd -> LEFT HALF BLOCK
    '\u2590'   #  0x00de -> RIGHT HALF BLOCK
    '\u2580'   #  0x00df -> UPPER HALF BLOCK
    '\u0440'   #  0x00e0 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0x00e1 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0x00e2 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0x00e3 -> CYRILLIC SMALL LETTER U
    '\u0444'   #  0x00e4 -> CYRILLIC SMALL LETTER EF
    '\u0445'   #  0x00e5 -> CYRILLIC SMALL LETTER HA
    '\u0446'   #  0x00e6 -> CYRILLIC SMALL LETTER TSE
    '\u0447'   #  0x00e7 -> CYRILLIC SMALL LETTER CHE
    '\u0448'   #  0x00e8 -> CYRILLIC SMALL LETTER SHA
    '\u0449'   #  0x00e9 -> CYRILLIC SMALL LETTER SHCHA
    '\u044a'   #  0x00ea -> CYRILLIC SMALL LETTER HARD SIGN
    '\u044b'   #  0x00eb -> CYRILLIC SMALL LETTER YERU
    '\u044c'   #  0x00ec -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044d'   #  0x00ed -> CYRILLIC SMALL LETTER E
    '\u044e'   #  0x00ee -> CYRILLIC SMALL LETTER YU
    '\u044f'   #  0x00ef -> CYRILLIC SMALL LETTER YA
    '\u0401'   #  0x00f0 -> CYRILLIC CAPITAL LETTER IO
    '\u0451'   #  0x00f1 -> CYRILLIC SMALL LETTER IO
    '\u0490'   #  0x00f2 -> CYRILLIC CAPITAL LETTER GHE WITH UPTURN
    '\u0491'   #  0x00f3 -> CYRILLIC SMALL LETTER GHE WITH UPTURN
    '\u0404'   #  0x00f4 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u0454'   #  0x00f5 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u0406'   #  0x00f6 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0456'   #  0x00f7 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0407'   #  0x00f8 -> CYRILLIC CAPITAL LETTER YI
    '\u0457'   #  0x00f9 -> CYRILLIC SMALL LETTER YI
    '\xb7'     #  0x00fa -> MIDDLE DOT
    '\u221a'   #  0x00fb -> SQUARE ROOT
    '\u2116'   #  0x00fc -> NUMERO SIGN
    '\xa4'     #  0x00fd -> CURRENCY SIGN
    '\u25a0'   #  0x00fe -> BLACK SQUARE
    '\xa0'     #  0x00ff -> NO-BREAK SPACE
)

### Encoding Map

encoding_map = {
    0x0000: 0x0000,     #  NULL
    0x0001: 0x0001,     #  START OF HEADING
    0x0002: 0x0002,     #  START OF TEXT
    0x0003: 0x0003,     #  END OF TEXT
    0x0004: 0x0004,     #  END OF TRANSMISSION
    0x0005: 0x0005,     #  ENQUIRY
    0x0006: 0x0006,     #  ACKNOWLEDGE
    0x0007: 0x0007,     #  BELL
    0x0008: 0x0008,     #  BACKSPACE
    0x0009: 0x0009,     #  HORIZONTAL TABULATION
    0x000a: 0x000a,     #  LINE FEED
    0x000b: 0x000b,     #  VERTICAL TABULATION
    0x000c: 0x000c,     #  FORM FEED
    0x000d: 0x000d,     #  CARRIAGE RETURN
    0x000e: 0x000e,     #  SHIFT OUT
    0x000f: 0x000f,     #  SHIFT IN
    0x0010: 0x0010,     #  DATA LINK ESCAPE
    0x0011: 0x0011,     #  DEVICE CONTROL ONE
    0x0012: 0x0012,     #  DEVICE CONTROL TWO
    0x0013: 0x0013,     #  DEVICE CONTROL THREE
    0x0014: 0x0014,     #  DEVICE CONTROL FOUR
    0x0015: 0x0015,     #  NEGATIVE ACKNOWLEDGE
    0x0016: 0x0016,     #  SYNCHRONOUS IDLE
    0x0017: 0x0017,     #  END OF TRANSMISSION BLOCK
    0x0018: 0x0018,     #  CANCEL
    0x0019: 0x0019,     #  END OF MEDIUM
    0x001a: 0x001a,     #  SUBSTITUTE
    0x001b: 0x001b,     #  ESCAPE
    0x001c: 0x001c,     #  FILE SEPARATOR
    0x001d: 0x001d,     #  GROUP SEPARATOR
    0x001e: 0x001e,     #  RECORD SEPARATOR
    0x001f: 0x001f,     #  UNIT SEPARATOR
    0x0020: 0x0020,     #  SPACE
    0x0021: 0x0021,     #  EXCLAMATION MARK
    0x0022: 0x0022,     #  QUOTATION MARK
    0x0023: 0x0023,     #  NUMBER SIGN
    0x0024: 0x0024,     #  DOLLAR SIGN
    0x0025: 0x0025,     #  PERCENT SIGN
    0x0026: 0x0026,     #  AMPERSAND
    0x0027: 0x0027,     #  APOSTROPHE
    0x0028: 0x0028,     #  LEFT PARENTHESIS
    0x0029: 0x0029,     #  RIGHT PARENTHESIS
    0x002a: 0x002a,     #  ASTERISK
    0x002b: 0x002b,     #  PLUS SIGN
    0x002c: 0x002c,     #  COMMA
    0x002d: 0x002d,     #  HYPHEN-MINUS
    0x002e: 0x002e,     #  FULL STOP
    0x002f: 0x002f,     #  SOLIDUS
    0x0030: 0x0030,     #  DIGIT ZERO
    0x0031: 0x0031,     #  DIGIT ONE
    0x0032: 0x0032,     #  DIGIT TWO
    0x0033: 0x0033,     #  DIGIT THREE
    0x0034: 0x0034,     #  DIGIT FOUR
    0x0035: 0x0035,     #  DIGIT FIVE
    0x0036: 0x0036,     #  DIGIT SIX
    0x0037: 0x0037,     #  DIGIT SEVEN
    0x0038: 0x0038,     #  DIGIT EIGHT
    0x0039: 0x0039,     #  DIGIT NINE
    0x003a: 0x003a,     #  COLON
    0x003b: 0x003b,     #  SEMICOLON
    0x003c: 0x003c,     #  LESS-THAN SIGN
    0x003d: 0x003d,     #  EQUALS SIGN
    0x003e: 0x003e,     #  GREATER-THAN SIGN
    0x003f: 0x003f,     #  QUESTION MARK
    0x0040: 0x0040,     #  COMMERCIAL AT
    0x0041: 0x0041,     #  LATIN CAPITAL LETTER A
    0x0042: 0x0042,     #  LATIN CAPITAL LETTER B
    0x0043: 0x0043,     #  LATIN CAPITAL LETTER C
    0x0044: 0x0044,     #  LATIN CAPITAL LETTER D
    0x0045: 0x0045,     #  LATIN CAPITAL LETTER E
    0x0046: 0x0046,     #  LATIN CAPITAL LETTER F
    0x0047: 0x0047,     #  LATIN CAPITAL LETTER G
    0x0048: 0x0048,     #  LATIN CAPITAL LETTER H
    0x0049: 0x0049,     #  LATIN CAPITAL LETTER I
    0x004a: 0x004a,     #  LATIN CAPITAL LETTER J
    0x004b: 0x004b,     #  LATIN CAPITAL LETTER K
    0x004c: 0x004c,     #  LATIN CAPITAL LETTER L
    0x004d: 0x004d,     #  LATIN CAPITAL LETTER M
    0x004e: 0x004e,     #  LATIN CAPITAL LETTER N
    0x004f: 0x004f,     #  LATIN CAPITAL LETTER O
    0x0050: 0x0050,     #  LATIN CAPITAL LETTER P
    0x0051: 0x0051,     #  LATIN CAPITAL LETTER Q
    0x0052: 0x0052,     #  LATIN CAPITAL LETTER R
    0x0053: 0x0053,     #  LATIN CAPITAL LETTER S
    0x0054: 0x0054,     #  LATIN CAPITAL LETTER T
    0x0055: 0x0055,     #  LATIN CAPITAL LETTER U
    0x0056: 0x0056,     #  LATIN CAPITAL LETTER V
    0x0057: 0x0057,     #  LATIN CAPITAL LETTER W
    0x0058: 0x0058,     #  LATIN CAPITAL LETTER X
    0x0059: 0x0059,     #  LATIN CAPITAL LETTER Y
    0x005a: 0x005a,     #  LATIN CAPITAL LETTER Z
    0x005b: 0x005b,     #  LEFT SQUARE BRACKET
    0x005c: 0x005c,     #  REVERSE SOLIDUS
    0x005d: 0x005d,     #  RIGHT SQUARE BRACKET
    0x005e: 0x005e,     #  CIRCUMFLEX ACCENT
    0x005f: 0x005f,     #  LOW LINE
    0x0060: 0x0060,     #  GRAVE ACCENT
    0x0061: 0x0061,     #  LATIN SMALL LETTER A
    0x0062: 0x0062,     #  LATIN SMALL LETTER B
    0x0063: 0x0063,     #  LATIN SMALL LETTER C
    0x0064: 0x0064,     #  LATIN SMALL LETTER D
    0x0065: 0x0065,     #  LATIN SMALL LETTER E
    0x0066: 0x0066,     #  LATIN SMALL LETTER F
    0x0067: 0x0067,     #  LATIN SMALL LETTER G
    0x0068: 0x0068,     #  LATIN SMALL LETTER H
    0x0069: 0x0069,     #  LATIN SMALL LETTER I
    0x006a: 0x006a,     #  LATIN SMALL LETTER J
    0x006b: 0x006b,     #  LATIN SMALL LETTER K
    0x006c: 0x006c,     #  LATIN SMALL LETTER L
    0x006d: 0x006d,     #  LATIN SMALL LETTER M
    0x006e: 0x006e,     #  LATIN SMALL LETTER N
    0x006f: 0x006f,     #  LATIN SMALL LETTER O
    0x0070: 0x0070,     #  LATIN SMALL LETTER P
    0x0071: 0x0071,     #  LATIN SMALL LETTER Q
    0x0072: 0x0072,     #  LATIN SMALL LETTER R
    0x0073: 0x0073,     #  LATIN SMALL LETTER S
    0x0074: 0x0074,     #  LATIN SMALL LETTER T
    0x0075: 0x0075,     #  LATIN SMALL LETTER U
    0x0076: 0x0076,     #  LATIN SMALL LETTER V
    0x0077: 0x0077,     #  LATIN SMALL LETTER W
    0x0078: 0x0078,     #  LATIN SMALL LETTER X
    0x0079: 0x0079,     #  LATIN SMALL LETTER Y
    0x007a: 0x007a,     #  LATIN SMALL LETTER Z
    0x007b: 0x007b,     #  LEFT CURLY BRACKET
    0x007c: 0x007c,     #  VERTICAL LINE
    0x007d: 0x007d,     #  RIGHT CURLY BRACKET
    0x007e: 0x007e,     #  TILDE
    0x007f: 0x007f,     #  DELETE
    0x00a0: 0x00ff,     #  NO-BREAK SPACE
    0x00a4: 0x00fd,     #  CURRENCY SIGN
    0x00b7: 0x00fa,     #  MIDDLE DOT
    0x0401: 0x00f0,     #  CYRILLIC CAPITAL LETTER IO
    0x0404: 0x00f4,     #  CYRILLIC CAPITAL LETTER UKRAINIAN IE
    0x0406: 0x00f6,     #  CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    0x0407: 0x00f8,     #  CYRILLIC CAPITAL LETTER YI
    0x0410: 0x0080,     #  CYRILLIC CAPITAL LETTER A
    0x0411: 0x0081,     #  CYRILLIC CAPITAL LETTER BE
    0x0412: 0x0082,     #  CYRILLIC CAPITAL LETTER VE
    0x0413: 0x0083,     #  CYRILLIC CAPITAL LETTER GHE
    0x0414: 0x0084,     #  CYRILLIC CAPITAL LETTER DE
    0x0415: 0x0085,     #  CYRILLIC CAPITAL LETTER IE
    0x0416: 0x0086,     #  CYRILLIC CAPITAL LETTER ZHE
    0x0417: 0x0087,     #  CYRILLIC CAPITAL LETTER ZE
    0x0418: 0x0088,     #  CYRILLIC CAPITAL LETTER I
    0x0419: 0x0089,     #  CYRILLIC CAPITAL LETTER SHORT I
    0x041a: 0x008a,     #  CYRILLIC CAPITAL LETTER KA
    0x041b: 0x008b,     #  CYRILLIC CAPITAL LETTER EL
    0x041c: 0x008c,     #  CYRILLIC CAPITAL LETTER EM
    0x041d: 0x008d,     #  CYRILLIC CAPITAL LETTER EN
    0x041e: 0x008e,     #  CYRILLIC CAPITAL LETTER O
    0x041f: 0x008f,     #  CYRILLIC CAPITAL LETTER PE
    0x0420: 0x0090,     #  CYRILLIC CAPITAL LETTER ER
    0x0421: 0x0091,     #  CYRILLIC CAPITAL LETTER ES
    0x0422: 0x0092,     #  CYRILLIC CAPITAL LETTER TE
    0x0423: 0x0093,     #  CYRILLIC CAPITAL LETTER U
    0x0424: 0x0094,     #  CYRILLIC CAPITAL LETTER EF
    0x0425: 0x0095,     #  CYRILLIC CAPITAL LETTER HA
    0x0426: 0x0096,     #  CYRILLIC CAPITAL LETTER TSE
    0x0427: 0x0097,     #  CYRILLIC CAPITAL LETTER CHE
    0x0428: 0x0098,     #  CYRILLIC CAPITAL LETTER SHA
    0x0429: 0x0099,     #  CYRILLIC CAPITAL LETTER SHCHA
    0x042a: 0x009a,     #  CYRILLIC CAPITAL LETTER HARD SIGN
    0x042b: 0x009b,     #  CYRILLIC CAPITAL LETTER YERU
    0x042c: 0x009c,     #  CYRILLIC CAPITAL LETTER SOFT SIGN
    0x042d: 0x009d,     #  CYRILLIC CAPITAL LETTER E
    0x042e: 0x009e,     #  CYRILLIC CAPITAL LETTER YU
    0x042f: 0x009f,     #  CYRILLIC CAPITAL LETTER YA
    0x0430: 0x00a0,     #  CYRILLIC SMALL LETTER A
    0x0431: 0x00a1,     #  CYRILLIC SMALL LETTER BE
    0x0432: 0x00a2,     #  CYRILLIC SMALL LETTER VE
    0x0433: 0x00a3,     #  CYRILLIC SMALL LETTER GHE
    0x0434: 0x00a4,     #  CYRILLIC SMALL LETTER DE
    0x0435: 0x00a5,     #  CYRILLIC SMALL LETTER IE
    0x0436: 0x00a6,     #  CYRILLIC SMALL LETTER ZHE
    0x0437: 0x00a7,     #  CYRILLIC SMALL LETTER ZE
    0x0438: 0x00a8,     #  CYRILLIC SMALL LETTER I
    0x0439: 0x00a9,     #  CYRILLIC SMALL LETTER SHORT I
    0x043a: 0x00aa,     #  CYRILLIC SMALL LETTER KA
    0x043b: 0x00ab,     #  CYRILLIC SMALL LETTER EL
    0x043c: 0x00ac,     #  CYRILLIC SMALL LETTER EM
    0x043d: 0x00ad,     #  CYRILLIC SMALL LETTER EN
    0x043e: 0x00ae,     #  CYRILLIC SMALL LETTER O
    0x043f: 0x00af,     #  CYRILLIC SMALL LETTER PE
    0x0440: 0x00e0,     #  CYRILLIC SMALL LETTER ER
    0x0441: 0x00e1,     #  CYRILLIC SMALL LETTER ES
    0x0442: 0x00e2,     #  CYRILLIC SMALL LETTER TE
    0x0443: 0x00e3,     #  CYRILLIC SMALL LETTER U
    0x0444: 0x00e4,     #  CYRILLIC SMALL LETTER EF
    0x0445: 0x00e5,     #  CYRILLIC SMALL LETTER HA
    0x0446: 0x00e6,     #  CYRILLIC SMALL LETTER TSE
    0x0447: 0x00e7,     #  CYRILLIC SMALL LETTER CHE
    0x0448: 0x00e8,     #  CYRILLIC SMALL LETTER SHA
    0x0449: 0x00e9,     #  CYRILLIC SMALL LETTER SHCHA
    0x044a: 0x00ea,     #  CYRILLIC SMALL LETTER HARD SIGN
    0x044b: 0x00eb,     #  CYRILLIC SMALL LETTER YERU
    0x044c: 0x00ec,     #  CYRILLIC SMALL LETTER SOFT SIGN
    0x044d: 0x00ed,     #  CYRILLIC SMALL LETTER E
    0x044e: 0x00ee,     #  CYRILLIC SMALL LETTER YU
    0x044f: 0x00ef,     #  CYRILLIC SMALL LETTER YA
    0x0451: 0x00f1,     #  CYRILLIC SMALL LETTER IO
    0x0454: 0x00f5,     #  CYRILLIC SMALL LETTER UKRAINIAN IE
    0x0456: 0x00f7,     #  CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    0x0457: 0x00f9,     #  CYRILLIC SMALL LETTER YI
    0x0490: 0x00f2,     #  CYRILLIC CAPITAL LETTER GHE WITH UPTURN
    0x0491: 0x00f3,     #  CYRILLIC SMALL LETTER GHE WITH UPTURN
    0x2116: 0x00fc,     #  NUMERO SIGN
    0x221a: 0x00fb,     #  SQUARE ROOT
    0x2500: 0x00c4,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x2502: 0x00b3,     #  BOX DRAWINGS LIGHT VERTICAL
    0x250c: 0x00da,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x2510: 0x00bf,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x2514: 0x00c0,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x2518: 0x00d9,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x251c: 0x00c3,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x2524: 0x00b4,     #  BOX DRAWINGS LIGHT VERTICAL AND LEFT
    0x252c: 0x00c2,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x2534: 0x00c1,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x253c: 0x00c5,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x2550: 0x00cd,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x2551: 0x00ba,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x2552: 0x00d5,     #  BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    0x2553: 0x00d6,     #  BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    0x2554: 0x00c9,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x2555: 0x00b8,     #  BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    0x2556: 0x00b7,     #  BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    0x2557: 0x00bb,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x2558: 0x00d4,     #  BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    0x2559: 0x00d3,     #  BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    0x255a: 0x00c8,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x255b: 0x00be,     #  BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    0x255c: 0x00bd,     #  BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    0x255d: 0x00bc,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x255e: 0x00c6,     #  BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    0x255f: 0x00c7,     #  BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    0x2560: 0x00cc,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x2561: 0x00b5,     #  BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    0x2562: 0x00b6,     #  BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    0x2563: 0x00b9,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x2564: 0x00d1,     #  BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    0x2565: 0x00d2,     #  BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    0x2566: 0x00cb,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x2567: 0x00cf,     #  BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    0x2568: 0x00d0,     #  BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    0x2569: 0x00ca,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x256a: 0x00d8,     #  BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    0x256b: 0x00d7,     #  BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    0x256c: 0x00ce,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x2580: 0x00df,     #  UPPER HALF BLOCK
    0x2584: 0x00dc,     #  LOWER HALF BLOCK
    0x2588: 0x00db,     #  FULL BLOCK
    0x258c: 0x00dd,     #  LEFT HALF BLOCK
    0x2590: 0x00de,     #  RIGHT HALF BLOCK
    0x2591: 0x00b0,     #  LIGHT SHADE
    0x2592: 0x00b1,     #  MEDIUM SHADE
    0x2593: 0x00b2,     #  DARK SHADE
    0x25a0: 0x00fe,     #  BLACK SQUARE
}
<|endoftext|>""" Python Character Mapping Codec cp775 generated from 'VENDORS/MICSFT/PC/CP775.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_map)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_map)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp775',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
### Decoding Map

decoding_map = codecs.make_identity_dict(range(256))
decoding_map.update({
    0x0080: 0x0106,     #  LATIN CAPITAL LETTER C WITH ACUTE
    0x0081: 0x00fc,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x0082: 0x00e9,     #  LATIN SMALL LETTER E WITH ACUTE
    0x0083: 0x0101,     #  LATIN SMALL LETTER A WITH MACRON
    0x0084: 0x00e4,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x0085: 0x0123,     #  LATIN SMALL LETTER G WITH CEDILLA
    0x0086: 0x00e5,     #  LATIN SMALL LETTER A WITH RING ABOVE
    0x0087: 0x0107,     #  LATIN SMALL LETTER C WITH ACUTE
    0x0088: 0x0142,     #  LATIN SMALL LETTER L WITH STROKE
    0x0089: 0x0113,     #  LATIN SMALL LETTER E WITH MACRON
    0x008a: 0x0156,     #  LATIN CAPITAL LETTER R WITH CEDILLA
    0x008b: 0x0157,     #  LATIN SMALL LETTER R WITH CEDILLA
    0x008c: 0x012b,     #  LATIN SMALL LETTER I WITH MACRON
    0x008d: 0x0179,     #  LATIN CAPITAL LETTER Z WITH ACUTE
    0x008e: 0x00c4,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x008f: 0x00c5,     #  LATIN CAPITAL LETTER A WITH RING ABOVE
    0x0090: 0x00c9,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x0091: 0x00e6,     #  LATIN SMALL LIGATURE AE
    0x0092: 0x00c6,     #  LATIN CAPITAL LIGATURE AE
    0x0093: 0x014d,     #  LATIN SMALL LETTER O WITH MACRON
    0x0094: 0x00f6,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x0095: 0x0122,     #  LATIN CAPITAL LETTER G WITH CEDILLA
    0x0096: 0x00a2,     #  CENT SIGN
    0x0097: 0x015a,     #  LATIN CAPITAL LETTER S WITH ACUTE
    0x0098: 0x015b,     #  LATIN SMALL LETTER S WITH ACUTE
    0x0099: 0x00d6,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x009a: 0x00dc,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x009b: 0x00f8,     #  LATIN SMALL LETTER O WITH STROKE
    0x009c: 0x00a3,     #  POUND SIGN
    0x009d: 0x00d8,     #  LATIN CAPITAL LETTER O WITH STROKE
    0x009e: 0x00d7,     #  MULTIPLICATION SIGN
    0x009f: 0x00a4,     #  CURRENCY SIGN
    0x00a0: 0x0100,     #  LATIN CAPITAL LETTER A WITH MACRON
    0x00a1: 0x012a,     #  LATIN CAPITAL LETTER I WITH MACRON
    0x00a2: 0x00f3,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00a3: 0x017b,     #  LATIN CAPITAL LETTER Z WITH DOT ABOVE
    0x00a4: 0x017c,     #  LATIN SMALL LETTER Z WITH DOT ABOVE
    0x00a5: 0x017a,     #  LATIN SMALL LETTER Z WITH ACUTE
    0x00a6: 0x201d,     #  RIGHT DOUBLE QUOTATION MARK
    0x00a7: 0x00a6,     #  BROKEN BAR
    0x00a8: 0x00a9,     #  COPYRIGHT SIGN
    0x00a9: 0x00ae,     #  REGISTERED SIGN
    0x00aa: 0x00ac,     #  NOT SIGN
    0x00ab: 0x00bd,     #  VULGAR FRACTION ONE HALF
    0x00ac: 0x00bc,     #  VULGAR FRACTION ONE QUARTER
    0x00ad: 0x0141,     #  LATIN CAPITAL LETTER L WITH STROKE
    0x00ae: 0x00ab,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00af: 0x00bb,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00b0: 0x2591,     #  LIGHT SHADE
    0x00b1: 0x2592,     #  MEDIUM SHADE
    0x00b2: 0x2593,     #  DARK SHADE
    0x00b3: 0x2502,     #  BOX DRAWINGS LIGHT VERTICAL
    0x00b4: 0x2524,     #  BOX DRAWINGS LIGHT VERTICAL AND LEFT
    0x00b5: 0x0104,     #  LATIN CAPITAL LETTER A WITH OGONEK
    0x00b6: 0x010c,     #  LATIN CAPITAL LETTER C WITH CARON
    0x00b7: 0x0118,     #  LATIN CAPITAL LETTER E WITH OGONEK
    0x00b8: 0x0116,     #  LATIN CAPITAL LETTER E WITH DOT ABOVE
    0x00b9: 0x2563,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x00ba: 0x2551,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x00bb: 0x2557,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x00bc: 0x255d,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x00bd: 0x012e,     #  LATIN CAPITAL LETTER I WITH OGONEK
    0x00be: 0x0160,     #  LATIN CAPITAL LETTER S WITH CARON
    0x00bf: 0x2510,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x00c0: 0x2514,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x00c1: 0x2534,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x00c2: 0x252c,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x00c3: 0x251c,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x00c4: 0x2500,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x00c5: 0x253c,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x00c6: 0x0172,     #  LATIN CAPITAL LETTER U WITH OGONEK
    0x00c7: 0x016a,     #  LATIN CAPITAL LETTER U WITH MACRON
    0x00c8: 0x255a,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x00c9: 0x2554,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x00ca: 0x2569,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x00cb: 0x2566,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x00cc: 0x2560,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x00cd: 0x2550,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x00ce: 0x256c,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x00cf: 0x017d,     #  LATIN CAPITAL LETTER Z WITH CARON
    0x00d0: 0x0105,     #  LATIN SMALL LETTER A WITH OGONEK
    0x00d1: 0x010d,     #  LATIN SMALL LETTER C WITH CARON
    0x00d2: 0x0119,     #  LATIN SMALL LETTER E WITH OGONEK
    0x00d3: 0x0117,     #  LATIN SMALL LETTER E WITH DOT ABOVE
    0x00d4: 0x012f,     #  LATIN SMALL LETTER I WITH OGONEK
    0x00d5: 0x0161,     #  LATIN SMALL LETTER S WITH CARON
    0x00d6: 0x0173,     #  LATIN SMALL LETTER U WITH OGONEK
    0x00d7: 0x016b,     #  LATIN SMALL LETTER U WITH MACRON
    0x00d8: 0x017e,     #  LATIN SMALL LETTER Z WITH CARON
    0x00d9: 0x2518,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x00da: 0x250c,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x00db: 0x2588,     #  FULL BLOCK
    0x00dc: 0x2584,     #  LOWER HALF BLOCK
    0x00dd: 0x258c,     #  LEFT HALF BLOCK
    0x00de: 0x2590,     #  RIGHT HALF BLOCK
    0x00df: 0x2580,     #  UPPER HALF BLOCK
    0x00e0: 0x00d3,     #  LATIN CAPITAL LETTER O WITH ACUTE
    0x00e1: 0x00df,     #  LATIN SMALL LETTER SHARP S (GERMAN)
    0x00e2: 0x014c,     #  LATIN CAPITAL LETTER O WITH MACRON
    0x00e3: 0x0143,     #  LATIN CAPITAL LETTER N WITH ACUTE
    0x00e4: 0x00f5,     #  LATIN SMALL LETTER O WITH TILDE
    0x00e5: 0x00d5,     #  LATIN CAPITAL LETTER O WITH TILDE
    0x00e6: 0x00b5,     #  MICRO SIGN
    0x00e7: 0x0144,     #  LATIN SMALL LETTER N WITH ACUTE
    0x00e8: 0x0136,     #  LATIN CAPITAL LETTER K WITH CEDILLA
    0x00e9: 0x0137,     #  LATIN SMALL LETTER K WITH CEDILLA
    0x00ea: 0x013b,     #  LATIN CAPITAL LETTER L WITH CEDILLA
    0x00eb: 0x013c,     #  LATIN SMALL LETTER L WITH CEDILLA
    0x00ec: 0x0146,     #  LATIN SMALL LETTER N WITH CEDILLA
    0x00ed: 0x0112,     #  LATIN CAPITAL LETTER E WITH MACRON
    0x00ee: 0x0145,     #  LATIN CAPITAL LETTER N WITH CEDILLA
    0x00ef: 0x2019,     #  RIGHT SINGLE QUOTATION MARK
    0x00f0: 0x00ad,     #  SOFT HYPHEN
    0x00f1: 0x00b1,     #  PLUS-MINUS SIGN
    0x00f2: 0x201c,     #  LEFT DOUBLE QUOTATION MARK
    0x00f3: 0x00be,     #  VULGAR FRACTION THREE QUARTERS
    0x00f4: 0x00b6,     #  PILCROW SIGN
    0x00f5: 0x00a7,     #  SECTION SIGN
    0x00f6: 0x00f7,     #  DIVISION SIGN
    0x00f7: 0x201e,     #  DOUBLE LOW-9 QUOTATION MARK
    0x00f8: 0x00b0,     #  DEGREE SIGN
    0x00f9: 0x2219,     #  BULLET OPERATOR
    0x00fa: 0x00b7,     #  MIDDLE DOT
    0x00fb: 0x00b9,     #  SUPERSCRIPT ONE
    0x00fc: 0x00b3,     #  SUPERSCRIPT THREE
    0x00fd: 0x00b2,     #  SUPERSCRIPT TWO
    0x00fe: 0x25a0,     #  BLACK SQUARE
    0x00ff: 0x00a0,     #  NO-BREAK SPACE
})

### Decoding Table

decoding_table = (
    '\x00'     #  0x0000 -> NULL
    '\x01'     #  0x0001 -> START OF HEADING
    '\x02'     #  0x0002 -> START OF TEXT
    '\x03'     #  0x0003 -> END OF TEXT
    '\x04'     #  0x0004 -> END OF TRANSMISSION
    '\x05'     #  0x0005 -> ENQUIRY
    '\x06'     #  0x0006 -> ACKNOWLEDGE
    '\x07'     #  0x0007 -> BELL
    '\x08'     #  0x0008 -> BACKSPACE
    '\t'       #  0x0009 -> HORIZONTAL TABULATION
    '\n'       #  0x000a -> LINE FEED
    '\x0b'     #  0x000b -> VERTICAL TABULATION
    '\x0c'     #  0x000c -> FORM FEED
    '\r'       #  0x000d -> CARRIAGE RETURN
    '\x0e'     #  0x000e -> SHIFT OUT
    '\x0f'     #  0x000f -> SHIFT IN
    '\x10'     #  0x0010 -> DATA LINK ESCAPE
    '\x11'     #  0x0011 -> DEVICE CONTROL ONE
    '\x12'     #  0x0012 -> DEVICE CONTROL TWO
    '\x13'     #  0x0013 -> DEVICE CONTROL THREE
    '\x14'     #  0x0014 -> DEVICE CONTROL FOUR
    '\x15'     #  0x0015 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x0016 -> SYNCHRONOUS IDLE
    '\x17'     #  0x0017 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x0018 -> CANCEL
    '\x19'     #  0x0019 -> END OF MEDIUM
    '\x1a'     #  0x001a -> SUBSTITUTE
    '\x1b'     #  0x001b -> ESCAPE
    '\x1c'     #  0x001c -> FILE SEPARATOR
    '\x1d'     #  0x001d -> GROUP SEPARATOR
    '\x1e'     #  0x001e -> RECORD SEPARATOR
    '\x1f'     #  0x001f -> UNIT SEPARATOR
    ' '        #  0x0020 -> SPACE
    '!'        #  0x0021 -> EXCLAMATION MARK
    '"'        #  0x0022 -> QUOTATION MARK
    '#'        #  0x0023 -> NUMBER SIGN
    '$'        #  0x0024 -> DOLLAR SIGN
    '%'        #  0x0025 -> PERCENT SIGN
    '&'        #  0x0026 -> AMPERSAND
    "'"        #  0x0027 -> APOSTROPHE
    '('        #  0x0028 -> LEFT PARENTHESIS
    ')'        #  0x0029 -> RIGHT PARENTHESIS
    '*'        #  0x002a -> ASTERISK
    '+'        #  0x002b -> PLUS SIGN
    ','        #  0x002c -> COMMA
    '-'        #  0x002d -> HYPHEN-MINUS
    '.'        #  0x002e -> FULL STOP
    '/'        #  0x002f -> SOLIDUS
    '0'        #  0x0030 -> DIGIT ZERO
    '1'        #  0x0031 -> DIGIT ONE
    '2'        #  0x0032 -> DIGIT TWO
    '3'        #  0x0033 -> DIGIT THREE
    '4'        #  0x0034 -> DIGIT FOUR
    '5'        #  0x0035 -> DIGIT FIVE
    '6'        #  0x0036 -> DIGIT SIX
    '7'        #  0x0037 -> DIGIT SEVEN
    '8'        #  0x0038 -> DIGIT EIGHT
    '9'        #  0x0039 -> DIGIT NINE
    ':'        #  0x003a -> COLON
    ';'        #  0x003b -> SEMICOLON
    '<'        #  0x003c -> LESS-THAN SIGN
    '='        #  0x003d -> EQUALS SIGN
    '>'        #  0x003e -> GREATER-THAN SIGN
    '?'        #  0x003f -> QUESTION MARK
    '@'        #  0x0040 -> COMMERCIAL AT
    'A'        #  0x0041 -> LATIN CAPITAL LETTER A
    'B'        #  0x0042 -> LATIN CAPITAL LETTER B
    'C'        #  0x0043 -> LATIN CAPITAL LETTER C
    'D'        #  0x0044 -> LATIN CAPITAL LETTER D
    'E'        #  0x0045 -> LATIN CAPITAL LETTER E
    'F'        #  0x0046 -> LATIN CAPITAL LETTER F
    'G'        #  0x0047 -> LATIN CAPITAL LETTER G
    'H'        #  0x0048 -> LATIN CAPITAL LETTER H
    'I'        #  0x0049 -> LATIN CAPITAL LETTER I
    'J'        #  0x004a -> LATIN CAPITAL LETTER J
    'K'        #  0x004b -> LATIN CAPITAL LETTER K
    'L'        #  0x004c -> LATIN CAPITAL LETTER L
    'M'        #  0x004d -> LATIN CAPITAL LETTER M
    'N'        #  0x004e -> LATIN CAPITAL LETTER N
    'O'        #  0x004f -> LATIN CAPITAL LETTER O
    'P'        #  0x0050 -> LATIN CAPITAL LETTER P
    'Q'        #  0x0051 -> LATIN CAPITAL LETTER Q
    'R'        #  0x0052 -> LATIN CAPITAL LETTER R
    'S'        #  0x0053 -> LATIN CAPITAL LETTER S
    'T'        #  0x0054 -> LATIN CAPITAL LETTER T
    'U'        #  0x0055 -> LATIN CAPITAL LETTER U
    'V'        #  0x0056 -> LATIN CAPITAL LETTER V
    'W'        #  0x0057 -> LATIN CAPITAL LETTER W
    'X'        #  0x0058 -> LATIN CAPITAL LETTER X
    'Y'        #  0x0059 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x005a -> LATIN CAPITAL LETTER Z
    '['        #  0x005b -> LEFT SQUARE BRACKET
    '\\'       #  0x005c -> REVERSE SOLIDUS
    ']'        #  0x005d -> RIGHT SQUARE BRACKET
    '^'        #  0x005e -> CIRCUMFLEX ACCENT
    '_'        #  0x005f -> LOW LINE
    '`'        #  0x0060 -> GRAVE ACCENT
    'a'        #  0x0061 -> LATIN SMALL LETTER A
    'b'        #  0x0062 -> LATIN SMALL LETTER B
    'c'        #  0x0063 -> LATIN SMALL LETTER C
    'd'        #  0x0064 -> LATIN SMALL LETTER D
    'e'        #  0x0065 -> LATIN SMALL LETTER E
    'f'        #  0x0066 -> LATIN SMALL LETTER F
    'g'        #  0x0067 -> LATIN SMALL LETTER G
    'h'        #  0x0068 -> LATIN SMALL LETTER H
    'i'        #  0x0069 -> LATIN SMALL LETTER I
    'j'        #  0x006a -> LATIN SMALL LETTER J
    'k'        #  0x006b -> LATIN SMALL LETTER K
    'l'        #  0x006c -> LATIN SMALL LETTER L
    'm'        #  0x006d -> LATIN SMALL LETTER M
    'n'        #  0x006e -> LATIN SMALL LETTER N
    'o'        #  0x006f -> LATIN SMALL LETTER O
    'p'        #  0x0070 -> LATIN SMALL LETTER P
    'q'        #  0x0071 -> LATIN SMALL LETTER Q
    'r'        #  0x0072 -> LATIN SMALL LETTER R
    's'        #  0x0073 -> LATIN SMALL LETTER S
    't'        #  0x0074 -> LATIN SMALL LETTER T
    'u'        #  0x0075 -> LATIN SMALL LETTER U
    'v'        #  0x0076 -> LATIN SMALL LETTER V
    'w'        #  0x0077 -> LATIN SMALL LETTER W
    'x'        #  0x0078 -> LATIN SMALL LETTER X
    'y'        #  0x0079 -> LATIN SMALL LETTER Y
    'z'        #  0x007a -> LATIN SMALL LETTER Z
    '{'        #  0x007b -> LEFT CURLY BRACKET
    '|'        #  0x007c -> VERTICAL LINE
    '}'        #  0x007d -> RIGHT CURLY BRACKET
    '~'        #  0x007e -> TILDE
    '\x7f'     #  0x007f -> DELETE
    '\u0106'   #  0x0080 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xfc'     #  0x0081 -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xe9'     #  0x0082 -> LATIN SMALL LETTER E WITH ACUTE
    '\u0101'   #  0x0083 -> LATIN SMALL LETTER A WITH MACRON
    '\xe4'     #  0x0084 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u0123'   #  0x0085 -> LATIN SMALL LETTER G WITH CEDILLA
    '\xe5'     #  0x0086 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\u0107'   #  0x0087 -> LATIN SMALL LETTER C WITH ACUTE
    '\u0142'   #  0x0088 -> LATIN SMALL LETTER L WITH STROKE
    '\u0113'   #  0x0089 -> LATIN SMALL LETTER E WITH MACRON
    '\u0156'   #  0x008a -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\u0157'   #  0x008b -> LATIN SMALL LETTER R WITH CEDILLA
    '\u012b'   #  0x008c -> LATIN SMALL LETTER I WITH MACRON
    '\u0179'   #  0x008d -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\xc4'     #  0x008e -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0x008f -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc9'     #  0x0090 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xe6'     #  0x0091 -> LATIN SMALL LIGATURE AE
    '\xc6'     #  0x0092 -> LATIN CAPITAL LIGATURE AE
    '\u014d'   #  0x0093 -> LATIN SMALL LETTER O WITH MACRON
    '\xf6'     #  0x0094 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\u0122'   #  0x0095 -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\xa2'     #  0x0096 -> CENT SIGN
    '\u015a'   #  0x0097 -> LATIN CAPITAL LETTER S WITH ACUTE
    '\u015b'   #  0x0098 -> LATIN SMALL LETTER S WITH ACUTE
    '\xd6'     #  0x0099 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x009a -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xf8'     #  0x009b -> LATIN SMALL LETTER O WITH STROKE
    '\xa3'     #  0x009c -> POUND SIGN
    '\xd8'     #  0x009d -> LATIN CAPITAL LETTER O WITH STROKE
    '\xd7'     #  0x009e -> MULTIPLICATION SIGN
    '\xa4'     #  0x009f -> CURRENCY SIGN
    '\u0100'   #  0x00a0 -> LATIN CAPITAL LETTER A WITH MACRON
    '\u012a'   #  0x00a1 -> LATIN CAPITAL LETTER I WITH MACRON
    '\xf3'     #  0x00a2 -> LATIN SMALL LETTER O WITH ACUTE
    '\u017b'   #  0x00a3 -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\u017c'   #  0x00a4 -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u017a'   #  0x00a5 -> LATIN SMALL LETTER Z WITH ACUTE
    '\u201d'   #  0x00a6 -> RIGHT DOUBLE QUOTATION MARK
    '\xa6'     #  0x00a7 -> BROKEN BAR
    '\xa9'     #  0x00a8 -> COPYRIGHT SIGN
    '\xae'     #  0x00a9 -> REGISTERED SIGN
    '\xac'     #  0x00aa -> NOT SIGN
    '\xbd'     #  0x00ab -> VULGAR FRACTION ONE HALF
    '\xbc'     #  0x00ac -> VULGAR FRACTION ONE QUARTER
    '\u0141'   #  0x00ad -> LATIN CAPITAL LETTER L WITH STROKE
    '\xab'     #  0x00ae -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0x00af -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2591'   #  0x00b0 -> LIGHT SHADE
    '\u2592'   #  0x00b1 -> MEDIUM SHADE
    '\u2593'   #  0x00b2 -> DARK SHADE
    '\u2502'   #  0x00b3 -> BOX DRAWINGS LIGHT VERTICAL
    '\u2524'   #  0x00b4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u0104'   #  0x00b5 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\u010c'   #  0x00b6 -> LATIN CAPITAL LETTER C WITH CARON
    '\u0118'   #  0x00b7 -> LATIN CAPITAL LETTER E WITH OGONEK
    '\u0116'   #  0x00b8 -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\u2563'   #  0x00b9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'   #  0x00ba -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'   #  0x00bb -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'   #  0x00bc -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u012e'   #  0x00bd -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u0160'   #  0x00be -> LATIN CAPITAL LETTER S WITH CARON
    '\u2510'   #  0x00bf -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x00c0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'   #  0x00c1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'   #  0x00c2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0x00c3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'   #  0x00c4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'   #  0x00c5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u0172'   #  0x00c6 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\u016a'   #  0x00c7 -> LATIN CAPITAL LETTER U WITH MACRON
    '\u255a'   #  0x00c8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'   #  0x00c9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'   #  0x00ca -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'   #  0x00cb -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'   #  0x00cc -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'   #  0x00cd -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'   #  0x00ce -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\u017d'   #  0x00cf -> LATIN CAPITAL LETTER Z WITH CARON
    '\u0105'   #  0x00d0 -> LATIN SMALL LETTER A WITH OGONEK
    '\u010d'   #  0x00d1 -> LATIN SMALL LETTER C WITH CARON
    '\u0119'   #  0x00d2 -> LATIN SMALL LETTER E WITH OGONEK
    '\u0117'   #  0x00d3 -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\u012f'   #  0x00d4 -> LATIN SMALL LETTER I WITH OGONEK
    '\u0161'   #  0x00d5 -> LATIN SMALL LETTER S WITH CARON
    '\u0173'   #  0x00d6 -> LATIN SMALL LETTER U WITH OGONEK
    '\u016b'   #  0x00d7 -> LATIN SMALL LETTER U WITH MACRON
    '\u017e'   #  0x00d8 -> LATIN SMALL LETTER Z WITH CARON
    '\u2518'   #  0x00d9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'   #  0x00da -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'   #  0x00db -> FULL BLOCK
    '\u2584'   #  0x00dc -> LOWER HALF BLOCK
    '\u258c'   #  0x00dd -> LEFT HALF BLOCK
    '\u2590'   #  0x00de -> RIGHT HALF BLOCK
    '\u2580'   #  0x00df -> UPPER HALF BLOCK
    '\xd3'     #  0x00e0 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xdf'     #  0x00e1 -> LATIN SMALL LETTER SHARP S (GERMAN)
    '\u014c'   #  0x00e2 -> LATIN CAPITAL LETTER O WITH MACRON
    '\u0143'   #  0x00e3 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\xf5'     #  0x00e4 -> LATIN SMALL LETTER O WITH TILDE
    '\xd5'     #  0x00e5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xb5'     #  0x00e6 -> MICRO SIGN
    '\u0144'   #  0x00e7 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0136'   #  0x00e8 -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\u0137'   #  0x00e9 -> LATIN SMALL LETTER K WITH CEDILLA
    '\u013b'   #  0x00ea -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\u013c'   #  0x00eb -> LATIN SMALL LETTER L WITH CEDILLA
    '\u0146'   #  0x00ec -> LATIN SMALL LETTER N WITH CEDILLA
    '\u0112'   #  0x00ed -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0145'   #  0x00ee -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\u2019'   #  0x00ef -> RIGHT SINGLE QUOTATION MARK
    '\xad'     #  0x00f0 -> SOFT HYPHEN
    '\xb1'     #  0x00f1 -> PLUS-MINUS SIGN
    '\u201c'   #  0x00f2 -> LEFT DOUBLE QUOTATION MARK
    '\xbe'     #  0x00f3 -> VULGAR FRACTION THREE QUARTERS
    '\xb6'     #  0x00f4 -> PILCROW SIGN
    '\xa7'     #  0x00f5 -> SECTION SIGN
    '\xf7'     #  0x00f6 -> DIVISION SIGN
    '\u201e'   #  0x00f7 -> DOUBLE LOW-9 QUOTATION MARK
    '\xb0'     #  0x00f8 -> DEGREE SIGN
    '\u2219'   #  0x00f9 -> BULLET OPERATOR
    '\xb7'     #  0x00fa -> MIDDLE DOT
    '\xb9'     #  0x00fb -> SUPERSCRIPT ONE
    '\xb3'     #  0x00fc -> SUPERSCRIPT THREE
    '\xb2'     #  0x00fd -> SUPERSCRIPT TWO
    '\u25a0'   #  0x00fe -> BLACK SQUARE
    '\xa0'     #  0x00ff -> NO-BREAK SPACE
)

### Encoding Map

encoding_map = {
    0x0000: 0x0000,     #  NULL
    0x0001: 0x0001,     #  START OF HEADING
    0x0002: 0x0002,     #  START OF TEXT
    0x0003: 0x0003,     #  END OF TEXT
    0x0004: 0x0004,     #  END OF TRANSMISSION
    0x0005: 0x0005,     #  ENQUIRY
    0x0006: 0x0006,     #  ACKNOWLEDGE
    0x0007: 0x0007,     #  BELL
    0x0008: 0x0008,     #  BACKSPACE
    0x0009: 0x0009,     #  HORIZONTAL TABULATION
    0x000a: 0x000a,     #  LINE FEED
    0x000b: 0x000b,     #  VERTICAL TABULATION
    0x000c: 0x000c,     #  FORM FEED
    0x000d: 0x000d,     #  CARRIAGE RETURN
    0x000e: 0x000e,     #  SHIFT OUT
    0x000f: 0x000f,     #  SHIFT IN
    0x0010: 0x0010,     #  DATA LINK ESCAPE
    0x0011: 0x0011,     #  DEVICE CONTROL ONE
    0x0012: 0x0012,     #  DEVICE CONTROL TWO
    0x0013: 0x0013,     #  DEVICE CONTROL THREE
    0x0014: 0x0014,     #  DEVICE CONTROL FOUR
    0x0015: 0x0015,     #  NEGATIVE ACKNOWLEDGE
    0x0016: 0x0016,     #  SYNCHRONOUS IDLE
    0x0017: 0x0017,     #  END OF TRANSMISSION BLOCK
    0x0018: 0x0018,     #  CANCEL
    0x0019: 0x0019,     #  END OF MEDIUM
    0x001a: 0x001a,     #  SUBSTITUTE
    0x001b: 0x001b,     #  ESCAPE
    0x001c: 0x001c,     #  FILE SEPARATOR
    0x001d: 0x001d,     #  GROUP SEPARATOR
    0x001e: 0x001e,     #  RECORD SEPARATOR
    0x001f: 0x001f,     #  UNIT SEPARATOR
    0x0020: 0x0020,     #  SPACE
    0x0021: 0x0021,     #  EXCLAMATION MARK
    0x0022: 0x0022,     #  QUOTATION MARK
    0x0023: 0x0023,     #  NUMBER SIGN
    0x0024: 0x0024,     #  DOLLAR SIGN
    0x0025: 0x0025,     #  PERCENT SIGN
    0x0026: 0x0026,     #  AMPERSAND
    0x0027: 0x0027,     #  APOSTROPHE
    0x0028: 0x0028,     #  LEFT PARENTHESIS
    0x0029: 0x0029,     #  RIGHT PARENTHESIS
    0x002a: 0x002a,     #  ASTERISK
    0x002b: 0x002b,     #  PLUS SIGN
    0x002c: 0x002c,     #  COMMA
    0x002d: 0x002d,     #  HYPHEN-MINUS
    0x002e: 0x002e,     #  FULL STOP
    0x002f: 0x002f,     #  SOLIDUS
    0x0030: 0x0030,     #  DIGIT ZERO
    0x0031: 0x0031,     #  DIGIT ONE
    0x0032: 0x0032,     #  DIGIT TWO
    0x0033: 0x0033,     #  DIGIT THREE
    0x0034: 0x0034,     #  DIGIT FOUR
    0x0035: 0x0035,     #  DIGIT FIVE
    0x0036: 0x0036,     #  DIGIT SIX
    0x0037: 0x0037,     #  DIGIT SEVEN
    0x0038: 0x0038,     #  DIGIT EIGHT
    0x0039: 0x0039,     #  DIGIT NINE
    0x003a: 0x003a,     #  COLON
    0x003b: 0x003b,     #  SEMICOLON
    0x003c: 0x003c,     #  LESS-THAN SIGN
    0x003d: 0x003d,     #  EQUALS SIGN
    0x003e: 0x003e,     #  GREATER-THAN SIGN
    0x003f: 0x003f,     #  QUESTION MARK
    0x0040: 0x0040,     #  COMMERCIAL AT
    0x0041: 0x0041,     #  LATIN CAPITAL LETTER A
    0x0042: 0x0042,     #  LATIN CAPITAL LETTER B
    0x0043: 0x0043,     #  LATIN CAPITAL LETTER C
    0x0044: 0x0044,     #  LATIN CAPITAL LETTER D
    0x0045: 0x0045,     #  LATIN CAPITAL LETTER E
    0x0046: 0x0046,     #  LATIN CAPITAL LETTER F
    0x0047: 0x0047,     #  LATIN CAPITAL LETTER G
    0x0048: 0x0048,     #  LATIN CAPITAL LETTER H
    0x0049: 0x0049,     #  LATIN CAPITAL LETTER I
    0x004a: 0x004a,     #  LATIN CAPITAL LETTER J
    0x004b: 0x004b,     #  LATIN CAPITAL LETTER K
    0x004c: 0x004c,     #  LATIN CAPITAL LETTER L
    0x004d: 0x004d,     #  LATIN CAPITAL LETTER M
    0x004e: 0x004e,     #  LATIN CAPITAL LETTER N
    0x004f: 0x004f,     #  LATIN CAPITAL LETTER O
    0x0050: 0x0050,     #  LATIN CAPITAL LETTER P
    0x0051: 0x0051,     #  LATIN CAPITAL LETTER Q
    0x0052: 0x0052,     #  LATIN CAPITAL LETTER R
    0x0053: 0x0053,     #  LATIN CAPITAL LETTER S
    0x0054: 0x0054,     #  LATIN CAPITAL LETTER T
    0x0055: 0x0055,     #  LATIN CAPITAL LETTER U
    0x0056: 0x0056,     #  LATIN CAPITAL LETTER V
    0x0057: 0x0057,     #  LATIN CAPITAL LETTER W
    0x0058: 0x0058,     #  LATIN CAPITAL LETTER X
    0x0059: 0x0059,     #  LATIN CAPITAL LETTER Y
    0x005a: 0x005a,     #  LATIN CAPITAL LETTER Z
    0x005b: 0x005b,     #  LEFT SQUARE BRACKET
    0x005c: 0x005c,     #  REVERSE SOLIDUS
    0x005d: 0x005d,     #  RIGHT SQUARE BRACKET
    0x005e: 0x005e,     #  CIRCUMFLEX ACCENT
    0x005f: 0x005f,     #  LOW LINE
    0x0060: 0x0060,     #  GRAVE ACCENT
    0x0061: 0x0061,     #  LATIN SMALL LETTER A
    0x0062: 0x0062,     #  LATIN SMALL LETTER B
    0x0063: 0x0063,     #  LATIN SMALL LETTER C
    0x0064: 0x0064,     #  LATIN SMALL LETTER D
    0x0065: 0x0065,     #  LATIN SMALL LETTER E
    0x0066: 0x0066,     #  LATIN SMALL LETTER F
    0x0067: 0x0067,     #  LATIN SMALL LETTER G
    0x0068: 0x0068,     #  LATIN SMALL LETTER H
    0x0069: 0x0069,     #  LATIN SMALL LETTER I
    0x006a: 0x006a,     #  LATIN SMALL LETTER J
    0x006b: 0x006b,     #  LATIN SMALL LETTER K
    0x006c: 0x006c,     #  LATIN SMALL LETTER L
    0x006d: 0x006d,     #  LATIN SMALL LETTER M
    0x006e: 0x006e,     #  LATIN SMALL LETTER N
    0x006f: 0x006f,     #  LATIN SMALL LETTER O
    0x0070: 0x0070,     #  LATIN SMALL LETTER P
    0x0071: 0x0071,     #  LATIN SMALL LETTER Q
    0x0072: 0x0072,     #  LATIN SMALL LETTER R
    0x0073: 0x0073,     #  LATIN SMALL LETTER S
    0x0074: 0x0074,     #  LATIN SMALL LETTER T
    0x0075: 0x0075,     #  LATIN SMALL LETTER U
    0x0076: 0x0076,     #  LATIN SMALL LETTER V
    0x0077: 0x0077,     #  LATIN SMALL LETTER W
    0x0078: 0x0078,     #  LATIN SMALL LETTER X
    0x0079: 0x0079,     #  LATIN SMALL LETTER Y
    0x007a: 0x007a,     #  LATIN SMALL LETTER Z
    0x007b: 0x007b,     #  LEFT CURLY BRACKET
    0x007c: 0x007c,     #  VERTICAL LINE
    0x007d: 0x007d,     #  RIGHT CURLY BRACKET
    0x007e: 0x007e,     #  TILDE
    0x007f: 0x007f,     #  DELETE
    0x00a0: 0x00ff,     #  NO-BREAK SPACE
    0x00a2: 0x0096,     #  CENT SIGN
    0x00a3: 0x009c,     #  POUND SIGN
    0x00a4: 0x009f,     #  CURRENCY SIGN
    0x00a6: 0x00a7,     #  BROKEN BAR
    0x00a7: 0x00f5,     #  SECTION SIGN
    0x00a9: 0x00a8,     #  COPYRIGHT SIGN
    0x00ab: 0x00ae,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00ac: 0x00aa,     #  NOT SIGN
    0x00ad: 0x00f0,     #  SOFT HYPHEN
    0x00ae: 0x00a9,     #  REGISTERED SIGN
    0x00b0: 0x00f8,     #  DEGREE SIGN
    0x00b1: 0x00f1,     #  PLUS-MINUS SIGN
    0x00b2: 0x00fd,     #  SUPERSCRIPT TWO
    0x00b3: 0x00fc,     #  SUPERSCRIPT THREE
    0x00b5: 0x00e6,     #  MICRO SIGN
    0x00b6: 0x00f4,     #  PILCROW SIGN
    0x00b7: 0x00fa,     #  MIDDLE DOT
    0x00b9: 0x00fb,     #  SUPERSCRIPT ONE
    0x00bb: 0x00af,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00bc: 0x00ac,     #  VULGAR FRACTION ONE QUARTER
    0x00bd: 0x00ab,     #  VULGAR FRACTION ONE HALF
    0x00be: 0x00f3,     #  VULGAR FRACTION THREE QUARTERS
    0x00c4: 0x008e,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x00c5: 0x008f,     #  LATIN CAPITAL LETTER A WITH RING ABOVE
    0x00c6: 0x0092,     #  LATIN CAPITAL LIGATURE AE
    0x00c9: 0x0090,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x00d3: 0x00e0,     #  LATIN CAPITAL LETTER O WITH ACUTE
    0x00d5: 0x00e5,     #  LATIN CAPITAL LETTER O WITH TILDE
    0x00d6: 0x0099,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x00d7: 0x009e,     #  MULTIPLICATION SIGN
    0x00d8: 0x009d,     #  LATIN CAPITAL LETTER O WITH STROKE
    0x00dc: 0x009a,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x00df: 0x00e1,     #  LATIN SMALL LETTER SHARP S (GERMAN)
    0x00e4: 0x0084,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x00e5: 0x0086,     #  LATIN SMALL LETTER A WITH RING ABOVE
    0x00e6: 0x0091,     #  LATIN SMALL LIGATURE AE
    0x00e9: 0x0082,     #  LATIN SMALL LETTER E WITH ACUTE
    0x00f3: 0x00a2,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00f5: 0x00e4,     #  LATIN SMALL LETTER O WITH TILDE
    0x00f6: 0x0094,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x00f7: 0x00f6,     #  DIVISION SIGN
    0x00f8: 0x009b,     #  LATIN SMALL LETTER O WITH STROKE
    0x00fc: 0x0081,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x0100: 0x00a0,     #  LATIN CAPITAL LETTER A WITH MACRON
    0x0101: 0x0083,     #  LATIN SMALL LETTER A WITH MACRON
    0x0104: 0x00b5,     #  LATIN CAPITAL LETTER A WITH OGONEK
    0x0105: 0x00d0,     #  LATIN SMALL LETTER A WITH OGONEK
    0x0106: 0x0080,     #  LATIN CAPITAL LETTER C WITH ACUTE
    0x0107: 0x0087,     #  LATIN SMALL LETTER C WITH ACUTE
    0x010c: 0x00b6,     #  LATIN CAPITAL LETTER C WITH CARON
    0x010d: 0x00d1,     #  LATIN SMALL LETTER C WITH CARON
    0x0112: 0x00ed,     #  LATIN CAPITAL LETTER E WITH MACRON
    0x0113: 0x0089,     #  LATIN SMALL LETTER E WITH MACRON
    0x0116: 0x00b8,     #  LATIN CAPITAL LETTER E WITH DOT ABOVE
    0x0117: 0x00d3,     #  LATIN SMALL LETTER E WITH DOT ABOVE
    0x0118: 0x00b7,     #  LATIN CAPITAL LETTER E WITH OGONEK
    0x0119: 0x00d2,     #  LATIN SMALL LETTER E WITH OGONEK
    0x0122: 0x0095,     #  LATIN CAPITAL LETTER G WITH CEDILLA
    0x0123: 0x0085,     #  LATIN SMALL LETTER G WITH CEDILLA
    0x012a: 0x00a1,     #  LATIN CAPITAL LETTER I WITH MACRON
    0x012b: 0x008c,     #  LATIN SMALL LETTER I WITH MACRON
    0x012e: 0x00bd,     #  LATIN CAPITAL LETTER I WITH OGONEK
    0x012f: 0x00d4,     #  LATIN SMALL LETTER I WITH OGONEK
    0x0136: 0x00e8,     #  LATIN CAPITAL LETTER K WITH CEDILLA
    0x0137: 0x00e9,     #  LATIN SMALL LETTER K WITH CEDILLA
    0x013b: 0x00ea,     #  LATIN CAPITAL LETTER L WITH CEDILLA
    0x013c: 0x00eb,     #  LATIN SMALL LETTER L WITH CEDILLA
    0x0141: 0x00ad,     #  LATIN CAPITAL LETTER L WITH STROKE
    0x0142: 0x0088,     #  LATIN SMALL LETTER L WITH STROKE
    0x0143: 0x00e3,     #  LATIN CAPITAL LETTER N WITH ACUTE
    0x0144: 0x00e7,     #  LATIN SMALL LETTER N WITH ACUTE
    0x0145: 0x00ee,     #  LATIN CAPITAL LETTER N WITH CEDILLA
    0x0146: 0x00ec,     #  LATIN SMALL LETTER N WITH CEDILLA
    0x014c: 0x00e2,     #  LATIN CAPITAL LETTER O WITH MACRON
    0x014d: 0x0093,     #  LATIN SMALL LETTER O WITH MACRON
    0x0156: 0x008a,     #  LATIN CAPITAL LETTER R WITH CEDILLA
    0x0157: 0x008b,     #  LATIN SMALL LETTER R WITH CEDILLA
    0x015a: 0x0097,     #  LATIN CAPITAL LETTER S WITH ACUTE
    0x015b: 0x0098,     #  LATIN SMALL LETTER S WITH ACUTE
    0x0160: 0x00be,     #  LATIN CAPITAL LETTER S WITH CARON
    0x0161: 0x00d5,     #  LATIN SMALL LETTER S WITH CARON
    0x016a: 0x00c7,     #  LATIN CAPITAL LETTER U WITH MACRON
    0x016b: 0x00d7,     #  LATIN SMALL LETTER U WITH MACRON
    0x0172: 0x00c6,     #  LATIN CAPITAL LETTER U WITH OGONEK
    0x0173: 0x00d6,     #  LATIN SMALL LETTER U WITH OGONEK
    0x0179: 0x008d,     #  LATIN CAPITAL LETTER Z WITH ACUTE
    0x017a: 0x00a5,     #  LATIN SMALL LETTER Z WITH ACUTE
    0x017b: 0x00a3,     #  LATIN CAPITAL LETTER Z WITH DOT ABOVE
    0x017c: 0x00a4,     #  LATIN SMALL LETTER Z WITH DOT ABOVE
    0x017d: 0x00cf,     #  LATIN CAPITAL LETTER Z WITH CARON
    0x017e: 0x00d8,     #  LATIN SMALL LETTER Z WITH CARON
    0x2019: 0x00ef,     #  RIGHT SINGLE QUOTATION MARK
    0x201c: 0x00f2,     #  LEFT DOUBLE QUOTATION MARK
    0x201d: 0x00a6,     #  RIGHT DOUBLE QUOTATION MARK
    0x201e: 0x00f7,     #  DOUBLE LOW-9 QUOTATION MARK
    0x2219: 0x00f9,     #  BULLET OPERATOR
    0x2500: 0x00c4,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x2502: 0x00b3,     #  BOX DRAWINGS LIGHT VERTICAL
    0x250c: 0x00da,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x2510: 0x00bf,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x2514: 0x00c0,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x2518: 0x00d9,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x251c: 0x00c3,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x2524: 0x00b4,     #  BOX DRAWINGS LIGHT VERTICAL AND LEFT
    0x252c: 0x00c2,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x2534: 0x00c1,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x253c: 0x00c5,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x2550: 0x00cd,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x2551: 0x00ba,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x2554: 0x00c9,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x2557: 0x00bb,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x255a: 0x00c8,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x255d: 0x00bc,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x2560: 0x00cc,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x2563: 0x00b9,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x2566: 0x00cb,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x2569: 0x00ca,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x256c: 0x00ce,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x2580: 0x00df,     #  UPPER HALF BLOCK
    0x2584: 0x00dc,     #  LOWER HALF BLOCK
    0x2588: 0x00db,     #  FULL BLOCK
    0x258c: 0x00dd,     #  LEFT HALF BLOCK
    0x2590: 0x00de,     #  RIGHT HALF BLOCK
    0x2591: 0x00b0,     #  LIGHT SHADE
    0x2592: 0x00b1,     #  MEDIUM SHADE
    0x2593: 0x00b2,     #  DARK SHADE
    0x25a0: 0x00fe,     #  BLACK SQUARE
}
<|endoftext|>""" Python Character Mapping Codec koi8_t
"""
# http://ru.wikipedia.org/wiki/КОИ-8
# http://www.opensource.apple.com/source/libiconv/libiconv-4/libiconv/tests/KOI8-T.TXT

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='koi8-t',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\u049b'   #  0x80 -> CYRILLIC SMALL LETTER KA WITH DESCENDER
    '\u0493'   #  0x81 -> CYRILLIC SMALL LETTER GHE WITH STROKE
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0492'   #  0x83 -> CYRILLIC CAPITAL LETTER GHE WITH STROKE
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\ufffe'   #  0x88 -> UNDEFINED
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u04b3'   #  0x8A -> CYRILLIC SMALL LETTER HA WITH DESCENDER
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u04b2'   #  0x8C -> CYRILLIC CAPITAL LETTER HA WITH DESCENDER
    '\u04b7'   #  0x8D -> CYRILLIC SMALL LETTER CHE WITH DESCENDER
    '\u04b6'   #  0x8E -> CYRILLIC CAPITAL LETTER CHE WITH DESCENDER
    '\ufffe'   #  0x8F -> UNDEFINED
    '\u049a'   #  0x90 -> CYRILLIC CAPITAL LETTER KA WITH DESCENDER
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\ufffe'   #  0xA0 -> UNDEFINED
    '\u04ef'   #  0xA1 -> CYRILLIC SMALL LETTER U WITH MACRON
    '\u04ee'   #  0xA2 -> CYRILLIC CAPITAL LETTER U WITH MACRON
    '\u0451'   #  0xA3 -> CYRILLIC SMALL LETTER IO
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u04e3'   #  0xA5 -> CYRILLIC SMALL LETTER I WITH MACRON
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\ufffe'   #  0xA8 -> UNDEFINED
    '\ufffe'   #  0xA9 -> UNDEFINED
    '\ufffe'   #  0xAA -> UNDEFINED
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\ufffe'   #  0xAF -> UNDEFINED
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\u0401'   #  0xB3 -> CYRILLIC CAPITAL LETTER IO
    '\ufffe'   #  0xB4 -> UNDEFINED
    '\u04e2'   #  0xB5 -> CYRILLIC CAPITAL LETTER I WITH MACRON
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\ufffe'   #  0xB8 -> UNDEFINED
    '\u2116'   #  0xB9 -> NUMERO SIGN
    '\ufffe'   #  0xBA -> UNDEFINED
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\ufffe'   #  0xBC -> UNDEFINED
    '\ufffe'   #  0xBD -> UNDEFINED
    '\ufffe'   #  0xBE -> UNDEFINED
    '\xa9'     #  0xBF -> COPYRIGHT SIGN
    '\u044e'   #  0xC0 -> CYRILLIC SMALL LETTER YU
    '\u0430'   #  0xC1 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xC2 -> CYRILLIC SMALL LETTER BE
    '\u0446'   #  0xC3 -> CYRILLIC SMALL LETTER TSE
    '\u0434'   #  0xC4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xC5 -> CYRILLIC SMALL LETTER IE
    '\u0444'   #  0xC6 -> CYRILLIC SMALL LETTER EF
    '\u0433'   #  0xC7 -> CYRILLIC SMALL LETTER GHE
    '\u0445'   #  0xC8 -> CYRILLIC SMALL LETTER HA
    '\u0438'   #  0xC9 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xCA -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xCB -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xCC -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xCD -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xCE -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xCF -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xD0 -> CYRILLIC SMALL LETTER PE
    '\u044f'   #  0xD1 -> CYRILLIC SMALL LETTER YA
    '\u0440'   #  0xD2 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xD3 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xD4 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xD5 -> CYRILLIC SMALL LETTER U
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0432'   #  0xD7 -> CYRILLIC SMALL LETTER VE
    '\u044c'   #  0xD8 -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044b'   #  0xD9 -> CYRILLIC SMALL LETTER YERU
    '\u0437'   #  0xDA -> CYRILLIC SMALL LETTER ZE
    '\u0448'   #  0xDB -> CYRILLIC SMALL LETTER SHA
    '\u044d'   #  0xDC -> CYRILLIC SMALL LETTER E
    '\u0449'   #  0xDD -> CYRILLIC SMALL LETTER SHCHA
    '\u0447'   #  0xDE -> CYRILLIC SMALL LETTER CHE
    '\u044a'   #  0xDF -> CYRILLIC SMALL LETTER HARD SIGN
    '\u042e'   #  0xE0 -> CYRILLIC CAPITAL LETTER YU
    '\u0410'   #  0xE1 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xE2 -> CYRILLIC CAPITAL LETTER BE
    '\u0426'   #  0xE3 -> CYRILLIC CAPITAL LETTER TSE
    '\u0414'   #  0xE4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xE5 -> CYRILLIC CAPITAL LETTER IE
    '\u0424'   #  0xE6 -> CYRILLIC CAPITAL LETTER EF
    '\u0413'   #  0xE7 -> CYRILLIC CAPITAL LETTER GHE
    '\u0425'   #  0xE8 -> CYRILLIC CAPITAL LETTER HA
    '\u0418'   #  0xE9 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xEA -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xEB -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xEC -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xED -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xEE -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xEF -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xF0 -> CYRILLIC CAPITAL LETTER PE
    '\u042f'   #  0xF1 -> CYRILLIC CAPITAL LETTER YA
    '\u0420'   #  0xF2 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xF3 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xF4 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xF5 -> CYRILLIC CAPITAL LETTER U
    '\u0416'   #  0xF6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0412'   #  0xF7 -> CYRILLIC CAPITAL LETTER VE
    '\u042c'   #  0xF8 -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042b'   #  0xF9 -> CYRILLIC CAPITAL LETTER YERU
    '\u0417'   #  0xFA -> CYRILLIC CAPITAL LETTER ZE
    '\u0428'   #  0xFB -> CYRILLIC CAPITAL LETTER SHA
    '\u042d'   #  0xFC -> CYRILLIC CAPITAL LETTER E
    '\u0429'   #  0xFD -> CYRILLIC CAPITAL LETTER SHCHA
    '\u0427'   #  0xFE -> CYRILLIC CAPITAL LETTER CHE
    '\u042a'   #  0xFF -> CYRILLIC CAPITAL LETTER HARD SIGN
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python 'undefined' Codec

    This codec will always raise a ValueError exception when being
    used. It is intended for use by the site.py file to switch off
    automatic string to Unicode coercion.

Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        raise UnicodeError("undefined encoding")

    def decode(self,input,errors='strict'):
        raise UnicodeError("undefined encoding")

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        raise UnicodeError("undefined encoding")

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        raise UnicodeError("undefined encoding")

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='undefined',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )
<|endoftext|>""" Python Character Mapping Codec cp1140 generated from 'python-mappings/CP1140.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp1140',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x9c'     #  0x04 -> CONTROL
    '\t'       #  0x05 -> HORIZONTAL TABULATION
    '\x86'     #  0x06 -> CONTROL
    '\x7f'     #  0x07 -> DELETE
    '\x97'     #  0x08 -> CONTROL
    '\x8d'     #  0x09 -> CONTROL
    '\x8e'     #  0x0A -> CONTROL
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x9d'     #  0x14 -> CONTROL
    '\x85'     #  0x15 -> CONTROL
    '\x08'     #  0x16 -> BACKSPACE
    '\x87'     #  0x17 -> CONTROL
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x92'     #  0x1A -> CONTROL
    '\x8f'     #  0x1B -> CONTROL
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    '\x80'     #  0x20 -> CONTROL
    '\x81'     #  0x21 -> CONTROL
    '\x82'     #  0x22 -> CONTROL
    '\x83'     #  0x23 -> CONTROL
    '\x84'     #  0x24 -> CONTROL
    '\n'       #  0x25 -> LINE FEED
    '\x17'     #  0x26 -> END OF TRANSMISSION BLOCK
    '\x1b'     #  0x27 -> ESCAPE
    '\x88'     #  0x28 -> CONTROL
    '\x89'     #  0x29 -> CONTROL
    '\x8a'     #  0x2A -> CONTROL
    '\x8b'     #  0x2B -> CONTROL
    '\x8c'     #  0x2C -> CONTROL
    '\x05'     #  0x2D -> ENQUIRY
    '\x06'     #  0x2E -> ACKNOWLEDGE
    '\x07'     #  0x2F -> BELL
    '\x90'     #  0x30 -> CONTROL
    '\x91'     #  0x31 -> CONTROL
    '\x16'     #  0x32 -> SYNCHRONOUS IDLE
    '\x93'     #  0x33 -> CONTROL
    '\x94'     #  0x34 -> CONTROL
    '\x95'     #  0x35 -> CONTROL
    '\x96'     #  0x36 -> CONTROL
    '\x04'     #  0x37 -> END OF TRANSMISSION
    '\x98'     #  0x38 -> CONTROL
    '\x99'     #  0x39 -> CONTROL
    '\x9a'     #  0x3A -> CONTROL
    '\x9b'     #  0x3B -> CONTROL
    '\x14'     #  0x3C -> DEVICE CONTROL FOUR
    '\x15'     #  0x3D -> NEGATIVE ACKNOWLEDGE
    '\x9e'     #  0x3E -> CONTROL
    '\x1a'     #  0x3F -> SUBSTITUTE
    ' '        #  0x40 -> SPACE
    '\xa0'     #  0x41 -> NO-BREAK SPACE
    '\xe2'     #  0x42 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x43 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe0'     #  0x44 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0x45 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe3'     #  0x46 -> LATIN SMALL LETTER A WITH TILDE
    '\xe5'     #  0x47 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe7'     #  0x48 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xf1'     #  0x49 -> LATIN SMALL LETTER N WITH TILDE
    '\xa2'     #  0x4A -> CENT SIGN
    '.'        #  0x4B -> FULL STOP
    '<'        #  0x4C -> LESS-THAN SIGN
    '('        #  0x4D -> LEFT PARENTHESIS
    '+'        #  0x4E -> PLUS SIGN
    '|'        #  0x4F -> VERTICAL LINE
    '&'        #  0x50 -> AMPERSAND
    '\xe9'     #  0x51 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0x52 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x53 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xe8'     #  0x54 -> LATIN SMALL LETTER E WITH GRAVE
    '\xed'     #  0x55 -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0x56 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x57 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xec'     #  0x58 -> LATIN SMALL LETTER I WITH GRAVE
    '\xdf'     #  0x59 -> LATIN SMALL LETTER SHARP S (GERMAN)
    '!'        #  0x5A -> EXCLAMATION MARK
    '$'        #  0x5B -> DOLLAR SIGN
    '*'        #  0x5C -> ASTERISK
    ')'        #  0x5D -> RIGHT PARENTHESIS
    ';'        #  0x5E -> SEMICOLON
    '\xac'     #  0x5F -> NOT SIGN
    '-'        #  0x60 -> HYPHEN-MINUS
    '/'        #  0x61 -> SOLIDUS
    '\xc2'     #  0x62 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc4'     #  0x63 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc0'     #  0x64 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0x65 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc3'     #  0x66 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc5'     #  0x67 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc7'     #  0x68 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xd1'     #  0x69 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xa6'     #  0x6A -> BROKEN BAR
    ','        #  0x6B -> COMMA
    '%'        #  0x6C -> PERCENT SIGN
    '_'        #  0x6D -> LOW LINE
    '>'        #  0x6E -> GREATER-THAN SIGN
    '?'        #  0x6F -> QUESTION MARK
    '\xf8'     #  0x70 -> LATIN SMALL LETTER O WITH STROKE
    '\xc9'     #  0x71 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0x72 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0x73 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xc8'     #  0x74 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xcd'     #  0x75 -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0x76 -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0x77 -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xcc'     #  0x78 -> LATIN CAPITAL LETTER I WITH GRAVE
    '`'        #  0x79 -> GRAVE ACCENT
    ':'        #  0x7A -> COLON
    '#'        #  0x7B -> NUMBER SIGN
    '@'        #  0x7C -> COMMERCIAL AT
    "'"        #  0x7D -> APOSTROPHE
    '='        #  0x7E -> EQUALS SIGN
    '"'        #  0x7F -> QUOTATION MARK
    '\xd8'     #  0x80 -> LATIN CAPITAL LETTER O WITH STROKE
    'a'        #  0x81 -> LATIN SMALL LETTER A
    'b'        #  0x82 -> LATIN SMALL LETTER B
    'c'        #  0x83 -> LATIN SMALL LETTER C
    'd'        #  0x84 -> LATIN SMALL LETTER D
    'e'        #  0x85 -> LATIN SMALL LETTER E
    'f'        #  0x86 -> LATIN SMALL LETTER F
    'g'        #  0x87 -> LATIN SMALL LETTER G
    'h'        #  0x88 -> LATIN SMALL LETTER H
    'i'        #  0x89 -> LATIN SMALL LETTER I
    '\xab'     #  0x8A -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0x8B -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xf0'     #  0x8C -> LATIN SMALL LETTER ETH (ICELANDIC)
    '\xfd'     #  0x8D -> LATIN SMALL LETTER Y WITH ACUTE
    '\xfe'     #  0x8E -> LATIN SMALL LETTER THORN (ICELANDIC)
    '\xb1'     #  0x8F -> PLUS-MINUS SIGN
    '\xb0'     #  0x90 -> DEGREE SIGN
    'j'        #  0x91 -> LATIN SMALL LETTER J
    'k'        #  0x92 -> LATIN SMALL LETTER K
    'l'        #  0x93 -> LATIN SMALL LETTER L
    'm'        #  0x94 -> LATIN SMALL LETTER M
    'n'        #  0x95 -> LATIN SMALL LETTER N
    'o'        #  0x96 -> LATIN SMALL LETTER O
    'p'        #  0x97 -> LATIN SMALL LETTER P
    'q'        #  0x98 -> LATIN SMALL LETTER Q
    'r'        #  0x99 -> LATIN SMALL LETTER R
    '\xaa'     #  0x9A -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0x9B -> MASCULINE ORDINAL INDICATOR
    '\xe6'     #  0x9C -> LATIN SMALL LIGATURE AE
    '\xb8'     #  0x9D -> CEDILLA
    '\xc6'     #  0x9E -> LATIN CAPITAL LIGATURE AE
    '\u20ac'   #  0x9F -> EURO SIGN
    '\xb5'     #  0xA0 -> MICRO SIGN
    '~'        #  0xA1 -> TILDE
    's'        #  0xA2 -> LATIN SMALL LETTER S
    't'        #  0xA3 -> LATIN SMALL LETTER T
    'u'        #  0xA4 -> LATIN SMALL LETTER U
    'v'        #  0xA5 -> LATIN SMALL LETTER V
    'w'        #  0xA6 -> LATIN SMALL LETTER W
    'x'        #  0xA7 -> LATIN SMALL LETTER X
    'y'        #  0xA8 -> LATIN SMALL LETTER Y
    'z'        #  0xA9 -> LATIN SMALL LETTER Z
    '\xa1'     #  0xAA -> INVERTED EXCLAMATION MARK
    '\xbf'     #  0xAB -> INVERTED QUESTION MARK
    '\xd0'     #  0xAC -> LATIN CAPITAL LETTER ETH (ICELANDIC)
    '\xdd'     #  0xAD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xde'     #  0xAE -> LATIN CAPITAL LETTER THORN (ICELANDIC)
    '\xae'     #  0xAF -> REGISTERED SIGN
    '^'        #  0xB0 -> CIRCUMFLEX ACCENT
    '\xa3'     #  0xB1 -> POUND SIGN
    '\xa5'     #  0xB2 -> YEN SIGN
    '\xb7'     #  0xB3 -> MIDDLE DOT
    '\xa9'     #  0xB4 -> COPYRIGHT SIGN
    '\xa7'     #  0xB5 -> SECTION SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xbc'     #  0xB7 -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xB8 -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xB9 -> VULGAR FRACTION THREE QUARTERS
    '['        #  0xBA -> LEFT SQUARE BRACKET
    ']'        #  0xBB -> RIGHT SQUARE BRACKET
    '\xaf'     #  0xBC -> MACRON
    '\xa8'     #  0xBD -> DIAERESIS
    '\xb4'     #  0xBE -> ACUTE ACCENT
    '\xd7'     #  0xBF -> MULTIPLICATION SIGN
    '{'        #  0xC0 -> LEFT CURLY BRACKET
    'A'        #  0xC1 -> LATIN CAPITAL LETTER A
    'B'        #  0xC2 -> LATIN CAPITAL LETTER B
    'C'        #  0xC3 -> LATIN CAPITAL LETTER C
    'D'        #  0xC4 -> LATIN CAPITAL LETTER D
    'E'        #  0xC5 -> LATIN CAPITAL LETTER E
    'F'        #  0xC6 -> LATIN CAPITAL LETTER F
    'G'        #  0xC7 -> LATIN CAPITAL LETTER G
    'H'        #  0xC8 -> LATIN CAPITAL LETTER H
    'I'        #  0xC9 -> LATIN CAPITAL LETTER I
    '\xad'     #  0xCA -> SOFT HYPHEN
    '\xf4'     #  0xCB -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0xCC -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf2'     #  0xCD -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xCE -> LATIN SMALL LETTER O WITH ACUTE
    '\xf5'     #  0xCF -> LATIN SMALL LETTER O WITH TILDE
    '}'        #  0xD0 -> RIGHT CURLY BRACKET
    'J'        #  0xD1 -> LATIN CAPITAL LETTER J
    'K'        #  0xD2 -> LATIN CAPITAL LETTER K
    'L'        #  0xD3 -> LATIN CAPITAL LETTER L
    'M'        #  0xD4 -> LATIN CAPITAL LETTER M
    'N'        #  0xD5 -> LATIN CAPITAL LETTER N
    'O'        #  0xD6 -> LATIN CAPITAL LETTER O
    'P'        #  0xD7 -> LATIN CAPITAL LETTER P
    'Q'        #  0xD8 -> LATIN CAPITAL LETTER Q
    'R'        #  0xD9 -> LATIN CAPITAL LETTER R
    '\xb9'     #  0xDA -> SUPERSCRIPT ONE
    '\xfb'     #  0xDB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xDC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xf9'     #  0xDD -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xDE -> LATIN SMALL LETTER U WITH ACUTE
    '\xff'     #  0xDF -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\\'       #  0xE0 -> REVERSE SOLIDUS
    '\xf7'     #  0xE1 -> DIVISION SIGN
    'S'        #  0xE2 -> LATIN CAPITAL LETTER S
    'T'        #  0xE3 -> LATIN CAPITAL LETTER T
    'U'        #  0xE4 -> LATIN CAPITAL LETTER U
    'V'        #  0xE5 -> LATIN CAPITAL LETTER V
    'W'        #  0xE6 -> LATIN CAPITAL LETTER W
    'X'        #  0xE7 -> LATIN CAPITAL LETTER X
    'Y'        #  0xE8 -> LATIN CAPITAL LETTER Y
    'Z'        #  0xE9 -> LATIN CAPITAL LETTER Z
    '\xb2'     #  0xEA -> SUPERSCRIPT TWO
    '\xd4'     #  0xEB -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd6'     #  0xEC -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd2'     #  0xED -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd5'     #  0xEF -> LATIN CAPITAL LETTER O WITH TILDE
    '0'        #  0xF0 -> DIGIT ZERO
    '1'        #  0xF1 -> DIGIT ONE
    '2'        #  0xF2 -> DIGIT TWO
    '3'        #  0xF3 -> DIGIT THREE
    '4'        #  0xF4 -> DIGIT FOUR
    '5'        #  0xF5 -> DIGIT FIVE
    '6'        #  0xF6 -> DIGIT SIX
    '7'        #  0xF7 -> DIGIT SEVEN
    '8'        #  0xF8 -> DIGIT EIGHT
    '9'        #  0xF9 -> DIGIT NINE
    '\xb3'     #  0xFA -> SUPERSCRIPT THREE
    '\xdb'     #  0xFB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xFC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xd9'     #  0xFD -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xFE -> LATIN CAPITAL LETTER U WITH ACUTE
    '\x9f'     #  0xFF -> CONTROL
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>#
# euc_jp.py: Python Unicode Codec for EUC_JP
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_jp, codecs
import _multibytecodec as mbc

codec = _codecs_jp.getcodec('euc_jp')

class Codec(codecs.Codec):
    encode = codec.encode
    decode = codec.decode

class IncrementalEncoder(mbc.MultibyteIncrementalEncoder,
                         codecs.IncrementalEncoder):
    codec = codec

class IncrementalDecoder(mbc.MultibyteIncrementalDecoder,
                         codecs.IncrementalDecoder):
    codec = codec

class StreamReader(Codec, mbc.MultibyteStreamReader, codecs.StreamReader):
    codec = codec

class StreamWriter(Codec, mbc.MultibyteStreamWriter, codecs.StreamWriter):
    codec = codec

def getregentry():
    return codecs.CodecInfo(
        name='euc_jp',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec iso8859_4 generated from 'MAPPINGS/ISO8859/8859-4.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='iso8859-4',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\x80'     #  0x80 -> <control>
    '\x81'     #  0x81 -> <control>
    '\x82'     #  0x82 -> <control>
    '\x83'     #  0x83 -> <control>
    '\x84'     #  0x84 -> <control>
    '\x85'     #  0x85 -> <control>
    '\x86'     #  0x86 -> <control>
    '\x87'     #  0x87 -> <control>
    '\x88'     #  0x88 -> <control>
    '\x89'     #  0x89 -> <control>
    '\x8a'     #  0x8A -> <control>
    '\x8b'     #  0x8B -> <control>
    '\x8c'     #  0x8C -> <control>
    '\x8d'     #  0x8D -> <control>
    '\x8e'     #  0x8E -> <control>
    '\x8f'     #  0x8F -> <control>
    '\x90'     #  0x90 -> <control>
    '\x91'     #  0x91 -> <control>
    '\x92'     #  0x92 -> <control>
    '\x93'     #  0x93 -> <control>
    '\x94'     #  0x94 -> <control>
    '\x95'     #  0x95 -> <control>
    '\x96'     #  0x96 -> <control>
    '\x97'     #  0x97 -> <control>
    '\x98'     #  0x98 -> <control>
    '\x99'     #  0x99 -> <control>
    '\x9a'     #  0x9A -> <control>
    '\x9b'     #  0x9B -> <control>
    '\x9c'     #  0x9C -> <control>
    '\x9d'     #  0x9D -> <control>
    '\x9e'     #  0x9E -> <control>
    '\x9f'     #  0x9F -> <control>
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0104'   #  0xA1 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\u0138'   #  0xA2 -> LATIN SMALL LETTER KRA
    '\u0156'   #  0xA3 -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u0128'   #  0xA5 -> LATIN CAPITAL LETTER I WITH TILDE
    '\u013b'   #  0xA6 -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    '\u0112'   #  0xAA -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0122'   #  0xAB -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u0166'   #  0xAC -> LATIN CAPITAL LETTER T WITH STROKE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    '\xaf'     #  0xAF -> MACRON
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0105'   #  0xB1 -> LATIN SMALL LETTER A WITH OGONEK
    '\u02db'   #  0xB2 -> OGONEK
    '\u0157'   #  0xB3 -> LATIN SMALL LETTER R WITH CEDILLA
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\u0129'   #  0xB5 -> LATIN SMALL LETTER I WITH TILDE
    '\u013c'   #  0xB6 -> LATIN SMALL LETTER L WITH CEDILLA
    '\u02c7'   #  0xB7 -> CARON
    '\xb8'     #  0xB8 -> CEDILLA
    '\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    '\u0113'   #  0xBA -> LATIN SMALL LETTER E WITH MACRON
    '\u0123'   #  0xBB -> LATIN SMALL LETTER G WITH CEDILLA
    '\u0167'   #  0xBC -> LATIN SMALL LETTER T WITH STROKE
    '\u014a'   #  0xBD -> LATIN CAPITAL LETTER ENG
    '\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    '\u014b'   #  0xBF -> LATIN SMALL LETTER ENG
    '\u0100'   #  0xC0 -> LATIN CAPITAL LETTER A WITH MACRON
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc3'     #  0xC3 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\u012e'   #  0xC7 -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0118'   #  0xCA -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u0116'   #  0xCC -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\u012a'   #  0xCF -> LATIN CAPITAL LETTER I WITH MACRON
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0145'   #  0xD1 -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\u014c'   #  0xD2 -> LATIN CAPITAL LETTER O WITH MACRON
    '\u0136'   #  0xD3 -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\u0172'   #  0xD9 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u0168'   #  0xDD -> LATIN CAPITAL LETTER U WITH TILDE
    '\u016a'   #  0xDE -> LATIN CAPITAL LETTER U WITH MACRON
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\u0101'   #  0xE0 -> LATIN SMALL LETTER A WITH MACRON
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe3'     #  0xE3 -> LATIN SMALL LETTER A WITH TILDE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\u012f'   #  0xE7 -> LATIN SMALL LETTER I WITH OGONEK
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u0119'   #  0xEA -> LATIN SMALL LETTER E WITH OGONEK
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u0117'   #  0xEC -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\u012b'   #  0xEF -> LATIN SMALL LETTER I WITH MACRON
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\u0146'   #  0xF1 -> LATIN SMALL LETTER N WITH CEDILLA
    '\u014d'   #  0xF2 -> LATIN SMALL LETTER O WITH MACRON
    '\u0137'   #  0xF3 -> LATIN SMALL LETTER K WITH CEDILLA
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\u0173'   #  0xF9 -> LATIN SMALL LETTER U WITH OGONEK
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u0169'   #  0xFD -> LATIN SMALL LETTER U WITH TILDE
    '\u016b'   #  0xFE -> LATIN SMALL LETTER U WITH MACRON
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>#
# euc_kr.py: Python Unicode Codec for EUC_KR
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_kr, codecs
import _multibytecodec as mbc

codec = _codecs_kr.getcodec('euc_kr')

class Codec(codecs.Codec):
    encode = codec.encode
    decode = codec.decode

class IncrementalEncoder(mbc.MultibyteIncrementalEncoder,
                         codecs.IncrementalEncoder):
    codec = codec

class IncrementalDecoder(mbc.MultibyteIncrementalDecoder,
                         codecs.IncrementalDecoder):
    codec = codec

class StreamReader(Codec, mbc.MultibyteStreamReader, codecs.StreamReader):
    codec = codec

class StreamWriter(Codec, mbc.MultibyteStreamWriter, codecs.StreamWriter):
    codec = codec

def getregentry():
    return codecs.CodecInfo(
        name='euc_kr',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec iso8859_14 generated from 'MAPPINGS/ISO8859/8859-14.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='iso8859-14',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\x80'     #  0x80 -> <control>
    '\x81'     #  0x81 -> <control>
    '\x82'     #  0x82 -> <control>
    '\x83'     #  0x83 -> <control>
    '\x84'     #  0x84 -> <control>
    '\x85'     #  0x85 -> <control>
    '\x86'     #  0x86 -> <control>
    '\x87'     #  0x87 -> <control>
    '\x88'     #  0x88 -> <control>
    '\x89'     #  0x89 -> <control>
    '\x8a'     #  0x8A -> <control>
    '\x8b'     #  0x8B -> <control>
    '\x8c'     #  0x8C -> <control>
    '\x8d'     #  0x8D -> <control>
    '\x8e'     #  0x8E -> <control>
    '\x8f'     #  0x8F -> <control>
    '\x90'     #  0x90 -> <control>
    '\x91'     #  0x91 -> <control>
    '\x92'     #  0x92 -> <control>
    '\x93'     #  0x93 -> <control>
    '\x94'     #  0x94 -> <control>
    '\x95'     #  0x95 -> <control>
    '\x96'     #  0x96 -> <control>
    '\x97'     #  0x97 -> <control>
    '\x98'     #  0x98 -> <control>
    '\x99'     #  0x99 -> <control>
    '\x9a'     #  0x9A -> <control>
    '\x9b'     #  0x9B -> <control>
    '\x9c'     #  0x9C -> <control>
    '\x9d'     #  0x9D -> <control>
    '\x9e'     #  0x9E -> <control>
    '\x9f'     #  0x9F -> <control>
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u1e02'   #  0xA1 -> LATIN CAPITAL LETTER B WITH DOT ABOVE
    '\u1e03'   #  0xA2 -> LATIN SMALL LETTER B WITH DOT ABOVE
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u010a'   #  0xA4 -> LATIN CAPITAL LETTER C WITH DOT ABOVE
    '\u010b'   #  0xA5 -> LATIN SMALL LETTER C WITH DOT ABOVE
    '\u1e0a'   #  0xA6 -> LATIN CAPITAL LETTER D WITH DOT ABOVE
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u1e80'   #  0xA8 -> LATIN CAPITAL LETTER W WITH GRAVE
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u1e82'   #  0xAA -> LATIN CAPITAL LETTER W WITH ACUTE
    '\u1e0b'   #  0xAB -> LATIN SMALL LETTER D WITH DOT ABOVE
    '\u1ef2'   #  0xAC -> LATIN CAPITAL LETTER Y WITH GRAVE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u0178'   #  0xAF -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\u1e1e'   #  0xB0 -> LATIN CAPITAL LETTER F WITH DOT ABOVE
    '\u1e1f'   #  0xB1 -> LATIN SMALL LETTER F WITH DOT ABOVE
    '\u0120'   #  0xB2 -> LATIN CAPITAL LETTER G WITH DOT ABOVE
    '\u0121'   #  0xB3 -> LATIN SMALL LETTER G WITH DOT ABOVE
    '\u1e40'   #  0xB4 -> LATIN CAPITAL LETTER M WITH DOT ABOVE
    '\u1e41'   #  0xB5 -> LATIN SMALL LETTER M WITH DOT ABOVE
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\u1e56'   #  0xB7 -> LATIN CAPITAL LETTER P WITH DOT ABOVE
    '\u1e81'   #  0xB8 -> LATIN SMALL LETTER W WITH GRAVE
    '\u1e57'   #  0xB9 -> LATIN SMALL LETTER P WITH DOT ABOVE
    '\u1e83'   #  0xBA -> LATIN SMALL LETTER W WITH ACUTE
    '\u1e60'   #  0xBB -> LATIN CAPITAL LETTER S WITH DOT ABOVE
    '\u1ef3'   #  0xBC -> LATIN SMALL LETTER Y WITH GRAVE
    '\u1e84'   #  0xBD -> LATIN CAPITAL LETTER W WITH DIAERESIS
    '\u1e85'   #  0xBE -> LATIN SMALL LETTER W WITH DIAERESIS
    '\u1e61'   #  0xBF -> LATIN SMALL LETTER S WITH DOT ABOVE
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc3'     #  0xC3 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc8'     #  0xC8 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0xCA -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xcc'     #  0xCC -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\u0174'   #  0xD0 -> LATIN CAPITAL LETTER W WITH CIRCUMFLEX
    '\xd1'     #  0xD1 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd2'     #  0xD2 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\u1e6a'   #  0xD7 -> LATIN CAPITAL LETTER T WITH DOT ABOVE
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\u0176'   #  0xDE -> LATIN CAPITAL LETTER Y WITH CIRCUMFLEX
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe3'     #  0xE3 -> LATIN SMALL LETTER A WITH TILDE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xec'     #  0xEC -> LATIN SMALL LETTER I WITH GRAVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u0175'   #  0xF0 -> LATIN SMALL LETTER W WITH CIRCUMFLEX
    '\xf1'     #  0xF1 -> LATIN SMALL LETTER N WITH TILDE
    '\xf2'     #  0xF2 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\u1e6b'   #  0xF7 -> LATIN SMALL LETTER T WITH DOT ABOVE
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\u0177'   #  0xFE -> LATIN SMALL LETTER Y WITH CIRCUMFLEX
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec cp1252 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1252.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp1252',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\u20ac'   #  0x80 -> EURO SIGN
    '\ufffe'   #  0x81 -> UNDEFINED
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\u02c6'   #  0x88 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u0160'   #  0x8A -> LATIN CAPITAL LETTER S WITH CARON
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u0152'   #  0x8C -> LATIN CAPITAL LIGATURE OE
    '\ufffe'   #  0x8D -> UNDEFINED
    '\u017d'   #  0x8E -> LATIN CAPITAL LETTER Z WITH CARON
    '\ufffe'   #  0x8F -> UNDEFINED
    '\ufffe'   #  0x90 -> UNDEFINED
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\u02dc'   #  0x98 -> SMALL TILDE
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\u0161'   #  0x9A -> LATIN SMALL LETTER S WITH CARON
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\u0153'   #  0x9C -> LATIN SMALL LIGATURE OE
    '\ufffe'   #  0x9D -> UNDEFINED
    '\u017e'   #  0x9E -> LATIN SMALL LETTER Z WITH CARON
    '\u0178'   #  0x9F -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\xa1'     #  0xA1 -> INVERTED EXCLAMATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\xaa'     #  0xAA -> FEMININE ORDINAL INDICATOR
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\xaf'     #  0xAF -> MACRON
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\xb8'     #  0xB8 -> CEDILLA
    '\xb9'     #  0xB9 -> SUPERSCRIPT ONE
    '\xba'     #  0xBA -> MASCULINE ORDINAL INDICATOR
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xbf'     #  0xBF -> INVERTED QUESTION MARK
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc3'     #  0xC3 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc8'     #  0xC8 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0xCA -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xcc'     #  0xCC -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xd0'     #  0xD0 -> LATIN CAPITAL LETTER ETH
    '\xd1'     #  0xD1 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd2'     #  0xD2 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xde'     #  0xDE -> LATIN CAPITAL LETTER THORN
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe3'     #  0xE3 -> LATIN SMALL LETTER A WITH TILDE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xec'     #  0xEC -> LATIN SMALL LETTER I WITH GRAVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf0'     #  0xF0 -> LATIN SMALL LETTER ETH
    '\xf1'     #  0xF1 -> LATIN SMALL LETTER N WITH TILDE
    '\xf2'     #  0xF2 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\xfe'     #  0xFE -> LATIN SMALL LETTER THORN
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

