    '\u03af'   #  0xDF -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03b0'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    '\u03b3'   #  0xE3 -> GREEK SMALL LETTER GAMMA
    '\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    '\u03b6'   #  0xE6 -> GREEK SMALL LETTER ZETA
    '\u03b7'   #  0xE7 -> GREEK SMALL LETTER ETA
    '\u03b8'   #  0xE8 -> GREEK SMALL LETTER THETA
    '\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    '\u03ba'   #  0xEA -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0xEB -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0xEC -> GREEK SMALL LETTER MU
    '\u03bd'   #  0xED -> GREEK SMALL LETTER NU
    '\u03be'   #  0xEE -> GREEK SMALL LETTER XI
    '\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    '\u03c1'   #  0xF1 -> GREEK SMALL LETTER RHO
    '\u03c2'   #  0xF2 -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    '\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    '\u03c5'   #  0xF5 -> GREEK SMALL LETTER UPSILON
    '\u03c6'   #  0xF6 -> GREEK SMALL LETTER PHI
    '\u03c7'   #  0xF7 -> GREEK SMALL LETTER CHI
    '\u03c8'   #  0xF8 -> GREEK SMALL LETTER PSI
    '\u03c9'   #  0xF9 -> GREEK SMALL LETTER OMEGA
    '\u03ca'   #  0xFA -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03cb'   #  0xFB -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u03cc'   #  0xFC -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u03cd'   #  0xFD -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03ce'   #  0xFE -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\ufffe'
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec cp1255 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1255.TXT' with gencodec.py.

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
        name='cp1255',
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
    '\ufffe'   #  0x8A -> UNDEFINED
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
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
    '\u02dc'   #  0x98 -> SMALL TILDE
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\xa1'     #  0xA1 -> INVERTED EXCLAMATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u20aa'   #  0xA4 -> NEW SHEQEL SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\xd7'     #  0xAA -> MULTIPLICATION SIGN
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
    '\xf7'     #  0xBA -> DIVISION SIGN
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xbf'     #  0xBF -> INVERTED QUESTION MARK
    '\u05b0'   #  0xC0 -> HEBREW POINT SHEVA
    '\u05b1'   #  0xC1 -> HEBREW POINT HATAF SEGOL
    '\u05b2'   #  0xC2 -> HEBREW POINT HATAF PATAH
    '\u05b3'   #  0xC3 -> HEBREW POINT HATAF QAMATS
    '\u05b4'   #  0xC4 -> HEBREW POINT HIRIQ
    '\u05b5'   #  0xC5 -> HEBREW POINT TSERE
    '\u05b6'   #  0xC6 -> HEBREW POINT SEGOL
    '\u05b7'   #  0xC7 -> HEBREW POINT PATAH
    '\u05b8'   #  0xC8 -> HEBREW POINT QAMATS
    '\u05b9'   #  0xC9 -> HEBREW POINT HOLAM
    '\ufffe'   #  0xCA -> UNDEFINED
    '\u05bb'   #  0xCB -> HEBREW POINT QUBUTS
    '\u05bc'   #  0xCC -> HEBREW POINT DAGESH OR MAPIQ
    '\u05bd'   #  0xCD -> HEBREW POINT METEG
    '\u05be'   #  0xCE -> HEBREW PUNCTUATION MAQAF
    '\u05bf'   #  0xCF -> HEBREW POINT RAFE
    '\u05c0'   #  0xD0 -> HEBREW PUNCTUATION PASEQ
    '\u05c1'   #  0xD1 -> HEBREW POINT SHIN DOT
    '\u05c2'   #  0xD2 -> HEBREW POINT SIN DOT
    '\u05c3'   #  0xD3 -> HEBREW PUNCTUATION SOF PASUQ
    '\u05f0'   #  0xD4 -> HEBREW LIGATURE YIDDISH DOUBLE VAV
    '\u05f1'   #  0xD5 -> HEBREW LIGATURE YIDDISH VAV YOD
    '\u05f2'   #  0xD6 -> HEBREW LIGATURE YIDDISH DOUBLE YOD
    '\u05f3'   #  0xD7 -> HEBREW PUNCTUATION GERESH
    '\u05f4'   #  0xD8 -> HEBREW PUNCTUATION GERSHAYIM
    '\ufffe'   #  0xD9 -> UNDEFINED
    '\ufffe'   #  0xDA -> UNDEFINED
    '\ufffe'   #  0xDB -> UNDEFINED
    '\ufffe'   #  0xDC -> UNDEFINED
    '\ufffe'   #  0xDD -> UNDEFINED
    '\ufffe'   #  0xDE -> UNDEFINED
    '\ufffe'   #  0xDF -> UNDEFINED
    '\u05d0'   #  0xE0 -> HEBREW LETTER ALEF
    '\u05d1'   #  0xE1 -> HEBREW LETTER BET
    '\u05d2'   #  0xE2 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0xE3 -> HEBREW LETTER DALET
    '\u05d4'   #  0xE4 -> HEBREW LETTER HE
    '\u05d5'   #  0xE5 -> HEBREW LETTER VAV
    '\u05d6'   #  0xE6 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0xE7 -> HEBREW LETTER HET
    '\u05d8'   #  0xE8 -> HEBREW LETTER TET
    '\u05d9'   #  0xE9 -> HEBREW LETTER YOD
    '\u05da'   #  0xEA -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0xEB -> HEBREW LETTER KAF
    '\u05dc'   #  0xEC -> HEBREW LETTER LAMED
    '\u05dd'   #  0xED -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0xEE -> HEBREW LETTER MEM
    '\u05df'   #  0xEF -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0xF0 -> HEBREW LETTER NUN
    '\u05e1'   #  0xF1 -> HEBREW LETTER SAMEKH
    '\u05e2'   #  0xF2 -> HEBREW LETTER AYIN
    '\u05e3'   #  0xF3 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0xF4 -> HEBREW LETTER PE
    '\u05e5'   #  0xF5 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0xF6 -> HEBREW LETTER TSADI
    '\u05e7'   #  0xF7 -> HEBREW LETTER QOF
    '\u05e8'   #  0xF8 -> HEBREW LETTER RESH
    '\u05e9'   #  0xF9 -> HEBREW LETTER SHIN
    '\u05ea'   #  0xFA -> HEBREW LETTER TAV
    '\ufffe'   #  0xFB -> UNDEFINED
    '\ufffe'   #  0xFC -> UNDEFINED
    '\u200e'   #  0xFD -> LEFT-TO-RIGHT MARK
    '\u200f'   #  0xFE -> RIGHT-TO-LEFT MARK
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec cp856 generated from 'MAPPINGS/VENDORS/MISC/CP856.TXT' with gencodec.py.

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
        name='cp856',
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
    '\u05d0'   #  0x80 -> HEBREW LETTER ALEF
    '\u05d1'   #  0x81 -> HEBREW LETTER BET
    '\u05d2'   #  0x82 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0x83 -> HEBREW LETTER DALET
    '\u05d4'   #  0x84 -> HEBREW LETTER HE
    '\u05d5'   #  0x85 -> HEBREW LETTER VAV
    '\u05d6'   #  0x86 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0x87 -> HEBREW LETTER HET
    '\u05d8'   #  0x88 -> HEBREW LETTER TET
    '\u05d9'   #  0x89 -> HEBREW LETTER YOD
    '\u05da'   #  0x8A -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0x8B -> HEBREW LETTER KAF
    '\u05dc'   #  0x8C -> HEBREW LETTER LAMED
    '\u05dd'   #  0x8D -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0x8E -> HEBREW LETTER MEM
    '\u05df'   #  0x8F -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0x90 -> HEBREW LETTER NUN
    '\u05e1'   #  0x91 -> HEBREW LETTER SAMEKH
    '\u05e2'   #  0x92 -> HEBREW LETTER AYIN
    '\u05e3'   #  0x93 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0x94 -> HEBREW LETTER PE
    '\u05e5'   #  0x95 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0x96 -> HEBREW LETTER TSADI
    '\u05e7'   #  0x97 -> HEBREW LETTER QOF
    '\u05e8'   #  0x98 -> HEBREW LETTER RESH
    '\u05e9'   #  0x99 -> HEBREW LETTER SHIN
    '\u05ea'   #  0x9A -> HEBREW LETTER TAV
    '\ufffe'   #  0x9B -> UNDEFINED
    '\xa3'     #  0x9C -> POUND SIGN
    '\ufffe'   #  0x9D -> UNDEFINED
    '\xd7'     #  0x9E -> MULTIPLICATION SIGN
    '\ufffe'   #  0x9F -> UNDEFINED
    '\ufffe'   #  0xA0 -> UNDEFINED
    '\ufffe'   #  0xA1 -> UNDEFINED
    '\ufffe'   #  0xA2 -> UNDEFINED
    '\ufffe'   #  0xA3 -> UNDEFINED
    '\ufffe'   #  0xA4 -> UNDEFINED
    '\ufffe'   #  0xA5 -> UNDEFINED
    '\ufffe'   #  0xA6 -> UNDEFINED
    '\ufffe'   #  0xA7 -> UNDEFINED
    '\ufffe'   #  0xA8 -> UNDEFINED
    '\xae'     #  0xA9 -> REGISTERED SIGN
    '\xac'     #  0xAA -> NOT SIGN
    '\xbd'     #  0xAB -> VULGAR FRACTION ONE HALF
    '\xbc'     #  0xAC -> VULGAR FRACTION ONE QUARTER
    '\ufffe'   #  0xAD -> UNDEFINED
    '\xab'     #  0xAE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xAF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2591'   #  0xB0 -> LIGHT SHADE
    '\u2592'   #  0xB1 -> MEDIUM SHADE
    '\u2593'   #  0xB2 -> DARK SHADE
    '\u2502'   #  0xB3 -> BOX DRAWINGS LIGHT VERTICAL
    '\u2524'   #  0xB4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\ufffe'   #  0xB5 -> UNDEFINED
    '\ufffe'   #  0xB6 -> UNDEFINED
    '\ufffe'   #  0xB7 -> UNDEFINED
    '\xa9'     #  0xB8 -> COPYRIGHT SIGN
    '\u2563'   #  0xB9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'   #  0xBA -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'   #  0xBB -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'   #  0xBC -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\xa2'     #  0xBD -> CENT SIGN
    '\xa5'     #  0xBE -> YEN SIGN
    '\u2510'   #  0xBF -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0xC0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'   #  0xC1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'   #  0xC2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0xC3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'   #  0xC4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'   #  0xC5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\ufffe'   #  0xC6 -> UNDEFINED
    '\ufffe'   #  0xC7 -> UNDEFINED
    '\u255a'   #  0xC8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'   #  0xC9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'   #  0xCA -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'   #  0xCB -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'   #  0xCC -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'   #  0xCD -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'   #  0xCE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\xa4'     #  0xCF -> CURRENCY SIGN
    '\ufffe'   #  0xD0 -> UNDEFINED
    '\ufffe'   #  0xD1 -> UNDEFINED
    '\ufffe'   #  0xD2 -> UNDEFINED
    '\ufffe'   #  0xD3 -> UNDEFINEDS
    '\ufffe'   #  0xD4 -> UNDEFINED
    '\ufffe'   #  0xD5 -> UNDEFINED
    '\ufffe'   #  0xD6 -> UNDEFINEDE
    '\ufffe'   #  0xD7 -> UNDEFINED
    '\ufffe'   #  0xD8 -> UNDEFINED
    '\u2518'   #  0xD9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'   #  0xDA -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'   #  0xDB -> FULL BLOCK
    '\u2584'   #  0xDC -> LOWER HALF BLOCK
    '\xa6'     #  0xDD -> BROKEN BAR
    '\ufffe'   #  0xDE -> UNDEFINED
    '\u2580'   #  0xDF -> UPPER HALF BLOCK
    '\ufffe'   #  0xE0 -> UNDEFINED
    '\ufffe'   #  0xE1 -> UNDEFINED
    '\ufffe'   #  0xE2 -> UNDEFINED
    '\ufffe'   #  0xE3 -> UNDEFINED
    '\ufffe'   #  0xE4 -> UNDEFINED
    '\ufffe'   #  0xE5 -> UNDEFINED
    '\xb5'     #  0xE6 -> MICRO SIGN
    '\ufffe'   #  0xE7 -> UNDEFINED
    '\ufffe'   #  0xE8 -> UNDEFINED
    '\ufffe'   #  0xE9 -> UNDEFINED
    '\ufffe'   #  0xEA -> UNDEFINED
    '\ufffe'   #  0xEB -> UNDEFINED
    '\ufffe'   #  0xEC -> UNDEFINED
    '\ufffe'   #  0xED -> UNDEFINED
    '\xaf'     #  0xEE -> MACRON
    '\xb4'     #  0xEF -> ACUTE ACCENT
    '\xad'     #  0xF0 -> SOFT HYPHEN
    '\xb1'     #  0xF1 -> PLUS-MINUS SIGN
    '\u2017'   #  0xF2 -> DOUBLE LOW LINE
    '\xbe'     #  0xF3 -> VULGAR FRACTION THREE QUARTERS
    '\xb6'     #  0xF4 -> PILCROW SIGN
    '\xa7'     #  0xF5 -> SECTION SIGN
    '\xf7'     #  0xF6 -> DIVISION SIGN
    '\xb8'     #  0xF7 -> CEDILLA
    '\xb0'     #  0xF8 -> DEGREE SIGN
    '\xa8'     #  0xF9 -> DIAERESIS
    '\xb7'     #  0xFA -> MIDDLE DOT
    '\xb9'     #  0xFB -> SUPERSCRIPT ONE
    '\xb3'     #  0xFC -> SUPERSCRIPT THREE
    '\xb2'     #  0xFD -> SUPERSCRIPT TWO
    '\u25a0'   #  0xFE -> BLACK SQUARE
    '\xa0'     #  0xFF -> NO-BREAK SPACE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec iso8859_15 generated from 'MAPPINGS/ISO8859/8859-15.TXT' with gencodec.py.

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
        name='iso8859-15',
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
    '\xa1'     #  0xA1 -> INVERTED EXCLAMATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u20ac'   #  0xA4 -> EURO SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\u0160'   #  0xA6 -> LATIN CAPITAL LETTER S WITH CARON
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u0161'   #  0xA8 -> LATIN SMALL LETTER S WITH CARON
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
    '\u017d'   #  0xB4 -> LATIN CAPITAL LETTER Z WITH CARON
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u017e'   #  0xB8 -> LATIN SMALL LETTER Z WITH CARON
    '\xb9'     #  0xB9 -> SUPERSCRIPT ONE
    '\xba'     #  0xBA -> MASCULINE ORDINAL INDICATOR
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u0152'   #  0xBC -> LATIN CAPITAL LIGATURE OE
    '\u0153'   #  0xBD -> LATIN SMALL LIGATURE OE
    '\u0178'   #  0xBE -> LATIN CAPITAL LETTER Y WITH DIAERESIS
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

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python 'utf-8-sig' Codec
This work similar to UTF-8 with the following changes:

* On encoding/writing a UTF-8 encoded BOM will be prepended/written as the
  first three bytes.

* On decoding/reading if the first three bytes are a UTF-8 encoded BOM, these
  bytes will be skipped.
"""
import codecs

### Codec APIs

def encode(input, errors='strict'):
    return (codecs.BOM_UTF8 + codecs.utf_8_encode(input, errors)[0],
            len(input))

def decode(input, errors='strict'):
    prefix = 0
    if input[:3] == codecs.BOM_UTF8:
        input = input[3:]
        prefix = 3
    (output, consumed) = codecs.utf_8_decode(input, errors, True)
    return (output, consumed+prefix)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors='strict'):
        codecs.IncrementalEncoder.__init__(self, errors)
        self.first = 1

    def encode(self, input, final=False):
        if self.first:
            self.first = 0
            return codecs.BOM_UTF8 + \
                   codecs.utf_8_encode(input, self.errors)[0]
        else:
            return codecs.utf_8_encode(input, self.errors)[0]

    def reset(self):
        codecs.IncrementalEncoder.reset(self)
        self.first = 1

    def getstate(self):
        return self.first

    def setstate(self, state):
        self.first = state

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def __init__(self, errors='strict'):
        codecs.BufferedIncrementalDecoder.__init__(self, errors)
        self.first = 1

    def _buffer_decode(self, input, errors, final):
        if self.first:
            if len(input) < 3:
                if codecs.BOM_UTF8.startswith(input):
                    # not enough data to decide if this really is a BOM
                    # => try again on the next call
                    return ("", 0)
                else:
                    self.first = 0
            else:
                self.first = 0
                if input[:3] == codecs.BOM_UTF8:
                    (output, consumed) = \
                       codecs.utf_8_decode(input[3:], errors, final)
                    return (output, consumed+3)
        return codecs.utf_8_decode(input, errors, final)

    def reset(self):
        codecs.BufferedIncrementalDecoder.reset(self)
        self.first = 1

    def getstate(self):
        state = codecs.BufferedIncrementalDecoder.getstate(self)
        # state[1] must be 0 here, as it isn't passed along to the caller
        return (state[0], self.first)

    def setstate(self, state):
        # state[1] will be ignored by BufferedIncrementalDecoder.setstate()
        codecs.BufferedIncrementalDecoder.setstate(self, state)
        self.first = state[1]

class StreamWriter(codecs.StreamWriter):
    def reset(self):
        codecs.StreamWriter.reset(self)
        try:
            del self.encode
        except AttributeError:
            pass

    def encode(self, input, errors='strict'):
        self.encode = codecs.utf_8_encode
        return encode(input, errors)

class StreamReader(codecs.StreamReader):
    def reset(self):
        codecs.StreamReader.reset(self)
        try:
            del self.decode
        except AttributeError:
            pass

    def decode(self, input, errors='strict'):
        if len(input) < 3:
            if codecs.BOM_UTF8.startswith(input):
                # not enough data to decide if this is a BOM
                # => try again on the next call
                return ("", 0)
        elif input[:3] == codecs.BOM_UTF8:
            self.decode = codecs.utf_8_decode
            (output, consumed) = codecs.utf_8_decode(input[3:],errors)
            return (output, consumed+3)
        # (else) no BOM present
        self.decode = codecs.utf_8_decode
        return codecs.utf_8_decode(input, errors)

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='utf-8-sig',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>#
# euc_jisx0213.py: Python Unicode Codec for EUC_JISX0213
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_jp, codecs
import _multibytecodec as mbc

codec = _codecs_jp.getcodec('euc_jisx0213')

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
        name='euc_jisx0213',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec cp1256 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1256.TXT' with gencodec.py.

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
        name='cp1256',
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
    '\u067e'   #  0x81 -> ARABIC LETTER PEH
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\u02c6'   #  0x88 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u0679'   #  0x8A -> ARABIC LETTER TTEH
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u0152'   #  0x8C -> LATIN CAPITAL LIGATURE OE
    '\u0686'   #  0x8D -> ARABIC LETTER TCHEH
    '\u0698'   #  0x8E -> ARABIC LETTER JEH
    '\u0688'   #  0x8F -> ARABIC LETTER DDAL
    '\u06af'   #  0x90 -> ARABIC LETTER GAF
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\u06a9'   #  0x98 -> ARABIC LETTER KEHEH
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\u0691'   #  0x9A -> ARABIC LETTER RREH
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\u0153'   #  0x9C -> LATIN SMALL LIGATURE OE
    '\u200c'   #  0x9D -> ZERO WIDTH NON-JOINER
    '\u200d'   #  0x9E -> ZERO WIDTH JOINER
    '\u06ba'   #  0x9F -> ARABIC LETTER NOON GHUNNA
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u060c'   #  0xA1 -> ARABIC COMMA
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u06be'   #  0xAA -> ARABIC LETTER HEH DOACHASHMEE
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
    '\u061b'   #  0xBA -> ARABIC SEMICOLON
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\u061f'   #  0xBF -> ARABIC QUESTION MARK
    '\u06c1'   #  0xC0 -> ARABIC LETTER HEH GOAL
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
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0637'   #  0xD8 -> ARABIC LETTER TAH
    '\u0638'   #  0xD9 -> ARABIC LETTER ZAH
    '\u0639'   #  0xDA -> ARABIC LETTER AIN
    '\u063a'   #  0xDB -> ARABIC LETTER GHAIN
    '\u0640'   #  0xDC -> ARABIC TATWEEL
    '\u0641'   #  0xDD -> ARABIC LETTER FEH
    '\u0642'   #  0xDE -> ARABIC LETTER QAF
    '\u0643'   #  0xDF -> ARABIC LETTER KAF
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\u0644'   #  0xE1 -> ARABIC LETTER LAM
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0645'   #  0xE3 -> ARABIC LETTER MEEM
    '\u0646'   #  0xE4 -> ARABIC LETTER NOON
    '\u0647'   #  0xE5 -> ARABIC LETTER HEH
    '\u0648'   #  0xE6 -> ARABIC LETTER WAW
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u0649'   #  0xEC -> ARABIC LETTER ALEF MAKSURA
    '\u064a'   #  0xED -> ARABIC LETTER YEH
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u064b'   #  0xF0 -> ARABIC FATHATAN
    '\u064c'   #  0xF1 -> ARABIC DAMMATAN
    '\u064d'   #  0xF2 -> ARABIC KASRATAN
    '\u064e'   #  0xF3 -> ARABIC FATHA
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u064f'   #  0xF5 -> ARABIC DAMMA
    '\u0650'   #  0xF6 -> ARABIC KASRA
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0651'   #  0xF8 -> ARABIC SHADDA
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\u0652'   #  0xFA -> ARABIC SUKUN
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u200e'   #  0xFD -> LEFT-TO-RIGHT MARK
    '\u200f'   #  0xFE -> RIGHT-TO-LEFT MARK
    '\u06d2'   #  0xFF -> ARABIC LETTER YEH BARREE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python 'mbcs' Codec for Windows


Cloned by Mark Hammond (mhammond@skippinet.com.au) from ascii.py,
which was written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""
# Import them explicitly to cause an ImportError
# on non-Windows systems
from codecs import mbcs_encode, mbcs_decode
# for IncrementalDecoder, IncrementalEncoder, ...
import codecs

### Codec APIs

encode = mbcs_encode

def decode(input, errors='strict'):
    return mbcs_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return mbcs_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = mbcs_decode

class StreamWriter(codecs.StreamWriter):
    encode = mbcs_encode

class StreamReader(codecs.StreamReader):
    decode = mbcs_decode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='mbcs',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec generated from 'VENDORS/MICSFT/PC/CP866.TXT' with gencodec.py.

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
        name='cp866',
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
    0x00f2: 0x0404,     #  CYRILLIC CAPITAL LETTER UKRAINIAN IE
    0x00f3: 0x0454,     #  CYRILLIC SMALL LETTER UKRAINIAN IE
    0x00f4: 0x0407,     #  CYRILLIC CAPITAL LETTER YI
    0x00f5: 0x0457,     #  CYRILLIC SMALL LETTER YI
    0x00f6: 0x040e,     #  CYRILLIC CAPITAL LETTER SHORT U
    0x00f7: 0x045e,     #  CYRILLIC SMALL LETTER SHORT U
    0x00f8: 0x00b0,     #  DEGREE SIGN
    0x00f9: 0x2219,     #  BULLET OPERATOR
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
    '\u0404'   #  0x00f2 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u0454'   #  0x00f3 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u0407'   #  0x00f4 -> CYRILLIC CAPITAL LETTER YI
    '\u0457'   #  0x00f5 -> CYRILLIC SMALL LETTER YI
    '\u040e'   #  0x00f6 -> CYRILLIC CAPITAL LETTER SHORT U
    '\u045e'   #  0x00f7 -> CYRILLIC SMALL LETTER SHORT U
    '\xb0'     #  0x00f8 -> DEGREE SIGN
    '\u2219'   #  0x00f9 -> BULLET OPERATOR
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
    0x00b0: 0x00f8,     #  DEGREE SIGN
    0x00b7: 0x00fa,     #  MIDDLE DOT
    0x0401: 0x00f0,     #  CYRILLIC CAPITAL LETTER IO
    0x0404: 0x00f2,     #  CYRILLIC CAPITAL LETTER UKRAINIAN IE
    0x0407: 0x00f4,     #  CYRILLIC CAPITAL LETTER YI
    0x040e: 0x00f6,     #  CYRILLIC CAPITAL LETTER SHORT U
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
    0x0454: 0x00f3,     #  CYRILLIC SMALL LETTER UKRAINIAN IE
    0x0457: 0x00f5,     #  CYRILLIC SMALL LETTER YI
    0x045e: 0x00f7,     #  CYRILLIC SMALL LETTER SHORT U
    0x2116: 0x00fc,     #  NUMERO SIGN
    0x2219: 0x00f9,     #  BULLET OPERATOR
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
<|endoftext|>"""Python 'zlib_codec' Codec - zlib compression encoding.

This codec de/encodes from bytes to bytes.

Written by Marc-Andre Lemburg (mal@lemburg.com).
"""

import codecs
import zlib # this codec needs the optional zlib module !

### Codec APIs

def zlib_encode(input, errors='strict'):
    assert errors == 'strict'
    return (zlib.compress(input), len(input))

def zlib_decode(input, errors='strict'):
    assert errors == 'strict'
    return (zlib.decompress(input), len(input))

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):
        return zlib_encode(input, errors)
    def decode(self, input, errors='strict'):
        return zlib_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors='strict'):
        assert errors == 'strict'
        self.errors = errors
        self.compressobj = zlib.compressobj()

    def encode(self, input, final=False):
        if final:
            c = self.compressobj.compress(input)
            return c + self.compressobj.flush()
        else:
            return self.compressobj.compress(input)

    def reset(self):
        self.compressobj = zlib.compressobj()

class IncrementalDecoder(codecs.IncrementalDecoder):
    def __init__(self, errors='strict'):
        assert errors == 'strict'
        self.errors = errors
        self.decompressobj = zlib.decompressobj()

    def decode(self, input, final=False):
        if final:
            c = self.decompressobj.decompress(input)
            return c + self.decompressobj.flush()
        else:
            return self.decompressobj.decompress(input)

    def reset(self):
        self.decompressobj = zlib.decompressobj()

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='zlib',
        encode=zlib_encode,
        decode=zlib_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
        _is_text_encoding=False,
    )
<|endoftext|>"""Python 'base64_codec' Codec - base64 content transfer encoding.

This codec de/encodes from bytes to bytes.

Written by Marc-Andre Lemburg (mal@lemburg.com).
"""

import codecs
import base64

### Codec APIs

def base64_encode(input, errors='strict'):
    assert errors == 'strict'
    return (base64.encodebytes(input), len(input))

def base64_decode(input, errors='strict'):
    assert errors == 'strict'
    return (base64.decodebytes(input), len(input))

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):
        return base64_encode(input, errors)
    def decode(self, input, errors='strict'):
        return base64_decode(input, errors)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        assert self.errors == 'strict'
        return base64.encodebytes(input)

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        assert self.errors == 'strict'
        return base64.decodebytes(input)

class StreamWriter(Codec, codecs.StreamWriter):
    charbuffertype = bytes

class StreamReader(Codec, codecs.StreamReader):
    charbuffertype = bytes

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='base64',
        encode=base64_encode,
        decode=base64_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
        _is_text_encoding=False,
    )
<|endoftext|>#
# gb2312.py: Python Unicode Codec for GB2312
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_cn, codecs
import _multibytecodec as mbc

codec = _codecs_cn.getcodec('gb2312')

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
        name='gb2312',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec generated from 'VENDORS/MICSFT/PC/CP864.TXT' with gencodec.py.

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
        name='cp864',
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
    0x0025: 0x066a,     #  ARABIC PERCENT SIGN
    0x0080: 0x00b0,     #  DEGREE SIGN
    0x0081: 0x00b7,     #  MIDDLE DOT
    0x0082: 0x2219,     #  BULLET OPERATOR
    0x0083: 0x221a,     #  SQUARE ROOT
    0x0084: 0x2592,     #  MEDIUM SHADE
    0x0085: 0x2500,     #  FORMS LIGHT HORIZONTAL
    0x0086: 0x2502,     #  FORMS LIGHT VERTICAL
    0x0087: 0x253c,     #  FORMS LIGHT VERTICAL AND HORIZONTAL
    0x0088: 0x2524,     #  FORMS LIGHT VERTICAL AND LEFT
    0x0089: 0x252c,     #  FORMS LIGHT DOWN AND HORIZONTAL
    0x008a: 0x251c,     #  FORMS LIGHT VERTICAL AND RIGHT
    0x008b: 0x2534,     #  FORMS LIGHT UP AND HORIZONTAL
    0x008c: 0x2510,     #  FORMS LIGHT DOWN AND LEFT
    0x008d: 0x250c,     #  FORMS LIGHT DOWN AND RIGHT
    0x008e: 0x2514,     #  FORMS LIGHT UP AND RIGHT
    0x008f: 0x2518,     #  FORMS LIGHT UP AND LEFT
    0x0090: 0x03b2,     #  GREEK SMALL BETA
    0x0091: 0x221e,     #  INFINITY
    0x0092: 0x03c6,     #  GREEK SMALL PHI
    0x0093: 0x00b1,     #  PLUS-OR-MINUS SIGN
    0x0094: 0x00bd,     #  FRACTION 1/2
    0x0095: 0x00bc,     #  FRACTION 1/4
    0x0096: 0x2248,     #  ALMOST EQUAL TO
    0x0097: 0x00ab,     #  LEFT POINTING GUILLEMET
    0x0098: 0x00bb,     #  RIGHT POINTING GUILLEMET
    0x0099: 0xfef7,     #  ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE ISOLATED FORM
    0x009a: 0xfef8,     #  ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE FINAL FORM
    0x009b: None,       #  UNDEFINED
    0x009c: None,       #  UNDEFINED
    0x009d: 0xfefb,     #  ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
    0x009e: 0xfefc,     #  ARABIC LIGATURE LAM WITH ALEF FINAL FORM
    0x009f: None,       #  UNDEFINED
    0x00a1: 0x00ad,     #  SOFT HYPHEN
    0x00a2: 0xfe82,     #  ARABIC LETTER ALEF WITH MADDA ABOVE FINAL FORM
    0x00a5: 0xfe84,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE FINAL FORM
    0x00a6: None,       #  UNDEFINED
    0x00a7: None,       #  UNDEFINED
    0x00a8: 0xfe8e,     #  ARABIC LETTER ALEF FINAL FORM
    0x00a9: 0xfe8f,     #  ARABIC LETTER BEH ISOLATED FORM
    0x00aa: 0xfe95,     #  ARABIC LETTER TEH ISOLATED FORM
    0x00ab: 0xfe99,     #  ARABIC LETTER THEH ISOLATED FORM
    0x00ac: 0x060c,     #  ARABIC COMMA
    0x00ad: 0xfe9d,     #  ARABIC LETTER JEEM ISOLATED FORM
    0x00ae: 0xfea1,     #  ARABIC LETTER HAH ISOLATED FORM
    0x00af: 0xfea5,     #  ARABIC LETTER KHAH ISOLATED FORM
    0x00b0: 0x0660,     #  ARABIC-INDIC DIGIT ZERO
    0x00b1: 0x0661,     #  ARABIC-INDIC DIGIT ONE
    0x00b2: 0x0662,     #  ARABIC-INDIC DIGIT TWO
    0x00b3: 0x0663,     #  ARABIC-INDIC DIGIT THREE
    0x00b4: 0x0664,     #  ARABIC-INDIC DIGIT FOUR
    0x00b5: 0x0665,     #  ARABIC-INDIC DIGIT FIVE
    0x00b6: 0x0666,     #  ARABIC-INDIC DIGIT SIX
    0x00b7: 0x0667,     #  ARABIC-INDIC DIGIT SEVEN
    0x00b8: 0x0668,     #  ARABIC-INDIC DIGIT EIGHT
    0x00b9: 0x0669,     #  ARABIC-INDIC DIGIT NINE
    0x00ba: 0xfed1,     #  ARABIC LETTER FEH ISOLATED FORM
    0x00bb: 0x061b,     #  ARABIC SEMICOLON
    0x00bc: 0xfeb1,     #  ARABIC LETTER SEEN ISOLATED FORM
    0x00bd: 0xfeb5,     #  ARABIC LETTER SHEEN ISOLATED FORM
    0x00be: 0xfeb9,     #  ARABIC LETTER SAD ISOLATED FORM
    0x00bf: 0x061f,     #  ARABIC QUESTION MARK
    0x00c0: 0x00a2,     #  CENT SIGN
    0x00c1: 0xfe80,     #  ARABIC LETTER HAMZA ISOLATED FORM
    0x00c2: 0xfe81,     #  ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    0x00c3: 0xfe83,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE ISOLATED FORM
    0x00c4: 0xfe85,     #  ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    0x00c5: 0xfeca,     #  ARABIC LETTER AIN FINAL FORM
    0x00c6: 0xfe8b,     #  ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    0x00c7: 0xfe8d,     #  ARABIC LETTER ALEF ISOLATED FORM
    0x00c8: 0xfe91,     #  ARABIC LETTER BEH INITIAL FORM
    0x00c9: 0xfe93,     #  ARABIC LETTER TEH MARBUTA ISOLATED FORM
    0x00ca: 0xfe97,     #  ARABIC LETTER TEH INITIAL FORM
    0x00cb: 0xfe9b,     #  ARABIC LETTER THEH INITIAL FORM
    0x00cc: 0xfe9f,     #  ARABIC LETTER JEEM INITIAL FORM
    0x00cd: 0xfea3,     #  ARABIC LETTER HAH INITIAL FORM
    0x00ce: 0xfea7,     #  ARABIC LETTER KHAH INITIAL FORM
    0x00cf: 0xfea9,     #  ARABIC LETTER DAL ISOLATED FORM
    0x00d0: 0xfeab,     #  ARABIC LETTER THAL ISOLATED FORM
    0x00d1: 0xfead,     #  ARABIC LETTER REH ISOLATED FORM
    0x00d2: 0xfeaf,     #  ARABIC LETTER ZAIN ISOLATED FORM
    0x00d3: 0xfeb3,     #  ARABIC LETTER SEEN INITIAL FORM
    0x00d4: 0xfeb7,     #  ARABIC LETTER SHEEN INITIAL FORM
    0x00d5: 0xfebb,     #  ARABIC LETTER SAD INITIAL FORM
    0x00d6: 0xfebf,     #  ARABIC LETTER DAD INITIAL FORM
    0x00d7: 0xfec1,     #  ARABIC LETTER TAH ISOLATED FORM
    0x00d8: 0xfec5,     #  ARABIC LETTER ZAH ISOLATED FORM
    0x00d9: 0xfecb,     #  ARABIC LETTER AIN INITIAL FORM
    0x00da: 0xfecf,     #  ARABIC LETTER GHAIN INITIAL FORM
    0x00db: 0x00a6,     #  BROKEN VERTICAL BAR
    0x00dc: 0x00ac,     #  NOT SIGN
    0x00dd: 0x00f7,     #  DIVISION SIGN
    0x00de: 0x00d7,     #  MULTIPLICATION SIGN
    0x00df: 0xfec9,     #  ARABIC LETTER AIN ISOLATED FORM
    0x00e0: 0x0640,     #  ARABIC TATWEEL
    0x00e1: 0xfed3,     #  ARABIC LETTER FEH INITIAL FORM
    0x00e2: 0xfed7,     #  ARABIC LETTER QAF INITIAL FORM
    0x00e3: 0xfedb,     #  ARABIC LETTER KAF INITIAL FORM
    0x00e4: 0xfedf,     #  ARABIC LETTER LAM INITIAL FORM
    0x00e5: 0xfee3,     #  ARABIC LETTER MEEM INITIAL FORM
    0x00e6: 0xfee7,     #  ARABIC LETTER NOON INITIAL FORM
    0x00e7: 0xfeeb,     #  ARABIC LETTER HEH INITIAL FORM
    0x00e8: 0xfeed,     #  ARABIC LETTER WAW ISOLATED FORM
    0x00e9: 0xfeef,     #  ARABIC LETTER ALEF MAKSURA ISOLATED FORM
    0x00ea: 0xfef3,     #  ARABIC LETTER YEH INITIAL FORM
    0x00eb: 0xfebd,     #  ARABIC LETTER DAD ISOLATED FORM
    0x00ec: 0xfecc,     #  ARABIC LETTER AIN MEDIAL FORM
    0x00ed: 0xfece,     #  ARABIC LETTER GHAIN FINAL FORM
    0x00ee: 0xfecd,     #  ARABIC LETTER GHAIN ISOLATED FORM
    0x00ef: 0xfee1,     #  ARABIC LETTER MEEM ISOLATED FORM
    0x00f0: 0xfe7d,     #  ARABIC SHADDA MEDIAL FORM
    0x00f1: 0x0651,     #  ARABIC SHADDAH
    0x00f2: 0xfee5,     #  ARABIC LETTER NOON ISOLATED FORM
    0x00f3: 0xfee9,     #  ARABIC LETTER HEH ISOLATED FORM
    0x00f4: 0xfeec,     #  ARABIC LETTER HEH MEDIAL FORM
    0x00f5: 0xfef0,     #  ARABIC LETTER ALEF MAKSURA FINAL FORM
    0x00f6: 0xfef2,     #  ARABIC LETTER YEH FINAL FORM
    0x00f7: 0xfed0,     #  ARABIC LETTER GHAIN MEDIAL FORM
    0x00f8: 0xfed5,     #  ARABIC LETTER QAF ISOLATED FORM
    0x00f9: 0xfef5,     #  ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE ISOLATED FORM
    0x00fa: 0xfef6,     #  ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE FINAL FORM
    0x00fb: 0xfedd,     #  ARABIC LETTER LAM ISOLATED FORM
    0x00fc: 0xfed9,     #  ARABIC LETTER KAF ISOLATED FORM
    0x00fd: 0xfef1,     #  ARABIC LETTER YEH ISOLATED FORM
    0x00fe: 0x25a0,     #  BLACK SQUARE
    0x00ff: None,       #  UNDEFINED
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
    '\u066a'   #  0x0025 -> ARABIC PERCENT SIGN
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
    '\xb0'     #  0x0080 -> DEGREE SIGN
    '\xb7'     #  0x0081 -> MIDDLE DOT
    '\u2219'   #  0x0082 -> BULLET OPERATOR
    '\u221a'   #  0x0083 -> SQUARE ROOT
    '\u2592'   #  0x0084 -> MEDIUM SHADE
    '\u2500'   #  0x0085 -> FORMS LIGHT HORIZONTAL
    '\u2502'   #  0x0086 -> FORMS LIGHT VERTICAL
    '\u253c'   #  0x0087 -> FORMS LIGHT VERTICAL AND HORIZONTAL
    '\u2524'   #  0x0088 -> FORMS LIGHT VERTICAL AND LEFT
    '\u252c'   #  0x0089 -> FORMS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0x008a -> FORMS LIGHT VERTICAL AND RIGHT
    '\u2534'   #  0x008b -> FORMS LIGHT UP AND HORIZONTAL
    '\u2510'   #  0x008c -> FORMS LIGHT DOWN AND LEFT
    '\u250c'   #  0x008d -> FORMS LIGHT DOWN AND RIGHT
    '\u2514'   #  0x008e -> FORMS LIGHT UP AND RIGHT
    '\u2518'   #  0x008f -> FORMS LIGHT UP AND LEFT
    '\u03b2'   #  0x0090 -> GREEK SMALL BETA
    '\u221e'   #  0x0091 -> INFINITY
    '\u03c6'   #  0x0092 -> GREEK SMALL PHI
    '\xb1'     #  0x0093 -> PLUS-OR-MINUS SIGN
    '\xbd'     #  0x0094 -> FRACTION 1/2
    '\xbc'     #  0x0095 -> FRACTION 1/4
    '\u2248'   #  0x0096 -> ALMOST EQUAL TO
    '\xab'     #  0x0097 -> LEFT POINTING GUILLEMET
    '\xbb'     #  0x0098 -> RIGHT POINTING GUILLEMET
    '\ufef7'   #  0x0099 -> ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE ISOLATED FORM
    '\ufef8'   #  0x009a -> ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE FINAL FORM
    '\ufffe'   #  0x009b -> UNDEFINED
    '\ufffe'   #  0x009c -> UNDEFINED
    '\ufefb'   #  0x009d -> ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
    '\ufefc'   #  0x009e -> ARABIC LIGATURE LAM WITH ALEF FINAL FORM
    '\ufffe'   #  0x009f -> UNDEFINED
    '\xa0'     #  0x00a0 -> NON-BREAKING SPACE
    '\xad'     #  0x00a1 -> SOFT HYPHEN
    '\ufe82'   #  0x00a2 -> ARABIC LETTER ALEF WITH MADDA ABOVE FINAL FORM
    '\xa3'     #  0x00a3 -> POUND SIGN
    '\xa4'     #  0x00a4 -> CURRENCY SIGN
    '\ufe84'   #  0x00a5 -> ARABIC LETTER ALEF WITH HAMZA ABOVE FINAL FORM
    '\ufffe'   #  0x00a6 -> UNDEFINED
    '\ufffe'   #  0x00a7 -> UNDEFINED
    '\ufe8e'   #  0x00a8 -> ARABIC LETTER ALEF FINAL FORM
    '\ufe8f'   #  0x00a9 -> ARABIC LETTER BEH ISOLATED FORM
    '\ufe95'   #  0x00aa -> ARABIC LETTER TEH ISOLATED FORM
    '\ufe99'   #  0x00ab -> ARABIC LETTER THEH ISOLATED FORM
    '\u060c'   #  0x00ac -> ARABIC COMMA
    '\ufe9d'   #  0x00ad -> ARABIC LETTER JEEM ISOLATED FORM
    '\ufea1'   #  0x00ae -> ARABIC LETTER HAH ISOLATED FORM
    '\ufea5'   #  0x00af -> ARABIC LETTER KHAH ISOLATED FORM
    '\u0660'   #  0x00b0 -> ARABIC-INDIC DIGIT ZERO
    '\u0661'   #  0x00b1 -> ARABIC-INDIC DIGIT ONE
    '\u0662'   #  0x00b2 -> ARABIC-INDIC DIGIT TWO
    '\u0663'   #  0x00b3 -> ARABIC-INDIC DIGIT THREE
    '\u0664'   #  0x00b4 -> ARABIC-INDIC DIGIT FOUR
    '\u0665'   #  0x00b5 -> ARABIC-INDIC DIGIT FIVE
    '\u0666'   #  0x00b6 -> ARABIC-INDIC DIGIT SIX
    '\u0667'   #  0x00b7 -> ARABIC-INDIC DIGIT SEVEN
    '\u0668'   #  0x00b8 -> ARABIC-INDIC DIGIT EIGHT
    '\u0669'   #  0x00b9 -> ARABIC-INDIC DIGIT NINE
    '\ufed1'   #  0x00ba -> ARABIC LETTER FEH ISOLATED FORM
    '\u061b'   #  0x00bb -> ARABIC SEMICOLON
    '\ufeb1'   #  0x00bc -> ARABIC LETTER SEEN ISOLATED FORM
    '\ufeb5'   #  0x00bd -> ARABIC LETTER SHEEN ISOLATED FORM
    '\ufeb9'   #  0x00be -> ARABIC LETTER SAD ISOLATED FORM
    '\u061f'   #  0x00bf -> ARABIC QUESTION MARK
    '\xa2'     #  0x00c0 -> CENT SIGN
    '\ufe80'   #  0x00c1 -> ARABIC LETTER HAMZA ISOLATED FORM
    '\ufe81'   #  0x00c2 -> ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    '\ufe83'   #  0x00c3 -> ARABIC LETTER ALEF WITH HAMZA ABOVE ISOLATED FORM
    '\ufe85'   #  0x00c4 -> ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    '\ufeca'   #  0x00c5 -> ARABIC LETTER AIN FINAL FORM
    '\ufe8b'   #  0x00c6 -> ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    '\ufe8d'   #  0x00c7 -> ARABIC LETTER ALEF ISOLATED FORM
    '\ufe91'   #  0x00c8 -> ARABIC LETTER BEH INITIAL FORM
    '\ufe93'   #  0x00c9 -> ARABIC LETTER TEH MARBUTA ISOLATED FORM
    '\ufe97'   #  0x00ca -> ARABIC LETTER TEH INITIAL FORM
    '\ufe9b'   #  0x00cb -> ARABIC LETTER THEH INITIAL FORM
    '\ufe9f'   #  0x00cc -> ARABIC LETTER JEEM INITIAL FORM
    '\ufea3'   #  0x00cd -> ARABIC LETTER HAH INITIAL FORM
    '\ufea7'   #  0x00ce -> ARABIC LETTER KHAH INITIAL FORM
    '\ufea9'   #  0x00cf -> ARABIC LETTER DAL ISOLATED FORM
    '\ufeab'   #  0x00d0 -> ARABIC LETTER THAL ISOLATED FORM
    '\ufead'   #  0x00d1 -> ARABIC LETTER REH ISOLATED FORM
    '\ufeaf'   #  0x00d2 -> ARABIC LETTER ZAIN ISOLATED FORM
    '\ufeb3'   #  0x00d3 -> ARABIC LETTER SEEN INITIAL FORM
    '\ufeb7'   #  0x00d4 -> ARABIC LETTER SHEEN INITIAL FORM
    '\ufebb'   #  0x00d5 -> ARABIC LETTER SAD INITIAL FORM
    '\ufebf'   #  0x00d6 -> ARABIC LETTER DAD INITIAL FORM
    '\ufec1'   #  0x00d7 -> ARABIC LETTER TAH ISOLATED FORM
    '\ufec5'   #  0x00d8 -> ARABIC LETTER ZAH ISOLATED FORM
    '\ufecb'   #  0x00d9 -> ARABIC LETTER AIN INITIAL FORM
    '\ufecf'   #  0x00da -> ARABIC LETTER GHAIN INITIAL FORM
    '\xa6'     #  0x00db -> BROKEN VERTICAL BAR
    '\xac'     #  0x00dc -> NOT SIGN
    '\xf7'     #  0x00dd -> DIVISION SIGN
    '\xd7'     #  0x00de -> MULTIPLICATION SIGN
    '\ufec9'   #  0x00df -> ARABIC LETTER AIN ISOLATED FORM
    '\u0640'   #  0x00e0 -> ARABIC TATWEEL
    '\ufed3'   #  0x00e1 -> ARABIC LETTER FEH INITIAL FORM
    '\ufed7'   #  0x00e2 -> ARABIC LETTER QAF INITIAL FORM
    '\ufedb'   #  0x00e3 -> ARABIC LETTER KAF INITIAL FORM
    '\ufedf'   #  0x00e4 -> ARABIC LETTER LAM INITIAL FORM
    '\ufee3'   #  0x00e5 -> ARABIC LETTER MEEM INITIAL FORM
    '\ufee7'   #  0x00e6 -> ARABIC LETTER NOON INITIAL FORM
    '\ufeeb'   #  0x00e7 -> ARABIC LETTER HEH INITIAL FORM
    '\ufeed'   #  0x00e8 -> ARABIC LETTER WAW ISOLATED FORM
    '\ufeef'   #  0x00e9 -> ARABIC LETTER ALEF MAKSURA ISOLATED FORM
    '\ufef3'   #  0x00ea -> ARABIC LETTER YEH INITIAL FORM
    '\ufebd'   #  0x00eb -> ARABIC LETTER DAD ISOLATED FORM
    '\ufecc'   #  0x00ec -> ARABIC LETTER AIN MEDIAL FORM
    '\ufece'   #  0x00ed -> ARABIC LETTER GHAIN FINAL FORM
    '\ufecd'   #  0x00ee -> ARABIC LETTER GHAIN ISOLATED FORM
    '\ufee1'   #  0x00ef -> ARABIC LETTER MEEM ISOLATED FORM
    '\ufe7d'   #  0x00f0 -> ARABIC SHADDA MEDIAL FORM
    '\u0651'   #  0x00f1 -> ARABIC SHADDAH
    '\ufee5'   #  0x00f2 -> ARABIC LETTER NOON ISOLATED FORM
    '\ufee9'   #  0x00f3 -> ARABIC LETTER HEH ISOLATED FORM
    '\ufeec'   #  0x00f4 -> ARABIC LETTER HEH MEDIAL FORM
    '\ufef0'   #  0x00f5 -> ARABIC LETTER ALEF MAKSURA FINAL FORM
    '\ufef2'   #  0x00f6 -> ARABIC LETTER YEH FINAL FORM
    '\ufed0'   #  0x00f7 -> ARABIC LETTER GHAIN MEDIAL FORM
    '\ufed5'   #  0x00f8 -> ARABIC LETTER QAF ISOLATED FORM
    '\ufef5'   #  0x00f9 -> ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE ISOLATED FORM
    '\ufef6'   #  0x00fa -> ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE FINAL FORM
    '\ufedd'   #  0x00fb -> ARABIC LETTER LAM ISOLATED FORM
    '\ufed9'   #  0x00fc -> ARABIC LETTER KAF ISOLATED FORM
    '\ufef1'   #  0x00fd -> ARABIC LETTER YEH ISOLATED FORM
    '\u25a0'   #  0x00fe -> BLACK SQUARE
    '\ufffe'   #  0x00ff -> UNDEFINED
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
    0x00a0: 0x00a0,     #  NON-BREAKING SPACE
    0x00a2: 0x00c0,     #  CENT SIGN
    0x00a3: 0x00a3,     #  POUND SIGN
    0x00a4: 0x00a4,     #  CURRENCY SIGN
    0x00a6: 0x00db,     #  BROKEN VERTICAL BAR
    0x00ab: 0x0097,     #  LEFT POINTING GUILLEMET
    0x00ac: 0x00dc,     #  NOT SIGN
    0x00ad: 0x00a1,     #  SOFT HYPHEN
    0x00b0: 0x0080,     #  DEGREE SIGN
    0x00b1: 0x0093,     #  PLUS-OR-MINUS SIGN
    0x00b7: 0x0081,     #  MIDDLE DOT
    0x00bb: 0x0098,     #  RIGHT POINTING GUILLEMET
    0x00bc: 0x0095,     #  FRACTION 1/4
    0x00bd: 0x0094,     #  FRACTION 1/2
    0x00d7: 0x00de,     #  MULTIPLICATION SIGN
    0x00f7: 0x00dd,     #  DIVISION SIGN
    0x03b2: 0x0090,     #  GREEK SMALL BETA
    0x03c6: 0x0092,     #  GREEK SMALL PHI
    0x060c: 0x00ac,     #  ARABIC COMMA
    0x061b: 0x00bb,     #  ARABIC SEMICOLON
    0x061f: 0x00bf,     #  ARABIC QUESTION MARK
    0x0640: 0x00e0,     #  ARABIC TATWEEL
    0x0651: 0x00f1,     #  ARABIC SHADDAH
    0x0660: 0x00b0,     #  ARABIC-INDIC DIGIT ZERO
    0x0661: 0x00b1,     #  ARABIC-INDIC DIGIT ONE
    0x0662: 0x00b2,     #  ARABIC-INDIC DIGIT TWO
    0x0663: 0x00b3,     #  ARABIC-INDIC DIGIT THREE
    0x0664: 0x00b4,     #  ARABIC-INDIC DIGIT FOUR
    0x0665: 0x00b5,     #  ARABIC-INDIC DIGIT FIVE
    0x0666: 0x00b6,     #  ARABIC-INDIC DIGIT SIX
    0x0667: 0x00b7,     #  ARABIC-INDIC DIGIT SEVEN
    0x0668: 0x00b8,     #  ARABIC-INDIC DIGIT EIGHT
    0x0669: 0x00b9,     #  ARABIC-INDIC DIGIT NINE
    0x066a: 0x0025,     #  ARABIC PERCENT SIGN
    0x2219: 0x0082,     #  BULLET OPERATOR
    0x221a: 0x0083,     #  SQUARE ROOT
    0x221e: 0x0091,     #  INFINITY
    0x2248: 0x0096,     #  ALMOST EQUAL TO
    0x2500: 0x0085,     #  FORMS LIGHT HORIZONTAL
    0x2502: 0x0086,     #  FORMS LIGHT VERTICAL
    0x250c: 0x008d,     #  FORMS LIGHT DOWN AND RIGHT
    0x2510: 0x008c,     #  FORMS LIGHT DOWN AND LEFT
    0x2514: 0x008e,     #  FORMS LIGHT UP AND RIGHT
    0x2518: 0x008f,     #  FORMS LIGHT UP AND LEFT
    0x251c: 0x008a,     #  FORMS LIGHT VERTICAL AND RIGHT
    0x2524: 0x0088,     #  FORMS LIGHT VERTICAL AND LEFT
    0x252c: 0x0089,     #  FORMS LIGHT DOWN AND HORIZONTAL
    0x2534: 0x008b,     #  FORMS LIGHT UP AND HORIZONTAL
    0x253c: 0x0087,     #  FORMS LIGHT VERTICAL AND HORIZONTAL
    0x2592: 0x0084,     #  MEDIUM SHADE
    0x25a0: 0x00fe,     #  BLACK SQUARE
    0xfe7d: 0x00f0,     #  ARABIC SHADDA MEDIAL FORM
    0xfe80: 0x00c1,     #  ARABIC LETTER HAMZA ISOLATED FORM
    0xfe81: 0x00c2,     #  ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    0xfe82: 0x00a2,     #  ARABIC LETTER ALEF WITH MADDA ABOVE FINAL FORM
    0xfe83: 0x00c3,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE ISOLATED FORM
    0xfe84: 0x00a5,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE FINAL FORM
    0xfe85: 0x00c4,     #  ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    0xfe8b: 0x00c6,     #  ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    0xfe8d: 0x00c7,     #  ARABIC LETTER ALEF ISOLATED FORM
    0xfe8e: 0x00a8,     #  ARABIC LETTER ALEF FINAL FORM
    0xfe8f: 0x00a9,     #  ARABIC LETTER BEH ISOLATED FORM
    0xfe91: 0x00c8,     #  ARABIC LETTER BEH INITIAL FORM
    0xfe93: 0x00c9,     #  ARABIC LETTER TEH MARBUTA ISOLATED FORM
    0xfe95: 0x00aa,     #  ARABIC LETTER TEH ISOLATED FORM
    0xfe97: 0x00ca,     #  ARABIC LETTER TEH INITIAL FORM
    0xfe99: 0x00ab,     #  ARABIC LETTER THEH ISOLATED FORM
    0xfe9b: 0x00cb,     #  ARABIC LETTER THEH INITIAL FORM
    0xfe9d: 0x00ad,     #  ARABIC LETTER JEEM ISOLATED FORM
    0xfe9f: 0x00cc,     #  ARABIC LETTER JEEM INITIAL FORM
    0xfea1: 0x00ae,     #  ARABIC LETTER HAH ISOLATED FORM
    0xfea3: 0x00cd,     #  ARABIC LETTER HAH INITIAL FORM
    0xfea5: 0x00af,     #  ARABIC LETTER KHAH ISOLATED FORM
    0xfea7: 0x00ce,     #  ARABIC LETTER KHAH INITIAL FORM
    0xfea9: 0x00cf,     #  ARABIC LETTER DAL ISOLATED FORM
    0xfeab: 0x00d0,     #  ARABIC LETTER THAL ISOLATED FORM
    0xfead: 0x00d1,     #  ARABIC LETTER REH ISOLATED FORM
    0xfeaf: 0x00d2,     #  ARABIC LETTER ZAIN ISOLATED FORM
    0xfeb1: 0x00bc,     #  ARABIC LETTER SEEN ISOLATED FORM
    0xfeb3: 0x00d3,     #  ARABIC LETTER SEEN INITIAL FORM
    0xfeb5: 0x00bd,     #  ARABIC LETTER SHEEN ISOLATED FORM
    0xfeb7: 0x00d4,     #  ARABIC LETTER SHEEN INITIAL FORM
    0xfeb9: 0x00be,     #  ARABIC LETTER SAD ISOLATED FORM
    0xfebb: 0x00d5,     #  ARABIC LETTER SAD INITIAL FORM
    0xfebd: 0x00eb,     #  ARABIC LETTER DAD ISOLATED FORM
    0xfebf: 0x00d6,     #  ARABIC LETTER DAD INITIAL FORM
    0xfec1: 0x00d7,     #  ARABIC LETTER TAH ISOLATED FORM
    0xfec5: 0x00d8,     #  ARABIC LETTER ZAH ISOLATED FORM
    0xfec9: 0x00df,     #  ARABIC LETTER AIN ISOLATED FORM
    0xfeca: 0x00c5,     #  ARABIC LETTER AIN FINAL FORM
    0xfecb: 0x00d9,     #  ARABIC LETTER AIN INITIAL FORM
    0xfecc: 0x00ec,     #  ARABIC LETTER AIN MEDIAL FORM
    0xfecd: 0x00ee,     #  ARABIC LETTER GHAIN ISOLATED FORM
    0xfece: 0x00ed,     #  ARABIC LETTER GHAIN FINAL FORM
    0xfecf: 0x00da,     #  ARABIC LETTER GHAIN INITIAL FORM
    0xfed0: 0x00f7,     #  ARABIC LETTER GHAIN MEDIAL FORM
    0xfed1: 0x00ba,     #  ARABIC LETTER FEH ISOLATED FORM
    0xfed3: 0x00e1,     #  ARABIC LETTER FEH INITIAL FORM
    0xfed5: 0x00f8,     #  ARABIC LETTER QAF ISOLATED FORM
    0xfed7: 0x00e2,     #  ARABIC LETTER QAF INITIAL FORM
    0xfed9: 0x00fc,     #  ARABIC LETTER KAF ISOLATED FORM
    0xfedb: 0x00e3,     #  ARABIC LETTER KAF INITIAL FORM
    0xfedd: 0x00fb,     #  ARABIC LETTER LAM ISOLATED FORM
    0xfedf: 0x00e4,     #  ARABIC LETTER LAM INITIAL FORM
    0xfee1: 0x00ef,     #  ARABIC LETTER MEEM ISOLATED FORM
    0xfee3: 0x00e5,     #  ARABIC LETTER MEEM INITIAL FORM
    0xfee5: 0x00f2,     #  ARABIC LETTER NOON ISOLATED FORM
    0xfee7: 0x00e6,     #  ARABIC LETTER NOON INITIAL FORM
    0xfee9: 0x00f3,     #  ARABIC LETTER HEH ISOLATED FORM
    0xfeeb: 0x00e7,     #  ARABIC LETTER HEH INITIAL FORM
    0xfeec: 0x00f4,     #  ARABIC LETTER HEH MEDIAL FORM
    0xfeed: 0x00e8,     #  ARABIC LETTER WAW ISOLATED FORM
    0xfeef: 0x00e9,     #  ARABIC LETTER ALEF MAKSURA ISOLATED FORM
    0xfef0: 0x00f5,     #  ARABIC LETTER ALEF MAKSURA FINAL FORM
    0xfef1: 0x00fd,     #  ARABIC LETTER YEH ISOLATED FORM
    0xfef2: 0x00f6,     #  ARABIC LETTER YEH FINAL FORM
    0xfef3: 0x00ea,     #  ARABIC LETTER YEH INITIAL FORM
    0xfef5: 0x00f9,     #  ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE ISOLATED FORM
    0xfef6: 0x00fa,     #  ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE FINAL FORM
    0xfef7: 0x0099,     #  ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE ISOLATED FORM
    0xfef8: 0x009a,     #  ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE FINAL FORM
    0xfefb: 0x009d,     #  ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
    0xfefc: 0x009e,     #  ARABIC LIGATURE LAM WITH ALEF FINAL FORM
}
<|endoftext|>""" Python Character Mapping Codec cp1253 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1253.TXT' with gencodec.py.

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
        name='cp1253',
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
    '\ufffe'   #  0x88 -> UNDEFINED
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\ufffe'   #  0x8A -> UNDEFINED
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
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
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0385'   #  0xA1 -> GREEK DIALYTIKA TONOS
    '\u0386'   #  0xA2 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\ufffe'   #  0xAA -> UNDEFINED
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u2015'   #  0xAF -> HORIZONTAL BAR
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\u0384'   #  0xB4 -> GREEK TONOS
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u0388'   #  0xB8 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0389'   #  0xB9 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\u038a'   #  0xBA -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u038c'   #  0xBC -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\u038e'   #  0xBE -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u038f'   #  0xBF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '\u0390'   #  0xC0 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u0391'   #  0xC1 -> GREEK CAPITAL LETTER ALPHA
    '\u0392'   #  0xC2 -> GREEK CAPITAL LETTER BETA
    '\u0393'   #  0xC3 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0xC4 -> GREEK CAPITAL LETTER DELTA
    '\u0395'   #  0xC5 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0xC6 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0xC7 -> GREEK CAPITAL LETTER ETA
    '\u0398'   #  0xC8 -> GREEK CAPITAL LETTER THETA
    '\u0399'   #  0xC9 -> GREEK CAPITAL LETTER IOTA
    '\u039a'   #  0xCA -> GREEK CAPITAL LETTER KAPPA
    '\u039b'   #  0xCB -> GREEK CAPITAL LETTER LAMDA
    '\u039c'   #  0xCC -> GREEK CAPITAL LETTER MU
    '\u039d'   #  0xCD -> GREEK CAPITAL LETTER NU
    '\u039e'   #  0xCE -> GREEK CAPITAL LETTER XI
    '\u039f'   #  0xCF -> GREEK CAPITAL LETTER OMICRON
    '\u03a0'   #  0xD0 -> GREEK CAPITAL LETTER PI
    '\u03a1'   #  0xD1 -> GREEK CAPITAL LETTER RHO
    '\ufffe'   #  0xD2 -> UNDEFINED
    '\u03a3'   #  0xD3 -> GREEK CAPITAL LETTER SIGMA
    '\u03a4'   #  0xD4 -> GREEK CAPITAL LETTER TAU
    '\u03a5'   #  0xD5 -> GREEK CAPITAL LETTER UPSILON
    '\u03a6'   #  0xD6 -> GREEK CAPITAL LETTER PHI
    '\u03a7'   #  0xD7 -> GREEK CAPITAL LETTER CHI
    '\u03a8'   #  0xD8 -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0xD9 -> GREEK CAPITAL LETTER OMEGA
    '\u03aa'   #  0xDA -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\u03ab'   #  0xDB -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '\u03ac'   #  0xDC -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u03ad'   #  0xDD -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xDE -> GREEK SMALL LETTER ETA WITH TONOS
    '\u03af'   #  0xDF -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03b0'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    '\u03b3'   #  0xE3 -> GREEK SMALL LETTER GAMMA
    '\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    '\u03b6'   #  0xE6 -> GREEK SMALL LETTER ZETA
    '\u03b7'   #  0xE7 -> GREEK SMALL LETTER ETA
    '\u03b8'   #  0xE8 -> GREEK SMALL LETTER THETA
    '\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    '\u03ba'   #  0xEA -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0xEB -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0xEC -> GREEK SMALL LETTER MU
    '\u03bd'   #  0xED -> GREEK SMALL LETTER NU
    '\u03be'   #  0xEE -> GREEK SMALL LETTER XI
    '\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    '\u03c1'   #  0xF1 -> GREEK SMALL LETTER RHO
    '\u03c2'   #  0xF2 -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    '\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    '\u03c5'   #  0xF5 -> GREEK SMALL LETTER UPSILON
    '\u03c6'   #  0xF6 -> GREEK SMALL LETTER PHI
    '\u03c7'   #  0xF7 -> GREEK SMALL LETTER CHI
    '\u03c8'   #  0xF8 -> GREEK SMALL LETTER PSI
    '\u03c9'   #  0xF9 -> GREEK SMALL LETTER OMEGA
    '\u03ca'   #  0xFA -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03cb'   #  0xFB -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u03cc'   #  0xFC -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u03cd'   #  0xFD -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03ce'   #  0xFE -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>"""Python Character Mapping Codec cp720 generated on Windows:
Vista 6.0.6002 SP2 Multiprocessor Free with the command:
  python Tools/unicode/genwincodec.py 720
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
        name='cp720',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'      #  0x00 -> CONTROL CHARACTER
    '\x01'      #  0x01 -> CONTROL CHARACTER
    '\x02'      #  0x02 -> CONTROL CHARACTER
    '\x03'      #  0x03 -> CONTROL CHARACTER
    '\x04'      #  0x04 -> CONTROL CHARACTER
    '\x05'      #  0x05 -> CONTROL CHARACTER
    '\x06'      #  0x06 -> CONTROL CHARACTER
    '\x07'      #  0x07 -> CONTROL CHARACTER
    '\x08'      #  0x08 -> CONTROL CHARACTER
    '\t'        #  0x09 -> CONTROL CHARACTER
    '\n'        #  0x0A -> CONTROL CHARACTER
    '\x0b'      #  0x0B -> CONTROL CHARACTER
    '\x0c'      #  0x0C -> CONTROL CHARACTER
    '\r'        #  0x0D -> CONTROL CHARACTER
    '\x0e'      #  0x0E -> CONTROL CHARACTER
    '\x0f'      #  0x0F -> CONTROL CHARACTER
    '\x10'      #  0x10 -> CONTROL CHARACTER
    '\x11'      #  0x11 -> CONTROL CHARACTER
    '\x12'      #  0x12 -> CONTROL CHARACTER
    '\x13'      #  0x13 -> CONTROL CHARACTER
    '\x14'      #  0x14 -> CONTROL CHARACTER
    '\x15'      #  0x15 -> CONTROL CHARACTER
    '\x16'      #  0x16 -> CONTROL CHARACTER
    '\x17'      #  0x17 -> CONTROL CHARACTER
    '\x18'      #  0x18 -> CONTROL CHARACTER
    '\x19'      #  0x19 -> CONTROL CHARACTER
    '\x1a'      #  0x1A -> CONTROL CHARACTER
    '\x1b'      #  0x1B -> CONTROL CHARACTER
    '\x1c'      #  0x1C -> CONTROL CHARACTER
    '\x1d'      #  0x1D -> CONTROL CHARACTER
    '\x1e'      #  0x1E -> CONTROL CHARACTER
    '\x1f'      #  0x1F -> CONTROL CHARACTER
    ' '         #  0x20 -> SPACE
    '!'         #  0x21 -> EXCLAMATION MARK
    '"'         #  0x22 -> QUOTATION MARK
    '#'         #  0x23 -> NUMBER SIGN
    '$'         #  0x24 -> DOLLAR SIGN
    '%'         #  0x25 -> PERCENT SIGN
    '&'         #  0x26 -> AMPERSAND
    "'"         #  0x27 -> APOSTROPHE
    '('         #  0x28 -> LEFT PARENTHESIS
    ')'         #  0x29 -> RIGHT PARENTHESIS
    '*'         #  0x2A -> ASTERISK
    '+'         #  0x2B -> PLUS SIGN
    ','         #  0x2C -> COMMA
    '-'         #  0x2D -> HYPHEN-MINUS
    '.'         #  0x2E -> FULL STOP
    '/'         #  0x2F -> SOLIDUS
    '0'         #  0x30 -> DIGIT ZERO
    '1'         #  0x31 -> DIGIT ONE
    '2'         #  0x32 -> DIGIT TWO
    '3'         #  0x33 -> DIGIT THREE
    '4'         #  0x34 -> DIGIT FOUR
    '5'         #  0x35 -> DIGIT FIVE
    '6'         #  0x36 -> DIGIT SIX
    '7'         #  0x37 -> DIGIT SEVEN
    '8'         #  0x38 -> DIGIT EIGHT
    '9'         #  0x39 -> DIGIT NINE
    ':'         #  0x3A -> COLON
    ';'         #  0x3B -> SEMICOLON
    '<'         #  0x3C -> LESS-THAN SIGN
    '='         #  0x3D -> EQUALS SIGN
    '>'         #  0x3E -> GREATER-THAN SIGN
    '?'         #  0x3F -> QUESTION MARK
    '@'         #  0x40 -> COMMERCIAL AT
    'A'         #  0x41 -> LATIN CAPITAL LETTER A
    'B'         #  0x42 -> LATIN CAPITAL LETTER B
    'C'         #  0x43 -> LATIN CAPITAL LETTER C
    'D'         #  0x44 -> LATIN CAPITAL LETTER D
    'E'         #  0x45 -> LATIN CAPITAL LETTER E
    'F'         #  0x46 -> LATIN CAPITAL LETTER F
    'G'         #  0x47 -> LATIN CAPITAL LETTER G
    'H'         #  0x48 -> LATIN CAPITAL LETTER H
    'I'         #  0x49 -> LATIN CAPITAL LETTER I
    'J'         #  0x4A -> LATIN CAPITAL LETTER J
    'K'         #  0x4B -> LATIN CAPITAL LETTER K
    'L'         #  0x4C -> LATIN CAPITAL LETTER L
    'M'         #  0x4D -> LATIN CAPITAL LETTER M
    'N'         #  0x4E -> LATIN CAPITAL LETTER N
    'O'         #  0x4F -> LATIN CAPITAL LETTER O
    'P'         #  0x50 -> LATIN CAPITAL LETTER P
    'Q'         #  0x51 -> LATIN CAPITAL LETTER Q
    'R'         #  0x52 -> LATIN CAPITAL LETTER R
    'S'         #  0x53 -> LATIN CAPITAL LETTER S
    'T'         #  0x54 -> LATIN CAPITAL LETTER T
    'U'         #  0x55 -> LATIN CAPITAL LETTER U
    'V'         #  0x56 -> LATIN CAPITAL LETTER V
    'W'         #  0x57 -> LATIN CAPITAL LETTER W
    'X'         #  0x58 -> LATIN CAPITAL LETTER X
    'Y'         #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'         #  0x5A -> LATIN CAPITAL LETTER Z
    '['         #  0x5B -> LEFT SQUARE BRACKET
    '\\'        #  0x5C -> REVERSE SOLIDUS
    ']'         #  0x5D -> RIGHT SQUARE BRACKET
    '^'         #  0x5E -> CIRCUMFLEX ACCENT
    '_'         #  0x5F -> LOW LINE
    '`'         #  0x60 -> GRAVE ACCENT
    'a'         #  0x61 -> LATIN SMALL LETTER A
    'b'         #  0x62 -> LATIN SMALL LETTER B
    'c'         #  0x63 -> LATIN SMALL LETTER C
    'd'         #  0x64 -> LATIN SMALL LETTER D
    'e'         #  0x65 -> LATIN SMALL LETTER E
    'f'         #  0x66 -> LATIN SMALL LETTER F
    'g'         #  0x67 -> LATIN SMALL LETTER G
    'h'         #  0x68 -> LATIN SMALL LETTER H
    'i'         #  0x69 -> LATIN SMALL LETTER I
    'j'         #  0x6A -> LATIN SMALL LETTER J
    'k'         #  0x6B -> LATIN SMALL LETTER K
    'l'         #  0x6C -> LATIN SMALL LETTER L
    'm'         #  0x6D -> LATIN SMALL LETTER M
    'n'         #  0x6E -> LATIN SMALL LETTER N
    'o'         #  0x6F -> LATIN SMALL LETTER O
    'p'         #  0x70 -> LATIN SMALL LETTER P
    'q'         #  0x71 -> LATIN SMALL LETTER Q
    'r'         #  0x72 -> LATIN SMALL LETTER R
    's'         #  0x73 -> LATIN SMALL LETTER S
    't'         #  0x74 -> LATIN SMALL LETTER T
    'u'         #  0x75 -> LATIN SMALL LETTER U
    'v'         #  0x76 -> LATIN SMALL LETTER V
    'w'         #  0x77 -> LATIN SMALL LETTER W
    'x'         #  0x78 -> LATIN SMALL LETTER X
    'y'         #  0x79 -> LATIN SMALL LETTER Y
    'z'         #  0x7A -> LATIN SMALL LETTER Z
    '{'         #  0x7B -> LEFT CURLY BRACKET
    '|'         #  0x7C -> VERTICAL LINE
    '}'         #  0x7D -> RIGHT CURLY BRACKET
    '~'         #  0x7E -> TILDE
    '\x7f'      #  0x7F -> CONTROL CHARACTER
    '\x80'
    '\x81'
    '\xe9'      #  0x82 -> LATIN SMALL LETTER E WITH ACUTE
    '\xe2'      #  0x83 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\x84'
    '\xe0'      #  0x85 -> LATIN SMALL LETTER A WITH GRAVE
    '\x86'
    '\xe7'      #  0x87 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xea'      #  0x88 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'      #  0x89 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xe8'      #  0x8A -> LATIN SMALL LETTER E WITH GRAVE
    '\xef'      #  0x8B -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xee'      #  0x8C -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\x8d'
    '\x8e'
    '\x8f'
    '\x90'
    '\u0651'    #  0x91 -> ARABIC SHADDA
    '\u0652'    #  0x92 -> ARABIC SUKUN
    '\xf4'      #  0x93 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xa4'      #  0x94 -> CURRENCY SIGN
    '\u0640'    #  0x95 -> ARABIC TATWEEL
    '\xfb'      #  0x96 -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xf9'      #  0x97 -> LATIN SMALL LETTER U WITH GRAVE
    '\u0621'    #  0x98 -> ARABIC LETTER HAMZA
    '\u0622'    #  0x99 -> ARABIC LETTER ALEF WITH MADDA ABOVE
    '\u0623'    #  0x9A -> ARABIC LETTER ALEF WITH HAMZA ABOVE
    '\u0624'    #  0x9B -> ARABIC LETTER WAW WITH HAMZA ABOVE
    '\xa3'      #  0x9C -> POUND SIGN
    '\u0625'    #  0x9D -> ARABIC LETTER ALEF WITH HAMZA BELOW
    '\u0626'    #  0x9E -> ARABIC LETTER YEH WITH HAMZA ABOVE
    '\u0627'    #  0x9F -> ARABIC LETTER ALEF
    '\u0628'    #  0xA0 -> ARABIC LETTER BEH
    '\u0629'    #  0xA1 -> ARABIC LETTER TEH MARBUTA
    '\u062a'    #  0xA2 -> ARABIC LETTER TEH
    '\u062b'    #  0xA3 -> ARABIC LETTER THEH
    '\u062c'    #  0xA4 -> ARABIC LETTER JEEM
    '\u062d'    #  0xA5 -> ARABIC LETTER HAH
    '\u062e'    #  0xA6 -> ARABIC LETTER KHAH
    '\u062f'    #  0xA7 -> ARABIC LETTER DAL
    '\u0630'    #  0xA8 -> ARABIC LETTER THAL
    '\u0631'    #  0xA9 -> ARABIC LETTER REH
    '\u0632'    #  0xAA -> ARABIC LETTER ZAIN
    '\u0633'    #  0xAB -> ARABIC LETTER SEEN
    '\u0634'    #  0xAC -> ARABIC LETTER SHEEN
    '\u0635'    #  0xAD -> ARABIC LETTER SAD
    '\xab'      #  0xAE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'      #  0xAF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2591'    #  0xB0 -> LIGHT SHADE
    '\u2592'    #  0xB1 -> MEDIUM SHADE
    '\u2593'    #  0xB2 -> DARK SHADE
    '\u2502'    #  0xB3 -> BOX DRAWINGS LIGHT VERTICAL
    '\u2524'    #  0xB4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u2561'    #  0xB5 -> BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    '\u2562'    #  0xB6 -> BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    '\u2556'    #  0xB7 -> BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    '\u2555'    #  0xB8 -> BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    '\u2563'    #  0xB9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'    #  0xBA -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'    #  0xBB -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'    #  0xBC -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u255c'    #  0xBD -> BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    '\u255b'    #  0xBE -> BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    '\u2510'    #  0xBF -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'    #  0xC0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'    #  0xC1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'    #  0xC2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'    #  0xC3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'    #  0xC4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'    #  0xC5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u255e'    #  0xC6 -> BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    '\u255f'    #  0xC7 -> BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    '\u255a'    #  0xC8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'    #  0xC9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'    #  0xCA -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'    #  0xCB -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'    #  0xCC -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'    #  0xCD -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'    #  0xCE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\u2567'    #  0xCF -> BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    '\u2568'    #  0xD0 -> BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    '\u2564'    #  0xD1 -> BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    '\u2565'    #  0xD2 -> BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    '\u2559'    #  0xD3 -> BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    '\u2558'    #  0xD4 -> BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    '\u2552'    #  0xD5 -> BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    '\u2553'    #  0xD6 -> BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    '\u256b'    #  0xD7 -> BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    '\u256a'    #  0xD8 -> BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    '\u2518'    #  0xD9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'    #  0xDA -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'    #  0xDB -> FULL BLOCK
    '\u2584'    #  0xDC -> LOWER HALF BLOCK
    '\u258c'    #  0xDD -> LEFT HALF BLOCK
    '\u2590'    #  0xDE -> RIGHT HALF BLOCK
    '\u2580'    #  0xDF -> UPPER HALF BLOCK
    '\u0636'    #  0xE0 -> ARABIC LETTER DAD
    '\u0637'    #  0xE1 -> ARABIC LETTER TAH
    '\u0638'    #  0xE2 -> ARABIC LETTER ZAH
    '\u0639'    #  0xE3 -> ARABIC LETTER AIN
    '\u063a'    #  0xE4 -> ARABIC LETTER GHAIN
    '\u0641'    #  0xE5 -> ARABIC LETTER FEH
    '\xb5'      #  0xE6 -> MICRO SIGN
    '\u0642'    #  0xE7 -> ARABIC LETTER QAF
    '\u0643'    #  0xE8 -> ARABIC LETTER KAF
    '\u0644'    #  0xE9 -> ARABIC LETTER LAM
    '\u0645'    #  0xEA -> ARABIC LETTER MEEM
    '\u0646'    #  0xEB -> ARABIC LETTER NOON
    '\u0647'    #  0xEC -> ARABIC LETTER HEH
    '\u0648'    #  0xED -> ARABIC LETTER WAW
    '\u0649'    #  0xEE -> ARABIC LETTER ALEF MAKSURA
    '\u064a'    #  0xEF -> ARABIC LETTER YEH
    '\u2261'    #  0xF0 -> IDENTICAL TO
    '\u064b'    #  0xF1 -> ARABIC FATHATAN
    '\u064c'    #  0xF2 -> ARABIC DAMMATAN
    '\u064d'    #  0xF3 -> ARABIC KASRATAN
    '\u064e'    #  0xF4 -> ARABIC FATHA
    '\u064f'    #  0xF5 -> ARABIC DAMMA
    '\u0650'    #  0xF6 -> ARABIC KASRA
    '\u2248'    #  0xF7 -> ALMOST EQUAL TO
    '\xb0'      #  0xF8 -> DEGREE SIGN
    '\u2219'    #  0xF9 -> BULLET OPERATOR
    '\xb7'      #  0xFA -> MIDDLE DOT
    '\u221a'    #  0xFB -> SQUARE ROOT
    '\u207f'    #  0xFC -> SUPERSCRIPT LATIN SMALL LETTER N
    '\xb2'      #  0xFD -> SUPERSCRIPT TWO
    '\u25a0'    #  0xFE -> BLACK SQUARE
    '\xa0'      #  0xFF -> NO-BREAK SPACE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec koi8_u generated from 'python-mappings/KOI8-U.TXT' with gencodec.py.

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
        name='koi8-u',
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
    '\u2500'   #  0x80 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u2502'   #  0x81 -> BOX DRAWINGS LIGHT VERTICAL
    '\u250c'   #  0x82 -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2510'   #  0x83 -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x84 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2518'   #  0x85 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u251c'   #  0x86 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2524'   #  0x87 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u252c'   #  0x88 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u2534'   #  0x89 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u253c'   #  0x8A -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u2580'   #  0x8B -> UPPER HALF BLOCK
    '\u2584'   #  0x8C -> LOWER HALF BLOCK
    '\u2588'   #  0x8D -> FULL BLOCK
    '\u258c'   #  0x8E -> LEFT HALF BLOCK
    '\u2590'   #  0x8F -> RIGHT HALF BLOCK
    '\u2591'   #  0x90 -> LIGHT SHADE
    '\u2592'   #  0x91 -> MEDIUM SHADE
    '\u2593'   #  0x92 -> DARK SHADE
    '\u2320'   #  0x93 -> TOP HALF INTEGRAL
    '\u25a0'   #  0x94 -> BLACK SQUARE
    '\u2219'   #  0x95 -> BULLET OPERATOR
    '\u221a'   #  0x96 -> SQUARE ROOT
    '\u2248'   #  0x97 -> ALMOST EQUAL TO
    '\u2264'   #  0x98 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0x99 -> GREATER-THAN OR EQUAL TO
    '\xa0'     #  0x9A -> NO-BREAK SPACE
    '\u2321'   #  0x9B -> BOTTOM HALF INTEGRAL
    '\xb0'     #  0x9C -> DEGREE SIGN
    '\xb2'     #  0x9D -> SUPERSCRIPT TWO
    '\xb7'     #  0x9E -> MIDDLE DOT
    '\xf7'     #  0x9F -> DIVISION SIGN
    '\u2550'   #  0xA0 -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u2551'   #  0xA1 -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2552'   #  0xA2 -> BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    '\u0451'   #  0xA3 -> CYRILLIC SMALL LETTER IO
    '\u0454'   #  0xA4 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u2554'   #  0xA5 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u0456'   #  0xA6 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0457'   #  0xA7 -> CYRILLIC SMALL LETTER YI (UKRAINIAN)
    '\u2557'   #  0xA8 -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u2558'   #  0xA9 -> BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    '\u2559'   #  0xAA -> BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    '\u255a'   #  0xAB -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u255b'   #  0xAC -> BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    '\u0491'   #  0xAD -> CYRILLIC SMALL LETTER UKRAINIAN GHE WITH UPTURN
    '\u255d'   #  0xAE -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u255e'   #  0xAF -> BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    '\u255f'   #  0xB0 -> BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    '\u2560'   #  0xB1 -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2561'   #  0xB2 -> BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    '\u0401'   #  0xB3 -> CYRILLIC CAPITAL LETTER IO
    '\u0404'   #  0xB4 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u2563'   #  0xB5 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u0406'   #  0xB6 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0407'   #  0xB7 -> CYRILLIC CAPITAL LETTER YI (UKRAINIAN)
    '\u2566'   #  0xB8 -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2567'   #  0xB9 -> BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    '\u2568'   #  0xBA -> BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    '\u2569'   #  0xBB -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u256a'   #  0xBC -> BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    '\u0490'   #  0xBD -> CYRILLIC CAPITAL LETTER UKRAINIAN GHE WITH UPTURN
    '\u256c'   #  0xBE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
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
<|endoftext|>#
# iso2022_jp.py: Python Unicode Codec for ISO2022_JP
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_iso2022, codecs
import _multibytecodec as mbc

codec = _codecs_iso2022.getcodec('iso2022_jp')

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
        name='iso2022_jp',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>"""
Python 'utf-32' Codec
"""
import codecs, sys

### Codec APIs

encode = codecs.utf_32_encode

def decode(input, errors='strict'):
    return codecs.utf_32_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors='strict'):
        codecs.IncrementalEncoder.__init__(self, errors)
        self.encoder = None

    def encode(self, input, final=False):
        if self.encoder is None:
            result = codecs.utf_32_encode(input, self.errors)[0]
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_32_le_encode
            else:
                self.encoder = codecs.utf_32_be_encode
            return result
        return self.encoder(input, self.errors)[0]

    def reset(self):
        codecs.IncrementalEncoder.reset(self)
        self.encoder = None

    def getstate(self):
        # state info we return to the caller:
        # 0: stream is in natural order for this platform
        # 2: endianness hasn't been determined yet
        # (we're never writing in unnatural order)
        return (2 if self.encoder is None else 0)

    def setstate(self, state):
        if state:
            self.encoder = None
        else:
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_32_le_encode
            else:
                self.encoder = codecs.utf_32_be_encode

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def __init__(self, errors='strict'):
        codecs.BufferedIncrementalDecoder.__init__(self, errors)
        self.decoder = None

    def _buffer_decode(self, input, errors, final):
        if self.decoder is None:
            (output, consumed, byteorder) = \
                codecs.utf_32_ex_decode(input, errors, 0, final)
            if byteorder == -1:
                self.decoder = codecs.utf_32_le_decode
            elif byteorder == 1:
                self.decoder = codecs.utf_32_be_decode
            elif consumed >= 4:
                raise UnicodeError("UTF-32 stream does not start with BOM")
            return (output, consumed)
        return self.decoder(input, self.errors, final)

    def reset(self):
        codecs.BufferedIncrementalDecoder.reset(self)
        self.decoder = None

    def getstate(self):
        # additional state info from the base class must be None here,
        # as it isn't passed along to the caller
        state = codecs.BufferedIncrementalDecoder.getstate(self)[0]
        # additional state info we pass to the caller:
        # 0: stream is in natural order for this platform
        # 1: stream is in unnatural order
        # 2: endianness hasn't been determined yet
        if self.decoder is None:
            return (state, 2)
        addstate = int((sys.byteorder == "big") !=
                       (self.decoder is codecs.utf_32_be_decode))
        return (state, addstate)

    def setstate(self, state):
        # state[1] will be ignored by BufferedIncrementalDecoder.setstate()
        codecs.BufferedIncrementalDecoder.setstate(self, state)
        state = state[1]
        if state == 0:
            self.decoder = (codecs.utf_32_be_decode
                            if sys.byteorder == "big"
                            else codecs.utf_32_le_decode)
        elif state == 1:
            self.decoder = (codecs.utf_32_le_decode
                            if sys.byteorder == "big"
                            else codecs.utf_32_be_decode)
        else:
            self.decoder = None

class StreamWriter(codecs.StreamWriter):
    def __init__(self, stream, errors='strict'):
        self.encoder = None
        codecs.StreamWriter.__init__(self, stream, errors)

    def reset(self):
        codecs.StreamWriter.reset(self)
        self.encoder = None

    def encode(self, input, errors='strict'):
        if self.encoder is None:
            result = codecs.utf_32_encode(input, errors)
            if sys.byteorder == 'little':
                self.encoder = codecs.utf_32_le_encode
            else:
                self.encoder = codecs.utf_32_be_encode
            return result
        else:
            return self.encoder(input, errors)

class StreamReader(codecs.StreamReader):

    def reset(self):
        codecs.StreamReader.reset(self)
        try:
            del self.decode
        except AttributeError:
            pass

    def decode(self, input, errors='strict'):
        (object, consumed, byteorder) = \
            codecs.utf_32_ex_decode(input, errors, 0, False)
        if byteorder == -1:
            self.decode = codecs.utf_32_le_decode
        elif byteorder == 1:
            self.decode = codecs.utf_32_be_decode
        elif consumed>=4:
            raise UnicodeError("UTF-32 stream does not start with BOM")
        return (object, consumed)

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='utf-32',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec cp437 generated from 'VENDORS/MICSFT/PC/CP437.TXT' with gencodec.py.

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
        name='cp437',
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
    0x0080: 0x00c7,     #  LATIN CAPITAL LETTER C WITH CEDILLA
    0x0081: 0x00fc,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x0082: 0x00e9,     #  LATIN SMALL LETTER E WITH ACUTE
    0x0083: 0x00e2,     #  LATIN SMALL LETTER A WITH CIRCUMFLEX
    0x0084: 0x00e4,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x0085: 0x00e0,     #  LATIN SMALL LETTER A WITH GRAVE
    0x0086: 0x00e5,     #  LATIN SMALL LETTER A WITH RING ABOVE
    0x0087: 0x00e7,     #  LATIN SMALL LETTER C WITH CEDILLA
    0x0088: 0x00ea,     #  LATIN SMALL LETTER E WITH CIRCUMFLEX
    0x0089: 0x00eb,     #  LATIN SMALL LETTER E WITH DIAERESIS
    0x008a: 0x00e8,     #  LATIN SMALL LETTER E WITH GRAVE
    0x008b: 0x00ef,     #  LATIN SMALL LETTER I WITH DIAERESIS
    0x008c: 0x00ee,     #  LATIN SMALL LETTER I WITH CIRCUMFLEX
    0x008d: 0x00ec,     #  LATIN SMALL LETTER I WITH GRAVE
    0x008e: 0x00c4,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x008f: 0x00c5,     #  LATIN CAPITAL LETTER A WITH RING ABOVE
    0x0090: 0x00c9,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x0091: 0x00e6,     #  LATIN SMALL LIGATURE AE
    0x0092: 0x00c6,     #  LATIN CAPITAL LIGATURE AE
    0x0093: 0x00f4,     #  LATIN SMALL LETTER O WITH CIRCUMFLEX
    0x0094: 0x00f6,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x0095: 0x00f2,     #  LATIN SMALL LETTER O WITH GRAVE
    0x0096: 0x00fb,     #  LATIN SMALL LETTER U WITH CIRCUMFLEX
    0x0097: 0x00f9,     #  LATIN SMALL LETTER U WITH GRAVE
    0x0098: 0x00ff,     #  LATIN SMALL LETTER Y WITH DIAERESIS
    0x0099: 0x00d6,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x009a: 0x00dc,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x009b: 0x00a2,     #  CENT SIGN
    0x009c: 0x00a3,     #  POUND SIGN
    0x009d: 0x00a5,     #  YEN SIGN
    0x009e: 0x20a7,     #  PESETA SIGN
    0x009f: 0x0192,     #  LATIN SMALL LETTER F WITH HOOK
    0x00a0: 0x00e1,     #  LATIN SMALL LETTER A WITH ACUTE
    0x00a1: 0x00ed,     #  LATIN SMALL LETTER I WITH ACUTE
    0x00a2: 0x00f3,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00a3: 0x00fa,     #  LATIN SMALL LETTER U WITH ACUTE
    0x00a4: 0x00f1,     #  LATIN SMALL LETTER N WITH TILDE
    0x00a5: 0x00d1,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x00a6: 0x00aa,     #  FEMININE ORDINAL INDICATOR
    0x00a7: 0x00ba,     #  MASCULINE ORDINAL INDICATOR
    0x00a8: 0x00bf,     #  INVERTED QUESTION MARK
    0x00a9: 0x2310,     #  REVERSED NOT SIGN
    0x00aa: 0x00ac,     #  NOT SIGN
    0x00ab: 0x00bd,     #  VULGAR FRACTION ONE HALF
    0x00ac: 0x00bc,     #  VULGAR FRACTION ONE QUARTER
    0x00ad: 0x00a1,     #  INVERTED EXCLAMATION MARK
    0x00ae: 0x00ab,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00af: 0x00bb,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
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
    0x00e0: 0x03b1,     #  GREEK SMALL LETTER ALPHA
    0x00e1: 0x00df,     #  LATIN SMALL LETTER SHARP S
    0x00e2: 0x0393,     #  GREEK CAPITAL LETTER GAMMA
    0x00e3: 0x03c0,     #  GREEK SMALL LETTER PI
    0x00e4: 0x03a3,     #  GREEK CAPITAL LETTER SIGMA
    0x00e5: 0x03c3,     #  GREEK SMALL LETTER SIGMA
    0x00e6: 0x00b5,     #  MICRO SIGN
    0x00e7: 0x03c4,     #  GREEK SMALL LETTER TAU
    0x00e8: 0x03a6,     #  GREEK CAPITAL LETTER PHI
    0x00e9: 0x0398,     #  GREEK CAPITAL LETTER THETA
    0x00ea: 0x03a9,     #  GREEK CAPITAL LETTER OMEGA
    0x00eb: 0x03b4,     #  GREEK SMALL LETTER DELTA
    0x00ec: 0x221e,     #  INFINITY
    0x00ed: 0x03c6,     #  GREEK SMALL LETTER PHI
    0x00ee: 0x03b5,     #  GREEK SMALL LETTER EPSILON
    0x00ef: 0x2229,     #  INTERSECTION
    0x00f0: 0x2261,     #  IDENTICAL TO
    0x00f1: 0x00b1,     #  PLUS-MINUS SIGN
    0x00f2: 0x2265,     #  GREATER-THAN OR EQUAL TO
    0x00f3: 0x2264,     #  LESS-THAN OR EQUAL TO
    0x00f4: 0x2320,     #  TOP HALF INTEGRAL
    0x00f5: 0x2321,     #  BOTTOM HALF INTEGRAL
    0x00f6: 0x00f7,     #  DIVISION SIGN
    0x00f7: 0x2248,     #  ALMOST EQUAL TO
    0x00f8: 0x00b0,     #  DEGREE SIGN
    0x00f9: 0x2219,     #  BULLET OPERATOR
    0x00fa: 0x00b7,     #  MIDDLE DOT
    0x00fb: 0x221a,     #  SQUARE ROOT
    0x00fc: 0x207f,     #  SUPERSCRIPT LATIN SMALL LETTER N
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
    '\xc7'     #  0x0080 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xfc'     #  0x0081 -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xe9'     #  0x0082 -> LATIN SMALL LETTER E WITH ACUTE
    '\xe2'     #  0x0083 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x0084 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe0'     #  0x0085 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe5'     #  0x0086 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe7'     #  0x0087 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xea'     #  0x0088 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x0089 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xe8'     #  0x008a -> LATIN SMALL LETTER E WITH GRAVE
    '\xef'     #  0x008b -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xee'     #  0x008c -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xec'     #  0x008d -> LATIN SMALL LETTER I WITH GRAVE
    '\xc4'     #  0x008e -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0x008f -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc9'     #  0x0090 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xe6'     #  0x0091 -> LATIN SMALL LIGATURE AE
    '\xc6'     #  0x0092 -> LATIN CAPITAL LIGATURE AE
    '\xf4'     #  0x0093 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x0094 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf2'     #  0x0095 -> LATIN SMALL LETTER O WITH GRAVE
    '\xfb'     #  0x0096 -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xf9'     #  0x0097 -> LATIN SMALL LETTER U WITH GRAVE
    '\xff'     #  0x0098 -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\xd6'     #  0x0099 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x009a -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xa2'     #  0x009b -> CENT SIGN
    '\xa3'     #  0x009c -> POUND SIGN
    '\xa5'     #  0x009d -> YEN SIGN
    '\u20a7'   #  0x009e -> PESETA SIGN
    '\u0192'   #  0x009f -> LATIN SMALL LETTER F WITH HOOK
    '\xe1'     #  0x00a0 -> LATIN SMALL LETTER A WITH ACUTE
    '\xed'     #  0x00a1 -> LATIN SMALL LETTER I WITH ACUTE
    '\xf3'     #  0x00a2 -> LATIN SMALL LETTER O WITH ACUTE
    '\xfa'     #  0x00a3 -> LATIN SMALL LETTER U WITH ACUTE
    '\xf1'     #  0x00a4 -> LATIN SMALL LETTER N WITH TILDE
    '\xd1'     #  0x00a5 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xaa'     #  0x00a6 -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0x00a7 -> MASCULINE ORDINAL INDICATOR
    '\xbf'     #  0x00a8 -> INVERTED QUESTION MARK
    '\u2310'   #  0x00a9 -> REVERSED NOT SIGN
    '\xac'     #  0x00aa -> NOT SIGN
    '\xbd'     #  0x00ab -> VULGAR FRACTION ONE HALF
    '\xbc'     #  0x00ac -> VULGAR FRACTION ONE QUARTER
    '\xa1'     #  0x00ad -> INVERTED EXCLAMATION MARK
    '\xab'     #  0x00ae -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0x00af -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
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
    '\u03b1'   #  0x00e0 -> GREEK SMALL LETTER ALPHA
    '\xdf'     #  0x00e1 -> LATIN SMALL LETTER SHARP S
    '\u0393'   #  0x00e2 -> GREEK CAPITAL LETTER GAMMA
    '\u03c0'   #  0x00e3 -> GREEK SMALL LETTER PI
    '\u03a3'   #  0x00e4 -> GREEK CAPITAL LETTER SIGMA
    '\u03c3'   #  0x00e5 -> GREEK SMALL LETTER SIGMA
    '\xb5'     #  0x00e6 -> MICRO SIGN
    '\u03c4'   #  0x00e7 -> GREEK SMALL LETTER TAU
    '\u03a6'   #  0x00e8 -> GREEK CAPITAL LETTER PHI
    '\u0398'   #  0x00e9 -> GREEK CAPITAL LETTER THETA
    '\u03a9'   #  0x00ea -> GREEK CAPITAL LETTER OMEGA
    '\u03b4'   #  0x00eb -> GREEK SMALL LETTER DELTA
    '\u221e'   #  0x00ec -> INFINITY
    '\u03c6'   #  0x00ed -> GREEK SMALL LETTER PHI
    '\u03b5'   #  0x00ee -> GREEK SMALL LETTER EPSILON
    '\u2229'   #  0x00ef -> INTERSECTION
    '\u2261'   #  0x00f0 -> IDENTICAL TO
    '\xb1'     #  0x00f1 -> PLUS-MINUS SIGN
    '\u2265'   #  0x00f2 -> GREATER-THAN OR EQUAL TO
