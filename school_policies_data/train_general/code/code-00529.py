### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec cp500 generated from 'MAPPINGS/VENDORS/MICSFT/EBCDIC/CP500.TXT' with gencodec.py.

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
        name='cp500',
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
    '['        #  0x4A -> LEFT SQUARE BRACKET
    '.'        #  0x4B -> FULL STOP
    '<'        #  0x4C -> LESS-THAN SIGN
    '('        #  0x4D -> LEFT PARENTHESIS
    '+'        #  0x4E -> PLUS SIGN
    '!'        #  0x4F -> EXCLAMATION MARK
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
    ']'        #  0x5A -> RIGHT SQUARE BRACKET
    '$'        #  0x5B -> DOLLAR SIGN
    '*'        #  0x5C -> ASTERISK
    ')'        #  0x5D -> RIGHT PARENTHESIS
    ';'        #  0x5E -> SEMICOLON
    '^'        #  0x5F -> CIRCUMFLEX ACCENT
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
    '\xa4'     #  0x9F -> CURRENCY SIGN
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
    '\xa2'     #  0xB0 -> CENT SIGN
    '\xa3'     #  0xB1 -> POUND SIGN
    '\xa5'     #  0xB2 -> YEN SIGN
    '\xb7'     #  0xB3 -> MIDDLE DOT
    '\xa9'     #  0xB4 -> COPYRIGHT SIGN
    '\xa7'     #  0xB5 -> SECTION SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xbc'     #  0xB7 -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xB8 -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xB9 -> VULGAR FRACTION THREE QUARTERS
    '\xac'     #  0xBA -> NOT SIGN
    '|'        #  0xBB -> VERTICAL LINE
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
<|endoftext|>""" Python Character Mapping Codec for PalmOS 3.5.

Written by Sjoerd Mullender (sjoerd@acm.org); based on iso8859_15.py.

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
        name='palmos',
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
    '\x81'     #  0x81 -> <control>
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
    '\u2666'   #  0x8D -> BLACK DIAMOND SUIT
    '\u2663'   #  0x8E -> BLACK CLUB SUIT
    '\u2665'   #  0x8F -> BLACK HEART SUIT
    '\u2660'   #  0x90 -> BLACK SPADE SUIT
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
    '\x9b'     #  0x9B -> <control>
    '\u0153'   #  0x9C -> LATIN SMALL LIGATURE OE
    '\x9d'     #  0x9D -> <control>
    '\x9e'     #  0x9E -> <control>
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
    '\xd0'     #  0xD0 -> LATIN CAPITAL LETTER ETH (Icelandic)
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
    '\xde'     #  0xDE -> LATIN CAPITAL LETTER THORN (Icelandic)
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S (German)
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
    '\xf0'     #  0xF0 -> LATIN SMALL LETTER ETH (Icelandic)
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
    '\xfe'     #  0xFE -> LATIN SMALL LETTER THORN (Icelandic)
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec tis_620 generated from 'python-mappings/TIS-620.TXT' with gencodec.py.

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
        name='tis-620',
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
    '\ufffe'
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
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
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
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec cp1257 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1257.TXT' with gencodec.py.

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
        name='cp1257',
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
    '\ufffe'   #  0x83 -> UNDEFINED
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\ufffe'   #  0x88 -> UNDEFINED
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\ufffe'   #  0x8A -> UNDEFINED
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x8C -> UNDEFINED
    '\xa8'     #  0x8D -> DIAERESIS
    '\u02c7'   #  0x8E -> CARON
    '\xb8'     #  0x8F -> CEDILLA
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
    '\xaf'     #  0x9D -> MACRON
    '\u02db'   #  0x9E -> OGONEK
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\ufffe'   #  0xA1 -> UNDEFINED
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\ufffe'   #  0xA5 -> UNDEFINED
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xd8'     #  0xA8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u0156'   #  0xAA -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\xc6'     #  0xAF -> LATIN CAPITAL LETTER AE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\xf8'     #  0xB8 -> LATIN SMALL LETTER O WITH STROKE
    '\xb9'     #  0xB9 -> SUPERSCRIPT ONE
    '\u0157'   #  0xBA -> LATIN SMALL LETTER R WITH CEDILLA
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xe6'     #  0xBF -> LATIN SMALL LETTER AE
    '\u0104'   #  0xC0 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\u012e'   #  0xC1 -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u0100'   #  0xC2 -> LATIN CAPITAL LETTER A WITH MACRON
    '\u0106'   #  0xC3 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\u0118'   #  0xC6 -> LATIN CAPITAL LETTER E WITH OGONEK
    '\u0112'   #  0xC7 -> LATIN CAPITAL LETTER E WITH MACRON
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0179'   #  0xCA -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\u0116'   #  0xCB -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\u0122'   #  0xCC -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u0136'   #  0xCD -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\u012a'   #  0xCE -> LATIN CAPITAL LETTER I WITH MACRON
    '\u013b'   #  0xCF -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\u0160'   #  0xD0 -> LATIN CAPITAL LETTER S WITH CARON
    '\u0143'   #  0xD1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\u0145'   #  0xD2 -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\u014c'   #  0xD4 -> LATIN CAPITAL LETTER O WITH MACRON
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0172'   #  0xD8 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\u0141'   #  0xD9 -> LATIN CAPITAL LETTER L WITH STROKE
    '\u015a'   #  0xDA -> LATIN CAPITAL LETTER S WITH ACUTE
    '\u016a'   #  0xDB -> LATIN CAPITAL LETTER U WITH MACRON
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u017b'   #  0xDD -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\u017d'   #  0xDE -> LATIN CAPITAL LETTER Z WITH CARON
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\u0105'   #  0xE0 -> LATIN SMALL LETTER A WITH OGONEK
    '\u012f'   #  0xE1 -> LATIN SMALL LETTER I WITH OGONEK
    '\u0101'   #  0xE2 -> LATIN SMALL LETTER A WITH MACRON
    '\u0107'   #  0xE3 -> LATIN SMALL LETTER C WITH ACUTE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\u0119'   #  0xE6 -> LATIN SMALL LETTER E WITH OGONEK
    '\u0113'   #  0xE7 -> LATIN SMALL LETTER E WITH MACRON
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u017a'   #  0xEA -> LATIN SMALL LETTER Z WITH ACUTE
    '\u0117'   #  0xEB -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\u0123'   #  0xEC -> LATIN SMALL LETTER G WITH CEDILLA
    '\u0137'   #  0xED -> LATIN SMALL LETTER K WITH CEDILLA
    '\u012b'   #  0xEE -> LATIN SMALL LETTER I WITH MACRON
    '\u013c'   #  0xEF -> LATIN SMALL LETTER L WITH CEDILLA
    '\u0161'   #  0xF0 -> LATIN SMALL LETTER S WITH CARON
    '\u0144'   #  0xF1 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0146'   #  0xF2 -> LATIN SMALL LETTER N WITH CEDILLA
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\u014d'   #  0xF4 -> LATIN SMALL LETTER O WITH MACRON
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0173'   #  0xF8 -> LATIN SMALL LETTER U WITH OGONEK
    '\u0142'   #  0xF9 -> LATIN SMALL LETTER L WITH STROKE
    '\u015b'   #  0xFA -> LATIN SMALL LETTER S WITH ACUTE
    '\u016b'   #  0xFB -> LATIN SMALL LETTER U WITH MACRON
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u017c'   #  0xFD -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u017e'   #  0xFE -> LATIN SMALL LETTER Z WITH CARON
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec generated from 'VENDORS/MICSFT/PC/CP857.TXT' with gencodec.py.

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
        name='cp857',
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
    0x008d: 0x0131,     #  LATIN SMALL LETTER DOTLESS I
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
    0x0098: 0x0130,     #  LATIN CAPITAL LETTER I WITH DOT ABOVE
    0x0099: 0x00d6,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x009a: 0x00dc,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x009b: 0x00f8,     #  LATIN SMALL LETTER O WITH STROKE
    0x009c: 0x00a3,     #  POUND SIGN
    0x009d: 0x00d8,     #  LATIN CAPITAL LETTER O WITH STROKE
    0x009e: 0x015e,     #  LATIN CAPITAL LETTER S WITH CEDILLA
    0x009f: 0x015f,     #  LATIN SMALL LETTER S WITH CEDILLA
    0x00a0: 0x00e1,     #  LATIN SMALL LETTER A WITH ACUTE
    0x00a1: 0x00ed,     #  LATIN SMALL LETTER I WITH ACUTE
    0x00a2: 0x00f3,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00a3: 0x00fa,     #  LATIN SMALL LETTER U WITH ACUTE
    0x00a4: 0x00f1,     #  LATIN SMALL LETTER N WITH TILDE
    0x00a5: 0x00d1,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x00a6: 0x011e,     #  LATIN CAPITAL LETTER G WITH BREVE
    0x00a7: 0x011f,     #  LATIN SMALL LETTER G WITH BREVE
    0x00a8: 0x00bf,     #  INVERTED QUESTION MARK
    0x00a9: 0x00ae,     #  REGISTERED SIGN
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
    0x00b5: 0x00c1,     #  LATIN CAPITAL LETTER A WITH ACUTE
    0x00b6: 0x00c2,     #  LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    0x00b7: 0x00c0,     #  LATIN CAPITAL LETTER A WITH GRAVE
    0x00b8: 0x00a9,     #  COPYRIGHT SIGN
    0x00b9: 0x2563,     #  BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    0x00ba: 0x2551,     #  BOX DRAWINGS DOUBLE VERTICAL
    0x00bb: 0x2557,     #  BOX DRAWINGS DOUBLE DOWN AND LEFT
    0x00bc: 0x255d,     #  BOX DRAWINGS DOUBLE UP AND LEFT
    0x00bd: 0x00a2,     #  CENT SIGN
    0x00be: 0x00a5,     #  YEN SIGN
    0x00bf: 0x2510,     #  BOX DRAWINGS LIGHT DOWN AND LEFT
    0x00c0: 0x2514,     #  BOX DRAWINGS LIGHT UP AND RIGHT
    0x00c1: 0x2534,     #  BOX DRAWINGS LIGHT UP AND HORIZONTAL
    0x00c2: 0x252c,     #  BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    0x00c3: 0x251c,     #  BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    0x00c4: 0x2500,     #  BOX DRAWINGS LIGHT HORIZONTAL
    0x00c5: 0x253c,     #  BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    0x00c6: 0x00e3,     #  LATIN SMALL LETTER A WITH TILDE
    0x00c7: 0x00c3,     #  LATIN CAPITAL LETTER A WITH TILDE
    0x00c8: 0x255a,     #  BOX DRAWINGS DOUBLE UP AND RIGHT
    0x00c9: 0x2554,     #  BOX DRAWINGS DOUBLE DOWN AND RIGHT
    0x00ca: 0x2569,     #  BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    0x00cb: 0x2566,     #  BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    0x00cc: 0x2560,     #  BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    0x00cd: 0x2550,     #  BOX DRAWINGS DOUBLE HORIZONTAL
    0x00ce: 0x256c,     #  BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    0x00cf: 0x00a4,     #  CURRENCY SIGN
    0x00d0: 0x00ba,     #  MASCULINE ORDINAL INDICATOR
    0x00d1: 0x00aa,     #  FEMININE ORDINAL INDICATOR
    0x00d2: 0x00ca,     #  LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    0x00d3: 0x00cb,     #  LATIN CAPITAL LETTER E WITH DIAERESIS
    0x00d4: 0x00c8,     #  LATIN CAPITAL LETTER E WITH GRAVE
    0x00d5: None,       #  UNDEFINED
    0x00d6: 0x00cd,     #  LATIN CAPITAL LETTER I WITH ACUTE
    0x00d7: 0x00ce,     #  LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    0x00d8: 0x00cf,     #  LATIN CAPITAL LETTER I WITH DIAERESIS
    0x00d9: 0x2518,     #  BOX DRAWINGS LIGHT UP AND LEFT
    0x00da: 0x250c,     #  BOX DRAWINGS LIGHT DOWN AND RIGHT
    0x00db: 0x2588,     #  FULL BLOCK
    0x00dc: 0x2584,     #  LOWER HALF BLOCK
    0x00dd: 0x00a6,     #  BROKEN BAR
    0x00de: 0x00cc,     #  LATIN CAPITAL LETTER I WITH GRAVE
    0x00df: 0x2580,     #  UPPER HALF BLOCK
    0x00e0: 0x00d3,     #  LATIN CAPITAL LETTER O WITH ACUTE
    0x00e1: 0x00df,     #  LATIN SMALL LETTER SHARP S
    0x00e2: 0x00d4,     #  LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    0x00e3: 0x00d2,     #  LATIN CAPITAL LETTER O WITH GRAVE
    0x00e4: 0x00f5,     #  LATIN SMALL LETTER O WITH TILDE
    0x00e5: 0x00d5,     #  LATIN CAPITAL LETTER O WITH TILDE
    0x00e6: 0x00b5,     #  MICRO SIGN
    0x00e7: None,       #  UNDEFINED
    0x00e8: 0x00d7,     #  MULTIPLICATION SIGN
    0x00e9: 0x00da,     #  LATIN CAPITAL LETTER U WITH ACUTE
    0x00ea: 0x00db,     #  LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    0x00eb: 0x00d9,     #  LATIN CAPITAL LETTER U WITH GRAVE
    0x00ed: 0x00ff,     #  LATIN SMALL LETTER Y WITH DIAERESIS
    0x00ee: 0x00af,     #  MACRON
    0x00ef: 0x00b4,     #  ACUTE ACCENT
    0x00f0: 0x00ad,     #  SOFT HYPHEN
    0x00f1: 0x00b1,     #  PLUS-MINUS SIGN
    0x00f2: None,       #  UNDEFINED
    0x00f3: 0x00be,     #  VULGAR FRACTION THREE QUARTERS
    0x00f4: 0x00b6,     #  PILCROW SIGN
    0x00f5: 0x00a7,     #  SECTION SIGN
    0x00f6: 0x00f7,     #  DIVISION SIGN
    0x00f7: 0x00b8,     #  CEDILLA
    0x00f8: 0x00b0,     #  DEGREE SIGN
    0x00f9: 0x00a8,     #  DIAERESIS
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
    '\u0131'   #  0x008d -> LATIN SMALL LETTER DOTLESS I
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
    '\u0130'   #  0x0098 -> LATIN CAPITAL LETTER I WITH DOT ABOVE
    '\xd6'     #  0x0099 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x009a -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xf8'     #  0x009b -> LATIN SMALL LETTER O WITH STROKE
    '\xa3'     #  0x009c -> POUND SIGN
    '\xd8'     #  0x009d -> LATIN CAPITAL LETTER O WITH STROKE
    '\u015e'   #  0x009e -> LATIN CAPITAL LETTER S WITH CEDILLA
    '\u015f'   #  0x009f -> LATIN SMALL LETTER S WITH CEDILLA
    '\xe1'     #  0x00a0 -> LATIN SMALL LETTER A WITH ACUTE
    '\xed'     #  0x00a1 -> LATIN SMALL LETTER I WITH ACUTE
    '\xf3'     #  0x00a2 -> LATIN SMALL LETTER O WITH ACUTE
    '\xfa'     #  0x00a3 -> LATIN SMALL LETTER U WITH ACUTE
    '\xf1'     #  0x00a4 -> LATIN SMALL LETTER N WITH TILDE
    '\xd1'     #  0x00a5 -> LATIN CAPITAL LETTER N WITH TILDE
    '\u011e'   #  0x00a6 -> LATIN CAPITAL LETTER G WITH BREVE
    '\u011f'   #  0x00a7 -> LATIN SMALL LETTER G WITH BREVE
    '\xbf'     #  0x00a8 -> INVERTED QUESTION MARK
    '\xae'     #  0x00a9 -> REGISTERED SIGN
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
    '\xc1'     #  0x00b5 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0x00b6 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc0'     #  0x00b7 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xa9'     #  0x00b8 -> COPYRIGHT SIGN
    '\u2563'   #  0x00b9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'   #  0x00ba -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'   #  0x00bb -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'   #  0x00bc -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\xa2'     #  0x00bd -> CENT SIGN
    '\xa5'     #  0x00be -> YEN SIGN
    '\u2510'   #  0x00bf -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x00c0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'   #  0x00c1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'   #  0x00c2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0x00c3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'   #  0x00c4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'   #  0x00c5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\xe3'     #  0x00c6 -> LATIN SMALL LETTER A WITH TILDE
    '\xc3'     #  0x00c7 -> LATIN CAPITAL LETTER A WITH TILDE
    '\u255a'   #  0x00c8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'   #  0x00c9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'   #  0x00ca -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'   #  0x00cb -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'   #  0x00cc -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'   #  0x00cd -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'   #  0x00ce -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\xa4'     #  0x00cf -> CURRENCY SIGN
    '\xba'     #  0x00d0 -> MASCULINE ORDINAL INDICATOR
    '\xaa'     #  0x00d1 -> FEMININE ORDINAL INDICATOR
    '\xca'     #  0x00d2 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0x00d3 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xc8'     #  0x00d4 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\ufffe'   #  0x00d5 -> UNDEFINED
    '\xcd'     #  0x00d6 -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0x00d7 -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0x00d8 -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\u2518'   #  0x00d9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'   #  0x00da -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'   #  0x00db -> FULL BLOCK
    '\u2584'   #  0x00dc -> LOWER HALF BLOCK
    '\xa6'     #  0x00dd -> BROKEN BAR
    '\xcc'     #  0x00de -> LATIN CAPITAL LETTER I WITH GRAVE
    '\u2580'   #  0x00df -> UPPER HALF BLOCK
    '\xd3'     #  0x00e0 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xdf'     #  0x00e1 -> LATIN SMALL LETTER SHARP S
    '\xd4'     #  0x00e2 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd2'     #  0x00e3 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xf5'     #  0x00e4 -> LATIN SMALL LETTER O WITH TILDE
    '\xd5'     #  0x00e5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xb5'     #  0x00e6 -> MICRO SIGN
    '\ufffe'   #  0x00e7 -> UNDEFINED
    '\xd7'     #  0x00e8 -> MULTIPLICATION SIGN
    '\xda'     #  0x00e9 -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0x00ea -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xd9'     #  0x00eb -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xec'     #  0x00ec -> LATIN SMALL LETTER I WITH GRAVE
    '\xff'     #  0x00ed -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\xaf'     #  0x00ee -> MACRON
    '\xb4'     #  0x00ef -> ACUTE ACCENT
    '\xad'     #  0x00f0 -> SOFT HYPHEN
    '\xb1'     #  0x00f1 -> PLUS-MINUS SIGN
    '\ufffe'   #  0x00f2 -> UNDEFINED
    '\xbe'     #  0x00f3 -> VULGAR FRACTION THREE QUARTERS
    '\xb6'     #  0x00f4 -> PILCROW SIGN
    '\xa7'     #  0x00f5 -> SECTION SIGN
    '\xf7'     #  0x00f6 -> DIVISION SIGN
    '\xb8'     #  0x00f7 -> CEDILLA
    '\xb0'     #  0x00f8 -> DEGREE SIGN
    '\xa8'     #  0x00f9 -> DIAERESIS
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
    0x00a1: 0x00ad,     #  INVERTED EXCLAMATION MARK
    0x00a2: 0x00bd,     #  CENT SIGN
    0x00a3: 0x009c,     #  POUND SIGN
    0x00a4: 0x00cf,     #  CURRENCY SIGN
    0x00a5: 0x00be,     #  YEN SIGN
    0x00a6: 0x00dd,     #  BROKEN BAR
    0x00a7: 0x00f5,     #  SECTION SIGN
    0x00a8: 0x00f9,     #  DIAERESIS
    0x00a9: 0x00b8,     #  COPYRIGHT SIGN
    0x00aa: 0x00d1,     #  FEMININE ORDINAL INDICATOR
    0x00ab: 0x00ae,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00ac: 0x00aa,     #  NOT SIGN
    0x00ad: 0x00f0,     #  SOFT HYPHEN
    0x00ae: 0x00a9,     #  REGISTERED SIGN
    0x00af: 0x00ee,     #  MACRON
    0x00b0: 0x00f8,     #  DEGREE SIGN
    0x00b1: 0x00f1,     #  PLUS-MINUS SIGN
    0x00b2: 0x00fd,     #  SUPERSCRIPT TWO
    0x00b3: 0x00fc,     #  SUPERSCRIPT THREE
    0x00b4: 0x00ef,     #  ACUTE ACCENT
    0x00b5: 0x00e6,     #  MICRO SIGN
    0x00b6: 0x00f4,     #  PILCROW SIGN
    0x00b7: 0x00fa,     #  MIDDLE DOT
    0x00b8: 0x00f7,     #  CEDILLA
    0x00b9: 0x00fb,     #  SUPERSCRIPT ONE
    0x00ba: 0x00d0,     #  MASCULINE ORDINAL INDICATOR
    0x00bb: 0x00af,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    0x00bc: 0x00ac,     #  VULGAR FRACTION ONE QUARTER
    0x00bd: 0x00ab,     #  VULGAR FRACTION ONE HALF
    0x00be: 0x00f3,     #  VULGAR FRACTION THREE QUARTERS
    0x00bf: 0x00a8,     #  INVERTED QUESTION MARK
    0x00c0: 0x00b7,     #  LATIN CAPITAL LETTER A WITH GRAVE
    0x00c1: 0x00b5,     #  LATIN CAPITAL LETTER A WITH ACUTE
    0x00c2: 0x00b6,     #  LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    0x00c3: 0x00c7,     #  LATIN CAPITAL LETTER A WITH TILDE
    0x00c4: 0x008e,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x00c5: 0x008f,     #  LATIN CAPITAL LETTER A WITH RING ABOVE
    0x00c6: 0x0092,     #  LATIN CAPITAL LIGATURE AE
    0x00c7: 0x0080,     #  LATIN CAPITAL LETTER C WITH CEDILLA
    0x00c8: 0x00d4,     #  LATIN CAPITAL LETTER E WITH GRAVE
    0x00c9: 0x0090,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x00ca: 0x00d2,     #  LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    0x00cb: 0x00d3,     #  LATIN CAPITAL LETTER E WITH DIAERESIS
    0x00cc: 0x00de,     #  LATIN CAPITAL LETTER I WITH GRAVE
    0x00cd: 0x00d6,     #  LATIN CAPITAL LETTER I WITH ACUTE
    0x00ce: 0x00d7,     #  LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    0x00cf: 0x00d8,     #  LATIN CAPITAL LETTER I WITH DIAERESIS
    0x00d1: 0x00a5,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x00d2: 0x00e3,     #  LATIN CAPITAL LETTER O WITH GRAVE
    0x00d3: 0x00e0,     #  LATIN CAPITAL LETTER O WITH ACUTE
    0x00d4: 0x00e2,     #  LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    0x00d5: 0x00e5,     #  LATIN CAPITAL LETTER O WITH TILDE
    0x00d6: 0x0099,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x00d7: 0x00e8,     #  MULTIPLICATION SIGN
    0x00d8: 0x009d,     #  LATIN CAPITAL LETTER O WITH STROKE
    0x00d9: 0x00eb,     #  LATIN CAPITAL LETTER U WITH GRAVE
    0x00da: 0x00e9,     #  LATIN CAPITAL LETTER U WITH ACUTE
    0x00db: 0x00ea,     #  LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    0x00dc: 0x009a,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x00df: 0x00e1,     #  LATIN SMALL LETTER SHARP S
    0x00e0: 0x0085,     #  LATIN SMALL LETTER A WITH GRAVE
    0x00e1: 0x00a0,     #  LATIN SMALL LETTER A WITH ACUTE
    0x00e2: 0x0083,     #  LATIN SMALL LETTER A WITH CIRCUMFLEX
    0x00e3: 0x00c6,     #  LATIN SMALL LETTER A WITH TILDE
    0x00e4: 0x0084,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x00e5: 0x0086,     #  LATIN SMALL LETTER A WITH RING ABOVE
    0x00e6: 0x0091,     #  LATIN SMALL LIGATURE AE
    0x00e7: 0x0087,     #  LATIN SMALL LETTER C WITH CEDILLA
    0x00e8: 0x008a,     #  LATIN SMALL LETTER E WITH GRAVE
    0x00e9: 0x0082,     #  LATIN SMALL LETTER E WITH ACUTE
    0x00ea: 0x0088,     #  LATIN SMALL LETTER E WITH CIRCUMFLEX
    0x00eb: 0x0089,     #  LATIN SMALL LETTER E WITH DIAERESIS
    0x00ec: 0x00ec,     #  LATIN SMALL LETTER I WITH GRAVE
    0x00ed: 0x00a1,     #  LATIN SMALL LETTER I WITH ACUTE
    0x00ee: 0x008c,     #  LATIN SMALL LETTER I WITH CIRCUMFLEX
    0x00ef: 0x008b,     #  LATIN SMALL LETTER I WITH DIAERESIS
    0x00f1: 0x00a4,     #  LATIN SMALL LETTER N WITH TILDE
    0x00f2: 0x0095,     #  LATIN SMALL LETTER O WITH GRAVE
    0x00f3: 0x00a2,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00f4: 0x0093,     #  LATIN SMALL LETTER O WITH CIRCUMFLEX
    0x00f5: 0x00e4,     #  LATIN SMALL LETTER O WITH TILDE
    0x00f6: 0x0094,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x00f7: 0x00f6,     #  DIVISION SIGN
    0x00f8: 0x009b,     #  LATIN SMALL LETTER O WITH STROKE
    0x00f9: 0x0097,     #  LATIN SMALL LETTER U WITH GRAVE
    0x00fa: 0x00a3,     #  LATIN SMALL LETTER U WITH ACUTE
    0x00fb: 0x0096,     #  LATIN SMALL LETTER U WITH CIRCUMFLEX
    0x00fc: 0x0081,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x00ff: 0x00ed,     #  LATIN SMALL LETTER Y WITH DIAERESIS
    0x011e: 0x00a6,     #  LATIN CAPITAL LETTER G WITH BREVE
    0x011f: 0x00a7,     #  LATIN SMALL LETTER G WITH BREVE
    0x0130: 0x0098,     #  LATIN CAPITAL LETTER I WITH DOT ABOVE
    0x0131: 0x008d,     #  LATIN SMALL LETTER DOTLESS I
    0x015e: 0x009e,     #  LATIN CAPITAL LETTER S WITH CEDILLA
    0x015f: 0x009f,     #  LATIN SMALL LETTER S WITH CEDILLA
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
    0x2591: 0x00b0,     #  LIGHT SHADE
    0x2592: 0x00b1,     #  MEDIUM SHADE
    0x2593: 0x00b2,     #  DARK SHADE
    0x25a0: 0x00fe,     #  BLACK SQUARE
}
<|endoftext|># This module implements the RFCs 3490 (IDNA) and 3491 (Nameprep)

import stringprep, re, codecs
from unicodedata import ucd_3_2_0 as unicodedata

# IDNA section 3.1
dots = re.compile("[\u002E\u3002\uFF0E\uFF61]")

# IDNA section 5
ace_prefix = b"xn--"
sace_prefix = "xn--"

# This assumes query strings, so AllowUnassigned is true
def nameprep(label):
    # Map
    newlabel = []
    for c in label:
        if stringprep.in_table_b1(c):
            # Map to nothing
            continue
        newlabel.append(stringprep.map_table_b2(c))
    label = "".join(newlabel)

    # Normalize
    label = unicodedata.normalize("NFKC", label)

    # Prohibit
    for c in label:
        if stringprep.in_table_c12(c) or \
           stringprep.in_table_c22(c) or \
           stringprep.in_table_c3(c) or \
           stringprep.in_table_c4(c) or \
           stringprep.in_table_c5(c) or \
           stringprep.in_table_c6(c) or \
           stringprep.in_table_c7(c) or \
           stringprep.in_table_c8(c) or \
           stringprep.in_table_c9(c):
            raise UnicodeError("Invalid character %r" % c)

    # Check bidi
    RandAL = [stringprep.in_table_d1(x) for x in label]
    if any(RandAL):
        # There is a RandAL char in the string. Must perform further
        # tests:
        # 1) The characters in section 5.8 MUST be prohibited.
        # This is table C.8, which was already checked
        # 2) If a string contains any RandALCat character, the string
        # MUST NOT contain any LCat character.
        if any(stringprep.in_table_d2(x) for x in label):
            raise UnicodeError("Violation of BIDI requirement 2")
        # 3) If a string contains any RandALCat character, a
        # RandALCat character MUST be the first character of the
        # string, and a RandALCat character MUST be the last
        # character of the string.
        if not RandAL[0] or not RandAL[-1]:
            raise UnicodeError("Violation of BIDI requirement 3")

    return label

def ToASCII(label):
    try:
        # Step 1: try ASCII
        label = label.encode("ascii")
    except UnicodeError:
        pass
    else:
        # Skip to step 3: UseSTD3ASCIIRules is false, so
        # Skip to step 8.
        if 0 < len(label) < 64:
            return label
        raise UnicodeError("label empty or too long")

    # Step 2: nameprep
    label = nameprep(label)

    # Step 3: UseSTD3ASCIIRules is false
    # Step 4: try ASCII
    try:
        label = label.encode("ascii")
    except UnicodeError:
        pass
    else:
        # Skip to step 8.
        if 0 < len(label) < 64:
            return label
        raise UnicodeError("label empty or too long")

    # Step 5: Check ACE prefix
    if label.startswith(sace_prefix):
        raise UnicodeError("Label starts with ACE prefix")

    # Step 6: Encode with PUNYCODE
    label = label.encode("punycode")

    # Step 7: Prepend ACE prefix
    label = ace_prefix + label

    # Step 8: Check size
    if 0 < len(label) < 64:
        return label
    raise UnicodeError("label empty or too long")

def ToUnicode(label):
    # Step 1: Check for ASCII
    if isinstance(label, bytes):
        pure_ascii = True
    else:
        try:
            label = label.encode("ascii")
            pure_ascii = True
        except UnicodeError:
            pure_ascii = False
    if not pure_ascii:
        # Step 2: Perform nameprep
        label = nameprep(label)
        # It doesn't say this, but apparently, it should be ASCII now
        try:
            label = label.encode("ascii")
        except UnicodeError:
            raise UnicodeError("Invalid character in IDN label")
    # Step 3: Check for ACE prefix
    if not label.startswith(ace_prefix):
        return str(label, "ascii")

    # Step 4: Remove ACE prefix
    label1 = label[len(ace_prefix):]

    # Step 5: Decode using PUNYCODE
    result = label1.decode("punycode")

    # Step 6: Apply ToASCII
    label2 = ToASCII(result)

    # Step 7: Compare the result of step 6 with the one of step 3
    # label2 will already be in lower case.
    if str(label, "ascii").lower() != str(label2, "ascii"):
        raise UnicodeError("IDNA does not round-trip", label, label2)

    # Step 8: return the result of step 5
    return result

### Codec APIs

class Codec(codecs.Codec):
    def encode(self, input, errors='strict'):

        if errors != 'strict':
            # IDNA is quite clear that implementations must be strict
            raise UnicodeError("unsupported error handling "+errors)

        if not input:
            return b'', 0

        try:
            result = input.encode('ascii')
        except UnicodeEncodeError:
            pass
        else:
            # ASCII name: fast path
            labels = result.split(b'.')
            for label in labels[:-1]:
                if not (0 < len(label) < 64):
                    raise UnicodeError("label empty or too long")
            if len(labels[-1]) >= 64:
                raise UnicodeError("label too long")
            return result, len(input)

        result = bytearray()
        labels = dots.split(input)
        if labels and not labels[-1]:
            trailing_dot = b'.'
            del labels[-1]
        else:
            trailing_dot = b''
        for label in labels:
            if result:
                # Join with U+002E
                result.extend(b'.')
            result.extend(ToASCII(label))
        return bytes(result+trailing_dot), len(input)

    def decode(self, input, errors='strict'):

        if errors != 'strict':
            raise UnicodeError("Unsupported error handling "+errors)

        if not input:
            return "", 0

        # IDNA allows decoding to operate on Unicode strings, too.
        if not isinstance(input, bytes):
            # XXX obviously wrong, see #3232
            input = bytes(input)

        if ace_prefix not in input:
            # Fast path
            try:
                return input.decode('ascii'), len(input)
            except UnicodeDecodeError:
                pass

        labels = input.split(b".")

        if labels and len(labels[-1]) == 0:
            trailing_dot = '.'
            del labels[-1]
        else:
            trailing_dot = ''

        result = []
        for label in labels:
            result.append(ToUnicode(label))

        return ".".join(result)+trailing_dot, len(input)

class IncrementalEncoder(codecs.BufferedIncrementalEncoder):
    def _buffer_encode(self, input, errors, final):
        if errors != 'strict':
            # IDNA is quite clear that implementations must be strict
            raise UnicodeError("unsupported error handling "+errors)

        if not input:
            return (b'', 0)

        labels = dots.split(input)
        trailing_dot = b''
        if labels:
            if not labels[-1]:
                trailing_dot = b'.'
                del labels[-1]
            elif not final:
                # Keep potentially unfinished label until the next call
                del labels[-1]
                if labels:
                    trailing_dot = b'.'

        result = bytearray()
        size = 0
        for label in labels:
            if size:
                # Join with U+002E
                result.extend(b'.')
                size += 1
            result.extend(ToASCII(label))
            size += len(label)

        result += trailing_dot
        size += len(trailing_dot)
        return (bytes(result), size)

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input, errors, final):
        if errors != 'strict':
            raise UnicodeError("Unsupported error handling "+errors)

        if not input:
            return ("", 0)

        # IDNA allows decoding to operate on Unicode strings, too.
        if isinstance(input, str):
            labels = dots.split(input)
        else:
            # Must be ASCII string
            input = str(input, "ascii")
            labels = input.split(".")

        trailing_dot = ''
        if labels:
            if not labels[-1]:
                trailing_dot = '.'
                del labels[-1]
            elif not final:
                # Keep potentially unfinished label until the next call
                del labels[-1]
                if labels:
                    trailing_dot = '.'

        result = []
        size = 0
        for label in labels:
            result.append(ToUnicode(label))
            if size:
                size += 1
            size += len(label)

        result = ".".join(result) + trailing_dot
        size += len(trailing_dot)
        return (result, size)

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='idna',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )
<|endoftext|>#
# iso2022_jp_1.py: Python Unicode Codec for ISO2022_JP_1
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_iso2022, codecs
import _multibytecodec as mbc

codec = _codecs_iso2022.getcodec('iso2022_jp_1')

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
        name='iso2022_jp_1',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec mac_croatian generated from 'MAPPINGS/VENDORS/APPLE/CROATIAN.TXT' with gencodec.py.

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
        name='mac-croatian',
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
    '\u2020'   #  0xA0 -> DAGGER
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    '\u2122'   #  0xAA -> TRADE MARK SIGN
    '\xb4'     #  0xAB -> ACUTE ACCENT
    '\xa8'     #  0xAC -> DIAERESIS
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    '\xd8'     #  0xAF -> LATIN CAPITAL LETTER O WITH STROKE
    '\u221e'   #  0xB0 -> INFINITY
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\u2206'   #  0xB4 -> INCREMENT
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\u2202'   #  0xB6 -> PARTIAL DIFFERENTIAL
    '\u2211'   #  0xB7 -> N-ARY SUMMATION
    '\u220f'   #  0xB8 -> N-ARY PRODUCT
    '\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    '\u222b'   #  0xBA -> INTEGRAL
    '\xaa'     #  0xBB -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0xBC -> MASCULINE ORDINAL INDICATOR
    '\u03a9'   #  0xBD -> GREEK CAPITAL LETTER OMEGA
    '\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    '\xf8'     #  0xBF -> LATIN SMALL LETTER O WITH STROKE
    '\xbf'     #  0xC0 -> INVERTED QUESTION MARK
    '\xa1'     #  0xC1 -> INVERTED EXCLAMATION MARK
    '\xac'     #  0xC2 -> NOT SIGN
    '\u221a'   #  0xC3 -> SQUARE ROOT
    '\u0192'   #  0xC4 -> LATIN SMALL LETTER F WITH HOOK
    '\u2248'   #  0xC5 -> ALMOST EQUAL TO
    '\u0106'   #  0xC6 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\xc0'     #  0xCB -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc3'     #  0xCC -> LATIN CAPITAL LETTER A WITH TILDE
    '\xd5'     #  0xCD -> LATIN CAPITAL LETTER O WITH TILDE
    '\u0152'   #  0xCE -> LATIN CAPITAL LIGATURE OE
    '\u0153'   #  0xCF -> LATIN SMALL LIGATURE OE
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u2014'   #  0xD1 -> EM DASH
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u25ca'   #  0xD7 -> LOZENGE
    '\uf8ff'   #  0xD8 -> Apple logo
    '\xa9'     #  0xD9 -> COPYRIGHT SIGN
    '\u2044'   #  0xDA -> FRACTION SLASH
    '\u20ac'   #  0xDB -> EURO SIGN
    '\u2039'   #  0xDC -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u203a'   #  0xDD -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\xc6'     #  0xDE -> LATIN CAPITAL LETTER AE
    '\xbb'     #  0xDF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2013'   #  0xE0 -> EN DASH
    '\xb7'     #  0xE1 -> MIDDLE DOT
    '\u201a'   #  0xE2 -> SINGLE LOW-9 QUOTATION MARK
    '\u201e'   #  0xE3 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2030'   #  0xE4 -> PER MILLE SIGN
    '\xc2'     #  0xE5 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\u0107'   #  0xE6 -> LATIN SMALL LETTER C WITH ACUTE
    '\xc1'     #  0xE7 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xc8'     #  0xE9 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xcd'     #  0xEA -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xEB -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xEC -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xcc'     #  0xED -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xEF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\xd2'     #  0xF1 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xda'     #  0xF2 -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xF3 -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xd9'     #  0xF4 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\u0131'   #  0xF5 -> LATIN SMALL LETTER DOTLESS I
    '\u02c6'   #  0xF6 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u02dc'   #  0xF7 -> SMALL TILDE
    '\xaf'     #  0xF8 -> MACRON
    '\u03c0'   #  0xF9 -> GREEK SMALL LETTER PI
    '\xcb'     #  0xFA -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u02da'   #  0xFB -> RING ABOVE
    '\xb8'     #  0xFC -> CEDILLA
    '\xca'     #  0xFD -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xe6'     #  0xFE -> LATIN SMALL LETTER AE
    '\u02c7'   #  0xFF -> CARON
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>#
# cp932.py: Python Unicode Codec for CP932
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_jp, codecs
import _multibytecodec as mbc

codec = _codecs_jp.getcodec('cp932')

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
        name='cp932',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec cp1006 generated from 'MAPPINGS/VENDORS/MISC/CP1006.TXT' with gencodec.py.

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
        name='cp1006',
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
    '\u06f0'   #  0xA1 -> EXTENDED ARABIC-INDIC DIGIT ZERO
    '\u06f1'   #  0xA2 -> EXTENDED ARABIC-INDIC DIGIT ONE
    '\u06f2'   #  0xA3 -> EXTENDED ARABIC-INDIC DIGIT TWO
    '\u06f3'   #  0xA4 -> EXTENDED ARABIC-INDIC DIGIT THREE
    '\u06f4'   #  0xA5 -> EXTENDED ARABIC-INDIC DIGIT FOUR
    '\u06f5'   #  0xA6 -> EXTENDED ARABIC-INDIC DIGIT FIVE
    '\u06f6'   #  0xA7 -> EXTENDED ARABIC-INDIC DIGIT SIX
    '\u06f7'   #  0xA8 -> EXTENDED ARABIC-INDIC DIGIT SEVEN
    '\u06f8'   #  0xA9 -> EXTENDED ARABIC-INDIC DIGIT EIGHT
    '\u06f9'   #  0xAA -> EXTENDED ARABIC-INDIC DIGIT NINE
    '\u060c'   #  0xAB -> ARABIC COMMA
    '\u061b'   #  0xAC -> ARABIC SEMICOLON
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u061f'   #  0xAE -> ARABIC QUESTION MARK
    '\ufe81'   #  0xAF -> ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    '\ufe8d'   #  0xB0 -> ARABIC LETTER ALEF ISOLATED FORM
    '\ufe8e'   #  0xB1 -> ARABIC LETTER ALEF FINAL FORM
    '\ufe8e'   #  0xB2 -> ARABIC LETTER ALEF FINAL FORM
    '\ufe8f'   #  0xB3 -> ARABIC LETTER BEH ISOLATED FORM
    '\ufe91'   #  0xB4 -> ARABIC LETTER BEH INITIAL FORM
    '\ufb56'   #  0xB5 -> ARABIC LETTER PEH ISOLATED FORM
    '\ufb58'   #  0xB6 -> ARABIC LETTER PEH INITIAL FORM
    '\ufe93'   #  0xB7 -> ARABIC LETTER TEH MARBUTA ISOLATED FORM
    '\ufe95'   #  0xB8 -> ARABIC LETTER TEH ISOLATED FORM
    '\ufe97'   #  0xB9 -> ARABIC LETTER TEH INITIAL FORM
    '\ufb66'   #  0xBA -> ARABIC LETTER TTEH ISOLATED FORM
    '\ufb68'   #  0xBB -> ARABIC LETTER TTEH INITIAL FORM
    '\ufe99'   #  0xBC -> ARABIC LETTER THEH ISOLATED FORM
    '\ufe9b'   #  0xBD -> ARABIC LETTER THEH INITIAL FORM
    '\ufe9d'   #  0xBE -> ARABIC LETTER JEEM ISOLATED FORM
    '\ufe9f'   #  0xBF -> ARABIC LETTER JEEM INITIAL FORM
    '\ufb7a'   #  0xC0 -> ARABIC LETTER TCHEH ISOLATED FORM
    '\ufb7c'   #  0xC1 -> ARABIC LETTER TCHEH INITIAL FORM
    '\ufea1'   #  0xC2 -> ARABIC LETTER HAH ISOLATED FORM
    '\ufea3'   #  0xC3 -> ARABIC LETTER HAH INITIAL FORM
    '\ufea5'   #  0xC4 -> ARABIC LETTER KHAH ISOLATED FORM
    '\ufea7'   #  0xC5 -> ARABIC LETTER KHAH INITIAL FORM
    '\ufea9'   #  0xC6 -> ARABIC LETTER DAL ISOLATED FORM
    '\ufb84'   #  0xC7 -> ARABIC LETTER DAHAL ISOLATED FORMN
    '\ufeab'   #  0xC8 -> ARABIC LETTER THAL ISOLATED FORM
    '\ufead'   #  0xC9 -> ARABIC LETTER REH ISOLATED FORM
    '\ufb8c'   #  0xCA -> ARABIC LETTER RREH ISOLATED FORM
    '\ufeaf'   #  0xCB -> ARABIC LETTER ZAIN ISOLATED FORM
    '\ufb8a'   #  0xCC -> ARABIC LETTER JEH ISOLATED FORM
    '\ufeb1'   #  0xCD -> ARABIC LETTER SEEN ISOLATED FORM
    '\ufeb3'   #  0xCE -> ARABIC LETTER SEEN INITIAL FORM
    '\ufeb5'   #  0xCF -> ARABIC LETTER SHEEN ISOLATED FORM
    '\ufeb7'   #  0xD0 -> ARABIC LETTER SHEEN INITIAL FORM
    '\ufeb9'   #  0xD1 -> ARABIC LETTER SAD ISOLATED FORM
    '\ufebb'   #  0xD2 -> ARABIC LETTER SAD INITIAL FORM
    '\ufebd'   #  0xD3 -> ARABIC LETTER DAD ISOLATED FORM
    '\ufebf'   #  0xD4 -> ARABIC LETTER DAD INITIAL FORM
    '\ufec1'   #  0xD5 -> ARABIC LETTER TAH ISOLATED FORM
    '\ufec5'   #  0xD6 -> ARABIC LETTER ZAH ISOLATED FORM
    '\ufec9'   #  0xD7 -> ARABIC LETTER AIN ISOLATED FORM
    '\ufeca'   #  0xD8 -> ARABIC LETTER AIN FINAL FORM
    '\ufecb'   #  0xD9 -> ARABIC LETTER AIN INITIAL FORM
    '\ufecc'   #  0xDA -> ARABIC LETTER AIN MEDIAL FORM
    '\ufecd'   #  0xDB -> ARABIC LETTER GHAIN ISOLATED FORM
    '\ufece'   #  0xDC -> ARABIC LETTER GHAIN FINAL FORM
    '\ufecf'   #  0xDD -> ARABIC LETTER GHAIN INITIAL FORM
    '\ufed0'   #  0xDE -> ARABIC LETTER GHAIN MEDIAL FORM
    '\ufed1'   #  0xDF -> ARABIC LETTER FEH ISOLATED FORM
    '\ufed3'   #  0xE0 -> ARABIC LETTER FEH INITIAL FORM
    '\ufed5'   #  0xE1 -> ARABIC LETTER QAF ISOLATED FORM
    '\ufed7'   #  0xE2 -> ARABIC LETTER QAF INITIAL FORM
    '\ufed9'   #  0xE3 -> ARABIC LETTER KAF ISOLATED FORM
    '\ufedb'   #  0xE4 -> ARABIC LETTER KAF INITIAL FORM
    '\ufb92'   #  0xE5 -> ARABIC LETTER GAF ISOLATED FORM
    '\ufb94'   #  0xE6 -> ARABIC LETTER GAF INITIAL FORM
    '\ufedd'   #  0xE7 -> ARABIC LETTER LAM ISOLATED FORM
    '\ufedf'   #  0xE8 -> ARABIC LETTER LAM INITIAL FORM
    '\ufee0'   #  0xE9 -> ARABIC LETTER LAM MEDIAL FORM
    '\ufee1'   #  0xEA -> ARABIC LETTER MEEM ISOLATED FORM
    '\ufee3'   #  0xEB -> ARABIC LETTER MEEM INITIAL FORM
    '\ufb9e'   #  0xEC -> ARABIC LETTER NOON GHUNNA ISOLATED FORM
    '\ufee5'   #  0xED -> ARABIC LETTER NOON ISOLATED FORM
    '\ufee7'   #  0xEE -> ARABIC LETTER NOON INITIAL FORM
    '\ufe85'   #  0xEF -> ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    '\ufeed'   #  0xF0 -> ARABIC LETTER WAW ISOLATED FORM
    '\ufba6'   #  0xF1 -> ARABIC LETTER HEH GOAL ISOLATED FORM
    '\ufba8'   #  0xF2 -> ARABIC LETTER HEH GOAL INITIAL FORM
    '\ufba9'   #  0xF3 -> ARABIC LETTER HEH GOAL MEDIAL FORM
    '\ufbaa'   #  0xF4 -> ARABIC LETTER HEH DOACHASHMEE ISOLATED FORM
    '\ufe80'   #  0xF5 -> ARABIC LETTER HAMZA ISOLATED FORM
    '\ufe89'   #  0xF6 -> ARABIC LETTER YEH WITH HAMZA ABOVE ISOLATED FORM
    '\ufe8a'   #  0xF7 -> ARABIC LETTER YEH WITH HAMZA ABOVE FINAL FORM
    '\ufe8b'   #  0xF8 -> ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    '\ufef1'   #  0xF9 -> ARABIC LETTER YEH ISOLATED FORM
    '\ufef2'   #  0xFA -> ARABIC LETTER YEH FINAL FORM
    '\ufef3'   #  0xFB -> ARABIC LETTER YEH INITIAL FORM
    '\ufbb0'   #  0xFC -> ARABIC LETTER YEH BARREE WITH HAMZA ABOVE ISOLATED FORM
    '\ufbae'   #  0xFD -> ARABIC LETTER YEH BARREE ISOLATED FORM
    '\ufe7c'   #  0xFE -> ARABIC SHADDA ISOLATED FORM
    '\ufe7d'   #  0xFF -> ARABIC SHADDA MEDIAL FORM
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python Character Mapping Codec iso8859_5 generated from 'MAPPINGS/ISO8859/8859-5.TXT' with gencodec.py.

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
        name='iso8859-5',
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
    '\u0401'   #  0xA1 -> CYRILLIC CAPITAL LETTER IO
    '\u0402'   #  0xA2 -> CYRILLIC CAPITAL LETTER DJE
    '\u0403'   #  0xA3 -> CYRILLIC CAPITAL LETTER GJE
    '\u0404'   #  0xA4 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u0405'   #  0xA5 -> CYRILLIC CAPITAL LETTER DZE
    '\u0406'   #  0xA6 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0407'   #  0xA7 -> CYRILLIC CAPITAL LETTER YI
    '\u0408'   #  0xA8 -> CYRILLIC CAPITAL LETTER JE
    '\u0409'   #  0xA9 -> CYRILLIC CAPITAL LETTER LJE
    '\u040a'   #  0xAA -> CYRILLIC CAPITAL LETTER NJE
    '\u040b'   #  0xAB -> CYRILLIC CAPITAL LETTER TSHE
    '\u040c'   #  0xAC -> CYRILLIC CAPITAL LETTER KJE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u040e'   #  0xAE -> CYRILLIC CAPITAL LETTER SHORT U
    '\u040f'   #  0xAF -> CYRILLIC CAPITAL LETTER DZHE
    '\u0410'   #  0xB0 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xB1 -> CYRILLIC CAPITAL LETTER BE
    '\u0412'   #  0xB2 -> CYRILLIC CAPITAL LETTER VE
    '\u0413'   #  0xB3 -> CYRILLIC CAPITAL LETTER GHE
    '\u0414'   #  0xB4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xB5 -> CYRILLIC CAPITAL LETTER IE
    '\u0416'   #  0xB6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0417'   #  0xB7 -> CYRILLIC CAPITAL LETTER ZE
    '\u0418'   #  0xB8 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xB9 -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xBA -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xBB -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xBC -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xBD -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xBE -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xBF -> CYRILLIC CAPITAL LETTER PE
    '\u0420'   #  0xC0 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xC1 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xC2 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xC3 -> CYRILLIC CAPITAL LETTER U
    '\u0424'   #  0xC4 -> CYRILLIC CAPITAL LETTER EF
    '\u0425'   #  0xC5 -> CYRILLIC CAPITAL LETTER HA
    '\u0426'   #  0xC6 -> CYRILLIC CAPITAL LETTER TSE
    '\u0427'   #  0xC7 -> CYRILLIC CAPITAL LETTER CHE
    '\u0428'   #  0xC8 -> CYRILLIC CAPITAL LETTER SHA
    '\u0429'   #  0xC9 -> CYRILLIC CAPITAL LETTER SHCHA
    '\u042a'   #  0xCA -> CYRILLIC CAPITAL LETTER HARD SIGN
    '\u042b'   #  0xCB -> CYRILLIC CAPITAL LETTER YERU
    '\u042c'   #  0xCC -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042d'   #  0xCD -> CYRILLIC CAPITAL LETTER E
    '\u042e'   #  0xCE -> CYRILLIC CAPITAL LETTER YU
    '\u042f'   #  0xCF -> CYRILLIC CAPITAL LETTER YA
    '\u0430'   #  0xD0 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xD1 -> CYRILLIC SMALL LETTER BE
    '\u0432'   #  0xD2 -> CYRILLIC SMALL LETTER VE
    '\u0433'   #  0xD3 -> CYRILLIC SMALL LETTER GHE
    '\u0434'   #  0xD4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xD5 -> CYRILLIC SMALL LETTER IE
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0437'   #  0xD7 -> CYRILLIC SMALL LETTER ZE
    '\u0438'   #  0xD8 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xD9 -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xDA -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xDB -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xDC -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xDD -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xDE -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xDF -> CYRILLIC SMALL LETTER PE
    '\u0440'   #  0xE0 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xE1 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xE2 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xE3 -> CYRILLIC SMALL LETTER U
    '\u0444'   #  0xE4 -> CYRILLIC SMALL LETTER EF
    '\u0445'   #  0xE5 -> CYRILLIC SMALL LETTER HA
    '\u0446'   #  0xE6 -> CYRILLIC SMALL LETTER TSE
    '\u0447'   #  0xE7 -> CYRILLIC SMALL LETTER CHE
    '\u0448'   #  0xE8 -> CYRILLIC SMALL LETTER SHA
    '\u0449'   #  0xE9 -> CYRILLIC SMALL LETTER SHCHA
    '\u044a'   #  0xEA -> CYRILLIC SMALL LETTER HARD SIGN
    '\u044b'   #  0xEB -> CYRILLIC SMALL LETTER YERU
    '\u044c'   #  0xEC -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044d'   #  0xED -> CYRILLIC SMALL LETTER E
    '\u044e'   #  0xEE -> CYRILLIC SMALL LETTER YU
    '\u044f'   #  0xEF -> CYRILLIC SMALL LETTER YA
    '\u2116'   #  0xF0 -> NUMERO SIGN
    '\u0451'   #  0xF1 -> CYRILLIC SMALL LETTER IO
    '\u0452'   #  0xF2 -> CYRILLIC SMALL LETTER DJE
    '\u0453'   #  0xF3 -> CYRILLIC SMALL LETTER GJE
    '\u0454'   #  0xF4 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u0455'   #  0xF5 -> CYRILLIC SMALL LETTER DZE
    '\u0456'   #  0xF6 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0457'   #  0xF7 -> CYRILLIC SMALL LETTER YI
    '\u0458'   #  0xF8 -> CYRILLIC SMALL LETTER JE
    '\u0459'   #  0xF9 -> CYRILLIC SMALL LETTER LJE
    '\u045a'   #  0xFA -> CYRILLIC SMALL LETTER NJE
    '\u045b'   #  0xFB -> CYRILLIC SMALL LETTER TSHE
    '\u045c'   #  0xFC -> CYRILLIC SMALL LETTER KJE
    '\xa7'     #  0xFD -> SECTION SIGN
    '\u045e'   #  0xFE -> CYRILLIC SMALL LETTER SHORT U
    '\u045f'   #  0xFF -> CYRILLIC SMALL LETTER DZHE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>#
# gbk.py: Python Unicode Codec for GBK
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_cn, codecs
import _multibytecodec as mbc

codec = _codecs_cn.getcodec('gbk')

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
        name='gbk',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python 'utf-7' Codec

Written by Brian Quinlan (brian@sweetapp.com).
"""
import codecs

### Codec APIs

encode = codecs.utf_7_encode

def decode(input, errors='strict'):
    return codecs.utf_7_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.utf_7_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = codecs.utf_7_decode

class StreamWriter(codecs.StreamWriter):
    encode = codecs.utf_7_encode

class StreamReader(codecs.StreamReader):
    decode = codecs.utf_7_decode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='utf-7',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec iso8859_2 generated from 'MAPPINGS/ISO8859/8859-2.TXT' with gencodec.py.

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
        name='iso8859-2',
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
    '\u02d8'   #  0xA2 -> BREVE
    '\u0141'   #  0xA3 -> LATIN CAPITAL LETTER L WITH STROKE
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u013d'   #  0xA5 -> LATIN CAPITAL LETTER L WITH CARON
    '\u015a'   #  0xA6 -> LATIN CAPITAL LETTER S WITH ACUTE
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    '\u015e'   #  0xAA -> LATIN CAPITAL LETTER S WITH CEDILLA
    '\u0164'   #  0xAB -> LATIN CAPITAL LETTER T WITH CARON
    '\u0179'   #  0xAC -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    '\u017b'   #  0xAF -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0105'   #  0xB1 -> LATIN SMALL LETTER A WITH OGONEK
    '\u02db'   #  0xB2 -> OGONEK
    '\u0142'   #  0xB3 -> LATIN SMALL LETTER L WITH STROKE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\u013e'   #  0xB5 -> LATIN SMALL LETTER L WITH CARON
    '\u015b'   #  0xB6 -> LATIN SMALL LETTER S WITH ACUTE
    '\u02c7'   #  0xB7 -> CARON
    '\xb8'     #  0xB8 -> CEDILLA
    '\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    '\u015f'   #  0xBA -> LATIN SMALL LETTER S WITH CEDILLA
    '\u0165'   #  0xBB -> LATIN SMALL LETTER T WITH CARON
    '\u017a'   #  0xBC -> LATIN SMALL LETTER Z WITH ACUTE
    '\u02dd'   #  0xBD -> DOUBLE ACUTE ACCENT
    '\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    '\u017c'   #  0xBF -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u0154'   #  0xC0 -> LATIN CAPITAL LETTER R WITH ACUTE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\u0102'   #  0xC3 -> LATIN CAPITAL LETTER A WITH BREVE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u0139'   #  0xC5 -> LATIN CAPITAL LETTER L WITH ACUTE
    '\u0106'   #  0xC6 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0118'   #  0xCA -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u011a'   #  0xCC -> LATIN CAPITAL LETTER E WITH CARON
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\u010e'   #  0xCF -> LATIN CAPITAL LETTER D WITH CARON
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0143'   #  0xD1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\u0147'   #  0xD2 -> LATIN CAPITAL LETTER N WITH CARON
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u0150'   #  0xD5 -> LATIN CAPITAL LETTER O WITH DOUBLE ACUTE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0158'   #  0xD8 -> LATIN CAPITAL LETTER R WITH CARON
    '\u016e'   #  0xD9 -> LATIN CAPITAL LETTER U WITH RING ABOVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\u0170'   #  0xDB -> LATIN CAPITAL LETTER U WITH DOUBLE ACUTE
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\u0162'   #  0xDE -> LATIN CAPITAL LETTER T WITH CEDILLA
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\u0155'   #  0xE0 -> LATIN SMALL LETTER R WITH ACUTE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0103'   #  0xE3 -> LATIN SMALL LETTER A WITH BREVE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u013a'   #  0xE5 -> LATIN SMALL LETTER L WITH ACUTE
    '\u0107'   #  0xE6 -> LATIN SMALL LETTER C WITH ACUTE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u0119'   #  0xEA -> LATIN SMALL LETTER E WITH OGONEK
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u011b'   #  0xEC -> LATIN SMALL LETTER E WITH CARON
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\u010f'   #  0xEF -> LATIN SMALL LETTER D WITH CARON
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\u0144'   #  0xF1 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0148'   #  0xF2 -> LATIN SMALL LETTER N WITH CARON
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u0151'   #  0xF5 -> LATIN SMALL LETTER O WITH DOUBLE ACUTE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0159'   #  0xF8 -> LATIN SMALL LETTER R WITH CARON
    '\u016f'   #  0xF9 -> LATIN SMALL LETTER U WITH RING ABOVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\u0171'   #  0xFB -> LATIN SMALL LETTER U WITH DOUBLE ACUTE
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\u0163'   #  0xFE -> LATIN SMALL LETTER T WITH CEDILLA
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
<|endoftext|>""" Python 'oem' Codec for Windows

"""
# Import them explicitly to cause an ImportError
# on non-Windows systems
from codecs import oem_encode, oem_decode
# for IncrementalDecoder, IncrementalEncoder, ...
import codecs

### Codec APIs

encode = oem_encode

def decode(input, errors='strict'):
    return oem_decode(input, errors, True)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return oem_encode(input, self.errors)[0]

class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    _buffer_decode = oem_decode

class StreamWriter(codecs.StreamWriter):
    encode = oem_encode

class StreamReader(codecs.StreamReader):
    decode = oem_decode

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='oem',
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>""" Python Character Mapping Codec generated from 'VENDORS/APPLE/ARABIC.TXT' with gencodec.py.

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
        name='mac-arabic',
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
    0x0080: 0x00c4,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x0081: 0x00a0,     #  NO-BREAK SPACE, right-left
    0x0082: 0x00c7,     #  LATIN CAPITAL LETTER C WITH CEDILLA
    0x0083: 0x00c9,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x0084: 0x00d1,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x0085: 0x00d6,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x0086: 0x00dc,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x0087: 0x00e1,     #  LATIN SMALL LETTER A WITH ACUTE
    0x0088: 0x00e0,     #  LATIN SMALL LETTER A WITH GRAVE
    0x0089: 0x00e2,     #  LATIN SMALL LETTER A WITH CIRCUMFLEX
    0x008a: 0x00e4,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x008b: 0x06ba,     #  ARABIC LETTER NOON GHUNNA
    0x008c: 0x00ab,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    0x008d: 0x00e7,     #  LATIN SMALL LETTER C WITH CEDILLA
    0x008e: 0x00e9,     #  LATIN SMALL LETTER E WITH ACUTE
    0x008f: 0x00e8,     #  LATIN SMALL LETTER E WITH GRAVE
    0x0090: 0x00ea,     #  LATIN SMALL LETTER E WITH CIRCUMFLEX
    0x0091: 0x00eb,     #  LATIN SMALL LETTER E WITH DIAERESIS
    0x0092: 0x00ed,     #  LATIN SMALL LETTER I WITH ACUTE
    0x0093: 0x2026,     #  HORIZONTAL ELLIPSIS, right-left
    0x0094: 0x00ee,     #  LATIN SMALL LETTER I WITH CIRCUMFLEX
    0x0095: 0x00ef,     #  LATIN SMALL LETTER I WITH DIAERESIS
    0x0096: 0x00f1,     #  LATIN SMALL LETTER N WITH TILDE
    0x0097: 0x00f3,     #  LATIN SMALL LETTER O WITH ACUTE
    0x0098: 0x00bb,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    0x0099: 0x00f4,     #  LATIN SMALL LETTER O WITH CIRCUMFLEX
    0x009a: 0x00f6,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x009b: 0x00f7,     #  DIVISION SIGN, right-left
    0x009c: 0x00fa,     #  LATIN SMALL LETTER U WITH ACUTE
    0x009d: 0x00f9,     #  LATIN SMALL LETTER U WITH GRAVE
    0x009e: 0x00fb,     #  LATIN SMALL LETTER U WITH CIRCUMFLEX
    0x009f: 0x00fc,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x00a0: 0x0020,     #  SPACE, right-left
    0x00a1: 0x0021,     #  EXCLAMATION MARK, right-left
    0x00a2: 0x0022,     #  QUOTATION MARK, right-left
    0x00a3: 0x0023,     #  NUMBER SIGN, right-left
    0x00a4: 0x0024,     #  DOLLAR SIGN, right-left
    0x00a5: 0x066a,     #  ARABIC PERCENT SIGN
    0x00a6: 0x0026,     #  AMPERSAND, right-left
    0x00a7: 0x0027,     #  APOSTROPHE, right-left
    0x00a8: 0x0028,     #  LEFT PARENTHESIS, right-left
    0x00a9: 0x0029,     #  RIGHT PARENTHESIS, right-left
    0x00aa: 0x002a,     #  ASTERISK, right-left
    0x00ab: 0x002b,     #  PLUS SIGN, right-left
    0x00ac: 0x060c,     #  ARABIC COMMA
    0x00ad: 0x002d,     #  HYPHEN-MINUS, right-left
    0x00ae: 0x002e,     #  FULL STOP, right-left
    0x00af: 0x002f,     #  SOLIDUS, right-left
    0x00b0: 0x0660,     #  ARABIC-INDIC DIGIT ZERO, right-left (need override)
    0x00b1: 0x0661,     #  ARABIC-INDIC DIGIT ONE, right-left (need override)
    0x00b2: 0x0662,     #  ARABIC-INDIC DIGIT TWO, right-left (need override)
    0x00b3: 0x0663,     #  ARABIC-INDIC DIGIT THREE, right-left (need override)
    0x00b4: 0x0664,     #  ARABIC-INDIC DIGIT FOUR, right-left (need override)
    0x00b5: 0x0665,     #  ARABIC-INDIC DIGIT FIVE, right-left (need override)
    0x00b6: 0x0666,     #  ARABIC-INDIC DIGIT SIX, right-left (need override)
    0x00b7: 0x0667,     #  ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    0x00b8: 0x0668,     #  ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    0x00b9: 0x0669,     #  ARABIC-INDIC DIGIT NINE, right-left (need override)
    0x00ba: 0x003a,     #  COLON, right-left
    0x00bb: 0x061b,     #  ARABIC SEMICOLON
    0x00bc: 0x003c,     #  LESS-THAN SIGN, right-left
    0x00bd: 0x003d,     #  EQUALS SIGN, right-left
    0x00be: 0x003e,     #  GREATER-THAN SIGN, right-left
    0x00bf: 0x061f,     #  ARABIC QUESTION MARK
    0x00c0: 0x274a,     #  EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
    0x00c1: 0x0621,     #  ARABIC LETTER HAMZA
    0x00c2: 0x0622,     #  ARABIC LETTER ALEF WITH MADDA ABOVE
    0x00c3: 0x0623,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE
    0x00c4: 0x0624,     #  ARABIC LETTER WAW WITH HAMZA ABOVE
    0x00c5: 0x0625,     #  ARABIC LETTER ALEF WITH HAMZA BELOW
    0x00c6: 0x0626,     #  ARABIC LETTER YEH WITH HAMZA ABOVE
    0x00c7: 0x0627,     #  ARABIC LETTER ALEF
    0x00c8: 0x0628,     #  ARABIC LETTER BEH
    0x00c9: 0x0629,     #  ARABIC LETTER TEH MARBUTA
    0x00ca: 0x062a,     #  ARABIC LETTER TEH
    0x00cb: 0x062b,     #  ARABIC LETTER THEH
    0x00cc: 0x062c,     #  ARABIC LETTER JEEM
    0x00cd: 0x062d,     #  ARABIC LETTER HAH
    0x00ce: 0x062e,     #  ARABIC LETTER KHAH
    0x00cf: 0x062f,     #  ARABIC LETTER DAL
    0x00d0: 0x0630,     #  ARABIC LETTER THAL
    0x00d1: 0x0631,     #  ARABIC LETTER REH
    0x00d2: 0x0632,     #  ARABIC LETTER ZAIN
    0x00d3: 0x0633,     #  ARABIC LETTER SEEN
    0x00d4: 0x0634,     #  ARABIC LETTER SHEEN
    0x00d5: 0x0635,     #  ARABIC LETTER SAD
    0x00d6: 0x0636,     #  ARABIC LETTER DAD
    0x00d7: 0x0637,     #  ARABIC LETTER TAH
    0x00d8: 0x0638,     #  ARABIC LETTER ZAH
    0x00d9: 0x0639,     #  ARABIC LETTER AIN
    0x00da: 0x063a,     #  ARABIC LETTER GHAIN
    0x00db: 0x005b,     #  LEFT SQUARE BRACKET, right-left
    0x00dc: 0x005c,     #  REVERSE SOLIDUS, right-left
    0x00dd: 0x005d,     #  RIGHT SQUARE BRACKET, right-left
    0x00de: 0x005e,     #  CIRCUMFLEX ACCENT, right-left
    0x00df: 0x005f,     #  LOW LINE, right-left
    0x00e0: 0x0640,     #  ARABIC TATWEEL
    0x00e1: 0x0641,     #  ARABIC LETTER FEH
    0x00e2: 0x0642,     #  ARABIC LETTER QAF
    0x00e3: 0x0643,     #  ARABIC LETTER KAF
    0x00e4: 0x0644,     #  ARABIC LETTER LAM
    0x00e5: 0x0645,     #  ARABIC LETTER MEEM
    0x00e6: 0x0646,     #  ARABIC LETTER NOON
    0x00e7: 0x0647,     #  ARABIC LETTER HEH
    0x00e8: 0x0648,     #  ARABIC LETTER WAW
    0x00e9: 0x0649,     #  ARABIC LETTER ALEF MAKSURA
    0x00ea: 0x064a,     #  ARABIC LETTER YEH
    0x00eb: 0x064b,     #  ARABIC FATHATAN
    0x00ec: 0x064c,     #  ARABIC DAMMATAN
    0x00ed: 0x064d,     #  ARABIC KASRATAN
    0x00ee: 0x064e,     #  ARABIC FATHA
    0x00ef: 0x064f,     #  ARABIC DAMMA
    0x00f0: 0x0650,     #  ARABIC KASRA
    0x00f1: 0x0651,     #  ARABIC SHADDA
    0x00f2: 0x0652,     #  ARABIC SUKUN
    0x00f3: 0x067e,     #  ARABIC LETTER PEH
    0x00f4: 0x0679,     #  ARABIC LETTER TTEH
    0x00f5: 0x0686,     #  ARABIC LETTER TCHEH
    0x00f6: 0x06d5,     #  ARABIC LETTER AE
    0x00f7: 0x06a4,     #  ARABIC LETTER VEH
    0x00f8: 0x06af,     #  ARABIC LETTER GAF
    0x00f9: 0x0688,     #  ARABIC LETTER DDAL
    0x00fa: 0x0691,     #  ARABIC LETTER RREH
    0x00fb: 0x007b,     #  LEFT CURLY BRACKET, right-left
    0x00fc: 0x007c,     #  VERTICAL LINE, right-left
    0x00fd: 0x007d,     #  RIGHT CURLY BRACKET, right-left
    0x00fe: 0x0698,     #  ARABIC LETTER JEH
    0x00ff: 0x06d2,     #  ARABIC LETTER YEH BARREE
})

### Decoding Table

decoding_table = (
    '\x00'     #  0x0000 -> CONTROL CHARACTER
    '\x01'     #  0x0001 -> CONTROL CHARACTER
    '\x02'     #  0x0002 -> CONTROL CHARACTER
    '\x03'     #  0x0003 -> CONTROL CHARACTER
    '\x04'     #  0x0004 -> CONTROL CHARACTER
    '\x05'     #  0x0005 -> CONTROL CHARACTER
    '\x06'     #  0x0006 -> CONTROL CHARACTER
    '\x07'     #  0x0007 -> CONTROL CHARACTER
    '\x08'     #  0x0008 -> CONTROL CHARACTER
    '\t'       #  0x0009 -> CONTROL CHARACTER
    '\n'       #  0x000a -> CONTROL CHARACTER
    '\x0b'     #  0x000b -> CONTROL CHARACTER
    '\x0c'     #  0x000c -> CONTROL CHARACTER
    '\r'       #  0x000d -> CONTROL CHARACTER
    '\x0e'     #  0x000e -> CONTROL CHARACTER
    '\x0f'     #  0x000f -> CONTROL CHARACTER
    '\x10'     #  0x0010 -> CONTROL CHARACTER
    '\x11'     #  0x0011 -> CONTROL CHARACTER
    '\x12'     #  0x0012 -> CONTROL CHARACTER
    '\x13'     #  0x0013 -> CONTROL CHARACTER
    '\x14'     #  0x0014 -> CONTROL CHARACTER
    '\x15'     #  0x0015 -> CONTROL CHARACTER
    '\x16'     #  0x0016 -> CONTROL CHARACTER
    '\x17'     #  0x0017 -> CONTROL CHARACTER
    '\x18'     #  0x0018 -> CONTROL CHARACTER
    '\x19'     #  0x0019 -> CONTROL CHARACTER
    '\x1a'     #  0x001a -> CONTROL CHARACTER
    '\x1b'     #  0x001b -> CONTROL CHARACTER
    '\x1c'     #  0x001c -> CONTROL CHARACTER
    '\x1d'     #  0x001d -> CONTROL CHARACTER
    '\x1e'     #  0x001e -> CONTROL CHARACTER
    '\x1f'     #  0x001f -> CONTROL CHARACTER
    ' '        #  0x0020 -> SPACE, left-right
    '!'        #  0x0021 -> EXCLAMATION MARK, left-right
    '"'        #  0x0022 -> QUOTATION MARK, left-right
    '#'        #  0x0023 -> NUMBER SIGN, left-right
    '$'        #  0x0024 -> DOLLAR SIGN, left-right
    '%'        #  0x0025 -> PERCENT SIGN, left-right
    '&'        #  0x0026 -> AMPERSAND, left-right
    "'"        #  0x0027 -> APOSTROPHE, left-right
    '('        #  0x0028 -> LEFT PARENTHESIS, left-right
    ')'        #  0x0029 -> RIGHT PARENTHESIS, left-right
    '*'        #  0x002a -> ASTERISK, left-right
    '+'        #  0x002b -> PLUS SIGN, left-right
    ','        #  0x002c -> COMMA, left-right; in Arabic-script context, displayed as 0x066C ARABIC THOUSANDS SEPARATOR
    '-'        #  0x002d -> HYPHEN-MINUS, left-right
    '.'        #  0x002e -> FULL STOP, left-right; in Arabic-script context, displayed as 0x066B ARABIC DECIMAL SEPARATOR
    '/'        #  0x002f -> SOLIDUS, left-right
    '0'        #  0x0030 -> DIGIT ZERO;  in Arabic-script context, displayed as 0x0660 ARABIC-INDIC DIGIT ZERO
    '1'        #  0x0031 -> DIGIT ONE;   in Arabic-script context, displayed as 0x0661 ARABIC-INDIC DIGIT ONE
    '2'        #  0x0032 -> DIGIT TWO;   in Arabic-script context, displayed as 0x0662 ARABIC-INDIC DIGIT TWO
    '3'        #  0x0033 -> DIGIT THREE; in Arabic-script context, displayed as 0x0663 ARABIC-INDIC DIGIT THREE
    '4'        #  0x0034 -> DIGIT FOUR;  in Arabic-script context, displayed as 0x0664 ARABIC-INDIC DIGIT FOUR
    '5'        #  0x0035 -> DIGIT FIVE;  in Arabic-script context, displayed as 0x0665 ARABIC-INDIC DIGIT FIVE
    '6'        #  0x0036 -> DIGIT SIX;   in Arabic-script context, displayed as 0x0666 ARABIC-INDIC DIGIT SIX
    '7'        #  0x0037 -> DIGIT SEVEN; in Arabic-script context, displayed as 0x0667 ARABIC-INDIC DIGIT SEVEN
    '8'        #  0x0038 -> DIGIT EIGHT; in Arabic-script context, displayed as 0x0668 ARABIC-INDIC DIGIT EIGHT
    '9'        #  0x0039 -> DIGIT NINE;  in Arabic-script context, displayed as 0x0669 ARABIC-INDIC DIGIT NINE
    ':'        #  0x003a -> COLON, left-right
    ';'        #  0x003b -> SEMICOLON, left-right
    '<'        #  0x003c -> LESS-THAN SIGN, left-right
    '='        #  0x003d -> EQUALS SIGN, left-right
    '>'        #  0x003e -> GREATER-THAN SIGN, left-right
    '?'        #  0x003f -> QUESTION MARK, left-right
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
    '['        #  0x005b -> LEFT SQUARE BRACKET, left-right
    '\\'       #  0x005c -> REVERSE SOLIDUS, left-right
    ']'        #  0x005d -> RIGHT SQUARE BRACKET, left-right
    '^'        #  0x005e -> CIRCUMFLEX ACCENT, left-right
    '_'        #  0x005f -> LOW LINE, left-right
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
    '{'        #  0x007b -> LEFT CURLY BRACKET, left-right
    '|'        #  0x007c -> VERTICAL LINE, left-right
    '}'        #  0x007d -> RIGHT CURLY BRACKET, left-right
    '~'        #  0x007e -> TILDE
    '\x7f'     #  0x007f -> CONTROL CHARACTER
    '\xc4'     #  0x0080 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xa0'     #  0x0081 -> NO-BREAK SPACE, right-left
    '\xc7'     #  0x0082 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc9'     #  0x0083 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xd1'     #  0x0084 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd6'     #  0x0085 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x0086 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x0087 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe0'     #  0x0088 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x0089 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x008a -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u06ba'   #  0x008b -> ARABIC LETTER NOON GHUNNA
    '\xab'     #  0x008c -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xe7'     #  0x008d -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x008e -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x008f -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x0090 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x0091 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xed'     #  0x0092 -> LATIN SMALL LETTER I WITH ACUTE
    '\u2026'   #  0x0093 -> HORIZONTAL ELLIPSIS, right-left
    '\xee'     #  0x0094 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x0095 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf1'     #  0x0096 -> LATIN SMALL LETTER N WITH TILDE
    '\xf3'     #  0x0097 -> LATIN SMALL LETTER O WITH ACUTE
    '\xbb'     #  0x0098 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xf4'     #  0x0099 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x009a -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0x009b -> DIVISION SIGN, right-left
    '\xfa'     #  0x009c -> LATIN SMALL LETTER U WITH ACUTE
    '\xf9'     #  0x009d -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x009e -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x009f -> LATIN SMALL LETTER U WITH DIAERESIS
    ' '        #  0x00a0 -> SPACE, right-left
    '!'        #  0x00a1 -> EXCLAMATION MARK, right-left
    '"'        #  0x00a2 -> QUOTATION MARK, right-left
    '#'        #  0x00a3 -> NUMBER SIGN, right-left
    '$'        #  0x00a4 -> DOLLAR SIGN, right-left
    '\u066a'   #  0x00a5 -> ARABIC PERCENT SIGN
    '&'        #  0x00a6 -> AMPERSAND, right-left
    "'"        #  0x00a7 -> APOSTROPHE, right-left
    '('        #  0x00a8 -> LEFT PARENTHESIS, right-left
    ')'        #  0x00a9 -> RIGHT PARENTHESIS, right-left
    '*'        #  0x00aa -> ASTERISK, right-left
    '+'        #  0x00ab -> PLUS SIGN, right-left
    '\u060c'   #  0x00ac -> ARABIC COMMA
    '-'        #  0x00ad -> HYPHEN-MINUS, right-left
    '.'        #  0x00ae -> FULL STOP, right-left
    '/'        #  0x00af -> SOLIDUS, right-left
    '\u0660'   #  0x00b0 -> ARABIC-INDIC DIGIT ZERO, right-left (need override)
    '\u0661'   #  0x00b1 -> ARABIC-INDIC DIGIT ONE, right-left (need override)
    '\u0662'   #  0x00b2 -> ARABIC-INDIC DIGIT TWO, right-left (need override)
    '\u0663'   #  0x00b3 -> ARABIC-INDIC DIGIT THREE, right-left (need override)
    '\u0664'   #  0x00b4 -> ARABIC-INDIC DIGIT FOUR, right-left (need override)
    '\u0665'   #  0x00b5 -> ARABIC-INDIC DIGIT FIVE, right-left (need override)
    '\u0666'   #  0x00b6 -> ARABIC-INDIC DIGIT SIX, right-left (need override)
    '\u0667'   #  0x00b7 -> ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    '\u0668'   #  0x00b8 -> ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    '\u0669'   #  0x00b9 -> ARABIC-INDIC DIGIT NINE, right-left (need override)
    ':'        #  0x00ba -> COLON, right-left
    '\u061b'   #  0x00bb -> ARABIC SEMICOLON
    '<'        #  0x00bc -> LESS-THAN SIGN, right-left
    '='        #  0x00bd -> EQUALS SIGN, right-left
    '>'        #  0x00be -> GREATER-THAN SIGN, right-left
    '\u061f'   #  0x00bf -> ARABIC QUESTION MARK
    '\u274a'   #  0x00c0 -> EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
    '\u0621'   #  0x00c1 -> ARABIC LETTER HAMZA
    '\u0622'   #  0x00c2 -> ARABIC LETTER ALEF WITH MADDA ABOVE
    '\u0623'   #  0x00c3 -> ARABIC LETTER ALEF WITH HAMZA ABOVE
    '\u0624'   #  0x00c4 -> ARABIC LETTER WAW WITH HAMZA ABOVE
    '\u0625'   #  0x00c5 -> ARABIC LETTER ALEF WITH HAMZA BELOW
    '\u0626'   #  0x00c6 -> ARABIC LETTER YEH WITH HAMZA ABOVE
    '\u0627'   #  0x00c7 -> ARABIC LETTER ALEF
    '\u0628'   #  0x00c8 -> ARABIC LETTER BEH
    '\u0629'   #  0x00c9 -> ARABIC LETTER TEH MARBUTA
    '\u062a'   #  0x00ca -> ARABIC LETTER TEH
    '\u062b'   #  0x00cb -> ARABIC LETTER THEH
    '\u062c'   #  0x00cc -> ARABIC LETTER JEEM
    '\u062d'   #  0x00cd -> ARABIC LETTER HAH
    '\u062e'   #  0x00ce -> ARABIC LETTER KHAH
    '\u062f'   #  0x00cf -> ARABIC LETTER DAL
    '\u0630'   #  0x00d0 -> ARABIC LETTER THAL
    '\u0631'   #  0x00d1 -> ARABIC LETTER REH
    '\u0632'   #  0x00d2 -> ARABIC LETTER ZAIN
    '\u0633'   #  0x00d3 -> ARABIC LETTER SEEN
    '\u0634'   #  0x00d4 -> ARABIC LETTER SHEEN
    '\u0635'   #  0x00d5 -> ARABIC LETTER SAD
    '\u0636'   #  0x00d6 -> ARABIC LETTER DAD
    '\u0637'   #  0x00d7 -> ARABIC LETTER TAH
    '\u0638'   #  0x00d8 -> ARABIC LETTER ZAH
    '\u0639'   #  0x00d9 -> ARABIC LETTER AIN
    '\u063a'   #  0x00da -> ARABIC LETTER GHAIN
    '['        #  0x00db -> LEFT SQUARE BRACKET, right-left
    '\\'       #  0x00dc -> REVERSE SOLIDUS, right-left
    ']'        #  0x00dd -> RIGHT SQUARE BRACKET, right-left
    '^'        #  0x00de -> CIRCUMFLEX ACCENT, right-left
    '_'        #  0x00df -> LOW LINE, right-left
    '\u0640'   #  0x00e0 -> ARABIC TATWEEL
    '\u0641'   #  0x00e1 -> ARABIC LETTER FEH
    '\u0642'   #  0x00e2 -> ARABIC LETTER QAF
    '\u0643'   #  0x00e3 -> ARABIC LETTER KAF
    '\u0644'   #  0x00e4 -> ARABIC LETTER LAM
    '\u0645'   #  0x00e5 -> ARABIC LETTER MEEM
    '\u0646'   #  0x00e6 -> ARABIC LETTER NOON
    '\u0647'   #  0x00e7 -> ARABIC LETTER HEH
    '\u0648'   #  0x00e8 -> ARABIC LETTER WAW
    '\u0649'   #  0x00e9 -> ARABIC LETTER ALEF MAKSURA
    '\u064a'   #  0x00ea -> ARABIC LETTER YEH
    '\u064b'   #  0x00eb -> ARABIC FATHATAN
    '\u064c'   #  0x00ec -> ARABIC DAMMATAN
    '\u064d'   #  0x00ed -> ARABIC KASRATAN
    '\u064e'   #  0x00ee -> ARABIC FATHA
    '\u064f'   #  0x00ef -> ARABIC DAMMA
    '\u0650'   #  0x00f0 -> ARABIC KASRA
    '\u0651'   #  0x00f1 -> ARABIC SHADDA
    '\u0652'   #  0x00f2 -> ARABIC SUKUN
    '\u067e'   #  0x00f3 -> ARABIC LETTER PEH
    '\u0679'   #  0x00f4 -> ARABIC LETTER TTEH
    '\u0686'   #  0x00f5 -> ARABIC LETTER TCHEH
    '\u06d5'   #  0x00f6 -> ARABIC LETTER AE
    '\u06a4'   #  0x00f7 -> ARABIC LETTER VEH
    '\u06af'   #  0x00f8 -> ARABIC LETTER GAF
    '\u0688'   #  0x00f9 -> ARABIC LETTER DDAL
    '\u0691'   #  0x00fa -> ARABIC LETTER RREH
    '{'        #  0x00fb -> LEFT CURLY BRACKET, right-left
    '|'        #  0x00fc -> VERTICAL LINE, right-left
    '}'        #  0x00fd -> RIGHT CURLY BRACKET, right-left
    '\u0698'   #  0x00fe -> ARABIC LETTER JEH
    '\u06d2'   #  0x00ff -> ARABIC LETTER YEH BARREE
)

### Encoding Map

encoding_map = {
    0x0000: 0x0000,     #  CONTROL CHARACTER
    0x0001: 0x0001,     #  CONTROL CHARACTER
    0x0002: 0x0002,     #  CONTROL CHARACTER
    0x0003: 0x0003,     #  CONTROL CHARACTER
    0x0004: 0x0004,     #  CONTROL CHARACTER
    0x0005: 0x0005,     #  CONTROL CHARACTER
    0x0006: 0x0006,     #  CONTROL CHARACTER
    0x0007: 0x0007,     #  CONTROL CHARACTER
    0x0008: 0x0008,     #  CONTROL CHARACTER
    0x0009: 0x0009,     #  CONTROL CHARACTER
    0x000a: 0x000a,     #  CONTROL CHARACTER
    0x000b: 0x000b,     #  CONTROL CHARACTER
    0x000c: 0x000c,     #  CONTROL CHARACTER
    0x000d: 0x000d,     #  CONTROL CHARACTER
    0x000e: 0x000e,     #  CONTROL CHARACTER
    0x000f: 0x000f,     #  CONTROL CHARACTER
    0x0010: 0x0010,     #  CONTROL CHARACTER
    0x0011: 0x0011,     #  CONTROL CHARACTER
    0x0012: 0x0012,     #  CONTROL CHARACTER
    0x0013: 0x0013,     #  CONTROL CHARACTER
    0x0014: 0x0014,     #  CONTROL CHARACTER
    0x0015: 0x0015,     #  CONTROL CHARACTER
    0x0016: 0x0016,     #  CONTROL CHARACTER
    0x0017: 0x0017,     #  CONTROL CHARACTER
    0x0018: 0x0018,     #  CONTROL CHARACTER
    0x0019: 0x0019,     #  CONTROL CHARACTER
    0x001a: 0x001a,     #  CONTROL CHARACTER
    0x001b: 0x001b,     #  CONTROL CHARACTER
    0x001c: 0x001c,     #  CONTROL CHARACTER
    0x001d: 0x001d,     #  CONTROL CHARACTER
    0x001e: 0x001e,     #  CONTROL CHARACTER
    0x001f: 0x001f,     #  CONTROL CHARACTER
    0x0020: 0x0020,     #  SPACE, left-right
    0x0020: 0x00a0,     #  SPACE, right-left
    0x0021: 0x0021,     #  EXCLAMATION MARK, left-right
    0x0021: 0x00a1,     #  EXCLAMATION MARK, right-left
    0x0022: 0x0022,     #  QUOTATION MARK, left-right
    0x0022: 0x00a2,     #  QUOTATION MARK, right-left
    0x0023: 0x0023,     #  NUMBER SIGN, left-right
    0x0023: 0x00a3,     #  NUMBER SIGN, right-left
    0x0024: 0x0024,     #  DOLLAR SIGN, left-right
    0x0024: 0x00a4,     #  DOLLAR SIGN, right-left
    0x0025: 0x0025,     #  PERCENT SIGN, left-right
    0x0026: 0x0026,     #  AMPERSAND, left-right
    0x0026: 0x00a6,     #  AMPERSAND, right-left
    0x0027: 0x0027,     #  APOSTROPHE, left-right
    0x0027: 0x00a7,     #  APOSTROPHE, right-left
    0x0028: 0x0028,     #  LEFT PARENTHESIS, left-right
    0x0028: 0x00a8,     #  LEFT PARENTHESIS, right-left
    0x0029: 0x0029,     #  RIGHT PARENTHESIS, left-right
    0x0029: 0x00a9,     #  RIGHT PARENTHESIS, right-left
    0x002a: 0x002a,     #  ASTERISK, left-right
    0x002a: 0x00aa,     #  ASTERISK, right-left
    0x002b: 0x002b,     #  PLUS SIGN, left-right
    0x002b: 0x00ab,     #  PLUS SIGN, right-left
    0x002c: 0x002c,     #  COMMA, left-right; in Arabic-script context, displayed as 0x066C ARABIC THOUSANDS SEPARATOR
    0x002d: 0x002d,     #  HYPHEN-MINUS, left-right
    0x002d: 0x00ad,     #  HYPHEN-MINUS, right-left
    0x002e: 0x002e,     #  FULL STOP, left-right; in Arabic-script context, displayed as 0x066B ARABIC DECIMAL SEPARATOR
    0x002e: 0x00ae,     #  FULL STOP, right-left
    0x002f: 0x002f,     #  SOLIDUS, left-right
    0x002f: 0x00af,     #  SOLIDUS, right-left
    0x0030: 0x0030,     #  DIGIT ZERO;  in Arabic-script context, displayed as 0x0660 ARABIC-INDIC DIGIT ZERO
    0x0031: 0x0031,     #  DIGIT ONE;   in Arabic-script context, displayed as 0x0661 ARABIC-INDIC DIGIT ONE
    0x0032: 0x0032,     #  DIGIT TWO;   in Arabic-script context, displayed as 0x0662 ARABIC-INDIC DIGIT TWO
    0x0033: 0x0033,     #  DIGIT THREE; in Arabic-script context, displayed as 0x0663 ARABIC-INDIC DIGIT THREE
    0x0034: 0x0034,     #  DIGIT FOUR;  in Arabic-script context, displayed as 0x0664 ARABIC-INDIC DIGIT FOUR
    0x0035: 0x0035,     #  DIGIT FIVE;  in Arabic-script context, displayed as 0x0665 ARABIC-INDIC DIGIT FIVE
    0x0036: 0x0036,     #  DIGIT SIX;   in Arabic-script context, displayed as 0x0666 ARABIC-INDIC DIGIT SIX
    0x0037: 0x0037,     #  DIGIT SEVEN; in Arabic-script context, displayed as 0x0667 ARABIC-INDIC DIGIT SEVEN
    0x0038: 0x0038,     #  DIGIT EIGHT; in Arabic-script context, displayed as 0x0668 ARABIC-INDIC DIGIT EIGHT
    0x0039: 0x0039,     #  DIGIT NINE;  in Arabic-script context, displayed as 0x0669 ARABIC-INDIC DIGIT NINE
    0x003a: 0x003a,     #  COLON, left-right
    0x003a: 0x00ba,     #  COLON, right-left
    0x003b: 0x003b,     #  SEMICOLON, left-right
    0x003c: 0x003c,     #  LESS-THAN SIGN, left-right
    0x003c: 0x00bc,     #  LESS-THAN SIGN, right-left
    0x003d: 0x003d,     #  EQUALS SIGN, left-right
    0x003d: 0x00bd,     #  EQUALS SIGN, right-left
    0x003e: 0x003e,     #  GREATER-THAN SIGN, left-right
    0x003e: 0x00be,     #  GREATER-THAN SIGN, right-left
    0x003f: 0x003f,     #  QUESTION MARK, left-right
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
    0x005b: 0x005b,     #  LEFT SQUARE BRACKET, left-right
    0x005b: 0x00db,     #  LEFT SQUARE BRACKET, right-left
    0x005c: 0x005c,     #  REVERSE SOLIDUS, left-right
    0x005c: 0x00dc,     #  REVERSE SOLIDUS, right-left
    0x005d: 0x005d,     #  RIGHT SQUARE BRACKET, left-right
    0x005d: 0x00dd,     #  RIGHT SQUARE BRACKET, right-left
    0x005e: 0x005e,     #  CIRCUMFLEX ACCENT, left-right
    0x005e: 0x00de,     #  CIRCUMFLEX ACCENT, right-left
    0x005f: 0x005f,     #  LOW LINE, left-right
    0x005f: 0x00df,     #  LOW LINE, right-left
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
    0x007b: 0x007b,     #  LEFT CURLY BRACKET, left-right
    0x007b: 0x00fb,     #  LEFT CURLY BRACKET, right-left
    0x007c: 0x007c,     #  VERTICAL LINE, left-right
    0x007c: 0x00fc,     #  VERTICAL LINE, right-left
    0x007d: 0x007d,     #  RIGHT CURLY BRACKET, left-right
    0x007d: 0x00fd,     #  RIGHT CURLY BRACKET, right-left
    0x007e: 0x007e,     #  TILDE
    0x007f: 0x007f,     #  CONTROL CHARACTER
    0x00a0: 0x0081,     #  NO-BREAK SPACE, right-left
    0x00ab: 0x008c,     #  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    0x00bb: 0x0098,     #  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    0x00c4: 0x0080,     #  LATIN CAPITAL LETTER A WITH DIAERESIS
    0x00c7: 0x0082,     #  LATIN CAPITAL LETTER C WITH CEDILLA
    0x00c9: 0x0083,     #  LATIN CAPITAL LETTER E WITH ACUTE
    0x00d1: 0x0084,     #  LATIN CAPITAL LETTER N WITH TILDE
    0x00d6: 0x0085,     #  LATIN CAPITAL LETTER O WITH DIAERESIS
    0x00dc: 0x0086,     #  LATIN CAPITAL LETTER U WITH DIAERESIS
    0x00e0: 0x0088,     #  LATIN SMALL LETTER A WITH GRAVE
    0x00e1: 0x0087,     #  LATIN SMALL LETTER A WITH ACUTE
    0x00e2: 0x0089,     #  LATIN SMALL LETTER A WITH CIRCUMFLEX
    0x00e4: 0x008a,     #  LATIN SMALL LETTER A WITH DIAERESIS
    0x00e7: 0x008d,     #  LATIN SMALL LETTER C WITH CEDILLA
    0x00e8: 0x008f,     #  LATIN SMALL LETTER E WITH GRAVE
    0x00e9: 0x008e,     #  LATIN SMALL LETTER E WITH ACUTE
    0x00ea: 0x0090,     #  LATIN SMALL LETTER E WITH CIRCUMFLEX
    0x00eb: 0x0091,     #  LATIN SMALL LETTER E WITH DIAERESIS
    0x00ed: 0x0092,     #  LATIN SMALL LETTER I WITH ACUTE
    0x00ee: 0x0094,     #  LATIN SMALL LETTER I WITH CIRCUMFLEX
    0x00ef: 0x0095,     #  LATIN SMALL LETTER I WITH DIAERESIS
    0x00f1: 0x0096,     #  LATIN SMALL LETTER N WITH TILDE
    0x00f3: 0x0097,     #  LATIN SMALL LETTER O WITH ACUTE
    0x00f4: 0x0099,     #  LATIN SMALL LETTER O WITH CIRCUMFLEX
    0x00f6: 0x009a,     #  LATIN SMALL LETTER O WITH DIAERESIS
    0x00f7: 0x009b,     #  DIVISION SIGN, right-left
    0x00f9: 0x009d,     #  LATIN SMALL LETTER U WITH GRAVE
    0x00fa: 0x009c,     #  LATIN SMALL LETTER U WITH ACUTE
    0x00fb: 0x009e,     #  LATIN SMALL LETTER U WITH CIRCUMFLEX
    0x00fc: 0x009f,     #  LATIN SMALL LETTER U WITH DIAERESIS
    0x060c: 0x00ac,     #  ARABIC COMMA
    0x061b: 0x00bb,     #  ARABIC SEMICOLON
    0x061f: 0x00bf,     #  ARABIC QUESTION MARK
    0x0621: 0x00c1,     #  ARABIC LETTER HAMZA
    0x0622: 0x00c2,     #  ARABIC LETTER ALEF WITH MADDA ABOVE
    0x0623: 0x00c3,     #  ARABIC LETTER ALEF WITH HAMZA ABOVE
    0x0624: 0x00c4,     #  ARABIC LETTER WAW WITH HAMZA ABOVE
    0x0625: 0x00c5,     #  ARABIC LETTER ALEF WITH HAMZA BELOW
    0x0626: 0x00c6,     #  ARABIC LETTER YEH WITH HAMZA ABOVE
    0x0627: 0x00c7,     #  ARABIC LETTER ALEF
    0x0628: 0x00c8,     #  ARABIC LETTER BEH
    0x0629: 0x00c9,     #  ARABIC LETTER TEH MARBUTA
    0x062a: 0x00ca,     #  ARABIC LETTER TEH
    0x062b: 0x00cb,     #  ARABIC LETTER THEH
    0x062c: 0x00cc,     #  ARABIC LETTER JEEM
    0x062d: 0x00cd,     #  ARABIC LETTER HAH
    0x062e: 0x00ce,     #  ARABIC LETTER KHAH
    0x062f: 0x00cf,     #  ARABIC LETTER DAL
    0x0630: 0x00d0,     #  ARABIC LETTER THAL
    0x0631: 0x00d1,     #  ARABIC LETTER REH
    0x0632: 0x00d2,     #  ARABIC LETTER ZAIN
    0x0633: 0x00d3,     #  ARABIC LETTER SEEN
    0x0634: 0x00d4,     #  ARABIC LETTER SHEEN
    0x0635: 0x00d5,     #  ARABIC LETTER SAD
    0x0636: 0x00d6,     #  ARABIC LETTER DAD
    0x0637: 0x00d7,     #  ARABIC LETTER TAH
    0x0638: 0x00d8,     #  ARABIC LETTER ZAH
    0x0639: 0x00d9,     #  ARABIC LETTER AIN
    0x063a: 0x00da,     #  ARABIC LETTER GHAIN
    0x0640: 0x00e0,     #  ARABIC TATWEEL
    0x0641: 0x00e1,     #  ARABIC LETTER FEH
    0x0642: 0x00e2,     #  ARABIC LETTER QAF
    0x0643: 0x00e3,     #  ARABIC LETTER KAF
    0x0644: 0x00e4,     #  ARABIC LETTER LAM
    0x0645: 0x00e5,     #  ARABIC LETTER MEEM
    0x0646: 0x00e6,     #  ARABIC LETTER NOON
    0x0647: 0x00e7,     #  ARABIC LETTER HEH
    0x0648: 0x00e8,     #  ARABIC LETTER WAW
    0x0649: 0x00e9,     #  ARABIC LETTER ALEF MAKSURA
    0x064a: 0x00ea,     #  ARABIC LETTER YEH
    0x064b: 0x00eb,     #  ARABIC FATHATAN
    0x064c: 0x00ec,     #  ARABIC DAMMATAN
    0x064d: 0x00ed,     #  ARABIC KASRATAN
    0x064e: 0x00ee,     #  ARABIC FATHA
    0x064f: 0x00ef,     #  ARABIC DAMMA
    0x0650: 0x00f0,     #  ARABIC KASRA
    0x0651: 0x00f1,     #  ARABIC SHADDA
    0x0652: 0x00f2,     #  ARABIC SUKUN
    0x0660: 0x00b0,     #  ARABIC-INDIC DIGIT ZERO, right-left (need override)
    0x0661: 0x00b1,     #  ARABIC-INDIC DIGIT ONE, right-left (need override)
    0x0662: 0x00b2,     #  ARABIC-INDIC DIGIT TWO, right-left (need override)
    0x0663: 0x00b3,     #  ARABIC-INDIC DIGIT THREE, right-left (need override)
    0x0664: 0x00b4,     #  ARABIC-INDIC DIGIT FOUR, right-left (need override)
    0x0665: 0x00b5,     #  ARABIC-INDIC DIGIT FIVE, right-left (need override)
    0x0666: 0x00b6,     #  ARABIC-INDIC DIGIT SIX, right-left (need override)
    0x0667: 0x00b7,     #  ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    0x0668: 0x00b8,     #  ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    0x0669: 0x00b9,     #  ARABIC-INDIC DIGIT NINE, right-left (need override)
    0x066a: 0x00a5,     #  ARABIC PERCENT SIGN
    0x0679: 0x00f4,     #  ARABIC LETTER TTEH
    0x067e: 0x00f3,     #  ARABIC LETTER PEH
    0x0686: 0x00f5,     #  ARABIC LETTER TCHEH
    0x0688: 0x00f9,     #  ARABIC LETTER DDAL
    0x0691: 0x00fa,     #  ARABIC LETTER RREH
    0x0698: 0x00fe,     #  ARABIC LETTER JEH
    0x06a4: 0x00f7,     #  ARABIC LETTER VEH
    0x06af: 0x00f8,     #  ARABIC LETTER GAF
    0x06ba: 0x008b,     #  ARABIC LETTER NOON GHUNNA
    0x06d2: 0x00ff,     #  ARABIC LETTER YEH BARREE
    0x06d5: 0x00f6,     #  ARABIC LETTER AE
    0x2026: 0x0093,     #  HORIZONTAL ELLIPSIS, right-left
    0x274a: 0x00c0,     #  EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
}
<|endoftext|>""" Python Character Mapping Codec mac_latin2 generated from 'MAPPINGS/VENDORS/MICSFT/MAC/LATIN2.TXT' with gencodec.py.

Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.
(c) Copyright 2000 Guido van Rossum.

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
        name='mac-latin2',
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
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u0100'   #  0x81 -> LATIN CAPITAL LETTER A WITH MACRON
    '\u0101'   #  0x82 -> LATIN SMALL LETTER A WITH MACRON
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0104'   #  0x84 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\u0105'   #  0x88 -> LATIN SMALL LETTER A WITH OGONEK
    '\u010c'   #  0x89 -> LATIN CAPITAL LETTER C WITH CARON
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u010d'   #  0x8B -> LATIN SMALL LETTER C WITH CARON
    '\u0106'   #  0x8C -> LATIN CAPITAL LETTER C WITH ACUTE
    '\u0107'   #  0x8D -> LATIN SMALL LETTER C WITH ACUTE
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\u0179'   #  0x8F -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\u017a'   #  0x90 -> LATIN SMALL LETTER Z WITH ACUTE
    '\u010e'   #  0x91 -> LATIN CAPITAL LETTER D WITH CARON
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\u010f'   #  0x93 -> LATIN SMALL LETTER D WITH CARON
    '\u0112'   #  0x94 -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0113'   #  0x95 -> LATIN SMALL LETTER E WITH MACRON
    '\u0116'   #  0x96 -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\u0117'   #  0x98 -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf5'     #  0x9B -> LATIN SMALL LETTER O WITH TILDE
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\u011a'   #  0x9D -> LATIN CAPITAL LETTER E WITH CARON
    '\u011b'   #  0x9E -> LATIN SMALL LETTER E WITH CARON
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u2020'   #  0xA0 -> DAGGER
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\u0118'   #  0xA2 -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
