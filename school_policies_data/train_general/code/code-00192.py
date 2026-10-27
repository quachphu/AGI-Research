    return int(match.group(), 0)
<|endoftext|># SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Taneli Hukkinen
# Licensed to PSF under a Contributor Agreement.

from __future__ import annotations

# Defer loading regular expressions until we actually need them in
# parse_value().
__lazy_modules__ = ["tomli._re", "tomllib._re"]

import sys

from ._re import (
    RE_DATETIME,
    RE_LOCALTIME,
    RE_NUMBER,
    match_to_datetime,
    match_to_localtime,
    match_to_number,
)

if sys.version_info < (3, 15):  # pragma: no cover
    from types import MappingProxyType as frozendict

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import IO, Any, Final

    from ._types import Key, ParseFloat, Pos

# Inline tables/arrays are implemented using recursion. Pathologically
# nested documents cause pure Python to raise RecursionError (which is OK),
# but mypyc binary wheels will crash unrecoverably (not OK). According to
# mypyc docs this will be fixed in the future:
# https://mypyc.readthedocs.io/en/latest/differences_from_python.html#stack-overflows
# Before mypyc's fix is in, recursion needs to be limited by this library.
# Choosing 400 as maximum inline table/array nesting level, as it's 4x the
# spec recommended minimum (https://github.com/toml-lang/toml/pull/1087)
# (unreleased at the time of writing) and less than half of what causes an
# unrecoverable crash in a worker (not main) thread on a python:3.13-alpine
# image.
MAX_INLINE_NESTING: Final = 400

# Pathologically excessive number of parts in a key runs into quadratic
# behavior (e.g. in Flags.is_).
# Even if keys aren't currently parsed using recursion, they name a
# recursive structure, so it makes sense to limit it using getrecursionlimit()
# and RecursionError.
MAX_KEY_PARTS: Final = sys.getrecursionlimit()

ASCII_CTRL: Final = frozenset(chr(i) for i in range(32)) | frozenset(chr(127))

# Neither of these sets include quotation mark or backslash. They are
# currently handled as separate cases in the parser functions.
ILLEGAL_BASIC_STR_CHARS: Final = ASCII_CTRL - frozenset("\t")
ILLEGAL_MULTILINE_BASIC_STR_CHARS: Final = ASCII_CTRL - frozenset("\t\n")

ILLEGAL_LITERAL_STR_CHARS: Final = ILLEGAL_BASIC_STR_CHARS
ILLEGAL_MULTILINE_LITERAL_STR_CHARS: Final = ILLEGAL_MULTILINE_BASIC_STR_CHARS

ILLEGAL_COMMENT_CHARS: Final = ILLEGAL_BASIC_STR_CHARS

TOML_WS: Final = frozenset(" \t")
TOML_WS_AND_NEWLINE: Final = TOML_WS | frozenset("\n")
BARE_KEY_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "-_"
)
KEY_INITIAL_CHARS: Final = BARE_KEY_CHARS | frozenset("\"'")
HEXDIGIT_CHARS: Final = frozenset("abcdef" "ABCDEF" "0123456789")

BASIC_STR_ESCAPE_REPLACEMENTS: Final = frozendict(
    {
        "\\b": "\u0008",  # backspace
        "\\t": "\u0009",  # tab
        "\\n": "\u000a",  # linefeed
        "\\f": "\u000c",  # form feed
        "\\r": "\u000d",  # carriage return
        "\\e": "\u001b",  # escape
        '\\"': "\u0022",  # quote
        "\\\\": "\u005c",  # backslash
    }
)


class DEPRECATED_DEFAULT:
    """Sentinel to be used as default arg during deprecation
    period of TOMLDecodeError's free-form arguments."""


class TOMLDecodeError(ValueError):
    """An error raised if a document is not valid TOML.

    Adds the following attributes to ValueError:
    msg: The unformatted error message
    doc: The TOML document being parsed
    pos: The index of doc where parsing failed
    lineno: The line corresponding to pos
    colno: The column corresponding to pos
    """

    def __init__(
        self,
        msg: str | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        doc: str | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        pos: Pos | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        *args: Any,
    ):
        if (
            args
            or not isinstance(msg, str)
            or not isinstance(doc, str)
            or not isinstance(pos, int)
        ):
            import warnings

            warnings.warn(
                "Free-form arguments for TOMLDecodeError are deprecated. "
                "Please set 'msg' (str), 'doc' (str) and 'pos' (int) arguments only.",
                DeprecationWarning,
                stacklevel=2,
            )
            if pos is not DEPRECATED_DEFAULT:
                args = pos, *args
            if doc is not DEPRECATED_DEFAULT:
                args = doc, *args
            if msg is not DEPRECATED_DEFAULT:
                args = msg, *args
            ValueError.__init__(self, *args)
            return

        lineno = doc.count("\n", 0, pos) + 1
        if lineno == 1:
            colno = pos + 1
        else:
            colno = pos - doc.rindex("\n", 0, pos)

        if pos >= len(doc):
            coord_repr = "end of document"
        else:
            coord_repr = f"line {lineno}, column {colno}"
        errmsg = f"{msg} (at {coord_repr})"
        ValueError.__init__(self, errmsg)

        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno


def load(__fp: IO[bytes], *, parse_float: ParseFloat = float) -> dict[str, Any]:
    """Parse TOML from a binary file object."""
    b = __fp.read()
    try:
        s = b.decode()
    except AttributeError:
        raise TypeError(
            "File must be opened in binary mode, e.g. use `open('foo.toml', 'rb')`"
        ) from None
    return loads(s, parse_float=parse_float)


def loads(__s: str, *, parse_float: ParseFloat = float) -> dict[str, Any]:
    """Parse TOML from a string."""

    # The spec allows converting "\r\n" to "\n", even in string
    # literals. Let's do so to simplify parsing.
    try:
        src = __s.replace("\r\n", "\n")
    except (AttributeError, TypeError):
        raise TypeError(
            f"Expected str object, not '{type(__s).__qualname__}'"
        ) from None
    pos = 0
    out = Output()
    header: Key = ()
    parse_float = make_safe_parse_float(parse_float)

    # Parse one statement at a time
    # (typically means one line in TOML source)
    while True:
        # 1. Skip line leading whitespace
        pos = skip_chars(src, pos, TOML_WS)

        # 2. Parse rules. Expect one of the following:
        #    - end of file
        #    - end of line
        #    - comment
        #    - key/value pair
        #    - append dict to list (and move to its namespace)
        #    - create dict (and move to its namespace)
        # Skip trailing whitespace when applicable.
        try:
            char = src[pos]
        except IndexError:
            break
        if char == "\n":
            pos += 1
            continue
        if char in KEY_INITIAL_CHARS:
            pos = key_value_rule(src, pos, out, header, parse_float)
            pos = skip_chars(src, pos, TOML_WS)
        elif char == "[":
            try:
                second_char: str | None = src[pos + 1]
            except IndexError:
                second_char = None
            out.flags.finalize_pending()
            if second_char == "[":
                pos, header = create_list_rule(src, pos, out)
            else:
                pos, header = create_dict_rule(src, pos, out)
            pos = skip_chars(src, pos, TOML_WS)
        elif char != "#":
            raise TOMLDecodeError("Invalid statement", src, pos)

        # 3. Skip comment
        pos = skip_comment(src, pos)

        # 4. Expect end of line or end of file
        try:
            char = src[pos]
        except IndexError:
            break
        if char != "\n":
            raise TOMLDecodeError(
                "Expected newline or end of document after a statement", src, pos
            )
        pos += 1

    return out.data.dict


class Flags:
    """Flags that map to parsed keys/namespaces."""

    # Marks an immutable namespace (inline array or inline table).
    FROZEN: Final = 0
    # Marks a nest that has been explicitly created and can no longer
    # be opened using the "[table]" syntax.
    EXPLICIT_NEST: Final = 1

    def __init__(self) -> None:
        self._flags: dict[str, dict[Any, Any]] = {}
        self._pending_flags: set[tuple[Key, int]] = set()

    def add_pending(self, key: Key, flag: int) -> None:
        self._pending_flags.add((key, flag))

    def finalize_pending(self) -> None:
        for key, flag in self._pending_flags:
            self.set(key, flag, recursive=False)
        self._pending_flags.clear()

    def unset_all(self, key: Key) -> None:
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return
            cont = cont[k]["nested"]
        cont.pop(key[-1], None)

    def set(self, key: Key, flag: int, *, recursive: bool) -> None:  # noqa: A003
        cont = self._flags
        key_parent, key_stem = key[:-1], key[-1]
        for k in key_parent:
            if k not in cont:
                cont[k] = {"flags": set(), "recursive_flags": set(), "nested": {}}
            cont = cont[k]["nested"]
        if key_stem not in cont:
            cont[key_stem] = {"flags": set(), "recursive_flags": set(), "nested": {}}
        cont[key_stem]["recursive_flags" if recursive else "flags"].add(flag)

    def is_(self, key: Key, flag: int) -> bool:
        if not key:
            return False  # document root has no flags
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return False
            inner_cont = cont[k]
            if flag in inner_cont["recursive_flags"]:
                return True
            cont = inner_cont["nested"]
        key_stem = key[-1]
        if key_stem in cont:
            inner_cont = cont[key_stem]
            return flag in inner_cont["flags"] or flag in inner_cont["recursive_flags"]
        return False


class NestedDict:
    def __init__(self) -> None:
        # The parsed content of the TOML document
        self.dict: dict[str, Any] = {}

    def get_or_create_nest(
        self,
        key: Key,
        *,
        access_lists: bool = True,
    ) -> dict[str, Any]:
        cont: Any = self.dict
        for k in key:
            if k not in cont:
                cont[k] = {}
            cont = cont[k]
            if access_lists and isinstance(cont, list):
                cont = cont[-1]
            if not isinstance(cont, dict):
                raise KeyError("There is no nest behind this key")
        return cont  # type: ignore[no-any-return]

    def append_nest_to_list(self, key: Key) -> None:
        cont = self.get_or_create_nest(key[:-1])
        last_key = key[-1]
        if last_key in cont:
            list_ = cont[last_key]
            if not isinstance(list_, list):
                raise KeyError("An object other than list found behind this key")
            list_.append({})
        else:
            cont[last_key] = [{}]


class Output:
    def __init__(self) -> None:
        self.data = NestedDict()
        self.flags = Flags()


def skip_chars(src: str, pos: Pos, chars: Iterable[str]) -> Pos:
    try:
        while src[pos] in chars:
            pos += 1
    except IndexError:
        pass
    return pos


def skip_until(
    src: str,
    pos: Pos,
    expect: str,
    *,
    error_on: frozenset[str],
    error_on_eof: bool,
) -> Pos:
    try:
        new_pos = src.index(expect, pos)
    except ValueError:
        new_pos = len(src)
        if error_on_eof:
            raise TOMLDecodeError(f"Expected {expect!r}", src, new_pos) from None

    if not error_on.isdisjoint(src[pos:new_pos]):
        while src[pos] not in error_on:
            pos += 1
        raise TOMLDecodeError(f"Found invalid character {src[pos]!r}", src, pos)
    return new_pos


def skip_comment(src: str, pos: Pos) -> Pos:
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char == "#":
        return skip_until(
            src, pos + 1, "\n", error_on=ILLEGAL_COMMENT_CHARS, error_on_eof=False
        )
    return pos


def skip_comments_and_array_ws(src: str, pos: Pos) -> Pos:
    while True:
        pos_before_skip = pos
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        pos = skip_comment(src, pos)
        if pos == pos_before_skip:
            return pos


def create_dict_rule(src: str, pos: Pos, out: Output) -> tuple[Pos, Key]:
    pos += 1  # Skip "["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if out.flags.is_(key, Flags.EXPLICIT_NEST) or out.flags.is_(key, Flags.FROZEN):
        raise TOMLDecodeError(f"Cannot declare {key} twice", src, pos)
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.get_or_create_nest(key)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None

    if not src.startswith("]", pos):
        raise TOMLDecodeError(
            "Expected ']' at the end of a table declaration", src, pos
        )
    return pos + 1, key


def create_list_rule(src: str, pos: Pos, out: Output) -> tuple[Pos, Key]:
    pos += 2  # Skip "[["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if out.flags.is_(key, Flags.FROZEN):
        raise TOMLDecodeError(f"Cannot mutate immutable namespace {key}", src, pos)
    # Free the namespace now that it points to another empty list item...
    out.flags.unset_all(key)
    # ...but this key precisely is still prohibited from table declaration
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.append_nest_to_list(key)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None

    if not src.startswith("]]", pos):
        raise TOMLDecodeError(
            "Expected ']]' at the end of an array declaration", src, pos
        )
    return pos + 2, key


def key_value_rule(
    src: str, pos: Pos, out: Output, header: Key, parse_float: ParseFloat
) -> Pos:
    pos, key, value = parse_key_value_pair(src, pos, parse_float, nest_lvl=0)
    key_parent, key_stem = key[:-1], key[-1]
    abs_key_parent = header + key_parent

    relative_path_cont_keys = (header + key[:i] for i in range(1, len(key)))
    for cont_key in relative_path_cont_keys:
        # Check that dotted key syntax does not redefine an existing table
        if out.flags.is_(cont_key, Flags.EXPLICIT_NEST):
            raise TOMLDecodeError(f"Cannot redefine namespace {cont_key}", src, pos)
        # Containers in the relative path can't be opened with the table syntax or
        # dotted key/value syntax in following table sections.
        out.flags.add_pending(cont_key, Flags.EXPLICIT_NEST)

    if out.flags.is_(abs_key_parent, Flags.FROZEN):
        raise TOMLDecodeError(
            f"Cannot mutate immutable namespace {abs_key_parent}", src, pos
        )

    try:
        nest = out.data.get_or_create_nest(abs_key_parent)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None
    if key_stem in nest:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos)
    # Mark inline table and array namespaces recursively immutable
    if isinstance(value, (dict, list)):
        out.flags.set(header + key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos


def parse_key_value_pair(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, Key, Any]:
    pos, key = parse_key(src, pos)
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char != "=":
        raise TOMLDecodeError("Expected '=' after a key in a key/value pair", src, pos)
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, value = parse_value(src, pos, parse_float, nest_lvl)
    return pos, key, value


def parse_key(src: str, pos: Pos) -> tuple[Pos, Key]:
    pos, key_part = parse_key_part(src, pos)
    key: Key = (key_part,)
    pos = skip_chars(src, pos, TOML_WS)
    while True:
        try:
            char: str | None = src[pos]
        except IndexError:
            char = None
        if char != ".":
            return pos, key
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)
        pos, key_part = parse_key_part(src, pos)
        key += (key_part,)
        if len(key) > MAX_KEY_PARTS:
            raise RecursionError(
                f"TOML key has more than the allowed {MAX_KEY_PARTS} parts"
            )
        pos = skip_chars(src, pos, TOML_WS)


def parse_key_part(src: str, pos: Pos) -> tuple[Pos, str]:
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char in BARE_KEY_CHARS:
        start_pos = pos
        pos = skip_chars(src, pos, BARE_KEY_CHARS)
        return pos, src[start_pos:pos]
    if char == "'":
        return parse_literal_str(src, pos)
    if char == '"':
        return parse_one_line_basic_str(src, pos)
    raise TOMLDecodeError("Invalid initial character for a key part", src, pos)


def parse_one_line_basic_str(src: str, pos: Pos) -> tuple[Pos, str]:
    pos += 1
    return parse_basic_str(src, pos, multiline=False)


def parse_array(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, list[Any]]:
    pos += 1
    array: list[Any] = []

    pos = skip_comments_and_array_ws(src, pos)
    if src.startswith("]", pos):
        return pos + 1, array
    while True:
        pos, val = parse_value(src, pos, parse_float, nest_lvl)
        array.append(val)
        pos = skip_comments_and_array_ws(src, pos)

        c = src[pos : pos + 1]
        if c == "]":
            return pos + 1, array
        if c != ",":
            raise TOMLDecodeError("Unclosed array", src, pos)
        pos += 1

        pos = skip_comments_and_array_ws(src, pos)
        if src.startswith("]", pos):
            return pos + 1, array


def parse_inline_table(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, dict[str, Any]]:
    pos += 1
    nested_dict = NestedDict()
    flags = Flags()

    pos = skip_comments_and_array_ws(src, pos)
    if src.startswith("}", pos):
        return pos + 1, nested_dict.dict
    while True:
        pos, key, value = parse_key_value_pair(src, pos, parse_float, nest_lvl)
        key_parent, key_stem = key[:-1], key[-1]
        if flags.is_(key, Flags.FROZEN):
            raise TOMLDecodeError(f"Cannot mutate immutable namespace {key}", src, pos)
        try:
            nest = nested_dict.get_or_create_nest(key_parent, access_lists=False)
        except KeyError:
            raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None
        if key_stem in nest:
            raise TOMLDecodeError(f"Duplicate inline table key {key_stem!r}", src, pos)
        nest[key_stem] = value
        pos = skip_comments_and_array_ws(src, pos)
        c = src[pos : pos + 1]
        if c == "}":
            return pos + 1, nested_dict.dict
        if c != ",":
            raise TOMLDecodeError("Unclosed inline table", src, pos)
        pos += 1
        pos = skip_comments_and_array_ws(src, pos)
        if src.startswith("}", pos):
            return pos + 1, nested_dict.dict
        if isinstance(value, (dict, list)):
            flags.set(key, Flags.FROZEN, recursive=True)


def parse_basic_str_escape(
    src: str, pos: Pos, *, multiline: bool = False
) -> tuple[Pos, str]:
    escape_id = src[pos : pos + 2]
    pos += 2
    if multiline and escape_id in {"\\ ", "\\\t", "\\\n"}:
        # Skip whitespace until next non-whitespace character or end of
        # the doc. Error if non-whitespace is found before newline.
        if escape_id != "\\\n":
            pos = skip_chars(src, pos, TOML_WS)
            try:
                char = src[pos]
            except IndexError:
                return pos, ""
            if char != "\n":
                raise TOMLDecodeError("Unescaped '\\' in a string", src, pos)
            pos += 1
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        return pos, ""
    if escape_id == "\\x":
        return parse_hex_char(src, pos, 2)
    if escape_id == "\\u":
        return parse_hex_char(src, pos, 4)
    if escape_id == "\\U":
        return parse_hex_char(src, pos, 8)
    try:
        return pos, BASIC_STR_ESCAPE_REPLACEMENTS[escape_id]
    except KeyError:
        raise TOMLDecodeError("Unescaped '\\' in a string", src, pos) from None


def parse_basic_str_escape_multiline(src: str, pos: Pos) -> tuple[Pos, str]:
    return parse_basic_str_escape(src, pos, multiline=True)


def parse_hex_char(src: str, pos: Pos, hex_len: int) -> tuple[Pos, str]:
    hex_str = src[pos : pos + hex_len]
    if len(hex_str) != hex_len or not HEXDIGIT_CHARS.issuperset(hex_str):
        raise TOMLDecodeError("Invalid hex value", src, pos)
    pos += hex_len
    hex_int = int(hex_str, 16)
    if not is_unicode_scalar_value(hex_int):
        raise TOMLDecodeError(
            "Escaped character is not a Unicode scalar value", src, pos
        )
    return pos, chr(hex_int)


def parse_literal_str(src: str, pos: Pos) -> tuple[Pos, str]:
    pos += 1  # Skip starting apostrophe
    start_pos = pos
    pos = skip_until(
        src, pos, "'", error_on=ILLEGAL_LITERAL_STR_CHARS, error_on_eof=True
    )
    return pos + 1, src[start_pos:pos]  # Skip ending apostrophe


def parse_multiline_str(src: str, pos: Pos, *, literal: bool) -> tuple[Pos, str]:
    pos += 3
    if src.startswith("\n", pos):
        pos += 1

    if literal:
        delim = "'"
        end_pos = skip_until(
            src,
            pos,
            "'''",
            error_on=ILLEGAL_MULTILINE_LITERAL_STR_CHARS,
            error_on_eof=True,
        )
        result = src[pos:end_pos]
        pos = end_pos + 3
    else:
        delim = '"'
        pos, result = parse_basic_str(src, pos, multiline=True)

    # Add at maximum two extra apostrophes/quotes if the end sequence
    # is 4 or 5 chars long instead of just 3.
    if not src.startswith(delim, pos):
        return pos, result
    pos += 1
    if not src.startswith(delim, pos):
        return pos, result + delim
    pos += 1
    return pos, result + (delim * 2)


def parse_basic_str(src: str, pos: Pos, *, multiline: bool) -> tuple[Pos, str]:
    if multiline:
        error_on = ILLEGAL_MULTILINE_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape_multiline
    else:
        error_on = ILLEGAL_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape
    result = ""
    start_pos = pos
    while True:
        try:
            char = src[pos]
        except IndexError:
            raise TOMLDecodeError("Unterminated string", src, pos) from None
        if char == '"':
            if not multiline:
                return pos + 1, result + src[start_pos:pos]
            if src.startswith('"""', pos):
                return pos + 3, result + src[start_pos:pos]
            pos += 1
            continue
        if char == "\\":
            result += src[start_pos:pos]
            pos, parsed_escape = parse_escapes(src, pos)
            result += parsed_escape
            start_pos = pos
            continue
        if char in error_on:
            raise TOMLDecodeError(f"Illegal character {char!r}", src, pos)
        pos += 1


def parse_value(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, Any]:
    if nest_lvl > MAX_INLINE_NESTING:
        # Pure Python should have raised RecursionError already.
        # This ensures mypyc binaries eventually do the same.
        raise RecursionError(  # pragma: no cover
            "TOML inline arrays/tables are nested more than the allowed"
            f" {MAX_INLINE_NESTING} levels"
        )

    try:
        char: str | None = src[pos]
    except IndexError:
        char = None

    # IMPORTANT: order conditions based on speed of checking and likelihood

    # Basic strings
    if char == '"':
        if src.startswith('"""', pos):
            return parse_multiline_str(src, pos, literal=False)
        return parse_one_line_basic_str(src, pos)

    # Literal strings
    if char == "'":
        if src.startswith("'''", pos):
            return parse_multiline_str(src, pos, literal=True)
        return parse_literal_str(src, pos)

    # Booleans
    if char == "t":
        if src.startswith("true", pos):
            return pos + 4, True
    if char == "f":
        if src.startswith("false", pos):
            return pos + 5, False

    # Arrays
    if char == "[":
        return parse_array(src, pos, parse_float, nest_lvl + 1)

    # Inline tables
    if char == "{":
        return parse_inline_table(src, pos, parse_float, nest_lvl + 1)

    # Dates and times
    datetime_match = RE_DATETIME.match(src, pos)
    if datetime_match:
        try:
            datetime_obj = match_to_datetime(datetime_match)
        except ValueError as e:
            raise TOMLDecodeError("Invalid date or datetime", src, pos) from e
        return datetime_match.end(), datetime_obj
    localtime_match = RE_LOCALTIME.match(src, pos)
    if localtime_match:
        return localtime_match.end(), match_to_localtime(localtime_match)

    # Integers and "normal" floats.
    # The regex will greedily match any type starting with a decimal
    # char, so needs to be located after handling of dates and times.
    number_match = RE_NUMBER.match(src, pos)
    if number_match:
        return number_match.end(), match_to_number(number_match, parse_float)

    # Special floats
    first_three = src[pos : pos + 3]
    if first_three in {"inf", "nan"}:
        return pos + 3, parse_float(first_three)
    first_four = src[pos : pos + 4]
    if first_four in {"-inf", "+inf", "-nan", "+nan"}:
        return pos + 4, parse_float(first_four)

    raise TOMLDecodeError("Invalid value", src, pos)


def is_unicode_scalar_value(codepoint: int) -> bool:
    return (0 <= codepoint <= 55295) or (57344 <= codepoint <= 1114111)


def make_safe_parse_float(parse_float: ParseFloat) -> ParseFloat:
    """A decorator to make `parse_float` safe.

    `parse_float` must not return dicts or lists, because these types
    would be mixed with parsed TOML tables and arrays, thus confusing
    the parser. The returned decorated callable raises `ValueError`
    instead of returning illegal types.
    """
    # The default `float` callable never returns illegal types. Optimize it.
    if parse_float is float:
        return float

    def safe_parse_float(float_str: str) -> Any:
        float_value = parse_float(float_str)
        if isinstance(float_value, (dict, list)):
            raise ValueError("parse_float must not return dicts or lists")
        return float_value

    return safe_parse_float
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
all Python Standard Library objects (currently: CH 1-15 @ 2.7)
and some other common objects (i.e. numpy.ndarray)
"""

__all__ = ['registered','failures','succeeds']

# helper imports
import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
import queue as Queue
#import dbm as anydbm #XXX: delete foo
from io import BytesIO as StringIO
import re
import array
import collections
import codecs
import struct
import dataclasses
import datetime
import calendar
import weakref
import pprint
import decimal
import numbers
import functools
import itertools
import operator
import tempfile
import shelve
import zlib
import gzip
import zipfile
import tarfile
import csv
import hashlib
import hmac
import os
import logging
import logging.handlers
import optparse
#import __hello__
import threading
import socket
import contextlib
import contextvars
try:
    import bz2
    import sqlite3
    import dbm.ndbm as dbm
    HAS_ALL = True
except ImportError: # Ubuntu
    HAS_ALL = False
try:
    #import curses
    #from curses import textpad, panel
    HAS_CURSES = True
except ImportError: # Windows
    HAS_CURSES = False
try:
    import ctypes
    HAS_CTYPES = True
    # if using `pypy`, pythonapi is not found
    IS_PYPY = not hasattr(ctypes, 'pythonapi')
except ImportError: # MacPorts
    HAS_CTYPES = False
    IS_PYPY = False

IS_PYODIDE = sys.platform == 'emscripten'

# helper objects
class _class:
    def _method(self):
        pass
#   @classmethod
#   def _clsmethod(cls): #XXX: test me
#       pass
#   @staticmethod
#   def _static(self): #XXX: test me
#       pass
class _class2:
    def __call__(self):
        pass
_instance2 = _class2()
class _newclass(object):
    def _method(self):
        pass
#   @classmethod
#   def _clsmethod(cls): #XXX: test me
#       pass
#   @staticmethod
#   def _static(self): #XXX: test me
#       pass
class _newclass2(object):
    __slots__ = ['descriptor']
def _function(x): yield x
def _function2():
    try: raise
    except Exception:
        from sys import exc_info
        e, er, tb = exc_info()
        return er, tb
if HAS_CTYPES:
    class _Struct(ctypes.Structure):
        pass
    _Struct._fields_ = [("_field", ctypes.c_int),("next", ctypes.POINTER(_Struct))]
_filedescrip, _tempfile = tempfile.mkstemp('r') # deleted in cleanup
if sys.hexversion < 0x30d00a1:
    _tmpf = tempfile.TemporaryFile('w') # emits OSError 9 in python 3.13
else:
    _tmpf = tempfile.NamedTemporaryFile('w').file # for > python 3.9

# objects used by dill for type declaration
registered = d = {}
# objects dill fails to pickle
failures = x = {}
# all other type objects
succeeds = a = {}

# types module (part of CH 8)
a['BooleanType'] = bool(1)
a['BuiltinFunctionType'] = len
a['BuiltinMethodType'] = a['BuiltinFunctionType']
a['BytesType'] = _bytes = codecs.latin_1_encode('\x00')[0] # bytes(1)
a['ClassType'] = _class
a['ComplexType'] = complex(1)
a['DictType'] = _dict = {}
a['DictionaryType'] = a['DictType']
a['FloatType'] = float(1)
a['FunctionType'] = _function
a['InstanceType'] = _instance = _class()
a['IntType'] = _int = int(1)
a['ListType'] = _list = []
a['NoneType'] = None
a['ObjectType'] = object()
a['StringType'] = _str = str(1)
a['TupleType'] = _tuple = ()
a['TypeType'] = type
a['LongType'] = _int
a['UnicodeType'] = _str
# built-in constants (CH 4)
a['CopyrightType'] = copyright
# built-in types (CH 5)
a['ClassObjectType'] = _newclass # <type 'type'>
a['ClassInstanceType'] = _newclass() # <type 'class'>
a['SetType'] = _set = set()
a['FrozenSetType'] = frozenset()
# built-in exceptions (CH 6)
a['ExceptionType'] = _exception = _function2()[0]
# string services (CH 7)
a['SREPatternType'] = _srepattern = re.compile('')
# data types (CH 8)
a['ArrayType'] = array.array("f")
a['DequeType'] = collections.deque([0])
a['DefaultDictType'] = collections.defaultdict(_function, _dict)
a['TZInfoType'] = datetime.tzinfo()
a['DateTimeType'] = datetime.datetime.today()
a['CalendarType'] = calendar.Calendar()
# numeric and mathematical types (CH 9)
a['DecimalType'] = decimal.Decimal(1)
# data compression and archiving (CH 12)
a['TarInfoType'] = tarfile.TarInfo()
# generic operating system services (CH 15)
a['LoggerType'] = _logger = logging.getLogger()
a['FormatterType'] = logging.Formatter() # pickle ok
a['FilterType'] = logging.Filter() # pickle ok
a['LogRecordType'] = logging.makeLogRecord(_dict) # pickle ok
a['OptionParserType'] = _oparser = optparse.OptionParser() # pickle ok
a['OptionGroupType'] = optparse.OptionGroup(_oparser,"foo") # pickle ok
a['OptionType'] = optparse.Option('--foo') # pickle ok
if HAS_CTYPES:
    z = x if (IS_PYPY and sys.hexversion < 0x30b0df0) else a
    z['CCharType'] = _cchar = ctypes.c_char()
    z['CWCharType'] = ctypes.c_wchar() # fail == 2.6
    z['CByteType'] = ctypes.c_byte()
    z['CUByteType'] = ctypes.c_ubyte()
    z['CShortType'] = ctypes.c_short()
    z['CUShortType'] = ctypes.c_ushort()
    z['CIntType'] = ctypes.c_int()
    z['CUIntType'] = ctypes.c_uint()
    z['CLongType'] = ctypes.c_long()
    z['CULongType'] = ctypes.c_ulong()
    z['CLongLongType'] = ctypes.c_longlong()
    z['CULongLongType'] = ctypes.c_ulonglong()
    z['CFloatType'] = ctypes.c_float()
    z['CDoubleType'] = ctypes.c_double()
    z['CSizeTType'] = ctypes.c_size_t()
    del z
    a['CLibraryLoaderType'] = ctypes.cdll
    a['StructureType'] = _Struct
    # if not IS_PYPY:
    #     a['BigEndianStructureType'] = ctypes.BigEndianStructure()
#NOTE: also LittleEndianStructureType and UnionType... abstract classes
#NOTE: remember for ctypesobj.contents creates a new python object
#NOTE: ctypes.c_int._objects is memberdescriptor for object's __dict__
#NOTE: base class of all ctypes data types is non-public _CData

import fractions
import io
from io import StringIO as TextIO
# built-in functions (CH 2)
a['ByteArrayType'] = bytearray([1])
# numeric and mathematical types (CH 9)
a['FractionType'] = fractions.Fraction()
a['NumberType'] = numbers.Number()
# generic operating system services (CH 15)
a['IOBaseType'] = io.IOBase()
a['RawIOBaseType'] = io.RawIOBase()
a['TextIOBaseType'] = io.TextIOBase()
a['BufferedIOBaseType'] = io.BufferedIOBase()
a['UnicodeIOType'] = TextIO() # the new StringIO
a['LoggerAdapterType'] = logging.LoggerAdapter(_logger,_dict) # pickle ok
if HAS_CTYPES:
    z = x if (IS_PYPY and sys.hexversion < 0x30b0df0) else a
    z['CBoolType'] = ctypes.c_bool(1)
    z['CLongDoubleType'] = ctypes.c_longdouble()
    del z
import argparse
# data types (CH 8)
a['OrderedDictType'] = collections.OrderedDict(_dict)
a['CounterType'] = collections.Counter(_dict)
if HAS_CTYPES:
    z = x if (IS_PYPY and sys.hexversion < 0x30b0df0) else a
    z['CSSizeTType'] = ctypes.c_ssize_t()
    del z
# generic operating system services (CH 15)
a['NullHandlerType'] = logging.NullHandler() # pickle ok  # new 2.7
a['ArgParseFileType'] = argparse.FileType() # pickle ok

# -- pickle fails on all below here -----------------------------------------
# types module (part of CH 8)
a['CodeType'] = compile('','','exec')
a['DictProxyType'] = type.__dict__
a['DictProxyType2'] = _newclass.__dict__
a['EllipsisType'] = Ellipsis
a['ClosedFileType'] = open(os.devnull, 'wb', buffering=0).close()
a['GetSetDescriptorType'] = array.array.typecode
a['LambdaType'] = _lambda = lambda x: lambda y: x #XXX: works when not imported!
a['MemberDescriptorType'] = _newclass2.descriptor
if not IS_PYPY:
    a['MemberDescriptorType2'] = datetime.timedelta.days
a['MethodType'] = _method = _class()._method #XXX: works when not imported!
a['ModuleType'] = datetime
a['NotImplementedType'] = NotImplemented
a['SliceType'] = slice(1)
a['UnboundMethodType'] = _class._method #XXX: works when not imported!
from dill._dill import get_file_type as openfile
d['TextWrapperType'] = openfile('r', buffering=-1) # same as mode='w','w+','r+'
if not IS_PYODIDE:
    d['BufferedRandomType'] = openfile('r+b', buffering=-1) # same as mode='w+b'
d['BufferedReaderType'] = openfile('rb', buffering=-1) # (default: buffering=-1)
d['BufferedWriterType'] = openfile('wb', buffering=-1)
try: # oddities: deprecated
    from _pyio import open as _open
    d['PyTextWrapperType'] = openfile('r', buffering=-1, open=_open)
    if not IS_PYODIDE:
        d['PyBufferedRandomType'] = openfile('r+b', buffering=-1, open=_open)
    d['PyBufferedReaderType'] = openfile('rb', buffering=-1, open=_open)
    d['PyBufferedWriterType'] = openfile('wb', buffering=-1, open=_open)
except ImportError:
    pass
del openfile
# other (concrete) object types
z = d if sys.hexversion < 0x30800a2 else a
z['CellType'] = (_lambda)(0).__closure__[0]
del z
a['XRangeType'] = _xrange = range(1)
a['MethodDescriptorType'] = type.__dict__['mro']
a['WrapperDescriptorType'] = type.__repr__
#a['WrapperDescriptorType2'] = type.__dict__['__module__']#XXX: GetSetDescriptor
a['ClassMethodDescriptorType'] = type.__dict__['__prepare__']
# built-in functions (CH 2)
_methodwrap = (1).__lt__
a['MethodWrapperType'] = _methodwrap
a['StaticMethodType'] = staticmethod(_method)
a['ClassMethodType'] = classmethod(_method)
a['PropertyType'] = property()
d['SuperType'] = super(Exception, _exception)
# string services (CH 7)
_in = _bytes
a['InputType'] = _cstrI = StringIO(_in)
a['OutputType'] = _cstrO = StringIO()
# data types (CH 8)
a['WeakKeyDictionaryType'] = weakref.WeakKeyDictionary()
a['WeakValueDictionaryType'] = weakref.WeakValueDictionary()
a['ReferenceType'] = weakref.ref(_instance)
a['DeadReferenceType'] = weakref.ref(_class())
a['ProxyType'] = weakref.proxy(_instance)
a['DeadProxyType'] = weakref.proxy(_class())
a['CallableProxyType'] = weakref.proxy(_instance2)
a['DeadCallableProxyType'] = weakref.proxy(_class2())
a['QueueType'] = Queue.Queue()
# numeric and mathematical types (CH 9)
d['PartialType'] = functools.partial(int,base=2)
a['IzipType'] = zip('0','1')
d['ItemGetterType'] = operator.itemgetter(0)
d['AttrGetterType'] = operator.attrgetter('__repr__')
# file and directory access (CH 10)
_fileW = _cstrO
# data persistence (CH 11)
if HAS_ALL:
    x['ConnectionType'] = _conn = sqlite3.connect(':memory:')
    x['CursorType'] = _conn.cursor()
a['ShelveType'] = shelve.Shelf({})
# data compression and archiving (CH 12)
if HAS_ALL:
    x['BZ2FileType'] = bz2.BZ2File(os.devnull)
    x['BZ2CompressorType'] = bz2.BZ2Compressor()
    x['BZ2DecompressorType'] = bz2.BZ2Decompressor()
#x['ZipFileType'] = _zip = zipfile.ZipFile(os.devnull,'w')
#_zip.write(_tempfile,'x') [causes annoying warning/error printed on import]
#a['ZipInfoType'] = _zip.getinfo('x')
a['TarFileType'] = tarfile.open(fileobj=_fileW,mode='w')
# file formats (CH 13)
x['DialectType'] = csv.get_dialect('excel')
if sys.hexversion < 0x30d00a1:
    import xdrlib
    a['PackerType'] = xdrlib.Packer()
# optional operating system services (CH 16)
a['LockType'] = threading.Lock()
a['RLockType'] = threading.RLock()
# generic operating system services (CH 15) # also closed/open and r/w/etc...
a['NamedLoggerType'] = _logger = logging.getLogger(__name__)
#a['FrozenModuleType'] = __hello__ #FIXME: prints "Hello world..."
# interprocess communication (CH 17)
x['SocketType'] = _socket = socket.socket()
x['SocketPairType'] = socket.socketpair()[0]
# python runtime services (CH 27)
a['GeneratorContextManagerType'] = contextlib.contextmanager(max)([1])
#a['ContextType'] = contextvars.Context() #XXX: ContextVar

try: # ipython
    __IPYTHON__ is True # is ipython
except NameError:
    # built-in constants (CH 4)
    a['QuitterType'] = quit
    d['ExitType'] = a['QuitterType']
try: # numpy #FIXME: slow... 0.05 to 0.1 sec to import numpy
    from numpy import ufunc as _numpy_ufunc
    from numpy import array as _numpy_array
    from numpy import int32 as _numpy_int32
    a['NumpyUfuncType'] = _numpy_ufunc
    a['NumpyArrayType'] = _numpy_array
    a['NumpyInt32Type'] = _numpy_int32
except ImportError:
    pass
# generic operating system services (CH 15)
a['FileHandlerType'] = logging.FileHandler(os.devnull)
a['RotatingFileHandlerType'] = logging.handlers.RotatingFileHandler(os.devnull)
a['SocketHandlerType'] = logging.handlers.SocketHandler('localhost',514)
a['MemoryHandlerType'] = logging.handlers.MemoryHandler(1)
# data types (CH 8)
a['WeakSetType'] = weakref.WeakSet() # 2.7
# generic operating system services (CH 15) [errors when dill is imported]
#a['ArgumentParserType'] = _parser = argparse.ArgumentParser('PROG')
#a['NamespaceType'] = _parser.parse_args() # pickle ok
#a['SubParsersActionType'] = _parser.add_subparsers()
#a['MutuallyExclusiveGroupType'] = _parser.add_mutually_exclusive_group()
#a['ArgumentGroupType'] = _parser.add_argument_group()

# -- dill fails in some versions below here ---------------------------------
# types module (part of CH 8)
d['FileType'] = open(os.devnull, 'rb', buffering=0) # same 'wb','wb+','rb+'
# built-in functions (CH 2)
# Iterators:
a['ListIteratorType'] = iter(_list) # empty vs non-empty
a['SetIteratorType'] = iter(_set) #XXX: empty vs non-empty #FIXME: list_iterator
a['TupleIteratorType']= iter(_tuple) # empty vs non-empty
a['XRangeIteratorType'] = iter(_xrange) # empty vs non-empty
a["BytesIteratorType"] = iter(b'')
a["BytearrayIteratorType"] = iter(bytearray(b''))
z = x if IS_PYPY else a
z["CallableIteratorType"] = iter(iter, None)
del z
x["MemoryIteratorType"] = iter(memoryview(b''))
a["ListReverseiteratorType"] = reversed([])
X = a['OrderedDictType']
d["OdictKeysType"] = X.keys()
d["OdictValuesType"] = X.values()
d["OdictItemsType"] = X.items()
a["OdictIteratorType"] = iter(X.keys()) #FIXME: list_iterator
del X
#FIXME: list_iterator
a['DictionaryItemIteratorType'] = iter(type.__dict__.items())
a['DictionaryKeyIteratorType'] = iter(type.__dict__.keys())
a['DictionaryValueIteratorType'] = iter(type.__dict__.values())
if sys.hexversion >= 0x30800a0:
    a["DictReversekeyiteratorType"] = reversed({}.keys())
    a["DictReversevalueiteratorType"] = reversed({}.values())
    a["DictReverseitemiteratorType"] = reversed({}.items())

try:
    import symtable
    #FIXME: fails to pickle
    x["SymtableEntryType"] = symtable.symtable("", "string", "exec")._table
except ImportError:
    pass

if sys.hexversion >= 0x30a00a0 and not IS_PYPY:
    x['LineIteratorType'] = compile('3', '', 'eval').co_lines()

if sys.hexversion >= 0x30b00b0 and not IS_PYPY:
    from types import GenericAlias
    d["GenericAliasIteratorType"] = iter(GenericAlias(list, (int,)))
    x['PositionsIteratorType'] = compile('3', '', 'eval').co_positions()

# data types (CH 8)
a['PrettyPrinterType'] = pprint.PrettyPrinter()
# file and directory access (CH 10)
a['TemporaryFileType'] = _tmpf
# data compression and archiving (CH 12)
x['GzipFileType'] = gzip.GzipFile(fileobj=_fileW)
# generic operating system services (CH 15)
a['StreamHandlerType'] = logging.StreamHandler()
# numeric and mathematical types (CH 9)
z = a if sys.hexversion < 0x30e00a1 else x
z['CountType'] = itertools.count(0) #FIXME: __reduce__ removed in 3.14.0a1
z['ChainType'] = itertools.chain('0','1')
z['ProductType'] = itertools.product('0','1')
z['CycleType'] = itertools.cycle('0')
z['PermutationsType'] = itertools.permutations('0')
z['CombinationsType'] = itertools.combinations('0',1)
z['RepeatType'] = itertools.repeat(0)
z['CompressType'] = itertools.compress('0',[1])
del z
#XXX: ...and etc

# -- dill fails on all below here -------------------------------------------
# types module (part of CH 8)
x['GeneratorType'] = _generator = _function(1) #XXX: priority
x['FrameType'] = _generator.gi_frame #XXX: inspect.currentframe()
x['TracebackType'] = _function2()[1] #(see: inspect.getouterframes,getframeinfo)
# other (concrete) object types
# (also: Capsule / CObject ?)
# built-in functions (CH 2)
# built-in types (CH 5)
# string services (CH 7)
x['StructType'] = struct.Struct('c')
x['CallableIteratorType'] = _srepattern.finditer('')
x['SREMatchType'] = _srepattern.match('')
x['SREScannerType'] = _srepattern.scanner('')
x['StreamReader'] = codecs.StreamReader(_cstrI) #XXX: ... and etc
# python object persistence (CH 11)
# x['DbShelveType'] = shelve.open('foo','n')#,protocol=2) #XXX: delete foo
if HAS_ALL:
    z = a if IS_PYPY else x
    z['DbmType'] = dbm.open(_tempfile,'n')
    del z
# x['DbCursorType'] = _dbcursor = anydbm.open('foo','n') #XXX: delete foo
# x['DbType'] = _dbcursor.db
# data compression and archiving (CH 12)
x['ZlibCompressType'] = zlib.compressobj()
x['ZlibDecompressType'] = zlib.decompressobj()
# file formats (CH 13)
x['CSVReaderType'] = csv.reader(_cstrI)
x['CSVWriterType'] = csv.writer(_cstrO)
x['CSVDictReaderType'] = csv.DictReader(_cstrI)
x['CSVDictWriterType'] = csv.DictWriter(_cstrO,{})
# cryptographic services (CH 14)
x['HashType'] = hashlib.md5()
if (sys.hexversion < 0x30800a1):
    x['HMACType'] = hmac.new(_in)
else:
    x['HMACType'] = hmac.new(_in, digestmod='md5')
# generic operating system services (CH 15)
if HAS_CURSES: pass
    #x['CursesWindowType'] = _curwin = curses.initscr() #FIXME: messes up tty
    #x['CursesTextPadType'] = textpad.Textbox(_curwin)
    #x['CursesPanelType'] = panel.new_panel(_curwin)
if HAS_CTYPES:
    x['CCharPType'] = ctypes.c_char_p()
    x['CWCharPType'] = ctypes.c_wchar_p()
    x['CVoidPType'] = ctypes.c_void_p()
    if sys.platform[:3] == 'win':
        x['CDLLType'] = _cdll = ctypes.cdll.msvcrt
    else:
        x['CDLLType'] = _cdll = ctypes.CDLL(None)
    if not IS_PYPY:
        x['PyDLLType'] = _pydll = ctypes.pythonapi
    x['FuncPtrType'] = _cdll._FuncPtr()
    x['CCharArrayType'] = ctypes.create_string_buffer(1)
    x['CWCharArrayType'] = ctypes.create_unicode_buffer(1)
    x['CParamType'] = ctypes.byref(_cchar)
    x['LPCCharType'] = ctypes.pointer(_cchar)
    x['LPCCharObjType'] = _lpchar = ctypes.POINTER(ctypes.c_char)
    x['NullPtrType'] = _lpchar()
    x['NullPyObjectType'] = ctypes.py_object()
    x['PyObjectType'] = ctypes.py_object(lambda :None)
    z = a if IS_PYPY else x
    z['FieldType'] = _field = _Struct._field
    z['CFUNCTYPEType'] = _cfunc = ctypes.CFUNCTYPE(ctypes.c_char)
    if sys.hexversion < 0x30c00b3:
        x['CFunctionType'] = _cfunc(str)
    del z
# numeric and mathematical types (CH 9)
a['MethodCallerType'] = operator.methodcaller('mro') # 2.6
# built-in types (CH 5)
x['MemoryType'] = memoryview(_in) # 2.7
x['MemoryType2'] = memoryview(bytearray(_in)) # 2.7
d['DictItemsType'] = _dict.items() # 2.7
d['DictKeysType'] = _dict.keys() # 2.7
d['DictValuesType'] = _dict.values() # 2.7
# generic operating system services (CH 15)
a['RawTextHelpFormatterType'] = argparse.RawTextHelpFormatter('PROG')
a['RawDescriptionHelpFormatterType'] = argparse.RawDescriptionHelpFormatter('PROG')
a['ArgDefaultsHelpFormatterType'] = argparse.ArgumentDefaultsHelpFormatter('PROG')
z = a if IS_PYPY else x
z['CmpKeyType'] = _cmpkey = functools.cmp_to_key(_methodwrap) # 2.7, >=3.2
z['CmpKeyObjType'] = _cmpkey('0') #2.7, >=3.2
del z
# oddities: removed, etc
x['BufferType'] = x['MemoryType']

from dill._dill import _testcapsule
if _testcapsule is not None:
    d['PyCapsuleType'] = _testcapsule
del _testcapsule

if hasattr(dataclasses, '_HAS_DEFAULT_FACTORY'):
    a['DataclassesHasDefaultFactoryType'] = dataclasses._HAS_DEFAULT_FACTORY

if hasattr(dataclasses, 'MISSING'):
    a['DataclassesMissingType'] = dataclasses.MISSING

if hasattr(dataclasses, 'KW_ONLY'):
    a['DataclassesKWOnlyType'] = dataclasses.KW_ONLY

if hasattr(dataclasses, '_FIELD_BASE'):
    a['DataclassesFieldBaseType'] = dataclasses._FIELD

# -- cleanup ----------------------------------------------------------------
a.update(d) # registered also succeed
if sys.platform[:3] == 'win':
    os.close(_filedescrip) # required on win32
os.remove(_tempfile)


# EOF
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
#
# inspired by inspect.py from Python-2.7.6
# inspect.py author: 'Ka-Ping Yee <ping@lfw.org>'
# inspect.py merged into original dill.source by Mike McKerns 4/13/14
"""
Extensions to python's 'inspect' module, which can be used
to retrieve information from live python objects. The methods
defined in this module are augmented to facilitate access to
source code of interactively defined functions and classes,
as well as provide access to source code for objects defined
in a file.
"""

__all__ = ['findsource', 'getsourcelines', 'getsource', 'indent', 'outdent', \
           '_wrap', 'dumpsource', 'getname', '_namespace', 'getimport', \
           '_importable', 'importable','isdynamic', 'isfrommain']

import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
                     isbuiltin, isclass, iscode, isframe, isfunction, ismethod,
                     ismodule, istraceback)
from tokenize import TokenError

from ._dill import IS_IPYTHON


def isfrommain(obj):
    "check if object was built in __main__"
    module = getmodule(obj)
    if module and module.__name__ == '__main__':
        return True
    return False


def isdynamic(obj):
    "check if object was built in the interpreter"
    try: file = getfile(obj)
    except TypeError: file = None
    if file == '<stdin>' and isfrommain(obj):
        return True
    return False


def _matchlambda(func, line):
    """check if lambda object 'func' matches raw line of code 'line'"""
    from .detect import code as getcode
    from .detect import freevars, globalvars, varnames
    dummy = lambda : '__this_is_a_big_dummy_function__'
    # process the line (removing leading whitespace, etc)
    lhs,rhs = line.split('lambda ',1)[-1].split(":", 1) #FIXME: if !1 inputs
    try: #FIXME: unsafe
        _ = eval("lambda %s : %s" % (lhs,rhs), globals(),locals())
    except Exception: _ = dummy
    # get code objects, for comparison
    _, code = getcode(_).co_code, getcode(func).co_code
    # check if func is in closure
    _f = [line.count(i) for i in freevars(func).keys()]
    if not _f: # not in closure
        # check if code matches
        if _ == code: return True
        return False
    # weak check on freevars
    if not all(_f): return False  #XXX: VERY WEAK
    # weak check on varnames and globalvars
    _f = varnames(func)
    _f = [line.count(i) for i in _f[0]+_f[1]]
    if _f and not all(_f): return False  #XXX: VERY WEAK
    _f = [line.count(i) for i in globalvars(func).keys()]
    if _f and not all(_f): return False  #XXX: VERY WEAK
    # check if func is a double lambda
    if (line.count('lambda ') > 1) and (lhs in freevars(func).keys()):
        _lhs,_rhs = rhs.split('lambda ',1)[-1].split(":",1) #FIXME: if !1 inputs
        try: #FIXME: unsafe
            _f = eval("lambda %s : %s" % (_lhs,_rhs), globals(),locals())
        except Exception: _f = dummy
        # get code objects, for comparison
        _, code = getcode(_f).co_code, getcode(func).co_code
        if len(_) != len(code): return False
        #NOTE: should be same code same order, but except for 't' and '\x88'
        _ = set((i,j) for (i,j) in zip(_,code) if i != j)
        if len(_) != 1: return False #('t','\x88')
        return True
    # check indentsize
    if not indentsize(line): return False #FIXME: is this a good check???
    # check if code 'pattern' matches
    #XXX: or pattern match against dis.dis(code)? (or use uncompyle2?)
    _ = _.split(_[0])  # 't' #XXX: remove matching values if starts the same?
    _f = code.split(code[0])  # '\x88'
    #NOTE: should be same code different order, with different first element
    _ = dict(re.match(r'([\W\D\S])(.*)', _[i]).groups() for i in range(1,len(_)))
    _f = dict(re.match(r'([\W\D\S])(.*)', _f[i]).groups() for i in range(1,len(_f)))
    if (_.keys() == _f.keys()) and (sorted(_.values()) == sorted(_f.values())):
        return True
    return False


def findsource(object):
    """Return the entire source file and starting line number for an object.
    For interactively-defined objects, the 'file' is the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of all the lines
    in the file and the line number indexes a line in that list.  An IOError
    is raised if the source code cannot be retrieved, while a TypeError is
    raised for objects where the source code is unavailable (e.g. builtins)."""

    module = getmodule(object)
    try: file = getfile(module)
    except TypeError: file = None
    is_module_main = (module and module.__name__ == '__main__' and not file)
    if IS_IPYTHON and is_module_main:
        #FIXME: quick fix for functions and classes in IPython interpreter
        try:
            file = getfile(object)
            sourcefile = getsourcefile(object)
        except TypeError:
            if isclass(object):
                for object_method in filter(isfunction, object.__dict__.values()):
                    # look for a method of the class
                    file_candidate = getfile(object_method)
                    if not file_candidate.startswith('<ipython-input-'):
                        continue
                    file = file_candidate
                    sourcefile = getsourcefile(object_method)
                    break
        if file:
            lines = linecache.getlines(file)
        else:
            # fallback to use history
            history = '\n'.join(get_ipython().history_manager.input_hist_parsed)
            lines = [line + '\n' for line in history.splitlines()]
    # use readline when working in interpreter (i.e. __main__ and not file)
    elif is_module_main:
        try:
            import readline
            err = ''
        except ImportError:
            import sys
            err = sys.exc_info()[1].args[0]
            if sys.platform[:3] == 'win':
                err += ", please install 'pyreadline'"
        if err:
            raise IOError(err)
        lbuf = readline.get_current_history_length()
        lines = [readline.get_history_item(i)+'\n' for i in range(1,lbuf+1)]
    else:
        try: # special handling for class instances
            if not isclass(object) and isclass(type(object)): # __class__
                file = getfile(module)
                sourcefile = getsourcefile(module)
            else: # builtins fail with a TypeError
                file = getfile(object)
                sourcefile = getsourcefile(object)
        except (TypeError, AttributeError): # fail with better error
            file = getfile(object)
            sourcefile = getsourcefile(object)
        if not sourcefile and file[:1] + file[-1:] != '<>':
            raise IOError('source code not available')
        file = sourcefile if sourcefile else file

        module = getmodule(object, file)
        if module:
            lines = linecache.getlines(file, module.__dict__)
        else:
            lines = linecache.getlines(file)

    if not lines:
        raise IOError('could not extract source code')

    #FIXME: all below may fail if exec used (i.e. exec('f = lambda x:x') )
    if ismodule(object):
        return lines, 0

    #NOTE: beneficial if search goes from end to start of buffer history
    name = pat1 = obj = ''
    pat2 = r'^(\s*@)'
#   pat1b = r'^(\s*%s\W*=)' % name #FIXME: finds 'f = decorate(f)', not exec
    if ismethod(object):
        name = object.__name__
        if name == '<lambda>': pat1 = r'(.*(?<!\w)lambda(:|\s))'
        else: pat1 = r'^(\s*def\s)'
        object = object.__func__
    if isfunction(object):
        name = object.__name__
        if name == '<lambda>':
            pat1 = r'(.*(?<!\w)lambda(:|\s))'
            obj = object #XXX: better a copy?
        else: pat1 = r'^(\s*def\s)'
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, 'co_firstlineno'):
            raise IOError('could not find function definition')
        stdin = object.co_filename == '<stdin>'
        if stdin:
            lnum = len(lines) - 1 # can't get lnum easily, so leverage pat
            if not pat1: pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)'
        else:
            lnum = object.co_firstlineno - 1
            pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)'
        pat1 = re.compile(pat1); pat2 = re.compile(pat2)
       #XXX: candidate_lnum = [n for n in range(lnum) if pat1.match(lines[n])]
        while lnum > 0: #XXX: won't find decorators in <stdin> ?
            line = lines[lnum]
            if pat1.match(line):
                if not stdin: break # co_firstlineno does the job
                if name == '<lambda>': # hackery needed to confirm a match
                    if _matchlambda(obj, line): break
                else: # not a lambda, just look for the name
                    if name in line: # need to check for decorator...
                        hats = 0
                        for _lnum in range(lnum-1,-1,-1):
                            if pat2.match(lines[_lnum]): hats += 1
                            else: break
                        lnum = lnum - hats
                        break
            lnum = lnum - 1
        return lines, lnum

    try: # turn instances into classes
        if not isclass(object) and isclass(type(object)): # __class__
            object = object.__class__ #XXX: sometimes type(class) is better?
            #XXX: we don't find how the instance was built
    except AttributeError: pass
    if isclass(object):
        name = object.__name__
        pat = re.compile(r'^(\s*)class\s*' + name + r'\b')
        # make some effort to find the best matching class definition:
        # use the one with the least indentation, which is the one
        # that's most probably not inside a function definition.
        candidates = []
        for i in range(len(lines)-1,-1,-1):
            match = pat.match(lines[i])
            if match:
                # if it's at toplevel, it's already the best one
                if lines[i][0] == 'c':
                    return lines, i
                # else add whitespace to candidate list
                candidates.append((match.group(1), i))
        if candidates:
            # this will sort by whitespace, and by line number,
            # less whitespace first  #XXX: should sort high lnum before low
            candidates.sort()
            return lines, candidates[0][1]
        else:
            raise IOError('could not find class definition')
    raise IOError('could not find code object')


def getblocks(object, lstrip=False, enclosing=False, locate=False):
    """Return a list of source lines and starting line number for an object.
    Interactively-defined objects refer to lines in the interpreter's history.

    If enclosing=True, then also return any enclosing code.
    If lstrip=True, ensure there is no indentation in the first line of code.
    If locate=True, then also return the line number for the block of code.

    DEPRECATED: use 'getsourcelines' instead
    """
    lines, lnum = findsource(object)

    if ismodule(object):
        if lstrip: lines = _outdent(lines)
        return ([lines], [0]) if locate is True else [lines]

    #XXX: 'enclosing' means: closures only? or classes and files?
    indent = indentsize(lines[lnum])
    block = getblock(lines[lnum:]) #XXX: catch any TokenError here?

    if not enclosing or not indent:
        if lstrip: block = _outdent(block)
        return ([block], [lnum]) if locate is True else [block]

    pat1 = r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))'; pat1 = re.compile(pat1)
    pat2 = r'^(\s*@)'; pat2 = re.compile(pat2)
   #pat3 = r'^(\s*class\s)'; pat3 = re.compile(pat3) #XXX: enclosing class?
    #FIXME: bound methods need enclosing class (and then instantiation)
    #       *or* somehow apply a partial using the instance

    skip = 0
    line = 0
    blocks = []; _lnum = []
    target = ''.join(block)
    while line <= lnum: #XXX: repeat lnum? or until line < lnum?
        # see if starts with ('def','lambda') and contains our target block
        if pat1.match(lines[line]):
            if not skip:
                try: code = getblock(lines[line:])
                except TokenError: code = [lines[line]]
            if indentsize(lines[line]) > indent: #XXX: should be >= ?
                line += len(code) - skip
            elif target in ''.join(code):
                blocks.append(code) # save code block as the potential winner
                _lnum.append(line - skip) # save the line number for the match
                line += len(code) - skip
            else:
                line += 1
            skip = 0
        # find skip: the number of consecutive decorators
        elif pat2.match(lines[line]):
            try: code = getblock(lines[line:])
            except TokenError: code = [lines[line]]
            skip = 1
            for _line in code[1:]: # skip lines that are decorators
                if not pat2.match(_line): break
                skip += 1
            line += skip
        # no match: reset skip and go to the next line
        else:
            line +=1
            skip = 0

    if not blocks:
        blocks = [block]
        _lnum = [lnum]
    if lstrip: blocks = [_outdent(block) for block in blocks]
    # return last match
    return (blocks, _lnum) if locate is True else blocks


def getsourcelines(object, lstrip=False, enclosing=False):
    """Return a list of source lines and starting line number for an object.
    Interactively-defined objects refer to lines in the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of the lines
    corresponding to the object and the line number indicates where in the
    original source file the first line of code was found.  An IOError is
    raised if the source code cannot be retrieved, while a TypeError is
    raised for objects where the source code is unavailable (e.g. builtins).

    If lstrip=True, ensure there is no indentation in the first line of code.
    If enclosing=True, then also return any enclosing code."""
    code, n = getblocks(object, lstrip=lstrip, enclosing=enclosing, locate=True)
    return code[-1], n[-1]


#NOTE: broke backward compatibility 4/16/14 (was lstrip=True, force=True)
def getsource(object, alias='', lstrip=False, enclosing=False, \
                                              force=False, builtin=False):
    """Return the text of the source code for an object. The source code for
    interactively-defined objects are extracted from the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a single string.  An
    IOError is raised if the source code cannot be retrieved, while a
    TypeError is raised for objects where the source code is unavailable
    (e.g. builtins).

    If alias is provided, then add a line of code that renames the object.
    If lstrip=True, ensure there is no indentation in the first line of code.
    If enclosing=True, then also return any enclosing code.
    If force=True, catch (TypeError,IOError) and try to use import hooks.
    If builtin=True, force an import for any builtins
    """
    # hascode denotes a callable
    hascode = _hascode(object)
    # is a class instance type (and not in builtins)
    instance = _isinstance(object)

    # get source lines; if fail, try to 'force' an import
    try: # fails for builtins, and other assorted object types
        lines, lnum = getsourcelines(object, enclosing=enclosing)
    except (TypeError, IOError): # failed to get source, resort to import hooks
        if not force: # don't try to get types that findsource can't get
            raise
        if not getmodule(object): # get things like 'None' and '1'
            if not instance: return getimport(object, alias, builtin=builtin)
            # special handling (numpy arrays, ...)
            _import = getimport(object, builtin=builtin)
            name = getname(object, force=True)
            _alias = "%s = " % alias if alias else ""
            if alias == name: _alias = ""
            return _import+_alias+"%s\n" % name
        else: #FIXME: could use a good bit of cleanup, since using getimport...
            if not instance: return getimport(object, alias, builtin=builtin)
            # now we are dealing with an instance...
            name = object.__class__.__name__
            module = object.__module__
            if not name.isidentifier() or not all(i.isidentifier() for i in module.split('.')): #XXX: does exclusing malicious code exclude valid use?
                raise SyntaxError('invalid syntax')
            if module in ['builtins','__builtin__']:
                return getimport(object, alias, builtin=builtin)
            else: #FIXME: leverage getimport? use 'from module import name'?
                lines, lnum = ["%s = __import__('%s', fromlist=['%s']).%s\n" % (name,module,name,name)], 0
                obj = eval(lines[0].lstrip(name + ' = '))
                lines, lnum = getsourcelines(obj, enclosing=enclosing)

    # strip leading indent (helps ensure can be imported)
    if lstrip or alias:
        lines = _outdent(lines)

    # instantiate, if there's a nice repr  #XXX: BAD IDEA???
    if instance: #and force: #XXX: move into findsource or getsourcelines ?
        if '(' in repr(object): lines.append('%r\n' % object)
       #else: #XXX: better to somehow to leverage __reduce__ ?
       #    reconstructor,args = object.__reduce__()
       #    _ = reconstructor(*args)
        else: # fall back to serialization #XXX: bad idea?
            #XXX: better not duplicate work? #XXX: better new/enclose=True?
            lines = dumpsource(object, alias='', new=force, enclose=False)
            lines, lnum = [line+'\n' for line in lines.split('\n')][:-1], 0
       #else: object.__code__ # raise AttributeError

    # add an alias to the source code
    if alias:
        if hascode:
            skip = 0
            for line in lines: # skip lines that are decorators
                if not line.startswith('@'): break
                skip += 1
            #XXX: use regex from findsource / getsourcelines ?
            if lines[skip].lstrip().startswith('def '): # we have a function
                if alias != object.__name__:
                    lines.append('\n%s = %s\n' % (alias, object.__name__))
            elif 'lambda ' in lines[skip]: # we have a lambda
                if alias != lines[skip].split('=')[0].strip():
                    lines[skip] = '%s = %s' % (alias, lines[skip])
            else: # ...try to use the object's name
                if alias != object.__name__:
                    lines.append('\n%s = %s\n' % (alias, object.__name__))
        else: # class or class instance
            if instance:
                if alias != lines[-1].split('=')[0].strip():
                    lines[-1] = ('%s = ' % alias) + lines[-1]
            else:
                name = getname(object, force=True) or object.__name__
                if alias != name:
                    lines.append('\n%s = %s\n' % (alias, name))
    return ''.join(lines)


def _hascode(object):
    '''True if object has an attribute that stores it's __code__'''
    return getattr(object,'__code__',None) or getattr(object,'func_code',None)

def _isinstance(object):
    '''True if object is a class instance type (and is not a builtin)'''
    if _hascode(object) or isclass(object) or ismodule(object):
        return False
    if istraceback(object) or isframe(object) or iscode(object):
        return False
    # special handling (numpy arrays, ...)
    if not getmodule(object) and getmodule(type(object)).__name__ in ['numpy']:
        return True
#   # check if is instance of a builtin
#   if not getmodule(object) and getmodule(type(object)).__name__ in ['__builtin__','builtins']:
#       return False
    _types = ('<class ',"<type 'instance'>")
    if not repr(type(object)).startswith(_types): #FIXME: weak hack
        return False
    if not getmodule(object) or object.__module__ in ['builtins','__builtin__'] or getname(object, force=True) in ['array']:
        return False
    return True # by process of elimination... it's what we want


def _intypes(object):
    '''check if object is in the 'types' module'''
    import types
    # allow user to pass in object or object.__name__
    if type(object) is not type(''):
        object = getname(object, force=True)
    if object == 'ellipsis': object = 'EllipsisType'
    return True if hasattr(types, object) else False


def _isstring(object): #XXX: isstringlike better?
    '''check if object is a string-like type'''
    return isinstance(object, (str, bytes))


def indent(code, spaces=4):
    '''indent a block of code with whitespace (default is 4 spaces)'''
    indent = indentsize(code)
    from numbers import Integral
    if isinstance(spaces, Integral): spaces = ' '*spaces
    # if '\t' is provided, will indent with a tab
    nspaces = indentsize(spaces)
    # blank lines (etc) need to be ignored
    lines = code.split('\n')
##  stq = "'''"; dtq = '"""'
##  in_stq = in_dtq = False
    for i in range(len(lines)):
        #FIXME: works... but shouldn't indent 2nd+ lines of multiline doc
        _indent = indentsize(lines[i])
        if indent > _indent: continue
        lines[i] = spaces+lines[i]
##      #FIXME: may fail when stq and dtq in same line (depends on ordering)
##      nstq, ndtq = lines[i].count(stq), lines[i].count(dtq)
##      if not in_dtq and not in_stq:
##          lines[i] = spaces+lines[i] # we indent
##          # entering a comment block
##          if nstq%2: in_stq = not in_stq
##          if ndtq%2: in_dtq = not in_dtq
##      # leaving a comment block
##      elif in_dtq and ndtq%2: in_dtq = not in_dtq
##      elif in_stq and nstq%2: in_stq = not in_stq
##      else: pass
    if lines[-1].strip() == '': lines[-1] = ''
    return '\n'.join(lines)


def _outdent(lines, spaces=None, all=True):
    '''outdent lines of code, accounting for docs and line continuations'''
    indent = indentsize(lines[0])
    if spaces is None or spaces > indent or spaces < 0: spaces = indent
    for i in range(len(lines) if all else 1):
        #FIXME: works... but shouldn't outdent 2nd+ lines of multiline doc
        _indent = indentsize(lines[i])
        if spaces > _indent: _spaces = _indent
        else: _spaces = spaces
        lines[i] = lines[i][_spaces:]
    return lines

def outdent(code, spaces=None, all=True):
    '''outdent a block of code (default is to strip all leading whitespace)'''
    indent = indentsize(code)
    if spaces is None or spaces > indent or spaces < 0: spaces = indent
    #XXX: will this delete '\n' in some cases?
    if not all: return code[spaces:]
    return '\n'.join(_outdent(code.split('\n'), spaces=spaces, all=all))


# _wrap provides an wrapper to correctly exec and load into locals
__globals__ = globals()
__locals__ = locals()
def _wrap(f):
    """ encapsulate a function and it's __import__ """
    def func(*args, **kwds):
        try:
            # _ = eval(getsource(f, force=True)) #XXX: safer but less robust
            exec(getimportable(f, alias='_'), __globals__, __locals__)
        except Exception:
            raise ImportError('cannot import name ' + f.__name__)
        return _(*args, **kwds)
    func.__name__ = f.__name__
    func.__doc__ = f.__doc__
    return func


def _enclose(object, alias=''): #FIXME: needs alias to hold returned object
    """create a function enclosure around the source of some object"""
    #XXX: dummy and stub should append a random string
    dummy = '__this_is_a_big_dummy_enclosing_function__'
    stub = '__this_is_a_stub_variable__'
    code = 'def %s():\n' % dummy
    code += indent(getsource(object, alias=stub, lstrip=True, force=True))
    code += indent('return %s\n' % stub)
    if alias: code += '%s = ' % alias
    code += '%s(); del %s\n' % (dummy, dummy)
   #code += "globals().pop('%s',lambda :None)()\n" % dummy
    return code


def dumpsource(object, alias='', new=False, enclose=True):
    """'dump to source', where the code includes a pickled object.

    If new=True and object is a class instance, then create a new
    instance using the unpacked class source code. If enclose, then
    create the object inside a function enclosure (thus minimizing
    any global namespace pollution).
    """
    from dill import dumps
    pik = repr(dumps(object))
    code = 'import dill\n'
    if enclose:
        stub = '__this_is_a_stub_variable__' #XXX: *must* be same _enclose.stub
        pre = '%s = ' % stub
        new = False #FIXME: new=True doesn't work with enclose=True
    else:
        stub = alias
        pre = '%s = ' % stub if alias else alias

    # if a 'new' instance is not needed, then just dump and load
    if not new or not _isinstance(object):
        code += pre + 'dill.loads(%s)\n' % pik
    else: #XXX: other cases where source code is needed???
        code += getsource(object.__class__, alias='', lstrip=True, force=True)
        mod = repr(object.__module__) # should have a module (no builtins here)
        code += pre + 'dill.loads(%s.replace(b%s,bytes(__name__,"UTF-8")))\n' % (pik,mod)
       #code += 'del %s' % object.__class__.__name__ #NOTE: kills any existing!

    if enclose:
        # generation of the 'enclosure'
        dummy = '__this_is_a_big_dummy_object__'
        dummy = _enclose(dummy, alias=alias)
        # hack to replace the 'dummy' with the 'real' code
        dummy = dummy.split('\n')
        code = dummy[0]+'\n' + indent(code) + '\n'.join(dummy[-3:])

    return code #XXX: better 'dumpsourcelines', returning list of lines?


def getname(obj, force=False, fqn=False): #XXX: throw(?) to raise error on fail?
    """get the name of the object. for lambdas, get the name of the pointer """
    if fqn: return '.'.join(_namespace(obj)) #NOTE: returns 'type'
    module = getmodule(obj)
    if not module: # things like "None" and "1"
        if not force: return None #NOTE: returns 'instance' NOT 'type' #FIXME?
        # handle some special cases
        if hasattr(obj, 'dtype') and not obj.shape:
            return getname(obj.__class__) + "(" + repr(obj.tolist()) + ")" 
        return repr(obj)
    try:
        #XXX: 'wrong' for decorators and curried functions ?
        #       if obj.func_closure: ...use logic from getimportable, etc ?
        name = obj.__name__
        if name == '<lambda>':
            return getsource(obj).split('=',1)[0].strip()
        # handle some special cases
        if module.__name__ in ['builtins','__builtin__']:
            if name == 'ellipsis': name = 'EllipsisType'
        return name
    except AttributeError: #XXX: better to just throw AttributeError ?
        if not force: return None
        name = repr(obj)
        if name.startswith('<'): # or name.split('('):
            return None
        return name


def _namespace(obj):
    """_namespace(obj); return namespace hierarchy (as a list of names)
    for the given object.  For an instance, find the class hierarchy.

    For example:

    >>> from functools import partial
    >>> p = partial(int, base=2)
    >>> _namespace(p)
    [\'functools\', \'partial\']
    """
    # mostly for functions and modules and such
    #FIXME: 'wrong' for decorators and curried functions
    try: #XXX: needs some work and testing on different types
        module = qual = str(getmodule(obj)).split()[1].strip('>').strip('"').strip("'")
        qual = qual.split('.')
        if ismodule(obj):
            return qual
        # get name of a lambda, function, etc
        name = getname(obj) or obj.__name__ # failing, raise AttributeError
        # check special cases (NoneType, ...)
        if module in ['builtins','__builtin__']: # BuiltinFunctionType
            if _intypes(name): return ['types'] + [name]
        return qual + [name] #XXX: can be wrong for some aliased objects
    except Exception: pass
    # special case: numpy.inf and numpy.nan (we don't want them as floats)
    if str(obj) in ['inf','nan','Inf','NaN']: # is more, but are they needed?
        return ['numpy'] + [str(obj)]
    # mostly for classes and class instances and such
    module = getattr(obj.__class__, '__module__', None)
    qual = str(obj.__class__)
    try: qual = qual[qual.index("'")+1:-2]
    except ValueError: pass # str(obj.__class__) made the 'try' unnecessary
    qual = qual.split(".")
    if module in ['builtins','__builtin__']:
        # check special cases (NoneType, Ellipsis, ...)
        if qual[-1] == 'ellipsis': qual[-1] = 'EllipsisType'
        if _intypes(qual[-1]): module = 'types' #XXX: BuiltinFunctionType
        qual = [module] + qual
    return qual


#NOTE: 05/25/14 broke backward compatibility: added 'alias' as 3rd argument
def _getimport(head, tail, alias='', verify=True, builtin=False):
    """helper to build a likely import string from head and tail of namespace.
    ('head','tail') are used in the following context: "from head import tail"

    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If alias is provided, then rename the object on import.
    """
    # special handling for a few common types
    if tail in ['Ellipsis', 'NotImplemented'] and head in ['types']:
        head = len.__module__
    elif tail in ['None'] and head in ['types']:
        _alias = '%s = ' % alias if alias else ''
        if alias == tail: _alias = ''
        return _alias+'%s\n' % tail
    # we don't need to import from builtins, so return ''
#   elif tail in ['NoneType','int','float','long','complex']: return '' #XXX: ?
    if head in ['builtins','__builtin__']:
        # special cases (NoneType, Ellipsis, ...) #XXX: BuiltinFunctionType
        if tail == 'ellipsis': tail = 'EllipsisType'
        if _intypes(tail): head = 'types'
        elif not builtin:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail: _alias = ''
            return _alias+'%s\n' % tail
        else: pass # handle builtins below
    # get likely import string
    if not head: _str = "import %s" % tail
    else: _str = "from %s import %s" % (head, tail)
    _alias = " as %s\n" % alias if alias else "\n"
    if alias == tail: _alias = "\n"
    _str += _alias
    # FIXME: fails on most decorators, currying, and such...
    #        (could look for magic __wrapped__ or __func__ attr)
    #        (could fix in 'namespace' to check obj for closure)
    if verify and not head.startswith('dill.'):# weird behavior for dill
       #print(_str)
        try: exec(_str) #XXX: check if == obj? (name collision)
        except ImportError: #XXX: better top-down or bottom-up recursion?
            _head = head.rsplit(".",1)[0] #(or get all, then compare == obj?)
            if not _head: raise
            if _head != head:
                _str = _getimport(_head, tail, alias, verify)
    return _str


#XXX: rename builtin to force? vice versa? verify to force? (as in getsource)
#NOTE: 05/25/14 broke backward compatibility: added 'alias' as 2nd argument
def getimport(obj, alias='', verify=True, builtin=False, enclosing=False):
    """get the likely import string for the given object

    obj is the object to inspect
    If verify=True, then test the import string before returning it.
    If builtin=True, then force an import for builtins where possible.
    If enclosing=True, get the import for the outermost enclosing callable.
    If alias is provided, then rename the object on import.
    """
    if enclosing:
        from .detect import outermost
        _obj = outermost(obj)
        obj = _obj if _obj else obj
    # get the namespace
    qual = _namespace(obj)
    head = '.'.join(qual[:-1])
    tail = qual[-1]
    # for named things... with a nice repr #XXX: move into _namespace?
    try: # look for '<...>' and be mindful it might be in lists, dicts, etc...
        name = repr(obj).split('<',1)[1].split('>',1)[1]
        name = None # we have a 'object'-style repr
    except Exception: # it's probably something 'importable'
        if head in ['builtins','__builtin__']:
            name = repr(obj) #XXX: catch [1,2], (1,2), set([1,2])... others?
        elif _isinstance(obj):
            name = getname(obj, force=True).split('(')[0]
        else:
            name = repr(obj).split('(')[0]
   #if not repr(obj).startswith('<'): name = repr(obj).split('(')[0]
   #else: name = None
    if name: # try using name instead of tail
        try: return _getimport(head, name, alias, verify, builtin)
        except ImportError: pass
        except SyntaxError:
            if head in ['builtins','__builtin__']:
                _alias = '%s = ' % alias if alias else ''
                if alias == name: _alias = ''
                return _alias+'%s\n' % name
            else: pass
    try:
       #if type(obj) is type(abs): _builtin = builtin # BuiltinFunctionType
       #else: _builtin = False
        return _getimport(head, tail, alias, verify, builtin)
    except ImportError:
        raise # could do some checking against obj
    except SyntaxError:
        if head in ['builtins','__builtin__']:
            _alias = '%s = ' % alias if alias else ''
            if alias == tail: _alias = ''
            return _alias+'%s\n' % tail
        raise # could do some checking against obj


def _importable(obj, alias='', source=None, enclosing=False, force=True, \
                                              builtin=True, lstrip=True):
    """get an import string (or the source code) for the given object

    This function will attempt to discover the name of the object, or the repr
    of the object, or the source code for the object. To attempt to force
    discovery of the source code, use source=True, to attempt to force the
    use of an import, use source=False; otherwise an import will be sought
    for objects not defined in __main__. The intent is to build a string
    that can be imported from a python file. obj is the object to inspect.
    If alias is provided, then rename the object with the given alias.

    If source=True, use these options:
      If enclosing=True, then also return any enclosing code.
      If force=True, catch (TypeError,IOError) and try to use import hooks.
      If lstrip=True, ensure there is no indentation in the first line of code.

    If source=False, use these options:
      If enclosing=True, get the import for the outermost enclosing callable.
      If force=True, then don't test the import string before returning it.
      If builtin=True, then force an import for builtins where possible.
    """
    if source is None:
        source = True if isfrommain(obj) else False
    if source: # first try to get the source
        try:
            return getsource(obj, alias, enclosing=enclosing, \
                             force=force, lstrip=lstrip, builtin=builtin)
        except Exception: pass
    try:
        if not _isinstance(obj):
            return getimport(obj, alias, enclosing=enclosing, \
                                  verify=(not force), builtin=builtin)
        # first 'get the import', then 'get the instance'
        _import = getimport(obj, enclosing=enclosing, \
                                 verify=(not force), builtin=builtin)
        name = getname(obj, force=True)
        if not name:
            raise AttributeError("object has no atribute '__name__'")
        _alias = "%s = " % alias if alias else ""
        if alias == name: _alias = ""
        return _import+_alias+"%s\n" % name

    except Exception: pass
    if not source: # try getsource, only if it hasn't been tried yet
        try:
            return getsource(obj, alias, enclosing=enclosing, \
                             force=force, lstrip=lstrip, builtin=builtin)
        except Exception: pass
    # get the name (of functions, lambdas, and classes)
    # or hope that obj can be built from the __repr__
    #XXX: what to do about class instances and such?
    obj = getname(obj, force=force)
    # we either have __repr__ or __name__ (or None)
    if not obj or obj.startswith('<'):
        raise AttributeError("object has no atribute '__name__'")
    _alias = '%s = ' % alias if alias else ''
    if alias == obj: _alias = ''
    return _alias+'%s\n' % obj
    #XXX: possible failsafe... (for example, for instances when source=False)
    #     "import dill; result = dill.loads(<pickled_object>); # repr(<object>)"

def _closuredimport(func, alias='', builtin=False):
    """get import for closured objects; return a dict of 'name' and 'import'"""
    import re
    from .detect import freevars, outermost
    free_vars = freevars(func)
    func_vars = {}
    # split into 'funcs' and 'non-funcs'
    for name,obj in list(free_vars.items()):
        if not isfunction(obj): continue
        # get import for 'funcs'
        fobj = free_vars.pop(name)
        src = getsource(fobj)
        if src.lstrip().startswith('@'): # we have a decorator
            src = getimport(fobj, alias=alias, builtin=builtin)
        else: # we have to "hack" a bit... and maybe be lucky
            encl = outermost(func)
            # pattern: 'func = enclosing(fobj'
            pat = r'.*[\w\s]=\s*'+getname(encl)+r'\('+getname(fobj)
            mod = getname(getmodule(encl))
            #HACK: get file containing 'outer' function; is func there?
            lines,_ = findsource(encl)
            candidate = [line for line in lines if getname(encl) in line and \
                         re.match(pat, line)]
            if not candidate:
                mod = getname(getmodule(fobj))
                #HACK: get file containing 'inner' function; is func there?
                lines,_ = findsource(fobj)
                candidate = [line for line in lines \
                             if getname(fobj) in line and re.match(pat, line)]
            if not len(candidate): raise TypeError('import could not be found')
            candidate = candidate[-1]
            name = candidate.split('=',1)[0].split()[-1].strip()
            src = _getimport(mod, name, alias=alias, builtin=builtin)
        func_vars[name] = src
    if not func_vars:
        name = outermost(func)
        mod = getname(getmodule(name))
        if not mod or name is func: # then it can be handled by getimport
            name = getname(func, force=True) #XXX: better key?
            src = getimport(func, alias=alias, builtin=builtin)
        else:
            lines,_ = findsource(name)
            # pattern: 'func = enclosing('
            candidate = [line for line in lines if getname(name) in line and \
                         re.match(r'.*[\w\s]=\s*'+getname(name)+r'\(', line)]
            if not len(candidate): raise TypeError('import could not be found')
            candidate = candidate[-1]
            name = candidate.split('=',1)[0].split()[-1].strip()
            src = _getimport(mod, name, alias=alias, builtin=builtin)
        func_vars[name] = src
    return func_vars

#XXX: should be able to use __qualname__
def _closuredsource(func, alias=''):
    """get source code for closured objects; return a dict of 'name'
    and 'code blocks'"""
    #FIXME: this entire function is a messy messy HACK
    #      - pollutes global namespace
    #      - fails if name of freevars are reused
    #      - can unnecessarily duplicate function code
    from .detect import freevars
    free_vars = freevars(func)
    func_vars = {}
    # split into 'funcs' and 'non-funcs'
    for name,obj in list(free_vars.items()):
        if not isfunction(obj):
            # get source for 'non-funcs'
            free_vars[name] = getsource(obj, force=True, alias=name)
            continue
        # get source for 'funcs'
        fobj = free_vars.pop(name)
        src = getsource(fobj, alias) # DO NOT include dependencies
        # if source doesn't start with '@', use name as the alias
        if not src.lstrip().startswith('@'): #FIXME: 'enclose' in dummy;
            src = importable(fobj,alias=name)#        wrong ref 'name'
            org = getsource(func, alias, enclosing=False, lstrip=True)
            src = (src, org) # undecorated first, then target
        else: #NOTE: reproduces the code!
            org = getsource(func, enclosing=True, lstrip=False)
            src = importable(fobj, alias, source=True) # include dependencies
            src = (org, src) # target first, then decorated
        func_vars[name] = src
    src = ''.join(free_vars.values())
    if not func_vars: #FIXME: 'enclose' in dummy; wrong ref 'name'
        org = getsource(func, alias, force=True, enclosing=False, lstrip=True)
        src = (src, org) # variables first, then target
    else:
        src = (src, None) # just variables        (better '' instead of None?)
    func_vars[None] = src
    # FIXME: remove duplicates (however, order is important...)
    return func_vars

def importable(obj, alias='', source=None, builtin=True):
    """get an importable string (i.e. source code or the import string)
    for the given object, including any required objects from the enclosing
    and global scope

    This function will attempt to discover the name of the object, or the repr
    of the object, or the source code for the object. To attempt to force
    discovery of the source code, use source=True, to attempt to force the
    use of an import, use source=False; otherwise an import will be sought
    for objects not defined in __main__. The intent is to build a string
    that can be imported from a python file.

    obj is the object to inspect. If alias is provided, then rename the
    object with the given alias. If builtin=True, then force an import for
    builtins where possible.
    """
    #NOTE: we always 'force', and 'lstrip' as necessary
    #NOTE: for 'enclosing', use importable(outermost(obj))
    if source is None:
        source = True if isfrommain(obj) else False
    elif builtin and isbuiltin(obj):
        source = False
    tried_source = tried_import = False
    while True:
        if not source: # we want an import
            try:
                if _isinstance(obj): # for instances, punt to _importable
                    return _importable(obj, alias, source=False, builtin=builtin)
                src = _closuredimport(obj, alias=alias, builtin=builtin)
                if len(src) == 0:
                    raise NotImplementedError('not implemented')
                if len(src) > 1:
                    raise NotImplementedError('not implemented')
                return list(src.values())[0]
            except Exception:
                if tried_source: raise
                tried_import = True
        # we want the source
        try:
            src = _closuredsource(obj, alias=alias)
            if len(src) == 0:
                raise NotImplementedError('not implemented')
            # groan... an inline code stitcher
            def _code_stitcher(block):
                "stitch together the strings in tuple 'block'"
                if block[0] and block[-1]: block = '\n'.join(block)
                elif block[0]: block = block[0]
                elif block[-1]: block = block[-1]
                else: block = ''
                return block
            # get free_vars first
            _src = _code_stitcher(src.pop(None))
            _src = [_src] if _src else []
            # get func_vars
            for xxx in src.values():
                xxx = _code_stitcher(xxx)
                if xxx: _src.append(xxx)
            # make a single source string
            if not len(_src):
                src = ''
            elif len(_src) == 1:
                src = _src[0]
            else:
                src = '\n'.join(_src)
            # get source code of objects referred to by obj in global scope
            from .detect import globalvars
            obj = globalvars(obj) #XXX: don't worry about alias? recurse? etc?
            obj = list(getsource(_obj,name,force=True) for (name,_obj) in obj.items() if not isbuiltin(_obj))
            obj = '\n'.join(obj) if obj else ''
            # combine all referred-to source (global then enclosing)
            if not obj: return src
            if not src: return obj
            return obj + src
        except Exception:
            if tried_import: raise
            tried_source = True
            source = not source
    # should never get here
    return


# backward compatibility
def getimportable(obj, alias='', byname=True, explicit=False):
    return importable(obj,alias,source=(not byname),builtin=explicit)
   #return outdent(_importable(obj,alias,source=(not byname),builtin=explicit))
def likely_import(obj, passive=False, explicit=False):
    return getimport(obj, verify=(not passive), builtin=explicit)
def _likely_import(first, last, passive=False, explicit=True):
    return _getimport(first, last, verify=(not passive), builtin=explicit)
_get_name = getname
getblocks_from_history = getblocks



# EOF
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

"""
Module to show if an object has changed since it was memorised
"""

import builtins
import os
import sys
import types
try:
    import numpy.ma
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# pypy doesn't use reference counting
getrefcount = getattr(sys, 'getrefcount', lambda x:0)

# memo of objects indexed by id to a tuple (attributes, sequence items)
# attributes is a dict indexed by attribute name to attribute id
# sequence items is either a list of ids, of a dictionary of keys to ids
memo = {}
id_to_obj = {}
# types that cannot have changing attributes
builtins_types = set((str, list, dict, set, frozenset, int))
dont_memo = set(id(i) for i in (memo, sys.modules, sys.path_importer_cache,
             os.environ, id_to_obj))


def get_attrs(obj):
    """
    Gets all the attributes of an object though its __dict__ or return None
    """
    if type(obj) in builtins_types \
       or type(obj) is type and obj in builtins_types:
        return
    return getattr(obj, '__dict__', None)


def get_seq(obj, cache={str: False, frozenset: False, list: True, set: True,
                        dict: True, tuple: True, type: False,
                        types.ModuleType: False, types.FunctionType: False,
                        types.BuiltinFunctionType: False}):
    """
    Gets all the items in a sequence or return None
    """
    try:
        o_type = obj.__class__
    except AttributeError:
        o_type = type(obj)
    hsattr = hasattr
    if o_type in cache:
        if cache[o_type]:
            if hsattr(obj, "copy"):
                return obj.copy()
            return obj
    elif HAS_NUMPY and o_type in (numpy.ndarray, numpy.ma.core.MaskedConstant):
        if obj.shape and obj.size:
            return obj
        else:
            return []
    elif hsattr(obj, "__contains__") and hsattr(obj, "__iter__") \
       and hsattr(obj, "__len__") and hsattr(o_type, "__contains__") \
       and hsattr(o_type, "__iter__") and hsattr(o_type, "__len__"):
        cache[o_type] = True
        if hsattr(obj, "copy"):
            return obj.copy()
        return obj
    else:
        cache[o_type] = False
        return None


def memorise(obj, force=False):
    """
    Adds an object to the memo, and recursively adds all the objects
    attributes, and if it is a container, its items. Use force=True to update
    an object already in the memo. Updating is not recursively done.
    """
    obj_id = id(obj)
    if obj_id in memo and not force or obj_id in dont_memo:
        return
    id_ = id
    g = get_attrs(obj)
    if g is None:
        attrs_id = None
    else:
        attrs_id = dict((key,id_(value)) for key, value in g.items())

    s = get_seq(obj)
    if s is None:
        seq_id = None
    elif hasattr(s, "items"):
        seq_id = dict((id_(key),id_(value)) for key, value in s.items())
    elif not hasattr(s, "__len__"): #XXX: avoid TypeError from unexpected case
        seq_id = None
    else:
        seq_id = [id_(i) for i in s]

    memo[obj_id] = attrs_id, seq_id
    id_to_obj[obj_id] = obj
    mem = memorise
    if g is not None:
        [mem(value) for key, value in g.items()]

    if s is not None:
        if hasattr(s, "items"):
            [(mem(key), mem(item))
             for key, item in s.items()]
        else:
            if hasattr(s, '__len__'):
                [mem(item) for item in s]
            else: mem(s)


def release_gone():
    itop, mp, src = id_to_obj.pop, memo.pop, getrefcount
    [(itop(id_), mp(id_)) for id_, obj in list(id_to_obj.items())
     if src(obj) < 4] #XXX: correct for pypy?


def whats_changed(obj, seen=None, simple=False, first=True):
    """
    Check an object against the memo. Returns a list in the form
    (attribute changes, container changed). Attribute changes is a dict of
    attribute name to attribute value. container changed is a boolean.
    If simple is true, just returns a boolean. None for either item means
    that it has not been checked yet
    """
    # Special cases
    if first:
        # ignore the _ variable, which only appears in interactive sessions
        if "_" in builtins.__dict__:
            del builtins._
        if seen is None:
            seen = {}

    obj_id = id(obj)

    if obj_id in seen:
        if simple:
            return any(seen[obj_id])
        return seen[obj_id]

    # Safety checks
    if obj_id in dont_memo:
        seen[obj_id] = [{}, False]
        if simple:
            return False
        return seen[obj_id]
    elif obj_id not in memo:
        if simple:
            return True
        else:
            raise RuntimeError("Object not memorised " + str(obj))

    seen[obj_id] = ({}, False)

    chngd = whats_changed
    id_ = id

    # compare attributes
    attrs = get_attrs(obj)
    if attrs is None:
        changed = {}
    else:
        obj_attrs = memo[obj_id][0]
        obj_get = obj_attrs.get
        changed = dict((key,None) for key in obj_attrs if key not in attrs)
        for key, o in attrs.items():
            if id_(o) != obj_get(key, None) or chngd(o, seen, True, False):
                changed[key] = o

    # compare sequence
    items = get_seq(obj)
    seq_diff = False
    if (items is not None) and (hasattr(items, '__len__')):
        obj_seq = memo[obj_id][1]
        if (len(items) != len(obj_seq)):
            seq_diff = True
        elif hasattr(obj, "items"):  # dict type obj
            obj_get = obj_seq.get
            for key, item in items.items():
                if id_(item) != obj_get(id_(key)) \
                   or chngd(key, seen, True, False) \
                   or chngd(item, seen, True, False):
                    seq_diff = True
                    break
        else:
            for i, j in zip(items, obj_seq):  # list type obj
                if id_(i) != j or chngd(i, seen, True, False):
                    seq_diff = True
                    break
    seen[obj_id] = changed, seq_diff
    if simple:
        return changed or seq_diff
    return changed, seq_diff


def has_changed(*args, **kwds):
    kwds['simple'] = True  # ignore simple if passed in
    return whats_changed(*args, **kwds)

__import__ = __import__


def _imp(*args, **kwds):
    """
    Replaces the default __import__, to allow a module to be memorised
    before the user can change it
    """
    before = set(sys.modules.keys())
    mod = __import__(*args, **kwds)
    after = set(sys.modules.keys()).difference(before)
    for m in after:
        memorise(sys.modules[m])
    return mod

builtins.__import__ = _imp
if hasattr(builtins, "_"):
    del builtins._

# memorise all already imported modules. This implies that this must be
# imported first for any changes to be recorded
for mod in list(sys.modules.values()):
    memorise(mod)
release_gone()
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

__all__ = ['parent', 'reference', 'at', 'parents', 'children']

import gc
import sys

from ._dill import _proxy_helper as reference
from ._dill import _locate_object as at

def parent(obj, objtype, ignore=()):
    """
>>> listiter = iter([4,5,6,7])
>>> obj = parent(listiter, list)
>>> obj == [4,5,6,7]  # actually 'is', but don't have handle any longer
True

NOTE: objtype can be a single type (e.g. int or list) or a tuple of types.

WARNING: if obj is a sequence (e.g. list), may produce unexpected results.
Parent finds *one* parent (e.g. the last member of the sequence).
    """
    depth = 1 #XXX: always looking for the parent (only, right?)
    chain = parents(obj, objtype, depth, ignore)
    parent = chain.pop()
    if parent is obj:
        return None
    return parent


def parents(obj, objtype, depth=1, ignore=()): #XXX: objtype=object ?
    """Find the chain of referents for obj. Chain will end with obj.

    objtype: an object type or tuple of types to search for
    depth: search depth (e.g. depth=2 is 'grandparents')
    ignore: an object or tuple of objects to ignore in the search
    """
    edge_func = gc.get_referents # looking for refs, not back_refs
    predicate = lambda x: isinstance(x, objtype) # looking for parent type
   #if objtype is None: predicate = lambda x: True #XXX: in obj.mro() ?
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth)[::-1]
    #XXX: should pop off obj... ?
    return chain


def children(obj, objtype, depth=1, ignore=()): #XXX: objtype=object ?
    """Find the chain of referrers for obj. Chain will start with obj.

    objtype: an object type or tuple of types to search for
    depth: search depth (e.g. depth=2 is 'grandchildren')
    ignore: an object or tuple of objects to ignore in the search

    NOTE: a common thing to ignore is all globals, 'ignore=(globals(),)'

    NOTE: repeated calls may yield different results, as python stores
    the last value in the special variable '_'; thus, it is often good
    to execute something to replace '_' (e.g. >>> 1+1).
    """
    edge_func = gc.get_referrers # looking for back_refs, not refs
    predicate = lambda x: isinstance(x, objtype) # looking for child type
   #if objtype is None: predicate = lambda x: True #XXX: in obj.mro() ?
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth, ignore)
    #XXX: should pop off obj... ?
    return chain


# more generic helper function (cut-n-paste from objgraph)
# Source at http://mg.pov.lt/objgraph/
# Copyright (c) 2008-2010 Marius Gedminas <marius@pov.lt>
# Copyright (c) 2010 Stefano Rivera <stefano@rivera.za.net>
# Released under the MIT licence (see objgraph/objgrah.py)

def find_chain(obj, predicate, edge_func, max_depth=20, extra_ignore=()):
    queue = [obj]
    depth = {id(obj): 0}
    parent = {id(obj): None}
    ignore = set(extra_ignore)
    ignore.add(id(extra_ignore))
    ignore.add(id(queue))
    ignore.add(id(depth))
    ignore.add(id(parent))
    ignore.add(id(ignore))
    ignore.add(id(sys._getframe()))  # this function
    ignore.add(id(sys._getframe(1))) # find_chain/find_backref_chain, likely
    gc.collect()
    while queue:
        target = queue.pop(0)
        if predicate(target):
            chain = [target]
            while parent[id(target)] is not None:
                target = parent[id(target)]
                chain.append(target)
            return chain
        tdepth = depth[id(target)]
        if tdepth < max_depth:
            referrers = edge_func(target)
            ignore.add(id(referrers))
            for source in referrers:
                if id(source) in ignore:
                    continue
                if id(source) not in depth:
                    depth[id(source)] = tdepth + 1
                    parent[id(source)] = target
                    queue.append(source)
    return [obj] # not found


# backward compatibility
refobject = at


# EOF
<|endoftext|>#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Leonardo Gama (@leogama)
# Copyright (c) 2022-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Logging utilities for dill.

The 'logger' object is dill's top-level logger.

The 'adapter' object wraps the logger and implements a 'trace()' method that
generates a detailed tree-style trace for the pickling call at log level INFO.

The 'trace()' function sets and resets dill's logger log level, enabling and
disabling the pickling trace.

The trace shows a tree structure depicting the depth of each object serialized
*with dill save functions*, but not the ones that use save functions from
'pickle._Pickler.dispatch'. If the information is available, it also displays
the size in bytes that the object contributed to the pickle stream (including
its child objects).  Sample trace output:

    >>> import dill, dill.tests
    >>> dill.detect.trace(True)
    >>> dill.dump_session(main=dill.tests)
    ┬ M1: <module 'dill.tests' from '.../dill/tests/__init__.py'>
    ├┬ F2: <function _import_module at 0x7f0d2dce1b80>
    │└ # F2 [32 B]
    ├┬ D2: <dict object at 0x7f0d2e98a540>
    │├┬ T4: <class '_frozen_importlib.ModuleSpec'>
    ││└ # T4 [35 B]
    │├┬ D2: <dict object at 0x7f0d2ef0e8c0>
    ││├┬ T4: <class '_frozen_importlib_external.SourceFileLoader'>
    │││└ # T4 [50 B]
    ││├┬ D2: <dict object at 0x7f0d2e988a40>
    │││└ # D2 [84 B]
    ││└ # D2 [413 B]
    │└ # D2 [763 B]
    └ # M1 [813 B]
"""

__all__ = ['adapter', 'logger', 'trace']

import codecs
import contextlib
import locale
import logging
import math
import os
from functools import partial
from typing import TextIO, Union

import dill

# Tree drawing characters: Unicode to ASCII map.
ASCII_MAP = str.maketrans({"│": "|", "├": "|", "┬": "+", "└": "`"})

## Notes about the design choices ##

# Here is some domumentation of the Standard Library's logging internals that
# can't be found completely in the official documentation.  dill's logger is
# obtained by calling logging.getLogger('dill') and therefore is an instance of
# logging.getLoggerClass() at the call time.  As this is controlled by the user,
# in order to add some functionality to it it's necessary to use a LoggerAdapter
# to wrap it, overriding some of the adapter's methods and creating new ones.
#
# Basic calling sequence
# ======================
#
# Python's logging functionality can be conceptually divided into five steps:
#   0. Check logging level -> abort if call level is greater than logger level
#   1. Gather information -> construct a LogRecord from passed arguments and context
#   2. Filter (optional) -> discard message if the record matches a filter
#   3. Format -> format message with args, then format output string with message plus record
#   4. Handle -> write the formatted string to output as defined in the handler
#
# dill.logging.logger.log ->        # or logger.info, etc.
#   Logger.log ->               \
#     Logger._log ->             }- accept 'extra' parameter for custom record entries
#       Logger.makeRecord ->    /
#         LogRecord.__init__
#       Logger.handle ->
#         Logger.callHandlers ->
#           Handler.handle ->
#             Filterer.filter ->
#               Filter.filter
#             StreamHandler.emit ->
#               Handler.format ->
#                 Formatter.format ->
#                   LogRecord.getMessage        # does: record.message = msg % args
#                   Formatter.formatMessage ->
#                     PercentStyle.format       # does: self._fmt % vars(record)
#
# NOTE: All methods from the second line on are from logging.__init__.py

class TraceAdapter(logging.LoggerAdapter):
    """
    Tracks object tree depth and calculates pickled object size.

    A single instance of this wraps the module's logger, as the logging API
    doesn't allow setting it directly with a custom Logger subclass.  The added
    'trace()' method receives a pickle instance as the first argument and
    creates extra values to be added in the LogRecord from it, then calls
    'info()'.

    Usage of logger with 'trace()' method:

    >>> from dill.logger import adapter as logger  #NOTE: not dill.logger.logger
    >>> ...
    >>> def save_atype(pickler, obj):
    >>>     logger.trace(pickler, "Message with %s and %r etc. placeholders", 'text', obj)
    >>>     ...
    """
    def __init__(self, logger):
        self.logger = logger
    def addHandler(self, handler):
        formatter = TraceFormatter("%(prefix)s%(message)s%(suffix)s", handler=handler)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    def removeHandler(self, handler):
        self.logger.removeHandler(handler)
    def process(self, msg, kwargs):
        # A no-op override, as we don't have self.extra.
        return msg, kwargs
    def trace_setup(self, pickler):
        # Called by Pickler.dump().
        if not dill._dill.is_dill(pickler, child=False):
            return
        if self.isEnabledFor(logging.INFO):
            pickler._trace_depth = 1
            pickler._size_stack = []
        else:
            pickler._trace_depth = None
    def trace(self, pickler, msg, *args, **kwargs):
        if not hasattr(pickler, '_trace_depth'):
            logger.info(msg, *args, **kwargs)
            return
        if pickler._trace_depth is None:
            return
        extra = kwargs.get('extra', {})
        pushed_obj = msg.startswith('#')
        size = None
        try:
            # Streams are not required to be tellable.
            size = pickler._file.tell()
            frame = pickler.framer.current_frame
            try:
                size += frame.tell()
            except AttributeError:
                # PyPy may use a BytesBuilder as frame
                size += len(frame)
        except (AttributeError, TypeError):
            pass
        if size is not None:
            if not pushed_obj:
                pickler._size_stack.append(size)
            else:
                size -= pickler._size_stack.pop()
                extra['size'] = size
        if pushed_obj:
            pickler._trace_depth -= 1
        extra['depth'] = pickler._trace_depth
        kwargs['extra'] = extra
        self.info(msg, *args, **kwargs)
        if not pushed_obj:
            pickler._trace_depth += 1

class TraceFormatter(logging.Formatter):
    """
    Generates message prefix and suffix from record.

    This Formatter adds prefix and suffix strings to the log message in trace
    mode (an also provides empty string defaults for normal logs).
    """
    def __init__(self, *args, handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            encoding = handler.stream.encoding
            if encoding is None:
                raise AttributeError
        except AttributeError:
            encoding = locale.getpreferredencoding()
        try:
            encoding = codecs.lookup(encoding).name
        except LookupError:
            self.is_utf8 = False
        else:
            self.is_utf8 = (encoding == codecs.lookup('utf-8').name)
    def format(self, record):
        fields = {'prefix': "", 'suffix': ""}
        if getattr(record, 'depth', 0) > 0:
            if record.msg.startswith("#"):
                prefix = (record.depth - 1)*"│" + "└"
            elif record.depth == 1:
                prefix = "┬"
            else:
                prefix = (record.depth - 2)*"│" + "├┬"
            if not self.is_utf8:
                prefix = prefix.translate(ASCII_MAP) + "-"
            fields['prefix'] = prefix + " "
        if hasattr(record, 'size') and record.size is not None and record.size >= 1:
            # Show object size in human-readable form.
            power = int(math.log(record.size, 2)) // 10
            size = record.size >> power*10
            fields['suffix'] = " [%d %sB]" % (size, "KMGTP"[power] + "i" if power else "")
        vars(record).update(fields)
        return super().format(record)

logger = logging.getLogger('dill')
logger.propagate = False
adapter = TraceAdapter(logger)
stderr_handler = logging._StderrHandler()
adapter.addHandler(stderr_handler)

def trace(arg: Union[bool, TextIO, str, os.PathLike] = None, *, mode: str = 'a') -> None:
    """print a trace through the stack when pickling; useful for debugging

    With a single boolean argument, enable or disable the tracing.

    Example usage:

        >>> import dill
        >>> dill.detect.trace(True)
        >>> dill.dump_session()

    Alternatively, ``trace()`` can be used as a context manager. With no
    arguments, it just takes care of restoring the tracing state on exit.
    Either a file handle, or a file name and (optionally) a file mode may be
    specitfied to redirect the tracing output in the ``with`` block context. A
    log function is yielded by the manager so the user can write extra
    information to the file.

    Example usage:

        >>> from dill import detect
        >>> D = {'a': 42, 'b': {'x': None}}
        >>> with detect.trace():
        >>>     dumps(D)
        ┬ D2: <dict object at 0x7f2721804800>
        ├┬ D2: <dict object at 0x7f27217f5c40>
        │└ # D2 [8 B]
        └ # D2 [22 B]
        >>> squared = lambda x: x**2
        >>> with detect.trace('output.txt', mode='w') as log:
        >>>     log("> D = %r", D)
        >>>     dumps(D)
        >>>     log("> squared = %r", squared)
        >>>     dumps(squared)

    Arguments:
        arg: a boolean value, or an optional file-like or path-like object for the context manager
        mode: mode string for ``open()`` if a file name is passed as the first argument
    """
    if repr(arg) not in ('False', 'True'):
        return TraceManager(file=arg, mode=mode)
    logger.setLevel(logging.INFO if arg else logging.WARNING)

class TraceManager(contextlib.AbstractContextManager):
    """context manager version of trace(); can redirect the trace to a file"""
    def __init__(self, file, mode):
        self.file = file
        self.mode = mode
        self.redirect = file is not None
        self.file_is_stream = hasattr(file, 'write')
    def __enter__(self):
        if self.redirect:
            stderr_handler.flush()
            if self.file_is_stream:
                self.handler = logging.StreamHandler(self.file)
            else:
                self.handler = logging.FileHandler(self.file, self.mode)
            adapter.removeHandler(stderr_handler)
            adapter.addHandler(self.handler)
        self.old_level = adapter.getEffectiveLevel()
        adapter.setLevel(logging.INFO)
        return adapter.info
    def __exit__(self, *exc_info):
        adapter.setLevel(self.old_level)
        if self.redirect:
            adapter.removeHandler(self.handler)
            adapter.addHandler(stderr_handler)
            if not self.file_is_stream:
                self.handler.close()
<|endoftext|># -*- coding: utf-8 -*-
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2015 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
dill: a utility for serialization of python objects

The primary functions in `dill` are :func:`dump` and
:func:`dumps` for serialization ("pickling") to a
file or to a string, respectively, and :func:`load`
and :func:`loads` for deserialization ("unpickling"),
similarly, from a file or from a string. Other notable
functions are :func:`~dill.dump_module` and
:func:`~dill.load_module`, which are used to save and
restore module objects, including an intepreter session.

Based on code written by Oren Tirosh and Armin Ronacher.
Extended to a (near) full set of the builtin types (in types module),
and coded to the pickle interface, by <mmckerns@caltech.edu>.
Initial port to python3 by Jonathan Dobson, continued by mmckerns.
Tested against "all" python types (Std. Lib. CH 1-15 @ 2.7) by mmckerns.
Tested against CH16+ Std. Lib. ... TBD.
"""

from __future__ import annotations

__all__ = [
    'dump','dumps','load','loads','copy',
    'Pickler','Unpickler','register','pickle','pickles','check',
    'DEFAULT_PROTOCOL','HIGHEST_PROTOCOL','HANDLE_FMODE','CONTENTS_FMODE','FILE_FMODE',
    'PickleError','PickleWarning','PicklingError','PicklingWarning','UnpicklingError',
    'UnpicklingWarning',
]

__module__ = 'dill'

import warnings
from .logger import adapter as logger
from .logger import trace as _trace
log = logger # backward compatibility (see issue #582)

import os
import sys
diff = None
_use_diff = False
OLD38 = (sys.hexversion < 0x3080000)
OLD39 = (sys.hexversion < 0x3090000)
OLD310 = (sys.hexversion < 0x30a0000)
OLD312a7 = (sys.hexversion < 0x30c00a7)
#XXX: get types from .objtypes ?
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _contextvars import Context as ContextType
from _thread import LockType
from _thread import RLock as RLockType
try:
    from _thread import _ExceptHookArgs as ExceptHookArgsType
except ImportError:
    ExceptHookArgsType = None
try:
    from _thread import _ThreadHandle as ThreadHandleType
except ImportError:
    ThreadHandleType = None
#from io import IOBase
from types import CodeType, FunctionType, MethodType, GeneratorType, \
    TracebackType, FrameType, ModuleType, BuiltinMethodType
BufferType = memoryview #XXX: unregistered
ClassType = type # no 'old-style' classes
EllipsisType = type(Ellipsis)
#FileType = IOBase
NotImplementedType = type(NotImplemented)
SliceType = slice
TypeType = type # 'new-style' classes #XXX: unregistered
XRangeType = range
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
# import zlib
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
GENERATOR_FAIL = False
import importlib.machinery
EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
try:
    import ctypes
    HAS_CTYPES = True
    # if using `pypy`, pythonapi is not found
    IS_PYPY = not hasattr(ctypes, 'pythonapi')
except ImportError:
    HAS_CTYPES = False
    IS_PYPY = False
NumpyUfuncType = None
NumpyDType = None
NumpyArrayType = None
try:
    if not importlib.machinery.PathFinder().find_spec('numpy'):
        raise ImportError("No module named 'numpy'")
    NumpyUfuncType = True
    NumpyDType = True
    NumpyArrayType = True
except ImportError:
    pass
def __hook__():
    global NumpyArrayType, NumpyDType, NumpyUfuncType
    from numpy import ufunc as NumpyUfuncType
    from numpy import ndarray as NumpyArrayType
    from numpy import dtype as NumpyDType
    return True
if NumpyArrayType: # then has numpy
    def ndarraysubclassinstance(obj_type):
        if all((c.__module__, c.__name__) != ('numpy', 'ndarray') for c in obj_type.__mro__):
            return False
        # anything below here is a numpy array (or subclass) instance
        __hook__() # import numpy (so the following works!!!)
        # verify that __reduce__ has not been overridden
        if obj_type.__reduce_ex__ is not NumpyArrayType.__reduce_ex__ \
                or obj_type.__reduce__ is not NumpyArrayType.__reduce__:
            return False
        return True
    def numpyufunc(obj_type):
        return any((c.__module__, c.__name__) == ('numpy', 'ufunc') for c in obj_type.__mro__)
    def numpydtype(obj_type):
        if all((c.__module__, c.__name__) != ('numpy', 'dtype') for c in obj_type.__mro__):
            return False
        # anything below here is a numpy dtype
        __hook__() # import numpy (so the following works!!!)
        return obj_type is type(NumpyDType) # handles subclasses
else:
    def ndarraysubclassinstance(obj): return False
    def numpyufunc(obj): return False
    def numpydtype(obj): return False

from types import GetSetDescriptorType, ClassMethodDescriptorType, \
     WrapperDescriptorType,  MethodDescriptorType, MemberDescriptorType, \
     MethodWrapperType #XXX: unused

# make sure to add these 'hand-built' types to _typemap
CellType = type((lambda x: lambda y: x)(0).__closure__[0])
PartialType = type(partial(int, base=2))
SuperType = type(super(Exception, TypeError()))
ItemGetterType = type(itemgetter(0))
AttrGetterType = type(attrgetter('__repr__'))

try:
    from functools import _lru_cache_wrapper as LRUCacheType
except ImportError:
    LRUCacheType = None

if not isinstance(LRUCacheType, type):
    LRUCacheType = None

def get_file_type(*args, **kwargs):
    open = kwargs.pop("open", __builtin__.open)
    f = open(os.devnull, *args, **kwargs)
    t = type(f)
    f.close()
    return t

IS_PYODIDE = sys.platform == 'emscripten'

FileType = get_file_type('rb', buffering=0)
TextWrapperType = get_file_type('r', buffering=-1)
BufferedRandomType = None if IS_PYODIDE else get_file_type('r+b', buffering=-1)
BufferedReaderType = get_file_type('rb', buffering=-1)
BufferedWriterType = get_file_type('wb', buffering=-1)
try:
    from _pyio import open as _open
    PyTextWrapperType = get_file_type('r', buffering=-1, open=_open)
    PyBufferedRandomType = None if IS_PYODIDE else get_file_type('r+b', buffering=-1, open=_open)
    PyBufferedReaderType = get_file_type('rb', buffering=-1, open=_open)
    PyBufferedWriterType = get_file_type('wb', buffering=-1, open=_open)
except ImportError:
    PyTextWrapperType = PyBufferedRandomType = PyBufferedReaderType = PyBufferedWriterType = None
from io import BytesIO as StringIO
InputType = OutputType = None
from socket import socket as SocketType
#FIXME: additionally calls ForkingPickler.register several times
from multiprocessing.reduction import _reduce_socket as reduce_socket
try: #pragma: no cover
    IS_IPYTHON = __IPYTHON__  # is True
    ExitType = None # IPython.core.autocall.ExitAutocall
    IPYTHON_SINGLETONS = ('exit', 'quit', 'get_ipython')
except NameError:
    IS_IPYTHON = False
    try: ExitType = type(exit) # apparently 'exit' can be removed
    except NameError: ExitType = None
    IPYTHON_SINGLETONS = ()

import inspect
import typing


### Shims for different versions of Python and dill
class Sentinel(object):
    """
    Create a unique sentinel object that is pickled as a constant.
    """
    def __init__(self, name, module_name=None):
        self.name = name
        if module_name is None:
            # Use the calling frame's module
            self.__module__ = inspect.currentframe().f_back.f_globals['__name__']
        else:
            self.__module__ = module_name # pragma: no cover
    def __repr__(self):
        return self.__module__ + '.' + self.name # pragma: no cover
    def __copy__(self):
        return self # pragma: no cover
    def __deepcopy__(self, memo):
        return self # pragma: no cover
    def __reduce__(self):
        return self.name
    def __reduce_ex__(self, protocol):
        return self.name

from . import _shims
from ._shims import Reduce, Getattr

### File modes
#: Pickles the file handle, preserving mode. The position of the unpickled
#: object is as for a new file handle.
HANDLE_FMODE = 0
#: Pickles the file contents, creating a new file if on load the file does
#: not exist. The position = min(pickled position, EOF) and mode is chosen
#: as such that "best" preserves behavior of the original file.
CONTENTS_FMODE = 1
#: Pickles the entire file (handle and contents), preserving mode and position.
FILE_FMODE = 2

### Shorthands (modified from python2.5/lib/pickle.py)
def copy(obj, *args, **kwds):
    """
    Use pickling to 'copy' an object (i.e. `loads(dumps(obj))`).

    See :func:`dumps` and :func:`loads` for keyword arguments.
    """
    ignore = kwds.pop('ignore', Unpickler.settings['ignore'])
    return loads(dumps(obj, *args, **kwds), ignore=ignore)

def dump(obj, file, protocol=None, byref=None, fmode=None, recurse=None, **kwds):#, strictio=None):
    """
    Pickle an object to a file.

    See :func:`dumps` for keyword arguments.
    """
    from .settings import settings
    protocol = settings['protocol'] if protocol is None else int(protocol)
    _kwds = kwds.copy()
    _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
    Pickler(file, protocol, **_kwds).dump(obj)
    return

def dumps(obj, protocol=None, byref=None, fmode=None, recurse=None, **kwds):#, strictio=None):
    """
    Pickle an object to a string.

    *protocol* is the pickler protocol, as defined for Python *pickle*.

    If *byref=True*, then dill behaves a lot more like pickle as certain
    objects (like modules) are pickled by reference as opposed to attempting
    to pickle the object itself.

    If *recurse=True*, then objects referred to in the global dictionary
    are recursively traced and pickled, instead of the default behavior
    of attempting to store the entire global dictionary. This is needed for
    functions defined via *exec()*.

    *fmode* (:const:`HANDLE_FMODE`, :const:`CONTENTS_FMODE`,
    or :const:`FILE_FMODE`) indicates how file handles will be pickled.
    For example, when pickling a data file handle for transfer to a remote
    compute service, *FILE_FMODE* will include the file contents in the
    pickle and cursor position so that a remote method can operate
    transparently on an object with an open file handle.

    Default values for keyword arguments can be set in :mod:`dill.settings`.
    """
    file = StringIO()
    dump(obj, file, protocol, byref, fmode, recurse, **kwds)#, strictio)
    return file.getvalue()

def load(file, ignore=None, **kwds):
    """
    Unpickle an object from a file.

    See :func:`loads` for keyword arguments.
    """
    return Unpickler(file, ignore=ignore, **kwds).load()

def loads(str, ignore=None, **kwds):
    """
    Unpickle an object from a string.

    If *ignore=False* then objects whose class is defined in the module
    *__main__* are updated to reference the existing class in *__main__*,
    otherwise they are left to refer to the reconstructed type, which may
    be different.

    Default values for keyword arguments can be set in :mod:`dill.settings`.
    """
    file = StringIO(str)
    return load(file, ignore, **kwds)

# def dumpzs(obj, protocol=None):
#     """pickle an object to a compressed string"""
#     return zlib.compress(dumps(obj, protocol))

# def loadzs(str):
#     """unpickle an object from a compressed string"""
#     return loads(zlib.decompress(str))

### End: Shorthands ###

class MetaCatchingDict(dict):
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __missing__(self, key):
        if issubclass(key, type):
            return save_type
        else:
            raise KeyError()

class PickleWarning(Warning, PickleError):
    pass

class PicklingWarning(PickleWarning, PicklingError):
    pass

class UnpicklingWarning(PickleWarning, UnpicklingError):
    pass

### Extend the Picklers
class Pickler(StockPickler):
    """python's Pickler extended to interpreter sessions"""
    dispatch: typing.Dict[type, typing.Callable[[Pickler, typing.Any], None]] \
            = MetaCatchingDict(StockPickler.dispatch.copy())
    """The dispatch table, a dictionary of serializing functions used
    by Pickler to save objects of specific types.  Use :func:`pickle`
    or :func:`register` to associate types to custom functions.

    :meta hide-value:
    """
    _session = False
    from .settings import settings

    def __init__(self, file, *args, **kwds):
        settings = Pickler.settings
        _byref = kwds.pop('byref', None)
       #_strictio = kwds.pop('strictio', None)
        _fmode = kwds.pop('fmode', None)
        _recurse = kwds.pop('recurse', None)
        StockPickler.__init__(self, file, *args, **kwds)
        self._main = _main_module
        self._diff_cache = {}
        self._byref = settings['byref'] if _byref is None else _byref
        self._strictio = False #_strictio
        self._fmode = settings['fmode'] if _fmode is None else _fmode
        self._recurse = settings['recurse'] if _recurse is None else _recurse
        self._postproc = OrderedDict()
        self._file = file

    def save(self, obj, save_persistent_id=True):
        # numpy hack
        obj_type = type(obj)
        if NumpyArrayType and not (obj_type is type or obj_type in Pickler.dispatch):
            # register if the object is a numpy ufunc
            # thanks to Paul Kienzle for pointing out ufuncs didn't pickle
            if numpyufunc(obj_type):
                @register(obj_type)
                def save_numpy_ufunc(pickler, obj):
                    logger.trace(pickler, "Nu: %s", obj)
                    name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
                    StockPickler.save_global(pickler, obj, name=name)
                    logger.trace(pickler, "# Nu")
                    return
                # NOTE: the above 'save' performs like:
                #   import copy_reg
                #   def udump(f): return f.__name__
                #   def uload(name): return getattr(numpy, name)
                #   copy_reg.pickle(NumpyUfuncType, udump, uload)
            # register if the object is a numpy dtype
            if numpydtype(obj_type):
                @register(obj_type)
                def save_numpy_dtype(pickler, obj):
                    logger.trace(pickler, "Dt: %s", obj)
                    pickler.save_reduce(_create_dtypemeta, (obj.type,), obj=obj)
                    logger.trace(pickler, "# Dt")
                    return
                # NOTE: the above 'save' performs like:
                #   import copy_reg
                #   def uload(name): return type(NumpyDType(name))
                #   def udump(f): return uload, (f.type,)
                #   copy_reg.pickle(NumpyDTypeType, udump, uload)
            # register if the object is a subclassed numpy array instance
            if ndarraysubclassinstance(obj_type):
                @register(obj_type)
                def save_numpy_array(pickler, obj):
                    logger.trace(pickler, "Nu: (%s, %s)", obj.shape, obj.dtype)
                    npdict = getattr(obj, '__dict__', None)
                    f, args, state = obj.__reduce__()
                    pickler.save_reduce(_create_array, (f,args,state,npdict), obj=obj)
                    logger.trace(pickler, "# Nu")
                    return
        # end numpy hack

        if GENERATOR_FAIL and obj_type is GeneratorType:
            msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
            raise PicklingError(msg)
        StockPickler.save(self, obj, save_persistent_id)

    save.__doc__ = StockPickler.save.__doc__

    def dump(self, obj): #NOTE: if settings change, need to update attributes
        logger.trace_setup(self)
        StockPickler.dump(self, obj)
    dump.__doc__ = StockPickler.dump.__doc__

class Unpickler(StockUnpickler):
    """python's Unpickler extended to interpreter sessions and more types"""
    from .settings import settings
    _session = False

    def find_class(self, module, name):
        if (module, name) == ('__builtin__', '__main__'):
            return self._main.__dict__ #XXX: above set w/save_module_dict
        elif (module, name) == ('__builtin__', 'NoneType'):
            return type(None) #XXX: special case: NoneType missing
        if module == 'dill.dill': module = 'dill._dill'
        return StockUnpickler.find_class(self, module, name)

    def __init__(self, *args, **kwds):
        settings = Pickler.settings
        _ignore = kwds.pop('ignore', None)
        StockUnpickler.__init__(self, *args, **kwds)
        self._main = _main_module
        self._ignore = settings['ignore'] if _ignore is None else _ignore

    def load(self): #NOTE: if settings change, need to update attributes
        obj = StockUnpickler.load(self)
        if type(obj).__module__ == getattr(_main_module, '__name__', '__main__'):
            if not self._ignore:
                # point obj class to main
                try: obj.__class__ = getattr(self._main, type(obj).__name__)
                except (AttributeError,TypeError): pass # defined in a file
       #_main_module.__dict__.update(obj.__dict__) #XXX: should update globals ?
        return obj
    load.__doc__ = StockUnpickler.load.__doc__
    pass

'''
def dispatch_table():
    """get the dispatch table of registered types"""
    return Pickler.dispatch
'''

pickle_dispatch_copy = StockPickler.dispatch.copy()

def pickle(t, func):
    """expose :attr:`~Pickler.dispatch` table for user-created extensions"""
    Pickler.dispatch[t] = func
    return

def register(t):
    """decorator to register types to Pickler's :attr:`~Pickler.dispatch` table"""
    def proxy(func):
        Pickler.dispatch[t] = func
        return func
    return proxy

def _revert_extension():
    """drop dill-registered types from pickle's dispatch table"""
    for type, func in list(StockPickler.dispatch.items()):
        if func.__module__ == __name__:
            del StockPickler.dispatch[type]
            if type in pickle_dispatch_copy:
                StockPickler.dispatch[type] = pickle_dispatch_copy[type]

def use_diff(on=True):
    """
    Reduces size of pickles by only including object which have changed.

    Decreases pickle size but increases CPU time needed.
    Also helps avoid some unpickleable objects.
    MUST be called at start of script, otherwise changes will not be recorded.
    """
    global _use_diff, diff
    _use_diff = on
    if _use_diff and diff is None:
        try:
            from . import diff as d
        except ImportError:
            import diff as d
        diff = d

def _create_typemap():
    import types
    d = dict(list(__builtin__.__dict__.items()) + \
             list(types.__dict__.items())).items()
    for key, value in d:
        if getattr(value, '__module__', None) == 'builtins' \
                and type(value) is type:
            yield key, value
    return
_reverse_typemap = dict(_create_typemap())
_reverse_typemap.update({
    'PartialType': PartialType,
    'SuperType': SuperType,
    'ItemGetterType': ItemGetterType,
    'AttrGetterType': AttrGetterType,
})
if sys.hexversion < 0x30800a2:
    _reverse_typemap.update({
        'CellType': CellType,
    })

# "Incidental" implementation specific types. Unpickling these types in another
# implementation of Python (PyPy -> CPython) is not guaranteed to work

# This dictionary should contain all types that appear in Python implementations
# but are not defined in https://docs.python.org/3/library/types.html#standard-interpreter-types
x=OrderedDict()
_incedental_reverse_typemap = {
    'FileType': FileType,
    'BufferedRandomType': BufferedRandomType,
    'BufferedReaderType': BufferedReaderType,
    'BufferedWriterType': BufferedWriterType,
    'TextWrapperType': TextWrapperType,
    'PyBufferedRandomType': PyBufferedRandomType,
    'PyBufferedReaderType': PyBufferedReaderType,
    'PyBufferedWriterType': PyBufferedWriterType,
    'PyTextWrapperType': PyTextWrapperType,
}

_incedental_reverse_typemap.update({
    "DictKeysType": type({}.keys()),
    "DictValuesType": type({}.values()),
    "DictItemsType": type({}.items()),

    "OdictKeysType": type(x.keys()),
    "OdictValuesType": type(x.values()),
    "OdictItemsType": type(x.items()),
})

if ExitType:
    _incedental_reverse_typemap['ExitType'] = ExitType
if InputType:
    _incedental_reverse_typemap['InputType'] = InputType
    _incedental_reverse_typemap['OutputType'] = OutputType

'''
try:
    import symtable
    _incedental_reverse_typemap["SymtableEntryType"] = type(symtable.symtable("", "string", "exec")._table)
except: #FIXME: fails to pickle
    pass

if sys.hexversion >= 0x30a00a0:
    _incedental_reverse_typemap['LineIteratorType'] = type(compile('3', '', 'eval').co_lines())
'''

if sys.hexversion >= 0x30b00b0 and not IS_PYPY:
    from types import GenericAlias
    _incedental_reverse_typemap["GenericAliasIteratorType"] = type(iter(GenericAlias(list, (int,))))
    '''
    _incedental_reverse_typemap['PositionsIteratorType'] = type(compile('3', '', 'eval').co_positions())
    '''

try:
    import winreg
    _incedental_reverse_typemap["HKEYType"] = winreg.HKEYType
except ImportError:
    pass

_reverse_typemap.update(_incedental_reverse_typemap)
_incedental_types = set(_incedental_reverse_typemap.values())

del x

_typemap = dict((v, k) for k, v in _reverse_typemap.items())

def _unmarshal(string):
    return marshal.loads(string)

def _load_type(name):
    return _reverse_typemap[name]

def _create_type(typeobj, *args):
    return typeobj(*args)

def _create_function(fcode, fglobals, fname=None, fdefaults=None,
                     fclosure=None, fdict=None, fkwdefaults=None):
    # same as FunctionType, but enable passing __dict__ to new function,
    # __dict__ is the storehouse for attributes added after function creation
    func = FunctionType(fcode, fglobals or dict(), fname, fdefaults, fclosure)
    if fdict is not None:
        func.__dict__.update(fdict) #XXX: better copy? option to copy?
    if fkwdefaults is not None:
        func.__kwdefaults__ = fkwdefaults
    # 'recurse' only stores referenced modules/objects in fglobals,
    # thus we need to make sure that we have __builtins__ as well
    if "__builtins__" not in func.__globals__:
        func.__globals__["__builtins__"] = globals()["__builtins__"]
    # assert id(fglobals) == id(func.__globals__)
    return func

class match:
    """
    Make avaialable a limited structural pattern matching-like syntax for Python < 3.10

    Patterns can be only tuples (without types) currently.
    Inspired by the package pattern-matching-PEP634.

    Usage:
    >>> with match(args) as m:
    >>>     if   m.case(('x', 'y')):
    >>>         # use m.x and m.y
    >>>     elif m.case(('x', 'y', 'z')):
    >>>         # use m.x, m.y and m.z

    Equivalent native code for Python >= 3.10:
    >>> match args:
    >>>     case (x, y):
    >>>         # use x and y
    >>>     case (x, y, z):
    >>>         # use x, y and z
    """
    def __init__(self, value):
        self.value = value
        self._fields = None
    def __enter__(self):
        return self
    def __exit__(self, *exc_info):
        return False
    def case(self, args): # *args, **kwargs):
        """just handles tuple patterns"""
        if len(self.value) != len(args): # + len(kwargs):
            return False
        #if not all(isinstance(arg, pat) for arg, pat in zip(self.value[len(args):], kwargs.values())):
        #    return False
        self.args = args # (*args, *kwargs)
        return True
    @property
    def fields(self):
        # Only bind names to values if necessary.
        if self._fields is None:
            self._fields = dict(zip(self.args, self.value))
        return self._fields
    def __getattr__(self, item):
        return self.fields[item]

ALL_CODE_PARAMS = [
    # Version     New attribute         CodeType parameters
    ((3,11,'a'), 'co_endlinetable',    'argcount posonlyargcount kwonlyargcount nlocals stacksize flags code consts names varnames filename name qualname firstlineno linetable endlinetable columntable exceptiontable freevars cellvars'),
    ((3,11),     'co_exceptiontable',  'argcount posonlyargcount kwonlyargcount nlocals stacksize flags code consts names varnames filename name qualname firstlineno linetable                          exceptiontable freevars cellvars'),
    ((3,11,'p'), 'co_qualname',        'argcount posonlyargcount kwonlyargcount nlocals stacksize flags code consts names varnames filename name qualname firstlineno linetable                                         freevars cellvars'),
    ((3,10),     'co_linetable',       'argcount posonlyargcount kwonlyargcount nlocals stacksize flags code consts names varnames filename name          firstlineno linetable                                         freevars cellvars'),
    ((3,8),      'co_posonlyargcount', 'argcount posonlyargcount kwonlyargcount nlocals stacksize flags code consts names varnames filename name          firstlineno lnotab                                            freevars cellvars'),
    ((3,7),      'co_kwonlyargcount',  'argcount                 kwonlyargcount nlocals stacksize flags code consts names varnames filename name          firstlineno lnotab                                            freevars cellvars'),
    ]
for version, new_attr, params in ALL_CODE_PARAMS:
    if hasattr(CodeType, new_attr):
        CODE_VERSION = version
        CODE_PARAMS = params.split()
        break
ENCODE_PARAMS = set(CODE_PARAMS).intersection(
        ['code', 'lnotab', 'linetable', 'endlinetable', 'columntable', 'exceptiontable'])

def _create_code(*args):
    if not isinstance(args[0], int): # co_lnotab stored from >= 3.10
        LNOTAB, *args = args
    else: # from < 3.10 (or pre-LNOTAB storage)
        LNOTAB = b''

    with match(args) as m:
        # Python 3.11/3.12a (18 members)
        if m.case((
            'argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags',     # args[0:6]
            'code', 'consts', 'names', 'varnames', 'filename', 'name', 'qualname', 'firstlineno', # args[6:14]
            'linetable', 'exceptiontable', 'freevars', 'cellvars'                                 # args[14:]
        )):
            if CODE_VERSION == (3,11):
                return CodeType(
                    *args[:6],
                    args[6].encode() if hasattr(args[6], 'encode') else args[6], # code
                    *args[7:14],
                    args[14].encode() if hasattr(args[14], 'encode') else args[14], # linetable
                    args[15].encode() if hasattr(args[15], 'encode') else args[15], # exceptiontable
                    args[16],
                    args[17],
                )
            fields = m.fields
        # PyPy 3.11 7.3.19+ (17 members)
        elif m.case((
            'argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', # args[0:6]
            'code', 'consts', 'names', 'varnames', 'filename', 'name', 'qualname',            # args[6:13]
            'firstlineno', 'linetable', 'freevars', 'cellvars'                                # args[13:]
        )):
            if CODE_VERSION == (3,11,'p'):
                return CodeType(
                    *args[:6],
                    args[6].encode() if hasattr(args[6], 'encode') else args[6], # code
                    *args[7:14],
                    args[14].encode() if hasattr(args[14], 'encode') else args[14], # linetable
                    args[15],
                    args[16],
                )
            fields = m.fields
        # Python 3.10 or 3.8/3.9 (16 members)
        elif m.case((
            'argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', # args[0:6]
            'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno',         # args[6:13]
            'LNOTAB_OR_LINETABLE', 'freevars', 'cellvars'                                     # args[13:]
        )):
            if CODE_VERSION == (3,10) or CODE_VERSION == (3,8):
                return CodeType(
                    *args[:6],
                    args[6].encode() if hasattr(args[6], 'encode') else args[6], # code
                    *args[7:13],
                    args[13].encode() if hasattr(args[13], 'encode') else args[13], # lnotab/linetable
                    args[14],
                    args[15],
                )
            fields = m.fields
            if CODE_VERSION >= (3,10):
                fields['linetable'] = m.LNOTAB_OR_LINETABLE
            else:
                fields['lnotab'] = LNOTAB if LNOTAB else m.LNOTAB_OR_LINETABLE
        # Python 3.7 (15 args)
        elif m.case((
            'argcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags',            # args[0:5]
            'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', # args[5:12]
            'lnotab', 'freevars', 'cellvars'                                          # args[12:]
        )):
            if CODE_VERSION == (3,7):
                return CodeType(
                    *args[:5],
                    args[5].encode() if hasattr(args[5], 'encode') else args[5], # code
                    *args[6:12],
                    args[12].encode() if hasattr(args[12], 'encode') else args[12], # lnotab
                    args[13],
                    args[14],
                )
            fields = m.fields
        # Python 3.11a (20 members)
        elif m.case((
            'argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags',     # args[0:6]
            'code', 'consts', 'names', 'varnames', 'filename', 'name', 'qualname', 'firstlineno', # args[6:14]
            'linetable', 'endlinetable', 'columntable', 'exceptiontable', 'freevars', 'cellvars'  # args[14:]
        )):
            if CODE_VERSION == (3,11,'a'):
                return CodeType(
                    *args[:6],
                    args[6].encode() if hasattr(args[6], 'encode') else args[6], # code
                    *args[7:14],
                    *(a.encode() if hasattr(a, 'encode') else a for a in args[14:18]), # linetable-exceptiontable
                    args[18],
                    args[19],
                )
            fields = m.fields
        else:
            raise UnpicklingError("pattern match for code object failed")

    # The args format doesn't match this version.
    fields.setdefault('posonlyargcount', 0)         # from python <= 3.7
    fields.setdefault('lnotab', LNOTAB)             # from python >= 3.10
    fields.setdefault('linetable', b'')             # from python <= 3.9
    fields.setdefault('qualname', fields['name'])   # from python <= 3.10
    fields.setdefault('exceptiontable', b'')        # from python <= 3.10
    fields.setdefault('endlinetable', None)         # from python != 3.11a
    fields.setdefault('columntable', None)          # from python != 3.11a

    args = (fields[k].encode() if k in ENCODE_PARAMS and hasattr(fields[k], 'encode') else fields[k]
            for k in CODE_PARAMS)
    return CodeType(*args)

def _create_ftype(ftypeobj, func, args, kwds):
    if kwds is None:
        kwds = {}
    if args is None:
        args = ()
    return ftypeobj(func, *args, **kwds)

def _create_typing_tuple(argz, *args): #NOTE: workaround python/cpython#94245
    if not argz:
        return typing.Tuple[()].copy_with(())
    if argz == ((),):
        return typing.Tuple[()]
    return typing.Tuple[argz]

if ThreadHandleType:
    def _create_thread_handle(ident, done, *args): #XXX: ignores 'blocking'
        from threading import _make_thread_handle
        handle = _make_thread_handle(ident)
        if done:
            handle._set_done()
        return handle

def _create_lock(locked, *args): #XXX: ignores 'blocking'
    from threading import Lock
    lock = Lock()
    if locked:
        if not lock.acquire(False):
            raise UnpicklingError("Cannot acquire lock")
    return lock

def _create_rlock(count, owner, *args): #XXX: ignores 'blocking'
    lock = RLockType()
    if owner is not None:
        lock._acquire_restore((count, owner))
    if owner and not lock._is_owned():
        raise UnpicklingError("Cannot acquire lock")
    return lock

# thanks to matsjoyce for adding all the different file modes
def _create_filehandle(name, mode, position, closed, open, strictio, fmode, fdata): # buffering=0
    # only pickles the handle, not the file contents... good? or StringIO(data)?
    # (for file contents see: http://effbot.org/librarybook/copy-reg.htm)
    # NOTE: handle special cases first (are there more special cases?)
    names = {'<stdin>':sys.__stdin__, '<stdout>':sys.__stdout__,
             '<stderr>':sys.__stderr__} #XXX: better fileno=(0,1,2) ?
    if name in list(names.keys()):
        f = names[name] #XXX: safer "f=sys.stdin"
    elif name == '<tmpfile>':
        f = os.tmpfile()
    elif name == '<fdopen>':
        import tempfile
        f = tempfile.TemporaryFile(mode)
    else:
        try:
            exists = os.path.exists(name)
        except Exception:
            exists = False
        if not exists:
            if strictio:
                raise FileNotFoundError("[Errno 2] No such file or directory: '%s'" % name)
            elif "r" in mode and fmode != FILE_FMODE:
                name = '<fdopen>' # or os.devnull?
            current_size = 0 # or maintain position?
        else:
            current_size = os.path.getsize(name)

        if position > current_size:
            if strictio:
                raise ValueError("invalid buffer size")
            elif fmode == CONTENTS_FMODE:
                position = current_size
        # try to open the file by name
        # NOTE: has different fileno
        try:
            #FIXME: missing: *buffering*, encoding, softspace
            if fmode == FILE_FMODE:
                f = open(name, mode if "w" in mode else "w")
                f.write(fdata)
                if "w" not in mode:
                    f.close()
                    f = open(name, mode)
            elif name == '<fdopen>': # file did not exist
                import tempfile
                f = tempfile.TemporaryFile(mode)
            # treat x mode as w mode
            elif fmode == CONTENTS_FMODE \
               and ("w" in mode or "x" in mode):
                # stop truncation when opening
                flags = os.O_CREAT
                if "+" in mode:
                    flags |= os.O_RDWR
                else:
                    flags |= os.O_WRONLY
                f = os.fdopen(os.open(name, flags), mode)
                # set name to the correct value
                r = getattr(f, "buffer", f)
                r = getattr(r, "raw", r)
                r.name = name
                assert f.name == name
            else:
                f = open(name, mode)
        except (IOError, FileNotFoundError):
            err = sys.exc_info()[1]
            raise UnpicklingError(err)
    if closed:
        f.close()
    elif position >= 0 and fmode != HANDLE_FMODE:
        f.seek(position)
    return f

def _create_stringi(value, position, closed):
    f = StringIO(value)
    if closed: f.close()
    else: f.seek(position)
    return f

def _create_stringo(value, position, closed):
    f = StringIO()
    if closed: f.close()
    else:
       f.write(value)
       f.seek(position)
    return f

class _itemgetter_helper(object):
    def __init__(self):
        self.items = []
    def __getitem__(self, item):
        self.items.append(item)
        return

class _attrgetter_helper(object):
    def __init__(self, attrs, index=None):
        self.attrs = attrs
        self.index = index
    def __getattribute__(self, attr):
        attrs = object.__getattribute__(self, "attrs")
        index = object.__getattribute__(self, "index")
        if index is None:
            index = len(attrs)
            attrs.append(attr)
        else:
            attrs[index] = ".".join([attrs[index], attr])
        return type(self)(attrs, index)

class _dictproxy_helper(dict):
   def __ror__(self, a):
        return a

_dictproxy_helper_instance = _dictproxy_helper()

__d = {}
try:
    # In CPython 3.9 and later, this trick can be used to exploit the
    # implementation of the __or__ function of MappingProxyType to get the true
    # mapping referenced by the proxy. It may work for other implementations,
    # but is not guaranteed.
    MAPPING_PROXY_TRICK = __d is (DictProxyType(__d) | _dictproxy_helper_instance)
except Exception:
    MAPPING_PROXY_TRICK = False
del __d

# _CELL_REF and _CELL_EMPTY are used to stay compatible with versions of dill
# whose _create_cell functions do not have a default value.
# _CELL_REF can be safely removed entirely (replaced by empty tuples for calls
# to _create_cell) once breaking changes are allowed.
_CELL_REF = None
_CELL_EMPTY = Sentinel('_CELL_EMPTY')

def _create_cell(contents=None):
    if contents is not _CELL_EMPTY:
        value = contents
    return (lambda: value).__closure__[0]

def _create_weakref(obj, *args):
    from weakref import ref
    if obj is None: # it's dead
        from collections import UserDict
        return ref(UserDict(), *args)
    return ref(obj, *args)

def _create_weakproxy(obj, callable=False, *args):
    from weakref import proxy
    if obj is None: # it's dead
        if callable: return proxy(lambda x:x, *args)
        from collections import UserDict
        return proxy(UserDict(), *args)
    return proxy(obj, *args)

def _eval_repr(repr_str):
    return eval(repr_str)

def _create_array(f, args, state, npdict=None):
   #array = numpy.core.multiarray._reconstruct(*args)
    array = f(*args)
    array.__setstate__(state)
    if npdict is not None: # we also have saved state in __dict__
        array.__dict__.update(npdict)
    return array

def _create_dtypemeta(scalar_type):
    if NumpyDType is True: __hook__() # a bit hacky I think
    if scalar_type is None:
        return NumpyDType
    return type(NumpyDType(scalar_type))

def _create_namedtuple(name, fieldnames, modulename, defaults=None):
    class_ = _import_module(modulename + '.' + name, safe=True)
    if class_ is not None:
        return class_
    import collections
    t = collections.namedtuple(name, fieldnames, defaults=defaults, module=modulename)
    return t

def _create_capsule(pointer, name, context, destructor):
    attr_found = False
    try:
        # based on https://github.com/python/cpython/blob/f4095e53ab708d95e019c909d5928502775ba68f/Objects/capsule.c#L209-L231
        uname = name.decode('utf8')
        for i in range(1, uname.count('.')+1):
            names = uname.rsplit('.', i)
            try:
                module = __import__(names[0])
            except ImportError:
                pass
            obj = module
            for attr in names[1:]:
                obj = getattr(obj, attr)
            capsule = obj
            attr_found = True
            break
    except Exception:
        pass

    if attr_found:
        if _PyCapsule_IsValid(capsule, name):
            return capsule
        raise UnpicklingError("%s object exists at %s but a PyCapsule object was expected." % (type(capsule), name))
    else:
        #warnings.warn('Creating a new PyCapsule %s for a C data structure that may not be present in memory. Segmentation faults or other memory errors are possible.' % (name,), UnpicklingWarning)
        capsule = _PyCapsule_New(pointer, name, destructor)
        _PyCapsule_SetContext(capsule, context)
        return capsule

def _getattr(objclass, name, repr_str):
    # hack to grab the reference directly
    try: #XXX: works only for __builtin__ ?
        attr = repr_str.split("'")[3]
        return eval(attr+'.__dict__["'+name+'"]')
    except Exception:
        try:
            attr = objclass.__dict__
            if type(attr) is DictProxyType:
                if sys.hexversion > 0x30f00a0 and name in ('__weakref__','__dict__'):
                    attr = _dictproxy_helper.__dict__[name]
                else:
                    attr = attr[name]
            else:
                attr = getattr(objclass,name)
        except (AttributeError, KeyError):
            attr = getattr(objclass,name)
        return attr

def _get_attr(self, name):
    # stop recursive pickling
    return getattr(self, name, None) or getattr(__builtin__, name)

def _import_module(import_name, safe=False):
    try:
        if import_name.startswith('__runtime__.'):
            return sys.modules[import_name]
        elif '.' in import_name:
            items = import_name.split('.')
            module = '.'.join(items[:-1])
            obj = items[-1]
            submodule = getattr(__import__(module, None, None, [obj]), obj)
            if isinstance(submodule, (ModuleType, type)):
                return submodule
            return __import__(import_name, None, None, [obj])
        else:
            return __import__(import_name)
    except (ImportError, AttributeError, KeyError):
        if safe:
            return None
        raise

# https://github.com/python/cpython/blob/a8912a0f8d9eba6d502c37d522221f9933e976db/Lib/pickle.py#L322-L333
def _getattribute(obj, name):
    for subpath in name.split('.'):
        if subpath == '<locals>':
            raise AttributeError("Can't get local attribute {!r} on {!r}"
                                 .format(name, obj))
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError("Can't get attribute {!r} on {!r}"
                                 .format(name, obj))
    return obj, parent

def _locate_function(obj, pickler=None):
    module_name = getattr(obj, '__module__', None)
    if module_name in ['__main__', None] or \
            pickler and is_dill(pickler, child=False) and pickler._session and module_name == pickler._main.__name__:
        return False
    if hasattr(obj, '__qualname__'):
        module = _import_module(module_name, safe=True)
        try:
            found, _ = _getattribute(module, obj.__qualname__)
            return found is obj
        except AttributeError:
            return False
    else:
        found = _import_module(module_name + '.' + obj.__name__, safe=True)
        return found is obj


def _setitems(dest, source):
    for k, v in source.items():
        dest[k] = v


def _save_with_postproc(pickler, reduction, is_pickler_dill=None, obj=Getattr.NO_DEFAULT, postproc_list=None):
    if obj is Getattr.NO_DEFAULT:
        obj = Reduce(reduction) # pragma: no cover

    if is_pickler_dill is None:
        is_pickler_dill = is_dill(pickler, child=True)
    if is_pickler_dill:
        # assert id(obj) not in pickler._postproc, str(obj) + ' already pushed on stack!'
        # if not hasattr(pickler, 'x'): pickler.x = 0
        # print(pickler.x*' ', 'push', obj, id(obj), pickler._recurse)
        # pickler.x += 1
        if postproc_list is None:
            postproc_list = []

        # Recursive object not supported. Default to a global instead.
        if id(obj) in pickler._postproc:
            name = '%s.%s ' % (obj.__module__, getattr(obj, '__qualname__', obj.__name__)) if hasattr(obj, '__module__') else ''
            warnings.warn('Cannot pickle %r: %shas recursive self-references that trigger a RecursionError.' % (obj, name), PicklingWarning)
            pickler.save_global(obj)
            return
        pickler._postproc[id(obj)] = postproc_list

    # TODO: Use state_setter in Python 3.8 to allow for faster cPickle implementations
    pickler.save_reduce(*reduction, obj=obj)

    if is_pickler_dill:
        # pickler.x -= 1
        # print(pickler.x*' ', 'pop', obj, id(obj))
        postproc = pickler._postproc.pop(id(obj))
        # assert postproc_list == postproc, 'Stack tampered!'
        for reduction in reversed(postproc):
            if reduction[0] is _setitems:
                # use the internal machinery of pickle.py to speedup when
                # updating a dictionary in postproc
                dest, source = reduction[1]
                if source:
                    pickler.write(pickler.get(pickler.memo[id(dest)][0]))
                    if sys.hexversion < 0x30e00a1:
                        pickler._batch_setitems(iter(source.items()))
                    else:
                        pickler._batch_setitems(iter(source.items()), obj=obj)
                else:
                    # Updating with an empty dictionary. Same as doing nothing.
                    continue
            else:
                pickler.save_reduce(*reduction)
            # pop None created by calling preprocessing step off stack
            pickler.write(POP)

#@register(CodeType)
#def save_code(pickler, obj):
#    logger.trace(pickler, "Co: %s", obj)
#    pickler.save_reduce(_unmarshal, (marshal.dumps(obj),), obj=obj)
#    logger.trace(pickler, "# Co")
#    return

# The following function is based on 'save_codeobject' from 'cloudpickle'
# Copyright (c) 2012, Regents of the University of California.
# Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
# License: https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE
@register(CodeType)
def save_code(pickler, obj):
    logger.trace(pickler, "Co: %s", obj)
    if hasattr(obj, "co_endlinetable"): # python 3.11a (20 args)
        args = (
            obj.co_lnotab, # for < python 3.10 [not counted in args]
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname,
            obj.co_firstlineno, obj.co_linetable, obj.co_endlinetable,
            obj.co_columntable, obj.co_exceptiontable, obj.co_freevars,
            obj.co_cellvars
        )
    elif hasattr(obj, "co_exceptiontable"): # python 3.11 (18 args)
        with warnings.catch_warnings():
            if not OLD312a7: # issue 597
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            args = (
                obj.co_lnotab, # for < python 3.10 [not counted in args]
                obj.co_argcount, obj.co_posonlyargcount,
                obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
                obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
                obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname,
                obj.co_firstlineno, obj.co_linetable, obj.co_exceptiontable,
                obj.co_freevars, obj.co_cellvars
            )
    elif hasattr(obj, "co_qualname"): # pypy 3.11 7.3.19+ (17 args)
        args = (
            obj.co_lnotab, obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname,
            obj.co_firstlineno, obj.co_linetable, obj.co_freevars,
            obj.co_cellvars
        )
    elif hasattr(obj, "co_linetable"): # python 3.10 (16 args)
        args = (
            obj.co_lnotab, # for < python 3.10 [not counted in args]
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_linetable, obj.co_freevars,
            obj.co_cellvars
        )
    elif hasattr(obj, "co_posonlyargcount"): # python 3.8 (16 args)
        args = (
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
            obj.co_cellvars
        )
    else: # python 3.7 (15 args)
        args = (
            obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals,
            obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts,
            obj.co_names, obj.co_varnames, obj.co_filename,
            obj.co_name, obj.co_firstlineno, obj.co_lnotab,
            obj.co_freevars, obj.co_cellvars
        )

    pickler.save_reduce(_create_code, args, obj=obj)
    logger.trace(pickler, "# Co")
    return

def _repr_dict(obj):
    """Make a short string representation of a dictionary."""
    return "<%s object at %#012x>" % (type(obj).__name__, id(obj))

@register(dict)
def save_module_dict(pickler, obj):
    if is_dill(pickler, child=False) and obj == pickler._main.__dict__ and \
            not (pickler._session and pickler._first_pass):
        logger.trace(pickler, "D1: %s", _repr_dict(obj)) # obj
        pickler.write(bytes('c__builtin__\n__main__\n', 'UTF-8'))
        logger.trace(pickler, "# D1")
    elif (not is_dill(pickler, child=False)) and (obj == _main_module.__dict__):
        logger.trace(pickler, "D3: %s", _repr_dict(obj)) # obj
        pickler.write(bytes('c__main__\n__dict__\n', 'UTF-8'))  #XXX: works in general?
        logger.trace(pickler, "# D3")
    elif '__name__' in obj and obj != _main_module.__dict__ \
            and type(obj['__name__']) is str \
            and obj is getattr(_import_module(obj['__name__'],True), '__dict__', None):
        logger.trace(pickler, "D4: %s", _repr_dict(obj)) # obj
        pickler.write(bytes('c%s\n__dict__\n' % obj['__name__'], 'UTF-8'))
        logger.trace(pickler, "# D4")
    else:
        logger.trace(pickler, "D2: %s", _repr_dict(obj)) # obj
        if is_dill(pickler, child=False) and pickler._session:
            # we only care about session the first pass thru
            pickler._first_pass = False
        StockPickler.save_dict(pickler, obj)
        logger.trace(pickler, "# D2")
    return


if not OLD310 and MAPPING_PROXY_TRICK:
    def save_dict_view(dicttype):
        def save_dict_view_for_function(func):
            def _save_dict_view(pickler, obj):
                logger.trace(pickler, "Dkvi: <%s>", obj)
                mapping = obj.mapping | _dictproxy_helper_instance
                pickler.save_reduce(func, (mapping,), obj=obj)
                logger.trace(pickler, "# Dkvi")
            return _save_dict_view
        return [
            (funcname, save_dict_view_for_function(getattr(dicttype, funcname)))
            for funcname in ('keys', 'values', 'items')
        ]
else:
    # The following functions are based on 'cloudpickle'
    # https://github.com/cloudpipe/cloudpickle/blob/5d89947288a18029672596a4d719093cc6d5a412/cloudpickle/cloudpickle.py#L922-L940
    # Copyright (c) 2012, Regents of the University of California.
    # Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
    # License: https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE
    def save_dict_view(dicttype):
        def save_dict_keys(pickler, obj):
            logger.trace(pickler, "Dk: <%s>", obj)
            dict_constructor = _shims.Reduce(dicttype.fromkeys, (list(obj),))
            pickler.save_reduce(dicttype.keys, (dict_constructor,), obj=obj)
            logger.trace(pickler, "# Dk")

        def save_dict_values(pickler, obj):
            logger.trace(pickler, "Dv: <%s>", obj)
            dict_constructor = _shims.Reduce(dicttype, (enumerate(obj),))
            pickler.save_reduce(dicttype.values, (dict_constructor,), obj=obj)
            logger.trace(pickler, "# Dv")

        def save_dict_items(pickler, obj):
            logger.trace(pickler, "Di: <%s>", obj)
            pickler.save_reduce(dicttype.items, (dicttype(obj),), obj=obj)
            logger.trace(pickler, "# Di")

        return (
            ('keys', save_dict_keys),
            ('values', save_dict_values),
            ('items', save_dict_items)
        )

for __dicttype in (
      dict,
      OrderedDict
):
    __obj = __dicttype()
    for __funcname, __savefunc in save_dict_view(__dicttype):
        __tview = type(getattr(__obj, __funcname)())
        if __tview not in Pickler.dispatch:
            Pickler.dispatch[__tview] = __savefunc
del __dicttype, __obj, __funcname, __tview, __savefunc


@register(ClassType)
def save_classobj(pickler, obj): #FIXME: enable pickler._byref
    if not _locate_function(obj, pickler):
        logger.trace(pickler, "C1: %s", obj)
        pickler.save_reduce(ClassType, (obj.__name__, obj.__bases__,
                                        obj.__dict__), obj=obj)
                                       #XXX: or obj.__dict__.copy()), obj=obj) ?
        logger.trace(pickler, "# C1")
    else:
        logger.trace(pickler, "C2: %s", obj)
        name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
        StockPickler.save_global(pickler, obj, name=name)
        logger.trace(pickler, "# C2")
    return

@register(typing._GenericAlias)
def save_generic_alias(pickler, obj):
    args = obj.__args__
    if type(obj.__reduce__()) is str:
        logger.trace(pickler, "Ga0: %s", obj)
        StockPickler.save_global(pickler, obj, name=obj.__reduce__())
        logger.trace(pickler, "# Ga0")
    elif obj.__origin__ is tuple and (not args or args == ((),)):
        logger.trace(pickler, "Ga1: %s", obj)
        pickler.save_reduce(_create_typing_tuple, (args,), obj=obj)
        logger.trace(pickler, "# Ga1")
    else:
        logger.trace(pickler, "Ga2: %s", obj)
        StockPickler.save_reduce(pickler, *obj.__reduce__(), obj=obj)
        logger.trace(pickler, "# Ga2")
    return

if ThreadHandleType:
    @register(ThreadHandleType)
    def save_thread_handle(pickler, obj):
        logger.trace(pickler, "Th: %s", obj)
        pickler.save_reduce(_create_thread_handle, (obj.ident, obj.is_done()), obj=obj)
        logger.trace(pickler, "# Th")
        return

@register(LockType) #XXX: copied Thread will have new Event (due to new Lock)
def save_lock(pickler, obj):
    logger.trace(pickler, "Lo: %s", obj)
    pickler.save_reduce(_create_lock, (obj.locked(),), obj=obj)
    logger.trace(pickler, "# Lo")
    return

@register(RLockType)
def save_rlock(pickler, obj):
    logger.trace(pickler, "RL: %s", obj)
    r = obj.__repr__() # don't use _release_save as it unlocks the lock
    count = int(r.split('count=')[1].split()[0].rstrip('>'))
    owner = int(r.split('owner=')[1].split()[0])
    pickler.save_reduce(_create_rlock, (count,owner,), obj=obj)
    logger.trace(pickler, "# RL")
    return

#@register(SocketType) #FIXME: causes multiprocess test_pickling FAIL
def save_socket(pickler, obj):
    logger.trace(pickler, "So: %s", obj)
    pickler.save_reduce(*reduce_socket(obj))
    logger.trace(pickler, "# So")
    return

def _save_file(pickler, obj, open_):
    if obj.closed:
        position = 0
    else:
        obj.flush()
        if obj in (sys.__stdout__, sys.__stderr__, sys.__stdin__):
            position = -1
        else:
            position = obj.tell()
    if is_dill(pickler, child=True) and pickler._fmode == FILE_FMODE:
        f = open_(obj.name, "r")
        fdata = f.read()
        f.close()
    else:
        fdata = ""
    if is_dill(pickler, child=True):
        strictio = pickler._strictio
        fmode = pickler._fmode
    else:
        strictio = False
        fmode = 0 # HANDLE_FMODE
    pickler.save_reduce(_create_filehandle, (obj.name, obj.mode, position,
                                             obj.closed, open_, strictio,
                                             fmode, fdata), obj=obj)
    return


@register(FileType) #XXX: in 3.x has buffer=0, needs different _create?
@register(BufferedReaderType)
@register(BufferedWriterType)
@register(TextWrapperType)
def save_file(pickler, obj):
    logger.trace(pickler, "Fi: %s", obj)
    f = _save_file(pickler, obj, open)
    logger.trace(pickler, "# Fi")
    return f

if BufferedRandomType:
    @register(BufferedRandomType)
    def save_file(pickler, obj):
        logger.trace(pickler, "Fi: %s", obj)
        f = _save_file(pickler, obj, open)
        logger.trace(pickler, "# Fi")
        return f

if PyTextWrapperType:
    @register(PyBufferedReaderType)
    @register(PyBufferedWriterType)
    @register(PyTextWrapperType)
    def save_file(pickler, obj):
        logger.trace(pickler, "Fi: %s", obj)
        f = _save_file(pickler, obj, _open)
        logger.trace(pickler, "# Fi")
        return f

    if PyBufferedRandomType:
        @register(PyBufferedRandomType)
        def save_file(pickler, obj):
            logger.trace(pickler, "Fi: %s", obj)
            f = _save_file(pickler, obj, _open)
            logger.trace(pickler, "# Fi")
            return f


# The following two functions are based on 'saveCStringIoInput'
# and 'saveCStringIoOutput' from spickle
# Copyright (c) 2011 by science+computing ag
# License: http://www.apache.org/licenses/LICENSE-2.0
if InputType:
    @register(InputType)
    def save_stringi(pickler, obj):
        logger.trace(pickler, "Io: %s", obj)
        if obj.closed:
            value = ''; position = 0
        else:
            value = obj.getvalue(); position = obj.tell()
        pickler.save_reduce(_create_stringi, (value, position, \
                                              obj.closed), obj=obj)
        logger.trace(pickler, "# Io")
        return

    @register(OutputType)
    def save_stringo(pickler, obj):
        logger.trace(pickler, "Io: %s", obj)
        if obj.closed:
            value = ''; position = 0
        else:
            value = obj.getvalue(); position = obj.tell()
        pickler.save_reduce(_create_stringo, (value, position, \
                                              obj.closed), obj=obj)
        logger.trace(pickler, "# Io")
        return

if LRUCacheType is not None:
    from functools import lru_cache
    @register(LRUCacheType)
    def save_lru_cache(pickler, obj):
        logger.trace(pickler, "LRU: %s", obj)
        if OLD39:
            kwargs = obj.cache_info()
            args = (kwargs.maxsize,)
        else:
            kwargs = obj.cache_parameters()
            args = (kwargs['maxsize'], kwargs['typed'])
        if args != lru_cache.__defaults__:
            wrapper = Reduce(lru_cache, args, is_callable=True)
        else:
            wrapper = lru_cache
        pickler.save_reduce(wrapper, (obj.__wrapped__,), obj=obj)
        logger.trace(pickler, "# LRU")
        return

@register(SuperType)
def save_super(pickler, obj):
    logger.trace(pickler, "Su: %s", obj)
    pickler.save_reduce(super, (obj.__thisclass__, obj.__self__), obj=obj)
    logger.trace(pickler, "# Su")
    return

if IS_PYPY:
    @register(MethodType)
    def save_instancemethod0(pickler, obj):
        code = getattr(obj.__func__, '__code__', None)
        if code is not None and type(code) is not CodeType \
              and getattr(obj.__self__, obj.__name__) == obj:
            # Some PyPy builtin functions have no module name
            logger.trace(pickler, "Me2: %s", obj)
            # TODO: verify that this works for all PyPy builtin methods
            pickler.save_reduce(getattr, (obj.__self__, obj.__name__), obj=obj)
            logger.trace(pickler, "# Me2")
            return

        logger.trace(pickler, "Me1: %s", obj)
        pickler.save_reduce(MethodType, (obj.__func__, obj.__self__), obj=obj)
        logger.trace(pickler, "# Me1")
        return
else:
    @register(MethodType)
    def save_instancemethod0(pickler, obj):
        logger.trace(pickler, "Me1: %s", obj)
        pickler.save_reduce(MethodType, (obj.__func__, obj.__self__), obj=obj)
        logger.trace(pickler, "# Me1")
        return

if not IS_PYPY:
    @register(MemberDescriptorType)
    @register(GetSetDescriptorType)
    @register(MethodDescriptorType)
    @register(WrapperDescriptorType)
    @register(ClassMethodDescriptorType)
    def save_wrapper_descriptor(pickler, obj):
        logger.trace(pickler, "Wr: %s", obj)
        pickler.save_reduce(_getattr, (obj.__objclass__, obj.__name__,
                                       obj.__repr__()), obj=obj)
        logger.trace(pickler, "# Wr")
        return
else:
    @register(MemberDescriptorType)
    @register(GetSetDescriptorType)
    def save_wrapper_descriptor(pickler, obj):
        logger.trace(pickler, "Wr: %s", obj)
        pickler.save_reduce(_getattr, (obj.__objclass__, obj.__name__,
                                       obj.__repr__()), obj=obj)
        logger.trace(pickler, "# Wr")
        return

@register(CellType)
def save_cell(pickler, obj):
    try:
        f = obj.cell_contents
    except ValueError: # cell is empty
        logger.trace(pickler, "Ce3: %s", obj)
        # _shims._CELL_EMPTY is defined in _shims.py to support PyPy 2.7.
        # It unpickles to a sentinel object _dill._CELL_EMPTY, also created in
        # _shims.py. This object is not present in Python 3 because the cell's
        # contents can be deleted in newer versions of Python. The reduce object
        # will instead unpickle to None if unpickled in Python 3.

        # When breaking changes are made to dill, (_shims._CELL_EMPTY,) can
        # be replaced by () OR the delattr function can be removed repending on
        # whichever is more convienient.
        pickler.save_reduce(_create_cell, (_shims._CELL_EMPTY,), obj=obj)
        # Call the function _delattr on the cell's cell_contents attribute
        # The result of this function call will be None
        pickler.save_reduce(_shims._delattr, (obj, 'cell_contents'))
        # pop None created by calling _delattr off stack
        pickler.write(POP)
        logger.trace(pickler, "# Ce3")
        return
    if is_dill(pickler, child=True):
        if id(f) in pickler._postproc:
            # Already seen. Add to its postprocessing.
            postproc = pickler._postproc[id(f)]
        else:
            # Haven't seen it. Add to the highest possible object and set its
            # value as late as possible to prevent cycle.
            postproc = next(iter(pickler._postproc.values()), None)
        if postproc is not None:
            logger.trace(pickler, "Ce2: %s", obj)
            # _CELL_REF is defined in _shims.py to support older versions of
            # dill. When breaking changes are made to dill, (_CELL_REF,) can
            # be replaced by ()
            pickler.save_reduce(_create_cell, (_CELL_REF,), obj=obj)
            postproc.append((_shims._setattr, (obj, 'cell_contents', f)))
            logger.trace(pickler, "# Ce2")
            return
    logger.trace(pickler, "Ce1: %s", obj)
    pickler.save_reduce(_create_cell, (f,), obj=obj)
    logger.trace(pickler, "# Ce1")
    return

if MAPPING_PROXY_TRICK:
    @register(DictProxyType)
    def save_dictproxy(pickler, obj):
        logger.trace(pickler, "Mp: %s", _repr_dict(obj)) # obj
        mapping = obj | _dictproxy_helper_instance
        pickler.save_reduce(DictProxyType, (mapping,), obj=obj)
        logger.trace(pickler, "# Mp")
        return
else:
    @register(DictProxyType)
    def save_dictproxy(pickler, obj):
        logger.trace(pickler, "Mp: %s", _repr_dict(obj)) # obj
        pickler.save_reduce(DictProxyType, (obj.copy(),), obj=obj)
        logger.trace(pickler, "# Mp")
        return

@register(SliceType)
def save_slice(pickler, obj):
    logger.trace(pickler, "Sl: %s", obj)
    pickler.save_reduce(slice, (obj.start, obj.stop, obj.step), obj=obj)
    logger.trace(pickler, "# Sl")
    return

@register(XRangeType)
@register(EllipsisType)
@register(NotImplementedType)
def save_singleton(pickler, obj):
    logger.trace(pickler, "Si: %s", obj)
    pickler.save_reduce(_eval_repr, (obj.__repr__(),), obj=obj)
    logger.trace(pickler, "# Si")
    return

def _proxy_helper(obj): # a dead proxy returns a reference to None
    """get memory address of proxy's reference object"""
    _repr = repr(obj)
    try: _str = str(obj)
    except ReferenceError: # it's a dead proxy
        return id(None)
    if _str == _repr: return id(obj) # it's a repr
    try: # either way, it's a proxy from here
        address = int(_str.rstrip('>').split(' at ')[-1], base=16)
    except ValueError: # special case: proxy of a 'type'
        if not IS_PYPY:
            address = int(_repr.rstrip('>').split(' at ')[-1], base=16)
        else:
            objects = iter(gc.get_objects())
            for _obj in objects:
                if repr(_obj) == _str: return id(_obj)
            # all bad below... nothing found so throw ReferenceError
            msg = "Cannot reference object for proxy at '%s'" % id(obj)
            raise ReferenceError(msg)
    return address

def _locate_object(address, module=None):
    """get object located at the given memory address (inverse of id(obj))"""
    special = [None, True, False] #XXX: more...?
    for obj in special:
        if address == id(obj): return obj
    if module:
        objects = iter(module.__dict__.values())
    else: objects = iter(gc.get_objects())
    for obj in objects:
        if address == id(obj): return obj
    # all bad below... nothing found so throw ReferenceError or TypeError
    try: address = hex(address)
    except TypeError:
        raise TypeError("'%s' is not a valid memory address" % str(address))
    raise ReferenceError("Cannot reference object at '%s'" % address)

@register(ReferenceType)
def save_weakref(pickler, obj):
    refobj = obj()
    logger.trace(pickler, "R1: %s", obj)
   #refobj = ctypes.pythonapi.PyWeakref_GetObject(obj) # dead returns "None"
    pickler.save_reduce(_create_weakref, (refobj,), obj=obj)
    logger.trace(pickler, "# R1")
    return

@register(ProxyType)
@register(CallableProxyType)
def save_weakproxy(pickler, obj):
    # Must do string substitution here and use %r to avoid ReferenceError.
    logger.trace(pickler, "R2: %r" % obj)
    refobj = _locate_object(_proxy_helper(obj))
    pickler.save_reduce(_create_weakproxy, (refobj, callable(obj)), obj=obj)
    logger.trace(pickler, "# R2")
    return

def _is_builtin_module(module):
    if not hasattr(module, "__file__"): return True
    if module.__file__ is None: return False
    # If a module file name starts with prefix, it should be a builtin
    # module, so should always be pickled as a reference.
    names = ["base_prefix", "base_exec_prefix", "exec_prefix", "prefix", "real_prefix"]
    rp = os.path.realpath
    # See https://github.com/uqfoundation/dill/issues/566
    return (
        any(
            module.__file__.startswith(getattr(sys, name))
            or rp(module.__file__).startswith(rp(getattr(sys, name)))
            for name in names
            if hasattr(sys, name)
        )
        or module.__file__.endswith(EXTENSION_SUFFIXES)
        or 'site-packages' in module.__file__
    )

def _is_imported_module(module):
    return getattr(module, '__loader__', None) is not None or module in sys.modules.values()

@register(ModuleType)
def save_module(pickler, obj):
    if False: #_use_diff:
        if obj.__name__.split('.', 1)[0] != "dill":
            try:
                changed = diff.whats_changed(obj, seen=pickler._diff_cache)[0]
            except RuntimeError:  # not memorised module, probably part of dill
                pass
            else:
                logger.trace(pickler, "M2: %s with diff", obj)
                logger.info("Diff: %s", changed.keys())
                pickler.save_reduce(_import_module, (obj.__name__,), obj=obj,
                                    state=changed)
                logger.trace(pickler, "# M2")
                return

        logger.trace(pickler, "M1: %s", obj)
        pickler.save_reduce(_import_module, (obj.__name__,), obj=obj)
        logger.trace(pickler, "# M1")
    else:
        builtin_mod = _is_builtin_module(obj)
        is_session_main = is_dill(pickler, child=True) and obj is pickler._main
        if (obj.__name__ not in ("builtins", "dill", "dill._dill") and not builtin_mod
                or is_session_main):
            logger.trace(pickler, "M1: %s", obj)
            # Hack for handling module-type objects in load_module().
            mod_name = obj.__name__ if _is_imported_module(obj) else '__runtime__.%s' % obj.__name__
            # Second references are saved as __builtin__.__main__ in save_module_dict().
            main_dict = obj.__dict__.copy()
            for item in ('__builtins__', '__loader__'):
                main_dict.pop(item, None)
            for item in IPYTHON_SINGLETONS: #pragma: no cover
                if getattr(main_dict.get(item), '__module__', '').startswith('IPython'):
                    del main_dict[item]
            pickler.save_reduce(_import_module, (mod_name,), obj=obj, state=main_dict)
            logger.trace(pickler, "# M1")
        elif obj.__name__ == "dill._dill":
            logger.trace(pickler, "M2: %s", obj)
            pickler.save_global(obj, name="_dill")
            logger.trace(pickler, "# M2")
        else:
            logger.trace(pickler, "M2: %s", obj)
            pickler.save_reduce(_import_module, (obj.__name__,), obj=obj)
            logger.trace(pickler, "# M2")
    return

# The following function is based on '_extract_class_dict' from 'cloudpickle'
# Copyright (c) 2012, Regents of the University of California.
# Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
# License: https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE
def _get_typedict_type(cls, clsdict, attrs, postproc_list):
    """Retrieve a copy of the dict of a class without the inherited methods"""
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)
    to_remove = []
    for name, value in dict.items(clsdict):
        try:
            base_value = inherited_dict[name]
            if value is base_value and hasattr(value, '__qualname__'):
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        dict.pop(clsdict, name)

    if issubclass(type(cls), type):
        clsdict.pop('__dict__', None)
        clsdict.pop('__weakref__', None)
        # clsdict.pop('__prepare__', None)
    return clsdict, attrs

def _get_typedict_abc(obj, _dict, attrs, postproc_list):
    if hasattr(abc, '_get_dump'):
        (registry, _, _, _) = abc._get_dump(obj)
        register = obj.register
        postproc_list.extend((register, (reg(),)) for reg in registry)
    elif hasattr(obj, '_abc_registry'):
        registry = obj._abc_registry
        register = obj.register
        postproc_list.extend((register, (reg,)) for reg in registry)
    else:
        raise PicklingError("Cannot find registry of ABC %s", obj)

    if '_abc_registry' in _dict:
        _dict.pop('_abc_registry', None)
        _dict.pop('_abc_cache', None)
        _dict.pop('_abc_negative_cache', None)
        # _dict.pop('_abc_negative_cache_version', None)
    else:
        _dict.pop('_abc_impl', None)
    return _dict, attrs

@register(TypeType)
def save_type(pickler, obj, postproc_list=None):
    if obj in _typemap:
        logger.trace(pickler, "T1: %s", obj)
        # if obj in _incedental_types:
        #     warnings.warn('Type %r may only exist on this implementation of Python and cannot be unpickled in other implementations.' % (obj,), PicklingWarning)
        pickler.save_reduce(_load_type, (_typemap[obj],), obj=obj)
        logger.trace(pickler, "# T1")
    elif obj.__bases__ == (tuple,) and all([hasattr(obj, attr) for attr in ('_fields','_asdict','_make','_replace')]):
        # special case: namedtuples
        logger.trace(pickler, "T6: %s", obj)

        obj_name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
        if obj.__name__ != obj_name:
            if postproc_list is None:
                postproc_list = []
            postproc_list.append((setattr, (obj, '__qualname__', obj_name)))

        if not obj._field_defaults:
            _save_with_postproc(pickler, (_create_namedtuple, (obj.__name__, obj._fields, obj.__module__)), obj=obj, postproc_list=postproc_list)
        else:
            defaults = [obj._field_defaults[field] for field in obj._fields if field in obj._field_defaults]
            _save_with_postproc(pickler, (_create_namedtuple, (obj.__name__, obj._fields, obj.__module__, defaults)), obj=obj, postproc_list=postproc_list)
        logger.trace(pickler, "# T6")
        return

    # special caes: NoneType, NotImplementedType, EllipsisType, EnumMeta, etc
    elif obj is type(None):
        logger.trace(pickler, "T7: %s", obj)
        #XXX: pickler.save_reduce(type, (None,), obj=obj)
        pickler.write(GLOBAL + b'__builtin__\nNoneType\n')
        logger.trace(pickler, "# T7")
    elif obj is NotImplementedType:
        logger.trace(pickler, "T7: %s", obj)
        pickler.save_reduce(type, (NotImplemented,), obj=obj)
        logger.trace(pickler, "# T7")
    elif obj is EllipsisType:
        logger.trace(pickler, "T7: %s", obj)
        pickler.save_reduce(type, (Ellipsis,), obj=obj)
        logger.trace(pickler, "# T7")
    elif obj is EnumMeta:
        logger.trace(pickler, "T7: %s", obj)
        pickler.write(GLOBAL + b'enum\nEnumMeta\n')
        logger.trace(pickler, "# T7")
    elif obj is ExceptHookArgsType: #NOTE: must be after NoneType for pypy
        logger.trace(pickler, "T7: %s", obj)
        pickler.write(GLOBAL + b'threading\nExceptHookArgs\n')
        logger.trace(pickler, "# T7")

    else:
        _byref = getattr(pickler, '_byref', None)
        obj_recursive = id(obj) in getattr(pickler, '_postproc', ())
        incorrectly_named = not _locate_function(obj, pickler)
        if not _byref and not obj_recursive and incorrectly_named: # not a function, but the name was held over
            if postproc_list is None:
                postproc_list = []

            # thanks to Tom Stepleton pointing out pickler._session unneeded
            logger.trace(pickler, "T2: %s", obj)
            _dict, attrs = _get_typedict_type(obj, obj.__dict__.copy(), None, postproc_list) # copy dict proxy to a dict

           #print (_dict)
           #print ("%s\n%s" % (type(obj), obj.__name__))
           #print ("%s\n%s" % (obj.__bases__, obj.__dict__))
            slots = _dict.get('__slots__', ())
            if type(slots) == str:
                # __slots__ accepts a single string
                slots = (slots,)

            for name in slots:
                _dict.pop(name, None)

            if isinstance(obj, abc.ABCMeta):
                logger.trace(pickler, "ABC: %s", obj)
                _dict, attrs = _get_typedict_abc(obj, _dict, attrs, postproc_list)
                logger.trace(pickler, "# ABC")

            qualname = getattr(obj, '__qualname__', None)
            if attrs is not None:
                for k, v in attrs.items():
                    postproc_list.append((setattr, (obj, k, v)))
                # TODO: Consider using the state argument to save_reduce?
            if qualname is not None:
                postproc_list.append((setattr, (obj, '__qualname__', qualname)))

            if not hasattr(obj, '__orig_bases__'):
                _save_with_postproc(pickler, (_create_type, (
                    type(obj), obj.__name__, obj.__bases__, _dict
                )), obj=obj, postproc_list=postproc_list)
            else:
                # This case will always work, but might be overkill.
                _metadict = {
                    'metaclass': type(obj)
                }

                if _dict:
                    _dict_update = PartialType(_setitems, source=_dict)
                else:
                    _dict_update = None

                _save_with_postproc(pickler, (new_class, (
                    obj.__name__, obj.__orig_bases__, _metadict, _dict_update
                )), obj=obj, postproc_list=postproc_list)
            logger.trace(pickler, "# T2")
        else:
            obj_name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
            logger.trace(pickler, "T4: %s", obj)
            if incorrectly_named:
                warnings.warn(
                    "Cannot locate reference to %r." % (obj,),
                    PicklingWarning,
                    stacklevel=3,
                )
            if obj_recursive:
                warnings.warn(
                    "Cannot pickle %r: %s.%s has recursive self-references that "
                    "trigger a RecursionError." % (obj, obj.__module__, obj_name),
                    PicklingWarning,
                    stacklevel=3,
                )
           #print (obj.__dict__)
           #print ("%s\n%s" % (type(obj), obj.__name__))
           #print ("%s\n%s" % (obj.__bases__, obj.__dict__))
            StockPickler.save_global(pickler, obj, name=obj_name)
            logger.trace(pickler, "# T4")
    return

@register(property)
@register(abc.abstractproperty)
def save_property(pickler, obj):
    logger.trace(pickler, "Pr: %s", obj)
    pickler.save_reduce(type(obj), (obj.fget, obj.fset, obj.fdel, obj.__doc__),
                        obj=obj)
    logger.trace(pickler, "# Pr")

@register(staticmethod)
@register(classmethod)
@register(abc.abstractstaticmethod)
@register(abc.abstractclassmethod)
def save_classmethod(pickler, obj):
    logger.trace(pickler, "Cm: %s", obj)
    orig_func = obj.__func__

    # if type(obj.__dict__) is dict:
    #     if obj.__dict__:
    #         state = obj.__dict__
    #     else:
    #         state = None
    # else:
    #     state = (None, {'__dict__', obj.__dict__})

    pickler.save_reduce(type(obj), (orig_func,), obj=obj)
    logger.trace(pickler, "# Cm")

@register(FunctionType)
def save_function(pickler, obj):
    if not _locate_function(obj, pickler):
        if type(obj.__code__) is not CodeType:
            # Some PyPy builtin functions have no module name, and thus are not
            # able to be located
            module_name = getattr(obj, '__module__', None)
            if module_name is None:
                module_name = __builtin__.__name__
            module = _import_module(module_name, safe=True)
            _pypy_builtin = False
            try:
                found, _ = _getattribute(module, obj.__qualname__)
                if getattr(found, '__func__', None) is obj:
                    _pypy_builtin = True
            except AttributeError:
                pass

            if _pypy_builtin:
                logger.trace(pickler, "F3: %s", obj)
                pickler.save_reduce(getattr, (found, '__func__'), obj=obj)
                logger.trace(pickler, "# F3")
                return

        logger.trace(pickler, "F1: %s", obj)
        _recurse = getattr(pickler, '_recurse', None)
        _postproc = getattr(pickler, '_postproc', None)
        _main_modified = getattr(pickler, '_main_modified', None)
        _original_main = getattr(pickler, '_original_main', __builtin__)#'None'
        postproc_list = []
        if _recurse:
            # recurse to get all globals referred to by obj
            from .detect import globalvars
            globs_copy = globalvars(obj, recurse=True, builtin=True)

            # Add the name of the module to the globs dictionary to prevent
            # the duplication of the dictionary. Pickle the unpopulated
            # globals dictionary and set the remaining items after the function
            # is created to correctly handle recursion.
            globs = {'__name__': obj.__module__}
        else:
            globs_copy = obj.__globals__

            # If the globals is the __dict__ from the module being saved as a
            # session, substitute it by the dictionary being actually saved.
            if _main_modified and globs_copy is _original_main.__dict__:
                globs_copy = getattr(pickler, '_main', _original_main).__dict__
                globs = globs_copy
            # If the globals is a module __dict__, do not save it in the pickle.
            elif globs_copy is not None and obj.__module__ is not None and \
                    getattr(_import_module(obj.__module__, True), '__dict__', None) is globs_copy:
                globs = globs_copy
            else:
                globs = {'__name__': obj.__module__}

        if globs_copy is not None and globs is not globs_copy:
            # In the case that the globals are copied, we need to ensure that
            # the globals dictionary is updated when all objects in the
            # dictionary are already created.
            glob_ids = {id(g) for g in globs_copy.values()}
            for stack_element in _postproc:
                if stack_element in glob_ids:
                    _postproc[stack_element].append((_setitems, (globs, globs_copy)))
                    break
            else:
                postproc_list.append((_setitems, (globs, globs_copy)))

        closure = obj.__closure__
        state_dict = {}
        for fattrname in ('__doc__', '__kwdefaults__', '__annotations__'):
            fattr = getattr(obj, fattrname, None)
            if fattr is not None:
                state_dict[fattrname] = fattr
        if obj.__qualname__ != obj.__name__:
            state_dict['__qualname__'] = obj.__qualname__
        if '__name__' not in globs or obj.__module__ != globs['__name__']:
            state_dict['__module__'] = obj.__module__

        state = obj.__dict__
        if type(state) is not dict:
            state_dict['__dict__'] = state
            state = None
        if state_dict:
            state = state, state_dict

        _save_with_postproc(pickler, (_create_function, (
                obj.__code__, globs, obj.__name__, obj.__defaults__,
                closure
        ), state), obj=obj, postproc_list=postproc_list)

        # Lift closure cell update to earliest function (#458)
        if _postproc:
            topmost_postproc = next(iter(_postproc.values()), None)
            if closure and topmost_postproc:
                for cell in closure:
                    possible_postproc = (setattr, (cell, 'cell_contents', obj))
                    try:
                        topmost_postproc.remove(possible_postproc)
                    except ValueError:
                        continue

                    # Change the value of the cell
                    pickler.save_reduce(*possible_postproc)
                    # pop None created by calling preprocessing step off stack
                    pickler.write(POP)

        logger.trace(pickler, "# F1")
    else:
        logger.trace(pickler, "F2: %s", obj)
        name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
        StockPickler.save_global(pickler, obj, name=name)
        logger.trace(pickler, "# F2")
    return

if HAS_CTYPES and hasattr(ctypes, 'pythonapi'):
    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    _PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    _PyCapsule_GetPointer.restype = ctypes.c_void_p
    _PyCapsule_GetDestructor = ctypes.pythonapi.PyCapsule_GetDestructor
    _PyCapsule_GetDestructor.argtypes = (ctypes.py_object,)
    _PyCapsule_GetDestructor.restype = ctypes.c_void_p
    _PyCapsule_GetContext = ctypes.pythonapi.PyCapsule_GetContext
    _PyCapsule_GetContext.argtypes = (ctypes.py_object,)
    _PyCapsule_GetContext.restype = ctypes.c_void_p
    _PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
    _PyCapsule_GetName.argtypes = (ctypes.py_object,)
    _PyCapsule_GetName.restype = ctypes.c_char_p
    _PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
    _PyCapsule_IsValid.argtypes = (ctypes.py_object, ctypes.c_char_p)
