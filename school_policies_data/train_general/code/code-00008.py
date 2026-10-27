    lang_rank: dict[str, int] = _FREQUENCIES_RANK[language]

    ordered_characters_count: int = len(ordered_characters)
    target_language_characters_count: int = len(FREQUENCIES[language])

    large_alphabet: bool = target_language_characters_count > 26
    large_alphabet_threshold: float = target_language_characters_count / 3

    expected_projection_ratio: float = (
        target_language_characters_count / ordered_characters_count
    )

    # Single pass: characters present in the language vocabulary, as
    # (language rank, popularity rank) pairs. The scoring below only ever
    # needs ranks, never the characters themselves.
    common_lr: list[int] = []
    common_orr: list[int] = []
    for popularity_rank, character in enumerate(ordered_characters):
        language_rank = lang_rank.get(character)
        if language_rank is not None:
            common_lr.append(language_rank)
            common_orr.append(popularity_rank)

    for character_rank_in_language, character_rank in zip(common_lr, common_orr):
        character_rank_projection: int = int(character_rank * expected_projection_ratio)

        if (
            not large_alphabet
            and abs(character_rank_projection - character_rank_in_language) > 4
        ):
            continue

        if (
            large_alphabet
            and abs(character_rank_projection - character_rank_in_language)
            < large_alphabet_threshold
        ):
            character_approved_count += 1
            continue

        if character_rank_in_language == 0:
            # before_match_count is structurally 0 here (no pair can have a
            # smaller language rank): the historic "before <= 4" acceptance
            # always holds. (The symmetric "after_len == 0" case is
            # impossible: language ranks are strictly below the language
            # character count, hence after_len >= 1.)
            character_approved_count += 1
            continue

        after_len: int = target_language_characters_count - character_rank_in_language

        # Count how many characters appear "before" in both orderings, and
        # how many appear "at or after" in both orderings. Both counts grow
        # monotonically and the approval thresholds
        # (before / rank >= 0.4 or after / after_len >= 0.4) are known
        # upfront, expressed below as exact integer comparisons: exit as
        # soon as one is crossed.
        before_match_count: int = 0
        after_match_count: int = 0

        for lr_i, orr_i in zip(common_lr, common_orr):
            if lr_i < character_rank_in_language:
                if orr_i < character_rank:
                    before_match_count += 1
                    if 5 * before_match_count >= 2 * character_rank_in_language:
                        character_approved_count += 1
                        break
            else:
                if orr_i >= character_rank:
                    after_match_count += 1
                    if 5 * after_match_count >= 2 * after_len:
                        character_approved_count += 1
                        break

    return character_approved_count / len(ordered_characters)


def alpha_unicode_split(decoded_sequence: str) -> list[str]:
    """
    Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
    Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
    One containing the latin letters and the other hebrew.
    """
    layers: dict[str, list[str]] = {}

    # Fast path: track single-layer key to skip dict iteration for single-script text.
    single_layer_key: str | None = None
    multi_layer: bool = False

    # Cache the last character_range and its resolved layer to avoid repeated
    # is_suspiciously_successive_range calls for consecutive same-range chars.
    prev_character_range: str | None = None
    prev_layer_target: str | None = None

    for character in decoded_sequence:
        # Reuse the per-codepoint CharInfo cache: info.alpha and info.range
        # are computed with the very same str.isalpha() / unicode_range()
        # calls this loop historically made per character occurrence.
        codepoint: int = ord(character)
        if codepoint < 128:
            info = _ASCII_CHAR_INFO[codepoint]
        else:
            info = _char_info(character)

        if not info.alpha:
            continue

        character_range: str | None = info.range

        if character_range is None:
            continue

        # Fast path: same range as previous character → reuse cached layer target.
        if character_range == prev_character_range:
            if prev_layer_target is not None:
                layers[prev_layer_target].append(character)
            continue

        layer_target_range: str | None = None

        if multi_layer:
            for discovered_range in layers:
                if not is_suspiciously_successive_range(
                    discovered_range, character_range
                ):
                    layer_target_range = discovered_range
                    break
        elif single_layer_key is not None:
            if not is_suspiciously_successive_range(single_layer_key, character_range):
                layer_target_range = single_layer_key

        if layer_target_range is None:
            layer_target_range = character_range

        if layer_target_range not in layers:
            layers[layer_target_range] = []
            if single_layer_key is None:
                single_layer_key = layer_target_range
            else:
                multi_layer = True

        layers[layer_target_range].append(character)

        # Cache for next iteration
        prev_character_range = character_range
        prev_layer_target = layer_target_range

    return ["".join(chars).lower() for chars in layers.values()]


def merge_coherence_ratios(results: list[CoherenceMatches]) -> CoherenceMatches:
    """
    This function merge results previously given by the function coherence_ratio.
    The return type is the same as coherence_ratio.
    """
    per_language_ratios: dict[str, list[float]] = {}
    for result in results:
        for sub_result in result:
            language, ratio = sub_result
            if language not in per_language_ratios:
                per_language_ratios[language] = [ratio]
                continue
            per_language_ratios[language].append(ratio)

    merge = [
        (
            language,
            round(
                sum(per_language_ratios[language]) / len(per_language_ratios[language]),
                4,
            ),
        )
        for language in per_language_ratios
    ]

    return sorted(merge, key=lambda x: x[1], reverse=True)


def filter_alt_coherence_matches(results: CoherenceMatches) -> CoherenceMatches:
    """
    We shall NOT return "English—" in CoherenceMatches because it is an alternative
    of "English". This function only keeps the best match and remove the em-dash in it.
    """
    index_results: dict[str, list[float]] = dict()

    for result in results:
        language, ratio = result
        no_em_name: str = language.replace("—", "")

        if no_em_name not in index_results:
            index_results[no_em_name] = []

        index_results[no_em_name].append(ratio)

    if any(len(index_results[e]) > 1 for e in index_results):
        filtered_results: CoherenceMatches = []

        for language in index_results:
            filtered_results.append((language, max(index_results[language])))

        return filtered_results

    return results


def coherence_ratio(
    decoded_sequence: str, threshold: float = 0.1, lg_inclusion: str | None = None
) -> CoherenceMatches:
    """
    Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
    A layer = Character extraction by alphabets/ranges.
    """

    results: list[tuple[str, float]] = []
    ignore_non_latin: bool = False

    sufficient_match_count: int = 0

    lg_inclusion_list = lg_inclusion.split(",") if lg_inclusion is not None else []
    if "Latin Based" in lg_inclusion_list:
        ignore_non_latin = True
        lg_inclusion_list.remove("Latin Based")

    for layer in alpha_unicode_split(decoded_sequence):
        # Native counting + stable sort reproduce Counter.most_common()
        # ordering exactly (ties keep first-appearance order) without the
        # interpreted Counter machinery in the compiled hot path.
        char_counts: dict[str, int] = {}
        for layer_character in layer:
            char_counts[layer_character] = char_counts.get(layer_character, 0) + 1

        character_count: int = len(layer)

        if character_count <= TOO_SMALL_SEQUENCE:
            continue

        popular_character_ordered: list[str] = [
            item[0]
            for item in sorted(
                char_counts.items(), key=lambda item: item[1], reverse=True
            )
        ]

        for language in lg_inclusion_list or alphabet_languages(
            popular_character_ordered, ignore_non_latin
        ):
            ratio: float = characters_popularity_compare(
                language, popular_character_ordered
            )

            if ratio < threshold:
                continue
            elif ratio >= 0.8:
                sufficient_match_count += 1

            results.append((language, round(ratio, 4)))

            if sufficient_match_count >= 3:
                break

    return sorted(
        filter_alt_coherence_matches(results), key=lambda x: x[1], reverse=True
    )
<|endoftext|>from __future__ import annotations

import argparse
import sys
import typing
from os.path import abspath, basename, dirname, join, realpath
from platform import python_version
from unicodedata import unidata_version

import charset_normalizer.md as md_module
from charset_normalizer import from_fp
from charset_normalizer.models import CliDetectionResult
from charset_normalizer.version import __version__


def query_yes_no(question: str, default: str = "yes") -> bool:  # Defensive:
    """Ask a yes/no question via input() and return the answer as a bool."""
    prompt = " [Y/n] " if default == "yes" else " [y/N] "

    while True:
        choice = input(question + prompt).strip().lower()
        if not choice:
            return default == "yes"
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("Please respond with 'y' or 'n'.")


class FileType:
    """Factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
        - encoding -- The file's encoding. Accepts the same values as the
            builtin open() function.
        - errors -- A string indicating how encoding and decoding errors are to
            be handled. Accepts the same value as the builtin open() function.

    Backported from CPython 3.12
    """

    def __init__(
        self,
        mode: str = "r",
        bufsize: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
    ):
        self._mode = mode
        self._bufsize = bufsize
        self._encoding = encoding
        self._errors = errors

    def __call__(self, string: str) -> typing.IO:  # type: ignore[type-arg]
        # the special argument "-" means sys.std{in,out}
        if string == "-":
            if "r" in self._mode:
                return sys.stdin.buffer if "b" in self._mode else sys.stdin
            elif any(c in self._mode for c in "wax"):
                return sys.stdout.buffer if "b" in self._mode else sys.stdout
            else:
                msg = f'argument "-" with mode {self._mode}'
                raise ValueError(msg)

        # all other arguments are used as file names
        try:
            return open(string, self._mode, self._bufsize, self._encoding, self._errors)
        except OSError as e:
            message = f"can't open '{string}': {e}"
            raise argparse.ArgumentTypeError(message)

    def __repr__(self) -> str:
        args = self._mode, self._bufsize
        kwargs = [("encoding", self._encoding), ("errors", self._errors)]
        args_str = ", ".join(
            [repr(arg) for arg in args if arg != -1]
            + [f"{kw}={arg!r}" for kw, arg in kwargs if arg is not None]
        )
        return f"{type(self).__name__}({args_str})"


def cli_detect(argv: list[str] | None = None) -> int:
    """
    CLI assistant using ARGV and ArgumentParser
    :param argv:
    :return: 0 if everything is fine, anything else equal trouble
    """
    parser = argparse.ArgumentParser(
        description="The Real First Universal Charset Detector. "
        "Discover originating encoding used on text file. "
        "Normalize text to unicode."
    )

    parser.add_argument(
        "files", type=FileType("rb"), nargs="+", help="File(s) to be analysed"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Display complementary information about file if any. "
        "Stdout will contain logs about the detection process.",
    )
    parser.add_argument(
        "-a",
        "--with-alternative",
        action="store_true",
        default=False,
        dest="alternatives",
        help="Output complementary possibilities if any. Top-level JSON WILL be a list.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        default=False,
        dest="normalize",
        help="Permit to normalize input file. If not set, program does not write anything.",
    )
    parser.add_argument(
        "-m",
        "--minimal",
        action="store_true",
        default=False,
        dest="minimal",
        help="Only output the charset detected to STDOUT. Disabling JSON output.",
    )
    parser.add_argument(
        "-r",
        "--replace",
        action="store_true",
        default=False,
        dest="replace",
        help="Replace file when trying to normalize it instead of creating a new one.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        dest="force",
        help="Replace file without asking if you are sure, use this flag with caution.",
    )
    parser.add_argument(
        "-i",
        "--no-preemptive",
        action="store_true",
        default=False,
        dest="no_preemptive",
        help="Disable looking at a charset declaration to hint the detector.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store",
        default=0.2,
        type=float,
        dest="threshold",
        help="Define a custom maximum amount of noise allowed in decoded content. 0. <= noise <= 1.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Charset-Normalizer {} - Python {} - Unicode {} - SpeedUp {}".format(
            __version__,
            python_version(),
            unidata_version,
            "OFF" if md_module.__file__.lower().endswith(".py") else "ON",
        ),
        help="Show version information and exit.",
    )

    args = parser.parse_args(argv)

    if args.replace is True and args.normalize is False:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("Use --replace in addition of --normalize only.", file=sys.stderr)
        return 1

    if args.force is True and args.replace is False:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("Use --force in addition of --replace only.", file=sys.stderr)
        return 1

    if args.threshold < 0.0 or args.threshold > 1.0:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("--threshold VALUE should be between 0. AND 1.", file=sys.stderr)
        return 1

    x_ = []

    for my_file in args.files:
        matches = from_fp(
            my_file,
            threshold=args.threshold,
            explain=args.verbose,
            preemptive_behaviour=args.no_preemptive is False,
        )

        best_guess = matches.best()

        if best_guess is None:
            print(
                'Unable to identify originating encoding for "{}". {}'.format(
                    my_file.name,
                    (
                        "Maybe try increasing maximum amount of chaos."
                        if args.threshold < 1.0
                        else ""
                    ),
                ),
                file=sys.stderr,
            )
            x_.append(
                CliDetectionResult(
                    abspath(my_file.name),
                    None,
                    [],
                    [],
                    "Unknown",
                    [],
                    False,
                    1.0,
                    0.0,
                    None,
                    True,
                )
            )
        else:
            cli_result = CliDetectionResult(
                abspath(my_file.name),
                best_guess.encoding,
                best_guess.encoding_aliases,
                [
                    cp
                    for cp in best_guess.could_be_from_charset
                    if cp != best_guess.encoding
                ],
                best_guess.language,
                best_guess.alphabets,
                best_guess.bom,
                best_guess.percent_chaos,
                best_guess.percent_coherence,
                None,
                True,
            )
            x_.append(cli_result)

            if len(matches) > 1 and args.alternatives:
                for el in matches:
                    if el != best_guess:
                        x_.append(
                            CliDetectionResult(
                                abspath(my_file.name),
                                el.encoding,
                                el.encoding_aliases,
                                [
                                    cp
                                    for cp in el.could_be_from_charset
                                    if cp != el.encoding
                                ],
                                el.language,
                                el.alphabets,
                                el.bom,
                                el.percent_chaos,
                                el.percent_coherence,
                                None,
                                False,
                            )
                        )

            if args.normalize is True:
                if best_guess.encoding.startswith("utf") is True:
                    print(
                        '"{}" file does not need to be normalized, as it already came from unicode.'.format(
                            my_file.name
                        ),
                        file=sys.stderr,
                    )
                    if my_file.closed is False:
                        my_file.close()
                    continue

                dir_path = dirname(realpath(my_file.name))
                file_name = basename(realpath(my_file.name))

                o_: list[str] = file_name.split(".")

                if args.replace is False:
                    o_.insert(-1, best_guess.encoding)
                    if my_file.closed is False:
                        my_file.close()
                elif (
                    args.force is False
                    and query_yes_no(
                        'Are you sure to normalize "{}" by replacing it ?'.format(
                            my_file.name
                        ),
                        "no",
                    )
                    is False
                ):
                    if my_file.closed is False:
                        my_file.close()
                    continue

                try:
                    cli_result.unicode_path = join(dir_path, ".".join(o_))

                    with open(cli_result.unicode_path, "wb") as fp:
                        fp.write(best_guess.output())
                except OSError as e:  # Defensive:
                    print(str(e), file=sys.stderr)
                    if my_file.closed is False:
                        my_file.close()
                    return 2

        if my_file.closed is False:
            my_file.close()

    if args.minimal is False:
        from json import dumps

        print(
            dumps(
                [el.__dict__ for el in x_] if len(x_) > 1 else x_[0].__dict__,
                ensure_ascii=True,
                indent=4,
            )
        )
    else:
        for my_file in args.files:
            print(
                ", ".join(
                    [
                        el.encoding or "undefined"
                        for el in x_
                        if el.path == abspath(my_file.name)
                    ]
                )
            )

    return 0


if __name__ == "__main__":  # Defensive:
    cli_detect()
<|endoftext|># High level events that make up HTTP/1.1 conversations. Loosely inspired by
# the corresponding events in hyper-h2:
#
#     http://python-hyper.org/h2/en/stable/api.html#events
#
# Don't subclass these. Stuff will break.

import re
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import bytesify, LocalProtocolError, validate

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = [
    "Event",
    "Request",
    "InformationalResponse",
    "Response",
    "Data",
    "EndOfMessage",
    "ConnectionClosed",
]

method_re = re.compile(method.encode("ascii"))
request_target_re = re.compile(request_target.encode("ascii"))


class Event(ABC):
    """
    Base class for h11 events.
    """

    __slots__ = ()


@dataclass(init=False, frozen=True)
class Request(Event):
    """The beginning of an HTTP request.

    Fields:

    .. attribute:: method

       An HTTP method, e.g. ``b"GET"`` or ``b"POST"``. Always a byte
       string. :term:`Bytes-like objects <bytes-like object>` and native
       strings containing only ascii characters will be automatically
       converted to byte strings.

    .. attribute:: target

       The target of an HTTP request, e.g. ``b"/index.html"``, or one of the
       more exotic formats described in `RFC 7320, section 5.3
       <https://tools.ietf.org/html/rfc7230#section-5.3>`_. Always a byte
       string. :term:`Bytes-like objects <bytes-like object>` and native
       strings containing only ascii characters will be automatically
       converted to byte strings.

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    """

    __slots__ = ("method", "headers", "target", "http_version")

    method: bytes
    headers: Headers
    target: bytes
    http_version: bytes

    def __init__(
        self,
        *,
        method: Union[bytes, str],
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        target: Union[bytes, str],
        http_version: Union[bytes, str] = b"1.1",
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(headers, Headers):
            object.__setattr__(self, "headers", headers)
        else:
            object.__setattr__(
                self, "headers", normalize_and_validate(headers, _parsed=_parsed)
            )
        if not _parsed:
            object.__setattr__(self, "method", bytesify(method))
            object.__setattr__(self, "target", bytesify(target))
            object.__setattr__(self, "http_version", bytesify(http_version))
        else:
            object.__setattr__(self, "method", method)
            object.__setattr__(self, "target", target)
            object.__setattr__(self, "http_version", http_version)

        # "A server MUST respond with a 400 (Bad Request) status code to any
        # HTTP/1.1 request message that lacks a Host header field and to any
        # request message that contains more than one Host header field or a
        # Host header field with an invalid field-value."
        # -- https://tools.ietf.org/html/rfc7230#section-5.4
        host_count = 0
        for name, value in self.headers:
            if name == b"host":
                host_count += 1
        if self.http_version == b"1.1" and host_count == 0:
            raise LocalProtocolError("Missing mandatory Host: header")
        if host_count > 1:
            raise LocalProtocolError("Found multiple Host: headers")

        validate(method_re, self.method, "Illegal method characters")
        validate(request_target_re, self.target, "Illegal target characters")

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class _ResponseBase(Event):
    __slots__ = ("headers", "http_version", "reason", "status_code")

    headers: Headers
    http_version: bytes
    reason: bytes
    status_code: int

    def __init__(
        self,
        *,
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        status_code: int,
        http_version: Union[bytes, str] = b"1.1",
        reason: Union[bytes, str] = b"",
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(headers, Headers):
            object.__setattr__(self, "headers", headers)
        else:
            object.__setattr__(
                self, "headers", normalize_and_validate(headers, _parsed=_parsed)
            )
        if not _parsed:
            object.__setattr__(self, "reason", bytesify(reason))
            object.__setattr__(self, "http_version", bytesify(http_version))
            if not isinstance(status_code, int):
                raise LocalProtocolError("status code must be integer")
            # Because IntEnum objects are instances of int, but aren't
            # duck-compatible (sigh), see gh-72.
            object.__setattr__(self, "status_code", int(status_code))
        else:
            object.__setattr__(self, "reason", reason)
            object.__setattr__(self, "http_version", http_version)
            object.__setattr__(self, "status_code", status_code)

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class InformationalResponse(_ResponseBase):
    """An HTTP informational response.

    Fields:

    .. attribute:: status_code

       The status code of this response, as an integer. For an
       :class:`InformationalResponse`, this is always in the range [100,
       200).

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for
       details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    .. attribute:: reason

       The reason phrase of this response, as a byte string. For example:
       ``b"OK"``, or ``b"Not Found"``.

    """

    def __post_init__(self) -> None:
        if not (100 <= self.status_code < 200):
            raise LocalProtocolError(
                "InformationalResponse status_code should be in range "
                "[100, 200), not {}".format(self.status_code)
            )

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Response(_ResponseBase):
    """The beginning of an HTTP response.

    Fields:

    .. attribute:: status_code

       The status code of this response, as an integer. For an
       :class:`Response`, this is always in the range [200,
       1000).

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    .. attribute:: reason

       The reason phrase of this response, as a byte string. For example:
       ``b"OK"``, or ``b"Not Found"``.

    """

    def __post_init__(self) -> None:
        if not (200 <= self.status_code < 1000):
            raise LocalProtocolError(
                "Response status_code should be in range [200, 1000), not {}".format(
                    self.status_code
                )
            )

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Data(Event):
    """Part of an HTTP message body.

    Fields:

    .. attribute:: data

       A :term:`bytes-like object` containing part of a message body. Or, if
       using the ``combine=False`` argument to :meth:`Connection.send`, then
       any object that your socket writing code knows what to do with, and for
       which calling :func:`len` returns the number of bytes that will be
       written -- see :ref:`sendfile` for details.

    .. attribute:: chunk_start

       A marker that indicates whether this data object is from the start of a
       chunked transfer encoding chunk. This field is ignored when when a Data
       event is provided to :meth:`Connection.send`: it is only valid on
       events emitted from :meth:`Connection.next_event`. You probably
       shouldn't use this attribute at all; see
       :ref:`chunk-delimiters-are-bad` for details.

    .. attribute:: chunk_end

       A marker that indicates whether this data object is the last for a
       given chunked transfer encoding chunk. This field is ignored when when
       a Data event is provided to :meth:`Connection.send`: it is only valid
       on events emitted from :meth:`Connection.next_event`. You probably
       shouldn't use this attribute at all; see
       :ref:`chunk-delimiters-are-bad` for details.

    """

    __slots__ = ("data", "chunk_start", "chunk_end")

    data: bytes
    chunk_start: bool
    chunk_end: bool

    def __init__(
        self, data: bytes, chunk_start: bool = False, chunk_end: bool = False
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "chunk_start", chunk_start)
        object.__setattr__(self, "chunk_end", chunk_end)

    # This is an unhashable type.
    __hash__ = None  # type: ignore


# XX FIXME: "A recipient MUST ignore (or consider as an error) any fields that
# are forbidden to be sent in a trailer, since processing them as if they were
# present in the header section might bypass external security filters."
# https://svn.tools.ietf.org/svn/wg/httpbis/specs/rfc7230.html#chunked.trailer.part
# Unfortunately, the list of forbidden fields is long and vague :-/
@dataclass(init=False, frozen=True)
class EndOfMessage(Event):
    """The end of an HTTP message.

    Fields:

    .. attribute:: headers

       Default value: ``[]``

       Any trailing headers attached to this message, represented as a list of
       (name, value) pairs. See :ref:`the header normalization rules
       <headers-format>` for details.

       Must be empty unless ``Transfer-Encoding: chunked`` is in use.

    """

    __slots__ = ("headers",)

    headers: Headers

    def __init__(
        self,
        *,
        headers: Union[
            Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]], None
        ] = None,
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if headers is None:
            headers = Headers([])
        elif not isinstance(headers, Headers):
            headers = normalize_and_validate(headers, _parsed=_parsed)

        object.__setattr__(self, "headers", headers)

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(frozen=True)
class ConnectionClosed(Event):
    """This event indicates that the sender has closed their outgoing
    connection.

    Note that this does not necessarily mean that they can't *receive* further
    data, because TCP connections are composed to two one-way channels which
    can be closed independently. See :ref:`closing` for details.

    No fields.
    """

    pass
<|endoftext|># This contains the main Connection class. Everything in h11 revolves around
# this.
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    overload,
    Tuple,
    Type,
    Union,
)

from ._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
    _SWITCH_CONNECT,
    _SWITCH_UPGRADE,
    CLIENT,
    ConnectionState,
    DONE,
    ERROR,
    MIGHT_SWITCH_PROTOCOL,
    SEND_BODY,
    SERVER,
    SWITCHED_PROTOCOL,
)
from ._util import (  # Import the internal things we need
    LocalProtocolError,
    RemoteProtocolError,
    Sentinel,
)
from ._writers import WRITERS, WritersType

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = ["Connection", "NEED_DATA", "PAUSED"]


class NEED_DATA(Sentinel, metaclass=Sentinel):
    pass


class PAUSED(Sentinel, metaclass=Sentinel):
    pass


# If we ever have this much buffered without it making a complete parseable
# event, we error out. The only time we really buffer is when reading the
# request/response line + headers together, so this is effectively the limit on
# the size of that.
#
# Some precedents for defaults:
# - node.js: 80 * 1024
# - tomcat: 8 * 1024
# - IIS: 16 * 1024
# - Apache: <8 KiB per line>
DEFAULT_MAX_INCOMPLETE_EVENT_SIZE = 16 * 1024


# RFC 7230's rules for connection lifecycles:
# - If either side says they want to close the connection, then the connection
#   must close.
# - HTTP/1.1 defaults to keep-alive unless someone says Connection: close
# - HTTP/1.0 defaults to close unless both sides say Connection: keep-alive
#   (and even this is a mess -- e.g. if you're implementing a proxy then
#   sending Connection: keep-alive is forbidden).
#
# We simplify life by simply not supporting keep-alive with HTTP/1.0 peers. So
# our rule is:
# - If someone says Connection: close, we will close
# - If someone uses HTTP/1.0, we will close.
def _keep_alive(event: Union[Request, Response]) -> bool:
    connection = get_comma_header(event.headers, b"connection")
    if b"close" in connection:
        return False
    if getattr(event, "http_version", b"1.1") < b"1.1":
        return False
    return True


def _body_framing(
    request_method: bytes, event: Union[Request, Response]
) -> Tuple[str, Union[Tuple[()], Tuple[int]]]:
    # Called when we enter SEND_BODY to figure out framing information for
    # this body.
    #
    # These are the only two events that can trigger a SEND_BODY state:
    assert type(event) in (Request, Response)
    # Returns one of:
    #
    #    ("content-length", count)
    #    ("chunked", ())
    #    ("http/1.0", ())
    #
    # which are (lookup key, *args) for constructing body reader/writer
    # objects.
    #
    # Reference: https://tools.ietf.org/html/rfc7230#section-3.3.3
    #
    # Step 1: some responses always have an empty body, regardless of what the
    # headers say.
    if type(event) is Response:
        if (
            event.status_code in (204, 304)
            or request_method == b"HEAD"
            or (request_method == b"CONNECT" and 200 <= event.status_code < 300)
        ):
            return ("content-length", (0,))
        # Section 3.3.3 also lists another case -- responses with status_code
        # < 200. For us these are InformationalResponses, not Responses, so
        # they can't get into this function in the first place.
        assert event.status_code >= 200

    # Step 2: check for Transfer-Encoding (T-E beats C-L):
    transfer_encodings = get_comma_header(event.headers, b"transfer-encoding")
    if transfer_encodings:
        assert transfer_encodings == [b"chunked"]
        return ("chunked", ())

    # Step 3: check for Content-Length
    content_lengths = get_comma_header(event.headers, b"content-length")
    if content_lengths:
        return ("content-length", (int(content_lengths[0]),))

    # Step 4: no applicable headers; fallback/default depends on type
    if type(event) is Request:
        return ("content-length", (0,))
    else:
        return ("http/1.0", ())


################################################################
#
# The main Connection class
#
################################################################


class Connection:
    """An object encapsulating the state of an HTTP connection.

    Args:
        our_role: If you're implementing a client, pass :data:`h11.CLIENT`. If
            you're implementing a server, pass :data:`h11.SERVER`.

        max_incomplete_event_size (int):
            The maximum number of bytes we're willing to buffer of an
            incomplete event. In practice this mostly sets a limit on the
            maximum size of the request/response line + headers. If this is
            exceeded, then :meth:`next_event` will raise
            :exc:`RemoteProtocolError`.

    """

    def __init__(
        self,
        our_role: Type[Sentinel],
        max_incomplete_event_size: int = DEFAULT_MAX_INCOMPLETE_EVENT_SIZE,
    ) -> None:
        self._max_incomplete_event_size = max_incomplete_event_size
        # State and role tracking
        if our_role not in (CLIENT, SERVER):
            raise ValueError(f"expected CLIENT or SERVER, not {our_role!r}")
        self.our_role = our_role
        self.their_role: Type[Sentinel]
        if our_role is CLIENT:
            self.their_role = SERVER
        else:
            self.their_role = CLIENT
        self._cstate = ConnectionState()

        # Callables for converting data->events or vice-versa given the
        # current state
        self._writer = self._get_io_object(self.our_role, None, WRITERS)
        self._reader = self._get_io_object(self.their_role, None, READERS)

        # Holds any unprocessed received data
        self._receive_buffer = ReceiveBuffer()
        # If this is true, then it indicates that the incoming connection was
        # closed *after* the end of whatever's in self._receive_buffer:
        self._receive_buffer_closed = False

        # Extra bits of state that don't fit into the state machine.
        #
        # These two are only used to interpret framing headers for figuring
        # out how to read/write response bodies. their_http_version is also
        # made available as a convenient public API.
        self.their_http_version: Optional[bytes] = None
        self._request_method: Optional[bytes] = None
        # This is pure flow-control and doesn't at all affect the set of legal
        # transitions, so no need to bother ConnectionState with it:
        self.client_is_waiting_for_100_continue = False

    @property
    def states(self) -> Dict[Type[Sentinel], Type[Sentinel]]:
        """A dictionary like::

           {CLIENT: <client state>, SERVER: <server state>}

        See :ref:`state-machine` for details.

        """
        return dict(self._cstate.states)

    @property
    def our_state(self) -> Type[Sentinel]:
        """The current state of whichever role we are playing. See
        :ref:`state-machine` for details.
        """
        return self._cstate.states[self.our_role]

    @property
    def their_state(self) -> Type[Sentinel]:
        """The current state of whichever role we are NOT playing. See
        :ref:`state-machine` for details.
        """
        return self._cstate.states[self.their_role]

    @property
    def they_are_waiting_for_100_continue(self) -> bool:
        return self.their_role is CLIENT and self.client_is_waiting_for_100_continue

    def start_next_cycle(self) -> None:
        """Attempt to reset our connection state for a new request/response
        cycle.

        If both client and server are in :data:`DONE` state, then resets them
        both to :data:`IDLE` state in preparation for a new request/response
        cycle on this same connection. Otherwise, raises a
        :exc:`LocalProtocolError`.

        See :ref:`keepalive-and-pipelining`.

        """
        old_states = dict(self._cstate.states)
        self._cstate.start_next_cycle()
        self._request_method = None
        # self.their_http_version gets left alone, since it presumably lasts
        # beyond a single request/response cycle
        assert not self.client_is_waiting_for_100_continue
        self._respond_to_state_changes(old_states)

    def _process_error(self, role: Type[Sentinel]) -> None:
        old_states = dict(self._cstate.states)
        self._cstate.process_error(role)
        self._respond_to_state_changes(old_states)

    def _server_switch_event(self, event: Event) -> Optional[Type[Sentinel]]:
        if type(event) is InformationalResponse and event.status_code == 101:
            return _SWITCH_UPGRADE
        if type(event) is Response:
            if (
                _SWITCH_CONNECT in self._cstate.pending_switch_proposals
                and 200 <= event.status_code < 300
            ):
                return _SWITCH_CONNECT
        return None

    # All events go through here
    def _process_event(self, role: Type[Sentinel], event: Event) -> None:
        # First, pass the event through the state machine to make sure it
        # succeeds.
        old_states = dict(self._cstate.states)
        if role is CLIENT and type(event) is Request:
            if event.method == b"CONNECT":
                self._cstate.process_client_switch_proposal(_SWITCH_CONNECT)
            if get_comma_header(event.headers, b"upgrade"):
                self._cstate.process_client_switch_proposal(_SWITCH_UPGRADE)
        server_switch_event = None
        if role is SERVER:
            server_switch_event = self._server_switch_event(event)
        self._cstate.process_event(role, type(event), server_switch_event)

        # Then perform the updates triggered by it.

        if type(event) is Request:
            self._request_method = event.method

        if role is self.their_role and type(event) in (
            Request,
            Response,
            InformationalResponse,
        ):
            event = cast(Union[Request, Response, InformationalResponse], event)
            self.their_http_version = event.http_version

        # Keep alive handling
        #
        # RFC 7230 doesn't really say what one should do if Connection: close
        # shows up on a 1xx InformationalResponse. I think the idea is that
        # this is not supposed to happen. In any case, if it does happen, we
        # ignore it.
        if type(event) in (Request, Response) and not _keep_alive(
            cast(Union[Request, Response], event)
        ):
            self._cstate.process_keep_alive_disabled()

        # 100-continue
        if type(event) is Request and has_expect_100_continue(event):
            self.client_is_waiting_for_100_continue = True
        if type(event) in (InformationalResponse, Response):
            self.client_is_waiting_for_100_continue = False
        if role is CLIENT and type(event) in (Data, EndOfMessage):
            self.client_is_waiting_for_100_continue = False

        self._respond_to_state_changes(old_states, event)

    def _get_io_object(
        self,
        role: Type[Sentinel],
        event: Optional[Event],
        io_dict: Union[ReadersType, WritersType],
    ) -> Optional[Callable[..., Any]]:
        # event may be None; it's only used when entering SEND_BODY
        state = self._cstate.states[role]
        if state is SEND_BODY:
            # Special case: the io_dict has a dict of reader/writer factories
            # that depend on the request/response framing.
            framing_type, args = _body_framing(
                cast(bytes, self._request_method), cast(Union[Request, Response], event)
            )
            return io_dict[SEND_BODY][framing_type](*args)  # type: ignore[index]
        else:
            # General case: the io_dict just has the appropriate reader/writer
            # for this state
            return io_dict.get((role, state))  # type: ignore[return-value]

    # This must be called after any action that might have caused
    # self._cstate.states to change.
    def _respond_to_state_changes(
        self,
        old_states: Dict[Type[Sentinel], Type[Sentinel]],
        event: Optional[Event] = None,
    ) -> None:
        # Update reader/writer
        if self.our_state != old_states[self.our_role]:
            self._writer = self._get_io_object(self.our_role, event, WRITERS)
        if self.their_state != old_states[self.their_role]:
            self._reader = self._get_io_object(self.their_role, event, READERS)

    @property
    def trailing_data(self) -> Tuple[bytes, bool]:
        """Data that has been received, but not yet processed, represented as
        a tuple with two elements, where the first is a byte-string containing
        the unprocessed data itself, and the second is a bool that is True if
        the receive connection was closed.

        See :ref:`switching-protocols` for discussion of why you'd want this.
        """
        return (bytes(self._receive_buffer), self._receive_buffer_closed)

    def receive_data(self, data: bytes) -> None:
        """Add data to our internal receive buffer.

        This does not actually do any processing on the data, just stores
        it. To trigger processing, you have to call :meth:`next_event`.

        Args:
            data (:term:`bytes-like object`):
                The new data that was just received.

                Special case: If *data* is an empty byte-string like ``b""``,
                then this indicates that the remote side has closed the
                connection (end of file). Normally this is convenient, because
                standard Python APIs like :meth:`file.read` or
                :meth:`socket.recv` use ``b""`` to indicate end-of-file, while
                other failures to read are indicated using other mechanisms
                like raising :exc:`TimeoutError`. When using such an API you
                can just blindly pass through whatever you get from ``read``
                to :meth:`receive_data`, and everything will work.

                But, if you have an API where reading an empty string is a
                valid non-EOF condition, then you need to be aware of this and
                make sure to check for such strings and avoid passing them to
                :meth:`receive_data`.

        Returns:
            Nothing, but after calling this you should call :meth:`next_event`
            to parse the newly received data.

        Raises:
            RuntimeError:
                Raised if you pass an empty *data*, indicating EOF, and then
                pass a non-empty *data*, indicating more data that somehow
                arrived after the EOF.

                (Calling ``receive_data(b"")`` multiple times is fine,
                and equivalent to calling it once.)

        """
        if data:
            if self._receive_buffer_closed:
                raise RuntimeError("received close, then received more data?")
            self._receive_buffer += data
        else:
            self._receive_buffer_closed = True

    def _extract_next_receive_event(
        self,
    ) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        state = self.their_state
        # We don't pause immediately when they enter DONE, because even in
        # DONE state we can still process a ConnectionClosed() event. But
        # if we have data in our buffer, then we definitely aren't getting
        # a ConnectionClosed() immediately and we need to pause.
        if state is DONE and self._receive_buffer:
            return PAUSED
        if state is MIGHT_SWITCH_PROTOCOL or state is SWITCHED_PROTOCOL:
            return PAUSED
        assert self._reader is not None
        event = self._reader(self._receive_buffer)
        if event is None:
            if not self._receive_buffer and self._receive_buffer_closed:
                # In some unusual cases (basically just HTTP/1.0 bodies), EOF
                # triggers an actual protocol event; in that case, we want to
                # return that event, and then the state will change and we'll
                # get called again to generate the actual ConnectionClosed().
                if hasattr(self._reader, "read_eof"):
                    event = self._reader.read_eof()
                else:
                    event = ConnectionClosed()
        if event is None:
            event = NEED_DATA
        return event  # type: ignore[no-any-return]

    def next_event(self) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        """Parse the next event out of our receive buffer, update our internal
        state, and return it.

        This is a mutating operation -- think of it like calling :func:`next`
        on an iterator.

        Returns:
            : One of three things:

            1) An event object -- see :ref:`events`.

            2) The special constant :data:`NEED_DATA`, which indicates that
               you need to read more data from your socket and pass it to
               :meth:`receive_data` before this method will be able to return
               any more events.

            3) The special constant :data:`PAUSED`, which indicates that we
               are not in a state where we can process incoming data (usually
               because the peer has finished their part of the current
               request/response cycle, and you have not yet called
               :meth:`start_next_cycle`). See :ref:`flow-control` for details.

        Raises:
            RemoteProtocolError:
                The peer has misbehaved. You should close the connection
                (possibly after sending some kind of 4xx response).

        Once this method returns :class:`ConnectionClosed` once, then all
        subsequent calls will also return :class:`ConnectionClosed`.

        If this method raises any exception besides :exc:`RemoteProtocolError`
        then that's a bug -- if it happens please file a bug report!

        If this method raises any exception then it also sets
        :attr:`Connection.their_state` to :data:`ERROR` -- see
        :ref:`error-handling` for discussion.

        """

        if self.their_state is ERROR:
            raise RemoteProtocolError("Can't receive data when peer state is ERROR")
        try:
            event = self._extract_next_receive_event()
            if event not in [NEED_DATA, PAUSED]:
                self._process_event(self.their_role, cast(Event, event))
            if event is NEED_DATA:
                if len(self._receive_buffer) > self._max_incomplete_event_size:
                    # 431 is "Request header fields too large" which is pretty
                    # much the only situation where we can get here
                    raise RemoteProtocolError(
                        "Receive buffer too long", error_status_hint=431
                    )
                if self._receive_buffer_closed:
                    # We're still trying to complete some event, but that's
                    # never going to happen because no more data is coming
                    raise RemoteProtocolError("peer unexpectedly closed connection")
            return event
        except BaseException as exc:
            self._process_error(self.their_role)
            if isinstance(exc, LocalProtocolError):
                exc._reraise_as_remote_protocol_error()
            else:
                raise

    @overload
    def send(self, event: ConnectionClosed) -> None:
        ...

    @overload
    def send(
        self, event: Union[Request, InformationalResponse, Response, Data, EndOfMessage]
    ) -> bytes:
        ...

    @overload
    def send(self, event: Event) -> Optional[bytes]:
        ...

    def send(self, event: Event) -> Optional[bytes]:
        """Convert a high-level event into bytes that can be sent to the peer,
        while updating our internal state machine.

        Args:
            event: The :ref:`event <events>` to send.

        Returns:
            If ``type(event) is ConnectionClosed``, then returns
            ``None``. Otherwise, returns a :term:`bytes-like object`.

        Raises:
            LocalProtocolError:
                Sending this event at this time would violate our
                understanding of the HTTP/1.1 protocol.

        If this method raises any exception then it also sets
        :attr:`Connection.our_state` to :data:`ERROR` -- see
        :ref:`error-handling` for discussion.

        """
        data_list = self.send_with_data_passthrough(event)
        if data_list is None:
            return None
        else:
            return b"".join(data_list)

    def send_with_data_passthrough(self, event: Event) -> Optional[List[bytes]]:
        """Identical to :meth:`send`, except that in situations where
        :meth:`send` returns a single :term:`bytes-like object`, this instead
        returns a list of them -- and when sending a :class:`Data` event, this
        list is guaranteed to contain the exact object you passed in as
        :attr:`Data.data`. See :ref:`sendfile` for discussion.

        """
        if self.our_state is ERROR:
            raise LocalProtocolError("Can't send data when our state is ERROR")
        try:
            if type(event) is Response:
                event = self._clean_up_response_headers_for_sending(event)
            # We want to call _process_event before calling the writer,
            # because if someone tries to do something invalid then this will
            # give a sensible error message, while our writers all just assume
            # they will only receive valid events. But, _process_event might
            # change self._writer. So we have to do a little dance:
            writer = self._writer
            self._process_event(self.our_role, event)
            if type(event) is ConnectionClosed:
                return None
            else:
                # In any situation where writer is None, process_event should
                # have raised ProtocolError
                assert writer is not None
                data_list: List[bytes] = []
                writer(event, data_list.append)
                return data_list
        except:
            self._process_error(self.our_role)
            raise

    def send_failed(self) -> None:
        """Notify the state machine that we failed to send the data it gave
        us.

        This causes :attr:`Connection.our_state` to immediately become
        :data:`ERROR` -- see :ref:`error-handling` for discussion.

        """
        self._process_error(self.our_role)

    # When sending a Response, we take responsibility for a few things:
    #
    # - Sometimes you MUST set Connection: close. We take care of those
    #   times. (You can also set it yourself if you want, and if you do then
    #   we'll respect that and close the connection at the right time. But you
    #   don't have to worry about that unless you want to.)
    #
    # - The user has to set Content-Length if they want it. Otherwise, for
    #   responses that have bodies (e.g. not HEAD), then we will automatically
    #   select the right mechanism for streaming a body of unknown length,
    #   which depends on depending on the peer's HTTP version.
    #
    # This function's *only* responsibility is making sure headers are set up
    # right -- everything downstream just looks at the headers. There are no
    # side channels.
    def _clean_up_response_headers_for_sending(self, response: Response) -> Response:
        assert type(response) is Response

        headers = response.headers
        need_close = False

        # HEAD requests need some special handling: they always act like they
        # have Content-Length: 0, and that's how _body_framing treats
        # them. But their headers are supposed to match what we would send if
        # the request was a GET. (Technically there is one deviation allowed:
        # we're allowed to leave out the framing headers -- see
        # https://tools.ietf.org/html/rfc7231#section-4.3.2 . But it's just as
        # easy to get them right.)
        method_for_choosing_headers = cast(bytes, self._request_method)
        if method_for_choosing_headers == b"HEAD":
            method_for_choosing_headers = b"GET"
        framing_type, _ = _body_framing(method_for_choosing_headers, response)
        if framing_type in ("chunked", "http/1.0"):
            # This response has a body of unknown length.
            # If our peer is HTTP/1.1, we use Transfer-Encoding: chunked
            # If our peer is HTTP/1.0, we use no framing headers, and close the
            # connection afterwards.
            #
            # Make sure to clear Content-Length (in principle user could have
            # set both and then we ignored Content-Length b/c
            # Transfer-Encoding overwrote it -- this would be naughty of them,
            # but the HTTP spec says that if our peer does this then we have
            # to fix it instead of erroring out, so we'll accord the user the
            # same respect).
            headers = set_comma_header(headers, b"content-length", [])
            if self.their_http_version is None or self.their_http_version < b"1.1":
                # Either we never got a valid request and are sending back an
                # error (their_http_version is None), so we assume the worst;
                # or else we did get a valid HTTP/1.0 request, so we know that
                # they don't understand chunked encoding.
                headers = set_comma_header(headers, b"transfer-encoding", [])
                # This is actually redundant ATM, since currently we
                # unconditionally disable keep-alive when talking to HTTP/1.0
                # peers. But let's be defensive just in case we add
                # Connection: keep-alive support later:
                if self._request_method != b"HEAD":
                    need_close = True
            else:
                headers = set_comma_header(headers, b"transfer-encoding", [b"chunked"])

        if not self._cstate.keep_alive or need_close:
            # Make sure Connection: close is set
            connection = set(get_comma_header(headers, b"connection"))
            connection.discard(b"keep-alive")
            connection.add(b"close")
            headers = set_comma_header(headers, b"connection", sorted(connection))

        return Response(
            headers=headers,
            status_code=response.status_code,
            http_version=response.http_version,
            reason=response.reason,
        )
<|endoftext|>################################################################
# The core state machine
################################################################
#
# Rule 1: everything that affects the state machine and state transitions must
# live here in this file. As much as possible goes into the table-based
# representation, but for the bits that don't quite fit, the actual code and
# state must nonetheless live here.
#
# Rule 2: this file does not know about what role we're playing; it only knows
# about HTTP request/response cycles in the abstract. This ensures that we
# don't cheat and apply different rules to local and remote parties.
#
#
# Theory of operation
# ===================
#
# Possibly the simplest way to think about this is that we actually have 5
# different state machines here. Yes, 5. These are:
#
# 1) The client state, with its complicated automaton (see the docs)
# 2) The server state, with its complicated automaton (see the docs)
# 3) The keep-alive state, with possible states {True, False}
# 4) The SWITCH_CONNECT state, with possible states {False, True}
# 5) The SWITCH_UPGRADE state, with possible states {False, True}
#
# For (3)-(5), the first state listed is the initial state.
#
# (1)-(3) are stored explicitly in member variables. The last
# two are stored implicitly in the pending_switch_proposals set as:
#   (state of 4) == (_SWITCH_CONNECT in pending_switch_proposals)
#   (state of 5) == (_SWITCH_UPGRADE in pending_switch_proposals)
#
# And each of these machines has two different kinds of transitions:
#
# a) Event-triggered
# b) State-triggered
#
# Event triggered is the obvious thing that you'd think it is: some event
# happens, and if it's the right event at the right time then a transition
# happens. But there are somewhat complicated rules for which machines can
# "see" which events. (As a rule of thumb, if a machine "sees" an event, this
# means two things: the event can affect the machine, and if the machine is
# not in a state where it expects that event then it's an error.) These rules
# are:
#
# 1) The client machine sees all h11.events objects emitted by the client.
#
# 2) The server machine sees all h11.events objects emitted by the server.
#
#    It also sees the client's Request event.
#
#    And sometimes, server events are annotated with a _SWITCH_* event. For
#    example, we can have a (Response, _SWITCH_CONNECT) event, which is
#    different from a regular Response event.
#
# 3) The keep-alive machine sees the process_keep_alive_disabled() event
#    (which is derived from Request/Response events), and this event
#    transitions it from True -> False, or from False -> False. There's no way
#    to transition back.
#
# 4&5) The _SWITCH_* machines transition from False->True when we get a
#    Request that proposes the relevant type of switch (via
#    process_client_switch_proposals), and they go from True->False when we
#    get a Response that has no _SWITCH_* annotation.
#
# So that's event-triggered transitions.
#
# State-triggered transitions are less standard. What they do here is couple
# the machines together. The way this works is, when certain *joint*
# configurations of states are achieved, then we automatically transition to a
# new *joint* state. So, for example, if we're ever in a joint state with
#
#   client: DONE
#   keep-alive: False
#
# then the client state immediately transitions to:
#
#   client: MUST_CLOSE
#
# This is fundamentally different from an event-based transition, because it
# doesn't matter how we arrived at the {client: DONE, keep-alive: False} state
# -- maybe the client transitioned SEND_BODY -> DONE, or keep-alive
# transitioned True -> False. Either way, once this precondition is satisfied,
# this transition is immediately triggered.
#
# What if two conflicting state-based transitions get enabled at the same
# time?  In practice there's only one case where this arises (client DONE ->
# MIGHT_SWITCH_PROTOCOL versus DONE -> MUST_CLOSE), and we resolve it by
# explicitly prioritizing the DONE -> MIGHT_SWITCH_PROTOCOL transition.
#
# Implementation
# --------------
#
# The event-triggered transitions for the server and client machines are all
# stored explicitly in a table. Ditto for the state-triggered transitions that
# involve just the server and client state.
#
# The transitions for the other machines, and the state-triggered transitions
# that involve the other machines, are written out as explicit Python code.
#
# It'd be nice if there were some cleaner way to do all this. This isn't
# *too* terrible, but I feel like it could probably be better.
#
# WARNING
# -------
#
# The script that generates the state machine diagrams for the docs knows how
# to read out the EVENT_TRIGGERED_TRANSITIONS and STATE_TRIGGERED_TRANSITIONS
# tables. But it can't automatically read the transitions that are written
# directly in Python code. So if you touch those, you need to also update the
# script to keep it in sync!
from typing import cast, Dict, Optional, Set, Tuple, Type, Union

from ._events import *
from ._util import LocalProtocolError, Sentinel

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = [
    "CLIENT",
    "SERVER",
    "IDLE",
    "SEND_RESPONSE",
    "SEND_BODY",
    "DONE",
    "MUST_CLOSE",
    "CLOSED",
    "MIGHT_SWITCH_PROTOCOL",
    "SWITCHED_PROTOCOL",
    "ERROR",
]


class CLIENT(Sentinel, metaclass=Sentinel):
    pass


class SERVER(Sentinel, metaclass=Sentinel):
    pass


# States
class IDLE(Sentinel, metaclass=Sentinel):
    pass


class SEND_RESPONSE(Sentinel, metaclass=Sentinel):
    pass


class SEND_BODY(Sentinel, metaclass=Sentinel):
    pass


class DONE(Sentinel, metaclass=Sentinel):
    pass


class MUST_CLOSE(Sentinel, metaclass=Sentinel):
    pass


class CLOSED(Sentinel, metaclass=Sentinel):
    pass


class ERROR(Sentinel, metaclass=Sentinel):
    pass


# Switch types
class MIGHT_SWITCH_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class SWITCHED_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_UPGRADE(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_CONNECT(Sentinel, metaclass=Sentinel):
    pass


EventTransitionType = Dict[
    Type[Sentinel],
    Dict[
        Type[Sentinel],
        Dict[Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]], Type[Sentinel]],
    ],
]

EVENT_TRIGGERED_TRANSITIONS: EventTransitionType = {
    CLIENT: {
        IDLE: {Request: SEND_BODY, ConnectionClosed: CLOSED},
        SEND_BODY: {Data: SEND_BODY, EndOfMessage: DONE},
        DONE: {ConnectionClosed: CLOSED},
        MUST_CLOSE: {ConnectionClosed: CLOSED},
        CLOSED: {ConnectionClosed: CLOSED},
        MIGHT_SWITCH_PROTOCOL: {},
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
    SERVER: {
        IDLE: {
            ConnectionClosed: CLOSED,
            Response: SEND_BODY,
            # Special case: server sees client Request events, in this form
            (Request, CLIENT): SEND_RESPONSE,
        },
        SEND_RESPONSE: {
            InformationalResponse: SEND_RESPONSE,
            Response: SEND_BODY,
            (InformationalResponse, _SWITCH_UPGRADE): SWITCHED_PROTOCOL,
            (Response, _SWITCH_CONNECT): SWITCHED_PROTOCOL,
        },
        SEND_BODY: {Data: SEND_BODY, EndOfMessage: DONE},
        DONE: {ConnectionClosed: CLOSED},
        MUST_CLOSE: {ConnectionClosed: CLOSED},
        CLOSED: {ConnectionClosed: CLOSED},
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
}

StateTransitionType = Dict[
    Tuple[Type[Sentinel], Type[Sentinel]], Dict[Type[Sentinel], Type[Sentinel]]
]

# NB: there are also some special-case state-triggered transitions hard-coded
# into _fire_state_triggered_transitions below.
STATE_TRIGGERED_TRANSITIONS: StateTransitionType = {
    # (Client state, Server state) -> new states
    # Protocol negotiation
    (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL): {CLIENT: SWITCHED_PROTOCOL},
    # Socket shutdown
    (CLOSED, DONE): {SERVER: MUST_CLOSE},
    (CLOSED, IDLE): {SERVER: MUST_CLOSE},
    (ERROR, DONE): {SERVER: MUST_CLOSE},
    (DONE, CLOSED): {CLIENT: MUST_CLOSE},
    (IDLE, CLOSED): {CLIENT: MUST_CLOSE},
    (DONE, ERROR): {CLIENT: MUST_CLOSE},
}


class ConnectionState:
    def __init__(self) -> None:
        # Extra bits of state that don't quite fit into the state model.

        # If this is False then it enables the automatic DONE -> MUST_CLOSE
        # transition. Don't set this directly; call .keep_alive_disabled()
        self.keep_alive = True

        # This is a subset of {UPGRADE, CONNECT}, containing the proposals
        # made by the client for switching protocols.
        self.pending_switch_proposals: Set[Type[Sentinel]] = set()

        self.states: Dict[Type[Sentinel], Type[Sentinel]] = {CLIENT: IDLE, SERVER: IDLE}

    def process_error(self, role: Type[Sentinel]) -> None:
        self.states[role] = ERROR
        self._fire_state_triggered_transitions()

    def process_keep_alive_disabled(self) -> None:
        self.keep_alive = False
        self._fire_state_triggered_transitions()

    def process_client_switch_proposal(self, switch_event: Type[Sentinel]) -> None:
        self.pending_switch_proposals.add(switch_event)
        self._fire_state_triggered_transitions()

    def process_event(
        self,
        role: Type[Sentinel],
        event_type: Type[Event],
        server_switch_event: Optional[Type[Sentinel]] = None,
    ) -> None:
        _event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]] = event_type
        if server_switch_event is not None:
            assert role is SERVER
            if server_switch_event not in self.pending_switch_proposals:
                raise LocalProtocolError(
                    "Received server _SWITCH_UPGRADE event without a pending proposal"
                )
            _event_type = (event_type, server_switch_event)
        if server_switch_event is None and _event_type is Response:
            self.pending_switch_proposals = set()
        self._fire_event_triggered_transitions(role, _event_type)
        # Special case: the server state does get to see Request
        # events.
        if _event_type is Request:
            assert role is CLIENT
            self._fire_event_triggered_transitions(SERVER, (Request, CLIENT))
        self._fire_state_triggered_transitions()

    def _fire_event_triggered_transitions(
        self,
        role: Type[Sentinel],
        event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]],
    ) -> None:
        state = self.states[role]
        try:
            new_state = EVENT_TRIGGERED_TRANSITIONS[role][state][event_type]
        except KeyError:
            event_type = cast(Type[Event], event_type)
            raise LocalProtocolError(
                "can't handle event type {} when role={} and state={}".format(
                    event_type.__name__, role, self.states[role]
                )
            ) from None
        self.states[role] = new_state

    def _fire_state_triggered_transitions(self) -> None:
        # We apply these rules repeatedly until converging on a fixed point
        while True:
            start_states = dict(self.states)

            # It could happen that both these special-case transitions are
            # enabled at the same time:
            #
            #    DONE -> MIGHT_SWITCH_PROTOCOL
            #    DONE -> MUST_CLOSE
            #
            # For example, this will always be true of a HTTP/1.0 client
            # requesting CONNECT.  If this happens, the protocol switch takes
            # priority. From there the client will either go to
            # SWITCHED_PROTOCOL, in which case it's none of our business when
            # they close the connection, or else the server will deny the
            # request, in which case the client will go back to DONE and then
            # from there to MUST_CLOSE.
            if self.pending_switch_proposals:
                if self.states[CLIENT] is DONE:
                    self.states[CLIENT] = MIGHT_SWITCH_PROTOCOL

            if not self.pending_switch_proposals:
                if self.states[CLIENT] is MIGHT_SWITCH_PROTOCOL:
                    self.states[CLIENT] = DONE

            if not self.keep_alive:
                for role in (CLIENT, SERVER):
                    if self.states[role] is DONE:
                        self.states[role] = MUST_CLOSE

            # Tabular state-triggered transitions
            joint_state = (self.states[CLIENT], self.states[SERVER])
            changes = STATE_TRIGGERED_TRANSITIONS.get(joint_state, {})
            self.states.update(changes)

            if self.states == start_states:
                # Fixed point reached
                return

    def start_next_cycle(self) -> None:
        if self.states != {CLIENT: DONE, SERVER: DONE}:
            raise LocalProtocolError(
                f"not in a reusable state. self.states={self.states}"
            )
        # Can't reach DONE/DONE with any of these active, but still, let's be
        # sure.
        assert self.keep_alive
        assert not self.pending_switch_proposals
        self.states = {CLIENT: IDLE, SERVER: IDLE}
<|endoftext|># Code to read HTTP data
#
# Strategy: each writer takes an event + a write-some-bytes function, which is
# calls.
#
# WRITERS is a dict describing how to pick a reader. It maps states to either:
# - a writer
# - or, for body writers, a dict of framin-dependent writer factories

from typing import Any, Callable, Dict, List, Tuple, Type, Union

from ._events import Data, EndOfMessage, Event, InformationalResponse, Request, Response
from ._headers import Headers
from ._state import CLIENT, IDLE, SEND_BODY, SEND_RESPONSE, SERVER
from ._util import LocalProtocolError, Sentinel

__all__ = ["WRITERS"]

Writer = Callable[[bytes], Any]


def write_headers(headers: Headers, write: Writer) -> None:
    # "Since the Host field-value is critical information for handling a
    # request, a user agent SHOULD generate Host as the first header field
    # following the request-line." - RFC 7230
    raw_items = headers._full_items
    for raw_name, name, value in raw_items:
        if name == b"host":
            write(b"%s: %s\r\n" % (raw_name, value))
    for raw_name, name, value in raw_items:
        if name != b"host":
            write(b"%s: %s\r\n" % (raw_name, value))
    write(b"\r\n")


def write_request(request: Request, write: Writer) -> None:
    if request.http_version != b"1.1":
        raise LocalProtocolError("I only send HTTP/1.1")
    write(b"%s %s HTTP/1.1\r\n" % (request.method, request.target))
    write_headers(request.headers, write)


# Shared between InformationalResponse and Response
def write_any_response(
    response: Union[InformationalResponse, Response], write: Writer
) -> None:
    if response.http_version != b"1.1":
        raise LocalProtocolError("I only send HTTP/1.1")
    status_bytes = str(response.status_code).encode("ascii")
    # We don't bother sending ascii status messages like "OK"; they're
    # optional and ignored by the protocol. (But the space after the numeric
    # status code is mandatory.)
    #
    # XX FIXME: could at least make an effort to pull out the status message
    # from stdlib's http.HTTPStatus table. Or maybe just steal their enums
    # (either by import or copy/paste). We already accept them as status codes
    # since they're of type IntEnum < int.
    write(b"HTTP/1.1 %s %s\r\n" % (status_bytes, response.reason))
    write_headers(response.headers, write)


class BodyWriter:
    def __call__(self, event: Event, write: Writer) -> None:
        if type(event) is Data:
            self.send_data(event.data, write)
        elif type(event) is EndOfMessage:
            self.send_eom(event.headers, write)
        else:  # pragma: no cover
            assert False

    def send_data(self, data: bytes, write: Writer) -> None:
        pass

    def send_eom(self, headers: Headers, write: Writer) -> None:
        pass


#
# These are all careful not to do anything to 'data' except call len(data) and
# write(data). This allows us to transparently pass-through funny objects,
# like placeholder objects referring to files on disk that will be sent via
# sendfile(2).
#
class ContentLengthWriter(BodyWriter):
    def __init__(self, length: int) -> None:
        self._length = length

    def send_data(self, data: bytes, write: Writer) -> None:
        self._length -= len(data)
        if self._length < 0:
            raise LocalProtocolError("Too much data for declared Content-Length")
        write(data)

    def send_eom(self, headers: Headers, write: Writer) -> None:
        if self._length != 0:
            raise LocalProtocolError("Too little data for declared Content-Length")
        if headers:
            raise LocalProtocolError("Content-Length and trailers don't mix")


class ChunkedWriter(BodyWriter):
    def send_data(self, data: bytes, write: Writer) -> None:
        # if we encoded 0-length data in the naive way, it would look like an
        # end-of-message.
        if not data:
            return
        write(b"%x\r\n" % len(data))
        write(data)
        write(b"\r\n")

    def send_eom(self, headers: Headers, write: Writer) -> None:
        write(b"0\r\n")
        write_headers(headers, write)


class Http10Writer(BodyWriter):
    def send_data(self, data: bytes, write: Writer) -> None:
        write(data)

    def send_eom(self, headers: Headers, write: Writer) -> None:
        if headers:
            raise LocalProtocolError("can't send trailers to HTTP/1.0 client")
        # no need to close the socket ourselves, that will be taken care of by
        # Connection: close machinery


WritersType = Dict[
    Union[Tuple[Type[Sentinel], Type[Sentinel]], Type[Sentinel]],
    Union[
        Dict[str, Type[BodyWriter]],
        Callable[[Union[InformationalResponse, Response], Writer], None],
        Callable[[Request, Writer], None],
    ],
]

WRITERS: WritersType = {
    (CLIENT, IDLE): write_request,
    (SERVER, IDLE): write_any_response,
    (SERVER, SEND_RESPONSE): write_any_response,
    SEND_BODY: {
        "chunked": ChunkedWriter,
        "content-length": ContentLengthWriter,
        "http/1.0": Http10Writer,
    },
}
<|endoftext|>import re
from typing import AnyStr, cast, List, overload, Sequence, Tuple, TYPE_CHECKING, Union

from ._abnf import field_name, field_value
from ._util import bytesify, LocalProtocolError, validate

if TYPE_CHECKING:
    from ._events import Request

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

CONTENT_LENGTH_MAX_DIGITS = 20  # allow up to 1 billion TB - 1


# Facts
# -----
#
# Headers are:
#   keys: case-insensitive ascii
#   values: mixture of ascii and raw bytes
#
# "Historically, HTTP has allowed field content with text in the ISO-8859-1
# charset [ISO-8859-1], supporting other charsets only through use of
# [RFC2047] encoding.  In practice, most HTTP header field values use only a
# subset of the US-ASCII charset [USASCII]. Newly defined header fields SHOULD
# limit their field values to US-ASCII octets.  A recipient SHOULD treat other
# octets in field content (obs-text) as opaque data."
# And it deprecates all non-ascii values
#
# Leading/trailing whitespace in header names is forbidden
#
# Values get leading/trailing whitespace stripped
#
# Content-Disposition actually needs to contain unicode semantically; to
# accomplish this it has a terrifically weird way of encoding the filename
# itself as ascii (and even this still has lots of cross-browser
# incompatibilities)
#
# Order is important:
# "a proxy MUST NOT change the order of these field values when forwarding a
# message"
# (and there are several headers where the order indicates a preference)
#
# Multiple occurences of the same header:
# "A sender MUST NOT generate multiple header fields with the same field name
# in a message unless either the entire field value for that header field is
# defined as a comma-separated list [or the header is Set-Cookie which gets a
# special exception]" - RFC 7230. (cookies are in RFC 6265)
#
# So every header aside from Set-Cookie can be merged by b", ".join if it
# occurs repeatedly. But, of course, they can't necessarily be split by
# .split(b","), because quoting.
#
# Given all this mess (case insensitive, duplicates allowed, order is
# important, ...), there doesn't appear to be any standard way to handle
# headers in Python -- they're almost like dicts, but... actually just
# aren't. For now we punt and just use a super simple representation: headers
# are a list of pairs
#
#   [(name1, value1), (name2, value2), ...]
#
# where all entries are bytestrings, names are lowercase and have no
# leading/trailing whitespace, and values are bytestrings with no
# leading/trailing whitespace. Searching and updating are done via naive O(n)
# methods.
#
# Maybe a dict-of-lists would be better?

_content_length_re = re.compile(rb"[0-9]+")
_field_name_re = re.compile(field_name.encode("ascii"))
_field_value_re = re.compile(field_value.encode("ascii"))


class Headers(Sequence[Tuple[bytes, bytes]]):
    """
    A list-like interface that allows iterating over headers as byte-pairs
    of (lowercased-name, value).

    Internally we actually store the representation as three-tuples,
    including both the raw original casing, in order to preserve casing
    over-the-wire, and the lowercased name, for case-insensitive comparisions.

    r = Request(
        method="GET",
        target="/",
        headers=[("Host", "example.org"), ("Connection", "keep-alive")],
        http_version="1.1",
    )
    assert r.headers == [
        (b"host", b"example.org"),
        (b"connection", b"keep-alive")
    ]
    assert r.headers.raw_items() == [
        (b"Host", b"example.org"),
        (b"Connection", b"keep-alive")
    ]
    """

    __slots__ = "_full_items"

    def __init__(self, full_items: List[Tuple[bytes, bytes, bytes]]) -> None:
        self._full_items = full_items

    def __bool__(self) -> bool:
        return bool(self._full_items)

    def __eq__(self, other: object) -> bool:
        return list(self) == list(other)  # type: ignore

    def __len__(self) -> int:
        return len(self._full_items)

    def __repr__(self) -> str:
        return "<Headers(%s)>" % repr(list(self))

    def __getitem__(self, idx: int) -> Tuple[bytes, bytes]:  # type: ignore[override]
        _, name, value = self._full_items[idx]
        return (name, value)

    def raw_items(self) -> List[Tuple[bytes, bytes]]:
        return [(raw_name, value) for raw_name, _, value in self._full_items]


HeaderTypes = Union[
    List[Tuple[bytes, bytes]],
    List[Tuple[bytes, str]],
    List[Tuple[str, bytes]],
    List[Tuple[str, str]],
]


@overload
def normalize_and_validate(headers: Headers, _parsed: Literal[True]) -> Headers:
    ...


@overload
def normalize_and_validate(headers: HeaderTypes, _parsed: Literal[False]) -> Headers:
    ...


@overload
def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    ...


def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    new_headers = []
    seen_content_length = None
    saw_transfer_encoding = False
    for name, value in headers:
        # For headers coming out of the parser, we can safely skip some steps,
        # because it always returns bytes and has already run these regexes
        # over the data:
        if not _parsed:
            name = bytesify(name)
            value = bytesify(value)
            validate(_field_name_re, name, "Illegal header name {!r}", name)
            validate(_field_value_re, value, "Illegal header value {!r}", value)
        assert isinstance(name, bytes)
        assert isinstance(value, bytes)

        raw_name = name
        name = name.lower()
        if name == b"content-length":
            lengths = {length.strip() for length in value.split(b",")}
            if len(lengths) != 1:
                raise LocalProtocolError("conflicting Content-Length headers")
            value = lengths.pop()
            validate(_content_length_re, value, "bad Content-Length")
            if len(value) > CONTENT_LENGTH_MAX_DIGITS:
                raise LocalProtocolError("bad Content-Length")
            if seen_content_length is None:
                seen_content_length = value
                new_headers.append((raw_name, name, value))
            elif seen_content_length != value:
                raise LocalProtocolError("conflicting Content-Length headers")
        elif name == b"transfer-encoding":
            # "A server that receives a request message with a transfer coding
            # it does not understand SHOULD respond with 501 (Not
            # Implemented)."
            # https://tools.ietf.org/html/rfc7230#section-3.3.1
            if saw_transfer_encoding:
                raise LocalProtocolError(
                    "multiple Transfer-Encoding headers", error_status_hint=501
                )
            # "All transfer-coding names are case-insensitive"
            # -- https://tools.ietf.org/html/rfc7230#section-4
            value = value.lower()
            if value != b"chunked":
                raise LocalProtocolError(
                    "Only Transfer-Encoding: chunked is supported",
                    error_status_hint=501,
                )
            saw_transfer_encoding = True
            new_headers.append((raw_name, name, value))
        else:
            new_headers.append((raw_name, name, value))
    return Headers(new_headers)


def get_comma_header(headers: Headers, name: bytes) -> List[bytes]:
    # Should only be used for headers whose value is a list of
    # comma-separated, case-insensitive values.
    #
    # The header name `name` is expected to be lower-case bytes.
    #
    # Connection: meets these criteria (including cast insensitivity).
    #
    # Content-Length: technically is just a single value (1*DIGIT), but the
    # standard makes reference to implementations that do multiple values, and
    # using this doesn't hurt. Ditto, case insensitivity doesn't things either
    # way.
    #
    # Transfer-Encoding: is more complex (allows for quoted strings), so
    # splitting on , is actually wrong. For example, this is legal:
    #
    #    Transfer-Encoding: foo; options="1,2", chunked
    #
    # and should be parsed as
    #
    #    foo; options="1,2"
    #    chunked
    #
    # but this naive function will parse it as
    #
    #    foo; options="1
    #    2"
    #    chunked
    #
    # However, this is okay because the only thing we are going to do with
    # any Transfer-Encoding is reject ones that aren't just "chunked", so
    # both of these will be treated the same anyway.
    #
    # Expect: the only legal value is the literal string
    # "100-continue". Splitting on commas is harmless. Case insensitive.
    #
    out: List[bytes] = []
    for _, found_name, found_raw_value in headers._full_items:
        if found_name == name:
            found_raw_value = found_raw_value.lower()
            for found_split_value in found_raw_value.split(b","):
                found_split_value = found_split_value.strip()
                if found_split_value:
                    out.append(found_split_value)
    return out


def set_comma_header(headers: Headers, name: bytes, new_values: List[bytes]) -> Headers:
    # The header name `name` is expected to be lower-case bytes.
    #
    # Note that when we store the header we use title casing for the header
    # names, in order to match the conventional HTTP header style.
    #
    # Simply calling `.title()` is a blunt approach, but it's correct
    # here given the cases where we're using `set_comma_header`...
    #
    # Connection, Content-Length, Transfer-Encoding.
    new_headers: List[Tuple[bytes, bytes]] = []
    for found_raw_name, found_name, found_raw_value in headers._full_items:
        if found_name != name:
            new_headers.append((found_raw_name, found_raw_value))
    for new_value in new_values:
        new_headers.append((name.title(), new_value))
    return normalize_and_validate(new_headers)


def has_expect_100_continue(request: "Request") -> bool:
    # https://tools.ietf.org/html/rfc7231#section-5.1.1
    # "A server that receives a 100-continue expectation in an HTTP/1.0 request
    # MUST ignore that expectation."
    if request.http_version < b"1.1":
        return False
    expect = get_comma_header(request.headers, b"expect")
    return b"100-continue" in expect
<|endoftext|># A highish-level implementation of the HTTP/1.1 wire protocol (RFC 7230),
# containing no networking code at all, loosely modelled on hyper-h2's generic
# implementation of HTTP/2 (and in particular the h2.connection.H2Connection
# class). There's still a bunch of subtle details you need to get right if you
# want to make this actually useful, because it doesn't implement all the
# semantics to check that what you're asking to write to the wire is sensible,
# but at least it gets you out of dealing with the wire itself.

from h11._connection import Connection, NEED_DATA, PAUSED
from h11._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from h11._state import (
    CLIENT,
    CLOSED,
    DONE,
    ERROR,
    IDLE,
    MIGHT_SWITCH_PROTOCOL,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
    SWITCHED_PROTOCOL,
)
from h11._util import LocalProtocolError, ProtocolError, RemoteProtocolError
from h11._version import __version__

PRODUCT_ID = "python-h11/" + __version__


__all__ = (
    "Connection",
    "NEED_DATA",
    "PAUSED",
    "ConnectionClosed",
    "Data",
    "EndOfMessage",
    "Event",
    "InformationalResponse",
    "Request",
    "Response",
    "CLIENT",
    "CLOSED",
    "DONE",
    "ERROR",
    "IDLE",
    "MUST_CLOSE",
    "SEND_BODY",
    "SEND_RESPONSE",
    "SERVER",
    "SWITCHED_PROTOCOL",
    "ProtocolError",
    "LocalProtocolError",
    "RemoteProtocolError",
)
<|endoftext|>import re
import sys
from typing import List, Optional, Union

__all__ = ["ReceiveBuffer"]


# Operations we want to support:
# - find next \r\n or \r\n\r\n (\n or \n\n are also acceptable),
#   or wait until there is one
# - read at-most-N bytes
# Goals:
# - on average, do this fast
# - worst case, do this in O(n) where n is the number of bytes processed
# Plan:
# - store bytearray, offset, how far we've searched for a separator token
# - use the how-far-we've-searched data to avoid rescanning
# - while doing a stream of uninterrupted processing, advance offset instead
#   of constantly copying
# WARNING:
# - I haven't benchmarked or profiled any of this yet.
#
# Note that starting in Python 3.4, deleting the initial n bytes from a
# bytearray is amortized O(n), thanks to some excellent work by Antoine
# Martin:
#
#     https://bugs.python.org/issue19087
#
# This means that if we only supported 3.4+, we could get rid of the code here
# involving self._start and self.compress, because it's doing exactly the same
# thing that bytearray now does internally.
#
# BUT unfortunately, we still support 2.7, and reading short segments out of a
# long buffer MUST be O(bytes read) to avoid DoS issues, so we can't actually
# delete this code. Yet:
#
#     https://pythonclock.org/
#
# (Two things to double-check first though: make sure PyPy also has the
# optimization, and benchmark to make sure it's a win, since we do have a
# slightly clever thing where we delay calling compress() until we've
# processed a whole event, which could in theory be slightly more efficient
# than the internal bytearray support.)
blank_line_regex = re.compile(b"\n\r?\n", re.MULTILINE)


class ReceiveBuffer:
    def __init__(self) -> None:
        self._data = bytearray()
        self._next_line_search = 0
        self._multiple_lines_search = 0

    def __iadd__(self, byteslike: Union[bytes, bytearray]) -> "ReceiveBuffer":
        self._data += byteslike
        return self

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self._data)

    # for @property unprocessed_data
    def __bytes__(self) -> bytes:
        return bytes(self._data)

    def _extract(self, count: int) -> bytearray:
        # extracting an initial slice of the data buffer and return it
        out = self._data[:count]
        del self._data[:count]

        self._next_line_search = 0
        self._multiple_lines_search = 0

        return out

    def maybe_extract_at_most(self, count: int) -> Optional[bytearray]:
        """
        Extract a fixed number of bytes from the buffer.
        """
        out = self._data[:count]
        if not out:
            return None

        return self._extract(count)

    def maybe_extract_next_line(self) -> Optional[bytearray]:
        """
        Extract the first line, if it is completed in the buffer.
        """
        # Only search in buffer space that we've not already looked at.
        search_start_index = max(0, self._next_line_search - 1)
        partial_idx = self._data.find(b"\r\n", search_start_index)

        if partial_idx == -1:
            self._next_line_search = len(self._data)
            return None

        # + 2 is to compensate len(b"\r\n")
        idx = partial_idx + 2

        return self._extract(idx)

    def maybe_extract_lines(self) -> Optional[List[bytearray]]:
        """
        Extract everything up to the first blank line, and return a list of lines.
        """
        # Handle the case where we have an immediate empty line.
        if self._data[:1] == b"\n":
            self._extract(1)
            return []

        if self._data[:2] == b"\r\n":
            self._extract(2)
            return []

        # Only search in buffer space that we've not already looked at.
        match = blank_line_regex.search(self._data, self._multiple_lines_search)
        if match is None:
            self._multiple_lines_search = max(0, len(self._data) - 2)
            return None

        # Truncate the buffer and return it.
        idx = match.span(0)[-1]
        out = self._extract(idx)
        lines = out.split(b"\n")

        for line in lines:
            if line.endswith(b"\r"):
                del line[-1]

        assert lines[-2] == lines[-1] == b""

        del lines[-2:]

        return lines

    # In theory we should wait until `\r\n` before starting to validate
    # incoming data. However it's interesting to detect (very) invalid data
    # early given they might not even contain `\r\n` at all (hence only
    # timeout will get rid of them).
    # This is not a 100% effective detection but more of a cheap sanity check
    # allowing for early abort in some useful cases.
    # This is especially interesting when peer is messing up with HTTPS and
    # sent us a TLS stream where we were expecting plain HTTP given all
    # versions of TLS so far start handshake with a 0x16 message type code.
    def is_next_line_obviously_invalid_request_line(self) -> bool:
        try:
            # HTTP header line must not contain non-printable characters
            # and should not start with a space
            return self._data[0] < 0x21
        except IndexError:
            return False
<|endoftext|>from typing import Any, Dict, NoReturn, Pattern, Tuple, Type, TypeVar, Union

__all__ = [
    "ProtocolError",
    "LocalProtocolError",
    "RemoteProtocolError",
    "validate",
    "bytesify",
]


class ProtocolError(Exception):
    """Exception indicating a violation of the HTTP/1.1 protocol.

    This as an abstract base class, with two concrete base classes:
    :exc:`LocalProtocolError`, which indicates that you tried to do something
    that HTTP/1.1 says is illegal, and :exc:`RemoteProtocolError`, which
    indicates that the remote peer tried to do something that HTTP/1.1 says is
    illegal. See :ref:`error-handling` for details.

    In addition to the normal :exc:`Exception` features, it has one attribute:

    .. attribute:: error_status_hint

       This gives a suggestion as to what status code a server might use if
       this error occurred as part of a request.

       For a :exc:`RemoteProtocolError`, this is useful as a suggestion for
       how you might want to respond to a misbehaving peer, if you're
       implementing a server.

       For a :exc:`LocalProtocolError`, this can be taken as a suggestion for
       how your peer might have responded to *you* if h11 had allowed you to
       continue.

       The default is 400 Bad Request, a generic catch-all for protocol
       violations.

    """

    def __init__(self, msg: str, error_status_hint: int = 400) -> None:
        if type(self) is ProtocolError:
            raise TypeError("tried to directly instantiate ProtocolError")
        Exception.__init__(self, msg)
        self.error_status_hint = error_status_hint


# Strategy: there are a number of public APIs where a LocalProtocolError can
# be raised (send(), all the different event constructors, ...), and only one
# public API where RemoteProtocolError can be raised
# (receive_data()). Therefore we always raise LocalProtocolError internally,
# and then receive_data will translate this into a RemoteProtocolError.
#
# Internally:
#   LocalProtocolError is the generic "ProtocolError".
# Externally:
#   LocalProtocolError is for local errors and RemoteProtocolError is for
#   remote errors.
class LocalProtocolError(ProtocolError):
    def _reraise_as_remote_protocol_error(self) -> NoReturn:
        # After catching a LocalProtocolError, use this method to re-raise it
        # as a RemoteProtocolError. This method must be called from inside an
        # except: block.
        #
        # An easy way to get an equivalent RemoteProtocolError is just to
        # modify 'self' in place.
        self.__class__ = RemoteProtocolError  # type: ignore
        # But the re-raising is somewhat non-trivial -- you might think that
        # now that we've modified the in-flight exception object, that just
        # doing 'raise' to re-raise it would be enough. But it turns out that
        # this doesn't work, because Python tracks the exception type
        # (exc_info[0]) separately from the exception object (exc_info[1]),
        # and we only modified the latter. So we really do need to re-raise
        # the new type explicitly.
        # On py3, the traceback is part of the exception object, so our
        # in-place modification preserved it and we can just re-raise:
        raise self


class RemoteProtocolError(ProtocolError):
    pass


def validate(
    regex: Pattern[bytes], data: bytes, msg: str = "malformed data", *format_args: Any
) -> Dict[str, bytes]:
    match = regex.fullmatch(data)
    if not match:
        if format_args:
            msg = msg.format(*format_args)
        raise LocalProtocolError(msg)
    return match.groupdict()


# Sentinel values
#
# - Inherit identity-based comparison and hashing from object
# - Have a nice repr
# - Have a *bonus property*: type(sentinel) is sentinel
#
# The bonus property is useful if you want to take the return value from
# next_event() and do some sort of dispatch based on type(event).

_T_Sentinel = TypeVar("_T_Sentinel", bound="Sentinel")


class Sentinel(type):
    def __new__(
        cls: Type[_T_Sentinel],
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwds: Any
    ) -> _T_Sentinel:
        assert bases == (Sentinel,)
        v = super().__new__(cls, name, bases, namespace, **kwds)
        v.__class__ = v  # type: ignore
        return v

    def __repr__(self) -> str:
        return self.__name__


# Used for methods, request targets, HTTP versions, header names, and header
# values. Accepts ascii-strings, or bytes/bytearray/memoryview/..., and always
# returns bytes.
def bytesify(s: Union[bytes, bytearray, memoryview, int, str]) -> bytes:
    # Fast-path:
    if type(s) is bytes:
        return s
    if isinstance(s, str):
        s = s.encode("ascii")
    if isinstance(s, int):
        raise TypeError("expected bytes-like object, not int")
    return bytes(s)
<|endoftext|># Code to read HTTP data
#
# Strategy: each reader is a callable which takes a ReceiveBuffer object, and
# either:
# 1) consumes some of it and returns an Event
# 2) raises a LocalProtocolError (for consistency -- e.g. we call validate()
#    and it might raise a LocalProtocolError, so simpler just to always use
#    this)
# 3) returns None, meaning "I need more data"
#
# If they have a .read_eof attribute, then this will be called if an EOF is
# received -- but this is optional. Either way, the actual ConnectionClosed
# event will be generated afterwards.
#
# READERS is a dict describing how to pick a reader. It maps states to either:
# - a reader
# - or, for body readers, a dict of per-framing reader factories

import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union

from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
    CLIENT,
    CLOSED,
    DONE,
    IDLE,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
)
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate

__all__ = ["READERS"]

header_field_re = re.compile(header_field.encode("ascii"))
obs_fold_re = re.compile(rb"[ \t]+")


def _obsolete_line_fold(lines: Iterable[bytes]) -> Iterable[bytes]:
    it = iter(lines)
    last: Optional[bytes] = None
    for line in it:
        match = obs_fold_re.match(line)
        if match:
            if last is None:
                raise LocalProtocolError("continuation line at start of headers")
            if not isinstance(last, bytearray):
                # Cast to a mutable type, avoiding copy on append to ensure O(n) time
                last = bytearray(last)
            last += b" "
            last += line[match.end() :]
        else:
            if last is not None:
                yield last
            last = line
    if last is not None:
        yield last


def _decode_header_lines(
    lines: Iterable[bytes],
) -> Iterable[Tuple[bytes, bytes]]:
    for line in _obsolete_line_fold(lines):
        matches = validate(header_field_re, line, "illegal header line: {!r}", line)
        yield (matches["field_name"], matches["field_value"])


request_line_re = re.compile(request_line.encode("ascii"))


def maybe_read_from_IDLE_client(buf: ReceiveBuffer) -> Optional[Request]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None
    if not lines:
        raise LocalProtocolError("no request line received")
    matches = validate(
        request_line_re, lines[0], "illegal request line: {!r}", lines[0]
    )
    return Request(
        headers=list(_decode_header_lines(lines[1:])), _parsed=True, **matches
    )


status_line_re = re.compile(status_line.encode("ascii"))


def maybe_read_from_SEND_RESPONSE_server(
    buf: ReceiveBuffer,
) -> Union[InformationalResponse, Response, None]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None
    if not lines:
        raise LocalProtocolError("no response line received")
    matches = validate(status_line_re, lines[0], "illegal status line: {!r}", lines[0])
    http_version = (
        b"1.1" if matches["http_version"] is None else matches["http_version"]
    )
    reason = b"" if matches["reason"] is None else matches["reason"]
    status_code = int(matches["status_code"])
    class_: Union[Type[InformationalResponse], Type[Response]] = (
        InformationalResponse if status_code < 200 else Response
    )
    return class_(
        headers=list(_decode_header_lines(lines[1:])),
        _parsed=True,
        status_code=status_code,
        reason=reason,
        http_version=http_version,
    )


class ContentLengthReader:
    def __init__(self, length: int) -> None:
        self._length = length
        self._remaining = length

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._remaining == 0:
            return EndOfMessage()
        data = buf.maybe_extract_at_most(self._remaining)
        if data is None:
            return None
        self._remaining -= len(data)
        return Data(data=data)

    def read_eof(self) -> NoReturn:
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(received {} bytes, expected {})".format(
                self._length - self._remaining, self._length
            )
        )


chunk_header_re = re.compile(chunk_header.encode("ascii"))


class ChunkedReader:
    def __init__(self) -> None:
        self._bytes_in_chunk = 0
        # After reading a chunk, we have to throw away the trailing \r\n.
        # This tracks the bytes that we need to match and throw away.
        self._bytes_to_discard = b""
        self._reading_trailer = False

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._reading_trailer:
            lines = buf.maybe_extract_lines()
            if lines is None:
                return None
            return EndOfMessage(headers=list(_decode_header_lines(lines)))
        if self._bytes_to_discard:
            data = buf.maybe_extract_at_most(len(self._bytes_to_discard))
            if data is None:
                return None
            if data != self._bytes_to_discard[: len(data)]:
                raise LocalProtocolError(
                    f"malformed chunk footer: {data!r} (expected {self._bytes_to_discard!r})"
                )
            self._bytes_to_discard = self._bytes_to_discard[len(data) :]
            if self._bytes_to_discard:
                return None
            # else, fall through and read some more
        assert self._bytes_to_discard == b""
        if self._bytes_in_chunk == 0:
            # We need to refill our chunk count
            chunk_header = buf.maybe_extract_next_line()
            if chunk_header is None:
                return None
            matches = validate(
                chunk_header_re,
                chunk_header,
                "illegal chunk header: {!r}",
                chunk_header,
            )
            # XX FIXME: we discard chunk extensions. Does anyone care?
            self._bytes_in_chunk = int(matches["chunk_size"], base=16)
            if self._bytes_in_chunk == 0:
                self._reading_trailer = True
                return self(buf)
            chunk_start = True
        else:
            chunk_start = False
        assert self._bytes_in_chunk > 0
        data = buf.maybe_extract_at_most(self._bytes_in_chunk)
        if data is None:
            return None
        self._bytes_in_chunk -= len(data)
        if self._bytes_in_chunk == 0:
            self._bytes_to_discard = b"\r\n"
            chunk_end = True
        else:
            chunk_end = False
        return Data(data=data, chunk_start=chunk_start, chunk_end=chunk_end)

    def read_eof(self) -> NoReturn:
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )


class Http10Reader:
    def __call__(self, buf: ReceiveBuffer) -> Optional[Data]:
        data = buf.maybe_extract_at_most(999999999)
        if data is None:
            return None
        return Data(data=data)

    def read_eof(self) -> EndOfMessage:
        return EndOfMessage()


def expect_nothing(buf: ReceiveBuffer) -> None:
    if buf:
        raise LocalProtocolError("Got data when expecting EOF")
    return None


ReadersType = Dict[
    Union[Type[Sentinel], Tuple[Type[Sentinel], Type[Sentinel]]],
    Union[Callable[..., Any], Dict[str, Callable[..., Any]]],
]

READERS: ReadersType = {
    (CLIENT, IDLE): maybe_read_from_IDLE_client,
    (SERVER, IDLE): maybe_read_from_SEND_RESPONSE_server,
    (SERVER, SEND_RESPONSE): maybe_read_from_SEND_RESPONSE_server,
    (CLIENT, DONE): expect_nothing,
    (CLIENT, MUST_CLOSE): expect_nothing,
    (CLIENT, CLOSED): expect_nothing,
    (SERVER, DONE): expect_nothing,
    (SERVER, MUST_CLOSE): expect_nothing,
    (SERVER, CLOSED): expect_nothing,
    SEND_BODY: {
        "chunked": ChunkedReader,
        "content-length": ContentLengthReader,
        "http/1.0": Http10Reader,
    },
}
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import ast
from collections.abc import Sequence
from concurrent import futures
# import threading submodule upfront to avoid partially initialized
# module bug (ARROW-11983)
import concurrent.futures.thread  # noqa
from copy import deepcopy
import decimal
from itertools import zip_longest
import json
import operator
import re
import warnings

try:
    import numpy as np
except ImportError:
    np = None
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes, is_threading_enabled  # noqa


_logical_type_map = {}
_numpy_logical_type_map = {}
_pandas_logical_type_map = {}


def get_logical_type_map():
    global _logical_type_map

    if not _logical_type_map:
        _logical_type_map.update({
            pa.lib.Type_NA: 'empty',
            pa.lib.Type_BOOL: 'bool',
            pa.lib.Type_INT8: 'int8',
            pa.lib.Type_INT16: 'int16',
            pa.lib.Type_INT32: 'int32',
            pa.lib.Type_INT64: 'int64',
            pa.lib.Type_UINT8: 'uint8',
            pa.lib.Type_UINT16: 'uint16',
            pa.lib.Type_UINT32: 'uint32',
            pa.lib.Type_UINT64: 'uint64',
            pa.lib.Type_HALF_FLOAT: 'float16',
            pa.lib.Type_FLOAT: 'float32',
            pa.lib.Type_DOUBLE: 'float64',
            pa.lib.Type_DATE32: 'date',
            pa.lib.Type_DATE64: 'date',
            pa.lib.Type_TIME32: 'time',
            pa.lib.Type_TIME64: 'time',
            pa.lib.Type_BINARY: 'bytes',
            pa.lib.Type_FIXED_SIZE_BINARY: 'bytes',
            pa.lib.Type_STRING: 'unicode',
        })
    return _logical_type_map


def get_logical_type(arrow_type):
    logical_type_map = get_logical_type_map()

    try:
        return logical_type_map[arrow_type.id]
    except KeyError:
        if isinstance(arrow_type, pa.lib.DictionaryType):
            return 'categorical'
        elif isinstance(arrow_type, pa.lib.ListType):
            return f'list[{get_logical_type(arrow_type.value_type)}]'
        elif isinstance(arrow_type, pa.lib.TimestampType):
            return 'datetimetz' if arrow_type.tz is not None else 'datetime'
        elif pa.types.is_decimal(arrow_type):
            return 'decimal'
        return 'object'


def get_numpy_logical_type_map():
    global _numpy_logical_type_map
    if not _numpy_logical_type_map:
        _numpy_logical_type_map.update({
            np.bool_: 'bool',
            np.int8: 'int8',
            np.int16: 'int16',
            np.int32: 'int32',
            np.int64: 'int64',
            np.uint8: 'uint8',
            np.uint16: 'uint16',
            np.uint32: 'uint32',
            np.uint64: 'uint64',
            np.float32: 'float32',
            np.float64: 'float64',
            'datetime64[D]': 'date',
            np.str_: 'string',
            np.bytes_: 'bytes',
        })
    return _numpy_logical_type_map


def get_logical_type_from_numpy(pandas_collection):
    numpy_logical_type_map = get_numpy_logical_type_map()
    try:
        return numpy_logical_type_map[pandas_collection.dtype.type]
    except KeyError:
        if hasattr(pandas_collection.dtype, 'tz'):
            return 'datetimetz'
        # See https://github.com/pandas-dev/pandas/issues/24739 (infer_dtype will
        # result in "datetime64" without unit, while pandas astype requires a unit)
        if str(pandas_collection.dtype).startswith('datetime64'):
            return str(pandas_collection.dtype)
        result = _pandas_api.infer_dtype(pandas_collection)
        if result == 'string':
            return 'unicode'
        return result


def get_extension_dtype_info(column):
    dtype = column.dtype
    if str(dtype) == 'category':
        cats = getattr(column, 'cat', column)
        assert cats is not None
        metadata = {
            'num_categories': len(cats.categories),
            'ordered': cats.ordered,
        }
        physical_dtype = str(cats.codes.dtype)
    elif hasattr(dtype, 'tz'):
        metadata = {'timezone': pa.lib.tzinfo_to_string(dtype.tz)}
        physical_dtype = f'datetime64[{dtype.unit}]'
    else:
        metadata = None
        physical_dtype = str(dtype)
    return physical_dtype, metadata


def get_column_metadata(column, name, arrow_type, field_name):
    """Construct the metadata for a given column

    Parameters
    ----------
    column : pandas.Series or pandas.Index
    name : str
    arrow_type : pyarrow.DataType
    field_name : str
        Equivalent to `name` when `column` is a `Series`, otherwise if `column`
        is a pandas Index then `field_name` will not be the same as `name`.
        This is the name of the field in the arrow Table's schema.

    Returns
    -------
    dict
    """
    logical_type = get_logical_type(arrow_type)

    string_dtype, extra_metadata = get_extension_dtype_info(column)
    if logical_type == 'decimal':
        extra_metadata = {
            'precision': arrow_type.precision,
            'scale': arrow_type.scale,
        }
        string_dtype = 'object'

    if (
        name is not None
        and not (isinstance(name, float) and np.isnan(name))
        and not isinstance(name, str)
    ):
        raise TypeError(
            f"Column name must be a string. Got column {name} of type "
            f"{type(name).__name__}"
        )

    assert isinstance(field_name, str), str(type(field_name))
    return {
        'name': name,
        'field_name': field_name,
        'pandas_type': logical_type,
        'numpy_type': string_dtype,
        'metadata': extra_metadata,
    }


def construct_metadata(columns_to_convert, df, column_names, index_levels,
                       index_descriptors, preserve_index, types,
                       column_field_names=None):
    """Returns a dictionary containing enough metadata to reconstruct a pandas
    DataFrame as an Arrow Table, including index columns.

    Parameters
    ----------
    columns_to_convert : list[pd.Series]
    df : pandas.DataFrame
    column_names : list[str | None]
    column_field_names: list[str]
    index_levels : List[pd.Index]
    index_descriptors : List[Dict]
    preserve_index : bool
    types : List[pyarrow.DataType]

    Returns
    -------
    dict
    """
    if column_field_names is None:
        # backwards compatibility for external projects that are using
        # `construct_metadata` such as cudf
        # see https://github.com/apache/arrow/pull/44963#discussion_r1875771953
        column_field_names = [str(name) for name in column_names]

    serialized_index_levels = [
        (level, descriptor)
        for level, descriptor in zip(index_levels, index_descriptors)
        if not isinstance(descriptor, dict)
    ]

    num_serialized_index_levels = len(serialized_index_levels)

    # Use ntypes instead of Python shorthand notation [:-len(x)] as [:-0]
    # behaves differently to what we want.
    ntypes = len(types)
    df_types = types[:ntypes - num_serialized_index_levels]
    index_types = types[ntypes - num_serialized_index_levels:]

    column_metadata = []
    for col, name, field_name, arrow_type in zip(columns_to_convert, column_names,
                                                 column_field_names, df_types):
        metadata = get_column_metadata(col, name=name,
                                       arrow_type=arrow_type,
                                       field_name=field_name)
        column_metadata.append(metadata)

    index_column_metadata = []
    if preserve_index is not False:
        non_str_index_names = []
        for (level, descriptor), arrow_type in zip(
            serialized_index_levels, index_types
        ):
            if level.name is not None and not isinstance(level.name, str):
                non_str_index_names.append(level.name)

            metadata = get_column_metadata(
                level,
                name=_column_name_to_strings(level.name),
                arrow_type=arrow_type,
                field_name=descriptor,
            )
            index_column_metadata.append(metadata)

        if len(non_str_index_names) > 0:
            warnings.warn(
                f"The DataFrame has non-str index name `{non_str_index_names}`"
                " which will be converted to string"
                " and not roundtrip correctly.",
                UserWarning, stacklevel=4)

        column_indexes = []

        levels = getattr(df.columns, 'levels', [df.columns])
        names = getattr(df.columns, 'names', [df.columns.name])
        for level, name in zip(levels, names):
            metadata = _get_simple_index_descriptor(level, name)
            column_indexes.append(metadata)
    else:
        index_descriptors = index_column_metadata = column_indexes = []

    attributes = df.attrs if hasattr(df, "attrs") else {}

    try:
        json.dumps(attributes)
    except Exception as e:
        attributes = {}
        warnings.warn(
            f"Could not serialize pd.DataFrame.attrs: {e},"
            f" defaulting to empty attributes.",
            UserWarning, stacklevel=4)

    return {
        b'pandas': json.dumps({
            'index_columns': index_descriptors,
            'column_indexes': column_indexes,
            'columns': column_metadata + index_column_metadata,
            'attributes': attributes,
            'creator': {
                'library': 'pyarrow',
                'version': pa.__version__
            },
            'pandas_version': _pandas_api.version
        }).encode('utf8')
    }


def _get_simple_index_descriptor(level, name):
    string_dtype, extra_metadata = get_extension_dtype_info(level)
    pandas_type = get_logical_type_from_numpy(level)
    if 'mixed' in pandas_type:
        warnings.warn(
            "The DataFrame has column names of mixed type. They will be "
            "converted to strings and not roundtrip correctly.",
            UserWarning, stacklevel=4)
    if pandas_type == 'unicode':
        assert not extra_metadata
        extra_metadata = {'encoding': 'UTF-8'}
    return {
        'name': name,
        'field_name': name,
        'pandas_type': pandas_type,
        'numpy_type': string_dtype,
        'metadata': extra_metadata,
    }


def _column_name_to_strings(name):
    """Convert a column name (or level) to either a string or a recursive
    collection of strings.

    Parameters
    ----------
    name : str or tuple

    Returns
    -------
    value : str or tuple

    Examples
    --------
    >>> name = 'foo'
    >>> _column_name_to_strings(name)
    'foo'
    >>> name = ('foo', 'bar')
    >>> _column_name_to_strings(name)
    "('foo', 'bar')"
    >>> import pandas as pd
    >>> name = (1, pd.Timestamp('2017-02-01 00:00:00'))
    >>> _column_name_to_strings(name)
    "('1', '2017-02-01 00:00:00')"
    """
    if isinstance(name, str):
        return name
    elif isinstance(name, bytes):
        # XXX: should we assume that bytes in Python 3 are UTF-8?
        return name.decode('utf8')
    elif isinstance(name, tuple):
        return str(tuple(map(_column_name_to_strings, name)))
    elif isinstance(name, Sequence):
        raise TypeError("Unsupported type for MultiIndex level")
    elif name is None or (isinstance(name, float) and np.isnan(name)):
        return name
    return str(name)


def _index_level_name(index, i, column_names):
    """Return the name of an index level or a default name if `index.name` is
    None or is already a column name.

    Parameters
    ----------
    index : pandas.Index
    i : int

    Returns
    -------
    name : str
    """
    if index.name is not None and index.name not in column_names:
        return _column_name_to_strings(index.name)
    else:
        j = i
        while f'__index_level_{j:d}__' in column_names:
            j += 1
        return f'__index_level_{j:d}__'


def _get_columns_to_convert(df, schema, preserve_index, columns):
    columns = _resolve_columns_of_interest(df, schema, columns)

    if not df.columns.is_unique:
        raise ValueError(
            f'Duplicate column names found: {list(df.columns)}'
        )

    if schema is not None:
        return _get_columns_to_convert_given_schema(df, schema, preserve_index)

    column_names = []
    column_field_names = []

    index_levels = (
        _get_index_level_values(df.index) if preserve_index is not False
        else []
    )

    columns_to_convert = []
    convert_fields = []

    for name in columns:
        col = df[name]
        name = _column_name_to_strings(name)

        if _pandas_api.is_sparse(col):
            raise TypeError(
                f"Sparse pandas data (column {name}) not supported.")

        columns_to_convert.append(col)
        convert_fields.append(None)
        column_names.append(name)
        column_field_names.append(str(name))

    index_descriptors = []
    index_column_names = []
    for i, index_level in enumerate(index_levels):
        name = _index_level_name(
            index_level, i, column_names + index_column_names
        )
        if (isinstance(index_level, _pandas_api.pd.RangeIndex) and
                preserve_index is None):
            descr = _get_range_index_descriptor(index_level)
        else:
            columns_to_convert.append(index_level)
            convert_fields.append(None)
            descr = name
            index_column_names.append(name)
        index_descriptors.append(descr)

    all_names = column_field_names + index_column_names

    # all_names : all of the columns in the resulting table including the data
    # columns and serialized index columns
    # column_names : the names of the data columns
    # index_column_names : the names of the serialized index columns
    # index_descriptors : descriptions of each index to be used for
    # reconstruction
    # index_levels : the extracted index level values
    # columns_to_convert : assembled raw data (both data columns and indexes)
    # to be converted to Arrow format
    # columns_fields : specified column to use for coercion / casting
    # during serialization, if a Schema was provided
    return (all_names, column_names, column_field_names, index_column_names,
            index_descriptors, index_levels, columns_to_convert, convert_fields)


def _get_columns_to_convert_given_schema(df, schema, preserve_index):
    """
    Specialized version of _get_columns_to_convert in case a Schema is
    specified.
    In that case, the Schema is used as the single point of truth for the
    table structure (types, which columns are included, order of columns, ...).
    """
    column_names = []
    columns_to_convert = []
    convert_fields = []
    index_descriptors = []
    index_column_names = []
    index_levels = []

    for name in schema.names:
        try:
            col = df[name]
            is_index = False
        except KeyError:
            try:
                col = _get_index_level(df, name)
            except (KeyError, IndexError):
                # name not found as index level
                raise KeyError(
                    f"name '{name}' present in the specified schema is not found "
                    "in the columns or index")
            if preserve_index is False:
                raise ValueError(
                    f"name '{name}' present in the specified schema corresponds "
                    "to the index, but 'preserve_index=False' was "
                    "specified")
            elif (preserve_index is None and
                    isinstance(col, _pandas_api.pd.RangeIndex)):
                raise ValueError(
                    f"name '{name}' is present in the schema, but it is a "
                    "RangeIndex which will not be converted as a column "
                    "in the Table, but saved as metadata-only not in "
                    "columns. Specify 'preserve_index=True' to force it "
                    "being added as a column, or remove it from the "
                    "specified schema")
            is_index = True

        if _pandas_api.is_sparse(col):
            raise TypeError(
                f"Sparse pandas data (column {name}) not supported.")

        field = schema.field(name)
        columns_to_convert.append(col)
        convert_fields.append(field)
        column_names.append(name)

        if is_index:
            index_column_names.append(name)
            index_descriptors.append(name)
            index_levels.append(col)

    all_names = column_names + index_column_names

    return (all_names, column_names, column_names, index_column_names,
            index_descriptors, index_levels, columns_to_convert, convert_fields)


def _get_index_level(df, name):
    """
    Get the index level of a DataFrame given 'name' (column name in an arrow
    Schema).
    """
    key = name
    if name not in df.index.names and _is_generated_index_name(name):
        # we know we have an autogenerated name => extract number and get
        # the index level positionally
        key = int(name[len("__index_level_"):-2])
    return df.index.get_level_values(key)


def _level_name(name):
    # preserve type when default serializable, otherwise str it
    try:
        json.dumps(name)
        return name
    except TypeError:
        return str(name)


def _get_range_index_descriptor(level):
    # public start/stop/step attributes added in pandas 0.25.0
    return {
        'kind': 'range',
        'name': _level_name(level.name),
        'start': _pandas_api.get_rangeindex_attribute(level, 'start'),
        'stop': _pandas_api.get_rangeindex_attribute(level, 'stop'),
        'step': _pandas_api.get_rangeindex_attribute(level, 'step')
    }


def _get_index_level_values(index):
    n = len(getattr(index, 'levels', [index]))
    return [index.get_level_values(i) for i in range(n)]


def _resolve_columns_of_interest(df, schema, columns):
    if schema is not None and columns is not None:
        raise ValueError('Schema and columns arguments are mutually '
                         'exclusive, pass only one of them')
    elif schema is not None:
        columns = schema.names
    elif columns is not None:
        columns = [c for c in columns if c in df.columns]
    else:
        columns = df.columns

    return columns


def dataframe_to_types(df, preserve_index, columns=None):
    (all_names,
     column_names,
     column_field_names,
     _,
     index_descriptors,
     index_columns,
     columns_to_convert,
     _) = _get_columns_to_convert(df, None, preserve_index, columns)

    types = []
    # If pandas knows type, skip conversion
    for c in columns_to_convert:
        values = c.values
        if _pandas_api.is_categorical(values):
            type_ = pa.array(c, from_pandas=True).type
        elif _pandas_api.is_extension_array_dtype(values):
            empty = c.head(0) if isinstance(
                c, _pandas_api.pd.Series) else c[:0]
            type_ = pa.array(empty, from_pandas=True).type
        else:
            values, type_ = get_datetimetz_type(values, c.dtype, None)
            type_ = pa.lib._ndarray_to_arrow_type(values, type_)
            if type_ is None:
                type_ = pa.array(c, from_pandas=True).type
        types.append(type_)

    metadata = construct_metadata(
        columns_to_convert, df, column_names, index_columns, index_descriptors,
        preserve_index, types, column_field_names=column_field_names
    )

    return all_names, types, metadata


def dataframe_to_arrays(df, schema, preserve_index, nthreads=1, columns=None,
                        safe=True):
    (all_names,
     column_names,
     column_field_names,
     index_column_names,
     index_descriptors,
     index_columns,
     columns_to_convert,
     convert_fields) = _get_columns_to_convert(df, schema, preserve_index,
                                               columns)

    # NOTE(wesm): If nthreads=None, then we use a heuristic to decide whether
    # using a thread pool is worth it. Currently the heuristic is whether the
    # nrows > 100 * ncols and ncols > 1.
    if nthreads is None:
        nrows, ncols = len(df), len(df.columns)
        if nrows > ncols * 100 and ncols > 1:
            nthreads = pa.cpu_count()
        else:
            nthreads = 1
    # if we don't have threading in libarrow, don't use threading here either
    if not is_threading_enabled():
        nthreads = 1

    def convert_column(col, field):
        if field is None:
            field_nullable = True
            type_ = None
        else:
            field_nullable = field.nullable
            type_ = field.type

        try:
            result = pa.array(col, type=type_, from_pandas=True, safe=safe)
        except (pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError) as e:
            e.args += (
                f"Conversion failed for column {col.name} with type {col.dtype}",)
            raise e
        if not field_nullable and result.null_count > 0:
            raise ValueError(f"Field {field} was non-nullable but pandas column "
                             f"had {result.null_count} null values")
        return result

    def _can_definitely_zero_copy(arr):
        return (isinstance(arr, np.ndarray) and
                arr.flags.contiguous and
                issubclass(arr.dtype.type, np.integer))

    if nthreads == 1:
        arrays = [convert_column(c, f)
                  for c, f in zip(columns_to_convert, convert_fields)]
    else:
        arrays = []
        with futures.ThreadPoolExecutor(nthreads) as executor:
            for c, f in zip(columns_to_convert, convert_fields):
                if _can_definitely_zero_copy(c.values):
                    arrays.append(convert_column(c, f))
                else:
                    arrays.append(executor.submit(convert_column, c, f))

        for i, maybe_fut in enumerate(arrays):
            if isinstance(maybe_fut, futures.Future):
                arrays[i] = maybe_fut.result()

    types = [x.type for x in arrays]

    if schema is None:
        fields = []
        for name, type_ in zip(all_names, types):
            fields.append(pa.field(name, type_))
        schema = pa.schema(fields)

    pandas_metadata = construct_metadata(
        columns_to_convert, df, column_names, index_columns, index_descriptors,
        preserve_index, types, column_field_names=column_field_names
    )
    metadata = deepcopy(schema.metadata) if schema.metadata else dict()
    metadata.update(pandas_metadata)
    schema = schema.with_metadata(metadata)

    # If dataframe is empty but with RangeIndex ->
    # remember the length of the indexes
    n_rows = None
    if len(arrays) == 0:
        try:
            kind = index_descriptors[0]["kind"]
            if kind == "range":
                start = index_descriptors[0]["start"]
                stop = index_descriptors[0]["stop"]
                step = index_descriptors[0]["step"]
                n_rows = len(range(start, stop, step))
        except IndexError:
            pass

    return arrays, schema, n_rows


def get_datetimetz_type(values, dtype, type_):
    if values.dtype.type != np.datetime64:
        return values, type_

    if _pandas_api.is_datetimetz(dtype) and type_ is None:
        # If no user type passed, construct a tz-aware timestamp type
        tz = dtype.tz
        unit = dtype.unit
        type_ = pa.timestamp(unit, tz)
    elif type_ is None:
        # Trust the NumPy dtype
        type_ = pa.from_numpy_dtype(values.dtype)

    return values, type_

# ----------------------------------------------------------------------
# Converting pyarrow.Table efficiently to pandas.DataFrame


def _reconstruct_block(item, columns=None, extension_columns=None, return_block=True):
    """
    Construct a pandas Block from the `item` dictionary coming from pyarrow's
    serialization or returned by arrow::python::ConvertTableToPandas.

    This function takes care of converting dictionary types to pandas
    categorical, Timestamp-with-timezones to the proper pandas Block, and
    conversion to pandas ExtensionBlock

    Parameters
    ----------
    item : dict
        For basic types, this is a dictionary in the form of
        {'block': np.ndarray of values, 'placement': pandas block placement}.
        Additional keys are present for other types (dictionary, timezone,
        object).
    columns :
        Column names of the table being constructed, used for extension types
    extension_columns : dict
        Dictionary of {column_name: pandas_dtype} that includes all columns
        and corresponding dtypes that will be converted to a pandas
        ExtensionBlock.

    Returns
    -------
    pandas Block

    """
    import pandas.core.internals as _int

    block_arr = item.get('block', None)
    placement = item['placement']
    if 'dictionary' in item:
        arr = _pandas_api.categorical_type.from_codes(
            block_arr, categories=item['dictionary'],
            ordered=item['ordered'])
    elif 'timezone' in item:
        unit, _ = np.datetime_data(block_arr.dtype)
        dtype = make_datetimetz(unit, item['timezone'])
        if _pandas_api.is_ge_v21():
            arr = _pandas_api.pd.array(
                block_arr.view("int64"), dtype=dtype, copy=False
            )
        else:
            arr = block_arr
            if return_block:
                block = _int.make_block(block_arr, placement=placement,
                                        klass=_int.DatetimeTZBlock,
                                        dtype=dtype)
                return block
    elif 'py_array' in item:
        # create ExtensionBlock
        arr = item['py_array']
        assert len(placement) == 1
        name = columns[placement[0]]
        pandas_dtype = extension_columns[name]
        if not hasattr(pandas_dtype, '__from_arrow__'):
            raise ValueError("This column does not support to be converted "
                             "to a pandas ExtensionArray")
        arr = pandas_dtype.__from_arrow__(arr)
    else:
        arr = block_arr

    if return_block:
        return _int.make_block(arr, placement=placement)
    else:
        return arr, placement


def make_datetimetz(unit, tz):
    if _pandas_api.is_v1():
        unit = 'ns'  # ARROW-3789: Coerce date/timestamp types to datetime64[ns]
    tz = pa.lib.string_to_tzinfo(tz, prefer_zoneinfo=_pandas_api.is_ge_v3())
    return _pandas_api.datetimetz_type(unit, tz=tz)


def table_to_dataframe(
    options, table, categories=None, ignore_metadata=False, types_mapper=None
):
    all_columns = []
    column_indexes = []
    attributes = {}
    pandas_metadata = table.schema.pandas_metadata

    if not ignore_metadata and pandas_metadata is not None:
        all_columns = pandas_metadata['columns']
        column_indexes = pandas_metadata.get('column_indexes', [])
        attributes = pandas_metadata.get('attributes', {})
        index_descriptors = pandas_metadata['index_columns']
        table = _add_any_metadata(table, pandas_metadata)
        table, index = _reconstruct_index(table, index_descriptors,
                                          all_columns, types_mapper)
        ext_columns_dtypes = _get_extension_dtypes(
            table, all_columns, types_mapper, options, categories)
    else:
        index = _pandas_api.pd.RangeIndex(table.num_rows)
        ext_columns_dtypes = _get_extension_dtypes(
            table, [], types_mapper, options, categories
        )

    _check_data_column_metadata_consistency(all_columns)
    columns = _deserialize_column_index(table, all_columns, column_indexes)

    column_names = table.column_names
    result = pa.lib.table_to_blocks(options, table, categories,
                                    list(ext_columns_dtypes.keys()))
    if _pandas_api.is_ge_v3():
        from pandas.api.internals import create_dataframe_from_blocks

        blocks = [
            _reconstruct_block(
                item, column_names, ext_columns_dtypes, return_block=False)
            for item in result
        ]
        df = create_dataframe_from_blocks(blocks, index=index, columns=columns)
        df.attrs = attributes

        return df
    else:
        from pandas.core.internals import BlockManager
        from pandas import DataFrame

        blocks = [
            _reconstruct_block(item, column_names, ext_columns_dtypes)
            for item in result
        ]
        axes = [columns, index]
        mgr = BlockManager(blocks, axes)
        if _pandas_api.is_ge_v21():
            df = DataFrame._from_mgr(mgr, mgr.axes)
        else:
            df = DataFrame(mgr)

        df.attrs = attributes

        return df


# Set of the string repr of all numpy dtypes that can be stored in a pandas
# dataframe (complex not included since not supported by Arrow)
_pandas_supported_numpy_types = {
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "object", "bool"
}


def _get_extension_dtypes(table, columns_metadata, types_mapper, options, categories):
    """
    Based on the stored column pandas metadata and the extension types
    in the arrow schema, infer which columns should be converted to a
    pandas extension dtype.

    The 'numpy_type' field in the column metadata stores the string
    representation of the original pandas dtype (and, despite its name,
    not the 'pandas_type' field).
    Based on this string representation, a pandas/numpy dtype is constructed
    and then we can check if this dtype supports conversion from arrow.

    """
    strings_to_categorical = options["strings_to_categorical"]
    categories = categories or []

    ext_columns = {}

    # older pandas version that does not yet support extension dtypes
    if _pandas_api.extension_dtype is None:
        return ext_columns

    # use the specified mapping of built-in arrow types to pandas dtypes
    if types_mapper:
        for field in table.schema:
            typ = field.type
            pandas_dtype = types_mapper(typ)
            if pandas_dtype is not None:
                ext_columns[field.name] = pandas_dtype

    # infer from extension type in the schema
    for field in table.schema:
        typ = field.type
        if field.name not in ext_columns and isinstance(typ, pa.BaseExtensionType):
            try:
                pandas_dtype = typ.to_pandas_dtype()
            except NotImplementedError:
                pass
            else:
                ext_columns[field.name] = pandas_dtype

    # infer the extension columns from the pandas metadata
    for col_meta in columns_metadata:
        try:
            name = col_meta['field_name']
        except KeyError:
            name = col_meta['name']
        dtype = col_meta['numpy_type']

        if name not in ext_columns and dtype not in _pandas_supported_numpy_types:
            # pandas_dtype is expensive, so avoid doing this for types
            # that are certainly numpy dtypes
            pandas_dtype = _pandas_api.pandas_dtype(dtype)
            if isinstance(pandas_dtype, _pandas_api.extension_dtype):
                if isinstance(pandas_dtype, _pandas_api.pd.StringDtype):
                    # when the metadata indicate to use the string dtype,
                    # ignore this in case:
                    # - it is specified to convert strings / this column to categorical
                    # - the column itself is dictionary encoded and would otherwise be
                    #   converted to categorical
                    if strings_to_categorical or name in categories:
                        continue
                    try:
                        if pa.types.is_dictionary(table.schema.field(name).type):
                            continue
                    except KeyError:
                        pass
                if hasattr(pandas_dtype, "__from_arrow__"):
                    ext_columns[name] = pandas_dtype

    # for pandas 3.0+, use pandas' new default string dtype
    if _pandas_api.uses_string_dtype() and not strings_to_categorical:
        for field in table.schema:
            if field.name not in ext_columns and (
                pa.types.is_string(field.type)
                or pa.types.is_large_string(field.type)
                or pa.types.is_string_view(field.type)
            ) and field.name not in categories:
                ext_columns[field.name] = _pandas_api.pd.StringDtype(na_value=np.nan)

    return ext_columns


def _check_data_column_metadata_consistency(all_columns):
    # It can never be the case in a released version of pyarrow that
    # c['name'] is None *and* 'field_name' is not a key in the column metadata,
    # because the change to allow c['name'] to be None and the change to add
    # 'field_name' are in the same release (0.8.0)
    assert all(
        (c['name'] is None and 'field_name' in c) or c['name'] is not None
        for c in all_columns
    )


def _deserialize_column_index(block_table, all_columns, column_indexes):
    if all_columns:
        columns_name_dict = {
            c.get('field_name', _column_name_to_strings(c['name'])): c['name']
            for c in all_columns
        }
        columns_values = [
            columns_name_dict.get(name, name) for name in block_table.column_names
        ]
    else:
        columns_values = block_table.column_names

    # Construct the base index
    if len(column_indexes) > 1:
        # If we're passed multiple column indexes then evaluate with
        # ast.literal_eval, since the column index values show up as a list of
        # tuples
        columns = _pandas_api.pd.MultiIndex.from_tuples(
            list(map(ast.literal_eval, columns_values)),
            names=[col_index['name'] for col_index in column_indexes],
        )
    else:
        columns = _pandas_api.pd.Index(
            columns_values, name=column_indexes[0]["name"] if column_indexes else None
        )

    # if we're reconstructing the index
    if len(column_indexes) > 0:
        columns = _reconstruct_columns_from_metadata(columns, column_indexes)

    return columns


def _reconstruct_index(table, index_descriptors, all_columns, types_mapper=None):
    # 0. 'field_name' is the name of the column in the arrow Table
    # 1. 'name' is the user-facing name of the column, that is, it came from
    #    pandas
    # 2. 'field_name' and 'name' differ for index columns
    # 3. We fall back on c['name'] for backwards compatibility
    field_name_to_metadata = {
        c.get('field_name', c['name']): c
        for c in all_columns
    }

    # Build up a list of index columns and names while removing those columns
    # from the original table
    index_arrays = []
    index_names = []
    result_table = table
    for descr in index_descriptors:
        if isinstance(descr, str):
            result_table, index_level, index_name = _extract_index_level(
                table, result_table, descr, field_name_to_metadata, types_mapper)
            if index_level is None:
                # ARROW-1883: the serialized index column was not found
                continue
        elif descr['kind'] == 'range':
            index_name = descr['name']
            index_level = _pandas_api.pd.RangeIndex(descr['start'],
                                                    descr['stop'],
                                                    step=descr['step'],
                                                    name=index_name)
            if len(index_level) != len(table):
                # Possibly the result of munged metadata
                continue
        else:
            raise ValueError(f"Unrecognized index kind: {descr['kind']}")
        index_arrays.append(index_level)
        index_names.append(index_name)

    pd = _pandas_api.pd

    # Reconstruct the row index
    if len(index_arrays) > 1:
        index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
    elif len(index_arrays) == 1:
        index = index_arrays[0]
        if not isinstance(index, pd.Index):
            # Box anything that wasn't boxed above
            index = pd.Index(index, name=index_names[0])
    else:
        index = pd.RangeIndex(table.num_rows)

    return result_table, index


def _extract_index_level(table, result_table, field_name,
                         field_name_to_metadata, types_mapper=None):
    logical_name = field_name_to_metadata[field_name]['name']
    index_name = _backwards_compatible_index_name(field_name, logical_name)
    i = table.schema.get_field_index(field_name)

    if i == -1:
        # The serialized index column was removed by the user
        return result_table, None, None

    col = table.column(i)
    index_level = col.to_pandas(types_mapper=types_mapper)
    index_level.name = None
    result_table = result_table.remove_column(
        result_table.schema.get_field_index(field_name)
    )
    return result_table, index_level, index_name


def _backwards_compatible_index_name(raw_name, logical_name):
    """Compute the name of an index column that is compatible with older
    versions of :mod:`pyarrow`.

    Parameters
    ----------
    raw_name : str
    logical_name : str

    Returns
    -------
    result : str

    Notes
    -----
    * Part of :func:`~pyarrow.pandas_compat.table_to_blockmanager`
    """
    # Part of table_to_blockmanager
    if raw_name == logical_name and _is_generated_index_name(raw_name):
        return None
    else:
        return logical_name


def _is_generated_index_name(name):
    pattern = r'^__index_level_\d+__$'
    return re.match(pattern, name) is not None


def get_pandas_logical_type_map():
    global _pandas_logical_type_map

    if not _pandas_logical_type_map:
        _pandas_logical_type_map.update({
            'date': 'datetime64[D]',
            'datetime': 'datetime64[ns]',
            'datetimetz': 'datetime64[ns]',
            'unicode': 'str',
            'bytes': np.bytes_,
            'string': 'str',
            'integer': np.int64,
            'floating': np.float64,
            'decimal': np.object_,
            'empty': np.object_,
        })
    return _pandas_logical_type_map


def _pandas_type_to_numpy_type(pandas_type):
    """Get the numpy dtype that corresponds to a pandas type.

    Parameters
    ----------
    pandas_type : str
        The result of a call to pandas.lib.infer_dtype.

    Returns
    -------
    dtype : np.dtype
        The dtype that corresponds to `pandas_type`.
    """
    pandas_logical_type_map = get_pandas_logical_type_map()
    try:
        return pandas_logical_type_map[pandas_type]
    except KeyError:
        if 'mixed' in pandas_type:
            # catching 'mixed', 'mixed-integer' and 'mixed-integer-float'
            return np.object_
        return np.dtype(pandas_type)


def _reconstruct_columns_from_metadata(columns, column_indexes):
    """Construct a pandas MultiIndex from `columns` and column index metadata
    in `column_indexes`.

    Parameters
    ----------
    columns : List[pd.Index]
        The columns coming from a pyarrow.Table
    column_indexes : List[Dict[str, str]]
        The column index metadata deserialized from the JSON schema metadata
        in a :class:`~pyarrow.Table`.

    Returns
    -------
    result : MultiIndex
        The index reconstructed using `column_indexes` metadata with levels of
        the correct type.

    Notes
    -----
    * Part of :func:`~pyarrow.pandas_compat.table_to_blockmanager`
    """
    pd = _pandas_api.pd
    # Get levels and labels, and provide sane defaults if the index has a
    # single level to avoid if/else spaghetti.
    levels = getattr(columns, 'levels', None) or [columns]
    labels = getattr(columns, 'codes', None) or [None]

    # Convert each level to the dtype provided in the metadata
    levels_dtypes = [
        (level, col_index.get('pandas_type', str(level.dtype)),
         col_index.get('numpy_type', None))
        for level, col_index in zip_longest(
            levels, column_indexes, fillvalue={}
        )
    ]

    new_levels = []
    encoder = operator.methodcaller('encode', 'UTF-8')

    for level, pandas_dtype, numpy_dtype in levels_dtypes:
        dtype = _pandas_type_to_numpy_type(pandas_dtype)
        # Since our metadata is UTF-8 encoded, Python turns things that were
        # bytes into unicode strings when json.loads-ing them. We need to
        # convert them back to bytes to preserve metadata.
        if dtype == np.bytes_:
            level = level.map(encoder)
        # ARROW-13756: if index is timezone aware DataTimeIndex
        elif pandas_dtype == "datetimetz":
            tz = pa.lib.string_to_tzinfo(
                column_indexes[0]['metadata']['timezone'],
                prefer_zoneinfo=_pandas_api.is_ge_v3())
            level = pd.to_datetime(level, utc=True).tz_convert(tz)
            if _pandas_api.is_ge_v3():
                # with pandas 3+, to_datetime returns a unit depending on the string
                # data, so we restore it to the original unit from the metadata
                level = level.as_unit(np.datetime_data(numpy_dtype)[0])
        # GH-41503: if the column index was decimal, restore to decimal
        elif pandas_dtype == "decimal":
            level = _pandas_api.pd.Index([decimal.Decimal(i) for i in level])
        elif (
            level.dtype == "str" and numpy_dtype == "object"
            and ("mixed" in pandas_dtype or pandas_dtype in ["unicode", "string"])
        ):
            # the metadata indicate that the original dataframe used object dtype,
            # but ignore this and keep string dtype if:
            # - the original columns used mixed types -> we don't attempt to faithfully
            #   roundtrip in this case, but keep the column names as strings
            # - the original columns were inferred to be strings but stored in object
            #   dtype -> we don't restore the object dtype because all metadata
            #   generated using pandas < 3 will have this case by default, and
            #   for pandas >= 3 we want to use the default string dtype for .columns
            new_levels.append(level)
            continue
        elif level.dtype != dtype:
            level = level.astype(dtype)
        # ARROW-9096: if original DataFrame was upcast we keep that
        if level.dtype != numpy_dtype and pandas_dtype != "datetimetz":
            level = level.astype(numpy_dtype)

        new_levels.append(level)

    if len(new_levels) > 1:
        return pd.MultiIndex(new_levels, labels, names=columns.names)
    else:
        return pd.Index(new_levels[0], dtype=new_levels[0].dtype, name=columns.name)


def _add_any_metadata(table, pandas_metadata):
    modified_columns = {}
    modified_fields = {}

    schema = table.schema

    index_columns = pandas_metadata['index_columns']
    # only take index columns into account if they are an actual table column
    index_columns = [idx_col for idx_col in index_columns
                     if isinstance(idx_col, str)]
    n_index_levels = len(index_columns)
    n_columns = len(pandas_metadata['columns']) - n_index_levels

    # Add time zones
    for i, col_meta in enumerate(pandas_metadata['columns']):

        raw_name = col_meta.get('field_name')
        if not raw_name:
            # deal with metadata written with arrow < 0.8 or fastparquet
            raw_name = col_meta['name']
            if i >= n_columns:
                # index columns
                raw_name = index_columns[i - n_columns]
            if raw_name is None:
                raw_name = 'None'

        idx = schema.get_field_index(raw_name)
        if idx != -1:
            if col_meta['pandas_type'] == 'datetimetz':
                col = table[idx]
                if not isinstance(col.type, pa.lib.TimestampType):
                    continue
                metadata = col_meta['metadata']
                if not metadata:
                    continue
                metadata_tz = metadata.get('timezone')
                if metadata_tz and metadata_tz != col.type.tz:
                    converted = col.to_pandas()
                    tz_aware_type = pa.timestamp('ns', tz=metadata_tz)
                    with_metadata = pa.Array.from_pandas(converted,
                                                         type=tz_aware_type)

                    modified_fields[idx] = pa.field(schema[idx].name,
                                                    tz_aware_type)
                    modified_columns[idx] = with_metadata

    if len(modified_columns) > 0:
        columns = []
        fields = []
        for i in range(len(table.schema)):
            if i in modified_columns:
                columns.append(modified_columns[i])
                fields.append(modified_fields[i])
            else:
                columns.append(table[i])
                fields.append(table.schema[i])
        return pa.Table.from_arrays(columns, schema=pa.schema(fields))
    else:
        return table


# ----------------------------------------------------------------------
# Helper functions used in lib


def make_tz_aware(series, tz):
    """
    Make a datetime64 Series timezone-aware for the given tz
    """
    tz = pa.lib.string_to_tzinfo(tz, prefer_zoneinfo=_pandas_api.is_ge_v3())
    series = (series.dt.tz_localize('utc')
                    .dt.tz_convert(tz))
    return series
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
FileSystem abstraction to interact with various local and remote filesystems.
"""

from pyarrow.util import _is_path_like, _stringify_path

from pyarrow._fs import (  # noqa
    FileSelector,
    FileType,
    FileInfo,
    FileSystem,
    LocalFileSystem,
    SubTreeFileSystem,
    _MockFileSystem,
    FileSystemHandler,
    PyFileSystem,
    _copy_files,
    _copy_files_selector,
)

# For backward compatibility.
FileStats = FileInfo

_not_imported = []
try:
    from pyarrow._azurefs import AzureFileSystem  # noqa
except ImportError:
    _not_imported.append("AzureFileSystem")

try:
    from pyarrow._hdfs import HadoopFileSystem  # noqa
except ImportError:
    _not_imported.append("HadoopFileSystem")

try:
    from pyarrow._gcsfs import GcsFileSystem  # noqa
except ImportError:
    _not_imported.append("GcsFileSystem")

try:
    from pyarrow._s3fs import (  # noqa
        AwsDefaultS3RetryStrategy, AwsStandardS3RetryStrategy,
        S3FileSystem, S3LogLevel, S3RetryStrategy, ensure_s3_initialized,
        finalize_s3, ensure_s3_finalized, initialize_s3, resolve_s3_region)
except ImportError:
    _not_imported.append("S3FileSystem")
else:
    # GH-38364: we don't initialize S3 eagerly as that could lead
    # to crashes at shutdown even when S3 isn't used.
    # Instead, S3 is initialized lazily using `ensure_s3_initialized`
    # in assorted places.
    import atexit
    atexit.register(ensure_s3_finalized)


def __getattr__(name):
    if name in _not_imported:
        raise ImportError(
            "The pyarrow installation is not built with support for "
            f"'{name}'"
        )

    raise AttributeError(
        f"module 'pyarrow.fs' has no attribute '{name}'"
    )


def _ensure_filesystem(filesystem, *, use_mmap=False):
    if isinstance(filesystem, FileSystem):
        return filesystem
    elif isinstance(filesystem, str):
        # create a filesystem from a URI string, note that the `path` part of the URI
        # is treated as a prefix if specified, so the filesystem is wrapped in a
        # SubTreeFileSystem
        if use_mmap:
            raise ValueError(
                "Specifying to use memory mapping not supported for "
                "filesystem specified as an URI string"
            )
        fs, path = FileSystem.from_uri(filesystem)
        prefix = fs.normalize_path(path)
        if prefix:
            # validate that the prefix is pointing to a directory
            prefix_info = fs.get_file_info([prefix])[0]
            if prefix_info.type != FileType.Directory:
                raise ValueError(
                    "The path component of the filesystem URI must point to a "
                    f"directory but it has a type: `{prefix_info.type.name}`. The path "
                    f"component is `{prefix_info.path}` and the given filesystem URI "
                    f"is `{filesystem}`"
                )
            fs = SubTreeFileSystem(prefix, fs)
        return fs
    else:
        # handle fsspec-compatible filesystems
        try:
            import fsspec
        except ImportError:
            pass
        else:
            if isinstance(filesystem, fsspec.AbstractFileSystem):
                if type(filesystem).__name__ == 'LocalFileSystem':
                    # In case its a simple LocalFileSystem, use native arrow one
                    return LocalFileSystem(use_mmap=use_mmap)
                return PyFileSystem(FSSpecHandler(filesystem))

        raise TypeError(
            f"Unrecognized filesystem: {type(filesystem)}. `filesystem` argument must "
            "be a FileSystem instance or a valid file system URI"
        )


def _resolve_filesystem_and_path(path, filesystem=None, *, memory_map=False):
    """
    Return filesystem/path from path which could be an URI or a plain
    filesystem path or a combination of fsspec protocol and URI.
    """
    if not _is_path_like(path):
        if filesystem is not None:
            raise ValueError(
                "'filesystem' passed but the specified path is file-like, so"
                " there is nothing to open with 'filesystem'."
            )
        return filesystem, path

    if filesystem is not None:
        filesystem = _ensure_filesystem(filesystem, use_mmap=memory_map)
        if isinstance(filesystem, LocalFileSystem):
            path = _stringify_path(path)
        elif not isinstance(path, str):
            raise TypeError(
                "Expected string path; path-like objects are only allowed "
                "with a local filesystem"
            )
        path = filesystem.normalize_path(path)
        return filesystem, path

    path = _stringify_path(path)

    # if filesystem is not given, try to automatically determine one
    # first check if the file exists as a local (relative) file path
    # if not then try to parse the path as an URI
    filesystem = LocalFileSystem(use_mmap=memory_map)

    try:
        file_info = filesystem.get_file_info(path)
    except ValueError:  # ValueError means path is likely an URI
        file_info = None
        exists_locally = False
    else:
        exists_locally = (file_info.type != FileType.NotFound)

    # if the file or directory doesn't exists locally, then assume that
    # the path is an URI describing the file system as well
    if not exists_locally:
        try:
            filesystem, path = FileSystem.from_uri(path)
        except ValueError as e:
            msg = str(e)
            if "empty scheme" in msg or "Cannot parse URI" in msg:
                # neither an URI nor a locally existing path, so assume that
                # local path was given and propagate a nicer file not found
                # error instead of a more confusing scheme parsing error
                pass
            else:
                raise e
    else:
        path = filesystem.normalize_path(path)

    return filesystem, path


def copy_files(source, destination,
               source_filesystem=None, destination_filesystem=None,
               *, chunk_size=1024*1024, use_threads=True):
    """
    Copy files between FileSystems.

    This functions allows you to recursively copy directories of files from
    one file system to another, such as from S3 to your local machine.

    Parameters
    ----------
    source : string
        Source file path or URI to a single file or directory.
        If a directory, files will be copied recursively from this path.
    destination : string
        Destination file path or URI. If `source` is a file, `destination`
        is also interpreted as the destination file (not directory).
        Directories will be created as necessary.
    source_filesystem : FileSystem, optional
        Source filesystem, needs to be specified if `source` is not a URI,
        otherwise inferred.
    destination_filesystem : FileSystem, optional
        Destination filesystem, needs to be specified if `destination` is not
        a URI, otherwise inferred.
    chunk_size : int, default 1MB
        The maximum size of block to read before flushing to the
        destination file. A larger chunk_size will use more memory while
        copying but may help accommodate high latency FileSystems.
    use_threads : bool, default True
        Whether to use multiple threads to accelerate copying.

    Examples
    --------
    Inspect an S3 bucket's files:

    >>> s3, path = fs.FileSystem.from_uri(
    ...            "s3://registry.opendata.aws/roda/ndjson/")
    >>> selector = fs.FileSelector(path)
    >>> s3.get_file_info(selector)
    [<FileInfo for 'registry.opendata.aws/roda/ndjson/index.ndjson':...]

    Copy one file from S3 bucket to a local directory:

    >>> fs.copy_files("s3://registry.opendata.aws/roda/ndjson/index.ndjson",
    ...               f"file:///{local_path}/index_copy.ndjson")

    >>> fs.LocalFileSystem().get_file_info(str(local_path)+
    ...                                    '/index_copy.ndjson')
    <FileInfo for '.../index_copy.ndjson': type=FileType.File, size=...>

    Copy file using a FileSystem object:

    >>> fs.copy_files("registry.opendata.aws/roda/ndjson/index.ndjson",
    ...               f"file:///{local_path}/index_copy.ndjson",
    ...               source_filesystem=fs.S3FileSystem())
    """
    source_fs, source_path = _resolve_filesystem_and_path(
        source, source_filesystem
    )
    destination_fs, destination_path = _resolve_filesystem_and_path(
        destination, destination_filesystem
    )

    file_info = source_fs.get_file_info(source_path)
    if file_info.type == FileType.Directory:
        source_sel = FileSelector(source_path, recursive=True)
        _copy_files_selector(source_fs, source_sel,
                             destination_fs, destination_path,
                             chunk_size, use_threads)
    else:
        _copy_files(source_fs, source_path,
                    destination_fs, destination_path,
                    chunk_size, use_threads)


class FSSpecHandler(FileSystemHandler):
    """
    Handler for fsspec-based Python filesystems.

    https://filesystem-spec.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    fs : FSSpec-compliant filesystem instance

    Examples
    --------
    >>> PyFileSystem(FSSpecHandler(fsspec_fs)) # doctest: +SKIP
    """

    def __init__(self, fs):
        self.fs = fs

    def __eq__(self, other):
        if isinstance(other, FSSpecHandler):
            return self.fs == other.fs
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FSSpecHandler):
            return self.fs != other.fs
        return NotImplemented

    def get_type_name(self):
        protocol = self.fs.protocol
        if isinstance(protocol, list):
            protocol = protocol[0]
        return f"fsspec+{protocol}"

    def normalize_path(self, path):
        return path

    @staticmethod
    def _create_file_info(path, info):
        size = info["size"]
        if info["type"] == "file":
            ftype = FileType.File
        elif info["type"] == "directory":
            ftype = FileType.Directory
            # some fsspec filesystems include a file size for directories
            size = None
        else:
            ftype = FileType.Unknown
        return FileInfo(path, ftype, size=size, mtime=info.get("mtime", None))

    def get_file_info(self, paths):
        infos = []
        for path in paths:
            try:
                info = self.fs.info(path)
            except FileNotFoundError:
                infos.append(FileInfo(path, FileType.NotFound))
            else:
                infos.append(self._create_file_info(path, info))
        return infos

    def get_file_info_selector(self, selector):
        if not self.fs.isdir(selector.base_dir):
            if self.fs.exists(selector.base_dir):
                raise NotADirectoryError(selector.base_dir)
            else:
                if selector.allow_not_found:
                    return []
                else:
                    raise FileNotFoundError(selector.base_dir)

        if selector.recursive:
            maxdepth = None
        else:
            maxdepth = 1

        infos = []
        selected_files = self.fs.find(
            selector.base_dir, maxdepth=maxdepth, withdirs=True, detail=True
        )
        for path, info in selected_files.items():
            _path = path.strip("/")
            base_dir = selector.base_dir.strip("/")
            # Need to exclude base directory from selected files if present
            # (fsspec filesystems, see GH-37555)
            if _path != base_dir:
                infos.append(self._create_file_info(path, info))

        return infos

    def create_dir(self, path, recursive):
        # mkdir also raises FileNotFoundError when base directory is not found
        try:
            self.fs.mkdir(path, create_parents=recursive)
        except FileExistsError:
            pass

    def delete_dir(self, path):
        self.fs.rm(path, recursive=True)

    def _delete_dir_contents(self, path, missing_dir_ok):
        try:
            subpaths = self.fs.listdir(path, detail=False)
        except FileNotFoundError:
            if missing_dir_ok:
                return
            raise
        for subpath in subpaths:
            if self.fs.isdir(subpath):
                self.fs.rm(subpath, recursive=True)
            elif self.fs.isfile(subpath):
                self.fs.rm(subpath)

    def delete_dir_contents(self, path, missing_dir_ok):
        if path.strip("/") == "":
            raise ValueError(
                "delete_dir_contents called on path '", path, "'")
        self._delete_dir_contents(path, missing_dir_ok)

    def delete_root_dir_contents(self):
        self._delete_dir_contents("/", False)

    def delete_file(self, path):
        # fs.rm correctly raises IsADirectoryError when `path` is a directory
        # instead of a file and `recursive` is not set to True
        if not self.fs.exists(path):
            raise FileNotFoundError(path)
        self.fs.rm(path)

    def move(self, src, dest):
        self.fs.mv(src, dest, recursive=True)

    def copy_file(self, src, dest):
        # fs.copy correctly raises IsADirectoryError when `src` is a directory
        # instead of a file
        self.fs.copy(src, dest)

    # TODO can we read/pass metadata (e.g. Content-Type) in the methods below?

    def open_input_stream(self, path):
        from pyarrow import PythonFile

        if not self.fs.isfile(path):
            raise FileNotFoundError(path)

        return PythonFile(self.fs.open(path, mode="rb"), mode="r")

    def open_input_file(self, path):
        from pyarrow import PythonFile

        if not self.fs.isfile(path):
            raise FileNotFoundError(path)

        return PythonFile(self.fs.open(path, mode="rb"), mode="r")

    def open_output_stream(self, path, metadata):
        from pyarrow import PythonFile

        return PythonFile(self.fs.open(path, mode="wb"), mode="w")

    def open_append_stream(self, path, metadata):
        from pyarrow import PythonFile

        return PythonFile(self.fs.open(path, mode="ab"), mode="w")
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Arrow file and stream reader/writer classes, and other messaging tools

import os

import pyarrow as pa

from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
                         Message, MessageReader,
                         RecordBatchReader, _ReadPandasMixin,
                         MetadataVersion, Alignment,
                         read_message, read_record_batch, read_schema,
                         read_tensor, write_tensor,
                         get_record_batch_size, get_tensor_size)
import pyarrow.lib as lib


class RecordBatchStreamReader(lib._RecordBatchStreamReader):
    """
    Reader for the Arrow streaming binary format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC deserialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, *, options=None, memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, options=options, memory_pool=memory_pool)


_ipc_writer_class_doc = """\
Parameters
----------
sink : str, pyarrow.NativeFile, or file-like Python object
    Either a file path, or a writable file object.
schema : pyarrow.Schema
    The Arrow schema for data to be written to the file.
options : pyarrow.ipc.IpcWriteOptions
    Options for IPC serialization.

    If None, default values will be used: the legacy format will not
    be used unless overridden by setting the environment variable
    ARROW_PRE_0_15_IPC_FORMAT=1, and the V5 metadata version will be
    used unless overridden by setting the environment variable
    ARROW_PRE_1_0_METADATA_VERSION=1."""


_ipc_file_writer_class_doc = (
    _ipc_writer_class_doc
    + "\n"
    + """\
metadata : dict | pyarrow.KeyValueMetadata, optional
    Key/value pairs (both must be bytes-like) that will be stored
    in the file footer and are retrievable via
    pyarrow.ipc.open_file(...).metadata."""
)


class RecordBatchStreamWriter(lib._RecordBatchStreamWriter):
    __doc__ = f"""Writer for the Arrow streaming binary format

{_ipc_writer_class_doc}"""

    def __init__(self, sink, schema, *, options=None):
        options = _get_legacy_format_default(options)
        self._open(sink, schema, options=options)


class RecordBatchFileReader(lib._RecordBatchFileReader):
    """
    Class for reading Arrow record batch data from the Arrow binary file format

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    footer_offset : int, default None
        If the file is embedded in some larger file, this is the byte offset to
        the very end of the file data
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, footer_offset=None, *, options=None,
                 memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, footer_offset=footer_offset,
                   options=options, memory_pool=memory_pool)


class RecordBatchFileWriter(lib._RecordBatchFileWriter):

    __doc__ = f"""Writer to create the Arrow binary file format

{_ipc_file_writer_class_doc}"""

    def __init__(self, sink, schema, *, options=None, metadata=None):
        options = _get_legacy_format_default(options)
        self._open(sink, schema, options=options, metadata=metadata)


def _get_legacy_format_default(options):
    if options:
        if not isinstance(options, IpcWriteOptions):
            raise TypeError(f"expected IpcWriteOptions, got {type(options)}")
        return options

    metadata_version = MetadataVersion.V5
    use_legacy_format = \
        bool(int(os.environ.get('ARROW_PRE_0_15_IPC_FORMAT', '0')))
    if bool(int(os.environ.get('ARROW_PRE_1_0_METADATA_VERSION', '0'))):
        metadata_version = MetadataVersion.V4
    return IpcWriteOptions(use_legacy_format=use_legacy_format,
                           metadata_version=metadata_version)


def _ensure_default_ipc_read_options(options):
    if options and not isinstance(options, IpcReadOptions):
        raise TypeError(f"expected IpcReadOptions, got {type(options)}")
    return options or IpcReadOptions()


def new_stream(sink, schema, *, options=None):
    return RecordBatchStreamWriter(sink, schema,
                                   options=options)


new_stream.__doc__ = f"""\
Create an Arrow columnar IPC stream writer instance

{_ipc_writer_class_doc}

Returns
-------
writer : RecordBatchStreamWriter
    A writer for the given sink
"""


def open_stream(source, *, options=None, memory_pool=None):
    """
    Create reader for Arrow streaming format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.

    Returns
    -------
    reader : RecordBatchStreamReader
        A reader for the given source
    """
    return RecordBatchStreamReader(source, options=options,
                                   memory_pool=memory_pool)


def new_file(sink, schema, *, options=None, metadata=None):
    return RecordBatchFileWriter(sink, schema, options=options, metadata=metadata)


new_file.__doc__ = f"""\
Create an Arrow columnar IPC file writer instance

{_ipc_file_writer_class_doc}

Returns
-------
writer : RecordBatchFileWriter
    A writer for the given sink
"""


def open_file(source, footer_offset=None, *, options=None, memory_pool=None):
    """
    Create reader for Arrow file format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
    footer_offset : int, default None
        If the file is embedded in some larger file, this is the byte offset to
        the very end of the file data.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.

    Returns
    -------
    reader : RecordBatchFileReader
        A reader for the given source
    """
    return RecordBatchFileReader(
        source, footer_offset=footer_offset,
        options=options, memory_pool=memory_pool)


def serialize_pandas(df, *, nthreads=None, preserve_index=None):
    """
    Serialize a pandas DataFrame into a buffer protocol compatible object.

    Parameters
    ----------
    df : pandas.DataFrame
    nthreads : int, default None
        Number of threads to use for conversion to Arrow, default all CPUs.
    preserve_index : bool, default None
        The default of None will store the index as a column, except for
        RangeIndex which is stored as metadata only. If True, always
        preserve the pandas index data as a column. If False, no index
        information is saved and the result will have a default RangeIndex.

    Returns
    -------
    buf : buffer
        An object compatible with the buffer protocol.
    """
    batch = pa.RecordBatch.from_pandas(df, nthreads=nthreads,
                                       preserve_index=preserve_index)
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def deserialize_pandas(buf, *, use_threads=True):
    """Deserialize a buffer protocol compatible object into a pandas DataFrame.

    Parameters
    ----------
    buf : buffer
        An object compatible with the buffer protocol.
    use_threads : bool, default True
        Whether to parallelize the conversion using multiple threads.

    Returns
    -------
    df : pandas.DataFrame
        The buffer deserialized as pandas DataFrame
    """
    buffer_reader = pa.BufferReader(buf)
    with pa.RecordBatchStreamReader(buffer_reader) as reader:
        table = reader.read_all()
    return table.to_pandas(use_threads=use_threads)
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Miscellaneous utility code

import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings


_DEPR_MSG = (
    "pyarrow.{} is deprecated as of {}, please use pyarrow.{} instead."
)


def doc(*docstrings, **params):
    """
    A decorator that takes docstring templates, concatenates them, and finally
    performs string substitution on them.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track of the original docstring template for potential future use.
    If the docstring is a template, it will be saved as a string.
    Otherwise, it will be saved as a callable and the docstring will be obtained via
    the __doc__ attribute.
    This decorator cannot be used on Cython classes due to a CPython constraint,
    which enforces the __doc__ attribute to be read-only.
    See https://github.com/python/cpython/issues/91309

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be prepended in order
        before the default docstring under the callable.
    **params
        The key/value pairs used to format the docstring template.
    """

    def decorator(decorated):
        docstring_components = []

        # collect docstrings and docstring templates
        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # append the callable's docstring last
        if decorated.__doc__:
            docstring_components.append(textwrap.dedent(decorated.__doc__))

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else textwrap.dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        decorated._docstring_components = (
            docstring_components
        )
        return decorated

    return decorator


def _deprecate_api(old_name, new_name, api, next_version, type=FutureWarning):
    msg = _DEPR_MSG.format(old_name, next_version, new_name)

    def wrapper(*args, **kwargs):
        warnings.warn(msg, type)
        return api(*args, **kwargs)
    return wrapper


def _deprecate_class(old_name, new_class, next_version,
                     instancecheck=True):
    """
    Raise warning if a deprecated class is used in an isinstance check.
    """
    class _DeprecatedMeta(type):
        def __instancecheck__(self, other):
            warnings.warn(
                _DEPR_MSG.format(old_name, next_version, new_class.__name__),
                FutureWarning,
                stacklevel=2
            )
            return isinstance(other, new_class)

    return _DeprecatedMeta(old_name, (new_class,), {})


def _is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _is_path_like(path):
    return isinstance(path, str) or hasattr(path, '__fspath__')


def _stringify_path(path):
    """
    Convert *path* to a string or unicode path if possible.
    """
    if isinstance(path, str):
        return os.path.expanduser(path)

    # checking whether path implements the filesystem protocol
    try:
        return os.path.expanduser(path.__fspath__())
    except AttributeError:
        pass

    raise TypeError("not a path-like object")


def product(seq):
    """
    Return a product of sequence items.
    """
    return functools.reduce(lambda a, b: a*b, seq, 1)


def get_contiguous_span(shape, strides, itemsize):
    """
    Return a contiguous span of N-D array data.

    Parameters
    ----------
    shape : tuple
    strides : tuple
    itemsize : int
      Specify array shape data

    Returns
    -------
    start, end : int
      The span end points.
    """
    if not strides:
        start = 0
        end = itemsize * product(shape)
    else:
        start = 0
        end = itemsize
        for i, dim in enumerate(shape):
            if dim == 0:
                start = end = 0
                break
            stride = strides[i]
            if stride > 0:
                end += stride * (dim - 1)
            elif stride < 0:
                start += stride * (dim - 1)
        if end - start != itemsize * product(shape):
            raise ValueError('array data is non-contiguous')
    return start, end


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with contextlib.closing(sock) as sock:
        sock.bind(('', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def guid():
    from uuid import uuid4
    return uuid4().hex


def _break_traceback_cycle_from_frame(frame):
    # Clear local variables in all inner frames, so as to break the
    # reference cycle.
    this_frame = sys._getframe(0)
    refs = gc.get_referrers(frame)
    while refs:
        for frame in refs:
            if frame is not this_frame and isinstance(frame, types.FrameType):
                break
        else:
            # No frame found in referrers (finished?)
            break
        refs = None
        # Clear the frame locals, to try and break the cycle (it is
        # somewhere along the chain of execution frames).
        frame.clear()
        # To visit the inner frame, we need to find it among the
        # referrers of this frame (while `frame.f_back` would let
        # us visit the outer frame).
        refs = gc.get_referrers(frame)
    refs = frame = this_frame = None


def _download_urllib(url, out_path):
    from urllib.request import urlopen, Request
    req = Request(url, headers={'User-Agent': 'pyarrow'})
    with urlopen(req) as response:
        with open(out_path, 'wb') as f:
            f.write(response.read())


def _download_requests(url, out_path):
    import requests
    with requests.get(url) as response:
        with open(out_path, 'wb') as f:
            f.write(response.content)


# TODO(GH-48593): Remove when libc++ supports std::chrono timezone
# https://github.com/apache/arrow/issues/48593
def download_tzdata_on_windows():
    r"""
    Download and extract latest IANA timezone database into the
    location expected by Arrow which is %USERPROFILE%\Downloads\tzdata.

    .. deprecated:: 24.0.0
       This function is deprecated and will be removed in a future version.
       PyArrow now uses the operating system's timezone database on Windows.
    """

    warnings.warn(
        "pyarrow.util.download_tzdata_on_windows is deprecated as of 24.0.0 "
        "and will be removed in a future version. PyArrow now uses the "
        "operating system's timezone database on most Windows builds.",
        FutureWarning,
        stacklevel=2
    )
    if sys.platform != 'win32':
        raise TypeError(f"Timezone database is already provided by {sys.platform}")

    import tarfile

    tzdata_url = "https://data.iana.org/time-zones/tzdata-latest.tar.gz"
    tzdata_path = os.path.expandvars(r"%USERPROFILE%\Downloads\tzdata")
    tzdata_compressed_path = os.path.join(tzdata_path, "tzdata.tar.gz")
    windows_zones_url = "https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/windowsZones.xml"  # noqa
    windows_zones_path = os.path.join(tzdata_path, "windowsZones.xml")
    os.makedirs(tzdata_path, exist_ok=True)

    # Try to download the files with requests and then fall back to urllib. This
    # works around possible issues in certain older environment (GH-45295)
    try:
        import requests  # noqa: F401
        download_fn = _download_requests
    except ImportError:
        download_fn = _download_urllib

    download_fn(tzdata_url, tzdata_compressed_path)
    download_fn(windows_zones_url, windows_zones_path)

    assert os.path.exists(tzdata_compressed_path)
    assert os.path.exists(windows_zones_path)

    tarfile.open(tzdata_compressed_path).extractall(tzdata_path)
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Tools for dealing with Arrow type metadata in Python


from enum import IntEnum

from pyarrow.lib import (is_boolean_value,  # noqa
                         is_integer_value,
                         is_float_value)

import pyarrow.lib as lib
from pyarrow.util import doc


_SIGNED_INTEGER_TYPES = {lib.Type_INT8, lib.Type_INT16, lib.Type_INT32,
                         lib.Type_INT64}
_UNSIGNED_INTEGER_TYPES = {lib.Type_UINT8, lib.Type_UINT16, lib.Type_UINT32,
                           lib.Type_UINT64}
_INTEGER_TYPES = _SIGNED_INTEGER_TYPES | _UNSIGNED_INTEGER_TYPES
_FLOATING_TYPES = {lib.Type_HALF_FLOAT, lib.Type_FLOAT, lib.Type_DOUBLE}
_DECIMAL_TYPES = {lib.Type_DECIMAL32, lib.Type_DECIMAL64, lib.Type_DECIMAL128,
                  lib.Type_DECIMAL256}
_DATE_TYPES = {lib.Type_DATE32, lib.Type_DATE64}
_TIME_TYPES = {lib.Type_TIME32, lib.Type_TIME64}
_INTERVAL_TYPES = {lib.Type_INTERVAL_MONTH_DAY_NANO}
_TEMPORAL_TYPES = ({lib.Type_TIMESTAMP,
                    lib.Type_DURATION} | _TIME_TYPES | _DATE_TYPES |
                   _INTERVAL_TYPES)
_UNION_TYPES = {lib.Type_SPARSE_UNION, lib.Type_DENSE_UNION}
_NESTED_TYPES = {lib.Type_LIST, lib.Type_FIXED_SIZE_LIST, lib.Type_LARGE_LIST,
                 lib.Type_LIST_VIEW, lib.Type_LARGE_LIST_VIEW,
                 lib.Type_STRUCT, lib.Type_MAP} | _UNION_TYPES


class TypesEnum(IntEnum):
    """
    An Enum that maps constant values to data types.
    Exposes the underlying data types representation for type checking purposes.
    Note that some of the types listed here are not supported by PyArrow yet:
    INTERVAL_MONTHS and INTERVAL_DAY_TIME.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from pyarrow.types import TypesEnum
    >>> int8_field = pa.field('int8_field', pa.int8())
    >>> int8_field.type.id == TypesEnum.INT8
    True

    >>> fixed_size_list = pa.list_(pa.uint16(), 3)
    >>> fixed_size_list.id == TypesEnum.LIST
    False

    >>> fixed_size_list.id == TypesEnum.FIXED_SIZE_LIST
    True
    """

    NA = lib.Type_NA
    BOOL = lib.Type_BOOL

    INT8 = lib.Type_INT8
    INT16 = lib.Type_INT16
    INT32 = lib.Type_INT32
    INT64 = lib.Type_INT64

    UINT8 = lib.Type_UINT8
    UINT16 = lib.Type_UINT16
    UINT32 = lib.Type_UINT32
    UINT64 = lib.Type_UINT64

    HALF_FLOAT = lib.Type_HALF_FLOAT
    FLOAT = lib.Type_FLOAT
    DOUBLE = lib.Type_DOUBLE

    BINARY = lib.Type_BINARY
    BINARY_VIEW = lib.Type_BINARY_VIEW
    LARGE_BINARY = lib.Type_LARGE_BINARY
    STRING = lib.Type_STRING
    STRING_VIEW = lib.Type_STRING_VIEW
    LARGE_STRING = lib.Type_LARGE_STRING
    FIXED_SIZE_BINARY = lib.Type_FIXED_SIZE_BINARY

    DECIMAL32 = lib.Type_DECIMAL32
    DECIMAL64 = lib.Type_DECIMAL64
    DECIMAL128 = lib.Type_DECIMAL128
    DECIMAL256 = lib.Type_DECIMAL256

    LIST = lib.Type_LIST
    LARGE_LIST = lib.Type_LARGE_LIST
    LIST_VIEW = lib.Type_LIST_VIEW
    LARGE_LIST_VIEW = lib.Type_LARGE_LIST_VIEW
    MAP = lib.Type_MAP
    FIXED_SIZE_LIST = lib.Type_FIXED_SIZE_LIST

    STRUCT = lib.Type_STRUCT
    SPARSE_UNION = lib.Type_SPARSE_UNION
    DENSE_UNION = lib.Type_DENSE_UNION
    RUN_END_ENCODED = lib.Type_RUN_END_ENCODED

    DATE32 = lib.Type_DATE32
    DATE64 = lib.Type_DATE64
    TIME32 = lib.Type_TIME32
    TIME64 = lib.Type_TIME64
    TIMESTAMP = lib.Type_TIMESTAMP

    INTERVAL_MONTHS = lib.Type_INTERVAL_MONTHS
    INTERVAL_DAY_TIME = lib.Type_INTERVAL_DAY_TIME
    INTERVAL_MONTH_DAY_NANO = lib.Type_INTERVAL_MONTH_DAY_NANO

    DURATION = lib.Type_DURATION
    DICTIONARY = lib.Type_DICTIONARY


@doc(datatype="null")
def is_null(t):
    """
    Return True if value is an instance of type: {datatype}.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_NA


@doc(is_null, datatype="boolean")
def is_boolean(t):
    return t.id == lib.Type_BOOL


@doc(is_null, datatype="any integer")
def is_integer(t):
    return t.id in _INTEGER_TYPES


@doc(is_null, datatype="signed integer")
def is_signed_integer(t):
    return t.id in _SIGNED_INTEGER_TYPES


@doc(is_null, datatype="unsigned integer")
def is_unsigned_integer(t):
    return t.id in _UNSIGNED_INTEGER_TYPES


@doc(is_null, datatype="int8")
def is_int8(t):
    return t.id == lib.Type_INT8


@doc(is_null, datatype="int16")
def is_int16(t):
    return t.id == lib.Type_INT16


@doc(is_null, datatype="int32")
def is_int32(t):
    return t.id == lib.Type_INT32


@doc(is_null, datatype="int64")
def is_int64(t):
    return t.id == lib.Type_INT64


@doc(is_null, datatype="uint8")
def is_uint8(t):
    return t.id == lib.Type_UINT8


@doc(is_null, datatype="uint16")
def is_uint16(t):
    return t.id == lib.Type_UINT16


@doc(is_null, datatype="uint32")
def is_uint32(t):
    return t.id == lib.Type_UINT32


@doc(is_null, datatype="uint64")
def is_uint64(t):
    return t.id == lib.Type_UINT64


@doc(is_null, datatype="floating point numeric")
def is_floating(t):
    return t.id in _FLOATING_TYPES


@doc(is_null, datatype="float16 (half-precision)")
def is_float16(t):
    return t.id == lib.Type_HALF_FLOAT


@doc(is_null, datatype="float32 (single precision)")
def is_float32(t):
    return t.id == lib.Type_FLOAT


@doc(is_null, datatype="float64 (double precision)")
def is_float64(t):
    return t.id == lib.Type_DOUBLE


@doc(is_null, datatype="list")
def is_list(t):
    return t.id == lib.Type_LIST


@doc(is_null, datatype="large list")
def is_large_list(t):
    return t.id == lib.Type_LARGE_LIST


@doc(is_null, datatype="fixed size list")
def is_fixed_size_list(t):
    return t.id == lib.Type_FIXED_SIZE_LIST


@doc(is_null, datatype="list view")
def is_list_view(t):
    return t.id == lib.Type_LIST_VIEW


@doc(is_null, datatype="large list view")
def is_large_list_view(t):
    return t.id == lib.Type_LARGE_LIST_VIEW


@doc(is_null, datatype="struct")
def is_struct(t):
    return t.id == lib.Type_STRUCT


@doc(is_null, datatype="union")
def is_union(t):
    return t.id in _UNION_TYPES


@doc(is_null, datatype="nested type")
def is_nested(t):
    return t.id in _NESTED_TYPES


@doc(is_null, datatype="run-end encoded")
def is_run_end_encoded(t):
    return t.id == lib.Type_RUN_END_ENCODED


@doc(is_null, datatype="date, time, timestamp or duration")
def is_temporal(t):
    return t.id in _TEMPORAL_TYPES


@doc(is_null, datatype="timestamp")
def is_timestamp(t):
    return t.id == lib.Type_TIMESTAMP


@doc(is_null, datatype="duration")
def is_duration(t):
    return t.id == lib.Type_DURATION


@doc(is_null, datatype="time")
def is_time(t):
    return t.id in _TIME_TYPES


@doc(is_null, datatype="time32")
def is_time32(t):
    return t.id == lib.Type_TIME32


@doc(is_null, datatype="time64")
def is_time64(t):
    return t.id == lib.Type_TIME64


@doc(is_null, datatype="variable-length binary")
def is_binary(t):
    return t.id == lib.Type_BINARY


@doc(is_null, datatype="large variable-length binary")
def is_large_binary(t):
    return t.id == lib.Type_LARGE_BINARY


@doc(method="is_string")
def is_unicode(t):
    """
    Alias for {method}.

    Parameters
    ----------
    t : DataType
    """
    return is_string(t)


@doc(is_null, datatype="string (utf8 unicode)")
def is_string(t):
    return t.id == lib.Type_STRING


@doc(is_unicode, method="is_large_string")
def is_large_unicode(t):
    return is_large_string(t)


@doc(is_null, datatype="large string (utf8 unicode)")
def is_large_string(t):
    return t.id == lib.Type_LARGE_STRING


@doc(is_null, datatype="fixed size binary")
def is_fixed_size_binary(t):
    return t.id == lib.Type_FIXED_SIZE_BINARY


@doc(is_null, datatype="variable-length binary view")
def is_binary_view(t):
    return t.id == lib.Type_BINARY_VIEW


@doc(is_null, datatype="variable-length string (utf-8) view")
def is_string_view(t):
    return t.id == lib.Type_STRING_VIEW


@doc(is_null, datatype="date")
def is_date(t):
    return t.id in _DATE_TYPES


@doc(is_null, datatype="date32 (days)")
def is_date32(t):
    return t.id == lib.Type_DATE32


@doc(is_null, datatype="date64 (milliseconds)")
def is_date64(t):
    return t.id == lib.Type_DATE64


@doc(is_null, datatype="map")
def is_map(t):
    return t.id == lib.Type_MAP


@doc(is_null, datatype="decimal")
def is_decimal(t):
    return t.id in _DECIMAL_TYPES


@doc(is_null, datatype="decimal32")
def is_decimal32(t):
    return t.id == lib.Type_DECIMAL32


@doc(is_null, datatype="decimal64")
def is_decimal64(t):
    return t.id == lib.Type_DECIMAL64


@doc(is_null, datatype="decimal128")
def is_decimal128(t):
    return t.id == lib.Type_DECIMAL128


@doc(is_null, datatype="decimal256")
def is_decimal256(t):
    return t.id == lib.Type_DECIMAL256


@doc(is_null, datatype="dictionary-encoded")
def is_dictionary(t):
    return t.id == lib.Type_DICTIONARY


@doc(is_null, datatype="interval")
def is_interval(t):
    return t.id == lib.Type_INTERVAL_MONTH_DAY_NANO


@doc(is_null, datatype="primitive type")
def is_primitive(t):
    return lib._is_primitive(t.id)
<|endoftext|># Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


