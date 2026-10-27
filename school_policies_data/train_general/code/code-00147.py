from ..datastructures import IfRange
from ..datastructures import ImmutableList
from ..datastructures import ImmutableMultiDict
from ..datastructures import LanguageAccept
from ..datastructures import MIMEAccept
from ..datastructures import MultiDict
from ..datastructures import Range
from ..datastructures import RequestCacheControl
from ..http import parse_accept_header
from ..http import parse_cache_control_header
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import parse_list_header
from ..http import parse_options_header
from ..http import parse_range_header
from ..http import parse_set_header
from ..user_agent import UserAgent
from ..utils import cached_property
from ..utils import header_property
from .http import parse_cookie
from .utils import get_content_length
from .utils import get_current_url
from .utils import get_host


class Request:
    """Represents the non-IO parts of a HTTP request, including the
    method, URL info, and headers.

    This class is not meant for general use. It should only be used when
    implementing WSGI, ASGI, or another HTTP application spec. Werkzeug
    provides a WSGI implementation at :cls:`werkzeug.wrappers.Request`.

    :param method: The method the request was made with, such as
        ``GET``.
    :param scheme: The URL scheme of the protocol the request used, such
        as ``https`` or ``wss``.
    :param server: The address of the server. ``(host, port)``,
        ``(path, None)`` for unix sockets, or ``None`` if not known.
    :param root_path: The prefix that the application is mounted under.
        This is prepended to generated URLs, but is not part of route
        matching.
    :param path: The path part of the URL after ``root_path``.
    :param query_string: The part of the URL after the "?".
    :param headers: The headers received with the request.
    :param remote_addr: The address of the client sending the request.

    .. versionchanged:: 3.0
        The ``charset``, ``url_charset``, and ``encoding_errors`` attributes
        were removed.

    .. versionadded:: 2.0
    """

    #: the class to use for `args` and `form`.  The default is an
    #: :class:`~werkzeug.datastructures.ImmutableMultiDict` which supports
    #: multiple values per key. A :class:`~werkzeug.datastructures.ImmutableDict`
    #: is faster but only remembers the last key. It is also
    #: possible to use mutable structures, but this is not recommended.
    #:
    #: .. versionadded:: 0.6
    parameter_storage_class: type[MultiDict[str, t.Any]] = ImmutableMultiDict

    #: The type to be used for dict values from the incoming WSGI
    #: environment. (For example for :attr:`cookies`.) By default an
    #: :class:`~werkzeug.datastructures.ImmutableMultiDict` is used.
    #:
    #: .. versionchanged:: 1.0.0
    #:     Changed to ``ImmutableMultiDict`` to support multiple values.
    #:
    #: .. versionadded:: 0.6
    dict_storage_class: type[MultiDict[str, t.Any]] = ImmutableMultiDict

    #: the type to be used for list values from the incoming WSGI environment.
    #: By default an :class:`~werkzeug.datastructures.ImmutableList` is used
    #: (for example for :attr:`access_list`).
    #:
    #: .. versionadded:: 0.6
    list_storage_class: type[list[t.Any]] = ImmutableList

    user_agent_class: type[UserAgent] = UserAgent
    """The class used and returned by the :attr:`user_agent` property to
    parse the header. Defaults to
    :class:`~werkzeug.user_agent.UserAgent`, which does no parsing. An
    extension can provide a subclass that uses a parser to provide other
    data.

    .. versionadded:: 2.0
    """

    #: Valid host names when handling requests. By default all hosts are
    #: trusted, which means that whatever the client says the host is
    #: will be accepted.
    #:
    #: Because ``Host`` and ``X-Forwarded-Host`` headers can be set to
    #: any value by a malicious client, it is recommended to either set
    #: this property or implement similar validation in the proxy (if
    #: the application is being run behind one).
    #:
    #: .. versionadded:: 0.9
    trusted_hosts: list[str] | None = None

    def __init__(
        self,
        method: str,
        scheme: str,
        server: tuple[str, int | None] | None,
        root_path: str,
        path: str,
        query_string: bytes,
        headers: Headers,
        remote_addr: str | None,
    ) -> None:
        #: The method the request was made with, such as ``GET``.
        self.method = method.upper()
        #: The URL scheme of the protocol the request used, such as
        #: ``https`` or ``wss``.
        self.scheme = scheme
        #: The address of the server. ``(host, port)``, ``(path, None)``
        #: for unix sockets, or ``None`` if not known.
        self.server = server
        #: The prefix that the application is mounted under, without a
        #: trailing slash. :attr:`path` comes after this.
        self.root_path = root_path.rstrip("/")
        #: The path part of the URL after :attr:`root_path`. This is the
        #: path used for routing within the application.
        self.path = "/" + path.lstrip("/")
        #: The part of the URL after the "?". This is the raw value, use
        #: :attr:`args` for the parsed values.
        self.query_string = query_string
        #: The headers received with the request.
        self.headers = headers
        #: The address of the client sending the request.
        self.remote_addr = remote_addr

    def __repr__(self) -> str:
        try:
            url = self.url
        except Exception as e:
            url = f"(invalid URL: {e})"

        return f"<{type(self).__name__} {url!r} [{self.method}]>"

    @cached_property
    def args(self) -> MultiDict[str, str]:
        """The parsed URL parameters (the part in the URL after the question
        mark).

        By default an
        :class:`~werkzeug.datastructures.ImmutableMultiDict`
        is returned from this function.  This can be changed by setting
        :attr:`parameter_storage_class` to a different type.  This might
        be necessary if the order of the form data is important.

        .. versionchanged:: 2.3
            Invalid bytes remain percent encoded.
        """
        return self.parameter_storage_class(
            parse_qsl(
                self.query_string.decode(),
                keep_blank_values=True,
                errors="werkzeug.url_quote",
            )
        )

    @cached_property
    def access_route(self) -> list[str]:
        """If a forwarded header exists this is a list of all ip addresses
        from the client ip to the last proxy server.
        """
        if "X-Forwarded-For" in self.headers:
            return self.list_storage_class(
                parse_list_header(self.headers["X-Forwarded-For"])
            )
        elif self.remote_addr is not None:
            return self.list_storage_class([self.remote_addr])
        return self.list_storage_class()

    @cached_property
    def full_path(self) -> str:
        """Requested path, including the query string."""
        return f"{self.path}?{self.query_string.decode()}"

    @property
    def is_secure(self) -> bool:
        """``True`` if the request was made with a secure protocol
        (HTTPS or WSS).
        """
        return self.scheme in {"https", "wss"}

    @cached_property
    def url(self) -> str:
        """The full request URL with the scheme, host, root path, path,
        and query string."""
        return get_current_url(
            self.scheme, self.host, self.root_path, self.path, self.query_string
        )

    @cached_property
    def base_url(self) -> str:
        """Like :attr:`url` but without the query string."""
        return get_current_url(self.scheme, self.host, self.root_path, self.path)

    @cached_property
    def root_url(self) -> str:
        """The request URL scheme, host, and root path. This is the root
        that the application is accessed from.
        """
        return get_current_url(self.scheme, self.host, self.root_path)

    @cached_property
    def host_url(self) -> str:
        """The request URL scheme and host only."""
        return get_current_url(self.scheme, self.host)

    @cached_property
    def host(self) -> str:
        """The host name the request was made to, including the port if
        it's non-standard. Validated with :attr:`trusted_hosts`.

        See :func:`.get_host` for a detailed explanation.
        """
        return get_host(
            self.scheme, self.headers.get("host"), self.server, self.trusted_hosts
        )

    @cached_property
    def cookies(self) -> ImmutableMultiDict[str, str]:
        """A :class:`dict` with the contents of all cookies transmitted with
        the request."""
        wsgi_combined_cookie = ";".join(self.headers.getlist("Cookie"))
        return parse_cookie(  # type: ignore
            wsgi_combined_cookie, cls=self.dict_storage_class
        )

    # Common Descriptors

    content_type = header_property[str](
        "Content-Type",
        doc="""The Content-Type entity-header field indicates the media
        type of the entity-body sent to the recipient or, in the case of
        the HEAD method, the media type that would have been sent had
        the request been a GET.""",
        read_only=True,
    )

    @cached_property
    def content_length(self) -> int | None:
        """The Content-Length entity-header field indicates the size of the
        entity-body in bytes or, in the case of the HEAD method, the size of
        the entity-body that would have been sent had the request been a
        GET.
        """
        return get_content_length(
            http_content_length=self.headers.get("Content-Length"),
            http_transfer_encoding=self.headers.get("Transfer-Encoding"),
        )

    content_encoding = header_property[str](
        "Content-Encoding",
        doc="""The Content-Encoding entity-header field is used as a
        modifier to the media-type. When present, its value indicates
        what additional content codings have been applied to the
        entity-body, and thus what decoding mechanisms must be applied
        in order to obtain the media-type referenced by the Content-Type
        header field.

        .. versionadded:: 0.9""",
        read_only=True,
    )
    content_md5 = header_property[str](
        "Content-MD5",
        doc="""The Content-MD5 entity-header field, as defined in
        RFC 1864, is an MD5 digest of the entity-body for the purpose of
        providing an end-to-end message integrity check (MIC) of the
        entity-body. (Note: a MIC is good for detecting accidental
        modification of the entity-body in transit, but is not proof
        against malicious attacks.)

        .. versionadded:: 0.9""",
        read_only=True,
    )
    referrer = header_property[str](
        "Referer",
        doc="""The Referer[sic] request-header field allows the client
        to specify, for the server's benefit, the address (URI) of the
        resource from which the Request-URI was obtained (the
        "referrer", although the header field is misspelled).""",
        read_only=True,
    )
    date = header_property(
        "Date",
        None,
        parse_date,
        doc="""The Date general-header field represents the date and
        time at which the message was originated, having the same
        semantics as orig-date in RFC 822.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
        read_only=True,
    )
    max_forwards = header_property(
        "Max-Forwards",
        None,
        int,
        doc="""The Max-Forwards request-header field provides a
        mechanism with the TRACE and OPTIONS methods to limit the number
        of proxies or gateways that can forward the request to the next
        inbound server.""",
        read_only=True,
    )

    def _parse_content_type(self) -> None:
        if not hasattr(self, "_parsed_content_type"):
            self._parsed_content_type = parse_options_header(
                self.headers.get("Content-Type", "")
            )

    @property
    def mimetype(self) -> str:
        """Like :attr:`content_type`, but without parameters (eg, without
        charset, type etc.) and always lowercase.  For example if the content
        type is ``text/HTML; charset=utf-8`` the mimetype would be
        ``'text/html'``.
        """
        self._parse_content_type()
        return self._parsed_content_type[0].lower()

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict.  For example if the content
        type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.
        """
        self._parse_content_type()
        return self._parsed_content_type[1]

    @cached_property
    def pragma(self) -> HeaderSet:
        """The Pragma general-header field is used to include
        implementation-specific directives that might apply to any recipient
        along the request/response chain.  All pragma directives specify
        optional behavior from the viewpoint of the protocol; however, some
        systems MAY require that behavior be consistent with the directives.
        """
        return parse_set_header(self.headers.get("Pragma", ""))

    # Accept

    @cached_property
    def accept_mimetypes(self) -> MIMEAccept:
        """List of mimetypes this client supports as
        :class:`~werkzeug.datastructures.MIMEAccept` object.
        """
        return parse_accept_header(self.headers.get("Accept"), MIMEAccept)

    @cached_property
    def accept_charsets(self) -> CharsetAccept:
        """List of charsets this client supports as
        :class:`~werkzeug.datastructures.CharsetAccept` object.
        """
        return parse_accept_header(self.headers.get("Accept-Charset"), CharsetAccept)

    @cached_property
    def accept_encodings(self) -> Accept:
        """List of encodings this client accepts.  Encodings in a HTTP term
        are compression encodings such as gzip.  For charsets have a look at
        :attr:`accept_charset`.
        """
        return parse_accept_header(self.headers.get("Accept-Encoding"))

    @cached_property
    def accept_languages(self) -> LanguageAccept:
        """List of languages this client accepts as
        :class:`~werkzeug.datastructures.LanguageAccept` object.

        .. versionchanged 0.5
           In previous versions this was a regular
           :class:`~werkzeug.datastructures.Accept` object.
        """
        return parse_accept_header(self.headers.get("Accept-Language"), LanguageAccept)

    # ETag

    @cached_property
    def cache_control(self) -> RequestCacheControl:
        """A :class:`~werkzeug.datastructures.RequestCacheControl` object
        for the incoming cache control headers.
        """
        cache_control = self.headers.get("Cache-Control")
        return parse_cache_control_header(cache_control, None, RequestCacheControl)

    @cached_property
    def if_match(self) -> ETags:
        """An object containing all the etags in the `If-Match` header.

        :rtype: :class:`~werkzeug.datastructures.ETags`
        """
        return parse_etags(self.headers.get("If-Match"))

    @cached_property
    def if_none_match(self) -> ETags:
        """An object containing all the etags in the `If-None-Match` header.

        :rtype: :class:`~werkzeug.datastructures.ETags`
        """
        return parse_etags(self.headers.get("If-None-Match"))

    @cached_property
    def if_modified_since(self) -> datetime | None:
        """The parsed `If-Modified-Since` header as a datetime object.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        return parse_date(self.headers.get("If-Modified-Since"))

    @cached_property
    def if_unmodified_since(self) -> datetime | None:
        """The parsed `If-Unmodified-Since` header as a datetime object.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        return parse_date(self.headers.get("If-Unmodified-Since"))

    @cached_property
    def if_range(self) -> IfRange:
        """The parsed ``If-Range`` header.

        .. versionchanged:: 2.0
            ``IfRange.date`` is timezone-aware.

        .. versionadded:: 0.7
        """
        return parse_if_range_header(self.headers.get("If-Range"))

    @cached_property
    def range(self) -> Range | None:
        """The parsed `Range` header.

        .. versionadded:: 0.7

        :rtype: :class:`~werkzeug.datastructures.Range`
        """
        return parse_range_header(self.headers.get("Range"))

    # User Agent

    @cached_property
    def user_agent(self) -> UserAgent:
        """The user agent. Use ``user_agent.string`` to get the header
        value. Set :attr:`user_agent_class` to a subclass of
        :class:`~werkzeug.user_agent.UserAgent` to provide parsing for
        the other properties or other extended data.

        .. versionchanged:: 2.1
            The built-in parser was removed. Set ``user_agent_class`` to a ``UserAgent``
            subclass to parse data from the string.
        """
        return self.user_agent_class(self.headers.get("User-Agent", ""))

    # Authorization

    @cached_property
    def authorization(self) -> Authorization | None:
        """The ``Authorization`` header parsed into an :class:`.Authorization` object.
        ``None`` if the header is not present.

        .. versionchanged:: 2.3
            :class:`Authorization` is no longer a ``dict``. The ``token`` attribute
            was added for auth schemes that use a token instead of parameters.
        """
        return Authorization.from_header(self.headers.get("Authorization"))

    # CORS

    origin = header_property[str](
        "Origin",
        doc=(
            "The host that the request originated from. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_origin` on"
            " the response to indicate which origins are allowed."
        ),
        read_only=True,
    )

    access_control_request_headers = header_property(
        "Access-Control-Request-Headers",
        load_func=parse_set_header,
        doc=(
            "Sent with a preflight request to indicate which headers"
            " will be sent with the cross origin request. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_headers`"
            " on the response to indicate which headers are allowed."
        ),
        read_only=True,
    )

    access_control_request_method = header_property[str](
        "Access-Control-Request-Method",
        doc=(
            "Sent with a preflight request to indicate which method"
            " will be used for the cross origin request. Set"
            " :attr:`~CORSResponseMixin.access_control_allow_methods`"
            " on the response to indicate which methods are allowed."
        ),
        read_only=True,
    )

    @property
    def is_json(self) -> bool:
        """Check if the mimetype indicates JSON data, either
        :mimetype:`application/json` or :mimetype:`application/*+json`.
        """
        mt = self.mimetype
        return (
            mt == "application/json"
            or mt.startswith("application/")
            and mt.endswith("+json")
        )
<|endoftext|>from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass
from dataclasses import field

from .converters import ValidationError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .rules import Rule
from .rules import RulePart


class SlashRequired(Exception):
    pass


@dataclass
class State:
    """A representation of a rule state.

    This includes the *rules* that correspond to the state and the
    possible *static* and *dynamic* transitions to the next state.
    """

    dynamic: list[tuple[RulePart, State]] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    static: dict[str, State] = field(default_factory=dict)


class StateMachineMatcher:
    def __init__(self, merge_slashes: bool) -> None:
        self._root = State()
        self.merge_slashes = merge_slashes

    def add(self, rule: Rule) -> None:
        state = self._root
        for part in rule._parts:
            if part.static:
                state.static.setdefault(part.content, State())
                state = state.static[part.content]
            else:
                for test_part, new_state in state.dynamic:
                    if test_part == part:
                        state = new_state
                        break
                else:
                    new_state = State()
                    state.dynamic.append((part, new_state))
                    state = new_state
        state.rules.append(rule)

    def update(self) -> None:
        # For every state the dynamic transitions should be sorted by
        # the weight of the transition
        state = self._root

        def _update_state(state: State) -> None:
            state.dynamic.sort(key=lambda entry: entry[0].weight)
            for new_state in state.static.values():
                _update_state(new_state)
            for _, new_state in state.dynamic:
                _update_state(new_state)

        _update_state(state)

    def match(
        self, domain: str, path: str, method: str, websocket: bool
    ) -> tuple[Rule, t.MutableMapping[str, t.Any]]:
        # To match to a rule we need to start at the root state and
        # try to follow the transitions until we find a match, or find
        # there is no transition to follow.

        have_match_for = set()
        websocket_mismatch = False

        def _match(
            state: State, parts: list[str], values: list[str]
        ) -> tuple[Rule, list[str]] | None:
            # This function is meant to be called recursively, and will attempt
            # to match the head part to the state's transitions.
            nonlocal have_match_for, websocket_mismatch

            # The base case is when all parts have been matched via
            # transitions. Hence if there is a rule with methods &
            # websocket that work return it and the dynamic values
            # extracted.
            if parts == []:
                for rule in state.rules:
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return rule, values

                # Test if there is a match with this path with a
                # trailing slash, if so raise an exception to report
                # that matching is possible with an additional slash
                if "" in state.static:
                    for rule in state.static[""].rules:
                        if websocket == rule.websocket and (
                            rule.methods is None or method in rule.methods
                        ):
                            if rule.strict_slashes:
                                raise SlashRequired()
                            else:
                                return rule, values
                return None

            part = parts[0]
            # To match this part try the static transitions first
            if part in state.static:
                rv = _match(state.static[part], parts[1:], values)
                if rv is not None:
                    return rv
            # No match via the static transitions, so try the dynamic
            # ones.
            for test_part, new_state in state.dynamic:
                target = part
                remaining = parts[1:]
                # A final part indicates a transition that always
                # consumes the remaining parts i.e. transitions to a
                # final state.
                if test_part.final:
                    target = "/".join(parts)
                    remaining = []
                match = re.compile(test_part.content).match(target)
                if match is not None:
                    if test_part.suffixed:
                        # If a part_isolating=False part has a slash suffix, remove the
                        # suffix from the match and check for the slash redirect next.
                        suffix = match.groups()[-1]
                        if suffix == "/":
                            remaining = [""]

                    converter_groups = sorted(
                        (
                            item
                            for item in match.groupdict().items()
                            if item[0].startswith("__werkzeug_")
                        ),
                        key=lambda item: int(item[0][11:]),
                    )
                    groups = [item[1] for item in converter_groups]
                    rv = _match(new_state, remaining, values + groups)
                    if rv is not None:
                        return rv

            # If there is no match and the only part left is a
            # trailing slash ("") consider rules that aren't
            # strict-slashes as these should match if there is a final
            # slash part.
            if parts == [""]:
                for rule in state.rules:
                    if rule.strict_slashes:
                        continue
                    if rule.methods is not None and method not in rule.methods:
                        have_match_for.update(rule.methods)
                    elif rule.websocket != websocket:
                        websocket_mismatch = True
                    else:
                        return rule, values

            return None

        try:
            rv = _match(self._root, [domain, *path.split("/")], [])
        except SlashRequired:
            raise RequestPath(f"{path}/") from None

        if self.merge_slashes and rv is None:
            # Try to match again, but with slashes merged
            path = re.sub("/{2,}", "/", path)
            try:
                rv = _match(self._root, [domain, *path.split("/")], [])
            except SlashRequired:
                raise RequestPath(f"{path}/") from None
            if rv is None or rv[0].merge_slashes is False:
                raise NoMatch(have_match_for, websocket_mismatch)
            else:
                raise RequestPath(f"{path}")
        elif rv is not None:
            rule, values = rv

            result = {}
            for name, value in zip(rule._converters.keys(), values):
                try:
                    value = rule._converters[name].to_python(value)
                except ValidationError as e:
                    raise NoMatch(have_match_for, websocket_mismatch) from e
                result[str(name)] = value
            if rule.defaults:
                result.update(rule.defaults)

            if rule.alias and rule.map.redirect_defaults:
                raise RequestAliasRedirect(result, rule.endpoint)

            return rule, result

        raise NoMatch(have_match_for, websocket_mismatch)
<|endoftext|>from __future__ import annotations

import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote

from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError

if t.TYPE_CHECKING:
    from .converters import BaseConverter
    from .map import Map


class Weighting(t.NamedTuple):
    number_static_weights: int
    static_weights: list[tuple[int, int]]
    number_argument_weights: int
    argument_weights: list[int]


@dataclass
class RulePart:
    """A part of a rule.

    Rules can be represented by parts as delimited by `/` with
    instances of this class representing those parts. The *content* is
    either the raw content if *static* or a regex string to match
    against. The *weight* can be used to order parts when matching.

    """

    content: str
    final: bool
    static: bool
    suffixed: bool
    weight: Weighting


_part_re = re.compile(
    r"""
    (?:
        (?P<slash>/)                                 # a slash
      |
        (?P<static>[^</]+)                           # static rule data
      |
        (?:
          <
            (?:
              (?P<converter>[a-zA-Z_][a-zA-Z0-9_]*)   # converter name
              (?:\((?P<arguments>.*?)\))?             # converter arguments
              :                                       # variable delimiter
            )?
            (?P<variable>[a-zA-Z_][a-zA-Z0-9_]*)      # variable name
           >
        )
    )
    """,
    re.VERBOSE,
)

_simple_rule_re = re.compile(r"<([^>]+)>")
_converter_args_re = re.compile(
    r"""
    \s*
    ((?P<name>\w+)\s*=\s*)?
    (?P<value>
        True|False|
        \d+.\d+|
        \d+.|
        \d+|
        [\w\d_.]+|
        [urUR]?(?P<stringval>"[^"]*?"|'[^']*')
    )\s*,
    """,
    re.VERBOSE,
)

_PYTHON_CONSTANTS = {"None": None, "True": True, "False": False}


def _find(value: str, target: str, pos: int) -> int:
    """Find the *target* in *value* after *pos*.

    Returns the *value* length if *target* isn't found.
    """
    try:
        return value.index(target, pos)
    except ValueError:
        return len(value)


def _pythonize(value: str) -> None | bool | int | float | str:
    if value in _PYTHON_CONSTANTS:
        return _PYTHON_CONSTANTS[value]
    for convert in int, float:
        try:
            return convert(value)
        except ValueError:
            pass
    if value[:1] == value[-1:] and value[0] in "\"'":
        value = value[1:-1]
    return str(value)


def parse_converter_args(argstr: str) -> tuple[tuple[t.Any, ...], dict[str, t.Any]]:
    argstr += ","
    args = []
    kwargs = {}
    position = 0

    for item in _converter_args_re.finditer(argstr):
        if item.start() != position:
            raise ValueError(
                f"Cannot parse converter argument '{argstr[position : item.start()]}'"
            )

        value = item.group("stringval")
        if value is None:
            value = item.group("value")
        value = _pythonize(value)
        if not item.group("name"):
            args.append(value)
        else:
            name = item.group("name")
            kwargs[name] = value
        position = item.end()

    return tuple(args), kwargs


class RuleFactory:
    """As soon as you have more complex URL setups it's a good idea to use rule
    factories to avoid repetitive tasks.  Some of them are builtin, others can
    be added by subclassing `RuleFactory` and overriding `get_rules`.
    """

    def get_rules(self, map: Map) -> t.Iterable[Rule]:
        """Subclasses of `RuleFactory` have to override this method and return
        an iterable of rules."""
        raise NotImplementedError()


class Subdomain(RuleFactory):
    """All URLs provided by this factory have the subdomain set to a
    specific domain. For example if you want to use the subdomain for
    the current language this can be a good setup::

        url_map = Map([
            Rule('/', endpoint='#select_language'),
            Subdomain('<string(length=2):lang_code>', [
                Rule('/', endpoint='index'),
                Rule('/about', endpoint='about'),
                Rule('/help', endpoint='help')
            ])
        ])

    All the rules except for the ``'#select_language'`` endpoint will now
    listen on a two letter long subdomain that holds the language code
    for the current request.
    """

    def __init__(self, subdomain: str, rules: t.Iterable[RuleFactory]) -> None:
        self.subdomain = subdomain
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.subdomain = self.subdomain
                yield rule


class Submount(RuleFactory):
    """Like `Subdomain` but prefixes the URL rule with a given string::

        url_map = Map([
            Rule('/', endpoint='index'),
            Submount('/blog', [
                Rule('/', endpoint='blog/index'),
                Rule('/entry/<entry_slug>', endpoint='blog/show')
            ])
        ])

    Now the rule ``'blog/show'`` matches ``/blog/entry/<entry_slug>``.
    """

    def __init__(self, path: str, rules: t.Iterable[RuleFactory]) -> None:
        self.path = path.rstrip("/")
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.rule = self.path + rule.rule
                yield rule


class EndpointPrefix(RuleFactory):
    """Prefixes all endpoints (which must be strings for this factory) with
    another string. This can be useful for sub applications::

        url_map = Map([
            Rule('/', endpoint='index'),
            EndpointPrefix('blog/', [Submount('/blog', [
                Rule('/', endpoint='index'),
                Rule('/entry/<entry_slug>', endpoint='show')
            ])])
        ])
    """

    def __init__(self, prefix: str, rules: t.Iterable[RuleFactory]) -> None:
        self.prefix = prefix
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.endpoint = self.prefix + rule.endpoint
                yield rule


class RuleTemplate:
    """Returns copies of the rules wrapped and expands string templates in
    the endpoint, rule, defaults or subdomain sections.

    Here a small example for such a rule template::

        from werkzeug.routing import Map, Rule, RuleTemplate

        resource = RuleTemplate([
            Rule('/$name/', endpoint='$name.list'),
            Rule('/$name/<int:id>', endpoint='$name.show')
        ])

        url_map = Map([resource(name='user'), resource(name='page')])

    When a rule template is called the keyword arguments are used to
    replace the placeholders in all the string parameters.
    """

    def __init__(self, rules: t.Iterable[Rule]) -> None:
        self.rules = list(rules)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> RuleTemplateFactory:
        return RuleTemplateFactory(self.rules, dict(*args, **kwargs))


class RuleTemplateFactory(RuleFactory):
    """A factory that fills in template variables into rules.  Used by
    `RuleTemplate` internally.

    :internal:
    """

    def __init__(
        self, rules: t.Iterable[RuleFactory], context: dict[str, t.Any]
    ) -> None:
        self.rules = rules
        self.context = context

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                new_defaults = subdomain = None
                if rule.defaults:
                    new_defaults = {}
                    for key, value in rule.defaults.items():
                        if isinstance(value, str):
                            value = Template(value).substitute(self.context)
                        new_defaults[key] = value
                if rule.subdomain is not None:
                    subdomain = Template(rule.subdomain).substitute(self.context)
                new_endpoint = rule.endpoint
                if isinstance(new_endpoint, str):
                    new_endpoint = Template(new_endpoint).substitute(self.context)
                yield Rule(
                    Template(rule.rule).substitute(self.context),
                    new_defaults,
                    subdomain,
                    rule.methods,
                    rule.build_only,
                    new_endpoint,
                    rule.strict_slashes,
                )


_ASTT = t.TypeVar("_ASTT", bound=ast.AST)


def _prefix_names(src: str, expected_type: type[_ASTT]) -> _ASTT:
    """ast parse and prefix names with `.` to avoid collision with user vars"""
    tree: ast.AST = ast.parse(src).body[0]
    if isinstance(tree, ast.Expr):
        tree = tree.value
    if not isinstance(tree, expected_type):
        raise TypeError(
            f"AST node is of type {type(tree).__name__}, not {expected_type.__name__}"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = f".{node.id}"
    return tree


_CALL_CONVERTER_CODE_FMT = "self._converters[{elem!r}].to_url()"
_IF_KWARGS_URL_ENCODE_CODE = """\
if kwargs:
    params = self._encode_query_vars(kwargs)
    q = "?" if params else ""
else:
    q = params = ""
"""
_IF_KWARGS_URL_ENCODE_AST = _prefix_names(_IF_KWARGS_URL_ENCODE_CODE, ast.If)
_URL_ENCODE_AST_NAMES = (
    _prefix_names("q", ast.Name),
    _prefix_names("params", ast.Name),
)


class Rule(RuleFactory):
    """A Rule represents one URL pattern.  There are some options for `Rule`
    that change the way it behaves and are passed to the `Rule` constructor.
    Note that besides the rule-string all arguments *must* be keyword arguments
    in order to not break the application on Werkzeug upgrades.

    `string`
        Rule strings basically are just normal URL paths with placeholders in
        the format ``<converter(arguments):name>`` where the converter and the
        arguments are optional.  If no converter is defined the `default`
        converter is used which means `string` in the normal configuration.

        URL rules that end with a slash are branch URLs, others are leaves.
        If you have `strict_slashes` enabled (which is the default), all
        branch URLs that are matched without a trailing slash will trigger a
        redirect to the same URL with the missing slash appended.

        The converters are defined on the `Map`.

    `endpoint`
        The endpoint for this rule. This can be anything. A reference to a
        function, a string, a number etc.  The preferred way is using a string
        because the endpoint is used for URL generation.

    `defaults`
        An optional dict with defaults for other rules with the same endpoint.
        This is a bit tricky but useful if you want to have unique URLs::

            url_map = Map([
                Rule('/all/', defaults={'page': 1}, endpoint='all_entries'),
                Rule('/all/page/<int:page>', endpoint='all_entries')
            ])

        If a user now visits ``http://example.com/all/page/1`` they will be
        redirected to ``http://example.com/all/``.  If `redirect_defaults` is
        disabled on the `Map` instance this will only affect the URL
        generation.

    `subdomain`
        The subdomain rule string for this rule. If not specified the rule
        only matches for the `default_subdomain` of the map.  If the map is
        not bound to a subdomain this feature is disabled.

        Can be useful if you want to have user profiles on different subdomains
        and all subdomains are forwarded to your application::

            url_map = Map([
                Rule('/', subdomain='<username>', endpoint='user/homepage'),
                Rule('/stats', subdomain='<username>', endpoint='user/stats')
            ])

    `methods`
        A sequence of http methods this rule applies to.  If not specified, all
        methods are allowed. For example this can be useful if you want different
        endpoints for `POST` and `GET`.  If methods are defined and the path
        matches but the method matched against is not in this list or in the
        list of another rule for that path the error raised is of the type
        `MethodNotAllowed` rather than `NotFound`.  If `GET` is present in the
        list of methods and `HEAD` is not, `HEAD` is added automatically.

    `strict_slashes`
        Override the `Map` setting for `strict_slashes` only for this rule. If
        not specified the `Map` setting is used.

    `merge_slashes`
        Override :attr:`Map.merge_slashes` for this rule.

    `build_only`
        Set this to True and the rule will never match but will create a URL
        that can be build. This is useful if you have resources on a subdomain
        or folder that are not handled by the WSGI application (like static data)

    `redirect_to`
        If given this must be either a string or callable.  In case of a
        callable it's called with the url adapter that triggered the match and
        the values of the URL as keyword arguments and has to return the target
        for the redirect, otherwise it has to be a string with placeholders in
        rule syntax::

            def foo_with_slug(adapter, id):
                # ask the database for the slug for the old id.  this of
                # course has nothing to do with werkzeug.
                return f'foo/{Foo.get_slug_for_id(id)}'

            url_map = Map([
                Rule('/foo/<slug>', endpoint='foo'),
                Rule('/some/old/url/<slug>', redirect_to='foo/<slug>'),
                Rule('/other/old/url/<int:id>', redirect_to=foo_with_slug)
            ])

        When the rule is matched the routing system will raise a
        `RequestRedirect` exception with the target for the redirect.

        Keep in mind that the URL will be joined against the URL root of the
        script so don't use a leading slash on the target URL unless you
        really mean root of that domain.

    `alias`
        If enabled this rule serves as an alias for another rule with the same
        endpoint and arguments.

    `host`
        If provided and the URL map has host matching enabled this can be
        used to provide a match rule for the whole host.  This also means
        that the subdomain feature is disabled.

    `websocket`
        If ``True``, this rule is only matches for WebSocket (``ws://``,
        ``wss://``) requests. By default, rules will only match for HTTP
        requests.

    .. versionchanged:: 2.1
        Percent-encoded newlines (``%0a``), which are decoded by WSGI
        servers, are considered when routing instead of terminating the
        match early.

    .. versionadded:: 1.0
        Added ``websocket``.

    .. versionadded:: 1.0
        Added ``merge_slashes``.

    .. versionadded:: 0.7
        Added ``alias`` and ``host``.

    .. versionchanged:: 0.6.1
       ``HEAD`` is added to ``methods`` if ``GET`` is present.
    """

    def __init__(
        self,
        string: str,
        defaults: t.Mapping[str, t.Any] | None = None,
        subdomain: str | None = None,
        methods: t.Iterable[str] | None = None,
        build_only: bool = False,
        endpoint: t.Any | None = None,
        strict_slashes: bool | None = None,
        merge_slashes: bool | None = None,
        redirect_to: str | t.Callable[..., str] | None = None,
        alias: bool = False,
        host: str | None = None,
        websocket: bool = False,
    ) -> None:
        if not string.startswith("/"):
            raise ValueError(f"URL rule '{string}' must start with a slash.")

        self.rule = string
        self.is_leaf = not string.endswith("/")
        self.is_branch = string.endswith("/")

        self.map: Map = None  # type: ignore
        self.strict_slashes = strict_slashes
        self.merge_slashes = merge_slashes
        self.subdomain = subdomain
        self.host = host
        self.defaults = defaults
        self.build_only = build_only
        self.alias = alias
        self.websocket = websocket

        if methods is not None:
            if isinstance(methods, str):
                raise TypeError("'methods' should be a list of strings.")

            methods = {x.upper() for x in methods}

            if "HEAD" not in methods and "GET" in methods:
                methods.add("HEAD")

            if websocket and methods - {"GET", "HEAD", "OPTIONS"}:
                raise ValueError(
                    "WebSocket rules can only use 'GET', 'HEAD', and 'OPTIONS' methods."
                )

        self.methods = methods
        self.endpoint: t.Any = endpoint
        self.redirect_to = redirect_to

        if defaults:
            self.arguments = set(map(str, defaults))
        else:
            self.arguments = set()

        self._converters: dict[str, BaseConverter] = {}
        self._trace: list[tuple[bool, str]] = []
        self._parts: list[RulePart] = []

    def empty(self) -> Rule:
        """
        Return an unbound copy of this rule.

        This can be useful if want to reuse an already bound URL for another
        map.  See ``get_empty_kwargs`` to override what keyword arguments are
        provided to the new copy.
        """
        return type(self)(self.rule, **self.get_empty_kwargs())

    def get_empty_kwargs(self) -> t.Mapping[str, t.Any]:
        """
        Provides kwargs for instantiating empty copy with empty()

        Use this method to provide custom keyword arguments to the subclass of
        ``Rule`` when calling ``some_rule.empty()``.  Helpful when the subclass
        has custom keyword arguments that are needed at instantiation.

        Must return a ``dict`` that will be provided as kwargs to the new
        instance of ``Rule``, following the initial ``self.rule`` value which
        is always provided as the first, required positional argument.
        """
        defaults = None
        if self.defaults:
            defaults = dict(self.defaults)
        return dict(
            defaults=defaults,
            subdomain=self.subdomain,
            methods=self.methods,
            build_only=self.build_only,
            endpoint=self.endpoint,
            strict_slashes=self.strict_slashes,
            redirect_to=self.redirect_to,
            alias=self.alias,
            host=self.host,
        )

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        yield self

    def refresh(self) -> None:
        """Rebinds and refreshes the URL.  Call this if you modified the
        rule in place.

        :internal:
        """
        self.bind(self.map, rebind=True)

    def bind(self, map: Map, rebind: bool = False) -> None:
        """Bind the url to a map and create a regular expression based on
        the information from the rule itself and the defaults from the map.

        :internal:
        """
        if self.map is not None and not rebind:
            raise RuntimeError(f"url rule {self!r} already bound to map {self.map!r}")
        self.map = map
        if self.strict_slashes is None:
            self.strict_slashes = map.strict_slashes
        if self.merge_slashes is None:
            self.merge_slashes = map.merge_slashes
        if self.subdomain is None:
            self.subdomain = map.default_subdomain
        self.compile()

    def get_converter(
        self,
        variable_name: str,
        converter_name: str,
        args: tuple[t.Any, ...],
        kwargs: t.Mapping[str, t.Any],
    ) -> BaseConverter:
        """Looks up the converter for the given parameter.

        .. versionadded:: 0.9
        """
        if converter_name not in self.map.converters:
            raise LookupError(f"the converter {converter_name!r} does not exist")
        return self.map.converters[converter_name](self.map, *args, **kwargs)

    def _encode_query_vars(self, query_vars: t.Mapping[str, t.Any]) -> str:
        items: t.Iterable[tuple[str, str]] = iter_multi_items(query_vars)

        if self.map.sort_parameters:
            items = sorted(items, key=self.map.sort_key)

        return _urlencode(items)

    def _parse_rule(self, rule: str) -> t.Iterable[RulePart]:
        content = ""
        static = True
        argument_weights = []
        static_weights: list[tuple[int, int]] = []
        final = False
        convertor_number = 0

        pos = 0
        while pos < len(rule):
            match = _part_re.match(rule, pos)
            if match is None:
                raise ValueError(f"malformed url rule: {rule!r}")

            data = match.groupdict()
            if data["static"] is not None:
                static_weights.append((len(static_weights), -len(data["static"])))
                self._trace.append((False, data["static"]))
                content += data["static"] if static else re.escape(data["static"])

            if data["variable"] is not None:
                if static:
                    # Switching content to represent regex, hence the need to escape
                    content = re.escape(content)
                static = False
                c_args, c_kwargs = parse_converter_args(data["arguments"] or "")
                convobj = self.get_converter(
                    data["variable"], data["converter"] or "default", c_args, c_kwargs
                )
                self._converters[data["variable"]] = convobj
                self.arguments.add(data["variable"])
                if not convobj.part_isolating:
                    final = True
                content += f"(?P<__werkzeug_{convertor_number}>{convobj.regex})"
                convertor_number += 1
                argument_weights.append(convobj.weight)
                self._trace.append((True, data["variable"]))

            if data["slash"] is not None:
                self._trace.append((False, "/"))
                if final:
                    content += "/"
                else:
                    if not static:
                        content += r"\Z"
                    weight = Weighting(
                        -len(static_weights),
                        static_weights,
                        -len(argument_weights),
                        argument_weights,
                    )
                    yield RulePart(
                        content=content,
                        final=final,
                        static=static,
                        suffixed=False,
                        weight=weight,
                    )
                    content = ""
                    static = True
                    argument_weights = []
                    static_weights = []
                    final = False
                    convertor_number = 0

            pos = match.end()

        suffixed = False
        if final and content.endswith("/"):
            # If a converter is part_isolating=False (matches slashes) and ends with a
            # slash, augment the regex to support slash redirects.
            suffixed = True
            content = content[:-1] + "(?<!/)(/?)"
        if not static:
            content += r"\Z"
        weight = Weighting(
            -len(static_weights),
            static_weights,
            -len(argument_weights),
            argument_weights,
        )
        yield RulePart(
            content=content,
            final=final,
            static=static,
            suffixed=suffixed,
            weight=weight,
        )
        if suffixed:
            yield RulePart(
                content="", final=False, static=True, suffixed=False, weight=weight
            )

    def compile(self) -> None:
        """Compiles the regular expression and stores it."""
        assert self.map is not None, "rule not bound"

        if self.map.host_matching:
            domain_rule = self.host or ""
        else:
            domain_rule = self.subdomain or ""
        self._parts = []
        self._trace = []
        self._converters = {}
        if domain_rule == "":
            self._parts = [
                RulePart(
                    content="",
                    final=False,
                    static=True,
                    suffixed=False,
                    weight=Weighting(0, [], 0, []),
                )
            ]
        else:
            self._parts.extend(self._parse_rule(domain_rule))
        self._trace.append((False, "|"))
        rule = self.rule
        if self.merge_slashes:
            rule = re.sub("/{2,}", "/", self.rule)
        self._parts.extend(self._parse_rule(rule))

        self._build: t.Callable[..., tuple[str, str]]
        self._build = self._compile_builder(False).__get__(self, None)
        self._build_unknown: t.Callable[..., tuple[str, str]]
        self._build_unknown = self._compile_builder(True).__get__(self, None)

    @staticmethod
    def _get_func_code(code: CodeType, name: str) -> t.Callable[..., tuple[str, str]]:
        globs: dict[str, t.Any] = {}
        locs: dict[str, t.Any] = {}
        exec(code, globs, locs)
        return locs[name]  # type: ignore

    def _compile_builder(
        self, append_unknown: bool = True
    ) -> t.Callable[..., tuple[str, str]]:
        defaults = self.defaults or {}
        dom_ops: list[tuple[bool, str]] = []
        url_ops: list[tuple[bool, str]] = []

        opl = dom_ops
        for is_dynamic, data in self._trace:
            if data == "|" and opl is dom_ops:
                opl = url_ops
                continue
            # this seems like a silly case to ever come up but:
            # if a default is given for a value that appears in the rule,
            # resolve it to a constant ahead of time
            if is_dynamic and data in defaults:
                data = self._converters[data].to_url(defaults[data])
                opl.append((False, data))
            elif not is_dynamic:
                # safe = https://url.spec.whatwg.org/#url-path-segment-string
                opl.append((False, quote(data, safe="!$&'()*+,/:;=@")))
            else:
                opl.append((True, data))

        def _convert(elem: str) -> ast.Call:
            ret = _prefix_names(_CALL_CONVERTER_CODE_FMT.format(elem=elem), ast.Call)
            ret.args = [ast.Name(elem, ast.Load())]
            return ret

        def _parts(ops: list[tuple[bool, str]]) -> list[ast.expr]:
            parts: list[ast.expr] = [
                _convert(elem) if is_dynamic else ast.Constant(elem)
                for is_dynamic, elem in ops
            ]
            parts = parts or [ast.Constant("")]
            # constant fold
            ret = [parts[0]]
            for p in parts[1:]:
                if isinstance(p, ast.Constant) and isinstance(ret[-1], ast.Constant):
                    ret[-1] = ast.Constant(ret[-1].value + p.value)  # type: ignore[operator]
                else:
                    ret.append(p)
            return ret

        dom_parts = _parts(dom_ops)
        url_parts = _parts(url_ops)
        body: list[ast.stmt]
        if not append_unknown:
            body = []
        else:
            body = [_IF_KWARGS_URL_ENCODE_AST]
            url_parts.extend(_URL_ENCODE_AST_NAMES)

        def _join(parts: list[ast.expr]) -> ast.expr:
            if len(parts) == 1:  # shortcut
                return parts[0]
            return ast.JoinedStr(parts)

        body.append(
            ast.Return(ast.Tuple([_join(dom_parts), _join(url_parts)], ast.Load()))
        )

        pargs = [
            elem
            for is_dynamic, elem in dom_ops + url_ops
            if is_dynamic and elem not in defaults
        ]
        kargs = [str(k) for k in defaults]

        func_ast = _prefix_names("def _(): pass", ast.FunctionDef)
        func_ast.name = f"<builder:{self.rule!r}>"
        func_ast.args.args.append(ast.arg(".self", None))
        for arg in pargs + kargs:
            func_ast.args.args.append(ast.arg(arg, None))
        func_ast.args.kwarg = ast.arg(".kwargs", None)
        for _ in kargs:
            func_ast.args.defaults.append(ast.Constant(""))
        func_ast.body = body

        # Use `ast.parse` instead of `ast.Module` for better portability, since the
        # signature of `ast.Module` can change.
        module = ast.parse("")
        module.body = [func_ast]

        # mark everything as on line 1, offset 0
        # less error-prone than `ast.fix_missing_locations`
        # bad line numbers cause an assert to fail in debug builds
        for node in ast.walk(module):
            if "lineno" in node._attributes:
                node.lineno = 1  # type: ignore[attr-defined]
            if "end_lineno" in node._attributes:
                node.end_lineno = node.lineno  # type: ignore[attr-defined]
            if "col_offset" in node._attributes:
                node.col_offset = 0  # type: ignore[attr-defined]
            if "end_col_offset" in node._attributes:
                node.end_col_offset = node.col_offset  # type: ignore[attr-defined]

        code = compile(module, "<werkzeug routing>", "exec")
        return self._get_func_code(code, func_ast.name)

    def build(
        self, values: t.Mapping[str, t.Any], append_unknown: bool = True
    ) -> tuple[str, str] | None:
        """Assembles the relative url for that rule and the subdomain.
        If building doesn't work for some reasons `None` is returned.

        :internal:
        """
        try:
            if append_unknown:
                return self._build_unknown(**values)
            else:
                return self._build(**values)
        except ValidationError:
            return None

    def provides_defaults_for(self, rule: Rule) -> bool:
        """Check if this rule has defaults for a given rule.

        :internal:
        """
        return bool(
            not self.build_only
            and self.defaults
            and self.endpoint == rule.endpoint
            and self != rule
            and self.arguments == rule.arguments
        )

    def suitable_for(
        self, values: t.Mapping[str, t.Any], method: str | None = None
    ) -> bool:
        """Check if the dict of values has enough data for url generation.

        :internal:
        """
        # if a method was given explicitly and that method is not supported
        # by this rule, this rule is not suitable.
        if (
            method is not None
            and self.methods is not None
            and method not in self.methods
        ):
            return False

        defaults = self.defaults or ()

        # all arguments required must be either in the defaults dict or
        # the value dictionary otherwise it's not suitable
        for key in self.arguments:
            if key not in defaults and key not in values:
                return False

        # in case defaults are given we ensure that either the value was
        # skipped or the value is the same as the default value.
        if defaults:
            for key, value in defaults.items():
                if key in values and value != values[key]:
                    return False

        return True

    def build_compare_key(self) -> tuple[int, int, int]:
        """The build compare key for sorting.

        :internal:
        """
        return (1 if self.alias else 0, -len(self.arguments), -len(self.defaults or ()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._trace == other._trace

    __hash__ = None  # type: ignore

    def __str__(self) -> str:
        return self.rule

    def __repr__(self) -> str:
        if self.map is None:
            return f"<{type(self).__name__} (unbound)>"
        parts = []
        for is_dynamic, data in self._trace:
            if is_dynamic:
                parts.append(f"<{data}>")
            else:
                parts.append(data)
        parts_str = "".join(parts).lstrip("|")
        methods = f" ({', '.join(self.methods)})" if self.methods is not None else ""
        return f"<{type(self).__name__} {parts_str!r}{methods} -> {self.endpoint}>"
<|endoftext|>from __future__ import annotations

import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit

from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

    from ..wrappers.request import Request
    from .converters import BaseConverter
    from .rules import RuleFactory


class Map:
    """The map class stores all the URL rules and some configuration
    parameters.  Some of the configuration values are only stored on the
    `Map` instance since those affect all rules, others are just defaults
    and can be overridden for each rule.  Note that you have to specify all
    arguments besides the `rules` as keyword arguments!

    :param rules: sequence of url rules for this map.
    :param default_subdomain: The default subdomain for rules without a
                              subdomain defined.
    :param strict_slashes: If a rule ends with a slash but the matched
        URL does not, redirect to the URL with a trailing slash.
    :param merge_slashes: Merge consecutive slashes when matching or
        building URLs. Matches will redirect to the normalized URL.
        Slashes in variable parts are not merged.
    :param redirect_defaults: This will redirect to the default rule if it
                              wasn't visited that way. This helps creating
                              unique URLs.
    :param converters: A dict of converters that adds additional converters
                       to the list of converters. If you redefine one
                       converter this will override the original one.
    :param sort_parameters: If set to `True` the url parameters are sorted.
                            See `url_encode` for more details.
    :param sort_key: The sort key function for `url_encode`.
    :param host_matching: if set to `True` it enables the host matching
                          feature and disables the subdomain one.  If
                          enabled the `host` parameter to rules is used
                          instead of the `subdomain` one.

    .. versionchanged:: 3.0
        The ``charset`` and ``encoding_errors`` parameters were removed.

    .. versionchanged:: 1.0
        If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules will match.

    .. versionchanged:: 1.0
        The ``merge_slashes`` parameter was added.

    .. versionchanged:: 0.7
        The ``encoding_errors`` and ``host_matching`` parameters were added.

    .. versionchanged:: 0.5
        The ``sort_parameters`` and ``sort_key``  parameters were added.
    """

    #: A dict of default converters to be used.
    default_converters = ImmutableDict(DEFAULT_CONVERTERS)

    #: The type of lock to use when updating.
    #:
    #: .. versionadded:: 1.0
    lock_class = Lock

    def __init__(
        self,
        rules: t.Iterable[RuleFactory] | None = None,
        default_subdomain: str = "",
        strict_slashes: bool = True,
        merge_slashes: bool = True,
        redirect_defaults: bool = True,
        converters: t.Mapping[str, type[BaseConverter]] | None = None,
        sort_parameters: bool = False,
        sort_key: t.Callable[[t.Any], t.Any] | None = None,
        host_matching: bool = False,
    ) -> None:
        self._matcher = StateMachineMatcher(merge_slashes)
        self._rules_by_endpoint: dict[t.Any, list[Rule]] = {}
        self._remap = True
        self._remap_lock = self.lock_class()

        self.default_subdomain = default_subdomain
        self.strict_slashes = strict_slashes
        self.redirect_defaults = redirect_defaults
        self.host_matching = host_matching

        self.converters = self.default_converters.copy()
        if converters:
            self.converters.update(converters)

        self.sort_parameters = sort_parameters
        self.sort_key = sort_key

        for rulefactory in rules or ():
            self.add(rulefactory)

    @property
    def merge_slashes(self) -> bool:
        return self._matcher.merge_slashes

    @merge_slashes.setter
    def merge_slashes(self, value: bool) -> None:
        self._matcher.merge_slashes = value

    def is_endpoint_expecting(self, endpoint: t.Any, *arguments: str) -> bool:
        """Iterate over all rules and check if the endpoint expects
        the arguments provided.  This is for example useful if you have
        some URLs that expect a language code and others that do not and
        you want to wrap the builder a bit so that the current language
        code is automatically added if not provided but endpoints expect
        it.

        :param endpoint: the endpoint to check.
        :param arguments: this function accepts one or more arguments
                          as positional arguments.  Each one of them is
                          checked.
        """
        self.update()
        arguments_set = set(arguments)
        for rule in self._rules_by_endpoint[endpoint]:
            if arguments_set.issubset(rule.arguments):
                return True
        return False

    @property
    def _rules(self) -> list[Rule]:
        return [rule for rules in self._rules_by_endpoint.values() for rule in rules]

    def iter_rules(self, endpoint: t.Any | None = None) -> t.Iterator[Rule]:
        """Iterate over all rules or the rules of an endpoint.

        :param endpoint: if provided only the rules for that endpoint
                         are returned.
        :return: an iterator
        """
        self.update()
        if endpoint is not None:
            return iter(self._rules_by_endpoint[endpoint])
        return iter(self._rules)

    def add(self, rulefactory: RuleFactory) -> None:
        """Add a new rule or factory to the map and bind it.  Requires that the
        rule is not bound to another map.

        :param rulefactory: a :class:`Rule` or :class:`RuleFactory`
        """
        for rule in rulefactory.get_rules(self):
            rule.bind(self)
            if not rule.build_only:
                self._matcher.add(rule)
            self._rules_by_endpoint.setdefault(rule.endpoint, []).append(rule)
        self._remap = True

    def bind(
        self,
        server_name: str,
        script_name: str | None = None,
        subdomain: str | None = None,
        url_scheme: str = "http",
        default_method: str = "GET",
        path_info: str | None = None,
        query_args: t.Mapping[str, t.Any] | str | None = None,
    ) -> MapAdapter:
        """Return a new :class:`MapAdapter` with the details specified to the
        call.  Note that `script_name` will default to ``'/'`` if not further
        specified or `None`.  The `server_name` at least is a requirement
        because the HTTP RFC requires absolute URLs for redirects and so all
        redirect exceptions raised by Werkzeug will contain the full canonical
        URL.

        If no path_info is passed to :meth:`match` it will use the default path
        info passed to bind.  While this doesn't really make sense for
        manual bind calls, it's useful if you bind a map to a WSGI
        environment which already contains the path info.

        `subdomain` will default to the `default_subdomain` for this map if
        no defined. If there is no `default_subdomain` you cannot use the
        subdomain feature.

        .. versionchanged:: 1.0
            If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules
            will match.

        .. versionchanged:: 0.15
            ``path_info`` defaults to ``'/'`` if ``None``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionchanged:: 0.7
            Added ``query_args``.
        """
        server_name = server_name.lower()
        if self.host_matching:
            if subdomain is not None:
                raise RuntimeError("host matching enabled and a subdomain was provided")
        elif subdomain is None:
            subdomain = self.default_subdomain
        if script_name is None:
            script_name = "/"
        if path_info is None:
            path_info = "/"

        # Port isn't part of IDNA, and might push a name over the 63 octet limit.
        server_name, port_sep, port = server_name.partition(":")

        try:
            server_name = server_name.encode("idna").decode("ascii")
        except UnicodeError as e:
            raise BadHost() from e

        return MapAdapter(
            self,
            f"{server_name}{port_sep}{port}",
            script_name,
            subdomain,
            url_scheme,
            path_info,
            default_method,
            query_args,
        )

    def bind_to_environ(
        self,
        environ: WSGIEnvironment | Request,
        server_name: str | None = None,
        subdomain: str | None = None,
    ) -> MapAdapter:
        """Like :meth:`bind` but you can pass it an WSGI environment and it
        will fetch the information from that dictionary.  Note that because of
        limitations in the protocol there is no way to get the current
        subdomain and real `server_name` from the environment.  If you don't
        provide it, Werkzeug will use `SERVER_NAME` and `SERVER_PORT` (or
        `HTTP_HOST` if provided) as used `server_name` with disabled subdomain
        feature.

        If `subdomain` is `None` but an environment and a server name is
        provided it will calculate the current subdomain automatically.
        Example: `server_name` is ``'example.com'`` and the `SERVER_NAME`
        in the wsgi `environ` is ``'staging.dev.example.com'`` the calculated
        subdomain will be ``'staging.dev'``.

        If the object passed as environ has an environ attribute, the value of
        this attribute is used instead.  This allows you to pass request
        objects.  Additionally `PATH_INFO` added as a default of the
        :class:`MapAdapter` so that you don't have to pass the path info to
        the match method.

        .. versionchanged:: 1.0.0
            If the passed server name specifies port 443, it will match
            if the incoming scheme is ``https`` without a port.

        .. versionchanged:: 1.0.0
            A warning is shown when the passed server name does not
            match the incoming WSGI server name.

        .. versionchanged:: 0.8
           This will no longer raise a ValueError when an unexpected server
           name was passed.

        .. versionchanged:: 0.5
            previously this method accepted a bogus `calculate_subdomain`
            parameter that did not have any effect.  It was removed because
            of that.

        :param environ: a WSGI environment.
        :param server_name: an optional server name hint (see above).
        :param subdomain: optionally the current subdomain (see above).
        """
        env = _get_environ(environ)
        wsgi_server_name = get_host(env).lower()
        scheme = env["wsgi.url_scheme"]
        upgrade = any(
            v.strip(" \t") == "upgrade"
            for v in env.get("HTTP_CONNECTION", "").lower().split(",")
        )

        if upgrade and env.get("HTTP_UPGRADE", "").lower() == "websocket":
            scheme = "wss" if scheme == "https" else "ws"

        if server_name is None:
            server_name = wsgi_server_name
        else:
            server_name = server_name.lower()

            # strip standard port to match get_host()
            if scheme in {"http", "ws"} and server_name.endswith(":80"):
                server_name = server_name[:-3]
            elif scheme in {"https", "wss"} and server_name.endswith(":443"):
                server_name = server_name[:-4]

        if subdomain is None and not self.host_matching:
            cur_server_name = wsgi_server_name.split(".")
            real_server_name = server_name.split(".")
            offset = -len(real_server_name)

            if cur_server_name[offset:] != real_server_name:
                # This can happen even with valid configs if the server was
                # accessed directly by IP address under some situations.
                # Instead of raising an exception like in Werkzeug 0.7 or
                # earlier we go by an invalid subdomain which will result
                # in a 404 error on matching.
                warnings.warn(
                    f"Current server name {wsgi_server_name!r} doesn't match configured"
                    f" server name {server_name!r}",
                    stacklevel=2,
                )
                subdomain = "<invalid>"
            else:
                subdomain = ".".join(filter(None, cur_server_name[:offset]))

        def _get_wsgi_string(name: str) -> str | None:
            val = env.get(name)
            if val is not None:
                return _wsgi_decoding_dance(val)
            return None

        script_name = _get_wsgi_string("SCRIPT_NAME")
        path_info = _get_wsgi_string("PATH_INFO")
        query_args = _get_wsgi_string("QUERY_STRING")
        return Map.bind(
            self,
            server_name,
            script_name,
            subdomain,
            scheme,
            env["REQUEST_METHOD"],
            path_info,
            query_args=query_args,
        )

    def update(self) -> None:
        """Called before matching and building to keep the compiled rules
        in the correct order after things changed.
        """
        if not self._remap:
            return

        with self._remap_lock:
            if not self._remap:
                return

            self._matcher.update()
            for rules in self._rules_by_endpoint.values():
                rules.sort(key=lambda x: x.build_compare_key())
            self._remap = False

    def __repr__(self) -> str:
        rules = self.iter_rules()
        return f"{type(self).__name__}({pformat(list(rules))})"


class MapAdapter:
    """Returned by :meth:`Map.bind` or :meth:`Map.bind_to_environ` and does
    the URL matching and building based on runtime information.
    """

    def __init__(
        self,
        map: Map,
        server_name: str,
        script_name: str,
        subdomain: str | None,
        url_scheme: str,
        path_info: str,
        default_method: str,
        query_args: t.Mapping[str, t.Any] | str | None = None,
    ):
        self.map = map
        self.server_name = server_name

        if not script_name.endswith("/"):
            script_name += "/"

        self.script_name = script_name
        self.subdomain = subdomain
        self.url_scheme = url_scheme
        self.path_info = path_info
        self.default_method = default_method
        self.query_args = query_args
        self.websocket = self.url_scheme in {"ws", "wss"}

    def dispatch(
        self,
        view_func: t.Callable[[str, t.Mapping[str, t.Any]], WSGIApplication],
        path_info: str | None = None,
        method: str | None = None,
        catch_http_exceptions: bool = False,
    ) -> WSGIApplication:
        """Does the complete dispatching process.  `view_func` is called with
        the endpoint and a dict with the values for the view.  It should
        look up the view function, call it, and return a response object
        or WSGI application.  http exceptions are not caught by default
        so that applications can display nicer error messages by just
        catching them by hand.  If you want to stick with the default
        error messages you can pass it ``catch_http_exceptions=True`` and
        it will catch the http exceptions.

        Here a small example for the dispatch usage::

            from werkzeug.wrappers import Request, Response
            from werkzeug.wsgi import responder
            from werkzeug.routing import Map, Rule

            def on_index(request):
                return Response('Hello from the index')

            url_map = Map([Rule('/', endpoint='index')])
            views = {'index': on_index}

            @responder
            def application(environ, start_response):
                request = Request(environ)
                urls = url_map.bind_to_environ(environ)
                return urls.dispatch(lambda e, v: views[e](request, **v),
                                     catch_http_exceptions=True)

        Keep in mind that this method might return exception objects, too, so
        use :class:`Response.force_type` to get a response object.

        :param view_func: a function that is called with the endpoint as
                          first argument and the value dict as second.  Has
                          to dispatch to the actual view function with this
                          information.  (see above)
        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param catch_http_exceptions: set to `True` to catch any of the
                                      werkzeug :class:`HTTPException`\\s.
        """
        try:
            try:
                endpoint, args = self.match(path_info, method)
            except RequestRedirect as e:
                return e
            return view_func(endpoint, args)
        except HTTPException as e:
            if catch_http_exceptions:
                return e
            raise

    @t.overload
    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: t.Literal[False] = False,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[t.Any, t.Mapping[str, t.Any]]: ...

    @t.overload
    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: t.Literal[True] = True,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[Rule, t.Mapping[str, t.Any]]: ...

    def match(
        self,
        path_info: str | None = None,
        method: str | None = None,
        return_rule: bool = False,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        websocket: bool | None = None,
    ) -> tuple[t.Any | Rule, t.Mapping[str, t.Any]]:
        """The usage is simple: you just pass the match method the current
        path info as well as the method (which defaults to `GET`).  The
        following things can then happen:

        - you receive a `NotFound` exception that indicates that no URL is
          matching.  A `NotFound` exception is also a WSGI application you
          can call to get a default page not found page (happens to be the
          same object as `werkzeug.exceptions.NotFound`)

        - you receive a `MethodNotAllowed` exception that indicates that there
          is a match for this URL but not for the current request method.
          This is useful for RESTful applications.

        - you receive a `RequestRedirect` exception with a `new_url`
          attribute.  This exception is used to notify you about a request
          Werkzeug requests from your WSGI application.  This is for example the
          case if you request ``/foo`` although the correct URL is ``/foo/``
          You can use the `RequestRedirect` instance as response-like object
          similar to all other subclasses of `HTTPException`.

        - you receive a ``WebsocketMismatch`` exception if the only
          match is a WebSocket rule but the bind is an HTTP request, or
          if the match is an HTTP rule but the bind is a WebSocket
          request.

        - you get a tuple in the form ``(endpoint, arguments)`` if there is
          a match (unless `return_rule` is True, in which case you get a tuple
          in the form ``(rule, arguments)``)

        If the path info is not passed to the match method the default path
        info of the map is used (defaults to the root URL if not defined
        explicitly).

        All of the exceptions raised are subclasses of `HTTPException` so they
        can be used as WSGI responses. They will all render generic error or
        redirect pages.

        Here is a small example for matching:

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.match("/", "GET")
        ('index', {})
        >>> urls.match("/downloads/42")
        ('downloads/show', {'id': 42})

        And here is what happens on redirect and missing URLs:

        >>> urls.match("/downloads")
        Traceback (most recent call last):
          ...
        RequestRedirect: http://example.com/downloads/
        >>> urls.match("/missing")
        Traceback (most recent call last):
          ...
        NotFound: 404 Not Found

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param return_rule: return the rule that matched instead of just the
                            endpoint (defaults to `False`).
        :param query_args: optional query arguments that are used for
                           automatic redirects as string or dictionary.  It's
                           currently not possible to use the query arguments
                           for URL matching.
        :param websocket: Match WebSocket instead of HTTP requests. A
            websocket request has a ``ws`` or ``wss``
            :attr:`url_scheme`. This overrides that detection.

        .. versionadded:: 1.0
            Added ``websocket``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionadded:: 0.7
            Added ``query_args``.

        .. versionadded:: 0.6
            Added ``return_rule``.
        """
        self.map.update()
        if path_info is None:
            path_info = self.path_info
        if query_args is None:
            query_args = self.query_args or {}
        method = (method or self.default_method).upper()

        if websocket is None:
            websocket = self.websocket

        domain_part = self.server_name

        if not self.map.host_matching and self.subdomain is not None:
            domain_part = self.subdomain

        path_part = f"/{path_info.lstrip('/')}" if path_info else ""

        try:
            result = self.map._matcher.match(domain_part, path_part, method, websocket)
        except RequestPath as e:
            # safe = https://url.spec.whatwg.org/#url-path-segment-string
            new_path = quote(e.path_info, safe="!$&'()*+,/:;=@")
            raise RequestRedirect(
                self.make_redirect_url(new_path, query_args)
            ) from None
        except RequestAliasRedirect as e:
            raise RequestRedirect(
                self.make_alias_redirect_url(
                    f"{domain_part}|{path_part}",
                    e.endpoint,
                    e.matched_values,
                    method,
                    query_args,
                )
            ) from None
        except NoMatch as e:
            if e.have_match_for:
                raise MethodNotAllowed(valid_methods=list(e.have_match_for)) from None

            if e.websocket_mismatch:
                raise WebsocketMismatch() from None

            raise NotFound() from None
        else:
            rule, rv = result

            if self.map.redirect_defaults:
                redirect_url = self.get_default_redirect(rule, method, rv, query_args)
                if redirect_url is not None:
                    raise RequestRedirect(redirect_url)

            if rule.redirect_to is not None:
                if isinstance(rule.redirect_to, str):

                    def _handle_match(match: t.Match[str]) -> str:
                        value = rv[match.group(1)]
                        return rule._converters[match.group(1)].to_url(value)

                    redirect_url = _simple_rule_re.sub(_handle_match, rule.redirect_to)
                else:
                    redirect_url = rule.redirect_to(self, **rv)

                if self.subdomain:
                    netloc = f"{self.subdomain}.{self.server_name}"
                else:
                    netloc = self.server_name

                raise RequestRedirect(
                    urljoin(
                        f"{self.url_scheme or 'http'}://{netloc}{self.script_name}",
                        redirect_url,
                    )
                )

            if return_rule:
                return rule, rv
            else:
                return rule.endpoint, rv

    def test(self, path_info: str | None = None, method: str | None = None) -> bool:
        """Test if a rule would match.  Works like `match` but returns `True`
        if the URL matches, or `False` if it does not exist.

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        """
        try:
            self.match(path_info, method)
        except RequestRedirect:
            pass
        except HTTPException:
            return False
        return True

    def allowed_methods(self, path_info: str | None = None) -> t.Iterable[str]:
        """Returns the valid methods that match for a given path.

        .. versionadded:: 0.7
        """
        try:
            self.match(path_info, method="--")
        except MethodNotAllowed as e:
            return e.valid_methods  # type: ignore
        except HTTPException:
            pass
        return []

    def get_host(self, domain_part: str | None) -> str:
        """Figures out the full host name for the given domain part.  The
        domain part is a subdomain in case host matching is disabled or
        a full host name.
        """
        if self.map.host_matching:
            if domain_part is None:
                return self.server_name

            return domain_part

        if domain_part is None:
            subdomain = self.subdomain
        else:
            subdomain = domain_part

        if subdomain:
            return f"{subdomain}.{self.server_name}"
        else:
            return self.server_name

    def get_default_redirect(
        self,
        rule: Rule,
        method: str,
        values: t.MutableMapping[str, t.Any],
        query_args: t.Mapping[str, t.Any] | str,
    ) -> str | None:
        """A helper that returns the URL to redirect to if it finds one.
        This is used for default redirecting only.

        :internal:
        """
        assert self.map.redirect_defaults
        for r in self.map._rules_by_endpoint[rule.endpoint]:
            # every rule that comes after this one, including ourself
            # has a lower priority for the defaults.  We order the ones
            # with the highest priority up for building.
            if r is rule:
                break
            if r.provides_defaults_for(rule) and r.suitable_for(values, method):
                values.update(r.defaults)  # type: ignore
                domain_part, path = r.build(values)  # type: ignore
                return self.make_redirect_url(path, query_args, domain_part=domain_part)
        return None

    def encode_query_args(self, query_args: t.Mapping[str, t.Any] | str) -> str:
        if not isinstance(query_args, str):
            return _urlencode(query_args)
        return query_args

    def make_redirect_url(
        self,
        path_info: str,
        query_args: t.Mapping[str, t.Any] | str | None = None,
        domain_part: str | None = None,
    ) -> str:
        """Creates a redirect URL.

        :internal:
        """
        if query_args is None:
            query_args = self.query_args

        if query_args:
            query_str = self.encode_query_args(query_args)
        else:
            query_str = None

        scheme = self.url_scheme or "http"
        host = self.get_host(domain_part)
        path = "/".join((self.script_name.strip("/"), path_info.lstrip("/")))
        return urlunsplit((scheme, host, path, query_str, None))

    def make_alias_redirect_url(
        self,
        path: str,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str,
        query_args: t.Mapping[str, t.Any] | str,
    ) -> str:
        """Internally called to make an alias redirect URL."""
        url = self.build(
            endpoint, values, method, append_unknown=False, force_external=True
        )
        if query_args:
            url += f"?{self.encode_query_args(query_args)}"
        assert url != path, "detected invalid alias setting. No canonical URL found"
        return url

    def _partial_build(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str | None,
        append_unknown: bool,
    ) -> tuple[str, str, bool] | None:
        """Helper for :meth:`build`.  Returns subdomain and path for the
        rule that accepts this endpoint, values and method.

        :internal:
        """
        # in case the method is none, try with the default method first
        if method is None:
            rv = self._partial_build(
                endpoint, values, self.default_method, append_unknown
            )
            if rv is not None:
                return rv

        # Default method did not match or a specific method is passed.
        # Check all for first match with matching host. If no matching
        # host is found, go with first result.
        first_match = None

        for rule in self.map._rules_by_endpoint.get(endpoint, ()):
            if rule.suitable_for(values, method):
                build_rv = rule.build(values, append_unknown)

                if build_rv is not None:
                    rv = (build_rv[0], build_rv[1], rule.websocket)
                    if self.map.host_matching:
                        if rv[0] == self.server_name:
                            return rv
                        elif first_match is None:
                            first_match = rv
                    else:
                        return rv

        return first_match

    def build(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any] | None = None,
        method: str | None = None,
        force_external: bool = False,
        append_unknown: bool = True,
        url_scheme: str | None = None,
    ) -> str:
        """Build a URL by filling in any variable slots in the endpoint's rule
        with the provided values.

        This produces relative URLs (path only, no host) by default, which is
        appropriate for most use cases. When using subdomain or host matching,
        the URL will be absolute if the endpoint is not on the same host as the
        current request. If you need an absolute URL no matter what, such as
        when adding a link to an email, pass ``force_external=True``.

        Characters that are not allowed directly in URLs will be percent-encoded
        using UTF-8.

        Additional values that don't match any rule parts are added to the query
        (``?key=value``) part of the URL.

        If a rule is not found, an :exc:`.BuildError` is raised. Converters may
        also raise exceptions when preparing a value.

        Unlike matching, building does not perform comprehensive validation of
        the values, it's assumed that they are trusted and correct. It's also
        possible to build a URL for one endpoint that will be matched by
        another, or will not match. This is very unlikely to happen or to matter
        in most applications, especially with testing. If you need a
        stronger guarantee, you can use a helper to verify that
        ``match(build(endpoint)) == endpoint``.

        :param endpoint: The endpoint for the rule to build.
        :param values: Values for the variable parts of the rule, which will be
            converted using the part's converter's
            :meth:`~.BaseConverter.to_url` method. Keys that don't match
            variable parts will be converted to a query string unless
            ``append_unknown`` is disabled.
        :param method: Further match the rule to build by this HTTP method, for
            when there are multiple rules for the same endpoint that vary based
            on method.
        :param force_external: Always produce an absolute URL, for external use.
            If ``scheme`` is not set, this produces a protocol-relative URL.
        :param append_unknown: Convert values that don't match any rule parts
            to a query string. Disable to ignore such values.
        :param url_scheme: Scheme to use for absolute URLs. Defaults to
            :attr:`url_scheme`.

        .. versionchanged:: 2.0
            Added the ``url_scheme`` parameter.

        .. versionadded:: 0.6
           Added the ``append_unknown`` parameter.
        """
        self.map.update()

        if values:
            if isinstance(values, MultiDict):
                values = {
                    k: (v[0] if len(v) == 1 else v)
                    for k, v in dict.items(values)
                    if len(v) != 0
                }
            else:  # plain dict
                values = {k: v for k, v in values.items() if v is not None}
        else:
            values = {}

        rv = self._partial_build(endpoint, values, method, append_unknown)
        if rv is None:
            raise BuildError(endpoint, values, method, self)

        domain_part, path, websocket = rv
        host = self.get_host(domain_part)

        if url_scheme is None:
            url_scheme = self.url_scheme

        # Always build WebSocket routes with the scheme (browsers
        # require full URLs). If bound to a WebSocket, ensure that HTTP
        # routes are built with an HTTP scheme.
        secure = url_scheme in {"https", "wss"}

        if websocket:
            force_external = True
            url_scheme = "wss" if secure else "ws"
        elif url_scheme:
            url_scheme = "https" if secure else "http"

        # shortcut this.
        if not force_external and (
            (self.map.host_matching and host == self.server_name)
            or (not self.map.host_matching and domain_part == self.subdomain)
        ):
            return f"{self.script_name.rstrip('/')}/{path.lstrip('/')}"

        scheme = f"{url_scheme}:" if url_scheme else ""
        return f"{scheme}//{host}{self.script_name[:-1]}/{path.lstrip('/')}"
<|endoftext|>"""When it comes to combining multiple controller or view functions
(however you want to call them) you need a dispatcher. A simple way
would be applying regular expression tests on the ``PATH_INFO`` and
calling registered callback functions that return the value then.

This module implements a much more powerful system than simple regular
expression matching because it can also convert values in the URLs and
build URLs.

Here a simple example that creates a URL map for an application with
two subdomains (www and kb) and some URL rules:

.. code-block:: python

    m = Map([
        # Static URLs
        Rule('/', endpoint='static/index'),
        Rule('/about', endpoint='static/about'),
        Rule('/help', endpoint='static/help'),
        # Knowledge Base
        Subdomain('kb', [
            Rule('/', endpoint='kb/index'),
            Rule('/browse/', endpoint='kb/browse'),
            Rule('/browse/<int:id>/', endpoint='kb/browse'),
            Rule('/browse/<int:id>/<int:page>', endpoint='kb/browse')
        ])
    ], default_subdomain='www')

If the application doesn't use subdomains it's perfectly fine to not set
the default subdomain and not use the `Subdomain` rule factory. The
endpoint in the rules can be anything, for example import paths or
unique identifiers. The WSGI application can use those endpoints to get the
handler for that URL.  It doesn't have to be a string at all but it's
recommended.

Now it's possible to create a URL adapter for one of the subdomains and
build URLs:

.. code-block:: python

    c = m.bind('example.com')

    c.build("kb/browse", dict(id=42))
    'http://kb.example.com/browse/42/'

    c.build("kb/browse", dict())
    'http://kb.example.com/browse/'

    c.build("kb/browse", dict(id=42, page=3))
    'http://kb.example.com/browse/42/3'

    c.build("static/about")
    '/about'

    c.build("static/index", force_external=True)
    'http://www.example.com/'

    c = m.bind('example.com', subdomain='kb')

    c.build("static/about")
    'http://www.example.com/about'

The first argument to bind is the server name *without* the subdomain.
Per default it will assume that the script is mounted on the root, but
often that's not the case so you can provide the real mount point as
second argument:

.. code-block:: python

    c = m.bind('example.com', '/applications/example')

The third argument can be the subdomain, if not given the default
subdomain is used.  For more details about binding have a look at the
documentation of the `MapAdapter`.

And here is how you can match URLs:

.. code-block:: python

    c = m.bind('example.com')

    c.match("/")
    ('static/index', {})

    c.match("/about")
    ('static/about', {})

    c = m.bind('example.com', '/', 'kb')

    c.match("/")
    ('kb/index', {})

    c.match("/browse/42/23")
    ('kb/browse', {'id': 42, 'page': 23})

If matching fails you get a ``NotFound`` exception, if the rule thinks
it's a good idea to redirect (for example because the URL was defined
to have a slash at the end but the request was missing that slash) it
will raise a ``RequestRedirect`` exception. Both are subclasses of
``HTTPException`` so you can use those errors as responses in the
application.

If matching succeeded but the URL rule was incompatible to the given
method (for example there were only rules for ``GET`` and ``HEAD`` but
routing tried to match a ``POST`` request) a ``MethodNotAllowed``
exception is raised.
"""

from .converters import AnyConverter as AnyConverter
from .converters import BaseConverter as BaseConverter
from .converters import FloatConverter as FloatConverter
from .converters import IntegerConverter as IntegerConverter
from .converters import PathConverter as PathConverter
from .converters import UnicodeConverter as UnicodeConverter
from .converters import UUIDConverter as UUIDConverter
from .converters import ValidationError as ValidationError
from .exceptions import BuildError as BuildError
from .exceptions import NoMatch as NoMatch
from .exceptions import RequestAliasRedirect as RequestAliasRedirect
from .exceptions import RequestPath as RequestPath
from .exceptions import RequestRedirect as RequestRedirect
from .exceptions import RoutingException as RoutingException
from .exceptions import WebsocketMismatch as WebsocketMismatch
from .map import Map as Map
from .map import MapAdapter as MapAdapter
from .matcher import StateMachineMatcher as StateMachineMatcher
from .rules import EndpointPrefix as EndpointPrefix
from .rules import parse_converter_args as parse_converter_args
from .rules import Rule as Rule
from .rules import RuleFactory as RuleFactory
from .rules import RuleTemplate as RuleTemplate
from .rules import RuleTemplateFactory as RuleTemplateFactory
from .rules import Subdomain as Subdomain
from .rules import Submount as Submount
<|endoftext|>from __future__ import annotations

import re
import typing as t
import uuid
from urllib.parse import quote

if t.TYPE_CHECKING:
    from .map import Map


class ValidationError(ValueError):
    """Validation error.  If a rule converter raises this exception the rule
    does not match the current URL and the next URL is tried.
    """


class BaseConverter:
    """Base class for all converters.

    .. versionchanged:: 2.3
        ``part_isolating`` defaults to ``False`` if ``regex`` contains a ``/``.
    """

    regex = "[^/]+"
    weight = 100
    part_isolating = True

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        # If the converter isn't inheriting its regex, disable part_isolating by default
        # if the regex contains a / character.
        if "regex" in cls.__dict__ and "part_isolating" not in cls.__dict__:
            cls.part_isolating = "/" not in cls.regex

    def __init__(self, map: Map, *args: t.Any, **kwargs: t.Any) -> None:
        self.map = map

    def to_python(self, value: str) -> t.Any:
        return value

    def to_url(self, value: t.Any) -> str:
        # safe = https://url.spec.whatwg.org/#url-path-segment-string
        return quote(str(value), safe="!$&'()*+,/:;=@")


class UnicodeConverter(BaseConverter):
    """This converter is the default converter and accepts any string but
    only one path segment.  Thus the string can not include a slash.

    This is the default validator.

    Example::

        Rule('/pages/<page>'),
        Rule('/<string(length=2):lang_code>')

    :param map: the :class:`Map`.
    :param minlength: the minimum length of the string.  Must be greater
                      or equal 1.
    :param maxlength: the maximum length of the string.
    :param length: the exact length of the string.
    """

    def __init__(
        self,
        map: Map,
        minlength: int = 1,
        maxlength: int | None = None,
        length: int | None = None,
    ) -> None:
        super().__init__(map)
        if length is not None:
            length_regex = f"{{{int(length)}}}"
        else:
            if maxlength is None:
                maxlength_value = ""
            else:
                maxlength_value = str(int(maxlength))
            length_regex = f"{{{int(minlength)},{maxlength_value}}}"
        self.regex = f"[^/]{length_regex}"


class AnyConverter(BaseConverter):
    """Matches one of the items provided.  Items can either be Python
    identifiers or strings::

        Rule('/<any(about, help, imprint, class, "foo,bar"):page_name>')

    :param map: the :class:`Map`.
    :param items: this function accepts the possible items as positional
                  arguments.

    .. versionchanged:: 2.2
        Value is validated when building a URL.
    """

    def __init__(self, map: Map, *items: str) -> None:
        super().__init__(map)
        self.items = set(items)
        self.regex = f"(?:{'|'.join([re.escape(x) for x in items])})"

    def to_url(self, value: t.Any) -> str:
        if value in self.items:
            return str(value)

        valid_values = ", ".join(f"'{item}'" for item in sorted(self.items))
        raise ValueError(f"'{value}' is not one of {valid_values}")


class PathConverter(BaseConverter):
    """Like the default :class:`UnicodeConverter`, but it also matches
    slashes.  This is useful for wikis and similar applications::

        Rule('/<path:wikipage>')
        Rule('/<path:wikipage>/edit')

    :param map: the :class:`Map`.
    """

    part_isolating = False
    regex = "[^/].*?"
    weight = 200


class NumberConverter(BaseConverter):
    """Baseclass for `IntegerConverter` and `FloatConverter`.

    :internal:
    """

    weight = 50
    num_convert: t.Callable[[t.Any], t.Any] = int

    def __init__(
        self,
        map: Map,
        fixed_digits: int = 0,
        min: int | None = None,
        max: int | None = None,
        signed: bool = False,
    ) -> None:
        if signed:
            self.regex = self.signed_regex
        super().__init__(map)
        self.fixed_digits = fixed_digits
        self.min = min
        self.max = max
        self.signed = signed

    def to_python(self, value: str) -> t.Any:
        if self.fixed_digits and len(value) != self.fixed_digits:
            raise ValidationError()

        try:
            value_num = self.num_convert(value)
        except ValueError as e:
            raise ValidationError() from e

        if (self.min is not None and value_num < self.min) or (
            self.max is not None and value_num > self.max
        ):
            raise ValidationError()
        return value_num

    def to_url(self, value: t.Any) -> str:
        value_str = str(self.num_convert(value))
        if self.fixed_digits:
            value_str = value_str.zfill(self.fixed_digits)
        return value_str

    @property
    def signed_regex(self) -> str:
        return f"-?{self.regex}"


class IntegerConverter(NumberConverter):
    """This converter only accepts integer values::

        Rule("/page/<int:page>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/page/<int(signed=True):page>")

    :param map: The :class:`Map`.
    :param fixed_digits: The number of fixed digits in the URL. If you
        set this to ``4`` for example, the rule will only match if the
        URL looks like ``/0001/``. The default is variable length.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+"


class FloatConverter(NumberConverter):
    """This converter only accepts floating point values::

        Rule("/probability/<float:probability>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/offset/<float(signed=True):offset>")

    :param map: The :class:`Map`.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+\.\d+"
    num_convert = float

    def __init__(
        self,
        map: Map,
        min: float | None = None,
        max: float | None = None,
        signed: bool = False,
    ) -> None:
        super().__init__(map, min=min, max=max, signed=signed)  # type: ignore


class UUIDConverter(BaseConverter):
    """This converter only accepts UUID strings::

        Rule('/object/<uuid:identifier>')

    .. versionadded:: 0.10

    :param map: the :class:`Map`.
    """

    regex = (
        r"[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-"
        r"[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}"
    )

    def to_python(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

    def to_url(self, value: uuid.UUID) -> str:
        return str(value)


#: the default converter mapping for the map.
DEFAULT_CONVERTERS: t.Mapping[str, type[BaseConverter]] = {
    "default": UnicodeConverter,
    "string": UnicodeConverter,
    "any": AnyConverter,
    "path": PathConverter,
    "int": IntegerConverter,
    "float": FloatConverter,
    "uuid": UUIDConverter,
}
<|endoftext|>from __future__ import annotations

import difflib
import typing as t

from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

    from ..wrappers.request import Request
    from ..wrappers.response import Response
    from .map import MapAdapter
    from .rules import Rule


class RoutingException(Exception):
    """Special exceptions that require the application to redirect, notifying
    about missing urls, etc.

    :internal:
    """


class RequestRedirect(HTTPException, RoutingException):
    """Raise if the map requests a redirect. This is for example the case if
    `strict_slashes` are activated and an url that requires a trailing slash.

    The attribute `new_url` contains the absolute destination url.
    """

    code = 308

    def __init__(self, new_url: str) -> None:
        super().__init__(new_url)
        self.new_url = new_url

    def get_response(
        self,
        environ: WSGIEnvironment | Request | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> Response:
        return redirect(self.new_url, self.code)


class RequestPath(RoutingException):
    """Internal exception."""

    __slots__ = ("path_info",)

    def __init__(self, path_info: str) -> None:
        super().__init__()
        self.path_info = path_info


class RequestAliasRedirect(RoutingException):  # noqa: B903
    """This rule is an alias and wants to redirect to the canonical URL."""

    def __init__(self, matched_values: t.Mapping[str, t.Any], endpoint: t.Any) -> None:
        super().__init__()
        self.matched_values = matched_values
        self.endpoint = endpoint


class BuildError(RoutingException, LookupError):
    """Raised if the build system cannot find a URL for an endpoint with the
    values provided.
    """

    def __init__(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str | None,
        adapter: MapAdapter | None = None,
    ) -> None:
        super().__init__(endpoint, values, method)
        self.endpoint = endpoint
        self.values = values
        self.method = method
        self.adapter = adapter

    @cached_property
    def suggested(self) -> Rule | None:
        return self.closest_rule(self.adapter)

    def closest_rule(self, adapter: MapAdapter | None) -> Rule | None:
        def _score_rule(rule: Rule) -> float:
            return sum(
                [
                    0.98
                    * difflib.SequenceMatcher(
                        # endpoints can be any type, compare as strings
                        None,
                        str(rule.endpoint),
                        str(self.endpoint),
                    ).ratio(),
                    0.01 * bool(set(self.values or ()).issubset(rule.arguments)),
                    0.01 * bool(rule.methods and self.method in rule.methods),
                ]
            )

        if adapter and adapter.map._rules:
            return max(adapter.map._rules, key=_score_rule)

        return None

    def __str__(self) -> str:
        message = [f"Could not build url for endpoint {self.endpoint!r}"]
        if self.method:
            message.append(f" ({self.method!r})")
        if self.values:
            message.append(f" with values {sorted(self.values)!r}")
        message.append(".")
        if self.suggested:
            if self.endpoint == self.suggested.endpoint:
                if (
                    self.method
                    and self.suggested.methods is not None
                    and self.method not in self.suggested.methods
                ):
                    message.append(
                        " Did you mean to use methods"
                        f" {sorted(self.suggested.methods)!r}?"
                    )
                missing_values = self.suggested.arguments.union(
                    set(self.suggested.defaults or ())
                ) - set(self.values.keys())
                if missing_values:
                    message.append(
                        f" Did you forget to specify values {sorted(missing_values)!r}?"
                    )
            else:
                message.append(f" Did you mean {self.suggested.endpoint!r} instead?")
        return "".join(message)


class WebsocketMismatch(BadRequest):
    """The only matched rule is either a WebSocket and the request is
    HTTP, or the rule is HTTP and the request is a WebSocket.
    """


class NoMatch(Exception):
    __slots__ = ("have_match_for", "websocket_mismatch")

    def __init__(self, have_match_for: set[str], websocket_mismatch: bool) -> None:
        self.have_match_for = have_match_for
        self.websocket_mismatch = websocket_mismatch
<|endoftext|>"""
Application Profiler
====================

This module provides a middleware that profiles each request with the
:mod:`cProfile` module. This can help identify bottlenecks in your code
that may be slowing down your application.

.. autoclass:: ProfilerMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import os.path
import sys
import time
import typing as t
from pstats import Stats

if sys.version_info >= (3, 15):
    from profiling.tracing import Profile
else:
    try:
        from cProfile import Profile
    except ImportError:
        from profile import Profile  # type: ignore[assignment]

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class ProfilerMiddleware:
    """Wrap a WSGI application and profile the execution of each
    request. Responses are buffered so that timings are more exact.

    If ``stream`` is given, :class:`pstats.Stats` are written to it
    after each request. If ``profile_dir`` is given, :mod:`cProfile`
    data files are saved to that directory, one file per request.

    The filename can be customized by passing ``filename_format``. If
    it is a string, it will be formatted using :meth:`str.format` with
    the following fields available:

    -   ``{method}`` - The request method; GET, POST, etc.
    -   ``{path}`` - The request path or 'root' should one not exist.
    -   ``{elapsed}`` - The elapsed time of the request in milliseconds.
    -   ``{time}`` - The time of the request.

    If it is a callable, it will be called with the WSGI ``environ`` and
    be expected to return a filename string. The ``environ`` dictionary
    will also have the ``"werkzeug.profiler"`` key populated with a
    dictionary containing the following fields (more may be added in the
    future):
    -   ``{elapsed}`` - The elapsed time of the request in milliseconds.
    -   ``{time}`` - The time of the request.

    :param app: The WSGI application to wrap.
    :param stream: Write stats to this stream. Disable with ``None``.
    :param sort_by: A tuple of columns to sort stats by. See
        :meth:`pstats.Stats.sort_stats`.
    :param restrictions: A tuple of restrictions to filter stats by. See
        :meth:`pstats.Stats.print_stats`.
    :param profile_dir: Save profile data files to this directory.
    :param filename_format: Format string for profile data file names,
        or a callable returning a name. See explanation above.

    .. code-block:: python

        from werkzeug.middleware.profiler import ProfilerMiddleware
        app = ProfilerMiddleware(app)

    .. versionchanged:: 3.0
        Added the ``"werkzeug.profiler"`` key to the ``filename_format(environ)``
        parameter with the  ``elapsed`` and ``time`` fields.

    .. versionchanged:: 0.15
        Stats are written even if ``profile_dir`` is given, and can be
        disable by passing ``stream=None``.

    .. versionadded:: 0.15
        Added ``filename_format``.

    .. versionadded:: 0.9
        Added ``restrictions`` and ``profile_dir``.
    """

    def __init__(
        self,
        app: WSGIApplication,
        stream: t.IO[str] | None = sys.stdout,
        sort_by: t.Iterable[str] = ("time", "calls"),
        restrictions: t.Iterable[str | int | float] = (),
        profile_dir: str | None = None,
        filename_format: str = "{method}.{path}.{elapsed:.0f}ms.{time:.0f}.prof",
    ) -> None:
        self._app = app
        self._stream = stream
        self._sort_by = sort_by
        self._restrictions = restrictions
        self._profile_dir = profile_dir
        self._filename_format = filename_format

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        response_body: list[bytes] = []

        def catching_start_response(status, headers, exc_info=None):  # type: ignore
            start_response(status, headers, exc_info)
            return response_body.append

        def runapp() -> None:
            app_iter = self._app(
                environ, t.cast("StartResponse", catching_start_response)
            )
            response_body.extend(app_iter)

            if hasattr(app_iter, "close"):
                app_iter.close()

        profile = Profile()
        start = time.time()
        profile.runcall(runapp)
        body = b"".join(response_body)
        elapsed = time.time() - start

        if self._profile_dir is not None:
            if callable(self._filename_format):
                environ["werkzeug.profiler"] = {
                    "elapsed": elapsed * 1000.0,
                    "time": time.time(),
                }
                filename = self._filename_format(environ)
            else:
                filename = self._filename_format.format(
                    method=environ["REQUEST_METHOD"],
                    path=environ["PATH_INFO"].strip("/").replace("/", ".") or "root",
                    elapsed=elapsed * 1000.0,
                    time=time.time(),
                )
            filename = os.path.join(self._profile_dir, filename)
            profile.dump_stats(filename)

        if self._stream is not None:
            stats = Stats(profile, stream=self._stream)
            stats.sort_stats(*self._sort_by)
            print("-" * 80, file=self._stream)
            path_info = environ.get("PATH_INFO", "")
            print(f"PATH: {path_info!r}", file=self._stream)
            stats.print_stats(*self._restrictions)
            print(f"{'-' * 80}\n", file=self._stream)

        return [body]
<|endoftext|>"""
X-Forwarded-For Proxy Fix
=========================

This module provides a middleware that adjusts the WSGI environ based on
``X-Forwarded-`` headers that proxies in front of an application may
set.

When an application is running behind a proxy server, WSGI may see the
request as coming from that server rather than the real client. Proxies
set various headers to track where the request actually came from.

This middleware should only be used if the application is actually
behind such a proxy, and should be configured with the number of proxies
that are chained in front of it. Not all proxies set all the headers.
Since incoming headers can be faked, you must set how many proxies are
setting each header so the middleware knows what to trust.

.. autoclass:: ProxyFix

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import typing as t

from ..http import parse_list_header

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class ProxyFix:
    """Adjust the WSGI environ based on ``X-Forwarded-`` that proxies in
    front of the application may set.

    -   ``X-Forwarded-For`` sets ``REMOTE_ADDR``.
    -   ``X-Forwarded-Proto`` sets ``wsgi.url_scheme``.
    -   ``X-Forwarded-Host`` sets ``HTTP_HOST``, ``SERVER_NAME``, and
        ``SERVER_PORT``.
    -   ``X-Forwarded-Port`` sets ``HTTP_HOST`` and ``SERVER_PORT``.
    -   ``X-Forwarded-Prefix`` sets ``SCRIPT_NAME``.

    You must tell the middleware how many proxies set each header so it
    knows what values to trust. It is a security issue to trust values
    that came from the client rather than a proxy.

    The original values of the headers are stored in the WSGI
    environ as ``werkzeug.proxy_fix.orig``, a dict.

    :param app: The WSGI application to wrap.
    :param x_for: Number of values to trust for ``X-Forwarded-For``.
    :param x_proto: Number of values to trust for ``X-Forwarded-Proto``.
    :param x_host: Number of values to trust for ``X-Forwarded-Host``.
    :param x_port: Number of values to trust for ``X-Forwarded-Port``.
    :param x_prefix: Number of values to trust for
        ``X-Forwarded-Prefix``.

    .. code-block:: python

        from werkzeug.middleware.proxy_fix import ProxyFix
        # App is behind one proxy that sets the -For and -Host headers.
        app = ProxyFix(app, x_for=1, x_host=1)

    .. versionchanged:: 1.0
        The ``num_proxies`` argument and attribute; the ``get_remote_addr`` method; and
        the environ keys ``orig_remote_addr``, ``orig_wsgi_url_scheme``, and
        ``orig_http_host`` were removed.

    .. versionchanged:: 0.15
        All headers support multiple values. Each header is configured with a separate
        number of trusted proxies.

    .. versionchanged:: 0.15
        Original WSGI environ values are stored in the ``werkzeug.proxy_fix.orig`` dict.

    .. versionchanged:: 0.15
        Support ``X-Forwarded-Port`` and ``X-Forwarded-Prefix``.

    .. versionchanged:: 0.15
        ``X-Forwarded-Host`` and ``X-Forwarded-Port`` modify
        ``SERVER_NAME`` and ``SERVER_PORT``.
    """

    def __init__(
        self,
        app: WSGIApplication,
        x_for: int = 1,
        x_proto: int = 1,
        x_host: int = 0,
        x_port: int = 0,
        x_prefix: int = 0,
    ) -> None:
        self.app = app
        self.x_for = x_for
        self.x_proto = x_proto
        self.x_host = x_host
        self.x_port = x_port
        self.x_prefix = x_prefix

    def _get_real_value(self, trusted: int, value: str | None) -> str | None:
        """Get the real value from a list header based on the configured
        number of trusted proxies.

        :param trusted: Number of values to trust in the header.
        :param value: Comma separated list header value to parse.
        :return: The real value, or ``None`` if there are fewer values
            than the number of trusted proxies.

        .. versionchanged:: 1.0
            Renamed from ``_get_trusted_comma``.

        .. versionadded:: 0.15
        """
        if not (trusted and value):
            return None
        values = parse_list_header(value)
        if len(values) >= trusted:
            return values[-trusted]
        return None

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Modify the WSGI environ based on the various ``Forwarded``
        headers before calling the wrapped application. Store the
        original environ values in ``werkzeug.proxy_fix.orig_{key}``.
        """
        environ_get = environ.get
        orig_remote_addr = environ_get("REMOTE_ADDR")
        orig_wsgi_url_scheme = environ_get("wsgi.url_scheme")
        orig_http_host = environ_get("HTTP_HOST")
        environ.update(
            {
                "werkzeug.proxy_fix.orig": {
                    "REMOTE_ADDR": orig_remote_addr,
                    "wsgi.url_scheme": orig_wsgi_url_scheme,
                    "HTTP_HOST": orig_http_host,
                    "SERVER_NAME": environ_get("SERVER_NAME"),
                    "SERVER_PORT": environ_get("SERVER_PORT"),
                    "SCRIPT_NAME": environ_get("SCRIPT_NAME"),
                }
            }
        )

        x_for = self._get_real_value(self.x_for, environ_get("HTTP_X_FORWARDED_FOR"))
        if x_for:
            environ["REMOTE_ADDR"] = x_for

        x_proto = self._get_real_value(
            self.x_proto, environ_get("HTTP_X_FORWARDED_PROTO")
        )
        if x_proto:
            environ["wsgi.url_scheme"] = x_proto

        x_host = self._get_real_value(self.x_host, environ_get("HTTP_X_FORWARDED_HOST"))
        if x_host:
            environ["HTTP_HOST"] = environ["SERVER_NAME"] = x_host
            # "]" to check for IPv6 address without port
            if ":" in x_host and not x_host.endswith("]"):
                environ["SERVER_NAME"], environ["SERVER_PORT"] = x_host.rsplit(":", 1)

        x_port = self._get_real_value(self.x_port, environ_get("HTTP_X_FORWARDED_PORT"))
        if x_port:
            host = environ.get("HTTP_HOST")
            if host:
                # "]" to check for IPv6 address without port
                if ":" in host and not host.endswith("]"):
                    host = host.rsplit(":", 1)[0]
                environ["HTTP_HOST"] = f"{host}:{x_port}"
            environ["SERVER_PORT"] = x_port

        x_prefix = self._get_real_value(
            self.x_prefix, environ_get("HTTP_X_FORWARDED_PREFIX")
        )
        if x_prefix:
            environ["SCRIPT_NAME"] = x_prefix

        return self.app(environ, start_response)
<|endoftext|>"""
Basic HTTP Proxy
================

.. autoclass:: ProxyMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import typing as t
from http import client
from urllib.parse import quote
from urllib.parse import urlsplit

from ..datastructures import EnvironHeaders
from ..http import is_hop_by_hop_header
from ..wsgi import get_input_stream

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class ProxyMiddleware:
    """Proxy requests under a path to an external server, routing other
    requests to the app.

    This middleware can only proxy HTTP requests, as HTTP is the only
    protocol handled by the WSGI server. Other protocols, such as
    WebSocket requests, cannot be proxied at this layer. This should
    only be used for development, in production a real proxy server
    should be used.

    The middleware takes a dict mapping a path prefix to a dict
    describing the host to be proxied to::

        app = ProxyMiddleware(app, {
            "/static/": {
                "target": "http://127.0.0.1:5001/",
            }
        })

    Each host has the following options:

    ``target``:
        The target URL to dispatch to. This is required.
    ``remove_prefix``:
        Whether to remove the prefix from the URL before dispatching it
        to the target. The default is ``False``.
    ``host``:
        ``"<auto>"`` (default):
            The host header is automatically rewritten to the URL of the
            target.
        ``None``:
            The host header is unmodified from the client request.
        Any other value:
            The host header is overwritten with the value.
    ``headers``:
        A dictionary of headers to be sent with the request to the
        target. The default is ``{}``.
    ``ssl_context``:
        A :class:`ssl.SSLContext` defining how to verify requests if the
        target is HTTPS. The default is ``None``.

    In the example above, everything under ``"/static/"`` is proxied to
    the server on port 5001. The host header is rewritten to the target,
    and the ``"/static/"`` prefix is removed from the URLs.

    :param app: The WSGI application to wrap.
    :param targets: Proxy target configurations. See description above.
    :param chunk_size: Size of chunks to read from input stream and
        write to target.
    :param timeout: Seconds before an operation to a target fails.

    .. versionadded:: 0.14
    """

    def __init__(
        self,
        app: WSGIApplication,
        targets: t.Mapping[str, dict[str, t.Any]],
        chunk_size: int = 2 << 13,
        timeout: int = 10,
    ) -> None:
        def _set_defaults(opts: dict[str, t.Any]) -> dict[str, t.Any]:
            opts.setdefault("remove_prefix", False)
            opts.setdefault("host", "<auto>")
            opts.setdefault("headers", {})
            opts.setdefault("ssl_context", None)
            return opts

        self.app = app
        self.targets = {
            f"/{k.strip('/')}/": _set_defaults(v) for k, v in targets.items()
        }
        self.chunk_size = chunk_size
        self.timeout = timeout

    def proxy_to(
        self, opts: dict[str, t.Any], path: str, prefix: str
    ) -> WSGIApplication:
        target = urlsplit(opts["target"])
        # socket can handle unicode host, but header must be ascii
        host = target.hostname.encode("idna").decode("ascii")

        def application(
            environ: WSGIEnvironment, start_response: StartResponse
        ) -> t.Iterable[bytes]:
            headers = list(EnvironHeaders(environ).items())
            headers[:] = [
                (k, v)
                for k, v in headers
                if not is_hop_by_hop_header(k)
                and k.lower() not in ("content-length", "host")
            ]
            headers.append(("Connection", "close"))

            if opts["host"] == "<auto>":
                headers.append(("Host", host))
            elif opts["host"] is None:
                headers.append(("Host", environ["HTTP_HOST"]))
            else:
                headers.append(("Host", opts["host"]))

            headers.extend(opts["headers"].items())
            remote_path = path

            if opts["remove_prefix"]:
                remote_path = remote_path[len(prefix) :].lstrip("/")
                remote_path = f"{target.path.rstrip('/')}/{remote_path}"

            content_length = environ.get("CONTENT_LENGTH")
            chunked = False

            if content_length not in ("", None):
                headers.append(("Content-Length", content_length))
            elif content_length is not None:
                headers.append(("Transfer-Encoding", "chunked"))
                chunked = True

            try:
                if target.scheme == "http":
                    con = client.HTTPConnection(
                        host, target.port or 80, timeout=self.timeout
                    )
                elif target.scheme == "https":
                    con = client.HTTPSConnection(
                        host,
                        target.port or 443,
                        timeout=self.timeout,
                        context=opts["ssl_context"],
                    )
                else:
                    raise RuntimeError(
                        "Target scheme must be 'http' or 'https', got"
                        f" {target.scheme!r}."
                    )

                con.connect()
                # safe = https://url.spec.whatwg.org/#url-path-segment-string
                # as well as percent for things that are already quoted
                remote_url = quote(remote_path, safe="!$&'()*+,/:;=@%")
                querystring = environ["QUERY_STRING"]

                if querystring:
                    remote_url = f"{remote_url}?{querystring}"

                con.putrequest(environ["REQUEST_METHOD"], remote_url, skip_host=True)

                for k, v in headers:
                    if k.lower() == "connection":
                        v = "close"

                    con.putheader(k, v)

                con.endheaders()
                stream = get_input_stream(environ)

                while True:
                    data = stream.read(self.chunk_size)

                    if not data:
                        break

                    if chunked:
                        con.send(b"%x\r\n%s\r\n" % (len(data), data))
                    else:
                        con.send(data)

                resp = con.getresponse()
            except OSError:
                from ..exceptions import BadGateway

                return BadGateway()(environ, start_response)

            start_response(
                f"{resp.status} {resp.reason}",
                [
                    (k.title(), v)
                    for k, v in resp.getheaders()
                    if not is_hop_by_hop_header(k)
                ],
            )

            def read() -> t.Iterator[bytes]:
                while True:
                    try:
                        data = resp.read(self.chunk_size)
                    except OSError:
                        break

                    if not data:
                        break

                    yield data

            return read()

        return application

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        path = environ["PATH_INFO"]
        app = self.app

        for prefix, opts in self.targets.items():
            if path.startswith(prefix):
                app = self.proxy_to(opts, path, prefix)
                break

        return app(environ, start_response)
<|endoftext|>"""
WSGI Protocol Linter
====================

This module provides a middleware that performs sanity checks on the
behavior of the WSGI server and application. It checks that the
:pep:`3333` WSGI spec is properly implemented. It also warns on some
common HTTP errors such as non-empty responses for 304 status codes.

.. autoclass:: LintMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn

from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class WSGIWarning(Warning):
    """Warning class for WSGI warnings."""


class HTTPWarning(Warning):
    """Warning class for HTTP warnings."""


def check_type(context: str, obj: object, need: type = str) -> None:
    if type(obj) is not need:
        warn(
            f"{context!r} requires {need.__name__!r}, got {type(obj).__name__!r}.",
            WSGIWarning,
            stacklevel=3,
        )


class InputStream:
    def __init__(self, stream: t.IO[bytes]) -> None:
        self._stream = stream

    def read(self, *args: t.Any) -> bytes:
        if len(args) == 0:
            warn(
                "WSGI does not guarantee an EOF marker on the input stream, thus making"
                " calls to 'wsgi.input.read()' unsafe. Conforming servers may never"
                " return from this call.",
                WSGIWarning,
                stacklevel=2,
            )
        elif len(args) != 1:
            warn(
                "Too many parameters passed to 'wsgi.input.read()'.",
                WSGIWarning,
                stacklevel=2,
            )
        return self._stream.read(*args)

    def readline(self, *args: t.Any) -> bytes:
        if len(args) == 0:
            warn(
                "Calls to 'wsgi.input.readline()' without arguments are unsafe. Use"
                " 'wsgi.input.read()' instead.",
                WSGIWarning,
                stacklevel=2,
            )
        elif len(args) == 1:
            warn(
                "'wsgi.input.readline()' was called with a size hint. WSGI does not"
                " support this, although it's available on all major servers.",
                WSGIWarning,
                stacklevel=2,
            )
        else:
            raise TypeError("Too many arguments passed to 'wsgi.input.readline()'.")
        return self._stream.readline(*args)

    def __iter__(self) -> t.Iterator[bytes]:
        try:
            return iter(self._stream)
        except TypeError:
            warn("'wsgi.input' is not iterable.", WSGIWarning, stacklevel=2)
            return iter(())

    def close(self) -> None:
        warn("The application closed the input stream!", WSGIWarning, stacklevel=2)
        self._stream.close()


class ErrorStream:
    def __init__(self, stream: t.IO[str]) -> None:
        self._stream = stream

    def write(self, s: str) -> None:
        check_type("wsgi.error.write()", s, str)
        self._stream.write(s)

    def flush(self) -> None:
        self._stream.flush()

    def writelines(self, seq: t.Iterable[str]) -> None:
        for line in seq:
            self.write(line)

    def close(self) -> None:
        warn("The application closed the error stream!", WSGIWarning, stacklevel=2)
        self._stream.close()


class GuardedWrite:
    def __init__(self, write: t.Callable[[bytes], object], chunks: list[int]) -> None:
        self._write = write
        self._chunks = chunks

    def __call__(self, s: bytes) -> None:
        check_type("write()", s, bytes)
        self._write(s)
        self._chunks.append(len(s))


class GuardedIterator:
    def __init__(
        self,
        iterator: t.Iterable[bytes],
        headers_set: tuple[int, Headers],
        chunks: list[int],
    ) -> None:
        self._iterator = iterator
        self._next = iter(iterator).__next__
        self.closed = False
        self.headers_set = headers_set
        self.chunks = chunks

    def __iter__(self) -> GuardedIterator:
        return self

    def __next__(self) -> bytes:
        if self.closed:
            warn("Iterated over closed 'app_iter'.", WSGIWarning, stacklevel=2)

        rv = self._next()

        if not self.headers_set:
            warn(
                "The application returned before it started the response.",
                WSGIWarning,
                stacklevel=2,
            )

        check_type("application iterator items", rv, bytes)
        self.chunks.append(len(rv))
        return rv

    def close(self) -> None:
        self.closed = True

        if hasattr(self._iterator, "close"):
            self._iterator.close()

        if self.headers_set:
            status_code, headers = self.headers_set
            bytes_sent = sum(self.chunks)
            content_length = headers.get("content-length", type=int)

            if status_code == 304:
                for key, _value in headers:
                    key = key.lower()
                    if key not in ("expires", "content-location") and is_entity_header(
                        key
                    ):
                        warn(
                            f"Entity header {key!r} found in 304 response.",
                            HTTPWarning,
                            stacklevel=2,
                        )
                if bytes_sent:
                    warn(
                        "304 responses must not have a body.",
                        HTTPWarning,
                        stacklevel=2,
                    )
            elif 100 <= status_code < 200 or status_code == 204:
                if content_length != 0:
                    warn(
                        f"{status_code} responses must have an empty content length.",
                        HTTPWarning,
                        stacklevel=2,
                    )
                if bytes_sent:
                    warn(
                        f"{status_code} responses must not have a body.",
                        HTTPWarning,
                        stacklevel=2,
                    )
            elif content_length is not None and content_length != bytes_sent:
                warn(
                    "Content-Length and the number of bytes sent to the"
                    " client do not match.",
                    WSGIWarning,
                    stacklevel=2,
                )

    def __del__(self) -> None:
        if not self.closed:
            try:
                warn(
                    "Iterator was garbage collected before it was closed.",
                    WSGIWarning,
                    stacklevel=2,
                )
            except Exception:
                pass


class LintMiddleware:
    """Warns about common errors in the WSGI and HTTP behavior of the
    server and wrapped application. Some of the issues it checks are:

    -   invalid status codes
    -   non-bytes sent to the WSGI server
    -   strings returned from the WSGI application
    -   non-empty conditional responses
    -   unquoted etags
    -   relative URLs in the Location header
    -   unsafe calls to wsgi.input
    -   unclosed iterators

    Error information is emitted using the :mod:`warnings` module.

    :param app: The WSGI application to wrap.

    .. code-block:: python

        from werkzeug.middleware.lint import LintMiddleware
        app = LintMiddleware(app)
    """

    def __init__(self, app: WSGIApplication) -> None:
        self.app = app

    def check_environ(self, environ: WSGIEnvironment) -> None:
        if type(environ) is not dict:  # noqa: E721
            warn(
                "WSGI environment is not a standard Python dict.",
                WSGIWarning,
                stacklevel=4,
            )
        for key in (
            "REQUEST_METHOD",
            "SERVER_NAME",
            "SERVER_PORT",
            "wsgi.version",
            "wsgi.input",
            "wsgi.errors",
            "wsgi.multithread",
            "wsgi.multiprocess",
            "wsgi.run_once",
        ):
            if key not in environ:
                warn(
                    f"Required environment key {key!r} not found",
                    WSGIWarning,
                    stacklevel=3,
                )
        if environ["wsgi.version"] != (1, 0):
            warn("Environ is not a WSGI 1.0 environ.", WSGIWarning, stacklevel=3)

        script_name = environ.get("SCRIPT_NAME", "")
        path_info = environ.get("PATH_INFO", "")

        if script_name and script_name[0] != "/":
            warn(
                f"'SCRIPT_NAME' does not start with a slash: {script_name!r}",
                WSGIWarning,
                stacklevel=3,
            )

        if path_info and path_info[0] != "/":
            warn(
                f"'PATH_INFO' does not start with a slash: {path_info!r}",
                WSGIWarning,
                stacklevel=3,
            )

    def check_start_response(
        self,
        status: str,
        headers: list[tuple[str, str]],
        exc_info: None | (tuple[type[BaseException], BaseException, TracebackType]),
    ) -> tuple[int, Headers]:
        check_type("status", status, str)
        status_code_str = status.split(None, 1)[0]

        if len(status_code_str) != 3 or not status_code_str.isdecimal():
            warn("Status code must be three digits.", WSGIWarning, stacklevel=3)

        if len(status) < 4 or status[3] != " ":
            warn(
                f"Invalid value for status {status!r}. Valid status strings are three"
                " digits, a space and a status explanation.",
                WSGIWarning,
                stacklevel=3,
            )

        status_code = int(status_code_str)

        if status_code < 100:
            warn("Status code < 100 detected.", WSGIWarning, stacklevel=3)

        if type(headers) is not list:  # noqa: E721
            warn("Header list is not a list.", WSGIWarning, stacklevel=3)

        for item in headers:
            if type(item) is not tuple or len(item) != 2:
                warn("Header items must be 2-item tuples.", WSGIWarning, stacklevel=3)
            name, value = item
            if type(name) is not str or type(value) is not str:  # noqa: E721
                warn(
                    "Header keys and values must be strings.", WSGIWarning, stacklevel=3
                )
            if name.lower() == "status":
                warn(
                    "The status header is not supported due to"
                    " conflicts with the CGI spec.",
                    WSGIWarning,
                    stacklevel=3,
                )

        if exc_info is not None and not isinstance(exc_info, tuple):
            warn("Invalid value for exc_info.", WSGIWarning, stacklevel=3)

        headers_obj = Headers(headers)
        self.check_headers(headers_obj)

        return status_code, headers_obj

    def check_headers(self, headers: Headers) -> None:
        etag = headers.get("etag")

        if etag is not None:
            if etag.startswith(("W/", "w/")):
                if etag.startswith("w/"):
                    warn(
                        "Weak etag indicator should be upper case.",
                        HTTPWarning,
                        stacklevel=4,
                    )

                etag = etag[2:]

            if not (etag[:1] == etag[-1:] == '"'):
                warn("Unquoted etag emitted.", HTTPWarning, stacklevel=4)

        location = headers.get("location")

        if location is not None:
            if not urlparse(location).netloc:
                warn(
                    "Absolute URLs required for location header.",
                    HTTPWarning,
                    stacklevel=4,
                )

    def check_iterator(self, app_iter: t.Iterable[bytes]) -> None:
        if isinstance(app_iter, str):
            warn(
                "The application returned a string. The response will send one"
                " character at a time to the client, which will kill performance."
                " Return a list or iterable instead.",
                WSGIWarning,
                stacklevel=3,
            )

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Iterable[bytes]:
        if len(args) != 2:
            warn("A WSGI app takes two arguments.", WSGIWarning, stacklevel=2)

        if kwargs:
            warn(
                "A WSGI app does not take keyword arguments.", WSGIWarning, stacklevel=2
            )

        environ: WSGIEnvironment = args[0]
        start_response: StartResponse = args[1]

        self.check_environ(environ)
        environ["wsgi.input"] = InputStream(environ["wsgi.input"])
        environ["wsgi.errors"] = ErrorStream(environ["wsgi.errors"])

        # Hook our own file wrapper in so that applications will always
        # iterate to the end and we can check the content length.
        environ["wsgi.file_wrapper"] = FileWrapper

        headers_set: list[t.Any] = []
        chunks: list[int] = []

        def checking_start_response(
            *args: t.Any, **kwargs: t.Any
        ) -> t.Callable[[bytes], None]:
            if len(args) not in {2, 3}:
                warn(
                    f"Invalid number of arguments: {len(args)}, expected 2 or 3.",
                    WSGIWarning,
                    stacklevel=2,
                )

            if kwargs:
                warn(
                    "'start_response' does not take keyword arguments.",
                    WSGIWarning,
                    stacklevel=2,
                )

            status: str = args[0]
            headers: list[tuple[str, str]] = args[1]
            exc_info: (
                None | (tuple[type[BaseException], BaseException, TracebackType])
            ) = args[2] if len(args) == 3 else None

            headers_set[:] = self.check_start_response(status, headers, exc_info)
            return GuardedWrite(start_response(status, headers, exc_info), chunks)

        app_iter = self.app(environ, t.cast("StartResponse", checking_start_response))
        self.check_iterator(app_iter)
        return GuardedIterator(
            app_iter, t.cast(tuple[int, Headers], headers_set), chunks
        )
<|endoftext|>"""
Application Dispatcher
======================

This middleware creates a single WSGI application that dispatches to
multiple other WSGI applications mounted at different URL paths.

A common example is writing a Single Page Application, where you have a
backend API and a frontend written in JavaScript that does the routing
in the browser rather than requesting different pages from the server.
The frontend is a single HTML and JS file that should be served for any
path besides "/api".

This example dispatches to an API app under "/api", an admin app
under "/admin", and an app that serves frontend files for all other
requests::

    app = DispatcherMiddleware(serve_frontend, {
        '/api': api_app,
        '/admin': admin_app,
    })

In production, you might instead handle this at the HTTP server level,
serving files or proxying to application servers based on location. The
API and admin apps would each be deployed with a separate WSGI server,
and the static files would be served directly by the HTTP server.

.. autoclass:: DispatcherMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class DispatcherMiddleware:
    """Combine multiple applications as a single WSGI application.
    Requests are dispatched to an application based on the path it is
    mounted under.

    :param app: The WSGI application to dispatch to if the request
        doesn't match a mounted path.
    :param mounts: Maps path prefixes to applications for dispatching.
    """

    def __init__(
        self,
        app: WSGIApplication,
        mounts: dict[str, WSGIApplication] | None = None,
    ) -> None:
        self.app = app
        self.mounts = mounts or {}

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        script = environ.get("PATH_INFO", "")
        path_info = ""

        while "/" in script:
            if script in self.mounts:
                app = self.mounts[script]
                break

            script, last_item = script.rsplit("/", 1)
            path_info = f"/{last_item}{path_info}"
        else:
            app = self.mounts.get(script, self.app)

        original_script_name = environ.get("SCRIPT_NAME", "")
        environ["SCRIPT_NAME"] = original_script_name + script
        environ["PATH_INFO"] = path_info
        return app(environ, start_response)
<|endoftext|>"""
Serve Shared Static Files
=========================

.. autoclass:: SharedDataMiddleware
    :members: is_allowed

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import collections.abc as cabc
import importlib.util
import mimetypes
import os
import posixpath
import typing as t
from datetime import datetime
from datetime import timezone
from io import BytesIO
from time import time
from zlib import adler32

from ..http import http_date
from ..http import is_resource_modified
from ..http import quote_etag
from ..security import safe_join
from ..utils import get_content_type
from ..wsgi import get_path_info
from ..wsgi import wrap_file

_TOpener = t.Callable[[], tuple[t.IO[bytes], datetime, int]]
_TLoader = t.Callable[[t.Optional[str]], tuple[t.Optional[str], t.Optional[_TOpener]]]

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class SharedDataMiddleware:
    """A WSGI middleware which provides static content for development
    environments or simple server setups. Its usage is quite simple::

        import os
        from werkzeug.middleware.shared_data import SharedDataMiddleware

        app = SharedDataMiddleware(app, {
            '/shared': os.path.join(os.path.dirname(__file__), 'shared')
        })

    The contents of the folder ``./shared`` will now be available on
    ``http://example.com/shared/``.  This is pretty useful during development
    because a standalone media server is not required. Files can also be
    mounted on the root folder and still continue to use the application because
    the shared data middleware forwards all unhandled requests to the
    application, even if the requests are below one of the shared folders.

    If `pkg_resources` is available you can also tell the middleware to serve
    files from package data::

        app = SharedDataMiddleware(app, {
            '/static': ('myapplication', 'static')
        })

    This will then serve the ``static`` folder in the `myapplication`
    Python package.

    The optional `disallow` parameter can be a list of :func:`~fnmatch.fnmatch`
    rules for files that are not accessible from the web.  If `cache` is set to
    `False` no caching headers are sent.

    Currently the middleware does not support non-ASCII filenames. If the
    encoding on the file system happens to match the encoding of the URI it may
    work but this could also be by accident. We strongly suggest using ASCII
    only file names for static files.

    The middleware will guess the mimetype using the Python `mimetype`
    module.  If it's unable to figure out the charset it will fall back
    to `fallback_mimetype`.

    :param app: the application to wrap.  If you don't want to wrap an
                application you can pass it :exc:`NotFound`.
    :param exports: a list or dict of exported files and folders.
    :param disallow: a list of :func:`~fnmatch.fnmatch` rules.
    :param cache: enable or disable caching headers.
    :param cache_timeout: the cache timeout in seconds for the headers.
    :param fallback_mimetype: The fallback mimetype for unknown files.

    .. versionchanged:: 1.0
        The default ``fallback_mimetype`` is
        ``application/octet-stream``. If a filename looks like a text
        mimetype, the ``utf-8`` charset is added to it.

    .. versionadded:: 0.6
        Added ``fallback_mimetype``.

    .. versionchanged:: 0.5
        Added ``cache_timeout``.
    """

    def __init__(
        self,
        app: WSGIApplication,
        exports: (
            cabc.Mapping[str, str | tuple[str, str]]
            | t.Iterable[tuple[str, str | tuple[str, str]]]
        ),
        disallow: None = None,
        cache: bool = True,
        cache_timeout: int = 60 * 60 * 12,
        fallback_mimetype: str = "application/octet-stream",
    ) -> None:
        self.app = app
        self.exports: list[tuple[str, _TLoader]] = []
        self.cache = cache
        self.cache_timeout = cache_timeout

        if isinstance(exports, cabc.Mapping):
            exports = exports.items()

        for key, value in exports:
            if isinstance(value, tuple):
                loader = self.get_package_loader(*value)
            elif isinstance(value, str):
                if os.path.isfile(value):
                    loader = self.get_file_loader(value)
                else:
                    loader = self.get_directory_loader(value)
            else:
                raise TypeError(f"unknown def {value!r}")

            self.exports.append((key, loader))

        if disallow is not None:
            from fnmatch import fnmatch

            self.is_allowed = lambda x: not fnmatch(x, disallow)

        self.fallback_mimetype = fallback_mimetype

    def is_allowed(self, filename: str) -> bool:
        """Subclasses can override this method to disallow the access to
        certain files.  However by providing `disallow` in the constructor
        this method is overwritten.
        """
        return True

    def _opener(self, filename: str) -> _TOpener:
        return lambda: (
            open(filename, "rb"),
            datetime.fromtimestamp(os.path.getmtime(filename), tz=timezone.utc),
            int(os.path.getsize(filename)),
        )

    def get_file_loader(self, filename: str) -> _TLoader:
        return lambda x: (os.path.basename(filename), self._opener(filename))

    def get_package_loader(self, package: str, package_path: str) -> _TLoader:
        load_time = datetime.now(timezone.utc)
        spec = importlib.util.find_spec(package)
        reader = spec.loader.get_resource_reader(package)  # type: ignore[union-attr]

        def loader(
            path: str | None,
        ) -> tuple[str | None, _TOpener | None]:
            if path is None:
                return None, None

            path = safe_join(package_path, path)

            if path is None:
                return None, None

            basename = posixpath.basename(path)

            try:
                resource = reader.open_resource(path)
            except OSError:
                return None, None

            if isinstance(resource, BytesIO):
                return (
                    basename,
                    lambda: (resource, load_time, len(resource.getvalue())),
                )

            return (
                basename,
                lambda: (
                    resource,
                    datetime.fromtimestamp(
                        os.path.getmtime(resource.name), tz=timezone.utc
                    ),
                    os.path.getsize(resource.name),
                ),
            )

        return loader

    def get_directory_loader(self, directory: str) -> _TLoader:
        def loader(
            path: str | None,
        ) -> tuple[str | None, _TOpener | None]:
            if path is not None:
                path = safe_join(directory, path)

                if path is None:
                    return None, None
            else:
                path = directory

            if os.path.isfile(path):
                return os.path.basename(path), self._opener(path)

            return None, None

        return loader

    def generate_etag(self, mtime: datetime, file_size: int, real_filename: str) -> str:
        fn_str = os.fsencode(real_filename)
        timestamp = mtime.timestamp()
        checksum = adler32(fn_str) & 0xFFFFFFFF
        return f"wzsdm-{timestamp}-{file_size}-{checksum}"

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        path = get_path_info(environ)
        file_loader = None

        for search_path, loader in self.exports:
            if search_path == path:
                real_filename, file_loader = loader(None)

                if file_loader is not None:
                    break

            if not search_path.endswith("/"):
                search_path += "/"

            if path.startswith(search_path):
                real_filename, file_loader = loader(path[len(search_path) :])

                if file_loader is not None:
                    break

        if file_loader is None or not self.is_allowed(real_filename):  # type: ignore
            return self.app(environ, start_response)

        guessed_type = mimetypes.guess_type(real_filename)  # type: ignore
        mime_type = get_content_type(guessed_type[0] or self.fallback_mimetype, "utf-8")
        f, mtime, file_size = file_loader()

        headers = [("Date", http_date())]

        if self.cache:
            timeout = self.cache_timeout
            etag = quote_etag(self.generate_etag(mtime, file_size, real_filename))  # type: ignore
            headers += [
                ("Etag", etag),
                ("Cache-Control", f"max-age={timeout}, public"),
            ]

            if not is_resource_modified(environ, etag, last_modified=mtime):
                f.close()
                start_response("304 Not Modified", headers)
                return []

            headers.append(("Expires", http_date(time() + timeout)))
        else:
            headers.append(("Cache-Control", "public"))

        headers.extend(
            (
                ("Content-Type", mime_type),
                ("Content-Length", str(file_size)),
                ("Last-Modified", http_date(mtime)),
            )
        )
        start_response("200 OK", headers)
        return wrap_file(environ, f)
<|endoftext|>"""Object representations for debugging purposes. Unlike the default
repr, these expose more information and produce HTML instead of ASCII.

Together with the CSS and JavaScript of the debugger this gives a
colorful and more compact output.
"""

from __future__ import annotations

import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only

from markupsafe import escape

missing = object()
_paragraph_re = re.compile(r"(?:\r\n|\r|\n){2,}")
RegexType = type(_paragraph_re)

HELP_HTML = """\
<div class=box>
  <h3>%(title)s</h3>
  <pre class=help>%(text)s</pre>
</div>\
"""
OBJECT_DUMP_HTML = """\
<div class=box>
  <h3>%(title)s</h3>
  %(repr)s
  <table>%(items)s</table>
</div>\
"""


def debug_repr(obj: object) -> str:
    """Creates a debug repr of an object as HTML string."""
    return DebugReprGenerator().repr(obj)


def dump(obj: object = missing) -> None:
    """Print the object details to stdout._write (for the interactive
    console of the web debugger.
    """
    gen = DebugReprGenerator()
    if obj is missing:
        rv = gen.dump_locals(sys._getframe(1).f_locals)
    else:
        rv = gen.dump_object(obj)
    sys.stdout._write(rv)  # type: ignore


class _Helper:
    """Displays an HTML version of the normal help, for the interactive
    debugger only because it requires a patched sys.stdout.
    """

    def __repr__(self) -> str:
        return "Type help(object) for help about object."

    def __call__(self, topic: t.Any | None = None) -> None:
        if topic is None:
            sys.stdout._write(f"<span class=help>{self!r}</span>")  # type: ignore
            return
        import pydoc

        pydoc.help(topic)
        rv = sys.stdout.reset()  # type: ignore
        paragraphs = _paragraph_re.split(rv)
        if len(paragraphs) > 1:
            title = paragraphs[0]
            text = "\n\n".join(paragraphs[1:])
        else:
            title = "Help"
            text = paragraphs[0]
        sys.stdout._write(HELP_HTML % {"title": title, "text": text})  # type: ignore


helper = _Helper()


def _add_subclass_info(inner: str, obj: object, base: type | tuple[type, ...]) -> str:
    if isinstance(base, tuple):
        for cls in base:
            if type(obj) is cls:
                return inner
    elif type(obj) is base:
        return inner
    module = ""
    if obj.__class__.__module__ not in ("__builtin__", "exceptions"):
        module = f'<span class="module">{obj.__class__.__module__}.</span>'
    return f"{module}{type(obj).__name__}({inner})"


def _sequence_repr_maker(
    left: str, right: str, base: type, limit: int = 8
) -> t.Callable[[DebugReprGenerator, t.Iterable[t.Any], bool], str]:
    def proxy(self: DebugReprGenerator, obj: t.Iterable[t.Any], recursive: bool) -> str:
        if recursive:
            return _add_subclass_info(f"{left}...{right}", obj, base)
        buf = [left]
        have_extended_section = False
        for idx, item in enumerate(obj):
            if idx:
                buf.append(", ")
            if idx == limit:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(self.repr(item))
        if have_extended_section:
            buf.append("</span>")
        buf.append(right)
        return _add_subclass_info("".join(buf), obj, base)

    return proxy


class DebugReprGenerator:
    def __init__(self) -> None:
        self._stack: list[t.Any] = []

    list_repr = _sequence_repr_maker("[", "]", list)
    tuple_repr = _sequence_repr_maker("(", ")", tuple)
    set_repr = _sequence_repr_maker("set([", "])", set)
    frozenset_repr = _sequence_repr_maker("frozenset([", "])", frozenset)
    deque_repr = _sequence_repr_maker(
        '<span class="module">collections.</span>deque([', "])", deque
    )

    def regex_repr(self, obj: t.Pattern[t.AnyStr]) -> str:
        pattern = repr(obj.pattern)
        pattern = codecs.decode(pattern, "unicode-escape", "ignore")
        pattern = f"r{pattern}"
        return f're.compile(<span class="string regex">{pattern}</span>)'

    def string_repr(self, obj: str | bytes, limit: int = 70) -> str:
        buf = ['<span class="string">']
        r = repr(obj)

        # shorten the repr when the hidden part would be at least 3 chars
        if len(r) - limit > 2:
            buf.extend(
                (
                    escape(r[:limit]),
                    '<span class="extended">',
                    escape(r[limit:]),
                    "</span>",
                )
            )
        else:
            buf.append(escape(r))

        buf.append("</span>")
        out = "".join(buf)

        # if the repr looks like a standard string, add subclass info if needed
        if r.startswith(("'", '"', 'b"', "b'")):
            return _add_subclass_info(out, obj, (bytes, str))

        # otherwise, assume the repr distinguishes the subclass already
        return out

    def dict_repr(
        self,
        d: dict[int, None] | dict[str, int] | dict[str | int, int],
        recursive: bool,
        limit: int = 5,
    ) -> str:
        if recursive:
            return _add_subclass_info("{...}", d, dict)
        buf = ["{"]
        have_extended_section = False
        for idx, (key, value) in enumerate(d.items()):
            if idx:
                buf.append(", ")
            if idx == limit - 1:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(
                f'<span class="pair"><span class="key">{self.repr(key)}</span>:'
                f' <span class="value">{self.repr(value)}</span></span>'
            )
        if have_extended_section:
            buf.append("</span>")
        buf.append("}")
        return _add_subclass_info("".join(buf), d, dict)

    def object_repr(self, obj: t.Any) -> str:
        r = repr(obj)
        return f'<span class="object">{escape(r)}</span>'

    def dispatch_repr(self, obj: t.Any, recursive: bool) -> str:
        if obj is helper:
            return f'<span class="help">{helper!r}</span>'
        if isinstance(obj, (int, float, complex)):
            return f'<span class="number">{obj!r}</span>'
        if isinstance(obj, str) or isinstance(obj, bytes):
            return self.string_repr(obj)
        if isinstance(obj, RegexType):
            return self.regex_repr(obj)
        if isinstance(obj, list):
            return self.list_repr(obj, recursive)
        if isinstance(obj, tuple):
            return self.tuple_repr(obj, recursive)
        if isinstance(obj, set):
            return self.set_repr(obj, recursive)
        if isinstance(obj, frozenset):
            return self.frozenset_repr(obj, recursive)
        if isinstance(obj, dict):
            return self.dict_repr(obj, recursive)
        if isinstance(obj, deque):
            return self.deque_repr(obj, recursive)
        return self.object_repr(obj)

    def fallback_repr(self) -> str:
        try:
            info = "".join(format_exception_only(*sys.exc_info()[:2]))
        except Exception:
            info = "?"
        return (
            '<span class="brokenrepr">'
            f"&lt;broken repr ({escape(info.strip())})&gt;</span>"
        )

    def repr(self, obj: object) -> str:
        recursive = False
        for item in self._stack:
            if item is obj:
                recursive = True
                break
        self._stack.append(obj)
        try:
            try:
                return self.dispatch_repr(obj, recursive)
            except Exception:
                return self.fallback_repr()
        finally:
            self._stack.pop()

    def dump_object(self, obj: object) -> str:
        repr = None
        items: list[tuple[str, str]] | None = None

        if isinstance(obj, dict):
            title = "Contents of"
            items = []
            for key, value in obj.items():
                if not isinstance(key, str):
                    items = None
                    break
                items.append((key, self.repr(value)))
        if items is None:
            items = []
            repr = self.repr(obj)
            for key in dir(obj):
                try:
                    items.append((key, self.repr(getattr(obj, key))))
                except Exception:
                    pass
            title = "Details for"
        title += f" {object.__repr__(obj)[1:-1]}"
        return self.render_object_dump(items, title, repr)

    def dump_locals(self, d: dict[str, t.Any]) -> str:
        items = [(key, self.repr(value)) for key, value in d.items()]
        return self.render_object_dump(items, "Local variables in frame")

    def render_object_dump(
        self, items: list[tuple[str, str]], title: str, repr: str | None = None
    ) -> str:
        html_items = []
        for key, value in items:
            html_items.append(f"<tr><th>{escape(key)}<td><pre class=repr>{value}</pre>")
        if not html_items:
            html_items.append("<tr><td><em>Nothing</em>")
        return OBJECT_DUMP_HTML % {
            "title": escape(title),
            "repr": f"<pre class=repr>{repr if repr else ''}</pre>",
            "items": "\n".join(html_items),
        }
<|endoftext|>from __future__ import annotations

import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType

from markupsafe import escape

from .repr import debug_repr
from .repr import dump
from .repr import helper

_stream: ContextVar[HTMLStringO] = ContextVar("werkzeug.debug.console.stream")
_ipy: ContextVar[_InteractiveConsole] = ContextVar("werkzeug.debug.console.ipy")


class HTMLStringO:
    """A StringO version that HTML escapes on write."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def seek(self, n: int, mode: int = 0) -> None:
        pass

    def readline(self) -> str:
        if len(self._buffer) == 0:
            return ""
        ret = self._buffer[0]
        del self._buffer[0]
        return ret

    def reset(self) -> str:
        val = "".join(self._buffer)
        del self._buffer[:]
        return val

    def _write(self, x: str) -> None:
        self._buffer.append(x)

    def write(self, x: str) -> None:
        self._write(escape(x))

    def writelines(self, x: t.Iterable[str]) -> None:
        self._write(escape("".join(x)))


class ThreadedStream:
    """Thread-local wrapper for sys.stdout for the interactive console."""

    @staticmethod
    def push() -> None:
        if not isinstance(sys.stdout, ThreadedStream):
            sys.stdout = t.cast(t.TextIO, ThreadedStream())

        _stream.set(HTMLStringO())

    @staticmethod
    def fetch() -> str:
        try:
            stream = _stream.get()
        except LookupError:
            return ""

        return stream.reset()

    @staticmethod
    def displayhook(obj: object) -> None:
        try:
            stream = _stream.get()
        except LookupError:
            return _displayhook(obj)  # type: ignore

        # stream._write bypasses escaping as debug_repr is
        # already generating HTML for us.
        if obj is not None:
            _ipy.get().locals["_"] = obj
            stream._write(debug_repr(obj))

    def __setattr__(self, name: str, value: t.Any) -> None:
        raise AttributeError(f"read only attribute {name}")

    def __dir__(self) -> list[str]:
        return dir(sys.__stdout__)

    def __getattribute__(self, name: str) -> t.Any:
        try:
            stream = _stream.get()
        except LookupError:
            stream = sys.__stdout__  # type: ignore[assignment]

        return getattr(stream, name)

    def __repr__(self) -> str:
        return repr(sys.__stdout__)


# add the threaded stream as display hook
_displayhook = sys.displayhook
sys.displayhook = ThreadedStream.displayhook


class _ConsoleLoader:
    def __init__(self) -> None:
        self._storage: dict[int, str] = {}

    def register(self, code: CodeType, source: str) -> None:
        self._storage[id(code)] = source
        # register code objects of wrapped functions too.
        for var in code.co_consts:
            if isinstance(var, CodeType):
                self._storage[id(var)] = source

    def get_source_by_code(self, code: CodeType) -> str | None:
        try:
            return self._storage[id(code)]
        except KeyError:
            return None


class _InteractiveConsole(code.InteractiveInterpreter):
    locals: dict[str, t.Any]

    def __init__(self, globals: dict[str, t.Any], locals: dict[str, t.Any]) -> None:
        self.loader = _ConsoleLoader()
        locals = {
            **globals,
            **locals,
            "dump": dump,
            "help": helper,
            "__loader__": self.loader,
        }
        super().__init__(locals)
        original_compile = self.compile

        def compile(source: str, filename: str, symbol: str) -> CodeType | None:
            code = original_compile(source, filename, symbol)

            if code is not None:
                self.loader.register(code, source)

            return code

        self.compile = compile  # type: ignore[assignment]
        self.more = False
        self.buffer: list[str] = []

    def runsource(self, source: str, **kwargs: t.Any) -> str:  # type: ignore
        source = f"{source.rstrip()}\n"
        ThreadedStream.push()
        prompt = "... " if self.more else ">>> "
        try:
            source_to_eval = "".join(self.buffer + [source])
            if super().runsource(source_to_eval, "<debugger>", "single"):
                self.more = True
                self.buffer.append(source)
            else:
                self.more = False
                del self.buffer[:]
        finally:
            output = ThreadedStream.fetch()
        return f"{prompt}{escape(source)}{output}"

    def runcode(self, code: CodeType) -> None:
        try:
            exec(code, self.locals)
        except Exception:
            self.showtraceback()

    def showtraceback(self) -> None:
        from .tbtools import DebugTraceback

        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=1)
        sys.stdout._write(te.render_traceback_html())  # type: ignore

    def showsyntaxerror(self, filename: str | None = None) -> None:
        from .tbtools import DebugTraceback

        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=4)
        sys.stdout._write(te.render_traceback_html())  # type: ignore

    def write(self, data: str) -> None:
        sys.stdout.write(data)


class Console:
    """An interactive console."""

    def __init__(
        self,
        globals: dict[str, t.Any] | None = None,
        locals: dict[str, t.Any] | None = None,
    ) -> None:
        if locals is None:
            locals = {}
        if globals is None:
            globals = {}
        self._ipy = _InteractiveConsole(globals, locals)

    def eval(self, code: str) -> str:
        _ipy.set(self._ipy)
        old_sys_stdout = sys.stdout
        try:
            return self._ipy.runsource(code)
        finally:
            sys.stdout = old_sys_stdout
<|endoftext|>from __future__ import annotations

import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from multiprocessing import Value
from os.path import basename
from os.path import join
from zlib import adler32

from .._internal import _log
from ..exceptions import NotFound
from ..exceptions import SecurityError
from ..http import parse_cookie
from ..sansio.utils import host_is_trusted
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

# A week
PIN_TIME = 60 * 60 * 24 * 7


def hash_pin(pin: str) -> str:
    return hashlib.sha1(f"{pin} added salt".encode("utf-8", "replace")).hexdigest()[:12]


_machine_id: str | bytes | None = None


def get_machine_id() -> str | bytes | None:
    global _machine_id

    if _machine_id is not None:
        return _machine_id

    def _generate() -> str | bytes | None:
        linux = b""

        # machine-id is stable across boots, boot_id is not.
        for filename in "/etc/machine-id", "/proc/sys/kernel/random/boot_id":
            try:
                with open(filename, "rb") as f:
                    value = f.readline().strip()
            except OSError:
                continue

            if value:
                linux += value
                break

        # Containers share the same machine id, add some cgroup
        # information. This is used outside containers too but should be
        # relatively stable across boots.
        try:
            with open("/proc/self/cgroup", "rb") as f:
                linux += f.readline().strip()
        except OSError:
            pass

        if linux:
            return linux

        # On OS X, use ioreg to get the computer's serial number.
        try:
            # subprocess may not be available, e.g. Google App Engine
            # https://github.com/pallets/werkzeug/issues/925
            from subprocess import PIPE
            from subprocess import Popen

            dump = Popen(
                ["ioreg", "-c", "IOPlatformExpertDevice", "-d", "2"], stdout=PIPE
            ).communicate()[0]
            match = re.search(b'"serial-number" = <([^>]+)', dump)

            if match is not None:
                return match.group(1)
        except (OSError, ImportError):
            pass

        # On Windows, use winreg to get the machine guid.
        if sys.platform == "win32":
            import winreg

            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    "SOFTWARE\\Microsoft\\Cryptography",
                    0,
                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
                ) as rk:
                    guid: str | bytes
                    guid_type: int
                    guid, guid_type = winreg.QueryValueEx(rk, "MachineGuid")

                    if guid_type == winreg.REG_SZ:
                        return guid.encode()

                    return guid
            except OSError:
                pass

        return None

    _machine_id = _generate()
    return _machine_id


class _ConsoleFrame:
    """Helper class so that we can reuse the frame console code for the
    standalone console.
    """

    def __init__(self, namespace: dict[str, t.Any]):
        self.console = Console(namespace)
        self.id = 0

    def eval(self, code: str) -> t.Any:
        return self.console.eval(code)


def get_pin_and_cookie_name(
    app: WSGIApplication,
) -> tuple[str, str] | tuple[None, None]:
    """Given an application object this returns a semi-stable 9 digit pin
    code and a random key.  The hope is that this is stable between
    restarts to not make debugging particularly frustrating.  If the pin
    was forcefully disabled this returns `None`.

    Second item in the resulting tuple is the cookie name for remembering.
    """
    pin = os.environ.get("WERKZEUG_DEBUG_PIN")
    rv = None
    num = None

    # Pin was explicitly disabled
    if pin == "off":
        return None, None

    # Pin was provided explicitly
    if pin is not None and pin.replace("-", "").isdecimal():
        # If there are separators in the pin, return it directly
        if "-" in pin:
            rv = pin
        else:
            num = pin

    modname = getattr(app, "__module__", t.cast(object, app).__class__.__module__)
    username: str | None

    try:
        # getuser imports the pwd module, which does not exist in Google
        # App Engine. It may also raise a KeyError if the UID does not
        # have a username, such as in Docker.
        username = getpass.getuser()
    # Python >= 3.13 only raises OSError
    except (ImportError, KeyError, OSError):
        username = None

    mod = sys.modules.get(modname)

    # This information only exists to make the cookie unique on the
    # computer, not as a security feature.
    probably_public_bits = [
        username,
        modname,
        getattr(app, "__name__", type(app).__name__),
        getattr(mod, "__file__", None),
    ]

    # This information is here to make it harder for an attacker to
    # guess the cookie name.  They are unlikely to be contained anywhere
    # within the unauthenticated debug page.
    private_bits = [str(uuid.getnode()), get_machine_id()]

    h = hashlib.sha1()
    for bit in chain(probably_public_bits, private_bits):
        if not bit:
            continue
        if isinstance(bit, str):
            bit = bit.encode()
        h.update(bit)
    h.update(b"cookiesalt")

    cookie_name = f"__wzd{h.hexdigest()[:20]}"

    # If we need to generate a pin we salt it a bit more so that we don't
    # end up with the same value and generate out 9 digits
    if num is None:
        h.update(b"pinsalt")
        num = f"{int(h.hexdigest(), 16):09d}"[:9]

    # Format the pincode in groups of digits for easier remembering if
    # we don't have a result yet.
    if rv is None:
        for group_size in 5, 4, 3:
            if len(num) % group_size == 0:
                rv = "-".join(
                    num[x : x + group_size].rjust(group_size, "0")
                    for x in range(0, len(num), group_size)
                )
                break
        else:
            rv = num

    return rv, cookie_name


class DebuggedApplication:
    """Enables debugging support for a given application::

        from werkzeug.debug import DebuggedApplication
        from myapp import app
        app = DebuggedApplication(app, evalex=True)

    The ``evalex`` argument allows evaluating expressions in any frame
    of a traceback. This works by preserving each frame with its local
    state. Some state, such as context globals, cannot be restored with
    the frame by default. When ``evalex`` is enabled,
    ``environ["werkzeug.debug.preserve_context"]`` will be a callable
    that takes a context manager, and can be called multiple times.
    Each context manager will be entered before evaluating code in the
    frame, then exited again, so they can perform setup and cleanup for
    each call.

    :param app: the WSGI application to run debugged.
    :param evalex: enable exception evaluation feature (interactive
                   debugging).  This requires a non-forking server.
    :param request_key: The key that points to the request object in this
                        environment.  This parameter is ignored in current
                        versions.
    :param console_path: the URL for a general purpose console.
    :param console_init_func: the function that is executed before starting
                              the general purpose console.  The return value
                              is used as initial namespace.
    :param show_hidden_frames: by default hidden traceback frames are skipped.
                               You can show them by setting this parameter
                               to `True`.
    :param pin_security: can be used to disable the pin based security system.
    :param pin_logging: enables the logging of the pin system.

    .. versionchanged:: 2.2
        Added the ``werkzeug.debug.preserve_context`` environ key.
    """

    _pin: str | None
    _pin_cookie: str

    def __init__(
        self,
        app: WSGIApplication,
        evalex: bool = False,
        request_key: str = "werkzeug.request",
        console_path: str = "/console",
        console_init_func: t.Callable[[], dict[str, t.Any]] | None = None,
        show_hidden_frames: bool = False,
        pin_security: bool = True,
        pin_logging: bool = True,
    ) -> None:
        if not console_init_func:
            console_init_func = None
        self.app = app
        self.evalex = evalex
        self.frames: dict[int, DebugFrameSummary | _ConsoleFrame] = {}
        self.frame_contexts: dict[int, list[t.ContextManager[None]]] = {}
        self.request_key = request_key
        self.console_path = console_path
        self.console_init_func = console_init_func
        self.show_hidden_frames = show_hidden_frames
        self.secret = gen_salt(20)
        self._failed_pin_auth = Value("B")

        self.pin_logging = pin_logging
        if pin_security:
            # Print out the pin for the debugger on standard out.
            if os.environ.get("WERKZEUG_RUN_MAIN") == "true" and pin_logging:
                _log("warning", " * Debugger is active!")
                if self.pin is None:
                    _log("warning", " * Debugger PIN disabled. DEBUGGER UNSECURED!")
                else:
                    _log("info", " * Debugger PIN: %s", self.pin)
        else:
            self.pin = None

        self.trusted_hosts: list[str] = [".localhost", "127.0.0.1"]
        """List of domains to allow requests to the debugger from. A leading dot
        allows all subdomains. This only allows ``".localhost"`` domains by
        default.

        .. versionadded:: 3.0.3
        """

    @property
    def pin(self) -> str | None:
        if not hasattr(self, "_pin"):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie  # type: ignore
        return self._pin

    @pin.setter
    def pin(self, value: str | None) -> None:
        if value is None:
            # Set _pin to None explicitly to prevent regeneration by the getter
            self._pin = None
        else:
            self._pin = value

    @property
    def pin_cookie_name(self) -> str:
        """The name of the pin cookie."""
        if not hasattr(self, "_pin_cookie"):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie  # type: ignore
        return self._pin_cookie

    def debug_application(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterator[bytes]:
        """Run the application and conserve the traceback frames."""
        contexts: list[t.ContextManager[t.Any]] = []

        if self.evalex:
            environ["werkzeug.debug.preserve_context"] = contexts.append

        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            yield from app_iter
            if hasattr(app_iter, "close"):
                app_iter.close()
        except Exception as e:
            if hasattr(app_iter, "close"):
                app_iter.close()  # type: ignore

            tb = DebugTraceback(e, skip=1, hide=not self.show_hidden_frames)

            for frame in tb.all_frames:
                self.frames[id(frame)] = frame
                self.frame_contexts[id(frame)] = contexts

            is_trusted = bool(self.check_pin_trust(environ))
            html = tb.render_debugger_html(
                evalex=self.evalex and self.check_host_trust(environ),
                secret=self.secret,
                evalex_trusted=is_trusted,
            )
            response = Response(html, status=500, mimetype="text/html")

            try:
                yield from response(environ, start_response)
            except Exception:
                # if we end up here there has been output but an error
                # occurred.  in that situation we can do nothing fancy any
                # more, better log something into the error log and fall
                # back gracefully.
                environ["wsgi.errors"].write(
                    "Debugging middleware caught exception in streamed "
                    "response at a point where response headers were already "
                    "sent.\n"
                )

            environ["wsgi.errors"].write("".join(tb.render_traceback_text()))

    def execute_command(
        self,
        request: Request,
        command: str,
        frame: DebugFrameSummary | _ConsoleFrame,
    ) -> Response:
        """Execute a command in a console."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        contexts = self.frame_contexts.get(id(frame), [])

        with ExitStack() as exit_stack:
            for cm in contexts:
                exit_stack.enter_context(cm)

            return Response(frame.eval(command), mimetype="text/html")

    def display_console(self, request: Request) -> Response:
        """Display a standalone shell."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        if 0 not in self.frames:
            if self.console_init_func is None:
                ns = {}
            else:
                ns = dict(self.console_init_func())
            ns.setdefault("app", self.app)
            self.frames[0] = _ConsoleFrame(ns)
        is_trusted = bool(self.check_pin_trust(request.environ))
        return Response(
            render_console_html(secret=self.secret, evalex_trusted=is_trusted),
            mimetype="text/html",
        )

    def get_resource(self, request: Request, filename: str) -> Response:
        """Return a static resource from the shared folder."""
        path = join("shared", basename(filename))

        try:
            data = pkgutil.get_data(__package__, path)
        except OSError:
            return NotFound()  # type: ignore[return-value]
        else:
            if data is None:
                return NotFound()  # type: ignore[return-value]

            etag = str(adler32(data) & 0xFFFFFFFF)
            return send_file(
                BytesIO(data), request.environ, download_name=filename, etag=etag
            )

    def check_pin_trust(self, environ: WSGIEnvironment) -> bool | None:
        """Checks if the request passed the pin test.  This returns `True` if the
        request is trusted on a pin/cookie basis and returns `False` if not.
        Additionally if the cookie's stored pin hash is wrong it will return
        `None` so that appropriate action can be taken.
        """
        if self.pin is None:
            return True

        # If we failed too many times, then we're locked out.
        if self._failed_pin_auth.value >= 10:
            return False

        val = parse_cookie(environ).get(self.pin_cookie_name)
        if not val or "|" not in val:
            return False
        ts_str, _, pin_hash = val.partition("|")

        try:
            ts = int(ts_str)
        except ValueError:
            return False

        if pin_hash != hash_pin(self.pin):
            return None
        return (time.time() - PIN_TIME) < ts

    def check_host_trust(self, environ: WSGIEnvironment) -> bool:
        return host_is_trusted(environ.get("HTTP_HOST"), self.trusted_hosts)

    def _fail_pin_auth(self) -> None:
        with self._failed_pin_auth.get_lock():
            count = self._failed_pin_auth.value
            self._failed_pin_auth.value = count + 1

        time.sleep(5.0 if count > 5 else 0.5)

    def pin_auth(self, request: Request) -> Response:
        """Authenticates with the pin."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        exhausted = False
        auth = False
        trust = self.check_pin_trust(request.environ)
        pin = t.cast(str, self.pin)

        # If the trust return value is `None` it means that the cookie is
        # set but the stored pin hash value is bad.  This means that the
        # pin was changed.  In this case we count a bad auth and unset the
        # cookie.  This way it becomes harder to guess the cookie name
        # instead of the pin as we still count up failures.
        bad_cookie = False
        if trust is None:
            self._fail_pin_auth()
            bad_cookie = True

        # If we're trusted, we're authenticated.
        elif trust:
            auth = True

        # If we failed too many times, then we're locked out.
        elif self._failed_pin_auth.value >= 10:
            exhausted = True

        # Otherwise go through pin based authentication
        else:
            entered_pin = request.args["pin"]

            if entered_pin.strip().replace("-", "") == pin.replace("-", ""):
                self._failed_pin_auth.value = 0
                auth = True
            else:
                self._fail_pin_auth()

        rv = Response(
            json.dumps({"auth": auth, "exhausted": exhausted}),
            mimetype="application/json",
        )
        if auth:
            rv.set_cookie(
                self.pin_cookie_name,
                f"{int(time.time())}|{hash_pin(pin)}",
                httponly=True,
                samesite="Strict",
                secure=request.is_secure,
            )
        elif bad_cookie:
            rv.delete_cookie(self.pin_cookie_name)
        return rv

    def log_pin_request(self, request: Request) -> Response:
        """Log the pin if needed."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        if self.pin_logging and self.pin is not None:
            _log(
                "info", " * To enable the debugger you need to enter the security pin:"
            )
            _log("info", " * Debugger pin code: %s", self.pin)
        return Response("")

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Dispatch the requests."""
        # important: don't ever access a function here that reads the incoming
        # form data!  Otherwise the application won't have access to that data
        # any more!
        request = Request(environ)
        response = self.debug_application
        if request.args.get("__debugger__") == "yes":
            cmd = request.args.get("cmd")
            arg = request.args.get("f")
            secret = request.args.get("s")
            frame = self.frames.get(request.args.get("frm", type=int))  # type: ignore
            if cmd == "resource" and arg:
                response = self.get_resource(request, arg)  # type: ignore
            elif cmd == "pinauth" and secret == self.secret:
                response = self.pin_auth(request)  # type: ignore
            elif cmd == "printpin" and secret == self.secret:
                response = self.log_pin_request(request)  # type: ignore
            elif (
                self.evalex
                and cmd is not None
                and frame is not None
                and self.secret == secret
                and self.check_pin_trust(environ)
            ):
                response = self.execute_command(request, cmd, frame)  # type: ignore
        elif (
            self.evalex
            and self.console_path is not None
            and request.path == self.console_path
        ):
            response = self.display_console(request)  # type: ignore
        return response(environ, start_response)
<|endoftext|>from __future__ import annotations

import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t

from markupsafe import escape

from ..utils import cached_property
from .console import Console

HEADER = """\
<!doctype html>
<html lang=en>
  <head>
    <title>%(title)s // Werkzeug Debugger</title>
    <link rel="stylesheet" href="?__debugger__=yes&amp;cmd=resource&amp;f=style.css">
    <link rel="shortcut icon"
        href="?__debugger__=yes&amp;cmd=resource&amp;f=console.png">
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=debugger.js"></script>
    <script>
      var CONSOLE_MODE = %(console)s,
          EVALEX = %(evalex)s,
          EVALEX_TRUSTED = %(evalex_trusted)s,
          SECRET = "%(secret)s";
    </script>
  </head>
  <body style="background-color: #fff">
    <div class="debugger">
"""

FOOTER = """\
      <div class="footer">
        Brought to you by <strong class="arthur">DON'T PANIC</strong>, your
        friendly Werkzeug powered traceback interpreter.
      </div>
    </div>

    <div class="pin-prompt">
      <div class="inner">
        <h3>Console Locked</h3>
        <p>
          The console is locked and needs to be unlocked by entering the PIN.
          You can find the PIN printed out on the standard output of your
          shell that runs the server.
        <form>
          <p>PIN:
            <input type=text name=pin size=14>
            <input type=submit name=btn value="Confirm Pin">
        </form>
      </div>
    </div>
  </body>
</html>
"""

PAGE_HTML = (
    HEADER
    + """\
<h1>%(exception_type)s</h1>
<div class="detail">
  <p class="errormsg">%(exception)s</p>
</div>
<h2 class="traceback">Traceback <em>(most recent call last)</em></h2>
%(summary)s
<div class="plain">
    <p>
      This is the Copy/Paste friendly version of the traceback.
    </p>
    <textarea cols="50" rows="10" name="code" readonly>%(plaintext)s</textarea>
</div>
<div class="explanation">
  The debugger caught an exception in your WSGI application.  You can now
  look at the traceback which led to the error.  <span class="nojavascript">
  If you enable JavaScript you can also use additional features such as code
  execution (if the evalex feature is enabled), automatic pasting of the
  exceptions and much more.</span>
</div>
"""
    + FOOTER
    + """
<!--

%(plaintext_cs)s

-->
"""
)

CONSOLE_HTML = (
    HEADER
    + """\
<h1>Interactive Console</h1>
<div class="explanation">
In this console you can execute Python expressions in the context of the
application.  The initial namespace was created by the debugger automatically.
</div>
<div class="console"><div class="inner">The Console requires JavaScript.</div></div>
"""
    + FOOTER
)

SUMMARY_HTML = """\
<div class="%(classes)s">
  %(title)s
  <ul>%(frames)s</ul>
  %(description)s
</div>
"""

FRAME_HTML = """\
<div class="frame" id="frame-%(id)d">
  <h4>File <cite class="filename">"%(filename)s"</cite>,
      line <em class="line">%(lineno)s</em>,
      in <code class="function">%(function_name)s</code></h4>
  <div class="source %(library)s">%(lines)s</div>
</div>
"""


def _process_traceback(
    exc: BaseException,
    te: traceback.TracebackException | None = None,
    *,
    skip: int = 0,
    hide: bool = True,
) -> traceback.TracebackException:
    if te is None:
        te = traceback.TracebackException.from_exception(exc, lookup_lines=False)

    # Get the frames the same way StackSummary.extract did, in order
    # to match each frame with the FrameSummary to augment.
    frame_gen = traceback.walk_tb(exc.__traceback__)
    limit = getattr(sys, "tracebacklimit", None)

    if limit is not None:
        if limit < 0:
            limit = 0

        frame_gen = itertools.islice(frame_gen, limit)

    if skip:
        frame_gen = itertools.islice(frame_gen, skip, None)
        del te.stack[:skip]

    new_stack: list[DebugFrameSummary] = []
    hidden = False

    # Match each frame with the FrameSummary that was generated.
    # Hide frames using Paste's __traceback_hide__ rules. Replace
    # all visible FrameSummary with DebugFrameSummary.
    for (f, _), fs in zip(frame_gen, te.stack):
        if hide:
            hide_value = f.f_locals.get("__traceback_hide__", False)

