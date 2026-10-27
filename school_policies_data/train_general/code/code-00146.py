        " (e.g. a bad password), or your browser doesn't understand"
        " how to supply the credentials required."
    )

    def __init__(
        self,
        description: str | None = None,
        response: SansIOResponse | None = None,
        www_authenticate: None | (WWWAuthenticate | t.Iterable[WWWAuthenticate]) = None,
    ) -> None:
        super().__init__(description, response)

        from .datastructures import WWWAuthenticate

        if isinstance(www_authenticate, WWWAuthenticate):
            www_authenticate = (www_authenticate,)

        self.www_authenticate = www_authenticate

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.www_authenticate:
            headers.extend(("WWW-Authenticate", str(x)) for x in self.www_authenticate)
        return headers


class Forbidden(HTTPException):
    """*403* `Forbidden`

    Raise if the user doesn't have the permission for the requested resource
    but was authenticated.
    """

    code = 403
    description = (
        "You don't have the permission to access the requested"
        " resource. It is either read-protected or not readable by the"
        " server."
    )


class NotFound(HTTPException):
    """*404* `Not Found`

    Raise if a resource does not exist and never existed.
    """

    code = 404
    description = (
        "The requested URL was not found on the server. If you entered"
        " the URL manually please check your spelling and try again."
    )


class MethodNotAllowed(HTTPException):
    """*405* `Method Not Allowed`

    Raise if the server used a method the resource does not handle.  For
    example `POST` if the resource is view only.  Especially useful for REST.

    The first argument for this exception should be a list of allowed methods.
    Strictly speaking the response would be invalid if you don't provide valid
    methods in the header which you can do with that list.
    """

    code = 405
    description = "The method is not allowed for the requested URL."

    def __init__(
        self,
        valid_methods: t.Iterable[str] | None = None,
        description: str | None = None,
        response: SansIOResponse | None = None,
    ) -> None:
        """Takes an optional list of valid http methods
        starting with werkzeug 0.3 the list will be mandatory."""
        super().__init__(description=description, response=response)
        self.valid_methods = valid_methods

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.valid_methods:
            headers.append(("Allow", ", ".join(self.valid_methods)))
        return headers


class NotAcceptable(HTTPException):
    """*406* `Not Acceptable`

    Raise if the server can't return any content conforming to the
    `Accept` headers of the client.
    """

    code = 406
    description = (
        "The resource identified by the request is only capable of"
        " generating response entities which have content"
        " characteristics not acceptable according to the accept"
        " headers sent in the request."
    )


class RequestTimeout(HTTPException):
    """*408* `Request Timeout`

    Raise to signalize a timeout.
    """

    code = 408
    description = (
        "The server closed the network connection because the browser"
        " didn't finish the request within the specified time."
    )


class Conflict(HTTPException):
    """*409* `Conflict`

    Raise to signal that a request cannot be completed because it conflicts
    with the current state on the server.

    .. versionadded:: 0.7
    """

    code = 409
    description = (
        "A conflict happened while processing the request. The"
        " resource might have been modified while the request was being"
        " processed."
    )


class Gone(HTTPException):
    """*410* `Gone`

    Raise if a resource existed previously and went away without new location.
    """

    code = 410
    description = (
        "The requested URL is no longer available on this server and"
        " there is no forwarding address. If you followed a link from a"
        " foreign page, please contact the author of this page."
    )


class LengthRequired(HTTPException):
    """*411* `Length Required`

    Raise if the browser submitted data but no ``Content-Length`` header which
    is required for the kind of processing the server does.
    """

    code = 411
    description = (
        "A request with this method requires a valid <code>Content-"
        "Length</code> header."
    )


class PreconditionFailed(HTTPException):
    """*412* `Precondition Failed`

    Status code used in combination with ``If-Match``, ``If-None-Match``, or
    ``If-Unmodified-Since``.
    """

    code = 412
    description = (
        "The precondition on the request for the URL failed positive evaluation."
    )


class RequestEntityTooLarge(HTTPException):
    """*413* `Request Entity Too Large`

    The status code one should return if the data submitted exceeded a given
    limit.
    """

    code = 413
    description = "The data value transmitted exceeds the capacity limit."


class RequestURITooLarge(HTTPException):
    """*414* `Request URI Too Large`

    Like *413* but for too long URLs.
    """

    code = 414
    description = (
        "The length of the requested URL exceeds the capacity limit for"
        " this server. The request cannot be processed."
    )


class UnsupportedMediaType(HTTPException):
    """*415* `Unsupported Media Type`

    The status code returned if the server is unable to handle the media type
    the client transmitted.
    """

    code = 415
    description = (
        "The server does not support the media type transmitted in the request."
    )


class RequestedRangeNotSatisfiable(HTTPException):
    """*416* `Requested Range Not Satisfiable`

    The client asked for an invalid part of the file.

    .. versionadded:: 0.7
    """

    code = 416
    description = "The server cannot provide the requested range."

    def __init__(
        self,
        length: int | None = None,
        units: str = "bytes",
        description: str | None = None,
        response: SansIOResponse | None = None,
    ) -> None:
        """Takes an optional `Content-Range` header value based on ``length``
        parameter.
        """
        super().__init__(description=description, response=response)
        self.length = length
        self.units = units

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.length is not None:
            headers.append(("Content-Range", f"{self.units} */{self.length}"))
        return headers


class ExpectationFailed(HTTPException):
    """*417* `Expectation Failed`

    The server cannot meet the requirements of the Expect request-header.

    .. versionadded:: 0.7
    """

    code = 417
    description = "The server could not meet the requirements of the Expect header"


class ImATeapot(HTTPException):
    """*418* `I'm a teapot`

    The server should return this if it is a teapot and someone attempted
    to brew coffee with it.

    .. versionadded:: 0.7
    """

    code = 418
    description = "This server is a teapot, not a coffee machine"


class MisdirectedRequest(HTTPException):
    """421 Misdirected Request

    Indicates that the request was directed to a server that is not able to
    produce a response.

    .. versionadded:: 3.1
    """

    code = 421
    description = "The server is not able to produce a response."


class UnprocessableEntity(HTTPException):
    """*422* `Unprocessable Entity`

    Used if the request is well formed, but the instructions are otherwise
    incorrect.
    """

    code = 422
    description = (
        "The request was well-formed but was unable to be followed due"
        " to semantic errors."
    )


class Locked(HTTPException):
    """*423* `Locked`

    Used if the resource that is being accessed is locked.
    """

    code = 423
    description = "The resource that is being accessed is locked."


class FailedDependency(HTTPException):
    """*424* `Failed Dependency`

    Used if the method could not be performed on the resource
    because the requested action depended on another action and that action failed.
    """

    code = 424
    description = (
        "The method could not be performed on the resource because the"
        " requested action depended on another action and that action"
        " failed."
    )


class PreconditionRequired(HTTPException):
    """*428* `Precondition Required`

    The server requires this request to be conditional, typically to prevent
    the lost update problem, which is a race condition between two or more
    clients attempting to update a resource through PUT or DELETE. By requiring
    each client to include a conditional header ("If-Match" or "If-Unmodified-
    Since") with the proper value retained from a recent GET request, the
    server ensures that each client has at least seen the previous revision of
    the resource.
    """

    code = 428
    description = (
        "This request is required to be conditional; try using"
        ' "If-Match" or "If-Unmodified-Since".'
    )


class _RetryAfter(HTTPException):
    """Adds an optional ``retry_after`` parameter which will set the
    ``Retry-After`` header. May be an :class:`int` number of seconds or
    a :class:`~datetime.datetime`.
    """

    def __init__(
        self,
        description: str | None = None,
        response: SansIOResponse | None = None,
        retry_after: datetime | int | None = None,
    ) -> None:
        super().__init__(description, response)
        self.retry_after = retry_after

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)

        if self.retry_after:
            if isinstance(self.retry_after, datetime):
                from .http import http_date

                value = http_date(self.retry_after)
            else:
                value = str(self.retry_after)

            headers.append(("Retry-After", value))

        return headers


class TooManyRequests(_RetryAfter):
    """*429* `Too Many Requests`

    The server is limiting the rate at which this user receives
    responses, and this request exceeds that rate. (The server may use
    any convenient method to identify users and their request rates).
    The server may include a "Retry-After" header to indicate how long
    the user should wait before retrying.

    :param retry_after: If given, set the ``Retry-After`` header to this
        value. May be an :class:`int` number of seconds or a
        :class:`~datetime.datetime`.

    .. versionchanged:: 1.0
        Added ``retry_after`` parameter.
    """

    code = 429
    description = "This user has exceeded an allotted request count. Try again later."


class RequestHeaderFieldsTooLarge(HTTPException):
    """*431* `Request Header Fields Too Large`

    The server refuses to process the request because the header fields are too
    large. One or more individual fields may be too large, or the set of all
    headers is too large.
    """

    code = 431
    description = "One or more header fields exceeds the maximum size."


class UnavailableForLegalReasons(HTTPException):
    """*451* `Unavailable For Legal Reasons`

    This status code indicates that the server is denying access to the
    resource as a consequence of a legal demand.
    """

    code = 451
    description = "Unavailable for legal reasons."


class InternalServerError(HTTPException):
    """*500* `Internal Server Error`

    Raise if an internal server error occurred.  This is a good fallback if an
    unknown error occurred in the dispatcher.

    .. versionchanged:: 1.0.0
        Added the :attr:`original_exception` attribute.
    """

    code = 500
    description = (
        "The server encountered an internal error and was unable to"
        " complete your request. Either the server is overloaded or"
        " there is an error in the application."
    )

    def __init__(
        self,
        description: str | None = None,
        response: SansIOResponse | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        #: The original exception that caused this 500 error. Can be
        #: used by frameworks to provide context when handling
        #: unexpected errors.
        self.original_exception = original_exception
        super().__init__(description=description, response=response)


class NotImplemented(HTTPException):
    """*501* `Not Implemented`

    Raise if the application does not support the action requested by the
    browser.
    """

    code = 501
    description = "The server does not support the action requested by the browser."


class BadGateway(HTTPException):
    """*502* `Bad Gateway`

    If you do proxying in your application you should return this status code
    if you received an invalid response from the upstream server it accessed
    in attempting to fulfill the request.
    """

    code = 502
    description = (
        "The proxy server received an invalid response from an upstream server."
    )


class ServiceUnavailable(_RetryAfter):
    """*503* `Service Unavailable`

    Status code you should return if a service is temporarily
    unavailable.

    :param retry_after: If given, set the ``Retry-After`` header to this
        value. May be an :class:`int` number of seconds or a
        :class:`~datetime.datetime`.

    .. versionchanged:: 1.0
        Added ``retry_after`` parameter.
    """

    code = 503
    description = (
        "The server is temporarily unable to service your request due"
        " to maintenance downtime or capacity problems. Please try"
        " again later."
    )


class GatewayTimeout(HTTPException):
    """*504* `Gateway Timeout`

    Status code you should return if a connection to an upstream server
    times out.
    """

    code = 504
    description = "The connection to an upstream server timed out."


class HTTPVersionNotSupported(HTTPException):
    """*505* `HTTP Version Not Supported`

    The server does not support the HTTP protocol version used in the request.
    """

    code = 505
    description = (
        "The server does not support the HTTP protocol version used in the request."
    )


default_exceptions: dict[int, type[HTTPException]] = {}


def _find_exceptions() -> None:
    for obj in globals().values():
        try:
            is_http_exception = issubclass(obj, HTTPException)
        except TypeError:
            is_http_exception = False
        if not is_http_exception or obj.code is None:
            continue
        old_obj = default_exceptions.get(obj.code, None)
        if old_obj is not None and issubclass(obj, old_obj):
            continue
        default_exceptions[obj.code] = obj


_find_exceptions()
del _find_exceptions


class Aborter:
    """When passed a dict of code -> exception items it can be used as
    callable that raises exceptions.  If the first argument to the
    callable is an integer it will be looked up in the mapping, if it's
    a WSGI application it will be raised in a proxy exception.

    The rest of the arguments are forwarded to the exception constructor.
    """

    def __init__(
        self,
        mapping: dict[int, type[HTTPException]] | None = None,
        extra: dict[int, type[HTTPException]] | None = None,
    ) -> None:
        if mapping is None:
            mapping = default_exceptions
        self.mapping = dict(mapping)
        if extra is not None:
            self.mapping.update(extra)

    def __call__(
        self, code: int | SansIOResponse, *args: t.Any, **kwargs: t.Any
    ) -> t.NoReturn:
        from .sansio.response import Response

        if isinstance(code, Response):
            raise HTTPException(response=code)

        if code not in self.mapping:
            raise LookupError(f"no exception for {code!r}")

        raise self.mapping[code](*args, **kwargs)


def abort(status: int | SansIOResponse, *args: t.Any, **kwargs: t.Any) -> t.NoReturn:
    """Raises an :py:exc:`HTTPException` for the given status code or WSGI
    application.

    If a status code is given, it will be looked up in the list of
    exceptions and will raise that exception.  If passed a WSGI application,
    it will wrap it in a proxy WSGI exception and raise that::

       abort(404)  # 404 Not Found
       abort(Response('Hello World'))

    """
    _aborter(status, *args, **kwargs)


_aborter: Aborter = Aborter()
<|endoftext|>from __future__ import annotations

import collections.abc as cabc


class ETags(cabc.Collection[str]):
    """A set that can be used to check if one etag is present in a collection
    of etags.
    """

    def __init__(
        self,
        strong_etags: cabc.Iterable[str] | None = None,
        weak_etags: cabc.Iterable[str] | None = None,
        star_tag: bool = False,
    ):
        if not star_tag and strong_etags:
            self._strong = frozenset(strong_etags)
        else:
            self._strong = frozenset()

        self._weak = frozenset(weak_etags or ())
        self.star_tag = star_tag

    def as_set(self, include_weak: bool = False) -> set[str]:
        """Convert the `ETags` object into a python set.  Per default all the
        weak etags are not part of this set."""
        rv = set(self._strong)
        if include_weak:
            rv.update(self._weak)
        return rv

    def is_weak(self, etag: str) -> bool:
        """Check if an etag is weak."""
        return etag in self._weak

    def is_strong(self, etag: str) -> bool:
        """Check if an etag is strong."""
        return etag in self._strong

    def contains_weak(self, etag: str) -> bool:
        """Check if an etag is part of the set including weak and strong tags."""
        return self.is_weak(etag) or self.contains(etag)

    def contains(self, etag: str) -> bool:
        """Check if an etag is part of the set ignoring weak tags.
        It is also possible to use the ``in`` operator.
        """
        if self.star_tag:
            return True
        return self.is_strong(etag)

    def contains_raw(self, etag: str) -> bool:
        """When passed a quoted tag it will check if this tag is part of the
        set.  If the tag is weak it is checked against weak and strong tags,
        otherwise strong only."""
        from ..http import unquote_etag

        etag, weak = unquote_etag(etag)
        if weak:
            return self.contains_weak(etag)
        return self.contains(etag)

    def to_header(self) -> str:
        """Convert the etags set into a HTTP header string."""
        if self.star_tag:
            return "*"
        return ", ".join(
            [f'"{x}"' for x in self._strong] + [f'W/"{x}"' for x in self._weak]
        )

    def __call__(
        self,
        etag: str | None = None,
        data: bytes | None = None,
        include_weak: bool = False,
    ) -> bool:
        if etag is None:
            if data is None:
                raise TypeError("'data' is required when 'etag' is not given.")

            from ..http import generate_etag

            etag = generate_etag(data)
        if include_weak:
            if etag in self._weak:
                return True
        return etag in self._strong

    def __bool__(self) -> bool:
        return bool(self.star_tag or self._strong or self._weak)

    def __str__(self) -> str:
        return self.to_header()

    def __len__(self) -> int:
        return len(self._strong)

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._strong)

    def __contains__(self, etag: str) -> bool:  # type: ignore[override]
        return self.contains(etag)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"
<|endoftext|>from __future__ import annotations

import codecs
import collections.abc as cabc
import re
import typing as t

from .structures import ImmutableList


class Accept(ImmutableList[tuple[str, float]]):
    """An :class:`Accept` object is just a list subclass for lists of
    ``(value, quality)`` tuples.  It is automatically sorted by specificity
    and quality.

    All :class:`Accept` objects work similar to a list but provide extra
    functionality for working with the data.  Containment checks are
    normalized to the rules of that header:

    >>> a = CharsetAccept([('ISO-8859-1', 1), ('utf-8', 0.7)])
    >>> a.best
    'ISO-8859-1'
    >>> 'iso-8859-1' in a
    True
    >>> 'UTF8' in a
    True
    >>> 'utf7' in a
    False

    To get the quality for an item you can use normal item lookup:

    >>> print a['utf-8']
    0.7
    >>> a['utf7']
    0

    .. versionchanged:: 0.5
       :class:`Accept` objects are forced immutable now.

    .. versionchanged:: 1.0.0
       :class:`Accept` internal values are no longer ordered
       alphabetically for equal quality tags. Instead the initial
       order is preserved.

    """

    def __init__(
        self, values: Accept | cabc.Iterable[tuple[str, float]] | None = ()
    ) -> None:
        if values is None:
            super().__init__()
            self.provided = False
        elif isinstance(values, Accept):
            self.provided = values.provided
            super().__init__(values)
        else:
            self.provided = True
            values = sorted(
                values, key=lambda x: (self._specificity(x[0]), x[1]), reverse=True
            )
            super().__init__(values)

    def _specificity(self, value: str) -> tuple[bool, ...]:
        """Returns a tuple describing the value's specificity."""
        return (value != "*",)

    def _value_matches(self, value: str, item: str) -> bool:
        """Check if a value matches a given accept item."""
        return item == "*" or item.lower() == value.lower()

    @t.overload
    def __getitem__(self, key: str) -> float: ...
    @t.overload
    def __getitem__(self, key: t.SupportsIndex) -> tuple[str, float]: ...
    @t.overload
    def __getitem__(self, key: slice) -> list[tuple[str, float]]: ...
    def __getitem__(
        self, key: str | t.SupportsIndex | slice
    ) -> float | tuple[str, float] | list[tuple[str, float]]:
        """Besides index lookup (getting item n) you can also pass it a string
        to get the quality for the item.  If the item is not in the list, the
        returned quality is ``0``.
        """
        if isinstance(key, str):
            return self.quality(key)
        return list.__getitem__(self, key)

    def quality(self, key: str) -> float:
        """Returns the quality of the key.

        .. versionadded:: 0.6
           In previous versions you had to use the item-lookup syntax
           (eg: ``obj[key]`` instead of ``obj.quality(key)``)
        """
        for item, quality in self:
            if self._value_matches(key, item):
                return quality
        return 0

    def __contains__(self, value: str) -> bool:  # type: ignore[override]
        for item, _quality in self:
            if self._value_matches(value, item):
                return True
        return False

    def __repr__(self) -> str:
        pairs_str = ", ".join(f"({x!r}, {y})" for x, y in self)
        return f"{type(self).__name__}([{pairs_str}])"

    def index(self, key: str | tuple[str, float]) -> int:  # type: ignore[override]
        """Get the position of an entry or raise :exc:`ValueError`.

        :param key: The key to be looked up.

        .. versionchanged:: 0.5
           This used to raise :exc:`IndexError`, which was inconsistent
           with the list API.
        """
        if isinstance(key, str):
            for idx, (item, _quality) in enumerate(self):
                if self._value_matches(key, item):
                    return idx
            raise ValueError(key)
        return list.index(self, key)

    def find(self, key: str | tuple[str, float]) -> int:
        """Get the position of an entry or return -1.

        :param key: The key to be looked up.
        """
        try:
            return self.index(key)
        except ValueError:
            return -1

    def values(self) -> cabc.Iterator[str]:
        """Iterate over all values."""
        for item in self:
            yield item[0]

    def to_header(self) -> str:
        """Convert the header set into an HTTP header string."""
        result = []
        for value, quality in self:
            if quality != 1:
                value = f"{value};q={quality}"
            result.append(value)
        return ",".join(result)

    def __str__(self) -> str:
        return self.to_header()

    def _best_single_match(self, match: str) -> tuple[str, float] | None:
        for client_item, quality in self:
            if self._value_matches(match, client_item):
                # self is sorted by specificity descending, we can exit
                return client_item, quality
        return None

    @t.overload
    def best_match(self, matches: cabc.Iterable[str]) -> str | None: ...
    @t.overload
    def best_match(self, matches: cabc.Iterable[str], default: str = ...) -> str: ...
    def best_match(
        self, matches: cabc.Iterable[str], default: str | None = None
    ) -> str | None:
        """Returns the best match from a list of possible matches based
        on the specificity and quality of the client. If two items have the
        same quality and specificity, the one is returned that comes first.

        :param matches: a list of matches to check for
        :param default: the value that is returned if none match
        """
        result = default
        best_quality: float = -1
        best_specificity: tuple[float, ...] = (-1,)
        for server_item in matches:
            match = self._best_single_match(server_item)
            if not match:
                continue
            client_item, quality = match
            specificity = self._specificity(client_item)
            if quality <= 0 or quality < best_quality:
                continue
            # better quality or same quality but more specific => better match
            if quality > best_quality or specificity > best_specificity:
                result = server_item
                best_quality = quality
                best_specificity = specificity
        return result

    @property
    def best(self) -> str | None:
        """The best match as value."""
        if self:
            return self[0][0]

        return None


_mime_split_re = re.compile(r"/|(?:[ \t]*;[ \t]*)")


def _normalize_mime(value: str) -> list[str]:
    return _mime_split_re.split(value.lower())


class MIMEAccept(Accept):
    """Like :class:`Accept` but with special methods and behavior for
    mimetypes.
    """

    def _specificity(self, value: str) -> tuple[bool, ...]:
        return tuple(x != "*" for x in _mime_split_re.split(value))

    def _value_matches(self, value: str, item: str) -> bool:
        # item comes from the client, can't match if it's invalid.
        if "/" not in item:
            return False

        # value comes from the application, tell the developer when it
        # doesn't look valid.
        if "/" not in value:
            raise ValueError(f"invalid mimetype {value!r}")

        # Split the match value into type, subtype, and a sorted list of parameters.
        normalized_value = _normalize_mime(value)
        value_type, value_subtype = normalized_value[:2]
        value_params = sorted(normalized_value[2:])

        # "*/*" is the only valid value that can start with "*".
        if value_type == "*" and value_subtype != "*":
            raise ValueError(f"invalid mimetype {value!r}")

        # Split the accept item into type, subtype, and parameters.
        normalized_item = _normalize_mime(item)
        item_type, item_subtype = normalized_item[:2]
        item_params = sorted(normalized_item[2:])

        # "*/not-*" from the client is invalid, can't match.
        if item_type == "*" and item_subtype != "*":
            return False

        return (
            (item_type == "*" and item_subtype == "*")
            or (value_type == "*" and value_subtype == "*")
        ) or (
            item_type == value_type
            and (
                item_subtype == "*"
                or value_subtype == "*"
                or (item_subtype == value_subtype and item_params == value_params)
            )
        )

    @property
    def accept_html(self) -> bool:
        """True if this object accepts HTML."""
        return "text/html" in self or self.accept_xhtml  # type: ignore[comparison-overlap]

    @property
    def accept_xhtml(self) -> bool:
        """True if this object accepts XHTML."""
        return "application/xhtml+xml" in self or "application/xml" in self  # type: ignore[comparison-overlap]

    @property
    def accept_json(self) -> bool:
        """True if this object accepts JSON."""
        return "application/json" in self  # type: ignore[comparison-overlap]


_locale_delim_re = re.compile(r"[_-]")


def _normalize_lang(value: str) -> list[str]:
    """Process a language tag for matching."""
    return _locale_delim_re.split(value.lower())


class LanguageAccept(Accept):
    """Like :class:`Accept` but with normalization for language tags."""

    def _value_matches(self, value: str, item: str) -> bool:
        return item == "*" or _normalize_lang(value) == _normalize_lang(item)

    @t.overload
    def best_match(self, matches: cabc.Iterable[str]) -> str | None: ...
    @t.overload
    def best_match(self, matches: cabc.Iterable[str], default: str = ...) -> str: ...
    def best_match(
        self, matches: cabc.Iterable[str], default: str | None = None
    ) -> str | None:
        """Given a list of supported values, finds the best match from
        the list of accepted values.

        Language tags are normalized for the purpose of matching, but
        are returned unchanged.

        If no exact match is found, this will fall back to matching
        the first subtag (primary language only), first with the
        accepted values then with the match values. This partial is not
        applied to any other language subtags.

        The default is returned if no exact or fallback match is found.

        :param matches: A list of supported languages to find a match.
        :param default: The value that is returned if none match.
        """
        # Look for an exact match first. If a client accepts "en-US",
        # "en-US" is a valid match at this point.
        result = super().best_match(matches)

        if result is not None:
            return result

        # Fall back to accepting primary tags. If a client accepts
        # "en-US", "en" is a valid match at this point. Need to use
        # re.split to account for 2 or 3 letter codes.
        fallback = Accept(
            [(_locale_delim_re.split(item[0], 1)[0], item[1]) for item in self]
        )
        result = fallback.best_match(matches)

        if result is not None:
            return result

        # Fall back to matching primary tags. If the client accepts
        # "en", "en-US" is a valid match at this point.
        fallback_matches = [_locale_delim_re.split(item, 1)[0] for item in matches]
        result = super().best_match(fallback_matches)

        # Return a value from the original match list. Find the first
        # original value that starts with the matched primary tag.
        if result is not None:
            return next(item for item in matches if item.startswith(result))

        return default


class CharsetAccept(Accept):
    """Like :class:`Accept` but with normalization for charsets."""

    def _value_matches(self, value: str, item: str) -> bool:
        def _normalize(name: str) -> str:
            try:
                return codecs.lookup(name).name
            except LookupError:
                return name.lower()

        return item == "*" or _normalize(value) == _normalize(item)
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t
from datetime import datetime

if t.TYPE_CHECKING:
    import typing_extensions as te

T = t.TypeVar("T")


class IfRange:
    """Very simple object that represents the `If-Range` header in parsed
    form.  It will either have neither a etag or date or one of either but
    never both.

    .. versionadded:: 0.7
    """

    def __init__(self, etag: str | None = None, date: datetime | None = None):
        #: The etag parsed and unquoted.  Ranges always operate on strong
        #: etags so the weakness information is not necessary.
        self.etag = etag
        #: The date in parsed format or `None`.
        self.date = date

    def to_header(self) -> str:
        """Converts the object back into an HTTP header."""
        if self.date is not None:
            return http.http_date(self.date)
        if self.etag is not None:
            return http.quote_etag(self.etag)
        return ""

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


class Range:
    """Represents a ``Range`` header. All methods only support only
    bytes as the unit. Stores a list of ranges if given, but the methods
    only work if only one range is provided.

    :raise ValueError: If the ranges provided are invalid.

    .. versionchanged:: 0.15
        The ranges passed in are validated.

    .. versionadded:: 0.7
    """

    def __init__(
        self, units: str, ranges: cabc.Sequence[tuple[int, int | None]]
    ) -> None:
        #: The units of this range.  Usually "bytes".
        self.units = units
        #: A list of ``(begin, end)`` tuples for the range header provided.
        #: The ranges are non-inclusive.
        self.ranges = ranges

        for start, end in ranges:
            if start is None or (end is not None and (start < 0 or start >= end)):
                raise ValueError(f"{(start, end)} is not a valid range.")

    def range_for_length(self, length: int | None) -> tuple[int, int] | None:
        """If the range is for bytes, the length is not None and there is
        exactly one range and it is satisfiable it returns a ``(start, stop)``
        tuple, otherwise `None`.
        """
        if self.units != "bytes" or length is None or len(self.ranges) != 1:
            return None
        start, end = self.ranges[0]
        if end is None:
            end = length
            if start < 0:
                start += length
        if http.is_byte_range_valid(start, end, length):
            return start, min(end, length)
        return None

    def make_content_range(self, length: int | None) -> ContentRange | None:
        """Creates a :class:`~werkzeug.datastructures.ContentRange` object
        from the current range and given content length.
        """
        rng = self.range_for_length(length)
        if rng is not None:
            return ContentRange(self.units, rng[0], rng[1], length)
        return None

    def to_header(self) -> str:
        """Converts the object back into an HTTP header."""
        ranges = []
        for begin, end in self.ranges:
            if end is None:
                ranges.append(f"{begin}-" if begin >= 0 else str(begin))
            else:
                ranges.append(f"{begin}-{end - 1}")
        return f"{self.units}={','.join(ranges)}"

    def to_content_range_header(self, length: int | None) -> str | None:
        """Converts the object into `Content-Range` HTTP header,
        based on given length
        """
        range = self.range_for_length(length)
        if range is not None:
            return f"{self.units} {range[0]}-{range[1] - 1}/{length}"
        return None

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


class _CallbackProperty(t.Generic[T]):
    def __set_name__(self, owner: type[ContentRange], name: str) -> None:
        self.attr = f"_{name}"

    @t.overload
    def __get__(self, instance: None, owner: None) -> te.Self: ...
    @t.overload
    def __get__(self, instance: ContentRange, owner: type[ContentRange]) -> T: ...
    def __get__(
        self, instance: ContentRange | None, owner: type[ContentRange] | None
    ) -> te.Self | T:
        if instance is None:
            return self

        return instance.__dict__[self.attr]  # type: ignore[no-any-return]

    def __set__(self, instance: ContentRange, value: T) -> None:
        instance.__dict__[self.attr] = value

        if instance.on_update is not None:
            instance.on_update(instance)


class ContentRange:
    """Represents the content range header.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        units: str | None,
        start: int | None,
        stop: int | None,
        length: int | None = None,
        on_update: cabc.Callable[[ContentRange], None] | None = None,
    ) -> None:
        self.on_update = on_update
        self.set(start, stop, length, units)

    #: The units to use, usually "bytes"
    units: str | None = _CallbackProperty()  # type: ignore[assignment]
    #: The start point of the range or `None`.
    start: int | None = _CallbackProperty()  # type: ignore[assignment]
    #: The stop point of the range (non-inclusive) or `None`.  Can only be
    #: `None` if also start is `None`.
    stop: int | None = _CallbackProperty()  # type: ignore[assignment]
    #: The length of the range or `None`.
    length: int | None = _CallbackProperty()  # type: ignore[assignment]

    def set(
        self,
        start: int | None,
        stop: int | None,
        length: int | None = None,
        units: str | None = "bytes",
    ) -> None:
        """Simple method to update the ranges."""
        assert http.is_byte_range_valid(start, stop, length), "Bad range provided"
        self._units: str | None = units
        self._start: int | None = start
        self._stop: int | None = stop
        self._length: int | None = length
        if self.on_update is not None:
            self.on_update(self)

    def unset(self) -> None:
        """Sets the units to `None` which indicates that the header should
        no longer be used.
        """
        self.set(None, None, units=None)

    def to_header(self) -> str:
        if self._units is None:
            return ""
        if self._length is None:
            length: str | int = "*"
        else:
            length = self._length
        if self._start is None:
            return f"{self._units} */{length}"
        return f"{self._units} {self._start}-{self._stop - 1}/{length}"  # type: ignore[operator]

    def __bool__(self) -> bool:
        return self._units is not None

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


# circular dependencies
from .. import http  # noqa: E402
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t
from copy import deepcopy

from .. import exceptions
from .._internal import _missing
from .mixins import ImmutableDictMixin
from .mixins import ImmutableListMixin
from .mixins import ImmutableMultiDictMixin
from .mixins import UpdateDictMixin

if t.TYPE_CHECKING:
    import typing_extensions as te

K = t.TypeVar("K")
V = t.TypeVar("V")
T = t.TypeVar("T")


def iter_multi_items(
    mapping: (
        MultiDict[K, V]
        | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
        | cabc.Iterable[tuple[K, V]]
    ),
) -> cabc.Iterator[tuple[K, V]]:
    """Iterates over the items of a mapping yielding keys and values
    without dropping any from more complex structures.
    """
    if isinstance(mapping, MultiDict):
        yield from mapping.items(multi=True)
    elif isinstance(mapping, cabc.Mapping):
        for key, value in mapping.items():
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    yield key, v
            else:
                yield key, value
    else:
        yield from mapping


class ImmutableList(ImmutableListMixin, list[V]):  # type: ignore[misc]
    """An immutable :class:`list`.

    .. versionadded:: 0.5

    :private:
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list.__repr__(self)})"


class TypeConversionDict(dict[K, V]):
    """Works like a regular dict but the :meth:`get` method can perform
    type conversions.  :class:`MultiDict` and :class:`CombinedMultiDict`
    are subclasses of this class and provide the same feature.

    .. versionadded:: 0.5
    """

    @t.overload  # type: ignore[override]
    def get(self, key: K) -> V | None: ...
    @t.overload
    def get(self, key: K, default: V) -> V: ...
    @t.overload
    def get(self, key: K, default: T) -> V | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[V], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[V], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: K,
        default: V | T | None = None,
        type: cabc.Callable[[V], T] | None = None,
    ) -> V | T | None:
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = TypeConversionDict(foo='42', bar='blub')
        >>> d.get('foo', type=int)
        42
        >>> d.get('bar', -1, type=int)
        -1

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`MultiDict`.  If a :exc:`ValueError` or a
                     :exc:`TypeError` is raised by this callable the default
                     value is returned.

        .. versionchanged:: 3.0.2
           Returns the default value on :exc:`TypeError`, too.
        """
        try:
            rv = self[key]
        except KeyError:
            return default

        if type is None:
            return rv

        try:
            return type(rv)
        except (ValueError, TypeError):
            return default


class ImmutableTypeConversionDict(ImmutableDictMixin[K, V], TypeConversionDict[K, V]):  # type: ignore[misc]
    """Works like a :class:`TypeConversionDict` but does not support
    modifications.

    .. versionadded:: 0.5
    """

    def copy(self) -> TypeConversionDict[K, V]:
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return TypeConversionDict(self)

    def __copy__(self) -> te.Self:
        return self


class MultiDict(TypeConversionDict[K, V]):
    """A :class:`MultiDict` is a dictionary subclass customized to deal with
    multiple values for the same key which is for example used by the parsing
    functions in the wrappers.  This is necessary because some HTML form
    elements pass multiple values for the same key.

    :class:`MultiDict` implements all standard dictionary methods.
    Internally, it saves all values for a key as a list, but the standard dict
    access methods will only return the first value for a key. If you want to
    gain access to the other values, too, you have to use the `list` methods as
    explained below.

    Basic Usage:

    >>> d = MultiDict([('a', 'b'), ('a', 'c')])
    >>> d
    MultiDict([('a', 'b'), ('a', 'c')])
    >>> d['a']
    'b'
    >>> d.getlist('a')
    ['b', 'c']
    >>> 'a' in d
    True

    It behaves like a normal dict thus all dict functions will only return the
    first value when multiple values for one key are found.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.

    A :class:`MultiDict` can be constructed from an iterable of
    ``(key, value)`` tuples, a dict, a :class:`MultiDict` or from Werkzeug 0.2
    onwards some keyword parameters.

    :param mapping: the initial value for the :class:`MultiDict`.  Either a
                    regular dict, an iterable of ``(key, value)`` tuples
                    or `None`.

    .. versionchanged:: 3.1
        Implement ``|`` and ``|=`` operators.
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        if mapping is None:
            super().__init__()
        elif isinstance(mapping, MultiDict):
            super().__init__((k, vs[:]) for k, vs in mapping.lists())  # type: ignore[misc]
        elif isinstance(mapping, cabc.Mapping):
            tmp = {}
            for key, value in mapping.items():
                if isinstance(value, (list, tuple, set)):
                    value = list(value)

                    if not value:
                        continue
                else:
                    value = [value]
                tmp[key] = value
            super().__init__(tmp)  # type: ignore[arg-type]
        else:
            tmp = {}
            for key, value in mapping:
                tmp.setdefault(key, []).append(value)
            super().__init__(tmp)  # type: ignore[arg-type]

    def __getstate__(self) -> t.Any:
        return dict(self.lists())

    def __setstate__(self, value: t.Any) -> None:
        super().clear()
        super().update(value)

    def __iter__(self) -> cabc.Iterator[K]:
        # https://github.com/python/cpython/issues/87412
        # If __iter__ is not overridden, Python uses a fast path for dict(md),
        # taking the data directly and getting lists of values, rather than
        # calling __getitem__ and getting only the first value.
        return super().__iter__()

    def __getitem__(self, key: K) -> V:
        """Return the first data value for this key;
        raises KeyError if not found.

        :param key: The key to be looked up.
        :raise KeyError: if the key does not exist.
        """

        if key in self:
            lst = super().__getitem__(key)
            if len(lst) > 0:  # type: ignore[arg-type]
                return lst[0]  # type: ignore[index,no-any-return]
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        """Like :meth:`add` but removes an existing key first.

        :param key: the key for the value.
        :param value: the value to set.
        """
        super().__setitem__(key, [value])  # type: ignore[assignment]

    def add(self, key: K, value: V) -> None:
        """Adds a new value for the key.

        .. versionadded:: 0.6

        :param key: the key for the value.
        :param value: the value to add.
        """
        super().setdefault(key, []).append(value)  # type: ignore[arg-type,attr-defined]

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        """Return the list of items for a given key. If that key is not in the
        `MultiDict`, the return value will be an empty list.  Just like `get`,
        `getlist` accepts a `type` parameter.  All items will be converted
        with the callable defined there.

        :param key: The key to be looked up.
        :param type: Callable to convert each value. If a ``ValueError`` or
            ``TypeError`` is raised, the value is omitted.
        :return: a :class:`list` of all the values for the key.

        .. versionchanged:: 3.1
            Catches ``TypeError`` in addition to ``ValueError``.
        """
        try:
            rv: list[V] = super().__getitem__(key)  # type: ignore[assignment]
        except KeyError:
            return []
        if type is None:
            return list(rv)
        result = []
        for item in rv:
            try:
                result.append(type(item))
            except (ValueError, TypeError):
                pass
        return result

    def setlist(self, key: K, new_list: cabc.Iterable[V]) -> None:
        """Remove the old values for a key and add new ones.  Note that the list
        you pass the values in will be shallow-copied before it is inserted in
        the dictionary.

        >>> d = MultiDict()
        >>> d.setlist('foo', ['1', '2'])
        >>> d['foo']
        '1'
        >>> d.getlist('foo')
        ['1', '2']

        :param key: The key for which the values are set.
        :param new_list: An iterable with the new values for the key.  Old values
                         are removed first.
        """
        super().__setitem__(key, list(new_list))  # type: ignore[assignment]

    @t.overload
    def setdefault(self, key: K) -> None: ...
    @t.overload
    def setdefault(self, key: K, default: V) -> V: ...
    def setdefault(self, key: K, default: V | None = None) -> V | None:
        """Returns the value for the key if it is in the dict, otherwise it
        returns `default` and sets that value for `key`.

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key is not
                        in the dict.  If not further specified it's `None`.
        """
        if key not in self:
            self[key] = default  # type: ignore[assignment]

        return self[key]

    def setlistdefault(
        self, key: K, default_list: cabc.Iterable[V] | None = None
    ) -> list[V]:
        """Like `setdefault` but sets multiple values.  The list returned
        is not a copy, but the list that is actually used internally.  This
        means that you can put new values into the dict by appending items
        to the list:

        >>> d = MultiDict({"foo": 1})
        >>> d.setlistdefault("foo").extend([2, 3])
        >>> d.getlist("foo")
        [1, 2, 3]

        :param key: The key to be looked up.
        :param default_list: An iterable of default values.  It is either copied
                             (in case it was a list) or converted into a list
                             before returned.
        :return: a :class:`list`
        """
        if key not in self:
            super().__setitem__(key, list(default_list or ()))  # type: ignore[assignment]

        return super().__getitem__(key)  # type: ignore[return-value]

    def items(self, multi: bool = False) -> cabc.Iterable[tuple[K, V]]:  # type: ignore[override]
        """Return an iterator of ``(key, value)`` pairs.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the first value of each key.
        """
        values: list[V]

        for key, values in super().items():  # type: ignore[assignment]
            if multi:
                for value in values:
                    yield key, value
            else:
                yield key, values[0]

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        """Return a iterator of ``(key, values)`` pairs, where values is the list
        of all values associated with the key."""
        values: list[V]

        for key, values in super().items():  # type: ignore[assignment]
            yield key, list(values)

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        """Returns an iterator of the first value on every key's value list."""
        values: list[V]

        for values in super().values():  # type: ignore[assignment]
            yield values[0]

    def listvalues(self) -> cabc.Iterable[list[V]]:
        """Return an iterator of all values associated with a key.  Zipping
        :meth:`keys` and this is the same as calling :meth:`lists`:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> zip(d.keys(), d.listvalues()) == d.lists()
        True
        """
        return super().values()  # type: ignore[return-value]

    def copy(self) -> te.Self:
        """Return a shallow copy of this object."""
        return self.__class__(self)

    def deepcopy(self, memo: t.Any = None) -> te.Self:
        """Return a deep copy of this object."""
        return self.__class__(deepcopy(self.to_dict(flat=False), memo))

    @t.overload
    def to_dict(self, flat: t.Literal[True] = ...) -> dict[K, V]: ...
    @t.overload
    def to_dict(self, flat: t.Literal[False]) -> dict[K, list[V]]: ...
    def to_dict(self, flat: bool = True) -> dict[K, V] | dict[K, list[V]]:
        """Return the contents as regular dict.  If `flat` is `True` the
        returned dict will only have the first item present, if `flat` is
        `False` all values will be returned as lists.

        :param flat: If set to `False` the dict returned will have lists
                     with all the values in it.  Otherwise it will only
                     contain the first value for each key.
        :return: a :class:`dict`
        """
        if flat:
            return dict(self.items())
        return dict(self.lists())

    def update(  # type: ignore[override]
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> None:
        """update() extends rather than replaces existing key lists:

        >>> a = MultiDict({'x': 1})
        >>> b = MultiDict({'x': 2, 'y': 3})
        >>> a.update(b)
        >>> a
        MultiDict([('y', 3), ('x', 1), ('x', 2)])

        If the value list for a key in ``other_dict`` is empty, no new values
        will be added to the dict and the key will not be created:

        >>> x = {'empty_list': []}
        >>> y = MultiDict()
        >>> y.update(x)
        >>> y
        MultiDict([])
        """
        for key, value in iter_multi_items(mapping):
            self.add(key, value)

    def __or__(  # type: ignore[override]
        self, other: cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
    ) -> MultiDict[K, V]:
        if not isinstance(other, cabc.Mapping):
            return NotImplemented

        rv = self.copy()
        rv.update(other)
        return rv

    def __ior__(  # type: ignore[override]
        self,
        other: (
            cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> te.Self:
        if not isinstance(other, (cabc.Mapping, cabc.Iterable)):
            return NotImplemented

        self.update(other)
        return self

    @t.overload
    def pop(self, key: K) -> V: ...
    @t.overload
    def pop(self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self, key: K, default: T) -> V | T: ...
    def pop(
        self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        """Pop the first item for a list on the dict.  Afterwards the
        key is removed from the dict, so additional values are discarded:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> d.pop("foo")
        1
        >>> "foo" in d
        False

        :param key: the key to pop.
        :param default: if provided the value to return if the key was
                        not in the dictionary.
        """
        lst: list[V]

        try:
            lst = super().pop(key)  # type: ignore[assignment]

            if len(lst) == 0:
                raise exceptions.BadRequestKeyError(key)

            return lst[0]
        except KeyError:
            if default is not _missing:
                return default

            raise exceptions.BadRequestKeyError(key) from None

    def popitem(self) -> tuple[K, V]:
        """Pop an item from the dict."""
        item: tuple[K, list[V]]

        try:
            item = super().popitem()  # type: ignore[assignment]

            if len(item[1]) == 0:
                raise exceptions.BadRequestKeyError(item[0])

            return item[0], item[1][0]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

    def poplist(self, key: K) -> list[V]:
        """Pop the list for a key from the dict.  If the key is not in the dict
        an empty list is returned.

        .. versionchanged:: 0.5
           If the key does no longer exist a list is returned instead of
           raising an error.
        """
        return super().pop(key, [])  # type: ignore[return-value]

    def popitemlist(self) -> tuple[K, list[V]]:
        """Pop a ``(key, list)`` tuple from the dict."""
        try:
            return super().popitem()  # type: ignore[return-value]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

    def __copy__(self) -> te.Self:
        return self.copy()

    def __deepcopy__(self, memo: t.Any) -> te.Self:
        return self.deepcopy(memo=memo)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.items(multi=True))!r})"


class _omd_bucket(t.Generic[K, V]):
    """Wraps values in the :class:`OrderedMultiDict`.  This makes it
    possible to keep an order over multiple different keys.  It requires
    a lot of extra memory and slows down access a lot, but makes it
    possible to access elements in O(1) and iterate in O(n).
    """

    __slots__ = ("prev", "key", "value", "next")

    def __init__(self, omd: _OrderedMultiDict[K, V], key: K, value: V) -> None:
        self.prev: _omd_bucket[K, V] | None = omd._last_bucket
        self.key: K = key
        self.value: V = value
        self.next: _omd_bucket[K, V] | None = None

        if omd._first_bucket is None:
            omd._first_bucket = self
        if omd._last_bucket is not None:
            omd._last_bucket.next = self
        omd._last_bucket = self

    def unlink(self, omd: _OrderedMultiDict[K, V]) -> None:
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        if omd._first_bucket is self:
            omd._first_bucket = self.next
        if omd._last_bucket is self:
            omd._last_bucket = self.prev


class _OrderedMultiDict(MultiDict[K, V]):
    """Works like a regular :class:`MultiDict` but preserves the
    order of the fields.  To convert the ordered multi dict into a
    list you can use the :meth:`items` method and pass it ``multi=True``.

    In general an :class:`OrderedMultiDict` is an order of magnitude
    slower than a :class:`MultiDict`.

    .. admonition:: note

       Due to a limitation in Python you cannot convert an ordered
       multi dict into a regular dict by using ``dict(multidict)``.
       Instead you have to use the :meth:`to_dict` method, otherwise
       the internal bucket objects are exposed.

    .. deprecated:: 3.1
        Will be removed in Werkzeug 3.2. Use ``MultiDict`` instead.
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        import warnings

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._first_bucket: _omd_bucket[K, V] | None = None
        self._last_bucket: _omd_bucket[K, V] | None = None
        if mapping is not None:
            self.update(mapping)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDict):
            return NotImplemented
        if isinstance(other, _OrderedMultiDict):
            iter1 = iter(self.items(multi=True))
            iter2 = iter(other.items(multi=True))
            try:
                for k1, v1 in iter1:
                    k2, v2 = next(iter2)
                    if k1 != k2 or v1 != v2:
                        return False
            except StopIteration:
                return False
            try:
                next(iter2)
            except StopIteration:
                return True
            return False
        if len(self) != len(other):
            return False
        for key, values in self.lists():
            if other.getlist(key) != values:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self.items(multi=True)),)

    def __getstate__(self) -> t.Any:
        return list(self.items(multi=True))

    def __setstate__(self, values: t.Any) -> None:
        self.clear()

        for key, value in values:
            self.add(key, value)

    def __getitem__(self, key: K) -> V:
        if key in self:
            return dict.__getitem__(self, key)[0].value  # type: ignore[index,no-any-return]
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.poplist(key)
        self.add(key, value)

    def __delitem__(self, key: K) -> None:
        self.pop(key)

    def keys(self) -> cabc.Iterable[K]:  # type: ignore[override]
        return (key for key, _ in self.items())

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self.keys())

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        return (value for key, value in self.items())

    def items(self, multi: bool = False) -> cabc.Iterable[tuple[K, V]]:  # type: ignore[override]
        ptr = self._first_bucket
        if multi:
            while ptr is not None:
                yield ptr.key, ptr.value
                ptr = ptr.next
        else:
            returned_keys = set()
            while ptr is not None:
                if ptr.key not in returned_keys:
                    returned_keys.add(ptr.key)
                    yield ptr.key, ptr.value
                ptr = ptr.next

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        returned_keys = set()
        ptr = self._first_bucket
        while ptr is not None:
            if ptr.key not in returned_keys:
                yield ptr.key, self.getlist(ptr.key)
                returned_keys.add(ptr.key)
            ptr = ptr.next

    def listvalues(self) -> cabc.Iterable[list[V]]:
        for _key, values in self.lists():
            yield values

    def add(self, key: K, value: V) -> None:
        dict.setdefault(self, key, []).append(_omd_bucket(self, key, value))  # type: ignore[misc]

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        rv: list[_omd_bucket[K, V]]

        try:
            rv = dict.__getitem__(self, key)  # type: ignore[index]
        except KeyError:
            return []
        if type is None:
            return [x.value for x in rv]
        result = []
        for item in rv:
            try:
                result.append(type(item.value))
            except (ValueError, TypeError):
                pass
        return result

    def setlist(self, key: K, new_list: cabc.Iterable[V]) -> None:
        self.poplist(key)
        for value in new_list:
            self.add(key, value)

    def setlistdefault(self, key: t.Any, default_list: t.Any = None) -> t.NoReturn:
        raise TypeError("setlistdefault is unsupported for ordered multi dicts")

    def update(  # type: ignore[override]
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> None:
        for key, value in iter_multi_items(mapping):
            self.add(key, value)

    def poplist(self, key: K) -> list[V]:
        buckets: cabc.Iterable[_omd_bucket[K, V]] = dict.pop(self, key, ())  # type: ignore[arg-type]
        for bucket in buckets:
            bucket.unlink(self)
        return [x.value for x in buckets]

    @t.overload
    def pop(self, key: K) -> V: ...
    @t.overload
    def pop(self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self, key: K, default: T) -> V | T: ...
    def pop(
        self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        buckets: list[_omd_bucket[K, V]]

        try:
            buckets = dict.pop(self, key)  # type: ignore[arg-type]
        except KeyError:
            if default is not _missing:
                return default

            raise exceptions.BadRequestKeyError(key) from None

        for bucket in buckets:
            bucket.unlink(self)

        return buckets[0].value

    def popitem(self) -> tuple[K, V]:
        key: K
        buckets: list[_omd_bucket[K, V]]

        try:
            key, buckets = dict.popitem(self)  # type: ignore[arg-type]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

        for bucket in buckets:
            bucket.unlink(self)

        return key, buckets[0].value

    def popitemlist(self) -> tuple[K, list[V]]:
        key: K
        buckets: list[_omd_bucket[K, V]]

        try:
            key, buckets = dict.popitem(self)  # type: ignore[arg-type]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

        for bucket in buckets:
            bucket.unlink(self)

        return key, [x.value for x in buckets]


class CombinedMultiDict(ImmutableMultiDictMixin[K, V], MultiDict[K, V]):  # type: ignore[misc]
    """A read only :class:`MultiDict` that you can pass multiple :class:`MultiDict`
    instances as sequence and it will combine the return values of all wrapped
    dicts:

    >>> from werkzeug.datastructures import CombinedMultiDict, MultiDict
    >>> post = MultiDict([('foo', 'bar')])
    >>> get = MultiDict([('blub', 'blah')])
    >>> combined = CombinedMultiDict([get, post])
    >>> combined['foo']
    'bar'
    >>> combined['blub']
    'blah'

    This works for all read operations and will raise a `TypeError` for
    methods that usually change data which isn't possible.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.
    """

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (self.dicts,)

    def __init__(self, dicts: cabc.Iterable[MultiDict[K, V]] | None = None) -> None:
        super().__init__()
        self.dicts: list[MultiDict[K, V]] = list(dicts or ())

    @classmethod
    def fromkeys(cls, keys: t.Any, value: t.Any = None) -> t.NoReturn:
        raise TypeError(f"cannot create {cls.__name__!r} instances by fromkeys")

    def __getitem__(self, key: K) -> V:
        for d in self.dicts:
            if key in d:
                return d[key]
        raise exceptions.BadRequestKeyError(key)

    @t.overload  # type: ignore[override]
    def get(self, key: K) -> V | None: ...
    @t.overload
    def get(self, key: K, default: V) -> V: ...
    @t.overload
    def get(self, key: K, default: T) -> V | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[V], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[V], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: K,
        default: V | T | None = None,
        type: cabc.Callable[[V], T] | None = None,
    ) -> V | T | None:
        for d in self.dicts:
            if key in d:
                if type is not None:
                    try:
                        return type(d[key])
                    except (ValueError, TypeError):
                        continue
                return d[key]
        return default

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        rv = []
        for d in self.dicts:
            rv.extend(d.getlist(key, type))  # type: ignore[arg-type]
        return rv

    def _keys_impl(self) -> set[K]:
        """This function exists so __len__ can be implemented more efficiently,
        saving one list creation from an iterator.
        """
        return set(k for d in self.dicts for k in d)

    def keys(self) -> cabc.Iterable[K]:  # type: ignore[override]
        return self._keys_impl()

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self._keys_impl())

    @t.overload  # type: ignore[override]
    def items(self) -> cabc.Iterable[tuple[K, V]]: ...
    @t.overload
    def items(self, multi: t.Literal[True]) -> cabc.Iterable[tuple[K, list[V]]]: ...
    def items(
        self, multi: bool = False
    ) -> cabc.Iterable[tuple[K, V]] | cabc.Iterable[tuple[K, list[V]]]:
        found = set()
        for d in self.dicts:
            for key, value in d.items(multi):
                if multi:
                    yield key, value
                elif key not in found:
                    found.add(key)
                    yield key, value

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        for _, value in self.items():
            yield value

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        rv: dict[K, list[V]] = {}
        for d in self.dicts:
            for key, values in d.lists():
                rv.setdefault(key, []).extend(values)
        return rv.items()

    def listvalues(self) -> cabc.Iterable[list[V]]:
        return (x[1] for x in self.lists())

    def copy(self) -> MultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.

        This returns a :class:`MultiDict` representing the data at the
        time of copying. The copy will no longer reflect changes to the
        wrapped dicts.

        .. versionchanged:: 0.15
            Return a mutable :class:`MultiDict`.
        """
        return MultiDict(self)

    def __len__(self) -> int:
        return len(self._keys_impl())

    def __contains__(self, key: K) -> bool:  # type: ignore[override]
        for d in self.dicts:
            if key in d:
                return True
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dicts!r})"


class ImmutableDict(ImmutableDictMixin[K, V], dict[K, V]):  # type: ignore[misc]
    """An immutable :class:`dict`.

    .. versionadded:: 0.5
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"

    def copy(self) -> dict[K, V]:
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return dict(self)

    def __copy__(self) -> te.Self:
        return self


class ImmutableMultiDict(ImmutableMultiDictMixin[K, V], MultiDict[K, V]):  # type: ignore[misc]
    """An immutable :class:`MultiDict`.

    .. versionadded:: 0.5
    """

    def copy(self) -> MultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return MultiDict(self)

    def __copy__(self) -> te.Self:
        return self


class _ImmutableOrderedMultiDict(  # type: ignore[misc]
    ImmutableMultiDictMixin[K, V], _OrderedMultiDict[K, V]
):
    """An immutable :class:`OrderedMultiDict`.

    .. deprecated:: 3.1
        Will be removed in Werkzeug 3.2. Use ``ImmutableMultiDict`` instead.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        super().__init__()

        if mapping is not None:
            for k, v in iter_multi_items(mapping):
                _OrderedMultiDict.add(self, k, v)

    def _iter_hashitems(self) -> cabc.Iterable[t.Any]:
        return enumerate(self.items(multi=True))

    def copy(self) -> _OrderedMultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return _OrderedMultiDict(self)

    def __copy__(self) -> te.Self:
        return self


class CallbackDict(UpdateDictMixin[K, V], dict[K, V]):
    """A dict that calls a function passed every time something is changed.
    The function is passed the dict instance.
    """

    def __init__(
        self,
        initial: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]] | None = None,
        on_update: cabc.Callable[[te.Self], None] | None = None,
    ) -> None:
        if initial is None:
            super().__init__()
        else:
            super().__init__(initial)

        self.on_update = on_update

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {super().__repr__()}>"


class HeaderSet(cabc.MutableSet[str]):
    """Similar to the :class:`ETags` class this implements a set-like structure.
    Unlike :class:`ETags` this is case insensitive and used for vary, allow, and
    content-language headers.

    If not constructed using the :func:`parse_set_header` function the
    instantiation works like this:

    >>> hs = HeaderSet(['foo', 'bar', 'baz'])
    >>> hs
    HeaderSet(['foo', 'bar', 'baz'])
    """

    def __init__(
        self,
        headers: cabc.Iterable[str] | None = None,
        on_update: cabc.Callable[[te.Self], None] | None = None,
    ) -> None:
        self._headers = list(headers or ())
        self._set = {x.lower() for x in self._headers}
        self.on_update = on_update

    def add(self, header: str) -> None:
        """Add a new header to the set."""
        self.update((header,))

    def remove(self: te.Self, header: str) -> None:
        """Remove a header from the set.  This raises an :exc:`KeyError` if the
        header is not in the set.

        .. versionchanged:: 0.5
            In older versions a :exc:`IndexError` was raised instead of a
            :exc:`KeyError` if the object was missing.

        :param header: the header to be removed.
        """
        key = header.lower()
        if key not in self._set:
            raise KeyError(header)
        self._set.remove(key)
        for idx, key in enumerate(self._headers):
            if key.lower() == header:
                del self._headers[idx]
                break
        if self.on_update is not None:
            self.on_update(self)

    def update(self: te.Self, iterable: cabc.Iterable[str]) -> None:
        """Add all the headers from the iterable to the set.

        :param iterable: updates the set with the items from the iterable.
        """
        inserted_any = False
        for header in iterable:
            key = header.lower()
            if key not in self._set:
                self._headers.append(header)
                self._set.add(key)
                inserted_any = True
        if inserted_any and self.on_update is not None:
            self.on_update(self)

    def discard(self, header: str) -> None:
        """Like :meth:`remove` but ignores errors.

        :param header: the header to be discarded.
        """
        try:
            self.remove(header)
        except KeyError:
            pass

    def find(self, header: str) -> int:
        """Return the index of the header in the set or return -1 if not found.

        :param header: the header to be looked up.
        """
        header = header.lower()
        for idx, item in enumerate(self._headers):
            if item.lower() == header:
                return idx
        return -1

    def index(self, header: str) -> int:
        """Return the index of the header in the set or raise an
        :exc:`IndexError`.

        :param header: the header to be looked up.
        """
        rv = self.find(header)
        if rv < 0:
            raise IndexError(header)
        return rv

    def clear(self: te.Self) -> None:
        """Clear the set."""
        self._set.clear()
        self._headers.clear()

        if self.on_update is not None:
            self.on_update(self)

    def as_set(self, preserve_casing: bool = False) -> set[str]:
        """Return the set as real python set type.  When calling this, all
        the items are converted to lowercase and the ordering is lost.

        :param preserve_casing: if set to `True` the items in the set returned
                                will have the original case like in the
                                :class:`HeaderSet`, otherwise they will
                                be lowercase.
        """
        if preserve_casing:
            return set(self._headers)
        return set(self._set)

    def to_header(self) -> str:
        """Convert the header set into an HTTP header string."""
        return ", ".join(map(http.quote_header_value, self._headers))

    def __getitem__(self, idx: t.SupportsIndex) -> str:
        return self._headers[idx]

    def __delitem__(self: te.Self, idx: t.SupportsIndex) -> None:
        rv = self._headers.pop(idx)
        self._set.remove(rv.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __setitem__(self: te.Self, idx: t.SupportsIndex, value: str) -> None:
        old = self._headers[idx]
        self._set.remove(old.lower())
        self._headers[idx] = value
        self._set.add(value.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __contains__(self, header: str) -> bool:  # type: ignore[override]
        return header.lower() in self._set

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._headers)

    def __bool__(self) -> bool:
        return bool(self._set)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._headers!r})"


# circular dependencies
from .. import http  # noqa: E402


def __getattr__(name: str) -> t.Any:
    import warnings

    if name == "OrderedMultiDict":
        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _OrderedMultiDict

    if name == "ImmutableOrderedMultiDict":
        warnings.warn(
            "'ImmutableOrderedMultiDict' is deprecated and will be removed in"
            " Werkzeug 3.2. Use 'ImmutableMultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ImmutableOrderedMultiDict

    raise AttributeError(name)
<|endoftext|>from __future__ import annotations

import typing as t

from .accept import Accept as Accept
from .accept import CharsetAccept as CharsetAccept
from .accept import LanguageAccept as LanguageAccept
from .accept import MIMEAccept as MIMEAccept
from .auth import Authorization as Authorization
from .auth import WWWAuthenticate as WWWAuthenticate
from .cache_control import RequestCacheControl as RequestCacheControl
from .cache_control import ResponseCacheControl as ResponseCacheControl
from .csp import ContentSecurityPolicy as ContentSecurityPolicy
from .etag import ETags as ETags
from .file_storage import FileMultiDict as FileMultiDict
from .file_storage import FileStorage as FileStorage
from .headers import EnvironHeaders as EnvironHeaders
from .headers import Headers as Headers
from .mixins import ImmutableDictMixin as ImmutableDictMixin
from .mixins import ImmutableHeadersMixin as ImmutableHeadersMixin
from .mixins import ImmutableListMixin as ImmutableListMixin
from .mixins import ImmutableMultiDictMixin as ImmutableMultiDictMixin
from .mixins import UpdateDictMixin as UpdateDictMixin
from .range import ContentRange as ContentRange
from .range import IfRange as IfRange
from .range import Range as Range
from .structures import CallbackDict as CallbackDict
from .structures import CombinedMultiDict as CombinedMultiDict
from .structures import HeaderSet as HeaderSet
from .structures import ImmutableDict as ImmutableDict
from .structures import ImmutableList as ImmutableList
from .structures import ImmutableMultiDict as ImmutableMultiDict
from .structures import ImmutableTypeConversionDict as ImmutableTypeConversionDict
from .structures import iter_multi_items as iter_multi_items
from .structures import MultiDict as MultiDict
from .structures import TypeConversionDict as TypeConversionDict


def __getattr__(name: str) -> t.Any:
    import warnings

    if name == "OrderedMultiDict":
        from .structures import _OrderedMultiDict

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _OrderedMultiDict

    if name == "ImmutableOrderedMultiDict":
        from .structures import _ImmutableOrderedMultiDict

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'ImmutableMultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ImmutableOrderedMultiDict

    raise AttributeError(name)
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import mimetypes
import os
import typing as t
from io import BytesIO
from os import fsdecode
from os import fspath

from .._internal import _plain_int
from .headers import Headers
from .structures import MultiDict


class FileStorage:
    """The :class:`FileStorage` class is a thin wrapper over incoming files.
    It is used by the request object to represent uploaded files.  All the
    attributes of the wrapper stream are proxied by the file storage so
    it's possible to do ``storage.read()`` instead of the long form
    ``storage.stream.read()``.
    """

    def __init__(
        self,
        stream: t.IO[bytes] | None = None,
        filename: str | None = None,
        name: str | None = None,
        content_type: str | None = None,
        content_length: int | None = None,
        headers: Headers | None = None,
    ):
        self.name = name
        self.stream = stream or BytesIO()

        # If no filename is provided, attempt to get the filename from
        # the stream object. Python names special streams like
        # ``<stderr>`` with angular brackets, skip these streams.
        if filename is None:
            filename = getattr(stream, "name", None)

            if filename is not None:
                filename = fsdecode(filename)

            if filename and filename.startswith("<") and filename.endswith(">"):
                filename = None
        else:
            filename = fsdecode(filename)

        self.filename = filename

        if headers is None:
            headers = Headers()
        self.headers = headers
        if content_type is not None:
            headers["Content-Type"] = content_type
        if content_length is not None:
            headers["Content-Length"] = str(content_length)

    def _parse_content_type(self) -> None:
        if not hasattr(self, "_parsed_content_type"):
            self._parsed_content_type = http.parse_options_header(self.content_type)

    @property
    def content_type(self) -> str | None:
        """The content-type sent in the header.  Usually not available"""
        return self.headers.get("content-type")

    @property
    def content_length(self) -> int:
        """The content-length sent in the header.  Usually not available"""
        if "content-length" in self.headers:
            try:
                return _plain_int(self.headers["content-length"])
            except ValueError:
                pass

        return 0

    @property
    def mimetype(self) -> str:
        """Like :attr:`content_type`, but without parameters (eg, without
        charset, type etc.) and always lowercase.  For example if the content
        type is ``text/HTML; charset=utf-8`` the mimetype would be
        ``'text/html'``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[0].lower()

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict.  For example if the content
        type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[1]

    def save(
        self, dst: str | os.PathLike[str] | t.IO[bytes], buffer_size: int = 16384
    ) -> None:
        """Save the file to a destination path or file object.  If the
        destination is a file object you have to close it yourself after the
        call.  The buffer size is the number of bytes held in memory during
        the copy process.  It defaults to 16KB.

        For secure file saving also have a look at :func:`secure_filename`.

        :param dst: a filename, :class:`os.PathLike`, or open file
            object to write to.
        :param buffer_size: Passed as the ``length`` parameter of
            :func:`shutil.copyfileobj`.

        .. versionchanged:: 1.0
            Supports :mod:`pathlib`.
        """
        from shutil import copyfileobj

        close_dst = False

        if hasattr(dst, "__fspath__"):
            dst = fspath(dst)

        if isinstance(dst, str):
            dst = open(dst, "wb")
            close_dst = True

        try:
            copyfileobj(self.stream, dst, buffer_size)
        finally:
            if close_dst:
                dst.close()

    def close(self) -> None:
        """Close the underlying file if possible."""
        try:
            self.stream.close()
        except Exception:
            pass

    def __bool__(self) -> bool:
        return bool(self.filename)

    def __getattr__(self, name: str) -> t.Any:
        try:
            return getattr(self.stream, name)
        except AttributeError:
            # SpooledTemporaryFile on Python < 3.11 doesn't implement IOBase,
            # get the attribute from its backing file instead.
            if hasattr(self.stream, "_file"):
                return getattr(self.stream._file, name)
            raise

    def __iter__(self) -> cabc.Iterator[bytes]:
        return iter(self.stream)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.filename!r} ({self.content_type!r})>"


class FileMultiDict(MultiDict[str, FileStorage]):
    """A special :class:`MultiDict` that has convenience methods to add
    files to it.  This is used for :class:`EnvironBuilder` and generally
    useful for unittesting.

    .. versionadded:: 0.5
    """

    def add_file(
        self,
        name: str,
        file: str | os.PathLike[str] | t.IO[bytes] | FileStorage,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Adds a new file to the dict.  `file` can be a file name or
        a :class:`file`-like or a :class:`FileStorage` object.

        :param name: the name of the field.
        :param file: a filename or :class:`file`-like object
        :param filename: an optional filename
        :param content_type: an optional content type
        """
        if isinstance(file, FileStorage):
            self.add(name, file)
            return

        if isinstance(file, (str, os.PathLike)):
            if filename is None:
                filename = os.fspath(file)

            file_obj: t.IO[bytes] = open(file, "rb")
        else:
            file_obj = file  # type: ignore[assignment]

        if filename and content_type is None:
            content_type = (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )

        self.add(name, FileStorage(file_obj, filename, name, content_type))


# circular dependencies
from .. import http  # noqa: E402
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import re
import typing as t

from .._internal import _missing
from ..exceptions import BadRequestKeyError
from .mixins import ImmutableHeadersMixin
from .structures import iter_multi_items
from .structures import MultiDict

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIEnvironment

T = t.TypeVar("T")


class Headers:
    """An object that stores some headers. It has a dict-like interface,
    but is ordered, can store the same key multiple times, and iterating
    yields ``(key, value)`` pairs instead of only keys.

    This data structure is useful if you want a nicer way to handle WSGI
    headers which are stored as tuples in a list.

    From Werkzeug 0.3 onwards, the :exc:`KeyError` raised by this class is
    also a subclass of the :class:`~exceptions.BadRequest` HTTP exception
    and will render a page for a ``400 BAD REQUEST`` if caught in a
    catch-all for HTTP exceptions.

    Headers is mostly compatible with the Python :class:`wsgiref.headers.Headers`
    class, with the exception of `__getitem__`.  :mod:`wsgiref` will return
    `None` for ``headers['missing']``, whereas :class:`Headers` will raise
    a :class:`KeyError`.

    To create a new ``Headers`` object, pass it a list, dict, or
    other ``Headers`` object with default values. These values are
    validated the same way values added later are.

    The header keys are assumed to be trusted and valid, they must not come
    from untrusted user input. Newlines in header values are not allowed and
    will raise a ``ValueError``.

    :param defaults: The list of default values for the :class:`Headers`.

    .. versionchanged:: 3.1
        Implement ``|`` and ``|=`` operators.

    .. versionchanged:: 2.1.0
        Default values are validated the same as values added later.

    .. versionchanged:: 0.9
       This data structure now stores unicode values similar to how the
       multi dicts do it.  The main difference is that bytes can be set as
       well which will automatically be latin1 decoded.

    .. versionchanged:: 0.9
       The :meth:`linked` function was removed without replacement as it
       was an API that does not support the changes to the encoding model.
    """

    def __init__(
        self,
        defaults: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
    ) -> None:
        self._list: list[tuple[str, str]] = []

        if defaults is not None:
            self.extend(defaults)

    @t.overload
    def __getitem__(self, key: str) -> str: ...
    @t.overload
    def __getitem__(self, key: int) -> tuple[str, str]: ...
    @t.overload
    def __getitem__(self, key: slice) -> te.Self: ...
    def __getitem__(self, key: str | int | slice) -> str | tuple[str, str] | te.Self:
        if isinstance(key, str):
            return self._get_key(key)

        if isinstance(key, int):
            return self._list[key]

        return self.__class__(self._list[key])

    def _get_key(self, key: str) -> str:
        ikey = key.lower()

        for k, v in self._list:
            if k.lower() == ikey:
                return v

        raise BadRequestKeyError(key)

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        def lowered(item: tuple[str, ...]) -> tuple[str, ...]:
            return item[0].lower(), *item[1:]

        return set(map(lowered, other._list)) == set(map(lowered, self._list))

    __hash__ = None  # type: ignore[assignment]

    @t.overload
    def get(self, key: str) -> str | None: ...
    @t.overload
    def get(self, key: str, default: str) -> str: ...
    @t.overload
    def get(self, key: str, default: T) -> str | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[str], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[str], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: str,
        default: str | T | None = None,
        type: cabc.Callable[[str], T] | None = None,
    ) -> str | T | None:
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = Headers([('Content-Length', '42')])
        >>> d.get('Content-Length', type=int)
        42

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the default value is returned.

        .. versionchanged:: 3.0
            The ``as_bytes`` parameter was removed.

        .. versionchanged:: 0.9
            The ``as_bytes`` parameter was added.
        """
        try:
            rv = self._get_key(key)
        except KeyError:
            return default

        if type is None:
            return rv

        try:
            return type(rv)
        except ValueError:
            return default

    @t.overload
    def getlist(self, key: str) -> list[str]: ...
    @t.overload
    def getlist(self, key: str, type: cabc.Callable[[str], T]) -> list[T]: ...
    def getlist(
        self, key: str, type: cabc.Callable[[str], T] | None = None
    ) -> list[str] | list[T]:
        """Return the list of items for a given key. If that key is not in the
        :class:`Headers`, the return value will be an empty list.  Just like
        :meth:`get`, :meth:`getlist` accepts a `type` parameter.  All items will
        be converted with the callable defined there.

        :param key: The key to be looked up.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the value will be removed from the list.
        :return: a :class:`list` of all the values for the key.

        .. versionchanged:: 3.0
            The ``as_bytes`` parameter was removed.

        .. versionchanged:: 0.9
            The ``as_bytes`` parameter was added.
        """
        ikey = key.lower()

        if type is not None:
            result = []

            for k, v in self:
                if k.lower() == ikey:
                    try:
                        result.append(type(v))
                    except ValueError:
                        continue

            return result

        return [v for k, v in self if k.lower() == ikey]

    def get_all(self, name: str) -> list[str]:
        """Return a list of all the values for the named field.

        This method is compatible with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.get_all` method.
        """
        return self.getlist(name)

    def items(self, lower: bool = False) -> t.Iterable[tuple[str, str]]:
        for key, value in self:
            if lower:
                key = key.lower()
            yield key, value

    def keys(self, lower: bool = False) -> t.Iterable[str]:
        for key, _ in self.items(lower):
            yield key

    def values(self) -> t.Iterable[str]:
        for _, value in self.items():
            yield value

    def extend(
        self,
        arg: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
        /,
        **kwargs: str,
    ) -> None:
        """Extend headers in this object with items from another object
        containing header items as well as keyword arguments.

        To replace existing keys instead of extending, use
        :meth:`update` instead.

        If provided, the first argument can be another :class:`Headers`
        object, a :class:`MultiDict`, :class:`dict`, or iterable of
        pairs.

        The header keys are assumed to be trusted and valid, they must not come
        from untrusted user input. Newlines in header values are not allowed and
        will raise a ``ValueError``.

        .. versionchanged:: 1.0
            Support :class:`MultiDict`. Allow passing ``kwargs``.
        """
        if arg is not None:
            for key, value in iter_multi_items(arg):
                self.add(key, value)

        for key, value in iter_multi_items(kwargs):
            self.add(key, value)

    def __delitem__(self, key: str | int | slice) -> None:
        if isinstance(key, str):
            self._del_key(key)
            return

        del self._list[key]

    def _del_key(self, key: str) -> None:
        key = key.lower()
        new = []

        for k, v in self._list:
            if k.lower() != key:
                new.append((k, v))

        self._list[:] = new

    def remove(self, key: str) -> None:
        """Remove a key.

        :param key: The key to be removed.
        """
        return self._del_key(key)

    @t.overload
    def pop(self) -> tuple[str, str]: ...
    @t.overload
    def pop(self, key: str) -> str: ...
    @t.overload
    def pop(self, key: int | None = ...) -> tuple[str, str]: ...
    @t.overload
    def pop(self, key: str, default: str) -> str: ...
    @t.overload
    def pop(self, key: str, default: T) -> str | T: ...
    def pop(
        self,
        key: str | int | None = None,
        default: str | T = _missing,  # type: ignore[assignment]
    ) -> str | tuple[str, str] | T:
        """Removes and returns a key or index.

        :param key: The key to be popped.  If this is an integer the item at
                    that position is removed, if it's a string the value for
                    that key is.  If the key is omitted or `None` the last
                    item is removed.
        :return: an item.
        """
        if key is None:
            return self._list.pop()

        if isinstance(key, int):
            return self._list.pop(key)

        try:
            rv = self._get_key(key)
        except KeyError:
            if default is not _missing:
                return default

            raise

        self.remove(key)
        return rv

    def popitem(self) -> tuple[str, str]:
        """Removes a key or index and returns a (key, value) item."""
        return self._list.pop()

    def __contains__(self, key: str) -> bool:
        """Check if a key is present."""
        try:
            self._get_key(key)
        except KeyError:
            return False

        return True

    def __iter__(self) -> t.Iterator[tuple[str, str]]:
        """Yield ``(key, value)`` tuples."""
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def add(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Add a new header tuple to the list.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes::

        >>> d = Headers()
        >>> d.add('Content-Type', 'text/plain')
        >>> d.add('Content-Disposition', 'attachment', filename='foo.png')

        The keyword argument dumping uses :func:`dump_options_header`
        behind the scenes.

        :param key: The header key. This is assumed to be trusted and valid, it
            must not come from untrusted user input.
        :param value: The header value. Newlines are not allowed and will raise
            a ``ValueError``.

        .. versionchanged:: 0.4.1
            keyword arguments were added for :mod:`wsgiref` compatibility.
        """
        if kwargs:
            value = _options_header_vkw(value, kwargs)

        value_str = _str_header_value(value)
        self._list.append((key, value_str))

    def add_header(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Add a new header tuple to the list.

        An alias for :meth:`add` for compatibility with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.add_header` method.
        """
        self.add(key, value, **kwargs)

    def clear(self) -> None:
        """Clears all headers."""
        self._list.clear()

    def set(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Remove all header tuples for `key` and add a new one.  The newly
        added key either appears at the end of the list if there was no
        entry or replaces the first one.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes.  See :meth:`add` for
        more information.

        .. versionchanged:: 0.6.1
           :meth:`set` now accepts the same arguments as :meth:`add`.

        :param key: The header key. This is assumed to be trusted and valid, it
            must not come from untrusted user input.
        :param value: The header value. Newlines are not allowed and will raise
            a ``ValueError``.
        """
        if kwargs:
            value = _options_header_vkw(value, kwargs)

        value_str = _str_header_value(value)

        if not self._list:
            self._list.append((key, value_str))
            return

        iter_list = iter(self._list)
        ikey = key.lower()

        for idx, (old_key, _) in enumerate(iter_list):
            if old_key.lower() == ikey:
                # replace first occurrence
                self._list[idx] = (key, value_str)
                break
        else:
            # no existing occurrences
            self._list.append((key, value_str))
            return

        # remove remaining occurrences
        self._list[idx + 1 :] = [t for t in iter_list if t[0].lower() != ikey]

    def setlist(self, key: str, values: cabc.Iterable[t.Any]) -> None:
        """Remove any existing values for a header and add new ones.

        :param key: The header key. This is assumed to be trusted and valid, it
            must not come from untrusted user input.
        :param value: An iterable of header values. Newlines are not allowed and
            will raise a ``ValueError``.

        .. versionadded:: 1.0
        """
        if values:
            values_iter = iter(values)
            self.set(key, next(values_iter))

            for value in values_iter:
                self.add(key, value)
        else:
            self.remove(key)

    def setdefault(self, key: str, default: t.Any) -> str:
        """Return the first value for the key if it is in the headers,
        otherwise set the header to the value given by ``default`` and
        return that.

        :param key: The header key. This is assumed to be trusted and valid, it
            must not come from untrusted user input.
        :param default: The header value to set if the key is not present.
            Newlines are not allowed and will raise a ``ValueError``.
        """
        try:
            return self._get_key(key)
        except KeyError:
            pass

        self.set(key, default)
        return self._get_key(key)

    def setlistdefault(self, key: str, default: cabc.Iterable[t.Any]) -> list[str]:
        """Return the list of values for the key if it is in the
        headers, otherwise set the header to the list of values given
        by ``default`` and return that.

        Unlike :meth:`MultiDict.setlistdefault`, modifying the returned
        list will not affect the headers.

        :param key: The header key. This is assumed to be trusted and valid, it
            must not come from untrusted user input.
        :param default: An iterable of header values to set if the key is not
            present. Newlines are not allowed and will raise a ``ValueError``.

        .. versionadded:: 1.0
        """
        if key not in self:
            self.setlist(key, default)

        return self.getlist(key)

    @t.overload
    def __setitem__(self, key: str, value: t.Any) -> None: ...
    @t.overload
    def __setitem__(self, key: int, value: tuple[str, t.Any]) -> None: ...
    @t.overload
    def __setitem__(
        self, key: slice, value: cabc.Iterable[tuple[str, t.Any]]
    ) -> None: ...
    def __setitem__(
        self,
        key: str | int | slice,
        value: t.Any | tuple[str, t.Any] | cabc.Iterable[tuple[str, t.Any]],
    ) -> None:
        """Like :meth:`set` but also supports index/slice based setting."""
        if isinstance(key, str):
            self.set(key, value)
        elif isinstance(key, int):
            self._list[key] = value[0], _str_header_value(value[1])  # type: ignore[index]
        else:
            self._list[key] = [(k, _str_header_value(v)) for k, v in value]  # type: ignore[str-unpack]

    def update(
        self,
        arg: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[
                str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]
            ]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
        /,
        **kwargs: t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any],
    ) -> None:
        """Replace headers in this object with items from another
        headers object and keyword arguments.

        To extend existing keys instead of replacing, use :meth:`extend`
        instead.

        If provided, the first argument can be another :class:`Headers`
        object, a :class:`MultiDict`, :class:`dict`, or iterable of
        pairs.

        The header keys are assumed to be trusted and valid, they must not come
        from untrusted user input. Newlines in header values are not allowed and
        will raise a ``ValueError``.

        .. versionadded:: 1.0
        """
        if arg is not None:
            if isinstance(arg, (Headers, MultiDict)):
                for key in arg.keys():
                    self.setlist(key, arg.getlist(key))
            elif isinstance(arg, cabc.Mapping):
                for key, value in arg.items():
                    if isinstance(value, (list, tuple, set)):
                        self.setlist(key, value)
                    else:
                        self.set(key, value)
            else:
                for key, value in arg:
                    self.set(key, value)

        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, set)):
                self.setlist(key, value)
            else:
                self.set(key, value)

    def __or__(
        self,
        other: cabc.Mapping[
            str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]
        ],
    ) -> te.Self:
        if not isinstance(other, cabc.Mapping):
            return NotImplemented

        rv = self.copy()
        rv.update(other)
        return rv

    def __ior__(
        self,
        other: (
            cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
        ),
    ) -> te.Self:
        if not isinstance(other, (cabc.Mapping, cabc.Iterable)):
            return NotImplemented

        self.update(other)
        return self

    def to_wsgi_list(self) -> list[tuple[str, str]]:
        """Convert the headers into a list suitable for WSGI.

        :return: list
        """
        return list(self)

    def copy(self) -> te.Self:
        return self.__class__(self._list)

    def __copy__(self) -> te.Self:
        return self.copy()

    def __str__(self) -> str:
        """Returns formatted headers suitable for HTTP transmission."""
        strs = []
        for key, value in self.to_wsgi_list():
            strs.append(f"{key}: {value}")
        strs.append("\r\n")
        return "\r\n".join(strs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"


def _options_header_vkw(value: str, kw: dict[str, t.Any]) -> str:
    return http.dump_options_header(
        value, {k.replace("_", "-"): v for k, v in kw.items()}
    )


_newline_re = re.compile(r"[\r\n]")


def _str_header_value(value: t.Any) -> str:
    if not isinstance(value, str):
        value = str(value)

    if _newline_re.search(value) is not None:
        raise ValueError("Header values must not contain newline characters.")

    return value  # type: ignore[no-any-return]


class EnvironHeaders(ImmutableHeadersMixin, Headers):  # type: ignore[misc]
    """Read only version of the headers from a WSGI environment.  This
    provides the same interface as `Headers` and is constructed from
    a WSGI environment.
    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for
    HTTP exceptions.
    """

    def __init__(self, environ: WSGIEnvironment) -> None:
        super().__init__()
        self.environ = environ

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvironHeaders):
            return NotImplemented

        return self.environ is other.environ

    __hash__ = None

    def __getitem__(self, key: str) -> str:  # type: ignore[override]
        return self._get_key(key)

    def _get_key(self, key: str) -> str:
        if not isinstance(key, str):
            raise BadRequestKeyError(key)

        key = key.upper().replace("-", "_")

        if key in {"CONTENT_TYPE", "CONTENT_LENGTH"}:
            return self.environ[key]  # type: ignore[no-any-return]

        return self.environ[f"HTTP_{key}"]  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> cabc.Iterator[tuple[str, str]]:
        for key, value in self.environ.items():
            if key.startswith("HTTP_") and key not in {
                "HTTP_CONTENT_TYPE",
                "HTTP_CONTENT_LENGTH",
            }:
                yield key[5:].replace("_", "-").title(), value
            elif key in {"CONTENT_TYPE", "CONTENT_LENGTH"} and value:
                yield key.replace("_", "-").title(), value

    def copy(self) -> t.NoReturn:
        raise TypeError(f"cannot create {type(self).__name__!r} copies")

    def __or__(self, other: t.Any) -> t.NoReturn:
        raise TypeError(f"cannot create {type(self).__name__!r} copies")


# circular dependencies
from .. import http  # noqa: E402
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t
from inspect import cleandoc

from .mixins import ImmutableDictMixin
from .structures import CallbackDict


def cache_control_property(
    key: str, empty: t.Any, type: type[t.Any] | None, *, doc: str | None = None
) -> t.Any:
    """Return a new property object for a cache header. Useful if you
    want to add support for a cache extension in a subclass.

    :param key: The attribute name present in the parsed cache-control header dict.
    :param empty: The value to use if the key is present without a value.
    :param type: The type to convert the string value to instead of a string. If
        conversion raises a ``ValueError``, the returned value is ``None``.
    :param doc: The docstring for the property. If not given, it is generated
        based on the other params.

    .. versionchanged:: 3.1
        Added the ``doc`` param.

    .. versionchanged:: 2.0
        Renamed from ``cache_property``.
    """
    if doc is None:
        parts = [f"The ``{key}`` attribute."]

        if type is bool:
            parts.append("A ``bool``, either present or not.")
        else:
            if type is None:
                parts.append("A ``str``,")
            else:
                parts.append(f"A ``{type.__name__}``,")

            if empty is not None:
                parts.append(f"``{empty!r}`` if present with no value,")

            parts.append("or ``None`` if not present.")

        doc = " ".join(parts)

    return property(
        lambda x: x._get_cache_value(key, empty, type),
        lambda x, v: x._set_cache_value(key, v, type),
        lambda x: x._del_cache_value(key),
        doc=cleandoc(doc),
    )


class _CacheControl(CallbackDict[str, t.Optional[str]]):
    """Subclass of a dict that stores values for a Cache-Control header.  It
    has accessors for all the cache-control directives specified in RFC 2616.
    The class does not differentiate between request and response directives.

    Because the cache-control directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`CacheControl` object again you can convert
    the object into a string or call the :meth:`to_header` method.  If you plan
    to subclass it and add your own items have a look at the sourcecode for
    that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionchanged:: 0.4
       Setting ``no_cache`` or ``private`` to ``True`` will set the
       implicit value ``"*"``.
    """

    no_store: bool = cache_control_property("no-store", None, bool)
    max_age: int | None = cache_control_property("max-age", None, int)
    no_transform: bool = cache_control_property("no-transform", None, bool)
    stale_if_error: int | None = cache_control_property("stale-if-error", None, int)

    def __init__(
        self,
        values: cabc.Mapping[str, t.Any] | cabc.Iterable[tuple[str, t.Any]] | None = (),
        on_update: cabc.Callable[[_CacheControl], None] | None = None,
    ):
        super().__init__(values, on_update)
        self.provided = values is not None

    def _get_cache_value(
        self, key: str, empty: t.Any, type: type[t.Any] | None
    ) -> t.Any:
        """Used internally by the accessor properties."""
        if type is bool:
            return key in self

        if key not in self:
            return None

        if (value := self[key]) is None:
            return empty

        if type is not None:
            try:
                value = type(value)
            except ValueError:
                return None

        return value

    def _set_cache_value(
        self, key: str, value: t.Any, type: type[t.Any] | None
    ) -> None:
        """Used internally by the accessor properties."""
        if type is bool:
            if value:
                self[key] = None
            else:
                self.pop(key, None)
        elif value is None or value is False:
            self.pop(key, None)
        elif value is True:
            self[key] = None
        else:
            if type is not None:
                value = type(value)

            self[key] = str(value)

    def _del_cache_value(self, key: str) -> None:
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self) -> str:
        """Convert the stored values into a cache control header."""
        return http.dump_header(self)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        kv_str = " ".join(f"{k}={v!r}" for k, v in sorted(self.items()))
        return f"<{type(self).__name__} {kv_str}>"

    cache_property = staticmethod(cache_control_property)


class RequestCacheControl(ImmutableDictMixin[str, t.Optional[str]], _CacheControl):  # type: ignore[misc]
    """A cache control for requests.  This is immutable and gives access
    to all the request-relevant cache control headers.

    To get a header of the :class:`RequestCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 3.1
       ``max_age`` is ``None`` if present without a value, rather
       than ``-1``.

    .. versionchanged:: 3.1
        ``no_cache`` is a boolean, it is ``True`` instead of ``"*"``
        when present.

    .. versionchanged:: 3.1
        ``max_stale`` is ``True`` if present without a value, rather
        than ``"*"``.

    .. versionchanged:: 3.1
       ``no_transform`` is a boolean. Previously it was mistakenly
       always ``None``.

    .. versionchanged:: 3.1
       ``min_fresh`` is ``None`` if present without a value, rather
       than ``"*"``.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
        Response-only properties are not present on this request class.
    """

    no_cache: bool = cache_control_property("no-cache", None, bool)
    max_stale: int | t.Literal[True] | None = cache_control_property(
        "max-stale",
        True,
        int,
    )
    min_fresh: int | None = cache_control_property("min-fresh", None, int)
    only_if_cached: bool = cache_control_property("only-if-cached", None, bool)


class ResponseCacheControl(_CacheControl):
    """A cache control for responses.  Unlike :class:`RequestCacheControl`
    this is mutable and gives access to response-relevant cache control
    headers.

    To get a header of the :class:`ResponseCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 3.1
        ``no_cache`` is ``True`` if present without a value, rather than
        ``"*"``.

    .. versionchanged:: 3.1
        ``private`` is ``True`` if present without a value, rather than
        ``"*"``.

    .. versionchanged:: 3.1
       ``no_transform`` is a boolean. Previously it was mistakenly
       always ``None``.

    .. versionchanged:: 3.1
        Added the ``must_understand``, ``stale_while_revalidate``, and
        ``stale_if_error`` properties.

    .. versionchanged:: 2.1.1
        ``s_maxage`` converts the value to an int.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
       Request-only properties are not present on this response class.
    """

    no_cache: str | t.Literal[True] | None = cache_control_property(
        "no-cache", True, None
    )
    public: bool = cache_control_property("public", None, bool)
    private: str | t.Literal[True] | None = cache_control_property(
        "private", True, None
    )
    must_revalidate: bool = cache_control_property("must-revalidate", None, bool)
    proxy_revalidate: bool = cache_control_property("proxy-revalidate", None, bool)
    s_maxage: int | None = cache_control_property("s-maxage", None, int)
    immutable: bool = cache_control_property("immutable", None, bool)
    must_understand: bool = cache_control_property("must-understand", None, bool)
    stale_while_revalidate: int | None = cache_control_property(
        "stale-while-revalidate", None, int
    )


# circular dependencies
from .. import http  # noqa: E402
<|endoftext|>from __future__ import annotations

import base64
import binascii
import collections.abc as cabc
import typing as t

from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict

if t.TYPE_CHECKING:
    import typing_extensions as te


class Authorization:
    """Represents the parts of an ``Authorization`` request header.

    :attr:`.Request.authorization` returns an instance if the header is set.

    An instance can be used with the test :class:`.Client` request methods' ``auth``
    parameter to send the header in test requests.

    Depending on the auth scheme, either :attr:`parameters` or :attr:`token` will be
    set. The ``Basic`` scheme's token is decoded into the ``username`` and ``password``
    parameters.

    For convenience, ``auth["key"]`` and ``auth.key`` both access the key in the
    :attr:`parameters` dict, along with ``auth.get("key")`` and ``"key" in auth``.

    .. versionchanged:: 2.3
        The ``token`` parameter and attribute was added to support auth schemes that use
        a token instead of parameters, such as ``Bearer``.

    .. versionchanged:: 2.3
        The object is no longer a ``dict``.

    .. versionchanged:: 0.5
        The object is an immutable dict.
    """

    def __init__(
        self,
        auth_type: str,
        data: dict[str, str | None] | None = None,
        token: str | None = None,
    ) -> None:
        self.type = auth_type
        """The authorization scheme, like ``basic``, ``digest``, or ``bearer``."""

        if data is None:
            data = {}

        self.parameters = data
        """A dict of parameters parsed from the header. Either this or :attr:`token`
        will have a value for a given scheme.
        """

        self.token = token
        """A token parsed from the header. Either this or :attr:`parameters` will have a
        value for a given scheme.

        .. versionadded:: 2.3
        """

    def __getattr__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def __getitem__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.parameters.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Authorization):
            return NotImplemented

        return (
            other.type == self.type
            and other.token == self.token
            and other.parameters == self.parameters
        )

    @classmethod
    def from_header(cls, value: str | None) -> te.Self | None:
        """Parse an ``Authorization`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
        if not value:
            return None

        scheme, _, rest = value.partition(" ")
        scheme = scheme.lower()
        rest = rest.strip(" \t")

        if scheme == "basic":
            try:
                username, _, password = (
                    base64.b64decode(rest, validate=True).decode().partition(":")
                )
            except (binascii.Error, UnicodeError):
                return None

            return cls(scheme, {"username": username, "password": password})

        if "=" in rest.rstrip("="):
            # = that is not trailing, this is parameters.
            return cls(scheme, parse_dict_header(rest), None)

        # No = or only trailing =, this is a token.
        return cls(scheme, None, rest)

    def to_header(self) -> str:
        """Produce an ``Authorization`` header value representing this data.

        .. versionadded:: 2.0
        """
        if self.type == "basic":
            value = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode("ascii")
            return f"Basic {value}"

        if self.token is not None:
            return f"{self.type.title()} {self.token}"

        return f"{self.type.title()} {dump_header(self.parameters)}"

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_header()}>"


class WWWAuthenticate:
    """Represents the parts of a ``WWW-Authenticate`` response header.

    Set :attr:`.Response.www_authenticate` to an instance of list of instances to set
    values for this header in the response. Modifying this instance will modify the
    header value.

    Depending on the auth scheme, either :attr:`parameters` or :attr:`token` should be
    set. The ``Basic`` scheme will encode ``username`` and ``password`` parameters to a
    token.

    For convenience, ``auth["key"]`` and ``auth.key`` both act on the :attr:`parameters`
    dict, and can be used to get, set, or delete parameters. ``auth.get("key")`` and
    ``"key" in auth`` are also provided.

    .. versionchanged:: 2.3
        The ``token`` parameter and attribute was added to support auth schemes that use
        a token instead of parameters, such as ``Bearer``.

    .. versionchanged:: 2.3
        The object is no longer a ``dict``.

    .. versionchanged:: 2.3
        The ``on_update`` parameter was removed.
    """

    def __init__(
        self,
        auth_type: str,
        values: dict[str, str | None] | None = None,
        token: str | None = None,
    ):
        self._type = auth_type.lower()
        self._parameters: dict[str, str | None] = CallbackDict(
            values, lambda _: self._trigger_on_update()
        )
        self._token = token
        self._on_update: cabc.Callable[[WWWAuthenticate], None] | None = None

    def _trigger_on_update(self) -> None:
        if self._on_update is not None:
            self._on_update(self)

    @property
    def type(self) -> str:
        """The authorization scheme, like ``basic``, ``digest``, or ``bearer``."""
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        self._type = value
        self._trigger_on_update()

    @property
    def parameters(self) -> dict[str, str | None]:
        """A dict of parameters for the header. Only one of this or :attr:`token` should
        have a value for a given scheme.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: dict[str, str]) -> None:
        self._parameters = CallbackDict(value, lambda _: self._trigger_on_update())
        self._trigger_on_update()

    @property
    def token(self) -> str | None:
        """A dict of parameters for the header. Only one of this or :attr:`token` should
        have a value for a given scheme.
        """
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        """A token for the header. Only one of this or :attr:`parameters` should have a
        value for a given scheme.

        .. versionadded:: 2.3
        """
        self._token = value
        self._trigger_on_update()

    def __getitem__(self, key: str) -> str | None:
        return self.parameters.get(key)

    def __setitem__(self, key: str, value: str | None) -> None:
        if value is None:
            if key in self.parameters:
                del self.parameters[key]
        else:
            self.parameters[key] = value

        self._trigger_on_update()

    def __delitem__(self, key: str) -> None:
        if key in self.parameters:
            del self.parameters[key]
            self._trigger_on_update()

    def __getattr__(self, name: str) -> str | None:
        return self[name]

    def __setattr__(self, name: str, value: str | None) -> None:
        if name in {"_type", "_parameters", "_token", "_on_update"}:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WWWAuthenticate):
            return NotImplemented

        return (
            other.type == self.type
            and other.token == self.token
            and other.parameters == self.parameters
        )

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.parameters.get(key, default)

    @classmethod
    def from_header(cls, value: str | None) -> te.Self | None:
        """Parse a ``WWW-Authenticate`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
        if not value:
            return None

        scheme, _, rest = value.partition(" ")
        scheme = scheme.lower()
        rest = rest.strip(" \t")

        if "=" in rest.rstrip("="):
            # = that is not trailing, this is parameters.
            return cls(scheme, parse_dict_header(rest), None)

        # No = or only trailing =, this is a token.
        return cls(scheme, None, rest)

    def to_header(self) -> str:
        """Produce a ``WWW-Authenticate`` header value representing this data."""
        if self.token is not None:
            return f"{self.type.title()} {self.token}"

        if not self.parameters:
            return self.type.title()

        if self.type == "digest":
            items = []

            for key, value in self.parameters.items():
                if key in {"realm", "domain", "nonce", "opaque", "qop"}:
                    value = quote_header_value(value, allow_token=False)
                else:
                    value = quote_header_value(value)

                items.append(f"{key}={value}")

            return f"Digest {', '.join(items)}"

        return f"{self.type.title()} {dump_header(self.parameters)}"

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_header()}>"
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t

from .structures import CallbackDict


def csp_property(key: str) -> t.Any:
    """Return a new property object for a content security policy header.
    Useful if you want to add support for a csp extension in a
    subclass.
    """
    return property(
        lambda x: x._get_value(key),
        lambda x, v: x._set_value(key, v),
        lambda x: x._del_value(key),
        f"accessor for {key!r}",
    )


class ContentSecurityPolicy(CallbackDict[str, str]):
    """Subclass of a dict that stores values for a Content Security Policy
    header. It has accessors for all the level 3 policies.

    Because the csp directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`ContentSecurityPolicy` object again
    you can convert the object into a string or call the
    :meth:`to_header` method.  If you plan to subclass it and add your
    own items have a look at the sourcecode for that class.

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """

    base_uri: str | None = csp_property("base-uri")
    child_src: str | None = csp_property("child-src")
    connect_src: str | None = csp_property("connect-src")
    default_src: str | None = csp_property("default-src")
    font_src: str | None = csp_property("font-src")
    form_action: str | None = csp_property("form-action")
    frame_ancestors: str | None = csp_property("frame-ancestors")
    frame_src: str | None = csp_property("frame-src")
    img_src: str | None = csp_property("img-src")
    manifest_src: str | None = csp_property("manifest-src")
    media_src: str | None = csp_property("media-src")
    navigate_to: str | None = csp_property("navigate-to")
    object_src: str | None = csp_property("object-src")
    prefetch_src: str | None = csp_property("prefetch-src")
    plugin_types: str | None = csp_property("plugin-types")
    report_to: str | None = csp_property("report-to")
    report_uri: str | None = csp_property("report-uri")
    sandbox: str | None = csp_property("sandbox")
    script_src: str | None = csp_property("script-src")
    script_src_attr: str | None = csp_property("script-src-attr")
    script_src_elem: str | None = csp_property("script-src-elem")
    style_src: str | None = csp_property("style-src")
    style_src_attr: str | None = csp_property("style-src-attr")
    style_src_elem: str | None = csp_property("style-src-elem")
    worker_src: str | None = csp_property("worker-src")

    def __init__(
        self,
        values: cabc.Mapping[str, str] | cabc.Iterable[tuple[str, str]] | None = (),
        on_update: cabc.Callable[[ContentSecurityPolicy], None] | None = None,
    ) -> None:
        super().__init__(values, on_update)
        self.provided = values is not None

    def _get_value(self, key: str) -> str | None:
        """Used internally by the accessor properties."""
        return self.get(key)

    def _set_value(self, key: str, value: str | None) -> None:
        """Used internally by the accessor properties."""
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value

    def _del_value(self, key: str) -> None:
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self) -> str:
        """Convert the stored values into a cache control header."""
        from ..http import dump_csp_header

        return dump_csp_header(self)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        kv_str = " ".join(f"{k}={v!r}" for k, v in sorted(self.items()))
        return f"<{type(self).__name__} {kv_str}>"
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t
from functools import update_wrapper
from itertools import repeat

from .._internal import _missing

if t.TYPE_CHECKING:
    import typing_extensions as te

K = t.TypeVar("K")
V = t.TypeVar("V")
T = t.TypeVar("T")
F = t.TypeVar("F", bound=cabc.Callable[..., t.Any])


def _immutable_error(self: t.Any) -> t.NoReturn:
    raise TypeError(f"{type(self).__name__!r} objects are immutable")


class ImmutableListMixin:
    """Makes a :class:`list` immutable.

    .. versionadded:: 0.5

    :private:
    """

    _hash_cache: int | None = None

    def __hash__(self) -> int:
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(tuple(self))  # type: ignore[arg-type]
        return rv

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self),)  # type: ignore[call-overload]

    def __delitem__(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __iadd__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __imul__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def append(self, item: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def remove(self, item: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def extend(self, iterable: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def insert(self, pos: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, index: t.Any = -1) -> t.NoReturn:
        _immutable_error(self)

    def reverse(self: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def sort(self, key: t.Any = None, reverse: t.Any = False) -> t.NoReturn:
        _immutable_error(self)


class ImmutableDictMixin(t.Generic[K, V]):
    """Makes a :class:`dict` immutable.

    .. versionchanged:: 3.1
        Disallow ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    _hash_cache: int | None = None

    @classmethod
    @t.overload
    def fromkeys(
        cls, keys: cabc.Iterable[K], value: None
    ) -> ImmutableDictMixin[K, t.Any | None]: ...
    @classmethod
    @t.overload
    def fromkeys(cls, keys: cabc.Iterable[K], value: V) -> ImmutableDictMixin[K, V]: ...
    @classmethod
    def fromkeys(
        cls, keys: cabc.Iterable[K], value: V | None = None
    ) -> ImmutableDictMixin[K, t.Any | None] | ImmutableDictMixin[K, V]:
        instance = super().__new__(cls)
        instance.__init__(zip(keys, repeat(value)))  # type: ignore[misc]
        return instance

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (dict(self),)  # type: ignore[call-overload]

    def _iter_hashitems(self) -> t.Iterable[t.Any]:
        return self.items()  # type: ignore[attr-defined,no-any-return]

    def __hash__(self) -> int:
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(frozenset(self._iter_hashitems()))
        return rv

    def setdefault(self, key: t.Any, default: t.Any = None) -> t.NoReturn:
        _immutable_error(self)

    def update(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __ior__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, key: t.Any, default: t.Any = None) -> t.NoReturn:
        _immutable_error(self)

    def popitem(self) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __delitem__(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def clear(self) -> t.NoReturn:
        _immutable_error(self)


class ImmutableMultiDictMixin(ImmutableDictMixin[K, V]):
    """Makes a :class:`MultiDict` immutable.

    .. versionadded:: 0.5

    :private:
    """

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self.items(multi=True)),)  # type: ignore[attr-defined]

    def _iter_hashitems(self) -> t.Iterable[t.Any]:
        return self.items(multi=True)  # type: ignore[attr-defined,no-any-return]

    def add(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def popitemlist(self) -> t.NoReturn:
        _immutable_error(self)

    def poplist(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlist(self, key: t.Any, new_list: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlistdefault(self, key: t.Any, default_list: t.Any = None) -> t.NoReturn:
        _immutable_error(self)


class ImmutableHeadersMixin:
    """Makes a :class:`Headers` immutable.  We do not mark them as
    hashable though since the only usecase for this datastructure
    in Werkzeug is a view on a mutable structure.

    .. versionchanged:: 3.1
        Disallow ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    def __delitem__(self, key: t.Any, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def set(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlist(self, key: t.Any, values: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def add(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def add_header(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def remove(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def extend(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def update(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __ior__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def insert(self, pos: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, key: t.Any = None, default: t.Any = _missing) -> t.NoReturn:
        _immutable_error(self)

    def popitem(self) -> t.NoReturn:
        _immutable_error(self)

    def setdefault(self, key: t.Any, default: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlistdefault(self, key: t.Any, default: t.Any) -> t.NoReturn:
        _immutable_error(self)


def _always_update(f: F) -> F:
    def wrapper(
        self: UpdateDictMixin[t.Any, t.Any], /, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        rv = f(self, *args, **kwargs)

        if self.on_update is not None:
            self.on_update(self)

        return rv

    return update_wrapper(wrapper, f)  # type: ignore[return-value]


class UpdateDictMixin(dict[K, V]):
    """Makes dicts call `self.on_update` on modifications.

    .. versionchanged:: 3.1
        Implement ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    on_update: cabc.Callable[[te.Self], None] | None = None

    def setdefault(self: te.Self, key: K, default: V | None = None) -> V:
        modified = key not in self
        rv = super().setdefault(key, default)  # type: ignore[arg-type]
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    @t.overload
    def pop(self: te.Self, key: K) -> V: ...
    @t.overload
    def pop(self: te.Self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self: te.Self, key: K, default: T) -> T: ...
    def pop(
        self: te.Self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        modified = key in self
        if default is _missing:
            rv: V | T = super().pop(key)
        else:
            rv = super().pop(key, default)
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    @_always_update
    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)

    @_always_update
    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)

    @_always_update
    def clear(self) -> None:
        super().clear()

    @_always_update
    def popitem(self) -> tuple[K, V]:
        return super().popitem()

    @_always_update
    def update(  # type: ignore[override]
        self,
        arg: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]] | None = None,
        /,
        **kwargs: V,
    ) -> None:
        if arg is None:
            super().update(**kwargs)  # type: ignore[call-overload]
        else:
            super().update(arg, **kwargs)

    @_always_update
    def __ior__(  # type: ignore[override]
        self, other: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]]
    ) -> te.Self:
        return super().__ior__(other)
<|endoftext|>from __future__ import annotations

import typing as t
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from http import HTTPStatus

from ..datastructures import CallbackDict
from ..datastructures import ContentRange
from ..datastructures import ContentSecurityPolicy
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..datastructures import ResponseCacheControl
from ..datastructures import WWWAuthenticate
from ..http import COEP
from ..http import COOP
from ..http import dump_age
from ..http import dump_cookie
from ..http import dump_header
from ..http import dump_options_header
from ..http import http_date
from ..http import HTTP_STATUS_CODES
from ..http import parse_age
from ..http import parse_cache_control_header
from ..http import parse_content_range_header
from ..http import parse_csp_header
from ..http import parse_date
from ..http import parse_options_header
from ..http import parse_set_header
from ..http import quote_etag
from ..http import unquote_etag
from ..utils import get_content_type
from ..utils import header_property

if t.TYPE_CHECKING:
    from ..datastructures.cache_control import _CacheControl


def _set_property(name: str, doc: str | None = None) -> property:
    def fget(self: Response) -> HeaderSet:
        def on_update(header_set: HeaderSet) -> None:
            if not header_set and name in self.headers:
                del self.headers[name]
            elif header_set:
                self.headers[name] = header_set.to_header()

        return parse_set_header(self.headers.get(name), on_update)

    def fset(
        self: Response,
        value: None | (str | dict[str, str | int] | t.Iterable[str]),
    ) -> None:
        if not value:
            del self.headers[name]
        elif isinstance(value, str):
            self.headers[name] = value
        else:
            self.headers[name] = dump_header(value)

    return property(fget, fset, doc=doc)


class Response:
    """Represents the non-IO parts of an HTTP response, specifically the
    status and headers but not the body.

    This class is not meant for general use. It should only be used when
    implementing WSGI, ASGI, or another HTTP application spec. Werkzeug
    provides a WSGI implementation at :cls:`werkzeug.wrappers.Response`.

    :param status: The status code for the response. Either an int, in
        which case the default status message is added, or a string in
        the form ``{code} {message}``, like ``404 Not Found``. Defaults
        to 200.
    :param headers: A :class:`~werkzeug.datastructures.Headers` object,
        or a list of ``(key, value)`` tuples that will be converted to a
        ``Headers`` object.
    :param mimetype: The mime type (content type without charset or
        other parameters) of the response. If the value starts with
        ``text/`` (or matches some other special cases), the charset
        will be added to create the ``content_type``.
    :param content_type: The full content type of the response.
        Overrides building the value from ``mimetype``.

    .. versionchanged:: 3.0
        The ``charset`` attribute was removed.

    .. versionadded:: 2.0
    """

    #: the default status if none is provided.
    default_status = 200

    #: the default mimetype if none is provided.
    default_mimetype: str | None = "text/plain"

    #: Warn if a cookie header exceeds this size. The default, 4093, should be
    #: safely `supported by most browsers <cookie_>`_. A cookie larger than
    #: this size will still be sent, but it may be ignored or handled
    #: incorrectly by some browsers. Set to 0 to disable this check.
    #:
    #: .. versionadded:: 0.13
    #:
    #: .. _`cookie`: http://browsercookielimits.squawky.net/
    max_cookie_size = 4093

    # A :class:`Headers` object representing the response headers.
    headers: Headers

    def __init__(
        self,
        status: int | str | HTTPStatus | None = None,
        headers: t.Mapping[str, str | t.Iterable[str]]
        | t.Iterable[tuple[str, str]]
        | None = None,
        mimetype: str | None = None,
        content_type: str | None = None,
    ) -> None:
        if isinstance(headers, Headers):
            self.headers = headers
        elif not headers:
            self.headers = Headers()
        else:
            self.headers = Headers(headers)

        if content_type is None:
            if mimetype is None and "content-type" not in self.headers:
                mimetype = self.default_mimetype
            if mimetype is not None:
                mimetype = get_content_type(mimetype, "utf-8")
            content_type = mimetype
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if status is None:
            status = self.default_status
        self.status = status

    def __repr__(self) -> str:
        return f"<{type(self).__name__} [{self.status}]>"

    @property
    def status_code(self) -> int:
        """The HTTP status code as a number."""
        return self._status_code

    @status_code.setter
    def status_code(self, code: int) -> None:
        self.status = code

    @property
    def status(self) -> str:
        """The HTTP status code as a string."""
        return self._status

    @status.setter
    def status(self, value: str | int | HTTPStatus) -> None:
        self._status, self._status_code = self._clean_status(value)

    def _clean_status(self, value: str | int | HTTPStatus) -> tuple[str, int]:
        if isinstance(value, (int, HTTPStatus)):
            status_code = int(value)
        else:
            value = value.strip()

            if not value:
                raise ValueError("Empty status argument")

            code_str, sep, _ = value.partition(" ")

            try:
                status_code = int(code_str)
            except ValueError:
                # only message
                return f"0 {value}", 0

            if sep:
                # code and message
                return value, status_code

        # only code, look up message
        try:
            status = f"{status_code} {HTTP_STATUS_CODES[status_code].upper()}"
        except KeyError:
            status = f"{status_code} UNKNOWN"

        return status, status_code

    def set_cookie(
        self,
        key: str,
        value: str = "",
        max_age: timedelta | int | None = None,
        expires: str | datetime | int | float | None = None,
        path: str | None = "/",
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
        partitioned: bool = False,
    ) -> None:
        """Sets a cookie.

        A warning is raised if the size of the cookie header exceeds
        :attr:`max_cookie_size`, but the header will still be set.

        :param key: the key (name) of the cookie to be set.
        :param value: the value of the cookie.
        :param max_age: should be a number of seconds, or `None` (default) if
                        the cookie should last only as long as the client's
                        browser session.
        :param expires: should be a `datetime` object or UNIX timestamp.
        :param path: limits the cookie to a given path, per default it will
                     span the whole domain.
        :param domain: if you want to set a cross-domain cookie.  For example,
                       ``domain="example.com"`` will set a cookie that is
                       readable by the domain ``www.example.com``,
                       ``foo.example.com`` etc.  Otherwise, a cookie will only
                       be readable by the domain that set it.
        :param secure: If ``True``, the cookie will only be available
            via HTTPS.
        :param httponly: Disallow JavaScript access to the cookie.
        :param samesite: Limit the scope of the cookie to only be
            attached to requests that are "same-site".
        :param partitioned: If ``True``, the cookie will be partitioned.

        .. versionchanged:: 3.1
            The ``partitioned`` parameter was added.
        """
        self.headers.add(
            "Set-Cookie",
            dump_cookie(
                key,
                value=value,
                max_age=max_age,
                expires=expires,
                path=path,
                domain=domain,
                secure=secure,
                httponly=httponly,
                max_size=self.max_cookie_size,
                samesite=samesite,
                partitioned=partitioned,
            ),
        )

    def delete_cookie(
        self,
        key: str,
        path: str | None = "/",
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
        partitioned: bool = False,
    ) -> None:
        """Delete a cookie.  Fails silently if key doesn't exist.

        :param key: the key (name) of the cookie to be deleted.
        :param path: if the cookie that should be deleted was limited to a
                     path, the path has to be defined here.
        :param domain: if the cookie that should be deleted was limited to a
                       domain, that domain has to be defined here.
        :param secure: If ``True``, the cookie will only be available
            via HTTPS.
        :param httponly: Disallow JavaScript access to the cookie.
        :param samesite: Limit the scope of the cookie to only be
            attached to requests that are "same-site".
        :param partitioned: If ``True``, the cookie will be partitioned.
        """
        self.set_cookie(
            key,
            expires=0,
            max_age=0,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
            partitioned=partitioned,
        )

    @property
    def is_json(self) -> bool:
        """Check if the mimetype indicates JSON data, either
        :mimetype:`application/json` or :mimetype:`application/*+json`.
        """
        mt = self.mimetype
        return mt is not None and (
            mt == "application/json"
            or mt.startswith("application/")
            and mt.endswith("+json")
        )

    # Common Descriptors

    @property
    def mimetype(self) -> str | None:
        """The mimetype (content type without charset etc.)"""
        ct = self.headers.get("content-type")

        if ct:
            return ct.partition(";")[0].strip()
        else:
            return None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.headers["Content-Type"] = get_content_type(value, "utf-8")

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict. For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.5
        """

        def on_update(d: CallbackDict[str, str]) -> None:
            self.headers["Content-Type"] = dump_options_header(self.mimetype, d)

        d = parse_options_header(self.headers.get("content-type", ""))[1]
        return CallbackDict(d, on_update)

    location = header_property[str](
        "Location",
        doc="""The Location response-header field is used to redirect
        the recipient to a location other than the Request-URI for
        completion of the request or identification of a new
        resource.""",
    )
    age = header_property(
        "Age",
        None,
        parse_age,
        dump_age,  # type: ignore
        doc="""The Age response-header field conveys the sender's
        estimate of the amount of time since the response (or its
        revalidation) was generated at the origin server.

        Age values are non-negative decimal integers, representing time
        in seconds.""",
    )
    content_type = header_property[str](
        "Content-Type",
        doc="""The Content-Type entity-header field indicates the media
        type of the entity-body sent to the recipient or, in the case of
        the HEAD method, the media type that would have been sent had
        the request been a GET.""",
    )
    content_length = header_property(
        "Content-Length",
        None,
        int,
        str,
        doc="""The Content-Length entity-header field indicates the size
        of the entity-body, in decimal number of OCTETs, sent to the
        recipient or, in the case of the HEAD method, the size of the
        entity-body that would have been sent had the request been a
        GET.""",
    )
    content_location = header_property[str](
        "Content-Location",
        doc="""The Content-Location entity-header field MAY be used to
        supply the resource location for the entity enclosed in the
        message when that entity is accessible from a location separate
        from the requested resource's URI.""",
    )
    content_encoding = header_property[str](
        "Content-Encoding",
        doc="""The Content-Encoding entity-header field is used as a
        modifier to the media-type. When present, its value indicates
        what additional content codings have been applied to the
        entity-body, and thus what decoding mechanisms must be applied
        in order to obtain the media-type referenced by the Content-Type
        header field.""",
    )
    content_md5 = header_property[str](
        "Content-MD5",
        doc="""The Content-MD5 entity-header field, as defined in
        RFC 1864, is an MD5 digest of the entity-body for the purpose of
        providing an end-to-end message integrity check (MIC) of the
        entity-body. (Note: a MIC is good for detecting accidental
        modification of the entity-body in transit, but is not proof
        against malicious attacks.)""",
    )
    date = header_property(
        "Date",
        None,
        parse_date,
        http_date,
        doc="""The Date general-header field represents the date and
        time at which the message was originated, having the same
        semantics as orig-date in RFC 822.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )
    expires = header_property(
        "Expires",
        None,
        parse_date,
        http_date,
        doc="""The Expires entity-header field gives the date/time after
        which the response is considered stale. A stale cache entry may
        not normally be returned by a cache.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )
    last_modified = header_property(
        "Last-Modified",
        None,
        parse_date,
        http_date,
        doc="""The Last-Modified entity-header field indicates the date
        and time at which the origin server believes the variant was
        last modified.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )

    @property
    def retry_after(self) -> datetime | None:
        """The Retry-After response-header field can be used with a
        503 (Service Unavailable) response to indicate how long the
        service is expected to be unavailable to the requesting client.

        Time in seconds until expiration or date.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        value = self.headers.get("retry-after")
        if value is None:
            return None

        try:
            seconds = int(value)
        except ValueError:
            return parse_date(value)

        return datetime.now(timezone.utc) + timedelta(seconds=seconds)

    @retry_after.setter
    def retry_after(self, value: datetime | int | str | None) -> None:
        if value is None:
            if "retry-after" in self.headers:
                del self.headers["retry-after"]
            return
        elif isinstance(value, datetime):
            value = http_date(value)
        else:
            value = str(value)
        self.headers["Retry-After"] = value

    vary = _set_property(
        "Vary",
        doc="""The Vary field value indicates the set of request-header
        fields that fully determines, while the response is fresh,
        whether a cache is permitted to use the response to reply to a
        subsequent request without revalidation.""",
    )
    content_language = _set_property(
        "Content-Language",
        doc="""The Content-Language entity-header field describes the
        natural language(s) of the intended audience for the enclosed
        entity. Note that this might not be equivalent to all the
        languages used within the entity-body.""",
    )
    allow = _set_property(
        "Allow",
        doc="""The Allow entity-header field lists the set of methods
        supported by the resource identified by the Request-URI. The
        purpose of this field is strictly to inform the recipient of
        valid methods associated with the resource. An Allow header
        field MUST be present in a 405 (Method Not Allowed)
        response.""",
    )

    # ETag

    @property
    def cache_control(self) -> ResponseCacheControl:
        """The Cache-Control general-header field is used to specify
        directives that MUST be obeyed by all caching mechanisms along the
        request/response chain.
        """

        def on_update(cache_control: _CacheControl) -> None:
            if not cache_control and "cache-control" in self.headers:
                del self.headers["cache-control"]
            elif cache_control:
                self.headers["Cache-Control"] = cache_control.to_header()

        return parse_cache_control_header(
            self.headers.get("cache-control"), on_update, ResponseCacheControl
        )

    def set_etag(self, etag: str, weak: bool = False) -> None:
        """Set the etag, and override the old one if there was one."""
        self.headers["ETag"] = quote_etag(etag, weak)

    def get_etag(self) -> tuple[str, bool] | tuple[None, None]:
        """Return a tuple in the form ``(etag, is_weak)``.  If there is no
        ETag the return value is ``(None, None)``.
        """
        return unquote_etag(self.headers.get("ETag"))

    accept_ranges = header_property[str](
        "Accept-Ranges",
        doc="""The `Accept-Ranges` header. Even though the name would
        indicate that multiple values are supported, it must be one
        string token only.

        The values ``'bytes'`` and ``'none'`` are common.

        .. versionadded:: 0.7""",
    )

    @property
    def content_range(self) -> ContentRange:
        """The ``Content-Range`` header as a
        :class:`~werkzeug.datastructures.ContentRange` object. Available
        even if the header is not set.

        .. versionadded:: 0.7
        """

        def on_update(rng: ContentRange) -> None:
            if not rng:
                del self.headers["content-range"]
            else:
                self.headers["Content-Range"] = rng.to_header()

        rv = parse_content_range_header(self.headers.get("content-range"), on_update)
        # always provide a content range object to make the descriptor
        # more user friendly.  It provides an unset() method that can be
        # used to remove the header quickly.
        if rv is None:
            rv = ContentRange(None, None, None, on_update=on_update)
        return rv

    @content_range.setter
    def content_range(self, value: ContentRange | str | None) -> None:
        if not value:
            del self.headers["content-range"]
        elif isinstance(value, str):
            self.headers["Content-Range"] = value
        else:
            self.headers["Content-Range"] = value.to_header()

    # Authorization

    @property
    def www_authenticate(self) -> WWWAuthenticate:
        """The ``WWW-Authenticate`` header parsed into a :class:`.WWWAuthenticate`
        object. Modifying the object will modify the header value.

        This header is not set by default. To set this header, assign an instance of
        :class:`.WWWAuthenticate` to this attribute.

        .. code-block:: python

            response.www_authenticate = WWWAuthenticate(
                "basic", {"realm": "Authentication Required"}
            )

        Multiple values for this header can be sent to give the client multiple options.
        Assign a list to set multiple headers. However, modifying the items in the list
        will not automatically update the header values, and accessing this attribute
        will only ever return the first value.

        To unset this header, assign ``None`` or use ``del``.

        .. versionchanged:: 2.3
            This attribute can be assigned to set the header. A list can be assigned
            to set multiple header values. Use ``del`` to unset the header.

        .. versionchanged:: 2.3
            :class:`WWWAuthenticate` is no longer a ``dict``. The ``token`` attribute
            was added for auth challenges that use a token instead of parameters.
        """
        value = WWWAuthenticate.from_header(self.headers.get("WWW-Authenticate"))

        if value is None:
            value = WWWAuthenticate("basic")

        def on_update(value: WWWAuthenticate) -> None:
            self.www_authenticate = value

        value._on_update = on_update
        return value

    @www_authenticate.setter
    def www_authenticate(
        self, value: WWWAuthenticate | list[WWWAuthenticate] | None
    ) -> None:
        if not value:  # None or empty list
            del self.www_authenticate
        elif isinstance(value, list):
            # Clear any existing header by setting the first item.
            self.headers.set("WWW-Authenticate", value[0].to_header())

            for item in value[1:]:
                # Add additional header lines for additional items.
                self.headers.add("WWW-Authenticate", item.to_header())
        else:
            self.headers.set("WWW-Authenticate", value.to_header())

            def on_update(value: WWWAuthenticate) -> None:
                self.www_authenticate = value

            # When setting a single value, allow updating it directly.
            value._on_update = on_update

    @www_authenticate.deleter
    def www_authenticate(self) -> None:
        if "WWW-Authenticate" in self.headers:
            del self.headers["WWW-Authenticate"]

    # CSP

    @property
    def content_security_policy(self) -> ContentSecurityPolicy:
        """The ``Content-Security-Policy`` header as a
        :class:`~werkzeug.datastructures.ContentSecurityPolicy` object. Available
        even if the header is not set.

        The Content-Security-Policy header adds an additional layer of
        security to help detect and mitigate certain types of attacks.
        """

        def on_update(csp: ContentSecurityPolicy) -> None:
            if not csp:
                del self.headers["content-security-policy"]
            else:
                self.headers["Content-Security-Policy"] = csp.to_header()

        rv = parse_csp_header(self.headers.get("content-security-policy"), on_update)
        if rv is None:
            rv = ContentSecurityPolicy(None, on_update=on_update)
        return rv

    @content_security_policy.setter
    def content_security_policy(
        self, value: ContentSecurityPolicy | str | None
    ) -> None:
        if not value:
            del self.headers["content-security-policy"]
        elif isinstance(value, str):
            self.headers["Content-Security-Policy"] = value
        else:
            self.headers["Content-Security-Policy"] = value.to_header()

    @property
    def content_security_policy_report_only(self) -> ContentSecurityPolicy:
        """The ``Content-Security-policy-report-only`` header as a
        :class:`~werkzeug.datastructures.ContentSecurityPolicy` object. Available
        even if the header is not set.

        The Content-Security-Policy-Report-Only header adds a csp policy
        that is not enforced but is reported thereby helping detect
        certain types of attacks.
        """

        def on_update(csp: ContentSecurityPolicy) -> None:
            if not csp:
                del self.headers["content-security-policy-report-only"]
            else:
                self.headers["Content-Security-policy-report-only"] = csp.to_header()

        rv = parse_csp_header(
            self.headers.get("content-security-policy-report-only"), on_update
        )
        if rv is None:
            rv = ContentSecurityPolicy(None, on_update=on_update)
        return rv

    @content_security_policy_report_only.setter
    def content_security_policy_report_only(
        self, value: ContentSecurityPolicy | str | None
    ) -> None:
        if not value:
            del self.headers["content-security-policy-report-only"]
        elif isinstance(value, str):
            self.headers["Content-Security-policy-report-only"] = value
        else:
            self.headers["Content-Security-policy-report-only"] = value.to_header()

    # CORS

    @property
    def access_control_allow_credentials(self) -> bool:
        """Whether credentials can be shared by the browser to
        JavaScript code. As part of the preflight request it indicates
        whether credentials can be used on the cross origin request.
        """
        return "Access-Control-Allow-Credentials" in self.headers

    @access_control_allow_credentials.setter
    def access_control_allow_credentials(self, value: bool | None) -> None:
        if value is True:
            self.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            self.headers.pop("Access-Control-Allow-Credentials", None)

    access_control_allow_headers = header_property(
        "Access-Control-Allow-Headers",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which headers can be sent with the cross origin request.",
    )

    access_control_allow_methods = header_property(
        "Access-Control-Allow-Methods",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which methods can be used for the cross origin request.",
    )

    access_control_allow_origin = header_property[str](
        "Access-Control-Allow-Origin",
        doc="The origin or '*' for any origin that may make cross origin requests.",
    )

    access_control_expose_headers = header_property(
        "Access-Control-Expose-Headers",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which headers can be shared by the browser to JavaScript code.",
    )

    access_control_max_age = header_property(
        "Access-Control-Max-Age",
        load_func=int,
        dump_func=str,
        doc="The maximum age in seconds the access control settings can be cached for.",
    )

    cross_origin_opener_policy = header_property[COOP](
        "Cross-Origin-Opener-Policy",
        load_func=lambda value: COOP(value),
        dump_func=lambda value: value.value,
        default=COOP.UNSAFE_NONE,
        doc="""Allows control over sharing of browsing context group with cross-origin
        documents. Values must be a member of the :class:`werkzeug.http.COOP` enum.""",
    )

    cross_origin_embedder_policy = header_property[COEP](
        "Cross-Origin-Embedder-Policy",
        load_func=lambda value: COEP(value),
        dump_func=lambda value: value.value,
        default=COEP.UNSAFE_NONE,
        doc="""Prevents a document from loading any cross-origin resources that do not
        explicitly grant the document permission. Values must be a member of the
        :class:`werkzeug.http.COEP` enum.""",
    )
<|endoftext|>from __future__ import annotations

import re
import typing as t
from urllib.parse import quote

from .._internal import _plain_int
from ..exceptions import SecurityError
from ..http import parse_set_header
from ..urls import uri_to_iri

_host_re = re.compile(
    r"""
    (
        [a-z0-9.-]+  # domain or ipv4
    |
        \[[a-f0-9]*:[a-f0-9.:]+]  # ipv6
    )
    (?::([1-9][0-9]{,4}))?  # optional port
    """,
    flags=re.ASCII | re.IGNORECASE | re.VERBOSE,
)


def host_is_trusted(
    hostname: str | None, trusted_list: t.Collection[str] | None = None
) -> bool:
    """Perform some checks on a ``Host`` header ``host:port``. The host must be
    made up of valid characters, but this does not check validity beyond that.
    If a list of trusted domains is given, the domain must match one.

    :param hostname: The ``Host`` header ``host:port`` to check.
    :param trusted_list: A list of trusted domains to match. These should
        already be IDNA encoded, but will be encoded if needed. The port is
        ignored for this check. If a name starts with a dot it will match as a
        suffix, accepting all subdomains. If empty or ``None``, all domains are
        allowed.

    .. versionchanged:: 3.2
        The value's characters are validated.

    .. versionchanged:: 3.2
        ``trusted_list`` defaults to ``None``.

    .. versionadded:: 0.9
    """
    if not hostname:
        return False

    if (m := _host_re.fullmatch(hostname)) is None:
        return False

    hostname, port_str = m.groups()

    if port_str and not (1 <= int(port_str) <= 65535):
        return False

    if not trusted_list:
        return True

    if isinstance(trusted_list, str):
        trusted_list = [trusted_list]

    for ref in trusted_list:
        if ref.startswith("."):
            ref = ref[1:]
            suffix_match = True
        else:
            suffix_match = False

        try:
            ref = ref.partition(":")[0].encode("idna").decode("ascii")
        except UnicodeEncodeError:
            return False

        if ref == hostname or (suffix_match and hostname.endswith(f".{ref}")):
            return True

    return False


def get_host(
    scheme: str,
    host_header: str | None,
    server: tuple[str, int | None] | None = None,
    trusted_hosts: t.Collection[str] | None = None,
) -> str:
    """Get and validate a request's ``host:port`` based on the given values.

    The ``Host`` header sent by the client is preferred. Otherwise, the server's
    configured address is used. The port is omitted if it matches the standard
    HTTP or HTTPS ports.

    The value is passed through :func:`host_is_trusted`. The host must be made
    up of valid characters, but this does not check validity beyond that. If a
    list of trusted domains is given, the domain must match one.

    If the host header is not available, such as for HTTP/0.9 and 1.0, or it has
    invalid characters, the empty string is returned. Subdomain and host
    routing, and external URL building, will not work in these cases.

    :param scheme: The protocol of the request. Used to omit the standard ports
        80 and 443.
    :param host_header: The ``Host`` header value.
    :param server: The server's configured address ``(host, port)``. The server
        may be using a Unix socket and give ``(path, None)``; this is ignored as
        it would not produce a useful host value.
    :param trusted_hosts: A list of trusted domains to match. These should
        already be IDNA encoded, but will be encoded if needed. The port is
        ignored for this check. If a name starts with a dot it will match as a
        suffix, accepting all subdomains. If empty or ``None``, all domains are
        allowed.

    :return: Host, with port if necessary.
    :raise .SecurityError: If the host is not trusted.

    .. versionchanged:: 3.1.8
        The empty string is again returned if no host header value is available,
        or if the characters are invalid.

    .. versionchanged:: 3.1.7
        The characters of the host value are validated. The empty string is no
        longer allowed if no header value is available.

    .. versionchanged:: 3.2
        When using the server address, Unix sockets are ignored.

    .. versionchanged:: 3.1.3
        If ``SERVER_NAME`` is IPv6, it is wrapped in ``[]``.
    """
    if host_header is not None:
        host = host_header
    # The port server[1] will be None for a Unix socket. Ignore in that case.
    elif server is not None and server[1] is not None:
        host = server[0]

        # If SERVER_NAME is IPv6, wrap it in [] to match Host header.
        # Check for : because domain or IPv4 can't have that.
        if ":" in host and host[0] != "[":
            host = f"[{host}]"

        host = f"{host}:{server[1]}"
    else:
        # Pass through empty host from HTTP/0.9 and 1.0.
        return ""

    if scheme in {"http", "ws"}:
        host = host.removesuffix(":80")
    elif scheme in {"https", "wss"}:
        host = host.removesuffix(":443")

    if not host_is_trusted(host, trusted_hosts):
        if trusted_hosts:
            raise SecurityError(f"Host {host!r} is not trusted.")

        # Invalid characters, treat as empty.
        return ""

    return host


def get_current_url(
    scheme: str,
    host: str,
    root_path: str | None = None,
    path: str | None = None,
    query_string: bytes | None = None,
) -> str:
    """Recreate the URL for a request. If an optional part isn't
    provided, it and subsequent parts are not included in the URL.

    The URL is an IRI, not a URI, so it may contain Unicode characters.
    Use :func:`~werkzeug.urls.iri_to_uri` to convert it to ASCII.

    :param scheme: The protocol the request used, like ``"https"``.
    :param host: The host the request was made to. See :func:`get_host`.
    :param root_path: Prefix that the application is mounted under. This
        is prepended to ``path``.
    :param path: The path part of the URL after ``root_path``.
    :param query_string: The portion of the URL after the "?".
    """
    url = [scheme, "://", host]

    if root_path is None:
        url.append("/")
        return uri_to_iri("".join(url))

    # safe = https://url.spec.whatwg.org/#url-path-segment-string
    # as well as percent for things that are already quoted
    url.append(quote(root_path.rstrip("/"), safe="!$&'()*+,/:;=@%"))
    url.append("/")

    if path is None:
        return uri_to_iri("".join(url))

    url.append(quote(path.lstrip("/"), safe="!$&'()*+,/:;=@%"))

    if query_string:
        url.append("?")
        url.append(quote(query_string, safe="!$&'()*+,/:;=?@%"))

    return uri_to_iri("".join(url))


def get_content_length(
    http_content_length: str | None = None,
    http_transfer_encoding: str | None = None,
) -> int | None:
    """Return the ``Content-Length`` header value as an int. If the header is not given
    or the ``Transfer-Encoding`` header is ``chunked``, ``None`` is returned to indicate
    a streaming request. If the value is not an integer, or negative, 0 is returned.

    :param http_content_length: The Content-Length HTTP header.
    :param http_transfer_encoding: The Transfer-Encoding HTTP header.

    .. versionadded:: 2.2
    """
    if (
        http_transfer_encoding is not None
        and "chunked" in parse_set_header(http_transfer_encoding)
    ) or http_content_length is None:
        return None

    try:
        return max(0, _plain_int(http_content_length))
    except ValueError:
        return 0
<|endoftext|>from __future__ import annotations

import re
import typing as t
from datetime import datetime

from .._internal import _dt_as_utc
from ..http import generate_etag
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import unquote_etag


def is_resource_modified(
    http_range: str | None = None,
    http_if_range: str | None = None,
    http_if_modified_since: str | None = None,
    http_if_none_match: str | None = None,
    http_if_match: str | None = None,
    etag: str | None = None,
    data: bytes | None = None,
    last_modified: datetime | str | None = None,
    ignore_if_range: bool = True,
) -> bool:
    """Convenience method for conditional requests.
    :param http_range: Range HTTP header
    :param http_if_range: If-Range HTTP header
    :param http_if_modified_since: If-Modified-Since HTTP header
    :param http_if_none_match: If-None-Match HTTP header
    :param http_if_match: If-Match HTTP header
    :param etag: the etag for the response for comparison.
    :param data: or alternatively the data of the response to automatically
                 generate an etag using :func:`generate_etag`.
    :param last_modified: an optional date of the last modification.
    :param ignore_if_range: If `False`, `If-Range` header will be taken into
                            account.
    :return: `True` if the resource was modified, otherwise `False`.

    .. versionadded:: 2.2
    """
    if etag is None and data is not None:
        etag = f'"{generate_etag(data)}"'
    elif data is not None:
        raise TypeError("both data and etag given")

    unmodified = False
    if isinstance(last_modified, str):
        last_modified = parse_date(last_modified)

    # HTTP doesn't use microsecond, remove it to avoid false positive
    # comparisons. Mark naive datetimes as UTC.
    if last_modified is not None:
        last_modified = _dt_as_utc(last_modified.replace(microsecond=0))

    if_range = None
    if not ignore_if_range and http_range is not None:
        # https://tools.ietf.org/html/rfc7233#section-3.2
        # A server MUST ignore an If-Range header field received in a request
        # that does not contain a Range header field.
        if_range = parse_if_range_header(http_if_range)

    if if_range is not None and if_range.date is not None:
        modified_since: datetime | None = if_range.date
    else:
        modified_since = parse_date(http_if_modified_since)

    if modified_since and last_modified and last_modified <= modified_since:
        unmodified = True

    if etag:
        etag, _ = unquote_etag(etag)

        if if_range is not None and if_range.etag is not None:
            unmodified = if_range.etag == etag
        else:
            if_none_match = parse_etags(http_if_none_match)
            if if_none_match:
                # https://tools.ietf.org/html/rfc7232#section-3.2
                # "A recipient MUST use the weak comparison function when comparing
                # entity-tags for If-None-Match"
                unmodified = if_none_match.contains_weak(etag)

            # https://tools.ietf.org/html/rfc7232#section-3.1
            # "Origin server MUST use the strong comparison function when
            # comparing entity-tags for If-Match"
            if_match = parse_etags(http_if_match)
            if if_match:
                unmodified = not if_match.is_strong(etag)

    return not unmodified


_cookie_re = re.compile(
    r"""
    [ \t]*  # ignore leading space
    ([^ \t=";]+)  # key
    (?:[ \t]*=[ \t]*  # optional =value, ignoring invalid space
        (
            "(?:[^\\"]|\\.)*"  # quoted value with backslash escapes
        |
            [^ \t";]*  # token value
        )
    )?
    [ \t]*  # ignore trailing space
    (?:;|\Z)  # only if followed by semicolon or end
    """,
    flags=re.ASCII | re.VERBOSE,
)
_cookie_unslash_re = re.compile(rb"\\([0-3][0-7]{2}|.)")


def _cookie_unslash_replace(m: t.Match[bytes]) -> bytes:
    v = m.group(1)

    if len(v) == 1:
        return v

    return int(v, 8).to_bytes(1, "big")


def parse_cookie(
    cookie: str | None = None,
    cls: type[ds.MultiDict[str, str]] | None = None,
) -> ds.MultiDict[str, str]:
    """Parse a cookie from a string.

    The same key can be provided multiple times, the values are stored
    in-order. The default :class:`MultiDict` will have the first value
    first, and all values can be retrieved with
    :meth:`MultiDict.getlist`.

    :param cookie: The cookie header as a string.
    :param cls: A dict-like class to store the parsed cookies in.
        Defaults to :class:`MultiDict`.

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` and ``errors`` parameters, were removed.

    .. versionadded:: 2.2
    """
    if cls is None:
        cls = t.cast("type[ds.MultiDict[str, str]]", ds.MultiDict)

    if not cookie:
        return cls()

    out = []
    pos = 0

    while True:
        if (m := _cookie_re.match(cookie, pos)) is None:
            # Skip invalid chars until the next semicolon.
            if (pos := cookie.find(";", pos) + 1) == 0:
                break

            continue

        ck, cv = m.groups()
        pos = m.end()

        if cv is None:
            cv = ""

        if cv.startswith('"') and cv.endswith('"'):
            # Work with bytes here, since a UTF-8 character could be multiple bytes.
            cv = _cookie_unslash_re.sub(
                _cookie_unslash_replace, cv[1:-1].encode()
            ).decode(errors="replace")

        out.append((ck, cv))

    return cls(out)


# circular dependencies
from .. import datastructures as ds  # noqa: E402
<|endoftext|>from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass
from enum import auto
from enum import Enum

from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header


class Event:
    pass


@dataclass(frozen=True)
class Preamble(Event):
    data: bytes


@dataclass(frozen=True)
class Field(Event):
    name: str
    headers: Headers


@dataclass(frozen=True)
class File(Event):
    name: str
    filename: str
    headers: Headers


@dataclass(frozen=True)
class Data(Event):
    data: bytes
    more_data: bool


@dataclass(frozen=True)
class Epilogue(Event):
    data: bytes


class NeedData(Event):
    pass


NEED_DATA = NeedData()


class State(Enum):
    PREAMBLE = auto()
    PART = auto()
    DATA = auto()
    DATA_START = auto()
    EPILOGUE = auto()
    COMPLETE = auto()


# Multipart line breaks MUST be CRLF (\r\n) by RFC-7578, except that
# many implementations break this and either use CR or LF alone.
LINE_BREAK = b"(?:\r\n|\n|\r)"
BLANK_LINE_RE = re.compile(b"(?:\r\n\r\n|\r\r|\n\n)", re.MULTILINE)
LINE_BREAK_RE = re.compile(LINE_BREAK, re.MULTILINE)
# Header values can be continued via a space or tab after the linebreak, as
# per RFC2231
HEADER_CONTINUATION_RE = re.compile(b"%s[ \t]" % LINE_BREAK, re.MULTILINE)
# This must be long enough to contain any line breaks plus any
# additional boundary markers (--) such that they will be found in a
# subsequent search
SEARCH_EXTRA_LENGTH = 8


class MultipartDecoder:
    """Decodes a multipart message as bytes into Python events.

    The part data is returned as available to allow the caller to save
    the data from memory to disk, if desired.

    .. versionchanged:: 3.1.4
        Handle chunks that split a``\r\n`` sequence.
    """

    def __init__(
        self,
        boundary: bytes,
        max_form_memory_size: int | None = None,
        *,
        max_parts: int | None = None,
    ) -> None:
        self.buffer = bytearray()
        self.complete = False
        self.max_form_memory_size = max_form_memory_size
        self.max_parts = max_parts
        self.state = State.PREAMBLE
        self.boundary = boundary

        # Note in the below \h i.e. horizontal whitespace is used
        # as [^\S\n\r] as \h isn't supported in python.

        # The preamble must end with a boundary where the boundary is
        # prefixed by a line break, RFC2046. Except that many
        # implementations including Werkzeug's tests omit the line
        # break prefix. In addition the first boundary could be the
        # epilogue boundary (for empty form-data) hence the matching
        # group to understand if it is an epilogue boundary.
        self.preamble_re = re.compile(
            rb"%s?--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )
        # A boundary must include a line break prefix and suffix, and
        # may include trailing whitespace. In addition the boundary
        # could be the epilogue boundary hence the matching group to
        # understand if it is an epilogue boundary.
        self.boundary_re = re.compile(
            rb"%s--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )
        self._search_position = 0
        self._parts_decoded = 0

    def receive_data(self, data: bytes | None) -> None:
        if data is None:
            self.complete = True
        elif (
            self.max_form_memory_size is not None
            and len(self.buffer) + len(data) > self.max_form_memory_size
        ):
            # Ensure that data within single event does not exceed limit.
            # Also checked across accumulated events in MultiPartParser.
            raise RequestEntityTooLarge()
        else:
            self.buffer.extend(data)

    def next_event(self) -> Event:
        event: Event = NEED_DATA
        if self.state == State.PREAMBLE:
            match = self.preamble_re.search(self.buffer, self._search_position)
            if match is not None:
                if match.group(1).startswith(b"--"):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data = bytes(self.buffer[: match.start()])
                del self.buffer[: match.end()]
                event = Preamble(data=data)
                self._search_position = 0
            else:
                # Update the search start position to be equal to the
                # current buffer length (already searched) minus a
                # safe buffer for part of the search target.
                self._search_position = max(
                    0, len(self.buffer) - len(self.boundary) - SEARCH_EXTRA_LENGTH
                )

        elif self.state == State.PART:
            match = BLANK_LINE_RE.search(self.buffer, self._search_position)
            if match is not None:
                headers = self._parse_headers(self.buffer[: match.start()])
                # The final header ends with a single CRLF, however a
                # blank line indicates the start of the
                # body. Therefore the end is after the first CRLF.
                headers_end = (match.start() + match.end()) // 2
                del self.buffer[:headers_end]

                if "content-disposition" not in headers:
                    raise ValueError("Missing Content-Disposition header")

                disposition, extra = parse_options_header(
                    headers["content-disposition"]
                )
                name = t.cast(str, extra.get("name"))
                filename = extra.get("filename")
                if filename is not None:
                    event = File(
                        filename=filename,
                        headers=headers,
                        name=name,
                    )
                else:
                    event = Field(
                        headers=headers,
                        name=name,
                    )
                self.state = State.DATA_START
                self._search_position = 0
                self._parts_decoded += 1

                if self.max_parts is not None and self._parts_decoded > self.max_parts:
                    raise RequestEntityTooLarge()
            else:
                # Update the search start position to be equal to the
                # current buffer length (already searched) minus a
                # safe buffer for part of the search target.
                self._search_position = max(0, len(self.buffer) - SEARCH_EXTRA_LENGTH)

        elif self.state == State.DATA_START:
            data, del_index, more_data = self._parse_data(self.buffer, start=True)
            del self.buffer[:del_index]
            event = Data(data=data, more_data=more_data)
            if more_data:
                self.state = State.DATA

        elif self.state == State.DATA:
            data, del_index, more_data = self._parse_data(self.buffer, start=False)
            del self.buffer[:del_index]
            if data or not more_data:
                event = Data(data=data, more_data=more_data)

        elif self.state == State.EPILOGUE and self.complete:
            event = Epilogue(data=bytes(self.buffer))
            del self.buffer[:]
            self.state = State.COMPLETE

        if self.complete and isinstance(event, NeedData):
            raise ValueError(f"Invalid form-data cannot parse beyond {self.state}")

        return event

    def _parse_headers(self, data: bytes | bytearray) -> Headers:
        headers: list[tuple[str, str]] = []
        # Merge the continued headers into one line
        data = HEADER_CONTINUATION_RE.sub(b" ", data)
        # Now there is one header per line
        for line in data.splitlines():
            line = line.strip(b" \t")

            if line != b"":
                name, _, value = line.decode().partition(":")
                headers.append((name.strip(" \t"), value.strip(" \t")))
        return Headers(headers)

    def _parse_data(
        self, data: bytes | bytearray, *, start: bool
    ) -> tuple[bytes, int, bool]:
        # Body parts must start with CRLF (or CR or LF)
        if start:
            match = LINE_BREAK_RE.match(data)
            data_start = t.cast(t.Match[bytes], match).end()
        else:
            data_start = 0

        if self.buffer.find(b"--" + self.boundary) == -1:
            # No complete boundary in the buffer, but there may be
            # a partial boundary at the end.
            data_end = del_index = (
                self._last_partial_boundary_index(data[data_start:]) + data_start
            )
            more_data = True
        else:
            match = self.boundary_re.search(data)
            if match is not None:
                if match.group(1).startswith(b"--"):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data_end = match.start()
                del_index = match.end()
            else:
                data_end = del_index = (
                    self._last_partial_boundary_index(data[data_start:]) + data_start
                )
            more_data = match is None
        return bytes(data[data_start:data_end]), del_index, more_data

    def _last_partial_boundary_index(self, data: bytes | bytearray) -> int:
        # Find the last index following which a partial boundary
        # could be present in the data. This will be the earliest
        # position of a LF or a CR, unless that position is more
        # than a complete boundary from the end in which case there
        # is no partial boundary.
        complete_boundary_index = len(data) - len(b"\r\n--" + self.boundary)
        try:
            last_nl = data.rindex(b"\n")
        except ValueError:
            last_nl = len(data)
        else:
            if last_nl < complete_boundary_index:
                last_nl = len(data)
        try:
            last_cr = data.rindex(b"\r")
        except ValueError:
            last_cr = len(data)
        else:
            if last_cr < complete_boundary_index:
                last_cr = len(data)
        return min(last_nl, last_cr)


class MultipartEncoder:
    """Used by the test client to create a multipart body. This is not secure
    and must only be used during testing, never in applications.

    :meta private:
    """

    def __init__(self, boundary: bytes) -> None:
        self.boundary = boundary
        self.state = State.PREAMBLE

    def send_event(self, event: Event) -> bytes:
        if isinstance(event, Preamble) and self.state == State.PREAMBLE:
            self.state = State.PART
            return event.data
        elif isinstance(event, (Field, File)) and self.state in {
            State.PREAMBLE,
            State.PART,
            State.DATA,
        }:
            data = b"\r\n--" + self.boundary + b"\r\n"
            data += b'Content-Disposition: form-data; name="%s"' % event.name.encode()
            if isinstance(event, File):
                data += b'; filename="%s"' % event.filename.encode()
            data += b"\r\n"
            for name, value in t.cast(Field, event).headers:
                if name.lower() != "content-disposition":
                    data += f"{name}: {value}\r\n".encode()
            self.state = State.DATA_START
            return data
        elif isinstance(event, Data) and self.state == State.DATA_START:
            self.state = State.DATA
            if len(event.data) > 0:
                return b"\r\n" + event.data
            else:
                return event.data
        elif isinstance(event, Data) and self.state == State.DATA:
            return event.data
        elif isinstance(event, Epilogue):
            self.state = State.COMPLETE
            return b"\r\n--" + self.boundary + b"--\r\n" + event.data
        else:
            raise ValueError(f"Cannot generate {event} in state: {self.state}")
<|endoftext|>from __future__ import annotations

import typing as t
from datetime import datetime
from urllib.parse import parse_qsl

from ..datastructures import Accept
from ..datastructures import Authorization
from ..datastructures import CharsetAccept
from ..datastructures import ETags
from ..datastructures import Headers
from ..datastructures import HeaderSet
