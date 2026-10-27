    global default timeout setting returned by :func:`socket.getdefaulttimeout`
    is used.  If *source_address* is set it must be a tuple of (host, port)
    for the socket to bind as a source address before making the connection.
    An host of '' or port 0 tells the OS to use the default.
    """

    host, port = address
    if host.startswith("["):
        host = host.strip("[]")
    err = None

    # Using the value from allowed_gai_family() in the context of getaddrinfo lets
    # us select whether to work with IPv4 DNS records, IPv6 records, or both.
    # The original create_connection function always returns all records.
    family = allowed_gai_family()

    try:
        host.encode("idna")
    except UnicodeError:
        raise LocationParseError(f"'{host}', label empty or too long") from None

    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)

            # If provided, set socket level options before connecting.
            _set_socket_options(sock, socket_options)

            if timeout is not _DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            # Break explicitly a reference cycle
            err = None
            return sock

        except OSError as _:
            err = _
            if sock is not None:
                sock.close()

    if err is not None:
        try:
            raise err
        finally:
            # Break explicitly a reference cycle
            err = None
    else:
        raise OSError("getaddrinfo returns an empty list")


def _set_socket_options(
    sock: socket.socket, options: _TYPE_SOCKET_OPTIONS | None
) -> None:
    if options is None:
        return

    for opt in options:
        sock.setsockopt(*opt)


def allowed_gai_family() -> socket.AddressFamily:
    """This function is designed to work in the context of
    getaddrinfo, where family=socket.AF_UNSPEC is the default and
    will perform a DNS search for both IPv6 and IPv4 records."""

    family = socket.AF_INET
    if HAS_IPV6:
        family = socket.AF_UNSPEC
    return family


def _has_ipv6(host: str) -> bool:
    """Returns True if the system can bind an IPv6 address."""
    sock = None
    has_ipv6 = False

    if socket.has_ipv6:
        # has_ipv6 returns true if cPython was compiled with IPv6 support.
        # It does not tell us if the system has IPv6 support enabled. To
        # determine that we must bind to an IPv6 address.
        # https://github.com/urllib3/urllib3/pull/611
        # https://bugs.python.org/issue658327
        try:
            sock = socket.socket(socket.AF_INET6)
            sock.bind((host, 0))
            has_ipv6 = True
        except Exception:
            pass

    if sock:
        sock.close()
    return has_ipv6


HAS_IPV6 = _has_ipv6("::1")
<|endoftext|>from __future__ import annotations

import select
import socket
from functools import partial

__all__ = ["wait_for_read", "wait_for_write"]


# How should we wait on sockets?
#
# There are two types of APIs you can use for waiting on sockets: the fancy
# modern stateful APIs like epoll/kqueue, and the older stateless APIs like
# select/poll. The stateful APIs are more efficient when you have a lots of
# sockets to keep track of, because you can set them up once and then use them
# lots of times. But we only ever want to wait on a single socket at a time
# and don't want to keep track of state, so the stateless APIs are actually
# more efficient. So we want to use select() or poll().
#
# Now, how do we choose between select() and poll()? On traditional Unixes,
# select() has a strange calling convention that makes it slow, or fail
# altogether, for high-numbered file descriptors. The point of poll() is to fix
# that, so on Unixes, we prefer poll().
#
# On Windows, there is no poll() (or at least Python doesn't provide a wrapper
# for it), but that's OK, because on Windows, select() doesn't have this
# strange calling convention; plain select() works fine.
#
# So: on Windows we use select(), and everywhere else we use poll(). We also
# fall back to select() in case poll() is somehow broken or missing.


def select_wait_for_socket(
    sock: socket.socket,
    read: bool = False,
    write: bool = False,
    timeout: float | None = None,
) -> bool:
    if not read and not write:
        raise RuntimeError("must specify at least one of read=True, write=True")
    rcheck = []
    wcheck = []
    if read:
        rcheck.append(sock)
    if write:
        wcheck.append(sock)
    # When doing a non-blocking connect, most systems signal success by
    # marking the socket writable. Windows, though, signals success by marked
    # it as "exceptional". We paper over the difference by checking the write
    # sockets for both conditions. (The stdlib selectors module does the same
    # thing.)
    fn = partial(select.select, rcheck, wcheck, wcheck)
    rready, wready, xready = fn(timeout)
    return bool(rready or wready or xready)


def poll_wait_for_socket(
    sock: socket.socket,
    read: bool = False,
    write: bool = False,
    timeout: float | None = None,
) -> bool:
    if not read and not write:
        raise RuntimeError("must specify at least one of read=True, write=True")
    mask = 0
    if read:
        mask |= select.POLLIN
    if write:
        mask |= select.POLLOUT
    poll_obj = select.poll()
    poll_obj.register(sock, mask)

    # For some reason, poll() takes timeout in milliseconds
    def do_poll(t: float | None) -> list[tuple[int, int]]:
        if t is not None:
            t *= 1000
        return poll_obj.poll(t)

    return bool(do_poll(timeout))


def _have_working_poll() -> bool:
    # Apparently some systems have a select.poll that fails as soon as you try
    # to use it, either due to strange configuration or broken monkeypatching
    # from libraries like eventlet/greenlet.
    try:
        poll_obj = select.poll()
        poll_obj.poll(0)
    except (AttributeError, OSError):
        return False
    else:
        return True


def wait_for_socket(
    sock: socket.socket,
    read: bool = False,
    write: bool = False,
    timeout: float | None = None,
) -> bool:
    # We delay choosing which implementation to use until the first time we're
    # called. We could do it at import time, but then we might make the wrong
    # decision if someone goes wild with monkeypatching select.poll after
    # we're imported.
    global wait_for_socket
    if _have_working_poll():
        wait_for_socket = poll_wait_for_socket
    elif hasattr(select, "select"):
        wait_for_socket = select_wait_for_socket
    return wait_for_socket(sock, read, write, timeout)


def wait_for_read(sock: socket.socket, timeout: float | None = None) -> bool:
    """Waits for reading to be available on a given socket.
    Returns True if the socket is readable, or False if the timeout expired.
    """
    return wait_for_socket(sock, read=True, timeout=timeout)


def wait_for_write(sock: socket.socket, timeout: float | None = None) -> bool:
    """Waits for writing to be available on a given socket.
    Returns True if the socket is readable, or False if the timeout expired.
    """
    return wait_for_socket(sock, write=True, timeout=timeout)
<|endoftext|>from __future__ import annotations

import io
import sys
import typing
from base64 import b64encode
from enum import Enum

from ..exceptions import UnrewindableBodyError
from .util import to_bytes

if typing.TYPE_CHECKING:
    from typing import Final

# Pass as a value within ``headers`` to skip
# emitting some HTTP headers that are added automatically.
# The only headers that are supported are ``Accept-Encoding``,
# ``Host``, and ``User-Agent``.
SKIP_HEADER = "@@@SKIP_HEADER@@@"
SKIPPABLE_HEADERS = frozenset(["accept-encoding", "host", "user-agent"])

ACCEPT_ENCODING = "gzip,deflate"
try:
    try:
        import brotlicffi as _unused_module_brotli  # type: ignore[import-not-found] # noqa: F401
    except ImportError:
        import brotli as _unused_module_brotli  # type: ignore[import-not-found] # noqa: F401
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ",br"

try:
    if sys.version_info >= (3, 14):
        from compression import zstd as _unused_module_zstd  # noqa: F401
    else:
        from backports import zstd as _unused_module_zstd  # noqa: F401
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ",zstd"


class _TYPE_FAILEDTELL(Enum):
    token = 0


_FAILEDTELL: Final[_TYPE_FAILEDTELL] = _TYPE_FAILEDTELL.token

_TYPE_BODY_POSITION = typing.Union[int, _TYPE_FAILEDTELL]

# When sending a request with these methods we aren't expecting
# a body so don't need to set an explicit 'Content-Length: 0'
# The reason we do this in the negative instead of tracking methods
# which 'should' have a body is because unknown methods should be
# treated as if they were 'POST' which *does* expect a body.
_METHODS_NOT_EXPECTING_BODY = {"GET", "HEAD", "DELETE", "TRACE", "OPTIONS", "CONNECT"}


def make_headers(
    keep_alive: bool | None = None,
    accept_encoding: bool | list[str] | str | None = None,
    user_agent: str | None = None,
    basic_auth: str | None = None,
    proxy_basic_auth: str | None = None,
    disable_cache: bool | None = None,
    *,
    basic_auth_encoding: str = "latin-1",
    proxy_basic_auth_encoding: str = "latin-1",
) -> dict[str, str]:
    """
    Shortcuts for generating request headers.

    :param keep_alive:
        If ``True``, adds 'connection: keep-alive' header.

    :param accept_encoding:
        Can be a boolean, list, or string.
        ``True`` translates to 'gzip,deflate'.  If the dependencies for
        Brotli (either the ``brotli`` or ``brotlicffi`` package) and/or
        Zstandard (the ``backports.zstd`` package for Python before 3.14)
        algorithms are installed, then their encodings are
        included in the string ('br' and 'zstd', respectively).
        List will get joined by comma.
        String will be used as provided.

    :param user_agent:
        String representing the user-agent you want, such as
        "python-urllib3/0.6"

    :param basic_auth:
        Colon-separated username:password string for 'authorization: basic ...'
        auth header. Basic authentication transmits Base64-encoded bytes, not
        Unicode text, so the string is encoded with ``basic_auth_encoding``
        before Base64 encoding.

        Must contain decoded credentials (not percent-encoded).
        For URLs parsed with :func:`parse_url`, use :attr:`Url.auth_decoded_joined`
        to get the properly formatted string.

    :param basic_auth_encoding:
        Text encoding used to encode ``basic_auth`` before Base64 encoding.
        Defaults to "latin-1" for backward compatibility. Use "utf-8" only
        when the peer expects UTF-8 Basic authentication credentials.

        Must contain decoded credentials (not percent-encoded).
        For URLs parsed with :func:`parse_url`, use :attr:`Url.auth_decoded_joined`
        to get the properly formatted string.

    :param proxy_basic_auth:
        Colon-separated username:password string for 'proxy-authorization: basic ...'
        auth header. Basic authentication transmits Base64-encoded bytes, not
        Unicode text, so the string is encoded with ``proxy_basic_auth_encoding``
        before Base64 encoding.

        Must contain decoded credentials (not percent-encoded).
        For URLs parsed with :func:`parse_url`, use :attr:`Url.auth_decoded_joined`
        to get the properly formatted string.

    :param proxy_basic_auth_encoding:
        Text encoding used to encode ``proxy_basic_auth`` before Base64
        encoding. Defaults to "latin-1" for backward compatibility. Use
        "utf-8" only when the proxy expects UTF-8 Basic authentication
        credentials.

        Must contain decoded credentials (not percent-encoded).
        For URLs parsed with :func:`parse_url`, use :attr:`Url.auth_decoded_joined`
        to get the properly formatted string.

    :param disable_cache:
        If ``True``, adds 'cache-control: no-cache' header.

    Example:

    .. code-block:: python

        import urllib3

        print(urllib3.util.make_headers(keep_alive=True, user_agent="Batman/1.0"))
        # {'connection': 'keep-alive', 'user-agent': 'Batman/1.0'}
        print(urllib3.util.make_headers(accept_encoding=True))
        # {'accept-encoding': 'gzip,deflate'}
    """
    headers: dict[str, str] = {}
    if accept_encoding:
        if isinstance(accept_encoding, str):
            pass
        elif isinstance(accept_encoding, list):
            accept_encoding = ",".join(accept_encoding)
        else:
            accept_encoding = ACCEPT_ENCODING
        headers["accept-encoding"] = accept_encoding

    if user_agent:
        headers["user-agent"] = user_agent

    if keep_alive:
        headers["connection"] = "keep-alive"

    if basic_auth:
        headers["authorization"] = (
            f"Basic {b64encode(basic_auth.encode(basic_auth_encoding)).decode()}"
        )

    if proxy_basic_auth:
        headers["proxy-authorization"] = (
            f"Basic {b64encode(proxy_basic_auth.encode(proxy_basic_auth_encoding)).decode()}"
        )

    if disable_cache:
        headers["cache-control"] = "no-cache"

    return headers


def set_file_position(
    body: typing.Any, pos: _TYPE_BODY_POSITION | None
) -> _TYPE_BODY_POSITION | None:
    """
    If a position is provided, move file to that point.
    Otherwise, we'll attempt to record a position for future use.
    """
    if pos is not None:
        rewind_body(body, pos)
    elif getattr(body, "tell", None) is not None:
        try:
            pos = body.tell()
        except OSError:
            # This differentiates from None, allowing us to catch
            # a failed `tell()` later when trying to rewind the body.
            pos = _FAILEDTELL

    return pos


def rewind_body(body: typing.IO[typing.AnyStr], body_pos: _TYPE_BODY_POSITION) -> None:
    """
    Attempt to rewind body to a certain position.
    Primarily used for request redirects and retries.

    :param body:
        File-like object that supports seek.

    :param int pos:
        Position to seek to in file.
    """
    body_seek = getattr(body, "seek", None)
    if body_seek is not None and isinstance(body_pos, int):
        try:
            body_seek(body_pos)
        except OSError as e:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect/retry."
            ) from e
    elif body_pos is _FAILEDTELL:
        raise UnrewindableBodyError(
            "Unable to record file position for rewinding "
            "request body during a redirect/retry."
        )
    elif body_seek is None:
        raise UnrewindableBodyError("body does not implement seek.")
    else:
        raise ValueError(
            f"body_pos must be of type integer, instead it was {type(body_pos)}."
        )


class ChunksAndContentLength(typing.NamedTuple):
    chunks: typing.Iterable[bytes] | None
    content_length: int | None


def body_to_chunks(
    body: typing.Any | None, method: str, blocksize: int
) -> ChunksAndContentLength:
    """Takes the HTTP request method, body, and blocksize and
    transforms them into an iterable of chunks to pass to
    socket.sendall() and an optional 'Content-Length' header.

    A 'Content-Length' of 'None' indicates the length of the body
    can't be determined so should use 'Transfer-Encoding: chunked'
    for framing instead.
    """

    chunks: typing.Iterable[bytes] | None
    content_length: int | None

    # No body, we need to make a recommendation on 'Content-Length'
    # based on whether that request method is expected to have
    # a body or not.
    if body is None:
        chunks = None
        if method.upper() not in _METHODS_NOT_EXPECTING_BODY:
            content_length = 0
        else:
            content_length = None

    # Bytes or strings become bytes
    elif isinstance(body, (str, bytes)):
        chunks = (to_bytes(body),)
        content_length = len(chunks[0])

    # File-like object, TODO: use seek() and tell() for length?
    elif hasattr(body, "read"):

        def chunk_readable() -> typing.Iterable[bytes]:
            encode = isinstance(body, io.TextIOBase)
            while True:
                datablock = body.read(blocksize)
                if not datablock:
                    break
                if encode:
                    datablock = datablock.encode("utf-8")
                yield datablock

        chunks = chunk_readable()
        content_length = None

    # Otherwise we need to start checking via duck-typing.
    else:
        try:
            # Check if the body implements the buffer API.
            mv = memoryview(body)
        except TypeError:
            try:
                # Check if the body is an iterable
                chunks = iter(body)
                content_length = None
            except TypeError:
                raise TypeError(
                    f"'body' must be a bytes-like object, file-like "
                    f"object, or iterable. Instead was {body!r}"
                ) from None
        else:
            # Since it implements the buffer API can be passed directly to socket.sendall()
            chunks = (body,)
            content_length = mv.nbytes

    return ChunksAndContentLength(chunks=chunks, content_length=content_length)
<|endoftext|>from __future__ import annotations

import email
import logging
import random
import re
import time
import typing
import warnings
from collections.abc import Collection
from itertools import takewhile
from types import TracebackType

from ..exceptions import (
    ConnectTimeoutError,
    InvalidHeader,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    ResponseError,
)
from .util import reraise

if typing.TYPE_CHECKING:
    from typing_extensions import Self

    from ..connectionpool import ConnectionPool
    from ..response import BaseHTTPResponse

log = logging.getLogger(__name__)


# Data structure for representing the metadata of requests that result in a retry.
class RequestHistory(typing.NamedTuple):
    method: str | None
    url: str | None
    error: Exception | None
    status: int | None
    redirect_location: str | None


class Retry:
    """Retry configuration.

    Each retry attempt will create a new Retry object with updated values, so
    they can be safely reused.

    Retries can be defined as a default for a pool:

    .. code-block:: python

        retries = Retry(connect=5, read=2, redirect=5)
        http = PoolManager(retries=retries)
        response = http.request("GET", "https://example.com/")

    Or per-request (which overrides the default for the pool):

    .. code-block:: python

        response = http.request("GET", "https://example.com/", retries=Retry(10))

    Retries can be disabled by passing ``False``:

    .. code-block:: python

        response = http.request("GET", "https://example.com/", retries=False)

    Errors will be wrapped in :class:`~urllib3.exceptions.MaxRetryError` unless
    retries are disabled, in which case the causing exception will be raised.

    :param int total:
        Total number of retries to allow. Takes precedence over other counts.

        Set to ``None`` to remove this constraint and fall back on other
        counts.

        Set to ``0`` to fail on the first retry.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param int connect:
        How many connection-related errors to retry on.

        These are errors raised before the request is sent to the remote server,
        which we assume has not triggered the server to process the request.

        Set to ``0`` to fail on the first retry of this type.

    :param int read:
        How many times to retry on read errors.

        These errors are raised after the request was sent to the server, so the
        request may have side-effects.

        Set to ``0`` to fail on the first retry of this type.

    :param int redirect:
        How many redirects to perform. Limit this to avoid infinite redirect
        loops.

        A redirect is a HTTP response with a status code 301, 302, 303, 307 or
        308.

        Set to ``0`` to fail on the first retry of this type.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param int status:
        How many times to retry on bad status codes.

        These are retries made on responses, where status code matches
        ``status_forcelist``.

        Set to ``0`` to fail on the first retry of this type.

    :param int other:
        How many times to retry on other errors.

        Other errors are errors that are not connect, read, redirect or status errors.
        These errors might be raised after the request was sent to the server, so the
        request might have side-effects.

        Set to ``0`` to fail on the first retry of this type.

        If ``total`` is not set, it's a good idea to set this to 0 to account
        for unexpected edge cases and avoid infinite retry loops.

    :param Collection allowed_methods:
        Set of uppercased HTTP method verbs that we should retry on.

        By default, we only retry on methods which are considered to be
        idempotent (multiple requests with the same parameters end with the
        same state). See :attr:`Retry.DEFAULT_ALLOWED_METHODS`.

        Set to a ``None`` value to retry on any verb.

    :param Collection status_forcelist:
        A set of integer HTTP status codes that we should force a retry on.
        A retry is initiated if the request method is in ``allowed_methods``
        and the response status code is in ``status_forcelist``.

        By default, this is disabled with ``None``.

    :param float backoff_factor:
        A backoff factor to apply between attempts after the second try
        (most errors are resolved immediately by a second try without a
        delay). urllib3 will sleep for::

            {backoff factor} * (2 ** ({number of previous retries}))

        seconds. If `backoff_jitter` is non-zero, this sleep is extended by::

            random.uniform(0, {backoff jitter})

        seconds. For example, if the backoff_factor is 0.1, then :func:`Retry.sleep` will
        sleep for [0.0s, 0.2s, 0.4s, 0.8s, ...] between retries. No backoff will ever
        be longer than `backoff_max`.

        By default, backoff is disabled (factor set to 0).

    :param float backoff_max:
        The maximum backoff time (in seconds) between retry attempts.
        This value caps the computed backoff from `backoff_factor`.

    :param float backoff_jitter:
        Random jitter amount (in seconds) added to the computed backoff.
        Jitter is sampled uniformly from `0` to `backoff_jitter`.

    :param bool raise_on_redirect: Whether, if the number of redirects is
        exhausted, to raise a MaxRetryError, or to return a response with a
        response code in the 3xx range.

    :param bool raise_on_status: Similar meaning to ``raise_on_redirect``:
        whether we should raise an exception, or return a response,
        if status falls in ``status_forcelist`` range and retries have
        been exhausted.

    :param tuple history: The history of the request encountered during
        each call to :meth:`~Retry.increment`. The list is in the order
        the requests occurred. Each list item is of class :class:`RequestHistory`.

    :param bool respect_retry_after_header:
        Whether to respect Retry-After header on status codes defined as
        :attr:`Retry.RETRY_AFTER_STATUS_CODES` or not.

    :param Collection remove_headers_on_redirect:
        Sequence of headers to remove from the request when a response
        indicating a redirect is returned before firing off the redirected
        request.

    :param int retry_after_max: Number of seconds to allow as the maximum for
        Retry-After headers. Defaults to :attr:`Retry.DEFAULT_RETRY_AFTER_MAX`.
        Any Retry-After headers larger than this value will be limited to this
        value.
    """

    #: Default methods to be used for ``allowed_methods``
    DEFAULT_ALLOWED_METHODS = frozenset(
        ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )

    #: Default status codes that trigger a retry when a Retry-After header is
    #: present and ``respect_retry_after_header`` is enabled.
    RETRY_AFTER_STATUS_CODES = frozenset([413, 429, 503])

    #: Default headers to be used for ``remove_headers_on_redirect``
    DEFAULT_REMOVE_HEADERS_ON_REDIRECT = frozenset(
        ["Cookie", "Authorization", "Proxy-Authorization"]
    )

    #: Default maximum backoff time.
    DEFAULT_BACKOFF_MAX = 120

    # This is undocumented in the RFC. Setting to 6 hours matches other popular libraries.
    #: Default maximum allowed value for Retry-After headers in seconds
    DEFAULT_RETRY_AFTER_MAX: typing.Final[int] = 21600

    # Backward compatibility; assigned outside of the class.
    DEFAULT: typing.ClassVar[Retry]

    def __init__(
        self,
        total: bool | int | None = 10,
        connect: int | None = None,
        read: int | None = None,
        redirect: bool | int | None = None,
        status: int | None = None,
        other: int | None = None,
        allowed_methods: typing.Collection[str] | None = DEFAULT_ALLOWED_METHODS,
        status_forcelist: typing.Collection[int] | None = None,
        backoff_factor: float = 0,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        raise_on_redirect: bool = True,
        raise_on_status: bool = True,
        history: tuple[RequestHistory, ...] | None = None,
        respect_retry_after_header: bool = True,
        remove_headers_on_redirect: typing.Collection[
            str
        ] = DEFAULT_REMOVE_HEADERS_ON_REDIRECT,
        backoff_jitter: float = 0.0,
        retry_after_max: int = DEFAULT_RETRY_AFTER_MAX,
    ) -> None:
        self.total = total
        self.connect = connect
        self.read = read
        self.status = status
        self.other = other

        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False

        self.redirect = redirect
        self.status_forcelist = status_forcelist or set()
        if not allowed_methods and isinstance(allowed_methods, Collection):
            warnings.warn(
                "Using an empty collection for 'allowed_methods' option to "
                "retry on any verb is deprecated and will skip retries for "
                "all verbs in urllib3 v3.0. Instead use "
                "Retry(..., allowed_methods=None).",
                FutureWarning,
                stacklevel=2,
            )
        self.allowed_methods = allowed_methods
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.retry_after_max = retry_after_max
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = history or ()
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = frozenset(
            h.lower() for h in remove_headers_on_redirect
        )
        self.backoff_jitter = backoff_jitter

    def new(self, **kw: typing.Any) -> Self:
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            other=self.other,
            allowed_methods=self.allowed_methods,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            backoff_max=self.backoff_max,
            retry_after_max=self.retry_after_max,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            remove_headers_on_redirect=self.remove_headers_on_redirect,
            respect_retry_after_header=self.respect_retry_after_header,
            backoff_jitter=self.backoff_jitter,
        )

        params.update(kw)
        return type(self)(**params)  # type: ignore[arg-type]

    @classmethod
    def from_int(
        cls,
        retries: Retry | bool | int | None,
        redirect: bool | int | None = True,
        default: Retry | bool | int | None = None,
    ) -> Retry:
        """Backwards-compatibility for the old retries format."""
        if retries is None:
            retries = default if default is not None else cls.DEFAULT

        if isinstance(retries, Retry):
            return retries

        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries

    def get_backoff_time(self) -> float:
        """Formula for computing the current backoff

        :rtype: float
        """
        # We want to consider only the last consecutive errors sequence (Ignore redirects).
        consecutive_errors_len = len(
            list(
                takewhile(lambda x: x.redirect_location is None, reversed(self.history))
            )
        )
        if consecutive_errors_len <= 1:
            return 0

        backoff_value = self.backoff_factor * (2 ** (consecutive_errors_len - 1))
        if self.backoff_jitter != 0.0:
            backoff_value += random.random() * self.backoff_jitter
        return float(max(0, min(self.backoff_max, backoff_value)))

    def parse_retry_after(self, retry_after: str) -> float:
        seconds: float
        # Whitespace: https://tools.ietf.org/html/rfc7230#section-3.2.4
        if re.match(r"^\s*[0-9]+\s*$", retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate_tz(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader(f"Invalid Retry-After header: {retry_after}")

            retry_date = email.utils.mktime_tz(retry_date_tuple)
            seconds = retry_date - time.time()

        seconds = max(seconds, 0)

        # Check the seconds do not exceed the specified maximum
        if seconds > self.retry_after_max:
            seconds = self.retry_after_max

        return seconds

    def get_retry_after(self, response: BaseHTTPResponse) -> float | None:
        """Get the value of Retry-After in seconds."""

        retry_after = response.headers.get("Retry-After")

        if retry_after is None:
            return None

        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response: BaseHTTPResponse) -> bool:
        retry_after = self.get_retry_after(response)
        if retry_after:
            time.sleep(retry_after)
            return True

        return False

    def _sleep_backoff(self) -> None:
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def sleep(self, response: BaseHTTPResponse | None = None) -> None:
        """Sleep between retry attempts.

        This method will respect a server's ``Retry-After`` response header
        and sleep the duration of the time requested. If that is not present, it
        will use an exponential backoff. By default, the backoff factor is 0 and
        this method will return immediately.
        """

        if self.respect_retry_after_header and response:
            slept = self.sleep_for_retry(response)
            if slept:
                return

        self._sleep_backoff()

    def _is_connection_error(self, err: Exception) -> bool:
        """Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
        if isinstance(err, ProxyError):
            err = err.original_error
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err: Exception) -> bool:
        """Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def _is_method_retryable(self, method: str) -> bool:
        """Checks if a given HTTP method should be retried upon, depending if
        it is included in the allowed_methods
        """
        if self.allowed_methods and method.upper() not in self.allowed_methods:
            return False
        return True

    def is_retry(
        self, method: str, status_code: int, has_retry_after: bool = False
    ) -> bool:
        """Is this method/status code retryable? (Based on allowlists and control
        variables such as the number of total retries to allow, whether to
        respect the Retry-After header, whether this header is present, and
        whether the returned status code is on the list of status codes to
        be retried upon on the presence of the aforementioned header)
        """
        if not self._is_method_retryable(method):
            return False

        if self.status_forcelist and status_code in self.status_forcelist:
            return True

        return bool(
            self.total
            and self.respect_retry_after_header
            and has_retry_after
            and (status_code in self.RETRY_AFTER_STATUS_CODES)
        )

    def is_exhausted(self) -> bool:
        """Are we out of retries?"""
        retry_counts = [
            x
            for x in (
                self.total,
                self.connect,
                self.read,
                self.redirect,
                self.status,
                self.other,
            )
            if x
        ]
        if not retry_counts:
            return False

        return min(retry_counts) < 0

    def increment(
        self,
        method: str | None = None,
        url: str | None = None,
        response: BaseHTTPResponse | None = None,
        error: Exception | None = None,
        _pool: ConnectionPool | None = None,
        _stacktrace: TracebackType | None = None,
    ) -> Self:
        """Return a new Retry object with incremented retry counters.

        :param response: A response object, or None, if the server did not
            return a response.
        :type response: :class:`~urllib3.response.BaseHTTPResponse`
        :param Exception error: An error encountered during the request, or
            None if the response was received successfully.

        :return: A new ``Retry`` object.
        """
        if self.total is False and error:
            # Disabled, indicate to re-raise the error.
            raise reraise(type(error), error, _stacktrace)

        total = self.total
        if total is not None:
            total -= 1

        connect = self.connect
        read = self.read
        redirect = self.redirect
        status_count = self.status
        other = self.other
        cause = "unknown"
        status = None
        redirect_location = None

        if error and self._is_connection_error(error):
            # Connect retry?
            if connect is False:
                raise reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1

        elif error and self._is_read_error(error):
            # Read retry?
            if read is False or method is None or not self._is_method_retryable(method):
                raise reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1

        elif error:
            # Other retry?
            if other is not None:
                other -= 1

        elif response and response.get_redirect_location():
            # Redirect retry?
            if redirect is not None:
                redirect -= 1
            cause = "too many redirects"
            response_redirect_location = response.get_redirect_location()
            if response_redirect_location:
                redirect_location = response_redirect_location
            status = response.status

        else:
            # Incrementing because of a server error like a 500 in
            # status_forcelist and the given method is in the allowed_methods
            cause = ResponseError.GENERIC_ERROR
            if response and response.status:
                if status_count is not None:
                    status_count -= 1
                cause = ResponseError.SPECIFIC_ERROR.format(status_code=response.status)
                status = response.status

        history = self.history + (
            RequestHistory(method, url, error, status, redirect_location),
        )

        new_retry = self.new(
            total=total,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status_count,
            other=other,
            history=history,
        )

        if new_retry.is_exhausted():
            reason = error or ResponseError(cause)
            raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]

        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)

        return new_retry

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(total={self.total}, connect={self.connect}, "
            f"read={self.read}, redirect={self.redirect}, status={self.status})"
        )


# For backwards compatibility (equivalent to pre-v1.9):
Retry.DEFAULT = Retry(3)
<|endoftext|>from __future__ import annotations

import typing

from .url import Url

if typing.TYPE_CHECKING:
    from ..connection import ProxyConfig


def connection_requires_http_tunnel(
    proxy_url: Url | None = None,
    proxy_config: ProxyConfig | None = None,
    destination_scheme: str | None = None,
) -> bool:
    """
    Returns True if the connection requires an HTTP CONNECT through the proxy.

    :param URL proxy_url:
        URL of the proxy.
    :param ProxyConfig proxy_config:
        Proxy configuration from poolmanager.py
    :param str destination_scheme:
        The scheme of the destination. (i.e https, http, etc)
    """
    # If we're not using a proxy, no way to use a tunnel.
    if proxy_url is None:
        return False

    # HTTP destinations never require tunneling, we always forward.
    if destination_scheme == "http":
        return False

    # Support for forwarding with HTTPS proxies and HTTPS destinations.
    if (
        proxy_url.scheme == "https"
        and proxy_config
        and proxy_config.use_forwarding_for_https
    ):
        return False

    # Otherwise always use a tunnel.
    return True
<|endoftext|>from __future__ import annotations

import threading


class _HTTP2ProbeCache:
    __slots__ = (
        "_lock",
        "_cache_locks",
        "_cache_values",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_locks: dict[tuple[str, int], threading.RLock] = {}
        self._cache_values: dict[tuple[str, int], bool | None] = {}

    def acquire_and_get(self, host: str, port: int) -> bool | None:
        # By the end of this block we know that
        # _cache_[values,locks] is available.
        value = None
        with self._lock:
            key = (host, port)
            try:
                value = self._cache_values[key]
                # If it's a known value we return right away.
                if value is not None:
                    return value
            except KeyError:
                self._cache_locks[key] = threading.RLock()
                self._cache_values[key] = None

        # If the value is unknown, we acquire the lock to signal
        # to the requesting thread that the probe is in progress
        # or that the current thread needs to return their findings.
        key_lock = self._cache_locks[key]
        key_lock.acquire()
        try:
            # If the by the time we get the lock the value has been
            # updated we want to return the updated value.
            value = self._cache_values[key]

        # In case an exception like KeyboardInterrupt is raised here.
        except BaseException as e:  # Defensive:
            assert not isinstance(e, KeyError)  # KeyError shouldn't be possible.
            key_lock.release()
            raise

        return value

    def set_and_release(
        self, host: str, port: int, supports_http2: bool | None
    ) -> None:
        key = (host, port)
        key_lock = self._cache_locks[key]
        with key_lock:  # Uses an RLock, so can be locked again from same thread.
            if supports_http2 is None and self._cache_values[key] is not None:
                raise ValueError(
                    "Cannot reset HTTP/2 support for origin after value has been set."
                )  # Defensive: not expected in normal usage

        self._cache_values[key] = supports_http2
        key_lock.release()

    def _values(self) -> dict[tuple[str, int], bool | None]:
        """This function is for testing purposes only. Gets the current state of the probe cache"""
        with self._lock:
            return {k: v for k, v in self._cache_values.items()}

    def _reset(self) -> None:
        """This function is for testing purposes only. Reset the cache values"""
        with self._lock:
            self._cache_locks = {}
            self._cache_values = {}


_HTTP2_PROBE_CACHE = _HTTP2ProbeCache()

set_and_release = _HTTP2_PROBE_CACHE.set_and_release
acquire_and_get = _HTTP2_PROBE_CACHE.acquire_and_get
_values = _HTTP2_PROBE_CACHE._values
_reset = _HTTP2_PROBE_CACHE._reset

__all__ = [
    "set_and_release",
    "acquire_and_get",
]
<|endoftext|>from __future__ import annotations

from importlib.metadata import version

__all__ = [
    "inject_into_urllib3",
    "extract_from_urllib3",
]

import typing

orig_HTTPSConnection: typing.Any = None


def inject_into_urllib3() -> None:
    # First check if h2 version is valid
    h2_version = version("h2")
    if not h2_version.startswith("4."):
        raise ImportError(
            "urllib3 v2 supports h2 version 4.x.x, currently "
            f"the 'h2' module is compiled with {h2_version!r}. "
            "See: https://github.com/urllib3/urllib3/issues/3290"
        )

    # Import here to avoid circular dependencies.
    from .. import connection as urllib3_connection
    from .. import util as urllib3_util
    from ..connectionpool import HTTPSConnectionPool
    from ..util import ssl_ as urllib3_util_ssl
    from .connection import HTTP2Connection

    global orig_HTTPSConnection
    orig_HTTPSConnection = urllib3_connection.HTTPSConnection

    HTTPSConnectionPool.ConnectionCls = HTTP2Connection
    urllib3_connection.HTTPSConnection = HTTP2Connection  # type: ignore[misc]

    # TODO: Offer 'http/1.1' as well, but for testing purposes this is handy.
    urllib3_util.ALPN_PROTOCOLS = ["h2"]
    urllib3_util_ssl.ALPN_PROTOCOLS = ["h2"]


def extract_from_urllib3() -> None:
    from .. import connection as urllib3_connection
    from .. import util as urllib3_util
    from ..connectionpool import HTTPSConnectionPool
    from ..util import ssl_ as urllib3_util_ssl

    HTTPSConnectionPool.ConnectionCls = orig_HTTPSConnection
    urllib3_connection.HTTPSConnection = orig_HTTPSConnection  # type: ignore[misc]

    urllib3_util.ALPN_PROTOCOLS = ["http/1.1"]
    urllib3_util_ssl.ALPN_PROTOCOLS = ["http/1.1"]
<|endoftext|>from __future__ import annotations

import logging
import re
import threading
import types
import typing

import h2.config
import h2.connection
import h2.events

from .._base_connection import _TYPE_BODY
from .._collections import HTTPHeaderDict
from ..connection import HTTPSConnection, _get_default_user_agent
from ..exceptions import ConnectionError
from ..response import BaseHTTPResponse

orig_HTTPSConnection = HTTPSConnection

T = typing.TypeVar("T")

log = logging.getLogger(__name__)

RE_IS_LEGAL_HEADER_NAME = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9a-z]+$")
RE_IS_ILLEGAL_HEADER_VALUE = re.compile(rb"[\0\x00\x0a\x0d\r\n]|^[ \r\n\t]|[ \r\n\t]$")


def _is_legal_header_name(name: bytes) -> bool:
    """
    "An implementation that validates fields according to the definitions in Sections
    5.1 and 5.5 of [HTTP] only needs an additional check that field names do not
    include uppercase characters." (https://httpwg.org/specs/rfc9113.html#n-field-validity)

    `http.client._is_legal_header_name` does not validate the field name according to the
    HTTP 1.1 spec, so we do that here, in addition to checking for uppercase characters.

    This does not allow for the `:` character in the header name, so should not
    be used to validate pseudo-headers.
    """
    return bool(RE_IS_LEGAL_HEADER_NAME.match(name))


def _is_illegal_header_value(value: bytes) -> bool:
    """
    "A field value MUST NOT contain the zero value (ASCII NUL, 0x00), line feed
    (ASCII LF, 0x0a), or carriage return (ASCII CR, 0x0d) at any position. A field
    value MUST NOT start or end with an ASCII whitespace character (ASCII SP or HTAB,
    0x20 or 0x09)." (https://httpwg.org/specs/rfc9113.html#n-field-validity)
    """
    return bool(RE_IS_ILLEGAL_HEADER_VALUE.search(value))


class _LockedObject(typing.Generic[T]):
    """
    A wrapper class that hides a specific object behind a lock.
    The goal here is to provide a simple way to protect access to an object
    that cannot safely be simultaneously accessed from multiple threads. The
    intended use of this class is simple: take hold of it with a context
    manager, which returns the protected object.
    """

    __slots__ = (
        "lock",
        "_obj",
    )

    def __init__(self, obj: T):
        self.lock = threading.RLock()
        self._obj = obj

    def __enter__(self) -> T:
        self.lock.acquire()
        return self._obj

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.lock.release()


class HTTP2Connection(HTTPSConnection):
    def __init__(
        self, host: str, port: int | None = None, **kwargs: typing.Any
    ) -> None:
        self._h2_conn = self._new_h2_conn()
        self._h2_stream: int | None = None
        self._headers: list[tuple[bytes, bytes]] = []

        if "proxy" in kwargs or "proxy_config" in kwargs:  # Defensive:
            raise NotImplementedError("Proxies aren't supported with HTTP/2")

        super().__init__(host, port, **kwargs)

        if self._tunnel_host is not None:
            raise NotImplementedError("Tunneling isn't supported with HTTP/2")

    def _new_h2_conn(self) -> _LockedObject[h2.connection.H2Connection]:
        config = h2.config.H2Configuration(client_side=True)
        return _LockedObject(h2.connection.H2Connection(config=config))

    def connect(self) -> None:
        super().connect()
        with self._h2_conn as conn:
            conn.initiate_connection()
            if data_to_send := conn.data_to_send():
                self.sock.sendall(data_to_send)

    def putrequest(  # type: ignore[override]
        self,
        method: str,
        url: str,
        **kwargs: typing.Any,
    ) -> None:
        """putrequest
        This deviates from the HTTPConnection method signature since we never need to override
        sending accept-encoding headers or the host header.
        """
        if "skip_host" in kwargs:
            raise NotImplementedError("`skip_host` isn't supported")
        if "skip_accept_encoding" in kwargs:
            raise NotImplementedError("`skip_accept_encoding` isn't supported")

        self._request_url = url or "/"
        self._validate_path(url)  # type: ignore[attr-defined]

        port = self.port if self.port is not None else 443
        if ":" in self.host:
            authority = f"[{self.host}]:{port}"
        else:
            authority = f"{self.host}:{port}"

        self._headers.append((b":scheme", b"https"))
        self._headers.append((b":method", method.encode()))
        self._headers.append((b":authority", authority.encode()))
        self._headers.append((b":path", url.encode()))

        with self._h2_conn as conn:
            self._h2_stream = conn.get_next_available_stream_id()

    def putheader(self, header: str | bytes, *values: str | bytes) -> None:  # type: ignore[override]
        # TODO SKIPPABLE_HEADERS from urllib3 are ignored.
        header = header.encode() if isinstance(header, str) else header
        header = header.lower()  # A lot of upstream code uses capitalized headers.
        if not _is_legal_header_name(header):
            raise ValueError(f"Illegal header name {str(header)}")

        for value in values:
            value = value.encode() if isinstance(value, str) else value
            if _is_illegal_header_value(value):
                raise ValueError(f"Illegal header value {str(value)}")
            self._headers.append((header, value))

    def endheaders(self, message_body: typing.Any = None) -> None:  # type: ignore[override]
        if self._h2_stream is None:
            raise ConnectionError("Must call `putrequest` first.")

        with self._h2_conn as conn:
            conn.send_headers(
                stream_id=self._h2_stream,
                headers=self._headers,
                end_stream=(message_body is None),
            )
            if data_to_send := conn.data_to_send():
                self.sock.sendall(data_to_send)
        self._headers = []  # Reset headers for the next request.

    def send(self, data: typing.Any) -> None:
        """Send data to the server.
        `data` can be: `str`, `bytes`, an iterable, or file-like objects
        that support a .read() method.
        """
        if self._h2_stream is None:
            raise ConnectionError("Must call `putrequest` first.")

        with self._h2_conn as conn:
            if data_to_send := conn.data_to_send():
                self.sock.sendall(data_to_send)

            if hasattr(data, "read"):  # file-like objects
                while True:
                    chunk = data.read(self.blocksize)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode()
                    conn.send_data(self._h2_stream, chunk, end_stream=False)
                    if data_to_send := conn.data_to_send():
                        self.sock.sendall(data_to_send)
                conn.end_stream(self._h2_stream)
                return

            if isinstance(data, str):  # str -> bytes
                data = data.encode()

            try:
                if isinstance(data, bytes):
                    conn.send_data(self._h2_stream, data, end_stream=True)
                    if data_to_send := conn.data_to_send():
                        self.sock.sendall(data_to_send)
                else:
                    for chunk in data:
                        conn.send_data(self._h2_stream, chunk, end_stream=False)
                        if data_to_send := conn.data_to_send():
                            self.sock.sendall(data_to_send)
                    conn.end_stream(self._h2_stream)
            except TypeError:
                raise TypeError(
                    "`data` should be str, bytes, iterable, or file. got %r"
                    % type(data)
                )

    def set_tunnel(
        self,
        host: str,
        port: int | None = None,
        headers: typing.Mapping[str, str] | None = None,
        scheme: str = "http",
    ) -> None:
        raise NotImplementedError(
            "HTTP/2 does not support setting up a tunnel through a proxy"
        )

    def getresponse(  # type: ignore[override]
        self,
    ) -> HTTP2Response:
        status = None
        data = bytearray()
        with self._h2_conn as conn:
            end_stream = False
            while not end_stream:
                # TODO: Arbitrary read value.
                if received_data := self.sock.recv(65535):
                    events = conn.receive_data(received_data)
                    for event in events:
                        if isinstance(event, h2.events.ResponseReceived):
                            headers = HTTPHeaderDict()
                            for header, value in event.headers:
                                if header == b":status":
                                    status = int(value.decode())
                                else:
                                    headers.add(
                                        header.decode("ascii"), value.decode("ascii")
                                    )

                        elif isinstance(event, h2.events.DataReceived):
                            data += event.data
                            conn.acknowledge_received_data(
                                event.flow_controlled_length, event.stream_id
                            )

                        elif isinstance(event, h2.events.StreamEnded):
                            end_stream = True

                if data_to_send := conn.data_to_send():
                    self.sock.sendall(data_to_send)

        assert status is not None
        return HTTP2Response(
            status=status,
            headers=headers,
            request_url=self._request_url,
            data=bytes(data),
        )

    def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        *,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """Send an HTTP/2 request"""
        if "chunked" in kwargs:
            # TODO this is often present from upstream.
            # raise NotImplementedError("`chunked` isn't supported with HTTP/2")
            pass

        if self.sock is not None:
            self.sock.settimeout(self.timeout)

        self.putrequest(method, url)

        headers = headers or {}
        for k, v in headers.items():
            if k.lower() == "transfer-encoding" and v == "chunked":
                continue
            else:
                self.putheader(k, v)

        if b"user-agent" not in dict(self._headers):
            self.putheader(b"user-agent", _get_default_user_agent())

        if body:
            self.endheaders(message_body=body)
            self.send(body)
        else:
            self.endheaders()

    def close(self) -> None:
        with self._h2_conn as conn:
            try:
                conn.close_connection()
                if data := conn.data_to_send():
                    self.sock.sendall(data)
            except Exception:
                pass

        # Reset all our HTTP/2 connection state.
        self._h2_conn = self._new_h2_conn()
        self._h2_stream = None
        self._headers = []

        super().close()


class HTTP2Response(BaseHTTPResponse):
    # TODO: This is a woefully incomplete response object, but works for non-streaming.
    def __init__(
        self,
        status: int,
        headers: HTTPHeaderDict,
        request_url: str,
        data: bytes,
        decode_content: bool = False,  # TODO: support decoding
    ) -> None:
        super().__init__(
            status=status,
            headers=headers,
            # Following CPython, we map HTTP versions to major * 10 + minor integers
            version=20,
            version_string="HTTP/2",
            # No reason phrase in HTTP/2
            reason=None,
            decode_content=decode_content,
            request_url=request_url,
        )
        self._data = data
        self.length_remaining = 0

    @property
    def data(self) -> bytes:
        return self._data

    def get_redirect_location(self) -> None:
        return None

    def close(self) -> None:
        pass
<|endoftext|>from __future__ import annotations

from typing import Any

from .core import decode, encode


def ToASCII(label: str) -> bytes:
    """Compatibility shim for :rfc:`3490` ``ToASCII``.

    Delegates to :func:`idna.encode` (IDNA 2008). Provided to ease porting
    of code written against the legacy :mod:`encodings.idna` API; new code
    should call :func:`idna.encode` directly.

    :param label: The label or domain to encode.
    :returns: The encoded form as ASCII :class:`bytes`.
    """
    return encode(label)


def ToUnicode(label: bytes | bytearray) -> str:
    """Compatibility shim for :rfc:`3490` ``ToUnicode``.

    Delegates to :func:`idna.decode` (IDNA 2008). Provided to ease porting
    of code written against the legacy :mod:`encodings.idna` API; new code
    should call :func:`idna.decode` directly.

    :param label: The label or domain to decode.
    :returns: The decoded Unicode form.
    """
    return decode(label)


def nameprep(s: Any) -> None:
    """Stub for :rfc:`3491` Nameprep, which is not used by IDNA 2008.

    IDNA 2008 (:rfc:`5891`) replaces Nameprep with the per-codepoint
    validity classes from :rfc:`5892`; this function exists only to
    return a clear error if legacy code attempts to call it.

    :raises NotImplementedError: Always.
    """
    raise NotImplementedError("IDNA 2008 does not utilise nameprep protocol")
<|endoftext|>"""
Given a list of integers, made up of (hopefully) a small number of long runs
of consecutive integers, compute a representation of the form
((start1, end1), (start2, end2) ...). Then answer the question "was x present
in the original list?" in time O(log(# runs)).
"""

import bisect


def intranges_from_list(list_: list[int]) -> tuple[int, ...]:
    """Represent a list of integers as a sequence of ranges:
    ((start_0, end_0), (start_1, end_1), ...), such that the original
    integers are exactly those x such that start_i <= x < end_i for some i.

    Ranges are encoded as single integers (start << 32 | end), not as tuples.
    """

    sorted_list = sorted(list_)
    ranges = []
    last_write = -1
    for i in range(len(sorted_list)):
        if i + 1 < len(sorted_list) and sorted_list[i] == sorted_list[i + 1] - 1:
            continue
        current_range = sorted_list[last_write + 1 : i + 1]
        ranges.append(_encode_range(current_range[0], current_range[-1] + 1))
        last_write = i

    return tuple(ranges)


def _encode_range(start: int, end: int) -> int:
    return (start << 32) | end


def _decode_range(r: int) -> tuple[int, int]:
    return (r >> 32), (r & ((1 << 32) - 1))


def intranges_contain(int_: int, ranges: tuple[int, ...]) -> bool:
    """Determine if `int_` falls into one of the ranges in `ranges`."""
    tuple_ = _encode_range(int_, 0)
    pos = bisect.bisect_left(ranges, tuple_)
    # we could be immediately ahead of a tuple (start, end)
    # with start < int_ <= end
    if pos > 0:
        left, right = _decode_range(ranges[pos - 1])
        if left <= int_ < right:
            return True
    # or we could be immediately behind a tuple (int_, end)
    if pos < len(ranges):
        left, _ = _decode_range(ranges[pos])
        if left == int_:
            return True
    return False
<|endoftext|>from __future__ import annotations

import codecs
from typing import Any

from .core import IDNAError, _max_domain_length, _unicode_dots_re, alabel, decode, encode, ulabel


class Codec(codecs.Codec):
    """Stateless IDNA 2008 codec.

    Implements the :class:`codecs.Codec` protocol so that the whole-domain
    encoder (:func:`idna.encode`) and decoder (:func:`idna.decode`) are
    accessible through the standard codec machinery as ``"idna2008"``.

    Only the ``"strict"`` error handler is supported; any other handler
    raises :exc:`~idna.IDNAError`.
    """

    def encode(self, data: str, errors: str = "strict") -> tuple[bytes, int]:  # ty: ignore[invalid-method-override]
        if errors != "strict":
            raise IDNAError(f'Unsupported error handling "{errors}"', code="unsupported_errors")

        if not data:
            return b"", 0

        return encode(data), len(data)

    def decode(self, data: bytes, errors: str = "strict") -> tuple[str, int]:  # ty: ignore[invalid-method-override]
        if errors != "strict":
            raise IDNAError(f'Unsupported error handling "{errors}"', code="unsupported_errors")

        if not data:
            return "", 0

        return decode(data), len(data)


class IncrementalEncoder(codecs.BufferedIncrementalEncoder):
    """Incremental IDNA 2008 encoder.

    Buffers a partial trailing label across calls until either the next
    label separator is seen or ``final=True``, so that streamed input is
    encoded one whole label at a time. Any of the four Unicode label
    separators (``U+002E``, ``U+3002``, ``U+FF0E``, ``U+FF61``) ends a
    label; the result always uses ``U+002E`` as the separator.

    The 253-octet domain length limit (254 with a trailing dot) that
    :func:`idna.encode` enforces is applied to the accumulated output, so
    that streaming a name and encoding it in one shot either both succeed
    with the same result or both raise :exc:`~idna.IDNAError`.

    Only the ``"strict"`` error handler is supported.
    """

    def __init__(self, errors: str = "strict") -> None:
        super().__init__(errors)
        self._emitted = 0  # octets returned so far
        self._trailing_dot = False  # whether the output so far ends with "."

    def reset(self) -> None:
        super().reset()
        self._emitted = 0
        self._trailing_dot = False

    def getstate(self) -> Any:
        if not self.buffer and not self._emitted:
            return 0
        return (self.buffer, self._emitted, self._trailing_dot)

    def setstate(self, state: Any) -> None:
        if state:
            self.buffer, self._emitted, self._trailing_dot = state
        else:
            self.reset()

    def _buffer_encode(self, data: str, errors: str, final: bool) -> tuple[bytes, int]:  # ty: ignore[invalid-method-override]
        if errors != "strict":
            raise IDNAError(f'Unsupported error handling "{errors}"', code="unsupported_errors")

        result_bytes = b""
        size = 0
        if data:
            labels = _unicode_dots_re.split(data)
            trailing_dot = b""
            if labels:
                if not labels[-1]:
                    trailing_dot = b"."
                    del labels[-1]
                elif not final:
                    # Keep potentially unfinished label until the next call
                    del labels[-1]
                    if labels:
                        trailing_dot = b"."

            result = []
            for label in labels:
                result.append(alabel(label))
                if size:
                    size += 1
                size += len(label)

            result_bytes = b".".join(result) + trailing_dot
            size += len(trailing_dot)

        # Mirror encode(): the whole name may not exceed 253 octets, or 254
        # when it ends with a dot. Until the input is final a trailing dot
        # may still arrive, so only the 254-octet ceiling applies before then.
        self._emitted += len(result_bytes)
        if result_bytes:
            self._trailing_dot = result_bytes.endswith(b".")
        may_end_with_dot = self._trailing_dot or not final
        if self._emitted > _max_domain_length + may_end_with_dot:
            raise IDNAError("Domain too long", code="domain_too_long")
        return result_bytes, size


class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    """Incremental IDNA 2008 decoder.

    Buffers a partial trailing label across calls until either the next
    label separator is seen or ``final=True``, so that streamed input is
    decoded one whole label at a time.

    The 254-octet input length limit that :func:`idna.decode` enforces is
    applied to the accumulated input, so that streaming a name and decoding
    it in one shot either both succeed with the same result or both raise
    :exc:`~idna.IDNAError`.

    Only the ``"strict"`` error handler is supported.
    """

    def __init__(self, errors: str = "strict") -> None:
        super().__init__(errors)
        self._consumed = 0  # input octets consumed so far

    def reset(self) -> None:
        super().reset()
        self._consumed = 0

    def getstate(self) -> tuple[bytes, int]:
        return (self.buffer, self._consumed)

    def setstate(self, state: tuple[bytes, int]) -> None:
        self.buffer, self._consumed = state

    def _buffer_decode(self, data: Any, errors: str, final: bool) -> tuple[str, int]:  # ty: ignore[invalid-method-override]
        if errors != "strict":
            raise IDNAError(f'Unsupported error handling "{errors}"', code="unsupported_errors")

        if not data:
            return ("", 0)

        if not isinstance(data, str):
            try:
                data = str(data, "ascii")
            except UnicodeDecodeError as err:
                raise IDNAError("Invalid ASCII in A-label", code="invalid_ascii") from err

        # Mirror decode(), which rejects input longer than 254 characters
        # (253 plus a possible trailing dot) before looking at any label.
        # ``data`` is the unconsumed buffer plus the new input, so this is
        # the total seen so far.
        if self._consumed + len(data) > _max_domain_length + 1:
            raise IDNAError("Domain too long", code="domain_too_long")

        labels = _unicode_dots_re.split(data)
        trailing_dot = ""
        if labels:
            if not labels[-1]:
                trailing_dot = "."
                del labels[-1]
            elif not final:
                # Keep potentially unfinished label until the next call
                del labels[-1]
                if labels:
                    trailing_dot = "."

        result = []
        size = 0
        for label in labels:
            result.append(ulabel(label))
            if size:
                size += 1
            size += len(label)

        result_str = ".".join(result) + trailing_dot
        size += len(trailing_dot)
        self._consumed += size
        return (result_str, size)


class StreamWriter(Codec, codecs.StreamWriter):
    pass


class StreamReader(Codec, codecs.StreamReader):
    pass


def search_function(name: str) -> codecs.CodecInfo | None:
    """Codec search function registered with :mod:`codecs`.

    Returns a :class:`codecs.CodecInfo` for the ``"idna2008"`` codec name
    so that ``str.encode("idna2008")`` and ``bytes.decode("idna2008")``
    invoke the IDNA 2008 codec defined in this module.

    :param name: The codec name being looked up.
    :returns: A :class:`codecs.CodecInfo` instance if ``name`` is
        ``"idna2008"``, otherwise ``None``.
    """
    if name != "idna2008":
        return None
    return codecs.CodecInfo(
        name=name,
        encode=Codec().encode,
        decode=Codec().decode,  # type: ignore
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )


codecs.register(search_function)
<|endoftext|>from .core import (
    IDNABidiError,
    IDNAError,
    InvalidCodepoint,
    InvalidCodepointContext,
    alabel,
    check_bidi,
    check_hyphen_ok,
    check_initial_combiner,
    check_label,
    check_nfc,
    decode,
    encode,
    ulabel,
    uts46_remap,
    valid_contextj,
    valid_contexto,
    valid_label_length,
    valid_string_length,
)
from .idnadata import __version__ as unicode_version
from .intranges import intranges_contain
from .package_data import __version__

__all__ = [
    "__version__",
    "unicode_version",
    "IDNABidiError",
    "IDNAError",
    "InvalidCodepoint",
    "InvalidCodepointContext",
    "alabel",
    "check_bidi",
    "check_hyphen_ok",
    "check_initial_combiner",
    "check_label",
    "check_nfc",
    "decode",
    "encode",
    "intranges_contain",
    "ulabel",
    "uts46_remap",
    "valid_contextj",
    "valid_contexto",
    "valid_label_length",
    "valid_string_length",
]
<|endoftext|>from __future__ import annotations

import bisect
import re
import unicodedata
import warnings
from typing import Literal

from . import idnadata
from .intranges import intranges_contain

_virama_combining_class = 9
_alabel_prefix = b"xn--"
_max_input_length = 1024
_max_domain_length = 253  # RFC 1035 octets, excluding any trailing dot
_STATUS_VALID, _STATUS_MAPPED, _STATUS_DEVIATION, _STATUS_IGNORED = b"VMDI"
_unicode_dots_re = re.compile("[\u002e\u3002\uff0e\uff61]")
_std3_disallowed_re = re.compile("[\x00-\x2c\x2f\x3a-\x40A-Z\x5b-\x60\x7b-\x7f]")
_bidi_rtl_first = frozenset({"R", "AL"})
_bidi_rtl_categories = frozenset({"R", "AL", "AN"})
_bidi_rtl_allowed = frozenset({"R", "AL", "AN", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"})
_bidi_rtl_valid_ending = frozenset({"R", "AL", "EN", "AN"})
_bidi_rtl_numeric = frozenset({"AN", "EN"})
_bidi_ltr_allowed = frozenset({"L", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"})
_bidi_ltr_valid_ending = frozenset({"L", "EN"})
_bidi_joiner_l_or_d = frozenset({"L", "D"})
_bidi_joiner_r_or_d = frozenset({"R", "D"})


def _joining_type(cp: int) -> str | None:
    for jt, ranges in idnadata.joining_types.items():
        if intranges_contain(cp, ranges):
            return jt
    return None


# Machine-readable identifiers for the rule an :class:`IDNAError` reports.
# These strings are stable and documented; exception message wording is not.
_ErrorCode = Literal[
    "input_too_long",
    "label_too_long",
    "domain_too_long",
    "empty_label",
    "empty_domain",
    "not_nfc",
    "hyphen_3_4",
    "hyphen_start_end",
    "leading_combiner",
    "disallowed_codepoint",
    "contextj",
    "contexto",
    "unknown_codepoint",
    "bidi_rule_1",
    "bidi_rule_2",
    "bidi_rule_3",
    "bidi_rule_4",
    "bidi_rule_5",
    "bidi_rule_6",
    "bidi_unknown_direction",
    "invalid_alabel",
    "non_canonical_alabel",
    "invalid_ascii",
    "invalid_utf8",
    "uts46_disallowed",
    "uts46_std3",
    "unsupported_errors",
]


class IDNAError(UnicodeError):
    """Base exception for all IDNA-encoding related problems.

    ``str(err)`` is a human-readable description of the failure. The
    exception also carries machine-readable attributes so callers do not
    need to parse the message:

    * ``code`` -- a short, stable identifier for the rule that failed, such
      as ``"disallowed_codepoint"`` or ``"bidi_rule_2"``; the full list is
      documented in the README. Message wording, by contrast, may change
      between releases.
    * ``text`` -- the label (or, for UTS #46 processing, the domain) that
      was being validated;
    * ``codepoint`` -- the offending codepoint, as an ``int``;
    * ``position`` -- the 1-based index of the offending character within
      ``text``, matching the position quoted in the message.

    Each is ``None`` when it does not apply.
    """

    code: str | None
    text: str | None
    codepoint: int | None
    position: int | None

    def __init__(
        self,
        *args: object,
        code: _ErrorCode | None = None,
        text: str | None = None,
        codepoint: int | None = None,
        position: int | None = None,
    ) -> None:
        super().__init__(*args)
        self.code = code
        self.text = text
        self.codepoint = codepoint
        self.position = position


class IDNABidiError(IDNAError):
    """Exception when bidirectional requirements are not satisfied"""


class InvalidCodepoint(IDNAError):
    """Exception when a disallowed or unallocated codepoint is used"""


class InvalidCodepointContext(IDNAError):
    """Exception when the codepoint is not valid in the context it is used"""


def _combining_class(cp: int) -> int:
    v = unicodedata.combining(chr(cp))
    if v == 0 and not unicodedata.name(chr(cp)):
        raise ValueError("Unknown character in unicodedata")
    return v


def _is_script(cp: str, script: str) -> bool:
    return intranges_contain(ord(cp), idnadata.scripts[script])


def _punycode(s: str) -> bytes:
    return s.encode("punycode")


def _unot(s: int) -> str:
    return f"U+{s:04X}"


def valid_label_length(label: bytes | str) -> bool:
    """Check that a label does not exceed the maximum permitted length.

    Per :rfc:`1035` (and :rfc:`5891` §4.2.4) a DNS label must not exceed
    63 octets. The argument may be either a :class:`str` (a U-label, where
    length is measured in characters) or :class:`bytes` (an A-label, where
    length is measured in octets).

    :param label: The label to check.
    :returns: ``True`` if the label is within the length limit, otherwise
        ``False``.
    """
    return len(label) <= 63


def valid_string_length(domain: bytes | str, trailing_dot: bool) -> bool:
    """Check that a full domain name does not exceed the maximum length.

    Per :rfc:`1035`, a domain name is limited to 253 octets when no trailing
    dot is present, or 254 octets when one is included.

    :param domain: The full (possibly multi-label) domain name.
    :param trailing_dot: ``True`` if ``domain`` includes a trailing ``.``.
    :returns: ``True`` if the domain is within the length limit, otherwise
        ``False``.
    """
    return len(domain) <= _max_domain_length + trailing_dot


def check_bidi(label: str, check_ltr: bool = False) -> bool:
    """Validate the Bidi Rule from :rfc:`5893` for a single label.

    The Bidi Rule constrains how bidirectional characters (Hebrew, Arabic,
    etc.) may appear within a label. By default the check is only applied
    when the label contains at least one right-to-left character (Unicode
    bidirectional categories ``R``, ``AL``, or ``AN``); set ``check_ltr``
    to ``True`` to apply it to LTR-only labels as well.

    :param label: The label to validate, as a Unicode string.
    :param check_ltr: If ``True``, apply the rules even when the label
        contains no RTL characters.
    :returns: ``True`` if the label satisfies the Bidi Rule.
    :raises IDNABidiError: If any of Bidi Rule conditions 1-6 are violated,
        or if the directional category of a codepoint cannot be determined.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    # Bidi rules should only be applied if string contains RTL characters
    bidi_label = False
    for idx, cp in enumerate(label, 1):
        direction = unicodedata.bidirectional(cp)
        if direction == "":
            # String likely comes from a newer version of Unicode
            raise IDNABidiError(
                f"Unknown directionality in label {label!r} at position {idx}",
                code="bidi_unknown_direction",
                text=label,
                codepoint=ord(cp),
                position=idx,
            )
        if direction in _bidi_rtl_categories:
            bidi_label = True
    if not bidi_label and not check_ltr:
        return True

    # Bidi rule 1
    direction = unicodedata.bidirectional(label[0])
    if direction in _bidi_rtl_first:
        rtl = True
    elif direction == "L":
        rtl = False
    else:
        raise IDNABidiError(
            f"First codepoint in label {label!r} must be directionality L, R or AL",
            code="bidi_rule_1",
            text=label,
            codepoint=ord(label[0]),
            position=1,
        )

    valid_ending = False
    ending_idx = 1
    number_type: str | None = None
    for idx, cp in enumerate(label, 1):
        direction = unicodedata.bidirectional(cp)

        if rtl:
            # Bidi rule 2
            if direction not in _bidi_rtl_allowed:
                raise IDNABidiError(
                    f"Invalid direction for codepoint at position {idx} in a right-to-left label",
                    code="bidi_rule_2",
                    text=label,
                    codepoint=ord(cp),
                    position=idx,
                )
            # Bidi rule 3
            if direction in _bidi_rtl_valid_ending:
                valid_ending = True
                ending_idx = idx
            elif direction != "NSM":
                valid_ending = False
                ending_idx = idx
            # Bidi rule 4
            if direction in _bidi_rtl_numeric:
                if not number_type:
                    number_type = direction
                elif number_type != direction:
                    raise IDNABidiError(
                        "Can not mix numeral types in a right-to-left label",
                        code="bidi_rule_4",
                        text=label,
                        codepoint=ord(cp),
                        position=idx,
                    )
        else:
            # Bidi rule 5
            if direction not in _bidi_ltr_allowed:
                raise IDNABidiError(
                    f"Invalid direction for codepoint at position {idx} in a left-to-right label",
                    code="bidi_rule_5",
                    text=label,
                    codepoint=ord(cp),
                    position=idx,
                )
            # Bidi rule 6
            if direction in _bidi_ltr_valid_ending:
                valid_ending = True
                ending_idx = idx
            elif direction != "NSM":
                valid_ending = False
                ending_idx = idx

    if not valid_ending:
        # Rules 3 and 6 concern the last character that is not a
        # non-spacing mark, which is what ``ending_idx`` tracks.
        raise IDNABidiError(
            "Label ends with illegal codepoint directionality",
            code="bidi_rule_3" if rtl else "bidi_rule_6",
            text=label,
            codepoint=ord(label[ending_idx - 1]),
            position=ending_idx,
        )

    return True


def check_initial_combiner(label: str) -> bool:
    """Reject labels that begin with a combining mark.

    Per :rfc:`5891` §4.2.3.2 a label must not start with a character of
    Unicode general category ``M`` (Mark).

    :param label: The label to check.
    :returns: ``True`` if the first character is not a combining mark.
    :raises IDNAError: If the label begins with a combining character.
    """
    if label and unicodedata.category(label[0])[0] == "M":
        raise IDNAError(
            "Label begins with an illegal combining character",
            code="leading_combiner",
            text=label,
            codepoint=ord(label[0]),
            position=1,
        )
    return True


def check_hyphen_ok(label: str) -> bool:
    """Validate the hyphen restrictions for a label.

    Per :rfc:`5891` §4.2.3.1 a label must not start or end with a hyphen
    (``U+002D``), and must not have hyphens in both the third and fourth
    positions (the prefix reserved for A-labels).

    :param label: The label to check.
    :returns: ``True`` if the hyphen restrictions are satisfied.
    :raises IDNAError: If any of the hyphen restrictions are violated.
    """
    if label[2:4] == "--":
        raise IDNAError("Label has disallowed hyphens in 3rd and 4th position", code="hyphen_3_4")
    if label.startswith("-") or label.endswith("-"):
        raise IDNAError("Label must not start or end with a hyphen", code="hyphen_start_end")
    return True


def check_nfc(label: str) -> None:
    """Require that a label is in Unicode Normalization Form C.

    :param label: The label to check.
    :raises IDNAError: If ``label`` differs from its NFC normalisation.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    if unicodedata.normalize("NFC", label) != label:
        raise IDNAError("Label must be in Normalization Form C", code="not_nfc")


def valid_contextj(label: str, pos: int) -> bool:
    """Validate the CONTEXTJ rules from :rfc:`5892` Appendix A.

    These rules govern the contextual use of the joiner codepoints
    ``U+200C`` (ZERO WIDTH NON-JOINER, Appendix A.1) and ``U+200D``
    (ZERO WIDTH JOINER, Appendix A.2) within a label.

    :param label: The label containing the codepoint.
    :param pos: Index of the joiner codepoint within ``label``.
    :returns: ``True`` if the codepoint at ``pos`` satisfies its CONTEXTJ
        rule, ``False`` otherwise (including when the codepoint at
        ``pos`` is not a recognised joiner).
    :raises ValueError: If an adjacent codepoint has no Unicode name when
        determining its combining class.
    :raises IDNAError: If ``label`` exceeds the defensive input length limit.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    cp_value = ord(label[pos])

    if cp_value == 0x200C:
        if pos > 0 and _combining_class(ord(label[pos - 1])) == _virama_combining_class:
            return True

        ok = False
        for i in range(pos - 1, -1, -1):
            joining_type = _joining_type(ord(label[i]))
            if joining_type == "T":
                continue
            if joining_type in _bidi_joiner_l_or_d:
                ok = True
                break
            break

        if not ok:
            return False

        ok = False
        for i in range(pos + 1, len(label)):
            joining_type = _joining_type(ord(label[i]))
            if joining_type == "T":
                continue
            if joining_type in _bidi_joiner_r_or_d:
                ok = True
                break
            break
        return ok

    if cp_value == 0x200D:
        return pos > 0 and _combining_class(ord(label[pos - 1])) == _virama_combining_class

    return False


def valid_contexto(label: str, pos: int, exception: bool = False) -> bool:
    """Validate the CONTEXTO rules from :rfc:`5892` Appendix A.

    Covers the contextual rules for codepoints such as MIDDLE DOT
    (``U+00B7``), Greek lower numeral sign, Hebrew punctuation, Katakana
    middle dot, and the Arabic-Indic / Extended Arabic-Indic digit ranges.

    :param label: The label containing the codepoint.
    :param pos: Index of the codepoint within ``label``.
    :param exception: Reserved for forward compatibility; currently unused.
    :returns: ``True`` if the codepoint at ``pos`` satisfies its CONTEXTO
        rule, ``False`` otherwise (including when the codepoint is not a
        recognised CONTEXTO codepoint).
    :raises IDNAError: If ``label`` exceeds the defensive input length limit.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    cp_value = ord(label[pos])

    if cp_value == 0x00B7:
        return 0 < pos < len(label) - 1 and ord(label[pos - 1]) == 0x006C and ord(label[pos + 1]) == 0x006C

    if cp_value == 0x0375:
        if pos < len(label) - 1 and len(label) > 1:
            return _is_script(label[pos + 1], "Greek")
        return False

    if cp_value in {0x05F3, 0x05F4}:
        if pos > 0:
            return _is_script(label[pos - 1], "Hebrew")
        return False

    if cp_value == 0x30FB:
        for cp in label:
            if cp == "\u30fb":
                continue
            if _is_script(cp, "Hiragana") or _is_script(cp, "Katakana") or _is_script(cp, "Han"):
                return True
        return False

    if 0x660 <= cp_value <= 0x669:
        return not any(0x6F0 <= ord(cp) <= 0x06F9 for cp in label)

    if 0x6F0 <= cp_value <= 0x6F9:
        return not any(0x660 <= ord(cp) <= 0x0669 for cp in label)

    return False


def check_label(label: str | bytes | bytearray) -> None:
    """Run the full set of IDNA 2008 validity checks on a single label.

    Applies, in order: NFC normalisation (:func:`check_nfc`), hyphen
    restrictions (:func:`check_hyphen_ok`), the no-leading-combiner rule
    (:func:`check_initial_combiner`), per-codepoint validity (PVALID,
    CONTEXTJ, CONTEXTO classes from :rfc:`5892`), and the Bidi Rule
    (:func:`check_bidi`).

    :param label: The label to validate. ``bytes`` or ``bytearray`` input
        is decoded as UTF-8 first.
    :raises IDNAError: If the label is empty or fails a structural rule.
    :raises InvalidCodepoint: If the label contains a DISALLOWED or
        UNASSIGNED codepoint.
    :raises InvalidCodepointContext: If a CONTEXTJ or CONTEXTO codepoint
        is not valid in its context.
    :raises IDNABidiError: If the Bidi Rule is violated.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    if isinstance(label, (bytes, bytearray)):
        try:
            label = label.decode("utf-8")
        except UnicodeDecodeError as err:
            raise IDNAError("Invalid UTF-8 in label", code="invalid_utf8") from err
    if len(label) == 0:
        raise IDNAError("Empty Label", code="empty_label")

    # Check against the domain length rather than the label length to
    # support some UTS #46 use cases, while still bounding the work done
    # by the label contextual rules below.
    if not valid_string_length(label, trailing_dot=True):
        raise IDNAError("Label too long", code="label_too_long")

    check_nfc(label)
    check_hyphen_ok(label)
    check_initial_combiner(label)

    for pos, cp in enumerate(label):
        cp_value = ord(cp)
        if intranges_contain(cp_value, idnadata.codepoint_classes["PVALID"]):
            continue
        if intranges_contain(cp_value, idnadata.codepoint_classes["CONTEXTJ"]):
            try:
                contextj_ok = valid_contextj(label, pos)
            except ValueError as err:
                raise IDNAError(
                    f"Unknown codepoint adjacent to joiner {_unot(cp_value)} at position {pos + 1} in {label!r}",
                    code="unknown_codepoint",
                    text=label,
                    codepoint=cp_value,
                    position=pos + 1,
                ) from err
            if not contextj_ok:
                raise InvalidCodepointContext(
                    f"Joiner {_unot(cp_value)} not allowed at position {pos + 1} in {label!r}",
                    code="contextj",
                    text=label,
                    codepoint=cp_value,
                    position=pos + 1,
                )
        elif intranges_contain(cp_value, idnadata.codepoint_classes["CONTEXTO"]):
            if not valid_contexto(label, pos):
                raise InvalidCodepointContext(
                    f"Codepoint {_unot(cp_value)} not allowed at position {pos + 1} in {label!r}",
                    code="contexto",
                    text=label,
                    codepoint=cp_value,
                    position=pos + 1,
                )
        else:
            raise InvalidCodepoint(
                f"Codepoint {_unot(cp_value)} at position {pos + 1} of {label!r} not allowed",
                code="disallowed_codepoint",
                text=label,
                codepoint=cp_value,
                position=pos + 1,
            )

    check_bidi(label)


def alabel(label: str) -> bytes:
    """Convert a single U-label into its A-label form.

    The result is the ASCII-Compatible Encoding (ACE) form per :rfc:`5891`
    §4: the label is validated, Punycode-encoded, and prefixed with
    ``xn--``. Pure ASCII labels that are already valid IDNA labels are
    returned unchanged (as :class:`bytes`).

    :param label: The label to convert, as a Unicode string.
    :returns: The A-label as ASCII-encoded :class:`bytes`.
    :raises IDNAError: If the label is invalid or the resulting A-label
        exceeds 63 octets.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    try:
        label_bytes = label.encode("ascii")
    except UnicodeEncodeError:
        pass
    else:
        ulabel(label_bytes)
        if not valid_label_length(label_bytes):
            raise IDNAError("Label too long", code="label_too_long")
        return label_bytes

    check_label(label)
    label_bytes = _alabel_prefix + _punycode(label)

    if not valid_label_length(label_bytes):
        raise IDNAError("Label too long", code="label_too_long")

    return label_bytes


def ulabel(label: str | bytes | bytearray) -> str:
    """Convert a single A-label into its U-label form.

    Performs the inverse of :func:`alabel`: an ``xn--``-prefixed label is
    Punycode-decoded and validated, and is rejected unless it is the
    canonical A-label for the decoded U-label (:rfc:`5891` §5.3). Labels
    that are already Unicode (or plain ASCII without the ACE prefix) are
    validated and returned as a Unicode string.

    :param label: The label to convert. ``bytes`` or ``bytearray`` input
        is treated as ASCII.
    :returns: The U-label as a Unicode string.
    :raises IDNAError: If the label is malformed or fails validation.
    """
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long")
    if not isinstance(label, (bytes, bytearray)):
        try:
            label_bytes = label.encode("ascii")
        except UnicodeEncodeError:
            check_label(label)
            return label
    else:
        label_bytes = bytes(label)
        if not label_bytes.isascii():
            raise IDNAError("Invalid ASCII in A-label", code="invalid_ascii")

    label_bytes = label_bytes.lower()
    if label_bytes.startswith(_alabel_prefix):
        label_bytes = label_bytes[len(_alabel_prefix) :]
        if not label_bytes:
            raise IDNAError("Malformed A-label, no Punycode eligible content found", code="invalid_alabel")
        if label_bytes.endswith(b"-"):
            raise IDNAError("A-label must not end with a hyphen", code="invalid_alabel")
    else:
        check_label(label_bytes)
        return label_bytes.decode("ascii")

    try:
        label = label_bytes.decode("punycode")
    except UnicodeError as err:
        raise IDNAError("Invalid A-label", code="invalid_alabel") from err
    # RFC 5891 §5.3: the label is rejected unless re-encoding the decoded
    # form reproduces the (lowercased) input. This catches "fake A-labels"
    # (RFC 5890 §2.3.2.1) such as ``xn---bbk``, a non-canonical Punycode
    # spelling of ``xn--bbk`` that would otherwise decode to the same
    # U-label and so display identically to a different wire-format name.
    if _punycode(label) != label_bytes:
        raise IDNAError("A-label is not the canonical Punycode encoding of its U-label", code="non_canonical_alabel")
    check_label(label)
    return label


def _check_std3(text: str, domain: str, offset: int) -> None:
    """Raise if ``text``, a slice of ``domain`` starting at ``offset`` that
    UTS #46 mapping left unchanged, contains an ASCII character disallowed
    under ``UseSTD3ASCIIRules``."""
    match = _std3_disallowed_re.search(text)
    if match:
        codepoint = ord(match.group())
        position = offset + match.start() + 1
        raise InvalidCodepoint(
            f"Codepoint {_unot(codepoint)} not allowed at position {position} in {domain!r}",
            code="uts46_std3",
            text=domain,
            codepoint=codepoint,
            position=position,
        )


def _warn_transitional() -> None:
    warnings.warn(
        "Transitional processing is deprecated in UTS #46 and has no effect. "
        "The transitional argument will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


def uts46_remap(domain: str, std3_rules: bool = True, transitional: bool = False) -> str:
    """Apply the UTS #46 character mapping to a domain string.

    Implements the mapping table from `UTS #46 §4
    <https://www.unicode.org/reports/tr46/>`_: each character is kept,
    replaced, or rejected based on its status (``V``, ``M``, ``D``,
    ``I``, ``X``). The result is returned in Normalisation Form C.

    :param domain: The full domain name to remap.
    :param std3_rules: If ``True``, apply UTS #46's ``UseSTD3ASCIIRules``:
        after mapping, any ASCII character other than a lowercase letter,
        digit, hyphen or the label separator ``.`` is rejected, whether it
        appeared in the input or was produced by a mapping (e.g. U+FF01
        FULLWIDTH EXCLAMATION MARK maps to ``!``). If ``False``, such
        characters are passed through.
    :param transitional: Deprecated and ignored. UTS #46 deprecated
        transitional processing in Unicode 15.1 and deviation (status
        ``D``) codepoints are now always kept, so this has no effect
        beyond emitting a :class:`DeprecationWarning`. It will be removed
        in a future version.
    :returns: The remapped domain, in Normalisation Form C.
    :raises InvalidCodepoint: If the domain contains a disallowed
        codepoint under the chosen rules.
    :raises IDNAError: If ``domain`` exceeds the defensive input length limit.
    """
    if transitional:
        _warn_transitional()
    if len(domain) > _max_input_length:
        raise IDNAError("Domain too long", code="input_too_long")
    if domain.isascii():
        # The only ASCII mapping in UTS #46 is upper- to lowercase, and
        # ASCII is invariant under NFC, so lowercasing is the whole job.
        result = domain.lower()
        if std3_rules:
            _check_std3(result, domain, 0)
        return result

    from .uts46data import uts46_replacements, uts46_starts, uts46_statuses

    # ``start`` marks the run of unchanged input not yet copied; a run is
    # only sliced out when a character must be replaced or dropped, so the
    # common no-change case makes no copy. STD3 is checked per output piece
    # to report a violation at its input position.
    output: list[str] = []
    start = 0
    for pos, char in enumerate(domain):
        code_point = ord(char)
        i = code_point if code_point < 256 else bisect.bisect_right(uts46_starts, code_point) - 1
        status = uts46_statuses[i]
        # UTS #46 §4: V valid, D deviation (kept), M mapped, I ignored,
        # anything else disallowed.
        if status == _STATUS_VALID:
            continue
        if status == _STATUS_MAPPED:
            replacement = uts46_replacements[i]
        elif status == _STATUS_DEVIATION:
            continue
        elif status == _STATUS_IGNORED:
            replacement = None
        else:
            raise InvalidCodepoint(
                f"Codepoint {_unot(code_point)} not allowed at position {pos + 1} in {domain!r}",
                code="uts46_disallowed",
                text=domain,
                codepoint=code_point,
                position=pos + 1,
            )
        if start < pos:
            run = domain[start:pos]
            if std3_rules:
                _check_std3(run, domain, start)
            output.append(run)
        if replacement:
            if std3_rules and _std3_disallowed_re.search(replacement):
                raise InvalidCodepoint(
                    f"Codepoint {_unot(code_point)} not allowed at position {pos + 1} in {domain!r}",
                    code="uts46_std3",
                    text=domain,
                    codepoint=code_point,
                    position=pos + 1,
                )
            output.append(replacement)
        start = pos + 1

    if start == 0:
        if std3_rules:
            _check_std3(domain, domain, 0)
        return unicodedata.normalize("NFC", domain)
    tail = domain[start:]
    if std3_rules:
        _check_std3(tail, domain, start)
    output.append(tail)
    return unicodedata.normalize("NFC", "".join(output))


def encode(
    s: str | bytes | bytearray,
    strict: bool = False,
    uts46: bool = False,
    std3_rules: bool = False,
    transitional: bool = False,
) -> bytes:
    """Encode a Unicode domain name into its ASCII (A-label) form.

    Splits the input on label separators (only ``U+002E`` if ``strict`` is
    set; otherwise also IDEOGRAPHIC FULL STOP ``U+3002``, FULLWIDTH FULL
    STOP ``U+FF0E``, and HALFWIDTH IDEOGRAPHIC FULL STOP ``U+FF61``),
    encodes each label with :func:`alabel`, and rejoins them with ``.``.
    Optionally pre-processes the input through :func:`uts46_remap`.

    :param s: The domain name to encode.
    :param strict: If ``True``, only ``U+002E`` is recognised as a label
        separator.
    :param uts46: If ``True``, apply UTS #46 mapping before encoding.
    :param std3_rules: Forwarded to :func:`uts46_remap` when ``uts46`` is
        ``True``.
    :param transitional: Deprecated and ignored (see :func:`uts46_remap`):
        emits a :class:`DeprecationWarning` and will be removed in a
        future version.
    :returns: The encoded domain as ASCII :class:`bytes`.
    :raises IDNAError: If the domain is empty, contains an invalid label,
        or exceeds the maximum domain length.
    """
    if transitional:
        _warn_transitional()
    if not isinstance(s, str):
        try:
            s = str(s, "ascii")
        except (UnicodeDecodeError, TypeError) as err:
            raise IDNAError(
                "should pass a unicode string to the function rather than a byte string.", code="invalid_ascii"
            ) from err
    if len(s) > _max_input_length:
        raise IDNAError("Domain too long", code="input_too_long")
    if uts46:
        s = uts46_remap(s, std3_rules)

    if not valid_string_length(s, trailing_dot=True):
        raise IDNAError("Domain too long", code="domain_too_long")

    trailing_dot = False
    result = []
    labels = s.split(".") if strict else _unicode_dots_re.split(s)
    if not labels or labels == [""]:
        raise IDNAError("Empty domain", code="empty_domain")
    if labels[-1] == "":
        del labels[-1]
        trailing_dot = True
    for label in labels:
        s = alabel(label)
        if s:
            result.append(s)
        else:
            raise IDNAError("Empty label", code="empty_label")
    if trailing_dot:
        result.append(b"")
    s = b".".join(result)
    if not valid_string_length(s, trailing_dot):
        raise IDNAError("Domain too long", code="domain_too_long")
    return s


def decode(
    s: str | bytes | bytearray,
    strict: bool = False,
    uts46: bool = False,
    std3_rules: bool = False,
    display: bool = False,
) -> str:
    """Decode an A-label-encoded domain name back to Unicode.

    Splits the input on label separators (see :func:`encode` for the
    rules), decodes each label with :func:`ulabel`, and rejoins them
    with ``.``. Optionally pre-processes the input through
    :func:`uts46_remap`.

    :param s: The domain name to decode.
    :param strict: If ``True``, only ``U+002E`` is recognised as a label
        separator.
    :param uts46: If ``True``, apply UTS #46 mapping before decoding.
    :param std3_rules: Forwarded to :func:`uts46_remap` when ``uts46`` is
        ``True``.
    :param display: If ``True``, any ``xn--`` label that fails IDNA
        validation is passed through unchanged (lowercased) rather than
        aborting the whole call. Intended for "decode for display"
        consumers (e.g. URL libraries, HTTP clients) that want to show
        the user the label as it appears on the wire when it cannot be
        rendered as Unicode. Matches the per-label recovery prescribed
        by UTS #46 §4 and the WHATWG URL "domain to Unicode" algorithm.
    :returns: The decoded domain as a Unicode string.
    :raises IDNAError: If the input is not valid ASCII, contains an
        invalid label, or is empty.
    """
    if not isinstance(s, str):
        try:
            s = str(s, "ascii")
        except (UnicodeDecodeError, TypeError) as err:
            raise IDNAError("Invalid ASCII in A-label", code="invalid_ascii") from err
    if len(s) > _max_input_length:
        raise IDNAError("Domain too long", code="input_too_long")
    if uts46:
        s = uts46_remap(s, std3_rules, False)
    if not valid_string_length(s, trailing_dot=True):
        raise IDNAError("Domain too long", code="domain_too_long")
    trailing_dot = False
    result = []
    labels = s.split(".") if strict else _unicode_dots_re.split(s)
    if not labels or labels == [""]:
        raise IDNAError("Empty domain", code="empty_domain")
    if not labels[-1]:
        del labels[-1]
        trailing_dot = True
    for label in labels:
        try:
            u = ulabel(label)
        except IDNAError:
            if display and label[:4].lower() == "xn--":
                u = label.lower()
            else:
                raise
        if u:
            result.append(u)
        else:
            raise IDNAError("Empty label", code="empty_label")
    if trailing_dot:
        result.append("")
    return ".".join(result)
<|endoftext|>"""Command-line interface for the :mod:`idna` package.

Invoked via ``python -m idna``. See :func:`main` for the entry point.
"""

from __future__ import annotations

import argparse
import sys
from itertools import chain
from typing import IO, TYPE_CHECKING

from . import IDNAError, decode, encode, unicode_version
from .core import _alabel_prefix, _unicode_dots_re
from .package_data import __version__

if TYPE_CHECKING:
    from collections.abc import Iterable


def _looks_like_alabel(s: str) -> bool:
    """Return True if any label in ``s`` carries the ``xn--`` ACE prefix."""
    prefix = _alabel_prefix.decode("ascii")
    return any(label.lower().startswith(prefix) for label in _unicode_dots_re.split(s))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idna",
        description=(
            "Convert a domain name between its Unicode (U-label) and "
            "ASCII-compatible (A-label) forms. With no mode flag, the "
            "direction is chosen from the first input — if it contains "
            "an xn-- label the stream is decoded, otherwise it is "
            "encoded — and the same mode is applied to every remaining "
            "input. UTS #46 mapping is applied by default; pass "
            "--strict to disable it. When no domains are given on the "
            "command line and stdin is piped, one domain per line is "
            "read from stdin."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-e",
        "--encode",
        dest="mode",
        action="store_const",
        const="encode",
        help="Encode the input to its ASCII A-label form.",
    )
    mode.add_argument(
        "-d",
        "--decode",
        dest="mode",
        action="store_const",
        const="decode",
        help="Decode the input from its ASCII A-label form.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Disable the default UTS #46 mapping and apply IDNA 2008 rules verbatim.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"idna {__version__} (Unicode {unicode_version})",
    )
    parser.add_argument(
        "domain",
        nargs="*",
        help="One or more domain names to convert. Omit to read from stdin.",
    )
    return parser


def _iter_stdin(stream: IO[str]) -> Iterable[str]:
    """Yield non-empty stripped lines from ``stream``, ignoring blanks."""
    for line in stream:
        stripped = line.strip()
        if stripped:
            yield stripped


def _convert_one(domain: str, mode: str, uts46: bool) -> bool:
    """Convert ``domain`` and write the result; return ``False`` on failure."""
    try:
        if mode == "decode":
            print(decode(domain, uts46=uts46))
        else:
            print(encode(domain, uts46=uts46).decode("ascii"))
    except IDNAError as err:
        print(f"idna: {mode} failed for {domain!r}: {err}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m idna``.

    When more than one domain is supplied (via positional arguments or
    piped stdin) and no mode flag is given, the first input determines
    the direction and that mode is applied uniformly to the rest.

    :param argv: Argument list excluding the program name. Defaults to
        :data:`sys.argv` when ``None``.
    :returns: ``0`` on success, ``1`` if any conversion fails.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    uts46 = not args.strict

    if args.domain:
        domains: Iterable[str] = args.domain
    elif not sys.stdin.isatty():
        domains = _iter_stdin(sys.stdin)
    else:
        parser.error("a domain argument is required when stdin is a terminal")

    iterator = iter(domains)
    first = next(iterator, None)
    if first is None:
        return 0

    mode = args.mode or ("decode" if _looks_like_alabel(first) else "encode")

    results = [_convert_one(domain, mode, uts46) for domain in chain([first], iterator)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch._functorch.deprecated import (
    combine_state_for_ensemble,
    functionalize,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    make_functional,
    make_functional_with_buffers,
    vjp,
    vmap,
)

# utilities. Maybe these should go in their own namespace in the future?
from torch._functorch.make_functional import (
    FunctionalModule,
    FunctionalModuleWithBuffers,
)

# Was never documented
from torch._functorch.python_key import make_fx


# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into torch._functorch
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile


__version__ = torch.__version__
<|endoftext|>"""Adapted from https://github.com/arogozhnikov/einops/blob/36c7bb16e57d6e57f8f3050f9e07abdf3f00469f/einops/parsing.py.

MIT License

Copyright (c) 2018 Alex Rogozhnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import keyword
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


_ellipsis: str = "\u2026"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis:
    """Used by `ParsedExpression` to represent an axis with a size (> 1), but no associated identifier.

    Note: Different instances of this class are not equal to each other, even if they have the same value.
    """

    def __init__(self, value: str) -> None:
        self.value = int(value)
        if self.value < 1:
            raise ValueError(
                f"Anonymous axis should have positive length, not {self.value}"
            )

    def __repr__(self) -> str:
        return f"{self.value}-axis"


class ParsedExpression:
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(
        self,
        expression: str,
        *,
        allow_underscore: bool = False,
        allow_duplicates: bool = False,
    ) -> None:
        """Parse the expression and store relevant metadata.

        Args:
            expression (str): the `einops`-pattern to parse
            allow_underscore (bool): whether to allow axis identifier names to begin with an underscore
            allow_duplicates (bool): whether to allow an identifier to appear more than once in the expression
        """
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: bool | None = None
        self.identifiers: set[str | AnonymousAxis] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: list[list[str | AnonymousAxis] | str] = []
        if "." in expression:
            if "..." not in expression:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...)"
                )
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "
                )
            expression = expression.replace("...", _ellipsis)
            self.has_ellipsis = True

        bracket_group: list[str | AnonymousAxis] | None = None

        def add_axis_name(x: str) -> None:
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise ValueError(
                        f"Indexing expression contains duplicate dimension '{x}'"
                    )
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(
                    x, allow_underscore=allow_underscore
                )
                if not (is_number or is_axis_name):
                    raise ValueError(f"Invalid axis identifier: {x}\n{reason}")
                axis_name: str | AnonymousAxis = AnonymousAxis(x) if is_number else x
                self.identifiers.add(axis_name)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([axis_name])
                else:
                    bracket_group.append(axis_name)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise ValueError(
                            "Axis composition is one-level (brackets inside brackets not allowed)"
                        )
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise ValueError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")

        if bracket_group is not None:
            raise ValueError(f"Imbalanced parentheses in expression: '{expression}'")
        if current_identifier is not None:
            add_axis_name(current_identifier)

    @staticmethod
    def check_axis_name_return_reason(
        name: str, allow_underscore: bool = False
    ) -> tuple[bool, str]:
        """Check if the given axis name is valid, and a message explaining why if not.

        Valid axes names are python identifiers except keywords, and should not start or end with an underscore.

        Args:
            name (str): the axis name to check
            allow_underscore (bool): whether axis names are allowed to start with an underscore

        Returns:
            tuple[bool, str]: whether the axis name is valid, a message explaining why if not
        """
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn(
                    f"It is discouraged to use axes names that are keywords: {name}",
                    RuntimeWarning,
                )
            if name == "axis":
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name and will raise an error in future",
                    FutureWarning,
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """Check if the name is a valid axis name.

        Args:
            name (str): the axis name to check

        Returns:
            bool: whether the axis name is valid
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


def parse_pattern(
    pattern: str, axes_lengths: Mapping[str, int]
) -> tuple[ParsedExpression, ParsedExpression]:
    """Parse an `einops`-style pattern into a left-hand side and right-hand side `ParsedExpression` object.

    Args:
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions

    Returns:
       tuple[ParsedExpression, ParsedExpression]: a tuple containing the left-hand side and right-hand side expressions
    """
    # adapted from einops.einops._prepare_transformation_recipe
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator") from None

    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(
            f"Ellipsis found in right side, but not left side of a pattern {pattern}"
        )
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(
            f"Ellipsis is parenthesis in the left side is not allowed: {pattern}"
        )

    return left, right


def validate_rearrange_expressions(
    left: ParsedExpression, right: ParsedExpression, axes_lengths: Mapping[str, int]
) -> None:
    """Perform expression validations that are specific to the `rearrange` operation.

    Args:
        left (ParsedExpression): left-hand side expression
        right (ParsedExpression): right-hand side expression
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions
    """
    for length in axes_lengths.values():
        if (length_type := type(length)) is not int:
            raise TypeError(
                f"rearrange axis lengths must be integers, got: {length_type}"
            )

    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise ValueError("rearrange only supports unnamed axes of size 1")

    difference = set.symmetric_difference(left.identifiers, right.identifiers)
    if len(difference) > 0:
        raise ValueError(
            f"Identifiers only on one side of rearrange expression (should be on both): {difference}"
        )

    unmatched_axes = axes_lengths.keys() - left.identifiers
    if len(unmatched_axes) > 0:
        raise ValueError(
            f"Identifiers not found in rearrange expression: {unmatched_axes}"
        )


def comma_separate(collection: Collection[str | Collection[str]]) -> str:
    """Convert a collection of strings representing first class dims into a comma-separated string.

    Args:
        collection (Collection[Union[str, Collection[str]]]): the collection of strings to convert

    Returns:
        str: the comma-separated string

    Examples:
        >>> comma_separate(("d0",))
        'd0'

        >>> comma_separate(("d0", "d1", "d2", "d3"))
        'd0, d1, d2, d3'

        >>> comma_separate([("d1", "d4")])
        '(d1, d4)'

        >>> comma_separate([("d0",), (), ("d1",), ("d2",), ("d3", "d4")])
        '(d0,), (), (d1,), (d2,), (d3, d4)'
    """
    return ", ".join(
        item
        if isinstance(item, str)
        else f"({comma_separate(item)}{',' if len(item) == 1 else ''})"
        for item in collection
    )
<|endoftext|>from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from functorch.dim import dims  # noqa: F401

from ._parsing import (
    _ellipsis,
    AnonymousAxis,
    comma_separate,
    parse_pattern,
    validate_rearrange_expressions,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["rearrange"]


@functools.lru_cache(256)
def _create_rearrange_callable(
    tensor_ndim: int, pattern: str, **axes_lengths: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Translate an `einops`-style pattern into a callable that performs the rearrange using first-class dimensions.

    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and
    specified axes lengths, this function can be memoized.

    Args:
        tensor_ndim (int): the number of dimensions in the tensor to rearrange
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: a callable that performs the rearrangement
    """
    left, right = parse_pattern(pattern, axes_lengths)
    validate_rearrange_expressions(left, right, axes_lengths)

    n_anon_dims = sum(not dim for dim in left.composition)
    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_named_dims = len(left.identifiers) - 1

        if (pattern_ndim := n_anon_dims + n_named_dims) > tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be less than or equal to the number of "
                f"dimensions in the tensor ({tensor_ndim})"
            )
    else:
        n_ellipsis_dims = 0
        n_named_dims = len(left.identifiers)

        if (pattern_ndim := len(left.composition)) != tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be equal to the number of dimensions in "
                f"the tensor ({tensor_ndim})"
            )
    n_dims = n_named_dims + n_ellipsis_dims + n_anon_dims

    if n_dims == 0:
        # an identity rearrangement on a 0-dimension tensor
        return lambda tensor: tensor

    first_class_dims: tuple[str, ...] = tuple(f"d{i}" for i in range(n_dims))
    identifier_dim_map: dict[str | AnonymousAxis, tuple[str, ...]] = {}
    anon_axes: list[AnonymousAxis] = []

    # map the left-hand side identifiers to strings representing first class dims
    dims_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                # non-unitary anon axes are not allowed in rearrange & unitary anon axes are represented as empty lists
                if not isinstance(identifier, str):
                    raise AssertionError(f"Expected str, got {type(identifier)}")
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
            if not dimension:
                # unitary anonymous axis
                anon_axis = AnonymousAxis("1")
                identifier_dim_map[anon_axis] = (first_class_dims[dims_i],)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dims_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_dim_map[identifier] = tuple(
                first_class_dims[dims_i + j] for j in range(n_ellipsis_dims)
            )
            dims_i += n_ellipsis_dims
        else:
            raise ValueError(f"Unexpected dimension: {dimension}")

    def composition_to_dims(
        composition: Sequence[list[str | AnonymousAxis] | str],
    ) -> list[str | tuple[str, ...]]:
        """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first
        class dims."""
        dim_composition: list[str | tuple[str, ...]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                dim_composition.append(
                    tuple(
                        dim
                        for identifier in dimension
                        for dim in identifier_dim_map[identifier]
                    )
                )
            elif dimension == _ellipsis:
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                raise ValueError(f"Unexpected dimension: {dimension}")
        return dim_composition

    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    anon_dims = tuple(identifier_dim_map[axis][0] for axis in anon_axes)
    specified_lengths = tuple(
        (identifier_dim_map[axis][0], length) for axis, length in axes_lengths.items()
    )

    custom_rearrange_callable_name = "do_rearrange"
    custom_rearrange_callable_code = (
        (
            f"def {custom_rearrange_callable_name}(tensor):\n"
            f"    {comma_separate(first_class_dims)} = dims({n_dims})\n"
        )
        + (
            "".join(
                f"    {dim}.size = {length}\n" for (dim, length) in specified_lengths
            )
            if specified_lengths
            else ""
        )
        + f"    tensor = tensor[{comma_separate(left_dims)}].order({comma_separate(right_dims)})\n"
        + (
            f"    return tensor.sum({comma_separate([anon_dims])}, keepdim=False)\n"
            if anon_dims
            else "    return tensor\n"
        )
    )

    namespace: dict = {}
    exec(custom_rearrange_callable_code, globals(), namespace)
    return namespace[custom_rearrange_callable_name]


def rearrange(
    tensor: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange
        pattern (str): the rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Tensor: the rearranged tensor

    Examples:
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, "b h w c -> b h w c").shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, "b h w c -> (b h) w c").shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, "b h w c -> h (b w) c").shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, "b h w c -> b c h w").shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, "b h w c -> b (c h w)").shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, "b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    rearrange_callable = _create_rearrange_callable(
        tensor.ndim, pattern, **axes_lengths
    )

    return rearrange_callable(tensor)
<|endoftext|>from torch._functorch import config
from torch._functorch.aot_autograd import (
    aot_function,
    aot_module,
    aot_module_simplified,
    compiled_function,
    compiled_module,
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
    make_boxed_compiler,
    make_boxed_func,
)
from torch._functorch.compilers import (
    debug_compile,
    default_decompositions,
    draw_graph_compile,
    memory_efficient_fusion,
    nnc_jit,
    nop,
    print_compile,
    ts_compile,
)
from torch._functorch.fx_minifier import minifier
from torch._functorch.partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)
from torch._functorch.python_key import pythonkey_decompose
<|endoftext|># PyTorch forward-mode is not mature yet
from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
from torch.func import functionalize
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from collections.abc import Callable
from types import (
    BuiltinMethodType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any


FUNC_TYPES = (
    FunctionType,
    MethodDescriptorType,
    BuiltinMethodType,
    WrapperDescriptorType,
)
PROPERTY_TYPES = (GetSetDescriptorType, property)


def _py_wrap_method(orig: Callable, __torch_function__: Callable) -> Callable:
    def impl(*args: Any, **kwargs: Any) -> Any:
        return __torch_function__(orig, None, args, kwargs)

    # Copy metadata using functools.update_wrapper for just __name__ and __doc__
    functools.update_wrapper(impl, orig, assigned=("__name__", "__doc__"), updated=())

    return impl


def wrap_type(to_patch: Any, pattern: type, __torch_function__: Callable) -> None:
    wrap_method = _py_wrap_method

    all: dict[str, Any] = {}
    for t in reversed(pattern.mro()[:-1]):  # skip object
        all.update(t.__dict__)

    def wrap_attr(orig: Any) -> property:
        return property(wrap_method(orig.__get__, __torch_function__))

    for name, obj in all.items():
        if name in (
            "__dict__",
            "__new__",
            "__init__",
            "__repr__",
            "__weakref__",
            "__doc__",
            "__module__",
            "__dir__",
        ):
            continue

        # skip things that have been overloaded
        # things that come from object like `__eq__` still need to be patched, however.
        if hasattr(to_patch, name) and getattr(to_patch, name) is not getattr(
            object, name, None
        ):
            continue

        if isinstance(obj, FUNC_TYPES):
            setattr(to_patch, name, wrap_method(obj, __torch_function__))
        elif isinstance(obj, PROPERTY_TYPES):
            setattr(to_patch, name, wrap_attr(obj))
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from ._dim_entry import _match_levels, DimEntry
from ._tensor_info import TensorInfo


if TYPE_CHECKING:
    from . import Dim


def _safe_index(lst: list, item: Any) -> int | None:
    """
    Helper function to find index of item in list.

    For DimEntry objects, uses __eq__ comparison which properly handles
    both positional and Dim entries.

    Returns the index if found, None if not found.
    """
    for i, list_item in enumerate(lst):
        # Use == for DimEntry objects as they have proper __eq__ implementation
        if isinstance(item, DimEntry) and isinstance(list_item, DimEntry):
            if list_item == item:
                return i
        elif list_item is item:
            return i
    return None


@dataclass
class IndexingInfo:
    can_call_original: bool = False
    advanced_indexing: bool = False
    self_tensor: torch.Tensor | None = None
    flat_inputs: list[Any] = field(default_factory=list)
    result_levels: list[DimEntry] = field(default_factory=list)
    has_device: bool = False


def has_dims(obj: Any) -> bool:
    """
    Check if an object has first-class dimensions.

    This function checks if the object is either a Dim or a functorch Tensor
    that has first-class dimensions, using the proper check_exact methods.
    """
    from . import Dim, Tensor

    return Dim.check_exact(obj) or Tensor.check_exact(obj)


def _bind_dims_to_size(sz: int, sd: int, dims: list, nsz: list, nsd: list) -> None:
    """
    Bind dimensions to size and calculate proper strides for dim packs.
    """
    from . import DimensionBindError

    rhs_prod = 1
    for i, dim in enumerate(dims):
        if not dim.is_bound:
            # Check for multiple unbound dimensions
            for j in range(i + 1, len(dims)):
                if not dims[j].is_bound:
                    raise DimensionBindError(
                        f"cannot infer the sizes of two dimensions at once {dim!r} and {dims[j]!r}"
                    )
                rhs_prod *= dims[j].size

            # Calculate the size for this unbound dimension
            if sz % rhs_prod != 0:
                tup = tuple(dim.size if dim.is_bound else "?" for dim in dims)
                raise DimensionBindError(
                    f"inferred dimension does not evenly fit into larger dimension: {sz} vs {tup}"
                )

            inferred_size = sz // rhs_prod
            dim.size = inferred_size
            rhs_prod = sz
            break
        else:
            rhs_prod *= dim.size

    # Final validation that dimensions match
    if rhs_prod != sz:
        tup = tuple(dims)
        raise DimensionBindError(
            f"Dimension sizes to do not match ({sz} != {rhs_prod}) when matching dimension pack {tup}"
        )

    # Calculate new sizes and strides for each dimension in the pack
    # First calculate all strides by iterating in reverse
    new_strides = [0] * len(dims)
    current_stride = sd
    for i in reversed(range(len(dims))):
        new_strides[i] = current_stride
        current_stride *= dims[i].size

    # Then append sizes and strides in forward order
    for i in range(len(dims)):
        nsz.append(dims[i].size)
        nsd.append(new_strides[i])


def slice_to_tuple(flat_inputs: list) -> tuple:
    return tuple(flat_inputs)


def extractIndices(index: Any, indices: list) -> bool:
    if isinstance(index, tuple):  # mpy::tuple_view::check
        indices.extend(index)
        return True
    elif isinstance(index, torch.Tensor):  # THPVariable_Check
        indices.append(index)
        return False
    elif not hasattr(index, "__iter__") or isinstance(
        index, (str, bytes)
    ):  # !mpy::is_sequence
        indices.append(index)
        return False

    # Handle sequence case (list)
    if isinstance(index, list):
        if len(index) >= 32:
            indices.extend(index)
            return True

        # Check each item in the sequence
        for item in index:
            if (
                isinstance(item, (torch.Tensor, slice))
                or hasattr(item, "__iter__")
                or item is ...
                or item is None
                or has_dims(item)
            ):
                indices.extend(index)
                return True

        # If we got here, treat as single index
        indices.append(index)
        return False

    # Default case
    indices.append(index)
    return False


def getitem(cls: Any, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    self = args[0]
    index = args[1]

    iinfo = getsetitem(self, index, has_dims(self))
    if iinfo.can_call_original:
        # Call original tensor __getitem__ directly, bypassing __torch_function__
        return torch.Tensor.__getitem__(self, index)

    return invoke_getitem(iinfo)


def setitem(self: Any, index: Any, rhs: Any) -> None:
    """Set values in tensor using first-class dimensions."""
    from . import DimensionBindError, TensorInfo

    iinfo = getsetitem(self, index, has_dims(self) or has_dims(rhs))

    if iinfo.can_call_original:
        # Call original tensor __setitem__ directly, bypassing __torch_function__
        torch._C.TensorBase.__setitem__(self, index, rhs)
        return

    # Handle RHS tensor with dimensions
    rhs_info = TensorInfo.create(rhs, False, False)

    if rhs_info:
        # Check that rhs dimensions are compatible with result dimensions
        for l in rhs_info.levels:
            if not l.is_positional():
                # Find this dimension in result levels
                found = False
                for result_level in iinfo.result_levels:
                    if (
                        not result_level.is_positional()
                        and result_level.dim() is l.dim()
                    ):
                        found = True
                        break

                if not found:
                    # Create tuple representation of result levels for error message
                    result_dims: list[int | Dim] = []
                    for rl in iinfo.result_levels:
                        if rl.is_positional():
                            result_dims.append(rl.position())
                        else:
                            result_dims.append(rl.dim())

                    raise DimensionBindError(
                        f"rhs of setitem contains dimension {l.dim()!r} which is not in the dimension on the left "
                        f"({tuple(result_dims)!r})"
                    )

        # Match RHS tensor to result levels
        if rhs_info.tensor is None:
            raise AssertionError("Cannot match levels on None tensor")
        matched_rhs = _match_levels(
            rhs_info.tensor, rhs_info.levels, iinfo.result_levels
        )
    else:
        matched_rhs = rhs

    # For advanced indexing with dimensions, we need special handling
    if iinfo.advanced_indexing:
        # Use advanced indexing - the flat_inputs already contain matched tensors
        tup = slice_to_tuple(iinfo.flat_inputs)
        if iinfo.self_tensor is None:
            raise RuntimeError("Cannot setitem on None tensor")
        torch._C.TensorBase.__setitem__(iinfo.self_tensor, tup, matched_rhs)
    else:
        # Simple copy operation
        if iinfo.self_tensor is None:
            raise RuntimeError("Cannot copy to None tensor")
        iinfo.self_tensor.copy_(matched_rhs)


def invoke_getitem(iinfo: IndexingInfo) -> Any:
    if iinfo.advanced_indexing:
        self_tensor = iinfo.self_tensor
        tup = slice_to_tuple(iinfo.flat_inputs)
        if self_tensor is None:
            raise RuntimeError("Cannot getitem on None tensor")
        rtensor = self_tensor[tup]
    else:
        rtensor = iinfo.self_tensor  # type: ignore[assignment]
        if rtensor is None:
            raise RuntimeError("Cannot getitem on None tensor")
        # rtensor is now guaranteed to be not None

    # Create a Tensor with the proper dimensions using the class method
    from . import Tensor

    return Tensor.from_positional(rtensor, iinfo.result_levels, iinfo.has_device)


def getsetitem(self: Any, index: Any, tensors_have_dims: bool) -> IndexingInfo:
    from . import DimList  # Import DimList for type checking

    can_call_original_getitem = not tensors_have_dims

    input_list = []
    if has_dims(index):
        input_list.append(index)
    else:
        is_sequence = extractIndices(index, input_list)
        # nothing about first class dims here, fallback to getitem
        if can_call_original_getitem and not is_sequence:
            return IndexingInfo(can_call_original=True)

    # Calculate how many dimensions have been indexed in order to compute the
    # size of ... or expand a potentially unbound dimension list.
    dims_indexed = 0
    expanding_object = -1
    unbound_dim_list = None
    dimlists = []  # Track DimList positions for later processing

    def check_expanding(i: int) -> None:
        nonlocal expanding_object
        if expanding_object != -1:
            from . import DimensionBindError

            raise DimensionBindError(
                f"at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets "
                f"{expanding_object} and {i}"
            )
        expanding_object = i

    def is_dimpack(s: Any) -> bool:
        from . import Dim

        return (
            isinstance(s, (tuple, list))
            and len(s) > 0
            and all(Dim.check_exact(item) for item in s)
        )

    has_dimpacks_or_none = False
    for i, s in enumerate(input_list):
        if has_dims(s):
            can_call_original_getitem = False
            dims_indexed += 1
        elif s is ...:
            check_expanding(i)
        elif isinstance(s, DimList):
            can_call_original_getitem = False
            if not s.is_bound:
                check_expanding(i)
                unbound_dim_list = s
            else:
                dims_indexed += len(s._dims)
            dimlists.append(i)
        elif s is None:
            has_dimpacks_or_none = True
        elif is_dimpack(s):
            can_call_original_getitem = False
            has_dimpacks_or_none = True
            dims_indexed += 1
        else:
            dims_indexed += 1

    # Early return if we can use original getitem
    if can_call_original_getitem:
        return IndexingInfo(can_call_original=True)

    self_info = TensorInfo.create(self, False, True)
    total_dims = len(self_info.levels)  # Total dimensions (positional + named)
    if dims_indexed > total_dims:
        raise ValueError(
            f"at least {dims_indexed} indices were supplied but the tensor only has {total_dims} dimensions"
        )

    # Expand any unbound dimension list, or expand ... into individual : slices.
    expanding_dims = total_dims - dims_indexed
    if expanding_object != -1:
        if unbound_dim_list is not None:
            # Bind unbound dimension list to the expanding dimensions
            unbound_dim_list.bind_len(expanding_dims)
        else:
            # Expand ... into slice(None) objects
            no_slices = [slice(None)] * expanding_dims
            input_list = (
                input_list[:expanding_object]
                + no_slices
                + input_list[expanding_object + 1 :]
            )

    # Flatten out any dimensions stored in dimlist elements directly into the inputs
    # Process in reverse order to maintain indices
    for i in range(len(dimlists) - 1, -1, -1):
        idx = dimlists[i]

        # We added more elements to input because of ...
        # so we need to also adjust the index to get back to where the
        # dimlist existed
        if (
            unbound_dim_list is None
            and expanding_object != -1
            and idx > expanding_object
        ):
            idx += expanding_dims

        dl = input_list[idx]

        # PRIVATE here naughty
        input_list = input_list[:idx] + dl._dims + input_list[idx + 1 :]

    return getsetitem_flat(self_info, input_list, [], [], has_dimpacks_or_none)


def getsetitem_flat(
    self_info: TensorInfo,
    input_list: list,
    keys: list[DimEntry],
    values: list,
    has_dimpacks_or_none: bool,
) -> IndexingInfo:
    from . import Dim

    # Track dimension usage
    seen_dims: list[Any] = []
    seen_dims_nuses: list[int] = []

    def add_dim(dim: Any) -> None:
        # Use safe indexing to avoid triggering __torch_function__ on Dim objects
        idx = _safe_index(seen_dims, dim)
        if idx is not None:
            seen_dims_nuses[idx] += 1
        else:
            seen_dims.append(dim)
            seen_dims_nuses.append(1)

    flat_inputs = []
    tensor_inputs: list[Any] = []
    device_holding_tensor = None

    def append_flat_handle(handle: Any) -> None:
        flat_inputs.append(handle)
        tensor_inputs.append(None)

    def append_tensor_input(ti: TensorInfo) -> None:
        flat_inputs.append(None)
        tensor_inputs.append(ti)
        nonlocal device_holding_tensor
        if ti.has_device and device_holding_tensor is None:
            device_holding_tensor = ti.tensor

    nsz = []
    nsd = []
    if self_info.tensor is None:
        raise RuntimeError("Cannot get size/stride on None tensor")
    sz = self_info.tensor.size()
    sd = self_info.tensor.stride()

    def append_size(i: int) -> None:
        if has_dimpacks_or_none:
            nsz.append(sz[i])
            nsd.append(sd[i])

    input_it = input_list[:]

    def parse_nones() -> None:
        nonlocal input_it
        while input_it and input_it[0] is None:
            append_flat_handle(slice(None))
            nsz.append(1)
            nsd.append(0)
            input_it = input_it[1:]

    def append_item(i: int, arg: Any) -> None:
        if Dim.check_exact(arg):
            d = arg
            if d._size == -1:
                d.size = sz[i]
            add_dim(d)
            append_size(i)
            append_flat_handle(arg)
            return

        info = TensorInfo.create(arg, False, False)
        if info:
            append_size(i)
            append_tensor_input(info)
            for level in info.levels:
                if not level.is_positional():
                    add_dim(level.dim())
            return

        if has_dimpacks_or_none:
            if isinstance(arg, (tuple, list)) and all(Dim.check_exact(d) for d in arg):
                # dim pack
                dim_pack = list(arg)
                for d in dim_pack:
                    add_dim(d)
                    append_flat_handle(d)
                _bind_dims_to_size(sz[i], sd[i], dim_pack, nsz, nsd)
                return

        append_size(i)
        append_flat_handle(arg)

    # Match indexing expressions with tensor dimensions
    for i, level in enumerate(self_info.levels):
        # Use safe indexing to avoid triggering __torch_function__ on DimEntry comparisons
        idx = _safe_index(keys, level)
        if idx is not None:
            append_item(i, values[idx])
        else:
            if level.is_positional():
                parse_nones()
                if not input_it:
                    append_flat_handle(slice(None))
                    append_size(i)
                else:
                    arg = input_it[0]
                    input_it = input_it[1:]
                    append_item(i, arg)
            else:
                add_dim(level.dim())
                append_flat_handle(level.dim())
                append_size(i)

    parse_nones()

    # Restride tensor if needed
    if has_dimpacks_or_none and nsz:
        if self_info.tensor is None:
            raise RuntimeError("Cannot restride None tensor")
        self_tensor = self_info.tensor.as_strided(
            nsz, nsd, self_info.tensor.storage_offset()
        )
    else:
        self_tensor = self_info.tensor

    # Determine result shape and indexing requirements
    result_levels: list[Any] = []
    index_levels = []
    tensor_insert_point = -1
    requires_getindex = False

    def mark_tensor_index() -> None:
        nonlocal tensor_insert_point
        if tensor_insert_point == -1:
            tensor_insert_point = len(result_levels)
        elif tensor_insert_point != len(result_levels):
            tensor_insert_point = 0

    for i, inp in enumerate(flat_inputs):
        if tensor_inputs[i] is not None:
            requires_getindex = True
            mark_tensor_index()
            for level in tensor_inputs[i].levels:
                if level not in index_levels:
                    index_levels.append(level)
        elif Dim.check_exact(inp):
            d = inp
            # Use safe indexing to avoid triggering __torch_function__
            dim_idx = _safe_index(seen_dims, d)
            if dim_idx is None:
                raise AssertionError(f"Dim {d} not found in seen_dims")
            if seen_dims_nuses[dim_idx] == 1:
                flat_inputs[i] = slice(None)
                result_levels.append(DimEntry(d))
            else:
                requires_getindex = True
                flat_inputs[i] = None
                tensor_inputs[i] = TensorInfo(
                    d._get_range(), [DimEntry(d)], False, None
                )
                if DimEntry(d) not in index_levels:
                    index_levels.append(DimEntry(d))
                mark_tensor_index()
        else:
            if inp != slice(None):
                requires_getindex = True
            if not isinstance(inp, int):
                result_levels.append(DimEntry(-1))

    # Insert indexing dimensions at first tensor use point
    if tensor_insert_point != -1:
        for level in reversed(index_levels):
            result_levels.insert(tensor_insert_point, level)

    # Match tensors to indexing shape
    if requires_getindex:
        for i in range(len(flat_inputs)):
            if tensor_inputs[i] is not None:
                t = tensor_inputs[i].tensor
                if t is None:
                    raise AssertionError("TensorInfo should have valid tensor data")
                if (
                    not tensor_inputs[i].has_device
                    and device_holding_tensor is not None
                ):
                    t = t.to(device_holding_tensor.device)
                flat_inputs[i] = _match_levels(t, tensor_inputs[i].levels, index_levels)

    # Number positional dimensions correctly
    seen_positionals = 0
    for i in reversed(range(len(result_levels))):
        if result_levels[i].is_positional():
            seen_positionals += 1
            result_levels[i] = DimEntry(-seen_positionals)

    return IndexingInfo(
        can_call_original=False,
        advanced_indexing=requires_getindex,
        self_tensor=self_tensor,
        flat_inputs=flat_inputs,
        result_levels=result_levels,
        has_device=self_info.has_device,
    )
<|endoftext|>import dis
from typing import Any


class _PyInstDecoder:
    """
    Decodes Python bytecode instructions to extract variable names
    """

    def __init__(self, code_object: Any, lasti: int) -> None:
        self.code_object = code_object
        self.instructions = list(dis.get_instructions(code_object))
        self.offset = self._find_instruction_index(lasti)

    def _find_instruction_index(self, lasti: int) -> int:
        """Find instruction index corresponding to lasti (byte offset)."""
        # Find the instruction at or before lasti
        # This should find the CALL instruction, not the next one
        best_idx = 0
        for i, instr in enumerate(self.instructions):
            if instr.offset <= lasti:
                best_idx = i
            else:
                break
        return best_idx

    def next(self) -> None:
        """Advance to the next instruction."""
        self.offset += 1

    def opcode(self) -> str | None:
        """Get the opcode name of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].opname
        return None

    def oparg(self) -> int:
        """Get the argument of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].arg or 0
        return 0

    def name(self) -> str | None:
        """
        Extract variable name from current instruction.
        """
        opname = self.opcode()
        if not opname:
            return None

        names = None
        if opname in ("STORE_NAME", "STORE_GLOBAL"):
            names = self.code_object.co_names
        elif opname == "STORE_FAST":
            names = self.code_object.co_varnames
        elif opname == "STORE_DEREF":
            names = self.code_object.co_cellvars
            if not names:
                names = self.code_object.co_freevars
        else:
            return None

        arg = self.oparg()
        if names and 0 <= arg < len(names):
            return names[arg]

        return None
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ._dim_entry import DimEntry


if TYPE_CHECKING:
    from . import Dim, Tensor


class EnableAllLayers:
    """
    RAII-style context manager for enabling functorch vmap layers.
    It manages the creation and cleanup of functorch dynamic layers.

    This is probably one of the more algorithmically important parts of first
    class dims. Intuitively, FCD can be thought of as another way of using
    vmap, where you don't actually have to vmap at the top level, instead the
    vmaps are implicitly determined by inspecting the bound dimensions on the
    FCD tensors involved in a compute (this is similar to our concept of
    non-lexical modes that we spent a long time talking about years ago). But
    under the hood you still need to actually enable the vmap mode. So once
    FCD has determined all of the dims we are batching over, it needs to
    enable all those layers so functorch can actually apply the batching
    rules. Therefore enable all layers!
    """

    levels_start: int
    levels_to_dim: list[Dim]

    def __init__(self, levels: list[DimEntry]):
        """
        Initialize and push dynamic layers for all first-class dimensions.

        Args:
            levels: List of dimension entries to create layers for
        """

        from . import Dim

        self.levels_start = 0
        self.levels_to_dim = []

        for l in levels:
            if not l.is_positional():
                d = l.dim()
                if not isinstance(d, Dim):
                    raise AssertionError(f"Expected Dim, got {type(d)}")
                self.levels_to_dim.append(d)

        # Sort by level for stable ordering
        self.levels_to_dim.sort(key=lambda d: d._level)

    def __enter__(self) -> EnableAllLayers:  # noqa: PYI034
        # Create functorch dynamic layers
        for i, dim in enumerate(self.levels_to_dim):
            batch_size = dim.size
            level = torch._C._functorch._vmap_increment_nesting(batch_size, "different")
            if i == 0:
                self.levels_start = level
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up dynamic layers in reverse order."""
        to_remove = self.levels_start + len(self.levels_to_dim) - 1
        for i in range(len(self.levels_to_dim)):
            popped = torch._C._functorch._vmap_decrement_nesting()
            if popped != to_remove - i:
                raise AssertionError(f"Expected layer {to_remove - i}, got {popped}")

    def from_batched(self, batchedtensor: torch.Tensor, has_device: bool) -> Tensor:
        """
        Create a Tensor from a batched tensor by unwrapping functorch layers.

        Args:
            batchedtensor: Batched tensor from functorch operation
            has_device: Whether tensor has device info

        Returns:
            Tensor with appropriate levels
        """
        # Create positional levels for base dimensions
        levels: list[DimEntry] = []
        for i in range(-batchedtensor.dim(), 0):
            levels.append(DimEntry(i))

        tensor = batchedtensor

        while torch._C._functorch.is_batchedtensor(tensor):
            level = torch._C._functorch.maybe_get_level(tensor)
            if level is None:
                raise AssertionError("Expected level to be non-None")
            if not (
                level >= self.levels_start
                and level < self.levels_start + len(self.levels_to_dim)
            ):
                raise AssertionError(f"Level {level} out of range")
            dim = DimEntry(self.levels_to_dim[level - self.levels_start])
            bdim = torch._C._functorch.maybe_get_bdim(tensor)
            if bdim is None:
                raise AssertionError("Expected bdim to be non-None")
            levels.insert(bdim, dim)
            tensor = torch._C._functorch.get_unwrapped(tensor)

        from . import Tensor

        result = Tensor()
        result._tensor = tensor
        result._batchtensor = batchedtensor
        result._has_device = has_device
        result._levels = levels
        return result

    def inplace_update_layers(
        self, batchtensor: torch.Tensor, levels: list[DimEntry]
    ) -> None:
        """
        Update the levels of a batched tensor in place.

        This requires the _maybe_unsafe_set_level binding that we'll add to functorch.

        Args:
            batchtensor: Batched tensor to update
            levels: New levels to set
        """
        # Check if tensor is batched
        if not torch._C._functorch.is_batchedtensor(batchtensor):
            return

        impl = batchtensor

        for i in reversed(range(len(self.levels_to_dim))):
            if impl is None:
                break

            if any(l == DimEntry(self.levels_to_dim[i]) for l in levels):
                # This is very interesting!  The level on batch tensor is
                # meaningless!  We set it RIGHT before we go into vmap
                torch._C._functorch._maybe_unsafe_set_level(impl, self.levels_start + i)
                impl = torch._C._functorch.get_unwrapped(impl)
<|endoftext|>from __future__ import annotations

import dis
import inspect
import sys
from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels
from ._enable_all_layers import EnableAllLayers
from ._py_inst_decoder import _PyInstDecoder
from ._tensor_info import TensorInfo


POINTWISE_OPTIMIZE = True
DOT_OPTIMIZED = True

# Global dimension level counter
_n_dims_created = 0


def _relevant_op(opcode: str | None) -> bool:
    """Check if opcode is relevant for variable assignment."""
    return bool(opcode and opcode.startswith("STORE_"))


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Handle tensor conversion for torch function integration."""
    return tensor


def _create_dim(name: str, size: int | None = None) -> Dim:
    """Create a new Dim object."""
    return Dim(name, size if size is not None else -1)


def dims(
    n: int | None = None, sizes: list[int | None] | None = None
) -> Dim | tuple[Dim, ...]:
    """
    Create and return one or more Dim objects.

    Uses bytecode inspection to determine variable names when possible.

    Args:
        n (int, optional): The number of dimensions to create. Can be omitted if sizes is specified.
        sizes (List[Optional[int]], optional): A list the same size as the number of dimensions to be
          created, specifying each dimensions size, or None to leave the size unset.

    Returns:
        Union[Dim, Tuple[Dim, ...]]: Single Dim if n=1, tuple of Dims otherwise.

    Examples:
        >>> batch, channel, width, height = dims(4)
        >>> batch, channel, width, height = dims(sizes=[None, 3, 224, 224])
        >>> single_dim = dims(1)
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    # Use bytecode inspection
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dims() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        def genobject(i: int) -> Dim:
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            size = sizes[i] if sizes is not None else None
            return _create_dim(name, size)

        # Validate sizes parameter
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        if specified_ndims == 1:
            return genobject(0)

        result = []
        for i in range(specified_ndims):
            result.append(genobject(i))

        return tuple(result)

    finally:
        del frame


class DimList:
    """
    A list of first-class dimensions that can be bound to tensor dimensions.

    A DimList can be in one of two states:
    1. Unbound: Created with just a name, no specific dimensions yet
    2. Bound: Either created with specific dimensions/sizes, or bound later via bind() or bind_len()
    """

    _name: str | None
    _dims: list[Dim]
    _bound: bool

    def __init__(
        self,
        len_or_dims: int | Sequence | None = None,
        name: str | None = None,
    ):
        """
        Initialize a new DimList object.

        Args:
            len_or_dims: Optional length (int) or sequence of dimensions/sizes
            name: Optional name for the dimension list
        """
        # Initialize attributes
        self._name = name
        self._dims: list = []
        self._bound = False

        if isinstance(len_or_dims, int):
            self.bind_len(len_or_dims)
        elif len_or_dims is not None:
            dims = []
            for i, item in enumerate(len_or_dims):
                if isinstance(item, int):
                    dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                    dims.append(Dim(dim_name, item))
                else:
                    dims.append(Dim(item))
            self._set_dims(dims)

    def _set_dims(self, dims: list) -> None:
        """Set the dimensions and mark as bound."""
        self._bound = True
        self._dims = dims

    def bind_len(self, size: int) -> None:
        """
        Bind this DimList to a specific length.

        Args:
            size: Number of dimensions to bind to

        Raises:
            DimensionBindError: If already bound to a different size
        """
        if self._bound:
            if len(self._dims) != size:
                raise DimensionBindError(
                    f"Dimlist has size {len(self._dims)} but it is being bound to size {size}"
                )
        else:
            self._bound = True
            self._dims = []
            for i in range(size):
                dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                self._dims.append(Dim(dim_name))

    def bind(self, sizes: Sequence[int]) -> None:
        """
        Bind this DimList to specific sizes.

        Args:
            sizes: Sequence of sizes for each dimension

        Raises:
            ValueError: If sizes is not a sequence
        """
        if not hasattr(sizes, "__len__") or not hasattr(sizes, "__getitem__"):
            raise ValueError("expected a sequence")

        size = len(sizes)
        self.bind_len(size)

        for i, dim_size in enumerate(sizes):
            self._dims[i].size = int(dim_size)

    def _size(self) -> int:
        if not self._bound:
            raise DimensionBindError("DimList not bound")
        return len(self._dims)

    def size(self) -> int:
        """Return the size (number of dimensions) of this DimList."""
        return self._size()

    def _set_bound(self, b: bool) -> None:
        """Set the bound status (for internal use)."""
        self._bound = b

    @property
    def is_bound(self) -> bool:
        """Property to check if DimList is bound."""
        return self._bound

    def __len__(self) -> int:
        """Return the length of the DimList."""
        return self.size()

    def __getitem__(self, key: int | slice) -> Dim | tuple[Dim, ...]:
        if not self._bound:
            raise DimensionBindError("DimList not bound")

        if isinstance(key, int):
            if key < 0 or key >= len(self._dims):
                raise IndexError("index out of bounds")
            return self._dims[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._dims))
            result = []
            for i in range(start, stop, step):
                result.append(self._dims[i])
            return tuple(result)
        else:
            raise ValueError("expected an int or a slice")

    def __repr__(self) -> str:
        """Return string representation of the DimList."""
        if self._bound:
            # Show as tuple representation
            return f"({', '.join(repr(dim) for dim in self._dims)})"
        elif self._name is not None:
            # Show as *name for unbound with name
            return f"*{self._name}"
        else:
            # Show as <unbound_dimlist> for unbound without name
            return "<unbound_dimlist>"

    def __str__(self) -> str:
        """Return string representation of the DimList."""
        return self.__repr__()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        return _Tensor.__torch_function__(func, types, args, kwargs)


def _create_dimlist(name: str, size: int | list[int | None] | None = None) -> DimList:
    """Create a DimList object with the given name and optional size."""
    dimlist = DimList(name=name)
    if size is not None:
        if isinstance(size, int):
            dimlist.bind_len(size)
        else:
            # size is a list of optional ints
            dimlist.bind_len(len(size))
            for i, s in enumerate(size):
                if s is not None:
                    dimlist._dims[i].size = s
    return dimlist


def dimlists(
    n: int | None = None, sizes: list[int | None] | None = None
) -> DimList | tuple[DimList, ...]:
    """
    Create and return one or more DimList objects.

    Similar to dims() but creates DimList objects instead.
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dimlists() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        # Generator function for dimlist names
        def genobject(i: int) -> str:
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            return name

        # Validate sizes
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        # Create dimlists
        if specified_ndims == 1:
            name = genobject(0)
            return _create_dimlist(name, sizes[0] if sizes is not None else None)

        result = []
        for i in range(specified_ndims):
            name = genobject(i)
            size = sizes[i] if sizes is not None else None
            result.append(_create_dimlist(name, size))

        return tuple(result)

    finally:
        del frame


class DimensionMismatchError(Exception):
    pass


class DimensionBindError(Exception):
    pass


from . import op_properties


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Safe print that avoids recursive torch function dispatches."""
    import sys

    # Convert any torch objects to basic representations
    safe_args = []
    for arg in args:
        if hasattr(arg, "__class__") and "torch" in str(type(arg)):
            safe_args.append(f"<{type(arg).__name__}>")
        else:
            safe_args.append(str(arg))

    print(*safe_args, **kwargs, file=sys.stderr)


class _Tensor:
    def _get_levels(self) -> list[Any]:
        raise NotImplementedError("_get_levels must be implemented by subclass")

    def _get_tensor(self) -> torch.Tensor | None:
        raise NotImplementedError("_get_tensor must be implemented by subclass")

    @property
    def ndim(self) -> int:
        raise NotImplementedError("ndim must be implemented by subclass")

    @property
    def dims(self) -> tuple[Any, ...]:
        return tuple(l.dim() for l in self._get_levels() if not l.is_positional())

    def dim(self) -> int:
        return self.ndim

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if DOT_OPTIMIZED and func is torch.Tensor.__mul__:
            # Check conditions: 2 args, both are tensor-like, both 0-dimensional
            if (
                len(args) == 2
                and not kwargs
                and isinstance(args[0], (_Tensor, torch.Tensor))
                and isinstance(args[1], (_Tensor, torch.Tensor))
            ):
                # Get tensor info for both operands
                lhs_info = TensorInfo.create(
                    args[0], ensure_batched=False, ensure_present=False
                )
                rhs_info = TensorInfo.create(
                    args[1], ensure_batched=False, ensure_present=False
                )

                if (
                    lhs_info
                    and rhs_info
                    and lhs_info.tensor is not None
                    and rhs_info.tensor is not None
                    and lhs_info.tensor.dim() == 0
                    and rhs_info.tensor.dim() == 0
                ):
                    if (
                        lhs_info.tensor.is_floating_point()
                        and rhs_info.tensor.is_floating_point()
                    ):
                        # Collect all unique levels and has_device
                        has_device = lhs_info.has_device or rhs_info.has_device
                        levels = []

                        for level in lhs_info.levels:
                            if level not in levels:
                                levels.append(level)
                        for level in rhs_info.levels:
                            if level not in levels:
                                levels.append(level)

                        # Debug print
                        # print(f"DEBUG: Creating delayed mul, levels: {levels}, has_device: {has_device}")

                        # Create delayed tensor
                        return Tensor.create_delayed(func, args, levels, has_device)

        if func is torch.Tensor.__getitem__:
            from functorch.dim._getsetitem import getitem

            return getitem(cls, func, types, args, kwargs)

        if func is torch.Tensor.__setitem__:
            from functorch.dim._getsetitem import setitem

            # args should be (tensor, index, value)
            if len(args) == 3:
                setitem(args[0], args[1], args[2])
                return None
            else:
                raise ValueError(f"Expected 3 args for __setitem__, got {len(args)}")

        # Fast-path for len; mostly to avoid infinite loop in TestMinFunctorchOnly.test_softmax_split
        if func is torch.Tensor.__len__:
            return args[0].size(0)

        # Special handling for torch.softmax - use the pre-wrapped version
        if func is torch.softmax:
            return softmax(*args, **kwargs)

        # Special handling for torch.stack - use the custom stack function
        if func is torch.stack:
            return stack(*args, **kwargs)

        if (
            func is torch.Tensor.split
            or func is torch._VF.split  # type: ignore[attr-defined]
            or func is torch._VF.split_with_sizes  # type: ignore[attr-defined]
            or func is torch.split
        ):
            return split(*args, **kwargs)

        return _Tensor._torch_function_fallback(func, types, args, kwargs)

    @staticmethod
    def _torch_function_fallback(
        func: Callable, types: tuple, args: tuple, kwargs: dict
    ) -> Any:
        """Fallback torch function implementation for non-special-cased functions."""
        is_pointwise = POINTWISE_OPTIMIZE and func in op_properties.pointwise
        # TODO: optimize pytree here
        flat_args, spec = tree_flatten((args, kwargs))
        device_holding_tensor = None

        infos: list[TensorInfo] = []
        result_levels: list[DimEntry] = []

        for f in flat_args:
            info = TensorInfo.create(f, not is_pointwise, False)
            infos.append(info)
            if info:
                if not (is_pointwise or info.batchedtensor is not None):
                    raise AssertionError(
                        "Expected pointwise or batchedtensor to be set"
                    )
                if device_holding_tensor is None and info.has_device:
                    device_holding_tensor = info.tensor
                # Collect all unique levels
                for level in info.levels:
                    if not isinstance(level, DimEntry):
                        raise AssertionError(f"Expected DimEntry, got {type(level)}")
                    if level not in result_levels:
                        result_levels.append(level)

        if is_pointwise:
            # Pointwise operation: match all tensors to common levels
            for i, info in enumerate(infos):
                if info and info.tensor is not None:
                    tensor = info.tensor
                    if device_holding_tensor is not None and not info.has_device:
                        tensor = tensor.to(device_holding_tensor.device)
                    ml = _match_levels(tensor, info.levels, result_levels)
                    flat_args[i] = handle_from_tensor(ml)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Wrap tensor results
            def wrap_tensor(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return Tensor.from_positional(
                        obj, result_levels, device_holding_tensor is not None
                    )
                return obj

            # Small fastpath
            if isinstance(result, torch.Tensor):
                return wrap_tensor(result)
            else:
                return tree_map(wrap_tensor, result)

        # Non-pointwise operation: use functorch vmap layers
        with EnableAllLayers(result_levels) as guard:
            # Update arguments with batched tensors
            for i, info in enumerate(infos):
                if info and info.batchedtensor is not None:
                    batched = info.batchedtensor
                    if device_holding_tensor is not None and not info.has_device:
                        batched = batched.to(device_holding_tensor.device)
                    guard.inplace_update_layers(batched, info.levels)
                    flat_args[i] = handle_from_tensor(batched)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Unwrap results from functorch layers
            def unwrap_tensor(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return guard.from_batched(obj, device_holding_tensor is not None)
                return obj

            if isinstance(result, torch.Tensor):
                return unwrap_tensor(result)
            else:
                return tree_map(unwrap_tensor, result)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Set values in tensor using first-class dimensions."""
        from functorch.dim._getsetitem import setitem

        return setitem(self, index, value)

    # expand and index are OK to be methods because they don't have torch.*
    # versions, but if they did they need the stack/cat treatment

    def expand(self, *args: Dim) -> _Tensor:
        """
        Expand tensor by adding new dimensions or expanding existing dimensions.

        If all arguments are Dim objects, adds new named dimensions.
        Otherwise, falls back to regular tensor expansion behavior.

        Args:
            args: Either Dim objects for new dimensions or sizes for regular expansion

        Returns:
            New tensor with expanded dimensions

        Example:
            >>> i, j = dims()
            >>> t = torch.randn(3, 4)
            >>> expanded = t[i].expand(j, k)  # Add j, k dimensions
            >>> expanded2 = t[i].expand(2, 4)  # Regular expand with sizes
        """
        info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

        for arg in args:
            if not isinstance(arg, Dim):
                # Not all args are Dims, fallback to regular expand
                if isinstance(self, torch.Tensor) and not isinstance(self, _Tensor):
                    return torch.Tensor.expand(self, *args)
                else:
                    return self.__torch_function__(
                        torch.Tensor.expand, (type(self),), (self,) + args
                    )

        # All args are Dim objects - proceed with first-class dimension expansion
        if not info:
            # No tensor info available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        # First-class dimension expansion - all args are Dim objects
        data = info.tensor
        if data is None:
            # No tensor data available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        levels = info.levels

        new_levels: list[DimEntry] = []
        new_sizes = []
        new_strides = []

        for d in args:
            # Check if dimension already exists in current levels or new_levels
            for level in levels:
                if not level.is_positional() and level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )
            for new_level in new_levels:
                if not new_level.is_positional() and new_level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )

            new_levels.append(DimEntry(d))
            new_sizes.append(d.size)
            new_strides.append(0)

        # Add existing levels
        new_levels.extend(levels)

        # Add existing sizes and strides
        orig_sizes = list(data.size())
        orig_strides = list(data.stride())
        new_sizes.extend(orig_sizes)
        new_strides.extend(orig_strides)

        # Create expanded tensor using as_strided
        expanded_data = data.as_strided(new_sizes, new_strides, data.storage_offset())

        # Return new tensor with expanded dimensions
        result = Tensor.from_positional(expanded_data, new_levels, info.has_device)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable

    def index(
        self,
        dims: int | Dim | tuple[int | Dim, ...] | list[int | Dim],
        indices: int
        | slice
        | torch.Tensor
        | tuple[int | slice | torch.Tensor, ...]
        | list[int | slice | torch.Tensor],
    ) -> _Tensor:
        """
        Index tensor using first-class dimensions.
        """
        from ._dim_entry import _match_levels
        from ._getsetitem import getsetitem_flat, invoke_getitem
        from ._wrap import _wrap_dim

        # Helper to check if obj is a dimpack (tuple/list) and extract items
        def maybe_dimpack(obj: Any, check_first: bool = False) -> tuple[Any, bool]:
            if isinstance(obj, (tuple, list)):
                return list(obj), True
            return None, False

        def parse_dim_entry(s: Any) -> Any:
            d = _wrap_dim(s, self.ndim, False)
            if d.is_none():
                raise TypeError(f"expected a dimension specifyer but found {repr(s)}")
            return d

        # Helper for dimension not present errors
        def dim_not_present(d: Any) -> None:
            if d.is_positional():
                raise TypeError(
                    f"dimension {d.position() + self.ndim} not in tensor of {self.ndim} dimensions"
                )
            else:
                raise TypeError(f"dimension {repr(d.dim())} not in tensor")

        dims_list: list[int | Dim] = []
        indices_list: list[int | slice | torch.Tensor] = []

        lhs_list = isinstance(dims, (tuple, list))
        rhs_list = isinstance(indices, (tuple, list))

        if lhs_list and rhs_list:
            # Type narrowing: we know dims and indices are sequences here
            dims_seq = dims  # type: ignore[assignment]
            indices_seq = indices  # type: ignore[assignment]
            if len(dims_seq) != len(indices_seq):  # type: ignore[arg-type]
                raise TypeError(
                    f"dims ({len(dims_seq)}) and indices ({len(indices_seq)}) must have the same length"  # type: ignore[arg-type]
                )
            dims_list.extend(dims_seq)  # type: ignore[arg-type]
            indices_list.extend(indices_seq)  # type: ignore[arg-type]
        else:
            dims_list.append(dims)  # type: ignore[arg-type]
            indices_list.append(indices)  # type: ignore[arg-type]

        # Create tensor info
        self_info = TensorInfo.create(self, False, False)

        new_levels: list[Any] = []
        to_flatten: list[Any] = []
        dims_list_flat = []

        # Process each dim specification
        for i in range(len(dims_list)):
            m, is_dimpack = maybe_dimpack(dims_list[i], check_first=False)
            if is_dimpack:
                if len(m) == 0:
                    dims_list_flat.append(DimEntry())  # Empty dimpack
                    continue

                first = parse_dim_entry(m[0])
                dims_list_flat.append(first)

                if len(m) == 1:
                    continue

                # Multi-element dimpack requires flattening
                if len(to_flatten) == 0:
                    new_levels.extend(self_info.levels)

                rest = []
                for j in range(1, len(m)):
                    d = parse_dim_entry(m[j])
                    removed = False
                    for k in range(len(new_levels)):
                        if new_levels[k] == d:
                            new_levels.pop(k)
                            removed = True
                            break
                    if not removed:
                        dim_not_present(d)
                    rest.append(d)

                # Find first in new_levels
                first_idx = None
                for k in range(len(new_levels)):
                    if new_levels[k] == first:
                        first_idx = k
                        break
                if first_idx is None:
                    dim_not_present(first)
                    continue  # Skip this iteration if dimension not found

                for j, r in enumerate(rest):
                    new_levels.insert(first_idx + 1 + j, r)
                to_flatten.extend(rest)
            else:
                dims_list_flat.append(parse_dim_entry(dims_list[i]))

        # Handle dimension flattening if needed
        if len(to_flatten) > 0:
            if self_info.tensor is None:
                raise AssertionError(
                    "Cannot perform dimension flattening on None tensor"
                )
            rearranged = _match_levels(self_info.tensor, self_info.levels, new_levels)
            sizes = rearranged.size()
            new_sizes: list[Any] = []
            reshape_levels = []

            for i in range(len(new_levels)):
                if new_levels[i] in to_flatten:
                    if len(new_sizes) == 0:
                        new_sizes.append(sizes[i])
                    else:
                        new_sizes[-1] *= sizes[i]
                else:
                    new_sizes.append(sizes[i])
                    reshape_levels.append(new_levels[i])

            self_info.tensor = rearranged.reshape(new_sizes)
            self_info.levels = reshape_levels

        # Check for dimpacks in indices
        has_dimpacks = False
        for idx in indices_list:
            if isinstance(idx, (tuple, list)):
                has_dimpacks = True
                break

        # Call getsetitem_flat with correct parameters
        info = getsetitem_flat(
            self_info,
            [],  # empty input_list
            dims_list_flat,  # keys
            indices_list,  # values
            has_dimpacks,
        )

        return invoke_getitem(info)

    def __repr__(self) -> str:
        tensor, levels, ndim = self._get_tensor(), self._get_levels(), self.ndim
        dims_repr = []
        for l in levels:
            if hasattr(l, "is_positional") and l.is_positional():
                # Convert negative positional to positive: -1 -> ndim-1, -2 -> ndim-2, etc.
                dims_repr.append(l.position() + ndim)
            elif hasattr(l, "dim"):
                dims_repr.append(l.dim())
            elif hasattr(l, "data"):
                dims_repr.append(l.data)
            else:
                dims_repr.append(l)
        return f"{tensor}\nwith dims={tuple(dims_repr)} sizes={tuple(tensor.size())}"  # type: ignore[union-attr]


TensorLike = (_Tensor, torch.Tensor)


class Dim(_Tensor):
    _level: int
    _name: str
    _size: int
    _range: torch.Tensor | None
    _batchtensor: torch.Tensor | None

    def __init__(self, name: str, s: int = -1) -> None:
        global _n_dims_created
        self._name = name
        self._size = s
        self._level = _n_dims_created
        _n_dims_created += 1
        self._range = None
        self._batchtensor = None

    @property
    def ndim(self) -> int:
        return 1

    @classmethod
    def check_exact(cls, obj: Any) -> bool:
        return type(obj) is cls

    @property
    def size(self) -> int:
        if self._size == -1:
            raise ValueError(f"dimension {self._name} is unbound")
        return self._size

    @size.setter
    def size(self, v: int) -> None:
        if self._size == -1:
            self._size = v
        elif self._size != v:
            raise DimensionBindError(
                f"Dim '{repr(self)}' previously bound to a dimension of size {self._size} "
                f"cannot bind to a dimension of size {v}"
            )

    @property
    def is_bound(self) -> bool:
        """Return True if this dimension is bound to a size."""
        return self._size != -1

    def _get_range(self) -> torch.Tensor:
        """
        Get a tensor representing the range [0, size) for this dimension.

        Returns:
            A 1D tensor with values [0, 1, 2, ..., size-1]
        """
        if self._range is None:
            self._range = torch.arange(self.size)
        return self._range

    def _get_batchtensor(self) -> torch.Tensor:
        """
        Get a batched tensor representation of this dimension.

        Returns:
            A batched tensor created from the range tensor
        """
        if self._batchtensor is None:
            self._batchtensor = torch._C._functorch._add_batch_dim(
                self._get_range(), 0, self._level
            )
        return self._batchtensor

    def __repr__(self) -> str:
        """String representation of a Dim object."""
        return self._name

    # note that Dim comes before tensor because we want the Dim API for things like size to take precedence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__


# Somewhat confusingly, an FCD tensor is also called Tensor.  This confusion
# is somewhat intentional, as FCD tensors are intended to be substitutable
# with regular Tensor (just with some positional dims hidden).
class Tensor(_Tensor):
    _tensor: torch.Tensor | None
    _batchtensor: torch.Tensor | None
    _levels: list[DimEntry]
    _has_device: bool
    _delayed: Callable[[], torch.Tensor] | None
    _delayed_orig: Callable | None
    _delayed_args: tuple | None

    @property
    def ndim(self) -> int:
        return sum(1 if l.is_positional() else 0 for l in self._levels)

    @classmethod
    def check_exact(cls, other: Any) -> bool:
        return type(other) is cls

    @classmethod
    def from_positional(
        cls, tensor: torch.Tensor, levels: list[DimEntry], has_device: bool
    ) -> _Tensor | torch.Tensor:
        """
        Create a functorch Tensor from a regular PyTorch tensor with specified dimension levels.

        This is the primary way to create Tensor objects with first-class dimensions.

        Args:
            tensor: The underlying PyTorch tensor
            levels: List of DimEntry objects specifying the dimension structure
            has_device: Whether the tensor is on a device (not CPU)

        Returns:
            A new Tensor instance with the specified dimensions, or a regular torch.Tensor
            if there are no named dimensions
        """
        seen_dims = 0
        last = 0

        for l in levels:
            if l.is_positional():
                # Validate consecutive positional dimensions
                if not (last == 0 or last + 1 == l.position()):
                    raise AssertionError(
                        f"Positional dimensions must be consecutive, got {last} then {l.position()}"
                    )
                last = l.position()
            else:
                # This is a named dimension
                seen_dims += 1

        # Validate final positional dimension
        if not (last == 0 or last == -1):
            raise AssertionError(
                f"Final positional dimension must be 0 or -1, got {last}"
            )

        if not seen_dims:
            return tensor

        # Create Tensor object with proper level management
        result = cls()
        result._tensor = tensor
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None  # Will be created lazily if needed
        result._delayed = None
        result._delayed_orig = None
        result._delayed_args = None

        # Validate tensor dimensionality matches levels
        if tensor.dim() != len(levels):
            raise AssertionError(
                f"Tensor has {tensor.dim()} dimensions but {len(levels)} levels provided"
            )

        return result

    @classmethod
    def create_delayed(
        cls, orig: Callable, args: tuple, levels: list[DimEntry], has_device: bool
    ) -> _Tensor:
        """
        Create a delayed tensor that defers the operation until later.
        """
        result = cls()
        result._tensor = None  # Will be computed when needed
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None
        result._delayed_orig = orig
        result._delayed_args = args

        # Create delayed evaluation function that unwraps Tensor objects
        def evaluate_delayed() -> torch.Tensor:
            unwrapped_args = []
            for arg in args:
                if hasattr(arg, "_get_tensor"):
                    unwrapped_args.append(arg._get_tensor())
                else:
                    unwrapped_args.append(arg)
            return orig(*unwrapped_args)

        result._delayed = evaluate_delayed

        return result

    def _get_tensor(self) -> torch.Tensor | None:
        """Get the underlying tensor, handling delayed operations if needed."""
        if (
            hasattr(self, "_delayed")
            and self._delayed is not None
            and self._tensor is None
        ):
            # Execute the delayed operation
            self._tensor = self._delayed()
            # Clear delayed operation to avoid re-execution
            self._delayed = None
            self._delayed_orig = None
            self._delayed_args = None
        return self._tensor

    def _get_levels(self) -> list[Any]:
        """Get the dimension levels."""
        return self._levels

    def _get_has_device(self) -> bool:
        """Get whether this tensor has device information."""
        return self._has_device

    def _get_batchtensor(self) -> torch.Tensor | None:
        """Get the batched tensor representation, creating it lazily if needed."""
        if self._batchtensor is None:
            self._batchtensor = self._add_batch_dims(
                self._get_tensor(), self._get_levels()
            )
        return self._batchtensor

    def _add_batch_dims(
        self, t: torch.Tensor | None, levels_: list[Any]
    ) -> torch.Tensor | None:
        levels = list(levels_)

        while True:
            min_real_index = -1
            min_index = -1
            min_value = float("inf")  # INT_MAX equivalent
            i = 0
            r = 0

            for r, l in enumerate(levels):
                if not l.is_none():
                    if not l.is_positional() and l.dim()._level < min_value:
                        min_value = l.dim()._level
                        min_index = i
                        min_real_index = r
                    i += 1

            if min_index == -1:
                return t

            if t is None:
                raise AssertionError("Expected t to be non-None")
            t = torch._C._functorch._add_batch_dim(t, min_index, int(min_value))

            levels[min_real_index] = DimEntry()
        return None

    def order(self, *dims: Any) -> _Tensor:
        """Reorder the dimensions of this tensor."""
        from ._order import order

        result = order(self, *dims)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable


def stack(tensors: Any, new_dim: Any, dim: int = 0) -> _Tensor:
    """
    Stack tensors along a new dimension.

    Args:
        tensors: Sequence of tensors to stack
        new_dim: The new Dim to create for stacking
        dim: The dimension position to insert the new dimension (default: 0)

    Returns:
        Stacked tensor with the new dimension
    """
    if not tensors:
        raise ValueError("stack expects a non-empty sequence of tensors")

    # Check if new_dim is a Dim object
    if not isinstance(new_dim, Dim):
        # Fall back to regular torch.stack
        result = torch.stack(tensors, dim=dim)
        return result  # type: ignore[return-value]

    # Collect all result_levels from input tensors
    result_levels = []
    infos = []

    for t in tensors:
        info = TensorInfo.create(t, ensure_batched=False, ensure_present=False)
        infos.append(info)
        for level in info.levels:
            if level not in result_levels:
                result_levels.append(level)

    # Set the new_dim size to match number of tensors
    new_dim.size = len(tensors)

    # Match all tensors to the common level structure using _match_levels
    inputs = []
    for info in infos:
        if info.tensor is None:
            raise AssertionError("Cannot stack tensors with None tensor data")
        matched_tensor = _match_levels(info.tensor, info.levels, result_levels)
        inputs.append(matched_tensor)

    # Calculate ndim and resolve the dim parameter
    ndim = ndim_of_levels(result_levels)
    rawdim = 0
    if dim is not None and not (isinstance(dim, int) and dim == 0):
        from ._wrap import _wrap_dim

        d = _wrap_dim(dim, ndim, False)
        try:
            idx = result_levels.index(d)
        except ValueError:
            raise TypeError(f"Dimension {dim} does not exist in inputs") from None
        rawdim = idx

    # Stack tensors at the resolved dimension
    result = torch.stack(inputs, rawdim)

    # Insert new dimension entry at the correct position
    result_levels.insert(rawdim, DimEntry(new_dim))

    # Return as a first-class tensor
    tensor_result = Tensor.from_positional(
        result, result_levels, infos[0].has_device if infos else True
    )
    return tensor_result  # type: ignore[return-value]


def split(tensor: Any, split_size_or_sections: Any, dim: Any = None) -> tuple:
    """
    Split tensor along a dimension.

    Can handle both regular integer sizes and Dim objects for split sizes.
    When Dim objects are used, they get bound to the resulting tensor dimensions.
    """
    from ._wrap import _wrap_dim

    # Check if dim is a Dim object
    dim_is_object = isinstance(dim, Dim)

    # Parse split_size_or_sections
    if isinstance(split_size_or_sections, int):
        # Single integer - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    # Check if it's a sequence
    sizes = []
    all_dims = True
    all_ints = True

    for item in split_size_or_sections:
        sizes.append(item)
        if isinstance(item, Dim):
            all_ints = False
        else:
            all_dims = False

    if all_ints:
        # All integers - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    if not all_dims:
        raise TypeError("split list must be ints or dims but got a mix")

    # All are Dim objects - handle first-class dimension split
    self_info = TensorInfo.create(tensor, ensure_batched=False, ensure_present=False)
    ndim = self_info.ndim()

    if not dim_is_object and ndim == 0:
        raise TypeError("split expects at least a 1-dimension tensor")

    # Wrap the dimension
    dim_l = _wrap_dim(dim, ndim, False) if dim is not None else DimEntry(-ndim)

    # Find the index of the dimension in levels
    idx = None
    for i, level in enumerate(self_info.levels):
        if level == dim_l:
