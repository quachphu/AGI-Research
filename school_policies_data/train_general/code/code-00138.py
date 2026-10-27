    def __enter__(self) -> Lock:
        self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self._lock.release()


class ThreadLock:
    """
    This is a threading-only lock for no-I/O contexts.

    In the sync case `ThreadLock` provides thread locking.
    In the async case `AsyncThreadLock` is a no-op.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> ThreadLock:
        self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self._lock.release()


class Event:
    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def wait(self, timeout: float | None = None) -> None:
        if timeout == float("inf"):  # pragma: no cover
            timeout = None
        if not self._event.wait(timeout=timeout):
            raise PoolTimeout()  # pragma: no cover


class Semaphore:
    def __init__(self, bound: int) -> None:
        self._semaphore = threading.Semaphore(value=bound)

    def acquire(self) -> None:
        self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()


class ShieldCancellation:
    # Thread-synchronous codebases don't support cancellation semantics.
    # We have this class because we need to mirror the async and sync
    # cases within our package, but it's just a no-op.
    def __enter__(self) -> ShieldCancellation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        pass
<|endoftext|>from importlib.metadata import version

from ._api import request, stream
from ._async import (
    AsyncConnectionInterface,
    AsyncConnectionPool,
    AsyncHTTP2Connection,
    AsyncHTTP11Connection,
    AsyncHTTPConnection,
    AsyncHTTPProxy,
    AsyncSOCKSProxy,
)
from ._backends.base import (
    SOCKET_OPTION,
    AsyncNetworkBackend,
    AsyncNetworkStream,
    NetworkBackend,
    NetworkStream,
)
from ._backends.mock import AsyncMockBackend, AsyncMockStream, MockBackend, MockStream
from ._backends.sync import SyncBackend
from ._exceptions import (
    ConnectError,
    ConnectionNotAvailable,
    ConnectTimeout,
    LocalProtocolError,
    NetworkError,
    PoolTimeout,
    ProtocolError,
    ProxyError,
    ReadError,
    ReadTimeout,
    RemoteProtocolError,
    TimeoutException,
    UnsupportedProtocol,
    WriteError,
    WriteTimeout,
)
from ._models import URL, Origin, Proxy, Request, Response
from ._ssl import default_ssl_context
from ._sync import (
    ConnectionInterface,
    ConnectionPool,
    HTTP2Connection,
    HTTP11Connection,
    HTTPConnection,
    HTTPProxy,
    SOCKSProxy,
)

# The 'httpcore2.AnyIOBackend' class is conditional on 'anyio' being installed.
try:
    from ._backends.anyio import AnyIOBackend
except ImportError:  # pragma: no cover

    class AnyIOBackend:  # type: ignore
        def __init__(self, *args, **kwargs):  # type: ignore
            msg = "Attempted to use 'httpcore2.AnyIOBackend' but 'anyio' is not installed."
            raise RuntimeError(msg)


# The 'httpcore2.TrioBackend' class is conditional on 'trio' being installed.
try:
    from ._backends.trio import TrioBackend
except ImportError:  # pragma: no cover

    class TrioBackend:  # type: ignore
        def __init__(self, *args, **kwargs):  # type: ignore
            msg = "Attempted to use 'httpcore2.TrioBackend' but 'trio' is not installed."
            raise RuntimeError(msg)


__all__ = [
    # top-level requests
    "request",
    "stream",
    # models
    "Origin",
    "URL",
    "Request",
    "Response",
    "Proxy",
    # async
    "AsyncHTTPConnection",
    "AsyncConnectionPool",
    "AsyncHTTPProxy",
    "AsyncHTTP11Connection",
    "AsyncHTTP2Connection",
    "AsyncConnectionInterface",
    "AsyncSOCKSProxy",
    # sync
    "HTTPConnection",
    "ConnectionPool",
    "HTTPProxy",
    "HTTP11Connection",
    "HTTP2Connection",
    "ConnectionInterface",
    "SOCKSProxy",
    # network backends, implementations
    "SyncBackend",
    "AnyIOBackend",
    "TrioBackend",
    # network backends, mock implementations
    "AsyncMockBackend",
    "AsyncMockStream",
    "MockBackend",
    "MockStream",
    # network backends, interface
    "AsyncNetworkStream",
    "AsyncNetworkBackend",
    "NetworkStream",
    "NetworkBackend",
    # util
    "default_ssl_context",
    "SOCKET_OPTION",
    # exceptions
    "ConnectionNotAvailable",
    "ProxyError",
    "ProtocolError",
    "LocalProtocolError",
    "RemoteProtocolError",
    "UnsupportedProtocol",
    "TimeoutException",
    "PoolTimeout",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "NetworkError",
    "ConnectError",
    "ReadError",
    "WriteError",
]

__version__ = version("httpcore2")


__locals = locals()
for __name in __all__:
    # Exclude SOCKET_OPTION, it causes AttributeError on Python 3.14
    if not __name.startswith(("__", "SOCKET_OPTION")):
        setattr(__locals[__name], "__module__", "httpcore2")  # noqa
<|endoftext|>from __future__ import annotations

import contextlib
import typing
from collections.abc import Generator

ExceptionMapping = typing.Mapping[type[Exception], type[Exception]]


@contextlib.contextmanager
def map_exceptions(map: ExceptionMapping) -> Generator[None]:
    try:
        yield
    except Exception as exc:  # noqa: PIE786
        for from_exc, to_exc in map.items():
            if isinstance(exc, from_exc):
                raise to_exc(exc) from exc
        raise  # pragma: no cover


class ConnectionNotAvailable(Exception):
    pass


class ProxyError(Exception):
    pass


class UnsupportedProtocol(Exception):
    pass


class ProtocolError(Exception):
    pass


class RemoteProtocolError(ProtocolError):
    pass


class LocalProtocolError(ProtocolError):
    pass


# Timeout errors


class TimeoutException(Exception):
    pass


class PoolTimeout(TimeoutException):
    pass


class ConnectTimeout(TimeoutException):
    pass


class ReadTimeout(TimeoutException):
    pass


class WriteTimeout(TimeoutException):
    pass


# Network errors


class NetworkError(Exception):
    pass


class ConnectError(NetworkError):
    pass


class ReadError(NetworkError):
    pass


class WriteError(NetworkError):
    pass
<|endoftext|>from __future__ import annotations

import select
import socket
import sys
import typing
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Generator,
    Iterable,
    Iterator,
)
from contextlib import AbstractAsyncContextManager, aclosing, contextmanager, nullcontext
from types import AsyncGeneratorType

T = typing.TypeVar("T")


def is_socket_readable(sock: socket.socket | None) -> bool:
    """
    Return whether a socket, as identified by its file descriptor, is readable.
    "A socket is readable" means that the read buffer isn't empty, i.e. that calling
    .recv() on it would immediately return some data.
    """
    # NOTE: we want check for readability without actually attempting to read, because
    # we don't want to block forever if it's not readable.

    # In the case that the socket no longer exists, or cannot return a file
    # descriptor, we treat it as being readable, as if it the next read operation
    # on it is ready to return the terminating `b""`.
    sock_fd = None if sock is None else sock.fileno()
    if sock_fd is None or sock_fd < 0:  # pragma: no cover
        return True

    # The implementation below was stolen from:
    # https://github.com/python-trio/trio/blob/20ee2b1b7376db637435d80e266212a35837ddcc/trio/_socket.py#L471-L478
    # See also: https://github.com/encode/httpcore/pull/193#issuecomment-703129316

    # Use select.select on Windows, and when poll is unavailable and select.poll
    # everywhere else. (E.g. When eventlet is in use. See #327)
    if sys.platform == "win32" or getattr(select, "poll", None) is None:  # pragma: no cover
        rready, _, _ = select.select([sock_fd], [], [], 0)
        return bool(rready)
    p = select.poll()
    p.register(sock_fd, select.POLLIN)
    return bool(p.poll(0))


def safe_async_iterate(
    iterable_or_iterator: AsyncIterable[T] | AsyncIterator[T], /
) -> AbstractAsyncContextManager[AsyncIterator[T]]:
    iterator = (
        iterable_or_iterator if isinstance(iterable_or_iterator, AsyncIterator) else iterable_or_iterator.__aiter__()
    )
    if isinstance(iterator, AsyncGeneratorType):
        return aclosing(iterator)
    return nullcontext(iterator)


@contextmanager
def safe_iterate(iterable_or_iterator: Iterable[T] | Iterator[T], /) -> Generator[Iterator[T], None, None]:
    # This is boilerplate code, only needed to make unasync happy
    iterator = iterable_or_iterator if isinstance(iterable_or_iterator, Iterator) else iterable_or_iterator.__iter__()
    yield iterator
<|endoftext|>from __future__ import annotations

import inspect
import logging
import types
import typing

from ._models import Request


class Trace:
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        request: Request | None = None,
        kwargs: dict[str, typing.Any] | None = None,
    ) -> None:
        self.name = name
        self.logger = logger
        self.trace_extension = None if request is None else request.extensions.get("trace")
        self.debug = self.logger.isEnabledFor(logging.DEBUG)
        self.kwargs = kwargs or {}
        self.return_value: typing.Any = None
        self.should_trace = self.debug or self.trace_extension is not None
        self.prefix = self.logger.name.split(".")[-1]

    def trace(self, name: str, info: dict[str, typing.Any]) -> None:
        if self.trace_extension is not None:
            prefix_and_name = f"{self.prefix}.{name}"
            ret = self.trace_extension(prefix_and_name, info)
            if inspect.iscoroutine(ret):  # pragma: no cover
                raise TypeError(
                    "If you are using a synchronous interface, "
                    "the callback of the `trace` extension should "
                    "be a normal function instead of an asynchronous function."
                )

        if self.debug:
            if not info or "return_value" in info and info["return_value"] is None:
                message = name
            else:
                args = " ".join([f"{key}={value!r}" for key, value in info.items()])
                message = f"{name} {args}"
            self.logger.debug(message)

    def __enter__(self) -> Trace:
        if self.should_trace:
            info = self.kwargs
            self.trace(f"{self.name}.started", info)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        if self.should_trace:
            if exc_value is None:
                info = {"return_value": self.return_value}
                self.trace(f"{self.name}.complete", info)
            else:
                info = {"exception": exc_value}
                self.trace(f"{self.name}.failed", info)

    async def atrace(self, name: str, info: dict[str, typing.Any]) -> None:
        if self.trace_extension is not None:
            prefix_and_name = f"{self.prefix}.{name}"
            coro = self.trace_extension(prefix_and_name, info)
            if not inspect.iscoroutine(coro):  # pragma: no cover
                raise TypeError(
                    "If you're using an asynchronous interface, "
                    "the callback of the `trace` extension should "
                    "be an asynchronous function rather than a normal function."
                )
            await coro

        if self.debug:
            if not info or "return_value" in info and info["return_value"] is None:
                message = name
            else:
                args = " ".join([f"{key}={value!r}" for key, value in info.items()])
                message = f"{name} {args}"
            self.logger.debug(message)

    async def __aenter__(self) -> Trace:
        if self.should_trace:
            info = self.kwargs
            await self.atrace(f"{self.name}.started", info)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        if self.should_trace:
            if exc_value is None:
                info = {"return_value": self.return_value}
                await self.atrace(f"{self.name}.complete", info)
            else:
                info = {"exception": exc_value}
                await self.atrace(f"{self.name}.failed", info)
<|endoftext|>from __future__ import annotations

import contextlib
import typing
from collections.abc import Generator

from ._models import URL, Extensions, HeaderTypes, Response
from ._sync.connection_pool import ConnectionPool


def request(
    method: bytes | str,
    url: URL | bytes | str,
    *,
    headers: HeaderTypes = None,
    content: bytes | typing.Iterator[bytes] | None = None,
    extensions: Extensions | None = None,
) -> Response:
    """
    Sends an HTTP request, returning the response.

    ```
    response = httpcore2.request("GET", "https://www.example.com/")
    ```

    Arguments:
        method: The HTTP method for the request. Typically one of `"GET"`,
            `"OPTIONS"`, `"HEAD"`, `"POST"`, `"PUT"`, `"PATCH"`, or `"DELETE"`.
        url: The URL of the HTTP request. Either as an instance of `httpcore2.URL`,
            or as str/bytes.
        headers: The HTTP request headers. Either as a dictionary of str/bytes,
            or as a list of two-tuples of str/bytes.
        content: The content of the request body. Either as bytes,
            or as a bytes iterator.
        extensions: A dictionary of optional extra information included on the request.
            Possible keys include `"timeout"`.

    Returns:
        An instance of `httpcore2.Response`.
    """
    with ConnectionPool() as pool:
        return pool.request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        )


@contextlib.contextmanager
def stream(
    method: bytes | str,
    url: URL | bytes | str,
    *,
    headers: HeaderTypes = None,
    content: bytes | typing.Iterator[bytes] | None = None,
    extensions: Extensions | None = None,
) -> Generator[Response]:
    """
    Sends an HTTP request, returning the response within a content manager.

    ```
    with httpcore2.stream("GET", "https://www.example.com/") as response:
        ...
    ```

    When using the `stream()` function, the body of the response will not be
    automatically read. If you want to access the response body you should
    either use `content = response.read()`, or `for chunk in response.iter_content()`.

    Arguments:
        method: The HTTP method for the request. Typically one of `"GET"`,
            `"OPTIONS"`, `"HEAD"`, `"POST"`, `"PUT"`, `"PATCH"`, or `"DELETE"`.
        url: The URL of the HTTP request. Either as an instance of `httpcore2.URL`,
            or as str/bytes.
        headers: The HTTP request headers. Either as a dictionary of str/bytes,
            or as a list of two-tuples of str/bytes.
        content: The content of the request body. Either as bytes,
            or as a bytes iterator.
        extensions: A dictionary of optional extra information included on the request.
            Possible keys include `"timeout"`.

    Returns:
        An instance of `httpcore2.Response`.
    """
    with ConnectionPool() as pool:
        with pool.stream(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        ) as response:
            yield response
<|endoftext|>import os
import ssl

import truststore


def default_ssl_context() -> ssl.SSLContext:
    if cafile := os.environ.get("SSL_CERT_FILE"):  # pragma: no cover
        return ssl.create_default_context(cafile=cafile)
    if capath := os.environ.get("SSL_CERT_DIR"):  # pragma: no cover
        return ssl.create_default_context(capath=capath)
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
<|endoftext|>from __future__ import annotations

import enum
import logging
import ssl
import time
import types
import typing
from collections.abc import AsyncGenerator

import h11

from .._backends.base import AsyncNetworkStream
from .._exceptions import (
    ConnectionNotAvailable,
    LocalProtocolError,
    RemoteProtocolError,
    WriteError,
    map_exceptions,
)
from .._models import Origin, Request, Response
from .._synchronization import AsyncLock, AsyncShieldCancellation
from .._trace import Trace
from .._utils import safe_async_iterate
from .interfaces import AsyncConnectionInterface

logger = logging.getLogger("httpcore2.http11")


# A subset of `h11.Event` types supported by `_send_event`
H11SendEvent = h11.Request | h11.Data | h11.EndOfMessage


class HTTPConnectionState(enum.IntEnum):
    NEW = 0
    ACTIVE = 1
    IDLE = 2
    CLOSED = 3


class AsyncHTTP11Connection(AsyncConnectionInterface):
    READ_NUM_BYTES = 64 * 1024
    MAX_INCOMPLETE_EVENT_SIZE = 100 * 1024

    def __init__(
        self,
        origin: Origin,
        stream: AsyncNetworkStream,
        keepalive_expiry: float | None = None,
    ) -> None:
        self._origin = origin
        self._network_stream = stream
        self._keepalive_expiry: float | None = keepalive_expiry
        self._expire_at: float | None = None
        self._state = HTTPConnectionState.NEW
        self._state_lock = AsyncLock()
        self._request_count = 0
        self._h11_state = h11.Connection(
            our_role=h11.CLIENT,
            max_incomplete_event_size=self.MAX_INCOMPLETE_EVENT_SIZE,
        )

    async def handle_async_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        async with self._state_lock:
            if self._state in (HTTPConnectionState.NEW, HTTPConnectionState.IDLE):
                self._request_count += 1
                self._state = HTTPConnectionState.ACTIVE
                self._expire_at = None
            else:
                raise ConnectionNotAvailable()

        try:
            kwargs = {"request": request}
            try:
                async with Trace("send_request_headers", logger, request, kwargs) as trace:
                    await self._send_request_headers(**kwargs)
                async with Trace("send_request_body", logger, request, kwargs) as trace:
                    await self._send_request_body(**kwargs)
            except WriteError:
                # If we get a write error while we're writing the request,
                # then we suppress this error and move on to attempting to
                # read the response. Servers can sometimes close the request
                # preemptively and then respond with a well formed HTTP
                # error response.
                pass

            async with Trace("receive_response_headers", logger, request, kwargs) as trace:
                (
                    http_version,
                    status,
                    reason_phrase,
                    headers,
                    trailing_data,
                ) = await self._receive_response_headers(**kwargs)
                trace.return_value = (
                    http_version,
                    status,
                    reason_phrase,
                    headers,
                )

            network_stream = self._network_stream

            # CONNECT or Upgrade request
            if (status == 101) or ((request.method == b"CONNECT") and (200 <= status < 300)):
                network_stream = AsyncHTTP11UpgradeStream(network_stream, trailing_data)

            return Response(
                status=status,
                headers=headers,
                content=HTTP11ConnectionByteStream(self, request),
                extensions={
                    "http_version": http_version,
                    "reason_phrase": reason_phrase,
                    "network_stream": network_stream,
                },
            )
        except BaseException as exc:
            with AsyncShieldCancellation():
                async with Trace("response_closed", logger, request) as trace:
                    await self._response_closed()
            raise exc

    # Sending the request...

    async def _send_request_headers(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        with map_exceptions({h11.LocalProtocolError: LocalProtocolError}):
            event = h11.Request(
                method=request.method,
                target=request.url.target,
                headers=request.headers,
            )
        await self._send_event(event, timeout=timeout)

    async def _send_request_body(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        assert isinstance(request.stream, typing.AsyncIterable)
        async with safe_async_iterate(request.stream) as iterator:
            async for chunk in iterator:
                event = h11.Data(data=chunk)
                await self._send_event(event, timeout=timeout)

        await self._send_event(h11.EndOfMessage(), timeout=timeout)

    async def _send_event(self, event: h11.Event, timeout: float | None = None) -> None:
        bytes_to_send = self._h11_state.send(event)
        if bytes_to_send is not None:
            await self._network_stream.write(bytes_to_send, timeout=timeout)

    # Receiving the response...

    async def _receive_response_headers(
        self, request: Request
    ) -> tuple[bytes, int, bytes, list[tuple[bytes, bytes]], bytes]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        while True:
            event = await self._receive_event(timeout=timeout)
            if isinstance(event, h11.Response):
                break
            if isinstance(event, h11.InformationalResponse) and event.status_code == 101:
                break

        http_version = b"HTTP/" + event.http_version

        # h11 version 0.11+ supports a `raw_items` interface to get the
        # raw header casing, rather than the enforced lowercase headers.
        headers = event.headers.raw_items()

        trailing_data, _ = self._h11_state.trailing_data

        return http_version, event.status_code, event.reason, headers, trailing_data

    async def _receive_response_body(self, request: Request) -> AsyncGenerator[bytes]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        while True:
            event = await self._receive_event(timeout=timeout)
            if isinstance(event, h11.Data):
                yield bytes(event.data)
            elif isinstance(event, (h11.EndOfMessage, h11.PAUSED)):
                break

    async def _receive_event(self, timeout: float | None = None) -> h11.Event | type[h11.PAUSED]:
        while True:
            with map_exceptions({h11.RemoteProtocolError: RemoteProtocolError}):
                event = self._h11_state.next_event()

            if event is h11.NEED_DATA:
                data = await self._network_stream.read(self.READ_NUM_BYTES, timeout=timeout)

                # If we feed this case through h11 we'll raise an exception like:
                #
                #     httpcore2.RemoteProtocolError: can't handle event type
                #     ConnectionClosed when role=SERVER and state=SEND_RESPONSE
                #
                # Which is accurate, but not very informative from an end-user
                # perspective. Instead we handle this case distinctly and treat
                # it as a ConnectError.
                if data == b"" and self._h11_state.their_state == h11.SEND_RESPONSE:
                    msg = "Server disconnected without sending a response."
                    raise RemoteProtocolError(msg)

                self._h11_state.receive_data(data)
            else:
                # mypy fails to narrow the type in the above if statement above
                return event  # type: ignore[return-value]

    async def _response_closed(self) -> None:
        async with self._state_lock:
            if self._h11_state.our_state is h11.DONE and self._h11_state.their_state is h11.DONE:
                self._state = HTTPConnectionState.IDLE
                self._h11_state.start_next_cycle()
                if self._keepalive_expiry is not None:
                    now = time.monotonic()
                    self._expire_at = now + self._keepalive_expiry
            else:
                await self.aclose()

    # Once the connection is no longer required...

    async def aclose(self) -> None:
        # Note that this method unilaterally closes the connection, and does
        # not have any kind of locking in place around it.
        self._state = HTTPConnectionState.CLOSED
        await self._network_stream.aclose()

    # The AsyncConnectionInterface methods provide information about the state of
    # the connection, allowing for a connection pooling implementation to
    # determine when to reuse and when to close the connection...

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    def is_connected(self) -> bool:
        return not self.is_closed()

    def is_available(self) -> bool:
        # Note that HTTP/1.1 connections in the "NEW" state are not treated as
        # being "available". The control flow which created the connection will
        # be able to send an outgoing request, but the connection will not be
        # acquired from the connection pool for any other request.
        return self._state == HTTPConnectionState.IDLE

    def has_expired(self) -> bool:
        now = time.monotonic()
        # Read `_expire_at` once into a local: on free-threaded builds another
        # thread may reset it to `None` between the check and the comparison.
        expire_at = self._expire_at
        keepalive_expired = expire_at is not None and now > expire_at

        # If the HTTP connection is idle but the socket is readable, then the
        # only valid state is that the socket is about to return b"", indicating
        # a server-initiated disconnect.
        server_disconnected = self._state == HTTPConnectionState.IDLE and self._network_stream.get_extra_info(
            "is_readable"
        )

        return keepalive_expired or server_disconnected

    def is_idle(self) -> bool:
        return self._state == HTTPConnectionState.IDLE

    def is_closed(self) -> bool:
        return self._state == HTTPConnectionState.CLOSED

    def info(self) -> str:
        origin = str(self._origin)
        return f"{origin!r}, HTTP/1.1, {self._state.name}, Request Count: {self._request_count}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        origin = str(self._origin)
        return f"<{class_name} [{origin!r}, {self._state.name}, Request Count: {self._request_count}]>"

    # These context managers are not used in the standard flow, but are
    # useful for testing or working with connection instances directly.

    async def __aenter__(self) -> AsyncHTTP11Connection:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        await self.aclose()


class HTTP11ConnectionByteStream:
    def __init__(self, connection: AsyncHTTP11Connection, request: Request) -> None:
        self._connection = connection
        self._request = request
        self._closed = False

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        kwargs = {"request": self._request}
        try:
            async with Trace("receive_response_body", logger, self._request, kwargs):
                async with safe_async_iterate(self._connection._receive_response_body(**kwargs)) as iterator:
                    async for chunk in iterator:
                        yield chunk
        except BaseException as exc:
            # If we get an exception while streaming the response,
            # we want to close the response (and possibly the connection)
            # before raising that exception.
            with AsyncShieldCancellation():
                await self.aclose()
            raise exc

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            async with Trace("response_closed", logger, self._request):
                await self._connection._response_closed()


class AsyncHTTP11UpgradeStream(AsyncNetworkStream):
    def __init__(self, stream: AsyncNetworkStream, leading_data: bytes) -> None:
        self._stream = stream
        self._leading_data = leading_data

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._leading_data:
            buffer = self._leading_data[:max_bytes]
            self._leading_data = self._leading_data[max_bytes:]
            return buffer
        else:
            return await self._stream.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._stream.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        return await self._stream.start_tls(ssl_context, server_hostname, timeout)

    def get_extra_info(self, info: str) -> typing.Any:
        return self._stream.get_extra_info(info)
<|endoftext|>from __future__ import annotations

import contextlib
import typing
from collections.abc import AsyncGenerator

from .._models import (
    URL,
    Extensions,
    HeaderTypes,
    Origin,
    Request,
    Response,
    enforce_bytes,
    enforce_headers,
    enforce_url,
    include_request_headers,
)


class AsyncRequestInterface:
    async def request(
        self,
        method: bytes | str,
        url: URL | bytes | str,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.AsyncIterator[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> Response:
        # Strict type checking on our parameters.
        method = enforce_bytes(method, name="method")
        url = enforce_url(url, name="url")
        headers = enforce_headers(headers, name="headers")

        # Include Host header, and optionally Content-Length or Transfer-Encoding.
        headers = include_request_headers(headers, url=url, content=content)

        request = Request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        )
        response = await self.handle_async_request(request)
        try:
            await response.aread()
        finally:
            await response.aclose()
        return response

    @contextlib.asynccontextmanager
    async def stream(
        self,
        method: bytes | str,
        url: URL | bytes | str,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.AsyncIterator[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> AsyncGenerator[Response]:
        # Strict type checking on our parameters.
        method = enforce_bytes(method, name="method")
        url = enforce_url(url, name="url")
        headers = enforce_headers(headers, name="headers")

        # Include Host header, and optionally Content-Length or Transfer-Encoding.
        headers = include_request_headers(headers, url=url, content=content)

        request = Request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        )
        response = await self.handle_async_request(request)
        try:
            yield response
        finally:
            await response.aclose()

    async def handle_async_request(self, request: Request) -> Response:
        raise NotImplementedError()  # pragma: no cover


class AsyncConnectionInterface(AsyncRequestInterface):
    async def aclose(self) -> None:
        raise NotImplementedError()  # pragma: no cover

    def info(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    def can_handle_request(self, origin: Origin) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def is_connected(self) -> bool:
        """
        Return `True` if the connection is open (the underlying socket has been
        established).  A connection in the NEW state (just created but not yet
        connected) returns `False`.

        Note: for some implementations `is_connected() != not is_closed()`.
        The default implementation returns `not self.is_closed()`, which is
        correct for connections that are never in the NEW (pre-TCP) state.
        """
        return not self.is_closed()  # pragma: no cover

    def is_available(self) -> bool:
        """
        Return `True` if the connection is currently able to accept an
        outgoing request.

        An HTTP/1.1 connection will only be available if it is currently idle.

        An HTTP/2 connection will be available so long as the stream ID space is
        not yet exhausted, and the connection is not in an error state.

        While the connection is being established we may not yet know if it is going
        to result in an HTTP/1.1 or HTTP/2 connection. The connection should be
        treated as being available, but might ultimately raise `NewConnectionRequired`
        required exceptions if multiple requests are attempted over a connection
        that ends up being established as HTTP/1.1.
        """
        raise NotImplementedError()  # pragma: no cover

    def has_expired(self) -> bool:
        """
        Return `True` if the connection is in a state where it should be closed.

        This either means that the connection is idle and it has passed the
        expiry time on its keep-alive, or that server has sent an EOF.
        """
        raise NotImplementedError()  # pragma: no cover

    def is_idle(self) -> bool:
        """
        Return `True` if the connection is currently idle.
        """
        raise NotImplementedError()  # pragma: no cover

    def can_multiplex(self) -> bool:
        """
        Return `True` if the connection can serve multiple requests
        concurrently, such as an established HTTP/2 connection.

        The default covers HTTP/1.1-style implementations, which serve a
        single request at a time.
        """
        return False

    def is_closed(self) -> bool:
        """
        Return `True` if the connection has been closed.

        Used when a response is closed to determine if the connection may be
        returned to the connection pool or not.
        """
        raise NotImplementedError()  # pragma: no cover
<|endoftext|>from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .http_proxy import AsyncHTTPProxy
from .interfaces import AsyncConnectionInterface

try:
    from .http2 import AsyncHTTP2Connection
except ImportError:  # pragma: no cover

    class AsyncHTTP2Connection:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # type: ignore
            raise RuntimeError(
                "Attempted to use http2 support, but the `h2` package is not "
                "installed. Use 'pip install httpcore[http2]'."
            )


try:
    from .socks_proxy import AsyncSOCKSProxy
except ImportError:  # pragma: no cover

    class AsyncSOCKSProxy:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # type: ignore
            raise RuntimeError(
                "Attempted to use SOCKS support, but the `socksio` package is not "
                "installed. Use 'pip install httpcore[socks]'."
            )


__all__ = [
    "AsyncHTTPConnection",
    "AsyncConnectionPool",
    "AsyncHTTPProxy",
    "AsyncHTTP11Connection",
    "AsyncHTTP2Connection",
    "AsyncConnectionInterface",
    "AsyncSOCKSProxy",
]
<|endoftext|>from __future__ import annotations

import base64
import logging
import ssl
import typing

from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ProxyError
from .._models import (
    URL,
    Origin,
    Request,
    Response,
    enforce_bytes,
    enforce_headers,
    enforce_url,
)
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface

ByteOrStr = bytes | str
HeadersAsSequence = typing.Sequence[tuple[ByteOrStr, ByteOrStr]]
HeadersAsMapping = typing.Mapping[ByteOrStr, ByteOrStr]


logger = logging.getLogger("httpcore2.proxy")


def merge_headers(
    default_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
    override_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
) -> list[tuple[bytes, bytes]]:
    """
    Append default_headers and override_headers, de-duplicating if a key exists
    in both cases.
    """
    default_headers = [] if default_headers is None else list(default_headers)
    override_headers = [] if override_headers is None else list(override_headers)
    has_override = set(key.lower() for key, _value in override_headers)
    default_headers = [(key, value) for key, value in default_headers if key.lower() not in has_override]
    return default_headers + override_headers


class AsyncHTTPProxy(AsyncConnectionPool):  # pragma: no cover
    """
    A connection pool that sends requests via an HTTP proxy.
    """

    def __init__(
        self,
        proxy_url: URL | bytes | str,
        proxy_auth: tuple[bytes | str, bytes | str] | None = None,
        proxy_headers: HeadersAsMapping | HeadersAsSequence | None = None,
        ssl_context: ssl.SSLContext | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: AsyncNetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            proxy_url: The URL to use when connecting to the proxy server.
                For example `"http://127.0.0.1:8080/"`.
            proxy_auth: Any proxy authentication as a two-tuple of
                (username, password). May be either bytes or ascii-only str.
            proxy_headers: Any HTTP headers to use for the proxy requests.
                For example `{"Proxy-Authorization": "Basic <username>:<password>"}`.
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            proxy_ssl_context: The same as `ssl_context`, but for a proxy server rather than a remote origin.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish
                a connection.
            local_address: Local address to connect from. Can also be used to
                connect using a particular address family. Using
                `local_address="0.0.0.0"` will connect using an `AF_INET` address
                (IPv4), while using `local_address="::"` will connect using an
                `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
        """
        super().__init__(
            ssl_context=ssl_context,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            http1=http1,
            http2=http2,
            network_backend=network_backend,
            retries=retries,
            local_address=local_address,
            uds=uds,
            socket_options=socket_options,
        )

        self._proxy_url = enforce_url(proxy_url, name="proxy_url")
        if self._proxy_url.scheme == b"http" and proxy_ssl_context is not None:  # pragma: no cover
            raise RuntimeError("The `proxy_ssl_context` argument is not allowed for the http scheme")

        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        if proxy_auth is not None:
            username = enforce_bytes(proxy_auth[0], name="proxy_auth")
            password = enforce_bytes(proxy_auth[1], name="proxy_auth")
            userpass = username + b":" + password
            authorization = b"Basic " + base64.b64encode(userpass)
            self._proxy_headers = [(b"Proxy-Authorization", authorization)] + self._proxy_headers

    def create_connection(self, origin: Origin) -> AsyncConnectionInterface:
        if origin.scheme == b"http":
            return AsyncForwardHTTPConnection(
                proxy_origin=self._proxy_url.origin,
                proxy_headers=self._proxy_headers,
                remote_origin=origin,
                keepalive_expiry=self._keepalive_expiry,
                network_backend=self._network_backend,
                proxy_ssl_context=self._proxy_ssl_context,
            )
        return AsyncTunnelHTTPConnection(
            proxy_origin=self._proxy_url.origin,
            proxy_headers=self._proxy_headers,
            remote_origin=origin,
            ssl_context=self._ssl_context,
            proxy_ssl_context=self._proxy_ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            network_backend=self._network_backend,
        )


class AsyncForwardHTTPConnection(AsyncConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        proxy_headers: HeadersAsMapping | HeadersAsSequence | None = None,
        keepalive_expiry: float | None = None,
        network_backend: AsyncNetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._connection = AsyncHTTPConnection(
            origin=proxy_origin,
            keepalive_expiry=keepalive_expiry,
            network_backend=network_backend,
            socket_options=socket_options,
            ssl_context=proxy_ssl_context,
        )
        self._proxy_origin = proxy_origin
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        self._remote_origin = remote_origin

    async def handle_async_request(self, request: Request) -> Response:
        headers = merge_headers(self._proxy_headers, request.headers)
        url = URL(
            scheme=self._proxy_origin.scheme,
            host=self._proxy_origin.host,
            port=self._proxy_origin.port,
            target=bytes(request.url),
        )
        proxy_request = Request(
            method=request.method,
            url=url,
            headers=headers,
            content=request.stream,
            extensions={key: value for key, value in request.extensions.items() if key != "sni_hostname"},
        )
        return await self._connection.handle_async_request(proxy_request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    async def aclose(self) -> None:
        await self._connection.aclose()

    def info(self) -> str:
        return self._connection.info()

    def is_connected(self) -> bool:
        return self._connection.is_connected()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"


class AsyncTunnelHTTPConnection(AsyncConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        ssl_context: ssl.SSLContext | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
        proxy_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        network_backend: AsyncNetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        self._connection: AsyncConnectionInterface = AsyncHTTPConnection(
            origin=proxy_origin,
            keepalive_expiry=keepalive_expiry,
            network_backend=network_backend,
            socket_options=socket_options,
            ssl_context=proxy_ssl_context,
        )
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._connect_lock = AsyncLock()
        self._connected = False

    async def handle_async_request(self, request: Request) -> Response:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("connect", None)

        async with self._connect_lock:
            if not self._connected:
                target = b"%b:%d" % (self._remote_origin.host, self._remote_origin.port)

                connect_url = URL(
                    scheme=self._proxy_origin.scheme,
                    host=self._proxy_origin.host,
                    port=self._proxy_origin.port,
                    target=target,
                )
                connect_headers = merge_headers([(b"Host", target), (b"Accept", b"*/*")], self._proxy_headers)
                connect_request = Request(
                    method=b"CONNECT",
                    url=connect_url,
                    headers=connect_headers,
                    extensions={key: value for key, value in request.extensions.items() if key != "sni_hostname"},
                )
                connect_response = await self._connection.handle_async_request(connect_request)

                if connect_response.status < 200 or connect_response.status > 299:
                    reason_bytes = connect_response.extensions.get("reason_phrase", b"")
                    reason_str = reason_bytes.decode("ascii", errors="ignore")
                    msg = f"{connect_response.status} {reason_str}"
                    await self._connection.aclose()
                    raise ProxyError(msg)

                stream = connect_response.extensions["network_stream"]

                # Upgrade the stream to SSL
                ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                alpn_protocols = (["h2", "http/1.1"] if self._http1 else ["h2"]) if self._http2 else ["http/1.1"]
                ssl_context.set_alpn_protocols(alpn_protocols)

                server_hostname = request.extensions.get("sni_hostname") or self._remote_origin.host.decode("ascii")
                kwargs = {
                    "ssl_context": ssl_context,
                    "server_hostname": server_hostname,
                    "timeout": timeout,
                }
                async with Trace("start_tls", logger, request, kwargs) as trace:
                    stream = await stream.start_tls(**kwargs)
                    trace.return_value = stream

                # Determine if we should be using HTTP/1.1 or HTTP/2
                ssl_object = stream.get_extra_info("ssl_object")
                http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"

                # Create the HTTP/1.1 or HTTP/2 connection
                if http2_negotiated or (self._http2 and not self._http1):
                    from .http2 import AsyncHTTP2Connection

                    self._connection = AsyncHTTP2Connection(
                        origin=self._remote_origin,
                        stream=stream,
                        keepalive_expiry=self._keepalive_expiry,
                    )
                else:
                    self._connection = AsyncHTTP11Connection(
                        origin=self._remote_origin,
                        stream=stream,
                        keepalive_expiry=self._keepalive_expiry,
                    )

                self._connected = True
        return await self._connection.handle_async_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    async def aclose(self) -> None:
        await self._connection.aclose()

    def info(self) -> str:
        return self._connection.info()

    def is_connected(self) -> bool:
        return self._connection.is_connected()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"
<|endoftext|>from __future__ import annotations

import logging
import ssl

import socksio

from .._backends.auto import AutoBackend
from .._backends.base import AsyncNetworkBackend, AsyncNetworkStream
from .._exceptions import ConnectionNotAvailable, ProxyError
from .._models import URL, Origin, Request, Response, enforce_bytes, enforce_url
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface

logger = logging.getLogger("httpcore2.socks")


AUTH_METHODS = {
    b"\x00": "NO AUTHENTICATION REQUIRED",
    b"\x01": "GSSAPI",
    b"\x02": "USERNAME/PASSWORD",
    b"\xff": "NO ACCEPTABLE METHODS",
}

REPLY_CODES = {
    b"\x00": "Succeeded",
    b"\x01": "General SOCKS server failure",
    b"\x02": "Connection not allowed by ruleset",
    b"\x03": "Network unreachable",
    b"\x04": "Host unreachable",
    b"\x05": "Connection refused",
    b"\x06": "TTL expired",
    b"\x07": "Command not supported",
    b"\x08": "Address type not supported",
}


async def _init_socks5_connection(
    stream: AsyncNetworkStream,
    *,
    host: bytes,
    port: int,
    auth: tuple[bytes, bytes] | None = None,
    timeouts: dict[str, float | None] | None = None,
) -> None:
    timeouts = timeouts or {}
    write_timeout = timeouts.get("write", None)
    read_timeout = timeouts.get("read", None)
    conn = socksio.socks5.SOCKS5Connection()

    # Auth method request
    auth_method = (
        socksio.socks5.SOCKS5AuthMethod.NO_AUTH_REQUIRED
        if auth is None
        else socksio.socks5.SOCKS5AuthMethod.USERNAME_PASSWORD
    )
    conn.send(socksio.socks5.SOCKS5AuthMethodsRequest([auth_method]))
    outgoing_bytes = conn.data_to_send()
    await stream.write(outgoing_bytes, timeout=write_timeout)

    # Auth method response
    incoming_bytes = await stream.read(max_bytes=4096, timeout=read_timeout)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socksio.socks5.SOCKS5AuthReply)
    if response.method != auth_method:
        requested = AUTH_METHODS.get(auth_method, "UNKNOWN")
        responded = AUTH_METHODS.get(response.method, "UNKNOWN")
        raise ProxyError(f"Requested {requested} from proxy server, but got {responded}.")

    if response.method == socksio.socks5.SOCKS5AuthMethod.USERNAME_PASSWORD:
        # Username/password request
        assert auth is not None
        username, password = auth
        conn.send(socksio.socks5.SOCKS5UsernamePasswordRequest(username, password))
        outgoing_bytes = conn.data_to_send()
        await stream.write(outgoing_bytes, timeout=write_timeout)

        # Username/password response
        incoming_bytes = await stream.read(max_bytes=4096, timeout=read_timeout)
        response = conn.receive_data(incoming_bytes)
        assert isinstance(response, socksio.socks5.SOCKS5UsernamePasswordReply)
        if not response.success:
            raise ProxyError("Invalid username/password")

    # Connect request
    conn.send(socksio.socks5.SOCKS5CommandRequest.from_address(socksio.socks5.SOCKS5Command.CONNECT, (host, port)))
    outgoing_bytes = conn.data_to_send()
    await stream.write(outgoing_bytes, timeout=write_timeout)

    # Connect response
    incoming_bytes = await stream.read(max_bytes=4096, timeout=read_timeout)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socksio.socks5.SOCKS5Reply)
    if response.reply_code != socksio.socks5.SOCKS5ReplyCode.SUCCEEDED:
        reply_code = REPLY_CODES.get(response.reply_code, "UNKOWN")
        raise ProxyError(f"Proxy Server could not connect: {reply_code}.")


class AsyncSOCKSProxy(AsyncConnectionPool):  # pragma: no cover
    """
    A connection pool that sends requests via an HTTP proxy.
    """

    def __init__(
        self,
        proxy_url: URL | bytes | str,
        proxy_auth: tuple[bytes | str, bytes | str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        network_backend: AsyncNetworkBackend | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            proxy_url: The URL to use when connecting to the proxy server.
                For example `"http://127.0.0.1:8080/"`.
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish
                a connection.
            local_address: Local address to connect from. Can also be used to
                connect using a particular address family. Using
                `local_address="0.0.0.0"` will connect using an `AF_INET` address
                (IPv4), while using `local_address="::"` will connect using an
                `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
        """
        super().__init__(
            ssl_context=ssl_context,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            http1=http1,
            http2=http2,
            network_backend=network_backend,
            retries=retries,
        )
        self._ssl_context = ssl_context
        self._proxy_url = enforce_url(proxy_url, name="proxy_url")
        if proxy_auth is not None:
            username, password = proxy_auth
            username_bytes = enforce_bytes(username, name="proxy_auth")
            password_bytes = enforce_bytes(password, name="proxy_auth")
            self._proxy_auth: tuple[bytes, bytes] | None = (
                username_bytes,
                password_bytes,
            )
        else:
            self._proxy_auth = None

    def create_connection(self, origin: Origin) -> AsyncConnectionInterface:
        return AsyncSocks5Connection(
            proxy_origin=self._proxy_url.origin,
            remote_origin=origin,
            proxy_auth=self._proxy_auth,
            ssl_context=self._ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            network_backend=self._network_backend,
        )


class AsyncSocks5Connection(AsyncConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        proxy_auth: tuple[bytes, bytes] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        network_backend: AsyncNetworkBackend | None = None,
    ) -> None:
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._proxy_auth = proxy_auth
        self._ssl_context = ssl_context
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2

        self._network_backend: AsyncNetworkBackend = AutoBackend() if network_backend is None else network_backend
        self._connect_lock = AsyncLock()
        self._connection: AsyncConnectionInterface | None = None
        self._connect_failed = False

    async def handle_async_request(self, request: Request) -> Response:
        timeouts = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname", None)
        timeout = timeouts.get("connect", None)

        async with self._connect_lock:
            if self._connection is None:
                try:
                    # Connect to the proxy
                    kwargs = {
                        "host": self._proxy_origin.host.decode("ascii"),
                        "port": self._proxy_origin.port,
                        "timeout": timeout,
                    }
                    async with Trace("connect_tcp", logger, request, kwargs) as trace:
                        stream = await self._network_backend.connect_tcp(**kwargs)
                        trace.return_value = stream

                    # Connect to the remote host using socks5
                    kwargs = {
                        "stream": stream,
                        "host": self._remote_origin.host.decode("ascii"),
                        "port": self._remote_origin.port,
                        "auth": self._proxy_auth,
                        "timeouts": timeouts,
                    }
                    async with Trace("setup_socks5_connection", logger, request, kwargs) as trace:
                        await _init_socks5_connection(**kwargs)
                        trace.return_value = stream

                    # Upgrade the stream to SSL
                    if self._remote_origin.scheme in (b"https", b"wss"):
                        ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                        alpn_protocols = (
                            (["h2", "http/1.1"] if self._http1 else ["h2"]) if self._http2 else ["http/1.1"]
                        )
                        ssl_context.set_alpn_protocols(alpn_protocols)

                        kwargs = {
                            "ssl_context": ssl_context,
                            "server_hostname": sni_hostname or self._remote_origin.host.decode("ascii"),
                            "timeout": timeout,
                        }
                        async with Trace("start_tls", logger, request, kwargs) as trace:
                            stream = await stream.start_tls(**kwargs)
                            trace.return_value = stream

                    # Determine if we should be using HTTP/1.1 or HTTP/2
                    ssl_object = stream.get_extra_info("ssl_object")
                    http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"

                    # Create the HTTP/1.1 or HTTP/2 connection
                    if http2_negotiated or (self._http2 and not self._http1):  # pragma: no cover
                        from .http2 import AsyncHTTP2Connection

                        self._connection = AsyncHTTP2Connection(
                            origin=self._remote_origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                    else:
                        self._connection = AsyncHTTP11Connection(
                            origin=self._remote_origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                except Exception as exc:
                    self._connect_failed = True
                    raise exc
            elif not self._connection.is_available():  # pragma: no cover
                raise ConnectionNotAvailable()

        return await self._connection.handle_async_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    async def aclose(self) -> None:
        if self._connection is not None:
            await self._connection.aclose()

    def is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected()

    def is_available(self) -> bool:
        if self._connection is None:  # pragma: no cover
            # If HTTP/2 support is enabled, and the resulting connection could
            # end up as HTTP/2 then we should indicate the connection as being
            # available to service multiple requests.
            return (
                self._http2 and (self._remote_origin.scheme == b"https" or not self._http1) and not self._connect_failed
            )
        return self._connection.is_available()

    def has_expired(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.is_closed()

    def info(self) -> str:
        if self._connection is None:  # pragma: no cover
            return "CONNECTION FAILED" if self._connect_failed else "CONNECTING"
        return self._connection.info()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"
<|endoftext|>from __future__ import annotations

import enum
import logging
import time
import types
import typing
from collections.abc import AsyncGenerator

import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings

from .._backends.base import AsyncNetworkStream
from .._exceptions import ConnectionNotAvailable, LocalProtocolError, RemoteProtocolError
from .._models import Origin, Request, Response
from .._synchronization import AsyncLock, AsyncSemaphore, AsyncShieldCancellation
from .._trace import Trace
from .._utils import safe_async_iterate
from .interfaces import AsyncConnectionInterface

logger = logging.getLogger("httpcore2.http2")


def has_body_headers(request: Request) -> bool:
    return any(k.lower() == b"content-length" or k.lower() == b"transfer-encoding" for k, _v in request.headers)


class HTTPConnectionState(enum.IntEnum):
    ACTIVE = 1
    IDLE = 2
    CLOSED = 3


class AsyncHTTP2Connection(AsyncConnectionInterface):
    READ_NUM_BYTES = 64 * 1024
    CONFIG = h2.config.H2Configuration(validate_inbound_headers=False)

    def __init__(
        self,
        origin: Origin,
        stream: AsyncNetworkStream,
        keepalive_expiry: float | None = None,
    ):
        self._origin = origin
        self._network_stream = stream
        self._keepalive_expiry: float | None = keepalive_expiry
        self._h2_state = h2.connection.H2Connection(config=self.CONFIG)
        self._state = HTTPConnectionState.IDLE
        self._expire_at: float | None = None
        self._request_count = 0
        self._init_lock = AsyncLock()
        self._state_lock = AsyncLock()
        self._read_lock = AsyncLock()
        self._write_lock = AsyncLock()
        self._sent_connection_init = False
        self._used_all_stream_ids = False
        self._connection_error = False

        # Mapping from stream ID to response stream events.
        self._events: dict[
            int,
            list[h2.events.ResponseReceived | h2.events.DataReceived | h2.events.StreamEnded | h2.events.StreamReset,],
        ] = {}

        # Connection terminated events are stored as state since
        # we need to handle them for all streams.
        self._connection_terminated: h2.events.ConnectionTerminated | None = None

        self._read_exception: Exception | None = None
        self._write_exception: Exception | None = None

    async def handle_async_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            # This cannot occur in normal operation, since the connection pool
            # will only send requests on connections that handle them.
            # It's in place simply for resilience as a guard against incorrect
            # usage, for anyone working directly with httpcore connections.
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        async with self._state_lock:
            if self._state in (HTTPConnectionState.ACTIVE, HTTPConnectionState.IDLE):
                self._request_count += 1
                self._expire_at = None
                self._state = HTTPConnectionState.ACTIVE
            else:
                raise ConnectionNotAvailable()

        async with self._init_lock:
            if not self._sent_connection_init:
                try:
                    sci_kwargs = {"request": request}
                    async with Trace("send_connection_init", logger, request, sci_kwargs):
                        await self._send_connection_init(**sci_kwargs)
                except BaseException as exc:
                    with AsyncShieldCancellation():
                        await self.aclose()
                    raise exc

                self._sent_connection_init = True

                # Initially start with just 1 until the remote server provides
                # its max_concurrent_streams value
                self._max_streams = 1

                local_settings_max_streams = self._h2_state.local_settings.max_concurrent_streams
                self._max_streams_semaphore = AsyncSemaphore(local_settings_max_streams)

                for _ in range(local_settings_max_streams - self._max_streams):
                    await self._max_streams_semaphore.acquire()

        await self._max_streams_semaphore.acquire()

        try:
            stream_id = self._h2_state.get_next_available_stream_id()
            self._events[stream_id] = []
        except h2.exceptions.NoAvailableStreamIDError:  # pragma: no cover
            self._used_all_stream_ids = True
            self._request_count -= 1
            await self._max_streams_semaphore.release()
            raise ConnectionNotAvailable()

        try:
            kwargs = {"request": request, "stream_id": stream_id}
            async with Trace("send_request_headers", logger, request, kwargs):
                await self._send_request_headers(request=request, stream_id=stream_id)
            async with Trace("send_request_body", logger, request, kwargs):
                await self._send_request_body(request=request, stream_id=stream_id)
            async with Trace("receive_response_headers", logger, request, kwargs) as trace:
                status, headers = await self._receive_response(request=request, stream_id=stream_id)
                trace.return_value = (status, headers)

            return Response(
                status=status,
                headers=headers,
                content=HTTP2ConnectionByteStream(self, request, stream_id=stream_id),
                extensions={
                    "http_version": b"HTTP/2",
                    "network_stream": self._network_stream,
                    "stream_id": stream_id,
                },
            )
        except BaseException as exc:  # noqa: PIE786
            with AsyncShieldCancellation():
                kwargs = {"stream_id": stream_id}
                async with Trace("response_closed", logger, request, kwargs):
                    await self._response_closed(stream_id=stream_id)

            if isinstance(exc, h2.exceptions.ProtocolError):
                # One case where h2 can raise a protocol error is when a
                # closed frame has been seen by the state machine.
                #
                # This happens when one stream is reading, and encounters
                # a GOAWAY event. Other flows of control may then raise
                # a protocol error at any point they interact with the 'h2_state'.
                #
                # In this case we'll have stored the event, and should raise
                # it as a RemoteProtocolError.
                if self._connection_terminated:  # pragma: no cover
                    raise RemoteProtocolError(self._connection_terminated)
                # If h2 raises a protocol error in some other state then we
                # must somehow have made a protocol violation.
                raise LocalProtocolError(exc)  # pragma: no cover

            raise exc

    async def _send_connection_init(self, request: Request) -> None:
        """
        The HTTP/2 connection requires some initial setup before we can start
        using individual request/response streams on it.
        """
        # Need to set these manually here instead of manipulating via
        # __setitem__() otherwise the H2Connection will emit SettingsUpdate
        # frames in addition to sending the undesired defaults.
        self._h2_state.local_settings = h2.settings.Settings(
            client=True,
            initial_values={
                # Disable PUSH_PROMISE frames from the server since we don't do anything
                # with them for now.  Maybe when we support caching?
                h2.settings.SettingCodes.ENABLE_PUSH: 0,
                # These two are taken from h2 for safe defaults
                h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 100,
                h2.settings.SettingCodes.MAX_HEADER_LIST_SIZE: 65536,
            },
        )

        # Some websites (*cough* Yahoo *cough*) balk at this setting being
        # present in the initial handshake since it's not defined in the original
        # RFC despite the RFC mandating ignoring settings you don't know about.
        del self._h2_state.local_settings[h2.settings.SettingCodes.ENABLE_CONNECT_PROTOCOL]

        self._h2_state.initiate_connection()
        self._h2_state.increment_flow_control_window(2**24)
        await self._write_outgoing_data(request)

    # Sending the request...

    async def _send_request_headers(self, request: Request, stream_id: int) -> None:
        """
        Send the request headers to a given stream ID.
        """
        end_stream = not has_body_headers(request)

        # In HTTP/2 the ':authority' pseudo-header is used instead of 'Host'.
        # In order to gracefully handle HTTP/1.1 and HTTP/2 we always require
        # HTTP/1.1 style headers, and map them appropriately if we end up on
        # an HTTP/2 connection.
        authority = [v for k, v in request.headers if k.lower() == b"host"][0]

        headers = [
            (b":method", request.method),
            (b":authority", authority),
            (b":scheme", request.url.scheme),
            (b":path", request.url.target),
        ] + [
            (k.lower(), v)
            for k, v in request.headers
            if k.lower()
            not in (
                b"host",
                b"transfer-encoding",
            )
        ]

        self._h2_state.send_headers(stream_id, headers, end_stream=end_stream)
        self._h2_state.increment_flow_control_window(2**24, stream_id=stream_id)
        await self._write_outgoing_data(request)

    async def _send_request_body(self, request: Request, stream_id: int) -> None:
        """
        Iterate over the request body sending it to a given stream ID.
        """
        if not has_body_headers(request):
            return

        assert isinstance(request.stream, typing.AsyncIterable)
        async with safe_async_iterate(request.stream) as iterator:
            async for chunk in iterator:
                await self._send_stream_data(request, stream_id, chunk)

        await self._send_end_stream(request, stream_id)

    async def _send_stream_data(self, request: Request, stream_id: int, data: bytes) -> None:
        """
        Send a single chunk of data in one or more data frames.
        """
        position = 0
        while position < len(data):
            max_flow = await self._wait_for_outgoing_flow(request, stream_id)
            chunk = data[position : position + max_flow]
            position += len(chunk)
            self._h2_state.send_data(stream_id, chunk)
            await self._write_outgoing_data(request)

    async def _send_end_stream(self, request: Request, stream_id: int) -> None:
        """
        Send an empty data frame on on a given stream ID with the END_STREAM flag set.
        """
        self._h2_state.end_stream(stream_id)
        await self._write_outgoing_data(request)

    # Receiving the response...

    async def _receive_response(self, request: Request, stream_id: int) -> tuple[int, list[tuple[bytes, bytes]]]:
        """
        Return the response status code and headers for a given stream ID.
        """
        while True:
            event = await self._receive_stream_event(request, stream_id)
            if isinstance(event, h2.events.ResponseReceived):
                break

        status_code = 200
        headers: list[tuple[bytes, bytes]] = []
        assert event.headers is not None
        for k, v in event.headers:
            if k == b":status":
                status_code = int(v.decode("ascii", errors="ignore"))
            elif not k.startswith(b":"):
                headers.append((k, v))

        return (status_code, headers)

    async def _receive_response_body(self, request: Request, stream_id: int) -> AsyncGenerator[bytes]:
        """
        Iterator that returns the bytes of the response body for a given stream ID.
        """
        while True:
            event = await self._receive_stream_event(request, stream_id)
            if isinstance(event, h2.events.DataReceived):
                assert event.flow_controlled_length is not None
                assert event.data is not None
                amount = event.flow_controlled_length
                self._h2_state.acknowledge_received_data(amount, stream_id)
                await self._write_outgoing_data(request)
                yield event.data
            elif isinstance(event, h2.events.StreamEnded):
                break

    async def _receive_stream_event(
        self, request: Request, stream_id: int
    ) -> h2.events.ResponseReceived | h2.events.DataReceived | h2.events.StreamEnded:
        """
        Return the next available event for a given stream ID.

        Will read more data from the network if required.
        """
        while not self._events.get(stream_id):
            await self._receive_events(request, stream_id)
        event = self._events[stream_id].pop(0)
        if isinstance(event, h2.events.StreamReset):
            raise RemoteProtocolError(event)
        return event

    async def _receive_events(self, request: Request, stream_id: int | None = None) -> None:
        """
        Read some data from the network until we see one or more events
        for a given stream ID.
        """
        async with self._read_lock:
            if self._connection_terminated is not None:
                last_stream_id = self._connection_terminated.last_stream_id
                if stream_id and last_stream_id and stream_id > last_stream_id:
                    self._request_count -= 1
                    raise ConnectionNotAvailable()
                raise RemoteProtocolError(self._connection_terminated)

            # This conditional is a bit icky. We don't want to block reading if we've
            # actually got an event to return for a given stream. We need to do that
            # check *within* the atomic read lock. Though it also need to be optional,
            # because when we call it from `_wait_for_outgoing_flow` we *do* want to
            # block until we've available flow control, event when we have events
            # pending for the stream ID we're attempting to send on.
            if stream_id is None or not self._events.get(stream_id):
                events = await self._read_incoming_data(request)
                for event in events:
                    if isinstance(event, h2.events.RemoteSettingsChanged):
                        async with Trace("receive_remote_settings", logger, request) as trace:
                            await self._receive_remote_settings_change(event)
                            trace.return_value = event

                    elif isinstance(
                        event,
                        (
                            h2.events.ResponseReceived,
                            h2.events.DataReceived,
                            h2.events.StreamEnded,
                            h2.events.StreamReset,
                        ),
                    ):
                        if event.stream_id in self._events:
                            self._events[event.stream_id].append(event)

                    elif isinstance(event, h2.events.ConnectionTerminated):
                        self._connection_terminated = event

        await self._write_outgoing_data(request)

    async def _receive_remote_settings_change(self, event: h2.events.RemoteSettingsChanged) -> None:
        max_concurrent_streams = event.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
        if max_concurrent_streams:
            new_max_streams = min(
                max_concurrent_streams.new_value,
                self._h2_state.local_settings.max_concurrent_streams,
            )
            if new_max_streams and new_max_streams != self._max_streams:
                while new_max_streams > self._max_streams:
                    await self._max_streams_semaphore.release()
                    self._max_streams += 1
                while new_max_streams < self._max_streams:
                    await self._max_streams_semaphore.acquire()
                    self._max_streams -= 1

    async def _response_closed(self, stream_id: int) -> None:
        async with self._state_lock:
            if stream_id in self._events:
                await self._max_streams_semaphore.release()
                del self._events[stream_id]
            if self._connection_terminated and not self._events:
                await self.aclose()

            elif self._state == HTTPConnectionState.ACTIVE and not self._events:
                self._state = HTTPConnectionState.IDLE
                if self._keepalive_expiry is not None:
                    now = time.monotonic()
                    self._expire_at = now + self._keepalive_expiry
                if self._used_all_stream_ids:  # pragma: no cover
                    await self.aclose()

    async def aclose(self) -> None:
        # Note that this method unilaterally closes the connection, and does
        # not have any kind of locking in place around it.
        self._h2_state.close_connection()
        self._state = HTTPConnectionState.CLOSED
        await self._network_stream.aclose()

    # Wrappers around network read/write operations...

    async def _read_incoming_data(self, request: Request) -> list[h2.events.Event]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        if self._read_exception is not None:
            raise self._read_exception  # pragma: no cover

        try:
            data = await self._network_stream.read(self.READ_NUM_BYTES, timeout)
            if data == b"":
                raise RemoteProtocolError("Server disconnected")
        except Exception as exc:
            # If we get a network error we should:
            #
            # 1. Save the exception and just raise it immediately on any future reads.
            #    (For example, this means that a single read timeout or disconnect will
            #    immediately close all pending streams. Without requiring multiple
            #    sequential timeouts.)
            # 2. Mark the connection as errored, so that we don't accept any other
            #    incoming requests.
            self._read_exception = exc
            self._connection_error = True
            raise exc

        events: list[h2.events.Event] = self._h2_state.receive_data(data)

        return events

    async def _write_outgoing_data(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        async with self._write_lock:
            data_to_send = self._h2_state.data_to_send()

            if self._write_exception is not None:
                raise self._write_exception  # pragma: no cover

            try:
                await self._network_stream.write(data_to_send, timeout)
            except Exception as exc:  # pragma: no cover
                # If we get a network error we should:
                #
                # 1. Save the exception and just raise it immediately on any future write.
                #    (For example, this means that a single write timeout or disconnect will
                #    immediately close all pending streams. Without requiring multiple
                #    sequential timeouts.)
                # 2. Mark the connection as errored, so that we don't accept any other
                #    incoming requests.
                self._write_exception = exc
                self._connection_error = True
                raise exc

    # Flow control...

    async def _wait_for_outgoing_flow(self, request: Request, stream_id: int) -> int:
        """
        Returns the maximum allowable outgoing flow for a given stream.

        If the allowable flow is zero, then waits on the network until
        WindowUpdated frames have increased the flow rate.
        https://tools.ietf.org/html/rfc7540#section-6.9
        """
        local_flow: int = self._h2_state.local_flow_control_window(stream_id)
        max_frame_size: int = self._h2_state.max_outbound_frame_size
        flow = min(local_flow, max_frame_size)
        while flow <= 0:
            await self._receive_events(request)
            local_flow = self._h2_state.local_flow_control_window(stream_id)
            max_frame_size = self._h2_state.max_outbound_frame_size
            flow = min(local_flow, max_frame_size)
        return flow

    # Interface for connection pooling...

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    def is_connected(self) -> bool:
        return not self.is_closed()

    def is_available(self) -> bool:
        return (
            self._state != HTTPConnectionState.CLOSED
            and not self._connection_error
            and not self._used_all_stream_ids
            and not (self._h2_state.state_machine.state == h2.connection.ConnectionState.CLOSED)
        )

    def has_expired(self) -> bool:
        now = time.monotonic()
        # Read `_expire_at` once into a local: on free-threaded builds another
        # thread may reset it to `None` between the check and the comparison.
        expire_at = self._expire_at
        return expire_at is not None and now > expire_at

    def is_idle(self) -> bool:
        return self._state == HTTPConnectionState.IDLE

    def can_multiplex(self) -> bool:
        return True

    def is_closed(self) -> bool:
        return self._state == HTTPConnectionState.CLOSED

    def info(self) -> str:
        origin = str(self._origin)
        return f"{origin!r}, HTTP/2, {self._state.name}, Request Count: {self._request_count}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        origin = str(self._origin)
        return f"<{class_name} [{origin!r}, {self._state.name}, Request Count: {self._request_count}]>"

    # These context managers are not used in the standard flow, but are
    # useful for testing or working with connection instances directly.

    async def __aenter__(self) -> AsyncHTTP2Connection:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        await self.aclose()


class HTTP2ConnectionByteStream:
    def __init__(self, connection: AsyncHTTP2Connection, request: Request, stream_id: int) -> None:
        self._connection = connection
        self._request = request
        self._stream_id = stream_id
        self._closed = False

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        kwargs = {"request": self._request, "stream_id": self._stream_id}
        try:
            async with Trace("receive_response_body", logger, self._request, kwargs):
                async with safe_async_iterate(
                    self._connection._receive_response_body(request=self._request, stream_id=self._stream_id)
                ) as iterator:
                    async for chunk in iterator:
                        yield chunk
        except BaseException as exc:
            # If we get an exception while streaming the response,
            # we want to close the response (and possibly the connection)
            # before raising that exception.
            with AsyncShieldCancellation():
                await self.aclose()
            raise exc

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            kwargs = {"stream_id": self._stream_id}
            async with Trace("response_closed", logger, self._request, kwargs):
                await self._connection._response_closed(stream_id=self._stream_id)
<|endoftext|>from __future__ import annotations

import ssl
import sys
import types
import typing
from collections.abc import AsyncGenerator

from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Proxy, Request, Response
from .._synchronization import AsyncEvent, AsyncShieldCancellation, AsyncThreadLock
from .._utils import safe_async_iterate
from .connection import AsyncHTTPConnection
from .interfaces import AsyncConnectionInterface, AsyncRequestInterface


class AsyncPoolRequest:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.connection: AsyncConnectionInterface | None = None
        self._connection_acquired = AsyncEvent()

    def assign_to_connection(self, connection: AsyncConnectionInterface | None) -> None:
        self.connection = connection
        self._connection_acquired.set()

    def clear_connection(self) -> None:
        self.connection = None
        self._connection_acquired = AsyncEvent()

    async def wait_for_connection(self, timeout: float | None = None) -> AsyncConnectionInterface:
        if self.connection is None:
            await self._connection_acquired.wait(timeout=timeout)
        assert self.connection is not None
        return self.connection

    def is_queued(self) -> bool:
        return self.connection is None


class AsyncConnectionPool(AsyncRequestInterface):
    """
    A connection pool for making HTTP requests.
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext | None = None,
        proxy: Proxy | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: AsyncNetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish a
                connection.
            local_address: Local address to connect from. Can also be used to connect
                using a particular address family. Using `local_address="0.0.0.0"`
                will connect using an `AF_INET` address (IPv4), while using
                `local_address="::"` will connect using an `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
            socket_options: Socket options that have to be included
             in the TCP socket when the connection was established.
        """
        self._ssl_context = ssl_context
        self._proxy = proxy
        self._max_connections = sys.maxsize if max_connections is None else max_connections
        self._max_keepalive_connections = (
            sys.maxsize if max_keepalive_connections is None else max_keepalive_connections
        )
        self._max_keepalive_connections = min(self._max_connections, self._max_keepalive_connections)

        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._retries = retries
        self._local_address = local_address
        self._uds = uds

        self._network_backend = AutoBackend() if network_backend is None else network_backend
        self._socket_options = socket_options

        # The mutable state on a connection pool is the queue of incoming requests,
        # and the set of connections that are servicing those requests.
        self._connections: list[AsyncConnectionInterface] = []
        self._requests: list[AsyncPoolRequest] = []

        # We only mutate the state of the connection pool within an 'optional_thread_lock'
        # context. This holds a threading lock unless we're running in async mode,
        # in which case it is a no-op.
        self._optional_thread_lock = AsyncThreadLock()

    def create_connection(self, origin: Origin) -> AsyncConnectionInterface:
        if self._proxy is not None:
            if self._proxy.url.scheme in (b"socks5", b"socks5h"):
                from .socks_proxy import AsyncSocks5Connection

                return AsyncSocks5Connection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_auth=self._proxy.auth,
                    remote_origin=origin,
                    ssl_context=self._ssl_context,
                    keepalive_expiry=self._keepalive_expiry,
                    http1=self._http1,
                    http2=self._http2,
                    network_backend=self._network_backend,
                )
            elif origin.scheme == b"http":
                from .http_proxy import AsyncForwardHTTPConnection

                return AsyncForwardHTTPConnection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_headers=self._proxy.headers,
                    proxy_ssl_context=self._proxy.ssl_context,
                    remote_origin=origin,
                    keepalive_expiry=self._keepalive_expiry,
                    network_backend=self._network_backend,
                )
            from .http_proxy import AsyncTunnelHTTPConnection

            return AsyncTunnelHTTPConnection(
                proxy_origin=self._proxy.url.origin,
                proxy_headers=self._proxy.headers,
                proxy_ssl_context=self._proxy.ssl_context,
                remote_origin=origin,
                ssl_context=self._ssl_context,
                keepalive_expiry=self._keepalive_expiry,
                http1=self._http1,
                http2=self._http2,
                network_backend=self._network_backend,
            )

        return AsyncHTTPConnection(
            origin=origin,
            ssl_context=self._ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            retries=self._retries,
            local_address=self._local_address,
            uds=self._uds,
            network_backend=self._network_backend,
            socket_options=self._socket_options,
        )

    @property
    def connections(self) -> list[AsyncConnectionInterface]:
        """
        Return a list of the connections currently in the pool.

        For example:

        ```python
        >>> pool.connections
        [
            <AsyncHTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 6]>,
            <AsyncHTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 9]> ,
            <AsyncHTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>,
        ]
        ```
        """
        return list(self._connections)

    async def handle_async_request(self, request: Request) -> Response:
        """
        Send an HTTP request, and return an HTTP response.

        This is the core implementation that is called into by `.request()` or `.stream()`.
        """
        scheme = request.url.scheme.decode()
        if scheme == "":
            raise UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol.")
        if scheme not in ("http", "https", "ws", "wss"):
            raise UnsupportedProtocol(f"Request URL has an unsupported protocol '{scheme}://'.")

        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("pool", None)

        with self._optional_thread_lock:
            # Add the incoming request to our request queue.
            pool_request = AsyncPoolRequest(request)
            self._requests.append(pool_request)

        try:
            while True:
                with self._optional_thread_lock:
                    # Assign incoming requests to available connections,
                    # closing or creating new connections as required.
                    closing = self._assign_requests_to_connections()
                await self._close_connections(closing)

                # Wait until this request has an assigned connection.
                connection = await pool_request.wait_for_connection(timeout=timeout)

                try:
                    # Send the request on the assigned connection.
                    response = await connection.handle_async_request(pool_request.request)
                except ConnectionNotAvailable:
                    # In some cases a connection may initially be available to
                    # handle a request, but then become unavailable.
                    #
                    # In this case we clear the connection and try again.
                    pool_request.clear_connection()
                else:
                    break  # pragma: no cover

        except BaseException as exc:
            with self._optional_thread_lock:
                # For any exception or cancellation we remove the request from
                # the queue, and then re-assign requests to connections.
                self._requests.remove(pool_request)
                closing = self._assign_requests_to_connections()

            await self._close_connections(closing)
            raise exc from None

        # Return the response. Note that in this case we still have to manage
        # the point at which the response is closed.
        assert isinstance(response.stream, typing.AsyncIterable)
        return Response(
            status=response.status,
            headers=response.headers,
            content=PoolByteStream(stream=response.stream, pool_request=pool_request, pool=self),
            extensions=response.extensions,
        )

    def _assign_requests_to_connections(self) -> list[AsyncConnectionInterface]:
        """
        Manage the state of the connection pool, assigning incoming
        requests to connections as available.

        Called whenever a new request is added or removed from the pool.

        Any closing connections are returned, allowing the I/O for closing
        those connections to be handled separately.
        """
        closing_connections: list[AsyncConnectionInterface] = []
        retained_connections: list[AsyncConnectionInterface] = []

        # Connections currently referenced by an in-flight request, including
        # connections that are in the process of being established and idle
        # connections reserved by an assigned-but-not-yet-sent request.
        request_connections = {r.connection for r in self._requests}

        # First we handle cleaning up any connections that are closed
        # or have expired their keep-alive, in a single pass. Reserved
        # connections skip the expiry check: they were checked when assigned,
        # and `has_expired()` on an idle connection probes the socket.
        for connection in self._connections:
            reserved = connection in request_connections
            if connection.is_closed():
                continue
            elif not (connection.is_connected() or reserved):
                # Garbage: a NEW-state connection whose request was cancelled
                # before the TCP handshake completed.  Drop it without closing
                # (there is no socket to close yet).
                continue
            elif not reserved and connection.has_expired():
                closing_connections.append(connection)
            else:
                retained_connections.append(connection)

        # Then we close any surplus idle connections, to enforce the
        # max_keepalive_connections setting. Reserved connections are not
        # surplus: a request is about to be sent on them.
        idle_surplus = (
            sum(connection.is_idle() and connection not in request_connections for connection in retained_connections)
            - self._max_keepalive_connections
        )
        if idle_surplus > 0:
            kept: list[AsyncConnectionInterface] = []
            for connection in retained_connections:
                if idle_surplus > 0 and connection.is_idle() and connection not in request_connections:
                    closing_connections.append(connection)
                    idle_surplus -= 1
                else:
                    kept.append(connection)
            retained_connections = kept

        self._connections = retained_connections

        # Snapshot the set of reusable connections once, rather than rebuilding
        # it per queued request — this is what brings the loop from O(N*M) to
        # O(N+M) in the common case.
        #
        # An idle connection already assigned to an in-flight request is
        # reserved: it stays IDLE until the winning task sends on it, so
        # without this exclusion the next pass would assign it again and the
        # loser would churn through `ConnectionNotAvailable`. Multiplexing
        # connections are exempt: they can take further requests while idle.
        available_connections = [
            connection
            for connection in self._connections
            if connection.is_available()
            and not (connection.is_idle() and connection in request_connections and not connection.can_multiplex())
        ]
        new_connection_budget = self._max_connections - len(self._connections)

        # Assign queued requests to connections. Once no connection is
        # available and no new connection may be created, no queued request
        # can be assigned, so the scan stops early: this keeps a pass on a
        # saturated pool O(connections) rather than O(in-flight requests).
        for pool_request in self._requests:
            if not available_connections and new_connection_budget <= 0:
                break
            if not pool_request.is_queued():
                continue
            origin = pool_request.request.url.origin

            # There are three cases for how we may be able to handle the request:
            #
            # 1. There is an existing connection that can handle the request.
            # 2. We can create a new connection to handle the request.
            # 3. We can close an idle connection and then create a new connection
            #    to handle the request.
            for idx, connection in enumerate(available_connections):
                if connection.can_handle_request(origin):
                    pool_request.assign_to_connection(connection)
                    if connection.is_idle() and not connection.can_multiplex():
                        # An idle HTTP/1.1 connection can only take this
                        # single request until it is released.
                        del available_connections[idx]
                    break
            else:
                if new_connection_budget > 0:
                    connection = self.create_connection(origin)
                    self._connections.append(connection)
                    pool_request.assign_to_connection(connection)
                    new_connection_budget -= 1
                    continue
                for idx, connection in enumerate(available_connections):
                    if connection.is_idle():
                        del available_connections[idx]
                        self._connections.remove(connection)
                        closing_connections.append(connection)
                        connection = self.create_connection(origin)
                        self._connections.append(connection)
                        pool_request.assign_to_connection(connection)
                        break

        return closing_connections

    async def _close_connections(self, closing: list[AsyncConnectionInterface]) -> None:
        # Close connections which have been removed from the pool.
        with AsyncShieldCancellation():
            for connection in closing:
                await connection.aclose()

    async def aclose(self) -> None:
        # Explicitly close the connection pool.
        # Clears all existing requests and connections.
        with self._optional_thread_lock:
            closing_connections = list(self._connections)
            self._connections = []
        await self._close_connections(closing_connections)

    async def __aenter__(self) -> AsyncConnectionPool:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        with self._optional_thread_lock:
            request_is_queued = [request.is_queued() for request in self._requests]
            connection_is_idle = [connection.is_idle() for connection in self._connections]

            num_active_requests = request_is_queued.count(False)
            num_queued_requests = request_is_queued.count(True)
            num_active_connections = connection_is_idle.count(False)
            num_idle_connections = connection_is_idle.count(True)

        requests_info = f"Requests: {num_active_requests} active, {num_queued_requests} queued"
        connection_info = f"Connections: {num_active_connections} active, {num_idle_connections} idle"

        return f"<{class_name} [{requests_info} | {connection_info}]>"


class PoolByteStream:
    def __init__(
        self,
        stream: typing.AsyncIterable[bytes],
        pool_request: AsyncPoolRequest,
        pool: AsyncConnectionPool,
    ) -> None:
        self._stream = stream
        self._pool_request = pool_request
        self._pool = pool
        self._closed = False

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        async with safe_async_iterate(self._stream) as iterator:
            async for chunk in iterator:
                yield chunk

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            with AsyncShieldCancellation():
                if hasattr(self._stream, "aclose"):
                    await self._stream.aclose()

            with self._pool._optional_thread_lock:
                self._pool._requests.remove(self._pool_request)
                closing = self._pool._assign_requests_to_connections()

            await self._pool._close_connections(closing)
<|endoftext|>from __future__ import annotations

import itertools
import logging
import ssl
import types
import typing

from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream
from .._exceptions import ConnectError, ConnectTimeout
from .._models import Origin, Request, Response
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface

RETRIES_BACKOFF_FACTOR = 0.5  # 0s, 0.5s, 1s, 2s, 4s, etc.


logger = logging.getLogger("httpcore2.connection")


def exponential_backoff(factor: float) -> typing.Iterator[float]:
    """
    Generate a geometric sequence that has a ratio of 2 and starts with 0.

    For example:
    - `factor = 2`: `0, 2, 4, 8, 16, 32, 64, ...`
    - `factor = 3`: `0, 3, 6, 12, 24, 48, 96, ...`
    """
    yield 0
    for n in itertools.count():
        yield factor * 2**n


class AsyncHTTPConnection(AsyncConnectionInterface):
    def __init__(
        self,
        origin: Origin,
        ssl_context: ssl.SSLContext | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: AsyncNetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        self._origin = origin
        self._ssl_context = ssl_context
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._retries = retries
        self._local_address = local_address
        self._uds = uds

        self._network_backend: AsyncNetworkBackend = AutoBackend() if network_backend is None else network_backend
        self._connection: AsyncConnectionInterface | None = None
        self._connect_failed: bool = False
        self._request_lock = AsyncLock()
        self._socket_options = socket_options

    async def handle_async_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        try:
            async with self._request_lock:
                if self._connection is None:
                    stream = await self._connect(request)

                    ssl_object = stream.get_extra_info("ssl_object")
                    http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"
                    if http2_negotiated or (self._http2 and not self._http1):
                        from .http2 import AsyncHTTP2Connection

                        self._connection = AsyncHTTP2Connection(
                            origin=self._origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                    else:
                        self._connection = AsyncHTTP11Connection(
                            origin=self._origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
        except BaseException as exc:
            self._connect_failed = True
            raise exc

        return await self._connection.handle_async_request(request)

    async def _connect(self, request: Request) -> AsyncNetworkStream:
        timeouts = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname", None)
        timeout = timeouts.get("connect", None)

        retries_left = self._retries
        delays = exponential_backoff(factor=RETRIES_BACKOFF_FACTOR)

        while True:
            try:
                if self._uds is None:
                    kwargs = {
                        "host": self._origin.host.decode("ascii"),
                        "port": self._origin.port,
                        "local_address": self._local_address,
                        "timeout": timeout,
                        "socket_options": self._socket_options,
                    }
                    async with Trace("connect_tcp", logger, request, kwargs) as trace:
                        stream = await self._network_backend.connect_tcp(**kwargs)
                        trace.return_value = stream
                else:
                    kwargs = {
                        "path": self._uds,
                        "timeout": timeout,
                        "socket_options": self._socket_options,
                    }
                    async with Trace("connect_unix_socket", logger, request, kwargs) as trace:
                        stream = await self._network_backend.connect_unix_socket(**kwargs)
                        trace.return_value = stream

                if self._origin.scheme in (b"https", b"wss"):
                    ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                    alpn_protocols = (["h2", "http/1.1"] if self._http1 else ["h2"]) if self._http2 else ["http/1.1"]
                    ssl_context.set_alpn_protocols(alpn_protocols)

                    kwargs = {
                        "ssl_context": ssl_context,
                        "server_hostname": sni_hostname or self._origin.host.decode("ascii"),
                        "timeout": timeout,
                    }
                    async with Trace("start_tls", logger, request, kwargs) as trace:
                        stream = await stream.start_tls(**kwargs)
                        trace.return_value = stream
                return stream
            except (ConnectError, ConnectTimeout):
                if retries_left <= 0:
                    raise
                retries_left -= 1
                delay = next(delays)
                async with Trace("retry", logger, request, kwargs) as trace:
                    await self._network_backend.sleep(delay)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    async def aclose(self) -> None:
        if self._connection is not None:
            async with Trace("close", logger, None, {}):
                await self._connection.aclose()

    def is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected()

    def is_available(self) -> bool:
        if self._connection is None:
            # If HTTP/2 support is enabled, and the resulting connection could
            # end up as HTTP/2 then we should indicate the connection as being
            # available to service multiple requests.
            return self._http2 and (self._origin.scheme == b"https" or not self._http1) and not self._connect_failed
        return self._connection.is_available()

    def has_expired(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.is_idle()

    def can_multiplex(self) -> bool:
        return self._connection is not None and self._connection.can_multiplex()

    def is_closed(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.is_closed()

    def info(self) -> str:
        if self._connection is None:
            return "CONNECTION FAILED" if self._connect_failed else "CONNECTING"
        return self._connection.info()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"

    # These context managers are not used in the standard flow, but are
    # useful for testing or working with connection instances directly.

    async def __aenter__(self) -> AsyncHTTPConnection:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        await self.aclose()
<|endoftext|>from __future__ import annotations

import enum
import logging
import ssl
import time
import types
import typing
from collections.abc import Generator

import h11

from .._backends.base import NetworkStream
from .._exceptions import (
    ConnectionNotAvailable,
    LocalProtocolError,
    RemoteProtocolError,
    WriteError,
    map_exceptions,
)
from .._models import Origin, Request, Response
from .._synchronization import Lock, ShieldCancellation
from .._trace import Trace
from .._utils import safe_iterate
from .interfaces import ConnectionInterface

logger = logging.getLogger("httpcore2.http11")


# A subset of `h11.Event` types supported by `_send_event`
H11SendEvent = h11.Request | h11.Data | h11.EndOfMessage


class HTTPConnectionState(enum.IntEnum):
    NEW = 0
    ACTIVE = 1
    IDLE = 2
    CLOSED = 3


class HTTP11Connection(ConnectionInterface):
    READ_NUM_BYTES = 64 * 1024
    MAX_INCOMPLETE_EVENT_SIZE = 100 * 1024

    def __init__(
        self,
        origin: Origin,
        stream: NetworkStream,
        keepalive_expiry: float | None = None,
    ) -> None:
        self._origin = origin
        self._network_stream = stream
        self._keepalive_expiry: float | None = keepalive_expiry
        self._expire_at: float | None = None
        self._state = HTTPConnectionState.NEW
        self._state_lock = Lock()
        self._request_count = 0
        self._h11_state = h11.Connection(
            our_role=h11.CLIENT,
            max_incomplete_event_size=self.MAX_INCOMPLETE_EVENT_SIZE,
        )

    def handle_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        with self._state_lock:
            if self._state in (HTTPConnectionState.NEW, HTTPConnectionState.IDLE):
                self._request_count += 1
                self._state = HTTPConnectionState.ACTIVE
                self._expire_at = None
            else:
                raise ConnectionNotAvailable()

        try:
            kwargs = {"request": request}
            try:
                with Trace("send_request_headers", logger, request, kwargs) as trace:
                    self._send_request_headers(**kwargs)
                with Trace("send_request_body", logger, request, kwargs) as trace:
                    self._send_request_body(**kwargs)
            except WriteError:
                # If we get a write error while we're writing the request,
                # then we suppress this error and move on to attempting to
                # read the response. Servers can sometimes close the request
                # preemptively and then respond with a well formed HTTP
                # error response.
                pass

            with Trace("receive_response_headers", logger, request, kwargs) as trace:
                (
                    http_version,
                    status,
                    reason_phrase,
                    headers,
                    trailing_data,
                ) = self._receive_response_headers(**kwargs)
                trace.return_value = (
                    http_version,
                    status,
                    reason_phrase,
                    headers,
                )

            network_stream = self._network_stream

            # CONNECT or Upgrade request
            if (status == 101) or ((request.method == b"CONNECT") and (200 <= status < 300)):
                network_stream = HTTP11UpgradeStream(network_stream, trailing_data)

            return Response(
                status=status,
                headers=headers,
                content=HTTP11ConnectionByteStream(self, request),
                extensions={
                    "http_version": http_version,
                    "reason_phrase": reason_phrase,
                    "network_stream": network_stream,
                },
            )
        except BaseException as exc:
            with ShieldCancellation():
                with Trace("response_closed", logger, request) as trace:
                    self._response_closed()
            raise exc

    # Sending the request...

    def _send_request_headers(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        with map_exceptions({h11.LocalProtocolError: LocalProtocolError}):
            event = h11.Request(
                method=request.method,
                target=request.url.target,
                headers=request.headers,
            )
        self._send_event(event, timeout=timeout)

    def _send_request_body(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        assert isinstance(request.stream, typing.Iterable)
        with safe_iterate(request.stream) as iterator:
            for chunk in iterator:
                event = h11.Data(data=chunk)
                self._send_event(event, timeout=timeout)

        self._send_event(h11.EndOfMessage(), timeout=timeout)

    def _send_event(self, event: h11.Event, timeout: float | None = None) -> None:
        bytes_to_send = self._h11_state.send(event)
        if bytes_to_send is not None:
            self._network_stream.write(bytes_to_send, timeout=timeout)

    # Receiving the response...

    def _receive_response_headers(
        self, request: Request
    ) -> tuple[bytes, int, bytes, list[tuple[bytes, bytes]], bytes]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        while True:
            event = self._receive_event(timeout=timeout)
            if isinstance(event, h11.Response):
                break
            if isinstance(event, h11.InformationalResponse) and event.status_code == 101:
                break

        http_version = b"HTTP/" + event.http_version

        # h11 version 0.11+ supports a `raw_items` interface to get the
        # raw header casing, rather than the enforced lowercase headers.
        headers = event.headers.raw_items()

        trailing_data, _ = self._h11_state.trailing_data

        return http_version, event.status_code, event.reason, headers, trailing_data

    def _receive_response_body(self, request: Request) -> Generator[bytes]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        while True:
            event = self._receive_event(timeout=timeout)
            if isinstance(event, h11.Data):
                yield bytes(event.data)
            elif isinstance(event, (h11.EndOfMessage, h11.PAUSED)):
                break

    def _receive_event(self, timeout: float | None = None) -> h11.Event | type[h11.PAUSED]:
        while True:
            with map_exceptions({h11.RemoteProtocolError: RemoteProtocolError}):
                event = self._h11_state.next_event()

            if event is h11.NEED_DATA:
                data = self._network_stream.read(self.READ_NUM_BYTES, timeout=timeout)

                # If we feed this case through h11 we'll raise an exception like:
                #
                #     httpcore2.RemoteProtocolError: can't handle event type
                #     ConnectionClosed when role=SERVER and state=SEND_RESPONSE
                #
                # Which is accurate, but not very informative from an end-user
                # perspective. Instead we handle this case distinctly and treat
                # it as a ConnectError.
                if data == b"" and self._h11_state.their_state == h11.SEND_RESPONSE:
                    msg = "Server disconnected without sending a response."
                    raise RemoteProtocolError(msg)

                self._h11_state.receive_data(data)
            else:
                # mypy fails to narrow the type in the above if statement above
                return event  # type: ignore[return-value]

    def _response_closed(self) -> None:
        with self._state_lock:
            if self._h11_state.our_state is h11.DONE and self._h11_state.their_state is h11.DONE:
                self._state = HTTPConnectionState.IDLE
                self._h11_state.start_next_cycle()
                if self._keepalive_expiry is not None:
                    now = time.monotonic()
                    self._expire_at = now + self._keepalive_expiry
            else:
                self.close()

    # Once the connection is no longer required...

    def close(self) -> None:
        # Note that this method unilaterally closes the connection, and does
        # not have any kind of locking in place around it.
        self._state = HTTPConnectionState.CLOSED
        self._network_stream.close()

    # The ConnectionInterface methods provide information about the state of
    # the connection, allowing for a connection pooling implementation to
    # determine when to reuse and when to close the connection...

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    def is_connected(self) -> bool:
        return not self.is_closed()

    def is_available(self) -> bool:
        # Note that HTTP/1.1 connections in the "NEW" state are not treated as
        # being "available". The control flow which created the connection will
        # be able to send an outgoing request, but the connection will not be
        # acquired from the connection pool for any other request.
        return self._state == HTTPConnectionState.IDLE

    def has_expired(self) -> bool:
        now = time.monotonic()
        # Read `_expire_at` once into a local: on free-threaded builds another
        # thread may reset it to `None` between the check and the comparison.
        expire_at = self._expire_at
        keepalive_expired = expire_at is not None and now > expire_at

        # If the HTTP connection is idle but the socket is readable, then the
        # only valid state is that the socket is about to return b"", indicating
        # a server-initiated disconnect.
        server_disconnected = self._state == HTTPConnectionState.IDLE and self._network_stream.get_extra_info(
            "is_readable"
        )

        return keepalive_expired or server_disconnected

    def is_idle(self) -> bool:
        return self._state == HTTPConnectionState.IDLE

    def is_closed(self) -> bool:
        return self._state == HTTPConnectionState.CLOSED

    def info(self) -> str:
        origin = str(self._origin)
        return f"{origin!r}, HTTP/1.1, {self._state.name}, Request Count: {self._request_count}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        origin = str(self._origin)
        return f"<{class_name} [{origin!r}, {self._state.name}, Request Count: {self._request_count}]>"

    # These context managers are not used in the standard flow, but are
    # useful for testing or working with connection instances directly.

    def __enter__(self) -> HTTP11Connection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self.close()


class HTTP11ConnectionByteStream:
    def __init__(self, connection: HTTP11Connection, request: Request) -> None:
        self._connection = connection
        self._request = request
        self._closed = False

    def __iter__(self) -> Generator[bytes]:
        kwargs = {"request": self._request}
        try:
            with Trace("receive_response_body", logger, self._request, kwargs):
                with safe_iterate(self._connection._receive_response_body(**kwargs)) as iterator:
                    for chunk in iterator:
                        yield chunk
        except BaseException as exc:
            # If we get an exception while streaming the response,
            # we want to close the response (and possibly the connection)
            # before raising that exception.
            with ShieldCancellation():
                self.close()
            raise exc

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with Trace("response_closed", logger, self._request):
                self._connection._response_closed()


class HTTP11UpgradeStream(NetworkStream):
    def __init__(self, stream: NetworkStream, leading_data: bytes) -> None:
        self._stream = stream
        self._leading_data = leading_data

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._leading_data:
            buffer = self._leading_data[:max_bytes]
            self._leading_data = self._leading_data[max_bytes:]
            return buffer
        else:
            return self._stream.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(buffer, timeout)

    def close(self) -> None:
        self._stream.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        return self._stream.start_tls(ssl_context, server_hostname, timeout)

    def get_extra_info(self, info: str) -> typing.Any:
        return self._stream.get_extra_info(info)
<|endoftext|>from __future__ import annotations

import contextlib
import typing
from collections.abc import Generator

from .._models import (
    URL,
    Extensions,
    HeaderTypes,
    Origin,
    Request,
    Response,
    enforce_bytes,
    enforce_headers,
    enforce_url,
    include_request_headers,
)


class RequestInterface:
    def request(
        self,
        method: bytes | str,
        url: URL | bytes | str,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.Iterator[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> Response:
        # Strict type checking on our parameters.
        method = enforce_bytes(method, name="method")
        url = enforce_url(url, name="url")
        headers = enforce_headers(headers, name="headers")

        # Include Host header, and optionally Content-Length or Transfer-Encoding.
        headers = include_request_headers(headers, url=url, content=content)

        request = Request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        )
        response = self.handle_request(request)
        try:
            response.read()
        finally:
            response.close()
        return response

    @contextlib.contextmanager
    def stream(
        self,
        method: bytes | str,
        url: URL | bytes | str,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.Iterator[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> Generator[Response]:
        # Strict type checking on our parameters.
        method = enforce_bytes(method, name="method")
        url = enforce_url(url, name="url")
        headers = enforce_headers(headers, name="headers")

        # Include Host header, and optionally Content-Length or Transfer-Encoding.
        headers = include_request_headers(headers, url=url, content=content)

        request = Request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            extensions=extensions,
        )
        response = self.handle_request(request)
        try:
            yield response
        finally:
            response.close()

    def handle_request(self, request: Request) -> Response:
        raise NotImplementedError()  # pragma: no cover


class ConnectionInterface(RequestInterface):
    def close(self) -> None:
        raise NotImplementedError()  # pragma: no cover

    def info(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    def can_handle_request(self, origin: Origin) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def is_connected(self) -> bool:
        """
        Return `True` if the connection is open (the underlying socket has been
        established).  A connection in the NEW state (just created but not yet
        connected) returns `False`.

        Note: for some implementations `is_connected() != not is_closed()`.
        The default implementation returns `not self.is_closed()`, which is
        correct for connections that are never in the NEW (pre-TCP) state.
        """
        return not self.is_closed()  # pragma: no cover

    def is_available(self) -> bool:
        """
        Return `True` if the connection is currently able to accept an
        outgoing request.

        An HTTP/1.1 connection will only be available if it is currently idle.

        An HTTP/2 connection will be available so long as the stream ID space is
        not yet exhausted, and the connection is not in an error state.

        While the connection is being established we may not yet know if it is going
        to result in an HTTP/1.1 or HTTP/2 connection. The connection should be
        treated as being available, but might ultimately raise `NewConnectionRequired`
        required exceptions if multiple requests are attempted over a connection
        that ends up being established as HTTP/1.1.
        """
        raise NotImplementedError()  # pragma: no cover

    def has_expired(self) -> bool:
        """
        Return `True` if the connection is in a state where it should be closed.

        This either means that the connection is idle and it has passed the
        expiry time on its keep-alive, or that server has sent an EOF.
        """
        raise NotImplementedError()  # pragma: no cover

    def is_idle(self) -> bool:
        """
        Return `True` if the connection is currently idle.
        """
        raise NotImplementedError()  # pragma: no cover

    def can_multiplex(self) -> bool:
        """
        Return `True` if the connection can serve multiple requests
        concurrently, such as an established HTTP/2 connection.

        The default covers HTTP/1.1-style implementations, which serve a
        single request at a time.
        """
        return False

    def is_closed(self) -> bool:
        """
        Return `True` if the connection has been closed.

        Used when a response is closed to determine if the connection may be
        returned to the connection pool or not.
        """
        raise NotImplementedError()  # pragma: no cover
<|endoftext|>from .connection import HTTPConnection
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .http_proxy import HTTPProxy
from .interfaces import ConnectionInterface

try:
    from .http2 import HTTP2Connection
except ImportError:  # pragma: no cover

    class HTTP2Connection:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # type: ignore
            raise RuntimeError(
                "Attempted to use http2 support, but the `h2` package is not "
                "installed. Use 'pip install httpcore[http2]'."
            )


try:
    from .socks_proxy import SOCKSProxy
except ImportError:  # pragma: no cover

    class SOCKSProxy:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # type: ignore
            raise RuntimeError(
                "Attempted to use SOCKS support, but the `socksio` package is not "
                "installed. Use 'pip install httpcore[socks]'."
            )


__all__ = [
    "HTTPConnection",
    "ConnectionPool",
    "HTTPProxy",
    "HTTP11Connection",
    "HTTP2Connection",
    "ConnectionInterface",
    "SOCKSProxy",
]
<|endoftext|>from __future__ import annotations

import base64
import logging
import ssl
import typing

from .._backends.base import SOCKET_OPTION, NetworkBackend
from .._exceptions import ProxyError
from .._models import (
    URL,
    Origin,
    Request,
    Response,
    enforce_bytes,
    enforce_headers,
    enforce_url,
)
from .._ssl import default_ssl_context
from .._synchronization import Lock
from .._trace import Trace
from .connection import HTTPConnection
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .interfaces import ConnectionInterface

ByteOrStr = bytes | str
HeadersAsSequence = typing.Sequence[tuple[ByteOrStr, ByteOrStr]]
HeadersAsMapping = typing.Mapping[ByteOrStr, ByteOrStr]


logger = logging.getLogger("httpcore2.proxy")


def merge_headers(
    default_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
    override_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
) -> list[tuple[bytes, bytes]]:
    """
    Append default_headers and override_headers, de-duplicating if a key exists
    in both cases.
    """
    default_headers = [] if default_headers is None else list(default_headers)
    override_headers = [] if override_headers is None else list(override_headers)
    has_override = set(key.lower() for key, _value in override_headers)
    default_headers = [(key, value) for key, value in default_headers if key.lower() not in has_override]
    return default_headers + override_headers


class HTTPProxy(ConnectionPool):  # pragma: no cover
    """
    A connection pool that sends requests via an HTTP proxy.
    """

    def __init__(
        self,
        proxy_url: URL | bytes | str,
        proxy_auth: tuple[bytes | str, bytes | str] | None = None,
        proxy_headers: HeadersAsMapping | HeadersAsSequence | None = None,
        ssl_context: ssl.SSLContext | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: NetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            proxy_url: The URL to use when connecting to the proxy server.
                For example `"http://127.0.0.1:8080/"`.
            proxy_auth: Any proxy authentication as a two-tuple of
                (username, password). May be either bytes or ascii-only str.
            proxy_headers: Any HTTP headers to use for the proxy requests.
                For example `{"Proxy-Authorization": "Basic <username>:<password>"}`.
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            proxy_ssl_context: The same as `ssl_context`, but for a proxy server rather than a remote origin.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish
                a connection.
            local_address: Local address to connect from. Can also be used to
                connect using a particular address family. Using
                `local_address="0.0.0.0"` will connect using an `AF_INET` address
                (IPv4), while using `local_address="::"` will connect using an
                `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
        """
        super().__init__(
            ssl_context=ssl_context,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            http1=http1,
            http2=http2,
            network_backend=network_backend,
            retries=retries,
            local_address=local_address,
            uds=uds,
            socket_options=socket_options,
        )

        self._proxy_url = enforce_url(proxy_url, name="proxy_url")
        if self._proxy_url.scheme == b"http" and proxy_ssl_context is not None:  # pragma: no cover
            raise RuntimeError("The `proxy_ssl_context` argument is not allowed for the http scheme")

        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        if proxy_auth is not None:
            username = enforce_bytes(proxy_auth[0], name="proxy_auth")
            password = enforce_bytes(proxy_auth[1], name="proxy_auth")
            userpass = username + b":" + password
            authorization = b"Basic " + base64.b64encode(userpass)
            self._proxy_headers = [(b"Proxy-Authorization", authorization)] + self._proxy_headers

    def create_connection(self, origin: Origin) -> ConnectionInterface:
        if origin.scheme == b"http":
            return ForwardHTTPConnection(
                proxy_origin=self._proxy_url.origin,
                proxy_headers=self._proxy_headers,
                remote_origin=origin,
                keepalive_expiry=self._keepalive_expiry,
                network_backend=self._network_backend,
                proxy_ssl_context=self._proxy_ssl_context,
            )
        return TunnelHTTPConnection(
            proxy_origin=self._proxy_url.origin,
            proxy_headers=self._proxy_headers,
            remote_origin=origin,
            ssl_context=self._ssl_context,
            proxy_ssl_context=self._proxy_ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            network_backend=self._network_backend,
        )


class ForwardHTTPConnection(ConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        proxy_headers: HeadersAsMapping | HeadersAsSequence | None = None,
        keepalive_expiry: float | None = None,
        network_backend: NetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._connection = HTTPConnection(
            origin=proxy_origin,
            keepalive_expiry=keepalive_expiry,
            network_backend=network_backend,
            socket_options=socket_options,
            ssl_context=proxy_ssl_context,
        )
        self._proxy_origin = proxy_origin
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        self._remote_origin = remote_origin

    def handle_request(self, request: Request) -> Response:
        headers = merge_headers(self._proxy_headers, request.headers)
        url = URL(
            scheme=self._proxy_origin.scheme,
            host=self._proxy_origin.host,
            port=self._proxy_origin.port,
            target=bytes(request.url),
        )
        proxy_request = Request(
            method=request.method,
            url=url,
            headers=headers,
            content=request.stream,
            extensions={key: value for key, value in request.extensions.items() if key != "sni_hostname"},
        )
        return self._connection.handle_request(proxy_request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    def close(self) -> None:
        self._connection.close()

    def info(self) -> str:
        return self._connection.info()

    def is_connected(self) -> bool:
        return self._connection.is_connected()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"


class TunnelHTTPConnection(ConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        ssl_context: ssl.SSLContext | None = None,
        proxy_ssl_context: ssl.SSLContext | None = None,
        proxy_headers: typing.Sequence[tuple[bytes, bytes]] | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        network_backend: NetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        self._connection: ConnectionInterface = HTTPConnection(
            origin=proxy_origin,
            keepalive_expiry=keepalive_expiry,
            network_backend=network_backend,
            socket_options=socket_options,
            ssl_context=proxy_ssl_context,
        )
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name="proxy_headers")
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._connect_lock = Lock()
        self._connected = False

    def handle_request(self, request: Request) -> Response:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("connect", None)

        with self._connect_lock:
            if not self._connected:
                target = b"%b:%d" % (self._remote_origin.host, self._remote_origin.port)

                connect_url = URL(
                    scheme=self._proxy_origin.scheme,
                    host=self._proxy_origin.host,
                    port=self._proxy_origin.port,
                    target=target,
                )
                connect_headers = merge_headers([(b"Host", target), (b"Accept", b"*/*")], self._proxy_headers)
                connect_request = Request(
                    method=b"CONNECT",
                    url=connect_url,
                    headers=connect_headers,
                    extensions={key: value for key, value in request.extensions.items() if key != "sni_hostname"},
                )
                connect_response = self._connection.handle_request(connect_request)

                if connect_response.status < 200 or connect_response.status > 299:
                    reason_bytes = connect_response.extensions.get("reason_phrase", b"")
                    reason_str = reason_bytes.decode("ascii", errors="ignore")
                    msg = f"{connect_response.status} {reason_str}"
                    self._connection.close()
                    raise ProxyError(msg)

                stream = connect_response.extensions["network_stream"]

                # Upgrade the stream to SSL
                ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                alpn_protocols = (["h2", "http/1.1"] if self._http1 else ["h2"]) if self._http2 else ["http/1.1"]
                ssl_context.set_alpn_protocols(alpn_protocols)

                server_hostname = request.extensions.get("sni_hostname") or self._remote_origin.host.decode("ascii")
                kwargs = {
                    "ssl_context": ssl_context,
                    "server_hostname": server_hostname,
                    "timeout": timeout,
                }
                with Trace("start_tls", logger, request, kwargs) as trace:
                    stream = stream.start_tls(**kwargs)
                    trace.return_value = stream

                # Determine if we should be using HTTP/1.1 or HTTP/2
                ssl_object = stream.get_extra_info("ssl_object")
                http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"

                # Create the HTTP/1.1 or HTTP/2 connection
                if http2_negotiated or (self._http2 and not self._http1):
                    from .http2 import HTTP2Connection

                    self._connection = HTTP2Connection(
                        origin=self._remote_origin,
                        stream=stream,
                        keepalive_expiry=self._keepalive_expiry,
                    )
                else:
                    self._connection = HTTP11Connection(
                        origin=self._remote_origin,
                        stream=stream,
                        keepalive_expiry=self._keepalive_expiry,
                    )

                self._connected = True
        return self._connection.handle_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    def close(self) -> None:
        self._connection.close()

    def info(self) -> str:
        return self._connection.info()

    def is_connected(self) -> bool:
        return self._connection.is_connected()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"
<|endoftext|>from __future__ import annotations

import logging
import ssl

import socksio

from .._backends.sync import SyncBackend
from .._backends.base import NetworkBackend, NetworkStream
from .._exceptions import ConnectionNotAvailable, ProxyError
from .._models import URL, Origin, Request, Response, enforce_bytes, enforce_url
from .._ssl import default_ssl_context
from .._synchronization import Lock
from .._trace import Trace
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .interfaces import ConnectionInterface

logger = logging.getLogger("httpcore2.socks")


AUTH_METHODS = {
    b"\x00": "NO AUTHENTICATION REQUIRED",
    b"\x01": "GSSAPI",
    b"\x02": "USERNAME/PASSWORD",
    b"\xff": "NO ACCEPTABLE METHODS",
}

REPLY_CODES = {
    b"\x00": "Succeeded",
    b"\x01": "General SOCKS server failure",
    b"\x02": "Connection not allowed by ruleset",
    b"\x03": "Network unreachable",
    b"\x04": "Host unreachable",
    b"\x05": "Connection refused",
    b"\x06": "TTL expired",
    b"\x07": "Command not supported",
    b"\x08": "Address type not supported",
}


def _init_socks5_connection(
    stream: NetworkStream,
    *,
    host: bytes,
    port: int,
    auth: tuple[bytes, bytes] | None = None,
    timeouts: dict[str, float | None] | None = None,
) -> None:
    timeouts = timeouts or {}
    write_timeout = timeouts.get("write", None)
    read_timeout = timeouts.get("read", None)
    conn = socksio.socks5.SOCKS5Connection()

    # Auth method request
    auth_method = (
        socksio.socks5.SOCKS5AuthMethod.NO_AUTH_REQUIRED
        if auth is None
        else socksio.socks5.SOCKS5AuthMethod.USERNAME_PASSWORD
    )
    conn.send(socksio.socks5.SOCKS5AuthMethodsRequest([auth_method]))
    outgoing_bytes = conn.data_to_send()
    stream.write(outgoing_bytes, timeout=write_timeout)

    # Auth method response
    incoming_bytes = stream.read(max_bytes=4096, timeout=read_timeout)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socksio.socks5.SOCKS5AuthReply)
    if response.method != auth_method:
        requested = AUTH_METHODS.get(auth_method, "UNKNOWN")
        responded = AUTH_METHODS.get(response.method, "UNKNOWN")
        raise ProxyError(f"Requested {requested} from proxy server, but got {responded}.")

    if response.method == socksio.socks5.SOCKS5AuthMethod.USERNAME_PASSWORD:
        # Username/password request
        assert auth is not None
        username, password = auth
        conn.send(socksio.socks5.SOCKS5UsernamePasswordRequest(username, password))
        outgoing_bytes = conn.data_to_send()
        stream.write(outgoing_bytes, timeout=write_timeout)

        # Username/password response
        incoming_bytes = stream.read(max_bytes=4096, timeout=read_timeout)
        response = conn.receive_data(incoming_bytes)
        assert isinstance(response, socksio.socks5.SOCKS5UsernamePasswordReply)
        if not response.success:
            raise ProxyError("Invalid username/password")

    # Connect request
    conn.send(socksio.socks5.SOCKS5CommandRequest.from_address(socksio.socks5.SOCKS5Command.CONNECT, (host, port)))
    outgoing_bytes = conn.data_to_send()
    stream.write(outgoing_bytes, timeout=write_timeout)

    # Connect response
    incoming_bytes = stream.read(max_bytes=4096, timeout=read_timeout)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socksio.socks5.SOCKS5Reply)
    if response.reply_code != socksio.socks5.SOCKS5ReplyCode.SUCCEEDED:
        reply_code = REPLY_CODES.get(response.reply_code, "UNKOWN")
        raise ProxyError(f"Proxy Server could not connect: {reply_code}.")


class SOCKSProxy(ConnectionPool):  # pragma: no cover
    """
    A connection pool that sends requests via an HTTP proxy.
    """

    def __init__(
        self,
        proxy_url: URL | bytes | str,
        proxy_auth: tuple[bytes | str, bytes | str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        network_backend: NetworkBackend | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            proxy_url: The URL to use when connecting to the proxy server.
                For example `"http://127.0.0.1:8080/"`.
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish
                a connection.
            local_address: Local address to connect from. Can also be used to
                connect using a particular address family. Using
                `local_address="0.0.0.0"` will connect using an `AF_INET` address
                (IPv4), while using `local_address="::"` will connect using an
                `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
        """
        super().__init__(
            ssl_context=ssl_context,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
            http1=http1,
            http2=http2,
            network_backend=network_backend,
            retries=retries,
        )
        self._ssl_context = ssl_context
        self._proxy_url = enforce_url(proxy_url, name="proxy_url")
        if proxy_auth is not None:
            username, password = proxy_auth
            username_bytes = enforce_bytes(username, name="proxy_auth")
            password_bytes = enforce_bytes(password, name="proxy_auth")
            self._proxy_auth: tuple[bytes, bytes] | None = (
                username_bytes,
                password_bytes,
            )
        else:
            self._proxy_auth = None

    def create_connection(self, origin: Origin) -> ConnectionInterface:
        return Socks5Connection(
            proxy_origin=self._proxy_url.origin,
            remote_origin=origin,
            proxy_auth=self._proxy_auth,
            ssl_context=self._ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            network_backend=self._network_backend,
        )


class Socks5Connection(ConnectionInterface):
    def __init__(
        self,
        proxy_origin: Origin,
        remote_origin: Origin,
        proxy_auth: tuple[bytes, bytes] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        network_backend: NetworkBackend | None = None,
    ) -> None:
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._proxy_auth = proxy_auth
        self._ssl_context = ssl_context
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2

        self._network_backend: NetworkBackend = SyncBackend() if network_backend is None else network_backend
        self._connect_lock = Lock()
        self._connection: ConnectionInterface | None = None
        self._connect_failed = False

    def handle_request(self, request: Request) -> Response:
        timeouts = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname", None)
        timeout = timeouts.get("connect", None)

        with self._connect_lock:
            if self._connection is None:
                try:
                    # Connect to the proxy
                    kwargs = {
                        "host": self._proxy_origin.host.decode("ascii"),
                        "port": self._proxy_origin.port,
                        "timeout": timeout,
                    }
                    with Trace("connect_tcp", logger, request, kwargs) as trace:
                        stream = self._network_backend.connect_tcp(**kwargs)
                        trace.return_value = stream

                    # Connect to the remote host using socks5
                    kwargs = {
                        "stream": stream,
                        "host": self._remote_origin.host.decode("ascii"),
                        "port": self._remote_origin.port,
                        "auth": self._proxy_auth,
                        "timeouts": timeouts,
                    }
                    with Trace("setup_socks5_connection", logger, request, kwargs) as trace:
                        _init_socks5_connection(**kwargs)
                        trace.return_value = stream

                    # Upgrade the stream to SSL
                    if self._remote_origin.scheme in (b"https", b"wss"):
                        ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                        alpn_protocols = (
                            (["h2", "http/1.1"] if self._http1 else ["h2"]) if self._http2 else ["http/1.1"]
                        )
                        ssl_context.set_alpn_protocols(alpn_protocols)

                        kwargs = {
                            "ssl_context": ssl_context,
                            "server_hostname": sni_hostname or self._remote_origin.host.decode("ascii"),
                            "timeout": timeout,
                        }
                        with Trace("start_tls", logger, request, kwargs) as trace:
                            stream = stream.start_tls(**kwargs)
                            trace.return_value = stream

                    # Determine if we should be using HTTP/1.1 or HTTP/2
                    ssl_object = stream.get_extra_info("ssl_object")
                    http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"

                    # Create the HTTP/1.1 or HTTP/2 connection
                    if http2_negotiated or (self._http2 and not self._http1):  # pragma: no cover
                        from .http2 import HTTP2Connection

                        self._connection = HTTP2Connection(
                            origin=self._remote_origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                    else:
                        self._connection = HTTP11Connection(
                            origin=self._remote_origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                except Exception as exc:
                    self._connect_failed = True
                    raise exc
            elif not self._connection.is_available():  # pragma: no cover
                raise ConnectionNotAvailable()

        return self._connection.handle_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected()

    def is_available(self) -> bool:
        if self._connection is None:  # pragma: no cover
            # If HTTP/2 support is enabled, and the resulting connection could
            # end up as HTTP/2 then we should indicate the connection as being
            # available to service multiple requests.
            return (
                self._http2 and (self._remote_origin.scheme == b"https" or not self._http1) and not self._connect_failed
            )
        return self._connection.is_available()

    def has_expired(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        if self._connection is None:  # pragma: no cover
            return self._connect_failed
        return self._connection.is_closed()

    def info(self) -> str:
        if self._connection is None:  # pragma: no cover
            return "CONNECTION FAILED" if self._connect_failed else "CONNECTING"
        return self._connection.info()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.info()}]>"
<|endoftext|>from __future__ import annotations

import enum
import logging
import time
import types
import typing
from collections.abc import Generator

import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings

from .._backends.base import NetworkStream
from .._exceptions import ConnectionNotAvailable, LocalProtocolError, RemoteProtocolError
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .._utils import safe_iterate
from .interfaces import ConnectionInterface

logger = logging.getLogger("httpcore2.http2")


def has_body_headers(request: Request) -> bool:
    return any(k.lower() == b"content-length" or k.lower() == b"transfer-encoding" for k, _v in request.headers)


class HTTPConnectionState(enum.IntEnum):
    ACTIVE = 1
    IDLE = 2
    CLOSED = 3


class HTTP2Connection(ConnectionInterface):
    READ_NUM_BYTES = 64 * 1024
    CONFIG = h2.config.H2Configuration(validate_inbound_headers=False)

    def __init__(
        self,
        origin: Origin,
        stream: NetworkStream,
        keepalive_expiry: float | None = None,
    ):
        self._origin = origin
        self._network_stream = stream
        self._keepalive_expiry: float | None = keepalive_expiry
        self._h2_state = h2.connection.H2Connection(config=self.CONFIG)
        self._state = HTTPConnectionState.IDLE
        self._expire_at: float | None = None
        self._request_count = 0
        self._init_lock = Lock()
        self._state_lock = Lock()
        self._read_lock = Lock()
        self._write_lock = Lock()
        self._sent_connection_init = False
        self._used_all_stream_ids = False
        self._connection_error = False

        # Mapping from stream ID to response stream events.
        self._events: dict[
            int,
            list[h2.events.ResponseReceived | h2.events.DataReceived | h2.events.StreamEnded | h2.events.StreamReset,],
        ] = {}

        # Connection terminated events are stored as state since
        # we need to handle them for all streams.
        self._connection_terminated: h2.events.ConnectionTerminated | None = None

        self._read_exception: Exception | None = None
        self._write_exception: Exception | None = None

    def handle_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            # This cannot occur in normal operation, since the connection pool
            # will only send requests on connections that handle them.
            # It's in place simply for resilience as a guard against incorrect
            # usage, for anyone working directly with httpcore connections.
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        with self._state_lock:
            if self._state in (HTTPConnectionState.ACTIVE, HTTPConnectionState.IDLE):
                self._request_count += 1
                self._expire_at = None
                self._state = HTTPConnectionState.ACTIVE
            else:
                raise ConnectionNotAvailable()

        with self._init_lock:
            if not self._sent_connection_init:
                try:
                    sci_kwargs = {"request": request}
                    with Trace("send_connection_init", logger, request, sci_kwargs):
                        self._send_connection_init(**sci_kwargs)
                except BaseException as exc:
                    with ShieldCancellation():
                        self.close()
                    raise exc

                self._sent_connection_init = True

                # Initially start with just 1 until the remote server provides
                # its max_concurrent_streams value
                self._max_streams = 1

                local_settings_max_streams = self._h2_state.local_settings.max_concurrent_streams
                self._max_streams_semaphore = Semaphore(local_settings_max_streams)

                for _ in range(local_settings_max_streams - self._max_streams):
                    self._max_streams_semaphore.acquire()

        self._max_streams_semaphore.acquire()

        try:
            stream_id = self._h2_state.get_next_available_stream_id()
            self._events[stream_id] = []
        except h2.exceptions.NoAvailableStreamIDError:  # pragma: no cover
            self._used_all_stream_ids = True
            self._request_count -= 1
            self._max_streams_semaphore.release()
            raise ConnectionNotAvailable()

        try:
            kwargs = {"request": request, "stream_id": stream_id}
            with Trace("send_request_headers", logger, request, kwargs):
                self._send_request_headers(request=request, stream_id=stream_id)
            with Trace("send_request_body", logger, request, kwargs):
                self._send_request_body(request=request, stream_id=stream_id)
            with Trace("receive_response_headers", logger, request, kwargs) as trace:
                status, headers = self._receive_response(request=request, stream_id=stream_id)
                trace.return_value = (status, headers)

            return Response(
                status=status,
                headers=headers,
                content=HTTP2ConnectionByteStream(self, request, stream_id=stream_id),
                extensions={
                    "http_version": b"HTTP/2",
                    "network_stream": self._network_stream,
                    "stream_id": stream_id,
                },
            )
        except BaseException as exc:  # noqa: PIE786
            with ShieldCancellation():
                kwargs = {"stream_id": stream_id}
                with Trace("response_closed", logger, request, kwargs):
                    self._response_closed(stream_id=stream_id)

            if isinstance(exc, h2.exceptions.ProtocolError):
                # One case where h2 can raise a protocol error is when a
                # closed frame has been seen by the state machine.
                #
                # This happens when one stream is reading, and encounters
                # a GOAWAY event. Other flows of control may then raise
                # a protocol error at any point they interact with the 'h2_state'.
                #
                # In this case we'll have stored the event, and should raise
                # it as a RemoteProtocolError.
                if self._connection_terminated:  # pragma: no cover
                    raise RemoteProtocolError(self._connection_terminated)
                # If h2 raises a protocol error in some other state then we
                # must somehow have made a protocol violation.
                raise LocalProtocolError(exc)  # pragma: no cover

            raise exc

    def _send_connection_init(self, request: Request) -> None:
        """
        The HTTP/2 connection requires some initial setup before we can start
        using individual request/response streams on it.
        """
        # Need to set these manually here instead of manipulating via
        # __setitem__() otherwise the H2Connection will emit SettingsUpdate
        # frames in addition to sending the undesired defaults.
        self._h2_state.local_settings = h2.settings.Settings(
            client=True,
            initial_values={
                # Disable PUSH_PROMISE frames from the server since we don't do anything
                # with them for now.  Maybe when we support caching?
                h2.settings.SettingCodes.ENABLE_PUSH: 0,
                # These two are taken from h2 for safe defaults
                h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 100,
                h2.settings.SettingCodes.MAX_HEADER_LIST_SIZE: 65536,
            },
        )

        # Some websites (*cough* Yahoo *cough*) balk at this setting being
        # present in the initial handshake since it's not defined in the original
        # RFC despite the RFC mandating ignoring settings you don't know about.
        del self._h2_state.local_settings[h2.settings.SettingCodes.ENABLE_CONNECT_PROTOCOL]

        self._h2_state.initiate_connection()
        self._h2_state.increment_flow_control_window(2**24)
        self._write_outgoing_data(request)

    # Sending the request...

    def _send_request_headers(self, request: Request, stream_id: int) -> None:
        """
        Send the request headers to a given stream ID.
        """
        end_stream = not has_body_headers(request)

        # In HTTP/2 the ':authority' pseudo-header is used instead of 'Host'.
        # In order to gracefully handle HTTP/1.1 and HTTP/2 we always require
        # HTTP/1.1 style headers, and map them appropriately if we end up on
        # an HTTP/2 connection.
        authority = [v for k, v in request.headers if k.lower() == b"host"][0]

        headers = [
            (b":method", request.method),
            (b":authority", authority),
            (b":scheme", request.url.scheme),
            (b":path", request.url.target),
        ] + [
            (k.lower(), v)
            for k, v in request.headers
            if k.lower()
            not in (
                b"host",
                b"transfer-encoding",
            )
        ]

        self._h2_state.send_headers(stream_id, headers, end_stream=end_stream)
        self._h2_state.increment_flow_control_window(2**24, stream_id=stream_id)
        self._write_outgoing_data(request)

    def _send_request_body(self, request: Request, stream_id: int) -> None:
        """
        Iterate over the request body sending it to a given stream ID.
        """
        if not has_body_headers(request):
            return

        assert isinstance(request.stream, typing.Iterable)
        with safe_iterate(request.stream) as iterator:
            for chunk in iterator:
                self._send_stream_data(request, stream_id, chunk)

        self._send_end_stream(request, stream_id)

    def _send_stream_data(self, request: Request, stream_id: int, data: bytes) -> None:
        """
        Send a single chunk of data in one or more data frames.
        """
        position = 0
        while position < len(data):
            max_flow = self._wait_for_outgoing_flow(request, stream_id)
            chunk = data[position : position + max_flow]
            position += len(chunk)
            self._h2_state.send_data(stream_id, chunk)
            self._write_outgoing_data(request)

    def _send_end_stream(self, request: Request, stream_id: int) -> None:
        """
        Send an empty data frame on on a given stream ID with the END_STREAM flag set.
        """
        self._h2_state.end_stream(stream_id)
        self._write_outgoing_data(request)

    # Receiving the response...

    def _receive_response(self, request: Request, stream_id: int) -> tuple[int, list[tuple[bytes, bytes]]]:
        """
        Return the response status code and headers for a given stream ID.
        """
        while True:
            event = self._receive_stream_event(request, stream_id)
            if isinstance(event, h2.events.ResponseReceived):
                break

        status_code = 200
        headers: list[tuple[bytes, bytes]] = []
        assert event.headers is not None
        for k, v in event.headers:
            if k == b":status":
                status_code = int(v.decode("ascii", errors="ignore"))
            elif not k.startswith(b":"):
                headers.append((k, v))

        return (status_code, headers)

    def _receive_response_body(self, request: Request, stream_id: int) -> Generator[bytes]:
        """
        Iterator that returns the bytes of the response body for a given stream ID.
        """
        while True:
            event = self._receive_stream_event(request, stream_id)
            if isinstance(event, h2.events.DataReceived):
                assert event.flow_controlled_length is not None
                assert event.data is not None
                amount = event.flow_controlled_length
                self._h2_state.acknowledge_received_data(amount, stream_id)
                self._write_outgoing_data(request)
                yield event.data
            elif isinstance(event, h2.events.StreamEnded):
                break

    def _receive_stream_event(
        self, request: Request, stream_id: int
    ) -> h2.events.ResponseReceived | h2.events.DataReceived | h2.events.StreamEnded:
        """
        Return the next available event for a given stream ID.

        Will read more data from the network if required.
        """
        while not self._events.get(stream_id):
            self._receive_events(request, stream_id)
        event = self._events[stream_id].pop(0)
        if isinstance(event, h2.events.StreamReset):
            raise RemoteProtocolError(event)
        return event

    def _receive_events(self, request: Request, stream_id: int | None = None) -> None:
        """
        Read some data from the network until we see one or more events
        for a given stream ID.
        """
        with self._read_lock:
            if self._connection_terminated is not None:
                last_stream_id = self._connection_terminated.last_stream_id
                if stream_id and last_stream_id and stream_id > last_stream_id:
                    self._request_count -= 1
                    raise ConnectionNotAvailable()
                raise RemoteProtocolError(self._connection_terminated)

            # This conditional is a bit icky. We don't want to block reading if we've
            # actually got an event to return for a given stream. We need to do that
            # check *within* the atomic read lock. Though it also need to be optional,
            # because when we call it from `_wait_for_outgoing_flow` we *do* want to
            # block until we've available flow control, event when we have events
            # pending for the stream ID we're attempting to send on.
            if stream_id is None or not self._events.get(stream_id):
                events = self._read_incoming_data(request)
                for event in events:
                    if isinstance(event, h2.events.RemoteSettingsChanged):
                        with Trace("receive_remote_settings", logger, request) as trace:
                            self._receive_remote_settings_change(event)
                            trace.return_value = event

                    elif isinstance(
                        event,
                        (
                            h2.events.ResponseReceived,
                            h2.events.DataReceived,
                            h2.events.StreamEnded,
                            h2.events.StreamReset,
                        ),
                    ):
                        if event.stream_id in self._events:
                            self._events[event.stream_id].append(event)

                    elif isinstance(event, h2.events.ConnectionTerminated):
                        self._connection_terminated = event

        self._write_outgoing_data(request)

    def _receive_remote_settings_change(self, event: h2.events.RemoteSettingsChanged) -> None:
        max_concurrent_streams = event.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
        if max_concurrent_streams:
            new_max_streams = min(
                max_concurrent_streams.new_value,
                self._h2_state.local_settings.max_concurrent_streams,
            )
            if new_max_streams and new_max_streams != self._max_streams:
                while new_max_streams > self._max_streams:
                    self._max_streams_semaphore.release()
                    self._max_streams += 1
                while new_max_streams < self._max_streams:
                    self._max_streams_semaphore.acquire()
                    self._max_streams -= 1

    def _response_closed(self, stream_id: int) -> None:
        with self._state_lock:
            if stream_id in self._events:
                self._max_streams_semaphore.release()
                del self._events[stream_id]
            if self._connection_terminated and not self._events:
                self.close()

            elif self._state == HTTPConnectionState.ACTIVE and not self._events:
                self._state = HTTPConnectionState.IDLE
                if self._keepalive_expiry is not None:
                    now = time.monotonic()
                    self._expire_at = now + self._keepalive_expiry
                if self._used_all_stream_ids:  # pragma: no cover
                    self.close()

    def close(self) -> None:
        # Note that this method unilaterally closes the connection, and does
        # not have any kind of locking in place around it.
        self._h2_state.close_connection()
        self._state = HTTPConnectionState.CLOSED
        self._network_stream.close()

    # Wrappers around network read/write operations...

    def _read_incoming_data(self, request: Request) -> list[h2.events.Event]:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("read", None)

        if self._read_exception is not None:
            raise self._read_exception  # pragma: no cover

        try:
            data = self._network_stream.read(self.READ_NUM_BYTES, timeout)
            if data == b"":
                raise RemoteProtocolError("Server disconnected")
        except Exception as exc:
            # If we get a network error we should:
            #
            # 1. Save the exception and just raise it immediately on any future reads.
            #    (For example, this means that a single read timeout or disconnect will
            #    immediately close all pending streams. Without requiring multiple
            #    sequential timeouts.)
            # 2. Mark the connection as errored, so that we don't accept any other
            #    incoming requests.
            self._read_exception = exc
            self._connection_error = True
            raise exc

        events: list[h2.events.Event] = self._h2_state.receive_data(data)

        return events

    def _write_outgoing_data(self, request: Request) -> None:
        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("write", None)

        with self._write_lock:
            data_to_send = self._h2_state.data_to_send()

            if self._write_exception is not None:
                raise self._write_exception  # pragma: no cover

            try:
                self._network_stream.write(data_to_send, timeout)
            except Exception as exc:  # pragma: no cover
                # If we get a network error we should:
                #
                # 1. Save the exception and just raise it immediately on any future write.
                #    (For example, this means that a single write timeout or disconnect will
                #    immediately close all pending streams. Without requiring multiple
                #    sequential timeouts.)
                # 2. Mark the connection as errored, so that we don't accept any other
                #    incoming requests.
                self._write_exception = exc
                self._connection_error = True
                raise exc

    # Flow control...

    def _wait_for_outgoing_flow(self, request: Request, stream_id: int) -> int:
        """
        Returns the maximum allowable outgoing flow for a given stream.

        If the allowable flow is zero, then waits on the network until
        WindowUpdated frames have increased the flow rate.
        https://tools.ietf.org/html/rfc7540#section-6.9
        """
        local_flow: int = self._h2_state.local_flow_control_window(stream_id)
        max_frame_size: int = self._h2_state.max_outbound_frame_size
        flow = min(local_flow, max_frame_size)
        while flow <= 0:
            self._receive_events(request)
            local_flow = self._h2_state.local_flow_control_window(stream_id)
            max_frame_size = self._h2_state.max_outbound_frame_size
            flow = min(local_flow, max_frame_size)
        return flow

    # Interface for connection pooling...

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    def is_connected(self) -> bool:
        return not self.is_closed()

    def is_available(self) -> bool:
        return (
            self._state != HTTPConnectionState.CLOSED
            and not self._connection_error
            and not self._used_all_stream_ids
            and not (self._h2_state.state_machine.state == h2.connection.ConnectionState.CLOSED)
        )

    def has_expired(self) -> bool:
        now = time.monotonic()
        # Read `_expire_at` once into a local: on free-threaded builds another
        # thread may reset it to `None` between the check and the comparison.
        expire_at = self._expire_at
        return expire_at is not None and now > expire_at

    def is_idle(self) -> bool:
        return self._state == HTTPConnectionState.IDLE

    def can_multiplex(self) -> bool:
        return True

    def is_closed(self) -> bool:
        return self._state == HTTPConnectionState.CLOSED

    def info(self) -> str:
        origin = str(self._origin)
        return f"{origin!r}, HTTP/2, {self._state.name}, Request Count: {self._request_count}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        origin = str(self._origin)
        return f"<{class_name} [{origin!r}, {self._state.name}, Request Count: {self._request_count}]>"

    # These context managers are not used in the standard flow, but are
    # useful for testing or working with connection instances directly.

    def __enter__(self) -> HTTP2Connection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self.close()


class HTTP2ConnectionByteStream:
    def __init__(self, connection: HTTP2Connection, request: Request, stream_id: int) -> None:
        self._connection = connection
        self._request = request
        self._stream_id = stream_id
        self._closed = False

    def __iter__(self) -> Generator[bytes]:
        kwargs = {"request": self._request, "stream_id": self._stream_id}
        try:
            with Trace("receive_response_body", logger, self._request, kwargs):
                with safe_iterate(
                    self._connection._receive_response_body(request=self._request, stream_id=self._stream_id)
                ) as iterator:
                    for chunk in iterator:
                        yield chunk
        except BaseException as exc:
            # If we get an exception while streaming the response,
            # we want to close the response (and possibly the connection)
            # before raising that exception.
            with ShieldCancellation():
                self.close()
            raise exc

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            kwargs = {"stream_id": self._stream_id}
            with Trace("response_closed", logger, self._request, kwargs):
                self._connection._response_closed(stream_id=self._stream_id)
<|endoftext|>from __future__ import annotations

import ssl
import sys
import types
import typing
from collections.abc import Generator

from .._backends.sync import SyncBackend
from .._backends.base import SOCKET_OPTION, NetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Proxy, Request, Response
from .._synchronization import Event, ShieldCancellation, ThreadLock
from .._utils import safe_iterate
from .connection import HTTPConnection
from .interfaces import ConnectionInterface, RequestInterface


class PoolRequest:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.connection: ConnectionInterface | None = None
        self._connection_acquired = Event()

    def assign_to_connection(self, connection: ConnectionInterface | None) -> None:
        self.connection = connection
        self._connection_acquired.set()

    def clear_connection(self) -> None:
        self.connection = None
        self._connection_acquired = Event()

    def wait_for_connection(self, timeout: float | None = None) -> ConnectionInterface:
        if self.connection is None:
            self._connection_acquired.wait(timeout=timeout)
        assert self.connection is not None
        return self.connection

    def is_queued(self) -> bool:
        return self.connection is None


class ConnectionPool(RequestInterface):
    """
    A connection pool for making HTTP requests.
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext | None = None,
        proxy: Proxy | None = None,
        max_connections: int | None = 10,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        local_address: str | None = None,
        uds: str | None = None,
        network_backend: NetworkBackend | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore2.default_ssl_context()`
                will be used.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish a
                connection.
            local_address: Local address to connect from. Can also be used to connect
                using a particular address family. Using `local_address="0.0.0.0"`
                will connect using an `AF_INET` address (IPv4), while using
                `local_address="::"` will connect using an `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
            socket_options: Socket options that have to be included
             in the TCP socket when the connection was established.
        """
        self._ssl_context = ssl_context
        self._proxy = proxy
        self._max_connections = sys.maxsize if max_connections is None else max_connections
        self._max_keepalive_connections = (
            sys.maxsize if max_keepalive_connections is None else max_keepalive_connections
        )
        self._max_keepalive_connections = min(self._max_connections, self._max_keepalive_connections)

        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._retries = retries
        self._local_address = local_address
        self._uds = uds

        self._network_backend = SyncBackend() if network_backend is None else network_backend
        self._socket_options = socket_options

        # The mutable state on a connection pool is the queue of incoming requests,
        # and the set of connections that are servicing those requests.
        self._connections: list[ConnectionInterface] = []
        self._requests: list[PoolRequest] = []

        # We only mutate the state of the connection pool within an 'optional_thread_lock'
        # context. This holds a threading lock unless we're running in async mode,
        # in which case it is a no-op.
        self._optional_thread_lock = ThreadLock()

    def create_connection(self, origin: Origin) -> ConnectionInterface:
        if self._proxy is not None:
            if self._proxy.url.scheme in (b"socks5", b"socks5h"):
                from .socks_proxy import Socks5Connection

                return Socks5Connection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_auth=self._proxy.auth,
                    remote_origin=origin,
                    ssl_context=self._ssl_context,
                    keepalive_expiry=self._keepalive_expiry,
                    http1=self._http1,
                    http2=self._http2,
                    network_backend=self._network_backend,
                )
            elif origin.scheme == b"http":
                from .http_proxy import ForwardHTTPConnection

                return ForwardHTTPConnection(
                    proxy_origin=self._proxy.url.origin,
                    proxy_headers=self._proxy.headers,
                    proxy_ssl_context=self._proxy.ssl_context,
                    remote_origin=origin,
                    keepalive_expiry=self._keepalive_expiry,
                    network_backend=self._network_backend,
                )
            from .http_proxy import TunnelHTTPConnection

            return TunnelHTTPConnection(
                proxy_origin=self._proxy.url.origin,
                proxy_headers=self._proxy.headers,
                proxy_ssl_context=self._proxy.ssl_context,
                remote_origin=origin,
                ssl_context=self._ssl_context,
                keepalive_expiry=self._keepalive_expiry,
                http1=self._http1,
                http2=self._http2,
                network_backend=self._network_backend,
            )

        return HTTPConnection(
            origin=origin,
            ssl_context=self._ssl_context,
            keepalive_expiry=self._keepalive_expiry,
            http1=self._http1,
            http2=self._http2,
            retries=self._retries,
            local_address=self._local_address,
            uds=self._uds,
            network_backend=self._network_backend,
            socket_options=self._socket_options,
        )

    @property
    def connections(self) -> list[ConnectionInterface]:
        """
        Return a list of the connections currently in the pool.

        For example:

        ```python
        >>> pool.connections
        [
            <HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 6]>,
            <HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 9]> ,
            <HTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>,
        ]
        ```
        """
        return list(self._connections)

    def handle_request(self, request: Request) -> Response:
        """
        Send an HTTP request, and return an HTTP response.

        This is the core implementation that is called into by `.request()` or `.stream()`.
        """
        scheme = request.url.scheme.decode()
        if scheme == "":
            raise UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol.")
        if scheme not in ("http", "https", "ws", "wss"):
            raise UnsupportedProtocol(f"Request URL has an unsupported protocol '{scheme}://'.")

        timeouts = request.extensions.get("timeout", {})
        timeout = timeouts.get("pool", None)

        with self._optional_thread_lock:
            # Add the incoming request to our request queue.
            pool_request = PoolRequest(request)
            self._requests.append(pool_request)

        try:
            while True:
                with self._optional_thread_lock:
                    # Assign incoming requests to available connections,
                    # closing or creating new connections as required.
                    closing = self._assign_requests_to_connections()
                self._close_connections(closing)

                # Wait until this request has an assigned connection.
                connection = pool_request.wait_for_connection(timeout=timeout)

                try:
                    # Send the request on the assigned connection.
                    response = connection.handle_request(pool_request.request)
                except ConnectionNotAvailable:
                    # In some cases a connection may initially be available to
                    # handle a request, but then become unavailable.
                    #
                    # In this case we clear the connection and try again.
                    pool_request.clear_connection()
                else:
                    break  # pragma: no cover

        except BaseException as exc:
            with self._optional_thread_lock:
                # For any exception or cancellation we remove the request from
                # the queue, and then re-assign requests to connections.
                self._requests.remove(pool_request)
