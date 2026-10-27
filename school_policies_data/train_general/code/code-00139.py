                closing = self._assign_requests_to_connections()

            self._close_connections(closing)
            raise exc from None

        # Return the response. Note that in this case we still have to manage
        # the point at which the response is closed.
        assert isinstance(response.stream, typing.Iterable)
        return Response(
            status=response.status,
            headers=response.headers,
            content=PoolByteStream(stream=response.stream, pool_request=pool_request, pool=self),
            extensions=response.extensions,
        )

    def _assign_requests_to_connections(self) -> list[ConnectionInterface]:
        """
        Manage the state of the connection pool, assigning incoming
        requests to connections as available.

        Called whenever a new request is added or removed from the pool.

        Any closing connections are returned, allowing the I/O for closing
        those connections to be handled separately.
        """
        closing_connections: list[ConnectionInterface] = []
        retained_connections: list[ConnectionInterface] = []

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
            kept: list[ConnectionInterface] = []
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

    def _close_connections(self, closing: list[ConnectionInterface]) -> None:
        # Close connections which have been removed from the pool.
        with ShieldCancellation():
            for connection in closing:
                connection.close()

    def close(self) -> None:
        # Explicitly close the connection pool.
        # Clears all existing requests and connections.
        with self._optional_thread_lock:
            closing_connections = list(self._connections)
            self._connections = []
        self._close_connections(closing_connections)

    def __enter__(self) -> ConnectionPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self.close()

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
        stream: typing.Iterable[bytes],
        pool_request: PoolRequest,
        pool: ConnectionPool,
    ) -> None:
        self._stream = stream
        self._pool_request = pool_request
        self._pool = pool
        self._closed = False

    def __iter__(self) -> Generator[bytes]:
        with safe_iterate(self._stream) as iterator:
            for chunk in iterator:
                yield chunk

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with ShieldCancellation():
                if hasattr(self._stream, "close"):
                    self._stream.close()

            with self._pool._optional_thread_lock:
                self._pool._requests.remove(self._pool_request)
                closing = self._pool._assign_requests_to_connections()

            self._pool._close_connections(closing)
<|endoftext|>from __future__ import annotations

import itertools
import logging
import ssl
import types
import typing

from .._backends.sync import SyncBackend
from .._backends.base import SOCKET_OPTION, NetworkBackend, NetworkStream
from .._exceptions import ConnectError, ConnectTimeout
from .._models import Origin, Request, Response
from .._ssl import default_ssl_context
from .._synchronization import Lock
from .._trace import Trace
from .http11 import HTTP11Connection
from .interfaces import ConnectionInterface

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


class HTTPConnection(ConnectionInterface):
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
        network_backend: NetworkBackend | None = None,
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

        self._network_backend: NetworkBackend = SyncBackend() if network_backend is None else network_backend
        self._connection: ConnectionInterface | None = None
        self._connect_failed: bool = False
        self._request_lock = Lock()
        self._socket_options = socket_options

    def handle_request(self, request: Request) -> Response:
        if not self.can_handle_request(request.url.origin):
            raise RuntimeError(f"Attempted to send request to {request.url.origin} on connection to {self._origin}")

        try:
            with self._request_lock:
                if self._connection is None:
                    stream = self._connect(request)

                    ssl_object = stream.get_extra_info("ssl_object")
                    http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == "h2"
                    if http2_negotiated or (self._http2 and not self._http1):
                        from .http2 import HTTP2Connection

                        self._connection = HTTP2Connection(
                            origin=self._origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
                    else:
                        self._connection = HTTP11Connection(
                            origin=self._origin,
                            stream=stream,
                            keepalive_expiry=self._keepalive_expiry,
                        )
        except BaseException as exc:
            self._connect_failed = True
            raise exc

        return self._connection.handle_request(request)

    def _connect(self, request: Request) -> NetworkStream:
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
                    with Trace("connect_tcp", logger, request, kwargs) as trace:
                        stream = self._network_backend.connect_tcp(**kwargs)
                        trace.return_value = stream
                else:
                    kwargs = {
                        "path": self._uds,
                        "timeout": timeout,
                        "socket_options": self._socket_options,
                    }
                    with Trace("connect_unix_socket", logger, request, kwargs) as trace:
                        stream = self._network_backend.connect_unix_socket(**kwargs)
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
                    with Trace("start_tls", logger, request, kwargs) as trace:
                        stream = stream.start_tls(**kwargs)
                        trace.return_value = stream
                return stream
            except (ConnectError, ConnectTimeout):
                if retries_left <= 0:
                    raise
                retries_left -= 1
                delay = next(delays)
                with Trace("retry", logger, request, kwargs) as trace:
                    self._network_backend.sleep(delay)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._origin

    def close(self) -> None:
        if self._connection is not None:
            with Trace("close", logger, None, {}):
                self._connection.close()

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

    def __enter__(self) -> HTTPConnection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        self.close()
<|endoftext|>from __future__ import annotations

import ssl
import typing

import trio

from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    ExceptionMapping,
    ReadError,
    ReadTimeout,
    WriteError,
    WriteTimeout,
    map_exceptions,
)
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream


class TrioStream(AsyncNetworkStream):
    def __init__(self, stream: trio.abc.Stream) -> None:
        self._stream = stream

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        timeout_or_inf = float("inf") if timeout is None else timeout
        exc_map: ExceptionMapping = {
            trio.TooSlowError: ReadTimeout,
            trio.BrokenResourceError: ReadError,
            trio.ClosedResourceError: ReadError,
        }
        with map_exceptions(exc_map):
            with trio.fail_after(timeout_or_inf):
                data: bytes = await self._stream.receive_some(max_bytes=max_bytes)
                return data

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return

        timeout_or_inf = float("inf") if timeout is None else timeout
        exc_map: ExceptionMapping = {
            trio.TooSlowError: WriteTimeout,
            trio.BrokenResourceError: WriteError,
            trio.ClosedResourceError: WriteError,
        }
        with map_exceptions(exc_map):
            with trio.fail_after(timeout_or_inf):
                await self._stream.send_all(data=buffer)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        timeout_or_inf = float("inf") if timeout is None else timeout
        exc_map: ExceptionMapping = {
            trio.TooSlowError: ConnectTimeout,
            trio.BrokenResourceError: ConnectError,
        }
        ssl_stream = trio.SSLStream(
            self._stream,
            ssl_context=ssl_context,
            server_hostname=server_hostname,
            https_compatible=True,
            server_side=False,
        )
        with map_exceptions(exc_map):
            try:
                with trio.fail_after(timeout_or_inf):
                    await ssl_stream.do_handshake()
            except Exception as exc:  # pragma: no cover
                await self.aclose()
                raise exc
        return TrioStream(ssl_stream)

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object" and isinstance(self._stream, trio.SSLStream):
            # Type checkers cannot see `_ssl_object` attribute because trio._ssl.SSLStream uses __getattr__/__setattr__.
            # Tracked at https://github.com/python-trio/trio/issues/542
            return self._stream._ssl_object  # type: ignore[attr-defined]
        if info == "client_addr":
            return self._get_socket_stream().socket.getsockname()
        if info == "server_addr":
            return self._get_socket_stream().socket.getpeername()
        if info == "socket":
            stream = self._stream
            while isinstance(stream, trio.SSLStream):
                stream = stream.transport_stream
            assert isinstance(stream, trio.SocketStream)
            return stream.socket
        if info == "is_readable":
            socket = self.get_extra_info("socket")
            return socket.is_readable()
        return None

    def _get_socket_stream(self) -> trio.SocketStream:
        stream = self._stream
        while isinstance(stream, trio.SSLStream):
            stream = stream.transport_stream
        assert isinstance(stream, trio.SocketStream)
        return stream


class TrioBackend(AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        # By default for TCP sockets, trio enables TCP_NODELAY.
        # https://trio.readthedocs.io/en/stable/reference-io.html#trio.SocketStream
        if socket_options is None:
            socket_options = []  # pragma: no cover
        timeout_or_inf = float("inf") if timeout is None else timeout
        exc_map: ExceptionMapping = {
            trio.TooSlowError: ConnectTimeout,
            trio.BrokenResourceError: ConnectError,
            OSError: ConnectError,
        }
        with map_exceptions(exc_map):
            with trio.fail_after(timeout_or_inf):
                stream: trio.abc.Stream = await trio.open_tcp_stream(host=host, port=port, local_address=local_address)
                for option in socket_options:
                    stream.setsockopt(*option)  # type: ignore[attr-defined] # pragma: no cover
        return TrioStream(stream)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: no cover
        if socket_options is None:
            socket_options = []
        timeout_or_inf = float("inf") if timeout is None else timeout
        exc_map: ExceptionMapping = {
            trio.TooSlowError: ConnectTimeout,
            trio.BrokenResourceError: ConnectError,
            OSError: ConnectError,
        }
        with map_exceptions(exc_map):
            with trio.fail_after(timeout_or_inf):
                stream: trio.abc.Stream = await trio.open_unix_socket(path)
                for option in socket_options:
                    stream.setsockopt(*option)  # type: ignore[attr-defined] # pragma: no cover
        return TrioStream(stream)

    async def sleep(self, seconds: float) -> None:
        await trio.sleep(seconds)  # pragma: no cover
<|endoftext|>from __future__ import annotations

import typing

from .._synchronization import current_async_library
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream


class AutoBackend(AsyncNetworkBackend):
    async def _init_backend(self) -> None:
        if not (hasattr(self, "_backend")):
            backend = current_async_library()
            if backend == "trio":
                from .trio import TrioBackend

                self._backend: AsyncNetworkBackend = TrioBackend()
            else:
                from .anyio import AnyIOBackend

                self._backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        await self._init_backend()
        return await self._backend.connect_tcp(
            host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: no cover
        await self._init_backend()
        return await self._backend.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:  # pragma: no cover
        await self._init_backend()
        return await self._backend.sleep(seconds)
<|endoftext|>from __future__ import annotations

import functools
import socket
import ssl
import sys
import typing

from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    ExceptionMapping,
    ReadError,
    ReadTimeout,
    WriteError,
    WriteTimeout,
    map_exceptions,
)
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, NetworkBackend, NetworkStream


class TLSinTLSStream(NetworkStream):  # pragma: no cover
    """
    Because the standard `SSLContext.wrap_socket` method does
    not work for `SSLSocket` objects, we need this class
    to implement TLS stream using an underlying `SSLObject`
    instance in order to support TLS on top of TLS.
    """

    # Defined in RFC 8449
    TLS_RECORD_SIZE = 16384

    def __init__(
        self,
        sock: socket.socket,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ):
        self._sock = sock
        self._incoming = ssl.MemoryBIO()
        self._outgoing = ssl.MemoryBIO()

        self.ssl_obj = ssl_context.wrap_bio(
            incoming=self._incoming,
            outgoing=self._outgoing,
            server_hostname=server_hostname,
        )

        self._sock.settimeout(timeout)
        self._perform_io(self.ssl_obj.do_handshake)

    def _perform_io(
        self,
        func: typing.Callable[..., typing.Any],
    ) -> typing.Any:
        ret = None

        while True:
            errno = None
            try:
                ret = func()
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError) as e:
                errno = e.errno

            self._sock.sendall(self._outgoing.read())

            if errno == ssl.SSL_ERROR_WANT_READ:
                buf = self._sock.recv(self.TLS_RECORD_SIZE)

                if buf:
                    self._incoming.write(buf)
                else:
                    self._incoming.write_eof()
            if errno is None:
                return ret

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        exc_map: ExceptionMapping = {socket.timeout: ReadTimeout, OSError: ReadError}
        with map_exceptions(exc_map):
            self._sock.settimeout(timeout)
            return typing.cast(bytes, self._perform_io(functools.partial(self.ssl_obj.read, max_bytes)))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        exc_map: ExceptionMapping = {socket.timeout: WriteTimeout, OSError: WriteError}
        with map_exceptions(exc_map):
            self._sock.settimeout(timeout)
            view = memoryview(buffer)  # zero-copy slicing; avoids copies
            while view:
                nsent = self._perform_io(functools.partial(self.ssl_obj.write, view))
                view = view[nsent:]

    def close(self) -> None:
        self._sock.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        raise NotImplementedError()

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object":
            return self.ssl_obj
        if info == "client_addr":
            return self._sock.getsockname()
        if info == "server_addr":
            return self._sock.getpeername()
        if info == "socket":
            return self._sock
        if info == "is_readable":
            return is_socket_readable(self._sock)
        return None


class SyncStream(NetworkStream):
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        exc_map: ExceptionMapping = {socket.timeout: ReadTimeout, OSError: ReadError}
        with map_exceptions(exc_map):
            self._sock.settimeout(timeout)
            return self._sock.recv(max_bytes)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return

        exc_map: ExceptionMapping = {socket.timeout: WriteTimeout, OSError: WriteError}
        with map_exceptions(exc_map):
            view = memoryview(buffer)  # zero-copy slicing; avoids copies
            while view:
                self._sock.settimeout(timeout)
                n = self._sock.send(view)
                view = view[n:]

    def close(self) -> None:
        self._sock.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        exc_map: ExceptionMapping = {
            socket.timeout: ConnectTimeout,
            OSError: ConnectError,
        }
        with map_exceptions(exc_map):
            try:
                if isinstance(self._sock, ssl.SSLSocket):  # pragma: no cover
                    # If the underlying socket has already been upgraded
                    # to the TLS layer (i.e. is an instance of SSLSocket),
                    # we need some additional smarts to support TLS-in-TLS.
                    return TLSinTLSStream(self._sock, ssl_context, server_hostname, timeout)
                else:
                    self._sock.settimeout(timeout)
                    sock = ssl_context.wrap_socket(self._sock, server_hostname=server_hostname)
            except Exception as exc:  # pragma: no cover
                self.close()
                raise exc
        return SyncStream(sock)

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object" and isinstance(self._sock, ssl.SSLSocket):
            return self._sock._sslobj  # type: ignore
        if info == "client_addr":
            return self._sock.getsockname()
        if info == "server_addr":
            return self._sock.getpeername()
        if info == "socket":
            return self._sock
        if info == "is_readable":
            return is_socket_readable(self._sock)
        return None


class SyncBackend(NetworkBackend):
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        # Note that we automatically include `TCP_NODELAY`
        # in addition to any other custom socket options.
        if socket_options is None:
            socket_options = []  # pragma: no cover
        address = (host, port)
        source_address = None if local_address is None else (local_address, 0)
        exc_map: ExceptionMapping = {
            socket.timeout: ConnectTimeout,
            OSError: ConnectError,
        }

        with map_exceptions(exc_map):
            sock = socket.create_connection(
                address,
                timeout,
                source_address=source_address,
            )
            for option in socket_options:
                sock.setsockopt(*option)  # pragma: no cover
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SyncStream(sock)

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:  # pragma: no cover
        if sys.platform == "win32":
            raise RuntimeError("Attempted to connect to a UNIX socket on a Windows system.")
        if socket_options is None:
            socket_options = []

        exc_map: ExceptionMapping = {
            socket.timeout: ConnectTimeout,
            OSError: ConnectError,
        }
        with map_exceptions(exc_map):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            for option in socket_options:
                sock.setsockopt(*option)
            sock.settimeout(timeout)
            sock.connect(path)
        return SyncStream(sock)
<|endoftext|>from __future__ import annotations

import ssl
import typing

from .._exceptions import ReadError
from .base import (
    SOCKET_OPTION,
    AsyncNetworkBackend,
    AsyncNetworkStream,
    NetworkBackend,
    NetworkStream,
)


class MockSSLObject:
    def __init__(self, http2: bool):
        self._http2 = http2

    def selected_alpn_protocol(self) -> str:
        return "h2" if self._http2 else "http/1.1"


class MockStream(NetworkStream):
    def __init__(self, buffer: list[bytes], http2: bool = False) -> None:
        self._buffer = buffer
        self._http2 = http2
        self._closed = False

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._closed:
            raise ReadError("Connection closed")
        if not self._buffer:
            return b""
        return self._buffer.pop(0)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        return self

    def get_extra_info(self, info: str) -> typing.Any:
        return MockSSLObject(http2=self._http2) if info == "ssl_object" else None

    def __repr__(self) -> str:
        return "<httpcore2.MockStream>"


class MockBackend(NetworkBackend):
    def __init__(self, buffer: list[bytes], http2: bool = False) -> None:
        self._buffer = buffer
        self._http2 = http2

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        return MockStream(list(self._buffer), http2=self._http2)

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        return MockStream(list(self._buffer), http2=self._http2)

    def sleep(self, seconds: float) -> None:
        pass


class AsyncMockStream(AsyncNetworkStream):
    def __init__(self, buffer: list[bytes], http2: bool = False) -> None:
        self._buffer = buffer
        self._http2 = http2
        self._closed = False

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._closed:
            raise ReadError("Connection closed")
        if not self._buffer:
            return b""
        return self._buffer.pop(0)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        pass

    async def aclose(self) -> None:
        self._closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        return self

    def get_extra_info(self, info: str) -> typing.Any:
        return MockSSLObject(http2=self._http2) if info == "ssl_object" else None

    def __repr__(self) -> str:
        return "<httpcore2.AsyncMockStream>"


class AsyncMockBackend(AsyncNetworkBackend):
    def __init__(self, buffer: list[bytes], http2: bool = False) -> None:
        self._buffer = buffer
        self._http2 = http2

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        return AsyncMockStream(list(self._buffer), http2=self._http2)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        return AsyncMockStream(list(self._buffer), http2=self._http2)

    async def sleep(self, seconds: float) -> None:
        pass
<|endoftext|>from __future__ import annotations

import ssl
import typing

import anyio
import anyio.abc
import anyio.streams.tls

from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    ReadError,
    ReadTimeout,
    WriteError,
    WriteTimeout,
    map_exceptions,
)
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream


class AnyIOStream(AsyncNetworkStream):
    def __init__(self, stream: anyio.abc.ByteStream) -> None:
        self._stream = stream

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        exc_map: dict[type[Exception], type[Exception]] = {
            TimeoutError: ReadTimeout,
            anyio.BrokenResourceError: ReadError,
            anyio.ClosedResourceError: ReadError,
            anyio.EndOfStream: ReadError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                try:
                    return await self._stream.receive(max_bytes=max_bytes)
                except anyio.EndOfStream:  # pragma: no cover
                    return b""

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return

        exc_map: dict[type[Exception], type[Exception]] = {
            TimeoutError: WriteTimeout,
            anyio.BrokenResourceError: WriteError,
            anyio.ClosedResourceError: WriteError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                await self._stream.send(item=buffer)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        exc_map: dict[type[Exception], type[Exception]] = {
            TimeoutError: ConnectTimeout,
            anyio.BrokenResourceError: ConnectError,
            anyio.EndOfStream: ConnectError,
            ssl.SSLError: ConnectError,
        }
        with map_exceptions(exc_map):
            try:
                with anyio.fail_after(timeout):
                    ssl_stream = await anyio.streams.tls.TLSStream.wrap(
                        self._stream,
                        ssl_context=ssl_context,
                        hostname=server_hostname,
                        standard_compatible=False,
                        server_side=False,
                    )
            except Exception as exc:  # pragma: no cover
                await self.aclose()
                raise exc
        return AnyIOStream(ssl_stream)

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object":
            return self._stream.extra(anyio.streams.tls.TLSAttribute.ssl_object, None)
        if info == "client_addr":
            return self._stream.extra(anyio.abc.SocketAttribute.local_address, None)
        if info == "server_addr":
            return self._stream.extra(anyio.abc.SocketAttribute.remote_address, None)
        if info == "socket":
            return self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
        if info == "is_readable":
            sock = self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
            return is_socket_readable(sock)
        return None


class AnyIOBackend(AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: no cover
        if socket_options is None:
            socket_options = []
        exc_map: dict[type[Exception], type[Exception]] = {
            TimeoutError: ConnectTimeout,
            OSError: ConnectError,
            anyio.BrokenResourceError: ConnectError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                stream: anyio.abc.ByteStream = await anyio.connect_tcp(
                    remote_host=host,
                    remote_port=port,
                    local_host=local_address,
                )
                # By default TCP sockets opened in `asyncio` include TCP_NODELAY.
                for option in socket_options:
                    stream._raw_socket.setsockopt(*option)  # type: ignore[attr-defined] # pragma: no cover
        return AnyIOStream(stream)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: no cover
        if socket_options is None:
            socket_options = []
        exc_map: dict[type[Exception], type[Exception]] = {
            TimeoutError: ConnectTimeout,
            OSError: ConnectError,
            anyio.BrokenResourceError: ConnectError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                stream: anyio.abc.ByteStream = await anyio.connect_unix(path)
                for option in socket_options:
                    stream._raw_socket.setsockopt(*option)  # type: ignore[attr-defined] # pragma: no cover
        return AnyIOStream(stream)

    async def sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)  # pragma: no cover
<|endoftext|>from __future__ import annotations

import ssl
import time
import typing

SOCKET_OPTION = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


class NetworkStream:
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        raise NotImplementedError()  # pragma: no cover

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        raise NotImplementedError()  # pragma: no cover

    def close(self) -> None:
        raise NotImplementedError()  # pragma: no cover

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> NetworkStream:
        raise NotImplementedError()  # pragma: no cover

    def get_extra_info(self, info: str) -> typing.Any:
        return None  # pragma: no cover


class NetworkBackend:
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        raise NotImplementedError()  # pragma: no cover

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> NetworkStream:
        raise NotImplementedError()  # pragma: no cover

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)  # pragma: no cover


class AsyncNetworkStream:
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        raise NotImplementedError()  # pragma: no cover

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        raise NotImplementedError()  # pragma: no cover

    async def aclose(self) -> None:
        raise NotImplementedError()  # pragma: no cover

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        raise NotImplementedError()  # pragma: no cover

    def get_extra_info(self, info: str) -> typing.Any:
        return None  # pragma: no cover


class AsyncNetworkBackend:
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        raise NotImplementedError()  # pragma: no cover

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        raise NotImplementedError()  # pragma: no cover

    async def sleep(self, seconds: float) -> None:
        raise NotImplementedError()  # pragma: no cover
<|endoftext|>import os
import sys
from typing import TYPE_CHECKING

__all__ = ("cached_property", "under_cached_property")


NO_EXTENSIONS = bool(os.environ.get("PROPCACHE_NO_EXTENSIONS"))  # type: bool
if sys.implementation.name != "cpython":
    NO_EXTENSIONS = True


# isort: off
if TYPE_CHECKING:
    from ._helpers_py import cached_property as cached_property_py
    from ._helpers_py import under_cached_property as under_cached_property_py

    cached_property = cached_property_py
    under_cached_property = under_cached_property_py
elif not NO_EXTENSIONS:  # pragma: no branch
    try:
        from ._helpers_c import cached_property as cached_property_c  # type: ignore[attr-defined, unused-ignore]
        from ._helpers_c import under_cached_property as under_cached_property_c  # type: ignore[attr-defined, unused-ignore]

        cached_property = cached_property_c
        under_cached_property = under_cached_property_c
    except ImportError:  # pragma: no cover
        from ._helpers_py import cached_property as cached_property_py
        from ._helpers_py import under_cached_property as under_cached_property_py

        cached_property = cached_property_py  # type: ignore[assignment, misc]
        under_cached_property = under_cached_property_py
else:
    from ._helpers_py import cached_property as cached_property_py
    from ._helpers_py import under_cached_property as under_cached_property_py

    cached_property = cached_property_py  # type: ignore[assignment, misc]
    under_cached_property = under_cached_property_py
# isort: on
<|endoftext|>"""propcache: An accelerated property cache for Python classes."""

from typing import TYPE_CHECKING

_PUBLIC_API = ("cached_property", "under_cached_property")

__version__ = "0.5.4"
__all__ = ()

# Imports have moved to `propcache.api` in 0.2.0+.
# This module is now a facade for the API.
if TYPE_CHECKING:
    from .api import cached_property as cached_property  # noqa: F401
    from .api import under_cached_property as under_cached_property  # noqa: F401


def _import_facade(attr: str) -> object:
    """Import the public API from the `api` module."""
    if attr in _PUBLIC_API:
        from . import api  # pylint: disable=import-outside-toplevel

        return getattr(api, attr)
    raise AttributeError(f"module '{__package__}' has no attribute '{attr}'")


def _dir_facade() -> list[str]:
    """Include the public API in the module's dir() output."""
    return [*_PUBLIC_API, *globals().keys()]


__getattr__ = _import_facade
__dir__ = _dir_facade
<|endoftext|>"""Various helper functions."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar, overload

__all__ = ("under_cached_property", "cached_property")


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

_T = TypeVar("_T")
# We use Mapping to make it possible to use TypedDict, but this isn't
# technically type safe as we need to assign into the dict.
_Cache = TypeVar("_Cache", bound=Mapping[str, Any])


class _CacheImpl(Protocol[_Cache]):
    _cache: _Cache


class under_cached_property(Generic[_T]):
    """Use as a class method decorator.

    It operates almost exactly like
    the Python `@property` decorator, but it puts the result of the
    method it decorates into the instance dict after the first call,
    effectively replacing the function it decorates with an instance
    variable.  It is, in Python parlance, a data descriptor.
    """

    def __init__(self, wrapped: Callable[[Any], _T]) -> None:
        self.wrapped = wrapped
        self.__doc__ = wrapped.__doc__
        self.name = wrapped.__name__

    @overload
    def __get__(self, inst: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(
        self, inst: _CacheImpl[Any], owner: type[object] | None = None
    ) -> _T: ...

    def __get__(
        self, inst: _CacheImpl[Any] | None, owner: type[object] | None = None
    ) -> _T | Self:
        if inst is None:
            return self
        try:
            return inst._cache[self.name]  # type: ignore[no-any-return]
        except KeyError:
            val = self.wrapped(inst)
            inst._cache[self.name] = val
            return val

    def __set__(self, inst: _CacheImpl[Any], value: _T) -> None:
        raise AttributeError("cached property is read-only")
<|endoftext|>import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template


def native_concat(values: t.Iterable[t.Any]) -> t.Optional[t.Any]:
    """Return a native Python type from the list of compiled nodes. If
    the result is a single node, its value is returned. Otherwise, the
    nodes are concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise,
    the string is returned.

    :param values: Iterable of outputs to concatenate.
    """
    head = list(islice(values, 2))

    if not head:
        return None

    if len(head) == 1:
        raw = head[0]
        if not isinstance(raw, str):
            return raw
    else:
        if isinstance(values, GeneratorType):
            values = chain(head, values)
        raw = "".join([str(v) for v in values])

    try:
        return literal_eval(
            # In Python 3.10+ ast.literal_eval removes leading spaces/tabs
            # from the given string. For backwards compatibility we need to
            # parse the string ourselves without removing leading spaces/tabs.
            parse(raw, mode="eval")
        )
    except (ValueError, SyntaxError, MemoryError):
        return raw


class NativeCodeGenerator(CodeGenerator):
    """A code generator which renders Python types by not adding
    ``str()`` around output nodes.
    """

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        return value

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        return repr("".join([str(v) for v in group]))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> t.Any:
        const = node.as_const(frame.eval_ctx)

        if not has_safe_repr(const):
            raise nodes.Impossible()

        if isinstance(node, nodes.TemplateData):
            return const

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if finalize.src is not None:
            self.write(")")


class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""

    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)  # type: ignore


class NativeTemplate(Template):
    environment_class = NativeEnvironment

    def render(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Render the template to produce a native Python type. If the
        result is a single node, its value is returned. Otherwise, the
        nodes are concatenated as strings. If the result can be parsed
        with :func:`ast.literal_eval`, the parsed value is returned.
        Otherwise, the string is returned.
        """
        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment_class.concat(  # type: ignore
                self.root_render_func(ctx)
            )
        except Exception:
            return self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment_class.concat(  # type: ignore
                [n async for n in self.root_render_func(ctx)]  # type: ignore
            )
        except Exception:
            return self.environment.handle_exception()


NativeEnvironment.template_class = NativeTemplate
<|endoftext|>import inspect
import typing as t
from functools import WRAPPER_ASSIGNMENTS
from functools import wraps

from .utils import _PassArg
from .utils import pass_eval_context

if t.TYPE_CHECKING:
    import typing_extensions as te

V = t.TypeVar("V")


def async_variant(normal_func):  # type: ignore
    def decorator(async_func):  # type: ignore
        pass_arg = _PassArg.from_obj(normal_func)
        need_eval_context = pass_arg is None

        if pass_arg is _PassArg.environment:

            def is_async(args: t.Any) -> bool:
                return t.cast(bool, args[0].is_async)

        else:

            def is_async(args: t.Any) -> bool:
                return t.cast(bool, args[0].environment.is_async)

        # Take the doc and annotations from the sync function, but the
        # name from the async function. Pallets-Sphinx-Themes
        # build_function_directive expects __wrapped__ to point to the
        # sync function.
        async_func_attrs = ("__module__", "__name__", "__qualname__")
        normal_func_attrs = tuple(set(WRAPPER_ASSIGNMENTS).difference(async_func_attrs))

        @wraps(normal_func, assigned=normal_func_attrs)
        @wraps(async_func, assigned=async_func_attrs, updated=())
        def wrapper(*args, **kwargs):  # type: ignore
            b = is_async(args)

            if need_eval_context:
                args = args[1:]

            if b:
                return async_func(*args, **kwargs)

            return normal_func(*args, **kwargs)

        if need_eval_context:
            wrapper = pass_eval_context(wrapper)

        wrapper.jinja_async_variant = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


_common_primitives = {int, float, bool, str, list, dict, tuple, type(None)}


async def auto_await(value: t.Union[t.Awaitable["V"], "V"]) -> "V":
    # Avoid a costly call to isawaitable
    if type(value) in _common_primitives:
        return t.cast("V", value)

    if inspect.isawaitable(value):
        return await t.cast("t.Awaitable[V]", value)

    return value


class _IteratorToAsyncIterator(t.Generic[V]):
    def __init__(self, iterator: "t.Iterator[V]"):
        self._iterator = iterator

    def __aiter__(self) -> "te.Self":
        return self

    async def __anext__(self) -> V:
        try:
            return next(self._iterator)
        except StopIteration as e:
            raise StopAsyncIteration(e.value) from e


def auto_aiter(
    iterable: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
) -> "t.AsyncIterator[V]":
    if hasattr(iterable, "__aiter__"):
        return iterable.__aiter__()
    else:
        return _IteratorToAsyncIterator(iter(iterable))


async def auto_to_list(
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
) -> t.List["V"]:
    return [x async for x in auto_aiter(value)]
<|endoftext|>import sys
import typing as t
from types import CodeType
from types import TracebackType

from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing

if t.TYPE_CHECKING:
    from .runtime import Context


def rewrite_traceback_stack(source: t.Optional[str] = None) -> BaseException:
    """Rewrite the current exception to replace any tracebacks from
    within compiled template code with tracebacks that look like they
    came from the template source.

    This must be called within an ``except`` block.

    :param source: For ``TemplateSyntaxError``, the original source if
        known.
    :return: The original exception with the rewritten traceback.
    """
    _, exc_value, tb = sys.exc_info()
    exc_value = t.cast(BaseException, exc_value)
    tb = t.cast(TracebackType, tb)

    if isinstance(exc_value, TemplateSyntaxError) and not exc_value.translated:
        exc_value.translated = True
        exc_value.source = source
        # Remove the old traceback, otherwise the frames from the
        # compiler still show up.
        exc_value.with_traceback(None)
        # Outside of runtime, so the frame isn't executing template
        # code, but it still needs to point at the template.
        tb = fake_traceback(
            exc_value, None, exc_value.filename or "<unknown>", exc_value.lineno
        )
    else:
        # Skip the frame for the render function.
        tb = tb.tb_next

    stack = []

    # Build the stack of traceback object, replacing any in template
    # code with the source file and line information.
    while tb is not None:
        # Skip frames decorated with @internalcode. These are internal
        # calls that aren't useful in template debugging output.
        if tb.tb_frame.f_code in internal_code:
            tb = tb.tb_next
            continue

        template = tb.tb_frame.f_globals.get("__jinja_template__")

        if template is not None:
            lineno = template.get_corresponding_lineno(tb.tb_lineno)
            fake_tb = fake_traceback(exc_value, tb, template.filename, lineno)
            stack.append(fake_tb)
        else:
            stack.append(tb)

        tb = tb.tb_next

    tb_next = None

    # Assign tb_next in reverse to avoid circular references.
    for tb in reversed(stack):
        tb.tb_next = tb_next
        tb_next = tb

    return exc_value.with_traceback(tb_next)


def fake_traceback(  # type: ignore
    exc_value: BaseException, tb: t.Optional[TracebackType], filename: str, lineno: int
) -> TracebackType:
    """Produce a new traceback object that looks like it came from the
    template source instead of the compiled code. The filename, line
    number, and location name will point to the template, and the local
    variables will be the current template context.

    :param exc_value: The original exception to be re-raised to create
        the new traceback.
    :param tb: The original traceback to get the local variables and
        code info from.
    :param filename: The template filename.
    :param lineno: The line number in the template source.
    """
    if tb is not None:
        # Replace the real locals with the context that would be
        # available at that point in the template.
        locals = get_template_locals(tb.tb_frame.f_locals)
        locals.pop("__jinja_exception__", None)
    else:
        locals = {}

    globals = {
        "__name__": filename,
        "__file__": filename,
        "__jinja_exception__": exc_value,
    }
    # Raise an exception at the correct line number.
    code: CodeType = compile(
        "\n" * (lineno - 1) + "raise __jinja_exception__", filename, "exec"
    )

    # Build a new code object that points to the template file and
    # replaces the location with a block name.
    location = "template"

    if tb is not None:
        function = tb.tb_frame.f_code.co_name

        if function == "root":
            location = "top-level template code"
        elif function.startswith("block_"):
            location = f"block {function[6:]!r}"

    if sys.version_info >= (3, 8):
        code = code.replace(co_name=location)
    else:
        code = CodeType(
            code.co_argcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            location,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )

    # Execute the new code, which is guaranteed to raise, and return
    # the new traceback without this frame.
    try:
        exec(code, globals, locals)
    except BaseException:
        return sys.exc_info()[2].tb_next  # type: ignore


def get_template_locals(real_locals: t.Mapping[str, t.Any]) -> t.Dict[str, t.Any]:
    """Based on the runtime locals, get the context that would be
    available at that point in the template.
    """
    # Start with the current template context.
    ctx: t.Optional[Context] = real_locals.get("context")

    if ctx is not None:
        data: t.Dict[str, t.Any] = ctx.get_all().copy()
    else:
        data = {}

    # Might be in a derived context that only sets local variables
    # rather than pushing a context. Local variables follow the scheme
    # l_depth_name. Find the highest-depth local that has a value for
    # each name.
    local_overrides: t.Dict[str, t.Tuple[int, t.Any]] = {}

    for name, value in real_locals.items():
        if not name.startswith("l_") or value is missing:
            # Not a template variable, or no longer relevant.
            continue

        try:
            _, depth_str, name = name.split("_", 2)
            depth = int(depth_str)
        except ValueError:
            continue

        cur_depth = local_overrides.get(name, (-1,))[0]

        if cur_depth < depth:
            local_overrides[name] = (depth, value)

    # Modify the context with any derived context.
    for name, (_, value) in local_overrides.items():
        if value is missing:
            data.pop(name, None)
        else:
            data[name] = value

    return data
<|endoftext|>"""Built-in template tests used with the ``is`` operator."""

import operator
import typing as t
from collections import abc
from numbers import Number

from .runtime import Undefined
from .utils import pass_environment

if t.TYPE_CHECKING:
    from .environment import Environment


def test_odd(value: int) -> bool:
    """Return true if the variable is odd."""
    return value % 2 == 1


def test_even(value: int) -> bool:
    """Return true if the variable is even."""
    return value % 2 == 0


def test_divisibleby(value: int, num: int) -> bool:
    """Check if a variable is divisible by a number."""
    return value % num == 0


def test_defined(value: t.Any) -> bool:
    """Return true if the variable is defined:

    .. sourcecode:: jinja

        {% if variable is defined %}
            value of variable: {{ variable }}
        {% else %}
            variable is not defined
        {% endif %}

    See the :func:`default` filter for a simple way to set undefined
    variables.
    """
    return not isinstance(value, Undefined)


def test_undefined(value: t.Any) -> bool:
    """Like :func:`defined` but the other way round."""
    return isinstance(value, Undefined)


@pass_environment
def test_filter(env: "Environment", value: str) -> bool:
    """Check if a filter exists by name. Useful if a filter may be
    optionally available.

    .. code-block:: jinja

        {% if 'markdown' is filter %}
            {{ value | markdown }}
        {% else %}
            {{ value }}
        {% endif %}

    .. versionadded:: 3.0
    """
    return value in env.filters


@pass_environment
def test_test(env: "Environment", value: str) -> bool:
    """Check if a test exists by name. Useful if a test may be
    optionally available.

    .. code-block:: jinja

        {% if 'loud' is test %}
            {% if value is loud %}
                {{ value|upper }}
            {% else %}
                {{ value|lower }}
            {% endif %}
        {% else %}
            {{ value }}
        {% endif %}

    .. versionadded:: 3.0
    """
    return value in env.tests


def test_none(value: t.Any) -> bool:
    """Return true if the variable is none."""
    return value is None


def test_boolean(value: t.Any) -> bool:
    """Return true if the object is a boolean value.

    .. versionadded:: 2.11
    """
    return value is True or value is False


def test_false(value: t.Any) -> bool:
    """Return true if the object is False.

    .. versionadded:: 2.11
    """
    return value is False


def test_true(value: t.Any) -> bool:
    """Return true if the object is True.

    .. versionadded:: 2.11
    """
    return value is True


# NOTE: The existing 'number' test matches booleans and floats
def test_integer(value: t.Any) -> bool:
    """Return true if the object is an integer.

    .. versionadded:: 2.11
    """
    return isinstance(value, int) and value is not True and value is not False


# NOTE: The existing 'number' test matches booleans and integers
def test_float(value: t.Any) -> bool:
    """Return true if the object is a float.

    .. versionadded:: 2.11
    """
    return isinstance(value, float)


def test_lower(value: str) -> bool:
    """Return true if the variable is lowercased."""
    return str(value).islower()


def test_upper(value: str) -> bool:
    """Return true if the variable is uppercased."""
    return str(value).isupper()


def test_string(value: t.Any) -> bool:
    """Return true if the object is a string."""
    return isinstance(value, str)


def test_mapping(value: t.Any) -> bool:
    """Return true if the object is a mapping (dict etc.).

    .. versionadded:: 2.6
    """
    return isinstance(value, abc.Mapping)


def test_number(value: t.Any) -> bool:
    """Return true if the variable is a number."""
    return isinstance(value, Number)


def test_sequence(value: t.Any) -> bool:
    """Return true if the variable is a sequence. Sequences are variables
    that are iterable.
    """
    try:
        len(value)
        value.__getitem__  # noqa B018
    except Exception:
        return False

    return True


def test_sameas(value: t.Any, other: t.Any) -> bool:
    """Check if an object points to the same memory address than another
    object:

    .. sourcecode:: jinja

        {% if foo.attribute is sameas false %}
            the foo attribute really is the `False` singleton
        {% endif %}
    """
    return value is other


def test_iterable(value: t.Any) -> bool:
    """Check if it's possible to iterate over an object."""
    try:
        iter(value)
    except TypeError:
        return False

    return True


def test_escaped(value: t.Any) -> bool:
    """Check if the value is escaped."""
    return hasattr(value, "__html__")


def test_in(value: t.Any, seq: t.Container[t.Any]) -> bool:
    """Check if value is in seq.

    .. versionadded:: 2.10
    """
    return value in seq


TESTS = {
    "odd": test_odd,
    "even": test_even,
    "divisibleby": test_divisibleby,
    "defined": test_defined,
    "undefined": test_undefined,
    "filter": test_filter,
    "test": test_test,
    "none": test_none,
    "boolean": test_boolean,
    "false": test_false,
    "true": test_true,
    "integer": test_integer,
    "float": test_float,
    "lower": test_lower,
    "upper": test_upper,
    "string": test_string,
    "mapping": test_mapping,
    "number": test_number,
    "sequence": test_sequence,
    "iterable": test_iterable,
    "callable": callable,
    "sameas": test_sameas,
    "escaped": test_escaped,
    "in": test_in,
    "==": operator.eq,
    "eq": operator.eq,
    "equalto": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    ">": operator.gt,
    "gt": operator.gt,
    "greaterthan": operator.gt,
    "ge": operator.ge,
    ">=": operator.ge,
    "<": operator.lt,
    "lt": operator.lt,
    "lessthan": operator.lt,
    "<=": operator.le,
    "le": operator.le,
}
<|endoftext|>"""Compiles nodes from the parser into Python code."""

import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword

from markupsafe import escape
from markupsafe import Markup

from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

F = t.TypeVar("F", bound=t.Callable[..., t.Any])

operators = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gteq": ">=",
    "lt": "<",
    "lteq": "<=",
    "in": "in",
    "notin": "not in",
}


def optimizeconst(f: F) -> F:
    def new_func(
        self: "CodeGenerator", node: nodes.Expr, frame: "Frame", **kwargs: t.Any
    ) -> t.Any:
        # Only optimize if the frame is not volatile
        if self.optimizer is not None and not frame.eval_ctx.volatile:
            new_node = self.optimizer.visit(node, frame.eval_ctx)

            if new_node != node:
                return self.visit(new_node, frame)

        return f(self, node, frame, **kwargs)

    return update_wrapper(new_func, f)  # type: ignore[return-value]


def _make_binop(op: str) -> t.Callable[["CodeGenerator", nodes.BinExpr, "Frame"], None]:
    @optimizeconst
    def visitor(self: "CodeGenerator", node: nodes.BinExpr, frame: Frame) -> None:
        if (
            self.environment.sandboxed and op in self.environment.intercepted_binops  # type: ignore
        ):
            self.write(f"environment.call_binop(context, {op!r}, ")
            self.visit(node.left, frame)
            self.write(", ")
            self.visit(node.right, frame)
        else:
            self.write("(")
            self.visit(node.left, frame)
            self.write(f" {op} ")
            self.visit(node.right, frame)

        self.write(")")

    return visitor


def _make_unop(
    op: str,
) -> t.Callable[["CodeGenerator", nodes.UnaryExpr, "Frame"], None]:
    @optimizeconst
    def visitor(self: "CodeGenerator", node: nodes.UnaryExpr, frame: Frame) -> None:
        if (
            self.environment.sandboxed and op in self.environment.intercepted_unops  # type: ignore
        ):
            self.write(f"environment.call_unop(context, {op!r}, ")
            self.visit(node.node, frame)
        else:
            self.write("(" + op)
            self.visit(node.node, frame)

        self.write(")")

    return visitor


def generate(
    node: nodes.Template,
    environment: "Environment",
    name: t.Optional[str],
    filename: t.Optional[str],
    stream: t.Optional[t.TextIO] = None,
    defer_init: bool = False,
    optimized: bool = True,
) -> t.Optional[str]:
    """Generate the python source for a node tree."""
    if not isinstance(node, nodes.Template):
        raise TypeError("Can't compile non template nodes")

    generator = environment.code_generator_class(
        environment, name, filename, stream, defer_init, optimized
    )
    generator.visit(node)

    if stream is None:
        return generator.stream.getvalue()  # type: ignore

    return None


def has_safe_repr(value: t.Any) -> bool:
    """Does the node have a safe representation?"""
    if value is None or value is NotImplemented or value is Ellipsis:
        return True

    if type(value) in {bool, int, float, complex, range, str, Markup}:
        return True

    if type(value) in {tuple, list, set, frozenset}:
        return all(has_safe_repr(v) for v in value)

    if type(value) is dict:  # noqa E721
        return all(has_safe_repr(k) and has_safe_repr(v) for k, v in value.items())

    return False


def find_undeclared(
    nodes: t.Iterable[nodes.Node], names: t.Iterable[str]
) -> t.Set[str]:
    """Check if the names passed are accessed undeclared.  The return value
    is a set of all the undeclared names from the sequence of names found.
    """
    visitor = UndeclaredNameVisitor(names)
    try:
        for node in nodes:
            visitor.visit(node)
    except VisitorExit:
        pass
    return visitor.undeclared


class MacroRef:
    def __init__(self, node: t.Union[nodes.Macro, nodes.CallBlock]) -> None:
        self.node = node
        self.accesses_caller = False
        self.accesses_kwargs = False
        self.accesses_varargs = False


class Frame:
    """Holds compile time information for us."""

    def __init__(
        self,
        eval_ctx: EvalContext,
        parent: t.Optional["Frame"] = None,
        level: t.Optional[int] = None,
    ) -> None:
        self.eval_ctx = eval_ctx

        # the parent of this frame
        self.parent = parent

        if parent is None:
            self.symbols = Symbols(level=level)

            # in some dynamic inheritance situations the compiler needs to add
            # write tests around output statements.
            self.require_output_check = False

            # inside some tags we are using a buffer rather than yield statements.
            # this for example affects {% filter %} or {% macro %}.  If a frame
            # is buffered this variable points to the name of the list used as
            # buffer.
            self.buffer: t.Optional[str] = None

            # the name of the block we're in, otherwise None.
            self.block: t.Optional[str] = None

        else:
            self.symbols = Symbols(parent.symbols, level=level)
            self.require_output_check = parent.require_output_check
            self.buffer = parent.buffer
            self.block = parent.block

        # a toplevel frame is the root + soft frames such as if conditions.
        self.toplevel = False

        # the root frame is basically just the outermost frame, so no if
        # conditions.  This information is used to optimize inheritance
        # situations.
        self.rootlevel = False

        # variables set inside of loops and blocks should not affect outer frames,
        # but they still needs to be kept track of as part of the active context.
        self.loop_frame = False
        self.block_frame = False

        # track whether the frame is being used in an if-statement or conditional
        # expression as it determines which errors should be raised during runtime
        # or compile time.
        self.soft_frame = False

    def copy(self) -> "te.Self":
        """Create a copy of the current one."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.symbols = self.symbols.copy()
        return rv

    def inner(self, isolated: bool = False) -> "Frame":
        """Return an inner frame."""
        if isolated:
            return Frame(self.eval_ctx, level=self.symbols.level + 1)
        return Frame(self.eval_ctx, self)

    def soft(self) -> "te.Self":
        """Return a soft frame.  A soft frame may not be modified as
        standalone thing as it shares the resources with the frame it
        was created of, but it's not a rootlevel frame any longer.

        This is only used to implement if-statements and conditional
        expressions.
        """
        rv = self.copy()
        rv.rootlevel = False
        rv.soft_frame = True
        return rv

    __copy__ = copy


class VisitorExit(RuntimeError):
    """Exception used by the `UndeclaredNameVisitor` to signal a stop."""


class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self) -> None:
        self.filters: t.Set[str] = set()
        self.tests: t.Set[str] = set()

    def visit_Filter(self, node: nodes.Filter) -> None:
        self.generic_visit(node)
        self.filters.add(node.name)

    def visit_Test(self, node: nodes.Test) -> None:
        self.generic_visit(node)
        self.tests.add(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting at blocks."""


class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names: t.Iterable[str]) -> None:
        self.names = set(names)
        self.undeclared: t.Set[str] = set()

    def visit_Name(self, node: nodes.Name) -> None:
        if node.ctx == "load" and node.name in self.names:
            self.undeclared.add(node.name)
            if self.undeclared == self.names:
                raise VisitorExit()
        else:
            self.names.discard(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting a blocks."""


class CompilerExit(Exception):
    """Raised if the compiler encountered a situation where it just
    doesn't make sense to further process the code.  Any block that
    raises such an exception is not further processed.
    """


class CodeGenerator(NodeVisitor):
    def __init__(
        self,
        environment: "Environment",
        name: t.Optional[str],
        filename: t.Optional[str],
        stream: t.Optional[t.TextIO] = None,
        defer_init: bool = False,
        optimized: bool = True,
    ) -> None:
        if stream is None:
            stream = StringIO()
        self.environment = environment
        self.name = name
        self.filename = filename
        self.stream = stream
        self.created_block_context = False
        self.defer_init = defer_init
        self.optimizer: t.Optional[Optimizer] = None

        if optimized:
            self.optimizer = Optimizer(environment)

        # aliases for imports
        self.import_aliases: t.Dict[str, str] = {}

        # a registry for all blocks.  Because blocks are moved out
        # into the global python scope they are registered here
        self.blocks: t.Dict[str, nodes.Block] = {}

        # the number of extends statements so far
        self.extends_so_far = 0

        # some templates have a rootlevel extends.  In this case we
        # can safely assume that we're a child template and do some
        # more optimizations.
        self.has_known_extends = False

        # the current line number
        self.code_lineno = 1

        # registry of all filters and tests (global, not block local)
        self.tests: t.Dict[str, str] = {}
        self.filters: t.Dict[str, str] = {}

        # the debug information
        self.debug_info: t.List[t.Tuple[int, int]] = []
        self._write_debug_info: t.Optional[int] = None

        # the number of new lines before the next write()
        self._new_lines = 0

        # the line number of the last written statement
        self._last_line = 0

        # true if nothing was written so far.
        self._first_write = True

        # used by the `temporary_identifier` method to get new
        # unique, temporary identifier
        self._last_identifier = 0

        # the current indentation
        self._indentation = 0

        # Tracks toplevel assignments
        self._assign_stack: t.List[t.Set[str]] = []

        # Tracks parameter definition blocks
        self._param_def_block: t.List[t.Set[str]] = []

        # Tracks the current context.
        self._context_reference_stack = ["context"]

    @property
    def optimized(self) -> bool:
        return self.optimizer is not None

    # -- Various compilation helpers

    def fail(self, msg: str, lineno: int) -> "te.NoReturn":
        """Fail with a :exc:`TemplateAssertionError`."""
        raise TemplateAssertionError(msg, lineno, self.name, self.filename)

    def temporary_identifier(self) -> str:
        """Get a new unique identifier."""
        self._last_identifier += 1
        return f"t_{self._last_identifier}"

    def buffer(self, frame: Frame) -> None:
        """Enable buffering for the frame from that point onwards."""
        frame.buffer = self.temporary_identifier()
        self.writeline(f"{frame.buffer} = []")

    def return_buffer_contents(
        self, frame: Frame, force_unescaped: bool = False
    ) -> None:
        """Return the buffer contents of the frame."""
        if not force_unescaped:
            if frame.eval_ctx.volatile:
                self.writeline("if context.eval_ctx.autoescape:")
                self.indent()
                self.writeline(f"return Markup(concat({frame.buffer}))")
                self.outdent()
                self.writeline("else:")
                self.indent()
                self.writeline(f"return concat({frame.buffer})")
                self.outdent()
                return
            elif frame.eval_ctx.autoescape:
                self.writeline(f"return Markup(concat({frame.buffer}))")
                return
        self.writeline(f"return concat({frame.buffer})")

    def indent(self) -> None:
        """Indent by one."""
        self._indentation += 1

    def outdent(self, step: int = 1) -> None:
        """Outdent by step."""
        self._indentation -= step

    def start_write(self, frame: Frame, node: t.Optional[nodes.Node] = None) -> None:
        """Yield or write into the frame buffer."""
        if frame.buffer is None:
            self.writeline("yield ", node)
        else:
            self.writeline(f"{frame.buffer}.append(", node)

    def end_write(self, frame: Frame) -> None:
        """End the writing process started by `start_write`."""
        if frame.buffer is not None:
            self.write(")")

    def simple_write(
        self, s: str, frame: Frame, node: t.Optional[nodes.Node] = None
    ) -> None:
        """Simple shortcut for start_write + write + end_write."""
        self.start_write(frame, node)
        self.write(s)
        self.end_write(frame)

    def blockvisit(self, nodes: t.Iterable[nodes.Node], frame: Frame) -> None:
        """Visit a list of nodes as block in a frame.  If the current frame
        is no buffer a dummy ``if 0: yield None`` is written automatically.
        """
        try:
            self.writeline("pass")
            for node in nodes:
                self.visit(node, frame)
        except CompilerExit:
            pass

    def write(self, x: str) -> None:
        """Write a string into the output stream."""
        if self._new_lines:
            if not self._first_write:
                self.stream.write("\n" * self._new_lines)
                self.code_lineno += self._new_lines
                if self._write_debug_info is not None:
                    self.debug_info.append((self._write_debug_info, self.code_lineno))
                    self._write_debug_info = None
            self._first_write = False
            self.stream.write("    " * self._indentation)
            self._new_lines = 0
        self.stream.write(x)

    def writeline(
        self, x: str, node: t.Optional[nodes.Node] = None, extra: int = 0
    ) -> None:
        """Combination of newline and write."""
        self.newline(node, extra)
        self.write(x)

    def newline(self, node: t.Optional[nodes.Node] = None, extra: int = 0) -> None:
        """Add one or more newlines before the next write."""
        self._new_lines = max(self._new_lines, 1 + extra)
        if node is not None and node.lineno != self._last_line:
            self._write_debug_info = node.lineno
            self._last_line = node.lineno

    def signature(
        self,
        node: t.Union[nodes.Call, nodes.Filter, nodes.Test],
        frame: Frame,
        extra_kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> None:
        """Writes a function call to the stream for the current node.
        A leading comma is added automatically.  The extra keyword
        arguments may not include python keywords otherwise a syntax
        error could occur.  The extra keyword arguments should be given
        as python dict.
        """
        # if any of the given keyword arguments is a python keyword
        # we have to make sure that no invalid call is created.
        kwarg_workaround = any(
            is_python_keyword(t.cast(str, k))
            for k in chain((x.key for x in node.kwargs), extra_kwargs or ())
        )

        for arg in node.args:
            self.write(", ")
            self.visit(arg, frame)

        if not kwarg_workaround:
            for kwarg in node.kwargs:
                self.write(", ")
                self.visit(kwarg, frame)
            if extra_kwargs is not None:
                for key, value in extra_kwargs.items():
                    self.write(f", {key}={value}")
        if node.dyn_args:
            self.write(", *")
            self.visit(node.dyn_args, frame)

        if kwarg_workaround:
            if node.dyn_kwargs is not None:
                self.write(", **dict({")
            else:
                self.write(", **{")
            for kwarg in node.kwargs:
                self.write(f"{kwarg.key!r}: ")
                self.visit(kwarg.value, frame)
                self.write(", ")
            if extra_kwargs is not None:
                for key, value in extra_kwargs.items():
                    self.write(f"{key!r}: {value}, ")
            if node.dyn_kwargs is not None:
                self.write("}, **")
                self.visit(node.dyn_kwargs, frame)
                self.write(")")
            else:
                self.write("}")

        elif node.dyn_kwargs is not None:
            self.write(", **")
            self.visit(node.dyn_kwargs, frame)

    def pull_dependencies(self, nodes: t.Iterable[nodes.Node]) -> None:
        """Find all filter and test names used in the template and
        assign them to variables in the compiled namespace. Checking
        that the names are registered with the environment is done when
        compiling the Filter and Test nodes. If the node is in an If or
        CondExpr node, the check is done at runtime instead.

        .. versionchanged:: 3.0
            Filters and tests in If and CondExpr nodes are checked at
            runtime instead of compile time.
        """
        visitor = DependencyFinderVisitor()

        for node in nodes:
            visitor.visit(node)

        for id_map, names, dependency in (
            (self.filters, visitor.filters, "filters"),
            (
                self.tests,
                visitor.tests,
                "tests",
            ),
        ):
            for name in sorted(names):
                if name not in id_map:
                    id_map[name] = self.temporary_identifier()

                # add check during runtime that dependencies used inside of executed
                # blocks are defined, as this step may be skipped during compile time
                self.writeline("try:")
                self.indent()
                self.writeline(f"{id_map[name]} = environment.{dependency}[{name!r}]")
                self.outdent()
                self.writeline("except KeyError:")
                self.indent()
                self.writeline("@internalcode")
                self.writeline(f"def {id_map[name]}(*unused):")
                self.indent()
                self.writeline(
                    f'raise TemplateRuntimeError("No {dependency[:-1]}'
                    f' named {name!r} found.")'
                )
                self.outdent()
                self.outdent()

    def enter_frame(self, frame: Frame) -> None:
        undefs = []
        for target, (action, param) in frame.symbols.loads.items():
            if action == VAR_LOAD_PARAMETER:
                pass
            elif action == VAR_LOAD_RESOLVE:
                self.writeline(f"{target} = {self.get_resolve_func()}({param!r})")
            elif action == VAR_LOAD_ALIAS:
                self.writeline(f"{target} = {param}")
            elif action == VAR_LOAD_UNDEFINED:
                undefs.append(target)
            else:
                raise NotImplementedError("unknown load instruction")
        if undefs:
            self.writeline(f"{' = '.join(undefs)} = missing")

    def leave_frame(self, frame: Frame, with_python_scope: bool = False) -> None:
        if not with_python_scope:
            undefs = []
            for target in frame.symbols.loads:
                undefs.append(target)
            if undefs:
                self.writeline(f"{' = '.join(undefs)} = missing")

    def choose_async(self, async_value: str = "async ", sync_value: str = "") -> str:
        return async_value if self.environment.is_async else sync_value

    def func(self, name: str) -> str:
        return f"{self.choose_async()}def {name}"

    def macro_body(
        self, node: t.Union[nodes.Macro, nodes.CallBlock], frame: Frame
    ) -> t.Tuple[Frame, MacroRef]:
        """Dump the function def of a macro or call block."""
        frame = frame.inner()
        frame.symbols.analyze_node(node)
        macro_ref = MacroRef(node)

        explicit_caller = None
        skip_special_params = set()
        args = []

        for idx, arg in enumerate(node.args):
            if arg.name == "caller":
                explicit_caller = idx
            if arg.name in ("kwargs", "varargs"):
                skip_special_params.add(arg.name)
            args.append(frame.symbols.ref(arg.name))

        undeclared = find_undeclared(node.body, ("caller", "kwargs", "varargs"))

        if "caller" in undeclared:
            # In older Jinja versions there was a bug that allowed caller
            # to retain the special behavior even if it was mentioned in
            # the argument list.  However thankfully this was only really
            # working if it was the last argument.  So we are explicitly
            # checking this now and error out if it is anywhere else in
            # the argument list.
            if explicit_caller is not None:
                try:
                    node.defaults[explicit_caller - len(node.args)]
                except IndexError:
                    self.fail(
                        "When defining macros or call blocks the "
                        'special "caller" argument must be omitted '
                        "or be given a default.",
                        node.lineno,
                    )
            else:
                args.append(frame.symbols.declare_parameter("caller"))
            macro_ref.accesses_caller = True
        if "kwargs" in undeclared and "kwargs" not in skip_special_params:
            args.append(frame.symbols.declare_parameter("kwargs"))
            macro_ref.accesses_kwargs = True
        if "varargs" in undeclared and "varargs" not in skip_special_params:
            args.append(frame.symbols.declare_parameter("varargs"))
            macro_ref.accesses_varargs = True

        # macros are delayed, they never require output checks
        frame.require_output_check = False
        frame.symbols.analyze_node(node)
        self.writeline(f"{self.func('macro')}({', '.join(args)}):", node)
        self.indent()

        self.buffer(frame)
        self.enter_frame(frame)

        self.push_parameter_definitions(frame)
        for idx, arg in enumerate(node.args):
            ref = frame.symbols.ref(arg.name)
            self.writeline(f"if {ref} is missing:")
            self.indent()
            try:
                default = node.defaults[idx - len(node.args)]
            except IndexError:
                self.writeline(
                    f'{ref} = undefined("parameter {arg.name!r} was not provided",'
                    f" name={arg.name!r})"
                )
            else:
                self.writeline(f"{ref} = ")
                self.visit(default, frame)
            self.mark_parameter_stored(ref)
            self.outdent()
        self.pop_parameter_definitions()

        self.blockvisit(node.body, frame)
        self.return_buffer_contents(frame, force_unescaped=True)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        return frame, macro_ref

    def macro_def(self, macro_ref: MacroRef, frame: Frame) -> None:
        """Dump the macro definition for the def created by macro_body."""
        arg_tuple = ", ".join(repr(x.name) for x in macro_ref.node.args)
        name = getattr(macro_ref.node, "name", None)
        if len(macro_ref.node.args) == 1:
            arg_tuple += ","
        self.write(
            f"Macro(environment, macro, {name!r}, ({arg_tuple}),"
            f" {macro_ref.accesses_kwargs!r}, {macro_ref.accesses_varargs!r},"
            f" {macro_ref.accesses_caller!r}, context.eval_ctx.autoescape)"
        )

    def position(self, node: nodes.Node) -> str:
        """Return a human readable position for the node."""
        rv = f"line {node.lineno}"
        if self.name is not None:
            rv = f"{rv} in {self.name!r}"
        return rv

    def dump_local_context(self, frame: Frame) -> str:
        items_kv = ", ".join(
            f"{name!r}: {target}"
            for name, target in frame.symbols.dump_stores().items()
        )
        return f"{{{items_kv}}}"

    def write_commons(self) -> None:
        """Writes a common preamble that is used by root and block functions.
        Primarily this sets up common local helpers and enforces a generator
        through a dead branch.
        """
        self.writeline("resolve = context.resolve_or_missing")
        self.writeline("undefined = environment.undefined")
        self.writeline("concat = environment.concat")
        # always use the standard Undefined class for the implicit else of
        # conditional expressions
        self.writeline("cond_expr_undefined = Undefined")
        self.writeline("if 0: yield None")

    def push_parameter_definitions(self, frame: Frame) -> None:
        """Pushes all parameter targets from the given frame into a local
        stack that permits tracking of yet to be assigned parameters.  In
        particular this enables the optimization from `visit_Name` to skip
        undefined expressions for parameters in macros as macros can reference
        otherwise unbound parameters.
        """
        self._param_def_block.append(frame.symbols.dump_param_targets())

    def pop_parameter_definitions(self) -> None:
        """Pops the current parameter definitions set."""
        self._param_def_block.pop()

    def mark_parameter_stored(self, target: str) -> None:
        """Marks a parameter in the current parameter definitions as stored.
        This will skip the enforced undefined checks.
        """
        if self._param_def_block:
            self._param_def_block[-1].discard(target)

    def push_context_reference(self, target: str) -> None:
        self._context_reference_stack.append(target)

    def pop_context_reference(self) -> None:
        self._context_reference_stack.pop()

    def get_context_ref(self) -> str:
        return self._context_reference_stack[-1]

    def get_resolve_func(self) -> str:
        target = self._context_reference_stack[-1]
        if target == "context":
            return "resolve"
        return f"{target}.resolve"

    def derive_context(self, frame: Frame) -> str:
        return f"{self.get_context_ref()}.derived({self.dump_local_context(frame)})"

    def parameter_is_undeclared(self, target: str) -> bool:
        """Checks if a given target is an undeclared parameter."""
        if not self._param_def_block:
            return False
        return target in self._param_def_block[-1]

    def push_assign_tracking(self) -> None:
        """Pushes a new layer for assignment tracking."""
        self._assign_stack.append(set())

    def pop_assign_tracking(self, frame: Frame) -> None:
        """Pops the topmost level for assignment tracking and updates the
        context variables if necessary.
        """
        vars = self._assign_stack.pop()
        if (
            not frame.block_frame
            and not frame.loop_frame
            and not frame.toplevel
            or not vars
        ):
            return
        public_names = [x for x in vars if x[:1] != "_"]
        if len(vars) == 1:
            name = next(iter(vars))
            ref = frame.symbols.ref(name)
            if frame.loop_frame:
                self.writeline(f"_loop_vars[{name!r}] = {ref}")
                return
            if frame.block_frame:
                self.writeline(f"_block_vars[{name!r}] = {ref}")
                return
            self.writeline(f"context.vars[{name!r}] = {ref}")
        else:
            if frame.loop_frame:
                self.writeline("_loop_vars.update({")
            elif frame.block_frame:
                self.writeline("_block_vars.update({")
            else:
                self.writeline("context.vars.update({")
            for idx, name in enumerate(sorted(vars)):
                if idx:
                    self.write(", ")
                ref = frame.symbols.ref(name)
                self.write(f"{name!r}: {ref}")
            self.write("})")
        if not frame.block_frame and not frame.loop_frame and public_names:
            if len(public_names) == 1:
                self.writeline(f"context.exported_vars.add({public_names[0]!r})")
            else:
                names_str = ", ".join(map(repr, sorted(public_names)))
                self.writeline(f"context.exported_vars.update(({names_str}))")

    # -- Statement Visitors

    def visit_Template(
        self, node: nodes.Template, frame: t.Optional[Frame] = None
    ) -> None:
        assert frame is None, "no root frame allowed"
        eval_ctx = EvalContext(self.environment, self.name)

        from .runtime import async_exported
        from .runtime import exported

        if self.environment.is_async:
            exported_names = sorted(exported + async_exported)
        else:
            exported_names = sorted(exported)

        self.writeline("from jinja2.runtime import " + ", ".join(exported_names))

        # if we want a deferred initialization we cannot move the
        # environment into a local name
        envenv = "" if self.defer_init else ", environment=environment"

        # do we have an extends tag at all?  If not, we can save some
        # overhead by just not processing any inheritance code.
        have_extends = node.find(nodes.Extends) is not None

        # find all blocks
        for block in node.find_all(nodes.Block):
            if block.name in self.blocks:
                self.fail(f"block {block.name!r} defined twice", block.lineno)
            self.blocks[block.name] = block

        # find all imports and import them
        for import_ in node.find_all(nodes.ImportedName):
            if import_.importname not in self.import_aliases:
                imp = import_.importname
                self.import_aliases[imp] = alias = self.temporary_identifier()
                if "." in imp:
                    module, obj = imp.rsplit(".", 1)
                    self.writeline(f"from {module} import {obj} as {alias}")
                else:
                    self.writeline(f"import {imp} as {alias}")

        # add the load name
        self.writeline(f"name = {self.name!r}")

        # generate the root render function.
        self.writeline(
            f"{self.func('root')}(context, missing=missing{envenv}):", extra=1
        )
        self.indent()
        self.write_commons()

        # process the root
        frame = Frame(eval_ctx)
        if "self" in find_undeclared(node.body, ("self",)):
            ref = frame.symbols.declare_parameter("self")
            self.writeline(f"{ref} = TemplateReference(context)")
        frame.symbols.analyze_node(node)
        frame.toplevel = frame.rootlevel = True
        frame.require_output_check = have_extends and not self.has_known_extends
        if have_extends:
            self.writeline("parent_template = None")
        self.enter_frame(frame)
        self.pull_dependencies(node.body)
        self.blockvisit(node.body, frame)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        # make sure that the parent root is called.
        if have_extends:
            if not self.has_known_extends:
                self.indent()
                self.writeline("if parent_template is not None:")
            self.indent()
            if not self.environment.is_async:
                self.writeline("yield from parent_template.root_render_func(context)")
            else:
                self.writeline("agen = parent_template.root_render_func(context)")
                self.writeline("try:")
                self.indent()
                self.writeline("async for event in agen:")
                self.indent()
                self.writeline("yield event")
                self.outdent()
                self.outdent()
                self.writeline("finally: await agen.aclose()")
            self.outdent(1 + (not self.has_known_extends))

        # at this point we now have the blocks collected and can visit them too.
        for name, block in self.blocks.items():
            self.writeline(
                f"{self.func('block_' + name)}(context, missing=missing{envenv}):",
                block,
                1,
            )
            self.indent()
            self.write_commons()
            # It's important that we do not make this frame a child of the
            # toplevel template.  This would cause a variety of
            # interesting issues with identifier tracking.
            block_frame = Frame(eval_ctx)
            block_frame.block_frame = True
            undeclared = find_undeclared(block.body, ("self", "super"))
            if "self" in undeclared:
                ref = block_frame.symbols.declare_parameter("self")
                self.writeline(f"{ref} = TemplateReference(context)")
            if "super" in undeclared:
                ref = block_frame.symbols.declare_parameter("super")
                self.writeline(f"{ref} = context.super({name!r}, block_{name})")
            block_frame.symbols.analyze_node(block)
            block_frame.block = name
            self.writeline("_block_vars = {}")
            self.enter_frame(block_frame)
            self.pull_dependencies(block.body)
            self.blockvisit(block.body, block_frame)
            self.leave_frame(block_frame, with_python_scope=True)
            self.outdent()

        blocks_kv_str = ", ".join(f"{x!r}: block_{x}" for x in self.blocks)
        self.writeline(f"blocks = {{{blocks_kv_str}}}", extra=1)
        debug_kv_str = "&".join(f"{k}={v}" for k, v in self.debug_info)
        self.writeline(f"debug_info = {debug_kv_str!r}")

    def visit_Block(self, node: nodes.Block, frame: Frame) -> None:
        """Call a block and register it for the template."""
        level = 0
        if frame.toplevel:
            # if we know that we are a child template, there is no need to
            # check if we are one
            if self.has_known_extends:
                return
            if self.extends_so_far > 0:
                self.writeline("if parent_template is None:")
                self.indent()
                level += 1

        if node.scoped:
            context = self.derive_context(frame)
        else:
            context = self.get_context_ref()

        if node.required:
            self.writeline(f"if len(context.blocks[{node.name!r}]) <= 1:", node)
            self.indent()
            self.writeline(
                f'raise TemplateRuntimeError("Required block {node.name!r} not found")',
                node,
            )
            self.outdent()

        if not self.environment.is_async and frame.buffer is None:
            self.writeline(
                f"yield from context.blocks[{node.name!r}][0]({context})", node
            )
        else:
            self.writeline(f"gen = context.blocks[{node.name!r}][0]({context})")
            self.writeline("try:")
            self.indent()
            self.writeline(
                f"{self.choose_async()}for event in gen:",
                node,
            )
            self.indent()
            self.simple_write("event", frame)
            self.outdent()
            self.outdent()
            self.writeline(
                f"finally: {self.choose_async('await gen.aclose()', 'gen.close()')}"
            )

        self.outdent(level)

    def visit_Extends(self, node: nodes.Extends, frame: Frame) -> None:
        """Calls the extender."""
        if not frame.toplevel:
            self.fail("cannot use extend from a non top-level scope", node.lineno)

        # if the number of extends statements in general is zero so
        # far, we don't have to add a check if something extended
        # the template before this one.
        if self.extends_so_far > 0:
            # if we have a known extends we just add a template runtime
            # error into the generated code.  We could catch that at compile
            # time too, but i welcome it not to confuse users by throwing the
            # same error at different times just "because we can".
            if not self.has_known_extends:
                self.writeline("if parent_template is not None:")
                self.indent()
            self.writeline('raise TemplateRuntimeError("extended multiple times")')

            # if we have a known extends already we don't need that code here
            # as we know that the template execution will end here.
            if self.has_known_extends:
                raise CompilerExit()
            else:
                self.outdent()

        self.writeline("parent_template = environment.get_template(", node)
        self.visit(node.template, frame)
        self.write(f", {self.name!r})")
        self.writeline("for name, parent_block in parent_template.blocks.items():")
        self.indent()
        self.writeline("context.blocks.setdefault(name, []).append(parent_block)")
        self.outdent()

        # if this extends statement was in the root level we can take
        # advantage of that information and simplify the generated code
        # in the top level from this point onwards
        if frame.rootlevel:
            self.has_known_extends = True

        # and now we have one more
        self.extends_so_far += 1

    def visit_Include(self, node: nodes.Include, frame: Frame) -> None:
        """Handles includes."""
        if node.ignore_missing:
            self.writeline("try:")
            self.indent()

        func_name = "get_or_select_template"
        if isinstance(node.template, nodes.Const):
            if isinstance(node.template.value, str):
                func_name = "get_template"
            elif isinstance(node.template.value, (tuple, list)):
                func_name = "select_template"
        elif isinstance(node.template, (nodes.Tuple, nodes.List)):
            func_name = "select_template"

        self.writeline(f"template = environment.{func_name}(", node)
        self.visit(node.template, frame)
        self.write(f", {self.name!r})")
        if node.ignore_missing:
            self.outdent()
            self.writeline("except TemplateNotFound:")
            self.indent()
            self.writeline("pass")
            self.outdent()
            self.writeline("else:")
            self.indent()

        def loop_body() -> None:
            self.indent()
            self.simple_write("event", frame)
            self.outdent()

        if node.with_context:
            self.writeline(
                f"gen = template.root_render_func("
                "template.new_context(context.get_all(), True,"
                f" {self.dump_local_context(frame)}))"
            )
            self.writeline("try:")
            self.indent()
            self.writeline(f"{self.choose_async()}for event in gen:")
            loop_body()
            self.outdent()
            self.writeline(
                f"finally: {self.choose_async('await gen.aclose()', 'gen.close()')}"
            )
        elif self.environment.is_async:
            self.writeline(
                "for event in (await template._get_default_module_async())"
                "._body_stream:"
            )
            loop_body()
        else:
            self.writeline("yield from template._get_default_module()._body_stream")

        if node.ignore_missing:
            self.outdent()

    def _import_common(
        self, node: t.Union[nodes.Import, nodes.FromImport], frame: Frame
    ) -> None:
        self.write(f"{self.choose_async('await ')}environment.get_template(")
        self.visit(node.template, frame)
        self.write(f", {self.name!r}).")

        if node.with_context:
            f_name = f"make_module{self.choose_async('_async')}"
            self.write(
                f"{f_name}(context.get_all(), True, {self.dump_local_context(frame)})"
            )
        else:
            self.write(f"_get_default_module{self.choose_async('_async')}(context)")

    def visit_Import(self, node: nodes.Import, frame: Frame) -> None:
        """Visit regular imports."""
        self.writeline(f"{frame.symbols.ref(node.target)} = ", node)
        if frame.toplevel:
            self.write(f"context.vars[{node.target!r}] = ")

        self._import_common(node, frame)

        if frame.toplevel and not node.target.startswith("_"):
            self.writeline(f"context.exported_vars.discard({node.target!r})")

    def visit_FromImport(self, node: nodes.FromImport, frame: Frame) -> None:
        """Visit named imports."""
        self.newline(node)
        self.write("included_template = ")
        self._import_common(node, frame)
        var_names = []
        discarded_names = []
        for name in node.names:
            if isinstance(name, tuple):
                name, alias = name
            else:
                alias = name
            self.writeline(
                f"{frame.symbols.ref(alias)} ="
                f" getattr(included_template, {name!r}, missing)"
            )
            self.writeline(f"if {frame.symbols.ref(alias)} is missing:")
            self.indent()
            # The position will contain the template name, and will be formatted
            # into a string that will be compiled into an f-string. Curly braces
            # in the name must be replaced with escapes so that they will not be
            # executed as part of the f-string.
            position = self.position(node).replace("{", "{{").replace("}", "}}")
            message = (
                "the template {included_template.__name__!r}"
                f" (imported on {position})"
                f" does not export the requested name {name!r}"
            )
            self.writeline(
                f"{frame.symbols.ref(alias)} = undefined(f{message!r}, name={name!r})"
            )
            self.outdent()
            if frame.toplevel:
                var_names.append(alias)
                if not alias.startswith("_"):
                    discarded_names.append(alias)

        if var_names:
            if len(var_names) == 1:
                name = var_names[0]
                self.writeline(f"context.vars[{name!r}] = {frame.symbols.ref(name)}")
            else:
                names_kv = ", ".join(
                    f"{name!r}: {frame.symbols.ref(name)}" for name in var_names
                )
                self.writeline(f"context.vars.update({{{names_kv}}})")
        if discarded_names:
            if len(discarded_names) == 1:
                self.writeline(f"context.exported_vars.discard({discarded_names[0]!r})")
            else:
                names_str = ", ".join(map(repr, discarded_names))
                self.writeline(
                    f"context.exported_vars.difference_update(({names_str}))"
                )

    def visit_For(self, node: nodes.For, frame: Frame) -> None:
        loop_frame = frame.inner()
        loop_frame.loop_frame = True
        test_frame = frame.inner()
        else_frame = frame.inner()

        # try to figure out if we have an extended loop.  An extended loop
        # is necessary if the loop is in recursive mode if the special loop
        # variable is accessed in the body if the body is a scoped block.
        extended_loop = (
            node.recursive
            or "loop"
            in find_undeclared(node.iter_child_nodes(only=("body",)), ("loop",))
            or any(block.scoped for block in node.find_all(nodes.Block))
        )

        loop_ref = None
        if extended_loop:
            loop_ref = loop_frame.symbols.declare_parameter("loop")

        loop_frame.symbols.analyze_node(node, for_branch="body")
        if node.else_:
            else_frame.symbols.analyze_node(node, for_branch="else")

        if node.test:
            loop_filter_func = self.temporary_identifier()
            test_frame.symbols.analyze_node(node, for_branch="test")
            self.writeline(f"{self.func(loop_filter_func)}(fiter):", node.test)
            self.indent()
            self.enter_frame(test_frame)
            self.writeline(self.choose_async("async for ", "for "))
            self.visit(node.target, loop_frame)
            self.write(" in ")
            self.write(self.choose_async("auto_aiter(fiter)", "fiter"))
            self.write(":")
            self.indent()
            self.writeline("if ", node.test)
            self.visit(node.test, test_frame)
            self.write(":")
            self.indent()
            self.writeline("yield ")
            self.visit(node.target, loop_frame)
            self.outdent(3)
            self.leave_frame(test_frame, with_python_scope=True)

        # if we don't have an recursive loop we have to find the shadowed
        # variables at that point.  Because loops can be nested but the loop
        # variable is a special one we have to enforce aliasing for it.
        if node.recursive:
            self.writeline(
                f"{self.func('loop')}(reciter, loop_render_func, depth=0):", node
            )
            self.indent()
            self.buffer(loop_frame)

            # Use the same buffer for the else frame
            else_frame.buffer = loop_frame.buffer

        # make sure the loop variable is a special one and raise a template
        # assertion error if a loop tries to write to loop
        if extended_loop:
            self.writeline(f"{loop_ref} = missing")

        for name in node.find_all(nodes.Name):
            if name.ctx == "store" and name.name == "loop":
                self.fail(
                    "Can't assign to special loop variable in for-loop target",
                    name.lineno,
                )

        if node.else_:
            iteration_indicator = self.temporary_identifier()
            self.writeline(f"{iteration_indicator} = 1")

        self.writeline(self.choose_async("async for ", "for "), node)
        self.visit(node.target, loop_frame)
        if extended_loop:
            self.write(f", {loop_ref} in {self.choose_async('Async')}LoopContext(")
        else:
            self.write(" in ")

        if node.test:
            self.write(f"{loop_filter_func}(")
        if node.recursive:
            self.write("reciter")
        else:
            if self.environment.is_async and not extended_loop:
                self.write("auto_aiter(")
            self.visit(node.iter, frame)
            if self.environment.is_async and not extended_loop:
                self.write(")")
        if node.test:
            self.write(")")

        if node.recursive:
            self.write(", undefined, loop_render_func, depth):")
        else:
            self.write(", undefined):" if extended_loop else ":")

        self.indent()
        self.enter_frame(loop_frame)

        self.writeline("_loop_vars = {}")
        self.blockvisit(node.body, loop_frame)
        if node.else_:
            self.writeline(f"{iteration_indicator} = 0")
        self.outdent()
        self.leave_frame(
            loop_frame, with_python_scope=node.recursive and not node.else_
        )

        if node.else_:
            self.writeline(f"if {iteration_indicator}:")
            self.indent()
            self.enter_frame(else_frame)
            self.blockvisit(node.else_, else_frame)
            self.leave_frame(else_frame)
            self.outdent()

        # if the node was recursive we have to return the buffer contents
        # and start the iteration code
        if node.recursive:
            self.return_buffer_contents(loop_frame)
            self.outdent()
            self.start_write(frame, node)
            self.write(f"{self.choose_async('await ')}loop(")
            if self.environment.is_async:
                self.write("auto_aiter(")
            self.visit(node.iter, frame)
            if self.environment.is_async:
                self.write(")")
            self.write(", loop)")
            self.end_write(frame)

        # at the end of the iteration, clear any assignments made in the
        # loop from the top level
        if self._assign_stack:
            self._assign_stack[-1].difference_update(loop_frame.symbols.stores)

    def visit_If(self, node: nodes.If, frame: Frame) -> None:
        if_frame = frame.soft()
        self.writeline("if ", node)
        self.visit(node.test, if_frame)
        self.write(":")
        self.indent()
        self.blockvisit(node.body, if_frame)
        self.outdent()
        for elif_ in node.elif_:
            self.writeline("elif ", elif_)
            self.visit(elif_.test, if_frame)
            self.write(":")
            self.indent()
            self.blockvisit(elif_.body, if_frame)
            self.outdent()
        if node.else_:
            self.writeline("else:")
            self.indent()
            self.blockvisit(node.else_, if_frame)
            self.outdent()

    def visit_Macro(self, node: nodes.Macro, frame: Frame) -> None:
        macro_frame, macro_ref = self.macro_body(node, frame)
        self.newline()
        if frame.toplevel:
            if not node.name.startswith("_"):
                self.write(f"context.exported_vars.add({node.name!r})")
            self.writeline(f"context.vars[{node.name!r}] = ")
        self.write(f"{frame.symbols.ref(node.name)} = ")
        self.macro_def(macro_ref, macro_frame)

    def visit_CallBlock(self, node: nodes.CallBlock, frame: Frame) -> None:
        call_frame, macro_ref = self.macro_body(node, frame)
        self.writeline("caller = ")
        self.macro_def(macro_ref, call_frame)
        self.start_write(frame, node)
        self.visit_Call(node.call, frame, forward_caller=True)
        self.end_write(frame)

    def visit_FilterBlock(self, node: nodes.FilterBlock, frame: Frame) -> None:
        filter_frame = frame.inner()
        filter_frame.symbols.analyze_node(node)
        self.enter_frame(filter_frame)
        self.buffer(filter_frame)
        self.blockvisit(node.body, filter_frame)
        self.start_write(frame, node)
        self.visit_Filter(node.filter, filter_frame)
        self.end_write(frame)
        self.leave_frame(filter_frame)

    def visit_With(self, node: nodes.With, frame: Frame) -> None:
        with_frame = frame.inner()
        with_frame.symbols.analyze_node(node)
        self.enter_frame(with_frame)
        for target, expr in zip(node.targets, node.values):
            self.newline()
            self.visit(target, with_frame)
            self.write(" = ")
            self.visit(expr, frame)
        self.blockvisit(node.body, with_frame)
        self.leave_frame(with_frame)

    def visit_ExprStmt(self, node: nodes.ExprStmt, frame: Frame) -> None:
        self.newline(node)
        self.visit(node.node, frame)

    class _FinalizeInfo(t.NamedTuple):
        const: t.Optional[t.Callable[..., str]]
        src: t.Optional[str]

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        """The default finalize function if the environment isn't
        configured with one. Or, if the environment has one, this is
        called on that function's output for constants.
        """
        return str(value)

    _finalize: t.Optional[_FinalizeInfo] = None

    def _make_finalize(self) -> _FinalizeInfo:
        """Build the finalize function to be used on constants and at
        runtime. Cached so it's only created once for all output nodes.

        Returns a ``namedtuple`` with the following attributes:

        ``const``
            A function to finalize constant data at compile time.

        ``src``
            Source code to output around nodes to be evaluated at
            runtime.
        """
        if self._finalize is not None:
            return self._finalize

        finalize: t.Optional[t.Callable[..., t.Any]]
        finalize = default = self._default_finalize
        src = None

        if self.environment.finalize:
            src = "environment.finalize("
            env_finalize = self.environment.finalize
            pass_arg = {
                _PassArg.context: "context",
                _PassArg.eval_context: "context.eval_ctx",
                _PassArg.environment: "environment",
            }.get(
                _PassArg.from_obj(env_finalize)  # type: ignore
            )
            finalize = None

            if pass_arg is None:

                def finalize(value: t.Any) -> t.Any:  # noqa: F811
                    return default(env_finalize(value))

            else:
                src = f"{src}{pass_arg}, "

                if pass_arg == "environment":

                    def finalize(value: t.Any) -> t.Any:  # noqa: F811
                        return default(env_finalize(self.environment, value))

        self._finalize = self._FinalizeInfo(finalize, src)
        return self._finalize

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        """Given a group of constant values converted from ``Output``
        child nodes, produce a string to write to the template module
        source.
        """
        return repr(concat(group))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> str:
        """Try to optimize a child of an ``Output`` node by trying to
        convert it to constant, finalized data at compile time.

        If :exc:`Impossible` is raised, the node is not constant and
        will be evaluated at runtime. Any other exception will also be
        evaluated at runtime for easier debugging.
        """
        const = node.as_const(frame.eval_ctx)

        if frame.eval_ctx.autoescape:
            const = escape(const)

        # Template data doesn't go through finalize.
        if isinstance(node, nodes.TemplateData):
            return str(const)

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code before visiting a child of an
        ``Output`` node.
        """
        if frame.eval_ctx.volatile:
            self.write("(escape if context.eval_ctx.autoescape else str)(")
        elif frame.eval_ctx.autoescape:
            self.write("escape(")
        else:
            self.write("str(")

        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code after visiting a child of an
        ``Output`` node.
        """
        self.write(")")

        if finalize.src is not None:
            self.write(")")

    def visit_Output(self, node: nodes.Output, frame: Frame) -> None:
        # If an extends is active, don't render outside a block.
        if frame.require_output_check:
            # A top-level extends is known to exist at compile time.
            if self.has_known_extends:
                return

            self.writeline("if parent_template is None:")
            self.indent()

        finalize = self._make_finalize()
        body: t.List[t.Union[t.List[t.Any], nodes.Expr]] = []

        # Evaluate constants at compile time if possible. Each item in
        # body will be either a list of static data or a node to be
        # evaluated at runtime.
        for child in node.nodes:
            try:
                if not (
                    # If the finalize function requires runtime context,
                    # constants can't be evaluated at compile time.
                    finalize.const
                    # Unless it's basic template data that won't be
                    # finalized anyway.
                    or isinstance(child, nodes.TemplateData)
                ):
                    raise nodes.Impossible()

                const = self._output_child_to_const(child, frame, finalize)
            except (nodes.Impossible, Exception):
                # The node was not constant and needs to be evaluated at
                # runtime. Or another error was raised, which is easier
                # to debug at runtime.
                body.append(child)
                continue

            if body and isinstance(body[-1], list):
                body[-1].append(const)
            else:
                body.append([const])

        if frame.buffer is not None:
            if len(body) == 1:
                self.writeline(f"{frame.buffer}.append(")
            else:
                self.writeline(f"{frame.buffer}.extend((")

            self.indent()

        for item in body:
            if isinstance(item, list):
                # A group of constant data to join and output.
                val = self._output_const_repr(item)

                if frame.buffer is None:
                    self.writeline("yield " + val)
                else:
                    self.writeline(val + ",")
            else:
                if frame.buffer is None:
                    self.writeline("yield ", item)
                else:
                    self.newline(item)

                # A node to be evaluated at runtime.
                self._output_child_pre(item, frame, finalize)
                self.visit(item, frame)
                self._output_child_post(item, frame, finalize)

                if frame.buffer is not None:
                    self.write(",")

        if frame.buffer is not None:
            self.outdent()
            self.writeline(")" if len(body) == 1 else "))")

        if frame.require_output_check:
            self.outdent()

    def visit_Assign(self, node: nodes.Assign, frame: Frame) -> None:
        self.push_assign_tracking()

        # ``a.b`` is allowed for assignment, and is parsed as an NSRef. However,
        # it is only valid if it references a Namespace object. Emit a check for
        # that for each ref here, before assignment code is emitted. This can't
        # be done in visit_NSRef as the ref could be in the middle of a tuple.
        seen_refs: t.Set[str] = set()

        for nsref in node.find_all(nodes.NSRef):
            if nsref.name in seen_refs:
                # Only emit the check for each reference once, in case the same
                # ref is used multiple times in a tuple, `ns.a, ns.b = c, d`.
                continue

            seen_refs.add(nsref.name)
            ref = frame.symbols.ref(nsref.name)
            self.writeline(f"if not isinstance({ref}, Namespace):")
            self.indent()
            self.writeline(
                "raise TemplateRuntimeError"
                '("cannot assign attribute on non-namespace object")'
            )
            self.outdent()

        self.newline(node)
        self.visit(node.target, frame)
        self.write(" = ")
        self.visit(node.node, frame)
        self.pop_assign_tracking(frame)

    def visit_AssignBlock(self, node: nodes.AssignBlock, frame: Frame) -> None:
        self.push_assign_tracking()
        block_frame = frame.inner()
        # This is a special case.  Since a set block always captures we
        # will disable output checks.  This way one can use set blocks
        # toplevel even in extended templates.
        block_frame.require_output_check = False
        block_frame.symbols.analyze_node(node)
        self.enter_frame(block_frame)
        self.buffer(block_frame)
        self.blockvisit(node.body, block_frame)
        self.newline(node)
        self.visit(node.target, frame)
        self.write(" = (Markup if context.eval_ctx.autoescape else identity)(")
        if node.filter is not None:
            self.visit_Filter(node.filter, block_frame)
        else:
            self.write(f"concat({block_frame.buffer})")
        self.write(")")
        self.pop_assign_tracking(frame)
        self.leave_frame(block_frame)

    # -- Expression Visitors

    def visit_Name(self, node: nodes.Name, frame: Frame) -> None:
        if node.ctx == "store" and (
            frame.toplevel or frame.loop_frame or frame.block_frame
        ):
            if self._assign_stack:
                self._assign_stack[-1].add(node.name)
        ref = frame.symbols.ref(node.name)

        # If we are looking up a variable we might have to deal with the
        # case where it's undefined.  We can skip that case if the load
        # instruction indicates a parameter which are always defined.
        if node.ctx == "load":
            load = frame.symbols.find_load(ref)
            if not (
                load is not None
                and load[0] == VAR_LOAD_PARAMETER
                and not self.parameter_is_undeclared(ref)
            ):
                self.write(
                    f"(undefined(name={node.name!r}) if {ref} is missing else {ref})"
                )
                return

        self.write(ref)

    def visit_NSRef(self, node: nodes.NSRef, frame: Frame) -> None:
        # NSRef is a dotted assignment target a.b=c, but uses a[b]=c internally.
        # visit_Assign emits code to validate that each ref is to a Namespace
        # object only. That can't be emitted here as the ref could be in the
        # middle of a tuple assignment.
        ref = frame.symbols.ref(node.name)
        self.writeline(f"{ref}[{node.attr!r}]")

    def visit_Const(self, node: nodes.Const, frame: Frame) -> None:
        val = node.as_const(frame.eval_ctx)
        if isinstance(val, float):
            self.write(str(val))
        else:
            self.write(repr(val))

    def visit_TemplateData(self, node: nodes.TemplateData, frame: Frame) -> None:
        try:
            self.write(repr(node.as_const(frame.eval_ctx)))
        except nodes.Impossible:
            self.write(
                f"(Markup if context.eval_ctx.autoescape else identity)({node.data!r})"
            )

    def visit_Tuple(self, node: nodes.Tuple, frame: Frame) -> None:
        self.write("(")
        idx = -1
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item, frame)
        self.write(",)" if idx == 0 else ")")

    def visit_List(self, node: nodes.List, frame: Frame) -> None:
        self.write("[")
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item, frame)
        self.write("]")

    def visit_Dict(self, node: nodes.Dict, frame: Frame) -> None:
        self.write("{")
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item.key, frame)
            self.write(": ")
            self.visit(item.value, frame)
        self.write("}")

    visit_Add = _make_binop("+")
    visit_Sub = _make_binop("-")
    visit_Mul = _make_binop("*")
    visit_Div = _make_binop("/")
    visit_FloorDiv = _make_binop("//")
    visit_Pow = _make_binop("**")
    visit_Mod = _make_binop("%")
    visit_And = _make_binop("and")
    visit_Or = _make_binop("or")
    visit_Pos = _make_unop("+")
    visit_Neg = _make_unop("-")
    visit_Not = _make_unop("not ")

    @optimizeconst
    def visit_Concat(self, node: nodes.Concat, frame: Frame) -> None:
        if frame.eval_ctx.volatile:
            func_name = "(markup_join if context.eval_ctx.volatile else str_join)"
        elif frame.eval_ctx.autoescape:
            func_name = "markup_join"
        else:
            func_name = "str_join"
        self.write(f"{func_name}((")
        for arg in node.nodes:
            self.visit(arg, frame)
            self.write(", ")
        self.write("))")

    @optimizeconst
    def visit_Compare(self, node: nodes.Compare, frame: Frame) -> None:
        self.write("(")
        self.visit(node.expr, frame)
        for op in node.ops:
            self.visit(op, frame)
        self.write(")")

    def visit_Operand(self, node: nodes.Operand, frame: Frame) -> None:
        self.write(f" {operators[node.op]} ")
        self.visit(node.expr, frame)

    @optimizeconst
    def visit_Getattr(self, node: nodes.Getattr, frame: Frame) -> None:
        if self.environment.is_async:
            self.write("(await auto_await(")

        self.write("environment.getattr(")
        self.visit(node.node, frame)
        self.write(f", {node.attr!r})")

        if self.environment.is_async:
            self.write("))")

    @optimizeconst
    def visit_Getitem(self, node: nodes.Getitem, frame: Frame) -> None:
        # slices bypass the environment getitem method.
        if isinstance(node.arg, nodes.Slice):
            self.visit(node.node, frame)
            self.write("[")
            self.visit(node.arg, frame)
            self.write("]")
        else:
            if self.environment.is_async:
                self.write("(await auto_await(")

            self.write("environment.getitem(")
            self.visit(node.node, frame)
            self.write(", ")
            self.visit(node.arg, frame)
            self.write(")")

            if self.environment.is_async:
                self.write("))")

    def visit_Slice(self, node: nodes.Slice, frame: Frame) -> None:
        if node.start is not None:
            self.visit(node.start, frame)
        self.write(":")
        if node.stop is not None:
            self.visit(node.stop, frame)
        if node.step is not None:
            self.write(":")
            self.visit(node.step, frame)

    @contextmanager
    def _filter_test_common(
        self, node: t.Union[nodes.Filter, nodes.Test], frame: Frame, is_filter: bool
    ) -> t.Iterator[None]:
        if self.environment.is_async:
            self.write("(await auto_await(")

        if is_filter:
            self.write(f"{self.filters[node.name]}(")
            func = self.environment.filters.get(node.name)
        else:
            self.write(f"{self.tests[node.name]}(")
            func = self.environment.tests.get(node.name)

        # When inside an If or CondExpr frame, allow the filter to be
        # undefined at compile time and only raise an error if it's
        # actually called at runtime. See pull_dependencies.
        if func is None and not frame.soft_frame:
            type_name = "filter" if is_filter else "test"
            self.fail(f"No {type_name} named {node.name!r}.", node.lineno)

        pass_arg = {
            _PassArg.context: "context",
            _PassArg.eval_context: "context.eval_ctx",
            _PassArg.environment: "environment",
        }.get(
            _PassArg.from_obj(func)  # type: ignore
        )

        if pass_arg is not None:
            self.write(f"{pass_arg}, ")

        # Back to the visitor function to handle visiting the target of
        # the filter or test.
        yield

        self.signature(node, frame)
        self.write(")")

        if self.environment.is_async:
            self.write("))")

    @optimizeconst
    def visit_Filter(self, node: nodes.Filter, frame: Frame) -> None:
        with self._filter_test_common(node, frame, True):
            # if the filter node is None we are inside a filter block
            # and want to write to the current buffer
            if node.node is not None:
                self.visit(node.node, frame)
            elif frame.eval_ctx.volatile:
                self.write(
                    f"(Markup(concat({frame.buffer}))"
                    f" if context.eval_ctx.autoescape else concat({frame.buffer}))"
                )
            elif frame.eval_ctx.autoescape:
                self.write(f"Markup(concat({frame.buffer}))")
            else:
                self.write(f"concat({frame.buffer})")

    @optimizeconst
    def visit_Test(self, node: nodes.Test, frame: Frame) -> None:
        with self._filter_test_common(node, frame, False):
            self.visit(node.node, frame)

    @optimizeconst
    def visit_CondExpr(self, node: nodes.CondExpr, frame: Frame) -> None:
        frame = frame.soft()

        def write_expr2() -> None:
            if node.expr2 is not None:
                self.visit(node.expr2, frame)
                return

            self.write(
                f'cond_expr_undefined("the inline if-expression on'
                f" {self.position(node)} evaluated to false and no else"
                f' section was defined.")'
            )

        self.write("(")
        self.visit(node.expr1, frame)
        self.write(" if ")
        self.visit(node.test, frame)
        self.write(" else ")
        write_expr2()
        self.write(")")

    @optimizeconst
    def visit_Call(
        self, node: nodes.Call, frame: Frame, forward_caller: bool = False
    ) -> None:
        if self.environment.is_async:
            self.write("(await auto_await(")
        if self.environment.sandboxed:
            self.write("environment.call(context, ")
        else:
            self.write("context.call(")
        self.visit(node.node, frame)
        extra_kwargs = {"caller": "caller"} if forward_caller else None
        loop_kwargs = {"_loop_vars": "_loop_vars"} if frame.loop_frame else {}
        block_kwargs = {"_block_vars": "_block_vars"} if frame.block_frame else {}
        if extra_kwargs:
            extra_kwargs.update(loop_kwargs, **block_kwargs)
        elif loop_kwargs or block_kwargs:
            extra_kwargs = dict(loop_kwargs, **block_kwargs)
        self.signature(node, frame, extra_kwargs)
        self.write(")")
        if self.environment.is_async:
            self.write("))")

    def visit_Keyword(self, node: nodes.Keyword, frame: Frame) -> None:
        self.write(node.key + "=")
        self.visit(node.value, frame)

    # -- Unused nodes for extensions

    def visit_MarkSafe(self, node: nodes.MarkSafe, frame: Frame) -> None:
        self.write("Markup(")
        self.visit(node.expr, frame)
        self.write(")")

    def visit_MarkSafeIfAutoescape(
        self, node: nodes.MarkSafeIfAutoescape, frame: Frame
    ) -> None:
        self.write("(Markup if context.eval_ctx.autoescape else identity)(")
        self.visit(node.expr, frame)
        self.write(")")

    def visit_EnvironmentAttribute(
        self, node: nodes.EnvironmentAttribute, frame: Frame
    ) -> None:
        self.write("environment." + node.name)

    def visit_ExtensionAttribute(
        self, node: nodes.ExtensionAttribute, frame: Frame
    ) -> None:
        self.write(f"environment.extensions[{node.identifier!r}].{node.name}")

    def visit_ImportedName(self, node: nodes.ImportedName, frame: Frame) -> None:
        self.write(self.import_aliases[node.importname])

    def visit_InternalName(self, node: nodes.InternalName, frame: Frame) -> None:
        self.write(node.name)

    def visit_ContextReference(
        self, node: nodes.ContextReference, frame: Frame
    ) -> None:
        self.write("context")

    def visit_DerivedContextReference(
        self, node: nodes.DerivedContextReference, frame: Frame
    ) -> None:
        self.write(self.derive_context(frame))

    def visit_Continue(self, node: nodes.Continue, frame: Frame) -> None:
        self.writeline("continue", node)

    def visit_Break(self, node: nodes.Break, frame: Frame) -> None:
        self.writeline("break", node)

    def visit_Scope(self, node: nodes.Scope, frame: Frame) -> None:
        scope_frame = frame.inner()
        scope_frame.symbols.analyze_node(node)
        self.enter_frame(scope_frame)
        self.blockvisit(node.body, scope_frame)
        self.leave_frame(scope_frame)

    def visit_OverlayScope(self, node: nodes.OverlayScope, frame: Frame) -> None:
        ctx = self.temporary_identifier()
        self.writeline(f"{ctx} = {self.derive_context(frame)}")
        self.writeline(f"{ctx}.vars = ")
        self.visit(node.context, frame)
        self.push_context_reference(ctx)

        scope_frame = frame.inner(isolated=True)
        scope_frame.symbols.analyze_node(node)
        self.enter_frame(scope_frame)
        self.blockvisit(node.body, scope_frame)
        self.leave_frame(scope_frame)
        self.pop_context_reference()

    def visit_EvalContextModifier(
        self, node: nodes.EvalContextModifier, frame: Frame
    ) -> None:
        for keyword in node.options:
            self.writeline(f"context.eval_ctx.{keyword.key} = ")
            self.visit(keyword.value, frame)
            try:
                val = keyword.value.as_const(frame.eval_ctx)
            except nodes.Impossible:
                frame.eval_ctx.volatile = True
            else:
                setattr(frame.eval_ctx, keyword.key, val)

    def visit_ScopedEvalContextModifier(
        self, node: nodes.ScopedEvalContextModifier, frame: Frame
    ) -> None:
        old_ctx_name = self.temporary_identifier()
        saved_ctx = frame.eval_ctx.save()
        self.writeline(f"{old_ctx_name} = context.eval_ctx.save()")
        self.visit_EvalContextModifier(node, frame)
        for child in node.body:
            self.visit(child, frame)
        frame.eval_ctx.revert(saved_ctx)
        self.writeline(f"context.eval_ctx.revert({old_ctx_name})")
<|endoftext|>"""Classes for managing templates and their runtime and compile time
options.
"""

import os
import typing
import typing as t
import weakref
from collections import ChainMap
from functools import lru_cache
from functools import partial
from functools import reduce
from types import CodeType

from markupsafe import Markup

from . import nodes
from .compiler import CodeGenerator
from .compiler import generate
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS  # type: ignore[attr-defined]
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS  # type: ignore[attr-defined]
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .exceptions import TemplateNotFound
from .exceptions import TemplateRuntimeError
from .exceptions import TemplatesNotFound
from .exceptions import TemplateSyntaxError
from .exceptions import UndefinedError
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .parser import Parser
from .runtime import Context
from .runtime import new_context
from .runtime import Undefined
from .utils import _PassArg
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import LRUCache
from .utils import missing

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .bccache import BytecodeCache
    from .ext import Extension
    from .loaders import BaseLoader

_env_bound = t.TypeVar("_env_bound", bound="Environment")


# for direct template usage we have up to ten living environments
@lru_cache(maxsize=10)
def get_spontaneous_environment(cls: t.Type[_env_bound], *args: t.Any) -> _env_bound:
    """Return a new spontaneous environment. A spontaneous environment
    is used for templates created directly rather than through an
    existing environment.

    :param cls: Environment class to create.
    :param args: Positional arguments passed to environment.
    """
    env = cls(*args)
    env.shared = True
    return env


def create_cache(
    size: int,
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Return the cache class for the given size."""
    if size == 0:
        return None

    if size < 0:
        return {}

    return LRUCache(size)  # type: ignore


def copy_cache(
    cache: t.Optional[t.MutableMapping[t.Any, t.Any]],
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Create an empty copy of the given cache."""
    if cache is None:
        return None

    if type(cache) is dict:  # noqa E721
        return {}

    return LRUCache(cache.capacity)  # type: ignore


def load_extensions(
    environment: "Environment",
    extensions: t.Sequence[t.Union[str, t.Type["Extension"]]],
) -> t.Dict[str, "Extension"]:
    """Load the extensions from the list and bind it to the environment.
    Returns a dict of instantiated extensions.
    """
    result = {}

    for extension in extensions:
        if isinstance(extension, str):
            extension = t.cast(t.Type["Extension"], import_string(extension))

        result[extension.identifier] = extension(environment)

    return result


def _environment_config_check(environment: _env_bound) -> _env_bound:
    """Perform a sanity check on the environment."""
    assert issubclass(
        environment.undefined, Undefined
    ), "'undefined' must be a subclass of 'jinja2.Undefined'."
    assert (
        environment.block_start_string
        != environment.variable_start_string
        != environment.comment_start_string
    ), "block, variable and comment start strings must be different."
    assert environment.newline_sequence in {
        "\r",
        "\r\n",
        "\n",
    }, "'newline_sequence' must be one of '\\n', '\\r\\n', or '\\r'."
    return environment


class Environment:
    r"""The core component of Jinja is the `Environment`.  It contains
    important shared variables like configuration, filters, tests,
    globals and others.  Instances of this class may be modified if
    they are not shared and if no template was loaded so far.
    Modifications on environments after the first template was loaded
    will lead to surprising effects and undefined behavior.

    Here are the possible initialization parameters:

        `block_start_string`
            The string marking the beginning of a block.  Defaults to ``'{%'``.

        `block_end_string`
            The string marking the end of a block.  Defaults to ``'%}'``.

        `variable_start_string`
            The string marking the beginning of a print statement.
            Defaults to ``'{{'``.

        `variable_end_string`
            The string marking the end of a print statement.  Defaults to
            ``'}}'``.

        `comment_start_string`
            The string marking the beginning of a comment.  Defaults to ``'{#'``.

        `comment_end_string`
            The string marking the end of a comment.  Defaults to ``'#}'``.

        `line_statement_prefix`
            If given and a string, this will be used as prefix for line based
            statements.  See also :ref:`line-statements`.

        `line_comment_prefix`
            If given and a string, this will be used as prefix for line based
            comments.  See also :ref:`line-statements`.

            .. versionadded:: 2.2

        `trim_blocks`
            If this is set to ``True`` the first newline after a block is
            removed (block, not variable tag!).  Defaults to `False`.

        `lstrip_blocks`
            If this is set to ``True`` leading spaces and tabs are stripped
            from the start of a line to a block.  Defaults to `False`.

        `newline_sequence`
            The sequence that starts a newline.  Must be one of ``'\r'``,
            ``'\n'`` or ``'\r\n'``.  The default is ``'\n'`` which is a
            useful default for Linux and OS X systems as well as web
            applications.

        `keep_trailing_newline`
            Preserve the trailing newline when rendering templates.
            The default is ``False``, which causes a single newline,
            if present, to be stripped from the end of the template.

            .. versionadded:: 2.7

        `extensions`
            List of Jinja extensions to use.  This can either be import paths
            as strings or extension classes.  For more information have a
            look at :ref:`the extensions documentation <jinja-extensions>`.

        `optimized`
            should the optimizer be enabled?  Default is ``True``.

        `undefined`
            :class:`Undefined` or a subclass of it that is used to represent
            undefined values in the template.

        `finalize`
            A callable that can be used to process the result of a variable
            expression before it is output.  For example one can convert
            ``None`` implicitly into an empty string here.

        `autoescape`
            If set to ``True`` the XML/HTML autoescaping feature is enabled by
            default.  For more details about autoescaping see
            :class:`~markupsafe.Markup`.  As of Jinja 2.4 this can also
            be a callable that is passed the template name and has to
            return ``True`` or ``False`` depending on autoescape should be
            enabled by default.

            .. versionchanged:: 2.4
               `autoescape` can now be a function

        `loader`
            The template loader for this environment.

        `cache_size`
            The size of the cache.  Per default this is ``400`` which means
            that if more than 400 templates are loaded the loader will clean
            out the least recently used template.  If the cache size is set to
            ``0`` templates are recompiled all the time, if the cache size is
            ``-1`` the cache will not be cleaned.

            .. versionchanged:: 2.8
               The cache size was increased to 400 from a low 50.

        `auto_reload`
            Some loaders load templates from locations where the template
            sources may change (ie: file system or database).  If
            ``auto_reload`` is set to ``True`` (default) every time a template is
            requested the loader checks if the source changed and if yes, it
            will reload the template.  For higher performance it's possible to
            disable that.

        `bytecode_cache`
            If set to a bytecode cache object, this object will provide a
            cache for the internal Jinja bytecode so that templates don't
            have to be parsed if they were not changed.

            See :ref:`bytecode-cache` for more information.

        `enable_async`
            If set to true this enables async template execution which
            allows using async functions and generators.
    """

    #: if this environment is sandboxed.  Modifying this variable won't make
    #: the environment sandboxed though.  For a real sandboxed environment
    #: have a look at jinja2.sandbox.  This flag alone controls the code
    #: generation by the compiler.
    sandboxed = False

    #: True if the environment is just an overlay
    overlayed = False

    #: the environment this environment is linked to if it is an overlay
    linked_to: t.Optional["Environment"] = None

    #: shared environments have this set to `True`.  A shared environment
    #: must not be modified
    shared = False

    #: the class that is used for code generation.  See
    #: :class:`~jinja2.compiler.CodeGenerator` for more information.
    code_generator_class: t.Type["CodeGenerator"] = CodeGenerator

    concat = "".join

    #: the context class that is used for templates.  See
    #: :class:`~jinja2.runtime.Context` for more information.
    context_class: t.Type[Context] = Context

    template_class: t.Type["Template"]

    def __init__(
        self,
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        loader: t.Optional["BaseLoader"] = None,
        cache_size: int = 400,
        auto_reload: bool = True,
        bytecode_cache: t.Optional["BytecodeCache"] = None,
        enable_async: bool = False,
    ):
        # !!Important notice!!
        #   The constructor accepts quite a few arguments that should be
        #   passed by keyword rather than position.  However it's important to
        #   not change the order of arguments because it's used at least
        #   internally in those cases:
        #       -   spontaneous environments (i18n extension and Template)
        #       -   unittests
        #   If parameter changes are required only add parameters at the end
        #   and don't change the arguments (or the defaults!) of the arguments
        #   existing already.

        # lexer / parser information
        self.block_start_string = block_start_string
        self.block_end_string = block_end_string
        self.variable_start_string = variable_start_string
        self.variable_end_string = variable_end_string
        self.comment_start_string = comment_start_string
        self.comment_end_string = comment_end_string
        self.line_statement_prefix = line_statement_prefix
        self.line_comment_prefix = line_comment_prefix
        self.trim_blocks = trim_blocks
        self.lstrip_blocks = lstrip_blocks
        self.newline_sequence = newline_sequence
        self.keep_trailing_newline = keep_trailing_newline

        # runtime information
        self.undefined: t.Type[Undefined] = undefined
        self.optimized = optimized
        self.finalize = finalize
        self.autoescape = autoescape

        # defaults
        self.filters = DEFAULT_FILTERS.copy()
        self.tests = DEFAULT_TESTS.copy()
        self.globals = DEFAULT_NAMESPACE.copy()

        # set the loader provided
        self.loader = loader
        self.cache = create_cache(cache_size)
        self.bytecode_cache = bytecode_cache
        self.auto_reload = auto_reload

        # configurable policies
        self.policies = DEFAULT_POLICIES.copy()

        # load extensions
        self.extensions = load_extensions(self, extensions)

        self.is_async = enable_async
        _environment_config_check(self)

    def add_extension(self, extension: t.Union[str, t.Type["Extension"]]) -> None:
        """Adds an extension after the environment was created.

        .. versionadded:: 2.5
        """
        self.extensions.update(load_extensions(self, [extension]))

    def extend(self, **attributes: t.Any) -> None:
        """Add the items to the instance of the environment if they do not exist
        yet.  This is used by :ref:`extensions <writing-extensions>` to register
        callbacks and configuration values without breaking inheritance.
        """
        for key, value in attributes.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def overlay(
        self,
        block_start_string: str = missing,
        block_end_string: str = missing,
        variable_start_string: str = missing,
        variable_end_string: str = missing,
        comment_start_string: str = missing,
        comment_end_string: str = missing,
        line_statement_prefix: t.Optional[str] = missing,
        line_comment_prefix: t.Optional[str] = missing,
        trim_blocks: bool = missing,
        lstrip_blocks: bool = missing,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = missing,
        keep_trailing_newline: bool = missing,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = missing,
        optimized: bool = missing,
        undefined: t.Type[Undefined] = missing,
        finalize: t.Optional[t.Callable[..., t.Any]] = missing,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = missing,
        loader: t.Optional["BaseLoader"] = missing,
        cache_size: int = missing,
        auto_reload: bool = missing,
        bytecode_cache: t.Optional["BytecodeCache"] = missing,
        enable_async: bool = missing,
    ) -> "te.Self":
        """Create a new overlay environment that shares all the data with the
        current environment except for cache and the overridden attributes.
        Extensions cannot be removed for an overlayed environment.  An overlayed
        environment automatically gets all the extensions of the environment it
        is linked to plus optional extra extensions.

        Creating overlays should happen after the initial environment was set
        up completely.  Not all attributes are truly linked, some are just
        copied over so modifications on the original environment may not shine
        through.

        .. versionchanged:: 3.1.5
            ``enable_async`` is applied correctly.

        .. versionchanged:: 3.1.2
            Added the ``newline_sequence``, ``keep_trailing_newline``,
            and ``enable_async`` parameters to match ``__init__``.
        """
        args = dict(locals())
        del args["self"], args["cache_size"], args["extensions"], args["enable_async"]

        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.overlayed = True
        rv.linked_to = self

        for key, value in args.items():
            if value is not missing:
                setattr(rv, key, value)

        if cache_size is not missing:
            rv.cache = create_cache(cache_size)
        else:
            rv.cache = copy_cache(self.cache)

        rv.extensions = {}
        for key, value in self.extensions.items():
            rv.extensions[key] = value.bind(rv)
        if extensions is not missing:
            rv.extensions.update(load_extensions(rv, extensions))

        if enable_async is not missing:
            rv.is_async = enable_async

        return _environment_config_check(rv)

    @property
    def lexer(self) -> Lexer:
        """The lexer for this environment."""
        return get_lexer(self)

    def iter_extensions(self) -> t.Iterator["Extension"]:
        """Iterates over the extensions by priority."""
        return iter(sorted(self.extensions.values(), key=lambda x: x.priority))

    def getitem(
        self, obj: t.Any, argument: t.Union[str, t.Any]
    ) -> t.Union[t.Any, Undefined]:
        """Get an item or attribute of an object but prefer the item."""
        try:
            return obj[argument]
        except (AttributeError, TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        return getattr(obj, attr)
                    except AttributeError:
                        pass
            return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Any:
        """Get an item or attribute of an object but prefer the attribute.
        Unlike :meth:`getitem` the attribute *must* be a string.
        """
        try:
            return getattr(obj, attribute)
        except AttributeError:
            pass
        try:
            return obj[attribute]
        except (TypeError, LookupError, AttributeError):
            return self.undefined(obj=obj, name=attribute)

    def _filter_test_common(
        self,
        name: t.Union[str, Undefined],
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]],
        kwargs: t.Optional[t.Mapping[str, t.Any]],
        context: t.Optional[Context],
        eval_ctx: t.Optional[EvalContext],
        is_filter: bool,
    ) -> t.Any:
        if is_filter:
            env_map = self.filters
            type_name = "filter"
        else:
            env_map = self.tests
            type_name = "test"

        func = env_map.get(name)  # type: ignore

        if func is None:
            msg = f"No {type_name} named {name!r}."

            if isinstance(name, Undefined):
                try:
                    name._fail_with_undefined_error()
                except Exception as e:
                    msg = f"{msg} ({e}; did you forget to quote the callable name?)"

            raise TemplateRuntimeError(msg)

        args = [value, *(args if args is not None else ())]
        kwargs = kwargs if kwargs is not None else {}
        pass_arg = _PassArg.from_obj(func)

        if pass_arg is _PassArg.context:
            if context is None:
                raise TemplateRuntimeError(
                    f"Attempted to invoke a context {type_name} without context."
                )

            args.insert(0, context)
        elif pass_arg is _PassArg.eval_context:
            if eval_ctx is None:
                if context is not None:
                    eval_ctx = context.eval_ctx
                else:
                    eval_ctx = EvalContext(self)

            args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            args.insert(0, self)

        return func(*args, **kwargs)

    def call_filter(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a filter on a value the same way the compiler does.

        This might return a coroutine if the filter is running from an
        environment in async mode and the filter supports async
        execution. It's your responsibility to await this if needed.

        .. versionadded:: 2.7
        """
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, True
        )

    def call_test(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a test on a value the same way the compiler does.

        This might return a coroutine if the test is running from an
        environment in async mode and the test supports async execution.
        It's your responsibility to await this if needed.

        .. versionchanged:: 3.0
            Tests support ``@pass_context``, etc. decorators. Added
            the ``context`` and ``eval_ctx`` parameters.

        .. versionadded:: 2.7
        """
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, False
        )

    @internalcode
    def parse(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> nodes.Template:
        """Parse the sourcecode and return the abstract syntax tree.  This
        tree of nodes is used by the compiler to convert the template into
        executable source- or bytecode.  This is useful for debugging or to
        extract information from templates.

        If you are :ref:`developing Jinja extensions <writing-extensions>`
        this gives you a good overview of the node tree generated.
        """
        try:
            return self._parse(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def _parse(
        self, source: str, name: t.Optional[str], filename: t.Optional[str]
    ) -> nodes.Template:
        """Internal parsing function used by `parse` and `compile`."""
        return Parser(self, source, name, filename).parse()

    def lex(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> t.Iterator[t.Tuple[int, str, str]]:
        """Lex the given sourcecode and return a generator that yields
        tokens as tuples in the form ``(lineno, token_type, value)``.
        This can be useful for :ref:`extension development <writing-extensions>`
        and debugging templates.

        This does not perform preprocessing.  If you want the preprocessing
        of the extensions to be applied you have to filter source through
        the :meth:`preprocess` method.
        """
        source = str(source)
        try:
            return self.lexer.tokeniter(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def preprocess(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> str:
        """Preprocesses the source with all extensions.  This is automatically
        called for all parsing and compiling methods but *not* for :meth:`lex`
        because there you usually only want the actual source tokenized.
        """
        return reduce(
            lambda s, e: e.preprocess(s, name, filename),
            self.iter_extensions(),
            str(source),
        )

    def _tokenize(
        self,
        source: str,
        name: t.Optional[str],
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> TokenStream:
        """Called by the parser to do the preprocessing and filtering
        for all the extensions.  Returns a :class:`~jinja2.lexer.TokenStream`.
        """
        source = self.preprocess(source, name, filename)
        stream = self.lexer.tokenize(source, name, filename, state)

        for ext in self.iter_extensions():
            stream = ext.filter_stream(stream)  # type: ignore

            if not isinstance(stream, TokenStream):
                stream = TokenStream(stream, name, filename)

        return stream

    def _generate(
        self,
        source: nodes.Template,
        name: t.Optional[str],
        filename: t.Optional[str],
        defer_init: bool = False,
    ) -> str:
        """Internal hook that can be overridden to hook a different generate
        method in.

        .. versionadded:: 2.5
        """
        return generate(  # type: ignore
            source,
            self,
            name,
            filename,
            defer_init=defer_init,
            optimized=self.optimized,
        )

    def _compile(self, source: str, filename: str) -> CodeType:
        """Internal hook that can be overridden to hook a different compile
        method in.

        .. versionadded:: 2.5
        """
        return compile(source, filename, "exec")

    @typing.overload
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: "te.Literal[False]" = False,
        defer_init: bool = False,
    ) -> CodeType: ...

    @typing.overload
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: "te.Literal[True]" = ...,
        defer_init: bool = False,
    ) -> str: ...

    @internalcode
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: bool = False,
        defer_init: bool = False,
    ) -> t.Union[str, CodeType]:
        """Compile a node or template source code.  The `name` parameter is
        the load name of the template after it was joined using
        :meth:`join_path` if necessary, not the filename on the file system.
        the `filename` parameter is the estimated filename of the template on
        the file system.  If the template came from a database or memory this
        can be omitted.

        The return value of this method is a python code object.  If the `raw`
        parameter is `True` the return value will be a string with python
        code equivalent to the bytecode returned otherwise.  This method is
        mainly used internally.

        `defer_init` is use internally to aid the module code generator.  This
        causes the generated code to be able to import without the global
        environment variable to be set.

        .. versionadded:: 2.4
           `defer_init` parameter added.
        """
        source_hint = None
        try:
            if isinstance(source, str):
                source_hint = source
                source = self._parse(source, name, filename)
            source = self._generate(source, name, filename, defer_init=defer_init)
            if raw:
                return source
            if filename is None:
                filename = "<template>"
            return self._compile(source, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source_hint)

    def compile_expression(
        self, source: str, undefined_to_none: bool = True
    ) -> "TemplateExpression":
        """A handy helper method that returns a callable that accepts keyword
        arguments that appear as variables in the expression.  If called it
        returns the result of the expression.

        This is useful if applications want to use the same rules as Jinja
        in template "configuration files" or similar situations.

        Example usage:

        >>> env = Environment()
        >>> expr = env.compile_expression('foo == 42')
        >>> expr(foo=23)
        False
        >>> expr(foo=42)
        True

        Per default the return value is converted to `None` if the
        expression returns an undefined value.  This can be changed
        by setting `undefined_to_none` to `False`.

        >>> env.compile_expression('var')() is None
        True
        >>> env.compile_expression('var', undefined_to_none=False)()
        Undefined

        .. versionadded:: 2.1
        """
        parser = Parser(self, source, state="variable")
        try:
            expr = parser.parse_expression()
            if not parser.stream.eos:
                raise TemplateSyntaxError(
                    "chunk after expression", parser.stream.current.lineno, None, None
                )
            expr.set_environment(self)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

        body = [nodes.Assign(nodes.Name("result", "store"), expr, lineno=1)]
        template = self.from_string(nodes.Template(body, lineno=1))
        return TemplateExpression(template, undefined_to_none)

    def compile_templates(
        self,
        target: t.Union[str, "os.PathLike[str]"],
        extensions: t.Optional[t.Collection[str]] = None,
        filter_func: t.Optional[t.Callable[[str], bool]] = None,
        zip: t.Optional[str] = "deflated",
        log_function: t.Optional[t.Callable[[str], None]] = None,
        ignore_errors: bool = True,
    ) -> None:
        """Finds all the templates the loader can find, compiles them
        and stores them in `target`.  If `zip` is `None`, instead of in a
        zipfile, the templates will be stored in a directory.
        By default a deflate zip algorithm is used. To switch to
        the stored algorithm, `zip` can be set to ``'stored'``.

        `extensions` and `filter_func` are passed to :meth:`list_templates`.
        Each template returned will be compiled to the target folder or
        zipfile.

        By default template compilation errors are ignored.  In case a
        log function is provided, errors are logged.  If you want template
        syntax errors to abort the compilation you can set `ignore_errors`
        to `False` and you will get an exception on syntax errors.

        .. versionadded:: 2.4
        """
        from .loaders import ModuleLoader

        if log_function is None:

            def log_function(x: str) -> None:
                pass

        assert log_function is not None
        assert self.loader is not None, "No loader configured."

        def write_file(filename: str, data: str) -> None:
            if zip:
                info = ZipInfo(filename)
                info.external_attr = 0o755 << 16
                zip_file.writestr(info, data)
            else:
                with open(os.path.join(target, filename), "wb") as f:
                    f.write(data.encode("utf8"))

        if zip is not None:
            from zipfile import ZIP_DEFLATED
            from zipfile import ZIP_STORED
            from zipfile import ZipFile
            from zipfile import ZipInfo

            zip_file = ZipFile(
                target, "w", dict(deflated=ZIP_DEFLATED, stored=ZIP_STORED)[zip]
            )
            log_function(f"Compiling into Zip archive {target!r}")
        else:
            if not os.path.isdir(target):
                os.makedirs(target)
            log_function(f"Compiling into folder {target!r}")

        try:
            for name in self.list_templates(extensions, filter_func):
                source, filename, _ = self.loader.get_source(self, name)
                try:
                    code = self.compile(source, name, filename, True, True)
                except TemplateSyntaxError as e:
                    if not ignore_errors:
                        raise
                    log_function(f'Could not compile "{name}": {e}')
                    continue

                filename = ModuleLoader.get_module_filename(name)

                write_file(filename, code)
                log_function(f'Compiled "{name}" as {filename}')
        finally:
            if zip:
                zip_file.close()

        log_function("Finished compiling templates")

    def list_templates(
        self,
        extensions: t.Optional[t.Collection[str]] = None,
        filter_func: t.Optional[t.Callable[[str], bool]] = None,
    ) -> t.List[str]:
        """Returns a list of templates for this environment.  This requires
        that the loader supports the loader's
        :meth:`~BaseLoader.list_templates` method.

        If there are other files in the template folder besides the
        actual templates, the returned list can be filtered.  There are two
        ways: either `extensions` is set to a list of file extensions for
        templates, or a `filter_func` can be provided which is a callable that
        is passed a template name and should return `True` if it should end up
        in the result list.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        assert self.loader is not None, "No loader configured."
        names = self.loader.list_templates()

        if extensions is not None:
            if filter_func is not None:
                raise TypeError(
                    "either extensions or filter_func can be passed, but not both"
                )

            def filter_func(x: str) -> bool:
                return "." in x and x.rsplit(".", 1)[1] in extensions

        if filter_func is not None:
            names = [name for name in names if filter_func(name)]

        return names

    def handle_exception(self, source: t.Optional[str] = None) -> "te.NoReturn":
        """Exception handling helper.  This is used internally to either raise
        rewritten exceptions or return a rendered traceback for the template.
        """
        from .debug import rewrite_traceback_stack

        raise rewrite_traceback_stack(source=source)

    def join_path(self, template: str, parent: str) -> str:
        """Join a template with the parent.  By default all the lookups are
        relative to the loader root so this method returns the `template`
        parameter unchanged, but if the paths should be relative to the
        parent template, this function can be used to calculate the real
        template name.

        Subclasses may override this method and implement template path
        joining here.
        """
        return template

    @internalcode
    def _load_template(
        self, name: str, globals: t.Optional[t.MutableMapping[str, t.Any]]
    ) -> "Template":
        if self.loader is None:
            raise TypeError("no loader for this environment specified")
        cache_key = (weakref.ref(self.loader), name)
        if self.cache is not None:
            template = self.cache.get(cache_key)
            if template is not None and (
                not self.auto_reload or template.is_up_to_date
            ):
                # template.globals is a ChainMap, modifying it will only
                # affect the template, not the environment globals.
                if globals:
                    template.globals.update(globals)

                return template

        template = self.loader.load(self, name, self.make_globals(globals))

        if self.cache is not None:
            self.cache[cache_key] = template
        return template

    @internalcode
    def get_template(
        self,
        name: t.Union[str, "Template"],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Load a template by name with :attr:`loader` and return a
        :class:`Template`. If the template does not exist a
        :exc:`TemplateNotFound` exception is raised.

        :param name: Name of the template to load. When loading
            templates from the filesystem, "/" is used as the path
            separator, even on Windows.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.4
            If ``name`` is a :class:`Template` object it is returned
            unchanged.
        """
        if isinstance(name, Template):
            return name
        if parent is not None:
            name = self.join_path(name, parent)

        return self._load_template(name, globals)

    @internalcode
    def select_template(
        self,
        names: t.Iterable[t.Union[str, "Template"]],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Like :meth:`get_template`, but tries loading multiple names.
        If none of the names can be loaded a :exc:`TemplatesNotFound`
        exception is raised.

        :param names: List of template names to try loading in order.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.11
            If ``names`` is :class:`Undefined`, an :exc:`UndefinedError`
            is raised instead. If no templates were found and ``names``
            contains :class:`Undefined`, the message is more helpful.

        .. versionchanged:: 2.4
            If ``names`` contains a :class:`Template` object it is
            returned unchanged.

        .. versionadded:: 2.3
        """
        if isinstance(names, Undefined):
            names._fail_with_undefined_error()

        if not names:
            raise TemplatesNotFound(
                message="Tried to select from an empty list of templates."
            )

        for name in names:
            if isinstance(name, Template):
                return name
            if parent is not None:
                name = self.join_path(name, parent)
            try:
                return self._load_template(name, globals)
            except (TemplateNotFound, UndefinedError):
                pass
        raise TemplatesNotFound(names)  # type: ignore

    @internalcode
    def get_or_select_template(
        self,
        template_name_or_list: t.Union[
            str, "Template", t.List[t.Union[str, "Template"]]
        ],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Use :meth:`select_template` if an iterable of template names
        is given, or :meth:`get_template` if one name is given.

        .. versionadded:: 2.3
        """
        if isinstance(template_name_or_list, (str, Undefined)):
            return self.get_template(template_name_or_list, parent, globals)
        elif isinstance(template_name_or_list, Template):
            return template_name_or_list
        return self.select_template(template_name_or_list, parent, globals)

    def from_string(
        self,
        source: t.Union[str, nodes.Template],
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
        template_class: t.Optional[t.Type["Template"]] = None,
    ) -> "Template":
        """Load a template from a source string without using
        :attr:`loader`.

        :param source: Jinja source to compile into a template.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.
        :param template_class: Return an instance of this
            :class:`Template` class.
        """
        gs = self.make_globals(globals)
        cls = template_class or self.template_class
        return cls.from_code(self, self.compile(source), gs, None)

    def make_globals(
        self, d: t.Optional[t.MutableMapping[str, t.Any]]
    ) -> t.MutableMapping[str, t.Any]:
        """Make the globals map for a template. Any given template
        globals overlay the environment :attr:`globals`.

        Returns a :class:`collections.ChainMap`. This allows any changes
        to a template's globals to only affect that template, while
        changes to the environment's globals are still reflected.
        However, avoid modifying any globals after a template is loaded.

        :param d: Dict of template-specific globals.

        .. versionchanged:: 3.0
            Use :class:`collections.ChainMap` to always prevent mutating
            environment globals.
        """
        if d is None:
            d = {}

        return ChainMap(d, self.globals)


class Template:
    """A compiled template that can be rendered.

    Use the methods on :class:`Environment` to create or load templates.
    The environment is used to configure how templates are compiled and
    behave.

    It is also possible to create a template object directly. This is
    not usually recommended. The constructor takes most of the same
    arguments as :class:`Environment`. All templates created with the
    same environment arguments share the same ephemeral ``Environment``
    instance behind the scenes.

    A template object should be considered immutable. Modifications on
    the object are not supported.
    """

    #: Type of environment to create when creating a template directly
    #: rather than through an existing environment.
    environment_class: t.Type[Environment] = Environment

    environment: Environment
    globals: t.MutableMapping[str, t.Any]
    name: t.Optional[str]
    filename: t.Optional[str]
    blocks: t.Dict[str, t.Callable[[Context], t.Iterator[str]]]
    root_render_func: t.Callable[[Context], t.Iterator[str]]
    _module: t.Optional["TemplateModule"]
    _debug_info: str
    _uptodate: t.Optional[t.Callable[[], bool]]

    def __new__(
        cls,
        source: t.Union[str, nodes.Template],
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        enable_async: bool = False,
    ) -> t.Any:  # it returns a `Template`, but this breaks the sphinx build...
        env = get_spontaneous_environment(
            cls.environment_class,  # type: ignore
            block_start_string,
            block_end_string,
            variable_start_string,
            variable_end_string,
            comment_start_string,
            comment_end_string,
            line_statement_prefix,
            line_comment_prefix,
            trim_blocks,
            lstrip_blocks,
            newline_sequence,
            keep_trailing_newline,
            frozenset(extensions),
            optimized,
            undefined,  # type: ignore
            finalize,
            autoescape,
            None,
            0,
            False,
            None,
            enable_async,
        )
        return env.from_string(source, template_class=cls)

    @classmethod
    def from_code(
        cls,
        environment: Environment,
        code: CodeType,
        globals: t.MutableMapping[str, t.Any],
        uptodate: t.Optional[t.Callable[[], bool]] = None,
    ) -> "Template":
        """Creates a template object from compiled code and the globals.  This
        is used by the loaders and environment to create a template object.
        """
        namespace = {"environment": environment, "__file__": code.co_filename}
        exec(code, namespace)
        rv = cls._from_namespace(environment, namespace, globals)
        rv._uptodate = uptodate
        return rv

    @classmethod
    def from_module_dict(
        cls,
        environment: Environment,
        module_dict: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        """Creates a template object from a module.  This is used by the
        module loader to create a template object.

        .. versionadded:: 2.4
        """
        return cls._from_namespace(environment, module_dict, globals)

    @classmethod
    def _from_namespace(
        cls,
        environment: Environment,
        namespace: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        t: Template = object.__new__(cls)
        t.environment = environment
        t.globals = globals
        t.name = namespace["name"]
        t.filename = namespace["__file__"]
        t.blocks = namespace["blocks"]

        # render function and module
        t.root_render_func = namespace["root"]
        t._module = None

        # debug and loader helpers
        t._debug_info = namespace["debug_info"]
        t._uptodate = None

        # store the reference
        namespace["environment"] = environment
        namespace["__jinja_template__"] = t

        return t

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This method accepts the same arguments as the `dict` constructor:
        A dict, a dict subclass or some keyword arguments.  If no arguments
        are given the context will be empty.  These two calls do the same::

            template.render(knights='that say nih')
            template.render({'knights': 'that say nih'})

        This will return the rendered template as a string.
        """
        if self.environment.is_async:
            import asyncio

            return asyncio.run(self.render_async(*args, **kwargs))

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(self.root_render_func(ctx))  # type: ignore
        except Exception:
            self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This works similar to :meth:`render` but returns a coroutine
        that when awaited returns the entire rendered template string.  This
        requires the async feature to be enabled.

        Example usage::

            await template.render_async(knights='that say nih; asynchronously')
        """
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(  # type: ignore
                [n async for n in self.root_render_func(ctx)]  # type: ignore
            )
        except Exception:
            return self.environment.handle_exception()

    def stream(self, *args: t.Any, **kwargs: t.Any) -> "TemplateStream":
        """Works exactly like :meth:`generate` but returns a
        :class:`TemplateStream`.
        """
        return TemplateStream(self.generate(*args, **kwargs))

    def generate(self, *args: t.Any, **kwargs: t.Any) -> t.Iterator[str]:
        """For very large templates it can be useful to not render the whole
        template at once but evaluate each statement after another and yield
        piece for piece.  This method basically does exactly that and returns
        a generator that yields one item after another as strings.

        It accepts the same arguments as :meth:`render`.
        """
        if self.environment.is_async:
            import asyncio

            async def to_list() -> t.List[str]:
                return [x async for x in self.generate_async(*args, **kwargs)]

            yield from asyncio.run(to_list())
            return

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            yield from self.root_render_func(ctx)
        except Exception:
            yield self.environment.handle_exception()

    async def generate_async(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncGenerator[str, object]:
        """An async version of :meth:`generate`.  Works very similarly but
        returns an async iterator instead.
        """
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            agen = self.root_render_func(ctx)
            try:
                async for event in agen:  # type: ignore
                    yield event
            finally:
                # we can't use async with aclosing(...) because that's only
                # in 3.10+
                await agen.aclose()  # type: ignore
        except Exception:
            yield self.environment.handle_exception()

    def new_context(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> Context:
        """Create a new :class:`Context` for this template.  The vars
        provided will be passed to the template.  Per default the globals
        are added to the context.  If shared is set to `True` the data
        is passed as is to the context without adding the globals.

        `locals` can be a dict of local variables for internal usage.
        """
        return new_context(
            self.environment, self.name, self.blocks, vars, shared, self.globals, locals
        )

    def make_module(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """This method works like the :attr:`module` attribute when called
        without arguments but it will evaluate the template on every call
        rather than caching it.  It's also possible to provide
        a dict which is then used as context.  The arguments are the same
        as for the :meth:`new_context` method.
        """
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(self, ctx)

    async def make_module_async(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """As template module creation can invoke template code for
        asynchronous executions this method must be used instead of the
        normal :meth:`make_module` one.  Likewise the module attribute
        becomes unavailable in async mode.
        """
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(
            self,
            ctx,
            [x async for x in self.root_render_func(ctx)],  # type: ignore
        )

    @internalcode
    def _get_default_module(self, ctx: t.Optional[Context] = None) -> "TemplateModule":
        """If a context is passed in, this means that the template was
        imported. Imported templates have access to the current
        template's globals by default, but they can only be accessed via
        the context during runtime.

        If there are new globals, we need to create a new module because
        the cached module is already rendered and will not have access
        to globals from the current context. This new module is not
        cached because the template can be imported elsewhere, and it
        should have access to only the current template's globals.
        """
        if self.environment.is_async:
            raise RuntimeError("Module is not available in async mode.")

        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                return self.make_module({k: ctx.parent[k] for k in keys})

        if self._module is None:
            self._module = self.make_module()

        return self._module

    async def _get_default_module_async(
        self, ctx: t.Optional[Context] = None
    ) -> "TemplateModule":
        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                return await self.make_module_async({k: ctx.parent[k] for k in keys})

        if self._module is None:
            self._module = await self.make_module_async()

        return self._module

    @property
    def module(self) -> "TemplateModule":
        """The template as module.  This is used for imports in the
        template runtime but is also useful if one wants to access
        exported template variables from the Python layer:

        >>> t = Template('{% macro foo() %}42{% endmacro %}23')
        >>> str(t.module)
        '23'
        >>> t.module.foo() == u'42'
        True

        This attribute is not available if async mode is enabled.
        """
        return self._get_default_module()

    def get_corresponding_lineno(self, lineno: int) -> int:
        """Return the source line number of a line number in the
        generated bytecode as they are not in sync.
        """
        for template_line, code_line in reversed(self.debug_info):
            if code_line <= lineno:
                return template_line
        return 1

    @property
    def is_up_to_date(self) -> bool:
        """If this variable is `False` there is a newer version available."""
        if self._uptodate is None:
            return True
        return self._uptodate()

    @property
    def debug_info(self) -> t.List[t.Tuple[int, int]]:
        """The debug info mapping."""
        if self._debug_info:
            return [
                tuple(map(int, x.split("=")))  # type: ignore
                for x in self._debug_info.split("&")
            ]

        return []

    def __repr__(self) -> str:
        if self.name is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.name)
        return f"<{type(self).__name__} {name}>"


class TemplateModule:
    """Represents an imported template.  All the exported names of the
    template are available as attributes on this object.  Additionally
    converting it into a string renders the contents.
