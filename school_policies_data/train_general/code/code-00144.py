        if isinstance(timeout, Timeout):
            # Passed as a single explicit Timeout.
            assert connect is UNSET
            assert read is UNSET
            assert write is UNSET
            assert pool is UNSET
            self.connect = timeout.connect
            self.read = timeout.read
            self.write = timeout.write
            self.pool = timeout.pool
        elif isinstance(timeout, tuple):
            # Passed as a tuple.
            self.connect = timeout[0]
            self.read = timeout[1]
            self.write = None if len(timeout) < 3 else timeout[2]
            self.pool = None if len(timeout) < 4 else timeout[3]
        elif not (
            isinstance(connect, UnsetType)
            or isinstance(read, UnsetType)
            or isinstance(write, UnsetType)
            or isinstance(pool, UnsetType)
        ):
            self.connect = connect
            self.read = read
            self.write = write
            self.pool = pool
        else:
            if isinstance(timeout, UnsetType):
                raise ValueError("httpx2.Timeout must either include a default, or set all four parameters explicitly.")
            self.connect = timeout if isinstance(connect, UnsetType) else connect
            self.read = timeout if isinstance(read, UnsetType) else read
            self.write = timeout if isinstance(write, UnsetType) else write
            self.pool = timeout if isinstance(pool, UnsetType) else pool

    def as_dict(self) -> dict[str, float | None]:
        return {
            "connect": self.connect,
            "read": self.read,
            "write": self.write,
            "pool": self.pool,
        }

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if len({self.connect, self.read, self.write, self.pool}) == 1:
            return f"{class_name}(timeout={self.connect})"
        return f"{class_name}(connect={self.connect}, read={self.read}, write={self.write}, pool={self.pool})"


@dataclasses.dataclass(kw_only=True)
class Limits:
    """
    Configuration for limits to various client behaviors.

    **Parameters:**

    * **max_connections** - The maximum number of concurrent connections that may be
            established.
    * **max_keepalive_connections** - Allow the connection pool to maintain
            keep-alive connections below this point. Should be less than or equal
            to `max_connections`.
    * **keepalive_expiry** - Time limit on idle keep-alive connections in seconds.
    """

    max_connections: int | None = None
    max_keepalive_connections: int | None = None
    keepalive_expiry: float | None = 5.0


class Proxy:
    def __init__(
        self,
        url: URL | str,
        *,
        ssl_context: ssl.SSLContext | None = None,
        auth: tuple[str, str] | None = None,
        headers: HeaderTypes | None = None,
    ) -> None:
        url = URL(url)
        headers = Headers(headers)

        if url.scheme not in ("http", "https", "socks5", "socks5h"):
            raise ValueError(f"Unknown scheme for proxy URL {url!r}")

        if url.username or url.password:
            # Remove any auth credentials from the URL.
            auth = (url.username, url.password)
            url = url.copy_with(username=None, password=None)

        self.url = url
        self.auth = auth
        self.headers = headers
        self.ssl_context = ssl_context

    @property
    def raw_auth(self) -> tuple[bytes, bytes] | None:
        # The proxy authentication as raw bytes.
        return None if self.auth is None else (self.auth[0].encode("utf-8"), self.auth[1].encode("utf-8"))

    def __repr__(self) -> str:
        # The authentication is represented with the password component masked.
        auth = (self.auth[0], "********") if self.auth else None

        # Build a nice concise representation.
        url_str = f"{str(self.url)!r}"
        auth_str = f", auth={auth!r}" if auth else ""
        headers_str = f", headers={dict(self.headers)!r}" if self.headers else ""
        return f"Proxy({url_str}{auth_str}{headers_str})"


DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)
DEFAULT_LIMITS = Limits(max_connections=100, max_keepalive_connections=20)
DEFAULT_MAX_REDIRECTS = 20

DEFAULT_MAX_EVENT_SIZE_BYTES = 1024 * 1024

DEFAULT_MAX_MESSAGE_SIZE_BYTES = 65_536
DEFAULT_QUEUE_SIZE = 512
DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS = 20.0
DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS = 20.0
<|endoftext|>from __future__ import annotations

import io
import mimetypes
import os
import re
import typing
from pathlib import Path

from ._types import (
    AsyncByteStream,
    FileContent,
    FileTypes,
    RequestData,
    RequestFiles,
    SyncByteStream,
)
from ._utils import (
    peek_filelike_length,
    primitive_value_to_str,
    to_bytes,
)

_HTML5_FORM_ENCODING_REPLACEMENTS = {'"': "%22", "\\": "\\\\"}
_HTML5_FORM_ENCODING_REPLACEMENTS.update({chr(c): f"%{c:02X}" for c in range(0x1F + 1) if c != 0x1B})
_HTML5_FORM_ENCODING_RE = re.compile(r"|".join([re.escape(c) for c in _HTML5_FORM_ENCODING_REPLACEMENTS.keys()]))
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_FORBIDDEN_HEADER_VALUE_CHARS_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _format_form_param(name: str, value: str) -> bytes:
    """
    Encode a name/value pair within a multipart form.
    """

    def replacer(match: typing.Match[str]) -> str:
        return _HTML5_FORM_ENCODING_REPLACEMENTS[match.group(0)]

    value = _HTML5_FORM_ENCODING_RE.sub(replacer, value)
    return f'{name}="{value}"'.encode()


def _guess_content_type(filename: str | None) -> str | None:
    """
    Guesses the mimetype based on a filename. Defaults to `application/octet-stream`.

    Returns `None` if `filename` is `None` or empty.
    """
    if filename:
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return None


def get_multipart_boundary_from_content_type(
    content_type: bytes | None,
) -> bytes | None:
    if not content_type or not content_type.startswith(b"multipart/form-data"):
        return None
    # parse boundary according to
    # https://www.rfc-editor.org/rfc/rfc2046#section-5.1.1
    if b";" in content_type:
        for section in content_type.split(b";"):
            if section.strip().lower().startswith(b"boundary="):
                return section.strip()[len(b"boundary=") :].strip(b'"')
    return None


class DataField:
    """
    A single form field item, within a multipart form field.
    """

    def __init__(self, name: str, value: str | bytes | int | float | None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Invalid type for name. Expected str, got {type(name)}: {name!r}")
        if value is not None and not isinstance(value, (str, bytes, int, float)):
            raise TypeError(f"Invalid type for value. Expected primitive type, got {type(value)}: {value!r}")
        self.name = name
        self.value: str | bytes = value if isinstance(value, bytes) else primitive_value_to_str(value)

    def render_headers(self) -> bytes:
        if not hasattr(self, "_headers"):
            name = _format_form_param("name", self.name)
            self._headers = b"".join([b"Content-Disposition: form-data; ", name, b"\r\n\r\n"])

        return self._headers

    def render_data(self) -> bytes:
        if not hasattr(self, "_data"):
            self._data = to_bytes(self.value)

        return self._data

    def get_length(self) -> int:
        headers = self.render_headers()
        data = self.render_data()
        return len(headers) + len(data)

    def render(self) -> typing.Iterator[bytes]:
        yield self.render_headers()
        yield self.render_data()


class FileField:
    """
    A single file field item, within a multipart form field.
    """

    CHUNK_SIZE = 64 * 1024

    def __init__(self, name: str, value: FileTypes) -> None:
        self.name = name

        fileobj: FileContent

        headers: dict[str, str] = {}
        content_type: str | None = None

        # This large tuple based API largely mirror's requests' API
        # It would be good to think of better APIs for this that we could
        # include in httpx 2.0 since variable length tuples(especially of 4 elements)
        # are quite unwieldy
        if isinstance(value, tuple):
            if len(value) == 2:
                # neither the 3rd parameter (content_type) nor the 4th (headers)
                # was included
                filename, fileobj = value
            elif len(value) == 3:
                filename, fileobj, content_type = value
            else:
                # all 4 parameters included
                filename, fileobj, content_type, headers = value  # type: ignore
        else:
            filename = Path(str(getattr(value, "name", "upload"))).name
            fileobj = value

        if content_type is None:
            content_type = _guess_content_type(filename)

        has_content_type_header = any("content-type" in key.lower() for key in headers)
        if content_type is not None and not has_content_type_header:
            # note that unlike requests, we ignore the content_type provided in the 3rd
            # tuple element if it is also included in the headers requests does
            # the opposite (it overwrites the headerwith the 3rd tuple element)
            headers["Content-Type"] = content_type

        if isinstance(fileobj, io.StringIO):
            raise TypeError("Multipart file uploads require 'io.BytesIO', not 'io.StringIO'.")
        if isinstance(fileobj, io.TextIOBase):
            raise TypeError("Multipart file uploads must be opened in binary mode, not text mode.")

        self.filename = filename
        self.file = fileobj
        self.headers = headers

    def get_length(self) -> int | None:
        headers = self.render_headers()

        if isinstance(self.file, (str, bytes)):
            return len(headers) + len(to_bytes(self.file))

        file_length = peek_filelike_length(self.file)

        # If we can't determine the filesize without reading it into memory,
        # then return `None` here, to indicate an unknown file length.
        if file_length is None:
            return None

        return len(headers) + file_length

    def render_headers(self) -> bytes:
        if not hasattr(self, "_headers"):
            parts = [
                b"Content-Disposition: form-data; ",
                _format_form_param("name", self.name),
            ]
            if self.filename:
                filename = _format_form_param("filename", self.filename)
                parts.extend([b"; ", filename])
            for header_name, header_value in self.headers.items():
                if _HEADER_NAME_RE.fullmatch(header_name) is None:
                    raise ValueError("Invalid multipart header name.")
                if _FORBIDDEN_HEADER_VALUE_CHARS_RE.search(header_value) is not None:
                    raise ValueError("Invalid control character in multipart header value.")
                key, val = f"\r\n{header_name}: ".encode(), header_value.encode()
                parts.extend([key, val])
            parts.append(b"\r\n\r\n")
            self._headers = b"".join(parts)

        return self._headers

    def render_data(self) -> typing.Iterator[bytes]:
        if isinstance(self.file, (str, bytes)):
            yield to_bytes(self.file)
            return

        if hasattr(self.file, "seek"):
            try:
                self.file.seek(0)
            except io.UnsupportedOperation:
                pass

        chunk = self.file.read(self.CHUNK_SIZE)
        while chunk:
            yield to_bytes(chunk)
            chunk = self.file.read(self.CHUNK_SIZE)

    def render(self) -> typing.Iterator[bytes]:
        yield self.render_headers()
        yield from self.render_data()


class MultipartStream(SyncByteStream, AsyncByteStream):
    """
    Request content as streaming multipart encoded form data.
    """

    def __init__(
        self,
        data: RequestData,
        files: RequestFiles,
        boundary: bytes | None = None,
    ) -> None:
        if boundary is None:
            boundary = os.urandom(16).hex().encode("ascii")

        self.boundary = boundary
        self.content_type = f"multipart/form-data; boundary={boundary.decode('ascii')}"
        self.fields = list(self._iter_fields(data, files))

    def _iter_fields(self, data: RequestData, files: RequestFiles) -> typing.Iterator[FileField | DataField]:
        for name, value in data.items():
            if isinstance(value, (tuple, list)):
                for item in value:
                    yield DataField(name=name, value=item)
            else:
                yield DataField(name=name, value=value)

        file_items = files.items() if isinstance(files, typing.Mapping) else files
        for name, value in file_items:
            yield FileField(name=name, value=value)

    def iter_chunks(self) -> typing.Iterator[bytes]:
        for field in self.fields:
            yield b"--%s\r\n" % self.boundary
            yield from field.render()
            yield b"\r\n"
        yield b"--%s--\r\n" % self.boundary

    def get_content_length(self) -> int | None:
        """
        Return the length of the multipart encoded content, or `None` if
        any of the files have a length that cannot be determined upfront.
        """
        boundary_length = len(self.boundary)
        length = 0

        for field in self.fields:
            field_length = field.get_length()
            if field_length is None:
                return None

            length += 2 + boundary_length + 2  # b"--{boundary}\r\n"
            length += field_length
            length += 2  # b"\r\n"

        length += 2 + boundary_length + 4  # b"--{boundary}--\r\n"
        return length

    # Content stream interface.

    def get_headers(self) -> dict[str, str]:
        content_length = self.get_content_length()
        content_type = self.content_type
        if content_length is None:
            return {"Transfer-Encoding": "chunked", "Content-Type": content_type}
        return {"Content-Length": str(content_length), "Content-Type": content_type}

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from self.iter_chunks()

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        for chunk in self.iter_chunks():
            yield chunk
<|endoftext|>from __future__ import annotations

import contextlib
import math
import typing
from types import TracebackType

import anyio
import wsproto
from anyio.streams.stapled import StapledObjectStream
from wsproto.frame_protocol import CloseReason

from .._models import Request, Response
from .._transports.asgi import ASGITransport, _ASGIApp
from .._types import AsyncByteStream
from ._exceptions import WebSocketDisconnect, WebSocketUpgradeError

Scope = dict[str, typing.Any]
Message = dict[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Scope], typing.Coroutine[None, None, None]]
ASGIApp = typing.Callable[[Scope, Receive, Send], typing.Coroutine[None, None, None]]


class ASGIWebSocketTransportError(Exception):
    pass


class UnhandledASGIMessageType(ASGIWebSocketTransportError):
    def __init__(self, message: Message) -> None:
        self.message = message


class UnhandledWebSocketEvent(ASGIWebSocketTransportError):
    def __init__(self, event: wsproto.events.Event) -> None:
        self.event = event


class ASGIWebSocketAsyncNetworkStream:
    def __init__(
        self,
        app: ASGIApp,
        scope: Scope,
        task_group: anyio.abc.TaskGroup,
        initial_receive_timeout: float = 1.0,
    ) -> None:
        self.app = app
        self.scope = scope
        self._receive_queue = StapledObjectStream(*anyio.create_memory_object_stream[Message](max_buffer_size=math.inf))
        self._send_queue = StapledObjectStream(*anyio.create_memory_object_stream[Message](max_buffer_size=math.inf))
        self._task_group = task_group
        self._initial_receive_timeout = initial_receive_timeout
        self.connection = wsproto.WSConnection(wsproto.ConnectionType.SERVER)
        self.connection.initiate_upgrade_connection(scope["headers"], scope["path"])
        self._aentered = False

    async def __aenter__(
        self,
    ) -> tuple[ASGIWebSocketAsyncNetworkStream, bytes]:
        if self._aentered:
            raise RuntimeError("Cannot use ASGIWebSocketAsyncNetworkStream in a context manager twice")
        self._aentered = True
        self._task_group.start_soon(self._run)
        async with contextlib.AsyncExitStack() as stack:
            stack.push_async_callback(self.aclose)
            await self.send({"type": "websocket.connect"})

            try:
                message = await self.receive(self._initial_receive_timeout)
            except TimeoutError as e:
                raise RuntimeError(
                    "WebSocket didn't accept the connection in time. Did you forget to call accept()?"
                ) from e

            if message["type"] == "websocket.close":
                await stack.aclose()
                raise WebSocketDisconnect(message["code"], message.get("reason"))

            # Websocket Denial Response extension
            # Ref: https://asgi.readthedocs.io/en/latest/extensions.html#websocket-denial-response
            if message["type"] == "websocket.http.response.start":
                status_code: int = message["status"]
                headers: list[tuple[bytes, bytes]] = message["headers"]
                body: list[bytes] = []
                while True:
                    message = await self.receive()
                    assert message["type"] == "websocket.http.response.body"
                    body.append(message["body"])
                    if not message.get("more_body", False):
                        break

                await stack.aclose()
                raise WebSocketUpgradeError(Response(status_code, headers=headers, content=b"".join(body)))

            assert message["type"] == "websocket.accept"
            retval = self, self._build_accept_response(message)
            self._exit_stack = stack.pop_all()
        return retval

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        message: Message = await self.receive(timeout=timeout)
        type = message["type"]

        if type not in {"websocket.send", "websocket.close"}:
            raise UnhandledASGIMessageType(message)

        event: wsproto.events.Event
        if type == "websocket.send":
            data_str: str | None = message.get("text")
            if data_str is not None:
                event = wsproto.events.TextMessage(data_str)
            data_bytes: bytes | None = message.get("bytes")
            if data_bytes is not None:
                event = wsproto.events.BytesMessage(data_bytes)
        elif type == "websocket.close":
            event = wsproto.events.CloseConnection(message["code"], message["reason"])

        return self.connection.send(event)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.connection.receive_data(buffer)
        for event in self.connection.events():
            if isinstance(event, wsproto.events.Request):
                pass
            elif isinstance(event, wsproto.events.CloseConnection):
                await self.send(
                    {
                        "type": "websocket.disconnect",
                        "code": event.code,
                        "reason": event.reason,
                    }
                )
            elif isinstance(event, wsproto.events.TextMessage):
                await self.send({"type": "websocket.receive", "text": event.data})
            elif isinstance(event, wsproto.events.BytesMessage):
                await self.send({"type": "websocket.receive", "bytes": event.data})
            else:
                raise UnhandledWebSocketEvent(event)

    async def aclose(self) -> None:
        with contextlib.suppress(anyio.ClosedResourceError):
            await self.send({"type": "websocket.disconnect"})
        await self._receive_queue.aclose()
        await self._send_queue.aclose()

    async def send(self, message: Message) -> None:
        await self._receive_queue.send(message)

    async def receive(self, timeout: float | None = None) -> Message:
        if timeout is None:
            timeout = math.inf
        with anyio.fail_after(timeout):
            return await self._send_queue.receive()

    async def _run(self) -> None:
        """
        The sub-thread in which the websocket session runs.
        """
        scope = self.scope
        receive = self._receive_queue.receive
        send = self._send_queue.send
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            message = {
                "type": "websocket.close",
                "code": CloseReason.INTERNAL_ERROR,
                "reason": str(e),
            }
            with contextlib.suppress(anyio.ClosedResourceError):
                await send(message)

    def _build_accept_response(self, message: Message) -> bytes:
        subprotocol = message.get("subprotocol", None)
        headers = message.get("headers", [])
        return self.connection.send(
            wsproto.events.AcceptConnection(
                subprotocol=subprotocol,
                extra_headers=headers,
            )
        )


class ASGIWebSocketTransport(ASGITransport):
    def __init__(
        self,
        app: _ASGIApp,
        raise_app_exceptions: bool = True,
        root_path: str = "",
        client: tuple[str, int] = ("127.0.0.1", 123),
        initial_receive_timeout: float = 1.0,
    ) -> None:
        super().__init__(app, raise_app_exceptions, root_path, client)
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._initial_receive_timeout = initial_receive_timeout

    async def __aenter__(self) -> ASGIWebSocketTransport:
        async with contextlib.AsyncExitStack() as stack:
            self._task_group = await stack.enter_async_context(anyio.create_task_group())
            self._exit_stack = stack.pop_all()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        await super().__aexit__(exc_type, exc_val, exc_tb)
        assert self._exit_stack is not None
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def handle_async_request(self, request: Request) -> Response:
        scheme = request.url.scheme
        headers = request.headers

        if scheme in {"ws", "wss"} or headers.get("upgrade") == "websocket":
            subprotocols: list[str] = []
            if (subprotocols_header := headers.get("sec-websocket-protocol")) is not None:
                subprotocols = subprotocols_header.split(",")

            scope = {
                "type": "websocket",
                "path": request.url.path,
                "raw_path": request.url.raw_path,
                "root_path": self.root_path,
                "scheme": scheme,
                "query_string": request.url.query,
                "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
                "client": self.client,
                "server": (request.url.host, request.url.port),
                "subprotocols": subprotocols,
            }
            return await self._handle_ws_request(request, scope)

        return await super().handle_async_request(request)

    async def _create_asgi_websocket_async_network_stream(
        self,
        *,
        task_status: anyio.abc.TaskStatus[tuple[ASGIWebSocketAsyncNetworkStream, bytes]],
    ) -> None:
        stream = ASGIWebSocketAsyncNetworkStream(
            self.app,  # type: ignore[arg-type]
            self.scope,
            self._task_group,
            self._initial_receive_timeout,
        )
        assert self._exit_stack is not None
        result = await self._exit_stack.enter_async_context(stream)
        task_status.started(result)

    async def _handle_ws_request(
        self,
        request: Request,
        scope: Scope,
    ) -> Response:
        assert isinstance(request.stream, AsyncByteStream)

        self.scope = scope
        stream, accept_response = await self._task_group.start(self._create_asgi_websocket_async_network_stream)
        accept_response_lines = accept_response.decode("utf-8").splitlines()
        headers = [
            typing.cast(tuple[str, str], line.split(": ", 1))
            for line in accept_response_lines[1:]
            if line.strip() != ""
        ]

        return Response(
            status_code=101,
            headers=headers,
            extensions={"network_stream": stream},
        )
<|endoftext|>"""
WebSocket support, derived from httpx-ws (https://github.com/frankie567/httpx-ws).

```
MIT License

Copyright (c) 2021 François Voron

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
```
"""

from ._api import (
    AsyncWebSocketClient,
    AsyncWebSocketSession,
    JSONMode,
    WebSocketClient,
    WebSocketSession,
)
from ._exceptions import (
    HTTPXWSException,
    WebSocketDisconnect,
    WebSocketInvalidTypeReceived,
    WebSocketNetworkError,
    WebSocketUpgradeError,
)
from ._transport import ASGIWebSocketTransport

__all__ = [
    "ASGIWebSocketTransport",
    "AsyncWebSocketClient",
    "AsyncWebSocketSession",
    "HTTPXWSException",
    "JSONMode",
    "WebSocketClient",
    "WebSocketDisconnect",
    "WebSocketInvalidTypeReceived",
    "WebSocketNetworkError",
    "WebSocketSession",
    "WebSocketUpgradeError",
]
<|endoftext|>from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import wsproto

    from .._models import Response


class HTTPXWSException(Exception):
    """
    Base exception class for HTTPX WS.
    """


class WebSocketUpgradeError(HTTPXWSException):
    """
    Raised when the initial connection didn't correctly upgrade to a WebSocket session.
    """

    def __init__(self, response: Response) -> None:
        self.response = response


class WebSocketDisconnect(HTTPXWSException):
    """
    Raised when the server closed the WebSocket session.

    Args:
        code:
            The integer close code to indicate why the connection has closed.
        reason:
            Additional reasoning for why the connection has closed.
    """

    def __init__(self, code: int = 1000, reason: str | None = None) -> None:
        self.code = code
        self.reason = reason or ""


class WebSocketInvalidTypeReceived(HTTPXWSException):
    """
    Raised when an event is not of the expected type.
    """

    def __init__(self, event: wsproto.events.Event) -> None:
        self.event = event


class WebSocketNetworkError(HTTPXWSException):
    """
    Raised when a network error occured,
    typically if the underlying stream has closed or timeout.
    """
<|endoftext|>from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import json
import queue
import secrets
import sys
import threading
import typing
from types import TracebackType

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

import anyio
import wsproto
import wsproto.utilities
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from wsproto.frame_protocol import CloseReason

from .._client import USE_CLIENT_DEFAULT
from .._config import (
    DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
    DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    DEFAULT_QUEUE_SIZE,
)
from .._models import Headers
from ._exceptions import (
    HTTPXWSException,
    WebSocketDisconnect,
    WebSocketInvalidTypeReceived,
    WebSocketNetworkError,
    WebSocketUpgradeError,
)
from ._ping import AsyncPingManager, PingManager
from ._transport import ASGIWebSocketAsyncNetworkStream

if typing.TYPE_CHECKING:
    from httpcore2 import AsyncNetworkStream, NetworkStream

    from .._client import AsyncClient, Client, UseClientDefault
    from .._models import Response
    from .._types import (
        AuthTypes,
        CookieTypes,
        HeaderTypes,
        QueryParamTypes,
        RequestExtensions,
        TimeoutTypes,
    )

JSONMode = typing.Literal["text", "binary"]
TaskFunction = typing.TypeVar("TaskFunction")
TaskResult = typing.TypeVar("TaskResult")
SyncSession = TypeVar("SyncSession", bound="WebSocketSession", default="WebSocketSession")
AsyncSession = TypeVar("AsyncSession", bound="AsyncWebSocketSession", default="AsyncWebSocketSession")


class ShouldClose(Exception):
    pass


class EndOfStream(Exception):
    pass


class WebSocketSession:
    """
    Sync context manager representing an opened WebSocket session.

    Attributes:
        subprotocol (typing.Optional[str]):
            Optional protocol that has been accepted by the server.
        response (Response | None):
            The webSocket handshake response.
    """

    subprotocol: str | None
    response: Response | None

    def __init__(
        self,
        stream: NetworkStream,
        *,
        max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
        keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
        response: Response | None = None,
    ) -> None:
        self.stream = stream
        self.connection = wsproto.connection.Connection(wsproto.ConnectionType.CLIENT)
        self.response = response
        if self.response is not None:
            self.subprotocol = self.response.headers.get("sec-websocket-protocol")
        else:
            self.subprotocol = None

        self._events: queue.Queue[wsproto.events.Event | HTTPXWSException] = queue.Queue(queue_size)

        self._ping_manager = PingManager()
        self._should_close = threading.Event()
        self._write_lock = threading.Lock()
        self._should_close_task: concurrent.futures.Future[bool] | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        self._max_message_size_bytes = max_message_size_bytes
        self._queue_size = queue_size
        self._keepalive_ping_interval_seconds = keepalive_ping_interval_seconds
        self._keepalive_ping_timeout_seconds = keepalive_ping_timeout_seconds

    def _get_executor_should_close_task(
        self,
    ) -> tuple[concurrent.futures.ThreadPoolExecutor, concurrent.futures.Future[bool]]:
        if self._should_close_task is None:
            self._executor = concurrent.futures.ThreadPoolExecutor()
            self._should_close_task = self._executor.submit(self._should_close.wait)
        assert self._executor is not None
        return self._executor, self._should_close_task

    def __enter__(self) -> WebSocketSession:
        self._background_receive_task = threading.Thread(
            target=self._background_receive, args=(self._max_message_size_bytes,)
        )
        self._background_receive_task.start()

        self._background_keepalive_ping_task: threading.Thread | None = None
        if self._keepalive_ping_interval_seconds is not None:
            self._background_keepalive_ping_task = threading.Thread(
                target=self._background_keepalive_ping,
                args=(
                    self._keepalive_ping_interval_seconds,
                    self._keepalive_ping_timeout_seconds,
                ),
            )
            self._background_keepalive_ping_task.start()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
        self._background_receive_task.join()
        if self._background_keepalive_ping_task is not None:
            self._background_keepalive_ping_task.join()

    def ping(self, payload: bytes = b"") -> threading.Event:
        """
        Send a Ping message.

        Args:
            payload:
                Payload to attach to the Ping event.
                Internally, it's used to track this specific event.
                If left empty, a random one will be generated.

        Returns:
            An event that can be used to wait for the corresponding Pong response.

        Examples:
            Send a Ping and wait for the Pong

                pong_callback = ws.ping()
                # Will block until the corresponding Pong is received.
                pong_callback.wait()
        """
        ping_id, callback = self._ping_manager.create(payload)
        event = wsproto.events.Ping(ping_id)
        self.send(event)
        return callback

    def send(self, event: wsproto.events.Event) -> None:
        """
        Send an Event message.

        Mainly useful to send events that are not supported by the library.
        Most of the time, [ping()][httpx_ws.WebSocketSession.ping],
        [send_text()][httpx_ws.WebSocketSession.send_text],
        [send_bytes()][httpx_ws.WebSocketSession.send_bytes]
        and [send_json()][httpx_ws.WebSocketSession.send_json] are preferred.

        Args:
            event: The event to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Examples:
            Send an event.

                event = wsproto.events.Message(b"Hello!")
                ws.send(event)
        """
        import httpcore2

        try:
            data = self.connection.send(event)
            with self._write_lock:
                self.stream.write(data)
        except httpcore2.WriteError as e:
            self.close(CloseReason.INTERNAL_ERROR, "Stream write error")
            raise WebSocketNetworkError() from e

    def send_text(self, data: str) -> None:
        """
        Send a text message.

        Args:
            data: The text to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Examples:
            Send a text message.

                ws.send_text("Hello!")
        """
        event = wsproto.events.TextMessage(data=data)
        self.send(event)

    def send_bytes(self, data: bytes) -> None:
        """
        Send a bytes message.

        Args:
            data: The data to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Examples:
            Send a bytes message.

                ws.send_bytes(b"Hello!")
        """
        event = wsproto.events.BytesMessage(data=data)
        self.send(event)

    def send_json(self, data: typing.Any, mode: JSONMode = "text") -> None:
        """
        Send JSON data.

        Args:
            data:
                The data to send. Must be serializable by [json.dumps][json.dumps].
            mode:
                The sending mode. Should either be `'text'` or `'bytes'`.

        Raises:
            WebSocketNetworkError: A network error occured.

        Examples:
            Send JSON data.

                data = {"message": "Hello!"}
                ws.send_json(data)
        """
        assert mode in ["text", "binary"]
        serialized_data = json.dumps(data)
        if mode == "text":
            self.send_text(serialized_data)
        else:
            self.send_bytes(serialized_data.encode("utf-8"))

    def receive(self, timeout: float | None = None) -> wsproto.events.Event:
        """
        Receive an event from the server.

        Mainly useful to receive raw [wsproto.events.Event][wsproto.events.Event].
        Most of the time, [receive_text()][httpx_ws.WebSocketSession.receive_text],
        [receive_bytes()][httpx_ws.WebSocketSession.receive_bytes],
        and [receive_json()][httpx_ws.WebSocketSession.receive_json] are preferred.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            A raw [wsproto.events.Event][wsproto.events.Event].

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.

        Examples:
            Wait for an event until one is available.

                try:
                    event = ws.receive()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for an event for 2 seconds.

                try:
                    event = ws.receive(timeout=2.)
                except TimeoutError:
                    print("No event received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        try:
            event = self._events.get(block=True, timeout=timeout)
        except queue.Empty as e:
            raise TimeoutError from e
        if isinstance(event, HTTPXWSException):
            raise event
        if isinstance(event, wsproto.events.CloseConnection):
            raise WebSocketDisconnect(event.code, event.reason)
        return event

    def receive_text(self, timeout: float | None = None) -> str:
        """
        Receive text from the server.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            Text data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event was not a text message.

        Examples:
            Wait for text until available.

                try:
                    text = ws.receive_text()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for text for 2 seconds.

                try:
                    event = ws.receive_text(timeout=2.)
                except TimeoutError:
                    print("No text received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        event = self.receive(timeout)
        if isinstance(event, wsproto.events.TextMessage):
            return event.data
        raise WebSocketInvalidTypeReceived(event)

    def receive_bytes(self, timeout: float | None = None) -> bytes:
        """
        Receive bytes from the server.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            Bytes data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event was not a bytes message.

        Examples:
            Wait for bytes until available.

                try:
                    data = ws.receive_bytes()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for bytes for 2 seconds.

                try:
                    data = ws.receive_bytes(timeout=2.)
                except TimeoutError:
                    print("No data received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        event = self.receive(timeout)
        if isinstance(event, wsproto.events.BytesMessage):
            return bytes(event.data)
        raise WebSocketInvalidTypeReceived(event)

    def receive_json(self, timeout: float | None = None, mode: JSONMode = "text") -> typing.Any:
        """
        Receive JSON data from the server.

        The received data should be parseable by [json.loads][json.loads].

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.
            mode:
                Receive mode. Should either be `'text'` or `'bytes'`.

        Returns:
            Parsed JSON data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event
                didn't correspond to the specified mode.

        Examples:
            Wait for data until available.

                try:
                    data = ws.receive_json()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for data for 2 seconds.

                try:
                    data = ws.receive_json(timeout=2.)
                except TimeoutError:
                    print("No data received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        assert mode in ["text", "binary"]
        data: str | bytes
        if mode == "text":
            data = self.receive_text(timeout)
        elif mode == "binary":
            data = self.receive_bytes(timeout)
        return json.loads(data)

    def close(self, code: int = 1000, reason: str | None = None) -> None:
        """
        Close the WebSocket session.

        Internally, it'll send the
        [CloseConnection][wsproto.events.CloseConnection] event.

        *This method is automatically called when exiting the context manager.*

        Args:
            code:
                The integer close code to indicate why the connection has closed.
            reason:
                Additional reasoning for why the connection has closed.

        Examples:
            Close the WebSocket session.

                ws.close()
        """
        import httpcore2

        self._should_close.set()
        if self._executor is not None:
            self._executor.shutdown(False)
        if self.connection.state not in {
            wsproto.connection.ConnectionState.LOCAL_CLOSING,
            wsproto.connection.ConnectionState.CLOSED,
        }:
            event = wsproto.events.CloseConnection(code, reason)
            data = self.connection.send(event)
            try:
                with self._write_lock:
                    self.stream.write(data)
            except httpcore2.WriteError:
                pass
        self.stream.close()

    def _background_receive(self, max_bytes: int) -> None:
        """
        Background thread listening for data from the server.

        Internally, it'll:

        * Answer to Ping events.
        * Acknowledge Pong events.
        * Put other events in the [_events][_events]
        queue that'll eventually be consumed by the user.

        Args:
            max_bytes: The maximum chunk size to read at each iteration.
        """
        import httpcore2

        partial_message_buffer: str | bytes | None = None
        partial_message_size = 0
        try:
            while not self._should_close.is_set():
                data = self._wait_until_closed(self._read_stream, max_bytes)
                self.connection.receive_data(data)
                for event in self.connection.events():
                    if isinstance(event, wsproto.events.Ping):
                        data = self.connection.send(event.response())
                        with self._write_lock:
                            self.stream.write(data)
                        continue
                    if isinstance(event, wsproto.events.Pong):
                        self._ping_manager.ack(event.payload)
                        continue
                    if isinstance(event, wsproto.events.CloseConnection):
                        self._should_close.set()
                    if isinstance(event, wsproto.events.Message):
                        partial_message_size += len(event.data.encode() if isinstance(event.data, str) else event.data)
                        if partial_message_size > max_bytes:
                            self.close(CloseReason.MESSAGE_TOO_BIG, "Message too big")
                            self._events.put(WebSocketDisconnect(CloseReason.MESSAGE_TOO_BIG, "Message too big"))
                            break
                        # Unfinished message: bufferize
                        if not event.message_finished:
                            if partial_message_buffer is None:
                                partial_message_buffer = event.data
                            else:
                                partial_message_buffer += event.data
                        # Finished message but no buffer: just emit the event
                        elif partial_message_buffer is None:
                            partial_message_size = 0
                            self._events.put(event)
                        # Finished message with buffer: emit the full event
                        else:
                            event_type = type(event)
                            full_message_event = event_type(partial_message_buffer + event.data)
                            partial_message_buffer = None
                            partial_message_size = 0
                            self._events.put(full_message_event)
                        continue
                    self._events.put(event)
        except (httpcore2.ReadError, httpcore2.WriteError, EndOfStream):
            self.close(CloseReason.INTERNAL_ERROR, "Stream error")
            self._events.put(WebSocketNetworkError())
        except ShouldClose:
            pass

    def _background_keepalive_ping(self, interval_seconds: float, timeout_seconds: float | None = None) -> None:
        try:
            while not self._should_close.is_set():
                should_close = self._wait_until_closed(self._should_close.wait, interval_seconds)
                if should_close:  # pragma: no cover
                    raise ShouldClose()
                try:
                    pong_callback = self.ping()
                except wsproto.utilities.LocalProtocolError:
                    return
                if timeout_seconds is not None:
                    acknowledged = self._wait_until_closed(pong_callback.wait, timeout_seconds)
                    if not acknowledged:
                        self.close(CloseReason.INTERNAL_ERROR, "Keepalive ping timeout")
                        self._events.put(WebSocketNetworkError())
        except ShouldClose:
            pass

    def _wait_until_closed(
        self, callable: typing.Callable[..., TaskResult], *args: typing.Any, **kwargs: typing.Any
    ) -> TaskResult:
        try:
            executor, should_close_task = self._get_executor_should_close_task()
            todo_task = executor.submit(callable, *args, **kwargs)
        except RuntimeError as e:
            raise ShouldClose() from e
        else:
            done, _ = concurrent.futures.wait(
                (todo_task, should_close_task),  # type: ignore[misc]
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if should_close_task in done:
                raise ShouldClose()
            assert todo_task in done
            result = todo_task.result()
        return result

    def _read_stream(self, max_bytes: int) -> bytes:
        data = self.stream.read(max_bytes)
        if data == b"":
            raise EndOfStream()
        return data


class AsyncWebSocketSession(anyio.AsyncContextManagerMixin):
    """
    Async context manager representing an opened WebSocket session.

    Internally, this session uses an anyio task group to manage background tasks.
    As a result, exceptions that are not caught inside the context manager
    and propagate out of the `async with` block will be wrapped
    in an [ExceptionGroup][ExceptionGroup].

    To handle them, use the `except*` syntax:

        async with AsyncWebSocketSession(stream) as ws:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                # Caught inside the context manager: plain exception.
                print("Connection closed")

        # If not caught inside:
        try:
            async with AsyncWebSocketSession(stream) as ws:
                data = await ws.receive_text()
        except* WebSocketDisconnect:
            # Propagated out of the context manager: wrapped in ExceptionGroup.
            print("Connection closed")

    Attributes:
        subprotocol (typing.Optional[str]):
            Optional protocol that has been accepted by the server.
        response (Response | None):
            The webSocket handshake response.
    """

    subprotocol: str | None
    response: Response | None
    _send_event: MemoryObjectSendStream[wsproto.events.Event | HTTPXWSException]
    _receive_event: MemoryObjectReceiveStream[wsproto.events.Event | HTTPXWSException]

    def __init__(
        self,
        stream: AsyncNetworkStream,
        *,
        max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
        keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
        response: Response | None = None,
    ) -> None:
        self.stream = stream
        self.connection = wsproto.connection.Connection(wsproto.ConnectionType.CLIENT)
        self.response = response
        if self.response is not None:
            self.subprotocol = self.response.headers.get("sec-websocket-protocol")
        else:
            self.subprotocol = None

        self._ping_manager = AsyncPingManager()
        self._should_close = anyio.Event()
        self._write_lock = anyio.Lock()

        self._max_message_size_bytes = max_message_size_bytes
        self._queue_size = queue_size

        # Always disable keepalive ping when emulating ASGI
        if isinstance(stream, ASGIWebSocketAsyncNetworkStream):
            self._keepalive_ping_interval_seconds = None
            self._keepalive_ping_timeout_seconds = None
        else:
            self._keepalive_ping_interval_seconds = keepalive_ping_interval_seconds
            self._keepalive_ping_timeout_seconds = keepalive_ping_timeout_seconds

    @contextlib.asynccontextmanager
    async def __asynccontextmanager__(self) -> typing.AsyncGenerator[AsyncWebSocketSession, None]:
        self._send_event, self._receive_event = anyio.create_memory_object_stream[
            wsproto.events.Event | HTTPXWSException
        ]()
        self._background_task_group = anyio.create_task_group()

        async with self._send_event, self._receive_event, self._background_task_group:
            self._background_task_group.start_soon(self._background_receive, self._max_message_size_bytes)
            if self._keepalive_ping_interval_seconds is not None:
                self._background_task_group.start_soon(
                    self._background_keepalive_ping,
                    self._keepalive_ping_interval_seconds,
                    self._keepalive_ping_timeout_seconds,
                )

            try:
                yield self
            finally:
                self._background_task_group.cancel_scope.cancel()
                with anyio.CancelScope(shield=True):
                    await self.close()

    async def ping(self, payload: bytes = b"") -> anyio.Event:
        """
        Send a Ping message.

        Args:
            payload:
                Payload to attach to the Ping event.
                Internally, it's used to track this specific event.
                If left empty, a random one will be generated.

        Returns:
            An event that can be used to wait for the corresponding Pong response.

        Examples:
            Send a Ping and wait for the Pong

                pong_callback = await ws.ping()
                # Will block until the corresponding Pong is received.
                await pong_callback.wait()
        """
        ping_id, callback = self._ping_manager.create(payload)
        event = wsproto.events.Ping(ping_id)
        await self.send(event)
        return callback

    async def send(self, event: wsproto.events.Event) -> None:
        """
        Send an Event message.

        Mainly useful to send events that are not supported by the library.
        Most of the time, [ping()][httpx_ws.AsyncWebSocketSession.ping],
        [send_text()][httpx_ws.AsyncWebSocketSession.send_text],
        [send_bytes()][httpx_ws.AsyncWebSocketSession.send_bytes]
        and [send_json()][httpx_ws.AsyncWebSocketSession.send_json] are preferred.

        Args:
            event: The event to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Send an event.

                event = await wsproto.events.Message(b"Hello!")
                ws.send(event)
        """
        import httpcore2

        try:
            data = self.connection.send(event)
            async with self._write_lock:
                await self.stream.write(data)
        except httpcore2.WriteError as e:
            await self.close(CloseReason.INTERNAL_ERROR, "Stream write error")
            raise WebSocketNetworkError() from e

    async def send_text(self, data: str) -> None:
        """
        Send a text message.

        Args:
            data: The text to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Send a text message.

                await ws.send_text("Hello!")
        """
        event = wsproto.events.TextMessage(data=data)
        await self.send(event)

    async def send_bytes(self, data: bytes) -> None:
        """
        Send a bytes message.

        Args:
            data: The data to send.

        Raises:
            WebSocketNetworkError: A network error occured.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Send a bytes message.

                await ws.send_bytes(b"Hello!")
        """
        event = wsproto.events.BytesMessage(data=data)
        await self.send(event)

    async def send_json(self, data: typing.Any, mode: JSONMode = "text") -> None:
        """
        Send JSON data.

        Args:
            data:
                The data to send. Must be serializable by [json.dumps][json.dumps].
            mode:
                The sending mode. Should either be `'text'` or `'bytes'`.

        Raises:
            WebSocketNetworkError: A network error occured.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Send JSON data.

                data = {"message": "Hello!"}
                await ws.send_json(data)
        """
        assert mode in ["text", "binary"]
        serialized_data = json.dumps(data)
        if mode == "text":
            await self.send_text(serialized_data)
        else:
            await self.send_bytes(serialized_data.encode("utf-8"))

    async def receive(self, timeout: float | None = None) -> wsproto.events.Event:
        """
        Receive an event from the server.

        Mainly useful to receive raw [wsproto.events.Event][wsproto.events.Event].
        Most of the time, [receive_text()][httpx_ws.AsyncWebSocketSession.receive_text],
        [receive_bytes()][httpx_ws.AsyncWebSocketSession.receive_bytes],
        and [receive_json()][httpx_ws.AsyncWebSocketSession.receive_json] are preferred.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            A raw [wsproto.events.Event][wsproto.events.Event].

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Wait for an event until one is available.

                try:
                    event = await ws.receive()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for an event for 2 seconds.

                try:
                    event = await ws.receive(timeout=2.)
                except TimeoutError:
                    print("No event received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        with anyio.fail_after(timeout):
            event = await self._receive_event.receive()
        if isinstance(event, HTTPXWSException):
            raise event
        if isinstance(event, wsproto.events.CloseConnection):
            raise WebSocketDisconnect(event.code, event.reason)
        return event

    async def receive_text(self, timeout: float | None = None) -> str:
        """
        Receive text from the server.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            Text data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event was not a text message.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Wait for text until available.

                try:
                    text = await ws.receive_text()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for text for 2 seconds.

                try:
                    event = await ws.receive_text(timeout=2.)
                except TimeoutError:
                    print("No text received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        event = await self.receive(timeout)
        if isinstance(event, wsproto.events.TextMessage):
            return event.data
        raise WebSocketInvalidTypeReceived(event)

    async def receive_bytes(self, timeout: float | None = None) -> bytes:
        """
        Receive bytes from the server.

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.

        Returns:
            Bytes data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event was not a bytes message.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Wait for bytes until available.

                try:
                    data = await ws.receive_bytes()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for bytes for 2 seconds.

                try:
                    data = await ws.receive_bytes(timeout=2.)
                except TimeoutError:
                    print("No data received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        event = await self.receive(timeout)
        if isinstance(event, wsproto.events.BytesMessage):
            return bytes(event.data)
        raise WebSocketInvalidTypeReceived(event)

    async def receive_json(self, timeout: float | None = None, mode: JSONMode = "text") -> typing.Any:
        """
        Receive JSON data from the server.

        The received data should be parseable by [json.loads][json.loads].

        Args:
            timeout:
                Number of seconds to wait for an event.
                If `None`, will block until an event is available.
            mode:
                Receive mode. Should either be `'text'` or `'bytes'`.

        Returns:
            Parsed JSON data.

        Raises:
            TimeoutError: No event was received before the timeout delay.
            WebSocketDisconnect: The server closed the websocket.
            WebSocketNetworkError: A network error occured.
            WebSocketInvalidTypeReceived: The received event
                didn't correspond to the specified mode.

        Note:
            Exceptions not caught inside the context manager will be
            wrapped in an [ExceptionGroup][ExceptionGroup]. Use `except*` to catch them
            outside the `async with` block.

        Examples:
            Wait for data until available.

                try:
                    data = await ws.receive_json()
                except WebSocketDisconnect:
                    print("Connection closed")

            Wait for data for 2 seconds.

                try:
                    data = await ws.receive_json(timeout=2.)
                except TimeoutError:
                    print("No data received.")
                except WebSocketDisconnect:
                    print("Connection closed")
        """
        assert mode in ["text", "binary"]
        data: str | bytes
        if mode == "text":
            data = await self.receive_text(timeout)
        elif mode == "binary":
            data = await self.receive_bytes(timeout)
        return json.loads(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        """
        Close the WebSocket session.

        Internally, it'll send the
        [CloseConnection][wsproto.events.CloseConnection] event.

        *This method is automatically called when exiting the context manager.*

        Args:
            code:
                The integer close code to indicate why the connection has closed.
            reason:
                Additional reasoning for why the connection has closed.

        Examples:
            Close the WebSocket session.

                await ws.close()
        """
        import httpcore2

        self._should_close.set()
        if self.connection.state not in {
            wsproto.connection.ConnectionState.LOCAL_CLOSING,
            wsproto.connection.ConnectionState.CLOSED,
        }:
            event = wsproto.events.CloseConnection(code, reason)
            data = self.connection.send(event)
            try:
                async with self._write_lock:
                    await self.stream.write(data)
            except httpcore2.WriteError:
                pass
        await self.stream.aclose()

    async def _background_receive(self, max_bytes: int) -> None:
        """
        Background task listening for data from the server.

        Internally, it'll:

        * Answer to Ping events.
        * Acknowledge Pong events.
        * Put other events in the [_events][_events]
        queue that'll eventually be consumed by the user.

        Args:
            max_bytes: The maximum chunk size to read at each iteration.
        """
        import httpcore2

        partial_message_buffer: str | bytes | None = None
        partial_message_size = 0
        try:
            while not self._should_close.is_set():
                data = await self._read_stream(max_bytes)
                self.connection.receive_data(data)
                for event in self.connection.events():
                    if isinstance(event, wsproto.events.Ping):
                        data = self.connection.send(event.response())
                        async with self._write_lock:
                            await self.stream.write(data)
                        continue
                    if isinstance(event, wsproto.events.Pong):
                        self._ping_manager.ack(event.payload)
                        continue
                    if isinstance(event, wsproto.events.CloseConnection):
                        self._should_close.set()
                    if isinstance(event, wsproto.events.Message):
                        partial_message_size += len(event.data.encode() if isinstance(event.data, str) else event.data)
                        if partial_message_size > max_bytes:
                            await self.close(CloseReason.MESSAGE_TOO_BIG, "Message too big")
                            await self._send_event.send(
                                WebSocketDisconnect(CloseReason.MESSAGE_TOO_BIG, "Message too big")
                            )
                            break
                        # Unfinished message: bufferize
                        if not event.message_finished:
                            if partial_message_buffer is None:
                                partial_message_buffer = event.data
                            else:
                                partial_message_buffer += event.data
                        # Finished message but no buffer: just emit the event
                        elif partial_message_buffer is None:
                            partial_message_size = 0
                            await self._send_event.send(event)
                        # Finished message with buffer: emit the full event
                        else:
                            event_type = type(event)
                            full_message_event = event_type(partial_message_buffer + event.data)
                            partial_message_buffer = None
                            partial_message_size = 0
                            await self._send_event.send(full_message_event)
                        continue
                    await self._send_event.send(event)
        except (httpcore2.ReadError, httpcore2.WriteError, EndOfStream):
            await self.close(CloseReason.INTERNAL_ERROR, "Stream error")
            await self._send_event.send(WebSocketNetworkError())

    async def _background_keepalive_ping(self, interval_seconds: float, timeout_seconds: float | None = None) -> None:
        while not self._should_close.is_set():
            await anyio.sleep(interval_seconds)

            try:
                pong_callback = await self.ping()
            # Connection is closing, exit the task
            except wsproto.utilities.LocalProtocolError:
                return

            if timeout_seconds is not None:
                try:
                    with anyio.fail_after(timeout_seconds):
                        await pong_callback.wait()
                except TimeoutError:
                    await self.close(CloseReason.INTERNAL_ERROR, "Keepalive ping timeout")
                    await self._send_event.send(WebSocketNetworkError())

    async def _read_stream(self, max_bytes: int) -> bytes:
        data = await self.stream.read(max_bytes)
        if data == b"":
            raise EndOfStream()
        return data


def _get_headers(
    subprotocols: list[str] | None,
) -> dict[str, typing.Any]:
    headers = {
        "connection": "upgrade",
        "upgrade": "websocket",
        "sec-websocket-key": base64.b64encode(secrets.token_bytes(16)).decode("utf-8"),
        "sec-websocket-version": "13",
    }
    if subprotocols is not None:
        headers["sec-websocket-protocol"] = ", ".join(subprotocols)
    return headers


class WebSocketClient(typing.Generic[SyncSession]):
    """
    A sync WebSocket client.

    This class provides an API for connecting to WebSocket.

    Attributes:
        client:
            HTTPX client to use.
        max_message_size_bytes:
            Maximum incoming message size in bytes.
            Larger messages, including fragmented messages whose
            cumulative size exceeds the limit, close the connection
            with `MESSAGE_TOO_BIG`.
            Defaults to 65 KiB.
        queue_size:
            Size of the queue where the received messages will be held
            until they are consumed.
            If the queue is full, the client will stop receive messages
            from the server until the queue has room available.
            Defaults to 512.
        keepalive_ping_interval_seconds:
            Interval at which the client will automatically send a Ping event
            to keep the connection alive. Set it to `None` to disable this mechanism.
            Defaults to 20 seconds.
        keepalive_ping_timeout_seconds:
            Maximum delay the client will wait for an answer to its Ping event.
            If the delay is exceeded,
            [WebSocketNetworkError][httpx_ws.WebSocketNetworkError]
            will be raised and the connection closed.
            Defaults to 20 seconds.
        session_class:
            The session class to use.
            Defaults to [WebSocketSession][httpx_ws.WebSocketSession].
    """

    def __init__(
        self,
        client: Client,
        *,
        max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
        keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
        session_class: type[SyncSession] = WebSocketSession,  # type: ignore[assignment]
    ) -> None:
        self.client = client
        self.max_message_size_bytes = max_message_size_bytes
        self.queue_size = queue_size
        self.keepalive_ping_interval_seconds = keepalive_ping_interval_seconds
        self.keepalive_ping_timeout_seconds = keepalive_ping_timeout_seconds
        self.session_class = session_class

    @contextlib.contextmanager
    def connect(
        self,
        url: str,
        *,
        subprotocols: list[str] | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> typing.Generator[SyncSession, None, None]:
        """
        Start a sync WebSocket session.

        It returns a context manager that'll automatically
        call [close()][httpx_ws.WebSocketSession.close] when exiting.

        Args:
            url: The WebSocket URL.
            subprotocols:
                Optional list of subprotocols to negotiate with the server.
            params:
                Query parameters to include in the handshake request.
            headers:
                Headers to include in the handshake request.
            cookies:
                Cookies to include in the handshake request.
            auth:
                Authentication to use for the handshake request.
            follow_redirects:
                Whether to follow redirects on the handshake request.
            timeout:
                Timeout configuration for the handshake request.
            extensions:
                Request extensions for the handshake request.

        Returns:
            A [context manager][contextlib.AbstractContextManager]
                for [WebSocketSession][httpx_ws.WebSocketSession].

        Examples:
            Initialize the client and connect to a WebSocket.

                with httpx2.Client() as client:
                    ws_client = WebSocketClient(client)
                    with ws_client.connect("http://localhost:8000/ws") as ws:
                        message = ws.receive_text()
                        print(message)
                        ws.send_text("Hello!")
        """
        with self.client.stream(
            "GET",
            url,
            params=params,
            headers=Headers(headers) | _get_headers(subprotocols),
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as response:
            if response.status_code != 101:
                raise WebSocketUpgradeError(response)

            session = self.session_class(
                response.extensions["network_stream"],
                max_message_size_bytes=self.max_message_size_bytes,
                queue_size=self.queue_size,
                keepalive_ping_interval_seconds=self.keepalive_ping_interval_seconds,
                keepalive_ping_timeout_seconds=self.keepalive_ping_timeout_seconds,
                response=response,
            )
            with session:
                yield session


@contextlib.contextmanager
def connect_ws(
    url: str,
    client: Client | None = None,
    *,
    max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
    subprotocols: list[str] | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
    follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    extensions: RequestExtensions | None = None,
) -> typing.Generator[WebSocketSession, None, None]:
    """
    Start a sync WebSocket session.

    It returns a context manager that'll automatically
    call [close()][httpx_ws.WebSocketSession.close] when exiting.

    Args:
        url: The WebSocket URL.
        client:
            HTTPX client to use.
            If not provided, a default one will be initialized.
        max_message_size_bytes:
            Maximum incoming message size in bytes.
            Larger messages, including fragmented messages whose
            cumulative size exceeds the limit, close the connection
            with `MESSAGE_TOO_BIG`.
            Defaults to 65 KiB.
        queue_size:
            Size of the queue where the received messages will be held
            until they are consumed.
            If the queue is full, the client will stop receive messages
            from the server until the queue has room available.
            Defaults to 512.
        keepalive_ping_interval_seconds:
            Interval at which the client will automatically send a Ping event
            to keep the connection alive. Set it to `None` to disable this mechanism.
            Defaults to 20 seconds.
        keepalive_ping_timeout_seconds:
            Maximum delay the client will wait for an answer to its Ping event.
            If the delay is exceeded,
            [WebSocketNetworkError][httpx_ws.WebSocketNetworkError]
            will be raised and the connection closed.
            Defaults to 20 seconds.
        subprotocols:
            Optional list of subprotocols to negotiate with the server.
        params:
            Query parameters to include in the handshake request.
        headers:
            Headers to include in the handshake request.
        cookies:
            Cookies to include in the handshake request.
        auth:
            Authentication to use for the handshake request.
        follow_redirects:
            Whether to follow redirects on the handshake request.
        timeout:
            Timeout configuration for the handshake request.
        extensions:
            Request extensions for the handshake request.

    Returns:
        A [context manager][contextlib.AbstractContextManager]
            for [WebSocketSession][httpx_ws.WebSocketSession].

    Examples:
        Without explicit HTTPX client.

            with connect_ws("http://localhost:8000/ws") as ws:
                message = ws.receive_text()
                print(message)
                ws.send_text("Hello!")

        With explicit HTTPX client.

            with httpx2.Client() as client:
                with connect_ws("http://localhost:8000/ws", client) as ws:
                    message = ws.receive_text()
                    print(message)
                    ws.send_text("Hello!")
    """
    if client is None:
        from .._client import Client

        owned_client: contextlib.AbstractContextManager[Client] = Client()
    else:
        owned_client = contextlib.nullcontext(client)

    with owned_client as client:
        ws_client = WebSocketClient(
            client=client,
            max_message_size_bytes=max_message_size_bytes,
            queue_size=queue_size,
            keepalive_ping_interval_seconds=keepalive_ping_interval_seconds,
            keepalive_ping_timeout_seconds=keepalive_ping_timeout_seconds,
        )
        with ws_client.connect(
            url,
            subprotocols=subprotocols,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as websocket:
            yield websocket


class AsyncWebSocketClient(typing.Generic[AsyncSession]):
    """
    An async WebSocket client.

    This class provides an API for connecting to WebSocket.

    Attributes:
        client:
            HTTPX client to use.
        max_message_size_bytes:
            Maximum incoming message size in bytes.
            Larger messages, including fragmented messages whose
            cumulative size exceeds the limit, close the connection
            with `MESSAGE_TOO_BIG`.
            Defaults to 65 KiB.
        queue_size:
            Size of the queue where the received messages will be held
            until they are consumed.
            If the queue is full, the client will stop receive messages
            from the server until the queue has room available.
            Defaults to 512.
        keepalive_ping_interval_seconds:
            Interval at which the client will automatically send a Ping event
            to keep the connection alive. Set it to `None` to disable this mechanism.
            Defaults to 20 seconds.
        keepalive_ping_timeout_seconds:
            Maximum delay the client will wait for an answer to its Ping event.
            If the delay is exceeded,
            [WebSocketNetworkError][httpx_ws.WebSocketNetworkError]
            will be raised in an [ExceptionGroup][ExceptionGroup] and the connection closed.
            Defaults to 20 seconds.
        session_class:
            The session class to use.
            Defaults to [AsyncWebSocketSession][httpx_ws.AsyncWebSocketSession].
    """

    def __init__(
        self,
        client: AsyncClient,
        *,
        max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
        keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
        session_class: type[AsyncSession] = AsyncWebSocketSession,  # type: ignore[assignment]
    ) -> None:
        self.client = client
        self.max_message_size_bytes = max_message_size_bytes
        self.queue_size = queue_size
        self.keepalive_ping_interval_seconds = keepalive_ping_interval_seconds
        self.keepalive_ping_timeout_seconds = keepalive_ping_timeout_seconds
        self.session_class = session_class

    @contextlib.asynccontextmanager
    async def connect(
        self,
        url: str,
        *,
        subprotocols: list[str] | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> typing.AsyncGenerator[AsyncSession, None]:
        """
        Start an async WebSocket session.

        It returns an async context manager that'll automatically
        call [close()][httpx_ws.AsyncWebSocketSession.close] when exiting.

        Args:
            url: The WebSocket URL.
            subprotocols:
                Optional list of subprotocols to negotiate with the server.
            params:
                Query parameters to include in the handshake request.
            headers:
                Headers to include in the handshake request.
            cookies:
                Cookies to include in the handshake request.
            auth:
                Authentication to use for the handshake request.
            follow_redirects:
                Whether to follow redirects on the handshake request.
            timeout:
                Timeout configuration for the handshake request.
            extensions:
                Request extensions for the handshake request.

        Returns:
            An [async context manager][contextlib.AbstractAsyncContextManager]
                for [AsyncWebSocketSession][httpx_ws.AsyncWebSocketSession].

        Examples:
            Initialize the client and connect to a WebSocket.

                async with httpx2.AsyncClient() as client:
                    ws_client = AsyncWebSocketClient(client)
                    async with ws_client.connect("http://localhost:8000/ws") as ws:
                        message = await ws.receive_text()
                        print(message)
                        await ws.send_text("Hello!")
        """
        async with self.client.stream(
            "GET",
            url,
            params=params,
            headers=Headers(headers) | _get_headers(subprotocols),
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as response:
            if response.status_code != 101:
                raise WebSocketUpgradeError(response)

            session = self.session_class(
                response.extensions["network_stream"],
                max_message_size_bytes=self.max_message_size_bytes,
                queue_size=self.queue_size,
                keepalive_ping_interval_seconds=self.keepalive_ping_interval_seconds,
                keepalive_ping_timeout_seconds=self.keepalive_ping_timeout_seconds,
                response=response,
            )
            async with session:
                yield session


@contextlib.asynccontextmanager
async def aconnect_ws(
    url: str,
    client: AsyncClient | None = None,
    *,
    max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
    subprotocols: list[str] | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
    follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    extensions: RequestExtensions | None = None,
) -> typing.AsyncGenerator[AsyncWebSocketSession, None]:
    """
    Start an async WebSocket session.

    It returns an async context manager that'll automatically
    call [close()][httpx_ws.AsyncWebSocketSession.close] when exiting.

    Args:
        url: The WebSocket URL.
        client:
            HTTPX client to use.
            If not provided, a default one will be initialized.
        max_message_size_bytes:
            Maximum incoming message size in bytes.
            Larger messages, including fragmented messages whose
            cumulative size exceeds the limit, close the connection
            with `MESSAGE_TOO_BIG`.
            Defaults to 65 KiB.
        queue_size:
            Size of the queue where the received messages will be held
            until they are consumed.
            If the queue is full, the client will stop receive messages
            from the server until the queue has room available.
            Defaults to 512.
        keepalive_ping_interval_seconds:
            Interval at which the client will automatically send a Ping event
            to keep the connection alive. Set it to `None` to disable this mechanism.
            Defaults to 20 seconds.
        keepalive_ping_timeout_seconds:
            Maximum delay the client will wait for an answer to its Ping event.
            If the delay is exceeded,
            [WebSocketNetworkError][httpx_ws.WebSocketNetworkError]
            will be raised in an [ExceptionGroup][ExceptionGroup] and the connection closed.
            Defaults to 20 seconds.
        subprotocols:
            Optional list of subprotocols to negotiate with the server.
        params:
            Query parameters to include in the handshake request.
        headers:
            Headers to include in the handshake request.
        cookies:
            Cookies to include in the handshake request.
        auth:
            Authentication to use for the handshake request.
        follow_redirects:
            Whether to follow redirects on the handshake request.
        timeout:
            Timeout configuration for the handshake request.
        extensions:
            Request extensions for the handshake request.

    Returns:
        An [async context manager][contextlib.AbstractAsyncContextManager]
            for [AsyncWebSocketSession][httpx_ws.AsyncWebSocketSession].

    Examples:
        Without explicit HTTPX client.

            async with aconnect_ws("http://localhost:8000/ws") as ws:
                message = await ws.receive_text()
                print(message)
                await ws.send_text("Hello!")

        With explicit HTTPX client.

            async with httpx2.AsyncClient() as client:
                async with aconnect_ws("http://localhost:8000/ws", client) as ws:
                    message = await ws.receive_text()
                    print(message)
                    await ws.send_text("Hello!")
    """
    if client is None:
        from .._client import AsyncClient

        owned_client: contextlib.AbstractAsyncContextManager[AsyncClient] = AsyncClient()
    else:
        owned_client = contextlib.nullcontext(client)

    async with owned_client as client:
        ws_client = AsyncWebSocketClient(
            client=client,
            max_message_size_bytes=max_message_size_bytes,
            queue_size=queue_size,
            keepalive_ping_interval_seconds=keepalive_ping_interval_seconds,
            keepalive_ping_timeout_seconds=keepalive_ping_timeout_seconds,
        )
        async with ws_client.connect(
            url,
            subprotocols=subprotocols,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as websocket:
            yield websocket
<|endoftext|>from __future__ import annotations

import secrets
import threading

import anyio


class PingManagerBase:
    def _generate_id(self) -> bytes:
        return secrets.token_bytes()


class PingManager(PingManagerBase):
    def __init__(self) -> None:
        self._pings: dict[bytes, threading.Event] = {}

    def create(self, ping_id: bytes | None = None) -> tuple[bytes, threading.Event]:
        ping_id = self._generate_id() if not ping_id else ping_id
        event = threading.Event()
        self._pings[ping_id] = event
        return ping_id, event

    def ack(self, ping_id: bytes | bytearray) -> None:
        event = self._pings.pop(bytes(ping_id), None)
        if event is not None:
            event.set()


class AsyncPingManager(PingManagerBase):
    def __init__(self) -> None:
        self._pings: dict[bytes, anyio.Event] = {}

    def create(self, ping_id: bytes | None = None) -> tuple[bytes, anyio.Event]:
        ping_id = self._generate_id() if not ping_id else ping_id
        event = anyio.Event()
        self._pings[ping_id] = event
        return ping_id, event

    def ack(self, ping_id: bytes | bytearray) -> None:
        event = self._pings.pop(bytes(ping_id), None)
        if event is not None:
            event.set()
<|endoftext|>"""
Custom transports, with nicely configured defaults.

The following additional keyword arguments are currently supported by httpcore...

* uds: str
* local_address: str
* retries: int

Example usages...

# Disable HTTP/2 on a single specific domain.
mounts = {
    "all://": httpx2.HTTPTransport(http2=True),
    "all://*example.org": httpx2.HTTPTransport()
}

# Using advanced httpcore configuration, with connection retries.
transport = httpx2.HTTPTransport(retries=1)
client = httpx2.Client(transport=transport)

# Using advanced httpcore configuration, with unix domain sockets.
transport = httpx2.HTTPTransport(uds="socket.uds")
client = httpx2.Client(transport=transport)
"""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Generator
from types import TracebackType

if typing.TYPE_CHECKING:  # pragma: no cover
    import ssl

    import httpx2

from .._config import DEFAULT_LIMITS, Limits, Proxy, create_ssl_context
from .._exceptions import (
    ConnectError,
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
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport

T = typing.TypeVar("T", bound="HTTPTransport")
A = typing.TypeVar("A", bound="AsyncHTTPTransport")

SOCKET_OPTION = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]

__all__ = ["AsyncHTTPTransport", "HTTPTransport"]

HTTPCORE_EXC_MAP: dict[type[Exception], type[httpx2.HTTPError]] = {}


def _load_httpcore_exceptions() -> dict[type[Exception], type[httpx2.HTTPError]]:
    import httpcore2

    return {
        httpcore2.TimeoutException: TimeoutException,
        httpcore2.ConnectTimeout: ConnectTimeout,
        httpcore2.ReadTimeout: ReadTimeout,
        httpcore2.WriteTimeout: WriteTimeout,
        httpcore2.PoolTimeout: PoolTimeout,
        httpcore2.NetworkError: NetworkError,
        httpcore2.ConnectError: ConnectError,
        httpcore2.ReadError: ReadError,
        httpcore2.WriteError: WriteError,
        httpcore2.ProxyError: ProxyError,
        httpcore2.UnsupportedProtocol: UnsupportedProtocol,
        httpcore2.ProtocolError: ProtocolError,
        httpcore2.LocalProtocolError: LocalProtocolError,
        httpcore2.RemoteProtocolError: RemoteProtocolError,
    }


@contextlib.contextmanager
def map_httpcore_exceptions() -> Generator[None]:
    global HTTPCORE_EXC_MAP
    if len(HTTPCORE_EXC_MAP) == 0:
        HTTPCORE_EXC_MAP = _load_httpcore_exceptions()
    try:
        yield
    except Exception as exc:
        mapped_exc = None

        for from_exc, to_exc in HTTPCORE_EXC_MAP.items():
            if not isinstance(exc, from_exc):
                continue
            # We want to map to the most specific exception we can find.
            # Eg if `exc` is an `httpcore2.ReadTimeout`, we want to map to
            # `httpx2.ReadTimeout`, not just `httpx2.TimeoutException`.
            if mapped_exc is None or issubclass(to_exc, mapped_exc):
                mapped_exc = to_exc

        if mapped_exc is None:  # pragma: no cover
            raise

        message = str(exc)
        raise mapped_exc(message) from exc


class ResponseStream(SyncByteStream):
    def __init__(self, httpcore_stream: typing.Iterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    def __iter__(self) -> typing.Iterator[bytes]:
        with map_httpcore_exceptions():
            yield from self._httpcore_stream

    def close(self) -> None:
        if hasattr(self._httpcore_stream, "close"):
            self._httpcore_stream.close()


class HTTPTransport(BaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        import httpcore2

        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
        ssl_context = create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)

        if proxy is None:
            self._pool = httpcore2.ConnectionPool(
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                uds=uds,
                local_address=local_address,
                retries=retries,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("http", "https"):
            self._pool = httpcore2.HTTPProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                proxy_headers=proxy.headers.raw,
                ssl_context=ssl_context,
                proxy_ssl_context=proxy.ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                ) from None

            self._pool = httpcore2.SOCKSProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
            )
        else:  # pragma: no cover
            raise ValueError(
                f"Proxy protocol must be either 'http', 'https', 'socks5', or 'socks5h', but got {proxy.url.scheme!r}."
            )

    def __enter__(self: T) -> T:  # Use generics for subclass support.
        self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with map_httpcore_exceptions():
            self._pool.__exit__(exc_type, exc_value, traceback)

    def handle_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, SyncByteStream)
        import httpcore2

        req = httpcore2.Request(
            method=request.method,
            url=httpcore2.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with map_httpcore_exceptions():
            resp = self._pool.handle_request(req)

        assert isinstance(resp.stream, typing.Iterable)

        return Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=ResponseStream(resp.stream),
            extensions=resp.extensions,
        )

    def close(self) -> None:
        self._pool.close()


class AsyncResponseStream(AsyncByteStream):
    def __init__(self, httpcore_stream: typing.AsyncIterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        with map_httpcore_exceptions():
            async for part in self._httpcore_stream:
                yield part

    async def aclose(self) -> None:
        if hasattr(self._httpcore_stream, "aclose"):
            await self._httpcore_stream.aclose()


class AsyncHTTPTransport(AsyncBaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        import httpcore2

        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
        ssl_context = create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)

        if proxy is None:
            self._pool = httpcore2.AsyncConnectionPool(
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                uds=uds,
                local_address=local_address,
                retries=retries,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("http", "https"):
            self._pool = httpcore2.AsyncHTTPProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                proxy_headers=proxy.headers.raw,
                proxy_ssl_context=proxy.ssl_context,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                ) from None

            self._pool = httpcore2.AsyncSOCKSProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
            )
        else:  # pragma: no cover
            raise ValueError(
                f"Proxy protocol must be either 'http', 'https', 'socks5', or 'socks5h', but got {proxy.url.scheme!r}."
            )

    async def __aenter__(self: A) -> A:  # Use generics for subclass support.
        await self._pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with map_httpcore_exceptions():
            await self._pool.__aexit__(exc_type, exc_value, traceback)

    async def handle_async_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, AsyncByteStream)
        import httpcore2

        req = httpcore2.Request(
            method=request.method,
            url=httpcore2.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with map_httpcore_exceptions():
            resp = await self._pool.handle_async_request(req)

        assert isinstance(resp.stream, typing.AsyncIterable)

        return Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=AsyncResponseStream(resp.stream),
            extensions=resp.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()
<|endoftext|>import sys

from .asgi import ASGITransport
from .base import AsyncBaseTransport, BaseTransport
from .mock import MockTransport
from .wsgi import WSGITransport

if sys.platform == "emscripten":  # pragma: nocover
    if sys.version_info < (3, 12):
        raise RuntimeError("httpx2 requires Python 3.12+ on Emscripten.")

    from httpx2_jsfetch import (
        AsyncJavascriptFetchTransport as AsyncHTTPTransport,
        JavascriptFetchTransport as HTTPTransport,
    )
else:
    from .default import AsyncHTTPTransport, HTTPTransport

__all__ = [
    "ASGITransport",
    "AsyncBaseTransport",
    "BaseTransport",
    "AsyncHTTPTransport",
    "HTTPTransport",
    "MockTransport",
    "WSGITransport",
]
<|endoftext|>from __future__ import annotations

import typing

from .._models import Request, Response
from .base import AsyncBaseTransport, BaseTransport

SyncHandler = typing.Callable[[Request], Response]
AsyncHandler = typing.Callable[[Request], typing.Coroutine[None, None, Response]]


__all__ = ["MockTransport"]


class MockTransport(AsyncBaseTransport, BaseTransport):
    def __init__(self, handler: SyncHandler | AsyncHandler) -> None:
        self.handler = handler

    def handle_request(
        self,
        request: Request,
    ) -> Response:
        request.read()
        response = self.handler(request)
        if not isinstance(response, Response):  # pragma: no cover
            raise TypeError("Cannot use an async handler in a sync Client")
        return response

    async def handle_async_request(self, request: Request) -> Response:
        await request.aread()
        response = self.handler(request)

        # Allow handler to *optionally* be an `async` function.
        # If it is, then the `response` variable need to be awaited to actually
        # return the result.

        if not isinstance(response, Response):
            response = await response

        return response
<|endoftext|>from __future__ import annotations

import typing

from .._models import Request, Response
from .._types import AsyncByteStream
from .base import AsyncBaseTransport

if typing.TYPE_CHECKING:
    import asyncio

    import trio

    Event = asyncio.Event | trio.Event


_Message = typing.MutableMapping[str, typing.Any]
_Receive = typing.Callable[[], typing.Awaitable[_Message]]
_Send = typing.Callable[[typing.MutableMapping[str, typing.Any]], typing.Awaitable[None]]
_ASGIApp = typing.Callable[[typing.MutableMapping[str, typing.Any], _Receive, _Send], typing.Awaitable[None]]

__all__ = ["ASGITransport"]


def is_running_trio() -> bool:
    try:
        # sniffio is a dependency of trio.

        # See https://github.com/python-trio/trio/issues/2802
        import sniffio

        if sniffio.current_async_library() == "trio":
            return True
    except ImportError:  # pragma: no cover
        pass

    return False


def create_event() -> Event:
    if is_running_trio():
        import trio

        return trio.Event()

    import asyncio

    return asyncio.Event()


class ASGIResponseStream(AsyncByteStream):
    def __init__(self, body: list[bytes]) -> None:
        self._body = body

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b"".join(self._body)


class ASGITransport(AsyncBaseTransport):
    """
    A custom AsyncTransport that handles sending requests directly to an ASGI app.

    ```python
    transport = httpx2.ASGITransport(
        app=app,
        root_path="/submount",
        client=("1.2.3.4", 123)
    )
    client = httpx2.AsyncClient(transport=transport)
    ```

    Arguments:
        app: The ASGI application.
        raise_app_exceptions: Boolean indicating if exceptions in the application
            should be raised. Default to `True`. Can be set to `False` for use cases
            such as testing the content of a client 500 response.
        root_path: The root path on which the ASGI application should be mounted.
        client: A two-tuple indicating the client IP and port of incoming requests.
    """

    def __init__(
        self,
        app: _ASGIApp,
        raise_app_exceptions: bool = True,
        root_path: str = "",
        client: tuple[str, int] = ("127.0.0.1", 123),
    ) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions
        self.root_path = root_path
        self.client = client

    async def handle_async_request(self, request: Request) -> Response:
        assert isinstance(request.stream, AsyncByteStream)

        # ASGI scope.
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": self.client,
            "root_path": self.root_path,
        }

        # Request.
        request_body_chunks = request.stream.__aiter__()
        request_complete = False

        # Response.
        status_code = None
        response_headers = None
        body_parts: list[bytes] = []
        response_started = False
        response_complete = create_event()

        # ASGI callables.

        async def receive() -> dict[str, typing.Any]:
            nonlocal request_complete

            if request_complete:
                await response_complete.wait()
                return {"type": "http.disconnect"}

            try:
                body = await request_body_chunks.__anext__()
            except StopAsyncIteration:
                request_complete = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": body, "more_body": True}

        async def send(message: typing.MutableMapping[str, typing.Any]) -> None:
            nonlocal status_code, response_headers, response_started

            if message["type"] == "http.response.start":
                assert not response_started

                status_code = message["status"]
                response_headers = message.get("headers", [])
                response_started = True

            elif message["type"] == "http.response.body":
                assert not response_complete.is_set()
                body = message.get("body", b"")
                more_body = message.get("more_body", False)

                if body and request.method != "HEAD":
                    body_parts.append(body)

                if not more_body:
                    response_complete.set()

        try:
            await self.app(scope, receive, send)
        except Exception:
            if self.raise_app_exceptions:
                raise

            response_complete.set()
            if status_code is None:
                status_code = 500
            if response_headers is None:
                response_headers = {}

        assert response_complete.is_set()
        assert status_code is not None
        assert response_headers is not None

        stream = ASGIResponseStream(body_parts)

        return Response(status_code, headers=response_headers, stream=stream)
<|endoftext|>from __future__ import annotations

import io
import itertools
import sys
import typing

from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport

if typing.TYPE_CHECKING:
    from _typeshed import OptExcInfo  # pragma: no cover
    from _typeshed.wsgi import WSGIApplication  # pragma: no cover

_T = typing.TypeVar("_T")


__all__ = ["WSGITransport"]


def _skip_leading_empty_chunks(body: typing.Iterable[_T]) -> typing.Iterable[_T]:
    body = iter(body)
    for chunk in body:
        if chunk:
            return itertools.chain([chunk], body)
    return []


class WSGIByteStream(SyncByteStream):
    def __init__(self, result: typing.Iterable[bytes]) -> None:
        self._close = getattr(result, "close", None)
        self._result = _skip_leading_empty_chunks(result)

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from self._result

    def close(self) -> None:
        if self._close is not None:
            self._close()


class WSGITransport(BaseTransport):
    """
    A custom transport that handles sending requests directly to an WSGI app.
    The simplest way to use this functionality is to use the `app` argument.

    ```
    client = httpx2.Client(app=app)
    ```

    Alternatively, you can setup the transport instance explicitly.
    This allows you to include any additional configuration arguments specific
    to the WSGITransport class:

    ```
    transport = httpx2.WSGITransport(
        app=app,
        script_name="/submount",
        remote_addr="1.2.3.4"
    )
    client = httpx2.Client(transport=transport)
    ```

    Arguments:
        app: The WSGI application.
        raise_app_exceptions: Boolean indicating if exceptions in the application
            should be raised. Default to `True`. Can be set to `False` for use cases
            such as testing the content of a client 500 response.
        script_name: The root path on which the WSGI application should be mounted.
        remote_addr: A string indicating the client IP of incoming requests.
    """

    def __init__(
        self,
        app: WSGIApplication,
        raise_app_exceptions: bool = True,
        script_name: str = "",
        remote_addr: str = "127.0.0.1",
        wsgi_errors: typing.TextIO | None = None,
    ) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions
        self.script_name = script_name
        self.remote_addr = remote_addr
        self.wsgi_errors = wsgi_errors

    def handle_request(self, request: Request) -> Response:
        request.read()
        wsgi_input = io.BytesIO(request.content)

        port = request.url.port or {"http": 80, "https": 443}[request.url.scheme]
        environ = {
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": request.url.scheme,
            "wsgi.input": wsgi_input,
            "wsgi.errors": self.wsgi_errors or sys.stderr,
            "wsgi.multithread": True,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "REQUEST_METHOD": request.method,
            "SCRIPT_NAME": self.script_name,
            "PATH_INFO": request.url.path,
            "QUERY_STRING": request.url.query.decode("ascii"),
            "SERVER_NAME": request.url.host,
            "SERVER_PORT": str(port),
            "SERVER_PROTOCOL": "HTTP/1.1",
            "REMOTE_ADDR": self.remote_addr,
        }
        if request.content and "Content-Length" not in request.headers:
            environ["CONTENT_LENGTH"] = str(len(request.content))
        for header_key, header_value in request.headers.raw:
            key = header_key.decode("ascii").upper().replace("-", "_")
            if key not in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                key = "HTTP_" + key
            environ[key] = header_value.decode("ascii")

        seen_status = None
        seen_response_headers = None
        seen_exc_info = None

        def start_response(
            status: str,
            response_headers: list[tuple[str, str]],
            exc_info: OptExcInfo | None = None,
        ) -> typing.Callable[[bytes], typing.Any]:
            nonlocal seen_status, seen_response_headers, seen_exc_info
            seen_status = status
            seen_response_headers = response_headers
            seen_exc_info = exc_info
            return lambda _: None

        result = self.app(environ, start_response)

        stream = WSGIByteStream(result)

        assert seen_status is not None
        assert seen_response_headers is not None
        if seen_exc_info and seen_exc_info[0] and self.raise_app_exceptions:
            raise seen_exc_info[1]

        status_code = int(seen_status.split()[0])
        headers = [(key.encode("ascii"), value.encode("ascii")) for key, value in seen_response_headers]

        return Response(status_code, headers=headers, stream=stream)
<|endoftext|>from __future__ import annotations

import typing
from types import TracebackType

from .._models import Request, Response

# TODO(Marcelo): When Python 3.10 reaches EOF, we can use `typing.Self` instead of defining those two.
T = typing.TypeVar("T", bound="BaseTransport")
A = typing.TypeVar("A", bound="AsyncBaseTransport")

__all__ = ["AsyncBaseTransport", "BaseTransport"]


class BaseTransport:
    def __enter__(self: T) -> T:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self.close()

    def handle_request(self, request: Request) -> Response:
        """
        Send a single HTTP request and return a response.

        Developers shouldn't typically ever need to call into this API directly,
        since the Client class provides all the higher level user-facing API
        niceties.

        In order to properly release any network resources, the response
        stream should *either* be consumed immediately, with a call to
        `response.stream.read()`, or else the `handle_request` call should
        be followed with a try/finally block to ensuring the stream is
        always closed.

        Example usage:

            with httpx2.HTTPTransport() as transport:
                req = httpx2.Request(
                    method=b"GET",
                    url=(b"https", b"www.example.com", 443, b"/"),
                    headers=[(b"Host", b"www.example.com")],
                )
                resp = transport.handle_request(req)
                body = resp.stream.read()
                print(resp.status_code, resp.headers, body)


        Takes a `Request` instance as the only argument.

        Returns a `Response` instance.
        """
        raise NotImplementedError("The 'handle_request' method must be implemented.")  # pragma: no cover

    def close(self) -> None:
        pass


class AsyncBaseTransport:
    async def __aenter__(self: A) -> A:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self.aclose()

    async def handle_async_request(
        self,
        request: Request,
    ) -> Response:
        raise NotImplementedError("The 'handle_async_request' method must be implemented.")  # pragma: no cover

    async def aclose(self) -> None:
        pass
<|endoftext|>from __future__ import annotations


class UserAgent:
    """Represents a parsed user agent header value.

    The default implementation does no parsing, only the :attr:`string`
    attribute is set. A subclass may parse the string to set the
    common attributes or expose other information. Set
    :attr:`werkzeug.wrappers.Request.user_agent_class` to use a
    subclass.

    :param string: The header value to parse.

    .. versionadded:: 2.0
        This replaces the previous ``useragents`` module, but does not
        provide a built-in parser.
    """

    platform: str | None = None
    """The OS name, if it could be parsed from the string."""

    browser: str | None = None
    """The browser name, if it could be parsed from the string."""

    version: str | None = None
    """The browser version, if it could be parsed from the string."""

    language: str | None = None
    """The browser language, if it could be parsed from the string."""

    def __init__(self, string: str) -> None:
        self.string: str = string
        """The original header value."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.browser}/{self.version}>"

    def __str__(self) -> str:
        return self.string

    def __bool__(self) -> bool:
        return bool(self.browser)

    def to_header(self) -> str:
        """Convert to a header value."""
        return self.string
<|endoftext|>from __future__ import annotations

import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter

from .wsgi import ClosingIterator

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

T = t.TypeVar("T")
F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def release_local(local: Local | LocalStack[t.Any]) -> None:
    """Release the data for the current context in a :class:`Local` or
    :class:`LocalStack` without using a :class:`LocalManager`.

    This should not be needed for modern use cases, and may be removed
    in the future.

    .. versionadded:: 0.6.1
    """
    local.__release_local__()


class Local:
    """Create a namespace of context-local data. This wraps a
    :class:`ContextVar` containing a :class:`dict` value.

    This may incur a performance penalty compared to using individual
    context vars, as it has to copy data to avoid mutating the dict
    between nested contexts.

    :param context_var: The :class:`~contextvars.ContextVar` to use as
        storage for this local. If not given, one will be created.
        Context vars not created at the global scope may interfere with
        garbage collection.

    .. versionchanged:: 2.0
        Uses ``ContextVar`` instead of a custom storage implementation.
    """

    __slots__ = ("__storage",)

    def __init__(self, context_var: ContextVar[dict[str, t.Any]] | None = None) -> None:
        if context_var is None:
            # A ContextVar not created at global scope interferes with
            # Python's garbage collection. However, a local only makes
            # sense defined at the global scope as well, in which case
            # the GC issue doesn't seem relevant.
            context_var = ContextVar(f"werkzeug.Local<{id(self)}>.storage")

        object.__setattr__(self, "_Local__storage", context_var)

    def __iter__(self) -> t.Iterator[tuple[str, t.Any]]:
        return iter(self.__storage.get({}).items())

    def __call__(
        self, name: str, *, unbound_message: str | None = None
    ) -> LocalProxy[t.Any]:
        """Create a :class:`LocalProxy` that access an attribute on this
        local namespace.

        :param name: Proxy this attribute.
        :param unbound_message: The error message that the proxy will
            show if the attribute isn't set.
        """
        return LocalProxy(self, name, unbound_message=unbound_message)

    def __release_local__(self) -> None:
        self.__storage.set({})

    def __getattr__(self, name: str) -> t.Any:
        values = self.__storage.get({})

        if name in values:
            return values[name]

        raise AttributeError(name)

    def __setattr__(self, name: str, value: t.Any) -> None:
        values = self.__storage.get({}).copy()
        values[name] = value
        self.__storage.set(values)

    def __delattr__(self, name: str) -> None:
        values = self.__storage.get({})

        if name in values:
            values = values.copy()
            del values[name]
            self.__storage.set(values)
        else:
            raise AttributeError(name)


class LocalStack(t.Generic[T]):
    """Create a stack of context-local data. This wraps a
    :class:`ContextVar` containing a :class:`list` value.

    This may incur a performance penalty compared to using individual
    context vars, as it has to copy data to avoid mutating the list
    between nested contexts.

    :param context_var: The :class:`~contextvars.ContextVar` to use as
        storage for this local. If not given, one will be created.
        Context vars not created at the global scope may interfere with
        garbage collection.

    .. versionchanged:: 2.0
        Uses ``ContextVar`` instead of a custom storage implementation.

    .. versionadded:: 0.6.1
    """

    __slots__ = ("_storage",)

    def __init__(self, context_var: ContextVar[list[T]] | None = None) -> None:
        if context_var is None:
            # A ContextVar not created at global scope interferes with
            # Python's garbage collection. However, a local only makes
            # sense defined at the global scope as well, in which case
            # the GC issue doesn't seem relevant.
            context_var = ContextVar(f"werkzeug.LocalStack<{id(self)}>.storage")

        self._storage = context_var

    def __release_local__(self) -> None:
        self._storage.set([])

    def push(self, obj: T) -> list[T]:
        """Add a new item to the top of the stack."""
        stack = self._storage.get([]).copy()
        stack.append(obj)
        self._storage.set(stack)
        return stack

    def pop(self) -> T | None:
        """Remove the top item from the stack and return it. If the
        stack is empty, return ``None``.
        """
        stack = self._storage.get([])

        if len(stack) == 0:
            return None

        rv = stack[-1]
        self._storage.set(stack[:-1])
        return rv

    @property
    def top(self) -> T | None:
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
        stack = self._storage.get([])

        if len(stack) == 0:
            return None

        return stack[-1]

    def __call__(
        self, name: str | None = None, *, unbound_message: str | None = None
    ) -> LocalProxy[t.Any]:
        """Create a :class:`LocalProxy` that accesses the top of this
        local stack.

        :param name: If given, the proxy access this attribute of the
            top item, rather than the item itself.
        :param unbound_message: The error message that the proxy will
            show if the stack is empty.
        """
        return LocalProxy(self, name, unbound_message=unbound_message)


class LocalManager:
    """Manage releasing the data for the current context in one or more
    :class:`Local` and :class:`LocalStack` objects.

    This should not be needed for modern use cases, and may be removed
    in the future.

    :param locals: A local or list of locals to manage.

    .. versionchanged:: 2.1
        The ``ident_func`` was removed.

    .. versionchanged:: 0.7
        The ``ident_func`` parameter was added.

    .. versionchanged:: 0.6.1
        The :func:`release_local` function can be used instead of a
        manager.
    """

    __slots__ = ("locals",)

    def __init__(
        self,
        locals: None
        | (Local | LocalStack[t.Any] | t.Iterable[Local | LocalStack[t.Any]]) = None,
    ) -> None:
        if locals is None:
            self.locals = []
        elif isinstance(locals, Local):
            self.locals = [locals]
        else:
            self.locals = list(locals)  # type: ignore[arg-type]

    def cleanup(self) -> None:
        """Release the data in the locals for this context. Call this at
        the end of each request or use :meth:`make_middleware`.
        """
        for local in self.locals:
            release_local(local)

    def make_middleware(self, app: WSGIApplication) -> WSGIApplication:
        """Wrap a WSGI application so that local data is released
        automatically after the response has been sent for a request.
        """

        def application(
            environ: WSGIEnvironment, start_response: StartResponse
        ) -> t.Iterable[bytes]:
            return ClosingIterator(app(environ, start_response), self.cleanup)

        return application

    def middleware(self, func: WSGIApplication) -> WSGIApplication:
        """Like :meth:`make_middleware` but used as a decorator on the
        WSGI application function.

        .. code-block:: python

            @manager.middleware
            def application(environ, start_response):
                ...
        """
        return update_wrapper(self.make_middleware(func), func)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} storages: {len(self.locals)}>"


class _ProxyLookup:
    """Descriptor that handles proxied attribute lookup for
    :class:`LocalProxy`.

    :param f: The built-in function this attribute is accessed through.
        Instead of looking up the special method, the function call
        is redone on the object.
    :param fallback: Return this function if the proxy is unbound
        instead of raising a :exc:`RuntimeError`.
    :param is_attr: This proxied name is an attribute, not a function.
        Call the fallback immediately to get the value.
    :param class_value: Value to return when accessed from the
        ``LocalProxy`` class directly. Used for ``__doc__`` so building
        docs still works.
    """

    __slots__ = ("bind_f", "fallback", "is_attr", "class_value", "name")

    def __init__(
        self,
        f: t.Callable[..., t.Any] | None = None,
        fallback: t.Callable[[LocalProxy[t.Any]], t.Any] | None = None,
        class_value: t.Any | None = None,
        is_attr: bool = False,
    ) -> None:
        bind_f: t.Callable[[LocalProxy[t.Any], t.Any], t.Callable[..., t.Any]] | None

        if hasattr(f, "__get__"):
            # A Python function, can be turned into a bound method.

            def bind_f(
                instance: LocalProxy[t.Any], obj: t.Any
            ) -> t.Callable[..., t.Any]:
                return f.__get__(obj, type(obj))  # type: ignore

        elif f is not None:
            # A C function, use partial to bind the first argument.

            def bind_f(
                instance: LocalProxy[t.Any], obj: t.Any
            ) -> t.Callable[..., t.Any]:
                return partial(f, obj)

        else:
            # Use getattr, which will produce a bound method.
            bind_f = None

        self.bind_f = bind_f
        self.fallback = fallback
        self.class_value = class_value
        self.is_attr = is_attr

    def __set_name__(self, owner: LocalProxy[t.Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: LocalProxy[t.Any], owner: type | None = None) -> t.Any:
        if instance is None:
            if self.class_value is not None:
                return self.class_value

            return self

        try:
            obj = instance._get_current_object()
        except RuntimeError:
            if self.fallback is None:
                raise

            fallback = self.fallback.__get__(instance, owner)

            if self.is_attr:
                # __class__ and __doc__ are attributes, not methods.
                # Call the fallback to get the value.
                return fallback()

            return fallback

        if self.bind_f is not None:
            return self.bind_f(instance, obj)

        return getattr(obj, self.name)

    def __repr__(self) -> str:
        return f"proxy {self.name}"

    def __call__(
        self, instance: LocalProxy[t.Any], *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        """Support calling unbound methods from the class. For example,
        this happens with ``copy.copy``, which does
        ``type(x).__copy__(x)``. ``type(x)`` can't be proxied, so it
        returns the proxy type and descriptor.
        """
        return self.__get__(instance, type(instance))(*args, **kwargs)


class _ProxyIOp(_ProxyLookup):
    """Look up an augmented assignment method on a proxied object. The
    method is wrapped to return the proxy instead of the object.
    """

    __slots__ = ()

    def __init__(
        self,
        f: t.Callable[..., t.Any] | None = None,
        fallback: t.Callable[[LocalProxy[t.Any]], t.Any] | None = None,
    ) -> None:
        super().__init__(f, fallback)

        def bind_f(instance: LocalProxy[t.Any], obj: t.Any) -> t.Callable[..., t.Any]:
            def i_op(self: t.Any, other: t.Any) -> LocalProxy[t.Any]:
                f(self, other)  # type: ignore
                return instance

            return i_op.__get__(obj, type(obj))  # type: ignore

        self.bind_f = bind_f


def _l_to_r_op(op: F) -> F:
    """Swap the argument order to turn an l-op into an r-op."""

    def r_op(obj: t.Any, other: t.Any) -> t.Any:
        return op(other, obj)

    return t.cast(F, r_op)


def _identity(o: T) -> T:
    return o


class LocalProxy(t.Generic[T]):
    """A proxy to the object bound to a context-local object. All
    operations on the proxy are forwarded to the bound object. If no
    object is bound, a ``RuntimeError`` is raised.

    :param local: The context-local object that provides the proxied
        object.
    :param name: Proxy this attribute from the proxied object.
    :param unbound_message: The error message to show if the
        context-local object is unbound.

    Proxy a :class:`~contextvars.ContextVar` to make it easier to
    access. Pass a name to proxy that attribute.

    .. code-block:: python

        _request_var = ContextVar("request")
        request = LocalProxy(_request_var)
        session = LocalProxy(_request_var, "session")

    Proxy an attribute on a :class:`Local` namespace by calling the
    local with the attribute name:

    .. code-block:: python

        data = Local()
        user = data("user")

    Proxy the top item on a :class:`LocalStack` by calling the local.
    Pass a name to proxy that attribute.

    .. code-block::

        app_stack = LocalStack()
        current_app = app_stack()
        g = app_stack("g")

    Pass a function to proxy the return value from that function. This
    was previously used to access attributes of local objects before
    that was supported directly.

    .. code-block:: python

        session = LocalProxy(lambda: request.session)

    ``__repr__`` and ``__class__`` are proxied, so ``repr(x)`` and
    ``isinstance(x, cls)`` will look like the proxied object. Use
    ``issubclass(type(x), LocalProxy)`` to check if an object is a
    proxy.

    .. code-block:: python

        repr(user)  # <User admin>
        isinstance(user, User)  # True
        issubclass(type(user), LocalProxy)  # True

    .. versionchanged:: 2.2.2
        ``__wrapped__`` is set when wrapping an object, not only when
        wrapping a function, to prevent doctest from failing.

    .. versionchanged:: 2.2
        Can proxy a ``ContextVar`` or ``LocalStack`` directly.

    .. versionchanged:: 2.2
        The ``name`` parameter can be used with any proxied object, not
        only ``Local``.

    .. versionchanged:: 2.2
        Added the ``unbound_message`` parameter.

    .. versionchanged:: 2.0
        Updated proxied attributes and methods to reflect the current
        data model.

    .. versionchanged:: 0.6.1
        The class can be instantiated with a callable.
    """

    __slots__ = ("__wrapped", "_get_current_object")

    _get_current_object: t.Callable[[], T]
    """Return the current object this proxy is bound to. If the proxy is
    unbound, this raises a ``RuntimeError``.

    This should be used if you need to pass the object to something that
    doesn't understand the proxy. It can also be useful for performance
    if you are accessing the object multiple times in a function, rather
    than going through the proxy multiple times.
    """

    def __init__(
        self,
        local: ContextVar[T] | Local | LocalStack[T] | t.Callable[[], T],
        name: str | None = None,
        *,
        unbound_message: str | None = None,
    ) -> None:
        if name is None:
            get_name = _identity
        else:
            get_name = attrgetter(name)  # type: ignore[assignment]

        if unbound_message is None:
            unbound_message = "object is not bound"

        if isinstance(local, Local):
            if name is None:
                raise TypeError("'name' is required when proxying a 'Local' object.")

            def _get_current_object() -> T:
                try:
                    return get_name(local)  # type: ignore[return-value]
                except AttributeError:
                    raise RuntimeError(unbound_message) from None

        elif isinstance(local, LocalStack):

            def _get_current_object() -> T:
                obj = local.top

                if obj is None:
                    raise RuntimeError(unbound_message)

                return get_name(obj)

        elif isinstance(local, ContextVar):

            def _get_current_object() -> T:
                try:
                    obj = local.get()
                except LookupError:
                    raise RuntimeError(unbound_message) from None

                return get_name(obj)

        elif callable(local):

            def _get_current_object() -> T:
                return get_name(local())

        else:
            raise TypeError(f"Don't know how to proxy '{type(local)}'.")

        object.__setattr__(self, "_LocalProxy__wrapped", local)
        object.__setattr__(self, "_get_current_object", _get_current_object)

    __doc__ = _ProxyLookup(
        class_value=__doc__, fallback=lambda self: type(self).__doc__, is_attr=True
    )
    __wrapped__ = _ProxyLookup(
        fallback=lambda self: self._LocalProxy__wrapped,  # type: ignore[attr-defined]
        is_attr=True,
    )
    # __del__ should only delete the proxy
    __repr__ = _ProxyLookup(
        repr, fallback=lambda self: f"<{type(self).__name__} unbound>"
    )
    __str__ = _ProxyLookup(str)
    __bytes__ = _ProxyLookup(bytes)
    __format__ = _ProxyLookup()
    __lt__ = _ProxyLookup(operator.lt)
    __le__ = _ProxyLookup(operator.le)
    __eq__ = _ProxyLookup(operator.eq)
    __ne__ = _ProxyLookup(operator.ne)
    __gt__ = _ProxyLookup(operator.gt)
    __ge__ = _ProxyLookup(operator.ge)
    __hash__ = _ProxyLookup(hash)
    __bool__ = _ProxyLookup(bool, fallback=lambda self: False)
    __getattr__ = _ProxyLookup(getattr)
    # __getattribute__ triggered through __getattr__
    __setattr__ = _ProxyLookup(setattr)
    __delattr__ = _ProxyLookup(delattr)
    __dir__ = _ProxyLookup(dir, fallback=lambda self: [])
    # __get__ (proxying descriptor not supported)
    # __set__ (descriptor)
    # __delete__ (descriptor)
    # __set_name__ (descriptor)
    # __objclass__ (descriptor)
    # __slots__ used by proxy itself
    # __dict__ (__getattr__)
    # __weakref__ (__getattr__)
    # __init_subclass__ (proxying metaclass not supported)
    # __prepare__ (metaclass)
    __class__ = _ProxyLookup(fallback=lambda self: type(self), is_attr=True)
    __instancecheck__ = _ProxyLookup(lambda self, other: isinstance(other, self))
    __subclasscheck__ = _ProxyLookup(lambda self, other: issubclass(other, self))
    # __class_getitem__ triggered through __getitem__
    __call__ = _ProxyLookup(lambda self, *args, **kwargs: self(*args, **kwargs))
    __len__ = _ProxyLookup(len)
    __length_hint__ = _ProxyLookup(operator.length_hint)
    __getitem__ = _ProxyLookup(operator.getitem)
    __setitem__ = _ProxyLookup(operator.setitem)
    __delitem__ = _ProxyLookup(operator.delitem)
    # __missing__ triggered through __getitem__
    __iter__ = _ProxyLookup(iter)
    __next__ = _ProxyLookup(next)
    __reversed__ = _ProxyLookup(reversed)
    __contains__ = _ProxyLookup(operator.contains)
    __add__ = _ProxyLookup(operator.add)
    __sub__ = _ProxyLookup(operator.sub)
    __mul__ = _ProxyLookup(operator.mul)
    __matmul__ = _ProxyLookup(operator.matmul)
    __truediv__ = _ProxyLookup(operator.truediv)
    __floordiv__ = _ProxyLookup(operator.floordiv)
    __mod__ = _ProxyLookup(operator.mod)
    __divmod__ = _ProxyLookup(divmod)
    __pow__ = _ProxyLookup(pow)
    __lshift__ = _ProxyLookup(operator.lshift)
    __rshift__ = _ProxyLookup(operator.rshift)
    __and__ = _ProxyLookup(operator.and_)
    __xor__ = _ProxyLookup(operator.xor)
    __or__ = _ProxyLookup(operator.or_)
    __radd__ = _ProxyLookup(_l_to_r_op(operator.add))
    __rsub__ = _ProxyLookup(_l_to_r_op(operator.sub))
    __rmul__ = _ProxyLookup(_l_to_r_op(operator.mul))
    __rmatmul__ = _ProxyLookup(_l_to_r_op(operator.matmul))
    __rtruediv__ = _ProxyLookup(_l_to_r_op(operator.truediv))
    __rfloordiv__ = _ProxyLookup(_l_to_r_op(operator.floordiv))
    __rmod__ = _ProxyLookup(_l_to_r_op(operator.mod))
    __rdivmod__ = _ProxyLookup(_l_to_r_op(divmod))
    __rpow__ = _ProxyLookup(_l_to_r_op(pow))
    __rlshift__ = _ProxyLookup(_l_to_r_op(operator.lshift))
    __rrshift__ = _ProxyLookup(_l_to_r_op(operator.rshift))
    __rand__ = _ProxyLookup(_l_to_r_op(operator.and_))
    __rxor__ = _ProxyLookup(_l_to_r_op(operator.xor))
    __ror__ = _ProxyLookup(_l_to_r_op(operator.or_))
    __iadd__ = _ProxyIOp(operator.iadd)
    __isub__ = _ProxyIOp(operator.isub)
    __imul__ = _ProxyIOp(operator.imul)
    __imatmul__ = _ProxyIOp(operator.imatmul)
    __itruediv__ = _ProxyIOp(operator.itruediv)
    __ifloordiv__ = _ProxyIOp(operator.ifloordiv)
    __imod__ = _ProxyIOp(operator.imod)
    __ipow__ = _ProxyIOp(operator.ipow)
    __ilshift__ = _ProxyIOp(operator.ilshift)
    __irshift__ = _ProxyIOp(operator.irshift)
    __iand__ = _ProxyIOp(operator.iand)
    __ixor__ = _ProxyIOp(operator.ixor)
    __ior__ = _ProxyIOp(operator.ior)
    __neg__ = _ProxyLookup(operator.neg)
    __pos__ = _ProxyLookup(operator.pos)
    __abs__ = _ProxyLookup(abs)
    __invert__ = _ProxyLookup(operator.invert)
    __complex__ = _ProxyLookup(complex)
    __int__ = _ProxyLookup(int)
    __float__ = _ProxyLookup(float)
    __index__ = _ProxyLookup(operator.index)
    __round__ = _ProxyLookup(round)
    __trunc__ = _ProxyLookup(math.trunc)
    __floor__ = _ProxyLookup(math.floor)
    __ceil__ = _ProxyLookup(math.ceil)
    __enter__ = _ProxyLookup()
    __exit__ = _ProxyLookup()
    __await__ = _ProxyLookup()
    __aiter__ = _ProxyLookup()
    __anext__ = _ProxyLookup()
    __aenter__ = _ProxyLookup()
    __aexit__ = _ProxyLookup()
    __copy__ = _ProxyLookup(copy.copy)
    __deepcopy__ = _ProxyLookup(copy.deepcopy)
    # __getnewargs_ex__ (pickle through proxy not supported)
    # __getnewargs__ (pickle)
    # __getstate__ (pickle)
    # __setstate__ (pickle)
    # __reduce__ (pickle)
    # __reduce_ex__ (pickle)
<|endoftext|>from __future__ import annotations

import hashlib
import hmac
import os
import posixpath
import secrets

SALT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_PBKDF2_ITERATIONS = 1_000_000

_os_alt_seps: list[str] = list(
    sep for sep in [os.sep, os.altsep] if sep is not None and sep != "/"
)
# https://chrisdenton.github.io/omnipath/Special%20Dos%20Device%20Names.html
_windows_device_files = {
    "AUX",
    "CON",
    "CONIN$",
    "CONOUT$",
    *(f"COM{c}" for c in "123456789¹²³"),
    *(f"LPT{c}" for c in "123456789¹²³"),
    "NUL",
    "PRN",
}


def gen_salt(length: int) -> str:
    """Generate a random string of SALT_CHARS with specified ``length``."""
    if length <= 0:
        raise ValueError("Salt length must be at least 1.")

    return "".join(secrets.choice(SALT_CHARS) for _ in range(length))


def _hash_internal(method: str, salt: str, password: str) -> tuple[str, str]:
    method, *args = method.split(":")
    salt_bytes = salt.encode()
    password_bytes = password.encode()

    # Security note: If the method contains args, they came from a stored,
    # trusted hash previously generated by generate_password_hash. An attacker
    # who could modify them can already compromise the entire application.

    if method == "scrypt":
        if not args:
            n = 2**15
            r = 8
            p = 1
        else:
            try:
                n, r, p = map(int, args)
            except ValueError:
                raise ValueError("'scrypt' takes 3 arguments.") from None

        maxmem = 132 * n * r * p  # ideally 128, but some extra seems needed
        return (
            hashlib.scrypt(
                password_bytes, salt=salt_bytes, n=n, r=r, p=p, maxmem=maxmem
            ).hex(),
            f"scrypt:{n}:{r}:{p}",
        )
    elif method == "pbkdf2":
        len_args = len(args)

        if len_args == 0:
            hash_name = "sha256"
            iterations = DEFAULT_PBKDF2_ITERATIONS
        elif len_args == 1:
            hash_name = args[0]
            iterations = DEFAULT_PBKDF2_ITERATIONS
        elif len_args == 2:
            hash_name = args[0]
            iterations = int(args[1])
        else:
            raise ValueError("'pbkdf2' takes 2 arguments.")

        return (
            hashlib.pbkdf2_hmac(
                hash_name, password_bytes, salt_bytes, iterations
            ).hex(),
            f"pbkdf2:{hash_name}:{iterations}",
        )
    else:
        raise ValueError(f"Invalid hash method '{method}'.")


def generate_password_hash(
    password: str, method: str = "scrypt", salt_length: int = 16
) -> str:
    """Securely hash a password for storage. A password can be compared to a stored hash
    using :func:`check_password_hash`.

    The following methods are supported:

    -   ``scrypt``, the default. The parameters are ``n``, ``r``, and ``p``, the default
        is ``scrypt:32768:8:1``. See :func:`hashlib.scrypt`.
    -   ``pbkdf2``, less secure. The parameters are ``hash_method`` and ``iterations``,
        the default is ``pbkdf2:sha256:600000``. See :func:`hashlib.pbkdf2_hmac`.

    Default parameters may be updated to reflect current guidelines, and methods may be
    deprecated and removed if they are no longer considered secure. To migrate old
    hashes, you may generate a new hash when checking an old hash, or you may contact
    users with a link to reset their password.

    :param password: The plaintext password.
    :param method: The key derivation function and parameters.
    :param salt_length: The number of characters to generate for the salt.

    .. versionchanged:: 3.1
        The default iterations for pbkdf2 was increased to 1,000,000.

    .. versionchanged:: 2.3
        Scrypt support was added.

    .. versionchanged:: 2.3
        The default iterations for pbkdf2 was increased to 600,000.

    .. versionchanged:: 2.3
        All plain hashes are deprecated and will not be supported in Werkzeug 3.0.
    """
    salt = gen_salt(salt_length)
    h, actual_method = _hash_internal(method, salt, password)
    return f"{actual_method}${salt}${h}"


def check_password_hash(pwhash: str, password: str) -> bool:
    """Securely check that the given stored password hash, previously generated using
    :func:`generate_password_hash`, matches the given password.

    Methods may be deprecated and removed if they are no longer considered secure. To
    migrate old hashes, you may generate a new hash when checking an old hash, or you
    may contact users with a link to reset their password.

    :param pwhash: The hashed password.
    :param password: The plaintext password.

    .. versionchanged:: 2.3
        All plain hashes are deprecated and will not be supported in Werkzeug 3.0.
    """
    try:
        method, salt, hashval = pwhash.split("$", 2)
    except ValueError:
        return False

    return hmac.compare_digest(_hash_internal(method, salt, password)[0], hashval)


def safe_join(directory: str, *untrusted: str) -> str | None:
    """Safely join zero or more untrusted path components to a trusted base
    directory to avoid escaping the base directory. Return ``None`` if the path
    is not safe.

    The untrusted path is assumed to be from/for a URL, such as for serving
    files. Therefore, it should only use the forward slash ``/`` path separator,
    and will be joined using that separator. On Windows, the backslash ``\\``
    separator is not allowed.

    This only considers the path name, it does not validate if it does or does
    not exist, if it is a symlink, or other properties of the file or
    filesystem. It's up to the application to ensure that the contents of
    ``directory`` can be trusted.

    On Windows, special device names such as ``CON`` and ``NUL`` are not
    allowed, as this is used for serving files with :func:`.send_from_directory`
    and those files will hang when read.

    NTFS alternate data streams (ADS) like ``filename:stream`` are
    allowed. :func:`.secure_filename` will remove the ``:`` when saving.

    :param directory: The trusted base directory.
    :param untrusted: The untrusted path components relative to the
        base directory.
    :return: A safe path, otherwise ``None``.

    .. versionchanged:: 3.1.9
        Special device names with empty ADS stream markers are not allowed on
        Windows.

    .. versionchanged:: 3.1.6
        Special device names in multi-segment paths are not allowed on Windows.

    .. versionchanged:: 3.1.5
        More special device names, regardless of extension or trailing spaces,
        are not allowed on Windows.

    .. versionchanged:: 3.1.4
        Special device names are not allowed on Windows.
    """
    if not directory:
        # Ensure we end up with ./path if directory="" is given,
        # otherwise the first untrusted part could become trusted.
        directory = "."

    parts = [directory]

    for part in untrusted:
        if not part:
            continue

        part = posixpath.normpath(part)

        if (
            os.path.isabs(part)
            # ntpath.isabs doesn't catch this
            or part.startswith("/")
            or part == ".."
            or part.startswith("../")
            or any(sep in part for sep in _os_alt_seps)
            or (
                os.name == "nt"
                and any(
                    p.partition(":")[0].partition(".")[0].strip().upper()
                    in _windows_device_files
                    for p in part.split("/")
                )
            )
        ):
            return None

        parts.append(part)

    return posixpath.join(*parts)
<|endoftext|>from __future__ import annotations

import io
import mimetypes
import os
import pkgutil
import re
import sys
import typing as t
import unicodedata
from datetime import datetime
from time import time
from urllib.parse import quote
from zlib import adler32

from markupsafe import escape

from ._internal import _DictAccessorProperty
from ._internal import _missing
from ._internal import _TAccessorValue
from .datastructures import Headers
from .exceptions import NotFound
from .exceptions import RequestedRangeNotSatisfiable
from .security import _windows_device_files
from .security import safe_join
from .wsgi import wrap_file

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

    from .wrappers.request import Request
    from .wrappers.response import Response

_T = t.TypeVar("_T")

_entity_re = re.compile(r"&([^;]+);")
_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.-]")


class cached_property(property, t.Generic[_T]):
    """A :func:`property` that is only evaluated once. Subsequent access
    returns the cached value. Setting the property sets the cached
    value. Deleting the property clears the cached value, accessing it
    again will evaluate it again.

    .. code-block:: python

        class Example:
            @cached_property
            def value(self):
                # calculate something important here
                return 42

        e = Example()
        e.value  # evaluates
        e.value  # uses cache
        e.value = 16  # sets cache
        del e.value  # clears cache

    If the class defines ``__slots__``, it must add ``_cache_{name}`` as
    a slot. Alternatively, it can add ``__dict__``, but that's usually
    not desirable.

    .. versionchanged:: 2.1
        Works with ``__slots__``.

    .. versionchanged:: 2.0
        ``del obj.name`` clears the cached value.
    """

    def __init__(
        self,
        fget: t.Callable[[t.Any], _T],
        name: str | None = None,
        doc: str | None = None,
    ) -> None:
        super().__init__(fget, doc=doc)
        self.__name__ = name or fget.__name__
        self.slot_name = f"_cache_{self.__name__}"
        self.__module__ = fget.__module__

    def __set__(self, obj: object, value: _T) -> None:
        if hasattr(obj, "__dict__"):
            obj.__dict__[self.__name__] = value
        else:
            setattr(obj, self.slot_name, value)

    def __get__(self, obj: object, type: type = None) -> _T:  # type: ignore
        if obj is None:
            return self  # type: ignore

        obj_dict = getattr(obj, "__dict__", None)

        if obj_dict is not None:
            value: _T = obj_dict.get(self.__name__, _missing)
        else:
            value = getattr(obj, self.slot_name, _missing)  # type: ignore[arg-type]

        if value is _missing:
            value = self.fget(obj)  # type: ignore

            if obj_dict is not None:
                obj.__dict__[self.__name__] = value
            else:
                setattr(obj, self.slot_name, value)

        return value

    def __delete__(self, obj: object) -> None:
        if hasattr(obj, "__dict__"):
            del obj.__dict__[self.__name__]
        else:
            setattr(obj, self.slot_name, _missing)


class environ_property(_DictAccessorProperty[_TAccessorValue]):
    """Maps request attributes to environment variables. This works not only
    for the Werkzeug request object, but also any other class with an
    environ attribute:

    >>> class Test(object):
    ...     environ = {'key': 'value'}
    ...     test = environ_property('key')
    >>> var = Test()
    >>> var.test
    'value'

    If you pass it a second value it's used as default if the key does not
    exist, the third one can be a converter that takes a value and converts
    it.  If it raises :exc:`ValueError` or :exc:`TypeError` the default value
    is used. If no default value is provided `None` is used.

    Per default the property is read only.  You have to explicitly enable it
    by passing ``read_only=False`` to the constructor.
    """

    read_only = True

    def lookup(self, obj: Request) -> WSGIEnvironment:
        return obj.environ


class header_property(_DictAccessorProperty[_TAccessorValue]):
    """Like `environ_property` but for headers."""

    def lookup(self, obj: Request | Response) -> Headers:  # type: ignore[override]
        return obj.headers


# https://cgit.freedesktop.org/xdg/shared-mime-info/tree/freedesktop.org.xml.in
# https://www.iana.org/assignments/media-types/media-types.xhtml
# Types listed in the XDG mime info that have a charset in the IANA registration.
_charset_mimetypes = {
    "application/ecmascript",
    "application/javascript",
    "application/sql",
    "application/xml",
    "application/xml-dtd",
    "application/xml-external-parsed-entity",
}


def get_content_type(mimetype: str, charset: str) -> str:
    """Returns the full content type string with charset for a mimetype.

    If the mimetype represents text, the charset parameter will be
    appended, otherwise the mimetype is returned unchanged.

    :param mimetype: The mimetype to be used as content type.
    :param charset: The charset to be appended for text mimetypes.
    :return: The content type.

    .. versionchanged:: 0.15
        Any type that ends with ``+xml`` gets a charset, not just those
        that start with ``application/``. Known text types such as
        ``application/javascript`` are also given charsets.
    """
    if (
        mimetype.startswith("text/")
        or mimetype in _charset_mimetypes
        or mimetype.endswith("+xml")
    ):
        mimetype += f"; charset={charset}"

    return mimetype


def secure_filename(filename: str) -> str:
    """Validate and modify a filename so that it is safe to use on a regular
    filesystem and in :func:`os.path.join`. This may produce an empty string if
    nothing in the filename could be safely preserved.

    Only the ASCII letters, digits, and ``._-`` characters are allowed, all
    others are removed or replaced. Unicode characters are decomposed and
    replaced with ASCII equivalents according to NFKD. ``/`` (and ``\\`` on
    Windows) and whitespace, are replaced with ``_``. Surrounding ``._``
    characters are removed.

    On Windows, special device names such as ``CON`` and ``NUL`` will be
    prefixed with ``_`` to avoid access.

    This only considers the filename, it does not validate if the file does or
    does not exist, if it is a symlink, or other properties of the file.

    :param filename: The filename to validate and modify.

    .. versionadded:: 0.5
    """
    # decompose combined chars, replace compat chars
    filename = unicodedata.normalize("NFKD", filename)
    # only keep ASCII chars
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")

    # replace any whitespace with _
    filename = "_".join(filename.split())
    # only allow ASCII letters, numbers, and _.-
    filename = _filename_ascii_strip_re.sub("", filename)
    # remove surrounding ._, left behind after replacing slash in ../
    filename = filename.strip("._")

    # prefix special device names on Windows
    if (
        os.name == "nt"
        and filename.partition(".")[0].strip().upper() in _windows_device_files
    ):
        filename = f"_{filename}"

    return filename


def redirect(
    location: str, code: int = 302, Response: type[Response] | None = None
) -> Response:
    """Returns a response object (a WSGI application) that, if called,
    redirects the client to the target location. Supported codes are
    301, 302, 303, 305, 307, and 308. 300 is not supported because
    it's not a real redirect and 304 because it's the answer for a
    request with a request with defined If-Modified-Since headers.

    .. versionadded:: 0.6
       The location can now be a unicode string that is encoded using
       the :func:`iri_to_uri` function.

    .. versionadded:: 0.10
        The class used for the Response object can now be passed in.

    :param location: the location the response should redirect to.
    :param code: the redirect status code. defaults to 302.
    :param class Response: a Response class to use when instantiating a
        response. The default is :class:`werkzeug.wrappers.Response` if
        unspecified.
    """
    if Response is None:
        from .wrappers import Response

    html_location = escape(location)
    response = Response(  # type: ignore[misc]
        "<!doctype html>\n"
        "<html lang=en>\n"
        "<title>Redirecting...</title>\n"
        "<h1>Redirecting...</h1>\n"
        "<p>You should be redirected automatically to the target URL: "
        f'<a href="{html_location}">{html_location}</a>. If not, click the link.\n',
        code,
        mimetype="text/html",
    )
    response.headers["Location"] = location
    return response


def append_slash_redirect(environ: WSGIEnvironment, code: int = 308) -> Response:
    """Redirect to the current URL with a slash appended.

    If the current URL is ``/user/42``, the redirect URL will be
    ``42/``. When joined to the current URL during response
    processing or by the browser, this will produce ``/user/42/``.

    The behavior is undefined if the path ends with a slash already. If
    called unconditionally on a URL, it may produce a redirect loop.

    :param environ: Use the path and query from this WSGI environment
        to produce the redirect URL.
    :param code: the status code for the redirect.

    .. versionchanged:: 2.1
        Produce a relative URL that only modifies the last segment.
        Relevant when the current path has multiple segments.

    .. versionchanged:: 2.1
        The default status code is 308 instead of 301. This preserves
        the request method and body.
    """
    tail = environ["PATH_INFO"].rpartition("/")[2]

    if not tail:
        new_path = "./"
    else:
        new_path = f"{tail}/"

    query_string = environ.get("QUERY_STRING")

    if query_string:
        new_path = f"{new_path}?{query_string}"

    return redirect(new_path, code)


def send_file(
    path_or_file: os.PathLike[str] | str | t.IO[bytes],
    environ: WSGIEnvironment,
    mimetype: str | None = None,
    as_attachment: bool = False,
    download_name: str | None = None,
    conditional: bool = True,
    etag: bool | str = True,
    last_modified: datetime | int | float | None = None,
    max_age: None | (int | t.Callable[[str | None], int | None]) = None,
    use_x_sendfile: bool = False,
    response_class: type[Response] | None = None,
    _root_path: os.PathLike[str] | str | None = None,
) -> Response:
    """Send the contents of a file to the client.

    The first argument can be a file path or a file-like object. Paths
    are preferred in most cases because Werkzeug can manage the file and
    get extra information from the path. Passing a file-like object
    requires that the file is opened in binary mode, and is mostly
    useful when building a file in memory with :class:`io.BytesIO`.

    Never pass file paths provided by a user. The path is assumed to be
    trusted, so a user could craft a path to access a file you didn't
    intend. Use :func:`send_from_directory` to safely serve user-provided paths.

    If the WSGI server sets a ``file_wrapper`` in ``environ``, it is
    used, otherwise Werkzeug's built-in wrapper is used. Alternatively,
    if the HTTP server supports ``X-Sendfile``, ``use_x_sendfile=True``
    will tell the server to send the given path, which is much more
    efficient than reading it in Python.

    :param path_or_file: The path to the file to send, relative to the
        current working directory if a relative path is given.
        Alternatively, a file-like object opened in binary mode. Make
        sure the file pointer is seeked to the start of the data.
    :param environ: The WSGI environ for the current request.
    :param mimetype: The MIME type to send for the file. If not
        provided, it will try to detect it from the file name.
    :param as_attachment: Indicate to a browser that it should offer to
        save the file instead of displaying it.
    :param download_name: The default name browsers will use when saving
        the file. Defaults to the passed file name.
    :param conditional: Enable conditional and range responses based on
        request headers. Requires passing a file path and ``environ``.
    :param etag: Calculate an ETag for the file, which requires passing
        a file path. Can also be a string to use instead.
    :param last_modified: The last modified time to send for the file,
        in seconds. If not provided, it will try to detect it from the
        file path.
    :param max_age: How long the client should cache the file, in
        seconds. If set, ``Cache-Control`` will be ``public``, otherwise
        it will be ``no-cache`` to prefer conditional caching.
    :param use_x_sendfile: Set the ``X-Sendfile`` header to let the
        server to efficiently send the file. Requires support from the
        HTTP server. Requires passing a file path.
    :param response_class: Build the response using this class. Defaults
        to :class:`~werkzeug.wrappers.Response`.
    :param _root_path: Do not use. For internal use only. Use
        :func:`send_from_directory` to safely send files under a path.

    .. versionchanged:: 2.0.2
        ``send_file`` only sets a detected ``Content-Encoding`` if
        ``as_attachment`` is disabled.

    .. versionadded:: 2.0
        Adapted from Flask's implementation.

    .. versionchanged:: 2.0
        ``download_name`` replaces Flask's ``attachment_filename``
         parameter. If ``as_attachment=False``, it is passed with
         ``Content-Disposition: inline`` instead.

    .. versionchanged:: 2.0
        ``max_age`` replaces Flask's ``cache_timeout`` parameter.
        ``conditional`` is enabled and ``max_age`` is not set by
        default.

    .. versionchanged:: 2.0
        ``etag`` replaces Flask's ``add_etags`` parameter. It can be a
        string to use instead of generating one.

    .. versionchanged:: 2.0
        If an encoding is returned when guessing ``mimetype`` from
        ``download_name``, set the ``Content-Encoding`` header.
    """
    if response_class is None:
        from .wrappers import Response

        response_class = Response

    path: str | None = None
    file: t.IO[bytes] | None = None
    size: int | None = None
    mtime: float | None = None
    headers = Headers()

    if isinstance(path_or_file, (os.PathLike, str)) or hasattr(
        path_or_file, "__fspath__"
    ):
        path_or_file = t.cast("os.PathLike[str] | str", path_or_file)

        # Flask will pass app.root_path, allowing its send_file wrapper
        # to not have to deal with paths.
        if _root_path is not None:
            path = os.path.join(_root_path, path_or_file)
        else:
            path = os.path.abspath(path_or_file)

        stat = os.stat(path)
        size = stat.st_size
        mtime = stat.st_mtime
    else:
        file = path_or_file

    if download_name is None and path is not None:
        download_name = os.path.basename(path)

    if mimetype is None:
        if download_name is None:
            raise TypeError(
                "Unable to detect the MIME type because a file name is"
                " not available. Either set 'download_name', pass a"
                " path instead of a file, or set 'mimetype'."
            )

        mimetype, encoding = mimetypes.guess_type(download_name)

        if mimetype is None:
            mimetype = "application/octet-stream"

        # Don't send encoding for attachments, it causes browsers to
        # save decompress tar.gz files.
        if encoding is not None and not as_attachment:
            headers.set("Content-Encoding", encoding)

    if download_name is not None:
        try:
            download_name.encode("ascii")
        except UnicodeEncodeError:
            simple = unicodedata.normalize("NFKD", download_name)
            simple = simple.encode("ascii", "ignore").decode("ascii")
            # safe = RFC 5987 attr-char
            quoted = quote(download_name, safe="!#$&+-.^_`|~")
            names = {"filename": simple, "filename*": f"UTF-8''{quoted}"}
        else:
            names = {"filename": download_name}

        value = "attachment" if as_attachment else "inline"
        headers.set("Content-Disposition", value, **names)
    elif as_attachment:
        raise TypeError(
            "No name provided for attachment. Either set"
            " 'download_name' or pass a path instead of a file."
        )

    if use_x_sendfile and path is not None:
        headers["X-Sendfile"] = path
        data = None
    else:
        if file is None:
            file = open(path, "rb")  # type: ignore
        elif isinstance(file, io.BytesIO):
            size = file.getbuffer().nbytes
        elif isinstance(file, io.TextIOBase):
            raise ValueError("Files must be opened in binary mode or use BytesIO.")

        data = wrap_file(environ, file)

    rv = response_class(
        data, mimetype=mimetype, headers=headers, direct_passthrough=True
    )

    if size is not None:
        rv.content_length = size

    if last_modified is not None:
        rv.last_modified = last_modified  # type: ignore
    elif mtime is not None:
        rv.last_modified = mtime  # type: ignore

    rv.cache_control.no_cache = True

    # Flask will pass app.get_send_file_max_age, allowing its send_file
    # wrapper to not have to deal with paths.
    if callable(max_age):
        max_age = max_age(path)

    if max_age is not None:
        if max_age > 0:
            rv.cache_control.no_cache = None
            rv.cache_control.public = True

        rv.cache_control.max_age = max_age
        rv.expires = int(time() + max_age)  # type: ignore

    if isinstance(etag, str):
        rv.set_etag(etag)
    elif etag and path is not None:
        check = adler32(path.encode()) & 0xFFFFFFFF
        rv.set_etag(f"{mtime}-{size}-{check}")

    if conditional:
        try:
            rv = rv.make_conditional(environ, accept_ranges=True, complete_length=size)
        except RequestedRangeNotSatisfiable:
            if file is not None:
                file.close()

            raise

        # Some x-sendfile implementations incorrectly ignore the 304
        # status code and send the file anyway.
        if rv.status_code == 304:
            rv.headers.pop("x-sendfile", None)

    return rv


def send_from_directory(
    directory: os.PathLike[str] | str,
    path: os.PathLike[str] | str,
    environ: WSGIEnvironment,
    **kwargs: t.Any,
) -> Response:
    """Send a file from within a directory using :func:`send_file`.

    This is a secure way to serve files from a folder, such as static
    files or uploads. Uses :func:`~werkzeug.security.safe_join` to
    ensure the path coming from the client is not maliciously crafted to
    point outside the specified directory.

    If the final path does not point to an existing regular file,
    returns a 404 :exc:`~werkzeug.exceptions.NotFound` error.

    :param directory: The directory that ``path`` must be located under. This *must not*
        be a value provided by the client, otherwise it becomes insecure.
    :param path: The path to the file to send, relative to ``directory``. This is the
        part of the path provided by the client, which is checked for security.
    :param environ: The WSGI environ for the current request.
    :param kwargs: Arguments to pass to :func:`send_file`.

    .. versionadded:: 2.0
        Adapted from Flask's implementation.
    """
    path_str = safe_join(os.fspath(directory), os.fspath(path))

    if path_str is None:
        raise NotFound()

    # Flask will pass app.root_path, allowing its send_from_directory
    # wrapper to not have to deal with paths.
    if "_root_path" in kwargs:
        path_str = os.path.join(kwargs["_root_path"], path_str)

    if not os.path.isfile(path_str):
        raise NotFound()

    return send_file(path_str, environ, **kwargs)


def import_string(import_name: str, silent: bool = False) -> t.Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If `silent` is True the return value will be `None` if the import fails.

    :param import_name: the dotted name for the object to import.
    :param silent: if set to `True` import errors are ignored and
                   `None` is returned instead.
    :return: imported object
    """
    import_name = import_name.replace(":", ".")
    try:
        try:
            __import__(import_name)
        except ImportError:
            if "." not in import_name:
                raise
        else:
            return sys.modules[import_name]

        module_name, obj_name = import_name.rsplit(".", 1)
        module = __import__(module_name, globals(), locals(), [obj_name])
        try:
            return getattr(module, obj_name)
        except AttributeError as e:
            raise ImportError(e) from None

    except ImportError as e:
        if not silent:
            raise ImportStringError(import_name, e).with_traceback(
                sys.exc_info()[2]
            ) from None

    return None


def find_modules(
    import_path: str, include_packages: bool = False, recursive: bool = False
) -> t.Iterator[str]:
    """Finds all the modules below a package.  This can be useful to
    automatically import all views / controllers so that their metaclasses /
    function decorators have a chance to register themselves on the
    application.

    Packages are not returned unless `include_packages` is `True`.  This can
    also recursively list modules but in that case it will import all the
    packages to get the correct load path of that module.

    :param import_path: the dotted name for the package to find child modules.
    :param include_packages: set to `True` if packages should be returned, too.
    :param recursive: set to `True` if recursion should happen.
    :return: generator
    """
    module = import_string(import_path)
    path = getattr(module, "__path__", None)
    if path is None:
        raise ValueError(f"{import_path!r} is not a package")
    basename = f"{module.__name__}."
    for _importer, modname, ispkg in pkgutil.iter_modules(path):
        modname = basename + modname
        if ispkg:
            if include_packages:
                yield modname
            if recursive:
                yield from find_modules(modname, include_packages, True)
        else:
            yield modname


class ImportStringError(ImportError):
    """Provides information about a failed :func:`import_string` attempt."""

    #: String in dotted notation that failed to be imported.
    import_name: str
    #: Wrapped exception.
    exception: BaseException

    def __init__(self, import_name: str, exception: BaseException) -> None:
        self.import_name = import_name
        self.exception = exception
        msg = import_name
        name = ""
        tracked = []
        for part in import_name.replace(":", ".").split("."):
            name = f"{name}.{part}" if name else part
            imported = import_string(name, silent=True)
            if imported:
                tracked.append((name, getattr(imported, "__file__", None)))
            else:
                track = [f"- {n!r} found in {i!r}." for n, i in tracked]
                track.append(f"- {name!r} not found.")
                track_str = "\n".join(track)
                msg = (
                    f"import_string() failed for {import_name!r}. Possible reasons"
                    f" are:\n\n"
                    "- missing __init__.py in a package;\n"
                    "- package or module path not included in sys.path;\n"
                    "- duplicated package or module name taking precedence in"
                    " sys.path;\n"
                    "- missing module, class, function or variable;\n\n"
                    f"Debugged import:\n\n{track_str}\n\n"
                    f"Original exception:\n\n{type(exception).__name__}: {exception}"
                )
                break

        super().__init__(msg)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.import_name!r}, {self.exception!r})>"
<|endoftext|>"""A WSGI and HTTP server for use **during development only**. This
server is convenient to use, but is not designed to be particularly
stable, secure, or efficient. Use a dedicate WSGI server and HTTP
server when deploying to production.

It provides features like interactive debugging and code reloading. Use
``run_simple`` to start the server. Put this in a ``run.py`` script:

.. code-block:: python

    from myapp import create_app
    from werkzeug import run_simple
"""

from __future__ import annotations

import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit

from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .http import parse_set_header
from .urls import uri_to_iri

try:
    import ssl

    connection_dropped_errors: tuple[type[Exception], ...] = (
        ConnectionError,
        socket.timeout,
        ssl.SSLEOFError,
    )
except ImportError:

    class _SslDummy:
        def __getattr__(self, name: str) -> t.Any:
            raise RuntimeError(  # noqa: B904
                "SSL is unavailable because this Python runtime was not"
                " compiled with SSL/TLS support."
            )

    ssl = _SslDummy()  # type: ignore
    connection_dropped_errors = (ConnectionError, socket.timeout)

_log_add_style = True

if os.name == "nt":
    try:
        __import__("colorama")
    except ImportError:
        _log_add_style = False

can_fork = hasattr(os, "fork")

if can_fork:
    ForkingMixIn = socketserver.ForkingMixIn
else:

    class ForkingMixIn:  # type: ignore
        pass


try:
    af_unix = socket.AF_UNIX
except AttributeError:
    af_unix = None  # type: ignore

LISTEN_QUEUE = 128

_TSSLContextArg = t.Optional[
    t.Union["ssl.SSLContext", tuple[str, t.Optional[str]], t.Literal["adhoc"]]
]

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        RSAPrivateKeyWithSerialization,
    )
    from cryptography.x509 import Certificate


class DechunkedInput(io.RawIOBase):
    """An input stream that handles ``Transfer-Encoding: chunked``. Only
    used by the dev server, which must not be used in production. A production
    WSGI server will have its own robust, secure chunk handler.
    """

    def __init__(self, rfile: t.IO[bytes]) -> None:
        self._rfile = rfile
        self._done = False
        self._len = 0

    def readable(self) -> bool:
        return True

    def read_chunk_len(self) -> int:
        try:
            line = self._rfile.readline(100).decode("latin1")
            _len = int(line.strip(" \t\r\n"), 16)
        except ValueError as e:
            raise OSError("Invalid chunk header") from e
        if _len < 0:
            raise OSError("Negative chunk length not allowed")
        return _len

    def readinto(self, buf: bytearray) -> int:  # type: ignore
        read = 0
        while not self._done and read < len(buf):
            if self._len == 0:
                # This is the first chunk or we fully consumed the previous
                # one. Read the next length of the next chunk
                self._len = self.read_chunk_len()

            if self._len == 0:
                # Found the final chunk of size 0. The stream is now exhausted,
                # but there is still a final newline that should be consumed
                self._done = True

            if self._len > 0:
                # There is data (left) in this chunk, so append it to the
                # buffer. If this operation fully consumes the chunk, this will
                # reset self._len to 0.
                n = min(len(buf), self._len)

                # If (read + chunk size) becomes more than len(buf), buf will
                # grow beyond the original size and read more data than
                # required. So only read as much data as can fit in buf.
                if read + n > len(buf):
                    buf[read:] = self._rfile.read(len(buf) - read)
                    self._len -= len(buf) - read
                    read = len(buf)
                else:
                    buf[read : read + n] = self._rfile.read(n)
                    self._len -= n
                    read += n

            if self._len == 0:
                # Skip the terminating newline of a chunk that has been fully
                # consumed. This also applies to the 0-sized final chunk
                terminator = self._rfile.readline(2)
                if terminator not in (b"\n", b"\r\n", b"\r"):
                    raise OSError("Missing chunk terminating newline")

        return read


class WSGIRequestHandler(BaseHTTPRequestHandler):
    """A request handler that implements WSGI dispatching."""

    server: BaseWSGIServer

    @property
    def server_version(self) -> str:  # type: ignore
        return self.server._server_version

    def make_environ(self) -> WSGIEnvironment:
        request_url = urlsplit(self.path)
        url_scheme = "http" if self.server.ssl_context is None else "https"

        if not self.client_address:
            self.client_address = ("<local>", 0)
        elif isinstance(self.client_address, str):
            self.client_address = (self.client_address, 0)

        # If there was no scheme but the path started with two slashes,
        # the first segment may have been incorrectly parsed as the
        # netloc, prepend it to the path again.
        if not request_url.scheme and request_url.netloc:
            path_info = f"/{request_url.netloc}{request_url.path}"
        else:
            path_info = request_url.path

        path_info = unquote(path_info)

        environ: WSGIEnvironment = {
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": url_scheme,
            "wsgi.input": self.rfile,
            "wsgi.errors": sys.stderr,
            "wsgi.multithread": self.server.multithread,
            "wsgi.multiprocess": self.server.multiprocess,
            "wsgi.run_once": False,
            "werkzeug.socket": self.connection,
            "SERVER_SOFTWARE": self.server_version,
            "REQUEST_METHOD": self.command,
            "SCRIPT_NAME": "",
            "PATH_INFO": _wsgi_encoding_dance(path_info),
            "QUERY_STRING": _wsgi_encoding_dance(request_url.query),
            # Non-standard, added by mod_wsgi, uWSGI
            "REQUEST_URI": _wsgi_encoding_dance(self.path),
            # Non-standard, added by gunicorn
            "RAW_URI": _wsgi_encoding_dance(self.path),
            "REMOTE_ADDR": self.address_string(),
            "REMOTE_PORT": self.port_integer(),
            "SERVER_NAME": self.server.server_address[0],
            "SERVER_PORT": str(self.server.server_address[1]),
            "SERVER_PROTOCOL": self.request_version,
        }

        for key, value in self.headers.items():
            if "_" in key:
                continue

            key = key.upper().replace("-", "_")
            value = value.replace("\r\n", "")
            if key not in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                key = f"HTTP_{key}"
                if key in environ:
                    value = f"{environ[key]},{value}"
            environ[key] = value

        if "chunked" in parse_set_header(environ.get("HTTP_TRANSFER_ENCODING")):
            environ["wsgi.input_terminated"] = True
            environ["wsgi.input"] = DechunkedInput(environ["wsgi.input"])

        # Per RFC 2616, if the URL is absolute, use that as the host.
        # We're using "has a scheme" to indicate an absolute URL.
        if request_url.scheme and request_url.netloc:
            environ["HTTP_HOST"] = request_url.netloc

        try:
            # binary_form=False gives nicer information, but wouldn't be compatible with
            # what Nginx or Apache could return.
            peer_cert = self.connection.getpeercert(binary_form=True)
            if peer_cert is not None:
                # Nginx and Apache use PEM format.
                environ["SSL_CLIENT_CERT"] = ssl.DER_cert_to_PEM_cert(peer_cert)
        except ValueError:
            # SSL handshake hasn't finished.
            self.server.log("error", "Cannot fetch SSL peer certificate info")
        except AttributeError:
            # Not using TLS, the socket will not have getpeercert().
            pass

        return environ

    def run_wsgi(self) -> None:
        if self.headers.get("Expect", "").lower().strip(" \t") == "100-continue":
            self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")

        self.environ = environ = self.make_environ()
        status_set: str | None = None
        headers_set: list[tuple[str, str]] | None = None
        status_sent: str | None = None
        headers_sent: list[tuple[str, str]] | None = None
        chunk_response: bool = False

        def write(data: bytes) -> None:
            nonlocal status_sent, headers_sent, chunk_response
            assert status_set is not None, "write() before start_response"
            assert headers_set is not None, "write() before start_response"
            if status_sent is None:
                status_sent = status_set
                headers_sent = headers_set
                try:
                    code_str, msg = status_sent.split(None, 1)
                except ValueError:
                    code_str, msg = status_sent, ""
                code = int(code_str)
                self.send_response(code, msg)
                header_keys = set()
                for key, value in headers_sent:
                    self.send_header(key, value)
                    header_keys.add(key.lower())

                # Use chunked transfer encoding if there is no content
                # length. Do not use for 1xx and 204 responses. 304
                # responses and HEAD requests are also excluded, which
                # is the more conservative behavior and matches other
                # parts of the code.
                # https://httpwg.org/specs/rfc7230.html#rfc.section.3.3.1
                if (
                    not (
                        "content-length" in header_keys
                        or environ["REQUEST_METHOD"] == "HEAD"
                        or (100 <= code < 200)
                        or code in {204, 304}
                    )
                    and self.protocol_version >= "HTTP/1.1"
                ):
                    chunk_response = True
                    self.send_header("Transfer-Encoding", "chunked")

                # Always close the connection. This disables HTTP/1.1
                # keep-alive connections. They aren't handled well by
                # Python's http.server because it doesn't know how to
                # drain the stream before the next request line.
                self.send_header("Connection", "close")
                self.end_headers()

            assert isinstance(data, bytes), "applications must write bytes"

            if data:
                if chunk_response:
                    self.wfile.write(hex(len(data))[2:].encode())
                    self.wfile.write(b"\r\n")

                self.wfile.write(data)

                if chunk_response:
                    self.wfile.write(b"\r\n")

            self.wfile.flush()

        def start_response(status, headers, exc_info=None):  # type: ignore
            nonlocal status_set, headers_set
            if exc_info:
                try:
                    if headers_sent:
                        raise exc_info[1].with_traceback(exc_info[2])
                finally:
                    exc_info = None
            elif headers_set:
                raise AssertionError("Headers already set")
            status_set = status
            headers_set = headers
            return write

        def execute(app: WSGIApplication) -> None:
            application_iter = app(environ, start_response)
            try:
                for data in application_iter:
                    write(data)
                if not headers_sent:
                    write(b"")
                if chunk_response:
                    self.wfile.write(b"0\r\n\r\n")
            finally:
                # Check for any remaining data in the read socket, and discard it. This
                # will read past request.max_content_length, but lets the client see a
                # 413 response instead of a connection reset failure. If we supported
                # keep-alive connections, this naive approach would break by reading the
                # next request line. Since we know that write (above) closes every
                # connection we can read everything.
                selector = selectors.DefaultSelector()
                selector.register(self.connection, selectors.EVENT_READ)
                total_size = 0
                total_reads = 0

                # A timeout of 0 tends to fail because a client needs a small amount of
                # time to continue sending its data.
                while selector.select(timeout=0.01):
                    # Only read 10MB into memory at a time.
                    data = self.rfile.read(10_000_000)
                    total_size += len(data)
                    total_reads += 1

                    # Stop reading on no data, >=10GB, or 1000 reads. If a client sends
                    # more than that, they'll get a connection reset failure.
                    if not data or total_size >= 10_000_000_000 or total_reads > 1000:
                        break

                selector.close()

                if hasattr(application_iter, "close"):
                    application_iter.close()

        try:
            execute(self.server.app)
        except connection_dropped_errors as e:
            self.connection_dropped(e, environ)
        except Exception as e:
            if self.server.passthrough_errors:
                raise

            if status_sent is not None and chunk_response:
                self.close_connection = True

            try:
                # if we haven't yet sent the headers but they are set
                # we roll back to be able to set them again.
                if status_sent is None:
                    status_set = None
                    headers_set = None
                execute(InternalServerError())
            except Exception:
                pass

            from .debug.tbtools import DebugTraceback

            msg = DebugTraceback(e).render_traceback_text()
            self.server.log("error", f"Error on request:\n{msg}")

    def handle(self) -> None:
        """Handles a request ignoring dropped connections."""
        try:
            super().handle()
        except (ConnectionError, socket.timeout) as e:
            self.connection_dropped(e)
        except Exception as e:
            if self.server.ssl_context is not None and is_ssl_error(e):
                self.log_error("SSL error occurred: %s", e)
            else:
                raise

    def connection_dropped(
        self, error: BaseException, environ: WSGIEnvironment | None = None
    ) -> None:
        """Called if the connection was closed by the client.  By default
        nothing happens.
        """

    def __getattr__(self, name: str) -> t.Any:
        # All HTTP methods are handled by run_wsgi.
        if name.startswith("do_"):
            return self.run_wsgi

        # All other attributes are forwarded to the base class.
        return getattr(super(), name)

    def address_string(self) -> str:
        if getattr(self, "environ", None):
            return self.environ["REMOTE_ADDR"]  # type: ignore

        if not self.client_address:
            return "<local>"

        return self.client_address[0]

    def port_integer(self) -> int:
        return self.client_address[1]

    # Escape control characters. This is defined (but private) in Python 3.12.
    _control_char_table = str.maketrans(
        {c: rf"\x{c:02x}" for c in [*range(0x20), *range(0x7F, 0xA0)]}
    )
    _control_char_table[ord("\\")] = r"\\"

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        try:
            path = uri_to_iri(self.path)
            msg = f"{self.command} {path} {self.request_version}"
        except AttributeError:
            # path isn't set if the requestline was bad
            msg = self.requestline

        # Escape control characters that may be in the decoded path.
        msg = msg.translate(self._control_char_table)
        code = str(code)

        if code.startswith("1"):  # 1xx - Informational
            msg = _ansi_style(msg, "bold")
        elif code == "200":  # 2xx - Success
            pass
        elif code == "304":  # 304 - Resource Not Modified
            msg = _ansi_style(msg, "cyan")
        elif code.startswith("3"):  # 3xx - Redirection
            msg = _ansi_style(msg, "green")
        elif code == "404":  # 404 - Resource Not Found
            msg = _ansi_style(msg, "yellow")
        elif code.startswith("4"):  # 4xx - Client Error
            msg = _ansi_style(msg, "bold", "red")
        else:  # 5xx, or any other response
            msg = _ansi_style(msg, "bold", "magenta")

        self.log("info", '"%s" %s %s', msg, code, size)

    def log_error(self, format: str, *args: t.Any) -> None:
        self.log("error", format, *args)

    def log_message(self, format: str, *args: t.Any) -> None:
        self.log("info", format, *args)

    def log(self, type: str, message: str, *args: t.Any) -> None:
        # an IPv6 scoped address contains "%" which breaks logging
        address_string = self.address_string().replace("%", "%%")
        _log(
            type,
            f"{address_string} - - [{self.log_date_time_string()}] {message}\n",
            *args,
        )


def _ansi_style(value: str, *styles: str) -> str:
    if not _log_add_style:
        return value

    codes = {
        "bold": 1,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "magenta": 35,
        "cyan": 36,
    }

    for style in styles:
        value = f"\x1b[{codes[style]}m{value}"

    return f"{value}\x1b[0m"


def generate_adhoc_ssl_pair(
    cn: str | None = None,
) -> tuple[Certificate, RSAPrivateKeyWithSerialization]:
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise TypeError(
            "Using ad-hoc certificates requires the cryptography library."
        ) from None

    backend = default_backend()
    pkey = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=backend
    )

    # pretty damn sure that this is not actually accepted by anyone
    if cn is None:
        cn = "*"

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Dummy Certificate"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )

    backend = default_backend()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(dt.now(timezone.utc))
        .not_valid_after(dt.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([x509.OID_SERVER_AUTH]), critical=False)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(cn), x509.DNSName(f"*.{cn}")]),
            critical=False,
        )
        .sign(pkey, hashes.SHA256(), backend)
    )
    return cert, pkey


def make_ssl_devcert(
    base_path: str, host: str | None = None, cn: str | None = None
) -> tuple[str, str]:
    """Creates an SSL key for development.  This should be used instead of
    the ``'adhoc'`` key which generates a new cert on each server start.
    It accepts a path for where it should store the key and cert and
    either a host or CN.  If a host is given it will use the CN
    ``*.host/CN=host``.

    For more information see :func:`run_simple`.

    .. versionadded:: 0.9

    :param base_path: the path to the certificate and key.  The extension
                      ``.crt`` is added for the certificate, ``.key`` is
                      added for the key.
    :param host: the name of the host.  This can be used as an alternative
                 for the `cn`.
    :param cn: the `CN` to use.
    """

    if host is not None:
        cn = host
    cert, pkey = generate_adhoc_ssl_pair(cn=cn)

    from cryptography.hazmat.primitives import serialization

    cert_file = f"{base_path}.crt"
    pkey_file = f"{base_path}.key"

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(pkey_file, "wb") as f:
        f.write(
            pkey.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return cert_file, pkey_file


def generate_adhoc_ssl_context() -> ssl.SSLContext:
    """Generates an adhoc SSL context for the development server."""
    import atexit
    import tempfile

    cert, pkey = generate_adhoc_ssl_pair()

    from cryptography.hazmat.primitives import serialization

    cert_handle, cert_file = tempfile.mkstemp()
    pkey_handle, pkey_file = tempfile.mkstemp()
    atexit.register(os.remove, pkey_file)
    atexit.register(os.remove, cert_file)

    os.write(cert_handle, cert.public_bytes(serialization.Encoding.PEM))
    os.write(
        pkey_handle,
        pkey.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )

    os.close(cert_handle)
    os.close(pkey_handle)
    ctx = load_ssl_context(cert_file, pkey_file)
    return ctx


def load_ssl_context(
    cert_file: str, pkey_file: str | None = None, protocol: int | None = None
) -> ssl.SSLContext:
    """Loads SSL context from cert/private key files and optional protocol.
    Many parameters are directly taken from the API of
    :py:class:`ssl.SSLContext`.

    :param cert_file: Path of the certificate to use.
    :param pkey_file: Path of the private key to use. If not given, the key
                      will be obtained from the certificate file.
