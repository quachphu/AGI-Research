    """Internal dependency holder class.

    Used to keep together the main dependencies used
    at the moment of send a signal.
    """

    def __init__(
        self,
        session: "ClientSession",
        trace_config: TraceConfig,
        trace_config_ctx: SimpleNamespace,
    ) -> None:
        self._trace_config = trace_config
        self._trace_config_ctx = trace_config_ctx
        self._session = session

    async def send_request_start(
        self, method: str, url: URL, headers: "CIMultiDict[str]"
    ) -> None:
        return await self._trace_config.on_request_start.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestStartParams(method, url, headers),
        )

    async def send_request_chunk_sent(
        self, method: str, url: URL, chunk: bytes
    ) -> None:
        return await self._trace_config.on_request_chunk_sent.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestChunkSentParams(method, url, chunk),
        )

    async def send_response_chunk_received(
        self, method: str, url: URL, chunk: bytes
    ) -> None:
        return await self._trace_config.on_response_chunk_received.send(
            self._session,
            self._trace_config_ctx,
            TraceResponseChunkReceivedParams(method, url, chunk),
        )

    async def send_request_end(
        self,
        method: str,
        url: URL,
        headers: "CIMultiDict[str]",
        response: ClientResponse,
    ) -> None:
        return await self._trace_config.on_request_end.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestEndParams(method, url, headers, response),
        )

    async def send_request_exception(
        self,
        method: str,
        url: URL,
        headers: "CIMultiDict[str]",
        exception: BaseException,
    ) -> None:
        return await self._trace_config.on_request_exception.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestExceptionParams(method, url, headers, exception),
        )

    async def send_request_redirect(
        self,
        method: str,
        url: URL,
        headers: "CIMultiDict[str]",
        response: ClientResponse,
    ) -> None:
        return await self._trace_config._on_request_redirect.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestRedirectParams(method, url, headers, response),
        )

    async def send_connection_queued_start(self) -> None:
        return await self._trace_config.on_connection_queued_start.send(
            self._session, self._trace_config_ctx, TraceConnectionQueuedStartParams()
        )

    async def send_connection_queued_end(self) -> None:
        return await self._trace_config.on_connection_queued_end.send(
            self._session, self._trace_config_ctx, TraceConnectionQueuedEndParams()
        )

    async def send_connection_create_start(self) -> None:
        return await self._trace_config.on_connection_create_start.send(
            self._session, self._trace_config_ctx, TraceConnectionCreateStartParams()
        )

    async def send_connection_create_end(self) -> None:
        return await self._trace_config.on_connection_create_end.send(
            self._session, self._trace_config_ctx, TraceConnectionCreateEndParams()
        )

    async def send_connection_reuseconn(self) -> None:
        return await self._trace_config.on_connection_reuseconn.send(
            self._session, self._trace_config_ctx, TraceConnectionReuseconnParams()
        )

    async def send_dns_resolvehost_start(self, host: str) -> None:
        return await self._trace_config.on_dns_resolvehost_start.send(
            self._session, self._trace_config_ctx, TraceDnsResolveHostStartParams(host)
        )

    async def send_dns_resolvehost_end(self, host: str) -> None:
        return await self._trace_config.on_dns_resolvehost_end.send(
            self._session, self._trace_config_ctx, TraceDnsResolveHostEndParams(host)
        )

    async def send_dns_cache_hit(self, host: str) -> None:
        return await self._trace_config.on_dns_cache_hit.send(
            self._session, self._trace_config_ctx, TraceDnsCacheHitParams(host)
        )

    async def send_dns_cache_miss(self, host: str) -> None:
        return await self._trace_config.on_dns_cache_miss.send(
            self._session, self._trace_config_ctx, TraceDnsCacheMissParams(host)
        )

    async def send_request_headers(
        self, method: str, url: URL, headers: "CIMultiDict[str]"
    ) -> None:
        return await self._trace_config._on_request_headers_sent.send(
            self._session,
            self._trace_config_ctx,
            TraceRequestHeadersSentParams(method, url, headers),
        )
<|endoftext|>__version__ = "3.14.5"

from typing import TYPE_CHECKING

from . import hdrs as hdrs
from .client import (
    BaseConnector,
    ClientConnectionError,
    ClientConnectionResetError,
    ClientConnectorCertificateError,
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientError,
    ClientHttpProxyError,
    ClientOSError,
    ClientPayloadError,
    ClientProxyConnectionError,
    ClientRequest,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientSSLError,
    ClientTimeout,
    ClientWebSocketResponse,
    ClientWSTimeout,
    ConnectionTimeoutError,
    ContentTypeError,
    Fingerprint,
    InvalidURL,
    InvalidUrlClientError,
    InvalidUrlRedirectClientError,
    NamedPipeConnector,
    NonHttpUrlClientError,
    NonHttpUrlRedirectClientError,
    RedirectClientError,
    RequestInfo,
    ServerConnectionError,
    ServerDisconnectedError,
    ServerFingerprintMismatch,
    ServerTimeoutError,
    SocketTimeoutError,
    TCPConnector,
    TooManyRedirects,
    UnixConnector,
    UploadAbortedError,
    UploadTracker,
    WSMessageTypeError,
    WSServerHandshakeError,
    request,
)
from .client_middleware_digest_auth import DigestAuthMiddleware
from .client_middlewares import ClientHandlerType, ClientMiddlewareType
from .compression_utils import set_zlib_backend
from .connector import (
    AddrInfoType as AddrInfoType,
    SocketFactoryType as SocketFactoryType,
)
from .cookiejar import CookieJar as CookieJar, DummyCookieJar as DummyCookieJar
from .formdata import FormData as FormData
from .helpers import BasicAuth, ChainMapProxy, ETag, encode_basic_auth
from .http import (
    HttpVersion as HttpVersion,
    HttpVersion10 as HttpVersion10,
    HttpVersion11 as HttpVersion11,
    WebSocketError as WebSocketError,
    WSCloseCode as WSCloseCode,
    WSMessage as WSMessage,
    WSMsgType as WSMsgType,
)
from .multipart import (
    BadContentDispositionHeader as BadContentDispositionHeader,
    BadContentDispositionParam as BadContentDispositionParam,
    BodyPartReader as BodyPartReader,
    MultipartReader as MultipartReader,
    MultipartWriter as MultipartWriter,
    content_disposition_filename as content_disposition_filename,
    parse_content_disposition as parse_content_disposition,
)
from .payload import (
    PAYLOAD_REGISTRY as PAYLOAD_REGISTRY,
    AsyncIterablePayload as AsyncIterablePayload,
    BufferedReaderPayload as BufferedReaderPayload,
    BytesIOPayload as BytesIOPayload,
    BytesPayload as BytesPayload,
    IOBasePayload as IOBasePayload,
    JsonPayload as JsonPayload,
    Payload as Payload,
    StringIOPayload as StringIOPayload,
    StringPayload as StringPayload,
    TextIOPayload as TextIOPayload,
    get_payload as get_payload,
    payload_type as payload_type,
)
from .payload_streamer import streamer as streamer
from .resolver import (
    AsyncResolver as AsyncResolver,
    DefaultResolver as DefaultResolver,
    ThreadedResolver as ThreadedResolver,
)
from .streams import (
    EMPTY_PAYLOAD as EMPTY_PAYLOAD,
    DataQueue as DataQueue,
    EofStream as EofStream,
    FlowControlDataQueue as FlowControlDataQueue,
    StreamReader as StreamReader,
)
from .tracing import (
    TraceConfig as TraceConfig,
    TraceConnectionCreateEndParams as TraceConnectionCreateEndParams,
    TraceConnectionCreateStartParams as TraceConnectionCreateStartParams,
    TraceConnectionQueuedEndParams as TraceConnectionQueuedEndParams,
    TraceConnectionQueuedStartParams as TraceConnectionQueuedStartParams,
    TraceConnectionReuseconnParams as TraceConnectionReuseconnParams,
    TraceDnsCacheHitParams as TraceDnsCacheHitParams,
    TraceDnsCacheMissParams as TraceDnsCacheMissParams,
    TraceDnsResolveHostEndParams as TraceDnsResolveHostEndParams,
    TraceDnsResolveHostStartParams as TraceDnsResolveHostStartParams,
    TraceRequestChunkSentParams as TraceRequestChunkSentParams,
    TraceRequestEndParams as TraceRequestEndParams,
    TraceRequestExceptionParams as TraceRequestExceptionParams,
    TraceRequestHeadersSentParams as TraceRequestHeadersSentParams,
    TraceRequestRedirectParams as TraceRequestRedirectParams,
    TraceRequestStartParams as TraceRequestStartParams,
    TraceResponseChunkReceivedParams as TraceResponseChunkReceivedParams,
)

if TYPE_CHECKING:
    # At runtime these are lazy-loaded at the bottom of the file.
    from .worker import (
        GunicornUVLoopWebWorker as GunicornUVLoopWebWorker,
        GunicornWebWorker as GunicornWebWorker,
    )

__all__: tuple[str, ...] = (
    "hdrs",
    # client
    "AddrInfoType",
    "BaseConnector",
    "ClientConnectionError",
    "ClientConnectionResetError",
    "ClientConnectorCertificateError",
    "ClientConnectorDNSError",
    "ClientConnectorError",
    "ClientConnectorSSLError",
    "ClientError",
    "ClientHttpProxyError",
    "ClientOSError",
    "ClientPayloadError",
    "ClientProxyConnectionError",
    "ClientResponse",
    "ClientRequest",
    "ClientResponseError",
    "ClientSSLError",
    "ClientSession",
    "ClientTimeout",
    "ClientWebSocketResponse",
    "ClientWSTimeout",
    "ConnectionTimeoutError",
    "ContentTypeError",
    "Fingerprint",
    "FlowControlDataQueue",
    "InvalidURL",
    "InvalidUrlClientError",
    "InvalidUrlRedirectClientError",
    "NonHttpUrlClientError",
    "NonHttpUrlRedirectClientError",
    "RedirectClientError",
    "RequestInfo",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ServerTimeoutError",
    "SocketFactoryType",
    "SocketTimeoutError",
    "TCPConnector",
    "TooManyRedirects",
    "UnixConnector",
    "UploadAbortedError",
    "UploadTracker",
    "NamedPipeConnector",
    "WSServerHandshakeError",
    "request",
    # client_middleware
    "ClientMiddlewareType",
    "ClientHandlerType",
    # cookiejar
    "CookieJar",
    "DummyCookieJar",
    # formdata
    "FormData",
    # helpers
    "BasicAuth",
    "ChainMapProxy",
    "DigestAuthMiddleware",
    "ETag",
    "encode_basic_auth",
    "set_zlib_backend",
    # http
    "HttpVersion",
    "HttpVersion10",
    "HttpVersion11",
    "WSMsgType",
    "WSCloseCode",
    "WSMessage",
    "WebSocketError",
    # multipart
    "BadContentDispositionHeader",
    "BadContentDispositionParam",
    "BodyPartReader",
    "MultipartReader",
    "MultipartWriter",
    "content_disposition_filename",
    "parse_content_disposition",
    # payload
    "AsyncIterablePayload",
    "BufferedReaderPayload",
    "BytesIOPayload",
    "BytesPayload",
    "IOBasePayload",
    "JsonPayload",
    "PAYLOAD_REGISTRY",
    "Payload",
    "StringIOPayload",
    "StringPayload",
    "TextIOPayload",
    "get_payload",
    "payload_type",
    # payload_streamer
    "streamer",
    # resolver
    "AsyncResolver",
    "DefaultResolver",
    "ThreadedResolver",
    # streams
    "DataQueue",
    "EMPTY_PAYLOAD",
    "EofStream",
    "StreamReader",
    # tracing
    "TraceConfig",
    "TraceConnectionCreateEndParams",
    "TraceConnectionCreateStartParams",
    "TraceConnectionQueuedEndParams",
    "TraceConnectionQueuedStartParams",
    "TraceConnectionReuseconnParams",
    "TraceDnsCacheHitParams",
    "TraceDnsCacheMissParams",
    "TraceDnsResolveHostEndParams",
    "TraceDnsResolveHostStartParams",
    "TraceRequestChunkSentParams",
    "TraceRequestEndParams",
    "TraceRequestExceptionParams",
    "TraceRequestHeadersSentParams",
    "TraceRequestRedirectParams",
    "TraceRequestStartParams",
    "TraceResponseChunkReceivedParams",
    # workers (imported lazily with __getattr__)
    "GunicornUVLoopWebWorker",
    "GunicornWebWorker",
    "WSMessageTypeError",
)


def __dir__() -> tuple[str, ...]:
    return __all__ + ("__doc__",)


def __getattr__(name: str) -> object:
    global GunicornUVLoopWebWorker, GunicornWebWorker

    # Importing gunicorn takes a long time (>100ms), so only import if actually needed.
    if name in ("GunicornUVLoopWebWorker", "GunicornWebWorker"):
        try:
            from .worker import GunicornUVLoopWebWorker as guv, GunicornWebWorker as gw
        except ImportError:
            return None

        GunicornUVLoopWebWorker = guv  # type: ignore[misc]
        GunicornWebWorker = gw  # type: ignore[misc]
        return guv if name == "GunicornUVLoopWebWorker" else gw

    raise AttributeError(f"module {__name__} has no attribute {name}")
<|endoftext|>import abc
import asyncio
import re
import string
import sys
from contextlib import suppress
from enum import IntEnum
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
)

from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL

from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import (
    HAS_BROTLI,
    HAS_ZSTD,
    BrotliDecompressor,
    ZLibDecompressor,
    ZSTDDecompressor,
)
from .helpers import (
    _EXC_SENTINEL,
    DEBUG,
    DEFAULT_CHUNK_SIZE,
    EMPTY_BODY_METHODS,
    EMPTY_BODY_STATUS_CODES,
    NO_EXTENSIONS,
    BaseTimerContext,
    set_exception,
)
from .http_exceptions import (
    BadHttpMessage,
    BadHttpMethod,
    BadStatusLine,
    ContentEncodingError,
    ContentLengthError,
    InvalidHeader,
    InvalidURLError,
    LineTooLong,
    TransferEncodingError,
)
from .http_writer import HttpVersion, HttpVersion10, HttpVersion11
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders

if TYPE_CHECKING:
    from .client_proto import ResponseHandler

__all__ = (
    "HeadersParser",
    "HttpParser",
    "HttpRequestParser",
    "HttpResponseParser",
    "RawRequestMessage",
    "RawResponseMessage",
)

_SEP = Literal[b"\r\n", b"\n"]

ASCIISET: Final[set[str]] = set(string.printable)

# See https://www.rfc-editor.org/rfc/rfc9110.html#name-overview
# and https://www.rfc-editor.org/rfc/rfc9110.html#name-tokens
#
#     method = token
#     tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." /
#             "^" / "_" / "`" / "|" / "~" / DIGIT / ALPHA
#     token = 1*tchar
_TCHAR_SPECIALS: Final[str] = re.escape("!#$%&'*+-.^_`|~")
TOKENRE: Final[Pattern[str]] = re.compile(f"[0-9A-Za-z{_TCHAR_SPECIALS}]+")
VERSRE: Final[Pattern[str]] = re.compile(r"HTTP/(\d)\.(\d)", re.ASCII)
DIGITS: Final[Pattern[str]] = re.compile(r"\d+", re.ASCII)
HEXDIGITS: Final[Pattern[bytes]] = re.compile(rb"[0-9a-fA-F]+")
# https://www.rfc-editor.org/rfc/rfc9110#section-5.5-5
_FIELD_VALUE_FORBIDDEN_CTL_RE: Final[Pattern[str]] = re.compile(
    r"[\x00-\x08\x0a-\x1f\x7f]"
)
_TARGET_FORBIDDEN_CTL_RE: Final[Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")


def _has_authority(target: str) -> bool:
    """Tell if an absolute-form request target has "//" and a non-empty authority.

    RFC 9110 section 4.2 requires a host for http and https. yarl's default
    WHATWG mode reads a host from "http:host/p" or "http:///host/p", so the
    target is checked before it is parsed.
    """
    rest = target.partition(":")[2]
    return rest[:2] == "//" and rest[2:3] not in ("", "/", "?", "#")


# RFC 9110 singleton headers — duplicates are rejected in strict mode.
# In lax mode (response parser default), the check is skipped entirely
# since real-world servers (e.g. Google APIs, Werkzeug) commonly send
# duplicate headers like Content-Type or Server.
# Lowercased for case-insensitive matching against wire names.
SINGLETON_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "content-length",
        "content-location",
        "content-range",
        "content-type",
        "etag",
        "host",
        "max-forwards",
        "server",
        "transfer-encoding",
        "user-agent",
    }
)


class RawRequestMessage(NamedTuple):
    method: str
    path: str
    version: HttpVersion
    headers: "CIMultiDictProxy[str]"
    raw_headers: RawHeaders
    should_close: bool
    compression: str | None
    upgrade: bool
    chunked: bool
    url: URL


class RawResponseMessage(NamedTuple):
    version: HttpVersion
    code: int
    reason: str
    headers: CIMultiDictProxy[str]
    raw_headers: RawHeaders
    should_close: bool
    compression: str | None
    upgrade: bool
    chunked: bool


_MsgT = TypeVar("_MsgT", RawRequestMessage, RawResponseMessage)


class PayloadState(IntEnum):
    PAYLOAD_COMPLETE = 0
    PAYLOAD_NEEDS_INPUT = 1
    PAYLOAD_HAS_PENDING_INPUT = 2


class ParseState(IntEnum):

    PARSE_NONE = 0
    PARSE_LENGTH = 1
    PARSE_CHUNKED = 2
    PARSE_UNTIL_EOF = 3


class ChunkState(IntEnum):
    PARSE_CHUNKED_SIZE = 0
    PARSE_CHUNKED_CHUNK = 1
    PARSE_CHUNKED_CHUNK_EOF = 2
    PARSE_MAYBE_TRAILERS = 3
    PARSE_TRAILERS = 4


class HeadersParser:
    def __init__(
        self,
        max_line_size: int = 8190,
        max_headers: int = 32768,
        max_field_size: int = 8190,
        lax: bool = False,
    ) -> None:
        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size
        self._lax = lax

    def parse_headers(
        self, lines: list[bytes]
    ) -> tuple["CIMultiDictProxy[str]", RawHeaders]:
        headers: CIMultiDict[str] = CIMultiDict()
        # note: "raw" does not mean inclusion of OWS before/after the field value
        raw_headers = []

        lines_idx = 0
        line = lines[lines_idx]
        line_count = len(lines)

        while line:
            # Parse initial header name : value pair.
            try:
                bname, bvalue = line.split(b":", 1)
            except ValueError:
                raise InvalidHeader(line) from None

            if len(bname) == 0:
                raise InvalidHeader(bname)

            # https://www.rfc-editor.org/rfc/rfc9112.html#section-5.1-2
            if {bname[0], bname[-1]} & {32, 9}:  # {" ", "\t"}
                raise InvalidHeader(line)

            bvalue = bvalue.lstrip(b" \t")
            name = bname.decode("utf-8", "surrogateescape")
            if not TOKENRE.fullmatch(name):
                raise InvalidHeader(bname)

            # next line
            lines_idx += 1
            line = lines[lines_idx]

            # consume continuation lines
            continuation = self._lax and line and line[0] in (32, 9)  # (' ', '\t')

            # Deprecated: https://www.rfc-editor.org/rfc/rfc9112.html#name-obsolete-line-folding
            if continuation:
                header_length = len(bvalue)
                bvalue_lst = [bvalue]
                while continuation:
                    header_length += len(line)
                    if header_length > self.max_field_size:
                        header_line = bname + b": " + b"".join(bvalue_lst)
                        raise LineTooLong(
                            header_line[:100] + b"...", self.max_field_size
                        )
                    bvalue_lst.append(line)

                    # next line
                    lines_idx += 1
                    if lines_idx < line_count:
                        line = lines[lines_idx]
                        if line:
                            continuation = line[0] in (32, 9)  # (' ', '\t')
                    else:
                        line = b""
                        break
                bvalue = b"".join(bvalue_lst)

            bvalue = bvalue.strip(b" \t")
            value = bvalue.decode("utf-8", "surrogateescape")

            # https://www.rfc-editor.org/rfc/rfc9110.html#section-5.5-5
            if self._lax:
                if "\n" in value or "\r" in value or "\x00" in value:
                    raise InvalidHeader(bvalue)
            elif _FIELD_VALUE_FORBIDDEN_CTL_RE.search(value):
                raise InvalidHeader(bvalue)

            if not self._lax and name in headers and name.lower() in SINGLETON_HEADERS:
                raise BadHttpMessage(f"Duplicate '{name}' header found.")
            headers.add(name, value)
            raw_headers.append((bname, bvalue))

        return (CIMultiDictProxy(headers), tuple(raw_headers))


def _is_supported_upgrade(headers: CIMultiDictProxy[str]) -> bool:
    """Check if the upgrade header is supported."""
    u = headers.get(hdrs.UPGRADE, "")
    # .lower() can transform non-ascii characters.
    return u.isascii() and u.lower() in {"tcp", "websocket"}


class HttpParser(abc.ABC, Generic[_MsgT]):
    lax: ClassVar[bool] = False

    def __init__(
        self,
        protocol: BaseProtocol | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        limit: int = 2**16,
        max_line_size: int = 8190,
        max_headers: int = 128,
        max_field_size: int = 8190,
        timer: BaseTimerContext | None = None,
        code: int | None = None,
        method: str | None = None,
        payload_exception: type[BaseException] | None = None,
        response_with_body: bool = True,
        read_until_eof: bool = False,
        auto_decompress: bool = True,
        max_msg_queue_size: int = 0,
    ) -> None:
        self.protocol = protocol
        self.loop = loop
        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size
        self.max_headers = max_headers
        self.timer = timer
        self.code = code
        self.method = method
        self.payload_exception = payload_exception
        self.response_with_body = response_with_body
        self.read_until_eof = read_until_eof

        self._lines: list[bytes] = []
        self._tail = b""
        self._upgraded = False
        self._pending_upgrade = False
        self._payload = None
        self._payload_parser: HttpPayloadParser | None = None
        self._payload_has_more_data = False
        self._auto_decompress = auto_decompress
        self._limit = limit
        self._headers_parser = HeadersParser(
            max_line_size, max_headers, max_field_size, self.lax
        )
        # Stop emitting messages once this many are queued unconsumed (0 = off).
        self._max_msg_queue_size = max_msg_queue_size
        self._msg_in_flight = 0

    @abc.abstractmethod
    def parse_message(self, lines: list[bytes]) -> _MsgT: ...

    @abc.abstractmethod
    def _is_chunked_te(self, te: str) -> bool: ...

    def pause_reading(self) -> None:
        assert self._payload_parser is not None
        self._payload_parser.pause_reading()

    def message_consumed(self) -> None:
        """Protocol drained a queued message; free a slot for parsing."""
        if self._msg_in_flight > 0:
            self._msg_in_flight -= 1

    def feed_eof(self) -> _MsgT | None:
        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            if self._payload_parser.done:
                self._payload_parser = None
        else:
            # try to extract partial message
            if self._tail:
                self._lines.append(self._tail)

            if self._lines:
                if self._lines[-1] != "\r\n":
                    self._lines.append(b"")
                with suppress(Exception):
                    return self.parse_message(self._lines)
        return None

    def feed_data(
        self,
        data: bytes,
        SEP: _SEP = b"\r\n",
        EMPTY: bytes = b"",
        CONTENT_LENGTH: istr = hdrs.CONTENT_LENGTH,
        METH_CONNECT: str = hdrs.METH_CONNECT,
        SEC_WEBSOCKET_KEY1: istr = hdrs.SEC_WEBSOCKET_KEY1,
    ) -> tuple[list[tuple[_MsgT, StreamReader]], bool, bytes]:

        messages = []

        if self._tail:
            data, self._tail = self._tail + data, b""

        data_len = len(data)
        start_pos = 0
        loop = self.loop
        max_line_length = self.max_line_size

        should_close = False
        while start_pos < data_len or self._payload_has_more_data:
            # read HTTP message (request/response line + headers), \r\n\r\n
            # and split by lines
            if self._payload_parser is None and not self._upgraded:
                if (
                    self._max_msg_queue_size
                    and self._msg_in_flight >= self._max_msg_queue_size
                ):
                    # Queue full: buffer the rest and stop. Safe pause point;
                    # any preceding body is consumed before the next request
                    # line. Resumes via feed_data(b"") when the queue drains.
                    self._tail = data[start_pos:]
                    # The remainder now lives in self._tail only. Don't return it.
                    data = EMPTY
                    break
                pos = data.find(SEP, start_pos)
                # consume \r\n
                if pos == start_pos and not self._lines:
                    start_pos = pos + len(SEP)
                    continue

                if pos >= start_pos:
                    if should_close:
                        raise BadHttpMessage("Data after `Connection: close`")

                    # line found
                    line = data[start_pos:pos]
                    if SEP == b"\n":  # For lax response parsing
                        line = line.rstrip(b"\r")
                    if len(line) > max_line_length:
                        raise LineTooLong(line[:100] + b"...", max_line_length)

                    self._lines.append(line)
                    # After processing the status/request line, everything is a header.
                    max_line_length = self.max_field_size

                    if len(self._lines) > self.max_headers:
                        raise BadHttpMessage("Too many headers received")

                    start_pos = pos + len(SEP)

                    # \r\n\r\n found
                    if self._lines[-1] == EMPTY:
                        max_trailers = self.max_headers - len(self._lines)
                        try:
                            msg: _MsgT = self.parse_message(self._lines)
                        finally:
                            self._lines.clear()

                        def get_content_length() -> int | None:
                            # payload length
                            length_hdr = msg.headers.get(CONTENT_LENGTH)
                            if length_hdr is None:
                                return None

                            # Shouldn't allow +/- or other number formats.
                            # https://www.rfc-editor.org/rfc/rfc9110#section-8.6-2
                            # msg.headers is already stripped of leading/trailing wsp
                            if not DIGITS.fullmatch(length_hdr):
                                raise InvalidHeader(CONTENT_LENGTH)

                            return int(length_hdr)

                        length = get_content_length()
                        # do not support old websocket spec
                        if SEC_WEBSOCKET_KEY1 in msg.headers:
                            raise InvalidHeader(SEC_WEBSOCKET_KEY1)

                        upgraded = msg.upgrade and _is_supported_upgrade(msg.headers)

                        method = getattr(msg, "method", self.method)
                        # code is only present on responses
                        code = getattr(msg, "code", 0)

                        assert self.protocol is not None
                        # calculate payload
                        # https://www.rfc-editor.org/info/rfc9112/#name-message-body-length
                        # https://www.rfc-editor.org/info/rfc9110/#section-9.3.1-6
                        # EMPTY_BODY_METHODS should only apply to responses.
                        # self.method is None on request parser.
                        empty_body = code in EMPTY_BODY_STATUS_CODES or bool(
                            self.method and self.method in EMPTY_BODY_METHODS
                        )
                        if not empty_body and (
                            (length is not None and length > 0) or msg.chunked
                        ):
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            payload_parser = HttpPayloadParser(
                                payload,
                                length=length,
                                chunked=msg.chunked,
                                method=method,
                                compression=msg.compression,
                                code=self.code,
                                response_with_body=self.response_with_body,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                                headers_parser=self._headers_parser,
                                max_line_size=self.max_line_size,
                                max_field_size=self.max_field_size,
                                max_trailers=max_trailers,
                                limit=self._limit,
                            )
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                                # https://www.rfc-editor.org/info/rfc9110/#section-7.8-15
                                # Defer any requested upgrade until the
                                # complete request has been read.
                                self._pending_upgrade = upgraded
                        elif method == METH_CONNECT:
                            assert isinstance(msg, RawRequestMessage)
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            self._upgraded = True
                            self._payload_parser = HttpPayloadParser(
                                payload,
                                method=msg.method,
                                compression=msg.compression,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                                headers_parser=self._headers_parser,
                                max_line_size=self.max_line_size,
                                max_field_size=self.max_field_size,
                                max_trailers=max_trailers,
                                limit=self._limit,
                            )
                        elif not empty_body and length is None and self.read_until_eof:
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            payload_parser = HttpPayloadParser(
                                payload,
                                length=length,
                                chunked=msg.chunked,
                                method=method,
                                compression=msg.compression,
                                code=self.code,
                                response_with_body=self.response_with_body,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                                headers_parser=self._headers_parser,
                                max_line_size=self.max_line_size,
                                max_field_size=self.max_field_size,
                                max_trailers=max_trailers,
                                limit=self._limit,
                            )
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        elif upgraded:
                            # No body to read, so the connection switches to
                            # the upgraded protocol immediately.
                            self._upgraded = True
                            payload = EMPTY_PAYLOAD
                        else:
                            payload = EMPTY_PAYLOAD

                        messages.append((msg, payload))
                        if self._max_msg_queue_size:
                            self._msg_in_flight += 1
                        should_close = msg.should_close
                else:
                    self._tail = data[start_pos:]
                    # A bare LF here means CRLF was required:
                    # reject instead of buffering, else a following request's
                    # bytes get appended to this line and leak in the error.
                    if b"\n" in self._tail:
                        raise BadHttpMessage("Bad line ending, expected CRLF")
                    if len(self._tail) > self.max_line_size:
                        raise LineTooLong(self._tail[:100] + b"...", self.max_line_size)
                    data = EMPTY
                    break

            # no parser, just store
            elif self._payload_parser is None and self._upgraded:
                assert not self._lines
                break

            # feed payload
            else:
                assert not self._lines
                assert self._payload_parser is not None
                try:
                    payload_state, data = self._payload_parser.feed_data(
                        data[start_pos:], SEP
                    )
                except Exception as underlying_exc:
                    reraised_exc: BaseException = underlying_exc
                    if self.payload_exception is not None:
                        reraised_exc = self.payload_exception(str(underlying_exc))

                    set_exception(
                        self._payload_parser.payload,
                        reraised_exc,
                        underlying_exc,
                    )

                    payload_state = PayloadState.PAYLOAD_COMPLETE
                    data = b""
                    if isinstance(
                        underlying_exc, (InvalidHeader, TransferEncodingError)
                    ):
                        raise

                self._payload_has_more_data = (
                    payload_state == PayloadState.PAYLOAD_HAS_PENDING_INPUT
                )

                if payload_state is not PayloadState.PAYLOAD_COMPLETE:
                    # We've either consumed all available data, or we're pausing
                    # until the reader buffer is freed up.
                    break

                start_pos = 0
                data_len = len(data)
                self._payload_parser = None
                if self._pending_upgrade:
                    # Body fully read: the deferred upgrade takes effect and
                    # the rest of the connection is the upgraded protocol.
                    self._upgraded = True
                    self._pending_upgrade = False

        if data and start_pos < data_len:
            data = data[start_pos:]
        else:
            data = EMPTY

        return messages, self._upgraded, data

    def parse_headers(
        self, lines: list[bytes]
    ) -> tuple[
        "CIMultiDictProxy[str]", RawHeaders, bool | None, str | None, bool, bool
    ]:
        """Parses RFC 5322 headers from a stream.

        Line continuations are supported. Returns list of header name
        and value pairs. Header name is in upper case.
        """
        headers, raw_headers = self._headers_parser.parse_headers(lines)
        close_conn = None
        encoding = None
        upgrade = False
        chunked = False

        # keep-alive and protocol switching
        # RFC 9110 section 7.6.1 defines Connection as a comma-separated list.
        conn_values = headers.getall(hdrs.CONNECTION, ())
        if conn_values:
            conn_tokens = {
                token.lower()
                for conn_value in conn_values
                for token in (part.strip(" \t") for part in conn_value.split(","))
                if token and token.isascii()
            }

            if "close" in conn_tokens:
                close_conn = True
            elif "keep-alive" in conn_tokens:
                close_conn = False

            # https://www.rfc-editor.org/rfc/rfc9110.html#name-101-switching-protocols
            if "upgrade" in conn_tokens and headers.get(hdrs.UPGRADE):
                upgrade = True

        # encoding
        enc = headers.get(hdrs.CONTENT_ENCODING, "")
        if enc.isascii():  # Must be checked before .lower()
            enc = enc.lower()
            if enc in {"gzip", "deflate", "br", "zstd"}:
                encoding = enc

        # chunking
        te = headers.get(hdrs.TRANSFER_ENCODING)
        if te is not None:
            if self._is_chunked_te(te):
                chunked = True

            if hdrs.CONTENT_LENGTH in headers:
                raise BadHttpMessage(
                    "Transfer-Encoding can't be present with Content-Length",
                )

        return (headers, raw_headers, close_conn, encoding, upgrade, chunked)

    def set_upgraded(self, val: bool) -> None:
        """Set connection upgraded (to websocket) mode.

        :param bool val: new state.
        """
        self._upgraded = val


class HttpRequestParser(HttpParser[RawRequestMessage]):
    """Read request status line.

    Exception .http_exceptions.BadStatusLine
    could be raised in case of any errors in status line.
    Returns RawRequestMessage.
    """

    def parse_message(self, lines: list[bytes]) -> RawRequestMessage:
        # request line
        line = lines[0].decode("utf-8", "surrogateescape")
        try:
            method, path, version = line.split(" ", maxsplit=2)
        except ValueError:
            raise BadHttpMethod(line) from None

        # method
        if not TOKENRE.fullmatch(method):
            raise BadHttpMethod(method)
        method = method.upper()

        # https://www.rfc-editor.org/rfc/rfc9112#section-3.2-4
        if _TARGET_FORBIDDEN_CTL_RE.search(path):
            raise InvalidURLError(
                path.encode(errors="surrogateescape").decode("latin1")
            )

        # version
        match = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o = HttpVersion(int(match.group(1)), int(match.group(2)))

        if method == "CONNECT":
            # authority-form,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.3
            try:
                url = URL.build(authority=path, encoded=True)
                host = url.raw_host
            except ValueError:
                # Duplicated so mypy understands that url must be defined below.
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )
            if host is None:
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )
        elif path.startswith("/"):
            # origin-form,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.1
            path_part, _hash_separator, url_fragment = path.partition("#")
            path_part, _question_mark_separator, qs_part = path_part.partition("?")

            # NOTE: `yarl.URL.build()` is used to mimic what the Cython-based
            # NOTE: parser does, otherwise it results into the same
            # NOTE: HTTP Request-Line input producing different
            # NOTE: `yarl.URL()` objects
            url = URL.build(
                path=path_part,
                query_string=qs_part,
                fragment=url_fragment,
                encoded=True,
            )
        elif path == "*" and method == "OPTIONS":
            # asterisk-form,
            url = URL(path, encoded=True)
        else:
            # absolute-form for proxy maybe,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.2
            if not _has_authority(path):
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )
            try:
                url = URL(path, encoded=True)
                host = url.raw_host
            except ValueError:
                # Duplicated so mypy understands that url must be defined below.
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )
            # https://www.rfc-editor.org/rfc/rfc9110#section-4.2.1-4
            if host is None:
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )

        # read headers
        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines[1:])

        if version_o == HttpVersion11 and hdrs.HOST not in headers:
            raise BadHttpMessage("Missing 'Host' header in request.")

        if close is None:  # then the headers weren't set in the request
            if version_o <= HttpVersion10:  # HTTP 1.0 must asks to not close
                close = True
            else:  # HTTP 1.1 must ask to close.
                close = False

        return RawRequestMessage(
            method,
            path,
            version_o,
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
            url,
        )

    def _is_chunked_te(self, te: str) -> bool:
        te = te.rsplit(",", maxsplit=1)[-1].strip(" \t")
        # .lower() transforms some non-ascii chars, so must check first.
        if te.isascii() and te.lower() == "chunked":
            return True
        # https://www.rfc-editor.org/rfc/rfc9112#section-6.3-2.4.3
        raise BadHttpMessage("Request has invalid `Transfer-Encoding`")


class HttpResponseParser(HttpParser[RawResponseMessage]):
    """Read response status line and headers.

    BadStatusLine could be raised in case of any errors in status line.
    Returns RawResponseMessage.
    """

    protocol: "ResponseHandler"

    # Lax mode should only be enabled on response parser.
    lax = not DEBUG

    def feed_data(
        self,
        data: bytes,
        SEP: _SEP | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[tuple[RawResponseMessage, StreamReader]], bool, bytes]:
        if SEP is None:
            SEP = b"\r\n" if DEBUG else b"\n"
        return super().feed_data(data, SEP, *args, **kwargs)

    def parse_message(self, lines: list[bytes]) -> RawResponseMessage:
        line = lines[0].decode("utf-8", "surrogateescape")
        try:
            version, status = line.split(maxsplit=1)
        except ValueError:
            raise BadStatusLine(line) from None

        try:
            status, reason = status.split(maxsplit=1)
        except ValueError:
            status = status.strip()
            reason = ""

        # version
        match = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o = HttpVersion(int(match.group(1)), int(match.group(2)))

        # The status code is a three-digit ASCII number, no padding
        if len(status) != 3 or not DIGITS.fullmatch(status):
            raise BadStatusLine(line)
        status_i = int(status)

        # read headers
        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines[1:])

        if close is None:
            if version_o <= HttpVersion10:
                close = True
            # https://www.rfc-editor.org/rfc/rfc9112.html#name-message-body-length
            elif 100 <= status_i < 200 or status_i in {204, 304}:
                close = False
            elif hdrs.CONTENT_LENGTH in headers or hdrs.TRANSFER_ENCODING in headers:
                close = False
            else:
                # https://www.rfc-editor.org/rfc/rfc9112.html#section-6.3-2.8
                close = True

        return RawResponseMessage(
            version_o,
            status_i,
            reason.strip(),
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        )

    def _is_chunked_te(self, te: str) -> bool:
        # https://www.rfc-editor.org/rfc/rfc9112#section-6.3-2.4.2
        return te.rsplit(",", maxsplit=1)[-1].strip(" \t").lower() == "chunked"


class HttpPayloadParser:
    def __init__(
        self,
        payload: StreamReader,
        length: int | None = None,
        chunked: bool = False,
        compression: str | None = None,
        code: int | None = None,
        method: str | None = None,
        response_with_body: bool = True,
        auto_decompress: bool = True,
        lax: bool = False,
        *,
        headers_parser: HeadersParser,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        max_trailers: int = 128,
        limit: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._length = 0
        self._paused = False
        self._type = ParseState.PARSE_UNTIL_EOF
        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
        self._chunk_size = 0
        self._chunk_tail = b""
        self._auto_decompress = auto_decompress
        self._lax = lax
        self._headers_parser = headers_parser
        self._max_line_size = max_line_size
        self._max_field_size = max_field_size
        self._max_trailers = max_trailers
        self._more_data_available = False
        self._trailer_lines: list[bytes] = []
        self.done = False
        self._eof_pending = False

        # payload decompression wrapper
        if response_with_body and compression and self._auto_decompress:
            real_payload: StreamReader | DeflateBuffer = DeflateBuffer(
                payload, compression, max_decompress_size=limit
            )
        else:
            real_payload = payload

        # payload parser
        if not response_with_body:
            # don't parse payload if it's not expected to be received
            self._type = ParseState.PARSE_NONE
            real_payload.feed_eof()
            self.done = True
        elif chunked:
            self._type = ParseState.PARSE_CHUNKED
        elif length is not None:
            self._type = ParseState.PARSE_LENGTH
            self._length = length
            self._length_expected = length
            if self._length == 0:
                real_payload.feed_eof()
                self.done = True

        self.payload = real_payload

    def pause_reading(self) -> None:
        self._paused = True

    def feed_eof(self) -> None:
        if self._type == ParseState.PARSE_UNTIL_EOF:
            self._eof_pending = True
            while self._more_data_available:
                if self._paused:
                    self._paused = False
                    return  # Will resume via feed_data(b"") later
                self._more_data_available = self.payload.feed_data(b"", 0)
            self.payload.feed_eof()
            self.done = True
            self._eof_pending = False
        elif self._type == ParseState.PARSE_LENGTH:
            if self._length:
                received = self._length_expected - self._length
                raise ContentLengthError(
                    f"Not enough data to satisfy content length header "
                    f"(received {received} of {self._length_expected} bytes)."
                )
            # Body has already been received, but parser paused.
            while self._more_data_available:
                if self._paused:
                    self._paused = False
                    return  # Will resume via feed_data(b"") later
                self._more_data_available = self.payload.feed_data(b"", 0)
            self.payload.feed_eof()
            self.done = True
        elif self._type == ParseState.PARSE_CHUNKED:
            raise TransferEncodingError(
                "Not enough data to satisfy transfer length header."
            )

    def feed_data(
        self, chunk: bytes, SEP: _SEP = b"\r\n", CHUNK_EXT: bytes = b";"
    ) -> tuple[PayloadState, bytes]:
        """Receive a chunk of data to process.

        Return:
            PayloadState - The current state of payload processing.
                           This function may be called with empty bytes after returning
                           PAYLOAD_HAS_PENDING_INPUT to continue processing after a pause.
            bytes - If payload is complete, this is the unconsumed bytes intended for the
                    next message/payload, b"" otherwise.
        """
        # Read specified amount of bytes
        if self._type == ParseState.PARSE_LENGTH:
            if self._chunk_tail:
                chunk = self._chunk_tail + chunk
                self._chunk_tail = b""

            required = self._length
            self._length = max(required - len(chunk), 0)
            self._more_data_available = self.payload.feed_data(
                chunk[:required], required
            )
            while self._more_data_available:
                if self._paused:
                    self._paused = False
                    self._chunk_tail = chunk[required:]
                    return PayloadState.PAYLOAD_HAS_PENDING_INPUT, b""
                self._more_data_available = self.payload.feed_data(b"", 0)

            if self._length == 0:
                self.payload.feed_eof()
                return PayloadState.PAYLOAD_COMPLETE, chunk[required:]
        # Chunked transfer encoding parser
        elif self._type == ParseState.PARSE_CHUNKED:
            if self._chunk_tail:
                # We should check the length is sane when not processing payload body.
                if self._chunk != ChunkState.PARSE_CHUNKED_CHUNK:
                    max_line_length = self._max_line_size
                    if self._chunk == ChunkState.PARSE_TRAILERS:
                        max_line_length = self._max_field_size
                    if len(self._chunk_tail) > max_line_length:
                        raise LineTooLong(
                            self._chunk_tail[:100] + b"...", max_line_length
                        )

                chunk = self._chunk_tail + chunk
                self._chunk_tail = b""

            while chunk or self._more_data_available:
                # read next chunk size
                if self._chunk == ChunkState.PARSE_CHUNKED_SIZE:
                    pos = chunk.find(SEP)
                    if pos >= 0:
                        # Only chunk-size lines reach here; trailers enforce
                        # _max_field_size separately in PARSE_TRAILERS below.
                        if pos > self._max_line_size:
                            raise LineTooLong(chunk[:100] + b"...", self._max_line_size)
                        i = chunk.find(CHUNK_EXT, 0, pos)
                        if i >= 0:
                            size_b = chunk[:i]  # strip chunk-extensions
                            # Verify no LF in the chunk-extension
                            if b"\n" in (ext := chunk[i:pos]):
                                exc = TransferEncodingError(
                                    f"Unexpected LF in chunk-extension: {ext!r}"
                                )
                                set_exception(self.payload, exc)
                                raise exc
                        else:
                            size_b = chunk[:pos]

                        if self._lax:  # Allow whitespace in lax mode.
                            size_b = size_b.strip()

                        if not re.fullmatch(HEXDIGITS, size_b):
                            exc = TransferEncodingError(
                                chunk[:pos].decode("ascii", "surrogateescape")
                            )
                            set_exception(self.payload, exc)
                            raise exc
                        size = int(bytes(size_b), 16)

                        chunk = chunk[pos + len(SEP) :]
                        if size == 0:  # eof marker
                            self._chunk = ChunkState.PARSE_TRAILERS
                            if self._lax and chunk.startswith(b"\r"):
                                chunk = chunk[1:]
                        else:
                            self._chunk = ChunkState.PARSE_CHUNKED_CHUNK
                            self._chunk_size = size
                            self.payload.begin_http_chunk_receiving()
                    else:
                        if b"\n" in chunk:
                            exc = TransferEncodingError(
                                "Bad chunk-size line ending, expected CRLF"
                            )
                            set_exception(self.payload, exc)
                            raise exc
                        self._chunk_tail = chunk
                        return PayloadState.PAYLOAD_NEEDS_INPUT, b""

                # read chunk and feed buffer
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK:
                    if self._paused:
                        self._paused = False
                        self._chunk_tail = chunk
                        return PayloadState.PAYLOAD_HAS_PENDING_INPUT, b""

                    required = self._chunk_size
                    self._chunk_size = max(required - len(chunk), 0)
                    self._more_data_available = self.payload.feed_data(
                        chunk[:required], required
                    )
                    chunk = chunk[required:]

                    if self._more_data_available:
                        continue

                    if self._chunk_size:
                        self._paused = False
                        return PayloadState.PAYLOAD_NEEDS_INPUT, b""
                    self._chunk = ChunkState.PARSE_CHUNKED_CHUNK_EOF
                    self.payload.end_http_chunk_receiving()

                # toss the CRLF at the end of the chunk
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK_EOF:
                    if self._lax and chunk.startswith(b"\r"):
                        chunk = chunk[1:]
                    if chunk[: len(SEP)] == SEP:
                        chunk = chunk[len(SEP) :]
                        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
                    elif len(chunk) >= len(SEP) or chunk != SEP[: len(chunk)]:
                        exc = TransferEncodingError(
                            "Chunk size mismatch: expected CRLF after chunk data"
                        )
                        set_exception(self.payload, exc)
                        raise exc
                    else:
                        self._chunk_tail = chunk
                        return PayloadState.PAYLOAD_NEEDS_INPUT, b""

                if self._chunk == ChunkState.PARSE_TRAILERS:
                    pos = chunk.find(SEP)
                    if pos < 0:  # No line found
                        if b"\n" in chunk:
                            exc = TransferEncodingError(
                                "Bad trailer line ending, expected CRLF"
                            )
                            set_exception(self.payload, exc)
                            raise exc
                        self._chunk_tail = chunk
                        return PayloadState.PAYLOAD_NEEDS_INPUT, b""

                    line = chunk[:pos]
                    chunk = chunk[pos + len(SEP) :]
                    if SEP == b"\n":  # For lax response parsing
                        line = line.rstrip(b"\r")

                    if len(line) > self._max_field_size:
                        raise LineTooLong(line[:100] + b"...", self._max_field_size)

                    self._trailer_lines.append(line)

                    if len(self._trailer_lines) > self._max_trailers:
                        raise BadHttpMessage("Too many trailers received")

                    # \r\n\r\n found, end of stream
                    if self._trailer_lines[-1] == b"":
                        # Headers and trailers are defined the same way,
                        # so we reuse the HeadersParser here.
                        try:
                            trailers, raw_trailers = self._headers_parser.parse_headers(
                                self._trailer_lines
                            )
                        finally:
                            self._trailer_lines.clear()
                        self.payload.feed_eof()
                        return PayloadState.PAYLOAD_COMPLETE, chunk

        # Read all bytes until eof
        elif self._type == ParseState.PARSE_UNTIL_EOF:
            self._more_data_available = self.payload.feed_data(chunk, len(chunk))
            while self._more_data_available:
                if self._paused:
                    self._paused = False
                    return PayloadState.PAYLOAD_HAS_PENDING_INPUT, b""
                self._more_data_available = self.payload.feed_data(b"", 0)

            if self._eof_pending:
                self.payload.feed_eof()
                self.done = True
                self._eof_pending = False
                return PayloadState.PAYLOAD_COMPLETE, b""

        return PayloadState.PAYLOAD_NEEDS_INPUT, b""


class DeflateBuffer:
    """DeflateStream decompress stream and feed data into specified stream."""

    decompressor: Any

    def __init__(
        self,
        out: StreamReader,
        encoding: str | None,
        max_decompress_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.out = out
        self.size = 0
        out.total_compressed_bytes = self.size
        self.encoding = encoding
        self._started_decoding = False

        self.decompressor: BrotliDecompressor | ZLibDecompressor | ZSTDDecompressor
        if encoding == "br":
            if not HAS_BROTLI:  # pragma: no cover
                raise ContentEncodingError(
                    "Can not decode content-encoding: brotli (br). "
                    "Please install `Brotli`"
                )
            self.decompressor = BrotliDecompressor()
        elif encoding == "zstd":
            if not HAS_ZSTD:
                raise ContentEncodingError(
                    "Can not decode content-encoding: zstandard (zstd). "
                    "Please install `backports.zstd`"
                )
            self.decompressor = ZSTDDecompressor()
        else:
            self.decompressor = ZLibDecompressor(encoding=encoding)

        self._max_decompress_size = max_decompress_size

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        set_exception(self.out, exc, exc_cause)

    def feed_data(self, chunk: bytes, size: int) -> bool:
        self.size += size
        self.out.total_compressed_bytes = self.size

        # Inspect the first real byte once to choose the decompressor. An empty
        # chunk (e.g. a chunk-size line arriving without body bytes) has no
        # header to sniff, so skip it and wait for the first data byte.
        if not self._started_decoding and chunk:
            # RFC1950
            # bits 0..3 = CM = 0b1000 = 8 = "deflate"
            # bits 4..7 = CINFO = 1..7 = windows size.
            if self.encoding == "deflate" and chunk[0] & 0xF != 8:
                # Change the decoder to decompress incorrectly compressed data
                # Actually we should issue a warning about non-RFC-compliant data.
                self.decompressor = ZLibDecompressor(
                    encoding=self.encoding, suppress_deflate_header=True
                )
            self._started_decoding = True

        low_water = self.out._low_water
        max_length = (
            0 if low_water >= sys.maxsize else max(self._max_decompress_size, low_water)
        )
        try:
            chunk = self.decompressor.decompress_sync(chunk, max_length=max_length)
        except Exception:
            raise ContentEncodingError(
                "Can not decode content-encoding: %s" % self.encoding
            )

        if chunk:
            self.out.feed_data(chunk, len(chunk))
        return self.decompressor.data_available  # type: ignore[no-any-return]

    def feed_eof(self) -> None:
        chunk = self.decompressor.flush()
        # This should never contain data as we defer the call until exhausting
        # the decompression. If .flush() is returning data, this may indicate a
        # zip bomb vulnerability as it will decompress all remaining data at once.
        assert not chunk

        if self.size > 0:
            if self.encoding == "deflate" and not self.decompressor.eof:
                raise ContentEncodingError("deflate")

        self.out.feed_eof()

    def begin_http_chunk_receiving(self) -> None:
        self.out.begin_http_chunk_receiving()

    def end_http_chunk_receiving(self) -> None:
        self.out.end_http_chunk_receiving()


HttpRequestParserPy = HttpRequestParser
HttpResponseParserPy = HttpResponseParser
RawRequestMessagePy = RawRequestMessage
RawResponseMessagePy = RawResponseMessage

try:
    if not NO_EXTENSIONS:
        from ._http_parser import (  # type: ignore[import-not-found,no-redef]
            HttpRequestParser,
            HttpResponseParser,
            RawRequestMessage,
            RawResponseMessage,
        )

        HttpRequestParserC = HttpRequestParser
        HttpResponseParserC = HttpResponseParser
        RawRequestMessageC = RawRequestMessage
        RawResponseMessageC = RawResponseMessage
except ImportError:  # pragma: no cover
    pass
<|endoftext|>import asyncio
import collections
import sys
import warnings
from collections.abc import Awaitable, Callable
from typing import Final, Generic, TypeVar

from .base_protocol import BaseProtocol
from .helpers import (
    _EXC_SENTINEL,
    DEFAULT_CHUNK_SIZE,
    BaseTimerContext,
    TimerNoop,
    set_exception,
    set_result,
)
from .http_exceptions import LineTooLong
from .log import internal_logger

__all__ = (
    "EMPTY_PAYLOAD",
    "EofStream",
    "StreamReader",
    "DataQueue",
)

_T = TypeVar("_T")


class EofStream(Exception):
    """eof stream indication."""


class AsyncStreamIterator(Generic[_T]):

    __slots__ = ("read_func",)

    def __init__(self, read_func: Callable[[], Awaitable[_T]]) -> None:
        self.read_func = read_func

    def __aiter__(self) -> "AsyncStreamIterator[_T]":
        return self

    async def __anext__(self) -> _T:
        try:
            rv = await self.read_func()
        except EofStream:
            raise StopAsyncIteration
        if rv == b"":
            raise StopAsyncIteration
        return rv


class ChunkTupleAsyncStreamIterator:

    __slots__ = ("_stream",)

    def __init__(self, stream: "StreamReader") -> None:
        self._stream = stream

    def __aiter__(self) -> "ChunkTupleAsyncStreamIterator":
        return self

    async def __anext__(self) -> tuple[bytes, bool]:
        rv = await self._stream.readchunk()
        if rv == (b"", False):
            raise StopAsyncIteration
        return rv


class StreamReader:
    """An enhancement of asyncio.StreamReader.

    Supports asynchronous iteration by line, chunk or as available::

        async for line in reader:
            ...
        async for chunk in reader.iter_chunked(1024):
            ...
        async for slice in reader.iter_any():
            ...

    """

    __slots__ = (
        "_protocol",
        "_low_water",
        "_high_water",
        "_low_water_chunks",
        "_high_water_chunks",
        "_loop",
        "_size",
        "_cursor",
        "_http_chunk_splits",
        "_buffer",
        "_buffer_offset",
        "_eof",
        "_waiter",
        "_eof_waiter",
        "_exception",
        "_timer",
        "_eof_callbacks",
        "_eof_counter",
        "total_bytes",
        "total_compressed_bytes",
    )

    def __init__(
        self,
        protocol: BaseProtocol,
        limit: int,
        *,
        timer: BaseTimerContext | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._protocol = protocol
        self._low_water = limit
        self._high_water = limit * 2
        if loop is None:
            loop = asyncio.get_event_loop()
        # Use max(4, ...) because there's always at least 1 chunk split remaining
        # (the current position), so we need low_water >= 2 to allow resume.
        # limit // 16 gets us a reasonable value of 16k with default 256KiB limit.
        self._high_water_chunks = max(4, limit // 16)
        self._low_water_chunks = self._high_water_chunks // 2
        self._loop = loop
        self._size = 0
        self._cursor = 0
        self._http_chunk_splits: collections.deque[int] | None = None
        self._buffer: collections.deque[bytes] = collections.deque()
        self._buffer_offset = 0
        self._eof = False
        self._waiter: asyncio.Future[None] | None = None
        self._eof_waiter: asyncio.Future[None] | None = None
        self._exception: BaseException | None = None
        self._timer = TimerNoop() if timer is None else timer
        self._eof_callbacks: list[Callable[[], None]] = []
        self._eof_counter = 0
        self.total_bytes = 0
        self.total_compressed_bytes: int | None = None

    def __repr__(self) -> str:
        info = [self.__class__.__name__]
        if self._size:
            info.append("%d bytes" % self._size)
        if self._eof:
            info.append("eof")
        if self._low_water != DEFAULT_CHUNK_SIZE:
            info.append("low=%d high=%d" % (self._low_water, self._high_water))
        if self._waiter:
            info.append("w=%r" % self._waiter)
        if self._exception:
            info.append("e=%r" % self._exception)
        return "<%s>" % " ".join(info)

    def __aiter__(self) -> AsyncStreamIterator[bytes]:
        return AsyncStreamIterator(self.readline)

    def iter_chunked(self, n: int) -> AsyncStreamIterator[bytes]:
        """Returns an asynchronous iterator that yields chunks of size n."""
        self.set_read_chunk_size(n)
        return AsyncStreamIterator(lambda: self.read(n))

    def iter_any(self) -> AsyncStreamIterator[bytes]:
        """Yield all available data as soon as it is received."""
        return AsyncStreamIterator(self.readany)

    def iter_chunks(self) -> ChunkTupleAsyncStreamIterator:
        """Yield chunks of data as they are received by the server.

        The yielded objects are tuples
        of (bytes, bool) as returned by the StreamReader.readchunk method.
        """
        return ChunkTupleAsyncStreamIterator(self)

    def get_read_buffer_limits(self) -> tuple[int, int]:
        return (self._low_water, self._high_water)

    def set_read_chunk_size(self, n: int) -> None:
        """Raise buffer limits to match the consumer's chunk size."""
        if n > self._low_water:
            self._low_water = n
            self._high_water = n * 2

    def exception(self) -> BaseException | None:
        return self._exception

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        self._exception = exc
        self._eof_callbacks.clear()

        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_exception(waiter, exc, exc_cause)

        waiter = self._eof_waiter
        if waiter is not None:
            self._eof_waiter = None
            set_exception(waiter, exc, exc_cause)

    def on_eof(self, callback: Callable[[], None]) -> None:
        if self._eof:
            try:
                callback()
            except Exception:
                internal_logger.exception("Exception in eof callback")
        else:
            self._eof_callbacks.append(callback)

    def feed_eof(self) -> None:
        self._eof = True

        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)

        waiter = self._eof_waiter
        if waiter is not None:
            self._eof_waiter = None
            set_result(waiter, None)

        # At EOF the parser is done, there won't be unprocessed data.
        self._protocol.resume_reading(resume_parser=False)

        for cb in self._eof_callbacks:
            try:
                cb()
            except Exception:
                internal_logger.exception("Exception in eof callback")

        self._eof_callbacks.clear()

    def is_eof(self) -> bool:
        """Return True if  'feed_eof' was called."""
        return self._eof

    def at_eof(self) -> bool:
        """Return True if the buffer is empty and 'feed_eof' was called."""
        return self._eof and not self._buffer

    async def wait_eof(self) -> None:
        if self._eof:
            return

        assert self._eof_waiter is None
        self._eof_waiter = self._loop.create_future()
        try:
            await self._eof_waiter
        finally:
            self._eof_waiter = None

    @property
    def total_raw_bytes(self) -> int:
        if self.total_compressed_bytes is None:
            return self.total_bytes
        return self.total_compressed_bytes

    def unread_data(self, data: bytes) -> None:
        """rollback reading some data from stream, inserting it to buffer head."""
        warnings.warn(
            "unread_data() is deprecated "
            "and will be removed in future releases (#3260)",
            DeprecationWarning,
            stacklevel=2,
        )
        if not data:
            return

        if self._buffer_offset:
            self._buffer[0] = self._buffer[0][self._buffer_offset :]
            self._buffer_offset = 0
        self._size += len(data)
        self._cursor -= len(data)
        self._buffer.appendleft(data)
        self._eof_counter = 0

    # TODO: size is ignored, remove the param later
    def feed_data(self, data: bytes, size: int = 0) -> bool:
        assert not self._eof, "feed_data after feed_eof"

        if not data:
            return False

        data_len = len(data)
        self._size += data_len
        self._buffer.append(data)
        self.total_bytes += data_len

        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)

        if self._size > self._high_water:
            self._protocol.pause_reading()
        return False

    def begin_http_chunk_receiving(self) -> None:
        if self._http_chunk_splits is None:
            if self.total_bytes:
                raise RuntimeError(
                    "Called begin_http_chunk_receiving when some data was already fed"
                )
            self._http_chunk_splits = collections.deque()

    def end_http_chunk_receiving(self) -> None:
        if self._http_chunk_splits is None:
            raise RuntimeError(
                "Called end_chunk_receiving without calling "
                "begin_chunk_receiving first"
            )

        # self._http_chunk_splits contains logical byte offsets from start of
        # the body transfer. Each offset is the offset of the end of a chunk.
        # "Logical" means bytes, accessible for a user.
        # If no chunks containing logical data were received, current position
        # is difinitely zero.
        pos = self._http_chunk_splits[-1] if self._http_chunk_splits else 0

        if self.total_bytes == pos:
            # We should not add empty chunks here. So we check for that.
            # Note, when chunked + gzip is used, we can receive a chunk
            # of compressed data, but that data may not be enough for gzip FSM
            # to yield any uncompressed data. That's why current position may
            # not change after receiving a chunk.
            return

        self._http_chunk_splits.append(self.total_bytes)

        # If we get too many small chunks before self._high_water is reached, then any
        # .read() call becomes computationally expensive, and could block the event loop
        # for too long, hence an additional self._high_water_chunks here.
        if len(self._http_chunk_splits) > self._high_water_chunks:
            self._protocol.pause_reading()

        # wake up readchunk when end of http chunk received
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)

    async def _wait(self, func_name: str) -> None:
        if not self._protocol.connected:
            raise RuntimeError("Connection closed.")

        # StreamReader uses a future to link the protocol feed_data() method
        # to a read coroutine. Running two read coroutines at the same time
        # would have an unexpected behaviour. It would not possible to know
        # which coroutine would get the next data.
        if self._waiter is not None:
            raise RuntimeError(
                "%s() called while another coroutine is "
                "already waiting for incoming data" % func_name
            )

        waiter = self._waiter = self._loop.create_future()
        try:
            with self._timer:
                await waiter
        finally:
            self._waiter = None

    async def readline(self, *, max_line_length: int | None = None) -> bytes:
        return await self.readuntil(max_size=max_line_length)

    async def readuntil(
        self, separator: bytes = b"\n", *, max_size: int | None = None
    ) -> bytes:
        seplen = len(separator)
        if seplen == 0:
            raise ValueError("Separator should be at least one-byte string")

        if self._exception is not None:
            raise self._exception

        chunk = b""
        chunk_size = 0
        not_enough = True
        max_size = max_size or self._high_water

        while not_enough:
            while self._buffer and not_enough:
                offset = self._buffer_offset
                n = -1
                if chunk and seplen > 1:
                    # The separator may straddle data already read and this chunk.
                    # tail and head are each shorter than the separator, so a match
                    # in tail + head must span both; n counts its bytes in head.
                    tail = chunk[1 - seplen :]
                    head = self._buffer[0][offset : offset + seplen - 1]
                    ichar = (tail + head).find(separator)
                    if ichar != -1:
                        n = ichar + seplen - len(tail)
                if n == -1:
                    ichar = self._buffer[0].find(separator, offset)
                    if ichar != -1:
                        n = ichar - offset + seplen
                # Read from current offset to found separator or to the end.
                data = self._read_nowait_chunk(n)
                chunk += data
                chunk_size += len(data)
                if n != -1:
                    not_enough = False

                if chunk_size > max_size:
                    raise LineTooLong(chunk[:100] + b"...", max_size)

            if self._eof:
                break

            if not_enough:
                await self._wait("readuntil")

        return chunk

    async def read(self, n: int = -1) -> bytes:
        if self._exception is not None:
            raise self._exception

        # migration problem; with DataQueue you have to catch
        # EofStream exception, so common way is to run payload.read() inside
        # infinite loop. what can cause real infinite loop with StreamReader
        # lets keep this code one major release.
        if __debug__:
            if self._eof and not self._buffer:
                self._eof_counter = getattr(self, "_eof_counter", 0) + 1
                if self._eof_counter > 5:
                    internal_logger.warning(
                        "Multiple access to StreamReader in eof state, "
                        "might be infinite loop.",
                        stack_info=True,
                    )

        if not n:
            return b""

        if n < 0:
            # Reading everything — remove decompression chunk limit.
            self.set_read_chunk_size(sys.maxsize)
            blocks = []
            while True:
                block = await self.readany()
                if not block:
                    break
                blocks.append(block)
            return b"".join(blocks)

        self.set_read_chunk_size(n)
        # TODO: should be `if` instead of `while`
        # because waiter maybe triggered on chunk end,
        # without feeding any data
        while not self._buffer and not self._eof:
            await self._wait("read")

        return self._read_nowait(n)

    async def readany(self) -> bytes:
        if self._exception is not None:
            raise self._exception

        # TODO: should be `if` instead of `while`
        # because waiter maybe triggered on chunk end,
        # without feeding any data
        while not self._buffer and not self._eof:
            await self._wait("readany")

        return self._read_nowait(-1)

    async def readchunk(self) -> tuple[bytes, bool]:
        """Returns a tuple of (data, end_of_http_chunk).

        When chunked transfer
        encoding is used, end_of_http_chunk is a boolean indicating if the end
        of the data corresponds to the end of a HTTP chunk , otherwise it is
        always False.
        """
        while True:
            if self._exception is not None:
                raise self._exception

            while self._http_chunk_splits:
                pos = self._http_chunk_splits.popleft()
                if pos == self._cursor:
                    return (b"", True)
                if pos > self._cursor:
                    return (self._read_nowait(pos - self._cursor), True)
                internal_logger.warning(
                    "Skipping HTTP chunk end due to data "
                    "consumption beyond chunk boundary"
                )

            if self._buffer:
                return (self._read_nowait_chunk(-1), False)
                # return (self._read_nowait(-1), False)

            if self._eof:
                # Special case for signifying EOF.
                # (b'', True) is not a final return value actually.
                return (b"", False)

            await self._wait("readchunk")

    async def readexactly(self, n: int) -> bytes:
        if self._exception is not None:
            raise self._exception

        blocks: list[bytes] = []
        while n > 0:
            block = await self.read(n)
            if not block:
                partial = b"".join(blocks)
                raise asyncio.IncompleteReadError(partial, len(partial) + n)
            blocks.append(block)
            n -= len(block)

        return b"".join(blocks)

    def read_nowait(self, n: int = -1) -> bytes:
        # default was changed to be consistent with .read(-1)
        #
        # I believe the most users don't know about the method and
        # they are not affected.
        if self._exception is not None:
            raise self._exception

        if self._waiter and not self._waiter.done():
            raise RuntimeError(
                "Called while some coroutine is waiting for incoming data."
            )

        return self._read_nowait(n)

    def _read_nowait_chunk(self, n: int) -> bytes:
        first_buffer = self._buffer[0]
        offset = self._buffer_offset
        if n != -1 and len(first_buffer) - offset > n:
            data = first_buffer[offset : offset + n]
            self._buffer_offset += n

        elif offset:
            self._buffer.popleft()
            data = first_buffer[offset:]
            self._buffer_offset = 0

        else:
            data = self._buffer.popleft()

        data_len = len(data)
        self._size -= data_len
        self._cursor += data_len

        chunk_splits = self._http_chunk_splits
        # Prevent memory leak: drop useless chunk splits
        while chunk_splits and chunk_splits[0] < self._cursor:
            chunk_splits.popleft()

        if self._size < self._low_water and (
            self._http_chunk_splits is None
            or len(self._http_chunk_splits) < self._low_water_chunks
        ):
            self._protocol.resume_reading()
        return data

    def _read_nowait(self, n: int) -> bytes:
        """Read not more than n bytes, or whole buffer if n == -1"""
        self._timer.assert_timeout()

        if n == -1:
            # Drain only chunks present now; _read_nowait_chunk() can
            # re-entrantly resume_reading() and refill the buffer.
            count = len(self._buffer)
            if count == 1:
                return self._read_nowait_chunk(-1)
            return b"".join([self._read_nowait_chunk(-1) for _ in range(count)])

        chunks: list[bytes] = []
        while self._buffer:
            chunk = self._read_nowait_chunk(n)
            chunks.append(chunk)
            n -= len(chunk)
            if n == 0:
                break

        return b"".join(chunks) if chunks else b""


class EmptyStreamReader(StreamReader):  # lgtm [py/missing-call-to-init]

    __slots__ = ("_read_eof_chunk",)

    def __init__(self) -> None:
        self._read_eof_chunk = False
        self.total_bytes = 0

    def __repr__(self) -> str:
        return "<%s>" % self.__class__.__name__

    def exception(self) -> BaseException | None:
        return None

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        pass

    def on_eof(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            internal_logger.exception("Exception in eof callback")

    def feed_eof(self) -> None:
        pass

    def is_eof(self) -> bool:
        return True

    def at_eof(self) -> bool:
        return True

    async def wait_eof(self) -> None:
        return

    def feed_data(self, data: bytes, n: int = 0) -> bool:
        return False

    def set_read_chunk_size(self, n: int) -> None:
        return

    async def readline(self, *, max_line_length: int | None = None) -> bytes:
        return b""

    async def read(self, n: int = -1) -> bytes:
        return b""

    # TODO add async def readuntil

    async def readany(self) -> bytes:
        return b""

    async def readchunk(self) -> tuple[bytes, bool]:
        if not self._read_eof_chunk:
            self._read_eof_chunk = True
            return (b"", False)

        return (b"", True)

    async def readexactly(self, n: int) -> bytes:
        raise asyncio.IncompleteReadError(b"", n)

    def read_nowait(self, n: int = -1) -> bytes:
        return b""


EMPTY_PAYLOAD: Final[StreamReader] = EmptyStreamReader()


class DataQueue(Generic[_T]):
    """DataQueue is a general-purpose blocking queue with one reader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._eof = False
        self._waiter: asyncio.Future[None] | None = None
        self._exception: BaseException | None = None
        self._buffer: collections.deque[tuple[_T, int]] = collections.deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_eof(self) -> bool:
        return self._eof

    def at_eof(self) -> bool:
        return self._eof and not self._buffer

    def exception(self) -> BaseException | None:
        return self._exception

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        self._eof = True
        self._exception = exc
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_exception(waiter, exc, exc_cause)

    def feed_data(self, data: _T, size: int = 0) -> None:
        self._buffer.append((data, size))
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_result(waiter, None)

    def feed_eof(self) -> None:
        self._eof = True
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_result(waiter, None)

    async def read(self) -> _T:
        if not self._buffer and not self._eof:
            assert not self._waiter
            self._waiter = self._loop.create_future()
            try:
                await self._waiter
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._waiter = None
                raise
        if self._buffer:
            data, _ = self._buffer.popleft()
            return data
        if self._exception is not None:
            raise self._exception
        raise EofStream

    def __aiter__(self) -> AsyncStreamIterator[_T]:
        return AsyncStreamIterator(self.read)


class FlowControlDataQueue(DataQueue[_T]):
    """FlowControlDataQueue resumes and pauses an underlying stream.

    It is a destination for parsed data.

    This class is deprecated and will be removed in version 4.0.
    """

    def __init__(
        self, protocol: BaseProtocol, limit: int, *, loop: asyncio.AbstractEventLoop
    ) -> None:
        super().__init__(loop=loop)
        self._size = 0
        self._protocol = protocol
        self._limit = limit * 2

    def feed_data(self, data: _T, size: int = 0) -> None:
        super().feed_data(data, size)
        self._size += size

        if self._size > self._limit and not self._protocol._reading_paused:
            self._protocol.pause_reading()

    async def read(self) -> _T:
        if not self._buffer and not self._eof:
            assert not self._waiter
            self._waiter = self._loop.create_future()
            try:
                await self._waiter
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._waiter = None
                raise
        if self._buffer:
            data, size = self._buffer.popleft()
            self._size -= size
            if self._size < self._limit and self._protocol._reading_paused:
                self._protocol.resume_reading()
            return data
        if self._exception is not None:
            raise self._exception
        raise EofStream
<|endoftext|>import sys
from collections.abc import Mapping
from http import HTTPStatus

from . import __version__
from .http_exceptions import HttpProcessingError as HttpProcessingError
from .http_parser import (
    HeadersParser as HeadersParser,
    HttpParser as HttpParser,
    HttpRequestParser as HttpRequestParser,
    HttpResponseParser as HttpResponseParser,
    RawRequestMessage as RawRequestMessage,
    RawResponseMessage as RawResponseMessage,
)
from .http_websocket import (
    WS_CLOSED_MESSAGE as WS_CLOSED_MESSAGE,
    WS_CLOSING_MESSAGE as WS_CLOSING_MESSAGE,
    WS_KEY as WS_KEY,
    WebSocketError as WebSocketError,
    WebSocketReader as WebSocketReader,
    WebSocketWriter as WebSocketWriter,
    WSCloseCode as WSCloseCode,
    WSMessage as WSMessage,
    WSMessageDecodeText as WSMessageDecodeText,
    WSMessageNoDecodeText as WSMessageNoDecodeText,
    WSMessageTextBytes as WSMessageTextBytes,
    WSMsgType as WSMsgType,
    ws_ext_gen as ws_ext_gen,
    ws_ext_parse as ws_ext_parse,
)
from .http_writer import (
    HttpVersion as HttpVersion,
    HttpVersion10 as HttpVersion10,
    HttpVersion11 as HttpVersion11,
    StreamWriter as StreamWriter,
)

__all__ = (
    "HttpProcessingError",
    "RESPONSES",
    "SERVER_SOFTWARE",
    # .http_writer
    "StreamWriter",
    "HttpVersion",
    "HttpVersion10",
    "HttpVersion11",
    # .http_parser
    "HeadersParser",
    "HttpParser",
    "HttpRequestParser",
    "HttpResponseParser",
    "RawRequestMessage",
    "RawResponseMessage",
    # .http_websocket
    "WS_CLOSED_MESSAGE",
    "WS_CLOSING_MESSAGE",
    "WS_KEY",
    "WebSocketReader",
    "WebSocketWriter",
    "ws_ext_gen",
    "ws_ext_parse",
    "WSMessage",
    "WSMessageDecodeText",
    "WSMessageNoDecodeText",
    "WSMessageTextBytes",
    "WebSocketError",
    "WSMsgType",
    "WSCloseCode",
)


SERVER_SOFTWARE: str = (
    f"Python/{sys.version_info[0]}.{sys.version_info[1]} aiohttp/{__version__}"
)

RESPONSES: Mapping[int, tuple[str, str]] = {
    v: (v.phrase, v.description) for v in HTTPStatus.__members__.values()
}
<|endoftext|>import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from collections.abc import Iterator, MutableMapping
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast, overload

from multidict import CIMultiDict, istr

from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import MAX_SYNC_CHUNK_SIZE, ZLibCompressor
from .helpers import (
    ETAG_ANY,
    QUOTED_ETAG_RE,
    ETag,
    HeadersMixin,
    ResponseKey,
    must_be_empty_body,
    parse_http_date,
    rfc822_formatted_time,
    sentinel,
    should_remove_content_length,
    validate_etag_value,
)
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONBytesEncoder, JSONEncoder, LooseHeaders

REASON_PHRASES = {http_status.value: http_status.phrase for http_status in HTTPStatus}

__all__ = (
    "ContentCoding",
    "StreamResponse",
    "Response",
    "json_response",
    "json_bytes_response",
)


if TYPE_CHECKING:
    from .web_request import BaseRequest

    BaseClass = MutableMapping[str, Any]
else:
    BaseClass = collections.abc.MutableMapping


_T = TypeVar("_T")


# TODO(py311): Convert to StrEnum for wider use
class ContentCoding(enum.Enum):
    # The content codings that we have support for.
    #
    # Additional registered codings are listed at:
    # https://www.iana.org/assignments/http-parameters/http-parameters.xhtml#content-coding
    deflate = "deflate"
    gzip = "gzip"
    identity = "identity"


CONTENT_CODINGS = {coding.value: coding for coding in ContentCoding}

############################################################
# HTTP Response classes
############################################################


class StreamResponse(MutableMapping[str | ResponseKey[Any], Any], HeadersMixin):

    _body: None | bytes | bytearray | Payload
    _length_check = True
    _body = None
    _keep_alive: bool | None = None
    _chunked: bool = False
    _compression: bool = False
    _compression_strategy: int | None = None
    _compression_force: ContentCoding | None = None
    _req: Optional["BaseRequest"] = None
    _payload_writer: AbstractStreamWriter | None = None
    _eof_sent: bool = False
    _must_be_empty_body: bool | None = None
    _body_length = 0
    _cookies: SimpleCookie | None = None
    _send_headers_immediately = True
    _seen_str_keys: set[str] = set()

    def __init__(
        self,
        *,
        status: int = 200,
        reason: str | None = None,
        headers: LooseHeaders | None = None,
        _real_headers: CIMultiDict[str] | None = None,
    ) -> None:
        """Initialize a new stream response object.

        _real_headers is an internal parameter used to pass a pre-populated
        headers object. It is used by the `Response` class to avoid copying
        the headers when creating a new response object. It is not intended
        to be used by external code.
        """
        self._state: dict[str | ResponseKey[Any], Any] = {}

        if _real_headers is not None:
            self._headers = _real_headers
        elif headers is not None:
            self._headers: CIMultiDict[str] = CIMultiDict(headers)
        else:
            self._headers = CIMultiDict()

        self._set_status(status, reason)

    @property
    def prepared(self) -> bool:
        return self._eof_sent or self._payload_writer is not None

    @property
    def task(self) -> "asyncio.Task[None] | None":
        if self._req:
            return self._req.task
        else:
            return None

    @property
    def status(self) -> int:
        return self._status

    @property
    def chunked(self) -> bool:
        return self._chunked

    @property
    def compression(self) -> bool:
        return self._compression

    @property
    def reason(self) -> str:
        return self._reason

    def set_status(
        self,
        status: int,
        reason: str | None = None,
    ) -> None:
        assert (
            not self.prepared
        ), "Cannot change the response status code after the headers have been sent"
        self._set_status(status, reason)

    def _set_status(self, status: int, reason: str | None) -> None:
        self._status = int(status)
        if reason is None:
            reason = REASON_PHRASES.get(self._status, "")
        elif "\r" in reason or "\n" in reason:
            raise ValueError("Reason cannot contain \\r or \\n")
        self._reason = reason

    @property
    def keep_alive(self) -> bool | None:
        return self._keep_alive

    def force_close(self) -> None:
        self._keep_alive = False

    @property
    def body_length(self) -> int:
        return self._body_length

    @property
    def output_length(self) -> int:
        warnings.warn("output_length is deprecated", DeprecationWarning)
        assert self._payload_writer
        return self._payload_writer.buffer_size

    def enable_chunked_encoding(self, chunk_size: int | None = None) -> None:
        """Enables automatic chunked transfer encoding."""
        if hdrs.CONTENT_LENGTH in self._headers:
            raise RuntimeError(
                "You can't enable chunked encoding when a content length is set"
            )
        if chunk_size is not None:
            warnings.warn("Chunk size is deprecated #1615", DeprecationWarning)
        self._chunked = True

    def enable_compression(
        self,
        force: bool | ContentCoding | None = None,
        strategy: int | None = None,
    ) -> None:
        """Enables response compression encoding."""
        # Backwards compatibility for when force was a bool <0.17.
        if isinstance(force, bool):
            force = ContentCoding.deflate if force else ContentCoding.identity
            warnings.warn(
                "Using boolean for force is deprecated #3318", DeprecationWarning
            )
        elif force is not None:
            assert isinstance(
                force, ContentCoding
            ), "force should one of None, bool or ContentEncoding"

        self._compression = True
        self._compression_force = force
        self._compression_strategy = strategy

    @property
    def headers(self) -> "CIMultiDict[str]":
        return self._headers

    @property
    def cookies(self) -> SimpleCookie:
        if self._cookies is None:
            self._cookies = SimpleCookie()
        return self._cookies

    def set_cookie(
        self,
        name: str,
        value: str,
        *,
        expires: str | None = None,
        domain: str | None = None,
        max_age: int | str | None = None,
        path: str = "/",
        secure: bool | None = None,
        httponly: bool | None = None,
        version: str | None = None,
        samesite: str | None = None,
        partitioned: bool | None = None,
    ) -> None:
        """Set or update response cookie.

        Sets new cookie or updates existent with new value.
        Also updates only those params which are not None.
        """
        if self._cookies is None:
            self._cookies = SimpleCookie()

        self._cookies[name] = value
        c = self._cookies[name]

        if expires is not None:
            c["expires"] = expires
        elif c.get("expires") == "Thu, 01 Jan 1970 00:00:00 GMT":
            del c["expires"]

        if domain is not None:
            c["domain"] = domain

        if max_age is not None:
            c["max-age"] = str(max_age)
        elif "max-age" in c:
            del c["max-age"]

        c["path"] = path

        if secure is not None:
            c["secure"] = secure
        if httponly is not None:
            c["httponly"] = httponly
        if version is not None:
            c["version"] = version
        if samesite is not None:
            c["samesite"] = samesite

        if partitioned is not None:
            c["partitioned"] = partitioned

    def del_cookie(
        self,
        name: str,
        *,
        domain: str | None = None,
        path: str = "/",
        secure: bool | None = None,
        httponly: bool | None = None,
        samesite: str | None = None,
    ) -> None:
        """Delete cookie.

        Creates new empty expired cookie.
        """
        # TODO: do we need domain/path here?
        if self._cookies is not None:
            self._cookies.pop(name, None)
        self.set_cookie(
            name,
            "",
            max_age=0,
            expires="Thu, 01 Jan 1970 00:00:00 GMT",
            domain=domain,
            path=path,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )

    @property
    def content_length(self) -> int | None:
        # Just a placeholder for adding setter
        return super().content_length

    @content_length.setter
    def content_length(self, value: int | None) -> None:
        if value is not None:
            value = int(value)
            if self._chunked:
                raise RuntimeError(
                    "You can't set content length when chunked encoding is enable"
                )
            self._headers[hdrs.CONTENT_LENGTH] = str(value)
        else:
            self._headers.pop(hdrs.CONTENT_LENGTH, None)

    @property
    def content_type(self) -> str:
        # Just a placeholder for adding setter
        return super().content_type

    @content_type.setter
    def content_type(self, value: str) -> None:
        self.content_type  # read header values if needed
        self._content_type = str(value)
        self._generate_content_type_header()

    @property
    def charset(self) -> str | None:
        # Just a placeholder for adding setter
        return super().charset

    @charset.setter
    def charset(self, value: str | None) -> None:
        ctype = self.content_type  # read header values if needed
        if ctype == "application/octet-stream":
            raise RuntimeError(
                "Setting charset for application/octet-stream "
                "doesn't make sense, setup content_type first"
            )
        assert self._content_dict is not None
        if value is None:
            self._content_dict.pop("charset", None)
        else:
            self._content_dict["charset"] = str(value).lower()
        self._generate_content_type_header()

    @property
    def last_modified(self) -> datetime.datetime | None:
        """The value of Last-Modified HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self._headers.get(hdrs.LAST_MODIFIED))

    @last_modified.setter
    def last_modified(
        self, value: int | float | datetime.datetime | str | None
    ) -> None:
        if value is None:
            self._headers.pop(hdrs.LAST_MODIFIED, None)
        elif isinstance(value, (int, float)):
            self._headers[hdrs.LAST_MODIFIED] = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT", time.gmtime(math.ceil(value))
            )
        elif isinstance(value, datetime.datetime):
            if value.microsecond:
                value = value.replace(microsecond=0) + datetime.timedelta(seconds=1)
            self._headers[hdrs.LAST_MODIFIED] = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT", value.utctimetuple()
            )
        elif isinstance(value, str):
            self._headers[hdrs.LAST_MODIFIED] = value
        else:
            msg = f"Unsupported type for last_modified: {type(value).__name__}"
            raise TypeError(msg)

    @property
    def etag(self) -> ETag | None:
        quoted_value = self._headers.get(hdrs.ETAG)
        if not quoted_value:
            return None
        elif quoted_value == ETAG_ANY:
            return ETag(value=ETAG_ANY)
        match = QUOTED_ETAG_RE.fullmatch(quoted_value)
        if not match:
            return None
        is_weak, value = match.group(1, 2)
        return ETag(
            is_weak=bool(is_weak),
            value=value,
        )

    @etag.setter
    def etag(self, value: ETag | str | None) -> None:
        if value is None:
            self._headers.pop(hdrs.ETAG, None)
        elif (isinstance(value, str) and value == ETAG_ANY) or (
            isinstance(value, ETag) and value.value == ETAG_ANY
        ):
            self._headers[hdrs.ETAG] = ETAG_ANY
        elif isinstance(value, str):
            validate_etag_value(value)
            self._headers[hdrs.ETAG] = f'"{value}"'
        elif isinstance(value, ETag) and isinstance(value.value, str):
            validate_etag_value(value.value)
            hdr_value = f'W/"{value.value}"' if value.is_weak else f'"{value.value}"'
            self._headers[hdrs.ETAG] = hdr_value
        else:
            raise ValueError(
                f"Unsupported etag type: {type(value)}. "
                f"etag must be str, ETag or None"
            )

    def _generate_content_type_header(
        self, CONTENT_TYPE: istr = hdrs.CONTENT_TYPE
    ) -> None:
        assert self._content_dict is not None
        assert self._content_type is not None
        params = "; ".join(f"{k}={v}" for k, v in self._content_dict.items())
        if params:
            ctype = self._content_type + "; " + params
        else:
            ctype = self._content_type
        self._headers[CONTENT_TYPE] = ctype

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if coding is ContentCoding.identity:
            return
        assert self._payload_writer is not None
        self._headers[hdrs.CONTENT_ENCODING] = coding.value
        self._payload_writer.enable_compression(
            coding.value, self._compression_strategy
        )
        # Compressed payload may have different content length,
        # remove the header
        self._headers.popall(hdrs.CONTENT_LENGTH, None)

    async def _start_compression(self, request: "BaseRequest") -> None:
        if self._compression_force:
            await self._do_start_compression(self._compression_force)
            return
        # Encoding comparisons should be case-insensitive
        # https://www.rfc-editor.org/rfc/rfc9110#section-8.4.1
        accept_encoding = request.headers.get(hdrs.ACCEPT_ENCODING, "").lower()
        for value, coding in CONTENT_CODINGS.items():
            if value in accept_encoding:
                await self._do_start_compression(coding)
                return

    async def prepare(self, request: "BaseRequest") -> AbstractStreamWriter | None:
        if self._eof_sent:
            return None
        if self._payload_writer is not None:
            return self._payload_writer
        self._must_be_empty_body = must_be_empty_body(request.method, self.status)
        return await self._start(request)

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        self._req = request
        writer = self._payload_writer = request._payload_writer

        await self._prepare_headers()
        await request._prepare_hook(self)
        await self._write_headers()

        return writer

    async def _prepare_headers(self) -> None:
        request = self._req
        assert request is not None
        writer = self._payload_writer
        assert writer is not None
        keep_alive = self._keep_alive
        if keep_alive is None:
            keep_alive = request.keep_alive
        self._keep_alive = keep_alive

        version = request.version

        headers = self._headers
        if self._cookies:
            for cookie in self._cookies.values():
                value = cookie.output(header="")[1:]
                headers.add(hdrs.SET_COOKIE, value)

        if self._compression:
            await self._start_compression(request)

        if self._chunked:
            if version != HttpVersion11:
                raise RuntimeError(
                    "Using chunked encoding is forbidden "
                    f"for HTTP/{request.version.major}.{request.version.minor}"
                )
            if not self._must_be_empty_body:
                writer.enable_chunking()
                headers[hdrs.TRANSFER_ENCODING] = "chunked"
        elif self._length_check:  # Disabled for WebSockets
            writer.length = self.content_length
            if writer.length is None:
                if version >= HttpVersion11:
                    if not self._must_be_empty_body:
                        writer.enable_chunking()
                        headers[hdrs.TRANSFER_ENCODING] = "chunked"
                elif not self._must_be_empty_body:
                    keep_alive = False

        # HTTP 1.1: https://tools.ietf.org/html/rfc7230#section-3.3.2
        # HTTP 1.0: https://tools.ietf.org/html/rfc1945#section-10.4
        if self._must_be_empty_body:
            if hdrs.CONTENT_LENGTH in headers and should_remove_content_length(
                request.method, self.status
            ):
                del headers[hdrs.CONTENT_LENGTH]
            # https://datatracker.ietf.org/doc/html/rfc9112#section-6.1-10
            # https://datatracker.ietf.org/doc/html/rfc9112#section-6.1-13
            if hdrs.TRANSFER_ENCODING in headers:
                del headers[hdrs.TRANSFER_ENCODING]
        elif (writer.length if self._length_check else self.content_length) != 0:
            # https://www.rfc-editor.org/rfc/rfc9110#section-8.3-5
            headers.setdefault(hdrs.CONTENT_TYPE, "application/octet-stream")
        if hdrs.DATE not in headers:
            headers[hdrs.DATE] = rfc822_formatted_time()
        headers.setdefault(hdrs.SERVER, SERVER_SOFTWARE)

        # connection header
        if hdrs.CONNECTION not in headers:
            if keep_alive:
                if version == HttpVersion10:
                    headers[hdrs.CONNECTION] = "keep-alive"
            elif version == HttpVersion11:
                headers[hdrs.CONNECTION] = "close"

    async def _write_headers(self) -> None:
        request = self._req
        assert request is not None
        writer = self._payload_writer
        assert writer is not None
        # status line
        version = request.version
        status_line = f"HTTP/{version[0]}.{version[1]} {self._status} {self._reason}"
        await writer.write_headers(status_line, self._headers)
        # Send headers immediately if not opted into buffering
        if self._send_headers_immediately:
            writer.send_headers()

    async def write(self, data: bytes | bytearray | memoryview) -> None:
        assert isinstance(
            data, (bytes, bytearray, memoryview)
        ), "data argument must be byte-ish (%r)" % type(data)

        if self._eof_sent:
            raise RuntimeError("Cannot call write() after write_eof()")
        if self._payload_writer is None:
            raise RuntimeError("Cannot call write() before prepare()")

        await self._payload_writer.write(data)

    async def drain(self) -> None:
        assert not self._eof_sent, "EOF has already been sent"
        assert self._payload_writer is not None, "Response has not been started"
        warnings.warn(
            "drain method is deprecated, use await resp.write()",
            DeprecationWarning,
            stacklevel=2,
        )
        await self._payload_writer.drain()

    async def write_eof(self, data: bytes = b"") -> None:
        assert isinstance(
            data, (bytes, bytearray, memoryview)
        ), "data argument must be byte-ish (%r)" % type(data)

        if self._eof_sent:
            return

        assert self._payload_writer is not None, "Response has not been started"

        await self._payload_writer.write_eof(data)
        self._eof_sent = True
        self._req = None
        self._body_length = self._payload_writer.output_size
        self._payload_writer = None

    def __repr__(self) -> str:
        if self._eof_sent:
            info = "eof"
        elif self.prepared:
            assert self._req is not None
            info = f"{self._req.method} {self._req.path} "
        else:
            info = "not prepared"
        return f"<{self.__class__.__name__} {self.reason} {info}>"

    @overload  # type: ignore[override]
    def __getitem__(self, key: ResponseKey[_T]) -> _T: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str | ResponseKey[_T]) -> Any:
        return self._state[key]

    @overload  # type: ignore[override]
    def __setitem__(self, key: ResponseKey[_T], value: _T) -> None: ...

    @overload
    def __setitem__(self, key: str, value: Any) -> None: ...

    def __setitem__(self, key: str | ResponseKey[_T], value: Any) -> None:
        if (
            not isinstance(key, ResponseKey)
            and key not in StreamResponse._seen_str_keys
        ):
            # Import here to break circular dependency
            from .web_exceptions import NotAppKeyWarning

            StreamResponse._seen_str_keys.add(key)
            warnings.warn(
                "It is recommended to use web.ResponseKey instances for keys.\n"
                + "https://docs.aiohttp.org/en/stable/web_advanced.html"
                + "#response-s-storage",
                category=NotAppKeyWarning,
                stacklevel=2,
            )
        self._state[key] = value

    def __delitem__(self, key: str | ResponseKey[_T]) -> None:
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self) -> Iterator[str | ResponseKey[Any]]:
        return iter(self._state)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other

    def __bool__(self) -> bool:
        return True


class Response(StreamResponse):

    _compressed_body: bytes | None = None
    _send_headers_immediately = False

    def __init__(
        self,
        *,
        body: Any = None,
        status: int = 200,
        reason: str | None = None,
        text: str | None = None,
        headers: LooseHeaders | None = None,
        content_type: str | None = None,
        charset: str | None = None,
        zlib_executor_size: int = MAX_SYNC_CHUNK_SIZE,
        zlib_executor: Executor | None = None,
    ) -> None:
        if body is not None and text is not None:
            raise ValueError("body and text are not allowed together")

        if headers is None:
            real_headers: CIMultiDict[str] = CIMultiDict()
        else:
            real_headers = CIMultiDict(headers)

        if content_type is not None and "charset" in content_type:
            raise ValueError("charset must not be in content_type argument")

        if text is not None:
            if hdrs.CONTENT_TYPE in real_headers:
                if content_type or charset:
                    raise ValueError(
                        "passing both Content-Type header and "
                        "content_type or charset params "
                        "is forbidden"
                    )
            else:
                # fast path for filling headers
                if not isinstance(text, str):
                    raise TypeError("text argument must be str (%r)" % type(text))
                if content_type is None:
                    content_type = "text/plain"
                if charset is None:
                    charset = "utf-8"
                real_headers[hdrs.CONTENT_TYPE] = content_type + "; charset=" + charset
                body = text.encode(charset)
                text = None
        elif hdrs.CONTENT_TYPE in real_headers:
            if content_type is not None or charset is not None:
                raise ValueError(
                    "passing both Content-Type header and "
                    "content_type or charset params "
                    "is forbidden"
                )
        elif content_type is not None:
            if charset is not None:
                content_type += "; charset=" + charset
            real_headers[hdrs.CONTENT_TYPE] = content_type

        super().__init__(status=status, reason=reason, _real_headers=real_headers)

        if text is not None:
            self.text = text
        else:
            self.body = body

        self._zlib_executor_size = zlib_executor_size
        self._zlib_executor = zlib_executor

    @property
    def body(self) -> bytes | bytearray | Payload | None:
        return self._body

    @body.setter
    def body(self, body: Any) -> None:
        if body is None:
            self._body = None
        elif isinstance(body, (bytes, bytearray)):
            self._body = body
        else:
            try:
                self._body = body = payload.PAYLOAD_REGISTRY.get(body)
            except payload.LookupError:
                raise ValueError("Unsupported body type %r" % type(body))

            headers = self._headers

            # set content-type
            if hdrs.CONTENT_TYPE not in headers:
                headers[hdrs.CONTENT_TYPE] = body.content_type

            # copy payload headers
            if body.headers:
                for key, value in body.headers.items():
                    if key not in headers:
                        headers[key] = value

        self._compressed_body = None

    @property
    def text(self) -> str | None:
        if self._body is None:
            return None
        # Note: When _body is a Payload (e.g. FilePayload), this may do blocking I/O
        # This is generally safe as most common payloads (BytesPayload, StringPayload)
        # don't do blocking I/O, but be careful with file-based payloads
        return self._body.decode(self.charset or "utf-8")

    @text.setter
    def text(self, text: str) -> None:
        assert text is None or isinstance(
            text, str
        ), "text argument must be str (%r)" % type(text)

        if self.content_type == "application/octet-stream":
            self.content_type = "text/plain"
        if self.charset is None:
            self.charset = "utf-8"

        self._body = text.encode(self.charset)
        self._compressed_body = None

    @property
    def content_length(self) -> int | None:
        if self._chunked:
            return None

        if hdrs.CONTENT_LENGTH in self._headers:
            return int(self._headers[hdrs.CONTENT_LENGTH])

        if self._compressed_body is not None:
            # Return length of the compressed body
            return len(self._compressed_body)
        elif isinstance(self._body, Payload):
            # A payload without content length, or a compressed payload
            return None
        elif self._body is not None:
            return len(self._body)
        else:
            return 0

    @content_length.setter
    def content_length(self, value: int | None) -> None:
        raise RuntimeError("Content length is set automatically")

    async def write_eof(self, data: bytes = b"") -> None:
        if self._eof_sent:
            return
        if self._compressed_body is None:
            body = self._body
        else:
            body = self._compressed_body
        assert not data, f"data arg is not supported, got {data!r}"
        assert self._req is not None
        assert self._payload_writer is not None
        if body is None or self._must_be_empty_body:
            await super().write_eof()
        elif isinstance(self._body, Payload):
            try:
                await self._body.write(self._payload_writer)
            finally:
                await self._body.close()
            await super().write_eof()
        else:
            await super().write_eof(cast(bytes, body))

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        if hdrs.CONTENT_LENGTH in self._headers:
            if should_remove_content_length(request.method, self.status):
                del self._headers[hdrs.CONTENT_LENGTH]
        elif not self._chunked:
            if isinstance(self._body, Payload):
                if (size := self._body.size) is not None:
                    self._headers[hdrs.CONTENT_LENGTH] = str(size)
            else:
                body_len = len(self._body) if self._body else "0"
                # https://www.rfc-editor.org/rfc/rfc9110.html#section-8.6-7
                if body_len != "0" or (
                    self.status != 304 and request.method != hdrs.METH_HEAD
                ):
                    self._headers[hdrs.CONTENT_LENGTH] = str(body_len)

        return await super()._start(request)

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if self._chunked or isinstance(self._body, Payload):
            return await super()._do_start_compression(coding)
        if coding is ContentCoding.identity:
            return
        # Instead of using _payload_writer.enable_compression,
        # compress the whole body
        compressor = ZLibCompressor(
            encoding=coding.value,
            max_sync_chunk_size=self._zlib_executor_size,
            executor=self._zlib_executor,
        )
        assert self._body is not None
        self._compressed_body = (
            await compressor.compress(self._body) + compressor.flush()
        )
        self._headers[hdrs.CONTENT_ENCODING] = coding.value
        self._headers[hdrs.CONTENT_LENGTH] = str(len(self._compressed_body))


def json_response(
    data: Any = sentinel,
    *,
    text: str | None = None,
    body: bytes | None = None,
    status: int = 200,
    reason: str | None = None,
    headers: LooseHeaders | None = None,
    content_type: str = "application/json",
    dumps: JSONEncoder = json.dumps,
) -> Response:
    if data is not sentinel:
        if text or body:
            raise ValueError("only one of data, text, or body should be specified")
        else:
            text = dumps(data)
    return Response(
        text=text,
        body=body,
        status=status,
        reason=reason,
        headers=headers,
        content_type=content_type,
    )


def json_bytes_response(
    data: Any = sentinel,
    *,
    dumps: JSONBytesEncoder,
    body: bytes | None = None,
    status: int = 200,
    reason: str | None = None,
    headers: LooseHeaders | None = None,
    content_type: str = "application/json",
) -> Response:
    """Create a JSON response using a bytes-returning encoder.

    Use this when your JSON encoder (like orjson) returns bytes
    instead of str, avoiding the encode/decode overhead.
    """
    if data is not sentinel:
        if body is not None:
            raise ValueError("only one of data or body should be specified")
        else:
            body = dumps(data)
    return Response(
        body=body,
        status=status,
        reason=reason,
        headers=headers,
        content_type=content_type,
    )
<|endoftext|>import asyncio
import signal
import socket
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from yarl import URL

from .abc import AbstractAccessLogger
from .typedefs import PathLike
from .web_app import Application
from .web_log import AccessLogger
from .web_server import Server

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    try:
        from ssl import SSLContext
    except ImportError:  # pragma: no cover
        SSLContext = object  # type: ignore[misc,assignment]

__all__ = (
    "BaseSite",
    "TCPSite",
    "UnixSite",
    "NamedPipeSite",
    "SockSite",
    "BaseRunner",
    "AppRunner",
    "ServerRunner",
    "GracefulExit",
)


class GracefulExit(SystemExit):
    code = 1


def _raise_graceful_exit() -> None:
    raise GracefulExit()


class BaseSite(ABC):
    __slots__ = ("_runner", "_ssl_context", "_backlog", "_server")

    def __init__(
        self,
        runner: "BaseRunner",
        *,
        shutdown_timeout: float = 60.0,
        ssl_context: SSLContext | None = None,
        backlog: int = 128,
    ) -> None:
        if runner.server is None:
            raise RuntimeError("Call runner.setup() before making a site")
        if shutdown_timeout != 60.0:
            msg = "shutdown_timeout should be set on BaseRunner"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            runner._shutdown_timeout = shutdown_timeout
        self._runner = runner
        self._ssl_context = ssl_context
        self._backlog = backlog
        self._server: asyncio.AbstractServer | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass  # pragma: no cover

    @abstractmethod
    async def start(self) -> None:
        self._runner._reg_site(self)

    async def stop(self) -> None:
        self._runner._check_site(self)
        if self._server is not None:  # Maybe not started yet
            self._server.close()

        self._runner._unreg_site(self)


class TCPSite(BaseSite):
    __slots__ = ("_host", "_port", "_bound_port", "_reuse_address", "_reuse_port")

    def __init__(
        self,
        runner: "BaseRunner",
        host: str | None = None,
        port: int | None = None,
        *,
        shutdown_timeout: float = 60.0,
        ssl_context: SSLContext | None = None,
        backlog: int = 128,
        reuse_address: bool | None = None,
        reuse_port: bool | None = None,
    ) -> None:
        super().__init__(
            runner,
            shutdown_timeout=shutdown_timeout,
            ssl_context=ssl_context,
            backlog=backlog,
        )
        self._host = host
        if port is None:
            port = 8443 if self._ssl_context else 8080
        self._port = port
        self._bound_port: int | None = None
        self._reuse_address = reuse_address
        self._reuse_port = reuse_port

    @property
    def port(self) -> int:
        """The port the server is listening on.

        If the server hasn't been started yet, this returns the requested port
        (which might be 0 for a dynamic port).
        After the server starts, it returns the actual bound port. This is
        especially useful when port=0 was requested, as it allows retrieving the
        dynamically assigned port after the site has started.
        """
        if self._bound_port is not None:
            return self._bound_port
        return self._port

    @property
    def name(self) -> str:
        scheme = "https" if self._ssl_context else "http"
        host = "0.0.0.0" if not self._host else self._host
        return str(URL.build(scheme=scheme, host=host, port=self.port))

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(
            server,
            self._host,
            self._port,
            ssl=self._ssl_context,
            backlog=self._backlog,
            reuse_address=self._reuse_address,
            reuse_port=self._reuse_port,
        )
        if self._server.sockets:
            self._bound_port = self._server.sockets[0].getsockname()[1]
        else:
            self._bound_port = self._port


class UnixSite(BaseSite):
    __slots__ = ("_path",)

    def __init__(
        self,
        runner: "BaseRunner",
        path: PathLike,
        *,
        shutdown_timeout: float = 60.0,
        ssl_context: SSLContext | None = None,
        backlog: int = 128,
    ) -> None:
        super().__init__(
            runner,
            shutdown_timeout=shutdown_timeout,
            ssl_context=ssl_context,
            backlog=backlog,
        )
        self._path = path

    @property
    def name(self) -> str:
        scheme = "https" if self._ssl_context else "http"
        return f"{scheme}://unix:{self._path}:"

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_unix_server(
            server,
            self._path,
            ssl=self._ssl_context,
            backlog=self._backlog,
        )


class NamedPipeSite(BaseSite):
    __slots__ = ("_path",)

    def __init__(
        self, runner: "BaseRunner", path: str, *, shutdown_timeout: float = 60.0
    ) -> None:
        loop = asyncio.get_event_loop()
        if not isinstance(
            loop, asyncio.ProactorEventLoop  # type: ignore[attr-defined]
        ):
            raise RuntimeError(
                "Named Pipes only available in proactor loop under windows"
            )
        super().__init__(runner, shutdown_timeout=shutdown_timeout)
        self._path = path

    @property
    def name(self) -> str:
        return self._path

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        _server = await loop.start_serving_pipe(  # type: ignore[attr-defined]
            server, self._path
        )
        self._server = _server[0]


class SockSite(BaseSite):
    __slots__ = ("_sock", "_name")

    def __init__(
        self,
        runner: "BaseRunner",
        sock: socket.socket,
        *,
        shutdown_timeout: float = 60.0,
        ssl_context: SSLContext | None = None,
        backlog: int = 128,
    ) -> None:
        super().__init__(
            runner,
            shutdown_timeout=shutdown_timeout,
            ssl_context=ssl_context,
            backlog=backlog,
        )
        self._sock = sock
        scheme = "https" if self._ssl_context else "http"
        if hasattr(socket, "AF_UNIX") and sock.family == socket.AF_UNIX:
            name = f"{scheme}://unix:{sock.getsockname()}:"
        else:
            host, port = sock.getsockname()[:2]
            name = str(URL.build(scheme=scheme, host=host, port=port))
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(
            server, sock=self._sock, ssl=self._ssl_context, backlog=self._backlog
        )


class BaseRunner(ABC):
    __slots__ = ("_handle_signals", "_kwargs", "_server", "_sites", "_shutdown_timeout")

    def __init__(
        self,
        *,
        handle_signals: bool = False,
        shutdown_timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        self._handle_signals = handle_signals
        self._kwargs = kwargs
        self._server: Server | None = None
        self._sites: list[BaseSite] = []
        self._shutdown_timeout = shutdown_timeout

    @property
    def server(self) -> Server | None:
        return self._server

    @property
    def addresses(self) -> list[Any]:
        ret: list[Any] = []
        for site in self._sites:
            server = site._server
            if server is not None:
                sockets = server.sockets  # type: ignore[attr-defined]
                if sockets is not None:
                    for sock in sockets:
                        ret.append(sock.getsockname())
        return ret

    @property
    def sites(self) -> set[BaseSite]:
        return set(self._sites)

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()

        if self._handle_signals:
            try:
                loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)
                loop.add_signal_handler(signal.SIGTERM, _raise_graceful_exit)
            except NotImplementedError:  # pragma: no cover
                # add_signal_handler is not implemented on Windows
                pass

        self._server = await self._make_server()

    @abstractmethod
    async def shutdown(self) -> None:
        """Call any shutdown hooks to help server close gracefully."""

    async def cleanup(self) -> None:
        # The loop over sites is intentional, an exception on gather()
        # leaves self._sites in unpredictable state.
        # The loop guaranties that a site is either deleted on success or
        # still present on failure
        for site in list(self._sites):
            await site.stop()

        if self._server:  # If setup succeeded
            # Yield to event loop to ensure incoming requests prior to stopping the sites
            # have all started to be handled before we proceed to close idle connections.
            await asyncio.sleep(0)
            self._server.pre_shutdown()
            await self.shutdown()
            await self._server.shutdown(self._shutdown_timeout)
        await self._cleanup_server()

        self._server = None
        if self._handle_signals:
            loop = asyncio.get_running_loop()
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except NotImplementedError:  # pragma: no cover
                # remove_signal_handler is not implemented on Windows
                pass

    @abstractmethod
    async def _make_server(self) -> Server:
        pass  # pragma: no cover

    @abstractmethod
    async def _cleanup_server(self) -> None:
        pass  # pragma: no cover

    def _reg_site(self, site: BaseSite) -> None:
        if site in self._sites:
            raise RuntimeError(f"Site {site} is already registered in runner {self}")
        self._sites.append(site)

    def _check_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f"Site {site} is not registered in runner {self}")

    def _unreg_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f"Site {site} is not registered in runner {self}")
        self._sites.remove(site)


class ServerRunner(BaseRunner):
    """Low-level web server runner"""

    __slots__ = ("_web_server",)

    def __init__(
        self, web_server: Server, *, handle_signals: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._web_server = web_server

    async def shutdown(self) -> None:
        pass

    async def _make_server(self) -> Server:
        return self._web_server

    async def _cleanup_server(self) -> None:
        pass


class AppRunner(BaseRunner):
    """Web Application runner"""

    __slots__ = ("_app",)

    def __init__(
        self,
        app: Application,
        *,
        handle_signals: bool = False,
        access_log_class: type[AbstractAccessLogger] = AccessLogger,
        **kwargs: Any,
    ) -> None:
        super().__init__(handle_signals=handle_signals, **kwargs)
        if not isinstance(app, Application):
            raise TypeError(
                f"The first argument should be web.Application instance, got {app!r}"
            )
        self._kwargs["access_log_class"] = access_log_class
        self._app = app

    @property
    def app(self) -> Application:
        return self._app

    async def shutdown(self) -> None:
        await self._app.shutdown()

    async def _make_server(self) -> Server:
        loop = asyncio.get_event_loop()
        self._app._set_loop(loop)
        self._app.on_startup.freeze()
        await self._app.startup()
        self._app.freeze()

        return self._app._make_handler(loop=loop, **self._kwargs)

    async def _cleanup_server(self) -> None:
        await self._app.cleanup()
<|endoftext|>"""Low-level http related exceptions."""

from textwrap import indent

from .typedefs import _CIMultiDict

__all__ = ("HttpProcessingError",)


class HttpProcessingError(Exception):
    """HTTP error.

    Shortcut for raising HTTP errors with custom code, message and headers.

    code: HTTP Error code.
    message: (optional) Error message.
    headers: (optional) Headers to be sent in response, a list of pairs
    """

    code = 0
    message = ""
    headers = None

    def __init__(
        self,
        *,
        code: int | None = None,
        message: str = "",
        headers: _CIMultiDict | None = None,
    ) -> None:
        if code is not None:
            self.code = code
        self.headers = headers
        self.message = message

    def __str__(self) -> str:
        msg = indent(self.message, "  ")
        return f"{self.code}, message:\n{msg}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.code}, message={self.message!r}>"


class BadHttpMessage(HttpProcessingError):

    code = 400
    message = "Bad Request"

    def __init__(self, message: str, *, headers: _CIMultiDict | None = None) -> None:
        super().__init__(message=message, headers=headers)
        self.args = (message,)


class HttpBadRequest(BadHttpMessage):

    code = 400
    message = "Bad Request"


class PayloadEncodingError(BadHttpMessage):
    """Base class for payload errors"""


class ContentEncodingError(PayloadEncodingError):
    """Content encoding error."""


class TransferEncodingError(PayloadEncodingError):
    """transfer encoding error."""


class ContentLengthError(PayloadEncodingError):
    """Not enough data to satisfy content length header."""


class DecompressSizeError(PayloadEncodingError):
    """Deprecated. Removed in v4."""


class LineTooLong(BadHttpMessage):
    def __init__(
        self,
        line: str | bytes,
        limit: str | int = "Unknown",
        actual_size: str = "Unknown",
    ) -> None:
        super().__init__(f"Got more than {limit} bytes when reading: {line!r}.")
        self.args = (line, limit, actual_size)


class InvalidHeader(BadHttpMessage):
    def __init__(self, hdr: bytes | str) -> None:
        hdr_s = hdr.decode(errors="backslashreplace") if isinstance(hdr, bytes) else hdr
        super().__init__(f"Invalid HTTP header: {hdr!r}")
        self.hdr = hdr_s
        self.args = (hdr,)


class BadStatusLine(BadHttpMessage):
    def __init__(self, line: str = "", error: str | None = None) -> None:
        if not isinstance(line, str):
            line = repr(line)
        super().__init__(error or f"Bad status line {line!r}")
        self.args = (line,)
        self.line = line


class BadHttpMethod(BadStatusLine):
    """Invalid HTTP method in status line."""

    def __init__(self, line: str = "", error: str | None = None) -> None:
        if error is None and line.startswith("\x16\x03"):
            error = "Received HTTPS traffic on an HTTP port"
        super().__init__(line, error or f"Bad HTTP method in status line {line!r}")


class InvalidURLError(BadHttpMessage):
    pass
<|endoftext|>import asyncio
import enum
import io
import json
import mimetypes
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import chain
from typing import IO, TYPE_CHECKING, Any, Final, TextIO

from multidict import CIMultiDict

from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
    _SENTINEL,
    DEFAULT_CHUNK_SIZE,
    content_disposition_header,
    guess_filename,
    parse_mimetype,
    sentinel,
)
from .http_writer import _safe_header
from .streams import StreamReader
from .typedefs import JSONBytesEncoder, JSONEncoder, _CIMultiDict

__all__ = (
    "PAYLOAD_REGISTRY",
    "get_payload",
    "payload_type",
    "Payload",
    "BytesPayload",
    "StringPayload",
    "IOBasePayload",
    "BytesIOPayload",
    "BufferedReaderPayload",
    "TextIOPayload",
    "StringIOPayload",
    "JsonPayload",
    "JsonBytesPayload",
    "AsyncIterablePayload",
)

TOO_LARGE_BYTES_BODY: Final[int] = 2**20  # 1 MB
_CLOSE_FUTURES: set[asyncio.Future[None]] = set()


class LookupError(Exception):
    """Raised when no payload factory is found for the given data type."""


class Order(str, enum.Enum):
    normal = "normal"
    try_first = "try_first"
    try_last = "try_last"


def get_payload(data: Any, *args: Any, **kwargs: Any) -> "Payload":
    return PAYLOAD_REGISTRY.get(data, *args, **kwargs)


def register_payload(
    factory: type["Payload"], type: Any, *, order: Order = Order.normal
) -> None:
    PAYLOAD_REGISTRY.register(factory, type, order=order)


class payload_type:
    def __init__(self, type: Any, *, order: Order = Order.normal) -> None:
        self.type = type
        self.order = order

    def __call__(self, factory: type["Payload"]) -> type["Payload"]:
        register_payload(factory, self.type, order=self.order)
        return factory


PayloadType = type["Payload"]
_PayloadRegistryItem = tuple[PayloadType, Any]


class PayloadRegistry:
    """Payload registry.

    note: we need zope.interface for more efficient adapter search
    """

    __slots__ = ("_first", "_normal", "_last", "_normal_lookup")

    def __init__(self) -> None:
        self._first: list[_PayloadRegistryItem] = []
        self._normal: list[_PayloadRegistryItem] = []
        self._last: list[_PayloadRegistryItem] = []
        self._normal_lookup: dict[Any, PayloadType] = {}

    def get(
        self,
        data: Any,
        *args: Any,
        _CHAIN: "type[chain[_PayloadRegistryItem]]" = chain,
        **kwargs: Any,
    ) -> "Payload":
        if self._first:
            for factory, type_ in self._first:
                if isinstance(data, type_):
                    return factory(data, *args, **kwargs)
        # Try the fast lookup first
        if lookup_factory := self._normal_lookup.get(type(data)):
            return lookup_factory(data, *args, **kwargs)
        # Bail early if its already a Payload
        if isinstance(data, Payload):
            return data
        # Fallback to the slower linear search
        for factory, type_ in _CHAIN(self._normal, self._last):
            if isinstance(data, type_):
                return factory(data, *args, **kwargs)
        raise LookupError()

    def register(
        self, factory: PayloadType, type: Any, *, order: Order = Order.normal
    ) -> None:
        if order is Order.try_first:
            self._first.append((factory, type))
        elif order is Order.normal:
            self._normal.append((factory, type))
            if isinstance(type, Iterable):
                for t in type:
                    self._normal_lookup[t] = factory
            else:
                self._normal_lookup[type] = factory
        elif order is Order.try_last:
            self._last.append((factory, type))
        else:
            raise ValueError(f"Unsupported order {order!r}")


class Payload(ABC):

    _default_content_type: str = "application/octet-stream"
    _size: int | None = None
    _consumed: bool = False  # Default: payload has not been consumed yet
    _autoclose: bool = False  # Default: assume resource needs explicit closing

    def __init__(
        self,
        value: Any,
        headers: (
            _CIMultiDict | dict[str, str] | Iterable[tuple[str, str]] | None
        ) = None,
        content_type: str | None | _SENTINEL = sentinel,
        filename: str | None = None,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._encoding = encoding
        self._filename = filename
        self._headers: _CIMultiDict = CIMultiDict()
        self._value = value
        if content_type is not sentinel and content_type is not None:
            self._headers[hdrs.CONTENT_TYPE] = content_type
        elif self._filename is not None:
            if sys.version_info >= (3, 13):
                guesser = mimetypes.guess_file_type
            else:
                guesser = mimetypes.guess_type
            content_type = guesser(self._filename)[0]
            if content_type is None:
                content_type = self._default_content_type
            self._headers[hdrs.CONTENT_TYPE] = content_type
        else:
            self._headers[hdrs.CONTENT_TYPE] = self._default_content_type
        if headers:
            self._headers.update(headers)

    @property
    def size(self) -> int | None:
        """Size of the payload in bytes.

        Returns the number of bytes that will be transmitted when the payload
        is written. For string payloads, this is the size after encoding to bytes,
        not the length of the string.
        """
        return self._size

    @property
    def filename(self) -> str | None:
        """Filename of the payload."""
        return self._filename

    @property
    def headers(self) -> _CIMultiDict:
        """Custom item headers"""
        return self._headers

    @property
    def _binary_headers(self) -> bytes:
        return (
            "".join(
                _safe_header(k) + ": " + _safe_header(v) + "\r\n"
                for k, v in self.headers.items()
            ).encode("utf-8")
            + b"\r\n"
        )

    @property
    def encoding(self) -> str | None:
        """Payload encoding"""
        return self._encoding

    @property
    def content_type(self) -> str:
        """Content type"""
        return self._headers[hdrs.CONTENT_TYPE]

    @property
    def consumed(self) -> bool:
        """Whether the payload has been consumed and cannot be reused."""
        return self._consumed

    @property
    def autoclose(self) -> bool:
        """
        Whether the payload can close itself automatically.

        Returns True if the payload has no file handles or resources that need
        explicit closing. If False, callers must await close() to release resources.
        """
        return self._autoclose

    def set_content_disposition(
        self,
        disptype: str,
        quote_fields: bool = True,
        _charset: str = "utf-8",
        **params: Any,
    ) -> None:
        """Sets ``Content-Disposition`` header."""
        self._headers[hdrs.CONTENT_DISPOSITION] = content_disposition_header(
            disptype, quote_fields=quote_fields, _charset=_charset, **params
        )

    @abstractmethod
    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """
        Return string representation of the value.

        This is named decode() to allow compatibility with bytes objects.
        """

    @abstractmethod
    async def write(self, writer: AbstractStreamWriter) -> None:
        """
        Write payload to the writer stream.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing

        This is a legacy method that writes the entire payload without length constraints.

        Important:
            For new implementations, use write_with_length() instead of this method.
            This method is maintained for backwards compatibility and will eventually
            delegate to write_with_length(writer, None) in all implementations.

        All payload subclasses must override this method for backwards compatibility,
        but new code should use write_with_length for more flexibility and control.

        """

    # write_with_length is new in aiohttp 3.12
    # it should be overridden by subclasses
    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        """
        Write payload with a specific content length constraint.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing
            content_length: Maximum number of bytes to write (None for unlimited)

        This method allows writing payload content with a specific length constraint,
        which is particularly useful for HTTP responses with Content-Length header.

        Note:
            This is the base implementation that provides backwards compatibility
            for subclasses that don't override this method. Specific payload types
            should override this method to implement proper length-constrained writing.

        """
        # Backwards compatibility for subclasses that don't override this method
        # and for the default implementation
        await self.write(writer)

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This is a convenience method that calls decode() and encodes the result
        to bytes using the specified encoding.
        """
        # Use instance encoding if available, otherwise use parameter
        actual_encoding = self._encoding or encoding
        return self.decode(actual_encoding, errors).encode(actual_encoding)

    def _close(self) -> None:
        """
        Async safe synchronous close operations for backwards compatibility.

        This method exists only for backwards compatibility with code that
        needs to clean up payloads synchronously. In the future, we will
        drop this method and only support the async close() method.

        WARNING: This method must be safe to call from within the event loop
        without blocking. Subclasses should not perform any blocking I/O here.

        WARNING: This method must be called from within an event loop for
        certain payload types (e.g., IOBasePayload). Calling it outside an
        event loop may raise RuntimeError.
        """
        # This is a no-op by default, but subclasses can override it
        # for non-blocking cleanup operations.

    async def close(self) -> None:
        """
        Close the payload if it holds any resources.

        IMPORTANT: This method must not await anything that might not finish
        immediately, as it may be called during cleanup/cancellation. Schedule
        any long-running operations without awaiting them.

        In the future, this will be the only close method supported.
        """
        self._close()


class BytesPayload(Payload):
    _value: bytes
    # _consumed = False (inherited) - Bytes are immutable and can be reused
    _autoclose = True  # No file handle, just bytes in memory

    def __init__(
        self, value: bytes | bytearray | memoryview, *args: Any, **kwargs: Any
    ) -> None:
        if "content_type" not in kwargs:
            kwargs["content_type"] = "application/octet-stream"

        super().__init__(value, *args, **kwargs)

        if isinstance(value, memoryview):
            self._size = value.nbytes
        elif isinstance(value, (bytes, bytearray)):
            self._size = len(value)
        else:
            raise TypeError(f"value argument must be byte-ish, not {type(value)!r}")

        if self._size > TOO_LARGE_BYTES_BODY:
            kwargs = {"source": self}
            warnings.warn(
                "Sending a large body directly with raw bytes might"
                " lock the event loop. You should probably pass an "
                "io.BytesIO object instead",
                ResourceWarning,
                **kwargs,
            )

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self._value.decode(encoding, errors)

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This method returns the raw bytes content of the payload.
        It is equivalent to accessing the _value attribute directly.
        """
        return self._value

    async def write(self, writer: AbstractStreamWriter) -> None:
        """
        Write the entire bytes payload to the writer stream.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing

        This method writes the entire bytes content without any length constraint.

        Note:
            For new implementations that need length control, use write_with_length().
            This method is maintained for backwards compatibility and is equivalent
            to write_with_length(writer, None).

        """
        await writer.write(self._value)

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        """
        Write bytes payload with a specific content length constraint.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing
            content_length: Maximum number of bytes to write (None for unlimited)

        This method writes either the entire byte sequence or a slice of it
        up to the specified content_length. For BytesPayload, this operation
        is performed efficiently using array slicing.

        """
        if content_length is not None:
            await writer.write(self._value[:content_length])
        else:
            await writer.write(self._value)


class StringPayload(BytesPayload):
    def __init__(
        self,
        value: str,
        *args: Any,
        encoding: str | None = None,
        content_type: str | None = None,
        **kwargs: Any,
    ) -> None:

        if encoding is None:
            if content_type is None:
                real_encoding = "utf-8"
                content_type = "text/plain; charset=utf-8"
            else:
                mimetype = parse_mimetype(content_type)
                real_encoding = mimetype.parameters.get("charset", "utf-8")
        else:
            if content_type is None:
                content_type = "text/plain; charset=%s" % encoding
            real_encoding = encoding

        super().__init__(
            value.encode(real_encoding),
            encoding=real_encoding,
            content_type=content_type,
            *args,
            **kwargs,
        )


class StringIOPayload(StringPayload):
    def __init__(self, value: IO[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(value.read(), *args, **kwargs)


class IOBasePayload(Payload):
    _value: io.IOBase
    # _consumed = False (inherited) - File can be re-read from the same position
    _start_position: int | None = None
    # _autoclose = False (inherited) - Has file handle that needs explicit closing

    def __init__(
        self, value: IO[Any], disposition: str = "attachment", *args: Any, **kwargs: Any
    ) -> None:
        if "filename" not in kwargs:
            kwargs["filename"] = guess_filename(value)

        super().__init__(value, *args, **kwargs)

        if self._filename is not None and disposition is not None:
            if hdrs.CONTENT_DISPOSITION not in self.headers:
                self.set_content_disposition(disposition, filename=self._filename)

    def _set_or_restore_start_position(self) -> None:
        """Set or restore the start position of the file-like object."""
        if self._start_position is None:
            try:
                self._start_position = self._value.tell()
            except (OSError, AttributeError):
                self._consumed = True  # Cannot seek, mark as consumed
            return
        try:
            self._value.seek(self._start_position)
        except (OSError, AttributeError):
            # Failed to seek back - mark as consumed since we've already read
            self._consumed = True

    def _read_and_available_len(
        self, remaining_content_len: int | None
    ) -> tuple[int | None, bytes]:
        """
        Read the file-like object and return both its total size and the first chunk.

        Args:
            remaining_content_len: Optional limit on how many bytes to read in this operation.
                If None, DEFAULT_CHUNK_SIZE will be used as the default chunk size.

        Returns:
            A tuple containing:
            - The total size of the remaining unread content (None if size cannot be determined)
            - The first chunk of bytes read from the file object

        This method is optimized to perform both size calculation and initial read
        in a single operation, which is executed in a single executor job to minimize
        context switches and file operations when streaming content.

        """
        self._set_or_restore_start_position()
        size = self.size  # Call size only once since it does I/O
        return size, self._value.read(
            min(
                DEFAULT_CHUNK_SIZE,
                size or DEFAULT_CHUNK_SIZE,
                remaining_content_len or DEFAULT_CHUNK_SIZE,
            )
        )

    def _read(self, remaining_content_len: int | None) -> bytes:
        """
        Read a chunk of data from the file-like object.

        Args:
            remaining_content_len: Optional maximum number of bytes to read.
                If None, DEFAULT_CHUNK_SIZE will be used as the default chunk size.

        Returns:
            A chunk of bytes read from the file object, respecting the
            remaining_content_len limit if specified.

        This method is used for subsequent reads during streaming after
        the initial _read_and_available_len call has been made.

        """
        return self._value.read(remaining_content_len or DEFAULT_CHUNK_SIZE)  # type: ignore[no-any-return]

    @property
    def size(self) -> int | None:
        """
        Size of the payload in bytes.

        Returns the total size of the payload content from the initial position.
        This ensures consistent Content-Length for requests, including 307/308 redirects
        where the same payload instance is reused.

        Returns None if the size cannot be determined (e.g., for unseekable streams).
        """
        try:
            # Store the start position on first access.
            # This is critical when the same payload instance is reused (e.g., 307/308
            # redirects). Without storing the initial position, after the payload is
            # read once, the file position would be at EOF, which would cause the
            # size calculation to return 0 (file_size - EOF position).
            # By storing the start position, we ensure the size calculation always
            # returns the correct total size for any subsequent use.
            if self._start_position is None:
                self._start_position = self._value.tell()

            # Return the total size from the start position
            # This ensures Content-Length is correct even after reading
            return os.fstat(self._value.fileno()).st_size - self._start_position
        except (AttributeError, OSError):
            return None

    async def write(self, writer: AbstractStreamWriter) -> None:
        """
        Write the entire file-like payload to the writer stream.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing

        This method writes the entire file content without any length constraint.
        It delegates to write_with_length() with no length limit for implementation
        consistency.

        Note:
            For new implementations that need length control, use write_with_length() directly.
            This method is maintained for backwards compatibility with existing code.

        """
        await self.write_with_length(writer, None)

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        """
        Write file-like payload with a specific content length constraint.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing
            content_length: Maximum number of bytes to write (None for unlimited)

        This method implements optimized streaming of file content with length constraints:

        1. File reading is performed in a thread pool to avoid blocking the event loop
        2. Content is read and written in chunks to maintain memory efficiency
        3. Writing stops when either:
           - All available file content has been written (when size is known)
           - The specified content_length has been reached
        4. File resources are properly closed even if the operation is cancelled

        The implementation carefully handles both known-size and unknown-size payloads,
        as well as constrained and unconstrained content lengths.

        """
        loop = asyncio.get_running_loop()
        total_written_len = 0
        remaining_content_len = content_length

        # Get initial data and available length
        available_len, chunk = await loop.run_in_executor(
            None, self._read_and_available_len, remaining_content_len
        )
        # Process data chunks until done
        while chunk:
            chunk_len = len(chunk)

            # Write data with or without length constraint
            if remaining_content_len is None:
                await writer.write(chunk)
            else:
                await writer.write(chunk[:remaining_content_len])
                remaining_content_len -= chunk_len

            total_written_len += chunk_len

            # Check if we're done writing
            if self._should_stop_writing(
                available_len, total_written_len, remaining_content_len
            ):
                return

            # Read next chunk
            chunk = await loop.run_in_executor(
                None,
                self._read,
                (
                    min(DEFAULT_CHUNK_SIZE, remaining_content_len)
                    if remaining_content_len is not None
                    else DEFAULT_CHUNK_SIZE
                ),
            )

    def _should_stop_writing(
        self,
        available_len: int | None,
        total_written_len: int,
        remaining_content_len: int | None,
    ) -> bool:
        """
        Determine if we should stop writing data.

        Args:
            available_len: Known size of the payload if available (None if unknown)
            total_written_len: Number of bytes already written
            remaining_content_len: Remaining bytes to be written for content-length limited responses

        Returns:
            True if we should stop writing data, based on either:
            - Having written all available data (when size is known)
            - Having written all requested content (when content-length is specified)

        """
        return (available_len is not None and total_written_len >= available_len) or (
            remaining_content_len is not None and remaining_content_len <= 0
        )

    def _close(self) -> None:
        """
        Async safe synchronous close operations for backwards compatibility.

        This method exists only for backwards
        compatibility. Use the async close() method instead.

        WARNING: This method MUST be called from within an event loop.
        Calling it outside an event loop will raise RuntimeError.
        """
        # Skip if already consumed
        if self._consumed:
            return
        self._consumed = True  # Mark as consumed to prevent further writes
        # Schedule file closing without awaiting to prevent cancellation issues
        loop = asyncio.get_running_loop()
        close_future = loop.run_in_executor(None, self._value.close)
        # Hold a strong reference to the future to prevent it from being
        # garbage collected before it completes.
        _CLOSE_FUTURES.add(close_future)
        close_future.add_done_callback(_CLOSE_FUTURES.remove)

    async def close(self) -> None:
        """
        Close the payload if it holds any resources.

        IMPORTANT: This method must not await anything that might not finish
        immediately, as it may be called during cleanup/cancellation. Schedule
        any long-running operations without awaiting them.
        """
        self._close()

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """
        Return string representation of the value.

        WARNING: This method does blocking I/O and should not be called in the event loop.
        """
        return self._read_all().decode(encoding, errors)

    def _read_all(self) -> bytes:
        """Read the entire file-like object and return its content as bytes."""
        self._set_or_restore_start_position()
        # Use readlines() to ensure we get all content
        return b"".join(self._value.readlines())

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This method reads the entire file content and returns it as bytes.
        It is equivalent to reading the file-like object directly.
        The file reading is performed in an executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_all)


class TextIOPayload(IOBasePayload):
    _value: io.TextIOBase
    # _autoclose = False (inherited) - Has text file handle that needs explicit closing

    def __init__(
        self,
        value: TextIO,
        *args: Any,
        encoding: str | None = None,
        content_type: str | None = None,
        **kwargs: Any,
    ) -> None:

        if encoding is None:
            if content_type is None:
                encoding = "utf-8"
                content_type = "text/plain; charset=utf-8"
            else:
                mimetype = parse_mimetype(content_type)
                encoding = mimetype.parameters.get("charset", "utf-8")
        else:
            if content_type is None:
                content_type = "text/plain; charset=%s" % encoding

        super().__init__(
            value,
            content_type=content_type,
            encoding=encoding,
            *args,
            **kwargs,
        )

    def _read_and_available_len(
        self, remaining_content_len: int | None
    ) -> tuple[int | None, bytes]:
        """
        Read the text file-like object and return both its total size and the first chunk.

        Args:
            remaining_content_len: Optional limit on how many bytes to read in this operation.
                If None, DEFAULT_CHUNK_SIZE will be used as the default chunk size.

        Returns:
            A tuple containing:
            - The total size of the remaining unread content (None if size cannot be determined)
            - The first chunk of bytes read from the file object, encoded using the payload's encoding

        This method is optimized to perform both size calculation and initial read
        in a single operation, which is executed in a single executor job to minimize
        context switches and file operations when streaming content.

        Note:
            TextIOPayload handles encoding of the text content before writing it
            to the stream. If no encoding is specified, UTF-8 is used as the default.

        """
        self._set_or_restore_start_position()
        size = self.size
        chunk = self._value.read(
            min(
                DEFAULT_CHUNK_SIZE,
                size or DEFAULT_CHUNK_SIZE,
                remaining_content_len or DEFAULT_CHUNK_SIZE,
            )
        )
        return size, chunk.encode(self._encoding) if self._encoding else chunk.encode()

    def _read(self, remaining_content_len: int | None) -> bytes:
        """
        Read a chunk of data from the text file-like object.

        Args:
            remaining_content_len: Optional maximum number of bytes to read.
                If None, DEFAULT_CHUNK_SIZE will be used as the default chunk size.

        Returns:
            A chunk of bytes read from the file object and encoded using the payload's
            encoding. The data is automatically converted from text to bytes.

        This method is used for subsequent reads during streaming after
        the initial _read_and_available_len call has been made. It properly
        handles text encoding, converting the text content to bytes using
        the specified encoding (or UTF-8 if none was provided).

        """
        chunk = self._value.read(remaining_content_len or DEFAULT_CHUNK_SIZE)
        return chunk.encode(self._encoding) if self._encoding else chunk.encode()

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """
        Return string representation of the value.

        WARNING: This method does blocking I/O and should not be called in the event loop.
        """
        self._set_or_restore_start_position()
        return self._value.read()

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This method reads the entire text file content and returns it as bytes.
        It encodes the text content using the specified encoding.
        The file reading is performed in an executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()

        # Use instance encoding if available, otherwise use parameter
        actual_encoding = self._encoding or encoding

        def _read_and_encode() -> bytes:
            self._set_or_restore_start_position()
            # TextIO read() always returns the full content
            return self._value.read().encode(actual_encoding, errors)

        return await loop.run_in_executor(None, _read_and_encode)


class BytesIOPayload(IOBasePayload):
    _value: io.BytesIO
    _size: int  # Always initialized in __init__
    _autoclose = True  # BytesIO is in-memory, safe to auto-close

    def __init__(self, value: io.BytesIO, *args: Any, **kwargs: Any) -> None:
        super().__init__(value, *args, **kwargs)
        with self._value.getbuffer() as buf:
            self._size = len(buf) - self._value.tell()

    @property
    def size(self) -> int:
        """Size of the payload in bytes.

        Returns the number of bytes in the BytesIO buffer that will be transmitted.
        This is calculated once during initialization for efficiency.
        """
        return self._size

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        self._set_or_restore_start_position()
        return self._value.read().decode(encoding, errors)

    async def write(self, writer: AbstractStreamWriter) -> None:
        return await self.write_with_length(writer, None)

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        """
        Write BytesIO payload with a specific content length constraint.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing
            content_length: Maximum number of bytes to write (None for unlimited)

        This implementation is specifically optimized for BytesIO objects:

        1. Reads content in chunks to maintain memory efficiency
        2. Yields control back to the event loop periodically to prevent blocking
           when dealing with large BytesIO objects
        3. Respects content_length constraints when specified
        4. Properly cleans up by closing the BytesIO object when done or on error

        The periodic yielding to the event loop is important for maintaining
        responsiveness when processing large in-memory buffers.

        """
        self._set_or_restore_start_position()
        loop_count = 0
        remaining_bytes = content_length
        while chunk := self._value.read(DEFAULT_CHUNK_SIZE):
            if loop_count > 0:
                # Avoid blocking the event loop
                # if they pass a large BytesIO object
                # and we are not in the first iteration
                # of the loop
                await asyncio.sleep(0)
            if remaining_bytes is None:
                await writer.write(chunk)
            else:
                await writer.write(chunk[:remaining_bytes])
                remaining_bytes -= len(chunk)
                if remaining_bytes <= 0:
                    return
            loop_count += 1

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This method reads the entire BytesIO content and returns it as bytes.
        It is equivalent to accessing the _value attribute directly.
        """
        self._set_or_restore_start_position()
        return self._value.read()

    async def close(self) -> None:
        """
        Close the BytesIO payload.

        This does nothing since BytesIO is in-memory and does not require explicit closing.
        """


class BufferedReaderPayload(IOBasePayload):
    _value: io.BufferedIOBase
    # _autoclose = False (inherited) - Has buffered file handle that needs explicit closing

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        self._set_or_restore_start_position()
        return self._value.read().decode(encoding, errors)


class JsonPayload(BytesPayload):
    def __init__(
        self,
        value: Any,
        encoding: str = "utf-8",
        content_type: str = "application/json",
        dumps: JSONEncoder = json.dumps,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            dumps(value).encode(encoding),
            content_type=content_type,
            encoding=encoding,
            *args,
            **kwargs,
        )


class JsonBytesPayload(BytesPayload):
    """JSON payload for encoders that return bytes directly.

    Use this when your JSON encoder (like orjson) returns bytes
    instead of str, avoiding the encode/decode overhead.
    """

    def __init__(
        self,
        value: Any,
        dumps: JSONBytesEncoder,
        content_type: str = "application/json",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dumps(value),
            content_type=content_type,
            *args,
            **kwargs,
        )


if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator

    _AsyncIterator = AsyncIterator[bytes]
    _AsyncIterable = AsyncIterable[bytes]
else:
    from collections.abc import AsyncIterable, AsyncIterator

    _AsyncIterator = AsyncIterator
    _AsyncIterable = AsyncIterable


class AsyncIterablePayload(Payload):

    _iter: _AsyncIterator | None = None
    _value: _AsyncIterable
    _cached_chunks: list[bytes] | None = None
    # _consumed stays False to allow reuse with cached content
    _autoclose = True  # Iterator doesn't need explicit closing

    def __init__(self, value: _AsyncIterable, *args: Any, **kwargs: Any) -> None:
        if not isinstance(value, AsyncIterable):
            raise TypeError(
                "value argument must support "
                "collections.abc.AsyncIterable interface, "
                f"got {type(value)!r}"
            )

        if "content_type" not in kwargs:
            kwargs["content_type"] = "application/octet-stream"

        super().__init__(value, *args, **kwargs)

        self._iter = value.__aiter__()

    async def write(self, writer: AbstractStreamWriter) -> None:
        """
        Write the entire async iterable payload to the writer stream.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing

        This method iterates through the async iterable and writes each chunk
        to the writer without any length constraint.

        Note:
            For new implementations that need length control, use write_with_length() directly.
            This method is maintained for backwards compatibility with existing code.

        """
        await self.write_with_length(writer, None)

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        """
        Write async iterable payload with a specific content length constraint.

        Args:
            writer: An AbstractStreamWriter instance that handles the actual writing
            content_length: Maximum number of bytes to write (None for unlimited)

        This implementation handles streaming of async iterable content with length constraints:

        1. If cached chunks are available, writes from them
        2. Otherwise iterates through the async iterable one chunk at a time
        3. Respects content_length constraints when specified
        4. Does NOT generate cache - that's done by as_bytes()

        """
        # If we have cached chunks, use them
        if self._cached_chunks is not None:
            remaining_bytes = content_length
            for chunk in self._cached_chunks:
                if remaining_bytes is None:
                    await writer.write(chunk)
                elif remaining_bytes > 0:
                    await writer.write(chunk[:remaining_bytes])
                    remaining_bytes -= len(chunk)
                else:
                    break
            return

        # If iterator is exhausted and we don't have cached chunks, nothing to write
        if self._iter is None:
            return

        # Stream from the iterator
        remaining_bytes = content_length
        # Nothing is cached, so advancing the iterator is irreversible: mark the
        # payload consumed up front so an interrupted write cannot be replayed
        # from a partially drained iterator.
        self._consumed = True

        try:
            while True:
                chunk = await anext(self._iter)
                if remaining_bytes is None:
                    await writer.write(chunk)
                # If we have a content length limit
                elif remaining_bytes > 0:
                    await writer.write(chunk[:remaining_bytes])
                    remaining_bytes -= len(chunk)
                # We still want to exhaust the iterator even
                # if we have reached the content length limit
                # since the file handle may not get closed by
                # the iterator if we don't do this
        except StopAsyncIteration:
            # Iterator is exhausted
            self._iter = None

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Decode the payload content as a string if cached chunks are available."""
        if self._cached_chunks is not None:
            return b"".join(self._cached_chunks).decode(encoding, errors)
        raise TypeError("Unable to decode - content not cached. Call as_bytes() first.")

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """
        Return bytes representation of the value.

        This method reads the entire async iterable content and returns it as bytes.
        It generates and caches the chunks for future reuse.
        """
        # If we have cached chunks, return them joined
        if self._cached_chunks is not None:
            return b"".join(self._cached_chunks)

        # If iterator is exhausted and no cache, return empty
        if self._iter is None:
            return b""

        # Read all chunks and cache them
        chunks: list[bytes] = []
        async for chunk in self._iter:
            chunks.append(chunk)

        # Iterator is exhausted, cache the chunks
        self._iter = None
        self._cached_chunks = chunks
        # Keep _consumed as False to allow reuse with cached chunks

        return b"".join(chunks)


class StreamReaderPayload(AsyncIterablePayload):
    def __init__(self, value: StreamReader, *args: Any, **kwargs: Any) -> None:
        super().__init__(value.iter_any(), *args, **kwargs)


PAYLOAD_REGISTRY = PayloadRegistry()
PAYLOAD_REGISTRY.register(BytesPayload, (bytes, bytearray, memoryview))
PAYLOAD_REGISTRY.register(StringPayload, str)
PAYLOAD_REGISTRY.register(StringIOPayload, io.StringIO)
PAYLOAD_REGISTRY.register(TextIOPayload, io.TextIOBase)
PAYLOAD_REGISTRY.register(BytesIOPayload, io.BytesIO)
PAYLOAD_REGISTRY.register(BufferedReaderPayload, (io.BufferedReader, io.BufferedRandom))
PAYLOAD_REGISTRY.register(IOBasePayload, io.IOBase)
PAYLOAD_REGISTRY.register(StreamReaderPayload, StreamReader)
# try_last for giving a chance to more specialized async interables like
# multipart.BodyPartReaderPayload override the default
PAYLOAD_REGISTRY.register(AsyncIterablePayload, AsyncIterable, order=Order.try_last)
<|endoftext|>import base64
import binascii
import builtins
import json
import re
import sys
import uuid
import warnings
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast
from urllib.parse import parse_qsl, unquote, urlencode

from multidict import CIMultiDict, CIMultiDictProxy

from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
    CONTENT_DISPOSITION,
    CONTENT_ENCODING,
    CONTENT_LENGTH,
    CONTENT_TRANSFER_ENCODING,
    CONTENT_TYPE,
)
from .helpers import CHAR, DEFAULT_CHUNK_SIZE, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .http_exceptions import BadHttpMessage
from .log import internal_logger
from .payload import (
    JsonPayload,
    LookupError,
    Order,
    Payload,
    StringPayload,
    get_payload,
    payload_type,
)
from .streams import StreamReader

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self", bound="BodyPartReader")

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    Buffer = Union[bytes, bytearray, "memoryview[int]", "memoryview[bytes]"]

_Buffer = TypeVar("_Buffer", bound=Buffer)

__all__ = (
    "MultipartReader",
    "MultipartWriter",
    "BodyPartReader",
    "BadContentDispositionHeader",
    "BadContentDispositionParam",
    "parse_content_disposition",
    "content_disposition_filename",
)


# The base64 alphabet plus the padding character.
_BASE64_CHARS = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)
_NON_BASE64_BYTES = bytes(b for b in range(256) if b not in _BASE64_CHARS)


if TYPE_CHECKING:
    from .client_reqrep import ClientResponse


class BadContentDispositionHeader(RuntimeWarning):
    pass


class BadContentDispositionParam(RuntimeWarning):
    pass


def parse_content_disposition(
    header: str | None,
) -> tuple[str | None, dict[str, str]]:
    def is_token(string: str) -> bool:
        return bool(string) and TOKEN >= set(string)

    def is_quoted(string: str) -> bool:
        return len(string) >= 2 and string[0] == string[-1] == '"'

    def is_rfc5987(string: str) -> bool:
        return is_token(string) and string.count("'") == 2

    def is_extended_param(string: str) -> bool:
        return string.endswith("*")

    def is_continuous_param(string: str) -> bool:
        pos = string.find("*") + 1
        if not pos:
            return False
        substring = string[pos:-1] if string.endswith("*") else string[pos:]
        return substring.isdigit()

    def unescape(text: str, *, chars: str = "".join(map(re.escape, CHAR))) -> str:
        return re.sub(f"\\\\([{chars}])", "\\1", text)

    if not header:
        return None, {}

    # https://www.rfc-editor.org/info/rfc9110/#section-5.6.6-2
    disptype, *parts = header.split(";")
    disptype = disptype.strip()
    if not is_token(disptype):
        warnings.warn(BadContentDispositionHeader(header))
        return None, {}

    params: dict[str, str] = {}
    while parts:
        item = parts.pop(0)

        if not item:  # To handle trailing semicolons
            warnings.warn(BadContentDispositionHeader(header))
            continue

        if "=" not in item:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        key, value = item.split("=", 1)
        key = key.lower().strip()
        value = value.lstrip()

        if key in params:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        if not is_token(key):
            warnings.warn(BadContentDispositionParam(item))
            continue

        elif is_continuous_param(key):
            if is_quoted(value):
                value = unescape(value[1:-1])
            elif not is_token(value):
                warnings.warn(BadContentDispositionParam(item))
                continue

        elif is_extended_param(key):
            if is_rfc5987(value):
                encoding, _, value = value.split("'", 2)
                encoding = encoding or "utf-8"
            else:
                warnings.warn(BadContentDispositionParam(item))
                continue

            try:
                value = unquote(value, encoding, "strict")
            except (builtins.LookupError, UnicodeDecodeError):
                # The charset is attacker-controlled here; an unknown name
                # raises the builtin LookupError (the bare name is shadowed in
                # this module by payload.LookupError).
                warnings.warn(BadContentDispositionParam(item))
                continue

        else:
            failed = True
            rstripped = value.rstrip()
            if is_quoted(rstripped):
                failed = False
                value = unescape(rstripped[1:-1].lstrip("\\/"))
            elif is_token(value):
                failed = False
            elif parts:
                # maybe just ; in filename, in any case this is just
                # one case fix, for proper fix we need to redesign parser
                _value = f"{value};{parts[0]}"
                if is_quoted(_value):
                    parts.pop(0)
                    value = unescape(_value[1:-1].lstrip("\\/"))
                    failed = False

            if failed:
                warnings.warn(BadContentDispositionHeader(header))
                return None, {}

        params[key] = value

    return disptype.lower(), params


def content_disposition_filename(
    params: Mapping[str, str], name: str = "filename"
) -> str | None:
    name_suf = "%s*" % name
    if not params:
        return None
    elif name_suf in params:
        return params[name_suf]
    elif name in params:
        return params[name]
    else:
        parts = []
        fnparams = sorted(
            (key, value) for key, value in params.items() if key.startswith(name_suf)
        )
        for num, (key, value) in enumerate(fnparams):
            _, tail = key.split("*", 1)
            if tail.endswith("*"):
                tail = tail[:-1]
            if tail == str(num):
                parts.append(value)
            else:
                break
        if not parts:
            return None
        value = "".join(parts)
        if "'" in value:
            encoding, _, value = value.split("'", 2)
            encoding = encoding or "utf-8"
            try:
                return unquote(value, encoding, "strict")
            except (builtins.LookupError, UnicodeDecodeError):
                # Both the charset name and the octets are attacker-controlled
                # here; an unknown encoding raises the builtin LookupError
                # (shadowed in this module by payload.LookupError) and
                # undecodable bytes raise UnicodeDecodeError.
                return None
        return value


class MultipartResponseWrapper:
    """Wrapper around the MultipartReader.

    It takes care about
    underlying connection and close it when it needs in.
    """

    def __init__(
        self,
        resp: "ClientResponse",
        stream: "MultipartReader",
    ) -> None:
        self.resp = resp
        self.stream = stream

    def __aiter__(self) -> "MultipartResponseWrapper":
        return self

    async def __anext__(
        self,
    ) -> Union["MultipartReader", "BodyPartReader"]:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    def at_eof(self) -> bool:
        """Returns True when all response data had been read."""
        return self.resp.content.at_eof()

    async def next(
        self,
    ) -> Union["MultipartReader", "BodyPartReader"] | None:
        """Emits next multipart reader object."""
        item = await self.stream.next()
        if self.stream.at_eof():
            await self.release()
        return item

    async def release(self) -> None:
        """Release the connection gracefully.

        All remaining content is read to the void.
        """
        await self.resp.release()


class BodyPartReader:
    """Multipart reader for single body part."""

    chunk_size = 8192

    def __init__(
        self,
        boundary: bytes,
        headers: "CIMultiDictProxy[str]",
        content: StreamReader,
        *,
        subtype: str = "mixed",
        default_charset: str | None = None,
        max_decompress_size: int = DEFAULT_CHUNK_SIZE,
        client_max_size: int = sys.maxsize,
        max_size_error_cls: type[Exception] = ValueError,
    ) -> None:
        self.headers = headers
        self._boundary = boundary
        self._boundary_len = len(boundary) + 2  # Boundary + \r\n
        self._content = content
        self._default_charset = default_charset
        self._at_eof = False
        self._is_form_data = subtype == "form-data"
        # https://datatracker.ietf.org/doc/html/rfc7578#section-4.8
        length = None if self._is_form_data else self.headers.get(CONTENT_LENGTH, None)
        if length is not None and not (length.isascii() and length.isdigit()):
            # Reject sign prefixes, underscores, whitespace and non-ASCII
            # digits that int() would otherwise accept.
            # https://www.rfc-editor.org/rfc/rfc9110#section-8.6
            raise ValueError(f"invalid Content-Length: {length!r}")
        self._length = int(length) if length is not None else None
        self._read_bytes = 0
        self._b64_carry = b""
        self._unread: deque[bytes] = deque()
        self._prev_chunk: bytes | None = None
        self._content_eof = 0
        self._cache: dict[str, Any] = {}
        self._max_decompress_size = max_decompress_size
        self._client_max_size = client_max_size
        self._max_size_error_cls = max_size_error_cls

    def __aiter__(self: Self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    async def next(self) -> bytes | None:
        item = await self.read()
        if not item:
            return None
        return item

    async def read(self, *, decode: bool = False) -> bytes:
        """Reads body part data.

        decode: Decodes data following by encoding
                method from Content-Encoding header. If it missed
                data remains untouched
        """
        if self._at_eof:
            return b""
        data = bytearray()
        while not self._at_eof:
            data.extend(await self.read_chunk(self.chunk_size))
            if len(data) > self._client_max_size:
                raise self._max_size_error_cls(self._client_max_size)
        if decode:
            decoded_data = bytearray()
            async for d in self.decode_iter(data):
                decoded_data.extend(d)
                if len(decoded_data) > self._client_max_size:
                    raise self._max_size_error_cls(self._client_max_size)
            return decoded_data
        return data

    async def read_chunk(self, size: int = chunk_size) -> bytes:
        """Reads body part content chunk of the specified size.

        size: chunk size
        """
        if self._at_eof:
            return b""
        carry = self._b64_carry
        want = size - len(carry)
        if carry:
            self._b64_carry = b""
            want = max(want, self._boundary_len)
        if self._length is not None:
            fresh = await self._read_chunk_from_length(want)
        else:
            fresh = await self._read_chunk_from_stream(want)
        chunk = carry + fresh
        self._read_bytes += len(fresh)

        # base64 decodes in quartets and every chunk is decoded on its own, so
        # a chunk should not end mid-quartet.
        encoding = self.headers.get(CONTENT_TRANSFER_ENCODING)
        if encoding and encoding.lower() == "base64":
            chunk = self._align_base64_chunk(chunk, len(carry) + want)

        if self._read_bytes == self._length:
            self._at_eof = True
        if self._at_eof and await self._content.readline() != b"\r\n":
            raise ValueError("Reader did not read all the data or it is malformed")
        return chunk

    def _align_base64_chunk(self, chunk: bytes, size: int) -> bytes:
        at_end = self._at_eof or (
            self._length is not None and self._read_bytes >= self._length
        )
        if not at_end and len(chunk) > size:
            self._b64_carry = chunk[size:]
            chunk = chunk[:size]

        remainder = len(chunk.translate(None, _NON_BASE64_BYTES)) % 4
        if not remainder or at_end:
            return chunk

        # Walk back over the trailing partial quartet and carry it into the
        # next chunk.
        cut = len(chunk)
        left = remainder
        while left:
            cut -= 1
            if chunk[cut] in _BASE64_CHARS:
                left -= 1
        if not cut:
            # No whole quartet to hand back, and carrying the lot would make
            # no progress: the caller asked for this many bytes, and a part
            # that holds no quartet within them holds none to give.
            return chunk

        self._b64_carry = chunk[cut:] + self._b64_carry
        return chunk[:cut]

    async def _read_chunk_from_length(self, size: int) -> bytes:
        # Reads body part content chunk of the specified size.
        # The body part must has Content-Length header with proper value.
        assert self._length is not None, "Content-Length required for chunked read"
        chunk_size = min(size, self._length - self._read_bytes)
        chunk = await self._content.read(chunk_size)
        if self._content.at_eof():
            self._at_eof = True
        return chunk

    async def _read_chunk_from_stream(self, size: int) -> bytes:
        # Reads content chunk of body part with unknown length.
        # The Content-Length header for body part is not necessary.
        assert (
            size >= self._boundary_len
        ), "Chunk size must be greater or equal than boundary length + 2"
        first_chunk = self._prev_chunk is None
        if first_chunk:
            # We need to re-add the CRLF that got removed from headers parsing.
            self._prev_chunk = b"\r\n" + await self._content.read(size)

        chunk = b""
        # content.read() may return less than size, so we need to loop to ensure
        # we have enough data to detect the boundary.
        while len(chunk) < self._boundary_len:
            chunk += await self._content.read(size)
            self._content_eof += int(self._content.at_eof())
            if self._content_eof > 2:
                raise ValueError("Reading after EOF")
            if self._content_eof:
                break
        if len(chunk) > size:
            self._content.unread_data(chunk[size:])
            chunk = chunk[:size]

        assert self._prev_chunk is not None
        window = self._prev_chunk + chunk
        sub = b"\r\n" + self._boundary
        if first_chunk:
            idx = window.find(sub)
        else:
            idx = window.find(sub, max(0, len(self._prev_chunk) - len(sub)))
        if idx >= 0:
            # pushing boundary back to content
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self._content.unread_data(window[idx:])
            self._prev_chunk = self._prev_chunk[:idx]
            chunk = window[len(self._prev_chunk) : idx]
            if not chunk:
                self._at_eof = True
        result = self._prev_chunk[2 if first_chunk else 0 :]  # Strip initial CRLF
        self._prev_chunk = chunk
        return result

    async def readline(self) -> bytes:
        """Reads body part by line by line."""
        if self._at_eof:
            return b""

        if self._unread:
            line = self._unread.popleft()
        else:
            line = await self._content.readline()

        if line.startswith(self._boundary):
            # the very last boundary may not come with \r\n,
            # so set single rules for everyone
            sline = line.rstrip(b"\r\n")
            boundary = self._boundary
            last_boundary = self._boundary + b"--"
            # ensure that we read exactly the boundary, not something alike
            if sline == boundary or sline == last_boundary:
                self._at_eof = True
                self._unread.append(line)
                return b""
        else:
            next_line = await self._content.readline()
            if next_line.startswith(self._boundary):
                line = line[:-2]  # strip CRLF but only once
            self._unread.append(next_line)

        return line

    async def release(self) -> None:
        """Like read(), but reads all the data to the void."""
        if self._at_eof:
            return
        while not self._at_eof:
            await self.read_chunk(self.chunk_size)

    async def text(self, *, encoding: str | None = None) -> str:
        """Like read(), but assumes that body part contains text data."""
        data = await self.read(decode=True)
        # see https://www.w3.org/TR/html5/forms.html#multipart/form-data-encoding-algorithm
        # and https://dvcs.w3.org/hg/xhr/raw-file/tip/Overview.html#dom-xmlhttprequest-send
        encoding = encoding or self.get_charset(default="utf-8")
        return data.decode(encoding)

    async def json(self, *, encoding: str | None = None) -> dict[str, Any] | None:
        """Like read(), but assumes that body parts contains JSON data."""
        data = await self.read(decode=True)
        if not data:
            return None
        encoding = encoding or self.get_charset(default="utf-8")
        return cast(dict[str, Any], json.loads(data.decode(encoding)))

    async def form(self, *, encoding: str | None = None) -> list[tuple[str, str]]:
        """Like read(), but assumes that body parts contain form urlencoded data."""
        data = await self.read(decode=True)
        if not data:
            return []
        if encoding is not None:
            real_encoding = encoding
        else:
            real_encoding = self.get_charset(default="utf-8")
        try:
            decoded_data = data.rstrip().decode(real_encoding)
        except UnicodeDecodeError:
            raise ValueError("data cannot be decoded with %s encoding" % real_encoding)

        return parse_qsl(
            decoded_data,
            keep_blank_values=True,
            encoding=real_encoding,
        )

    def at_eof(self) -> bool:
        """Returns True if the boundary was reached or False otherwise."""
        return self._at_eof

    def _apply_content_transfer_decoding(self, data: _Buffer) -> _Buffer | bytes:
        """Apply Content-Transfer-Encoding decoding if header is present."""
        if CONTENT_TRANSFER_ENCODING in self.headers:
            return self._decode_content_transfer(data)
        return data

    def _needs_content_decoding(self) -> bool:
        """Check if Content-Encoding decoding should be applied."""
        # https://datatracker.ietf.org/doc/html/rfc7578#section-4.8
        return not self._is_form_data and CONTENT_ENCODING in self.headers

    def decode(self, data: _Buffer) -> _Buffer | bytes:
        """Decodes data synchronously.

        Decodes data according the specified Content-Encoding
        or Content-Transfer-Encoding headers value.

        Note: For large payloads, consider using decode_iter() instead
        to avoid blocking the event loop during decompression.
        """
        decoded = self._apply_content_transfer_decoding(data)
        if self._needs_content_decoding():
            return self._decode_content(decoded)
        return decoded

    async def decode_iter(self, data: _Buffer) -> AsyncIterator[_Buffer | bytes]:
        """Async generator that yields decoded data chunks.

        Decodes data according the specified Content-Encoding
        or Content-Transfer-Encoding headers value.

        This method offloads decompression to an executor for large payloads
        to avoid blocking the event loop.
        """
        decoded = self._apply_content_transfer_decoding(data)
        if self._needs_content_decoding():
            async for d in self._decode_content_async(decoded):
                yield d
        else:
            yield decoded

    def _decode_content(self, data: _Buffer) -> _Buffer | bytes:
        encoding = self.headers.get(CONTENT_ENCODING, "").lower()
        if encoding == "identity":
            return data
        if encoding in {"deflate", "gzip"}:
            return ZLibDecompressor(
                encoding=encoding,
                suppress_deflate_header=True,
            ).decompress_sync(data, max_length=self._max_decompress_size)

        raise RuntimeError(f"unknown content encoding: {encoding}")

    async def _decode_content_async(
        self, data: _Buffer
    ) -> AsyncIterator[_Buffer | bytes]:
        encoding = self.headers.get(CONTENT_ENCODING, "").lower()
        if encoding == "identity":
            yield data
        elif encoding in {"deflate", "gzip"}:
            d = ZLibDecompressor(
                encoding=encoding,
                suppress_deflate_header=True,
            )
            yield await d.decompress(data, max_length=self._max_decompress_size)
