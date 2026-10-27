            while d.data_available:
                yield await d.decompress(b"", max_length=self._max_decompress_size)
        else:
            raise RuntimeError(f"unknown content encoding: {encoding}")

    def _decode_content_transfer(self, data: _Buffer) -> _Buffer | bytes:
        encoding = self.headers.get(CONTENT_TRANSFER_ENCODING, "").lower()

        if encoding == "base64":
            return base64.b64decode(data)
        elif encoding == "quoted-printable":
            return binascii.a2b_qp(data)
        elif encoding in ("binary", "8bit", "7bit"):
            return data
        else:
            raise RuntimeError(f"unknown content transfer encoding: {encoding}")

    def get_charset(self, default: str) -> str:
        """Returns charset parameter from Content-Type header or default."""
        ctype = self.headers.get(CONTENT_TYPE, "")
        mimetype = parse_mimetype(ctype)
        return mimetype.parameters.get("charset", self._default_charset or default)

    @reify
    def name(self) -> str | None:
        """Returns name specified in Content-Disposition header.

        If the header is missing or malformed, returns None.
        """
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "name")

    @reify
    def filename(self) -> str | None:
        """Returns filename specified in Content-Disposition header.

        Returns None if the header is missing or malformed.
        """
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "filename")


@payload_type(BodyPartReader, order=Order.try_first)
class BodyPartReaderPayload(Payload):
    _value: BodyPartReader
    # _autoclose = False (inherited) - Streaming reader that may have resources

    def __init__(self, value: BodyPartReader, *args: Any, **kwargs: Any) -> None:
        super().__init__(value, *args, **kwargs)

        params: dict[str, str] = {}
        if value.name is not None:
            params["name"] = value.name
        if value.filename is not None:
            params["filename"] = value.filename

        if params:
            self.set_content_disposition("attachment", True, **params)

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise TypeError("Unable to decode.")

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """Raises TypeError as body parts should be consumed via write().

        This is intentional: BodyPartReader payloads are designed for streaming
        large data (potentially gigabytes) and must be consumed only once via
        the write() method to avoid memory exhaustion. They cannot be buffered
        in memory for reuse.
        """
        raise TypeError("Unable to read body part as bytes. Use write() to consume.")

    async def write(self, writer: AbstractStreamWriter) -> None:
        field = self._value
        # Reading the part drains the underlying stream irreversibly, so mark the
        # payload consumed up front: even an interrupted write leaves nothing that
        # a retry or redirect could replay.
        self._consumed = True
        while chunk := await field.read_chunk(size=DEFAULT_CHUNK_SIZE):
            async for d in field.decode_iter(chunk):
                await writer.write(d)


class MultipartReader:
    """Multipart body reader."""

    #: Response wrapper, used when multipart readers constructs from response.
    response_wrapper_cls = MultipartResponseWrapper
    #: Multipart reader class, used to handle multipart/* body parts.
    #: None points to type(self)
    multipart_reader_cls: type["MultipartReader"] | None = None
    #: Body part reader class for non multipart/* content types.
    part_reader_cls = BodyPartReader

    def __init__(
        self,
        headers: Mapping[str, str],
        content: StreamReader,
        *,
        client_max_size: int = sys.maxsize,
        max_field_size: int = 8190,
        max_headers: int = 128,
        max_size_error_cls: type[Exception] = ValueError,
    ) -> None:
        self._mimetype = parse_mimetype(headers[CONTENT_TYPE])
        assert self._mimetype.type == "multipart", "multipart/* content type expected"
        if "boundary" not in self._mimetype.parameters:
            raise ValueError(
                "boundary missed for Content-Type: %s" % headers[CONTENT_TYPE]
            )

        self.headers = headers
        self._boundary = ("--" + self._get_boundary()).encode()
        self._client_max_size = client_max_size
        self._content = content
        self._default_charset: str | None = None
        self._last_part: MultipartReader | BodyPartReader | None = None
        self._max_field_size = max_field_size
        self._max_headers = max_headers
        self._max_size_error_cls = max_size_error_cls
        self._at_eof = False
        self._at_bof = True
        self._unread: list[bytes] = []

    def __aiter__(self: Self) -> Self:
        return self

    async def __anext__(
        self,
    ) -> Union["MultipartReader", BodyPartReader] | None:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    @classmethod
    def from_response(
        cls,
        response: "ClientResponse",
    ) -> MultipartResponseWrapper:
        """Constructs reader instance from HTTP response.

        :param response: :class:`~aiohttp.client.ClientResponse` instance
        """
        obj = cls.response_wrapper_cls(
            response, cls(response.headers, response.content)
        )
        return obj

    def at_eof(self) -> bool:
        """Returns True if the final boundary was reached, false otherwise."""
        return self._at_eof

    async def next(
        self,
    ) -> Union["MultipartReader", BodyPartReader] | None:
        """Emits the next multipart body part."""
        # So, if we're at BOF, we need to skip till the boundary.
        if self._at_eof:
            return None
        await self._maybe_release_last_part()
        if self._at_bof:
            await self._read_until_first_boundary()
            self._at_bof = False
        else:
            await self._read_boundary()
        if self._at_eof:  # we just read the last boundary, nothing to do there
            return None

        part = await self.fetch_next_part()
        # https://datatracker.ietf.org/doc/html/rfc7578#section-4.6
        if (
            self._last_part is None
            and self._mimetype.subtype == "form-data"
            and isinstance(part, BodyPartReader)
        ):
            _, params = parse_content_disposition(part.headers.get(CONTENT_DISPOSITION))
            if params.get("name") == "_charset_":
                # Longest encoding in https://encoding.spec.whatwg.org/encodings.json
                # is 19 characters, so 32 should be more than enough for any valid encoding.
                charset = await part.read_chunk(32)
                if len(charset) > 31:
                    raise RuntimeError("Invalid default charset")
                self._default_charset = charset.strip().decode()
                part = await self.fetch_next_part()
        self._last_part = part
        return self._last_part

    async def release(self) -> None:
        """Reads all the body parts to the void till the final boundary."""
        while not self._at_eof:
            item = await self.next()
            if item is None:
                break
            await item.release()

    async def fetch_next_part(
        self,
    ) -> Union["MultipartReader", BodyPartReader]:
        """Returns the next body part reader."""
        headers = await self._read_headers()
        return self._get_part_reader(headers)

    def _get_part_reader(
        self,
        headers: "CIMultiDictProxy[str]",
    ) -> Union["MultipartReader", BodyPartReader]:
        """Dispatches the response by the `Content-Type` header.

        Returns a suitable reader instance.

        :param dict headers: Response headers
        """
        ctype = headers.get(CONTENT_TYPE, "")
        mimetype = parse_mimetype(ctype)

        if mimetype.type == "multipart":
            if self.multipart_reader_cls is None:
                return type(self)(
                    headers,
                    self._content,
                    client_max_size=self._client_max_size,
                    max_field_size=self._max_field_size,
                    max_headers=self._max_headers,
                    max_size_error_cls=self._max_size_error_cls,
                )
            return self.multipart_reader_cls(
                headers,
                self._content,
                client_max_size=self._client_max_size,
                max_field_size=self._max_field_size,
                max_headers=self._max_headers,
                max_size_error_cls=self._max_size_error_cls,
            )
        else:
            return self.part_reader_cls(
                self._boundary,
                headers,
                self._content,
                subtype=self._mimetype.subtype,
                default_charset=self._default_charset,
                client_max_size=self._client_max_size,
                max_size_error_cls=self._max_size_error_cls,
            )

    def _get_boundary(self) -> str:
        boundary = self._mimetype.parameters["boundary"]
        if len(boundary) > 70:
            raise ValueError("boundary %r is too long (70 chars max)" % boundary)

        return boundary

    async def _readline(self) -> bytes:
        if self._unread:
            return self._unread.pop()
        return await self._content.readline()

    async def _read_until_first_boundary(self) -> None:
        while True:
            chunk = await self._readline()
            if chunk == b"":
                raise ValueError(
                    "Could not find starting boundary %r" % (self._boundary)
                )
            chunk = chunk.rstrip()
            if chunk == self._boundary:
                return
            elif chunk == self._boundary + b"--":
                self._at_eof = True
                return

    async def _read_boundary(self) -> None:
        chunk = (await self._readline()).rstrip()
        if chunk == self._boundary:
            pass
        elif chunk == self._boundary + b"--":
            self._at_eof = True
            epilogue = await self._readline()
            next_line = await self._readline()

            # the epilogue is expected and then either the end of input or the
            # parent multipart boundary, if the parent boundary is found then
            # it should be marked as unread and handed to the parent for
            # processing
            if next_line[:2] == b"--":
                self._unread.append(next_line)
            # otherwise the request is likely missing an epilogue and both
            # lines should be passed to the parent for processing
            # (this handles the old behavior gracefully)
            else:
                self._unread.extend([next_line, epilogue])
        else:
            raise ValueError(f"Invalid boundary {chunk!r}, expected {self._boundary!r}")

    async def _read_headers(self) -> "CIMultiDictProxy[str]":
        lines = []
        while True:
            chunk = await self._content.readline(max_line_length=self._max_field_size)
            chunk = chunk.rstrip(b"\r\n")
            lines.append(chunk)
            if not chunk:
                break
            if len(lines) > self._max_headers:
                raise BadHttpMessage("Too many headers received")
        parser = HeadersParser(max_field_size=self._max_field_size)
        headers, raw_headers = parser.parse_headers(lines)
        return headers

    async def _maybe_release_last_part(self) -> None:
        """Ensures that the last read body part is read completely."""
        if self._last_part is not None:
            if not self._last_part.at_eof():
                await self._last_part.release()
            self._unread.extend(self._last_part._unread)
            self._last_part = None


_Part = tuple[Payload, str, str]


class MultipartWriter(Payload):
    """Multipart body writer."""

    _value: None
    # _consumed = False (inherited) - Can be encoded multiple times
    _autoclose = True  # No file handles, just collects parts in memory

    def __init__(self, subtype: str = "mixed", boundary: str | None = None) -> None:
        boundary = boundary if boundary is not None else uuid.uuid4().hex
        # The underlying Payload API demands a str (utf-8), not bytes,
        # so we need to ensure we don't lose anything during conversion.
        # As a result, require the boundary to be ASCII only.
        # In both situations.

        try:
            self._boundary = boundary.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("boundary should contain ASCII only chars") from None
        ctype = f"multipart/{subtype}; boundary={self._boundary_value}"

        super().__init__(None, content_type=ctype)

        self._parts: list[_Part] = []
        self._is_form_data = subtype == "form-data"

    def __enter__(self) -> "MultipartWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    @property
    def consumed(self) -> bool:
        """Whether the writer or any of its parts can no longer be replayed."""
        return self._consumed or any(part.consumed for part, _, _ in self._parts)

    def __iter__(self) -> Iterator[_Part]:
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __bool__(self) -> bool:
        return True

    _valid_tchar_regex = re.compile(rb"\A[!#$%&'*+\-.^_`|~\w]+\Z")
    _invalid_qdtext_char_regex = re.compile(rb"[\x00-\x08\x0A-\x1F\x7F]")

    @property
    def _boundary_value(self) -> str:
        """Wrap boundary parameter value in quotes, if necessary.

        Reads self.boundary and returns a unicode string.
        """
        # Refer to RFCs 7231, 7230, 5234.
        #
        # parameter      = token "=" ( token / quoted-string )
        # token          = 1*tchar
        # quoted-string  = DQUOTE *( qdtext / quoted-pair ) DQUOTE
        # qdtext         = HTAB / SP / %x21 / %x23-5B / %x5D-7E / obs-text
        # obs-text       = %x80-FF
        # quoted-pair    = "\" ( HTAB / SP / VCHAR / obs-text )
        # tchar          = "!" / "#" / "$" / "%" / "&" / "'" / "*"
        #                  / "+" / "-" / "." / "^" / "_" / "`" / "|" / "~"
        #                  / DIGIT / ALPHA
        #                  ; any VCHAR, except delimiters
        # VCHAR           = %x21-7E
        value = self._boundary
        if re.match(self._valid_tchar_regex, value):
            return value.decode("ascii")  # cannot fail

        if re.search(self._invalid_qdtext_char_regex, value):
            raise ValueError("boundary value contains invalid characters")

        # escape %x5C and %x22
        quoted_value_content = value.replace(b"\\", b"\\\\")
        quoted_value_content = quoted_value_content.replace(b'"', b'\\"')

        return '"' + quoted_value_content.decode("ascii") + '"'

    @property
    def boundary(self) -> str:
        return self._boundary.decode("ascii")

    def append(self, obj: Any, headers: Mapping[str, str] | None = None) -> Payload:
        if headers is None:
            headers = CIMultiDict()

        if isinstance(obj, Payload):
            obj.headers.update(headers)
            return self.append_payload(obj)
        else:
            try:
                payload = get_payload(obj, headers=headers)
            except LookupError:
                raise TypeError("Cannot create payload from %r" % obj)
            else:
                return self.append_payload(payload)

    def append_payload(self, payload: Payload) -> Payload:
        """Adds a new body part to multipart writer."""
        encoding: str | None = None
        te_encoding: str | None = None
        if self._is_form_data:
            # https://datatracker.ietf.org/doc/html/rfc7578#section-4.7
            # https://datatracker.ietf.org/doc/html/rfc7578#section-4.8
            assert (
                not {CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TRANSFER_ENCODING}
                & payload.headers.keys()
            )
            # Set default Content-Disposition in case user doesn't create one
            if CONTENT_DISPOSITION not in payload.headers:
                name = f"section-{len(self._parts)}"
                payload.set_content_disposition("form-data", name=name)
        else:
            # compression
            encoding = payload.headers.get(CONTENT_ENCODING, "").lower()
            if encoding and encoding not in ("deflate", "gzip", "identity"):
                raise RuntimeError(f"unknown content encoding: {encoding}")
            if encoding == "identity":
                encoding = None

            # te encoding
            te_encoding = payload.headers.get(CONTENT_TRANSFER_ENCODING, "").lower()
            if te_encoding not in ("", "base64", "quoted-printable", "binary"):
                raise RuntimeError(f"unknown content transfer encoding: {te_encoding}")
            if te_encoding == "binary":
                te_encoding = None

            # size
            size = payload.size
            if size is not None and not (encoding or te_encoding):
                payload.headers[CONTENT_LENGTH] = str(size)

        self._parts.append((payload, encoding, te_encoding))  # type: ignore[arg-type]
        return payload

    def append_json(
        self, obj: Any, headers: Mapping[str, str] | None = None
    ) -> Payload:
        """Helper to append JSON part."""
        if headers is None:
            headers = CIMultiDict()

        return self.append_payload(JsonPayload(obj, headers=headers))

    def append_form(
        self,
        obj: Sequence[tuple[str, str]] | Mapping[str, str],
        headers: Mapping[str, str] | None = None,
    ) -> Payload:
        """Helper to append form urlencoded part."""
        assert isinstance(obj, (Sequence, Mapping))

        if headers is None:
            headers = CIMultiDict()

        if isinstance(obj, Mapping):
            obj = list(obj.items())
        data = urlencode(obj, doseq=True)

        return self.append_payload(
            StringPayload(
                data, headers=headers, content_type="application/x-www-form-urlencoded"
            )
        )

    @property
    def size(self) -> int | None:
        """Size of the payload."""
        total = 0
        for part, encoding, te_encoding in self._parts:
            part_size = part.size
            if encoding or te_encoding or part_size is None:
                return None

            total += int(
                2
                + len(self._boundary)
                + 2
                + part_size  # b'--'+self._boundary+b'\r\n'
                + len(part._binary_headers)
                + 2  # b'\r\n'
            )

        total += 2 + len(self._boundary) + 4  # b'--'+self._boundary+b'--\r\n'
        return total

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Return string representation of the multipart data.

        WARNING: This method may do blocking I/O if parts contain file payloads.
        It should not be called in the event loop. Use as_bytes().decode() instead.
        """
        return "".join(
            "--"
            + self.boundary
            + "\r\n"
            + part._binary_headers.decode(encoding, errors)
            + part.decode()
            for part, _e, _te in self._parts
        )

    async def as_bytes(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """Return bytes representation of the multipart data.

        This method is async-safe and calls as_bytes on underlying payloads.
        """
        parts: list[bytes] = []

        # Process each part
        for part, _e, _te in self._parts:
            # Add boundary
            parts.append(b"--" + self._boundary + b"\r\n")

            # Add headers
            parts.append(part._binary_headers)

            # Add payload content using as_bytes for async safety
            part_bytes = await part.as_bytes(encoding, errors)
            parts.append(part_bytes)

            # Add trailing CRLF
            parts.append(b"\r\n")

        # Add closing boundary
        parts.append(b"--" + self._boundary + b"--\r\n")

        return b"".join(parts)

    async def write(
        self, writer: AbstractStreamWriter, close_boundary: bool = True
    ) -> None:
        """Write body."""
        for part, encoding, te_encoding in self._parts:
            if self._is_form_data:
                # https://datatracker.ietf.org/doc/html/rfc7578#section-4.2
                assert CONTENT_DISPOSITION in part.headers
                assert "name=" in part.headers[CONTENT_DISPOSITION]

            await writer.write(b"--" + self._boundary + b"\r\n")
            await writer.write(part._binary_headers)

            if encoding or te_encoding:
                w = MultipartPayloadWriter(writer)
                if encoding:
                    w.enable_compression(encoding)
                if te_encoding:
                    w.enable_encoding(te_encoding)
                await part.write(w)  # type: ignore[arg-type]
                await w.write_eof()
            else:
                await part.write(writer)

            await writer.write(b"\r\n")

        if close_boundary:
            await writer.write(b"--" + self._boundary + b"--\r\n")

    async def close(self) -> None:
        """
        Close all part payloads that need explicit closing.

        IMPORTANT: This method must not await anything that might not finish
        immediately, as it may be called during cleanup/cancellation. Schedule
        any long-running operations without awaiting them.
        """
        if self._consumed:
            return
        self._consumed = True

        # Close all parts that need explicit closing
        # We catch and log exceptions to ensure all parts get a chance to close
        # we do not use asyncio.gather() here because we are not allowed
        # to suspend given we may be called during cleanup
        for idx, (part, _, _) in enumerate(self._parts):
            if not part.autoclose and not part.consumed:
                try:
                    await part.close()
                except Exception as exc:
                    internal_logger.error(
                        "Failed to close multipart part %d: %s", idx, exc, exc_info=True
                    )


class MultipartPayloadWriter:
    def __init__(self, writer: AbstractStreamWriter) -> None:
        self._writer = writer
        self._encoding: str | None = None
        self._compress: ZLibCompressor | None = None
        self._encoding_buffer: bytearray | None = None

    def enable_encoding(self, encoding: str) -> None:
        if encoding == "base64":
            self._encoding = encoding
            self._encoding_buffer = bytearray()
        elif encoding == "quoted-printable":
            self._encoding = "quoted-printable"

    def enable_compression(
        self, encoding: str = "deflate", strategy: int | None = None
    ) -> None:
        self._compress = ZLibCompressor(
            encoding=encoding,
            suppress_deflate_header=True,
            strategy=strategy,
        )

    async def write_eof(self) -> None:
        if self._compress is not None:
            chunk = self._compress.flush()
            if chunk:
                self._compress = None
                await self.write(chunk)

        if self._encoding == "base64":
            if self._encoding_buffer:
                await self._writer.write(base64.b64encode(self._encoding_buffer))

    async def write(self, chunk: bytes) -> None:
        if self._compress is not None:
            if chunk:
                chunk = await self._compress.compress(chunk)
                if not chunk:
                    return

        if self._encoding == "base64":
            buf = self._encoding_buffer
            assert buf is not None
            buf.extend(chunk)

            if buf:
                div, mod = divmod(len(buf), 3)
                enc_chunk, self._encoding_buffer = (buf[: div * 3], buf[div * 3 :])
                if enc_chunk:
                    b64chunk = base64.b64encode(enc_chunk)
                    await self._writer.write(b64chunk)
        elif self._encoding == "quoted-printable":
            await self._writer.write(binascii.b2a_qp(chunk))
        else:
            await self._writer.write(chunk)
<|endoftext|>"""HTTP related errors."""

import asyncio
import warnings
from typing import TYPE_CHECKING, Union

from multidict import MultiMapping

from .typedefs import StrOrURL

if TYPE_CHECKING:
    import ssl

    SSLContext = ssl.SSLContext
else:
    try:
        import ssl

        SSLContext = ssl.SSLContext
    except ImportError:  # pragma: no cover
        ssl = SSLContext = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .client_reqrep import ClientResponse, ConnectionKey, Fingerprint, RequestInfo
    from .http_parser import RawResponseMessage
else:
    RequestInfo = ClientResponse = ConnectionKey = RawResponseMessage = None

__all__ = (
    "ClientError",
    "ClientConnectionError",
    "ClientConnectionResetError",
    "ClientOSError",
    "ClientConnectorError",
    "ClientProxyConnectionError",
    "ClientSSLError",
    "ClientConnectorDNSError",
    "ClientConnectorSSLError",
    "ClientConnectorCertificateError",
    "ConnectionTimeoutError",
    "SocketTimeoutError",
    "ServerConnectionError",
    "ServerTimeoutError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ClientResponseError",
    "ClientHttpProxyError",
    "WSServerHandshakeError",
    "ContentTypeError",
    "ClientPayloadError",
    "UploadAbortedError",
    "InvalidURL",
    "InvalidUrlClientError",
    "RedirectClientError",
    "NonHttpUrlClientError",
    "InvalidUrlRedirectClientError",
    "NonHttpUrlRedirectClientError",
    "WSMessageTypeError",
)


class ClientError(Exception):
    """Base class for client connection errors."""


class ClientResponseError(ClientError):
    """Base class for exceptions that occur after getting a response.

    request_info: An instance of RequestInfo.
    history: A sequence of responses, if redirects occurred.
    status: HTTP status code.
    message: Error message.
    headers: Response headers.
    """

    def __init__(
        self,
        request_info: RequestInfo,
        history: tuple[ClientResponse, ...],
        *,
        code: int | None = None,
        status: int | None = None,
        message: str = "",
        headers: MultiMapping[str] | None = None,
    ) -> None:
        self.request_info = request_info
        if code is not None:
            if status is not None:
                raise ValueError(
                    "Both code and status arguments are provided; "
                    "code is deprecated, use status instead"
                )
            warnings.warn(
                "code argument is deprecated, use status instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if status is not None:
            self.status = status
        elif code is not None:
            self.status = code
        else:
            self.status = 0
        self.message = message
        self.headers = headers
        self.history = history
        self.args = (request_info, history)

    def __str__(self) -> str:
        return f"{self.status}, message={self.message!r}, url={str(self.request_info.real_url)!r}"

    def __repr__(self) -> str:
        args = f"{self.request_info!r}, {self.history!r}"
        if self.status != 0:
            args += f", status={self.status!r}"
        if self.message != "":
            args += f", message={self.message!r}"
        if self.headers is not None:
            args += f", headers={self.headers!r}"
        return f"{type(self).__name__}({args})"

    @property
    def code(self) -> int:
        warnings.warn(
            "code property is deprecated, use status instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.status

    @code.setter
    def code(self, value: int) -> None:
        warnings.warn(
            "code property is deprecated, use status instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.status = value


class ContentTypeError(ClientResponseError):
    """ContentType found is not valid."""


class WSServerHandshakeError(ClientResponseError):
    """websocket server handshake error."""


class ClientHttpProxyError(ClientResponseError):
    """HTTP proxy error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
    proxy responds with status other than ``200 OK``
    on ``CONNECT`` request.
    """


class TooManyRedirects(ClientResponseError):
    """Client was redirected too many times."""


class ClientConnectionError(ClientError):
    """Base class for client socket errors."""


class ClientConnectionResetError(ClientConnectionError, ConnectionResetError):
    """ConnectionResetError"""


class ClientOSError(ClientConnectionError, OSError):
    """OSError error."""


class ClientConnectorError(ClientOSError):
    """Client connector error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        a connection can not be established.
    """

    def __init__(self, connection_key: ConnectionKey, os_error: OSError) -> None:
        self._conn_key = connection_key
        self._os_error = os_error
        super().__init__(os_error.errno, os_error.strerror)
        self.args = (connection_key, os_error)

    @property
    def os_error(self) -> OSError:
        return self._os_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> int | None:
        return self._conn_key.port

    @property
    def ssl(self) -> Union[SSLContext, bool, "Fingerprint"]:
        return self._conn_key.ssl

    def __str__(self) -> str:
        return "Cannot connect to host {0.host}:{0.port} ssl:{1} [{2}]".format(
            self, "default" if self.ssl is True else self.ssl, self.strerror
        )

    # OSError.__reduce__ does too much black magick
    __reduce__ = BaseException.__reduce__


class ClientConnectorDNSError(ClientConnectorError):
    """DNS resolution failed during client connection.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        DNS resolution fails.
    """


class ClientProxyConnectionError(ClientConnectorError):
    """Proxy connection error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        connection to proxy can not be established.
    """


class UnixClientConnectorError(ClientConnectorError):
    """Unix connector error.

    Raised in :py:class:`aiohttp.connector.UnixConnector`
    if connection to unix socket can not be established.
    """

    def __init__(
        self, path: str, connection_key: ConnectionKey, os_error: OSError
    ) -> None:
        self._path = path
        super().__init__(connection_key, os_error)

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return "Cannot connect to unix socket {0.path} ssl:{1} [{2}]".format(
            self, "default" if self.ssl is True else self.ssl, self.strerror
        )


class ServerConnectionError(ClientConnectionError):
    """Server connection errors."""


class ServerDisconnectedError(ServerConnectionError):
    """Server disconnected."""

    def __init__(self, message: RawResponseMessage | str | None = None) -> None:
        if message is None:
            message = "Server disconnected"

        self.args = (message,)
        self.message = message


class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError):
    """Server timeout error."""


class ConnectionTimeoutError(ServerTimeoutError):
    """Connection timeout error."""


class SocketTimeoutError(ServerTimeoutError):
    """Socket timeout error."""


class ServerFingerprintMismatch(ServerConnectionError):
    """SSL certificate does not match expected fingerprint."""

    def __init__(self, expected: bytes, got: bytes, host: str, port: int) -> None:
        self.expected = expected
        self.got = got
        self.host = host
        self.port = port
        self.args = (expected, got, host, port)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} expected={self.expected!r} got={self.got!r} host={self.host!r} port={self.port!r}>"


class ClientPayloadError(ClientError):
    """Response payload error."""


class UploadAbortedError(ClientError):
    """The request body was never fully sent."""


class InvalidURL(ClientError, ValueError):
    """Invalid URL.

    URL used for fetching is malformed, e.g. it doesn't contains host
    part.
    """

    # Derive from ValueError for backward compatibility

    def __init__(self, url: StrOrURL, description: str | None = None) -> None:
        # The type of url is not yarl.URL because the exception can be raised
        # on URL(url) call
        self._url = url
        self._description = description

        if description:
            super().__init__(url, description)
        else:
            super().__init__(url)

    @property
    def url(self) -> StrOrURL:
        return self._url

    @property
    def description(self) -> "str | None":
        return self._description

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self}>"

    def __str__(self) -> str:
        if self._description:
            return f"{self._url} - {self._description}"
        return str(self._url)


class InvalidUrlClientError(InvalidURL):
    """Invalid URL client error."""


class RedirectClientError(ClientError):
    """Client redirect error."""


class NonHttpUrlClientError(ClientError):
    """Non http URL client error."""


class InvalidUrlRedirectClientError(InvalidUrlClientError, RedirectClientError):
    """Invalid URL redirect client error."""


class NonHttpUrlRedirectClientError(NonHttpUrlClientError, RedirectClientError):
    """Non http URL redirect client error."""


class ClientSSLError(ClientConnectorError):
    """Base error for ssl.*Errors."""


if ssl is not None:
    cert_errors = (ssl.CertificateError,)
    cert_errors_bases = (
        ClientSSLError,
        ssl.CertificateError,
    )

    ssl_errors = (ssl.SSLError,)
    ssl_error_bases = (ClientSSLError, ssl.SSLError)
else:  # pragma: no cover
    cert_errors = tuple()
    cert_errors_bases = (
        ClientSSLError,
        ValueError,
    )

    ssl_errors = tuple()
    ssl_error_bases = (ClientSSLError,)


class ClientConnectorSSLError(*ssl_error_bases):  # type: ignore[misc]
    """Response ssl error."""


class ClientConnectorCertificateError(*cert_errors_bases):  # type: ignore[misc]
    """Response certificate error."""

    _conn_key: ConnectionKey

    def __init__(
        # TODO: If we require ssl in future, this can become ssl.CertificateError
        self,
        connection_key: ConnectionKey,
        certificate_error: Exception,
    ) -> None:
        if isinstance(certificate_error, cert_errors + (OSError,)):
            # ssl.CertificateError has errno and strerror, so we should be fine
            os_error = certificate_error
        else:
            os_error = OSError()

        super().__init__(connection_key, os_error)
        self._certificate_error = certificate_error
        self.args = (connection_key, certificate_error)

    @property
    def certificate_error(self) -> Exception:
        return self._certificate_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> int | None:
        return self._conn_key.port

    @property
    def ssl(self) -> bool:
        return self._conn_key.is_ssl

    def __str__(self) -> str:
        return (
            f"Cannot connect to host {self.host}:{self.port} ssl:{self.ssl} "
            f"[{self.certificate_error.__class__.__name__}: "
            f"{self.certificate_error.args}]"
        )


class WSMessageTypeError(TypeError):
    """WebSocket message type is not valid."""
<|endoftext|>"""
Internal cookie handling helpers.

This module contains internal utilities for cookie parsing and manipulation.
These are not part of the public API and may change without notice.
"""

import re
from collections.abc import Sequence
from http.cookies import CookieError, Morsel

from .log import internal_logger

__all__ = (
    "parse_set_cookie_headers",
    "parse_cookie_header",
    "preserve_morsel_with_coded_value",
)

# Cookie parsing constants
# Allow more characters in cookie names to handle real-world cookies
# that don't strictly follow RFC standards (fixes #2683)
# RFC 6265 defines cookie-name token as per RFC 2616 Section 2.2,
# but many servers send cookies with characters like {} [] () etc.
# This makes the cookie parser more tolerant of real-world cookies
# while still providing some validation to catch obviously malformed names.
_COOKIE_NAME_RE = re.compile(r"^[!#$%&\'()*+\-./0-9:<=>?@A-Z\[\]^_`a-z{|}~]+$")
_COOKIE_KNOWN_ATTRS = frozenset(  # AKA Morsel._reserved
    (
        "path",
        "domain",
        "max-age",
        "expires",
        "secure",
        "httponly",
        "samesite",
        "partitioned",
        "version",
        "comment",
    )
)
_COOKIE_BOOL_ATTRS = frozenset(  # AKA Morsel._flags
    ("secure", "httponly", "partitioned")
)

# Implementation limits as permitted by RFC 6265 Section 6.1.
_MAX_COOKIES_PER_RESPONSE = 50
_MAX_COOKIE_PAIR_LENGTH = 4096
_MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH = 1024
# Like Chromium's ParsedCookie::kMaxPairs, but more lenient.
_MAX_COOKIE_ATTRIBUTES = 32

# Where the next attribute can start: skips runs of ";" and whitespace.
_ATTRIBUTE_START_RE = re.compile(r"[^;\s]", re.ASCII)
_COOKIE_FORBIDDEN_CTL_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")
_COOKIE_CTL_RE = re.compile(r"[\x00-\x1f\x7f]")

# SimpleCookie's pattern for parsing cookies with relaxed validation
# Based on http.cookies pattern but extended to allow more characters in cookie names
# to handle real-world cookies (fixes #2683)
_COOKIE_PATTERN = re.compile(
    r"""
    \s*                            # Optional whitespace at start of cookie
    (?P<key>                       # Start of group 'key'
    # aiohttp has extended to include [] for compatibility with real-world cookies
    [\w\d!#%&'~_`><@,:/\$\*\+\-\.\^\|\)\(\?\}\{\[\]]+   # Any word of at least one letter
    )                              # End of group 'key'
    (                              # Optional group: there may not be a value.
    \s*=\s*                          # Equal Sign
    (?P<val>                         # Start of group 'val'
    "(?:[^\\"]|\\.)*"                  # Any double-quoted string (properly closed)
    |                                  # or
    "[^";]*                            # Unmatched opening quote (differs from SimpleCookie - issue #7993)
    |                                  # or
    # Special case for "expires" attr - RFC 822, RFC 850, RFC 1036, RFC 1123
    (\w{3,6}day|\w{3}),\s              # Day of the week or abbreviated day (with comma)
    [\w\d\s-]{9,11}\s[\d:]{8}\s        # Date and time in specific format
    (GMT|[+-]\d{4})                     # Timezone: GMT or RFC 2822 offset like -0000, +0100
                                        # NOTE: RFC 2822 timezone support is an aiohttp extension
                                        # for issue #4493 - SimpleCookie does NOT support this
    |                                  # or
    # ANSI C asctime() format: "Wed Jun  9 10:18:14 2021"
    # NOTE: This is an aiohttp extension for issue #4327 - SimpleCookie does NOT support this format
    \w{3}\s+\w{3}\s+[\s\d]\d\s+\d{2}:\d{2}:\d{2}\s+\d{4}
    |                                  # or
    [\w\d!#%&'~_`><@,:/\$\*\+\-\.\^\|\)\(\?\}\{\=\[\]]*      # Any word or empty string
    )                                # End of group 'val'
    )?                             # End of optional value group
    \s*                            # Any number of spaces.
    (\s+|;|$)                      # Ending either at space, semicolon, or EOS.
    """,
    re.VERBOSE | re.ASCII,
)


def preserve_morsel_with_coded_value(cookie: Morsel[str]) -> Morsel[str]:
    """
    Preserve a Morsel's coded_value exactly as received from the server.

    This function ensures that cookie encoding is preserved exactly as sent by
    the server, which is critical for compatibility with old servers that have
    strict requirements about cookie formats.

    This addresses the issue described in https://github.com/aio-libs/aiohttp/pull/1453
    where Python's SimpleCookie would re-encode cookies, breaking authentication
    with certain servers.

    Args:
        cookie: A Morsel object from SimpleCookie

    Returns:
        A Morsel object with preserved coded_value

    """
    # Don't look up cookie.key in the Morsel: it maps attribute names, so a
    # cookie named ``path`` would return the Path attribute.
    mrsl_val: Morsel[str] = Morsel()
    # We use __setstate__ instead of the public set() API because it allows us to
    # bypass validation and set already validated state. This is more stable than
    # setting protected attributes directly and unlikely to change since it would
    # break pickling.
    try:
        mrsl_val.__setstate__(  # type: ignore[attr-defined]
            {
                "key": cookie.key,
                "value": cookie.value,
                "coded_value": cookie.coded_value,
            }
        )
    except CookieError:
        return cookie
    return mrsl_val


_unquote_sub = re.compile(r"\\(?:([0-3][0-7][0-7])|(.))").sub


def _unquote_replace(m: re.Match[str]) -> str:
    """
    Replace function for _unquote_sub regex substitution.

    Handles escaped characters in cookie values:
    - Octal sequences are converted to their character representation
    - Other escaped characters are unescaped by removing the backslash
    """
    if m[1]:
        return chr(int(m[1], 8))
    return m[2]


def _unquote(value: str) -> str:
    """
    Unquote a cookie value.

    Vendored from http.cookies._unquote to ensure compatibility.

    Note: The original implementation checked for None, but we've removed
    that check since all callers already ensure the value is not None.
    """
    # If there aren't any doublequotes,
    # then there can't be any special characters.  See RFC 2109.
    if len(value) < 2:
        return value
    if value[0] != '"' or value[-1] != '"':
        return value

    # We have to assume that we must decode this string.
    # Down to work.

    # Remove the "s
    value = value[1:-1]

    # Check for special sequences.  Examples:
    #    \012 --> \n
    #    \"   --> "
    #
    return _unquote_sub(_unquote_replace, value)


def parse_cookie_header(header: str) -> list[tuple[str, Morsel[str]]]:
    """
    Parse a Cookie header according to RFC 6265 Section 5.4.

    Cookie headers contain only name-value pairs separated by semicolons.
    There are no attributes in Cookie headers - even names that match
    attribute names (like 'path' or 'secure') should be treated as cookies.

    This parser uses _COOKIE_PATTERN to properly handle quoted values that may
    contain semicolons. When the regex fails to match a malformed cookie, it
    falls back to simple parsing to ensure subsequent cookies are not lost
    https://github.com/aio-libs/aiohttp/issues/11632

    Args:
        header: The Cookie header value to parse

    Returns:
        List of (name, Morsel) tuples for compatibility with SimpleCookie.update()
    """
    if not header:
        return []

    cookies: list[tuple[str, Morsel[str]]] = []
    morsel: Morsel[str]
    i = 0
    n = len(header)

    invalid_names = []
    while i < n:
        # Find the next cookie-pair
        match = _COOKIE_PATTERN.match(header, i)
        if not match:
            # Fallback for malformed cookies https://github.com/aio-libs/aiohttp/issues/11632
            # Find next semicolon to skip or attempt simple key=value parsing
            next_semi = header.find(";", i)
            eq_pos = header.find("=", i)

            # Try to extract key=value if '=' comes before ';'
            if eq_pos != -1 and (next_semi == -1 or eq_pos < next_semi):
                end_pos = next_semi if next_semi != -1 else n
                key = header[i:eq_pos].strip()
                value = header[eq_pos + 1 : end_pos].strip()

                # Validate the name (same as regex path)
                if not _COOKIE_NAME_RE.match(key):
                    invalid_names.append(key)
                else:
                    morsel = Morsel()
                    try:
                        morsel.__setstate__(  # type: ignore[attr-defined]
                            {
                                "key": key,
                                "value": _unquote(value),
                                "coded_value": value,
                            }
                        )
                    except CookieError:
                        pass
                    else:
                        cookies.append((key, morsel))

            # Move to next cookie or end
            i = next_semi + 1 if next_semi != -1 else n
            continue

        key = match.group("key")
        value = match.group("val") or ""
        i = match.end(0)

        # Validate the name
        if not key or not _COOKIE_NAME_RE.match(key):
            invalid_names.append(key)
            continue

        # Create new morsel
        morsel = Morsel()
        # Preserve the original value as coded_value (with quotes if present)
        # We use __setstate__ instead of the public set() API because it allows us to
        # bypass validation and set already validated state. This is more stable than
        # setting protected attributes directly and unlikely to change since it would
        # break pickling.
        try:
            morsel.__setstate__(  # type: ignore[attr-defined]
                {"key": key, "value": _unquote(value), "coded_value": value}
            )
        except CookieError:
            continue

        cookies.append((key, morsel))

    if invalid_names:
        internal_logger.debug(
            "Cannot load cookie. Illegal cookie names: %r", invalid_names
        )

    return cookies


def _encoded_length(text: str) -> int | None:
    """Return the UTF-8 length of text, or None if it can't be encoded."""
    if text.isascii():
        return len(text)
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _apply_cookie_attributes(morsel: Morsel[str], attributes: str) -> None:
    """Set the recognized attributes from the text after the cookie-pair.

    The field must already be free of control characters other than tab.
    """
    # Walk with the cookie pattern so a quoted value keeps its ";". Missing
    # semicolons between attributes are tolerated; every match is an attribute
    # of this cookie, never a new cookie.
    index = 0
    end = len(attributes)
    for _ in range(_MAX_COOKIE_ATTRIBUTES):
        if index >= end:
            break
        if (attribute_match := _COOKIE_PATTERN.match(attributes, index)) is None:
            # Skip a malformed segment and any run of ";" or whitespace after it.
            if (
                not (next_index := attributes.find(";", index) + 1)
                or (start := _ATTRIBUTE_START_RE.search(attributes, next_index)) is None
            ):
                break
            index = start.start()
            continue
        index = attribute_match.end()
        lower_key = attribute_match.group("key").lower()
        if lower_key not in _COOKIE_KNOWN_ATTRS:
            # RFC 6265 Section 5.2: ignore unknown attributes.
            continue
        if lower_key not in morsel._reserved:  # type: ignore[attr-defined]
            # Python versions before 3.14 do not expose Partitioned.
            continue

        # The key and value are checked here, so skip Morsel.__setitem__'s checks.
        if lower_key in _COOKIE_BOOL_ATTRS:
            # Like Firefox, a flag's value is ignored.
            dict.__setitem__(morsel, lower_key, True)
            continue
        attr_value = attribute_match.group("val") or ""
        if (
            len(attr_value) <= _MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH
            and attr_value.isascii()
            and attr_value[:1] != '"'
            and "\t" not in attr_value
        ):
            # The field has no other control characters and nothing to unquote.
            dict.__setitem__(morsel, lower_key, attr_value)
            continue
        attr_value_length = _encoded_length(attr_value)
        if (
            attr_value_length is None
            or attr_value_length > _MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH
        ):
            continue
        # Patched CPython rejects control characters, even ones produced by
        # unquoting, so ignore such an attribute rather than raise.
        if _COOKIE_CTL_RE.search(value := _unquote(attr_value)) is None:
            dict.__setitem__(morsel, lower_key, value)


def _parse_quoted_pair(header: str) -> tuple[str, str, int] | None:
    """Parse a first pair whose quoted value contains ";".

    aiohttp has always accepted these. Returns the name, the quoted value and
    where the attributes start, or None to split at the first ";" instead.
    """
    match = _COOKIE_PATTERN.match(header)
    if match is None or not (value := match.group("val")):
        return None
    if not (value.startswith('"') and value.endswith('"') and ";" in value):
        return None
    tail = header[match.end("val") :].lstrip(" \t")
    if tail and not tail.startswith(";"):
        return None
    return match.group("key"), value, len(header) - len(tail) + bool(tail)


def parse_set_cookie_headers(headers: Sequence[str]) -> list[tuple[str, Morsel[str]]]:
    """Parse Set-Cookie fields into at most one cookie per field.

    Fields with control characters, oversized or nameless pairs, names outside
    the allowlist, or tabs in the pair are skipped.
    """
    parsed_cookies: list[tuple[str, Morsel[str]]] = []
    over_limit = 0

    for position, header in enumerate(headers):
        if len(parsed_cookies) >= _MAX_COOKIES_PER_RESPONSE:
            over_limit = len(headers) - position
            break
        if not header or _COOKIE_FORBIDDEN_CTL_RE.search(header) is not None:
            continue

        if (pair_end := header.find(";")) == -1:
            pair_end = attributes_start = len(header)
        else:
            attributes_start = pair_end + 1
        # A quote before the first ";" may start a quoted value containing ";".
        if header.find('"', 0, pair_end) != -1 and (
            quoted := _parse_quoted_pair(header)
        ):
            key, coded_value, attributes_start = quoted
        else:
            key, sep, coded_value = header[:pair_end].partition("=")
            if not sep:
                continue

        key = key.strip(" \t")
        coded_value = coded_value.strip(" \t")

        if not key:
            continue
        if not _COOKIE_NAME_RE.match(key):
            internal_logger.warning("Can not load cookies: Illegal cookie name %r", key)
            continue
        pair_length = _encoded_length(key + coded_value)
        if pair_length is None or pair_length > _MAX_COOKIE_PAIR_LENGTH:
            continue

        value = _unquote(coded_value)
        if _COOKIE_CTL_RE.search(value) is not None:
            continue
        morsel: Morsel[str] = Morsel()
        try:
            morsel.__setstate__(  # type: ignore[attr-defined]
                {"key": key, "value": value, "coded_value": coded_value}
            )
        except CookieError:
            continue

        _apply_cookie_attributes(morsel, header[attributes_start:])
        parsed_cookies.append((key, morsel))

    if (invalid := len(headers) - over_limit - len(parsed_cookies)) or over_limit:
        internal_logger.debug(
            "Ignored %d invalid Set-Cookie field(s) and %d over the %d-cookie limit",
            invalid,
            over_limit,
            _MAX_COOKIES_PER_RESPONSE,
        )
    return parsed_cookies
<|endoftext|>import asyncio
import base64
import binascii
import hashlib
import json
import sys
from asyncio.base_events import BaseEventLoop
from collections.abc import Callable, Iterable
from typing import Any, Final, Generic, Literal, cast, overload

import attr
from multidict import CIMultiDict

from . import hdrs
from ._websocket.reader import WebSocketDataQueue
from .abc import AbstractStreamWriter
from .client_exceptions import WSMessageTypeError
from .helpers import (
    DEFAULT_CHUNK_SIZE,
    calculate_timeout_when,
    set_exception,
    set_result,
)
from .http import (
    WS_CLOSED_MESSAGE,
    WS_CLOSING_MESSAGE,
    WS_KEY,
    WebSocketError,
    WebSocketReader,
    WebSocketWriter,
    WSCloseCode,
    WSMessage,
    WSMessageDecodeText,
    WSMessageNoDecodeText,
    WSMsgType as WSMsgType,
    ws_ext_gen,
    ws_ext_parse,
)
from .http_websocket import _INTERNAL_RECEIVE_TYPES
from .log import ws_logger
from .streams import EofStream
from .typedefs import JSONBytesEncoder, JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing import Union

    Buffer = Union[bytes, bytearray, "memoryview[int]", "memoryview[bytes]"]

if sys.version_info >= (3, 11):
    import asyncio as async_timeout
    from typing import Self
else:
    import async_timeout
    from typing_extensions import Self

__all__ = (
    "WebSocketResponse",
    "WebSocketReady",
    "WSMsgType",
)

THRESHOLD_CONNLOST_ACCESS: Final[int] = 5

# TypeVar for whether text messages are decoded to str (True) or kept as bytes (False)
_DecodeText = TypeVar("_DecodeText", bound=bool, covariant=True, default=Literal[True])


@attr.s(auto_attribs=True, frozen=True, slots=True)
class WebSocketReady:
    ok: bool
    protocol: str | None

    def __bool__(self) -> bool:
        return self.ok


class WebSocketResponse(StreamResponse, Generic[_DecodeText]):

    _length_check: bool = False
    _ws_protocol: str | None = None
    _writer: WebSocketWriter | None = None
    _reader: WebSocketDataQueue | None = None
    _parser: WebSocketReader | None = None
    _closed: bool = False
    _closing: bool = False
    _conn_lost: int = 0
    _close_code: int | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _waiting: bool = False
    _close_wait: asyncio.Future[None] | None = None
    _exception: BaseException | None = None
    _heartbeat_when: float = 0.0
    _heartbeat_cb: asyncio.TimerHandle | None = None
    _pong_response_cb: asyncio.TimerHandle | None = None
    _ping_task: asyncio.Task[None] | None = None
    _need_heartbeat_reset: bool = False
    _heartbeat_reset_handle: asyncio.Handle | None = None

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        receive_timeout: float | None = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: float | None = None,
        protocols: Iterable[str] = (),
        compress: bool = True,
        max_msg_size: int = 4 * 1024 * 1024,
        writer_limit: int = DEFAULT_CHUNK_SIZE,
        decode_text: bool = True,
    ) -> None:
        super().__init__(status=101)
        self._protocols = protocols
        self._timeout = timeout
        self._receive_timeout = receive_timeout
        self._autoclose = autoclose
        self._autoping = autoping
        self._heartbeat = heartbeat
        if heartbeat is not None:
            self._pong_heartbeat = heartbeat / 2.0
        self._compress: bool | int = compress
        self._max_msg_size = max_msg_size
        self._writer_limit = writer_limit
        self._decode_text = decode_text
        self._need_heartbeat_reset = False
        self._heartbeat_reset_handle = None

    def _cancel_heartbeat(self) -> None:
        self._cancel_pong_response_cb()
        if self._heartbeat_reset_handle is not None:
            self._heartbeat_reset_handle.cancel()
            self._heartbeat_reset_handle = None
        self._need_heartbeat_reset = False
        if self._heartbeat_cb is not None:
            self._heartbeat_cb.cancel()
            self._heartbeat_cb = None
        if self._ping_task is not None:
            self._ping_task.cancel()
            self._ping_task = None

    def _cancel_pong_response_cb(self) -> None:
        if self._pong_response_cb is not None:
            self._pong_response_cb.cancel()
            self._pong_response_cb = None

    def _on_data_received(self) -> None:
        if self._heartbeat is None or self._need_heartbeat_reset:
            return
        loop = self._loop
        assert loop is not None
        # Coalesce multiple chunks received in the same loop tick into a single
        # heartbeat reset. Resetting immediately per chunk increases timer churn.
        self._need_heartbeat_reset = True
        self._heartbeat_reset_handle = loop.call_soon(self._flush_heartbeat_reset)

    def _flush_heartbeat_reset(self) -> None:
        self._heartbeat_reset_handle = None
        if not self._need_heartbeat_reset:
            return
        self._reset_heartbeat()
        self._need_heartbeat_reset = False

    def _reset_heartbeat(self) -> None:
        if self._heartbeat is None:
            return
        self._cancel_pong_response_cb()
        req = self._req
        timeout_ceil_threshold = (
            req._protocol._timeout_ceil_threshold if req is not None else 5
        )
        loop = self._loop
        assert loop is not None
        now = loop.time()
        when = calculate_timeout_when(now, self._heartbeat, timeout_ceil_threshold)
        self._heartbeat_when = when
        if self._heartbeat_cb is None:
            # We do not cancel the previous heartbeat_cb here because
            # it generates a significant amount of TimerHandle churn
            # which causes asyncio to rebuild the heap frequently.
            # Instead _send_heartbeat() will reschedule the next
            # heartbeat if it fires too early.
            self._heartbeat_cb = loop.call_at(when, self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        self._heartbeat_cb = None

        # If heartbeat reset is pending (data is being received), skip sending
        # the ping and let the reset callback handle rescheduling the heartbeat.
        if self._need_heartbeat_reset:
            return

        loop = self._loop
        assert loop is not None and self._writer is not None
        now = loop.time()
        if now < self._heartbeat_when:
            # Heartbeat fired too early, reschedule
            self._heartbeat_cb = loop.call_at(
                self._heartbeat_when, self._send_heartbeat
            )
            return

        req = self._req
        timeout_ceil_threshold = (
            req._protocol._timeout_ceil_threshold if req is not None else 5
        )
        when = calculate_timeout_when(now, self._pong_heartbeat, timeout_ceil_threshold)
        self._cancel_pong_response_cb()
        self._pong_response_cb = loop.call_at(when, self._pong_not_received)

        coro = self._writer.send_frame(b"", WSMsgType.PING)
        if sys.version_info >= (3, 14):
            # Try to send the ping immediately to avoid having to schedule
            # the task on the event loop.
            if isinstance(loop, BaseEventLoop):
                ping_task = asyncio.create_task(coro, eager_start=True)
            else:
                ping_task = asyncio.Task(coro, loop=loop, eager_start=True)
        elif sys.version_info >= (3, 12):
            ping_task = asyncio.Task(coro, loop=loop, eager_start=True)
        else:
            ping_task = asyncio.create_task(coro)

        if not ping_task.done():
            self._ping_task = ping_task
            ping_task.add_done_callback(self._ping_task_done)
        else:
            self._ping_task_done(ping_task)

    def _ping_task_done(self, task: "asyncio.Task[None]") -> None:
        """Callback for when the ping task completes."""
        if not task.cancelled() and (exc := task.exception()):
            self._handle_ping_pong_exception(exc)
        self._ping_task = None

    def _pong_not_received(self) -> None:
        if self._req is not None and self._req.transport is not None:
            self._handle_ping_pong_exception(
                asyncio.TimeoutError(
                    f"No PONG received after {self._pong_heartbeat} seconds"
                )
            )

    def _handle_ping_pong_exception(self, exc: BaseException) -> None:
        """Handle exceptions raised during ping/pong processing."""
        if self._closed:
            return
        self._set_closed()
        # close() is never reached after this; release the parser here.
        self._parser = None
        self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
        self._exception = exc
        if self._waiting and not self._closing and self._reader is not None:
            self._reader.feed_data(WSMessage(WSMsgType.ERROR, exc, None), 0)

    def _set_closed(self) -> None:
        """Set the connection to closed.

        Cancel any heartbeat timers and set the closed flag.
        """
        self._closed = True
        self._cancel_heartbeat()

    async def prepare(self, request: BaseRequest) -> AbstractStreamWriter:
        # make pre-check to don't hide it by do_handshake() exceptions
        if self._payload_writer is not None:
            return self._payload_writer

        protocol, writer = self._pre_start(request)
        payload_writer = await super().prepare(request)
        assert payload_writer is not None
        self._post_start(request, protocol, writer)
        await payload_writer.drain()
        return payload_writer

    def _handshake(
        self, request: BaseRequest
    ) -> tuple["CIMultiDict[str]", str | None, int, bool]:
        headers = request.headers
        if "websocket" != headers.get(hdrs.UPGRADE, "").lower().strip():
            raise HTTPBadRequest(
                text=(
                    f"No WebSocket UPGRADE hdr: {headers.get(hdrs.UPGRADE)}\n Can "
                    '"Upgrade" only to "WebSocket".'
                )
            )

        if not request._message.upgrade:
            raise HTTPBadRequest(
                text=f"No CONNECTION upgrade hdr: {headers.get(hdrs.CONNECTION)}"
            )

        # find common sub-protocol between client and server
        protocol: str | None = None
        if hdrs.SEC_WEBSOCKET_PROTOCOL in headers:
            req_protocols = [
                str(proto.strip())
                for proto in headers[hdrs.SEC_WEBSOCKET_PROTOCOL].split(",")
            ]

            for proto in req_protocols:
                if proto in self._protocols:
                    protocol = proto
                    break
            else:
                # No overlap found: Return no protocol as per spec
                ws_logger.warning(
                    "%s: Client protocols %r don’t overlap server-known ones %r",
                    request.remote,
                    req_protocols,
                    self._protocols,
                )

        # check supported version
        version = headers.get(hdrs.SEC_WEBSOCKET_VERSION, "")
        if version not in ("13", "8", "7"):
            raise HTTPBadRequest(text=f"Unsupported version: {version}")

        # check client handshake for validity
        key = headers.get(hdrs.SEC_WEBSOCKET_KEY)
        try:
            if not key or len(base64.b64decode(key)) != 16:
                raise HTTPBadRequest(text=f"Handshake error: {key!r}")
        except binascii.Error:
            raise HTTPBadRequest(text=f"Handshake error: {key!r}") from None

        accept_val = base64.b64encode(
            hashlib.sha1(key.encode() + WS_KEY).digest()
        ).decode()
        response_headers = CIMultiDict(
            {
                hdrs.UPGRADE: "websocket",
                hdrs.CONNECTION: "upgrade",
                hdrs.SEC_WEBSOCKET_ACCEPT: accept_val,
            }
        )

        notakeover = False
        compress = 0
        if self._compress:
            extensions = headers.get(hdrs.SEC_WEBSOCKET_EXTENSIONS)
            # Server side always get return with no exception.
            # If something happened, just drop compress extension
            compress, notakeover = ws_ext_parse(extensions, isserver=True)
            if compress:
                enabledext = ws_ext_gen(
                    compress=compress, isserver=True, server_notakeover=notakeover
                )
                response_headers[hdrs.SEC_WEBSOCKET_EXTENSIONS] = enabledext

        if protocol:
            response_headers[hdrs.SEC_WEBSOCKET_PROTOCOL] = protocol
        return (
            response_headers,
            protocol,
            compress,
            notakeover,
        )

    def _pre_start(self, request: BaseRequest) -> tuple[str | None, WebSocketWriter]:
        self._loop = request._loop

        headers, protocol, compress, notakeover = self._handshake(request)

        self.set_status(101)
        self.headers.update(headers)
        self.force_close()
        self._compress = compress
        transport = request._protocol.transport
        if transport is None:
            raise ConnectionResetError("Connection lost")
        writer = WebSocketWriter(
            request._protocol,
            transport,
            compress=compress,
            notakeover=notakeover,
            limit=self._writer_limit,
        )

        return protocol, writer

    def _post_start(
        self, request: BaseRequest, protocol: str | None, writer: WebSocketWriter
    ) -> None:
        self._ws_protocol = protocol
        self._writer = writer

        self._reset_heartbeat()

        loop = self._loop
        assert loop is not None
        self._reader = WebSocketDataQueue(
            request._protocol, DEFAULT_CHUNK_SIZE, loop=loop
        )
        # Owns the parser so a stalled reader parked on the queue by weakref
        # stays alive while this response can still be drained.
        self._parser = WebSocketReader(
            self._reader,
            self._max_msg_size,
            compress=bool(self._compress),
            decode_text=self._decode_text,
        )
        cb = None if self._heartbeat is None else self._on_data_received
        request.protocol.set_parser(self._parser, data_received_cb=cb)
        # disable HTTP keepalive for WebSocket
        request.protocol.keep_alive(False)

    def can_prepare(self, request: BaseRequest) -> WebSocketReady:
        if self._writer is not None:
            raise RuntimeError("Already started")
        try:
            _, protocol, _, _ = self._handshake(request)
        except HTTPException:
            return WebSocketReady(False, None)
        else:
            return WebSocketReady(True, protocol)

    @property
    def prepared(self) -> bool:
        return self._writer is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_code(self) -> int | None:
        return self._close_code

    @property
    def ws_protocol(self) -> str | None:
        return self._ws_protocol

    @property
    def compress(self) -> int | bool:
        return self._compress

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get optional transport information.

        If no value associated with ``name`` is found, ``default`` is returned.
        """
        writer = self._writer
        if writer is None:
            return default
        transport = writer.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def exception(self) -> BaseException | None:
        return self._exception

    async def ping(self, message: bytes = b"") -> None:
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")
        await self._writer.send_frame(message, WSMsgType.PING)

    async def pong(self, message: bytes = b"") -> None:
        # unsolicited pong
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")
        await self._writer.send_frame(message, WSMsgType.PONG)

    async def send_frame(
        self, message: bytes, opcode: WSMsgType, compress: int | None = None
    ) -> None:
        """Send a frame over the websocket."""
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")
        await self._writer.send_frame(message, opcode, compress)

    async def send_str(self, data: str, compress: int | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")
        if not isinstance(data, str):
            raise TypeError("data argument must be str (%r)" % type(data))
        await self._writer.send_frame(
            data.encode("utf-8"), WSMsgType.TEXT, compress=compress
        )

    async def send_bytes(self, data: bytes, compress: int | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data argument must be byte-ish (%r)" % type(data))
        await self._writer.send_frame(data, WSMsgType.BINARY, compress=compress)

    async def send_json(
        self,
        data: Any,
        compress: int | None = None,
        *,
        dumps: JSONEncoder = json.dumps,
    ) -> None:
        await self.send_str(dumps(data), compress=compress)

    async def send_json_bytes(
        self,
        data: Any,
        compress: int | None = None,
        *,
        dumps: JSONBytesEncoder,
    ) -> None:
        """Send JSON data using a bytes-returning encoder as a binary frame.

        Use this when your JSON encoder (like orjson) returns bytes
        instead of str, avoiding the encode/decode overhead.
        """
        await self.send_bytes(dumps(data), compress=compress)

    async def write_eof(self) -> None:  # type: ignore[override]
        if self._eof_sent:
            return
        if self._payload_writer is None:
            raise RuntimeError("Response has not been started")

        await self.close()
        self._eof_sent = True

    async def close(
        self, *, code: int = WSCloseCode.OK, message: bytes = b"", drain: bool = True
    ) -> bool:
        """Close websocket connection."""
        if self._writer is None:
            raise RuntimeError("Call .prepare() first")

        if self._closed:
            return False
        self._set_closed()

        try:
            try:
                await self._writer.close(code, message)
                writer = self._payload_writer
                assert writer is not None
                if drain:
                    await writer.drain()
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
                raise
            except Exception as exc:
                self._exception = exc
                self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
                return True

            reader = self._reader
            assert reader is not None
            # we need to break `receive()` cycle before we can call
            # `reader.read()` as `close()` may be called from different task
            if self._waiting:
                assert self._loop is not None
                assert self._close_wait is None
                self._close_wait = self._loop.create_future()
                reader.feed_data(WS_CLOSING_MESSAGE, 0)
                await self._close_wait

            if self._closing:
                self._close_transport()
                return True

            try:
                async with async_timeout.timeout(self._timeout):
                    while True:
                        msg = await reader.read()
                        if msg.type is WSMsgType.CLOSE:
                            self._set_code_close_transport(msg.data)
                            return True
            except asyncio.CancelledError:
                self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
                raise
            except Exception as exc:
                self._exception = exc
                self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
                return True
        finally:
            # Once closed the response can no longer be drained; release the
            # parser and the stash it retains.
            self._parser = None

    def _set_closing(self, code: WSCloseCode) -> None:
        """Set the close code and mark the connection as closing."""
        self._closing = True
        self._close_code = code
        self._cancel_heartbeat()

    def _set_code_close_transport(self, code: WSCloseCode) -> None:
        """Set the close code and close the transport."""
        self._close_code = code
        self._close_transport()

    def _close_transport(self) -> None:
        """Close the transport."""
        if self._req is not None and self._req.transport is not None:
            self._req.transport.close()

    @overload
    async def receive(
        self: "WebSocketResponse[Literal[True]]", timeout: float | None = None
    ) -> WSMessageDecodeText: ...

    @overload
    async def receive(
        self: "WebSocketResponse[Literal[False]]", timeout: float | None = None
    ) -> WSMessageNoDecodeText: ...

    @overload
    async def receive(
        self: "WebSocketResponse[_DecodeText]", timeout: float | None = None
    ) -> WSMessageDecodeText | WSMessageNoDecodeText: ...

    async def receive(
        self, timeout: float | None = None
    ) -> WSMessageDecodeText | WSMessageNoDecodeText:
        if self._reader is None:
            raise RuntimeError("Call .prepare() first")

        receive_timeout = timeout or self._receive_timeout
        while True:
            if self._waiting:
                raise RuntimeError("Concurrent call to receive() is not allowed")

            if self._closed:
                self._conn_lost += 1
                if self._conn_lost >= THRESHOLD_CONNLOST_ACCESS:
                    raise RuntimeError("WebSocket connection is closed.")
                return WS_CLOSED_MESSAGE
            elif self._closing:
                return WS_CLOSING_MESSAGE

            try:
                self._waiting = True
                try:
                    if receive_timeout:
                        # Entering the context manager and creating
                        # Timeout() object can take almost 50% of the
                        # run time in this loop so we avoid it if
                        # there is no read timeout.
                        async with async_timeout.timeout(receive_timeout):
                            msg = await self._reader.read()
                    else:
                        msg = await self._reader.read()
                finally:
                    self._waiting = False
                    if self._close_wait:
                        set_result(self._close_wait, None)
            except asyncio.TimeoutError:
                raise
            except EofStream:
                self._close_code = WSCloseCode.OK
                await self.close()
                return WSMessage(WSMsgType.CLOSED, None, None)
            except WebSocketError as exc:
                self._close_code = exc.code
                await self.close(code=exc.code)
                return WSMessage(WSMsgType.ERROR, exc, None)
            except Exception as exc:
                self._exception = exc
                self._set_closing(WSCloseCode.ABNORMAL_CLOSURE)
                await self.close()
                return WSMessage(WSMsgType.ERROR, exc, None)

            if msg.type not in _INTERNAL_RECEIVE_TYPES:
                # If its not a close/closing/ping/pong message
                # we can return it immediately
                return msg

            if msg.type is WSMsgType.CLOSE:
                self._set_closing(msg.data)
                # Could be closed while awaiting reader.
                if not self._closed and self._autoclose:
                    # The client is likely going to close the
                    # connection out from under us so we do not
                    # want to drain any pending writes as it will
                    # likely result writing to a broken pipe.
                    await self.close(drain=False)
            elif msg.type is WSMsgType.CLOSING:
                self._set_closing(WSCloseCode.OK)
            elif msg.type is WSMsgType.PING and self._autoping:
                await self.pong(msg.data)
                continue
            elif msg.type is WSMsgType.PONG and self._autoping:
                continue

            return msg

    @overload
    async def receive_str(
        self: "WebSocketResponse[Literal[True]]", *, timeout: float | None = None
    ) -> str: ...

    @overload
    async def receive_str(
        self: "WebSocketResponse[Literal[False]]", *, timeout: float | None = None
    ) -> bytes: ...

    @overload
    async def receive_str(
        self: "WebSocketResponse[_DecodeText]", *, timeout: float | None = None
    ) -> str | bytes: ...

    async def receive_str(self, *, timeout: float | None = None) -> str | bytes:
        """Receive TEXT message.

        Returns str when decode_text=True (default), bytes when decode_text=False.
        """
        msg = await self.receive(timeout)
        if msg.type is not WSMsgType.TEXT:
            raise WSMessageTypeError(
                f"Received message {msg.type}:{msg.data!r} is not WSMsgType.TEXT"
            )
        return cast(str, msg.data)

    async def receive_bytes(self, *, timeout: float | None = None) -> bytes:
        msg = await self.receive(timeout)
        if msg.type is not WSMsgType.BINARY:
            raise WSMessageTypeError(
                f"Received message {msg.type}:{msg.data!r} is not WSMsgType.BINARY"
            )
        return cast(bytes, msg.data)

    @overload
    async def receive_json(
        self: "WebSocketResponse[Literal[True]]",
        *,
        loads: JSONDecoder = ...,
        timeout: float | None = None,
    ) -> Any: ...

    @overload
    async def receive_json(
        self: "WebSocketResponse[Literal[False]]",
        *,
        loads: Callable[[bytes], Any] = ...,
        timeout: float | None = None,
    ) -> Any: ...

    @overload
    async def receive_json(
        self: "WebSocketResponse[_DecodeText]",
        *,
        loads: JSONDecoder | Callable[[bytes], Any] = ...,
        timeout: float | None = None,
    ) -> Any: ...

    async def receive_json(
        self,
        *,
        loads: JSONDecoder | Callable[[bytes], Any] = json.loads,
        timeout: float | None = None,
    ) -> Any:
        data = await self.receive_str(timeout=timeout)
        return loads(data)  # type: ignore[arg-type]

    async def write(self, data: Buffer) -> None:
        raise RuntimeError("Cannot call .write() for websocket")

    def __aiter__(self) -> Self:
        return self

    @overload
    async def __anext__(
        self: "WebSocketResponse[Literal[True]]",
    ) -> WSMessageDecodeText: ...

    @overload
    async def __anext__(
        self: "WebSocketResponse[Literal[False]]",
    ) -> WSMessageNoDecodeText: ...

    @overload
    async def __anext__(
        self: "WebSocketResponse[_DecodeText]",
    ) -> WSMessageDecodeText | WSMessageNoDecodeText: ...

    async def __anext__(self) -> WSMessageDecodeText | WSMessageNoDecodeText:
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    def _cancel(self, exc: BaseException) -> None:
        # web_protocol calls this from connection_lost
        # or when the server is shutting down.
        self._closing = True
        self._cancel_heartbeat()
        if self._reader is not None:
            set_exception(self._reader, exc)
<|endoftext|>"""HTTP Client for asyncio."""

import asyncio
import base64
import hashlib
import json
import os
import sys
import traceback
import warnings
from collections.abc import (
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Sequence,
)
from contextlib import suppress
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    overload,
)

import attr
from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from yarl import URL

from . import hdrs, http, payload
from ._websocket.reader import WebSocketDataQueue
from .abc import AbstractCookieJar
from .client_exceptions import (
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
    ClientResponseError,
    ClientSSLError,
    ConnectionTimeoutError,
    ContentTypeError,
    InvalidURL,
    InvalidUrlClientError,
    InvalidUrlRedirectClientError,
    NonHttpUrlClientError,
    NonHttpUrlRedirectClientError,
    RedirectClientError,
    ServerConnectionError,
    ServerDisconnectedError,
    ServerFingerprintMismatch,
    ServerTimeoutError,
    SocketTimeoutError,
    TooManyRedirects,
    UploadAbortedError,
    WSMessageTypeError,
    WSServerHandshakeError,
)
from .client_middlewares import ClientMiddlewareType, build_client_middlewares
from .client_reqrep import (
    ClientRequest as ClientRequest,
    ClientResponse as ClientResponse,
    Fingerprint as Fingerprint,
    RequestInfo as RequestInfo,
    UploadTracker,
    _merge_ssl_params,
)
from .client_ws import (
    DEFAULT_WS_CLIENT_TIMEOUT,
    ClientWebSocketResponse as ClientWebSocketResponse,
    ClientWSTimeout as ClientWSTimeout,
)
from .connector import (
    HTTP_AND_EMPTY_SCHEMA_SET,
    BaseConnector as BaseConnector,
    NamedPipeConnector as NamedPipeConnector,
    TCPConnector as TCPConnector,
    UnixConnector as UnixConnector,
)
from .cookiejar import CookieJar, _UnlimitedCookieJar
from .helpers import (
    _SENTINEL,
    DEBUG,
    DEFAULT_CHUNK_SIZE,
    EMPTY_BODY_METHODS,
    BasicAuth,
    TimeoutHandle,
    basicauth_from_netrc,
    get_env_proxy_for_url,
    netrc_from_env,
    sentinel,
    strip_auth_from_url,
)
from .http import WS_KEY, HttpVersion, WebSocketReader, WebSocketWriter
from .http_websocket import WSHandshakeError, ws_ext_gen, ws_ext_parse
from .tracing import Trace, TraceConfig
from .typedefs import (
    JSONBytesEncoder,
    JSONEncoder,
    LooseCookies,
    LooseHeaders,
    Query,
    StrOrURL,
)

__all__ = (
    # client_exceptions
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
    "ClientResponseError",
    "ClientSSLError",
    "ConnectionTimeoutError",
    "ContentTypeError",
    "InvalidURL",
    "InvalidUrlClientError",
    "RedirectClientError",
    "NonHttpUrlClientError",
    "InvalidUrlRedirectClientError",
    "NonHttpUrlRedirectClientError",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ServerTimeoutError",
    "SocketTimeoutError",
    "TooManyRedirects",
    "UploadAbortedError",
    "WSServerHandshakeError",
    # client_reqrep
    "ClientRequest",
    "ClientResponse",
    "Fingerprint",
    "RequestInfo",
    "UploadTracker",
    # connector
    "BaseConnector",
    "TCPConnector",
    "UnixConnector",
    "NamedPipeConnector",
    # client_ws
    "ClientWebSocketResponse",
    # client
    "ClientSession",
    "ClientTimeout",
    "ClientWSTimeout",
    "request",
    "WSMessageTypeError",
)


if TYPE_CHECKING:
    from ssl import SSLContext
else:
    SSLContext = Any

if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack


# URL parsers strip leading and trailing C0 control characters and spaces and
# drop tabs and newlines before splitting a URL.
_C0_CONTROL_OR_SPACE = "".join(map(chr, range(0x21)))
_REMOVE_TAB_OR_NEWLINE = str.maketrans("", "", "\t\n\r")


def _has_no_authority(location: str, scheme: str) -> bool:
    """Tell if a URL with a scheme has no "//" after the scheme.

    Browsers read a backslash like a slash there for http and https.
    """
    location = location.strip(_C0_CONTROL_OR_SPACE).translate(_REMOVE_TAB_OR_NEWLINE)
    rest = location[len(scheme) + 1 : len(scheme) + 3]
    return rest[:1] not in ("/", "\\") or rest[1:] not in ("/", "\\")


class _RequestOptions(TypedDict, total=False):
    params: Query
    data: Any
    json: Any
    cookies: LooseCookies | None
    headers: LooseHeaders | None
    skip_auto_headers: Iterable[str] | None
    auth: BasicAuth | None
    allow_redirects: bool
    max_redirects: int
    compress: str | bool | None
    chunked: bool | None
    expect100: bool
    raise_for_status: None | bool | Callable[[ClientResponse], Awaitable[None]]
    read_until_eof: bool
    proxy: StrOrURL | None
    proxy_auth: BasicAuth | None
    timeout: "ClientTimeout | _SENTINEL | None"
    ssl: SSLContext | bool | Fingerprint
    server_hostname: str | None
    proxy_headers: LooseHeaders | None
    trace_request_ctx: object
    read_bufsize: int | None
    auto_decompress: bool | None
    max_line_size: int | None
    max_field_size: int | None
    max_headers: int | None
    middlewares: Sequence[ClientMiddlewareType] | None
    upload_tracker: UploadTracker | None


class _WSConnectOptions(TypedDict, total=False):
    method: str
    protocols: Iterable[str]
    timeout: "ClientWSTimeout | _SENTINEL"
    receive_timeout: float | None
    autoclose: bool
    autoping: bool
    heartbeat: float | None
    auth: BasicAuth | None
    origin: str | None
    params: Query
    headers: LooseHeaders | None
    proxy: StrOrURL | None
    proxy_auth: BasicAuth | None
    ssl: SSLContext | bool | Fingerprint
    verify_ssl: bool | None
    fingerprint: bytes | None
    ssl_context: SSLContext | None
    server_hostname: str | None
    proxy_headers: LooseHeaders | None
    compress: int
    max_msg_size: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ClientTimeout:
    total: float | None = None
    connect: float | None = None
    sock_read: float | None = None
    sock_connect: float | None = None
    ceil_threshold: float = 5

    # pool_queue_timeout: Optional[float] = None
    # dns_resolution_timeout: Optional[float] = None
    # socket_connect_timeout: Optional[float] = None
    # connection_acquiring_timeout: Optional[float] = None
    # new_connection_timeout: Optional[float] = None
    # http_header_timeout: Optional[float] = None
    # response_body_timeout: Optional[float] = None

    # to create a timeout specific for a single request, either
    # - create a completely new one to overwrite the default
    # - or use http://www.attrs.org/en/stable/api.html#attr.evolve
    # to overwrite the defaults


# 5 Minute default read timeout
DEFAULT_TIMEOUT: Final[ClientTimeout] = ClientTimeout(total=5 * 60, sock_connect=30)

# https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"})

_RetType_co = TypeVar(
    "_RetType_co",
    bound="ClientResponse | ClientWebSocketResponse[bool]",
    covariant=True,
)
_CharsetResolver = Callable[[ClientResponse, bytes], str]


class ClientSession:
    """First-class interface for making HTTP requests."""

    ATTRS = frozenset(
        [
            "_base_url",
            "_base_url_origin",
            "_source_traceback",
            "_connector",
            "_loop",
            "_cookie_jar",
            "_connector_owner",
            "_default_auth",
            "_version",
            "_json_serialize",
            "_json_serialize_bytes",
            "_requote_redirect_url",
            "_timeout",
            "_raise_for_status",
            "_auto_decompress",
            "_trust_env",
            "_default_headers",
            "_skip_auto_headers",
            "_request_class",
            "_response_class",
            "_ws_response_class",
            "_trace_configs",
            "_read_bufsize",
            "_max_line_size",
            "_max_field_size",
            "_max_headers",
            "_resolve_charset",
            "_default_proxy",
            "_default_proxy_auth",
            "_retry_connection",
            "_middlewares",
            "requote_redirect_url",
        ]
    )

    _source_traceback: traceback.StackSummary | None = None
    _connector: BaseConnector | None = None

    def __init__(
        self,
        base_url: StrOrURL | None = None,
        *,
        connector: BaseConnector | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        cookies: LooseCookies | None = None,
        headers: LooseHeaders | None = None,
        proxy: StrOrURL | None = None,
        proxy_auth: BasicAuth | None = None,
        skip_auto_headers: Iterable[str] | None = None,
        auth: BasicAuth | None = None,
        json_serialize: JSONEncoder = json.dumps,
        json_serialize_bytes: JSONBytesEncoder | None = None,
        request_class: type[ClientRequest] = ClientRequest,
        response_class: type[ClientResponse] = ClientResponse,
        ws_response_class: type[ClientWebSocketResponse] = ClientWebSocketResponse,
        version: HttpVersion = http.HttpVersion11,
        cookie_jar: AbstractCookieJar | None = None,
        connector_owner: bool = True,
        raise_for_status: bool | Callable[[ClientResponse], Awaitable[None]] = False,
        read_timeout: float | _SENTINEL = sentinel,
        conn_timeout: float | None = None,
        timeout: object | ClientTimeout = sentinel,
        auto_decompress: bool = True,
        trust_env: bool = False,
        requote_redirect_url: bool = True,
        trace_configs: list[TraceConfig] | None = None,
        read_bufsize: int = DEFAULT_CHUNK_SIZE,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        max_headers: int = 128,
        fallback_charset_resolver: _CharsetResolver = lambda r, b: "utf-8",
        middlewares: Sequence[ClientMiddlewareType] = (),
        ssl_shutdown_timeout: _SENTINEL | None | float = sentinel,
    ) -> None:
        # We initialise _connector to None immediately, as it's referenced in __del__()
        # and could cause issues if an exception occurs during initialisation.
        self._connector: BaseConnector | None = None

        if loop is None:
            if connector is not None:
                loop = connector._loop

        loop = loop or asyncio.get_running_loop()

        if base_url is None or isinstance(base_url, URL):
            self._base_url: URL | None = base_url
            self._base_url_origin = None if base_url is None else base_url.origin()
        else:
            self._base_url = URL(base_url)
            self._base_url_origin = self._base_url.origin()
            assert self._base_url.absolute, "Only absolute URLs are supported"
        if self._base_url is not None and not self._base_url.path.endswith("/"):
            raise ValueError("base_url must have a trailing '/'")

        if timeout is sentinel or timeout is None:
            self._timeout = DEFAULT_TIMEOUT
            if read_timeout is not sentinel:
                warnings.warn(
                    "read_timeout is deprecated, use timeout argument instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._timeout = attr.evolve(self._timeout, total=read_timeout)
            if conn_timeout is not None:
                self._timeout = attr.evolve(self._timeout, connect=conn_timeout)
                warnings.warn(
                    "conn_timeout is deprecated, use timeout argument instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
        else:
            if not isinstance(timeout, ClientTimeout):
                raise ValueError(
                    f"timeout parameter cannot be of {type(timeout)} type, "
                    "please use 'timeout=ClientTimeout(...)'",
                )
            self._timeout = timeout
            if read_timeout is not sentinel:
                raise ValueError(
                    "read_timeout and timeout parameters "
                    "conflict, please setup "
                    "timeout.read"
                )
            if conn_timeout is not None:
                raise ValueError(
                    "conn_timeout and timeout parameters "
                    "conflict, please setup "
                    "timeout.connect"
                )

        if ssl_shutdown_timeout is not sentinel:
            warnings.warn(
                "The ssl_shutdown_timeout parameter is deprecated and will be removed in aiohttp 4.0",
                DeprecationWarning,
                stacklevel=2,
            )

        if connector is None:
            connector = TCPConnector(
                loop=loop, ssl_shutdown_timeout=ssl_shutdown_timeout
            )

        if connector._loop is not loop:
            raise RuntimeError("Session and connector has to use same event loop")

        self._loop = loop

        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

        if cookie_jar is None:
            cookie_jar = CookieJar(loop=loop)
        self._cookie_jar = cookie_jar

        if cookies:
            self._cookie_jar.update_cookies(cookies)

        if auth is not None:
            warnings.warn(
                "The 'auth' parameter is deprecated and will be removed in v4;"
                " pass headers={'Authorization': "
                "aiohttp.encode_basic_auth(login, password)} instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if proxy_auth is not None:
            warnings.warn(
                "The 'proxy_auth' parameter is deprecated and will be removed in v4;"
                " pass proxy_headers={'Proxy-Authorization': "
                "aiohttp.encode_basic_auth(login, password)} instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self._connector = connector
        self._connector_owner = connector_owner
        self._default_auth = auth
        self._version = version
        self._json_serialize = json_serialize
        self._json_serialize_bytes = json_serialize_bytes
        self._raise_for_status = raise_for_status
        self._auto_decompress = auto_decompress
        self._trust_env = trust_env
        self._requote_redirect_url = requote_redirect_url
        self._read_bufsize = read_bufsize
        self._max_line_size = max_line_size
        self._max_field_size = max_field_size
        self._max_headers = max_headers

        # Convert to list of tuples
        if headers:
            real_headers: CIMultiDict[str] = CIMultiDict(headers)
        else:
            real_headers = CIMultiDict()
        self._default_headers: CIMultiDict[str] = real_headers
        if skip_auto_headers is not None:
            self._skip_auto_headers = frozenset(istr(i) for i in skip_auto_headers)
        else:
            self._skip_auto_headers = frozenset()

        self._request_class = request_class
        self._response_class = response_class
        self._ws_response_class = ws_response_class

        self._trace_configs = trace_configs or []
        for trace_config in self._trace_configs:
            trace_config.freeze()

        self._resolve_charset = fallback_charset_resolver

        self._default_proxy = proxy
        self._default_proxy_auth = proxy_auth
        self._retry_connection: bool = True
        self._middlewares = middlewares

    def __init_subclass__(cls: type["ClientSession"]) -> None:
        warnings.warn(
            f"Inheritance class {cls.__name__} from ClientSession is discouraged",
            DeprecationWarning,
            stacklevel=2,
        )

    if DEBUG:

        def __setattr__(self, name: str, val: Any) -> None:
            if name not in self.ATTRS:
                warnings.warn(
                    f"Setting custom ClientSession.{name} attribute is discouraged",
                    DeprecationWarning,
                    stacklevel=2,
                )
            super().__setattr__(name, val)

    def __del__(self, _warnings: Any = warnings) -> None:
        if not self.closed:
            kwargs = {"source": self}
            _warnings.warn(
                f"Unclosed client session {self!r}", ResourceWarning, **kwargs
            )
            context = {"client_session": self, "message": "Unclosed client session"}
            if self._source_traceback is not None:
                context["source_traceback"] = self._source_traceback
            self._loop.call_exception_handler(context)

    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        def request(
            self,
            method: str,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

    else:

        def request(
            self, method: str, url: StrOrURL, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP request."""
            return _RequestContextManager(self._request(method, url, **kwargs))

    def _build_url(self, str_or_url: StrOrURL) -> URL:
        url = URL(str_or_url)
        if self._base_url and not url.absolute:
            return self._base_url.join(url)
        return url

    async def _request(
        self,
        method: str,
        str_or_url: StrOrURL,
        *,
        params: Query = None,
        data: Any = None,
        json: Any = None,
        cookies: LooseCookies | None = None,
        headers: LooseHeaders | None = None,
        skip_auto_headers: Iterable[str] | None = None,
        auth: BasicAuth | None = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        compress: str | bool | None = None,
        chunked: bool | None = None,
        expect100: bool = False,
        raise_for_status: (
            None | bool | Callable[[ClientResponse], Awaitable[None]]
        ) = None,
        read_until_eof: bool = True,
        proxy: StrOrURL | None = None,
        proxy_auth: BasicAuth | None = None,
        timeout: ClientTimeout | _SENTINEL = sentinel,
        verify_ssl: bool | None = None,
        fingerprint: bytes | None = None,
        ssl_context: SSLContext | None = None,
        ssl: SSLContext | bool | Fingerprint = True,
        server_hostname: str | None = None,
        proxy_headers: LooseHeaders | None = None,
        trace_request_ctx: object = None,
        read_bufsize: int | None = None,
        auto_decompress: bool | None = None,
        max_line_size: int | None = None,
        max_field_size: int | None = None,
        max_headers: int | None = None,
        middlewares: Sequence[ClientMiddlewareType] | None = None,
        upload_tracker: UploadTracker | None = None,
    ) -> ClientResponse:

        # NOTE: timeout clamps existing connect and read timeouts.  We cannot
        # set the default to None because we need to detect if the user wants
        # to use the existing timeouts by setting timeout to None.

        # Bound outside the settle-guaranteeing try block: a rebind error
        # must not settle a tracker owned by another request.
        if upload_tracker is not None:
            upload_tracker._bind()

        tm: TimeoutHandle | None = None
        handle: asyncio.TimerHandle | None = None
        # Only traces that saw send_request_start; they must also see a terminal event.
        traces: list[Trace] = []
        req: ClientRequest | None = None
        resp: ClientResponse | None = None
        try:
            if self.closed:
                raise RuntimeError("Session is closed")

            method = method.upper()

            ssl = _merge_ssl_params(ssl, verify_ssl, ssl_context, fingerprint)

            if auth is not None:
                warnings.warn(
                    "The 'auth' parameter is deprecated and will be removed in v4;"
                    " pass headers={'Authorization': "
                    "aiohttp.encode_basic_auth(login, password)} instead",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if proxy_auth is not None:
                warnings.warn(
                    "The 'proxy_auth' parameter is deprecated and will be removed in v4;"
                    " pass proxy_headers={'Proxy-Authorization': "
                    "aiohttp.encode_basic_auth(login, password)} instead",
                    DeprecationWarning,
                    stacklevel=3,
                )

            if data is not None and json is not None:
                raise ValueError(
                    "data and json parameters can not be used at the same time"
                )
            elif json is not None:
                if self._json_serialize_bytes is not None:
                    data = payload.JsonBytesPayload(
                        json, dumps=self._json_serialize_bytes
                    )
                else:
                    data = payload.JsonPayload(json, dumps=self._json_serialize)

            if not isinstance(chunked, bool) and chunked is not None:
                warnings.warn("Chunk size is deprecated #1615", DeprecationWarning)

            redirects = 0
            history: list[ClientResponse] = []
            version = self._version
            params = params or {}

            # Merge with default headers and transform to CIMultiDict
            headers = self._prepare_headers(headers)

            try:
                url = self._build_url(str_or_url)
            except ValueError as e:
                raise InvalidUrlClientError(str_or_url) from e

            assert self._connector is not None
            if url.scheme not in self._connector.allowed_protocol_schema_set:
                raise NonHttpUrlClientError(url)

            skip_headers: Iterable[istr] | None
            if skip_auto_headers is not None:
                skip_headers = {
                    istr(i) for i in skip_auto_headers
                } | self._skip_auto_headers
            elif self._skip_auto_headers:
                skip_headers = self._skip_auto_headers
            else:
                skip_headers = None

            if proxy is None:
                proxy = self._default_proxy
            if proxy_auth is None:
                proxy_auth = self._default_proxy_auth

            if proxy is None:
                proxy_headers = None
            else:
                proxy_headers = self._prepare_headers(proxy_headers)
                try:
                    proxy = URL(proxy)
                except ValueError as e:
                    raise InvalidURL(proxy) from e

            if timeout is sentinel:
                real_timeout: ClientTimeout = self._timeout
            else:
                if not isinstance(timeout, ClientTimeout):
                    real_timeout = ClientTimeout(total=timeout)
                else:
                    real_timeout = timeout
            # timeout is cumulative for all request operations
            # (request, redirects, responses, data consuming)
            tm = TimeoutHandle(
                self._loop,
                real_timeout.total,
                ceil_threshold=real_timeout.ceil_threshold,
            )
            handle = tm.start()

            if read_bufsize is None:
                read_bufsize = self._read_bufsize

            if auto_decompress is None:
                auto_decompress = self._auto_decompress

            if max_line_size is None:
                max_line_size = self._max_line_size

            if max_field_size is None:
                max_field_size = self._max_field_size

            if max_headers is None:
                max_headers = self._max_headers

            for trace_config in self._trace_configs:
                trace = Trace(
                    self,
                    trace_config,
                    trace_config.trace_config_ctx(trace_request_ctx=trace_request_ctx),
                )
                await trace.send_request_start(
                    method, url.update_query(params), headers
                )
                traces.append(trace)

            timer = tm.timer()
            with timer:
                # https://www.rfc-editor.org/rfc/rfc9112.html#name-retrying-requests
                retry_persistent_connection = (
                    self._retry_connection and method in IDEMPOTENT_METHODS
                )
                while True:
                    url, auth_from_url = strip_auth_from_url(url)
                    if not url.raw_host:
                        # NOTE: Bail early, otherwise, causes `InvalidURL` through
                        # NOTE: `self._request_class()` below.
                        err_exc_cls = (
                            InvalidUrlRedirectClientError
                            if redirects
                            else InvalidUrlClientError
                        )
                        raise err_exc_cls(url)
                    # If `auth` was passed for an already authenticated URL,
                    # disallow only if this is the initial URL; this is to avoid issues
                    # with sketchy redirects that are not the caller's responsibility
                    if not history and (auth and auth_from_url):
                        raise ValueError(
                            "Cannot combine AUTH argument with "
                            "credentials encoded in URL"
                        )

                    # Override the auth with the one from the URL only if we
                    # have no auth, or if we got an auth from a redirect URL
                    if auth is None or (history and auth_from_url is not None):
                        auth = auth_from_url

                    if (
                        auth is None
                        and self._default_auth
                        and (
                            not self._base_url or self._base_url_origin == url.origin()
                        )
                    ):
                        auth = self._default_auth

                    # Try netrc if auth is still None and trust_env is enabled.
                    if auth is None and self._trust_env and url.host is not None:
                        auth = await self._loop.run_in_executor(
                            None, self._get_netrc_auth, url.host
                        )

                    # It would be confusing if we support explicit
                    # Authorization header with auth argument
                    if (
                        headers is not None
                        and auth is not None
                        and hdrs.AUTHORIZATION in headers
                    ):
                        raise ValueError(
                            "Cannot combine AUTHORIZATION header "
                            "with AUTH argument or credentials "
                            "encoded in URL"
                        )

                    all_cookies = self._cookie_jar.filter_cookies(url)

                    if cookies is not None:
                        tmp_cookie_jar = _UnlimitedCookieJar(
                            unsafe=self._cookie_jar.unsafe,
                            quote_cookie=self._cookie_jar.quote_cookie,
                            treat_as_secure_origin=frozenset(
                                self._cookie_jar.treat_as_secure_origin
                            ),
                        )
                        tmp_cookie_jar.update_cookies(cookies)
                        req_cookies = tmp_cookie_jar.filter_cookies(url)
                        if req_cookies:
                            all_cookies.load(req_cookies)

                    proxy_: URL | None = None
                    if proxy is not None:
                        proxy_ = URL(proxy)
                    elif self._trust_env:
                        with suppress(LookupError):
                            proxy_, proxy_auth = await asyncio.to_thread(
                                get_env_proxy_for_url, url
                            )

                    req = self._request_class(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        skip_auto_headers=skip_headers,
                        data=data,
                        cookies=all_cookies,
                        auth=auth,
                        version=version,
                        compress=compress,
                        chunked=chunked,
                        expect100=expect100,
                        loop=self._loop,
                        response_class=self._response_class,
                        proxy=proxy_,
                        proxy_auth=proxy_auth,
                        timer=timer,
                        session=self,
                        ssl=ssl if ssl is not None else True,
                        server_hostname=server_hostname,
                        proxy_headers=proxy_headers,
                        traces=traces,
                        trust_env=self.trust_env,
                    )
                    req._upload_tracker = upload_tracker

                    async def _connect_and_send_request(
                        req: ClientRequest,
                    ) -> ClientResponse:
                        # connection timeout
                        assert self._connector is not None
                        if (tracker := req._upload_tracker) is not None:
                            req._upload_gen = tracker._attempt_started()
                        try:
                            conn = await self._connector.connect(
                                req, traces=traces, timeout=real_timeout
                            )
                        except asyncio.TimeoutError as exc:
                            raise ConnectionTimeoutError(
                                f"Connection timeout to host {req.url}"
                            ) from exc

                        assert conn.protocol is not None
                        conn.protocol.set_response_params(
                            timer=timer,
                            skip_payload=req.method in EMPTY_BODY_METHODS,
                            read_until_eof=read_until_eof,
                            auto_decompress=auto_decompress,
                            read_timeout=real_timeout.sock_read,
                            read_bufsize=read_bufsize,
                            timeout_ceil_threshold=self._connector._timeout_ceil_threshold,
                            max_line_size=max_line_size,
                            max_field_size=max_field_size,
                            max_headers=max_headers,
                        )
                        try:
                            resp = await req.send(conn)
                            try:
                                await resp.start(conn)
                            except BaseException:
                                resp.close()
                                raise
                        except BaseException:
                            conn.close()
                            raise
                        return resp

                    # Apply middleware (if any) - per-request middleware overrides session middleware
                    effective_middlewares = (
                        self._middlewares if middlewares is None else middlewares
                    )

                    if effective_middlewares:
                        handler = build_client_middlewares(
                            _connect_and_send_request, effective_middlewares
                        )
                    else:
                        handler = _connect_and_send_request

                    try:
                        resp = await handler(req)
                    # Client connector errors should not be retried
                    except (
                        ConnectionTimeoutError,
                        ClientConnectorError,
                        ClientConnectorCertificateError,
                        ClientConnectorSSLError,
                    ):
                        raise
                    except (ClientOSError, ServerDisconnectedError):
                        if not retry_persistent_connection:
                            raise
                        retry_persistent_connection = False
                        if data is not None:
                            # Rebuilding from `data` would resend only the unread
                            # remainder of a file object; reuse the payload, which
                            # rewinds itself once the cancelled writer has settled.
                            await req.close()
                            if req._body is not None and req._body.consumed:
                                raise
                            data = req._body
                        continue
                    except ClientError:
                        raise
                    except OSError as exc:
                        if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                            raise
                        raise ClientOSError(*exc.args) from exc

                    # Update cookies from raw headers to preserve duplicates
                    if resp._raw_cookie_headers:
                        self._cookie_jar.update_cookies_from_headers(
                            resp._raw_cookie_headers, resp.url
                        )

                    # redirects
                    if resp.status in (301, 302, 303, 307, 308) and allow_redirects:

                        for trace in traces:
                            await trace.send_request_redirect(
                                method, url.update_query(params), headers, resp
                            )

                        redirects += 1
                        history.append(resp)
                        if max_redirects and redirects >= max_redirects:
                            if req._body is not None:
                                await req._body.close()
                            resp.close()
                            raise TooManyRedirects(
                                history[0].request_info, tuple(history)
                            )

                        # For 301 and 302, mimic IE, now changed in RFC
                        # https://github.com/kennethreitz/requests/pull/269
                        if (resp.status == 303 and resp.method != hdrs.METH_HEAD) or (
                            resp.status in (301, 302) and resp.method == hdrs.METH_POST
                        ):
                            method = hdrs.METH_GET
                            data = None
                            if headers.get(hdrs.CONTENT_LENGTH):
                                headers.pop(hdrs.CONTENT_LENGTH)
                        else:
                            # For 307/308, always preserve the request body
                            # For 301/302 with non-POST methods, preserve the request body
                            # https://www.rfc-editor.org/rfc/rfc9110#section-15.4.3-3.1
                            # Use the existing payload to avoid recreating it from
                            # a potentially consumed file.
                            #
                            # If the payload is already consumed and cannot be replayed,
                            # fail fast instead of silently sending an empty body.
                            if req._body is not None and req._body.consumed:
                                resp.close()
                                raise ClientPayloadError(
                                    "Cannot follow redirect with a consumed request "
                                    "body. Use bytes, a seekable file-like object, "
                                    "or set allow_redirects=False."
                                )
                            data = req._body

                        r_url = resp.headers.get(hdrs.LOCATION) or resp.headers.get(
                            hdrs.URI
                        )
                        if r_url is None:
                            # see github.com/aio-libs/aiohttp/issues/2022
                            break
                        else:
                            # reading from correct redirection
                            # response is forbidden
                            resp.release()

                        try:
                            parsed_redirect_url = URL(
                                r_url, encoded=not self._requote_redirect_url
                            )
                        except ValueError as e:
                            if req._body is not None:
                                await req._body.close()
                            resp.close()
                            raise InvalidUrlRedirectClientError(
                                r_url,
                                "Server attempted redirecting to a location that does not look like a URL",
                            ) from e

                        scheme = parsed_redirect_url.scheme
                        if scheme not in HTTP_AND_EMPTY_SCHEMA_SET:
                            if req._body is not None:
                                await req._body.close()
                            resp.close()
                            raise NonHttpUrlRedirectClientError(r_url)
                        elif not scheme or (
                            scheme == url.scheme and _has_no_authority(r_url, scheme)
                        ):
                            # "http:/path" or "http:path" is a reference to
                            # the current URL, as browsers resolve it.
                            parsed_redirect_url = url.join(parsed_redirect_url)

                        try:
                            redirect_origin = parsed_redirect_url.origin()
                        except ValueError as origin_val_err:
                            if req._body is not None:
                                await req._body.close()
                            resp.close()
                            raise InvalidUrlRedirectClientError(
                                parsed_redirect_url,
                                "Invalid redirect URL origin",
                            ) from origin_val_err

                        if url.origin() != redirect_origin:
                            auth = None
                            cookies = None
                            headers.popall(hdrs.AUTHORIZATION, None)
                            headers.popall(hdrs.COOKIE, None)
                            headers.popall(hdrs.PROXY_AUTHORIZATION, None)

                        url = parsed_redirect_url
                        params = {}
                        resp.release()
                        continue

                    break

            if req._body is not None:
                await req._body.close()
            # check response status
            if raise_for_status is None:
                raise_for_status = self._raise_for_status

            if raise_for_status is None:
                pass
            elif callable(raise_for_status):
                await raise_for_status(resp)
            elif raise_for_status:
                resp.raise_for_status()

            # register connection
            if handle is not None:
                if resp.connection is not None:
                    resp.connection.add_callback(handle.cancel)
                else:
                    handle.cancel()

            resp._history = tuple(history)

            for trace in traces:
                await trace.send_request_end(
                    method, url.update_query(params), headers, resp
                )
            if upload_tracker is not None:
                upload_tracker._finalize()
            return resp

        except BaseException as e:
            # cleanup timer
            if tm is not None:
                tm.close()
            if handle:
                handle.cancel()

            if upload_tracker is not None:
                upload_tracker._finalize()

            if resp is not None:
                # A failure occurred after the response was received.
                resp.close()

            if req is not None and req._body is not None:
                await req._body.close()

            for trace in traces:
                await trace.send_request_exception(
                    method, url.update_query(params), headers, e  # type: ignore[arg-type]
                )
            raise

    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        @overload
        def ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: Literal[True] = ...,
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "_BaseRequestContextManager[ClientWebSocketResponse[Literal[True]]]": ...

        @overload
        def ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: Literal[False],
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "_BaseRequestContextManager[ClientWebSocketResponse[Literal[False]]]": ...

        @overload
        def ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: bool = ...,
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "_BaseRequestContextManager[ClientWebSocketResponse[bool]]": ...

    def ws_connect(
        self,
        url: StrOrURL,
        *,
        method: str = hdrs.METH_GET,
        protocols: Iterable[str] = (),
        timeout: ClientWSTimeout | _SENTINEL = sentinel,
        receive_timeout: float | None = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: float | None = None,
        auth: BasicAuth | None = None,
        origin: str | None = None,
        params: Query = None,
        headers: LooseHeaders | None = None,
        proxy: StrOrURL | None = None,
        proxy_auth: BasicAuth | None = None,
        ssl: SSLContext | bool | Fingerprint = True,
        verify_ssl: bool | None = None,
        fingerprint: bytes | None = None,
        ssl_context: SSLContext | None = None,
        server_hostname: str | None = None,
        proxy_headers: LooseHeaders | None = None,
        compress: int = 0,
        max_msg_size: int = 4 * 1024 * 1024,
        decode_text: bool = True,
    ) -> "_BaseRequestContextManager[ClientWebSocketResponse[bool]]":
        """Initiate websocket connection."""
        return _WSRequestContextManager(
            self._ws_connect(
                url,
                method=method,
                protocols=protocols,
                timeout=timeout,
                receive_timeout=receive_timeout,
                autoclose=autoclose,
                autoping=autoping,
                heartbeat=heartbeat,
                auth=auth,
                origin=origin,
                params=params,
                headers=headers,
                proxy=proxy,
                proxy_auth=proxy_auth,
                ssl=ssl,
                verify_ssl=verify_ssl,
                fingerprint=fingerprint,
                ssl_context=ssl_context,
                server_hostname=server_hostname,
                proxy_headers=proxy_headers,
                compress=compress,
                max_msg_size=max_msg_size,
                decode_text=decode_text,
            )
        )

    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        @overload
        async def _ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: Literal[True] = ...,
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "ClientWebSocketResponse[Literal[True]]": ...

        @overload
        async def _ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: Literal[False],
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "ClientWebSocketResponse[Literal[False]]": ...

        @overload
        async def _ws_connect(
            self,
            url: StrOrURL,
            *,
            decode_text: bool = ...,
            **kwargs: Unpack[_WSConnectOptions],
        ) -> "ClientWebSocketResponse[bool]": ...

    async def _ws_connect(
        self,
        url: StrOrURL,
        *,
        method: str = hdrs.METH_GET,
        protocols: Iterable[str] = (),
        timeout: ClientWSTimeout | _SENTINEL = sentinel,
        receive_timeout: float | None = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: float | None = None,
        auth: BasicAuth | None = None,
        origin: str | None = None,
        params: Query = None,
        headers: LooseHeaders | None = None,
        proxy: StrOrURL | None = None,
        proxy_auth: BasicAuth | None = None,
        ssl: SSLContext | bool | Fingerprint = True,
        verify_ssl: bool | None = None,
        fingerprint: bytes | None = None,
        ssl_context: SSLContext | None = None,
        server_hostname: str | None = None,
        proxy_headers: LooseHeaders | None = None,
        compress: int = 0,
        max_msg_size: int = 4 * 1024 * 1024,
        decode_text: bool = True,
    ) -> "ClientWebSocketResponse[bool]":
        if auth is not None:
            warnings.warn(
                "The 'auth' parameter is deprecated and will be removed in v4;"
                " pass headers={'Authorization': "
                "aiohttp.encode_basic_auth(login, password)} instead",
                DeprecationWarning,
                stacklevel=3,
            )
        if proxy_auth is not None:
            warnings.warn(
                "The 'proxy_auth' parameter is deprecated and will be removed in v4;"
                " pass proxy_headers={'Proxy-Authorization': "
                "aiohttp.encode_basic_auth(login, password)} instead",
                DeprecationWarning,
                stacklevel=3,
            )
        if timeout is not sentinel:
            if isinstance(timeout, ClientWSTimeout):
                ws_timeout = timeout
            else:
                warnings.warn(
                    "parameter 'timeout' of type 'float' "
                    "is deprecated, please use "
                    "'timeout=ClientWSTimeout(ws_close=...)'",
                    DeprecationWarning,
                    stacklevel=2,
                )
                ws_timeout = ClientWSTimeout(ws_close=timeout)
        else:
            ws_timeout = DEFAULT_WS_CLIENT_TIMEOUT
        if receive_timeout is not None:
            warnings.warn(
                "float parameter 'receive_timeout' "
                "is deprecated, please use parameter "
                "'timeout=ClientWSTimeout(ws_receive=...)'",
                DeprecationWarning,
                stacklevel=2,
            )
            ws_timeout = attr.evolve(ws_timeout, ws_receive=receive_timeout)

        if headers is None:
            real_headers: CIMultiDict[str] = CIMultiDict()
        else:
            real_headers = CIMultiDict(headers)

        default_headers = {
            hdrs.UPGRADE: "websocket",
            hdrs.CONNECTION: "Upgrade",
            hdrs.SEC_WEBSOCKET_VERSION: "13",
        }

        for key, value in default_headers.items():
            real_headers.setdefault(key, value)

        sec_key = base64.b64encode(os.urandom(16))
        real_headers[hdrs.SEC_WEBSOCKET_KEY] = sec_key.decode()

        if protocols:
            real_headers[hdrs.SEC_WEBSOCKET_PROTOCOL] = ",".join(protocols)
        if origin is not None:
            real_headers[hdrs.ORIGIN] = origin
        if compress:
            extstr = ws_ext_gen(compress=compress)
            real_headers[hdrs.SEC_WEBSOCKET_EXTENSIONS] = extstr

        # For the sake of backward compatibility, if user passes in None, convert it to True
        if ssl is None:
            warnings.warn(
                "ssl=None is deprecated, please use ssl=True",
                DeprecationWarning,
                stacklevel=2,
            )
            ssl = True
        ssl = _merge_ssl_params(ssl, verify_ssl, ssl_context, fingerprint)

        # send request
        resp = await self.request(
            method,
            url,
            params=params,
            headers=real_headers,
            read_until_eof=False,
            auth=auth,
            proxy=proxy,
            proxy_auth=proxy_auth,
            ssl=ssl,
            server_hostname=server_hostname,
            proxy_headers=proxy_headers,
        )

        try:
            # check handshake
            if resp.status != 101:
                raise WSServerHandshakeError(
                    resp.request_info,
                    resp.history,
                    message="Invalid response status",
                    status=resp.status,
                    headers=resp.headers,
                )

            if resp.headers.get(hdrs.UPGRADE, "").lower() != "websocket":
                raise WSServerHandshakeError(
                    resp.request_info,
                    resp.history,
                    message="Invalid upgrade header",
                    status=resp.status,
                    headers=resp.headers,
                )

            if resp.headers.get(hdrs.CONNECTION, "").lower() != "upgrade":
                raise WSServerHandshakeError(
                    resp.request_info,
                    resp.history,
                    message="Invalid connection header",
                    status=resp.status,
                    headers=resp.headers,
                )

            # key calculation
            r_key = resp.headers.get(hdrs.SEC_WEBSOCKET_ACCEPT, "")
            match = base64.b64encode(hashlib.sha1(sec_key + WS_KEY).digest()).decode()
            if r_key != match:
                raise WSServerHandshakeError(
                    resp.request_info,
                    resp.history,
                    message="Invalid challenge response",
                    status=resp.status,
                    headers=resp.headers,
                )

            # websocket protocol
            protocol = None
            if protocols and hdrs.SEC_WEBSOCKET_PROTOCOL in resp.headers:
                resp_protocols = [
                    proto.strip()
                    for proto in resp.headers[hdrs.SEC_WEBSOCKET_PROTOCOL].split(",")
                ]

                for proto in resp_protocols:
                    if proto in protocols:
                        protocol = proto
                        break

            # websocket compress
            notakeover = False
            if compress:
                compress_hdrs = resp.headers.get(hdrs.SEC_WEBSOCKET_EXTENSIONS)
                if compress_hdrs:
                    try:
                        compress, notakeover = ws_ext_parse(compress_hdrs)
                    except WSHandshakeError as exc:
                        raise WSServerHandshakeError(
                            resp.request_info,
                            resp.history,
                            message=exc.args[0],
                            status=resp.status,
                            headers=resp.headers,
                        ) from exc
                else:
                    compress = 0
                    notakeover = False

            conn = resp.connection
            assert conn is not None
            conn_proto = conn.protocol
            assert conn_proto is not None

            # For WS connection the read_timeout must be either receive_timeout or greater
            # None == no timeout, i.e. infinite timeout, so None is the max timeout possible
            if ws_timeout.ws_receive is None:
                # Reset regardless
                conn_proto.read_timeout = None
            elif conn_proto.read_timeout is not None:
                conn_proto.read_timeout = max(
                    ws_timeout.ws_receive, conn_proto.read_timeout
                )

            transport = conn.transport
            assert transport is not None
            reader = WebSocketDataQueue(conn_proto, DEFAULT_CHUNK_SIZE, loop=self._loop)
            writer = WebSocketWriter(
                conn_proto,
                transport,
                use_mask=True,
                compress=compress,
                notakeover=notakeover,
            )
        except BaseException:
            resp.close()
            raise
        else:
            ws_resp = self._ws_response_class(
                reader,
                writer,
                protocol,
                resp,
                ws_timeout,
                autoclose,
                autoping,
                self._loop,
                heartbeat=heartbeat,
                compress=compress,
                client_notakeover=notakeover,
            )
            ws_resp._parser = WebSocketReader(
                reader,
                max_msg_size,
                compress=bool(compress),
                decode_text=decode_text,
            )
            cb = None if heartbeat is None else ws_resp._on_data_received
            conn_proto.set_parser(ws_resp._parser, reader, data_received_cb=cb)
            return ws_resp

    def _prepare_headers(self, headers: LooseHeaders | None) -> "CIMultiDict[str]":
        """Add default headers and transform it to CIMultiDict"""
        # Convert headers to MultiDict
        result = CIMultiDict(self._default_headers)
        if headers:
            if not isinstance(headers, (MultiDictProxy, MultiDict)):
                headers = CIMultiDict(headers)
            added_names: set[str] = set()
            for key, value in headers.items():
                if key in added_names:
                    result.add(key, value)
                else:
                    result[key] = value
                    added_names.add(key)
        return result

    def _get_netrc_auth(self, host: str) -> BasicAuth | None:
        """
        Get auth from netrc for the given host.

        This method is designed to be called in an executor to avoid
        blocking I/O in the event loop.
        """
        netrc_obj = netrc_from_env()
        try:
            return basicauth_from_netrc(netrc_obj, host)
        except LookupError:
            return None

    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        def get(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def options(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def head(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def post(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def put(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def patch(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

        def delete(
            self,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...

    else:

        def get(
            self, url: StrOrURL, *, allow_redirects: bool = True, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP GET request."""
            return _RequestContextManager(
                self._request(
                    hdrs.METH_GET, url, allow_redirects=allow_redirects, **kwargs
                )
            )

        def options(
            self, url: StrOrURL, *, allow_redirects: bool = True, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP OPTIONS request."""
            return _RequestContextManager(
                self._request(
                    hdrs.METH_OPTIONS, url, allow_redirects=allow_redirects, **kwargs
                )
            )

        def head(
            self, url: StrOrURL, *, allow_redirects: bool = False, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP HEAD request."""
            return _RequestContextManager(
                self._request(
                    hdrs.METH_HEAD, url, allow_redirects=allow_redirects, **kwargs
                )
            )

        def post(
            self, url: StrOrURL, *, data: Any = None, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP POST request."""
            return _RequestContextManager(
                self._request(hdrs.METH_POST, url, data=data, **kwargs)
            )

        def put(
            self, url: StrOrURL, *, data: Any = None, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP PUT request."""
            return _RequestContextManager(
                self._request(hdrs.METH_PUT, url, data=data, **kwargs)
            )

        def patch(
            self, url: StrOrURL, *, data: Any = None, **kwargs: Any
        ) -> "_RequestContextManager":
            """Perform HTTP PATCH request."""
            return _RequestContextManager(
                self._request(hdrs.METH_PATCH, url, data=data, **kwargs)
            )

        def delete(self, url: StrOrURL, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP DELETE request."""
            return _RequestContextManager(
                self._request(hdrs.METH_DELETE, url, **kwargs)
            )

    async def close(self) -> None:
        """Close underlying connector.

        Release all acquired resources.
        """
        if not self.closed:
            if self._connector is not None and self._connector_owner:
                await self._connector.close()
            self._connector = None

    @property
    def closed(self) -> bool:
        """Is client session closed.

        A readonly property.
        """
        return self._connector is None or self._connector.closed

    @property
    def connector(self) -> BaseConnector | None:
        """Connector instance used for the session."""
        return self._connector

    @property
    def cookie_jar(self) -> AbstractCookieJar:
        """The session cookies."""
        return self._cookie_jar

    @property
    def version(self) -> tuple[int, int]:
        """The session HTTP protocol version."""
        return self._version

    @property
    def requote_redirect_url(self) -> bool:
        """Do URL requoting on redirection handling."""
        return self._requote_redirect_url

    @requote_redirect_url.setter
    def requote_redirect_url(self, val: bool) -> None:
        """Do URL requoting on redirection handling."""
        warnings.warn(
            "session.requote_redirect_url modification is deprecated #2778",
            DeprecationWarning,
            stacklevel=2,
        )
        self._requote_redirect_url = val

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Session's loop."""
        warnings.warn(
            "client.loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        return self._loop

    @property
    def timeout(self) -> ClientTimeout:
        """Timeout for the session."""
        return self._timeout

    @property
    def headers(self) -> "CIMultiDict[str]":
        """The default headers of the client session."""
        return self._default_headers

    @property
    def skip_auto_headers(self) -> frozenset[istr]:
        """Headers for which autogeneration should be skipped"""
        return self._skip_auto_headers

    @property
    def auth(self) -> BasicAuth | None:
        """An object that represents HTTP Basic Authorization"""
        return self._default_auth

    @property
    def json_serialize(self) -> JSONEncoder:
        """Json serializer callable"""
        return self._json_serialize

    @property
    def connector_owner(self) -> bool:
        """Should connector be closed on session closing"""
        return self._connector_owner

    @property
    def raise_for_status(
        self,
    ) -> bool | Callable[[ClientResponse], Awaitable[None]]:
        """Should `ClientResponse.raise_for_status()` be called for each response."""
        return self._raise_for_status

    @property
    def auto_decompress(self) -> bool:
        """Should the body response be automatically decompressed."""
        return self._auto_decompress

    @property
    def trust_env(self) -> bool:
        """
        Should proxies information from environment or netrc be trusted.

        Information is from HTTP_PROXY / HTTPS_PROXY environment variables
        or ~/.netrc file if present.
        """
        return self._trust_env

    @property
    def trace_configs(self) -> list[TraceConfig]:
        """A list of TraceConfig instances used for client tracing"""
        return self._trace_configs

    def detach(self) -> None:
        """Detach connector from session without closing the former.

        Session is switched to closed state anyway.
        """
        self._connector = None

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "ClientSession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()


class _BaseRequestContextManager(
    Coroutine[Any, Any, _RetType_co], Generic[_RetType_co]
):

    __slots__ = ("_coro", "_resp")

    def __init__(self, coro: Coroutine[asyncio.Future[Any], None, _RetType_co]) -> None:
        self._coro: Coroutine[asyncio.Future[Any], None, _RetType_co] = coro

    def send(self, arg: None) -> asyncio.Future[Any]:
        return self._coro.send(arg)

    def throw(self, *args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        return self._coro.throw(*args, **kwargs)

    def close(self) -> None:
        return self._coro.close()

    def __await__(self) -> Generator[Any, None, _RetType_co]:
        ret = self._coro.__await__()
        return ret

    def __iter__(self) -> Generator[Any, None, _RetType_co]:
        return self.__await__()

    async def __aenter__(self) -> _RetType_co:
        self._resp: _RetType_co = await self._coro
        return await self._resp.__aenter__()  # type: ignore[return-value]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._resp.__aexit__(exc_type, exc, tb)


_RequestContextManager = _BaseRequestContextManager[ClientResponse]
_WSRequestContextManager = _BaseRequestContextManager[ClientWebSocketResponse[bool]]


class _SessionRequestContextManager:

    __slots__ = ("_coro", "_resp", "_session")

    def __init__(
        self,
        coro: Coroutine[asyncio.Future[Any], None, ClientResponse],
        session: ClientSession,
    ) -> None:
        self._coro = coro
        self._resp: ClientResponse | None = None
        self._session = session

    async def __aenter__(self) -> ClientResponse:
        try:
            self._resp = await self._coro
        except BaseException:
            await self._session.close()
            raise
        else:
            return self._resp

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._resp is not None
        self._resp.close()
        await self._session.close()


if sys.version_info >= (3, 11) and TYPE_CHECKING:

    def request(
        method: str,
        url: StrOrURL,
        *,
        version: HttpVersion = http.HttpVersion11,
        connector: BaseConnector | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Unpack[_RequestOptions],
    ) -> _SessionRequestContextManager: ...

else:

    def request(
        method: str,
        url: StrOrURL,
        *,
        version: HttpVersion = http.HttpVersion11,
        connector: BaseConnector | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ) -> _SessionRequestContextManager:
        """Constructs and sends a request.

        Returns response object.
        method - HTTP method
        url - request url
        params - (optional) Dictionary or bytes to be sent in the query
        string of the new request
        data - (optional) Dictionary, bytes, or file-like object to
        send in the body of the request
        json - (optional) Any json compatible python object
        headers - (optional) Dictionary of HTTP Headers to send with
        the request
        cookies - (optional) Dict object to send with the request
        auth - (optional) BasicAuth named tuple represent HTTP Basic Auth
        auth - aiohttp.helpers.BasicAuth
        allow_redirects - (optional) If set to False, do not follow
        redirects
        version - Request HTTP version.
        compress - Set to True if request has to be compressed
        with deflate encoding.
        chunked - Set to chunk size for chunked transfer encoding.
        expect100 - Expect 100-continue response from server.
        connector - BaseConnector sub-class instance to support
        connection pooling.
        read_until_eof - Read response until eof if response
        does not have Content-Length header.
        loop - Optional event loop.
        timeout - Optional ClientTimeout settings structure, 5min
        total timeout by default.
        Usage::
        >>> import aiohttp
        >>> async with aiohttp.request('GET', 'http://python.org/') as resp:
        ...    print(resp)
        ...    data = await resp.read()
        <ClientResponse(https://www.python.org/) [200 OK]>
        """
        connector_owner = False
        if connector is None:
            connector_owner = True
            connector = TCPConnector(loop=loop, force_close=True)

        session = ClientSession(
            loop=loop,
            cookies=kwargs.pop("cookies", None),
            version=version,
            timeout=kwargs.pop("timeout", sentinel),
            connector=connector,
            connector_owner=connector_owner,
        )

        return _SessionRequestContextManager(
            session._request(method, url, **kwargs),
            session,
        )
<|endoftext|>"""
Payload implementation for coroutines as data provider.

As a simple case, you can upload data from file::

   @aiohttp.streamer
   async def file_sender(writer, file_name=None):
      with open(file_name, 'rb') as f:
          chunk = f.read(2**16)
          while chunk:
              await writer.write(chunk)

              chunk = f.read(2**16)

Then you can use `file_sender` like this:

    async with session.post('http://httpbin.org/post',
                            data=file_sender(file_name='huge_file')) as resp:
        print(await resp.text())

..note:: Coroutine must accept `writer` as first argument

"""

import types
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from .abc import AbstractStreamWriter
from .payload import Payload, payload_type

__all__ = ("streamer",)


class _stream_wrapper:
    def __init__(
        self,
        coro: Callable[..., Awaitable[None]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.coro = types.coroutine(coro)
        self.args = args
        self.kwargs = kwargs

    async def __call__(self, writer: AbstractStreamWriter) -> None:
        await self.coro(writer, *self.args, **self.kwargs)


class streamer:
    def __init__(self, coro: Callable[..., Awaitable[None]]) -> None:
        warnings.warn(
            "@streamer is deprecated, use async generators instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.coro = coro

    def __call__(self, *args: Any, **kwargs: Any) -> _stream_wrapper:
        return _stream_wrapper(self.coro, args, kwargs)


@payload_type(_stream_wrapper)
class StreamWrapperPayload(Payload):
    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise TypeError("Unable to decode.")


@payload_type(streamer)
class StreamPayload(StreamWrapperPayload):
    def __init__(self, value: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(value(), *args, **kwargs)

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)
<|endoftext|>"""WebSocket protocol versions 13 and 8."""

from ._websocket.helpers import WS_KEY, ws_ext_gen, ws_ext_parse
from ._websocket.models import (
    WS_CLOSED_MESSAGE,
    WS_CLOSING_MESSAGE,
    WebSocketError,
    WSCloseCode,
    WSHandshakeError,
    WSMessage,
    WSMessageDecodeText,
    WSMessageNoDecodeText,
    WSMessageTextBytes,
    WSMsgType,
)
from ._websocket.reader import WebSocketReader
from ._websocket.writer import WebSocketWriter

# Messages that the WebSocketResponse.receive needs to handle internally
_INTERNAL_RECEIVE_TYPES = frozenset(
    (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.PING, WSMsgType.PONG)
)


__all__ = (
    "WS_CLOSED_MESSAGE",
    "WS_CLOSING_MESSAGE",
    "WS_KEY",
    "WebSocketReader",
    "WebSocketWriter",
    "WSMessage",
    "WSMessageDecodeText",
    "WSMessageNoDecodeText",
    "WSMessageTextBytes",
    "WebSocketError",
    "WSMsgType",
    "WSCloseCode",
    "ws_ext_gen",
    "ws_ext_parse",
    "WSHandshakeError",
)
<|endoftext|>import asyncio
from contextlib import suppress
from typing import Any, Callable

from .base_protocol import BaseProtocol
from .client_exceptions import (
    ClientConnectionError,
    ClientOSError,
    ClientPayloadError,
    ServerDisconnectedError,
    SocketTimeoutError,
)
from .helpers import (
    _EXC_SENTINEL,
    DEFAULT_CHUNK_SIZE,
    EMPTY_BODY_STATUS_CODES,
    BaseTimerContext,
    set_exception,
    set_result,
)
from .http import HttpResponseParser, RawResponseMessage
from .http_exceptions import HttpProcessingError
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader


class ResponseHandler(BaseProtocol, DataQueue[tuple[RawResponseMessage, StreamReader]]):
    """Helper class to adapt between Protocol and StreamReader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        BaseProtocol.__init__(self, loop=loop, parser=None)
        DataQueue.__init__(self, loop)

        self._should_close = False

        self._payload: StreamReader | None = None
        self._skip_payload = False
        self._payload_parser = None
        self._data_received_cb: Callable[[], None] | None = None

        self._timer = None
        self._tail = b""
        self._payload_parser_failed = False
        self._read_bufsize = DEFAULT_CHUNK_SIZE

        self._read_timeout: float | None = None
        self._read_timeout_handle: asyncio.TimerHandle | None = None

        self._timeout_ceil_threshold: float | None = 5

        self._closed: None | asyncio.Future[None] = None
        self._connection_lost_called = False

    @property
    def closed(self) -> None | asyncio.Future[None]:
        """Future that is set when the connection is closed.

        This property returns a Future that will be completed when the connection
        is closed. The Future is created lazily on first access to avoid creating
        futures that will never be awaited.

        Returns:
            - A Future[None] if the connection is still open or was closed after
              this property was accessed
            - None if connection_lost() was already called before this property
              was ever accessed (indicating no one is waiting for the closure)
        """
        if self._closed is None and not self._connection_lost_called:
            self._closed = self._loop.create_future()
        return self._closed

    @property
    def upgraded(self) -> bool:
        return self._upgraded

    @property
    def should_close(self) -> bool:
        return bool(
            self._should_close
            or (self._payload is not None and not self._payload.is_eof())
            or self._upgraded
            or self._exception is not None
            or self._payload_parser is not None
            or self._buffer
            or self._tail
        )

    def force_close(self) -> None:
        self._should_close = True

    def close(self) -> None:
        self._exception = None  # Break cyclic references
        transport = self.transport
        if transport is not None:
            transport.close()
            self.transport = None
            self._payload = None
            self._drop_timeout()

    def abort(self) -> None:
        self._exception = None  # Break cyclic references
        transport = self.transport
        if transport is not None:
            transport.abort()
            self.transport = None
            self._payload = None
            self._drop_timeout()

    def is_connected(self) -> bool:
        return self.transport is not None and not self.transport.is_closing()

    def connection_lost(self, exc: BaseException | None) -> None:
        self._connection_lost_called = True
        self._drop_timeout()

        original_connection_error = exc
        reraised_exc = original_connection_error

        connection_closed_cleanly = original_connection_error is None

        if self._closed is not None:
            # If someone is waiting for the closed future,
            # we should set it to None or an exception. If
            # self._closed is None, it means that
            # connection_lost() was called already
            # or nobody is waiting for it.
            if connection_closed_cleanly:
                set_result(self._closed, None)
            else:
                assert original_connection_error is not None
                set_exception(
                    self._closed,
                    ClientConnectionError(
                        f"Connection lost: {original_connection_error !s}",
                    ),
                    original_connection_error,
                )

        if self._payload_parser is not None:
            with suppress(Exception):  # FIXME: log this somehow?
                self._payload_parser.feed_eof()

        uncompleted = None
        if self._parser is not None:
            try:
                uncompleted = self._parser.feed_eof()
            except Exception as underlying_exc:
                if self._payload is not None:
                    client_payload_exc_msg = (
                        f"Response payload is not completed: {underlying_exc !r}"
                    )
                    if not connection_closed_cleanly:
                        client_payload_exc_msg = (
                            f"{client_payload_exc_msg !s}. "
                            f"{original_connection_error !r}"
                        )
                    set_exception(
                        self._payload,
                        ClientPayloadError(client_payload_exc_msg),
                        underlying_exc,
                    )

        if not self.is_eof():
            if isinstance(original_connection_error, OSError):
                reraised_exc = ClientOSError(*original_connection_error.args)
            if connection_closed_cleanly:
                reraised_exc = ServerDisconnectedError(uncompleted)
            # assigns self._should_close to True as side effect,
            # we do it anyway below
            underlying_non_eof_exc = (
                _EXC_SENTINEL
                if connection_closed_cleanly
                else original_connection_error
            )
            assert underlying_non_eof_exc is not None
            assert reraised_exc is not None
            self.set_exception(reraised_exc, underlying_non_eof_exc)

        self._should_close = True
        self._parser = None
        self._payload = None
        self._payload_parser = None
        self._payload_parser_failed = False
        self._reading_paused = False
        self._buffer_paused = False

        super().connection_lost(reraised_exc)

    def eof_received(self) -> None:
        # should call parser.feed_eof() most likely
        self._drop_timeout()

    def pause_reading(self) -> None:
        super().pause_reading()
        self._drop_timeout()

    def resume_reading(self, resume_parser: bool = True) -> None:
        was_paused = self._reading_paused
        super().resume_reading(resume_parser)
        # sock_read measures the peer, so it stays stopped while the tail
        # bound is still holding the transport.
        if was_paused and not self._buffer_paused:
            self._reschedule_timeout()

    def _drain_tail(self) -> None:
        if self._tail:
            data, self._tail = self._tail, b""
            self.data_received(data)
        # Tested empty-first so a read_bufsize of 0 cannot wedge the connection.
        still_bound = bool(self._tail) and len(self._tail) >= self._read_bufsize
        if self._buffer_paused and not still_bound:
            self._resume_reading_for_buffer()
        if self._reading_paused or self._buffer_paused:
            # The drain restarted sock_read through data_received(); anything
            # still holding the transport means it stays stopped.
            self._drop_timeout()

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        self._should_close = True
        self._drop_timeout()
        super().set_exception(exc, exc_cause)

    def set_parser(
        self,
        parser: Any,
        payload: Any,
        data_received_cb: Callable[[], None] | None = None,
    ) -> None:
        # TODO: actual types are:
        #   parser: WebSocketReader
        #   payload: WebSocketDataQueue
        # but they are not generi enough
        # Need an ABC for both types
        self._payload = payload
        self._payload_parser = parser
        self._data_received_cb = data_received_cb

        self._drop_timeout()

        self._drain_tail()

    def set_response_params(
        self,
        *,
        timer: BaseTimerContext | None = None,
        skip_payload: bool = False,
        read_until_eof: bool = False,
        auto_decompress: bool = True,
        read_timeout: float | None = None,
        read_bufsize: int = DEFAULT_CHUNK_SIZE,
        timeout_ceil_threshold: float = 5,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        max_headers: int = 128,
    ) -> None:
        self._skip_payload = skip_payload

        self._read_bufsize = read_bufsize
        self._read_timeout = read_timeout

        self._timeout_ceil_threshold = timeout_ceil_threshold

        self._parser = HttpResponseParser(
            self,
            self._loop,
            read_bufsize,
            timer=timer,
            payload_exception=ClientPayloadError,
            response_with_body=not skip_payload,
            read_until_eof=read_until_eof,
            auto_decompress=auto_decompress,
            max_line_size=max_line_size,
            max_field_size=max_field_size,
            max_headers=max_headers,
        )

        self._drain_tail()

    def _drop_timeout(self) -> None:
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()
            self._read_timeout_handle = None

    def _reschedule_timeout(self) -> None:
        timeout = self._read_timeout
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()

        if timeout:
            self._read_timeout_handle = self._loop.call_later(
                timeout, self._on_read_timeout
            )
        else:
            self._read_timeout_handle = None

    def start_timeout(self) -> None:
        self._reschedule_timeout()

    @property
    def read_timeout(self) -> float | None:
        return self._read_timeout

    @read_timeout.setter
    def read_timeout(self, read_timeout: float | None) -> None:
        self._read_timeout = read_timeout

    def _on_read_timeout(self) -> None:
        exc = SocketTimeoutError("Timeout on reading data from socket")
        self.set_exception(exc)
        if self._payload is not None:
            set_exception(self._payload, exc)

    def data_received(self, data: bytes) -> None:
        if self._payload_parser_failed:
            # Dropped, not closed on, so queued messages stay answerable and
            # the peer's FIN still arrives. Discarded bytes are not progress,
            # so they must not hold off sock_read either.
            return

        # If no data, then we are resuming decompression. We haven't received
        # data from the socket, so we can avoid the reschedule overhead.
        if data:
            self._reschedule_timeout()

        # custom payload parser - currently always WebSocketReader
        if self._payload_parser is not None:
            if self._data_received_cb is not None:
                self._data_received_cb()
            # WebSocketReader signals EOF only for a protocol error, never
            # for a clean close, and has already put the error on the queue.
            protocol_error, _ = self._payload_parser.feed_data(data)
            if protocol_error:
                self._payload = None
                self._payload_parser = None
                self._payload_parser_failed = True
            return

        if self._upgraded or self._parser is None:
            # i.e. websocket connection, websocket parser is not set yet
            self._tail += data
            # Nothing drains this until set_parser() runs, and an await
            # after the 101 can hold that off; stop reading instead.
            # Tested non-empty so a read_bufsize of 0 cannot pause on nothing.
            if (
                not self._buffer_paused
                and self._tail
                and len(self._tail) >= self._read_bufsize
            ):
                self._pause_reading_for_buffer()
                # sock_read measures the peer; this pause is ours.
                self._drop_timeout()
            return

        # parse http messages
        try:
            messages, upgraded, tail = self._parser.feed_data(data)
        except BaseException as underlying_exc:
            if self.transport is not None:
                # connection.release() could be called BEFORE
                # data_received(), the transport is already
                # closed in this case
                self.transport.close()
            if not isinstance(underlying_exc, Exception):
                raise
            # should_close is True after the call
            if isinstance(underlying_exc, HttpProcessingError):
                exc = HttpProcessingError(
                    code=underlying_exc.code,
                    message=underlying_exc.message,
                    headers=underlying_exc.headers,
                )
            else:
                exc = HttpProcessingError()
            self.set_exception(exc, underlying_exc)
            return

        self._upgraded = upgraded

        payload: StreamReader | None = None
        for message, payload in messages:
            if message.should_close:
                self._should_close = True

            self._payload = payload

            if self._skip_payload or message.code in EMPTY_BODY_STATUS_CODES:
                self.feed_data((message, EMPTY_PAYLOAD), 0)
            else:
                self.feed_data((message, payload), 0)

        if payload is not None:
            # new message(s) was processed
            # register timeout handler unsubscribing
            # either on end-of-stream or immediately for
            # EMPTY_PAYLOAD
            if payload is not EMPTY_PAYLOAD:
                payload.on_eof(self._drop_timeout)
            else:
                self._drop_timeout()

        if upgraded and tail:
            self.data_received(tail)
<|endoftext|>import abc
import asyncio
import base64
import fnmatch
import functools
import hashlib
import html
import inspect
import keyword
import os
import platform
import re
import sys
import warnings
from collections.abc import (
    Awaitable,
    Callable,
    Container,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sized,
)
from functools import wraps
from pathlib import Path
from re import Pattern
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NoReturn, Optional, TypedDict, cast

from yarl import URL, __version__ as yarl_version

from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG, DEFAULT_CHUNK_SIZE
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
    HTTPException,
    HTTPExpectationFailed,
    HTTPForbidden,
    HTTPMethodNotAllowed,
    HTTPNotFound,
)
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef

__all__ = (
    "UrlDispatcher",
    "UrlMappingMatchInfo",
    "AbstractResource",
    "Resource",
    "PlainResource",
    "DynamicResource",
    "AbstractRoute",
    "ResourceRoute",
    "StaticResource",
    "View",
)


if TYPE_CHECKING:
    from .web_app import Application

    BaseDict = dict[str, str]
else:
    BaseDict = dict

CIRCULAR_SYMLINK_ERROR = (RuntimeError,) if sys.version_info < (3, 13) else ()

YARL_VERSION: Final[tuple[int, ...]] = tuple(map(int, yarl_version.split(".")[:2]))

HTTP_METHOD_RE: Final[Pattern[str]] = re.compile(
    r"^[0-9A-Za-z!#\$%&'\*\+\-\.\^_`\|~]+$"
)
ROUTE_RE: Final[Pattern[str]] = re.compile(
    r"(\{[_a-zA-Z][^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
)
PATH_SEP: Final[str] = re.escape("/")

IS_WINDOWS: Final[bool] = platform.system() == "Windows"

_ExpectHandler = Callable[[Request], Awaitable[StreamResponse | None]]
_Resolve = tuple[Optional["UrlMappingMatchInfo"], set[str]]

html_escape = functools.partial(html.escape, quote=True)


class _InfoDict(TypedDict, total=False):
    path: str

    formatter: str
    pattern: Pattern[str]

    directory: Path
    prefix: str
    routes: Mapping[str, "AbstractRoute"]

    app: "Application"

    domain: str

    rule: "AbstractRuleMatching"

    http_exception: HTTPException


class AbstractResource(Sized, Iterable["AbstractRoute"]):
    def __init__(self, *, name: str | None = None) -> None:
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        """Exposes the resource's canonical path.

        For example '/foo/bar/{name}'

        """

    @abc.abstractmethod  # pragma: no branch
    def url_for(self, **kwargs: str) -> URL:
        """Construct url for resource with additional params."""

    @abc.abstractmethod  # pragma: no branch
    async def resolve(self, request: Request) -> _Resolve:
        """Resolve resource.

        Return (UrlMappingMatchInfo, allowed_methods) pair.
        """

    @abc.abstractmethod
    def add_prefix(self, prefix: str) -> None:
        """Add a prefix to processed URLs.

        Required for subapplications support.
        """

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        """Return a dict with additional info useful for introspection"""

    def freeze(self) -> None:
        pass

    @abc.abstractmethod
    def raw_match(self, path: str) -> bool:
        """Perform a raw match against path"""


class AbstractRoute(abc.ABC):
    def __init__(
        self,
        method: str,
        handler: Handler | type[AbstractView],
        *,
        expect_handler: _ExpectHandler | None = None,
        resource: AbstractResource | None = None,
    ) -> None:

        if expect_handler is None:
            expect_handler = _default_expect_handler

        assert inspect.iscoroutinefunction(expect_handler) or (
            sys.version_info < (3, 14) and asyncio.iscoroutinefunction(expect_handler)
        ), f"Coroutine is expected, got {expect_handler!r}"

        method = method.upper()
        if not HTTP_METHOD_RE.match(method):
            raise ValueError(f"{method} is not allowed HTTP method")

        assert callable(handler), handler
        if inspect.iscoroutinefunction(handler) or (
            sys.version_info < (3, 14) and asyncio.iscoroutinefunction(handler)
        ):
            pass
        elif inspect.isgeneratorfunction(handler):
            if TYPE_CHECKING:
                assert False
            warnings.warn(
                "Bare generators are deprecated, use @coroutine wrapper",
                DeprecationWarning,
            )
        elif isinstance(handler, type) and issubclass(handler, AbstractView):
            pass
        else:
            warnings.warn(
                "Bare functions are deprecated, use async ones", DeprecationWarning
            )

            @wraps(handler)
            async def handler_wrapper(request: Request) -> StreamResponse:
                result = old_handler(request)  # type: ignore[call-arg]
                if asyncio.iscoroutine(result):
                    result = await result
                assert isinstance(result, StreamResponse)
                return result

            old_handler = handler
            handler = handler_wrapper

        self._method = method
        self._handler = handler
        self._expect_handler = expect_handler
        self._resource = resource

    @property
    def method(self) -> str:
        return self._method

    @property
    def handler(self) -> Handler:
        return self._handler

    @property
    @abc.abstractmethod
    def name(self) -> str | None:
        """Optional route's name, always equals to resource's name."""

    @property
    def resource(self) -> AbstractResource | None:
        return self._resource

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        """Return a dict with additional info useful for introspection"""

    @abc.abstractmethod  # pragma: no branch
    def url_for(self, *args: str, **kwargs: str) -> URL:
        """Construct url for route with additional params."""

    async def handle_expect_header(self, request: Request) -> StreamResponse | None:
        return await self._expect_handler(request)


class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):

    __slots__ = ("_route", "_apps", "_current_app", "_frozen")

    def __init__(self, match_dict: dict[str, str], route: AbstractRoute) -> None:
        super().__init__(match_dict)
        self._route = route
        self._apps: list[Application] = []
        self._current_app: Application | None = None
        self._frozen = False

    @property
    def handler(self) -> Handler:
        return self._route.handler

    @property
    def route(self) -> AbstractRoute:
        return self._route

    @property
    def expect_handler(self) -> _ExpectHandler:
        return self._route.handle_expect_header

    @property
    def http_exception(self) -> HTTPException | None:
        return None

    def get_info(self) -> _InfoDict:  # type: ignore[override]
        return self._route.get_info()

    @property
    def apps(self) -> tuple["Application", ...]:
        return tuple(self._apps)

    def add_app(self, app: "Application") -> None:
        if self._frozen:
            raise RuntimeError("Cannot change apps stack after .freeze() call")
        if self._current_app is None:
            self._current_app = app
        self._apps.insert(0, app)

    @property
    def current_app(self) -> "Application":
        app = self._current_app
        assert app is not None
        return app

    @current_app.setter
    def current_app(self, app: "Application") -> None:
        if DEBUG:  # pragma: no cover
            if app not in self._apps:
                raise RuntimeError(
                    f"Expected one of the following apps {self._apps!r}, got {app!r}"
                )
        self._current_app = app

    def freeze(self) -> None:
        self._frozen = True

    def __repr__(self) -> str:
        return f"<MatchInfo {super().__repr__()}: {self._route}>"


class MatchInfoError(UrlMappingMatchInfo):

    __slots__ = ("_exception",)

    def __init__(self, http_exception: HTTPException) -> None:
        self._exception = http_exception
        super().__init__({}, SystemRoute(self._exception))

    @property
    def http_exception(self) -> HTTPException:
        return self._exception

    def __repr__(self) -> str:
        return f"<MatchInfoError {self._exception.status}: {self._exception.reason}>"


async def _default_expect_handler(request: Request) -> None:
    """Default handler for Expect header.

    Just send "100 Continue" to client.
    raise HTTPExpectationFailed if value of header is not "100-continue"
    """
    expect = request.headers.get(hdrs.EXPECT, "")
    if request.version == HttpVersion11:
        if expect.lower() == "100-continue":
            await request.writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            # Reset output_size as we haven't started the main body yet.
            request.writer.output_size = 0
        else:
            raise HTTPExpectationFailed(text="Unknown Expect: %s" % expect)


class Resource(AbstractResource):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._routes: dict[str, ResourceRoute] = {}
        self._any_route: ResourceRoute | None = None
        self._allowed_methods: set[str] = set()

    def add_route(
        self,
        method: str,
        handler: type[AbstractView] | Handler,
        *,
        expect_handler: _ExpectHandler | None = None,
    ) -> "ResourceRoute":
        if route := self._routes.get(method, self._any_route):
            raise RuntimeError(
                "Added route will never be executed, "
                f"method {route.method} is already "
                "registered"
            )

        route_obj = ResourceRoute(method, handler, self, expect_handler=expect_handler)
        self.register_route(route_obj)
        return route_obj

    def register_route(self, route: "ResourceRoute") -> None:
        assert isinstance(
            route, ResourceRoute
        ), f"Instance of Route class is required, got {route!r}"
        if route.method == hdrs.METH_ANY:
            self._any_route = route
        self._allowed_methods.add(route.method)
        self._routes[route.method] = route

    async def resolve(self, request: Request) -> _Resolve:
        if (match_dict := self._match(request.rel_url.path_safe)) is None:
            return None, set()
        if route := self._routes.get(request.method, self._any_route):
            return UrlMappingMatchInfo(match_dict, route), self._allowed_methods
        return None, self._allowed_methods

    @abc.abstractmethod
    def _match(self, path: str) -> dict[str, str] | None:
        pass  # pragma: no cover

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator["ResourceRoute"]:
        return iter(self._routes.values())

    # TODO: implement all abstract methods


class PlainResource(Resource):
    def __init__(self, path: str, *, name: str | None = None) -> None:
        super().__init__(name=name)
        assert not path or path.startswith("/")
        self._path = path

    @property
    def canonical(self) -> str:
        return self._path

    def freeze(self) -> None:
        if not self._path:
            self._path = "/"

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._path = prefix + self._path

    def _match(self, path: str) -> dict[str, str] | None:
        # string comparison is about 10 times faster than regexp matching
        if self._path == path:
            return {}
        return None

    def raw_match(self, path: str) -> bool:
        return self._path == path

    def get_info(self) -> _InfoDict:
        return {"path": self._path}

    def url_for(self) -> URL:  # type: ignore[override]
        return URL.build(path=self._path, encoded=True)

    def __repr__(self) -> str:
        name = "'" + self.name + "' " if self.name is not None else ""
        return f"<PlainResource {name} {self._path}>"


class DynamicResource(Resource):

    DYN = re.compile(r"\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*)\}")
    DYN_WITH_RE = re.compile(r"\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*):(?P<re>.+)\}")
    GOOD = r"[^{}/]+"

    def __init__(self, path: str, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._orig_path = path
        pattern = ""
        formatter = ""
        for part in ROUTE_RE.split(path):
            match = self.DYN.fullmatch(part)
            if match:
                pattern += "(?P<{}>{})".format(match.group("var"), self.GOOD)
                formatter += "{" + match.group("var") + "}"
                continue

            match = self.DYN_WITH_RE.fullmatch(part)
            if match:
                pattern += "(?P<{var}>{re})".format(**match.groupdict())
                formatter += "{" + match.group("var") + "}"
                continue

            if "{" in part or "}" in part:
                raise ValueError(f"Invalid path '{path}'['{part}']")

            part = _requote_path(part)
            formatter += part
            pattern += re.escape(part)

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Bad pattern '{pattern}': {exc}") from None
        assert compiled.pattern.startswith(PATH_SEP)
        assert formatter.startswith("/")
        self._pattern = compiled
        self._formatter = formatter

    @property
    def canonical(self) -> str:
        return self._formatter

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._pattern = re.compile(re.escape(prefix) + self._pattern.pattern)
        self._formatter = prefix + self._formatter

    def _match(self, path: str) -> dict[str, str] | None:
        match = self._pattern.fullmatch(path)
        if match is None:
            return None
        return {
            key: _unquote_path_safe(value) for key, value in match.groupdict().items()
        }

    def raw_match(self, path: str) -> bool:
        return self._orig_path == path

    def get_info(self) -> _InfoDict:
        return {"formatter": self._formatter, "pattern": self._pattern}

    def url_for(self, **parts: str) -> URL:
        url = self._formatter.format_map({k: _quote_path(v) for k, v in parts.items()})
        return URL.build(path=url, encoded=True)

    def __repr__(self) -> str:
        name = "'" + self.name + "' " if self.name is not None else ""
        return f"<DynamicResource {name} {self._formatter}>"


class PrefixResource(AbstractResource):
    def __init__(self, prefix: str, *, name: str | None = None) -> None:
        assert not prefix or prefix.startswith("/"), prefix
        assert prefix in ("", "/") or not prefix.endswith("/"), prefix
        super().__init__(name=name)
        self._prefix = _requote_path(prefix)
        self._prefix2 = self._prefix + "/"

    @property
    def canonical(self) -> str:
        return self._prefix

    def add_prefix(self, prefix: str) -> None:
        assert prefix.startswith("/")
        assert not prefix.endswith("/")
        assert len(prefix) > 1
        self._prefix = prefix + self._prefix
        self._prefix2 = self._prefix + "/"

    def raw_match(self, prefix: str) -> bool:
        return False

    # TODO: impl missing abstract methods


class StaticResource(PrefixResource):
    VERSION_KEY = "v"

    def __init__(
        self,
        prefix: str,
        directory: PathLike,
        *,
        name: str | None = None,
        expect_handler: _ExpectHandler | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        show_index: bool = False,
        follow_symlinks: bool = False,
        append_version: bool = False,
    ) -> None:
        super().__init__(prefix, name=name)
        try:
            directory = Path(directory).expanduser().resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"'{directory}' does not exist") from error
        if not directory.is_dir():
            raise ValueError(f"'{directory}' is not a directory")
        self._directory = directory
        self._show_index = show_index
        self._chunk_size = chunk_size
        self._follow_symlinks = follow_symlinks
        self._expect_handler = expect_handler
        self._append_version = append_version

        self._routes = {
            "GET": ResourceRoute(
                "GET", self._handle, self, expect_handler=expect_handler
            ),
            "HEAD": ResourceRoute(
                "HEAD", self._handle, self, expect_handler=expect_handler
            ),
        }
        self._allowed_methods = set(self._routes)

    def url_for(  # type: ignore[override]
        self,
        *,
        filename: PathLike,
        append_version: bool | None = None,
    ) -> URL:
        if append_version is None:
            append_version = self._append_version
        filename = str(filename).lstrip("/")

        url = URL.build(path=self._prefix, encoded=True)
        # filename is not encoded
        if YARL_VERSION < (1, 6):
            url = url / filename.replace("%", "%25")
        else:
            url = url / filename

        if append_version:
            unresolved_path = self._directory.joinpath(filename)
            try:
                if self._follow_symlinks:
                    normalized_path = Path(os.path.normpath(unresolved_path))
                    normalized_path.relative_to(self._directory)
                    filepath = normalized_path.resolve()
                else:
                    filepath = unresolved_path.resolve()
                    filepath.relative_to(self._directory)
            except (ValueError, FileNotFoundError):
                # ValueError for case when path point to symlink
                # with follow_symlinks is False
                return url  # relatively safe
            if filepath.is_file():
                # TODO cache file content
                # with file watcher for cache invalidation
                with filepath.open("rb") as f:
                    file_bytes = f.read()
                h = self._get_file_hash(file_bytes)
                url = url.with_query({self.VERSION_KEY: h})
                return url
        return url

    @staticmethod
    def _get_file_hash(byte_array: bytes) -> str:
        m = hashlib.sha256()  # todo sha256 can be configurable param
        m.update(byte_array)
        b64 = base64.urlsafe_b64encode(m.digest())
        return b64.decode("ascii")

    def get_info(self) -> _InfoDict:
        return {
            "directory": self._directory,
            "prefix": self._prefix,
            "routes": self._routes,
        }

    def set_options_route(self, handler: Handler) -> None:
        if "OPTIONS" in self._routes:
            raise RuntimeError("OPTIONS route was set already")
        self._routes["OPTIONS"] = ResourceRoute(
            "OPTIONS", handler, self, expect_handler=self._expect_handler
        )
        self._allowed_methods.add("OPTIONS")

    async def resolve(self, request: Request) -> _Resolve:
        path = request.rel_url.path_safe
        method = request.method
        # We normalise here to avoid matches that traverse below the static root.
        # e.g. /static/../../../../home/user/webapp/static/
        norm_path = os.path.normpath(path)
        if IS_WINDOWS:
            norm_path = norm_path.replace("\\", "/")
        if not norm_path.startswith(self._prefix2) and norm_path != self._prefix:
            return None, set()

        allowed_methods = self._allowed_methods
        if method not in allowed_methods:
            return None, allowed_methods

        match_dict = {"filename": _unquote_path_safe(path[len(self._prefix) + 1 :])}
        return (UrlMappingMatchInfo(match_dict, self._routes[method]), allowed_methods)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        return iter(self._routes.values())

    async def _handle(self, request: Request) -> StreamResponse:
        filename = request.match_info["filename"]
        if Path(filename).is_absolute():
            # filename is an absolute path e.g. //network/share or D:\path
            # which could be a UNC path leading to NTLM credential theft
            raise HTTPNotFound()
        unresolved_path = self._directory.joinpath(filename)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._resolve_path_to_response, unresolved_path
        )

    def _resolve_path_to_response(self, unresolved_path: Path) -> StreamResponse:
        """Take the unresolved path and query the file system to form a response."""
        # Check for access outside the root directory. For follow symlinks, URI
        # cannot traverse out, but symlinks can. Otherwise, no access outside
        # root is permitted.
        try:
            if self._follow_symlinks:
                normalized_path = Path(os.path.normpath(unresolved_path))
                normalized_path.relative_to(self._directory)
                file_path = normalized_path.resolve()
            else:
                file_path = unresolved_path.resolve()
                file_path.relative_to(self._directory)
        except (ValueError, *CIRCULAR_SYMLINK_ERROR) as error:
            # ValueError is raised for the relative check. Circular symlinks
            # raise here on resolving for python < 3.13.
            raise HTTPNotFound() from error

        # if path is a directory, return the contents if permitted. Note the
        # directory check will raise if a segment is not readable.
        try:
            if file_path.is_dir():
                if self._show_index:
                    return Response(
                        text=self._directory_as_html(file_path),
                        content_type="text/html",
                    )
                else:
                    raise HTTPForbidden()
        except PermissionError as error:
            raise HTTPForbidden() from error

        # Return the file response, which handles all other checks.
        return FileResponse(file_path, chunk_size=self._chunk_size)

    def _directory_as_html(self, dir_path: Path) -> str:
        """returns directory's index as html."""
        assert dir_path.is_dir()

        relative_path_to_dir = dir_path.relative_to(self._directory).as_posix()
        index_of = f"Index of /{html_escape(relative_path_to_dir)}"
        h1 = f"<h1>{index_of}</h1>"

        index_list = []
        dir_index = dir_path.iterdir()
        for _file in sorted(dir_index):
            # show file url as relative to static path
            rel_path = _file.relative_to(self._directory).as_posix()
            quoted_file_url = _quote_path(f"{self._prefix}/{rel_path}")

            # if file is a directory, add '/' to the end of the name
            if _file.is_dir():
                file_name = f"{_file.name}/"
            else:
                file_name = _file.name

            index_list.append(
                f'<li><a href="{quoted_file_url}">{html_escape(file_name)}</a></li>'
            )
        ul = "<ul>\n{}\n</ul>".format("\n".join(index_list))
        body = f"<body>\n{h1}\n{ul}\n</body>"

        head_str = f"<head>\n<title>{index_of}</title>\n</head>"
        html = f"<html>\n{head_str}\n{body}\n</html>"

        return html

    def __repr__(self) -> str:
        name = "'" + self.name + "'" if self.name is not None else ""
        return f"<StaticResource {name} {self._prefix} -> {self._directory!r}>"


class PrefixedSubAppResource(PrefixResource):
    def __init__(self, prefix: str, app: "Application") -> None:
        super().__init__(prefix)
        self._app = app
        self._add_prefix_to_resources(prefix)

    def add_prefix(self, prefix: str) -> None:
        super().add_prefix(prefix)
        self._add_prefix_to_resources(prefix)

    def _add_prefix_to_resources(self, prefix: str) -> None:
        router = self._app.router
        for resource in router.resources():
            # Since the canonical path of a resource is about
            # to change, we need to unindex it and then reindex
            router.unindex_resource(resource)
            resource.add_prefix(prefix)
            router.index_resource(resource)

    def url_for(self, *args: str, **kwargs: str) -> URL:
        raise RuntimeError(".url_for() is not supported by sub-application root")

    def get_info(self) -> _InfoDict:
        return {"app": self._app, "prefix": self._prefix}

    async def resolve(self, request: Request) -> _Resolve:
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return match_info, methods

    def __len__(self) -> int:
        return len(self._app.router.routes())

    def __iter__(self) -> Iterator[AbstractRoute]:
        return iter(self._app.router.routes())

    def __repr__(self) -> str:
        return f"<PrefixedSubAppResource {self._prefix} -> {self._app!r}>"


class AbstractRuleMatching(abc.ABC):
    @abc.abstractmethod  # pragma: no branch
    async def match(self, request: Request) -> bool:
        """Return bool if the request satisfies the criteria"""

    @abc.abstractmethod  # pragma: no branch
    def get_info(self) -> _InfoDict:
        """Return a dict with additional info useful for introspection"""

    @property
    @abc.abstractmethod  # pragma: no branch
    def canonical(self) -> str:
        """Return a str"""


class Domain(AbstractRuleMatching):
    re_part = re.compile(r"(?!-)[a-z\d-]{1,63}(?<!-)")

    def __init__(self, domain: str) -> None:
        super().__init__()
        self._domain = self.validation(domain)

    @property
    def canonical(self) -> str:
        return self._domain

    def validation(self, domain: str) -> str:
        if not isinstance(domain, str):
            raise TypeError("Domain must be str")
        domain = domain.rstrip(".").lower()
        if not domain:
            raise ValueError("Domain cannot be empty")
        elif "://" in domain:
            raise ValueError("Scheme not supported")
        url = URL("http://" + domain)
        assert url.raw_host is not None
        if not all(self.re_part.fullmatch(x) for x in url.raw_host.split(".")):
            raise ValueError("Domain not valid")
        if url.port == 80:
            return url.raw_host
        return f"{url.raw_host}:{url.port}"

    async def match(self, request: Request) -> bool:
        host = request.headers.get(hdrs.HOST)
        if not host:
            return False
        return self.match_domain(host)

    def match_domain(self, host: str) -> bool:
        host = host.lower()
        # The port only matters when the domain was registered with one.
        return host == self._domain or host.partition(":")[0] == self._domain

    def get_info(self) -> _InfoDict:
        return {"domain": self._domain}


class MaskDomain(Domain):
    re_part = re.compile(r"(?!-)[a-z\d\*-]{1,63}(?<!-)")

    def __init__(self, domain: str) -> None:
        super().__init__(domain)
        self._pattern = (
            self._domain.replace(".", r"\.").replace("*", "[^:]*") + "(:.*)?"
        )
        host_mask, _, self._port = self._domain.partition(":")
        self._mask = re.compile(fnmatch.translate(host_mask), re.I)

    @property
    def canonical(self) -> str:
        return self._pattern

    def match_domain(self, host: str) -> bool:
        hostname, _, port = host.partition(":")
        if self._port and port != self._port:
            return False
        return self._mask.match(hostname) is not None


class MatchedSubAppResource(PrefixedSubAppResource):
    def __init__(self, rule: AbstractRuleMatching, app: "Application") -> None:
        AbstractResource.__init__(self)
        self._prefix = ""
        self._app = app
        self._rule = rule

    @property
    def canonical(self) -> str:
        return self._rule.canonical

    def get_info(self) -> _InfoDict:
        return {"app": self._app, "rule": self._rule}

    async def resolve(self, request: Request) -> _Resolve:
        if not await self._rule.match(request):
            return None, set()
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return match_info, methods

    def __repr__(self) -> str:
        return f"<MatchedSubAppResource -> {self._app!r}>"


class ResourceRoute(AbstractRoute):
    """A route with resource"""

    def __init__(
        self,
        method: str,
        handler: Handler | type[AbstractView],
        resource: AbstractResource,
        *,
        expect_handler: _ExpectHandler | None = None,
    ) -> None:
        super().__init__(
            method, handler, expect_handler=expect_handler, resource=resource
        )

    def __repr__(self) -> str:
        return f"<ResourceRoute [{self.method}] {self._resource} -> {self.handler!r}"

    @property
    def name(self) -> str | None:
        if self._resource is None:
            return None
        return self._resource.name

    def url_for(self, *args: str, **kwargs: str) -> URL:
        """Construct url for route with additional params."""
        assert self._resource is not None
        return self._resource.url_for(*args, **kwargs)

    def get_info(self) -> _InfoDict:
        assert self._resource is not None
        return self._resource.get_info()


class SystemRoute(AbstractRoute):
    def __init__(self, http_exception: HTTPException) -> None:
        super().__init__(hdrs.METH_ANY, self._handle)
        self._http_exception = http_exception

    def url_for(self, *args: str, **kwargs: str) -> URL:
        raise RuntimeError(".url_for() is not allowed for SystemRoute")

    @property
    def name(self) -> str | None:
        return None

    def get_info(self) -> _InfoDict:
        return {"http_exception": self._http_exception}

    async def _handle(self, request: Request) -> StreamResponse:
        raise self._http_exception

    @property
    def status(self) -> int:
        return self._http_exception.status

    @property
    def reason(self) -> str:
        return self._http_exception.reason

    def __repr__(self) -> str:
        return f"<SystemRoute {self.status}: {self.reason}>"


class View(AbstractView):
    async def _iter(self) -> StreamResponse:
        if self.request.method not in hdrs.METH_ALL:
            self._raise_allowed_methods()
        method: Callable[[], Awaitable[StreamResponse]] | None
        method = getattr(self, self.request.method.lower(), None)
        if method is None:
            self._raise_allowed_methods()
        ret = await method()
        assert isinstance(ret, StreamResponse)
        return ret

    def __await__(self) -> Generator[None, None, StreamResponse]:
        return self._iter().__await__()

    def _raise_allowed_methods(self) -> NoReturn:
        allowed_methods = {m for m in hdrs.METH_ALL if hasattr(self, m.lower())}
        raise HTTPMethodNotAllowed(self.request.method, allowed_methods)


class ResourcesView(Sized, Iterable[AbstractResource], Container[AbstractResource]):
    def __init__(self, resources: list[AbstractResource]) -> None:
        self._resources = resources

    def __len__(self) -> int:
        return len(self._resources)

    def __iter__(self) -> Iterator[AbstractResource]:
        yield from self._resources

    def __contains__(self, resource: object) -> bool:
        return resource in self._resources


class RoutesView(Sized, Iterable[AbstractRoute], Container[AbstractRoute]):
    def __init__(self, resources: list[AbstractResource]):
        self._routes: list[AbstractRoute] = []
        for resource in resources:
            for route in resource:
                self._routes.append(route)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        yield from self._routes

    def __contains__(self, route: object) -> bool:
        return route in self._routes


