            if self._raw_cookie_headers is not None:
                # Parse cookies for response.cookies (SimpleCookie for backward compatibility)
                cookies = SimpleCookie()
                # Use parse_set_cookie_headers for more lenient parsing that handles
                # malformed cookies better than SimpleCookie.load
                cookies.update(parse_set_cookie_headers(self._raw_cookie_headers))
                self._cookies = cookies
            else:
                self._cookies = SimpleCookie()
        return self._cookies

    @cookies.setter
    def cookies(self, cookies: SimpleCookie) -> None:
        self._cookies = cookies
        # Generate raw cookie headers from the SimpleCookie
        if cookies:
            self._raw_cookie_headers = tuple(
                morsel.OutputString() for morsel in cookies.values()
            )
        else:
            self._raw_cookie_headers = None

    @reify
    def url(self) -> URL:
        return self._url

    @reify
    def url_obj(self) -> URL:
        warnings.warn("Deprecated, use .url #1654", DeprecationWarning, stacklevel=2)
        return self._url

    @reify
    def real_url(self) -> URL:
        return self._real_url

    @reify
    def host(self) -> str:
        assert self._url.host is not None
        return self._url.host

    @reify
    def headers(self) -> "CIMultiDictProxy[str]":
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        return self._raw_headers

    @reify
    def request_info(self) -> RequestInfo:
        return self._request_info

    @reify
    def content_disposition(self) -> ContentDisposition | None:
        raw = self._headers.get(hdrs.CONTENT_DISPOSITION)
        if raw is None:
            return None
        disposition_type, params_dct = multipart.parse_content_disposition(raw)
        params = MappingProxyType(params_dct)
        filename = multipart.content_disposition_filename(params)
        return ContentDisposition(disposition_type, params, filename)

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._closed:
            return

        if self._connection is not None:
            self._connection.release()
            self._cleanup_writer()

            if self._loop.get_debug():
                kwargs = {"source": self}
                _warnings.warn(f"Unclosed response {self!r}", ResourceWarning, **kwargs)
                context = {"client_response": self, "message": "Unclosed response"}
                if self._source_traceback:
                    context["source_traceback"] = self._source_traceback
                self._loop.call_exception_handler(context)

    def __repr__(self) -> str:
        out = io.StringIO()
        ascii_encodable_url = str(self.url)
        if self.reason:
            ascii_encodable_reason = self.reason.encode(
                "ascii", "backslashreplace"
            ).decode("ascii")
        else:
            ascii_encodable_reason = "None"
        print(
            f"<ClientResponse({ascii_encodable_url}) [{self.status} {ascii_encodable_reason}]>",
            file=out,
        )
        print(self.headers, file=out)
        return out.getvalue()

    @property
    def connection(self) -> Optional["Connection"]:
        return self._connection

    @reify
    def history(self) -> tuple["ClientResponse", ...]:
        """A sequence of of responses, if redirects occurred."""
        return self._history

    @reify
    def links(self) -> "MultiDictProxy[MultiDictProxy[str | URL]]":
        links_str = ", ".join(self.headers.getall("link", []))

        if not links_str:
            return MultiDictProxy(MultiDict())

        links: MultiDict[MultiDictProxy[str | URL]] = MultiDict()

        for val in re.split(r",(?=\s*<)", links_str):
            match = re.match(r"\s*<(.*)>(.*)", val)
            if match is None:  # pragma: no cover
                # the check exists to suppress mypy error
                continue
            url, params_str = match.groups()
            params = params_str.split(";")[1:]

            link: MultiDict[str | URL] = MultiDict()

            for param in params:
                match = _LINK_PARAM_RE.match(param.strip())
                if match is None:  # Malformed param
                    continue
                key, _, value_quoted, value_unquoted = match.groups()

                link.add(key, value_unquoted if value_quoted is None else value_quoted)

            key = link.get("rel", url)

            link.add("url", self.url.join(URL(url)))

            links.add(str(key), MultiDictProxy(link))

        return MultiDictProxy(links)

    async def start(self, connection: "Connection") -> "ClientResponse":
        """Start response processing."""
        self._closed = False
        self._protocol = connection.protocol
        self._connection = connection

        with self._timer:
            while True:
                # read response
                try:
                    protocol = self._protocol
                    message, payload = await protocol.read()  # type: ignore[union-attr]
                except http.HttpProcessingError as exc:
                    raise ClientResponseError(
                        self.request_info,
                        self.history,
                        status=exc.code,
                        message=exc.message,
                        headers=exc.headers,
                    ) from exc

                if message.code < 100 or message.code > 199 or message.code == 101:
                    break

                if self._continue is not None:
                    set_result(self._continue, True)
                    self._continue = None

        # payload eof handler
        payload.on_eof(self._response_eof)

        # response status
        self.version = message.version
        self.status = message.code
        self.reason = message.reason

        # headers
        self._headers = message.headers  # type is CIMultiDictProxy
        self._raw_headers = message.raw_headers  # type is Tuple[bytes, bytes]

        # payload
        self.content = payload

        # cookies
        if cookie_hdrs := self.headers.getall(hdrs.SET_COOKIE, ()):
            # Store raw cookie headers for CookieJar
            self._raw_cookie_headers = tuple(cookie_hdrs)
        return self

    def _response_eof(self) -> None:
        if self._closed:
            return

        # protocol could be None because connection could be detached
        protocol = self._connection and self._connection.protocol
        if protocol is not None and protocol.upgraded:
            return

        self._closed = True
        self._cleanup_writer()
        self._release_connection()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._released:
            self._notify_content()

        self._closed = True
        if self._loop is None or self._loop.is_closed():
            return

        self._cleanup_writer()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def release(self) -> Any:
        if not self._released:
            self._notify_content()

        self._closed = True

        self._cleanup_writer()
        self._release_connection()
        return noop()

    @property
    def ok(self) -> bool:
        """Returns ``True`` if ``status`` is less than ``400``, ``False`` if not.

        This is **not** a check for ``200 OK`` but a check that the response
        status is under 400.
        """
        return 400 > self.status

    def raise_for_status(self) -> None:
        if not self.ok:
            # reason should always be not None for a started response
            assert self.reason is not None

            # If we're in a context we can rely on __aexit__() to release as the
            # exception propagates.
            if not self._in_context:
                self.release()

            raise ClientResponseError(
                self.request_info,
                self.history,
                status=self.status,
                message=self.reason,
                headers=self.headers,
            )

    def _release_connection(self) -> None:
        if self._connection is not None:
            if self.__writer is None:
                self._connection.release()
                self._connection = None
            else:
                self.__writer.add_done_callback(lambda f: self._release_connection())

    async def _wait_released(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if (
                    sys.version_info >= (3, 11)
                    and (task := asyncio.current_task())
                    and task.cancelling()
                ):
                    raise
        self._release_connection()

    def _cleanup_writer(self) -> None:
        if self.__writer is not None:
            self.__writer.cancel()
        if self._stream_writer is not None:
            self._output_size = self._stream_writer.output_size
            self._stream_writer = None
        self._session = None

    def _notify_content(self) -> None:
        content = self.content
        if content and content.exception() is None:
            set_exception(content, _CONNECTION_CLOSED_EXCEPTION)
        self._released = True

    async def wait_for_close(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if (
                    sys.version_info >= (3, 11)
                    and (task := asyncio.current_task())
                    and task.cancelling()
                ):
                    raise
        self.release()

    async def read(self) -> bytes:
        """Read response payload."""
        if self._body is None:
            try:
                self._body = await self.content.read()
                for trace in self._traces:
                    await trace.send_response_chunk_received(
                        self.method, self.url, self._body
                    )
            except BaseException:
                self.close()
                raise
        elif self._released:  # Response explicitly released
            raise ClientConnectionError("Connection closed")

        protocol = self._connection and self._connection.protocol
        if protocol is None or not protocol.upgraded:
            await self._wait_released()  # Underlying connection released
        return self._body

    def get_encoding(self) -> str:
        ctype = self.headers.get(hdrs.CONTENT_TYPE, "").lower()
        mimetype = helpers.parse_mimetype(ctype)

        encoding = mimetype.parameters.get("charset")
        if encoding:
            with contextlib.suppress(LookupError, ValueError):
                return codecs.lookup(encoding).name

        if mimetype.type == "application" and (
            mimetype.subtype == "json" or mimetype.subtype == "rdap"
        ):
            # RFC 7159 states that the default encoding is UTF-8.
            # RFC 7483 defines application/rdap+json
            return "utf-8"

        if self._body is None:
            raise RuntimeError(
                "Cannot compute fallback encoding of a not yet read body"
            )

        return self._resolve_charset(self, self._body)

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        """Read response payload and decode."""
        if self._body is None:
            await self.read()

        if encoding is None:
            encoding = self.get_encoding()

        return self._body.decode(encoding, errors=errors)  # type: ignore[union-attr]

    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: JSONDecoder = DEFAULT_JSON_DECODER,
        content_type: str | None = "application/json",
    ) -> Any:
        """Read and decodes JSON response."""
        if self._body is None:
            await self.read()

        if content_type:
            ctype = self.headers.get(hdrs.CONTENT_TYPE, "").lower()
            if not _is_expected_content_type(ctype, content_type):
                raise ContentTypeError(
                    self.request_info,
                    self.history,
                    status=self.status,
                    message=(
                        "Attempt to decode JSON with unexpected mimetype: %s" % ctype
                    ),
                    headers=self.headers,
                )

        stripped = self._body.strip()  # type: ignore[union-attr]
        if not stripped:
            return None

        if encoding is None:
            encoding = self.get_encoding()

        return loads(stripped.decode(encoding))

    async def __aenter__(self) -> "ClientResponse":
        self._in_context = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._in_context = False
        # similar to _RequestContextManager, we do not need to check
        # for exceptions, response object can close connection
        # if state is broken
        self.release()
        await self.wait_for_close()


class ClientRequest:
    GET_METHODS = {
        hdrs.METH_GET,
        hdrs.METH_HEAD,
        hdrs.METH_OPTIONS,
        hdrs.METH_TRACE,
    }
    POST_METHODS = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT}
    ALL_METHODS = GET_METHODS.union(POST_METHODS).union({hdrs.METH_DELETE})

    DEFAULT_HEADERS = {
        hdrs.ACCEPT: "*/*",
        hdrs.ACCEPT_ENCODING: _gen_default_accept_encoding(),
    }

    # Type of body depends on PAYLOAD_REGISTRY, which is dynamic.
    _body: None | payload.Payload = None
    auth = None
    response = None

    __writer: Optional["asyncio.Task[None]"] = None  # async task for streaming data

    # These class defaults help create_autospec() work correctly.
    # If autospec is improved in future, maybe these can be removed.
    url = URL()
    method = "GET"

    _continue = None  # waiter future for '100 Continue' response

    _upload_tracker: UploadTracker | None = None
    _upload_gen = 0

    _skip_auto_headers: "CIMultiDict[None] | None" = None

    # N.B.
    # Adding __del__ method with self._writer closing doesn't make sense
    # because _writer is instance method, thus it keeps a reference to self.
    # Until writer has finished finalizer will not be called.

    def __init__(
        self,
        method: str,
        url: URL,
        *,
        params: Query = None,
        headers: LooseHeaders | None = None,
        skip_auto_headers: Iterable[str] | None = None,
        data: Any = None,
        cookies: LooseCookies | None = None,
        auth: BasicAuth | None = None,
        version: http.HttpVersion = http.HttpVersion11,
        compress: str | bool | None = None,
        chunked: bool | None = None,
        expect100: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        response_class: type["ClientResponse"] | None = None,
        proxy: URL | None = None,
        proxy_auth: BasicAuth | None = None,
        timer: BaseTimerContext | None = None,
        session: Optional["ClientSession"] = None,
        ssl: SSLContext | bool | Fingerprint = True,
        proxy_headers: LooseHeaders | None = None,
        traces: list["Trace"] | None = None,
        trust_env: bool = False,
        server_hostname: str | None = None,
    ):
        if loop is None:
            loop = asyncio.get_event_loop()
        if match := _CONTAINS_CONTROL_CHAR_RE.search(method):
            raise ValueError(
                f"Method cannot contain non-token characters {method!r} "
                f"(found at least {match.group()!r})"
            )
        # URL forbids subclasses, so a simple type check is enough.
        assert type(url) is URL, url
        if proxy is not None:
            assert type(proxy) is URL, proxy
        # FIXME: session is None in tests only, need to fix tests
        # assert session is not None
        if TYPE_CHECKING:
            assert session is not None
        self._session = session
        if params:
            url = url.extend_query(params)
        self.original_url = url
        self.url = url.with_fragment(None) if url.raw_fragment else url
        self.method = method.upper()
        self.chunked = chunked
        self.compress = compress
        self.loop = loop
        self.length = None
        if response_class is None:
            real_response_class = ClientResponse
        else:
            real_response_class = response_class
        self.response_class: type[ClientResponse] = real_response_class
        self._timer = timer if timer is not None else TimerNoop()
        self._ssl = ssl if ssl is not None else True
        self.server_hostname = server_hostname

        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

        self.update_version(version)
        self.update_host(url)
        self.update_headers(headers)
        self.update_auto_headers(skip_auto_headers)
        self.update_cookies(cookies)
        self.update_content_encoding(data)
        self.update_auth(auth, trust_env)
        self.update_proxy(proxy, proxy_auth, proxy_headers)

        self.update_body_from_data(data)
        if data is not None or self.method not in self.GET_METHODS:
            self.update_transfer_encoding()
        self.update_expect_continue(expect100)
        self._traces = [] if traces is None else traces

    def __reset_writer(self, _: object = None) -> None:
        self.__writer = None

    def _get_content_length(self) -> int | None:
        """Extract and validate Content-Length header value.

        Returns parsed Content-Length value or None if not set.
        Raises ValueError if header exists but cannot be parsed as an integer.
        """
        if hdrs.CONTENT_LENGTH not in self.headers:
            return None

        content_length_hdr = self.headers[hdrs.CONTENT_LENGTH]
        if not _DIGITS_RE.fullmatch(content_length_hdr):
            raise ValueError(f"Invalid Content-Length header: {content_length_hdr!r}")
        return int(content_length_hdr)

    @property
    def skip_auto_headers(self) -> CIMultiDict[None]:
        return self._skip_auto_headers or CIMultiDict()

    @property
    def _writer(self) -> Optional["asyncio.Task[None]"]:
        return self.__writer

    @_writer.setter
    def _writer(self, writer: "asyncio.Task[None]") -> None:
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        writer.add_done_callback(self.__reset_writer)

    def is_ssl(self) -> bool:
        return self.url.scheme in _SSL_SCHEMES

    @property
    def ssl(self) -> Union["SSLContext", bool, Fingerprint]:
        return self._ssl

    @property
    def connection_key(self) -> ConnectionKey:
        if proxy_headers := self.proxy_headers:
            h: int | None = hash(tuple(proxy_headers.items()))
        else:
            h = None
        url = self.url
        return tuple.__new__(
            ConnectionKey,
            (
                url.raw_host or "",
                url.port,
                url.scheme in _SSL_SCHEMES,
                self._ssl,
                self.proxy,
                self.proxy_auth,
                h,
                self.server_hostname,
            ),
        )

    @property
    def host(self) -> str:
        ret = self.url.raw_host
        assert ret is not None
        return ret

    @property
    def port(self) -> int | None:
        return self.url.port

    @property
    def body(self) -> payload.Payload | Literal[b""]:
        """Request body."""
        # empty body is represented as bytes for backwards compatibility
        return self._body or b""

    @body.setter
    def body(self, value: Any) -> None:
        """Set request body with warning for non-autoclose payloads.

        WARNING: This setter must be called from within an event loop and is not
        thread-safe. Setting body outside of an event loop may raise RuntimeError
        when closing file-based payloads.

        DEPRECATED: Direct assignment to body is deprecated and will be removed
        in a future version. Use await update_body() instead for proper resource
        management.
        """
        # Close existing payload if present
        if self._body is not None:
            # Warn if the payload needs manual closing
            # stacklevel=3: user code -> body setter -> _warn_if_unclosed_payload
            _warn_if_unclosed_payload(self._body, stacklevel=3)
            # NOTE: In the future, when we remove sync close support,
            # this setter will need to be removed and only the async
            # update_body() method will be available. For now, we call
            # _close() for backwards compatibility.
            self._body._close()
        self._update_body(value)

    @property
    def request_info(self) -> RequestInfo:
        headers: CIMultiDictProxy[str] = CIMultiDictProxy(self.headers)
        # These are created on every request, so we use a NamedTuple
        # for performance reasons. We don't use the RequestInfo.__new__
        # method because it has a different signature which is provided
        # for backwards compatibility only.
        return tuple.__new__(
            RequestInfo, (self.url, self.method, headers, self.original_url)
        )

    @property
    def session(self) -> "ClientSession":
        """Return the ClientSession instance.

        This property provides access to the ClientSession that initiated
        this request, allowing middleware to make additional requests
        using the same session.
        """
        return self._session

    def update_host(self, url: URL) -> None:
        """Update destination host, port and connection type (ssl)."""
        # get host/port
        if not url.raw_host:
            raise InvalidURL(url)

        # basic auth info
        if url.raw_user or url.raw_password:
            self.auth = _basic_auth_no_warn(url.user or "", url.password or "")

    def update_version(self, version: http.HttpVersion | str) -> None:
        """Convert request version to two elements tuple.

        parser HTTP version '1.1' => (1, 1)
        """
        if isinstance(version, str):
            v = [part.strip() for part in version.split(".", 1)]
            try:
                version = http.HttpVersion(int(v[0]), int(v[1]))
            except ValueError:
                raise ValueError(
                    f"Can not parse http version number: {version}"
                ) from None
        self.version = version

    def update_headers(self, headers: LooseHeaders | None) -> None:
        """Update request headers."""
        self.headers: CIMultiDict[str] = CIMultiDict()

        # Build the host header
        host = self.url.host_port_subcomponent

        # host_port_subcomponent is None when the URL is a relative URL.
        # but we know we do not have a relative URL here.
        assert host is not None
        self.headers[hdrs.HOST] = host

        if not headers:
            return

        if isinstance(headers, (dict, MultiDictProxy, MultiDict)):
            headers = headers.items()

        for key, value in headers:  # type: ignore[str-unpack]
            # A special case for Host header
            if key in hdrs.HOST_ALL:
                self.headers[key] = value
            else:
                self.headers.add(key, value)

    def update_auto_headers(self, skip_auto_headers: Iterable[str] | None) -> None:
        if skip_auto_headers is not None:
            self._skip_auto_headers = CIMultiDict(
                (hdr, None) for hdr in sorted(skip_auto_headers)
            )
            used_headers = self.headers.copy()
            used_headers.extend(self._skip_auto_headers)  # type: ignore[arg-type]
        else:
            # Fast path when there are no headers to skip
            # which is the most common case.
            used_headers = self.headers

        for hdr, val in self.DEFAULT_HEADERS.items():
            if hdr not in used_headers:
                self.headers[hdr] = val

        if hdrs.USER_AGENT not in used_headers:
            self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE

    def update_cookies(self, cookies: LooseCookies | None) -> None:
        """Update request cookies header."""
        if not cookies:
            return

        c = SimpleCookie()
        if hdrs.COOKIE in self.headers:
            # parse_cookie_header for RFC 6265 compliant Cookie header parsing
            c.update(parse_cookie_header(self.headers.get(hdrs.COOKIE, "")))
            del self.headers[hdrs.COOKIE]

        if isinstance(cookies, Mapping):
            iter_cookies = cookies.items()
        else:
            iter_cookies = cookies  # type: ignore[assignment]
        for name, value in iter_cookies:
            if isinstance(value, Morsel):
                # Use helper to preserve coded_value exactly as sent by server
                c[name] = preserve_morsel_with_coded_value(value)
            else:
                c[name] = value  # type: ignore[assignment]

        self.headers[hdrs.COOKIE] = c.output(header="", sep=";").strip()

    def update_content_encoding(self, data: Any) -> None:
        """Set request content encoding."""
        if not data:
            # Don't compress an empty body.
            self.compress = None
            return

        if self.headers.get(hdrs.CONTENT_ENCODING):
            if self.compress:
                raise ValueError(
                    "compress can not be set if Content-Encoding header is set"
                )
        elif self.compress:
            if not isinstance(self.compress, str):
                self.compress = "deflate"
            self.headers[hdrs.CONTENT_ENCODING] = self.compress
            self.chunked = True  # enable chunked, no need to deal with length

    def update_transfer_encoding(self) -> None:
        """Analyze transfer-encoding header."""
        te = self.headers.get(hdrs.TRANSFER_ENCODING, "").lower()

        if "chunked" in te:
            if self.chunked:
                raise ValueError(
                    "chunked can not be set "
                    'if "Transfer-Encoding: chunked" header is set'
                )

        elif self.chunked:
            if hdrs.CONTENT_LENGTH in self.headers:
                raise ValueError(
                    "chunked can not be set if Content-Length header is set"
                )

            self.headers[hdrs.TRANSFER_ENCODING] = "chunked"

    def update_auth(self, auth: BasicAuth | None, trust_env: bool = False) -> None:
        """Set basic auth."""
        if auth is None:
            auth = self.auth
        if auth is None:
            return

        if not isinstance(auth, helpers.BasicAuth):
            raise TypeError("BasicAuth() tuple is required instead")

        self.headers[hdrs.AUTHORIZATION] = auth.encode()

    def update_body_from_data(self, body: Any, _stacklevel: int = 3) -> None:
        """Update request body from data."""
        if self._body is not None:
            _warn_if_unclosed_payload(self._body, stacklevel=_stacklevel)

        if body is None:
            self._body = None
            # Set Content-Length to 0 when body is None for methods that expect a body
            if (
                self.method not in self.GET_METHODS
                and not self.chunked
                and hdrs.CONTENT_LENGTH not in self.headers
            ):
                self.headers[hdrs.CONTENT_LENGTH] = "0"
            return

        # FormData
        maybe_payload = body() if isinstance(body, FormData) else body

        try:
            body_payload = payload.PAYLOAD_REGISTRY.get(maybe_payload, disposition=None)
        except payload.LookupError:
            body_payload = FormData(maybe_payload)()  # type: ignore[arg-type]

        self._body = body_payload
        # enable chunked encoding if needed
        if not self.chunked and hdrs.CONTENT_LENGTH not in self.headers:
            if (size := body_payload.size) is not None:
                self.headers[hdrs.CONTENT_LENGTH] = str(size)
            else:
                self.chunked = True

        # copy payload headers
        assert body_payload.headers
        headers = self.headers
        skip_headers = self._skip_auto_headers
        for key, value in body_payload.headers.items():
            if key in headers or (skip_headers is not None and key in skip_headers):
                continue
            headers[key] = value

    def _update_body(self, body: Any) -> None:
        """Update request body after its already been set."""
        # Remove existing Content-Length header since body is changing
        if hdrs.CONTENT_LENGTH in self.headers:
            del self.headers[hdrs.CONTENT_LENGTH]

        # Remove existing Transfer-Encoding header to avoid conflicts
        if self.chunked and hdrs.TRANSFER_ENCODING in self.headers:
            del self.headers[hdrs.TRANSFER_ENCODING]

        # Now update the body using the existing method
        # Called from _update_body, add 1 to stacklevel from caller
        self.update_body_from_data(body, _stacklevel=4)

        # Update transfer encoding headers if needed (same logic as __init__)
        if body is not None or self.method not in self.GET_METHODS:
            self.update_transfer_encoding()

    async def update_body(self, body: Any) -> None:
        """
        Update request body and close previous payload if needed.

        This method safely updates the request body by first closing any existing
        payload to prevent resource leaks, then setting the new body.

        IMPORTANT: Always use this method instead of setting request.body directly.
        Direct assignment to request.body will leak resources if the previous body
        contains file handles, streams, or other resources that need cleanup.

        Args:
            body: The new body content. Can be:
                - bytes/bytearray: Raw binary data
                - str: Text data (will be encoded using charset from Content-Type)
                - FormData: Form data that will be encoded as multipart/form-data
                - Payload: A pre-configured payload object
                - AsyncIterable: An async iterable of bytes chunks
                - File-like object: Will be read and sent as binary data
                - None: Clears the body

        Usage:
            # CORRECT: Use update_body
            await request.update_body(b"new request data")

            # WRONG: Don't set body directly
            # request.body = b"new request data"  # This will leak resources!

            # Update with form data
            form_data = FormData()
            form_data.add_field('field', 'value')
            await request.update_body(form_data)

            # Clear body
            await request.update_body(None)

        Note:
            This method is async because it may need to close file handles or
            other resources associated with the previous payload. Always await
            this method to ensure proper cleanup.

        Warning:
            Setting request.body directly is highly discouraged and can lead to:
            - Resource leaks (unclosed file handles, streams)
            - Memory leaks (unreleased buffers)
            - Unexpected behavior with streaming payloads

            It is not recommended to change the payload type in middleware. If the
            body was already set (e.g., as bytes), it's best to keep the same type
            rather than converting it (e.g., to str) as this may result in unexpected
            behavior.

        See Also:
            - update_body_from_data: Synchronous body update without cleanup
            - body property: Direct body access (STRONGLY DISCOURAGED)

        """
        # Close existing payload if it exists and needs closing
        if self._body is not None:
            await self._body.close()
        self._update_body(body)

    def update_expect_continue(self, expect: bool = False) -> None:
        if expect:
            self.headers[hdrs.EXPECT] = "100-continue"
        elif (
            hdrs.EXPECT in self.headers
            and self.headers[hdrs.EXPECT].lower() == "100-continue"
        ):
            expect = True

        if expect:
            self._continue = self.loop.create_future()

    def update_proxy(
        self,
        proxy: URL | None,
        proxy_auth: BasicAuth | None,
        proxy_headers: LooseHeaders | None,
    ) -> None:
        self.proxy = proxy
        if proxy is None:
            self.proxy_auth = None
            self.proxy_headers = None
            return

        if proxy_auth and not isinstance(proxy_auth, helpers.BasicAuth):
            raise ValueError("proxy_auth must be None or BasicAuth() tuple")
        self.proxy_auth = proxy_auth

        if proxy_headers is not None and not isinstance(
            proxy_headers, (MultiDict, MultiDictProxy)
        ):
            proxy_headers = CIMultiDict(proxy_headers)
        self.proxy_headers = proxy_headers

    async def write_bytes(
        self,
        writer: AbstractStreamWriter,
        conn: "Connection",
        content_length: int | None = None,
    ) -> None:
        """
        Write the request body to the connection stream.

        This method handles writing different types of request bodies:
        1. Payload objects (using their specialized write_with_length method)
        2. Bytes/bytearray objects
        3. Iterable body content

        Args:
            writer: The stream writer to write the body to
            conn: The connection being used for this request
            content_length: Optional maximum number of bytes to write from the body
                            (None means write the entire body)

        The method properly handles:
        - Waiting for 100-Continue responses if required
        - Content length constraints for chunked encoding
        - Error handling for network issues, cancellation, and other exceptions
        - Signaling EOF and timeout management
        - Upload progress reporting to the request's UploadTracker

        Raises:
            ClientOSError: When there's an OS-level error writing the body
            ClientConnectionError: When there's a general connection error
            asyncio.CancelledError: When the operation is cancelled

        """
        tracker = self._upload_tracker
        gen = self._upload_gen
        if tracker is not None:
            writer.on_body_write = functools.partial(tracker._add_bytes, gen)
            tracker._attempt_writing(gen, writer)
        try:
            # 100 response
            if self._continue is not None:
                # Force headers to be sent before waiting for 100-continue
                writer.send_headers()
                await writer.drain()
                await self._continue

            protocol = conn.protocol
            assert protocol is not None
            try:
                # This should be a rare case but the
                # self._body can be set to None while
                # the task is being started or we wait above
                # for the 100-continue response.
                # The more likely case is we have an empty
                # payload, but 100-continue is still expected.
                if self._body is not None:
                    await self._body.write_with_length(writer, content_length)
            except OSError as underlying_exc:
                reraised_exc = underlying_exc

                # Distinguish between timeout and other OS errors for better error reporting
                exc_is_not_timeout = underlying_exc.errno is not None or not isinstance(
                    underlying_exc, asyncio.TimeoutError
                )
                if exc_is_not_timeout:
                    reraised_exc = ClientOSError(
                        underlying_exc.errno,
                        f"Can not write request body for {self.url !s}",
                    )

                set_exception(protocol, reraised_exc, underlying_exc)
                if tracker is not None:
                    tracker._attempt_failed(gen, reraised_exc)
            except Exception as underlying_exc:
                wrapped_exc = ClientConnectionError(
                    "Failed to send bytes into the underlying connection "
                    f"{conn !s}: {underlying_exc!r}",
                )
                set_exception(protocol, wrapped_exc, underlying_exc)
                if tracker is not None:
                    tracker._attempt_failed(gen, wrapped_exc)
            else:
                # Successfully wrote the body, signal EOF and start response timeout
                await writer.write_eof()
                if tracker is not None:
                    tracker._attempt_finished(gen)
                protocol.start_timeout()
        except BaseException as underlying_exc:
            # Cancellation, or a failure escaping the inner handlers
            # (100-continue preamble, write_eof), leaves the body unsent:
            # the connection can't be reused, and the tracker must still
            # be notified. Record the attempt first: the tracker
            # transitions cannot fail, while conn.close() conceivably can.
            if tracker is not None:
                tracker._attempt_failed(
                    gen,
                    (
                        None
                        if isinstance(underlying_exc, asyncio.CancelledError)
                        else underlying_exc
                    ),
                )
            conn.close()
            raise

    async def send(self, conn: "Connection") -> "ClientResponse":
        # Specify request target:
        # - CONNECT request must send authority form URI
        # - not CONNECT proxy must send absolute form URI
        # - most common is origin form URI
        if self.method == hdrs.METH_CONNECT:
            connect_host = self.url.host_subcomponent
            assert connect_host is not None
            path = f"{connect_host}:{self.url.port}"
        elif self.proxy and not self.is_ssl():
            path = str(self.url)
        else:
            path = self.url.raw_path_qs

        protocol = conn.protocol
        assert protocol is not None
        writer = StreamWriter(
            protocol,
            self.loop,
            on_chunk_sent=(
                functools.partial(self._on_chunk_request_sent, self.method, self.url)
                if self._traces
                else None
            ),
            on_headers_sent=(
                functools.partial(self._on_headers_request_sent, self.method, self.url)
                if self._traces
                else None
            ),
        )

        if self.compress:
            writer.enable_compression(self.compress)  # type: ignore[arg-type]

        if self.chunked is not None:
            writer.enable_chunking()

        # set default content-type
        if (
            self.method in self.POST_METHODS
            and (
                self._skip_auto_headers is None
                or hdrs.CONTENT_TYPE not in self._skip_auto_headers
            )
            and hdrs.CONTENT_TYPE not in self.headers
        ):
            self.headers[hdrs.CONTENT_TYPE] = "application/octet-stream"

        v = self.version
        if hdrs.CONNECTION not in self.headers:
            if conn._connector.force_close:
                if v == HttpVersion11:
                    self.headers[hdrs.CONNECTION] = "close"
            elif v == HttpVersion10:
                self.headers[hdrs.CONNECTION] = "keep-alive"

        # status + headers
        status_line = f"{self.method} {path} HTTP/{v.major}.{v.minor}"

        # Buffer headers for potential coalescing with body
        await writer.write_headers(status_line, self.headers)

        task: asyncio.Task[None] | None
        if self._body or self._continue is not None or protocol.writing_paused:
            coro = self.write_bytes(writer, conn, self._get_content_length())
            if sys.version_info >= (3, 14):
                # Try to write bytes immediately to avoid having to schedule
                # the task on the event loop.
                loop = asyncio.get_running_loop()
                if isinstance(loop, BaseEventLoop):
                    task = asyncio.create_task(coro, eager_start=True)
                else:
                    task = asyncio.Task(coro, loop=loop, eager_start=True)
            elif sys.version_info >= (3, 12):
                task = asyncio.Task(
                    coro, loop=asyncio.get_running_loop(), eager_start=True
                )
            else:
                task = asyncio.create_task(coro)
            if task.done():
                task = None
            else:
                self._writer = task
        else:
            # We have nothing to write because
            # - there is no body
            # - the protocol does not have writing paused
            # - we are not waiting for a 100-continue response
            protocol.start_timeout()
            writer.set_eof()
            task = None
            if (tracker := self._upload_tracker) is not None:
                # The request went out without a body-writer task: the
                # dispatched attempt is trivially complete with zero bytes.
                tracker._attempt_finished(self._upload_gen)
        response_class = self.response_class
        assert response_class is not None
        self.response = response_class(
            self.method,
            self.original_url,
            writer=task,
            continue100=self._continue,
            timer=self._timer,
            request_info=self.request_info,
            traces=self._traces,
            loop=self.loop,
            session=self._session,
            stream_writer=writer,
        )
        return self.response

    async def close(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if (
                    sys.version_info >= (3, 11)
                    and (task := asyncio.current_task())
                    and task.cancelling()
                ):
                    raise

    def terminate(self) -> None:
        if self.__writer is not None:
            if not self.loop.is_closed():
                self.__writer.cancel()
            self.__writer.remove_done_callback(self.__reset_writer)
            self.__writer = None

    async def _on_chunk_request_sent(self, method: str, url: URL, chunk: bytes) -> None:
        for trace in self._traces:
            await trace.send_request_chunk_sent(method, url, chunk)

    async def _on_headers_request_sent(
        self, method: str, url: URL, headers: "CIMultiDict[str]"
    ) -> None:
        for trace in self._traces:
            await trace.send_request_headers(method, url, headers)
<|endoftext|>import asyncio
import calendar
import contextlib
import datetime
import heapq
import itertools
import json
import math
import os
import pathlib
import pickle
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from http.cookies import BaseCookie, Morsel, SimpleCookie
from types import MappingProxyType
from typing import Union, cast

from yarl import URL

from ._cookie_helpers import (
    _COOKIE_CTL_RE,
    _encoded_length,
    parse_set_cookie_headers,
    preserve_morsel_with_coded_value,
)
from .abc import AbstractCookieJar, ClearCookiePredicate
from .helpers import is_ip_address
from .log import internal_logger
from .typedefs import LooseCookies, PathLike, StrOrURL

__all__ = ("CookieJar", "DummyCookieJar")


CookieItem = Union[str, "Morsel[str]"]
_JarKey = tuple[str, str]  # (domain, path)
_CookieKey = tuple[str, str, str]  # (domain, path, name)
# A built morsel, its encoded name=value length and the cookie's key.
_MorselEntry = tuple["Morsel[str]", int, _CookieKey]
_UNLIMITED_COOKIE_KEY: _CookieKey = ("", "", "")

# We cache these string methods here as their use is in performance critical code.
_FORMAT_PATH = "{}/{}".format
_FORMAT_DOMAIN_REVERSED = "{1}.{0}".format

# The minimum number of scheduled cookie expirations before we start cleaning up
# the expiration heap. This is a performance optimization to avoid cleaning up the
# heap too often when there are only a few scheduled expirations.
_MIN_SCHEDULED_COOKIE_EXPIRATION = 100
_SIMPLE_COOKIE = SimpleCookie()

# Private, non-configurable limits for the built-in jar.
# Eviction follows Firefox's CookieStorage.
_MAX_COOKIES_PER_DOMAIN = 180
_COOKIE_QUOTA_PER_DOMAIN = 150
_MAX_COOKIES_TOTAL = 3300
_COOKIE_QUOTA_TOTAL = 3000
_MAX_COOKIES_PER_REQUEST = 150
_MAX_COOKIE_HEADER_LENGTH = 8190
_COOKIE_HEADER_PREFIX_LENGTH = len(b"Cookie: ")

# Not persisted; the absolute deadline is saved instead.
_RELATIVE_EXPIRY_ATTRS = frozenset(("max-age", "expires"))


class _RestrictedCookieUnpickler(pickle._Unpickler):
    """A restricted unpickler that only allows cookie-related types.

    This prevents arbitrary code execution when loading pickled cookie data
    from untrusted sources. Only types that are expected in a serialized
    CookieJar are permitted.

    Subclasses :class:`pickle._Unpickler` (the pure-Python implementation)
    rather than :class:`pickle.Unpickler` because the accelerated unpickler
    on some implementations (notably PyPy) does not dispatch through
    :meth:`find_class` overrides.

    See: https://docs.python.org/3/library/pickle.html#restricting-globals
    """

    _ALLOWED_CLASSES: frozenset[tuple[str, str]] = frozenset(
        {
            # Core cookie types
            ("http.cookies", "SimpleCookie"),
            ("http.cookies", "Morsel"),
            # Container types used by CookieJar._cookies
            ("collections", "defaultdict"),
            # builtins that pickle uses for reconstruction
            ("builtins", "tuple"),
            ("builtins", "set"),
            ("builtins", "frozenset"),
            ("builtins", "dict"),
        }
    )

    def find_class(self, module: str, name: str) -> type:
        if (module, name) not in self._ALLOWED_CLASSES:
            raise pickle.UnpicklingError(
                f"Forbidden class: {module}.{name}. "
                "CookieJar.load() only allows cookie-related types for security. "
                "See https://docs.python.org/3/library/pickle.html#restricting-globals"
            )
        return super().find_class(module, name)  # type: ignore[no-any-return]


class CookieJar(AbstractCookieJar):
    """Implements cookie storage adhering to RFC 6265."""

    _limits_enabled = True

    DATE_TOKENS_RE = re.compile(
        r"[\x09\x20-\x2F\x3B-\x40\x5B-\x60\x7B-\x7E]*"
        r"(?P<token>[\x00-\x08\x0A-\x1F\d:a-zA-Z\x7F-\xFF]+)"
    )

    DATE_HMS_TIME_RE = re.compile(r"(\d{1,2}):(\d{1,2}):(\d{1,2})")

    DATE_DAY_OF_MONTH_RE = re.compile(r"(\d{1,2})")

    DATE_MONTH_RE = re.compile(
        "(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)",
        re.I,
    )

    DATE_YEAR_RE = re.compile(r"(\d{2,4})")

    # calendar.timegm() fails for timestamps after datetime.datetime.max
    # Minus one as a loss of precision occurs when timestamp() is called.
    MAX_TIME = (
        int(datetime.datetime.max.replace(tzinfo=datetime.timezone.utc).timestamp()) - 1
    )
    try:
        calendar.timegm(time.gmtime(MAX_TIME))
    except OSError:
        # Hit the maximum representable time on Windows
        # https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/localtime-localtime32-localtime64
        MAX_TIME = calendar.timegm((3000, 12, 31, 23, 59, 59, -1, -1, -1))
    except OverflowError:
        # #4515: datetime.max may not be representable on 32-bit platforms
        MAX_TIME = 2**31 - 1
    # Avoid minuses in the future, 3x faster
    SUB_MAX_TIME = MAX_TIME - 1

    def __init__(
        self,
        *,
        unsafe: bool = False,
        quote_cookie: bool = True,
        treat_as_secure_origin: StrOrURL | Iterable[StrOrURL] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(loop=loop)
        self._cookies: defaultdict[_JarKey, SimpleCookie] = defaultdict(SimpleCookie)
        self._morsel_cache: defaultdict[_JarKey, dict[str, _MorselEntry]] = defaultdict(
            dict
        )
        # Cookie identity is (domain, path, name).
        self._host_only_cookies: set[_CookieKey] = set()
        self._unsafe = unsafe
        self._quote_cookie = quote_cookie
        if treat_as_secure_origin is None:
            self._treat_as_secure_origin: frozenset[URL] = frozenset()
        elif isinstance(treat_as_secure_origin, URL):
            self._treat_as_secure_origin = frozenset({treat_as_secure_origin.origin()})
        elif isinstance(treat_as_secure_origin, str):
            self._treat_as_secure_origin = frozenset(
                {URL(treat_as_secure_origin).origin()}
            )
        else:
            self._treat_as_secure_origin = frozenset(
                {
                    URL(url).origin() if isinstance(url, str) else url.origin()
                    for url in treat_as_secure_origin
                }
            )
        self._expire_heap: list[tuple[float, _CookieKey]] = []
        self._expirations: dict[_CookieKey, float] = {}
        self._access_generations: dict[_CookieKey, int] = {}
        self._next_access_generation = 0
        self._domain_counts: dict[str, int] = {}
        self._cookie_count = 0

    @property
    def unsafe(self) -> bool:
        return self._unsafe

    @property
    def quote_cookie(self) -> bool:
        return self._quote_cookie

    @property
    def treat_as_secure_origin(self) -> frozenset[URL]:
        """Return origins considered secure even over cleartext connections."""
        return self._treat_as_secure_origin

    @property
    def cookies(self) -> MappingProxyType[tuple[str, str], SimpleCookie]:
        """Return the cookies stored in this jar."""
        return MappingProxyType(self._cookies)

    @property
    def host_only_cookies(self) -> frozenset[tuple[str, str, str]]:
        """Return the host-only cookies stored in this jar."""
        return frozenset(self._host_only_cookies)

    def save(self, file_path: PathLike) -> None:
        """Save cookies to a file using JSON format.

        :param file_path: Path to file where cookies will be serialized,
            :class:`str` or :class:`pathlib.Path` instance.
        """
        file_path = pathlib.Path(file_path)
        data: dict[str, dict[str, dict[str, str | bool | float]]] = {}
        for (domain, path), cookie in self._cookies.items():
            key = f"{domain}|{path}"
            data[key] = {}
            for name, morsel in cookie.items():
                morsel_data: dict[str, str | bool | float] = {
                    "key": morsel.key,
                    "value": morsel.value,
                    "coded_value": morsel.coded_value,
                }
                # Skip relative expiry; the absolute deadline is saved below.
                for attr in morsel._reserved:  # type: ignore[attr-defined]
                    if attr in _RELATIVE_EXPIRY_ATTRS:
                        continue
                    attr_val = morsel[attr]
                    if attr_val:
                        morsel_data[attr] = attr_val
                # Persist or it reloads as a domain cookie and leaks to subdomains.
                if (domain, path, name) in self._host_only_cookies:
                    morsel_data["host_only"] = True
                if (exp := self._expirations.get((domain, path, name))) is not None:
                    morsel_data["expires_timestamp"] = exp
                data[key][name] = morsel_data

        # Cookie persistence may include authentication/session tokens.
        # Use 0o600 at creation time to avoid umask-dependent overexposure
        # and enforce least-privilege access to sensitive credential data.
        with open(
            file_path,
            mode="w",
            encoding="utf-8",
            opener=lambda path, flags: os.open(path, flags, 0o600),
        ) as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: PathLike) -> None:
        """Load cookies from a file.

        Tries to load JSON format first. Falls back to loading legacy
        pickle format (using a restricted unpickler) for backward
        compatibility with existing cookie files.

        Replaces the current jar contents; loaded cookies pass through the
        same acceptance rules as :meth:`update_cookies`. A malformed file
        raises and leaves the jar unchanged.

        :param file_path: Path to file from where cookies will be
            imported, :class:`str` or :class:`pathlib.Path` instance.
        """
        file_path = pathlib.Path(file_path)
        # Try JSON format first; only a file that isn't JSON is a legacy pickle.
        try:
            with file_path.open(mode="r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass
        else:
            self._load_json_data(data)
            return
        # Fall back to legacy pickle format with restricted unpickler
        with file_path.open(mode="rb") as f:
            pickled = _RestrictedCookieUnpickler(f).load()
        loaded = self._new_loading_jar()
        for (domain, _), cookies in pickled.items():
            response_url = URL.build(scheme="https", host=domain) if domain else URL()
            for name, morsel in cookies.items():
                loaded._update_cookies({name: morsel}, response_url, copy_morsels=False)
        self._replace_contents(loaded)

    def _new_loading_jar(self) -> "CookieJar":
        return CookieJar(
            unsafe=self._unsafe,
            quote_cookie=self._quote_cookie,
            treat_as_secure_origin=self._treat_as_secure_origin,
            loop=self._loop,
        )

    def _replace_contents(self, loaded: "CookieJar") -> None:
        # Refill the existing mapping so views from the cookies property stay live.
        cookies = self._cookies
        cookies.clear()
        cookies.update(loaded._cookies)
        self.__dict__.update(loaded.__dict__)
        self._cookies = cookies

    def _load_json_data(
        self, data: dict[str, dict[str, dict[str, str | bool | float]]]
    ) -> None:
        """Replace contents, routing cookies through _update_cookies()."""
        # Fill a new jar and adopt its state only once every record is in,
        # so a bad file leaves this one unchanged.
        loaded = self._new_loading_jar()
        for compound_key, cookie_data in data.items():
            domain, path = compound_key.split("|", 1)
            for name, morsel_data in cookie_data.items():
                expiration: float | None = None
                if (raw_expiration := morsel_data.get("expires_timestamp")) is not None:
                    try:
                        expiration = float(raw_expiration)
                        if not math.isfinite(expiration):
                            raise ValueError(expiration)
                    except (OverflowError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Cookie {name!r} in {compound_key!r}: "
                            "expiration timestamp must be a finite number"
                        ) from exc
                morsel: Morsel[str] = Morsel()
                # Use __setstate__ to bypass validation, same pattern
                # used in _build_morsel and _cookie_helpers.
                morsel.__setstate__(  # type: ignore[attr-defined]
                    {
                        "key": morsel_data["key"],
                        "value": morsel_data["value"],
                        "coded_value": morsel_data["coded_value"],
                    }
                )
                # Restore morsel attributes
                for attr in morsel._reserved:  # type: ignore[attr-defined]
                    if attr in morsel_data and attr not in (
                        "key",
                        "value",
                        "coded_value",
                    ):
                        morsel[attr] = morsel_data[attr]
                # Drop the domain so update_cookies() re-marks it host-only.
                if morsel_data.get("host_only"):
                    morsel["domain"] = ""
                response_url = (
                    URL.build(scheme="https", host=domain) if domain else URL()
                )
                loaded._update_cookies(
                    {name: morsel},
                    response_url,
                    copy_morsels=False,
                    expiration=expiration,
                )
        self._replace_contents(loaded)

    def clear(self, predicate: ClearCookiePredicate | None = None) -> None:
        if predicate is None:
            self._expire_heap.clear()
            self._cookies.clear()
            self._morsel_cache.clear()
            self._host_only_cookies.clear()
            self._expirations.clear()
            self._access_generations.clear()
            self._domain_counts.clear()
            self._cookie_count = 0
            return

        now = time.time()
        to_del = [
            key
            for (domain, path), cookie in self._cookies.items()
            for name, morsel in cookie.items()
            if (
                (key := (domain, path, name)) in self._expirations
                and self._expirations[key] <= now
            )
            or predicate(morsel)
        ]
        if to_del:
            self._delete_cookies(to_del)

    def clear_domain(self, domain: str) -> None:
        self.clear(lambda x: self._is_domain_match(domain, x["domain"]))

    def __iter__(self) -> "Iterator[Morsel[str]]":
        self._do_expiration()
        for val in self._cookies.values():
            yield from val.values()

    def __len__(self) -> int:
        """Return number of cookies.

        This function does not iterate self to avoid unnecessary expiration
        checks.
        """
        return self._cookie_count

    def _do_expiration(self) -> None:
        """Remove expired cookies."""
        if not (expire_heap_len := len(self._expire_heap)):
            return

        # If the expiration heap grows larger than the number expirations
        # times two, we clean it up to avoid keeping expired entries in
        # the heap and consuming memory. We guard this with a minimum
        # threshold to avoid cleaning up the heap too often when there are
        # only a few scheduled expirations.
        if (
            expire_heap_len > _MIN_SCHEDULED_COOKIE_EXPIRATION
            and expire_heap_len > len(self._expirations) * 2
        ):
            # Remove any expired entries from the expiration heap
            # that do not match the expiration time in the expirations
            # as it means the cookie has been re-added to the heap
            # with a different expiration time.
            self._expire_heap = [
                entry
                for entry in self._expire_heap
                if self._expirations.get(entry[1]) == entry[0]
            ]
            heapq.heapify(self._expire_heap)

        now = time.time()
        to_del: list[_CookieKey] = []
        # Find any expired cookies and add them to the to-delete list
        while self._expire_heap:
            when, cookie_key = self._expire_heap[0]
            if when > now:
                break
            heapq.heappop(self._expire_heap)
            # Check if the cookie hasn't been re-added to the heap
            # with a different expiration time as it will be removed
            # later when it reaches the top of the heap and its
            # expiration time is met.
            if self._expirations.get(cookie_key) == when:
                to_del.append(cookie_key)

        if to_del:
            self._delete_cookies(to_del)

    def _delete_cookies(self, to_del: list[_CookieKey]) -> None:
        for domain, path, name in to_del:
            cookie_key = (domain, path, name)
            jar_key = (domain, path)
            self._host_only_cookies.discard(cookie_key)
            if (cookies := self._cookies.get(jar_key)) is not None:
                if cookies.pop(name, None) is not None:
                    self._cookie_count -= 1
                    if domain_count := self._domain_counts[domain] - 1:
                        self._domain_counts[domain] = domain_count
                    else:
                        del self._domain_counts[domain]
                if not cookies:
                    del self._cookies[jar_key]
            if (morsels := self._morsel_cache.get(jar_key)) is not None:
                morsels.pop(name, None)
                if not morsels:
                    del self._morsel_cache[jar_key]
            self._expirations.pop(cookie_key, None)
            self._access_generations.pop(cookie_key, None)

    def _make_room(self, domain: str, secure: bool) -> bool:
        """Evict for a new cookie like Firefox; False drops the new cookie.

        Mirrors AddCookie and FindStaleCookies: expired, then non-secure,
        cookies go first, least recently used first; secure cookies are only
        evicted for another secure cookie.
        """
        if self._domain_counts.get(domain, 0) >= _MAX_COOKIES_PER_DOMAIN:
            self._do_expiration()
            count = self._domain_counts.get(domain, 0)
            # Leave room so the domain ends at its quota.
            if (excess := count - _COOKIE_QUOTA_PER_DOMAIN + 1) <= 0:
                return True
            non_secure: list[tuple[int, _CookieKey]] = []
            secure_keys: list[tuple[int, _CookieKey]] = []
            for (cookie_domain, path), cookies in self._cookies.items():
                if cookie_domain != domain:
                    continue
                for name, cookie in cookies.items():
                    cookie_key = (domain, path, name)
                    entry = (self._access_generations.get(cookie_key, 0), cookie_key)
                    (secure_keys if cookie["secure"] else non_secure).append(entry)
            if not (candidates := non_secure):
                if count < _MAX_COOKIES_PER_DOMAIN:  # expiry freed a slot
                    return True
                if not secure:
                    return False
                candidates = secure_keys
            candidates.sort()
            evicted = candidates[:excess]
            self._delete_cookies([key for _, key in evicted])
            internal_logger.debug(
                "Evicted %d cookie(s) for %s over the %d-cookie domain limit",
                len(evicted),
                domain,
                _MAX_COOKIES_PER_DOMAIN,
            )
        else:
            self._purge_cookies()
        return True

    def _purge_cookies(self) -> None:
        """Evict expired, then least recently used, cookies down to the total."""
        self._do_expiration()
        if (excess := self._cookie_count - _COOKIE_QUOTA_TOTAL) <= 0:
            return
        generations = self._access_generations
        self._delete_cookies(sorted(generations, key=generations.__getitem__)[:excess])
        internal_logger.debug(
            "Evicted %d cookie(s) over the %d-cookie total limit",
            excess,
            _MAX_COOKIES_TOTAL,
        )

    def _expire_cookie(self, when: float, domain: str, path: str, name: str) -> None:
        cookie_key = (domain, path, name)
        if self._expirations.get(cookie_key) == when:
            # Avoid adding duplicates to the heap
            return
        heapq.heappush(self._expire_heap, (when, cookie_key))
        self._expirations[cookie_key] = when

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        """Update cookies."""
        self._update_cookies(cookies, response_url, copy_morsels=True)

    def update_cookies_from_headers(
        self, headers: Sequence[str], response_url: URL
    ) -> None:
        """Update cookies from raw Set-Cookie headers."""
        hostname = response_url.raw_host
        if not self._unsafe and is_ip_address(hostname):
            # Reject IP-origin cookies before parsing the fields.
            return
        if headers and (cookies_to_update := parse_set_cookie_headers(headers)):
            # The freshly parsed Morsels are not shared with the caller,
            # so they can be stored and normalized without a defensive copy.
            # The parser only produces cookies that can be sent.
            self._update_cookies(
                cookies_to_update,
                response_url,
                copy_morsels=False,
                check_sendable=False,
            )

    def _update_cookies(
        self,
        cookies: LooseCookies,
        response_url: URL,
        *,
        copy_morsels: bool,
        expiration: float | None = None,
        check_sendable: bool = True,
    ) -> None:
        hostname = response_url.raw_host

        if not self._unsafe and is_ip_address(hostname):
            # Don't accept cookies from IPs
            return

        if isinstance(cookies, Mapping):
            cookies = cookies.items()

        for name, cookie in cookies:
            if not isinstance(cookie, Morsel):
                tmp = SimpleCookie()
                tmp[name] = cookie  # type: ignore[assignment]
                cookie = tmp[name]
            elif copy_morsels:
                # TODO(https://github.com/python/typeshed/pull/16346): Remove cast
                cookie = cast("Morsel[str]", cookie.copy())

            domain = cookie["domain"]

            # ignore domains with trailing dots
            if domain and domain[-1] == ".":
                domain = ""
                del cookie["domain"]

            if domain and domain[0] == ".":
                # Remove leading dot
                domain = domain[1:]
                cookie["domain"] = domain

            if domain and hostname and not self._is_domain_match(domain, hostname):
                # Setting cookies for different domains is not allowed
                continue

            path = cookie["path"]
            if not path or path[0] != "/":
                # Set the cookie's path to the response path
                path = response_url.path
                if not path.startswith("/"):
                    path = "/"
                else:
                    # Cut everything from the last slash to the end
                    path = "/" + path[1 : path.rfind("/")]
                cookie["path"] = path
            path = path.rstrip("/")

            if not domain and hostname is not None:
                host_only = True
                domain = hostname
            else:
                host_only = False

            deadline: float | None = None
            if expiration is not None:
                # Saved jars pass an absolute deadline.
                deadline = expiration
            elif max_age := cookie["max-age"]:
                try:
                    delta_seconds = int(max_age)
                    # https://datatracker.ietf.org/doc/html/rfc6265#section-5.2.2
                    if delta_seconds <= 0:
                        deadline = 0.0
                    else:
                        # Cap first to protect against OverflowError on next line.
                        delta_seconds = min(delta_seconds, self.MAX_TIME)
                        deadline = min(time.time() + delta_seconds, self.MAX_TIME)
                except ValueError:
                    cookie["max-age"] = ""

            elif expires := cookie["expires"]:
                if expire_time := self._parse_date(expires):
                    deadline = expire_time
                else:
                    cookie["expires"] = ""

            if check_sendable and self._limits_enabled and not self._can_send(cookie):
                internal_logger.debug(
                    "Dropped cookie %r for %s: it can't be sent in a Cookie header",
                    name,
                    domain,
                )
                continue

            key = (domain, path)
            cookie_key = (domain, path, name)
            bucket = self._cookies.get(key)
            stored = None if bucket is None else bucket.get(name)
            if (
                stored is None
                and self._limits_enabled
                and (
                    self._domain_counts.get(domain, 0) >= _MAX_COOKIES_PER_DOMAIN
                    or self._cookie_count >= _MAX_COOKIES_TOTAL
                )
                # Like Firefox, a cookie that only deletes never evicts others.
                and (deadline is None or deadline > time.time())
                and not self._make_room(domain, bool(cookie["secure"]))
            ):
                internal_logger.debug(
                    "Dropped cookie %r for %s: its domain is full of Secure cookies",
                    name,
                    domain,
                )
                continue

            if host_only:
                self._host_only_cookies.add(cookie_key)
                cookie["domain"] = domain
            else:
                # A cookie with an explicit Domain attribute replaces any
                # host-only cookie with the same (domain, path, name) identity.
                self._host_only_cookies.discard(cookie_key)
            if deadline is not None:
                self._expire_cookie(deadline, domain, path, name)
            elif self._expirations:
                # RFC 6265 5.3 step 11: a replacement doesn't inherit expiry.
                self._expirations.pop(cookie_key, None)

            if stored != cookie:
                # Don't blow away the cache if the same
                # cookie gets set again
                if stored is None:
                    self._cookie_count += 1
                    self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1
                self._cookies[key][name] = cookie
                if (morsels := self._morsel_cache.get(key)) is not None:
                    morsels.pop(name, None)
            if self._limits_enabled:
                # Like Firefox, each stored cookie gets its own access time.
                self._next_access_generation += 1
                self._access_generations[cookie_key] = self._next_access_generation

        self._do_expiration()

    def filter_cookies(self, request_url: URL = URL()) -> "BaseCookie[str]":
        """Returns this jar's cookies filtered by their attributes."""
        # We always use BaseCookie now since all
        # cookies set on on filtered are fully constructed
        # Morsels, not just names and values.
        filtered: BaseCookie[str] = BaseCookie()
        if not self._cookies:
            # Skip do_expiration() if there are no cookies.
            return filtered
        self._do_expiration()
        if not self._cookies:
            # Skip rest of function if no non-expired cookies.
            return filtered
        if type(request_url) is not URL:
            warnings.warn(
                "filter_cookies expects yarl.URL instances only,"
                f"and will stop working in 4.x, got {type(request_url)}",
                DeprecationWarning,
                stacklevel=2,
            )
            request_url = URL(request_url)
        hostname = request_url.raw_host or ""

        is_not_secure = request_url.scheme not in ("https", "wss")
        if is_not_secure and self._treat_as_secure_origin:
            request_origin = URL()
            with contextlib.suppress(ValueError):
                request_origin = request_url.origin()
            is_not_secure = request_origin not in self._treat_as_secure_origin

        # Last (most specific) matching cookie per name wins.
        selected: dict[str, _MorselEntry] = {}

        # Send shared cookie
        key = ("", "")
        if key in self._cookies:
            morsels = self._morsel_cache[key]
            for name, cookie in self._cookies[key].items():
                if is_not_secure and cookie["secure"]:
                    continue
                if (entry := morsels.get(name)) is None:
                    entry = morsels[name] = self._morsel_entry(key, name, cookie)
                selected[name] = entry

        if not is_ip_address(hostname):
            # Get all the subdomains that might match a cookie (e.g. "foo.bar.com", "bar.com", "com")
            domains: Iterable[str] = itertools.accumulate(
                reversed(hostname.split(".")), _FORMAT_DOMAIN_REVERSED
            )
        elif self._unsafe:
            domains = (hostname,)
        else:
            domains = ()

        # Get all the path prefixes that might match a cookie (e.g. "", "/foo", "/foo/bar")
        paths = itertools.accumulate(request_url.path.split("/"), _FORMAT_PATH)
        # Create every combination of (domain, path) pairs.
        pairs = itertools.product(domains, paths)

        path_len = len(request_url.path)
        # Point 2: https://www.rfc-editor.org/rfc/rfc6265.html#section-5.4
        for p in pairs:
            if p == ("", "") or p not in self._cookies:
                continue
            morsels = self._morsel_cache[p]
            for name, cookie in self._cookies[p].items():
                domain = cookie["domain"]

                if domain != hostname and p + (name,) in self._host_only_cookies:
                    continue

                # Skip edge case when the cookie has a trailing slash but request doesn't.
                if len(cookie["path"]) > path_len:
                    continue

                if is_not_secure and cookie["secure"]:
                    continue

                if (entry := morsels.get(name)) is None:
                    entry = morsels[name] = self._morsel_entry(p, name, cookie)
                # Re-insert so the order tracks specificity, for the header and
                # the over-limit fallback.
                selected.pop(name, None)
                selected[name] = entry

        if self._limits_enabled:
            # Each cookie after the first adds a "; " separator.
            header_length = _COOKIE_HEADER_PREFIX_LENGTH - 2 + 2 * len(selected)
            header_length += sum(entry[1] for entry in selected.values())
            if (
                len(selected) > _MAX_COOKIES_PER_REQUEST
                or header_length > _MAX_COOKIE_HEADER_LENGTH
            ):
                return self._filter_cookies_within_limits(selected, hostname)
            self._touch_cookies([entry[2] for entry in selected.values()])
        # Bypass BaseCookie.__setitem__, which stores Morsel values as-is.
        dict.update(filtered, {name: entry[0] for name, entry in selected.items()})
        return filtered

    def _filter_cookies_within_limits(
        self, selected: dict[str, _MorselEntry], hostname: str
    ) -> "BaseCookie[str]":
        """Admit the most specific cookies first so a host's own cookies win."""
        header_length = _COOKIE_HEADER_PREFIX_LENGTH - 2  # no "; " before the first
        admitted: list[tuple[str, _MorselEntry]] = []
        for name, entry in reversed(selected.items()):
            if len(admitted) >= _MAX_COOKIES_PER_REQUEST:
                break
            if header_length + 2 + entry[1] > _MAX_COOKIE_HEADER_LENGTH:
                continue
            header_length += 2 + entry[1]
            admitted.append((name, entry))

        filtered: BaseCookie[str] = BaseCookie()
        for name, entry in reversed(admitted):
            dict.__setitem__(filtered, name, entry[0])
        self._touch_cookies([entry[2] for _, entry in admitted])
        internal_logger.debug(
            "Omitted %d cookie(s) for %s over the %d-cookie"
            " or %d-octet Cookie header limit",
            len(selected) - len(admitted),
            hostname,
            _MAX_COOKIES_PER_REQUEST,
            _MAX_COOKIE_HEADER_LENGTH,
        )
        return filtered

    @staticmethod
    def _can_send(cookie: Morsel[str]) -> bool:
        """Return whether the cookie can be written to a Cookie header."""
        # Whichever of value and coded_value is sent, CPython builds with the
        # CVE-2026-3644 patch also reject control characters in either.
        text = f"{cookie.key}={cookie.value}{cookie.coded_value}"
        return _COOKIE_CTL_RE.search(text) is None and _encoded_length(text) is not None

    def _morsel_entry(
        self, jar_key: _JarKey, name: str, cookie: Morsel[str]
    ) -> _MorselEntry:
        """Build the morsel to send with its length in the Cookie header."""
        morsel = self._build_morsel(cookie)
        length = _encoded_length(f"{morsel.key}={morsel.coded_value}")
        assert length is not None  # Checked before storing.
        return morsel, length, jar_key + (name,)

    def _touch_cookies(self, cookie_keys: Sequence[_CookieKey]) -> None:
        """Record one access generation for cookies used together, like Firefox."""
        self._next_access_generation += 1
        generation = self._next_access_generation
        generations = self._access_generations
        for cookie_key in cookie_keys:
            generations[cookie_key] = generation

    def _build_morsel(self, cookie: Morsel[str]) -> Morsel[str]:
        """Build a morsel for sending, respecting quote_cookie setting."""
        if self._quote_cookie and cookie.coded_value and cookie.coded_value[0] == '"':
            return preserve_morsel_with_coded_value(cookie)
        morsel: Morsel[str] = Morsel()
        if self._quote_cookie:
            value, coded_value = _SIMPLE_COOKIE.value_encode(cookie.value)
        else:
            coded_value = value = cookie.value
        # We use __setstate__ instead of the public set() API because it allows us to
        # bypass validation and set already validated state. This is more stable than
        # setting protected attributes directly and unlikely to change since it would
        # break pickling.
        morsel.__setstate__({"key": cookie.key, "value": value, "coded_value": coded_value})  # type: ignore[attr-defined]
        return morsel

    @staticmethod
    def _is_domain_match(domain: str, hostname: str) -> bool:
        """Implements domain matching adhering to RFC 6265."""
        if hostname == domain:
            return True

        if not hostname.endswith(domain):
            return False

        non_matching = hostname[: -len(domain)]

        if not non_matching.endswith("."):
            return False

        return not is_ip_address(hostname)

    @classmethod
    def _parse_date(cls, date_str: str) -> int | None:
        """Implements date string parsing adhering to RFC 6265."""
        if not date_str:
            return None

        found_time = False
        found_day = False
        found_month = False
        found_year = False

        hour = minute = second = 0
        day = 0
        month = 0
        year = 0

        for token_match in cls.DATE_TOKENS_RE.finditer(date_str):
            token = token_match.group("token")

            if not found_time:
                time_match = cls.DATE_HMS_TIME_RE.match(token)
                if time_match:
                    found_time = True
                    hour, minute, second = (int(s) for s in time_match.groups())
                    continue

            if not found_day:
                day_match = cls.DATE_DAY_OF_MONTH_RE.match(token)
                if day_match:
                    found_day = True
                    day = int(day_match.group())
                    continue

            if not found_month:
                month_match = cls.DATE_MONTH_RE.match(token)
                if month_match:
                    found_month = True
                    assert month_match.lastindex is not None
                    month = month_match.lastindex
                    continue

            if not found_year:
                year_match = cls.DATE_YEAR_RE.match(token)
                if year_match:
                    found_year = True
                    year = int(year_match.group())

        if 70 <= year <= 99:
            year += 1900
        elif 0 <= year <= 69:
            year += 2000

        if False in (found_day, found_month, found_year, found_time):
            return None

        if not 1 <= day <= 31:
            return None

        if year < 1601 or hour > 23 or minute > 59 or second > 59:
            return None

        return calendar.timegm((year, month, day, hour, minute, second, -1, -1, -1))


class _UnlimitedCookieJar(CookieJar):
    """Jar for per-request cookies, which the jar limits don't apply to."""

    _limits_enabled = False

    def _morsel_entry(
        self, jar_key: _JarKey, name: str, cookie: Morsel[str]
    ) -> _MorselEntry:
        # The length and LRU key only serve the limits. These cookies skip the
        # storage check, so an unencodable one is left for the writer to report.
        return self._build_morsel(cookie), 0, _UNLIMITED_COOKIE_KEY


class DummyCookieJar(AbstractCookieJar):
    """Implements a dummy cookie storage.

    It can be used with the ClientSession when no cookie processing is needed.

    """

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        super().__init__(loop=loop)

    def __iter__(self) -> "Iterator[Morsel[str]]":
        while False:
            yield None

    def __len__(self) -> int:
        return 0

    @property
    def unsafe(self) -> bool:
        return False

    @property
    def quote_cookie(self) -> bool:
        return True

    @property
    def cookies(self) -> MappingProxyType[tuple[str, str], SimpleCookie]:
        """Return an empty mapping."""
        return MappingProxyType({})

    @property
    def host_only_cookies(self) -> frozenset[tuple[str, str, str]]:
        """Return an empty frozenset."""
        return frozenset()

    def clear(self, predicate: ClearCookiePredicate | None = None) -> None:
        pass

    def clear_domain(self, domain: str) -> None:
        pass

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        pass

    def update_cookies_from_headers(
        self, headers: Sequence[str], response_url: URL
    ) -> None:
        """Ignore raw Set-Cookie fields without parsing them."""

    def filter_cookies(self, request_url: URL) -> "BaseCookie[str]":
        return SimpleCookie()
<|endoftext|>import asyncio
from typing import TYPE_CHECKING, Any, cast

from .client_exceptions import ClientConnectionResetError
from .helpers import set_exception
from .tcp_helpers import tcp_nodelay

if TYPE_CHECKING:
    from .http_parser import HttpParser

# Raised by transport.pause_reading()/resume_reading() when the transport
# does not support flow control; safe to ignore.
# NOTE: Catch these with a plain try/except/pass, never contextlib.suppress():
# pause/resume run on the hot read path and suppress() is ~6x slower than
# try/except here (it builds a context manager and unpacks this tuple per call).
PAUSE_RESUME_READING_ERRORS = (AttributeError, NotImplementedError, RuntimeError)


class BaseProtocol(asyncio.Protocol):
    __slots__ = (
        "_loop",
        "_paused",
        "_parser",
        "_drain_waiter",
        "_connection_lost",
        "_reading_paused",
        "_buffer_paused",
        "_upgraded",
        "transport",
    )

    def __init__(
        self, loop: asyncio.AbstractEventLoop, parser: "HttpParser[Any] | None" = None
    ) -> None:
        self._loop: asyncio.AbstractEventLoop = loop
        self._paused = False
        self._drain_waiter: asyncio.Future[None] | None = None
        self._reading_paused = False
        self._buffer_paused = False
        self._parser = parser
        self._upgraded = False

        self.transport: asyncio.Transport | None = None

    @property
    def connected(self) -> bool:
        """Return True if the connection is open."""
        return self.transport is not None

    @property
    def writing_paused(self) -> bool:
        return self._paused

    def pause_writing(self) -> None:
        assert not self._paused
        self._paused = True

    def resume_writing(self) -> None:
        assert self._paused
        self._paused = False

        waiter = self._drain_waiter
        if waiter is not None:
            self._drain_waiter = None
            if not waiter.done():
                waiter.set_result(None)

    def pause_reading(self) -> None:
        self._reading_paused = True
        # Parser shouldn't be paused on websockets.
        if not self._upgraded:
            assert self._parser is not None
            self._parser.pause_reading()
        self._pause_transport_reading()

    def _pause_reading_for_buffer(self) -> None:
        """Hold the transport for a buffer this protocol cannot drain yet.

        One flag covers every buffer a protocol owns; the server already
        multiplexes its message queue and its tail through it. Re-check the
        conditions on resume rather than adding another flag.
        """
        self._buffer_paused = True
        self._pause_transport_reading()

    def _resume_reading_for_buffer(self) -> None:
        """Release that hold, unless flow control is still holding it too."""
        self._buffer_paused = False
        if not self._reading_paused:
            self._resume_transport_reading()

    def _pause_transport_reading(self) -> None:
        if self.transport is not None:
            try:
                self.transport.pause_reading()
            except PAUSE_RESUME_READING_ERRORS:
                # Transport lacks flow control; nothing to pause. Intentionally
                # ignored (see PAUSE_RESUME_READING_ERRORS; do not use suppress).
                pass

    def _resume_transport_reading(self) -> None:
        if self.transport is not None:
            try:
                self.transport.resume_reading()
            except PAUSE_RESUME_READING_ERRORS:
                # Transport lacks flow control; nothing to resume. Intentionally
                # ignored (see PAUSE_RESUME_READING_ERRORS; do not use suppress).
                pass

    def resume_reading(self, resume_parser: bool = True) -> None:
        self._reading_paused = False

        # This will resume parsing any unprocessed data from the last pause.
        if not self._upgraded and resume_parser:
            self.data_received(b"")

        # Reading may have been paused again in the above call if there was a lot of
        # compressed data still pending.
        if not self._reading_paused and not self._buffer_paused:
            self._resume_transport_reading()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        tr = cast(asyncio.Transport, transport)
        tcp_nodelay(tr, True)
        self.transport = tr

    def connection_lost(self, exc: BaseException | None) -> None:
        # Wake up the writer if currently paused.
        self.transport = None
        if not self._paused:
            return
        waiter = self._drain_waiter
        if waiter is None:
            return
        self._drain_waiter = None
        if waiter.done():
            return
        if exc is None:
            waiter.set_result(None)
        else:
            set_exception(
                waiter,
                ConnectionError("Connection lost"),
                exc,
            )

    async def _drain_helper(self) -> None:
        if self.transport is None:
            raise ClientConnectionResetError("Connection lost")
        if not self._paused:
            return
        waiter = self._drain_waiter
        if waiter is None:
            waiter = self._loop.create_future()
            self._drain_waiter = waiter
        await waiter
<|endoftext|>import asyncio
import functools
import random
import socket
import sys
import traceback
import warnings
from asyncio.base_events import BaseEventLoop
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from http import HTTPStatus
from itertools import chain, cycle, islice
from time import monotonic
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import aiohappyeyeballs
from aiohappyeyeballs import AddrInfoType, SocketFactoryType

from . import hdrs, helpers
from .abc import AbstractResolver, ResolveResult
from .client_exceptions import (
    ClientConnectionError,
    ClientConnectorCertificateError,
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientHttpProxyError,
    ClientProxyConnectionError,
    InvalidUrlClientError,
    ServerFingerprintMismatch,
    UnixClientConnectorError,
    cert_errors,
    ssl_errors,
)
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import (
    _SENTINEL,
    ceil_timeout,
    is_canonical_ipv4_address,
    is_ip_address,
    noop,
    sentinel,
    set_exception,
    set_result,
)
from .log import client_logger
from .resolver import DefaultResolver

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    Buffer = Union[bytes, bytearray, "memoryview[int]", "memoryview[bytes]"]

if TYPE_CHECKING:
    import ssl

    SSLContext = ssl.SSLContext
else:
    try:
        import ssl

        SSLContext = ssl.SSLContext
    except ImportError:  # pragma: no cover
        ssl = None  # type: ignore[assignment]
        SSLContext = object  # type: ignore[misc,assignment]

EMPTY_SCHEMA_SET = frozenset({""})
HTTP_SCHEMA_SET = frozenset({"http", "https"})
WS_SCHEMA_SET = frozenset({"ws", "wss"})

HTTP_AND_EMPTY_SCHEMA_SET = HTTP_SCHEMA_SET | EMPTY_SCHEMA_SET
HIGH_LEVEL_SCHEMA_SET = HTTP_AND_EMPTY_SCHEMA_SET | WS_SCHEMA_SET

NEEDS_CLEANUP_CLOSED = (3, 13, 0) <= sys.version_info < (
    3,
    13,
    1,
) or sys.version_info < (3, 12, 8)
# Cleanup closed is no longer needed after https://github.com/python/cpython/pull/118960
# which first appeared in Python 3.12.8 and 3.13.1


__all__ = (
    "BaseConnector",
    "TCPConnector",
    "UnixConnector",
    "NamedPipeConnector",
    "AddrInfoType",
    "SocketFactoryType",
)


if TYPE_CHECKING:
    from .client import ClientTimeout
    from .client_reqrep import ConnectionKey
    from .tracing import Trace


class _DeprecationWaiter:
    __slots__ = ("_awaitable", "_awaited")

    def __init__(self, awaitable: Awaitable[Any]) -> None:
        self._awaitable = awaitable
        self._awaited = False

    def __await__(self) -> Any:
        self._awaited = True
        return self._awaitable.__await__()

    def __del__(self) -> None:
        if not self._awaited:
            warnings.warn(
                "Connector.close() is a coroutine, "
                "please use await connector.close()",
                DeprecationWarning,
            )


async def _wait_for_close(waiters: list[Awaitable[object]]) -> None:
    """Wait for all waiters to finish closing."""
    results = await asyncio.gather(*waiters, return_exceptions=True)
    for res in results:
        if isinstance(res, Exception):
            client_logger.debug("Error while closing connector: %r", res)


class Connection:

    _source_traceback = None

    def __init__(
        self,
        connector: "BaseConnector",
        key: "ConnectionKey",
        protocol: ResponseHandler,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._key = key
        self._connector = connector
        self._loop = loop
        self._protocol: ResponseHandler | None = protocol
        self._callbacks: list[Callable[[], None]] = []

        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

    def __repr__(self) -> str:
        return f"Connection<{self._key}>"

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._protocol is not None:
            kwargs = {"source": self}
            _warnings.warn(f"Unclosed connection {self!r}", ResourceWarning, **kwargs)
            if self._loop.is_closed():
                return

            self._connector._release(self._key, self._protocol, should_close=True)

            context = {"client_connection": self, "message": "Unclosed connection"}
            if self._source_traceback is not None:
                context["source_traceback"] = self._source_traceback
            self._loop.call_exception_handler(context)

    def __bool__(self) -> Literal[True]:
        """Force subclasses to not be falsy, to make checks simpler."""
        return True

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        warnings.warn(
            "connector.loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        return self._loop

    @property
    def transport(self) -> asyncio.Transport | None:
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def protocol(self) -> ResponseHandler | None:
        return self._protocol

    def add_callback(self, callback: Callable[[], None]) -> None:
        if callback is not None:
            self._callbacks.append(callback)

    def _notify_release(self) -> None:
        callbacks, self._callbacks = self._callbacks[:], []

        for cb in callbacks:
            with suppress(Exception):
                cb()

    def close(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(self._key, self._protocol, should_close=True)
            self._protocol = None

    def release(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(self._key, self._protocol)
            self._protocol = None

    @property
    def closed(self) -> bool:
        return self._protocol is None or not self._protocol.is_connected()


class _ConnectTunnelConnection(Connection):
    """Special connection wrapper for CONNECT tunnels that must never be pooled.

    This connection wraps the proxy connection that will be upgraded with TLS.
    It must never be released to the pool because:
    1. Its 'closed' future will never complete, causing session.close() to hang
    2. It represents an intermediate state, not a reusable connection
    3. The real connection (with TLS) will be created separately
    """

    def release(self) -> None:
        """Do nothing - don't pool or close the connection.

        These connections are an intermediate state during the CONNECT tunnel
        setup and will be cleaned up naturally after the TLS upgrade. If they
        were to be pooled, they would never be properly closed, causing
        session.close() to wait forever for their 'closed' future.
        """


class _TransportPlaceholder:
    """placeholder for BaseConnector.connect function"""

    __slots__ = ("closed", "transport")

    def __init__(self, closed_future: asyncio.Future[Exception | None]) -> None:
        """Initialize a placeholder for a transport."""
        self.closed = closed_future
        self.transport = None

    def close(self) -> None:
        """Close the placeholder."""

    def abort(self) -> None:
        """Abort the placeholder (does nothing)."""


class BaseConnector:
    """Base connector class.

    keepalive_timeout - (optional) Keep-alive timeout. None means a
        reusable connection never expires from idling alone (a
        disconnected one is discarded the next time the pool is
        consulted for that host, not proactively). This is different
        from force_close=True, which disables reuse entirely.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    timeout_ceil_threshold - Trigger ceiling of timeout values when
                             it's above timeout_ceil_threshold.
    loop - Optional event loop.
    """

    _closed = True  # prevent AttributeError in __del__ if ctor was failed
    _source_traceback = None

    # abort transport after 2 seconds (cleanup broken connections)
    _cleanup_closed_period = 2.0

    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET

    def __init__(
        self,
        *,
        keepalive_timeout: _SENTINEL | None | float = sentinel,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        timeout_ceil_threshold: float = 5,
    ) -> None:

        if force_close:
            if keepalive_timeout is not None and keepalive_timeout is not sentinel:
                raise ValueError(
                    "keepalive_timeout cannot be set if force_close is True"
                )
            keepalive_timeout = None
        else:
            if keepalive_timeout is sentinel:
                keepalive_timeout = 15.0

        loop = loop or asyncio.get_running_loop()
        self._timeout_ceil_threshold = timeout_ceil_threshold

        self._closed = False
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

        # Connection pool of reusable connections.
        # We use a deque to store connections because it has O(1) popleft()
        # and O(1) append() operations to implement a FIFO queue.
        self._conns: defaultdict[
            ConnectionKey, deque[tuple[ResponseHandler, float]]
        ] = defaultdict(deque)
        self._limit = limit
        self._limit_per_host = limit_per_host
        self._acquired: set[ResponseHandler] = set()
        self._acquired_per_host: defaultdict[ConnectionKey, set[ResponseHandler]] = (
            defaultdict(set)
        )
        self._keepalive_timeout = keepalive_timeout
        self._force_close = force_close

        # {host_key: FIFO list of waiters}
        # The FIFO is implemented with an OrderedDict with None keys because
        # python does not have an ordered set.
        self._waiters: defaultdict[
            ConnectionKey, OrderedDict[asyncio.Future[None], None]
        ] = defaultdict(OrderedDict)

        self._loop = loop
        self._factory = functools.partial(ResponseHandler, loop=loop)

        # start keep-alive connection cleanup task
        self._cleanup_handle: asyncio.TimerHandle | None = None

        # start cleanup closed transports task
        self._cleanup_closed_handle: asyncio.TimerHandle | None = None

        if enable_cleanup_closed and not NEEDS_CLEANUP_CLOSED:
            warnings.warn(
                "enable_cleanup_closed ignored because "
                "https://github.com/python/cpython/pull/118960 is fixed "
                f"in Python version {sys.version_info}",
                DeprecationWarning,
                stacklevel=2,
            )
            enable_cleanup_closed = False

        self._cleanup_closed_disabled = not enable_cleanup_closed
        self._cleanup_closed_transports: list[asyncio.Transport | None] = []
        self._placeholder_future: asyncio.Future[Exception | None] = (
            loop.create_future()
        )
        self._placeholder_future.set_result(None)
        self._cleanup_closed()

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._closed:
            return
        if not self._conns:
            return

        conns = [repr(c) for c in self._conns.values()]

        self._close()

        kwargs = {"source": self}
        _warnings.warn(f"Unclosed connector {self!r}", ResourceWarning, **kwargs)
        context = {
            "connector": self,
            "connections": conns,
            "message": "Unclosed connector",
        }
        if self._source_traceback is not None:
            context["source_traceback"] = self._source_traceback
        self._loop.call_exception_handler(context)

    def __enter__(self) -> "BaseConnector":
        warnings.warn(
            '"with Connector():" is deprecated, '
            'use "async with Connector():" instead',
            DeprecationWarning,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        self._close()

    async def __aenter__(self) -> "BaseConnector":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ) -> None:
        await self.close()

    @property
    def force_close(self) -> bool:
        """Ultimately close connection on releasing if True."""
        return self._force_close

    @property
    def limit(self) -> int:
        """The total number for simultaneous connections.

        If limit is 0 the connector has no limit.
        The default limit size is 100.
        """
        return self._limit

    @property
    def limit_per_host(self) -> int:
        """The limit for simultaneous connections to the same endpoint.

        Endpoints are the same if they are have equal
        (host, port, is_ssl) triple.
        """
        return self._limit_per_host

    def _cleanup(self) -> None:
        """Cleanup unused transports."""
        if self._cleanup_handle:
            self._cleanup_handle.cancel()
            # _cleanup_handle should be unset, otherwise _release() will not
            # recreate it ever!
            self._cleanup_handle = None

        now = monotonic()
        timeout = self._keepalive_timeout
        # weakref_handle() never schedules this method for a None timeout
        assert timeout is not None

        if self._conns:
            connections = defaultdict(deque)
            deadline = now - timeout
            for key, conns in self._conns.items():
                alive: deque[tuple[ResponseHandler, float]] = deque()
                for proto, use_time in conns:
                    if proto.is_connected() and use_time - deadline >= 0:
                        alive.append((proto, use_time))
                        continue
                    transport = proto.transport
                    proto.close()
                    if not self._cleanup_closed_disabled and key.is_ssl:
                        self._cleanup_closed_transports.append(transport)

                if alive:
                    connections[key] = alive

            self._conns = connections

        if self._conns:
            self._cleanup_handle = helpers.weakref_handle(
                self,
                "_cleanup",
                timeout,
                self._loop,
                timeout_ceil_threshold=self._timeout_ceil_threshold,
            )

    def _cleanup_closed(self) -> None:
        """Double confirmation for transport close.

        Some broken ssl servers may leave socket open without proper close.
        """
        if self._cleanup_closed_handle:
            self._cleanup_closed_handle.cancel()

        for transport in self._cleanup_closed_transports:
            if transport is not None:
                transport.abort()

        self._cleanup_closed_transports = []

        if not self._cleanup_closed_disabled:
            self._cleanup_closed_handle = helpers.weakref_handle(
                self,
                "_cleanup_closed",
                self._cleanup_closed_period,
                self._loop,
                timeout_ceil_threshold=self._timeout_ceil_threshold,
            )

    def close(self, *, abort_ssl: bool = False) -> Awaitable[None]:
        """Close all opened transports.

        :param abort_ssl: If True, SSL connections will be aborted immediately
                         without performing the shutdown handshake. This provides
                         faster cleanup at the cost of less graceful disconnection.
        """
        if not (waiters := self._close(abort_ssl=abort_ssl)):
            # If there are no connections to close, we can return a noop
            # awaitable to avoid scheduling a task on the event loop.
            return _DeprecationWaiter(noop())
        coro = _wait_for_close(waiters)
        if sys.version_info >= (3, 12):
            # Optimization for Python 3.12, try to close connections
            # immediately to avoid having to schedule the task on the event loop.
            task = asyncio.Task(coro, loop=self._loop, eager_start=True)
        else:
            task = self._loop.create_task(coro)
        return _DeprecationWaiter(task)

    def _close(self, *, abort_ssl: bool = False) -> list[Awaitable[object]]:
        waiters: list[Awaitable[object]] = []

        if self._closed:
            return waiters

        self._closed = True

        try:
            if self._loop.is_closed():
                return waiters

            # cancel cleanup task
            if self._cleanup_handle:
                self._cleanup_handle.cancel()

            # cancel cleanup close task
            if self._cleanup_closed_handle:
                self._cleanup_closed_handle.cancel()

            for data in self._conns.values():
                for proto, _ in data:
                    if (
                        abort_ssl
                        and proto.transport
                        and proto.transport.get_extra_info("sslcontext") is not None
                    ):
                        proto.abort()
                    else:
                        proto.close()
                    if closed := proto.closed:
                        waiters.append(closed)

            for proto in self._acquired:
                if (
                    abort_ssl
                    and proto.transport
                    and proto.transport.get_extra_info("sslcontext") is not None
                ):
                    proto.abort()
                else:
                    proto.close()
                if closed := proto.closed:
                    waiters.append(closed)

            for transport in self._cleanup_closed_transports:
                if transport is not None:
                    transport.abort()

            return waiters

        finally:
            self._conns.clear()
            self._acquired.clear()
            for keyed_waiters in self._waiters.values():
                for keyed_waiter in keyed_waiters:
                    keyed_waiter.cancel()
            self._waiters.clear()
            self._cleanup_handle = None
            self._cleanup_closed_transports.clear()
            self._cleanup_closed_handle = None

    @property
    def closed(self) -> bool:
        """Is connector closed.

        A readonly property.
        """
        return self._closed

    def _available_connections(self, key: "ConnectionKey") -> int:
        """
        Return number of available connections.

        The limit, limit_per_host and the connection key are taken into account.

        If it returns less than 1 means that there are no connections
        available.
        """
        # check total available connections
        # If there are no limits, this will always return 1
        total_remain = 1

        if self._limit and (total_remain := self._limit - len(self._acquired)) <= 0:
            return total_remain

        # check limit per host
        if host_remain := self._limit_per_host:
            if acquired := self._acquired_per_host.get(key):
                host_remain -= len(acquired)
            if total_remain > host_remain:
                return host_remain

        return total_remain

    def _update_proxy_auth_header_and_build_proxy_req(
        self, req: ClientRequest
    ) -> ClientRequest:
        """Set Proxy-Authorization header for non-SSL proxy requests and builds the proxy request for SSL proxy requests."""
        url = req.proxy
        assert url is not None
        headers: dict[str, str] = {}
        if req.proxy_headers is not None:
            headers = req.proxy_headers  # type: ignore[assignment]
        headers[hdrs.HOST] = req.headers[hdrs.HOST]
        proxy_req = ClientRequest(
            hdrs.METH_GET,
            url,
            headers=headers,
            auth=req.proxy_auth,
            loop=self._loop,
            ssl=req.ssl,
        )
        auth = proxy_req.headers.pop(hdrs.AUTHORIZATION, None)
        if auth is not None:
            if not req.is_ssl():
                req.headers[hdrs.PROXY_AUTHORIZATION] = auth
            else:
                proxy_req.headers[hdrs.PROXY_AUTHORIZATION] = auth
        return proxy_req

    async def connect(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> Connection:
        """Get from pool or create new connection."""
        key = req.connection_key
        if (conn := await self._get(key, traces)) is not None:
            # If we do not have to wait and we can get a connection from the pool
            # we can avoid the timeout ceil logic and directly return the connection
            if req.proxy:
                self._update_proxy_auth_header_and_build_proxy_req(req)
            return conn

        async with ceil_timeout(timeout.connect, timeout.ceil_threshold):
            if self._available_connections(key) <= 0:
                await self._wait_for_available_connection(key, traces)
                if (conn := await self._get(key, traces)) is not None:
                    if req.proxy:
                        self._update_proxy_auth_header_and_build_proxy_req(req)
                    return conn

            placeholder = cast(
                ResponseHandler, _TransportPlaceholder(self._placeholder_future)
            )
            self._acquired.add(placeholder)
            if self._limit_per_host:
                self._acquired_per_host[key].add(placeholder)

            try:
                # Traces are done inside the try block to ensure that the
                # that the placeholder is still cleaned up if an exception
                # is raised.
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_start()
                proto = await self._create_connection(req, traces, timeout)
                if traces:
                    for trace in traces:
                        await trace.send_connection_create_end()
            except BaseException:
                self._release_acquired(key, placeholder)
                raise
            else:
                if self._closed:
                    proto.close()
                    raise ClientConnectionError("Connector is closed.")

        # The connection was successfully created, drop the placeholder
        # and add the real connection to the acquired set. There should
        # be no awaits after the proto is added to the acquired set
        # to ensure that the connection is not left in the acquired set
        # on cancellation.
        self._acquired.remove(placeholder)
        self._acquired.add(proto)
        if self._limit_per_host:
            acquired_per_host = self._acquired_per_host[key]
            acquired_per_host.remove(placeholder)
            acquired_per_host.add(proto)
        return Connection(self, key, proto, self._loop)

    async def _wait_for_available_connection(
        self, key: "ConnectionKey", traces: list["Trace"]
    ) -> None:
        """Wait for an available connection slot."""
        # We loop here because there is a race between
        # the connection limit check and the connection
        # being acquired. If the connection is acquired
        # between the check and the await statement, we
        # need to loop again to check if the connection
        # slot is still available.
        attempts = 0
        while True:
            fut: asyncio.Future[None] = self._loop.create_future()
            keyed_waiters = self._waiters[key]
            keyed_waiters[fut] = None
            if attempts:
                # If we have waited before, we need to move the waiter
                # to the front of the queue as otherwise we might get
                # starved and hit the timeout.
                keyed_waiters.move_to_end(fut, last=False)

            try:
                # Traces happen in the try block to ensure that the
                # the waiter is still cleaned up if an exception is raised.
                if traces:
                    for trace in traces:
                        await trace.send_connection_queued_start()
                await fut
                if traces:
                    for trace in traces:
                        await trace.send_connection_queued_end()
            finally:
                # pop the waiter from the queue if its still
                # there and not already removed by _release_waiter
                keyed_waiters.pop(fut, None)
                if not self._waiters.get(key, True):
                    del self._waiters[key]

            if self._available_connections(key) > 0:
                break
            attempts += 1

    async def _get(
        self, key: "ConnectionKey", traces: list["Trace"]
    ) -> Connection | None:
        """Get next reusable connection for the key or None.

        The connection will be marked as acquired.
        """
        if (conns := self._conns.get(key)) is None:
            return None

        t1 = monotonic()
        while conns:
            proto, t0 = conns.popleft()
            # We will we reuse the connection if its connected and
            # the keepalive timeout has not been exceeded
            if proto.is_connected() and (
                self._keepalive_timeout is None or t1 - t0 <= self._keepalive_timeout
            ):
                if not conns:
                    # The very last connection was reclaimed: drop the key
                    del self._conns[key]
                self._acquired.add(proto)
                if self._limit_per_host:
                    self._acquired_per_host[key].add(proto)
                if traces:
                    for trace in traces:
                        try:
                            await trace.send_connection_reuseconn()
                        except BaseException:
                            self._release_acquired(key, proto)
                            raise
                return Connection(self, key, proto, self._loop)

            # Connection cannot be reused, close it
            transport = proto.transport
            proto.close()
            # only for SSL transports
            if not self._cleanup_closed_disabled and key.is_ssl:
                self._cleanup_closed_transports.append(transport)

        # No more connections: drop the key
        del self._conns[key]
        return None

    def _release_waiter(self) -> None:
        """
        Iterates over all waiters until one to be released is found.

        The one to be released is not finished and
        belongs to a host that has available connections.
        """
        if not self._waiters:
            return

        # Having the dict keys ordered this avoids to iterate
        # at the same order at each call.
        queues = list(self._waiters)
        random.shuffle(queues)

        for key in queues:
            if self._available_connections(key) < 1:
                continue

            waiters = self._waiters[key]
            while waiters:
                waiter, _ = waiters.popitem(last=False)
                if not waiter.done():
                    waiter.set_result(None)
                    return

    def _release_acquired(self, key: "ConnectionKey", proto: ResponseHandler) -> None:
        """Release acquired connection."""
        if self._closed:
            # acquired connection is already released on connector closing
            return

        self._acquired.discard(proto)
        if self._limit_per_host and (conns := self._acquired_per_host.get(key)):
            conns.discard(proto)
            if not conns:
                del self._acquired_per_host[key]
        self._release_waiter()

    def _release(
        self,
        key: "ConnectionKey",
        protocol: ResponseHandler,
        *,
        should_close: bool = False,
    ) -> None:
        if self._closed:
            # acquired connection is already released on connector closing
            return

        self._release_acquired(key, protocol)

        if self._force_close or should_close or protocol.should_close:
            transport = protocol.transport
            protocol.close()

            if key.is_ssl and not self._cleanup_closed_disabled:
                self._cleanup_closed_transports.append(transport)
            return

        self._conns[key].append((protocol, monotonic()))

        if self._cleanup_handle is None:
            self._cleanup_handle = helpers.weakref_handle(
                self,
                "_cleanup",
                self._keepalive_timeout,
                self._loop,
                timeout_ceil_threshold=self._timeout_ceil_threshold,
            )

    async def _create_connection(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        raise NotImplementedError()


class _DNSCacheTable:
    def __init__(self, ttl: float | None = None, max_size: int = 1000) -> None:
        self._addrs_rr: OrderedDict[
            tuple[str, int], tuple[Iterator[ResolveResult], int]
        ] = OrderedDict()
        self._timestamps: dict[tuple[str, int], float] = {}
        self._ttl = ttl
        self._max_size = max_size

    def __contains__(self, host: object) -> bool:
        return host in self._addrs_rr

    def add(self, key: tuple[str, int], addrs: list[ResolveResult]) -> None:
        if key in self._addrs_rr:
            self._addrs_rr.move_to_end(key)

        self._addrs_rr[key] = (cycle(addrs), len(addrs))

        if self._ttl is not None:
            self._timestamps[key] = monotonic()

        if len(self._addrs_rr) > self._max_size:
            oldest_key, _ = self._addrs_rr.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

    def remove(self, key: tuple[str, int]) -> None:
        self._addrs_rr.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self) -> None:
        self._addrs_rr.clear()
        self._timestamps.clear()

    def next_addrs(self, key: tuple[str, int]) -> list[ResolveResult]:
        loop, length = self._addrs_rr[key]
        addrs = list(islice(loop, length))
        # Consume one more element to shift internal state of `cycle`
        next(loop)
        self._addrs_rr.move_to_end(key)
        return addrs

    def expired(self, key: tuple[str, int]) -> bool:
        if self._ttl is None:
            return False

        return self._timestamps[key] + self._ttl < monotonic()


def _make_ssl_context(verified: bool) -> SSLContext:
    """Create SSL context.

    This method is not async-friendly and should be called from a thread
    because it will load certificates from disk and do other blocking I/O.
    """
    if ssl is None:
        # No ssl support
        return None
    if verified:
        sslcontext = ssl.create_default_context()
    else:
        sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        sslcontext.options |= ssl.OP_NO_SSLv2
        sslcontext.options |= ssl.OP_NO_SSLv3
        sslcontext.check_hostname = False
        sslcontext.verify_mode = ssl.CERT_NONE
        sslcontext.options |= ssl.OP_NO_COMPRESSION
        sslcontext.set_default_verify_paths()
    sslcontext.set_alpn_protocols(("http/1.1",))
    return sslcontext


# The default SSLContext objects are created at import time
# since they do blocking I/O to load certificates from disk,
# and imports should always be done before the event loop starts
# or in a thread.
_SSL_CONTEXT_VERIFIED = _make_ssl_context(True)
_SSL_CONTEXT_UNVERIFIED = _make_ssl_context(False)


class TCPConnector(BaseConnector):
    """TCP connector.

    verify_ssl - Set to True to check ssl certifications.
    fingerprint - Pass the binary sha256
        digest of the expected certificate in DER format to verify
        that the certificate the server presents matches. See also
        https://en.wikipedia.org/wiki/HTTP_Public_Key_Pinning
    resolver - Enable DNS lookups and use this
        resolver
    use_dns_cache - Use memory cache for DNS lookups.
    ttl_dns_cache - Max seconds having cached a DNS entry, None forever.
    family - socket address family
    local_addr - local tuple of (host, port) to bind socket to

    keepalive_timeout - (optional) Keep-alive timeout. None means a
        reusable connection never expires from idling alone (a
        disconnected one is discarded the next time the pool is
        consulted for that host, not proactively). This is different
        from force_close=True, which disables reuse entirely.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    happy_eyeballs_delay - This is the “Connection Attempt Delay”
                           as defined in RFC 8305. To disable
                           the happy eyeballs algorithm, set to None.
    interleave - “First Address Family Count” as defined in RFC 8305
    loop - Optional event loop.
    socket_factory - A SocketFactoryType function that, if supplied,
                     will be used to create sockets given an
                     AddrInfoType.
    ssl_shutdown_timeout - DEPRECATED. Will be removed in aiohttp 4.0.
                           Grace period for SSL shutdown handshake on TLS
                           connections. Default is 0 seconds (immediate abort).
                           This parameter allowed for a clean SSL shutdown by
                           notifying the remote peer of connection closure,
                           while avoiding excessive delays during connector cleanup.
                           Note: Only takes effect on Python 3.11+.
    """

    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({"tcp"})

    def __init__(
        self,
        *,
        verify_ssl: bool = True,
        fingerprint: bytes | None = None,
        use_dns_cache: bool = True,
        ttl_dns_cache: int | None = 10,
        dns_cache_max_size: int = 1000,
        family: socket.AddressFamily = socket.AddressFamily.AF_UNSPEC,
        ssl_context: SSLContext | None = None,
        ssl: bool | Fingerprint | SSLContext = True,
        local_addr: tuple[str, int] | None = None,
        resolver: AbstractResolver | None = None,
        keepalive_timeout: None | float | _SENTINEL = sentinel,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        timeout_ceil_threshold: float = 5,
        happy_eyeballs_delay: float | None = 0.25,
        interleave: int | None = None,
        socket_factory: SocketFactoryType | None = None,
        ssl_shutdown_timeout: _SENTINEL | None | float = sentinel,
    ):
        super().__init__(
            keepalive_timeout=keepalive_timeout,
            force_close=force_close,
            limit=limit,
            limit_per_host=limit_per_host,
            enable_cleanup_closed=enable_cleanup_closed,
            loop=loop,
            timeout_ceil_threshold=timeout_ceil_threshold,
        )

        self._ssl = _merge_ssl_params(ssl, verify_ssl, ssl_context, fingerprint)

        self._resolver: AbstractResolver
        if resolver is None:
            self._resolver = DefaultResolver(loop=self._loop)
            self._resolver_owner = True
        else:
            self._resolver = resolver
            self._resolver_owner = False

        self._use_dns_cache = use_dns_cache
        self._cached_hosts = _DNSCacheTable(
            ttl=ttl_dns_cache, max_size=dns_cache_max_size
        )
        self._throttle_dns_futures: dict[tuple[str, int], set[asyncio.Future[None]]] = (
            {}
        )
        self._family = family
        self._local_addr_infos = aiohappyeyeballs.addr_to_addr_infos(local_addr)
        self._happy_eyeballs_delay = happy_eyeballs_delay
        self._interleave = interleave
        self._resolve_host_tasks: set[asyncio.Task[list[ResolveResult]]] = set()
        self._socket_factory = socket_factory
        self._ssl_shutdown_timeout: float | None
        # Handle ssl_shutdown_timeout with warning for Python < 3.11
        if ssl_shutdown_timeout is sentinel:
            self._ssl_shutdown_timeout = 0
        else:
            # Deprecation warning for ssl_shutdown_timeout parameter
            warnings.warn(
                "The ssl_shutdown_timeout parameter is deprecated and will be removed in aiohttp 4.0",
                DeprecationWarning,
                stacklevel=2,
            )
            if (
                sys.version_info < (3, 11)
                and ssl_shutdown_timeout is not None
                and ssl_shutdown_timeout != 0
            ):
                warnings.warn(
                    f"ssl_shutdown_timeout={ssl_shutdown_timeout} is ignored on Python < 3.11; "
                    "only ssl_shutdown_timeout=0 is supported. The timeout will be ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._ssl_shutdown_timeout = ssl_shutdown_timeout

    def _close(self, *, abort_ssl: bool = False) -> list[Awaitable[object]]:
        """Close all ongoing DNS calls."""
        for fut in chain.from_iterable(self._throttle_dns_futures.values()):
            fut.cancel()

        waiters = super()._close(abort_ssl=abort_ssl)

        for t in self._resolve_host_tasks:
            t.cancel()
            waiters.append(t)

        return waiters

    async def close(self, *, abort_ssl: bool = False) -> None:
        """
        Close all opened transports.

        :param abort_ssl: If True, SSL connections will be aborted immediately
                         without performing the shutdown handshake. If False (default),
                         the behavior is determined by ssl_shutdown_timeout:
                         - If ssl_shutdown_timeout=0: connections are aborted
                         - If ssl_shutdown_timeout>0: graceful shutdown is performed
        """
        # Use abort_ssl param if explicitly set, otherwise use ssl_shutdown_timeout default
        await super().close(abort_ssl=abort_ssl or self._ssl_shutdown_timeout == 0)
        if self._resolver_owner:
            await self._resolver.close()

    @property
    def family(self) -> int:
        """Socket family like AF_INET."""
        return self._family

    @property
    def use_dns_cache(self) -> bool:
        """True if local DNS caching is enabled."""
        return self._use_dns_cache

    def clear_dns_cache(self, host: str | None = None, port: int | None = None) -> None:
        """Remove specified host/port or clear all dns local cache."""
        if host is not None and port is not None:
            self._cached_hosts.remove((host, port))
        elif host is not None or port is not None:
            raise ValueError("either both host and port or none of them are allowed")
        else:
            self._cached_hosts.clear()

    async def _resolve_host(
        self, host: str, port: int, traces: Sequence["Trace"] | None = None
    ) -> list[ResolveResult]:
        """Resolve host and return list of addresses."""
        if is_ip_address(host):
            # Reject legacy numeric IPv4 forms (e.g. 2130706433, 127.1) that
            # socket would map onto an address, slipping past a connector-level
            # policy that only sees the raw host.
            if ":" not in host and not is_canonical_ipv4_address(host):
                raise InvalidUrlClientError(host, "is not a canonical IPv4 address")
            return [
                {
                    "hostname": host,
                    "host": host,
                    "port": port,
                    "family": self._family,
                    "proto": 0,
                    "flags": 0,
                }
            ]

        if not self._use_dns_cache:

            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)

            if self._closed:
                raise ClientConnectionError("Connector is closed")

            res = await self._resolver.resolve(host, port, family=self._family)

            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)

            return res

        key = (host, port)
        if key in self._cached_hosts and not self._cached_hosts.expired(key):
            # get result early, before any await (#4014)
            result = self._cached_hosts.next_addrs(key)

            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            return result

        futures: set[asyncio.Future[None]]
        #
        # If multiple connectors are resolving the same host, we wait
        # for the first one to resolve and then use the result for all of them.
        # We use a throttle to ensure that we only resolve the host once
        # and then use the result for all the waiters.
        #
        if key in self._throttle_dns_futures:
            # get futures early, before any await (#4014)
            futures = self._throttle_dns_futures[key]
            future: asyncio.Future[None] = self._loop.create_future()
            futures.add(future)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            try:
                await future
            finally:
                futures.discard(future)
            return self._cached_hosts.next_addrs(key)

        # update dict early, before any await (#4014)
        self._throttle_dns_futures[key] = futures = set()
        # In this case we need to create a task to ensure that we can shield
        # the task from cancellation as cancelling this lookup should not cancel
        # the underlying lookup or else the cancel event will get broadcast to
        # all the waiters across all connections.
        #
        coro = self._resolve_host_with_throttle(key, host, port, futures, traces)
        if sys.version_info >= (3, 14):
            # Try to send immediately to avoid having to schedule the task.
            loop = asyncio.get_running_loop()
            if isinstance(loop, BaseEventLoop):
                resolved_host_task = asyncio.create_task(coro, eager_start=True)
            else:
                resolved_host_task = asyncio.Task(coro, loop=loop, eager_start=True)
        elif sys.version_info >= (3, 12):
            resolved_host_task = asyncio.Task(
                coro, loop=asyncio.get_running_loop(), eager_start=True
            )
        else:
            resolved_host_task = asyncio.create_task(coro)

        if not resolved_host_task.done():
            self._resolve_host_tasks.add(resolved_host_task)
            resolved_host_task.add_done_callback(self._resolve_host_tasks.discard)

        try:
            return await asyncio.shield(resolved_host_task)
        except asyncio.CancelledError:

            def drop_exception(fut: "asyncio.Future[list[ResolveResult]]") -> None:
                with suppress(Exception, asyncio.CancelledError):
                    fut.result()

            resolved_host_task.add_done_callback(drop_exception)
            raise

    async def _resolve_host_with_throttle(
        self,
        key: tuple[str, int],
        host: str,
        port: int,
        futures: set["asyncio.Future[None]"],
        traces: Sequence["Trace"] | None,
    ) -> list[ResolveResult]:
        """Resolve host and set result for all waiters.

        This method must be run in a task and shielded from cancellation
        to avoid cancelling the underlying lookup.
        """
        try:
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_miss(host)

                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)

            addrs = await self._resolver.resolve(host, port, family=self._family)
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)

            self._cached_hosts.add(key, addrs)
            for fut in futures:
                set_result(fut, None)
        except BaseException as e:
            # any DNS exception is set for the waiters to raise the same exception.
            # This coro is always run in task that is shielded from cancellation so
            # we should never be propagating cancellation here.
            for fut in futures:
                set_exception(fut, e)
            raise
        finally:
            self._throttle_dns_futures.pop(key)

        return self._cached_hosts.next_addrs(key)

    async def _create_connection(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        """Create connection.

        Has same keyword arguments as BaseEventLoop.create_connection.
        """
        if req.proxy:
            _, proto = await self._create_proxy_connection(req, traces, timeout)
        else:
            _, proto = await self._create_direct_connection(req, traces, timeout)

        return proto

    def _get_ssl_context(self, req: ClientRequest) -> SSLContext | None:
        """Logic to get the correct SSL context

        0. if req.ssl is false, return None

        1. if ssl_context is specified in req, use it
        2. if _ssl_context is specified in self, use it
        3. otherwise:
            1. if verify_ssl is not specified in req, use self.ssl_context
               (will generate a default context according to self.verify_ssl)
            2. if verify_ssl is True in req, generate a default SSL context
            3. if verify_ssl is False in req, generate a SSL context that
               won't verify
        """
        if not req.is_ssl():
            return None

        if ssl is None:  # pragma: no cover
            raise RuntimeError("SSL is not supported.")
        sslcontext = req.ssl
        if isinstance(sslcontext, ssl.SSLContext):
            return sslcontext
        if sslcontext is not True:
            # not verified or fingerprinted
            return _SSL_CONTEXT_UNVERIFIED
        sslcontext = self._ssl
        if isinstance(sslcontext, ssl.SSLContext):
            return sslcontext
        if sslcontext is not True:
            # not verified or fingerprinted
            return _SSL_CONTEXT_UNVERIFIED
        return _SSL_CONTEXT_VERIFIED

    def _get_fingerprint(self, req: ClientRequest) -> Optional["Fingerprint"]:
        ret = req.ssl
        if isinstance(ret, Fingerprint):
            return ret
        ret = self._ssl
        if isinstance(ret, Fingerprint):
            return ret
        return None

    async def _wrap_create_connection(
        self,
        *args: Any,
        addr_infos: list[AddrInfoType],
        req: ClientRequest,
        timeout: "ClientTimeout",
        client_error: type[Exception] = ClientConnectorError,
        **kwargs: Any,
    ) -> tuple[asyncio.Transport, ResponseHandler]:
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                sock = await aiohappyeyeballs.start_connection(
                    addr_infos=addr_infos,
                    local_addr_infos=self._local_addr_infos,
                    happy_eyeballs_delay=self._happy_eyeballs_delay,
                    interleave=self._interleave,
                    loop=self._loop,
                    socket_factory=self._socket_factory,
                )
                # Add ssl_shutdown_timeout for Python 3.11+ when SSL is used
                if (
                    kwargs.get("ssl")
                    and self._ssl_shutdown_timeout
                    and sys.version_info >= (3, 11)
                ):
                    kwargs["ssl_shutdown_timeout"] = self._ssl_shutdown_timeout
                return await self._loop.create_connection(*args, **kwargs, sock=sock)
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc

    async def _wrap_existing_connection(
        self,
        *args: Any,
        req: ClientRequest,
        timeout: "ClientTimeout",
        client_error: type[Exception] = ClientConnectorError,
        **kwargs: Any,
    ) -> tuple[asyncio.Transport, ResponseHandler]:
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                return await self._loop.create_connection(*args, **kwargs)
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc

    def _fail_on_no_start_tls(self, req: "ClientRequest") -> None:
        """Raise a :py:exc:`RuntimeError` on missing ``start_tls()``.

        It is necessary for TLS-in-TLS so that it is possible to
        send HTTPS queries through HTTPS proxies.

        This doesn't affect regular HTTP requests, though.
        """
        if not req.is_ssl():
            return

        proxy_url = req.proxy
        assert proxy_url is not None
        if proxy_url.scheme != "https":
            return

        self._check_loop_for_start_tls()

    def _check_loop_for_start_tls(self) -> None:
        try:
            self._loop.start_tls
        except AttributeError as attr_exc:
            raise RuntimeError(
                "An HTTPS request is being sent through an HTTPS proxy. "
                "This needs support for TLS in TLS but it is not implemented "
                "in your runtime for the stdlib asyncio.\n\n"
                "Please upgrade to Python 3.11 or higher. For more details, "
                "please see:\n"
                "* https://bugs.python.org/issue37179\n"
                "* https://github.com/python/cpython/pull/28073\n"
                "* https://docs.aiohttp.org/en/stable/"
                "client_advanced.html#proxy-support\n"
                "* https://github.com/aio-libs/aiohttp/discussions/6044\n",
            ) from attr_exc

    def _loop_supports_start_tls(self) -> bool:
        try:
            self._check_loop_for_start_tls()
        except RuntimeError:
            return False
        else:
            return True

    def _warn_about_tls_in_tls(
        self,
        underlying_transport: asyncio.Transport,
        req: ClientRequest,
    ) -> None:
        """Issue a warning if the requested URL has HTTPS scheme."""
        if req.request_info.url.scheme != "https":
            return

        # TLS-in-TLS only applies when the proxy itself is HTTPS.
        # When the proxy is HTTP, start_tls upgrades a plain TCP connection,
        # which is standard TLS and works on all event loops and Python versions.
        if req.proxy is None or req.proxy.scheme != "https":
            return

        # Check if uvloop is being used, which supports TLS in TLS,
        # otherwise assume that asyncio's native transport is being used.
        if type(underlying_transport).__module__.startswith("uvloop"):
            return

        # Support in asyncio was added in Python 3.11 (bpo-44011)
        asyncio_supports_tls_in_tls = sys.version_info >= (3, 11) or getattr(
            underlying_transport,
            "_start_tls_compatible",
            False,
        )

        if asyncio_supports_tls_in_tls:
            return

        warnings.warn(
            "An HTTPS request is being sent through an HTTPS proxy. "
            "This support for TLS in TLS is known to be disabled "
            "in the stdlib asyncio (Python <3.11). This is why you'll probably see "
            "an error in the log below.\n\n"
            "It is possible to enable it via monkeypatching. "
            "For more details, see:\n"
            "* https://bugs.python.org/issue37179\n"
            "* https://github.com/python/cpython/pull/28073\n\n"
            "You can temporarily patch this as follows:\n"
            "* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n"
            "* https://github.com/aio-libs/aiohttp/discussions/6044\n",
            RuntimeWarning,
            source=self,
            # Why `4`? At least 3 of the calls in the stack originate
            # from the methods in this class.
            stacklevel=3,
        )

    async def _start_tls_connection(
        self,
        underlying_transport: asyncio.Transport,
        req: ClientRequest,
        timeout: "ClientTimeout",
        client_error: type[Exception] = ClientConnectorError,
    ) -> tuple[asyncio.BaseTransport, ResponseHandler]:
        """Wrap the raw TCP transport with TLS."""
        tls_proto = self._factory()  # Create a brand new proto for TLS
        sslcontext = self._get_ssl_context(req)
        if TYPE_CHECKING:
            # _start_tls_connection is unreachable in the current code path
            # if sslcontext is None.
            assert sslcontext is not None

        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                try:
                    # ssl_shutdown_timeout is only available in Python 3.11+
                    if sys.version_info >= (3, 11) and self._ssl_shutdown_timeout:
                        tls_transport = await self._loop.start_tls(
                            underlying_transport,
                            tls_proto,
                            sslcontext,
                            server_hostname=req.server_hostname or req.host,
                            ssl_handshake_timeout=timeout.total or None,
                            ssl_shutdown_timeout=self._ssl_shutdown_timeout,
                        )
                    else:
                        tls_transport = await self._loop.start_tls(
                            underlying_transport,
                            tls_proto,
                            sslcontext,
                            server_hostname=req.server_hostname or req.host,
                            ssl_handshake_timeout=timeout.total or None,
                        )
                except BaseException:
                    # We need to close the underlying transport since
                    # `start_tls()` probably failed before it had a
                    # chance to do this:
                    if self._ssl_shutdown_timeout == 0:
                        underlying_transport.abort()
                    else:
                        underlying_transport.close()
                    raise
                if tls_transport is not None:
                    fingerprint = self._get_fingerprint(req)
                    if fingerprint:
                        try:
                            fingerprint.check(tls_transport)
                        except ServerFingerprintMismatch:
                            tls_transport.close()
                            if not self._cleanup_closed_disabled:
                                self._cleanup_closed_transports.append(tls_transport)
                            raise
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc
        except TypeError as type_err:
            # Example cause looks like this:
            # TypeError: transport <asyncio.sslproto._SSLProtocolTransport
            # object at 0x7f760615e460> is not supported by start_tls()

            raise ClientConnectionError(
                "Cannot initialize a TLS-in-TLS connection to host "
                f"{req.host!s}:{req.port:d} through an underlying connection "
                f"to an HTTPS proxy {req.proxy!s} ssl:{req.ssl or 'default'} "
                f"[{type_err!s}]"
            ) from type_err
        else:
            if tls_transport is None:
                msg = "Failed to start TLS (possibly caused by closing transport)"
                raise client_error(req.connection_key, OSError(msg))
            tls_proto.connection_made(
                tls_transport
            )  # Kick the state machine of the new TLS protocol

        return tls_transport, tls_proto

    def _convert_hosts_to_addr_infos(
        self, hosts: list[ResolveResult]
    ) -> list[AddrInfoType]:
        """Converts the list of hosts to a list of addr_infos.

        The list of hosts is the result of a DNS lookup. The list of
        addr_infos is the result of a call to `socket.getaddrinfo()`.
        """
        addr_infos: list[AddrInfoType] = []
        for hinfo in hosts:
            host = hinfo["host"]
            is_ipv6 = ":" in host
            family = socket.AF_INET6 if is_ipv6 else socket.AF_INET
            if self._family and self._family != family:
                continue
            addr = (host, hinfo["port"], 0, 0) if is_ipv6 else (host, hinfo["port"])
            addr_infos.append(
                (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", addr)
            )
        return addr_infos

    async def _create_direct_connection(
        self,
        req: ClientRequest,
        traces: list["Trace"],
        timeout: "ClientTimeout",
        *,
        client_error: type[Exception] = ClientConnectorError,
    ) -> tuple[asyncio.Transport, ResponseHandler]:
        sslcontext = self._get_ssl_context(req)
        fingerprint = self._get_fingerprint(req)

        host = req.url.raw_host
        assert host is not None
        # Replace multiple trailing dots with a single one.
        # A trailing dot is only present for fully-qualified domain names.
        # See https://github.com/aio-libs/aiohttp/pull/7364.
        if host.endswith(".."):
            host = host.rstrip(".") + "."
        port = req.port
        assert port is not None
        try:
            # Cancelling this lookup should not cancel the underlying lookup
            #  or else the cancel event will get broadcast to all the waiters
            #  across all connections.
            hosts = await self._resolve_host(host, port, traces=traces)
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            # in case of proxy it is not ClientProxyConnectionError
            # it is problem of resolving proxy ip itself
            raise ClientConnectorDNSError(req.connection_key, exc) from exc

        last_exc: Exception | None = None
        addr_infos = self._convert_hosts_to_addr_infos(hosts)
        while addr_infos:
            # Strip trailing dots, certificates contain FQDN without dots.
            # See https://github.com/aio-libs/aiohttp/issues/3636
            server_hostname = (
                (req.server_hostname or host).rstrip(".") if sslcontext else None
            )

            try:
                transp, proto = await self._wrap_create_connection(
                    self._factory,
                    timeout=timeout,
                    ssl=sslcontext,
                    addr_infos=addr_infos,
                    server_hostname=server_hostname,
                    req=req,
                    client_error=client_error,
                )
            except (ClientConnectorError, asyncio.TimeoutError) as exc:
                last_exc = exc
                aiohappyeyeballs.pop_addr_infos_interleave(addr_infos, self._interleave)
                continue

            if req.is_ssl() and fingerprint:
                try:
                    fingerprint.check(transp)
                except ServerFingerprintMismatch as exc:
                    transp.close()
                    if not self._cleanup_closed_disabled:
                        self._cleanup_closed_transports.append(transp)
                    last_exc = exc
                    # Remove the bad peer from the list of addr_infos
                    sock: socket.socket = transp.get_extra_info("socket")
                    bad_peer = sock.getpeername()
                    aiohappyeyeballs.remove_addr_infos(addr_infos, bad_peer)
                    continue

            return transp, proto
        else:
            assert last_exc is not None
            raise last_exc

    async def _create_proxy_connection(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> tuple[asyncio.BaseTransport, ResponseHandler]:
        self._fail_on_no_start_tls(req)
        runtime_has_start_tls = self._loop_supports_start_tls()
        proxy_req = self._update_proxy_auth_header_and_build_proxy_req(req)

        # create connection to proxy server
        transport, proto = await self._create_direct_connection(
            proxy_req, [], timeout, client_error=ClientProxyConnectionError
        )

        if req.is_ssl():
            if runtime_has_start_tls:
                self._warn_about_tls_in_tls(transport, req)

            # For HTTPS requests over HTTP proxy
            # we must notify proxy to tunnel connection
            # so we send CONNECT command:
            #   CONNECT www.python.org:443 HTTP/1.1
            #   Host: www.python.org
            #
            # next we must do TLS handshake and so on
            # to do this we must wrap raw socket into secure one
            # asyncio handles this perfectly
            proxy_req.method = hdrs.METH_CONNECT
            proxy_req.url = req.url
            key = req.connection_key._replace(
                proxy=None, proxy_auth=None, proxy_headers_hash=None
            )
            conn = _ConnectTunnelConnection(self, key, proto, self._loop)
            try:
                proxy_resp = await proxy_req.send(conn)
            except BaseException:
                conn.close()
                raise
            try:
                protocol = conn._protocol
                assert protocol is not None

                # read_until_eof=True will ensure the connection isn't closed
                # once the response is received and processed allowing
                # START_TLS to work on the connection below.
                protocol.set_response_params(
                    read_until_eof=runtime_has_start_tls,
                    timeout_ceil_threshold=self._timeout_ceil_threshold,
                )
                resp = await proxy_resp.start(conn)
            except BaseException:
                proxy_resp.close()
                conn.close()
                raise
            else:
                conn._protocol = None
                try:
                    if resp.status != 200:
                        message = resp.reason
                        if message is None:
                            message = HTTPStatus(resp.status).phrase
                        raise ClientHttpProxyError(
                            proxy_resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=message,
                            headers=resp.headers,
                        )
                    if not runtime_has_start_tls:
                        rawsock = transport.get_extra_info("socket", default=None)
                        if rawsock is None:
                            raise RuntimeError(
                                "Transport does not expose socket instance"
                            )
                        # Duplicate the socket, so now we can close proxy transport
                        rawsock = rawsock.dup()
                except BaseException:
                    # It shouldn't be closed in `finally` because it's fed to
                    # `loop.start_tls()` and the docs say not to touch it after
                    # passing there.
                    transport.close()
                    raise
                finally:
                    if not runtime_has_start_tls:
                        transport.close()

                if not runtime_has_start_tls:
                    # HTTP proxy with support for upgrade to HTTPS
                    sslcontext = self._get_ssl_context(req)
                    return await self._wrap_existing_connection(
                        self._factory,
                        timeout=timeout,
                        ssl=sslcontext,
                        sock=rawsock,
                        server_hostname=req.host,
                        req=req,
                    )

                return await self._start_tls_connection(
                    # Access the old transport for the last time before it's
                    # closed and forgotten forever:
                    transport,
                    req=req,
                    timeout=timeout,
                )
            finally:
                proxy_resp.close()

        return transport, proto


class UnixConnector(BaseConnector):
    """Unix socket connector.

    path - Unix socket path.
    keepalive_timeout - (optional) Keep-alive timeout. None means a
        reusable connection never expires from idling alone (a
        disconnected one is discarded the next time the pool is
        consulted for that host, not proactively). This is different
        from force_close=True, which disables reuse entirely.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """

    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({"unix"})

    def __init__(
        self,
        path: str,
        force_close: bool = False,
        keepalive_timeout: _SENTINEL | float | None = sentinel,
        limit: int = 100,
        limit_per_host: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(
            force_close=force_close,
            keepalive_timeout=keepalive_timeout,
            limit=limit,
            limit_per_host=limit_per_host,
            loop=loop,
        )
        self._path = path

    @property
    def path(self) -> str:
        """Path to unix socket."""
        return self._path

    async def _create_connection(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                _, proto = await self._loop.create_unix_connection(
                    self._factory, self._path
                )
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc

        return proto


class NamedPipeConnector(BaseConnector):
    """Named pipe connector.

    Only supported by the proactor event loop.
    See also: https://docs.python.org/3/library/asyncio-eventloop.html

    path - Windows named pipe path.
    keepalive_timeout - (optional) Keep-alive timeout. None means a
        reusable connection never expires from idling alone (a
        disconnected one is discarded the next time the pool is
        consulted for that host, not proactively). This is different
        from force_close=True, which disables reuse entirely.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """

    allowed_protocol_schema_set = HIGH_LEVEL_SCHEMA_SET | frozenset({"npipe"})

    def __init__(
        self,
        path: str,
        force_close: bool = False,
        keepalive_timeout: _SENTINEL | float | None = sentinel,
        limit: int = 100,
        limit_per_host: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(
            force_close=force_close,
            keepalive_timeout=keepalive_timeout,
            limit=limit,
            limit_per_host=limit_per_host,
            loop=loop,
        )
        if not isinstance(
            self._loop,
            asyncio.ProactorEventLoop,  # type: ignore[attr-defined]
        ):
            raise RuntimeError(
                "Named Pipes only available in proactor loop under windows"
            )
        self._path = path

    @property
    def path(self) -> str:
        """Path to the named pipe."""
        return self._path

    async def _create_connection(
        self, req: ClientRequest, traces: list["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                _, proto = await self._loop.create_pipe_connection(  # type: ignore[attr-defined]
                    self._factory, self._path
                )
                # the drain is required so that the connection_made is called
                # and transport is set otherwise it is not set before the
                # `assert conn.transport is not None`
                # in client.py's _request method
                await asyncio.sleep(0)
                # other option is to manually set transport like
                # `proto.transport = trans`
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorError(req.connection_key, exc) from exc

        return cast(ResponseHandler, proto)
<|endoftext|>import asyncio
import sys
import zlib
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Any, Final, Generic, Protocol, TypedDict, TypeVar, cast

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing import Union

    Buffer = Union[bytes, bytearray, "memoryview[int]", "memoryview[bytes]"]

try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli

    HAS_BROTLI = True
except ImportError:  # pragma: no cover
    HAS_BROTLI = False

try:
    if sys.version_info >= (3, 14):
        from compression.zstd import ZstdDecompressor  # noqa: I900
    else:  # TODO(PY314): Remove mentions of backports.zstd across codebase
        from backports.zstd import ZstdDecompressor

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


MAX_SYNC_CHUNK_SIZE = 4096

# Unlimited decompression constants - different libraries use different conventions
ZLIB_MAX_LENGTH_UNLIMITED = 0  # zlib uses 0 to mean unlimited
ZSTD_MAX_LENGTH_UNLIMITED = -1  # zstd uses -1 to mean unlimited

# Concatenated members are decoded through a window that starts small and
# doubles. A fresh decompressor copies everything past the member it decodes
# into unused_data, so handing it the whole remaining buffer at every boundary
# is quadratic over a stream of small members.
MEMBER_WINDOW_MIN = 64
MEMBER_WINDOW_MAX = 65536

# Cap on concatenated members decoded in one call. Real payloads are unlikely
# to have more than a few members.
MAX_DECOMPRESS_MEMBERS = 1024


class TooManyMembersError(ValueError):
    """A stream concatenated more members than the caller allows."""


class ZLibCompressObjProtocol(Protocol):
    def compress(self, data: Buffer) -> bytes: ...
    def flush(self, mode: int = ..., /) -> bytes: ...


class ZLibDecompressObjProtocol(Protocol):
    def decompress(self, data: Buffer, max_length: int = ...) -> bytes: ...
    def flush(self, length: int = ..., /) -> bytes: ...

    @property
    def eof(self) -> bool: ...

    @property
    def unconsumed_tail(self) -> bytes: ...

    @property
    def unused_data(self) -> bytes: ...


class ZLibBackendProtocol(Protocol):
    MAX_WBITS: int
    Z_FULL_FLUSH: int
    Z_SYNC_FLUSH: int
    Z_BEST_SPEED: int
    Z_FINISH: int

    def compressobj(
        self,
        level: int = ...,
        method: int = ...,
        wbits: int = ...,
        memLevel: int = ...,
        strategy: int = ...,
        zdict: Buffer | None = ...,
    ) -> ZLibCompressObjProtocol: ...
    def decompressobj(
        self, wbits: int = ..., zdict: Buffer = ...
    ) -> ZLibDecompressObjProtocol: ...

    def compress(
        self, data: Buffer, /, level: int = ..., wbits: int = ...
    ) -> bytes: ...
    def decompress(
        self, data: Buffer, /, wbits: int = ..., bufsize: int = ...
    ) -> bytes: ...


class CompressObjArgs(TypedDict, total=False):
    wbits: int
    strategy: int
    level: int


class ZLibBackendWrapper:
    def __init__(self, _zlib_backend: ZLibBackendProtocol):
        self._zlib_backend: ZLibBackendProtocol = _zlib_backend

    @property
    def name(self) -> str:
        return getattr(self._zlib_backend, "__name__", "undefined")

    @property
    def MAX_WBITS(self) -> int:
        return self._zlib_backend.MAX_WBITS

    @property
    def Z_FULL_FLUSH(self) -> int:
        return self._zlib_backend.Z_FULL_FLUSH

    @property
    def Z_SYNC_FLUSH(self) -> int:
        return self._zlib_backend.Z_SYNC_FLUSH

    @property
    def Z_BEST_SPEED(self) -> int:
        return self._zlib_backend.Z_BEST_SPEED

    @property
    def Z_FINISH(self) -> int:
        return self._zlib_backend.Z_FINISH

    def compressobj(self, *args: Any, **kwargs: Any) -> ZLibCompressObjProtocol:
        return self._zlib_backend.compressobj(*args, **kwargs)

    def decompressobj(self, *args: Any, **kwargs: Any) -> ZLibDecompressObjProtocol:
        return self._zlib_backend.decompressobj(*args, **kwargs)

    def compress(self, data: Buffer, *args: Any, **kwargs: Any) -> bytes:
        return self._zlib_backend.compress(data, *args, **kwargs)

    def decompress(self, data: Buffer, *args: Any, **kwargs: Any) -> bytes:
        return self._zlib_backend.decompress(data, *args, **kwargs)

    # Everything not explicitly listed in the Protocol we just pass through
    def __getattr__(self, attrname: str) -> Any:
        return getattr(self._zlib_backend, attrname)


ZLibBackend: ZLibBackendWrapper = ZLibBackendWrapper(zlib)


def set_zlib_backend(new_zlib_backend: ZLibBackendProtocol) -> None:
    ZLibBackend._zlib_backend = new_zlib_backend


def encoding_to_mode(
    encoding: str | None = None,
    suppress_deflate_header: bool = False,
) -> int:
    if encoding == "gzip":
        return 16 + ZLibBackend.MAX_WBITS

    return -ZLibBackend.MAX_WBITS if suppress_deflate_header else ZLibBackend.MAX_WBITS


class MemberDecompressObjProtocol(Protocol):
    def decompress(self, data: Buffer, max_length: int = ...) -> bytes: ...

    @property
    def eof(self) -> bool: ...

    @property
    def unused_data(self) -> bytes: ...


_DecompressObjT = TypeVar("_DecompressObjT", bound=MemberDecompressObjProtocol)


class DecompressionBaseHandler(ABC):
    def __init__(
        self,
        executor: Executor | None = None,
        max_sync_chunk_size: int | None = MAX_SYNC_CHUNK_SIZE,
    ):
        """Base class for decompression handlers."""
        self._executor = executor
        self._max_sync_chunk_size = max_sync_chunk_size

    @abstractmethod
    def decompress_sync(
        self, data: Buffer, max_length: int = ZLIB_MAX_LENGTH_UNLIMITED
    ) -> bytes:
        """Decompress the given data."""

    async def decompress(
        self, data: Buffer, max_length: int = ZLIB_MAX_LENGTH_UNLIMITED
    ) -> bytes:
        """Decompress the given data."""
        if (
            self._max_sync_chunk_size is not None
            and len(data) > self._max_sync_chunk_size
        ):
            return await asyncio.get_event_loop().run_in_executor(
                self._executor, self.decompress_sync, data, max_length
            )
        return self.decompress_sync(data, max_length)

    @property
    @abstractmethod
    def data_available(self) -> bool:
        """Return True if more output is available by passing b""."""


class ConcatDecompressionHandler(DecompressionBaseHandler, Generic[_DecompressObjT]):
    """Handler for a codec whose streams may concatenate independent members.

    Concatenated gzip/deflate members and multi-frame zstd
    (https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1) decode the
    same way: a decompressor handles one member, then flags eof and leaves the
    rest of the input in unused_data, so every member after it needs a fresh
    one.
    """

    # Sentinel this codec's decompress() takes to mean "no output limit".
    _unlimited: int
    _decompressor: _DecompressObjT
    # Input a max_length-capped walk stopped short of, fed back on the next call.
    _pending_unused_data: bytes | None = None

    @abstractmethod
    def _new_decompressor(self) -> _DecompressObjT:
        """Return a decompressor for the next member."""

    def _decompress_members(self, first: bytes, max_length: int) -> bytes:
        """Decode the members following the one ``first`` came from."""
        remaining = memoryview(self._decompressor.unused_data)
        parts = [first]
        produced = len(first)
        pos = 0
        window = MEMBER_WINDOW_MIN
        budget = max_length
        members = 1

        while pos < len(remaining):
            if self._decompressor.eof:
                members += 1
                if members > MAX_DECOMPRESS_MEMBERS:
                    raise TooManyMembersError(
                        f"Compressed stream has more than "
                        f"{MAX_DECOMPRESS_MEMBERS} members"
                    )
                # Replace the spent decompressor before the budget check below
                # can break out of the loop: it still lists these bytes in its
                # unused_data and would hand them back on the next call.
                self._decompressor = self._new_decompressor()
                window = MEMBER_WINDOW_MIN
            if max_length != self._unlimited:
                budget = max_length - produced
                if budget <= 0:
                    self._pending_unused_data = bytes(remaining[pos:])
                    break

            end = min(pos + window, len(remaining))
            chunk = self._decompressor.decompress(remaining[pos:end], budget)
            if chunk:
                parts.append(chunk)
                produced += len(chunk)

            if self._decompressor.eof:
                pos = end - len(self._decompressor.unused_data)
            else:
                pos = end
                # Doubling the window on each iteration avoids too many calls
                # when a large member is present, while protecting us from
                # quadratic usage when members are of window+1 length.
                window = min(window * 2, MEMBER_WINDOW_MAX)

        return b"".join(parts)


class ZLibCompressor:
    def __init__(
        self,
        encoding: str | None = None,
        suppress_deflate_header: bool = False,
        level: int | None = None,
        wbits: int | None = None,
        strategy: int | None = None,
        executor: Executor | None = None,
        max_sync_chunk_size: int | None = MAX_SYNC_CHUNK_SIZE,
    ):
        self._executor = executor
        self._max_sync_chunk_size = max_sync_chunk_size
        self._mode = (
            encoding_to_mode(encoding, suppress_deflate_header)
            if wbits is None
            else wbits
        )
        self._zlib_backend: Final = ZLibBackendWrapper(ZLibBackend._zlib_backend)

        kwargs: CompressObjArgs = {}
        kwargs["wbits"] = self._mode
        if strategy is not None:
            kwargs["strategy"] = strategy
        if level is not None:
            kwargs["level"] = level
        self._compressor = self._zlib_backend.compressobj(**kwargs)

    def compress_sync(self, data: Buffer) -> bytes:
        return self._compressor.compress(data)

    async def compress(self, data: Buffer) -> bytes:
        """Compress the data and returned the compressed bytes.

        Note that flush() must be called after the last call to compress()

        If the data size is large than the max_sync_chunk_size, the compression
        will be done in the executor. Otherwise, the compression will be done
        in the event loop.

        **WARNING: This method is NOT cancellation-safe when used with flush().**
        If this operation is cancelled, the compressor state may be corrupted.
        The connection MUST be closed after cancellation to avoid data corruption
        in subsequent compress operations.

        For cancellation-safe compression (e.g., WebSocket), the caller MUST wrap
        compress() + flush() + send operations in a shield and lock to ensure atomicity.
        """
        # For large payloads, offload compression to executor to avoid blocking event loop
        should_use_executor = (
            self._max_sync_chunk_size is not None
            and len(data) > self._max_sync_chunk_size
        )
        if should_use_executor:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._compressor.compress, data
            )
        return self.compress_sync(data)

    def flush(self, mode: int | None = None) -> bytes:
        """Flush the compressor synchronously.

        **WARNING: This method is NOT cancellation-safe when called after compress().**
        The flush() operation accesses shared compressor state. If compress() was
        cancelled, calling flush() may result in corrupted data. The connection MUST
        be closed after compress() cancellation.

        For cancellation-safe compression (e.g., WebSocket), the caller MUST wrap
        compress() + flush() + send operations in a shield and lock to ensure atomicity.
        """
        return self._compressor.flush(
            mode if mode is not None else self._zlib_backend.Z_FINISH
        )


class ZLibDecompressor(ConcatDecompressionHandler[ZLibDecompressObjProtocol]):
    _unlimited = ZLIB_MAX_LENGTH_UNLIMITED

    def __init__(
        self,
        encoding: str | None = None,
        suppress_deflate_header: bool = False,
        executor: Executor | None = None,
        max_sync_chunk_size: int | None = MAX_SYNC_CHUNK_SIZE,
    ):
        super().__init__(executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        self._mode = encoding_to_mode(encoding, suppress_deflate_header)
        self._zlib_backend: Final = ZLibBackendWrapper(ZLibBackend._zlib_backend)
        self._decompressor = self._new_decompressor()
        self._last_empty = False

    def _new_decompressor(self) -> ZLibDecompressObjProtocol:
        return self._zlib_backend.decompressobj(wbits=self._mode)

    def decompress_sync(
        self, data: Buffer, max_length: int = ZLIB_MAX_LENGTH_UNLIMITED
    ) -> bytes:
        if self._pending_unused_data is not None:
            data = self._pending_unused_data + bytes(data)
            self._pending_unused_data = None
        result = self._decompressor.decompress(
            self._decompressor.unconsumed_tail + data, max_length
        )

        # Concatenated gzip/deflate stream: decode the members after this one.
        if self._decompressor.eof and self._decompressor.unused_data:
            result = self._decompress_members(result, max_length)

        # Only way to know that isal has no further data is checking we get no output
        self._last_empty = result == b""

        # Member ended exactly at chunk boundary — no unused_data, but the
        # next feed_data() call would fail on the spent decompressor.
        # Only reset for gzip; deflate's feed_eof() relies on eof=True to
        # confirm the stream is complete.
        if self._decompressor.eof and self._mode > self._zlib_backend.MAX_WBITS:
            self._decompressor = self._new_decompressor()

        return result

    def flush(self, length: int = 0) -> bytes:
        return (
            self._decompressor.flush(length)
            if length > 0
            else self._decompressor.flush()
        )

    @property
    def data_available(self) -> bool:
        return (
            bool(self._decompressor.unconsumed_tail)
            or not self._last_empty
            or self._pending_unused_data is not None
        )

    @property
    def eof(self) -> bool:
        return self._decompressor.eof


class BrotliDecompressor(DecompressionBaseHandler):
    # Supports both 'brotlipy' and 'Brotli' packages
    # since they share an import name. The top branches
    # are for 'brotlipy' and bottom branches for 'Brotli'
    def __init__(
        self,
        executor: Executor | None = None,
        max_sync_chunk_size: int | None = MAX_SYNC_CHUNK_SIZE,
    ) -> None:
        """Decompress data using the Brotli library."""
        if not HAS_BROTLI:
            raise RuntimeError(
                "The brotli decompression is not available. "
                "Please install `Brotli` module"
            )
        self._obj = brotli.Decompressor()
        self._last_empty = False
        super().__init__(executor=executor, max_sync_chunk_size=max_sync_chunk_size)

    def decompress_sync(
        self, data: Buffer, max_length: int = ZLIB_MAX_LENGTH_UNLIMITED
    ) -> bytes:
        """Decompress the given data."""
        if hasattr(self._obj, "decompress"):
            if max_length == ZLIB_MAX_LENGTH_UNLIMITED:
                result = cast(bytes, self._obj.decompress(data))
            else:
                result = cast(bytes, self._obj.decompress(data, max_length))
        else:
            if max_length == ZLIB_MAX_LENGTH_UNLIMITED:
                result = cast(bytes, self._obj.process(data))
            else:
                result = cast(bytes, self._obj.process(data, max_length))
        # Only way to know that brotli has no further data is checking we get no output
        self._last_empty = result == b""
        return result

    def flush(self) -> bytes:
        """Flush the decompressor."""
        if hasattr(self._obj, "flush"):
            return cast(bytes, self._obj.flush())
        return b""

    @property
    def data_available(self) -> bool:
        return not self._obj.is_finished() and not self._last_empty


class ZSTDDecompressor(ConcatDecompressionHandler["ZstdDecompressor"]):
    _unlimited = ZSTD_MAX_LENGTH_UNLIMITED

    def __init__(
        self,
        executor: Executor | None = None,
        max_sync_chunk_size: int | None = MAX_SYNC_CHUNK_SIZE,
    ) -> None:
        if not HAS_ZSTD:
            raise RuntimeError(
                "The zstd decompression is not available. "
                "Please install `backports.zstd` module"
            )
        super().__init__(executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        self._decompressor = self._new_decompressor()

    def _new_decompressor(self) -> "ZstdDecompressor":
        return ZstdDecompressor()

    def decompress_sync(
        self, data: Buffer, max_length: int = ZLIB_MAX_LENGTH_UNLIMITED
    ) -> bytes:
        # zstd uses -1 for unlimited, while zlib uses 0 for unlimited
        # Convert the zlib convention (0=unlimited) to zstd convention (-1=unlimited)
        zstd_max_length = (
            ZSTD_MAX_LENGTH_UNLIMITED
            if max_length == ZLIB_MAX_LENGTH_UNLIMITED
            else max_length
        )
        if self._pending_unused_data is not None:
            data = self._pending_unused_data + data
            self._pending_unused_data = None
        result = self._decompressor.decompress(data, zstd_max_length)

        # Concatenated zstd stream: decode the frames after this one.
        if self._decompressor.eof and self._decompressor.unused_data:
            result = self._decompress_members(result, zstd_max_length)

        # Frame ended exactly at chunk boundary — no unused_data, but the
        # next feed_data() call would fail on the spent decompressor.
        # Prepare a fresh one for the next chunk.
        if self._decompressor.eof:
            self._decompressor = self._new_decompressor()

        return result

    def flush(self) -> bytes:
        return b""

    @property
    def data_available(self) -> bool:
        return (
            not self._decompressor.needs_input and not self._decompressor.eof
        ) or self._pending_unused_data is not None
<|endoftext|>import asyncio
import logging
import os
import socket
import sys
import warnings
from argparse import ArgumentParser
from collections.abc import Awaitable, Callable, Iterable, Iterable as TypingIterable
from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from .abc import AbstractAccessLogger
from .helpers import AppKey, RequestKey, ResponseKey
from .log import access_logger
from .typedefs import PathLike
from .web_app import Application as Application, CleanupError as CleanupError
from .web_exceptions import (
    HTTPAccepted as HTTPAccepted,
    HTTPBadGateway as HTTPBadGateway,
    HTTPBadRequest as HTTPBadRequest,
    HTTPClientError as HTTPClientError,
    HTTPConflict as HTTPConflict,
    HTTPCreated as HTTPCreated,
    HTTPError as HTTPError,
    HTTPException as HTTPException,
    HTTPExpectationFailed as HTTPExpectationFailed,
    HTTPFailedDependency as HTTPFailedDependency,
    HTTPForbidden as HTTPForbidden,
    HTTPFound as HTTPFound,
    HTTPGatewayTimeout as HTTPGatewayTimeout,
    HTTPGone as HTTPGone,
    HTTPInsufficientStorage as HTTPInsufficientStorage,
    HTTPInternalServerError as HTTPInternalServerError,
    HTTPLengthRequired as HTTPLengthRequired,
    HTTPMethodNotAllowed as HTTPMethodNotAllowed,
    HTTPMisdirectedRequest as HTTPMisdirectedRequest,
    HTTPMove as HTTPMove,
    HTTPMovedPermanently as HTTPMovedPermanently,
    HTTPMultipleChoices as HTTPMultipleChoices,
    HTTPNetworkAuthenticationRequired as HTTPNetworkAuthenticationRequired,
    HTTPNoContent as HTTPNoContent,
    HTTPNonAuthoritativeInformation as HTTPNonAuthoritativeInformation,
    HTTPNotAcceptable as HTTPNotAcceptable,
    HTTPNotExtended as HTTPNotExtended,
    HTTPNotFound as HTTPNotFound,
    HTTPNotImplemented as HTTPNotImplemented,
    HTTPNotModified as HTTPNotModified,
    HTTPOk as HTTPOk,
    HTTPPartialContent as HTTPPartialContent,
    HTTPPaymentRequired as HTTPPaymentRequired,
    HTTPPermanentRedirect as HTTPPermanentRedirect,
    HTTPPreconditionFailed as HTTPPreconditionFailed,
    HTTPPreconditionRequired as HTTPPreconditionRequired,
    HTTPProxyAuthenticationRequired as HTTPProxyAuthenticationRequired,
    HTTPRedirection as HTTPRedirection,
    HTTPRequestEntityTooLarge as HTTPRequestEntityTooLarge,
    HTTPRequestHeaderFieldsTooLarge as HTTPRequestHeaderFieldsTooLarge,
    HTTPRequestRangeNotSatisfiable as HTTPRequestRangeNotSatisfiable,
    HTTPRequestTimeout as HTTPRequestTimeout,
    HTTPRequestURITooLong as HTTPRequestURITooLong,
    HTTPResetContent as HTTPResetContent,
    HTTPSeeOther as HTTPSeeOther,
    HTTPServerError as HTTPServerError,
    HTTPServiceUnavailable as HTTPServiceUnavailable,
    HTTPSuccessful as HTTPSuccessful,
    HTTPTemporaryRedirect as HTTPTemporaryRedirect,
    HTTPTooManyRequests as HTTPTooManyRequests,
    HTTPUnauthorized as HTTPUnauthorized,
    HTTPUnavailableForLegalReasons as HTTPUnavailableForLegalReasons,
    HTTPUnprocessableEntity as HTTPUnprocessableEntity,
    HTTPUnsupportedMediaType as HTTPUnsupportedMediaType,
    HTTPUpgradeRequired as HTTPUpgradeRequired,
    HTTPUseProxy as HTTPUseProxy,
    HTTPVariantAlsoNegotiates as HTTPVariantAlsoNegotiates,
    HTTPVersionNotSupported as HTTPVersionNotSupported,
    NotAppKeyWarning as NotAppKeyWarning,
)
from .web_fileresponse import FileResponse as FileResponse
from .web_log import AccessLogger
from .web_middlewares import (
    middleware as middleware,
    normalize_path_middleware as normalize_path_middleware,
)
from .web_protocol import (
    PayloadAccessError as PayloadAccessError,
    RequestHandler as RequestHandler,
    RequestPayloadError as RequestPayloadError,
)
from .web_request import (
    BaseRequest as BaseRequest,
    FileField as FileField,
    Request as Request,
)
from .web_response import (
    ContentCoding as ContentCoding,
    Response as Response,
    StreamResponse as StreamResponse,
    json_bytes_response as json_bytes_response,
    json_response as json_response,
)
from .web_routedef import (
    AbstractRouteDef as AbstractRouteDef,
    RouteDef as RouteDef,
    RouteTableDef as RouteTableDef,
    StaticDef as StaticDef,
    delete as delete,
    get as get,
    head as head,
    options as options,
    patch as patch,
    post as post,
    put as put,
    route as route,
    static as static,
    view as view,
)
from .web_runner import (
    AppRunner as AppRunner,
    BaseRunner as BaseRunner,
    BaseSite as BaseSite,
    GracefulExit as GracefulExit,
    NamedPipeSite as NamedPipeSite,
    ServerRunner as ServerRunner,
    SockSite as SockSite,
    TCPSite as TCPSite,
    UnixSite as UnixSite,
)
from .web_server import Server as Server
from .web_urldispatcher import (
    AbstractResource as AbstractResource,
    AbstractRoute as AbstractRoute,
    DynamicResource as DynamicResource,
    PlainResource as PlainResource,
    PrefixedSubAppResource as PrefixedSubAppResource,
    Resource as Resource,
    ResourceRoute as ResourceRoute,
    StaticResource as StaticResource,
    UrlDispatcher as UrlDispatcher,
    UrlMappingMatchInfo as UrlMappingMatchInfo,
    View as View,
)
from .web_ws import (
    WebSocketReady as WebSocketReady,
    WebSocketResponse as WebSocketResponse,
    WSMsgType as WSMsgType,
)

__all__ = (
    # web_app
    "AppKey",
    "Application",
    "CleanupError",
    # web_exceptions
    "NotAppKeyWarning",
    "HTTPAccepted",
    "HTTPBadGateway",
    "HTTPBadRequest",
    "HTTPClientError",
    "HTTPConflict",
    "HTTPCreated",
    "HTTPError",
    "HTTPException",
    "HTTPExpectationFailed",
    "HTTPFailedDependency",
    "HTTPForbidden",
    "HTTPFound",
    "HTTPGatewayTimeout",
    "HTTPGone",
    "HTTPInsufficientStorage",
    "HTTPInternalServerError",
    "HTTPLengthRequired",
    "HTTPMethodNotAllowed",
    "HTTPMisdirectedRequest",
    "HTTPMove",
    "HTTPMovedPermanently",
    "HTTPMultipleChoices",
    "HTTPNetworkAuthenticationRequired",
    "HTTPNoContent",
    "HTTPNonAuthoritativeInformation",
    "HTTPNotAcceptable",
    "HTTPNotExtended",
    "HTTPNotFound",
    "HTTPNotImplemented",
    "HTTPNotModified",
    "HTTPOk",
    "HTTPPartialContent",
    "HTTPPaymentRequired",
    "HTTPPermanentRedirect",
    "HTTPPreconditionFailed",
    "HTTPPreconditionRequired",
    "HTTPProxyAuthenticationRequired",
    "HTTPRedirection",
    "HTTPRequestEntityTooLarge",
    "HTTPRequestHeaderFieldsTooLarge",
    "HTTPRequestRangeNotSatisfiable",
    "HTTPRequestTimeout",
    "HTTPRequestURITooLong",
    "HTTPResetContent",
    "HTTPSeeOther",
    "HTTPServerError",
    "HTTPServiceUnavailable",
    "HTTPSuccessful",
    "HTTPTemporaryRedirect",
    "HTTPTooManyRequests",
    "HTTPUnauthorized",
    "HTTPUnavailableForLegalReasons",
    "HTTPUnprocessableEntity",
    "HTTPUnsupportedMediaType",
    "HTTPUpgradeRequired",
    "HTTPUseProxy",
    "HTTPVariantAlsoNegotiates",
    "HTTPVersionNotSupported",
    # web_fileresponse
    "FileResponse",
    # web_middlewares
    "middleware",
    "normalize_path_middleware",
    # web_protocol
    "PayloadAccessError",
    "RequestHandler",
    "RequestPayloadError",
    # web_request
    "BaseRequest",
    "FileField",
    "Request",
    "RequestKey",
    # web_response
    "ContentCoding",
    "Response",
    "StreamResponse",
    "json_bytes_response",
    "json_response",
    "ResponseKey",
    # web_routedef
    "AbstractRouteDef",
    "RouteDef",
    "RouteTableDef",
    "StaticDef",
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "route",
    "static",
    "view",
    # web_runner
    "AppRunner",
    "BaseRunner",
    "BaseSite",
    "GracefulExit",
    "ServerRunner",
    "SockSite",
    "TCPSite",
    "UnixSite",
    "NamedPipeSite",
    # web_server
    "Server",
    # web_urldispatcher
    "AbstractResource",
    "AbstractRoute",
    "DynamicResource",
    "PlainResource",
    "PrefixedSubAppResource",
    "Resource",
    "ResourceRoute",
    "StaticResource",
    "UrlDispatcher",
    "UrlMappingMatchInfo",
    "View",
    # web_ws
    "WebSocketReady",
    "WebSocketResponse",
    "WSMsgType",
    # web
    "run_app",
)


if TYPE_CHECKING:
    from ssl import SSLContext
else:
    try:
        from ssl import SSLContext
    except ImportError:  # pragma: no cover
        SSLContext = object  # type: ignore[misc,assignment]

# Only display warning when using -Wdefault, -We, -X dev or similar.
warnings.filterwarnings("ignore", category=NotAppKeyWarning, append=True)

HostSequence = TypingIterable[str]


async def _run_app(
    app: Application | Awaitable[Application],
    *,
    host: str | HostSequence | None = None,
    port: int | None = None,
    path: PathLike | TypingIterable[PathLike] | None = None,
    sock: socket.socket | TypingIterable[socket.socket] | None = None,
    ssl_context: SSLContext | None = None,
    print: Callable[..., None] | None = print,
    backlog: int = 128,
    reuse_address: bool | None = None,
    reuse_port: bool | None = None,
    **kwargs: Any,  # TODO(PY311): Use Unpack
) -> None:
    # An internal function to actually do all dirty job for application running
    if asyncio.iscoroutine(app):
        app = await app

    app = cast(Application, app)

    runner = AppRunner(app, **kwargs)

    await runner.setup()

    sites: list[BaseSite] = []

    try:
        if host is not None:
            if isinstance(host, str):
                sites.append(
                    TCPSite(
                        runner,
                        host,
                        port,
                        ssl_context=ssl_context,
                        backlog=backlog,
                        reuse_address=reuse_address,
                        reuse_port=reuse_port,
                    )
                )
            else:
                for h in host:
                    sites.append(
                        TCPSite(
                            runner,
                            h,
                            port,
                            ssl_context=ssl_context,
                            backlog=backlog,
                            reuse_address=reuse_address,
                            reuse_port=reuse_port,
                        )
                    )
        elif path is None and sock is None or port is not None:
            sites.append(
                TCPSite(
                    runner,
                    port=port,
                    ssl_context=ssl_context,
                    backlog=backlog,
                    reuse_address=reuse_address,
                    reuse_port=reuse_port,
                )
            )

        if path is not None:
            if isinstance(path, (str, os.PathLike)):
                sites.append(
                    UnixSite(
                        runner,
                        path,
                        ssl_context=ssl_context,
                        backlog=backlog,
                    )
                )
            else:
                for p in path:
                    sites.append(
                        UnixSite(
                            runner,
                            p,
                            ssl_context=ssl_context,
                            backlog=backlog,
                        )
                    )

        if sock is not None:
            if not isinstance(sock, Iterable):
                sites.append(
                    SockSite(
                        runner,
                        sock,
                        ssl_context=ssl_context,
                        backlog=backlog,
                    )
                )
            else:
                for s in sock:
                    sites.append(
                        SockSite(
                            runner,
                            s,
                            ssl_context=ssl_context,
                            backlog=backlog,
                        )
                    )
        for site in sites:
            await site.start()

        if print:  # pragma: no branch
            names = sorted(str(s.name) for s in runner.sites)
            print(
                "======== Running on {} ========\n"
                "(Press CTRL+C to quit)".format(", ".join(names))
            )

        # sleep forever by 1 hour intervals,
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def _cancel_tasks(
    to_cancel: set["asyncio.Task[Any]"], loop: asyncio.AbstractEventLoop
) -> None:
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def run_app(
    app: Application | Awaitable[Application],
    *,
    host: str | HostSequence | None = None,
    port: int | None = None,
    path: PathLike | TypingIterable[PathLike] | None = None,
    sock: socket.socket | TypingIterable[socket.socket] | None = None,
    shutdown_timeout: float = 60.0,
    keepalive_timeout: float = 75.0,
    ssl_context: SSLContext | None = None,
    print: Callable[..., None] | None = print,
    backlog: int = 128,
    access_log_class: type[AbstractAccessLogger] = AccessLogger,
    access_log_format: str = AccessLogger.LOG_FORMAT,
    access_log: logging.Logger | None = access_logger,
    handle_signals: bool = True,
    reuse_address: bool | None = None,
    reuse_port: bool | None = None,
    handler_cancellation: bool = False,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: Any,
) -> None:
    """Run an app locally"""
    if loop is None:
        loop = asyncio.new_event_loop()

    # Configure if and only if in debugging mode and using the default logger
    if loop.get_debug() and access_log and access_log.name == "aiohttp.access":
        if access_log.level == logging.NOTSET:
            access_log.setLevel(logging.DEBUG)
        if not access_log.hasHandlers():
            access_log.addHandler(logging.StreamHandler())

    main_task = loop.create_task(
        _run_app(
            app,
            host=host,
            port=port,
            path=path,
            sock=sock,
            shutdown_timeout=shutdown_timeout,
            keepalive_timeout=keepalive_timeout,
            ssl_context=ssl_context,
            print=print,
            backlog=backlog,
            access_log_class=access_log_class,
            access_log_format=access_log_format,
            access_log=access_log,
            handle_signals=handle_signals,
            reuse_address=reuse_address,
            reuse_port=reuse_port,
            handler_cancellation=handler_cancellation,
            **kwargs,
        )
    )

    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_task)
    except (GracefulExit, KeyboardInterrupt):  # pragma: no cover
        pass
    finally:
        try:
            # Skip when ``main_task`` is already done (e.g. raised during startup).
            # Re-running ``loop.run_until_complete`` on a finished task calls
            # ``Future.result`` again, which does
            # ``raise self._exception.with_traceback(self._exception_tb)`` and
            # resets ``exc.__traceback__`` to the originally saved tb — by then
            # shallow — clobbering the deep traceback the caller would otherwise
            # see (frames from ``cleanup_ctx`` / ``on_startup`` and the user code
            # that actually raised).
            if not main_task.done():
                main_task.cancel()
                with suppress(asyncio.CancelledError):
                    loop.run_until_complete(main_task)
        finally:
            _cancel_tasks(asyncio.all_tasks(loop), loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def main(argv: list[str]) -> None:
    arg_parser = ArgumentParser(
        description="aiohttp.web Application server", prog="aiohttp.web"
    )
    arg_parser.add_argument(
        "entry_func",
        help=(
            "Callable returning the `aiohttp.web.Application` instance to "
            "run. Should be specified in the 'module:function' syntax."
        ),
        metavar="entry-func",
    )
    arg_parser.add_argument(
        "-H",
        "--hostname",
        help="TCP/IP hostname to serve on (default: localhost)",
        default=None,
    )
    arg_parser.add_argument(
        "-P",
        "--port",
        help="TCP/IP port to serve on (default: %(default)r)",
        type=int,
        default=8080,
    )
    arg_parser.add_argument(
        "-U",
        "--path",
        help="Unix file system path to serve on. Can be combined with hostname "
        "to serve on both Unix and TCP.",
    )
    args, extra_argv = arg_parser.parse_known_args(argv)

    # Import logic
    mod_str, _, func_str = args.entry_func.partition(":")
    if not func_str or not mod_str:
        arg_parser.error("'entry-func' not in 'module:function' syntax")
    if mod_str.startswith("."):
        arg_parser.error("relative module names not supported")
    try:
        module = import_module(mod_str)
    except ImportError as ex:
        arg_parser.error(f"unable to import {mod_str}: {ex}")
    try:
        func = getattr(module, func_str)
    except AttributeError:
        arg_parser.error(f"module {mod_str!r} has no attribute {func_str!r}")

    # Compatibility logic
    if args.path is not None and not hasattr(socket, "AF_UNIX"):
        arg_parser.error(
            "file system paths not supported by your operating environment"
        )

    logging.basicConfig(level=logging.DEBUG)

    if args.path and args.hostname is None:
        host = port = None
    else:
        host = args.hostname or "localhost"
        port = args.port

    app = func(extra_argv)
    run_app(app, host=host, port=port, path=args.path)
    arg_parser.exit(message="Stopped\n")


if __name__ == "__main__":  # pragma: no branch
    main(sys.argv[1:])  # pragma: no cover
<|endoftext|>"""Utilities shared by tests."""

import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload
from unittest import IsolatedAsyncioTestCase, mock

from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import aiohttp
from aiohttp.client import (
    _BaseRequestContextManager,
    _RequestContextManager,
    _RequestOptions,
    _WSRequestContextManager,
)

from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import StrOrURL
from .web import (
    Application,
    AppRunner,
    BaseRequest,
    BaseRunner,
    Request,
    Server,
    ServerRunner,
    SockSite,
    UrlMappingMatchInfo,
)
from .web_protocol import _RequestHandler

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    SSLContext = Any

if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

_ApplicationNone = TypeVar("_ApplicationNone", Application, None)
_Request = TypeVar("_Request", bound=BaseRequest)

REUSE_ADDRESS = os.name == "posix" and sys.platform != "cygwin"


def get_unused_port_socket(
    host: str, family: socket.AddressFamily = socket.AF_INET
) -> socket.socket:
    return get_port_socket(host, 0, family)


def get_port_socket(
    host: str, port: int, family: socket.AddressFamily
) -> socket.socket:
    s = socket.socket(family, socket.SOCK_STREAM)
    if REUSE_ADDRESS:
        # Windows has different semantics for SO_REUSEADDR,
        # so don't set it. Ref:
        # https://docs.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    return s


def unused_port() -> int:
