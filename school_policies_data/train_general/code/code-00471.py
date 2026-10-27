    _has_connected_to_proxy: bool
    _response_options: _ResponseOptions | None
    _tunnel_host: str | None
    _tunnel_port: int | None
    _tunnel_scheme: str | None

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: None | (
            connection._TYPE_SOCKET_OPTIONS
        ) = default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=Timeout.resolve_default_timeout(timeout),
            source_address=source_address,
            blocksize=blocksize,
        )
        self.socket_options = socket_options
        self.proxy = proxy
        self.proxy_config = proxy_config

        self._has_connected_to_proxy = False
        self._response_options = None
        self._tunnel_host: str | None = None
        self._tunnel_port: int | None = None
        self._tunnel_scheme: str | None = None

    def __str__(self) -> str:
        return f"{type(self).__name__}(host={self.host!r}, port={self.port!r})"

    def __repr__(self) -> str:
        return f"<{self} at {id(self):#x}>"

    @property
    def host(self) -> str:
        """
        Getter method to remove any trailing dots that indicate the hostname is an FQDN.

        In general, SSL certificates don't include the trailing dot indicating a
        fully-qualified domain name, and thus, they don't validate properly when
        checked against a domain name that includes the dot. In addition, some
        servers may not expect to receive the trailing dot when provided.

        However, the hostname with trailing dot is critical to DNS resolution; doing a
        lookup with the trailing dot will properly only resolve the appropriate FQDN,
        whereas a lookup without a trailing dot will search the system's search domain
        list. Thus, it's important to keep the original host around for use only in
        those cases where it's appropriate (i.e., when doing DNS lookup to establish the
        actual TCP connection across which we're going to send HTTP requests).
        """
        return self._dns_host.rstrip(".")

    @host.setter
    def host(self, value: str) -> None:
        """
        Setter for the `host` property.

        We assume that only urllib3 uses the _dns_host attribute; httplib itself
        only uses `host`, and it seems reasonable that other libraries follow suit.
        """
        self._dns_host = value

    def _new_conn(self) -> socket.socket:
        """Establish a socket connection and set nodelay settings on it.

        :return: New socket connection.
        """
        try:
            sock = connection.create_connection(
                (self._dns_host, self.port),
                self.timeout,
                source_address=self.source_address,
                socket_options=self.socket_options,
            )
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e
        except SocketTimeout as e:
            raise ConnectTimeoutError(
                self,
                f"Connection to {self.host} timed out. (connect timeout={self.timeout})",
            ) from e

        except OSError as e:
            raise NewConnectionError(
                self, f"Failed to establish a new connection: {e}"
            ) from e

        sys.audit("http.client.connect", self, self.host, self.port)

        return sock

    def set_tunnel(
        self,
        host: str,
        port: int | None = None,
        headers: typing.Mapping[str, str] | None = None,
        scheme: str = "http",
    ) -> None:
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid proxy scheme for tunneling: {scheme!r}, must be either 'http' or 'https'"
            )
        super().set_tunnel(host, port=port, headers=headers)
        self._tunnel_scheme = scheme

    if sys.version_info < (3, 11, 16) or ((3, 12) <= sys.version_info < (3, 12, 14)):
        # Taken from python/cpython#100986 which was backported in 3.11.9 and 3.12.3.
        # When using connection_from_host, host will come without brackets.
        # With a security patch from python/cpython#146211.
        def _wrap_ipv6(self, ip: bytes) -> bytes:
            if b":" in ip and ip[0] != b"["[0]:
                return b"[" + ip + b"]"
            return ip

        # Copied from CPython 3.12.13 Lib/http/client.py
        _is_legal_header_name = staticmethod(re.compile(rb"[^:\s][^:\r\n]*").fullmatch)
        _is_illegal_header_value = staticmethod(
            re.compile(rb"\n(?![ \t])|\r(?![ \t\n])").search
        )
        _contains_disallowed_url_pchar_re = re.compile("[\x00-\x20\x7f]")

        if sys.version_info < (3, 11, 16):
            # `_tunnel` copied from 3.11.15 backporting
            # https://github.com/python/cpython/commit/0d4026432591d43185568dd31cef6a034c4b9261
            # and https://github.com/python/cpython/commit/6fbc61070fda2ffb8889e77e3b24bca4249ab4d1
            # plus a fix from https://github.com/python/cpython/commit/56b7100b04e44ea27989242b176beb8f016b2c53
            def _tunnel(self) -> None:
                if self._contains_disallowed_url_pchar_re.search(self._tunnel_host):  # type: ignore[arg-type]
                    raise ValueError(
                        "Tunnel host can't contain control characters %r"
                        % (self._tunnel_host,)
                    )
                _MAXLINE = http.client._MAXLINE  # type: ignore[attr-defined]
                connect = b"CONNECT %s:%d HTTP/1.0\r\n" % (  # type: ignore[str-format]
                    self._wrap_ipv6(self._tunnel_host.encode("ascii")),  # type: ignore[union-attr]
                    self._tunnel_port,
                )
                headers = [connect]
                for header, value in self._tunnel_headers.items():  # type: ignore[attr-defined]
                    header_bytes = header.encode("latin-1")
                    value_bytes = value.encode("latin-1")
                    if not self._is_legal_header_name(header_bytes):
                        raise ValueError(f"Invalid header name {header_bytes!r}")
                    if self._is_illegal_header_value(value_bytes):
                        raise ValueError(f"Invalid header value {value_bytes!r}")
                    headers.append(b"%s: %s\r\n" % (header_bytes, value_bytes))
                headers.append(b"\r\n")
                # Making a single send() call instead of one per line encourages
                # the host OS to use a more optimal packet size instead of
                # potentially emitting a series of small packets.
                self.send(b"".join(headers))
                del headers

                response = self.response_class(self.sock, method=self._method)  # type: ignore[attr-defined]
                try:
                    version, code, message = response._read_status()  # type: ignore[attr-defined]

                    if code != http.HTTPStatus.OK:
                        self.close()
                        raise OSError(
                            f"Tunnel connection failed: {code} {message.strip()}"
                        )
                    while True:
                        line = response.fp.readline(_MAXLINE + 1)
                        if len(line) > _MAXLINE:
                            raise http.client.LineTooLong("header line")
                        if not line:
                            # for sites which EOF without sending a trailer
                            break
                        if line in (b"\r\n", b"\n", b""):
                            break

                        if self.debuglevel > 0:
                            print("header:", line.decode())
                finally:
                    response.close()

        elif (3, 12) <= sys.version_info < (3, 12, 14):
            # `_tunnel` copied from 3.12.13 backporting
            # https://github.com/python/cpython/commit/23aef575c7629abcd4aaf028ebd226fb41a4b3c8
            # plus a fix from https://github.com/python/cpython/commit/c00c386faa579ad71196d33408644478488e43ec
            def _tunnel(self) -> None:  # noqa: F811
                if self._contains_disallowed_url_pchar_re.search(self._tunnel_host):  # type: ignore[arg-type]
                    raise ValueError(
                        "Tunnel host can't contain control characters %r"
                        % (self._tunnel_host,)
                    )
                connect = b"CONNECT %s:%d HTTP/1.1\r\n" % (  # type: ignore[str-format]
                    self._wrap_ipv6(self._tunnel_host.encode("idna")),  # type: ignore[union-attr]
                    self._tunnel_port,
                )
                headers = [connect]
                for header, value in self._tunnel_headers.items():  # type: ignore[attr-defined]
                    header_bytes = header.encode("latin-1")
                    value_bytes = value.encode("latin-1")
                    if not self._is_legal_header_name(header_bytes):
                        raise ValueError(f"Invalid header name {header_bytes!r}")
                    if self._is_illegal_header_value(value_bytes):
                        raise ValueError(f"Invalid header value {value_bytes!r}")
                    headers.append(b"%s: %s\r\n" % (header_bytes, value_bytes))
                headers.append(b"\r\n")
                # Making a single send() call instead of one per line encourages
                # the host OS to use a more optimal packet size instead of
                # potentially emitting a series of small packets.
                self.send(b"".join(headers))
                del headers

                response = self.response_class(self.sock, method=self._method)  # type: ignore[attr-defined]
                try:
                    version, code, message = response._read_status()  # type: ignore[attr-defined]

                    self._raw_proxy_headers = http.client._read_headers(response.fp)  # type: ignore[attr-defined]

                    if self.debuglevel > 0:
                        for header in self._raw_proxy_headers:
                            print("header:", header.decode())

                    if code != http.HTTPStatus.OK:
                        self.close()
                        raise OSError(
                            f"Tunnel connection failed: {code} {message.strip()}"
                        )

                finally:
                    response.close()

    def connect(self) -> None:
        self.sock = self._new_conn()
        if self._tunnel_host:
            # If we're tunneling it means we're connected to our proxy.
            self._has_connected_to_proxy = True

            # TODO: Fix tunnel so it doesn't depend on self.sock state.
            self._tunnel()

        # If there's a proxy to be connected to we are fully connected.
        # This is set twice (once above and here) due to forwarding proxies
        # not using tunnelling.
        self._has_connected_to_proxy = bool(self.proxy)

        if self._has_connected_to_proxy:
            self.proxy_is_verified = False

    @property
    def is_closed(self) -> bool:
        return self.sock is None

    @property
    def is_connected(self) -> bool:
        if self.sock is None:
            return False
        return not wait_for_read(self.sock, timeout=0.0)

    @property
    def has_connected_to_proxy(self) -> bool:
        return self._has_connected_to_proxy

    @property
    def proxy_is_forwarding(self) -> bool:
        """
        Return True if a forwarding proxy is configured, else return False
        """
        return bool(self.proxy) and self._tunnel_host is None

    @property
    def proxy_is_tunneling(self) -> bool:
        """
        Return True if a tunneling proxy is configured, else return False
        """
        return self._tunnel_host is not None

    def close(self) -> None:
        try:
            super().close()
        finally:
            # Reset all stateful properties so connection
            # can be re-used without leaking prior configs.
            self.sock = None
            self.is_verified = False
            self.proxy_is_verified = None
            self._has_connected_to_proxy = False
            self._response_options = None
            self._tunnel_host = None
            self._tunnel_port = None
            self._tunnel_scheme = None

    def putrequest(
        self,
        method: str,
        url: str,
        skip_host: bool = False,
        skip_accept_encoding: bool = False,
    ) -> None:
        """"""
        # Empty docstring because the indentation of CPython's implementation
        # is broken but we don't want this method in our documentation.
        match = _CONTAINS_CONTROL_CHAR_RE.search(method)
        if match:
            raise ValueError(
                f"Method cannot contain non-token characters {method!r} (found at least {match.group()!r})"
            )

        return super().putrequest(
            method, url, skip_host=skip_host, skip_accept_encoding=skip_accept_encoding
        )

    def putheader(self, header: str, *values: str) -> None:  # type: ignore[override]
        """"""
        if not any(isinstance(v, str) and v == SKIP_HEADER for v in values):
            super().putheader(header, *values)
        elif to_str(header.lower()) not in SKIPPABLE_HEADERS:
            skippable_headers = "', '".join(
                [str.title(header) for header in sorted(SKIPPABLE_HEADERS)]
            )
            raise ValueError(
                f"urllib3.util.SKIP_HEADER only supports '{skippable_headers}'"
            )

    # `request` method's signature intentionally violates LSP.
    # urllib3's API is different from `http.client.HTTPConnection` and the subclassing is only incidental.
    def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        *,
        chunked: bool = False,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
    ) -> None:
        # Update the inner socket's timeout value to send the request.
        # This only triggers if the connection is re-used.
        if self.sock is not None:
            self.sock.settimeout(self.timeout)

        # Store these values to be fed into the HTTPResponse
        # object later. TODO: Remove this in favor of a real
        # HTTP lifecycle mechanism.

        # We have to store these before we call .request()
        # because sometimes we can still salvage a response
        # off the wire even if we aren't able to completely
        # send the request body.
        self._response_options = _ResponseOptions(
            request_method=method,
            request_url=url,
            preload_content=preload_content,
            decode_content=decode_content,
            enforce_content_length=enforce_content_length,
        )

        if headers is None:
            headers = {}
        header_keys = frozenset(to_str(k.lower()) for k in headers)
        skip_accept_encoding = "accept-encoding" in header_keys
        skip_host = "host" in header_keys
        self.putrequest(
            method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host
        )

        # Transform the body into an iterable of sendall()-able chunks
        # and detect if an explicit Content-Length is doable.
        chunks_and_cl = body_to_chunks(body, method=method, blocksize=self.blocksize)
        chunks = chunks_and_cl.chunks
        content_length = chunks_and_cl.content_length

        # When chunked is explicit set to 'True' we respect that.
        if chunked:
            if "transfer-encoding" not in header_keys:
                self.putheader("Transfer-Encoding", "chunked")
        else:
            # Detect whether a framing mechanism is already in use. If so
            # we respect that value, otherwise we pick chunked vs content-length
            # depending on the type of 'body'.
            if "content-length" in header_keys:
                chunked = False
            elif "transfer-encoding" in header_keys:
                chunked = True

            # Otherwise we go off the recommendation of 'body_to_chunks()'.
            else:
                chunked = False
                if content_length is None:
                    if chunks is not None:
                        chunked = True
                        self.putheader("Transfer-Encoding", "chunked")
                else:
                    self.putheader("Content-Length", str(content_length))

        # Now that framing headers are out of the way we send all the other headers.
        if "user-agent" not in header_keys:
            self.putheader("User-Agent", _get_default_user_agent())
        for header, value in headers.items():
            self.putheader(header, value)
        self.endheaders()

        # If we're given a body we start sending that in chunks.
        if chunks is not None:
            for chunk in chunks:
                # Sending empty chunks isn't allowed for TE: chunked
                # as it indicates the end of the body.
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if chunked:
                    self.send(b"%x\r\n%b\r\n" % (len(chunk), chunk))
                else:
                    self.send(chunk)

        # Regardless of whether we have a body or not, if we're in
        # chunked mode we want to send an explicit empty chunk.
        if chunked:
            self.send(b"0\r\n\r\n")

    def request_chunked(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
    ) -> None:
        """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
        warnings.warn(
            "HTTPConnection.request_chunked() is deprecated and will be removed "
            "in urllib3 v3.0. Instead use HTTPConnection.request(..., chunked=True).",
            category=FutureWarning,
            stacklevel=2,
        )
        self.request(method, url, body=body, headers=headers, chunked=True)

    def getresponse(  # type: ignore[override]
        self,
    ) -> HTTPResponse:
        """
        Get the response from the server.

        If the HTTPConnection is in the correct state, returns an instance of HTTPResponse or of whatever object is returned by the response_class variable.

        If a request has not been sent or if a previous response has not be handled, ResponseNotReady is raised. If the HTTP response indicates that the connection should be closed, then it will be closed before the response is returned. When the connection is closed, the underlying socket is closed.
        """
        # Raise the same error as http.client.HTTPConnection
        if self._response_options is None:
            raise ResponseNotReady()

        # Reset this attribute for being used again.
        resp_options = self._response_options
        self._response_options = None

        # Since the connection's timeout value may have been updated
        # we need to set the timeout on the socket.
        self.sock.settimeout(self.timeout)

        # This is needed here to avoid circular import errors
        from .response import HTTPResponse

        # Save a reference to the shutdown function before ownership is passed
        # to httplib_response
        # TODO should we implement it everywhere?
        _shutdown = getattr(self.sock, "shutdown", None)

        # Get the response from http.client.HTTPConnection
        httplib_response = super().getresponse()

        try:
            assert_header_parsing(httplib_response.msg)
        except (HeaderParsingError, TypeError) as hpe:
            log.warning(
                "Failed to parse headers (url=%s): %s",
                _url_from_connection(self, resp_options.request_url),
                hpe,
                exc_info=True,
            )

        header_items = _normalize_header_values(httplib_response.msg)
        headers = HTTPHeaderDict(header_items)

        response = HTTPResponse(
            body=httplib_response,
            headers=headers,
            status=httplib_response.status,
            version=httplib_response.version,
            version_string=getattr(self, "_http_vsn_str", "HTTP/?"),
            reason=httplib_response.reason,
            preload_content=resp_options.preload_content,
            decode_content=resp_options.decode_content,
            original_response=httplib_response,
            enforce_content_length=resp_options.enforce_content_length,
            request_method=resp_options.request_method,
            request_url=resp_options.request_url,
            sock_shutdown=_shutdown,
        )
        return response


class HTTPSConnection(HTTPConnection):
    """
    Many of the parameters to this constructor are passed to the underlying SSL
    socket by means of :py:func:`urllib3.util.ssl_wrap_socket`.
    """

    default_port = port_by_scheme["https"]  # type: ignore[misc]

    cert_reqs: int | str | None = None
    ca_certs: str | None = None
    ca_cert_dir: str | None = None
    ca_cert_data: None | str | bytes = None
    ssl_version: int | str | None = None
    ssl_minimum_version: int | None = None
    ssl_maximum_version: int | None = None
    assert_fingerprint: str | None = None
    _connect_callback: typing.Callable[..., None] | None = None

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: None | (
            connection._TYPE_SOCKET_OPTIONS
        ) = HTTPConnection.default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
        cert_reqs: int | str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        server_hostname: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        ca_certs: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
        ssl_minimum_version: int | None = None,
        ssl_maximum_version: int | None = None,
        ssl_version: int | str | None = None,  # Deprecated
        cert_file: str | None = None,
        key_file: str | None = None,
        key_password: str | None = None,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            socket_options=socket_options,
            proxy=proxy,
            proxy_config=proxy_config,
        )

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ssl_version = ssl_version
        self.ssl_minimum_version = ssl_minimum_version
        self.ssl_maximum_version = ssl_maximum_version
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

        # cert_reqs depends on ssl_context so calculate last.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)
        self.cert_reqs = cert_reqs
        self._connect_callback = None

    def set_cert(
        self,
        key_file: str | None = None,
        cert_file: str | None = None,
        cert_reqs: int | str | None = None,
        key_password: str | None = None,
        ca_certs: str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
    ) -> None:
        """
        This method should only be called once, before the connection is used.
        """
        warnings.warn(
            "HTTPSConnection.set_cert() is deprecated and will be removed "
            "in urllib3 v3.0. Instead provide the parameters to the "
            "HTTPSConnection constructor.",
            category=FutureWarning,
            stacklevel=2,
        )

        # If cert_reqs is not provided we'll assume CERT_REQUIRED unless we also
        # have an SSLContext object in which case we'll use its verify_mode.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

    def connect(self) -> None:
        # Today we don't need to be doing this step before the /actual/ socket
        # connection, however in the future we'll need to decide whether to
        # create a new socket or re-use an existing "shared" socket as a part
        # of the HTTP/2 handshake dance.
        if self._tunnel_host is not None and self._tunnel_port is not None:
            probe_http2_host = self._tunnel_host
            probe_http2_port = self._tunnel_port
        else:
            probe_http2_host = self.host
            probe_http2_port = self.port

        # Check if the target origin supports HTTP/2.
        # If the value comes back as 'None' it means that the current thread
        # is probing for HTTP/2 support. Otherwise, we're waiting for another
        # probe to complete, or we get a value right away.
        target_supports_http2: bool | None
        if "h2" in ssl_.ALPN_PROTOCOLS:
            target_supports_http2 = http2_probe.acquire_and_get(
                host=probe_http2_host, port=probe_http2_port
            )
        else:
            # If HTTP/2 isn't going to be offered it doesn't matter if
            # the target supports HTTP/2. Don't want to make a probe.
            target_supports_http2 = False

        if self._connect_callback is not None:
            self._connect_callback(
                "before connect",
                thread_id=threading.get_ident(),
                target_supports_http2=target_supports_http2,
            )

        try:
            sock: socket.socket | ssl.SSLSocket
            self.sock = sock = self._new_conn()
            server_hostname: str = self.host
            tls_in_tls = False

            # Do we need to establish a tunnel?
            if self.proxy_is_tunneling:
                # We're tunneling to an HTTPS origin so need to do TLS-in-TLS.
                if self._tunnel_scheme == "https":
                    # _connect_tls_proxy will verify and assign proxy_is_verified
                    self.sock = sock = self._connect_tls_proxy(self.host, sock)
                    tls_in_tls = True
                elif self._tunnel_scheme == "http":
                    self.proxy_is_verified = False

                # If we're tunneling it means we're connected to our proxy.
                self._has_connected_to_proxy = True

                self._tunnel()
                # Override the host with the one we're requesting data from.
                server_hostname = typing.cast(str, self._tunnel_host)

            if self.server_hostname is not None:
                server_hostname = self.server_hostname

            is_time_off = datetime.date.today() < RECENT_DATE
            if is_time_off:
                warnings.warn(
                    (
                        f"System time is way off (before {RECENT_DATE}). This will probably "
                        "lead to SSL verification errors"
                    ),
                    SystemTimeWarning,
                )

            # Remove trailing '.' from fqdn hostnames to allow certificate validation
            server_hostname_rm_dot = server_hostname.rstrip(".")

            # Forwarding proxies should use proxy SSL context for
            # wrapping since that's the connection being established,
            # whereas tunneling proxies should use the connection's SSL
            # context.
            # However, for backwards compatibility reasons, if the proxy
            # is forwarding but no proxy SSL context is provided, we
            # fall back to using the connection's SSL context until
            # urllib3 v3.0. Appropriate warning is emitted in
            # ``ProxyManager.__init__``.
            wrapped_socket: ssl.SSLSocket | SSLTransport
            if self.proxy_is_forwarding and self.proxy_config is not None:
                wrapped_socket = self._connect_tls_proxy(self.host, sock)
                is_verified = self.proxy_is_verified is True
            else:
                wrapped_socket, is_verified = _ssl_wrap_socket_and_match_hostname(
                    sock=sock,
                    cert_reqs=self.cert_reqs,
                    ssl_version=self.ssl_version,
                    ssl_minimum_version=self.ssl_minimum_version,
                    ssl_maximum_version=self.ssl_maximum_version,
                    ca_certs=self.ca_certs,
                    ca_cert_dir=self.ca_cert_dir,
                    ca_cert_data=self.ca_cert_data,
                    cert_file=self.cert_file,
                    key_file=self.key_file,
                    key_password=self.key_password,
                    server_hostname=server_hostname_rm_dot,
                    ssl_context=self.ssl_context,
                    tls_in_tls=tls_in_tls,
                    assert_hostname=self.assert_hostname,
                    assert_fingerprint=self.assert_fingerprint,
                )
            self.sock = wrapped_socket

        # If an error occurs during connection/handshake we may need to release
        # our lock so another connection can probe the origin.
        except BaseException:
            if self._connect_callback is not None:
                self._connect_callback(
                    "after connect failure",
                    thread_id=threading.get_ident(),
                    target_supports_http2=target_supports_http2,
                )

            if target_supports_http2 is None:
                http2_probe.set_and_release(
                    host=probe_http2_host, port=probe_http2_port, supports_http2=None
                )
            raise

        # If this connection doesn't know if the origin supports HTTP/2
        # we report back to the HTTP/2 probe our result.
        if target_supports_http2 is None:
            supports_http2 = wrapped_socket.selected_alpn_protocol() == "h2"
            http2_probe.set_and_release(
                host=probe_http2_host,
                port=probe_http2_port,
                supports_http2=supports_http2,
            )

        # Forwarding proxies can never have a verified target since
        # the proxy is the one doing the verification. Should instead
        # use a CONNECT tunnel in order to verify the target.
        # See: https://github.com/urllib3/urllib3/issues/3267.
        if self.proxy_is_forwarding:
            self.is_verified = False
        else:
            self.is_verified = is_verified

        # If there's a proxy to be connected to we are fully connected.
        # This is set twice (once above and here) due to forwarding proxies
        # not using tunnelling.
        self._has_connected_to_proxy = bool(self.proxy)

        # Set `self.proxy_is_verified` unless it's already set while
        # establishing a tunnel.
        if self._has_connected_to_proxy and self.proxy_is_verified is None:
            self.proxy_is_verified = is_verified

    def _connect_tls_proxy(self, hostname: str, sock: socket.socket) -> ssl.SSLSocket:
        """
        Establish a TLS connection to the proxy using proxy-specific policy.
        """
        proxy_config = typing.cast(ProxyConfig, self.proxy_config)
        proxy_ssl_context = proxy_config.ssl_context

        ssl_context: ssl.SSLContext | None
        cert_reqs: int | str | None
        if proxy_ssl_context is not None:
            # Prefer the proxy's cert policy for the proxy connection
            ssl_context = proxy_ssl_context
            cert_reqs = proxy_ssl_context.verify_mode
            ca_certs = None
            ca_cert_dir = None
            ca_cert_data = None
            ssl_version = None
            ssl_minimum_version = None
            ssl_maximum_version = None
        else:
            # Otherwise we inherit the pool's cert policies
            ssl_context = self.ssl_context if self.proxy_is_forwarding else None
            cert_reqs = self.cert_reqs
            ca_certs = self.ca_certs
            ca_cert_dir = self.ca_cert_dir
            ca_cert_data = self.ca_cert_data
            ssl_version = self.ssl_version
            ssl_minimum_version = self.ssl_minimum_version
            ssl_maximum_version = self.ssl_maximum_version

        sock_and_verified = _ssl_wrap_socket_and_match_hostname(
            sock,
            cert_reqs=cert_reqs,
            ssl_version=ssl_version,
            ssl_minimum_version=ssl_minimum_version,
            ssl_maximum_version=ssl_maximum_version,
            ca_certs=ca_certs,
            ca_cert_dir=ca_cert_dir,
            ca_cert_data=ca_cert_data,
            server_hostname=hostname,
            ssl_context=ssl_context,
            assert_hostname=proxy_config.assert_hostname,
            assert_fingerprint=proxy_config.assert_fingerprint,
            # Features that aren't implemented for proxies yet:
            cert_file=None,
            key_file=None,
            key_password=None,
            tls_in_tls=False,
        )
        self.proxy_is_verified = sock_and_verified.is_verified
        return sock_and_verified.socket  # type: ignore[return-value]


class _WrappedAndVerifiedSocket(typing.NamedTuple):
    """
    Wrapped socket and whether the connection is
    verified after the TLS handshake
    """

    socket: ssl.SSLSocket | SSLTransport
    is_verified: bool


def _ssl_wrap_socket_and_match_hostname(
    sock: socket.socket,
    *,
    cert_reqs: None | str | int,
    ssl_version: None | str | int,
    ssl_minimum_version: int | None,
    ssl_maximum_version: int | None,
    cert_file: str | None,
    key_file: str | None,
    key_password: str | None,
    ca_certs: str | None,
    ca_cert_dir: str | None,
    ca_cert_data: None | str | bytes,
    assert_hostname: None | str | typing.Literal[False],
    assert_fingerprint: str | None,
    server_hostname: str | None,
    ssl_context: ssl.SSLContext | None,
    tls_in_tls: bool = False,
) -> _WrappedAndVerifiedSocket:
    """Logic for constructing an SSLContext from all TLS parameters, passing
    that down into ssl_wrap_socket, and then doing certificate verification
    either via hostname or fingerprint. This function exists to guarantee
    that both proxies and targets have the same behavior when connecting via TLS.
    """
    default_ssl_context = False
    if ssl_context is None:
        default_ssl_context = True
        context = create_urllib3_context(
            ssl_version=resolve_ssl_version(ssl_version),
            ssl_minimum_version=ssl_minimum_version,
            ssl_maximum_version=ssl_maximum_version,
            cert_reqs=resolve_cert_reqs(cert_reqs),
        )
    else:
        context = ssl_context

    context.verify_mode = resolve_cert_reqs(cert_reqs)

    # In some cases, we want to verify hostnames ourselves
    if (
        # `ssl` can't verify fingerprints or alternate hostnames
        assert_fingerprint
        or assert_hostname
        # assert_hostname can be set to False to disable hostname checking
        or assert_hostname is False
        # We still support OpenSSL 1.0.2, which prevents us from verifying
        # hostnames easily: https://github.com/pyca/pyopenssl/pull/933
        or ssl_.IS_PYOPENSSL
        or not ssl_.HAS_NEVER_CHECK_COMMON_NAME
    ):
        context.check_hostname = False

    # Try to load OS default certs if none are given. We need to do the hasattr() check
    # for custom pyOpenSSL SSLContext objects because they don't support
    # load_default_certs().
    if (
        not ca_certs
        and not ca_cert_dir
        and not ca_cert_data
        and default_ssl_context
        and hasattr(context, "load_default_certs")
    ):
        context.load_default_certs()

    # Ensure that IPv6 addresses are in the proper format and don't have a
    # scope ID. Python's SSL module fails to recognize scoped IPv6 addresses
    # and interprets them as DNS hostnames.
    if server_hostname is not None:
        normalized = server_hostname.strip("[]")
        if "%" in normalized:
            normalized = normalized[: normalized.rfind("%")]
        if is_ipaddress(normalized):
            server_hostname = normalized

    ssl_sock = ssl_wrap_socket(
        sock=sock,
        keyfile=key_file,
        certfile=cert_file,
        key_password=key_password,
        ca_certs=ca_certs,
        ca_cert_dir=ca_cert_dir,
        ca_cert_data=ca_cert_data,
        server_hostname=server_hostname,
        ssl_context=context,
        tls_in_tls=tls_in_tls,
    )

    try:
        if assert_fingerprint:
            _assert_fingerprint(
                ssl_sock.getpeercert(binary_form=True), assert_fingerprint
            )
        elif (
            context.verify_mode != ssl.CERT_NONE
            and not context.check_hostname
            and assert_hostname is not False
        ):
            cert: _TYPE_PEER_CERT_RET_DICT = ssl_sock.getpeercert()  # type: ignore[assignment]

            # Need to signal to our match_hostname whether to use 'commonName' or not.
            # If we're using our own constructed SSLContext we explicitly set 'False'
            # because PyPy hard-codes 'True' from SSLContext.hostname_checks_common_name.
            if default_ssl_context:
                hostname_checks_common_name = False
            else:
                hostname_checks_common_name = (
                    getattr(context, "hostname_checks_common_name", False) or False
                )

            _match_hostname(
                cert,
                assert_hostname or server_hostname,  # type: ignore[arg-type]
                hostname_checks_common_name,
            )

        return _WrappedAndVerifiedSocket(
            socket=ssl_sock,
            is_verified=context.verify_mode == ssl.CERT_REQUIRED
            or bool(assert_fingerprint),
        )
    except BaseException:
        ssl_sock.close()
        raise


def _match_hostname(
    cert: _TYPE_PEER_CERT_RET_DICT | None,
    asserted_hostname: str,
    hostname_checks_common_name: bool = False,
) -> None:
    # Our upstream implementation of ssl.match_hostname()
    # only applies this normalization to IP addresses so it doesn't
    # match DNS SANs so we do the same thing!
    stripped_hostname = asserted_hostname.strip("[]")
    if is_ipaddress(stripped_hostname):
        asserted_hostname = stripped_hostname

    try:
        match_hostname(cert, asserted_hostname, hostname_checks_common_name)
    except CertificateError as e:
        log.warning(
            "Certificate did not match expected hostname: %s. Certificate: %s",
            asserted_hostname,
            cert,
        )
        # Add cert to exception and reraise so client code can inspect
        # the cert when catching the exception, if they want to
        e._peer_cert = cert  # type: ignore[attr-defined]
        raise


def _wrap_proxy_error(err: Exception, proxy_scheme: str | None) -> ProxyError:
    # Look for the phrase 'wrong version number', if found
    # then we should warn the user that we're very sure that
    # this proxy is HTTP-only and they have a configuration issue.
    error_normalized = " ".join(re.split("[^a-z]", str(err).lower()))
    is_likely_http_proxy = (
        "wrong version number" in error_normalized
        or "unknown protocol" in error_normalized
        or "record layer failure" in error_normalized
    )
    http_proxy_warning = (
        ". Your proxy appears to only use HTTP and not HTTPS, "
        "try changing your proxy URL to be HTTP. See: "
        "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
        "#https-proxy-error-http-proxy"
    )
    new_err = ProxyError(
        f"Unable to connect to proxy"
        f"{http_proxy_warning if is_likely_http_proxy and proxy_scheme == 'https' else ''}",
        err,
    )
    new_err.__cause__ = err
    return new_err


def _get_default_user_agent() -> str:
    return f"python-urllib3/{__version__}"


class DummyConnection:
    """Used to detect a failed ConnectionCls import."""


if not ssl:
    HTTPSConnection = DummyConnection  # type: ignore[misc, assignment] # noqa: F811


VerifiedHTTPSConnection = HTTPSConnection


def _url_from_connection(
    conn: HTTPConnection | HTTPSConnection, path: str | None = None
) -> str:
    """Returns the URL from a given connection. This is mainly used for testing and logging."""

    scheme = "https" if isinstance(conn, HTTPSConnection) else "http"

    return Url(scheme=scheme, host=conn.host, port=conn.port, path=path).url
<|endoftext|>from __future__ import annotations

import typing

from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from .util.url import Url

_TYPE_BODY = typing.Union[
    bytes, typing.IO[typing.Any], typing.Iterable[bytes | str], str
]


class ProxyConfig(typing.NamedTuple):
    ssl_context: ssl.SSLContext | None
    use_forwarding_for_https: bool
    assert_hostname: None | str | typing.Literal[False]
    assert_fingerprint: str | None


class _ResponseOptions(typing.NamedTuple):
    # TODO: Remove this in favor of a better
    # HTTP request/response lifecycle tracking.
    request_method: str
    request_url: str
    preload_content: bool
    decode_content: bool
    enforce_content_length: bool


if typing.TYPE_CHECKING:
    import ssl
    from typing import Protocol

    from .response import BaseHTTPResponse

    class BaseHTTPConnection(Protocol):
        default_port: typing.ClassVar[int]
        default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]

        host: str
        port: int
        timeout: None | (
            float
        )  # Instance doesn't store _DEFAULT_TIMEOUT, must be resolved.
        blocksize: int
        source_address: tuple[str, int] | None
        socket_options: _TYPE_SOCKET_OPTIONS | None

        proxy: Url | None
        proxy_config: ProxyConfig | None

        is_verified: bool
        proxy_is_verified: bool | None

        def __init__(
            self,
            host: str,
            port: int | None = None,
            *,
            timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
            source_address: tuple[str, int] | None = None,
            blocksize: int = 8192,
            socket_options: _TYPE_SOCKET_OPTIONS | None = ...,
            proxy: Url | None = None,
            proxy_config: ProxyConfig | None = None,
        ) -> None: ...

        def set_tunnel(
            self,
            host: str,
            port: int | None = None,
            headers: typing.Mapping[str, str] | None = None,
            scheme: str = "http",
        ) -> None: ...

        def connect(self) -> None: ...

        def request(
            self,
            method: str,
            url: str,
            body: _TYPE_BODY | None = None,
            headers: typing.Mapping[str, str] | None = None,
            # We know *at least* botocore is depending on the order of the
            # first 3 parameters so to be safe we only mark the later ones
            # as keyword-only to ensure we have space to extend.
            *,
            chunked: bool = False,
            preload_content: bool = True,
            decode_content: bool = True,
            enforce_content_length: bool = True,
        ) -> None: ...

        def getresponse(self) -> BaseHTTPResponse: ...

        def close(self) -> None: ...

        @property
        def is_closed(self) -> bool:
            """Whether the connection either is brand new or has been previously closed.
            If this property is True then both ``is_connected`` and ``has_connected_to_proxy``
            properties must be False.
            """

        @property
        def is_connected(self) -> bool:
            """Whether the connection is actively connected to any origin (proxy or target)"""

        @property
        def has_connected_to_proxy(self) -> bool:
            """Whether the connection has successfully connected to its proxy.
            This returns False if no proxy is in use. Used to determine whether
            errors are coming from the proxy layer or from tunnelling to the target origin.
            """

    class BaseHTTPSConnection(BaseHTTPConnection, Protocol):
        default_port: typing.ClassVar[int]
        default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]

        # Certificate verification methods
        cert_reqs: int | str | None
        assert_hostname: None | str | typing.Literal[False]
        assert_fingerprint: str | None
        ssl_context: ssl.SSLContext | None

        # Trusted CAs
        ca_certs: str | None
        ca_cert_dir: str | None
        ca_cert_data: None | str | bytes

        # TLS version
        ssl_minimum_version: int | None
        ssl_maximum_version: int | None
        ssl_version: int | str | None  # Deprecated

        # Client certificates
        cert_file: str | None
        key_file: str | None
        key_password: str | None

        def __init__(
            self,
            host: str,
            port: int | None = None,
            *,
            timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
            source_address: tuple[str, int] | None = None,
            blocksize: int = 16384,
            socket_options: _TYPE_SOCKET_OPTIONS | None = ...,
            proxy: Url | None = None,
            proxy_config: ProxyConfig | None = None,
            cert_reqs: int | str | None = None,
            assert_hostname: None | str | typing.Literal[False] = None,
            assert_fingerprint: str | None = None,
            server_hostname: str | None = None,
            ssl_context: ssl.SSLContext | None = None,
            ca_certs: str | None = None,
            ca_cert_dir: str | None = None,
            ca_cert_data: None | str | bytes = None,
            ssl_minimum_version: int | None = None,
            ssl_maximum_version: int | None = None,
            ssl_version: int | str | None = None,  # Deprecated
            cert_file: str | None = None,
            key_file: str | None = None,
            key_password: str | None = None,
        ) -> None: ...
<|endoftext|>from __future__ import annotations

import socket
import typing
import warnings
from email.errors import MessageDefect
from http.client import IncompleteRead as httplib_IncompleteRead

if typing.TYPE_CHECKING:
    from .connection import HTTPConnection
    from .connectionpool import ConnectionPool
    from .response import HTTPResponse
    from .util.retry import Retry

# Base Exceptions


class HTTPError(Exception):
    """Base exception used by this module."""


class HTTPWarning(Warning):
    """Base warning used by this module."""


_TYPE_REDUCE_RESULT = tuple[typing.Callable[..., object], tuple[object, ...]]


class PoolError(HTTPError):
    """Base exception for errors caused within a pool."""

    def __init__(self, pool: ConnectionPool, message: str) -> None:
        self.pool = pool
        self._message = message
        super().__init__(f"{pool}: {message}")

    def __reduce__(self) -> _TYPE_REDUCE_RESULT:
        # For pickling purposes.
        return self.__class__, (None, self._message)


class RequestError(PoolError):
    """Base exception for PoolErrors that have associated URLs."""

    def __init__(self, pool: ConnectionPool, url: str | None, message: str) -> None:
        self.url = url
        super().__init__(pool, message)

    def __reduce__(self) -> _TYPE_REDUCE_RESULT:
        # For pickling purposes.
        return self.__class__, (None, self.url, self._message)


class SSLError(HTTPError):
    """Raised when SSL certificate fails in an HTTPS connection."""


class ProxyError(HTTPError):
    """Raised when the connection to a proxy fails."""

    # The original error is also available as __cause__.
    original_error: Exception

    def __init__(self, message: str, error: Exception) -> None:
        super().__init__(message, error)
        self.original_error = error


class DecodeError(HTTPError):
    """Raised when automatic decoding based on Content-Type fails."""


class ProtocolError(HTTPError):
    """Raised when something unexpected happens mid-request/response."""


#: Renamed to ProtocolError but aliased for backwards compatibility.
ConnectionError = ProtocolError


# Leaf Exceptions


class MaxRetryError(RequestError):
    """Raised when the maximum number of retries is exceeded.

    :param pool: The connection pool
    :type pool: :class:`~urllib3.connectionpool.HTTPConnectionPool`
    :param str url: The requested Url
    :param reason: The underlying error
    :type reason: :class:`Exception`

    """

    def __init__(
        self, pool: ConnectionPool, url: str | None, reason: Exception | None = None
    ) -> None:
        self.reason = reason

        message = f"Max retries exceeded with url: {url} (Caused by {reason!r})"

        super().__init__(pool, url, message)

    def __reduce__(self) -> _TYPE_REDUCE_RESULT:
        # For pickling purposes.
        return self.__class__, (None, self.url, self.reason)


class HostChangedError(RequestError):
    """Raised when an existing pool gets a request for a foreign host."""

    def __init__(
        self, pool: ConnectionPool, url: str, retries: Retry | int = 3
    ) -> None:
        message = f"Tried to open a foreign host with url: {url}"
        super().__init__(pool, url, message)
        self.retries = retries


class TimeoutStateError(HTTPError):
    """Raised when passing an invalid state to a timeout"""


class TimeoutError(HTTPError):
    """Raised when a socket timeout error occurs.

    Catching this error will catch both :exc:`ReadTimeoutErrors
    <ReadTimeoutError>` and :exc:`ConnectTimeoutErrors <ConnectTimeoutError>`.
    """


class ReadTimeoutError(TimeoutError, RequestError):
    """Raised when a socket timeout occurs while receiving data from a server"""


# This timeout error does not have a URL attached and needs to inherit from the
# base HTTPError
class ConnectTimeoutError(TimeoutError):
    """Raised when a socket timeout occurs while connecting to a server"""


class NewConnectionError(ConnectTimeoutError, HTTPError):
    """Raised when we fail to establish a new connection. Usually ECONNREFUSED."""

    def __init__(self, conn: HTTPConnection, message: str) -> None:
        self.conn = conn
        self._message = message
        super().__init__(f"{conn}: {message}")

    def __reduce__(self) -> _TYPE_REDUCE_RESULT:
        # For pickling purposes.
        return self.__class__, (None, self._message)

    @property
    def pool(self) -> HTTPConnection:
        warnings.warn(
            "The 'pool' property is deprecated and will be removed "
            "in urllib3 v3.0. Use 'conn' instead.",
            FutureWarning,
            stacklevel=2,
        )

        return self.conn


class NameResolutionError(NewConnectionError):
    """Raised when host name resolution fails."""

    def __init__(self, host: str, conn: HTTPConnection, reason: socket.gaierror):
        message = f"Failed to resolve '{host}' ({reason})"
        self._host = host
        self._reason = reason
        super().__init__(conn, message)

    def __reduce__(self) -> _TYPE_REDUCE_RESULT:
        # For pickling purposes.
        return self.__class__, (self._host, None, self._reason)


class EmptyPoolError(PoolError):
    """Raised when a pool runs out of connections and no more are allowed."""


class FullPoolError(PoolError):
    """Raised when we try to add a connection to a full pool in blocking mode."""


class ClosedPoolError(PoolError):
    """Raised when a request enters a pool after the pool has been closed."""


class LocationValueError(ValueError, HTTPError):
    """Raised when there is something wrong with a given URL input."""


class LocationParseError(LocationValueError):
    """Raised when get_host or similar fails to parse the URL input."""

    def __init__(self, location: str) -> None:
        message = f"Failed to parse: {location}"
        super().__init__(message)

        self.location = location


class URLSchemeUnknown(LocationValueError):
    """Raised when a URL input has an unsupported scheme."""

    def __init__(self, scheme: str):
        message = f"Not supported URL scheme {scheme}"
        super().__init__(message)

        self.scheme = scheme


class ResponseError(HTTPError):
    """Used as a container for an error reason supplied in a MaxRetryError."""

    GENERIC_ERROR = "too many error responses"
    SPECIFIC_ERROR = "too many {status_code} error responses"


class SecurityWarning(HTTPWarning):
    """Warned when performing security reducing actions"""


class InsecureRequestWarning(SecurityWarning):
    """Warned when making an unverified HTTPS request."""


class NotOpenSSLWarning(SecurityWarning):
    """Warned when using unsupported SSL library"""


class SystemTimeWarning(SecurityWarning):
    """Warned when system time is suspected to be wrong"""


class InsecurePlatformWarning(SecurityWarning):
    """Warned when certain TLS/SSL configuration is not available on a platform."""


class DependencyWarning(HTTPWarning):
    """
    Warned when an attempt is made to import a module with missing optional
    dependencies.
    """


class ResponseNotChunked(ProtocolError, ValueError):
    """Response needs to be chunked in order to read it as chunks."""


class BodyNotHttplibCompatible(HTTPError):
    """
    Body should be :class:`http.client.HTTPResponse` like
    (have an fp attribute which returns raw chunks) for read_chunked().
    """


class IncompleteRead(HTTPError, httplib_IncompleteRead):
    """
    Response length doesn't match expected Content-Length

    Subclass of :class:`http.client.IncompleteRead` to allow int value
    for ``partial`` to avoid creating large objects on streamed reads.
    """

    partial: int  # type: ignore[assignment]
    expected: int

    def __init__(self, partial: int, expected: int) -> None:
        self.partial = partial
        self.expected = expected

    def __repr__(self) -> str:
        return "IncompleteRead(%i bytes read, %i more expected)" % (
            self.partial,
            self.expected,
        )


class InvalidChunkLength(HTTPError, httplib_IncompleteRead):
    """Invalid chunk length in a chunked response."""

    def __init__(self, response: HTTPResponse, length: bytes) -> None:
        self.partial: int = response.tell()  # type: ignore[assignment]
        self.expected: int | None = response.length_remaining
        self.response = response
        self.length = length

    def __repr__(self) -> str:
        return "InvalidChunkLength(got length %r, %i bytes read)" % (
            self.length,
            self.partial,
        )


class InvalidHeader(HTTPError):
    """The header provided was somehow invalid."""


class ProxySchemeUnknown(AssertionError, URLSchemeUnknown):
    """ProxyManager does not support the supplied scheme"""

    # TODO(t-8ch): Stop inheriting from AssertionError in v2.0.

    def __init__(self, scheme: str | None) -> None:
        # 'localhost' is here because our URL parser parses
        # localhost:8080 -> scheme=localhost, remove if we fix this.
        if scheme == "localhost":
            scheme = None
        if scheme is None:
            message = "Proxy URL had no scheme, should start with http:// or https://"
        else:
            message = f"Proxy URL had unsupported scheme {scheme}, should use http:// or https://"
        super().__init__(message)


class ProxySchemeUnsupported(ValueError):
    """Fetching HTTPS resources through HTTPS proxies is unsupported"""


class HeaderParsingError(HTTPError):
    """Raised by assert_header_parsing, but we convert it to a log.warning statement."""

    def __init__(
        self, defects: list[MessageDefect], unparsed_data: bytes | str | None
    ) -> None:
        message = f"{defects or 'Unknown'}, unparsed data: {unparsed_data!r}"
        super().__init__(message)


class UnrewindableBodyError(HTTPError):
    """urllib3 encountered an error when trying to rewind a body"""
<|endoftext|>"""
This module contains provisional support for SOCKS proxies from within
urllib3. This module supports SOCKS4, SOCKS4A (an extension of SOCKS4), and
SOCKS5. To enable its functionality, either install PySocks or install this
module with the ``socks`` extra.

The SOCKS implementation supports the full range of urllib3 features. It also
supports the following SOCKS features:

- SOCKS4A (``proxy_url='socks4a://...``)
- SOCKS4 (``proxy_url='socks4://...``)
- SOCKS5 with remote DNS (``proxy_url='socks5h://...``)
- SOCKS5 with local DNS (``proxy_url='socks5://...``)
- Usernames and passwords for the SOCKS proxy

.. note::
   It is recommended to use ``socks5h://`` or ``socks4a://`` schemes in
   your ``proxy_url`` to ensure that DNS resolution is done from the remote
   server instead of client-side when connecting to a domain name.

SOCKS4 supports IPv4 and domain names with the SOCKS4A extension. SOCKS5
supports IPv4, IPv6, and domain names.

When connecting to a SOCKS4 proxy the ``username`` portion of the ``proxy_url``
will be sent as the ``userid`` section of the SOCKS request:

.. code-block:: python

    proxy_url="socks4a://<userid>@proxy-host"

When connecting to a SOCKS5 proxy the ``username`` and ``password`` portion
of the ``proxy_url`` will be sent as the username/password to authenticate
with the proxy:

.. code-block:: python

    proxy_url="socks5h://<username>:<password>@proxy-host"

"""

from __future__ import annotations

try:
    import socks
except ImportError:
    import warnings

    from ..exceptions import DependencyWarning

    warnings.warn(
        (
            "SOCKS support in urllib3 requires the installation of optional "
            "dependencies: specifically, PySocks.  For more information, see "
            "https://urllib3.readthedocs.io/en/latest/advanced-usage.html#socks-proxies"
        ),
        DependencyWarning,
    )
    raise

import typing
from socket import timeout as SocketTimeout

from ..connection import HTTPConnection, HTTPSConnection
from ..connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import ConnectTimeoutError, NewConnectionError
from ..poolmanager import PoolManager
from ..util.url import parse_url

try:
    import ssl
except ImportError:
    ssl = None  # type: ignore[assignment]


class _TYPE_SOCKS_OPTIONS(typing.TypedDict):
    socks_version: int
    proxy_host: str | None
    proxy_port: str | None
    username: str | None
    password: str | None
    rdns: bool


class SOCKSConnection(HTTPConnection):
    """
    A plain-text HTTP connection that connects via a SOCKS proxy.
    """

    def __init__(
        self,
        _socks_options: _TYPE_SOCKS_OPTIONS,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        self._socks_options = _socks_options
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> socks.socksocket:
        """
        Establish a new connection via the SOCKS proxy.
        """
        extra_kw: dict[str, typing.Any] = {}
        if self.source_address:
            extra_kw["source_address"] = self.source_address

        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options

        try:
            conn = socks.create_connection(
                (self.host, self.port),
                proxy_type=self._socks_options["socks_version"],
                proxy_addr=self._socks_options["proxy_host"],
                proxy_port=self._socks_options["proxy_port"],  # type: ignore[arg-type]
                proxy_username=self._socks_options["username"],
                proxy_password=self._socks_options["password"],
                proxy_rdns=self._socks_options["rdns"],
                timeout=self.timeout,  # type: ignore[arg-type]
                **extra_kw,
            )

        except SocketTimeout as e:
            raise ConnectTimeoutError(
                self,
                f"Connection to {self.host} timed out. (connect timeout={self.timeout})",
            ) from e

        except socks.ProxyError as e:
            # This is fragile as hell, but it seems to be the only way to raise
            # useful errors here.
            if e.socket_err:
                error = e.socket_err
                if isinstance(error, SocketTimeout):
                    raise ConnectTimeoutError(
                        self,
                        f"Connection to {self.host} timed out. (connect timeout={self.timeout})",
                    ) from e
                else:
                    # Adding `from e` messes with coverage somehow, so it's omitted.
                    # See #2386.
                    raise NewConnectionError(
                        self, f"Failed to establish a new connection: {error}"
                    )
            else:  # Defensive: see https://github.com/urllib3/urllib3/pull/3728#pullrequestreview-3816302703
                raise NewConnectionError(
                    self, f"Failed to establish a new connection: {e}"
                ) from e

        except OSError as e:  # Defensive: PySocks should catch all these.
            raise NewConnectionError(
                self, f"Failed to establish a new connection: {e}"
            ) from e

        return conn


# We don't need to duplicate the Verified/Unverified distinction from
# urllib3/connection.py here because the HTTPSConnection will already have been
# correctly set to either the Verified or Unverified form by that module. This
# means the SOCKSHTTPSConnection will automatically be the correct type.
class SOCKSHTTPSConnection(SOCKSConnection, HTTPSConnection):
    pass


class SOCKSHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = SOCKSConnection


class SOCKSHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = SOCKSHTTPSConnection


class SOCKSProxyManager(PoolManager):
    """
    A version of the urllib3 ProxyManager that routes connections via the
    defined SOCKS proxy.
    """

    pool_classes_by_scheme = {
        "http": SOCKSHTTPConnectionPool,
        "https": SOCKSHTTPSConnectionPool,
    }

    def __init__(
        self,
        proxy_url: str,
        username: str | None = None,
        password: str | None = None,
        num_pools: int = 10,
        headers: typing.Mapping[str, str] | None = None,
        **connection_pool_kw: typing.Any,
    ):
        parsed = parse_url(proxy_url)

        if username is None and password is None:
            username, password = parsed.auth_decoded
        if parsed.scheme == "socks5":
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = False
        elif parsed.scheme == "socks5h":
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = True
        elif parsed.scheme == "socks4":
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = False
        elif parsed.scheme == "socks4a":
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = True
        else:
            raise ValueError(f"Unable to determine SOCKS version from {proxy_url}")

        self.proxy_url = proxy_url

        socks_options = {
            "socks_version": socks_version,
            "proxy_host": parsed.host,
            "proxy_port": parsed.port,
            "username": username,
            "password": password,
            "rdns": rdns,
        }
        connection_pool_kw["_socks_options"] = socks_options

        super().__init__(num_pools, headers, **connection_pool_kw)

        self.pool_classes_by_scheme = SOCKSProxyManager.pool_classes_by_scheme
<|endoftext|>"""
Module for using pyOpenSSL as a TLS backend. This module was relevant before
the standard library ``ssl`` module supported SNI, but now that we've dropped
support for Python 2.7 all relevant Python versions support SNI so
**this module is no longer recommended**.

This needs the following packages installed:

* `pyOpenSSL`_ (tested with 19.0.0)
* `cryptography`_ (minimum 2.3, from pyopenssl)
* `idna`_ (minimum 2.1, from cryptography)

However, pyOpenSSL depends on cryptography, so while we use all three directly here we
end up having relatively few packages required.

You can install them with the following command:

.. code-block:: bash

    $ python -m pip install pyopenssl cryptography idna

To activate certificate checking, call
:func:`~urllib3.contrib.pyopenssl.inject_into_urllib3` from your Python code
before you begin making HTTP requests. This can be done in a ``sitecustomize``
module, or at any other time before your application begins using ``urllib3``,
like this:

.. code-block:: python

    try:
        import urllib3.contrib.pyopenssl
        urllib3.contrib.pyopenssl.inject_into_urllib3()
    except ImportError:
        pass

.. _pyopenssl: https://www.pyopenssl.org
.. _cryptography: https://cryptography.io
.. _idna: https://github.com/kjd/idna
"""

from __future__ import annotations

import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509.oid import NameOID

try:
    from cryptography.x509 import UnsupportedExtension  # type: ignore[attr-defined]
except ImportError:
    # UnsupportedExtension is gone in cryptography >= 2.1.0
    class UnsupportedExtension(Exception):  # type: ignore[no-redef]
        pass


import logging
import ssl
import typing
from socket import socket as socket_cls

from .. import util

if typing.TYPE_CHECKING:
    from OpenSSL.crypto import X509


__all__ = ["inject_into_urllib3", "extract_from_urllib3"]

# Map from urllib3 to PyOpenSSL compatible parameter-values.
_openssl_versions: dict[int, int] = {
    util.ssl_.PROTOCOL_TLS: OpenSSL.SSL.SSLv23_METHOD,  # type: ignore[attr-defined]
    util.ssl_.PROTOCOL_TLS_CLIENT: OpenSSL.SSL.SSLv23_METHOD,  # type: ignore[attr-defined]
}

if hasattr(ssl, "PROTOCOL_TLSv1") and hasattr(OpenSSL.SSL, "TLSv1_METHOD"):
    _openssl_versions[ssl.PROTOCOL_TLSv1] = OpenSSL.SSL.TLSv1_METHOD

if hasattr(ssl, "PROTOCOL_TLSv1_1") and hasattr(OpenSSL.SSL, "TLSv1_1_METHOD"):
    _openssl_versions[ssl.PROTOCOL_TLSv1_1] = OpenSSL.SSL.TLSv1_1_METHOD

if hasattr(ssl, "PROTOCOL_TLSv1_2") and hasattr(OpenSSL.SSL, "TLSv1_2_METHOD"):
    _openssl_versions[ssl.PROTOCOL_TLSv1_2] = OpenSSL.SSL.TLSv1_2_METHOD


_stdlib_to_openssl_verify = {
    ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE,
    ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER,
    ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER
    + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT,
}
_openssl_to_stdlib_verify = {v: k for k, v in _stdlib_to_openssl_verify.items()}

# The SSLvX values are the most likely to be missing in the future
# but we check them all just to be sure.
_OP_NO_SSLv2_OR_SSLv3: int = getattr(OpenSSL.SSL, "OP_NO_SSLv2", 0) | getattr(
    OpenSSL.SSL, "OP_NO_SSLv3", 0
)
_OP_NO_TLSv1: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1", 0)
_OP_NO_TLSv1_1: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_1", 0)
_OP_NO_TLSv1_2: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_2", 0)
_OP_NO_TLSv1_3: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_3", 0)

_openssl_to_ssl_minimum_version: dict[int, int] = {
    ssl.TLSVersion.MINIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.TLSv1: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1,
    ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1,
    ssl.TLSVersion.TLSv1_3: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2
    ),
    ssl.TLSVersion.MAXIMUM_SUPPORTED: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2
    ),
}
_openssl_to_ssl_maximum_version: dict[int, int] = {
    ssl.TLSVersion.MINIMUM_SUPPORTED: (
        _OP_NO_SSLv2_OR_SSLv3
        | _OP_NO_TLSv1
        | _OP_NO_TLSv1_1
        | _OP_NO_TLSv1_2
        | _OP_NO_TLSv1_3
    ),
    ssl.TLSVersion.TLSv1: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3
    ),
    ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3,
    ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_3,
    ssl.TLSVersion.TLSv1_3: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.MAXIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3,
}

# OpenSSL will only write 16K at a time
SSL_WRITE_BLOCKSIZE = 16384

orig_util_SSLContext = util.ssl_.SSLContext


log = logging.getLogger(__name__)


def inject_into_urllib3() -> None:
    "Monkey-patch urllib3 with PyOpenSSL-backed SSL-support."

    _validate_dependencies_met()

    util.SSLContext = PyOpenSSLContext  # type: ignore[assignment]
    util.ssl_.SSLContext = PyOpenSSLContext  # type: ignore[assignment]
    util.IS_PYOPENSSL = True
    util.ssl_.IS_PYOPENSSL = True


def extract_from_urllib3() -> None:
    "Undo monkey-patching by :func:`inject_into_urllib3`."

    util.SSLContext = orig_util_SSLContext
    util.ssl_.SSLContext = orig_util_SSLContext
    util.IS_PYOPENSSL = False
    util.ssl_.IS_PYOPENSSL = False


def _validate_dependencies_met() -> None:
    """
    Verifies that PyOpenSSL's package-level dependencies have been met.
    Throws `ImportError` if they are not met.
    """
    # Method added in `cryptography==1.1`; not available in older versions
    from cryptography.x509.extensions import Extensions

    if getattr(Extensions, "get_extension_for_class", None) is None:
        raise ImportError(
            "'cryptography' module missing required functionality.  "
            "Try upgrading to v1.3.4 or newer."
        )

    # pyOpenSSL 0.14 and above use cryptography for OpenSSL bindings. The _x509
    # attribute is only present on those versions.
    from OpenSSL.crypto import X509

    x509 = X509()
    if getattr(x509, "_x509", None) is None:
        raise ImportError(
            "'pyOpenSSL' module missing required functionality. "
            "Try upgrading to v0.14 or newer."
        )


def _dnsname_to_stdlib(name: str) -> str | None:
    """
    Converts a dNSName SubjectAlternativeName field to the form used by the
    standard library on the given Python version.

    Cryptography produces a dNSName as a unicode string that was idna-decoded
    from ASCII bytes. We need to idna-encode that string to get it back, and
    then on Python 3 we also need to convert to unicode via UTF-8 (the stdlib
    uses PyUnicode_FromStringAndSize on it, which decodes via UTF-8).

    If the name cannot be idna-encoded then we return None signalling that
    the name given should be skipped.
    """

    def idna_encode(name: str) -> bytes | None:
        """
        Borrowed wholesale from the Python Cryptography Project. It turns out
        that we can't just safely call `idna.encode`: it can explode for
        wildcard names. This avoids that problem.
        """
        import idna

        try:
            for prefix in ["*.", "."]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    return prefix.encode("ascii") + idna.encode(name)
            return idna.encode(name)
        except idna.core.IDNAError:
            return None

    # Don't send IPv6 addresses through the IDNA encoder.
    if ":" in name:
        return name

    encoded_name = idna_encode(name)
    if encoded_name is None:
        return None
    return encoded_name.decode("utf-8")


def get_subj_alt_name(peer_cert: X509) -> list[tuple[str, str]]:
    """
    Given an PyOpenSSL certificate, provides all the subject alternative names.
    """
    cert = peer_cert.to_cryptography()

    # We want to find the SAN extension. Ask Cryptography to locate it (it's
    # faster than looping in Python)
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        # No such extension, return the empty list.
        return []
    except (
        x509.DuplicateExtension,
        UnsupportedExtension,
        x509.UnsupportedGeneralNameType,
        UnicodeError,
    ) as e:
        # A problem has been found with the quality of the certificate. Assume
        # no SAN field is present.
        log.warning(
            "A problem was encountered with the certificate that prevented "
            "urllib3 from finding the SubjectAlternativeName field. This can "
            "affect certificate validation. The error was %s",
            e,
        )
        return []

    # We want to return dNSName and iPAddress fields. We need to cast the IPs
    # back to strings because the match_hostname function wants them as
    # strings.
    # Sadly the DNS names need to be idna encoded and then, on Python 3, UTF-8
    # decoded. This is pretty frustrating, but that's what the standard library
    # does with certificates, and so we need to attempt to do the same.
    # We also want to skip over names which cannot be idna encoded.
    names = [
        ("DNS", name)
        for name in map(_dnsname_to_stdlib, ext.get_values_for_type(x509.DNSName))
        if name is not None
    ]
    names.extend(
        ("IP Address", str(name)) for name in ext.get_values_for_type(x509.IPAddress)
    )

    return names


def _get_common_name(peer_cert: X509) -> str | None:
    """
    Given a pyOpenSSL certificate, return the subject's common name.
    """
    cert = peer_cert.to_cryptography()
    names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return typing.cast(str, names[0].value) if names else None


class WrappedSocket:
    """API-compatibility wrapper for Python OpenSSL's Connection-class."""

    def __init__(
        self,
        connection: OpenSSL.SSL.Connection,
        socket: socket_cls,
        suppress_ragged_eofs: bool = True,
    ) -> None:
        self.connection = connection
        self.socket = socket
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self._io_refs = 0
        self._closed = False

    def fileno(self) -> int:
        return self.socket.fileno()

    # Copy-pasted from Python 3.5 source code
    def _decref_socketios(self) -> None:
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def recv(self, *args: typing.Any, **kwargs: typing.Any) -> bytes:
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, "Unexpected EOF"):
                return b""
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b""
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise TimeoutError("The read operation timed out") from e
            else:
                return self.recv(*args, **kwargs)

        # TLS 1.3 post-handshake authentication
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"read error: {e!r}") from e
        else:
            return data

    def recv_into(self, *args: typing.Any, **kwargs: typing.Any) -> int:
        try:
            return self.connection.recv_into(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, "Unexpected EOF"):
                return 0
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return 0
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise TimeoutError("The read operation timed out") from e
            else:
                return self.recv_into(*args, **kwargs)

        # TLS 1.3 post-handshake authentication
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"read error: {e!r}") from e

    def settimeout(self, timeout: float) -> None:
        return self.socket.settimeout(timeout)

    def _send_until_done(self, data: bytes) -> int:
        while True:
            try:
                return self.connection.send(data)
            except OpenSSL.SSL.WantWriteError as e:
                if not util.wait_for_write(self.socket, self.socket.gettimeout()):
                    raise TimeoutError() from e
                continue
            except OpenSSL.SSL.SysCallError as e:
                raise OSError(e.args[0], str(e)) from e

    def sendall(self, data: bytes) -> None:
        total_sent = 0
        while total_sent < len(data):
            sent = self._send_until_done(
                data[total_sent : total_sent + SSL_WRITE_BLOCKSIZE]
            )
            total_sent += sent

    def shutdown(self, how: int) -> None:
        try:
            self.connection.shutdown()
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"shutdown error: {e!r}") from e

    def close(self) -> None:
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    def _real_close(self) -> None:
        try:
            return self.connection.close()  # type: ignore[no-any-return]
        except OpenSSL.SSL.Error:
            return

    @typing.overload
    def getpeercert(
        self, binary_form: typing.Literal[False] = False
    ) -> dict[str, list[typing.Any]] | None: ...

    @typing.overload
    def getpeercert(self, binary_form: typing.Literal[True]) -> bytes | None: ...

    def getpeercert(
        self, binary_form: bool = False
    ) -> dict[str, list[typing.Any]] | bytes | None:
        x509 = self.connection.get_peer_certificate()

        if x509 is None:
            return None

        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)

        return {
            "subject": ((("commonName", _get_common_name(x509)),),),  # type: ignore[dict-item]
            "subjectAltName": get_subj_alt_name(x509),
        }

    def version(self) -> str:
        return self.connection.get_protocol_version_name()

    def selected_alpn_protocol(self) -> str | None:
        alpn_proto = self.connection.get_alpn_proto_negotiated()
        return alpn_proto.decode() if alpn_proto else None


WrappedSocket.makefile = socket_cls.makefile  # type: ignore[attr-defined]


class PyOpenSSLContext:
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object. I am responsible
    for translating the interface of the standard library ``SSLContext`` object
    to calls into PyOpenSSL.
    """

    def __init__(self, protocol: int) -> None:
        self.protocol = _openssl_versions[protocol]
        self._ctx = OpenSSL.SSL.Context(self.protocol)
        self._options = 0
        self.check_hostname = False
        self._minimum_version: int = ssl.TLSVersion.MINIMUM_SUPPORTED
        self._maximum_version: int = ssl.TLSVersion.MAXIMUM_SUPPORTED
        self._verify_flags: int = ssl.VERIFY_X509_TRUSTED_FIRST

    @property
    def options(self) -> int:
        return self._options

    @options.setter
    def options(self, value: int) -> None:
        self._options = value
        self._set_ctx_options()

    @property
    def verify_flags(self) -> int:
        return self._verify_flags

    @verify_flags.setter
    def verify_flags(self, value: int) -> None:
        self._verify_flags = value
        self._ctx.get_cert_store().set_flags(self._verify_flags)  # type: ignore[union-attr]

    @property
    def verify_mode(self) -> int:
        return _openssl_to_stdlib_verify[self._ctx.get_verify_mode()]

    @verify_mode.setter
    def verify_mode(self, value: ssl.VerifyMode) -> None:
        self._ctx.set_verify(_stdlib_to_openssl_verify[value], _verify_callback)

    def set_default_verify_paths(self) -> None:
        self._ctx.set_default_verify_paths()

    def set_ciphers(self, ciphers: bytes | str) -> None:
        if isinstance(ciphers, str):
            ciphers = ciphers.encode("utf-8")
        self._ctx.set_cipher_list(ciphers)

    def load_verify_locations(
        self,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: bytes | None = None,
    ) -> None:
        if cafile is not None:
            cafile = cafile.encode("utf-8")  # type: ignore[assignment]
        if capath is not None:
            capath = capath.encode("utf-8")  # type: ignore[assignment]
        try:
            self._ctx.load_verify_locations(cafile, capath)
            if cadata is not None:
                self._ctx.load_verify_locations(cadata)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"unable to load trusted certificates: {e!r}") from e

    def load_cert_chain(
        self,
        certfile: str,
        keyfile: str | None = None,
        password: str | bytes | None = None,
    ) -> None:
        try:
            self._ctx.use_certificate_chain_file(certfile)
            if password is not None:
                if not isinstance(password, bytes):
                    password = password.encode("utf-8")
                # pyOpenSSL added cryptography-key support in 24.3.0.
                # Keep using the older password-callback path until 2026's
                # versions because set_passwd_cb() became deprecated in 26.3.0.
                if int(OpenSSL.__version__.split(".")[0]) >= 26:
                    with open(keyfile or certfile, "rb") as key_file:
                        key_data = key_file.read()
                    try:
                        private_key = load_pem_private_key(key_data, password)
                    except TypeError:
                        # Match SSLContext by ignoring passwords supplied
                        # for unencrypted keys.
                        # Incorrect passwords for encrypted keys raise
                        # ValueError instead.
                        private_key = load_pem_private_key(key_data, None)
                    # cryptography's loader returns a wider private-key union
                    # than pyOpenSSL accepts, so we add `type: ignore` here.
                    self._ctx.use_privatekey(private_key)  # type: ignore[arg-type]
                else:
                    self._ctx.set_passwd_cb(lambda *_: password)
                    self._ctx.use_privatekey_file(keyfile or certfile)
            else:
                self._ctx.use_privatekey_file(keyfile or certfile)
        except (OpenSSL.SSL.Error, TypeError, ValueError) as e:
            raise ssl.SSLError(f"Unable to load certificate chain: {e!r}") from e

    def set_alpn_protocols(self, protocols: list[bytes | str]) -> None:
        protocols = [util.util.to_bytes(p, "ascii") for p in protocols]
        return self._ctx.set_alpn_protos(protocols)  # type: ignore[arg-type]

    def wrap_socket(
        self,
        sock: socket_cls,
        server_side: bool = False,
        do_handshake_on_connect: bool = True,
        suppress_ragged_eofs: bool = True,
        server_hostname: bytes | str | None = None,
    ) -> WrappedSocket:
        cnx = OpenSSL.SSL.Connection(self._ctx, sock)

        # If server_hostname is an IP, don't use it for SNI, per RFC6066 Section 3
        if server_hostname and not util.ssl_.is_ipaddress(server_hostname):
            if isinstance(server_hostname, str):
                server_hostname = server_hostname.encode("utf-8")
            cnx.set_tlsext_host_name(server_hostname)

        cnx.set_connect_state()

        while True:
            try:
                cnx.do_handshake()
            except OpenSSL.SSL.WantReadError as e:
                if not util.wait_for_read(sock, sock.gettimeout()):
                    raise TimeoutError("select timed out") from e
                continue
            except OpenSSL.SSL.Error as e:
                raise ssl.SSLError(f"bad handshake: {e!r}") from e
            break

        return WrappedSocket(cnx, sock)

    def _set_ctx_options(self) -> None:
        self._ctx.set_options(
            self._options
            | _openssl_to_ssl_minimum_version[self._minimum_version]
            | _openssl_to_ssl_maximum_version[self._maximum_version]
        )

    @property
    def minimum_version(self) -> int:
        return self._minimum_version

    @minimum_version.setter
    def minimum_version(self, minimum_version: int) -> None:
        self._minimum_version = minimum_version
        self._set_ctx_options()

    @property
    def maximum_version(self) -> int:
        return self._maximum_version

    @maximum_version.setter
    def maximum_version(self, maximum_version: int) -> None:
        self._maximum_version = maximum_version
        self._set_ctx_options()


def _verify_callback(
    cnx: OpenSSL.SSL.Connection,
    x509: X509,
    err_no: int,
    err_depth: int,
    return_code: int,
) -> bool:
    return err_no == 0
<|endoftext|>from __future__ import annotations

import json as _json
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException as HTTPException
from io import BytesIO, IOBase

from ...exceptions import InvalidHeader, TimeoutError
from ...response import BaseHTTPResponse
from ...util.retry import Retry
from .request import EmscriptenRequest

if typing.TYPE_CHECKING:
    from ..._base_connection import BaseHTTPConnection, BaseHTTPSConnection

log = logging.getLogger(__name__)


@dataclass
class EmscriptenResponse:
    status_code: int
    headers: dict[str, str]
    body: IOBase | bytes
    request: EmscriptenRequest


class EmscriptenHttpResponseWrapper(BaseHTTPResponse):
    def __init__(
        self,
        internal_response: EmscriptenResponse,
        url: str | None = None,
        connection: BaseHTTPConnection | BaseHTTPSConnection | None = None,
    ):
        self._pool = None  # set by pool class
        self._body = None
        self._uncached_read_occurred = False
        self._response = internal_response
        self._url = url
        self._connection = connection
        self._closed = False
        super().__init__(
            headers=internal_response.headers,
            status=internal_response.status_code,
            request_url=url,
            version=0,
            version_string="HTTP/?",
            reason="",
            decode_content=True,
        )
        self.length_remaining = self._init_length(self._response.request.method)
        self.length_is_certain = False

    @property
    def url(self) -> str | None:
        return self._url

    @url.setter
    def url(self, url: str | None) -> None:
        self._url = url

    @property
    def connection(self) -> BaseHTTPConnection | BaseHTTPSConnection | None:
        return self._connection

    @property
    def retries(self) -> Retry | None:
        return self._retries

    @retries.setter
    def retries(self, retries: Retry | None) -> None:
        # Override the request_url if retries has a redirect location.
        self._retries = retries

    def stream(
        self, amt: int | None = 2**16, decode_content: bool | None = None
    ) -> typing.Generator[bytes]:
        """
        A generator wrapper for the read() method. A call will block until
        ``amt`` bytes have been read from the connection or until the
        connection is closed.

        :param amt:
            How much of the content to read. The generator will return up to
            much data per iteration, but may return less. This is particularly
            likely when using compressed data. However, the empty string will
            never be returned.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        while True:
            data = self.read(amt=amt, decode_content=decode_content)

            if data:
                yield data
            else:
                break

    def _init_length(self, request_method: str | None) -> int | None:
        length: int | None
        content_length: str | None = self.headers.get("content-length")

        if content_length is not None:
            try:
                # RFC 7230 section 3.3.2 specifies multiple content lengths can
                # be sent in a single Content-Length header
                # (e.g. Content-Length: 42, 42). This line ensures the values
                # are all valid ints and that as long as the `set` length is 1,
                # all values are the same. Otherwise, the header is invalid.
                lengths = {int(val) for val in content_length.split(",")}
                if len(lengths) > 1:
                    raise InvalidHeader(
                        "Content-Length contained multiple "
                        "unmatching values (%s)" % content_length
                    )
                length = lengths.pop()
            except ValueError:
                length = None
            else:
                if length < 0:
                    length = None

        else:  # if content_length is None
            length = None

        # Check for responses that shouldn't include a body
        if (
            self.status in (204, 304)
            or 100 <= self.status < 200
            or request_method == "HEAD"
        ):
            length = 0

        return length

    def read(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,  # ignored because browser decodes always
        cache_content: bool = False,
    ) -> bytes:
        if (
            self._closed
            or self._response is None
            or (isinstance(self._response.body, IOBase) and self._response.body.closed)
        ):
            return b""

        with self._error_catcher():
            # body has been preloaded as a string by XmlHttpRequest
            if not isinstance(self._response.body, IOBase):
                self.length_remaining = len(self._response.body)
                self.length_is_certain = True
                # wrap body in IOStream
                self._response.body = BytesIO(self._response.body)
            if amt is not None and amt >= 0:
                # don't cache partial content
                cache_content = False
                data = self._response.body.read(amt)
                self._uncached_read_occurred = True
            else:  # read all we can (and cache it)
                data = self._response.body.read()
                if cache_content and not self._uncached_read_occurred:
                    self._body = data
                else:
                    self._uncached_read_occurred = True
            if self.length_remaining is not None:
                self.length_remaining = max(self.length_remaining - len(data), 0)
            if len(data) == 0 or (
                self.length_is_certain and self.length_remaining == 0
            ):
                # definitely finished reading, close response stream
                self._response.body.close()
            return typing.cast(bytes, data)

    def read_chunked(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
    ) -> typing.Generator[bytes]:
        # chunked is handled by browser
        while True:
            bytes = self.read(amt, decode_content)
            if not bytes:
                break
            yield bytes

    def release_conn(self) -> None:
        if not self._pool or not self._connection:
            return None

        self._pool._put_conn(self._connection)
        self._connection = None

    def drain_conn(self) -> None:
        self.close()

    @property
    def data(self) -> bytes:
        if self._body:
            return self._body
        else:
            return self.read(cache_content=True)

    def json(self) -> typing.Any:
        """
        Deserializes the body of the HTTP response as a Python object.

        The body of the HTTP response must be encoded using UTF-8, as per
        `RFC 8529 Section 8.1 <https://www.rfc-editor.org/rfc/rfc8259#section-8.1>`_.

        To use a custom JSON decoder pass the result of :attr:`HTTPResponse.data` to
        your custom decoder instead.

        If the body of the HTTP response is not decodable to UTF-8, a
        `UnicodeDecodeError` will be raised. If the body of the HTTP response is not a
        valid JSON document, a `json.JSONDecodeError` will be raised.

        Read more :ref:`here <json_content>`.

        :returns: The body of the HTTP response as a Python object.
        """
        data = self.data.decode("utf-8")
        return _json.loads(data)

    def close(self) -> None:
        if not self._closed:
            if isinstance(self._response.body, IOBase):
                self._response.body.close()
            if self._connection:
                self._connection.close()
                self._connection = None
            self._closed = True

    @contextmanager
    def _error_catcher(self) -> typing.Generator[None]:
        """
        Catch Emscripten specific exceptions thrown by fetch.py,
        instead re-raising urllib3 variants, so that low-level exceptions
        are not leaked in the high-level api.

        On exit, release the connection back to the pool.
        """
        from .fetch import _RequestError, _TimeoutError  # avoid circular import

        clean_exit = False

        try:
            yield
            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        except _TimeoutError as e:
            raise TimeoutError(str(e))
        except _RequestError as e:
            raise HTTPException(str(e))
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                # The response may not be closed but we're not going to use it
                # anymore so close it now
                if (
                    isinstance(self._response.body, IOBase)
                    and not self._response.body.closed
                ):
                    self._response.body.close()
                # release the connection back to the pool
                self.release_conn()
            else:
                # If we have read everything from the response stream,
                # return the connection back to the pool.
                if (
                    isinstance(self._response.body, IOBase)
                    and self._response.body.closed
                ):
                    self.release_conn()
<|endoftext|>"""
Support for streaming http requests in emscripten.

A few caveats -

If your browser (or Node.js) has WebAssembly JavaScript Promise Integration enabled
https://github.com/WebAssembly/js-promise-integration/blob/main/proposals/js-promise-integration/Overview.md
*and* you launch pyodide using `pyodide.runPythonAsync`, this will fetch data using the
JavaScript asynchronous fetch api (wrapped via `pyodide.ffi.call_sync`). In this case
timeouts and streaming should just work.

Otherwise, it uses a combination of XMLHttpRequest and a web-worker for streaming.

This approach has several caveats:

Firstly, you can't do streaming http in the main UI thread, because atomics.wait isn't allowed.
Streaming only works if you're running pyodide in a web worker.

Secondly, this uses an extra web worker and SharedArrayBuffer to do the asynchronous fetch
operation, so it requires that you have crossOriginIsolation enabled, by serving over https
(or from localhost) with the two headers below set:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

You can tell if cross origin isolation is successfully enabled by looking at the global crossOriginIsolated variable in
JavaScript console. If it isn't, streaming requests will fallback to XMLHttpRequest, i.e. getting the whole
request into a buffer and then returning it. it shows a warning in the JavaScript console in this case.

Finally, the webworker which does the streaming fetch is created on initial import, but will only be started once
control is returned to javascript. Call `await wait_for_streaming_ready()` to wait for streaming fetch.

NB: in this code, there are a lot of JavaScript objects. They are named js_*
to make it clear what type of object they are.
"""

from __future__ import annotations

import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
    JsArray,
    JsException,
    JsProxy,
    to_js,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer

from .request import EmscriptenRequest
from .response import EmscriptenResponse

"""
There are some headers that trigger unintended CORS preflight requests.
See also https://github.com/koenvo/pyodide-http/issues/22
"""
HEADERS_TO_IGNORE = ("user-agent",)

SUCCESS_HEADER = -1
SUCCESS_EOF = -2
ERROR_TIMEOUT = -3
ERROR_EXCEPTION = -4


class _RequestError(Exception):
    def __init__(
        self,
        message: str | None = None,
        *,
        request: EmscriptenRequest | None = None,
        response: EmscriptenResponse | None = None,
    ):
        self.request = request
        self.response = response
        self.message = message
        super().__init__(self.message)


class _StreamingError(_RequestError):
    pass


class _TimeoutError(_RequestError):
    pass


def _obj_from_dict(dict_val: dict[str, Any]) -> JsProxy:
    return to_js(dict_val, dict_converter=js.Object.fromEntries)


class _ReadStream(io.RawIOBase):
    def __init__(
        self,
        int_buffer: JsArray,
        byte_buffer: JsArray,
        timeout: float,
        worker: JsProxy,
        connection_id: int,
        request: EmscriptenRequest,
    ):
        self.int_buffer = int_buffer
        self.byte_buffer = byte_buffer
        self.read_pos = 0
        self.read_len = 0
        self.connection_id = connection_id
        self.worker = worker
        self.timeout = int(1000 * timeout) if timeout > 0 else None
        self.is_live = True
        self._is_closed = False
        self.request: EmscriptenRequest | None = request

    def __del__(self) -> None:
        self.close()

    # this is compatible with _base_connection
    def is_closed(self) -> bool:
        return self._is_closed

    # for compatibility with RawIOBase
    @property
    def closed(self) -> bool:
        return self.is_closed()

    def close(self) -> None:
        if self.is_closed():
            return
        self.read_len = 0
        self.read_pos = 0
        self.int_buffer = None
        self.byte_buffer = None
        self._is_closed = True
        self.request = None
        if self.is_live:
            self.worker.postMessage(_obj_from_dict({"close": self.connection_id}))
            self.is_live = False
        super().close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def readinto(self, byte_obj: Buffer) -> int:
        if not self.int_buffer:
            raise _StreamingError(
                "No buffer for stream in _ReadStream.readinto",
                request=self.request,
                response=None,
            )
        if self.read_len == 0:
            # wait for the worker to send something
            js.Atomics.store(self.int_buffer, 0, ERROR_TIMEOUT)
            self.worker.postMessage(_obj_from_dict({"getMore": self.connection_id}))
            if (
                js.Atomics.wait(self.int_buffer, 0, ERROR_TIMEOUT, self.timeout)
                == "timed-out"
            ):
                raise _TimeoutError
            data_len = self.int_buffer[0]
            if data_len > 0:
                self.read_len = data_len
                self.read_pos = 0
            elif data_len == ERROR_EXCEPTION:
                string_len = self.int_buffer[1]
                # decode the error string
                js_decoder = js.TextDecoder.new()
                json_str = js_decoder.decode(self.byte_buffer.slice(0, string_len))
                raise _StreamingError(
                    f"Exception thrown in fetch: {json_str}",
                    request=self.request,
                    response=None,
                )
            else:
                # EOF, free the buffers and return zero
                # and free the request
                self.is_live = False
                self.close()
                return 0
        # copy from int32array to python bytes
        ret_length = min(self.read_len, len(memoryview(byte_obj)))
        subarray = self.byte_buffer.subarray(
            self.read_pos, self.read_pos + ret_length
        ).to_py()
        memoryview(byte_obj)[0:ret_length] = subarray
        self.read_len -= ret_length
        self.read_pos += ret_length
        return ret_length


class _StreamingFetcher:
    def __init__(self) -> None:
        # make web-worker and data buffer on startup
        self.streaming_ready = False
        streaming_worker_code = (
            files(__package__)
            .joinpath("emscripten_fetch_worker.js")
            .read_text(encoding="utf-8")
        )
        js_data_blob = js.Blob.new(
            to_js([streaming_worker_code], create_pyproxies=False),
            _obj_from_dict({"type": "application/javascript"}),
        )

        def promise_resolver(js_resolve_fn: JsProxy, js_reject_fn: JsProxy) -> None:
            def onMsg(e: JsProxy) -> None:
                self.streaming_ready = True
                js_resolve_fn(e)

            def onErr(e: JsProxy) -> None:
                js_reject_fn(e)  # Defensive: never happens in ci

            self.js_worker.onmessage = onMsg
            self.js_worker.onerror = onErr

        js_data_url = js.URL.createObjectURL(js_data_blob)
        self.js_worker = js.globalThis.Worker.new(js_data_url)
        self.js_worker_ready_promise = js.globalThis.Promise.new(promise_resolver)

    def send(self, request: EmscriptenRequest) -> EmscriptenResponse:
        headers = {
            k: v for k, v in request.headers.items() if k not in HEADERS_TO_IGNORE
        }

        body = request.body
        fetch_data = {"headers": headers, "body": to_js(body), "method": request.method}
        # start the request off in the worker
        timeout = int(1000 * request.timeout) if request.timeout > 0 else None
        js_shared_buffer = js.SharedArrayBuffer.new(1048576)
        js_int_buffer = js.Int32Array.new(js_shared_buffer)
        js_byte_buffer = js.Uint8Array.new(js_shared_buffer, 8)

        js.Atomics.store(js_int_buffer, 0, ERROR_TIMEOUT)
        js.Atomics.notify(js_int_buffer, 0)
        js_absolute_url = js.URL.new(request.url, js.location).href
        self.js_worker.postMessage(
            _obj_from_dict(
                {
                    "buffer": js_shared_buffer,
                    "url": js_absolute_url,
                    "fetchParams": fetch_data,
                }
            )
        )
        # wait for the worker to send something
        js.Atomics.wait(js_int_buffer, 0, ERROR_TIMEOUT, timeout)
        if js_int_buffer[0] == ERROR_TIMEOUT:
            raise _TimeoutError(
                "Timeout connecting to streaming request",
                request=request,
                response=None,
            )
        elif js_int_buffer[0] == SUCCESS_HEADER:
            # got response
            # header length is in second int of intBuffer
            string_len = js_int_buffer[1]
            # decode the rest to a JSON string
            js_decoder = js.TextDecoder.new()
            # this does a copy (the slice) because decode can't work on shared array
            # for some silly reason
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            # get it as an object
            response_obj = json.loads(json_str)
            return EmscriptenResponse(
                request=request,
                status_code=response_obj["status"],
                headers=response_obj["headers"],
                body=_ReadStream(
                    js_int_buffer,
                    js_byte_buffer,
                    request.timeout,
                    self.js_worker,
                    response_obj["connectionID"],
                    request,
                ),
            )
        elif js_int_buffer[0] == ERROR_EXCEPTION:
            string_len = js_int_buffer[1]
            # decode the error string
            js_decoder = js.TextDecoder.new()
            json_str = js_decoder.decode(js_byte_buffer.slice(0, string_len))
            raise _StreamingError(
                f"Exception thrown in fetch: {json_str}", request=request, response=None
            )
        else:
            raise _StreamingError(
                f"Unknown status from worker in fetch: {js_int_buffer[0]}",
                request=request,
                response=None,
            )


class _JSPIReadStream(io.RawIOBase):
    """
    A read stream that uses pyodide.ffi.run_sync to read from a JavaScript fetch
    response. This requires support for WebAssembly JavaScript Promise Integration
    in the containing browser, and for pyodide to be launched via runPythonAsync.

    :param js_read_stream:
        The JavaScript stream reader

    :param timeout:
        Timeout in seconds

    :param request:
        The request we're handling

    :param response:
        The response this stream relates to

    :param js_abort_controller:
        A JavaScript AbortController object, used for timeouts
    """

    def __init__(
        self,
        js_read_stream: Any,
        timeout: float,
        request: EmscriptenRequest,
        response: EmscriptenResponse,
        js_abort_controller: Any,  # JavaScript AbortController for timeouts
    ):
        self.js_read_stream = js_read_stream
        self.timeout = timeout
        self._is_closed = False
        self._is_done = False
        self.request: EmscriptenRequest | None = request
        self.response: EmscriptenResponse | None = response
        self.current_buffer = None
        self.current_buffer_pos = 0
        self.js_abort_controller = js_abort_controller

    def __del__(self) -> None:
        self.close()

    # this is compatible with _base_connection
    def is_closed(self) -> bool:
        return self._is_closed

    # for compatibility with RawIOBase
    @property
    def closed(self) -> bool:
        return self.is_closed()

    def close(self) -> None:
        if self.is_closed():
            return
        self.read_len = 0
        self.read_pos = 0
        self.js_read_stream.cancel()
        self.js_read_stream = None
        self._is_closed = True
        self._is_done = True
        self.request = None
        self.response = None
        super().close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def _get_next_buffer(self) -> bool:
        result_js = _run_sync_with_timeout(
            self.js_read_stream.read(),
            self.timeout,
            self.js_abort_controller,
            request=self.request,
            response=self.response,
        )
        if result_js.done:
            self._is_done = True
            return False
        else:
            self.current_buffer = result_js.value.to_py()
            self.current_buffer_pos = 0
            return True

    def readinto(self, byte_obj: Buffer) -> int:
        if self.current_buffer is None:
            if not self._get_next_buffer() or self.current_buffer is None:
                self.close()
                return 0
        ret_length = min(
            len(byte_obj), len(self.current_buffer) - self.current_buffer_pos
        )
        byte_obj[0:ret_length] = self.current_buffer[
            self.current_buffer_pos : self.current_buffer_pos + ret_length
        ]
        self.current_buffer_pos += ret_length
        if self.current_buffer_pos == len(self.current_buffer):
            self.current_buffer = None
        return ret_length


# check if we are in a worker or not
def is_in_browser_main_thread() -> bool:
    return hasattr(js, "window") and hasattr(js, "self") and js.self == js.window


def is_cross_origin_isolated() -> bool:
    return hasattr(js, "crossOriginIsolated") and js.crossOriginIsolated


def is_in_node() -> bool:
    return (
        hasattr(js, "process")
        and hasattr(js.process, "release")
        and hasattr(js.process.release, "name")
        and js.process.release.name == "node"
    )


def is_worker_available() -> bool:
    return hasattr(js, "Worker") and hasattr(js, "Blob")


_fetcher: _StreamingFetcher | None = None

if is_worker_available() and (
    (is_cross_origin_isolated() and not is_in_browser_main_thread())
    and (not is_in_node())
):
    _fetcher = _StreamingFetcher()
else:
    _fetcher = None


NODE_JSPI_ERROR = (
    "urllib3 only works in Node.js with pyodide.runPythonAsync"
    " and requires the flag --experimental-wasm-stack-switching in "
    " versions of node <24."
)


def send_streaming_request(request: EmscriptenRequest) -> EmscriptenResponse | None:
    if has_jspi():
        return send_jspi_request(request, True)
    elif is_in_node():
        raise _RequestError(
            message=NODE_JSPI_ERROR,
            request=request,
            response=None,
        )

    if _fetcher and streaming_ready():
        return _fetcher.send(request)
    else:
        _show_streaming_warning()
        return None


_SHOWN_TIMEOUT_WARNING = False


def _show_timeout_warning() -> None:
    global _SHOWN_TIMEOUT_WARNING
    if not _SHOWN_TIMEOUT_WARNING:
        _SHOWN_TIMEOUT_WARNING = True
        message = "Warning: Timeout is not available on main browser thread"
        js.console.warn(message)


_SHOWN_STREAMING_WARNING = False


def _show_streaming_warning() -> None:
    global _SHOWN_STREAMING_WARNING
    if not _SHOWN_STREAMING_WARNING:
        _SHOWN_STREAMING_WARNING = True
        message = "Can't stream HTTP requests because: \n"
        if not is_cross_origin_isolated():
            message += "  Page is not cross-origin isolated\n"
        if is_in_browser_main_thread():
            message += "  Python is running in main browser thread\n"
        if not is_worker_available():
            message += " Worker or Blob classes are not available in this environment."  # Defensive: this is always False in browsers that we test in
        if streaming_ready() is False:
            message += """ Streaming fetch worker isn't ready. If you want to be sure that streaming fetch
is working, you need to call: 'await urllib3.contrib.emscripten.fetch.wait_for_streaming_ready()`"""
        from js import console

        console.warn(message)


def send_request(request: EmscriptenRequest) -> EmscriptenResponse:
    if has_jspi():
        return send_jspi_request(request, False)
    elif is_in_node():
        raise _RequestError(
            message=NODE_JSPI_ERROR,
            request=request,
            response=None,
        )
    try:
        js_xhr = js.XMLHttpRequest.new()

        if not is_in_browser_main_thread():
            js_xhr.responseType = "arraybuffer"
            if request.timeout:
                js_xhr.timeout = int(request.timeout * 1000)
        else:
            js_xhr.overrideMimeType("text/plain; charset=ISO-8859-15")
            if request.timeout:
                # timeout isn't available on the main thread - show a warning in console
                # if it is set
                _show_timeout_warning()

        js_xhr.open(request.method, request.url, False)
        for name, value in request.headers.items():
            if name.lower() not in HEADERS_TO_IGNORE:
                js_xhr.setRequestHeader(name, value)

        js_xhr.send(to_js(request.body))

        headers = dict(Parser().parsestr(js_xhr.getAllResponseHeaders()))

        if not is_in_browser_main_thread():
            body = js_xhr.response.to_py().tobytes()
        else:
            body = js_xhr.response.encode("ISO-8859-15")
        return EmscriptenResponse(
            status_code=js_xhr.status, headers=headers, body=body, request=request
        )
    except JsException as err:
        if err.name == "TimeoutError":
            raise _TimeoutError(err.message, request=request)
        elif err.name == "NetworkError":
            raise _RequestError(err.message, request=request)
        else:
            # general http error
            raise _RequestError(err.message, request=request)


def send_jspi_request(
    request: EmscriptenRequest, streaming: bool
) -> EmscriptenResponse:
    """
    Send a request using WebAssembly JavaScript Promise Integration
    to wrap the asynchronous JavaScript fetch api (experimental).

    :param request:
        Request to send

    :param streaming:
        Whether to stream the response

    :return: The response object
    :rtype: EmscriptenResponse
    """
    timeout = request.timeout
    js_abort_controller = js.AbortController.new()
    headers = {k: v for k, v in request.headers.items() if k not in HEADERS_TO_IGNORE}
    req_body = request.body
    fetch_data = {
        "headers": headers,
        "body": to_js(req_body),
        "method": request.method,
        "signal": js_abort_controller.signal,
    }
    # Node.js returns the whole response (unlike opaqueredirect in browsers),
    # so urllib3 can set `redirect: manual` to control redirects itself.
    # https://stackoverflow.com/a/78524615
    if _is_node_js():
        fetch_data["redirect"] = "manual"
    # Call JavaScript fetch (async api, returns a promise)
    fetcher_promise_js = js.fetch(request.url, _obj_from_dict(fetch_data))
    # Now suspend WebAssembly until we resolve that promise
    # or time out.
    response_js = _run_sync_with_timeout(
        fetcher_promise_js,
        timeout,
        js_abort_controller,
        request=request,
        response=None,
    )
    headers = {}
    header_iter = response_js.headers.entries()
    while True:
        iter_value_js = header_iter.next()
        if getattr(iter_value_js, "done", False):
            break
        else:
            headers[str(iter_value_js.value[0])] = str(iter_value_js.value[1])
    status_code = response_js.status
    body: bytes | io.RawIOBase = b""

    response = EmscriptenResponse(
        status_code=status_code, headers=headers, body=b"", request=request
    )
    if streaming:
        # get via inputstream
        if response_js.body is not None:
            # get a reader from the fetch response
            body_stream_js = response_js.body.getReader()
            body = _JSPIReadStream(
                body_stream_js, timeout, request, response, js_abort_controller
            )
    else:
        # get directly via arraybuffer
        # n.b. this is another async JavaScript call.
        body = _run_sync_with_timeout(
            response_js.arrayBuffer(),
            timeout,
            js_abort_controller,
            request=request,
            response=response,
        ).to_py()
    response.body = body
    return response


def _run_sync_with_timeout(
    promise: Any,
    timeout: float,
    js_abort_controller: Any,
    request: EmscriptenRequest | None,
    response: EmscriptenResponse | None,
) -> Any:
    """
    Await a JavaScript promise synchronously with a timeout which is implemented
    via the AbortController

    :param promise:
        Javascript promise to await

    :param timeout:
        Timeout in seconds

    :param js_abort_controller:
        A JavaScript AbortController object, used on timeout

    :param request:
        The request being handled

    :param response:
        The response being handled (if it exists yet)

    :raises _TimeoutError: If the request times out
    :raises _RequestError: If the request raises a JavaScript exception

    :return: The result of awaiting the promise.
    """
    timer_id = None
    if timeout > 0:
        timer_id = js.setTimeout(
            js_abort_controller.abort.bind(js_abort_controller), int(timeout * 1000)
        )
    try:
        from pyodide.ffi import run_sync

        # run_sync here uses WebAssembly JavaScript Promise Integration to
        # suspend python until the JavaScript promise resolves.
        return run_sync(promise)
    except JsException as err:
        if err.name == "AbortError":
            raise _TimeoutError(
                message="Request timed out", request=request, response=response
            )
        else:
            raise _RequestError(message=err.message, request=request, response=response)
    finally:
        if timer_id is not None:
            js.clearTimeout(timer_id)


def has_jspi() -> bool:
    """
    Return true if jspi can be used.

    This requires both browser support and also WebAssembly
    to be in the correct state - i.e. that the javascript
    call into python was async not sync.

    :return: True if jspi can be used.
    :rtype: bool
    """
    try:
        from pyodide.ffi import can_run_sync, run_sync  # noqa: F401

        return bool(can_run_sync())
    except ImportError:
        return False


def _is_node_js() -> bool:
    """
    Check if we are in Node.js.

    :return: True if we are in Node.js.
    :rtype: bool
    """
    return (
        hasattr(js, "process")
        and hasattr(js.process, "release")
        # According to the Node.js documentation, the release name is always "node".
        and js.process.release.name == "node"
    )


def streaming_ready() -> bool | None:
    if _fetcher:
        return _fetcher.streaming_ready
    else:
        return None  # no fetcher, return None to signify that


async def wait_for_streaming_ready() -> bool:
    if _fetcher:
        await _fetcher.js_worker_ready_promise
        return True
    else:
        return False
<|endoftext|>from __future__ import annotations

import urllib3.connection

from ...connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from .connection import EmscriptenHTTPConnection, EmscriptenHTTPSConnection


def inject_into_urllib3() -> None:
    # override connection classes to use emscripten specific classes
    # n.b. mypy complains about the overriding of classes below
    # if it isn't ignored
    HTTPConnectionPool.ConnectionCls = EmscriptenHTTPConnection
    HTTPSConnectionPool.ConnectionCls = EmscriptenHTTPSConnection
    urllib3.connection.HTTPConnection = EmscriptenHTTPConnection  # type: ignore[misc,assignment]
    urllib3.connection.HTTPSConnection = EmscriptenHTTPSConnection  # type: ignore[misc,assignment]
    urllib3.connection.VerifiedHTTPSConnection = EmscriptenHTTPSConnection  # type: ignore[assignment]
<|endoftext|>from __future__ import annotations

import os
import typing

# use http.client.HTTPException for consistency with non-emscripten
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady

from ..._base_connection import _TYPE_BODY
from ...connection import HTTPConnection, ProxyConfig, port_by_scheme
from ...exceptions import TimeoutError
from ...response import BaseHTTPResponse
from ...util.connection import _TYPE_SOCKET_OPTIONS
from ...util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from ...util.url import Url
from .fetch import _RequestError, _TimeoutError, send_request, send_streaming_request
from .request import EmscriptenRequest
from .response import EmscriptenHttpResponseWrapper, EmscriptenResponse

if typing.TYPE_CHECKING:
    from ..._base_connection import BaseHTTPConnection, BaseHTTPSConnection


class EmscriptenHTTPConnection:
    default_port: typing.ClassVar[int] = port_by_scheme["http"]
    default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]

    timeout: None | (float)

    host: str
    port: int
    blocksize: int
    source_address: tuple[str, int] | None
    socket_options: _TYPE_SOCKET_OPTIONS | None

    proxy: Url | None
    proxy_config: ProxyConfig | None

    is_verified: bool = False
    proxy_is_verified: bool | None = None

    response_class: type[BaseHTTPResponse] = EmscriptenHttpResponseWrapper
    _response: EmscriptenResponse | None

    def __init__(
        self,
        host: str,
        port: int = 0,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 8192,
        socket_options: _TYPE_SOCKET_OPTIONS | None = None,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout if isinstance(timeout, float) else 0.0
        self.scheme = "http"
        self._closed = True
        self._response = None
        # ignore these things because we don't
        # have control over that stuff
        self.proxy = None
        self.proxy_config = None
        self.blocksize = blocksize
        self.source_address = None
        self.socket_options = None
        self.is_verified = False

    def set_tunnel(
        self,
        host: str,
        port: int | None = 0,
        headers: typing.Mapping[str, str] | None = None,
        scheme: str = "http",
    ) -> None:
        pass

    def connect(self) -> None:
        pass

    def request(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        # We know *at least* botocore is depending on the order of the
        # first 3 parameters so to be safe we only mark the later ones
        # as keyword-only to ensure we have space to extend.
        *,
        chunked: bool = False,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
    ) -> None:
        self._closed = False
        if url.startswith("/"):
            if self.port is not None:
                port = f":{self.port}"
            else:
                port = ""
            # no scheme / host / port included, make a full url
            url = f"{self.scheme}://{self.host}{port}{url}"
        request = EmscriptenRequest(
            url=url,
            method=method,
            timeout=self.timeout if self.timeout else 0,
            decode_content=decode_content,
        )
        request.set_body(body)
        if headers:
            for k, v in headers.items():
                request.set_header(k, v)
        self._response = None
        try:
            if not preload_content:
                self._response = send_streaming_request(request)
            if self._response is None:
                self._response = send_request(request)
        except _TimeoutError as e:
            raise TimeoutError(e.message) from e
        except _RequestError as e:
            raise HTTPException(e.message) from e

    def getresponse(self) -> BaseHTTPResponse:
        if self._response is not None:
            return EmscriptenHttpResponseWrapper(
                internal_response=self._response,
                url=self._response.request.url,
                connection=self,
            )
        else:
            raise ResponseNotReady()

    def close(self) -> None:
        self._closed = True
        self._response = None

    @property
    def is_closed(self) -> bool:
        """Whether the connection either is brand new or has been previously closed.
        If this property is True then both ``is_connected`` and ``has_connected_to_proxy``
        properties must be False.
        """
        return self._closed

    @property
    def is_connected(self) -> bool:
        """Whether the connection is actively connected to any origin (proxy or target)"""
        return True

    @property
    def has_connected_to_proxy(self) -> bool:
        """Whether the connection has successfully connected to its proxy.
        This returns False if no proxy is in use. Used to determine whether
        errors are coming from the proxy layer or from tunnelling to the target origin.
        """
        return False


class EmscriptenHTTPSConnection(EmscriptenHTTPConnection):
    default_port = port_by_scheme["https"]
    # all this is basically ignored, as browser handles https
    cert_reqs: int | str | None = None
    ca_certs: str | None = None
    ca_cert_dir: str | None = None
    ca_cert_data: None | str | bytes = None
    cert_file: str | None
    key_file: str | None
    key_password: str | None
    ssl_context: typing.Any | None
    ssl_version: int | str | None = None
    ssl_minimum_version: int | None = None
    ssl_maximum_version: int | None = None
    assert_hostname: None | str | typing.Literal[False]
    assert_fingerprint: str | None = None

    def __init__(
        self,
        host: str,
        port: int = 0,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: (
            None | _TYPE_SOCKET_OPTIONS
        ) = HTTPConnection.default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
        cert_reqs: int | str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        server_hostname: str | None = None,
        ssl_context: typing.Any | None = None,
        ca_certs: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
        ssl_minimum_version: int | None = None,
        ssl_maximum_version: int | None = None,
        ssl_version: int | str | None = None,  # Deprecated
        cert_file: str | None = None,
        key_file: str | None = None,
        key_password: str | None = None,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            socket_options=socket_options,
            proxy=proxy,
            proxy_config=proxy_config,
        )
        self.scheme = "https"

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ssl_version = ssl_version
        self.ssl_minimum_version = ssl_minimum_version
        self.ssl_maximum_version = ssl_maximum_version
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

        self.cert_reqs = None

        # The browser will automatically verify all requests.
        # We have no control over that setting.
        self.is_verified = True

    def set_cert(
        self,
        key_file: str | None = None,
        cert_file: str | None = None,
        cert_reqs: int | str | None = None,
        key_password: str | None = None,
        ca_certs: str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
    ) -> None:
        pass


# verify that this class implements BaseHTTP(s) connection correctly
if typing.TYPE_CHECKING:
    _supports_http_protocol: BaseHTTPConnection = EmscriptenHTTPConnection("", 0)
    _supports_https_protocol: BaseHTTPSConnection = EmscriptenHTTPSConnection("", 0)
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass, field

from ..._base_connection import _TYPE_BODY


@dataclass
class EmscriptenRequest:
    method: str
    url: str
    params: dict[str, str] | None = None
    body: _TYPE_BODY | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 0
    decode_content: bool = True

    def set_header(self, name: str, value: str) -> None:
        self.headers[name.capitalize()] = value

    def set_body(self, body: _TYPE_BODY | None) -> None:
        self.body = body
<|endoftext|>from __future__ import annotations

import re
import typing
from functools import partial
from urllib.parse import unquote as _unquote

from ..exceptions import LocationParseError
from .util import to_str

# We only want to normalize urls with an HTTP(S) scheme.
# urllib3 infers URLs without a scheme (None) to be http.
_NORMALIZABLE_SCHEMES = ("http", "https", None)

# Almost all of these patterns were derived from the
# 'rfc3986' module: https://github.com/python-hyper/rfc3986
_PERCENT_RE = re.compile(r"%[a-fA-F0-9]{2}")
_HOST_PERCENT_RE = re.compile(r"%[a-fA-F0-9]{2}|%")
_SCHEME_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+-]*:|/)")
_HOST_INVALID_CHAR_RE = re.compile(r"[\x00-\x20\x7f]")
_URI_RE = re.compile(
    r"^(?:([a-zA-Z][a-zA-Z0-9+.-]*):)?"
    r"(?://([^\\/?#]*))?"
    r"([^?#]*)"
    r"(?:\?([^#]*))?"
    r"(?:#(.*))?$",
    re.UNICODE | re.DOTALL,
)

_IPV4_PAT = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
_HEX_PAT = "[0-9A-Fa-f]{1,4}"
_LS32_PAT = "(?:{hex}:{hex}|{ipv4})".format(hex=_HEX_PAT, ipv4=_IPV4_PAT)
_subs = {"hex": _HEX_PAT, "ls32": _LS32_PAT}
_variations = [
    #                            6( h16 ":" ) ls32
    "(?:%(hex)s:){6}%(ls32)s",
    #                       "::" 5( h16 ":" ) ls32
    "::(?:%(hex)s:){5}%(ls32)s",
    # [               h16 ] "::" 4( h16 ":" ) ls32
    "(?:%(hex)s)?::(?:%(hex)s:){4}%(ls32)s",
    # [ *1( h16 ":" ) h16 ] "::" 3( h16 ":" ) ls32
    "(?:(?:%(hex)s:)?%(hex)s)?::(?:%(hex)s:){3}%(ls32)s",
    # [ *2( h16 ":" ) h16 ] "::" 2( h16 ":" ) ls32
    "(?:(?:%(hex)s:){0,2}%(hex)s)?::(?:%(hex)s:){2}%(ls32)s",
    # [ *3( h16 ":" ) h16 ] "::"    h16 ":"   ls32
    "(?:(?:%(hex)s:){0,3}%(hex)s)?::%(hex)s:%(ls32)s",
    # [ *4( h16 ":" ) h16 ] "::"              ls32
    "(?:(?:%(hex)s:){0,4}%(hex)s)?::%(ls32)s",
    # [ *5( h16 ":" ) h16 ] "::"              h16
    "(?:(?:%(hex)s:){0,5}%(hex)s)?::%(hex)s",
    # [ *6( h16 ":" ) h16 ] "::"
    "(?:(?:%(hex)s:){0,6}%(hex)s)?::",
]

_UNRESERVED_PAT = r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._\-~"
_IPV6_PAT = "(?:" + "|".join([x % _subs for x in _variations]) + ")"
_ZONE_ID_PAT = "(?:%25|%)(?:[" + _UNRESERVED_PAT + "]|%[a-fA-F0-9]{2})+"
_IPV6_ADDRZ_PAT = r"\[" + _IPV6_PAT + r"(?:" + _ZONE_ID_PAT + r")?\]"
_REG_NAME_PAT = r"(?:[^\[\]%:/?#]|%[a-fA-F0-9]{2})*"
_TARGET_RE = re.compile(r"^(/[^?#]*)(?:\?([^#]*))?(?:#.*)?$")

_IPV4_RE = re.compile(
    r"^(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:\.(?:0[xX][0-9a-fA-F]+|[0-9]+)){0,3}$"
)
_IPV6_RE = re.compile("^" + _IPV6_PAT + "$")
_IPV6_ADDRZ_RE = re.compile("^" + _IPV6_ADDRZ_PAT + "$")
_BRACELESS_IPV6_ADDRZ_RE = re.compile("^" + _IPV6_ADDRZ_PAT[2:-2] + "$")
_ZONE_ID_RE = re.compile("(" + _ZONE_ID_PAT + r")\]$")

_HOST_PORT_PAT = ("^(%s|%s|%s)(?::0*?(|0|[1-9][0-9]{0,4}))?$") % (
    _REG_NAME_PAT,
    _IPV4_PAT,
    _IPV6_ADDRZ_PAT,
)
_HOST_PORT_RE = re.compile(_HOST_PORT_PAT, re.UNICODE | re.DOTALL)

_UNRESERVED_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~"
)
_SUB_DELIM_CHARS = set("!$&'()*+,;=")
_USERINFO_CHARS = _UNRESERVED_CHARS | _SUB_DELIM_CHARS | {":"}
_PATH_CHARS = _USERINFO_CHARS | {"@", "/"}
_QUERY_CHARS = _FRAGMENT_CHARS = _PATH_CHARS | {"?"}


class Url(
    typing.NamedTuple(
        "Url",
        [
            ("scheme", typing.Optional[str]),
            ("auth", typing.Optional[str]),
            ("host", typing.Optional[str]),
            ("port", typing.Optional[int]),
            ("path", typing.Optional[str]),
            ("query", typing.Optional[str]),
            ("fragment", typing.Optional[str]),
        ],
    )
):
    """
    Data structure for representing an HTTP URL. Used as a return value for
    :func:`parse_url`. Both the scheme and host are normalized as they are
    both case-insensitive according to RFC 3986.

    :param auth: User information as defined in RFC 3986 3.2.1. This
        component is kept percent-encoded. Use :attr:`auth_decoded` or
        :attr:`auth_decoded_joined` to get the decoded form.
    """

    def __new__(  # type: ignore[no-untyped-def]
        cls,
        scheme: str | None = None,
        auth: str | None = None,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        query: str | None = None,
        fragment: str | None = None,
    ):
        if path and not path.startswith("/"):
            path = "/" + path
        if scheme is not None:
            scheme = scheme.lower()
        return super().__new__(cls, scheme, auth, host, port, path, query, fragment)

    @property
    def auth_decoded(self) -> tuple[None, None] | tuple[str, str | None]:
        """
        User information with %-escapes decoded as UTF-8, returned as a
        ``(username, password)`` tuple.

        Both values are ``None`` if ``auth`` is ``None``.
        ``password`` is ``None`` if not present in the auth component.
        """
        if self.auth is None:
            return None, None
        username, sep, password = self.auth.partition(":")
        return (
            _unquote(username, encoding="utf-8"),
            _unquote(password, encoding="utf-8") if sep else None,
        )

    @property
    def auth_decoded_joined(self) -> str | None:
        """
        User information with %-escapes decoded as UTF-8, as a string
        prepared for encoding into an 'authorization: basic ...' header.

        This property does not choose the encoding used for an
        'authorization: basic ...' header. Use the ``basic_auth_encoding`` or
        ``proxy_basic_auth_encoding`` parameters of
        :func:`urllib3.util.make_headers` when converting the returned string
        into Basic authentication bytes.

        This is a convenience property that joins the username and
        password with a colon, if both are present.
        If only the username is present, a trailing colon is still
        appended.
        If ``auth`` is ``None``, this returns ``None``.
        """
        username, password = self.auth_decoded
        if username is None:
            return None
        return f"{username}:{password or ''}"

    @property
    def hostname(self) -> str | None:
        """For backwards-compatibility with urlparse. We're nice like that."""
        return self.host

    @property
    def request_uri(self) -> str:
        """Absolute path including the query string."""
        uri = self.path or "/"

        if self.query is not None:
            uri += "?" + self.query

        return uri

    @property
    def authority(self) -> str | None:
        """
        Authority component as defined in RFC 3986 3.2.
        This includes userinfo (auth), host and port.

        i.e.
            userinfo@host:port
        """
        userinfo = self.auth
        netloc = self.netloc
        if netloc is None or userinfo is None:
            return netloc
        else:
            return f"{userinfo}@{netloc}"

    @property
    def netloc(self) -> str | None:
        """
        Network location including host and port.

        If you need the equivalent of urllib.parse's ``netloc``,
        use the ``authority`` property instead.
        """
        if self.host is None:
            return None
        if self.port is not None:
            return f"{self.host}:{self.port}"
        return self.host

    @property
    def url(self) -> str:
        """
        Convert self into a url

        This function should more or less round-trip with :func:`.parse_url`. The
        returned url may not be exactly the same as the url inputted to
        :func:`.parse_url`, but it should be equivalent by the RFC (e.g., urls
        with a blank port will have : removed).

        Example:

        .. code-block:: python

            import urllib3

            U = urllib3.util.parse_url("https://google.com/mail/")

            print(U.url)
            # "https://google.com/mail/"

            print( urllib3.util.Url("https", "username:password",
                                    "host.com", 80, "/path", "query", "fragment"
                                    ).url
                )
            # "https://username:password@host.com:80/path?query#fragment"
        """
        scheme, auth, host, port, path, query, fragment = self
        url = ""

        # We use "is not None" we want things to happen with empty strings (or 0 port)
        if scheme is not None:
            url += scheme + "://"
        if auth is not None:
            url += auth + "@"
        if host is not None:
            url += host
        if port is not None:
            url += ":" + str(port)
        if path is not None:
            url += path
        if query is not None:
            url += "?" + query
        if fragment is not None:
            url += "#" + fragment

        return url

    def __str__(self) -> str:
        return self.url


@typing.overload
def _encode_invalid_chars(
    component: str, allowed_chars: typing.Container[str]
) -> str:  # Abstract
    ...


@typing.overload
def _encode_invalid_chars(
    component: None, allowed_chars: typing.Container[str]
) -> None:  # Abstract
    ...


def _encode_invalid_chars(
    component: str | None, allowed_chars: typing.Container[str]
) -> str | None:
    """Percent-encodes a URI component without reapplying
    onto an already percent-encoded component.
    """
    if component is None:
        return component

    component = to_str(component)

    # Fast path: an ASCII component that consists only of allowed characters
    # and contains no '%' needs neither encoding nor percent-normalization.
    # Components containing '%' always take the general path below because
    # how '%' is treated depends on whether the whole component is already
    # percent-encoded.
    if (
        "%" not in component
        and component.isascii()
        and all(map(allowed_chars.__contains__, component))
    ):
        return component

    # Normalize existing percent-encoded bytes.
    # Try to see if the component we're encoding is already percent-encoded
    # so we can skip all '%' characters but still encode all others.
    component, percent_encodings = _PERCENT_RE.subn(
        lambda match: match.group(0).upper(), component
    )

    uri_bytes = component.encode("utf-8", "surrogatepass")
    is_percent_encoded = percent_encodings == uri_bytes.count(b"%")
    encoded_component = bytearray()

    # Iterating over bytes yields integers; 0x25 is '%'.
    for byte_ord in uri_bytes:
        if (is_percent_encoded and byte_ord == 0x25) or (
            byte_ord < 128 and chr(byte_ord) in allowed_chars
        ):
            encoded_component.append(byte_ord)
        else:
            encoded_component += b"%%%02X" % byte_ord

    return encoded_component.decode()


def _remove_path_dot_segments(path: str) -> str:
    # See http://tools.ietf.org/html/rfc3986#section-5.2.4 for pseudo-code
    segments = path.split("/")  # Turn the path into a list of segments
    output = []  # Initialize the variable to use to store output

    for segment in segments:
        # '.' is the current directory, so ignore it, it is superfluous
        if segment == ".":
            continue
        # Anything other than '..', should be appended to the output
        if segment != "..":
            output.append(segment)
        # In this case segment == '..', if we can, we should pop the last
        # element
        elif output:
            output.pop()

    # If the path starts with '/' and the output is empty or the first string
    # is non-empty
    if path.startswith("/") and (not output or output[0]):
        output.insert(0, "")

    # If the path starts with '/.' or '/..' ensure we add one more empty
    # string to add a trailing '/'
    if path.endswith(("/.", "/..")):
        output.append("")

    return "/".join(output)


@typing.overload
def _normalize_host(host: None, scheme: str | None) -> None: ...


@typing.overload
def _normalize_host(host: str, scheme: str | None) -> str: ...


def _normalize_host(host: str | None, scheme: str | None) -> str | None:
    if host:
        invalid_host_char = _HOST_INVALID_CHAR_RE.search(host)
        if invalid_host_char:
            raise LocationParseError(
                f"Host {host!r} contains invalid character "
                f"{invalid_host_char.group()!r}"
            )
        if scheme in _NORMALIZABLE_SCHEMES:
            is_ipv6 = _IPV6_ADDRZ_RE.match(host)
            if is_ipv6:
                # IPv6 hosts of the form 'a::b%zone' are encoded in a URL as
                # such per RFC 6874: 'a::b%25zone'. Unquote the ZoneID
                # separator as necessary to return a valid RFC 4007 scoped IP.
                match = _ZONE_ID_RE.search(host)
                if match:
                    start, end = match.span(1)
                    zone_id = host[start:end]

                    if zone_id.startswith("%25") and zone_id != "%25":
                        zone_id = zone_id[3:]
                    else:
                        zone_id = zone_id[1:]
                    zone_id = _PERCENT_RE.sub(
                        partial(_normalize_zone_id_percent_encoding, error_host=host),
                        zone_id,
                    )
                    zone_id = _encode_invalid_chars(zone_id, _UNRESERVED_CHARS)
                    return f"{host[:start].lower()}%{zone_id}{host[end:]}"
                else:
                    return host.lower()
            elif not _IPV4_RE.match(host):
                if "%" in host:
                    host = _HOST_PERCENT_RE.sub(_normalize_host_percent_encoding, host)
                return to_str(
                    b".".join([_idna_encode(label) for label in host.split(".")]),
                    "ascii",
                )
    return host


def _decode_percent_encoding(match: re.Match[str], *, error_host: str) -> str:
    percent_encoded_octet = match.group(0)
    decoded_octet = chr(int(percent_encoded_octet[1:], 16))
    # Keep this narrower than _HOST_INVALID_CHAR_RE because "%20" is valid
    # percent-encoding and remains encoded.
    if decoded_octet < "\x20" or decoded_octet == "\x7f":
        raise LocationParseError(
            f"Host {error_host!r} contains invalid percent-encoded "
            f"control character {percent_encoded_octet!r}"
        )
    return decoded_octet


def _normalize_host_percent_encoding(match: re.Match[str]) -> str:
    # Reject invalid percent encodings
    if match.group(0) == "%":
        raise LocationParseError(f"{match.string!r} is not a valid host")
    decoded_octet = _decode_percent_encoding(match, error_host=match.string)
    if decoded_octet in _UNRESERVED_CHARS:
        return decoded_octet
    return match.group(0).upper()


def _normalize_zone_id_percent_encoding(
    match: re.Match[str], *, error_host: str
) -> str:
    _decode_percent_encoding(match, error_host=error_host)
    return match.group(0).upper()


def _idna_encode(name: str) -> bytes:
    if not name.isascii():
        try:
            import idna
        except ImportError:
            raise LocationParseError(
                "Unable to parse URL without the 'idna' module"
            ) from None

        try:
            return idna.encode(name.lower(), strict=True, std3_rules=True)
        except idna.IDNAError:
            raise LocationParseError(
                f"Name '{name}' is not a valid IDNA label"
            ) from None

    return _PERCENT_RE.sub(lambda match: match.group(0).upper(), name.lower()).encode(
        "ascii"
    )


def _encode_target(target: str) -> str:
    """Percent-encodes a request target so that there are no invalid characters

    Pre-condition for this function is that 'target' must start with '/'.
    If that is the case then _TARGET_RE will always produce a match.
    """
    match = _TARGET_RE.match(target)
    if not match:  # Defensive:
        raise LocationParseError(f"{target!r} is not a valid request URI")

    path, query = match.groups()
    encoded_target = _encode_invalid_chars(path, _PATH_CHARS)
    if query is not None:
        query = _encode_invalid_chars(query, _QUERY_CHARS)
        encoded_target += "?" + query
    return encoded_target


def parse_url(url: str) -> Url:
    """
    Given a url, return a parsed :class:`.Url` namedtuple. Best-effort is
    performed to parse incomplete urls. Fields not provided will be None.
    This parser is RFC 3986 and RFC 6874 compliant.

    The parser logic and helper functions are based heavily on
    work done in the ``rfc3986`` module.

    :param str url: URL to parse into a :class:`.Url` namedtuple.

    Partly backwards-compatible with :mod:`urllib.parse`.

    Example:

    .. code-block:: python

        import urllib3

        print( urllib3.util.parse_url('http://google.com/mail/'))
        # Url(scheme='http', host='google.com', port=None, path='/mail/', ...)

        print( urllib3.util.parse_url('google.com:80'))
        # Url(scheme=None, host='google.com', port=80, path=None, ...)

        print( urllib3.util.parse_url('/foo?bar'))
        # Url(scheme=None, host=None, port=None, path='/foo', query='bar', ...)
    """
    if not url:
        # Empty
        return Url()

    if not _SCHEME_RE.search(url):
        url = "//" + url

    scheme: str | None
    authority: str | None
    auth: str | None
    host: str | None
    port: str | None
    port_int: int | None
    path: str | None
    query: str | None
    fragment: str | None

    scheme, authority, path, query, fragment = _URI_RE.match(url).groups()  # type: ignore[union-attr]
    normalize_uri = scheme is None or scheme.lower() in _NORMALIZABLE_SCHEMES

    if scheme:
        scheme = scheme.lower()

    if authority:
        auth, _, host_port = authority.rpartition("@")
        auth = auth or None
        host_port_match = _HOST_PORT_RE.fullmatch(host_port)
        if not host_port_match:
            invalid_host_char = _HOST_INVALID_CHAR_RE.search(host_port)
            if invalid_host_char:
                raise LocationParseError(
                    f"Host {host_port!r} contains invalid character "
                    f"{invalid_host_char.group()!r}"
                )
            raise LocationParseError(f"{host_port!r} is not a valid host or port")
        host, port = host_port_match.groups()
        if auth and normalize_uri:
            auth = _encode_invalid_chars(auth, _USERINFO_CHARS)
        if port == "":
            port = None
    else:
        auth, host, port = None, None, None

    if port is not None:
        port_int = int(port)
        if not (0 <= port_int <= 65535):
            raise LocationParseError(url)
    else:
        port_int = None

    host = _normalize_host(host, scheme)

    if normalize_uri and path:
        path = _remove_path_dot_segments(path)
        path = _encode_invalid_chars(path, _PATH_CHARS)
    if normalize_uri and query:
        query = _encode_invalid_chars(query, _QUERY_CHARS)
    if normalize_uri and fragment:
        fragment = _encode_invalid_chars(fragment, _FRAGMENT_CHARS)

    # For the sake of backwards compatibility we put empty
    # string values for path if there are any defined values
    # beyond the path in the URL.
    # TODO: Remove this when we break backwards compatibility.
    if not path:
        if query is not None or fragment is not None:
            path = ""
        else:
            path = None

    return Url(
        scheme=scheme,
        auth=auth,
        host=host,
        port=port_int,
        path=path,
        query=query,
        fragment=fragment,
    )
<|endoftext|>from __future__ import annotations

import http.client as httplib
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect

from ..exceptions import HeaderParsingError


def is_fp_closed(obj: object) -> bool:
    """
    Checks whether a given file-like object is closed.

    :param obj:
        The file-like object to check.
    """

    try:
        # Check `isclosed()` first, in case Python3 doesn't set `closed`.
        # GH Issue #928
        return obj.isclosed()  # type: ignore[no-any-return, attr-defined]
    except AttributeError:
        pass

    try:
        # Check via the official file-like-object way.
        return obj.closed  # type: ignore[no-any-return, attr-defined]
    except AttributeError:
        pass

    try:
        # Check if the object is a container for another file-like object that
        # gets released on exhaustion (e.g. HTTPResponse).
        return obj.fp is None  # type: ignore[attr-defined]
    except AttributeError:
        pass

    raise ValueError("Unable to determine whether fp is closed.")


def assert_header_parsing(headers: httplib.HTTPMessage) -> None:
    """
    Asserts whether all headers have been successfully parsed.
    Extracts encountered errors from the result of parsing headers.

    Only works on Python 3.

    :param http.client.HTTPMessage headers: Headers to verify.

    :raises urllib3.exceptions.HeaderParsingError:
        If parsing errors are found.
    """

    # This will fail silently if we pass in the wrong kind of parameter.
    # To make debugging easier add an explicit check.
    if not isinstance(headers, httplib.HTTPMessage):
        raise TypeError(f"expected httplib.Message, got {type(headers)}.")

    unparsed_data = None

    # get_payload is actually email.message.Message.get_payload;
    # we're only interested in the result if it's not a multipart message
    if not headers.is_multipart():
        payload = headers.get_payload()

        if isinstance(payload, (bytes, str)):
            unparsed_data = payload

    # httplib is assuming a response body is available
    # when parsing headers even when httplib only sends
    # header data to parse_headers() This results in
    # defects on multipart responses in particular.
    # See: https://github.com/urllib3/urllib3/issues/800

    # So we ignore the following defects:
    # - StartBoundaryNotFoundDefect:
    #     The claimed start boundary was never found.
    # - MultipartInvariantViolationDefect:
    #     A message claimed to be a multipart but no subparts were found.
    defects = [
        defect
        for defect in headers.defects
        if not isinstance(
            defect, (StartBoundaryNotFoundDefect, MultipartInvariantViolationDefect)
        )
    ]

    if defects or unparsed_data:
        raise HeaderParsingError(defects=defects, unparsed_data=unparsed_data)


def is_response_to_head(response: httplib.HTTPResponse) -> bool:
    """
    Checks whether the request of a response has been a HEAD-request.

    :param http.client.HTTPResponse response:
        Response to check if the originating request
        used 'HEAD' as a method.
    """
    # FIXME: Can we do this somehow without accessing private httplib _method?
    method_str = response._method  # type: str  # type: ignore[attr-defined]
    return method_str.upper() == "HEAD"
<|endoftext|>from __future__ import annotations

import io
import socket
import ssl
import typing

from ..exceptions import ProxySchemeUnsupported

if typing.TYPE_CHECKING:
    from typing_extensions import Self

    from .ssl_ import _TYPE_PEER_CERT_RET, _TYPE_PEER_CERT_RET_DICT


_WriteBuffer = typing.Union[bytearray, memoryview]
_ReturnValue = typing.TypeVar("_ReturnValue")

SSL_BLOCKSIZE = 16384


class SSLTransport:
    """
    The SSLTransport wraps an existing socket and establishes an SSL connection.

    Contrary to Python's implementation of SSLSocket, it allows you to chain
    multiple TLS connections together. It's particularly useful if you need to
    implement TLS within TLS.

    The class supports most of the socket API operations.
    """

    @staticmethod
    def _validate_ssl_context_for_tls_in_tls(ssl_context: ssl.SSLContext) -> None:
        """
        Raises a ProxySchemeUnsupported if the provided ssl_context can't be used
        for TLS in TLS.

        The only requirement is that the ssl_context provides the 'wrap_bio'
        methods.
        """

        if not hasattr(ssl_context, "wrap_bio"):
            raise ProxySchemeUnsupported(
                "TLS in TLS requires SSLContext.wrap_bio() which isn't "
                "available on non-native SSLContext"
            )

    def __init__(
        self,
        socket: socket.socket,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        suppress_ragged_eofs: bool = True,
    ) -> None:
        """
        Create an SSLTransport around socket using the provided ssl_context.
        """
        self.incoming = ssl.MemoryBIO()
        self.outgoing = ssl.MemoryBIO()

        self.suppress_ragged_eofs = suppress_ragged_eofs
        self.socket = socket

        self.sslobj = ssl_context.wrap_bio(
            self.incoming, self.outgoing, server_hostname=server_hostname
        )

        # Perform initial handshake.
        self._ssl_io_loop(self.sslobj.do_handshake)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: typing.Any) -> None:
        self.close()

    def fileno(self) -> int:
        return self.socket.fileno()

    def read(self, len: int = 1024, buffer: typing.Any | None = None) -> int | bytes:
        return self._wrap_ssl_read(len, buffer)

    def recv(self, buflen: int = 1024, flags: int = 0) -> int | bytes:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to recv")
        return self._wrap_ssl_read(buflen)

    def recv_into(
        self,
        buffer: _WriteBuffer,
        nbytes: int | None = None,
        flags: int = 0,
    ) -> None | int | bytes:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to recv_into")
        if nbytes is None:
            nbytes = len(buffer)
        return self.read(nbytes, buffer)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to sendall")
        count = 0
        with memoryview(data) as view, view.cast("B") as byte_view:
            amount = len(byte_view)
            while count < amount:
                v = self.send(byte_view[count:])
                count += v

    def send(self, data: bytes, flags: int = 0) -> int:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to send")
        return self._ssl_io_loop(self.sslobj.write, data)

    def makefile(
        self,
        mode: str,
        buffering: int | None = None,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> typing.BinaryIO | typing.TextIO | socket.SocketIO:
        """
        Python's httpclient uses makefile and buffered io when reading HTTP
        messages and we need to support it.

        This is unfortunately a copy and paste of socket.py makefile with small
        changes to point to the socket directly.
        """
        if not set(mode) <= {"r", "w", "b"}:
            raise ValueError(f"invalid mode {mode!r} (only r, w, b allowed)")

        writing = "w" in mode
        reading = "r" in mode or not writing
        assert reading or writing
        binary = "b" in mode
        rawmode = ""
        if reading:
            rawmode += "r"
        if writing:
            rawmode += "w"
        raw = socket.SocketIO(self, rawmode)  # type: ignore[arg-type]
        self.socket._io_refs += 1  # type: ignore[attr-defined]
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError("unbuffered streams must be binary")
            return raw
        buffer: typing.BinaryIO
        if reading and writing:
            buffer = io.BufferedRWPair(raw, raw, buffering)  # type: ignore[assignment]
        elif reading:
            buffer = io.BufferedReader(raw, buffering)
        else:
            assert writing
            buffer = io.BufferedWriter(raw, buffering)
        if binary:
            return buffer
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode  # type: ignore[misc]
        return text

    def unwrap(self) -> None:
        self._ssl_io_loop(self.sslobj.unwrap)

    def close(self) -> None:
        self.socket.close()

    @typing.overload
    def getpeercert(
        self, binary_form: typing.Literal[False] = ...
    ) -> _TYPE_PEER_CERT_RET_DICT | None: ...

    @typing.overload
    def getpeercert(self, binary_form: typing.Literal[True]) -> bytes | None: ...

    def getpeercert(self, binary_form: bool = False) -> _TYPE_PEER_CERT_RET:
        return self.sslobj.getpeercert(binary_form)  # type: ignore[return-value]

    def version(self) -> str | None:
        return self.sslobj.version()

    def cipher(self) -> tuple[str, str, int] | None:
        return self.sslobj.cipher()

    def selected_alpn_protocol(self) -> str | None:
        return self.sslobj.selected_alpn_protocol()

    def shared_ciphers(self) -> list[tuple[str, str, int]] | None:
        return self.sslobj.shared_ciphers()

    def compression(self) -> str | None:
        return self.sslobj.compression()

    def settimeout(self, value: float | None) -> None:
        self.socket.settimeout(value)

    def gettimeout(self) -> float | None:
        return self.socket.gettimeout()

    def _decref_socketios(self) -> None:
        self.socket._decref_socketios()  # type: ignore[attr-defined]

    def _wrap_ssl_read(self, len: int, buffer: bytearray | None = None) -> int | bytes:
        try:
            return self._ssl_io_loop(self.sslobj.read, len, buffer)
        except ssl.SSLError as e:
            if e.errno == ssl.SSL_ERROR_EOF and self.suppress_ragged_eofs:
                return 0  # eof, return 0.
            else:
                raise

    # func is sslobj.do_handshake or sslobj.unwrap
    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[], None]) -> None: ...

    # func is sslobj.write, arg1 is data
    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[bytes], int], arg1: bytes) -> int: ...

    # func is sslobj.read, arg1 is len, arg2 is buffer
    @typing.overload
    def _ssl_io_loop(
        self,
        func: typing.Callable[[int, bytearray | None], bytes],
        arg1: int,
        arg2: bytearray | None,
    ) -> bytes: ...

    def _ssl_io_loop(
        self,
        func: typing.Callable[..., _ReturnValue],
        arg1: None | bytes | int = None,
        arg2: bytearray | None = None,
    ) -> _ReturnValue:
        """Performs an I/O loop between incoming/outgoing and the socket."""
        should_loop = True
        ret = None

        while should_loop:
            errno = None
            try:
                if arg1 is None and arg2 is None:
                    ret = func()
                elif arg2 is None:
                    ret = func(arg1)
                else:
                    ret = func(arg1, arg2)
            except ssl.SSLError as e:
                if e.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE):
                    # WANT_READ, and WANT_WRITE are expected, others are not.
                    raise e
                errno = e.errno

            buf = self.outgoing.read()
            self.socket.sendall(buf)

            if errno is None:
                should_loop = False
            elif errno == ssl.SSL_ERROR_WANT_READ:
                buf = self.socket.recv(SSL_BLOCKSIZE)
                if buf:
                    self.incoming.write(buf)
                else:
                    self.incoming.write_eof()
        return typing.cast(_ReturnValue, ret)
<|endoftext|>from __future__ import annotations

import typing
from types import TracebackType


def to_bytes(
    x: str | bytes, encoding: str | None = None, errors: str | None = None
) -> bytes:
    if isinstance(x, bytes):
        return x
    elif not isinstance(x, str):
        raise TypeError(f"not expecting type {type(x).__name__}")
    if encoding or errors:
        return x.encode(encoding or "utf-8", errors=errors or "strict")
    return x.encode()


def to_str(
    x: str | bytes, encoding: str | None = None, errors: str | None = None
) -> str:
    if isinstance(x, str):
        return x
    elif not isinstance(x, bytes):
        raise TypeError(f"not expecting type {type(x).__name__}")
    if encoding or errors:
        return x.decode(encoding or "utf-8", errors=errors or "strict")
    return x.decode()


def reraise(
    tp: type[BaseException] | None,
    value: BaseException,
    tb: TracebackType | None = None,
) -> typing.NoReturn:
    try:
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None  # type: ignore[assignment]
        tb = None
<|endoftext|># For backwards compatibility, provide imports that used to be here.
from __future__ import annotations

from .connection import is_connection_dropped
from .request import SKIP_HEADER, SKIPPABLE_HEADERS, make_headers
from .response import is_fp_closed
from .retry import Retry
from .ssl_ import (
    ALPN_PROTOCOLS,
    IS_PYOPENSSL,
    SSLContext,
    assert_fingerprint,
    create_urllib3_context,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .timeout import Timeout
from .url import Url, parse_url
from .wait import wait_for_read, wait_for_write

__all__ = (
    "IS_PYOPENSSL",
    "SSLContext",
    "ALPN_PROTOCOLS",
    "Retry",
    "Timeout",
    "Url",
    "assert_fingerprint",
    "create_urllib3_context",
    "is_connection_dropped",
    "is_fp_closed",
    "parse_url",
    "make_headers",
    "resolve_cert_reqs",
    "resolve_ssl_version",
    "ssl_wrap_socket",
    "wait_for_read",
    "wait_for_write",
    "SKIP_HEADER",
    "SKIPPABLE_HEADERS",
)
<|endoftext|>from __future__ import annotations

import hashlib
import hmac
import os
import socket
import sys
import typing
import warnings
from binascii import Error as BinasciiError
from binascii import unhexlify

from ..exceptions import ProxySchemeUnsupported, SSLError
from .url import _BRACELESS_IPV6_ADDRZ_RE, _IPV4_RE

SSLContext = None
SSLTransport = None
HAS_NEVER_CHECK_COMMON_NAME = False
IS_PYOPENSSL = False
ALPN_PROTOCOLS = ["http/1.1"]

_TYPE_VERSION_INFO = tuple[int, int, int, str, int]

# Maps the length of a digest to a possible hash function producing this digest
HASHFUNC_MAP = {
    length: getattr(hashlib, algorithm, None)
    for length, algorithm in ((32, "md5"), (40, "sha1"), (64, "sha256"))
}


def _is_has_never_check_common_name_reliable(
    openssl_version: str,
) -> bool:
    # As of May 2023, all released versions of LibreSSL fail to reject certificates with
    # only common names, see https://github.com/urllib3/urllib3/pull/3024
    is_openssl = openssl_version.startswith("OpenSSL ")

    return is_openssl


if typing.TYPE_CHECKING:
    from ssl import VerifyMode
    from typing import TypedDict

    from .ssltransport import SSLTransport as SSLTransportType

    class _TYPE_PEER_CERT_RET_DICT(TypedDict, total=False):
        subjectAltName: tuple[tuple[str, str], ...]
        subject: tuple[tuple[tuple[str, str], ...], ...]
        serialNumber: str


# Mapping from 'ssl.PROTOCOL_TLSX' to 'TLSVersion.X'
_SSL_VERSION_TO_TLS_VERSION: dict[int, int] = {}

try:  # Do we have ssl at all?
    import ssl
    from ssl import (  # type: ignore[assignment]
        CERT_REQUIRED,
        HAS_NEVER_CHECK_COMMON_NAME,
        OP_NO_COMPRESSION,
        OP_NO_TICKET,
        OPENSSL_VERSION,
        PROTOCOL_TLS,
        PROTOCOL_TLS_CLIENT,
        VERIFY_X509_PARTIAL_CHAIN,
        VERIFY_X509_STRICT,
        OP_NO_SSLv2,
        OP_NO_SSLv3,
        SSLContext,
        TLSVersion,
    )

    PROTOCOL_SSLv23 = PROTOCOL_TLS

    # Setting SSLContext.hostname_checks_common_name = False didn't work with
    # LibreSSL, check details in the used function.
    if HAS_NEVER_CHECK_COMMON_NAME and not _is_has_never_check_common_name_reliable(
        OPENSSL_VERSION,
    ):  # Defensive:
        HAS_NEVER_CHECK_COMMON_NAME = False

    # Need to be careful here in case old TLS versions get
    # removed in future 'ssl' module implementations.
    for attr in ("TLSv1", "TLSv1_1", "TLSv1_2"):
        try:
            _SSL_VERSION_TO_TLS_VERSION[getattr(ssl, f"PROTOCOL_{attr}")] = getattr(
                TLSVersion, attr
            )
        except AttributeError:  # Defensive:
            continue

    from .ssltransport import SSLTransport  # type: ignore[assignment]
except ImportError:
    OP_NO_COMPRESSION = 0x20000  # type: ignore[assignment,misc]
    OP_NO_TICKET = 0x4000  # type: ignore[assignment,misc]
    OP_NO_SSLv2 = 0x1000000  # type: ignore[assignment,misc]
    OP_NO_SSLv3 = 0x2000000  # type: ignore[assignment,misc]
    PROTOCOL_SSLv23 = PROTOCOL_TLS = 2  # type: ignore[assignment,misc]
    PROTOCOL_TLS_CLIENT = 16  # type: ignore[assignment,misc]
    VERIFY_X509_PARTIAL_CHAIN = 0x80000  # type: ignore[assignment,misc]
    VERIFY_X509_STRICT = 0x20  # type: ignore[assignment,misc]


_TYPE_PEER_CERT_RET = typing.Union["_TYPE_PEER_CERT_RET_DICT", bytes, None]


def assert_fingerprint(cert: bytes | None, fingerprint: str) -> None:
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """

    if cert is None:
        raise SSLError("No certificate for the peer.")

    fingerprint = fingerprint.replace(":", "").lower()
    digest_length = len(fingerprint)
    if digest_length not in HASHFUNC_MAP:
        raise SSLError(f"Fingerprint of invalid length: {fingerprint}")
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if hashfunc is None:
        raise SSLError(
            f"Hash function implementation unavailable for fingerprint length: {digest_length}"
        )

    # We need encode() here for py32; works on py2 and p33.
    try:
        fingerprint_bytes = unhexlify(fingerprint.encode())
    except BinasciiError as e:
        raise SSLError(e) from e

    cert_digest = hashfunc(cert).digest()

    if not hmac.compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError(
            f'Fingerprints did not match. Expected "{fingerprint}", got "{cert_digest.hex()}"'
        )


def resolve_cert_reqs(candidate: None | int | str) -> VerifyMode:
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_REQUIRED`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbreviation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    if candidate is None:
        return CERT_REQUIRED

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "CERT_" + candidate)
        return res  # type: ignore[no-any-return]

    return candidate  # type: ignore[return-value]


def resolve_ssl_version(candidate: None | int | str) -> int:
    """
    like resolve_cert_reqs
    """
    if candidate is None:
        return PROTOCOL_TLS

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "PROTOCOL_" + candidate)
        return typing.cast(int, res)

    return candidate


def create_urllib3_context(
    ssl_version: int | None = None,
    cert_reqs: int | None = None,
    options: int | None = None,
    ciphers: str | None = None,
    ssl_minimum_version: int | None = None,
    ssl_maximum_version: int | None = None,
    verify_flags: int | None = None,
) -> ssl.SSLContext:
    """Creates and configures an :class:`ssl.SSLContext` instance for use with urllib3.

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.

        This parameter is deprecated instead use 'ssl_minimum_version'.
    :param ssl_minimum_version:
        The minimum version of TLS to be used. Use the 'ssl.TLSVersion' enum for specifying the value.
    :param ssl_maximum_version:
        The maximum version of TLS to be used. Use the 'ssl.TLSVersion' enum for specifying the value.
        Not recommended to set to anything other than 'ssl.TLSVersion.MAXIMUM_SUPPORTED' which is the
        default value.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``, and ``ssl.OP_NO_TICKET``.
    :param ciphers:
        Which cipher suites to allow the server to select. Defaults to either system configured
        ciphers if OpenSSL 1.1.1+, otherwise uses a secure default set of ciphers.
    :param verify_flags:
        The flags for certificate verification operations. These default to
        ``ssl.VERIFY_X509_PARTIAL_CHAIN`` and ``ssl.VERIFY_X509_STRICT`` for Python 3.13+.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    if SSLContext is None:
        raise TypeError("Can't create an SSLContext object without an ssl module")

    # This means 'ssl_version' was specified as an exact value.
    if ssl_version not in (None, PROTOCOL_TLS, PROTOCOL_TLS_CLIENT):
        # Disallow setting 'ssl_version' and 'ssl_minimum|maximum_version'
        # to avoid conflicts.
        if ssl_minimum_version is not None or ssl_maximum_version is not None:
            raise ValueError(
                "Can't specify both 'ssl_version' and either "
                "'ssl_minimum_version' or 'ssl_maximum_version'"
            )

        # 'ssl_version' is deprecated and will be removed in the future.
        else:
            # Use 'ssl_minimum_version' and 'ssl_maximum_version' instead.
            ssl_minimum_version = _SSL_VERSION_TO_TLS_VERSION.get(
                ssl_version, TLSVersion.MINIMUM_SUPPORTED
            )
            ssl_maximum_version = _SSL_VERSION_TO_TLS_VERSION.get(
                ssl_version, TLSVersion.MAXIMUM_SUPPORTED
            )

            # This warning message is pushing users to use 'ssl_minimum_version'
            # instead of both min/max. Best practice is to only set the minimum version and
            # keep the maximum version to be it's default value: 'TLSVersion.MAXIMUM_SUPPORTED'
            warnings.warn(
                "'ssl_version' option is deprecated and will be "
                "removed in urllib3 v3.0. Instead use 'ssl_minimum_version'",
                category=FutureWarning,
                stacklevel=2,
            )

    context = SSLContext(PROTOCOL_TLS_CLIENT)
    if ssl_minimum_version is not None:
        context.minimum_version = ssl_minimum_version
    else:  # pyOpenSSL defaults to 'MINIMUM_SUPPORTED' so explicitly set TLSv1.2 here
        context.minimum_version = TLSVersion.TLSv1_2

    if ssl_maximum_version is not None:
        context.maximum_version = ssl_maximum_version

    # Unless we're given ciphers defer to either system ciphers in
    # the case of OpenSSL 1.1.1+ or use our own secure default ciphers.
    if ciphers:
        context.set_ciphers(ciphers)

    # Setting the default here, as we may have no ssl module on import
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs

    if options is None:
        options = 0
        # SSLv2 is easily broken and is considered harmful and dangerous
        options |= OP_NO_SSLv2
        # SSLv3 has several problems and is now dangerous
        options |= OP_NO_SSLv3
        # Disable compression to prevent CRIME attacks for OpenSSL 1.0+
        # (issue #309)
        options |= OP_NO_COMPRESSION
        # TLSv1.2 only. Unless set explicitly, do not request tickets.
        # This may save some bandwidth on wire, and although the ticket is encrypted,
        # there is a risk associated with it being on wire,
        # if the server is not rotating its ticketing keys properly.
        options |= OP_NO_TICKET

    context.options |= options

    if verify_flags is None:
        verify_flags = 0
        # In Python 3.13+ ssl.create_default_context() sets VERIFY_X509_PARTIAL_CHAIN
        # and VERIFY_X509_STRICT so we do the same
        if sys.version_info >= (3, 13):
            verify_flags |= VERIFY_X509_PARTIAL_CHAIN
            verify_flags |= VERIFY_X509_STRICT

    context.verify_flags |= verify_flags

    # Enable post-handshake authentication for TLS 1.3, see GH #1634. PHA is
    # necessary for conditional client cert authentication with TLS 1.3.
    # The attribute is None for OpenSSL <= 1.1.0 or does not exist when using
    # an SSLContext created by pyOpenSSL.
    if getattr(context, "post_handshake_auth", None) is not None:
        context.post_handshake_auth = True

    # The order of the below lines setting verify_mode and check_hostname
    # matter due to safe-guards SSLContext has to prevent an SSLContext with
    # check_hostname=True, verify_mode=NONE/OPTIONAL.
    # We always set 'check_hostname=False' for pyOpenSSL so we rely on our own
    # 'ssl.match_hostname()' implementation.
    if cert_reqs == ssl.CERT_REQUIRED and not IS_PYOPENSSL:
        context.verify_mode = cert_reqs
        context.check_hostname = True
    else:
        context.check_hostname = False
        context.verify_mode = cert_reqs

    context.hostname_checks_common_name = False

    if "SSLKEYLOGFILE" in os.environ:
        sslkeylogfile = os.path.expandvars(os.environ.get("SSLKEYLOGFILE"))
    else:
        sslkeylogfile = None
    if sslkeylogfile:
        context.keylog_filename = sslkeylogfile

    return context


@typing.overload
def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = ...,
    certfile: str | None = ...,
    cert_reqs: int | None = ...,
    ca_certs: str | None = ...,
    server_hostname: str | None = ...,
    ssl_version: int | None = ...,
    ciphers: str | None = ...,
    ssl_context: ssl.SSLContext | None = ...,
    ca_cert_dir: str | None = ...,
    key_password: str | None = ...,
    ca_cert_data: None | str | bytes = ...,
    tls_in_tls: typing.Literal[False] = ...,
) -> ssl.SSLSocket: ...


@typing.overload
def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = ...,
    certfile: str | None = ...,
    cert_reqs: int | None = ...,
    ca_certs: str | None = ...,
    server_hostname: str | None = ...,
    ssl_version: int | None = ...,
    ciphers: str | None = ...,
    ssl_context: ssl.SSLContext | None = ...,
    ca_cert_dir: str | None = ...,
    key_password: str | None = ...,
    ca_cert_data: None | str | bytes = ...,
    tls_in_tls: bool = ...,
) -> ssl.SSLSocket | SSLTransportType: ...


def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = None,
    certfile: str | None = None,
    cert_reqs: int | None = None,
    ca_certs: str | None = None,
    server_hostname: str | None = None,
    ssl_version: int | None = None,
    ciphers: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    ca_cert_dir: str | None = None,
    key_password: str | None = None,
    ca_cert_data: None | str | bytes = None,
    tls_in_tls: bool = False,
) -> ssl.SSLSocket | SSLTransportType:
    """
    All arguments except for server_hostname, ssl_context, tls_in_tls, ca_cert_data and
    ca_cert_dir have the same meaning as they do when using
    :func:`ssl.create_default_context`, :meth:`ssl.SSLContext.load_cert_chain`,
    :meth:`ssl.SSLContext.set_ciphers` and :meth:`ssl.SSLContext.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    :param key_password:
        Optional password if the keyfile is encrypted.
    :param ca_cert_data:
        Optional string containing CA certificates in PEM format suitable for
        passing as the cadata parameter to SSLContext.load_verify_locations()
    :param tls_in_tls:
        Use SSLTransport to wrap the existing socket.
    """
    context = ssl_context
    if context is None:
        # Note: This branch of code and all the variables in it are only used in tests.
        # We should consider deprecating and removing this code.
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)

    if ca_certs or ca_cert_dir or ca_cert_data:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir, ca_cert_data)
        except OSError as e:
            raise SSLError(e) from e

    elif ssl_context is None and hasattr(context, "load_default_certs"):
        # try to load OS default certs; works well on Windows.
        context.load_default_certs()

    # Attempt to detect if we get the goofy behavior of the
    # keyfile being encrypted and OpenSSL asking for the
    # passphrase via the terminal and instead error out.
    if keyfile and key_password is None and _is_key_file_encrypted(keyfile):
        raise SSLError("Client private key is encrypted, password is required")

    if certfile:
        if key_password is None:
            context.load_cert_chain(certfile, keyfile)
        else:
            context.load_cert_chain(certfile, keyfile, key_password)

    context.set_alpn_protocols(ALPN_PROTOCOLS)

    ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)
    return ssl_sock


def is_ipaddress(hostname: str | bytes) -> bool:
    """Detects whether the hostname given is an IPv4 or IPv6 address.
    Also detects IPv6 addresses with Zone IDs.

    :param str hostname: Hostname to examine.
    :return: True if the hostname is an IP address, False otherwise.
    """
    if isinstance(hostname, bytes):
        # IDN A-label bytes are ASCII compatible.
        hostname = hostname.decode("ascii")
    return bool(_IPV4_RE.match(hostname) or _BRACELESS_IPV6_ADDRZ_RE.match(hostname))


def _is_key_file_encrypted(key_file: str) -> bool:
    """Detects if a key file is encrypted or not."""
    with open(key_file) as f:
        for line in f:
            # Look for Proc-Type: 4,ENCRYPTED
            if "ENCRYPTED" in line:
                return True

    return False


def _ssl_wrap_socket_impl(
    sock: socket.socket,
    ssl_context: ssl.SSLContext,
    tls_in_tls: bool,
    server_hostname: str | None = None,
) -> ssl.SSLSocket | SSLTransportType:
    if tls_in_tls:
        if not SSLTransport:
            # Import error, ssl is not available.
            raise ProxySchemeUnsupported(
                "TLS in TLS requires support for the 'ssl' module"
            )

        SSLTransport._validate_ssl_context_for_tls_in_tls(ssl_context)
        return SSLTransport(sock, ssl_context, server_hostname)

    return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
<|endoftext|>from __future__ import annotations

import time
import typing
from enum import Enum
from socket import getdefaulttimeout

from ..exceptions import TimeoutStateError

if typing.TYPE_CHECKING:
    from typing import Final


class _TYPE_DEFAULT(Enum):
    # This value should never be passed to socket.settimeout() so for safety we use a -1.
    # socket.settimout() raises a ValueError for negative values.
    token = -1


_DEFAULT_TIMEOUT: Final[_TYPE_DEFAULT] = _TYPE_DEFAULT.token

_TYPE_TIMEOUT = typing.Optional[typing.Union[float, _TYPE_DEFAULT]]


class Timeout:
    """Timeout configuration.

    Timeouts can be defined as a default for a pool:

    .. code-block:: python

        import urllib3

        timeout = urllib3.util.Timeout(connect=2.0, read=7.0)

        http = urllib3.PoolManager(timeout=timeout)

        resp = http.request("GET", "https://example.com/")

        print(resp.status)

    Or per-request (which overrides the default for the pool):

    .. code-block:: python

       response = http.request("GET", "https://example.com/", timeout=Timeout(10))

    Timeouts can be disabled by setting all the parameters to ``None``:

    .. code-block:: python

       no_timeout = Timeout(connect=None, read=None)
       response = http.request("GET", "https://example.com/", timeout=no_timeout)


    :param total:
        This combines the connect and read timeouts into one; the read timeout
        will be set to the time leftover from the connect attempt. In the
        event that both a connect timeout and a total are specified, or a read
        timeout and a total are specified, the shorter timeout will be applied.

        Defaults to None.

    :type total: int, float, or None

    :param connect:
        The maximum amount of time (in seconds) to wait for a connection
        attempt to a server to succeed. Omitting the parameter will default the
        connect timeout to the system default, probably `the global default
        timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout for connection attempts.

    :type connect: int, float, or None

    :param read:
        The maximum amount of time (in seconds) to wait between consecutive
        read operations for a response from the server. Omitting the parameter
        will default the read timeout to the system default, probably `the
        global default timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout.

    :type read: int, float, or None

    .. note::

        Many factors can affect the total amount of time for urllib3 to return
        an HTTP response.

        For example, Python's DNS resolver does not obey the timeout specified
        on the socket. Other factors that can affect total request time include
        high CPU load, high swap, the program running at a low priority level,
        or other behaviors.

        In addition, the read and total timeouts only measure the time between
        read operations on the socket connecting the client and the server,
        not the total amount of time for the request to return a complete
        response. For most requests, the timeout is raised because the server
        has not sent the first byte in the specified time. This is not always
        the case; if a server streams one byte every fifteen seconds, a timeout
        of 20 seconds will not trigger, even though the request will take
        several minutes to complete.
    """

    #: A sentinel object representing the default timeout value
    DEFAULT_TIMEOUT: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT

    def __init__(
        self,
        total: _TYPE_TIMEOUT = None,
        connect: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        read: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
    ) -> None:
        self._connect = self._validate_timeout(connect, "connect")
        self._read = self._validate_timeout(read, "read")
        self.total = self._validate_timeout(total, "total")
        self._start_connect: float | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(connect={self._connect!r}, read={self._read!r}, total={self.total!r})"

    # __str__ provided for backwards compatibility
    __str__ = __repr__

    @staticmethod
    def resolve_default_timeout(timeout: _TYPE_TIMEOUT) -> float | None:
        return getdefaulttimeout() if timeout is _DEFAULT_TIMEOUT else timeout

    @classmethod
    def _validate_timeout(cls, value: _TYPE_TIMEOUT, name: str) -> _TYPE_TIMEOUT:
        """Check that a timeout attribute is valid.

        :param value: The timeout value to validate
        :param name: The name of the timeout attribute to validate. This is
            used to specify in error messages.
        :return: The validated and casted version of the given value.
        :raises ValueError: If it is a numeric value less than or equal to
            zero, or the type is not an integer, float, or None.
        """
        if value is None or value is _DEFAULT_TIMEOUT:
            return value

        if isinstance(value, bool):
            raise ValueError(
                "Timeout cannot be a boolean value. It must "
                "be an int, float or None."
            )
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            ) from None

        try:
            if value <= 0:
                raise ValueError(
                    "Attempted to set %s timeout to %s, but the "
                    "timeout cannot be set to a value less "
                    "than or equal to 0." % (name, value)
                )
        except TypeError:
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            ) from None

        return value

    @classmethod
    def from_float(cls, timeout: _TYPE_TIMEOUT) -> Timeout:
        """Create a new Timeout from a legacy timeout value.

        The timeout value used by httplib.py sets the same timeout on the
        connect(), and recv() socket requests. This creates a :class:`Timeout`
        object that sets the individual timeouts to the ``timeout`` value
        passed to this function.

        :param timeout: The legacy timeout value.
        :type timeout: integer, float, :attr:`urllib3.util.Timeout.DEFAULT_TIMEOUT`, or None
        :return: Timeout object
        :rtype: :class:`Timeout`
        """
        return Timeout(read=timeout, connect=timeout)

    def clone(self) -> Timeout:
        """Create a copy of the timeout object

        Timeout properties are stored per-pool but each request needs a fresh
        Timeout object to ensure each one has its own start/stop configured.

        :return: a copy of the timeout object
        :rtype: :class:`Timeout`
        """
        # We can't use copy.deepcopy because that will also create a new object
        # for _GLOBAL_DEFAULT_TIMEOUT, which socket.py uses as a sentinel to
        # detect the user default.
        return Timeout(connect=self._connect, read=self._read, total=self.total)

    def start_connect(self) -> float:
        """Start the timeout clock, used during a connect() attempt

        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to start a timer that has been started already.
        """
        if self._start_connect is not None:
            raise TimeoutStateError("Timeout timer has already been started.")
        self._start_connect = time.monotonic()
        return self._start_connect

    def get_connect_duration(self) -> float:
        """Gets the time elapsed since the call to :meth:`start_connect`.

        :return: Elapsed time in seconds.
        :rtype: float
        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to get duration for a timer that hasn't been started.
        """
        if self._start_connect is None:
            raise TimeoutStateError(
                "Can't get connect duration for timer that has not started."
            )
        return time.monotonic() - self._start_connect

    @property
    def connect_timeout(self) -> _TYPE_TIMEOUT:
        """Get the value to use when setting a connection timeout.

        This will be a positive float or integer, the value None
        (never timeout), or the default system timeout.

        :return: Connect timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        """
        if self.total is None:
            return self._connect

        if self._connect is None or self._connect is _DEFAULT_TIMEOUT:
            return self.total

        return min(self._connect, self.total)  # type: ignore[type-var]

    @property
    def read_timeout(self) -> float | None:
        """Get the value for the read timeout.

        This assumes some time has elapsed in the connection timeout and
        computes the read timeout appropriately.

        If self.total is set, the read timeout is dependent on the amount of
        time taken by the connect timeout. If the connection time has not been
        established, a :exc:`~urllib3.exceptions.TimeoutStateError` will be
        raised.

        :return: Value to use for the read timeout.
        :rtype: int, float or None
        :raises urllib3.exceptions.TimeoutStateError: If :meth:`start_connect`
            has not yet been called on this object.
        """
        if (
            self.total is not None
            and self.total is not _DEFAULT_TIMEOUT
            and self._read is not None
            and self._read is not _DEFAULT_TIMEOUT
        ):
            # In case the connect timeout has not yet been established.
            if self._start_connect is None:
                return self._read
            return max(0, min(self.total - self.get_connect_duration(), self._read))
        elif self.total is not None and self.total is not _DEFAULT_TIMEOUT:
            return max(0, self.total - self.get_connect_duration())
        else:
            return self.resolve_default_timeout(self._read)
<|endoftext|>"""The match_hostname() function from Python 3.5, essential when using SSL."""

# Note: This file is under the PSF license as the code comes from the python
# stdlib.   http://docs.python.org/3/license.html
# It is modified to remove commonName support.

from __future__ import annotations

import ipaddress
import re
import typing
from ipaddress import IPv4Address, IPv6Address

if typing.TYPE_CHECKING:
    from .ssl_ import _TYPE_PEER_CERT_RET_DICT

__version__ = "3.5.0.1"


class CertificateError(ValueError):
    pass


def _dnsname_match(
    dn: typing.Any, hostname: str, max_wildcards: int = 1
) -> typing.Match[str] | None | bool:
    """Matching according to RFC 6125, section 6.4.3

    http://tools.ietf.org/html/rfc6125#section-6.4.3
    """
    pats = []
    if not dn:
        return False

    # Ported from python3-syntax:
    # leftmost, *remainder = dn.split(r'.')
    parts = dn.split(r".")
    leftmost = parts[0]
    remainder = parts[1:]

    wildcards = leftmost.count("*")
    if wildcards > max_wildcards:
        # Issue #17980: avoid denials of service by refusing more
        # than one wildcard per fragment.  A survey of established
        # policy among SSL implementations showed it to be a
        # reasonable choice.
        raise CertificateError(
            "too many wildcards in certificate DNS name: " + repr(dn)
        )

    # speed up common case w/o wildcards
    if not wildcards:
        return bool(dn.lower() == hostname.lower())

    # RFC 6125, section 6.4.3, subitem 1.
    # The client SHOULD NOT attempt to match a presented identifier in which
    # the wildcard character comprises a label other than the left-most label.
    if leftmost == "*":
        # When '*' is a fragment by itself, it matches a non-empty dotless
        # fragment.
        pats.append("[^.]+")
    elif leftmost.startswith("xn--") or hostname.startswith("xn--"):
        # RFC 6125, section 6.4.3, subitem 3.
        # The client SHOULD NOT attempt to match a presented identifier
        # where the wildcard character is embedded within an A-label or
        # U-label of an internationalized domain name.
        pats.append(re.escape(leftmost))
    else:
        # Otherwise, '*' matches any dotless string, e.g. www*
        pats.append(re.escape(leftmost).replace(r"\*", "[^.]*"))

    # add the remaining fragments, ignore any wildcards
    for frag in remainder:
        pats.append(re.escape(frag))

    pat = re.compile(r"\A" + r"\.".join(pats) + r"\Z", re.IGNORECASE)
    return pat.match(hostname)


def _ipaddress_match(ipname: str, host_ip: IPv4Address | IPv6Address) -> bool:
    """Exact matching of IP addresses.

    RFC 9110 section 4.3.5: "A reference identity of IP-ID contains the decoded
    bytes of the IP address. An IP version 4 address is 4 octets, and an IP
    version 6 address is 16 octets. [...] A reference identity of type IP-ID
    matches if the address is identical to an iPAddress value of the
    subjectAltName extension of the certificate."
    """
    # OpenSSL may add a trailing newline to a subjectAltName's IP address
    # Divergence from upstream: ipaddress can't handle byte str
    ip = ipaddress.ip_address(ipname.rstrip())
    return bool(ip.packed == host_ip.packed)


def match_hostname(
    cert: _TYPE_PEER_CERT_RET_DICT | None,
    hostname: str,
    hostname_checks_common_name: bool = False,
) -> None:
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed, but IP addresses are not accepted for *hostname*.

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    if not cert:
        raise ValueError(
            "empty or no certificate, match_hostname needs a "
            "SSL socket or SSL context with either "
            "CERT_OPTIONAL or CERT_REQUIRED"
        )

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP address (common case)
        host_ip = None
    dnsnames = []
    san: tuple[tuple[str, str], ...] = cert.get("subjectAltName", ())
    key: str
    value: str
    for key, value in san:
        if key == "DNS":
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == "IP Address":
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)

    # We only check 'commonName' if it's enabled and we're not verifying
    # an IP address. IP addresses aren't valid within 'commonName'.
    if hostname_checks_common_name and host_ip is None and not dnsnames:
        for sub in cert.get("subject", ()):
            for key, value in sub:
                if key == "commonName":
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(
                        value
                    )  # Defensive: for older PyPy and OpenSSL versions

    if len(dnsnames) > 1:
        raise CertificateError(
            "hostname %r "
            "doesn't match either of %s" % (hostname, ", ".join(map(repr, dnsnames)))
        )
    elif len(dnsnames) == 1:
        raise CertificateError(f"hostname {hostname!r} doesn't match {dnsnames[0]!r}")
    else:
        raise CertificateError("no appropriate subjectAltName fields were found")
<|endoftext|>from __future__ import annotations

import socket
import typing

from ..exceptions import LocationParseError
from .timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT

_TYPE_SOCKET_OPTIONS = list[tuple[int, int, typing.Union[int, bytes]]]

if typing.TYPE_CHECKING:
    from .._base_connection import BaseHTTPConnection


def is_connection_dropped(conn: BaseHTTPConnection) -> bool:  # Platform-specific
    """
    Returns True if the connection is dropped and should be closed.
    :param conn: :class:`urllib3.connection.HTTPConnection` object.
    """
    return not conn.is_connected


# This function is copied from socket.py in the Python 2.7 standard
# library test suite. Added to its signature is only `socket_options`.
# One additional modification is that we avoid binding to IPv6 servers
# discovered in DNS if the system doesn't have IPv6 functionality.
def create_connection(
    address: tuple[str, int],
    timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
    socket_options: _TYPE_SOCKET_OPTIONS | None = None,
) -> socket.socket:
    """Connect to *address* and return the socket object.

    Convenience function.  Connect to *address* (a 2-tuple ``(host,
    port)``) and return the socket object.  Passing the optional
    *timeout* parameter will set the timeout on the socket instance
    before attempting to connect.  If no *timeout* is supplied, the
