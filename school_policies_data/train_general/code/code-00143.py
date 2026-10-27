    def params(self, params: QueryParamTypes) -> None:
        self._params = QueryParams(params)

    def build_request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Request:
        """
        Build and return a request instance.

        * The `params`, `headers` and `cookies` arguments
        are merged with any values set on the client.
        * The `url` argument is merged with any `base_url` set on the client.

        See also: [Request instances][0]

        [0]: advanced/clients.md#request-instances
        """
        url = self._merge_url(url)
        headers = self._merge_headers(headers)
        cookies = self._merge_cookies(cookies)
        params = self._merge_queryparams(params)
        extensions = {} if extensions is None else extensions
        if "timeout" not in extensions:
            timeout = self.timeout if isinstance(timeout, UseClientDefault) else Timeout(timeout)
            extensions = dict(**extensions, timeout=timeout.as_dict())
        return Request(
            method,
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            extensions=extensions,
        )

    def _merge_url(self, url: URL | str) -> URL:
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        merge_url = URL(url)
        if merge_url.is_relative_url:
            # To merge URLs we always append to the base URL. To get this
            # behaviour correct we always ensure the base URL ends in a '/'
            # separator, and strip any leading '/' from the merge URL.
            #
            # So, eg...
            #
            # >>> client = Client(base_url="https://www.example.com/subpath")
            # >>> client.base_url
            # URL('https://www.example.com/subpath/')
            # >>> client.build_request("GET", "/path").url
            # URL('https://www.example.com/subpath/path')
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b"/")
            return self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url

    def _merge_cookies(self, cookies: CookieTypes | None = None) -> CookieTypes | None:
        """
        Merge a cookies argument together with any cookies on the client,
        to create the cookies used for the outgoing request.
        """
        if cookies or self.cookies:
            merged_cookies = Cookies(self.cookies)
            merged_cookies.update(cookies)
            return merged_cookies
        return cookies

    def _merge_headers(self, headers: HeaderTypes | None = None) -> HeaderTypes | None:
        """
        Merge a headers argument together with any headers on the client,
        to create the headers used for the outgoing request.
        """
        merged_headers = Headers(self.headers)
        merged_headers.update(headers)
        return merged_headers

    def _merge_queryparams(self, params: QueryParamTypes | None = None) -> QueryParamTypes | None:
        """
        Merge a queryparams argument together with any queryparams on the client,
        to create the queryparams used for the outgoing request.
        """
        if params or self.params:
            merged_queryparams = QueryParams(self.params)
            return merged_queryparams.merge(params)
        return params

    def _build_auth(self, auth: AuthTypes | None) -> Auth | None:
        if auth is None:
            return None
        elif isinstance(auth, tuple):
            return BasicAuth(username=auth[0], password=auth[1])
        elif isinstance(auth, Auth):
            return auth
        elif callable(auth):
            return FunctionAuth(func=auth)
        else:
            raise TypeError(f'Invalid "auth" argument: {auth!r}')

    def _build_request_auth(
        self,
        request: Request,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
    ) -> Auth:
        auth = self._auth if isinstance(auth, UseClientDefault) else self._build_auth(auth)

        if auth is not None:
            return auth

        username, password = request.url.username, request.url.password
        if username or password:
            return BasicAuth(username=username, password=password)

        return Auth()

    def _build_redirect_request(self, request: Request, response: Response) -> Request:
        """
        Given a request and a redirect response, return a new request that
        should be used to effect the redirect.
        """
        method = self._redirect_method(request, response)
        url = self._redirect_url(request, response)
        headers = self._redirect_headers(request, url, method)
        stream = self._redirect_stream(request, method)
        cookies = Cookies(self.cookies)
        return Request(
            method=method,
            url=url,
            headers=headers,
            cookies=cookies,
            stream=stream,
            extensions=request.extensions,
        )

    def _redirect_method(self, request: Request, response: Response) -> str:
        """
        When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = request.method

        # https://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.SEE_OTHER and method != "HEAD":
            method = "GET"

        # Do what the browsers do, despite standards...
        # Turn 302s into GETs.
        # QUERY is excluded, per RFC 10008 Section 2.5.
        # https://datatracker.ietf.org/doc/html/rfc10008#section-2.5
        if response.status_code == codes.FOUND and method not in ("HEAD", "QUERY"):
            method = "GET"

        # If a POST is responded to with a 301, turn it into a GET.
        # This bizarre behaviour is explained in 'requests' issue 1704.
        if response.status_code == codes.MOVED_PERMANENTLY and method == "POST":
            method = "GET"

        return method

    def _redirect_url(self, request: Request, response: Response) -> URL:
        """
        Return the URL for the redirect to follow.
        """
        location = response.headers["Location"]

        try:
            url = URL(location)
        except InvalidURL as exc:
            raise RemoteProtocolError(f"Invalid URL in location header: {exc}.", request=request) from None

        # Handle malformed 'Location' headers that are "absolute" form, have no host.
        # See: https://github.com/encode/httpx/issues/771
        if url.scheme and not url.host:
            url = url.copy_with(host=request.url.host)

        # Facilitate relative 'Location' headers, as allowed by RFC 7231.
        # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
        if url.is_relative_url:
            url = request.url.join(url)

        # Attach previous fragment if needed (RFC 7231 7.1.2)
        if request.url.fragment and not url.fragment:
            url = url.copy_with(fragment=request.url.fragment)

        return url

    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers:
        """
        Return the headers that should be used for the redirect request.
        """
        headers = Headers(request.headers)

        if url.origin != request.url.origin:
            if not _is_https_redirect(request.url, url):
                # Strip Authorization headers when responses are redirected
                # away from the origin. (Except for direct HTTP to HTTPS redirects.)
                headers.pop("Authorization", None)

            # Update the Host header.
            headers["Host"] = url.netloc.decode("ascii")

        if method != request.method and method == "GET":
            # If we've switch to a 'GET' request, then strip any headers which
            # are only relevant to the request body.
            headers.pop("Content-Length", None)
            headers.pop("Transfer-Encoding", None)

        # We should use the client cookie store to determine any cookie header,
        # rather than whatever was on the original outgoing request.
        headers.pop("Cookie", None)

        return headers

    def _redirect_stream(self, request: Request, method: str) -> SyncByteStream | AsyncByteStream | None:
        """
        Return the body that should be used for the redirect request.
        """
        if method != request.method and method == "GET":
            return None

        return request.stream

    def _set_timeout(self, request: Request) -> None:
        if "timeout" not in request.extensions:
            timeout = self.timeout if isinstance(self.timeout, UseClientDefault) else Timeout(self.timeout)
            request.extensions = dict(**request.extensions, timeout=timeout.as_dict())


class Client(BaseClient):
    """
    An HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.

    It can be shared between threads.

    Usage:

    ```python
    >>> client = httpx2.Client()
    >>> response = client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **mounts** - *(optional)* A dictionary mapping URL patterns to transports,
    used to route requests through specific transports based on the URL.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: None | (typing.Mapping[str, BaseTransport | None]) = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: None | (typing.Mapping[str, list[EventHook]]) = None,
        base_url: URL | str = "",
        transport: BaseTransport | None = None,
        default_encoding: str | typing.Callable[[bytes], str | None] = "utf-8",
    ) -> None:
        super().__init__(
            auth=auth,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            event_hooks=event_hooks,
            base_url=base_url,
            trust_env=trust_env,
            default_encoding=default_encoding,
        )

        if http2:
            try:
                import h2  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using http2=True, but the 'h2' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[http2]`."
                ) from None

        allow_env_proxies = trust_env and transport is None
        proxy_map = self._get_proxy_map(proxy, allow_env_proxies)

        self._transport = self._init_transport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            transport=transport,
        )
        self._mounts: dict[URLPattern, BaseTransport | None] = {
            URLPattern(key): None
            if proxy is None
            else self._init_proxy_transport(
                proxy,
                verify=verify,
                cert=cert,
                trust_env=trust_env,
                http1=http1,
                http2=http2,
                limits=limits,
            )
            for key, proxy in proxy_map.items()
        }
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})

        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: BaseTransport | None = None,
    ) -> BaseTransport:
        if transport is not None:
            return transport

        return HTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
        )

    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> BaseTransport:
        return HTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            proxy=proxy,
        )

    def _transport_for_url(self, url: URL) -> BaseTransport:
        """
        Returns the transport instance that should be used for a given URL.
        This will either be the standard connection pool, or a proxy.
        """
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport

        return self._transport

    def request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Build and send a request.

        Equivalent to:

        ```python
        request = client.build_request(...)
        response = client.send(request, ...)
        ```

        See `Client.build_request()`, `Client.send()` and
        [Merging of configuration][0] for how the various parameters
        are merged with client-level configuration.

        [0]: advanced/clients.md#merging-of-configuration
        """
        if cookies is not None:
            message = (
                "Setting per-request cookies=<...> is being deprecated, because "
                "the expected behaviour on cookie persistence is ambiguous. Set "
                "cookies directly on the client instance instead."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)

        request = self.build_request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            extensions=extensions,
        )
        return self.send(request, auth=auth, follow_redirects=follow_redirects)

    @contextmanager
    def stream(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Generator[Response]:
        """
        Alternative to `httpx2.request()` that streams the response body
        instead of loading it into memory at once.

        **Parameters**: See `httpx2.request`.

        See also: [Streaming Responses][0]

        [0]: quickstart.md#streaming-responses
        """
        request = self.build_request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            extensions=extensions,
        )
        response = self.send(
            request=request,
            auth=auth,
            follow_redirects=follow_redirects,
            stream=True,
        )
        try:
            yield response
        finally:
            response.close()

    @contextmanager
    def sse(
        self,
        url: URL | str,
        *,
        method: str = "GET",
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
        max_event_size: int | None = DEFAULT_MAX_EVENT_SIZE_BYTES,
    ) -> Generator[EventSource]:
        """
        Connect to a server-sent events endpoint and yield an `EventSource`.

        Iterating the `EventSource` yields `ServerSentEvent` instances.

        **Parameters**: See `httpx2.request`.
        """
        with self.stream(
            method,
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers={"Accept": "text/event-stream", "Cache-Control": "no-store"} | Headers(headers),
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as response:
            yield EventSource(response, max_event_size=max_event_size)

    @contextmanager
    def websocket(
        self,
        url: URL | str,
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
    ) -> Generator[WebSocketSession]:
        """
        Open a WebSocket session, using this client's configuration.

        The session is closed automatically when exiting the context manager.

        ```python
        with httpx2.Client() as client:
            with client.websocket("ws://localhost:8000/ws") as ws:
                ws.send_text("Hello!")
                message = ws.receive_text()
        ```
        """
        try:
            from .websockets._api import connect_ws
        except ImportError:  # pragma: no cover
            raise ImportError(
                "WebSocket support requires the `wsproto` package. Install it with `pip install httpx2[ws]`."
            )

        with connect_ws(
            str(url),
            self,
            max_message_size_bytes=max_message_size_bytes,
            queue_size=queue_size,
            keepalive_ping_interval_seconds=keepalive_ping_interval_seconds,
            keepalive_ping_timeout_seconds=keepalive_ping_timeout_seconds,
            subprotocols=subprotocols,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as session:
            yield session

    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Response:
        """
        Send a request.

        The request is sent as-is, unmodified.

        Typically you'll want to build one with `Client.build_request()`
        so that any client-level configuration is merged into the request,
        but passing an explicit `httpx2.Request()` is supported as well.

        See also: [Request instances][0]

        [0]: advanced/clients.md#request-instances
        """
        if self._state == ClientState.CLOSED:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects

        self._set_timeout(request)

        auth = self._build_request_auth(request, auth)

        response = self._send_handling_auth(
            request,
            auth=auth,
            follow_redirects=follow_redirects,
            history=[],
        )
        try:
            if not stream:
                response.read()

            return response

        except BaseException as exc:
            response.close()
            raise exc

    def _send_handling_auth(
        self,
        request: Request,
        auth: Auth,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response:
        auth_flow = auth.sync_auth_flow(request)
        try:
            request = next(auth_flow)

            while True:
                response = self._send_handling_redirects(
                    request,
                    follow_redirects=follow_redirects,
                    history=history,
                )
                try:
                    try:
                        next_request = auth_flow.send(response)
                    except StopIteration:
                        return response

                    response.history = list(history)
                    response.read()
                    request = next_request
                    history.append(response)

                except BaseException as exc:
                    response.close()
                    raise exc
        finally:
            auth_flow.close()

    def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response:
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects("Exceeded maximum allowed redirects.", request=request)

            for hook in self._event_hooks["request"]:
                hook(request)

            response = self._send_single_request(request)
            try:
                for hook in self._event_hooks["response"]:
                    hook(response)
                response.history = list(history)

                if not response.has_redirect_location:
                    return response

                request = self._build_redirect_request(request, response)
                history = history + [response]

                if follow_redirects:
                    response.read()
                else:
                    response.next_request = request
                    return response

            except BaseException as exc:
                response.close()
                raise exc

    def _send_single_request(self, request: Request) -> Response:
        """
        Sends a single request, without handling any redirections.
        """
        transport = self._transport_for_url(request.url)
        start = time.perf_counter()

        if not isinstance(request.stream, SyncByteStream):
            raise RuntimeError("Attempted to send an async request with a sync Client instance.")

        with request_context(request=request):
            response = transport.handle_request(request)

        assert isinstance(response.stream, SyncByteStream)

        response.request = request
        response.stream = BoundSyncStream(response.stream, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding

        logger.info(
            'HTTP Request: %s %s "%s %d %s"',
            request.method,
            request.url,
            response.http_version,
            response.status_code,
            response.reason_phrase,
        )

        return response

    def get(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `GET` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def options(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send an `OPTIONS` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "OPTIONS",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def head(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `HEAD` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "HEAD",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def post(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `POST` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "POST",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def put(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `PUT` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "PUT",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def patch(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `PATCH` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "PATCH",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def delete(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `DELETE` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "DELETE",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def query(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `QUERY` request.

        **Parameters**: See `httpx2.request`.
        """
        return self.request(
            "QUERY",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    def close(self) -> None:
        """
        Close transport and proxies.
        """
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED

            self._transport.close()
            for transport in self._mounts.values():
                if transport is not None:
                    transport.close()

    def __enter__(self: T) -> T:
        if self._state != ClientState.UNOPENED:
            msg = {
                ClientState.OPENED: "Cannot open a client instance more than once.",
                ClientState.CLOSED: ("Cannot reopen a client instance, once it has been closed."),
            }[self._state]
            raise RuntimeError(msg)

        self._state = ClientState.OPENED

        self._transport.__enter__()
        for transport in self._mounts.values():
            if transport is not None:
                transport.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self._state = ClientState.CLOSED

        self._transport.__exit__(exc_type, exc_value, traceback)
        for transport in self._mounts.values():
            if transport is not None:
                transport.__exit__(exc_type, exc_value, traceback)


class AsyncClient(BaseClient):
    """
    An asynchronous HTTP client, with connection pooling, HTTP/2, redirects,
    cookie persistence, etc.

    It can be shared between tasks.

    Usage:

    ```python
    >>> async with httpx2.AsyncClient() as client:
    >>>     response = await client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **mounts** - *(optional)* A dictionary mapping URL patterns to transports,
    used to route requests through specific transports based on the URL.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: None | (typing.Mapping[str, AsyncBaseTransport | None]) = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: None | (typing.Mapping[str, list[EventHook]]) = None,
        base_url: URL | str = "",
        transport: AsyncBaseTransport | None = None,
        trust_env: bool = True,
        default_encoding: str | typing.Callable[[bytes], str | None] = "utf-8",
    ) -> None:
        super().__init__(
            auth=auth,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            event_hooks=event_hooks,
            base_url=base_url,
            trust_env=trust_env,
            default_encoding=default_encoding,
        )

        if http2:
            try:
                import h2  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using http2=True, but the 'h2' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[http2]`."
                ) from None

        allow_env_proxies = trust_env and transport is None
        proxy_map = self._get_proxy_map(proxy, allow_env_proxies)

        self._transport = self._init_transport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            transport=transport,
        )

        self._mounts: dict[URLPattern, AsyncBaseTransport | None] = {
            URLPattern(key): None
            if proxy is None
            else self._init_proxy_transport(
                proxy,
                verify=verify,
                cert=cert,
                trust_env=trust_env,
                http1=http1,
                http2=http2,
                limits=limits,
            )
            for key, proxy in proxy_map.items()
        }
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: AsyncBaseTransport | None = None,
    ) -> AsyncBaseTransport:
        if transport is not None:
            return transport

        return AsyncHTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
        )

    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> AsyncBaseTransport:
        return AsyncHTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            proxy=proxy,
        )

    def _transport_for_url(self, url: URL) -> AsyncBaseTransport:
        """
        Returns the transport instance that should be used for a given URL.
        This will either be the standard connection pool, or a proxy.
        """
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport

        return self._transport

    async def request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Build and send a request.

        Equivalent to:

        ```python
        request = client.build_request(...)
        response = await client.send(request, ...)
        ```

        See `AsyncClient.build_request()`, `AsyncClient.send()`
        and [Merging of configuration][0] for how the various parameters
        are merged with client-level configuration.

        [0]: advanced/clients.md#merging-of-configuration
        """

        if cookies is not None:  # pragma: no cover
            message = (
                "Setting per-request cookies=<...> is being deprecated, because "
                "the expected behaviour on cookie persistence is ambiguous. Set "
                "cookies directly on the client instance instead."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)

        request = self.build_request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            extensions=extensions,
        )
        return await self.send(request, auth=auth, follow_redirects=follow_redirects)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> AsyncGenerator[Response]:
        """
        Alternative to `httpx2.request()` that streams the response body
        instead of loading it into memory at once.

        **Parameters**: See `httpx2.request`.

        See also: [Streaming Responses][0]

        [0]: quickstart.md#streaming-responses
        """
        request = self.build_request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            extensions=extensions,
        )
        response = await self.send(
            request=request,
            auth=auth,
            follow_redirects=follow_redirects,
            stream=True,
        )
        try:
            yield response
        finally:
            await response.aclose()

    @asynccontextmanager
    async def sse(
        self,
        url: URL | str,
        *,
        method: str = "GET",
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
        max_event_size: int | None = DEFAULT_MAX_EVENT_SIZE_BYTES,
    ) -> AsyncGenerator[EventSource]:
        """
        Connect to a server-sent events endpoint and yield an `EventSource`.

        Iterating the `EventSource` yields `ServerSentEvent` instances.

        **Parameters**: See `httpx2.request`.
        """
        async with self.stream(
            method,
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers={"Accept": "text/event-stream", "Cache-Control": "no-store"} | Headers(headers),
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as response:
            yield EventSource(response, max_event_size=max_event_size)

    @asynccontextmanager
    async def websocket(
        self,
        url: URL | str,
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
    ) -> AsyncGenerator[AsyncWebSocketSession]:
        """
        Open a WebSocket session, using this client's configuration.

        The session is closed automatically when exiting the context manager.

        ```python
        async with httpx2.AsyncClient() as client:
            async with client.websocket("ws://localhost:8000/ws") as ws:
                await ws.send_text("Hello!")
                message = await ws.receive_text()
        ```
        """
        try:
            from .websockets._api import aconnect_ws
        except ImportError:  # pragma: no cover
            raise ImportError(
                "WebSocket support requires the `wsproto` package. Install it with `pip install httpx2[ws]`."
            )

        async with aconnect_ws(
            str(url),
            self,
            max_message_size_bytes=max_message_size_bytes,
            queue_size=queue_size,
            keepalive_ping_interval_seconds=keepalive_ping_interval_seconds,
            keepalive_ping_timeout_seconds=keepalive_ping_timeout_seconds,
            subprotocols=subprotocols,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        ) as session:
            yield session

    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Response:
        """
        Send a request.

        The request is sent as-is, unmodified.

        Typically you'll want to build one with `AsyncClient.build_request()`
        so that any client-level configuration is merged into the request,
        but passing an explicit `httpx2.Request()` is supported as well.

        See also: [Request instances][0]

        [0]: advanced/clients.md#request-instances
        """
        if self._state == ClientState.CLOSED:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects

        self._set_timeout(request)

        auth = self._build_request_auth(request, auth)

        response = await self._send_handling_auth(
            request,
            auth=auth,
            follow_redirects=follow_redirects,
            history=[],
        )
        try:
            if not stream:
                await response.aread()

            return response

        except BaseException as exc:
            await response.aclose()
            raise exc

    async def _send_handling_auth(
        self,
        request: Request,
        auth: Auth,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response:
        auth_flow = auth.async_auth_flow(request)
        try:
            request = await auth_flow.__anext__()

            while True:
                response = await self._send_handling_redirects(
                    request,
                    follow_redirects=follow_redirects,
                    history=history,
                )
                try:
                    try:
                        next_request = await auth_flow.asend(response)
                    except StopAsyncIteration:
                        return response

                    response.history = list(history)
                    await response.aread()
                    request = next_request
                    history.append(response)

                except BaseException as exc:
                    await response.aclose()
                    raise exc
        finally:
            await auth_flow.aclose()

    async def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response:
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects("Exceeded maximum allowed redirects.", request=request)

            for hook in self._event_hooks["request"]:
                await hook(request)

            response = await self._send_single_request(request)
            try:
                for hook in self._event_hooks["response"]:
                    await hook(response)

                response.history = list(history)

                if not response.has_redirect_location:
                    return response

                request = self._build_redirect_request(request, response)
                history = history + [response]

                if follow_redirects:
                    await response.aread()
                else:
                    response.next_request = request
                    return response

            except BaseException as exc:
                await response.aclose()
                raise exc

    async def _send_single_request(self, request: Request) -> Response:
        """
        Sends a single request, without handling any redirections.
        """
        transport = self._transport_for_url(request.url)
        start = time.perf_counter()

        if not isinstance(request.stream, AsyncByteStream):
            raise RuntimeError("Attempted to send a sync request with an AsyncClient instance.")

        with request_context(request=request):
            response = await transport.handle_async_request(request)

        assert isinstance(response.stream, AsyncByteStream)
        response.request = request
        response.stream = BoundAsyncStream(response.stream, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding

        logger.info(
            'HTTP Request: %s %s "%s %d %s"',
            request.method,
            request.url,
            response.http_version,
            response.status_code,
            response.reason_phrase,
        )

        return response

    async def get(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `GET` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def options(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send an `OPTIONS` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "OPTIONS",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def head(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `HEAD` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "HEAD",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def post(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `POST` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "POST",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def put(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `PUT` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "PUT",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def patch(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `PATCH` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "PATCH",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def delete(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `DELETE` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "DELETE",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def query(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response:
        """
        Send a `QUERY` request.

        **Parameters**: See `httpx2.request`.
        """
        return await self.request(
            "QUERY",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def aclose(self) -> None:
        """
        Close transport and proxies.
        """
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED

            await self._transport.aclose()
            for proxy in self._mounts.values():
                if proxy is not None:
                    await proxy.aclose()

    async def __aenter__(self: U) -> U:
        if self._state != ClientState.UNOPENED:
            msg = {
                ClientState.OPENED: "Cannot open a client instance more than once.",
                ClientState.CLOSED: ("Cannot reopen a client instance, once it has been closed."),
            }[self._state]
            raise RuntimeError(msg)

        self._state = ClientState.OPENED

        await self._transport.__aenter__()
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self._state = ClientState.CLOSED

        await self._transport.__aexit__(exc_type, exc_value, traceback)
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aexit__(exc_type, exc_value, traceback)
<|endoftext|>from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from types import ModuleType

__all__ = ["alias_httpx"]


class _AliasLoader(importlib.abc.Loader):
    _original_spec: importlib.machinery.ModuleSpec | None

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._real_name)
        self._original_spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        module.__spec__ = self._original_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, alias: str, real: str) -> None:
        self._alias = alias
        self._real = real

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self._alias and not fullname.startswith(self._alias + "."):
            return None
        real_name = self._real + fullname.removeprefix(self._alias)
        if real_name not in sys.modules and importlib.util.find_spec(real_name) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(real_name))


def _alias(alias: str, module: ModuleType) -> None:
    existing = sys.modules.get(alias)
    if existing is not None and existing is not module:
        raise RuntimeError(f"{alias} was already imported; call `alias_httpx()` before any `import {alias}`.")

    if not any(isinstance(finder, _AliasFinder) and finder._alias == alias for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder(alias, module.__name__))
    sys.modules[alias] = module


def alias_httpx() -> None:
    """
    Make `import httpx` resolve to `httpx2`, and `import httpcore` to `httpcore2`, process-wide.

    Intended for applications migrating from `httpx`, so that dependencies still
    importing `httpx` or `httpcore` share the `httpx2` classes. Libraries should never call this.

    Must be called before anything imports `httpx` or `httpcore`. Calling it again is a no-op.
    """
    import httpcore2
    import httpx2

    _alias("httpx", httpx2)
    _alias("httpcore", httpcore2)
<|endoftext|>"""
Handlers for Content-Encoding.

See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding
"""

from __future__ import annotations

import codecs
import io
import itertools
import sys
import typing
import zlib

from ._exceptions import DecodingError

# Brotli support is optional
try:
    # The C bindings in `brotli` are recommended for CPython.
    import brotli
except ImportError:  # pragma: no cover
    try:
        # The CFFI bindings in `brotlicffi` are recommended for PyPy
        # and other environments.
        import brotlicffi as brotli
    except ImportError:
        brotli = None


# Zstandard support is optional on Python <= 3.13.
# On Python 3.14+, the stdlib includes an optional built-in zstd implementation.
if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 14):
        from compression.zstd import ZstdDecompressor, ZstdError
    else:
        from backports.zstd import ZstdDecompressor, ZstdError

    _zstandard_installed: bool
else:  # pragma: no cover
    try:
        if sys.version_info >= (3, 14):
            from compression.zstd import ZstdDecompressor, ZstdError
        else:
            from backports.zstd import ZstdDecompressor, ZstdError

        _zstandard_installed = True
    except ImportError:
        _zstandard_installed = False


MAX_DECODE_CHUNK_SIZE = 2**20  # 1 MiB


class Decompressor(typing.Protocol):
    @property
    def unconsumed_tail(self) -> bytes: ...

    def decompress(self, data: bytes, max_length: int) -> bytes: ...

    def flush(self) -> bytes: ...


class ZlibDecompressor:
    """
    Drain a `zlib`/`gzip` decompressor in bounded pieces so a small compressed
    input cannot inflate to an unbounded buffer in a single call.
    """

    def __init__(self, decompressor: Decompressor) -> None:
        self.decompressor = decompressor

    def decompress(self, data: bytes) -> typing.Iterator[bytes]:
        decompressed = self.decompressor.decompress(data, MAX_DECODE_CHUNK_SIZE)
        while decompressed:
            yield decompressed
            decompressed = self.decompressor.decompress(self.decompressor.unconsumed_tail, MAX_DECODE_CHUNK_SIZE)

    def flush(self) -> bytes:
        return self.decompressor.flush()


class ContentDecoder:
    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        raise NotImplementedError()  # pragma: no cover

    def flush(self) -> typing.Iterator[bytes]:
        raise NotImplementedError()  # pragma: no cover


class IdentityDecoder(ContentDecoder):
    """
    Handle unencoded data.
    """

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        yield data

    def flush(self) -> typing.Iterator[bytes]:
        yield from ()


class DeflateDecoder(ContentDecoder):
    """
    Handle 'deflate' decoding.

    See: https://stackoverflow.com/questions/1838699
    """

    def __init__(self) -> None:
        self.first_attempt = True
        self.decompressor = ZlibDecompressor(zlib.decompressobj())

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        was_first_attempt = self.first_attempt
        self.first_attempt = False
        try:
            yield from self.decompressor.decompress(data)
        except zlib.error as exc:
            if was_first_attempt:
                self.decompressor = ZlibDecompressor(zlib.decompressobj(-zlib.MAX_WBITS))
                yield from self.decode(data)
            else:
                raise DecodingError(str(exc)) from exc

    def flush(self) -> typing.Iterator[bytes]:
        try:
            yield self.decompressor.flush()
        except zlib.error as exc:  # pragma: no cover
            raise DecodingError(str(exc)) from exc


class GZipDecoder(ContentDecoder):
    """
    Handle 'gzip' decoding.

    See: https://stackoverflow.com/questions/1838699
    """

    def __init__(self) -> None:
        self.decompressor = ZlibDecompressor(zlib.decompressobj(zlib.MAX_WBITS | 16))

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        try:
            yield from self.decompressor.decompress(data)
        except zlib.error as exc:
            raise DecodingError(str(exc)) from exc

    def flush(self) -> typing.Iterator[bytes]:
        try:
            yield self.decompressor.flush()
        except zlib.error as exc:  # pragma: no cover
            raise DecodingError(str(exc)) from exc


class BrotliDecoder(ContentDecoder):
    """
    Handle 'brotli' decoding.

    Requires `pip install brotlipy`. See: https://brotlipy.readthedocs.io/
        or   `pip install brotli`. See https://github.com/google/brotli
    Supports both 'brotlipy' and 'Brotli' packages since they share an import
    name. The top branches are for 'brotlipy' and bottom branches for 'Brotli'
    """

    def __init__(self) -> None:
        if brotli is None:  # pragma: no cover
            raise ImportError(
                "Using 'BrotliDecoder', but neither of the 'brotlicffi' or 'brotli' "
                "packages have been installed. "
                "Make sure to install httpx using `pip install httpx[brotli]`."
            ) from None

        self.decompressor = brotli.Decompressor()
        self.seen_data = False
        self._decompress: typing.Callable[..., bytes]
        if hasattr(self.decompressor, "decompress"):
            # The 'brotlicffi' package.
            self._decompress = self.decompressor.decompress  # pragma: no cover
        else:
            # The 'brotli' package.
            self._decompress = self.decompressor.process

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        if not data:
            return
        self.seen_data = True
        try:
            # The C backend may allocate nearly twice the requested threshold.
            output_buffer_limit = MAX_DECODE_CHUNK_SIZE // 2
            decompressed = self._decompress(data, output_buffer_limit=output_buffer_limit)
            while decompressed:
                yield decompressed
                decompressed = self._decompress(b"", output_buffer_limit=output_buffer_limit)
        except brotli.error as exc:
            raise DecodingError(str(exc)) from exc

    def flush(self) -> typing.Iterator[bytes]:
        if not self.seen_data:
            return
        try:
            if hasattr(self.decompressor, "finish"):
                # Only available in the 'brotlicffi' package.

                # As the decompressor decompresses eagerly, this
                # will never actually emit any data. However, it will potentially throw
                # errors if a truncated or damaged data stream has been used.
                self.decompressor.finish()  # pragma: no cover
        except brotli.error as exc:  # pragma: no cover
            raise DecodingError(str(exc)) from exc
        yield from ()


class ZStandardDecoder(ContentDecoder):
    """Handle 'zstd' RFC 8878 decoding.

    On Python 3.13 and below, this requires `pip install httpx2[zstd]`.
    """

    # inspired by the ZstdDecoder implementation in urllib3
    def __init__(self) -> None:
        if not _zstandard_installed:  # pragma: no cover
            raise ImportError(
                "Using 'ZStandardDecoder', ...Make sure to install httpx using `pip install httpx[zstd]`."
            ) from None

        self.decompressor = ZstdDecompressor()
        self.seen_data = False

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        if not data:
            return
        self.seen_data = True
        try:
            if self.decompressor.eof:
                data = self.decompressor.unused_data + data
                self.decompressor = ZstdDecompressor()
            while True:
                decompressed = self.decompressor.decompress(data, MAX_DECODE_CHUNK_SIZE)
                while decompressed:
                    yield decompressed
                    if self.decompressor.needs_input or self.decompressor.eof:
                        break
                    decompressed = self.decompressor.decompress(b"", MAX_DECODE_CHUNK_SIZE)
                if not (self.decompressor.eof and self.decompressor.unused_data):
                    break
                data = self.decompressor.unused_data
                self.decompressor = ZstdDecompressor()
        except ZstdError as exc:
            raise DecodingError(str(exc)) from exc

    def flush(self) -> typing.Iterator[bytes]:
        if not self.seen_data:
            return
        if not self.decompressor.eof:
            raise DecodingError("Zstandard data is incomplete")  # pragma: no cover
        yield from ()


class MultiDecoder(ContentDecoder):
    """
    Handle the case where multiple encodings have been applied.
    """

    max_decode_links: typing.ClassVar[int] = 5

    def __init__(self, encodings: typing.Sequence[str]) -> None:
        """
        'encodings' should be the content codings in the order in which
        each was applied.
        """
        codings = [encoding for encoding in encodings if encoding in SUPPORTED_DECODERS]
        if len(codings) > self.max_decode_links:
            raise DecodingError(f"Cannot apply more than {self.max_decode_links} content encodings.")
        # Note that we reverse the order for decoding.
        self.children: list[ContentDecoder] = [SUPPORTED_DECODERS[coding]() for coding in reversed(codings)]

    def decode(self, data: bytes) -> typing.Iterator[bytes]:
        streams: typing.Iterator[bytes] = iter((data,))
        for child in self.children:
            streams = self._pipe(child.decode, streams)
        yield from streams

    def flush(self) -> typing.Iterator[bytes]:
        streams: typing.Iterator[bytes] = iter(())
        for child in self.children:
            streams = itertools.chain(self._pipe(child.decode, streams), child.flush())
        yield from streams

    @staticmethod
    def _pipe(
        decode: typing.Callable[[bytes], typing.Iterator[bytes]],
        upstream: typing.Iterator[bytes],
    ) -> typing.Iterator[bytes]:
        for chunk in upstream:
            yield from decode(chunk)


class ByteChunker:
    """
    Handles returning byte content in fixed-size chunks.
    """

    def __init__(self, chunk_size: int | None = None) -> None:
        self._buffer = io.BytesIO()
        self._chunk_size = chunk_size

    def decode(self, content: bytes) -> list[bytes]:
        if self._chunk_size is None:
            return [content] if content else []

        self._buffer.write(content)
        if self._buffer.tell() >= self._chunk_size:
            value = self._buffer.getvalue()
            chunks = [value[i : i + self._chunk_size] for i in range(0, len(value), self._chunk_size)]
            if len(chunks[-1]) == self._chunk_size:
                self._buffer.seek(0)
                self._buffer.truncate()
                return chunks
            else:
                self._buffer.seek(0)
                self._buffer.write(chunks[-1])
                self._buffer.truncate()
                return chunks[:-1]
        else:
            return []

    def flush(self) -> list[bytes]:
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()
        return [value] if value else []


class TextChunker:
    """
    Handles returning text content in fixed-size chunks.
    """

    def __init__(self, chunk_size: int | None = None) -> None:
        self._buffer = io.StringIO()
        self._chunk_size = chunk_size

    def decode(self, content: str) -> list[str]:
        if self._chunk_size is None:
            return [content] if content else []

        self._buffer.write(content)
        if self._buffer.tell() >= self._chunk_size:
            value = self._buffer.getvalue()
            chunks = [value[i : i + self._chunk_size] for i in range(0, len(value), self._chunk_size)]
            if len(chunks[-1]) == self._chunk_size:
                self._buffer.seek(0)
                self._buffer.truncate()
                return chunks
            else:
                self._buffer.seek(0)
                self._buffer.write(chunks[-1])
                self._buffer.truncate()
                return chunks[:-1]
        else:
            return []

    def flush(self) -> list[str]:
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()
        return [value] if value else []


class TextDecoder:
    """
    Handles incrementally decoding bytes into text
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.decoder = codecs.getincrementaldecoder(encoding)(errors="replace")

    def decode(self, data: bytes) -> str:
        return self.decoder.decode(data)

    def flush(self) -> str:
        return self.decoder.decode(b"", True)


class LineDecoder:
    """
    Handles incrementally reading lines from text.

    Has the same behaviour as the stdllib splitlines,
    but handling the input iteratively.
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.trailing_cr: bool = False

    def decode(self, text: str) -> list[str]:
        # See https://docs.python.org/3/library/stdtypes.html#str.splitlines
        NEWLINE_CHARS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"

        # We always push a trailing `\r` into the next decode iteration.
        if self.trailing_cr:
            text = "\r" + text
            self.trailing_cr = False
        if text.endswith("\r"):
            self.trailing_cr = True
            text = text[:-1]

        if not text:
            # NOTE: the edge case input of empty text doesn't occur in practice,
            # because other httpx internals filter out this value
            return []  # pragma: no cover

        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = text.splitlines()

        if len(lines) == 1 and not trailing_newline:
            # No new lines, buffer the input and continue.
            self.buffer.append(lines[0])
            return []

        if self.buffer:
            # Include any existing buffer in the first portion of the
            # splitlines result.
            lines = ["".join(self.buffer) + lines[0]] + lines[1:]
            self.buffer = []

        if not trailing_newline:
            # If the last segment of splitlines is not newline terminated,
            # then drop it from our output and start a new buffer.
            self.buffer = [lines.pop()]

        return lines

    def flush(self) -> list[str]:
        if not self.buffer and not self.trailing_cr:
            return []

        lines = ["".join(self.buffer)]
        self.buffer = []
        self.trailing_cr = False
        return lines


SUPPORTED_DECODERS: dict[str, type[ContentDecoder]] = {
    "identity": IdentityDecoder,
    "gzip": GZipDecoder,
    "deflate": DeflateDecoder,
    "br": BrotliDecoder,
    "zstd": ZStandardDecoder,
}


if brotli is None:
    SUPPORTED_DECODERS.pop("br")  # pragma: no cover
if not _zstandard_installed:
    SUPPORTED_DECODERS.pop("zstd")  # pragma: no cover
<|endoftext|>from __future__ import annotations

import hashlib
import os
import re
import time
import typing
from base64 import b64encode
from urllib.request import parse_http_list

from ._exceptions import ProtocolError
from ._models import Cookies, Request, Response
from ._utils import to_bytes, to_str

if typing.TYPE_CHECKING:
    from hashlib import _Hash


__all__ = ["Auth", "BasicAuth", "DigestAuth", "FunctionAuth", "NetRCAuth"]


class Auth:
    """
    Base class for all authentication schemes.

    To implement a custom authentication scheme, subclass `Auth` and override
    the `.auth_flow()` method.

    If the authentication scheme does I/O such as disk access or network calls, or uses
    synchronization primitives such as locks, you should override `.sync_auth_flow()`
    and/or `.async_auth_flow()` instead of `.auth_flow()` to provide specialized
    implementations that will be used by `Client` and `AsyncClient` respectively.
    """

    requires_request_body = False
    requires_response_body = False

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        """
        Execute the authentication flow.

        To dispatch a request, `yield` it:

        ```
        yield request
        ```

        The client will `.send()` the response back into the flow generator. You can
        access it like so:

        ```
        response = yield request
        ```

        A `return` (or reaching the end of the generator) will result in the
        client returning the last response obtained from the server.

        You can dispatch as many requests as is necessary.
        """
        yield request

    def sync_auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        """
        Execute the authentication flow synchronously.

        By default, this defers to `.auth_flow()`. You should override this method
        when the authentication scheme does I/O and/or uses concurrency primitives.
        """
        if self.requires_request_body:
            request.read()

        flow = self.auth_flow(request)
        request = next(flow)

        while True:
            response = yield request
            if self.requires_response_body:
                response.read()

            try:
                request = flow.send(response)
            except StopIteration:
                break

    async def async_auth_flow(self, request: Request) -> typing.AsyncGenerator[Request, Response]:
        """
        Execute the authentication flow asynchronously.

        By default, this defers to `.auth_flow()`. You should override this method
        when the authentication scheme does I/O and/or uses concurrency primitives.
        """
        if self.requires_request_body:
            await request.aread()

        flow = self.auth_flow(request)
        request = next(flow)

        while True:
            response = yield request
            if self.requires_response_body:
                await response.aread()

            try:
                request = flow.send(response)
            except StopIteration:
                break


class FunctionAuth(Auth):
    """
    Allows the 'auth' argument to be passed as a simple callable function,
    that takes the request, and returns a new, modified request.
    """

    def __init__(self, func: typing.Callable[[Request], Request]) -> None:
        self._func = func

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        yield self._func(request)


class BasicAuth(Auth):
    """
    Allows the 'auth' argument to be passed as a (username, password) pair,
    and uses HTTP Basic authentication.
    """

    def __init__(self, username: str | bytes, password: str | bytes) -> None:
        self._auth_header = self._build_auth_header(username, password)

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        request.headers["Authorization"] = self._auth_header
        yield request

    def _build_auth_header(self, username: str | bytes, password: str | bytes) -> str:
        userpass = b":".join((to_bytes(username), to_bytes(password)))
        token = b64encode(userpass).decode()
        return f"Basic {token}"


class NetRCAuth(Auth):
    """
    Use a 'netrc' file to lookup basic auth credentials based on the url host.
    """

    def __init__(self, file: str | None = None) -> None:
        # Lazily import 'netrc'.
        # There's no need for us to load this module unless 'NetRCAuth' is being used.
        import netrc

        self._netrc_info = netrc.netrc(file)

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        auth_info = self._netrc_info.authenticators(request.url.host)
        if auth_info is None or not auth_info[2]:
            # The netrc file did not have authentication credentials for this host.
            yield request
        else:
            # Build a basic auth header with credentials from the netrc file.
            request.headers["Authorization"] = self._build_auth_header(username=auth_info[0], password=auth_info[2])
            yield request

    def _build_auth_header(self, username: str | bytes, password: str | bytes) -> str:
        userpass = b":".join((to_bytes(username), to_bytes(password)))
        token = b64encode(userpass).decode()
        return f"Basic {token}"


class DigestAuth(Auth):
    _ALGORITHM_TO_HASH_FUNCTION: dict[str, typing.Callable[[bytes], _Hash]] = {
        "MD5": hashlib.md5,
        "MD5-SESS": hashlib.md5,
        "SHA": hashlib.sha1,
        "SHA-SESS": hashlib.sha1,
        "SHA-256": hashlib.sha256,
        "SHA-256-SESS": hashlib.sha256,
        "SHA-512": hashlib.sha512,
        "SHA-512-SESS": hashlib.sha512,
    }

    def __init__(self, username: str | bytes, password: str | bytes) -> None:
        self._username = to_bytes(username)
        self._password = to_bytes(password)
        self._last_challenge: _DigestAuthChallenge | None = None
        self._nonce_count = 1

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        if self._last_challenge:
            request.headers["Authorization"] = self._build_auth_header(request, self._last_challenge)

        response = yield request

        if response.status_code != 401 or "www-authenticate" not in response.headers:
            # If the response is not a 401 then we don't
            # need to build an authenticated request.
            return

        for auth_header in response.headers.get_list("www-authenticate"):
            if auth_header.lower().startswith("digest "):
                break
        else:
            # If the response does not include a 'WWW-Authenticate: Digest ...'
            # header, then we don't need to build an authenticated request.
            return

        self._last_challenge = self._parse_challenge(request, response, auth_header)
        self._nonce_count = 1

        request.headers["Authorization"] = self._build_auth_header(request, self._last_challenge)
        if response.cookies:
            Cookies(response.cookies).set_cookie_header(request=request)
        yield request

    def _parse_challenge(self, request: Request, response: Response, auth_header: str) -> _DigestAuthChallenge:
        """
        Returns a challenge from a Digest WWW-Authenticate header.
        These take the form of:
        `Digest realm="realm@host.com",qop="auth,auth-int",nonce="abc",opaque="xyz"`
        """
        scheme, _, fields = auth_header.partition(" ")

        # This method should only ever have been called with a Digest auth header.
        assert scheme.lower() == "digest"

        header_dict: dict[str, str] = {}
        for field in parse_http_list(fields):
            key, value = field.strip().split("=", 1)
            header_dict[key] = value.strip('"')

        try:
            realm = header_dict["realm"].encode()
            nonce = header_dict["nonce"].encode()
            algorithm = header_dict.get("algorithm", "MD5")
            opaque = header_dict["opaque"].encode() if "opaque" in header_dict else None
            qop = header_dict["qop"].encode() if "qop" in header_dict else None
            return _DigestAuthChallenge(realm=realm, nonce=nonce, algorithm=algorithm, opaque=opaque, qop=qop)
        except KeyError as exc:
            message = "Malformed Digest WWW-Authenticate header"
            raise ProtocolError(message, request=request) from exc

    def _build_auth_header(self, request: Request, challenge: _DigestAuthChallenge) -> str:
        hash_func = self._ALGORITHM_TO_HASH_FUNCTION[challenge.algorithm.upper()]

        def digest(data: bytes) -> bytes:
            return hash_func(data).hexdigest().encode()

        A1 = b":".join((self._username, challenge.realm, self._password))

        path = request.url.raw_path
        A2 = b":".join((request.method.encode(), path))
        # TODO: implement auth-int
        HA2 = digest(A2)

        nc_value = b"%08x" % self._nonce_count
        cnonce = self._get_client_nonce(self._nonce_count, challenge.nonce)
        self._nonce_count += 1

        HA1 = digest(A1)
        if challenge.algorithm.lower().endswith("-sess"):
            HA1 = digest(b":".join((HA1, challenge.nonce, cnonce)))

        qop = self._resolve_qop(challenge.qop, request=request)
        if qop is None:
            # Following RFC 2069
            digest_data = [HA1, challenge.nonce, HA2]
        else:
            # Following RFC 2617/7616
            digest_data = [HA1, challenge.nonce, nc_value, cnonce, qop, HA2]

        format_args = {
            "username": self._username,
            "realm": challenge.realm,
            "nonce": challenge.nonce,
            "uri": path,
            "response": digest(b":".join(digest_data)),
            "algorithm": challenge.algorithm.encode(),
        }
        if challenge.opaque:
            format_args["opaque"] = challenge.opaque
        if qop:
            format_args["qop"] = b"auth"
            format_args["nc"] = nc_value
            format_args["cnonce"] = cnonce

        return "Digest " + self._get_header_value(format_args)

    def _get_client_nonce(self, nonce_count: int, nonce: bytes) -> bytes:
        s = str(nonce_count).encode()
        s += nonce
        s += time.ctime().encode()
        s += os.urandom(8)

        return hashlib.sha1(s).hexdigest()[:16].encode()

    def _get_header_value(self, header_fields: dict[str, bytes]) -> str:
        NON_QUOTED_FIELDS = ("algorithm", "qop", "nc")
        QUOTED_TEMPLATE = '{}="{}"'
        NON_QUOTED_TEMPLATE = "{}={}"

        header_value = ""
        for i, (field, value) in enumerate(header_fields.items()):
            if i > 0:
                header_value += ", "
            template = QUOTED_TEMPLATE if field not in NON_QUOTED_FIELDS else NON_QUOTED_TEMPLATE
            header_value += template.format(field, to_str(value))

        return header_value

    def _resolve_qop(self, qop: bytes | None, request: Request) -> bytes | None:
        if qop is None:
            return None
        qops = re.split(b", ?", qop)
        if b"auth" in qops:
            return b"auth"

        if qops == [b"auth-int"]:
            raise NotImplementedError("Digest auth-int support is not yet implemented")

        message = f'Unexpected qop value "{qop!r}" in digest auth'
        raise ProtocolError(message, request=request)


class _DigestAuthChallenge(typing.NamedTuple):
    realm: bytes
    nonce: bytes
    algorithm: str
    opaque: bytes | None
    qop: bytes | None
<|endoftext|>"""
Type definitions for type checking purposes.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from http.cookiejar import CookieJar
from typing import IO, TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ._auth import Auth  # noqa: F401
    from ._config import Proxy, Timeout  # noqa: F401
    from ._models import Cookies, Headers, Request  # noqa: F401
    from ._urls import URL, QueryParams  # noqa: F401


PrimitiveData = str | int | float | bool | None

URLTypes = Union["URL", str]

QueryParamTypes = Union[
    "QueryParams",
    Mapping[str, PrimitiveData | Sequence[PrimitiveData]],
    list[tuple[str, PrimitiveData]],
    tuple[tuple[str, PrimitiveData], ...],
    str,
    bytes,
]

HeaderTypes = Union[
    "Headers",
    Mapping[str, str],
    Mapping[bytes, bytes],
    Sequence[tuple[str, str]],
    Sequence[tuple[bytes, bytes]],
]

CookieTypes = Union["Cookies", CookieJar, dict[str, str], list[tuple[str, str]]]

TimeoutTypes = Union[float | None, tuple[float | None, float | None, float | None, float | None], "Timeout"]
ProxyTypes = Union["URL", str, "Proxy"]
CertTypes = str | tuple[str, str] | tuple[str, str, str]

AuthTypes = Union[tuple[str | bytes, str | bytes], Callable[["Request"], "Request"], "Auth"]

RequestContent = str | bytes | Iterable[bytes] | AsyncIterable[bytes]
ResponseContent = str | bytes | Iterable[bytes] | AsyncIterable[bytes]
ResponseExtensions = Mapping[str, Any]

RequestData = Mapping[str, Any]

FileContent = IO[bytes] | bytes | str
FileTypes = (
    # # file (or bytes)
    FileContent
    # # (filename, file (or bytes))
    | tuple[str | None, FileContent]
    # # (filename, file (or bytes), content_type)
    | tuple[str | None, FileContent, str | None]
    | tuple[str | None, FileContent, str | None, Mapping[str, str]]
)
RequestFiles = Mapping[str, FileTypes] | Sequence[tuple[str, FileTypes]]

RequestExtensions = Mapping[str, Any]

__all__ = ["AsyncByteStream", "SyncByteStream"]


class SyncByteStream:
    def __iter__(self) -> Iterator[bytes]:
        raise NotImplementedError("The '__iter__' method must be implemented.")  # pragma: no cover
        yield b""  # pragma: no cover

    def close(self) -> None:
        """
        Subclasses can override this method to release any network resources
        after a request/response cycle is complete.
        """


class AsyncByteStream:
    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise NotImplementedError("The '__aiter__' method must be implemented.")  # pragma: no cover
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        pass
<|endoftext|>from __future__ import annotations

import inspect
import warnings
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping
from json import dumps as json_dumps
from typing import (
    Any,
)
from urllib.parse import urlencode

from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import (
    AsyncByteStream,
    RequestContent,
    RequestData,
    RequestFiles,
    ResponseContent,
    SyncByteStream,
)
from ._utils import peek_filelike_remaining_length, primitive_value_to_str

__all__ = ["ByteStream"]


class ByteStream(AsyncByteStream, SyncByteStream):
    def __init__(self, stream: bytes) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[bytes]:
        yield self._stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._stream


class IteratorByteStream(SyncByteStream):
    CHUNK_SIZE = 65_536

    def __init__(self, stream: Iterable[bytes]) -> None:
        self._stream = stream
        self._is_stream_consumed = False
        self._is_generator = inspect.isgenerator(stream)

    def __iter__(self) -> Iterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()

        self._is_stream_consumed = True
        if hasattr(self._stream, "read"):
            # File-like interfaces should use 'read' directly.
            chunk = self._stream.read(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = self._stream.read(self.CHUNK_SIZE)
        else:
            # Otherwise iterate.
            yield from self._stream


class AsyncIteratorByteStream(AsyncByteStream):
    CHUNK_SIZE = 65_536

    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream = stream
        self._is_stream_consumed = False
        self._is_generator = inspect.isasyncgen(stream)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()

        self._is_stream_consumed = True
        if hasattr(self._stream, "aread"):
            # File-like interfaces should use 'aread' directly.
            chunk = await self._stream.aread(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = await self._stream.aread(self.CHUNK_SIZE)
        else:
            # Otherwise iterate, making sure the wrapped stream is closed even if the
            # consumer stops early (e.g. an exception is raised part-way through decoding).
            stream = self._stream.__aiter__()
            try:
                async for part in stream:
                    yield part
            finally:
                if isinstance(stream, AsyncGenerator):
                    await stream.aclose()


class UnattachedStream(AsyncByteStream, SyncByteStream):
    """
    If a request or response is serialized using pickle, then it is no longer
    attached to a stream for I/O purposes. Any stream operations should result
    in `httpx2.StreamClosed`.
    """

    def __iter__(self) -> Iterator[bytes]:
        raise StreamClosed()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise StreamClosed()
        yield b""  # pragma: no cover


def encode_content(
    content: str | bytes | Iterable[bytes] | AsyncIterable[bytes],
) -> tuple[dict[str, str], SyncByteStream | AsyncByteStream]:
    if isinstance(content, (bytes, str)):
        body = content.encode("utf-8") if isinstance(content, str) else content
        content_length = len(body)
        headers = {"Content-Length": str(content_length)} if body else {}
        return headers, ByteStream(body)

    elif isinstance(content, Iterable) and not isinstance(content, dict):
        # `not isinstance(content, dict)` is a bit oddly specific, but it
        # catches a case that's easy for users to make in error, and would
        # otherwise pass through here, like any other bytes-iterable,
        # because `dict` happens to be iterable. See issue #2491.
        content_length_or_none = peek_filelike_remaining_length(content)

        if content_length_or_none is None:
            headers = {"Transfer-Encoding": "chunked"}
        else:
            headers = {"Content-Length": str(content_length_or_none)}
        return headers, IteratorByteStream(content)

    elif isinstance(content, AsyncIterable):
        headers = {"Transfer-Encoding": "chunked"}
        return headers, AsyncIteratorByteStream(content)

    raise TypeError(f"Unexpected type for 'content', {type(content)!r}")


def encode_urlencoded_data(data: RequestData) -> tuple[dict[str, str], ByteStream]:
    plain_data: list[tuple[str, str]] = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            plain_data.extend([(key, primitive_value_to_str(item)) for item in value])
        else:
            plain_data.append((key, primitive_value_to_str(value)))
    body = urlencode(plain_data, doseq=True).encode("utf-8")
    content_length = str(len(body))
    content_type = "application/x-www-form-urlencoded"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)


def encode_multipart_data(
    data: RequestData, files: RequestFiles, boundary: bytes | None
) -> tuple[dict[str, str], MultipartStream]:
    multipart = MultipartStream(data=data, files=files, boundary=boundary)
    headers = multipart.get_headers()
    return headers, multipart


def encode_text(text: str) -> tuple[dict[str, str], ByteStream]:
    body = text.encode("utf-8")
    content_length = str(len(body))
    content_type = "text/plain; charset=utf-8"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)


def encode_html(html: str) -> tuple[dict[str, str], ByteStream]:
    body = html.encode("utf-8")
    content_length = str(len(body))
    content_type = "text/html; charset=utf-8"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)


def encode_json(json: Any) -> tuple[dict[str, str], ByteStream]:
    body = json_dumps(json, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)


def encode_request(
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: Any | None = None,
    boundary: bytes | None = None,
) -> tuple[dict[str, str], SyncByteStream | AsyncByteStream]:
    """
    Handles encoding the given `content`, `data`, `files`, and `json`,
    returning a two-tuple of (<headers>, <stream>).
    """
    if data is not None and not isinstance(data, Mapping):
        # We prefer to separate `content=<bytes|str|byte iterator|bytes aiterator>`
        # for raw request content, and `data=<form data>` for url encoded or
        # multipart form content.
        #
        # However for compat with requests, we *do* still support
        # `data=<bytes...>` usages. We deal with that case here, treating it
        # as if `content=<...>` had been supplied instead.
        message = "Use 'content=<...>' to upload raw bytes/text content."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return encode_content(data)

    if content is not None:
        return encode_content(content)
    elif files:
        return encode_multipart_data(data or {}, files, boundary)
    elif data:
        return encode_urlencoded_data(data)
    elif json is not None:
        return encode_json(json)

    return {}, ByteStream(b"")


def encode_response(
    content: ResponseContent | None = None,
    text: str | None = None,
    html: str | None = None,
    json: Any | None = None,
) -> tuple[dict[str, str], SyncByteStream | AsyncByteStream]:
    """
    Handles encoding the given `content`, returning a two-tuple of
    (<headers>, <stream>).
    """
    if content is not None:
        return encode_content(content)
    elif text is not None:
        return encode_text(text)
    elif html is not None:
        return encode_html(html)
    elif json is not None:
        return encode_json(json)

    return {}, ByteStream(b"")
<|endoftext|>from .__version__ import __description__, __title__, __version__
from ._alias import *
from ._api import *
from ._auth import *
from ._client import *
from ._config import *
from ._content import *
from ._exceptions import *
from ._models import *
from ._sse import *
from ._status_codes import *
from ._transports import *
from ._types import *
from ._urls import *

__all__ = [
    # __version__.py
    "__description__",
    "__title__",
    "__version__",
    # _alias.py
    "alias_httpx",
    # _api.py
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "query",
    "request",
    "stream",
    "websocket",
    # _auth.py
    "Auth",
    "BasicAuth",
    "DigestAuth",
    "FunctionAuth",
    "NetRCAuth",
    # _client.py
    "AsyncClient",
    "Client",
    "USE_CLIENT_DEFAULT",
    # _config.py
    "create_ssl_context",
    "Limits",
    "Proxy",
    "Timeout",
    # _content.py
    "ByteStream",
    # _exceptions.py
    "CloseError",
    "ConnectError",
    "ConnectTimeout",
    "CookieConflict",
    "DecodingError",
    "HTTPError",
    "HTTPStatusError",
    "InvalidURL",
    "LocalProtocolError",
    "NetworkError",
    "PoolTimeout",
    "ProtocolError",
    "ProxyError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "RequestError",
    "RequestNotRead",
    "ResponseNotRead",
    "StreamClosed",
    "StreamConsumed",
    "StreamError",
    "TimeoutException",
    "TooManyRedirects",
    "TransportError",
    "UnsupportedProtocol",
    "WriteError",
    "WriteTimeout",
    # _models.py
    "Cookies",
    "Headers",
    "Request",
    "Response",
    # _sse.py
    "EventSource",
    "ServerSentEvent",
    "SSEError",
    # _status_codes.py
    "codes",
    # _transports
    "ASGITransport",
    "AsyncBaseTransport",
    "AsyncHTTPTransport",
    "BaseTransport",
    "HTTPTransport",
    "MockTransport",
    "WSGITransport",
    # _types.py
    "AsyncByteStream",
    "SyncByteStream",
    # _urls.py
    "Origin",
    "QueryParams",
    "URL",
]


__locals = locals()
for __name in __all__:
    if not __name.startswith("__"):
        setattr(__locals[__name], "__module__", "httpx2")  # noqa


def __getattr__(name: str) -> object:  # pragma: no cover
    if name == "main":
        import warnings

        warnings.warn(
            "`httpx2.main` is deprecated and will be removed in a future release. "
            "Use the `httpx2` CLI entry point instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ._main import main

        return main

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
<|endoftext|>from __future__ import annotations

import ipaddress
import sys
import typing
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode

import idna

if sys.version_info >= (3, 13):
    from warnings import deprecated  # pragma: no cover
else:
    from typing_extensions import deprecated  # pragma: no cover

from ._exceptions import HTTPXDeprecationWarning
from ._types import QueryParamTypes
from ._urlparse import urlparse
from ._utils import primitive_value_to_str

__all__ = ["URL", "Origin", "QueryParams"]


class URL:
    """
    url = httpx2.URL("HTTPS://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink")

    assert url.scheme == "https"
    assert url.username == "jo@email.com"
    assert url.password == "a secret"
    assert url.userinfo == b"jo%40email.com:a%20secret"
    assert url.host == "müller.de"
    assert url.raw_host == b"xn--mller-kva.de"
    assert url.port == 1234
    assert url.netloc == b"xn--mller-kva.de:1234"
    assert url.path == "/pa th"
    assert url.query == b"search=ab"
    assert url.raw_path == b"/pa%20th?search=ab"
    assert url.fragment == "anchorlink"

    The components of a URL are broken down like this:

       https://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink
    [scheme]   [  username  ] [password] [ host ][port][ path ] [ query ] [fragment]
               [       userinfo        ] [   netloc   ][    raw_path    ]

    Note that:

    * `url.scheme` is normalized to always be lowercased.

    * `url.host` is normalized to always be lowercased. Internationalized domain
      names are represented in unicode, without IDNA encoding applied. For instance:

      url = httpx2.URL("http://中国.icom.museum")
      assert url.host == "中国.icom.museum"
      url = httpx2.URL("http://xn--fiqs8s.icom.museum")
      assert url.host == "中国.icom.museum"

    * `url.raw_host` is normalized to always be lowercased, and is IDNA encoded.

      url = httpx2.URL("http://中国.icom.museum")
      assert url.raw_host == b"xn--fiqs8s.icom.museum"
      url = httpx2.URL("http://xn--fiqs8s.icom.museum")
      assert url.raw_host == b"xn--fiqs8s.icom.museum"

    * `url.port` is either None or an integer. URLs that include the default port for
      "http", "https", "ws", "wss", and "ftp" schemes have their port
      normalized to `None`.

      assert httpx2.URL("http://example.com") == httpx2.URL("http://example.com:80")
      assert httpx2.URL("http://example.com").port is None
      assert httpx2.URL("http://example.com:80").port is None

    * `url.userinfo` is raw bytes, without URL escaping. Usually you'll want to work
      with `url.username` and `url.password` instead, which handle the URL escaping.

    * `url.raw_path` is raw bytes of both the path and query, without URL escaping.
      This portion is used as the target when constructing HTTP requests. Usually you'll
      want to work with `url.path` instead.

    * `url.query` is raw bytes, without URL escaping. A URL query string portion can
      only be properly URL escaped when decoding the parameter names and values
      themselves.
    """

    def __init__(self, url: URL | str = "", **kwargs: typing.Any) -> None:
        if kwargs:
            allowed = {
                "scheme": str,
                "username": str,
                "password": str,
                "userinfo": bytes,
                "host": str,
                "port": int,
                "netloc": bytes,
                "path": str,
                "query": bytes,
                "raw_path": bytes,
                "fragment": str,
                "params": object,
            }

            # Perform type checking for all supported keyword arguments.
            for key, value in kwargs.items():
                if key not in allowed:
                    message = f"{key!r} is an invalid keyword argument for URL()"
                    raise TypeError(message)
                if value is not None and not isinstance(value, allowed[key]):
                    expected = allowed[key].__name__
                    seen = type(value).__name__
                    message = f"Argument {key!r} must be {expected} but got {seen}"
                    raise TypeError(message)
                if isinstance(value, bytes):
                    kwargs[key] = value.decode("ascii")

            if "params" in kwargs:
                # Replace any "params" keyword with the raw "query" instead.
                #
                # Ensure that empty params use `kwargs["query"] = None` rather
                # than `kwargs["query"] = ""`, so that generated URLs do not
                # include an empty trailing "?".
                params = kwargs.pop("params")
                kwargs["query"] = None if not params else str(QueryParams(params))

        if isinstance(url, str):
            self._uri_reference = urlparse(url, **kwargs)
        elif isinstance(url, URL):
            self._uri_reference = url._uri_reference.copy_with(**kwargs)
        else:
            raise TypeError(f"Invalid type for url.  Expected str or httpx2.URL, got {type(url)}: {url!r}")

    @property
    def scheme(self) -> str:
        """
        The URL scheme, such as "http", "https".
        Always normalised to lowercase.
        """
        return self._uri_reference.scheme

    @property
    def raw_scheme(self) -> bytes:
        """
        The raw bytes representation of the URL scheme, such as b"http", b"https".
        Always normalised to lowercase.
        """
        return self._uri_reference.scheme.encode("ascii")

    @property
    def userinfo(self) -> bytes:
        """
        The URL userinfo as a raw bytestring.
        For example: b"jo%40email.com:a%20secret".
        """
        return self._uri_reference.userinfo.encode("ascii")

    @property
    def username(self) -> str:
        """
        The URL username as a string, with URL decoding applied.
        For example: "jo@email.com"
        """
        userinfo = self._uri_reference.userinfo
        return unquote(userinfo.partition(":")[0])

    @property
    def password(self) -> str:
        """
        The URL password as a string, with URL decoding applied.
        For example: "a secret"
        """
        userinfo = self._uri_reference.userinfo
        return unquote(userinfo.partition(":")[2])

    @property
    def host(self) -> str:
        """
        The URL host as a string.
        Always normalized to lowercase, with IDNA hosts decoded into unicode.

        Examples:

        url = httpx2.URL("http://www.EXAMPLE.org")
        assert url.host == "www.example.org"

        url = httpx2.URL("http://中国.icom.museum")
        assert url.host == "中国.icom.museum"

        url = httpx2.URL("http://xn--fiqs8s.icom.museum")
        assert url.host == "中国.icom.museum"

        url = httpx2.URL("https://[::ffff:192.168.0.1]")
        assert url.host == "::ffff:192.168.0.1"
        """
        host: str = self._uri_reference.host

        if "xn--" in host:
            host = idna.decode(host, display=True)

        return host

    @property
    def raw_host(self) -> bytes:
        """
        The raw bytes representation of the URL host.
        Always normalized to lowercase, and IDNA encoded.

        Examples:

        url = httpx2.URL("http://www.EXAMPLE.org")
        assert url.raw_host == b"www.example.org"

        url = httpx2.URL("http://中国.icom.museum")
        assert url.raw_host == b"xn--fiqs8s.icom.museum"

        url = httpx2.URL("http://xn--fiqs8s.icom.museum")
        assert url.raw_host == b"xn--fiqs8s.icom.museum"

        url = httpx2.URL("https://[::ffff:192.168.0.1]")
        assert url.raw_host == b"::ffff:192.168.0.1"
        """
        return self._uri_reference.host.encode("ascii")

    @property
    def port(self) -> int | None:
        """
        The URL port as an integer.

        Note that the URL class performs port normalization as per the WHATWG spec.
        Default ports for "http", "https", "ws", "wss", and "ftp" schemes are always
        treated as `None`.

        For example:

        assert httpx2.URL("http://www.example.com") == httpx2.URL("http://www.example.com:80")
        assert httpx2.URL("http://www.example.com:80").port is None
        """
        return self._uri_reference.port

    @property
    def netloc(self) -> bytes:
        """
        Either `<host>` or `<host>:<port>` as bytes.
        Always normalized to lowercase, and IDNA encoded.

        This property may be used for generating the value of a request
        "Host" header.
        """
        return self._uri_reference.netloc.encode("ascii")

    @property
    def path(self) -> str:
        """
        The URL path as a string. Excluding the query string, and URL decoded.

        For example:

        url = httpx2.URL("https://example.com/pa%20th")
        assert url.path == "/pa th"
        """
        path = self._uri_reference.path or "/"
        return unquote(path)

    @property
    def query(self) -> bytes:
        """
        The URL query string, as raw bytes, excluding the leading b"?".

        This is necessarily a bytewise interface, because we cannot
        perform URL decoding of this representation until we've parsed
        the keys and values into a QueryParams instance.

        For example:

        url = httpx2.URL("https://example.com/?filter=some%20search%20terms")
        assert url.query == b"filter=some%20search%20terms"
        """
        query = self._uri_reference.query or ""
        return query.encode("ascii")

    @property
    def params(self) -> QueryParams:
        """
        The URL query parameters, neatly parsed and packaged into an immutable
        multidict representation.
        """
        return QueryParams(self._uri_reference.query)

    @property
    def raw_path(self) -> bytes:
        """
        The complete URL path and query string as raw bytes.
        Used as the target when constructing HTTP requests.

        For example:

        GET /users?search=some%20text HTTP/1.1
        Host: www.example.org
        Connection: close
        """
        path = self._uri_reference.path or "/"
        if self._uri_reference.query is not None:
            path += "?" + self._uri_reference.query
        return path.encode("ascii")

    @property
    def fragment(self) -> str:
        """
        The URL fragments, as used in HTML anchors.
        As a string, without the leading '#'.
        """
        return unquote(self._uri_reference.fragment or "")

    @property
    def is_absolute_url(self) -> bool:
        """
        Return `True` for absolute URLs such as 'http://example.com/path',
        and `False` for relative URLs such as '/path'.
        """
        # We don't use `.is_absolute` from `rfc3986` because it treats
        # URLs with a fragment portion as not absolute.
        # What we actually care about is if the URL provides
        # a scheme and hostname to which connections should be made.
        return bool(self._uri_reference.scheme and self._uri_reference.host)

    @property
    def is_relative_url(self) -> bool:
        """
        Return `False` for absolute URLs such as 'http://example.com/path',
        and `True` for relative URLs such as '/path'.
        """
        return not self.is_absolute_url

    @property
    def origin(self) -> Origin:
        """
        The URL origin as an immutable, hashable `Origin` value.

        The origin consists of the URL scheme, host, and effective port.
        User information, path, query, and fragment components are excluded.
        """
        return Origin(self)

    def copy_with(self, **kwargs: typing.Any) -> URL:
        """
        Copy this URL, returning a new URL with some components altered.
        Accepts the same set of parameters as the components that are made
        available via properties on the `URL` class.

        For example:

        url = httpx2.URL("https://www.example.com").copy_with(
            username="jo@gmail.com", password="a secret"
        )
        assert url == "https://jo%40email.com:a%20secret@www.example.com"
        """
        return URL(self, **kwargs)

    def copy_set_param(self, key: str, value: typing.Any = None) -> URL:
        return self.copy_with(params=self.params.set(key, value))

    def copy_add_param(self, key: str, value: typing.Any = None) -> URL:
        return self.copy_with(params=self.params.add(key, value))

    def copy_remove_param(self, key: str) -> URL:
        return self.copy_with(params=self.params.remove(key))

    def copy_merge_params(self, params: QueryParamTypes) -> URL:
        return self.copy_with(params=self.params.merge(params))

    def join(self, url: URL | str) -> URL:
        """
        Return an absolute URL, using this URL as the base.

        Eg.

        url = httpx2.URL("https://www.example.com/test")
        url = url.join("/new/path")
        assert url == "https://www.example.com/new/path"
        """
        from urllib.parse import urljoin

        return URL(urljoin(str(self), str(URL(url))))

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: typing.Any) -> bool:
        return isinstance(other, (URL, str)) and str(self) == str(URL(other))

    def __str__(self) -> str:
        return str(self._uri_reference)

    def __repr__(self) -> str:
        scheme, userinfo, host, port, path, query, fragment = self._uri_reference

        if ":" in userinfo:
            # Mask any password component.
            userinfo = f"{userinfo.split(':')[0]}:[secure]"

        authority = "".join(
            [
                f"{userinfo}@" if userinfo else "",
                f"[{host}]" if ":" in host else host,
                f":{port}" if port is not None else "",
            ]
        )
        url = "".join(
            [
                f"{self.scheme}:" if scheme else "",
                f"//{authority}" if authority else "",
                path,
                f"?{query}" if query is not None else "",
                f"#{fragment}" if fragment is not None else "",
            ]
        )

        return f"{self.__class__.__name__}({url!r})"

    @property
    @deprecated("URL.raw is deprecated.", category=HTTPXDeprecationWarning)
    def raw(self) -> tuple[bytes, bytes, int, bytes]:  # pragma: no cover
        import collections

        RawURL = collections.namedtuple("RawURL", ["raw_scheme", "raw_host", "port", "raw_path"])
        return RawURL(
            raw_scheme=self.raw_scheme,
            raw_host=self.raw_host,
            port=self.port,
            raw_path=self.raw_path,
        )


_ORIGIN_DEFAULT_PORTS = {
    "ftp": 21,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}


@dataclass(frozen=True, slots=True, init=False)
class Origin:
    """
    The scheme, host, and effective port of a URL.

    Origins are normalized, immutable, comparable, and hashable.

    See RFC 9110, Section 4.3.1: https://www.rfc-editor.org/rfc/rfc9110.html#name-uri-origin
    """

    scheme: str
    host: str
    port: int | None

    def __init__(self, url: URL | str) -> None:
        if not isinstance(url, URL):
            url = URL(url)

        if url.is_relative_url:
            raise ValueError("URL must be absolute to have an origin")

        host = url.host
        if b":" in url.raw_host:
            # IPv6 addresses may have multiple equivalent string forms. Store
            # their canonical compressed form so origin equality is numeric.
            host = str(ipaddress.IPv6Address(url.raw_host.decode("ascii")))

        port = url.port
        if port is None:
            port = _ORIGIN_DEFAULT_PORTS.get(url.scheme)

        object.__setattr__(self, "scheme", url.scheme)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)


class QueryParams(typing.Mapping[str, str]):
    """
    URL query parameters, as a multi-dict.
    """

    def __init__(self, *args: QueryParamTypes | None, **kwargs: typing.Any) -> None:
        assert len(args) < 2, "Too many arguments."
        assert not (args and kwargs), "Cannot mix named and unnamed arguments."

        value = args[0] if args else kwargs

        if value is None or isinstance(value, (str, bytes)):
            value = value.decode("ascii") if isinstance(value, bytes) else value
            self._dict = parse_qs(value, keep_blank_values=True)
        elif isinstance(value, QueryParams):
            self._dict = {k: list(v) for k, v in value._dict.items()}
        else:
            dict_value: dict[typing.Any, list[typing.Any]] = {}
            if isinstance(value, (list, tuple)):
                # Convert list inputs like:
                #     [("a", "123"), ("a", "456"), ("b", "789")]
                # To a dict representation, like:
                #     {"a": ["123", "456"], "b": ["789"]}
                for item in value:
                    dict_value.setdefault(item[0], []).append(item[1])
            else:
                # Convert dict inputs like:
                #    {"a": "123", "b": ["456", "789"]}
                # To dict inputs where values are always lists, like:
                #    {"a": ["123"], "b": ["456", "789"]}
                dict_value = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in value.items()}

            # Ensure that keys and values are neatly coerced to strings.
            # We coerce values `True` and `False` to JSON-like "true" and "false"
            # representations, and coerce `None` values to the empty string.
            self._dict = {str(k): [primitive_value_to_str(item) for item in v] for k, v in dict_value.items()}

    def keys(self) -> typing.KeysView[str]:
        """
        Return all the keys in the query params.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert list(q.keys()) == ["a", "b"]
        """
        return self._dict.keys()

    def values(self) -> typing.ValuesView[str]:
        """
        Return all the values in the query params. If a key occurs more than once
        only the first item for that key is returned.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert list(q.values()) == ["123", "789"]
        """
        return {k: v[0] for k, v in self._dict.items()}.values()

    def items(self) -> typing.ItemsView[str, str]:
        """
        Return all items in the query params. If a key occurs more than once
        only the first item for that key is returned.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert list(q.items()) == [("a", "123"), ("b", "789")]
        """
        return {k: v[0] for k, v in self._dict.items()}.items()

    def multi_items(self) -> list[tuple[str, str]]:
        """
        Return all items in the query params. Allow duplicate keys to occur.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert list(q.multi_items()) == [("a", "123"), ("a", "456"), ("b", "789")]
        """
        multi_items: list[tuple[str, str]] = []
        for k, v in self._dict.items():
            multi_items.extend([(k, i) for i in v])
        return multi_items

    def get(self, key: typing.Any, default: typing.Any = None) -> typing.Any:
        """
        Get a value from the query param for a given key. If the key occurs
        more than once, then only the first value is returned.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert q.get("a") == "123"
        """
        if key in self._dict:
            return self._dict[str(key)][0]
        return default

    def get_list(self, key: str) -> list[str]:
        """
        Get all values from the query param for a given key.

        Usage:

        q = httpx2.QueryParams("a=123&a=456&b=789")
        assert q.get_list("a") == ["123", "456"]
        """
        return list(self._dict.get(str(key), []))

    def set(self, key: str, value: typing.Any = None) -> QueryParams:
        """
        Return a new QueryParams instance, setting the value of a key.

        Usage:

        q = httpx2.QueryParams("a=123")
        q = q.set("a", "456")
        assert q == httpx2.QueryParams("a=456")
        """
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict[str(key)] = [primitive_value_to_str(value)]
        return q

    def add(self, key: str, value: typing.Any = None) -> QueryParams:
        """
        Return a new QueryParams instance, setting or appending the value of a key.

        Usage:

        q = httpx2.QueryParams("a=123")
        q = q.add("a", "456")
        assert q == httpx2.QueryParams("a=123&a=456")
        """
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict[str(key)] = q.get_list(key) + [primitive_value_to_str(value)]
        return q

    def remove(self, key: str) -> QueryParams:
        """
        Return a new QueryParams instance, removing the value of a key.

        Usage:

        q = httpx2.QueryParams("a=123")
        q = q.remove("a")
        assert q == httpx2.QueryParams("")
        """
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict.pop(str(key), None)
        return q

    def merge(self, params: QueryParamTypes | None = None) -> QueryParams:
        """
        Return a new QueryParams instance, updated with.

        Usage:

        q = httpx2.QueryParams("a=123")
        q = q.merge({"b": "456"})
        assert q == httpx2.QueryParams("a=123&b=456")

        q = httpx2.QueryParams("a=123")
        q = q.merge({"a": "456", "b": "789"})
        assert q == httpx2.QueryParams("a=456&b=789")
        """
        q = QueryParams(params)
        q._dict = {**self._dict, **q._dict}
        return q

    def __getitem__(self, key: typing.Any) -> str:
        return self._dict[key][0]

    def __contains__(self, key: typing.Any) -> bool:
        return key in self._dict

    def __iter__(self) -> typing.Iterator[typing.Any]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._dict)

    def __bool__(self) -> bool:
        return bool(self._dict)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return sorted(self.multi_items()) == sorted(other.multi_items())

    def __str__(self) -> str:
        return urlencode(self.multi_items())

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        query_string = str(self)
        return f"{class_name}({query_string!r})"

    def update(self, params: QueryParamTypes | None = None) -> None:
        raise RuntimeError("QueryParams are immutable since 0.18.0. Use `q = q.merge(...)` to create an updated copy.")

    def __setitem__(self, key: str, value: str) -> None:
        raise RuntimeError(
            "QueryParams are immutable since 0.18.0. Use `q = q.set(key, value)` to create an updated copy."
        )
<|endoftext|>"""
Our exception hierarchy:

* HTTPError
  x RequestError
    + TransportError
      - TimeoutException
        · ConnectTimeout
        · ReadTimeout
        · WriteTimeout
        · PoolTimeout
      - NetworkError
        · ConnectError
        · ReadError
        · WriteError
        · CloseError
      - ProtocolError
        · LocalProtocolError
        · RemoteProtocolError
      - ProxyError
      - UnsupportedProtocol
    + DecodingError
    + TooManyRedirects
  x HTTPStatusError
* InvalidURL
* CookieConflict
* StreamError
  x StreamConsumed
  x StreamClosed
  x ResponseNotRead
  x RequestNotRead
"""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Generator

if typing.TYPE_CHECKING:
    from ._models import Request, Response  # pragma: no cover

__all__ = [
    "CloseError",
    "ConnectError",
    "ConnectTimeout",
    "CookieConflict",
    "DecodingError",
    "HTTPError",
    "HTTPStatusError",
    "InvalidURL",
    "LocalProtocolError",
    "NetworkError",
    "PoolTimeout",
    "ProtocolError",
    "ProxyError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "RequestError",
    "RequestNotRead",
    "ResponseNotRead",
    "StreamClosed",
    "StreamConsumed",
    "StreamError",
    "TimeoutException",
    "TooManyRedirects",
    "TransportError",
    "UnsupportedProtocol",
    "WriteError",
    "WriteTimeout",
]


class HTTPXDeprecationWarning(UserWarning):
    """A custom deprecation warning for HTTPX.

    Unlike the built-in `DeprecationWarning`, this inherits from `UserWarning` to ensure it is visible by default,
    helping users discover deprecated features without needing to enable warnings explicitly.

    Reference: https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries
    """


class HTTPError(Exception):
    """
    Base class for `RequestError` and `HTTPStatusError`.

    Useful for `try...except` blocks when issuing a request,
    and then calling `.raise_for_status()`.

    For example:

    ```
    try:
        response = httpx2.get("https://www.example.com")
        response.raise_for_status()
    except httpx2.HTTPError as exc:
        print(f"HTTP Exception for {exc.request.url} - {exc}")
    ```
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._request: Request | None = None

    @property
    def request(self) -> Request:
        if self._request is None:
            raise RuntimeError("The .request property has not been set.")
        return self._request

    @request.setter
    def request(self, request: Request) -> None:
        self._request = request


class RequestError(HTTPError):
    """
    Base class for all exceptions that may occur when issuing a `.request()`.
    """

    def __init__(self, message: str, *, request: Request | None = None) -> None:
        super().__init__(message)
        # At the point an exception is raised we won't typically have a request
        # instance to associate it with.
        #
        # The 'request_context' context manager is used within the Client and
        # Response methods in order to ensure that any raised exceptions
        # have a `.request` property set on them.
        self._request = request


class TransportError(RequestError):
    """
    Base class for all exceptions that occur at the level of the Transport API.
    """


# Timeout exceptions...


class TimeoutException(TransportError):
    """
    The base class for timeout errors.

    An operation has timed out.
    """


class ConnectTimeout(TimeoutException):
    """
    Timed out while connecting to the host.
    """


class ReadTimeout(TimeoutException):
    """
    Timed out while receiving data from the host.
    """


class WriteTimeout(TimeoutException):
    """
    Timed out while sending data to the host.
    """


class PoolTimeout(TimeoutException):
    """
    Timed out waiting to acquire a connection from the pool.
    """


# Core networking exceptions...


class NetworkError(TransportError):
    """
    The base class for network-related errors.

    An error occurred while interacting with the network.
    """


class ReadError(NetworkError):
    """
    Failed to receive data from the network.
    """


class WriteError(NetworkError):
    """
    Failed to send data through the network.
    """


class ConnectError(NetworkError):
    """
    Failed to establish a connection.
    """


class CloseError(NetworkError):
    """
    Failed to close a connection.
    """


# Other transport exceptions...


class ProxyError(TransportError):
    """
    An error occurred while establishing a proxy connection.
    """


class UnsupportedProtocol(TransportError):
    """
    Attempted to make a request to an unsupported protocol.

    For example issuing a request to `ftp://www.example.com`.
    """


class ProtocolError(TransportError):
    """
    The protocol was violated.
    """


class LocalProtocolError(ProtocolError):
    """
    A protocol was violated by the client.

    For example if the user instantiated a `Request` instance explicitly,
    failed to include the mandatory `Host:` header, and then issued it directly
    using `client.send()`.
    """


class RemoteProtocolError(ProtocolError):
    """
    The protocol was violated by the server.

    For example, returning malformed HTTP.
    """


# Other request exceptions...


class DecodingError(RequestError):
    """
    Decoding of the response failed, due to a malformed encoding.
    """


class TooManyRedirects(RequestError):
    """
    Too many redirects.
    """


# Client errors


class HTTPStatusError(HTTPError):
    """
    The response had an error HTTP status of 4xx or 5xx.

    May be raised when calling `response.raise_for_status()`
    """

    def __init__(self, message: str, *, request: Request, response: Response) -> None:
        super().__init__(message)
        self.request = request
        self.response = response


class InvalidURL(Exception):
    """
    URL is improperly formed or cannot be parsed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class CookieConflict(Exception):
    """
    Attempted to lookup a cookie by name, but multiple cookies existed.

    Can occur when calling `response.cookies.get(...)`.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# Stream exceptions...

# These may occur as the result of a programming error, by accessing
# the request/response stream in an invalid manner.


class StreamError(RuntimeError):
    """
    The base class for stream exceptions.

    The developer made an error in accessing the request stream in
    an invalid way.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class StreamConsumed(StreamError):
    """
    Attempted to read or stream content, but the content has already
    been streamed.
    """

    def __init__(self) -> None:
        message = (
            "Attempted to read or stream some content, but the content has "
            "already been streamed. For requests, this could be due to passing "
            "a generator as request content, and then receiving a redirect "
            "response or a secondary request as part of an authentication flow."
            "For responses, this could be due to attempting to stream the response "
            "content more than once."
        )
        super().__init__(message)


class StreamClosed(StreamError):
    """
    Attempted to read or stream response content, but the request has been
    closed.
    """

    def __init__(self) -> None:
        message = "Attempted to read or stream content, but the stream has been closed."
        super().__init__(message)


class ResponseNotRead(StreamError):
    """
    Attempted to access streaming response content, without having called `read()`.
    """

    def __init__(self) -> None:
        message = "Attempted to access streaming response content, without having called `read()`."
        super().__init__(message)


class RequestNotRead(StreamError):
    """
    Attempted to access streaming request content, without having called `read()`.
    """

    def __init__(self) -> None:
        message = "Attempted to access streaming request content, without having called `read()`."
        super().__init__(message)


@contextlib.contextmanager
def request_context(request: Request | None = None) -> Generator[None]:
    """
    A context manager that can be used to attach the given request context
    to any `RequestError` exceptions that are raised within the block.
    """
    try:
        yield
    except RequestError as exc:
        if request is not None:
            exc.request = request
        raise exc
<|endoftext|>"""
An implementation of `urlparse` that provides URL validation and normalization
as described by RFC3986.

We rely on this implementation rather than the one in Python's stdlib, because:

* It provides more complete URL validation.
* It properly differentiates between an empty querystring and an absent querystring,
  to distinguish URLs with a trailing '?'.
* It handles scheme, hostname, port, and path normalization.
* It supports IDNA hostnames, normalizing them to their encoded form.
* The API supports passing individual components, as well as the complete URL string.

Previously we relied on the excellent `rfc3986` package to handle URL parsing and
validation, but this module provides a simpler alternative, with less indirection
required.
"""

from __future__ import annotations

import ipaddress
import re
import typing

import idna

from ._exceptions import InvalidURL

MAX_URL_LENGTH = 65536

# https://datatracker.ietf.org/doc/html/rfc3986.html#section-2.3
UNRESERVED_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
SUB_DELIMS = "!$&'()*+,;="

PERCENT_ENCODED_REGEX = re.compile("%[A-Fa-f0-9]{2}")

# https://url.spec.whatwg.org/#percent-encoded-bytes

# The fragment percent-encode set is the C0 control percent-encode set
# and U+0020 SPACE, U+0022 ("), U+003C (<), U+003E (>), and U+0060 (`).
FRAG_SAFE = "".join([chr(i) for i in range(0x20, 0x7F) if i not in (0x20, 0x22, 0x3C, 0x3E, 0x60)])

# The query percent-encode set is the C0 control percent-encode set
# and U+0020 SPACE, U+0022 ("), U+0023 (#), U+003C (<), and U+003E (>).
QUERY_SAFE = "".join([chr(i) for i in range(0x20, 0x7F) if i not in (0x20, 0x22, 0x23, 0x3C, 0x3E)])

# The path percent-encode set is the query percent-encode set
# and U+003F (?), U+0060 (`), U+007B ({), and U+007D (}).
PATH_SAFE = "".join(
    [chr(i) for i in range(0x20, 0x7F) if i not in (0x20, 0x22, 0x23, 0x3C, 0x3E) + (0x3F, 0x60, 0x7B, 0x7D)]
)

# The userinfo percent-encode set is the path percent-encode set
# and U+002F (/), U+003A (:), U+003B (;), U+003D (=), U+0040 (@),
# U+005B ([) to U+005E (^), inclusive, and U+007C (|).
USERNAME_SAFE = "".join(
    [
        chr(i)
        for i in range(0x20, 0x7F)
        if i
        not in (0x20, 0x22, 0x23, 0x3C, 0x3E)
        + (0x3F, 0x60, 0x7B, 0x7D)
        + (0x2F, 0x3A, 0x3B, 0x3D, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x7C)
    ]
)
PASSWORD_SAFE = "".join(
    [
        chr(i)
        for i in range(0x20, 0x7F)
        if i
        not in (0x20, 0x22, 0x23, 0x3C, 0x3E)
        + (0x3F, 0x60, 0x7B, 0x7D)
        + (0x2F, 0x3A, 0x3B, 0x3D, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x7C)
    ]
)
# Note... The terminology 'userinfo' percent-encode set in the WHATWG document
# is used for the username and password quoting. For the joint userinfo component
# we remove U+003A (:) from the safe set.
USERINFO_SAFE = "".join(
    [
        chr(i)
        for i in range(0x20, 0x7F)
        if i
        not in (0x20, 0x22, 0x23, 0x3C, 0x3E)
        + (0x3F, 0x60, 0x7B, 0x7D)
        + (0x2F, 0x3B, 0x3D, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x7C)
    ]
)


# {scheme}:      (optional)
# //{authority}  (optional)
# {path}
# ?{query}       (optional)
# #{fragment}    (optional)
URL_REGEX = re.compile(
    (
        r"(?:(?P<scheme>{scheme}):)?"
        r"(?://(?P<authority>{authority}))?"
        r"(?P<path>{path})"
        r"(?:\?(?P<query>{query}))?"
        r"(?:#(?P<fragment>{fragment}))?"
    ).format(
        scheme="([a-zA-Z][a-zA-Z0-9+.-]*)?",
        authority="[^/?#]*",
        path="[^?#]*",
        query="[^#]*",
        fragment=".*",
    )
)

# {userinfo}@    (optional)
# {host}
# :{port}        (optional)
AUTHORITY_REGEX = re.compile(
    (r"(?:(?P<userinfo>{userinfo})@)?" r"(?P<host>{host})" r":?(?P<port>{port})?").format(
        userinfo=".*",  # Any character sequence.
        host="(\\[.*\\]|[^:@]*)",  # Either any character sequence excluding ':' or '@',
        # or an IPv6 address enclosed within square brackets.
        port=".*",  # Any character sequence.
    )
)


# If we call urlparse with an individual component, then we need to regex
# validate that component individually.
# Note that we're duplicating the same strings as above. Shock! Horror!!
COMPONENT_REGEX = {
    "scheme": re.compile("([a-zA-Z][a-zA-Z0-9+.-]*)?"),
    "authority": re.compile("[^/?#]*"),
    "path": re.compile("[^?#]*"),
    "query": re.compile("[^#]*"),
    "fragment": re.compile(".*"),
    "userinfo": re.compile("[^@]*"),
    "host": re.compile("(\\[.*\\]|[^:]*)"),
    "port": re.compile(".*"),
}


# We use these simple regexs as a first pass before handing off to
# the stdlib 'ipaddress' module for IP address validation.
IPv4_STYLE_HOSTNAME = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
IPv6_STYLE_HOSTNAME = re.compile(r"^\[.*\]$")


class ParseResult(typing.NamedTuple):
    scheme: str
    userinfo: str
    host: str
    port: int | None
    path: str
    query: str | None
    fragment: str | None

    @property
    def authority(self) -> str:
        return "".join(
            [
                f"{self.userinfo}@" if self.userinfo else "",
                f"[{self.host}]" if ":" in self.host else self.host,
                f":{self.port}" if self.port is not None else "",
            ]
        )

    @property
    def netloc(self) -> str:
        return "".join(
            [
                f"[{self.host}]" if ":" in self.host else self.host,
                f":{self.port}" if self.port is not None else "",
            ]
        )

    def copy_with(self, **kwargs: str | None) -> ParseResult:
        if not kwargs:
            return self

        defaults = {
            "scheme": self.scheme,
            "authority": self.authority,
            "path": self.path,
            "query": self.query,
            "fragment": self.fragment,
        }
        defaults.update(kwargs)
        return urlparse("", **defaults)

    def __str__(self) -> str:
        authority = self.authority
        return "".join(
            [
                f"{self.scheme}:" if self.scheme else "",
                f"//{authority}" if authority else "",
                self.path,
                f"?{self.query}" if self.query is not None else "",
                f"#{self.fragment}" if self.fragment is not None else "",
            ]
        )


def urlparse(url: str = "", **kwargs: str | None) -> ParseResult:
    # Initial basic checks on allowable URLs.
    # ---------------------------------------

    # Hard limit the maximum allowable URL length.
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURL("URL too long")

    # If a URL includes any ASCII control characters including \t, \r, \n,
    # then treat it as invalid.
    if any(char.isascii() and not char.isprintable() for char in url):
        char = next(char for char in url if char.isascii() and not char.isprintable())
        idx = url.find(char)
        error = f"Invalid non-printable ASCII character in URL, {char!r} at position {idx}."
        raise InvalidURL(error)

    # Some keyword arguments require special handling.
    # ------------------------------------------------

    # Coerce "port" to a string, if it is provided as an integer.
    if "port" in kwargs:
        port = kwargs["port"]
        kwargs["port"] = str(port) if isinstance(port, int) else port

    # Replace "netloc" with "host and "port".
    if "netloc" in kwargs:
        netloc = kwargs.pop("netloc") or ""
        kwargs["host"], _, kwargs["port"] = netloc.partition(":")

    # Replace "username" and/or "password" with "userinfo".
    if "username" in kwargs or "password" in kwargs:
        username = quote(kwargs.pop("username", "") or "", safe=USERNAME_SAFE)
        password = quote(kwargs.pop("password", "") or "", safe=PASSWORD_SAFE)
        kwargs["userinfo"] = f"{username}:{password}" if password else username

    # Replace "raw_path" with "path" and "query".
    if "raw_path" in kwargs:
        raw_path = kwargs.pop("raw_path") or ""
        kwargs["path"], separator, kwargs["query"] = raw_path.partition("?")
        if not separator:
            kwargs["query"] = None

    # Ensure that IPv6 "host" addresses are always escaped with "[...]".
    if "host" in kwargs:
        host = kwargs.get("host") or ""
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            kwargs["host"] = f"[{host}]"

    # If any keyword arguments are provided, ensure they are valid.
    # -------------------------------------------------------------

    for key, value in kwargs.items():
        if value is not None:
            if len(value) > MAX_URL_LENGTH:
                raise InvalidURL(f"URL component '{key}' too long")

            # If a component includes any ASCII control characters including \t, \r, \n,
            # then treat it as invalid.
            if any(char.isascii() and not char.isprintable() for char in value):
                char = next(char for char in value if char.isascii() and not char.isprintable())
                idx = value.find(char)
                error = f"Invalid non-printable ASCII character in URL {key} component, {char!r} at position {idx}."
                raise InvalidURL(error)

            # Ensure that keyword arguments match as a valid regex.
            if not COMPONENT_REGEX[key].fullmatch(value):
                raise InvalidURL(f"Invalid URL component '{key}'")

    # The URL_REGEX will always match, but may have empty components.
    url_match = URL_REGEX.match(url)
    assert url_match is not None
    url_dict = url_match.groupdict()

    # * 'scheme', 'authority', and 'path' may be empty strings.
    # * 'query' may be 'None', indicating no trailing "?" portion.
    #   Any string including the empty string, indicates a trailing "?".
    # * 'fragment' may be 'None', indicating no trailing "#" portion.
    #   Any string including the empty string, indicates a trailing "#".
    scheme = kwargs.get("scheme", url_dict["scheme"]) or ""
    authority = kwargs.get("authority", url_dict["authority"]) or ""
    path = kwargs.get("path", url_dict["path"]) or ""
    query = kwargs.get("query", url_dict["query"])
    frag = kwargs.get("fragment", url_dict["fragment"])

    # The AUTHORITY_REGEX will always match, but may have empty components.
    authority_match = AUTHORITY_REGEX.match(authority)
    assert authority_match is not None
    authority_dict = authority_match.groupdict()

    # * 'userinfo' and 'host' may be empty strings.
    # * 'port' may be 'None'.
    userinfo = kwargs.get("userinfo", authority_dict["userinfo"]) or ""
    host = kwargs.get("host", authority_dict["host"]) or ""
    port = kwargs.get("port", authority_dict["port"])

    # Normalize and validate each component.
    # We end up with a parsed representation of the URL,
    # with components that are plain ASCII bytestrings.
    parsed_scheme: str = scheme.lower()
    parsed_userinfo: str = quote(userinfo, safe=USERINFO_SAFE)
    parsed_host: str = encode_host(host)
    parsed_port: int | None = normalize_port(port, scheme)

    has_scheme = parsed_scheme != ""
    has_authority = parsed_userinfo != "" or parsed_host != "" or parsed_port is not None
    validate_path(path, has_scheme=has_scheme, has_authority=has_authority)
    if has_scheme or has_authority:
        path = normalize_path(path)

    parsed_path: str = quote(path, safe=PATH_SAFE)
    parsed_query: str | None = None if query is None else quote(query, safe=QUERY_SAFE)
    parsed_frag: str | None = None if frag is None else quote(frag, safe=FRAG_SAFE)

    # The parsed ASCII bytestrings are our canonical form.
    # All properties of the URL are derived from these.
    return ParseResult(
        parsed_scheme,
        parsed_userinfo,
        parsed_host,
        parsed_port,
        parsed_path,
        parsed_query,
        parsed_frag,
    )


def encode_host(host: str) -> str:
    if not host:
        return ""

    elif IPv4_STYLE_HOSTNAME.match(host):
        # Validate IPv4 hostnames like #.#.#.#
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # IPv4address = dec-octet "." dec-octet "." dec-octet "." dec-octet
        try:
            ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError:
            raise InvalidURL(f"Invalid IPv4 address: {host!r}")
        return host

    elif IPv6_STYLE_HOSTNAME.match(host):
        # Validate IPv6 hostnames like [...]
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # "A host identified by an Internet Protocol literal address, version 6
        # [RFC3513] or later, is distinguished by enclosing the IP literal
        # within square brackets ("[" and "]").  This is the only place where
        # square bracket characters are allowed in the URI syntax."
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ipaddress.AddressValueError:
            raise InvalidURL(f"Invalid IPv6 address: {host!r}")
        return host[1:-1]

    elif host.isascii():
        # Regular ASCII hostnames
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # reg-name    = *( unreserved / pct-encoded / sub-delims )
        WHATWG_SAFE = '"`{}%|\\'
        return quote(host.lower(), safe=SUB_DELIMS + WHATWG_SAFE)

    # IDNA hostnames
    try:
        return idna.encode(host.lower()).decode("ascii")
    except idna.IDNAError:
        raise InvalidURL(f"Invalid IDNA hostname: {host!r}")


def normalize_port(port: str | int | None, scheme: str) -> int | None:
    # From https://tools.ietf.org/html/rfc3986#section-3.2.3
    #
    # "A scheme may define a default port.  For example, the "http" scheme
    # defines a default port of "80", corresponding to its reserved TCP
    # port number.  The type of port designated by the port number (e.g.,
    # TCP, UDP, SCTP) is defined by the URI scheme.  URI producers and
    # normalizers should omit the port component and its ":" delimiter if
    # port is empty or if its value would be the same as that of the
    # scheme's default."
    if port is None or port == "":
        return None

    try:
        port_as_int = int(port)
    except ValueError:
        raise InvalidURL(f"Invalid port: {port!r}")

    # See https://url.spec.whatwg.org/#url-miscellaneous
    default_port = {"ftp": 21, "http": 80, "https": 443, "ws": 80, "wss": 443}.get(scheme)
    if port_as_int == default_port:
        return None
    return port_as_int


def validate_path(path: str, has_scheme: bool, has_authority: bool) -> None:
    """
    Path validation rules that depend on if the URL contains
    a scheme or authority component.

    See https://datatracker.ietf.org/doc/html/rfc3986.html#section-3.3
    """
    if has_authority:
        # If a URI contains an authority component, then the path component
        # must either be empty or begin with a slash ("/") character."
        if path and not path.startswith("/"):
            raise InvalidURL("For absolute URLs, path must be empty or begin with '/'")

    if not has_scheme and not has_authority:
        # If a URI does not contain an authority component, then the path cannot begin
        # with two slash characters ("//").
        if path.startswith("//"):
            raise InvalidURL("Relative URLs cannot have a path starting with '//'")

        # In addition, a URI reference (Section 4.1) may be a relative-path reference,
        # in which case the first path segment cannot contain a colon (":") character.
        if path.startswith(":"):
            raise InvalidURL("Relative URLs cannot have a path starting with ':'")


def normalize_path(path: str) -> str:
    """
    Drop "." and ".." segments from a URL path.

    For example:

        normalize_path("/path/./to/somewhere/..") == "/path/to"
    """
    # Fast return when no '.' characters in the path.
    if "." not in path:
        return path

    components = path.split("/")

    # Fast return when no '.' or '..' components in the path.
    if "." not in components and ".." not in components:
        return path

    # https://datatracker.ietf.org/doc/html/rfc3986#section-5.2.4
    output: list[str] = []
    for component in components:
        if component == ".":
            pass
        elif component == "..":
            if output and output != [""]:
                output.pop()
        else:
            output.append(component)
    return "/".join(output)


def PERCENT(string: str) -> str:
    return "".join([f"%{byte:02X}" for byte in string.encode("utf-8")])


def percent_encoded(string: str, safe: str) -> str:
    """
    Use percent-encoding to quote a string.
    """
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe

    # Fast path for strings that don't need escaping.
    if not string.rstrip(NON_ESCAPED_CHARS):
        return string

    return "".join([char if char in NON_ESCAPED_CHARS else PERCENT(char) for char in string])


def quote(string: str, safe: str) -> str:
    """
    Use percent-encoding to quote a string, omitting existing '%xx' escape sequences.

    See: https://www.rfc-editor.org/rfc/rfc3986#section-2.1

    * `string`: The string to be percent-escaped.
    * `safe`: A string containing characters that may be treated as safe, and do not
        need to be escaped. Unreserved characters are always treated as safe.
        See: https://www.rfc-editor.org/rfc/rfc3986#section-2.3
    """
    parts: list[str] = []
    current_position = 0
    for match in re.finditer(PERCENT_ENCODED_REGEX, string):
        start_position, end_position = match.start(), match.end()
        matched_text = match.group(0)
        # Add any text up to the '%xx' escape sequence.
        if start_position != current_position:
            leading_text = string[current_position:start_position]
            parts.append(percent_encoded(leading_text, safe=safe))

        # Add the '%xx' escape sequence.
        parts.append(matched_text)
        current_position = end_position

    # Add any text after the final '%xx' escape sequence.
    if current_position != len(string):
        trailing_text = string[current_position:]
        parts.append(percent_encoded(trailing_text, safe=safe))

    return "".join(parts)
<|endoftext|>from __future__ import annotations

import ipaddress
import os
import re
import typing
from urllib.request import getproxies

from ._types import PrimitiveData

if typing.TYPE_CHECKING:
    from ._urls import URL


def primitive_value_to_str(value: PrimitiveData) -> str:
    """
    Coerce a primitive data type into a string value.

    Note that we prefer JSON-style 'true'/'false' for boolean values here.
    """
    if value is True:
        return "true"
    elif value is False:
        return "false"
    elif value is None:
        return ""
    return str(value)


def get_environment_proxies() -> dict[str, str | None]:
    """Gets proxy information from the environment"""

    # urllib.request.getproxies() falls back on System
    # Registry and Config for proxies on Windows and macOS.
    # We don't want to propagate non-HTTP proxies into
    # our configuration such as 'TRAVIS_APT_PROXY'.
    proxy_info = getproxies()
    mounts: dict[str, str | None] = {}

    for scheme in ("http", "https", "all"):
        if proxy_info.get(scheme):
            hostname = proxy_info[scheme]
            mounts[f"{scheme}://"] = hostname if "://" in hostname else f"http://{hostname}"

    no_proxy_hosts = [host.strip() for host in proxy_info.get("no", "").split(",")]
    for hostname in no_proxy_hosts:
        # See https://curl.haxx.se/libcurl/c/CURLOPT_NOPROXY.html for details
        # on how names in `NO_PROXY` are handled.
        if hostname == "*":
            # If NO_PROXY=* is used or if "*" occurs as any one of the comma
            # separated hostnames, then we should just bypass any information
            # from HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, and always ignore
            # proxies.
            return {}
        elif hostname:
            # NO_PROXY=.google.com is marked as "all://*.google.com,
            #   which disables "www.google.com" but not "google.com"
            # NO_PROXY=google.com is marked as "all://*google.com,
            #   which disables "www.google.com" and "google.com".
            #   (But not "wwwgoogle.com")
            # NO_PROXY can include domains, IPv6, IPv4 addresses and "localhost"
            #   NO_PROXY=example.com,::1,localhost,192.168.0.0/16
            if "://" in hostname:
                mounts[hostname] = None
            elif is_ipv4_hostname(hostname):
                mounts[f"all://{hostname}"] = None
            elif is_ipv6_hostname(hostname):
                if "/" in hostname:
                    addr, _, subnet = hostname.partition("/")
                    mounts[f"all://[{addr}]/{subnet}"] = None
                else:
                    mounts[f"all://[{hostname}]"] = None
            elif hostname.lower() == "localhost":
                mounts[f"all://{hostname}"] = None
            else:
                mounts[f"all://*{hostname}"] = None

    return mounts


def to_bytes(value: str | bytes, encoding: str = "utf-8") -> bytes:
    return value.encode(encoding) if isinstance(value, str) else value


def to_str(value: str | bytes, encoding: str = "utf-8") -> str:
    return value if isinstance(value, str) else value.decode(encoding)


def to_bytes_or_str(value: str, match_type_of: typing.AnyStr) -> typing.AnyStr:
    return value if isinstance(match_type_of, str) else value.encode()


def peek_filelike_length(stream: typing.Any) -> int | None:
    """
    Given a file-like stream object, return its length in number of bytes
    without reading it into memory.
    """
    try:
        # Is it an actual file?
        fd = stream.fileno()
        # Yup, seems to be an actual file.
        length = os.fstat(fd).st_size
    except (AttributeError, OSError):
        # No... Maybe it's something that supports random access, like `io.BytesIO`?
        try:
            # Assuming so, go to end of stream to figure out its length,
            # then put it back in place.
            offset = stream.tell()
            length = stream.seek(0, os.SEEK_END)
            stream.seek(offset)
        except (AttributeError, OSError):
            # Not even that? Sorry, we're doomed...
            return None

    return length


def peek_filelike_remaining_length(stream: typing.Any) -> int | None:
    """Return the remaining byte length without changing the stream position."""
    try:
        offset: int = stream.tell()
        length: int = stream.seek(0, os.SEEK_END)
        stream.seek(offset)
    except (AttributeError, OSError):
        return None

    return max(0, length - offset)


class URLPattern:
    """
    A utility class currently used for making lookups against proxy keys...

    # Wildcard matching...
    >>> pattern = URLPattern("all://")
    >>> pattern.matches(httpx2.URL("http://example.com"))
    True

    # Witch scheme matching...
    >>> pattern = URLPattern("https://")
    >>> pattern.matches(httpx2.URL("https://example.com"))
    True
    >>> pattern.matches(httpx2.URL("http://example.com"))
    False

    # With domain matching...
    >>> pattern = URLPattern("https://example.com")
    >>> pattern.matches(httpx2.URL("https://example.com"))
    True
    >>> pattern.matches(httpx2.URL("http://example.com"))
    False
    >>> pattern.matches(httpx2.URL("https://other.com"))
    False

    # Wildcard scheme, with domain matching...
    >>> pattern = URLPattern("all://example.com")
    >>> pattern.matches(httpx2.URL("https://example.com"))
    True
    >>> pattern.matches(httpx2.URL("http://example.com"))
    True
    >>> pattern.matches(httpx2.URL("https://other.com"))
    False

    # With port matching...
    >>> pattern = URLPattern("https://example.com:1234")
    >>> pattern.matches(httpx2.URL("https://example.com:1234"))
    True
    >>> pattern.matches(httpx2.URL("https://example.com"))
    False
    """

    def __init__(self, pattern: str) -> None:
        from ._urls import URL

        if pattern and ":" not in pattern:
            raise ValueError(
                f"Proxy keys should use proper URL forms rather "
                f"than plain scheme strings. "
                f'Instead of "{pattern}", use "{pattern}://"'
            )

        url = URL(pattern)
        self.pattern = pattern
        self.scheme = "" if url.scheme == "all" else url.scheme
        self.host = "" if url.host == "*" else url.host
        self.port = url.port
        if not url.host or url.host == "*":
            self.host_regex: typing.Pattern[str] | None = None
        elif url.host.startswith("*."):
            # *.example.com should match "www.example.com", but not "example.com"
            domain = re.escape(url.host[2:])
            self.host_regex = re.compile(f"^.+\\.{domain}$")
        elif url.host.startswith("*"):
            # *example.com should match "www.example.com" and "example.com"
            domain = re.escape(url.host[1:])
            self.host_regex = re.compile(f"^(.+\\.)?{domain}$")
        else:
            # example.com should match "example.com" but not "www.example.com"
            domain = re.escape(url.host)
            self.host_regex = re.compile(f"^{domain}$")

    def matches(self, other: URL) -> bool:
        if self.scheme and self.scheme != other.scheme:
            return False
        if self.host and self.host_regex is not None and not self.host_regex.match(other.host):
            return False
        if self.port is not None and self.port != other.port:
            return False
        return True

    @property
    def priority(self) -> tuple[int, int, int]:
        """
        The priority allows URLPattern instances to be sortable, so that
        we can match from most specific to least specific.
        """
        # URLs with a port should take priority over URLs without a port.
        port_priority = 0 if self.port is not None else 1
        # Longer hostnames should match first.
        host_priority = -len(self.host)
        # Longer schemes should match first.
        scheme_priority = -len(self.scheme)
        return (port_priority, host_priority, scheme_priority)

    def __hash__(self) -> int:
        return hash(self.pattern)

    def __lt__(self, other: URLPattern) -> bool:
        return self.priority < other.priority

    def __eq__(self, other: typing.Any) -> bool:
        return isinstance(other, URLPattern) and self.pattern == other.pattern


def is_ipv4_hostname(hostname: str) -> bool:
    try:
        ipaddress.IPv4Address(hostname.split("/")[0])
    except Exception:
        return False
    return True


def is_ipv6_hostname(hostname: str) -> bool:
    try:
        ipaddress.IPv6Address(hostname.split("/")[0])
    except Exception:
        return False
    return True
<|endoftext|>"""
Server-sent events support, derived from httpx-sse (https://github.com/florimondmanca/httpx-sse).

```
MIT License

Copyright (c) 2022 Florimond Manca

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

from __future__ import annotations

import json as jsonlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from ._config import DEFAULT_MAX_EVENT_SIZE_BYTES
from ._exceptions import TransportError, request_context
from ._models import Response

__all__ = ["EventSource", "SSEError", "ServerSentEvent"]


class SSEError(TransportError):
    """
    An error that occurred while connecting to a server-sent events endpoint.
    """


@dataclass(frozen=True)
class ServerSentEvent:
    event: str = "message"
    data: str = ""
    id: str = ""
    retry: int | None = None

    def json(self) -> object:
        return jsonlib.loads(self.data)


class _SSEEventDecoder:
    def __init__(self, max_event_size: int | None = None) -> None:
        self._max_event_size = max_event_size
        self._event = ""
        self._data: list[str] = []
        self._event_size = 0
        self._last_event_id = ""
        self._retry: int | None = None
        self._pending = False

    def decode(self, line: str) -> ServerSentEvent | None:
        if not line:
            self._event_size = 0
            if not self._pending:
                return None

            sse = ServerSentEvent(
                event=self._event or "message",
                data="\n".join(self._data),
                id=self._last_event_id,
                retry=self._retry,
            )
            self._event = ""
            self._data = []
            self._retry = None
            self._pending = False
            return sse

        if line.startswith(":"):
            return None

        self._event_size += len(line.encode("utf-8"))
        self._check_size()

        fieldname, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value

        if fieldname == "event":
            self._event = value
            self._pending = True
        elif fieldname == "data":
            self._data.append(value)
            self._pending = True
        elif fieldname == "id":
            if "\0" not in value:
                self._last_event_id = value
                self._pending = True
        elif fieldname == "retry":
            try:
                self._retry = int(value)
                self._pending = True
            except ValueError:
                pass

        return None

    def check_pending(self, pending_size: int) -> None:
        """
        Bound the total bytes buffered for the in-progress event, including
        a trailing line that has not been terminated by a newline yet.
        """
        self._check_size(pending_size)

    def _check_size(self, pending_size: int = 0) -> None:
        if self._max_event_size is not None and self._event_size + pending_size > self._max_event_size:
            raise SSEError(f"Server-sent event exceeded the {self._max_event_size} byte limit.")


class _SSELineDecoder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._pending_size = 0
        self._trailing_cr = False

    @property
    def pending_size(self) -> int:
        return self._pending_size

    def decode(self, text: str) -> list[str]:
        if self._trailing_cr:
            text = "\r" + text
            self._trailing_cr = False
        if text.endswith("\r"):
            self._trailing_cr = True
            text = text[:-1]

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in text:
            self._append(text)
            return []

        lines = text.split("\n")
        self._append(lines[0])
        lines[0] = self._consume_pending()
        self._append(lines.pop())
        return lines

    def flush(self) -> list[str]:
        if self._trailing_cr:
            self._append("\n")
            self._trailing_cr = False
        buffer = self._consume_pending()
        if not buffer:
            return []
        return buffer.split("\n")

    def _append(self, text: str) -> None:
        if text:
            self._parts.append(text)
            self._pending_size += len(text.encode("utf-8"))

    def _consume_pending(self) -> str:
        pending = "".join(self._parts)
        self._parts = []
        self._pending_size = 0
        return pending


class _SSEParser:
    def __init__(self, max_event_size: int | None = None) -> None:
        self._event_decoder = _SSEEventDecoder(max_event_size)
        self._line_decoder = _SSELineDecoder()

    def decode(self, text: str) -> Iterator[ServerSentEvent]:
        yield from self._decode_lines(self._line_decoder.decode(text))
        self._event_decoder.check_pending(self._line_decoder.pending_size)

    def flush(self) -> Iterator[ServerSentEvent]:
        yield from self._decode_lines(self._line_decoder.flush())

    def _decode_lines(self, lines: list[str]) -> Iterator[ServerSentEvent]:
        for line in lines:
            sse = self._event_decoder.decode(line)
            if sse is not None:
                yield sse


class EventSource:
    def __init__(self, response: Response, max_event_size: int | None = DEFAULT_MAX_EVENT_SIZE_BYTES) -> None:
        self._response = response
        self._max_event_size = max_event_size

    @property
    def response(self) -> Response:
        return self._response

    def _check_content_type(self) -> None:
        content_type, _, _ = self._response.headers.get("content-type", "").partition(";")
        if content_type.strip().lower() != "text/event-stream":
            raise SSEError(f"Expected response with content type 'text/event-stream', got {content_type.strip()!r}.")

    def __iter__(self) -> Iterator[ServerSentEvent]:
        with request_context(request=self._response.request):
            self._check_content_type()
            parser = _SSEParser(self._max_event_size)
            for chunk in self._response.iter_text():
                yield from parser.decode(chunk)
            yield from parser.flush()

    async def __aiter__(self) -> AsyncIterator[ServerSentEvent]:
        with request_context(request=self._response.request):
            self._check_content_type()
            parser = _SSEParser(self._max_event_size)
            async for chunk in self._response.aiter_text():
                for sse in parser.decode(chunk):
                    yield sse
            for sse in parser.flush():
                yield sse
<|endoftext|>from __future__ import annotations

import warnings
from enum import EnumMeta, IntEnum

__all__ = ["codes"]

# Constant names that were renamed by RFC 9110, kept as deprecated aliases
# mapping each old name to its canonical replacement.
_DEPRECATED_CODES = {
    "REQUEST_ENTITY_TOO_LARGE": "CONTENT_TOO_LARGE",
    "REQUEST_URI_TOO_LONG": "URI_TOO_LONG",
    "REQUESTED_RANGE_NOT_SATISFIABLE": "RANGE_NOT_SATISFIABLE",
    "UNPROCESSABLE_ENTITY": "UNPROCESSABLE_CONTENT",
}


class _CodesMeta(EnumMeta):
    def __getattribute__(cls, name: str) -> object:
        if name in _DEPRECATED_CODES:
            new_name = _DEPRECATED_CODES[name]
            warnings.warn(
                f"'{name}' is deprecated. Use '{new_name}' instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return super().__getattribute__(name)


class codes(IntEnum, metaclass=_CodesMeta):
    """HTTP status codes and reason phrases

    Status codes from the following RFCs are all observed:

        * RFC 9110: HTTP Semantics, obsoletes 7231 & 7238
        * RFC 6585: Additional HTTP Status Codes
        * RFC 3229: Delta encoding in HTTP
        * RFC 4918: HTTP Extensions for WebDAV, obsoletes 2518
        * RFC 5842: Binding Extensions to WebDAV
        * RFC 2295: Transparent Content Negotiation in HTTP
        * RFC 2774: An HTTP Extension Framework
        * RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)
        * RFC 2324: Hyper Text Coffee Pot Control Protocol (HTCPCP/1.0)
        * RFC 7725: An HTTP Status Code to Report Legal Obstacles
        * RFC 8297: An HTTP Status Code for Indicating Hints
        * RFC 8470: Using Early Data in HTTP
    """

    def __new__(cls, value: int, phrase: str = "") -> codes:
        obj = int.__new__(cls, value)
        obj._value_ = value

        obj.phrase = phrase  # type: ignore[attr-defined]
        return obj

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def get_reason_phrase(cls, value: int) -> str:
        try:
            return codes(value).phrase  # type: ignore
        except ValueError:
            return ""

    @classmethod
    def is_informational(cls, value: int) -> bool:
        """
        Returns `True` for 1xx status codes, `False` otherwise.
        """
        return 100 <= value <= 199

    @classmethod
    def is_success(cls, value: int) -> bool:
        """
        Returns `True` for 2xx status codes, `False` otherwise.
        """
        return 200 <= value <= 299

    @classmethod
    def is_redirect(cls, value: int) -> bool:
        """
        Returns `True` for 3xx status codes, `False` otherwise.
        """
        return 300 <= value <= 399

    @classmethod
    def is_client_error(cls, value: int) -> bool:
        """
        Returns `True` for 4xx status codes, `False` otherwise.
        """
        return 400 <= value <= 499

    @classmethod
    def is_server_error(cls, value: int) -> bool:
        """
        Returns `True` for 5xx status codes, `False` otherwise.
        """
        return 500 <= value <= 599

    @classmethod
    def is_error(cls, value: int) -> bool:
        """
        Returns `True` for 4xx or 5xx status codes, `False` otherwise.
        """
        return 400 <= value <= 599

    # informational
    CONTINUE = 100, "Continue"
    SWITCHING_PROTOCOLS = 101, "Switching Protocols"
    PROCESSING = 102, "Processing"
    EARLY_HINTS = 103, "Early Hints"

    # success
    OK = 200, "OK"
    CREATED = 201, "Created"
    ACCEPTED = 202, "Accepted"
    NON_AUTHORITATIVE_INFORMATION = 203, "Non-Authoritative Information"
    NO_CONTENT = 204, "No Content"
    RESET_CONTENT = 205, "Reset Content"
    PARTIAL_CONTENT = 206, "Partial Content"
    MULTI_STATUS = 207, "Multi-Status"
    ALREADY_REPORTED = 208, "Already Reported"
    IM_USED = 226, "IM Used"

    # redirection
    MULTIPLE_CHOICES = 300, "Multiple Choices"
    MOVED_PERMANENTLY = 301, "Moved Permanently"
    FOUND = 302, "Found"
    SEE_OTHER = 303, "See Other"
    NOT_MODIFIED = 304, "Not Modified"
    USE_PROXY = 305, "Use Proxy"
    TEMPORARY_REDIRECT = 307, "Temporary Redirect"
    PERMANENT_REDIRECT = 308, "Permanent Redirect"

    # client error
    BAD_REQUEST = 400, "Bad Request"
    UNAUTHORIZED = 401, "Unauthorized"
    PAYMENT_REQUIRED = 402, "Payment Required"
    FORBIDDEN = 403, "Forbidden"
    NOT_FOUND = 404, "Not Found"
    METHOD_NOT_ALLOWED = 405, "Method Not Allowed"
    NOT_ACCEPTABLE = 406, "Not Acceptable"
    PROXY_AUTHENTICATION_REQUIRED = 407, "Proxy Authentication Required"
    REQUEST_TIMEOUT = 408, "Request Timeout"
    CONFLICT = 409, "Conflict"
    GONE = 410, "Gone"
    LENGTH_REQUIRED = 411, "Length Required"
    PRECONDITION_FAILED = 412, "Precondition Failed"
    CONTENT_TOO_LARGE = 413, "Content Too Large"
    URI_TOO_LONG = 414, "URI Too Long"
    UNSUPPORTED_MEDIA_TYPE = 415, "Unsupported Media Type"
    RANGE_NOT_SATISFIABLE = 416, "Range Not Satisfiable"
    EXPECTATION_FAILED = 417, "Expectation Failed"
    IM_A_TEAPOT = 418, "I'm a teapot"
    MISDIRECTED_REQUEST = 421, "Misdirected Request"
    UNPROCESSABLE_CONTENT = 422, "Unprocessable Content"
    LOCKED = 423, "Locked"
    FAILED_DEPENDENCY = 424, "Failed Dependency"
    TOO_EARLY = 425, "Too Early"
    UPGRADE_REQUIRED = 426, "Upgrade Required"
    PRECONDITION_REQUIRED = 428, "Precondition Required"
    TOO_MANY_REQUESTS = 429, "Too Many Requests"
    REQUEST_HEADER_FIELDS_TOO_LARGE = 431, "Request Header Fields Too Large"
    UNAVAILABLE_FOR_LEGAL_REASONS = 451, "Unavailable For Legal Reasons"

    # server errors
    INTERNAL_SERVER_ERROR = 500, "Internal Server Error"
    NOT_IMPLEMENTED = 501, "Not Implemented"
    BAD_GATEWAY = 502, "Bad Gateway"
    SERVICE_UNAVAILABLE = 503, "Service Unavailable"
    GATEWAY_TIMEOUT = 504, "Gateway Timeout"
    HTTP_VERSION_NOT_SUPPORTED = 505, "HTTP Version Not Supported"
    VARIANT_ALSO_NEGOTIATES = 506, "Variant Also Negotiates"
    INSUFFICIENT_STORAGE = 507, "Insufficient Storage"
    LOOP_DETECTED = 508, "Loop Detected"
    NOT_EXTENDED = 510, "Not Extended"
    NETWORK_AUTHENTICATION_REQUIRED = 511, "Network Authentication Required"

    # Deprecated aliases for names replaced by RFC 9110.
    REQUEST_ENTITY_TOO_LARGE = CONTENT_TOO_LARGE
    REQUEST_URI_TOO_LONG = URI_TOO_LONG
    REQUESTED_RANGE_NOT_SATISFIABLE = RANGE_NOT_SATISFIABLE
    UNPROCESSABLE_ENTITY = UNPROCESSABLE_CONTENT


# Include lower-case styles for `requests` compatibility.
for name, code in codes.__members__.items():
    setattr(codes, name.lower(), int(code))
<|endoftext|>from __future__ import annotations

import typing
from collections.abc import Generator
from contextlib import contextmanager

from ._client import Client
from ._config import (
    DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
    DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_TIMEOUT_CONFIG,
)
from ._models import Response
from ._types import (
    AuthTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
    TimeoutTypes,
)
from ._urls import URL

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

    from .websockets._api import WebSocketSession


__all__ = [
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "query",
    "request",
    "stream",
    "websocket",
]


def request(
    method: str,
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    trust_env: bool = True,
) -> Response:
    """Sends an HTTP request.

    Parameters:
        method: HTTP method for the new `Request` object: `GET`, `OPTIONS`, `HEAD`, `POST`, `PUT`, `PATCH`, `DELETE`,
            or `QUERY`.
        url: URL for the new `Request` object.
        params: *(optional)* Query parameters to include in the URL, as a string, dictionary, or sequence of two-tuples.
        content: *(optional)* Binary content to include in the body of the request, as bytes or a byte iterator.
        data: *(optional)* Form data to include in the body of the request, as a dictionary.
        files: *(optional)* A dictionary of upload files to include in the body of the request.
        json: *(optional)* A JSON serializable object to include in the body of the request.
        headers: *(optional)* Dictionary of HTTP headers to include in the request.
        cookies: *(optional)* Dictionary of Cookie items to include in the request.
        auth: *(optional)* An authentication class to use when sending the request.
        proxy: *(optional)* A proxy URL where all the traffic should be routed.
        timeout: *(optional)* The timeout configuration to use when sending the request.
        follow_redirects: *(optional)* Enables or disables HTTP redirects.
        verify: *(optional)* Either `True` to use an SSL context with the default CA bundle, `False` to disable
            verification, or an instance of `ssl.SSLContext` to use a custom context.
        trust_env: *(optional)* Enables or disables usage of environment variables for configuration.

    Returns:
        The `Response` object.

    Usage:

    ```
    >>> import httpx2
    >>> response = httpx2.request('GET', 'https://httpbin.org/get')
    >>> response
    <Response [200 OK]>
    ```
    """
    with Client(
        cookies=cookies,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    ) as client:
        return client.request(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects,
        )


@contextmanager
def stream(
    method: str,
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    trust_env: bool = True,
) -> Generator[Response]:
    """
    Alternative to `httpx2.request()` that streams the response body
    instead of loading it into memory at once.

    **Parameters**: See `httpx2.request`.

    See also: [Streaming Responses][0]

    [0]: quickstart.md#streaming-responses
    """
    with Client(
        cookies=cookies,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    ) as client:
        with client.stream(
            method=method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects,
        ) as response:
            yield response


def get(
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `GET` request.

    **Parameters**: See `httpx2.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `GET` requests should not include a request body.
    """
    return request(
        "GET",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def options(
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends an `OPTIONS` request.

    **Parameters**: See `httpx2.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `OPTIONS` requests should not include a request body.
    """
    return request(
        "OPTIONS",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def head(
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `HEAD` request.

    **Parameters**: See `httpx2.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `HEAD` requests should not include a request body.
    """
    return request(
        "HEAD",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def post(
    url: URL | str,
    *,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `POST` request.

    **Parameters**: See `httpx2.request`.
    """
    return request(
        "POST",
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def put(
    url: URL | str,
    *,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `PUT` request.

    **Parameters**: See `httpx2.request`.
    """
    return request(
        "PUT",
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def patch(
    url: URL | str,
    *,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `PATCH` request.

    **Parameters**: See `httpx2.request`.
    """
    return request(
        "PATCH",
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def delete(
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    verify: ssl.SSLContext | str | bool = True,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `DELETE` request.

    **Parameters**: See `httpx2.request`.

    Note that the `data`, `files`, `json` and `content` parameters are not available
    on this function, as `DELETE` requests should not include a request body.
    """
    return request(
        "DELETE",
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


def query(
    url: URL | str,
    *,
    content: RequestContent | None = None,
    data: RequestData | None = None,
    files: RequestFiles | None = None,
    json: typing.Any | None = None,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    verify: ssl.SSLContext | str | bool = True,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    trust_env: bool = True,
) -> Response:
    """
    Sends a `QUERY` request.

    **Parameters**: See `httpx2.request`.
    """
    return request(
        "QUERY",
        url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        follow_redirects=follow_redirects,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    )


@contextmanager
def websocket(
    url: URL | str,
    *,
    params: QueryParamTypes | None = None,
    headers: HeaderTypes | None = None,
    cookies: CookieTypes | None = None,
    auth: AuthTypes | None = None,
    proxy: ProxyTypes | None = None,
    follow_redirects: bool = False,
    timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
    verify: ssl.SSLContext | str | bool = True,
    trust_env: bool = True,
    subprotocols: list[str] | None = None,
    max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    keepalive_ping_interval_seconds: float | None = DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    keepalive_ping_timeout_seconds: float | None = DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
) -> Generator[WebSocketSession]:
    """
    Open a WebSocket session.

    The session is closed automatically when exiting the context manager.

    ```python
    with httpx2.websocket("ws://localhost:8000/ws") as ws:
        ws.send_text("Hello!")
        message = ws.receive_text()
    ```

    **Parameters**: See `httpx2.request` and `httpx2.Client.websocket`.
    """
    with Client(
        cookies=cookies,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        trust_env=trust_env,
    ) as client:
        with client.websocket(
            url,
            params=params,
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects,
            subprotocols=subprotocols,
            max_message_size_bytes=max_message_size_bytes,
            queue_size=queue_size,
            keepalive_ping_interval_seconds=keepalive_ping_interval_seconds,
            keepalive_ping_timeout_seconds=keepalive_ping_timeout_seconds,
        ) as session:
            yield session
<|endoftext|>from __future__ import annotations

import dataclasses
import os
import typing

from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes
from ._urls import URL

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

__all__ = ["Limits", "Proxy", "Timeout", "create_ssl_context"]


class UnsetType:
    pass  # pragma: no cover


UNSET = UnsetType()


def create_ssl_context(
    verify: ssl.SSLContext | str | bool = True,
    cert: CertTypes | None = None,
    trust_env: bool = True,
) -> ssl.SSLContext:
    import ssl
    import warnings

    import truststore

    if verify is True:
        if trust_env and os.environ.get("SSL_CERT_FILE"):  # pragma: no cover
            ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
        elif trust_env and os.environ.get("SSL_CERT_DIR"):  # pragma: no cover
            ctx = ssl.create_default_context(capath=os.environ["SSL_CERT_DIR"])
        else:
            # Default case: rely on the system trust store via `truststore`.
            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    elif verify is False:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif isinstance(verify, str):
        if cert:
            raise TypeError(
                "`verify=<str>` cannot be combined with `cert=...`. "
                "Build an `ssl.SSLContext` and pass it as `verify=<ctx>`, "
                "using `.load_cert_chain()` to configure the certificate chain."
            )
        message = (
            "`verify=<str>` is deprecated. "
            "Use `verify=ssl.create_default_context(cafile=...)` "
            "or `verify=ssl.create_default_context(capath=...)` instead."
        )
        warnings.warn(message, DeprecationWarning)
        if os.path.isdir(verify):  # pragma: no cover
            return ssl.create_default_context(capath=verify)
        return ssl.create_default_context(cafile=verify)
    else:
        ctx = verify

    if cert:  # pragma: no cover
        message = (
            "`cert=...` is deprecated. Use `verify=<ssl_context>` instead,"
            "with `.load_cert_chain()` to configure the certificate chain."
        )
        warnings.warn(message, DeprecationWarning)
        if isinstance(cert, str):
            ctx.load_cert_chain(cert)
        else:
            ctx.load_cert_chain(*cert)

    return ctx


@dataclasses.dataclass(init=False, repr=False)
class Timeout:
    """
    Timeout configuration.

    **Usage**:

    Timeout(None)               # No timeouts.
    Timeout(5.0)                # 5s timeout on all operations.
    Timeout(None, connect=5.0)  # 5s timeout on connect, no other timeouts.
    Timeout(5.0, connect=10.0)  # 10s timeout on connect. 5s timeout elsewhere.
    Timeout(5.0, pool=None)     # No timeout on acquiring connection from pool.
                                # 5s timeout elsewhere.
    """

    connect: float | None
    read: float | None
    write: float | None
    pool: float | None

    def __init__(
        self,
        timeout: TimeoutTypes | UnsetType = UNSET,
        *,
        connect: None | float | UnsetType = UNSET,
        read: None | float | UnsetType = UNSET,
        write: None | float | UnsetType = UNSET,
        pool: None | float | UnsetType = UNSET,
    ) -> None:
