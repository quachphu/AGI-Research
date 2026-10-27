    """Return a port that is unused on the current host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return cast(int, s.getsockname()[1])


class BaseTestServer(ABC):
    __test__ = False

    def __init__(
        self,
        *,
        scheme: str = "",
        loop: asyncio.AbstractEventLoop | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        skip_url_asserts: bool = False,
        socket_factory: Callable[
            [str, int, socket.AddressFamily], socket.socket
        ] = get_port_socket,
        **kwargs: Any,
    ) -> None:
        self._loop = loop
        self.runner: BaseRunner | None = None
        self._root: URL | None = None
        self.host = host
        self.port = port
        self._closed = False
        self.scheme = scheme
        self.skip_url_asserts = skip_url_asserts
        self.socket_factory = socket_factory

    async def start_server(
        self, loop: asyncio.AbstractEventLoop | None = None, **kwargs: Any
    ) -> None:
        if self.runner:
            return
        self._loop = loop
        self._ssl = kwargs.pop("ssl", None)
        self.runner = await self._make_runner(handler_cancellation=True, **kwargs)
        await self.runner.setup()
        if not self.port:
            self.port = 0
        absolute_host = self.host
        try:
            version = ipaddress.ip_address(self.host).version
        except ValueError:
            version = 4
        if version == 6:
            absolute_host = f"[{self.host}]"
        family = socket.AF_INET6 if version == 6 else socket.AF_INET
        _sock = self.socket_factory(self.host, self.port, family)
        self.host, self.port = _sock.getsockname()[:2]
        site = SockSite(self.runner, sock=_sock, ssl_context=self._ssl)
        await site.start()
        server = site._server
        assert server is not None
        sockets = server.sockets  # type: ignore[attr-defined]
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]
        if not self.scheme:
            self.scheme = "https" if self._ssl else "http"
        self._root = URL(f"{self.scheme}://{absolute_host}:{self.port}")

    @abstractmethod  # pragma: no cover
    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        pass

    def make_url(self, path: StrOrURL) -> URL:
        assert self._root is not None
        url = URL(path)
        if not self.skip_url_asserts:
            assert not url.absolute
            return self._root.join(url)
        else:
            return URL(str(self._root) + str(path))

    @property
    def started(self) -> bool:
        return self.runner is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def handler(self) -> Server:
        # for backward compatibility
        # web.Server instance
        runner = self.runner
        assert runner is not None
        assert runner.server is not None
        return runner.server

    async def close(self) -> None:
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run when the object is garbage collected, and on
        exit when used as a context manager.

        """
        if self.started and not self.closed:
            assert self.runner is not None
            await self.runner.cleanup()
            self._root = None
            self.port = None
            self._closed = True

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "BaseTestServer":
        await self.start_server(loop=self._loop)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


class TestServer(BaseTestServer):
    def __init__(
        self,
        app: Application,
        *,
        scheme: str = "",
        host: str = "127.0.0.1",
        port: int | None = None,
        **kwargs: Any,
    ):
        self.app = app
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        return AppRunner(self.app, **kwargs)


class RawTestServer(BaseTestServer):
    def __init__(
        self,
        handler: _RequestHandler,
        *,
        scheme: str = "",
        host: str = "127.0.0.1",
        port: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._handler = handler
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, debug: bool = True, **kwargs: Any) -> ServerRunner:
        srv = Server(self._handler, loop=self._loop, debug=debug, **kwargs)
        return ServerRunner(srv, debug=debug, **kwargs)


class TestClient(Generic[_Request, _ApplicationNone]):
    """
    A test client implementation.

    To write functional tests for aiohttp based servers.

    """

    __test__ = False

    @overload
    def __init__(
        self: "TestClient[Request, Application]",
        server: TestServer,
        *,
        cookie_jar: AbstractCookieJar | None = None,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self: "TestClient[_Request, None]",
        server: BaseTestServer,
        *,
        cookie_jar: AbstractCookieJar | None = None,
        **kwargs: Any,
    ) -> None: ...
    def __init__(
        self,
        server: BaseTestServer,
        *,
        cookie_jar: AbstractCookieJar | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(server, BaseTestServer):
            raise TypeError(
                "server must be TestServer instance, found type: %r" % type(server)
            )
        self._server = server
        self._loop = loop
        if cookie_jar is None:
            cookie_jar = aiohttp.CookieJar(unsafe=True, loop=loop)
        self._session = ClientSession(loop=loop, cookie_jar=cookie_jar, **kwargs)
        self._session._retry_connection = False
        self._closed = False
        self._responses: list[ClientResponse] = []
        self._websockets: list[ClientWebSocketResponse[bool]] = []

    async def start_server(self) -> None:
        await self._server.start_server(loop=self._loop)

    @property
    def host(self) -> str:
        return self._server.host

    @property
    def port(self) -> int | None:
        return self._server.port

    @property
    def server(self) -> BaseTestServer:
        return self._server

    @property
    def app(self) -> _ApplicationNone:
        return getattr(self._server, "app", None)  # type: ignore[return-value]

    @property
    def session(self) -> ClientSession:
        """An internal aiohttp.ClientSession.

        Unlike the methods on the TestClient, client session requests
        do not automatically include the host in the url queried, and
        will require an absolute path to the resource.

        """
        return self._session

    def make_url(self, path: StrOrURL) -> URL:
        return self._server.make_url(path)

    async def _request(
        self, method: str, path: StrOrURL, **kwargs: Any
    ) -> ClientResponse:
        resp = await self._session.request(method, self.make_url(path), **kwargs)
        # save it to close later
        self._responses.append(resp)
        return resp

    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        def request(
            self, method: str, path: StrOrURL, **kwargs: Unpack[_RequestOptions]
        ) -> _RequestContextManager: ...

        def get(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def options(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def head(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def post(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def put(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def patch(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

        def delete(
            self,
            path: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> _RequestContextManager: ...

    else:

        def request(
            self, method: str, path: StrOrURL, **kwargs: Any
        ) -> _RequestContextManager:
            """Routes a request to tested http server.

            The interface is identical to aiohttp.ClientSession.request,
            except the loop kwarg is overridden by the instance used by the
            test server.

            """
            return _RequestContextManager(self._request(method, path, **kwargs))

        def get(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP GET request."""
            return _RequestContextManager(self._request(hdrs.METH_GET, path, **kwargs))

        def post(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP POST request."""
            return _RequestContextManager(self._request(hdrs.METH_POST, path, **kwargs))

        def options(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP OPTIONS request."""
            return _RequestContextManager(
                self._request(hdrs.METH_OPTIONS, path, **kwargs)
            )

        def head(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP HEAD request."""
            return _RequestContextManager(self._request(hdrs.METH_HEAD, path, **kwargs))

        def put(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP PUT request."""
            return _RequestContextManager(self._request(hdrs.METH_PUT, path, **kwargs))

        def patch(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP PATCH request."""
            return _RequestContextManager(
                self._request(hdrs.METH_PATCH, path, **kwargs)
            )

        def delete(self, path: StrOrURL, **kwargs: Any) -> _RequestContextManager:
            """Perform an HTTP PATCH request."""
            return _RequestContextManager(
                self._request(hdrs.METH_DELETE, path, **kwargs)
            )

    @overload
    def ws_connect(
        self, path: StrOrURL, *, decode_text: Literal[True] = ..., **kwargs: Any
    ) -> "_BaseRequestContextManager[ClientWebSocketResponse[Literal[True]]]": ...

    @overload
    def ws_connect(
        self, path: StrOrURL, *, decode_text: Literal[False], **kwargs: Any
    ) -> "_BaseRequestContextManager[ClientWebSocketResponse[Literal[False]]]": ...

    @overload
    def ws_connect(
        self, path: StrOrURL, *, decode_text: bool = ..., **kwargs: Any
    ) -> "_BaseRequestContextManager[ClientWebSocketResponse[bool]]": ...

    def ws_connect(
        self, path: StrOrURL, *, decode_text: bool = True, **kwargs: Any
    ) -> "_BaseRequestContextManager[ClientWebSocketResponse[bool]]":
        """Initiate websocket connection.

        The api corresponds to aiohttp.ClientSession.ws_connect.

        """
        return _WSRequestContextManager(
            self._ws_connect(path, decode_text=decode_text, **kwargs)
        )

    @overload
    async def _ws_connect(
        self, path: StrOrURL, *, decode_text: Literal[True] = ..., **kwargs: Any
    ) -> "ClientWebSocketResponse[Literal[True]]": ...

    @overload
    async def _ws_connect(
        self, path: StrOrURL, *, decode_text: Literal[False], **kwargs: Any
    ) -> "ClientWebSocketResponse[Literal[False]]": ...

    @overload
    async def _ws_connect(
        self, path: StrOrURL, *, decode_text: bool = ..., **kwargs: Any
    ) -> "ClientWebSocketResponse[bool]": ...

    async def _ws_connect(
        self, path: StrOrURL, *, decode_text: bool = True, **kwargs: Any
    ) -> "ClientWebSocketResponse[bool]":
        ws = await self._session.ws_connect(
            self.make_url(path), decode_text=decode_text, **kwargs
        )
        self._websockets.append(ws)
        return ws

    async def close(self) -> None:
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run on exit when used as a(n) (asynchronous)
        context manager.

        """
        if not self._closed:
            for resp in self._responses:
                resp.close()
            for ws in self._websockets:
                await ws.close()
            await self._session.close()
            await self._server.close()
            self._closed = True

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> Self:
        await self.start_server()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class AioHTTPTestCase(IsolatedAsyncioTestCase):
    """A base class to allow for unittest web applications using aiohttp.

    Provides the following:

    * self.client (aiohttp.test_utils.TestClient): an aiohttp test client.
    * self.loop (asyncio.BaseEventLoop): the event loop in which the
        application and server are running.
    * self.app (aiohttp.web.Application): the application returned by
        self.get_application()

    Note that the TestClient's methods are asynchronous: you have to
    execute function on the test client using asynchronous methods.
    """

    async def get_application(self) -> Application:
        """Get application.

        This method should be overridden
        to return the aiohttp.web.Application
        object to test.
        """
        return self.get_app()

    def get_app(self) -> Application:
        """Obsolete method used to constructing web application.

        Use .get_application() coroutine instead.
        """
        raise RuntimeError("Did you forget to define get_application()?")

    async def asyncSetUp(self) -> None:
        self.loop = asyncio.get_running_loop()
        return await self.setUpAsync()

    async def setUpAsync(self) -> None:
        self.app = await self.get_application()
        self.server = await self.get_server(self.app)
        self.client = await self.get_client(self.server)

        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        return await self.tearDownAsync()

    async def tearDownAsync(self) -> None:
        await self.client.close()

    async def get_server(self, app: Application) -> TestServer:
        """Return a TestServer instance."""
        return TestServer(app, loop=self.loop)

    async def get_client(self, server: TestServer) -> TestClient[Request, Application]:
        """Return a TestClient instance."""
        return TestClient(server, loop=self.loop)


def unittest_run_loop(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    A decorator dedicated to use with asynchronous AioHTTPTestCase test methods.

    In 3.8+, this does nothing.
    """
    warnings.warn(
        "Decorator `@unittest_run_loop` is no longer needed in aiohttp 3.8+",
        DeprecationWarning,
        stacklevel=2,
    )
    return func


_LOOP_FACTORY = Callable[[], asyncio.AbstractEventLoop]


@contextlib.contextmanager
def loop_context(
    loop_factory: _LOOP_FACTORY = asyncio.new_event_loop, fast: bool = False
) -> Iterator[asyncio.AbstractEventLoop]:
    """A contextmanager that creates an event_loop, for test purposes.

    Handles the creation and cleanup of a test loop.
    """
    loop = setup_test_loop(loop_factory)
    yield loop
    teardown_test_loop(loop, fast=fast)


def setup_test_loop(
    loop_factory: _LOOP_FACTORY = asyncio.new_event_loop,
) -> asyncio.AbstractEventLoop:
    """Create and return an asyncio.BaseEventLoop instance.

    The caller should also call teardown_test_loop,
    once they are done with the loop.
    """
    loop = loop_factory()
    asyncio.set_event_loop(loop)
    return loop


def teardown_test_loop(loop: asyncio.AbstractEventLoop, fast: bool = False) -> None:
    """Teardown and cleanup an event_loop created by setup_test_loop."""
    closed = loop.is_closed()
    if not closed:
        loop.call_soon(loop.stop)
        loop.run_forever()
        loop.close()

    if not fast:
        gc.collect()

    asyncio.set_event_loop(None)


def _create_app_mock() -> mock.MagicMock:
    def get_dict(app: Any, key: str) -> Any:
        return app.__app_dict[key]

    def set_dict(app: Any, key: str, value: Any) -> None:
        app.__app_dict[key] = value

    app = mock.MagicMock(spec=Application)
    app.__app_dict = {}
    app.__getitem__ = get_dict
    app.__setitem__ = set_dict

    app._debug = False
    app.on_response_prepare = Signal(app)
    app.on_response_prepare.freeze()
    return app


def _create_transport(sslcontext: SSLContext | None = None) -> mock.Mock:
    transport = mock.Mock()

    def get_extra_info(key: str) -> SSLContext | tuple[str, int] | None:
        if key == "sslcontext":
            return sslcontext
        return ("127.0.0.1", 80) if key == "sockname" else None

    transport.get_extra_info.side_effect = get_extra_info
    return transport


def make_mocked_request(
    method: str,
    path: str,
    headers: Any = None,
    *,
    match_info: Any = sentinel,
    version: HttpVersion = HttpVersion(1, 1),
    closing: bool = False,
    app: Any = None,
    writer: Any = sentinel,
    protocol: Any = sentinel,
    transport: Any = sentinel,
    payload: StreamReader = EMPTY_PAYLOAD,
    sslcontext: SSLContext | None = None,
    client_max_size: int = 1024**2,
    client_max_fields: int = 1000,
    loop: Any = ...,
) -> Request:
    """Creates mocked web.Request testing purposes.

    Useful in unit tests, when spinning full web server is overkill or
    specific conditions and errors are hard to trigger.
    """
    task = mock.Mock()
    if loop is ...:
        # no loop passed, try to get the current one if
        # its is running as we need a real loop to create
        # executor jobs to be able to do testing
        # with a real executor
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = mock.Mock()
            loop.create_future.return_value = ()

    if version < HttpVersion(1, 1):
        closing = True

    if headers:
        headers = CIMultiDictProxy(CIMultiDict(headers))
        raw_hdrs = tuple(
            (k.encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()
        )
    else:
        headers = CIMultiDictProxy(CIMultiDict())
        raw_hdrs = ()

    chunked = "chunked" in headers.get(hdrs.TRANSFER_ENCODING, "").lower()
    upgrade = headers.get(hdrs.CONNECTION, "").lower() == "upgrade" and bool(
        headers.get(hdrs.UPGRADE)
    )

    message = RawRequestMessage(
        method.upper(),
        path,
        version,
        headers,
        raw_hdrs,
        closing,
        None,
        upgrade,
        chunked,
        URL(path),
    )
    if app is None:
        app = _create_app_mock()

    if transport is sentinel:
        transport = _create_transport(sslcontext)

    if protocol is sentinel:
        protocol = mock.Mock()
        protocol.max_field_size = 8190
        protocol.max_line_length = 8190
        protocol.max_headers = 128
        protocol.transport = transport
        type(protocol).peername = mock.PropertyMock(
            return_value=transport.get_extra_info("peername")
        )
        type(protocol).sockname = mock.PropertyMock(
            return_value=transport.get_extra_info("sockname")
        )
        type(protocol).ssl_context = mock.PropertyMock(return_value=sslcontext)

    if writer is sentinel:
        writer = mock.Mock()
        writer.write_headers = make_mocked_coro(None)
        writer.write = make_mocked_coro(None)
        writer.write_eof = make_mocked_coro(None)
        writer.drain = make_mocked_coro(None)
        writer.transport = transport

    protocol.transport = transport
    protocol.writer = writer

    req = Request(
        message,
        payload,
        protocol,
        writer,
        task,
        loop,
        client_max_size=client_max_size,
        client_max_fields=client_max_fields,
    )

    match_info = UrlMappingMatchInfo(
        {} if match_info is sentinel else match_info, mock.Mock()
    )
    match_info.add_app(app)
    req._match_info = match_info

    return req


def make_mocked_coro(
    return_value: Any = sentinel, raise_exception: Any = sentinel
) -> Any:
    """Creates a coroutine mock."""

    async def mock_coro(*args: Any, **kwargs: Any) -> Any:
        if raise_exception is not sentinel:
            raise raise_exception
        if not inspect.isawaitable(return_value):
            return return_value
        await return_value

    return mock.Mock(wraps=mock_coro)
<|endoftext|>import asyncio
import contextlib
import inspect
import warnings
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Protocol, overload

import pytest

from .test_utils import (
    BaseTestServer,
    RawTestServer,
    TestClient,
    TestServer,
    loop_context,
    setup_test_loop,
    teardown_test_loop,
    unused_port as _unused_port,
)
from .web import Application, BaseRequest, Request
from .web_protocol import _RequestHandler

try:
    import uvloop
except ImportError:  # pragma: no cover
    uvloop = None  # type: ignore[assignment]


class AiohttpClient(Protocol):
    @overload
    async def __call__(
        self,
        __param: Application,
        *,
        server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestClient[Request, Application]: ...
    @overload
    async def __call__(
        self,
        __param: BaseTestServer,
        *,
        server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestClient[BaseRequest, None]: ...


class AiohttpServer(Protocol):
    def __call__(
        self, app: Application, *, port: int | None = None, **kwargs: Any
    ) -> Awaitable[TestServer]: ...


class AiohttpRawServer(Protocol):
    def __call__(
        self, handler: _RequestHandler, *, port: int | None = None, **kwargs: Any
    ) -> Awaitable[RawTestServer]: ...


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    parser.addoption(
        "--aiohttp-fast",
        action="store_true",
        default=False,
        help="run tests faster by disabling extra checks",
    )
    parser.addoption(
        "--aiohttp-loop",
        action="store",
        default="pyloop",
        help="run tests with specific loop: pyloop, uvloop or all",
    )
    parser.addoption(
        "--aiohttp-enable-loop-debug",
        action="store_true",
        default=False,
        help="enable event loop debug mode",
    )


def pytest_fixture_setup(fixturedef):  # type: ignore[no-untyped-def]
    """Set up pytest fixture.

    Allow fixtures to be coroutines. Run coroutine fixtures in an event loop.
    """
    func = fixturedef.func

    if inspect.isasyncgenfunction(func):
        # async generator fixture
        is_async_gen = True
    elif inspect.iscoroutinefunction(func):
        # regular async fixture
        is_async_gen = False
    else:
        # not an async fixture, nothing to do
        return

    strip_request = False
    if "request" not in fixturedef.argnames:
        fixturedef.argnames += ("request",)
        strip_request = True

    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        request = kwargs["request"]
        if strip_request:
            del kwargs["request"]

        # if neither the fixture nor the test use the 'loop' fixture,
        # 'getfixturevalue' will fail because the test is not parameterized
        # (this can be removed someday if 'loop' is no longer parameterized)
        if "loop" not in request.fixturenames:
            raise Exception(
                "Asynchronous fixtures must depend on the 'loop' fixture or "
                "be used in tests depending from it."
            )

        _loop = request.getfixturevalue("loop")

        if is_async_gen:
            # for async generators, we need to advance the generator once,
            # then advance it again in a finalizer
            gen = func(*args, **kwargs)

            def finalizer():  # type: ignore[no-untyped-def]
                try:
                    return _loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    pass

            request.addfinalizer(finalizer)
            return _loop.run_until_complete(gen.__anext__())
        else:
            return _loop.run_until_complete(func(*args, **kwargs))

    fixturedef.func = wrapper


@pytest.fixture
def fast(request):  # type: ignore[no-untyped-def]
    """--fast config option"""
    return request.config.getoption("--aiohttp-fast")


@pytest.fixture
def loop_debug(request):  # type: ignore[no-untyped-def]
    """--enable-loop-debug config option"""
    return request.config.getoption("--aiohttp-enable-loop-debug")


@contextlib.contextmanager
def _runtime_warning_context():  # type: ignore[no-untyped-def]
    """Context manager which checks for RuntimeWarnings.

    This exists specifically to
    avoid "coroutine 'X' was never awaited" warnings being missed.

    If RuntimeWarnings occur in the context a RuntimeError is raised.
    """
    with warnings.catch_warnings(record=True) as _warnings:
        yield
        rw = [
            f"{w.filename}:{w.lineno}:{w.message}"
            for w in _warnings
            if w.category == RuntimeWarning
        ]
        if rw:
            raise RuntimeError(
                "{} Runtime Warning{},\n{}".format(
                    len(rw), "" if len(rw) == 1 else "s", "\n".join(rw)
                )
            )


@contextlib.contextmanager
def _passthrough_loop_context(loop, fast=False):  # type: ignore[no-untyped-def]
    """Passthrough loop context.

    Sets up and tears down a loop unless one is passed in via the loop
    argument when it's passed straight through.
    """
    if loop:
        # loop already exists, pass it straight through
        yield loop
    else:
        # this shadows loop_context's standard behavior
        loop = setup_test_loop()
        yield loop
        teardown_test_loop(loop, fast=fast)


def pytest_pycollect_makeitem(collector, name, obj):  # type: ignore[no-untyped-def]
    """Fix pytest collecting for coroutines."""
    if collector.funcnamefilter(name) and inspect.iscoroutinefunction(obj):
        return list(collector._genfunctions(name, obj))


def pytest_pyfunc_call(pyfuncitem):  # type: ignore[no-untyped-def]
    """Run coroutines in an event loop instead of a normal function call."""
    fast = pyfuncitem.config.getoption("--aiohttp-fast")
    if inspect.iscoroutinefunction(pyfuncitem.function):
        warnings.warn(
            "aiohttp.pytest_plugin will be removed in v4. Please install pytest-aiohttp.",
            DeprecationWarning,
        )
        existing_loop = (
            pyfuncitem.funcargs.get("proactor_loop")
            or pyfuncitem.funcargs.get("selector_loop")
            or pyfuncitem.funcargs.get("uvloop_loop")
            or pyfuncitem.funcargs.get("loop", None)
        )

        with _runtime_warning_context():
            with _passthrough_loop_context(existing_loop, fast=fast) as _loop:
                testargs = {
                    arg: pyfuncitem.funcargs[arg]
                    for arg in pyfuncitem._fixtureinfo.argnames
                }
                _loop.run_until_complete(pyfuncitem.obj(**testargs))

        return True


def pytest_generate_tests(metafunc):  # type: ignore[no-untyped-def]
    if "loop_factory" not in metafunc.fixturenames:
        return

    loops = metafunc.config.option.aiohttp_loop
    avail_factories: dict[str, Callable[[], asyncio.AbstractEventLoop]]
    avail_factories = {"pyloop": asyncio.new_event_loop}

    if uvloop is not None:  # pragma: no cover
        avail_factories["uvloop"] = uvloop.new_event_loop

    if loops == "all":
        loops = "pyloop,uvloop?"

    factories = {}  # type: ignore[var-annotated]
    for name in loops.split(","):
        required = not name.endswith("?")
        name = name.strip(" ?")
        if name not in avail_factories:  # pragma: no cover
            if required:
                raise ValueError(
                    "Unknown loop '%s', available loops: %s"
                    % (name, list(factories.keys()))
                )
            else:
                continue
        factories[name] = avail_factories[name]
    metafunc.parametrize(
        "loop_factory", list(factories.values()), ids=list(factories.keys())
    )


@pytest.fixture
def loop(
    loop_factory: Callable[[], asyncio.AbstractEventLoop],
    fast: bool,
    loop_debug: bool,
) -> Iterator[asyncio.AbstractEventLoop]:
    """Return an instance of the event loop."""
    with loop_context(loop_factory, fast=fast) as _loop:
        if loop_debug:
            _loop.set_debug(True)  # pragma: no cover
        asyncio.set_event_loop(_loop)
        yield _loop


@pytest.fixture
def proactor_loop() -> Iterator[asyncio.AbstractEventLoop]:
    factory = asyncio.ProactorEventLoop  # type: ignore[attr-defined]

    with loop_context(factory) as _loop:
        asyncio.set_event_loop(_loop)
        yield _loop


@pytest.fixture
def unused_port(aiohttp_unused_port: Callable[[], int]) -> Callable[[], int]:
    warnings.warn(
        "Deprecated, use aiohttp_unused_port fixture instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return aiohttp_unused_port


@pytest.fixture
def aiohttp_unused_port() -> Callable[[], int]:
    """Return a port that is unused on the current host."""
    return _unused_port


@pytest.fixture
def aiohttp_server(loop: asyncio.AbstractEventLoop) -> Iterator[AiohttpServer]:
    """Factory to create a TestServer instance, given an app.

    aiohttp_server(app, **kwargs)
    """
    servers = []

    async def go(
        app: Application,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        **kwargs: Any,
    ) -> TestServer:
        server = TestServer(app, host=host, port=port)
        await server.start_server(loop=loop, **kwargs)
        servers.append(server)
        return server

    yield go

    async def finalize() -> None:
        while servers:
            await servers.pop().close()

    loop.run_until_complete(finalize())


@pytest.fixture
def test_server(aiohttp_server):  # type: ignore[no-untyped-def]  # pragma: no cover
    warnings.warn(
        "Deprecated, use aiohttp_server fixture instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return aiohttp_server


@pytest.fixture
def aiohttp_raw_server(loop: asyncio.AbstractEventLoop) -> Iterator[AiohttpRawServer]:
    """Factory to create a RawTestServer instance, given a web handler.

    aiohttp_raw_server(handler, **kwargs)
    """
    servers = []

    async def go(
        handler: _RequestHandler, *, port: int | None = None, **kwargs: Any
    ) -> RawTestServer:
        server = RawTestServer(handler, port=port)
        await server.start_server(loop=loop, **kwargs)
        servers.append(server)
        return server

    yield go

    async def finalize() -> None:
        while servers:
            await servers.pop().close()

    loop.run_until_complete(finalize())


@pytest.fixture
def raw_test_server(  # type: ignore[no-untyped-def]  # pragma: no cover
    aiohttp_raw_server,
):
    warnings.warn(
        "Deprecated, use aiohttp_raw_server fixture instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return aiohttp_raw_server


@pytest.fixture
def aiohttp_client(loop: asyncio.AbstractEventLoop) -> Iterator[AiohttpClient]:
    """Factory to create a TestClient instance.

    aiohttp_client(app, **kwargs)
    aiohttp_client(server, **kwargs)
    aiohttp_client(raw_server, **kwargs)
    """
    clients = []

    @overload
    async def go(
        __param: Application,
        *,
        server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestClient[Request, Application]: ...

    @overload
    async def go(
        __param: BaseTestServer,
        *,
        server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestClient[BaseRequest, None]: ...

    async def go(
        __param: Application | BaseTestServer,
        *args: Any,
        server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TestClient[Any, Any]:
        if isinstance(__param, Callable) and not isinstance(  # type: ignore[arg-type]
            __param, (Application, BaseTestServer)
        ):
            __param = __param(loop, *args, **kwargs)
            kwargs = {}
        else:
            assert not args, "args should be empty"

        if isinstance(__param, Application):
            server_kwargs = server_kwargs or {}
            server = TestServer(__param, loop=loop, **server_kwargs)
            client = TestClient(server, loop=loop, **kwargs)
        elif isinstance(__param, BaseTestServer):
            client = TestClient(__param, loop=loop, **kwargs)
        else:
            raise ValueError("Unknown argument type: %r" % type(__param))

        await client.start_server()
        clients.append(client)
        return client

    yield go

    async def finalize() -> None:
        while clients:
            await clients.pop().close()

    loop.run_until_complete(finalize())


@pytest.fixture
def test_client(aiohttp_client):  # type: ignore[no-untyped-def]  # pragma: no cover
    warnings.warn(
        "Deprecated, use aiohttp_client fixture instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return aiohttp_client
<|endoftext|>import io
import warnings
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode

from multidict import MultiDict, MultiDictProxy

from . import hdrs, multipart, payload
from .helpers import guess_filename
from .http_writer import _safe_header
from .payload import Payload

__all__ = ("FormData",)


class FormData:
    """Helper class for form body generation.

    Supports multipart/form-data and application/x-www-form-urlencoded.
    """

    def __init__(
        self,
        fields: Iterable[Any] = (),
        quote_fields: bool = True,
        charset: str | None = None,
        *,
        default_to_multipart: bool = False,
    ) -> None:
        self._writer = multipart.MultipartWriter("form-data")
        self._fields: list[Any] = []
        self._is_multipart = default_to_multipart
        self._quote_fields = quote_fields
        self._charset = charset

        if isinstance(fields, dict):
            fields = list(fields.items())
        elif not isinstance(fields, (list, tuple)):
            fields = (fields,)
        self.add_fields(*fields)

    @property
    def is_multipart(self) -> bool:
        return self._is_multipart

    def add_field(
        self,
        name: str,
        value: Any,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        content_transfer_encoding: str | None = None,
    ) -> None:

        if isinstance(value, io.IOBase):
            self._is_multipart = True
        elif isinstance(value, (bytes, bytearray, memoryview)):
            msg = (
                "In v4, passing bytes will no longer create a file field. "
                "Please explicitly use the filename parameter or pass a BytesIO object."
            )
            if filename is None and content_transfer_encoding is None:
                warnings.warn(msg, DeprecationWarning)
                filename = name

        _safe_header(name)
        type_options: MultiDict[str] = MultiDict({"name": name})
        if filename is not None and not isinstance(filename, str):
            raise TypeError("filename must be an instance of str. Got: %s" % filename)
        if filename is None and isinstance(value, io.IOBase):
            filename = guess_filename(value, name)
        if filename is not None:
            _safe_header(filename)
            type_options["filename"] = filename
            self._is_multipart = True

        headers = {}
        if content_type is not None:
            if not isinstance(content_type, str):
                raise TypeError(
                    "content_type must be an instance of str. Got: %s" % content_type
                )
            _safe_header(content_type)
            headers[hdrs.CONTENT_TYPE] = content_type
            self._is_multipart = True
        if content_transfer_encoding is not None:
            if not isinstance(content_transfer_encoding, str):
                raise TypeError(
                    "content_transfer_encoding must be an instance"
                    " of str. Got: %s" % content_transfer_encoding
                )
            msg = (
                "content_transfer_encoding is deprecated. "
                "To maintain compatibility with v4 please pass a BytesPayload."
            )
            warnings.warn(msg, DeprecationWarning)
            self._is_multipart = True

        self._fields.append((type_options, headers, value))

    def add_fields(self, *fields: Any) -> None:
        to_add = list(fields)

        while to_add:
            rec = to_add.pop(0)

            if isinstance(rec, io.IOBase):
                k = guess_filename(rec, "unknown")
                self.add_field(k, rec)  # type: ignore[arg-type]

            elif isinstance(rec, (MultiDictProxy, MultiDict)):
                to_add.extend(rec.items())

            elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                k, fp = rec
                self.add_field(k, fp)

            else:
                raise TypeError(
                    "Only io.IOBase, multidict and (name, file) "
                    "pairs allowed, use .add_field() for passing "
                    f"more complex parameters, got {rec!r}"
                )

    def _gen_form_urlencoded(self) -> payload.BytesPayload:
        # form data (x-www-form-urlencoded)
        data = []
        for type_options, _, value in self._fields:
            data.append((type_options["name"], value))

        charset = self._charset if self._charset is not None else "utf-8"

        if charset == "utf-8":
            content_type = "application/x-www-form-urlencoded"
        else:
            content_type = "application/x-www-form-urlencoded; charset=%s" % charset

        return payload.BytesPayload(
            urlencode(data, doseq=True, encoding=charset).encode(),
            content_type=content_type,
        )

    def _gen_form_data(self) -> multipart.MultipartWriter:
        """Encode a list of fields using the multipart/form-data MIME format"""
        for dispparams, headers, value in self._fields:
            try:
                if hdrs.CONTENT_TYPE in headers:
                    part = payload.get_payload(
                        value,
                        content_type=headers[hdrs.CONTENT_TYPE],
                        headers=headers,
                        encoding=self._charset,
                    )
                else:
                    part = payload.get_payload(
                        value, headers=headers, encoding=self._charset
                    )
            except Exception as exc:
                raise TypeError(
                    "Can not serialize value type: %r\n "
                    "headers: %r\n value: %r" % (type(value), headers, value)
                ) from exc

            if dispparams:
                part.set_content_disposition(
                    "form-data", quote_fields=self._quote_fields, **dispparams
                )
                # FIXME cgi.FieldStorage doesn't likes body parts with
                # Content-Length which were sent via chunked transfer encoding
                assert part.headers is not None
                part.headers.popall(hdrs.CONTENT_LENGTH, None)

            self._writer.append_payload(part)

        self._fields.clear()
        return self._writer

    def __call__(self) -> Payload:
        if self._is_multipart:
            return self._gen_form_data()
        else:
            return self._gen_form_urlencoded()
<|endoftext|>"""Helper methods to tune a TCP connection"""

import asyncio
import socket
from contextlib import suppress
from typing import Optional  # noqa

__all__ = ("tcp_keepalive", "tcp_nodelay")


if hasattr(socket, "SO_KEEPALIVE"):

    def tcp_keepalive(transport: asyncio.Transport) -> None:
        sock = transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

else:

    def tcp_keepalive(transport: asyncio.Transport) -> None:  # pragma: no cover
        pass


def tcp_nodelay(transport: asyncio.Transport, value: bool) -> None:
    sock = transport.get_extra_info("socket")

    if sock is None:
        return

    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return

    value = bool(value)

    # socket may be closed already, on windows OSError get raised
    with suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, value)
<|endoftext|>import asyncio
import socket
import sys
import weakref
from typing import Any, Final, Optional

from .abc import AbstractResolver, ResolveResult

__all__ = ("ThreadedResolver", "AsyncResolver", "DefaultResolver")


try:
    import aiodns

    aiodns_default = hasattr(aiodns.DNSResolver, "getaddrinfo")
except ImportError:  # pragma: no cover
    aiodns = None  # type: ignore[assignment]
    aiodns_default = False


_NUMERIC_SOCKET_FLAGS = socket.AI_NUMERICHOST | socket.AI_NUMERICSERV
_NAME_SOCKET_FLAGS = socket.NI_NUMERICHOST | socket.NI_NUMERICSERV
_AI_ADDRCONFIG = socket.AI_ADDRCONFIG
if hasattr(socket, "AI_MASK"):
    _AI_ADDRCONFIG &= socket.AI_MASK
_IS_WINDOWS = sys.platform == "win32"


def _is_windows_localhost(host: str) -> bool:
    return _IS_WINDOWS and host.rstrip(".").casefold() == "localhost"


class ThreadedResolver(AbstractResolver):
    """Threaded resolver.

    Uses an Executor for synchronous getaddrinfo() calls.
    concurrent.futures.ThreadPoolExecutor is used by default.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        try:
            infos = await self._loop.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                family=family,
                flags=_AI_ADDRCONFIG,
            )
        except socket.gaierror:
            if not _is_windows_localhost(host):
                raise
            infos = await self._loop.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                family=family,
                flags=0,
            )

        hosts: list[ResolveResult] = []
        for family, _, proto, _, address in infos:
            if family == socket.AF_INET6:
                if len(address) < 3:
                    # IPv6 is not supported by Python build,
                    # or IPv6 is not enabled in the host
                    continue
                if address[3]:
                    # This is essential for link-local IPv6 addresses.
                    # LL IPv6 is a VERY rare case. Strictly speaking, we should use
                    # getnameinfo() unconditionally, but performance makes sense.
                    resolved_host, _port = await self._loop.getnameinfo(
                        address, _NAME_SOCKET_FLAGS
                    )
                    port = int(_port)
                else:
                    resolved_host, port = address[:2]
            else:  # IPv4
                assert family == socket.AF_INET
                resolved_host, port = address  # type: ignore[misc]
            hosts.append(
                ResolveResult(
                    hostname=host,
                    host=resolved_host,
                    port=port,
                    family=family,
                    proto=proto,
                    flags=_NUMERIC_SOCKET_FLAGS,
                )
            )

        return hosts

    async def close(self) -> None:
        pass


class AsyncResolver(AbstractResolver):
    """Use the `aiodns` package to make asynchronous DNS lookups"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if aiodns is None:
            raise RuntimeError("Resolver requires aiodns library")

        self._loop = loop or asyncio.get_running_loop()
        self._manager: _DNSResolverManager | None = None
        # If custom args are provided, create a dedicated resolver instance
        # This means each AsyncResolver with custom args gets its own
        # aiodns.DNSResolver instance
        if args or kwargs:
            self._resolver = aiodns.DNSResolver(*args, **kwargs)
            return
        # Use the shared resolver from the manager for default arguments
        self._manager = _DNSResolverManager()
        self._resolver = self._manager.get_resolver(self, self._loop)

        if not hasattr(self._resolver, "gethostbyname"):
            # aiodns 1.1 is not available, fallback to DNSResolver.query
            self.resolve = self._resolve_with_query  # type: ignore

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        try:
            try:
                resp = await self._resolver.getaddrinfo(
                    host,
                    port=port,
                    type=socket.SOCK_STREAM,
                    family=family,
                    flags=_AI_ADDRCONFIG,
                )
            except aiodns.error.DNSError:
                if not _is_windows_localhost(host):
                    raise
                resp = await self._resolver.getaddrinfo(
                    host,
                    port=port,
                    type=socket.SOCK_STREAM,
                    family=family,
                    flags=0,
                )
        except aiodns.error.DNSError as exc:
            msg = exc.args[1] if len(exc.args) >= 1 else "DNS lookup failed"
            raise OSError(None, msg) from exc
        hosts: list[ResolveResult] = []
        for node in resp.nodes:
            address: tuple[bytes, int] | tuple[bytes, int, int, int] = node.addr
            if node.family == socket.AF_INET6:
                if len(address) > 3 and address[3]:
                    # This is essential for link-local IPv6 addresses.
                    # LL IPv6 is a VERY rare case. Strictly speaking, we should use
                    # getnameinfo() unconditionally, but performance makes sense.
                    result = await self._resolver.getnameinfo(
                        (address[0].decode("ascii"), *address[1:]),
                        _NAME_SOCKET_FLAGS,
                    )
                    resolved_host = result.node
                else:
                    resolved_host = address[0].decode("ascii")
                    port = address[1]
            else:  # IPv4
                assert node.family == socket.AF_INET
                resolved_host = address[0].decode("ascii")
                port = address[1]
            hosts.append(
                ResolveResult(
                    hostname=host,
                    host=resolved_host,
                    port=port,
                    family=node.family,
                    proto=0,
                    flags=_NUMERIC_SOCKET_FLAGS,
                )
            )

        if not hosts:
            raise OSError(None, "DNS lookup failed")

        return hosts

    async def _resolve_with_query(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        qtype: Final = "AAAA" if family == socket.AF_INET6 else "A"

        try:
            resp = await self._resolver.query(host, qtype)
        except aiodns.error.DNSError as exc:
            msg = exc.args[1] if len(exc.args) >= 1 else "DNS lookup failed"
            raise OSError(None, msg) from exc

        hosts = []
        for rr in resp:
            hosts.append(
                {
                    "hostname": host,
                    "host": rr.host,
                    "port": port,
                    "family": family,
                    "proto": 0,
                    "flags": socket.AI_NUMERICHOST,
                }
            )

        if not hosts:
            raise OSError(None, "DNS lookup failed")

        return hosts

    async def close(self) -> None:
        if self._manager:
            # Release the resolver from the manager if using the shared resolver
            self._manager.release_resolver(self, self._loop)
            self._manager = None  # Clear reference to manager
            self._resolver = None  # type: ignore[assignment] # Clear reference to resolver
            return
        # Otherwise cancel our dedicated resolver
        if self._resolver is not None:
            self._resolver.cancel()
        self._resolver = None  # type: ignore[assignment] # Clear reference


class _DNSResolverManager:
    """Manager for aiodns.DNSResolver objects.

    This class manages shared aiodns.DNSResolver instances
    with no custom arguments across different event loops.
    """

    _instance: Optional["_DNSResolverManager"] = None

    def __new__(cls) -> "_DNSResolverManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # Use WeakKeyDictionary to allow event loops to be garbage collected
        self._loop_data: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            tuple[aiodns.DNSResolver, weakref.WeakSet[AsyncResolver]],
        ] = weakref.WeakKeyDictionary()

    def get_resolver(
        self, client: "AsyncResolver", loop: asyncio.AbstractEventLoop
    ) -> "aiodns.DNSResolver":
        """Get or create the shared aiodns.DNSResolver instance for a specific event loop.

        Args:
            client: The AsyncResolver instance requesting the resolver.
                   This is required to track resolver usage.
            loop: The event loop to use for the resolver.
        """
        # Create a new resolver and client set for this loop if it doesn't exist
        if loop not in self._loop_data:
            resolver = aiodns.DNSResolver(loop=loop)
            client_set: weakref.WeakSet[AsyncResolver] = weakref.WeakSet()
            self._loop_data[loop] = (resolver, client_set)
        else:
            # Get the existing resolver and client set
            resolver, client_set = self._loop_data[loop]

        # Register this client with the loop
        client_set.add(client)
        return resolver

    def release_resolver(
        self, client: "AsyncResolver", loop: asyncio.AbstractEventLoop
    ) -> None:
        """Release the resolver for an AsyncResolver client when it's closed.

        Args:
            client: The AsyncResolver instance to release.
            loop: The event loop the resolver was using.
        """
        # Remove client from its loop's tracking
        current_loop_data = self._loop_data.get(loop)
        if current_loop_data is None:
            return
        resolver, client_set = current_loop_data
        client_set.discard(client)
        # If no more clients for this loop, cancel and remove its resolver
        if not client_set:
            if resolver is not None:
                resolver.cancel()
            del self._loop_data[loop]


_DefaultType = type[AsyncResolver | ThreadedResolver]
DefaultResolver: _DefaultType = AsyncResolver if aiodns_default else ThreadedResolver
<|endoftext|>import asyncio
import logging
import warnings
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import lru_cache, partial, update_wrapper
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast, overload

from aiosignal import Signal
from frozenlist import FrozenList

from . import hdrs
from .abc import (
    AbstractAccessLogger,
    AbstractMatchInfo,
    AbstractRouter,
    AbstractStreamWriter,
)
from .helpers import DEBUG, AppKey
from .http_parser import RawRequestMessage
from .log import web_logger
from .streams import StreamReader
from .typedefs import Handler, Middleware
from .web_exceptions import NotAppKeyWarning
from .web_log import AccessLogger
from .web_middlewares import _fix_request_current_app
from .web_protocol import RequestHandler
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_server import Server
from .web_urldispatcher import (
    AbstractResource,
    AbstractRoute,
    Domain,
    MaskDomain,
    MatchedSubAppResource,
    PrefixedSubAppResource,
    SystemRoute,
    UrlDispatcher,
)

__all__ = ("Application", "CleanupError")


if TYPE_CHECKING:
    _AppSignal = Signal["Application"]
    _RespPrepareSignal = Signal[Request, StreamResponse]
    _Middlewares = FrozenList[Middleware]
    _MiddlewaresHandlers = Optional[Sequence[tuple[Middleware, bool]]]
    _Subapps = list["Application"]
else:
    # No type checker mode, skip types
    _AppSignal = Signal
    _RespPrepareSignal = Signal
    _Middlewares = FrozenList
    _MiddlewaresHandlers = Optional[Sequence]
    _Subapps = list

_T = TypeVar("_T")
_U = TypeVar("_U")
_Resource = TypeVar("_Resource", bound=AbstractResource)


def _build_middlewares(
    handler: Handler, apps: tuple["Application", ...]
) -> Callable[[Request], Awaitable[StreamResponse]]:
    """Apply middlewares to handler."""
    for app in apps[::-1]:
        for m, _ in app._middlewares_handlers:  # type: ignore[union-attr]
            handler = update_wrapper(partial(m, handler=handler), handler)
    return handler


_cached_build_middleware = lru_cache(maxsize=1024)(_build_middlewares)


class Application(MutableMapping[str | AppKey[Any], Any]):
    ATTRS = frozenset(
        [
            "logger",
            "_debug",
            "_router",
            "_loop",
            "_handler_args",
            "_middlewares",
            "_middlewares_handlers",
            "_has_legacy_middlewares",
            "_run_middlewares",
            "_state",
            "_frozen",
            "_pre_frozen",
            "_subapps",
            "_on_response_prepare",
            "_on_startup",
            "_on_shutdown",
            "_on_cleanup",
            "_client_max_size",
            "_client_max_fields",
            "_cleanup_ctx",
        ]
    )

    def __init__(
        self,
        *,
        logger: logging.Logger = web_logger,
        router: UrlDispatcher | None = None,
        middlewares: Iterable[Middleware] = (),
        handler_args: Mapping[str, Any] | None = None,
        client_max_size: int = 1024**2,
        client_max_fields: int = 1000,
        loop: asyncio.AbstractEventLoop | None = None,
        debug: Any = ...,  # mypy doesn't support ellipsis
    ) -> None:
        if router is None:
            router = UrlDispatcher()
        else:
            warnings.warn(
                "router argument is deprecated", DeprecationWarning, stacklevel=2
            )
        assert isinstance(router, AbstractRouter), router

        if loop is not None:
            warnings.warn(
                "loop argument is deprecated", DeprecationWarning, stacklevel=2
            )

        if debug is not ...:
            warnings.warn(
                "debug argument is deprecated", DeprecationWarning, stacklevel=2
            )
        self._debug = debug
        self._router: UrlDispatcher = router
        self._loop = loop
        self._handler_args = handler_args
        self.logger = logger

        self._middlewares: _Middlewares = FrozenList(middlewares)

        # initialized on freezing
        self._middlewares_handlers: _MiddlewaresHandlers = None
        # initialized on freezing
        self._run_middlewares: bool | None = None
        self._has_legacy_middlewares: bool = True

        self._state: dict[AppKey[Any] | str, object] = {}
        self._frozen = False
        self._pre_frozen = False
        self._subapps: _Subapps = []

        self._on_response_prepare: _RespPrepareSignal = Signal(self)
        self._on_startup: _AppSignal = Signal(self)
        self._on_shutdown: _AppSignal = Signal(self)
        self._on_cleanup: _AppSignal = Signal(self)
        self._cleanup_ctx = CleanupContext()
        self._on_startup.append(self._cleanup_ctx._on_startup)
        self._on_cleanup.append(self._cleanup_ctx._on_cleanup)
        self._client_max_size = client_max_size
        self._client_max_fields = client_max_fields

    def __init_subclass__(cls: type["Application"]) -> None:
        warnings.warn(
            f"Inheritance class {cls.__name__} from web.Application is discouraged",
            DeprecationWarning,
            stacklevel=3,
        )

    if DEBUG:  # pragma: no cover

        def __setattr__(self, name: str, val: Any) -> None:
            if name not in self.ATTRS:
                warnings.warn(
                    f"Setting custom web.Application.{name} attribute "
                    "is discouraged",
                    DeprecationWarning,
                    stacklevel=2,
                )
            super().__setattr__(name, val)

    # MutableMapping API

    def __eq__(self, other: object) -> bool:
        return self is other

    @overload  # type: ignore[override]
    def __getitem__(self, key: AppKey[_T]) -> _T: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str | AppKey[_T]) -> Any:
        return self._state[key]

    def _check_frozen(self) -> None:
        if self._frozen:
            warnings.warn(
                "Changing state of started or joined application is deprecated",
                DeprecationWarning,
                stacklevel=3,
            )

    @overload  # type: ignore[override]
    def __setitem__(self, key: AppKey[_T], value: _T) -> None: ...

    @overload
    def __setitem__(self, key: str, value: Any) -> None: ...

    def __setitem__(self, key: str | AppKey[_T], value: Any) -> None:
        self._check_frozen()
        if not isinstance(key, AppKey):
            warnings.warn(
                "It is recommended to use web.AppKey instances for keys.\n"
                + "https://docs.aiohttp.org/en/stable/web_advanced.html"
                + "#application-s-config",
                category=NotAppKeyWarning,
                stacklevel=2,
            )
        self._state[key] = value

    def __delitem__(self, key: str | AppKey[_T]) -> None:
        self._check_frozen()
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self) -> Iterator[str | AppKey[Any]]:
        return iter(self._state)

    def __hash__(self) -> int:
        return id(self)

    @overload  # type: ignore[override]
    def get(self, key: AppKey[_T], default: None = ...) -> _T | None: ...

    @overload
    def get(self, key: AppKey[_T], default: _U) -> _T | _U: ...

    @overload
    def get(self, key: str, default: Any = ...) -> Any: ...

    def get(self, key: str | AppKey[_T], default: Any = None) -> Any:
        return self._state.get(key, default)

    ########
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        # Technically the loop can be None
        # but we mask it by explicit type cast
        # to provide more convenient type annotation
        warnings.warn("loop property is deprecated", DeprecationWarning, stacklevel=2)
        return cast(asyncio.AbstractEventLoop, self._loop)

    def _set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        if loop is None:
            loop = asyncio.get_event_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(
                "web.Application instance initialized with different loop"
            )

        self._loop = loop

        # set loop debug
        if self._debug is ...:
            self._debug = loop.get_debug()

        # set loop to sub applications
        for subapp in self._subapps:
            subapp._set_loop(loop)

    @property
    def pre_frozen(self) -> bool:
        return self._pre_frozen

    def pre_freeze(self) -> None:
        if self._pre_frozen:
            return

        self._pre_frozen = True
        self._middlewares.freeze()
        self._router.freeze()
        self._on_response_prepare.freeze()
        self._cleanup_ctx.freeze()
        self._on_startup.freeze()
        self._on_shutdown.freeze()
        self._on_cleanup.freeze()
        self._middlewares_handlers = tuple(self._prepare_middleware())
        self._has_legacy_middlewares = any(
            not new_style for _, new_style in self._middlewares_handlers
        )

        # If current app and any subapp do not have middlewares avoid run all
        # of the code footprint that it implies, which have a middleware
        # hardcoded per app that sets up the current_app attribute. If no
        # middlewares are configured the handler will receive the proper
        # current_app without needing all of this code.
        self._run_middlewares = True if self.middlewares else False

        for subapp in self._subapps:
            subapp.pre_freeze()
            self._run_middlewares = self._run_middlewares or subapp._run_middlewares

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        if self._frozen:
            return

        self.pre_freeze()
        self._frozen = True
        for subapp in self._subapps:
            subapp.freeze()

    @property
    def debug(self) -> bool:
        warnings.warn("debug property is deprecated", DeprecationWarning, stacklevel=2)
        return self._debug  # type: ignore[no-any-return]

    def _reg_subapp_signals(self, subapp: "Application") -> None:
        def reg_handler(signame: str) -> None:
            subsig = getattr(subapp, signame)

            async def handler(app: "Application") -> None:
                await subsig.send(subapp)

            appsig = getattr(self, signame)
            appsig.append(handler)

        reg_handler("on_startup")
        reg_handler("on_shutdown")
        reg_handler("on_cleanup")

    def add_subapp(self, prefix: str, subapp: "Application") -> PrefixedSubAppResource:
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be str")
        prefix = prefix.rstrip("/")
        if not prefix:
            raise ValueError("Prefix cannot be empty")
        factory = partial(PrefixedSubAppResource, prefix, subapp)
        return self._add_subapp(factory, subapp)

    def _add_subapp(
        self, resource_factory: Callable[[], _Resource], subapp: "Application"
    ) -> _Resource:
        if self.frozen:
            raise RuntimeError("Cannot add sub application to frozen application")
        if subapp.frozen:
            raise RuntimeError("Cannot add frozen application")
        resource = resource_factory()
        self.router.register_resource(resource)
        self._reg_subapp_signals(subapp)
        self._subapps.append(subapp)
        subapp.pre_freeze()
        if self._loop is not None:
            subapp._set_loop(self._loop)
        return resource

    def add_domain(self, domain: str, subapp: "Application") -> MatchedSubAppResource:
        if not isinstance(domain, str):
            raise TypeError("Domain must be str")
        elif "*" in domain:
            rule: Domain = MaskDomain(domain)
        else:
            rule = Domain(domain)
        factory = partial(MatchedSubAppResource, rule, subapp)
        return self._add_subapp(factory, subapp)

    def add_routes(self, routes: Iterable[AbstractRouteDef]) -> list[AbstractRoute]:
        return self.router.add_routes(routes)

    @property
    def on_response_prepare(self) -> _RespPrepareSignal:
        return self._on_response_prepare

    @property
    def on_startup(self) -> _AppSignal:
        return self._on_startup

    @property
    def on_shutdown(self) -> _AppSignal:
        return self._on_shutdown

    @property
    def on_cleanup(self) -> _AppSignal:
        return self._on_cleanup

    @property
    def cleanup_ctx(self) -> "CleanupContext":
        return self._cleanup_ctx

    @property
    def router(self) -> UrlDispatcher:
        return self._router

    @property
    def middlewares(self) -> _Middlewares:
        return self._middlewares

    def _make_handler(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        access_log_class: type[AbstractAccessLogger] = AccessLogger,
        **kwargs: Any,
    ) -> Server:

        if not issubclass(access_log_class, AbstractAccessLogger):
            raise TypeError(
                "access_log_class must be subclass of "
                f"aiohttp.abc.AbstractAccessLogger, got {access_log_class}"
            )

        self._set_loop(loop)
        self.freeze()

        kwargs["debug"] = self._debug
        kwargs["access_log_class"] = access_log_class
        if self._handler_args:
            for k, v in self._handler_args.items():
                kwargs[k] = v

        return Server(
            self._handle,  # type: ignore[arg-type]
            request_factory=self._make_request,
            loop=self._loop,
            **kwargs,
        )

    def make_handler(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        access_log_class: type[AbstractAccessLogger] = AccessLogger,
        **kwargs: Any,
    ) -> Server:

        warnings.warn(
            "Application.make_handler(...) is deprecated, use AppRunner API instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return self._make_handler(
            loop=loop, access_log_class=access_log_class, **kwargs
        )

    async def startup(self) -> None:
        """Causes on_startup signal

        Should be called in the event loop along with the request handler.
        """
        await self.on_startup.send(self)

    async def shutdown(self) -> None:
        """Causes on_shutdown signal

        Should be called before cleanup()
        """
        await self.on_shutdown.send(self)

    async def cleanup(self) -> None:
        """Causes on_cleanup signal

        Should be called after shutdown()
        """
        if self.on_cleanup.frozen:
            await self.on_cleanup.send(self)
        else:
            # If an exception occurs in startup, ensure cleanup contexts are completed.
            await self._cleanup_ctx._on_cleanup(self)

    def _make_request(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: RequestHandler,
        writer: AbstractStreamWriter,
        task: "asyncio.Task[None]",
        _cls: type[Request] = Request,
    ) -> Request:
        if TYPE_CHECKING:
            assert self._loop is not None
        return _cls(
            message,
            payload,
            protocol,
            writer,
            task,
            self._loop,
            client_max_size=self._client_max_size,
            client_max_fields=self._client_max_fields,
        )

    def _prepare_middleware(self) -> Iterator[tuple[Middleware, bool]]:
        for m in reversed(self._middlewares):
            if getattr(m, "__middleware_version__", None) == 1:
                yield m, True
            else:
                warnings.warn(
                    f'old-style middleware "{m!r}" deprecated, see #2252',
                    DeprecationWarning,
                    stacklevel=2,
                )
                yield m, False

        yield _fix_request_current_app(self), True

    async def _handle(self, request: Request) -> StreamResponse:
        loop = asyncio.get_event_loop()
        debug = loop.get_debug()
        match_info = await self._router.resolve(request)
        if debug:  # pragma: no cover
            if not isinstance(match_info, AbstractMatchInfo):
                raise TypeError(
                    "match_info should be AbstractMatchInfo "
                    f"instance, not {match_info!r}"
                )
        match_info.add_app(self)

        match_info.freeze()

        request._match_info = match_info

        if request.headers.get(hdrs.EXPECT):
            resp = await match_info.expect_handler(request)
            await request.writer.drain()
            if resp is not None:
                return resp

        handler = match_info.handler

        if self._run_middlewares:
            # If its a SystemRoute, don't cache building the middlewares since
            # they are constructed for every MatchInfoError as a new handler
            # is made each time.
            if not self._has_legacy_middlewares and not isinstance(
                match_info.route, SystemRoute
            ):
                handler = _cached_build_middleware(handler, match_info.apps)
            else:
                for app in match_info.apps[::-1]:
                    for m, new_style in app._middlewares_handlers:  # type: ignore[union-attr]
                        if new_style:
                            handler = update_wrapper(
                                partial(m, handler=handler), handler
                            )
                        else:
                            handler = await m(app, handler)  # type: ignore[arg-type,assignment]

        return await handler(request)

    def __call__(self) -> "Application":
        """gunicorn compatibility"""
        return self

    def __repr__(self) -> str:
        return f"<Application 0x{id(self):x}>"

    def __bool__(self) -> bool:
        return True


class CleanupError(RuntimeError):
    @property
    def exceptions(self) -> list[BaseException]:
        return cast(list[BaseException], self.args[1])


_CleanupContextCallable = (
    Callable[[Application], AbstractAsyncContextManager[None]]
    | Callable[[Application], AsyncIterator[None]]
)


class CleanupContext(FrozenList[_CleanupContextCallable]):
    def __init__(self) -> None:
        super().__init__()
        self._exits: list[AbstractAsyncContextManager[None]] = []

    async def _on_startup(self, app: Application) -> None:
        for cb in self:
            ctx = cb(app)

            if not isinstance(ctx, AbstractAsyncContextManager):
                ctx = asynccontextmanager(cb)(app)  # type: ignore[arg-type]

            await ctx.__aenter__()
            self._exits.append(ctx)

    async def _on_cleanup(self, app: Application) -> None:
        errors = []
        for it in reversed(self._exits):
            try:
                await it.__aexit__(None, None, None)
            except (Exception, asyncio.CancelledError) as exc:
                errors.append(exc)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            else:
                raise CleanupError("Multiple errors on cleanup stage", errors)
<|endoftext|>"""WebSocket client for asyncio."""

import asyncio
import sys
from asyncio.base_events import BaseEventLoop
from collections.abc import Callable
from types import TracebackType
from typing import Any, Generic, Literal, Optional, cast, overload

import attr

from ._websocket.reader import WebSocketDataQueue, WebSocketReader
from .client_exceptions import ClientError, ServerTimeoutError, WSMessageTypeError
from .client_reqrep import ClientResponse
from .helpers import calculate_timeout_when, set_result
from .http import (
    WS_CLOSED_MESSAGE,
    WS_CLOSING_MESSAGE,
    WebSocketError,
    WSCloseCode,
    WSMessage,
    WSMessageDecodeText,
    WSMessageNoDecodeText,
    WSMsgType,
)
from .http_websocket import _INTERNAL_RECEIVE_TYPES, WebSocketWriter
from .streams import EofStream
from .typedefs import (
    DEFAULT_JSON_DECODER,
    DEFAULT_JSON_ENCODER,
    JSONBytesEncoder,
    JSONDecoder,
    JSONEncoder,
)

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 11):
    import asyncio as async_timeout
    from typing import Self
else:
    import async_timeout
    from typing_extensions import Self

# TypeVar for whether text messages are decoded to str (True) or kept as bytes (False)
# Covariant because it only affects return types, not input types
_DecodeText = TypeVar("_DecodeText", bound=bool, covariant=True, default=Literal[True])


@attr.s(frozen=True, slots=True)
class ClientWSTimeout:
    ws_receive = attr.ib(type=Optional[float], default=None)
    ws_close = attr.ib(type=Optional[float], default=None)


DEFAULT_WS_CLIENT_TIMEOUT = ClientWSTimeout(ws_receive=None, ws_close=10.0)


class ClientWebSocketResponse(Generic[_DecodeText]):
    def __init__(
        self,
        reader: WebSocketDataQueue,
        writer: WebSocketWriter,
        protocol: str | None,
        response: ClientResponse,
        timeout: ClientWSTimeout,
        autoclose: bool,
        autoping: bool,
        loop: asyncio.AbstractEventLoop,
        *,
        heartbeat: float | None = None,
        compress: int = 0,
        client_notakeover: bool = False,
    ) -> None:
        self._response = response
        self._conn = response.connection

        self._writer = writer
        self._reader = reader
        # Set by ClientSession._ws_connect; owns the parser so a stalled
        # reader parked on the queue by weakref stays alive while this
        # response can still be drained.
        self._parser: WebSocketReader | None = None
        self._protocol = protocol
        self._closed = False
        self._closing = False
        self._close_code: int | None = None
        self._timeout = timeout
        self._autoclose = autoclose
        self._autoping = autoping
        self._heartbeat = heartbeat
        self._heartbeat_cb: asyncio.TimerHandle | None = None
        self._heartbeat_when: float = 0.0
        if heartbeat is not None:
            self._pong_heartbeat = heartbeat / 2.0
        self._pong_response_cb: asyncio.TimerHandle | None = None
        self._loop = loop
        self._waiting: bool = False
        self._close_wait: asyncio.Future[None] | None = None
        self._exception: BaseException | None = None
        self._compress = compress
        self._client_notakeover = client_notakeover
        self._ping_task: asyncio.Task[None] | None = None
        self._need_heartbeat_reset = False
        self._heartbeat_reset_handle: asyncio.Handle | None = None

        self._reset_heartbeat()

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
        loop = self._loop
        assert loop is not None
        conn = self._conn
        timeout_ceil_threshold = (
            conn._connector._timeout_ceil_threshold if conn is not None else 5
        )
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
        now = loop.time()
        if now < self._heartbeat_when:
            # Heartbeat fired too early, reschedule
            self._heartbeat_cb = loop.call_at(
                self._heartbeat_when, self._send_heartbeat
            )
            return

        conn = self._conn
        timeout_ceil_threshold = (
            conn._connector._timeout_ceil_threshold if conn is not None else 5
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
        self._handle_ping_pong_exception(
            ServerTimeoutError(f"No PONG received after {self._pong_heartbeat} seconds")
        )

    def _handle_ping_pong_exception(self, exc: BaseException) -> None:
        """Handle exceptions raised during ping/pong processing."""
        if self._closed:
            return
        self._set_closed()
        # close() is never reached after this; release the parser here.
        self._parser = None
        self._close_code = WSCloseCode.ABNORMAL_CLOSURE
        self._exception = exc
        self._response.close()
        if self._waiting and not self._closing:
            self._reader.feed_data(WSMessage(WSMsgType.ERROR, exc, None), 0)

    def _set_closed(self) -> None:
        """Set the connection to closed.

        Cancel any heartbeat timers and set the closed flag.
        """
        self._closed = True
        self._cancel_heartbeat()

    def _set_closing(self) -> None:
        """Set the connection to closing.

        Cancel any heartbeat timers and set the closing flag.
        """
        self._closing = True
        self._cancel_heartbeat()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_code(self) -> int | None:
        return self._close_code

    @property
    def protocol(self) -> str | None:
        return self._protocol

    @property
    def compress(self) -> int:
        return self._compress

    @property
    def client_notakeover(self) -> bool:
        return self._client_notakeover

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """extra info from connection transport"""
        conn = self._response.connection
        if conn is None:
            return default
        transport = conn.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def exception(self) -> BaseException | None:
        return self._exception

    async def ping(self, message: bytes = b"") -> None:
        await self._writer.send_frame(message, WSMsgType.PING)

    async def pong(self, message: bytes = b"") -> None:
        await self._writer.send_frame(message, WSMsgType.PONG)

    async def send_frame(
        self, message: bytes, opcode: WSMsgType, compress: int | None = None
    ) -> None:
        """Send a frame over the websocket."""
        await self._writer.send_frame(message, opcode, compress)

    async def send_str(self, data: str, compress: int | None = None) -> None:
        if not isinstance(data, str):
            raise TypeError("data argument must be str (%r)" % type(data))
        await self._writer.send_frame(
            data.encode("utf-8"), WSMsgType.TEXT, compress=compress
        )

    async def send_bytes(self, data: bytes, compress: int | None = None) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data argument must be byte-ish (%r)" % type(data))
        await self._writer.send_frame(data, WSMsgType.BINARY, compress=compress)

    async def send_json(
        self,
        data: Any,
        compress: int | None = None,
        *,
        dumps: JSONEncoder = DEFAULT_JSON_ENCODER,
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

    async def close(self, *, code: int = WSCloseCode.OK, message: bytes = b"") -> bool:
        # we need to break `receive()` cycle first,
        # `close()` may be called from different task
        if self._waiting and not self._closing:
            assert self._loop is not None
            self._close_wait = self._loop.create_future()
            self._set_closing()
            self._reader.feed_data(WS_CLOSING_MESSAGE, 0)
            await self._close_wait

        if self._closed:
            return False

        self._set_closed()
        try:
            try:
                await self._writer.close(code, message)
            except asyncio.CancelledError:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._response.close()
                raise
            except Exception as exc:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._exception = exc
                self._response.close()
                return True

            if self._close_code:
                self._response.close()
                return True

            while True:
                try:
                    async with async_timeout.timeout(self._timeout.ws_close):
                        msg = await self._reader.read()
                except asyncio.CancelledError:
                    self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                    self._response.close()
                    raise
                except Exception as exc:
                    self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                    self._exception = exc
                    self._response.close()
                    return True

                if msg.type is WSMsgType.CLOSE:
                    self._close_code = msg.data
                    self._response.close()
                    return True
        finally:
            # Once closed the response can no longer be drained; release the
            # parser and the stash it retains.
            self._parser = None

    @overload
    async def receive(
        self: "ClientWebSocketResponse[Literal[True]]", timeout: float | None = None
    ) -> WSMessageDecodeText: ...

    @overload
    async def receive(
        self: "ClientWebSocketResponse[Literal[False]]", timeout: float | None = None
    ) -> WSMessageNoDecodeText: ...

    @overload
    async def receive(
        self: "ClientWebSocketResponse[_DecodeText]", timeout: float | None = None
    ) -> WSMessageDecodeText | WSMessageNoDecodeText: ...

    async def receive(
        self, timeout: float | None = None
    ) -> WSMessageDecodeText | WSMessageNoDecodeText:
        receive_timeout = timeout or self._timeout.ws_receive

        while True:
            if self._waiting:
                raise RuntimeError("Concurrent call to receive() is not allowed")

            if self._closed:
                return WS_CLOSED_MESSAGE
            elif self._closing:
                await self.close()
                return WS_CLOSED_MESSAGE

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
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                raise
            except EofStream:
                self._close_code = WSCloseCode.OK
                await self.close()
                return WSMessage(WSMsgType.CLOSED, None, None)
            except ClientError:
                # Likely ServerDisconnectedError when connection is lost.
                # close() is not called on this path, so release the parser
                # and the stash it retains here.
                self._parser = None
                self._set_closed()
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                return WS_CLOSED_MESSAGE
            except WebSocketError as exc:
                self._close_code = exc.code
                await self.close(code=exc.code)
                return WSMessage(WSMsgType.ERROR, exc, None)
            except Exception as exc:
                self._exception = exc
                self._set_closing()
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                await self.close()
                return WSMessage(WSMsgType.ERROR, exc, None)

            if msg.type not in _INTERNAL_RECEIVE_TYPES:
                # If its not a close/closing/ping/pong message
                # we can return it immediately
                return msg

            if msg.type is WSMsgType.CLOSE:
                self._set_closing()
                self._close_code = msg.data
                if not self._closed and self._autoclose:
                    await self.close()
            elif msg.type is WSMsgType.CLOSING:
                self._set_closing()
            elif msg.type is WSMsgType.PING and self._autoping:
                await self.pong(msg.data)
                continue
            elif msg.type is WSMsgType.PONG and self._autoping:
                continue

            return msg

    @overload
    async def receive_str(
        self: "ClientWebSocketResponse[Literal[True]]", *, timeout: float | None = None
    ) -> str: ...

    @overload
    async def receive_str(
        self: "ClientWebSocketResponse[Literal[False]]", *, timeout: float | None = None
    ) -> bytes: ...

    @overload
    async def receive_str(
        self: "ClientWebSocketResponse[_DecodeText]", *, timeout: float | None = None
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
        self: "ClientWebSocketResponse[Literal[True]]",
        *,
        loads: JSONDecoder = ...,
        timeout: float | None = None,
    ) -> Any: ...

    @overload
    async def receive_json(
        self: "ClientWebSocketResponse[Literal[False]]",
        *,
        loads: Callable[[bytes], Any] = ...,
        timeout: float | None = None,
    ) -> Any: ...

    @overload
    async def receive_json(
        self: "ClientWebSocketResponse[_DecodeText]",
        *,
        loads: JSONDecoder | Callable[[bytes], Any] = ...,
        timeout: float | None = None,
    ) -> Any: ...

    async def receive_json(
        self,
        *,
        loads: JSONDecoder | Callable[[bytes], Any] = DEFAULT_JSON_DECODER,
        timeout: float | None = None,
    ) -> Any:
        data = await self.receive_str(timeout=timeout)
        return loads(data)  # type: ignore[arg-type]

    def __aiter__(self) -> Self:
        return self

    @overload
    async def __anext__(
        self: "ClientWebSocketResponse[Literal[True]]",
    ) -> WSMessageDecodeText: ...

    @overload
    async def __anext__(
        self: "ClientWebSocketResponse[Literal[False]]",
    ) -> WSMessageNoDecodeText: ...

    @overload
    async def __anext__(
        self: "ClientWebSocketResponse[_DecodeText]",
    ) -> WSMessageDecodeText | WSMessageNoDecodeText: ...

    async def __anext__(self) -> WSMessageDecodeText | WSMessageNoDecodeText:
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
<|endoftext|>import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa

from yarl import URL

from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response

__all__ = (
    "HTTPException",
    "HTTPError",
    "HTTPRedirection",
    "HTTPSuccessful",
    "HTTPOk",
    "HTTPCreated",
    "HTTPAccepted",
    "HTTPNonAuthoritativeInformation",
    "HTTPNoContent",
    "HTTPResetContent",
    "HTTPPartialContent",
    "HTTPMove",
    "HTTPMultipleChoices",
    "HTTPMovedPermanently",
    "HTTPFound",
    "HTTPSeeOther",
    "HTTPNotModified",
    "HTTPUseProxy",
    "HTTPTemporaryRedirect",
    "HTTPPermanentRedirect",
    "HTTPClientError",
    "HTTPBadRequest",
    "HTTPUnauthorized",
    "HTTPPaymentRequired",
    "HTTPForbidden",
    "HTTPNotFound",
    "HTTPMethodNotAllowed",
    "HTTPNotAcceptable",
    "HTTPProxyAuthenticationRequired",
    "HTTPRequestTimeout",
    "HTTPConflict",
    "HTTPGone",
    "HTTPLengthRequired",
    "HTTPPreconditionFailed",
    "HTTPRequestEntityTooLarge",
    "HTTPRequestURITooLong",
    "HTTPUnsupportedMediaType",
    "HTTPRequestRangeNotSatisfiable",
    "HTTPExpectationFailed",
    "HTTPMisdirectedRequest",
    "HTTPUnprocessableEntity",
    "HTTPFailedDependency",
    "HTTPUpgradeRequired",
    "HTTPPreconditionRequired",
    "HTTPTooManyRequests",
    "HTTPRequestHeaderFieldsTooLarge",
    "HTTPUnavailableForLegalReasons",
    "HTTPServerError",
    "HTTPInternalServerError",
    "HTTPNotImplemented",
    "HTTPBadGateway",
    "HTTPServiceUnavailable",
    "HTTPGatewayTimeout",
    "HTTPVersionNotSupported",
    "HTTPVariantAlsoNegotiates",
    "HTTPInsufficientStorage",
    "HTTPNotExtended",
    "HTTPNetworkAuthenticationRequired",
)


class NotAppKeyWarning(UserWarning):
    """Warning when not using AppKey in Application."""


############################################################
# HTTP Exceptions
############################################################


class HTTPException(Response, Exception):

    # You should set in subclasses:
    # status = 200

    status_code = -1
    empty_body = False

    __http_exception__ = True

    def __init__(
        self,
        *,
        headers: LooseHeaders | None = None,
        reason: str | None = None,
        body: Any = None,
        text: str | None = None,
        content_type: str | None = None,
    ) -> None:
        if body is not None:
            warnings.warn(
                "body argument is deprecated for http web exceptions",
                DeprecationWarning,
            )
        if reason is not None and ("\r" in reason or "\n" in reason):
            raise ValueError("Reason cannot contain \\r or \\n")
        Response.__init__(
            self,
            status=self.status_code,
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        Exception.__init__(self, self.reason)
        if self.body is None and not self.empty_body:
            self.text = f"{self.status}: {self.reason}"

    def __bool__(self) -> bool:
        return True


class HTTPError(HTTPException):
    """Base class for exceptions with status codes in the 400s and 500s."""


class HTTPRedirection(HTTPException):
    """Base class for exceptions with status codes in the 300s."""


class HTTPSuccessful(HTTPException):
    """Base class for exceptions with status codes in the 200s."""


class HTTPOk(HTTPSuccessful):
    status_code = 200


class HTTPCreated(HTTPSuccessful):
    status_code = 201


class HTTPAccepted(HTTPSuccessful):
    status_code = 202


class HTTPNonAuthoritativeInformation(HTTPSuccessful):
    status_code = 203


class HTTPNoContent(HTTPSuccessful):
    status_code = 204
    empty_body = True


class HTTPResetContent(HTTPSuccessful):
    status_code = 205
    empty_body = True


class HTTPPartialContent(HTTPSuccessful):
    status_code = 206


############################################################
# 3xx redirection
############################################################


class HTTPMove(HTTPRedirection):
    def __init__(
        self,
        location: StrOrURL,
        *,
        headers: LooseHeaders | None = None,
        reason: str | None = None,
        body: Any = None,
        text: str | None = None,
        content_type: str | None = None,
    ) -> None:
        if not location:
            raise ValueError("HTTP redirects need a location to redirect to.")
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self.headers["Location"] = str(URL(location))
        self.location = location


class HTTPMultipleChoices(HTTPMove):
    status_code = 300


class HTTPMovedPermanently(HTTPMove):
    status_code = 301


class HTTPFound(HTTPMove):
    status_code = 302


# This one is safe after a POST (the redirected location will be
# retrieved with GET):
class HTTPSeeOther(HTTPMove):
    status_code = 303


class HTTPNotModified(HTTPRedirection):
    # FIXME: this should include a date or etag header
    status_code = 304
    empty_body = True


class HTTPUseProxy(HTTPMove):
    # Not a move, but looks a little like one
    status_code = 305


class HTTPTemporaryRedirect(HTTPMove):
    status_code = 307


class HTTPPermanentRedirect(HTTPMove):
    status_code = 308


############################################################
# 4xx client error
############################################################


class HTTPClientError(HTTPError):
    pass


class HTTPBadRequest(HTTPClientError):
    status_code = 400


class HTTPUnauthorized(HTTPClientError):
    status_code = 401


class HTTPPaymentRequired(HTTPClientError):
    status_code = 402


class HTTPForbidden(HTTPClientError):
    status_code = 403


class HTTPNotFound(HTTPClientError):
    status_code = 404


class HTTPMethodNotAllowed(HTTPClientError):
    status_code = 405

    def __init__(
        self,
        method: str,
        allowed_methods: Iterable[str],
        *,
        headers: LooseHeaders | None = None,
        reason: str | None = None,
        body: Any = None,
        text: str | None = None,
        content_type: str | None = None,
    ) -> None:
        allow = ",".join(sorted(allowed_methods))
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self.headers["Allow"] = allow
        self.allowed_methods: set[str] = set(allowed_methods)
        self.method = method.upper()


class HTTPNotAcceptable(HTTPClientError):
    status_code = 406


class HTTPProxyAuthenticationRequired(HTTPClientError):
    status_code = 407


class HTTPRequestTimeout(HTTPClientError):
    status_code = 408


class HTTPConflict(HTTPClientError):
    status_code = 409


class HTTPGone(HTTPClientError):
    status_code = 410


class HTTPLengthRequired(HTTPClientError):
    status_code = 411


class HTTPPreconditionFailed(HTTPClientError):
    status_code = 412


class HTTPRequestEntityTooLarge(HTTPClientError):
    status_code = 413

    def __init__(self, max_size: float, actual_size: float = 0, **kwargs: Any) -> None:
        kwargs.setdefault("text", f"Maximum request body size {max_size} exceeded.")
        super().__init__(**kwargs)


class HTTPRequestURITooLong(HTTPClientError):
    status_code = 414


class HTTPUnsupportedMediaType(HTTPClientError):
    status_code = 415


class HTTPRequestRangeNotSatisfiable(HTTPClientError):
    status_code = 416


class HTTPExpectationFailed(HTTPClientError):
    status_code = 417


class HTTPMisdirectedRequest(HTTPClientError):
    status_code = 421


class HTTPUnprocessableEntity(HTTPClientError):
    status_code = 422


class HTTPFailedDependency(HTTPClientError):
    status_code = 424


class HTTPUpgradeRequired(HTTPClientError):
    status_code = 426


class HTTPPreconditionRequired(HTTPClientError):
    status_code = 428


class HTTPTooManyRequests(HTTPClientError):
    status_code = 429


class HTTPRequestHeaderFieldsTooLarge(HTTPClientError):
    status_code = 431


class HTTPUnavailableForLegalReasons(HTTPClientError):
    status_code = 451

    def __init__(
        self,
        link: StrOrURL | None,
        *,
        headers: LooseHeaders | None = None,
        reason: str | None = None,
        body: Any = None,
        text: str | None = None,
        content_type: str | None = None,
    ) -> None:
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self._link = None
        if link:
            self._link = URL(link)
            self.headers["Link"] = f'<{str(self._link)}>; rel="blocked-by"'

    @property
    def link(self) -> URL | None:
        return self._link


############################################################
# 5xx Server Error
############################################################
#  Response status codes beginning with the digit "5" indicate cases in
#  which the server is aware that it has erred or is incapable of
#  performing the request. Except when responding to a HEAD request, the
#  server SHOULD include an entity containing an explanation of the error
#  situation, and whether it is a temporary or permanent condition. User
#  agents SHOULD display any included entity to the user. These response
#  codes are applicable to any request method.


class HTTPServerError(HTTPError):
    pass


class HTTPInternalServerError(HTTPServerError):
    status_code = 500


class HTTPNotImplemented(HTTPServerError):
    status_code = 501


class HTTPBadGateway(HTTPServerError):
    status_code = 502


class HTTPServiceUnavailable(HTTPServerError):
    status_code = 503


class HTTPGatewayTimeout(HTTPServerError):
    status_code = 504


class HTTPVersionNotSupported(HTTPServerError):
    status_code = 505


class HTTPVariantAlsoNegotiates(HTTPServerError):
    status_code = 506


class HTTPInsufficientStorage(HTTPServerError):
    status_code = 507


class HTTPNotExtended(HTTPServerError):
    status_code = 510


class HTTPNetworkAuthenticationRequired(HTTPServerError):
    status_code = 511
<|endoftext|>"""Async gunicorn worker for aiohttp.web"""

import asyncio
import inspect
import os
import re
import signal
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional

from gunicorn.config import AccessLogFormat as GunicornAccessLogFormat
from gunicorn.workers import base

from aiohttp import web

from .helpers import set_result
from .web_app import Application
from .web_log import AccessLogger

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


__all__ = ("GunicornWebWorker", "GunicornUVLoopWebWorker")


class GunicornWebWorker(base.Worker):  # type: ignore[misc,no-any-unimported]
    DEFAULT_AIOHTTP_LOG_FORMAT = AccessLogger.LOG_FORMAT
    DEFAULT_GUNICORN_LOG_FORMAT = GunicornAccessLogFormat.default

    def __init__(self, *args: Any, **kw: Any) -> None:  # pragma: no cover
        super().__init__(*args, **kw)

        self._task: asyncio.Task[None] | None = None
        self.exit_code = 0
        self._notify_waiter: asyncio.Future[bool] | None = None

    def init_process(self) -> None:
        # create new event_loop after fork
        try:
            asyncio.get_event_loop().close()
        except RuntimeError:
            # No loop was running
            pass

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        super().init_process()

    def run(self) -> None:
        # base.Worker.init_process() sets self.booted = True before
        # invoking run(), but for the aiohttp worker the real boot work
        # (factory call, runner setup, binding sockets) happens here.
        # Reset until _run() reaches the serve loop so that the arbiter
        # can tell a startup failure from a normal worker exit and
        # halt instead of endlessly respawning workers.
        self.booted = False

        self._task = self.loop.create_task(self._run())
        try:
            self.loop.run_until_complete(self._task)
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

        sys.exit(self.exit_code)

    async def _run(self) -> None:
        runner = None
        if isinstance(self.wsgi, Application):
            app = self.wsgi
        elif inspect.iscoroutinefunction(self.wsgi) or (
            sys.version_info < (3, 14) and asyncio.iscoroutinefunction(self.wsgi)
        ):
            wsgi = await self.wsgi()
            if isinstance(wsgi, web.AppRunner):
                runner = wsgi
                app = runner.app
            else:
                app = wsgi
        else:
            raise RuntimeError(
                "wsgi app should be either Application or "
                f"async function returning Application, got {self.wsgi}"
            )

        if runner is None:
            access_log = self.log.access_log if self.cfg.accesslog else None
            runner = web.AppRunner(
                app,
                logger=self.log,
                keepalive_timeout=self.cfg.keepalive,
                access_log=access_log,
                access_log_format=self._get_valid_log_format(
                    self.cfg.access_log_format
                ),
                shutdown_timeout=self.cfg.graceful_timeout / 100 * 95,
            )
        await runner.setup()

        ctx = self._create_ssl_context(self.cfg) if self.cfg.is_ssl else None

        runner = runner
        assert runner is not None
        server = runner.server
        assert server is not None
        for sock in self.sockets:
            site = web.SockSite(
                runner,
                sock,
                ssl_context=ctx,
            )
            await site.start()

        # Sockets are bound; tell the arbiter the worker is ready to
        # accept requests. Any failure before this point propagates out
        # of run() with self.booted=False so the arbiter exits with
        # WORKER_BOOT_ERROR instead of treating this as a clean exit.
        self.booted = True

        # If our parent changed then we shut down.
        pid = os.getpid()
        try:
            while self.alive:  # type: ignore[has-type]
                self.notify()

                cnt = server.requests_count
                if self.max_requests and cnt > self.max_requests:
                    self.alive = False
                    self.log.info("Max requests, shutting down: %s", self)

                elif pid == os.getpid() and self.ppid != os.getppid():
                    self.alive = False
                    self.log.info("Parent changed, shutting down: %s", self)
                else:
                    await self._wait_next_notify()
        except Exception:
            pass

        await runner.cleanup()

    def _wait_next_notify(self) -> "asyncio.Future[bool]":
        self._notify_waiter_done()

        loop = self.loop
        assert loop is not None
        self._notify_waiter = waiter = loop.create_future()
        self.loop.call_later(1.0, self._notify_waiter_done, waiter)

        return waiter

    def _notify_waiter_done(
        self, waiter: Optional["asyncio.Future[bool]"] = None
    ) -> None:
        if waiter is None:
            waiter = self._notify_waiter
        if waiter is not None:
            set_result(waiter, True)

        if waiter is self._notify_waiter:
            self._notify_waiter = None

    def init_signals(self) -> None:
        # Set up signals through the event loop API.

        self.loop.add_signal_handler(
            signal.SIGQUIT, self.handle_quit, signal.SIGQUIT, None
        )

        self.loop.add_signal_handler(
            signal.SIGTERM, self.handle_exit, signal.SIGTERM, None
        )

        self.loop.add_signal_handler(
            signal.SIGINT, self.handle_quit, signal.SIGINT, None
        )

        self.loop.add_signal_handler(
            signal.SIGWINCH, self.handle_winch, signal.SIGWINCH, None
        )

        self.loop.add_signal_handler(
            signal.SIGUSR1, self.handle_usr1, signal.SIGUSR1, None
        )

        self.loop.add_signal_handler(
            signal.SIGABRT, self.handle_abort, signal.SIGABRT, None
        )

        # Don't let SIGTERM and SIGUSR1 disturb active requests
        # by interrupting system calls
        signal.siginterrupt(signal.SIGTERM, False)
        signal.siginterrupt(signal.SIGUSR1, False)

        # Reset SIGCHLD to default so Gunicorn doesn't swallow subprocess
        # return codes. Without this, workers inherit the master arbiter's
        # SIGCHLD handler, causing spurious "Worker exited" errors when
        # application code spawns subprocesses.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    def handle_quit(self, sig: int, frame: FrameType | None) -> None:
        self.alive = False

        # worker_int callback
        self.cfg.worker_int(self)

        # wakeup closing process
        self._notify_waiter_done()

    def handle_abort(self, sig: int, frame: FrameType | None) -> None:
        self.alive = False
        self.exit_code = 1
        self.cfg.worker_abort(self)
        sys.exit(1)

    @staticmethod
    def _create_ssl_context(cfg: Any) -> "SSLContext":
        """Creates SSLContext instance for usage in asyncio.create_server.

        See ssl.SSLSocket.__init__ for more details.
        """
        if ssl is None:  # pragma: no cover
            raise RuntimeError("SSL is not supported.")

        ctx = ssl.SSLContext(cfg.ssl_version)
        ctx.load_cert_chain(cfg.certfile, cfg.keyfile)
        ctx.verify_mode = cfg.cert_reqs
        if cfg.ca_certs:
            ctx.load_verify_locations(cfg.ca_certs)
        if cfg.ciphers:
            ctx.set_ciphers(cfg.ciphers)
        return ctx

    def _get_valid_log_format(self, source_format: str) -> str:
        if source_format == self.DEFAULT_GUNICORN_LOG_FORMAT:
            return self.DEFAULT_AIOHTTP_LOG_FORMAT
        elif re.search(r"%\([^\)]+\)", source_format):
            raise ValueError(
                "Gunicorn's style options in form of `%(name)s` are not "
                "supported for the log formatting. Please use aiohttp's "
                "format specification to configure access log formatting: "
                "http://docs.aiohttp.org/en/stable/logging.html"
                "#format-specification"
            )
        else:
            return source_format


class GunicornUVLoopWebWorker(GunicornWebWorker):
    def init_process(self) -> None:
        import uvloop

        # Close any existing event loop before setting a
        # new policy.
        try:
            asyncio.get_event_loop().close()
        except RuntimeError:
            # No loop was running
            pass

        # Setup uvloop policy, so that every
        # asyncio.get_event_loop() will create an instance
        # of uvloop event loop.
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        super().init_process()
<|endoftext|>"""Http related parsers and protocol."""

import asyncio
import re
import sys
from typing import (  # noqa
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
)

from multidict import CIMultiDict

from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .client_exceptions import ClientConnectionResetError
from .compression_utils import ZLibCompressor
from .helpers import NO_EXTENSIONS

__all__ = ("StreamWriter", "HttpVersion", "HttpVersion10", "HttpVersion11")

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing import Union

    Buffer = Union[bytes, bytearray, "memoryview[int]", "memoryview[bytes]"]


MIN_PAYLOAD_FOR_WRITELINES = 2048
IS_PY313_BEFORE_313_2 = (3, 13, 0) <= sys.version_info < (3, 13, 2)
IS_PY_BEFORE_312_9 = sys.version_info < (3, 12, 9)
SKIP_WRITELINES = IS_PY313_BEFORE_313_2 or IS_PY_BEFORE_312_9
# writelines is not safe for use
# on Python 3.12+ until 3.12.9
# on Python 3.13+ until 3.13.2
# and on older versions it not any faster than write
# CVE-2024-12254: https://github.com/python/cpython/pull/127656


class HttpVersion(NamedTuple):
    major: int
    minor: int


HttpVersion10 = HttpVersion(1, 0)
HttpVersion11 = HttpVersion(1, 1)


_T_OnChunkSent = Optional[Callable[[Buffer], Awaitable[None]]]
_T_OnHeadersSent = Optional[Callable[["CIMultiDict[str]"], Awaitable[None]]]


class StreamWriter(AbstractStreamWriter):

    length: int | None = None
    chunked: bool = False
    _eof: bool = False
    _compress: ZLibCompressor | None = None

    def __init__(
        self,
        protocol: BaseProtocol,
        loop: asyncio.AbstractEventLoop,
        on_chunk_sent: _T_OnChunkSent = None,
        on_headers_sent: _T_OnHeadersSent = None,
    ) -> None:
        self._protocol = protocol
        self.loop = loop
        self._on_chunk_sent: _T_OnChunkSent = on_chunk_sent
        self._on_headers_sent: _T_OnHeadersSent = on_headers_sent
        self._headers_buf: bytes | None = None
        self._headers_written: bool = False
        self.on_body_write = None

    @property
    def transport(self) -> asyncio.Transport | None:
        return self._protocol.transport

    @property
    def protocol(self) -> BaseProtocol:
        return self._protocol

    def enable_chunking(self) -> None:
        self.chunked = True

    def enable_compression(
        self, encoding: str = "deflate", strategy: int | None = None
    ) -> None:
        self._compress = ZLibCompressor(encoding=encoding, strategy=strategy)

    def _write(self, chunk: Buffer) -> None:
        size = len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self._protocol.transport
        if transport is None or transport.is_closing():
            raise ClientConnectionResetError("Cannot write to closing transport")
        transport.write(chunk)

    def _writelines(self, chunks: Iterable[Buffer]) -> None:
        size = 0
        for chunk in chunks:
            size += len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self._protocol.transport
        if transport is None or transport.is_closing():
            raise ClientConnectionResetError("Cannot write to closing transport")
        if SKIP_WRITELINES or size < MIN_PAYLOAD_FOR_WRITELINES:
            transport.write(b"".join(chunks))
        else:
            transport.writelines(chunks)

    def _write_chunked_payload(self, chunk: Buffer) -> None:
        """Write a chunk with proper chunked encoding."""
        chunk_len_pre = f"{len(chunk):x}\r\n".encode("ascii")
        self._writelines((chunk_len_pre, chunk, b"\r\n"))

    def _send_headers_with_payload(self, chunk: Buffer, is_eof: bool) -> None:
        """Send buffered headers with payload, coalescing into single write."""
        # Mark headers as written
        self._headers_written = True
        headers_buf = self._headers_buf
        self._headers_buf = None

        if TYPE_CHECKING:
            # Safe because callers (write() and write_eof()) only invoke this method
            # after checking that self._headers_buf is truthy
            assert headers_buf is not None

        if not self.chunked:
            # Non-chunked: coalesce headers with body
            if chunk:
                self._writelines((headers_buf, chunk))
            else:
                self._write(headers_buf)
            return

        # Coalesce headers with chunked data
        if chunk:
            chunk_len_pre = f"{len(chunk):x}\r\n".encode("ascii")
            if is_eof:
                self._writelines((headers_buf, chunk_len_pre, chunk, b"\r\n0\r\n\r\n"))
            else:
                self._writelines((headers_buf, chunk_len_pre, chunk, b"\r\n"))
        elif is_eof:
            self._writelines((headers_buf, b"0\r\n\r\n"))
        else:
            self._write(headers_buf)

    async def write(
        self, chunk: Buffer, *, drain: bool = True, LIMIT: int = 0x10000
    ) -> None:
        """
        Writes chunk of data to a stream.

        write_eof() indicates end of stream.
        writer can't be used after write_eof() method being called.
        write() return drain future.
        """
        if self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)

        notify = self.on_body_write
        if isinstance(chunk, memoryview):
            body_size = chunk.nbytes
            if body_size != len(chunk):
                # just reshape it
                chunk = chunk.cast("c")
        else:
            body_size = len(chunk) if notify is not None else 0

        if self._compress is not None:
            chunk = await self._compress.compress(chunk)
            if not chunk:
                if notify is not None and body_size:
                    notify(body_size, self.output_size)
                return

        if self.length is not None:
            chunk_len = len(chunk)
            if self.length >= chunk_len:
                self.length = self.length - chunk_len
            else:
                # Bytes clipped at the declared length are discarded, not sent.
                body_size = self.length
                chunk = chunk[: self.length]
                self.length = 0
                if not chunk:
                    return

        # Handle buffered headers for small payload optimization
        if self._headers_buf and not self._headers_written:
            self._send_headers_with_payload(chunk, False)
        elif chunk:
            if self.chunked:
                self._write_chunked_payload(chunk)
            else:
                self._write(chunk)
        else:
            return

        if notify is not None and body_size:
            notify(body_size, self.output_size)
        if drain and self.buffer_size > LIMIT:
            self.buffer_size = 0
            await self.drain()

    async def write_headers(
        self, status_line: str, headers: "CIMultiDict[str]"
    ) -> None:
        """Write headers to the stream."""
        if self._on_headers_sent is not None:
            await self._on_headers_sent(headers)
        # status + headers
        buf = _serialize_headers(status_line, headers)
        self._headers_written = False
        self._headers_buf = buf

    def send_headers(self) -> None:
        """Force sending buffered headers if not already sent."""
        if not self._headers_buf or self._headers_written:
            return

        self._headers_written = True
        headers_buf = self._headers_buf
        self._headers_buf = None

        if TYPE_CHECKING:
            # Safe because we only enter this block when self._headers_buf is truthy
            assert headers_buf is not None

        self._write(headers_buf)

    def set_eof(self) -> None:
        """Indicate that the message is complete."""
        if self._eof:
            return

        # If headers haven't been sent yet, send them now
        # This handles the case where there's no body at all
        if self._headers_buf and not self._headers_written:
            self._headers_written = True
            headers_buf = self._headers_buf
            self._headers_buf = None

            if TYPE_CHECKING:
                # Safe because we only enter this block when self._headers_buf is truthy
                assert headers_buf is not None

            # Combine headers and chunked EOF marker in a single write
            if self.chunked:
                self._writelines((headers_buf, b"0\r\n\r\n"))
            else:
                self._write(headers_buf)
        elif self.chunked and self._headers_written:
            # Headers already sent, just send the final chunk marker
            self._write(b"0\r\n\r\n")

        self._eof = True

    async def write_eof(self, chunk: bytes = b"") -> None:
        if self._eof:
            return

        if chunk and self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)

        # Handle body/compression
        if self._compress:
            chunks: list[bytes] = []
            chunks_len = 0
            if chunk and (compressed_chunk := await self._compress.compress(chunk)):
                chunks_len = len(compressed_chunk)
                chunks.append(compressed_chunk)

            flush_chunk = self._compress.flush()
            chunks_len += len(flush_chunk)
            chunks.append(flush_chunk)
            assert chunks_len

            # Send buffered headers with compressed data if not yet sent
            if self._headers_buf and not self._headers_written:
                self._headers_written = True
                headers_buf = self._headers_buf
                self._headers_buf = None

                if self.chunked:
                    # Coalesce headers with compressed chunked data
                    chunk_len_pre = f"{chunks_len:x}\r\n".encode("ascii")
                    self._writelines(
                        (headers_buf, chunk_len_pre, *chunks, b"\r\n0\r\n\r\n")
                    )
                else:
                    # Coalesce headers with compressed data
                    self._writelines((headers_buf, *chunks))
            # Headers already sent, just write compressed data
            elif self.chunked:
                chunk_len_pre = f"{chunks_len:x}\r\n".encode("ascii")
                self._writelines((chunk_len_pre, *chunks, b"\r\n0\r\n\r\n"))
            elif len(chunks) > 1:
                self._writelines(chunks)
            else:
                self._write(chunks[0])
        # No compression - send buffered headers if not yet sent
        elif self._headers_buf and not self._headers_written:
            # Use helper to send headers with payload
            self._send_headers_with_payload(chunk, True)
        # Handle remaining body
        elif self.chunked:
            if chunk:
                # Write final chunk with EOF marker
                self._writelines(
                    (f"{len(chunk):x}\r\n".encode("ascii"), chunk, b"\r\n0\r\n\r\n")
                )
            else:
                self._write(b"0\r\n\r\n")
        elif chunk:
            self._write(chunk)
        else:
            self._eof = True
            return

        if chunk and (notify := self.on_body_write) is not None:
            notify(len(chunk), self.output_size)
        await self.drain()
        self._eof = True

    async def drain(self) -> None:
        """Flush the write buffer.

        The intended use is to write

          await w.write(data)
          await w.drain()
        """
        protocol = self._protocol
        if protocol.transport is not None and protocol._paused:
            await protocol._drain_helper()


# https://www.rfc-editor.org/info/rfc9110/#section-5.5-5
# https://www.rfc-editor.org/info/rfc9112/#section-4-3
_FORBIDDEN_HEADER_CHARS_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _safe_header(string: str) -> str:
    if _FORBIDDEN_HEADER_CHARS_RE.search(string) is not None:
        raise ValueError(
            "Forbidden control character detected in headers. "
            "Potential header injection attack."
        )
    return string


def _py_serialize_headers(status_line: str, headers: "CIMultiDict[str]") -> bytes:
    _safe_header(status_line)
    headers_gen = (_safe_header(k) + ": " + _safe_header(v) for k, v in headers.items())
    line = status_line + "\r\n" + "\r\n".join(headers_gen) + "\r\n\r\n"
    return line.encode("utf-8")


_serialize_headers = _py_serialize_headers

try:
    import aiohttp._http_writer as _http_writer  # type: ignore[import-not-found]

    _c_serialize_headers = _http_writer._serialize_headers
    if not NO_EXTENSIONS:
        _serialize_headers = _c_serialize_headers
except ImportError:
    pass
<|endoftext|>import asyncio
import logging
import socket
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence, Sized
from http.cookies import BaseCookie, Morsel, SimpleCookie
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

from multidict import CIMultiDict
from yarl import URL

from ._cookie_helpers import parse_set_cookie_headers
from .typedefs import LooseCookies

if TYPE_CHECKING:
    from .web_app import Application
    from .web_exceptions import HTTPException
    from .web_request import BaseRequest, Request
    from .web_response import StreamResponse
else:
    BaseRequest = Request = Application = StreamResponse = Any
    HTTPException = Any


class AbstractRouter(ABC):
    def __init__(self) -> None:
        self._frozen = False

    def post_init(self, app: Application) -> None:
        """Post init stage.

        Not an abstract method for sake of backward compatibility,
        but if the router wants to be aware of the application
        it can override this.
        """

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freeze router."""
        self._frozen = True

    @abstractmethod
    async def resolve(self, request: Request) -> "AbstractMatchInfo":
        """Return MATCH_INFO for given request"""


class AbstractMatchInfo(ABC):

    __slots__ = ()

    @property  # pragma: no branch
    @abstractmethod
    def handler(self) -> Callable[[Request], Awaitable[StreamResponse]]:
        """Execute matched request handler"""

    @property
    @abstractmethod
    def expect_handler(
        self,
    ) -> Callable[[Request], Awaitable[StreamResponse | None]]:
        """Expect handler for 100-continue processing"""

    @property  # pragma: no branch
    @abstractmethod
    def http_exception(self) -> HTTPException | None:
        """HTTPException instance raised on router's resolving, or None"""

    @abstractmethod  # pragma: no branch
    def get_info(self) -> dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    @property  # pragma: no branch
    @abstractmethod
    def apps(self) -> tuple[Application, ...]:
        """Stack of nested applications.

        Top level application is left-most element.

        """

    @abstractmethod
    def add_app(self, app: Application) -> None:
        """Add application to the nested apps stack."""

    @abstractmethod
    def freeze(self) -> None:
        """Freeze the match info.

        The method is called after route resolution.

        After the call .add_app() is forbidden.

        """


class AbstractView(ABC):
    """Abstract class based view."""

    def __init__(self, request: Request) -> None:
        self._request = request

    @property
    def request(self) -> Request:
        """Request instance."""
        return self._request

    @abstractmethod
    def __await__(self) -> Generator[None, None, StreamResponse]:
        """Execute the view handler."""


class ResolveResult(TypedDict):
    """Resolve result.

    This is the result returned from an AbstractResolver's
    resolve method.

    :param hostname: The hostname that was provided.
    :param host: The IP address that was resolved.
    :param port: The port that was resolved.
    :param family: The address family that was resolved.
    :param proto: The protocol that was resolved.
    :param flags: The flags that were resolved.
    """

    hostname: str
    host: str
    port: int
    family: int
    proto: int
    flags: int


class AbstractResolver(ABC):
    """Abstract DNS resolver."""

    @abstractmethod
    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        """Return IP address for given hostname"""

    @abstractmethod
    async def close(self) -> None:
        """Release resolver"""


if TYPE_CHECKING:
    IterableBase = Iterable[Morsel[str]]
else:
    IterableBase = Iterable


ClearCookiePredicate = Callable[["Morsel[str]"], bool]


class AbstractCookieJar(Sized, IterableBase):
    """Abstract Cookie Jar."""

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    @property
    @abstractmethod
    def unsafe(self) -> bool:
        """Return True if cookies can be used with IP addresses."""

    @property
    @abstractmethod
    def quote_cookie(self) -> bool:
        """Return True if cookies should be quoted."""

    @property
    def treat_as_secure_origin(self) -> frozenset[URL]:
        """Return origins considered secure even over cleartext connections."""
        return frozenset()

    @property
    @abstractmethod
    def cookies(self) -> MappingProxyType[tuple[str, str], SimpleCookie]:
        """Return the cookies stored in this jar."""

    @property
    @abstractmethod
    def host_only_cookies(self) -> frozenset[tuple[str, str, str]]:
        """Return the host-only cookies stored in this jar."""

    @abstractmethod
    def clear(self, predicate: ClearCookiePredicate | None = None) -> None:
        """Clear all cookies if no predicate is passed."""

    @abstractmethod
    def clear_domain(self, domain: str) -> None:
        """Clear all cookies for domain and all subdomains."""

    @abstractmethod
    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        """Update cookies."""

    def update_cookies_from_headers(
        self, headers: Sequence[str], response_url: URL
    ) -> None:
        """Update cookies from raw Set-Cookie headers."""
        if headers and (cookies_to_update := parse_set_cookie_headers(headers)):
            self.update_cookies(cookies_to_update, response_url)

    @abstractmethod
    def filter_cookies(self, request_url: URL) -> "BaseCookie[str]":
        """Return the jar's cookies filtered by their attributes."""


class AbstractStreamWriter(ABC):
    """Abstract stream writer."""

    buffer_size: int = 0
    output_size: int = 0
    length: int | None = 0
    # Called with each accepted body chunk's byte count (before any
    # transport-level transformation such as compression or chunked
    # framing) and the writer's total output_size after the chunk.
    # Assigned by the client request machinery for upload progress
    # tracking; write()/write_eof() implementations should invoke it
    # for every body chunk they accept.
    on_body_write: Callable[[int, int], None] | None = None

    @property
    def transport(self) -> asyncio.WriteTransport | None:
        """The transport this writer writes to, if any.

        Used by upload progress tracking to observe the unsent buffer;
        writers without one report progress only on completion.
        """
        return None

    @abstractmethod
    async def write(self, chunk: bytes | bytearray | memoryview) -> None:
        """Write chunk into stream."""

    @abstractmethod
    async def write_eof(self, chunk: bytes = b"") -> None:
        """Write last chunk."""

    @abstractmethod
    async def drain(self) -> None:
        """Flush the write buffer."""

    @abstractmethod
    def enable_compression(
        self, encoding: str = "deflate", strategy: int | None = None
    ) -> None:
        """Enable HTTP body compression"""

    @abstractmethod
    def enable_chunking(self) -> None:
        """Enable HTTP chunked mode"""

    @abstractmethod
    async def write_headers(
        self, status_line: str, headers: "CIMultiDict[str]"
    ) -> None:
        """Write HTTP headers"""

    def send_headers(self) -> None:
        """Force sending buffered headers if not already sent.

        Required only if write_headers() buffers headers instead of sending immediately.
        For backwards compatibility, this method does nothing by default.
        """


class AbstractAccessLogger(ABC):
    """Abstract writer to access log."""

    __slots__ = ("logger", "log_format")

    def __init__(self, logger: logging.Logger, log_format: str) -> None:
        self.logger = logger
        self.log_format = log_format

    @abstractmethod
    def log(self, request: BaseRequest, response: StreamResponse, time: float) -> None:
        """Emit log to logger."""

    @property
    def enabled(self) -> bool:
        """Check if logger is enabled."""
        return True
<|endoftext|>import json
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, Union

from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy, istr
from yarl import URL, Query as _Query

Query = _Query

DEFAULT_JSON_ENCODER = json.dumps
DEFAULT_JSON_DECODER = json.loads

if TYPE_CHECKING:
    _CIMultiDict = CIMultiDict[str]
    _CIMultiDictProxy = CIMultiDictProxy[str]
    _MultiDict = MultiDict[str]
    _MultiDictProxy = MultiDictProxy[str]
    from http.cookies import BaseCookie, Morsel

    from .web import Request, StreamResponse
else:
    _CIMultiDict = CIMultiDict
    _CIMultiDictProxy = CIMultiDictProxy
    _MultiDict = MultiDict
    _MultiDictProxy = MultiDictProxy

Byteish = Union[bytes, bytearray, memoryview]
JSONEncoder = Callable[[Any], str]
JSONBytesEncoder = Callable[[Any], bytes]
JSONDecoder = Callable[[str], Any]
LooseHeaders = Union[
    Mapping[str, str],
    Mapping[istr, str],
    _CIMultiDict,
    _CIMultiDictProxy,
    Iterable[tuple[str | istr, str]],
]
RawHeaders = tuple[tuple[bytes, bytes], ...]
StrOrURL = Union[str, URL]

LooseCookiesMappings = Mapping[str, Union[str, "BaseCookie[str]", "Morsel[Any]"]]
LooseCookiesIterables = Iterable[
    tuple[str, Union[str, "BaseCookie[str]", "Morsel[Any]"]]
]
LooseCookies = Union[
    LooseCookiesMappings,
    LooseCookiesIterables,
    "BaseCookie[str]",
]

Handler = Callable[["Request"], Awaitable["StreamResponse"]]


class Middleware(Protocol):
    def __call__(
        self, request: "Request", handler: Handler
    ) -> Awaitable["StreamResponse"]: ...


PathLike = Union[str, "os.PathLike[str]"]
<|endoftext|>import asyncio
import datetime
import io
import re
import string
import sys
import types
import warnings
from collections.abc import Iterator, Mapping, MutableMapping
from re import Pattern
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any, Final, Optional, TypeVar, cast, overload

import attr
from multidict import (
    CIMultiDict,
    CIMultiDictProxy,
    MultiDict,
    MultiDictProxy,
    MultiMapping,
)
from yarl import URL, query_to_pairs

from . import hdrs
from ._cookie_helpers import parse_cookie_header
from .abc import AbstractStreamWriter
from .helpers import (
    _SENTINEL,
    DEBUG,
    DEFAULT_CHUNK_SIZE,
    ETAG_ANY,
    LIST_QUOTED_ETAG_RE,
    ChainMapProxy,
    ETag,
    HeadersMixin,
    RequestKey,
    parse_http_date,
    reify,
    sentinel,
    set_exception,
)
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import (
    DEFAULT_JSON_DECODER,
    JSONDecoder,
    LooseHeaders,
    RawHeaders,
    StrOrURL,
)
from .web_exceptions import HTTPRequestEntityTooLarge, NotAppKeyWarning
from .web_response import StreamResponse

if sys.version_info >= (3, 11):
    from tempfile import SpooledTemporaryFile
else:
    import tempfile

    class SpooledTemporaryFile(tempfile.SpooledTemporaryFile[bytes], io.IOBase):
        """Make the spooled file satisfy the documented `FileField.file` type.

        `tempfile.SpooledTemporaryFile` only became an `io.IOBase` subclass in
        3.11 (python/cpython#70363), and never grew the three capability
        predicates before then. Both underlying files (`io.BytesIO` before
        rollover, a `w+b` temp file after) are readable, writable and seekable.
        """

        def readable(self) -> bool:
            return True

        def writable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return True


__all__ = ("BaseRequest", "FileField", "Request")


if TYPE_CHECKING:
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo


_T = TypeVar("_T")


# Multipart file parts are buffered in memory and only spilled to a temporary
# file once they outgrow this size. A temp file per part would let a body made
# of many tiny parts exhaust the process's file descriptors; spooling bounds
# that at ``client_max_size // _FILE_SPOOL_MAX_SIZE`` descriptors per request.
_FILE_SPOOL_MAX_SIZE: Final[int] = 1024**2


@attr.s(auto_attribs=True, frozen=True, slots=True)
class FileField:
    name: str
    filename: str
    file: IO[bytes]
    content_type: str
    headers: CIMultiDictProxy[str]


_Post = str | bytes | bytearray | FileField
_TCHAR: Final[str] = string.digits + string.ascii_letters + r"!#$%&'*+.^_`|~-"
# '-' at the end to prevent interpretation as range in a char class

_TOKEN: Final[str] = rf"[{_TCHAR}]+"

_QDTEXT: Final[str] = r"[{}]".format(
    r"".join(chr(c) for c in (0x09, 0x20, 0x21) + tuple(range(0x23, 0x7F)))
)
# qdtext includes 0x5C to escape 0x5D ('\]')
# qdtext excludes obs-text (because obsoleted, and encoding not specified)

_QUOTED_PAIR: Final[str] = r"\\[\t !-~]"

_QUOTED_STRING: Final[str] = rf'"(?:{_QUOTED_PAIR}|{_QDTEXT})*"'

# This does not have a ReDOS/performance concern as long as it used with re.match().
_FORWARDED_PAIR: Final[str] = rf"({_TOKEN})=({_TOKEN}|{_QUOTED_STRING})(:\d{{1,4}})?"

_QUOTED_PAIR_REPLACE_RE: Final[Pattern[str]] = re.compile(r"\\([\t !-~])")
# same pattern as _QUOTED_PAIR but contains a capture group

_FORWARDED_PAIR_RE: Final[Pattern[str]] = re.compile(_FORWARDED_PAIR)

############################################################
# HTTP Request
############################################################


def _too_many_fields(max_fields: int) -> HTTPRequestEntityTooLarge:
    return HTTPRequestEntityTooLarge(
        max_fields, text=f"Maximum number of form fields {max_fields} exceeded."
    )


class BaseRequest(MutableMapping[str | RequestKey[Any], Any], HeadersMixin):
    POST_METHODS = {
        hdrs.METH_PATCH,
        hdrs.METH_POST,
        hdrs.METH_PUT,
        hdrs.METH_TRACE,
        hdrs.METH_DELETE,
    }

    ATTRS = HeadersMixin.ATTRS | frozenset(
        [
            "_message",
            "_protocol",
            "_payload_writer",
            "_payload",
            "_headers",
            "_method",
            "_version",
            "_rel_url",
            "_post",
            "_read_bytes",
            "_state",
            "_cache",
            "_task",
            "_client_max_size",
            "_client_max_fields",
            "_loop",
            "_transport_sslcontext",
            "_transport_peername",
            "_transport_sockname",
        ]
    )
    _post: MultiDictProxy[_Post] | None = None
    _read_bytes: bytes | None = None
    _seen_str_keys: set[str] = set()

    def __init__(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: "RequestHandler",
        payload_writer: AbstractStreamWriter,
        task: "asyncio.Task[None]",
        loop: asyncio.AbstractEventLoop,
        *,
        client_max_size: int = 1024**2,
        client_max_fields: int = 1000,
        state: dict[RequestKey[Any] | str, Any] | None = None,
        scheme: str | None = None,
        host: str | None = None,
        remote: str | None = None,
    ) -> None:
        self._message = message
        self._protocol = protocol
        self._payload_writer = payload_writer

        self._payload = payload
        self._headers: CIMultiDictProxy[str] = message.headers
        self._method = message.method
        self._version = message.version
        self._cache: dict[str, Any] = {}
        url = message.url
        if url.absolute:
            if scheme is None and url.scheme:
                # Absolute URL is peer-controlled, use the protocol.
                scheme = "https" if protocol.ssl_context else "http"
            if scheme is not None:
                # Authority-form (CONNECT) has no scheme: leave the target
                # unchanged unless clone(scheme=...) overrides it.
                url = url.with_scheme(scheme)
                self._cache["scheme"] = scheme
            if host is not None:
                url = url.with_host(host)
            # absolute URL is given,
            # override auto-calculating url, host, and scheme
            # all other properties should be good
            self._cache["url"] = url
            self._cache["host"] = url.host
            self._rel_url = url.relative()
        else:
            self._rel_url = url
            if scheme is not None:
                self._cache["scheme"] = scheme
            if host is not None:
                self._cache["host"] = host

        self._state = {} if state is None else state
        self._task = task
        self._client_max_size = client_max_size
        self._client_max_fields = client_max_fields
        self._loop = loop

        self._transport_sslcontext = protocol.ssl_context
        self._transport_peername = protocol.peername
        self._transport_sockname = protocol.sockname

        if remote is not None:
            self._cache["remote"] = remote

    def clone(
        self,
        *,
        method: str | _SENTINEL = sentinel,
        rel_url: StrOrURL | _SENTINEL = sentinel,
        headers: LooseHeaders | _SENTINEL = sentinel,
        scheme: str | _SENTINEL = sentinel,
        host: str | _SENTINEL = sentinel,
        remote: str | _SENTINEL = sentinel,
        client_max_size: int | _SENTINEL = sentinel,
        client_max_fields: int | _SENTINEL = sentinel,
    ) -> "BaseRequest":
        """Clone itself with replacement some attributes.

        Creates and returns a new instance of Request object. If no parameters
        are given, an exact copy is returned. If a parameter is not passed, it
        will reuse the one from the current request object.
        """
        if self._read_bytes:
            raise RuntimeError("Cannot clone request after reading its content")

        dct: dict[str, Any] = {}
        if method is not sentinel:
            dct["method"] = method
        if rel_url is not sentinel:
            new_url: URL = URL(rel_url)
            dct["url"] = new_url
            dct["path"] = str(new_url)
        if headers is not sentinel:
            # a copy semantic
            dct["headers"] = CIMultiDictProxy(CIMultiDict(headers))
            dct["raw_headers"] = tuple(
                (k.encode("utf-8"), v.encode("utf-8"))
                for k, v in dct["headers"].items()
            )

        message = self._message._replace(**dct)

        kwargs = {}
        if scheme is not sentinel:
            kwargs["scheme"] = scheme
        if host is not sentinel:
            kwargs["host"] = host
        if remote is not sentinel:
            kwargs["remote"] = remote
        if client_max_size is sentinel:
            client_max_size = self._client_max_size
        if client_max_fields is sentinel:
            client_max_fields = self._client_max_fields

        return self.__class__(
            message,
            self._payload,
            self._protocol,
            self._payload_writer,
            self._task,
            self._loop,
            client_max_size=client_max_size,
            client_max_fields=client_max_fields,
            state=self._state.copy(),
            **kwargs,
        )

    @property
    def task(self) -> "asyncio.Task[None]":
        return self._task

    @property
    def protocol(self) -> "RequestHandler":
        return self._protocol

    @property
    def transport(self) -> asyncio.Transport | None:
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def writer(self) -> AbstractStreamWriter:
        return self._payload_writer

    @property
    def client_max_size(self) -> int:
        return self._client_max_size

    @property
    def client_max_fields(self) -> int:
        return self._client_max_fields

    @reify
    def message(self) -> RawRequestMessage:
        warnings.warn("Request.message is deprecated", DeprecationWarning, stacklevel=3)
        return self._message

    @reify
    def rel_url(self) -> URL:
        return self._rel_url

    @reify
    def loop(self) -> asyncio.AbstractEventLoop:
        warnings.warn(
            "request.loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        return self._loop

    # MutableMapping API

    @overload  # type: ignore[override]
    def __getitem__(self, key: RequestKey[_T]) -> _T: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str | RequestKey[_T]) -> Any:
        return self._state[key]

    @overload  # type: ignore[override]
    def __setitem__(self, key: RequestKey[_T], value: _T) -> None: ...

    @overload
    def __setitem__(self, key: str, value: Any) -> None: ...

    def __setitem__(self, key: str | RequestKey[_T], value: Any) -> None:
        if not isinstance(key, RequestKey) and key not in BaseRequest._seen_str_keys:
            BaseRequest._seen_str_keys.add(key)
            warnings.warn(
                "It is recommended to use web.RequestKey instances for keys.\n"
                + "https://docs.aiohttp.org/en/stable/web_advanced.html"
                + "#request-s-storage",
                category=NotAppKeyWarning,
                stacklevel=2,
            )
        self._state[key] = value

    def __delitem__(self, key: str | RequestKey[_T]) -> None:
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self) -> Iterator[str | RequestKey[Any]]:
        return iter(self._state)

    ########

    @reify
    def secure(self) -> bool:
        """A bool indicating if the request is handled with SSL."""
        return self.scheme == "https"

    @reify
    def forwarded(self) -> tuple[Mapping[str, str], ...]:
        """A tuple containing all parsed Forwarded header(s).

        Makes an effort to parse Forwarded headers as specified by RFC 7239:

        - It adds one (immutable) dictionary per Forwarded 'field-value', ie
          per proxy. The element corresponds to the data in the Forwarded
          field-value added by the first proxy encountered by the client. Each
          subsequent item corresponds to those added by later proxies.
        - It checks that every value has valid syntax in general as specified
          in section 4: either a 'token' or a 'quoted-string'.
        - It un-escapes found escape sequences.
        - It does NOT validate 'by' and 'for' contents as specified in section
          6.
        - It does NOT validate 'host' contents (Host ABNF).
        - It does NOT validate 'proto' contents for valid URI scheme names.

        Returns a tuple containing one or more immutable dicts
        """
        elems = []
        for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
            length = len(field_value)
            pos = 0
            need_separator = False
            elem: dict[str, str] = {}
            elems.append(types.MappingProxyType(elem))
            while 0 <= pos < length:
                match = _FORWARDED_PAIR_RE.match(field_value, pos)
                if match is not None:  # got a valid forwarded-pair
                    if need_separator:
                        # bad syntax here, skip to next comma
                        pos = field_value.find(",", pos)
                    else:
                        name, value, port = match.groups()
                        if value[0] == '"':
                            # quoted string: remove quotes and unescape
                            value = _QUOTED_PAIR_REPLACE_RE.sub(r"\1", value[1:-1])
                        if port:
                            value += port
                        elem[name.lower()] = value
                        pos += len(match.group(0))
                        need_separator = True
                elif field_value[pos] == ",":  # next forwarded-element
                    need_separator = False
                    elem = {}
                    elems.append(types.MappingProxyType(elem))
                    pos += 1
                elif field_value[pos] == ";":  # next forwarded-pair
                    need_separator = False
                    pos += 1
                elif field_value[pos] in " \t":
                    # Allow whitespace even between forwarded-pairs, though
                    # RFC 7239 doesn't. This simplifies code and is in line
                    # with Postel's law.
                    pos += 1
                else:
                    # bad syntax here, skip to next comma
                    pos = field_value.find(",", pos)
        return tuple(elems)

    @reify
    def scheme(self) -> str:
        """A string representing the scheme of the request.

        Hostname is resolved in this order:

        - overridden value by .clone(scheme=new_scheme) call.
        - type of connection to peer: HTTPS if socket is SSL, HTTP otherwise.

        'http' or 'https'.
        """
        if self._transport_sslcontext:
            return "https"
        else:
            return "http"

    @reify
    def method(self) -> str:
        """Read only property for getting HTTP method.

        The value is upper-cased str like 'GET', 'POST', 'PUT' etc.
        """
        return self._method

    @reify
    def version(self) -> HttpVersion:
        """Read only property for getting HTTP version of request.

        Returns aiohttp.protocol.HttpVersion instance.
        """
        return self._version

    @reify
    def host(self) -> str:
        """Hostname of the request.

        Hostname is resolved in this order:

        - overridden value by .clone(host=new_host) call.
        - HOST HTTP header
        - local socket address the request arrived on
          (transport ``sockname``)
        - empty string if no transport information is available

        For example, 'example.com' or 'localhost:8080'.

        For historical reasons, the port number may be included.
        """
        host = self._message.headers.get(hdrs.HOST)
        if host is not None:
            return host
        sockname = self._transport_sockname
        if sockname is None:
            return ""
        if isinstance(sockname, tuple):
            # AF_INET6 returns a 4-tuple (host, port, flowinfo, scopeid);
            # bracket the bare address so it matches the Host-header shape
            # and is a valid URL authority component.
            if len(sockname) == 4:
                return f"[{sockname[0]}]"
            return str(sockname[0])
        return str(sockname)

    @reify
    def remote(self) -> str | None:
        """Remote IP of client initiated HTTP request.

        The IP is resolved in this order:

        - overridden value by .clone(remote=new_remote) call.
        - peername of opened socket
        """
        if self._transport_peername is None:
            return None
        if isinstance(self._transport_peername, (list, tuple)):
            return str(self._transport_peername[0])
        return str(self._transport_peername)

    @reify
    def url(self) -> URL:
        """The full URL of the request."""
        # authority is used here because it may include the port number
        # and we want yarl to parse it correctly
        return URL.build(scheme=self.scheme, authority=self.host).join(self._rel_url)

    @reify
    def path(self) -> str:
        """The URL including *PATH INFO* without the host or scheme.

        E.g., ``/app/blog``
        """
        return self._rel_url.path

    @reify
    def path_qs(self) -> str:
        """The URL including PATH_INFO and the query string.

        E.g, /app/blog?id=10
        """
        return str(self._rel_url)

    @reify
    def raw_path(self) -> str:
        """The URL including raw *PATH INFO* without the host or scheme.

        Warning, the path is unquoted and may contains non valid URL characters

        E.g., ``/my%2Fpath%7Cwith%21some%25strange%24characters``
        """
        return self._message.path

    @reify
    def query(self) -> "MultiMapping[str]":
        """A multidict with all the variables in the query string."""
        return self._rel_url.query

    @reify
    def query_string(self) -> str:
        """The query string in the URL.

        E.g., id=10
        """
        return self._rel_url.query_string

    @reify
    def headers(self) -> CIMultiDictProxy[str]:
        """A case-insensitive multidict proxy with all headers."""
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        """A sequence of pairs for all headers."""
        return self._message.raw_headers

    @reify
    def if_modified_since(self) -> datetime.datetime | None:
        """The value of If-Modified-Since HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_MODIFIED_SINCE))

    @reify
    def if_unmodified_since(self) -> datetime.datetime | None:
        """The value of If-Unmodified-Since HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_UNMODIFIED_SINCE))

    @staticmethod
    def _etag_values(etag_header: str) -> Iterator[ETag]:
        """Extract `ETag` objects from raw header."""
        if etag_header == ETAG_ANY:
            yield ETag(
                is_weak=False,
                value=ETAG_ANY,
            )
        else:
            for match in LIST_QUOTED_ETAG_RE.finditer(etag_header):
                is_weak, value, garbage = match.group(2, 3, 4)
                # Any symbol captured by 4th group means
                # that the following sequence is invalid.
                if garbage:
                    break

                yield ETag(
                    is_weak=bool(is_weak),
                    value=value,
                )

    @classmethod
    def _if_match_or_none_impl(
        cls, header_value: str | None
    ) -> tuple[ETag, ...] | None:
        if not header_value:
            return None

        return tuple(cls._etag_values(header_value))

    @reify
    def if_match(self) -> tuple[ETag, ...] | None:
        """The value of If-Match HTTP header, or None.

        This header is represented as a `tuple` of `ETag` objects.
        """
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_MATCH))

    @reify
    def if_none_match(self) -> tuple[ETag, ...] | None:
        """The value of If-None-Match HTTP header, or None.

        This header is represented as a `tuple` of `ETag` objects.
        """
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_NONE_MATCH))

    @reify
    def if_range(self) -> datetime.datetime | None:
        """The value of If-Range HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_RANGE))

    @reify
    def keep_alive(self) -> bool:
        """Is keepalive enabled by client?"""
        return not self._message.should_close

    @reify
    def cookies(self) -> Mapping[str, str]:
        """Return request cookies.

        A read-only dictionary-like object.
        """
        # Use parse_cookie_header for RFC 6265 compliant Cookie header parsing
        # that accepts special characters in cookie names (fixes #2683)
        parsed = parse_cookie_header(self.headers.get(hdrs.COOKIE, ""))
        # Extract values from Morsel objects
        return MappingProxyType({name: morsel.value for name, morsel in parsed})

    @reify
    def http_range(self) -> slice:
        """The content of Range HTTP header.

        Return a slice instance.

        """
        rng = self._headers.get(hdrs.RANGE)
        start, end = None, None
        if rng is not None:
            try:
                pattern = r"^bytes=(\d*)-(\d*)$"
                # https://www.rfc-editor.org/info/rfc9110/#section-14.1-4
                start, end = re.findall(pattern, rng, re.ASCII | re.IGNORECASE)[0]
            except IndexError:  # pattern was not found in header
                raise ValueError("range not in acceptable format")

            end = int(end) if end else None
            start = int(start) if start else None

            if start is None and end is not None:
                # end with no start is to return tail of content
                start = -end
                end = None

            if start is not None and end is not None:
                # end is inclusive in range header, exclusive for slice
                end += 1

                if start >= end:
                    raise ValueError("start cannot be after end")

            if start is end is None:  # No valid range supplied
                raise ValueError("No start or end of range specified")

        return slice(start, end, 1)

    @reify
    def content(self) -> StreamReader:
        """Return raw payload stream."""
        return self._payload

    @property
    def has_body(self) -> bool:
        """Return True if request's HTTP BODY can be read, False otherwise."""
        warnings.warn(
            "Deprecated, use .can_read_body #2005", DeprecationWarning, stacklevel=2
        )
        return not self._payload.at_eof()

    @property
    def can_read_body(self) -> bool:
        """Return True if request's HTTP BODY can be read, False otherwise."""
        return not self._payload.at_eof()

    @reify
    def body_exists(self) -> bool:
        """Return True if request has HTTP BODY, False otherwise."""
        return type(self._payload) is not EmptyStreamReader

    async def release(self) -> None:
        """Release request.

        Eat unread part of HTTP BODY if present.
        """
        while not self._payload.at_eof():
            await self._payload.readany()

    async def read(self) -> bytes:
        """Read request body if present.

        Returns bytes object with full request content.
        """
        if self._read_bytes is None:
            # Raise the buffer limits so compressed payloads decompress in
            # larger chunks instead of many small pause/resume cycles.
            if self._client_max_size:
                self._payload.set_read_chunk_size(self._client_max_size)
            body = bytearray()
            while True:
                chunk = await self._payload.readany()
                body.extend(chunk)
                if self._client_max_size:
                    body_size = len(body)
                    if body_size > self._client_max_size:
                        raise HTTPRequestEntityTooLarge(self._client_max_size)
                if not chunk:
                    break
            self._read_bytes = bytes(body)
        return self._read_bytes

    async def text(self) -> str:
        """Return BODY as text using encoding from .charset."""
        bytes_body = await self.read()
        encoding = self.charset or "utf-8"
        return bytes_body.decode(encoding)

    async def json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER) -> Any:
        """Return BODY as JSON."""
        body = await self.text()
        return loads(body)

    async def multipart(self) -> MultipartReader:
        """Return async iterator to process BODY as multipart."""
        return MultipartReader(
            self._headers,
            self._payload,
            client_max_size=self._client_max_size,
            max_field_size=self._protocol.max_field_size,
            max_headers=self._protocol.max_headers,
            max_size_error_cls=HTTPRequestEntityTooLarge,
        )

    async def post(self) -> "MultiDictProxy[_Post]":
        """Return POST parameters."""
        if self._post is not None:
            return self._post
        if self._method not in self.POST_METHODS:
            self._post = MultiDictProxy(MultiDict())
            return self._post

        content_type = self.content_type
        if content_type not in (
            "",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ):
            self._post = MultiDictProxy(MultiDict())
            return self._post

        out: MultiDict[_Post]

        if content_type == "multipart/form-data":
            out = MultiDict()
            multipart = await self.multipart()
            max_size = self._client_max_size
            max_fields = self._client_max_fields

            payload = self._payload
            while (field := await multipart.next()) is not None:
                # This check is needed for empty payloads, which still add
                # overhead without entering the loop and the check below.
                if 0 < max_size < payload.total_bytes:
                    raise HTTPRequestEntityTooLarge(max_size)
                if 0 < max_fields <= len(out):
                    raise _too_many_fields(max_fields)

                field_ct = field.headers.get(hdrs.CONTENT_TYPE)

                if isinstance(field, BodyPartReader):
                    if field.name is None:
                        raise ValueError("Multipart field missing name.")

                    # Note that according to RFC 7578, the Content-Type header
                    # is optional, even for files, so we can't assume it's
                    # present.
                    # https://tools.ietf.org/html/rfc7578#section-4.4
                    if field.filename:
                        tmp = SpooledTemporaryFile(_FILE_SPOOL_MAX_SIZE)
                        # rolled means the temp file now uses the disk, at
                        # which point we want to run in the executor.
                        rolled = False
                        while chunk := await field.read_chunk(size=DEFAULT_CHUNK_SIZE):
                            # Bounds one part, mid-read.
                            if 0 < max_size < payload.total_bytes:
                                if rolled:
                                    await self._loop.run_in_executor(None, tmp.close)
                                else:
                                    tmp.close()
                                raise HTTPRequestEntityTooLarge(max_size)
                            async for decoded_chunk in field.decode_iter(chunk):
                                # Update before writing, so we know if next
                                # write is going to hit the disk.
                                rolled = rolled or (
                                    tmp.tell() + len(decoded_chunk)
                                    > _FILE_SPOOL_MAX_SIZE
                                )
                                if rolled:
                                    await self._loop.run_in_executor(
                                        None, tmp.write, decoded_chunk
                                    )
                                else:
                                    tmp.write(decoded_chunk)
                        if rolled:
                            await self._loop.run_in_executor(None, tmp.seek, 0)
                        else:
                            tmp.seek(0)

                        if field_ct is None:
                            field_ct = "application/octet-stream"

                        ff = FileField(
                            field.name,
                            field.filename,
                            tmp,
                            field_ct,
                            field.headers,
                        )
                        out.add(field.name, ff)
                    else:
                        # deal with ordinary data
                        raw_data = bytearray()
                        while chunk := await field.read_chunk():
                            if 0 < max_size < payload.total_bytes:
                                raise HTTPRequestEntityTooLarge(max_size)
                            raw_data.extend(chunk)

                        value = bytearray()
                        # form-data doesn't support compression, so don't need to check size again.
                        async for d in field.decode_iter(raw_data):
                            value.extend(d)

                        if field_ct is None or field_ct.startswith("text/"):
                            charset = field.get_charset(default="utf-8")
                            out.add(field.name, value.decode(charset))
                        else:
                            out.add(field.name, value)
                else:
                    raise ValueError(
                        "To decode nested multipart you need to use custom reader",
                    )
        elif not (data := await self.read()):
            out = MultiDict()
        else:
            charset = self.charset or "utf-8"
            query = data.rstrip().decode(charset)
            max_fields = self._client_max_fields
            try:
                out = MultiDict(
                    query_to_pairs(
                        query,
                        max_fields=max_fields if max_fields > 0 else None,
                        encoding=charset,
                    )
                )
            except ValueError:
                raise _too_many_fields(max_fields) from None

        self._post = MultiDictProxy(out)
        return self._post

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Extra info from protocol transport"""
        protocol = self._protocol
        if protocol is None:
            return default

        transport = protocol.transport
        if transport is None:
            return default

        return transport.get_extra_info(name, default)

    def __repr__(self) -> str:
        ascii_encodable_path = self.path.encode("ascii", "backslashreplace").decode(
            "ascii"
        )
        return f"<{self.__class__.__name__} {self._method} {ascii_encodable_path} >"

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def __bool__(self) -> bool:
        return True

    async def _prepare_hook(self, response: StreamResponse) -> None:
        return

    def _cancel(self, exc: BaseException) -> None:
        set_exception(self._payload, exc)

    def _finish(self) -> None:
        if self._post is None or self.content_type != "multipart/form-data":
            return

        # Release the temp files created within multipart request body.
        for file_name, file_field_object in self._post.items():
            if isinstance(file_field_object, FileField):
                file_field_object.file.close()


class Request(BaseRequest):

    ATTRS = BaseRequest.ATTRS | frozenset(["_match_info"])

    _match_info: Optional["UrlMappingMatchInfo"] = None

    if DEBUG:

        def __setattr__(self, name: str, val: Any) -> None:
            if name not in self.ATTRS:
                warnings.warn(
                    f"Setting custom {self.__class__.__name__}.{name} attribute "
                    "is discouraged",
                    DeprecationWarning,
                    stacklevel=2,
                )
            super().__setattr__(name, val)

    def clone(
        self,
        *,
        method: str | _SENTINEL = sentinel,
        rel_url: StrOrURL | _SENTINEL = sentinel,
        headers: LooseHeaders | _SENTINEL = sentinel,
        scheme: str | _SENTINEL = sentinel,
        host: str | _SENTINEL = sentinel,
        remote: str | _SENTINEL = sentinel,
        client_max_size: int | _SENTINEL = sentinel,
        client_max_fields: int | _SENTINEL = sentinel,
    ) -> "Request":
        ret = super().clone(
            method=method,
            rel_url=rel_url,
            headers=headers,
            scheme=scheme,
            host=host,
            remote=remote,
            client_max_size=client_max_size,
            client_max_fields=client_max_fields,
        )
        new_ret = cast(Request, ret)
        new_ret._match_info = self._match_info
        return new_ret

    @reify
    def match_info(self) -> "UrlMappingMatchInfo":
        """Result of route resolving."""
        match_info = self._match_info
        assert match_info is not None
        return match_info

    @property
    def app(self) -> "Application":
        """Application instance."""
        match_info = self._match_info
        assert match_info is not None
        return match_info.current_app

    @property
    def config_dict(self) -> ChainMapProxy:
        match_info = self._match_info
        assert match_info is not None
        lst = match_info.apps
        app = self.app
        idx = lst.index(app)
        sublist = list(reversed(lst[: idx + 1]))
        return ChainMapProxy(sublist)

    async def _prepare_hook(self, response: StreamResponse) -> None:
        match_info = self._match_info
        if match_info is None:
            return
        for app in match_info._apps:
            if on_response_prepare := app.on_response_prepare:
                await on_response_prepare.send(self, response)
<|endoftext|>"""Client middleware support."""

from collections.abc import Awaitable, Callable, Sequence

from .client_reqrep import ClientRequest, ClientResponse

__all__ = ("ClientMiddlewareType", "ClientHandlerType", "build_client_middlewares")

# Type alias for client request handlers - functions that process requests and return responses
ClientHandlerType = Callable[[ClientRequest], Awaitable[ClientResponse]]

# Type for client middleware - similar to server but uses ClientRequest/ClientResponse
ClientMiddlewareType = Callable[
    [ClientRequest, ClientHandlerType], Awaitable[ClientResponse]
]


def build_client_middlewares(
    handler: ClientHandlerType,
    middlewares: Sequence[ClientMiddlewareType],
) -> ClientHandlerType:
    """
    Apply middlewares to request handler.

    The middlewares are applied in reverse order, so the first middleware
    in the list wraps all subsequent middlewares and the handler.

    This implementation avoids using partial/update_wrapper to minimize overhead
    and doesn't cache to avoid holding references to stateful middleware.
    """
    # Optimize for single middleware case
    if len(middlewares) == 1:
        middleware = middlewares[0]

        async def single_middleware_handler(req: ClientRequest) -> ClientResponse:
            return await middleware(req, handler)

        return single_middleware_handler

    # Build the chain for multiple middlewares
    current_handler = handler

    for middleware in reversed(middlewares):
        # Create a new closure that captures the current state
        def make_wrapper(
            mw: ClientMiddlewareType, next_h: ClientHandlerType
        ) -> ClientHandlerType:
            async def wrapped(req: ClientRequest) -> ClientResponse:
                return await mw(req, next_h)

            return wrapped

        current_handler = make_wrapper(middleware, current_handler)

    return current_handler
<|endoftext|>import asyncio
import io
import os
import pathlib
import sys
from contextlib import suppress
from enum import Enum, auto
from mimetypes import MimeTypes
from stat import S_ISREG
from types import MappingProxyType
from typing import (  # noqa
    IO,
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Final,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import DEFAULT_CHUNK_SIZE, ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
    HTTPForbidden,
    HTTPNotFound,
    HTTPNotModified,
    HTTPPartialContent,
    HTTPPreconditionFailed,
    HTTPRequestRangeNotSatisfiable,
)
from .web_response import StreamResponse

__all__ = ("FileResponse",)

if TYPE_CHECKING:
    from .web_request import BaseRequest


_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]


NOSENDFILE: Final[bool] = bool(os.environ.get("AIOHTTP_NOSENDFILE"))

CONTENT_TYPES: Final[MimeTypes] = MimeTypes()

# File extension to IANA encodings map that will be checked in the order defined.
ENCODING_EXTENSIONS = MappingProxyType(
    {ext: CONTENT_TYPES.encodings_map[ext] for ext in (".br", ".gz")}
)

FALLBACK_CONTENT_TYPE = "application/octet-stream"

# Provide additional MIME type/extension pairs to be recognized.
# https://en.wikipedia.org/wiki/List_of_archive_formats#Compression_only
ADDITIONAL_CONTENT_TYPES = MappingProxyType(
    {
        "application/gzip": ".gz",
        "application/x-brotli": ".br",
        "application/x-bzip2": ".bz2",
        "application/x-compress": ".Z",
        "application/x-xz": ".xz",
    }
)


class _FileResponseResult(Enum):
    """The result of the file response."""

    SEND_FILE = auto()  # Ie a regular file to send
    NOT_ACCEPTABLE = auto()  # Ie a socket, or non-regular file
    PRE_CONDITION_FAILED = auto()  # Ie If-Match or If-None-Match failed
    NOT_MODIFIED = auto()  # 304 Not Modified


# Add custom pairs and clear the encodings map so guess_type ignores them.
CONTENT_TYPES.encodings_map.clear()
for content_type, extension in ADDITIONAL_CONTENT_TYPES.items():
    CONTENT_TYPES.add_type(content_type, extension)


_CLOSE_FUTURES: set[asyncio.Future[None]] = set()


class FileResponse(StreamResponse):
    """A response object can be used to send files."""

    def __init__(
        self,
        path: PathLike,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        status: int = 200,
        reason: str | None = None,
        headers: LooseHeaders | None = None,
    ) -> None:
        super().__init__(status=status, reason=reason, headers=headers)

        self._path = pathlib.Path(path)
        self._chunk_size = chunk_size

    def _seek_and_read(self, fobj: IO[Any], offset: int, chunk_size: int) -> bytes:
        fobj.seek(offset)
        return fobj.read(chunk_size)  # type: ignore[no-any-return]

    async def _sendfile_fallback(
        self, writer: AbstractStreamWriter, fobj: IO[Any], offset: int, count: int
    ) -> AbstractStreamWriter:
        # To keep memory usage low,fobj is transferred in chunks
        # controlled by the constructor's chunk_size argument.

        chunk_size = self._chunk_size
        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(
            None, self._seek_and_read, fobj, offset, min(chunk_size, count)
        )
        while chunk:
            await writer.write(chunk)
            count = count - len(chunk)
            if count <= 0:
                break
            chunk = await loop.run_in_executor(None, fobj.read, min(chunk_size, count))

        await writer.drain()
        return writer

    async def _sendfile(
        self, request: "BaseRequest", fobj: IO[Any], offset: int, count: int
    ) -> AbstractStreamWriter:
        writer = await super().prepare(request)
        assert writer is not None

        if NOSENDFILE or self.compression:
            return await self._sendfile_fallback(writer, fobj, offset, count)

        loop = request._loop
        transport = request.transport
        if transport is None:
            raise ConnectionResetError("Connection lost")

        try:
            await loop.sendfile(transport, fobj, offset, count)
        except NotImplementedError:
            return await self._sendfile_fallback(writer, fobj, offset, count)

        await super().write_eof()
        return writer

    @staticmethod
    def _etag_match(etag_value: str, etags: tuple[ETag, ...], *, weak: bool) -> bool:
        if len(etags) == 1 and etags[0].value == ETAG_ANY:
            return True
        return any(
            etag.value == etag_value for etag in etags if weak or not etag.is_weak
        )

    async def _not_modified(
        self, request: "BaseRequest", etag_value: str, last_modified: float
    ) -> AbstractStreamWriter | None:
        self.set_status(HTTPNotModified.status_code)
        self._length_check = False
        self.etag = etag_value
        self.last_modified = last_modified
        # Delete any Content-Length headers provided by user. HTTP 304
        # should always have empty response body
        return await super().prepare(request)

    async def _precondition_failed(
        self, request: "BaseRequest"
    ) -> AbstractStreamWriter | None:
        self.set_status(HTTPPreconditionFailed.status_code)
        self.content_length = 0
        return await super().prepare(request)

    def _make_response(
        self, request: "BaseRequest", accept_encoding: str
    ) -> tuple[
        _FileResponseResult, io.BufferedReader | None, os.stat_result, str | None
    ]:
        """Return the response result, io object, stat result, and encoding.

        If an uncompressed file is returned, the encoding is set to
        :py:data:`None`.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        file_path, st, file_encoding = self._get_file_path_stat_encoding(
            accept_encoding
        )
        if not file_path:
            return _FileResponseResult.NOT_ACCEPTABLE, None, st, None

        etag_value = f"{st.st_mtime_ns:x}-{st.st_size:x}"

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.1-2
        if (ifmatch := request.if_match) is not None and not self._etag_match(
            etag_value, ifmatch, weak=False
        ):
            return _FileResponseResult.PRE_CONDITION_FAILED, None, st, file_encoding

        if (
            (unmodsince := request.if_unmodified_since) is not None
            and ifmatch is None
            and st.st_mtime > unmodsince.timestamp()
        ):
            return _FileResponseResult.PRE_CONDITION_FAILED, None, st, file_encoding

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.2-2
        if (ifnonematch := request.if_none_match) is not None and self._etag_match(
            etag_value, ifnonematch, weak=True
        ):
            return _FileResponseResult.NOT_MODIFIED, None, st, file_encoding

        if (
            (modsince := request.if_modified_since) is not None
            and ifnonematch is None
            and st.st_mtime <= modsince.timestamp()
        ):
            return _FileResponseResult.NOT_MODIFIED, None, st, file_encoding

        fobj = file_path.open("rb")
        with suppress(OSError):
            # fstat() may not be available on all platforms
            # Once we open the file, we want the fstat() to ensure
            # the file has not changed between the first stat()
            # and the open().
            st = os.stat(fobj.fileno())
        return _FileResponseResult.SEND_FILE, fobj, st, file_encoding

    def _get_file_path_stat_encoding(
        self, accept_encoding: str
    ) -> tuple[pathlib.Path | None, os.stat_result, str | None]:
        file_path = self._path
        for file_extension, file_encoding in ENCODING_EXTENSIONS.items():
            if file_encoding not in accept_encoding:
                continue

            compressed_path = file_path.with_suffix(file_path.suffix + file_extension)
            with suppress(OSError):
                # Do not follow symlinks and ignore any non-regular files.
                st = compressed_path.lstat()
                if S_ISREG(st.st_mode):
                    return compressed_path, st, file_encoding

        # Fallback to the uncompressed file
        st = file_path.stat()
        return file_path if S_ISREG(st.st_mode) else None, st, None

    async def prepare(self, request: "BaseRequest") -> AbstractStreamWriter | None:
        loop = asyncio.get_running_loop()
        # Encoding comparisons should be case-insensitive
        # https://www.rfc-editor.org/rfc/rfc9110#section-8.4.1
        accept_encoding = request.headers.get(hdrs.ACCEPT_ENCODING, "").lower()
        try:
            response_result, fobj, st, file_encoding = await loop.run_in_executor(
                None, self._make_response, request, accept_encoding
            )
        except PermissionError:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)
        except OSError:
            # Most likely to be FileNotFoundError or OSError for circular
            # symlinks in python >= 3.13, so respond with 404.
            self.set_status(HTTPNotFound.status_code)
            return await super().prepare(request)

        # Forbid special files like sockets, pipes, devices, etc.
        if response_result is _FileResponseResult.NOT_ACCEPTABLE:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)

        if response_result is _FileResponseResult.PRE_CONDITION_FAILED:
            return await self._precondition_failed(request)

        if response_result is _FileResponseResult.NOT_MODIFIED:
            etag_value = f"{st.st_mtime_ns:x}-{st.st_size:x}"
            last_modified = st.st_mtime
            return await self._not_modified(request, etag_value, last_modified)

        assert fobj is not None
        try:
            return await self._prepare_open_file(request, fobj, st, file_encoding)
        finally:
            # We do not await here because we do not want to wait
            # for the executor to finish before returning the response
            # so the connection can begin servicing another request
            # as soon as possible.
            close_future = loop.run_in_executor(None, fobj.close)
            # Hold a strong reference to the future to prevent it from being
            # garbage collected before it completes.
            _CLOSE_FUTURES.add(close_future)
            close_future.add_done_callback(_CLOSE_FUTURES.remove)

    async def _prepare_open_file(
        self,
        request: "BaseRequest",
        fobj: io.BufferedReader,
        st: os.stat_result,
        file_encoding: str | None,
    ) -> AbstractStreamWriter | None:
        status = self._status
        file_size: int = st.st_size
        file_mtime: float = st.st_mtime
        count: int = file_size
        start: int | None = None

        if (ifrange := request.if_range) is None or file_mtime <= ifrange.timestamp():
            # If-Range header check:
            # condition = cached date >= last modification date
            # return 206 if True else 200.
            # if False:
            #   Range header would not be processed, return 200
            # if True but Range header missing
            #   return 200
            try:
                rng = request.http_range
                start = rng.start
                end: int | None = rng.stop
            except ValueError:
                # https://tools.ietf.org/html/rfc7233:
                # A server generating a 416 (Range Not Satisfiable) response to
                # a byte-range request SHOULD send a Content-Range header field
                # with an unsatisfied-range value.
                # The complete-length in a 416 response indicates the current
                # length of the selected representation.
                #
                # Will do the same below. Many servers ignore this and do not
                # send a Content-Range header with HTTP 416
                self._headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)

            # If a range request has been made, convert start, end slice
            # notation into file pointer offset and count
            if start is not None:
                if start < 0 and end is None:  # return tail of file
                    start += file_size
                    if start < 0:
                        # if Range:bytes=-1000 in request header but file size
                        # is only 200, there would be trouble without this
                        start = 0
                    count = file_size - start
                else:
                    # rfc7233:If the last-byte-pos value is
                    # absent, or if the value is greater than or equal to
                    # the current length of the representation data,
                    # the byte range is interpreted as the remainder
                    # of the representation (i.e., the server replaces the
                    # value of last-byte-pos with a value that is one less than
                    # the current length of the selected representation).
                    count = (
                        min(end if end is not None else file_size, file_size) - start
                    )

                if start >= file_size:
                    # HTTP 416 should be returned in this case.
                    #
                    # According to https://tools.ietf.org/html/rfc7233:
                    # If a valid byte-range-set includes at least one
                    # byte-range-spec with a first-byte-pos that is less than
                    # the current length of the representation, or at least one
                    # suffix-byte-range-spec with a non-zero suffix-length,
                    # then the byte-range-set is satisfiable. Otherwise, the
                    # byte-range-set is unsatisfiable.
                    self._headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                    self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                    return await super().prepare(request)

                status = HTTPPartialContent.status_code
                # Even though you are sending the whole file, you should still
                # return a HTTP 206 for a Range request.
                self.set_status(status)

        # If the Content-Type header is not already set, guess it based on the
        # extension of the request path. The encoding returned by guess_type
        #  can be ignored since the map was cleared above.
        if hdrs.CONTENT_TYPE not in self._headers:
            if sys.version_info >= (3, 13):
                guesser = CONTENT_TYPES.guess_file_type
            else:
                guesser = CONTENT_TYPES.guess_type
            self.content_type = guesser(self._path)[0] or FALLBACK_CONTENT_TYPE

        if file_encoding:
            self._headers[hdrs.CONTENT_ENCODING] = file_encoding
            self._headers[hdrs.VARY] = hdrs.ACCEPT_ENCODING
            # Disable compression if we are already sending
            # a compressed file since we don't want to double
            # compress.
            self._compression = False

        self.etag = f"{st.st_mtime_ns:x}-{st.st_size:x}"
        self.last_modified = file_mtime
        self.content_length = count

        self._headers[hdrs.ACCEPT_RANGES] = "bytes"

        if status == HTTPPartialContent.status_code:
            real_start = start
            assert real_start is not None
            self._headers[hdrs.CONTENT_RANGE] = (
                f"bytes {real_start}-{real_start + count - 1}/{file_size}"
            )

        # If we are sending 0 bytes calling sendfile() will throw a ValueError
        if count == 0 or must_be_empty_body(request.method, status):
            return await super().prepare(request)

        # be aware that start could be None or int=0 here.
        offset = start or 0

        return await self._sendfile(request, fobj, offset, count)
<|endoftext|>from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar

import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL

from .client_reqrep import ClientResponse

if TYPE_CHECKING:
    from .client import ClientSession

    _ParamT_contra = TypeVar("_ParamT_contra", contravariant=True)
    _TracingSignal = Signal[ClientSession, SimpleNamespace, _ParamT_contra]


__all__ = (
    "TraceConfig",
    "TraceRequestStartParams",
    "TraceRequestEndParams",
    "TraceRequestExceptionParams",
    "TraceConnectionQueuedStartParams",
    "TraceConnectionQueuedEndParams",
    "TraceConnectionCreateStartParams",
    "TraceConnectionCreateEndParams",
    "TraceConnectionReuseconnParams",
    "TraceDnsResolveHostStartParams",
    "TraceDnsResolveHostEndParams",
    "TraceDnsCacheHitParams",
    "TraceDnsCacheMissParams",
    "TraceRequestRedirectParams",
    "TraceRequestChunkSentParams",
    "TraceResponseChunkReceivedParams",
    "TraceRequestHeadersSentParams",
)


class TraceConfig:
    """First-class used to trace requests launched via ClientSession objects."""

    def __init__(
        self, trace_config_ctx_factory: type[SimpleNamespace] = SimpleNamespace
    ) -> None:
        self._on_request_start: _TracingSignal[TraceRequestStartParams] = Signal(self)
        self._on_request_chunk_sent: _TracingSignal[TraceRequestChunkSentParams] = (
            Signal(self)
        )
        self._on_response_chunk_received: _TracingSignal[
            TraceResponseChunkReceivedParams
        ] = Signal(self)
        self._on_request_end: _TracingSignal[TraceRequestEndParams] = Signal(self)
        self._on_request_exception: _TracingSignal[TraceRequestExceptionParams] = (
            Signal(self)
        )
        self._on_request_redirect: _TracingSignal[TraceRequestRedirectParams] = Signal(
            self
        )
        self._on_connection_queued_start: _TracingSignal[
            TraceConnectionQueuedStartParams
        ] = Signal(self)
        self._on_connection_queued_end: _TracingSignal[
            TraceConnectionQueuedEndParams
        ] = Signal(self)
        self._on_connection_create_start: _TracingSignal[
            TraceConnectionCreateStartParams
        ] = Signal(self)
        self._on_connection_create_end: _TracingSignal[
            TraceConnectionCreateEndParams
        ] = Signal(self)
        self._on_connection_reuseconn: _TracingSignal[
            TraceConnectionReuseconnParams
        ] = Signal(self)
        self._on_dns_resolvehost_start: _TracingSignal[
            TraceDnsResolveHostStartParams
        ] = Signal(self)
        self._on_dns_resolvehost_end: _TracingSignal[TraceDnsResolveHostEndParams] = (
            Signal(self)
        )
        self._on_dns_cache_hit: _TracingSignal[TraceDnsCacheHitParams] = Signal(self)
        self._on_dns_cache_miss: _TracingSignal[TraceDnsCacheMissParams] = Signal(self)
        self._on_request_headers_sent: _TracingSignal[TraceRequestHeadersSentParams] = (
            Signal(self)
        )

        self._trace_config_ctx_factory = trace_config_ctx_factory

    def trace_config_ctx(self, trace_request_ctx: Any = None) -> SimpleNamespace:
        """Return a new trace_config_ctx instance"""
        return self._trace_config_ctx_factory(trace_request_ctx=trace_request_ctx)

    def freeze(self) -> None:
        self._on_request_start.freeze()
        self._on_request_chunk_sent.freeze()
        self._on_response_chunk_received.freeze()
        self._on_request_end.freeze()
        self._on_request_exception.freeze()
        self._on_request_redirect.freeze()
        self._on_connection_queued_start.freeze()
        self._on_connection_queued_end.freeze()
        self._on_connection_create_start.freeze()
        self._on_connection_create_end.freeze()
        self._on_connection_reuseconn.freeze()
        self._on_dns_resolvehost_start.freeze()
        self._on_dns_resolvehost_end.freeze()
        self._on_dns_cache_hit.freeze()
        self._on_dns_cache_miss.freeze()
        self._on_request_headers_sent.freeze()

    @property
    def on_request_start(self) -> "_TracingSignal[TraceRequestStartParams]":
        return self._on_request_start

    @property
    def on_request_chunk_sent(
        self,
    ) -> "_TracingSignal[TraceRequestChunkSentParams]":
        return self._on_request_chunk_sent

    @property
    def on_response_chunk_received(
        self,
    ) -> "_TracingSignal[TraceResponseChunkReceivedParams]":
        return self._on_response_chunk_received

    @property
    def on_request_end(self) -> "_TracingSignal[TraceRequestEndParams]":
        return self._on_request_end

    @property
    def on_request_exception(
        self,
    ) -> "_TracingSignal[TraceRequestExceptionParams]":
        return self._on_request_exception

    @property
    def on_request_redirect(
        self,
    ) -> "_TracingSignal[TraceRequestRedirectParams]":
        return self._on_request_redirect

    @property
    def on_connection_queued_start(
        self,
    ) -> "_TracingSignal[TraceConnectionQueuedStartParams]":
        return self._on_connection_queued_start

    @property
    def on_connection_queued_end(
        self,
    ) -> "_TracingSignal[TraceConnectionQueuedEndParams]":
        return self._on_connection_queued_end

    @property
    def on_connection_create_start(
        self,
    ) -> "_TracingSignal[TraceConnectionCreateStartParams]":
        return self._on_connection_create_start

    @property
    def on_connection_create_end(
        self,
    ) -> "_TracingSignal[TraceConnectionCreateEndParams]":
        return self._on_connection_create_end

    @property
    def on_connection_reuseconn(
        self,
    ) -> "_TracingSignal[TraceConnectionReuseconnParams]":
        return self._on_connection_reuseconn

    @property
    def on_dns_resolvehost_start(
        self,
    ) -> "_TracingSignal[TraceDnsResolveHostStartParams]":
        return self._on_dns_resolvehost_start

    @property
    def on_dns_resolvehost_end(
        self,
    ) -> "_TracingSignal[TraceDnsResolveHostEndParams]":
        return self._on_dns_resolvehost_end

    @property
    def on_dns_cache_hit(self) -> "_TracingSignal[TraceDnsCacheHitParams]":
        return self._on_dns_cache_hit

    @property
    def on_dns_cache_miss(self) -> "_TracingSignal[TraceDnsCacheMissParams]":
        return self._on_dns_cache_miss

    @property
    def on_request_headers_sent(
        self,
    ) -> "_TracingSignal[TraceRequestHeadersSentParams]":
        return self._on_request_headers_sent


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestStartParams:
    """Parameters sent by the `on_request_start` signal"""

    method: str
    url: URL
    headers: "CIMultiDict[str]"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestChunkSentParams:
    """Parameters sent by the `on_request_chunk_sent` signal"""

    method: str
    url: URL
    chunk: bytes


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceResponseChunkReceivedParams:
    """Parameters sent by the `on_response_chunk_received` signal"""

    method: str
    url: URL
    chunk: bytes


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestEndParams:
    """Parameters sent by the `on_request_end` signal"""

    method: str
    url: URL
    headers: "CIMultiDict[str]"
    response: ClientResponse


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestExceptionParams:
    """Parameters sent by the `on_request_exception` signal"""

    method: str
    url: URL
    headers: "CIMultiDict[str]"
    exception: BaseException


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestRedirectParams:
    """Parameters sent by the `on_request_redirect` signal"""

    method: str
    url: URL
    headers: "CIMultiDict[str]"
    response: ClientResponse


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionQueuedStartParams:
    """Parameters sent by the `on_connection_queued_start` signal"""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionQueuedEndParams:
    """Parameters sent by the `on_connection_queued_end` signal"""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionCreateStartParams:
    """Parameters sent by the `on_connection_create_start` signal"""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionCreateEndParams:
    """Parameters sent by the `on_connection_create_end` signal"""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionReuseconnParams:
    """Parameters sent by the `on_connection_reuseconn` signal"""


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceDnsResolveHostStartParams:
    """Parameters sent by the `on_dns_resolvehost_start` signal"""

    host: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceDnsResolveHostEndParams:
    """Parameters sent by the `on_dns_resolvehost_end` signal"""

    host: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceDnsCacheHitParams:
    """Parameters sent by the `on_dns_cache_hit` signal"""

    host: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceDnsCacheMissParams:
    """Parameters sent by the `on_dns_cache_miss` signal"""

    host: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceRequestHeadersSentParams:
    """Parameters sent by the `on_request_headers_sent` signal"""

    method: str
    url: URL
    headers: "CIMultiDict[str]"


class Trace:
