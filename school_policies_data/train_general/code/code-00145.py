    :param protocol: A ``PROTOCOL`` constant from the :mod:`ssl` module.
        Defaults to :data:`ssl.PROTOCOL_TLS_SERVER`.
    """
    if protocol is None:
        protocol = ssl.PROTOCOL_TLS_SERVER

    ctx = ssl.SSLContext(protocol)
    ctx.load_cert_chain(cert_file, pkey_file)
    return ctx


def is_ssl_error(error: Exception | None = None) -> bool:
    """Checks if the given error (or the current one) is an SSL error."""
    if error is None:
        error = t.cast(Exception, sys.exc_info()[1])
    return isinstance(error, ssl.SSLError)


def select_address_family(host: str, port: int) -> socket.AddressFamily:
    """Return ``AF_INET4``, ``AF_INET6``, or ``AF_UNIX`` depending on
    the host and port."""
    if host.startswith("unix://"):
        return socket.AF_UNIX
    elif ":" in host and hasattr(socket, "AF_INET6"):
        return socket.AF_INET6
    return socket.AF_INET


def get_sockaddr(
    host: str, port: int, family: socket.AddressFamily
) -> tuple[str, int] | str:
    """Return a fully qualified socket address that can be passed to
    :func:`socket.bind`."""
    if family == af_unix:
        # Absolute path avoids IDNA encoding error when path starts with dot.
        return os.path.abspath(host.partition("://")[2])
    try:
        res = socket.getaddrinfo(
            host, port, family, socket.SOCK_STREAM, socket.IPPROTO_TCP
        )
    except socket.gaierror:
        return host, port
    return res[0][4]  # type: ignore


def get_interface_ip(family: socket.AddressFamily) -> str:
    """Get the IP address of an external interface. Used when binding to
    0.0.0.0 or ::1 to show a more useful URL.

    :meta private:
    """
    # arbitrary private address
    host = "fd31:f903:5ab5:1::1" if family == socket.AF_INET6 else "10.253.155.219"

    with socket.socket(family, socket.SOCK_DGRAM) as s:
        try:
            s.connect((host, 58162))
        except OSError:
            return "::1" if family == socket.AF_INET6 else "127.0.0.1"

        return s.getsockname()[0]  # type: ignore


class BaseWSGIServer(HTTPServer):
    """A WSGI server that that handles one request at a time.

    Use :func:`make_server` to create a server instance.
    """

    multithread = False
    multiprocess = False
    request_queue_size = LISTEN_QUEUE
    allow_reuse_address = True

    def __init__(
        self,
        host: str,
        port: int,
        app: WSGIApplication,
        handler: type[WSGIRequestHandler] | None = None,
        passthrough_errors: bool = False,
        ssl_context: _TSSLContextArg | None = None,
        fd: int | None = None,
    ) -> None:
        if handler is None:
            handler = WSGIRequestHandler

        # If the handler doesn't directly set a protocol version and
        # thread or process workers are used, then allow chunked
        # responses and keep-alive connections by enabling HTTP/1.1.
        if "protocol_version" not in vars(handler) and (
            self.multithread or self.multiprocess
        ):
            handler.protocol_version = "HTTP/1.1"

        self.host = host
        self.port = port
        self.app = app
        self.passthrough_errors = passthrough_errors

        self.address_family = address_family = select_address_family(host, port)
        server_address = get_sockaddr(host, int(port), address_family)

        # Remove a leftover Unix socket file from a previous run. Don't
        # remove a file that was set up by run_simple.
        if address_family == af_unix and fd is None:
            server_address = t.cast(str, server_address)

            if os.path.exists(server_address):
                os.unlink(server_address)

        # Bind and activate will be handled manually, it should only
        # happen if we're not using a socket that was already set up.
        super().__init__(
            server_address,  # type: ignore[arg-type]
            handler,
            bind_and_activate=False,
        )

        if fd is None:
            # No existing socket descriptor, do bind_and_activate=True.
            try:
                self.server_bind()
                self.server_activate()
            except OSError as e:
                # Catch connection issues and show them without the traceback. Show
                # extra instructions for address not found, and for macOS.
                self.server_close()
                print(e.strerror, file=sys.stderr)

                if e.errno == errno.EADDRINUSE:
                    print(
                        f"Port {port} is in use by another program. Either identify and"
                        " stop that program, or start the server with a different"
                        " port.",
                        file=sys.stderr,
                    )

                    if sys.platform == "darwin" and port == 5000:
                        print(
                            "On macOS, try searching for and disabling"
                            " 'AirPlay Receiver' in System Settings.",
                            file=sys.stderr,
                        )

                sys.exit(1)
            except BaseException:
                self.server_close()
                raise
        else:
            # TCPServer automatically opens a socket even if bind_and_activate is False.
            # Close it to silence a ResourceWarning.
            self.server_close()

            # Use the passed in socket directly.
            self.socket = socket.fromfd(fd, address_family, socket.SOCK_STREAM)
            self.server_address = self.socket.getsockname()

        if address_family != af_unix:
            # If port was 0, this will record the bound port.
            self.port = self.server_address[1]

        if ssl_context is not None:
            if isinstance(ssl_context, tuple):
                ssl_context = load_ssl_context(*ssl_context)
            elif ssl_context == "adhoc":
                ssl_context = generate_adhoc_ssl_context()

            self.socket = ssl_context.wrap_socket(self.socket, server_side=True)
            self.ssl_context: ssl.SSLContext | None = ssl_context
        else:
            self.ssl_context = None

        import importlib.metadata

        self._server_version = f"Werkzeug/{importlib.metadata.version('werkzeug')}"

    def log(self, type: str, message: str, *args: t.Any) -> None:
        _log(type, message, *args)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        try:
            super().serve_forever(poll_interval=poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.server_close()

    def handle_error(
        self, request: t.Any, client_address: tuple[str, int] | str
    ) -> None:
        if self.passthrough_errors:
            raise

        return super().handle_error(request, client_address)

    def log_startup(self) -> None:
        """Show information about the address when starting the server."""
        dev_warning = (
            "WARNING: This is a development server. Do not use it in a production"
            " deployment. Use a production WSGI server instead."
        )
        dev_warning = _ansi_style(dev_warning, "bold", "red")
        messages = [dev_warning]

        if self.address_family == af_unix:
            messages.append(f" * Running on {self.host}")
        else:
            scheme = "http" if self.ssl_context is None else "https"
            display_hostname = self.host

            if self.host in {"0.0.0.0", "::"}:
                messages.append(f" * Running on all addresses ({self.host})")

                if self.host == "0.0.0.0":
                    localhost = "127.0.0.1"
                    display_hostname = get_interface_ip(socket.AF_INET)
                else:
                    localhost = "[::1]"
                    display_hostname = get_interface_ip(socket.AF_INET6)

                messages.append(f" * Running on {scheme}://{localhost}:{self.port}")

            if ":" in display_hostname:
                display_hostname = f"[{display_hostname}]"

            messages.append(f" * Running on {scheme}://{display_hostname}:{self.port}")

        _log("info", "\n".join(messages))


class ThreadedWSGIServer(socketserver.ThreadingMixIn, BaseWSGIServer):
    """A WSGI server that handles concurrent requests in separate
    threads.

    Use :func:`make_server` to create a server instance.
    """

    multithread = True
    daemon_threads = True


class ForkingWSGIServer(ForkingMixIn, BaseWSGIServer):
    """A WSGI server that handles concurrent requests in separate forked
    processes.

    Use :func:`make_server` to create a server instance.
    """

    multiprocess = True

    def __init__(
        self,
        host: str,
        port: int,
        app: WSGIApplication,
        processes: int = 40,
        handler: type[WSGIRequestHandler] | None = None,
        passthrough_errors: bool = False,
        ssl_context: _TSSLContextArg | None = None,
        fd: int | None = None,
    ) -> None:
        if not can_fork:
            raise ValueError("Your platform does not support forking.")

        super().__init__(host, port, app, handler, passthrough_errors, ssl_context, fd)
        self.max_children = processes


def make_server(
    host: str,
    port: int,
    app: WSGIApplication,
    threaded: bool = False,
    processes: int = 1,
    request_handler: type[WSGIRequestHandler] | None = None,
    passthrough_errors: bool = False,
    ssl_context: _TSSLContextArg | None = None,
    fd: int | None = None,
) -> BaseWSGIServer:
    """Create an appropriate WSGI server instance based on the value of
    ``threaded`` and ``processes``.

    This is called from :func:`run_simple`, but can be used separately
    to have access to the server object, such as to run it in a separate
    thread.

    See :func:`run_simple` for parameter docs.
    """
    if threaded and processes > 1:
        raise ValueError("Cannot have a multi-thread and multi-process server.")

    if threaded:
        return ThreadedWSGIServer(
            host, port, app, request_handler, passthrough_errors, ssl_context, fd=fd
        )

    if processes > 1:
        return ForkingWSGIServer(
            host,
            port,
            app,
            processes,
            request_handler,
            passthrough_errors,
            ssl_context,
            fd=fd,
        )

    return BaseWSGIServer(
        host, port, app, request_handler, passthrough_errors, ssl_context, fd=fd
    )


def is_running_from_reloader() -> bool:
    """Check if the server is running as a subprocess within the
    Werkzeug reloader.

    .. versionadded:: 0.10
    """
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def run_simple(
    hostname: str,
    port: int,
    application: WSGIApplication,
    use_reloader: bool = False,
    use_debugger: bool = False,
    use_evalex: bool = True,
    extra_files: t.Iterable[str] | None = None,
    exclude_patterns: t.Iterable[str] | None = None,
    reloader_interval: int = 1,
    reloader_type: str = "auto",
    threaded: bool = False,
    processes: int = 1,
    request_handler: type[WSGIRequestHandler] | None = None,
    static_files: dict[str, str | tuple[str, str]] | None = None,
    passthrough_errors: bool = False,
    ssl_context: _TSSLContextArg | None = None,
) -> None:
    """Start a development server for a WSGI application. Various
    optional features can be enabled.

    .. warning::

        Do not use the development server when deploying to production.
        It is intended for use only during local development. It is not
        designed to be particularly efficient, stable, or secure.

    :param hostname: The host to bind to, for example ``'localhost'``.
        Can be a domain, IPv4 or IPv6 address, or file path starting
        with ``unix://`` for a Unix socket.
    :param port: The port to bind to, for example ``8080``. Using ``0``
        tells the OS to pick a random free port.
    :param application: The WSGI application to run.
    :param use_reloader: Use a reloader process to restart the server
        process when files are changed.
    :param use_debugger: Use Werkzeug's debugger, which will show
        formatted tracebacks on unhandled exceptions.
    :param use_evalex: Make the debugger interactive. A Python terminal
        can be opened for any frame in the traceback. Some protection is
        provided by requiring a PIN, but this should never be enabled
        on a publicly visible server.
    :param extra_files: The reloader will watch these files for changes
        in addition to Python modules. For example, watch a
        configuration file.
    :param exclude_patterns: The reloader will ignore changes to any
        files matching these :mod:`fnmatch` patterns. For example,
        ignore cache files.
    :param reloader_interval: How often the reloader tries to check for
        changes.
    :param reloader_type: The reloader to use. The ``'stat'`` reloader
        is built in, but may require significant CPU to watch files. The
        ``'watchdog'`` reloader is much more efficient but requires
        installing the ``watchdog`` package first.
    :param threaded: Handle concurrent requests using threads. Cannot be
        used with ``processes``.
    :param processes: Handle concurrent requests using up to this number
        of processes. Cannot be used with ``threaded``.
    :param request_handler: Use a different
        :class:`~BaseHTTPServer.BaseHTTPRequestHandler` subclass to
        handle requests.
    :param static_files: A dict mapping URL prefixes to directories to
        serve static files from using
        :class:`~werkzeug.middleware.SharedDataMiddleware`.
    :param passthrough_errors: Don't catch unhandled exceptions at the
        server level, let the server crash instead. If ``use_debugger``
        is enabled, the debugger will still catch such errors.
    :param ssl_context: Configure TLS to serve over HTTPS. Can be an
        :class:`ssl.SSLContext` object, a ``(cert_file, key_file)``
        tuple to create a typical context, or the string ``'adhoc'`` to
        generate a temporary self-signed certificate.

    .. versionchanged:: 2.1
        Instructions are shown for dealing with an "address already in
        use" error.

    .. versionchanged:: 2.1
        Running on ``0.0.0.0`` or ``::`` shows the loopback IP in
        addition to a real IP.

    .. versionchanged:: 2.1
        The command-line interface was removed.

    .. versionchanged:: 2.0
        Running on ``0.0.0.0`` or ``::`` shows a real IP address that
        was bound as well as a warning not to run the development server
        in production.

    .. versionchanged:: 2.0
        The ``exclude_patterns`` parameter was added.

    .. versionchanged:: 0.15
        Bind to a Unix socket by passing a ``hostname`` that starts with
        ``unix://``.

    .. versionchanged:: 0.10
        Improved the reloader and added support for changing the backend
        through the ``reloader_type`` parameter.

    .. versionchanged:: 0.9
        A command-line interface was added.

    .. versionchanged:: 0.8
        ``ssl_context`` can be a tuple of paths to the certificate and
        private key files.

    .. versionchanged:: 0.6
        The ``ssl_context`` parameter was added.

    .. versionchanged:: 0.5
       The ``static_files`` and ``passthrough_errors`` parameters were
       added.
    """
    if not isinstance(port, int):
        raise TypeError("port must be an integer")

    if static_files:
        from .middleware.shared_data import SharedDataMiddleware

        application = SharedDataMiddleware(application, static_files)

    if use_debugger:
        from .debug import DebuggedApplication

        application = DebuggedApplication(application, evalex=use_evalex)
        # Allow the specified hostname to use the debugger, in addition to
        # localhost domains.
        application.trusted_hosts.append(hostname)

    if not is_running_from_reloader():
        fd = None
    else:
        fd = int(os.environ["WERKZEUG_SERVER_FD"])

    srv = make_server(
        hostname,
        port,
        application,
        threaded,
        processes,
        request_handler,
        passthrough_errors,
        ssl_context,
        fd=fd,
    )
    srv.socket.set_inheritable(True)
    os.environ["WERKZEUG_SERVER_FD"] = str(srv.fileno())

    if not is_running_from_reloader():
        srv.log_startup()
        _log("info", _ansi_style("Press CTRL+C to quit", "yellow"))

    if use_reloader:
        from ._reloader import run_with_reloader

        try:
            run_with_reloader(
                srv.serve_forever,
                extra_files=extra_files,
                exclude_patterns=exclude_patterns,
                interval=reloader_interval,
                reloader_type=reloader_type,
            )
        finally:
            srv.server_close()
    else:
        srv.serve_forever()
<|endoftext|>from __future__ import annotations

import logging
import re
import sys
import typing as t
from datetime import datetime
from datetime import timezone

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

    from .wrappers.request import Request

_logger: logging.Logger | None = None


class _Missing:
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


def _wsgi_decoding_dance(s: str) -> str:
    return s.encode("latin1").decode(errors="replace")


def _wsgi_encoding_dance(s: str) -> str:
    return s.encode().decode("latin1")


def _get_environ(obj: WSGIEnvironment | Request) -> WSGIEnvironment:
    env = getattr(obj, "environ", obj)
    assert isinstance(env, dict), (
        f"{type(obj).__name__!r} is not a WSGI environment (has to be a dict)"
    )
    return env


def _has_level_handler(logger: logging.Logger) -> bool:
    """Check if there is a handler in the logging chain that will handle
    the given logger's effective level.
    """
    level = logger.getEffectiveLevel()
    current = logger

    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True

        if not current.propagate:
            break

        current = current.parent  # type: ignore

    return False


class _ColorStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """On Windows, wrap stream with Colorama for ANSI style support."""

    def __init__(self) -> None:
        try:
            import colorama
        except ImportError:
            stream = None
        else:
            stream = colorama.AnsiToWin32(sys.stderr)

        super().__init__(stream)


def _log(type: str, message: str, *args: t.Any, **kwargs: t.Any) -> None:
    """Log a message to the 'werkzeug' logger.

    The logger is created the first time it is needed. If there is no
    level set, it is set to :data:`logging.INFO`. If there is no handler
    for the logger's effective level, a :class:`logging.StreamHandler`
    is added.
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger("werkzeug")

        if _logger.level == logging.NOTSET:
            _logger.setLevel(logging.INFO)

        if not _has_level_handler(_logger):
            _logger.addHandler(_ColorStreamHandler())

    getattr(_logger, type)(message.rstrip(), *args, **kwargs)


@t.overload
def _dt_as_utc(dt: None) -> None: ...


@t.overload
def _dt_as_utc(dt: datetime) -> datetime: ...


def _dt_as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return dt

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)

    return dt


_TAccessorValue = t.TypeVar("_TAccessorValue")


class _DictAccessorProperty(t.Generic[_TAccessorValue]):
    """Baseclass for `environ_property` and `header_property`."""

    read_only = False

    def __init__(
        self,
        name: str,
        default: _TAccessorValue | None = None,
        load_func: t.Callable[[str], _TAccessorValue] | None = None,
        dump_func: t.Callable[[_TAccessorValue], str] | None = None,
        read_only: bool | None = None,
        doc: str | None = None,
    ) -> None:
        self.name = name
        self.default = default
        self.load_func = load_func
        self.dump_func = dump_func
        if read_only is not None:
            self.read_only = read_only
        self.__doc__ = doc

    def lookup(self, instance: t.Any) -> t.MutableMapping[str, t.Any]:
        raise NotImplementedError

    @t.overload
    def __get__(
        self, instance: None, owner: type
    ) -> _DictAccessorProperty[_TAccessorValue]: ...

    @t.overload
    def __get__(self, instance: t.Any, owner: type) -> _TAccessorValue: ...

    def __get__(
        self, instance: t.Any | None, owner: type
    ) -> _TAccessorValue | _DictAccessorProperty[_TAccessorValue]:
        if instance is None:
            return self

        storage = self.lookup(instance)

        if self.name not in storage:
            return self.default  # type: ignore

        value = storage[self.name]

        if self.load_func is not None:
            try:
                return self.load_func(value)
            except (ValueError, TypeError):
                return self.default  # type: ignore

        return value  # type: ignore

    def __set__(self, instance: t.Any, value: _TAccessorValue) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        if self.dump_func is not None:
            self.lookup(instance)[self.name] = self.dump_func(value)
        else:
            self.lookup(instance)[self.name] = value

    def __delete__(self, instance: t.Any) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        self.lookup(instance).pop(self.name, None)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


_plain_int_re = re.compile(r"-?[0-9]+", re.ASCII)


def _plain_int(value: str) -> int:
    """Parse an int only if it is only ASCII digits and ``-``.

    This disallows ``+``, ``_``, and non-ASCII digits, which are accepted by ``int`` but
    are not allowed in HTTP header values.

    Any leading or trailing whitespace is stripped
    """
    value = value.strip(" \t")
    if _plain_int_re.fullmatch(value) is None:
        raise ValueError

    return int(value)
<|endoftext|>from __future__ import annotations

import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote

from ._internal import _dt_as_utc
from ._internal import _plain_int

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

_token_chars = frozenset(
    "!#$%&'*+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ^_`abcdefghijklmnopqrstuvwxyz|~"
)
_entity_headers = frozenset(
    [
        "allow",
        "content-encoding",
        "content-language",
        "content-length",
        "content-location",
        "content-md5",
        "content-range",
        "content-type",
        "expires",
        "last-modified",
    ]
)
_hop_by_hop_headers = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)
HTTP_STATUS_CODES = {
    100: "Continue",
    101: "Switching Protocols",
    102: "Processing",
    103: "Early Hints",  # see RFC 8297
    200: "OK",
    201: "Created",
    202: "Accepted",
    203: "Non Authoritative Information",
    204: "No Content",
    205: "Reset Content",
    206: "Partial Content",
    207: "Multi Status",
    208: "Already Reported",  # see RFC 5842
    226: "IM Used",  # see RFC 3229
    300: "Multiple Choices",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    305: "Use Proxy",
    306: "Switch Proxy",  # unused
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",  # unused
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    407: "Proxy Authentication Required",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    411: "Length Required",
    412: "Precondition Failed",
    413: "Request Entity Too Large",
    414: "Request URI Too Long",
    415: "Unsupported Media Type",
    416: "Requested Range Not Satisfiable",
    417: "Expectation Failed",
    418: "I'm a teapot",  # see RFC 2324
    421: "Misdirected Request",  # see RFC 7540
    422: "Unprocessable Entity",
    423: "Locked",
    424: "Failed Dependency",
    425: "Too Early",  # see RFC 8470
    426: "Upgrade Required",
    428: "Precondition Required",  # see RFC 6585
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    449: "Retry With",  # proprietary MS extension
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    505: "HTTP Version Not Supported",
    506: "Variant Also Negotiates",  # see RFC 2295
    507: "Insufficient Storage",
    508: "Loop Detected",  # see RFC 5842
    510: "Not Extended",
    511: "Network Authentication Failed",
}


class COEP(Enum):
    """Cross Origin Embedder Policies"""

    UNSAFE_NONE = "unsafe-none"
    REQUIRE_CORP = "require-corp"


class COOP(Enum):
    """Cross Origin Opener Policies"""

    UNSAFE_NONE = "unsafe-none"
    SAME_ORIGIN_ALLOW_POPUPS = "same-origin-allow-popups"
    SAME_ORIGIN = "same-origin"


def quote_header_value(value: t.Any, allow_token: bool = True) -> str:
    """Add double quotes around a header value. If the header contains only ASCII token
    characters, it will be returned unchanged. If the header contains ``"`` or ``\\``
    characters, they will be escaped with an additional ``\\`` character.

    This is the reverse of :func:`unquote_header_value`.

    :param value: The value to quote. Will be converted to a string.
    :param allow_token: Disable to quote the value even if it only has token characters.

    .. versionchanged:: 3.0
        Passing bytes is not supported.

    .. versionchanged:: 3.0
        The ``extra_chars`` parameter is removed.

    .. versionchanged:: 2.3
        The value is quoted if it is the empty string.

    .. versionadded:: 0.5
    """
    value_str = str(value)

    if not value_str:
        return '""'

    if allow_token:
        token_chars = _token_chars

        if token_chars.issuperset(value_str):
            return value_str

    value_str = value_str.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value_str}"'


_unslash_re = re.compile(r"\\(.)", re.A)


def unquote_header_value(value: str) -> str:
    """Remove double quotes and backslash escapes from a header value.

    This is the reverse of :func:`quote_header_value`.

    :param value: The header value to unquote.

    .. versionchanged:: 3.2
        Removes escape preceding any character.

    .. versionchanged:: 3.0
        The ``is_filename`` parameter is removed.
    """
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return _unslash_re.sub(r"\g<1>", value[1:-1])

    return value


def dump_options_header(header: str | None, options: t.Mapping[str, t.Any]) -> str:
    """Produce a header value and ``key=value`` parameters separated by semicolons
    ``;``. For example, the ``Content-Type`` header.

    .. code-block:: python

        dump_options_header("text/html", {"charset": "UTF-8"})
        'text/html; charset=UTF-8'

    This is the reverse of :func:`parse_options_header`.

    If a value contains non-token characters, it will be quoted.

    If a value is ``None``, the parameter is skipped.

    In some keys for some headers, a UTF-8 value can be encoded using a special
    ``key*=UTF-8''value`` form, where ``value`` is percent encoded. This function will
    not produce that format automatically, but if a given key ends with an asterisk
    ``*``, the value is assumed to have that form and will not be quoted further.

    :param header: The primary header value.
    :param options: Parameters to encode as ``key=value`` pairs.

    .. versionchanged:: 2.3
        Keys with ``None`` values are skipped rather than treated as a bare key.

    .. versionchanged:: 2.2.3
        If a key ends with ``*``, its value will not be quoted.
    """
    segments = []

    if header is not None:
        segments.append(header)

    for key, value in options.items():
        if value is None:
            continue

        if key.endswith("*"):
            segments.append(f"{key}={value}")
        else:
            segments.append(f"{key}={quote_header_value(value)}")

    return "; ".join(segments)


def dump_header(iterable: dict[str, t.Any] | t.Iterable[t.Any]) -> str:
    """Produce a header value from a list of items or ``key=value`` pairs, separated by
    commas ``,``.

    This is the reverse of :func:`parse_list_header`, :func:`parse_dict_header`, and
    :func:`parse_set_header`.

    If a value contains non-token characters, it will be quoted.

    If a value is ``None``, the key is output alone.

    In some keys for some headers, a UTF-8 value can be encoded using a special
    ``key*=UTF-8''value`` form, where ``value`` is percent encoded. This function will
    not produce that format automatically, but if a given key ends with an asterisk
    ``*``, the value is assumed to have that form and will not be quoted further.

    .. code-block:: python

        dump_header(["foo", "bar baz"])
        'foo, "bar baz"'

        dump_header({"foo": "bar baz"})
        'foo="bar baz"'

    :param iterable: The items to create a header from.

    .. versionchanged:: 3.0
        The ``allow_token`` parameter is removed.

    .. versionchanged:: 2.2.3
        If a key ends with ``*``, its value will not be quoted.
    """
    if isinstance(iterable, dict):
        items = []

        for key, value in iterable.items():
            if value is None:
                items.append(key)
            elif key.endswith("*"):
                items.append(f"{key}={value}")
            else:
                items.append(f"{key}={quote_header_value(value)}")
    else:
        items = [quote_header_value(x) for x in iterable]

    return ", ".join(items)


def dump_csp_header(header: ds.ContentSecurityPolicy) -> str:
    """Dump a Content Security Policy header.

    These are structured into policies such as "default-src 'self';
    script-src 'self'".

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """
    return "; ".join(f"{key} {value}" for key, value in header.items())


def parse_list_header(value: str) -> list[str]:
    """Parse a header value that consists of a list of comma separated items according
    to `RFC 9110 <https://httpwg.org/specs/rfc9110.html#abnf.extension>`__.

    Surrounding quotes are removed from items, but internal quotes are left for
    future parsing. Empty values are discarded.

    .. code-block:: python

        parse_list_header('token, "quoted value"')
        ['token', 'quoted value']

    This is the reverse of :func:`dump_header`.

    :param value: The header value to parse.

    .. versionchanged:: 3.2
        Quotes and escapes are kept if only part of an item is quoted. Empty
        values are omitted. An empty list is returned if the value contains an
        unclosed quoted string.
    """
    items = []
    item = ""
    escape = False
    quote = False

    for char in value:
        if escape:
            escape = False
            item += char
            continue

        if quote:
            if char == "\\":
                escape = True
            elif char == '"':
                quote = False

            item += char
            continue

        if char == ",":
            items.append(item)
            item = ""
            continue

        if char == '"':
            quote = True

        item += char

    if quote:
        # invalid, unclosed quoted string
        return []

    items.append(item)
    return [
        unquote_header_value(item)
        for item in (item.strip(" \t") for item in items)
        if item
    ]


def parse_dict_header(value: str) -> dict[str, str | None]:
    """Parse a list header using :func:`parse_list_header`, then parse each item as a
    ``key=value`` pair.

    .. code-block:: python

        parse_dict_header('a=b, c="d, e", f')
        {"a": "b", "c": "d, e", "f": None}

    This is the reverse of :func:`dump_header`.

    If a key does not have a value, it is ``None``.

    This handles charsets for values as described in
    `RFC 2231 <https://www.rfc-editor.org/rfc/rfc2231#section-3>`__. Only ASCII, UTF-8,
    and ISO-8859-1 charsets are accepted, otherwise the value remains quoted.

    :param value: The header value to parse.

    .. versionchanged:: 3.2
        An empty dict is returned if the value contains an unclosed quoted
        string.

    .. versionchanged:: 3.0
        Passing bytes is not supported.

    .. versionchanged:: 3.0
        The ``cls`` argument is removed.

    .. versionchanged:: 2.3
        Added support for ``key*=charset''value`` encoded items.

    .. versionchanged:: 0.9
       The ``cls`` argument was added.
    """
    result: dict[str, str | None] = {}

    for item in parse_list_header(value):
        key, has_value, value = item.partition("=")
        key = key.strip(" \t")

        if not key:
            # =value is not valid
            continue

        if not has_value:
            result[key] = None
            continue

        value = value.strip(" \t")
        encoding: str | None = None

        if key.endswith("*"):
            # key*=charset''value becomes key=value, where value is percent encoded
            # adapted from parse_options_header, without the continuation handling
            key = key[:-1]

            if (m_charset := _charset_value_re.match(value)) is not None:
                # If there is a charset marker in the value, split it off.
                encoding, value = m_charset.groups()
                encoding = encoding.lower()

            # A safe list of encodings. Modern clients should only send ASCII or UTF-8.
            # This list will not be extended further. An invalid encoding will leave the
            # value quoted.
            if encoding in {"ascii", "us-ascii", "utf-8", "iso-8859-1"}:
                # invalid bytes are replaced during unquoting
                value = unquote(value, encoding=encoding)

        result[key] = unquote_header_value(value)

    return result


# https://httpwg.org/specs/rfc9110.html#parameter
_parameter_key_re = re.compile(r"([\w!#$%&'*+\-.^`|~]+)=", flags=re.ASCII)
# https://www.rfc-editor.org/rfc/rfc2231#section-3
_parameter_mark_re = re.compile(r"(?:\*(\d+))?(\*)?$", flags=re.ASCII)
_parameter_token_value_re = re.compile(r"[\w!#$%&'*+\-.^`|~]+", flags=re.ASCII)
# https://www.rfc-editor.org/rfc/rfc2231#section-4
_charset_value_re = re.compile(
    r"""
    ([\w!#$%&*+\-.^`|~]*)'  # charset part, could be empty
    [\w!#$%&*+\-.^`|~]*'  # don't care about language part, usually empty
    ([\w!#$%&'*+\-.^`|~]+)$  # one or more token chars with percent encoding
    """,
    re.ASCII | re.VERBOSE,
)
_parameter_delimiter_re = re.compile(r";[ \t]*")


def parse_options_header(value: str | None) -> tuple[str, dict[str, str]]:
    """Parse a header that consists of a value with ``key=value`` parameters separated
    by semicolons ``;``. For example, the ``Content-Type`` header.

    .. code-block:: python

        parse_options_header("text/html; charset=UTF-8")
        ('text/html', {'charset': 'UTF-8'})

        parse_options_header("")
        ("", {})

    This is the reverse of :func:`dump_options_header`.

    This parses valid parameter parts as described in
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#parameter>`__. Invalid parts are
    skipped.

    This handles continuations and charsets as described in
    `RFC 2231 <https://www.rfc-editor.org/rfc/rfc2231#section-3>`__, although not as
    strictly as the RFC. Only ASCII, UTF-8, and ISO-8859-1 charsets are accepted,
    otherwise the value remains quoted.

    Clients may not be consistent in how they handle a quote character within a quoted
    value. The `HTML Standard <https://html.spec.whatwg.org/#multipart-form-data>`__
    replaces it with ``%22`` in multipart form data.
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#quoted.strings>`__ uses backslash
    escapes in HTTP headers. Both are decoded to the ``"`` character.

    Clients may not be consistent in how they handle non-ASCII characters. HTML
    documents must declare ``<meta charset=UTF-8>``, otherwise browsers may replace with
    HTML character references, which can be decoded using :func:`html.unescape`.

    :param value: The header value to parse.
    :return: ``(value, options)``, where ``options`` is a dict

    .. versionchanged:: 2.3
        Invalid parts, such as keys with no value, quoted keys, and incorrectly quoted
        values, are discarded instead of treating as ``None``.

    .. versionchanged:: 2.3
        Only ASCII, UTF-8, and ISO-8859-1 are accepted for charset values.

    .. versionchanged:: 2.3
        Escaped quotes in quoted values, like ``%22`` and ``\\"``, are handled.

    .. versionchanged:: 2.2
        Option names are always converted to lowercase.

    .. versionchanged:: 2.2
        The ``multiple`` parameter was removed.

    .. versionchanged:: 0.15
        :rfc:`2231` parameter continuations are handled.

    .. versionadded:: 0.5
    """
    if value is None:
        return "", {}

    value, _, rest = value.partition(";")
    value = value.strip(" \t")
    rest = rest.strip(" \t")

    if not value or not rest:
        # empty (invalid) value, or value without options
        return value, {}

    # Collect all valid key=value parts without processing the value.
    parts: list[tuple[str, str]] = []
    scan_pos = 0
    length = len(rest)

    while scan_pos < length:
        if (m_key := _parameter_key_re.match(rest, scan_pos)) is not None:
            pk = m_key.group(1).lower()
            scan_pos = m_key.end()

            # Value may be a token.
            if (m_value := _parameter_token_value_re.match(rest, scan_pos)) is not None:
                parts.append((pk, m_value.group()))

            # Value may be a quoted string, find the closing quote.
            elif rest.startswith('"', scan_pos):
                quote_pos = scan_pos + 1

                while quote_pos < length:
                    if rest.startswith(("\\\\", '\\"'), quote_pos):
                        # Consume escaped slashes and quotes.
                        quote_pos += 2
                    elif rest.startswith('"', quote_pos):
                        # Stop at an unescaped quote.
                        quote_pos += 1
                        parts.append((pk, rest[scan_pos:quote_pos]))
                        scan_pos = quote_pos
                        break
                    else:
                        # Consume any other character.
                        quote_pos += 1
                else:
                    # Scanned all without finding closing quote.
                    break

        # Continue to the next delimited part, skipping spaces.
        if m_delim := _parameter_delimiter_re.search(rest, scan_pos):
            scan_pos = m_delim.end()
        else:
            break

    options: dict[str, str] = {}
    continuations: dict[str, list[str]] = {}
    continuation_encodings: dict[str, str] = {}

    # For each collected part, process optional charset and continuation,
    # unquote quoted values.
    for pk, pv in parts:
        encoding: str | None = None
        has_continuation = has_charset = False

        # Check for and remove markers at end of key.
        if (
            m_mark := _parameter_mark_re.search(pk)
        ) is not None and m_mark.end() - m_mark.start() > 0:
            has_continuation = m_mark.group(1) is not None
            has_charset = m_mark.group(2) is not None
            pk = pk[: m_mark.start()]

        # key*=charset''value becomes key=value, where value is percent encoded
        if has_charset:
            if (m_charset := _charset_value_re.match(pv)) is not None:
                # If there is a valid charset marker in the value, split it off.
                encoding, pv = m_charset.groups()
                # This might be the empty string, handled next.
                encoding = encoding.lower()

            # Use the prior continuation encoding if not set on this part.
            if not encoding and has_continuation:
                encoding = continuation_encodings.get(pk)

            # A safe list of encodings. Modern clients should only send ASCII or UTF-8.
            # This list will not be extended further. An invalid encoding will leave the
            # value quoted.
            if encoding in {"ascii", "us-ascii", "utf-8", "iso-8859-1"}:
                # invalid bytes are replaced during unquoting
                pv = unquote(pv, encoding=encoding)
        else:
            # Remove quotes, replace header slash escapes, replace multipart
            # percent-encoded quotes.
            pv = unquote_header_value(pv).replace("%22", '"')

        # key*0=a; key*1=b becomes key=ab
        if has_continuation:
            # Remove prior normal option.
            if pk in options:
                del options[pk]

            # For simplicity, this uses the scan order rather than enforcing
            # sequential numbering.
            if pk in continuations:
                continuations[pk].append(pv)
            else:
                continuations[pk] = [pv]

            # Further parts don't require their own charset marker.
            if encoding:
                continuation_encodings[pk] = encoding
        else:
            # Remove prior continuation option.
            if pk in continuations:
                del continuations[pk]
                continuation_encodings.pop(pk, None)

            options[pk] = pv

    options.update({pk: "".join(pv) for pk, pv in continuations.items()})
    return value, options


_q_value_re = re.compile(r"-?\d+(\.\d+)?", re.ASCII)
_TAnyAccept = t.TypeVar("_TAnyAccept", bound="ds.Accept")


@t.overload
def parse_accept_header(value: str | None) -> ds.Accept: ...


@t.overload
def parse_accept_header(value: str | None, cls: type[_TAnyAccept]) -> _TAnyAccept: ...


def parse_accept_header(
    value: str | None, cls: type[_TAnyAccept] | None = None
) -> _TAnyAccept:
    """Parse an ``Accept`` header according to
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#field.accept>`__.

    Returns an :class:`.Accept` instance, which can sort and inspect items based on
    their quality parameter. When parsing ``Accept-Charset``, ``Accept-Encoding``, or
    ``Accept-Language``, pass the appropriate :class:`.Accept` subclass.

    :param value: The header value to parse.
    :param cls: The :class:`.Accept` class to wrap the result in.
    :return: An instance of ``cls``.

    .. versionchanged:: 2.3
        Parse according to RFC 9110. Items with invalid ``q`` values are skipped.
    """
    if cls is None:
        cls = t.cast(type[_TAnyAccept], ds.Accept)

    if not value:
        return cls(None)

    result = []

    for item in parse_list_header(value):
        item, options = parse_options_header(item)

        if "q" in options:
            # pop q, remaining options are reconstructed
            q_str = options.pop("q").strip(" \t")

            if _q_value_re.fullmatch(q_str) is None:
                # ignore an invalid q
                continue

            q = float(q_str)

            if q < 0 or q > 1:
                # ignore an invalid q
                continue
        else:
            q = 1

        if options:
            # reconstruct the media type with any options
            item = dump_options_header(item, options)

        result.append((item, q))

    return cls(result)


_TAnyCC = t.TypeVar("_TAnyCC", bound="ds.cache_control._CacheControl")


@t.overload
def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
) -> ds.RequestCacheControl: ...


@t.overload
def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
    cls: type[_TAnyCC] = ...,
) -> _TAnyCC: ...


def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
    cls: type[_TAnyCC] | None = None,
) -> _TAnyCC:
    """Parse a cache control header.  The RFC differs between response and
    request cache control, this method does not.  It's your responsibility
    to not use the wrong control statements.

    .. versionadded:: 0.5
       The `cls` was added.  If not specified an immutable
       :class:`~werkzeug.datastructures.RequestCacheControl` is returned.

    :param value: a cache control header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.CacheControl`
                      object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.RequestCacheControl` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast("type[_TAnyCC]", ds.RequestCacheControl)

    if not value:
        return cls((), on_update)

    return cls(parse_dict_header(value), on_update)


_TAnyCSP = t.TypeVar("_TAnyCSP", bound="ds.ContentSecurityPolicy")


@t.overload
def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
) -> ds.ContentSecurityPolicy: ...


@t.overload
def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
    cls: type[_TAnyCSP] = ...,
) -> _TAnyCSP: ...


def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
    cls: type[_TAnyCSP] | None = None,
) -> _TAnyCSP:
    """Parse a Content Security Policy header.

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    :param value: a csp header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.ContentSecurityPolicy` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast("type[_TAnyCSP]", ds.ContentSecurityPolicy)

    if value is None:
        return cls((), on_update)

    items = []

    for policy in value.split(";"):
        policy = policy.strip(" \t")

        # Ignore badly formatted policies (no space)
        if " " in policy:
            directive, _, value = policy.strip(" \t").partition(" ")
            items.append((directive.strip(" \t"), value.strip(" \t")))

    return cls(items, on_update)


def parse_set_header(
    value: str | None,
    on_update: t.Callable[[ds.HeaderSet], None] | None = None,
) -> ds.HeaderSet:
    """Parse a set-like header and return a
    :class:`~werkzeug.datastructures.HeaderSet` object:

    >>> hs = parse_set_header('token, "quoted value"')

    The return value is an object that treats the items case-insensitively
    and keeps the order of the items:

    >>> 'TOKEN' in hs
    True
    >>> hs.index('quoted value')
    1
    >>> hs
    HeaderSet(['token', 'quoted value'])

    To create a header from the :class:`HeaderSet` again, use the
    :func:`dump_header` function.

    :param value: a set header to be parsed.
    :param on_update: an optional callable that is called every time a
                      value on the :class:`~werkzeug.datastructures.HeaderSet`
                      object is changed.
    :return: a :class:`~werkzeug.datastructures.HeaderSet`
    """
    if not value:
        return ds.HeaderSet(None, on_update)
    return ds.HeaderSet(parse_list_header(value), on_update)


def parse_if_range_header(value: str | None) -> ds.IfRange:
    """Parses an if-range header which can be an etag or a date.  Returns
    a :class:`~werkzeug.datastructures.IfRange` object.

    .. versionchanged:: 2.0
        If the value represents a datetime, it is timezone-aware.

    .. versionadded:: 0.7
    """
    if not value:
        return ds.IfRange()
    date = parse_date(value)
    if date is not None:
        return ds.IfRange(date=date)
    # drop weakness information
    return ds.IfRange(unquote_etag(value)[0])


def parse_range_header(
    value: str | None, make_inclusive: bool = True
) -> ds.Range | None:
    """Parses a range header into a :class:`~werkzeug.datastructures.Range`
    object.  If the header is missing or malformed `None` is returned.
    `ranges` is a list of ``(start, stop)`` tuples where the ranges are
    non-inclusive.

    .. versionadded:: 0.7
    """
    if not value or "=" not in value:
        return None

    ranges = []
    last_end = 0
    units, _, rng = value.partition("=")
    units = units.strip(" \t").lower()

    for item in rng.split(","):
        item = item.strip(" \t")
        if "-" not in item:
            return None
        if item.startswith("-"):
            if last_end < 0:
                return None
            try:
                begin = _plain_int(item)
            except ValueError:
                return None

            # -0 will parse to 0, and is an invalid suffix length.
            if begin == 0:
                return None

            end = None
            last_end = -1
        elif "-" in item:
            begin_str, _, end_str = item.partition("-")
            begin_str = begin_str.strip(" \t")
            end_str = end_str.strip(" \t")

            try:
                begin = _plain_int(begin_str)
            except ValueError:
                return None

            if begin < last_end or last_end < 0:
                return None
            if end_str:
                try:
                    end = _plain_int(end_str) + 1
                except ValueError:
                    return None

                if begin >= end:
                    return None
            else:
                end = None
            last_end = end if end is not None else -1
        ranges.append((begin, end))

    return ds.Range(units, ranges)


def parse_content_range_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentRange], None] | None = None,
) -> ds.ContentRange | None:
    """Parses a range header into a
    :class:`~werkzeug.datastructures.ContentRange` object or `None` if
    parsing is not possible.

    .. versionadded:: 0.7

    :param value: a content range header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.ContentRange`
                      object is changed.
    """
    if value is None:
        return None
    try:
        units, _, rangedef = (value or "").strip(" \t").partition(" ")
    except ValueError:
        return None

    if "/" not in rangedef:
        return None
    rng, _, length_str = rangedef.partition("/")
    if length_str == "*":
        length = None
    else:
        try:
            length = _plain_int(length_str)
        except ValueError:
            return None

    if rng == "*":
        if not is_byte_range_valid(None, None, length):
            return None

        return ds.ContentRange(units, None, None, length, on_update=on_update)
    elif "-" not in rng:
        return None

    start_str, _, stop_str = rng.partition("-")
    try:
        start = _plain_int(start_str)
        stop = _plain_int(stop_str) + 1
    except ValueError:
        return None

    if is_byte_range_valid(start, stop, length):
        return ds.ContentRange(units, start, stop, length, on_update=on_update)

    return None


def quote_etag(etag: str, weak: bool = False) -> str:
    """Quote an etag.

    :param etag: the etag to quote.
    :param weak: set to `True` to tag it "weak".
    """
    if '"' in etag:
        raise ValueError("invalid etag")

    if weak:
        return f'W/"{etag}"'

    return f'"{etag}"'


@t.overload
def unquote_etag(etag: str) -> tuple[str, bool]: ...
@t.overload
def unquote_etag(etag: None) -> tuple[None, None]: ...
def unquote_etag(
    etag: str | None,
) -> tuple[str, bool] | tuple[None, None]:
    """Unquote a single etag:

    >>> unquote_etag('W/"bar"')
    ('bar', True)
    >>> unquote_etag('"bar"')
    ('bar', False)

    :param etag: the etag identifier to unquote.
    :return: a ``(etag, weak)`` tuple.
    """
    if not etag:
        return None, None

    weak = False
    start = 0

    if etag.startswith(("W/", "w/")):
        weak = True
        start = 2

    if etag.startswith('"', start) and etag.endswith('"', start):
        return etag[start + 1 : -1], weak

    # invalid unquoted
    return etag[start:], weak


_etag_re = re.compile(
    r"""
    [ \t]*  # ignore leading space
    ([Ww]/)?  # optional weak marker
    (?:
        "([^"]*)"  # quoted value
    |
        ([^" \t,]+)  # invalid unquoted value, exclude syntax characters
    )
    [ \t]*  # ignore trailing space
    (?:,|\Z)  # only if followed by comma or end
    """,
    flags=re.ASCII | re.VERBOSE,
)


def parse_etags(value: str | None) -> ds.ETags:
    """Parse an etag header.

    :param value: the tag header to parse
    :return: an :class:`~werkzeug.datastructures.ETags` object.
    """
    if not value:
        return ds.ETags()

    if value == "*":
        return ds.ETags(star_tag=True)

    strong = []
    weak = []
    pos = 0

    while True:
        if (m := _etag_re.match(value, pos)) is None:
            # Skip invalid chars until the next comma.
            if (pos := value.find(",", pos) + 1) == 0:
                break

            continue

        is_weak, tag, invalid_unquoted = m.groups()
        pos = m.end()
        tag = tag or invalid_unquoted

        if is_weak:
            weak.append(tag)
        else:
            strong.append(tag)

    return ds.ETags(strong, weak)


def generate_etag(data: bytes) -> str:
    """Generate an etag for some data.

    .. versionchanged:: 2.0
        Use SHA-1. MD5 may not be available in some environments.
    """
    return sha1(data).hexdigest()


def parse_date(value: str | None) -> datetime | None:
    """Parse an :rfc:`2822` date into a timezone-aware
    :class:`datetime.datetime` object, or ``None`` if parsing fails.

    This is a wrapper for :func:`email.utils.parsedate_to_datetime`. It
    returns ``None`` if parsing fails instead of raising an exception,
    and always returns a timezone-aware datetime object. If the string
    doesn't have timezone information, it is assumed to be UTC.

    :param value: A string with a supported date format.

    .. versionchanged:: 2.0
        Return a timezone-aware datetime object. Use
        ``email.utils.parsedate_to_datetime``.
    """
    if value is None:
        return None

    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt


def http_date(
    timestamp: datetime | date | int | float | struct_time | None = None,
) -> str:
    """Format a datetime object or timestamp into an :rfc:`2822` date
    string.

    This is a wrapper for :func:`email.utils.format_datetime`. It
    assumes naive datetime objects are in UTC instead of raising an
    exception.

    :param timestamp: The datetime or timestamp to format. Defaults to
        the current time.

    .. versionchanged:: 2.0
        Use ``email.utils.format_datetime``. Accept ``date`` objects.
    """
    if isinstance(timestamp, date):
        if not isinstance(timestamp, datetime):
            # Assume plain date is midnight UTC.
            timestamp = datetime.combine(timestamp, time(), tzinfo=timezone.utc)
        else:
            # Ensure datetime is timezone-aware.
            timestamp = _dt_as_utc(timestamp)

        return email.utils.format_datetime(timestamp, usegmt=True)

    if isinstance(timestamp, struct_time):
        timestamp = mktime(timestamp)

    return email.utils.formatdate(timestamp, usegmt=True)


def parse_age(value: str | None = None) -> timedelta | None:
    """Parses a base-10 integer count of seconds into a timedelta.

    If parsing fails, the return value is `None`.

    :param value: a string consisting of an integer represented in base-10
    :return: a :class:`datetime.timedelta` object or `None`.
    """
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    try:
        return timedelta(seconds=seconds)
    except OverflowError:
        return None


def dump_age(age: timedelta | int | None = None) -> str | None:
    """Formats the duration as a base-10 integer.

    :param age: should be an integer number of seconds,
                a :class:`datetime.timedelta` object, or,
                if the age is unknown, `None` (default).
    """
    if age is None:
        return None
    if isinstance(age, timedelta):
        age = int(age.total_seconds())
    else:
        age = int(age)

    if age < 0:
        raise ValueError("age cannot be negative")

    return str(age)


def is_resource_modified(
    environ: WSGIEnvironment,
    etag: str | None = None,
    data: bytes | None = None,
    last_modified: datetime | str | None = None,
    ignore_if_range: bool = True,
) -> bool:
    """Convenience method for conditional requests.

    :param environ: the WSGI environment of the request to be checked.
    :param etag: the etag for the response for comparison.
    :param data: or alternatively the data of the response to automatically
                 generate an etag using :func:`generate_etag`.
    :param last_modified: an optional date of the last modification.
    :param ignore_if_range: If `False`, `If-Range` header will be taken into
                            account.
    :return: `True` if the resource was modified, otherwise `False`.

    .. versionchanged:: 2.0
        SHA-1 is used to generate an etag value for the data. MD5 may
        not be available in some environments.

    .. versionchanged:: 1.0.0
        The check is run for methods other than ``GET`` and ``HEAD``.
    """
    return _sansio_http.is_resource_modified(
        http_range=environ.get("HTTP_RANGE"),
        http_if_range=environ.get("HTTP_IF_RANGE"),
        http_if_modified_since=environ.get("HTTP_IF_MODIFIED_SINCE"),
        http_if_none_match=environ.get("HTTP_IF_NONE_MATCH"),
        http_if_match=environ.get("HTTP_IF_MATCH"),
        etag=etag,
        data=data,
        last_modified=last_modified,
        ignore_if_range=ignore_if_range,
    )


def remove_entity_headers(
    headers: ds.Headers | list[tuple[str, str]],
    allowed: t.Iterable[str] = ("expires", "content-location"),
) -> None:
    """Remove all entity headers from a list or :class:`Headers` object.  This
    operation works in-place.  `Expires` and `Content-Location` headers are
    by default not removed.  The reason for this is :rfc:`2616` section
    10.3.5 which specifies some entity headers that should be sent.

    .. versionchanged:: 0.5
       added `allowed` parameter.

    :param headers: a list or :class:`Headers` object.
    :param allowed: a list of headers that should still be allowed even though
                    they are entity headers.
    """
    allowed = {x.lower() for x in allowed}
    headers[:] = [
        (key, value)
        for key, value in headers
        if not is_entity_header(key) or key.lower() in allowed
    ]


def remove_hop_by_hop_headers(headers: ds.Headers | list[tuple[str, str]]) -> None:
    """Remove all HTTP/1.1 "Hop-by-Hop" headers from a list or
    :class:`Headers` object.  This operation works in-place.

    .. versionadded:: 0.5

    :param headers: a list or :class:`Headers` object.
    """
    headers[:] = [
        (key, value) for key, value in headers if not is_hop_by_hop_header(key)
    ]


def is_entity_header(header: str) -> bool:
    """Check if a header is an entity header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an entity header, `False` otherwise.
    """
    return header.lower() in _entity_headers


def is_hop_by_hop_header(header: str) -> bool:
    """Check if a header is an HTTP/1.1 "Hop-by-Hop" header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an HTTP/1.1 "Hop-by-Hop" header, `False` otherwise.
    """
    return header.lower() in _hop_by_hop_headers


def parse_cookie(
    header: WSGIEnvironment | str | None,
    cls: type[ds.MultiDict[str, str]] | None = None,
) -> ds.MultiDict[str, str]:
    """Parse a cookie from a string or WSGI environ.

    The same key can be provided multiple times, the values are stored
    in-order. The default :class:`MultiDict` will have the first value
    first, and all values can be retrieved with
    :meth:`MultiDict.getlist`.

    :param header: The cookie header as a string, or a WSGI environ dict
        with a ``HTTP_COOKIE`` key.
    :param cls: A dict-like class to store the parsed cookies in.
        Defaults to :class:`MultiDict`.

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` and ``errors`` parameters, were removed.

    .. versionchanged:: 1.0
        Returns a :class:`MultiDict` instead of a ``TypeConversionDict``.

    .. versionchanged:: 0.5
        Returns a :class:`TypeConversionDict` instead of a regular dict. The ``cls``
        parameter was added.
    """
    if isinstance(header, dict):
        cookie = header.get("HTTP_COOKIE")
    else:
        cookie = header

    if cookie:
        cookie = cookie.encode("latin1").decode()

    return _sansio_http.parse_cookie(cookie=cookie, cls=cls)


_cookie_no_quote_re = re.compile(r"[\w!#$%&'()*+\-./:<=>?@\[\]^`{|}~]*", re.A)
_cookie_slash_re = re.compile(rb"[\x00-\x19\",;\\\x7f-\xff]", re.A)
_cookie_slash_map = {b'"': b'\\"', b"\\": b"\\\\"}
_cookie_slash_map.update(
    (v.to_bytes(1, "big"), b"\\%03o" % v)
    for v in [*range(0x20), *b",;", *range(0x7F, 256)]
)


def dump_cookie(
    key: str,
    value: str = "",
    max_age: timedelta | int | None = None,
    expires: str | datetime | int | float | None = None,
    path: str | None = "/",
    domain: str | None = None,
    secure: bool = False,
    httponly: bool = False,
    sync_expires: bool = True,
    max_size: int = 4093,
    samesite: str | None = None,
    partitioned: bool = False,
) -> str:
    """Create a Set-Cookie header without the ``Set-Cookie`` prefix.

    The return value is usually restricted to ascii as the vast majority
    of values are properly escaped, but that is no guarantee. It's
    tunneled through latin1 as required by :pep:`3333`.

    The return value is not ASCII safe if the key contains unicode
    characters.  This is technically against the specification but
    happens in the wild.  It's strongly recommended to not use
    non-ASCII values for the keys.

    :param key: The cookie key. This is assumed to be trusted and valid, it
        must not come from untrusted user input.
    :param value: The cookie value. It will be quoted if it contains characters
        not allowed by RFC 6265; ``\123`` octal escapes are used to encode
        non-ASCII bytes.
    :param max_age: should be a number of seconds, or `None` (default) if
                    the cookie should last only as long as the client's
                    browser session.  Additionally `timedelta` objects
                    are accepted, too.
    :param expires: should be a `datetime` object or unix timestamp.
    :param path: limits the cookie to a given path, per default it will
                 span the whole domain.
    :param domain: Use this if you want to set a cross-domain cookie. For
                   example, ``domain="example.com"`` will set a cookie
                   that is readable by the domain ``www.example.com``,
                   ``foo.example.com`` etc. Otherwise, a cookie will only
                   be readable by the domain that set it.
    :param secure: The cookie will only be available via HTTPS
    :param httponly: disallow JavaScript to access the cookie.  This is an
                     extension to the cookie standard and probably not
                     supported by all browsers.
    :param charset: the encoding for string values.
    :param sync_expires: automatically set expires if max_age is defined
                         but expires not.
    :param max_size: Warn if the final header value exceeds this size. The
        default, 4093, should be safely `supported by most browsers
        <cookie_>`_. Set to 0 to disable this check.
    :param samesite: Limits the scope of the cookie such that it will
        only be attached to requests if those requests are same-site.
    :param partitioned: Opts the cookie into partitioned storage. This
        will also set secure to True

    .. _`cookie`: http://browsercookielimits.squawky.net/

    .. versionchanged:: 3.1
        The ``partitioned`` parameter was added.

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` parameter, were removed.

    .. versionchanged:: 2.3.3
        The ``path`` parameter is ``/`` by default.

    .. versionchanged:: 2.3.1
        The value allows more characters without quoting.

    .. versionchanged:: 2.3
        ``localhost`` and other names without a dot are allowed for the domain. A
        leading dot is ignored.

    .. versionchanged:: 2.3
        The ``path`` parameter is ``None`` by default.

    .. versionchanged:: 1.0.0
        The string ``'None'`` is accepted for ``samesite``.
    """
    if path is not None:
        # safe = https://url.spec.whatwg.org/#url-path-segment-string
        # as well as percent for things that are already quoted
        # excluding semicolon since it's part of the header syntax
        path = quote(path, safe="%!$&'()*+,/:=@")

    if domain:
        domain = domain.partition(":")[0].lstrip(".").encode("idna").decode("ascii")

    if isinstance(max_age, timedelta):
        max_age = int(max_age.total_seconds())

    if expires is not None:
        if not isinstance(expires, str):
            expires = http_date(expires)
    elif max_age is not None and sync_expires:
        expires = http_date(datetime.now(tz=timezone.utc).timestamp() + max_age)

    if samesite is not None:
        samesite = samesite.title()

        if samesite not in {"Strict", "Lax", "None"}:
            raise ValueError("SameSite must be 'Strict', 'Lax', or 'None'.")

    if partitioned:
        secure = True

    # Quote value if it contains characters not allowed by RFC 6265. Slash-escape with
    # three octal digits, which matches http.cookies, although the RFC suggests base64.
    if not _cookie_no_quote_re.fullmatch(value):
        # Work with bytes here, since a UTF-8 character could be multiple bytes.
        value = _cookie_slash_re.sub(
            lambda m: _cookie_slash_map[m.group()], value.encode()
        ).decode("ascii")
        value = f'"{value}"'

    # Send a non-ASCII key as mojibake. Everything else should already be ASCII.
    # TODO Remove encoding dance, it seems like clients accept UTF-8 keys
    buf = [f"{key.encode().decode('latin1')}={value}"]

    for k, v in (
        ("Domain", domain),
        ("Expires", expires),
        ("Max-Age", max_age),
        ("Secure", secure),
        ("HttpOnly", httponly),
        ("Path", path),
        ("SameSite", samesite),
        ("Partitioned", partitioned),
    ):
        if v is None or v is False:
            continue

        if v is True:
            buf.append(k)
            continue

        buf.append(f"{k}={v}")

    rv = "; ".join(buf)

    # Warn if the final value of the cookie is larger than the limit. If the cookie is
    # too large, then it may be silently ignored by the browser, which can be quite hard
    # to debug.
    cookie_size = len(rv)

    if max_size and cookie_size > max_size:
        value_size = len(value)
        warnings.warn(
            f"The '{key}' cookie is too large: the value was {value_size} bytes but the"
            f" header required {cookie_size - value_size} extra bytes. The final size"
            f" was {cookie_size} bytes but the limit is {max_size} bytes. Browsers may"
            " silently ignore cookies larger than this.",
            stacklevel=2,
        )

    return rv


def is_byte_range_valid(
    start: int | None, stop: int | None, length: int | None
) -> bool:
    """Checks if a given byte content range is valid for the given length.

    .. versionadded:: 0.7
    """
    if (start is None) != (stop is None):
        return False
    elif start is None:
        return length is None or length >= 0
    elif length is None:
        return 0 <= start < stop  # type: ignore
    elif start >= stop:  # type: ignore
        return False
    return 0 <= start < length


# circular dependencies
from . import datastructures as ds  # noqa: E402
from .sansio import http as _sansio_http  # noqa: E402
<|endoftext|>from __future__ import annotations

import codecs
import re
import typing as t
import urllib.parse
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from .datastructures import iter_multi_items


def _codec_error_url_quote(e: UnicodeError) -> tuple[str, int]:
    """Used in :func:`uri_to_iri` after unquoting to re-quote any
    invalid bytes.
    """
    # the docs state that UnicodeError does have these attributes,
    # but mypy isn't picking them up
    out = quote(e.object[e.start : e.end], safe="")  # type: ignore
    return out, e.end  # type: ignore


codecs.register_error("werkzeug.url_quote", _codec_error_url_quote)


def _make_unquote_part(name: str, chars: str) -> t.Callable[[str], str]:
    """Create a function that unquotes all percent encoded characters except those
    given. This allows working with unquoted characters if possible while not changing
    the meaning of a given part of a URL.
    """
    choices = "|".join(f"{ord(c):02X}" for c in sorted(chars))
    pattern = re.compile(f"((?:%(?:{choices}))+)", re.I)

    def _unquote_partial(value: str) -> str:
        parts = iter(pattern.split(value))
        out = []

        for part in parts:
            out.append(unquote(part, "utf-8", "werkzeug.url_quote"))
            out.append(next(parts, ""))

        return "".join(out)

    _unquote_partial.__name__ = f"_unquote_{name}"
    return _unquote_partial


# characters that should remain quoted in URL parts
# based on https://url.spec.whatwg.org/#percent-encoded-bytes
# always keep all controls, space, and % quoted
_always_unsafe = bytes((*range(0x21), 0x25, 0x7F)).decode()
_unquote_fragment = _make_unquote_part("fragment", _always_unsafe)
_unquote_query = _make_unquote_part("query", _always_unsafe + "&=+#")
_unquote_path = _make_unquote_part("path", _always_unsafe + "/?#")
_unquote_user = _make_unquote_part("user", _always_unsafe + ":@/?#")


def uri_to_iri(uri: str) -> str:
    """Convert a URI to an IRI. All valid UTF-8 characters are unquoted,
    leaving all reserved and invalid characters quoted. If the URL has
    a domain, it is decoded from Punycode.

    >>> uri_to_iri("http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF")
    'http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF'

    :param uri: The URI to convert.

    .. versionchanged:: 3.1.9
        Empty username, password, and port 0 are preserved.

    .. versionchanged:: 3.0
        Passing a tuple or bytes, and the ``charset`` and ``errors`` parameters,
        are removed.

    .. versionchanged:: 2.3
        Which characters remain quoted is specific to each part of the URL.

    .. versionchanged:: 0.15
        All reserved and invalid characters remain quoted. Previously,
        only some reserved characters were preserved, and invalid bytes
        were replaced instead of left quoted.

    .. versionadded:: 0.6
    """
    parts = urlsplit(uri)
    path = _unquote_path(parts.path)
    query = _unquote_query(parts.query)
    fragment = _unquote_fragment(parts.fragment)

    if parts.hostname:
        netloc = _decode_idna(parts.hostname)
    else:
        netloc = ""

    if ":" in netloc:
        netloc = f"[{netloc}]"

    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"

    if parts.username is not None:
        auth = _unquote_user(parts.username)

        if parts.password is not None:
            password = _unquote_user(parts.password)
            auth = f"{auth}:{password}"

        netloc = f"{auth}@{netloc}"

    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def iri_to_uri(iri: str) -> str:
    """Convert an IRI to a URI. All non-ASCII and unsafe characters are
    quoted. If the URL has a domain, it is encoded to Punycode.

    >>> iri_to_uri('http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF')
    'http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF'

    :param iri: The IRI to convert.

    .. versionchanged:: 3.1.9
        Empty username, password, and port 0 are preserved.

    .. versionchanged:: 3.0
        Passing a tuple or bytes, the ``charset`` and ``errors`` parameters,
        and the ``safe_conversion`` parameter, are removed.

    .. versionchanged:: 2.3
        Which characters remain unquoted is specific to each part of the URL.

    .. versionchanged:: 0.15
        All reserved characters remain unquoted. Previously, only some reserved
        characters were left unquoted.

    .. versionchanged:: 0.9.6
       The ``safe_conversion`` parameter was added.

    .. versionadded:: 0.6
    """
    parts = urlsplit(iri)
    # safe = https://url.spec.whatwg.org/#url-path-segment-string
    # as well as percent for things that are already quoted
    path = quote(parts.path, safe="%!$&'()*+,/:;=@")
    query = quote(parts.query, safe="%!$&'()*+,/:;=?@")
    fragment = quote(parts.fragment, safe="%!#$&'()*+,/:;=?@")

    if parts.hostname:
        netloc = parts.hostname.encode("idna").decode("ascii")
    else:
        netloc = ""

    if ":" in netloc:
        netloc = f"[{netloc}]"

    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"

    if parts.username is not None:
        auth = quote(parts.username, safe="%!$&'()*+,;=")

        if parts.password is not None:
            password = quote(parts.password, safe="%!$&'()*+,;=")
            auth = f"{auth}:{password}"

        netloc = f"{auth}@{netloc}"

    return urlunsplit((parts.scheme, netloc, path, query, fragment))


# Python < 3.12
# itms-services was worked around in previous iri_to_uri implementations, but
# we can tell Python directly that it needs to preserve the //.
if "itms-services" not in urllib.parse.uses_netloc:
    urllib.parse.uses_netloc.append("itms-services")


def _decode_idna(domain: str) -> str:
    try:
        data = domain.encode("ascii")
    except UnicodeEncodeError:
        # If the domain is not ASCII, it's decoded already.
        return domain

    try:
        # Try decoding in one shot.
        return data.decode("idna")
    except UnicodeDecodeError:
        pass

    # Decode each part separately, leaving invalid parts as punycode.
    parts = []

    for part in data.split(b"."):
        try:
            parts.append(part.decode("idna"))
        except UnicodeDecodeError:
            parts.append(part.decode("ascii"))

    return ".".join(parts)


def _urlencode(query: t.Mapping[str, str] | t.Iterable[tuple[str, str]]) -> str:
    items = [x for x in iter_multi_items(query) if x[1] is not None]
    # safe = https://url.spec.whatwg.org/#percent-encoded-bytes
    return urlencode(items, safe="!$'()*,/:;?@")
<|endoftext|>from __future__ import annotations

import dataclasses
import mimetypes
import sys
import typing as t
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from ._internal import _get_environ
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_cookie
from .http import parse_date
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import _urlencode
from .urls import iri_to_uri
from .utils import cached_property
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def stream_encode_multipart(
    data: t.Mapping[str, t.Any],
    use_tempfile: bool = True,
    threshold: int = 1024 * 500,
    boundary: str | None = None,
) -> tuple[t.IO[bytes], int, str]:
    """Encode a dict of values (either strings or file descriptors or
    :class:`FileStorage` objects.) into a multipart encoded string stored
    in a file descriptor.

    This is not secure and must only be used during testing, never in applications.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.
    """
    if boundary is None:
        boundary = f"---------------WerkzeugFormPart_{time()}{random()}"

    stream: t.IO[bytes] = BytesIO()
    total_length = 0
    on_disk = False
    write_binary: t.Callable[[bytes], int]

    if use_tempfile:

        def write_binary(s: bytes) -> int:
            nonlocal stream, total_length, on_disk

            if on_disk:
                return stream.write(s)
            else:
                length = len(s)

                if length + total_length <= threshold:
                    stream.write(s)
                else:
                    new_stream = t.cast(t.IO[bytes], TemporaryFile("wb+"))
                    new_stream.write(stream.getvalue())  # type: ignore
                    new_stream.write(s)
                    stream = new_stream
                    on_disk = True

                total_length += length
                return length

    else:
        write_binary = stream.write

    encoder = MultipartEncoder(boundary.encode())
    write_binary(encoder.send_event(Preamble(data=b"")))
    for key, value in _iter_data(data):
        reader = getattr(value, "read", None)
        if reader is not None:
            filename = getattr(value, "filename", getattr(value, "name", None))
            content_type = getattr(value, "content_type", None)
            if content_type is None:
                content_type = (
                    filename
                    and mimetypes.guess_type(filename)[0]
                    or "application/octet-stream"
                )
            headers = value.headers
            headers.update([("Content-Type", content_type)])
            if filename is None:
                write_binary(encoder.send_event(Field(name=key, headers=headers)))
            else:
                write_binary(
                    encoder.send_event(
                        File(name=key, filename=filename, headers=headers)
                    )
                )
            while True:
                chunk = reader(16384)

                if not chunk:
                    write_binary(encoder.send_event(Data(data=chunk, more_data=False)))
                    break

                write_binary(encoder.send_event(Data(data=chunk, more_data=True)))
        else:
            if not isinstance(value, str):
                value = str(value)
            write_binary(encoder.send_event(Field(name=key, headers=Headers())))
            write_binary(encoder.send_event(Data(data=value.encode(), more_data=False)))

    write_binary(encoder.send_event(Epilogue(data=b"")))

    length = stream.tell()
    stream.seek(0)
    return stream, length, boundary


def encode_multipart(
    values: t.Mapping[str, t.Any], boundary: str | None = None
) -> tuple[str, bytes]:
    """Like `stream_encode_multipart` but returns a tuple in the form
    (``boundary``, ``data``) where data is bytes.

    This is not secure and must only be used during testing, never in applications.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.
    """
    stream, length, boundary = stream_encode_multipart(
        values, use_tempfile=False, boundary=boundary
    )
    return boundary, stream.read()


def _iter_data(data: t.Mapping[str, t.Any]) -> t.Iterator[tuple[str, t.Any]]:
    """Iterate over a mapping that might have a list of values, yielding
    all key, value pairs. Almost like iter_multi_items but only allows
    lists, not tuples, of values so tuples can be used for files.
    """
    if isinstance(data, MultiDict):
        yield from data.items(multi=True)
    else:
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    yield key, v
            else:
                yield key, value


_TAnyMultiDict = t.TypeVar("_TAnyMultiDict", bound="MultiDict[t.Any, t.Any]")


class EnvironBuilder:
    """This class can be used to conveniently create a WSGI environment
    for testing purposes.  It can be used to quickly create WSGI environments
    or request objects from arbitrary data.

    The signature of this class is also used in some other places as of
    Werkzeug 0.5 (:func:`create_environ`, :meth:`Response.from_values`,
    :meth:`Client.open`).  Because of this most of the functionality is
    available through the constructor alone.

    Files and regular form data can be manipulated independently of each
    other with the :attr:`form` and :attr:`files` attributes, but are
    passed with the same argument to the constructor: `data`.

    `data` can be any of these values:

    -   a `str` or `bytes` object: The object is converted into an
        :attr:`input_stream`, the :attr:`content_length` is set and you have to
        provide a :attr:`content_type`.
    -   a `dict` or :class:`MultiDict`: The keys have to be strings. The values
        have to be either any of the following objects, or a list of any of the
        following objects:

        -   a :class:`file`-like object:  These are converted into
            :class:`FileStorage` objects automatically.
        -   a `tuple`:  The :meth:`~FileMultiDict.add_file` method is called
            with the key and the unpacked `tuple` items as positional
            arguments.
        -   a `str`:  The string is set as form data for the associated key.
    -   a file-like object: The object content is loaded in memory and then
        handled like a regular `str` or a `bytes`.

    :param path: the path of the request.  In the WSGI environment this will
                 end up as `PATH_INFO`.  If the `query_string` is not defined
                 and there is a question mark in the `path` everything after
                 it is used as query string.
    :param base_url: the base URL is a URL that is used to extract the WSGI
                     URL scheme, host (server name + server port) and the
                     script root (`SCRIPT_NAME`).
    :param query_string: an optional string or dict with URL parameters.
    :param method: the HTTP method to use, defaults to `GET`.
    :param input_stream: an optional input stream.  Do not specify this and
                         `data`.  As soon as an input stream is set you can't
                         modify :attr:`args` and :attr:`files` unless you
                         set the :attr:`input_stream` to `None` again.
    :param content_type: The content type for the request.  As of 0.5 you
                         don't have to provide this when specifying files
                         and form data via `data`.
    :param content_length: The content length for the request.  You don't
                           have to specify this when providing data via
                           `data`.
    :param errors_stream: an optional error stream that is used for
                          `wsgi.errors`.  Defaults to :data:`stderr`.
    :param multithread: controls `wsgi.multithread`.  Defaults to `False`.
    :param multiprocess: controls `wsgi.multiprocess`.  Defaults to `False`.
    :param run_once: controls `wsgi.run_once`.  Defaults to `False`.
    :param headers: an optional list or :class:`Headers` object of headers.
    :param data: a string or dict of form data or a file-object.
                 See explanation above.
    :param json: An object to be serialized and assigned to ``data``.
        Defaults the content type to ``"application/json"``.
        Serialized with the function assigned to :attr:`json_dumps`.
    :param environ_base: an optional dict of environment defaults.
    :param environ_overrides: an optional dict of environment overrides.
    :param auth: An authorization object to use for the
        ``Authorization`` header value. A ``(username, password)`` tuple
        is a shortcut for ``Basic`` authorization.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.

    .. versionchanged:: 2.1
        ``CONTENT_TYPE`` and ``CONTENT_LENGTH`` are not duplicated as
        header keys in the environ.

    .. versionchanged:: 2.0
        ``REQUEST_URI`` and ``RAW_URI`` is the full raw URI including
        the query string, not only the path.

    .. versionchanged:: 2.0
        The default :attr:`request_class` is ``Request`` instead of
        ``BaseRequest``.

    .. versionadded:: 2.0
       Added the ``auth`` parameter.

    .. versionadded:: 0.15
        The ``json`` param and :meth:`json_dumps` method.

    .. versionadded:: 0.15
        The environ has keys ``REQUEST_URI`` and ``RAW_URI`` containing
        the path before percent-decoding. This is not part of the WSGI
        PEP, but many WSGI servers include it.

    .. versionchanged:: 0.6
       ``path`` and ``base_url`` can now be unicode strings that are
       encoded with :func:`iri_to_uri`.
    """

    #: the server protocol to use.  defaults to HTTP/1.1
    server_protocol = "HTTP/1.1"

    #: the wsgi version to use.  defaults to (1, 0)
    wsgi_version = (1, 0)

    #: The default request class used by :meth:`get_request`.
    request_class = Request

    import json

    #: The serialization function used when ``json`` is passed.
    json_dumps = staticmethod(json.dumps)
    del json

    _args: MultiDict[str, str] | None
    _query_string: str | None
    _input_stream: t.IO[bytes] | None
    _form: MultiDict[str, str] | None
    _files: FileMultiDict | None

    def __init__(
        self,
        path: str = "/",
        base_url: str | None = None,
        query_string: t.Mapping[str, str] | str | None = None,
        method: str = "GET",
        input_stream: t.IO[bytes] | None = None,
        content_type: str | None = None,
        content_length: int | None = None,
        errors_stream: t.IO[str] | None = None,
        multithread: bool = False,
        multiprocess: bool = False,
        run_once: bool = False,
        headers: Headers | t.Iterable[tuple[str, str]] | None = None,
        data: None | (t.IO[bytes] | str | bytes | t.Mapping[str, t.Any]) = None,
        environ_base: t.Mapping[str, t.Any] | None = None,
        environ_overrides: t.Mapping[str, t.Any] | None = None,
        mimetype: str | None = None,
        json: t.Mapping[str, t.Any] | None = None,
        auth: Authorization | tuple[str, str] | None = None,
    ) -> None:
        if query_string is not None and "?" in path:
            raise ValueError("Query string is defined in the path and as an argument")
        request_uri = urlsplit(path)
        if query_string is None and "?" in path:
            query_string = request_uri.query

        self.path = iri_to_uri(request_uri.path)
        self.request_uri = path
        if base_url is not None:
            base_url = iri_to_uri(base_url)
        self.base_url = base_url
        if isinstance(query_string, str):
            self.query_string = query_string
        else:
            if query_string is None:
                query_string = MultiDict()
            elif not isinstance(query_string, MultiDict):
                query_string = MultiDict(query_string)
            self.args = query_string
        self.method = method
        if headers is None:
            headers = Headers()
        elif not isinstance(headers, Headers):
            headers = Headers(headers)
        self.headers = headers
        if content_type is not None:
            self.content_type = content_type
        if errors_stream is None:
            errors_stream = sys.stderr
        self.errors_stream = errors_stream
        self.multithread = multithread
        self.multiprocess = multiprocess
        self.run_once = run_once
        self.environ_base = environ_base
        self.environ_overrides = environ_overrides
        self.input_stream = input_stream
        self.content_length = content_length
        self.closed = False

        if auth is not None:
            if isinstance(auth, tuple):
                auth = Authorization(
                    "basic", {"username": auth[0], "password": auth[1]}
                )

            self.headers.set("Authorization", auth.to_header())

        if json is not None:
            if data is not None:
                raise TypeError("can't provide both json and data")

            data = self.json_dumps(json)

            if self.content_type is None:
                self.content_type = "application/json"

        if data:
            if input_stream is not None:
                raise TypeError("can't provide input stream and data")
            if hasattr(data, "read"):
                data = data.read()
            if isinstance(data, str):
                data = data.encode()
            if isinstance(data, bytes):
                self.input_stream = BytesIO(data)
                if self.content_length is None:
                    self.content_length = len(data)
            else:
                for key, value in _iter_data(data):
                    if isinstance(value, (tuple, dict)) or hasattr(value, "read"):
                        self._add_file_from_data(key, value)
                    else:
                        self.form.setlistdefault(key).append(value)

        if mimetype is not None:
            self.mimetype = mimetype

    @classmethod
    def from_environ(cls, environ: WSGIEnvironment, **kwargs: t.Any) -> EnvironBuilder:
        """Turn an environ dict back into a builder. Any extra kwargs
        override the args extracted from the environ.

        .. versionchanged:: 2.0
            Path and query values are passed through the WSGI decoding
            dance to avoid double encoding.

        .. versionadded:: 0.15
        """
        headers = Headers(EnvironHeaders(environ))
        out = {
            "path": _wsgi_decoding_dance(environ["PATH_INFO"]),
            "base_url": cls._make_base_url(
                environ["wsgi.url_scheme"],
                headers.pop("Host"),
                _wsgi_decoding_dance(environ["SCRIPT_NAME"]),
            ),
            "query_string": _wsgi_decoding_dance(environ["QUERY_STRING"]),
            "method": environ["REQUEST_METHOD"],
            "input_stream": environ["wsgi.input"],
            "content_type": headers.pop("Content-Type", None),
            "content_length": headers.pop("Content-Length", None),
            "errors_stream": environ["wsgi.errors"],
            "multithread": environ["wsgi.multithread"],
            "multiprocess": environ["wsgi.multiprocess"],
            "run_once": environ["wsgi.run_once"],
            "headers": headers,
        }
        out.update(kwargs)
        return cls(**out)

    def _add_file_from_data(
        self,
        key: str,
        value: (t.IO[bytes] | tuple[t.IO[bytes], str] | tuple[t.IO[bytes], str, str]),
    ) -> None:
        """Called in the EnvironBuilder to add files from the data dict."""
        if isinstance(value, tuple):
            self.files.add_file(key, *value)
        else:
            self.files.add_file(key, value)

    @staticmethod
    def _make_base_url(scheme: str, host: str, script_root: str) -> str:
        return urlunsplit((scheme, host, script_root, "", "")).rstrip("/") + "/"

    @property
    def base_url(self) -> str:
        """The base URL is used to extract the URL scheme, host name,
        port, and root path.
        """
        return self._make_base_url(self.url_scheme, self.host, self.script_root)

    @base_url.setter
    def base_url(self, value: str | None) -> None:
        if value is None:
            scheme = "http"
            netloc = "localhost"
            script_root = ""
        else:
            scheme, netloc, script_root, qs, anchor = urlsplit(value)
            if qs or anchor:
                raise ValueError("base url must not contain a query string or fragment")
        self.script_root = script_root.rstrip("/")
        self.host = netloc
        self.url_scheme = scheme

    @property
    def content_type(self) -> str | None:
        """The content type for the request.  Reflected from and to
        the :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        ct = self.headers.get("Content-Type")
        if ct is None and not self._input_stream:
            if self._files:
                return "multipart/form-data"
            if self._form:
                return "application/x-www-form-urlencoded"
            return None
        return ct

    @content_type.setter
    def content_type(self, value: str | None) -> None:
        if value is None:
            self.headers.pop("Content-Type", None)
        else:
            self.headers["Content-Type"] = value

    @property
    def mimetype(self) -> str | None:
        """The mimetype (content type without charset etc.)

        .. versionadded:: 0.14
        """
        ct = self.content_type
        return ct.partition(";")[0].strip() if ct else None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.content_type = get_content_type(value, "utf-8")

    @property
    def mimetype_params(self) -> t.Mapping[str, str]:
        """The mimetype parameters as dict.  For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.14
        """

        def on_update(d: CallbackDict[str, str]) -> None:
            self.headers["Content-Type"] = dump_options_header(self.mimetype, d)

        d = parse_options_header(self.headers.get("content-type", ""))[1]
        return CallbackDict(d, on_update)

    @property
    def content_length(self) -> int | None:
        """The content length as integer.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        return self.headers.get("Content-Length", type=int)

    @content_length.setter
    def content_length(self, value: int | None) -> None:
        if value is None:
            self.headers.pop("Content-Length", None)
        else:
            self.headers["Content-Length"] = str(value)

    def _get_form(self, name: str, storage: type[_TAnyMultiDict]) -> _TAnyMultiDict:
        """Common behavior for getting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param storage: Storage class used for the data.
        """
        if self.input_stream is not None:
            raise AttributeError("an input stream is defined")

        rv = getattr(self, name)

        if rv is None:
            rv = storage()
            setattr(self, name, rv)

        return rv  # type: ignore

    def _set_form(self, name: str, value: MultiDict[str, t.Any]) -> None:
        """Common behavior for setting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param value: Value to assign to the attribute.
        """
        self._input_stream = None
        setattr(self, name, value)

    @property
    def form(self) -> MultiDict[str, str]:
        """A :class:`MultiDict` of form values."""
        return self._get_form("_form", MultiDict)

    @form.setter
    def form(self, value: MultiDict[str, str]) -> None:
        self._set_form("_form", value)

    @property
    def files(self) -> FileMultiDict:
        """A :class:`FileMultiDict` of uploaded files. Use
        :meth:`~FileMultiDict.add_file` to add new files.
        """
        return self._get_form("_files", FileMultiDict)

    @files.setter
    def files(self, value: FileMultiDict) -> None:
        self._set_form("_files", value)

    @property
    def input_stream(self) -> t.IO[bytes] | None:
        """An optional input stream. This is mutually exclusive with
        setting :attr:`form` and :attr:`files`, setting it will clear
        those. Do not provide this if the method is not ``POST`` or
        another method that has a body.
        """
        return self._input_stream

    @input_stream.setter
    def input_stream(self, value: t.IO[bytes] | None) -> None:
        self._input_stream = value
        self._form = None
        self._files = None

    @property
    def query_string(self) -> str:
        """The query string.  If you set this to a string
        :attr:`args` will no longer be available.
        """
        if self._query_string is None:
            if self._args is not None:
                return _urlencode(self._args)
            return ""
        return self._query_string

    @query_string.setter
    def query_string(self, value: str | None) -> None:
        self._query_string = value
        self._args = None

    @property
    def args(self) -> MultiDict[str, str]:
        """The URL arguments as :class:`MultiDict`."""
        if self._query_string is not None:
            raise AttributeError("a query string is defined")
        if self._args is None:
            self._args = MultiDict()
        return self._args

    @args.setter
    def args(self, value: MultiDict[str, str] | None) -> None:
        self._query_string = None
        self._args = value

    @property
    def server_name(self) -> str:
        """The server name (read-only, use :attr:`host` to set)"""
        return self.host.partition(":")[0]

    @property
    def server_port(self) -> int:
        """The server port as integer (read-only, use :attr:`host` to set)"""
        _, sep, port = self.host.partition(":")

        if sep:
            try:
                return int(port)
            except ValueError:
                pass

        if self.url_scheme == "https":
            return 443
        return 80

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Closes all files.  If you put real :class:`file` objects into the
        :attr:`files` dict you can call this method to automatically close
        them all in one go.
        """
        if self.closed:
            return
        try:
            files = self.files.values()
        except AttributeError:
            files = ()
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        self.closed = True

    def get_environ(self) -> WSGIEnvironment:
        """Return the built environ.

        .. versionchanged:: 0.15
            The content type and length headers are set based on
            input stream detection. Previously this only set the WSGI
            keys.
        """
        input_stream = self.input_stream
        content_length = self.content_length

        mimetype = self.mimetype
        content_type = self.content_type

        if input_stream is not None:
            start_pos = input_stream.tell()
            input_stream.seek(0, 2)
            end_pos = input_stream.tell()
            input_stream.seek(start_pos)
            content_length = end_pos - start_pos
        elif mimetype == "multipart/form-data":
            input_stream, content_length, boundary = stream_encode_multipart(
                CombinedMultiDict([self.form, self.files])
            )
            content_type = f'{mimetype}; boundary="{boundary}"'
        elif mimetype == "application/x-www-form-urlencoded":
            form_encoded = _urlencode(self.form).encode("ascii")
            content_length = len(form_encoded)
            input_stream = BytesIO(form_encoded)
        else:
            input_stream = BytesIO()

        result: WSGIEnvironment = {}
        if self.environ_base:
            result.update(self.environ_base)

        def _path_encode(x: str) -> str:
            return _wsgi_encoding_dance(unquote(x))

        raw_uri = _wsgi_encoding_dance(self.request_uri)
        result.update(
            {
                "REQUEST_METHOD": self.method,
                "SCRIPT_NAME": _path_encode(self.script_root),
                "PATH_INFO": _path_encode(self.path),
                "QUERY_STRING": _wsgi_encoding_dance(self.query_string),
                # Non-standard, added by mod_wsgi, uWSGI
                "REQUEST_URI": raw_uri,
                # Non-standard, added by gunicorn
                "RAW_URI": raw_uri,
                "SERVER_NAME": self.server_name,
                "SERVER_PORT": str(self.server_port),
                "HTTP_HOST": self.host,
                "SERVER_PROTOCOL": self.server_protocol,
                "wsgi.version": self.wsgi_version,
                "wsgi.url_scheme": self.url_scheme,
                "wsgi.input": input_stream,
                "wsgi.errors": self.errors_stream,
                "wsgi.multithread": self.multithread,
                "wsgi.multiprocess": self.multiprocess,
                "wsgi.run_once": self.run_once,
            }
        )

        headers = self.headers.copy()
        # Don't send these as headers, they're part of the environ.
        headers.remove("Content-Type")
        headers.remove("Content-Length")

        if content_type is not None:
            result["CONTENT_TYPE"] = content_type

        if content_length is not None:
            result["CONTENT_LENGTH"] = str(content_length)

        combined_headers = defaultdict(list)

        for key, value in headers.to_wsgi_list():
            combined_headers[f"HTTP_{key.upper().replace('-', '_')}"].append(value)

        for key, values in combined_headers.items():
            result[key] = ", ".join(values)

        if self.environ_overrides:
            result.update(self.environ_overrides)

        return result

    def get_request(self, cls: type[Request] | None = None) -> Request:
        """Returns a request with the data.  If the request class is not
        specified :attr:`request_class` is used.

        :param cls: The request wrapper to use.
        """
        if cls is None:
            cls = self.request_class

        return cls(self.get_environ())


class ClientRedirectError(Exception):
    """If a redirect loop is detected when using follow_redirects=True with
    the :cls:`Client`, then this exception is raised.
    """


class Client:
    """Simulate sending requests to a WSGI application without running a WSGI or HTTP
    server.

    :param application: The WSGI application to make requests to.
    :param response_wrapper: A :class:`.Response` class to wrap response data with.
        Defaults to :class:`.TestResponse`. If it's not a subclass of ``TestResponse``,
        one will be created.
    :param use_cookies: Persist cookies from ``Set-Cookie`` response headers to the
        ``Cookie`` header in subsequent requests. Domain and path matching is supported,
        but other cookie parameters are ignored.
    :param allow_subdomain_redirects: Allow requests to follow redirects to subdomains.
        Enable this if the application handles subdomains and redirects between them.

    .. versionchanged:: 2.3
        Simplify cookie implementation, support domain and path matching.

    .. versionchanged:: 2.1
        All data is available as properties on the returned response object. The
        response cannot be returned as a tuple.

    .. versionchanged:: 2.0
        ``response_wrapper`` is always a subclass of :class:``TestResponse``.

    .. versionchanged:: 0.5
        Added the ``use_cookies`` parameter.
    """

    def __init__(
        self,
        application: WSGIApplication,
        response_wrapper: type[Response] | None = None,
        use_cookies: bool = True,
        allow_subdomain_redirects: bool = False,
    ) -> None:
        self.application = application

        if response_wrapper in {None, Response}:
            response_wrapper = TestResponse
        elif response_wrapper is not None and not issubclass(
            response_wrapper, TestResponse
        ):
            response_wrapper = type(
                "WrapperTestResponse",
                (TestResponse, response_wrapper),
                {},
            )

        self.response_wrapper = t.cast(type["TestResponse"], response_wrapper)

        if use_cookies:
            self._cookies: dict[tuple[str, str, str], Cookie] | None = {}
        else:
            self._cookies = None

        self.allow_subdomain_redirects = allow_subdomain_redirects

    def get_cookie(
        self, key: str, domain: str = "localhost", path: str = "/"
    ) -> Cookie | None:
        """Return a :class:`.Cookie` if it exists. Cookies are uniquely identified by
        ``(domain, path, key)``.

        :param key: The decoded form of the key for the cookie.
        :param domain: The domain the cookie was set for.
        :param path: The path the cookie was set for.

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        return self._cookies.get((domain, path, key))

    def set_cookie(
        self,
        key: str,
        value: str = "",
        *,
        domain: str = "localhost",
        origin_only: bool = True,
        path: str = "/",
        **kwargs: t.Any,
    ) -> None:
        """Set a cookie to be sent in subsequent requests.

        This is a convenience to skip making a test request to a route that would set
        the cookie. To test the cookie, make a test request to a route that uses the
        cookie value.

        The client uses ``domain``, ``origin_only``, and ``path`` to determine which
        cookies to send with a request. It does not use other cookie parameters that
        browsers use, since they're not applicable in tests.

        :param key: The key part of the cookie.
        :param value: The value part of the cookie.
        :param domain: Send this cookie with requests that match this domain. If
            ``origin_only`` is true, it must be an exact match, otherwise it may be a
            suffix match.
        :param origin_only: Whether the domain must be an exact match to the request.
        :param path: Send this cookie with requests that match this path either exactly
            or as a prefix.
        :param kwargs: Passed to :func:`.dump_cookie`.

        .. versionchanged:: 3.0
            The parameter ``server_name`` is removed. The first parameter is
            ``key``. Use the ``domain`` and ``origin_only`` parameters instead.

        .. versionchanged:: 2.3
            The ``origin_only`` parameter was added.

        .. versionchanged:: 2.3
            The ``domain`` parameter defaults to ``localhost``.
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        cookie = Cookie._from_response_header(
            domain, "/", dump_cookie(key, value, domain=domain, path=path, **kwargs)
        )
        cookie.origin_only = origin_only

        if cookie._should_delete:
            self._cookies.pop(cookie._storage_key, None)
        else:
            self._cookies[cookie._storage_key] = cookie

    def delete_cookie(
        self,
        key: str,
        *,
        domain: str = "localhost",
        path: str = "/",
    ) -> None:
        """Delete a cookie if it exists. Cookies are uniquely identified by
        ``(domain, path, key)``.

        :param key: The decoded form of the key for the cookie.
        :param domain: The domain the cookie was set for.
        :param path: The path the cookie was set for.

        .. versionchanged:: 3.0
            The ``server_name`` parameter is removed. The first parameter is
            ``key``. Use the ``domain`` parameter instead.

        .. versionchanged:: 3.0
            The ``secure``, ``httponly`` and ``samesite`` parameters are removed.

        .. versionchanged:: 2.3
            The ``domain`` parameter defaults to ``localhost``.
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        self._cookies.pop((domain, path, key), None)

    def _add_cookies_to_wsgi(self, environ: WSGIEnvironment) -> None:
        """If cookies are enabled, set the ``Cookie`` header in the environ to the
        cookies that are applicable to the request host and path.

        :meta private:

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            return

        url = urlsplit(get_current_url(environ))
        server_name = url.hostname or "localhost"
        value = "; ".join(
            c._to_request_header()
            for c in self._cookies.values()
            if c._matches_request(server_name, url.path)
        )

        if value:
            environ["HTTP_COOKIE"] = value
        else:
            environ.pop("HTTP_COOKIE", None)

    def _update_cookies_from_response(
        self, server_name: str, path: str, headers: list[str]
    ) -> None:
        """If cookies are enabled, update the stored cookies from any ``Set-Cookie``
        headers in the response.

        :meta private:

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            return

        for header in headers:
            cookie = Cookie._from_response_header(server_name, path, header)

            if cookie._should_delete:
                self._cookies.pop(cookie._storage_key, None)
            else:
                self._cookies[cookie._storage_key] = cookie

    def run_wsgi_app(
        self, environ: WSGIEnvironment, buffered: bool = False
    ) -> tuple[t.Iterable[bytes], str, Headers]:
        """Runs the wrapped WSGI app with the given environment.

        :meta private:
        """
        self._add_cookies_to_wsgi(environ)
        rv = run_wsgi_app(self.application, environ, buffered=buffered)
        url = urlsplit(get_current_url(environ))
        self._update_cookies_from_response(
            url.hostname or "localhost", url.path, rv[2].getlist("Set-Cookie")
        )
        return rv

    def resolve_redirect(
        self, response: TestResponse, buffered: bool = False
    ) -> TestResponse:
        """Perform a new request to the location given by the redirect
        response to the previous request.

        :meta private:
        """
        scheme, netloc, path, qs, anchor = urlsplit(response.location)
        builder = EnvironBuilder.from_environ(
            response.request.environ, path=path, query_string=qs
        )

        to_name_parts = netloc.partition(":")[0].split(".")
        from_name_parts = builder.server_name.split(".")

        if to_name_parts != [""]:
            # The new location has a host, use it for the base URL.
            builder.url_scheme = scheme
            builder.host = netloc
        else:
            # A local redirect with autocorrect_location_header=False
            # doesn't have a host, so use the request's host.
            to_name_parts = from_name_parts

        # Explain why a redirect to a different server name won't be followed.
        if to_name_parts != from_name_parts:
            if to_name_parts[-len(from_name_parts) :] == from_name_parts:
                if not self.allow_subdomain_redirects:
                    raise RuntimeError("Following subdomain redirects is not enabled.")
            else:
                raise RuntimeError("Following external redirects is not supported.")

        path_parts = path.split("/")
        root_parts = builder.script_root.split("/")

        if path_parts[: len(root_parts)] == root_parts:
            # Strip the script root from the path.
            builder.path = path[len(builder.script_root) :]
        else:
            # The new location is not under the script root, so use the
            # whole path and clear the previous root.
            builder.path = path
            builder.script_root = ""

        # Only 307 and 308 preserve all of the original request.
        if response.status_code not in {307, 308}:
            # HEAD is preserved, everything else becomes GET.
            if builder.method != "HEAD":
                builder.method = "GET"

            # Clear the body and the headers that describe it.

            if builder.input_stream is not None:
                builder.input_stream.close()
                builder.input_stream = None

            builder.content_type = None
            builder.content_length = None
            builder.headers.pop("Transfer-Encoding", None)

        return self.open(builder, buffered=buffered)

    def open(
        self,
        *args: t.Any,
        buffered: bool = False,
        follow_redirects: bool = False,
        **kwargs: t.Any,
    ) -> TestResponse:
        """Generate an environ dict from the given arguments, make a
        request to the application using it, and return the response.

        :param args: Passed to :class:`EnvironBuilder` to create the
            environ for the request. If a single arg is passed, it can
            be an existing :class:`EnvironBuilder` or an environ dict.
        :param buffered: Convert the iterator returned by the app into
            a list. If the iterator has a ``close()`` method, it is
            called automatically.
        :param follow_redirects: Make additional requests to follow HTTP
            redirects until a non-redirect status is returned.
            :attr:`TestResponse.history` lists the intermediate
            responses.

        .. versionchanged:: 2.1
            Removed the ``as_tuple`` parameter.

        .. versionchanged:: 2.0
            The request input stream is closed when calling
            ``response.close()``. Input streams for redirects are
            automatically closed.

        .. versionchanged:: 0.5
            If a dict is provided as file in the dict for the ``data``
            parameter the content type has to be called ``content_type``
            instead of ``mimetype``. This change was made for
            consistency with :class:`werkzeug.FileWrapper`.

        .. versionchanged:: 0.5
            Added the ``follow_redirects`` parameter.
        """
        request: Request | None = None

        if not kwargs and len(args) == 1:
            arg = args[0]

            if isinstance(arg, EnvironBuilder):
                request = arg.get_request()
            elif isinstance(arg, dict):
                request = EnvironBuilder.from_environ(arg).get_request()
            elif isinstance(arg, Request):
                request = arg

        if request is None:
            builder = EnvironBuilder(*args, **kwargs)

            try:
                request = builder.get_request()
            finally:
                builder.close()

        response_parts = self.run_wsgi_app(request.environ, buffered=buffered)
        response = self.response_wrapper(*response_parts, request=request)

        redirects = set()
        history: list[TestResponse] = []

        if not follow_redirects:
            return response

        while response.status_code in {
            301,
            302,
            303,
            305,
            307,
            308,
        }:
            # Exhaust intermediate response bodies to ensure middleware
            # that returns an iterator runs any cleanup code.
            if not buffered:
                response.make_sequence()
                response.close()

            new_redirect_entry = (response.location, response.status_code)

            if new_redirect_entry in redirects:
                raise ClientRedirectError(
                    f"Loop detected: A {response.status_code} redirect"
                    f" to {response.location} was already made."
                )

            redirects.add(new_redirect_entry)
            response.history = tuple(history)
            history.append(response)
            response = self.resolve_redirect(response, buffered=buffered)
        else:
            # This is the final request after redirects.
            response.history = tuple(history)
            # Close the input stream when closing the response, in case
            # the input is an open temporary file.
            response.call_on_close(request.input_stream.close)
            return response

    def get(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``GET``."""
        kw["method"] = "GET"
        return self.open(*args, **kw)

    def post(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``POST``."""
        kw["method"] = "POST"
        return self.open(*args, **kw)

    def put(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``PUT``."""
        kw["method"] = "PUT"
        return self.open(*args, **kw)

    def delete(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``DELETE``."""
        kw["method"] = "DELETE"
        return self.open(*args, **kw)

    def patch(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``PATCH``."""
        kw["method"] = "PATCH"
        return self.open(*args, **kw)

    def options(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``OPTIONS``."""
        kw["method"] = "OPTIONS"
        return self.open(*args, **kw)

    def head(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``HEAD``."""
        kw["method"] = "HEAD"
        return self.open(*args, **kw)

    def trace(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``TRACE``."""
        kw["method"] = "TRACE"
        return self.open(*args, **kw)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.application!r}>"


def create_environ(*args: t.Any, **kwargs: t.Any) -> WSGIEnvironment:
    """Create a new WSGI environ dict based on the values passed.  The first
    parameter should be the path of the request which defaults to '/'.  The
    second one can either be an absolute path (in that case the host is
    localhost:80) or a full path to the request with scheme, netloc port and
    the path to the script.

    This accepts the same arguments as the :class:`EnvironBuilder`
    constructor.

    .. versionchanged:: 0.5
       This function is now a thin wrapper over :class:`EnvironBuilder` which
       was added in 0.5.  The `headers`, `environ_base`, `environ_overrides`
       and `charset` parameters were added.
    """
    builder = EnvironBuilder(*args, **kwargs)

    try:
        return builder.get_environ()
    finally:
        builder.close()


def run_wsgi_app(
    app: WSGIApplication, environ: WSGIEnvironment, buffered: bool = False
) -> tuple[t.Iterable[bytes], str, Headers]:
    """Return a tuple in the form (app_iter, status, headers) of the
    application output.  This works best if you pass it an application that
    returns an iterator all the time.

    Sometimes applications may use the `write()` callable returned
    by the `start_response` function.  This tries to resolve such edge
    cases automatically.  But if you don't get the expected output you
    should set `buffered` to `True` which enforces buffering.

    If passed an invalid WSGI application the behavior of this function is
    undefined.  Never pass non-conforming WSGI applications to this function.

    :param app: the application to execute.
    :param buffered: set to `True` to enforce buffering.
    :return: tuple in the form ``(app_iter, status, headers)``
    """
    # Copy environ to ensure any mutations by the app (ProxyFix, for
    # example) don't affect subsequent requests (such as redirects).
    environ = _get_environ(environ).copy()
    status: str
    response: tuple[str, list[tuple[str, str]]] | None = None
    buffer: list[bytes] = []

    def start_response(status, headers, exc_info=None):  # type: ignore
        nonlocal response

        if exc_info:
            try:
                raise exc_info[1].with_traceback(exc_info[2])
            finally:
                exc_info = None

        response = (status, headers)
        return buffer.append

    app_rv = app(environ, start_response)
    close_func = getattr(app_rv, "close", None)
    app_iter: t.Iterable[bytes] = iter(app_rv)

    # when buffering we emit the close call early and convert the
    # application iterator into a regular list
    if buffered:
        try:
            app_iter = list(app_iter)
        finally:
            if close_func is not None:
                close_func()

    # otherwise we iterate the application iter until we have a response, chain
    # the already received data with the already collected data and wrap it in
    # a new `ClosingIterator` if we need to restore a `close` callable from the
    # original return value.
    else:
        for item in app_iter:
            buffer.append(item)

            if response is not None:
                break

        if buffer:
            app_iter = chain(buffer, app_iter)

        if close_func is not None and app_iter is not app_rv:
            app_iter = ClosingIterator(app_iter, close_func)

    status, headers = response  # type: ignore
    return app_iter, status, Headers(headers)


class TestResponse(Response):
    """:class:`~werkzeug.wrappers.Response` subclass that provides extra
    information about requests made with the test :class:`Client`.

    Test client requests will always return an instance of this class.
    If a custom response class is passed to the client, it is
    subclassed along with this to support test information.

    If the test request included large files, or if the application is
    serving a file, call :meth:`close` to close any open files and
    prevent Python showing a ``ResourceWarning``.

    .. versionchanged:: 2.2
        Set the ``default_mimetype`` to None to prevent a mimetype being
        assumed if missing.

    .. versionchanged:: 2.1
        Response instances cannot be treated as tuples.

    .. versionadded:: 2.0
        Test client methods always return instances of this class.
    """

    default_mimetype = None
    # Don't assume a mimetype, instead use whatever the response provides

    request: Request
    """A request object with the environ used to make the request that
    resulted in this response.
    """

    history: tuple[TestResponse, ...]
    """A list of intermediate responses. Populated when the test request
    is made with ``follow_redirects`` enabled.
    """

    # Tell Pytest to ignore this, it's not a test class.
    __test__ = False

    def __init__(
        self,
        response: t.Iterable[bytes],
        status: str,
        headers: Headers,
        request: Request,
        history: tuple[TestResponse] = (),  # type: ignore
        **kwargs: t.Any,
    ) -> None:
        super().__init__(response, status, headers, **kwargs)
        self.request = request
        self.history = history
        self._compat_tuple = response, status, headers

    @cached_property
    def text(self) -> str:
        """The response data as text. A shortcut for
        ``response.get_data(as_text=True)``.

        .. versionadded:: 2.1
        """
        return self.get_data(as_text=True)


@dataclasses.dataclass
class Cookie:
    """A cookie key, value, and parameters.

    The class itself is not a public API. Its attributes are documented for inspection
    with :meth:`.Client.get_cookie` only.

    .. versionadded:: 2.3
    """

    key: str
    """The cookie key, encoded as a client would see it."""

    value: str
    """The cookie key, encoded as a client would see it."""

    decoded_key: str
    """The cookie key, decoded as the application would set and see it."""

    decoded_value: str
    """The cookie value, decoded as the application would set and see it."""

    expires: datetime | None
    """The time at which the cookie is no longer valid."""

    max_age: int | None
    """The number of seconds from when the cookie was set at which it is
    no longer valid.
    """

    domain: str
    """The domain that the cookie was set for, or the request domain if not set."""

    origin_only: bool
    """Whether the cookie will be sent for exact domain matches only. This is ``True``
    if the ``Domain`` parameter was not present.
    """

    path: str
    """The path that the cookie was set for."""

    secure: bool | None
    """The ``Secure`` parameter."""

    http_only: bool | None
    """The ``HttpOnly`` parameter."""

    same_site: str | None
    """The ``SameSite`` parameter."""

    def _matches_request(self, server_name: str, path: str) -> bool:
        return (
            server_name == self.domain
            or (
                not self.origin_only
                and server_name.endswith(self.domain)
                and server_name[: -len(self.domain)].endswith(".")
            )
        ) and (
            path == self.path
            or (
                path.startswith(self.path)
                and path[len(self.path) - self.path.endswith("/") :].startswith("/")
            )
        )

    def _to_request_header(self) -> str:
        return f"{self.key}={self.value}"

    @classmethod
    def _from_response_header(cls, server_name: str, path: str, header: str) -> te.Self:
        header, _, parameters_str = header.partition(";")
        key, _, value = header.partition("=")
        decoded_key, decoded_value = next(parse_cookie(header).items())  # type: ignore[call-overload]
        params = {}

        for item in parameters_str.split(";"):
            k, sep, v = item.partition("=")
            params[k.strip(" \t").lower()] = v.strip(" \t") if sep else None

        return cls(
            key=key.strip(" \t"),
            value=value.strip(" \t"),
            decoded_key=decoded_key,
            decoded_value=decoded_value,
            expires=parse_date(params.get("expires")),
            max_age=int(params["max-age"] or 0) if "max-age" in params else None,
            domain=params.get("domain") or server_name,
            origin_only="domain" not in params,
            path=params.get("path") or path.rpartition("/")[0] or "/",
            secure="secure" in params,
            http_only="httponly" in params,
            same_site=params.get("samesite"),
        )

    @property
    def _storage_key(self) -> tuple[str, str, str]:
        return self.domain, self.path, self.decoded_key

    @property
    def _should_delete(self) -> bool:
        return self.max_age == 0 or (
            self.expires is not None and self.expires.timestamp() == 0
        )
<|endoftext|>"""A small application that can be used to test a WSGI server and check
it for WSGI compliance.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import typing as t
from textwrap import wrap

from markupsafe import escape

from .wrappers.request import Request
from .wrappers.response import Response

TEMPLATE = """\
<!doctype html>
<html lang=en>
<title>WSGI Information</title>
<style type="text/css">
  @import url(https://fonts.googleapis.com/css?family=Ubuntu);

  body       { font-family: 'Lucida Grande', 'Lucida Sans Unicode', 'Geneva',
               'Verdana', sans-serif; background-color: white; color: #000;
               font-size: 15px; text-align: center; }
  div.box    { text-align: left; width: 45em; margin: auto; padding: 50px 0;
               background-color: white; }
  h1, h2     { font-family: 'Ubuntu', 'Lucida Grande', 'Lucida Sans Unicode',
               'Geneva', 'Verdana', sans-serif; font-weight: normal; }
  h1         { margin: 0 0 30px 0; }
  h2         { font-size: 1.4em; margin: 1em 0 0.5em 0; }
  table      { width: 100%%; border-collapse: collapse; border: 1px solid #AFC5C9 }
  table th   { background-color: #AFC1C4; color: white; font-size: 0.72em;
               font-weight: normal; width: 18em; vertical-align: top;
               padding: 0.5em 0 0.1em 0.5em; }
  table td   { border: 1px solid #AFC5C9; padding: 0.1em 0 0.1em 0.5em; }
  code       { font-family: 'Consolas', 'Monaco', 'Bitstream Vera Sans Mono',
               monospace; font-size: 0.7em; }
  ul li      { line-height: 1.5em; }
  ul.path    { font-size: 0.7em; margin: 0 -30px; padding: 8px 30px;
               list-style: none; background: #E8EFF0; }
  ul.path li { line-height: 1.6em; }
  li.virtual { color: #999; text-decoration: underline; }
  li.exp     { background: white; }
</style>
<div class="box">
  <h1>WSGI Information</h1>
  <p>
    This page displays all available information about the WSGI server and
    the underlying Python interpreter.
  <h2 id="python-interpreter">Python Interpreter</h2>
  <table>
    <tr>
      <th>Python Version
      <td>%(python_version)s
    <tr>
      <th>Platform
      <td>%(platform)s [%(os)s]
    <tr>
      <th>API Version
      <td>%(api_version)s
    <tr>
      <th>Byteorder
      <td>%(byteorder)s
    <tr>
      <th>Werkzeug Version
      <td>%(werkzeug_version)s
  </table>
  <h2 id="wsgi-environment">WSGI Environment</h2>
  <table>%(wsgi_env)s</table>
  <h2 id="installed-eggs">Installed Eggs</h2>
  <p>
    The following python packages were installed on the system as
    Python eggs:
  <ul>%(python_eggs)s</ul>
  <h2 id="sys-path">System Path</h2>
  <p>
    The following paths are the current contents of the load path.  The
    following entries are looked up for Python packages.  Note that not
    all items in this path are folders.  Gray and underlined items are
    entries pointing to invalid resources or used by custom import hooks
    such as the zip importer.
  <p>
    Items with a bright background were expanded for display from a relative
    path.  If you encounter such paths in the output you might want to check
    your setup as relative paths are usually problematic in multithreaded
    environments.
  <ul class="path">%(sys_path)s</ul>
</div>
"""


def iter_sys_path() -> t.Iterator[tuple[str, bool, bool]]:
    if os.name == "posix":

        def strip(x: str) -> str:
            prefix = os.path.expanduser("~")
            if x.startswith(prefix):
                x = f"~{x[len(prefix) :]}"
            return x

    else:

        def strip(x: str) -> str:
            return x

    cwd = os.path.abspath(os.getcwd())
    for item in sys.path:
        path = os.path.join(cwd, item or os.path.curdir)
        yield strip(os.path.normpath(path)), not os.path.isdir(path), path != item


@Request.application
def test_app(req: Request) -> Response:
    """Simple test application that dumps the environment.  You can use
    it to check if Werkzeug is working properly:

    .. sourcecode:: pycon

        >>> from werkzeug.serving import run_simple
        >>> from werkzeug.testapp import test_app
        >>> run_simple('localhost', 3000, test_app)
         * Running on http://localhost:3000/

    The application displays important information from the WSGI environment,
    the Python interpreter and the installed libraries.
    """
    try:
        import pkg_resources
    except ImportError:
        eggs: t.Iterable[t.Any] = ()
    else:
        eggs = sorted(
            pkg_resources.working_set,
            key=lambda x: x.project_name.lower(),
        )
    python_eggs = []
    for egg in eggs:
        try:
            version = egg.version
        except (ValueError, AttributeError):
            version = "unknown"
        python_eggs.append(
            f"<li>{escape(egg.project_name)} <small>[{escape(version)}]</small>"
        )

    wsgi_env = []
    sorted_environ = sorted(req.environ.items(), key=lambda x: repr(x[0]).lower())
    for key, value in sorted_environ:
        value = "".join(wrap(str(escape(repr(value)))))
        wsgi_env.append(f"<tr><th>{escape(key)}<td><code>{value}</code>")

    sys_path = []
    for item, virtual, expanded in iter_sys_path():
        css = []
        if virtual:
            css.append("virtual")
        if expanded:
            css.append("exp")
        class_str = f' class="{" ".join(css)}"' if css else ""
        sys_path.append(f"<li{class_str}>{escape(item)}")

    context = {
        "python_version": "<br>".join(escape(sys.version).splitlines()),
        "platform": escape(sys.platform),
        "os": escape(os.name),
        "api_version": sys.api_version,
        "byteorder": sys.byteorder,
        "werkzeug_version": _get_werkzeug_version(),
        "python_eggs": "\n".join(python_eggs),
        "wsgi_env": "\n".join(wsgi_env),
        "sys_path": "\n".join(sys_path),
    }
    return Response(TEMPLATE % context, mimetype="text/html")


_werkzeug_version = ""


def _get_werkzeug_version() -> str:
    global _werkzeug_version

    if not _werkzeug_version:
        _werkzeug_version = importlib.metadata.version("werkzeug")

    return _werkzeug_version


if __name__ == "__main__":
    from .serving import run_simple

    run_simple("localhost", 5000, test_app, use_reloader=True)
<|endoftext|>from __future__ import annotations

import typing as t
from io import BytesIO
from urllib.parse import parse_qsl

from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream

# there are some platforms where SpooledTemporaryFile is not available.
# In that case we need to provide a fallback.
try:
    from tempfile import SpooledTemporaryFile
except ImportError:
    from tempfile import TemporaryFile

    SpooledTemporaryFile = None  # type: ignore

if t.TYPE_CHECKING:
    import typing as te

    from _typeshed.wsgi import WSGIEnvironment

    t_parse_result = tuple[
        t.IO[bytes], MultiDict[str, str], MultiDict[str, FileStorage]
    ]

    class TStreamFactory(te.Protocol):
        def __call__(
            self,
            total_content_length: int | None,
            content_type: str | None,
            filename: str | None,
            content_length: int | None = None,
        ) -> t.IO[bytes]: ...


F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def default_stream_factory(
    total_content_length: int | None,
    content_type: str | None,
    filename: str | None,
    content_length: int | None = None,
) -> t.IO[bytes]:
    max_size = 1024 * 500

    if SpooledTemporaryFile is not None:
        return t.cast(t.IO[bytes], SpooledTemporaryFile(max_size=max_size, mode="rb+"))
    elif total_content_length is None or total_content_length > max_size:
        return t.cast(t.IO[bytes], TemporaryFile("rb+"))

    return BytesIO()


def parse_form_data(
    environ: WSGIEnvironment,
    stream_factory: TStreamFactory | None = None,
    max_form_memory_size: int | None = None,
    max_content_length: int | None = None,
    cls: type[MultiDict[str, t.Any]] | None = None,
    silent: bool = True,
    *,
    max_form_parts: int | None = None,
) -> t_parse_result:
    """Parse the form data in the environ and return it as tuple in the form
    ``(stream, form, files)``.  You should only call this method if the
    transport method is `POST`, `PUT`, or `PATCH`.

    If the mimetype of the data transmitted is `multipart/form-data` the
    files multidict will be filled with `FileStorage` objects.  If the
    mimetype is unknown the input stream is wrapped and returned as first
    argument, else the stream is empty.

    This is a shortcut for the common usage of :class:`FormDataParser`.

    :param environ: the WSGI environment to be used for parsing.
    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param max_content_length: If the data is larger than this many bytes, raise
        :exc:`.RequestEntityTooLarge`. This is used by :meth:`parse_from_environ`
        to set up a limited stream. When using :meth:`parse`, you must get a
        limited stream with :func:`.get_input_stream`.
    :param max_form_memory_size: If a ``multipart/form-data`` text part is larger
        than this many bytes, raise :exc:`~exceptions.RequestEntityTooLarge`.
        File parts are written to disk after this size. This is an additional
        check, it does not replace using a limited stream and
        ``max_content_length``.
    :param max_form_parts: If more than this number of ``multipart/form-data``
        parts are received, raise :exc:`.RequestEntityTooLarge`. This is an
        additional check, it does not replace using a limited stream and
        ``max_content_length``.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    :return: A tuple in the form ``(stream, form, files)``.

    .. versionchanged:: 3.1.9
        ``max_form_memory_size`` is not applied to
        ``application/x-www-form-urlencoded``.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionchanged:: 2.3
        Added the ``max_form_parts`` parameter.

    .. versionadded:: 0.5.1
       Added the ``silent`` parameter.

    .. versionadded:: 0.5
       Added the ``max_form_memory_size``, ``max_content_length``, and ``cls``
       parameters.
    """
    return FormDataParser(
        stream_factory=stream_factory,
        max_form_memory_size=max_form_memory_size,
        max_content_length=max_content_length,
        max_form_parts=max_form_parts,
        silent=silent,
        cls=cls,
    ).parse_from_environ(environ)


class FormDataParser:
    """This class implements parsing of form data for Werkzeug.  By itself
    it can parse multipart and url encoded form data.  It can be subclassed
    and extended but for most mimetypes it is a better idea to use the
    untouched stream and expose it as separate attributes on a request
    object.

    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param max_content_length: If the data is larger than this many bytes, raise
        :exc:`.RequestEntityTooLarge`. This is used by :meth:`parse_from_environ`
        to set up a limited stream. When using :meth:`parse`, you must get a
        limited stream with :func:`.get_input_stream`.
    :param max_form_memory_size: If a ``multipart/form-data`` text part is larger
        than this many bytes, raise :exc:`~exceptions.RequestEntityTooLarge`.
        File parts are written to disk after this size. This is an additional
        check, it does not replace using a limited stream and
        ``max_content_length``.
    :param max_form_parts: If more than this number of ``multipart/form-data``
        parts are received, raise :exc:`.RequestEntityTooLarge`. This is an
        additional check, it does not replace using a limited stream and
        ``max_content_length``.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionchanged:: 3.0
        The ``parse_functions`` attribute and ``get_parse_func`` methods were removed.

    .. versionchanged:: 2.2.3
        Added the ``max_form_parts`` parameter.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        stream_factory: TStreamFactory | None = None,
        max_form_memory_size: int | None = None,
        max_content_length: int | None = None,
        cls: type[MultiDict[str, t.Any]] | None = None,
        silent: bool = True,
        *,
        max_form_parts: int | None = None,
    ) -> None:
        if stream_factory is None:
            stream_factory = default_stream_factory

        self.stream_factory = stream_factory
        self.max_content_length = max_content_length
        self.max_form_memory_size = max_form_memory_size
        self.max_form_parts = max_form_parts

        if cls is None:
            cls = t.cast("type[MultiDict[str, t.Any]]", MultiDict)

        self.cls = cls
        self.silent = silent

    def parse_from_environ(self, environ: WSGIEnvironment) -> t_parse_result:
        """Parses the information from the environment as form data.

        :param environ: the WSGI environment to be used for parsing.
        :return: A tuple in the form ``(stream, form, files)``.
        """
        stream = get_input_stream(environ, max_content_length=self.max_content_length)
        content_length = get_content_length(environ)
        mimetype, options = parse_options_header(environ.get("CONTENT_TYPE"))
        return self.parse(
            stream,
            content_length=content_length,
            mimetype=mimetype,
            options=options,
        )

    def parse(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: int | None,
        options: dict[str, str] | None = None,
    ) -> t_parse_result:
        """Parses the information from the given stream, mimetype,
        content length and mimetype parameters.

        :param stream: A limited input stream, from :meth:`.get_input_stream`.
        :param mimetype: The mimetype used to choose how to parse the form.
            ``multipart/form-data`` and ``application/x-www-form-urlencoded``
            are supported.
        :param content_length: the content length of the incoming data
        :param options: optional mimetype parameters (used for
                        the multipart boundary for instance)
        :return: A tuple in the form ``(stream, form, files)``.

        .. versionchanged:: 3.0
            The invalid ``application/x-url-encoded`` content type is not
            treated as ``application/x-www-form-urlencoded``.
        """
        if mimetype == "multipart/form-data":
            parse_func = self._parse_multipart
        elif mimetype == "application/x-www-form-urlencoded":
            parse_func = self._parse_urlencoded
        else:
            return stream, self.cls(), self.cls()

        if options is None:
            options = {}

        try:
            return parse_func(stream, mimetype, content_length, options)
        except ValueError:
            if not self.silent:
                raise

        return stream, self.cls(), self.cls()

    def _parse_multipart(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: int | None,
        options: dict[str, str],
    ) -> t_parse_result:
        parser = MultiPartParser(
            stream_factory=self.stream_factory,
            max_form_memory_size=self.max_form_memory_size,
            max_form_parts=self.max_form_parts,
            cls=self.cls,
        )
        boundary = options.get("boundary", "").encode("ascii")

        if not boundary:
            raise ValueError("Missing boundary")

        form, files = parser.parse(stream, boundary, content_length)
        return stream, form, files

    def _parse_urlencoded(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: int | None,
        options: dict[str, str],
    ) -> t_parse_result:
        # The stream must already be limited to max_content_length.
        # max_form_memory_size can't apply, since the entire stream is read at
        # once instead of incrementally parsed.
        # max_form_parts could be applied here, but doesn't impose a useful
        # limit given the previous points and because the parser is much simpler
        # and faster already.
        items = parse_qsl(
            stream.read().decode(),
            keep_blank_values=True,
            errors="werkzeug.url_quote",
        )
        return stream, self.cls(items), self.cls()


class MultiPartParser:
    def __init__(
        self,
        stream_factory: TStreamFactory | None = None,
        max_form_memory_size: int | None = None,
        cls: type[MultiDict[str, t.Any]] | None = None,
        buffer_size: int = 64 * 1024,
        max_form_parts: int | None = None,
    ) -> None:
        self.max_form_memory_size = max_form_memory_size
        self.max_form_parts = max_form_parts

        if stream_factory is None:
            stream_factory = default_stream_factory

        self.stream_factory = stream_factory

        if cls is None:
            cls = t.cast("type[MultiDict[str, t.Any]]", MultiDict)

        self.cls = cls
        self.buffer_size = buffer_size

    def fail(self, message: str) -> te.NoReturn:
        raise ValueError(message)

    def get_part_charset(self, headers: Headers) -> str:
        # Figure out input charset for current part
        content_type = headers.get("content-type")

        if content_type:
            parameters = parse_options_header(content_type)[1]
            ct_charset = parameters.get("charset", "").lower()

            # A safe list of encodings. Modern clients should only send ASCII or UTF-8.
            # This list will not be extended further.
            if ct_charset in {"ascii", "us-ascii", "utf-8", "iso-8859-1"}:
                return ct_charset

        return "utf-8"

    def start_file_streaming(
        self, event: File, total_content_length: int | None
    ) -> t.IO[bytes]:
        content_type = event.headers.get("content-type")

        try:
            content_length = _plain_int(event.headers["content-length"])
        except (KeyError, ValueError):
            content_length = 0

        container = self.stream_factory(
            total_content_length=total_content_length,
            filename=event.filename,
            content_type=content_type,
            content_length=content_length,
        )
        return container

    def parse(
        self, stream: t.IO[bytes], boundary: bytes, content_length: int | None
    ) -> tuple[MultiDict[str, str], MultiDict[str, FileStorage]]:
        current_part: Field | File
        field_size: int | None = None
        container: t.IO[bytes] | list[bytes]
        _write: t.Callable[[bytes], t.Any]

        parser = MultipartDecoder(
            boundary,
            max_form_memory_size=self.max_form_memory_size,
            max_parts=self.max_form_parts,
        )

        fields = []
        files = []

        for data in _chunk_iter(stream.read, self.buffer_size):
            parser.receive_data(data)
            event = parser.next_event()
            while not isinstance(event, (Epilogue, NeedData)):
                if isinstance(event, Field):
                    current_part = event
                    field_size = 0
                    container = []
                    _write = container.append
                elif isinstance(event, File):
                    current_part = event
                    field_size = None
                    container = self.start_file_streaming(event, content_length)
                    _write = container.write
                elif isinstance(event, Data):
                    if self.max_form_memory_size is not None and field_size is not None:
                        # Ensure that accumulated data events do not exceed limit.
                        # Also checked within single event in MultipartDecoder.
                        field_size += len(event.data)

                        if field_size > self.max_form_memory_size:
                            raise RequestEntityTooLarge()

                    _write(event.data)
                    if not event.more_data:
                        if isinstance(current_part, Field):
                            value = b"".join(container).decode(
                                self.get_part_charset(current_part.headers), "replace"
                            )
                            fields.append((current_part.name, value))
                        else:
                            container = t.cast(t.IO[bytes], container)
                            container.seek(0)
                            files.append(
                                (
                                    current_part.name,
                                    FileStorage(
                                        container,
                                        current_part.filename,
                                        current_part.name,
                                        headers=current_part.headers,
                                    ),
                                )
                            )

                event = parser.next_event()

        return self.cls(fields), self.cls(files)


def _chunk_iter(read: t.Callable[[int], bytes], size: int) -> t.Iterator[bytes | None]:
    """Read data in chunks for multipart/form-data parsing. Stop if no data is read.
    Yield ``None`` at the end to signal end of parsing.
    """
    while True:
        data = read(size)

        if not data:
            break

        yield data

    yield None
<|endoftext|>from __future__ import annotations

import io
import typing as t
from functools import partial
from functools import update_wrapper

from .exceptions import ClientDisconnected
from .exceptions import RequestEntityTooLarge
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def responder(f: t.Callable[..., WSGIApplication]) -> WSGIApplication:
    """Marks a function as responder.  Decorate a function with it and it
    will automatically call the return value as WSGI application.

    Example::

        @responder
        def application(environ, start_response):
            return Response('Hello World!')
    """
    return update_wrapper(lambda *a: f(*a)(*a[-2:]), f)


def get_current_url(
    environ: WSGIEnvironment,
    root_only: bool = False,
    strip_querystring: bool = False,
    host_only: bool = False,
    trusted_hosts: t.Collection[str] | None = None,
) -> str:
    """Recreate the URL for a request from the parts in a WSGI
    environment.

    The URL is an IRI, not a URI, so it may contain Unicode characters.
    Use :func:`~werkzeug.urls.iri_to_uri` to convert it to ASCII.

    :param environ: The WSGI environment to get the URL parts from.
    :param root_only: Only build the root path, don't include the
        remaining path or query string.
    :param strip_querystring: Don't include the query string.
    :param host_only: Only build the scheme and host.
    :param trusted_hosts: A list of trusted host names to validate the
        host against.
    """
    parts = {
        "scheme": environ["wsgi.url_scheme"],
        "host": get_host(environ, trusted_hosts),
    }

    if not host_only:
        parts["root_path"] = environ.get("SCRIPT_NAME", "")

        if not root_only:
            parts["path"] = environ.get("PATH_INFO", "")

            if not strip_querystring:
                parts["query_string"] = environ.get("QUERY_STRING", "").encode("latin1")

    return _sansio_utils.get_current_url(**parts)


def _get_server(
    environ: WSGIEnvironment,
) -> tuple[str, int | None] | None:
    name = environ.get("SERVER_NAME")

    if name is None:
        return None

    try:
        port: int | None = int(environ.get("SERVER_PORT", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # unix socket
        port = None

    return name, port


def get_host(
    environ: WSGIEnvironment, trusted_hosts: t.Collection[str] | None = None
) -> str:
    """Get and validate a request's ``host:port`` based on the values in the
    given WSGI environ.

    The ``Host`` header sent by the client is preferred. Otherwise, the server's
    configured address is used. If the server address is a Unix socket, it is
    ignored. The port is omitted if it matches the standard HTTP or HTTPS ports.

    The value is passed through :func:`host_is_trusted`. The host must be made
    up of valid characters, but this does not check validity beyond that. If a
    list of trusted domains is given, the domain must match one.

    :param environ: The WSGI environ.
    :param trusted_hosts: A list of trusted domains to match. These should
        already be IDNA encoded, but will be encoded if needed. The port is
        ignored for this check. If a name starts with a dot it will match as a
        suffix, accepting all subdomains. If empty or ``None``, all domains are
        allowed.

    :return: Host, with port if necessary.
    :raise .SecurityError: If the host is not trusted.

    .. versionchanged:: 3.2
        The characters of the host value are validated. The empty string is no
        longer allowed if no header value is available.

    .. versionchanged:: 3.2
        When using the server address, Unix sockets are ignored.

    .. versionchanged:: 3.1.3
        If ``SERVER_NAME`` is IPv6, it is wrapped in ``[]``.
    """
    return _sansio_utils.get_host(
        environ["wsgi.url_scheme"],
        environ.get("HTTP_HOST"),
        _get_server(environ),
        trusted_hosts,
    )


def get_content_length(environ: WSGIEnvironment) -> int | None:
    """Return the ``Content-Length`` header value as an int. If the header is not given
    or the ``Transfer-Encoding`` header is ``chunked``, ``None`` is returned to indicate
    a streaming request. If the value is not an integer, or negative, 0 is returned.

    :param environ: The WSGI environ to get the content length from.

    .. versionadded:: 0.9
    """
    return _sansio_utils.get_content_length(
        http_content_length=environ.get("CONTENT_LENGTH"),
        http_transfer_encoding=environ.get("HTTP_TRANSFER_ENCODING"),
    )


def get_input_stream(
    environ: WSGIEnvironment,
    safe_fallback: bool = True,
    max_content_length: int | None = None,
) -> t.IO[bytes]:
    """Return the WSGI input stream, wrapped so that it may be read safely
    without going past the ``Content-Length`` header value or
    ``max_content_length``. This must be used to pass a safe stream to other
    functions, which are not responsible for enforcing this limit.

    If ``Content-Length`` exceeds ``max_content_length``, a
    :exc:`RequestEntityTooLarge`` ``413 Content Too Large`` error is raised.

    If the WSGI server sets ``environ["wsgi.input_terminated"]``, it indicates that the
    server handles terminating the stream, so it is safe to read directly. For example,
    a server that knows how to handle chunked requests safely would set this.

    If ``max_content_length`` is set, it can be enforced on streams if
    ``wsgi.input_terminated`` is set. Otherwise, an empty stream is returned unless the
    user explicitly disables this safe fallback.

    If the limit is reached before the underlying stream is exhausted (such as a file
    that is too large, or an infinite stream), the remaining contents of the stream
    cannot be read safely. Depending on how the server handles this, clients may show a
    "connection reset" failure instead of seeing the 413 response.

    :param environ: The WSGI environ containing the stream.
    :param safe_fallback: Return an empty stream when ``Content-Length`` is not set.
        Disabling this allows infinite streams, which can be a denial-of-service risk.
    :param max_content_length: The maximum length that content-length or streaming
        requests may not exceed.

    .. versionchanged:: 2.3.2
        ``max_content_length`` is only applied to streaming requests if the server sets
        ``wsgi.input_terminated``.

    .. versionchanged:: 2.3
        Check ``max_content_length`` and raise an error if it is exceeded.

    .. versionadded:: 0.9
    """
    stream = t.cast(t.IO[bytes], environ["wsgi.input"])
    content_length = get_content_length(environ)

    if content_length is not None and max_content_length is not None:
        if content_length > max_content_length:
            raise RequestEntityTooLarge()

    # A WSGI server can set this to indicate that it terminates the input stream. In
    # that case the stream is safe without wrapping, or can enforce a max length.
    if "wsgi.input_terminated" in environ:
        if max_content_length is not None:
            # If this is moved above, it can cause the stream to hang if a read attempt
            # is made when the client sends no data. For example, the development server
            # does not handle buffering except for chunked encoding.
            return t.cast(
                t.IO[bytes], LimitedStream(stream, max_content_length, is_max=True)
            )

        return stream

    # No limit given, return an empty stream unless the user explicitly allows the
    # potentially infinite stream. An infinite stream is dangerous if it's not expected,
    # as it can tie up a worker indefinitely.
    if content_length is None:
        return io.BytesIO() if safe_fallback else stream

    return t.cast(t.IO[bytes], LimitedStream(stream, content_length))


def get_path_info(environ: WSGIEnvironment) -> str:
    """Return ``PATH_INFO`` from  the WSGI environment.

    :param environ: WSGI environment to get the path from.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionadded:: 0.9
    """
    path: bytes = environ.get("PATH_INFO", "").encode("latin1")
    return path.decode(errors="replace")


class ClosingIterator:
    """The WSGI specification requires that all middlewares and gateways
    respect the `close` callback of the iterable returned by the application.
    Because it is useful to add another close action to a returned iterable
    and adding a custom iterable is a boring task this class can be used for
    that::

        return ClosingIterator(app(environ, start_response), [cleanup_session,
                                                              cleanup_locals])

    If there is just one close function it can be passed instead of the list.

    A closing iterator is not needed if the application uses response objects
    and finishes the processing if the response is started::

        try:
            return response(environ, start_response)
        finally:
            cleanup_session()
            cleanup_locals()
    """

    def __init__(
        self,
        iterable: t.Iterable[bytes],
        callbacks: None
        | (t.Callable[[], None] | t.Iterable[t.Callable[[], None]]) = None,
    ) -> None:
        iterator = iter(iterable)
        self._next = t.cast(t.Callable[[], bytes], partial(next, iterator))
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        iterable_close = getattr(iterable, "close", None)
        if iterable_close:
            callbacks.insert(0, iterable_close)
        self._callbacks = callbacks

    def __iter__(self) -> ClosingIterator:
        return self

    def __next__(self) -> bytes:
        return self._next()

    def close(self) -> None:
        for callback in self._callbacks:
            callback()


def wrap_file(
    environ: WSGIEnvironment, file: t.IO[bytes], buffer_size: int = 8192
) -> t.Iterable[bytes]:
    """Wraps a file.  This uses the WSGI server's file wrapper if available
    or otherwise the generic :class:`FileWrapper`.

    .. versionadded:: 0.5

    If the file wrapper from the WSGI server is used it's important to not
    iterate over it from inside the application but to pass it through
    unchanged.  If you want to pass out a file wrapper inside a response
    object you have to set :attr:`Response.direct_passthrough` to `True`.

    More information about file wrappers are available in :pep:`333`.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """
    return environ.get("wsgi.file_wrapper", FileWrapper)(  # type: ignore
        file, buffer_size
    )


class FileWrapper:
    """This class can be used to convert a :class:`file`-like object into
    an iterable.  It yields `buffer_size` blocks until the file is fully
    read.

    You should not use this class directly but rather use the
    :func:`wrap_file` function that uses the WSGI server's file wrapper
    support if it's available.

    .. versionadded:: 0.5

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """

    def __init__(self, file: t.IO[bytes], buffer_size: int = 8192) -> None:
        self.file = file
        self.buffer_size = buffer_size

    def close(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    def seekable(self) -> bool:
        if hasattr(self.file, "seekable"):
            return self.file.seekable()
        if hasattr(self.file, "seek"):
            return True
        return False

    def seek(self, *args: t.Any) -> None:
        if hasattr(self.file, "seek"):
            self.file.seek(*args)

    def tell(self) -> int | None:
        if hasattr(self.file, "tell"):
            return self.file.tell()
        return None

    def __iter__(self) -> FileWrapper:
        return self

    def __next__(self) -> bytes:
        data = self.file.read(self.buffer_size)
        if data:
            return data
        raise StopIteration()


class _RangeWrapper:
    # private for now, but should we make it public in the future ?

    """This class can be used to convert an iterable object into
    an iterable that will only yield a piece of the underlying content.
    It yields blocks until the underlying stream range is fully read.
    The yielded blocks will have a size that can't exceed the original
    iterator defined block size, but that can be smaller.

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param iterable: an iterable object with a :meth:`__next__` method.
    :param start_byte: byte from which read will start.
    :param byte_range: how many bytes to read.
    """

    def __init__(
        self,
        iterable: t.Iterable[bytes] | t.IO[bytes],
        start_byte: int = 0,
        byte_range: int | None = None,
    ):
        self.iterable = iter(iterable)
        self.byte_range = byte_range
        self.start_byte = start_byte
        self.end_byte = None

        if byte_range is not None:
            self.end_byte = start_byte + byte_range

        self.read_length = 0
        self.seekable = hasattr(iterable, "seekable") and iterable.seekable()
        self.end_reached = False

    def __iter__(self) -> _RangeWrapper:
        return self

    def _next_chunk(self) -> bytes:
        try:
            chunk = next(self.iterable)
            self.read_length += len(chunk)
            return chunk
        except StopIteration:
            self.end_reached = True
            raise

    def _first_iteration(self) -> tuple[bytes | None, int]:
        chunk = None
        if self.seekable:
            self.iterable.seek(self.start_byte)  # type: ignore
            self.read_length = self.iterable.tell()  # type: ignore
            contextual_read_length = self.read_length
        else:
            while self.read_length <= self.start_byte:
                chunk = self._next_chunk()
            if chunk is not None:
                chunk = chunk[self.start_byte - self.read_length :]
            contextual_read_length = self.start_byte
        return chunk, contextual_read_length

    def _next(self) -> bytes:
        if self.end_reached:
            raise StopIteration()
        chunk = None
        contextual_read_length = self.read_length
        if self.read_length == 0:
            chunk, contextual_read_length = self._first_iteration()
        if chunk is None:
            chunk = self._next_chunk()
        if self.end_byte is not None and self.read_length >= self.end_byte:
            self.end_reached = True
            return chunk[: self.end_byte - contextual_read_length]
        return chunk

    def __next__(self) -> bytes:
        chunk = self._next()
        if chunk:
            return chunk
        self.end_reached = True
        raise StopIteration()

    def close(self) -> None:
        if hasattr(self.iterable, "close"):
            self.iterable.close()


class LimitedStream(io.RawIOBase):
    """Wrap a stream so that it doesn't read more than a given limit. This is used to
    limit ``wsgi.input`` to the ``Content-Length`` header value or
    :attr:`.Request.max_content_length`.

    When attempting to read after the limit has been reached, :meth:`on_exhausted` is
    called. When the limit is a maximum, this raises :exc:`.RequestEntityTooLarge`.

    If reading from the stream returns zero bytes or raises an error,
    :meth:`on_disconnect` is called, which raises :exc:`.ClientDisconnected`. When the
    limit is a maximum and zero bytes were read, no error is raised, since it may be the
    end of the stream.

    If the limit is reached before the underlying stream is exhausted (such as a file
    that is too large, or an infinite stream), the remaining contents of the stream
    cannot be read safely. Depending on how the server handles this, clients may show a
    "connection reset" failure instead of seeing the 413 response.

    :param stream: The stream to read from. Must be a readable binary IO object.
    :param limit: The limit in bytes to not read past. Should be either the
        ``Content-Length`` header value or ``request.max_content_length``.
    :param is_max: Whether the given ``limit`` is ``request.max_content_length`` instead
        of the ``Content-Length`` header value. This changes how exhausted and
        disconnect events are handled.

    .. versionchanged:: 2.3
        Handle ``max_content_length`` differently than ``Content-Length``.

    .. versionchanged:: 2.3
        Implements ``io.RawIOBase`` rather than ``io.IOBase``.
    """

    def __init__(self, stream: t.IO[bytes], limit: int, is_max: bool = False) -> None:
        self._stream = stream
        self._pos = 0
        self.limit = limit
        self._limit_is_max = is_max

    @property
    def is_exhausted(self) -> bool:
        """Whether the current stream position has reached the limit."""
        return self._pos >= self.limit

    def on_exhausted(self) -> None:
        """Called when attempting to read after the limit has been reached.

        The default behavior is to do nothing, unless the limit is a maximum, in which
        case it raises :exc:`.RequestEntityTooLarge`.

        .. versionchanged:: 2.3
            Raises ``RequestEntityTooLarge`` if the limit is a maximum.

        .. versionchanged:: 2.3
            Any return value is ignored.
        """
        if self._limit_is_max:
            raise RequestEntityTooLarge()

    def on_disconnect(self, error: Exception | None = None) -> None:
        """Called when an attempted read receives zero bytes before the limit was
        reached. This indicates that the client disconnected before sending the full
        request body.

        The default behavior is to raise :exc:`.ClientDisconnected`, unless the limit is
        a maximum and no error was raised.

        .. versionchanged:: 2.3
            Added the ``error`` parameter. Do nothing if the limit is a maximum and no
            error was raised.

        .. versionchanged:: 2.3
            Any return value is ignored.
        """
        if not self._limit_is_max or error is not None:
            raise ClientDisconnected()

        # If the limit is a maximum, then we may have read zero bytes because the
        # streaming body is complete. There's no way to distinguish that from the
        # client disconnecting early.

    def exhaust(self) -> bytes:
        """Exhaust the stream by reading until the limit is reached or the client
        disconnects, returning the remaining data.

        .. versionchanged:: 2.3
            Return the remaining data.

        .. versionchanged:: 2.2.3
            Handle case where wrapped stream returns fewer bytes than requested.
        """
        if not self.is_exhausted:
            return self.readall()

        return b""

    def readinto(self, b: bytearray) -> int | None:  # type: ignore[override]
        size = len(b)
        remaining = self.limit - self._pos

        if remaining <= 0:
            self.on_exhausted()
            return 0

        if hasattr(self._stream, "readinto"):
            # Use stream.readinto if it's available.
            if size <= remaining:
                # The size fits in the remaining limit, use the buffer directly.
                try:
                    out_size: int | None = self._stream.readinto(b)
                except (OSError, ValueError) as e:
                    self.on_disconnect(error=e)
                    return 0
            else:
                # Use a temp buffer with the remaining limit as the size.
                temp_b = bytearray(remaining)

                try:
                    out_size = self._stream.readinto(temp_b)
                except (OSError, ValueError) as e:
                    self.on_disconnect(error=e)
                    return 0

                if out_size:
                    # Need to slice both sides. Math here is complicated.
                    # size=10, remaining=5, out_size=3, remaining > out_size
                    # Without slicing temp_b (b[:s] = tb), would try to
                    # put 5 bytes into 3 bytes, causing b to resize to 12.
                    b[:out_size] = temp_b[:out_size]
        else:
            # WSGI requires that stream.read is available.
            try:
                data = self._stream.read(min(size, remaining))
            except (OSError, ValueError) as e:
                self.on_disconnect(error=e)
                return 0

            out_size = len(data)
            b[:out_size] = data

        if not out_size:
            # Read zero bytes from the stream.
            self.on_disconnect()
            return 0

        self._pos += out_size
        return out_size

    def readall(self) -> bytes:
        if self.is_exhausted:
            self.on_exhausted()
            return b""

        out = bytearray()

        # The parent implementation uses "while True", which results in an extra read.
        while not self.is_exhausted:
            data = self.read(1024 * 64)

            # Stream may return empty before a max limit is reached.
            if not data:
                break

            out.extend(data)

        return bytes(out)

    def tell(self) -> int:
        """Return the current stream position.

        .. versionadded:: 0.9
        """
        return self._pos

    def readable(self) -> bool:
        return True
<|endoftext|>from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath

from ._internal import _log

# The various system prefixes where imports are found. Base values are
# different when running in a virtualenv. All reloaders will ignore the
# base paths (usually the system installation). The stat reloader won't
# scan the virtualenv paths, it will only include modules that are
# already imported.
_ignore_always = tuple({sys.base_prefix, sys.base_exec_prefix})
prefix = {*_ignore_always, sys.prefix, sys.exec_prefix}

if hasattr(sys, "real_prefix"):
    # virtualenv < 20
    prefix.add(sys.real_prefix)

_stat_ignore_scan = tuple(prefix)
del prefix
# Ignore __pycache__ since a change there will always have a change to
# the source file (or initial pyc file) as well. Ignore common version control
# internals. Ignore common tool caches.
_ignore_common_dirs = {
    "__pycache__",
    ".git",
    ".hg",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
}


def _iter_module_paths() -> t.Iterator[str]:
    """Find the filesystem paths associated with imported modules."""
    # List is in case the value is modified by the app while updating.
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)

        if name is None or name.startswith(_ignore_always):
            continue

        while not os.path.isfile(name):
            # Zip file, find the base file without the module path.
            old = name
            name = os.path.dirname(name)

            if name == old:  # skip if it was all directories somehow
                break
        else:
            yield name


def _remove_by_pattern(paths: set[str], exclude_patterns: set[str]) -> None:
    for pattern in exclude_patterns:
        paths.difference_update(fnmatch.filter(paths, pattern))


def _find_stat_paths(
    extra_files: set[str], exclude_patterns: set[str]
) -> t.Iterable[str]:
    """Find paths for the stat reloader to watch. Returns imported
    module files, Python files under non-system paths. Extra files and
    Python files under extra directories can also be scanned.

    System paths have to be excluded for efficiency. Non-system paths,
    such as a project root or ``sys.path.insert``, should be the paths
    of interest to the user anyway.
    """
    paths = set()

    for path in chain(list(sys.path), extra_files):
        path = os.path.abspath(path)

        if os.path.isfile(path):
            # zip file on sys.path, or extra file
            paths.add(path)
            continue

        parent_has_py = {os.path.dirname(path): True}

        for root, dirs, files in os.walk(path):
            if (
                root.startswith(_stat_ignore_scan)
                or os.path.basename(root) in _ignore_common_dirs
            ):
                dirs.clear()
                continue

            has_py = False

            for name in files:
                if name.endswith((".py", ".pyc")):
                    has_py = True
                    paths.add(os.path.join(root, name))

            # Optimization: stop scanning a directory if neither it nor
            # its parent contained Python files.
            if not (has_py or parent_has_py[os.path.dirname(root)]):
                dirs.clear()
                continue

            parent_has_py[root] = has_py

    paths.update(_iter_module_paths())
    _remove_by_pattern(paths, exclude_patterns)
    return paths


def _find_watchdog_paths(
    extra_files: set[str], exclude_patterns: set[str]
) -> t.Iterable[str]:
    """Find paths for the stat reloader to watch. Looks at the same
    sources as the stat reloader, but watches everything under
    directories instead of individual files.
    """
    dirs = set()

    for name in chain(list(sys.path), extra_files):
        name = os.path.abspath(name)

        if os.path.isfile(name):
            name = os.path.dirname(name)

        dirs.add(name)

    for name in _iter_module_paths():
        dirs.add(os.path.dirname(name))

    _remove_by_pattern(dirs, exclude_patterns)
    return _find_common_roots(dirs)


def _find_common_roots(paths: t.Iterable[str]) -> t.Iterable[str]:
    root: dict[str, dict[str, t.Any]] = {}

    for chunks in sorted((PurePath(x).parts for x in paths), key=len, reverse=True):
        node = root

        for chunk in chunks:
            node = node.setdefault(chunk, {})

        node.clear()

    rv = set()

    def _walk(node: t.Mapping[str, dict[str, t.Any]], path: tuple[str, ...]) -> None:
        for prefix, child in node.items():
            _walk(child, path + (prefix,))

        # If there are no more nodes, and a path has been accumulated, add it.
        # Path may be empty if the "" entry is in sys.path.
        if not node and path:
            rv.add(os.path.join(*path))

    _walk(root, ())
    return rv


def _get_args_for_reloading() -> list[str]:
    """Determine how the script was executed, and return the args needed
    to execute it again in a new process.
    """
    if sys.version_info >= (3, 10):
        # sys.orig_argv, added in Python 3.10, contains the exact args used to invoke
        # Python. Still replace argv[0] with sys.executable for accuracy.
        return [sys.executable, *sys.orig_argv[1:]]

    rv = [sys.executable]
    py_script = sys.argv[0]
    args = sys.argv[1:]
    # Need to look at main module to determine how it was executed.
    __main__ = sys.modules["__main__"]

    # The value of __package__ indicates how Python was called. It may
    # not exist if a setuptools script is installed as an egg. It may be
    # set incorrectly for entry points created with pip on Windows.
    if getattr(__main__, "__package__", None) is None or (
        os.name == "nt"
        and __main__.__package__ == ""
        and not os.path.exists(py_script)
        and os.path.exists(f"{py_script}.exe")
    ):
        # Executed a file, like "python app.py".
        py_script = os.path.abspath(py_script)

        if os.name == "nt":
            # Windows entry points have ".exe" extension and should be
            # called directly.
            if not os.path.exists(py_script) and os.path.exists(f"{py_script}.exe"):
                py_script += ".exe"

            if (
                os.path.splitext(sys.executable)[1] == ".exe"
                and os.path.splitext(py_script)[1] == ".exe"
            ):
                rv.pop(0)

        rv.append(py_script)
    else:
        # Executed a module, like "python -m werkzeug.serving".
        if os.path.isfile(py_script):
            # Rewritten by Python from "-m script" to "/path/to/script.py".
            py_module = t.cast(str, __main__.__package__)
            name = os.path.splitext(os.path.basename(py_script))[0]

            if name != "__main__":
                py_module += f".{name}"
        else:
            # Incorrectly rewritten by pydevd debugger from "-m script" to "script".
            py_module = py_script

        rv.extend(("-m", py_module.lstrip(".")))

    rv.extend(args)
    return rv


class ReloaderLoop:
    name = ""

    def __init__(
        self,
        extra_files: t.Iterable[str] | None = None,
        exclude_patterns: t.Iterable[str] | None = None,
        interval: int | float = 1,
    ) -> None:
        self.extra_files: set[str] = {os.path.abspath(x) for x in extra_files or ()}
        self.exclude_patterns: set[str] = set(exclude_patterns or ())
        self.interval = interval

    def __enter__(self) -> ReloaderLoop:
        """Do any setup, then run one step of the watch to populate the
        initial filesystem state.
        """
        self.run_step()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        """Clean up any resources associated with the reloader."""
        pass

    def run(self) -> None:
        """Continually run the watch step, sleeping for the configured
        interval after each step.
        """
        while True:
            self.run_step()
            time.sleep(self.interval)

    def run_step(self) -> None:
        """Run one step for watching the filesystem. Called once to set
        up initial state, then repeatedly to update it.
        """
        pass

    def restart_with_reloader(self) -> int:
        """Spawn a new Python interpreter with the same arguments as the
        current one, but running the reloader thread.
        """
        while True:
            _log("info", f" * Restarting with {self.name}")
            args = _get_args_for_reloading()
            new_environ = os.environ.copy()
            new_environ["WERKZEUG_RUN_MAIN"] = "true"
            exit_code = subprocess.call(args, env=new_environ, close_fds=False)

            if exit_code != 3:
                return exit_code

    def trigger_reload(self, filename: str) -> None:
        self.log_reload(filename)
        sys.exit(3)

    def log_reload(self, filename: str | bytes) -> None:
        filename = os.path.abspath(filename)
        _log("info", f" * Detected change in {filename!r}, reloading")


class StatReloaderLoop(ReloaderLoop):
    name = "stat"

    def __enter__(self) -> ReloaderLoop:
        self.mtimes: dict[str, float] = {}
        return super().__enter__()

    def run_step(self) -> None:
        for name in _find_stat_paths(self.extra_files, self.exclude_patterns):
            try:
                mtime = os.stat(name).st_mtime
            except OSError:
                continue

            old_time = self.mtimes.get(name)

            if old_time is None:
                self.mtimes[name] = mtime
                continue

            if mtime > old_time:
                self.trigger_reload(name)


class WatchdogReloaderLoop(ReloaderLoop):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        from watchdog.events import EVENT_TYPE_CLOSED
        from watchdog.events import EVENT_TYPE_CREATED
        from watchdog.events import EVENT_TYPE_DELETED
        from watchdog.events import EVENT_TYPE_MODIFIED
        from watchdog.events import EVENT_TYPE_MOVED
        from watchdog.events import FileModifiedEvent
        from watchdog.events import PatternMatchingEventHandler
        from watchdog.observers import Observer

        super().__init__(*args, **kwargs)
        trigger_reload = self.trigger_reload

        class EventHandler(PatternMatchingEventHandler):
            def on_any_event(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
                if event.event_type not in {
                    EVENT_TYPE_CLOSED,
                    EVENT_TYPE_CREATED,
                    EVENT_TYPE_DELETED,
                    EVENT_TYPE_MODIFIED,
                    EVENT_TYPE_MOVED,
                }:
                    # skip events that don't involve changes to the file
                    return

                trigger_reload(event.src_path)

        reloader_name = Observer.__name__.lower()  # type: ignore[attr-defined]

        if reloader_name.endswith("observer"):
            reloader_name = reloader_name[:-8]

        self.name = f"watchdog ({reloader_name})"
        self.observer = Observer()
        extra_patterns = (p for p in self.extra_files if not os.path.isdir(p))
        self.event_handler = EventHandler(
            patterns=["*.py", "*.pyc", "*.zip", *extra_patterns],
            ignore_patterns=[
                *[f"*/{d}/*" for d in _ignore_common_dirs],
                *self.exclude_patterns,
            ],
        )
        self.should_reload = threading.Event()

    def trigger_reload(self, filename: str | bytes) -> None:
        # This is called inside an event handler, which means throwing
        # SystemExit has no effect.
        # https://github.com/gorakhargosh/watchdog/issues/294
        self.should_reload.set()
        self.log_reload(filename)

    def __enter__(self) -> ReloaderLoop:
        self.watches: dict[str, t.Any] = {}
        self.observer.start()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.observer.stop()
        self.observer.join()

    def run(self) -> None:
        while not self.should_reload.wait(timeout=self.interval):
            self.run_step()

        sys.exit(3)

    def run_step(self) -> None:
        to_delete = set(self.watches)

        for path in _find_watchdog_paths(self.extra_files, self.exclude_patterns):
            if path not in self.watches:
                try:
                    self.watches[path] = self.observer.schedule(
                        self.event_handler, path, recursive=True
                    )
                except OSError:
                    # Clear this path from list of watches. We don't want
                    # the same error message showing again in the next
                    # iteration.
                    self.watches[path] = None

            to_delete.discard(path)

        for path in to_delete:
            watch = self.watches.pop(path, None)

            if watch is not None:
                self.observer.unschedule(watch)


reloader_loops: dict[str, type[ReloaderLoop]] = {
    "stat": StatReloaderLoop,
    "watchdog": WatchdogReloaderLoop,
}

try:
    __import__("watchdog.observers")
except ImportError:
    reloader_loops["auto"] = reloader_loops["stat"]
else:
    reloader_loops["auto"] = reloader_loops["watchdog"]


def ensure_echo_on() -> None:
    """Ensure that echo mode is enabled. Some tools such as PDB disable
    it which causes usability issues after a reload."""
    # tcgetattr will fail if stdin isn't a tty
    if sys.stdin is None or not sys.stdin.isatty():
        return

    try:
        import termios
    except ImportError:
        return

    attributes = termios.tcgetattr(sys.stdin)

    if not attributes[3] & termios.ECHO:
        attributes[3] |= termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, attributes)


def run_with_reloader(
    main_func: t.Callable[[], None],
    extra_files: t.Iterable[str] | None = None,
    exclude_patterns: t.Iterable[str] | None = None,
    interval: int | float = 1,
    reloader_type: str = "auto",
) -> None:
    """Run the given function in an independent Python interpreter."""
    import signal

    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    reloader = reloader_loops[reloader_type](
        extra_files=extra_files, exclude_patterns=exclude_patterns, interval=interval
    )

    try:
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            ensure_echo_on()
            t = threading.Thread(target=main_func, args=())
            t.daemon = True

            # Enter the reloader to set up initial state, then start
            # the app thread and reloader update loop.
            with reloader:
                t.start()
                reloader.run()
        else:
            sys.exit(reloader.restart_with_reloader())
    except KeyboardInterrupt:
        pass
<|endoftext|>"""Implements a number of Python exceptions which can be raised from within
a view to trigger a standard HTTP non-200 response.

Usage Example
-------------

.. code-block:: python

    from werkzeug.wrappers.request import Request
    from werkzeug.exceptions import HTTPException, NotFound

    def view(request):
        raise NotFound()

    @Request.application
    def application(request):
        try:
            return view(request)
        except HTTPException as e:
            return e

As you can see from this example those exceptions are callable WSGI
applications. However, they are not Werkzeug response objects. You
can get a response object by calling ``get_response()`` on a HTTP
exception.

Keep in mind that you may have to pass an environ (WSGI) or scope
(ASGI) to ``get_response()`` because some errors fetch additional
information relating to the request.

If you want to hook in a different exception page to say, a 404 status
code, you can add a second except for a specific subclass of an error:

.. code-block:: python

    @Request.application
    def application(request):
        try:
            return view(request)
        except NotFound as e:
            return not_found(request)
        except HTTPException as e:
            return e

"""

from __future__ import annotations

import typing as t
from datetime import datetime

from markupsafe import escape
from markupsafe import Markup

from ._internal import _get_environ

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIEnvironment

    from .datastructures import WWWAuthenticate
    from .sansio.response import Response as SansIOResponse
    from .wrappers.request import Request as WSGIRequest
    from .wrappers.response import Response as WSGIResponse


class HTTPException(Exception):
    """The base class for all HTTP exceptions. This exception can be called as a WSGI
    application to render a default error page or you can catch the subclasses
    of it independently and render nicer error messages.

    .. versionchanged:: 2.1
        Removed the ``wrap`` class method.
    """

    code: int | None = None
    description: str | None = None

    def __init__(
        self,
        description: str | None = None,
        response: SansIOResponse | None = None,
    ) -> None:
        super().__init__()
        if description is not None:
            self.description = description
        self.response = response

    @property
    def name(self) -> str:
        """The status name."""
        from .http import HTTP_STATUS_CODES

        return HTTP_STATUS_CODES.get(self.code, "Unknown Error")  # type: ignore

    def get_description(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> str:
        """Get the description."""
        if self.description is None:
            description = ""
        else:
            description = self.description

        description = escape(description).replace("\n", Markup("<br>"))
        return f"<p>{description}</p>"

    def get_body(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> str:
        """Get the HTML body."""
        return (
            "<!doctype html>\n"
            "<html lang=en>\n"
            f"<title>{self.code} {escape(self.name)}</title>\n"
            f"<h1>{escape(self.name)}</h1>\n"
            f"{self.get_description(environ)}\n"
        )

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        """Get a list of headers."""
        return [("Content-Type", "text/html; charset=utf-8")]

    @t.overload
    def get_response(
        self,
        environ: WSGIEnvironment | WSGIRequest | None = ...,
        scope: None = None,
    ) -> WSGIResponse: ...
    @t.overload
    def get_response(
        self,
        environ: None = None,
        scope: dict[str, t.Any] = ...,
    ) -> SansIOResponse: ...
    def get_response(
        self,
        environ: WSGIEnvironment | WSGIRequest | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> WSGIResponse | SansIOResponse:
        """Get a response object.

        :param environ: A WSGI environ dict or request object. If given, may be
            used to customize the response based on the request.
        :param scope: An ASGI scope dict. If given, may be used to customize the
            response based on the request.
        :return: A WSGI :class:`werkzeug.wrappers.Response` if called without
            arguments or with ``environ``. A sans-IO
            :class:`werkzeug.sansio.Response` for ASGI if called with
            ``scope``.
        """
        from .wrappers.response import Response

        if self.response is not None:
            return self.response
        if environ is not None:
            environ = _get_environ(environ)
        headers = self.get_headers(environ, scope)
        return Response(self.get_body(environ, scope), self.code, headers)

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Call the exception as WSGI application.

        :param environ: the WSGI environment.
        :param start_response: the response callable provided by the WSGI
                               server.
        """
        response = self.get_response(environ)
        return response(environ, start_response)

    def __str__(self) -> str:
        code = self.code if self.code is not None else "???"
        return f"{code} {self.name}: {self.description}"

    def __repr__(self) -> str:
        code = self.code if self.code is not None else "???"
        return f"<{type(self).__name__} '{code}: {self.name}'>"


class BadRequest(HTTPException):
    """*400* `Bad Request`

    Raise if the browser sends something to the application the application
    or server cannot handle.
    """

    code = 400
    description = (
        "The browser (or proxy) sent a request that this server could not understand."
    )


class BadRequestKeyError(BadRequest, KeyError):
    """An exception that is used to signal both a :exc:`KeyError` and a
    :exc:`BadRequest`. Used by many of the datastructures.
    """

    _description = BadRequest.description
    #: Show the KeyError along with the HTTP error message in the
    #: response. This should be disabled in production, but can be
    #: useful in a debug mode.
    show_exception = False

    def __init__(self, arg: object | None = None, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)

        if arg is None:
            KeyError.__init__(self)
        else:
            KeyError.__init__(self, arg)

    @property
    def description(self) -> str:
        if self.show_exception:
            return f"{self._description}\n{KeyError.__name__}: {KeyError.__str__(self)}"

        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value


class ClientDisconnected(BadRequest):
    """Internal exception that is raised if Werkzeug detects a disconnected
    client.  Since the client is already gone at that point attempting to
    send the error message to the client might not work and might ultimately
    result in another exception in the server.  Mainly this is here so that
    it is silenced by default as far as Werkzeug is concerned.

    Since disconnections cannot be reliably detected and are unspecified
    by WSGI to a large extent this might or might not be raised if a client
    is gone.

    .. versionadded:: 0.8
    """


class SecurityError(BadRequest):
    """Raised if something triggers a security error.  This is otherwise
    exactly like a bad request error.

    .. versionadded:: 0.9
    """


class BadHost(BadRequest):
    """Raised if the submitted host is badly formatted.

    .. versionadded:: 0.11.2
    """


class Unauthorized(HTTPException):
    """*401* ``Unauthorized``

    Raise if the user is not authorized to access a resource.

    The ``www_authenticate`` argument should be used to set the
    ``WWW-Authenticate`` header. This is used for HTTP basic auth and
    other schemes. Use :class:`~werkzeug.datastructures.WWWAuthenticate`
    to create correctly formatted values. Strictly speaking a 401
    response is invalid if it doesn't provide at least one value for
    this header, although real clients typically don't care.

    :param description: Override the default message used for the body
        of the response.
    :param www-authenticate: A single value, or list of values, for the
        WWW-Authenticate header(s).

    .. versionchanged:: 2.0
        Serialize multiple ``www_authenticate`` items into multiple
        ``WWW-Authenticate`` headers, rather than joining them
        into a single value, for better interoperability.

    .. versionchanged:: 0.15.3
        If the ``www_authenticate`` argument is not set, the
        ``WWW-Authenticate`` header is not set.

    .. versionchanged:: 0.15.3
        The ``response`` argument was restored.

    .. versionchanged:: 0.15.1
        ``description`` was moved back as the first argument, restoring
         its previous position.

    .. versionchanged:: 0.15.0
        ``www_authenticate`` was added as the first argument, ahead of
        ``description``.
    """

    code = 401
    description = (
        "The server could not verify that you are authorized to access"
        " the URL requested. You either supplied the wrong credentials"
