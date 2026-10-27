            if hide_value in {"before", "before_and_this"}:
                new_stack = []
                hidden = False

                if hide_value == "before_and_this":
                    continue
            elif hide_value in {"reset", "reset_and_this"}:
                hidden = False

                if hide_value == "reset_and_this":
                    continue
            elif hide_value in {"after", "after_and_this"}:
                hidden = True

                if hide_value == "after_and_this":
                    continue
            elif hide_value or hidden:
                continue

        frame_args: dict[str, t.Any] = {
            "filename": fs.filename,
            "lineno": fs.lineno,
            "name": fs.name,
            "locals": f.f_locals,
            "globals": f.f_globals,
        }

        if sys.version_info >= (3, 11):
            frame_args["colno"] = fs.colno
            frame_args["end_colno"] = fs.end_colno

        new_stack.append(DebugFrameSummary(**frame_args))

    # The codeop module is used to compile code from the interactive
    # debugger. Hide any codeop frames from the bottom of the traceback.
    while new_stack:
        module = new_stack[0].global_ns.get("__name__")

        if module is None:
            module = new_stack[0].local_ns.get("__name__")

        if module == "codeop":
            del new_stack[0]
        else:
            break

    te.stack[:] = new_stack

    if te.__context__:
        context_exc = t.cast(BaseException, exc.__context__)
        te.__context__ = _process_traceback(context_exc, te.__context__, hide=hide)

    if te.__cause__:
        cause_exc = t.cast(BaseException, exc.__cause__)
        te.__cause__ = _process_traceback(cause_exc, te.__cause__, hide=hide)

    return te


class DebugTraceback:
    __slots__ = ("_te", "_cache_all_tracebacks", "_cache_all_frames")

    def __init__(
        self,
        exc: BaseException,
        te: traceback.TracebackException | None = None,
        *,
        skip: int = 0,
        hide: bool = True,
    ) -> None:
        self._te = _process_traceback(exc, te, skip=skip, hide=hide)

    def __str__(self) -> str:
        return f"<{type(self).__name__} {self._te}>"

    @cached_property
    def all_tracebacks(
        self,
    ) -> list[tuple[str | None, traceback.TracebackException]]:
        out: list[tuple[str | None, traceback.TracebackException]] = []
        current: traceback.TracebackException | None = self._te

        while current is not None:
            if current.__cause__ is not None:
                chained_msg = (
                    "The above exception was the direct cause of the"
                    " following exception"
                )
                chained_exc = current.__cause__
            elif current.__context__ is not None and not current.__suppress_context__:
                chained_msg = (
                    "During handling of the above exception, another exception occurred"
                )
                chained_exc = current.__context__
            else:
                chained_msg = None
                chained_exc = None

            out.append((chained_msg, current))
            current = chained_exc

        return out

    @cached_property
    def all_frames(self) -> list[DebugFrameSummary]:
        return [
            f  # type: ignore[misc]
            for _, te in self.all_tracebacks
            for f in te.stack
        ]

    def render_traceback_text(self) -> str:
        return "".join(self._te.format())

    def render_traceback_html(self, include_title: bool = True) -> str:
        library_frames = [f.is_library for f in self.all_frames]
        mark_library = 0 < sum(library_frames) < len(library_frames)
        rows = []

        if not library_frames:
            classes = "traceback noframe-traceback"
        else:
            classes = "traceback"

            for msg, current in reversed(self.all_tracebacks):
                row_parts = []

                if msg is not None:
                    row_parts.append(f'<li><div class="exc-divider">{msg}:</div>')

                for frame in current.stack:
                    frame = t.cast(DebugFrameSummary, frame)
                    info = f' title="{escape(frame.info)}"' if frame.info else ""
                    row_parts.append(f"<li{info}>{frame.render_html(mark_library)}")

                rows.append("\n".join(row_parts))

        if sys.version_info < (3, 13):
            exc_type_str = self._te.exc_type.__name__
        else:
            exc_type_str = self._te.exc_type_str

        is_syntax_error = exc_type_str == "SyntaxError"

        if include_title:
            if is_syntax_error:
                title = "Syntax Error"
            else:
                title = "Traceback <em>(most recent call last)</em>:"
        else:
            title = ""

        exc_full = escape("".join(self._te.format_exception_only()))

        if is_syntax_error:
            description = f"<pre class=syntaxerror>{exc_full}</pre>"
        else:
            description = f"<blockquote>{exc_full}</blockquote>"

        return SUMMARY_HTML % {
            "classes": classes,
            "title": f"<h3>{title}</h3>",
            "frames": "\n".join(rows),
            "description": description,
        }

    def render_debugger_html(
        self, evalex: bool, secret: str, evalex_trusted: bool
    ) -> str:
        exc_lines = list(self._te.format_exception_only())
        plaintext = "".join(self._te.format())

        if sys.version_info < (3, 13):
            exc_type_str = self._te.exc_type.__name__
        else:
            exc_type_str = self._te.exc_type_str

        return PAGE_HTML % {
            "evalex": "true" if evalex else "false",
            "evalex_trusted": "true" if evalex_trusted else "false",
            "console": "false",
            "title": escape(exc_lines[0]),
            "exception": escape("".join(exc_lines)),
            "exception_type": escape(exc_type_str),
            "summary": self.render_traceback_html(include_title=False),
            "plaintext": escape(plaintext),
            "plaintext_cs": re.sub("-{2,}", "-", plaintext),
            "secret": secret,
        }


class DebugFrameSummary(traceback.FrameSummary):
    """A :class:`traceback.FrameSummary` that can evaluate code in the
    frame's namespace.
    """

    __slots__ = (
        "local_ns",
        "global_ns",
        "_cache_info",
        "_cache_is_library",
        "_cache_console",
    )

    def __init__(
        self,
        *,
        locals: dict[str, t.Any],
        globals: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> None:
        super().__init__(locals=None, **kwargs)
        self.local_ns = locals
        self.global_ns = globals

    @cached_property
    def info(self) -> str | None:
        return self.local_ns.get("__traceback_info__")

    @cached_property
    def is_library(self) -> bool:
        return any(
            self.filename.startswith((path, os.path.realpath(path)))
            for path in sysconfig.get_paths().values()
        )

    @cached_property
    def console(self) -> Console:
        return Console(self.global_ns, self.local_ns)

    def eval(self, code: str) -> t.Any:
        return self.console.eval(code)

    def render_html(self, mark_library: bool) -> str:
        context = 5
        lines = linecache.getlines(self.filename)
        line_idx = self.lineno - 1  # type: ignore[operator]
        start_idx = max(0, line_idx - context)
        stop_idx = min(len(lines), line_idx + context + 1)
        rendered_lines = []

        def render_line(line: str, cls: str) -> None:
            line = line.expandtabs().rstrip()
            stripped_line = line.strip()
            prefix = len(line) - len(stripped_line)
            colno = getattr(self, "colno", 0)
            end_colno = getattr(self, "end_colno", 0)

            if cls == "current" and colno and end_colno:
                arrow = (
                    f'\n<span class="ws">{" " * prefix}</span>'
                    f"{' ' * (colno - prefix)}{'^' * (end_colno - colno)}"
                )
            else:
                arrow = ""

            rendered_lines.append(
                f'<pre class="line {cls}"><span class="ws">{" " * prefix}</span>'
                f"{escape(stripped_line) if stripped_line else ' '}"
                f"{arrow if arrow else ''}</pre>"
            )

        if line_idx < len(lines):
            for line in lines[start_idx:line_idx]:
                render_line(line, "before")

            render_line(lines[line_idx], "current")

            for line in lines[line_idx + 1 : stop_idx]:
                render_line(line, "after")

        return FRAME_HTML % {
            "id": id(self),
            "filename": escape(self.filename),
            "lineno": self.lineno,
            "function_name": escape(self.name),
            "lines": "\n".join(rendered_lines),
            "library": "library" if mark_library and self.is_library else "",
        }


def render_console_html(secret: str, evalex_trusted: bool) -> str:
    return CONSOLE_HTML % {
        "evalex": "true",
        "evalex_trusted": "true" if evalex_trusted else "false",
        "console": "true",
        "title": "Console",
        "secret": secret,
    }
<|endoftext|>from __future__ import annotations

import json
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin

from .._internal import _get_environ
from ..datastructures import Headers
from ..http import generate_etag
from ..http import http_date
from ..http import is_resource_modified
from ..http import parse_etags
from ..http import parse_range_header
from ..http import remove_entity_headers
from ..sansio.response import Response as _SansIOResponse
from ..urls import iri_to_uri
from ..utils import cached_property
from ..wsgi import _RangeWrapper
from ..wsgi import ClosingIterator
from ..wsgi import get_current_url

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

    from .request import Request


def _iter_encoded(iterable: t.Iterable[str | bytes]) -> t.Iterator[bytes]:
    for item in iterable:
        if isinstance(item, str):
            yield item.encode()
        else:
            yield item


class Response(_SansIOResponse):
    """Represents an outgoing WSGI HTTP response with body, status, and
    headers. Has properties and methods for using the functionality
    defined by various HTTP specs.

    The response body is flexible to support different use cases. The
    simple form is passing bytes, or a string which will be encoded as
    UTF-8. Passing an iterable of bytes or strings makes this a
    streaming response. A generator is particularly useful for building
    a CSV file in memory or using SSE (Server Sent Events). A file-like
    object is also iterable, although the
    :func:`~werkzeug.utils.send_file` helper should be used in that
    case.

    The response object is itself a WSGI application callable. When
    called (:meth:`__call__`) with ``environ`` and ``start_response``,
    it will pass its status and headers to ``start_response`` then
    return its body as an iterable.

    .. code-block:: python

        from werkzeug.wrappers.response import Response

        def index():
            return Response("Hello, World!")

        def application(environ, start_response):
            path = environ.get("PATH_INFO") or "/"

            if path == "/":
                response = index()
            else:
                response = Response("Not Found", status=404)

            return response(environ, start_response)

    :param response: The data for the body of the response. A string or
        bytes, or tuple or list of strings or bytes, for a fixed-length
        response, or any other iterable of strings or bytes for a
        streaming response. Defaults to an empty body.
    :param status: The status code for the response. Either an int, in
        which case the default status message is added, or a string in
        the form ``{code} {message}``, like ``404 Not Found``. Defaults
        to 200.
    :param headers: A :class:`~werkzeug.datastructures.Headers` object,
        or a list of ``(key, value)`` tuples that will be converted to a
        ``Headers`` object.
    :param mimetype: The mime type (content type without charset or
        other parameters) of the response. If the value starts with
        ``text/`` (or matches some other special cases), the charset
        will be added to create the ``content_type``.
    :param content_type: The full content type of the response.
        Overrides building the value from ``mimetype``.
    :param direct_passthrough: Pass the response body directly through
        as the WSGI iterable. This can be used when the body is a binary
        file or other iterator of bytes, to skip some unnecessary
        checks. Use :func:`~werkzeug.utils.send_file` instead of setting
        this manually.

    .. versionchanged:: 2.1
        Old ``BaseResponse`` and mixin classes were removed.

    .. versionchanged:: 2.0
        Combine ``BaseResponse`` and mixins into a single ``Response``
        class.

    .. versionchanged:: 0.5
        The ``direct_passthrough`` parameter was added.
    """

    #: if set to `False` accessing properties on the response object will
    #: not try to consume the response iterator and convert it into a list.
    #:
    #: .. versionadded:: 0.6.2
    #:
    #:    That attribute was previously called `implicit_seqence_conversion`.
    #:    (Notice the typo).  If you did use this feature, you have to adapt
    #:    your code to the name change.
    implicit_sequence_conversion = True

    #: If a redirect ``Location`` header is a relative URL, make it an
    #: absolute URL, including scheme and domain.
    #:
    #: .. versionchanged:: 2.1
    #:     This is disabled by default, so responses will send relative
    #:     redirects.
    #:
    #: .. versionadded:: 0.8
    autocorrect_location_header = False

    #: Should this response object automatically set the content-length
    #: header if possible?  This is true by default.
    #:
    #: .. versionadded:: 0.8
    automatically_set_content_length = True

    #: The response body to send as the WSGI iterable. A list of strings
    #: or bytes represents a fixed-length response, any other iterable
    #: is a streaming response. Strings are encoded to bytes as UTF-8.
    #:
    #: Do not set to a plain string or bytes, that will cause sending
    #: the response to be very inefficient as it will iterate one byte
    #: at a time.
    response: t.Iterable[str] | t.Iterable[bytes]

    def __init__(
        self,
        response: t.Iterable[bytes] | bytes | t.Iterable[str] | str | None = None,
        status: int | str | HTTPStatus | None = None,
        headers: t.Mapping[str, str | t.Iterable[str]]
        | t.Iterable[tuple[str, str]]
        | None = None,
        mimetype: str | None = None,
        content_type: str | None = None,
        direct_passthrough: bool = False,
    ) -> None:
        super().__init__(
            status=status,
            headers=headers,
            mimetype=mimetype,
            content_type=content_type,
        )

        #: Pass the response body directly through as the WSGI iterable.
        #: This can be used when the body is a binary file or other
        #: iterator of bytes, to skip some unnecessary checks. Use
        #: :func:`~werkzeug.utils.send_file` instead of setting this
        #: manually.
        self.direct_passthrough = direct_passthrough
        self._on_close: list[t.Callable[[], t.Any]] = []

        # we set the response after the headers so that if a class changes
        # the charset attribute, the data is set in the correct charset.
        if response is None:
            self.response = []
        elif isinstance(response, (str, bytes, bytearray)):
            self.set_data(response)
        else:
            self.response = response

    def call_on_close(self, func: t.Callable[[], t.Any]) -> t.Callable[[], t.Any]:
        """Adds a function to the internal list of functions that should
        be called as part of closing down the response.  Since 0.7 this
        function also returns the function that was passed so that this
        can be used as a decorator.

        .. versionadded:: 0.6
        """
        self._on_close.append(func)
        return func

    def __repr__(self) -> str:
        if self.is_sequence:
            body_info = f"{sum(map(len, self.iter_encoded()))} bytes"
        else:
            body_info = "streamed" if self.is_streamed else "likely-streamed"
        return f"<{type(self).__name__} {body_info} [{self.status}]>"

    @classmethod
    def force_type(
        cls, response: Response, environ: WSGIEnvironment | None = None
    ) -> Response:
        """Enforce that the WSGI response is a response object of the current
        type.  Werkzeug will use the :class:`Response` internally in many
        situations like the exceptions.  If you call :meth:`get_response` on an
        exception you will get back a regular :class:`Response` object, even
        if you are using a custom subclass.

        This method can enforce a given response type, and it will also
        convert arbitrary WSGI callables into response objects if an environ
        is provided::

            # convert a Werkzeug response object into an instance of the
            # MyResponseClass subclass.
            response = MyResponseClass.force_type(response)

            # convert any WSGI application into a response object
            response = MyResponseClass.force_type(response, environ)

        This is especially useful if you want to post-process responses in
        the main dispatcher and use functionality provided by your subclass.

        Keep in mind that this will modify response objects in place if
        possible!

        :param response: a response object or wsgi application.
        :param environ: a WSGI environment object.
        :return: a response object.
        """
        if not isinstance(response, Response):
            if environ is None:
                raise TypeError(
                    "cannot convert WSGI application into response"
                    " objects without an environ"
                )

            from ..test import run_wsgi_app

            response = Response(*run_wsgi_app(response, environ))

        response.__class__ = cls
        return response

    @classmethod
    def from_app(
        cls, app: WSGIApplication, environ: WSGIEnvironment, buffered: bool = False
    ) -> Response:
        """Create a new response object from an application output.  This
        works best if you pass it an application that returns a generator all
        the time.  Sometimes applications may use the `write()` callable
        returned by the `start_response` function.  This tries to resolve such
        edge cases automatically.  But if you don't get the expected output
        you should set `buffered` to `True` which enforces buffering.

        :param app: the WSGI application to execute.
        :param environ: the WSGI environment to execute against.
        :param buffered: set to `True` to enforce buffering.
        :return: a response object.
        """
        from ..test import run_wsgi_app

        return cls(*run_wsgi_app(app, environ, buffered))

    @t.overload
    def get_data(self, as_text: t.Literal[False] = False) -> bytes: ...

    @t.overload
    def get_data(self, as_text: t.Literal[True]) -> str: ...

    def get_data(self, as_text: bool = False) -> bytes | str:
        """The string representation of the response body.  Whenever you call
        this property the response iterable is encoded and flattened.  This
        can lead to unwanted behavior if you stream big data.

        This behavior can be disabled by setting
        :attr:`implicit_sequence_conversion` to `False`.

        If `as_text` is set to `True` the return value will be a decoded
        string.

        .. versionadded:: 0.9
        """
        self._ensure_sequence()
        rv = b"".join(self.iter_encoded())

        if as_text:
            return rv.decode()

        return rv

    def set_data(self, value: bytes | str) -> None:
        """Sets a new string as response.  The value must be a string or
        bytes. If a string is set it's encoded to the charset of the
        response (utf-8 by default).

        .. versionadded:: 0.9
        """
        if isinstance(value, str):
            value = value.encode()
        self.response = [value]
        if self.automatically_set_content_length:
            self.headers["Content-Length"] = str(len(value))

    data = property(
        get_data,
        set_data,
        doc="A descriptor that calls :meth:`get_data` and :meth:`set_data`.",
    )

    def calculate_content_length(self) -> int | None:
        """Returns the content length if available or `None` otherwise."""
        try:
            self._ensure_sequence()
        except RuntimeError:
            return None
        return sum(len(x) for x in self.iter_encoded())

    def _ensure_sequence(self, mutable: bool = False) -> None:
        """This method can be called by methods that need a sequence.  If
        `mutable` is true, it will also ensure that the response sequence
        is a standard Python list.

        .. versionadded:: 0.6
        """
        if self.is_sequence:
            # if we need a mutable object, we ensure it's a list.
            if mutable and not isinstance(self.response, list):
                self.response = list(self.response)  # type: ignore
            return
        if self.direct_passthrough:
            raise RuntimeError(
                "Attempted implicit sequence conversion but the"
                " response object is in direct passthrough mode."
            )
        if not self.implicit_sequence_conversion:
            raise RuntimeError(
                "The response object required the iterable to be a"
                " sequence, but the implicit conversion was disabled."
                " Call make_sequence() yourself."
            )
        self.make_sequence()

    def make_sequence(self) -> None:
        """Converts the response iterator in a list.  By default this happens
        automatically if required.  If `implicit_sequence_conversion` is
        disabled, this method is not automatically called and some properties
        might raise exceptions.  This also encodes all the items.

        .. versionadded:: 0.6
        """
        if not self.is_sequence:
            # if we consume an iterable we have to ensure that the close
            # method of the iterable is called if available when we tear
            # down the response
            close = getattr(self.response, "close", None)
            self.response = list(self.iter_encoded())
            if close is not None:
                self.call_on_close(close)

    def iter_encoded(self) -> t.Iterator[bytes]:
        """Iter the response encoded with the encoding of the response.
        If the response object is invoked as WSGI application the return
        value of this method is used as application iterator unless
        :attr:`direct_passthrough` was activated.
        """
        # Encode in a separate function so that self.response is fetched
        # early.  This allows us to wrap the response with the return
        # value from get_app_iter or iter_encoded.
        return _iter_encoded(self.response)

    @property
    def is_streamed(self) -> bool:
        """If the response is streamed (the response is not an iterable with
        a length information) this property is `True`.  In this case streamed
        means that there is no information about the number of iterations.
        This is usually `True` if a generator is passed to the response object.

        This is useful for checking before applying some sort of post
        filtering that should not take place for streamed responses.
        """
        try:
            len(self.response)  # type: ignore
        except (TypeError, AttributeError):
            return True
        return False

    @property
    def is_sequence(self) -> bool:
        """If the iterator is buffered, this property will be `True`.  A
        response object will consider an iterator to be buffered if the
        response attribute is a list or tuple.

        .. versionadded:: 0.6
        """
        return isinstance(self.response, (tuple, list))

    def close(self) -> None:
        """Close the wrapped response if possible.  You can also use the object
        in a with statement which will automatically close it.

        .. versionadded:: 0.9
           Can now be used in a with statement.
        """
        if hasattr(self.response, "close"):
            self.response.close()
        for func in self._on_close:
            func()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, exc_type, exc_value, tb):  # type: ignore
        self.close()

    def freeze(self) -> None:
        """Make the response object ready to be pickled. Does the
        following:

        *   Buffer the response into a list, ignoring
            :attr:`implicity_sequence_conversion` and
            :attr:`direct_passthrough`.
        *   Set the ``Content-Length`` header.
        *   Generate an ``ETag`` header if one is not already set.

        .. versionchanged:: 2.1
            Removed the ``no_etag`` parameter.

        .. versionchanged:: 2.0
            An ``ETag`` header is always added.

        .. versionchanged:: 0.6
            The ``Content-Length`` header is set.
        """
        # Always freeze the encoded response body, ignore
        # implicit_sequence_conversion and direct_passthrough.
        self.response = list(self.iter_encoded())
        self.headers["Content-Length"] = str(sum(map(len, self.response)))
        self.add_etag()

    def get_wsgi_headers(self, environ: WSGIEnvironment) -> Headers:
        """This is automatically called right before the response is started
        and returns headers modified for the given environment.  It returns a
        copy of the headers from the response with some modifications applied
        if necessary.

        For example the location header (if present) is joined with the root
        URL of the environment.  Also the content length is automatically set
        to zero here for certain status codes.

        .. versionchanged:: 0.6
           Previously that function was called `fix_headers` and modified
           the response object in place.  Also since 0.6, IRIs in location
           and content-location headers are handled properly.

           Also starting with 0.6, Werkzeug will attempt to set the content
           length if it is able to figure it out on its own.  This is the
           case if all the strings in the response iterable are already
           encoded and the iterable is buffered.

        :param environ: the WSGI environment of the request.
        :return: returns a new :class:`~werkzeug.datastructures.Headers`
                 object.
        """
        headers = Headers(self.headers)
        location: str | None = None
        content_location: str | None = None
        content_length: str | int | None = None
        status = self.status_code

        # iterate over the headers to find all values in one go.  Because
        # get_wsgi_headers is used each response that gives us a tiny
        # speedup.
        for key, value in headers:
            ikey = key.lower()
            if ikey == "location":
                location = value
            elif ikey == "content-location":
                content_location = value
            elif ikey == "content-length":
                content_length = value

        if location is not None:
            location = iri_to_uri(location)

            if self.autocorrect_location_header:
                # Make the location header an absolute URL.
                current_url = get_current_url(environ, strip_querystring=True)
                current_url = iri_to_uri(current_url)
                location = urljoin(current_url, location)

            headers["Location"] = location

        # make sure the content location is a URL
        if content_location is not None:
            headers["Content-Location"] = iri_to_uri(content_location)

        if 100 <= status < 200 or status == 204:
            # Per section 3.3.2 of RFC 7230, "a server MUST NOT send a
            # Content-Length header field in any response with a status
            # code of 1xx (Informational) or 204 (No Content)."
            headers.remove("Content-Length")
        elif status == 304:
            remove_entity_headers(headers)

        # if we can determine the content length automatically, we
        # should try to do that.  But only if this does not involve
        # flattening the iterator or encoding of strings in the
        # response. We however should not do that if we have a 304
        # response.
        if (
            self.automatically_set_content_length
            and self.is_sequence
            and content_length is None
            and status not in (204, 304)
            and not (100 <= status < 200)
        ):
            content_length = sum(len(x) for x in self.iter_encoded())
            headers["Content-Length"] = str(content_length)

        return headers

    def get_app_iter(self, environ: WSGIEnvironment) -> t.Iterable[bytes]:
        """Returns the application iterator for the given environ.  Depending
        on the request method and the current status code the return value
        might be an empty response rather than the one from the response.

        If the request method is `HEAD` or the status code is in a range
        where the HTTP specification requires an empty response, an empty
        iterable is returned.

        .. versionadded:: 0.6

        :param environ: the WSGI environment of the request.
        :return: a response iterable.
        """
        status = self.status_code
        if (
            environ["REQUEST_METHOD"] == "HEAD"
            or 100 <= status < 200
            or status in (204, 304)
        ):
            iterable: t.Iterable[bytes] = ()
        elif self.direct_passthrough:
            return self.response  # type: ignore
        else:
            iterable = self.iter_encoded()
        return ClosingIterator(iterable, self.close)

    def get_wsgi_response(
        self, environ: WSGIEnvironment
    ) -> tuple[t.Iterable[bytes], str, list[tuple[str, str]]]:
        """Returns the final WSGI response as tuple.  The first item in
        the tuple is the application iterator, the second the status and
        the third the list of headers.  The response returned is created
        specially for the given environment.  For example if the request
        method in the WSGI environment is ``'HEAD'`` the response will
        be empty and only the headers and status code will be present.

        .. versionadded:: 0.6

        :param environ: the WSGI environment of the request.
        :return: an ``(app_iter, status, headers)`` tuple.
        """
        headers = self.get_wsgi_headers(environ)
        app_iter = self.get_app_iter(environ)
        return app_iter, self.status, headers.to_wsgi_list()

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Process this response as WSGI application.

        :param environ: the WSGI environment.
        :param start_response: the response callable provided by the WSGI
                               server.
        :return: an application iterator
        """
        app_iter, status, headers = self.get_wsgi_response(environ)
        start_response(status, headers)
        return app_iter

    # JSON

    #: A module or other object that has ``dumps`` and ``loads``
    #: functions that match the API of the built-in :mod:`json` module.
    json_module = json

    @property
    def json(self) -> t.Any | None:
        """The parsed JSON data if :attr:`mimetype` indicates JSON
        (:mimetype:`application/json`, see :attr:`is_json`).

        Calls :meth:`get_json` with default arguments.
        """
        return self.get_json()

    @t.overload
    def get_json(self, force: bool = ..., silent: t.Literal[False] = ...) -> t.Any: ...

    @t.overload
    def get_json(self, force: bool = ..., silent: bool = ...) -> t.Any | None: ...

    def get_json(self, force: bool = False, silent: bool = False) -> t.Any | None:
        """Parse :attr:`data` as JSON. Useful during testing.

        If the mimetype does not indicate JSON
        (:mimetype:`application/json`, see :attr:`is_json`), this
        returns ``None``.

        Unlike :meth:`Request.get_json`, the result is not cached.

        :param force: Ignore the mimetype and always try to parse JSON.
        :param silent: Silence parsing errors and return ``None``
            instead.
        """
        if not (force or self.is_json):
            return None

        data = self.get_data()

        try:
            return self.json_module.loads(data)
        except ValueError:
            if not silent:
                raise

            return None

    # Stream

    @cached_property
    def stream(self) -> ResponseStream:
        """The response iterable as write-only stream."""
        return ResponseStream(self)

    def _wrap_range_response(self, start: int, length: int) -> None:
        """Wrap existing Response in case of Range Request context."""
        if self.status_code == 206:
            self.response = _RangeWrapper(self.response, start, length)  # type: ignore

    def _is_range_request_processable(self, environ: WSGIEnvironment) -> bool:
        """Return ``True`` if `Range` header is present and if underlying
        resource is considered unchanged when compared with `If-Range` header.
        """
        return "HTTP_RANGE" in environ and (
            "HTTP_IF_RANGE" not in environ
            or not is_resource_modified(
                environ,
                self.headers.get("ETag"),
                last_modified=self.headers.get("Last-Modified"),
                ignore_if_range=False,
            )
        )

    def _process_range_request(
        self,
        environ: WSGIEnvironment,
        complete_length: int | None,
        accept_ranges: bool | str,
    ) -> bool:
        """Handle Range Request related headers (RFC7233).  If `Accept-Ranges`
        header is valid, and Range Request is processable, we set the headers
        as described by the RFC, and wrap the underlying response in a
        RangeWrapper.

        Returns ``True`` if Range Request can be fulfilled, ``False`` otherwise.

        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchanged:: 3.2
            Adds the ``Accept-Ranges`` header if ``accept_ranges`` is passed,
            even if this is not a satisfiable range request.

        .. versionchanged:: 2.0
            Returns ``False`` if the length is 0.
        """
        from ..exceptions import RequestedRangeNotSatisfiable

        if not accept_ranges:
            return False

        if accept_ranges is True:
            accept_ranges = "bytes"

        self.accept_ranges = accept_ranges

        if not (complete_length and self._is_range_request_processable(environ)):
            return False

        parsed_range = parse_range_header(environ.get("HTTP_RANGE"))

        if parsed_range is None:
            raise RequestedRangeNotSatisfiable(complete_length)

        range_tuple = parsed_range.range_for_length(complete_length)
        content_range_header = parsed_range.to_content_range_header(complete_length)

        if range_tuple is None or content_range_header is None:
            raise RequestedRangeNotSatisfiable(complete_length)

        content_length = range_tuple[1] - range_tuple[0]
        self.content_length = content_length
        self.content_range = content_range_header
        self.status_code = 206
        self._wrap_range_response(range_tuple[0], content_length)
        return True

    def make_conditional(
        self,
        request_or_environ: WSGIEnvironment | Request,
        accept_ranges: bool | str = False,
        complete_length: int | None = None,
    ) -> Response:
        """Make the response conditional to the request.  This method works
        best if an etag was defined for the response already.  The `add_etag`
        method can be used to do that.  If called without etag just the date
        header is set.

        This does nothing if the request method in the request or environ is
        anything but GET or HEAD.

        For optimal performance when handling range requests, it's recommended
        that your response data object implements `seekable`, `seek` and `tell`
        methods as described by :py:class:`io.IOBase`.  Objects returned by
        :meth:`~werkzeug.wsgi.wrap_file` automatically implement those methods.

        It does not remove the body of the response because that's something
        the :meth:`__call__` function does for us automatically.

        Returns self so that you can do ``return resp.make_conditional(req)``
        but modifies the object in-place.

        :param request_or_environ: a request object or WSGI environment to be
                                   used to make the response conditional
                                   against.
        :param accept_ranges: This parameter dictates the value of
                              `Accept-Ranges` header. If ``False`` (default),
                              the header is not set. If ``True``, it will be set
                              to ``"bytes"``. If it's a string, it will use this
                              value.
        :param complete_length: Will be used only in valid Range Requests.
                                It will set `Content-Range` complete length
                                value and compute `Content-Length` real value.
                                This parameter is mandatory for successful
                                Range Requests completion.
        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchangedd: 3.2
            Adds the ``Accept-Ranges`` header if ``accept_ranges`` is passed,
            even if this is not a satisfiable range request.

        .. versionchanged:: 2.0
            Range processing is skipped if length is 0 instead of
            raising a 416 Range Not Satisfiable error.
        """
        environ = _get_environ(request_or_environ)
        if environ["REQUEST_METHOD"] in ("GET", "HEAD"):
            # if the date is not in the headers, add it now.  We however
            # will not override an already existing header.  Unfortunately
            # this header will be overridden by many WSGI servers including
            # wsgiref.
            if "date" not in self.headers:
                self.headers["Date"] = http_date()
            is206 = self._process_range_request(environ, complete_length, accept_ranges)
            if not is206 and not is_resource_modified(
                environ,
                self.headers.get("ETag"),
                last_modified=self.headers.get("Last-Modified"),
            ):
                if parse_etags(environ.get("HTTP_IF_MATCH")):
                    self.status_code = 412
                else:
                    self.status_code = 304
            if (
                self.automatically_set_content_length
                and "content-length" not in self.headers
            ):
                length = self.calculate_content_length()
                if length is not None:
                    self.headers["Content-Length"] = str(length)
        return self

    def add_etag(self, overwrite: bool = False, weak: bool = False) -> None:
        """Add an etag for the current response if there is none yet.

        .. versionchanged:: 2.0
            SHA-1 is used to generate the value. MD5 may not be
            available in some environments.
        """
        if overwrite or "etag" not in self.headers:
            self.set_etag(generate_etag(self.get_data()), weak)


class ResponseStream:
    """A file descriptor like object used by :meth:`Response.stream` to
    represent the body of the stream. It directly pushes into the
    response iterable of the response object.
    """

    mode = "wb+"

    def __init__(self, response: Response):
        self.response = response
        self.closed = False

    def write(self, value: bytes) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        self.response._ensure_sequence(mutable=True)
        self.response.response.append(value)  # type: ignore
        self.response.headers.pop("Content-Length", None)
        return len(value)

    def writelines(self, seq: t.Iterable[bytes]) -> None:
        for item in seq:
            self.write(item)

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def isatty(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return False

    def tell(self) -> int:
        self.response._ensure_sequence()
        return sum(map(len, self.response.response))

    @property
    def encoding(self) -> str:
        return "utf-8"
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import functools
import json
import typing as t
from io import BytesIO

from .._internal import _wsgi_decoding_dance
from ..datastructures import CombinedMultiDict
from ..datastructures import EnvironHeaders
from ..datastructures import FileStorage
from ..datastructures import ImmutableMultiDict
from ..datastructures import iter_multi_items
from ..datastructures import MultiDict
from ..exceptions import BadRequest
from ..exceptions import UnsupportedMediaType
from ..formparser import default_stream_factory
from ..formparser import FormDataParser
from ..sansio.request import Request as _SansIORequest
from ..utils import cached_property
from ..utils import environ_property
from ..wsgi import _get_server
from ..wsgi import get_input_stream

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class Request(_SansIORequest):
    """Represents an incoming WSGI HTTP request, with headers and body
    taken from the WSGI environment. Has properties and methods for
    using the functionality defined by various HTTP specs. The data in
    requests object is read-only.

    Text data is assumed to use UTF-8 encoding, which should be true for
    the vast majority of modern clients. Using an encoding set by the
    client is unsafe in Python due to extra encodings it provides, such
    as ``zip``. To change the assumed encoding, subclass and replace
    :attr:`charset`.

    :param environ: The WSGI environ is generated by the WSGI server and
        contains information about the server configuration and client
        request.
    :param populate_request: Add this request object to the WSGI environ
        as ``environ['werkzeug.request']``. Can be useful when
        debugging.
    :param shallow: Makes reading from :attr:`stream` (and any method
        that would read from it) raise a :exc:`RuntimeError`. Useful to
        prevent consuming the form data in middleware, which would make
        it unavailable to the final application.

    .. versionchanged:: 3.0
        The ``charset``, ``url_charset``, and ``encoding_errors`` parameters
        were removed.

    .. versionchanged:: 2.1
        Old ``BaseRequest`` and mixin classes were removed.

    .. versionchanged:: 2.1
        Remove the ``disable_data_descriptor`` attribute.

    .. versionchanged:: 2.0
        Combine ``BaseRequest`` and mixins into a single ``Request``
        class.

    .. versionchanged:: 0.5
        Read-only mode is enforced with immutable classes for all data.
    """

    max_content_length: int | None = None
    """Stop reading data after this many bytes and raise
    :exc:`.RequestEntityTooLarge`. This amount of memory, plus additional object
    overhead, could be used during one request.

    It's better to configure this in the WSGI server or HTTP server, rather than
    the WSGI application, which is why it's not set by default. Not setting it
    anywhere would be an issue with that application, not with Werkzeug.

    This is applied to :attr:`stream` and anything that reads it such as
    :attr:`data` and :meth:`parse_form_data`.

    .. versionadded:: 0.5
    """

    max_form_memory_size: int | None = 500_000
    """Stop parsing ``multipart/form-data`` if any non-file part is larger than
    this number of bytes, and raise :exc:`.RequestEntityTooLarge`. File parts
    can be larger, they are moved to disk at this limit. The default is 500kB.
    This is an additional check, it does not replace :attr:`max_content_length`.

    .. versionchanged:: 3.1.9
        This is not applied to ``application/x-www-form-urlencoded``.

    .. versionchanged:: 3.1
        Defaults to 500kb instead of unlimited.

    .. versionadded:: 0.5
    """

    max_form_parts = 1000
    """Stop parsing ``multipart/form-data`` if more than this number of parts
    are received, and raise :exc:`.RequestEntityTooLarge`. This is useful to
    stop a very large number of very small fields, especially files. The
    default is 1000. This is an additional check, it does not replace
    :attr:`max_content_length`.

    .. versionadded:: 2.2.3
    """

    #: The form data parser that should be used.  Can be replaced to customize
    #: the form date parsing.
    form_data_parser_class: type[FormDataParser] = FormDataParser

    #: The WSGI environment containing HTTP headers and information from
    #: the WSGI server.
    environ: WSGIEnvironment

    #: Set when creating the request object. If ``True``, reading from
    #: the request body will cause a ``RuntimeException``. Useful to
    #: prevent modifying the stream from middleware.
    shallow: bool

    def __init__(
        self,
        environ: WSGIEnvironment,
        populate_request: bool = True,
        shallow: bool = False,
    ) -> None:
        super().__init__(
            method=environ.get("REQUEST_METHOD", "GET"),
            scheme=environ.get("wsgi.url_scheme", "http"),
            server=_get_server(environ),
            root_path=_wsgi_decoding_dance(environ.get("SCRIPT_NAME") or ""),
            path=_wsgi_decoding_dance(environ.get("PATH_INFO") or ""),
            query_string=environ.get("QUERY_STRING", "").encode("latin1"),
            headers=EnvironHeaders(environ),
            remote_addr=environ.get("REMOTE_ADDR"),
        )
        self.environ = environ
        self.shallow = shallow

        if populate_request and not shallow:
            self.environ["werkzeug.request"] = self

    @classmethod
    def from_values(cls, *args: t.Any, **kwargs: t.Any) -> Request:
        """Create a new request object based on the values provided.  If
        environ is given missing values are filled from there.  This method is
        useful for small scripts when you need to simulate a request from an URL.
        Do not use this method for unittesting, there is a full featured client
        object (:class:`Client`) that allows to create multipart requests,
        support for cookies etc.

        This accepts the same options as the
        :class:`~werkzeug.test.EnvironBuilder`.

        .. versionchanged:: 0.5
           This method now accepts the same arguments as
           :class:`~werkzeug.test.EnvironBuilder`.  Because of this the
           `environ` parameter is now called `environ_overrides`.

        :return: request object
        """
        from ..test import EnvironBuilder

        builder = EnvironBuilder(*args, **kwargs)
        try:
            return builder.get_request(cls)
        finally:
            builder.close()

    @classmethod
    def application(cls, f: t.Callable[[Request], WSGIApplication]) -> WSGIApplication:
        """Decorate a function as responder that accepts the request as
        the last argument.  This works like the :func:`responder`
        decorator but the function is passed the request object as the
        last argument and the request object will be closed
        automatically::

            @Request.application
            def my_wsgi_app(request):
                return Response('Hello World!')

        As of Werkzeug 0.14 HTTP exceptions are automatically caught and
        converted to responses instead of failing.

        :param f: the WSGI callable to decorate
        :return: a new WSGI callable
        """
        #: return a callable that wraps the -2nd argument with the request
        #: and calls the function with all the arguments up to that one and
        #: the request.  The return value is then called with the latest
        #: two arguments.  This makes it possible to use this decorator for
        #: both standalone WSGI functions as well as bound methods and
        #: partially applied functions.
        from ..exceptions import HTTPException

        @functools.wraps(f)
        def application(*args: t.Any) -> cabc.Iterable[bytes]:
            request = cls(args[-2])
            with request:
                try:
                    resp = f(*args[:-2] + (request,))
                except HTTPException as e:
                    resp = t.cast("WSGIApplication", e.get_response(args[-2]))
                return resp(*args[-2:])

        return t.cast("WSGIApplication", application)

    def _get_file_stream(
        self,
        total_content_length: int | None,
        content_type: str | None,
        filename: str | None = None,
        content_length: int | None = None,
    ) -> t.IO[bytes]:
        """Called to get a stream for the file upload.

        This must provide a file-like class with `read()`, `readline()`
        and `seek()` methods that is both writeable and readable.

        The default implementation returns a temporary file if the total
        content length is higher than 500KB.  Because many browsers do not
        provide a content length for the files only the total content
        length matters.

        :param total_content_length: the total content length of all the
                                     data in the request combined.  This value
                                     is guaranteed to be there.
        :param content_type: the mimetype of the uploaded file.
        :param filename: the filename of the uploaded file.  May be `None`.
        :param content_length: the length of this file.  This value is usually
                               not provided because webbrowsers do not provide
                               this value.
        """
        return default_stream_factory(
            total_content_length=total_content_length,
            filename=filename,
            content_type=content_type,
            content_length=content_length,
        )

    @property
    def want_form_data_parsed(self) -> bool:
        """``True`` if the request method carries content. By default
        this is true if a ``Content-Type`` is sent.

        .. versionadded:: 0.8
        """
        return bool(self.environ.get("CONTENT_TYPE"))

    def make_form_data_parser(self) -> FormDataParser:
        """Creates the form data parser. Instantiates the
        :attr:`form_data_parser_class` with some parameters.

        .. versionadded:: 0.8
        """
        return self.form_data_parser_class(
            stream_factory=self._get_file_stream,
            max_form_memory_size=self.max_form_memory_size,
            max_content_length=self.max_content_length,
            max_form_parts=self.max_form_parts,
            cls=self.parameter_storage_class,
        )

    def _load_form_data(self) -> None:
        """Method used internally to retrieve submitted data.  After calling
        this sets `form` and `files` on the request object to multi dicts
        filled with the incoming form data.  As a matter of fact the input
        stream will be empty afterwards.  You can also call this method to
        force the parsing of the form data.

        .. versionadded:: 0.8
        """
        # abort early if we have already consumed the stream
        if "form" in self.__dict__:
            return

        if self.want_form_data_parsed:
            parser = self.make_form_data_parser()
            data = parser.parse(
                self._get_stream_for_parsing(),
                self.mimetype,
                self.content_length,
                self.mimetype_params,
            )
        else:
            data = (
                self.stream,
                self.parameter_storage_class(),
                self.parameter_storage_class(),
            )

        # inject the values into the instance dict so that we bypass
        # our cached_property non-data descriptor.
        d = self.__dict__
        d["stream"], d["form"], d["files"] = data

    def _get_stream_for_parsing(self) -> t.IO[bytes]:
        """This is the same as accessing :attr:`stream` with the difference
        that if it finds cached data from calling :meth:`get_data` first it
        will create a new stream out of the cached data.

        .. versionadded:: 0.9.3
        """
        cached_data = getattr(self, "_cached_data", None)
        if cached_data is not None:
            return BytesIO(cached_data)
        return self.stream

    def close(self) -> None:
        """Closes associated resources of this request object.  This
        closes all file handles explicitly.  You can also use the request
        object in a with statement which will automatically close it.

        .. versionadded:: 0.9
        """
        files = self.__dict__.get("files")
        for _key, value in iter_multi_items(files or ()):
            value.close()

    def __enter__(self) -> Request:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:  # type: ignore
        self.close()

    @cached_property
    def stream(self) -> t.IO[bytes]:
        """The WSGI input stream, with safety checks. This stream can only be consumed
        once.

        Use :meth:`get_data` to get the full data as bytes or text. The :attr:`data`
        attribute will contain the full bytes only if they do not represent form data.
        The :attr:`form` attribute will contain the parsed form data in that case.

        Unlike :attr:`input_stream`, this stream guards against infinite streams or
        reading past :attr:`content_length` or :attr:`max_content_length`.

        If ``max_content_length`` is set, it can be enforced on streams if
        ``wsgi.input_terminated`` is set. Otherwise, an empty stream is returned.

        If the limit is reached before the underlying stream is exhausted (such as a
        file that is too large, or an infinite stream), the remaining contents of the
        stream cannot be read safely. Depending on how the server handles this, clients
        may show a "connection reset" failure instead of seeing the 413 response.

        .. versionchanged:: 2.3
            Check ``max_content_length`` preemptively and while reading.

        .. versionchanged:: 0.9
            The stream is always set (but may be consumed) even if form parsing was
            accessed first.
        """
        if self.shallow:
            raise RuntimeError(
                "This request was created with 'shallow=True', reading"
                " from the input stream is disabled."
            )

        return get_input_stream(
            self.environ, max_content_length=self.max_content_length
        )

    input_stream = environ_property[t.IO[bytes]](
        "wsgi.input",
        doc="""The raw WSGI input stream, without any safety checks.

        This is dangerous to use. It does not guard against infinite streams or reading
        past :attr:`content_length` or :attr:`max_content_length`.

        Use :attr:`stream` instead.
        """,
    )

    @cached_property
    def data(self) -> bytes:
        """The raw data read from :attr:`stream`. Will be empty if the request
        represents form data.

        To get the raw data even if it represents form data, use :meth:`get_data`.
        """
        return self.get_data(parse_form_data=True)

    @t.overload
    def get_data(
        self,
        cache: bool = True,
        as_text: t.Literal[False] = False,
        parse_form_data: bool = False,
    ) -> bytes: ...

    @t.overload
    def get_data(
        self,
        cache: bool = True,
        as_text: t.Literal[True] = ...,
        parse_form_data: bool = False,
    ) -> str: ...

    def get_data(
        self, cache: bool = True, as_text: bool = False, parse_form_data: bool = False
    ) -> bytes | str:
        """This reads the buffered incoming data from the client into one
        bytes object.  By default this is cached but that behavior can be
        changed by setting `cache` to `False`.

        Usually it's a bad idea to call this method without checking the
        content length first as a client could send dozens of megabytes or more
        to cause memory problems on the server.

        Note that if the form data was already parsed this method will not
        return anything as form data parsing does not cache the data like
        this method does.  To implicitly invoke form data parsing function
        set `parse_form_data` to `True`.  When this is done the return value
        of this method will be an empty string if the form parser handles
        the data.  This generally is not necessary as if the whole data is
        cached (which is the default) the form parser will used the cached
        data to parse the form data.  Please be generally aware of checking
        the content length first in any case before calling this method
        to avoid exhausting server memory.

        If `as_text` is set to `True` the return value will be a decoded
        string.

        .. versionadded:: 0.9
        """
        rv = getattr(self, "_cached_data", None)
        if rv is None:
            if parse_form_data:
                self._load_form_data()
            rv = self.stream.read()
            if cache:
                self._cached_data = rv
        if as_text:
            rv = rv.decode(errors="replace")
        return rv

    @cached_property
    def form(self) -> ImmutableMultiDict[str, str]:
        """The form parameters.  By default an
        :class:`~werkzeug.datastructures.ImmutableMultiDict`
        is returned from this function.  This can be changed by setting
        :attr:`parameter_storage_class` to a different type.  This might
        be necessary if the order of the form data is important.

        Please keep in mind that file uploads will not end up here, but instead
        in the :attr:`files` attribute.

        .. versionchanged:: 0.9

            Previous to Werkzeug 0.9 this would only contain form data for POST
            and PUT requests.
        """
        self._load_form_data()
        return self.form

    @cached_property
    def values(self) -> CombinedMultiDict[str, str]:
        """A :class:`werkzeug.datastructures.CombinedMultiDict` that
        combines :attr:`args` and :attr:`form`.

        For GET requests, only ``args`` are present, not ``form``.

        .. versionchanged:: 2.0
            For GET requests, only ``args`` are present, not ``form``.
        """
        sources = [self.args]

        if self.method != "GET":
            # GET requests can have a body, and some caching proxies
            # might not treat that differently than a normal GET
            # request, allowing form data to "invisibly" affect the
            # cache without indication in the query string / URL.
            sources.append(self.form)

        args = []

        for d in sources:
            if not isinstance(d, MultiDict):
                d = MultiDict(d)

            args.append(d)

        return CombinedMultiDict(args)

    @cached_property
    def files(self) -> ImmutableMultiDict[str, FileStorage]:
        """:class:`~werkzeug.datastructures.MultiDict` object containing
        all uploaded files.  Each key in :attr:`files` is the name from the
        ``<input type="file" name="">``.  Each value in :attr:`files` is a
        Werkzeug :class:`~werkzeug.datastructures.FileStorage` object.

        It basically behaves like a standard file object you know from Python,
        with the difference that it also has a
        :meth:`~werkzeug.datastructures.FileStorage.save` function that can
        store the file on the filesystem.

        Note that :attr:`files` will only contain data if the request method was
        POST, PUT or PATCH and the ``<form>`` that posted to the request had
        ``enctype="multipart/form-data"``.  It will be empty otherwise.

        See the :class:`~werkzeug.datastructures.MultiDict` /
        :class:`~werkzeug.datastructures.FileStorage` documentation for
        more details about the used data structure.
        """
        self._load_form_data()
        return self.files

    @property
    def script_root(self) -> str:
        """Alias for :attr:`self.root_path`. ``environ["SCRIPT_NAME"]``
        without a trailing slash.
        """
        return self.root_path

    @cached_property
    def url_root(self) -> str:
        """Alias for :attr:`root_url`. The URL with scheme, host, and
        root path. For example, ``https://example.com/app/``.
        """
        return self.root_url

    remote_user = environ_property[str](
        "REMOTE_USER",
        doc="""If the server supports user authentication, and the
        script is protected, this attribute contains the username the
        user has authenticated as.""",
    )
    is_multithread = environ_property[bool](
        "wsgi.multithread",
        doc="""boolean that is `True` if the application is served by a
        multithreaded WSGI server.""",
    )
    is_multiprocess = environ_property[bool](
        "wsgi.multiprocess",
        doc="""boolean that is `True` if the application is served by a
        WSGI server that spawns multiple processes.""",
    )
    is_run_once = environ_property[bool](
        "wsgi.run_once",
        doc="""boolean that is `True` if the application will be
        executed only once in a process lifetime.  This is the case for
        CGI for example, but it's not guaranteed that the execution only
        happens one time.""",
    )

    # JSON

    #: A module or other object that has ``dumps`` and ``loads``
    #: functions that match the API of the built-in :mod:`json` module.
    json_module = json

    @property
    def json(self) -> t.Any:
        """The parsed JSON data if :attr:`mimetype` indicates JSON
        (:mimetype:`application/json`, see :attr:`is_json`).

        Calls :meth:`get_json` with default arguments.

        If the request content type is not ``application/json``, this
        will raise a 415 Unsupported Media Type error.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.

        .. versionchanged:: 2.1
            Raise a 400 error if the content type is incorrect.
        """
        return self.get_json()

    # Cached values for ``(silent=False, silent=True)``. Initialized
    # with sentinel values.
    _cached_json: tuple[t.Any, t.Any] = (Ellipsis, Ellipsis)

    @t.overload
    def get_json(
        self, force: bool = ..., silent: t.Literal[False] = ..., cache: bool = ...
    ) -> t.Any: ...

    @t.overload
    def get_json(
        self, force: bool = ..., silent: bool = ..., cache: bool = ...
    ) -> t.Any | None: ...

    def get_json(
        self, force: bool = False, silent: bool = False, cache: bool = True
    ) -> t.Any | None:
        """Parse :attr:`data` as JSON.

        If the mimetype does not indicate JSON
        (:mimetype:`application/json`, see :attr:`is_json`), or parsing
        fails, :meth:`on_json_loading_failed` is called and
        its return value is used as the return value. By default this
        raises a 415 Unsupported Media Type resp.

        :param force: Ignore the mimetype and always try to parse JSON.
        :param silent: Silence mimetype and parsing errors, and
            return ``None`` instead.
        :param cache: Store the parsed JSON to return for subsequent
            calls.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.

        .. versionchanged:: 2.1
            Raise a 400 error if the content type is incorrect.
        """
        if cache and self._cached_json[silent] is not Ellipsis:
            return self._cached_json[silent]

        if not (force or self.is_json):
            if not silent:
                return self.on_json_loading_failed(None)
            else:
                return None

        data = self.get_data(cache=cache)

        try:
            rv = self.json_module.loads(data)
        except ValueError as e:
            if silent:
                rv = None

                if cache:
                    normal_rv, _ = self._cached_json
                    self._cached_json = (normal_rv, rv)
            else:
                rv = self.on_json_loading_failed(e)

                if cache:
                    _, silent_rv = self._cached_json
                    self._cached_json = (rv, silent_rv)
        else:
            if cache:
                self._cached_json = (rv, rv)

        return rv

    def on_json_loading_failed(self, e: ValueError | None) -> t.Any:
        """Called if :meth:`get_json` fails and isn't silenced.

        If this method returns a value, it is used as the return value
        for :meth:`get_json`. The default implementation raises
        :exc:`~werkzeug.exceptions.BadRequest`.

        :param e: If parsing failed, this is the exception. It will be
            ``None`` if the content type wasn't ``application/json``.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.
        """
        if e is not None:
            raise BadRequest(f"Failed to decode JSON object: {e}")

        raise UnsupportedMediaType(
            "Did not attempt to load JSON data because the request"
            " Content-Type was not 'application/json'."
        )
<|endoftext|># This file is generated by SciPy's build process
# It contains system_info results at the time of building this package.
from enum import Enum

__all__ = ["show"]
_built_with_meson = True


class DisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def _cleanup(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return { k: _cleanup(v) for k, v in d.items() if v != '' and _cleanup(v) != '' }
    else:
        return d


CONFIG = _cleanup(
    {
        "Compilers": {
            "c": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.2.1",
                "commands": r"cc",
                "args": r"",
                "linker args": r"",
            },
            "cython": {
                "name": r"cython",
                "linker": r"cython",
                "version": r"3.0.12",
                "commands": r"cython",
                "args": r"",
                "linker args": r"",
            },
            "c++": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.2.1",
                "commands": r"c++",
                "args": r"",
                "linker args": r"",
            },
            "fortran": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.2.1",
                "commands": r"gfortran",
                "args": r"",
                "linker args": r"",
            },
            "pythran": {
                "version": r"0.17.0",
                "include directory": r"../../tmp/pip-build-env-jujiy7q_/overlay/lib/python3.10/site-packages/pythran"
            },
        },
        "Machine Information": {
            "host": {
                "cpu": r"x86_64",
                "family": r"x86_64",
                "endian": r"little",
                "system": r"linux",
            },
            "build": {
                "cpu": r"x86_64",
                "family": r"x86_64",
                "endian": r"little",
                "system": r"linux",
            },
            "cross-compiled": bool("False".lower().replace('false', '')),
        },
        "Build Dependencies": {
            "blas": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.28",
                "detection method": "pkgconfig",
                "include directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas32/include",
                "lib directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas32/lib",
                "openblas configuration": r"OpenBLAS 0.3.28 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
                "pc file directory": r"/project",
            },
            "lapack": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.28",
                "detection method": "pkgconfig",
                "include directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas32/include",
                "lib directory": r"/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas32/lib",
                "openblas configuration": r"OpenBLAS 0.3.28 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
                "pc file directory": r"/project",
            },
            "pybind11": {
                "name": "pybind11",
                "version": "2.13.6",
                "detection method": "config-tool",
                "include directory": r"unknown",
            },
        },
        "Python Information": {
            "path": r"/opt/python/cp310-cp310/bin/python",
            "version": "3.10",
        },
    }
)


def _check_pyyaml():
    import yaml

    return yaml


def show(mode=DisplayModes.stdout.value):
    """
    Show libraries and system information on which SciPy was built
    and is being used

    Parameters
    ----------
    mode : {`'stdout'`, `'dicts'`}, optional.
        Indicates how to display the config information.
        `'stdout'` prints to console, `'dicts'` returns a dictionary
        of the configuration.

    Returns
    -------
    out : {`dict`, `None`}
        If mode is `'dicts'`, a dict is returned, else None

    Notes
    -----
    1. The `'stdout'` mode will give more readable
       output if ``pyyaml`` is installed

    """
    if mode == DisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = _check_pyyaml()

            print(yaml.dump(CONFIG))
        except ModuleNotFoundError:
            import warnings
            import json

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(CONFIG, indent=2))
    elif mode == DisplayModes.dicts.value:
        return CONFIG
    else:
        raise AttributeError(
            f"Invalid `mode`, use one of: {', '.join([e.value for e in DisplayModes])}"
        )
<|endoftext|>"""
SciPy: A scientific computing package for Python
================================================

Documentation is available in the docstrings and
online at https://docs.scipy.org.

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import scipy.cluster``.

::

 cluster                      --- Vector Quantization / Kmeans
 constants                    --- Physical and mathematical constants and units
 datasets                     --- Dataset methods
 differentiate                --- Finite difference differentiation tools
 fft                          --- Discrete Fourier transforms
 fftpack                      --- Legacy discrete Fourier transforms
 integrate                    --- Integration routines
 interpolate                  --- Interpolation Tools
 io                           --- Data input and output
 linalg                       --- Linear algebra routines
 ndimage                      --- N-D image package
 odr                          --- Orthogonal Distance Regression
 optimize                     --- Optimization Tools
 signal                       --- Signal Processing Tools
 sparse                       --- Sparse Matrices
 spatial                      --- Spatial data structures and algorithms
 special                      --- Special functions
 stats                        --- Statistical Functions

Public API in the main SciPy namespace
--------------------------------------
::

 __version__       --- SciPy version string
 LowLevelCallable  --- Low-level callback function
 show_config       --- Show scipy build configuration
 test              --- Run scipy unittests

"""

import importlib as _importlib

from numpy import __version__ as __numpy_version__


try:
    from scipy.__config__ import show as show_config
except ImportError as e:
    msg = """Error importing SciPy: you cannot import SciPy while
    being in scipy source directory; please exit the SciPy source
    tree first and relaunch your Python interpreter."""
    raise ImportError(msg) from e


from scipy.version import version as __version__


# Allow distributors to run custom init code
from . import _distributor_init
del _distributor_init


from scipy._lib import _pep440
# In maintenance branch, change to np_maxversion N+3 if numpy is at N
np_minversion = '1.23.5'
np_maxversion = '2.5.0'
if (_pep440.parse(__numpy_version__) < _pep440.Version(np_minversion) or
        _pep440.parse(__numpy_version__) >= _pep440.Version(np_maxversion)):
    import warnings
    warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
                  f" is required for this version of SciPy (detected "
                  f"version {__numpy_version__})",
                  UserWarning, stacklevel=2)
del _pep440


# This is the first import of an extension module within SciPy. If there's
# a general issue with the install, such that extension modules are missing
# or cannot be imported, this is where we'll get a failure - so give an
# informative error message.
try:
    from scipy._lib._ccallback import LowLevelCallable
except ImportError as e:
    msg = "The `scipy` install you are using seems to be broken, " + \
          "(extension modules cannot be imported), " + \
          "please try reinstalling."
    raise ImportError(msg) from e


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester


submodules = [
    'cluster',
    'constants',
    'datasets',
    'differentiate',
    'fft',
    'fftpack',
    'integrate',
    'interpolate',
    'io',
    'linalg',
    'ndimage',
    'odr',
    'optimize',
    'signal',
    'sparse',
    'spatial',
    'special',
    'stats'
]

__all__ = submodules + [
    'LowLevelCallable',
    'test',
    'show_config',
    '__version__',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'scipy.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'scipy' has no attribute '{name}'"
            )
<|endoftext|>"""
fitpack --- curve and surface fitting with splines

fitpack is based on a collection of Fortran routines DIERCKX
by P. Dierckx (see http://www.netlib.org/dierckx/) transformed
to double routines by Pearu Peterson.
"""
# Created by Pearu Peterson, June,August 2003
__all__ = [
    'UnivariateSpline',
    'InterpolatedUnivariateSpline',
    'LSQUnivariateSpline',
    'BivariateSpline',
    'LSQBivariateSpline',
    'SmoothBivariateSpline',
    'LSQSphereBivariateSpline',
    'SmoothSphereBivariateSpline',
    'RectBivariateSpline',
    'RectSphereBivariateSpline']


import warnings
from threading import Lock

from numpy import zeros, concatenate, ravel, diff, array
import numpy as np

from . import _fitpack_impl
from . import _dfitpack as dfitpack


dfitpack_int = dfitpack.types.intvar.dtype
FITPACK_LOCK = Lock()


# ############### Univariate spline ####################

_curfit_messages = {1: """
The required storage space exceeds the available storage space, as
specified by the parameter nest: nest too small. If nest is already
large (say nest > m/2), it may also indicate that s is too small.
The approximation returned is the weighted least-squares spline
according to the knots t[0],t[1],...,t[n-1]. (n=nest) the parameter fp
gives the corresponding weighted sum of squared residuals (fp>s).
""",
                    2: """
A theoretically impossible result was found during the iteration
process for finding a smoothing spline with fp = s: s too small.
There is an approximation returned but the corresponding weighted sum
of squared residuals does not satisfy the condition abs(fp-s)/s < tol.""",
                    3: """
The maximal number of iterations maxit (set to 20 by the program)
allowed for finding a smoothing spline with fp=s has been reached: s
too small.
There is an approximation returned but the corresponding weighted sum
of squared residuals does not satisfy the condition abs(fp-s)/s < tol.""",
                    10: """
Error on entry, no approximation returned. The following conditions
must hold:
xb<=x[0]<x[1]<...<x[m-1]<=xe, w[i]>0, i=0..m-1
if iopt=-1:
  xb<t[k+1]<t[k+2]<...<t[n-k-2]<xe"""
                    }


# UnivariateSpline, ext parameter can be an int or a string
_extrap_modes = {0: 0, 'extrapolate': 0,
                 1: 1, 'zeros': 1,
                 2: 2, 'raise': 2,
                 3: 3, 'const': 3}


class UnivariateSpline:
    """
    1-D smoothing spline fit to a given set of data points.

    .. legacy:: class

        Specifically, we recommend using `make_splrep` instead.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
    specifies the number of knots by specifying a smoothing condition.

    Parameters
    ----------
    x : (N,) array_like
        1-D array of independent input data. Must be increasing;
        must be strictly increasing if `s` is 0.
    y : (N,) array_like
        1-D array of dependent input data, of the same length as `x`.
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If `w` is None,
        weights are all 1. Default is None.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        `bbox` is None, ``bbox=[x[0], x[-1]]``. Default is None.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        ``k = 3`` is a cubic spline. Default is 3.
    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        However, because of numerical issues, the actual condition is::

            abs(sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) - s) < 0.001 * s

        If `s` is None, `s` will be set as `len(w)` for a smoothing spline
        that uses all data points.
        If 0, spline will interpolate through all data points. This is
        equivalent to `InterpolatedUnivariateSpline`.
        Default is None.
        The user can use the `s` to control the tradeoff between closeness
        and smoothness of fit. Larger `s` means more smoothing while smaller
        values of `s` indicate less smoothing.
        Recommended values of `s` depend on the weights, `w`. If the weights
        represent the inverse of the standard-deviation of `y`, then a good
        `s` value should be found in the range (m-sqrt(2*m),m+sqrt(2*m))
        where m is the number of datapoints in `x`, `y`, and `w`. This means
        ``s = len(w)`` should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``y[i]``.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 or 'const', return the boundary value.

        Default is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    **NaN handling**: If the input arrays contain ``nan`` values, the result
    is not useful, since the underlying spline fitting routines cannot deal
    with ``nan``. A workaround is to use zero weights for not-a-number
    data points:

    >>> import numpy as np
    >>> from scipy.interpolate import UnivariateSpline
    >>> x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> spl = UnivariateSpline(x, y, w=~w)

    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)

    References
    ----------
    Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:

    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
       integration of experimental data using spline functions",
       J.Comp.Appl.Maths 1 (1975) 165-184.
    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
       1286-1304.
    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import UnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> plt.plot(x, y, 'ro', ms=5)

    Use the default value for the smoothing parameter:

    >>> spl = UnivariateSpline(x, y)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3)

    Manually change the amount of smoothing:

    >>> spl.set_smoothing_factor(0.5)
    >>> plt.plot(xs, spl(xs), 'b', lw=3)
    >>> plt.show()

    """

    def __init__(self, x, y, w=None, bbox=[None]*2, k=3, s=None,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, s, ext,
                                                      check_finite)

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        with FITPACK_LOCK:
            data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                    xe=bbox[1], s=s)
        if data[-1] == 1:
            # nest too small, setting to maximum bound
            data = self._reset_nest(data)
        self._data = data
        self._reset_class()

    @staticmethod
    def validate_input(x, y, w, bbox, k, s, ext, check_finite):
        x, y, bbox = np.asarray(x), np.asarray(y), np.asarray(bbox)
        if w is not None:
            w = np.asarray(w)
        if check_finite:
            w_finite = np.isfinite(w).all() if w is not None else True
            if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not w_finite):
                raise ValueError("x and y array must not contain "
                                 "NaNs or infs.")
        if s is None or s > 0:
            if not np.all(diff(x) >= 0.0):
                raise ValueError("x must be increasing if s > 0")
        else:
            if not np.all(diff(x) > 0.0):
                raise ValueError("x must be strictly increasing if s = 0")
        if x.size != y.size:
            raise ValueError("x and y should have a same length")
        elif w is not None and not x.size == y.size == w.size:
            raise ValueError("x, y, and w should have a same length")
        elif bbox.shape != (2,):
            raise ValueError("bbox shape should be (2,)")
        elif not (1 <= k <= 5):
            raise ValueError("k should be 1 <= k <= 5")
        elif s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        try:
            ext = _extrap_modes[ext]
        except KeyError as e:
            raise ValueError(f"Unknown extrapolation mode {ext}.") from e

        return x, y, w, bbox, ext

    @classmethod
    def _from_tck(cls, tck, ext=0):
        """Construct a spline object from given tck"""
        self = cls.__new__(cls)
        t, c, k = tck
        self._eval_args = tck
        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = (None, None, None, None, None, k, None, len(t), t,
                      c, None, None, None, None)
        self.ext = ext
        return self

    def _reset_class(self):
        data = self._data
        n, t, c, k, ier = data[7], data[8], data[9], data[5], data[-1]
        self._eval_args = t[:n], c[:n], k
        if ier == 0:
            # the spline returned has a residual sum of squares fp
            # such that abs(fp-s)/s <= tol with tol a relative
            # tolerance set to 0.001 by the program
            pass
        elif ier == -1:
            # the spline returned is an interpolating spline
            self._set_class(InterpolatedUnivariateSpline)
        elif ier == -2:
            # the spline returned is the weighted least-squares
            # polynomial of degree k. In this extreme case fp gives
            # the upper bound fp0 for the smoothing factor s.
            self._set_class(LSQUnivariateSpline)
        else:
            # error
            if ier == 1:
                self._set_class(LSQUnivariateSpline)
            message = _curfit_messages.get(ier, f'ier={ier}')
            warnings.warn(message, stacklevel=3)

    def _set_class(self, cls):
        self._spline_class = cls
        if self.__class__ in (UnivariateSpline, InterpolatedUnivariateSpline,
                              LSQUnivariateSpline):
            self.__class__ = cls
        else:
            # It's an unknown subclass -- don't change class. cf. #731
            pass

    def _reset_nest(self, data, nest=None):
        n = data[10]
        if nest is None:
            k, m = data[5], len(data[0])
            nest = m+k+1  # this is the maximum bound for nest
        else:
            if not n <= nest:
                raise ValueError("`nest` can only be increased")
        t, c, fpint, nrdata = (np.resize(data[j], nest) for j in
                               [8, 9, 11, 12])

        args = data[:8] + (t, c, n, fpint, nrdata, data[13])
        with FITPACK_LOCK:
            data = dfitpack.fpcurf1(*args)
        return data

    def set_smoothing_factor(self, s):
        """ Continue spline computation with the given smoothing
        factor s and with the knots found at the last call.

        This routine modifies the spline in place.

        """
        data = self._data
        if data[6] == -1:
            warnings.warn('smoothing factor unchanged for'
                          'LSQ spline with fixed knots',
                          stacklevel=2)
            return
        args = data[:6] + (s,) + data[7:]
        with FITPACK_LOCK:
            data = dfitpack.fpcurf1(*args)
        if data[-1] == 1:
            # nest too small, setting to maximum bound
            data = self._reset_nest(data)
        self._data = data
        self._reset_class()

    def __call__(self, x, nu=0, ext=None):
        """
        Evaluate spline (or its nu-th derivative) at positions x.

        Parameters
        ----------
        x : array_like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: `x` can be unordered but the
            evaluation is more efficient if `x` is (partially) ordered.
        nu  : int
            The order of derivative of the spline to compute.
        ext : int
            Controls the value returned for elements of `x` not in the
            interval defined by the knot sequence.

            * if ext=0 or 'extrapolate', return the extrapolated value.
            * if ext=1 or 'zeros', return 0
            * if ext=2 or 'raise', raise a ValueError
            * if ext=3 or 'const', return the boundary value.

            The default value is 0, passed from the initialization of
            UnivariateSpline.

        """
        x = np.asarray(x)
        # empty input yields empty output
        if x.size == 0:
            return array([])
        if ext is None:
            ext = self.ext
        else:
            try:
                ext = _extrap_modes[ext]
            except KeyError as e:
                raise ValueError(f"Unknown extrapolation mode {ext}.") from e
        with FITPACK_LOCK:
            return _fitpack_impl.splev(x, self._eval_args, der=nu, ext=ext)

    def get_knots(self):
        """ Return positions of interior knots of the spline.

        Internally, the knot vector contains ``2*k`` additional boundary knots.
        """
        data = self._data
        k, n = data[5], data[7]
        return data[8][k:n-k]

    def get_coeffs(self):
        """Return spline coefficients."""
        data = self._data
        k, n = data[5], data[7]
        return data[9][:n-k-1]

    def get_residual(self):
        """Return weighted sum of squared residuals of the spline approximation.

           This is equivalent to::

                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)

        """
        return self._data[10]

    def integral(self, a, b):
        """ Return definite integral of the spline between two given points.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.

        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0

        which agrees with :math:`\\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.

        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:

        >>> spl.integral(-1, 4)
        9.0
        >>> spl.integral(-1, 0)
        0.0

        """
        with FITPACK_LOCK:
            return _fitpack_impl.splint(a, b, self._eval_args)

    def derivatives(self, x):
        """ Return all derivatives of the spline at the point x.

        Parameters
        ----------
        x : float
            The point to evaluate the derivatives at.

        Returns
        -------
        der : ndarray, shape(k+1,)
            Derivatives of the orders 0 to k.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)
        array([2.25, 3.0, 2.0, 0])

        """
        with FITPACK_LOCK:
            return _fitpack_impl.spalde(x, self._eval_args)

    def roots(self):
        """ Return the zeros of the spline.

        Notes
        -----
        Restriction: only cubic splines are supported by FITPACK. For non-cubic
        splines, use `PPoly.root` (see below for an example).

        Examples
        --------

        For some data, this method may miss a root. This happens when one of
        the spline knots (which FITPACK places automatically) happens to
        coincide with the true root. A workaround is to convert to `PPoly`,
        which uses a different root-finding algorithm.

        For example,

        >>> x = [1.96, 1.97, 1.98, 1.99, 2.00, 2.01, 2.02, 2.03, 2.04, 2.05]
        >>> y = [-6.365470e-03, -4.790580e-03, -3.204320e-03, -1.607270e-03,
        ...      4.440892e-16,  1.616930e-03,  3.243000e-03,  4.877670e-03,
        ...      6.520430e-03,  8.170770e-03]
        >>> from scipy.interpolate import UnivariateSpline
        >>> spl = UnivariateSpline(x, y, s=0)
        >>> spl.roots()
        array([], dtype=float64)

        Converting to a PPoly object does find the roots at `x=2`:

        >>> from scipy.interpolate import splrep, PPoly
        >>> tck = splrep(x, y, s=0)
        >>> ppoly = PPoly.from_spline(tck)
        >>> ppoly.roots(extrapolate=False)
        array([2.])

        See Also
        --------
        sproot
        PPoly.roots

        """
        k = self._data[5]
        if k == 3:
            t = self._eval_args[0]
            mest = 3 * (len(t) - 7)
            with FITPACK_LOCK:
                return _fitpack_impl.sproot(self._eval_args, mest=mest)
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')

    def derivative(self, n=1):
        """
        Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.

        See Also
        --------
        splder, antiderivative

        Notes
        -----

        .. versionadded:: 0.13.0

        Examples
        --------
        This can be used for finding maxima of a curve:

        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 10, 70)
        >>> y = np.sin(x)
        >>> spl = UnivariateSpline(x, y, k=4, s=0)

        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):

        >>> spl.derivative().roots() / np.pi
        array([ 0.50000001,  1.5       ,  2.49999998])

        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.

        """
        with FITPACK_LOCK:
            tck = _fitpack_impl.splder(self._eval_args, n)
        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext
        return UnivariateSpline._from_tck(tck, ext=ext)

    def antiderivative(self, n=1):
        """
        Construct a new spline representing the antiderivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k+n representing the antiderivative of this
            spline.

        Notes
        -----

        .. versionadded:: 0.13.0

        See Also
        --------
        splantider, derivative

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, np.pi/2, 70)
        >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
        >>> spl = UnivariateSpline(x, y, s=0)

        The derivative is the inverse operation of the antiderivative,
        although some floating point error accumulates:

        >>> spl(1.7), spl.antiderivative().derivative()(1.7)
        (array(2.1565429877197317), array(2.1565429877201865))

        Antiderivative can be used to evaluate definite integrals:

        >>> ispl = spl.antiderivative()
        >>> ispl(np.pi/2) - ispl(0)
        2.2572053588768486

        This is indeed an approximation to the complete elliptic integral
        :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

        >>> from scipy.special import ellipk
        >>> ellipk(0.8)
        2.2572053268208538

        """
        with FITPACK_LOCK:
            tck = _fitpack_impl.splantider(self._eval_args, n)
        return UnivariateSpline._from_tck(tck, self.ext)


class InterpolatedUnivariateSpline(UnivariateSpline):
    """
    1-D interpolating spline for a given set of data points.

    .. legacy:: class

        Specifically, we recommend using `make_interp_spline` instead.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with  `s` = 0.

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) array_like
        input dimension of data points
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all 1.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be ``1 <= k <= 5``. Default is
        ``k = 3``, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQUnivariateSpline :
        a spline for which knots are user-selected
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> spl = InterpolatedUnivariateSpline(x, y)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
    >>> plt.show()

    Notice that the ``spl(x)`` interpolates `y`:

    >>> spl.get_residual()
    0.0

    """

    def __init__(self, x, y, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                            ext, check_finite)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        with FITPACK_LOCK:
            self._data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                          xe=bbox[1], s=0)
        self._reset_class()


_fpchec_error_string = """The input parameters have been rejected by fpchec. \
This means that at least one of the following conditions is violated:

1) k+1 <= n-k-1 <= m
2) t(1) <= t(2) <= ... <= t(k+1)
   t(n-k) <= t(n-k+1) <= ... <= t(n)
3) t(k+1) < t(k+2) < ... < t(n-k)
4) t(k+1) <= x(i) <= t(n-k)
5) The conditions specified by Schoenberg and Whitney must hold
   for at least one subset of data points, i.e., there must be a
   subset of data points y(j) such that
       t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
"""


class LSQUnivariateSpline(UnivariateSpline):
    """
    1-D spline with explicit internal knots.

    .. legacy:: class

        Specifically, we recommend using `make_lsq_spline` instead.


    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `t`
    specifies the internal knots of the spline

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be increasing
    y : (N,) array_like
        Input dimension of data points
    t : (M,) array_like
        interior knots of the spline.  Must be in ascending order and::

            bbox[0] < t[0] < ... < t[-1] < bbox[-1]

    w : (N,) array_like, optional
        weights for spline fitting. Must be positive. If None (default),
        weights are all 1.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox = [x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is `k` = 3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    Raises
    ------
    ValueError
        If the interior knots do not satisfy the Schoenberg-Whitney conditions

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Fit a smoothing spline with a pre-defined internal knots:

    >>> t = [-1, 0, 1]
    >>> spl = LSQUnivariateSpline(x, y, t)

    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3)
    >>> plt.show()

    Check the knot vector:

    >>> spl.get_knots()
    array([-3., -1., 0., 1., 3.])

    Constructing lsq spline using the knots from another spline:

    >>> x = np.arange(10)
    >>> s = UnivariateSpline(x, x, s=0)
    >>> s.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])
    >>> knt = s.get_knots()
    >>> s1 = LSQUnivariateSpline(x, x, knt[1:-1])    # Chop 1st and last knot
    >>> s1.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])

    """

    def __init__(self, x, y, t, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                                      ext, check_finite)
        if not np.all(diff(x) >= 0.0):
            raise ValueError('x must be increasing')

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        xb = bbox[0]
        xe = bbox[1]
        if xb is None:
            xb = x[0]
        if xe is None:
            xe = x[-1]
        t = concatenate(([xb]*(k+1), t, [xe]*(k+1)))
        n = len(t)
        if not np.all(t[k+1:n-k]-t[k:n-k-1] > 0, axis=0):
            raise ValueError('Interior knots t must satisfy '
                             'Schoenberg-Whitney conditions')
        with FITPACK_LOCK:
            if not dfitpack.fpchec(x, t, k) == 0:
                raise ValueError(_fpchec_error_string)
            data = dfitpack.fpcurfm1(x, y, k, t, w=w, xb=xb, xe=xe)
        self._data = data[:-3] + (None, None, data[-1])
        self._reset_class()


# ############### Bivariate spline ####################

class _BivariateSplineBase:
    """ Base class for Bivariate spline s(x,y) interpolation on the rectangle
    [xb,xe] x [yb, ye] calculated from a given set of data points
    (x,y,z).

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    BivariateSpline :
        a base class for bivariate splines.
    SphereBivariateSpline :
        a bivariate spline on a spherical grid
    """

    @classmethod
    def _from_tck(cls, tck):
        """Construct a spline object from given tck and degree"""
        self = cls.__new__(cls)
        if len(tck) != 5:
            raise ValueError("tck should be a 5 element tuple of tx,"
                             " ty, c, kx, ky")
        self.tck = tck[:3]
        self.degrees = tck[3:]
        return self

    def get_residual(self):
        """ Return weighted sum of squared residuals of the spline
        approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
        """
        return self.fp

    def get_knots(self):
        """ Return a tuple (tx,ty) where tx,ty contain knots positions
        of the spline with respect to x-, y-variable, respectively.
        The position of interior and additional knots are given as
        t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
        """
        return self.tck[:2]

    def get_coeffs(self):
        """ Return spline coefficients."""
        return self.tck[2]

    def __call__(self, x, y, dx=0, dy=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        x, y : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points ``(x[i],
            y[i]), i=0, ..., len(x)-1``.  Standard Numpy broadcasting
            is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays x, y. The arrays must be
            sorted to increasing order.

            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dx : int
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int
            Order of y-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to bilinearly interpolate an exponentially decaying
        function in 2 dimensions.

        >>> import numpy as np
        >>> from scipy.interpolate import RectBivariateSpline

        We sample the function on a coarse grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> xarr = np.linspace(-3, 3, 100)
        >>> yarr = np.linspace(-3, 3, 100)
        >>> xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")

        The function to interpolate decays faster along one axis than the other.

        >>> zdata = np.exp(-np.sqrt((xgrid / 2) ** 2 + ygrid**2))

        Next we sample on a finer grid using interpolation (kx=ky=1 for bilinear).

        >>> rbs = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)
        >>> xarr_fine = np.linspace(-3, 3, 200)
        >>> yarr_fine = np.linspace(-3, 3, 200)
        >>> xgrid_fine, ygrid_fine = np.meshgrid(xarr_fine, yarr_fine, indexing="ij")
        >>> zdata_interp = rbs(xgrid_fine, ygrid_fine, grid=False)

        And check that the result agrees with the input by plotting both.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1, aspect="equal")
        >>> ax2 = fig.add_subplot(1, 2, 2, aspect="equal")
        >>> ax1.imshow(zdata)
        >>> ax2.imshow(zdata_interp)
        >>> plt.show()
        """
        x = np.asarray(x)
        y = np.asarray(y)

        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        if grid:
            if x.size == 0 or y.size == 0:
                return np.zeros((x.size, y.size), dtype=self.tck[2].dtype)

            if (x.size >= 2) and (not np.all(np.diff(x) >= 0.0)):
                raise ValueError("x must be strictly increasing when `grid` is True")
            if (y.size >= 2) and (not np.all(np.diff(y) >= 0.0)):
                raise ValueError("y must be strictly increasing when `grid` is True")

            if dx or dy:
                with FITPACK_LOCK:
                    z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError(f"Error code returned by parder: {ier}")
            else:
                with FITPACK_LOCK:
                    z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError(f"Error code returned by bispev: {ier}")
        else:
            # standard Numpy broadcasting
            if x.shape != y.shape:
                x, y = np.broadcast_arrays(x, y)

            shape = x.shape
            x = x.ravel()
            y = y.ravel()

            if x.size == 0 or y.size == 0:
                return np.zeros(shape, dtype=self.tck[2].dtype)

            if dx or dy:
                with FITPACK_LOCK:
                    z, ier = dfitpack.pardeu(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError(f"Error code returned by pardeu: {ier}")
            else:
                with FITPACK_LOCK:
                    z, ier = dfitpack.bispeu(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError(f"Error code returned by bispeu: {ier}")

            z = z.reshape(shape)
        return z

    def partial_derivative(self, dx, dy):
        """Construct a new spline representing a partial derivative of this
        spline.

        Parameters
        ----------
        dx, dy : int
            Orders of the derivative in x and y respectively. They must be
            non-negative integers and less than the respective degree of the
            original spline (self) in that direction (``kx``, ``ky``).

        Returns
        -------
        spline :
            A new spline of degrees (``kx - dx``, ``ky - dy``) representing the
            derivative of this spline.

        Notes
        -----

        .. versionadded:: 1.9.0

        """
        if dx == 0 and dy == 0:
            return self
        else:
            kx, ky = self.degrees
            if not (dx >= 0 and dy >= 0):
                raise ValueError("order of derivative must be positive or"
                                 " zero")
            if not (dx < kx and dy < ky):
                raise ValueError("order of derivative must be less than"
                                 " degree of spline")
            tx, ty, c = self.tck[:3]
            with FITPACK_LOCK:
                newc, ier = dfitpack.pardtc(tx, ty, c, kx, ky, dx, dy)
            if ier != 0:
                # This should not happen under normal conditions.
                raise ValueError("Unexpected error code returned by"
                                 " pardtc: %d" % ier)
            nx = len(tx)
            ny = len(ty)
            newtx = tx[dx:nx - dx]
            newty = ty[dy:ny - dy]
            newkx, newky = kx - dx, ky - dy
            newclen = (nx - dx - kx - 1) * (ny - dy - ky - 1)
            return _DerivedBivariateSpline._from_tck((newtx, newty,
                                                      newc[:newclen],
                                                      newkx, newky))


_surfit_messages = {1: """
The required storage space exceeds the available storage space: nxest
or nyest too small, or s too small.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    2: """
A theoretically impossible result was found during the iteration
process for finding a smoothing spline with fp = s: s too small or
badly chosen eps.
Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
                    3: """
the maximal number of iterations maxit (set to 20 by the program)
allowed for finding a smoothing spline with fp=s has been reached:
s too small.
Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
                    4: """
No more knots can be added because the number of b-spline coefficients
(nx-kx-1)*(ny-ky-1) already exceeds the number of data points m:
either s or m too small.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    5: """
No more knots can be added because the additional knot would (quasi)
coincide with an old one: s too small or too large a weight to an
inaccurate data point.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    10: """
Error on entry, no approximation returned. The following conditions
must hold:
xb<=x[i]<=xe, yb<=y[i]<=ye, w[i]>0, i=0..m-1
If iopt==-1, then
  xb<tx[kx+1]<tx[kx+2]<...<tx[nx-kx-2]<xe
  yb<ty[ky+1]<ty[ky+2]<...<ty[ny-ky-2]<ye""",
                    -3: """
The coefficients of the spline returned have been computed as the
minimal norm least-squares solution of a (numerically) rank deficient
system (deficiency=%i). If deficiency is large, the results may be
inaccurate. Deficiency may strongly depend on the value of eps."""
                    }


class BivariateSpline(_BivariateSplineBase):
    """
    Base class for bivariate splines.

    This describes a spline ``s(x, y)`` of degrees ``kx`` and ``ky`` on
    the rectangle ``[xb, xe] * [yb, ye]`` calculated from a given set
    of data points ``(x, y, z)``.

    This class is meant to be subclassed, not instantiated directly.
    To construct these splines, call either `SmoothBivariateSpline` or
    `LSQBivariateSpline` or `RectBivariateSpline`.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    """

    def ev(self, xi, yi, dx=0, dy=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(xi[i], yi[i]),
        i=0,...,len(xi)-1``.

        Parameters
        ----------
        xi, yi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dx : int, optional
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int, optional
            Order of y-derivative

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to bilinearly interpolate an exponentially decaying
        function in 2 dimensions.

        >>> import numpy as np
        >>> from scipy.interpolate import RectBivariateSpline
        >>> def f(x, y):
        ...     return np.exp(-np.sqrt((x / 2) ** 2 + y**2))

        We sample the function on a coarse grid and set up the interpolator. Note that
        the default ``indexing="xy"`` of meshgrid would result in an unexpected
        (transposed) result after interpolation.

        >>> xarr = np.linspace(-3, 3, 21)
        >>> yarr = np.linspace(-3, 3, 21)
        >>> xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")
        >>> zdata = f(xgrid, ygrid)
        >>> rbs = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)

        Next we sample the function along a diagonal slice through the coordinate space
        on a finer grid using interpolation.

        >>> xinterp = np.linspace(-3, 3, 201)
        >>> yinterp = np.linspace(3, -3, 201)
        >>> zinterp = rbs.ev(xinterp, yinterp)

        And check that the interpolation passes through the function evaluations as a
        function of the distance from the origin along the slice.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sqrt(xarr**2 + yarr**2), np.diag(zdata), "or")
        >>> ax1.plot(np.sqrt(xinterp**2 + yinterp**2), zinterp, "-b")
        >>> plt.show()
        """
        return self.__call__(xi, yi, dx=dx, dy=dy, grid=False)

    def integral(self, xa, xb, ya, yb):
        """
        Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

        Parameters
        ----------
        xa, xb : float
            The end-points of the x integration interval.
        ya, yb : float
            The end-points of the y integration interval.

        Returns
        -------
        integ : float
            The value of the resulting integral.

        """
        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        with FITPACK_LOCK:
            return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)

    @staticmethod
    def _validate_input(x, y, z, w, kx, ky, eps):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if not x.size == y.size == z.size:
            raise ValueError('x, y, and z should have a same length')

        if w is not None:
            w = np.asarray(w)
            if x.size != w.size:
                raise ValueError('x, y, z, and w should have a same length')
            elif not np.all(w >= 0.0):
                raise ValueError('w should be positive')
        if (eps is not None) and (not 0.0 < eps < 1.0):
            raise ValueError('eps should be between (0, 1)')
        if not x.size >= (kx + 1) * (ky + 1):
            raise ValueError('The length of x, y and z should be at least'
                             ' (kx+1) * (ky+1)')
        return x, y, z, w


class _DerivedBivariateSpline(_BivariateSplineBase):
    """Bivariate spline constructed from the coefficients and knots of another
    spline.

    Notes
    -----
    The class is not meant to be instantiated directly from the data to be
    interpolated or smoothed. As a result, its ``fp`` attribute and
    ``get_residual`` method are inherited but overridden; ``AttributeError`` is
    raised when they are accessed.

    The other inherited attributes can be used as usual.
    """
    _invalid_why = ("is unavailable, because _DerivedBivariateSpline"
                    " instance is not constructed from data that are to be"
                    " interpolated or smoothed, but derived from the"
                    " underlying knots and coefficients of another spline"
                    " object")

    @property
    def fp(self):
        raise AttributeError(f"attribute \"fp\" {self._invalid_why}")

    def get_residual(self):
        raise AttributeError(f"method \"get_residual\" {self._invalid_why}")


class SmoothBivariateSpline(BivariateSpline):
    """
    Smooth bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    w : array_like, optional
        Positive 1-D sequence of weights, of same length as `x`, `y` and `z`.
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``z[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    This routine constructs spline knot vectors automatically via the FITPACK
    algorithm. The spline knots may be placed away from the data points. For
    some data sets, this routine may fail to construct an interpolating spline,
    even if one is requested via ``s=0`` parameter. In such situations, it is
    recommended to use `bisplrep` / `bisplev` directly instead of this routine
    and, if needed, increase the values of ``nxest`` and ``nyest`` parameters
    of `bisplrep`.

    For linear interpolation, prefer `LinearNDInterpolator`.
    See ``https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff``
    for discussion.

    """

    def __init__(self, x, y, z, w=None, bbox=[None] * 4, kx=3, ky=3, s=None,
                 eps=1e-16):

        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        xb, xe, yb, ye = bbox
        with FITPACK_LOCK:
            nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(
                x, y, z, w, xb, xe, yb, ye, kx, ky, s=s, eps=eps, lwrk2=1)
            if ier > 10:          # lwrk2 was to small, re-run
                nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(
                    x, y, z, w, xb, xe, yb, ye, kx, ky, s=s, eps=eps,
                    lwrk2=ier)
        if ier in [0, -1, -2]:  # normal return
            pass
        else:
            message = _surfit_messages.get(ier, f'ier={ier}')
            warnings.warn(message, stacklevel=2)

        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx-kx-1)*(ny-ky-1)]
        self.degrees = kx, ky


class LSQBivariateSpline(BivariateSpline):
    """
    Weighted least-squares bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    tx, ty : array_like
        Strictly ordered 1-D sequences of knots coordinates.
    w : array_like, optional
        Positive 1-D array of weights, of the same length as `x`, `y` and `z`.
    bbox : (4,) array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    """

    def __init__(self, x, y, z, tx, ty, w=None, bbox=[None]*4, kx=3, ky=3,
                 eps=None):

        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')

        nx = 2*kx+2+len(tx)
        ny = 2*ky+2+len(ty)
        # The Fortran subroutine "surfit" (called as dfitpack.surfit_lsq)
        # requires that the knot arrays passed as input should be "real
        # array(s) of dimension nmax" where "nmax" refers to the greater of nx
        # and ny. We pad the tx1/ty1 arrays here so that this is satisfied, and
        # slice them to the desired sizes upon return.
        nmax = max(nx, ny)
        tx1 = zeros((nmax,), float)
        ty1 = zeros((nmax,), float)
        tx1[kx+1:nx-kx-1] = tx
        ty1[ky+1:ny-ky-1] = ty

        xb, xe, yb, ye = bbox
        with FITPACK_LOCK:
            tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z, nx, tx1, ny, ty1,
                                                    w, xb, xe, yb, ye,
                                                    kx, ky, eps, lwrk2=1)
            if ier > 10:
                tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z,
                                                        nx, tx1, ny, ty1, w,
                                                        xb, xe, yb, ye,
                                                        kx, ky, eps, lwrk2=ier)
        if ier in [0, -1, -2]:  # normal return
            pass
        else:
            if ier < -2:
                deficiency = (nx-kx-1)*(ny-ky-1)+ier
                message = _surfit_messages.get(-3) % (deficiency)
            else:
                message = _surfit_messages.get(ier, f'ier={ier}')
            warnings.warn(message, stacklevel=2)
        self.fp = fp
        self.tck = tx1[:nx], ty1[:ny], c
        self.degrees = kx, ky


class RectBivariateSpline(BivariateSpline):
    """
    Bivariate spline approximation over a rectangular mesh.

    Can be used for both smoothing and interpolating data.

    Parameters
    ----------
    x,y : array_like
        1-D arrays of coordinates in strictly ascending order.
        Evaluated points outside the data range will be extrapolated.
    z : array_like
        2-D array of data with shape (x.size,y.size).
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain, which means the start and end spline knots of
        each dimension are set by these values. By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s`` where f is a spline
        function. Default is ``s=0``, which is for interpolation.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    """

    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0):
        x, y, bbox = ravel(x), ravel(y), ravel(bbox)
        z = np.asarray(z)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')
        if not np.all(diff(y) > 0.0):
            raise ValueError('y must be strictly increasing')
        if not x.size == z.shape[0]:
            raise ValueError('x dimension of z must have same number of '
                             'elements as x')
        if not y.size == z.shape[1]:
            raise ValueError('y dimension of z must have same number of '
                             'elements as y')
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        z = ravel(z)
        xb, xe, yb, ye = bbox
        with FITPACK_LOCK:
            nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(x, y, z, xb, xe, yb,
                                                            ye, kx, ky, s)

        if ier not in [0, -1, -2]:
            msg = _surfit_messages.get(ier, f'ier={ier}')
            raise ValueError(msg)

        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)]
        self.degrees = kx, ky


_spherefit_messages = _surfit_messages.copy()
_spherefit_messages[10] = """
ERROR. On entry, the input data are controlled on validity. The following
       restrictions must be satisfied:
            -1<=iopt<=1,  m>=2, ntest>=8 ,npest >=8, 0<eps<1,
            0<=teta(i)<=pi, 0<=phi(i)<=2*pi, w(i)>0, i=1,...,m
            lwrk1 >= 185+52*v+10*u+14*u*v+8*(u-1)*v**2+8*m
            kwrk >= m+(ntest-7)*(npest-7)
            if iopt=-1: 8<=nt<=ntest , 9<=np<=npest
                        0<tt(5)<tt(6)<...<tt(nt-4)<pi
                        0<tp(5)<tp(6)<...<tp(np-4)<2*pi
            if iopt>=0: s>=0
            if one of these conditions is found to be violated,control
            is immediately repassed to the calling program. in that
            case there is no approximation returned."""
_spherefit_messages[-3] = """
WARNING. The coefficients of the spline returned have been computed as the
         minimal norm least-squares solution of a (numerically) rank
         deficient system (deficiency=%i, rank=%i). Especially if the rank
         deficiency, which is computed by 6+(nt-8)*(np-7)+ier, is large,
         the results may be inaccurate. They could also seriously depend on
         the value of eps."""


class SphereBivariateSpline(_BivariateSplineBase):
    """
    Bivariate spline s(x,y) of degrees 3 on a sphere, calculated from a
    given set of data points (theta,phi,r).

    .. versionadded:: 0.11.0

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQUnivariateSpline :
        a univariate spline using weighted least-squares fitting
    """

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points
            ``(theta[i], phi[i]), i=0, ..., len(x)-1``.  Standard
            Numpy broadcasting is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays theta, phi. The arrays
            must be sorted to increasing order.
            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int
            Order of phi-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        Examples
        --------

        Suppose that we want to use splines to interpolate a bivariate function on a
        sphere. The value of the function is known on a grid of longitudes and
        colatitudes.

        >>> import numpy as np
        >>> from scipy.interpolate import RectSphereBivariateSpline
        >>> def f(theta, phi):
        ...     return np.sin(theta) * np.cos(phi)

        We evaluate the function on the grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> thetaarr = np.linspace(0, np.pi, 22)[1:-1]
        >>> phiarr = np.linspace(0, 2 * np.pi, 21)[:-1]
        >>> thetagrid, phigrid = np.meshgrid(thetaarr, phiarr, indexing="ij")
        >>> zdata = f(thetagrid, phigrid)

        We next set up the interpolator and use it to evaluate the function
        on a finer grid.

        >>> rsbs = RectSphereBivariateSpline(thetaarr, phiarr, zdata)
        >>> thetaarr_fine = np.linspace(0, np.pi, 200)
        >>> phiarr_fine = np.linspace(0, 2 * np.pi, 200)
        >>> zdata_fine = rsbs(thetaarr_fine, phiarr_fine)

        Finally we plot the coarsly-sampled input data alongside the
        finely-sampled interpolated data to check that they agree.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1)
        >>> ax2 = fig.add_subplot(1, 2, 2)
        >>> ax1.imshow(zdata)
        >>> ax2.imshow(zdata_fine)
        >>> plt.show()
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if theta.size > 0 and (theta.min() < 0. or theta.max() > np.pi):
            raise ValueError("requested theta out of bounds.")

        return _BivariateSplineBase.__call__(self, theta, phi,
                                             dx=dtheta, dy=dphi, grid=grid)

    def ev(self, theta, phi, dtheta=0, dphi=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(theta[i], phi[i]),
        i=0,...,len(theta)-1``.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
            The ordering of axes is consistent with
            np.meshgrid(..., indexing="ij") and inconsistent with the
            default ordering np.meshgrid(..., indexing="xy").
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int, optional
            Order of phi-derivative

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to use splines to interpolate a bivariate function on a
        sphere. The value of the function is known on a grid of longitudes and
        colatitudes.

        >>> import numpy as np
        >>> from scipy.interpolate import RectSphereBivariateSpline
        >>> def f(theta, phi):
        ...     return np.sin(theta) * np.cos(phi)

        We evaluate the function on the grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> thetaarr = np.linspace(0, np.pi, 22)[1:-1]
        >>> phiarr = np.linspace(0, 2 * np.pi, 21)[:-1]
        >>> thetagrid, phigrid = np.meshgrid(thetaarr, phiarr, indexing="ij")
        >>> zdata = f(thetagrid, phigrid)

        We next set up the interpolator and use it to evaluate the function
        at points not on the original grid.

        >>> rsbs = RectSphereBivariateSpline(thetaarr, phiarr, zdata)
        >>> thetainterp = np.linspace(thetaarr[0], thetaarr[-1], 200)
        >>> phiinterp = np.linspace(phiarr[0], phiarr[-1], 200)
        >>> zinterp = rsbs.ev(thetainterp, phiinterp)

        Finally we plot the original data for a diagonal slice through the
        initial grid, and the spline approximation along the same slice.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sin(thetaarr) * np.sin(phiarr), np.diag(zdata), "or")
        >>> ax1.plot(np.sin(thetainterp) * np.sin(phiinterp), zinterp, "-b")
        >>> plt.show()
        """
        return self.__call__(theta, phi, dtheta=dtheta, dphi=dphi, grid=False)


class SmoothSphereBivariateSpline(SphereBivariateSpline):
    """
    Smooth bivariate spline approximation in spherical coordinates.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    w : array_like, optional
        Positive 1-D sequence of weights.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``r[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> import numpy as np
    >>> theta = np.linspace(0., np.pi, 7)
    >>> phi = np.linspace(0., 2*np.pi, 9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> from scipy.interpolate import SmoothSphereBivariateSpline
    >>> lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                   data.T.ravel(), s=3.5)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2 * np.pi, 90)

    >>> data_smth = lut(fine_lats, fine_lons)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_smth, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, w=None, s=0., eps=1E-16):

        theta, phi, r = np.asarray(theta), np.asarray(phi), np.asarray(r)

        # input validation
        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2.0 * np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not s >= 0.0:
            raise ValueError('s should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')

        with FITPACK_LOCK:
            nt_, tt_, np_, tp_, c, fp, ier = dfitpack.spherfit_smth(theta, phi,
                                                                    r, w=w, s=s,
                                                                    eps=eps)
        if ier not in [0, -1, -2]:
            message = _spherefit_messages.get(ier, f'ier={ier}')
            raise ValueError(message)

        self.fp = fp
        self.tck = tt_[:nt_], tp_[:np_], c[:(nt_ - 4) * (np_ - 4)]
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if phi.size > 0 and (phi.min() < 0. or phi.max() > 2. * np.pi):
            raise ValueError("requested phi out of bounds.")

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)


class LSQSphereBivariateSpline(SphereBivariateSpline):
    """
    Weighted least-squares bivariate spline approximation in spherical
    coordinates.

    Determines a smoothing bicubic spline according to a given
    set of knots in the `theta` and `phi` directions.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    tt, tp : array_like
        Strictly ordered 1-D sequences of knots coordinates.
        Coordinates must satisfy ``0 < tt[i] < pi``, ``0 < tp[i] < 2*pi``.
    w : array_like, optional
        Positive 1-D sequence of weights, of the same length as `theta`, `phi`
        and `r`.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the
        open interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> from scipy.interpolate import LSQSphereBivariateSpline
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> theta = np.linspace(0, np.pi, num=7)
    >>> phi = np.linspace(0, 2*np.pi, num=9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object. Here, we must also specify the
    coordinates of the knots to use.

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> knotst, knotsp = theta.copy(), phi.copy()
    >>> knotst[0] += .0001
    >>> knotst[-1] -= .0001
    >>> knotsp[0] += .0001
    >>> knotsp[-1] -= .0001
    >>> lut = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                data.T.ravel(), knotst, knotsp)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2*np.pi, 90)
    >>> data_lsq = lut(fine_lats, fine_lons)

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_lsq, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, tt, tp, w=None, eps=1E-16):

        theta, phi, r = np.asarray(theta), np.asarray(phi), np.asarray(r)
        tt, tp = np.asarray(tt), np.asarray(tp)

        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2*np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if not ((0.0 < tt).all() and (tt < np.pi).all()):
            raise ValueError('tt should be between (0, pi)')
        if not ((0.0 < tp).all() and (tp < 2*np.pi).all()):
            raise ValueError('tp should be between (0, 2pi)')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')

        nt_, np_ = 8 + len(tt), 8 + len(tp)
        tt_, tp_ = zeros((nt_,), float), zeros((np_,), float)
        tt_[4:-4], tp_[4:-4] = tt, tp
        tt_[-4:], tp_[-4:] = np.pi, 2. * np.pi
        with FITPACK_LOCK:
            tt_, tp_, c, fp, ier = dfitpack.spherfit_lsq(theta, phi, r, tt_, tp_,
                                                        w=w, eps=eps)
        if ier > 0:
            message = _spherefit_messages.get(ier, f'ier={ier}')
            raise ValueError(message)

        self.fp = fp
        self.tck = tt_, tp_, c
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if phi.size > 0 and (phi.min() < 0. or phi.max() > 2. * np.pi):
            raise ValueError("requested phi out of bounds.")

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)


_spfit_messages = _surfit_messages.copy()
_spfit_messages[10] = """
ERROR: on entry, the input data are controlled on validity
       the following restrictions must be satisfied.
          -1<=iopt(1)<=1, 0<=iopt(2)<=1, 0<=iopt(3)<=1,
          -1<=ider(1)<=1, 0<=ider(2)<=1, ider(2)=0 if iopt(2)=0.
          -1<=ider(3)<=1, 0<=ider(4)<=1, ider(4)=0 if iopt(3)=0.
          mu >= mumin (see above), mv >= 4, nuest >=8, nvest >= 8,
          kwrk>=5+mu+mv+nuest+nvest,
          lwrk >= 12+nuest*(mv+nvest+3)+nvest*24+4*mu+8*mv+max(nuest,mv+nvest)
          0< u(i-1)<u(i)< pi,i=2,..,mu,
          -pi<=v(1)< pi, v(1)<v(i-1)<v(i)<v(1)+2*pi, i=3,...,mv
          if iopt(1)=-1: 8<=nu<=min(nuest,mu+6+iopt(2)+iopt(3))
                         0<tu(5)<tu(6)<...<tu(nu-4)< pi
                         8<=nv<=min(nvest,mv+7)
                         v(1)<tv(5)<tv(6)<...<tv(nv-4)<v(1)+2*pi
                         the schoenberg-whitney conditions, i.e. there must be
                         subset of grid coordinates uu(p) and vv(q) such that
                            tu(p) < uu(p) < tu(p+4) ,p=1,...,nu-4
                            (iopt(2)=1 and iopt(3)=1 also count for a uu-value
                            tv(q) < vv(q) < tv(q+4) ,q=1,...,nv-4
                            (vv(q) is either a value v(j) or v(j)+2*pi)
          if iopt(1)>=0: s>=0
          if s=0: nuest>=mu+6+iopt(2)+iopt(3), nvest>=mv+7
       if one of these conditions is found to be violated,control is
       immediately repassed to the calling program. in that case there is no
       approximation returned."""


class RectSphereBivariateSpline(SphereBivariateSpline):
    """
    Bivariate spline approximation over a rectangular mesh on a sphere.

    Can be used for smoothing data.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    u : array_like
        1-D array of colatitude coordinates in strictly ascending order.
        Coordinates must be given in radians and lie within the open interval
        ``(0, pi)``.
    v : array_like
        1-D array of longitude coordinates in strictly ascending order.
        Coordinates must be given in radians. First element (``v[0]``) must lie
        within the interval ``[-pi, pi)``. Last element (``v[-1]``) must satisfy
        ``v[-1] <= v[0] + 2*pi``.
    r : array_like
        2-D array of data with shape ``(u.size, v.size)``.
    s : float, optional
        Positive smoothing factor defined for estimation condition
        (``s=0`` is for interpolation).
    pole_continuity : bool or (bool, bool), optional
        Order of continuity at the poles ``u=0`` (``pole_continuity[0]``) and
        ``u=pi`` (``pole_continuity[1]``).  The order of continuity at the pole
        will be 1 or 0 when this is True or False, respectively.
        Defaults to False.
    pole_values : float or (float, float), optional
        Data values at the poles ``u=0`` and ``u=pi``.  Either the whole
        parameter or each individual element can be None.  Defaults to None.
    pole_exact : bool or (bool, bool), optional
        Data value exactness at the poles ``u=0`` and ``u=pi``.  If True, the
        value is considered to be the right function value, and it will be
        fitted exactly. If False, the value will be considered to be a data
        value just like the other data values.  Defaults to False.
    pole_flat : bool or (bool, bool), optional
        For the poles at ``u=0`` and ``u=pi``, specify whether or not the
        approximation has vanishing derivatives.  Defaults to False.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    Currently, only the smoothing spline approximation (``iopt[0] = 0`` and
    ``iopt[0] = 1`` in the FITPACK routine) is supported.  The exact
    least-squares spline approximation is not implemented yet.

    When actually performing the interpolation, the requested `v` values must
    lie within the same length 2pi interval that the original `v` values were
    chosen from.

    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/spgrid.f

    Examples
    --------
    Suppose we have global data on a coarse grid

    >>> import numpy as np
    >>> lats = np.linspace(10, 170, 9) * np.pi / 180.
    >>> lons = np.linspace(0, 350, 18) * np.pi / 180.
    >>> data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
    ...               np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

    We want to interpolate it to a global one-degree grid

    >>> new_lats = np.linspace(1, 180, 180) * np.pi / 180
    >>> new_lons = np.linspace(1, 360, 360) * np.pi / 180
    >>> new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    We need to set up the interpolator object

    >>> from scipy.interpolate import RectSphereBivariateSpline
    >>> lut = RectSphereBivariateSpline(lats, lons, data)

    Finally we interpolate the data.  The `RectSphereBivariateSpline` object
    only takes 1-D arrays as input, therefore we need to do some reshaping.

    >>> data_interp = lut.ev(new_lats.ravel(),
    ...                      new_lons.ravel()).reshape((360, 180)).T

    Looking at the original and the interpolated data, one can see that the
    interpolant reproduces the original data very well:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(212)
    >>> ax2.imshow(data_interp, interpolation='nearest')
    >>> plt.show()

    Choosing the optimal value of ``s`` can be a delicate task. Recommended
    values for ``s`` depend on the accuracy of the data values.  If the user
    has an idea of the statistical errors on the data, she can also find a
    proper estimate for ``s``. By assuming that, if she specifies the
    right ``s``, the interpolator will use a spline ``f(u,v)`` which exactly
    reproduces the function underlying the data, she can evaluate
    ``sum((r(i,j)-s(u(i),v(j)))**2)`` to find a good estimate for this ``s``.
    For example, if she knows that the statistical errors on her
    ``r(i,j)``-values are not greater than 0.1, she may expect that a good
    ``s`` should have a value not larger than ``u.size * v.size * (0.1)**2``.

    If nothing is known about the statistical error in ``r(i,j)``, ``s`` must
    be determined by trial and error.  The best is then to start with a very
    large value of ``s`` (to determine the least-squares polynomial and the
    corresponding upper bound ``fp0`` for ``s``) and then to progressively
    decrease the value of ``s`` (say by a factor 10 in the beginning, i.e.
    ``s = fp0 / 10, fp0 / 100, ...``  and more carefully as the approximation
    shows more detail) to obtain closer fits.

    The interpolation results for different values of ``s`` give some insight
    into this process:

    >>> fig2 = plt.figure()
    >>> s = [3e9, 2e9, 1e9, 1e8]
    >>> for idx, sval in enumerate(s, 1):
    ...     lut = RectSphereBivariateSpline(lats, lons, data, s=sval)
    ...     data_interp = lut.ev(new_lats.ravel(),
    ...                          new_lons.ravel()).reshape((360, 180)).T
    ...     ax = fig2.add_subplot(2, 2, idx)
    ...     ax.imshow(data_interp, interpolation='nearest')
    ...     ax.set_title(f"s = {sval:g}")
    >>> plt.show()

    """

    def __init__(self, u, v, r, s=0., pole_continuity=False, pole_values=None,
                 pole_exact=False, pole_flat=False):
        iopt = np.array([0, 0, 0], dtype=dfitpack_int)
        ider = np.array([-1, 0, -1, 0], dtype=dfitpack_int)
        if pole_values is None:
            pole_values = (None, None)
        elif isinstance(pole_values, (float, np.float32, np.float64)):
            pole_values = (pole_values, pole_values)
        if isinstance(pole_continuity, bool):
            pole_continuity = (pole_continuity, pole_continuity)
        if isinstance(pole_exact, bool):
            pole_exact = (pole_exact, pole_exact)
        if isinstance(pole_flat, bool):
            pole_flat = (pole_flat, pole_flat)

        r0, r1 = pole_values
        iopt[1:] = pole_continuity
        if r0 is None:
            ider[0] = -1
        else:
            ider[0] = pole_exact[0]

        if r1 is None:
            ider[2] = -1
        else:
            ider[2] = pole_exact[1]

        ider[1], ider[3] = pole_flat

        u, v = np.ravel(u), np.ravel(v)
        r = np.asarray(r)

        if not (0.0 < u[0] and u[-1] < np.pi):
            raise ValueError('u should be between (0, pi)')
        if not -np.pi <= v[0] < np.pi:
            raise ValueError('v[0] should be between [-pi, pi)')
        if not v[-1] <= v[0] + 2*np.pi:
            raise ValueError('v[-1] should be v[0] + 2pi or less ')

        if not np.all(np.diff(u) > 0.0):
            raise ValueError('u must be strictly increasing')
        if not np.all(np.diff(v) > 0.0):
            raise ValueError('v must be strictly increasing')

        if not u.size == r.shape[0]:
            raise ValueError('u dimension of r must have same number of '
                             'elements as u')
        if not v.size == r.shape[1]:
            raise ValueError('v dimension of r must have same number of '
                             'elements as v')

        if pole_continuity[1] is False and pole_flat[1] is True:
            raise ValueError('if pole_continuity is False, so must be '
                             'pole_flat')
        if pole_continuity[0] is False and pole_flat[0] is True:
            raise ValueError('if pole_continuity is False, so must be '
                             'pole_flat')

        if not s >= 0.0:
            raise ValueError('s should be positive')

        r = np.ravel(r)
        with FITPACK_LOCK:
            nu, tu, nv, tv, c, fp, ier = dfitpack.regrid_smth_spher(iopt, ider,
                                                                    u.copy(),
                                                                    v.copy(),
                                                                    r.copy(),
                                                                    r0, r1, s)

        if ier not in [0, -1, -2]:
            msg = _spfit_messages.get(ier, f'ier={ier}')
            raise ValueError(msg)

        self.fp = fp
        self.tck = tu[:nu], tv[:nv], c[:(nu - 4) * (nv-4)]
        self.degrees = (3, 3)
        self.v0 = v[0]

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)
<|endoftext|>__all__ = ['RegularGridInterpolator', 'interpn']

import itertools

import numpy as np

import scipy.sparse.linalg as ssl

from ._interpnd import _ndim_coords_from_arrays
from ._cubic import PchipInterpolator
from ._rgi_cython import evaluate_linear_2d, find_indices
from ._bsplines import make_interp_spline
from ._fitpack2 import RectBivariateSpline
from ._ndbspline import make_ndbspl


def _check_points(points):
    descending_dimensions = []
    grid = []
    for i, p in enumerate(points):
        # early make points float
        # see https://github.com/scipy/scipy/pull/17230
        p = np.asarray(p, dtype=float)
        if not np.all(p[1:] > p[:-1]):
            if np.all(p[1:] < p[:-1]):
                # input is descending, so make it ascending
                descending_dimensions.append(i)
                p = np.flip(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        # see https://github.com/scipy/scipy/issues/17716
        p = np.ascontiguousarray(p)
        grid.append(p)
    return tuple(grid), tuple(descending_dimensions)


def _check_dimensionality(points, values):
    if len(points) > values.ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), values.ndim))
    for i, p in enumerate(points):
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))


class RegularGridInterpolator:
    """
    Interpolator on a regular or rectilinear grid in arbitrary dimensions.

    The data must be defined on a rectilinear grid; that is, a rectangular
    grid with even or uneven spacing. Linear, nearest-neighbor, spline
    interpolations are supported. After setting up the interpolator object,
    the interpolation method may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data is
        accepted.

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip". This
        parameter will become the default for the object's ``__call__``
        method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True.

    fill_value : float or None, optional
        The value to use for points outside of the interpolation domain.
        If None, values outside the domain are extrapolated.
        Default is ``np.nan``.

    solver : callable, optional
        Only used for methods "slinear", "cubic" and "quintic".
        Sparse linear algebra solver for construction of the NdBSpline instance.
        Default is the iterative solver `scipy.sparse.linalg.gcrotmk`.

        .. versionadded:: 1.13

    solver_args: dict, optional
        Additional arguments to pass to `solver`, if any.

        .. versionadded:: 1.13

    Methods
    -------
    __call__

    Attributes
    ----------
    grid : tuple of ndarrays
        The points defining the regular grid in n dimensions.
        This tuple defines the full grid via
        ``np.meshgrid(*grid, indexing='ij')``
    values : ndarray
        Data values at the grid.
    method : str
        Interpolation method.
    fill_value : float or ``None``
        Use this value for out-of-bounds arguments to `__call__`.
    bounds_error : bool
        If ``True``, out-of-bounds argument raise a ``ValueError``.

    Notes
    -----
    Contrary to `LinearNDInterpolator` and `NearestNDInterpolator`, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    In other words, this class assumes that the data is defined on a
    *rectilinear* grid.

    .. versionadded:: 0.14

    The 'slinear'(k=1), 'cubic'(k=3), and 'quintic'(k=5) methods are
    tensor-product spline interpolators, where `k` is the spline degree,
    If any dimension has fewer points than `k` + 1, an error will be raised.

    .. versionadded:: 1.9

    If the input data is such that dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    **Choosing a solver for spline methods**

    Spline methods, "slinear", "cubic" and "quintic" involve solving a
    large sparse linear system at instantiation time. Depending on data,
    the default solver may or may not be adequate. When it is not, you may
    need to experiment with an optional `solver` argument, where you may
    choose between the direct solver (`scipy.sparse.linalg.spsolve`) or
    iterative solvers from `scipy.sparse.linalg`. You may need to supply
    additional parameters via the optional `solver_args` parameter (for instance,
    you may supply the starting value or target tolerance). See the
    `scipy.sparse.linalg` documentation for the full list of available options.

    Alternatively, you may instead use the legacy methods, "slinear_legacy",
    "cubic_legacy" and "quintic_legacy". These methods allow faster construction
    but evaluations will be much slower.

    Examples
    --------
    **Evaluate a function on the points of a 3-D grid**

    As a first example, we evaluate a simple example function on the points of
    a 3-D grid:

    >>> from scipy.interpolate import RegularGridInterpolator
    >>> import numpy as np
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    >>> data = f(xg, yg, zg)

    ``data`` is now a 3-D array with ``data[i, j, k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> interp = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3],
    ...                 [3.3, 5.2, 7.1]])
    >>> interp(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to

    >>> f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
    (125.54200000000002, 145.894)

    **Interpolate and extrapolate a 2D dataset**

    As a second example, we interpolate and extrapolate a 2D data set:

    >>> x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
    >>> def ff(x, y):
    ...     return x**2 + y**2

    >>> xg, yg = np.meshgrid(x, y, indexing='ij')
    >>> data = ff(xg, yg)
    >>> interp = RegularGridInterpolator((x, y), data,
    ...                                  bounds_error=False, fill_value=None)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
    ...            s=60, c='k', label='data')

    Evaluate and plot the interpolator on a finer grid

    >>> xx = np.linspace(-4, 9, 31)
    >>> yy = np.linspace(-4, 9, 31)
    >>> X, Y = np.meshgrid(xx, yy, indexing='ij')

    >>> # interpolator
    >>> ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
    ...                   alpha=0.4, color='m', label='linear interp')

    >>> # ground truth
    >>> ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
    ...                   alpha=0.4, label='ground truth')
    >>> plt.legend()
    >>> plt.show()

    Other examples are given
    :ref:`in the tutorial <tutorial-interpolate_regular_grid_interpolator>`.

    See Also
    --------
    NearestNDInterpolator : Nearest neighbor interpolator on *unstructured*
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolator on *unstructured* data
                           in N dimensions

    interpn : a convenience function which wraps `RegularGridInterpolator`

    scipy.ndimage.map_coordinates : interpolation on grids with equal spacing
                                    (suitable for e.g., N-D image resampling)

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
           :doi:`10.1090/S0025-5718-1988-0917826-0`

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3,
                          "slinear_legacy": 1, "cubic_legacy": 3, "quintic_legacy": 5,}
    _SPLINE_METHODS_recursive = {"slinear_legacy", "cubic_legacy",
                                "quintic_legacy", "pchip"}
    _SPLINE_METHODS_ndbspl = {"slinear", "cubic", "quintic"}
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan, *, solver=None, solver_args=None):
        if method not in self._ALL_METHODS:
            raise ValueError(f"Method '{method}' is not defined")
        elif method in self._SPLINE_METHODS:
            self._validate_grid_dimensions(points, method)
        self.method = method
        self._spline = None
        self.bounds_error = bounds_error
        self.grid, self._descending_dimensions = _check_points(points)
        self.values = self._check_values(values)
        self._check_dimensionality(self.grid, self.values)
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = np.flip(values, axis=self._descending_dimensions)
        if self.method == "pchip" and np.iscomplexobj(self.values):
            msg = ("`PchipInterpolator` only works with real values. If you are trying "
                   "to use the real components of the passed array, use `np.real` on "
                   "the array before passing to `RegularGridInterpolator`.")
            raise ValueError(msg)
        if method in self._SPLINE_METHODS_ndbspl:
            if solver_args is None:
                solver_args = {}
            self._spline = self._construct_spline(method, solver, **solver_args)
        else:
            if solver is not None or solver_args:
                raise ValueError(
                    f"{method =} does not accept the 'solver' argument. Got "
                    f" {solver = } and with arguments {solver_args}."
                )

    def _construct_spline(self, method, solver=None, **solver_args):
        if solver is None:
            solver = ssl.gcrotmk
        spl = make_ndbspl(
                self.grid, self.values, self._SPLINE_DEGREE_MAP[method],
                solver=solver, **solver_args
              )
        return spl

    def _check_dimensionality(self, grid, values):
        _check_dimensionality(grid, values)

    def _check_points(self, points):
        return _check_points(points)

    def _check_values(self, values):
        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        return values

    def _check_fill_value(self, values, fill_value):
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value

    def __call__(self, xi, method=None, *, nu=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str, optional
            The method of interpolation to perform. Supported are "linear",
            "nearest", "slinear", "cubic", "quintic" and "pchip". Default is
            the method chosen when the interpolator was created.

        nu : sequence of ints, length ndim, optional
            If not None, the orders of the derivatives to evaluate.
            Each entry must be non-negative.
            Only allowed for methods "slinear", "cubic" and "quintic".

            .. versionadded:: 1.13

        Returns
        -------
        values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.

        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

        Examples
        --------
        Here we define a nearest-neighbor interpolator of a simple function

        >>> import numpy as np
        >>> x, y = np.array([0, 1, 2]), np.array([1, 3, 7])
        >>> def f(x, y):
        ...     return x**2 + y**2
        >>> data = f(*np.meshgrid(x, y, indexing='ij', sparse=True))
        >>> from scipy.interpolate import RegularGridInterpolator
        >>> interp = RegularGridInterpolator((x, y), data, method='nearest')

        By construction, the interpolator uses the nearest-neighbor
        interpolation

        >>> interp([[1.5, 1.3], [0.3, 4.5]])
        array([2., 9.])

        We can however evaluate the linear interpolant by overriding the
        `method` parameter

        >>> interp([[1.5, 1.3], [0.3, 4.5]], method='linear')
        array([ 4.7, 24.3])
        """
        _spline = self._spline
        method = self.method if method is None else method
        is_method_changed = self.method != method
        if method not in self._ALL_METHODS:
            raise ValueError(f"Method '{method}' is not defined")
        if is_method_changed and method in self._SPLINE_METHODS_ndbspl:
            _spline = self._construct_spline(method)

        if nu is not None and method not in self._SPLINE_METHODS_ndbspl:
            raise ValueError(
                f"Can only compute derivatives for methods "
                f"{self._SPLINE_METHODS_ndbspl}, got {method =}."
            )

        xi, xi_shape, ndim, nans, out_of_bounds = self._prepare_xi(xi)

        if method == "linear":
            indices, norm_distances = self._find_indices(xi.T)
            if (ndim == 2 and hasattr(self.values, 'dtype') and
                    self.values.ndim == 2 and self.values.flags.writeable and
                    self.values.dtype in (np.float64, np.complex128) and
                    self.values.dtype.byteorder == '='):
                # until cython supports const fused types, the fast path
                # cannot support non-writeable values
                # a fast path
                out = np.empty(indices.shape[1], dtype=self.values.dtype)
                result = evaluate_linear_2d(self.values,
                                            indices,
                                            norm_distances,
                                            self.grid,
                                            out)
            else:
                result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_nearest(indices, norm_distances)
        elif method in self._SPLINE_METHODS:
            if is_method_changed:
                self._validate_grid_dimensions(self.grid, method)
            if method in self._SPLINE_METHODS_recursive:
                result = self._evaluate_spline(xi, method)
            else:
                result = _spline(xi, nu=nu)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        # f(nan) = nan, if any
        if np.any(nans):
            result[nans] = np.nan
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             f"{xi.shape[-1]} but this "
                             f"RegularGridInterpolator has dimension {ndim}")

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.asarray(xi, dtype=float)

        # find nans in input
        nans = np.any(np.isnan(xi), axis=-1)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)

        return xi, xi_shape, ndim, nans, out_of_bounds

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # Compute shifting up front before zipping everything together
        shift_norm_distances = [1 - yi for yi in norm_distances]
        shift_indices = [i + 1 for i in indices]

        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
        zipped1 = zip(indices, shift_norm_distances)
        zipped2 = zip(shift_indices, norm_distances)

        # Take all products of zipped1 and zipped2 and iterate over them
        # to get the terms in the above formula. This corresponds to iterating
        # over the vertices of a hypercube.
        hypercube = itertools.product(*zip(zipped1, zipped2))
        value = np.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = np.array([1.])
            for w in weights:
                weight = weight * w
            term = np.asarray(self.values[edge_indices]) * weight[vslice]
            value = value + term   # cannot use += because broadcasting
        return value

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [np.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _validate_grid_dimensions(self, points, method):
        k = self._SPLINE_DEGREE_MAP[method]
        for i, point in enumerate(points):
            ndim = len(np.atleast_1d(point))
            if ndim <= k:
                raise ValueError(f"There are {ndim} points in dimension {i},"
                                 f" but method {method} requires at least "
                                 f" {k+1} points per dimension.")

    def _evaluate_spline(self, xi, method):
        # ensure xi is 2D list of points to evaluate (`m` is the number of
        # points and `n` is the number of interpolation dimensions,
        # ``n == len(self.grid)``.)
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # Reorder the axes: n-dimensional process iterates over the
        # interpolation axes from the last axis downwards: E.g. for a 4D grid
        # the order of axes is 3, 2, 1, 0. Each 1D interpolation works along
        # the 0th axis of its argument array (for 1D routine it's its ``y``
        # array). Thus permute the interpolation axes of `values` *and keep
        # trailing dimensions trailing*.
        axes = tuple(range(self.values.ndim))
        axx = axes[:n][::-1] + axes[n:]
        values = self.values.transpose(axx)

        if method == 'pchip':
            _eval_func = self._do_pchip
        else:
            _eval_func = self._do_spline_fit
        k = self._SPLINE_DEGREE_MAP[method]

        # Non-stationary procedure: difficult to vectorize this part entirely
        # into numpy-level operations. Unfortunately this requires explicit
        # looping over each point in xi.

        # can at least vectorize the first pass across all points in the
        # last variable of xi.
        last_dim = n - 1
        first_values = _eval_func(self.grid[last_dim],
                                  values,
                                  xi[:, last_dim],
                                  k)

        # the rest of the dimensions have to be on a per point-in-xi basis
        shape = (m, *self.values.shape[n:])
        result = np.empty(shape, dtype=self.values.dtype)
        for j in range(m):
            # Main process: Apply 1D interpolate in each dimension
            # sequentially, starting with the last dimension.
            # These are then "folded" into the next dimension in-place.
            folded_values = first_values[j, ...]
            for i in range(last_dim-1, -1, -1):
                # Interpolate for each 1D from the last dimensions.
                # This collapses each 1D sequence into a scalar.
                folded_values = _eval_func(self.grid[i],
                                           folded_values,
                                           xi[j, i],
                                           k)
            result[j, ...] = folded_values

        return result

    @staticmethod
    def _do_spline_fit(x, y, pt, k):
        local_interp = make_interp_spline(x, y, k=k, axis=0)
        values = local_interp(pt)
        return values

    @staticmethod
    def _do_pchip(x, y, pt, k):
        local_interp = PchipInterpolator(x, y, axis=0)
        values = local_interp(pt)
        return values

    def _find_indices(self, xi):
        return find_indices(self.grid, xi)

    def _find_out_of_bounds(self, xi):
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds


def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data is
        accepted.

        .. deprecated:: 1.13.0
            Complex data is deprecated with ``method="pchip"`` and will raise an
            error in SciPy 1.15.0. This is because ``PchipInterpolator`` only
            works with real values. If you are trying to use the real components of
            the passed array, use ``np.real`` on ``values``.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic", "pchip", and "splinef2d".
        "splinef2d" is only supported for 2-dimensional data.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at `xi`. See notes for behaviour when
        ``xi.ndim == 1``.

    See Also
    --------
    NearestNDInterpolator : Nearest neighbor interpolation on unstructured
                            data in N dimensions
    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions
    RegularGridInterpolator : interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).
    RectBivariateSpline : Bivariate spline approximation over a rectangular mesh
    scipy.ndimage.map_coordinates : interpolation on grids with equal spacing
                                    (suitable for e.g., N-D image resampling)

    Notes
    -----

    .. versionadded:: 0.14

    In the case that ``xi.ndim == 1`` a new axis is inserted into
    the 0 position of the returned array, values_x, so its shape is
    instead ``(1,) + values.shape[ndim:]``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolation.

    Examples
    --------
    Evaluate a simple example function on the points of a regular 3-D grid:

    >>> import numpy as np
    >>> from scipy.interpolate import interpn
    >>> def value_func_3d(x, y, z):
    ...     return 2 * x + 3 * y - z
    >>> x = np.linspace(0, 4, 5)
    >>> y = np.linspace(0, 5, 6)
    >>> z = np.linspace(0, 6, 7)
    >>> points = (x, y, z)
    >>> values = value_func_3d(*np.meshgrid(*points, indexing='ij'))

    Evaluate the interpolating function at a point

    >>> point = np.array([2.21, 3.12, 1.15])
    >>> print(interpn(points, values, point))
    [12.63]

    """
    # sanity check 'method' kwarg
    if method not in ["linear", "nearest", "cubic", "quintic", "pchip",
                      "splinef2d", "slinear",
                      "slinear_legacy", "cubic_legacy", "quintic_legacy"]:
        raise ValueError("interpn only understands the methods 'linear', "
                         "'nearest', 'slinear', 'cubic', 'quintic', 'pchip', "
                         f"and 'splinef2d'. You provided {method}.")

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim
    if ndim > 2 and method == "splinef2d":
        raise ValueError("The method splinef2d can only be used for "
                         "2-dimensional input data")
    if not bounds_error and fill_value is None and method == "splinef2d":
        raise ValueError("The method splinef2d does not support extrapolation.")

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))
    if len(points) != ndim and method == 'splinef2d':
        raise ValueError("The method splinef2d can only be used for "
                         "scalar data with one point per coordinate")

    grid, descending_dimensions = _check_points(points)
    _check_dimensionality(grid, values)

    # sanity check requested xi
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[-1], len(grid)))

    if bounds_error:
        for i, p in enumerate(xi.T):
            if not np.logical_and(np.all(grid[i][0] <= p),
                                  np.all(p <= grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds "
                                 "in dimension %d" % i)

    # perform interpolation
    if method in RegularGridInterpolator._ALL_METHODS:
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
    elif method == "splinef2d":
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])

        # RectBivariateSpline doesn't support fill_value; we need to wrap here
        idx_valid = np.all((grid[0][0] <= xi[:, 0], xi[:, 0] <= grid[0][-1],
                            grid[1][0] <= xi[:, 1], xi[:, 1] <= grid[1][-1]),
                           axis=0)
        result = np.empty_like(xi[:, 0])

        # make a copy of values for RectBivariateSpline
        interp = RectBivariateSpline(points[0], points[1], values[:])
        result[idx_valid] = interp.ev(xi[idx_valid, 0], xi[idx_valid, 1])
        result[np.logical_not(idx_valid)] = fill_value

        return result.reshape(xi_shape[:-1])
    else:
        raise ValueError(f"unknown {method = }")
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BarycentricInterpolator',
    'KroghInterpolator',
    'approximate_taylor_polynomial',
    'barycentric_interpolate',
    'krogh_interpolate',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="polyint",
                                   private_modules=["_polyint"], all=__all__,
                                   attribute=name)
<|endoftext|>import warnings

import numpy as np
from scipy.special import factorial
from scipy._lib._util import (_asarray_validated, float_factorial, check_random_state,
                              _transition_to_rng)


__all__ = ["KroghInterpolator", "krogh_interpolate",
           "BarycentricInterpolator", "barycentric_interpolate",
           "approximate_taylor_polynomial"]


def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()


class _Interpolator1D:
    """
    Common features in univariate interpolation

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis
        Axis along which the interpolation goes in the original array
    _y_extra_shape
        Additional trailing shape of the input arrays, excluding
        the interpolation axis.
    dtype
        Dtype of the y-data arrays. Can be set via _set_dtype, which
        forces it to be float or complex.

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the interpolant.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of `x`.

