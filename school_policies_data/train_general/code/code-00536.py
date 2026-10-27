        else:
            self._server_hostname = None
        self._sslcontext = sslcontext
        # SSL-specific extra info. More info are set when the handshake
        # completes.
        self._extra = dict(sslcontext=sslcontext)

        # App data write buffering
        self._write_backlog = collections.deque()
        self._write_buffer_size = 0

        self._waiter = waiter
        self._loop = loop
        self._set_app_protocol(app_protocol)
        self._app_transport = _SSLProtocolTransport(self._loop, self)
        # _SSLPipe instance (None until the connection is made)
        self._sslpipe = None
        self._session_established = False
        self._in_handshake = False
        self._in_shutdown = False
        # transport, ex: SelectorSocketTransport
        self._transport = None
        self._call_connection_made = call_connection_made
        self._ssl_handshake_timeout = ssl_handshake_timeout

    def _set_app_protocol(self, app_protocol):
        self._app_protocol = app_protocol
        self._app_protocol_is_buffer = \
            isinstance(app_protocol, protocols.BufferedProtocol)

    def _wakeup_waiter(self, exc=None):
        if self._waiter is None:
            return
        if not self._waiter.cancelled():
            if exc is not None:
                self._waiter.set_exception(exc)
            else:
                self._waiter.set_result(None)
        self._waiter = None

    def connection_made(self, transport):
        """Called when the low-level connection is made.

        Start the SSL handshake.
        """
        self._transport = transport
        self._sslpipe = _SSLPipe(self._sslcontext,
                                 self._server_side,
                                 self._server_hostname)
        self._start_handshake()

    def connection_lost(self, exc):
        """Called when the low-level connection is lost or closed.

        The argument is an exception object or None (the latter
        meaning a regular EOF is received or the connection was
        aborted or closed).
        """
        if self._session_established:
            self._session_established = False
            self._loop.call_soon(self._app_protocol.connection_lost, exc)
        else:
            # Most likely an exception occurred while in SSL handshake.
            # Just mark the app transport as closed so that its __del__
            # doesn't complain.
            if self._app_transport is not None:
                self._app_transport._closed = True
        self._transport = None
        self._app_transport = None
        if getattr(self, '_handshake_timeout_handle', None):
            self._handshake_timeout_handle.cancel()
        self._wakeup_waiter(exc)
        self._app_protocol = None
        self._sslpipe = None

    def pause_writing(self):
        """Called when the low-level transport's buffer goes over
        the high-water mark.
        """
        self._app_protocol.pause_writing()

    def resume_writing(self):
        """Called when the low-level transport's buffer drains below
        the low-water mark.
        """
        self._app_protocol.resume_writing()

    def data_received(self, data):
        """Called when some SSL data is received.

        The argument is a bytes object.
        """
        if self._sslpipe is None:
            # transport closing, sslpipe is destroyed
            return

        try:
            ssldata, appdata = self._sslpipe.feed_ssldata(data)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as e:
            self._fatal_error(e, 'SSL error in data received')
            return

        for chunk in ssldata:
            self._transport.write(chunk)

        for chunk in appdata:
            if chunk:
                try:
                    if self._app_protocol_is_buffer:
                        protocols._feed_data_to_buffered_proto(
                            self._app_protocol, chunk)
                    else:
                        self._app_protocol.data_received(chunk)
                except (SystemExit, KeyboardInterrupt):
                    raise
                except BaseException as ex:
                    self._fatal_error(
                        ex, 'application protocol failed to receive SSL data')
                    return
            else:
                self._start_shutdown()
                break

    def eof_received(self):
        """Called when the other end of the low-level stream
        is half-closed.

        If this returns a false value (including None), the transport
        will close itself.  If it returns a true value, closing the
        transport is up to the protocol.
        """
        try:
            if self._loop.get_debug():
                logger.debug("%r received EOF", self)

            self._wakeup_waiter(ConnectionResetError)

            if not self._in_handshake:
                keep_open = self._app_protocol.eof_received()
                if keep_open:
                    logger.warning('returning true from eof_received() '
                                   'has no effect when using ssl')
        finally:
            self._transport.close()

    def _get_extra_info(self, name, default=None):
        if name in self._extra:
            return self._extra[name]
        elif self._transport is not None:
            return self._transport.get_extra_info(name, default)
        else:
            return default

    def _start_shutdown(self):
        if self._in_shutdown:
            return
        if self._in_handshake:
            self._abort()
        else:
            self._in_shutdown = True
            self._write_appdata(b'')

    def _write_appdata(self, data):
        self._write_backlog.append((data, 0))
        self._write_buffer_size += len(data)
        self._process_write_backlog()

    def _start_handshake(self):
        if self._loop.get_debug():
            logger.debug("%r starts SSL handshake", self)
            self._handshake_start_time = self._loop.time()
        else:
            self._handshake_start_time = None
        self._in_handshake = True
        # (b'', 1) is a special value in _process_write_backlog() to do
        # the SSL handshake
        self._write_backlog.append((b'', 1))
        self._handshake_timeout_handle = \
            self._loop.call_later(self._ssl_handshake_timeout,
                                  self._check_handshake_timeout)
        self._process_write_backlog()

    def _check_handshake_timeout(self):
        if self._in_handshake is True:
            msg = (
                f"SSL handshake is taking longer than "
                f"{self._ssl_handshake_timeout} seconds: "
                f"aborting the connection"
            )
            self._fatal_error(ConnectionAbortedError(msg))

    def _on_handshake_complete(self, handshake_exc):
        self._in_handshake = False
        self._handshake_timeout_handle.cancel()

        sslobj = self._sslpipe.ssl_object
        try:
            if handshake_exc is not None:
                raise handshake_exc

            peercert = sslobj.getpeercert()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if isinstance(exc, ssl.CertificateError):
                msg = 'SSL handshake failed on verifying the certificate'
            else:
                msg = 'SSL handshake failed'
            self._fatal_error(exc, msg)
            return

        if self._loop.get_debug():
            dt = self._loop.time() - self._handshake_start_time
            logger.debug("%r: SSL handshake took %.1f ms", self, dt * 1e3)

        # Add extra info that becomes available after handshake.
        self._extra.update(peercert=peercert,
                           cipher=sslobj.cipher(),
                           compression=sslobj.compression(),
                           ssl_object=sslobj,
                           )
        if self._call_connection_made:
            self._app_protocol.connection_made(self._app_transport)
        self._wakeup_waiter()
        self._session_established = True
        # In case transport.write() was already called. Don't call
        # immediately _process_write_backlog(), but schedule it:
        # _on_handshake_complete() can be called indirectly from
        # _process_write_backlog(), and _process_write_backlog() is not
        # reentrant.
        self._loop.call_soon(self._process_write_backlog)

    def _process_write_backlog(self):
        # Try to make progress on the write backlog.
        if self._transport is None or self._sslpipe is None:
            return

        try:
            for i in range(len(self._write_backlog)):
                data, offset = self._write_backlog[0]
                if data:
                    ssldata, offset = self._sslpipe.feed_appdata(data, offset)
                elif offset:
                    ssldata = self._sslpipe.do_handshake(
                        self._on_handshake_complete)
                    offset = 1
                else:
                    ssldata = self._sslpipe.shutdown(self._finalize)
                    offset = 1

                for chunk in ssldata:
                    self._transport.write(chunk)

                if offset < len(data):
                    self._write_backlog[0] = (data, offset)
                    # A short write means that a write is blocked on a read
                    # We need to enable reading if it is paused!
                    assert self._sslpipe.need_ssldata
                    if self._transport._paused:
                        self._transport.resume_reading()
                    break

                # An entire chunk from the backlog was processed. We can
                # delete it and reduce the outstanding buffer size.
                del self._write_backlog[0]
                self._write_buffer_size -= len(data)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if self._in_handshake:
                # Exceptions will be re-raised in _on_handshake_complete.
                self._on_handshake_complete(exc)
            else:
                self._fatal_error(exc, 'Fatal error on SSL transport')

    def _fatal_error(self, exc, message='Fatal error on transport'):
        if isinstance(exc, OSError):
            if self._loop.get_debug():
                logger.debug("%r: %s", self, message, exc_info=True)
        else:
            self._loop.call_exception_handler({
                'message': message,
                'exception': exc,
                'transport': self._transport,
                'protocol': self,
            })
        if self._transport:
            self._transport._force_close(exc)

    def _finalize(self):
        self._sslpipe = None

        if self._transport is not None:
            self._transport.close()

    def _abort(self):
        try:
            if self._transport is not None:
                self._transport.abort()
        finally:
            self._finalize()
<|endoftext|>"""Abstract Protocol base classes."""

__all__ = (
    'BaseProtocol', 'Protocol', 'DatagramProtocol',
    'SubprocessProtocol', 'BufferedProtocol',
)


class BaseProtocol:
    """Common base class for protocol interfaces.

    Usually user implements protocols that derived from BaseProtocol
    like Protocol or ProcessProtocol.

    The only case when BaseProtocol should be implemented directly is
    write-only transport like write pipe
    """

    __slots__ = ()

    def connection_made(self, transport):
        """Called when a connection is made.

        The argument is the transport representing the pipe connection.
        To receive data, wait for data_received() calls.
        When the connection is closed, connection_lost() is called.
        """

    def connection_lost(self, exc):
        """Called when the connection is lost or closed.

        The argument is an exception object or None (the latter
        meaning a regular EOF is received or the connection was
        aborted or closed).
        """

    def pause_writing(self):
        """Called when the transport's buffer goes over the high-water mark.

        Pause and resume calls are paired -- pause_writing() is called
        once when the buffer goes strictly over the high-water mark
        (even if subsequent writes increases the buffer size even
        more), and eventually resume_writing() is called once when the
        buffer size reaches the low-water mark.

        Note that if the buffer size equals the high-water mark,
        pause_writing() is not called -- it must go strictly over.
        Conversely, resume_writing() is called when the buffer size is
        equal or lower than the low-water mark.  These end conditions
        are important to ensure that things go as expected when either
        mark is zero.

        NOTE: This is the only Protocol callback that is not called
        through EventLoop.call_soon() -- if it were, it would have no
        effect when it's most needed (when the app keeps writing
        without yielding until pause_writing() is called).
        """

    def resume_writing(self):
        """Called when the transport's buffer drains below the low-water mark.

        See pause_writing() for details.
        """


class Protocol(BaseProtocol):
    """Interface for stream protocol.

    The user should implement this interface.  They can inherit from
    this class but don't need to.  The implementations here do
    nothing (they don't raise exceptions).

    When the user wants to requests a transport, they pass a protocol
    factory to a utility function (e.g., EventLoop.create_connection()).

    When the connection is made successfully, connection_made() is
    called with a suitable transport object.  Then data_received()
    will be called 0 or more times with data (bytes) received from the
    transport; finally, connection_lost() will be called exactly once
    with either an exception object or None as an argument.

    State machine of calls:

      start -> CM [-> DR*] [-> ER?] -> CL -> end

    * CM: connection_made()
    * DR: data_received()
    * ER: eof_received()
    * CL: connection_lost()
    """

    __slots__ = ()

    def data_received(self, data):
        """Called when some data is received.

        The argument is a bytes object.
        """

    def eof_received(self):
        """Called when the other end calls write_eof() or equivalent.

        If this returns a false value (including None), the transport
        will close itself.  If it returns a true value, closing the
        transport is up to the protocol.
        """


class BufferedProtocol(BaseProtocol):
    """Interface for stream protocol with manual buffer control.

    Event methods, such as `create_server` and `create_connection`,
    accept factories that return protocols that implement this interface.

    The idea of BufferedProtocol is that it allows to manually allocate
    and control the receive buffer.  Event loops can then use the buffer
    provided by the protocol to avoid unnecessary data copies.  This
    can result in noticeable performance improvement for protocols that
    receive big amounts of data.  Sophisticated protocols can allocate
    the buffer only once at creation time.

    State machine of calls:

      start -> CM [-> GB [-> BU?]]* [-> ER?] -> CL -> end

    * CM: connection_made()
    * GB: get_buffer()
    * BU: buffer_updated()
    * ER: eof_received()
    * CL: connection_lost()
    """

    __slots__ = ()

    def get_buffer(self, sizehint):
        """Called to allocate a new receive buffer.

        *sizehint* is a recommended minimal size for the returned
        buffer.  When set to -1, the buffer size can be arbitrary.

        Must return an object that implements the
        :ref:`buffer protocol <bufferobjects>`.
        It is an error to return a zero-sized buffer.
        """

    def buffer_updated(self, nbytes):
        """Called when the buffer was updated with the received data.

        *nbytes* is the total number of bytes that were written to
        the buffer.
        """

    def eof_received(self):
        """Called when the other end calls write_eof() or equivalent.

        If this returns a false value (including None), the transport
        will close itself.  If it returns a true value, closing the
        transport is up to the protocol.
        """


class DatagramProtocol(BaseProtocol):
    """Interface for datagram protocol."""

    __slots__ = ()

    def datagram_received(self, data, addr):
        """Called when some datagram is received."""

    def error_received(self, exc):
        """Called when a send or receive operation raises an OSError.

        (Other than BlockingIOError or InterruptedError.)
        """


class SubprocessProtocol(BaseProtocol):
    """Interface for protocol for subprocess calls."""

    __slots__ = ()

    def pipe_data_received(self, fd, data):
        """Called when the subprocess writes data into stdout/stderr pipe.

        fd is int file descriptor.
        data is bytes object.
        """

    def pipe_connection_lost(self, fd, exc):
        """Called when a file descriptor associated with the child process is
        closed.

        fd is the int file descriptor that was closed.
        """

    def process_exited(self):
        """Called when subprocess has exited."""


def _feed_data_to_buffered_proto(proto, data):
    data_len = len(data)
    while data_len:
        buf = proto.get_buffer(data_len)
        buf_len = len(buf)
        if not buf_len:
            raise RuntimeError('get_buffer() returned an empty buffer')

        if buf_len >= data_len:
            buf[:data_len] = data
            proto.buffer_updated(data_len)
            return
        else:
            buf[:buf_len] = data[:buf_len]
            proto.buffer_updated(buf_len)
            data = data[buf_len:]
            data_len = len(data)
<|endoftext|>__all__ = (
    'StreamReader', 'StreamWriter', 'StreamReaderProtocol',
    'open_connection', 'start_server')

import collections
import socket
import sys
import warnings
import weakref

if hasattr(socket, 'AF_UNIX'):
    __all__ += ('open_unix_connection', 'start_unix_server')

from . import coroutines
from . import events
from . import exceptions
from . import format_helpers
from . import protocols
from .log import logger
from .tasks import sleep


_DEFAULT_LIMIT = 2 ** 16  # 64 KiB


async def open_connection(host=None, port=None, *,
                          limit=_DEFAULT_LIMIT, **kwds):
    """A wrapper for create_connection() returning a (reader, writer) pair.

    The reader returned is a StreamReader instance; the writer is a
    StreamWriter instance.

    The arguments are all the usual arguments to create_connection()
    except protocol_factory; most common are positional host and port,
    with various optional keyword arguments following.

    Additional optional keyword arguments are loop (to set the event loop
    instance to use) and limit (to set the buffer limit passed to the
    StreamReader).

    (If you want to customize the StreamReader and/or
    StreamReaderProtocol classes, just copy the code -- there's
    really nothing special here except some convenience.)
    """
    loop = events.get_running_loop()
    reader = StreamReader(limit=limit, loop=loop)
    protocol = StreamReaderProtocol(reader, loop=loop)
    transport, _ = await loop.create_connection(
        lambda: protocol, host, port, **kwds)
    writer = StreamWriter(transport, protocol, reader, loop)
    return reader, writer


async def start_server(client_connected_cb, host=None, port=None, *,
                       limit=_DEFAULT_LIMIT, **kwds):
    """Start a socket server, call back for each client connected.

    The first parameter, `client_connected_cb`, takes two parameters:
    client_reader, client_writer.  client_reader is a StreamReader
    object, while client_writer is a StreamWriter object.  This
    parameter can either be a plain callback function or a coroutine;
    if it is a coroutine, it will be automatically converted into a
    Task.

    The rest of the arguments are all the usual arguments to
    loop.create_server() except protocol_factory; most common are
    positional host and port, with various optional keyword arguments
    following.  The return value is the same as loop.create_server().

    Additional optional keyword arguments are loop (to set the event loop
    instance to use) and limit (to set the buffer limit passed to the
    StreamReader).

    The return value is the same as loop.create_server(), i.e. a
    Server object which can be used to stop the service.
    """
    loop = events.get_running_loop()

    def factory():
        reader = StreamReader(limit=limit, loop=loop)
        protocol = StreamReaderProtocol(reader, client_connected_cb,
                                        loop=loop)
        return protocol

    return await loop.create_server(factory, host, port, **kwds)


if hasattr(socket, 'AF_UNIX'):
    # UNIX Domain Sockets are supported on this platform

    async def open_unix_connection(path=None, *,
                                   limit=_DEFAULT_LIMIT, **kwds):
        """Similar to `open_connection` but works with UNIX Domain Sockets."""
        loop = events.get_running_loop()

        reader = StreamReader(limit=limit, loop=loop)
        protocol = StreamReaderProtocol(reader, loop=loop)
        transport, _ = await loop.create_unix_connection(
            lambda: protocol, path, **kwds)
        writer = StreamWriter(transport, protocol, reader, loop)
        return reader, writer

    async def start_unix_server(client_connected_cb, path=None, *,
                                limit=_DEFAULT_LIMIT, **kwds):
        """Similar to `start_server` but works with UNIX Domain Sockets."""
        loop = events.get_running_loop()

        def factory():
            reader = StreamReader(limit=limit, loop=loop)
            protocol = StreamReaderProtocol(reader, client_connected_cb,
                                            loop=loop)
            return protocol

        return await loop.create_unix_server(factory, path, **kwds)


class FlowControlMixin(protocols.Protocol):
    """Reusable flow control logic for StreamWriter.drain().

    This implements the protocol methods pause_writing(),
    resume_writing() and connection_lost().  If the subclass overrides
    these it must call the super methods.

    StreamWriter.drain() must wait for _drain_helper() coroutine.
    """

    def __init__(self, loop=None):
        if loop is None:
            self._loop = events._get_event_loop(stacklevel=4)
        else:
            self._loop = loop
        self._paused = False
        self._drain_waiters = collections.deque()
        self._connection_lost = False

    def pause_writing(self):
        assert not self._paused
        self._paused = True
        if self._loop.get_debug():
            logger.debug("%r pauses writing", self)

    def resume_writing(self):
        assert self._paused
        self._paused = False
        if self._loop.get_debug():
            logger.debug("%r resumes writing", self)

        for waiter in self._drain_waiters:
            if not waiter.done():
                waiter.set_result(None)

    def connection_lost(self, exc):
        self._connection_lost = True
        # Wake up the writer(s) if currently paused.
        if not self._paused:
            return

        for waiter in self._drain_waiters:
            if not waiter.done():
                if exc is None:
                    waiter.set_result(None)
                else:
                    waiter.set_exception(exc)

    async def _drain_helper(self):
        if self._connection_lost:
            raise ConnectionResetError('Connection lost')
        if not self._paused:
            return
        waiter = self._loop.create_future()
        self._drain_waiters.append(waiter)
        try:
            await waiter
        finally:
            self._drain_waiters.remove(waiter)

    def _get_close_waiter(self, stream):
        raise NotImplementedError


class StreamReaderProtocol(FlowControlMixin, protocols.Protocol):
    """Helper class to adapt between Protocol and StreamReader.

    (This is a helper class instead of making StreamReader itself a
    Protocol subclass, because the StreamReader has other potential
    uses, and to prevent the user of the StreamReader to accidentally
    call inappropriate methods of the protocol.)
    """

    _source_traceback = None

    def __init__(self, stream_reader, client_connected_cb=None, loop=None):
        super().__init__(loop=loop)
        if stream_reader is not None:
            self._stream_reader_wr = weakref.ref(stream_reader)
            self._source_traceback = stream_reader._source_traceback
        else:
            self._stream_reader_wr = None
        if client_connected_cb is not None:
            # This is a stream created by the `create_server()` function.
            # Keep a strong reference to the reader until a connection
            # is established.
            self._strong_reader = stream_reader
        self._reject_connection = False
        self._stream_writer = None
        self._task = None
        self._transport = None
        self._client_connected_cb = client_connected_cb
        self._over_ssl = False
        self._closed = self._loop.create_future()

    @property
    def _stream_reader(self):
        if self._stream_reader_wr is None:
            return None
        return self._stream_reader_wr()

    def connection_made(self, transport):
        if self._reject_connection:
            context = {
                'message': ('An open stream was garbage collected prior to '
                            'establishing network connection; '
                            'call "stream.close()" explicitly.')
            }
            if self._source_traceback:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)
            transport.abort()
            return
        self._transport = transport
        reader = self._stream_reader
        if reader is not None:
            reader.set_transport(transport)
        self._over_ssl = transport.get_extra_info('sslcontext') is not None
        if self._client_connected_cb is not None:
            self._stream_writer = StreamWriter(transport, self,
                                               reader,
                                               self._loop)
            res = self._client_connected_cb(reader,
                                            self._stream_writer)
            if coroutines.iscoroutine(res):
                self._task = self._loop.create_task(res)
            self._strong_reader = None

    def connection_lost(self, exc):
        reader = self._stream_reader
        if reader is not None:
            if exc is None:
                reader.feed_eof()
            else:
                reader.set_exception(exc)
        if not self._closed.done():
            if exc is None:
                self._closed.set_result(None)
            else:
                self._closed.set_exception(exc)
        super().connection_lost(exc)
        self._stream_reader_wr = None
        self._stream_writer = None
        self._task = None
        self._transport = None

    def data_received(self, data):
        reader = self._stream_reader
        if reader is not None:
            reader.feed_data(data)

    def eof_received(self):
        reader = self._stream_reader
        if reader is not None:
            reader.feed_eof()
        if self._over_ssl:
            # Prevent a warning in SSLProtocol.eof_received:
            # "returning true from eof_received()
            # has no effect when using ssl"
            return False
        return True

    def _get_close_waiter(self, stream):
        return self._closed

    def __del__(self):
        # Prevent reports about unhandled exceptions.
        # Better than self._closed._log_traceback = False hack
        try:
            closed = self._closed
        except AttributeError:
            pass  # failed constructor
        else:
            if closed.done() and not closed.cancelled():
                closed.exception()


class StreamWriter:
    """Wraps a Transport.

    This exposes write(), writelines(), [can_]write_eof(),
    get_extra_info() and close().  It adds drain() which returns an
    optional Future on which you can wait for flow control.  It also
    adds a transport property which references the Transport
    directly.
    """

    def __init__(self, transport, protocol, reader, loop):
        self._transport = transport
        self._protocol = protocol
        # drain() expects that the reader has an exception() method
        assert reader is None or isinstance(reader, StreamReader)
        self._reader = reader
        self._loop = loop
        self._complete_fut = self._loop.create_future()
        self._complete_fut.set_result(None)

    def __repr__(self):
        info = [self.__class__.__name__, f'transport={self._transport!r}']
        if self._reader is not None:
            info.append(f'reader={self._reader!r}')
        return '<{}>'.format(' '.join(info))

    @property
    def transport(self):
        return self._transport

    def write(self, data):
        self._transport.write(data)

    def writelines(self, data):
        self._transport.writelines(data)

    def write_eof(self):
        return self._transport.write_eof()

    def can_write_eof(self):
        return self._transport.can_write_eof()

    def close(self):
        return self._transport.close()

    def is_closing(self):
        return self._transport.is_closing()

    async def wait_closed(self):
        await self._protocol._get_close_waiter(self)

    def get_extra_info(self, name, default=None):
        return self._transport.get_extra_info(name, default)

    async def drain(self):
        """Flush the write buffer.

        The intended use is to write

          w.write(data)
          await w.drain()
        """
        if self._reader is not None:
            exc = self._reader.exception()
            if exc is not None:
                raise exc
        if self._transport.is_closing():
            # Wait for protocol.connection_lost() call
            # Raise connection closing error if any,
            # ConnectionResetError otherwise
            # Yield to the event loop so connection_lost() may be
            # called.  Without this, _drain_helper() would return
            # immediately, and code that calls
            #     write(...); await drain()
            # in a loop would never call connection_lost(), so it
            # would not see an error when the socket is closed.
            await sleep(0)
        await self._protocol._drain_helper()


class StreamReader:

    _source_traceback = None

    def __init__(self, limit=_DEFAULT_LIMIT, loop=None):
        # The line length limit is  a security feature;
        # it also doubles as half the buffer limit.

        if limit <= 0:
            raise ValueError('Limit cannot be <= 0')

        self._limit = limit
        if loop is None:
            self._loop = events._get_event_loop()
        else:
            self._loop = loop
        self._buffer = bytearray()
        self._eof = False    # Whether we're done.
        self._waiter = None  # A future used by _wait_for_data()
        self._exception = None
        self._transport = None
        self._paused = False
        if self._loop.get_debug():
            self._source_traceback = format_helpers.extract_stack(
                sys._getframe(1))

    def __repr__(self):
        info = ['StreamReader']
        if self._buffer:
            info.append(f'{len(self._buffer)} bytes')
        if self._eof:
            info.append('eof')
        if self._limit != _DEFAULT_LIMIT:
            info.append(f'limit={self._limit}')
        if self._waiter:
            info.append(f'waiter={self._waiter!r}')
        if self._exception:
            info.append(f'exception={self._exception!r}')
        if self._transport:
            info.append(f'transport={self._transport!r}')
        if self._paused:
            info.append('paused')
        return '<{}>'.format(' '.join(info))

    def exception(self):
        return self._exception

    def set_exception(self, exc):
        self._exception = exc

        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            if not waiter.cancelled():
                waiter.set_exception(exc)

    def _wakeup_waiter(self):
        """Wakeup read*() functions waiting for data or EOF."""
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            if not waiter.cancelled():
                waiter.set_result(None)

    def set_transport(self, transport):
        assert self._transport is None, 'Transport already set'
        self._transport = transport

    def _maybe_resume_transport(self):
        if self._paused and len(self._buffer) <= self._limit:
            self._paused = False
            self._transport.resume_reading()

    def feed_eof(self):
        self._eof = True
        self._wakeup_waiter()

    def at_eof(self):
        """Return True if the buffer is empty and 'feed_eof' was called."""
        return self._eof and not self._buffer

    def feed_data(self, data):
        assert not self._eof, 'feed_data after feed_eof'

        if not data:
            return

        self._buffer.extend(data)
        self._wakeup_waiter()

        if (self._transport is not None and
                not self._paused and
                len(self._buffer) > 2 * self._limit):
            try:
                self._transport.pause_reading()
            except NotImplementedError:
                # The transport can't be paused.
                # We'll just have to buffer all data.
                # Forget the transport so we don't keep trying.
                self._transport = None
            else:
                self._paused = True

    async def _wait_for_data(self, func_name):
        """Wait until feed_data() or feed_eof() is called.

        If stream was paused, automatically resume it.
        """
        # StreamReader uses a future to link the protocol feed_data() method
        # to a read coroutine. Running two read coroutines at the same time
        # would have an unexpected behaviour. It would not possible to know
        # which coroutine would get the next data.
        if self._waiter is not None:
            raise RuntimeError(
                f'{func_name}() called while another coroutine is '
                f'already waiting for incoming data')

        assert not self._eof, '_wait_for_data after EOF'

        # Waiting for data while paused will make deadlock, so prevent it.
        # This is essential for readexactly(n) for case when n > self._limit.
        if self._paused:
            self._paused = False
            self._transport.resume_reading()

        self._waiter = self._loop.create_future()
        try:
            await self._waiter
        finally:
            self._waiter = None

    async def readline(self):
        """Read chunk of data from the stream until newline (b'\n') is found.

        On success, return chunk that ends with newline. If only partial
        line can be read due to EOF, return incomplete line without
        terminating newline. When EOF was reached while no bytes read, empty
        bytes object is returned.

        If limit is reached, ValueError will be raised. In that case, if
        newline was found, complete line including newline will be removed
        from internal buffer. Else, internal buffer will be cleared. Limit is
        compared against part of the line without newline.

        If stream was paused, this function will automatically resume it if
        needed.
        """
        sep = b'\n'
        seplen = len(sep)
        try:
            line = await self.readuntil(sep)
        except exceptions.IncompleteReadError as e:
            return e.partial
        except exceptions.LimitOverrunError as e:
            if self._buffer.startswith(sep, e.consumed):
                del self._buffer[:e.consumed + seplen]
            else:
                self._buffer.clear()
            self._maybe_resume_transport()
            raise ValueError(e.args[0])
        return line

    async def readuntil(self, separator=b'\n'):
        """Read data from the stream until ``separator`` is found.

        On success, the data and separator will be removed from the
        internal buffer (consumed). Returned data will include the
        separator at the end.

        Configured stream limit is used to check result. Limit sets the
        maximal length of data that can be returned, not counting the
        separator.

        If an EOF occurs and the complete separator is still not found,
        an IncompleteReadError exception will be raised, and the internal
        buffer will be reset.  The IncompleteReadError.partial attribute
        may contain the separator partially.

        If the data cannot be read because of over limit, a
        LimitOverrunError exception  will be raised, and the data
        will be left in the internal buffer, so it can be read again.
        """
        seplen = len(separator)
        if seplen == 0:
            raise ValueError('Separator should be at least one-byte string')

        if self._exception is not None:
            raise self._exception

        # Consume whole buffer except last bytes, which length is
        # one less than seplen. Let's check corner cases with
        # separator='SEPARATOR':
        # * we have received almost complete separator (without last
        #   byte). i.e buffer='some textSEPARATO'. In this case we
        #   can safely consume len(separator) - 1 bytes.
        # * last byte of buffer is first byte of separator, i.e.
        #   buffer='abcdefghijklmnopqrS'. We may safely consume
        #   everything except that last byte, but this require to
        #   analyze bytes of buffer that match partial separator.
        #   This is slow and/or require FSM. For this case our
        #   implementation is not optimal, since require rescanning
        #   of data that is known to not belong to separator. In
        #   real world, separator will not be so long to notice
        #   performance problems. Even when reading MIME-encoded
        #   messages :)

        # `offset` is the number of bytes from the beginning of the buffer
        # where there is no occurrence of `separator`.
        offset = 0

        # Loop until we find `separator` in the buffer, exceed the buffer size,
        # or an EOF has happened.
        while True:
            buflen = len(self._buffer)

            # Check if we now have enough data in the buffer for `separator` to
            # fit.
            if buflen - offset >= seplen:
                isep = self._buffer.find(separator, offset)

                if isep != -1:
                    # `separator` is in the buffer. `isep` will be used later
                    # to retrieve the data.
                    break

                # see upper comment for explanation.
                offset = buflen + 1 - seplen
                if offset > self._limit:
                    raise exceptions.LimitOverrunError(
                        'Separator is not found, and chunk exceed the limit',
                        offset)

            # Complete message (with full separator) may be present in buffer
            # even when EOF flag is set. This may happen when the last chunk
            # adds data which makes separator be found. That's why we check for
            # EOF *ater* inspecting the buffer.
            if self._eof:
                chunk = bytes(self._buffer)
                self._buffer.clear()
                raise exceptions.IncompleteReadError(chunk, None)

            # _wait_for_data() will resume reading if stream was paused.
            await self._wait_for_data('readuntil')

        if isep > self._limit:
            raise exceptions.LimitOverrunError(
                'Separator is found, but chunk is longer than limit', isep)

        chunk = self._buffer[:isep + seplen]
        del self._buffer[:isep + seplen]
        self._maybe_resume_transport()
        return bytes(chunk)

    async def read(self, n=-1):
        """Read up to `n` bytes from the stream.

        If `n` is not provided or set to -1,
        read until EOF, then return all read bytes.
        If EOF was received and the internal buffer is empty,
        return an empty bytes object.

        If `n` is 0, return an empty bytes object immediately.

        If `n` is positive, return at most `n` available bytes
        as soon as at least 1 byte is available in the internal buffer.
        If EOF is received before any byte is read, return an empty
        bytes object.

        Returned value is not limited with limit, configured at stream
        creation.

        If stream was paused, this function will automatically resume it if
        needed.
        """

        if self._exception is not None:
            raise self._exception

        if n == 0:
            return b''

        if n < 0:
            # This used to just loop creating a new waiter hoping to
            # collect everything in self._buffer, but that would
            # deadlock if the subprocess sends more than self.limit
            # bytes.  So just call self.read(self._limit) until EOF.
            blocks = []
            while True:
                block = await self.read(self._limit)
                if not block:
                    break
                blocks.append(block)
            return b''.join(blocks)

        if not self._buffer and not self._eof:
            await self._wait_for_data('read')

        # This will work right even if buffer is less than n bytes
        data = bytes(self._buffer[:n])
        del self._buffer[:n]

        self._maybe_resume_transport()
        return data

    async def readexactly(self, n):
        """Read exactly `n` bytes.

        Raise an IncompleteReadError if EOF is reached before `n` bytes can be
        read. The IncompleteReadError.partial attribute of the exception will
        contain the partial read bytes.

        if n is zero, return empty bytes object.

        Returned value is not limited with limit, configured at stream
        creation.

        If stream was paused, this function will automatically resume it if
        needed.
        """
        if n < 0:
            raise ValueError('readexactly size can not be less than zero')

        if self._exception is not None:
            raise self._exception

        if n == 0:
            return b''

        while len(self._buffer) < n:
            if self._eof:
                incomplete = bytes(self._buffer)
                self._buffer.clear()
                raise exceptions.IncompleteReadError(incomplete, n)

            await self._wait_for_data('readexactly')

        if len(self._buffer) == n:
            data = bytes(self._buffer)
            self._buffer.clear()
        else:
            data = bytes(self._buffer[:n])
            del self._buffer[:n]
        self._maybe_resume_transport()
        return data

    def __aiter__(self):
        return self

    async def __anext__(self):
        val = await self.readline()
        if val == b'':
            raise StopAsyncIteration
        return val
<|endoftext|>__all__ = ('Queue', 'PriorityQueue', 'LifoQueue', 'QueueFull', 'QueueEmpty')

import collections
import heapq
from types import GenericAlias

from . import locks
from . import mixins


class QueueEmpty(Exception):
    """Raised when Queue.get_nowait() is called on an empty Queue."""
    pass


class QueueFull(Exception):
    """Raised when the Queue.put_nowait() method is called on a full Queue."""
    pass


class Queue(mixins._LoopBoundMixin):
    """A queue, useful for coordinating producer and consumer coroutines.

    If maxsize is less than or equal to zero, the queue size is infinite. If it
    is an integer greater than 0, then "await put()" will block when the
    queue reaches maxsize, until an item is removed by get().

    Unlike the standard library Queue, you can reliably know this Queue's size
    with qsize(), since your single-threaded asyncio application won't be
    interrupted between calling qsize() and doing an operation on the Queue.
    """

    def __init__(self, maxsize=0, *, loop=mixins._marker):
        super().__init__(loop=loop)
        self._maxsize = maxsize

        # Futures.
        self._getters = collections.deque()
        # Futures.
        self._putters = collections.deque()
        self._unfinished_tasks = 0
        self._finished = locks.Event()
        self._finished.set()
        self._init(maxsize)

    # These three are overridable in subclasses.

    def _init(self, maxsize):
        self._queue = collections.deque()

    def _get(self):
        return self._queue.popleft()

    def _put(self, item):
        self._queue.append(item)

    # End of the overridable methods.

    def _wakeup_next(self, waiters):
        # Wake up the next waiter (if any) that isn't cancelled.
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break

    def __repr__(self):
        return f'<{type(self).__name__} at {id(self):#x} {self._format()}>'

    def __str__(self):
        return f'<{type(self).__name__} {self._format()}>'

    __class_getitem__ = classmethod(GenericAlias)

    def _format(self):
        result = f'maxsize={self._maxsize!r}'
        if getattr(self, '_queue', None):
            result += f' _queue={list(self._queue)!r}'
        if self._getters:
            result += f' _getters[{len(self._getters)}]'
        if self._putters:
            result += f' _putters[{len(self._putters)}]'
        if self._unfinished_tasks:
            result += f' tasks={self._unfinished_tasks}'
        return result

    def qsize(self):
        """Number of items in the queue."""
        return len(self._queue)

    @property
    def maxsize(self):
        """Number of items allowed in the queue."""
        return self._maxsize

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return not self._queue

    def full(self):
        """Return True if there are maxsize items in the queue.

        Note: if the Queue was initialized with maxsize=0 (the default),
        then full() is never True.
        """
        if self._maxsize <= 0:
            return False
        else:
            return self.qsize() >= self._maxsize

    async def put(self, item):
        """Put an item into the queue.

        Put an item into the queue. If the queue is full, wait until a free
        slot is available before adding item.
        """
        while self.full():
            putter = self._get_loop().create_future()
            self._putters.append(putter)
            try:
                await putter
            except:
                putter.cancel()  # Just in case putter is not done yet.
                try:
                    # Clean self._putters from canceled putters.
                    self._putters.remove(putter)
                except ValueError:
                    # The putter could be removed from self._putters by a
                    # previous get_nowait call.
                    pass
                if not self.full() and not putter.cancelled():
                    # We were woken up by get_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._putters)
                raise
        return self.put_nowait(item)

    def put_nowait(self, item):
        """Put an item into the queue without blocking.

        If no free slot is immediately available, raise QueueFull.
        """
        if self.full():
            raise QueueFull
        self._put(item)
        self._unfinished_tasks += 1
        self._finished.clear()
        self._wakeup_next(self._getters)

    async def get(self):
        """Remove and return an item from the queue.

        If queue is empty, wait until an item is available.
        """
        while self.empty():
            getter = self._get_loop().create_future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # Just in case getter is not done yet.
                try:
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                except ValueError:
                    # The getter could be removed from self._getters by a
                    # previous put_nowait call.
                    pass
                if not self.empty() and not getter.cancelled():
                    # We were woken up by put_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._getters)
                raise
        return self.get_nowait()

    def get_nowait(self):
        """Remove and return an item from the queue.

        Return an item if one is immediately available, else raise QueueEmpty.
        """
        if self.empty():
            raise QueueEmpty
        item = self._get()
        self._wakeup_next(self._putters)
        return item

    def task_done(self):
        """Indicate that a formerly enqueued task is complete.

        Used by queue consumers. For each get() used to fetch a task,
        a subsequent call to task_done() tells the queue that the processing
        on the task is complete.

        If a join() is currently blocking, it will resume when all items have
        been processed (meaning that a task_done() call was received for every
        item that had been put() into the queue).

        Raises ValueError if called more times than there were items placed in
        the queue.
        """
        if self._unfinished_tasks <= 0:
            raise ValueError('task_done() called too many times')
        self._unfinished_tasks -= 1
        if self._unfinished_tasks == 0:
            self._finished.set()

    async def join(self):
        """Block until all items in the queue have been gotten and processed.

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer calls task_done() to
        indicate that the item was retrieved and all work on it is complete.
        When the count of unfinished tasks drops to zero, join() unblocks.
        """
        if self._unfinished_tasks > 0:
            await self._finished.wait()


class PriorityQueue(Queue):
    """A subclass of Queue; retrieves entries in priority order (lowest first).

    Entries are typically tuples of the form: (priority number, data).
    """

    def _init(self, maxsize):
        self._queue = []

    def _put(self, item, heappush=heapq.heappush):
        heappush(self._queue, item)

    def _get(self, heappop=heapq.heappop):
        return heappop(self._queue)


class LifoQueue(Queue):
    """A subclass of Queue that retrieves most recently added entries first."""

    def _init(self, maxsize):
        self._queue = []

    def _put(self, item):
        self._queue.append(item)

    def _get(self):
        return self._queue.pop()
<|endoftext|>"""Abstract Transport class."""

__all__ = (
    'BaseTransport', 'ReadTransport', 'WriteTransport',
    'Transport', 'DatagramTransport', 'SubprocessTransport',
)


class BaseTransport:
    """Base class for transports."""

    __slots__ = ('_extra',)

    def __init__(self, extra=None):
        if extra is None:
            extra = {}
        self._extra = extra

    def get_extra_info(self, name, default=None):
        """Get optional transport information."""
        return self._extra.get(name, default)

    def is_closing(self):
        """Return True if the transport is closing or closed."""
        raise NotImplementedError

    def close(self):
        """Close the transport.

        Buffered data will be flushed asynchronously.  No more data
        will be received.  After all buffered data is flushed, the
        protocol's connection_lost() method will (eventually) be
        called with None as its argument.
        """
        raise NotImplementedError

    def set_protocol(self, protocol):
        """Set a new protocol."""
        raise NotImplementedError

    def get_protocol(self):
        """Return the current protocol."""
        raise NotImplementedError


class ReadTransport(BaseTransport):
    """Interface for read-only transports."""

    __slots__ = ()

    def is_reading(self):
        """Return True if the transport is receiving."""
        raise NotImplementedError

    def pause_reading(self):
        """Pause the receiving end.

        No data will be passed to the protocol's data_received()
        method until resume_reading() is called.
        """
        raise NotImplementedError

    def resume_reading(self):
        """Resume the receiving end.

        Data received will once again be passed to the protocol's
        data_received() method.
        """
        raise NotImplementedError


class WriteTransport(BaseTransport):
    """Interface for write-only transports."""

    __slots__ = ()

    def set_write_buffer_limits(self, high=None, low=None):
        """Set the high- and low-water limits for write flow control.

        These two values control when to call the protocol's
        pause_writing() and resume_writing() methods.  If specified,
        the low-water limit must be less than or equal to the
        high-water limit.  Neither value can be negative.

        The defaults are implementation-specific.  If only the
        high-water limit is given, the low-water limit defaults to an
        implementation-specific value less than or equal to the
        high-water limit.  Setting high to zero forces low to zero as
        well, and causes pause_writing() to be called whenever the
        buffer becomes non-empty.  Setting low to zero causes
        resume_writing() to be called only once the buffer is empty.
        Use of zero for either limit is generally sub-optimal as it
        reduces opportunities for doing I/O and computation
        concurrently.
        """
        raise NotImplementedError

    def get_write_buffer_size(self):
        """Return the current size of the write buffer."""
        raise NotImplementedError

    def get_write_buffer_limits(self):
        """Get the high and low watermarks for write flow control. 
        Return a tuple (low, high) where low and high are 
        positive number of bytes."""
        raise NotImplementedError

    def write(self, data):
        """Write some data bytes to the transport.

        This does not block; it buffers the data and arranges for it
        to be sent out asynchronously.
        """
        raise NotImplementedError

    def writelines(self, list_of_data):
        """Write a list (or any iterable) of data bytes to the transport.

        The default implementation concatenates the arguments and
        calls write() on the result.
        """
        data = b''.join(list_of_data)
        self.write(data)

    def write_eof(self):
        """Close the write end after flushing buffered data.

        (This is like typing ^D into a UNIX program reading from stdin.)

        Data may still be received.
        """
        raise NotImplementedError

    def can_write_eof(self):
        """Return True if this transport supports write_eof(), False if not."""
        raise NotImplementedError

    def abort(self):
        """Close the transport immediately.

        Buffered data will be lost.  No more data will be received.
        The protocol's connection_lost() method will (eventually) be
        called with None as its argument.
        """
        raise NotImplementedError


class Transport(ReadTransport, WriteTransport):
    """Interface representing a bidirectional transport.

    There may be several implementations, but typically, the user does
    not implement new transports; rather, the platform provides some
    useful transports that are implemented using the platform's best
    practices.

    The user never instantiates a transport directly; they call a
    utility function, passing it a protocol factory and other
    information necessary to create the transport and protocol.  (E.g.
    EventLoop.create_connection() or EventLoop.create_server().)

    The utility function will asynchronously create a transport and a
    protocol and hook them up by calling the protocol's
    connection_made() method, passing it the transport.

    The implementation here raises NotImplemented for every method
    except writelines(), which calls write() in a loop.
    """

    __slots__ = ()


class DatagramTransport(BaseTransport):
    """Interface for datagram (UDP) transports."""

    __slots__ = ()

    def sendto(self, data, addr=None):
        """Send data to the transport.

        This does not block; it buffers the data and arranges for it
        to be sent out asynchronously.
        addr is target socket address.
        If addr is None use target address pointed on transport creation.
        """
        raise NotImplementedError

    def abort(self):
        """Close the transport immediately.

        Buffered data will be lost.  No more data will be received.
        The protocol's connection_lost() method will (eventually) be
        called with None as its argument.
        """
        raise NotImplementedError


class SubprocessTransport(BaseTransport):

    __slots__ = ()

    def get_pid(self):
        """Get subprocess id."""
        raise NotImplementedError

    def get_returncode(self):
        """Get subprocess returncode.

        See also
        http://docs.python.org/3/library/subprocess#subprocess.Popen.returncode
        """
        raise NotImplementedError

    def get_pipe_transport(self, fd):
        """Get transport for pipe with number fd."""
        raise NotImplementedError

    def send_signal(self, signal):
        """Send signal to subprocess.

        See also:
        docs.python.org/3/library/subprocess#subprocess.Popen.send_signal
        """
        raise NotImplementedError

    def terminate(self):
        """Stop the subprocess.

        Alias for close() method.

        On Posix OSs the method sends SIGTERM to the subprocess.
        On Windows the Win32 API function TerminateProcess()
         is called to stop the subprocess.

        See also:
        http://docs.python.org/3/library/subprocess#subprocess.Popen.terminate
        """
        raise NotImplementedError

    def kill(self):
        """Kill the subprocess.

        On Posix OSs the function sends SIGKILL to the subprocess.
        On Windows kill() is an alias for terminate().

        See also:
        http://docs.python.org/3/library/subprocess#subprocess.Popen.kill
        """
        raise NotImplementedError


class _FlowControlMixin(Transport):
    """All the logic for (write) flow control in a mix-in base class.

    The subclass must implement get_write_buffer_size().  It must call
    _maybe_pause_protocol() whenever the write buffer size increases,
    and _maybe_resume_protocol() whenever it decreases.  It may also
    override set_write_buffer_limits() (e.g. to specify different
    defaults).

    The subclass constructor must call super().__init__(extra).  This
    will call set_write_buffer_limits().

    The user may call set_write_buffer_limits() and
    get_write_buffer_size(), and their protocol's pause_writing() and
    resume_writing() may be called.
    """

    __slots__ = ('_loop', '_protocol_paused', '_high_water', '_low_water')

    def __init__(self, extra=None, loop=None):
        super().__init__(extra)
        assert loop is not None
        self._loop = loop
        self._protocol_paused = False
        self._set_write_buffer_limits()

    def _maybe_pause_protocol(self):
        size = self.get_write_buffer_size()
        if size <= self._high_water:
            return
        if not self._protocol_paused:
            self._protocol_paused = True
            try:
                self._protocol.pause_writing()
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                self._loop.call_exception_handler({
                    'message': 'protocol.pause_writing() failed',
                    'exception': exc,
                    'transport': self,
                    'protocol': self._protocol,
                })

    def _maybe_resume_protocol(self):
        if (self._protocol_paused and
                self.get_write_buffer_size() <= self._low_water):
            self._protocol_paused = False
            try:
                self._protocol.resume_writing()
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                self._loop.call_exception_handler({
                    'message': 'protocol.resume_writing() failed',
                    'exception': exc,
                    'transport': self,
                    'protocol': self._protocol,
                })

    def get_write_buffer_limits(self):
        return (self._low_water, self._high_water)

    def _set_write_buffer_limits(self, high=None, low=None):
        if high is None:
            if low is None:
                high = 64 * 1024
            else:
                high = 4 * low
        if low is None:
            low = high // 4

        if not high >= low >= 0:
            raise ValueError(
                f'high ({high!r}) must be >= low ({low!r}) must be >= 0')

        self._high_water = high
        self._low_water = low

    def set_write_buffer_limits(self, high=None, low=None):
        self._set_write_buffer_limits(high=high, low=low)
        self._maybe_pause_protocol()

    def get_write_buffer_size(self):
        raise NotImplementedError
<|endoftext|>import ast
import asyncio
import code
import concurrent.futures
import inspect
import sys
import threading
import types
import warnings

from . import futures


class AsyncIOInteractiveConsole(code.InteractiveConsole):

    def __init__(self, locals, loop):
        super().__init__(locals)
        self.compile.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

        self.loop = loop

    def runcode(self, code):
        future = concurrent.futures.Future()

        def callback():
            global repl_future
            global repl_future_interrupted

            repl_future = None
            repl_future_interrupted = False

            func = types.FunctionType(code, self.locals)
            try:
                coro = func()
            except SystemExit:
                raise
            except KeyboardInterrupt as ex:
                repl_future_interrupted = True
                future.set_exception(ex)
                return
            except BaseException as ex:
                future.set_exception(ex)
                return

            if not inspect.iscoroutine(coro):
                future.set_result(coro)
                return

            try:
                repl_future = self.loop.create_task(coro)
                futures._chain_future(repl_future, future)
            except BaseException as exc:
                future.set_exception(exc)

        loop.call_soon_threadsafe(callback)

        try:
            return future.result()
        except SystemExit:
            raise
        except BaseException:
            if repl_future_interrupted:
                self.write("\nKeyboardInterrupt\n")
            else:
                self.showtraceback()


class REPLThread(threading.Thread):

    def run(self):
        try:
            banner = (
                f'asyncio REPL {sys.version} on {sys.platform}\n'
                f'Use "await" directly instead of "asyncio.run()".\n'
                f'Type "help", "copyright", "credits" or "license" '
                f'for more information.\n'
                f'{getattr(sys, "ps1", ">>> ")}import asyncio'
            )

            console.interact(
                banner=banner,
                exitmsg='exiting asyncio REPL...')
        finally:
            warnings.filterwarnings(
                'ignore',
                message=r'^coroutine .* was never awaited$',
                category=RuntimeWarning)

            loop.call_soon_threadsafe(loop.stop)


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    repl_locals = {'asyncio': asyncio}
    for key in {'__name__', '__package__',
                '__loader__', '__spec__',
                '__builtins__', '__file__'}:
        repl_locals[key] = locals()[key]

    console = AsyncIOInteractiveConsole(repl_locals, loop)

    repl_future = None
    repl_future_interrupted = False

    try:
        import readline  # NoQA
    except ImportError:
        pass

    repl_thread = REPLThread()
    repl_thread.daemon = True
    repl_thread.start()

    while True:
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            if repl_future and not repl_future.done():
                repl_future.cancel()
                repl_future_interrupted = True
            continue
        else:
            break
<|endoftext|>__all__ = 'coroutine', 'iscoroutinefunction', 'iscoroutine'

import collections.abc
import functools
import inspect
import os
import sys
import traceback
import types
import warnings

from . import base_futures
from . import constants
from . import format_helpers
from .log import logger


def _is_debug_mode():
    # If you set _DEBUG to true, @coroutine will wrap the resulting
    # generator objects in a CoroWrapper instance (defined below).  That
    # instance will log a message when the generator is never iterated
    # over, which may happen when you forget to use "await" or "yield from"
    # with a coroutine call.
    # Note that the value of the _DEBUG flag is taken
    # when the decorator is used, so to be of any use it must be set
    # before you define your coroutines.  A downside of using this feature
    # is that tracebacks show entries for the CoroWrapper.__next__ method
    # when _DEBUG is true.
    return sys.flags.dev_mode or (not sys.flags.ignore_environment and
                                  bool(os.environ.get('PYTHONASYNCIODEBUG')))


_DEBUG = _is_debug_mode()


class CoroWrapper:
    # Wrapper for coroutine object in _DEBUG mode.

    def __init__(self, gen, func=None):
        assert inspect.isgenerator(gen) or inspect.iscoroutine(gen), gen
        self.gen = gen
        self.func = func  # Used to unwrap @coroutine decorator
        self._source_traceback = format_helpers.extract_stack(sys._getframe(1))
        self.__name__ = getattr(gen, '__name__', None)
        self.__qualname__ = getattr(gen, '__qualname__', None)

    def __repr__(self):
        coro_repr = _format_coroutine(self)
        if self._source_traceback:
            frame = self._source_traceback[-1]
            coro_repr += f', created at {frame[0]}:{frame[1]}'

        return f'<{self.__class__.__name__} {coro_repr}>'

    def __iter__(self):
        return self

    def __next__(self):
        return self.gen.send(None)

    def send(self, value):
        return self.gen.send(value)

    def throw(self, type, value=None, traceback=None):
        return self.gen.throw(type, value, traceback)

    def close(self):
        return self.gen.close()

    @property
    def gi_frame(self):
        return self.gen.gi_frame

    @property
    def gi_running(self):
        return self.gen.gi_running

    @property
    def gi_code(self):
        return self.gen.gi_code

    def __await__(self):
        return self

    @property
    def gi_yieldfrom(self):
        return self.gen.gi_yieldfrom

    def __del__(self):
        # Be careful accessing self.gen.frame -- self.gen might not exist.
        gen = getattr(self, 'gen', None)
        frame = getattr(gen, 'gi_frame', None)
        if frame is not None and frame.f_lasti == -1:
            msg = f'{self!r} was never yielded from'
            tb = getattr(self, '_source_traceback', ())
            if tb:
                tb = ''.join(traceback.format_list(tb))
                msg += (f'\nCoroutine object created at '
                        f'(most recent call last, truncated to '
                        f'{constants.DEBUG_STACK_DEPTH} last lines):\n')
                msg += tb.rstrip()
            logger.error(msg)


def coroutine(func):
    """Decorator to mark coroutines.

    If the coroutine is not yielded from before it is destroyed,
    an error message is logged.
    """
    warnings.warn('"@coroutine" decorator is deprecated since Python 3.8, use "async def" instead',
                  DeprecationWarning,
                  stacklevel=2)
    if inspect.iscoroutinefunction(func):
        # In Python 3.5 that's all we need to do for coroutines
        # defined with "async def".
        return func

    if inspect.isgeneratorfunction(func):
        coro = func
    else:
        @functools.wraps(func)
        def coro(*args, **kw):
            res = func(*args, **kw)
            if (base_futures.isfuture(res) or inspect.isgenerator(res) or
                    isinstance(res, CoroWrapper)):
                res = yield from res
            else:
                # If 'res' is an awaitable, run it.
                try:
                    await_meth = res.__await__
                except AttributeError:
                    pass
                else:
                    if isinstance(res, collections.abc.Awaitable):
                        res = yield from await_meth()
            return res

    coro = types.coroutine(coro)
    if not _DEBUG:
        wrapper = coro
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            w = CoroWrapper(coro(*args, **kwds), func=func)
            if w._source_traceback:
                del w._source_traceback[-1]
            # Python < 3.5 does not implement __qualname__
            # on generator objects, so we set it manually.
            # We use getattr as some callables (such as
            # functools.partial may lack __qualname__).
            w.__name__ = getattr(func, '__name__', None)
            w.__qualname__ = getattr(func, '__qualname__', None)
            return w

    wrapper._is_coroutine = _is_coroutine  # For iscoroutinefunction().
    return wrapper


# A marker for iscoroutinefunction.
_is_coroutine = object()


def iscoroutinefunction(func):
    """Return True if func is a decorated coroutine function."""
    return (inspect.iscoroutinefunction(func) or
            getattr(func, '_is_coroutine', None) is _is_coroutine)


# Prioritize native coroutine check to speed-up
# asyncio.iscoroutine.
_COROUTINE_TYPES = (types.CoroutineType, types.GeneratorType,
                    collections.abc.Coroutine, CoroWrapper)
_iscoroutine_typecache = set()


def iscoroutine(obj):
    """Return True if obj is a coroutine object."""
    if type(obj) in _iscoroutine_typecache:
        return True

    if isinstance(obj, _COROUTINE_TYPES):
        # Just in case we don't want to cache more than 100
        # positive types.  That shouldn't ever happen, unless
        # someone stressing the system on purpose.
        if len(_iscoroutine_typecache) < 100:
            _iscoroutine_typecache.add(type(obj))
        return True
    else:
        return False


def _format_coroutine(coro):
    assert iscoroutine(coro)

    is_corowrapper = isinstance(coro, CoroWrapper)

    def get_name(coro):
        # Coroutines compiled with Cython sometimes don't have
        # proper __qualname__ or __name__.  While that is a bug
        # in Cython, asyncio shouldn't crash with an AttributeError
        # in its __repr__ functions.
        if is_corowrapper:
            return format_helpers._format_callback(coro.func, (), {})

        if hasattr(coro, '__qualname__') and coro.__qualname__:
            coro_name = coro.__qualname__
        elif hasattr(coro, '__name__') and coro.__name__:
            coro_name = coro.__name__
        else:
            # Stop masking Cython bugs, expose them in a friendly way.
            coro_name = f'<{type(coro).__name__} without __name__>'
        return f'{coro_name}()'

    def is_running(coro):
        try:
            return coro.cr_running
        except AttributeError:
            try:
                return coro.gi_running
            except AttributeError:
                return False

    coro_code = None
    if hasattr(coro, 'cr_code') and coro.cr_code:
        coro_code = coro.cr_code
    elif hasattr(coro, 'gi_code') and coro.gi_code:
        coro_code = coro.gi_code

    coro_name = get_name(coro)

    if not coro_code:
        # Built-in types might not have __qualname__ or __name__.
        if is_running(coro):
            return f'{coro_name} running'
        else:
            return coro_name

    coro_frame = None
    if hasattr(coro, 'gi_frame') and coro.gi_frame:
        coro_frame = coro.gi_frame
    elif hasattr(coro, 'cr_frame') and coro.cr_frame:
        coro_frame = coro.cr_frame

    # If Cython's coroutine has a fake code object without proper
    # co_filename -- expose that.
    filename = coro_code.co_filename or '<empty co_filename>'

    lineno = 0
    if (is_corowrapper and
            coro.func is not None and
            not inspect.isgeneratorfunction(coro.func)):
        source = format_helpers._get_function_source(coro.func)
        if source is not None:
            filename, lineno = source
        if coro_frame is None:
            coro_repr = f'{coro_name} done, defined at {filename}:{lineno}'
        else:
            coro_repr = f'{coro_name} running, defined at {filename}:{lineno}'

    elif coro_frame is not None:
        lineno = coro_frame.f_lineno
        coro_repr = f'{coro_name} running at {filename}:{lineno}'

    else:
        lineno = coro_code.co_firstlineno
        coro_repr = f'{coro_name} done, defined at {filename}:{lineno}'

    return coro_repr
<|endoftext|>"""Event loop mixins."""

import threading
from . import events

_global_lock = threading.Lock()

# Used as a sentinel for loop parameter
_marker = object()


class _LoopBoundMixin:
    _loop = None

    def __init__(self, *, loop=_marker):
        if loop is not _marker:
            raise TypeError(
                f'As of 3.10, the *loop* parameter was removed from '
                f'{type(self).__name__}() since it is no longer necessary'
            )

    def _get_loop(self):
        loop = events._get_running_loop()

        if self._loop is None:
            with _global_lock:
                if self._loop is None:
                    self._loop = loop
        if loop is not self._loop:
            raise RuntimeError(f'{self!r} is bound to a different event loop')
        return loop
<|endoftext|>__all__ = 'create_subprocess_exec', 'create_subprocess_shell'

import subprocess

from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger


PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEVNULL = subprocess.DEVNULL


class SubprocessStreamProtocol(streams.FlowControlMixin,
                               protocols.SubprocessProtocol):
    """Like StreamReaderProtocol, but for a subprocess."""

    def __init__(self, limit, loop):
        super().__init__(loop=loop)
        self._limit = limit
        self.stdin = self.stdout = self.stderr = None
        self._transport = None
        self._process_exited = False
        self._pipe_fds = []
        self._stdin_closed = self._loop.create_future()

    def __repr__(self):
        info = [self.__class__.__name__]
        if self.stdin is not None:
            info.append(f'stdin={self.stdin!r}')
        if self.stdout is not None:
            info.append(f'stdout={self.stdout!r}')
        if self.stderr is not None:
            info.append(f'stderr={self.stderr!r}')
        return '<{}>'.format(' '.join(info))

    def connection_made(self, transport):
        self._transport = transport

        stdout_transport = transport.get_pipe_transport(1)
        if stdout_transport is not None:
            self.stdout = streams.StreamReader(limit=self._limit,
                                               loop=self._loop)
            self.stdout.set_transport(stdout_transport)
            self._pipe_fds.append(1)

        stderr_transport = transport.get_pipe_transport(2)
        if stderr_transport is not None:
            self.stderr = streams.StreamReader(limit=self._limit,
                                               loop=self._loop)
            self.stderr.set_transport(stderr_transport)
            self._pipe_fds.append(2)

        stdin_transport = transport.get_pipe_transport(0)
        if stdin_transport is not None:
            self.stdin = streams.StreamWriter(stdin_transport,
                                              protocol=self,
                                              reader=None,
                                              loop=self._loop)

    def pipe_data_received(self, fd, data):
        if fd == 1:
            reader = self.stdout
        elif fd == 2:
            reader = self.stderr
        else:
            reader = None
        if reader is not None:
            reader.feed_data(data)

    def pipe_connection_lost(self, fd, exc):
        if fd == 0:
            pipe = self.stdin
            if pipe is not None:
                pipe.close()
            self.connection_lost(exc)
            if exc is None:
                self._stdin_closed.set_result(None)
            else:
                self._stdin_closed.set_exception(exc)
            return
        if fd == 1:
            reader = self.stdout
        elif fd == 2:
            reader = self.stderr
        else:
            reader = None
        if reader is not None:
            if exc is None:
                reader.feed_eof()
            else:
                reader.set_exception(exc)

        if fd in self._pipe_fds:
            self._pipe_fds.remove(fd)
        self._maybe_close_transport()

    def process_exited(self):
        self._process_exited = True
        self._maybe_close_transport()

    def _maybe_close_transport(self):
        if len(self._pipe_fds) == 0 and self._process_exited:
            self._transport.close()
            self._transport = None

    def _get_close_waiter(self, stream):
        if stream is self.stdin:
            return self._stdin_closed


class Process:
    def __init__(self, transport, protocol, loop):
        self._transport = transport
        self._protocol = protocol
        self._loop = loop
        self.stdin = protocol.stdin
        self.stdout = protocol.stdout
        self.stderr = protocol.stderr
        self.pid = transport.get_pid()

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.pid}>'

    @property
    def returncode(self):
        return self._transport.get_returncode()

    async def wait(self):
        """Wait until the process exit and return the process return code."""
        return await self._transport._wait()

    def send_signal(self, signal):
        self._transport.send_signal(signal)

    def terminate(self):
        self._transport.terminate()

    def kill(self):
        self._transport.kill()

    async def _feed_stdin(self, input):
        debug = self._loop.get_debug()
        self.stdin.write(input)
        if debug:
            logger.debug(
                '%r communicate: feed stdin (%s bytes)', self, len(input))
        try:
            await self.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            # communicate() ignores BrokenPipeError and ConnectionResetError
            if debug:
                logger.debug('%r communicate: stdin got %r', self, exc)

        if debug:
            logger.debug('%r communicate: close stdin', self)
        self.stdin.close()

    async def _noop(self):
        return None

    async def _read_stream(self, fd):
        transport = self._transport.get_pipe_transport(fd)
        if fd == 2:
            stream = self.stderr
        else:
            assert fd == 1
            stream = self.stdout
        if self._loop.get_debug():
            name = 'stdout' if fd == 1 else 'stderr'
            logger.debug('%r communicate: read %s', self, name)
        output = await stream.read()
        if self._loop.get_debug():
            name = 'stdout' if fd == 1 else 'stderr'
            logger.debug('%r communicate: close %s', self, name)
        transport.close()
        return output

    async def communicate(self, input=None):
        if input is not None:
            stdin = self._feed_stdin(input)
        else:
            stdin = self._noop()
        if self.stdout is not None:
            stdout = self._read_stream(1)
        else:
            stdout = self._noop()
        if self.stderr is not None:
            stderr = self._read_stream(2)
        else:
            stderr = self._noop()
        stdin, stdout, stderr = await tasks.gather(stdin, stdout, stderr)
        await self.wait()
        return (stdout, stderr)


async def create_subprocess_shell(cmd, stdin=None, stdout=None, stderr=None,
                                  limit=streams._DEFAULT_LIMIT, **kwds):
    loop = events.get_running_loop()
    protocol_factory = lambda: SubprocessStreamProtocol(limit=limit,
                                                        loop=loop)
    transport, protocol = await loop.subprocess_shell(
        protocol_factory,
        cmd, stdin=stdin, stdout=stdout,
        stderr=stderr, **kwds)
    return Process(transport, protocol, loop)


async def create_subprocess_exec(program, *args, stdin=None, stdout=None,
                                 stderr=None, limit=streams._DEFAULT_LIMIT,
                                 **kwds):
    loop = events.get_running_loop()
    protocol_factory = lambda: SubprocessStreamProtocol(limit=limit,
                                                        loop=loop)
    transport, protocol = await loop.subprocess_exec(
        protocol_factory,
        program, *args,
        stdin=stdin, stdout=stdout,
        stderr=stderr, **kwds)
    return Process(transport, protocol, loop)
<|endoftext|>import collections
import subprocess
import warnings

from . import protocols
from . import transports
from .log import logger


class BaseSubprocessTransport(transports.SubprocessTransport):

    def __init__(self, loop, protocol, args, shell,
                 stdin, stdout, stderr, bufsize,
                 waiter=None, extra=None, **kwargs):
        super().__init__(extra)
        self._closed = False
        self._protocol = protocol
        self._loop = loop
        self._proc = None
        self._pid = None
        self._returncode = None
        self._exit_waiters = []
        self._pending_calls = collections.deque()
        self._pipes = {}
        self._finished = False

        if stdin == subprocess.PIPE:
            self._pipes[0] = None
        if stdout == subprocess.PIPE:
            self._pipes[1] = None
        if stderr == subprocess.PIPE:
            self._pipes[2] = None

        # Create the child process: set the _proc attribute
        try:
            self._start(args=args, shell=shell, stdin=stdin, stdout=stdout,
                        stderr=stderr, bufsize=bufsize, **kwargs)
        except:
            self.close()
            raise

        self._pid = self._proc.pid
        self._extra['subprocess'] = self._proc

        if self._loop.get_debug():
            if isinstance(args, (bytes, str)):
                program = args
            else:
                program = args[0]
            logger.debug('process %r created: pid %s',
                         program, self._pid)

        self._loop.create_task(self._connect_pipes(waiter))

    def __repr__(self):
        info = [self.__class__.__name__]
        if self._closed:
            info.append('closed')
        if self._pid is not None:
            info.append(f'pid={self._pid}')
        if self._returncode is not None:
            info.append(f'returncode={self._returncode}')
        elif self._pid is not None:
            info.append('running')
        else:
            info.append('not started')

        stdin = self._pipes.get(0)
        if stdin is not None:
            info.append(f'stdin={stdin.pipe}')

        stdout = self._pipes.get(1)
        stderr = self._pipes.get(2)
        if stdout is not None and stderr is stdout:
            info.append(f'stdout=stderr={stdout.pipe}')
        else:
            if stdout is not None:
                info.append(f'stdout={stdout.pipe}')
            if stderr is not None:
                info.append(f'stderr={stderr.pipe}')

        return '<{}>'.format(' '.join(info))

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        raise NotImplementedError

    def set_protocol(self, protocol):
        self._protocol = protocol

    def get_protocol(self):
        return self._protocol

    def is_closing(self):
        return self._closed

    def close(self):
        if self._closed:
            return
        self._closed = True

        for proto in self._pipes.values():
            if proto is None:
                continue
            proto.pipe.close()

        if (self._proc is not None and
                # has the child process finished?
                self._returncode is None and
                # the child process has finished, but the
                # transport hasn't been notified yet?
                self._proc.poll() is None):

            if self._loop.get_debug():
                logger.warning('Close running child process: kill %r', self)

            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

            # Don't clear the _proc reference yet: _post_init() may still run

    def __del__(self, _warn=warnings.warn):
        if not self._closed:
            _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
            self.close()

    def get_pid(self):
        return self._pid

    def get_returncode(self):
        return self._returncode

    def get_pipe_transport(self, fd):
        if fd in self._pipes:
            return self._pipes[fd].pipe
        else:
            return None

    def _check_proc(self):
        if self._proc is None:
            raise ProcessLookupError()

    def send_signal(self, signal):
        self._check_proc()
        self._proc.send_signal(signal)

    def terminate(self):
        self._check_proc()
        self._proc.terminate()

    def kill(self):
        self._check_proc()
        self._proc.kill()

    async def _connect_pipes(self, waiter):
        try:
            proc = self._proc
            loop = self._loop

            if proc.stdin is not None:
                _, pipe = await loop.connect_write_pipe(
                    lambda: WriteSubprocessPipeProto(self, 0),
                    proc.stdin)
                self._pipes[0] = pipe

            if proc.stdout is not None:
                _, pipe = await loop.connect_read_pipe(
                    lambda: ReadSubprocessPipeProto(self, 1),
                    proc.stdout)
                self._pipes[1] = pipe

            if proc.stderr is not None:
                _, pipe = await loop.connect_read_pipe(
                    lambda: ReadSubprocessPipeProto(self, 2),
                    proc.stderr)
                self._pipes[2] = pipe

            assert self._pending_calls is not None

            loop.call_soon(self._protocol.connection_made, self)
            for callback, data in self._pending_calls:
                loop.call_soon(callback, *data)
            self._pending_calls = None
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if waiter is not None and not waiter.cancelled():
                waiter.set_exception(exc)
        else:
            if waiter is not None and not waiter.cancelled():
                waiter.set_result(None)

    def _call(self, cb, *data):
        if self._pending_calls is not None:
            self._pending_calls.append((cb, data))
        else:
            self._loop.call_soon(cb, *data)

    def _pipe_connection_lost(self, fd, exc):
        self._call(self._protocol.pipe_connection_lost, fd, exc)
        self._try_finish()

    def _pipe_data_received(self, fd, data):
        self._call(self._protocol.pipe_data_received, fd, data)

    def _process_exited(self, returncode):
        assert returncode is not None, returncode
        assert self._returncode is None, self._returncode
        if self._loop.get_debug():
            logger.info('%r exited with return code %r', self, returncode)
        self._returncode = returncode
        if self._proc.returncode is None:
            # asyncio uses a child watcher: copy the status into the Popen
            # object. On Python 3.6, it is required to avoid a ResourceWarning.
            self._proc.returncode = returncode
        self._call(self._protocol.process_exited)
        self._try_finish()

        # wake up futures waiting for wait()
        for waiter in self._exit_waiters:
            if not waiter.cancelled():
                waiter.set_result(returncode)
        self._exit_waiters = None

    async def _wait(self):
        """Wait until the process exit and return the process return code.

        This method is a coroutine."""
        if self._returncode is not None:
            return self._returncode

        waiter = self._loop.create_future()
        self._exit_waiters.append(waiter)
        return await waiter

    def _try_finish(self):
        assert not self._finished
        if self._returncode is None:
            return
        if all(p is not None and p.disconnected
               for p in self._pipes.values()):
            self._finished = True
            self._call(self._call_connection_lost, None)

    def _call_connection_lost(self, exc):
        try:
            self._protocol.connection_lost(exc)
        finally:
            self._loop = None
            self._proc = None
            self._protocol = None


class WriteSubprocessPipeProto(protocols.BaseProtocol):

    def __init__(self, proc, fd):
        self.proc = proc
        self.fd = fd
        self.pipe = None
        self.disconnected = False

    def connection_made(self, transport):
        self.pipe = transport

    def __repr__(self):
        return f'<{self.__class__.__name__} fd={self.fd} pipe={self.pipe!r}>'

    def connection_lost(self, exc):
        self.disconnected = True
        self.proc._pipe_connection_lost(self.fd, exc)
        self.proc = None

    def pause_writing(self):
        self.proc._protocol.pause_writing()

    def resume_writing(self):
        self.proc._protocol.resume_writing()


class ReadSubprocessPipeProto(WriteSubprocessPipeProto,
                              protocols.Protocol):

    def data_received(self, data):
        self.proc._pipe_data_received(self.fd, data)
<|endoftext|>"""asyncio exceptions."""


__all__ = ('CancelledError', 'InvalidStateError', 'TimeoutError',
           'IncompleteReadError', 'LimitOverrunError',
           'SendfileNotAvailableError')


class CancelledError(BaseException):
    """The Future or Task was cancelled."""


class TimeoutError(Exception):
    """The operation exceeded the given deadline."""


class InvalidStateError(Exception):
    """The operation is not allowed in this state."""


class SendfileNotAvailableError(RuntimeError):
    """Sendfile syscall is not available.

    Raised if OS does not support sendfile syscall for given socket or
    file type.
    """


class IncompleteReadError(EOFError):
    """
    Incomplete read error. Attributes:

    - partial: read bytes string before the end of stream was reached
    - expected: total number of expected bytes (or None if unknown)
    """
    def __init__(self, partial, expected):
        r_expected = 'undefined' if expected is None else repr(expected)
        super().__init__(f'{len(partial)} bytes read on a total of '
                         f'{r_expected} expected bytes')
        self.partial = partial
        self.expected = expected

    def __reduce__(self):
        return type(self), (self.partial, self.expected)


class LimitOverrunError(Exception):
    """Reached the buffer limit while looking for a separator.

    Attributes:
    - consumed: total number of to be consumed bytes.
    """
    def __init__(self, message, consumed):
        super().__init__(message)
        self.consumed = consumed

    def __reduce__(self):
        return type(self), (self.args[0], self.consumed)
<|endoftext|>""" Routines for manipulating RFC2047 encoded words.

This is currently a package-private API, but will be considered for promotion
to a public API if there is demand.

"""

# An ecoded word looks like this:
#
#        =?charset[*lang]?cte?encoded_string?=
#
# for more information about charset see the charset module.  Here it is one
# of the preferred MIME charset names (hopefully; you never know when parsing).
# cte (Content Transfer Encoding) is either 'q' or 'b' (ignoring case).  In
# theory other letters could be used for other encodings, but in practice this
# (almost?) never happens.  There could be a public API for adding entries
# to the CTE tables, but YAGNI for now.  'q' is Quoted Printable, 'b' is
# Base64.  The meaning of encoded_string should be obvious.  'lang' is optional
# as indicated by the brackets (they are not part of the syntax) but is almost
# never encountered in practice.
#
# The general interface for a CTE decoder is that it takes the encoded_string
# as its argument, and returns a tuple (cte_decoded_string, defects).  The
# cte_decoded_string is the original binary that was encoded using the
# specified cte.  'defects' is a list of MessageDefect instances indicating any
# problems encountered during conversion.  'charset' and 'lang' are the
# corresponding strings extracted from the EW, case preserved.
#
# The general interface for a CTE encoder is that it takes a binary sequence
# as input and returns the cte_encoded_string, which is an ascii-only string.
#
# Each decoder must also supply a length function that takes the binary
# sequence as its argument and returns the length of the resulting encoded
# string.
#
# The main API functions for the module are decode, which calls the decoder
# referenced by the cte specifier, and encode, which adds the appropriate
# RFC 2047 "chrome" to the encoded string, and can optionally automatically
# select the shortest possible encoding.  See their docstrings below for
# details.

import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors

__all__ = ['decode_q',
           'encode_q',
           'decode_b',
           'encode_b',
           'len_q',
           'len_b',
           'decode',
           'encode',
           ]

#
# Quoted Printable
#

# regex based decoder.
_q_byte_subber = functools.partial(re.compile(br'=([a-fA-F0-9]{2})').sub,
        lambda m: bytes.fromhex(m.group(1).decode()))

def decode_q(encoded):
    encoded = encoded.replace(b'_', b' ')
    return _q_byte_subber(encoded), []


# dict mapping bytes to their encoded form
class _QByteMap(dict):

    safe = b'-!*+/' + ascii_letters.encode('ascii') + digits.encode('ascii')

    def __missing__(self, key):
        if key in self.safe:
            self[key] = chr(key)
        else:
            self[key] = "={:02X}".format(key)
        return self[key]

_q_byte_map = _QByteMap()

# In headers spaces are mapped to '_'.
_q_byte_map[ord(' ')] = '_'

def encode_q(bstring):
    return ''.join(_q_byte_map[x] for x in bstring)

def len_q(bstring):
    return sum(len(_q_byte_map[x]) for x in bstring)


#
# Base64
#

def decode_b(encoded):
    # First try encoding with validate=True, fixing the padding if needed.
    # This will succeed only if encoded includes no invalid characters.
    pad_err = len(encoded) % 4
    missing_padding = b'==='[:4-pad_err] if pad_err else b''
    try:
        return (
            base64.b64decode(encoded + missing_padding, validate=True),
            [errors.InvalidBase64PaddingDefect()] if pad_err else [],
        )
    except binascii.Error:
        # Since we had correct padding, this is likely an invalid char error.
        #
        # The non-alphabet characters are ignored as far as padding
        # goes, but we don't know how many there are.  So try without adding
        # padding to see if it works.
        try:
            return (
                base64.b64decode(encoded, validate=False),
                [errors.InvalidBase64CharactersDefect()],
            )
        except binascii.Error:
            # Add as much padding as could possibly be necessary (extra padding
            # is ignored).
            try:
                return (
                    base64.b64decode(encoded + b'==', validate=False),
                    [errors.InvalidBase64CharactersDefect(),
                     errors.InvalidBase64PaddingDefect()],
                )
            except binascii.Error:
                # This only happens when the encoded string's length is 1 more
                # than a multiple of 4, which is invalid.
                #
                # bpo-27397: Just return the encoded string since there's no
                # way to decode.
                return encoded, [errors.InvalidBase64LengthDefect()]

def encode_b(bstring):
    return base64.b64encode(bstring).decode('ascii')

def len_b(bstring):
    groups_of_3, leftover = divmod(len(bstring), 3)
    # 4 bytes out for each 3 bytes (or nonzero fraction thereof) in.
    return groups_of_3 * 4 + (4 if leftover else 0)


_cte_decoders = {
    'q': decode_q,
    'b': decode_b,
    }

def decode(ew):
    """Decode encoded word and return (string, charset, lang, defects) tuple.

    An RFC 2047/2243 encoded word has the form:

        =?charset*lang?cte?encoded_string?=

    where '*lang' may be omitted but the other parts may not be.

    This function expects exactly such a string (that is, it does not check the
    syntax and may raise errors if the string is not well formed), and returns
    the encoded_string decoded first from its Content Transfer Encoding and
    then from the resulting bytes into unicode using the specified charset.  If
    the cte-decoded string does not successfully decode using the specified
    character set, a defect is added to the defects list and the unknown octets
    are replaced by the unicode 'unknown' character \\uFDFF.

    The specified charset and language are returned.  The default for language,
    which is rarely if ever encountered, is the empty string.

    """
    _, charset, cte, cte_string, _ = ew.split('?')
    charset, _, lang = charset.partition('*')
    cte = cte.lower()
    # Recover the original bytes and do CTE decoding.
    bstring = cte_string.encode('ascii', 'surrogateescape')
    bstring, defects = _cte_decoders[cte](bstring)
    # Turn the CTE decoded bytes into unicode.
    try:
        string = bstring.decode(charset)
    except UnicodeDecodeError:
        defects.append(errors.UndecodableBytesDefect("Encoded word "
            f"contains bytes not decodable using {charset!r} charset"))
        string = bstring.decode(charset, 'surrogateescape')
    except (LookupError, UnicodeEncodeError):
        string = bstring.decode('ascii', 'surrogateescape')
        if charset.lower() != 'unknown-8bit':
            defects.append(errors.CharsetError(f"Unknown charset {charset!r} "
                f"in encoded word; decoded as unknown bytes"))
    return string, charset, lang, defects


_cte_encoders = {
    'q': encode_q,
    'b': encode_b,
    }

_cte_encode_length = {
    'q': len_q,
    'b': len_b,
    }

def encode(string, charset='utf-8', encoding=None, lang=''):
    """Encode string using the CTE encoding that produces the shorter result.

    Produces an RFC 2047/2243 encoded word of the form:

        =?charset*lang?cte?encoded_string?=

    where '*lang' is omitted unless the 'lang' parameter is given a value.
    Optional argument charset (defaults to utf-8) specifies the charset to use
    to encode the string to binary before CTE encoding it.  Optional argument
    'encoding' is the cte specifier for the encoding that should be used ('q'
    or 'b'); if it is None (the default) the encoding which produces the
    shortest encoded sequence is used, except that 'q' is preferred if it is up
    to five characters longer.  Optional argument 'lang' (default '') gives the
    RFC 2243 language string to specify in the encoded word.

    """
    if charset == 'unknown-8bit':
        bstring = string.encode('ascii', 'surrogateescape')
    else:
        bstring = string.encode(charset)
    if encoding is None:
        qlen = _cte_encode_length['q'](bstring)
        blen = _cte_encode_length['b'](bstring)
        # Bias toward q.  5 is arbitrary.
        encoding = 'q' if qlen - blen < 5 else 'b'
    encoded = _cte_encoders[encoding](bstring)
    if lang:
        lang = '*' + lang
    return "=?{}{}?{}?{}?=".format(charset, lang, encoding, encoded)
<|endoftext|># Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""email package exception classes."""


class MessageError(Exception):
    """Base class for errors in the email package."""


class MessageParseError(MessageError):
    """Base class for message parsing errors."""


class HeaderParseError(MessageParseError):
    """Error while parsing headers."""


class BoundaryError(MessageParseError):
    """Couldn't find terminating boundary."""


class MultipartConversionError(MessageError, TypeError):
    """Conversion to a multipart is prohibited."""


class CharsetError(MessageError):
    """An illegal charset was given."""


class HeaderWriteError(MessageError):
    """Error while writing headers."""


# These are parsing defects which the parser was able to work around.
class MessageDefect(ValueError):
    """Base class for a message defect."""

    def __init__(self, line=None):
        if line is not None:
            super().__init__(line)
        self.line = line

class NoBoundaryInMultipartDefect(MessageDefect):
    """A message claimed to be a multipart but had no boundary parameter."""

class StartBoundaryNotFoundDefect(MessageDefect):
    """The claimed start boundary was never found."""

class CloseBoundaryNotFoundDefect(MessageDefect):
    """A start boundary was found, but not the corresponding close boundary."""

class FirstHeaderLineIsContinuationDefect(MessageDefect):
    """A message had a continuation line as its first header line."""

class MisplacedEnvelopeHeaderDefect(MessageDefect):
    """A 'Unix-from' header was found in the middle of a header block."""

class MissingHeaderBodySeparatorDefect(MessageDefect):
    """Found line with no leading whitespace and no colon before blank line."""
# XXX: backward compatibility, just in case (it was never emitted).
MalformedHeaderDefect = MissingHeaderBodySeparatorDefect

class MultipartInvariantViolationDefect(MessageDefect):
    """A message claimed to be a multipart but no subparts were found."""

class InvalidMultipartContentTransferEncodingDefect(MessageDefect):
    """An invalid content transfer encoding was set on the multipart itself."""

class UndecodableBytesDefect(MessageDefect):
    """Header contained bytes that could not be decoded"""

class InvalidBase64PaddingDefect(MessageDefect):
    """base64 encoded sequence had an incorrect length"""

class InvalidBase64CharactersDefect(MessageDefect):
    """base64 encoded sequence had characters not in base64 alphabet"""

class InvalidBase64LengthDefect(MessageDefect):
    """base64 encoded sequence had invalid length (1 mod 4)"""

# These errors are specific to header parsing.

class HeaderDefect(MessageDefect):
    """Base class for a header defect."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

class InvalidHeaderDefect(HeaderDefect):
    """Header is not valid, message gives details."""

class HeaderMissingRequiredValue(HeaderDefect):
    """A header that must have a value had none"""

class NonPrintableDefect(HeaderDefect):
    """ASCII characters outside the ascii-printable range found"""

    def __init__(self, non_printables):
        super().__init__(non_printables)
        self.non_printables = non_printables

    def __str__(self):
        return ("the following ASCII non-printables found in header: "
            "{}".format(self.non_printables))

class ObsoleteHeaderDefect(HeaderDefect):
    """Header uses syntax declared obsolete by RFC 5322"""

class NonASCIILocalPartDefect(HeaderDefect):
    """local_part contains non-ASCII characters"""
    # This defect only occurs during unicode parsing, not when
    # parsing messages decoded from binary.

class InvalidDateDefect(HeaderDefect):
    """Header has unparsable or invalid date"""
<|endoftext|># Copyright (C) 2001-2007 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Basic message object for the email package object model."""

__all__ = ['Message', 'EmailMessage']

import re
import uu
import quopri
from io import BytesIO, StringIO

# Intrapackage imports
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
Charset = _charset.Charset

SEMISPACE = '; '

# Regular expression that matches `special' characters in parameters, the
# existence of which force quoting of the parameter value.
tspecials = re.compile(r'[ \(\)<>@,;:\\"/\[\]\?=]')


def _splitparam(param):
    # Split header parameters.  BAW: this may be too simple.  It isn't
    # strictly RFC 2045 (section 5.1) compliant, but it catches most headers
    # found in the wild.  We may eventually need a full fledged parser.
    # RDM: we might have a Header here; for now just stringify it.
    a, sep, b = str(param).partition(';')
    if not sep:
        return a.strip(), None
    return a.strip(), b.strip()

def _formatparam(param, value=None, quote=True):
    """Convenience function to format and return a key=value pair.

    This will quote the value if needed or if quote is true.  If value is a
    three tuple (charset, language, value), it will be encoded according
    to RFC2231 rules.  If it contains non-ascii characters it will likewise
    be encoded according to RFC2231 rules, using the utf-8 charset and
    a null language.
    """
    if value is not None and len(value) > 0:
        # A tuple is used for RFC 2231 encoded parameter values where items
        # are (charset, language, value).  charset is a string, not a Charset
        # instance.  RFC 2231 encoded values are never quoted, per RFC.
        if isinstance(value, tuple):
            # Encode as per RFC 2231
            param += '*'
            value = utils.encode_rfc2231(value[2], value[0], value[1])
            return '%s=%s' % (param, value)
        else:
            try:
                value.encode('ascii')
            except UnicodeEncodeError:
                param += '*'
                value = utils.encode_rfc2231(value, 'utf-8', '')
                return '%s=%s' % (param, value)
        # BAW: Please check this.  I think that if quote is set it should
        # force quoting even if not necessary.
        if quote or tspecials.search(value):
            return '%s="%s"' % (param, utils.quote(value))
        else:
            return '%s=%s' % (param, value)
    else:
        return param

def _parseparam(s):
    # RDM This might be a Header, so for now stringify it.
    s = ';' + str(s)
    plist = []
    while s[:1] == ';':
        s = s[1:]
        end = s.find(';')
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(';', end + 1)
        if end < 0:
            end = len(s)
        f = s[:end]
        if '=' in f:
            i = f.index('=')
            f = f[:i].strip().lower() + '=' + f[i+1:].strip()
        plist.append(f.strip())
        s = s[end:]
    return plist


def _unquotevalue(value):
    # This is different than utils.collapse_rfc2231_value() because it doesn't
    # try to convert the value to a unicode.  Message.get_param() and
    # Message.get_params() are both currently defined to return the tuple in
    # the face of RFC 2231 parameters.
    if isinstance(value, tuple):
        return value[0], value[1], utils.unquote(value[2])
    else:
        return utils.unquote(value)



class Message:
    """Basic message object.

    A message object is defined as something that has a bunch of RFC 2822
    headers and a payload.  It may optionally have an envelope header
    (a.k.a. Unix-From or From_ header).  If the message is a container (i.e. a
    multipart or a message/rfc822), then the payload is a list of Message
    objects, otherwise it is a string.

    Message objects implement part of the `mapping' interface, which assumes
    there is exactly one occurrence of the header per message.  Some headers
    do in fact appear multiple times (e.g. Received) and for those headers,
    you must use the explicit API to set or get all the headers.  Not all of
    the mapping methods are implemented.
    """
    def __init__(self, policy=compat32):
        self.policy = policy
        self._headers = []
        self._unixfrom = None
        self._payload = None
        self._charset = None
        # Defaults for multipart messages
        self.preamble = self.epilogue = None
        self.defects = []
        # Default content type
        self._default_type = 'text/plain'

    def __str__(self):
        """Return the entire formatted message as a string.
        """
        return self.as_string()

    def as_string(self, unixfrom=False, maxheaderlen=0, policy=None):
        """Return the entire formatted message as a string.

        Optional 'unixfrom', when true, means include the Unix From_ envelope
        header.  For backward compatibility reasons, if maxheaderlen is
        not specified it defaults to 0, so you must override it explicitly
        if you want a different maxheaderlen.  'policy' is passed to the
        Generator instance used to serialize the message; if it is not
        specified the policy associated with the message instance is used.

        If the message object contains binary data that is not encoded
        according to RFC standards, the non-compliant data will be replaced by
        unicode "unknown character" code points.
        """
        from email.generator import Generator
        policy = self.policy if policy is None else policy
        fp = StringIO()
        g = Generator(fp,
                      mangle_from_=False,
                      maxheaderlen=maxheaderlen,
                      policy=policy)
        g.flatten(self, unixfrom=unixfrom)
        return fp.getvalue()

    def __bytes__(self):
        """Return the entire formatted message as a bytes object.
        """
        return self.as_bytes()

    def as_bytes(self, unixfrom=False, policy=None):
        """Return the entire formatted message as a bytes object.

        Optional 'unixfrom', when true, means include the Unix From_ envelope
        header.  'policy' is passed to the BytesGenerator instance used to
        serialize the message; if not specified the policy associated with
        the message instance is used.
        """
        from email.generator import BytesGenerator
        policy = self.policy if policy is None else policy
        fp = BytesIO()
        g = BytesGenerator(fp, mangle_from_=False, policy=policy)
        g.flatten(self, unixfrom=unixfrom)
        return fp.getvalue()

    def is_multipart(self):
        """Return True if the message consists of multiple parts."""
        return isinstance(self._payload, list)

    #
    # Unix From_ line
    #
    def set_unixfrom(self, unixfrom):
        self._unixfrom = unixfrom

    def get_unixfrom(self):
        return self._unixfrom

    #
    # Payload manipulation.
    #
    def attach(self, payload):
        """Add the given payload to the current payload.

        The current payload will always be a list of objects after this method
        is called.  If you want to set the payload to a scalar object, use
        set_payload() instead.
        """
        if self._payload is None:
            self._payload = [payload]
        else:
            try:
                self._payload.append(payload)
            except AttributeError:
                raise TypeError("Attach is not valid on a message with a"
                                " non-multipart payload")

    def get_payload(self, i=None, decode=False):
        """Return a reference to the payload.

        The payload will either be a list object or a string.  If you mutate
        the list object, you modify the message's payload in place.  Optional
        i returns that index into the payload.

        Optional decode is a flag indicating whether the payload should be
        decoded or not, according to the Content-Transfer-Encoding header
        (default is False).

        When True and the message is not a multipart, the payload will be
        decoded if this header's value is `quoted-printable' or `base64'.  If
        some other encoding is used, or the header is missing, or if the
        payload has bogus data (i.e. bogus base64 or uuencoded data), the
        payload is returned as-is.

        If the message is a multipart and the decode flag is True, then None
        is returned.
        """
        # Here is the logic table for this code, based on the email5.0.0 code:
        #   i     decode  is_multipart  result
        # ------  ------  ------------  ------------------------------
        #  None   True    True          None
        #   i     True    True          None
        #  None   False   True          _payload (a list)
        #   i     False   True          _payload element i (a Message)
        #   i     False   False         error (not a list)
        #   i     True    False         error (not a list)
        #  None   False   False         _payload
        #  None   True    False         _payload decoded (bytes)
        # Note that Barry planned to factor out the 'decode' case, but that
        # isn't so easy now that we handle the 8 bit data, which needs to be
        # converted in both the decode and non-decode path.
        if self.is_multipart():
            if decode:
                return None
            if i is None:
                return self._payload
            else:
                return self._payload[i]
        # For backward compatibility, Use isinstance and this error message
        # instead of the more logical is_multipart test.
        if i is not None and not isinstance(self._payload, list):
            raise TypeError('Expected list, got %s' % type(self._payload))
        payload = self._payload
        # cte might be a Header, so for now stringify it.
        cte = str(self.get('content-transfer-encoding', '')).lower()
        # payload may be bytes here.
        if isinstance(payload, str):
            if utils._has_surrogates(payload):
                bpayload = payload.encode('ascii', 'surrogateescape')
                if not decode:
                    try:
                        payload = bpayload.decode(self.get_param('charset', 'ascii'), 'replace')
                    except LookupError:
                        payload = bpayload.decode('ascii', 'replace')
            elif decode:
                try:
                    bpayload = payload.encode('ascii')
                except UnicodeError:
                    # This won't happen for RFC compliant messages (messages
                    # containing only ASCII code points in the unicode input).
                    # If it does happen, turn the string into bytes in a way
                    # guaranteed not to fail.
                    bpayload = payload.encode('raw-unicode-escape')
        if not decode:
            return payload
        if cte == 'quoted-printable':
            return quopri.decodestring(bpayload)
        elif cte == 'base64':
            # XXX: this is a bit of a hack; decode_b should probably be factored
            # out somewhere, but I haven't figured out where yet.
            value, defects = decode_b(b''.join(bpayload.splitlines()))
            for defect in defects:
                self.policy.handle_defect(self, defect)
            return value
        elif cte in ('x-uuencode', 'uuencode', 'uue', 'x-uue'):
            in_file = BytesIO(bpayload)
            out_file = BytesIO()
            try:
                uu.decode(in_file, out_file, quiet=True)
                return out_file.getvalue()
            except uu.Error:
                # Some decoding problem
                return bpayload
        if isinstance(payload, str):
            return bpayload
        return payload

    def set_payload(self, payload, charset=None):
        """Set the payload to the given value.

        Optional charset sets the message's default character set.  See
        set_charset() for details.
        """
        if hasattr(payload, 'encode'):
            if charset is None:
                self._payload = payload
                return
            if not isinstance(charset, Charset):
                charset = Charset(charset)
            payload = payload.encode(charset.output_charset)
        if hasattr(payload, 'decode'):
            self._payload = payload.decode('ascii', 'surrogateescape')
        else:
            self._payload = payload
        if charset is not None:
            self.set_charset(charset)

    def set_charset(self, charset):
        """Set the charset of the payload to a given character set.

        charset can be a Charset instance, a string naming a character set, or
        None.  If it is a string it will be converted to a Charset instance.
        If charset is None, the charset parameter will be removed from the
        Content-Type field.  Anything else will generate a TypeError.

        The message will be assumed to be of type text/* encoded with
        charset.input_charset.  It will be converted to charset.output_charset
        and encoded properly, if needed, when generating the plain text
        representation of the message.  MIME headers (MIME-Version,
        Content-Type, Content-Transfer-Encoding) will be added as needed.
        """
        if charset is None:
            self.del_param('charset')
            self._charset = None
            return
        if not isinstance(charset, Charset):
            charset = Charset(charset)
        self._charset = charset
        if 'MIME-Version' not in self:
            self.add_header('MIME-Version', '1.0')
        if 'Content-Type' not in self:
            self.add_header('Content-Type', 'text/plain',
                            charset=charset.get_output_charset())
        else:
            self.set_param('charset', charset.get_output_charset())
        if charset != charset.get_output_charset():
            self._payload = charset.body_encode(self._payload)
        if 'Content-Transfer-Encoding' not in self:
            cte = charset.get_body_encoding()
            try:
                cte(self)
            except TypeError:
                # This 'if' is for backward compatibility, it allows unicode
                # through even though that won't work correctly if the
                # message is serialized.
                payload = self._payload
                if payload:
                    try:
                        payload = payload.encode('ascii', 'surrogateescape')
                    except UnicodeError:
                        payload = payload.encode(charset.output_charset)
                self._payload = charset.body_encode(payload)
                self.add_header('Content-Transfer-Encoding', cte)

    def get_charset(self):
        """Return the Charset instance associated with the message's payload.
        """
        return self._charset

    #
    # MAPPING INTERFACE (partial)
    #
    def __len__(self):
        """Return the total number of headers, including duplicates."""
        return len(self._headers)

    def __getitem__(self, name):
        """Get a header value.

        Return None if the header is missing instead of raising an exception.

        Note that if the header appeared multiple times, exactly which
        occurrence gets returned is undefined.  Use get_all() to get all
        the values matching a header field name.
        """
        return self.get(name)

    def __setitem__(self, name, val):
        """Set the value of a header.

        Note: this does not overwrite an existing header with the same field
        name.  Use __delitem__() first to delete any existing headers.
        """
        max_count = self.policy.header_max_count(name)
        if max_count:
            lname = name.lower()
            found = 0
            for k, v in self._headers:
                if k.lower() == lname:
                    found += 1
                    if found >= max_count:
                        raise ValueError("There may be at most {} {} headers "
                                         "in a message".format(max_count, name))
        self._headers.append(self.policy.header_store_parse(name, val))

    def __delitem__(self, name):
        """Delete all occurrences of a header, if present.

        Does not raise an exception if the header is missing.
        """
        name = name.lower()
        newheaders = []
        for k, v in self._headers:
            if k.lower() != name:
                newheaders.append((k, v))
        self._headers = newheaders

    def __contains__(self, name):
        return name.lower() in [k.lower() for k, v in self._headers]

    def __iter__(self):
        for field, value in self._headers:
            yield field

    def keys(self):
        """Return a list of all the message's header field names.

        These will be sorted in the order they appeared in the original
        message, or were added to the message, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [k for k, v in self._headers]

    def values(self):
        """Return a list of all the message's header values.

        These will be sorted in the order they appeared in the original
        message, or were added to the message, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [self.policy.header_fetch_parse(k, v)
                for k, v in self._headers]

    def items(self):
        """Get all the message's header fields and values.

        These will be sorted in the order they appeared in the original
        message, or were added to the message, and may contain duplicates.
        Any fields deleted and re-inserted are always appended to the header
        list.
        """
        return [(k, self.policy.header_fetch_parse(k, v))
                for k, v in self._headers]

    def get(self, name, failobj=None):
        """Get a header value.

        Like __getitem__() but return failobj instead of None when the field
        is missing.
        """
        name = name.lower()
        for k, v in self._headers:
            if k.lower() == name:
                return self.policy.header_fetch_parse(k, v)
        return failobj

    #
    # "Internal" methods (public API, but only intended for use by a parser
    # or generator, not normal application code.
    #

    def set_raw(self, name, value):
        """Store name and value in the model without modification.

        This is an "internal" API, intended only for use by a parser.
        """
        self._headers.append((name, value))

    def raw_items(self):
        """Return the (name, value) header pairs without modification.

        This is an "internal" API, intended only for use by a generator.
        """
        return iter(self._headers.copy())

    #
    # Additional useful stuff
    #

    def get_all(self, name, failobj=None):
        """Return a list of all the values for the named field.

        These will be sorted in the order they appeared in the original
        message, and may contain duplicates.  Any fields deleted and
        re-inserted are always appended to the header list.

        If no such fields exist, failobj is returned (defaults to None).
        """
        values = []
        name = name.lower()
        for k, v in self._headers:
            if k.lower() == name:
                values.append(self.policy.header_fetch_parse(k, v))
        if not values:
            return failobj
        return values

    def add_header(self, _name, _value, **_params):
        """Extended header setting.

        name is the header field to add.  keyword arguments can be used to set
        additional parameters for the header field, with underscores converted
        to dashes.  Normally the parameter will be added as key="value" unless
        value is None, in which case only the key will be added.  If a
        parameter value contains non-ASCII characters it can be specified as a
        three-tuple of (charset, language, value), in which case it will be
        encoded according to RFC2231 rules.  Otherwise it will be encoded using
        the utf-8 charset and a language of ''.

        Examples:

        msg.add_header('content-disposition', 'attachment', filename='bud.gif')
        msg.add_header('content-disposition', 'attachment',
                       filename=('utf-8', '', Fußballer.ppt'))
        msg.add_header('content-disposition', 'attachment',
                       filename='Fußballer.ppt'))
        """
        parts = []
        for k, v in _params.items():
            if v is None:
                parts.append(k.replace('_', '-'))
            else:
                parts.append(_formatparam(k.replace('_', '-'), v))
        if _value is not None:
            parts.insert(0, _value)
        self[_name] = SEMISPACE.join(parts)

    def replace_header(self, _name, _value):
        """Replace a header.

        Replace the first matching header found in the message, retaining
        header order and case.  If no matching header was found, a KeyError is
        raised.
        """
        _name = _name.lower()
        for i, (k, v) in zip(range(len(self._headers)), self._headers):
            if k.lower() == _name:
                self._headers[i] = self.policy.header_store_parse(k, _value)
                break
        else:
            raise KeyError(_name)

    #
    # Use these three methods instead of the three above.
    #

    def get_content_type(self):
        """Return the message's content type.

        The returned string is coerced to lower case of the form
        `maintype/subtype'.  If there was no Content-Type header in the
        message, the default type as given by get_default_type() will be
        returned.  Since according to RFC 2045, messages always have a default
        type this will always return a value.

        RFC 2045 defines a message's default type to be text/plain unless it
        appears inside a multipart/digest container, in which case it would be
        message/rfc822.
        """
        missing = object()
        value = self.get('content-type', missing)
        if value is missing:
            # This should have no parameters
            return self.get_default_type()
        ctype = _splitparam(value)[0].lower()
        # RFC 2045, section 5.2 says if its invalid, use text/plain
        if ctype.count('/') != 1:
            return 'text/plain'
        return ctype

    def get_content_maintype(self):
        """Return the message's main content type.

        This is the `maintype' part of the string returned by
        get_content_type().
        """
        ctype = self.get_content_type()
        return ctype.split('/')[0]

    def get_content_subtype(self):
        """Returns the message's sub-content type.

        This is the `subtype' part of the string returned by
        get_content_type().
        """
        ctype = self.get_content_type()
        return ctype.split('/')[1]

    def get_default_type(self):
        """Return the `default' content type.

        Most messages have a default content type of text/plain, except for
        messages that are subparts of multipart/digest containers.  Such
        subparts have a default content type of message/rfc822.
        """
        return self._default_type

    def set_default_type(self, ctype):
        """Set the `default' content type.

        ctype should be either "text/plain" or "message/rfc822", although this
        is not enforced.  The default content type is not stored in the
        Content-Type header.
        """
        self._default_type = ctype

    def _get_params_preserve(self, failobj, header):
        # Like get_params() but preserves the quoting of values.  BAW:
        # should this be part of the public interface?
        missing = object()
        value = self.get(header, missing)
        if value is missing:
            return failobj
        params = []
        for p in _parseparam(value):
            try:
                name, val = p.split('=', 1)
                name = name.strip()
                val = val.strip()
            except ValueError:
                # Must have been a bare attribute
                name = p.strip()
                val = ''
            params.append((name, val))
        params = utils.decode_params(params)
        return params

    def get_params(self, failobj=None, header='content-type', unquote=True):
        """Return the message's Content-Type parameters, as a list.

        The elements of the returned list are 2-tuples of key/value pairs, as
        split on the `=' sign.  The left hand side of the `=' is the key,
        while the right hand side is the value.  If there is no `=' sign in
        the parameter the value is the empty string.  The value is as
        described in the get_param() method.

        Optional failobj is the object to return if there is no Content-Type
        header.  Optional header is the header to search instead of
        Content-Type.  If unquote is True, the value is unquoted.
        """
        missing = object()
        params = self._get_params_preserve(missing, header)
        if params is missing:
            return failobj
        if unquote:
            return [(k, _unquotevalue(v)) for k, v in params]
        else:
            return params

    def get_param(self, param, failobj=None, header='content-type',
                  unquote=True):
        """Return the parameter value if found in the Content-Type header.

        Optional failobj is the object to return if there is no Content-Type
        header, or the Content-Type header has no such parameter.  Optional
        header is the header to search instead of Content-Type.

        Parameter keys are always compared case insensitively.  The return
        value can either be a string, or a 3-tuple if the parameter was RFC
        2231 encoded.  When it's a 3-tuple, the elements of the value are of
        the form (CHARSET, LANGUAGE, VALUE).  Note that both CHARSET and
        LANGUAGE can be None, in which case you should consider VALUE to be
        encoded in the us-ascii charset.  You can usually ignore LANGUAGE.
        The parameter value (either the returned string, or the VALUE item in
        the 3-tuple) is always unquoted, unless unquote is set to False.

        If your application doesn't care whether the parameter was RFC 2231
        encoded, it can turn the return value into a string as follows:

            rawparam = msg.get_param('foo')
            param = email.utils.collapse_rfc2231_value(rawparam)

        """
        if header not in self:
            return failobj
        for k, v in self._get_params_preserve(failobj, header):
            if k.lower() == param.lower():
                if unquote:
                    return _unquotevalue(v)
                else:
                    return v
        return failobj

    def set_param(self, param, value, header='Content-Type', requote=True,
                  charset=None, language='', replace=False):
        """Set a parameter in the Content-Type header.

        If the parameter already exists in the header, its value will be
        replaced with the new value.

        If header is Content-Type and has not yet been defined for this
        message, it will be set to "text/plain" and the new parameter and
        value will be appended as per RFC 2045.

        An alternate header can be specified in the header argument, and all
        parameters will be quoted as necessary unless requote is False.

        If charset is specified, the parameter will be encoded according to RFC
        2231.  Optional language specifies the RFC 2231 language, defaulting
        to the empty string.  Both charset and language should be strings.
        """
        if not isinstance(value, tuple) and charset:
            value = (charset, language, value)

        if header not in self and header.lower() == 'content-type':
            ctype = 'text/plain'
        else:
            ctype = self.get(header)
        if not self.get_param(param, header=header):
            if not ctype:
                ctype = _formatparam(param, value, requote)
            else:
                ctype = SEMISPACE.join(
                    [ctype, _formatparam(param, value, requote)])
        else:
            ctype = ''
            for old_param, old_value in self.get_params(header=header,
                                                        unquote=requote):
                append_param = ''
                if old_param.lower() == param.lower():
                    append_param = _formatparam(param, value, requote)
                else:
                    append_param = _formatparam(old_param, old_value, requote)
                if not ctype:
                    ctype = append_param
                else:
                    ctype = SEMISPACE.join([ctype, append_param])
        if ctype != self.get(header):
            if replace:
                self.replace_header(header, ctype)
            else:
                del self[header]
                self[header] = ctype

    def del_param(self, param, header='content-type', requote=True):
        """Remove the given parameter completely from the Content-Type header.

        The header will be re-written in place without the parameter or its
        value. All values will be quoted as necessary unless requote is
        False.  Optional header specifies an alternative to the Content-Type
        header.
        """
        if header not in self:
            return
        new_ctype = ''
        for p, v in self.get_params(header=header, unquote=requote):
            if p.lower() != param.lower():
                if not new_ctype:
                    new_ctype = _formatparam(p, v, requote)
                else:
                    new_ctype = SEMISPACE.join([new_ctype,
                                                _formatparam(p, v, requote)])
        if new_ctype != self.get(header):
            del self[header]
            self[header] = new_ctype

    def set_type(self, type, header='Content-Type', requote=True):
        """Set the main type and subtype for the Content-Type header.

        type must be a string in the form "maintype/subtype", otherwise a
        ValueError is raised.

        This method replaces the Content-Type header, keeping all the
        parameters in place.  If requote is False, this leaves the existing
        header's quoting as is.  Otherwise, the parameters will be quoted (the
        default).

        An alternative header can be specified in the header argument.  When
        the Content-Type header is set, we'll always also add a MIME-Version
        header.
        """
        # BAW: should we be strict?
        if not type.count('/') == 1:
            raise ValueError
        # Set the Content-Type, you get a MIME-Version
        if header.lower() == 'content-type':
            del self['mime-version']
            self['MIME-Version'] = '1.0'
        if header not in self:
            self[header] = type
            return
        params = self.get_params(header=header, unquote=requote)
        del self[header]
        self[header] = type
        # Skip the first param; it's the old type.
        for p, v in params[1:]:
            self.set_param(p, v, header, requote)

    def get_filename(self, failobj=None):
        """Return the filename associated with the payload if present.

        The filename is extracted from the Content-Disposition header's
        `filename' parameter, and it is unquoted.  If that header is missing
        the `filename' parameter, this method falls back to looking for the
        `name' parameter.
        """
        missing = object()
        filename = self.get_param('filename', missing, 'content-disposition')
        if filename is missing:
            filename = self.get_param('name', missing, 'content-type')
        if filename is missing:
            return failobj
        return utils.collapse_rfc2231_value(filename).strip()

    def get_boundary(self, failobj=None):
        """Return the boundary associated with the payload if present.

        The boundary is extracted from the Content-Type header's `boundary'
        parameter, and it is unquoted.
        """
        missing = object()
        boundary = self.get_param('boundary', missing)
        if boundary is missing:
            return failobj
        # RFC 2046 says that boundaries may begin but not end in w/s
        return utils.collapse_rfc2231_value(boundary).rstrip()

    def set_boundary(self, boundary):
        """Set the boundary parameter in Content-Type to 'boundary'.

        This is subtly different than deleting the Content-Type header and
        adding a new one with a new boundary parameter via add_header().  The
        main difference is that using the set_boundary() method preserves the
        order of the Content-Type header in the original message.

        HeaderParseError is raised if the message has no Content-Type header.
        """
        missing = object()
        params = self._get_params_preserve(missing, 'content-type')
        if params is missing:
            # There was no Content-Type header, and we don't know what type
            # to set it to, so raise an exception.
            raise errors.HeaderParseError('No Content-Type header found')
        newparams = []
        foundp = False
        for pk, pv in params:
            if pk.lower() == 'boundary':
                newparams.append(('boundary', '"%s"' % boundary))
                foundp = True
            else:
                newparams.append((pk, pv))
        if not foundp:
            # The original Content-Type header had no boundary attribute.
            # Tack one on the end.  BAW: should we raise an exception
            # instead???
            newparams.append(('boundary', '"%s"' % boundary))
        # Replace the existing Content-Type header with the new value
        newheaders = []
        for h, v in self._headers:
            if h.lower() == 'content-type':
                parts = []
                for k, v in newparams:
                    if v == '':
                        parts.append(k)
                    else:
                        parts.append('%s=%s' % (k, v))
                val = SEMISPACE.join(parts)
                newheaders.append(self.policy.header_store_parse(h, val))

            else:
                newheaders.append((h, v))
        self._headers = newheaders

    def get_content_charset(self, failobj=None):
        """Return the charset parameter of the Content-Type header.

        The returned string is always coerced to lower case.  If there is no
        Content-Type header, or if that header has no charset parameter,
        failobj is returned.
        """
        missing = object()
        charset = self.get_param('charset', missing)
        if charset is missing:
            return failobj
        if isinstance(charset, tuple):
            # RFC 2231 encoded, so decode it, and it better end up as ascii.
            pcharset = charset[0] or 'us-ascii'
            try:
                # LookupError will be raised if the charset isn't known to
                # Python.  UnicodeError will be raised if the encoded text
                # contains a character not in the charset.
                as_bytes = charset[2].encode('raw-unicode-escape')
                charset = str(as_bytes, pcharset)
            except (LookupError, UnicodeError):
                charset = charset[2]
        # charset characters must be in us-ascii range
        try:
            charset.encode('us-ascii')
        except UnicodeError:
            return failobj
        # RFC 2046, $4.1.2 says charsets are not case sensitive
        return charset.lower()

    def get_charsets(self, failobj=None):
        """Return a list containing the charset(s) used in this message.

        The returned list of items describes the Content-Type headers'
        charset parameter for this message and all the subparts in its
        payload.

        Each item will either be a string (the value of the charset parameter
        in the Content-Type header of that part) or the value of the
        'failobj' parameter (defaults to None), if the part does not have a
        main MIME type of "text", or the charset is not defined.

        The list will contain one string for each part of the message, plus
        one for the container message (i.e. self), so that a non-multipart
        message will still return a list of length 1.
        """
        return [part.get_content_charset(failobj) for part in self.walk()]

    def get_content_disposition(self):
        """Return the message's content-disposition if it exists, or None.

        The return values can be either 'inline', 'attachment' or None
        according to the rfc2183.
        """
        value = self.get('content-disposition')
        if value is None:
            return None
        c_d = _splitparam(value)[0].lower()
        return c_d

    # I.e. def walk(self): ...
    from email.iterators import walk


class MIMEPart(Message):

    def __init__(self, policy=None):
        if policy is None:
            from email.policy import default
            policy = default
        super().__init__(policy)


    def as_string(self, unixfrom=False, maxheaderlen=None, policy=None):
        """Return the entire formatted message as a string.

        Optional 'unixfrom', when true, means include the Unix From_ envelope
        header.  maxheaderlen is retained for backward compatibility with the
        base Message class, but defaults to None, meaning that the policy value
        for max_line_length controls the header maximum length.  'policy' is
        passed to the Generator instance used to serialize the message; if it
        is not specified the policy associated with the message instance is
        used.
        """
        policy = self.policy if policy is None else policy
        if maxheaderlen is None:
            maxheaderlen = policy.max_line_length
        return super().as_string(unixfrom, maxheaderlen, policy)

    def __str__(self):
        return self.as_string(policy=self.policy.clone(utf8=True))

    def is_attachment(self):
        c_d = self.get('content-disposition')
        return False if c_d is None else c_d.content_disposition == 'attachment'

    def _find_body(self, part, preferencelist):
        if part.is_attachment():
            return
        maintype, subtype = part.get_content_type().split('/')
        if maintype == 'text':
            if subtype in preferencelist:
                yield (preferencelist.index(subtype), part)
            return
        if maintype != 'multipart' or not self.is_multipart():
            return
        if subtype != 'related':
            for subpart in part.iter_parts():
                yield from self._find_body(subpart, preferencelist)
            return
        if 'related' in preferencelist:
            yield (preferencelist.index('related'), part)
        candidate = None
        start = part.get_param('start')
        if start:
            for subpart in part.iter_parts():
                if subpart['content-id'] == start:
                    candidate = subpart
                    break
        if candidate is None:
            subparts = part.get_payload()
            candidate = subparts[0] if subparts else None
        if candidate is not None:
            yield from self._find_body(candidate, preferencelist)

    def get_body(self, preferencelist=('related', 'html', 'plain')):
        """Return best candidate mime part for display as 'body' of message.

        Do a depth first search, starting with self, looking for the first part
        matching each of the items in preferencelist, and return the part
        corresponding to the first item that has a match, or None if no items
        have a match.  If 'related' is not included in preferencelist, consider
        the root part of any multipart/related encountered as a candidate
        match.  Ignore parts with 'Content-Disposition: attachment'.
        """
        best_prio = len(preferencelist)
        body = None
        for prio, part in self._find_body(self, preferencelist):
            if prio < best_prio:
                best_prio = prio
                body = part
                if prio == 0:
                    break
        return body

    _body_types = {('text', 'plain'),
                   ('text', 'html'),
                   ('multipart', 'related'),
                   ('multipart', 'alternative')}
    def iter_attachments(self):
        """Return an iterator over the non-main parts of a multipart.

        Skip the first of each occurrence of text/plain, text/html,
        multipart/related, or multipart/alternative in the multipart (unless
        they have a 'Content-Disposition: attachment' header) and include all
        remaining subparts in the returned iterator.  When applied to a
        multipart/related, return all parts except the root part.  Return an
        empty iterator when applied to a multipart/alternative or a
        non-multipart.
        """
        maintype, subtype = self.get_content_type().split('/')
        if maintype != 'multipart' or subtype == 'alternative':
            return
        payload = self.get_payload()
        # Certain malformed messages can have content type set to `multipart/*`
        # but still have single part body, in which case payload.copy() can
        # fail with AttributeError.
        try:
            parts = payload.copy()
        except AttributeError:
            # payload is not a list, it is most probably a string.
            return

        if maintype == 'multipart' and subtype == 'related':
            # For related, we treat everything but the root as an attachment.
            # The root may be indicated by 'start'; if there's no start or we
            # can't find the named start, treat the first subpart as the root.
            start = self.get_param('start')
            if start:
                found = False
                attachments = []
                for part in parts:
                    if part.get('content-id') == start:
                        found = True
                    else:
                        attachments.append(part)
                if found:
                    yield from attachments
                    return
            parts.pop(0)
            yield from parts
            return
        # Otherwise we more or less invert the remaining logic in get_body.
        # This only really works in edge cases (ex: non-text related or
        # alternatives) if the sending agent sets content-disposition.
        seen = []   # Only skip the first example of each candidate type.
        for part in parts:
            maintype, subtype = part.get_content_type().split('/')
            if ((maintype, subtype) in self._body_types and
                    not part.is_attachment() and subtype not in seen):
                seen.append(subtype)
                continue
            yield part

    def iter_parts(self):
        """Return an iterator over all immediate subparts of a multipart.

        Return an empty iterator for a non-multipart.
        """
        if self.is_multipart():
            yield from self.get_payload()

    def get_content(self, *args, content_manager=None, **kw):
        if content_manager is None:
            content_manager = self.policy.content_manager
        return content_manager.get_content(self, *args, **kw)

    def set_content(self, *args, content_manager=None, **kw):
        if content_manager is None:
            content_manager = self.policy.content_manager
        content_manager.set_content(self, *args, **kw)

    def _make_multipart(self, subtype, disallowed_subtypes, boundary):
        if self.get_content_maintype() == 'multipart':
            existing_subtype = self.get_content_subtype()
            disallowed_subtypes = disallowed_subtypes + (subtype,)
            if existing_subtype in disallowed_subtypes:
                raise ValueError("Cannot convert {} to {}".format(
                    existing_subtype, subtype))
        keep_headers = []
        part_headers = []
        for name, value in self._headers:
            if name.lower().startswith('content-'):
                part_headers.append((name, value))
            else:
                keep_headers.append((name, value))
        if part_headers:
            # There is existing content, move it to the first subpart.
            part = type(self)(policy=self.policy)
            part._headers = part_headers
            part._payload = self._payload
            self._payload = [part]
        else:
            self._payload = []
        self._headers = keep_headers
        self['Content-Type'] = 'multipart/' + subtype
        if boundary is not None:
            self.set_param('boundary', boundary)

    def make_related(self, boundary=None):
        self._make_multipart('related', ('alternative', 'mixed'), boundary)

    def make_alternative(self, boundary=None):
        self._make_multipart('alternative', ('mixed',), boundary)

    def make_mixed(self, boundary=None):
        self._make_multipart('mixed', (), boundary)

    def _add_multipart(self, _subtype, *args, _disp=None, **kw):
        if (self.get_content_maintype() != 'multipart' or
                self.get_content_subtype() != _subtype):
            getattr(self, 'make_' + _subtype)()
        part = type(self)(policy=self.policy)
        part.set_content(*args, **kw)
        if _disp and 'content-disposition' not in part:
            part['Content-Disposition'] = _disp
        self.attach(part)

    def add_related(self, *args, **kw):
        self._add_multipart('related', *args, _disp='inline', **kw)

    def add_alternative(self, *args, **kw):
        self._add_multipart('alternative', *args, **kw)

    def add_attachment(self, *args, **kw):
        self._add_multipart('mixed', *args, _disp='attachment', **kw)

    def clear(self):
        self._headers = []
        self._payload = None

    def clear_content(self):
        self._headers = [(n, v) for n, v in self._headers
                         if not n.lower().startswith('content-')]
        self._payload = None


class EmailMessage(MIMEPart):

    def set_content(self, *args, **kw):
        super().set_content(*args, **kw)
        if 'MIME-Version' not in self:
            self['MIME-Version'] = '1.0'
<|endoftext|># Copyright (C) 2002-2007 Python Software Foundation
# Author: Ben Gertzfield, Barry Warsaw
# Contact: email-sig@python.org

"""Header encoding and decoding functionality."""

__all__ = [
    'Header',
    'decode_header',
    'make_header',
    ]

import re
import binascii

import email.quoprimime
import email.base64mime

from email.errors import HeaderParseError
from email import charset as _charset
Charset = _charset.Charset

NL = '\n'
SPACE = ' '
BSPACE = b' '
SPACE8 = ' ' * 8
EMPTYSTRING = ''
MAXLINELEN = 78
FWS = ' \t'

USASCII = Charset('us-ascii')
UTF8 = Charset('utf-8')

# Match encoded-word strings in the form =?charset?q?Hello_World?=
ecre = re.compile(r'''
  =\?                   # literal =?
  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset
  \?                    # literal ?
  (?P<encoding>[qQbB])  # either a "q" or a "b", case insensitive
  \?                    # literal ?
  (?P<encoded>.*?)      # non-greedy up to the next ?= is the encoded string
  \?=                   # literal ?=
  ''', re.VERBOSE | re.MULTILINE)

# Field name regexp, including trailing colon, but not separating whitespace,
# according to RFC 2822.  Character range is from tilde to exclamation mark.
# For use with .match()
fcre = re.compile(r'[\041-\176]+:$')

# Find a header embedded in a putative header value.  Used to check for
# header injection attack.
_embedded_header = re.compile(r'\n[^ \t]+:')



# Helpers
_max_append = email.quoprimime._max_append



def decode_header(header):
    """Decode a message header value without converting charset.

    Returns a list of (string, charset) pairs containing each of the decoded
    parts of the header.  Charset is None for non-encoded parts of the header,
    otherwise a lower-case string containing the name of the character set
    specified in the encoded string.

    header may be a string that may or may not contain RFC2047 encoded words,
    or it may be a Header object.

    An email.errors.HeaderParseError may be raised when certain decoding error
    occurs (e.g. a base64 decoding exception).
    """
    # If it is a Header object, we can just return the encoded chunks.
    if hasattr(header, '_chunks'):
        return [(_charset._encode(string, str(charset)), str(charset))
                    for string, charset in header._chunks]
    # If no encoding, just return the header with no charset.
    if not ecre.search(header):
        return [(header, None)]
    # First step is to parse all the encoded parts into triplets of the form
    # (encoded_string, encoding, charset).  For unencoded strings, the last
    # two parts will be None.
    words = []
    for line in header.splitlines():
        parts = ecre.split(line)
        first = True
        while parts:
            unencoded = parts.pop(0)
            if first:
                unencoded = unencoded.lstrip()
                first = False
            if unencoded:
                words.append((unencoded, None, None))
            if parts:
                charset = parts.pop(0).lower()
                encoding = parts.pop(0).lower()
                encoded = parts.pop(0)
                words.append((encoded, encoding, charset))
    # Now loop over words and remove words that consist of whitespace
    # between two encoded strings.
    droplist = []
    for n, w in enumerate(words):
        if n>1 and w[1] and words[n-2][1] and words[n-1][0].isspace():
            droplist.append(n-1)
    for d in reversed(droplist):
        del words[d]

    # The next step is to decode each encoded word by applying the reverse
    # base64 or quopri transformation.  decoded_words is now a list of the
    # form (decoded_word, charset).
    decoded_words = []
    for encoded_string, encoding, charset in words:
        if encoding is None:
            # This is an unencoded word.
            decoded_words.append((encoded_string, charset))
        elif encoding == 'q':
            word = email.quoprimime.header_decode(encoded_string)
            decoded_words.append((word, charset))
        elif encoding == 'b':
            paderr = len(encoded_string) % 4   # Postel's law: add missing padding
            if paderr:
                encoded_string += '==='[:4 - paderr]
            try:
                word = email.base64mime.decode(encoded_string)
            except binascii.Error:
                raise HeaderParseError('Base64 decoding error')
            else:
                decoded_words.append((word, charset))
        else:
            raise AssertionError('Unexpected encoding: ' + encoding)
    # Now convert all words to bytes and collapse consecutive runs of
    # similarly encoded words.
    collapsed = []
    last_word = last_charset = None
    for word, charset in decoded_words:
        if isinstance(word, str):
            word = bytes(word, 'raw-unicode-escape')
        if last_word is None:
            last_word = word
            last_charset = charset
        elif charset != last_charset:
            collapsed.append((last_word, last_charset))
            last_word = word
            last_charset = charset
        elif last_charset is None:
            last_word += BSPACE + word
        else:
            last_word += word
    collapsed.append((last_word, last_charset))
    return collapsed



def make_header(decoded_seq, maxlinelen=None, header_name=None,
                continuation_ws=' '):
    """Create a Header from a sequence of pairs as returned by decode_header()

    decode_header() takes a header value string and returns a sequence of
    pairs of the format (decoded_string, charset) where charset is the string
    name of the character set.

    This function takes one of those sequence of pairs and returns a Header
    instance.  Optional maxlinelen, header_name, and continuation_ws are as in
    the Header constructor.
    """
    h = Header(maxlinelen=maxlinelen, header_name=header_name,
               continuation_ws=continuation_ws)
    for s, charset in decoded_seq:
        # None means us-ascii but we can simply pass it on to h.append()
        if charset is not None and not isinstance(charset, Charset):
            charset = Charset(charset)
        h.append(s, charset)
    return h



class Header:
    def __init__(self, s=None, charset=None,
                 maxlinelen=None, header_name=None,
                 continuation_ws=' ', errors='strict'):
        """Create a MIME-compliant header that can contain many character sets.

        Optional s is the initial header value.  If None, the initial header
        value is not set.  You can later append to the header with .append()
        method calls.  s may be a byte string or a Unicode string, but see the
        .append() documentation for semantics.

        Optional charset serves two purposes: it has the same meaning as the
        charset argument to the .append() method.  It also sets the default
        character set for all subsequent .append() calls that omit the charset
        argument.  If charset is not provided in the constructor, the us-ascii
        charset is used both as s's initial charset and as the default for
        subsequent .append() calls.

        The maximum line length can be specified explicitly via maxlinelen. For
        splitting the first line to a shorter value (to account for the field
        header which isn't included in s, e.g. `Subject') pass in the name of
        the field in header_name.  The default maxlinelen is 78 as recommended
        by RFC 2822.

        continuation_ws must be RFC 2822 compliant folding whitespace (usually
        either a space or a hard tab) which will be prepended to continuation
        lines.

        errors is passed through to the .append() call.
        """
        if charset is None:
            charset = USASCII
        elif not isinstance(charset, Charset):
            charset = Charset(charset)
        self._charset = charset
        self._continuation_ws = continuation_ws
        self._chunks = []
        if s is not None:
            self.append(s, charset, errors)
        if maxlinelen is None:
            maxlinelen = MAXLINELEN
        self._maxlinelen = maxlinelen
        if header_name is None:
            self._headerlen = 0
        else:
            # Take the separating colon and space into account.
            self._headerlen = len(header_name) + 2

    def __str__(self):
        """Return the string value of the header."""
        self._normalize()
        uchunks = []
        lastcs = None
        lastspace = None
        for string, charset in self._chunks:
            # We must preserve spaces between encoded and non-encoded word
            # boundaries, which means for us we need to add a space when we go
            # from a charset to None/us-ascii, or from None/us-ascii to a
            # charset.  Only do this for the second and subsequent chunks.
            # Don't add a space if the None/us-ascii string already has
            # a space (trailing or leading depending on transition)
            nextcs = charset
            if nextcs == _charset.UNKNOWN8BIT:
                original_bytes = string.encode('ascii', 'surrogateescape')
                string = original_bytes.decode('ascii', 'replace')
            if uchunks:
                hasspace = string and self._nonctext(string[0])
                if lastcs not in (None, 'us-ascii'):
                    if nextcs in (None, 'us-ascii') and not hasspace:
                        uchunks.append(SPACE)
                        nextcs = None
                elif nextcs not in (None, 'us-ascii') and not lastspace:
                    uchunks.append(SPACE)
            lastspace = string and self._nonctext(string[-1])
            lastcs = nextcs
            uchunks.append(string)
        return EMPTYSTRING.join(uchunks)

    # Rich comparison operators for equality only.  BAW: does it make sense to
    # have or explicitly disable <, <=, >, >= operators?
    def __eq__(self, other):
        # other may be a Header or a string.  Both are fine so coerce
        # ourselves to a unicode (of the unencoded header value), swap the
        # args and do another comparison.
        return other == str(self)

    def append(self, s, charset=None, errors='strict'):
        """Append a string to the MIME header.

        Optional charset, if given, should be a Charset instance or the name
        of a character set (which will be converted to a Charset instance).  A
        value of None (the default) means that the charset given in the
        constructor is used.

        s may be a byte string or a Unicode string.  If it is a byte string
        (i.e. isinstance(s, str) is false), then charset is the encoding of
        that byte string, and a UnicodeError will be raised if the string
        cannot be decoded with that charset.  If s is a Unicode string, then
        charset is a hint specifying the character set of the characters in
        the string.  In either case, when producing an RFC 2822 compliant
        header using RFC 2047 rules, the string will be encoded using the
        output codec of the charset.  If the string cannot be encoded to the
        output codec, a UnicodeError will be raised.

        Optional `errors' is passed as the errors argument to the decode
        call if s is a byte string.
        """
        if charset is None:
            charset = self._charset
        elif not isinstance(charset, Charset):
            charset = Charset(charset)
        if not isinstance(s, str):
            input_charset = charset.input_codec or 'us-ascii'
            if input_charset == _charset.UNKNOWN8BIT:
                s = s.decode('us-ascii', 'surrogateescape')
            else:
                s = s.decode(input_charset, errors)
        # Ensure that the bytes we're storing can be decoded to the output
        # character set, otherwise an early error is raised.
        output_charset = charset.output_codec or 'us-ascii'
        if output_charset != _charset.UNKNOWN8BIT:
            try:
                s.encode(output_charset, errors)
            except UnicodeEncodeError:
                if output_charset!='us-ascii':
                    raise
                charset = UTF8
        self._chunks.append((s, charset))

    def _nonctext(self, s):
        """True if string s is not a ctext character of RFC822.
        """
        return s.isspace() or s in ('(', ')', '\\')

    def encode(self, splitchars=';, \t', maxlinelen=None, linesep='\n'):
        r"""Encode a message header into an RFC-compliant format.

        There are many issues involved in converting a given string for use in
        an email header.  Only certain character sets are readable in most
        email clients, and as header strings can only contain a subset of
        7-bit ASCII, care must be taken to properly convert and encode (with
        Base64 or quoted-printable) header strings.  In addition, there is a
        75-character length limit on any given encoded header field, so
        line-wrapping must be performed, even with double-byte character sets.

        Optional maxlinelen specifies the maximum length of each generated
        line, exclusive of the linesep string.  Individual lines may be longer
        than maxlinelen if a folding point cannot be found.  The first line
        will be shorter by the length of the header name plus ": " if a header
        name was specified at Header construction time.  The default value for
        maxlinelen is determined at header construction time.

        Optional splitchars is a string containing characters which should be
        given extra weight by the splitting algorithm during normal header
        wrapping.  This is in very rough support of RFC 2822's `higher level
        syntactic breaks':  split points preceded by a splitchar are preferred
        during line splitting, with the characters preferred in the order in
        which they appear in the string.  Space and tab may be included in the
        string to indicate whether preference should be given to one over the
        other as a split point when other split chars do not appear in the line
        being split.  Splitchars does not affect RFC 2047 encoded lines.

        Optional linesep is a string to be used to separate the lines of
        the value.  The default value is the most useful for typical
        Python applications, but it can be set to \r\n to produce RFC-compliant
        line separators when needed.
        """
        self._normalize()
        if maxlinelen is None:
            maxlinelen = self._maxlinelen
        # A maxlinelen of 0 means don't wrap.  For all practical purposes,
        # choosing a huge number here accomplishes that and makes the
        # _ValueFormatter algorithm much simpler.
        if maxlinelen == 0:
            maxlinelen = 1000000
        formatter = _ValueFormatter(self._headerlen, maxlinelen,
                                    self._continuation_ws, splitchars)
        lastcs = None
        hasspace = lastspace = None
        for string, charset in self._chunks:
            if hasspace is not None:
                hasspace = string and self._nonctext(string[0])
                if lastcs not in (None, 'us-ascii'):
                    if not hasspace or charset not in (None, 'us-ascii'):
                        formatter.add_transition()
                elif charset not in (None, 'us-ascii') and not lastspace:
                    formatter.add_transition()
            lastspace = string and self._nonctext(string[-1])
            lastcs = charset
            hasspace = False
            lines = string.splitlines()
            if lines:
                formatter.feed('', lines[0], charset)
            else:
                formatter.feed('', '', charset)
            for line in lines[1:]:
                formatter.newline()
                if charset.header_encoding is not None:
                    formatter.feed(self._continuation_ws, ' ' + line.lstrip(),
                                   charset)
                else:
                    sline = line.lstrip()
                    fws = line[:len(line)-len(sline)]
                    formatter.feed(fws, sline, charset)
            if len(lines) > 1:
                formatter.newline()
        if self._chunks:
            formatter.add_transition()
        value = formatter._str(linesep)
        if _embedded_header.search(value):
            raise HeaderParseError("header value appears to contain "
                "an embedded header: {!r}".format(value))
        return value

    def _normalize(self):
        # Step 1: Normalize the chunks so that all runs of identical charsets
        # get collapsed into a single unicode string.
        chunks = []
        last_charset = None
        last_chunk = []
        for string, charset in self._chunks:
            if charset == last_charset:
                last_chunk.append(string)
            else:
                if last_charset is not None:
                    chunks.append((SPACE.join(last_chunk), last_charset))
                last_chunk = [string]
                last_charset = charset
        if last_chunk:
            chunks.append((SPACE.join(last_chunk), last_charset))
        self._chunks = chunks



class _ValueFormatter:
    def __init__(self, headerlen, maxlen, continuation_ws, splitchars):
        self._maxlen = maxlen
        self._continuation_ws = continuation_ws
        self._continuation_ws_len = len(continuation_ws)
        self._splitchars = splitchars
        self._lines = []
        self._current_line = _Accumulator(headerlen)

    def _str(self, linesep):
        self.newline()
        return linesep.join(self._lines)

    def __str__(self):
        return self._str(NL)

    def newline(self):
        end_of_line = self._current_line.pop()
        if end_of_line != (' ', ''):
            self._current_line.push(*end_of_line)
        if len(self._current_line) > 0:
            if self._current_line.is_onlyws() and self._lines:
                self._lines[-1] += str(self._current_line)
            else:
                self._lines.append(str(self._current_line))
        self._current_line.reset()

    def add_transition(self):
        self._current_line.push(' ', '')

    def feed(self, fws, string, charset):
        # If the charset has no header encoding (i.e. it is an ASCII encoding)
        # then we must split the header at the "highest level syntactic break"
        # possible. Note that we don't have a lot of smarts about field
        # syntax; we just try to break on semi-colons, then commas, then
        # whitespace.  Eventually, this should be pluggable.
        if charset.header_encoding is None:
            self._ascii_split(fws, string, self._splitchars)
            return
        # Otherwise, we're doing either a Base64 or a quoted-printable
        # encoding which means we don't need to split the line on syntactic
        # breaks.  We can basically just find enough characters to fit on the
        # current line, minus the RFC 2047 chrome.  What makes this trickier
        # though is that we have to split at octet boundaries, not character
        # boundaries but it's only safe to split at character boundaries so at
        # best we can only get close.
        encoded_lines = charset.header_encode_lines(string, self._maxlengths())
        # The first element extends the current line, but if it's None then
        # nothing more fit on the current line so start a new line.
        try:
            first_line = encoded_lines.pop(0)
        except IndexError:
            # There are no encoded lines, so we're done.
            return
        if first_line is not None:
            self._append_chunk(fws, first_line)
        try:
            last_line = encoded_lines.pop()
        except IndexError:
            # There was only one line.
            return
        self.newline()
        self._current_line.push(self._continuation_ws, last_line)
        # Everything else are full lines in themselves.
        for line in encoded_lines:
            self._lines.append(self._continuation_ws + line)

    def _maxlengths(self):
        # The first line's length.
        yield self._maxlen - len(self._current_line)
        while True:
            yield self._maxlen - self._continuation_ws_len

    def _ascii_split(self, fws, string, splitchars):
        # The RFC 2822 header folding algorithm is simple in principle but
        # complex in practice.  Lines may be folded any place where "folding
        # white space" appears by inserting a linesep character in front of the
        # FWS.  The complication is that not all spaces or tabs qualify as FWS,
        # and we are also supposed to prefer to break at "higher level
        # syntactic breaks".  We can't do either of these without intimate
        # knowledge of the structure of structured headers, which we don't have
        # here.  So the best we can do here is prefer to break at the specified
        # splitchars, and hope that we don't choose any spaces or tabs that
        # aren't legal FWS.  (This is at least better than the old algorithm,
        # where we would sometimes *introduce* FWS after a splitchar, or the
        # algorithm before that, where we would turn all white space runs into
        # single spaces or tabs.)
        parts = re.split("(["+FWS+"]+)", fws+string)
        if parts[0]:
            parts[:0] = ['']
        else:
            parts.pop(0)
        for fws, part in zip(*[iter(parts)]*2):
            self._append_chunk(fws, part)

    def _append_chunk(self, fws, string):
        self._current_line.push(fws, string)
        if len(self._current_line) > self._maxlen:
            # Find the best split point, working backward from the end.
            # There might be none, on a long first line.
            for ch in self._splitchars:
                for i in range(self._current_line.part_count()-1, 0, -1):
                    if ch.isspace():
                        fws = self._current_line[i][0]
                        if fws and fws[0]==ch:
                            break
                    prevpart = self._current_line[i-1][1]
                    if prevpart and prevpart[-1]==ch:
                        break
                else:
                    continue
                break
            else:
                fws, part = self._current_line.pop()
                if self._current_line._initial_size > 0:
                    # There will be a header, so leave it on a line by itself.
                    self.newline()
                    if not fws:
                        # We don't use continuation_ws here because the whitespace
                        # after a header should always be a space.
                        fws = ' '
                self._current_line.push(fws, part)
                return
            remainder = self._current_line.pop_from(i)
            self._lines.append(str(self._current_line))
            self._current_line.reset(remainder)


class _Accumulator(list):

    def __init__(self, initial_size=0):
        self._initial_size = initial_size
        super().__init__()

    def push(self, fws, string):
        self.append((fws, string))

    def pop_from(self, i=0):
        popped = self[i:]
        self[i:] = []
        return popped

    def pop(self):
        if self.part_count()==0:
            return ('', '')
        return super().pop()

    def __len__(self):
        return sum((len(fws)+len(part) for fws, part in self),
                   self._initial_size)

    def __str__(self):
        return EMPTYSTRING.join((EMPTYSTRING.join((fws, part))
                                for fws, part in self))

    def reset(self, startval=None):
        if startval is None:
            startval = []
        self[:] = startval
        self._initial_size = 0

    def is_onlyws(self):
        return self._initial_size==0 and (not self or str(self).isspace())

    def part_count(self):
        return super().__len__()
<|endoftext|>"""This will be the home for the policy that hooks in the new
code that adds all the email6 features.
"""

import re
import sys
from email._policybase import Policy, Compat32, compat32, _extend_docstrings
from email.utils import _has_surrogates
from email.headerregistry import HeaderRegistry as HeaderRegistry
from email.contentmanager import raw_data_manager
from email.message import EmailMessage

__all__ = [
    'Compat32',
    'compat32',
    'Policy',
    'EmailPolicy',
    'default',
    'strict',
    'SMTP',
    'HTTP',
    ]

linesep_splitter = re.compile(r'\n|\r')

@_extend_docstrings
class EmailPolicy(Policy):

    """+
    PROVISIONAL

    The API extensions enabled by this policy are currently provisional.
    Refer to the documentation for details.

    This policy adds new header parsing and folding algorithms.  Instead of
    simple strings, headers are custom objects with custom attributes
    depending on the type of the field.  The folding algorithm fully
    implements RFCs 2047 and 5322.

    In addition to the settable attributes listed above that apply to
    all Policies, this policy adds the following additional attributes:

    utf8                -- if False (the default) message headers will be
                           serialized as ASCII, using encoded words to encode
                           any non-ASCII characters in the source strings.  If
                           True, the message headers will be serialized using
                           utf8 and will not contain encoded words (see RFC
                           6532 for more on this serialization format).

    refold_source       -- if the value for a header in the Message object
                           came from the parsing of some source, this attribute
                           indicates whether or not a generator should refold
                           that value when transforming the message back into
                           stream form.  The possible values are:

                           none  -- all source values use original folding
                           long  -- source values that have any line that is
                                    longer than max_line_length will be
                                    refolded
                           all  -- all values are refolded.

                           The default is 'long'.

    header_factory      -- a callable that takes two arguments, 'name' and
                           'value', where 'name' is a header field name and
                           'value' is an unfolded header field value, and
                           returns a string-like object that represents that
                           header.  A default header_factory is provided that
                           understands some of the RFC5322 header field types.
                           (Currently address fields and date fields have
                           special treatment, while all other fields are
                           treated as unstructured.  This list will be
                           completed before the extension is marked stable.)

    content_manager     -- an object with at least two methods: get_content
                           and set_content.  When the get_content or
                           set_content method of a Message object is called,
                           it calls the corresponding method of this object,
                           passing it the message object as its first argument,
                           and any arguments or keywords that were passed to
                           it as additional arguments.  The default
                           content_manager is
                           :data:`~email.contentmanager.raw_data_manager`.

    """

    message_factory = EmailMessage
    utf8 = False
    refold_source = 'long'
    header_factory = HeaderRegistry()
    content_manager = raw_data_manager

    def __init__(self, **kw):
        # Ensure that each new instance gets a unique header factory
        # (as opposed to clones, which share the factory).
        if 'header_factory' not in kw:
            object.__setattr__(self, 'header_factory', HeaderRegistry())
        super().__init__(**kw)

    def header_max_count(self, name):
        """+
        The implementation for this class returns the max_count attribute from
        the specialized header class that would be used to construct a header
        of type 'name'.
        """
        return self.header_factory[name].max_count

    # The logic of the next three methods is chosen such that it is possible to
    # switch a Message object between a Compat32 policy and a policy derived
    # from this class and have the results stay consistent.  This allows a
    # Message object constructed with this policy to be passed to a library
    # that only handles Compat32 objects, or to receive such an object and
    # convert it to use the newer style by just changing its policy.  It is
    # also chosen because it postpones the relatively expensive full rfc5322
    # parse until as late as possible when parsing from source, since in many
    # applications only a few headers will actually be inspected.

    def header_source_parse(self, sourcelines):
        """+
        The name is parsed as everything up to the ':' and returned unmodified.
        The value is determined by stripping leading whitespace off the
        remainder of the first line, joining all subsequent lines together, and
        stripping any trailing carriage return or linefeed characters.  (This
        is the same as Compat32).

        """
        name, value = sourcelines[0].split(':', 1)
        value = value.lstrip(' \t') + ''.join(sourcelines[1:])
        return (name, value.rstrip('\r\n'))

    def header_store_parse(self, name, value):
        """+
        The name is returned unchanged.  If the input value has a 'name'
        attribute and it matches the name ignoring case, the value is returned
        unchanged.  Otherwise the name and value are passed to header_factory
        method, and the resulting custom header object is returned as the
        value.  In this case a ValueError is raised if the input value contains
        CR or LF characters.

        """
        if hasattr(value, 'name') and value.name.lower() == name.lower():
            return (name, value)
        if isinstance(value, str) and len(value.splitlines())>1:
            # XXX this error message isn't quite right when we use splitlines
            # (see issue 22233), but I'm not sure what should happen here.
            raise ValueError("Header values may not contain linefeed "
                             "or carriage return characters")
        return (name, self.header_factory(name, value))

    def header_fetch_parse(self, name, value):
        """+
        If the value has a 'name' attribute, it is returned to unmodified.
        Otherwise the name and the value with any linesep characters removed
        are passed to the header_factory method, and the resulting custom
        header object is returned.  Any surrogateescaped bytes get turned
        into the unicode unknown-character glyph.

        """
        if hasattr(value, 'name'):
            return value
        # We can't use splitlines here because it splits on more than \r and \n.
        value = ''.join(linesep_splitter.split(value))
        return self.header_factory(name, value)

    def fold(self, name, value):
        """+
        Header folding is controlled by the refold_source policy setting.  A
        value is considered to be a 'source value' if and only if it does not
        have a 'name' attribute (having a 'name' attribute means it is a header
        object of some sort).  If a source value needs to be refolded according
        to the policy, it is converted into a custom header object by passing
        the name and the value with any linesep characters removed to the
        header_factory method.  Folding of a custom header object is done by
        calling its fold method with the current policy.

        Source values are split into lines using splitlines.  If the value is
        not to be refolded, the lines are rejoined using the linesep from the
        policy and returned.  The exception is lines containing non-ascii
        binary data.  In that case the value is refolded regardless of the
        refold_source setting, which causes the binary data to be CTE encoded
        using the unknown-8bit charset.

        """
        return self._fold(name, value, refold_binary=True)

    def fold_binary(self, name, value):
        """+
        The same as fold if cte_type is 7bit, except that the returned value is
        bytes.

        If cte_type is 8bit, non-ASCII binary data is converted back into
        bytes.  Headers with binary data are not refolded, regardless of the
        refold_header setting, since there is no way to know whether the binary
        data consists of single byte characters or multibyte characters.

        If utf8 is true, headers are encoded to utf8, otherwise to ascii with
        non-ASCII unicode rendered as encoded words.

        """
        folded = self._fold(name, value, refold_binary=self.cte_type=='7bit')
        charset = 'utf8' if self.utf8 else 'ascii'
        return folded.encode(charset, 'surrogateescape')

    def _fold(self, name, value, refold_binary=False):
        if hasattr(value, 'name'):
            return value.fold(policy=self)
        maxlen = self.max_line_length if self.max_line_length else sys.maxsize
        lines = value.splitlines()
        refold = (self.refold_source == 'all' or
                  self.refold_source == 'long' and
                    (lines and len(lines[0])+len(name)+2 > maxlen or
                     any(len(x) > maxlen for x in lines[1:])))
        if refold or refold_binary and _has_surrogates(value):
            return self.header_factory(name, ''.join(lines)).fold(policy=self)
        return name + ': ' + self.linesep.join(lines) + self.linesep


default = EmailPolicy()
# Make the default policy use the class default header_factory
del default.header_factory
strict = default.clone(raise_on_defect=True)
SMTP = default.clone(linesep='\r\n')
HTTP = default.clone(linesep='\r\n', max_line_length=None)
SMTPUTF8 = SMTP.clone(utf8=True)
<|endoftext|>"""Header value parser implementing various email-related RFC parsing rules.

The parsing methods defined in this module implement various email related
parsing rules.  Principal among them is RFC 5322, which is the followon
to RFC 2822 and primarily a clarification of the former.  It also implements
RFC 2047 encoded word decoding.

RFC 5322 goes to considerable trouble to maintain backward compatibility with
RFC 822 in the parse phase, while cleaning up the structure on the generation
phase.  This parser supports correct RFC 5322 generation by tagging white space
as folding white space only when folding is allowed in the non-obsolete rule
sets.  Actually, the parser is even more generous when accepting input than RFC
5322 mandates, following the spirit of Postel's Law, which RFC 5322 encourages.
Where possible deviations from the standard are annotated on the 'defects'
attribute of tokens that deviate.

The general structure of the parser follows RFC 5322, and uses its terminology
where there is a direct correspondence.  Where the implementation requires a
somewhat different structure than that used by the formal grammar, new terms
that mimic the closest existing terms are used.  Thus, it really helps to have
a copy of RFC 5322 handy when studying this code.

Input to the parser is a string that has already been unfolded according to
RFC 5322 rules.  According to the RFC this unfolding is the very first step, and
this parser leaves the unfolding step to a higher level message parser, which
will have already detected the line breaks that need unfolding while
determining the beginning and end of each header.

The output of the parser is a TokenList object, which is a list subclass.  A
TokenList is a recursive data structure.  The terminal nodes of the structure
are Terminal objects, which are subclasses of str.  These do not correspond
directly to terminal objects in the formal grammar, but are instead more
practical higher level combinations of true terminals.

All TokenList and Terminal objects have a 'value' attribute, which produces the
semantically meaningful value of that part of the parse subtree.  The value of
all whitespace tokens (no matter how many sub-tokens they may contain) is a
single space, as per the RFC rules.  This includes 'CFWS', which is herein
included in the general class of whitespace tokens.  There is one exception to
the rule that whitespace tokens are collapsed into single spaces in values: in
the value of a 'bare-quoted-string' (a quoted-string with no leading or
trailing whitespace), any whitespace that appeared between the quotation marks
is preserved in the returned value.  Note that in all Terminal strings quoted
pairs are turned into their unquoted values.

All TokenList and Terminal objects also have a string value, which attempts to
be a "canonical" representation of the RFC-compliant form of the substring that
produced the parsed subtree, including minimal use of quoted pair quoting.
Whitespace runs are not collapsed.

Comment tokens also have a 'content' attribute providing the string found
between the parens (including any nested comments) with whitespace preserved.

All TokenList and Terminal objects have a 'defects' attribute which is a
possibly empty list all of the defects found while creating the token.  Defects
may appear on any token in the tree, and a composite list of all defects in the
subtree is available through the 'all_defects' attribute of any node.  (For
Terminal notes x.defects == x.all_defects.)

Each object in a parse tree is called a 'token', and each has a 'token_type'
attribute that gives the name from the RFC 5322 grammar that it represents.
Not all RFC 5322 nodes are produced, and there is one non-RFC 5322 node that
may be produced: 'ptext'.  A 'ptext' is a string of printable ascii characters.
It is returned in place of lists of (ctext/quoted-pair) and
(qtext/quoted-pair).

XXX: provide complete list of token types.
"""

import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils

#
# Useful constants and functions
#

WSP = set(' \t')
CFWS_LEADER = WSP | set('(')
SPECIALS = set(r'()<>@,:;.\"[]')
ATOM_ENDS = SPECIALS | WSP
DOT_ATOM_ENDS = ATOM_ENDS - set('.')
# '.', '"', and '(' do not end phrases in order to support obs-phrase
PHRASE_ENDS = SPECIALS - set('."(')
TSPECIALS = (SPECIALS | set('/?=')) - set('.')
TOKEN_ENDS = TSPECIALS | WSP
ASPECIALS = TSPECIALS | set("*'%")
ATTRIBUTE_ENDS = ASPECIALS | WSP
EXTENDED_ATTRIBUTE_ENDS = ATTRIBUTE_ENDS - set('%')
NLSET = {'\n', '\r'}
SPECIALSNL = SPECIALS | NLSET

def make_parenthesis_pairs(value):
    """Escape parenthesis and backslash for use within a comment."""
    return str(value).replace('\\', '\\\\') \
        .replace('(', '\\(').replace(')', '\\)')


def quote_string(value):
    return '"'+str(value).replace('\\', '\\\\').replace('"', r'\"')+'"'

# Match a RFC 2047 word, looks like =?utf-8?q?someword?=
rfc2047_matcher = re.compile(r'''
   =\?            # literal =?
   [^?]*          # charset
   \?             # literal ?
   [qQbB]         # literal 'q' or 'b', case insensitive
   \?             # literal ?
  .*?             # encoded word
  \?=             # literal ?=
''', re.VERBOSE | re.MULTILINE)


#
# TokenList and its subclasses
#

class TokenList(list):

    token_type = None
    syntactic_break = True
    ew_combine_allowed = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.defects = []

    def __str__(self):
        return ''.join(str(x) for x in self)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                             super().__repr__())

    @property
    def value(self):
        return ''.join(x.value for x in self if x.value)

    @property
    def all_defects(self):
        return sum((x.all_defects for x in self), self.defects)

    def startswith_fws(self):
        return self[0].startswith_fws()

    @property
    def as_ew_allowed(self):
        """True if all top level tokens of this part may be RFC2047 encoded."""
        return all(part.as_ew_allowed for part in self)

    @property
    def comments(self):
        comments = []
        for token in self:
            comments.extend(token.comments)
        return comments

    def fold(self, *, policy):
        return _refold_parse_tree(self, policy=policy)

    def pprint(self, indent=''):
        print(self.ppstr(indent=indent))

    def ppstr(self, indent=''):
        return '\n'.join(self._pp(indent=indent))

    def _pp(self, indent=''):
        yield '{}{}/{}('.format(
            indent,
            self.__class__.__name__,
            self.token_type)
        for token in self:
            if not hasattr(token, '_pp'):
                yield (indent + '    !! invalid element in token '
                                        'list: {!r}'.format(token))
            else:
                yield from token._pp(indent+'    ')
        if self.defects:
            extra = ' Defects: {}'.format(self.defects)
        else:
            extra = ''
        yield '{}){}'.format(indent, extra)


class WhiteSpaceTokenList(TokenList):

    @property
    def value(self):
        return ' '

    @property
    def comments(self):
        return [x.content for x in self if x.token_type=='comment']


class UnstructuredTokenList(TokenList):
    token_type = 'unstructured'


class Phrase(TokenList):
    token_type = 'phrase'

class Word(TokenList):
    token_type = 'word'


class CFWSList(WhiteSpaceTokenList):
    token_type = 'cfws'


class Atom(TokenList):
    token_type = 'atom'


class Token(TokenList):
    token_type = 'token'
    encode_as_ew = False


class EncodedWord(TokenList):
    token_type = 'encoded-word'
    cte = None
    charset = None
    lang = None


class QuotedString(TokenList):

    token_type = 'quoted-string'

    @property
    def content(self):
        for x in self:
            if x.token_type == 'bare-quoted-string':
                return x.value

    @property
    def quoted_value(self):
        res = []
        for x in self:
            if x.token_type == 'bare-quoted-string':
                res.append(str(x))
            else:
                res.append(x.value)
        return ''.join(res)

    @property
    def stripped_value(self):
        for token in self:
            if token.token_type == 'bare-quoted-string':
                return token.value


class BareQuotedString(QuotedString):

    token_type = 'bare-quoted-string'

    def __str__(self):
        return quote_string(''.join(str(x) for x in self))

    @property
    def value(self):
        return ''.join(str(x) for x in self)


class Comment(WhiteSpaceTokenList):

    token_type = 'comment'

    def __str__(self):
        return ''.join(sum([
                            ["("],
                            [self.quote(x) for x in self],
                            [")"],
                            ], []))

    def quote(self, value):
        if value.token_type == 'comment':
            return str(value)
        return str(value).replace('\\', '\\\\').replace(
                                  '(', r'\(').replace(
                                  ')', r'\)')

    @property
    def content(self):
        return ''.join(str(x) for x in self)

    @property
    def comments(self):
        return [self.content]

class AddressList(TokenList):

    token_type = 'address-list'

    @property
    def addresses(self):
        return [x for x in self if x.token_type=='address']

    @property
    def mailboxes(self):
        return sum((x.mailboxes
                    for x in self if x.token_type=='address'), [])

    @property
    def all_mailboxes(self):
        return sum((x.all_mailboxes
                    for x in self if x.token_type=='address'), [])


class Address(TokenList):

    token_type = 'address'

    @property
    def display_name(self):
        if self[0].token_type == 'group':
            return self[0].display_name

    @property
    def mailboxes(self):
        if self[0].token_type == 'mailbox':
            return [self[0]]
        elif self[0].token_type == 'invalid-mailbox':
            return []
        return self[0].mailboxes

    @property
    def all_mailboxes(self):
        if self[0].token_type == 'mailbox':
            return [self[0]]
        elif self[0].token_type == 'invalid-mailbox':
            return [self[0]]
        return self[0].all_mailboxes

class MailboxList(TokenList):

    token_type = 'mailbox-list'

    @property
    def mailboxes(self):
        return [x for x in self if x.token_type=='mailbox']

    @property
    def all_mailboxes(self):
        return [x for x in self
            if x.token_type in ('mailbox', 'invalid-mailbox')]


class GroupList(TokenList):

    token_type = 'group-list'

    @property
    def mailboxes(self):
        if not self or self[0].token_type != 'mailbox-list':
            return []
        return self[0].mailboxes

    @property
    def all_mailboxes(self):
        if not self or self[0].token_type != 'mailbox-list':
            return []
        return self[0].all_mailboxes


class Group(TokenList):

    token_type = "group"

    @property
    def mailboxes(self):
        if self[2].token_type != 'group-list':
            return []
        return self[2].mailboxes

    @property
    def all_mailboxes(self):
        if self[2].token_type != 'group-list':
            return []
        return self[2].all_mailboxes

    @property
    def display_name(self):
        return self[0].display_name


class NameAddr(TokenList):

    token_type = 'name-addr'

    @property
    def display_name(self):
        if len(self) == 1:
            return None
        return self[0].display_name

    @property
    def local_part(self):
        return self[-1].local_part

    @property
    def domain(self):
        return self[-1].domain

    @property
    def route(self):
        return self[-1].route

    @property
    def addr_spec(self):
        return self[-1].addr_spec


class AngleAddr(TokenList):

    token_type = 'angle-addr'

    @property
    def local_part(self):
        for x in self:
            if x.token_type == 'addr-spec':
                return x.local_part

    @property
    def domain(self):
        for x in self:
            if x.token_type == 'addr-spec':
