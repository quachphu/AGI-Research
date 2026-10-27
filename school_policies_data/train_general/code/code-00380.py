class FigureCanvasTemplate(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc.

    Note: GUI templates will want to connect events for button presses,
    mouse movements and key presses to functions that call the base
    class methods button_press_event, button_release_event,
    motion_notify_event, key_press_event, and key_release_event.  See the
    implementations of the interactive backends for examples.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level Figure instance
    """

    # The instantiated manager class.  For further customization,
    # ``FigureManager.create_with_canvas`` can also be overridden; see the
    # wx-based backends for an example.
    manager_class = FigureManagerTemplate

    def draw(self):
        """
        Draw the figure using the renderer.

        It is important that this method actually walk the artist tree
        even if not output is produced because this will trigger
        deferred work (like computing limits auto-limits and tick
        values) that users may want access to before saving to disk.
        """
        renderer = RendererTemplate(self.figure.dpi)
        self.figure.draw(renderer)

    # You should provide a print_xxx function for every file format
    # you can write.

    # If the file type is not in the base set of filetypes,
    # you should add it to the class-scope filetypes dictionary as follows:
    filetypes = {**FigureCanvasBase.filetypes, 'foo': 'My magic Foo format'}

    def print_foo(self, filename, **kwargs):
        """
        Write out format foo.

        This method is normally called via `.Figure.savefig` and
        `.FigureCanvasBase.print_figure`, which take care of setting the figure
        facecolor, edgecolor, and dpi to the desired output values, and will
        restore them to the original values.  Therefore, `print_foo` does not
        need to handle these settings.
        """
        self.draw()

    def get_default_filetype(self):
        return 'foo'


########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate
<|endoftext|>"""Displays Agg images in the browser, with interactivity."""

# The WebAgg backend is divided into two modules:
#
# - `backend_webagg_core.py` contains code necessary to embed a WebAgg
#   plot inside of a web application, and communicate in an abstract
#   way over a web socket.
#
# - `backend_webagg.py` contains a concrete implementation of a basic
#   application, implemented with tornado.

from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading

try:
    import tornado.web
    import tornado.ioloop
    import tornado.websocket
except ImportError as err:
    raise RuntimeError("The WebAgg backend requires Tornado.") from err

import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
    TimerAsyncio, TimerTornado)


webagg_server_thread = threading.Thread(
    target=lambda: tornado.ioloop.IOLoop.instance().start())


class FigureManagerWebAgg(core.FigureManagerWebAgg):
    _toolbar2_class = core.NavigationToolbar2WebAgg

    @classmethod
    def pyplot_show(cls, *, block=None):
        WebAggApplication.initialize()

        url = "http://{address}:{port}{prefix}".format(
            address=WebAggApplication.address,
            port=WebAggApplication.port,
            prefix=WebAggApplication.url_prefix)

        if mpl.rcParams['webagg.open_in_browser']:
            import webbrowser
            if not webbrowser.open(url):
                print(f"To view figure, visit {url}")
        else:
            print(f"To view figure, visit {url}")

        WebAggApplication.start()


class FigureCanvasWebAgg(core.FigureCanvasWebAggCore):
    manager_class = FigureManagerWebAgg


class WebAggApplication(tornado.web.Application):
    initialized = False
    started = False

    class FavIcon(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Content-Type', 'image/png')
            self.write(Path(mpl.get_data_path(),
                            'images/matplotlib.png').read_bytes())

    class SingleFigurePage(tornado.web.RequestHandler):
        def __init__(self, application, request, *, url_prefix='', **kwargs):
            self.url_prefix = url_prefix
            super().__init__(application, request, **kwargs)

        def get(self, fignum):
            fignum = int(fignum)
            manager = Gcf.get_fig_manager(fignum)

            ws_uri = f'ws://{self.request.host}{self.url_prefix}/'
            self.render(
                "single_figure.html",
                prefix=self.url_prefix,
                ws_uri=ws_uri,
                fig_id=fignum,
                toolitems=core.NavigationToolbar2WebAgg.toolitems,
                canvas=manager.canvas)

    class AllFiguresPage(tornado.web.RequestHandler):
        def __init__(self, application, request, *, url_prefix='', **kwargs):
            self.url_prefix = url_prefix
            super().__init__(application, request, **kwargs)

        def get(self):
            ws_uri = f'ws://{self.request.host}{self.url_prefix}/'
            self.render(
                "all_figures.html",
                prefix=self.url_prefix,
                ws_uri=ws_uri,
                figures=sorted(Gcf.figs.items()),
                toolitems=core.NavigationToolbar2WebAgg.toolitems)

    class MplJs(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Content-Type', 'application/javascript')

            js_content = core.FigureManagerWebAgg.get_javascript()

            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        def get(self, fignum, fmt):
            fignum = int(fignum)
            manager = Gcf.get_fig_manager(fignum)
            self.set_header(
                'Content-Type', mimetypes.types_map.get(fmt, 'binary'))
            buff = BytesIO()
            manager.canvas.figure.savefig(buff, format=fmt)
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        supports_binary = True

        def open(self, fignum):
            self.fignum = int(fignum)
            self.manager = Gcf.get_fig_manager(self.fignum)
            self.manager.add_web_socket(self)
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            self.manager.remove_web_socket(self)

        def on_message(self, message):
            message = json.loads(message)
            # The 'supports_binary' message is on a client-by-client
            # basis.  The others affect the (shared) canvas as a
            # whole.
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = Gcf.get_fig_manager(self.fignum)
                # It is possible for a figure to be closed,
                # but a stale figure UI is still sending messages
                # from the browser.
                if manager is not None:
                    manager.handle_json(message)

        def send_json(self, content):
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = "data:image/png;base64,{}".format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)

    def __init__(self, url_prefix=''):
        if url_prefix:
            assert url_prefix[0] == '/' and url_prefix[-1] != '/', \
                'url_prefix must start with a "/" and not end with one.'

        super().__init__(
            [
                # Static files for the CSS and JS
                (url_prefix + r'/_static/(.*)',
                 tornado.web.StaticFileHandler,
                 {'path': core.FigureManagerWebAgg.get_static_file_path()}),

                # Static images for the toolbar
                (url_prefix + r'/_images/(.*)',
                 tornado.web.StaticFileHandler,
                 {'path': Path(mpl.get_data_path(), 'images')}),

                # A Matplotlib favicon
                (url_prefix + r'/favicon.ico', self.FavIcon),

                # The page that contains all of the pieces
                (url_prefix + r'/([0-9]+)', self.SingleFigurePage,
                 {'url_prefix': url_prefix}),

                # The page that contains all of the figures
                (url_prefix + r'/?', self.AllFiguresPage,
                 {'url_prefix': url_prefix}),

                (url_prefix + r'/js/mpl.js', self.MplJs),

                # Sends images and events to the browser, and receives
                # events from the browser
                (url_prefix + r'/([0-9]+)/ws', self.WebSocket),

                # Handles the downloading (i.e., saving) of static images
                (url_prefix + r'/([0-9]+)/download.([a-z0-9.]+)',
                 self.Download),
            ],
            template_path=core.FigureManagerWebAgg.get_static_file_path())

    @classmethod
    def initialize(cls, url_prefix='', port=None, address=None):
        if cls.initialized:
            return

        # Create the class instance
        app = cls(url_prefix=url_prefix)

        cls.url_prefix = url_prefix

        # This port selection algorithm is borrowed, more or less
        # verbatim, from IPython.
        def random_ports(port, n):
            """
            Generate a list of n random ports near the given port.

            The first 5 ports will be sequential, and the remaining n-5 will be
            randomly selected in the range [port-2*n, port+2*n].
            """
            for i in range(min(5, n)):
                yield port + i
            for i in range(n - 5):
                yield port + random.randint(-2 * n, 2 * n)

        if address is None:
            cls.address = mpl.rcParams['webagg.address']
        else:
            cls.address = address
        cls.port = mpl.rcParams['webagg.port']
        for port in random_ports(cls.port,
                                 mpl.rcParams['webagg.port_retries']):
            try:
                app.listen(port, cls.address)
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    raise
            else:
                cls.port = port
                break
        else:
            raise SystemExit(
                "The webagg server could not be started because an available "
                "port could not be found")

        cls.initialized = True

    @classmethod
    def start(cls):
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            cls.started = True

        if cls.started:
            return

        """
        IOLoop.running() was removed as of Tornado 2.4; see for example
        https://groups.google.com/forum/#!topic/python-tornado/QLMzkpQBGOY
        Thus there is no correct way to check if the loop has already been
        launched. We may end up with two concurrently running loops in that
        unlucky case with all the expected consequences.
        """
        ioloop = tornado.ioloop.IOLoop.instance()

        def shutdown():
            ioloop.stop()
            print("Server is stopped")
            sys.stdout.flush()
            cls.started = False

        @contextmanager
        def catch_sigint():
            old_handler = signal.signal(
                signal.SIGINT,
                lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
            try:
                yield
            finally:
                signal.signal(signal.SIGINT, old_handler)

        # Set the flag to True *before* blocking on ioloop.start()
        cls.started = True

        print("Press Ctrl+C to stop WebAgg server")
        sys.stdout.flush()
        with catch_sigint():
            ioloop.start()


def ipython_inline_display(figure):
    import tornado.template

    WebAggApplication.initialize()
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if not webagg_server_thread.is_alive():
            webagg_server_thread.start()

    fignum = figure.number
    tpl = Path(core.FigureManagerWebAgg.get_static_file_path(),
               "ipython_inline_figure.html").read_text()
    t = tornado.template.Template(tpl)
    return t.generate(
        prefix=WebAggApplication.url_prefix,
        fig_id=fignum,
        toolitems=core.NavigationToolbar2WebAgg.toolitems,
        canvas=figure.canvas,
        port=WebAggApplication.port).decode('utf-8')


@_Backend.export
class _BackendWebAgg(_Backend):
    FigureCanvas = FigureCanvasWebAgg
    FigureManager = FigureManagerWebAgg
<|endoftext|>import ctypes

from .backend_cairo import cairo, FigureCanvasCairo
from .backend_qt import _BackendQT, FigureCanvasQT
from .qt_compat import QT_API, QtCore, QtGui


class FigureCanvasQTCairo(FigureCanvasCairo, FigureCanvasQT):
    def draw(self):
        if hasattr(self._renderer.gc, "ctx"):
            self._renderer.dpi = self.figure.dpi
            self.figure.draw(self._renderer)
        super().draw()

    def paintEvent(self, event):
        width = int(self.device_pixel_ratio * self.width())
        height = int(self.device_pixel_ratio * self.height())
        if (width, height) != self._renderer.get_canvas_width_height():
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            self._renderer.set_context(cairo.Context(surface))
            self._renderer.dpi = self.figure.dpi
            self.figure.draw(self._renderer)
        buf = self._renderer.gc.ctx.get_target().get_data()
        if QT_API == "PyQt6":
            from PyQt6 import sip
            ptr = int(sip.voidptr(buf))
        else:
            ptr = buf
        qimage = QtGui.QImage(
            ptr, width, height,
            QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        # Adjust the buf reference count to work around a memory leak bug in
        # QImage under PySide.
        if QT_API == "PySide2" and QtCore.__version_info__ < (5, 12):
            ctypes.c_long.from_address(id(buf)).value = 1
        qimage.setDevicePixelRatio(self.device_pixel_ratio)
        painter = QtGui.QPainter(self)
        painter.eraseRect(event.rect())
        painter.drawImage(0, 0, qimage)
        self._draw_rect_callback(painter)
        painter.end()


@_BackendQT.export
class _BackendQTCairo(_BackendQT):
    FigureCanvas = FigureCanvasQTCairo
<|endoftext|>import numpy as np

from .. import cbook
from . import backend_agg, backend_gtk4
from .backend_gtk4 import GLib, Gtk, _BackendGTK4

import cairo  # Presence of cairo is already checked by _backend_gtk.


class FigureCanvasGTK4Agg(backend_agg.FigureCanvasAgg,
                          backend_gtk4.FigureCanvasGTK4):

    def on_draw_event(self, widget, ctx):
        if self._idle_draw_id:
            GLib.source_remove(self._idle_draw_id)
            self._idle_draw_id = 0
            self.draw()

        scale = self.device_pixel_ratio
        allocation = self.get_allocation()

        Gtk.render_background(
            self.get_style_context(), ctx,
            allocation.x, allocation.y,
            allocation.width, allocation.height)

        buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
            np.asarray(self.get_renderer().buffer_rgba()))
        height, width, _ = buf.shape
        image = cairo.ImageSurface.create_for_data(
            buf.ravel().data, cairo.FORMAT_ARGB32, width, height)
        image.set_device_scale(scale, scale)
        ctx.set_source_surface(image, 0, 0)
        ctx.paint()

        return False


@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
<|endoftext|>import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
     _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase


_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# SimpleXMLWriter class
#
# Based on an original by Fredrik Lundh, but modified here to:
#   1. Support modern Python idioms
#   2. Remove encoding support (it's handled by the file writer instead)
#   3. Support proper indentation
#   4. Minify things a little bit

# --------------------------------------------------------------------
# The SimpleXMLWriter module is
#
# Copyright (c) 2001-2004 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------


def _escape_cdata(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


_escape_xml_comment = re.compile(r'-(?=-)')


def _escape_comment(s):
    s = _escape_cdata(s)
    return _escape_xml_comment.sub('- ', s)


def _escape_attrib(s):
    s = s.replace("&", "&amp;")
    s = s.replace("'", "&apos;")
    s = s.replace('"', "&quot;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def _quote_escape_attrib(s):
    return ('"' + _escape_cdata(s) + '"' if '"' not in s else
            "'" + _escape_cdata(s) + "'" if "'" not in s else
            '"' + _escape_attrib(s) + '"')


def _short_float_fmt(x):
    """
    Create a short string representation of a float, which is %f
    formatting with trailing zeros and the decimal point removed.
    """
    return f'{x:f}'.rstrip('0').rstrip('.')


class XMLWriter:
    """
    Parameters
    ----------
    file : writable text file-like object
    """

    def __init__(self, file):
        self.__write = file.write
        if hasattr(file, "flush"):
            self.flush = file.flush
        self.__open = 0  # true if start tag is open
        self.__tags = []
        self.__data = []
        self.__indentation = " " * 64

    def __flush(self, indent=True):
        # flush internal buffers
        if self.__open:
            if indent:
                self.__write(">\n")
            else:
                self.__write(">")
            self.__open = 0
        if self.__data:
            data = ''.join(self.__data)
            self.__write(_escape_cdata(data))
            self.__data = []

    def start(self, tag, attrib={}, **extra):
        """
        Open a new element.  Attributes can be given as keyword
        arguments, or as a string/string dictionary. The method returns
        an opaque identifier that can be passed to the :meth:`close`
        method, to close all open elements up to and including this one.

        Parameters
        ----------
        tag
            Element tag.
        attrib
            Attribute dictionary.  Alternatively, attributes can be given as
            keyword arguments.

        Returns
        -------
        An element identifier.
        """
        self.__flush()
        tag = _escape_cdata(tag)
        self.__data = []
        self.__tags.append(tag)
        self.__write(self.__indentation[:len(self.__tags) - 1])
        self.__write(f"<{tag}")
        for k, v in {**attrib, **extra}.items():
            if v:
                k = _escape_cdata(k)
                v = _quote_escape_attrib(v)
                self.__write(f' {k}={v}')
        self.__open = 1
        return len(self.__tags) - 1

    def comment(self, comment):
        """
        Add a comment to the output stream.

        Parameters
        ----------
        comment : str
            Comment text.
        """
        self.__flush()
        self.__write(self.__indentation[:len(self.__tags)])
        self.__write(f"<!-- {_escape_comment(comment)} -->\n")

    def data(self, text):
        """
        Add character data to the output stream.

        Parameters
        ----------
        text : str
            Character data.
        """
        self.__data.append(text)

    def end(self, tag=None, indent=True):
        """
        Close the current element (opened by the most recent call to
        :meth:`start`).

        Parameters
        ----------
        tag
            Element tag.  If given, the tag must match the start tag.  If
            omitted, the current element is closed.
        indent : bool, default: True
        """
        if tag:
            assert self.__tags, f"unbalanced end({tag})"
            assert _escape_cdata(tag) == self.__tags[-1], \
                f"expected end({self.__tags[-1]}), got {tag}"
        else:
            assert self.__tags, "unbalanced end()"
        tag = self.__tags.pop()
        if self.__data:
            self.__flush(indent)
        elif self.__open:
            self.__open = 0
            self.__write("/>\n")
            return
        if indent:
            self.__write(self.__indentation[:len(self.__tags)])
        self.__write(f"</{tag}>\n")

    def close(self, id):
        """
        Close open elements, up to (and including) the element identified
        by the given identifier.

        Parameters
        ----------
        id
            Element identifier, as returned by the :meth:`start` method.
        """
        while len(self.__tags) > id:
            self.end()

    def element(self, tag, text=None, attrib={}, **extra):
        """
        Add an entire element.  This is the same as calling :meth:`start`,
        :meth:`data`, and :meth:`end` in sequence. The *text* argument can be
        omitted.
        """
        self.start(tag, attrib, **extra)
        if text:
            self.data(text)
        self.end(indent=False)

    def flush(self):
        """Flush the output stream."""
        pass  # replaced by the constructor


def _generate_transform(transform_list):
    parts = []
    for type, value in transform_list:
        if (type == 'scale' and (value == (1,) or value == (1, 1))
                or type == 'translate' and value == (0, 0)
                or type == 'rotate' and value == (0,)):
            continue
        if type == 'matrix' and isinstance(value, Affine2DBase):
            value = value.to_values()
        parts.append('{}({})'.format(
            type, ' '.join(_short_float_fmt(x) for x in value)))
    return ' '.join(parts)


def _generate_css(attrib):
    return "; ".join(f"{k}: {v}" for k, v in attrib.items())


_capstyle_d = {'projecting': 'square', 'butt': 'butt', 'round': 'round'}


def _check_is_str(info, key):
    if not isinstance(info, str):
        raise TypeError(f'Invalid type for {key} metadata. Expected str, not '
                        f'{type(info)}.')


def _check_is_iterable_of_str(infos, key):
    if np.iterable(infos):
        for info in infos:
            if not isinstance(info, str):
                raise TypeError(f'Invalid type for {key} metadata. Expected '
                                f'iterable of str, not {type(info)}.')
    else:
        raise TypeError(f'Invalid type for {key} metadata. Expected str or '
                        f'iterable of str, not {type(infos)}.')


class RendererSVG(RendererBase):
    def __init__(self, width, height, svgwriter, basename=None, image_dpi=72,
                 *, metadata=None):
        self.width = width
        self.height = height
        self.writer = XMLWriter(svgwriter)
        self.image_dpi = image_dpi  # actual dpi at which we rasterize stuff

        if basename is None:
            basename = getattr(svgwriter, "name", "")
            if not isinstance(basename, str):
                basename = ""
        self.basename = basename

        self._groupd = {}
        self._image_counter = itertools.count()
        self._clip_path_ids = {}
        self._clipd = {}
        self._markers = {}
        self._path_collection_id = 0
        self._hatchd = {}
        self._has_gouraud = False
        self._n_gradients = 0

        super().__init__()
        self._glyph_map = dict()
        str_height = _short_float_fmt(height)
        str_width = _short_float_fmt(width)
        svgwriter.write(svgProlog)
        self._start_id = self.writer.start(
            'svg',
            width=f'{str_width}pt',
            height=f'{str_height}pt',
            viewBox=f'0 0 {str_width} {str_height}',
            xmlns="http://www.w3.org/2000/svg",
            version="1.1",
            id=mpl.rcParams['svg.id'],
            attrib={'xmlns:xlink': "http://www.w3.org/1999/xlink"})
        self._write_metadata(metadata)
        self._write_default_style()

    def _get_clippath_id(self, clippath):
        """
        Returns a stable and unique identifier for the *clippath* argument
        object within the current rendering context.

        This allows plots that include custom clip paths to produce identical
        SVG output on each render, provided that the :rc:`svg.hashsalt` config
        setting and the ``SOURCE_DATE_EPOCH`` build-time environment variable
        are set to fixed values.
        """
        if clippath not in self._clip_path_ids:
            self._clip_path_ids[clippath] = len(self._clip_path_ids)
        return self._clip_path_ids[clippath]

    def finalize(self):
        self._write_clips()
        self._write_hatches()
        self.writer.close(self._start_id)
        self.writer.flush()

    def _write_metadata(self, metadata):
        # Add metadata following the Dublin Core Metadata Initiative, and the
        # Creative Commons Rights Expression Language. This is mainly for
        # compatibility with Inkscape.
        if metadata is None:
            metadata = {}
        metadata = {
            'Format': 'image/svg+xml',
            'Type': 'http://purl.org/dc/dcmitype/StillImage',
            'Creator':
                f'Matplotlib v{mpl.__version__}, https://matplotlib.org/',
            **metadata
        }
        writer = self.writer

        if 'Title' in metadata:
            title = metadata['Title']
            _check_is_str(title, 'Title')
            writer.element('title', text=title)

        # Special handling.
        date = metadata.get('Date', None)
        if date is not None:
            if isinstance(date, str):
                dates = [date]
            elif isinstance(date, (datetime.datetime, datetime.date)):
                dates = [date.isoformat()]
            elif np.iterable(date):
                dates = []
                for d in date:
                    if isinstance(d, str):
                        dates.append(d)
                    elif isinstance(d, (datetime.datetime, datetime.date)):
                        dates.append(d.isoformat())
                    else:
                        raise TypeError(
                            f'Invalid type for Date metadata. '
                            f'Expected iterable of str, date, or datetime, '
                            f'not {type(d)}.')
            else:
                raise TypeError(f'Invalid type for Date metadata. '
                                f'Expected str, date, datetime, or iterable '
                                f'of the same, not {type(date)}.')
            metadata['Date'] = '/'.join(dates)
        elif 'Date' not in metadata:
            # Do not add `Date` if the user explicitly set `Date` to `None`
            # Get source date from SOURCE_DATE_EPOCH, if set.
            # See https://reproducible-builds.org/specs/source-date-epoch/
            date = os.getenv("SOURCE_DATE_EPOCH")
            if date:
                date = datetime.datetime.fromtimestamp(int(date), datetime.timezone.utc)
                metadata['Date'] = date.replace(tzinfo=UTC).isoformat()
            else:
                metadata['Date'] = datetime.datetime.today().isoformat()

        mid = None
        def ensure_metadata(mid):
            if mid is not None:
                return mid
            mid = writer.start('metadata')
            writer.start('rdf:RDF', attrib={
                'xmlns:dc': "http://purl.org/dc/elements/1.1/",
                'xmlns:cc': "http://creativecommons.org/ns#",
                'xmlns:rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            })
            writer.start('cc:Work')
            return mid

        uri = metadata.pop('Type', None)
        if uri is not None:
            mid = ensure_metadata(mid)
            writer.element('dc:type', attrib={'rdf:resource': uri})

        # Single value only.
        for key in ['Title', 'Coverage', 'Date', 'Description', 'Format',
                    'Identifier', 'Language', 'Relation', 'Source']:
            info = metadata.pop(key, None)
            if info is not None:
                mid = ensure_metadata(mid)
                _check_is_str(info, key)
                writer.element(f'dc:{key.lower()}', text=info)

        # Multiple Agent values.
        for key in ['Creator', 'Contributor', 'Publisher', 'Rights']:
            agents = metadata.pop(key, None)
            if agents is None:
                continue

            if isinstance(agents, str):
                agents = [agents]

            _check_is_iterable_of_str(agents, key)
            # Now we know that we have an iterable of str
            mid = ensure_metadata(mid)
            writer.start(f'dc:{key.lower()}')
            for agent in agents:
                writer.start('cc:Agent')
                writer.element('dc:title', text=agent)
                writer.end('cc:Agent')
            writer.end(f'dc:{key.lower()}')

        # Multiple values.
        keywords = metadata.pop('Keywords', None)
        if keywords is not None:
            if isinstance(keywords, str):
                keywords = [keywords]
            _check_is_iterable_of_str(keywords, 'Keywords')
            # Now we know that we have an iterable of str
            mid = ensure_metadata(mid)
            writer.start('dc:subject')
            writer.start('rdf:Bag')
            for keyword in keywords:
                writer.element('rdf:li', text=keyword)
            writer.end('rdf:Bag')
            writer.end('dc:subject')

        if mid is not None:
            writer.close(mid)

        if metadata:
            raise ValueError('Unknown metadata key(s) passed to SVG writer: ' +
                             ','.join(metadata))

    def _write_default_style(self):
        writer = self.writer
        default_style = _generate_css({
            'stroke-linejoin': 'round',
            'stroke-linecap': 'butt'})
        writer.start('defs')
        writer.element('style', type='text/css', text='*{%s}' % default_style)
        writer.end('defs')

    def _make_id(self, type, content):
        salt = mpl.rcParams['svg.hashsalt']
        if salt is None:
            salt = str(uuid.uuid4())
        m = hashlib.sha256()
        m.update(salt.encode('utf8'))
        m.update(str(content).encode('utf8'))
        return f'{type}{m.hexdigest()[:10]}'

    def _make_flip_transform(self, transform):
        return transform + Affine2D().scale(1, -1).translate(0, self.height)

    def _get_hatch(self, gc, rgbFace):
        """
        Create a new hatch pattern
        """
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        edge = gc.get_hatch_color()
        if edge is not None:
            edge = tuple(edge)
        lw = gc.get_hatch_linewidth()
        dictkey = (gc.get_hatch(), rgbFace, edge, lw)
        oid = self._hatchd.get(dictkey)
        if oid is None:
            oid = self._make_id('h', dictkey)
            self._hatchd[dictkey] = ((gc.get_hatch_path(), rgbFace, edge, lw), oid)
        else:
            _, oid = oid
        return oid

    def _write_hatches(self):
        if not len(self._hatchd):
            return
        HATCH_SIZE = 72
        writer = self.writer
        writer.start('defs')
        for (path, face, stroke, lw), oid in self._hatchd.values():
            writer.start(
                'pattern',
                id=oid,
                patternUnits="userSpaceOnUse",
                x="0", y="0", width=str(HATCH_SIZE),
                height=str(HATCH_SIZE))
            path_data = self._convert_path(
                path,
                Affine2D()
                .scale(HATCH_SIZE).scale(1.0, -1.0).translate(0, HATCH_SIZE),
                simplify=False)
            if face is None:
                fill = 'none'
            else:
                fill = rgb2hex(face)
            writer.element(
                'rect',
                x="0", y="0", width=str(HATCH_SIZE+1),
                height=str(HATCH_SIZE+1),
                fill=fill)
            hatch_style = {
                    'fill': rgb2hex(stroke),
                    'stroke': rgb2hex(stroke),
                    'stroke-width': str(lw),
                    'stroke-linecap': 'butt',
                    'stroke-linejoin': 'miter'
                    }
            if stroke[3] < 1:
                hatch_style['stroke-opacity'] = str(stroke[3])
            writer.element(
                'path',
                d=path_data,
                style=_generate_css(hatch_style)
                )
            writer.end('pattern')
        writer.end('defs')

    def _get_style_dict(self, gc, rgbFace):
        """Generate a style string from the GraphicsContext and rgbFace."""
        attrib = {}

        forced_alpha = gc.get_forced_alpha()

        if gc.get_hatch() is not None:
            attrib['fill'] = f"url(#{self._get_hatch(gc, rgbFace)})"
            if (rgbFace is not None and len(rgbFace) == 4 and rgbFace[3] != 1.0
                    and not forced_alpha):
                attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])
        else:
            if rgbFace is None:
                attrib['fill'] = 'none'
            else:
                if tuple(rgbFace[:3]) != (0, 0, 0):
                    attrib['fill'] = rgb2hex(rgbFace)
                if (len(rgbFace) == 4 and rgbFace[3] != 1.0
                        and not forced_alpha):
                    attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])

        if forced_alpha and gc.get_alpha() != 1.0:
            attrib['opacity'] = _short_float_fmt(gc.get_alpha())

        offset, seq = gc.get_dashes()
        if seq is not None:
            attrib['stroke-dasharray'] = ','.join(
                _short_float_fmt(val) for val in seq)
            attrib['stroke-dashoffset'] = _short_float_fmt(float(offset))

        linewidth = gc.get_linewidth()
        if linewidth:
            rgb = gc.get_rgb()
            attrib['stroke'] = rgb2hex(rgb)
            if not forced_alpha and rgb[3] != 1.0:
                attrib['stroke-opacity'] = _short_float_fmt(rgb[3])
            if linewidth != 1.0:
                attrib['stroke-width'] = _short_float_fmt(linewidth)
            if gc.get_joinstyle() != 'round':
                attrib['stroke-linejoin'] = gc.get_joinstyle()
            if gc.get_capstyle() != 'butt':
                attrib['stroke-linecap'] = _capstyle_d[gc.get_capstyle()]

        return attrib

    def _get_style(self, gc, rgbFace):
        return _generate_css(self._get_style_dict(gc, rgbFace))

    def _get_clip_attrs(self, gc):
        cliprect = gc.get_clip_rectangle()
        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            clippath_trans = self._make_flip_transform(clippath_trans)
            dictkey = (self._get_clippath_id(clippath), str(clippath_trans))
        elif cliprect is not None:
            x, y, w, h = cliprect.bounds
            y = self.height-(y+h)
            dictkey = (x, y, w, h)
        else:
            return {}
        clip = self._clipd.get(dictkey)
        if clip is None:
            oid = self._make_id('p', dictkey)
            if clippath is not None:
                self._clipd[dictkey] = ((clippath, clippath_trans), oid)
            else:
                self._clipd[dictkey] = (dictkey, oid)
        else:
            _, oid = clip
        return {'clip-path': f'url(#{oid})'}

    def _write_clips(self):
        if not len(self._clipd):
            return
        writer = self.writer
        writer.start('defs')
        for clip, oid in self._clipd.values():
            writer.start('clipPath', id=oid)
            if len(clip) == 2:
                clippath, clippath_trans = clip
                path_data = self._convert_path(
                    clippath, clippath_trans, simplify=False)
                writer.element('path', d=path_data)
            else:
                x, y, w, h = clip
                writer.element(
                    'rect',
                    x=_short_float_fmt(x),
                    y=_short_float_fmt(y),
                    width=_short_float_fmt(w),
                    height=_short_float_fmt(h))
            writer.end('clipPath')
        writer.end('defs')

    def open_group(self, s, gid=None):
        # docstring inherited
        if gid:
            self.writer.start('g', id=gid)
        else:
            self._groupd[s] = self._groupd.get(s, 0) + 1
            self.writer.start('g', id=f"{s}_{self._groupd[s]:d}")

    def close_group(self, s):
        # docstring inherited
        self.writer.end('g')

    def option_image_nocomposite(self):
        # docstring inherited
        return not mpl.rcParams['image.composite_image']

    def _convert_path(self, path, transform=None, clip=None, simplify=None,
                      sketch=None):
        if clip:
            clip = (0.0, 0.0, self.width, self.height)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, sketch, 6,
            [b'M', b'L', b'Q', b'C', b'z'], False).decode('ascii')

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        trans_and_flip = self._make_flip_transform(transform)
        clip = (rgbFace is None and gc.get_hatch_path() is None)
        simplify = path.should_simplify and clip
        path_data = self._convert_path(
            path, trans_and_flip, clip=clip, simplify=simplify,
            sketch=gc.get_sketch_params())

        if gc.get_url() is not None:
            self.writer.start('a', {'xlink:href': gc.get_url()})
        self.writer.element('path', d=path_data, **self._get_clip_attrs(gc),
                            style=self._get_style(gc, rgbFace))
        if gc.get_url() is not None:
            self.writer.end('a')

    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # docstring inherited

        if not len(path.vertices):
            return

        writer = self.writer
        path_data = self._convert_path(
            marker_path,
            marker_trans + Affine2D().scale(1.0, -1.0),
            simplify=False)
        style = self._get_style_dict(gc, rgbFace)
        dictkey = (path_data, _generate_css(style))
        oid = self._markers.get(dictkey)
        style = _generate_css({k: v for k, v in style.items()
                              if k.startswith('stroke')})

        if oid is None:
            oid = self._make_id('m', dictkey)
            writer.start('defs')
            writer.element('path', id=oid, d=path_data, style=style)
            writer.end('defs')
            self._markers[dictkey] = oid

        writer.start('g', **self._get_clip_attrs(gc))
        if gc.get_url() is not None:
            self.writer.start('a', {'xlink:href': gc.get_url()})
        trans_and_flip = self._make_flip_transform(trans)
        attrib = {'xlink:href': f'#{oid}'}
        clip = (0, 0, self.width*72, self.height*72)
        for vertices, code in path.iter_segments(
                trans_and_flip, clip=clip, simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                attrib['x'] = _short_float_fmt(x)
                attrib['y'] = _short_float_fmt(y)
                attrib['style'] = self._get_style(gc, rgbFace)
                writer.element('use', attrib=attrib)
        if gc.get_url() is not None:
            self.writer.end('a')
        writer.end('g')

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # Is the optimization worth it? Rough calculation:
        # cost of emitting a path in-line is
        #    (len_path + 5) * uses_per_path
        # cost of definition+use is
        #    (len_path + 3) + 9 * uses_per_path
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = \
            len_path + 9 * uses_per_path + 3 < (len_path + 5) * uses_per_path
        if not should_do_optimization:
            return super().draw_path_collection(
                gc, master_transform, paths, all_transforms,
                offsets, offset_trans, facecolors, edgecolors,
                linewidths, linestyles, antialiaseds, urls,
                offset_position)

        writer = self.writer
        path_codes = []
        writer.start('defs')
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                master_transform, paths, all_transforms)):
            transform = Affine2D(transform.get_matrix()).scale(1.0, -1.0)
            d = self._convert_path(path, transform, simplify=False)
            oid = 'C{:x}_{:x}_{}'.format(
                self._path_collection_id, i, self._make_id('', d))
            writer.element('path', id=oid, d=d)
            path_codes.append(oid)
        writer.end('defs')

        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, path_codes, offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            url = gc0.get_url()
            if url is not None:
                writer.start('a', attrib={'xlink:href': url})
            clip_attrs = self._get_clip_attrs(gc0)
            if clip_attrs:
                writer.start('g', **clip_attrs)
            attrib = {
                'xlink:href': f'#{path_id}',
                'x': _short_float_fmt(xo),
                'y': _short_float_fmt(self.height - yo),
                'style': self._get_style(gc0, rgbFace)
                }
            writer.element('use', attrib=attrib)
            if clip_attrs:
                writer.end('g')
            if url is not None:
                writer.end('a')

        self._path_collection_id += 1

    def _draw_gouraud_triangle(self, transformed_points, colors):
        # This uses a method described here:
        #
        #   http://www.svgopen.org/2005/papers/Converting3DFaceToSVG/index.html
        #
        # that uses three overlapping linear gradients to simulate a
        # Gouraud triangle.  Each gradient goes from fully opaque in
        # one corner to fully transparent along the opposite edge.
        # The line between the stop points is perpendicular to the
        # opposite edge.  Underlying these three gradients is a solid
        # triangle whose color is the average of all three points.

        avg_color = np.average(colors, axis=0)
        if avg_color[-1] == 0:
            # Skip fully-transparent triangles
            return

        writer = self.writer
        writer.start('defs')
        for i in range(3):
            x1, y1 = transformed_points[i]
            x2, y2 = transformed_points[(i + 1) % 3]
            x3, y3 = transformed_points[(i + 2) % 3]
            rgba_color = colors[i]

            if x2 == x3:
                xb = x2
                yb = y1
            elif y2 == y3:
                xb = x1
                yb = y2
            else:
                m1 = (y2 - y3) / (x2 - x3)
                b1 = y2 - (m1 * x2)
                m2 = -(1.0 / m1)
                b2 = y1 - (m2 * x1)
                xb = (-b1 + b2) / (m1 - m2)
                yb = m2 * xb + b2

            writer.start(
                'linearGradient',
                id=f"GR{self._n_gradients:x}_{i:d}",
                gradientUnits="userSpaceOnUse",
                x1=_short_float_fmt(x1), y1=_short_float_fmt(y1),
                x2=_short_float_fmt(xb), y2=_short_float_fmt(yb))
            writer.element(
                'stop',
                offset='1',
                style=_generate_css({
                    'stop-color': rgb2hex(avg_color),
                    'stop-opacity': _short_float_fmt(rgba_color[-1])}))
            writer.element(
                'stop',
                offset='0',
                style=_generate_css({'stop-color': rgb2hex(rgba_color),
                                    'stop-opacity': "0"}))

            writer.end('linearGradient')

        writer.end('defs')

        # triangle formation using "path"
        dpath = (f"M {_short_float_fmt(x1)},{_short_float_fmt(y1)}"
                 f" L {_short_float_fmt(x2)},{_short_float_fmt(y2)}"
                 f" {_short_float_fmt(x3)},{_short_float_fmt(y3)} Z")

        writer.element(
            'path',
            attrib={'d': dpath,
                    'fill': rgb2hex(avg_color),
                    'fill-opacity': '1',
                    'shape-rendering': "crispEdges"})

        writer.start(
                'g',
                attrib={'stroke': "none",
                        'stroke-width': "0",
                        'shape-rendering': "crispEdges",
                        'filter': "url(#colorMat)"})

        writer.element(
            'path',
            attrib={'d': dpath,
                    'fill': f'url(#GR{self._n_gradients:x}_0)',
                    'shape-rendering': "crispEdges"})

        writer.element(
            'path',
            attrib={'d': dpath,
                    'fill': f'url(#GR{self._n_gradients:x}_1)',
                    'filter': 'url(#colorAdd)',
                    'shape-rendering': "crispEdges"})

        writer.element(
            'path',
            attrib={'d': dpath,
                    'fill': f'url(#GR{self._n_gradients:x}_2)',
                    'filter': 'url(#colorAdd)',
                    'shape-rendering': "crispEdges"})

        writer.end('g')

        self._n_gradients += 1

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        writer = self.writer
        writer.start('g', **self._get_clip_attrs(gc))
        transform = transform.frozen()
        trans_and_flip = self._make_flip_transform(transform)

        if not self._has_gouraud:
            self._has_gouraud = True
            writer.start(
                'filter',
                id='colorAdd')
            writer.element(
                'feComposite',
                attrib={'in': 'SourceGraphic'},
                in2='BackgroundImage',
                operator='arithmetic',
                k2="1", k3="1")
            writer.end('filter')
            # feColorMatrix filter to correct opacity
            writer.start(
                'filter',
                id='colorMat')
            writer.element(
                'feColorMatrix',
                attrib={'type': 'matrix'},
                values='1 0 0 0 0 \n0 1 0 0 0 \n0 0 1 0 0 \n1 1 1 1 0 \n0 0 0 0 1 ')
            writer.end('filter')

        for points, colors in zip(triangles_array, colors_array):
            self._draw_gouraud_triangle(trans_and_flip.transform(points), colors)
        writer.end('g')

    def option_scale_image(self):
        # docstring inherited
        return True

    def get_image_magnification(self):
        return self.image_dpi / 72.0

    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]

        if w == 0 or h == 0:
            return

        clip_attrs = self._get_clip_attrs(gc)
        if clip_attrs:
            # Can't apply clip-path directly to the image because the image has
            # a transformation, which would also be applied to the clip-path.
            self.writer.start('g', **clip_attrs)

        url = gc.get_url()
        if url is not None:
            self.writer.start('a', attrib={'xlink:href': url})

        attrib = {}
        oid = gc.get_gid()
        if mpl.rcParams['svg.image_inline']:
            buf = BytesIO()
            Image.fromarray(im).save(buf, format="png")
            oid = oid or self._make_id('image', buf.getvalue())
            attrib['xlink:href'] = (
                "data:image/png;base64,\n" +
                base64.b64encode(buf.getvalue()).decode('ascii'))
        else:
            if self.basename is None:
                raise ValueError("Cannot save image data to filesystem when "
                                 "writing SVG to an in-memory buffer")
            filename = f'{self.basename}.image{next(self._image_counter)}.png'
            _log.info('Writing image file for inclusion: %s', filename)
            Image.fromarray(im).save(filename)
            oid = oid or 'Im_' + self._make_id('image', filename)
            attrib['xlink:href'] = filename
        attrib['id'] = oid

        if transform is None:
            w = 72.0 * w / self.image_dpi
            h = 72.0 * h / self.image_dpi

            self.writer.element(
                'image',
                transform=_generate_transform([
                    ('scale', (1, -1)), ('translate', (0, -h))]),
                x=_short_float_fmt(x),
                y=_short_float_fmt(-(self.height - y - h)),
                width=_short_float_fmt(w), height=_short_float_fmt(h),
                attrib=attrib)
        else:
            alpha = gc.get_alpha()
            if alpha != 1.0:
                attrib['opacity'] = _short_float_fmt(alpha)

            flipped = (
                Affine2D().scale(1.0 / w, 1.0 / h) +
                transform +
                Affine2D()
                .translate(x, y)
                .scale(1.0, -1.0)
                .translate(0.0, self.height))

            attrib['transform'] = _generate_transform(
                [('matrix', flipped.frozen())])
            attrib['style'] = (
                'image-rendering:crisp-edges;'
                'image-rendering:pixelated')
            self.writer.element(
                'image',
                width=_short_float_fmt(w), height=_short_float_fmt(h),
                attrib=attrib)

        if url is not None:
            self.writer.end('a')
        if clip_attrs:
            self.writer.end('g')

    def _update_glyph_map_defs(self, glyph_map_new):
        """
        Emit definitions for not-yet-defined glyphs, and record them as having
        been defined.
        """
        writer = self.writer
        if glyph_map_new:
            writer.start('defs')
            for char_id, (vertices, codes) in glyph_map_new.items():
                char_id = self._adjust_char_id(char_id)
                # x64 to go back to FreeType's internal (integral) units.
                path_data = self._convert_path(
                    Path(vertices * 64, codes), simplify=False)
                writer.element(
                    'path', id=char_id, d=path_data,
                    transform=_generate_transform([('scale', (1 / 64,))]))
            writer.end('defs')
            self._glyph_map.update(glyph_map_new)

    def _adjust_char_id(self, char_id):
        return char_id.replace("%20", "_")

    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext=None):
        # docstring inherited
        writer = self.writer

        writer.comment(s)

        glyph_map = self._glyph_map

        text2path = self._text2path
        color = rgb2hex(gc.get_rgb())
        fontsize = prop.get_size_in_points()

        style = {}
        if color != '#000000':
            style['fill'] = color
        alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
        if alpha != 1:
            style['opacity'] = _short_float_fmt(alpha)
        font_scale = fontsize / text2path.FONT_SCALE
        attrib = {
            'style': _generate_css(style),
            'transform': _generate_transform([
                ('translate', (x, y)),
                ('rotate', (-angle,)),
                ('scale', (font_scale, -font_scale))]),
        }
        writer.start('g', attrib=attrib)

        if not ismath:
            font = text2path._get_font(prop)
            _glyphs = text2path.get_glyphs_with_font(
                font, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            glyph_info, glyph_map_new, rects = _glyphs
            self._update_glyph_map_defs(glyph_map_new)

            for glyph_id, xposition, yposition, scale in glyph_info:
                writer.element(
                    'use',
                    transform=_generate_transform([
                        ('translate', (xposition, yposition)),
                        ('scale', (scale,)),
                        ]),
                    attrib={'xlink:href': f'#{glyph_id}'})

        else:
            if ismath == "TeX":
                _glyphs = text2path.get_glyphs_tex(
                    prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            else:
                _glyphs = text2path.get_glyphs_mathtext(
                    prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            glyph_info, glyph_map_new, rects = _glyphs
            self._update_glyph_map_defs(glyph_map_new)

            for char_id, xposition, yposition, scale in glyph_info:
                char_id = self._adjust_char_id(char_id)
                writer.element(
                    'use',
                    transform=_generate_transform([
                        ('translate', (xposition, yposition)),
                        ('scale', (scale,)),
                        ]),
                    attrib={'xlink:href': f'#{char_id}'})

            for verts, codes in rects:
                path = Path(verts, codes)
                path_data = self._convert_path(path, simplify=False)
                writer.element('path', d=path_data)

        writer.end('g')

    def _draw_text_as_text(self, gc, x, y, s, prop, angle, ismath, mtext=None):
        # NOTE: If you change the font styling CSS, then be sure the check for
        # svg.fonttype = none in `lib/matplotlib/testing/compare.py::convert` remains in
        # sync. Also be sure to re-generate any SVG using this mode, or else such tests
        # will fail to use the right converter for the expected images, and they will
        # fail strangely.
        writer = self.writer

        color = rgb2hex(gc.get_rgb())
        font_style = {}
        color_style = {}
        if color != '#000000':
            color_style['fill'] = color

        alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
        if alpha != 1:
            color_style['opacity'] = _short_float_fmt(alpha)

        if not ismath:
            attrib = {}

            # Separate font style in their separate attributes
            if prop.get_style() != 'normal':
                font_style['font-style'] = prop.get_style()
            if prop.get_variant() != 'normal':
                font_style['font-variant'] = prop.get_variant()
            weight = prop.get_weight()
            weight = fm.weight_dict.get(weight, weight)  # convert to int
            if weight != 400:
                font_style['font-weight'] = f'{weight}'

            def _normalize_sans(name):
                return 'sans-serif' if name in ['sans', 'sans serif'] else name

            def _expand_family_entry(fn):
                fn = _normalize_sans(fn)
                # prepend generic font families with all configured font names
                if fn in fm.font_family_aliases:
                    # get all of the font names and fix spelling of sans-serif
                    # (we accept 3 ways CSS only supports 1)
                    for name in fm.FontManager._expand_aliases(fn):
                        yield _normalize_sans(name)
                # whether a generic name or a family name, it must appear at
                # least once
                yield fn

            def _get_all_quoted_names(prop):
                # only quote specific names, not generic names
                return [name if name in fm.font_family_aliases else repr(name)
                        for entry in prop.get_family()
                        for name in _expand_family_entry(entry)]

            font_style['font-size'] = f'{_short_float_fmt(prop.get_size())}px'
            # ensure expansion, quoting, and dedupe of font names
            font_style['font-family'] = ", ".join(
                dict.fromkeys(_get_all_quoted_names(prop))
                )

            if prop.get_stretch() != 'normal':
                font_style['font-stretch'] = prop.get_stretch()
            attrib['style'] = _generate_css({**font_style, **color_style})

            if mtext and (angle == 0 or mtext.get_rotation_mode() == "anchor"):
                # If text anchoring can be supported, get the original
                # coordinates and add alignment information.

                # Get anchor coordinates.
                transform = mtext.get_transform()
                ax, ay = transform.transform(mtext.get_unitless_position())
                ay = self.height - ay

                # Don't do vertical anchor alignment. Most applications do not
                # support 'alignment-baseline' yet. Apply the vertical layout
                # to the anchor point manually for now.
                angle_rad = np.deg2rad(angle)
                dir_vert = np.array([np.sin(angle_rad), np.cos(angle_rad)])
                v_offset = np.dot(dir_vert, [(x - ax), (y - ay)])
                ax = ax + v_offset * dir_vert[0]
                ay = ay + v_offset * dir_vert[1]

                ha_mpl_to_svg = {'left': 'start', 'right': 'end',
                                 'center': 'middle'}
                font_style['text-anchor'] = ha_mpl_to_svg[mtext.get_ha()]

                attrib['x'] = _short_float_fmt(ax)
                attrib['y'] = _short_float_fmt(ay)
                attrib['style'] = _generate_css({**font_style, **color_style})
                attrib['transform'] = _generate_transform([
                    ("rotate", (-angle, ax, ay))])

            else:
                attrib['transform'] = _generate_transform([
                    ('translate', (x, y)),
                    ('rotate', (-angle,))])

            writer.element('text', s, attrib=attrib)

        else:
            writer.comment(s)

            width, height, descent, glyphs, rects = \
                self._text2path.mathtext_parser.parse(s, 72, prop)

            # Apply attributes to 'g', not 'text', because we likely have some
            # rectangles as well with the same style and transformation.
            writer.start('g',
                         style=_generate_css({**font_style, **color_style}),
                         transform=_generate_transform([
                             ('translate', (x, y)),
                             ('rotate', (-angle,))]),
                         )

            writer.start('text')

            # Sort the characters by font, and output one tspan for each.
            spans = {}
            for font, fontsize, thetext, new_x, new_y in glyphs:
                entry = fm.ttfFontProperty(font)
                font_style = {}
                # Separate font style in its separate attributes
                if entry.style != 'normal':
                    font_style['font-style'] = entry.style
                if entry.variant != 'normal':
                    font_style['font-variant'] = entry.variant
                if entry.weight != 400:
                    font_style['font-weight'] = f'{entry.weight}'
                font_style['font-size'] = f'{_short_float_fmt(fontsize)}px'
                font_style['font-family'] = f'{entry.name!r}'  # ensure quoting
                if entry.stretch != 'normal':
                    font_style['font-stretch'] = entry.stretch
                style = _generate_css({**font_style, **color_style})
                if thetext == 32:
                    thetext = 0xa0  # non-breaking space
                spans.setdefault(style, []).append((new_x, -new_y, thetext))

            for style, chars in spans.items():
                chars.sort()  # Sort by increasing x position
                for x, y, t in chars:  # Output one tspan for each character
                    writer.element(
                        'tspan',
                        chr(t),
                        x=_short_float_fmt(x),
                        y=_short_float_fmt(y),
                        style=style)

            writer.end('text')

            for x, y, width, height in rects:
                writer.element(
                    'rect',
                    x=_short_float_fmt(x),
                    y=_short_float_fmt(-y-1),
                    width=_short_float_fmt(width),
                    height=_short_float_fmt(height)
                    )

            writer.end('g')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        clip_attrs = self._get_clip_attrs(gc)
        if clip_attrs:
            # Cannot apply clip-path directly to the text, because
            # it has a transformation
            self.writer.start('g', **clip_attrs)

        if gc.get_url() is not None:
            self.writer.start('a', {'xlink:href': gc.get_url()})

        if mpl.rcParams['svg.fonttype'] == 'path':
            self._draw_text_as_path(gc, x, y, s, prop, angle, ismath, mtext)
        else:
            self._draw_text_as_text(gc, x, y, s, prop, angle, ismath, mtext)

        if gc.get_url() is not None:
            self.writer.end('a')

        if clip_attrs:
            self.writer.end('g')

    def flipy(self):
        # docstring inherited
        return True

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        return self._text2path.get_text_width_height_descent(s, prop, ismath)


class FigureCanvasSVG(FigureCanvasBase):
    filetypes = {'svg': 'Scalable Vector Graphics',
                 'svgz': 'Scalable Vector Graphics'}

    fixed_dpi = 72

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict[str, Any], optional
            Metadata in the SVG file defined as key-value pairs of strings,
            datetimes, or lists of strings, e.g., ``{'Creator': 'My software',
            'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}``.

            The standard keys and their value types are:

            * *str*: ``'Coverage'``, ``'Description'``, ``'Format'``,
              ``'Identifier'``, ``'Language'``, ``'Relation'``, ``'Source'``,
              ``'Title'``, and ``'Type'``.
            * *str* or *list of str*: ``'Contributor'``, ``'Creator'``,
              ``'Keywords'``, ``'Publisher'``, and ``'Rights'``.
            * *str*, *date*, *datetime*, or *tuple* of same: ``'Date'``. If a
              non-*str*, then it will be formatted as ISO 8601.

            Values have been predefined for ``'Creator'``, ``'Date'``,
            ``'Format'``, and ``'Type'``. They can be removed by setting them
            to `None`.

            Information is encoded as `Dublin Core Metadata`__.

            .. _DC: https://www.dublincore.org/specifications/dublin-core/

            __ DC_
        """
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            if not cbook.file_requires_unicode(fh):
                fh = codecs.getwriter('utf-8')(fh)
            dpi = self.figure.dpi
            self.figure.dpi = 72
            width, height = self.figure.get_size_inches()
            w, h = width * 72, height * 72
            renderer = MixedModeRenderer(
                self.figure, width, height, dpi,
                RendererSVG(w, h, fh, image_dpi=dpi, metadata=metadata),
                bbox_inches_restore=bbox_inches_restore)
            self.figure.draw(renderer)
            renderer.finalize()

    def print_svgz(self, filename, **kwargs):
        with (cbook.open_file_cm(filename, "wb") as fh,
              gzip.GzipFile(mode='w', fileobj=fh) as gzipwriter):
            return self.print_svg(gzipwriter, **kwargs)

    def get_default_filetype(self):
        return 'svg'

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()


FigureManagerSVG = FigureManagerBase


svgProlog = """\
<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
"""


@_Backend.export
class _BackendSVG(_Backend):
    backend_version = mpl.__version__
    FigureCanvas = FigureCanvasSVG
<|endoftext|>import functools
import logging
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
    ToolContainerBase, MouseButton,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent)

try:
    from gi import require_version as gi_require_version
except ImportError as err:
    raise ImportError("The GTK3 backends require PyGObject") from err

try:
    # :raises ValueError: If module/version is already loaded, already
    # required, or unavailable.
    gi_require_version("Gtk", "3.0")
except ValueError as e:
    # in this case we want to re-raise as ImportError so the
    # auto-backend selection logic correctly skips.
    raise ImportError(e) from e

from gi.repository import Gio, GLib, GObject, Gtk, Gdk
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
    _BackendGTK, _FigureCanvasGTK, _FigureManagerGTK, _NavigationToolbar2GTK,
    TimerGTK as TimerGTK3,
)


_log = logging.getLogger(__name__)


@functools.cache
def _mpl_to_gtk_cursor(mpl_cursor):
    return Gdk.Cursor.new_from_name(
        Gdk.Display.get_default(),
        _backend_gtk.mpl_to_gtk_cursor_name(mpl_cursor))


class FigureCanvasGTK3(_FigureCanvasGTK, Gtk.DrawingArea):
    required_interactive_framework = "gtk3"
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK3)
    # Setting this as a static constant prevents
    # this resulting expression from leaking
    event_mask = (Gdk.EventMask.BUTTON_PRESS_MASK
                  | Gdk.EventMask.BUTTON_RELEASE_MASK
                  | Gdk.EventMask.EXPOSURE_MASK
                  | Gdk.EventMask.KEY_PRESS_MASK
                  | Gdk.EventMask.KEY_RELEASE_MASK
                  | Gdk.EventMask.ENTER_NOTIFY_MASK
                  | Gdk.EventMask.LEAVE_NOTIFY_MASK
                  | Gdk.EventMask.POINTER_MOTION_MASK
                  | Gdk.EventMask.SCROLL_MASK)

    def __init__(self, figure=None):
        super().__init__(figure=figure)

        self._idle_draw_id = 0
        self._rubberband_rect = None

        self.connect('scroll_event',         self.scroll_event)
        self.connect('button_press_event',   self.button_press_event)
        self.connect('button_release_event', self.button_release_event)
        self.connect('configure_event',      self.configure_event)
        self.connect('screen-changed',       self._update_device_pixel_ratio)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)
        self.connect('draw',                 self.on_draw_event)
        self.connect('draw',                 self._post_draw)
        self.connect('key_press_event',      self.key_press_event)
        self.connect('key_release_event',    self.key_release_event)
        self.connect('motion_notify_event',  self.motion_notify_event)
        self.connect('enter_notify_event',   self.enter_notify_event)
        self.connect('leave_notify_event',   self.leave_notify_event)
        self.connect('size_allocate',        self.size_allocate)

        self.set_events(self.__class__.event_mask)

        self.set_can_focus(True)

        css = Gtk.CssProvider()
        css.load_from_data(b".matplotlib-canvas { background-color: white; }")
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        style_ctx.add_class("matplotlib-canvas")

    def destroy(self):
        CloseEvent("close_event", self)._process()

    def set_cursor(self, cursor):
        # docstring inherited
        window = self.get_property("window")
        if window is not None:
            window.set_cursor(_mpl_to_gtk_cursor(cursor))
            context = GLib.MainContext.default()
            context.iteration(True)

    def _mpl_coords(self, event=None):
        """
        Convert the position of a GTK event, or of the current cursor position
        if *event* is None, to Matplotlib coordinates.

        GTK use logical pixels, but the figure is scaled to physical pixels for
        rendering.  Transform to physical pixels so that all of the down-stream
        transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        if event is None:
            window = self.get_window()
            t, x, y, state = window.get_device_position(
                window.get_display().get_device_manager().get_client_pointer())
        else:
            x, y = event.x, event.y
        x = x * self.device_pixel_ratio
        # flip y so y=0 is bottom of canvas
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        return x, y

    def scroll_event(self, widget, event):
        step = 1 if event.direction == Gdk.ScrollDirection.UP else -1
        MouseEvent("scroll_event", self,
                   *self._mpl_coords(event), step=step,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def button_press_event(self, widget, event):
        MouseEvent("button_press_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def button_release_event(self, widget, event):
        MouseEvent("button_release_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def key_press_event(self, widget, event):
        KeyEvent("key_press_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        return True  # stop event propagation

    def key_release_event(self, widget, event):
        KeyEvent("key_release_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        return True  # stop event propagation

    def motion_notify_event(self, widget, event):
        MouseEvent("motion_notify_event", self, *self._mpl_coords(event),
                   buttons=self._mpl_buttons(event.state),
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def enter_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        LocationEvent("figure_enter_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    def leave_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        LocationEvent("figure_leave_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    def size_allocate(self, widget, allocation):
        dpival = self.figure.dpi
        winch = allocation.width * self.device_pixel_ratio / dpival
        hinch = allocation.height * self.device_pixel_ratio / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        ResizeEvent("resize_event", self)._process()
        self.draw_idle()

    @staticmethod
    def _mpl_buttons(event_state):
        modifiers = [
            (MouseButton.LEFT, Gdk.ModifierType.BUTTON1_MASK),
            (MouseButton.MIDDLE, Gdk.ModifierType.BUTTON2_MASK),
            (MouseButton.RIGHT, Gdk.ModifierType.BUTTON3_MASK),
            (MouseButton.BACK, Gdk.ModifierType.BUTTON4_MASK),
            (MouseButton.FORWARD, Gdk.ModifierType.BUTTON5_MASK),
        ]
        # State *before* press/release.
        return [name for name, mask in modifiers if event_state & mask]

    @staticmethod
    def _mpl_modifiers(event_state, *, exclude=None):
        modifiers = [
            ("ctrl", Gdk.ModifierType.CONTROL_MASK, "control"),
            ("alt", Gdk.ModifierType.MOD1_MASK, "alt"),
            ("shift", Gdk.ModifierType.SHIFT_MASK, "shift"),
            ("super", Gdk.ModifierType.MOD4_MASK, "super"),
        ]
        return [name for name, mask, key in modifiers
                if exclude != key and event_state & mask]

    def _get_key(self, event):
        unikey = chr(Gdk.keyval_to_unicode(event.keyval))
        key = cbook._unikey_or_keysym_to_mplkey(
            unikey, Gdk.keyval_name(event.keyval))
        mods = self._mpl_modifiers(event.state, exclude=key)
        if "shift" in mods and unikey.isprintable():
            mods.remove("shift")
        return "+".join([*mods, key])

    def _update_device_pixel_ratio(self, *args, **kwargs):
        # We need to be careful in cases with mixed resolution displays if
        # device_pixel_ratio changes.
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            # The easiest way to resize the canvas is to emit a resize event
            # since we implement all the logic for resizing the canvas for that
            # event.
            self.queue_resize()
            self.queue_draw()

    def configure_event(self, widget, event):
        if widget.get_property("window") is None:
            return
        w = event.width * self.device_pixel_ratio
        h = event.height * self.device_pixel_ratio
        if w < 3 or h < 3:
            return  # empty fig
        # resize the figure (in inches)
        dpi = self.figure.dpi
        self.figure.set_size_inches(w / dpi, h / dpi, forward=False)
        return False  # finish event propagation?

    def _draw_rubberband(self, rect):
        self._rubberband_rect = rect
        # TODO: Only update the rubberband area.
        self.queue_draw()

    def _post_draw(self, widget, ctx):
        if self._rubberband_rect is None:
            return

        x0, y0, w, h = (dim / self.device_pixel_ratio
                        for dim in self._rubberband_rect)
        x1 = x0 + w
        y1 = y0 + h

        # Draw the lines from x0, y0 towards x1, y1 so that the
        # dashes don't "jump" when moving the zoom box.
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)

        ctx.set_antialias(1)
        ctx.set_line_width(1)
        ctx.set_dash((3, 3), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()

        ctx.set_dash((3, 3), 3)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

    def on_draw_event(self, widget, ctx):
        # to be overwritten by GTK3Agg or GTK3Cairo
        pass

    def draw(self):
        # docstring inherited
        if self.is_drawable():
            self.queue_draw()

    def draw_idle(self):
        # docstring inherited
        if self._idle_draw_id != 0:
            return
        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        self._idle_draw_id = GLib.idle_add(idle_draw)

    def flush_events(self):
        # docstring inherited
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(True)


class NavigationToolbar2GTK3(_NavigationToolbar2GTK, Gtk.Toolbar):
    def __init__(self, canvas):
        GObject.GObject.__init__(self)

        self.set_style(Gtk.ToolbarStyle.ICONS)

        self._gtk_ids = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert(Gtk.SeparatorToolItem(), -1)
                continue
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(
                    str(cbook._get_data_path('images',
                                             f'{image_file}-symbolic.svg'))),
                Gtk.IconSize.LARGE_TOOLBAR)
            self._gtk_ids[text] = button = (
                Gtk.ToggleToolButton() if callback in ['zoom', 'pan'] else
                Gtk.ToolButton())
            button.set_label(text)
            button.set_icon_widget(image)
            # Save the handler id, so that we can block it as needed.
            button._signal_handler = button.connect(
                'clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.insert(button, -1)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        label = Gtk.Label()
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        toolitem.set_expand(True)  # Push real message to the right.
        toolitem.add(label)

        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        toolitem.add(self.message)

        self.show_all()

        _NavigationToolbar2GTK.__init__(self, canvas)

    def save_figure(self, *args):
        dialog = Gtk.FileChooserDialog(
            title="Save the figure",
            transient_for=self.canvas.get_toplevel(),
            action=Gtk.FileChooserAction.SAVE,
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                     Gtk.STOCK_SAVE,   Gtk.ResponseType.OK),
        )
        for name, fmts \
                in self.canvas.get_supported_filetypes_grouped().items():
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            if self.canvas.get_default_filetype() in fmts:
                dialog.set_filter(ff)

        @functools.partial(dialog.connect, "notify::filter")
        def on_notify_filter(*args):
            name = dialog.get_filter().get_name()
            fmt = self.canvas.get_supported_filetypes_grouped()[name][0]
            dialog.set_current_name(
                str(Path(dialog.get_current_name()).with_suffix(f'.{fmt}')))

        dialog.set_current_folder(mpl.rcParams["savefig.directory"])
        dialog.set_current_name(self.canvas.get_default_filename())
        dialog.set_do_overwrite_confirmation(True)

        response = dialog.run()
        fname = dialog.get_filename()
        ff = dialog.get_filter()  # Doesn't autoadjust to filename :/
        fmt = self.canvas.get_supported_filetypes_grouped()[ff.get_name()][0]
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            return None
        # Save dir for next time, unless empty str (which means use cwd).
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
        try:
            self.canvas.figure.savefig(fname, format=fmt)
            return fname
        except Exception as e:
            dialog = Gtk.MessageDialog(
                transient_for=self.canvas.get_toplevel(), text=str(e),
                message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK)
            dialog.run()
            dialog.destroy()


class ToolbarGTK3(ToolContainerBase, Gtk.Box):
    _icon_extension = '-symbolic.svg'

    def __init__(self, toolmanager):
        ToolContainerBase.__init__(self, toolmanager)
        Gtk.Box.__init__(self)
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)
        self.pack_end(self._message, False, False, 0)
        self.show_all()
        self._groups = {}
        self._toolitems = {}

    def add_toolitem(self, name, group, position, image_file, description,
                     toggle):
        if toggle:
            button = Gtk.ToggleToolButton()
        else:
            button = Gtk.ToolButton()
        button.set_label(name)

        if image_file is not None:
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(image_file),
                Gtk.IconSize.LARGE_TOOLBAR)
            button.set_icon_widget(image)

        if position is None:
            position = -1

        self._add_button(button, group, position)
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)
        button.show_all()
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    def _add_button(self, button, group, position):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            toolbar = Gtk.Toolbar()
            toolbar.set_style(Gtk.ToolbarStyle.ICONS)
            self.pack_start(toolbar, False, False, 0)
            toolbar.show_all()
            self._groups[group] = toolbar
        self._groups[group].insert(button, position)

    def _call_tool(self, btn, name):
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for toolitem, signal in self._toolitems[name]:
            toolitem.handler_block(signal)
            toolitem.set_active(toggled)
            toolitem.handler_unblock(signal)

    def remove_toolitem(self, name):
        for toolitem, _signal in self._toolitems.pop(name, []):
            for group in self._groups:
                if toolitem in self._groups[group]:
                    self._groups[group].remove(toolitem)

    def _add_separator(self):
        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.VERTICAL)
        self.pack_start(sep, False, True, 0)
        sep.show_all()

    def set_message(self, s):
        self._message.set_label(s)


@backend_tools._register_tool_class(FigureCanvasGTK3)
class SaveFigureGTK3(backend_tools.SaveFigureBase):
    def trigger(self, *args, **kwargs):
        NavigationToolbar2GTK3.save_figure(
            self._make_classic_style_pseudo_toolbar())


@backend_tools._register_tool_class(FigureCanvasGTK3)
class HelpGTK3(backend_tools.ToolHelpBase):
    def _normalize_shortcut(self, key):
        """
        Convert Matplotlib key presses to GTK+ accelerator identifiers.

        Related to `FigureCanvasGTK3._get_key`.
        """
        special = {
            'backspace': 'BackSpace',
            'pagedown': 'Page_Down',
            'pageup': 'Page_Up',
            'scroll_lock': 'Scroll_Lock',
        }

        parts = key.split('+')
        mods = ['<' + mod + '>' for mod in parts[:-1]]
        key = parts[-1]

        if key in special:
            key = special[key]
        elif len(key) > 1:
            key = key.capitalize()
        elif key.isupper():
            mods += ['<shift>']

        return ''.join(mods) + key

    def _is_valid_shortcut(self, key):
        """
        Check for a valid shortcut to be displayed.

        - GTK will never send 'cmd+' (see `FigureCanvasGTK3._get_key`).
        - The shortcut window only shows keyboard shortcuts, not mouse buttons.
        """
        return 'cmd+' not in key and not key.startswith('MouseButton.')

    def _show_shortcuts_window(self):
        section = Gtk.ShortcutsSection()

        for name, tool in sorted(self.toolmanager.tools.items()):
            if not tool.description:
                continue

            # Putting everything in a separate group allows GTK to
            # automatically split them into separate columns/pages, which is
            # useful because we have lots of shortcuts, some with many keys
            # that are very wide.
            group = Gtk.ShortcutsGroup()
            section.add(group)
            # A hack to remove the title since we have no group naming.
            group.forall(lambda widget, data: widget.set_visible(False), None)

            shortcut = Gtk.ShortcutsShortcut(
                accelerator=' '.join(
                    self._normalize_shortcut(key)
                    for key in self.toolmanager.get_tool_keymap(name)
                    if self._is_valid_shortcut(key)),
                title=tool.name,
                subtitle=tool.description)
            group.add(shortcut)

        window = Gtk.ShortcutsWindow(
            title='Help',
            modal=True,
            transient_for=self._figure.canvas.get_toplevel())
        section.show()  # Must be done explicitly before add!
        window.add(section)

        window.show_all()

    def _show_shortcuts_dialog(self):
        dialog = Gtk.MessageDialog(
            self._figure.canvas.get_toplevel(),
            0, Gtk.MessageType.INFO, Gtk.ButtonsType.OK, self._get_help_text(),
            title="Help")
        dialog.run()
        dialog.destroy()

    def trigger(self, *args):
        if Gtk.check_version(3, 20, 0) is None:
            self._show_shortcuts_window()
        else:
            self._show_shortcuts_dialog()


@backend_tools._register_tool_class(FigureCanvasGTK3)
class ToolCopyToClipboardGTK3(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs):
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        window = self.canvas.get_window()
        x, y, width, height = window.get_geometry()
        pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
        clipboard.set_image(pb)


Toolbar = ToolbarGTK3
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.ConfigureSubplotsGTK)
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.RubberbandGTK)


class FigureManagerGTK3(_FigureManagerGTK):
    _toolbar2_class = NavigationToolbar2GTK3
    _toolmanager_toolbar_class = ToolbarGTK3


@_BackendGTK.export
class _BackendGTK3(_BackendGTK):
    FigureCanvas = FigureCanvasGTK3
    FigureManager = FigureManagerGTK3
<|endoftext|>import numpy as np

from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing


class MixedModeRenderer:
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """
    def __init__(self, figure, width, height, dpi, vector_renderer,
                 raster_renderer_class=None,
                 bbox_inches_restore=None):
        """
        Parameters
        ----------
        figure : `~matplotlib.figure.Figure`
            The figure instance.
        width : float
            The width of the canvas in logical units
        height : float
            The height of the canvas in logical units
        dpi : float
            The dpi of the canvas
        vector_renderer : `~matplotlib.backend_bases.RendererBase`
            An instance of a subclass of
            `~matplotlib.backend_bases.RendererBase` that will be used for the
            vector drawing.
        raster_renderer_class : `~matplotlib.backend_bases.RendererBase`
            The renderer class to use for the raster drawing.  If not provided,
            this will use the Agg backend (which is currently the only viable
            option anyway.)

        """
        if raster_renderer_class is None:
            raster_renderer_class = RendererAgg

        self._raster_renderer_class = raster_renderer_class
        self._width = width
        self._height = height
        self.dpi = dpi

        self._vector_renderer = vector_renderer

        self._raster_renderer = None

        # A reference to the figure is needed as we need to change
        # the figure dpi before and after the rasterization. Although
        # this looks ugly, I couldn't find a better solution. -JJL
        self.figure = figure
        self._figdpi = figure.dpi

        self._bbox_inches_restore = bbox_inches_restore

        self._renderer = vector_renderer

    def __getattr__(self, attr):
        # Proxy everything that hasn't been overridden to the base
        # renderer. Things that *are* overridden can call methods
        # on self._renderer directly, but must not cache/store
        # methods (because things like RendererAgg change their
        # methods on the fly in order to optimise proxying down
        # to the underlying C implementation).
        return getattr(self._renderer, attr)

    def start_rasterizing(self):
        """
        Enter "raster" mode.  All subsequent drawing commands (until
        `stop_rasterizing` is called) will be drawn with the raster backend.
        """
        # change the dpi of the figure temporarily.
        self.figure.dpi = self.dpi
        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore)
            self._bbox_inches_restore = r

        self._raster_renderer = self._raster_renderer_class(
            self._width*self.dpi, self._height*self.dpi, self.dpi)
        self._renderer = self._raster_renderer

    def stop_rasterizing(self):
        """
        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        """

        self._renderer = self._vector_renderer

        height = self._height * self.dpi
        img = np.asarray(self._raster_renderer.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(img[..., 3])
        cropped_img = img[slice_y, slice_x]
        if cropped_img.size:
            gc = self._renderer.new_gc()
            # TODO: If the mixedmode resolution differs from the figure's
            #       dpi, the image must be scaled (dpi->_figdpi). Not all
            #       backends support this.
            self._renderer.draw_image(
                gc,
                slice_x.start * self._figdpi / self.dpi,
                (height - slice_y.stop) * self._figdpi / self.dpi,
                cropped_img[::-1])
        self._raster_renderer = None

        # restore the figure dpi.
        self.figure.dpi = self._figdpi

        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore,
                                               self._figdpi)
            self._bbox_inches_restore = r
<|endoftext|>"""
A PostScript backend, which can produce both PostScript .ps and .eps.
"""

import bisect
import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import math
import os
import pathlib
import shutil
import struct
from tempfile import TemporaryDirectory
import textwrap
import time

import fontTools
import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LoadFlags
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps


_log = logging.getLogger(__name__)
debugPS = False


papersize = {'letter': (8.5, 11),
             'legal': (8.5, 14),
             'ledger': (11, 17),
             'a0': (33.11, 46.81),
             'a1': (23.39, 33.11),
             'a2': (16.54, 23.39),
             'a3': (11.69, 16.54),
             'a4': (8.27, 11.69),
             'a5': (5.83, 8.27),
             'a6': (4.13, 5.83),
             'a7': (2.91, 4.13),
             'a8': (2.05, 2.91),
             'a9': (1.46, 2.05),
             'a10': (1.02, 1.46),
             'b0': (40.55, 57.32),
             'b1': (28.66, 40.55),
             'b2': (20.27, 28.66),
             'b3': (14.33, 20.27),
             'b4': (10.11, 14.33),
             'b5': (7.16, 10.11),
             'b6': (5.04, 7.16),
             'b7': (3.58, 5.04),
             'b8': (2.51, 3.58),
             'b9': (1.76, 2.51),
             'b10': (1.26, 1.76)}


def _nums_to_str(*args, sep=" "):
    return sep.join(f"{arg:1.3f}".rstrip("0").rstrip(".") for arg in args)


def _move_path_to_path_or_stream(src, dst):
    """
    Move the contents of file at *src* to path-or-filelike *dst*.

    If *dst* is a path, the metadata of *src* are *not* copied.
    """
    if is_writable_file_like(dst):
        fh = (open(src, encoding='latin-1')
              if file_requires_unicode(dst)
              else open(src, 'rb'))
        with fh:
            shutil.copyfileobj(fh, dst)
    else:
        shutil.move(src, dst, copy_function=shutil.copyfile)


def _font_to_ps_type3(font_path, chars):
    """
    Subset *chars* from the font at *font_path* into a Type 3 font.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.

    Returns
    -------
    str
        The string representation of a Type 3 font, which can be included
        verbatim into a PostScript file.
    """
    font = get_font(font_path, hinting_factor=1)
    glyph_ids = [font.get_char_index(c) for c in chars]

    preamble = """\
%!PS-Adobe-3.0 Resource-Font
%%Creator: Converted from TrueType to Type 3 by Matplotlib.
10 dict begin
/FontName /{font_name} def
/PaintType 0 def
/FontMatrix [{inv_units_per_em} 0 0 {inv_units_per_em} 0 0] def
/FontBBox [{bbox}] def
/FontType 3 def
/Encoding [{encoding}] def
/CharStrings {num_glyphs} dict dup begin
/.notdef 0 def
""".format(font_name=font.postscript_name,
           inv_units_per_em=1 / font.units_per_EM,
           bbox=" ".join(map(str, font.bbox)),
           encoding=" ".join(f"/{font.get_glyph_name(glyph_id)}"
                             for glyph_id in glyph_ids),
           num_glyphs=len(glyph_ids) + 1)
    postamble = """
end readonly def

/BuildGlyph {
 exch begin
 CharStrings exch
 2 copy known not {pop /.notdef} if
 true 3 1 roll get exec
 end
} _d

/BuildChar {
 1 index /Encoding get exch get
 1 index /BuildGlyph get exec
} _d

FontName currentdict end definefont pop
"""

    entries = []
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LoadFlags.NO_SCALE)
        v, c = font.get_path()
        entries.append(
            "/%(name)s{%(bbox)s sc\n" % {
                "name": font.get_glyph_name(glyph_id),
                "bbox": " ".join(map(str, [g.horiAdvance, 0, *g.bbox])),
            }
            + _path.convert_to_string(
                # Convert back to TrueType's internal units (1/64's).
                # (Other dimensions are already in these units.)
                Path(v * 64, c), None, None, False, None, 0,
                # No code for quad Beziers triggers auto-conversion to cubics.
                # Drop intermediate closepolys (relying on the outline
                # decomposer always explicitly moving to the closing point
                # first).
                [b"m", b"l", b"", b"c", b""], True).decode("ascii")
            + "ce} _d"
        )

    return preamble + "\n".join(entries) + postamble


def _font_to_ps_type42(font_path, chars, fh):
    """
    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.
    fh : file-like
        Where to write the font.
    """
    subset_str = ''.join(chr(c) for c in chars)
    _log.debug("SUBSET %s characters: %s", font_path, subset_str)
    try:
        kw = {}
        # fix this once we support loading more fonts from a collection
        # https://github.com/matplotlib/matplotlib/issues/3135#issuecomment-571085541
        if font_path.endswith('.ttc'):
            kw['fontNumber'] = 0
        with (fontTools.ttLib.TTFont(font_path, **kw) as font,
              _backend_pdf_ps.get_glyphs_subset(font_path, subset_str) as subset):
            fontdata = _backend_pdf_ps.font_as_file(subset).getvalue()
            _log.debug(
                "SUBSET %s %d -> %d", font_path, os.stat(font_path).st_size,
                len(fontdata)
            )
            fh.write(_serialize_type42(font, subset, fontdata))
    except RuntimeError:
        _log.warning(
            "The PostScript backend does not currently support the selected font (%s).",
            font_path)
        raise


def _serialize_type42(font, subset, fontdata):
    """
    Output a PostScript Type-42 format representation of font

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The original font object
    subset : fontTools.ttLib.ttFont.TTFont
        The subset font object
    fontdata : bytes
        The raw font data in TTF format

    Returns
    -------
    str
        The Type-42 formatted font
    """
    version, breakpoints = _version_and_breakpoints(font.get('loca'), fontdata)
    post = font['post']
    name = font['name']
    chars = _generate_charstrings(subset)
    sfnts = _generate_sfnts(fontdata, subset, breakpoints)
    return textwrap.dedent(f"""
        %%!PS-TrueTypeFont-{version[0]}.{version[1]}-{font['head'].fontRevision:.7f}
        10 dict begin
        /FontType 42 def
        /FontMatrix [1 0 0 1 0 0] def
        /FontName /{name.getDebugName(6)} def
        /FontInfo 7 dict dup begin
        /FullName ({name.getDebugName(4)}) def
        /FamilyName ({name.getDebugName(1)}) def
        /Version ({name.getDebugName(5)}) def
        /ItalicAngle {post.italicAngle} def
        /isFixedPitch {'true' if post.isFixedPitch else 'false'} def
        /UnderlinePosition {post.underlinePosition} def
        /UnderlineThickness {post.underlineThickness} def
        end readonly def
        /Encoding StandardEncoding def
        /FontBBox [{_nums_to_str(*_bounds(font))}] def
        /PaintType 0 def
        /CIDMap 0 def
        {chars}
        {sfnts}
        FontName currentdict end definefont pop
        """)


def _version_and_breakpoints(loca, fontdata):
    """
    Read the version number of the font and determine sfnts breakpoints.

    When a TrueType font file is written as a Type 42 font, it has to be
    broken into substrings of at most 65535 bytes. These substrings must
    begin at font table boundaries or glyph boundaries in the glyf table.
    This function determines all possible breakpoints and it is the caller's
    responsibility to do the splitting.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    loca : fontTools.ttLib._l_o_c_a.table__l_o_c_a or None
        The loca table of the font if available
    fontdata : bytes
        The raw data of the font

    Returns
    -------
    version : tuple[int, int]
        A 2-tuple of the major version number and minor version number.
    breakpoints : list[int]
        The breakpoints is a sorted list of offsets into fontdata; if loca is not
        available, just the table boundaries.
    """
    v1, v2, numTables = struct.unpack('>3h', fontdata[:6])
    version = (v1, v2)

    tables = {}
    for i in range(numTables):
        tag, _, offset, _ = struct.unpack('>4sIII', fontdata[12 + i*16:12 + (i+1)*16])
        tables[tag.decode('ascii')] = offset
    if loca is not None:
        glyf_breakpoints = {tables['glyf'] + offset for offset in loca.locations[:-1]}
    else:
        glyf_breakpoints = set()
    breakpoints = sorted({*tables.values(), *glyf_breakpoints, len(fontdata)})

    return version, breakpoints


def _bounds(font):
    """
    Compute the font bounding box, as if all glyphs were written
    at the same start position.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The font

    Returns
    -------
    tuple
        (xMin, yMin, xMax, yMax) of the combined bounding box
        of all the glyphs in the font
    """
    gs = font.getGlyphSet(False)
    pen = fontTools.pens.boundsPen.BoundsPen(gs)
    for name in gs.keys():
        gs[name].draw(pen)
    return pen.bounds or (0, 0, 0, 0)


def _generate_charstrings(font):
    """
    Transform font glyphs into CharStrings

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        The font

    Returns
    -------
    str
        A definition of the CharStrings dictionary in PostScript
    """
    go = font.getGlyphOrder()
    s = f'/CharStrings {len(go)} dict dup begin\n'
    for i, name in enumerate(go):
        s += f'/{name} {i} def\n'
    s += 'end readonly def'
    return s


def _generate_sfnts(fontdata, font, breakpoints):
    """
    Transform font data into PostScript sfnts format.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    fontdata : bytes
        The raw data of the font
    font : fontTools.ttLib.ttFont.TTFont
        The fontTools font object
    breakpoints : list
        Sorted offsets of possible breakpoints

    Returns
    -------
    str
        The sfnts array for the font definition, consisting
        of hex-encoded strings in PostScript format
    """
    s = '/sfnts['
    pos = 0
    while pos < len(fontdata):
        i = bisect.bisect_left(breakpoints, pos + 65534)
        newpos = breakpoints[i-1]
        if newpos <= pos:
            # have to accept a larger string
            newpos = breakpoints[-1]
        s += f'<{fontdata[pos:newpos].hex()}00>'  # Always NUL terminate.
        pos = newpos
    s += ']def'
    return '\n'.join(s[i:i+100] for i in range(0, len(s), 100))


def _log_if_debug_on(meth):
    """
    Wrap `RendererPS` method *meth* to emit a PS comment with the method name,
    if the global flag `debugPS` is set.
    """
    @functools.wraps(meth)
    def wrapper(self, *args, **kwargs):
        if debugPS:
            self._pswriter.write(f"% {meth.__name__}\n")
        return meth(self, *args, **kwargs)

    return wrapper


class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    _afm_font_dir = cbook._get_data_path("fonts/afm")
    _use_afm_rc_name = "ps.useafm"

    def __init__(self, width, height, pswriter, imagedpi=72):
        # Although postscript itself is dpi independent, we need to inform the
        # image code about a requested dpi to generate high resolution images
        # and them scale them before embedding them.
        super().__init__(width, height)
        self._pswriter = pswriter
        if mpl.rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi

        # current renderer state (None=uninitialised)
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0

        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self._logwarn_once = functools.cache(_log.warning)

    def _is_transparent(self, rgb_or_rgba):
        if rgb_or_rgba is None:
            return True  # Consistent with rgbFace semantics.
        elif len(rgb_or_rgba) == 4:
            if rgb_or_rgba[3] == 0:
                return True
            if rgb_or_rgba[3] != 1:
                self._logwarn_once(
                    "The PostScript backend does not support transparency; "
                    "partially transparent artists will be rendered opaque.")
            return False
        else:  # len() == 3.
            return False

    def set_color(self, r, g, b, store=True):
        if (r, g, b) != self.color:
            self._pswriter.write(f"{_nums_to_str(r)} setgray\n"
                                 if r == g == b else
                                 f"{_nums_to_str(r, g, b)} setrgbcolor\n")
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=True):
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write(f"{_nums_to_str(linewidth)} setlinewidth\n")
            if store:
                self.linewidth = linewidth

    @staticmethod
    def _linejoin_cmd(linejoin):
        # Support for directly passing integer values is for backcompat.
        linejoin = {'miter': 0, 'round': 1, 'bevel': 2, 0: 0, 1: 1, 2: 2}[
            linejoin]
        return f"{linejoin:d} setlinejoin\n"

    def set_linejoin(self, linejoin, store=True):
        if linejoin != self.linejoin:
            self._pswriter.write(self._linejoin_cmd(linejoin))
            if store:
                self.linejoin = linejoin

    @staticmethod
    def _linecap_cmd(linecap):
        # Support for directly passing integer values is for backcompat.
        linecap = {'butt': 0, 'round': 1, 'projecting': 2, 0: 0, 1: 1, 2: 2}[
            linecap]
        return f"{linecap:d} setlinecap\n"

    def set_linecap(self, linecap, store=True):
        if linecap != self.linecap:
            self._pswriter.write(self._linecap_cmd(linecap))
            if store:
                self.linecap = linecap

    def set_linedash(self, offset, seq, store=True):
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return

        self._pswriter.write(f"[{_nums_to_str(*seq)}] {_nums_to_str(offset)} setdash\n"
                             if seq is not None and len(seq) else
                             "[] 0 setdash\n")
        if store:
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=True):
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            self._pswriter.write(f"/{fontname} {fontsize:1.3f} selectfont\n")
            if store:
                self.fontname = fontname
                self.fontsize = fontsize

    def create_hatch(self, hatch, linewidth):
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        pageheight = self.height * 72
        self._pswriter.write(f"""\
  << /PatternType 1
     /PaintType 2
     /TilingType 2
     /BBox[0 0 {sidelen:d} {sidelen:d}]
     /XStep {sidelen:d}
     /YStep {sidelen:d}

     /PaintProc {{
        pop
        {linewidth:g} setlinewidth
{self._convert_path(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False)}
        gsave
        fill
        grestore
        stroke
     }} bind
   >>
   matrix
   0 {pageheight:g} translate
   makepattern
   /{name} exch def
""")
        self._hatches[hatch] = name
        return name

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return self.image_magnification

    def _convert_path(self, path, transform, clip=False, simplify=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, None,
            6, [b"m", b"l", b"", b"c", b"cl"], True).decode("ascii")

    def _get_clip_cmd(self, gc):
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append(f"{_nums_to_str(*rect.p0, *rect.size)} rectclip\n")
        path, trf = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = "c%d" % len(self._clip_paths)
                self._pswriter.write(f"""\
/{custom_clip_cmd} {{
{self._convert_path(path, trf, simplify=False)}
clip
newpath
}} bind def
""")
                self._clip_paths[key] = custom_clip_cmd
            clip.append(f"{custom_clip_cmd}\n")
        return "".join(clip)

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        imagecmd = "false 3 colorimage"
        data = im[::-1, :, :3]  # Vertically flipped rgb values.
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        if transform is None:
            matrix = "1 0 0 1 0 0"
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = " ".join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0

        self._pswriter.write(f"""\
gsave
{self._get_clip_cmd(gc)}
{x:g} {y:g} translate
[{matrix}] concat
{xscale:g} {yscale:g} scale
/DataString {w:d} string def
{w:d} {h:d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]
{{
currentfile DataString readhexstring pop
}} bind {imagecmd}
{hexdata}
grestore
""")

    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)

    @_log_if_debug_on
    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # docstring inherited

        ps_color = (
            None
            if self._is_transparent(rgbFace)
            else f'{_nums_to_str(rgbFace[0])} setgray'
            if rgbFace[0] == rgbFace[1] == rgbFace[2]
            else f'{_nums_to_str(*rgbFace[:3])} setrgbcolor')

        # construct the generic marker command:

        # don't want the translate to be global
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']

        lw = gc.get_linewidth()
        alpha = (gc.get_alpha()
                 if gc.get_forced_alpha() or len(gc.get_rgb()) == 3
                 else gc.get_rgb()[3])
        stroke = lw > 0 and alpha > 0
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)
            ps_cmd.append(self._linejoin_cmd(gc.get_joinstyle()))
            ps_cmd.append(self._linecap_cmd(gc.get_capstyle()))

        ps_cmd.append(self._convert_path(marker_path, marker_trans,
                                         simplify=False))

        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])
            if stroke:
                ps_cmd.append('grestore')

        if stroke:
            ps_cmd.append('stroke')
        ps_cmd.extend(['grestore', '} bind def'])

        for vertices, code in path.iter_segments(
                trans,
                clip=(0, 0, self.width*72, self.height*72),
                simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append(f"{x:g} {y:g} o")

        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)

    @_log_if_debug_on
    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # Is the optimization worth it? Rough calculation:
        # cost of emitting a path in-line is
        #     (len_path + 2) * uses_per_path
        # cost of definition+use is
        #     (len_path + 3) + 3 * uses_per_path
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = \
            len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
        if not should_do_optimization:
            return RendererBase.draw_path_collection(
                self, gc, master_transform, paths, all_transforms,
                offsets, offset_trans, facecolors, edgecolors,
                linewidths, linestyles, antialiaseds, urls,
                offset_position)

        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                master_transform, paths, all_transforms)):
            name = 'p%d_%d' % (self._path_collection_id, i)
            path_bytes = self._convert_path(path, transform, simplify=False)
            self._pswriter.write(f"""\
/{name} {{
newpath
translate
{path_bytes}
}} bind def
""")
            path_codes.append(name)

        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, path_codes, offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            ps = f"{xo:g} {yo:g} {path_id}"
            self._draw_ps(ps, gc0, rgbFace)

        self._path_collection_id += 1

    @_log_if_debug_on
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        if self._is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        if not hasattr(self, "psfrag"):
            self._logwarn_once(
                "The PS backend determines usetex status solely based on "
                "rcParams['text.usetex'] and does not support having "
                "usetex=True only for some elements; this element will thus "
                "be rendered as if usetex=False.")
            self.draw_text(gc, x, y, s, prop, angle, False, mtext)
            return

        w, h, bl = self.get_text_width_height_descent(s, prop, ismath="TeX")
        fontsize = prop.get_size_in_points()
        thetext = 'psmarker%d' % self.textcnt
        color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(
                       mpl.rcParams['font.family'][0], r'{\rmfamily %s}')
        s = fontcmd % s
        tex = r'\color[rgb]{%s} %s' % (color, s)

        # Stick to bottom-left alignment, so subtract descent from the text-normal
        # direction since text is normally positioned by its baseline.
        rangle = np.radians(angle + 90)
        pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
        self.psfrag.append(
            r'\psfrag{%s}[bl][bl][1][%f]{\fontsize{%f}{%f}%s}' % (
                thetext, angle, fontsize, fontsize*1.25, tex))

        self._pswriter.write(f"""\
gsave
{pos} moveto
({thetext})
show
grestore
""")
        self.textcnt += 1

    @_log_if_debug_on
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        if self._is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        stream = []  # list of (ps_name, x, char_name)

        if mpl.rcParams['ps.useafm']:
            font = self._get_font_afm(prop)
            ps_name = (font.postscript_name.encode("ascii", "replace")
                        .decode("ascii"))
            scale = 0.001 * prop.get_size_in_points()
            thisx = 0
            last_name = None  # kerns returns 0 for None.
            for c in s:
                name = uni2type1.get(ord(c), f"uni{ord(c):04X}")
                try:
                    width = font.get_width_from_char_name(name)
                except KeyError:
                    name = 'question'
                    width = font.get_width_char('?')
                kern = font.get_kern_dist_from_name(last_name, name)
                last_name = name
                thisx += kern * scale
                stream.append((ps_name, thisx, name))
                thisx += width * scale

        else:
            font = self._get_font_ttf(prop)
            self._character_tracker.track(font, s)
            for item in _text_helpers.layout(s, font):
                ps_name = (item.ft_object.postscript_name
                           .encode("ascii", "replace").decode("ascii"))
                glyph_name = item.ft_object.get_glyph_name(item.glyph_idx)
                stream.append((ps_name, item.x, glyph_name))
        self.set_color(*gc.get_rgb())

        for ps_name, group in itertools. \
                groupby(stream, lambda entry: entry[0]):
            self.set_font(ps_name, prop.get_size_in_points(), False)
            thetext = "\n".join(f"{x:g} 0 m /{name:s} glyphshow"
                                for _, x, name in group)
            self._pswriter.write(f"""\
gsave
{self._get_clip_cmd(gc)}
{x:g} {y:g} translate
{angle:g} rotate
{thetext}
grestore
""")

    @_log_if_debug_on
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw the math text using matplotlib.mathtext."""
        width, height, descent, glyphs, rects = \
            self._text2path.mathtext_parser.parse(s, 72, prop)
        self.set_color(*gc.get_rgb())
        self._pswriter.write(
            f"gsave\n"
            f"{x:g} {y:g} translate\n"
            f"{angle:g} rotate\n")
        lastfont = None
        for font, fontsize, num, ox, oy in glyphs:
            self._character_tracker.track_glyph(font, num)
            if (font.postscript_name, fontsize) != lastfont:
                lastfont = font.postscript_name, fontsize
                self._pswriter.write(
                    f"/{font.postscript_name} {fontsize} selectfont\n")
            glyph_name = (
                font.get_name_char(chr(num)) if isinstance(font, AFM) else
                font.get_glyph_name(font.get_char_index(num)))
            self._pswriter.write(
                f"{ox:g} {oy:g} moveto\n"
                f"/{glyph_name} glyphshow\n")
        for ox, oy, w, h in rects:
            self._pswriter.write(f"{ox} {oy} {w} {h} rectfill\n")
        self._pswriter.write("grestore\n")

    @_log_if_debug_on
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4

        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))

        xmin, ymin = points_min
        xmax, ymax = points_max

        data = np.empty(
            shape[0] * shape[1],
            dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        data['points'] = (flat_points - points_min) * factor
        data['colors'] = flat_colors[:, :3] * 255.0
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        self._pswriter.write(f"""\
gsave
<< /ShadingType 4
   /ColorSpace [/DeviceRGB]
   /BitsPerCoordinate 32
   /BitsPerComponent 8
   /BitsPerFlag 8
   /AntiAlias true
   /Decode [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]
   /DataSource <
{hexdata}
>
>>
shfill
grestore
""")

    def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
        """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """
        write = self._pswriter.write
        mightstroke = (gc.get_linewidth() > 0
                       and not self._is_transparent(gc.get_rgb()))
        if not mightstroke:
            stroke = False
        if self._is_transparent(rgbFace):
            fill = False
        hatch = gc.get_hatch()

        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            self.set_linejoin(gc.get_joinstyle())
            self.set_linecap(gc.get_capstyle())
            self.set_linedash(*gc.get_dashes())
        if mightstroke or hatch:
            self.set_color(*gc.get_rgb()[:3])
        write('gsave\n')

        write(self._get_clip_cmd(gc))

        write(ps.strip())
        write("\n")

        if fill:
            if stroke or hatch:
                write("gsave\n")
            self.set_color(*rgbFace[:3], store=False)
            write("fill\n")
            if stroke or hatch:
                write("grestore\n")

        if hatch:
            hatch_name = self.create_hatch(hatch, gc.get_hatch_linewidth())
            write("gsave\n")
            write(_nums_to_str(*gc.get_hatch_color()[:3]))
            write(f" {hatch_name} setpattern fill grestore\n")

        if stroke:
            write("stroke\n")

        write("grestore\n")


class _Orientation(Enum):
    portrait, landscape = range(2)

    def swap_if_landscape(self, shape):
        return shape[::-1] if self.name == "landscape" else shape


class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72
    filetypes = {'ps': 'Postscript',
                 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def _print_ps(
            self, fmt, outfile, *,
            metadata=None, papertype=None, orientation='portrait',
            bbox_inches_restore=None, **kwargs):

        dpi = self.figure.dpi
        self.figure.dpi = 72  # Override the dpi kwarg

        dsc_comments = {}
        if isinstance(outfile, (str, os.PathLike)):
            filename = pathlib.Path(outfile).name
            dsc_comments["Title"] = \
                filename.encode("ascii", "replace").decode("ascii")
        dsc_comments["Creator"] = (metadata or {}).get(
            "Creator",
            f"Matplotlib v{mpl.__version__}, https://matplotlib.org/")
        # See https://reproducible-builds.org/specs/source-date-epoch/
        source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
        dsc_comments["CreationDate"] = (
            datetime.datetime.fromtimestamp(
                int(source_date_epoch),
                datetime.timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
            if source_date_epoch
            else time.ctime())
        dsc_comments = "\n".join(
            f"%%{k}: {v}" for k, v in dsc_comments.items())

        if papertype is None:
            papertype = mpl.rcParams['ps.papersize']
        papertype = papertype.lower()
        _api.check_in_list(['figure', *papersize], papertype=papertype)

        orientation = _api.check_getitem(
            _Orientation, orientation=orientation.lower())

        printer = (self._print_figure_tex
                   if mpl.rcParams['text.usetex'] else
                   self._print_figure)
        printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments,
                orientation=orientation, papertype=papertype,
                bbox_inches_restore=bbox_inches_restore, **kwargs)

    def _print_figure(
            self, fmt, outfile, *,
            dpi, dsc_comments, orientation, papertype,
            bbox_inches_restore=None):
        """
        Render the figure to a filesystem path or a file-like object.

        Parameters are as for `.print_figure`, except that *dsc_comments* is a
        string containing Document Structuring Convention comments,
        generated from the *metadata* parameter to `.print_figure`.
        """
        is_eps = fmt == 'eps'
        if not (isinstance(outfile, (str, os.PathLike))
                or is_writable_file_like(outfile)):
            raise ValueError("outfile must be a path or a file-like object")

        # find the appropriate papertype
        width, height = self.figure.get_size_inches()
        if is_eps or papertype == 'figure':
            paper_width, paper_height = width, height
        else:
            paper_width, paper_height = orientation.swap_if_landscape(
                papersize[papertype])

        # center the figure on the paper
        xo = 72 * 0.5 * (paper_width - width)
        yo = 72 * 0.5 * (paper_height - height)

        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        rotation = 0
        if orientation is _Orientation.landscape:
            llx, lly, urx, ury = lly, llx, ury, urx
            xo, yo = 72 * paper_height - yo, xo
            rotation = 90
        bbox = (llx, lly, urx, ury)

        self._pswriter = StringIO()

        # mixed mode rendering
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(
            self.figure, width, height, dpi, ps_renderer,
            bbox_inches_restore=bbox_inches_restore)

        self.figure.draw(renderer)

        def print_figure_impl(fh):
            # write the PostScript headers
            if is_eps:
                print("%!PS-Adobe-3.0 EPSF-3.0", file=fh)
            else:
                print("%!PS-Adobe-3.0", file=fh)
                if papertype != 'figure':
                    print(f"%%DocumentPaperSizes: {papertype}", file=fh)
                print("%%Pages: 1", file=fh)
            print(f"%%LanguageLevel: 3\n"
                  f"{dsc_comments}\n"
                  f"%%Orientation: {orientation.name}\n"
                  f"{_get_bbox_header(bbox)}\n"
                  f"%%EndComments\n",
                  end="", file=fh)

            Ndict = len(_psDefs)
            print("%%BeginProlog", file=fh)
            if not mpl.rcParams['ps.useafm']:
                Ndict += len(ps_renderer._character_tracker.used)
            print("/mpldict %d dict def" % Ndict, file=fh)
            print("mpldict begin", file=fh)
            print("\n".join(_psDefs), file=fh)
            if not mpl.rcParams['ps.useafm']:
                for font_path, chars \
                        in ps_renderer._character_tracker.used.items():
                    if not chars:
                        continue
                    fonttype = mpl.rcParams['ps.fonttype']
                    # Can't use more than 255 chars from a single Type 3 font.
                    if len(chars) > 255:
                        fonttype = 42
                    fh.flush()
                    if fonttype == 3:
                        fh.write(_font_to_ps_type3(font_path, chars))
                    else:  # Type 42 only.
                        _font_to_ps_type42(font_path, chars, fh)
            print("end", file=fh)
            print("%%EndProlog", file=fh)

            if not is_eps:
                print("%%Page: 1 1", file=fh)
            print("mpldict begin", file=fh)

            print("%s translate" % _nums_to_str(xo, yo), file=fh)
            if rotation:
                print("%d rotate" % rotation, file=fh)
            print(f"0 0 {_nums_to_str(width*72, height*72)} rectclip", file=fh)

            # write the figure
            print(self._pswriter.getvalue(), file=fh)

            # write the trailer
            print("end", file=fh)
            print("showpage", file=fh)
            if not is_eps:
                print("%%EOF", file=fh)
            fh.flush()

        if mpl.rcParams['ps.usedistiller']:
            # We are going to use an external program to process the output.
            # Write to a temporary file.
            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, "tmp.ps")
                with open(tmpfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)
                if mpl.rcParams['ps.usedistiller'] == 'ghostscript':
                    _try_distill(gs_distill,
                                 tmpfile, is_eps, ptype=papertype, bbox=bbox)
                elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                    _try_distill(xpdf_distill,
                                 tmpfile, is_eps, ptype=papertype, bbox=bbox)
                _move_path_to_path_or_stream(tmpfile, outfile)

        else:  # Write directly to outfile.
            with cbook.open_file_cm(outfile, "w", encoding="latin-1") as file:
                if not file_requires_unicode(file):
                    file = codecs.getwriter("latin-1")(file)
                print_figure_impl(file)

    def _print_figure_tex(
            self, fmt, outfile, *,
            dpi, dsc_comments, orientation, papertype,
            bbox_inches_restore=None):
        """
        If :rc:`text.usetex` is True, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        The rest of the behavior is as for `._print_figure`.
        """
        is_eps = fmt == 'eps'

        width, height = self.figure.get_size_inches()
        xo = 0
        yo = 0

        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        bbox = (llx, lly, urx, ury)

        self._pswriter = StringIO()

        # mixed mode rendering
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure,
                                     width, height, dpi, ps_renderer,
                                     bbox_inches_restore=bbox_inches_restore)

        self.figure.draw(renderer)

        # write to a temp file, we'll move it to outfile when done
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir, "tmp.ps")
            tmppath.write_text(
                f"""\
%!PS-Adobe-3.0 EPSF-3.0
%%LanguageLevel: 3
{dsc_comments}
{_get_bbox_header(bbox)}
%%EndComments
%%BeginProlog
/mpldict {len(_psDefs)} dict def
mpldict begin
{"".join(_psDefs)}
end
%%EndProlog
mpldict begin
{_nums_to_str(xo, yo)} translate
0 0 {_nums_to_str(width*72, height*72)} rectclip
{self._pswriter.getvalue()}
end
showpage
""",
                encoding="latin-1")

            if orientation is _Orientation.landscape:  # now, ready to rotate
                width, height = height, width
                bbox = (lly, llx, ury, urx)

            # set the paper size to the figure size if is_eps. The
            # resulting ps file has the given size with correct bounding
            # box so that there is no need to call 'pstoeps'
            if is_eps or papertype == 'figure':
                paper_width, paper_height = orientation.swap_if_landscape(
                    self.figure.get_size_inches())
            else:
                paper_width, paper_height = papersize[papertype]

            psfrag_rotated = _convert_psfrags(
                tmppath, ps_renderer.psfrag, paper_width, paper_height,
                orientation.name)

            if (mpl.rcParams['ps.usedistiller'] == 'ghostscript'
                    or mpl.rcParams['text.usetex']):
                _try_distill(gs_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)
            elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                _try_distill(xpdf_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)

            _move_path_to_path_or_stream(tmppath, outfile)

    print_ps = functools.partialmethod(_print_ps, "ps")
    print_eps = functools.partialmethod(_print_ps, "eps")

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()


def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    with mpl.rc_context({
            "text.latex.preamble":
            mpl.rcParams["text.latex.preamble"] +
            mpl.texmanager._usepackage_if_not_loaded("color") +
            mpl.texmanager._usepackage_if_not_loaded("graphicx") +
            mpl.texmanager._usepackage_if_not_loaded("psfrag") +
            r"\geometry{papersize={%(width)sin,%(height)sin},margin=0in}"
            % {"width": paper_width, "height": paper_height}
    }):
        dvifile = TexManager().make_dvi(
            "\n"
            r"\begin{figure}""\n"
            r"  \centering\leavevmode""\n"
            r"  %(psfrags)s""\n"
            r"  \includegraphics*[angle=%(angle)s]{%(epsfile)s}""\n"
            r"\end{figure}"
            % {
                "psfrags": "\n".join(psfrags),
                "angle": 90 if orientation == 'landscape' else 0,
                "epsfile": tmppath.resolve().as_posix(),
            },
            fontsize=10)  # tex's default fontsize.

    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, "tmp.ps")
        # -R1 is a security flag used to prevent shell command execution
        cbook._check_and_log_subprocess(
            ['dvips', '-q', '-R1', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmppath)

    # check if the dvips created a ps in landscape paper.  Somehow,
    # above latex+dvips results in a ps file in a landscape mode for a
    # certain figure sizes (e.g., 8.3in, 5.8in which is a5). And the
    # bounding box of the final output got messed up. We check see if
    # the generated ps file is in landscape and return this
    # information. The return value is used in pstoeps step to recover
    # the correct bounding box. 2010-06-05 JJL
    with open(tmppath) as fh:
        psfrag_rotated = "Landscape" in fh.read(1000)
    return psfrag_rotated


def _try_distill(func, tmppath, *args, **kwargs):
    try:
        func(str(tmppath), *args, **kwargs)
    except mpl.ExecutableNotFoundError as exc:
        _log.warning("%s.  Distillation step skipped.", exc)


def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """

    if eps:
        paper_option = ["-dEPSCrop"]
    elif ptype == "figure":
        # The bbox will have its lower-left corner at (0, 0), so upper-right
        # corner corresponds with paper size.
        paper_option = [f"-dDEVICEWIDTHPOINTS={bbox[2]}",
                        f"-dDEVICEHEIGHTPOINTS={bbox[3]}"]
    else:
        paper_option = [f"-sPAPERSIZE={ptype}"]

    psfile = tmpfile + '.ps'
    dpi = mpl.rcParams['ps.distiller.res']

    cbook._check_and_log_subprocess(
        [mpl._get_executable_info("gs").executable,
         "-dBATCH", "-dNOPAUSE", "-dSAFER", "-r%d" % dpi, "-sDEVICE=ps2write",
         *paper_option, f"-sOutputFile={psfile}", tmpfile],
        _log)

    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)

    # While it is best if above steps preserve the original bounding
    # box, there seem to be cases when it is not. For those cases,
    # the original bbox can be restored during the pstoeps step.

    if eps:
        # For some versions of gs, above steps result in a ps file where the
        # original bbox is no more correct. Do not adjust bbox for now.
        pstoeps(tmpfile, bbox, rotated=rotated)


def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    mpl._get_executable_info("gs")  # Effectively checks for ps2pdf.
    mpl._get_executable_info("pdftops")

    if eps:
        paper_option = ["-dEPSCrop"]
    elif ptype == "figure":
        # The bbox will have its lower-left corner at (0, 0), so upper-right
        # corner corresponds with paper size.
        paper_option = [f"-dDEVICEWIDTHPOINTS#{bbox[2]}",
                        f"-dDEVICEHEIGHTPOINTS#{bbox[3]}"]
    else:
        paper_option = [f"-sPAPERSIZE#{ptype}"]

    with TemporaryDirectory() as tmpdir:
        tmppdf = pathlib.Path(tmpdir, "tmp.pdf")
        tmpps = pathlib.Path(tmpdir, "tmp.ps")
        # Pass options as `-foo#bar` instead of `-foo=bar` to keep Windows
        # happy (https://ghostscript.com/doc/9.56.1/Use.htm#MS_Windows).
        cbook._check_and_log_subprocess(
            ["ps2pdf",
             "-dSAFER",
             "-dAutoFilterColorImages#false",
             "-dAutoFilterGrayImages#false",
             "-sAutoRotatePages#None",
             "-sGrayImageFilter#FlateEncode",
             "-sColorImageFilter#FlateEncode",
             *paper_option,
             tmpfile, tmppdf], _log)
        cbook._check_and_log_subprocess(
            ["pdftops", "-paper", "match", "-level3", tmppdf, tmpps], _log)
        shutil.move(tmpps, tmpfile)
    if eps:
        pstoeps(tmpfile)


@_api.deprecated("3.9")
def get_bbox_header(lbrt, rotated=False):
    """
    Return a postscript header string for the given bbox lbrt=(l, b, r, t).
    Optionally, return rotate command.
    """
    return _get_bbox_header(lbrt), (_get_rotate_command(lbrt) if rotated else "")


def _get_bbox_header(lbrt):
    """Return a PostScript header string for bounding box *lbrt*=(l, b, r, t)."""
    l, b, r, t = lbrt
    return (f"%%BoundingBox: {int(l)} {int(b)} {math.ceil(r)} {math.ceil(t)}\n"
            f"%%HiResBoundingBox: {l:.6f} {b:.6f} {r:.6f} {t:.6f}")


def _get_rotate_command(lbrt):
    """Return a PostScript 90° rotation command for bounding box *lbrt*=(l, b, r, t)."""
    l, b, r, t = lbrt
    return f"{l+r:.2f} {0:.2f} translate\n90 rotate"


def pstoeps(tmpfile, bbox=None, rotated=False):
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """

    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        # Modify the header:
        for line in tmph:
            if line.startswith(b'%!PS'):
                write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
                if bbox:
                    write(_get_bbox_header(bbox).encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                write(line)
                write(b'%%BeginProlog\n'
                      b'save\n'
                      b'countdictstack\n'
                      b'mark\n'
                      b'newpath\n'
                      b'/showpage {} def\n'
                      b'/setpagedevice {pop} def\n'
                      b'%%EndProlog\n'
                      b'%%Page 1 1\n')
                if rotated:  # The output eps file need to be rotated.
                    write(_get_rotate_command(bbox).encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound',
                                           b'%%DocumentMedia', b'%%Pages')):
                pass
            else:
                write(line)
        # Now rewrite the rest of the file, and modify the trailer.
        # This is done in a second loop such that the header of the embedded
        # eps file is not modified.
        for line in tmph:
            if line.startswith(b'%%EOF'):
                write(b'cleartomark\n'
                      b'countdictstack\n'
                      b'exch sub { end } repeat\n'
                      b'restore\n'
                      b'showpage\n'
                      b'%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                pass
            else:
                write(line)

    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)


FigureManagerPS = FigureManagerBase


# The following Python dictionary psDefs contains the entries for the
# PostScript dictionary mpldict.  This dictionary implements most of
# the matplotlib primitives and some abbreviations.
#
# References:
# https://www.adobe.com/content/dam/acom/en/devnet/actionscript/articles/PLRM.pdf
# http://preserve.mactech.com/articles/mactech/Vol.09/09.04/PostscriptTutorial
# http://www.math.ubc.ca/people/faculty/cass/graphics/text/www/
#

# The usage comments use the notation of the operator summary
# in the PostScript Language reference manual.
_psDefs = [
    # name proc  *_d*  -
    # Note that this cannot be bound to /d, because when embedding a Type3 font
    # we may want to define a "d" glyph using "/d{...} d" which would locally
    # overwrite the definition.
    "/_d { bind def } bind def",
    # x y  *m*  -
    "/m { moveto } _d",
    # x y  *l*  -
    "/l { lineto } _d",
    # x y  *r*  -
    "/r { rlineto } _d",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } _d",
    # *cl*  -
    "/cl { closepath } _d",
    # *ce*  -
    "/ce { closepath eofill } _d",
    # wx wy llx lly urx ury  *setcachedevice*  -
    "/sc { setcachedevice } _d",
]


@_Backend.export
class _BackendPS(_Backend):
    backend_version = 'Level II'
    FigureCanvas = FigureCanvasPS
<|endoftext|>import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref

from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
    _create_pdf_info_dict, _datetime_to_pdf)
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib._pylab_helpers import Gcf

_log = logging.getLogger(__name__)


_DOCUMENTCLASS = r"\documentclass{article}"


# Note: When formatting floating point values, it is important to use the
# %f/{:f} format rather than %s/{} to avoid triggering scientific notation,
# which is not recognized by TeX.

def _get_preamble():
    """Prepare a LaTeX preamble based on the rcParams configuration."""
    font_size_pt = FontProperties(
        size=mpl.rcParams["font.size"]
    ).get_size_in_points()
    return "\n".join([
        # Remove Matplotlib's custom command \mathdefault.  (Not using
        # \mathnormal instead since this looks odd with Computer Modern.)
        r"\def\mathdefault#1{#1}",
        # Use displaystyle for all math.
        r"\everymath=\expandafter{\the\everymath\displaystyle}",
        # Set up font sizes to match font.size setting.
        # If present, use the KOMA package scrextend to adjust the standard
        # LaTeX font commands (\tiny, ..., \normalsize, ..., \Huge) accordingly.
        # Otherwise, only set \normalsize, manually.
        r"\IfFileExists{scrextend.sty}{",
        r"  \usepackage[fontsize=%fpt]{scrextend}" % font_size_pt,
        r"}{",
        r"  \renewcommand{\normalsize}{\fontsize{%f}{%f}\selectfont}"
        % (font_size_pt, 1.2 * font_size_pt),
        r"  \normalsize",
        r"}",
        # Allow pgf.preamble to override the above definitions.
        mpl.rcParams["pgf.preamble"],
        *([
            r"\ifdefined\pdftexversion\else  % non-pdftex case.",
            r"  \usepackage{fontspec}",
        ] + [
            r"  \%s{%s}[Path=\detokenize{%s/}]"
            % (command, path.name, path.parent.as_posix())
            for command, path in zip(
                ["setmainfont", "setsansfont", "setmonofont"],
                [pathlib.Path(fm.findfont(family))
                 for family in ["serif", "sans\\-serif", "monospace"]]
            )
        ] + [r"\fi"] if mpl.rcParams["pgf.rcfonts"] else []),
        # Documented as "must come last".
        mpl.texmanager._usepackage_if_not_loaded("underscore", option="strings"),
    ])


# It's better to use only one unit for all coordinates, since the
# arithmetic in latex seems to produce inaccurate conversions.
latex_pt_to_in = 1. / 72.27
latex_in_to_pt = 1. / latex_pt_to_in
mpl_pt_to_in = 1. / 72.
mpl_in_to_pt = 1. / mpl_pt_to_in


def _tex_escape(text):
    r"""
    Do some necessary and/or useful substitutions for texts to be included in
    LaTeX documents.
    """
    return text.replace("\N{MINUS SIGN}", r"\ensuremath{-}")


def _writeln(fh, line):
    # Ending lines with a % prevents TeX from inserting spurious spaces
    # (https://tex.stackexchange.com/questions/7453).
    fh.write(line)
    fh.write("%\n")


def _escape_and_apply_props(s, prop):
    """
    Generate a TeX string that renders string *s* with font properties *prop*,
    also applying any required escapes to *s*.
    """
    commands = []

    families = {"serif": r"\rmfamily", "sans": r"\sffamily",
                "sans-serif": r"\sffamily", "monospace": r"\ttfamily"}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif not mpl.rcParams["pgf.rcfonts"]:
        commands.append(r"\fontfamily{\familydefault}")
    elif any(font.name == family for font in fm.fontManager.ttflist):
        commands.append(
            r"\ifdefined\pdftexversion\else\setmainfont{%s}\rmfamily\fi" % family)
    else:
        _log.warning("Ignoring unknown font: %s", family)

    size = prop.get_size_in_points()
    commands.append(r"\fontsize{%f}{%f}" % (size, size * 1.2))

    styles = {"normal": r"", "italic": r"\itshape", "oblique": r"\slshape"}
    commands.append(styles[prop.get_style()])

    boldstyles = ["semibold", "demibold", "demi", "bold", "heavy",
                  "extra bold", "black"]
    if prop.get_weight() in boldstyles:
        commands.append(r"\bfseries")

    commands.append(r"\selectfont")
    return (
        "{"
        + "".join(commands)
        + r"\catcode`\^=\active\def^{\ifmmode\sp\else\^{}\fi}"
        # It should normally be enough to set the catcode of % to 12 ("normal
        # character"); this works on TeXLive 2021 but not on 2018, so we just
        # make it active too.
        + r"\catcode`\%=\active\def%{\%}"
        + _tex_escape(s)
        + "}"
    )


def _metadata_to_str(key, value):
    """Convert metadata key/value to a form that hyperref accepts."""
    if isinstance(value, datetime.datetime):
        value = _datetime_to_pdf(value)
    elif key == 'Trapped':
        value = value.name.decode('ascii')
    else:
        value = str(value)
    return f'{key}={{{value}}}'


def make_pdf_to_png_converter():
    """Return a function that converts a pdf file to a png file."""
    try:
        mpl._get_executable_info("pdftocairo")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            ["pdftocairo", "-singlefile", "-transp", "-png", "-r", "%d" % dpi,
             pdffile, os.path.splitext(pngfile)[0]],
            stderr=subprocess.STDOUT)
    try:
        gs_info = mpl._get_executable_info("gs")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            [gs_info.executable,
             '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT',
             '-dUseCIEColor', '-dTextAlphaBits=4',
             '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE',
             '-sDEVICE=pngalpha', '-sOutputFile=%s' % pngfile,
             '-r%d' % dpi, pdffile],
            stderr=subprocess.STDOUT)
    raise RuntimeError("No suitable pdf to png renderer found.")


class LatexError(Exception):
    def __init__(self, message, latex_output=""):
        super().__init__(message)
        self.latex_output = latex_output

    def __str__(self):
        s, = self.args
        if self.latex_output:
            s += "\n" + self.latex_output
        return s


class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custom preamble in `.rcParams`.
    """

    @staticmethod
    def _build_latex_header():
        latex_header = [
            _DOCUMENTCLASS,
            # Include TeX program name as a comment for cache invalidation.
            # TeX does not allow this to be the first line.
            rf"% !TeX program = {mpl.rcParams['pgf.texsystem']}",
            # Test whether \includegraphics supports interpolate option.
            r"\usepackage{graphicx}",
            _get_preamble(),
            r"\begin{document}",
            r"\typeout{pgf_backend_query_start}",
        ]
        return "\n".join(latex_header)

    @classmethod
    def _get_cached_or_new(cls):
        """
        Return the previous LatexManager if the header and tex system did not
        change, or a new instance otherwise.
        """
        return cls._get_cached_or_new_impl(cls._build_latex_header())

    @classmethod
    @functools.lru_cache(1)
    def _get_cached_or_new_impl(cls, header):  # Helper for _get_cached_or_new.
        return cls()

    def _stdin_writeln(self, s):
        if self.latex is None:
            self._setup_latex_process()
        self.latex.stdin.write(s)
        self.latex.stdin.write("\n")
        self.latex.stdin.flush()

    def _expect(self, s):
        s = list(s)
        chars = []
        while True:
            c = self.latex.stdout.read(1)
            chars.append(c)
            if chars[-len(s):] == s:
                break
            if not c:
                self.latex.kill()
                self.latex = None
                raise LatexError("LaTeX process halted", "".join(chars))
        return "".join(chars)

    def _expect_prompt(self):
        return self._expect("\n*")

    def __init__(self):
        # create a tmp directory for running latex, register it for deletion
        self._tmpdir = TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self._finalize_tmpdir = weakref.finalize(self, self._tmpdir.cleanup)

        # test the LaTeX setup to ensure a clean startup of the subprocess
        self._setup_latex_process(expect_reply=False)
        stdout, stderr = self.latex.communicate("\n\\makeatletter\\@@end\n")
        if self.latex.returncode != 0:
            raise LatexError(
                f"LaTeX errored (probably missing font or error in preamble) "
                f"while processing the following input:\n"
                f"{self._build_latex_header()}",
                stdout)
        self.latex = None  # Will be set up on first use.
        # Per-instance cache.
        self._get_box_metrics = functools.lru_cache(self._get_box_metrics)

    def _setup_latex_process(self, *, expect_reply=True):
        # Open LaTeX process for real work; register it for deletion.  On
        # Windows, we must ensure that the subprocess has quit before being
        # able to delete the tmpdir in which it runs; in order to do so, we
        # must first `kill()` it, and then `communicate()` with or `wait()` on
        # it.
        try:
            self.latex = subprocess.Popen(
                [mpl.rcParams["pgf.texsystem"], "-halt-on-error", "-no-shell-escape"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                encoding="utf-8", cwd=self.tmpdir)
        except FileNotFoundError as err:
            raise RuntimeError(
                f"{mpl.rcParams['pgf.texsystem']!r} not found; install it or change "
                f"rcParams['pgf.texsystem'] to an available TeX implementation"
            ) from err
        except OSError as err:
            raise RuntimeError(
                f"Error starting {mpl.rcParams['pgf.texsystem']!r}") from err

        def finalize_latex(latex):
            latex.kill()
            try:
                latex.communicate()
            except RuntimeError:
                latex.wait()

        self._finalize_latex = weakref.finalize(
            self, finalize_latex, self.latex)
        # write header with 'pgf_backend_query_start' token
        self._stdin_writeln(self._build_latex_header())
        if expect_reply:  # read until 'pgf_backend_query_start' token appears
            self._expect("*pgf_backend_query_start")
            self._expect_prompt()

    def get_width_height_descent(self, text, prop):
        """
        Get the width, total height, and descent (in TeX points) for a text
        typeset by the current LaTeX environment.
        """
        return self._get_box_metrics(_escape_and_apply_props(text, prop))

    def _get_box_metrics(self, tex):
        """
        Get the width, total height and descent (in TeX points) for a TeX
        command's output in the current LaTeX environment.
        """
        # This method gets wrapped in __init__ for per-instance caching.
        self._stdin_writeln(  # Send textbox to TeX & request metrics typeout.
            # \sbox doesn't handle catcode assignments inside its argument,
            # so repeat the assignment of the catcode of "^" and "%" outside.
            r"{\catcode`\^=\active\catcode`\%%=\active\sbox0{%s}"
            r"\typeout{\the\wd0,\the\ht0,\the\dp0}}"
            % tex)
        try:
            answer = self._expect_prompt()
        except LatexError as err:
            # Here and below, use '{}' instead of {!r} to avoid doubling all
            # backslashes.
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, err.latex_output)) from err
        try:
            # Parse metrics from the answer string.  Last line is prompt, and
            # next-to-last-line is blank line from \typeout.
            width, height, offset = answer.splitlines()[-3].split(",")
        except Exception as err:
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, answer)) from err
        w, h, o = float(width[:-2]), float(height[:-2]), float(offset[:-2])
        # The height returned from LaTeX goes from base to top;
        # the height Matplotlib expects goes from bottom to top.
        return w, h + o, o


@functools.lru_cache(1)
def _get_image_inclusion_command():
    man = LatexManager._get_cached_or_new()
    man._stdin_writeln(
        r"\includegraphics[interpolate=true]{%s}"
        # Don't mess with backslashes on Windows.
        % cbook._get_data_path("images/matplotlib.png").as_posix())
    try:
        man._expect_prompt()
        return r"\includegraphics"
    except LatexError:
        # Discard the broken manager.
        LatexManager._get_cached_or_new_impl.cache_clear()
        return r"\pgfimage"


class RendererPgf(RendererBase):

    def __init__(self, figure, fh):
        """
        Create a new PGF renderer that translates any drawing instruction
        into text commands to be interpreted in a latex pgfpicture environment.

        Attributes
        ----------
        figure : `~matplotlib.figure.Figure`
            Matplotlib figure to initialize height, width and dpi from.
        fh : file-like
            File handle for the output of the drawing commands.
        """

        super().__init__()
        self.dpi = figure.dpi
        self.fh = fh
        self.figure = figure
        self.image_counter = 0

    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        # docstring inherited

        _writeln(self.fh, r"\begin{pgfscope}")

        # convert from display units to in
        f = 1. / self.dpi

        # set style and clip
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)

        # build marker definition
        bl, tr = marker_path.get_extents(marker_trans).get_points()
        coords = bl[0] * f, bl[1] * f, tr[0] * f, tr[1] * f
        _writeln(self.fh,
                 r"\pgfsys@defobject{currentmarker}"
                 r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}{" % coords)
        self._print_pgf_path(None, marker_path, marker_trans)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0,
                            fill=rgbFace is not None)
        _writeln(self.fh, r"}")

        maxcoord = 16383 / 72.27 * self.dpi  # Max dimensions in LaTeX.
        clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)

        # draw marker for each vertex
        for point, code in path.iter_segments(trans, simplify=False,
                                              clip=clip):
            x, y = point[0] * f, point[1] * f
            _writeln(self.fh, r"\begin{pgfscope}")
            _writeln(self.fh, r"\pgfsys@transformshift{%fin}{%fin}" % (x, y))
            _writeln(self.fh, r"\pgfsys@useobject{currentmarker}{}")
            _writeln(self.fh, r"\end{pgfscope}")

        _writeln(self.fh, r"\end{pgfscope}")

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        _writeln(self.fh, r"\begin{pgfscope}")
        # draw the path
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        self._print_pgf_path(gc, path, transform, rgbFace)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0,
                            fill=rgbFace is not None)
        _writeln(self.fh, r"\end{pgfscope}")

        # if present, draw pattern on top
        if gc.get_hatch():
            _writeln(self.fh, r"\begin{pgfscope}")
            self._print_pgf_path_styles(gc, rgbFace)

            # combine clip and path for clipping
            self._print_pgf_clip(gc)
            self._print_pgf_path(gc, path, transform, rgbFace)
            _writeln(self.fh, r"\pgfusepath{clip}")

            # build pattern definition
            _writeln(self.fh,
                     r"\pgfsys@defobject{currentpattern}"
                     r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}{")
            _writeln(self.fh, r"\begin{pgfscope}")
            _writeln(self.fh,
                     r"\pgfpathrectangle"
                     r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}")
            _writeln(self.fh, r"\pgfusepath{clip}")
            scale = mpl.transforms.Affine2D().scale(self.dpi)
            self._print_pgf_path(None, gc.get_hatch_path(), scale)
            self._pgf_path_draw(stroke=True)
            _writeln(self.fh, r"\end{pgfscope}")
            _writeln(self.fh, r"}")
            # repeat pattern, filling the bounding rect of the path
            f = 1. / self.dpi
            (xmin, ymin), (xmax, ymax) = \
                path.get_extents(transform).get_points()
            xmin, xmax = f * xmin, f * xmax
            ymin, ymax = f * ymin, f * ymax
            repx, repy = math.ceil(xmax - xmin), math.ceil(ymax - ymin)
            _writeln(self.fh,
                     r"\pgfsys@transformshift{%fin}{%fin}" % (xmin, ymin))
            for iy in range(repy):
                for ix in range(repx):
                    _writeln(self.fh, r"\pgfsys@useobject{currentpattern}{}")
                    _writeln(self.fh, r"\pgfsys@transformshift{1in}{0in}")
                _writeln(self.fh, r"\pgfsys@transformshift{-%din}{0in}" % repx)
                _writeln(self.fh, r"\pgfsys@transformshift{0in}{1in}")

            _writeln(self.fh, r"\end{pgfscope}")

    def _print_pgf_clip(self, gc):
        f = 1. / self.dpi
        # check for clip box
        bbox = gc.get_clip_rectangle()
        if bbox:
            p1, p2 = bbox.get_points()
            w, h = p2 - p1
            coords = p1[0] * f, p1[1] * f, w * f, h * f
            _writeln(self.fh,
                     r"\pgfpathrectangle"
                     r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}"
                     % coords)
            _writeln(self.fh, r"\pgfusepath{clip}")

        # check for clip path
        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            self._print_pgf_path(gc, clippath, clippath_trans)
            _writeln(self.fh, r"\pgfusepath{clip}")

    def _print_pgf_path_styles(self, gc, rgbFace):
        # cap style
        capstyles = {"butt": r"\pgfsetbuttcap",
                     "round": r"\pgfsetroundcap",
                     "projecting": r"\pgfsetrectcap"}
        _writeln(self.fh, capstyles[gc.get_capstyle()])

        # join style
        joinstyles = {"miter": r"\pgfsetmiterjoin",
                      "round": r"\pgfsetroundjoin",
                      "bevel": r"\pgfsetbeveljoin"}
        _writeln(self.fh, joinstyles[gc.get_joinstyle()])

        # filling
        has_fill = rgbFace is not None

        if gc.get_forced_alpha():
            fillopacity = strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]
            fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0

        if has_fill:
            _writeln(self.fh,
                     r"\definecolor{currentfill}{rgb}{%f,%f,%f}"
                     % tuple(rgbFace[:3]))
            _writeln(self.fh, r"\pgfsetfillcolor{currentfill}")
        if has_fill and fillopacity != 1.0:
            _writeln(self.fh, r"\pgfsetfillopacity{%f}" % fillopacity)

        # linewidth and color
        lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
        stroke_rgba = gc.get_rgb()
        _writeln(self.fh, r"\pgfsetlinewidth{%fpt}" % lw)
        _writeln(self.fh,
                 r"\definecolor{currentstroke}{rgb}{%f,%f,%f}"
                 % stroke_rgba[:3])
        _writeln(self.fh, r"\pgfsetstrokecolor{currentstroke}")
        if strokeopacity != 1.0:
            _writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % strokeopacity)

        # line style
        dash_offset, dash_list = gc.get_dashes()
        if dash_list is None:
            _writeln(self.fh, r"\pgfsetdash{}{0pt}")
        else:
            _writeln(self.fh,
                     r"\pgfsetdash{%s}{%fpt}"
                     % ("".join(r"{%fpt}" % dash for dash in dash_list),
                        dash_offset))

    def _print_pgf_path(self, gc, path, transform, rgbFace=None):
        f = 1. / self.dpi
        # check for clip box / ignore clip for filled paths
        bbox = gc.get_clip_rectangle() if gc else None
        maxcoord = 16383 / 72.27 * self.dpi  # Max dimensions in LaTeX.
        if bbox and (rgbFace is None):
            p1, p2 = bbox.get_points()
            clip = (max(p1[0], -maxcoord), max(p1[1], -maxcoord),
                    min(p2[0], maxcoord), min(p2[1], maxcoord))
        else:
            clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
        # build path
        for points, code in path.iter_segments(transform, clip=clip):
            if code == Path.MOVETO:
                x, y = tuple(points)
                _writeln(self.fh,
                         r"\pgfpathmoveto{\pgfqpoint{%fin}{%fin}}" %
                         (f * x, f * y))
            elif code == Path.CLOSEPOLY:
                _writeln(self.fh, r"\pgfpathclose")
            elif code == Path.LINETO:
                x, y = tuple(points)
                _writeln(self.fh,
                         r"\pgfpathlineto{\pgfqpoint{%fin}{%fin}}" %
                         (f * x, f * y))
            elif code == Path.CURVE3:
                cx, cy, px, py = tuple(points)
                coords = cx * f, cy * f, px * f, py * f
                _writeln(self.fh,
                         r"\pgfpathquadraticcurveto"
                         r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}"
                         % coords)
            elif code == Path.CURVE4:
                c1x, c1y, c2x, c2y, px, py = tuple(points)
                coords = c1x * f, c1y * f, c2x * f, c2y * f, px * f, py * f
                _writeln(self.fh,
                         r"\pgfpathcurveto"
                         r"{\pgfqpoint{%fin}{%fin}}"
                         r"{\pgfqpoint{%fin}{%fin}}"
                         r"{\pgfqpoint{%fin}{%fin}}"
                         % coords)

        # apply pgf decorators
        sketch_params = gc.get_sketch_params() if gc else None
        if sketch_params is not None:
            # Only "length" directly maps to "segment length" in PGF's API.
            # PGF uses "amplitude" to pass the combined deviation in both x-
            # and y-direction, while matplotlib only varies the length of the
            # wiggle along the line ("randomness" and "length" parameters)
            # and has a separate "scale" argument for the amplitude.
            # -> Use "randomness" as PRNG seed to allow the user to force the
            # same shape on multiple sketched lines
            scale, length, randomness = sketch_params
            if scale is not None:
                # make matplotlib and PGF rendering visually similar
                length *= 0.5
                scale *= 2
                # PGF guarantees that repeated loading is a no-op
                _writeln(self.fh, r"\usepgfmodule{decorations}")
                _writeln(self.fh, r"\usepgflibrary{decorations.pathmorphing}")
                _writeln(self.fh, r"\pgfkeys{/pgf/decoration/.cd, "
                         f"segment length = {(length * f):f}in, "
                         f"amplitude = {(scale * f):f}in}}")
                _writeln(self.fh, f"\\pgfmathsetseed{{{int(randomness)}}}")
                _writeln(self.fh, r"\pgfdecoratecurrentpath{random steps}")

    def _pgf_path_draw(self, stroke=True, fill=False):
        actions = []
        if stroke:
            actions.append("stroke")
        if fill:
            actions.append("fill")
        _writeln(self.fh, r"\pgfusepath{%s}" % ",".join(actions))

    def option_scale_image(self):
        # docstring inherited
        return True

    def option_image_nocomposite(self):
        # docstring inherited
        return not mpl.rcParams['image.composite_image']

    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return

        if not os.path.exists(getattr(self.fh, "name", "")):
            raise ValueError(
                "streamed pgf-code does not support raster graphics, consider "
                "using the pgf-to-pdf option")

        # save the images to png files
        path = pathlib.Path(self.fh.name)
        fname_img = "%s-img%d.png" % (path.stem, self.image_counter)
        Image.fromarray(im[::-1]).save(path.parent / fname_img)
        self.image_counter += 1

        # reference the image in the pgf picture
        _writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)
        f = 1. / self.dpi  # from display coords to inch
        if transform is None:
            _writeln(self.fh,
                     r"\pgfsys@transformshift{%fin}{%fin}" % (x * f, y * f))
            w, h = w * f, h * f
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            _writeln(self.fh,
                     r"\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}" %
                     (tr1 * f, tr2 * f, tr3 * f, tr4 * f,
                      (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1  # scale is already included in the transform
        interp = str(transform is None).lower()  # interpolation in PDF reader
        _writeln(self.fh,
                 r"\pgftext[left,bottom]"
                 r"{%s[interpolate=%s,width=%fin,height=%fin]{%s}}" %
                 (_get_image_inclusion_command(),
                  interp, w, h, fname_img))
        _writeln(self.fh, r"\end{pgfscope}")

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        self.draw_text(gc, x, y, s, prop, angle, ismath="TeX", mtext=mtext)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        # prepare string for tex
        s = _escape_and_apply_props(s, prop)

        _writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)

        alpha = gc.get_alpha()
        if alpha != 1.0:
            _writeln(self.fh, r"\pgfsetfillopacity{%f}" % alpha)
            _writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % alpha)
        rgb = tuple(gc.get_rgb())[:3]
        _writeln(self.fh, r"\definecolor{textcolor}{rgb}{%f,%f,%f}" % rgb)
        _writeln(self.fh, r"\pgfsetstrokecolor{textcolor}")
        _writeln(self.fh, r"\pgfsetfillcolor{textcolor}")
        s = r"\color{textcolor}" + s

        dpi = self.figure.dpi
        text_args = []
        if mtext and (
                (angle == 0 or
                 mtext.get_rotation_mode() == "anchor") and
                mtext.get_verticalalignment() != "center_baseline"):
            # if text anchoring can be supported, get the original coordinates
            # and add alignment information
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform(pos)
            halign = {"left": "left", "right": "right", "center": ""}
            valign = {"top": "top", "bottom": "bottom",
                      "baseline": "base", "center": ""}
            text_args.extend([
                f"x={x/dpi:f}in",
                f"y={y/dpi:f}in",
                halign[mtext.get_horizontalalignment()],
                valign[mtext.get_verticalalignment()],
            ])
        else:
            # if not, use the text layout provided by Matplotlib.
            text_args.append(f"x={x/dpi:f}in, y={y/dpi:f}in, left, base")

        if angle != 0:
            text_args.append("rotate=%f" % angle)

        _writeln(self.fh, r"\pgftext[%s]{%s}" % (",".join(text_args), s))
        _writeln(self.fh, r"\end{pgfscope}")

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        # get text metrics in units of latex pt, convert to display units
        w, h, d = (LatexManager._get_cached_or_new()
                   .get_width_height_descent(s, prop))
        # TODO: this should be latex_pt_to_in instead of mpl_pt_to_in
        # but having a little bit more space around the text looks better,
        # plus the bounding box reported by LaTeX is VERY narrow
        f = mpl_pt_to_in * self.dpi
        return w * f, h * f, d * f

    def flipy(self):
        # docstring inherited
        return False

    def get_canvas_width_height(self):
        # docstring inherited
        return (self.figure.get_figwidth() * self.dpi,
                self.figure.get_figheight() * self.dpi)

    def points_to_pixels(self, points):
        # docstring inherited
        return points * mpl_pt_to_in * self.dpi


class FigureCanvasPgf(FigureCanvasBase):
    filetypes = {"pgf": "LaTeX PGF picture",
                 "pdf": "LaTeX compiled PGF picture",
                 "png": "Portable Network Graphics", }

    def get_default_filetype(self):
        return 'pdf'

    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None):

        header_text = """%% Creator: Matplotlib, PGF backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Also ensure that all the required font packages are loaded; for instance,
%% the lmodern package is sometimes necessary when using math font.
%%   \\usepackage{lmodern}
%%
%% Figures using additional raster images can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%%
%% and then include the figures with
%%   \\import{<path to file>}{<filename>.pgf}
%%
"""

        # append the preamble used by the backend as a comment for debugging
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in _get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # get figure size in inch
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.dpi

        # create pgfpicture environment and write the pgf code
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write("\n")
        _writeln(fh, r"\begingroup")
        _writeln(fh, r"\makeatletter")
        _writeln(fh, r"\begin{pgfpicture}")
        _writeln(fh,
                 r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                 % (w, h))
        _writeln(fh, r"\pgfusepath{use as bounding box, clip}")
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     RendererPgf(self.figure, fh),
                                     bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        # end the pgfpicture environment
        _writeln(fh, r"\end{pgfpicture}")
        _writeln(fh, r"\makeatother")
        _writeln(fh, r"\endgroup")

    def print_pgf(self, fname_or_fh, **kwargs):
        """
        Output pgf macros for drawing the figure so it can be included and
        rendered in latex documents.
        """
        with cbook.open_file_cm(fname_or_fh, "w", encoding="utf-8") as file:
            if not cbook.file_requires_unicode(file):
                file = codecs.getwriter("utf-8")(file)
            self._print_pgf_to_fh(file, **kwargs)

    def print_pdf(self, fname_or_fh, *, metadata=None, **kwargs):
        """Use LaTeX to compile a pgf generated figure to pdf."""
        w, h = self.figure.get_size_inches()

        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in info_dict.items())

        # print figure to pgf and compile it with latex
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            self.print_pgf(tmppath / "figure.pgf", **kwargs)
            (tmppath / "figure.tex").write_text(
                "\n".join([
                    _DOCUMENTCLASS,
                    r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
                    r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
                    % (w, h),
                    r"\usepackage{pgf}",
                    _get_preamble(),
                    r"\begin{document}",
                    r"\centering",
                    r"\input{figure.pgf}",
                    r"\end{document}",
                ]), encoding="utf-8")
            texcommand = mpl.rcParams["pgf.texsystem"]
            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", "figure.tex"], _log, cwd=tmpdir)
            with ((tmppath / "figure.pdf").open("rb") as orig,
                  cbook.open_file_cm(fname_or_fh, "wb") as dest):
                shutil.copyfileobj(orig, dest)  # copy file contents to target

    def print_png(self, fname_or_fh, **kwargs):
        """Use LaTeX to compile a pgf figure to pdf and convert it to png."""
        converter = make_pdf_to_png_converter()
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            pdf_path = tmppath / "figure.pdf"
            png_path = tmppath / "figure.png"
            self.print_pdf(pdf_path, **kwargs)
            converter(pdf_path, png_path, dpi=self.figure.dpi)
            with (png_path.open("rb") as orig,
                  cbook.open_file_cm(fname_or_fh, "wb") as dest):
                shutil.copyfileobj(orig, dest)  # copy file contents to target

    def get_renderer(self):
        return RendererPgf(self.figure, None)

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()


FigureManagerPgf = FigureManagerBase


@_Backend.export
class _BackendPgf(_Backend):
    FigureCanvas = FigureCanvasPgf


class PdfPages:
    """
    A multi-page PDF file using the pgf backend

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Initialize:
    >>> with PdfPages('foo.pdf') as pdf:
    ...     # As many times as you like, create a figure fig and save it:
    ...     fig = plt.figure()
    ...     pdf.savefig(fig)
    ...     # When no figure is specified the current figure is saved
    ...     pdf.savefig()
    """

    def __init__(self, filename, *, metadata=None):
        """
        Create a new PdfPages object.

        Parameters
        ----------
        filename : str or path-like
            Plots using `PdfPages.savefig` will be written to a file at this
            location. Any older file with the same name is overwritten.

        metadata : dict, optional
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.

            Note that some versions of LaTeX engines may ignore the 'Producer'
            key and set it to themselves.
        """
        self._output_name = filename
        self._n_figures = 0
        self._metadata = (metadata or {}).copy()
        self._info_dict = _create_pdf_info_dict('pgf', self._metadata)
        self._file = BytesIO()

    def _write_header(self, width_inches, height_inches):
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in self._info_dict.items())
        latex_header = "\n".join([
            _DOCUMENTCLASS,
            r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
            r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
            % (width_inches, height_inches),
            r"\usepackage{pgf}",
            _get_preamble(),
            r"\setlength{\parindent}{0pt}",
            r"\begin{document}%",
        ])
        self._file.write(latex_header.encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Finalize this object, running LaTeX in a temporary directory
        and moving the final pdf file to *filename*.
        """
        self._file.write(rb'\end{document}\n')
        if self._n_figures > 0:
            self._run_latex()
        self._file.close()

    def _run_latex(self):
        texcommand = mpl.rcParams["pgf.texsystem"]
        with TemporaryDirectory() as tmpdir:
            tex_source = pathlib.Path(tmpdir, "pdf_pages.tex")
            tex_source.write_bytes(self._file.getvalue())
            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", tex_source],
                _log, cwd=tmpdir)
            shutil.move(tex_source.with_suffix(".pdf"), self._output_name)

    def savefig(self, figure=None, **kwargs):
        """
        Save a `.Figure` to this file as a new page.

        Any other keyword arguments are passed to `~.Figure.savefig`.

        Parameters
        ----------
        figure : `.Figure` or int, default: the active figure
            The figure, or index of the figure, that is saved to the file.
        """
        if not isinstance(figure, Figure):
            if figure is None:
                manager = Gcf.get_active()
            else:
                manager = Gcf.get_fig_manager(figure)
            if manager is None:
                raise ValueError(f"No figure {figure}")
            figure = manager.canvas.figure

        width, height = figure.get_size_inches()
        if self._n_figures == 0:
            self._write_header(width, height)
        else:
            # \pdfpagewidth and \pdfpageheight exist on pdftex, xetex, and
            # luatex<0.85; they were renamed to \pagewidth and \pageheight
            # on luatex>=0.85.
            self._file.write(
                rb'\newpage'
                rb'\ifdefined\pdfpagewidth\pdfpagewidth\else\pagewidth\fi=%fin'
                rb'\ifdefined\pdfpageheight\pdfpageheight\else\pageheight\fi=%fin'
                b'%%\n' % (width, height)
            )
        figure.savefig(self._file, format="pgf", backend="pgf", **kwargs)
        self._n_figures += 1

    def get_pagecount(self):
        """Return the current number of pages in the multipage pdf file."""
        return self._n_figures
<|endoftext|>"""
An `Anti-Grain Geometry`_ (AGG) backend.

Features that are implemented:

* capstyles and join styles
* dashes
* linewidth
* lines, rectangles, ellipses
* clipping to a rectangle
* output to RGBA and Pillow-supported image formats
* alpha blending
* DPI scaling properly - everything scales properly (dashes, linewidths, etc)
* draw polygon
* freetype2 w/ ft2font

Still TODO:

* integrate screen dpi w/ ppi and text

.. _Anti-Grain Geometry: http://agg.sourceforge.net/antigrain.com
"""

from contextlib import nullcontext
from math import radians, cos, sin

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import LoadFlags
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg


def get_hinting_flag():
    mapping = {
        'default': LoadFlags.DEFAULT,
        'no_autohint': LoadFlags.NO_AUTOHINT,
        'force_autohint': LoadFlags.FORCE_AUTOHINT,
        'no_hinting': LoadFlags.NO_HINTING,
        True: LoadFlags.FORCE_AUTOHINT,
        False: LoadFlags.NO_HINTING,
        'either': LoadFlags.DEFAULT,
        'native': LoadFlags.NO_AUTOHINT,
        'auto': LoadFlags.FORCE_AUTOHINT,
        'none': LoadFlags.NO_HINTING,
    }
    return mapping[mpl.rcParams['text.hinting']]


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    def __init__(self, width, height, dpi):
        super().__init__()

        self.dpi = dpi
        self.width = width
        self.height = height
        self._renderer = _RendererAgg(int(width), int(height), dpi)
        self._filter_renderers = []

        self._update_methods()
        self.mathtext_parser = MathTextParser('agg')

        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)

    def __getstate__(self):
        # We only want to preserve the init keywords of the Renderer.
        # Anything else can be re-created.
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        self.__init__(state['width'], state['height'], state['dpi'])

    def _update_methods(self):
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.copy_from_bbox = self._renderer.copy_from_bbox

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        nmax = mpl.rcParams['agg.path.chunksize']  # here at least for testing
        npts = path.vertices.shape[0]

        if (npts > nmax > 100 and path.should_simplify and
                rgbFace is None and gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO  # move to end of last chunk
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                try:
                    self._renderer.draw_path(gc, p, transform, rgbFace)
                except OverflowError:
                    msg = (
                        "Exceeded cell block limit in Agg.\n\n"
                        "Please reduce the value of "
                        f"rcParams['agg.path.chunksize'] (currently {nmax}) "
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']:.2f} by "
                        "default and path.simplify_threshold = "
                        f"{path.simplify_threshold:.2f} on the input)."
                    )
                    raise OverflowError(msg) from None
        else:
            try:
                self._renderer.draw_path(gc, path, transform, rgbFace)
            except OverflowError:
                cant_chunk = ''
                if rgbFace is not None:
                    cant_chunk += "- cannot split filled path\n"
                if gc.get_hatch() is not None:
                    cant_chunk += "- cannot split hatched path\n"
                if not path.should_simplify:
                    cant_chunk += "- path.should_simplify is False\n"
                if len(cant_chunk):
                    msg = (
                        "Exceeded cell block limit in Agg, however for the "
                        "following reasons:\n\n"
                        f"{cant_chunk}\n"
                        "we cannot automatically split up this path to draw."
                        "\n\nPlease manually simplify your path."
                    )

                else:
                    inc_threshold = (
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']} "
                        "by default and path.simplify_threshold "
                        f"= {path.simplify_threshold} "
                        "on the input)."
                        )
                    if nmax > 100:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please reduce "
                            "the value of rcParams['agg.path.chunksize'] "
                            f"(currently {nmax}) {inc_threshold}"
                        )
                    else:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please set "
                            "the value of rcParams['agg.path.chunksize'], "
                            f"(currently {nmax}) to be greater than 100 "
                            + inc_threshold
                        )

                raise OverflowError(msg) from None

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw mathtext using :mod:`matplotlib.mathtext`."""
        ox, oy, width, height, descent, font_image = \
            self.mathtext_parser.parse(s, self.dpi, prop,
                                       antialiased=gc.get_antialiased())

        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = round(x + ox + xd)
        y = round(y - oy + yd)
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        font = self._prepare_font(prop)
        # We pass '0' for angle here, since it will be rotated (in raster
        # space) in the following call to draw_text_image).
        font.set_text(s, 0, flags=get_hinting_flag())
        font.draw_glyphs_to_bitmap(
            antialiased=gc.get_antialiased())
        d = font.get_descent() / 64.0
        # The descent needs to be adjusted for the angle.
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xo + xd)
        y = round(y + yo + yd)
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited

        _api.check_in_list(["TeX", True, False], ismath=ismath)
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)

        if ismath:
            ox, oy, width, height, descent, font_image = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        font = self._prepare_font(prop)
        font.set_text(s, 0.0, flags=get_hinting_flag())
        w, h = font.get_width_height()  # width and height of unrotated string
        d = font.get_descent()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d /= 64.0
        return w, h, d

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        # todo, handle props, angle, origins
        size = prop.get_size_in_points()

        texmanager = self.get_texmanager()

        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)

        w, h, d = self.get_text_width_height_descent(s, prop, ismath="TeX")
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)
        self._renderer.draw_text_image(Z, x, y, angle, gc)

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    def _prepare_font(self, font_prop):
        """
        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.
        """
        font = get_font(_fontManager._find_fonts_by_props(font_prop))
        font.clear()
        size = font_prop.get_size_in_points()
        font.set_size(size, self.dpi)
        return font

    def points_to_pixels(self, points):
        # docstring inherited
        return points * self.dpi / 72

    def buffer_rgba(self):
        return memoryview(self._renderer)

    def tostring_argb(self):
        return np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()

    def clear(self):
        self._renderer.clear()

    def option_image_nocomposite(self):
        # docstring inherited

        # It is generally faster to composite each image directly to
        # the Figure, and there's no file size benefit to compositing
        # with the Agg backend
        return True

    def option_scale_image(self):
        # docstring inherited
        return False

    def restore_region(self, region, bbox=None, xy=None):
        """
        Restore the saved region. If bbox (instance of BboxBase, or
        its extents) is given, only the region specified by the bbox
        will be restored. *xy* (a pair of floats) optionally
        specifies the new position (the LLC of the original region,
        not the LLC of the bbox) where the region will be restored.

        >>> region = renderer.copy_from_bbox()
        >>> x1, y1, x2, y2 = region.get_extents()
        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),
        ...                         xy=(x1-dx, y1))

        """
        if bbox is not None or xy is not None:
            if bbox is None:
                x1, y1, x2, y2 = region.get_extents()
            elif isinstance(bbox, BboxBase):
                x1, y1, x2, y2 = bbox.extents
            else:
                x1, y1, x2, y2 = bbox

            if xy is None:
                ox, oy = x1, y1
            else:
                ox, oy = xy

            # The incoming data is float, but the _renderer type-checking wants
            # to see integers.
            self._renderer.restore_region(region, int(x1), int(y1),
                                          int(x2), int(y2), int(ox), int(oy))

        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        """
        Start filtering. It simply creates a new canvas (the old one is saved).
        """
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height),
                                      self.dpi)
        self._update_methods()

    def stop_filter(self, post_processing):
        """
        Save the current canvas as an image and apply post processing.

        The *post_processing* function::

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """
        orig_img = np.asarray(self.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]

        self._renderer = self._filter_renderers.pop()
        self._update_methods()

        if cropped_img.size:
            img, ox, oy = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255., np.uint8)
            self._renderer.draw_image(
                gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy,
                img[::-1])


class FigureCanvasAgg(FigureCanvasBase):
