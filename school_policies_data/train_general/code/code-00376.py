    def _switch_on_zoom_mode(self, event):
        self._zoom_mode = event.key
        self._mouse_move(event)

    def _switch_off_zoom_mode(self, event):
        self._zoom_mode = None
        self._mouse_move(event)

    def _mouse_move(self, event):
        """Callback for mouse moves in zoom-to-rectangle mode."""

        if self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, view = self._xypress[0]
            (x1, y1), (x2, y2) = np.clip(
                [[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
            if self._zoom_mode == "x":
                y1, y2 = a.bbox.intervaly
            elif self._zoom_mode == "y":
                x1, x2 = a.bbox.intervalx
            self.toolmanager.trigger_tool(
                'rubberband', self, data=(x1, y1, x2, y2))

    def _release(self, event):
        """Callback for mouse button releases in zoom-to-rectangle mode."""

        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []

        if not self._xypress:
            self._cancel_action()
            return

        done_ax = []

        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, _ind, view = cur_xypress
            # ignore singular clicks - 5 pixels is a threshold
            if abs(x - lastx) < 5 or abs(y - lasty) < 5:
                self._cancel_action()
                return

            # detect twinx, twiny Axes and avoid double zooming
            twinx = any(a.get_shared_x_axes().joined(a, a1) for a1 in done_ax)
            twiny = any(a.get_shared_y_axes().joined(a, a1) for a1 in done_ax)
            done_ax.append(a)

            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue

            a._set_view_from_bbox((lastx, lasty, x, y), direction,
                                  self._zoom_mode, twinx, twiny)

        self._zoom_mode = None
        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()


class ToolPan(ZoomPanBase):
    """Pan Axes with left mouse, zoom with right."""

    default_keymap = property(lambda self: mpl.rcParams['keymap.pan'])
    description = 'Pan axes with left mouse, zoom with right'
    image = 'mpl-data/images/move'
    cursor = cursors.MOVE
    radio_group = 'default'

    def __init__(self, *args):
        super().__init__(*args)
        self._id_drag = None

    def _cancel_action(self):
        self._button_pressed = None
        self._xypress = []
        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)
        self.figure.canvas.draw_idle()

    def _press(self, event):
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return

        x, y = event.x, event.y

        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_pan()):
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                self.toolmanager.messagelock(self)
                self._id_drag = self.figure.canvas.mpl_connect(
                    'motion_notify_event', self._mouse_move)

    def _release(self, event):
        if self._button_pressed is None:
            self._cancel_action()
            return

        self.figure.canvas.mpl_disconnect(self._id_drag)
        self.toolmanager.messagelock.release(self)

        for a, _ind in self._xypress:
            a.end_pan()
        if not self._xypress:
            self._cancel_action()
            return

        self.toolmanager.get_tool(_views_positions).push_current()
        self._cancel_action()

    def _mouse_move(self, event):
        for a, _ind in self._xypress:
            # safer to use the recorded button at the _press than current
            # button: # multiple button can get pressed during motion...
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        self.toolmanager.canvas.draw_idle()


class ToolHelpBase(ToolBase):
    description = 'Print tool list, shortcuts and description'
    default_keymap = property(lambda self: mpl.rcParams['keymap.help'])
    image = 'mpl-data/images/help'

    @staticmethod
    def format_shortcut(key_sequence):
        """
        Convert a shortcut string from the notation used in rc config to the
        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.
        """
        return (key_sequence if len(key_sequence) == 1 else
                re.sub(r"\+[A-Z]", r"+Shift\g<0>", key_sequence).title())

    def _format_tool_keymap(self, name):
        keymaps = self.toolmanager.get_tool_keymap(name)
        return ", ".join(self.format_shortcut(keymap) for keymap in keymaps)

    def _get_help_entries(self):
        return [(name, self._format_tool_keymap(name), tool.description)
                for name, tool in sorted(self.toolmanager.tools.items())
                if tool.description]

    def _get_help_text(self):
        entries = self._get_help_entries()
        entries = ["{}: {}\n\t{}".format(*entry) for entry in entries]
        return "\n".join(entries)

    def _get_help_html(self):
        fmt = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>"
        rows = [fmt.format(
            "<b>Action</b>", "<b>Shortcuts</b>", "<b>Description</b>")]
        rows += [fmt.format(*row) for row in self._get_help_entries()]
        return ("<style>td {padding: 0px 4px}</style>"
                "<table><thead>" + rows[0] + "</thead>"
                "<tbody>".join(rows[1:]) + "</tbody></table>")


class ToolCopyToClipboardBase(ToolBase):
    """Tool to copy the figure to the clipboard."""

    description = 'Copy the canvas figure to clipboard'
    default_keymap = property(lambda self: mpl.rcParams['keymap.copy'])

    def trigger(self, *args, **kwargs):
        message = "Copy tool is not available"
        self.toolmanager.message_event(message, self)


#: The default tools to add to a tool manager.
default_tools = {'home': ToolHome, 'back': ToolBack, 'forward': ToolForward,
                 'zoom': ToolZoom, 'pan': ToolPan,
                 'subplots': ConfigureSubplotsBase,
                 'save': SaveFigureBase,
                 'grid': ToolGrid,
                 'grid_minor': ToolMinorGrid,
                 'fullscreen': ToolFullScreen,
                 'quit': ToolQuit,
                 'quit_all': ToolQuitAll,
                 'xscale': ToolXScale,
                 'yscale': ToolYScale,
                 'position': ToolCursorPosition,
                 _views_positions: ToolViewsPositions,
                 'cursor': ToolSetCursor,
                 'rubberband': RubberbandBase,
                 'help': ToolHelpBase,
                 'copy': ToolCopyToClipboardBase,
                 }

#: The default tools to add to a container.
default_toolbar_tools = [['navigation', ['home', 'back', 'forward']],
                         ['zoompan', ['pan', 'zoom', 'subplots']],
                         ['io', ['save', 'help']]]


def add_tools_to_manager(toolmanager, tools=None):
    """
    Add multiple tools to a `.ToolManager`.

    Parameters
    ----------
    toolmanager : `.backend_managers.ToolManager`
        Manager to which the tools are added.
    tools : {str: class_like}, optional
        The tools to add in a {name: tool} dict, see
        `.backend_managers.ToolManager.add_tool` for more info. If not specified, then
        defaults to `.default_tools`.
    """
    if tools is None:
        tools = default_tools
    for name, tool in tools.items():
        toolmanager.add_tool(name, tool)


def add_tools_to_container(container, tools=None):
    """
    Add multiple tools to the container.

    Parameters
    ----------
    container : Container
        `.backend_bases.ToolContainerBase` object that will get the tools
        added.
    tools : list, optional
        List in the form ``[[group1, [tool1, tool2 ...]], [group2, [...]]]``
        where the tools ``[tool1, tool2, ...]`` will display in group1.
        See `.backend_bases.ToolContainerBase.add_tool` for details. If not specified,
        then defaults to `.default_toolbar_tools`.
    """
    if tools is None:
        tools = default_toolbar_tools
    for group, grouptools in tools:
        for position, tool in enumerate(grouptools):
            container.add_tool(tool, group, position)
<|endoftext|>"""
`pylab` is a historic interface and its use is strongly discouraged. The equivalent
replacement is `matplotlib.pyplot`.  See :ref:`api_interfaces` for a full overview
of Matplotlib interfaces.

`pylab` was designed to support a MATLAB-like way of working with all plotting related
functions directly available in the global namespace. This was achieved through a
wildcard import (``from pylab import *``).

.. warning::
   The use of `pylab` is discouraged for the following reasons:

   ``from pylab import *`` imports all the functions from `matplotlib.pyplot`, `numpy`,
   `numpy.fft`, `numpy.linalg`, and `numpy.random`, and some additional functions into
   the global namespace.

   Such a pattern is considered bad practice in modern python, as it clutters the global
   namespace. Even more severely, in the case of `pylab`, this will overwrite some
   builtin functions (e.g. the builtin `sum` will be replaced by `numpy.sum`), which
   can lead to unexpected behavior.

"""

from matplotlib.cbook import flatten, silent_list

import matplotlib as mpl

from matplotlib.dates import (
    date2num, num2date, datestr2num, drange, DateFormatter, DateLocator,
    RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator,
    HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR,
    SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY,
    relativedelta)

# bring all the symbols in so folks can import them from
# pylab in one fell swoop

## We are still importing too many things from mlab; more cleanup is needed.

from matplotlib.mlab import (
    detrend, detrend_linear, detrend_mean, detrend_none, window_hanning,
    window_none)

from matplotlib import cbook, mlab, pyplot as plt
from matplotlib.pyplot import *

from numpy import *
from numpy.fft import *
from numpy.random import *
from numpy.linalg import *

import numpy as np
import numpy.ma as ma

# don't let numpy's datetime hide stdlib
import datetime

# This is needed, or bytes will be numpy.random.bytes from
# "from numpy.random import *" above
bytes = __import__("builtins").bytes
# We also don't want the numpy version of these functions
abs = __import__("builtins").abs
bool = __import__("builtins").bool
max = __import__("builtins").max
min = __import__("builtins").min
pow = __import__("builtins").pow
round = __import__("builtins").round
<|endoftext|>"""
Classes for the efficient drawing of large collections of objects that
share most properties, e.g., a large number of line segments or
polygons.

The classes are not meant to be as flexible as their single element
counterparts (e.g., you may not be able to select all line styles) but
they are meant to be fast for common use cases (e.g., a large set of solid
line segments).
"""

import itertools
import functools
import math
from numbers import Number, Real
import warnings

import numpy as np

import matplotlib as mpl
from . import (_api, _path, artist, cbook, colorizer as mcolorizer, colors as mcolors,
               _docstring, hatch as mhatch, lines as mlines, path as mpath, transforms)
from ._enums import JoinStyle, CapStyle


# "color" is excluded; it is a compound setter, and its docstring differs
# in LineCollection.
@_api.define_aliases({
    "antialiased": ["antialiaseds", "aa"],
    "edgecolor": ["edgecolors", "ec"],
    "facecolor": ["facecolors", "fc"],
    "linestyle": ["linestyles", "dashes", "ls"],
    "linewidth": ["linewidths", "lw"],
    "offset_transform": ["transOffset"],
})
class Collection(mcolorizer.ColorizingArtist):
    r"""
    Base class for Collections. Must be subclassed to be usable.

    A Collection represents a sequence of `.Patch`\es that can be drawn
    more efficiently together than individually. For example, when a single
    path is being drawn repeatedly at different offsets, the renderer can
    typically execute a ``draw_marker()`` call much more efficiently than a
    series of repeated calls to ``draw_path()`` with the offsets put in
    one-by-one.

    Most properties of a collection can be configured per-element. Therefore,
    Collections have "plural" versions of many of the properties of a `.Patch`
    (e.g. `.Collection.get_paths` instead of `.Patch.get_path`). Exceptions are
    the *zorder*, *hatch*, *pickradius*, *capstyle* and *joinstyle* properties,
    which can only be set globally for the whole collection.

    Besides these exceptions, all properties can be specified as single values
    (applying to all elements) or sequences of values. The property of the
    ``i``\th element of the collection is::

      prop[i % len(prop)]

    Each Collection can optionally be used as its own `.ScalarMappable` by
    passing the *norm* and *cmap* parameters to its constructor. If the
    Collection's `.ScalarMappable` matrix ``_A`` has been set (via a call
    to `.Collection.set_array`), then at draw time this internal scalar
    mappable will be used to set the ``facecolors`` and ``edgecolors``,
    ignoring those that were manually passed in.
    """
    #: Either a list of 3x3 arrays or an Nx3x3 array (representing N
    #: transforms), suitable for the `all_transforms` argument to
    #: `~matplotlib.backend_bases.RendererBase.draw_path_collection`;
    #: each 3x3 array is used to initialize an
    #: `~matplotlib.transforms.Affine2D` object.
    #: Each kind of collection defines this based on its arguments.
    _transforms = np.empty((0, 3, 3))

    # Whether to draw an edge by default.  Set on a
    # subclass-by-subclass basis.
    _edge_default = False

    @_docstring.interpd
    def __init__(self, *,
                 edgecolors=None,
                 facecolors=None,
                 linewidths=None,
                 linestyles='solid',
                 capstyle=None,
                 joinstyle=None,
                 antialiaseds=None,
                 offsets=None,
                 offset_transform=None,
                 norm=None,  # optional for ScalarMappable
                 cmap=None,  # ditto
                 colorizer=None,
                 pickradius=5.0,
                 hatch=None,
                 urls=None,
                 zorder=1,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        edgecolors : :mpltype:`color` or list of colors, default: :rc:`patch.edgecolor`
            Edge color for each patch making up the collection. The special
            value 'face' can be passed to make the edgecolor match the
            facecolor.
        facecolors : :mpltype:`color` or list of colors, default: :rc:`patch.facecolor`
            Face color for each patch making up the collection.
        linewidths : float or list of floats, default: :rc:`patch.linewidth`
            Line width for each patch making up the collection.
        linestyles : str or tuple or list thereof, default: 'solid'
            Valid strings are ['solid', 'dashed', 'dashdot', 'dotted', '-',
            '--', '-.', ':']. Dash tuples should be of the form::

                (offset, onoffseq),

            where *onoffseq* is an even length tuple of on and off ink lengths
            in points. For examples, see
            :doc:`/gallery/lines_bars_and_markers/linestyles`.
        capstyle : `.CapStyle`-like, default: 'butt'
            Style to use for capping lines for all paths in the collection.
            Allowed values are %(CapStyle)s.
        joinstyle : `.JoinStyle`-like, default: 'round'
            Style to use for joining lines for all paths in the collection.
            Allowed values are %(JoinStyle)s.
        antialiaseds : bool or list of bool, default: :rc:`patch.antialiased`
            Whether each patch in the collection should be drawn with
            antialiasing.
        offsets : (float, float) or list thereof, default: (0, 0)
            A vector by which to translate each patch after rendering (default
            is no translation). The translation is performed in screen (pixel)
            coordinates (i.e. after the Artist's transform is applied).
        offset_transform : `~.Transform`, default: `.IdentityTransform`
            A single transform which will be applied to each *offsets* vector
            before it is used.
        cmap, norm
            Data normalization and colormapping parameters. See
            `.ScalarMappable` for a detailed description.
        hatch : str, optional
            Hatching pattern to use in filled paths, if any. Valid strings are
            ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']. See
            :doc:`/gallery/shapes_and_collections/hatch_style_reference` for
            the meaning of each hatch type.
        pickradius : float, default: 5.0
            If ``pickradius <= 0``, then `.Collection.contains` will return
            ``True`` whenever the test point is inside of one of the polygons
            formed by the control points of a Path in the Collection. On the
            other hand, if it is greater than 0, then we instead check if the
            test point is contained in a stroke of width ``2*pickradius``
            following any of the Paths in the Collection.
        urls : list of str, default: None
            A URL for each patch to link to once drawn. Currently only works
            for the SVG backend. See :doc:`/gallery/misc/hyperlinks_sgskip` for
            examples.
        zorder : float, default: 1
            The drawing order, shared by all Patches in the Collection. See
            :doc:`/gallery/misc/zorder_demo` for all defaults and examples.
        **kwargs
            Remaining keyword arguments will be used to set properties as
            ``Collection.set_{key}(val)`` for each key-value pair in *kwargs*.
        """

        super().__init__(self._get_colorizer(cmap, norm, colorizer))
        # list of un-scaled dash patterns
        # this is needed scaling the dash pattern by linewidth
        self._us_linestyles = [(0, None)]
        # list of dash patterns
        self._linestyles = [(0, None)]
        # list of unbroadcast/scaled linewidths
        self._us_lw = [0]
        self._linewidths = [0]

        self._gapcolor = None  # Currently only used by LineCollection.

        # Flags set by _set_mappable_flags: are colors from mapping an array?
        self._face_is_mapped = None
        self._edge_is_mapped = None
        self._mapped_colors = None  # calculated in update_scalarmappable
        self._hatch_color = mcolors.to_rgba(mpl.rcParams['hatch.color'])
        self._hatch_linewidth = mpl.rcParams['hatch.linewidth']
        self.set_facecolor(facecolors)
        self.set_edgecolor(edgecolors)
        self.set_linewidth(linewidths)
        self.set_linestyle(linestyles)
        self.set_antialiased(antialiaseds)
        self.set_pickradius(pickradius)
        self.set_urls(urls)
        self.set_hatch(hatch)
        self.set_zorder(zorder)

        if capstyle:
            self.set_capstyle(capstyle)
        else:
            self._capstyle = None

        if joinstyle:
            self.set_joinstyle(joinstyle)
        else:
            self._joinstyle = None

        if offsets is not None:
            offsets = np.asanyarray(offsets, float)
            # Broadcast (2,) -> (1, 2) but nothing else.
            if offsets.shape == (2,):
                offsets = offsets[None, :]

        self._offsets = offsets
        self._offset_transform = offset_transform

        self._path_effects = None
        self._internal_update(kwargs)
        self._paths = None

    def get_paths(self):
        return self._paths

    def set_paths(self, paths):
        self._paths = paths
        self.stale = True

    def get_transforms(self):
        return self._transforms

    def get_offset_transform(self):
        """Return the `.Transform` instance used by this artist offset."""
        if self._offset_transform is None:
            self._offset_transform = transforms.IdentityTransform()
        elif (not isinstance(self._offset_transform, transforms.Transform)
              and hasattr(self._offset_transform, '_as_mpl_transform')):
            self._offset_transform = \
                self._offset_transform._as_mpl_transform(self.axes)
        return self._offset_transform

    def set_offset_transform(self, offset_transform):
        """
        Set the artist offset transform.

        Parameters
        ----------
        offset_transform : `.Transform`
        """
        self._offset_transform = offset_transform

    def get_datalim(self, transData):
        # Calculate the data limits and return them as a `.Bbox`.
        #
        # This operation depends on the transforms for the data in the
        # collection and whether the collection has offsets:
        #
        # 1. offsets = None, transform child of transData: use the paths for
        # the automatic limits (i.e. for LineCollection in streamline).
        # 2. offsets != None: offset_transform is child of transData:
        #
        #    a. transform is child of transData: use the path + offset for
        #       limits (i.e for bar).
        #    b. transform is not a child of transData: just use the offsets
        #       for the limits (i.e. for scatter)
        #
        # 3. otherwise return a null Bbox.

        transform = self.get_transform()
        offset_trf = self.get_offset_transform()
        if not (isinstance(offset_trf, transforms.IdentityTransform)
                or offset_trf.contains_branch(transData)):
            # if the offsets are in some coords other than data,
            # then don't use them for autoscaling.
            return transforms.Bbox.null()

        paths = self.get_paths()
        if not len(paths):
            # No paths to transform
            return transforms.Bbox.null()

        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(p) for p in paths]
            # Don't convert transform to transform.get_affine() here because
            # we may have transform.contains_branch(transData) but not
            # transforms.get_affine().contains_branch(transData).  But later,
            # be careful to only apply the affine part that remains.

        offsets = self.get_offsets()

        if any(transform.contains_branch_seperately(transData)):
            # collections that are just in data units (like quiver)
            # can properly have the axes limits set by their shape +
            # offset.  LineCollections that have no offsets can
            # also use this algorithm (like streamplot).
            if isinstance(offsets, np.ma.MaskedArray):
                offsets = offsets.filled(np.nan)
                # get_path_collection_extents handles nan but not masked arrays
            return mpath.get_path_collection_extents(
                transform.get_affine() - transData, paths,
                self.get_transforms(),
                offset_trf.transform_non_affine(offsets),
                offset_trf.get_affine().frozen())

        # NOTE: None is the default case where no offsets were passed in
        if self._offsets is not None:
            # this is for collections that have their paths (shapes)
            # in physical, axes-relative, or figure-relative units
            # (i.e. like scatter). We can't uniquely set limits based on
            # those shapes, so we just set the limits based on their
            # location.
            offsets = (offset_trf - transData).transform(offsets)
            # note A-B means A B^{-1}
            offsets = np.ma.masked_invalid(offsets)
            if not offsets.mask.all():
                bbox = transforms.Bbox.null()
                bbox.update_from_data_xy(offsets)
                return bbox
        return transforms.Bbox.null()

    def get_window_extent(self, renderer=None):
        # TODO: check to ensure that this does not fail for
        # cases other than scatter plot legend
        return self.get_datalim(transforms.IdentityTransform())

    def _prepare_points(self):
        # Helper for drawing and hit testing.

        transform = self.get_transform()
        offset_trf = self.get_offset_transform()
        offsets = self.get_offsets()
        paths = self.get_paths()

        if self.have_units():
            paths = []
            for path in self.get_paths():
                vertices = path.vertices
                xs, ys = vertices[:, 0], vertices[:, 1]
                xs = self.convert_xunits(xs)
                ys = self.convert_yunits(ys)
                paths.append(mpath.Path(np.column_stack([xs, ys]), path.codes))
            xs = self.convert_xunits(offsets[:, 0])
            ys = self.convert_yunits(offsets[:, 1])
            offsets = np.ma.column_stack([xs, ys])

        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(path)
                     for path in paths]
            transform = transform.get_affine()
        if not offset_trf.is_affine:
            offsets = offset_trf.transform_non_affine(offsets)
            # This might have changed an ndarray into a masked array.
            offset_trf = offset_trf.get_affine()

        if isinstance(offsets, np.ma.MaskedArray):
            offsets = offsets.filled(np.nan)
            # Changing from a masked array to nan-filled ndarray
            # is probably most efficient at this point.

        return transform, offset_trf, offsets, paths

    @artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, self.get_gid())

        self.update_scalarmappable()

        transform, offset_trf, offsets, paths = self._prepare_points()

        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_snap(self.get_snap())

        if self._hatch:
            gc.set_hatch(self._hatch)
            gc.set_hatch_color(self._hatch_color)
            gc.set_hatch_linewidth(self._hatch_linewidth)

        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        # If the collection is made up of a single shape/color/stroke,
        # it can be rendered once and blitted multiple times, using
        # `draw_markers` rather than `draw_path_collection`.  This is
        # *much* faster for Agg, and results in smaller file sizes in
        # PDF/SVG/PS.

        trans = self.get_transforms()
        facecolors = self.get_facecolor()
        edgecolors = self.get_edgecolor()
        do_single_path_optimization = False
        if (len(paths) == 1 and len(trans) <= 1 and
                len(facecolors) == 1 and len(edgecolors) == 1 and
                len(self._linewidths) == 1 and
                all(ls[1] is None for ls in self._linestyles) and
                len(self._antialiaseds) == 1 and len(self._urls) == 1 and
                self.get_hatch() is None):
            if len(trans):
                combined_transform = transforms.Affine2D(trans[0]) + transform
            else:
                combined_transform = transform
            extents = paths[0].get_extents(combined_transform)
            if (extents.width < self.get_figure(root=True).bbox.width
                    and extents.height < self.get_figure(root=True).bbox.height):
                do_single_path_optimization = True

        if self._joinstyle:
            gc.set_joinstyle(self._joinstyle)

        if self._capstyle:
            gc.set_capstyle(self._capstyle)

        if do_single_path_optimization:
            gc.set_foreground(tuple(edgecolors[0]))
            gc.set_linewidth(self._linewidths[0])
            gc.set_dashes(*self._linestyles[0])
            gc.set_antialiased(self._antialiaseds[0])
            gc.set_url(self._urls[0])
            renderer.draw_markers(
                gc, paths[0], combined_transform.frozen(),
                mpath.Path(offsets), offset_trf, tuple(facecolors[0]))
        else:
            if self._gapcolor is not None:
                # First draw paths within the gaps.
                ipaths, ilinestyles = self._get_inverse_paths_linestyles()
                renderer.draw_path_collection(
                    gc, transform.frozen(), ipaths,
                    self.get_transforms(), offsets, offset_trf,
                    [mcolors.to_rgba("none")], self._gapcolor,
                    self._linewidths, ilinestyles,
                    self._antialiaseds, self._urls,
                    "screen")

            renderer.draw_path_collection(
                gc, transform.frozen(), paths,
                self.get_transforms(), offsets, offset_trf,
                self.get_facecolor(), self.get_edgecolor(),
                self._linewidths, self._linestyles,
                self._antialiaseds, self._urls,
                "screen")  # offset_position, kept for backcompat.

        gc.restore()
        renderer.close_group(self.__class__.__name__)
        self.stale = False

    def set_pickradius(self, pickradius):
        """
        Set the pick radius used for containment tests.

        Parameters
        ----------
        pickradius : float
            Pick radius, in points.
        """
        if not isinstance(pickradius, Real):
            raise ValueError(
                f"pickradius must be a real-valued number, not {pickradius!r}")
        self._pickradius = pickradius

    def get_pickradius(self):
        return self._pickradius

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns ``bool, dict(ind=itemlist)``, where every item in itemlist
        contains the event.
        """
        if self._different_canvas(mouseevent) or not self.get_visible():
            return False, {}
        pickradius = (
            float(self._picker)
            if isinstance(self._picker, Number) and
               self._picker is not True  # the bool, not just nonzero or 1
            else self._pickradius)
        if self.axes:
            self.axes._unstale_viewLim()
        transform, offset_trf, offsets, paths = self._prepare_points()
        # Tests if the point is contained on one of the polygons formed
        # by the control points of each of the paths. A point is considered
        # "on" a path if it would lie within a stroke of width 2*pickradius
        # following the path. If pickradius <= 0, then we instead simply check
        # if the point is *inside* of the path instead.
        ind = _path.point_in_path_collection(
            mouseevent.x, mouseevent.y, pickradius,
            transform.frozen(), paths, self.get_transforms(),
            offsets, offset_trf, pickradius <= 0)
        return len(ind) > 0, dict(ind=ind)

    def set_urls(self, urls):
        """
        Parameters
        ----------
        urls : list of str or None

        Notes
        -----
        URLs are currently only implemented by the SVG backend. They are
        ignored by all other backends.
        """
        self._urls = urls if urls is not None else [None]
        self.stale = True

    def get_urls(self):
        """
        Return a list of URLs, one for each element of the collection.

        The list contains *None* for elements without a URL. See
        :doc:`/gallery/misc/hyperlinks_sgskip` for an example.
        """
        return self._urls

    def set_hatch(self, hatch):
        r"""
        Set the hatching pattern

        *hatch* can be one of::

          /   - diagonal hatching
          \   - back diagonal
          |   - vertical
          -   - horizontal
          +   - crossed
          x   - crossed diagonal
          o   - small circle
          O   - large circle
          .   - dots
          *   - stars

        Letters can be combined, in which case all the specified
        hatchings are done.  If same letter repeats, it increases the
        density of hatching of that pattern.

        Unlike other properties such as linewidth and colors, hatching
        can only be specified for the collection as a whole, not separately
        for each member.

        Parameters
        ----------
        hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        """
        # Use validate_hatch(list) after deprecation.
        mhatch._validate_hatch_pattern(hatch)
        self._hatch = hatch
        self.stale = True

    def get_hatch(self):
        """Return the current hatching pattern."""
        return self._hatch

    def set_hatch_linewidth(self, lw):
        """Set the hatch linewidth."""
        self._hatch_linewidth = lw

    def get_hatch_linewidth(self):
        """Return the hatch linewidth."""
        return self._hatch_linewidth

    def set_offsets(self, offsets):
        """
        Set the offsets for the collection.

        Parameters
        ----------
        offsets : (N, 2) or (2,) array-like
        """
        offsets = np.asanyarray(offsets)
        if offsets.shape == (2,):  # Broadcast (2,) -> (1, 2) but nothing else.
            offsets = offsets[None, :]
        cstack = (np.ma.column_stack if isinstance(offsets, np.ma.MaskedArray)
                  else np.column_stack)
        self._offsets = cstack(
            (np.asanyarray(self.convert_xunits(offsets[:, 0]), float),
             np.asanyarray(self.convert_yunits(offsets[:, 1]), float)))
        self.stale = True

    def get_offsets(self):
        """Return the offsets for the collection."""
        # Default to zeros in the no-offset (None) case
        return np.zeros((1, 2)) if self._offsets is None else self._offsets

    def _get_default_linewidth(self):
        # This may be overridden in a subclass.
        return mpl.rcParams['patch.linewidth']  # validated as float

    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.  *lw* can be a scalar
        or a sequence; if it is a sequence the patches will cycle
        through the sequence

        Parameters
        ----------
        lw : float or list of floats
        """
        if lw is None:
            lw = self._get_default_linewidth()
        # get the un-scaled/broadcast lw
        self._us_lw = np.atleast_1d(lw)

        # scale all of the dash patterns.
        self._linewidths, self._linestyles = self._bcast_lwls(
            self._us_lw, self._us_linestyles)
        self.stale = True

    def set_linestyle(self, ls):
        """
        Set the linestyle(s) for the collection.

        ===========================   =================
        linestyle                     description
        ===========================   =================
        ``'-'`` or ``'solid'``        solid line
        ``'--'`` or  ``'dashed'``     dashed line
        ``'-.'`` or  ``'dashdot'``    dash-dotted line
        ``':'`` or ``'dotted'``       dotted line
        ===========================   =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq),

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : str or tuple or list thereof
            Valid values for individual linestyles include {'-', '--', '-.',
            ':', '', (offset, on-off-seq)}. See `.Line2D.set_linestyle` for a
            complete description.
        """
        # get the list of raw 'unscaled' dash patterns
        self._us_linestyles = mlines._get_dash_patterns(ls)

        # broadcast and scale the lw and dash patterns
        self._linewidths, self._linestyles = self._bcast_lwls(
            self._us_lw, self._us_linestyles)

    @_docstring.interpd
    def set_capstyle(self, cs):
        """
        Set the `.CapStyle` for the collection (for all its elements).

        Parameters
        ----------
        cs : `.CapStyle` or %(CapStyle)s
        """
        self._capstyle = CapStyle(cs)

    @_docstring.interpd
    def get_capstyle(self):
        """
        Return the cap style for the collection (for all its elements).

        Returns
        -------
        %(CapStyle)s or None
        """
        return self._capstyle.name if self._capstyle else None

    @_docstring.interpd
    def set_joinstyle(self, js):
        """
        Set the `.JoinStyle` for the collection (for all its elements).

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
        self._joinstyle = JoinStyle(js)

    @_docstring.interpd
    def get_joinstyle(self):
        """
        Return the join style for the collection (for all its elements).

        Returns
        -------
        %(JoinStyle)s or None
        """
        return self._joinstyle.name if self._joinstyle else None

    @staticmethod
    def _bcast_lwls(linewidths, dashes):
        """
        Internal helper function to broadcast + scale ls/lw

        In the collection drawing code, the linewidth and linestyle are cycled
        through as circular buffers (via ``v[i % len(v)]``).  Thus, if we are
        going to scale the dash pattern at set time (not draw time) we need to
        do the broadcasting now and expand both lists to be the same length.

        Parameters
        ----------
        linewidths : list
            line widths of collection
        dashes : list
            dash specification (offset, (dash pattern tuple))

        Returns
        -------
        linewidths, dashes : list
            Will be the same length, dashes are scaled by paired linewidth
        """
        if mpl.rcParams['_internal.classic_mode']:
            return linewidths, dashes
        # make sure they are the same length so we can zip them
        if len(dashes) != len(linewidths):
            l_dashes = len(dashes)
            l_lw = len(linewidths)
            gcd = math.gcd(l_dashes, l_lw)
            dashes = list(dashes) * (l_lw // gcd)
            linewidths = list(linewidths) * (l_dashes // gcd)

        # scale the dash patterns
        dashes = [mlines._scale_dashes(o, d, lw)
                  for (o, d), lw in zip(dashes, linewidths)]

        return linewidths, dashes

    def get_antialiased(self):
        """
        Get the antialiasing state for rendering.

        Returns
        -------
        array of bools
        """
        return self._antialiaseds

    def set_antialiased(self, aa):
        """
        Set the antialiasing state for rendering.

        Parameters
        ----------
        aa : bool or list of bools
        """
        if aa is None:
            aa = self._get_default_antialiased()
        self._antialiaseds = np.atleast_1d(np.asarray(aa, bool))
        self.stale = True

    def _get_default_antialiased(self):
        # This may be overridden in a subclass.
        return mpl.rcParams['patch.antialiased']

    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.

        Parameters
        ----------
        c : :mpltype:`color` or list of RGBA tuples

        See Also
        --------
        Collection.set_facecolor, Collection.set_edgecolor
            For setting the edge or face color individually.
        """
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def _get_default_facecolor(self):
        # This may be overridden in a subclass.
        return mpl.rcParams['patch.facecolor']

    def _set_facecolor(self, c):
        if c is None:
            c = self._get_default_facecolor()

        self._facecolors = mcolors.to_rgba_array(c, self._alpha)
        self.stale = True

    def set_facecolor(self, c):
        """
        Set the facecolor(s) of the collection. *c* can be a color (all patches
        have same color), or a sequence of colors; if it is a sequence the
        patches will cycle through the sequence.

        If *c* is 'none', the patch will not be filled.

        Parameters
        ----------
        c : :mpltype:`color` or list of :mpltype:`color`
        """
        if isinstance(c, str) and c.lower() in ("none", "face"):
            c = c.lower()
        self._original_facecolor = c
        self._set_facecolor(c)

    def get_facecolor(self):
        return self._facecolors

    def get_edgecolor(self):
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        else:
            return self._edgecolors

    def _get_default_edgecolor(self):
        # This may be overridden in a subclass.
        return mpl.rcParams['patch.edgecolor']

    def _set_edgecolor(self, c):
        set_hatch_color = True
        if c is None:
            if (mpl.rcParams['patch.force_edgecolor']
                    or self._edge_default
                    or cbook._str_equal(self._original_facecolor, 'none')):
                c = self._get_default_edgecolor()
            else:
                c = 'none'
                set_hatch_color = False
        if cbook._str_lower_equal(c, 'face'):
            self._edgecolors = 'face'
            self.stale = True
            return
        self._edgecolors = mcolors.to_rgba_array(c, self._alpha)
        if set_hatch_color and len(self._edgecolors):
            self._hatch_color = tuple(self._edgecolors[0])
        self.stale = True

    def set_edgecolor(self, c):
        """
        Set the edgecolor(s) of the collection.

        Parameters
        ----------
        c : :mpltype:`color` or list of :mpltype:`color` or 'face'
            The collection edgecolor(s).  If a sequence, the patches cycle
            through it.  If 'face', match the facecolor.
        """
        # We pass through a default value for use in LineCollection.
        # This allows us to maintain None as the default indicator in
        # _original_edgecolor.
        if isinstance(c, str) and c.lower() in ("none", "face"):
            c = c.lower()
        self._original_edgecolor = c
        self._set_edgecolor(c)

    def set_alpha(self, alpha):
        """
        Set the transparency of the collection.

        Parameters
        ----------
        alpha : float or array of float or None
            If not None, *alpha* values must be between 0 and 1, inclusive.
            If an array is provided, its length must match the number of
            elements in the collection.  Masked values and nans are not
            supported.
        """
        artist.Artist._set_alpha_for_array(self, alpha)
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)

    set_alpha.__doc__ = artist.Artist._set_alpha_for_array.__doc__

    def get_linewidth(self):
        return self._linewidths

    def get_linestyle(self):
        return self._linestyles

    def _set_mappable_flags(self):
        """
        Determine whether edges and/or faces are color-mapped.

        This is a helper for update_scalarmappable.
        It sets Boolean flags '_edge_is_mapped' and '_face_is_mapped'.

        Returns
        -------
        mapping_change : bool
            True if either flag is True, or if a flag has changed.
        """
        # The flags are initialized to None to ensure this returns True
        # the first time it is called.
        edge0 = self._edge_is_mapped
        face0 = self._face_is_mapped
        # After returning, the flags must be Booleans, not None.
        self._edge_is_mapped = False
        self._face_is_mapped = False
        if self._A is not None:
            if not cbook._str_equal(self._original_facecolor, 'none'):
                self._face_is_mapped = True
                if cbook._str_equal(self._original_edgecolor, 'face'):
                    self._edge_is_mapped = True
            else:
                if self._original_edgecolor is None:
                    self._edge_is_mapped = True

        mapped = self._face_is_mapped or self._edge_is_mapped
        changed = (edge0 is None or face0 is None
                   or self._edge_is_mapped != edge0
                   or self._face_is_mapped != face0)
        return mapped or changed

    def update_scalarmappable(self):
        """
        Update colors from the scalar mappable array, if any.

        Assign colors to edges and faces based on the array and/or
        colors that were directly set, as appropriate.
        """
        if not self._set_mappable_flags():
            return
        # Allow possibility to call 'self.set_array(None)'.
        if self._A is not None:
            # QuadMesh can map 2d arrays (but pcolormesh supplies 1d array)
            if self._A.ndim > 1 and not isinstance(self, _MeshData):
                raise ValueError('Collections can only map rank 1 arrays')
            if np.iterable(self._alpha):
                if self._alpha.size != self._A.size:
                    raise ValueError(
                        f'Data array shape, {self._A.shape} '
                        'is incompatible with alpha array shape, '
                        f'{self._alpha.shape}. '
                        'This can occur with the deprecated '
                        'behavior of the "flat" shading option, '
                        'in which a row and/or column of the data '
                        'array is dropped.')
                # pcolormesh, scatter, maybe others flatten their _A
                self._alpha = self._alpha.reshape(self._A.shape)
            self._mapped_colors = self.to_rgba(self._A, self._alpha)

        if self._face_is_mapped:
            self._facecolors = self._mapped_colors
        else:
            self._set_facecolor(self._original_facecolor)
        if self._edge_is_mapped:
            self._edgecolors = self._mapped_colors
        else:
            self._set_edgecolor(self._original_edgecolor)
        self.stale = True

    def get_fill(self):
        """Return whether face is colored."""
        return not cbook._str_lower_equal(self._original_facecolor, "none")

    def update_from(self, other):
        """Copy properties from other to self."""

        artist.Artist.update_from(self, other)
        self._antialiaseds = other._antialiaseds
        self._mapped_colors = other._mapped_colors
        self._edge_is_mapped = other._edge_is_mapped
        self._original_edgecolor = other._original_edgecolor
        self._edgecolors = other._edgecolors
        self._face_is_mapped = other._face_is_mapped
        self._original_facecolor = other._original_facecolor
        self._facecolors = other._facecolors
        self._linewidths = other._linewidths
        self._linestyles = other._linestyles
        self._us_linestyles = other._us_linestyles
        self._pickradius = other._pickradius
        self._hatch = other._hatch

        # update_from for scalarmappable
        self._A = other._A
        self.norm = other.norm
        self.cmap = other.cmap
        self.stale = True


class _CollectionWithSizes(Collection):
    """
    Base class for collections that have an array of sizes.
    """
    _factor = 1.0

    def get_sizes(self):
        """
        Return the sizes ('areas') of the elements in the collection.

        Returns
        -------
        array
            The 'area' of each element.
        """
        return self._sizes

    def set_sizes(self, sizes, dpi=72.0):
        """
        Set the sizes of each member of the collection.

        Parameters
        ----------
        sizes : `numpy.ndarray` or None
            The size to set for each element of the collection.  The
            value is the 'area' of the element.
        dpi : float, default: 72
            The dpi of the canvas.
        """
        if sizes is None:
            self._sizes = np.array([])
            self._transforms = np.empty((0, 3, 3))
        else:
            self._sizes = np.asarray(sizes)
            self._transforms = np.zeros((len(self._sizes), 3, 3))
            scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
            self._transforms[:, 0, 0] = scale
            self._transforms[:, 1, 1] = scale
            self._transforms[:, 2, 2] = 1.0
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        super().draw(renderer)


class PathCollection(_CollectionWithSizes):
    r"""
    A collection of `~.path.Path`\s, as created by e.g. `~.Axes.scatter`.
    """

    def __init__(self, paths, sizes=None, **kwargs):
        """
        Parameters
        ----------
        paths : list of `.path.Path`
            The paths that will make up the `.Collection`.
        sizes : array-like
            The factor by which to scale each drawn `~.path.Path`. One unit
            squared in the Path's data space is scaled to be ``sizes**2``
            points when rendered.
        **kwargs
            Forwarded to `.Collection`.
        """

        super().__init__(**kwargs)
        self.set_paths(paths)
        self.set_sizes(sizes)
        self.stale = True

    def get_paths(self):
        return self._paths

    def legend_elements(self, prop="colors", num="auto",
                        fmt=None, func=lambda x: x, **kwargs):
        """
        Create legend handles and labels for a PathCollection.

        Each legend handle is a `.Line2D` representing the Path that was drawn,
        and each label is a string that represents the Path.

        This is useful for obtaining a legend for a `~.Axes.scatter` plot;
        e.g.::

            scatter = plt.scatter([1, 2, 3],  [4, 5, 6],  c=[7, 2, 3], num=None)
            plt.legend(*scatter.legend_elements())

        creates three legend elements, one for each color with the numerical
        values passed to *c* as the labels.

        Also see the :ref:`automatedlegendcreation` example.

        Parameters
        ----------
        prop : {"colors", "sizes"}, default: "colors"
            If "colors", the legend handles will show the different colors of
            the collection. If "sizes", the legend will show the different
            sizes. To set both, use *kwargs* to directly edit the `.Line2D`
            properties.
        num : int, None, "auto" (default), array-like, or `~.ticker.Locator`
            Target number of elements to create.
            If None, use all unique elements of the mappable array. If an
            integer, target to use *num* elements in the normed range.
            If *"auto"*, try to determine which option better suits the nature
            of the data.
            The number of created elements may slightly deviate from *num* due
            to a `~.ticker.Locator` being used to find useful locations.
            If a list or array, use exactly those elements for the legend.
            Finally, a `~.ticker.Locator` can be provided.
        fmt : str, `~matplotlib.ticker.Formatter`, or None (default)
            The format or formatter to use for the labels. If a string must be
            a valid input for a `.StrMethodFormatter`. If None (the default),
            use a `.ScalarFormatter`.
        func : function, default: ``lambda x: x``
            Function to calculate the labels.  Often the size (or color)
            argument to `~.Axes.scatter` will have been pre-processed by the
            user using a function ``s = f(x)`` to make the markers visible;
            e.g. ``size = np.log10(x)``.  Providing the inverse of this
            function here allows that pre-processing to be inverted, so that
            the legend labels have the correct values; e.g. ``func = lambda
            x: 10**x``.
        **kwargs
            Allowed keyword arguments are *color* and *size*. E.g. it may be
            useful to set the color of the markers if *prop="sizes"* is used;
            similarly to set the size of the markers if *prop="colors"* is
            used. Any further parameters are passed onto the `.Line2D`
            instance. This may be useful to e.g. specify a different
            *markeredgecolor* or *alpha* for the legend handles.

        Returns
        -------
        handles : list of `.Line2D`
            Visual representation of each element of the legend.
        labels : list of str
            The string labels for elements of the legend.
        """
        handles = []
        labels = []
        hasarray = self.get_array() is not None
        if fmt is None:
            fmt = mpl.ticker.ScalarFormatter(useOffset=False, useMathText=True)
        elif isinstance(fmt, str):
            fmt = mpl.ticker.StrMethodFormatter(fmt)
        fmt.create_dummy_axis()

        if prop == "colors":
            if not hasarray:
                warnings.warn("Collection without array used. Make sure to "
                              "specify the values to be colormapped via the "
                              "`c` argument.")
                return handles, labels
            u = np.unique(self.get_array())
            size = kwargs.pop("size", mpl.rcParams["lines.markersize"])
        elif prop == "sizes":
            u = np.unique(self.get_sizes())
            color = kwargs.pop("color", "k")
        else:
            raise ValueError("Valid values for `prop` are 'colors' or "
                             f"'sizes'. You supplied '{prop}' instead.")

        fu = func(u)
        fmt.axis.set_view_interval(fu.min(), fu.max())
        fmt.axis.set_data_interval(fu.min(), fu.max())
        if num == "auto":
            num = 9
            if len(u) <= num:
                num = None
        if num is None:
            values = u
            label_values = func(values)
        else:
            if prop == "colors":
                arr = self.get_array()
            elif prop == "sizes":
                arr = self.get_sizes()
            if isinstance(num, mpl.ticker.Locator):
                loc = num
            elif np.iterable(num):
                loc = mpl.ticker.FixedLocator(num)
            else:
                num = int(num)
                loc = mpl.ticker.MaxNLocator(nbins=num, min_n_ticks=num-1,
                                             steps=[1, 2, 2.5, 3, 5, 6, 8, 10])
            label_values = loc.tick_values(func(arr).min(), func(arr).max())
            cond = ((label_values >= func(arr).min()) &
                    (label_values <= func(arr).max()))
            label_values = label_values[cond]
            yarr = np.linspace(arr.min(), arr.max(), 256)
            xarr = func(yarr)
            ix = np.argsort(xarr)
            values = np.interp(label_values, xarr[ix], yarr[ix])

        kw = {"markeredgewidth": self.get_linewidths()[0],
              "alpha": self.get_alpha(),
              **kwargs}

        for val, lab in zip(values, label_values):
            if prop == "colors":
                color = self.cmap(self.norm(val))
            elif prop == "sizes":
                size = np.sqrt(val)
                if np.isclose(size, 0.0):
                    continue
            h = mlines.Line2D([0], [0], ls="", color=color, ms=size,
                              marker=self.get_paths()[0], **kw)
            handles.append(h)
            if hasattr(fmt, "set_locs"):
                fmt.set_locs(label_values)
            l = fmt(lab)
            labels.append(l)

        return handles, labels


class PolyCollection(_CollectionWithSizes):

    def __init__(self, verts, sizes=None, *, closed=True, **kwargs):
        """
        Parameters
        ----------
        verts : list of array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (M, 2).
        sizes : array-like, default: None
            Squared scaling factors for the polygons. The coordinates of each
            polygon *verts_i* are multiplied by the square-root of the
            corresponding entry in *sizes* (i.e., *sizes* specify the scaling
            of areas). The scaling is applied before the Artist master
            transform.
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        **kwargs
            Forwarded to `.Collection`.
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self.set_verts(verts, closed)
        self.stale = True

    def set_verts(self, verts, closed=True):
        """
        Set the vertices of the polygons.

        Parameters
        ----------
        verts : list of array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (M, 2).
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        """
        self.stale = True
        if isinstance(verts, np.ma.MaskedArray):
            verts = verts.astype(float).filled(np.nan)

        # No need to do anything fancy if the path isn't closed.
        if not closed:
            self._paths = [mpath.Path(xy) for xy in verts]
            return

        # Fast path for arrays
        if isinstance(verts, np.ndarray) and len(verts.shape) == 3:
            verts_pad = np.concatenate((verts, verts[:, :1]), axis=1)
            # Creating the codes once is much faster than having Path do it
            # separately each time by passing closed=True.
            codes = np.empty(verts_pad.shape[1], dtype=mpath.Path.code_type)
            codes[:] = mpath.Path.LINETO
            codes[0] = mpath.Path.MOVETO
            codes[-1] = mpath.Path.CLOSEPOLY
            self._paths = [mpath.Path(xy, codes) for xy in verts_pad]
            return

        self._paths = []
        for xy in verts:
            if len(xy):
                self._paths.append(mpath.Path._create_closed(xy))
            else:
                self._paths.append(mpath.Path(xy))

    set_paths = set_verts

    def set_verts_and_codes(self, verts, codes):
        """Initialize vertices with path codes."""
        if len(verts) != len(codes):
            raise ValueError("'codes' must be a 1D list or array "
                             "with the same length of 'verts'")
        self._paths = [mpath.Path(xy, cds) if len(xy) else mpath.Path(xy)
                       for xy, cds in zip(verts, codes)]
        self.stale = True


class FillBetweenPolyCollection(PolyCollection):
    """
    `.PolyCollection` that fills the area between two x- or y-curves.
    """
    def __init__(
            self, t_direction, t, f1, f2, *,
            where=None, interpolate=False, step=None, **kwargs):
        """
        Parameters
        ----------
        t_direction : {{'x', 'y'}}
            The axes on which the variable lies.

            - 'x': the curves are ``(t, f1)`` and ``(t, f2)``.
            - 'y': the curves are ``(f1, t)`` and ``(f2, t)``.

        t : array-like
            The ``t_direction`` coordinates of the nodes defining the curves.

        f1 : array-like or float
            The other coordinates of the nodes defining the first curve.

        f2 : array-like or float
            The other coordinates of the nodes defining the second curve.

        where : array-like of bool, optional
            Define *where* to exclude some {dir} regions from being filled.
            The filled regions are defined by the coordinates ``t[where]``.
            More precisely, fill between ``t[i]`` and ``t[i+1]`` if
            ``where[i] and where[i+1]``.  Note that this definition implies
            that an isolated *True* value between two *False* values in *where*
            will not result in filling.  Both sides of the *True* position
            remain unfilled due to the adjacent *False* values.

        interpolate : bool, default: False
            This option is only relevant if *where* is used and the two curves
            are crossing each other.

            Semantically, *where* is often used for *f1* > *f2* or
            similar.  By default, the nodes of the polygon defining the filled
            region will only be placed at the positions in the *t* array.
            Such a polygon cannot describe the above semantics close to the
            intersection.  The t-sections containing the intersection are
            simply clipped.

            Setting *interpolate* to *True* will calculate the actual
            intersection point and extend the filled region up to this point.

        step : {{'pre', 'post', 'mid'}}, optional
            Define *step* if the filling should be a step function,
            i.e. constant in between *t*.  The value determines where the
            step will occur:

            - 'pre': The f value is continued constantly to the left from
              every *t* position, i.e. the interval ``(t[i-1], t[i]]`` has the
              value ``f[i]``.
            - 'post': The y value is continued constantly to the right from
              every *x* position, i.e. the interval ``[t[i], t[i+1])`` has the
              value ``f[i]``.
            - 'mid': Steps occur half-way between the *t* positions.

        **kwargs
            Forwarded to `.PolyCollection`.

        See Also
        --------
        .Axes.fill_between, .Axes.fill_betweenx
        """
        self.t_direction = t_direction
        self._interpolate = interpolate
        self._step = step
        verts = self._make_verts(t, f1, f2, where)
        super().__init__(verts, **kwargs)

    @staticmethod
    def _f_dir_from_t(t_direction):
        """The direction that is other than `t_direction`."""
        if t_direction == "x":
            return "y"
        elif t_direction == "y":
            return "x"
        else:
            msg = f"t_direction must be 'x' or 'y', got {t_direction!r}"
            raise ValueError(msg)

    @property
    def _f_direction(self):
        """The direction that is other than `self.t_direction`."""
        return self._f_dir_from_t(self.t_direction)

    def set_data(self, t, f1, f2, *, where=None):
        """
        Set new values for the two bounding curves.

        Parameters
        ----------
        t : array-like
            The ``self.t_direction`` coordinates of the nodes defining the curves.

        f1 : array-like or float
            The other coordinates of the nodes defining the first curve.

        f2 : array-like or float
            The other coordinates of the nodes defining the second curve.

        where : array-like of bool, optional
            Define *where* to exclude some {dir} regions from being filled.
            The filled regions are defined by the coordinates ``t[where]``.
            More precisely, fill between ``t[i]`` and ``t[i+1]`` if
            ``where[i] and where[i+1]``.  Note that this definition implies
            that an isolated *True* value between two *False* values in *where*
            will not result in filling.  Both sides of the *True* position
            remain unfilled due to the adjacent *False* values.

        See Also
        --------
        .PolyCollection.set_verts, .Line2D.set_data
        """
        t, f1, f2 = self.axes._fill_between_process_units(
            self.t_direction, self._f_direction, t, f1, f2)

        verts = self._make_verts(t, f1, f2, where)
        self.set_verts(verts)

    def get_datalim(self, transData):
        """Calculate the data limits and return them as a `.Bbox`."""
        datalim = transforms.Bbox.null()
        datalim.update_from_data_xy((self.get_transform() - transData).transform(
            np.concatenate([self._bbox, [self._bbox.minpos]])))
        return datalim

    def _make_verts(self, t, f1, f2, where):
        """
        Make verts that can be forwarded to `.PolyCollection`.
        """
        self._validate_shapes(self.t_direction, self._f_direction, t, f1, f2)

        where = self._get_data_mask(t, f1, f2, where)
        t, f1, f2 = np.broadcast_arrays(np.atleast_1d(t), f1, f2, subok=True)

        self._bbox = transforms.Bbox.null()
        self._bbox.update_from_data_xy(self._fix_pts_xy_order(np.concatenate([
            np.stack((t[where], f[where]), axis=-1) for f in (f1, f2)])))

        return [
            self._make_verts_for_region(t, f1, f2, idx0, idx1)
            for idx0, idx1 in cbook.contiguous_regions(where)
        ]

    def _get_data_mask(self, t, f1, f2, where):
        """
        Return a bool array, with True at all points that should eventually be rendered.

        The array is True at a point if none of the data inputs
        *t*, *f1*, *f2* is masked and if the input *where* is true at that point.
        """
        if where is None:
            where = True
        else:
            where = np.asarray(where, dtype=bool)
            if where.size != t.size:
                msg = "where size ({}) does not match {!r} size ({})".format(
                    where.size, self.t_direction, t.size)
                raise ValueError(msg)
        return where & ~functools.reduce(
            np.logical_or, map(np.ma.getmaskarray, [t, f1, f2]))

    @staticmethod
    def _validate_shapes(t_dir, f_dir, t, f1, f2):
        """Validate that t, f1 and f2 are 1-dimensional and have the same length."""
        names = (d + s for d, s in zip((t_dir, f_dir, f_dir), ("", "1", "2")))
        for name, array in zip(names, [t, f1, f2]):
            if array.ndim > 1:
                raise ValueError(f"{name!r} is not 1-dimensional")
            if t.size > 1 and array.size > 1 and t.size != array.size:
                msg = "{!r} has size {}, but {!r} has an unequal size of {}".format(
                    t_dir, t.size, name, array.size)
                raise ValueError(msg)

    def _make_verts_for_region(self, t, f1, f2, idx0, idx1):
        """
        Make ``verts`` for a contiguous region between ``idx0`` and ``idx1``, taking
        into account ``step`` and ``interpolate``.
        """
        t_slice = t[idx0:idx1]
        f1_slice = f1[idx0:idx1]
        f2_slice = f2[idx0:idx1]
        if self._step is not None:
            step_func = cbook.STEP_LOOKUP_MAP["steps-" + self._step]
            t_slice, f1_slice, f2_slice = step_func(t_slice, f1_slice, f2_slice)

        if self._interpolate:
            start = self._get_interpolating_points(t, f1, f2, idx0)
            end = self._get_interpolating_points(t, f1, f2, idx1)
        else:
            # Handle scalar f2 (e.g. 0): the fill should go all
            # the way down to 0 even if none of the dep1 sample points do.
            start = t_slice[0], f2_slice[0]
            end = t_slice[-1], f2_slice[-1]

        pts = np.concatenate((
            np.asarray([start]),
            np.stack((t_slice, f1_slice), axis=-1),
            np.asarray([end]),
            np.stack((t_slice, f2_slice), axis=-1)[::-1]))

        return self._fix_pts_xy_order(pts)

    @classmethod
    def _get_interpolating_points(cls, t, f1, f2, idx):
        """Calculate interpolating points."""
        im1 = max(idx - 1, 0)
        t_values = t[im1:idx+1]
        diff_values = f1[im1:idx+1] - f2[im1:idx+1]
        f1_values = f1[im1:idx+1]

        if len(diff_values) == 2:
            if np.ma.is_masked(diff_values[1]):
                return t[im1], f1[im1]
            elif np.ma.is_masked(diff_values[0]):
                return t[idx], f1[idx]

        diff_root_t = cls._get_diff_root(0, diff_values, t_values)
        diff_root_f = cls._get_diff_root(diff_root_t, t_values, f1_values)
        return diff_root_t, diff_root_f

    @staticmethod
    def _get_diff_root(x, xp, fp):
        """Calculate diff root."""
        order = xp.argsort()
        return np.interp(x, xp[order], fp[order])

    def _fix_pts_xy_order(self, pts):
        """
        Fix pts calculation results with `self.t_direction`.

        In the workflow, it is assumed that `self.t_direction` is 'x'. If this
        is not true, we need to exchange the coordinates.
        """
        return pts[:, ::-1] if self.t_direction == "y" else pts


class RegularPolyCollection(_CollectionWithSizes):
    """A collection of n-sided regular polygons."""

    _path_generator = mpath.Path.unit_regular_polygon
    _factor = np.pi ** (-1/2)

    def __init__(self,
                 numsides,
                 *,
                 rotation=0,
                 sizes=(1,),
                 **kwargs):
        """
        Parameters
        ----------
        numsides : int
            The number of sides of the polygon.
        rotation : float
            The rotation of the polygon in radians.
        sizes : tuple of float
            The area of the circle circumscribing the polygon in points^2.
        **kwargs
            Forwarded to `.Collection`.

        Examples
        --------
        See :doc:`/gallery/event_handling/lasso_demo` for a complete example::

            offsets = np.random.rand(20, 2)
            facecolors = [cm.jet(x) for x in np.random.rand(20)]

            collection = RegularPolyCollection(
                numsides=5, # a pentagon
                rotation=0, sizes=(50,),
                facecolors=facecolors,
                edgecolors=("black",),
                linewidths=(1,),
                offsets=offsets,
                offset_transform=ax.transData,
                )
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self._numsides = numsides
        self._paths = [self._path_generator(numsides)]
        self._rotation = rotation
        self.set_transform(transforms.IdentityTransform())

    def get_numsides(self):
        return self._numsides

    def get_rotation(self):
        return self._rotation

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        self._transforms = [
            transforms.Affine2D(x).rotate(-self._rotation).get_matrix()
            for x in self._transforms
        ]
        # Explicitly not super().draw, because set_sizes must be called before
        # updating self._transforms.
        Collection.draw(self, renderer)


class StarPolygonCollection(RegularPolyCollection):
    """Draw a collection of regular stars with *numsides* points."""
    _path_generator = mpath.Path.unit_regular_star


class AsteriskPolygonCollection(RegularPolyCollection):
    """Draw a collection of regular asterisks with *numsides* points."""
    _path_generator = mpath.Path.unit_regular_asterisk


class LineCollection(Collection):
    r"""
    Represents a sequence of `.Line2D`\s that should be drawn together.

    This class extends `.Collection` to represent a sequence of
    `.Line2D`\s instead of just a sequence of `.Patch`\s.
    Just as in `.Collection`, each property of a *LineCollection* may be either
    a single value or a list of values. This list is then used cyclically for
    each element of the LineCollection, so the property of the ``i``\th element
    of the collection is::

      prop[i % len(prop)]

    The properties of each member of a *LineCollection* default to their values
    in :rc:`lines.*` instead of :rc:`patch.*`, and the property *colors* is
    added in place of *edgecolors*.
    """

    _edge_default = True

    def __init__(self, segments,  # Can be None.
                 *,
                 zorder=2,        # Collection.zorder is 1
                 **kwargs
                 ):
        """
        Parameters
        ----------
        segments : list of (N, 2) array-like
            A sequence ``[line0, line1, ...]`` where each line is a (N, 2)-shape
            array-like containing points::

                line0 = [(x0, y0), (x1, y1), ...]

            Each line can contain a different number of points.
        linewidths : float or list of float, default: :rc:`lines.linewidth`
            The width of each line in points.
        colors : :mpltype:`color` or list of color, default: :rc:`lines.color`
            A sequence of RGBA tuples (e.g., arbitrary color strings, etc, not
            allowed).
        antialiaseds : bool or list of bool, default: :rc:`lines.antialiased`
            Whether to use antialiasing for each line.
        zorder : float, default: 2
            zorder of the lines once drawn.

        facecolors : :mpltype:`color` or list of :mpltype:`color`, default: 'none'
            When setting *facecolors*, each line is interpreted as a boundary
            for an area, implicitly closing the path from the last point to the
            first point. The enclosed area is filled with *facecolor*.
            In order to manually specify what should count as the "interior" of
            each line, please use `.PathCollection` instead, where the
            "interior" can be specified by appropriate usage of
            `~.path.Path.CLOSEPOLY`.

        **kwargs
            Forwarded to `.Collection`.
        """
        # Unfortunately, mplot3d needs this explicit setting of 'facecolors'.
        kwargs.setdefault('facecolors', 'none')
        super().__init__(
            zorder=zorder,
            **kwargs)
        self.set_segments(segments)

    def set_segments(self, segments):
        if segments is None:
            return

        self._paths = [mpath.Path(seg) if isinstance(seg, np.ma.MaskedArray)
                       else mpath.Path(np.asarray(seg, float))
                       for seg in segments]
        self.stale = True

    set_verts = set_segments  # for compatibility with PolyCollection
    set_paths = set_segments

    def get_segments(self):
        """
        Returns
        -------
        list
            List of segments in the LineCollection. Each list item contains an
            array of vertices.
        """
        segments = []

        for path in self._paths:
            vertices = [
                vertex
                for vertex, _
                # Never simplify here, we want to get the data-space values
                # back and there in no way to know the "right" simplification
                # threshold so never try.
                in path.iter_segments(simplify=False)
            ]
            vertices = np.asarray(vertices)
            segments.append(vertices)

        return segments

    def _get_default_linewidth(self):
        return mpl.rcParams['lines.linewidth']

    def _get_default_antialiased(self):
        return mpl.rcParams['lines.antialiased']

    def _get_default_edgecolor(self):
        return mpl.rcParams['lines.color']

    def _get_default_facecolor(self):
        return 'none'

    def set_alpha(self, alpha):
        # docstring inherited
        super().set_alpha(alpha)
        if self._gapcolor is not None:
            self.set_gapcolor(self._original_gapcolor)

    def set_color(self, c):
        """
        Set the edgecolor(s) of the LineCollection.

        Parameters
        ----------
        c : :mpltype:`color` or list of :mpltype:`color`
            Single color (all lines have same color), or a
            sequence of RGBA tuples; if it is a sequence the lines will
            cycle through the sequence.
        """
        self.set_edgecolor(c)

    set_colors = set_color

    def get_color(self):
        return self._edgecolors

    get_colors = get_color  # for compatibility with old versions

    def set_gapcolor(self, gapcolor):
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : :mpltype:`color` or list of :mpltype:`color` or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
        self._original_gapcolor = gapcolor
        self._set_gapcolor(gapcolor)

    def _set_gapcolor(self, gapcolor):
        if gapcolor is not None:
            gapcolor = mcolors.to_rgba_array(gapcolor, self._alpha)
        self._gapcolor = gapcolor
        self.stale = True

    def get_gapcolor(self):
        return self._gapcolor

    def _get_inverse_paths_linestyles(self):
        """
        Returns the path and pattern for the gaps in the non-solid lines.

        This path and pattern is the inverse of the path and pattern used to
        construct the non-solid lines. For solid lines, we set the inverse path
        to nans to prevent drawing an inverse line.
        """
        path_patterns = [
            (mpath.Path(np.full((1, 2), np.nan)), ls)
            if ls == (0, None) else
            (path, mlines._get_inverse_dash_pattern(*ls))
            for (path, ls) in
            zip(self._paths, itertools.cycle(self._linestyles))]

        return zip(*path_patterns)


class EventCollection(LineCollection):
    """
    A collection of locations along a single axis at which an "event" occurred.

    The events are given by a 1-dimensional array. They do not have an
    amplitude and are displayed as parallel lines.
    """

    _edge_default = True

    def __init__(self,
                 positions,  # Cannot be None.
                 orientation='horizontal',
                 *,
                 lineoffset=0,
                 linelength=1,
                 linewidth=None,
                 color=None,
                 linestyle='solid',
                 antialiased=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        positions : 1D array-like
            Each value is an event.
        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The sequence of events is plotted along this direction.
            The marker lines of the single events are along the orthogonal
            direction.
        lineoffset : float, default: 0
            The offset of the center of the markers from the origin, in the
            direction orthogonal to *orientation*.
        linelength : float, default: 1
            The total height of the marker (i.e. the marker stretches from
            ``lineoffset - linelength/2`` to ``lineoffset + linelength/2``).
        linewidth : float or list thereof, default: :rc:`lines.linewidth`
            The line width of the event lines, in points.
        color : :mpltype:`color` or list of :mpltype:`color`, default: :rc:`lines.color`
            The color of the event lines.
        linestyle : str or tuple or list thereof, default: 'solid'
            Valid strings are ['solid', 'dashed', 'dashdot', 'dotted',
            '-', '--', '-.', ':']. Dash tuples should be of the form::

                (offset, onoffseq),

            where *onoffseq* is an even length tuple of on and off ink
            in points.
        antialiased : bool or list thereof, default: :rc:`lines.antialiased`
            Whether to use antialiasing for drawing the lines.
        **kwargs
            Forwarded to `.LineCollection`.

        Examples
        --------
        .. plot:: gallery/lines_bars_and_markers/eventcollection_demo.py
        """
        super().__init__([],
                         linewidths=linewidth, linestyles=linestyle,
                         colors=color, antialiaseds=antialiased,
                         **kwargs)
        self._is_horizontal = True  # Initial value, may be switched below.
        self._linelength = linelength
        self._lineoffset = lineoffset
        self.set_orientation(orientation)
        self.set_positions(positions)

    def get_positions(self):
        """
        Return an array containing the floating-point values of the positions.
        """
        pos = 0 if self.is_horizontal() else 1
        return [segment[0, pos] for segment in self.get_segments()]

    def set_positions(self, positions):
        """Set the positions of the events."""
        if positions is None:
            positions = []
        if np.ndim(positions) != 1:
            raise ValueError('positions must be one-dimensional')
        lineoffset = self.get_lineoffset()
        linelength = self.get_linelength()
        pos_idx = 0 if self.is_horizontal() else 1
        segments = np.empty((len(positions), 2, 2))
        segments[:, :, pos_idx] = np.sort(positions)[:, None]
        segments[:, 0, 1 - pos_idx] = lineoffset + linelength / 2
        segments[:, 1, 1 - pos_idx] = lineoffset - linelength / 2
        self.set_segments(segments)

    def add_positions(self, position):
        """Add one or more events at the specified positions."""
        if position is None or (hasattr(position, 'len') and
                                len(position) == 0):
            return
        positions = self.get_positions()
        positions = np.hstack([positions, np.asanyarray(position)])
        self.set_positions(positions)
    extend_positions = append_positions = add_positions

    def is_horizontal(self):
        """True if the eventcollection is horizontal, False if vertical."""
        return self._is_horizontal

    def get_orientation(self):
        """
        Return the orientation of the event line ('horizontal' or 'vertical').
        """
        return 'horizontal' if self.is_horizontal() else 'vertical'

    def switch_orientation(self):
        """
        Switch the orientation of the event line, either from vertical to
        horizontal or vice versus.
        """
        segments = self.get_segments()
        for i, segment in enumerate(segments):
            segments[i] = np.fliplr(segment)
        self.set_segments(segments)
        self._is_horizontal = not self.is_horizontal()
        self.stale = True

    def set_orientation(self, orientation):
        """
        Set the orientation of the event line.

        Parameters
        ----------
        orientation : {'horizontal', 'vertical'}
        """
        is_horizontal = _api.check_getitem(
            {"horizontal": True, "vertical": False},
            orientation=orientation)
        if is_horizontal == self.is_horizontal():
            return
        self.switch_orientation()

    def get_linelength(self):
        """Return the length of the lines used to mark each event."""
        return self._linelength

    def set_linelength(self, linelength):
        """Set the length of the lines used to mark each event."""
        if linelength == self.get_linelength():
            return
        lineoffset = self.get_lineoffset()
        segments = self.get_segments()
        pos = 1 if self.is_horizontal() else 0
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.
            segment[1, pos] = lineoffset - linelength / 2.
        self.set_segments(segments)
        self._linelength = linelength

    def get_lineoffset(self):
        """Return the offset of the lines used to mark each event."""
        return self._lineoffset

    def set_lineoffset(self, lineoffset):
        """Set the offset of the lines used to mark each event."""
        if lineoffset == self.get_lineoffset():
            return
        linelength = self.get_linelength()
        segments = self.get_segments()
        pos = 1 if self.is_horizontal() else 0
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.
            segment[1, pos] = lineoffset - linelength / 2.
        self.set_segments(segments)
        self._lineoffset = lineoffset

    def get_linewidth(self):
        """Get the width of the lines used to mark each event."""
        return super().get_linewidth()[0]

    def get_linewidths(self):
        return super().get_linewidth()

    def get_color(self):
        """Return the color of the lines used to mark each event."""
        return self.get_colors()[0]


class CircleCollection(_CollectionWithSizes):
    """A collection of circles, drawn using splines."""

    _factor = np.pi ** (-1/2)

    def __init__(self, sizes, **kwargs):
        """
        Parameters
        ----------
        sizes : float or array-like
            The area of each circle in points^2.
        **kwargs
            Forwarded to `.Collection`.
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self.set_transform(transforms.IdentityTransform())
        self._paths = [mpath.Path.unit_circle()]


class EllipseCollection(Collection):
    """A collection of ellipses, drawn using splines."""

    def __init__(self, widths, heights, angles, *, units='points', **kwargs):
        """
        Parameters
        ----------
        widths : array-like
            The lengths of the first axes (e.g., major axis lengths).
        heights : array-like
            The lengths of second axes.
        angles : array-like
            The angles of the first axes, degrees CCW from the x-axis.
        units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
            The units in which majors and minors are given; 'width' and
            'height' refer to the dimensions of the axes, while 'x' and 'y'
            refer to the *offsets* data units. 'xy' differs from all others in
            that the angle as plotted varies with the aspect ratio, and equals
            the specified angle only when the aspect ratio is unity.  Hence
            it behaves the same as the `~.patches.Ellipse` with
            ``axes.transData`` as its transform.
        **kwargs
            Forwarded to `Collection`.
        """
        super().__init__(**kwargs)
        self.set_widths(widths)
        self.set_heights(heights)
        self.set_angles(angles)
        self._units = units
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))
        self._paths = [mpath.Path.unit_circle()]

    def _set_transforms(self):
        """Calculate transforms immediately before drawing."""

        ax = self.axes
        fig = self.get_figure(root=False)

        if self._units == 'xy':
            sc = 1
        elif self._units == 'x':
            sc = ax.bbox.width / ax.viewLim.width
        elif self._units == 'y':
            sc = ax.bbox.height / ax.viewLim.height
        elif self._units == 'inches':
            sc = fig.dpi
        elif self._units == 'points':
            sc = fig.dpi / 72.0
        elif self._units == 'width':
            sc = ax.bbox.width
        elif self._units == 'height':
            sc = ax.bbox.height
        elif self._units == 'dots':
            sc = 1.0
        else:
            raise ValueError(f'Unrecognized units: {self._units!r}')

        self._transforms = np.zeros((len(self._widths), 3, 3))
        widths = self._widths * sc
        heights = self._heights * sc
        sin_angle = np.sin(self._angles)
        cos_angle = np.cos(self._angles)
        self._transforms[:, 0, 0] = widths * cos_angle
        self._transforms[:, 0, 1] = heights * -sin_angle
        self._transforms[:, 1, 0] = widths * sin_angle
        self._transforms[:, 1, 1] = heights * cos_angle
        self._transforms[:, 2, 2] = 1.0

        _affine = transforms.Affine2D
        if self._units == 'xy':
            m = ax.transData.get_affine().get_matrix().copy()
            m[:2, 2:] = 0
            self.set_transform(_affine(m))

    def set_widths(self, widths):
        """Set the lengths of the first axes (e.g., major axis)."""
        self._widths = 0.5 * np.asarray(widths).ravel()
        self.stale = True

    def set_heights(self, heights):
        """Set the lengths of second axes (e.g., minor axes)."""
        self._heights = 0.5 * np.asarray(heights).ravel()
        self.stale = True

    def set_angles(self, angles):
        """Set the angles of the first axes, degrees CCW from the x-axis."""
        self._angles = np.deg2rad(angles).ravel()
        self.stale = True

    def get_widths(self):
        """Get the lengths of the first axes (e.g., major axis)."""
        return self._widths * 2

    def get_heights(self):
        """Set the lengths of second axes (e.g., minor axes)."""
        return self._heights * 2

    def get_angles(self):
        """Get the angles of the first axes, degrees CCW from the x-axis."""
        return np.rad2deg(self._angles)

    @artist.allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)


class PatchCollection(Collection):
    """
    A generic collection of patches.

    PatchCollection draws faster than a large number of equivalent individual
    Patches. It also makes it easier to assign a colormap to a heterogeneous
    collection of patches.
    """

    def __init__(self, patches, *, match_original=False, **kwargs):
        """
        Parameters
        ----------
        patches : list of `.Patch`
            A sequence of Patch objects.  This list may include
            a heterogeneous assortment of different patch types.

        match_original : bool, default: False
            If True, use the colors and linewidths of the original
            patches.  If False, new colors may be assigned by
            providing the standard collection arguments, facecolor,
            edgecolor, linewidths, norm or cmap.

        **kwargs
            All other parameters are forwarded to `.Collection`.

            If any of *edgecolors*, *facecolors*, *linewidths*, *antialiaseds*
            are None, they default to their `.rcParams` patch setting, in
            sequence form.

        Notes
        -----
        The use of `~matplotlib.cm.ScalarMappable` functionality is optional.
        If the `~matplotlib.cm.ScalarMappable` matrix ``_A`` has been set (via
        a call to `~.ScalarMappable.set_array`), at draw time a call to scalar
        mappable will be made to set the face colors.
        """

        if match_original:
            def determine_facecolor(patch):
                if patch.get_fill():
                    return patch.get_facecolor()
                return [0, 0, 0, 0]

            kwargs['facecolors'] = [determine_facecolor(p) for p in patches]
            kwargs['edgecolors'] = [p.get_edgecolor() for p in patches]
            kwargs['linewidths'] = [p.get_linewidth() for p in patches]
            kwargs['linestyles'] = [p.get_linestyle() for p in patches]
            kwargs['antialiaseds'] = [p.get_antialiased() for p in patches]

        super().__init__(**kwargs)

        self.set_paths(patches)

    def set_paths(self, patches):
        paths = [p.get_transform().transform_path(p.get_path())
                 for p in patches]
        self._paths = paths


class TriMesh(Collection):
    """
    Class for the efficient drawing of a triangular mesh using Gouraud shading.

    A triangular mesh is a `~matplotlib.tri.Triangulation` object.
    """
    def __init__(self, triangulation, **kwargs):
        super().__init__(**kwargs)
        self._triangulation = triangulation
        self._shading = 'gouraud'

        self._bbox = transforms.Bbox.unit()

        # Unfortunately this requires a copy, unless Triangulation
        # was rewritten.
        xy = np.hstack((triangulation.x.reshape(-1, 1),
                        triangulation.y.reshape(-1, 1)))
        self._bbox.update_from_data_xy(xy)

    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        return self._paths

    def set_paths(self):
        self._paths = self.convert_mesh_to_paths(self._triangulation)

    @staticmethod
    def convert_mesh_to_paths(tri):
        """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support meshes.
        """
        triangles = tri.get_masked_triangles()
        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)
        return [mpath.Path(x) for x in verts]

    @artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())
        transform = self.get_transform()

        # Get a list of triangles and the color at each vertex.
        tri = self._triangulation
        triangles = tri.get_masked_triangles()

        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)

        self.update_scalarmappable()
        colors = self._facecolors[triangles]

        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])
        renderer.draw_gouraud_triangles(gc, verts, colors, transform.frozen())
        gc.restore()
        renderer.close_group(self.__class__.__name__)


class _MeshData:
    r"""
    Class for managing the two dimensional coordinates of Quadrilateral meshes
    and the associated data with them. This class is a mixin and is intended to
    be used with another collection that will implement the draw separately.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    shading : {'flat', 'gouraud'}, default: 'flat'
    """
    def __init__(self, coordinates, *, shading='flat'):
        _api.check_shape((None, None, 2), coordinates=coordinates)
        self._coordinates = coordinates
        self._shading = shading

    def set_array(self, A):
        """
        Set the data values.

        Parameters
        ----------
        A : array-like
            The mesh data. Supported array shapes are:

            - (M, N) or (M*N,): a mesh with scalar data. The values are mapped
              to colors using normalization and a colormap. See parameters
              *norm*, *cmap*, *vmin*, *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

            If the values are provided as a 2D grid, the shape must match the
            coordinates grid. If the values are 1D, they are reshaped to 2D.
            M, N follow from the coordinates grid, where the coordinates grid
            shape is (M, N) for 'gouraud' *shading* and (M+1, N+1) for 'flat'
            shading.
        """
        height, width = self._coordinates.shape[0:-1]
        if self._shading == 'flat':
            h, w = height - 1, width - 1
        else:
            h, w = height, width
        ok_shapes = [(h, w, 3), (h, w, 4), (h, w), (h * w,)]
        if A is not None:
            shape = np.shape(A)
            if shape not in ok_shapes:
                raise ValueError(
                    f"For X ({width}) and Y ({height}) with {self._shading} "
                    f"shading, A should have shape "
                    f"{' or '.join(map(str, ok_shapes))}, not {A.shape}")
        return super().set_array(A)

    def get_coordinates(self):
        """
        Return the vertices of the mesh as an (M+1, N+1, 2) array.

        M, N are the number of quadrilaterals in the rows / columns of the
        mesh, corresponding to (M+1, N+1) vertices.
        The last dimension specifies the components (x, y).
        """
        return self._coordinates

    def get_edgecolor(self):
        # docstring inherited
        # Note that we want to return an array of shape (N*M, 4)
        # a flattened RGBA collection
        return super().get_edgecolor().reshape(-1, 4)

    def get_facecolor(self):
        # docstring inherited
        # Note that we want to return an array of shape (N*M, 4)
        # a flattened RGBA collection
        return super().get_facecolor().reshape(-1, 4)

    @staticmethod
    def _convert_mesh_to_paths(coordinates):
        """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support quadmeshes.
        """
        if isinstance(coordinates, np.ma.MaskedArray):
            c = coordinates.data
        else:
            c = coordinates
        points = np.concatenate([
            c[:-1, :-1],
            c[:-1, 1:],
            c[1:, 1:],
            c[1:, :-1],
            c[:-1, :-1]
        ], axis=2).reshape((-1, 5, 2))
        return [mpath.Path(x) for x in points]

    def _convert_mesh_to_triangles(self, coordinates):
        """
        Convert a given mesh into a sequence of triangles, each point
        with its own color.  The result can be used to construct a call to
        `~.RendererBase.draw_gouraud_triangles`.
        """
        if isinstance(coordinates, np.ma.MaskedArray):
            p = coordinates.data
        else:
            p = coordinates

        p_a = p[:-1, :-1]
        p_b = p[:-1, 1:]
        p_c = p[1:, 1:]
        p_d = p[1:, :-1]
        p_center = (p_a + p_b + p_c + p_d) / 4.0
        triangles = np.concatenate([
            p_a, p_b, p_center,
            p_b, p_c, p_center,
            p_c, p_d, p_center,
            p_d, p_a, p_center,
        ], axis=2).reshape((-1, 3, 2))

        c = self.get_facecolor().reshape((*coordinates.shape[:2], 4))
        z = self.get_array()
        mask = z.mask if np.ma.is_masked(z) else None
        if mask is not None:
            c[mask, 3] = np.nan
        c_a = c[:-1, :-1]
        c_b = c[:-1, 1:]
        c_c = c[1:, 1:]
        c_d = c[1:, :-1]
        c_center = (c_a + c_b + c_c + c_d) / 4.0
        colors = np.concatenate([
            c_a, c_b, c_center,
            c_b, c_c, c_center,
            c_c, c_d, c_center,
            c_d, c_a, c_center,
        ], axis=2).reshape((-1, 3, 4))
        tmask = np.isnan(colors[..., 2, 3])
        return triangles[~tmask], colors[~tmask]


class QuadMesh(_MeshData, Collection):
    r"""
    Class for the efficient drawing of a quadrilateral mesh.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    antialiased : bool, default: True

    shading : {'flat', 'gouraud'}, default: 'flat'

    Notes
    -----
    Unlike other `.Collection`\s, the default *pickradius* of `.QuadMesh` is 0,
    i.e. `~.Artist.contains` checks whether the test point is within any of the
    mesh quadrilaterals.

    """

    def __init__(self, coordinates, *, antialiased=True, shading='flat',
                 **kwargs):
        kwargs.setdefault("pickradius", 0)
        super().__init__(coordinates=coordinates, shading=shading)
        Collection.__init__(self, **kwargs)

        self._antialiased = antialiased
        self._bbox = transforms.Bbox.unit()
        self._bbox.update_from_data_xy(self._coordinates.reshape(-1, 2))
        self.set_mouseover(False)

    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        return self._paths

    def set_paths(self):
        self._paths = self._convert_mesh_to_paths(self._coordinates)
        self.stale = True

    def get_datalim(self, transData):
        return (self.get_transform() - transData).transform_bbox(self._bbox)

    @artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, self.get_gid())
        transform = self.get_transform()
        offset_trf = self.get_offset_transform()
        offsets = self.get_offsets()

        if self.have_units():
            xs = self.convert_xunits(offsets[:, 0])
            ys = self.convert_yunits(offsets[:, 1])
            offsets = np.column_stack([xs, ys])

        self.update_scalarmappable()

        if not transform.is_affine:
            coordinates = self._coordinates.reshape((-1, 2))
            coordinates = transform.transform(coordinates)
            coordinates = coordinates.reshape(self._coordinates.shape)
            transform = transforms.IdentityTransform()
        else:
            coordinates = self._coordinates

        if not offset_trf.is_affine:
            offsets = offset_trf.transform_non_affine(offsets)
            offset_trf = offset_trf.get_affine()

        gc = renderer.new_gc()
        gc.set_snap(self.get_snap())
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])

        if self._shading == 'gouraud':
            triangles, colors = self._convert_mesh_to_triangles(coordinates)
            renderer.draw_gouraud_triangles(
                gc, triangles, colors, transform.frozen())
        else:
            renderer.draw_quad_mesh(
                gc, transform.frozen(),
                coordinates.shape[1] - 1, coordinates.shape[0] - 1,
                coordinates, offsets, offset_trf,
                # Backends expect flattened rgba arrays (n*m, 4) for fc and ec
                self.get_facecolor().reshape((-1, 4)),
                self._antialiased, self.get_edgecolors().reshape((-1, 4)))
        gc.restore()
        renderer.close_group(self.__class__.__name__)
        self.stale = False

    def get_cursor_data(self, event):
        contained, info = self.contains(event)
        if contained and self.get_array() is not None:
            return self.get_array().ravel()[info["ind"]]
        return None


class PolyQuadMesh(_MeshData, PolyCollection):
    """
    Class for drawing a quadrilateral mesh as individual Polygons.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    Notes
    -----
    Unlike `.QuadMesh`, this class will draw each cell as an individual Polygon.
    This is significantly slower, but allows for more flexibility when wanting
    to add additional properties to the cells, such as hatching.

    Another difference from `.QuadMesh` is that if any of the vertices or data
    of a cell are masked, that Polygon will **not** be drawn and it won't be in
    the list of paths returned.
    """

    def __init__(self, coordinates, **kwargs):
        super().__init__(coordinates=coordinates)
        PolyCollection.__init__(self, verts=[], **kwargs)
        # Setting the verts updates the paths of the PolyCollection
        # This is called after the initializers to make sure the kwargs
        # have all been processed and available for the masking calculations
        self._set_unmasked_verts()

    def _get_unmasked_polys(self):
        """Get the unmasked regions using the coordinates and array"""
        # mask(X) | mask(Y)
        mask = np.any(np.ma.getmaskarray(self._coordinates), axis=-1)

        # We want the shape of the polygon, which is the corner of each X/Y array
        mask = (mask[0:-1, 0:-1] | mask[1:, 1:] | mask[0:-1, 1:] | mask[1:, 0:-1])
        arr = self.get_array()
        if arr is not None:
            arr = np.ma.getmaskarray(arr)
            if arr.ndim == 3:
                # RGB(A) case
                mask |= np.any(arr, axis=-1)
            elif arr.ndim == 2:
                mask |= arr
            else:
                mask |= arr.reshape(self._coordinates[:-1, :-1, :].shape[:2])
        return ~mask

    def _set_unmasked_verts(self):
        X = self._coordinates[..., 0]
        Y = self._coordinates[..., 1]

        unmask = self._get_unmasked_polys()
        X1 = np.ma.filled(X[:-1, :-1])[unmask]
        Y1 = np.ma.filled(Y[:-1, :-1])[unmask]
        X2 = np.ma.filled(X[1:, :-1])[unmask]
        Y2 = np.ma.filled(Y[1:, :-1])[unmask]
        X3 = np.ma.filled(X[1:, 1:])[unmask]
        Y3 = np.ma.filled(Y[1:, 1:])[unmask]
        X4 = np.ma.filled(X[:-1, 1:])[unmask]
        Y4 = np.ma.filled(Y[:-1, 1:])[unmask]
        npoly = len(X1)

        xy = np.ma.stack([X1, Y1, X2, Y2, X3, Y3, X4, Y4, X1, Y1], axis=-1)
        verts = xy.reshape((npoly, 5, 2))
        self.set_verts(verts)

    def get_edgecolor(self):
        # docstring inherited
        # We only want to return the facecolors of the polygons
        # that were drawn.
        ec = super().get_edgecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        if len(ec) != len(unmasked_polys):
            # Mapping is off
            return ec
        return ec[unmasked_polys, :]

    def get_facecolor(self):
        # docstring inherited
        # We only want to return the facecolors of the polygons
        # that were drawn.
        fc = super().get_facecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        if len(fc) != len(unmasked_polys):
            # Mapping is off
            return fc
        return fc[unmasked_polys, :]

    def set_array(self, A):
        # docstring inherited
        prev_unmask = self._get_unmasked_polys()
        super().set_array(A)
        # If the mask has changed at all we need to update
        # the set of Polys that we are drawing
        if not np.array_equal(prev_unmask, self._get_unmasked_polys()):
            self._set_unmasked_verts()
<|endoftext|>import numbers

import numpy as np

from matplotlib import _api, _docstring, transforms
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis
from matplotlib.transforms import Transform


class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation, location, functions, transform=None,
                 **kwargs):
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """
        _api.check_in_list(["x", "y"], orientation=orientation)
        self._functions = functions
        self._parent = parent
        self._orientation = orientation
        self._ticks_set = False

        fig = self._parent.get_figure(root=False)
        if self._orientation == 'x':
            super().__init__(fig, [0, 1., 1, 0.0001], **kwargs)
            self._axis = self.xaxis
            self._locstrings = ['top', 'bottom']
            self._otherstrings = ['left', 'right']
        else:  # 'y'
            super().__init__(fig, [0, 1., 0.0001, 1], **kwargs)
            self._axis = self.yaxis
            self._locstrings = ['right', 'left']
            self._otherstrings = ['top', 'bottom']
        self._parentscale = None
        # this gets positioned w/o constrained_layout so exclude:

        self.set_location(location, transform)
        self.set_functions(functions)

        # styling:
        otheraxis = self.yaxis if self._orientation == 'x' else self.xaxis
        otheraxis.set_major_locator(mticker.NullLocator())
        otheraxis.set_ticks_position('none')

        self.spines[self._otherstrings].set_visible(False)
        self.spines[self._locstrings].set_visible(True)

        if self._pos < 0.5:
            # flip the location strings...
            self._locstrings = self._locstrings[::-1]
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the Axes.

        Parameters
        ----------
        align : {'top', 'bottom', 'left', 'right'}
            Either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
        _api.check_in_list(self._locstrings, align=align)
        if align == self._locstrings[1]:  # Need to change the orientation.
            self._locstrings = self._locstrings[::-1]
        self.spines[self._locstrings[0]].set_visible(True)
        self.spines[self._locstrings[1]].set_visible(False)
        self._axis.set_ticks_position(align)
        self._axis.set_label_position(align)

    def set_location(self, location, transform=None):
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized coordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent Axes to put the new Axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).

        transform : `.Transform`, optional
            Transform for the location to use. Defaults to
            the parent's ``transAxes``, so locations are normally relative to
            the parent axes.

            .. versionadded:: 3.9
        """

        _api.check_isinstance((transforms.Transform, None), transform=transform)

        # This puts the rectangle into figure-relative coordinates.
        if isinstance(location, str):
            _api.check_in_list(self._locstrings, location=location)
            self._pos = 1. if location in ('top', 'right') else 0.
        elif isinstance(location, numbers.Real):
            self._pos = location
        else:
            raise ValueError(
                f"location must be {self._locstrings[0]!r}, "
                f"{self._locstrings[1]!r}, or a float, not {location!r}")

        self._loc = location

        if self._orientation == 'x':
            # An x-secondary axes is like an inset axes from x = 0 to x = 1 and
            # from y = pos to y = pos + eps, in the parent's transAxes coords.
            bounds = [0, self._pos, 1., 1e-10]

            # If a transformation is provided, use its y component rather than
            # the parent's transAxes. This can be used to place axes in the data
            # coords, for instance.
            if transform is not None:
                transform = transforms.blended_transform_factory(
                    self._parent.transAxes, transform)
        else:  # 'y'
            bounds = [self._pos, 0, 1e-10, 1]
            if transform is not None:
                transform = transforms.blended_transform_factory(
                    transform, self._parent.transAxes)  # Use provided x axis

        # If no transform is provided, use the parent's transAxes
        if transform is None:
            transform = self._parent.transAxes

        # this locator lets the axes move in the parent axes coordinates.
        # so it never needs to know where the parent is explicitly in
        # figure coordinates.
        # it gets called in ax.apply_aspect() (of all places)
        self.set_axes_locator(_TransformedBoundsLocator(bounds, transform))

    def apply_aspect(self, position=None):
        # docstring inherited.
        self._set_lims()
        super().apply_aspect(position)

    @_docstring.copy(Axis.set_ticks)
    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        ret = self._axis.set_ticks(ticks, labels, minor=minor, **kwargs)
        self.stale = True
        self._ticks_set = True
        return ret

    def set_functions(self, functions):
        """
        Set how the secondary axis converts limits from the parent Axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """

        if (isinstance(functions, tuple) and len(functions) == 2 and
                callable(functions[0]) and callable(functions[1])):
            # make an arbitrary convert from a two-tuple of functions
            # forward and inverse.
            self._functions = functions
        elif isinstance(functions, Transform):
            self._functions = (
                 functions.transform,
                 lambda x: functions.inverted().transform(x)
            )
        elif functions is None:
            self._functions = (lambda x: x, lambda x: x)
        else:
            raise ValueError('functions argument of secondary Axes '
                             'must be a two-tuple of callable functions '
                             'with the first function being the transform '
                             'and the second being the inverse')
        self._set_scale()

    def draw(self, renderer):
        """
        Draw the secondary Axes.

        Consults the parent Axes for its limits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_functions` (or *functions*
        parameter when Axes initialized.)
        """
        self._set_lims()
        # this sets the scale in case the parent has set its scale.
        self._set_scale()
        super().draw(renderer)

    def _set_scale(self):
        """
        Check if parent has set its scale
        """

        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        else:  # 'y'
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        if pscale == self._parentscale:
            return

        if self._ticks_set:
            ticks = self._axis.get_ticklocs()

        # need to invert the roles here for the ticks to line up.
        set_scale('functionlog' if pscale == 'log' else 'function',
                  functions=self._functions[::-1])

        # OK, set_scale sets the locators, but if we've called
        # axsecond.set_ticks, we want to keep those.
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))

        # If the parent scale doesn't change, we can skip this next time.
        self._parentscale = pscale

    def _set_lims(self):
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary Axes.
        """
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        else:  # 'y'
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        if neworder != order:
            # Flip because the transform will take care of the flipping.
            lims = lims[::-1]
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        """
        Secondary Axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
        _api.warn_external("Secondary Axes can't set the aspect ratio")

    def set_color(self, color):
        """
        Change the color of the secondary Axes and all decorators.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        axis = self._axis_map[self._orientation]
        axis.set_tick_params(colors=color)
        for spine in self.spines.values():
            if spine.axis is axis:
                spine.set_color(color)
        axis.label.set_color(color)


_secax_docstring = '''
Warnings
--------
This method is experimental as of 3.1, and the API may change.

Parameters
----------
location : {'top', 'bottom', 'left', 'right'} or float
    The position to put the secondary axis.  Strings can be 'top' or
    'bottom' for orientation='x' and 'right' or 'left' for
    orientation='y'. A float indicates the relative position on the
    parent Axes to put the new Axes, 0.0 being the bottom (or left)
    and 1.0 being the top (or right).

functions : 2-tuple of func, or Transform with an inverse

    If a 2-tuple of functions, the user specifies the transform
    function and its inverse.  i.e.
    ``functions=(lambda x: 2 / x, lambda x: 2 / x)`` would be an
    reciprocal transform with a factor of 2. Both functions must accept
    numpy arrays as input.

    The user can also directly supply a subclass of
    `.transforms.Transform` so long as it has an inverse.

    See :doc:`/gallery/subplots_axes_and_figures/secondary_axis`
    for examples of making these conversions.

transform : `.Transform`, optional
    If specified, *location* will be
    placed relative to this transform (in the direction of the axis)
    rather than the parent's axis. i.e. a secondary x-axis will
    use the provided y transform and the x transform of the parent.

    .. versionadded:: 3.9

Returns
-------
ax : axes._secondary_axes.SecondaryAxis

Other Parameters
----------------
**kwargs : `~matplotlib.axes.Axes` properties.
    Other miscellaneous Axes parameters.
'''
_docstring.interpd.register(_secax_docstring=_secax_docstring)
<|endoftext|>from . import _base
from ._axes import Axes

# Backcompat.
Subplot = Axes


class _SubplotBaseMeta(type):
    def __instancecheck__(self, obj):
        return (isinstance(obj, _base._AxesBase)
                and obj.get_subplotspec() is not None)


class SubplotBase(metaclass=_SubplotBaseMeta):
    pass


def subplot_class_factory(cls): return cls
<|endoftext|>from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import re
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

_log = logging.getLogger(__name__)


class _axis_method_wrapper:
    """
    Helper to generate Axes methods wrapping Axis methods.

    After ::

        get_foo = _axis_method_wrapper("xaxis", "get_bar")

    (in the body of a class) ``get_foo`` is a method that forwards it arguments
    to the ``get_bar`` method of the ``xaxis`` attribute, and gets its
    signature and docstring from ``Axis.get_bar``.

    The docstring of ``get_foo`` is built by replacing "this Axis" by "the
    {attr_name}" (i.e., "the xaxis", "the yaxis") in the wrapped method's
    dedented docstring; additional replacements can be given in *doc_sub*.
    """

    def __init__(self, attr_name, method_name, *, doc_sub=None):
        self.attr_name = attr_name
        self.method_name = method_name
        # Immediately put the docstring in ``self.__doc__`` so that docstring
        # manipulations within the class body work as expected.
        doc = inspect.getdoc(getattr(maxis.Axis, method_name))
        self._missing_subs = []
        if doc:
            doc_sub = {"this Axis": f"the {self.attr_name}", **(doc_sub or {})}
            for k, v in doc_sub.items():
                if k not in doc:  # Delay raising error until we know qualname.
                    self._missing_subs.append(k)
                doc = doc.replace(k, v)
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        # This is called at the end of the class body as
        # ``self.__set_name__(cls, name_under_which_self_is_assigned)``; we
        # rely on that to give the wrapper the correct __name__/__qualname__.
        get_method = attrgetter(f"{self.attr_name}.{self.method_name}")

        def wrapper(self, *args, **kwargs):
            return get_method(self)(*args, **kwargs)

        wrapper.__module__ = owner.__module__
        wrapper.__name__ = name
        wrapper.__qualname__ = f"{owner.__qualname__}.{name}"
        wrapper.__doc__ = self.__doc__
        # Manually copy the signature instead of using functools.wraps because
        # displaying the Axis method source when asking for the Axes method
        # source would be confusing.
        wrapper.__signature__ = inspect.signature(
            getattr(maxis.Axis, self.method_name))

        if self._missing_subs:
            raise ValueError(
                "The definition of {} expected that the docstring of Axis.{} "
                "contains {!r} as substrings".format(
                    wrapper.__qualname__, self.method_name,
                    ", ".join(map(repr, self._missing_subs))))

        setattr(owner, name, wrapper)


class _TransformedBoundsLocator:
    """
    Axes locator for `.Axes.inset_axes` and similarly positioned Axes.

    The locator is a callable object used in `.Axes.set_aspect` to compute the
    Axes location depending on the renderer.
    """

    def __init__(self, bounds, transform):
        """
        *bounds* (a ``[l, b, w, h]`` rectangle) and *transform* together
        specify the position of the inset Axes.
        """
        self._bounds = bounds
        self._transform = transform

    def __call__(self, ax, renderer):
        # Subtracting transSubfigure will typically rely on inverted(),
        # freezing the transform; thus, this needs to be delayed until draw
        # time as transSubfigure may otherwise change after this is evaluated.
        return mtransforms.TransformedBbox(
            mtransforms.Bbox.from_bounds(*self._bounds),
            self._transform - ax.get_figure(root=False).transSubfigure)


def _process_plot_format(fmt, *, ambiguous_fmt_datakey=False):
    """
    Convert a MATLAB style color/line style format string to a (*linestyle*,
    *marker*, *color*) tuple.

    Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    See Also
    --------
    matplotlib.Line2D.lineStyles, matplotlib.colors.cnames
        All possible styles and color format strings.
    """

    linestyle = None
    marker = None
    color = None

    # First check whether fmt is just a colorspec, but specifically exclude the
    # grayscale string "1" (not "1.0"), which is interpreted as the tri_down
    # marker "1".  The grayscale string "0" could be unambiguously understood
    # as a color (black) but also excluded for consistency.
    if fmt not in ["0", "1"]:
        try:
            color = mcolors.to_rgba(fmt)
            return linestyle, marker, color
        except ValueError:
            pass

    errfmt = ("{!r} is neither a data key nor a valid format string ({})"
              if ambiguous_fmt_datakey else
              "{!r} is not a valid format string ({})")

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(errfmt.format(fmt, "two marker symbols"))
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(errfmt.format(fmt, "two color symbols"))
            color = c
            i += 1
        elif c == "C":
            cn_color = re.match(r"C\d+", fmt[i:])
            if not cn_color:
                raise ValueError(errfmt.format(fmt, "'C' must be followed by a number"))
            color = mcolors.to_rgba(cn_color[0])
            i += len(cn_color[0])
        else:
            raise ValueError(errfmt.format(fmt, f"unrecognized character {c!r}"))

    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

    return linestyle, marker, color


class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """

    def __init__(self, output='Line2D'):
        _api.check_in_list(['Line2D', 'Polygon', 'coordinates'], output=output)
        self.output = output
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        if cycler is None:
            cycler = mpl.rcParams['axes.prop_cycle']
        self._idx = 0
        self._cycler_items = [*cycler]

    def __call__(self, axes, *args, data=None, return_kwargs=False, **kwargs):
        axes._process_unit_info(kwargs=kwargs)

        for pos_only in "xy":
            if pos_only in kwargs:
                raise _api.kwarg_error(inspect.stack()[1].function, pos_only)

        if not args:
            return

        if data is None:  # Process dict views
            args = [cbook.sanitize_sequence(a) for a in args]
        else:  # Process the 'data' kwarg.
            replaced = [mpl._replacer(data, arg) for arg in args]
            if len(args) == 1:
                label_namer_idx = 0
            elif len(args) == 2:  # Can be x, y or y, c.
                # Figure out what the second argument is.
                # 1) If the second argument cannot be a format shorthand, the
                #    second argument is the label_namer.
                # 2) Otherwise (it could have been a format shorthand),
                #    a) if we did perform a substitution, emit a warning, and
                #       use it as label_namer.
                #    b) otherwise, it is indeed a format shorthand; use the
                #       first argument as label_namer.
                try:
                    _process_plot_format(args[1])
                except ValueError:  # case 1)
                    label_namer_idx = 1
                else:
                    if replaced[1] is not args[1]:  # case 2a)
                        _api.warn_external(
                            f"Second argument {args[1]!r} is ambiguous: could "
                            f"be a format string but is in 'data'; using as "
                            f"data.  If it was intended as data, set the "
                            f"format string to an empty string to suppress "
                            f"this warning.  If it was intended as a format "
                            f"string, explicitly pass the x-values as well.  "
                            f"Alternatively, rename the entry in 'data'.",
                            RuntimeWarning)
                        label_namer_idx = 1
                    else:  # case 2b)
                        label_namer_idx = 0
            elif len(args) == 3:
                label_namer_idx = 1
            else:
                raise ValueError(
                    "Using arbitrary long args with data is not supported due "
                    "to ambiguity of arguments; use multiple plotting calls "
                    "instead")
            if kwargs.get("label") is None:
                kwargs["label"] = mpl._label_from_arg(
                    replaced[label_namer_idx], args[label_namer_idx])
            args = replaced
        ambiguous_fmt_datakey = data is not None and len(args) == 2

        if len(args) >= 4 and not cbook.is_scalar_or_string(
                kwargs.get("label")):
            raise ValueError("plot() with multiple groups of data (i.e., "
                             "pairs of x and y) does not support multiple "
                             "labels")

        # Repeatedly grab (x, y) or (x, y, format) from the front of args and
        # massage them into arguments to plot() or fill().

        while args:
            this, args = args[:2], args[2:]
            if args and isinstance(args[0], str):
                this += args[0],
                args = args[1:]
            yield from self._plot_args(
                axes, this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey,
                return_kwargs=return_kwargs
            )

    def get_next_color(self):
        """Return the next color in the cycle."""
        entry = self._cycler_items[self._idx]
        if "color" in entry:
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            return entry["color"]
        else:
            return "k"

    def _getdefaults(self, kw, ignore=frozenset()):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        defaults = self._cycler_items[self._idx]
        if any(kw.get(k, None) is None for k in {*defaults} - ignore):
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            # Return a new dict to avoid exposing _cycler_items entries to mutation.
            return {k: v for k, v in defaults.items() if k not in ignore}
        else:
            return {}

    def _setdefaults(self, defaults, kw):
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
        for k in defaults:
            if kw.get(k, None) is None:
                kw[k] = defaults[k]

    def _make_line(self, axes, x, y, kw, kwargs):
        kw = {**kw, **kwargs}  # Don't modify the original kw.
        self._setdefaults(self._getdefaults(kw), kw)
        seg = mlines.Line2D(x, y, **kw)
        return seg, kw

    def _make_coordinates(self, axes, x, y, kw, kwargs):
        kw = {**kw, **kwargs}  # Don't modify the original kw.
        self._setdefaults(self._getdefaults(kw), kw)
        return (x, y), kw

    def _make_polygon(self, axes, x, y, kw, kwargs):
        # Polygon doesn't directly support unitized inputs.
        x = axes.convert_xunits(x)
        y = axes.convert_yunits(y)

        kw = kw.copy()  # Don't modify the original kw.
        kwargs = kwargs.copy()

        # Ignore 'marker'-related properties as they aren't Polygon
        # properties, but they are Line2D properties, and so they are
        # likely to appear in the default cycler construction.
        # This is done here to the defaults dictionary as opposed to the
        # other two dictionaries because we do want to capture when a
        # *user* explicitly specifies a marker which should be an error.
        # We also want to prevent advancing the cycler if there are no
        # defaults needed after ignoring the given properties.
        ignores = ({'marker', 'markersize', 'markeredgecolor',
                    'markerfacecolor', 'markeredgewidth'}
                   # Also ignore anything provided by *kwargs*.
                   | {k for k, v in kwargs.items() if v is not None})

        # Only using the first dictionary to use as basis
        # for getting defaults for back-compat reasons.
        # Doing it with both seems to mess things up in
        # various places (probably due to logic bugs elsewhere).
        default_dict = self._getdefaults(kw, ignores)
        self._setdefaults(default_dict, kw)

        # Looks like we don't want "color" to be interpreted to
        # mean both facecolor and edgecolor for some reason.
        # So the "kw" dictionary is thrown out, and only its
        # 'color' value is kept and translated as a 'facecolor'.
        # This design should probably be revisited as it increases
        # complexity.
        facecolor = kw.get('color', None)

        # Throw out 'color' as it is now handled as a facecolor
        default_dict.pop('color', None)

        # To get other properties set from the cycler
        # modify the kwargs dictionary.
        self._setdefaults(default_dict, kwargs)

        seg = mpatches.Polygon(np.column_stack((x, y)),
                               facecolor=facecolor,
                               fill=kwargs.get('fill', True),
                               closed=kw['closed'])
        seg.set(**kwargs)
        return seg, kwargs

    def _plot_args(self, axes, tup, kwargs, *,
                   return_kwargs=False, ambiguous_fmt_datakey=False):
        """
        Process the arguments of ``plot([x], y, [fmt], **kwargs)`` calls.

        This processes a single set of ([x], y, [fmt]) parameters; i.e. for
        ``plot(x, y, x2, y2)`` it will be called twice. Once for (x, y) and
        once for (x2, y2).

        x and y may be 2D and thus can still represent multiple datasets.

        For multiple datasets, if the keyword argument *label* is a list, this
        will unpack the list and assign the individual labels to the datasets.

        Parameters
        ----------
        tup : tuple
            A tuple of the positional parameters. This can be one of

            - (y,)
            - (x, y)
            - (y, fmt)
            - (x, y, fmt)

        kwargs : dict
            The keyword arguments passed to ``plot()``.

        return_kwargs : bool
            Whether to also return the effective keyword arguments after label
            unpacking as well.

        ambiguous_fmt_datakey : bool
            Whether the format string in *tup* could also have been a
            misspelled data key.

        Returns
        -------
        result
            If *return_kwargs* is false, a list of Artists representing the
            dataset(s).
            If *return_kwargs* is true, a list of (Artist, effective_kwargs)
            representing the dataset(s). See *return_kwargs*.
            The Artist is either `.Line2D` (if called from ``plot()``) or
            `.Polygon` otherwise.
        """
        if len(tup) > 1 and isinstance(tup[-1], str):
            # xy is tup with fmt stripped (could still be (y,) only)
            *xy, fmt = tup
            linestyle, marker, color = _process_plot_format(
                fmt, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
        elif len(tup) == 3:
            raise ValueError('third arg must be a format string')
        else:
            xy = tup
            linestyle, marker, color = None, None, None

        # Don't allow any None value; these would be up-converted to one
        # element array of None which causes problems downstream.
        if any(v is None for v in tup):
            raise ValueError("x, y, and format string must not be None")

        kw = {}
        for prop_name, val in zip(('linestyle', 'marker', 'color'),
                                  (linestyle, marker, color)):
            if val is not None:
                # check for conflicts between fmt and kwargs
                if (fmt.lower() != 'none'
                        and prop_name in kwargs
                        and val != 'None'):
                    # Technically ``plot(x, y, 'o', ls='--')`` is a conflict
                    # because 'o' implicitly unsets the linestyle
                    # (linestyle='None').
                    # We'll gracefully not warn in this case because an
                    # explicit set via kwargs can be seen as intention to
                    # override an implicit unset.
                    # Note: We don't val.lower() != 'none' because val is not
                    # necessarily a string (can be a tuple for colors). This
                    # is safe, because *val* comes from _process_plot_format()
                    # which only returns 'None'.
                    _api.warn_external(
                        f"{prop_name} is redundantly defined by the "
                        f"'{prop_name}' keyword argument and the fmt string "
                        f'"{fmt}" (-> {prop_name}={val!r}). The keyword '
                        f"argument will take precedence.")
                kw[prop_name] = val

        if len(xy) == 2:
            x = _check_1d(xy[0])
            y = _check_1d(xy[1])
        else:
            x, y = index_of(xy[-1])

        if axes.xaxis is not None:
            axes.xaxis.update_units(x)
        if axes.yaxis is not None:
            axes.yaxis.update_units(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same first dimension, but "
                             f"have shapes {x.shape} and {y.shape}")
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError(f"x and y can be no greater than 2D, but have "
                             f"shapes {x.shape} and {y.shape}")
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.output == 'Line2D':
            make_artist = self._make_line
        elif self.output == 'Polygon':
            kw['closed'] = kwargs.get('closed', True)
            make_artist = self._make_polygon
        elif self.output == 'coordinates':
            make_artist = self._make_coordinates
        else:
            _api.check_in_list(['Line2D', 'Polygon', 'coordinates'], output=self.output)

        ncx, ncy = x.shape[1], y.shape[1]
        if ncx > 1 and ncy > 1 and ncx != ncy:
            raise ValueError(f"x has {ncx} columns but y has {ncy} columns")
        if ncx == 0 or ncy == 0:
            return []

        label = kwargs.get('label')
        n_datasets = max(ncx, ncy)

        if cbook.is_scalar_or_string(label):
            labels = [label] * n_datasets
        elif len(label) == n_datasets:
            labels = label
        elif n_datasets == 1:
            msg = (f'Passing label as a length {len(label)} sequence when '
                    'plotting a single dataset is deprecated in Matplotlib 3.9 '
                    'and will error in 3.11.  To keep the current behavior, '
                    'cast the sequence to string before passing.')
            _api.warn_deprecated('3.9', message=msg)
            labels = [label]
        else:
            raise ValueError(
                f"label must be scalar or have the same length as the input "
                f"data, but found {len(label)} for {n_datasets} datasets.")

        result = (make_artist(axes, x[:, j % ncx], y[:, j % ncy], kw,
                              {**kwargs, 'label': label})
                  for j, label in enumerate(labels))

        if return_kwargs:
            return list(result)
        else:
            return [l[0] for l in result]


@_api.define_aliases({"facecolor": ["fc"]})
class _AxesBase(martist.Artist):
    name = "rectilinear"

    # axis names are the prefixes for the attributes that contain the
    # respective axis; e.g. 'x' <-> self.xaxis, containing an XAxis.
    # Note that PolarAxes uses these attributes as well, so that we have
    # 'x' <-> self.xaxis, containing a ThetaAxis. In particular we do not
    # have 'theta' in _axis_names.
    # In practice, this is ('x', 'y') for all 2D Axes and ('x', 'y', 'z')
    # for Axes3D.
    _axis_names = ("x", "y")
    _shared_axes = {name: cbook.Grouper() for name in _axis_names}
    _twinned_axes = cbook.Grouper()

    _subclass_uses_cla = False

    dataLim: mtransforms.Bbox
    """The bounding `.Bbox` enclosing all data displayed in the Axes."""

    spines: mspines.Spines
    """
    The `.Spines` container for the Axes' spines, i.e. the lines denoting the
    data area boundaries.
    """

    xaxis: maxis.XAxis
    """
    The `.XAxis` instance.

    Common axis-related configuration can be achieved through high-level wrapper
    methods on Axes; e.g. `ax.set_xticks <.Axes.set_xticks>` is a shortcut for
    `ax.xaxis.set_ticks <.Axis.set_ticks>`. The *xaxis* attribute gives direct
    direct access to the underlying `~.axis.Axis` if you need more control.

    See also

    - :ref:`Axis wrapper methods on Axes <axes-api-axis>`
    - :doc:`Axis API </api/axis_api>`
    """

    yaxis: maxis.YAxis
    """
    The `.YAxis` instance.

    Common axis-related configuration can be achieved through high-level wrapper
    methods on Axes; e.g. `ax.set_yticks <.Axes.set_yticks>` is a shortcut for
    `ax.yaxis.set_ticks <.Axis.set_ticks>`. The *yaxis* attribute gives direct
    access to the underlying `~.axis.Axis` if you need more control.

    See also

    - :ref:`Axis wrapper methods on Axes <axes-api-axis>`
    - :doc:`Axis API </api/axis_api>`
    """

    @property
    def _axis_map(self):
        """A mapping of axis names, e.g. 'x', to `Axis` instances."""
        return {name: getattr(self, f"{name}axis")
                for name in self._axis_names}

    def __str__(self):
        return "{0}({1[0]:g},{1[1]:g};{1[2]:g}x{1[3]:g})".format(
            type(self).__name__, self._position.bounds)

    def __init__(self, fig,
                 *args,
                 facecolor=None,  # defaults to rc axes.facecolor
                 frameon=True,
                 sharex=None,  # use Axes instance's xaxis info
                 sharey=None,  # use Axes instance's yaxis info
                 label='',
                 xscale=None,
                 yscale=None,
                 box_aspect=None,
                 forward_navigation_events="auto",
                 **kwargs
                 ):
        """
        Build an Axes in a figure.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The Axes is built in the `.Figure` *fig*.

        *args
            ``*args`` can be a single ``(left, bottom, width, height)``
            rectangle or a single `.Bbox`.  This specifies the rectangle (in
            figure coordinates) where the Axes is positioned.

            ``*args`` can also consist of three numbers or a single three-digit
            number; in the latter case, the digits are considered as
            independent numbers.  The numbers are interpreted as ``(nrows,
            ncols, index)``: ``(nrows, ncols)`` specifies the size of an array
            of subplots, and ``index`` is the 1-based index of the subplot
            being created.  Finally, ``*args`` can also directly be a
            `.SubplotSpec` instance.

        sharex, sharey : `~matplotlib.axes.Axes`, optional
            The x- or y-`~.matplotlib.axis` is shared with the x- or y-axis in
            the input `~.axes.Axes`.  Note that it is not possible to unshare
            axes.

        frameon : bool, default: True
            Whether the Axes frame is visible.

        box_aspect : float, optional
            Set a fixed aspect for the Axes box, i.e. the ratio of height to
            width. See `~.axes.Axes.set_box_aspect` for details.

        forward_navigation_events : bool or "auto", default: "auto"
            Control whether pan/zoom events are passed through to Axes below
            this one. "auto" is *True* for axes with an invisible patch and
            *False* otherwise.

        **kwargs
            Other optional keyword arguments:

            %(Axes:kwdoc)s

        Returns
        -------
        `~.axes.Axes`
            The new `~.axes.Axes` object.
        """

        super().__init__()
        if "rect" in kwargs:
            if args:
                raise TypeError(
                    "'rect' cannot be used together with positional arguments")
            rect = kwargs.pop("rect")
            _api.check_isinstance((mtransforms.Bbox, Iterable), rect=rect)
            args = (rect,)
        subplotspec = None
        if len(args) == 1 and isinstance(args[0], mtransforms.Bbox):
            self._position = args[0].frozen()
        elif len(args) == 1 and np.iterable(args[0]):
            self._position = mtransforms.Bbox.from_bounds(*args[0])
        else:
            self._position = self._originalPosition = mtransforms.Bbox.unit()
            subplotspec = SubplotSpec._from_subplot_args(fig, args)
        if self._position.width < 0 or self._position.height < 0:
            raise ValueError('Width and height specified must be non-negative')
        self._originalPosition = self._position.frozen()
        self.axes = self
        self._aspect = 'auto'
        self._adjustable = 'box'
        self._anchor = 'C'
        self._stale_viewlims = dict.fromkeys(self._axis_names, False)
        self._forward_navigation_events = forward_navigation_events
        self._sharex = sharex
        self._sharey = sharey
        self.set_label(label)
        self.set_figure(fig)
        # The subplotspec needs to be set after the figure (so that
        # figure-level subplotpars are taken into account), but the figure
        # needs to be set after self._position is initialized.
        if subplotspec:
            self.set_subplotspec(subplotspec)
        else:
            self._subplotspec = None
        self.set_box_aspect(box_aspect)
        self._axes_locator = None  # Optionally set via update(kwargs).

        self._children = []

        # placeholder for any colorbars added that use this Axes.
        # (see colorbar.py):
        self._colorbars = []
        self.spines = mspines.Spines.from_dict(self._gen_axes_spines())

        # this call may differ for non-sep axes, e.g., polar
        self._init_axis()
        if facecolor is None:
            facecolor = mpl.rcParams['axes.facecolor']
        self._facecolor = facecolor
        self._frameon = frameon
        self.set_axisbelow(mpl.rcParams['axes.axisbelow'])

        self._rasterization_zorder = None
        self.clear()

        # funcs used to format x and y - fall back on major formatters
        self.fmt_xdata = None
        self.fmt_ydata = None

        self.set_navigate(True)
        self.set_navigate_mode(None)

        if xscale:
            self.set_xscale(xscale)
        if yscale:
            self.set_yscale(yscale)

        self._internal_update(kwargs)

        for name, axis in self._axis_map.items():
            axis.callbacks._connect_picklable(
                'units', self._unit_change_handler(name))

        rcParams = mpl.rcParams
        self.tick_params(
            top=rcParams['xtick.top'] and rcParams['xtick.minor.top'],
            bottom=rcParams['xtick.bottom'] and rcParams['xtick.minor.bottom'],
            labeltop=(rcParams['xtick.labeltop'] and
                      rcParams['xtick.minor.top']),
            labelbottom=(rcParams['xtick.labelbottom'] and
                         rcParams['xtick.minor.bottom']),
            left=rcParams['ytick.left'] and rcParams['ytick.minor.left'],
            right=rcParams['ytick.right'] and rcParams['ytick.minor.right'],
            labelleft=(rcParams['ytick.labelleft'] and
                       rcParams['ytick.minor.left']),
            labelright=(rcParams['ytick.labelright'] and
                        rcParams['ytick.minor.right']),
            which='minor')

        self.tick_params(
            top=rcParams['xtick.top'] and rcParams['xtick.major.top'],
            bottom=rcParams['xtick.bottom'] and rcParams['xtick.major.bottom'],
            labeltop=(rcParams['xtick.labeltop'] and
                      rcParams['xtick.major.top']),
            labelbottom=(rcParams['xtick.labelbottom'] and
                         rcParams['xtick.major.bottom']),
            left=rcParams['ytick.left'] and rcParams['ytick.major.left'],
            right=rcParams['ytick.right'] and rcParams['ytick.major.right'],
            labelleft=(rcParams['ytick.labelleft'] and
                       rcParams['ytick.major.left']),
            labelright=(rcParams['ytick.labelright'] and
                        rcParams['ytick.major.right']),
            which='major')

    def __init_subclass__(cls, **kwargs):
        parent_uses_cla = super(cls, cls)._subclass_uses_cla
        if 'cla' in cls.__dict__:
            _api.warn_deprecated(
                '3.6',
                pending=True,
                message=f'Overriding `Axes.cla` in {cls.__qualname__} is '
                'pending deprecation in %(since)s and will be fully '
                'deprecated in favor of `Axes.clear` in the future. '
                'Please report '
                f'this to the {cls.__module__!r} author.')
        cls._subclass_uses_cla = 'cla' in cls.__dict__ or parent_uses_cla
        super().__init_subclass__(**kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        # Prune the sharing & twinning info to only contain the current group.
        state["_shared_axes"] = {
            name: self._shared_axes[name].get_siblings(self)
            for name in self._axis_names if self in self._shared_axes[name]}
        state["_twinned_axes"] = (self._twinned_axes.get_siblings(self)
                                  if self in self._twinned_axes else None)
        return state

    def __setstate__(self, state):
        # Merge the grouping info back into the global groupers.
        shared_axes = state.pop("_shared_axes")
        for name, shared_siblings in shared_axes.items():
            self._shared_axes[name].join(*shared_siblings)
        twinned_siblings = state.pop("_twinned_axes")
        if twinned_siblings:
            self._twinned_axes.join(*twinned_siblings)
        self.__dict__ = state
        self._stale = True

    def __repr__(self):
        fields = []
        if self.get_label():
            fields += [f"label={self.get_label()!r}"]
        if hasattr(self, "get_title"):
            titles = {}
            for k in ["left", "center", "right"]:
                title = self.get_title(loc=k)
                if title:
                    titles[k] = title
            if titles:
                fields += [f"title={titles}"]
        for name, axis in self._axis_map.items():
            if axis.label and axis.label.get_text():
                fields += [f"{name}label={axis.label.get_text()!r}"]
        return f"<{self.__class__.__name__}: " + ", ".join(fields) + ">"

    def get_subplotspec(self):
        """Return the `.SubplotSpec` associated with the subplot, or None."""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the `.SubplotSpec`. associated with the subplot."""
        self._subplotspec = subplotspec
        self._set_position(subplotspec.get_position(self.get_figure(root=False)))

    def get_gridspec(self):
        """Return the `.GridSpec` associated with the subplot, or None."""
        return self._subplotspec.get_gridspec() if self._subplotspec else None

    def get_window_extent(self, renderer=None):
        """
        Return the Axes bounding box in display space.

        This bounding box does not include the spines, ticks, ticklabels,
        or other labels.  For a bounding box including these elements use
        `~matplotlib.axes.Axes.get_tightbbox`.

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axis.Axis.get_tightbbox
        matplotlib.spines.Spine.get_window_extent
        """
        return self.bbox

    def _init_axis(self):
        # This is moved out of __init__ because non-separable axes don't use it
        self.xaxis = maxis.XAxis(self, clear=False)
        self.spines.bottom.register_axis(self.xaxis)
        self.spines.top.register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self, clear=False)
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    def set_figure(self, fig):
        # docstring inherited
        super().set_figure(fig)

        self.bbox = mtransforms.TransformedBbox(self._position,
                                                fig.transSubfigure)
        # these will be updated later as data is added
        self.dataLim = mtransforms.Bbox.null()
        self._viewLim = mtransforms.Bbox.unit()
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        self._set_lim_and_transforms()

    def _unstale_viewLim(self):
        # We should arrange to store this information once per share-group
        # instead of on every axis.
        need_scale = {
            name: any(ax._stale_viewlims[name]
                      for ax in self._shared_axes[name].get_siblings(self))
            for name in self._axis_names}
        if any(need_scale.values()):
            for name in need_scale:
                for ax in self._shared_axes[name].get_siblings(self):
                    ax._stale_viewlims[name] = False
            self.autoscale_view(**{f"scale{name}": scale
                                   for name, scale in need_scale.items()})

    @property
    def viewLim(self):
        """The view limits as `.Bbox` in data coordinates."""
        self._unstale_viewLim()
        return self._viewLim

    def _request_autoscale_view(self, axis="all", tight=None):
        """
        Mark a single axis, or all of them, as stale wrt. autoscaling.

        No computation is performed until the next autoscaling; thus, separate
        calls to control individual axises incur negligible performance cost.

        Parameters
        ----------
        axis : str, default: "all"
            Either an element of ``self._axis_names``, or "all".
        tight : bool or None, default: None
        """
        axis_names = _api.check_getitem(
            {**{k: [k] for k in self._axis_names}, "all": self._axis_names},
            axis=axis)
        for name in axis_names:
            self._stale_viewlims[name] = True
        if tight is not None:
            self._tight = tight

    def _set_lim_and_transforms(self):
        """
        Set the *_xaxis_transform*, *_yaxis_transform*, *transScale*,
        *transData*, *transLimits* and *transAxes* transformations.

        .. note::

            This method is primarily used by rectilinear projections of the
            `~matplotlib.axes.Axes` class, and is meant to be overridden by
            new kinds of projection Axes that need different transformations
            and limits. (See `~matplotlib.projections.polar.PolarAxes` for an
            example.)
        """
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # Transforms the x and y axis separately by a scale factor.
        # It is assumed that this part will have non-linear components
        # (e.g., for a log scale).
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self._viewLim, self.transScale))

        # The parentheses are important for efficiency here -- they
        # group the last two (which are usually affines) separately
        # from the first (which, with log-scaling can be non-affine).
        self.transData = self.transScale + (self.transLimits + self.transAxes)

        self._xaxis_transform = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

    def get_xaxis_transform(self, which='grid'):
        """
        Get the transformation used for drawing x-axis labels, ticks
        and gridlines.  The x-direction is in data coordinates and the
        y-direction is in axis coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
        """
        if which == 'grid':
            return self._xaxis_transform
        elif which == 'tick1':
            # for cartesian projection, this is bottom spine
            return self.spines.bottom.get_spine_transform()
        elif which == 'tick2':
            # for cartesian projection, this is top spine
            return self.spines.top.get_spine_transform()
        else:
            raise ValueError(f'unknown value for which: {which!r}')

    def get_xaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing x-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in data coordinates and the y-direction is in
            axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        labels_align = mpl.rcParams["xtick.alignment"]
        return (self.get_xaxis_transform(which='tick1') +
                mtransforms.ScaledTranslation(
                    0, -1 * pad_points / 72,
                    self.get_figure(root=False).dpi_scale_trans),
                "top", labels_align)

    def get_xaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing secondary x-axis labels, which will
            add *pad_points* of padding (in points) between the axis and the
            label.  The x-direction is in data coordinates and the y-direction
            is in axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        labels_align = mpl.rcParams["xtick.alignment"]
        return (self.get_xaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(
                    0, pad_points / 72,
                    self.get_figure(root=False).dpi_scale_trans),
                "bottom", labels_align)

    def get_yaxis_transform(self, which='grid'):
        """
        Get the transformation used for drawing y-axis labels, ticks
        and gridlines.  The x-direction is in axis coordinates and the
        y-direction is in data coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
        """
        if which == 'grid':
            return self._yaxis_transform
        elif which == 'tick1':
            # for cartesian projection, this is bottom spine
            return self.spines.left.get_spine_transform()
        elif which == 'tick2':
            # for cartesian projection, this is top spine
            return self.spines.right.get_spine_transform()
        else:
            raise ValueError(f'unknown value for which: {which!r}')

    def get_yaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing y-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in axis coordinates and the y-direction is in
            data coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        labels_align = mpl.rcParams["ytick.alignment"]
        return (self.get_yaxis_transform(which='tick1') +
                mtransforms.ScaledTranslation(
                    -1 * pad_points / 72, 0,
                    self.get_figure(root=False).dpi_scale_trans),
                labels_align, "right")

    def get_yaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing secondart y-axis labels, which will
            add *pad_points* of padding (in points) between the axis and the
            label.  The x-direction is in axis coordinates and the y-direction
            is in data coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        labels_align = mpl.rcParams["ytick.alignment"]
        return (self.get_yaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(
                    pad_points / 72, 0,
                    self.get_figure(root=False).dpi_scale_trans),
                labels_align, "left")

    def _update_transScale(self):
        self.transScale.set(
            mtransforms.blended_transform_factory(
                self.xaxis.get_transform(), self.yaxis.get_transform()))

    def get_position(self, original=False):
        """
        Return the position of the Axes within the figure as a `.Bbox`.

        Parameters
        ----------
        original : bool
            If ``True``, return the original position. Otherwise, return the
            active position. For an explanation of the positions see
            `.set_position`.

        Returns
        -------
        `.Bbox`

        """
        if original:
            return self._originalPosition.frozen()
        else:
            locator = self.get_axes_locator()
            if not locator:
                self.apply_aspect()
            return self._position.frozen()

    def set_position(self, pos, which='both'):
        """
        Set the Axes position.

        Axes have two position attributes. The 'original' position is the
        position allocated for the Axes. The 'active' position is the
        position the Axes is actually drawn at. These positions are usually
        the same unless a fixed aspect is set to the Axes. See
        `.Axes.set_aspect` for details.

        Parameters
        ----------
        pos : [left, bottom, width, height] or `~matplotlib.transforms.Bbox`
            The new position of the Axes in `.Figure` coordinates.

        which : {'both', 'active', 'original'}, default: 'both'
            Determines which position variables to change.

        See Also
        --------
        matplotlib.transforms.Bbox.from_bounds
        matplotlib.transforms.Bbox.from_extents
        """
        self._set_position(pos, which=which)
        # because this is being called externally to the library we
        # don't let it be in the layout.
        self.set_in_layout(False)

    def _set_position(self, pos, which='both'):
        """
        Private version of set_position.

        Call this internally to get the same functionality of `set_position`,
        but not to take the axis out of the constrained_layout hierarchy.
        """
        if not isinstance(pos, mtransforms.BboxBase):
            pos = mtransforms.Bbox.from_bounds(*pos)
        for ax in self._twinned_axes.get_siblings(self):
            if which in ('both', 'active'):
                ax._position.set(pos)
            if which in ('both', 'original'):
                ax._originalPosition.set(pos)
        self.stale = True

    def reset_position(self):
        """
        Reset the active position to the original position.

        This undoes changes to the active position (as defined in
        `.set_position`) which may have been performed to satisfy fixed-aspect
        constraints.
        """
        for ax in self._twinned_axes.get_siblings(self):
            pos = ax.get_position(original=True)
            ax.set_position(pos, which='active')

    def set_axes_locator(self, locator):
        """
        Set the Axes locator.

        Parameters
        ----------
        locator : Callable[[Axes, Renderer], Bbox]
        """
        self._axes_locator = locator
        self.stale = True

    def get_axes_locator(self):
        """
        Return the axes_locator.
        """
        return self._axes_locator

    def _set_artist_props(self, a):
        """Set the boilerplate props for artists added to Axes."""
        a.set_figure(self.get_figure(root=False))
        if not a.is_transform_set():
            a.set_transform(self.transData)

        a.axes = self
        if a.get_mouseover():
            self._mouseover_set.add(a)

    def _gen_axes_patch(self):
        """
        Returns
        -------
        Patch
            The patch used to draw the background of the Axes.  It is also used
            as the clipping path for any data elements on the Axes.

            In the standard Axes, this is a rectangle, but in other projections
            it may not be.

        Notes
        -----
        Intended to be overridden by new projection types.
        """
        return mpatches.Rectangle((0.0, 0.0), 1.0, 1.0)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        """
        Returns
        -------
        dict
            Mapping of spine names to `.Line2D` or `.Patch` instances that are
            used to draw Axes spines.

            In the standard Axes, spines are single line segments, but in other
            projections they may not be.

        Notes
        -----
        Intended to be overridden by new projection types.
        """
        return {side: mspines.Spine.linear_spine(self, side)
                for side in ['left', 'right', 'bottom', 'top']}

    def sharex(self, other):
        """
        Share the x-axis with *other*.

        This is equivalent to passing ``sharex=other`` when constructing the
        Axes, and cannot be used if the x-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(_AxesBase, other=other)
        if self._sharex is not None and other is not self._sharex:
            raise ValueError("x-axis is already shared")
        self._shared_axes["x"].join(self, other)
        self._sharex = other
        self.xaxis.major = other.xaxis.major  # Ticker instances holding
        self.xaxis.minor = other.xaxis.minor  # locator and formatter.
        x0, x1 = other.get_xlim()
        self.set_xlim(x0, x1, emit=False, auto=other.get_autoscalex_on())
        self.xaxis._scale = other.xaxis._scale

    def sharey(self, other):
        """
        Share the y-axis with *other*.

        This is equivalent to passing ``sharey=other`` when constructing the
        Axes, and cannot be used if the y-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(_AxesBase, other=other)
        if self._sharey is not None and other is not self._sharey:
            raise ValueError("y-axis is already shared")
        self._shared_axes["y"].join(self, other)
        self._sharey = other
        self.yaxis.major = other.yaxis.major  # Ticker instances holding
        self.yaxis.minor = other.yaxis.minor  # locator and formatter.
        y0, y1 = other.get_ylim()
        self.set_ylim(y0, y1, emit=False, auto=other.get_autoscaley_on())
        self.yaxis._scale = other.yaxis._scale

    def __clear(self):
        """Clear the Axes."""
        # The actual implementation of clear() as long as clear() has to be
        # an adapter delegating to the correct implementation.
        # The implementation can move back into clear() when the
        # deprecation on cla() subclassing expires.

        # stash the current visibility state
        if hasattr(self, 'patch'):
            patch_visible = self.patch.get_visible()
        else:
            patch_visible = True

        xaxis_visible = self.xaxis.get_visible()
        yaxis_visible = self.yaxis.get_visible()

        for axis in self._axis_map.values():
            axis.clear()  # Also resets the scale to linear.
        for spine in self.spines.values():
            spine._clear()  # Use _clear to not clear Axis again

        self.ignore_existing_data_limits = True
        self.callbacks = cbook.CallbackRegistry(
            signals=["xlim_changed", "ylim_changed", "zlim_changed"])

        # update the minor locator for x and y axis based on rcParams
        if mpl.rcParams['xtick.minor.visible']:
            self.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if mpl.rcParams['ytick.minor.visible']:
            self.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        self._xmargin = mpl.rcParams['axes.xmargin']
        self._ymargin = mpl.rcParams['axes.ymargin']
        self._tight = None
        self._use_sticky_edges = True

        self._get_lines = _process_plot_var_args()
        self._get_patches_for_fill = _process_plot_var_args('Polygon')

        self._gridOn = mpl.rcParams['axes.grid']
        # Swap children to minimize time we spend in an invalid state
        old_children, self._children = self._children, []
        for chld in old_children:
            chld._remove_method = None
            chld._parent_figure = None
            chld.axes = None
        # Use list.clear to break the `artist._remove_method` reference cycle
        old_children.clear()
        self._mouseover_set = _OrderedSet()
        self.child_axes = []
        self._current_image = None  # strictly for pyplot via _sci, _gci
        self._projection_init = None  # strictly for pyplot.subplot
        self.legend_ = None
        self.containers = []

        self.grid(False)  # Disable grid on init to use rcParameter
        self.grid(self._gridOn, which=mpl.rcParams['axes.grid.which'],
                  axis=mpl.rcParams['axes.grid.axis'])
        props = font_manager.FontProperties(
            size=mpl.rcParams['axes.titlesize'],
            weight=mpl.rcParams['axes.titleweight'])

        y = mpl.rcParams['axes.titley']
        if y is None:
            y = 1.0
            self._autotitlepos = True
        else:
            self._autotitlepos = False

        self.title = mtext.Text(
            x=0.5, y=y, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='center',
            )
        self._left_title = mtext.Text(
            x=0.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='left', )
        self._right_title = mtext.Text(
            x=1.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='right',
            )
        title_offset_points = mpl.rcParams['axes.titlepad']
        # refactor this out so it can be called in ax.set_title if
        # pad argument used...
        self._set_title_offset_trans(title_offset_points)

        for _title in (self.title, self._left_title, self._right_title):
            self._set_artist_props(_title)

        # The patch draws the background of the Axes.  We want this to be below
        # the other artists.  We use the frame to draw the edges so we are
        # setting the edgecolor to None.
        self.patch = self._gen_axes_patch()
        self.patch.set_figure(self.get_figure(root=False))
        self.patch.set_facecolor(self._facecolor)
        self.patch.set_edgecolor('none')
        self.patch.set_linewidth(0)
        self.patch.set_transform(self.transAxes)

        self.set_axis_on()

        self.xaxis.set_clip_path(self.patch)
        self.yaxis.set_clip_path(self.patch)

        if self._sharex is not None:
            self.xaxis.set_visible(xaxis_visible)
            self.patch.set_visible(patch_visible)
        if self._sharey is not None:
            self.yaxis.set_visible(yaxis_visible)
            self.patch.set_visible(patch_visible)

        # This comes last, as the call to _set_lim may trigger an autoscale (in
        # case of shared axes), requiring children to be already set up.
        for name, axis in self._axis_map.items():
            share = getattr(self, f"_share{name}")
            if share is not None:
                getattr(self, f"share{name}")(share)
            else:
                # Although the scale was set to linear as part of clear,
                # polar requires that _set_scale is called again
                if self.name == "polar":
                    axis._set_scale("linear")
                axis._set_lim(0, 1, auto=True)
        self._update_transScale()

        self.stale = True

    def clear(self):
        """Clear the Axes."""
        # Act as an alias, or as the superclass implementation depending on the
        # subclass implementation.
        if self._subclass_uses_cla:
            self.cla()
        else:
            self.__clear()

    def cla(self):
        """Clear the Axes."""
        # Act as an alias, or as the superclass implementation depending on the
        # subclass implementation.
        if self._subclass_uses_cla:
            self.__clear()
        else:
            self.clear()

    class ArtistList(Sequence):
        """
        A sublist of Axes children based on their type.

        The type-specific children sublists were made immutable in Matplotlib
        3.7.  In the future these artist lists may be replaced by tuples. Use
        as if this is a tuple already.
        """
        def __init__(self, axes, prop_name,
                     valid_types=None, invalid_types=None):
            """
            Parameters
            ----------
            axes : `~matplotlib.axes.Axes`
                The Axes from which this sublist will pull the children
                Artists.
            prop_name : str
                The property name used to access this sublist from the Axes;
                used to generate deprecation warnings.
            valid_types : list of type, optional
                A list of types that determine which children will be returned
                by this sublist. If specified, then the Artists in the sublist
                must be instances of any of these types. If unspecified, then
                any type of Artist is valid (unless limited by
                *invalid_types*.)
            invalid_types : tuple, optional
                A list of types that determine which children will *not* be
                returned by this sublist. If specified, then Artists in the
                sublist will never be an instance of these types. Otherwise, no
                types will be excluded.
            """
            self._axes = axes
            self._prop_name = prop_name
            self._type_check = lambda artist: (
                (not valid_types or isinstance(artist, valid_types)) and
                (not invalid_types or not isinstance(artist, invalid_types))
            )

        def __repr__(self):
            return f'<Axes.ArtistList of {len(self)} {self._prop_name}>'

        def __len__(self):
            return sum(self._type_check(artist)
                       for artist in self._axes._children)

        def __iter__(self):
            for artist in list(self._axes._children):
                if self._type_check(artist):
                    yield artist

        def __getitem__(self, key):
            return [artist
                    for artist in self._axes._children
                    if self._type_check(artist)][key]

        def __add__(self, other):
            if isinstance(other, (list, _AxesBase.ArtistList)):
                return [*self, *other]
            if isinstance(other, (tuple, _AxesBase.ArtistList)):
                return (*self, *other)
            return NotImplemented

        def __radd__(self, other):
            if isinstance(other, list):
                return other + list(self)
            if isinstance(other, tuple):
                return other + tuple(self)
            return NotImplemented

    @property
    def artists(self):
        return self.ArtistList(self, 'artists', invalid_types=(
            mcoll.Collection, mimage.AxesImage, mlines.Line2D, mpatches.Patch,
            mtable.Table, mtext.Text))

    @property
    def collections(self):
        return self.ArtistList(self, 'collections',
                               valid_types=mcoll.Collection)

    @property
    def images(self):
        return self.ArtistList(self, 'images', valid_types=mimage.AxesImage)

    @property
    def lines(self):
        return self.ArtistList(self, 'lines', valid_types=mlines.Line2D)

    @property
    def patches(self):
        return self.ArtistList(self, 'patches', valid_types=mpatches.Patch)

    @property
    def tables(self):
        return self.ArtistList(self, 'tables', valid_types=mtable.Table)

    @property
    def texts(self):
        return self.ArtistList(self, 'texts', valid_types=mtext.Text)

    def get_facecolor(self):
        """Get the facecolor of the Axes."""
        return self.patch.get_facecolor()

    def set_facecolor(self, color):
        """
        Set the facecolor of the Axes.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        self._facecolor = color
        self.stale = True
        return self.patch.set_facecolor(color)

    def _set_title_offset_trans(self, title_offset_points):
        """
        Set the offset for the title either from :rc:`axes.titlepad`
        or from set_title kwarg ``pad``.
        """
        self.titleOffsetTrans = mtransforms.ScaledTranslation(
                0.0, title_offset_points / 72,
                self.get_figure(root=False).dpi_scale_trans)
        for _title in (self.title, self._left_title, self._right_title):
            _title.set_transform(self.transAxes + self.titleOffsetTrans)
            _title.set_clip_box(None)

    def set_prop_cycle(self, *args, **kwargs):
        """
        Set the property cycle of the Axes.

        The property cycle controls the style properties such as color,
        marker and linestyle of future plot commands. The style properties
        of data already added to the Axes are not modified.

        Call signatures::

          set_prop_cycle(cycler)
          set_prop_cycle(label=values, label2=values2, ...)
          set_prop_cycle(label, values)

        Form 1 sets given `~cycler.Cycler` object.

        Form 2 creates a `~cycler.Cycler` which cycles over one or more
        properties simultaneously and set it as the property cycle of the
        Axes. If multiple properties are given, their value lists must have
        the same length. This is just a shortcut for explicitly creating a
        cycler and passing it to the function, i.e. it's short for
        ``set_prop_cycle(cycler(label=values, label2=values2, ...))``.

        Form 3 creates a `~cycler.Cycler` for a single property and set it
        as the property cycle of the Axes. This form exists for compatibility
        with the original `cycler.cycler` interface. Its use is discouraged
        in favor of the kwarg form, i.e. ``set_prop_cycle(label=values)``.

        Parameters
        ----------
        cycler : `~cycler.Cycler` or ``None``
            Set the given Cycler. *None* resets to the cycle defined by the
            current style.

            .. ACCEPTS: `~cycler.Cycler`

        label : str
            The property key. Must be a valid `.Artist` property.
            For example, 'color' or 'linestyle'. Aliases are allowed,
            such as 'c' for 'color' and 'lw' for 'linewidth'.

        values : iterable
            Finite-length iterable of the property values. These values
            are validated and will raise a ValueError if invalid.

        See Also
        --------
        matplotlib.rcsetup.cycler
            Convenience function for creating validated cyclers for properties.
        cycler.cycler
            The original function for creating unvalidated cyclers.

        Examples
        --------
        Setting the property cycle for a single property:

        >>> ax.set_prop_cycle(color=['red', 'green', 'blue'])

        Setting the property cycle for simultaneously cycling over multiple
        properties (e.g. red circle, green plus, blue cross):

        >>> ax.set_prop_cycle(color=['red', 'green', 'blue'],
        ...                   marker=['o', '+', 'x'])

        """
        if args and kwargs:
            raise TypeError("Cannot supply both positional and keyword "
                            "arguments to this method.")
        # Can't do `args == (None,)` as that crashes cycler.
        if len(args) == 1 and args[0] is None:
            prop_cycle = None
        else:
            prop_cycle = cycler(*args, **kwargs)
        self._get_lines.set_prop_cycle(prop_cycle)
        self._get_patches_for_fill.set_prop_cycle(prop_cycle)

    def get_aspect(self):
        """
        Return the aspect ratio of the Axes scaling.

        This is either "auto" or a float giving the ratio of y/x-scale.
        """
        return self._aspect

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """
        Set the aspect ratio of the Axes scaling, i.e. y/x-scale.

        Parameters
        ----------
        aspect : {'auto', 'equal'} or float
            Possible values:

            - 'auto': fill the position rectangle with data.
            - 'equal': same as ``aspect=1``, i.e. same scaling for x and y.
            - *float*: The displayed size of 1 unit in y-data coordinates will
              be *aspect* times the displayed size of 1 unit in x-data
              coordinates; e.g. for ``aspect=2`` a square in data coordinates
              will be rendered with a height of twice its width.

        adjustable : None or {'box', 'datalim'}, optional
            If not ``None``, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or (float, float), optional
            If not ``None``, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way
            to specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_anchor
            Set the position in case of extra space.
        """
        if cbook._str_equal(aspect, 'equal'):
            aspect = 1
        if not cbook._str_equal(aspect, 'auto'):
            aspect = float(aspect)  # raise ValueError if necessary
            if aspect <= 0 or not np.isfinite(aspect):
                raise ValueError("aspect must be finite and positive ")

        if share:
            axes = {sibling for name in self._axis_names
                    for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            axes = [self]

        for ax in axes:
            ax._aspect = aspect

        if adjustable is None:
            adjustable = self._adjustable
        self.set_adjustable(adjustable, share=share)  # Handle sharing.

        if anchor is not None:
            self.set_anchor(anchor, share=share)
        self.stale = True

    def get_adjustable(self):
        """
        Return whether the Axes will adjust its physical dimension ('box') or
        its data limits ('datalim') to achieve the desired aspect ratio.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.
        """
        return self._adjustable

    def set_adjustable(self, adjustable, share=False):
        """
        Set how the Axes adjusts to achieve the required aspect ratio.

        Parameters
        ----------
        adjustable : {'box', 'datalim'}
            If 'box', change the physical dimensions of the Axes.
            If 'datalim', change the ``x`` or ``y`` data limits. This
            may ignore explicitly defined axis limits.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.

        Notes
        -----
        Shared Axes (of which twinned Axes are a special case)
        impose restrictions on how aspect ratios can be imposed.
        For twinned Axes, use 'datalim'.  For Axes that share both
        x and y, use 'box'.  Otherwise, either 'datalim' or 'box'
        may be used.  These limitations are partly a requirement
        to avoid over-specification, and partly a result of the
        particular implementation we are currently using, in
        which the adjustments for aspect ratios are done sequentially
        and independently on each Axes as it is drawn.
        """
        _api.check_in_list(["box", "datalim"], adjustable=adjustable)
        if share:
            axs = {sibling for name in self._axis_names
                   for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            axs = [self]
        if (adjustable == "datalim"
                and any(getattr(ax.get_data_ratio, "__func__", None)
                        != _AxesBase.get_data_ratio
                        for ax in axs)):
            # Limits adjustment by apply_aspect assumes that the axes' aspect
            # ratio can be computed from the data limits and scales.
            raise ValueError("Cannot set Axes adjustable to 'datalim' for "
                             "Axes which override 'get_data_ratio'")
        for ax in axs:
            ax._adjustable = adjustable
        self.stale = True

    def get_box_aspect(self):
        """
        Return the Axes box aspect, i.e. the ratio of height to width.

        The box aspect is ``None`` (i.e. chosen depending on the available
        figure space) unless explicitly specified.

        See Also
        --------
        matplotlib.axes.Axes.set_box_aspect
            for a description of box aspect.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        return self._box_aspect

    def set_box_aspect(self, aspect=None):
        """
        Set the Axes box aspect, i.e. the ratio of height to width.

        This defines the aspect of the Axes in figure space and is not to be
        confused with the data aspect (see `~.Axes.set_aspect`).

        Parameters
        ----------
        aspect : float or None
            Changes the physical dimensions of the Axes, such that the ratio
            of the Axes height to the Axes width in physical units is equal to
            *aspect*. Defining a box aspect will change the *adjustable*
            property to 'datalim' (see `~.Axes.set_adjustable`).

            *None* will disable a fixed box aspect so that height and width
            of the Axes are chosen independently.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        axs = {*self._twinned_axes.get_siblings(self),
               *self._twinned_axes.get_siblings(self)}

        if aspect is not None:
            aspect = float(aspect)
            # when box_aspect is set to other than ´None`,
            # adjustable must be "datalim"
            for ax in axs:
                ax.set_adjustable("datalim")

        for ax in axs:
            ax._box_aspect = aspect
            ax.stale = True

    def get_anchor(self):
        """
        Get the anchor location.

        See Also
        --------
        matplotlib.axes.Axes.set_anchor
            for a description of the anchor.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        return self._anchor

    def set_anchor(self, anchor, share=False):
        """
        Define the anchor location.

        The actual drawing area (active position) of the Axes may be smaller
        than the Bbox (original position) when a fixed aspect is required. The
        anchor defines where the drawing area will be located within the
        available space.

        Parameters
        ----------
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).  str inputs are shorthands
            for (*x*, *y*) coordinates, as shown in the following diagram::

               ┌─────────────────┬─────────────────┬─────────────────┐
               │ 'NW' (0.0, 1.0) │ 'N' (0.5, 1.0)  │ 'NE' (1.0, 1.0) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'W'  (0.0, 0.5) │ 'C' (0.5, 0.5)  │ 'E'  (1.0, 0.5) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'SW' (0.0, 0.0) │ 'S' (0.5, 0.0)  │ 'SE' (1.0, 0.0) │
               └─────────────────┴─────────────────┴─────────────────┘

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        if not (anchor in mtransforms.Bbox.coefs or len(anchor) == 2):
            raise ValueError('argument must be among %s' %
                             ', '.join(mtransforms.Bbox.coefs))
        if share:
            axes = {sibling for name in self._axis_names
                    for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            axes = [self]
        for ax in axes:
            ax._anchor = anchor

        self.stale = True

    def get_data_ratio(self):
        """
        Return the aspect ratio of the scaled data.

        Notes
        -----
        This method is intended to be overridden by new projection types.
        """
        txmin, txmax = self.xaxis.get_transform().transform(self.get_xbound())
        tymin, tymax = self.yaxis.get_transform().transform(self.get_ybound())
        xsize = max(abs(txmax - txmin), 1e-30)
        ysize = max(abs(tymax - tymin), 1e-30)
        return ysize / xsize

    def apply_aspect(self, position=None):
        """
        Adjust the Axes for a specified data aspect ratio.

        Depending on `.get_adjustable` this will modify either the
        Axes box (position) or the view limits. In the former case,
        `~matplotlib.axes.Axes.get_anchor` will affect the position.

        Parameters
        ----------
        position : None or .Bbox

            .. note::
                This parameter exists for historic reasons and is considered
                internal. End users should not use it.

            If not ``None``, this defines the position of the
            Axes within the figure as a Bbox. See `~.Axes.get_position`
            for further details.

        Notes
        -----
        This is called automatically when each Axes is drawn.  You may need
        to call it yourself if you need to update the Axes position and/or
        view limits before the Figure is drawn.

        An alternative with a broader scope is `.Figure.draw_without_rendering`,
        which updates all stale components of a figure, not only the positioning /
        view limits of a single Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            For a description of aspect ratio handling.
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_anchor
            Set the position in case of extra space.
        matplotlib.figure.Figure.draw_without_rendering
            Update all stale components of a figure.

        Examples
        --------
        A typical usage example would be the following. `~.Axes.imshow` sets the
        aspect to 1, but adapting the Axes position and extent to reflect this is
        deferred until rendering for performance reasons. If you want to know the
        Axes size before, you need to call `.apply_aspect` to get the correct
        values.

        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.zeros((3, 3)))
        >>> ax.bbox.width, ax.bbox.height
        (496.0, 369.59999999999997)
        >>> ax.apply_aspect()
        >>> ax.bbox.width, ax.bbox.height
        (369.59999999999997, 369.59999999999997)
        """
        if position is None:
            position = self.get_position(original=True)

        aspect = self.get_aspect()

        if aspect == 'auto' and self._box_aspect is None:
            self._set_position(position, which='active')
            return

        trans = self.get_figure(root=False).transSubfigure
        bb = mtransforms.Bbox.unit().transformed(trans)
        # this is the physical aspect of the panel (or figure):
        fig_aspect = bb.height / bb.width

        if self._adjustable == 'box':
            if self in self._twinned_axes:
                raise RuntimeError("Adjustable 'box' is not allowed in a "
                                   "twinned Axes; use 'datalim' instead")
            box_aspect = aspect * self.get_data_ratio()
            pb = position.frozen()
            pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
            self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')
            return

        # The following is only seen if self._adjustable == 'datalim'
        if self._box_aspect is not None:
            pb = position.frozen()
            pb1 = pb.shrunk_to_aspect(self._box_aspect, pb, fig_aspect)
            self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')
            if aspect == "auto":
                return

        # reset active to original in case it had been changed by prior use
        # of 'box'
        if self._box_aspect is None:
            self._set_position(position, which='active')
        else:
            position = pb1.anchored(self.get_anchor(), pb)

        x_trf = self.xaxis.get_transform()
        y_trf = self.yaxis.get_transform()
        xmin, xmax = x_trf.transform(self.get_xbound())
        ymin, ymax = y_trf.transform(self.get_ybound())
        xsize = max(abs(xmax - xmin), 1e-30)
        ysize = max(abs(ymax - ymin), 1e-30)

