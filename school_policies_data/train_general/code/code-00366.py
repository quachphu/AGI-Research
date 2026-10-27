            renderer = self.copy_with_path_effect([path_effect])
            # Recursively call this method, only next time we will only have
            # one path effect.
            renderer.draw_markers(gc, marker_path, marker_trans, path,
                                  *args, **kwargs)

    def draw_path_collection(self, gc, master_transform, paths, *args,
                             **kwargs):
        # We do a little shimmy so that all paths are drawn for each path
        # effect in turn. Essentially, we induce recursion (depth 1) which is
        # terminated once we have just a single path effect to work with.
        if len(self._path_effects) == 1:
            # Call the base path effect function - this uses the unoptimised
            # approach of calling "draw_path" multiple times.
            return super().draw_path_collection(gc, master_transform, paths,
                                                *args, **kwargs)

        for path_effect in self._path_effects:
            renderer = self.copy_with_path_effect([path_effect])
            # Recursively call this method, only next time we will only have
            # one path effect.
            renderer.draw_path_collection(gc, master_transform, paths,
                                          *args, **kwargs)

    def open_group(self, s, gid=None):
        return self._renderer.open_group(s, gid)

    def close_group(self, s):
        return self._renderer.close_group(s)


class Normal(AbstractPathEffect):
    """
    The "identity" PathEffect.

    The Normal PathEffect's sole purpose is to draw the original artist with
    no special path effect.
    """


def _subclass_with_normal(effect_class):
    """
    Create a PathEffect class combining *effect_class* and a normal draw.
    """

    class withEffect(effect_class):
        def draw_path(self, renderer, gc, tpath, affine, rgbFace):
            super().draw_path(renderer, gc, tpath, affine, rgbFace)
            renderer.draw_path(gc, tpath, affine, rgbFace)

    withEffect.__name__ = f"with{effect_class.__name__}"
    withEffect.__qualname__ = f"with{effect_class.__name__}"
    withEffect.__doc__ = f"""
    A shortcut PathEffect for applying `.{effect_class.__name__}` and then
    drawing the original Artist.

    With this class you can use ::

        artist.set_path_effects([patheffects.with{effect_class.__name__}()])

    as a shortcut for ::

        artist.set_path_effects([patheffects.{effect_class.__name__}(),
                                 patheffects.Normal()])
    """
    # Docstring inheritance doesn't work for locally-defined subclasses.
    withEffect.draw_path.__doc__ = effect_class.draw_path.__doc__
    return withEffect


class Stroke(AbstractPathEffect):
    """A line based PathEffect which re-draws a stroke."""

    def __init__(self, offset=(0, 0), **kwargs):
        """
        The path will be stroked with its gc updated with the given
        keyword arguments, i.e., the keyword arguments should be valid
        gc parameter values.
        """
        super().__init__(offset)
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        gc0 = renderer.new_gc()  # Don't modify gc, but a copy!
        gc0.copy_properties(gc)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer), rgbFace)
        gc0.restore()


withStroke = _subclass_with_normal(effect_class=Stroke)


class SimplePatchShadow(AbstractPathEffect):
    """A simple shadow via a filled patch."""

    def __init__(self, offset=(2, -2),
                 shadow_rgbFace=None, alpha=None,
                 rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            The (x, y) offset of the shadow in points.
        shadow_rgbFace : :mpltype:`color`
            The shadow color.
        alpha : float, default: 0.3
            The alpha transparency of the created shadow patch.
        rho : float, default: 0.3
            A scale factor to apply to the rgbFace color if *shadow_rgbFace*
            is not specified.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        """
        super().__init__(offset)

        if shadow_rgbFace is None:
            self._shadow_rgbFace = shadow_rgbFace
        else:
            self._shadow_rgbFace = mcolors.to_rgba(shadow_rgbFace)

        if alpha is None:
            alpha = 0.3

        self._alpha = alpha
        self._rho = rho

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Overrides the standard draw_path to add the shadow offset and
        necessary color changes for the shadow.
        """
        gc0 = renderer.new_gc()  # Don't modify gc, but a copy!
        gc0.copy_properties(gc)

        if self._shadow_rgbFace is None:
            r, g, b = (rgbFace or (1., 1., 1.))[:3]
            # Scale the colors by a factor to improve the shadow effect.
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_rgbFace

        gc0.set_foreground("none")
        gc0.set_alpha(self._alpha)
        gc0.set_linewidth(0)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer),
            shadow_rgbFace)
        gc0.restore()


withSimplePatchShadow = _subclass_with_normal(effect_class=SimplePatchShadow)


class SimpleLineShadow(AbstractPathEffect):
    """A simple shadow via a line."""

    def __init__(self, offset=(2, -2),
                 shadow_color='k', alpha=0.3, rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            The (x, y) offset to apply to the path, in points.
        shadow_color : :mpltype:`color`, default: 'black'
            The shadow color.
            A value of ``None`` takes the original artist's color
            with a scale factor of *rho*.
        alpha : float, default: 0.3
            The alpha transparency of the created shadow patch.
        rho : float, default: 0.3
            A scale factor to apply to the rgbFace color if *shadow_color*
            is ``None``.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.
        """
        super().__init__(offset)
        if shadow_color is None:
            self._shadow_color = shadow_color
        else:
            self._shadow_color = mcolors.to_rgba(shadow_color)
        self._alpha = alpha
        self._rho = rho
        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Overrides the standard draw_path to add the shadow offset and
        necessary color changes for the shadow.
        """
        gc0 = renderer.new_gc()  # Don't modify gc, but a copy!
        gc0.copy_properties(gc)

        if self._shadow_color is None:
            r, g, b = (gc0.get_foreground() or (1., 1., 1.))[:3]
            # Scale the colors by a factor to improve the shadow effect.
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_color

        gc0.set_foreground(shadow_rgbFace)
        gc0.set_alpha(self._alpha)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer))
        gc0.restore()


class PathPatchEffect(AbstractPathEffect):
    """
    Draws a `.PathPatch` instance whose Path comes from the original
    PathEffect artist.
    """

    def __init__(self, offset=(0, 0), **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        **kwargs
            All keyword arguments are passed through to the
            :class:`~matplotlib.patches.PathPatch` constructor. The
            properties which cannot be overridden are "path", "clip_box"
            "transform" and "clip_path".
        """
        super().__init__(offset=offset)
        self.patch = mpatches.PathPatch([], **kwargs)

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        self.patch._path = tpath
        self.patch.set_transform(affine + self._offset_transform(renderer))
        self.patch.set_clip_box(gc.get_clip_rectangle())
        clip_path = gc.get_clip_path()
        if clip_path and self.patch.get_clip_path() is None:
            self.patch.set_clip_path(*clip_path)
        self.patch.draw(renderer)


class TickedStroke(AbstractPathEffect):
    """
    A line-based PathEffect which draws a path with a ticked style.

    This line style is frequently used to represent constraints in
    optimization.  The ticks may be used to indicate that one side
    of the line is invalid or to represent a closed boundary of a
    domain (i.e. a wall or the edge of a pipe).

    The spacing, length, and angle of ticks can be controlled.

    This line style is sometimes referred to as a hatched line.

    See also the :doc:`/gallery/misc/tickedstroke_demo` example.
    """

    def __init__(self, offset=(0, 0),
                 spacing=10.0, angle=45.0, length=np.sqrt(2),
                 **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between ticks in points.
        angle : float, default: 45.0
            The angle between the path and the tick in degrees.  The angle
            is measured as if you were an ant walking along the curve, with
            zero degrees pointing directly ahead, 90 to your left, -90
            to your right, and 180 behind you. To change side of the ticks,
            change sign of the angle.
        length : float, default: 1.414
            The length of the tick relative to spacing.
            Recommended length = 1.414 (sqrt(2)) when angle=45, length=1.0
            when angle=90 and length=2.0 when angle=60.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        Examples
        --------
        See :doc:`/gallery/misc/tickedstroke_demo`.
        """
        super().__init__(offset)

        self._spacing = spacing
        self._angle = angle
        self._length = length
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        # Do not modify the input! Use copy instead.
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        trans = affine + self._offset_transform(renderer)

        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])

        # Convert spacing parameter to pixels.
        spacing_px = renderer.points_to_pixels(self._spacing)

        # Transform before evaluation because to_polygons works at resolution
        # of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(tpath)

        # Evaluate path to straight line segments that can be used to
        # construct line ticks.
        polys = transpath.to_polygons(closed_only=False)

        for p in polys:
            x = p[:, 0]
            y = p[:, 1]

            # Can not interpolate points or draw line if only one point in
            # polyline.
            if x.size < 2:
                continue

            # Find distance between points on the line
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])

            # Build parametric coordinate along curve
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]

            num = int(np.ceil(s_total / spacing_px)) - 1
            # Pick parameter values for ticks.
            s_tick = np.linspace(spacing_px/2, s_total - spacing_px/2, num)

            # Find points along the parameterized curve
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)

            # Find unit vectors in local direction of curve
            delta_s = self._spacing * .001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s

            # Normalize slope into unit slope vector.
            n = np.hypot(u, v)
            mask = n == 0
            n[mask] = 1.0

            uv = np.array([u / n, v / n]).T
            uv[mask] = np.array([0, 0]).T

            # Rotate and scale unit vector into tick vector
            dxy = np.dot(uv, trans_matrix) * self._length * spacing_px

            # Build tick endpoints
            x_end = x_tick + dxy[:, 0]
            y_end = y_tick + dxy[:, 1]

            # Interleave ticks to form Path vertices
            xyt = np.empty((2 * num, 2), dtype=x_tick.dtype)
            xyt[0::2, 0] = x_tick
            xyt[1::2, 0] = x_end
            xyt[0::2, 1] = y_tick
            xyt[1::2, 1] = y_end

            # Build up vector of Path codes
            codes = np.tile([Path.MOVETO, Path.LINETO], num)

            # Construct and draw resulting path
            h = Path(xyt, codes)
            # Transform back to data space during render
            renderer.draw_path(gc0, h, affine.inverted() + trans, rgbFace)

        gc0.restore()


withTickedStroke = _subclass_with_normal(effect_class=TickedStroke)
<|endoftext|>"""
A module for reading dvi files output by TeX. Several limitations make
this not (currently) useful as a general-purpose dvi preprocessor, but
it is currently used by the pdf backend for processing usetex text.

Interface::

  with Dvi(filename, 72) as dvi:
      # iterate over pages:
      for page in dvi:
          w, h, d = page.width, page.height, page.descent
          for x, y, font, glyph, width in page.text:
              fontname = font.texname
              pointsize = font.size
              ...
          for x, y, height, width in page.boxes:
              ...
"""

from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

from matplotlib import _api, cbook

_log = logging.getLogger(__name__)

# Many dvi related files are looked for by external processes, require
# additional parsing, and are used many times per rendering, which is why they
# are cached using lru_cache().

# Dvi is a bytecode format documented in
# https://ctan.org/pkg/dvitype
# https://texdoc.org/serve/dvitype.pdf/0
#
# The file consists of a preamble, some number of pages, a postamble,
# and a finale. Different opcodes are allowed in different contexts,
# so the Dvi object has a parser state:
#
#   pre:       expecting the preamble
#   outer:     between pages (followed by a page or the postamble,
#              also e.g. font definitions are allowed)
#   page:      processing a page
#   post_post: state after the postamble (our current implementation
#              just stops reading)
#   finale:    the finale (unimplemented in our current implementation)

_dvistate = enum.Enum('DviState', 'pre outer inpage post_post finale')

# The marks on a page consist of text and boxes. A page also has dimensions.
Page = namedtuple('Page', 'text boxes height width descent')
Box = namedtuple('Box', 'x y height width')


# Also a namedtuple, for backcompat.
class Text(namedtuple('Text', 'x y font glyph width')):
    """
    A glyph in the dvi file.

    The *x* and *y* attributes directly position the glyph.  The *font*,
    *glyph*, and *width* attributes are kept public for back-compatibility,
    but users wanting to draw the glyph themselves are encouraged to instead
    load the font specified by `font_path` at `font_size`, warp it with the
    effects specified by `font_effects`, and load the glyph specified by
    `glyph_name_or_index`.
    """

    def _get_pdftexmap_entry(self):
        return PsfontsMap(find_tex_file("pdftex.map"))[self.font.texname]

    @property
    def font_path(self):
        """The `~pathlib.Path` to the font for this glyph."""
        psfont = self._get_pdftexmap_entry()
        if psfont.filename is None:
            raise ValueError("No usable font file found for {} ({}); "
                             "the font may lack a Type-1 version"
                             .format(psfont.psname.decode("ascii"),
                                     psfont.texname.decode("ascii")))
        return Path(psfont.filename)

    @property
    def font_size(self):
        """The font size."""
        return self.font.size

    @property
    def font_effects(self):
        """
        The "font effects" dict for this glyph.

        This dict contains the values for this glyph of SlantFont and
        ExtendFont (if any), read off :file:`pdftex.map`.
        """
        return self._get_pdftexmap_entry().effects

    @property
    def glyph_name_or_index(self):
        """
        Either the glyph name or the native charmap glyph index.

        If :file:`pdftex.map` specifies an encoding for this glyph's font, that
        is a mapping of glyph indices to Adobe glyph names; use it to convert
        dvi indices to glyph names.  Callers can then convert glyph names to
        glyph indices (with FT_Get_Name_Index/get_name_index), and load the
        glyph using FT_Load_Glyph/load_glyph.

        If :file:`pdftex.map` specifies no encoding, the indices directly map
        to the font's "native" charmap; glyphs should directly load using
        FT_Load_Char/load_char after selecting the native charmap.
        """
        entry = self._get_pdftexmap_entry()
        return (_parse_enc(entry.encoding)[self.glyph]
                if entry.encoding is not None else self.glyph)


# Opcode argument parsing
#
# Each of the following functions takes a Dvi object and delta, which is the
# difference between the opcode and the minimum opcode with the same meaning.
# Dvi opcodes often encode the number of argument bytes in this delta.
_arg_mapping = dict(
    # raw: Return delta as is.
    raw=lambda dvi, delta: delta,
    # u1: Read 1 byte as an unsigned number.
    u1=lambda dvi, delta: dvi._read_arg(1, signed=False),
    # u4: Read 4 bytes as an unsigned number.
    u4=lambda dvi, delta: dvi._read_arg(4, signed=False),
    # s4: Read 4 bytes as a signed number.
    s4=lambda dvi, delta: dvi._read_arg(4, signed=True),
    # slen: Read delta bytes as a signed number, or None if delta is None.
    slen=lambda dvi, delta: dvi._read_arg(delta, signed=True) if delta else None,
    # slen1: Read (delta + 1) bytes as a signed number.
    slen1=lambda dvi, delta: dvi._read_arg(delta + 1, signed=True),
    # ulen1: Read (delta + 1) bytes as an unsigned number.
    ulen1=lambda dvi, delta: dvi._read_arg(delta + 1, signed=False),
    # olen1: Read (delta + 1) bytes as an unsigned number if less than 4 bytes,
    # as a signed number if 4 bytes.
    olen1=lambda dvi, delta: dvi._read_arg(delta + 1, signed=(delta == 3)),
)


def _dispatch(table, min, max=None, state=None, args=('raw',)):
    """
    Decorator for dispatch by opcode. Sets the values in *table*
    from *min* to *max* to this method, adds a check that the Dvi state
    matches *state* if not None, reads arguments from the file according
    to *args*.

    Parameters
    ----------
    table : dict[int, callable]
        The dispatch table to be filled in.

    min, max : int
        Range of opcodes that calls the registered function; *max* defaults to
        *min*.

    state : _dvistate, optional
        State of the Dvi object in which these opcodes are allowed.

    args : list[str], default: ['raw']
        Sequence of argument specifications:

        - 'raw': opcode minus minimum
        - 'u1': read one unsigned byte
        - 'u4': read four bytes, treat as an unsigned number
        - 's4': read four bytes, treat as a signed number
        - 'slen': read (opcode - minimum) bytes, treat as signed
        - 'slen1': read (opcode - minimum + 1) bytes, treat as signed
        - 'ulen1': read (opcode - minimum + 1) bytes, treat as unsigned
        - 'olen1': read (opcode - minimum + 1) bytes, treat as unsigned
          if under four bytes, signed if four bytes
    """
    def decorate(method):
        get_args = [_arg_mapping[x] for x in args]

        @wraps(method)
        def wrapper(self, byte):
            if state is not None and self.state != state:
                raise ValueError("state precondition failed")
            return method(self, *[f(self, byte-min) for f in get_args])
        if max is None:
            table[min] = wrapper
        else:
            for i in range(min, max+1):
                assert table[i] is None
                table[i] = wrapper
        return wrapper
    return decorate


class Dvi:
    """
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
        ...     for page in dvi:
        ...         print(''.join(chr(t.glyph) for t in page.text))
    """
    # dispatch table
    _dtable = [None] * 256
    _dispatch = partial(_dispatch, _dtable)

    def __init__(self, filename, dpi):
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
        _log.debug('Dvi: %s', filename)
        self.file = open(filename, 'rb')
        self.dpi = dpi
        self.fonts = {}
        self.state = _dvistate.pre
        self._missing_font = None

    def __enter__(self):
        """Context manager enter method, does nothing."""
        return self

    def __exit__(self, etype, evalue, etrace):
        """
        Context manager exit method, closes the underlying file if it is open.
        """
        self.close()

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Yields
        ------
        Page
            Details of all the text and box objects on the page.
            The Page tuple contains lists of Text and Box tuples and
            the page dimensions, and the Text and Box tuples contain
            coordinates transformed into a standard Cartesian
            coordinate system at the dpi value given when initializing.
            The coordinates are floating point numbers, but otherwise
            precision is not lost and coordinate values are not clipped to
            integers.
        """
        while self._read():
            yield self._output()

    def close(self):
        """Close the underlying file if it is open."""
        if not self.file.closed:
            self.file.close()

    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
        minx = miny = np.inf
        maxx = maxy = -np.inf
        maxy_pure = -np.inf
        for elt in self.text + self.boxes:
            if isinstance(elt, Box):
                x, y, h, w = elt
                e = 0  # zero depth
            else:  # glyph
                x, y, font, g, w = elt
                h, e = font._height_depth_of(g)
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)
        if self._baseline_v is not None:
            maxy_pure = self._baseline_v  # This should normally be the case.
            self._baseline_v = None

        if not self.text and not self.boxes:  # Avoid infs/nans from inf+/-inf.
            return Page(text=[], boxes=[], width=0, height=0, descent=0)

        if self.dpi is None:
            # special case for ease of debugging: output raw dvi coordinates
            return Page(text=self.text, boxes=self.boxes,
                        width=maxx-minx, height=maxy_pure-miny,
                        descent=maxy-maxy_pure)

        # convert from TeX's "scaled points" to dpi units
        d = self.dpi / (72.27 * 2**16)
        descent = (maxy - maxy_pure) * d

        text = [Text((x-minx)*d, (maxy-y)*d - descent, f, g, w*d)
                for (x, y, f, g, w) in self.text]
        boxes = [Box((x-minx)*d, (maxy-y)*d - descent, h*d, w*d)
                 for (x, y, h, w) in self.boxes]

        return Page(text=text, boxes=boxes, width=(maxx-minx)*d,
                    height=(maxy_pure-miny)*d, descent=descent)

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        # Pages appear to start with the sequence
        #   bop (begin of page)
        #   xxx comment
        #   <push, ..., pop>  # if using chemformula
        #   down
        #   push
        #     down
        #     <push, push, xxx, right, xxx, pop, pop>  # if using xcolor
        #     down
        #     push
        #       down (possibly multiple)
        #       push  <=  here, v is the baseline position.
        #         etc.
        # (dviasm is useful to explore this structure.)
        # Thus, we use the vertical position at the first time the stack depth
        # reaches 3, while at least three "downs" have been executed (excluding
        # those popped out (corresponding to the chemformula preamble)), as the
        # baseline (the "down" count is necessary to handle xcolor).
        down_stack = [0]
        self._baseline_v = None
        while True:
            byte = self.file.read(1)[0]
            self._dtable[byte](self, byte)
            if self._missing_font:
                raise self._missing_font.to_exception()
            name = self._dtable[byte].__name__
            if name == "_push":
                down_stack.append(down_stack[-1])
            elif name == "_pop":
                down_stack.pop()
            elif name == "_down":
                down_stack[-1] += 1
            if (self._baseline_v is None
                    and len(getattr(self, "stack", [])) == 3
                    and down_stack[-1] >= 4):
                self._baseline_v = self.v
            if byte == 140:                         # end of page
                return True
            if self.state is _dvistate.post_post:   # end of file
                self.close()
                return False

    def _read_arg(self, nbytes, signed=False):
        """
        Read and return a big-endian integer *nbytes* long.
        Signedness is determined by the *signed* keyword.
        """
        return int.from_bytes(self.file.read(nbytes), "big", signed=signed)

    @_dispatch(min=0, max=127, state=_dvistate.inpage)
    def _set_char_immediate(self, char):
        self._put_char_real(char)
        if isinstance(self.fonts[self.f], cbook._ExceptionInfo):
            return
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(min=128, max=131, state=_dvistate.inpage, args=('olen1',))
    def _set_char(self, char):
        self._put_char_real(char)
        if isinstance(self.fonts[self.f], cbook._ExceptionInfo):
            return
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(132, state=_dvistate.inpage, args=('s4', 's4'))
    def _set_rule(self, a, b):
        self._put_rule_real(a, b)
        self.h += b

    @_dispatch(min=133, max=136, state=_dvistate.inpage, args=('olen1',))
    def _put_char(self, char):
        self._put_char_real(char)

    def _put_char_real(self, char):
        font = self.fonts[self.f]
        if isinstance(font, cbook._ExceptionInfo):
            self._missing_font = font
        elif font._vf is None:
            self.text.append(Text(self.h, self.v, font, char,
                                  font._width_of(char)))
        else:
            scale = font._scale
            for x, y, f, g, w in font._vf[char].text:
                newf = DviFont(scale=_mul2012(scale, f._scale),
                               tfm=f._tfm, texname=f.texname, vf=f._vf)
                self.text.append(Text(self.h + _mul2012(x, scale),
                                      self.v + _mul2012(y, scale),
                                      newf, g, newf._width_of(g)))
            self.boxes.extend([Box(self.h + _mul2012(x, scale),
                                   self.v + _mul2012(y, scale),
                                   _mul2012(a, scale), _mul2012(b, scale))
                               for x, y, a, b in font._vf[char].boxes])

    @_dispatch(137, state=_dvistate.inpage, args=('s4', 's4'))
    def _put_rule(self, a, b):
        self._put_rule_real(a, b)

    def _put_rule_real(self, a, b):
        if a > 0 and b > 0:
            self.boxes.append(Box(self.h, self.v, a, b))

    @_dispatch(138)
    def _nop(self, _):
        pass

    @_dispatch(139, state=_dvistate.outer, args=('s4',)*11)
    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        self.state = _dvistate.inpage
        self.h = self.v = self.w = self.x = self.y = self.z = 0
        self.stack = []
        self.text = []          # list of Text objects
        self.boxes = []         # list of Box objects

    @_dispatch(140, state=_dvistate.inpage)
    def _eop(self, _):
        self.state = _dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    @_dispatch(141, state=_dvistate.inpage)
    def _push(self, _):
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    @_dispatch(142, state=_dvistate.inpage)
    def _pop(self, _):
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    @_dispatch(min=143, max=146, state=_dvistate.inpage, args=('slen1',))
    def _right(self, b):
        self.h += b

    @_dispatch(min=147, max=151, state=_dvistate.inpage, args=('slen',))
    def _right_w(self, new_w):
        if new_w is not None:
            self.w = new_w
        self.h += self.w

    @_dispatch(min=152, max=156, state=_dvistate.inpage, args=('slen',))
    def _right_x(self, new_x):
        if new_x is not None:
            self.x = new_x
        self.h += self.x

    @_dispatch(min=157, max=160, state=_dvistate.inpage, args=('slen1',))
    def _down(self, a):
        self.v += a

    @_dispatch(min=161, max=165, state=_dvistate.inpage, args=('slen',))
    def _down_y(self, new_y):
        if new_y is not None:
            self.y = new_y
        self.v += self.y

    @_dispatch(min=166, max=170, state=_dvistate.inpage, args=('slen',))
    def _down_z(self, new_z):
        if new_z is not None:
            self.z = new_z
        self.v += self.z

    @_dispatch(min=171, max=234, state=_dvistate.inpage)
    def _fnt_num_immediate(self, k):
        self.f = k

    @_dispatch(min=235, max=238, state=_dvistate.inpage, args=('olen1',))
    def _fnt_num(self, new_f):
        self.f = new_f

    @_dispatch(min=239, max=242, args=('ulen1',))
    def _xxx(self, datalen):
        special = self.file.read(datalen)
        _log.debug(
            'Dvi._xxx: encountered special: %s',
            ''.join([chr(ch) if 32 <= ch < 127 else '<%02x>' % ch
                     for ch in special]))

    @_dispatch(min=243, max=246, args=('olen1', 'u4', 'u4', 'u4', 'u1', 'u1'))
    def _fnt_def(self, k, c, s, d, a, l):
        self._fnt_def_real(k, c, s, d, a, l)

    def _fnt_def_real(self, k, c, s, d, a, l):
        n = self.file.read(a + l)
        fontname = n[-l:].decode('ascii')
        try:
            tfm = _tfmfile(fontname)
        except FileNotFoundError as exc:
            # Explicitly allow defining missing fonts for Vf support; we only
            # register an error when trying to load a glyph from a missing font
            # and throw that error in Dvi._read.  For Vf, _finalize_packet
            # checks whether a missing glyph has been used, and in that case
            # skips the glyph definition.
            self.fonts[k] = cbook._ExceptionInfo.from_exception(exc)
            return
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError(f'tfm checksum mismatch: {n}')
        try:
            vf = _vffile(fontname)
        except FileNotFoundError:
            vf = None
        self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)

    @_dispatch(247, state=_dvistate.pre, args=('u1', 'u4', 'u4', 'u4', 'u1'))
    def _pre(self, i, num, den, mag, k):
        self.file.read(k)  # comment in the dvi file
        if i != 2:
            raise ValueError(f"Unknown dvi format {i}")
        if num != 25400000 or den != 7227 * 2**16:
            raise ValueError("Nonstandard units in dvi file")
            # meaning: TeX always uses those exact values, so it
            # should be enough for us to support those
            # (There are 72.27 pt to an inch so 7227 pt =
            # 7227 * 2**16 sp to 100 in. The numerator is multiplied
            # by 10^5 to get units of 10**-7 meters.)
        if mag != 1000:
            raise ValueError("Nonstandard magnification in dvi file")
            # meaning: LaTeX seems to frown on setting \mag, so
            # I think we can assume this is constant
        self.state = _dvistate.outer

    @_dispatch(248, state=_dvistate.outer)
    def _post(self, _):
        self.state = _dvistate.post_post
        # TODO: actually read the postamble and finale?
        # currently post_post just triggers closing the file

    @_dispatch(249)
    def _post_post(self, _):
        raise NotImplementedError

    @_dispatch(min=250, max=255)
    def _malformed(self, offset):
        raise ValueError(f"unknown command: byte {250 + offset}")


class DviFont:
    """
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    tfm : Tfm
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally by TeX and friends, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
    size : float
       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.
    widths : list
       Widths of glyphs in glyph-space units, typically 1/1000ths of
       the point size.

    """
    __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')

    def __init__(self, scale, tfm, texname, vf):
        _api.check_isinstance(bytes, texname=texname)
        self._scale = scale
        self._tfm = tfm
        self.texname = texname
        self._vf = vf
        self.size = scale * (72.0 / (72.27 * 2**16))
        try:
            nchars = max(tfm.width) + 1
        except ValueError:
            nchars = 0
        self.widths = [(1000*tfm.width.get(char, 0)) >> 20
                       for char in range(nchars)]

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.texname == other.texname and self.size == other.size)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"<{type(self).__name__}: {self.texname}>"

    def _width_of(self, char):
        """Width of char in dvi units."""
        width = self._tfm.width.get(char, None)
        if width is not None:
            return _mul2012(width, self._scale)
        _log.debug('No width for char %d in font %s.', char, self.texname)
        return 0

    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""
        result = []
        for metric, name in ((self._tfm.height, "height"),
                             (self._tfm.depth, "depth")):
            value = metric.get(char, None)
            if value is None:
                _log.debug('No %s for char %d in font %s',
                           name, char, self.texname)
                result.append(0)
            else:
                result.append(_mul2012(value, self._scale))
        # cmsyXX (symbols font) glyph 0 ("minus") has a nonzero descent
        # so that TeX aligns equations properly
        # (https://tex.stackexchange.com/q/526103/)
        # but we actually care about the rasterization depth to align
        # the dvipng-generated images.
        if re.match(br'^cmsy\d+$', self.texname) and char == 0:
            result[-1] = 0
        return result


class Vf(Dvi):
    r"""
    A virtual font (\*.vf file) containing subroutines for dvi files.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    The virtual font format is a derivative of dvi:
    http://mirrors.ctan.org/info/knuth/virtual-fonts
    This class reuses some of the machinery of `Dvi`
    but replaces the `_read` loop and dispatch mechanism.

    Examples
    --------
    ::

        vf = Vf(filename)
        glyph = vf[code]
        glyph.text, glyph.boxes, glyph.width
    """

    def __init__(self, filename):
        super().__init__(filename, 0)
        try:
            self._first_font = None
            self._chars = {}
            self._read()
        finally:
            self.close()

    def __getitem__(self, code):
        return self._chars[code]

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        packet_char = packet_ends = None
        packet_len = packet_width = None
        while True:
            byte = self.file.read(1)[0]
            # If we are in a packet, execute the dvi instructions
            if self.state is _dvistate.inpage:
                byte_at = self.file.tell()-1
                if byte_at == packet_ends:
                    self._finalize_packet(packet_char, packet_width)
                    packet_len = packet_char = packet_width = None
                    # fall through to out-of-packet code
                elif byte_at > packet_ends:
                    raise ValueError("Packet length mismatch in vf file")
                else:
                    if byte in (139, 140) or byte >= 243:
                        raise ValueError(f"Inappropriate opcode {byte} in vf file")
                    Dvi._dtable[byte](self, byte)
                    continue

            # We are outside a packet
            if byte < 242:          # a short packet (length given by byte)
                packet_len = byte
                packet_char = self._read_arg(1)
                packet_width = self._read_arg(3)
                packet_ends = self._init_packet(byte)
                self.state = _dvistate.inpage
            elif byte == 242:       # a long packet
                packet_len = self._read_arg(4)
                packet_char = self._read_arg(4)
                packet_width = self._read_arg(4)
                self._init_packet(packet_len)
            elif 243 <= byte <= 246:
                k = self._read_arg(byte - 242, byte == 246)
                c = self._read_arg(4)
                s = self._read_arg(4)
                d = self._read_arg(4)
                a = self._read_arg(1)
                l = self._read_arg(1)
                self._fnt_def_real(k, c, s, d, a, l)
                if self._first_font is None:
                    self._first_font = k
            elif byte == 247:       # preamble
                i = self._read_arg(1)
                k = self._read_arg(1)
                x = self.file.read(k)
                cs = self._read_arg(4)
                ds = self._read_arg(4)
                self._pre(i, x, cs, ds)
            elif byte == 248:       # postamble (just some number of 248s)
                break
            else:
                raise ValueError(f"Unknown vf opcode {byte}")

    def _init_packet(self, pl):
        if self.state != _dvistate.outer:
            raise ValueError("Misplaced packet in vf file")
        self.h = self.v = self.w = self.x = self.y = self.z = 0
        self.stack = []
        self.text = []
        self.boxes = []
        self.f = self._first_font
        self._missing_font = None
        return self.file.tell() + pl

    def _finalize_packet(self, packet_char, packet_width):
        if not self._missing_font:  # Otherwise we don't have full glyph definition.
            self._chars[packet_char] = Page(
                text=self.text, boxes=self.boxes, width=packet_width,
                height=None, descent=None)
        self.state = _dvistate.outer

    def _pre(self, i, x, cs, ds):
        if self.state is not _dvistate.pre:
            raise ValueError("pre command in middle of vf file")
        if i != 202:
            raise ValueError(f"Unknown vf format {i}")
        if len(x):
            _log.debug('vf file comment: %s', x)
        self.state = _dvistate.outer
        # cs = checksum, ds = design size


def _mul2012(num1, num2):
    """Multiply two numbers in 20.12 fixed point format."""
    # Separated into a function because >> has surprising precedence
    return (num1*num2) >> 20


class Tfm:
    """
    A TeX Font Metric file.

    This implementation covers only the bare minimum needed by the Dvi class.

    Parameters
    ----------
    filename : str or path-like

    Attributes
    ----------
    checksum : int
       Used for verifying against the dvi file.
    design_size : int
       Design size of the font (unknown units)
    width, height, depth : dict
       Dimensions of each character, need to be scaled by the factor
       specified in the dvi file. These are dicts because indexing may
       not start from 0.
    """
    __slots__ = ('checksum', 'design_size', 'width', 'height', 'depth')

    def __init__(self, filename):
        _log.debug('opening tfm file %s', filename)
        with open(filename, 'rb') as file:
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = struct.unpack('!6H', header1[2:14])
            _log.debug('lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d',
                       lh, bc, ec, nw, nh, nd)
            header2 = file.read(4*lh)
            self.checksum, self.design_size = struct.unpack('!2I', header2[:8])
            # there is also encoding information etc.
            char_info = file.read(4*(ec-bc+1))
            widths = struct.unpack(f'!{nw}i', file.read(4*nw))
            heights = struct.unpack(f'!{nh}i', file.read(4*nh))
            depths = struct.unpack(f'!{nd}i', file.read(4*nd))
        self.width = {}
        self.height = {}
        self.depth = {}
        for idx, char in enumerate(range(bc, ec+1)):
            byte0 = char_info[4*idx]
            byte1 = char_info[4*idx+1]
            self.width[char] = widths[byte0]
            self.height[char] = heights[byte1 >> 4]
            self.depth[char] = depths[byte1 & 0xf]


PsFont = namedtuple('PsFont', 'texname psname effects encoding filename')


class PsfontsMap:
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX's native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
    the 35 PostScript fonts (i.e., have no filename for them, as in
    the Times-Bold example above), while the pdf-related files perhaps
    only avoid the "Base 14" pdf fonts. But the user may have
    configured these files differently.

    Examples
    --------
    >>> map = PsfontsMap(find_tex_file('pdftex.map'))
    >>> entry = map[b'ptmbo8r']
    >>> entry.texname
    b'ptmbo8r'
    >>> entry.psname
    b'Times-Bold'
    >>> entry.encoding
    '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
    >>> entry.effects
    {'slant': 0.16700000000000001}
    >>> entry.filename
    """
    __slots__ = ('_filename', '_unparsed', '_parsed')

    # Create a filename -> PsfontsMap cache, so that calling
    # `PsfontsMap(filename)` with the same filename a second time immediately
    # returns the same object.
    @lru_cache
    def __new__(cls, filename):
        self = object.__new__(cls)
        self._filename = os.fsdecode(filename)
        # Some TeX distributions have enormous pdftex.map files which would
        # take hundreds of milliseconds to parse, but it is easy enough to just
        # store the unparsed lines (keyed by the first word, which is the
        # texname) and parse them on-demand.
        with open(filename, 'rb') as file:
            self._unparsed = {}
            for line in file:
                tfmname = line.split(b' ', 1)[0]
                self._unparsed.setdefault(tfmname, []).append(line)
        self._parsed = {}
        return self

    def __getitem__(self, texname):
        assert isinstance(texname, bytes)
        if texname in self._unparsed:
            for line in self._unparsed.pop(texname):
                if self._parse_and_cache_line(line):
                    break
        try:
            return self._parsed[texname]
        except KeyError:
            raise LookupError(
                f"An associated PostScript font (required by Matplotlib) "
                f"could not be found for TeX font {texname.decode('ascii')!r} "
                f"in {self._filename!r}; this problem can often be solved by "
                f"installing a suitable PostScript font package in your TeX "
                f"package manager") from None

    def _parse_and_cache_line(self, line):
        """
        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        """
        # If the map file specifies multiple encodings for a font, we
        # follow pdfTeX in choosing the last one specified. Such
        # entries are probably mistakes but they have occurred.
        # https://tex.stackexchange.com/q/10826/

        if not line or line.startswith((b" ", b"%", b"*", b";", b"#")):
            return
        tfmname = basename = special = encodingfile = fontfile = None
        is_subsetted = is_t1 = is_truetype = False
        matches = re.finditer(br'"([^"]*)(?:"|$)|(\S+)', line)
        for match in matches:
            quoted, unquoted = match.groups()
            if unquoted:
                if unquoted.startswith(b"<<"):  # font
                    fontfile = unquoted[2:]
                elif unquoted.startswith(b"<["):  # encoding
                    encodingfile = unquoted[2:]
                elif unquoted.startswith(b"<"):  # font or encoding
                    word = (
                        # <foo => foo
                        unquoted[1:]
                        # < by itself => read the next word
                        or next(filter(None, next(matches).groups())))
                    if word.endswith(b".enc"):
                        encodingfile = word
                    else:
                        fontfile = word
                        is_subsetted = True
                elif tfmname is None:
                    tfmname = unquoted
                elif basename is None:
                    basename = unquoted
            elif quoted:
                special = quoted
        effects = {}
        if special:
            words = reversed(special.split())
            for word in words:
                if word == b"SlantFont":
                    effects["slant"] = float(next(words))
                elif word == b"ExtendFont":
                    effects["extend"] = float(next(words))

        # Verify some properties of the line that would cause it to be ignored
        # otherwise.
        if fontfile is not None:
            if fontfile.endswith((b".ttf", b".ttc")):
                is_truetype = True
            elif not fontfile.endswith(b".otf"):
                is_t1 = True
        elif basename is not None:
            is_t1 = True
        if is_truetype and is_subsetted and encodingfile is None:
            return
        if not is_t1 and ("slant" in effects or "extend" in effects):
            return
        if abs(effects.get("slant", 0)) > 1:
            return
        if abs(effects.get("extend", 0)) > 2:
            return

        if basename is None:
            basename = tfmname
        if encodingfile is not None:
            encodingfile = find_tex_file(encodingfile)
        if fontfile is not None:
            fontfile = find_tex_file(fontfile)
        self._parsed[tfmname] = PsFont(
            texname=tfmname, psname=basename, effects=effects,
            encoding=encodingfile, filename=fontfile)
        return True


def _parse_enc(path):
    r"""
    Parse a \*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : `os.PathLike`

    Returns
    -------
    list
        The nth entry of the list is the PostScript glyph name of the nth
        glyph.
    """
    no_comments = re.sub("%.*", "", Path(path).read_text(encoding="ascii"))
    array = re.search(r"(?s)\[(.*)\]", no_comments).group(1)
    lines = [line for line in array.split() if line]
    if all(line.startswith("/") for line in lines):
        return [line[1:] for line in lines]
    else:
        raise ValueError(f"Failed to parse {path} as Postscript encoding")


class _LuatexKpsewhich:
    @lru_cache  # A singleton.
    def __new__(cls):
        self = object.__new__(cls)
        self._proc = self._new_proc()
        return self

    def _new_proc(self):
        return subprocess.Popen(
            ["luatex", "--luaonly",
             str(cbook._get_data_path("kpsewhich.lua"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def search(self, filename):
        if self._proc.poll() is not None:  # Dead, restart it.
            self._proc = self._new_proc()
        self._proc.stdin.write(os.fsencode(filename) + b"\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline().rstrip()
        return None if out == b"nil" else os.fsdecode(out)


@lru_cache
def find_tex_file(filename):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """

    # we expect these to always be ascii encoded, but use utf-8
    # out of caution
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='replace')

    try:
        lk = _LuatexKpsewhich()
    except FileNotFoundError:
        lk = None  # Fallback to directly calling kpsewhich, as below.

    if lk:
        path = lk.search(filename)
    else:
        if sys.platform == 'win32':
            # On Windows only, kpathsea can use utf-8 for cmd args and output.
            # The `command_line_encoding` environment variable is set to force
            # it to always use utf-8 encoding.  See Matplotlib issue #11848.
            kwargs = {'env': {**os.environ, 'command_line_encoding': 'utf-8'},
                      'encoding': 'utf-8'}
        else:  # On POSIX, run through the equivalent of os.fsdecode().
            kwargs = {'encoding': sys.getfilesystemencoding(),
                      'errors': 'surrogateescape'}

        try:
            path = (cbook._check_and_log_subprocess(['kpsewhich', filename],
                                                    _log, **kwargs)
                    .rstrip('\n'))
        except (FileNotFoundError, RuntimeError):
            path = None

    if path:
        return path
    else:
        raise FileNotFoundError(
            f"Matplotlib's TeX implementation searched for a file named "
            f"{filename!r} in your texmf tree, but could not find it")


@lru_cache
def _fontfile(cls, suffix, texname):
    return cls(find_tex_file(texname + suffix))


_tfmfile = partial(_fontfile, Tfm, ".tfm")
_vffile = partial(_fontfile, Vf, ".vf")


if __name__ == '__main__':
    from argparse import ArgumentParser
    import itertools

    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("dpi", nargs="?", type=float, default=None)
    args = parser.parse_args()
    with Dvi(args.filename, args.dpi) as dvi:
        fontmap = PsfontsMap(find_tex_file('pdftex.map'))
        for page in dvi:
            print(f"=== new page === "
                  f"(w: {page.width}, h: {page.height}, d: {page.descent})")
            for font, group in itertools.groupby(
                    page.text, lambda text: text.font):
                print(f"font: {font.texname.decode('latin-1')!r}\t"
                      f"scale: {font._scale / 2 ** 20}")
                print("x", "y", "glyph", "chr", "w", "(glyphs)", sep="\t")
                for text in group:
                    print(text.x, text.y, text.glyph,
                          chr(text.glyph) if chr(text.glyph).isprintable()
                          else ".",
                          text.width, sep="\t")
            if page.boxes:
                print("x", "y", "h", "w", "", "(boxes)", sep="\t")
                for box in page.boxes:
                    print(box.x, box.y, box.height, box.width, sep="\t")
<|endoftext|>"""
The image module supports basic image loading, rescaling and display
operations.
"""

import math
import os
import logging
from pathlib import Path
import warnings

import numpy as np
import PIL.Image
import PIL.PngImagePlugin

import matplotlib as mpl
from matplotlib import _api, cbook
# For clarity, names from _image are given explicitly in this module
from matplotlib import _image
# For user convenience, the names from _image are also imported into
# the image namespace
from matplotlib._image import *  # noqa: F401, F403
import matplotlib.artist as martist
import matplotlib.colorizer as mcolorizer
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
    Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
    IdentityTransform, TransformedBbox)

_log = logging.getLogger(__name__)

# map interpolation strings to module constants
_interpd_ = {
    'auto': _image.NEAREST,  # this will use nearest or Hanning...
    'none': _image.NEAREST,  # fall back to nearest when not supported
    'nearest': _image.NEAREST,
    'bilinear': _image.BILINEAR,
    'bicubic': _image.BICUBIC,
    'spline16': _image.SPLINE16,
    'spline36': _image.SPLINE36,
    'hanning': _image.HANNING,
    'hamming': _image.HAMMING,
    'hermite': _image.HERMITE,
    'kaiser': _image.KAISER,
    'quadric': _image.QUADRIC,
    'catrom': _image.CATROM,
    'gaussian': _image.GAUSSIAN,
    'bessel': _image.BESSEL,
    'mitchell': _image.MITCHELL,
    'sinc': _image.SINC,
    'lanczos': _image.LANCZOS,
    'blackman': _image.BLACKMAN,
    'antialiased': _image.NEAREST,  # this will use nearest or Hanning...
}

interpolations_names = set(_interpd_)


def composite_images(images, renderer, magnification=1.0):
    """
    Composite a number of RGBA images into one.  The images are
    composited in the order in which they appear in the *images* list.

    Parameters
    ----------
    images : list of Images
        Each must have a `make_image` method.  For each image,
        `can_composite` should return `True`, though this is not
        enforced by this function.  Each image must have a purely
        affine transformation with no shear.

    renderer : `.RendererBase`

    magnification : float, default: 1
        The additional magnification to apply for the renderer in use.

    Returns
    -------
    image : (M, N, 4) `numpy.uint8` array
        The composited RGBA image.
    offset_x, offset_y : float
        The (left, bottom) offset where the composited image should be placed
        in the output figure.
    """
    if len(images) == 0:
        return np.empty((0, 0, 4), dtype=np.uint8), 0, 0

    parts = []
    bboxes = []
    for image in images:
        data, x, y, trans = image.make_image(renderer, magnification)
        if data is not None:
            x *= magnification
            y *= magnification
            parts.append((data, x, y, image._get_scalar_alpha()))
            bboxes.append(
                Bbox([[x, y], [x + data.shape[1], y + data.shape[0]]]))

    if len(parts) == 0:
        return np.empty((0, 0, 4), dtype=np.uint8), 0, 0

    bbox = Bbox.union(bboxes)

    output = np.zeros(
        (int(bbox.height), int(bbox.width), 4), dtype=np.uint8)

    for data, x, y, alpha in parts:
        trans = Affine2D().translate(x - bbox.x0, y - bbox.y0)
        _image.resample(data, output, trans, _image.NEAREST,
                        resample=False, alpha=alpha)

    return output, bbox.x0 / magnification, bbox.y0 / magnification


def _draw_list_compositing_images(
        renderer, parent, artists, suppress_composite=None):
    """
    Draw a sorted list of artists, compositing images into a single
    image where possible.

    For internal Matplotlib use only: It is here to reduce duplication
    between `Figure.draw` and `Axes.draw`, but otherwise should not be
    generally useful.
    """
    has_images = any(isinstance(x, _ImageBase) for x in artists)

    # override the renderer default if suppressComposite is not None
    not_composite = (suppress_composite if suppress_composite is not None
                     else renderer.option_image_nocomposite())

    if not_composite or not has_images:
        for a in artists:
            a.draw(renderer)
    else:
        # Composite any adjacent images together
        image_group = []
        mag = renderer.get_image_magnification()

        def flush_images():
            if len(image_group) == 1:
                image_group[0].draw(renderer)
            elif len(image_group) > 1:
                data, l, b = composite_images(image_group, renderer, mag)
                if data.size != 0:
                    gc = renderer.new_gc()
                    gc.set_clip_rectangle(parent.bbox)
                    gc.set_clip_path(parent.get_clip_path())
                    renderer.draw_image(gc, round(l), round(b), data)
                    gc.restore()
            del image_group[:]

        for a in artists:
            if (isinstance(a, _ImageBase) and a.can_composite() and
                    a.get_clip_on() and not a.get_clip_path()):
                image_group.append(a)
            else:
                flush_images()
                a.draw(renderer)
        flush_images()


def _resample(
        image_obj, data, out_shape, transform, *, resample=None, alpha=1):
    """
    Convenience wrapper around `._image.resample` to resample *data* to
    *out_shape* (with a third dimension if *data* is RGBA) that takes care of
    allocating the output array and fetching the relevant properties from the
    Image object *image_obj*.
    """
    # AGG can only handle coordinates smaller than 24-bit signed integers,
    # so raise errors if the input data is larger than _image.resample can
    # handle.
    msg = ('Data with more than {n} cannot be accurately displayed. '
           'Downsampling to less than {n} before displaying. '
           'To remove this warning, manually downsample your data.')
    if data.shape[1] > 2**23:
        warnings.warn(msg.format(n='2**23 columns'))
        step = int(np.ceil(data.shape[1] / 2**23))
        data = data[:, ::step]
        transform = Affine2D().scale(step, 1) + transform
    if data.shape[0] > 2**24:
        warnings.warn(msg.format(n='2**24 rows'))
        step = int(np.ceil(data.shape[0] / 2**24))
        data = data[::step, :]
        transform = Affine2D().scale(1, step) + transform
    # decide if we need to apply anti-aliasing if the data is upsampled:
    # compare the number of displayed pixels to the number of
    # the data pixels.
    interpolation = image_obj.get_interpolation()
    if interpolation in ['antialiased', 'auto']:
        # don't antialias if upsampling by an integer number or
        # if zooming in more than a factor of 3
        pos = np.array([[0, 0], [data.shape[1], data.shape[0]]])
        disp = transform.transform(pos)
        dispx = np.abs(np.diff(disp[:, 0]))
        dispy = np.abs(np.diff(disp[:, 1]))
        if ((dispx > 3 * data.shape[1] or
                dispx == data.shape[1] or
                dispx == 2 * data.shape[1]) and
            (dispy > 3 * data.shape[0] or
                dispy == data.shape[0] or
                dispy == 2 * data.shape[0])):
            interpolation = 'nearest'
        else:
            interpolation = 'hanning'
    out = np.zeros(out_shape + data.shape[2:], data.dtype)  # 2D->2D, 3D->3D.
    if resample is None:
        resample = image_obj.get_resample()
    _image.resample(data, out, transform,
                    _interpd_[interpolation],
                    resample,
                    alpha,
                    image_obj.get_filternorm(),
                    image_obj.get_filterrad())
    return out


def _rgb_to_rgba(A):
    """
    Convert an RGB image to RGBA, as required by the image resample C++
    extension.
    """
    rgba = np.zeros((A.shape[0], A.shape[1], 4), dtype=A.dtype)
    rgba[:, :, :3] = A
    if rgba.dtype == np.uint8:
        rgba[:, :, 3] = 255
    else:
        rgba[:, :, 3] = 1.0
    return rgba


class _ImageBase(mcolorizer.ColorizingArtist):
    """
    Base class for images.

    interpolation and cmap default to their rc settings

    cmap is a colors.Colormap instance
    norm is a colors.Normalize instance to map luminance to 0-1

    extent is data axes (left, right, bottom, top) for making image plots
    registered with data plots.  Default is to label the pixel
    centers with the zero-based row and column indices.

    Additional kwargs are matplotlib.artist properties
    """
    zorder = 0

    def __init__(self, ax,
                 cmap=None,
                 norm=None,
                 colorizer=None,
                 interpolation=None,
                 origin=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 *,
                 interpolation_stage=None,
                 **kwargs
                 ):
        super().__init__(self._get_colorizer(cmap, norm, colorizer))
        if origin is None:
            origin = mpl.rcParams['image.origin']
        _api.check_in_list(["upper", "lower"], origin=origin)
        self.origin = origin
        self.set_filternorm(filternorm)
        self.set_filterrad(filterrad)
        self.set_interpolation(interpolation)
        self.set_interpolation_stage(interpolation_stage)
        self.set_resample(resample)
        self.axes = ax

        self._imcache = None

        self._internal_update(kwargs)

    def __str__(self):
        try:
            shape = self.get_shape()
            return f"{type(self).__name__}(shape={shape!r})"
        except RuntimeError:
            return type(self).__name__

    def __getstate__(self):
        # Save some space on the pickle by not saving the cache.
        return {**super().__getstate__(), "_imcache": None}

    def get_size(self):
        """Return the size of the image as tuple (numrows, numcols)."""
        return self.get_shape()[:2]

    def get_shape(self):
        """
        Return the shape of the image as tuple (numrows, numcols, channels).
        """
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : float or 2D array-like or None
        """
        martist.Artist._set_alpha_for_array(self, alpha)
        if np.ndim(alpha) not in (0, 2):
            raise TypeError('alpha must be a float, two-dimensional '
                            'array, or None')
        self._imcache = None

    def _get_scalar_alpha(self):
        """
        Get a scalar alpha value to be applied to the artist as a whole.

        If the alpha value is a matrix, the method returns 1.0 because pixels
        have individual alpha values (see `~._ImageBase._make_image` for
        details). If the alpha value is a scalar, the method returns said value
        to be applied to the artist as a whole because pixels do not have
        individual alpha values.
        """
        return 1.0 if self._alpha is None or np.ndim(self._alpha) > 0 \
            else self._alpha

    def changed(self):
        """
        Call this whenever the mappable is changed so observers can update.
        """
        self._imcache = None
        super().changed()

    def _make_image(self, A, in_bbox, out_bbox, clip_bbox, magnification=1.0,
                    unsampled=False, round_to_pixel_border=True):
        """
        Normalize, rescale, and colormap the image *A* from the given *in_bbox*
        (in data space), to the given *out_bbox* (in pixel space) clipped to
        the given *clip_bbox* (also in pixel space), and magnified by the
        *magnification* factor.

        Parameters
        ----------
        A : ndarray

            - a (M, N) array interpreted as scalar (greyscale) image,
              with one of the dtypes `~numpy.float32`, `~numpy.float64`,
              `~numpy.float128`, `~numpy.uint16` or `~numpy.uint8`.
            - (M, N, 4) RGBA image with a dtype of `~numpy.float32`,
              `~numpy.float64`, `~numpy.float128`, or `~numpy.uint8`.

        in_bbox : `~matplotlib.transforms.Bbox`

        out_bbox : `~matplotlib.transforms.Bbox`

        clip_bbox : `~matplotlib.transforms.Bbox`

        magnification : float, default: 1

        unsampled : bool, default: False
            If True, the image will not be scaled, but an appropriate
            affine transformation will be returned instead.

        round_to_pixel_border : bool, default: True
            If True, the output image size will be rounded to the nearest pixel
            boundary.  This makes the images align correctly with the Axes.
            It should not be used if exact scaling is needed, such as for
            `.FigureImage`.

        Returns
        -------
        image : (M, N, 4) `numpy.uint8` array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : `~matplotlib.transforms.Affine2D`
            The affine transformation from image to pixel space.
        """
        if A is None:
            raise RuntimeError('You must first set the image '
                               'array or the image attribute')
        if A.size == 0:
            raise RuntimeError("_make_image must get a non-empty image. "
                               "Your Artist's draw method must filter before "
                               "this method is called.")

        clipped_bbox = Bbox.intersection(out_bbox, clip_bbox)

        if clipped_bbox is None:
            return None, 0, 0, None

        out_width_base = clipped_bbox.width * magnification
        out_height_base = clipped_bbox.height * magnification

        if out_width_base == 0 or out_height_base == 0:
            return None, 0, 0, None

        if self.origin == 'upper':
            # Flip the input image using a transform.  This avoids the
            # problem with flipping the array, which results in a copy
            # when it is converted to contiguous in the C wrapper
            t0 = Affine2D().translate(0, -A.shape[0]).scale(1, -1)
        else:
            t0 = IdentityTransform()

        t0 += (
            Affine2D()
            .scale(
                in_bbox.width / A.shape[1],
                in_bbox.height / A.shape[0])
            .translate(in_bbox.x0, in_bbox.y0)
            + self.get_transform())

        t = (t0
             + (Affine2D()
                .translate(-clipped_bbox.x0, -clipped_bbox.y0)
                .scale(magnification)))

        # So that the image is aligned with the edge of the Axes, we want to
        # round up the output width to the next integer.  This also means
        # scaling the transform slightly to account for the extra subpixel.
        if ((not unsampled) and t.is_affine and round_to_pixel_border and
                (out_width_base % 1.0 != 0.0 or out_height_base % 1.0 != 0.0)):
            out_width = math.ceil(out_width_base)
            out_height = math.ceil(out_height_base)
            extra_width = (out_width - out_width_base) / out_width_base
            extra_height = (out_height - out_height_base) / out_height_base
            t += Affine2D().scale(1.0 + extra_width, 1.0 + extra_height)
        else:
            out_width = int(out_width_base)
            out_height = int(out_height_base)
        out_shape = (out_height, out_width)

        if not unsampled:
            if not (A.ndim == 2 or A.ndim == 3 and A.shape[-1] in (3, 4)):
                raise ValueError(f"Invalid shape {A.shape} for image data")

            # if antialiased, this needs to change as window sizes
            # change:
            interpolation_stage = self._interpolation_stage
            if interpolation_stage in ['antialiased', 'auto']:
                pos = np.array([[0, 0], [A.shape[1], A.shape[0]]])
                disp = t.transform(pos)
                dispx = np.abs(np.diff(disp[:, 0])) / A.shape[1]
                dispy = np.abs(np.diff(disp[:, 1])) / A.shape[0]
                if (dispx < 3) or (dispy < 3):
                    interpolation_stage = 'rgba'
                else:
                    interpolation_stage = 'data'

            if A.ndim == 2 and interpolation_stage == 'data':
                # if we are a 2D array, then we are running through the
                # norm + colormap transformation.  However, in general the
                # input data is not going to match the size on the screen so we
                # have to resample to the correct number of pixels

                if A.dtype.kind == 'f':  # Float dtype: scale to same dtype.
                    scaled_dtype = np.dtype("f8" if A.dtype.itemsize > 4 else "f4")
                    if scaled_dtype.itemsize < A.dtype.itemsize:
                        _api.warn_external(f"Casting input data from {A.dtype}"
                                           f" to {scaled_dtype} for imshow.")
                else:  # Int dtype, likely.
                    # TODO slice input array first
                    # Scale to appropriately sized float: use float32 if the
                    # dynamic range is small, to limit the memory footprint.
                    da = A.max().astype("f8") - A.min().astype("f8")
                    scaled_dtype = "f8" if da > 1e8 else "f4"

                # resample the input data to the correct resolution and shape
                A_resampled = _resample(self, A.astype(scaled_dtype), out_shape, t)

                # if using NoNorm, cast back to the original datatype
                if isinstance(self.norm, mcolors.NoNorm):
                    A_resampled = A_resampled.astype(A.dtype)

                # Compute out_mask (what screen pixels include "bad" data
                # pixels) and out_alpha (to what extent screen pixels are
                # covered by data pixels: 0 outside the data extent, 1 inside
                # (even for bad data), and intermediate values at the edges).
                mask = (np.where(A.mask, np.float32(np.nan), np.float32(1))
                        if A.mask.shape == A.shape  # nontrivial mask
                        else np.ones_like(A, np.float32))
                # we always have to interpolate the mask to account for
                # non-affine transformations
                out_alpha = _resample(self, mask, out_shape, t, resample=True)
                del mask  # Make sure we don't use mask anymore!
                out_mask = np.isnan(out_alpha)
                out_alpha[out_mask] = 1
                # Apply the pixel-by-pixel alpha values if present
                alpha = self.get_alpha()
                if alpha is not None and np.ndim(alpha) > 0:
                    out_alpha *= _resample(self, alpha, out_shape, t, resample=True)
                # mask and run through the norm
                resampled_masked = np.ma.masked_array(A_resampled, out_mask)
                output = self.norm(resampled_masked)
            else:
                if A.ndim == 2:  # interpolation_stage = 'rgba'
                    self.norm.autoscale_None(A)
                    A = self.to_rgba(A)
                alpha = self.get_alpha()
                if alpha is None:  # alpha parameter not specified
                    if A.shape[2] == 3:  # image has no alpha channel
                        output_alpha = 255 if A.dtype == np.uint8 else 1.0
                    else:
                        output_alpha = _resample(  # resample alpha channel
                            self, A[..., 3], out_shape, t)
                    output = _resample(  # resample rgb channels
                        self, _rgb_to_rgba(A[..., :3]), out_shape, t)
                elif np.ndim(alpha) > 0:  # Array alpha
                    # user-specified array alpha overrides the existing alpha channel
                    output_alpha = _resample(self, alpha, out_shape, t)
                    output = _resample(
                        self, _rgb_to_rgba(A[..., :3]), out_shape, t)
                else:  # Scalar alpha
                    if A.shape[2] == 3:  # broadcast scalar alpha
                        output_alpha = (255 * alpha) if A.dtype == np.uint8 else alpha
                    else:  # or apply scalar alpha to existing alpha channel
                        output_alpha = _resample(self, A[..., 3], out_shape, t) * alpha
                    output = _resample(
                        self, _rgb_to_rgba(A[..., :3]), out_shape, t)
                output[..., 3] = output_alpha  # recombine rgb and alpha

            # output is now either a 2D array of normed (int or float) data
            # or an RGBA array of re-sampled input
            output = self.to_rgba(output, bytes=True, norm=False)
            # output is now a correctly sized RGBA array of uint8

            # Apply alpha *after* if the input was greyscale without a mask
            if A.ndim == 2:
                alpha = self._get_scalar_alpha()
                alpha_channel = output[:, :, 3]
                alpha_channel[:] = (  # Assignment will cast to uint8.
                    alpha_channel.astype(np.float32) * out_alpha * alpha)

        else:
            if self._imcache is None:
                self._imcache = self.to_rgba(A, bytes=True, norm=(A.ndim == 2))
            output = self._imcache

            # Subset the input image to only the part that will be displayed.
            subset = TransformedBbox(clip_bbox, t0.inverted()).frozen()
            output = output[
                int(max(subset.ymin, 0)):
                int(min(subset.ymax + 1, output.shape[0])),
                int(max(subset.xmin, 0)):
                int(min(subset.xmax + 1, output.shape[1]))]

            t = Affine2D().translate(
                int(max(subset.xmin, 0)), int(max(subset.ymin, 0))) + t

        return output, clipped_bbox.x0, clipped_bbox.y0, t

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        """
        Normalize, rescale, and colormap this image's data for rendering using
        *renderer*, with the given *magnification*.

        If *unsampled* is True, the image will not be scaled, but an
        appropriate affine transformation will be returned instead.

        Returns
        -------
        image : (M, N, 4) `numpy.uint8` array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : `~matplotlib.transforms.Affine2D`
            The affine transformation from image to pixel space.
        """
        raise NotImplementedError('The make_image method must be overridden')

    def _check_unsampled_image(self):
        """
        Return whether the image is better to be drawn unsampled.

        The derived class needs to override it.
        """
        return False

    @martist.allow_rasterization
    def draw(self, renderer):
        # if not visible, declare victory and return
        if not self.get_visible():
            self.stale = False
            return
        # for empty images, there is nothing to draw!
        if self.get_array().size == 0:
            self.stale = False
            return
        # actually render the image.
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_alpha(self._get_scalar_alpha())
        gc.set_url(self.get_url())
        gc.set_gid(self.get_gid())
        if (renderer.option_scale_image()  # Renderer supports transform kwarg.
                and self._check_unsampled_image()
                and self.get_transform().is_affine):
            im, l, b, trans = self.make_image(renderer, unsampled=True)
            if im is not None:
                trans = Affine2D().scale(im.shape[1], im.shape[0]) + trans
                renderer.draw_image(gc, l, b, im, trans)
        else:
            im, l, b, trans = self.make_image(
                renderer, renderer.get_image_magnification())
            if im is not None:
                renderer.draw_image(gc, l, b, im)
        gc.restore()
        self.stale = False

    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
        if (self._different_canvas(mouseevent)
                # This doesn't work for figimage.
                or not self.axes.contains(mouseevent)[0]):
            return False, {}
        # TODO: make sure this is consistent with patch and patch
        # collection on nonlinear transformed coordinates.
        # TODO: consider returning image coordinates (shouldn't
        # be too difficult given that the image is rectilinear
        trans = self.get_transform().inverted()
        x, y = trans.transform([mouseevent.x, mouseevent.y])
        xmin, xmax, ymin, ymax = self.get_extent()
        # This checks xmin <= x <= xmax *or* xmax <= x <= xmin.
        inside = (x is not None and (x - xmin) * (x - xmax) <= 0
                  and y is not None and (y - ymin) * (y - ymax) <= 0)
        return inside, {}

    def write_png(self, fname):
        """Write the image to png file *fname*."""
        im = self.to_rgba(self._A[::-1] if self.origin == 'lower' else self._A,
                          bytes=True, norm=True)
        PIL.Image.fromarray(im).save(fname, format="png")

    @staticmethod
    def _normalize_image_array(A):
        """
        Check validity of image-like input *A* and normalize it to a format suitable for
        Image subclasses.
        """
        A = cbook.safe_masked_invalid(A, copy=True)
        if A.dtype != np.uint8 and not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            f"converted to float")
        if A.ndim == 3 and A.shape[-1] == 1:
            A = A.squeeze(-1)  # If just (M, N, 1), assume scalar and apply colormap.
        if not (A.ndim == 2 or A.ndim == 3 and A.shape[-1] in [3, 4]):
            raise TypeError(f"Invalid shape {A.shape} for image data")
        if A.ndim == 3:
            # If the input data has values outside the valid range (after
            # normalisation), we issue a warning and then clip X to the bounds
            # - otherwise casting wraps extreme values, hiding outliers and
            # making reliable interpretation impossible.
            high = 255 if np.issubdtype(A.dtype, np.integer) else 1
            if A.min() < 0 or high < A.max():
                _log.warning(
                    'Clipping input data to the valid range for imshow with '
                    'RGB data ([0..1] for floats or [0..255] for integers). '
                    'Got range [%s..%s].',
                    A.min(), A.max()
                )
                A = np.clip(A, 0, high)
            # Cast unsupported integer types to uint8
            if A.dtype != np.uint8 and np.issubdtype(A.dtype, np.integer):
                A = A.astype(np.uint8)
        return A

    def set_data(self, A):
        """
        Set the image array.

        Note that this function does *not* update the normalization used.

        Parameters
        ----------
        A : array-like or `PIL.Image.Image`
        """
        if isinstance(A, PIL.Image.Image):
            A = pil_to_array(A)  # Needed e.g. to apply png palette.
        self._A = self._normalize_image_array(A)
        self._imcache = None
        self.stale = True

    def set_array(self, A):
        """
        Retained for backwards compatibility - use set_data instead.

        Parameters
        ----------
        A : array-like
        """
        # This also needs to be here to override the inherited
        # cm.ScalarMappable.set_array method so it is not invoked by mistake.
        self.set_data(A)

    def get_interpolation(self):
        """
        Return the interpolation method the image uses when resizing.

        One of 'auto', 'antialiased', 'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos',
        or 'none'.
        """
        return self._interpolation

    def set_interpolation(self, s):
        """
        Set the interpolation method the image uses when resizing.

        If None, use :rc:`image.interpolation`. If 'none', the image is
        shown as is without interpolating. 'none' is only supported in
        agg, ps and pdf backends and will fall back to 'nearest' mode
        for other backends.

        Parameters
        ----------
        s : {'auto', 'nearest', 'bilinear', 'bicubic', 'spline16', \
'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', \
'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'none'} or None
        """
        s = mpl._val_or_rc(s, 'image.interpolation').lower()
        _api.check_in_list(interpolations_names, interpolation=s)
        self._interpolation = s
        self.stale = True

    def get_interpolation_stage(self):
        """
        Return when interpolation happens during the transform to RGBA.

        One of 'data', 'rgba', 'auto'.
        """
        return self._interpolation_stage

    def set_interpolation_stage(self, s):
        """
        Set when interpolation happens during the transform to RGBA.

        Parameters
        ----------
        s : {'data', 'rgba', 'auto'} or None
            Whether to apply up/downsampling interpolation in data or RGBA
            space.  If None, use :rc:`image.interpolation_stage`.
            If 'auto' we will check upsampling rate and if less
            than 3 then use 'rgba', otherwise use 'data'.
        """
        s = mpl._val_or_rc(s, 'image.interpolation_stage')
        _api.check_in_list(['data', 'rgba', 'auto'], s=s)
        self._interpolation_stage = s
        self.stale = True

    def can_composite(self):
        """Return whether the image can be composited with its neighbors."""
        trans = self.get_transform()
        return (
            self._interpolation != 'none' and
            trans.is_affine and
            trans.is_separable)

    def set_resample(self, v):
        """
        Set whether image resampling is used.

        Parameters
        ----------
        v : bool or None
            If None, use :rc:`image.resample`.
        """
        v = mpl._val_or_rc(v, 'image.resample')
        self._resample = v
        self.stale = True

    def get_resample(self):
        """Return whether image resampling is used."""
        return self._resample

    def set_filternorm(self, filternorm):
        """
        Set whether the resize filter normalizes the weights.

        See help for `~.Axes.imshow`.

        Parameters
        ----------
        filternorm : bool
        """
        self._filternorm = bool(filternorm)
        self.stale = True

    def get_filternorm(self):
        """Return whether the resize filter normalizes the weights."""
        return self._filternorm

    def set_filterrad(self, filterrad):
        """
        Set the resize filter radius only applicable to some
        interpolation schemes -- see help for imshow

        Parameters
        ----------
        filterrad : positive float
        """
        r = float(filterrad)
        if r <= 0:
            raise ValueError("The filter radius must be a positive number")
        self._filterrad = r
        self.stale = True

    def get_filterrad(self):
        """Return the filterrad setting."""
        return self._filterrad


class AxesImage(_ImageBase):
    """
    An image with pixels on a regular grid, attached to an Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes the image will belong to.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar
        data to colors.
    norm : str or `~matplotlib.colors.Normalize`
        Maps luminance to 0-1.
    interpolation : str, default: :rc:`image.interpolation`
        Supported values are 'none', 'auto', 'nearest', 'bilinear',
        'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
        'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
        'sinc', 'lanczos', 'blackman'.
    interpolation_stage : {'data', 'rgba'}, default: 'data'
        If 'data', interpolation
        is carried out on the data provided by the user.  If 'rgba', the
        interpolation is carried out after the colormapping has been
        applied (visual interpolation).
    origin : {'upper', 'lower'}, default: :rc:`image.origin`
        Place the [0, 0] index of the array in the upper left or lower left
        corner of the Axes. The convention 'upper' is typically used for
        matrices and images.
    extent : tuple, optional
        The data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.
    filternorm : bool, default: True
        A parameter for the antigrain image resize filter
        (see the antigrain documentation).
        If filternorm is set, the filter normalizes integer values and corrects
        the rounding errors. It doesn't do anything with the source floating
        point values, it corrects only integers according to the rule of 1.0
        which means that any sum of pixel weights must be equal to 1.0. So,
        the filter function must produce a graph of the proper shape.
    filterrad : float > 0, default: 4
        The filter radius for filters that have a radius parameter, i.e. when
        interpolation is one of: 'sinc', 'lanczos' or 'blackman'.
    resample : bool, default: False
        When True, use a full resampling method. When False, only resample when
        the output image is larger than the input image.
    **kwargs : `~matplotlib.artist.Artist` properties
    """

    def __init__(self, ax,
                 *,
                 cmap=None,
                 norm=None,
                 colorizer=None,
                 interpolation=None,
                 origin=None,
                 extent=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 interpolation_stage=None,
                 **kwargs
                 ):

        self._extent = extent

        super().__init__(
            ax,
            cmap=cmap,
            norm=norm,
            colorizer=colorizer,
            interpolation=interpolation,
            origin=origin,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            interpolation_stage=interpolation_stage,
            **kwargs
        )

    def get_window_extent(self, renderer=None):
        x0, x1, y0, y1 = self._extent
        bbox = Bbox.from_extents([x0, y0, x1, y1])
        return bbox.transformed(self.get_transform())

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        trans = self.get_transform()
        # image is created in the canvas coordinate.
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        clip = ((self.get_clip_box() or self.axes.bbox) if self.get_clip_on()
                else self.get_figure(root=True).bbox)
        return self._make_image(self._A, bbox, transformed_bbox, clip,
                                magnification, unsampled=unsampled)

    def _check_unsampled_image(self):
        """Return whether the image would be better drawn unsampled."""
        return self.get_interpolation() == "none"

    def set_extent(self, extent, **kwargs):
        """
        Set the image extent.

        Parameters
        ----------
        extent : 4-tuple of float
            The position and size of the image as tuple
            ``(left, right, bottom, top)`` in data coordinates.
        **kwargs
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar Axes) entries are applied, if present.

        Notes
        -----
        This updates `.Axes.dataLim`, and, if autoscaling, sets `.Axes.viewLim`
        to tightly fit the image, regardless of `~.Axes.dataLim`.  Autoscaling
        state is not changed, so a subsequent call to `.Axes.autoscale_view`
        will redo the autoscaling in accord with `~.Axes.dataLim`.
        """
        (xmin, xmax), (ymin, ymax) = self.axes._process_unit_info(
            [("x", [extent[0], extent[1]]),
             ("y", [extent[2], extent[3]])],
            kwargs)
        if kwargs:
            raise _api.kwarg_error("set_extent", kwargs)
        xmin = self.axes._validate_converted_limits(
            xmin, self.convert_xunits)
        xmax = self.axes._validate_converted_limits(
            xmax, self.convert_xunits)
        ymin = self.axes._validate_converted_limits(
            ymin, self.convert_yunits)
        ymax = self.axes._validate_converted_limits(
            ymax, self.convert_yunits)
        extent = [xmin, xmax, ymin, ymax]

        self._extent = extent
        corners = (xmin, ymin), (xmax, ymax)
        self.axes.update_datalim(corners)
        self.sticky_edges.x[:] = [xmin, xmax]
        self.sticky_edges.y[:] = [ymin, ymax]
        if self.axes.get_autoscalex_on():
            self.axes.set_xlim((xmin, xmax), auto=None)
        if self.axes.get_autoscaley_on():
            self.axes.set_ylim((ymin, ymax), auto=None)
        self.stale = True

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
        if self._extent is not None:
            return self._extent
        else:
            sz = self.get_size()
            numrows, numcols = sz
            if self.origin == 'upper':
                return (-0.5, numcols-0.5, numrows-0.5, -0.5)
            else:
                return (-0.5, numcols-0.5, -0.5, numrows-0.5)

    def get_cursor_data(self, event):
        """
        Return the image value at the event position or *None* if the event is
        outside the image.

        See Also
        --------
        matplotlib.artist.Artist.get_cursor_data
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == 'upper':
            ymin, ymax = ymax, ymin
        arr = self.get_array()
        data_extent = Bbox([[xmin, ymin], [xmax, ymax]])
        array_extent = Bbox([[0, 0], [arr.shape[1], arr.shape[0]]])
        trans = self.get_transform().inverted()
        trans += BboxTransform(boxin=data_extent, boxout=array_extent)
        point = trans.transform([event.x, event.y])
        if any(np.isnan(point)):
            return None
        j, i = point.astype(int)
        # Clip the coordinates at array bounds
        if not (0 <= i < arr.shape[0]) or not (0 <= j < arr.shape[1]):
            return None
        else:
            return arr[i, j]


class NonUniformImage(AxesImage):
    """
    An image with pixels on a rectilinear grid.

    In contrast to `.AxesImage`, where pixels are on a regular grid,
    NonUniformImage allows rows and columns with individual heights / widths.

    See also :doc:`/gallery/images_contours_and_fields/image_nonuniform`.
    """

    def __init__(self, ax, *, interpolation='nearest', **kwargs):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes the image will belong to.
        interpolation : {'nearest', 'bilinear'}, default: 'nearest'
            The interpolation scheme used in the resampling.
        **kwargs
            All other keyword arguments are identical to those of `.AxesImage`.
        """
        super().__init__(ax, **kwargs)
        self.set_interpolation(interpolation)

    def _check_unsampled_image(self):
        """Return False. Do not use unsampled image."""
        return False

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        if unsampled:
            raise ValueError('unsampled not supported on NonUniformImage')
        A = self._A
        if A.ndim == 2:
            if A.dtype != np.uint8:
                A = self.to_rgba(A, bytes=True)
            else:
                A = np.repeat(A[:, :, np.newaxis], 4, 2)
                A[:, :, 3] = 255
        else:
            if A.dtype != np.uint8:
                A = (255*A).astype(np.uint8)
            if A.shape[2] == 3:
                B = np.zeros(tuple([*A.shape[0:2], 4]), np.uint8)
                B[:, :, 0:3] = A
                B[:, :, 3] = 255
                A = B
        l, b, r, t = self.axes.bbox.extents
        width = int(((round(r) + 0.5) - (round(l) - 0.5)) * magnification)
        height = int(((round(t) + 0.5) - (round(b) - 0.5)) * magnification)

        invertedTransform = self.axes.transData.inverted()
        x_pix = invertedTransform.transform(
            [(x, b) for x in np.linspace(l, r, width)])[:, 0]
        y_pix = invertedTransform.transform(
            [(l, y) for y in np.linspace(b, t, height)])[:, 1]

        if self._interpolation == "nearest":
            x_mid = (self._Ax[:-1] + self._Ax[1:]) / 2
            y_mid = (self._Ay[:-1] + self._Ay[1:]) / 2
            x_int = x_mid.searchsorted(x_pix)
            y_int = y_mid.searchsorted(y_pix)
            # The following is equal to `A[y_int[:, None], x_int[None, :]]`,
            # but many times faster.  Both casting to uint32 (to have an
            # effectively 1D array) and manual index flattening matter.
            im = (
                np.ascontiguousarray(A).view(np.uint32).ravel()[
                    np.add.outer(y_int * A.shape[1], x_int)]
                .view(np.uint8).reshape((height, width, 4)))
        else:  # self._interpolation == "bilinear"
            # Use np.interp to compute x_int/x_float has similar speed.
            x_int = np.clip(
                self._Ax.searchsorted(x_pix) - 1, 0, len(self._Ax) - 2)
            y_int = np.clip(
                self._Ay.searchsorted(y_pix) - 1, 0, len(self._Ay) - 2)
            idx_int = np.add.outer(y_int * A.shape[1], x_int)
            x_frac = np.clip(
                np.divide(x_pix - self._Ax[x_int], np.diff(self._Ax)[x_int],
                          dtype=np.float32),  # Downcasting helps with speed.
                0, 1)
            y_frac = np.clip(
                np.divide(y_pix - self._Ay[y_int], np.diff(self._Ay)[y_int],
                          dtype=np.float32),
                0, 1)
            f00 = np.outer(1 - y_frac, 1 - x_frac)
            f10 = np.outer(y_frac, 1 - x_frac)
            f01 = np.outer(1 - y_frac, x_frac)
            f11 = np.outer(y_frac, x_frac)
            im = np.empty((height, width, 4), np.uint8)
            for chan in range(4):
                ac = A[:, :, chan].reshape(-1)  # reshape(-1) avoids a copy.
                # Shifting the buffer start (`ac[offset:]`) avoids an array
                # addition (`ac[idx_int + offset]`).
                buf = f00 * ac[idx_int]
                buf += f10 * ac[A.shape[1]:][idx_int]
                buf += f01 * ac[1:][idx_int]
                buf += f11 * ac[A.shape[1] + 1:][idx_int]
                im[:, :, chan] = buf  # Implicitly casts to uint8.
        return im, l, b, IdentityTransform()

    def set_data(self, x, y, A):
        """
        Set the grid for the pixel centers, and the pixel values.

        Parameters
        ----------
        x, y : 1D array-like
            Monotonic arrays of shapes (N,) and (M,), respectively, specifying
            pixel centers.
        A : array-like
            (M, N) `~numpy.ndarray` or masked array of values to be
            colormapped, or (M, N, 3) RGB array, or (M, N, 4) RGBA array.
        """
        A = self._normalize_image_array(A)
        x = np.array(x, np.float32)
        y = np.array(y, np.float32)
        if not (x.ndim == y.ndim == 1 and A.shape[:2] == y.shape + x.shape):
            raise TypeError("Axes don't match array shape")
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None
        self.stale = True

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_interpolation(self, s):
        """
        Parameters
        ----------
        s : {'nearest', 'bilinear'} or None
            If None, use :rc:`image.interpolation`.
        """
        if s is not None and s not in ('nearest', 'bilinear'):
            raise NotImplementedError('Only nearest neighbor and '
                                      'bilinear interpolations are supported')
        super().set_interpolation(s)

    def get_extent(self):
        if self._A is None:
            raise RuntimeError('Must set data first')
        return self._Ax[0], self._Ax[-1], self._Ay[0], self._Ay[-1]

    def set_filternorm(self, filternorm):
        pass

    def set_filterrad(self, filterrad):
        pass

    def set_norm(self, norm):
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        super().set_norm(norm)

    def set_cmap(self, cmap):
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        super().set_cmap(cmap)

    def get_cursor_data(self, event):
        # docstring inherited
        x, y = event.xdata, event.ydata
        if (x < self._Ax[0] or x > self._Ax[-1] or
                y < self._Ay[0] or y > self._Ay[-1]):
            return None
        j = np.searchsorted(self._Ax, x) - 1
        i = np.searchsorted(self._Ay, y) - 1
        return self._A[i, j]


class PcolorImage(AxesImage):
    """
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    """

    def __init__(self, ax,
                 x=None,
                 y=None,
                 A=None,
                 *,
                 cmap=None,
                 norm=None,
                 colorizer=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes the image will belong to.
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map
            scalar data to colors.
        norm : str or `~matplotlib.colors.Normalize`
            Maps luminance to 0-1.
        **kwargs : `~matplotlib.artist.Artist` properties
        """
        super().__init__(ax, norm=norm, cmap=cmap, colorizer=colorizer)
        self._internal_update(kwargs)
        if A is not None:
            self.set_data(x, y, A)

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        if unsampled:
            raise ValueError('unsampled not supported on PColorImage')

        if self._imcache is None:
            A = self.to_rgba(self._A, bytes=True)
            self._imcache = np.pad(A, [(1, 1), (1, 1), (0, 0)], "constant")
        padded_A = self._imcache
        bg = mcolors.to_rgba(self.axes.patch.get_facecolor(), 0)
        bg = (np.array(bg) * 255).astype(np.uint8)
        if (padded_A[0, 0] != bg).all():
            padded_A[[0, -1], :] = padded_A[:, [0, -1]] = bg

        l, b, r, t = self.axes.bbox.extents
        width = (round(r) + 0.5) - (round(l) - 0.5)
        height = (round(t) + 0.5) - (round(b) - 0.5)
        width = round(width * magnification)
        height = round(height * magnification)
        vl = self.axes.viewLim

        x_pix = np.linspace(vl.x0, vl.x1, width)
        y_pix = np.linspace(vl.y0, vl.y1, height)
        x_int = self._Ax.searchsorted(x_pix)
        y_int = self._Ay.searchsorted(y_pix)
        im = (  # See comment in NonUniformImage.make_image re: performance.
            padded_A.view(np.uint32).ravel()[
                np.add.outer(y_int * padded_A.shape[1], x_int)]
            .view(np.uint8).reshape((height, width, 4)))
        return im, l, b, IdentityTransform()

    def _check_unsampled_image(self):
        return False

    def set_data(self, x, y, A):
        """
        Set the grid for the rectangle boundaries, and the data values.

        Parameters
        ----------
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array
        """
        A = self._normalize_image_array(A)
        x = np.arange(0., A.shape[1] + 1) if x is None else np.array(x, float).ravel()
        y = np.arange(0., A.shape[0] + 1) if y is None else np.array(y, float).ravel()
        if A.shape[:2] != (y.size - 1, x.size - 1):
            raise ValueError(
                "Axes don't match array shape. Got %s, expected %s." %
                (A.shape[:2], (y.size - 1, x.size - 1)))
        # For efficient cursor readout, ensure x and y are increasing.
        if x[-1] < x[0]:
            x = x[::-1]
            A = A[:, ::-1]
        if y[-1] < y[0]:
            y = y[::-1]
            A = A[::-1]
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None
        self.stale = True

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def get_cursor_data(self, event):
        # docstring inherited
        x, y = event.xdata, event.ydata
        if (x < self._Ax[0] or x > self._Ax[-1] or
                y < self._Ay[0] or y > self._Ay[-1]):
            return None
        j = np.searchsorted(self._Ax, x) - 1
        i = np.searchsorted(self._Ay, y) - 1
        return self._A[i, j]


class FigureImage(_ImageBase):
    """An image attached to a figure."""

    zorder = 0

    _interpolation = 'nearest'

    def __init__(self, fig,
                 *,
                 cmap=None,
                 norm=None,
                 colorizer=None,
                 offsetx=0,
                 offsety=0,
                 origin=None,
                 **kwargs
                 ):
        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
        super().__init__(
            None,
            norm=norm,
            cmap=cmap,
            colorizer=colorizer,
            origin=origin
        )
        self.set_figure(fig)
        self.ox = offsetx
        self.oy = offsety
        self._internal_update(kwargs)
        self.magnification = 1.0

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
        numrows, numcols = self.get_size()
        return (-0.5 + self.ox, numcols-0.5 + self.ox,
                -0.5 + self.oy, numrows-0.5 + self.oy)

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        fig = self.get_figure(root=True)
        fac = renderer.dpi/fig.dpi
        # fac here is to account for pdf, eps, svg backends where
        # figure.dpi is set to 72.  This means we need to scale the
        # image (using magnification) and offset it appropriately.
        bbox = Bbox([[self.ox/fac, self.oy/fac],
                     [(self.ox/fac + self._A.shape[1]),
                     (self.oy/fac + self._A.shape[0])]])
        width, height = fig.get_size_inches()
        width *= renderer.dpi
        height *= renderer.dpi
        clip = Bbox([[0, 0], [width, height]])
        return self._make_image(
            self._A, bbox, bbox, clip, magnification=magnification / fac,
            unsampled=unsampled, round_to_pixel_border=False)

    def set_data(self, A):
        """Set the image array."""
        super().set_data(A)
        self.stale = True


class BboxImage(_ImageBase):
    """The Image class whose size is determined by the given bbox."""

    def __init__(self, bbox,
                 *,
                 cmap=None,
                 norm=None,
                 colorizer=None,
                 interpolation=None,
                 origin=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 **kwargs
                 ):
        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
        super().__init__(
            None,
            cmap=cmap,
            norm=norm,
            colorizer=colorizer,
            interpolation=interpolation,
            origin=origin,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            **kwargs
        )
        self.bbox = bbox

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure()._get_renderer()

        if isinstance(self.bbox, BboxBase):
            return self.bbox
        elif callable(self.bbox):
            return self.bbox(renderer)
        else:
            raise ValueError("Unknown type of bbox")

    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
        if self._different_canvas(mouseevent) or not self.get_visible():
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        inside = self.get_window_extent().contains(x, y)
        return inside, {}

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        width, height = renderer.get_canvas_width_height()
        bbox_in = self.get_window_extent(renderer).frozen()
        bbox_in._points /= [width, height]
        bbox_out = self.get_window_extent(renderer)
        clip = Bbox([[0, 0], [width, height]])
        self._transform = BboxTransformTo(clip)
        return self._make_image(
            self._A,
            bbox_in, bbox_out, clip, magnification, unsampled=unsampled)


def imread(fname, format=None):
    """
    Read an image from a file into an array.

    .. note::

        This function exists for historical reasons.  It is recommended to
        use `PIL.Image.open` instead for loading images.

    Parameters
    ----------
    fname : str or file-like
        The image file to read: a filename, a URL or a file-like object opened
        in read-binary mode.

        Passing a URL is deprecated.  Please open the URL
        for reading and pass the result to Pillow, e.g. with
        ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.
    format : str, optional
        The image file format assumed for reading the data.  The image is
        loaded as a PNG file if *format* is set to "png", if *fname* is a path
        or opened file with a ".png" extension, or if it is a URL.  In all
        other cases, *format* is ignored and the format is auto-detected by
        `PIL.Image.open`.

    Returns
    -------
    `numpy.array`
        The image data. The returned array has shape

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.

        PNG images are returned as float arrays (0-1).  All other formats are
        returned as int arrays, with a bit depth determined by the file's
        contents.
    """
    # hide imports to speed initial import on systems with slow linkers
    from urllib import parse

    if format is None:
        if isinstance(fname, str):
            parsed = parse.urlparse(fname)
            # If the string is a URL (Windows paths appear as if they have a
            # length-1 scheme), assume png.
            if len(parsed.scheme) > 1:
                ext = 'png'
            else:
                ext = Path(fname).suffix.lower()[1:]
        elif hasattr(fname, 'geturl'):  # Returned by urlopen().
            # We could try to parse the url's path and use the extension, but
            # returning png is consistent with the block above.  Note that this
            # if clause has to come before checking for fname.name as
            # urlopen("file:///...") also has a name attribute (with the fixed
            # value "<urllib response>").
            ext = 'png'
        elif hasattr(fname, 'name'):
            ext = Path(fname.name).suffix.lower()[1:]
        else:
            ext = 'png'
    else:
        ext = format
    img_open = (
        PIL.PngImagePlugin.PngImageFile if ext == 'png' else PIL.Image.open)
    if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
        # Pillow doesn't handle URLs directly.
        raise ValueError(
            "Please open the URL for reading and pass the "
            "result to Pillow, e.g. with "
            "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
            )
    with img_open(fname) as image:
        return (_pil_png_to_float_array(image)
                if isinstance(image, PIL.PngImagePlugin.PngImageFile) else
                pil_to_array(image))


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
           origin=None, dpi=100, *, metadata=None, pil_kwargs=None):
    """
    Colormap and save an array as an image file.

    RGB(A) images are passed through.  Single channel images will be
    colormapped according to *cmap* and *norm*.

    .. note::

       If you want to save a single channel image as gray scale please use an
       image I/O library (such as pillow, tifffile, or imageio) directly.

    Parameters
    ----------
    fname : str or path-like or file-like
        A path or a file-like object to store the image in.
        If *format* is not set, then the output format is inferred from the
        extension of *fname*, if any, and from :rc:`savefig.format` otherwise.
        If *format* is set, it determines the output format.
    arr : array-like
        The image data. Accepts NumPy arrays or sequences
        (e.g., lists or tuples). The shape can be one of
        MxN (luminance), MxNx3 (RGB) or MxNx4 (RGBA).
    vmin, vmax : float, optional
        *vmin* and *vmax* set the color scaling for the image by fixing the
        values that map to the colormap color limits. If either *vmin*
        or *vmax* is None, that limit is determined from the *arr*
        min/max value.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        A Colormap instance or registered colormap name. The colormap
        maps scalar data to colors. It is ignored for RGB(A) data.
    format : str, optional
        The file format, e.g. 'png', 'pdf', 'svg', ...  The behavior when this
        is unset is documented under *fname*.
    origin : {'upper', 'lower'}, default: :rc:`image.origin`
        Indicates whether the ``(0, 0)`` index of the array is in the upper
        left or lower left corner of the Axes.
    dpi : float
        The DPI to store in the metadata of the file.  This does not affect the
        resolution of the output image.  Depending on file format, this may be
        rounded to the nearest integer.
    metadata : dict, optional
        Metadata in the image file.  The supported keys depend on the output
        format, see the documentation of the respective backends for more
        information.
        Currently only supported for "png", "pdf", "ps", "eps", and "svg".
    pil_kwargs : dict, optional
        Keyword arguments passed to `PIL.Image.Image.save`.  If the 'pnginfo'
        key is present, it completely overrides *metadata*, including the
        default 'Software' key.
    """
    from matplotlib.figure import Figure

    # Normalizing input (e.g., list or tuples) to NumPy array if needed
    arr = np.asanyarray(arr)

    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if format is None:
        format = (Path(fname).suffix[1:] if isinstance(fname, str)
                  else mpl.rcParams["savefig.format"]).lower()
    if format in ["pdf", "ps", "eps", "svg"]:
        # Vector formats that are not handled by PIL.
        if pil_kwargs is not None:
            raise ValueError(
                f"Cannot use 'pil_kwargs' when saving to {format}")
        fig = Figure(dpi=dpi, frameon=False)
        fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                     resize=True)
        fig.savefig(fname, dpi=dpi, format=format, transparent=True,
                    metadata=metadata)
    else:
        # Don't bother creating an image; this avoids rounding errors on the
        # size when dividing and then multiplying by dpi.
        if origin is None:
            origin = mpl.rcParams["image.origin"]
        else:
            _api.check_in_list(('upper', 'lower'), origin=origin)
        if origin == "lower":
            arr = arr[::-1]
        if (isinstance(arr, memoryview) and arr.format == "B"
                and arr.ndim == 3 and arr.shape[-1] == 4):
            # Such an ``arr`` would also be handled fine by sm.to_rgba below
            # (after casting with asarray), but it is useful to special-case it
            # because that's what backend_agg passes, and can be in fact used
            # as is, saving a few operations.
            rgba = arr
        else:
            sm = mcolorizer.Colorizer(cmap=cmap)
            sm.set_clim(vmin, vmax)
            rgba = sm.to_rgba(arr, bytes=True)
        if pil_kwargs is None:
            pil_kwargs = {}
        else:
            # we modify this below, so make a copy (don't modify caller's dict)
            pil_kwargs = pil_kwargs.copy()
        pil_shape = (rgba.shape[1], rgba.shape[0])
        rgba = np.require(rgba, requirements='C')
        image = PIL.Image.frombuffer(
            "RGBA", pil_shape, rgba, "raw", "RGBA", 0, 1)
        if format == "png":
            # Only use the metadata kwarg if pnginfo is not set, because the
            # semantics of duplicate keys in pnginfo is unclear.
            if "pnginfo" in pil_kwargs:
                if metadata:
                    _api.warn_external("'metadata' is overridden by the "
                                       "'pnginfo' entry in 'pil_kwargs'.")
            else:
                metadata = {
                    "Software": (f"Matplotlib version{mpl.__version__}, "
                                 f"https://matplotlib.org/"),
                    **(metadata if metadata is not None else {}),
                }
                pil_kwargs["pnginfo"] = pnginfo = PIL.PngImagePlugin.PngInfo()
                for k, v in metadata.items():
                    if v is not None:
                        pnginfo.add_text(k, v)
        elif metadata is not None:
            raise ValueError(f"metadata not supported for format {format!r}")
        if format in ["jpg", "jpeg"]:
            format = "jpeg"  # Pillow doesn't recognize "jpg".
            facecolor = mpl.rcParams["savefig.facecolor"]
            if cbook._str_equal(facecolor, "auto"):
                facecolor = mpl.rcParams["figure.facecolor"]
            color = tuple(int(x * 255) for x in mcolors.to_rgb(facecolor))
            background = PIL.Image.new("RGB", pil_shape, color)
            background.paste(image, image)
            image = background
        pil_kwargs.setdefault("format", format)
        pil_kwargs.setdefault("dpi", (dpi, dpi))
        image.save(fname, **pil_kwargs)


def pil_to_array(pilImage):
    """
    Load a `PIL image`_ and return it as a numpy int array.

    .. _PIL image: https://pillow.readthedocs.io/en/latest/reference/Image.html

    Returns
    -------
    numpy.array

        The array shape depends on the image type:

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.
    """
    if pilImage.mode in ['RGBA', 'RGBX', 'RGB', 'L']:
        # return MxNx4 RGBA, MxNx3 RBA, or MxN luminance array
        return np.asarray(pilImage)
    elif pilImage.mode.startswith('I;16'):
        # return MxN luminance array of uint16
        raw = pilImage.tobytes('raw', pilImage.mode)
        if pilImage.mode.endswith('B'):
            x = np.frombuffer(raw, '>u2')
        else:
            x = np.frombuffer(raw, '<u2')
        return x.reshape(pilImage.size[::-1]).astype('=u2')
    else:  # try to convert to an rgba image
        try:
            pilImage = pilImage.convert('RGBA')
        except ValueError as err:
            raise RuntimeError('Unknown image mode') from err
        return np.asarray(pilImage)  # return MxNx4 RGBA array


def _pil_png_to_float_array(pil_png):
    """Convert a PIL `PNGImageFile` to a 0-1 float array."""
    # Unlike pil_to_array this converts to 0-1 float32s for backcompat with the
    # old libpng-based loader.
    # The supported rawmodes are from PIL.PngImagePlugin._MODES.  When
    # mode == "RGB(A)", the 16-bit raw data has already been coarsened to 8-bit
    # by Pillow.
    mode = pil_png.mode
    rawmode = pil_png.png.im_rawmode
    if rawmode == "1":  # Grayscale.
        return np.asarray(pil_png, np.float32)
    if rawmode == "L;2":  # Grayscale.
        return np.divide(pil_png, 2**2 - 1, dtype=np.float32)
    if rawmode == "L;4":  # Grayscale.
        return np.divide(pil_png, 2**4 - 1, dtype=np.float32)
    if rawmode == "L":  # Grayscale.
        return np.divide(pil_png, 2**8 - 1, dtype=np.float32)
    if rawmode == "I;16B":  # Grayscale.
        return np.divide(pil_png, 2**16 - 1, dtype=np.float32)
    if mode == "RGB":  # RGB.
        return np.divide(pil_png, 2**8 - 1, dtype=np.float32)
    if mode == "P":  # Palette.
        return np.divide(pil_png.convert("RGBA"), 2**8 - 1, dtype=np.float32)
    if mode == "LA":  # Grayscale + alpha.
        return np.divide(pil_png.convert("RGBA"), 2**8 - 1, dtype=np.float32)
    if mode == "RGBA":  # RGBA.
        return np.divide(pil_png, 2**8 - 1, dtype=np.float32)
    raise ValueError(f"Unknown PIL rawmode: {rawmode}")


def thumbnail(infile, thumbfile, scale=0.1, interpolation='bilinear',
              preview=False):
    """
    Make a thumbnail of image in *infile* with output filename *thumbfile*.

    See :doc:`/gallery/misc/image_thumbnail_sgskip`.

    Parameters
    ----------
    infile : str or file-like
        The image file. Matplotlib relies on Pillow_ for image reading, and
        thus supports a wide range of file formats, including PNG, JPG, TIFF
        and others.

        .. _Pillow: https://python-pillow.github.io

    thumbfile : str or file-like
        The thumbnail filename.

    scale : float, default: 0.1
        The scale factor for the thumbnail.

    interpolation : str, default: 'bilinear'
        The interpolation scheme used in the resampling. See the
        *interpolation* parameter of `~.Axes.imshow` for possible values.

    preview : bool, default: False
        If True, the default backend (presumably a user interface
        backend) will be used which will cause a figure to be raised if
        `~matplotlib.pyplot.show` is called.  If it is False, the figure is
        created using `.FigureCanvasBase` and the drawing backend is selected
        as `.Figure.savefig` would normally do.

    Returns
    -------
    `.Figure`
        The figure instance containing the thumbnail.
    """

    im = imread(infile)
    rows, cols, depth = im.shape

    # This doesn't really matter (it cancels in the end) but the API needs it.
    dpi = 100

    height = rows / dpi * scale
    width = cols / dpi * scale

    if preview:
        # Let the UI backend do everything.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(width, height), dpi=dpi)
    else:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasBase(fig)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])
    ax.imshow(im, aspect='auto', resample=True, interpolation=interpolation)
    fig.savefig(thumbfile, dpi=dpi)
    return fig
<|endoftext|>"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""
from collections.abc import Hashable, Sequence
import pathlib
from typing import Any, Callable, Literal, TypeAlias, TypeVar, Union

from . import path
from ._enums import JoinStyle, CapStyle
from .artist import Artist
from .backend_bases import RendererBase
from .markers import MarkerStyle
from .transforms import Bbox, Transform

RGBColorType: TypeAlias = tuple[float, float, float] | str
RGBAColorType: TypeAlias = (
    str |  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float] |
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float] |
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float]
)

ColorType: TypeAlias = RGBColorType | RGBAColorType

RGBColourType: TypeAlias = RGBColorType
RGBAColourType: TypeAlias = RGBAColorType
ColourType: TypeAlias = ColorType

LineStyleType: TypeAlias = str | tuple[float, Sequence[float]]
DrawStyleType: TypeAlias = Literal["default", "steps", "steps-pre", "steps-mid",
                                   "steps-post"]
MarkEveryType: TypeAlias = (
    None |
    int | tuple[int, int] | slice | list[int] |
    float | tuple[float, float] |
    list[bool]
)

MarkerType: TypeAlias = str | path.Path | MarkerStyle
FillStyleType: TypeAlias = Literal["full", "left", "right", "bottom", "top", "none"]
JoinStyleType: TypeAlias = JoinStyle | Literal["miter", "round", "bevel"]
CapStyleType: TypeAlias = CapStyle | Literal["butt", "projecting", "round"]

CoordsBaseType = Union[
    str,
    Artist,
    Transform,
    Callable[
        [RendererBase],
        Union[Bbox, Transform]
    ]
]
CoordsType = Union[
    CoordsBaseType,
    tuple[CoordsBaseType, CoordsBaseType]
]

RcStyleType: TypeAlias = (
    str |
    dict[str, Any] |
    pathlib.Path |
    Sequence[str | pathlib.Path | dict[str, Any]]
)

_HT = TypeVar("_HT", bound=Hashable)
HashableList: TypeAlias = list[_HT | "HashableList[_HT]"]
"""A nested list of Hashable values."""
<|endoftext|>"""
Implementation details for :mod:`.mathtext`.
"""

from __future__ import annotations

import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple

import numpy as np
from pyparsing import (
    Empty, Forward, Literal, Group, NotAny, OneOrMore, Optional,
    ParseBaseException, ParseException, ParseExpression, ParseFatalException,
    ParserElement, ParseResults, QuotedString, Regex, StringEnd, ZeroOrMore,
    pyparsing_common, nested_expr, one_of)

import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
    latex_to_bakoma, stix_glyph_fixes, stix_virtual_fonts, tex2uni)
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, Kerning, LoadFlags


if T.TYPE_CHECKING:
    from collections.abc import Iterable
    from .ft2font import Glyph

ParserElement.enable_packrat()
_log = logging.getLogger("matplotlib.mathtext")


##############################################################################
# FONTS


def get_unicode_index(symbol: str) -> int:  # Publicly exported.
    r"""
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\pi') or a Type1
        symbol name (e.g. 'phi').
    """
    try:  # This will succeed if symbol is a single Unicode char
        return ord(symbol)
    except TypeError:
        pass
    try:  # Is symbol a TeX symbol (i.e. \alpha)
        return tex2uni[symbol.strip("\\")]
    except KeyError as err:
        raise ValueError(
            f"{symbol!r} is not a valid Unicode character or TeX/Type1 symbol"
            ) from err


class VectorParse(NamedTuple):
    """
    The namedtuple type returned by ``MathTextParser("path").parse(...)``.

    Attributes
    ----------
    width, height, depth : float
        The global metrics.
    glyphs : list
        The glyphs including their positions.
    rect : list
        The list of rectangles.
    """
    width: float
    height: float
    depth: float
    glyphs: list[tuple[FT2Font, float, int, float, float]]
    rects: list[tuple[float, float, float, float]]

VectorParse.__module__ = "matplotlib.mathtext"


class RasterParse(NamedTuple):
    """
    The namedtuple type returned by ``MathTextParser("agg").parse(...)``.

    Attributes
    ----------
    ox, oy : float
        The offsets are always zero.
    width, height, depth : float
        The global metrics.
    image : FT2Image
        A raster image.
    """
    ox: float
    oy: float
    width: float
    height: float
    depth: float
    image: FT2Image

RasterParse.__module__ = "matplotlib.mathtext"


class Output:
    r"""
    Result of `ship`\ping a box: lists of positioned glyphs and rectangles.

    This class is not exposed to end users, but converted to a `VectorParse` or
    a `RasterParse` by `.MathTextParser.parse`.
    """

    def __init__(self, box: Box):
        self.box = box
        self.glyphs: list[tuple[float, float, FontInfo]] = []  # (ox, oy, info)
        self.rects: list[tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)

    def to_vector(self) -> VectorParse:
        w, h, d = map(
            np.ceil, [self.box.width, self.box.height, self.box.depth])
        gs = [(info.font, info.fontsize, info.num, ox, h - oy + info.offset)
              for ox, oy, info in self.glyphs]
        rs = [(x1, h - y2, x2 - x1, y2 - y1)
              for x1, y1, x2, y2 in self.rects]
        return VectorParse(w, h + d, d, gs, rs)

    def to_raster(self, *, antialiased: bool) -> RasterParse:
        # Metrics y's and mathtext y's are oriented in opposite directions,
        # hence the switch between ymin and ymax.
        xmin = min([*[ox + info.metrics.xmin for ox, oy, info in self.glyphs],
                    *[x1 for x1, y1, x2, y2 in self.rects], 0]) - 1
        ymin = min([*[oy - info.metrics.ymax for ox, oy, info in self.glyphs],
                    *[y1 for x1, y1, x2, y2 in self.rects], 0]) - 1
        xmax = max([*[ox + info.metrics.xmax for ox, oy, info in self.glyphs],
                    *[x2 for x1, y1, x2, y2 in self.rects], 0]) + 1
        ymax = max([*[oy - info.metrics.ymin for ox, oy, info in self.glyphs],
                    *[y2 for x1, y1, x2, y2 in self.rects], 0]) + 1
        w = xmax - xmin
        h = ymax - ymin - self.box.depth
        d = ymax - ymin - self.box.height
        image = FT2Image(int(np.ceil(w)), int(np.ceil(h + max(d, 0))))

        # Ideally, we could just use self.glyphs and self.rects here, shifting
        # their coordinates by (-xmin, -ymin), but this yields slightly
        # different results due to floating point slop; shipping twice is the
        # old approach and keeps baseline images backcompat.
        shifted = ship(self.box, (-xmin, -ymin))

        for ox, oy, info in shifted.glyphs:
            info.font.draw_glyph_to_bitmap(
                image, int(ox), int(oy - info.metrics.iceberg), info.glyph,
                antialiased=antialiased)
        for x1, y1, x2, y2 in shifted.rects:
            height = max(int(y2 - y1) - 1, 0)
            if height == 0:
                center = (y2 + y1) / 2
                y = int(center - (height + 1) / 2)
            else:
                y = int(y1)
            image.draw_rect_filled(int(x1), y, int(np.ceil(x2)), y + height)
        return RasterParse(0, 0, w, h + d, d, image)


class FontMetrics(NamedTuple):
    """
    Metrics of a font.

    Attributes
    ----------
    advance : float
        The advance distance (in points) of the glyph.
    height : float
        The height of the glyph in points.
    width : float
        The width of the glyph in points.
    xmin, xmax, ymin, ymax : float
        The ink rectangle of the glyph.
    iceberg : float
        The distance from the baseline to the top of the glyph. (This corresponds to
        TeX's definition of "height".)
    slanted : bool
        Whether the glyph should be considered as "slanted" (currently used for kerning
        sub/superscripts).
    """
    advance: float
    height: float
    width: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    iceberg: float
    slanted: bool


class FontInfo(NamedTuple):
    font: FT2Font
    fontsize: float
    postscript_name: str
    metrics: FontMetrics
    num: int
    glyph: Glyph
    offset: float


class Fonts(abc.ABC):
    """
    An abstract base class for a system of fonts to use for mathtext.

    The class must be able to take symbol keys and font file names and
    return the character metrics.  It also delegates to a backend class
    to do the actual drawing.
    """

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        """
        Parameters
        ----------
        default_font_prop : `~.font_manager.FontProperties`
            The default non-math font, or the base font for Unicode (generic)
            font rendering.
        load_glyph_flags : `.ft2font.LoadFlags`
            Flags passed to the glyph loader (e.g. ``FT_Load_Glyph`` and
            ``FT_Load_Char`` for FreeType-based fonts).
        """
        self.default_font_prop = default_font_prop
        self.load_glyph_flags = load_glyph_flags

    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float,
                 font2: str, fontclass2: str, sym2: str, fontsize2: float,
                 dpi: float) -> float:
        """
        Get the kerning distance for font between *sym1* and *sym2*.

        See `~.Fonts.get_metrics` for a detailed description of the parameters.
        """
        return 0.

    def _get_font(self, font: str) -> FT2Font:
        raise NotImplementedError

    def _get_info(self, font: str, font_class: str, sym: str, fontsize: float,
                  dpi: float) -> FontInfo:
        raise NotImplementedError

    def get_metrics(self, font: str, font_class: str, sym: str, fontsize: float,
                    dpi: float) -> FontMetrics:
        r"""
        Parameters
        ----------
        font : str
            One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
            "default", "regular", "bb", "frak", "scr".  "default" and "regular"
            are synonyms and use the non-math font.
        font_class : str
            One of the TeX font names (as for *font*), but **not** "bb",
            "frak", or "scr".  This is used to combine two font classes.  The
            only supported combination currently is ``get_metrics("frak", "bf",
            ...)``.
        sym : str
            A symbol in raw TeX form, e.g., "1", "x", or "\sigma".
        fontsize : float
            Font size in points.
        dpi : float
            Rendering dots-per-inch.

        Returns
        -------
        FontMetrics
        """
        info = self._get_info(font, font_class, sym, fontsize, dpi)
        return info.metrics

    def render_glyph(self, output: Output, ox: float, oy: float, font: str,
                     font_class: str, sym: str, fontsize: float, dpi: float) -> None:
        """
        At position (*ox*, *oy*), draw the glyph specified by the remaining
        parameters (see `get_metrics` for their detailed description).
        """
        info = self._get_info(font, font_class, sym, fontsize, dpi)
        output.glyphs.append((ox, oy, info))

    def render_rect_filled(self, output: Output,
                           x1: float, y1: float, x2: float, y2: float) -> None:
        """
        Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        output.rects.append((x1, y1, x2, y2))

    def get_xheight(self, font: str, fontsize: float, dpi: float) -> float:
        """
        Get the xheight for the given *font* and *fontsize*.
        """
        raise NotImplementedError()

    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        """
        Get the line thickness that matches the given font.  Used as a
        base unit for drawing lines such as in a fraction or radical.
        """
        raise NotImplementedError()

    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        """
        Override if your font provides multiple sizes of the same
        symbol.  Should return a list of symbols matching *sym* in
        various sizes.  The expression renderer will select the most
        appropriate size for a given situation from this list.
        """
        return [(fontname, sym)]


class TruetypeFonts(Fonts, metaclass=abc.ABCMeta):
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        super().__init__(default_font_prop, load_glyph_flags)
        # Per-instance cache.
        self._get_info = functools.cache(self._get_info)  # type: ignore[method-assign]
        self._fonts = {}
        self.fontmap: dict[str, str] = {}

        filename = findfont(self.default_font_prop)
        default_font = get_font(filename)
        self._fonts['default'] = default_font
        self._fonts['regular'] = default_font

    def _get_font(self, font: str) -> FT2Font:
        basename = self.fontmap.get(font, font)
        cached_font = self._fonts.get(basename)
        if cached_font is None and os.path.exists(basename):
            cached_font = get_font(basename)
            self._fonts[basename] = cached_font
            self._fonts[cached_font.postscript_name] = cached_font
            self._fonts[cached_font.postscript_name.lower()] = cached_font
        return T.cast(FT2Font, cached_font)  # FIXME: Not sure this is guaranteed.

    def _get_offset(self, font: FT2Font, glyph: Glyph, fontsize: float,
                    dpi: float) -> float:
        if font.postscript_name == 'Cmex10':
            return (glyph.height / 64 / 2) + (fontsize/3 * dpi/72)
        return 0.

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        raise NotImplementedError

    # The return value of _get_info is cached per-instance.
    def _get_info(self, fontname: str, font_class: str, sym: str, fontsize: float,
                  dpi: float) -> FontInfo:
        font, num, slanted = self._get_glyph(fontname, font_class, sym)
        font.set_size(fontsize, dpi)
        glyph = font.load_char(num, flags=self.load_glyph_flags)

        xmin, ymin, xmax, ymax = (val / 64 for val in glyph.bbox)
        offset = self._get_offset(font, glyph, fontsize, dpi)
        metrics = FontMetrics(
            advance=glyph.linearHoriAdvance / 65536,
            height=glyph.height / 64,
            width=glyph.width / 64,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin + offset,
            ymax=ymax + offset,
            # iceberg is the equivalent of TeX's "height"
            iceberg=glyph.horiBearingY / 64 + offset,
            slanted=slanted
        )

        return FontInfo(
            font=font,
            fontsize=fontsize,
            postscript_name=font.postscript_name,
            metrics=metrics,
            num=num,
            glyph=glyph,
            offset=offset
        )

    def get_xheight(self, fontname: str, fontsize: float, dpi: float) -> float:
        font = self._get_font(fontname)
        font.set_size(fontsize, dpi)
        pclt = font.get_sfnt_table('pclt')
        if pclt is None:
            # Some fonts don't store the xHeight, so we do a poor man's xHeight
            metrics = self.get_metrics(
                fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi)
            return metrics.iceberg
        xHeight = (pclt['xHeight'] / 64.0) * (fontsize / 12.0) * (dpi / 100.0)
        return xHeight

    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        # This function used to grab underline thickness from the font
        # metrics, but that information is just too un-reliable, so it
        # is now hardcoded.
        return ((0.75 / 12.0) * fontsize * dpi) / 72.0

    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float,
                 font2: str, fontclass2: str, sym2: str, fontsize2: float,
                 dpi: float) -> float:
        if font1 == font2 and fontsize1 == fontsize2:
            info1 = self._get_info(font1, fontclass1, sym1, fontsize1, dpi)
            info2 = self._get_info(font2, fontclass2, sym2, fontsize2, dpi)
            font = info1.font
            return font.get_kerning(info1.num, info2.num, Kerning.DEFAULT) / 64
        return super().get_kern(font1, fontclass1, sym1, fontsize1,
                                font2, fontclass2, sym2, fontsize2, dpi)


class BakomaFonts(TruetypeFonts):
    """
    Use the Bakoma TrueType fonts for rendering.

    Symbols are strewn about a number of font files, each of which has
    its own proprietary 8-bit encoding.
    """
    _fontmap = {
        'cal': 'cmsy10',
        'rm':  'cmr10',
        'tt':  'cmtt10',
        'it':  'cmmi10',
        'bf':  'cmb10',
        'sf':  'cmss10',
        'ex':  'cmex10',
    }

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        self._stix_fallback = StixFonts(default_font_prop, load_glyph_flags)

        super().__init__(default_font_prop, load_glyph_flags)
        for key, val in self._fontmap.items():
            fullpath = findfont(val)
            self.fontmap[key] = fullpath
            self.fontmap[val] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        font = None
        if fontname in self.fontmap and sym in latex_to_bakoma:
            basename, num = latex_to_bakoma[sym]
            slanted = (basename == "cmmi10") or sym in self._slanted_symbols
            font = self._get_font(basename)
        elif len(sym) == 1:
            slanted = (fontname == "it")
            font = self._get_font(fontname)
            if font is not None:
                num = ord(sym)
        if font is not None and font.get_char_index(num) != 0:
            return font, num, slanted
        else:
            return self._stix_fallback._get_glyph(fontname, font_class, sym)

    # The Bakoma fonts contain many pre-sized alternatives for the
    # delimiters.  The AutoSizedChar class will use these alternatives
    # and select the best (closest sized) glyph.
    _size_alternatives = {
        '(':           [('rm', '('), ('ex', '\xa1'), ('ex', '\xb3'),
                        ('ex', '\xb5'), ('ex', '\xc3')],
        ')':           [('rm', ')'), ('ex', '\xa2'), ('ex', '\xb4'),
                        ('ex', '\xb6'), ('ex', '\x21')],
        '{':           [('cal', '{'), ('ex', '\xa9'), ('ex', '\x6e'),
                        ('ex', '\xbd'), ('ex', '\x28')],
        '}':           [('cal', '}'), ('ex', '\xaa'), ('ex', '\x6f'),
                        ('ex', '\xbe'), ('ex', '\x29')],
        # The fourth size of '[' is mysteriously missing from the BaKoMa
        # font, so I've omitted it for both '[' and ']'
        '[':           [('rm', '['), ('ex', '\xa3'), ('ex', '\x68'),
                        ('ex', '\x22')],
        ']':           [('rm', ']'), ('ex', '\xa4'), ('ex', '\x69'),
                        ('ex', '\x23')],
        r'\lfloor':    [('ex', '\xa5'), ('ex', '\x6a'),
                        ('ex', '\xb9'), ('ex', '\x24')],
        r'\rfloor':    [('ex', '\xa6'), ('ex', '\x6b'),
                        ('ex', '\xba'), ('ex', '\x25')],
        r'\lceil':     [('ex', '\xa7'), ('ex', '\x6c'),
                        ('ex', '\xbb'), ('ex', '\x26')],
        r'\rceil':     [('ex', '\xa8'), ('ex', '\x6d'),
                        ('ex', '\xbc'), ('ex', '\x27')],
        r'\langle':    [('ex', '\xad'), ('ex', '\x44'),
                        ('ex', '\xbf'), ('ex', '\x2a')],
        r'\rangle':    [('ex', '\xae'), ('ex', '\x45'),
                        ('ex', '\xc0'), ('ex', '\x2b')],
        r'\__sqrt__':  [('ex', '\x70'), ('ex', '\x71'),
                        ('ex', '\x72'), ('ex', '\x73')],
        r'\backslash': [('ex', '\xb2'), ('ex', '\x2f'),
                        ('ex', '\xc2'), ('ex', '\x2d')],
        r'/':          [('rm', '/'), ('ex', '\xb1'), ('ex', '\x2e'),
                        ('ex', '\xcb'), ('ex', '\x2c')],
        r'\widehat':   [('rm', '\x5e'), ('ex', '\x62'), ('ex', '\x63'),
                        ('ex', '\x64')],
        r'\widetilde': [('rm', '\x7e'), ('ex', '\x65'), ('ex', '\x66'),
                        ('ex', '\x67')],
        r'<':          [('cal', 'h'), ('ex', 'D')],
        r'>':          [('cal', 'i'), ('ex', 'E')]
        }

    for alias, target in [(r'\leftparen', '('),
                          (r'\rightparen', ')'),
                          (r'\leftbrace', '{'),
                          (r'\rightbrace', '}'),
                          (r'\leftbracket', '['),
                          (r'\rightbracket', ']'),
                          (r'\{', '{'),
                          (r'\}', '}'),
                          (r'\[', '['),
                          (r'\]', ']')]:
        _size_alternatives[alias] = _size_alternatives[target]

    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        return self._size_alternatives.get(sym, [(fontname, sym)])


class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol cannot be found in the font.
    """

    # Some glyphs are not present in the `cmr10` font, and must be brought in
    # from `cmsy10`. Map the Unicode indices of those glyphs to the indices at
    # which they are found in `cmsy10`.
    _cmr10_substitutions = {
        0x00D7: 0x00A3,  # Multiplication sign.
        0x2212: 0x00A1,  # Minus sign.
    }

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        # This must come first so the backend's owner is set correctly
        fallback_rc = mpl.rcParams['mathtext.fallback']
        font_cls: type[TruetypeFonts] | None = {
            'stix': StixFonts,
            'stixsans': StixSansFonts,
            'cm': BakomaFonts
        }.get(fallback_rc)
        self._fallback_font = (font_cls(default_font_prop, load_glyph_flags)
                               if font_cls else None)

        super().__init__(default_font_prop, load_glyph_flags)
        for texfont in "cal rm tt it bf sf bfit".split():
            prop = mpl.rcParams['mathtext.' + texfont]
            font = findfont(prop)
            self.fontmap[texfont] = font
        prop = FontProperties('cmex10')
        font = findfont(prop)
        self.fontmap['ex'] = font

        # include STIX sized alternatives for glyphs if fallback is STIX
        if isinstance(self._fallback_font, StixFonts):
            stixsizedaltfonts = {
                '0': 'STIXGeneral',
                '1': 'STIXSizeOneSym',
                '2': 'STIXSizeTwoSym',
                '3': 'STIXSizeThreeSym',
                '4': 'STIXSizeFourSym',
                '5': 'STIXSizeFiveSym',
            }

            for size, name in stixsizedaltfonts.items():
                fullpath = findfont(name)
                self.fontmap[size] = fullpath
                self.fontmap[name] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _map_virtual_font(self, fontname: str, font_class: str,
                          uniindex: int) -> tuple[str, int]:
        return fontname, uniindex

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        try:
            uniindex = get_unicode_index(sym)
            found_symbol = True
        except ValueError:
            uniindex = ord('?')
            found_symbol = False
            _log.warning("No TeX to Unicode mapping for %a.", sym)

        fontname, uniindex = self._map_virtual_font(
            fontname, font_class, uniindex)

        new_fontname = fontname

        # Only characters in the "Letter" class should be italicized in 'it'
        # mode.  Greek capital letters should be Roman.
        if found_symbol:
            if fontname == 'it' and uniindex < 0x10000:
                char = chr(uniindex)
                if (unicodedata.category(char)[0] != "L"
                        or unicodedata.name(char).startswith("GREEK CAPITAL")):
                    new_fontname = 'rm'

            slanted = (new_fontname == 'it') or sym in self._slanted_symbols
            found_symbol = False
            font = self._get_font(new_fontname)
            if font is not None:
                if (uniindex in self._cmr10_substitutions
                        and font.family_name == "cmr10"):
                    font = get_font(
                        cbook._get_data_path("fonts/ttf/cmsy10.ttf"))
                    uniindex = self._cmr10_substitutions[uniindex]
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    found_symbol = True

        if not found_symbol:
            if self._fallback_font:
                if (fontname in ('it', 'regular')
                        and isinstance(self._fallback_font, StixFonts)):
                    fontname = 'rm'

                g = self._fallback_font._get_glyph(fontname, font_class, sym)
                family = g[0].family_name
                if family in BakomaFonts._fontmap.values():
                    family = "Computer Modern"
                _log.info("Substituting symbol %s from %s", sym, family)
                return g

            else:
                if (fontname in ('it', 'regular')
                        and isinstance(self, StixFonts)):
                    return self._get_glyph('rm', font_class, sym)
                _log.warning("Font %r does not have a glyph for %a [U+%x], "
                             "substituting with a dummy symbol.",
                             new_fontname, sym, uniindex)
                font = self._get_font('rm')
                uniindex = 0xA4  # currency char, for lack of anything better
                slanted = False

        return font, uniindex, slanted

    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        if self._fallback_font:
            return self._fallback_font.get_sized_alternatives_for_symbol(fontname, sym)
        return [(fontname, sym)]


class DejaVuFonts(UnicodeFonts, metaclass=abc.ABCMeta):
    _fontmap: dict[str, str] = {}

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        # This must come first so the backend's owner is set correctly
        if isinstance(self, DejaVuSerifFonts):
            self._fallback_font = StixFonts(default_font_prop, load_glyph_flags)
        else:
            self._fallback_font = StixSansFonts(default_font_prop, load_glyph_flags)
        self.bakoma = BakomaFonts(default_font_prop, load_glyph_flags)
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        # Include Stix sized alternatives for glyphs
        self._fontmap.update({
            '1': 'STIXSizeOneSym',
            '2': 'STIXSizeTwoSym',
            '3': 'STIXSizeThreeSym',
            '4': 'STIXSizeFourSym',
            '5': 'STIXSizeFiveSym',
        })
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        # Override prime symbol to use Bakoma.
        if sym == r'\prime':
            return self.bakoma._get_glyph(fontname, font_class, sym)
        else:
            # check whether the glyph is available in the display font
            uniindex = get_unicode_index(sym)
            font = self._get_font('ex')
            if font is not None:
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    return super()._get_glyph('ex', font_class, sym)
            # otherwise return regular glyph
            return super()._get_glyph(fontname, font_class, sym)


class DejaVuSerifFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Serif fonts

    If a glyph is not found it will fallback to Stix Serif
    """
    _fontmap = {
        'rm': 'DejaVu Serif',
        'it': 'DejaVu Serif:italic',
        'bf': 'DejaVu Serif:weight=bold',
        'bfit': 'DejaVu Serif:italic:bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Serif Display',
        '0': 'DejaVu Serif',
    }


class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """
    _fontmap = {
        'rm': 'DejaVu Sans',
        'it': 'DejaVu Sans:italic',
        'bf': 'DejaVu Sans:weight=bold',
        'bfit': 'DejaVu Sans:italic:bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Sans Display',
        '0': 'DejaVu Sans',
    }


class StixFonts(UnicodeFonts):
    """
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:

    - supports "virtual fonts" which are complete alpha numeric
      character sets with different font styles at special Unicode
      code points, such as "Blackboard".

    - handles sized alternative characters for the STIXSizeX fonts.
    """
    _fontmap = {
        'rm': 'STIXGeneral',
        'it': 'STIXGeneral:italic',
        'bf': 'STIXGeneral:weight=bold',
        'bfit': 'STIXGeneral:italic:bold',
        'nonunirm': 'STIXNonUnicode',
        'nonuniit': 'STIXNonUnicode:italic',
        'nonunibf': 'STIXNonUnicode:weight=bold',
        '0': 'STIXGeneral',
        '1': 'STIXSizeOneSym',
        '2': 'STIXSizeTwoSym',
        '3': 'STIXSizeThreeSym',
        '4': 'STIXSizeFourSym',
        '5': 'STIXSizeFiveSym',
    }
    _fallback_font = None
    _sans = False

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: LoadFlags):
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _map_virtual_font(self, fontname: str, font_class: str,
                          uniindex: int) -> tuple[str, int]:
        # Handle these "fonts" that are actually embedded in
        # other fonts.
        font_mapping = stix_virtual_fonts.get(fontname)
        if (self._sans and font_mapping is None
                and fontname not in ('regular', 'default')):
            font_mapping = stix_virtual_fonts['sf']
            doing_sans_conversion = True
        else:
            doing_sans_conversion = False

        if isinstance(font_mapping, dict):
            try:
                mapping = font_mapping[font_class]
            except KeyError:
                mapping = font_mapping['rm']
        elif isinstance(font_mapping, list):
            mapping = font_mapping
        else:
            mapping = None

        if mapping is not None:
            # Binary search for the source glyph
            lo = 0
            hi = len(mapping)
            while lo < hi:
                mid = (lo+hi)//2
                range = mapping[mid]
                if uniindex < range[0]:
                    hi = mid
                elif uniindex <= range[1]:
                    break
                else:
                    lo = mid + 1

            if range[0] <= uniindex <= range[1]:
                uniindex = uniindex - range[0] + range[3]
                fontname = range[2]
            elif not doing_sans_conversion:
                # This will generate a dummy character
                uniindex = 0x1
                fontname = mpl.rcParams['mathtext.default']

        # Fix some incorrect glyphs.
        if fontname in ('rm', 'it'):
            uniindex = stix_glyph_fixes.get(uniindex, uniindex)

        # Handle private use area glyphs
        if fontname in ('it', 'rm', 'bf', 'bfit') and 0xe000 <= uniindex <= 0xf8ff:
            fontname = 'nonuni' + fontname

        return fontname, uniindex

    @functools.cache
    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        fixes = {
            '\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']',
            '<': '\N{MATHEMATICAL LEFT ANGLE BRACKET}',
            '>': '\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
        }
        sym = fixes.get(sym, sym)
        try:
            uniindex = get_unicode_index(sym)
        except ValueError:
            return [(fontname, sym)]
        alternatives = [(str(i), chr(uniindex)) for i in range(6)
                        if self._get_font(str(i)).get_char_index(uniindex) != 0]
        # The largest size of the radical symbol in STIX has incorrect
        # metrics that cause it to be disconnected from the stem.
        if sym == r'\__sqrt__':
            alternatives = alternatives[:-1]
        return alternatives


class StixSansFonts(StixFonts):
    """
    A font handling class for the STIX fonts (that uses sans-serif
    characters by default).
    """
    _sans = True


##############################################################################
# TeX-LIKE BOX MODEL

# The following is based directly on the document 'woven' from the
# TeX82 source code.  This information is also available in printed
# form:
#
#    Knuth, Donald E.. 1986.  Computers and Typesetting, Volume B:
#    TeX: The Program.  Addison-Wesley Professional.
#
# The most relevant "chapters" are:
#    Data structures for boxes and their friends
#    Shipping pages out (ship())
#    Packaging (hpack() and vpack())
#    Data structures for math mode
#    Subroutines for math mode
#    Typesetting math formulas
#
# Many of the docstrings below refer to a numbered "node" in that
# book, e.g., node123
#
# Note that (as TeX) y increases downward, unlike many other parts of
# matplotlib.

# How much text shrinks when going to the next-smallest level.
SHRINK_FACTOR   = 0.7
# The number of different sizes of chars to use, beyond which they will not
# get any smaller
NUM_SIZE_LEVELS = 6


class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """
    # Percentage of x-height of additional horiz. space after sub/superscripts
    script_space: T.ClassVar[float] = 0.05

    # Percentage of x-height that sub/superscripts drop below the baseline
    subdrop: T.ClassVar[float] = 0.4

    # Percentage of x-height that superscripts are raised from the baseline
    sup1: T.ClassVar[float] = 0.7

    # Percentage of x-height that subscripts drop below the baseline
    sub1: T.ClassVar[float] = 0.3

    # Percentage of x-height that subscripts drop below the baseline when a
    # superscript is present
    sub2: T.ClassVar[float] = 0.5

    # Percentage of x-height that sub/superscripts are offset relative to the
    # nucleus edge for non-slanted nuclei
    delta: T.ClassVar[float] = 0.025

    # Additional percentage of last character height above 2/3 of the
    # x-height that superscripts are offset relative to the subscript
    # for slanted nuclei
    delta_slanted: T.ClassVar[float] = 0.2

    # Percentage of x-height that superscripts and subscripts are offset for
    # integrals
    delta_integral: T.ClassVar[float] = 0.1


class ComputerModernFontConstants(FontConstantsBase):
    script_space = 0.075
    subdrop = 0.2
    sup1 = 0.45
    sub1 = 0.2
    sub2 = 0.3
    delta = 0.075
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXFontConstants(FontConstantsBase):
    script_space = 0.1
    sup1 = 0.8
    sub2 = 0.6
    delta = 0.05
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXSansFontConstants(FontConstantsBase):
    script_space = 0.05
    sup1 = 0.8
    delta_slanted = 0.6
    delta_integral = 0.3


class DejaVuSerifFontConstants(FontConstantsBase):
    pass


class DejaVuSansFontConstants(FontConstantsBase):
    pass


# Maps font family names to the FontConstantBase subclass to use
_font_constant_mapping = {
    'DejaVu Sans': DejaVuSansFontConstants,
    'DejaVu Sans Mono': DejaVuSansFontConstants,
    'DejaVu Serif': DejaVuSerifFontConstants,
    'cmb10': ComputerModernFontConstants,
    'cmex10': ComputerModernFontConstants,
    'cmmi10': ComputerModernFontConstants,
    'cmr10': ComputerModernFontConstants,
    'cmss10': ComputerModernFontConstants,
    'cmsy10': ComputerModernFontConstants,
    'cmtt10': ComputerModernFontConstants,
    'STIXGeneral': STIXFontConstants,
    'STIXNonUnicode': STIXFontConstants,
    'STIXSizeFiveSym': STIXFontConstants,
    'STIXSizeFourSym': STIXFontConstants,
    'STIXSizeThreeSym': STIXFontConstants,
    'STIXSizeTwoSym': STIXFontConstants,
    'STIXSizeOneSym': STIXFontConstants,
    # Map the fonts we used to ship, just for good measure
    'Bitstream Vera Sans': DejaVuSansFontConstants,
    'Bitstream Vera': DejaVuSansFontConstants,
    }


def _get_font_constant_set(state: ParserState) -> type[FontConstantsBase]:
    constants = _font_constant_mapping.get(
        state.fontset._get_font(state.font).family_name, FontConstantsBase)
    # STIX sans isn't really its own fonts, just different code points
    # in the STIX fonts, so we have to detect this one separately.
    if (constants is STIXFontConstants and
            isinstance(state.fontset, StixSansFonts)):
        return STIXSansFontConstants
    return constants


class Node:
    """A node in the TeX box model."""

    def __init__(self) -> None:
        self.size = 0

    def __repr__(self) -> str:
        return type(self).__name__

    def get_kerning(self, next: Node | None) -> float:
        return 0.0

    def shrink(self) -> None:
        """
        Shrinks one level smaller.  There are only three levels of
        sizes, after which things will no longer get smaller.
        """
        self.size += 1

    def render(self, output: Output, x: float, y: float) -> None:
        """Render this node."""


class Box(Node):
    """A node with a physical location."""

    def __init__(self, width: float, height: float, depth: float) -> None:
        super().__init__()
        self.width  = width
        self.height = height
        self.depth  = depth

    def shrink(self) -> None:
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width  *= SHRINK_FACTOR
            self.height *= SHRINK_FACTOR
            self.depth  *= SHRINK_FACTOR

    def render(self, output: Output,  # type: ignore[override]
               x1: float, y1: float, x2: float, y2: float) -> None:
        pass


class Vbox(Box):
    """A box with only height (zero width)."""

    def __init__(self, height: float, depth: float):
        super().__init__(0., height, depth)


class Hbox(Box):
    """A box with only width (zero height and depth)."""

    def __init__(self, width: float):
        super().__init__(width, 0., 0.)


class Char(Node):
    """
    A single character.

    Unlike TeX, the font information and metrics are stored with each `Char`
    to make it easier to lookup the font metrics when needed.  Note that TeX
    boxes have a width, height, and depth, unlike Type1 and TrueType which use
    a full bounding box and an advance in the x-direction.  The metrics must
    be converted to the TeX model, and the advance (if different from width)
    must be converted into a `Kern` node when the `Char` is added to its parent
    `Hlist`.
    """

    def __init__(self, c: str, state: ParserState):
        super().__init__()
        self.c = c
        self.fontset = state.fontset
        self.font = state.font
        self.font_class = state.font_class
        self.fontsize = state.fontsize
        self.dpi = state.dpi
        # The real width, height and depth will be set during the
        # pack phase, after we know the real fontsize
        self._update_metrics()

    def __repr__(self) -> str:
        return '`%s`' % self.c

    def _update_metrics(self) -> None:
        metrics = self._metrics = self.fontset.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
        if self.c == ' ':
            self.width = metrics.advance
        else:
            self.width = metrics.width
        self.height = metrics.iceberg
        self.depth = -(metrics.iceberg - metrics.height)

    def is_slanted(self) -> bool:
        return self._metrics.slanted

    def get_kerning(self, next: Node | None) -> float:
        """
        Return the amount of kerning between this and the given character.

        This method is called when characters are strung together into `Hlist`
        to create `Kern` nodes.
        """
        advance = self._metrics.advance - self.width
        kern = 0.
        if isinstance(next, Char):
            kern = self.fontset.get_kern(
                self.font, self.font_class, self.c, self.fontsize,
                next.font, next.font_class, next.c, next.fontsize,
                self.dpi)
        return advance + kern

    def render(self, output: Output, x: float, y: float) -> None:
        self.fontset.render_glyph(
            output, x, y,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)

    def shrink(self) -> None:
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.fontsize *= SHRINK_FACTOR
            self.width    *= SHRINK_FACTOR
            self.height   *= SHRINK_FACTOR
            self.depth    *= SHRINK_FACTOR


class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """
    def _update_metrics(self) -> None:
        metrics = self._metrics = self.fontset.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
        self.width = metrics.xmax - metrics.xmin
        self.height = metrics.ymax - metrics.ymin
        self.depth = 0

    def shrink(self) -> None:
        super().shrink()
        self._update_metrics()

    def render(self, output: Output, x: float, y: float) -> None:
        self.fontset.render_glyph(
            output, x - self._metrics.xmin, y + self._metrics.ymin,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)


class List(Box):
    """A list of nodes (either horizontal or vertical)."""

    def __init__(self, elements: T.Sequence[Node]):
        super().__init__(0., 0., 0.)
        self.shift_amount = 0.   # An arbitrary offset
        self.children = [*elements]  # The child nodes of this list
        # The following parameters are set in the vpack and hpack functions
        self.glue_set     = 0.   # The glue setting of this list
        self.glue_sign    = 0    # 0: normal, -1: shrinking, 1: stretching
        self.glue_order   = 0    # The order of infinity (0 - 3) for the glue

    def __repr__(self) -> str:
        return '{}<w={:.02f} h={:.02f} d={:.02f} s={:.02f}>[{}]'.format(
            super().__repr__(),
            self.width, self.height,
            self.depth, self.shift_amount,
            ', '.join([repr(x) for x in self.children]))

    def _set_glue(self, x: float, sign: int, totals: list[float],
                  error_type: str) -> None:
        self.glue_order = o = next(
            # Highest order of glue used by the members of this list.
            (i for i in range(len(totals))[::-1] if totals[i] != 0), 0)
        self.glue_sign = sign
        if totals[o] != 0.:
            self.glue_set = x / totals[o]
        else:
            self.glue_sign = 0
            self.glue_ratio = 0.
        if o == 0:
            if len(self.children):
                _log.warning("%s %s: %r",
                             error_type, type(self).__name__, self)

    def shrink(self) -> None:
        for child in self.children:
            child.shrink()
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.shift_amount *= SHRINK_FACTOR
            self.glue_set     *= SHRINK_FACTOR


class Hlist(List):
    """A horizontal list of boxes."""

    def __init__(self, elements: T.Sequence[Node], w: float = 0.0,
                 m: T.Literal['additional', 'exactly'] = 'additional',
                 do_kern: bool = True):
        super().__init__(elements)
        if do_kern:
            self.kern()
        self.hpack(w=w, m=m)

    def kern(self) -> None:
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
        new_children = []
        num_children = len(self.children)
        if num_children:
            for i in range(num_children):
                elem = self.children[i]
                if i < num_children - 1:
                    next = self.children[i + 1]
                else:
                    next = None

                new_children.append(elem)
                kerning_distance = elem.get_kerning(next)
                if kerning_distance != 0.:
                    kern = Kern(kerning_distance)
                    new_children.append(kern)
            self.children = new_children

    def hpack(self, w: float = 0.0,
              m: T.Literal['additional', 'exactly'] = 'additional') -> None:
        r"""
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """
        # I don't know why these get reset in TeX.  Shift_amount is pretty
        # much useless if we do.
        # self.shift_amount = 0.
        h = 0.
        d = 0.
        x = 0.
        total_stretch = [0.] * 4
        total_shrink = [0.] * 4
        for p in self.children:
            if isinstance(p, Char):
                x += p.width
                h = max(h, p.height)
                d = max(d, p.depth)
            elif isinstance(p, Box):
                x += p.width
                if not np.isinf(p.height) and not np.isinf(p.depth):
                    s = getattr(p, 'shift_amount', 0.)
                    h = max(h, p.height - s)
                    d = max(d, p.depth + s)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += p.width
        self.height = h
        self.depth = d

        if m == 'additional':
            w += x
        self.width = w
        x = w - x

        if x == 0.:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.
            return
        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overful")
        else:
            self._set_glue(x, -1, total_shrink, "Underful")


class Vlist(List):
    """A vertical list of boxes."""

    def __init__(self, elements: T.Sequence[Node], h: float = 0.0,
                 m: T.Literal['additional', 'exactly'] = 'additional'):
        super().__init__(elements)
        self.vpack(h=h, m=m)

    def vpack(self, h: float = 0.0,
              m: T.Literal['additional', 'exactly'] = 'additional',
              l: float = np.inf) -> None:
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *h*; or a box
            with the natural height of the contents, plus *h* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """
        # I don't know why these get reset in TeX.  Shift_amount is pretty
        # much useless if we do.
        # self.shift_amount = 0.
        w = 0.
        d = 0.
        x = 0.
        total_stretch = [0.] * 4
        total_shrink = [0.] * 4
        for p in self.children:
            if isinstance(p, Box):
                x += d + p.height
                d = p.depth
                if not np.isinf(p.width):
                    s = getattr(p, 'shift_amount', 0.)
                    w = max(w, p.width + s)
            elif isinstance(p, Glue):
                x += d
                d = 0.
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += d + p.width
                d = 0.
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in Vlist")

        self.width = w
        if d > l:
            x += d - l
            self.depth = l
        else:
            self.depth = d

        if m == 'additional':
            h += x
        self.height = h
        x = h - x

        if x == 0:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.
            return

        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overful")
        else:
            self._set_glue(x, -1, total_shrink, "Underful")


class Rule(Box):
    """
    A solid black rectangle.

    It has *width*, *depth*, and *height* fields just as in an `Hlist`.
    However, if any of these dimensions is inf, the actual value will be
    determined by running the rule up to the boundary of the innermost
    enclosing box.  This is called a "running dimension".  The width is never
    running in an `Hlist`; the height and depth are never running in a `Vlist`.
    """

    def __init__(self, width: float, height: float, depth: float, state: ParserState):
        super().__init__(width, height, depth)
        self.fontset = state.fontset

    def render(self, output: Output,  # type: ignore[override]
               x: float, y: float, w: float, h: float) -> None:
        self.fontset.render_rect_filled(output, x, y, x + w, y + h)


class Hrule(Rule):
    """Convenience class to create a horizontal rule."""

    def __init__(self, state: ParserState, thickness: float | None = None):
        if thickness is None:
            thickness = state.get_current_underline_thickness()
        height = depth = thickness * 0.5
        super().__init__(np.inf, height, depth, state)


class Vrule(Rule):
    """Convenience class to create a vertical rule."""

    def __init__(self, state: ParserState):
        thickness = state.get_current_underline_thickness()
        super().__init__(thickness, np.inf, np.inf, state)


class _GlueSpec(NamedTuple):
    width: float
    stretch: float
    stretch_order: int
    shrink: float
    shrink_order: int


_GlueSpec._named = {  # type: ignore[attr-defined]
    'fil':         _GlueSpec(0., 1., 1, 0., 0),
    'fill':        _GlueSpec(0., 1., 2, 0., 0),
    'filll':       _GlueSpec(0., 1., 3, 0., 0),
    'neg_fil':     _GlueSpec(0., 0., 0, 1., 1),
    'neg_fill':    _GlueSpec(0., 0., 0, 1., 2),
    'neg_filll':   _GlueSpec(0., 0., 0, 1., 3),
    'empty':       _GlueSpec(0., 0., 0, 0., 0),
    'ss':          _GlueSpec(0., 1., 1, -1., 1),
}


class Glue(Node):
    """
    Most of the information in this object is stored in the underlying
    ``_GlueSpec`` class, which is shared between multiple glue objects.
    (This is a memory optimization which probably doesn't matter anymore, but
    it's easier to stick to what TeX does.)
    """

    def __init__(self,
                 glue_type: _GlueSpec | T.Literal["fil", "fill", "filll",
                                                  "neg_fil", "neg_fill", "neg_filll",
                                                  "empty", "ss"]):
        super().__init__()
        if isinstance(glue_type, str):
            glue_spec = _GlueSpec._named[glue_type]  # type: ignore[attr-defined]
        elif isinstance(glue_type, _GlueSpec):
            glue_spec = glue_type
        else:
            raise ValueError("glue_type must be a glue spec name or instance")
        self.glue_spec = glue_spec

    def shrink(self) -> None:
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            g = self.glue_spec
            self.glue_spec = g._replace(width=g.width * SHRINK_FACTOR)


class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements: list[Node]):
        super().__init__([Glue('ss'), *elements, Glue('ss')], do_kern=False)


class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements: list[Node]):
        super().__init__([Glue('ss'), *elements, Glue('ss')])


class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """

    height = 0
    depth = 0

    def __init__(self, width: float):
        super().__init__()
        self.width = width

    def __repr__(self) -> str:
        return "k%.02f" % self.width

    def shrink(self) -> None:
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR


class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, height: float, depth: float, state: ParserState,
                 always: bool = False, factor: float | None = None):
        alternatives = state.fontset.get_sized_alternatives_for_symbol(
            state.font, c)

        xHeight = state.fontset.get_xheight(
            state.font, state.fontsize, state.dpi)

        state = state.copy()
        target_total = height + depth
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)
            # Ensure that size 0 is chosen when the text is regular sized but
            # with descender glyphs by subtracting 0.2 * xHeight
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break

        shift = 0.0
        if state.font != '0' or len(alternatives) == 1:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)

            shift = (depth - char.depth)

        super().__init__([char])
        self.shift_amount = shift


class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, width: float, state: ParserState, always: bool = False,
                 char_class: type[Char] = Char):
        alternatives = state.fontset.get_sized_alternatives_for_symbol(
            state.font, c)

        state = state.copy()
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)
            if char.width >= width:
                break

        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)

        super().__init__([char])
        self.width = char.width


def ship(box: Box, xy: tuple[float, float] = (0, 0)) -> Output:
    """
    Ship out *box* at offset *xy*, converting it to an `Output`.

    Since boxes can be inside of boxes inside of boxes, the main work of `ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become local variables here.
    """
    ox, oy = xy
    cur_v = 0.
    cur_h = 0.
    off_h = ox
    off_v = oy + box.height
    output = Output(box)

    def clamp(value: float) -> float:
        return -1e9 if value < -1e9 else +1e9 if value > +1e9 else value

    def hlist_out(box: Hlist) -> None:
        nonlocal cur_v, cur_h, off_h, off_v

        cur_g = 0
        cur_glue = 0.
        glue_order = box.glue_order
        glue_sign = box.glue_sign
        base_line = cur_v
        left_edge = cur_h

        for p in box.children:
            if isinstance(p, Char):
                p.render(output, cur_h + off_h, cur_v + off_v)
                cur_h += p.width
            elif isinstance(p, Kern):
                cur_h += p.width
            elif isinstance(p, List):
                # node623
                if len(p.children) == 0:
                    cur_h += p.width
                else:
                    edge = cur_h
                    cur_v = base_line + p.shift_amount
                    if isinstance(p, Hlist):
                        hlist_out(p)
                    elif isinstance(p, Vlist):
                        # p.vpack(box.height + box.depth, 'exactly')
                        vlist_out(p)
                    else:
                        assert False, "unreachable code"
                    cur_h = edge + p.width
                    cur_v = base_line
            elif isinstance(p, Box):
                # node624
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_height):
                    rule_height = box.height
                if np.isinf(rule_depth):
                    rule_depth = box.depth
                if rule_height > 0 and rule_width > 0:
                    cur_v = base_line + rule_depth
                    p.render(output,
                             cur_h + off_h, cur_v + off_v,
                             rule_width, rule_height)
                    cur_v = base_line
                cur_h += rule_width
            elif isinstance(p, Glue):
                # node625
                glue_spec = p.glue_spec
                rule_width = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_width += cur_g
                cur_h += rule_width

    def vlist_out(box: Vlist) -> None:
        nonlocal cur_v, cur_h, off_h, off_v

        cur_g = 0
        cur_glue = 0.
        glue_order = box.glue_order
        glue_sign = box.glue_sign
        left_edge = cur_h
        cur_v -= box.height
        top_edge = cur_v

        for p in box.children:
            if isinstance(p, Kern):
                cur_v += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    cur_v += p.height + p.depth
                else:
                    cur_v += p.height
                    cur_h = left_edge + p.shift_amount
                    save_v = cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        hlist_out(p)
                    elif isinstance(p, Vlist):
                        vlist_out(p)
                    else:
                        assert False, "unreachable code"
                    cur_v = save_v + p.depth
                    cur_h = left_edge
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                if rule_height > 0 and rule_depth > 0:
                    cur_v += rule_height
                    p.render(output,
                             cur_h + off_h, cur_v + off_v,
                             rule_width, rule_height)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:  # shrinking
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                cur_v += rule_height
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in vlist")

    assert isinstance(box, Hlist)
    hlist_out(box)
    return output


##############################################################################
# PARSER


def Error(msg: str) -> ParserElement:
    """Helper class to raise parser errors."""
    def raise_error(s: str, loc: int, toks: ParseResults) -> T.Any:
        raise ParseFatalException(s, loc, msg)

    return Empty().set_parse_action(raise_error)


class ParserState:
    """
    Parser state.

    States are pushed and popped from a stack as necessary, and the "current"
    state is always at the top of the stack.

    Upon entering and leaving a group { } or math/non-math, the stack is pushed
    and popped accordingly.
    """

    def __init__(self, fontset: Fonts, font: str, font_class: str, fontsize: float,
                 dpi: float):
        self.fontset = fontset
        self._font = font
        self.font_class = font_class
        self.fontsize = fontsize
        self.dpi = dpi

    def copy(self) -> ParserState:
        return copy.copy(self)

    @property
    def font(self) -> str:
        return self._font

    @font.setter
    def font(self, name: str) -> None:
        if name in ('rm', 'it', 'bf', 'bfit'):
            self.font_class = name
        self._font = name

    def get_current_underline_thickness(self) -> float:
        """Return the underline thickness for this state."""
        return self.fontset.get_underline_thickness(
            self.font, self.fontsize, self.dpi)


def cmd(expr: str, args: ParserElement) -> ParserElement:
    r"""
    Helper to define TeX commands.

    ``cmd("\cmd", args)`` is equivalent to
    ``"\cmd" - (args | Error("Expected \cmd{arg}{...}"))`` where the names in
    the error message are taken from element names in *args*.  If *expr*
    already includes arguments (e.g. "\cmd{arg}{...}"), then they are stripped
    when constructing the parse element, but kept (and *expr* is used as is) in
    the error message.
    """

    def names(elt: ParserElement) -> T.Generator[str, None, None]:
        if isinstance(elt, ParseExpression):
            for expr in elt.exprs:
                yield from names(expr)
        elif elt.resultsName:
            yield elt.resultsName

    csname = expr.split("{", 1)[0]
    err = (csname + "".join("{%s}" % name for name in names(args))
           if expr == csname else expr)
    return csname - (args | Error(f"Expected {err}"))


class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = 0
        TEXTSTYLE = 1
        SCRIPTSTYLE = 2
        SCRIPTSCRIPTSTYLE = 3

    _binary_operators = set(
      '+ * - \N{MINUS SIGN}'
      r'''
      \pm             \sqcap                   \rhd
      \mp             \sqcup                   \unlhd
      \times          \vee                     \unrhd
      \div            \wedge                   \oplus
      \ast            \setminus                \ominus
      \star           \wr                      \otimes
      \circ           \diamond                 \oslash
      \bullet         \bigtriangleup           \odot
      \cdot           \bigtriangledown         \bigcirc
      \cap            \triangleleft            \dagger
      \cup            \triangleright           \ddagger
      \uplus          \lhd                     \amalg
      \dotplus        \dotminus                \Cap
      \Cup            \barwedge                \boxdot
      \boxminus       \boxplus                 \boxtimes
      \curlyvee       \curlywedge              \divideontimes
      \doublebarwedge \leftthreetimes          \rightthreetimes
      \slash          \veebar                  \barvee
      \cupdot         \intercal                \amalg
      \circledcirc    \circleddash             \circledast
      \boxbar         \obar                    \merge
      \minuscolon     \dotsminusdots
      '''.split())

    _relation_symbols = set(r'''
      = < > :
      \leq          \geq          \equiv       \models
      \prec         \succ         \sim         \perp
      \preceq       \succeq       \simeq       \mid
      \ll           \gg           \asymp       \parallel
      \subset       \supset       \approx      \bowtie
      \subseteq     \supseteq     \cong        \Join
      \sqsubset     \sqsupset     \neq         \smile
      \sqsubseteq   \sqsupseteq   \doteq       \frown
      \in           \ni           \propto      \vdash
      \dashv        \dots         \doteqdot    \leqq
      \geqq         \lneqq        \gneqq       \lessgtr
      \leqslant     \geqslant     \eqgtr       \eqless
      \eqslantless  \eqslantgtr   \lesseqgtr   \backsim
      \backsimeq    \lesssim      \gtrsim      \precsim
      \precnsim     \gnsim        \lnsim       \succsim
      \succnsim     \nsim         \lesseqqgtr  \gtreqqless
      \gtreqless    \subseteqq    \supseteqq   \subsetneqq
      \supsetneqq   \lessapprox   \approxeq    \gtrapprox
      \precapprox   \succapprox   \precnapprox \succnapprox
      \npreccurlyeq \nsucccurlyeq \nsqsubseteq \nsqsupseteq
      \sqsubsetneq  \sqsupsetneq  \nlesssim    \ngtrsim
      \nlessgtr     \ngtrless     \lnapprox    \gnapprox
      \napprox      \approxeq     \approxident \lll
      \ggg          \nparallel    \Vdash       \Vvdash
      \nVdash       \nvdash       \vDash       \nvDash
      \nVDash       \oequal       \simneqq     \triangle
      \triangleq         \triangleeq         \triangleleft
      \triangleright     \ntriangleleft      \ntriangleright
      \trianglelefteq    \ntrianglelefteq    \trianglerighteq
      \ntrianglerighteq  \blacktriangleleft  \blacktriangleright
      \equalparallel     \measuredrightangle \varlrtriangle
      \Doteq        \Bumpeq       \Subset      \Supset
      \backepsilon  \because      \therefore   \bot
      \top          \bumpeq       \circeq      \coloneq
      \curlyeqprec  \curlyeqsucc  \eqcirc      \eqcolon
      \eqsim        \fallingdotseq \gtrdot     \gtrless
      \ltimes       \rtimes       \lessdot     \ne
      \ncong        \nequiv       \ngeq        \ngtr
      \nleq         \nless        \nmid        \notin
      \nprec        \nsubset      \nsubseteq   \nsucc
      \nsupset      \nsupseteq    \pitchfork   \preccurlyeq
      \risingdotseq \subsetneq    \succcurlyeq \supsetneq
      \varpropto    \vartriangleleft \scurel
      \vartriangleright \rightangle \equal     \backcong
      \eqdef        \wedgeq       \questeq     \between
      \veeeq        \disin        \varisins    \isins
      \isindot      \varisinobar  \isinobar    \isinvb
      \isinE        \nisd         \varnis      \nis
      \varniobar    \niobar       \bagmember   \ratio
      \Equiv        \stareq       \measeq      \arceq
      \rightassert  \rightModels  \smallin     \smallowns
      \notsmallowns \nsimeq'''.split())

    _arrow_symbols = set(r"""
     \leftarrow \longleftarrow \uparrow \Leftarrow \Longleftarrow
     \Uparrow \rightarrow \longrightarrow \downarrow \Rightarrow
     \Longrightarrow \Downarrow \leftrightarrow \updownarrow
     \longleftrightarrow \updownarrow \Leftrightarrow
     \Longleftrightarrow \Updownarrow \mapsto \longmapsto \nearrow
     \hookleftarrow \hookrightarrow \searrow \leftharpoonup
     \rightharpoonup \swarrow \leftharpoondown \rightharpoondown
     \nwarrow \rightleftharpoons \leadsto \dashrightarrow
     \dashleftarrow \leftleftarrows \leftrightarrows \Lleftarrow
     \Rrightarrow \twoheadleftarrow \leftarrowtail \looparrowleft
     \leftrightharpoons \curvearrowleft \circlearrowleft \Lsh
     \upuparrows \upharpoonleft \downharpoonleft \multimap
     \leftrightsquigarrow \rightrightarrows \rightleftarrows
     \rightrightarrows \rightleftarrows \twoheadrightarrow
     \rightarrowtail \looparrowright \rightleftharpoons
     \curvearrowright \circlearrowright \Rsh \downdownarrows
     \upharpoonright \downharpoonright \rightsquigarrow \nleftarrow
     \nrightarrow \nLeftarrow \nRightarrow \nleftrightarrow
     \nLeftrightarrow \to \Swarrow \Searrow \Nwarrow \Nearrow
     \leftsquigarrow \overleftarrow \overleftrightarrow \cwopencirclearrow
     \downzigzagarrow \cupleftarrow \rightzigzagarrow \twoheaddownarrow
     \updownarrowbar \twoheaduparrow \rightarrowbar \updownarrows
     \barleftarrow \mapsfrom \mapsdown \mapsup \Ldsh \Rdsh
     """.split())

    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols

    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())

    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())

    _overunder_functions = set("lim liminf limsup sup max min".split())

    _dropsub_symbols = set(r'\int \oint \iint \oiint \iiint \oiiint \iiiint'.split())

    _fontnames = set("rm cal it tt sf bf bfit "
                     "default bb frak scr regular".split())

    _function_names = set("""
