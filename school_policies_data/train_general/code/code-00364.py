            lower, upper = -1e250, 1e250

        if self.extend in ('both', 'min'):
            self._levels.insert(0, lower)
        if self.extend in ('both', 'max'):
            self._levels.append(upper)
        self._levels = np.asarray(self._levels)

        if not self.filled:
            self.layers = self.levels
            return

        # Layer values are mid-way between levels in screen space.
        if self.logscale:
            # Avoid overflow by taking sqrt before multiplying.
            self.layers = (np.sqrt(self._levels[:-1])
                           * np.sqrt(self._levels[1:]))
        else:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the colormapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            # Generate integers for direct indexing.
            i0, i1 = 0, len(self.levels)
            if self.filled:
                i1 -= 1
                # Out of range indices for over and under:
                if self.extend in ('both', 'min'):
                    i0 -= 1
                if self.extend in ('both', 'max'):
                    i1 += 1
            self.cvalues = list(range(i0, i1))
            self.set_norm(mcolors.NoNorm())
        else:
            self.cvalues = self.layers
        self.norm.autoscale_None(self.levels)
        self.set_array(self.cvalues)
        self.update_scalarmappable()
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False

    def _process_linewidths(self, linewidths):
        Nlev = len(self.levels)
        if linewidths is None:
            default_linewidth = mpl.rcParams['contour.linewidth']
            if default_linewidth is None:
                default_linewidth = mpl.rcParams['lines.linewidth']
            return [default_linewidth] * Nlev
        elif not np.iterable(linewidths):
            return [linewidths] * Nlev
        else:
            linewidths = list(linewidths)
            return (linewidths * math.ceil(Nlev / len(linewidths)))[:Nlev]

    def _process_linestyles(self, linestyles):
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            if self.monochrome:
                eps = - (self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = self.negative_linestyles
        else:
            if isinstance(linestyles, str):
                tlinestyles = [linestyles] * Nlev
            elif np.iterable(linestyles):
                tlinestyles = list(linestyles)
                if len(tlinestyles) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linestyles)))
                    tlinestyles = tlinestyles * nreps
                if len(tlinestyles) > Nlev:
                    tlinestyles = tlinestyles[:Nlev]
            else:
                raise ValueError("Unrecognized type for linestyles kwarg")
        return tlinestyles

    def _find_nearest_contour(self, xy, indices=None):
        """
        Find the point in the unfilled contour plot that is closest (in screen
        space) to point *xy*.

        Parameters
        ----------
        xy : tuple[float, float]
            The reference point (in screen space).
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all levels
            are considered.

        Returns
        -------
        idx_level_min : int
            The index of the contour level closest to *xy*.
        idx_vtx_min : int
            The index of the `.Path` segment closest to *xy* (at that level).
        proj : (float, float)
            The point in the contour plot closest to *xy*.
        """

        # Convert each contour segment to pixel coordinates and then compare the given
        # point to those coordinates for each contour. This is fast enough in normal
        # cases, but speedups may be possible.

        if self.filled:
            raise ValueError("Method does not support filled contours")

        if indices is None:
            indices = range(len(self._paths))

        d2min = np.inf
        idx_level_min = idx_vtx_min = proj_min = None

        for idx_level in indices:
            path = self._paths[idx_level]
            idx_vtx_start = 0
            for subpath in path._iter_connected_components():
                if not len(subpath.vertices):
                    continue
                lc = self.get_transform().transform(subpath.vertices)
                d2, proj, leg = _find_closest_point_on_path(lc, xy)
                if d2 < d2min:
                    d2min = d2
                    idx_level_min = idx_level
                    idx_vtx_min = leg[1] + idx_vtx_start
                    proj_min = proj
                idx_vtx_start += len(subpath)

        return idx_level_min, idx_vtx_min, proj_min

    def find_nearest_contour(self, x, y, indices=None, pixel=True):
        """
        Find the point in the contour plot that is closest to ``(x, y)``.

        This method does not support filled contours.

        Parameters
        ----------
        x, y : float
            The reference point.
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all
            levels are considered.
        pixel : bool, default: True
            If *True*, measure distance in pixel (screen) space, which is
            useful for manual contour labeling; else, measure distance in axes
            space.

        Returns
        -------
        path : int
            The index of the path that is closest to ``(x, y)``.  Each path corresponds
            to one contour level.
        subpath : int
            The index within that closest path of the subpath that is closest to
            ``(x, y)``.  Each subpath corresponds to one unbroken contour line.
        index : int
            The index of the vertices within that subpath that are closest to
            ``(x, y)``.
        xmin, ymin : float
            The point in the contour plot that is closest to ``(x, y)``.
        d2 : float
            The squared distance from ``(xmin, ymin)`` to ``(x, y)``.
        """
        segment = index = d2 = None

        with ExitStack() as stack:
            if not pixel:
                # _find_nearest_contour works in pixel space. We want axes space, so
                # effectively disable the transformation here by setting to identity.
                stack.enter_context(self._cm_set(
                    transform=mtransforms.IdentityTransform()))

            i_level, i_vtx, (xmin, ymin) = self._find_nearest_contour((x, y), indices)

        if i_level is not None:
            cc_cumlens = np.cumsum(
                [*map(len, self._paths[i_level]._iter_connected_components())])
            segment = cc_cumlens.searchsorted(i_vtx, "right")
            index = i_vtx if segment == 0 else i_vtx - cc_cumlens[segment - 1]
            d2 = (xmin-x)**2 + (ymin-y)**2

        return (i_level, segment, index, xmin, ymin, d2)

    def draw(self, renderer):
        paths = self._paths
        n_paths = len(paths)
        if not self.filled or all(hatch is None for hatch in self.hatches):
            super().draw(renderer)
            return
        # In presence of hatching, draw contours one at a time.
        edgecolors = self.get_edgecolors()
        if edgecolors.size == 0:
            edgecolors = ("none",)
        for idx in range(n_paths):
            with cbook._setattr_cm(self, _paths=[paths[idx]]), self._cm_set(
                hatch=self.hatches[idx % len(self.hatches)],
                array=[self.get_array()[idx]],
                linewidths=[self.get_linewidths()[idx % len(self.get_linewidths())]],
                linestyles=[self.get_linestyles()[idx % len(self.get_linestyles())]],
                edgecolors=edgecolors[idx % len(edgecolors)],
            ):
                super().draw(renderer)


@_docstring.interpd
class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    This class is typically not instantiated directly by the user but by
    `~.Axes.contour` and `~.Axes.contourf`.

    %(contour_set_attributes)s
    """

    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        if args and isinstance(args[0], QuadContourSet):
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._corner_mask = args[0]._corner_mask
            contour_generator = args[0]._contour_generator
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
            self._algorithm = args[0]._algorithm
        else:
            import contourpy

            if algorithm is None:
                algorithm = mpl.rcParams['contour.algorithm']
            mpl.rcParams.validate["contour.algorithm"](algorithm)
            self._algorithm = algorithm

            if corner_mask is None:
                if self._algorithm == "mpl2005":
                    # mpl2005 does not support corner_mask=True so if not
                    # specifically requested then disable it.
                    corner_mask = False
                else:
                    corner_mask = mpl.rcParams['contour.corner_mask']
            self._corner_mask = corner_mask

            x, y, z = self._contour_args(args, kwargs)

            contour_generator = contourpy.contour_generator(
                x, y, z, name=self._algorithm, corner_mask=self._corner_mask,
                line_type=contourpy.LineType.SeparateCode,
                fill_type=contourpy.FillType.OuterCode,
                chunk_size=self.nchunk)

            t = self.get_transform()

            # if the transform is not trans data, and some part of it
            # contains transData, transform the xs and ys to data coordinates
            if (t != self.axes.transData and
                    any(t.contains_branch_seperately(self.axes.transData))):
                trans_to_data = t - self.axes.transData
                pts = np.vstack([x.flat, y.flat]).T
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]

            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]

        self._contour_generator = contour_generator

        return kwargs

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        nargs = len(args)

        if 0 < nargs <= 2:
            z, *args = args
            z = ma.asarray(z)
            x, y = self._initialize_x_y(z)
        elif 2 < nargs <= 4:
            x, y, z_orig, *args = args
            x, y, z = self._check_xyz(x, y, z_orig, kwargs)

        else:
            raise _api.nargs_error(fn, takes="from 1 to 4", given=nargs)
        z = ma.masked_invalid(z, copy=False)
        self.zmax = z.max().astype(float)
        self.zmin = z.min().astype(float)
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            _api.warn_external('Log scale: values of z <= 0 have been masked')
            self.zmin = z.min().astype(float)
        self._process_contour_level_args(args, z.dtype)
        return (x, y, z)

    def _check_xyz(self, x, y, z, kwargs):
        """
        Check that the shapes of the input arrays match; if x and y are 1D,
        convert them to 2D using meshgrid.
        """
        x, y = self.axes._process_unit_info([("x", x), ("y", y)], kwargs)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = ma.asarray(z)

        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        if z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        Ny, Nx = z.shape

        if x.ndim != y.ndim:
            raise TypeError(f"Number of dimensions of x ({x.ndim}) and y "
                            f"({y.ndim}) do not match")
        if x.ndim == 1:
            nx, = x.shape
            ny, = y.shape
            if nx != Nx:
                raise TypeError(f"Length of x ({nx}) must match number of "
                                f"columns in z ({Nx})")
            if ny != Ny:
                raise TypeError(f"Length of y ({ny}) must match number of "
                                f"rows in z ({Ny})")
            x, y = np.meshgrid(x, y)
        elif x.ndim == 2:
            if x.shape != z.shape:
                raise TypeError(
                    f"Shapes of x {x.shape} and z {z.shape} do not match")
            if y.shape != z.shape:
                raise TypeError(
                    f"Shapes of y {y.shape} and z {z.shape} do not match")
        else:
            raise TypeError(f"Inputs x and y must be 1D or 2D, not {x.ndim}D")

        return x, y, z

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        elif z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        else:
            Ny, Nx = z.shape
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent
        dx = (x1 - x0) / Nx
        dy = (y1 - y0) / Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x, y)


_docstring.interpd.register(contour_doc="""
`.contour` and `.contourf` draw contour lines and filled contours,
respectively.  Except as noted, function signatures and return values
are the same for both versions.

Parameters
----------
X, Y : array-like, optional
    The coordinates of the values in *Z*.

    *X* and *Y* must both be 2D with the same shape as *Z* (e.g.
    created via `numpy.meshgrid`), or they must both be 1-D such
    that ``len(X) == N`` is the number of columns in *Z* and
    ``len(Y) == M`` is the number of rows in *Z*.

    *X* and *Y* must both be ordered monotonically.

    If not given, they are assumed to be integer indices, i.e.
    ``X = range(N)``, ``Y = range(M)``.

Z : (M, N) array-like
    The height values over which the contour is drawn.  Color-mapping is
    controlled by *cmap*, *norm*, *vmin*, and *vmax*.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries
    to automatically choose no more than *n+1* "nice" contour levels
    between minimum and maximum numeric values of *Z*.

    If array-like, draw contour lines at the specified levels.
    The values must be in increasing order.

Returns
-------
`~.contour.QuadContourSet`

Other Parameters
----------------
corner_mask : bool, default: :rc:`contour.corner_mask`
    Enable/disable corner masking, which only has an effect if *Z* is
    a masked array.  If ``False``, any quad touching a masked point is
    masked out.  If ``True``, only the triangular corners of quads
    nearest those points are always masked out, other triangular
    corners comprising three unmasked points are contoured as usual.

colors : :mpltype:`color` or list of :mpltype:`color`, optional
    The colors of the levels, i.e. the lines for `.contour` and the
    areas for `.contourf`.

    The sequence is cycled for the levels in ascending order. If the
    sequence is shorter than the number of levels, it's repeated.

    As a shortcut, a single color may be used in place of one-element lists, i.e.
    ``'red'`` instead of ``['red']`` to color all levels with the same color.

    .. versionchanged:: 3.10
        Previously a single color had to be expressed as a string, but now any
        valid color format may be passed.

    By default (value *None*), the colormap specified by *cmap*
    will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

%(cmap_doc)s

    This parameter is ignored if *colors* is set.

%(norm_doc)s

    This parameter is ignored if *colors* is set.

%(vmin_vmax_doc)s

    If *vmin* or *vmax* are not given, the default color scaling is based on
    *levels*.

    This parameter is ignored if *colors* is set.

%(colorizer_doc)s

    This parameter is ignored if *colors* is set.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *Z* by specifying
    the position of ``Z[0, 0]``.  This is only relevant, if *X*, *Y*
    are not given.

    - *None*: ``Z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    - 'lower': ``Z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    - 'upper': ``Z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left
      corner.
    - 'image': Use the value from :rc:`image.origin`.

extent : (x0, x1, y0, y1), optional
    If *origin* is not *None*, then *extent* is interpreted as in
    `.imshow`: it gives the outer pixel boundaries. In this case, the
    position of Z[0, 0] is the center of the pixel, not a corner. If
    *origin* is *None*, then (*x0*, *y0*) is the position of Z[0, 0],
    and (*x1*, *y1*) is the position of Z[-1, -1].

    This argument is ignored if *X* and *Y* are specified in the call
    to contour.

locator : ticker.Locator subclass, optional
    The locator is used to determine the contour levels if they
    are not given explicitly via *levels*.
    Defaults to `~.ticker.MaxNLocator`.

extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    Determines the ``contourf``-coloring of values that are outside the
    *levels* range.

    If 'neither', values outside the *levels* range are not colored.
    If 'min', 'max' or 'both', color the values below, above or below
    and above the *levels* range.

    Values below ``min(levels)`` and above ``max(levels)`` are mapped
    to the under/over values of the `.Colormap`. Note that most
    colormaps do not have dedicated colors for these by default, so
    that the over and under values are the edge values of the colormap.
    You may want to set these values explicitly using
    `.Colormap.set_under` and `.Colormap.set_over`.

    .. note::

        An existing `.QuadContourSet` does not get notified if
        properties of its colormap are changed. Therefore, an explicit
        call `~.ContourSet.changed()` is needed after modifying the
        colormap. The explicit call can be left out, if a colorbar is
        assigned to the `.QuadContourSet` because it internally calls
        `~.ContourSet.changed()`.

    Example::

        x = np.arange(1, 10)
        y = x.reshape(-1, 1)
        h = x * y

        cs = plt.contourf(h, levels=[10, 30, 50],
            colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
        cs.changed()

xunits, yunits : registered units, optional
    Override axis units by specifying an instance of a
    :class:`matplotlib.units.ConversionInterface`.

antialiased : bool, optional
    Enable antialiasing, overriding the defaults.  For
    filled contours, the default is *False*.  For line contours,
    it is taken from :rc:`lines.antialiased`.

nchunk : int >= 0, optional
    If 0, no subdivision of the domain.  Specify a positive integer to
    divide the domain into subdomains of *nchunk* by *nchunk* quads.
    Chunking reduces the maximum length of polygons generated by the
    contouring algorithm which reduces the rendering workload passed
    on to the backend and also requires slightly less RAM.  It can
    however introduce rendering artifacts at chunk boundaries depending
    on the backend, the *antialiased* flag and value of *alpha*.

linewidths : float or array-like, default: :rc:`contour.linewidth`
    *Only applies to* `.contour`.

    The line width of the contour lines.

    If a number, all levels will be plotted with this linewidth.

    If a sequence, the levels in ascending order will be plotted with
    the linewidths in the order specified.

    If None, this falls back to :rc:`lines.linewidth`.

linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
    *Only applies to* `.contour`.

    If *linestyles* is *None*, the default is 'solid' unless the lines are
    monochrome. In that case, negative contours will instead take their
    linestyle from the *negative_linestyles* argument.

    *linestyles* can also be an iterable of the above strings specifying a set
    of linestyles to be used. If this iterable is shorter than the number of
    contour levels it will be repeated as necessary.

negative_linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, \
                       optional
    *Only applies to* `.contour`.

    If *linestyles* is *None* and the lines are monochrome, this argument
    specifies the line style for negative contours.

    If *negative_linestyles* is *None*, the default is taken from
    :rc:`contour.negative_linestyle`.

    *negative_linestyles* can also be an iterable of the above strings
    specifying a set of linestyles to be used. If this iterable is shorter than
    the number of contour levels it will be repeated as necessary.

hatches : list[str], optional
    *Only applies to* `.contourf`.

    A list of cross hatch patterns to use on the filled areas.
    If None, no hatching will be added to the contour.

algorithm : {'mpl2005', 'mpl2014', 'serial', 'threaded'}, optional
    Which contouring algorithm to use to calculate the contour lines and
    polygons. The algorithms are implemented in
    `ContourPy <https://github.com/contourpy/contourpy>`_, consult the
    `ContourPy documentation <https://contourpy.readthedocs.io>`_ for
    further information.

    The default is taken from :rc:`contour.algorithm`.

clip_path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath`
    Set the clip path.  See `~matplotlib.artist.Artist.set_clip_path`.

    .. versionadded:: 3.8

data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

Notes
-----
1. `.contourf` differs from the MATLAB version in that it does not draw
   the polygon edges. To draw edges, add line contours with calls to
   `.contour`.

2. `.contourf` fills intervals that are closed at the top; that is, for
   boundaries *z1* and *z2*, the filled region is::

      z1 < Z <= z2

   except for the lowest interval, which is closed on both sides (i.e.
   it includes the lowest value).

3. `.contour` and `.contourf` use a `marching squares
   <https://en.wikipedia.org/wiki/Marching_squares>`_ algorithm to
   compute contour locations.  More information can be found in
   `ContourPy documentation <https://contourpy.readthedocs.io>`_.
""" % _docstring.interpd.params)
<|endoftext|>"""
Module for creating Sankey diagrams using Matplotlib.
"""

import logging
from types import SimpleNamespace

import numpy as np

import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib import _docstring

_log = logging.getLogger(__name__)

__author__ = "Kevin L. Davies"
__credits__ = ["Yannick Copin"]
__license__ = "BSD"
__version__ = "2011/09/16"

# Angles [deg/90]
RIGHT = 0
UP = 1
# LEFT = 2
DOWN = 3


class Sankey:
    """
    Sankey diagram.

      Sankey diagrams are a specific type of flow diagram, in which
      the width of the arrows is shown proportionally to the flow
      quantity.  They are typically used to visualize energy or
      material or cost transfers between processes.
      `Wikipedia (6/1/2011) <https://en.wikipedia.org/wiki/Sankey_diagram>`_

    """

    def __init__(self, ax=None, scale=1.0, unit='', format='%G', gap=0.25,
                 radius=0.1, shoulder=0.03, offset=0.15, head_angle=100,
                 margin=0.4, tolerance=1e-6, **kwargs):
        """
        Create a new Sankey instance.

        The optional arguments listed below are applied to all subdiagrams so
        that there is consistent alignment and formatting.

        In order to draw a complex Sankey diagram, create an instance of
        `Sankey` by calling it without any kwargs::

            sankey = Sankey()

        Then add simple Sankey sub-diagrams::

            sankey.add() # 1
            sankey.add() # 2
            #...
            sankey.add() # n

        Finally, create the full diagram::

            sankey.finish()

        Or, instead, simply daisy-chain those calls::

            Sankey().add().add...  .add().finish()

        Other Parameters
        ----------------
        ax : `~matplotlib.axes.Axes`
            Axes onto which the data should be plotted.  If *ax* isn't
            provided, new Axes will be created.
        scale : float
            Scaling factor for the flows.  *scale* sizes the width of the paths
            in order to maintain proper layout.  The same scale is applied to
            all subdiagrams.  The value should be chosen such that the product
            of the scale and the sum of the inputs is approximately 1.0 (and
            the product of the scale and the sum of the outputs is
            approximately -1.0).
        unit : str
            The physical unit associated with the flow quantities.  If *unit*
            is None, then none of the quantities are labeled.
        format : str or callable
            A Python number formatting string or callable used to label the
            flows with their quantities (i.e., a number times a unit, where the
            unit is given). If a format string is given, the label will be
            ``format % quantity``. If a callable is given, it will be called
            with ``quantity`` as an argument.
        gap : float
            Space between paths that break in/break away to/from the top or
            bottom.
        radius : float
            Inner radius of the vertical paths.
        shoulder : float
            Size of the shoulders of output arrows.
        offset : float
            Text offset (from the dip or tip of the arrow).
        head_angle : float
            Angle, in degrees, of the arrow heads (and negative of the angle of
            the tails).
        margin : float
            Minimum space between Sankey outlines and the edge of the plot
            area.
        tolerance : float
            Acceptable maximum of the magnitude of the sum of flows.  The
            magnitude of the sum of connected flows cannot be greater than
            *tolerance*.
        **kwargs
            Any additional keyword arguments will be passed to `add`, which
            will create the first subdiagram.

        See Also
        --------
        Sankey.add
        Sankey.finish

        Examples
        --------
        .. plot:: gallery/specialty_plots/sankey_basics.py
        """
        # Check the arguments.
        if gap < 0:
            raise ValueError(
                "'gap' is negative, which is not allowed because it would "
                "cause the paths to overlap")
        if radius > gap:
            raise ValueError(
                "'radius' is greater than 'gap', which is not allowed because "
                "it would cause the paths to overlap")
        if head_angle < 0:
            raise ValueError(
                "'head_angle' is negative, which is not allowed because it "
                "would cause inputs to look like outputs and vice versa")
        if tolerance < 0:
            raise ValueError(
                "'tolerance' is negative, but it must be a magnitude")

        # Create Axes if necessary.
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

        self.diagrams = []

        # Store the inputs.
        self.ax = ax
        self.unit = unit
        self.format = format
        self.scale = scale
        self.gap = gap
        self.radius = radius
        self.shoulder = shoulder
        self.offset = offset
        self.margin = margin
        self.pitch = np.tan(np.pi * (1 - head_angle / 180.0) / 2.0)
        self.tolerance = tolerance

        # Initialize the vertices of tight box around the diagram(s).
        self.extent = np.array((np.inf, -np.inf, np.inf, -np.inf))

        # If there are any kwargs, create the first subdiagram.
        if len(kwargs):
            self.add(**kwargs)

    def _arc(self, quadrant=0, cw=True, radius=1, center=(0, 0)):
        """
        Return the codes and vertices for a rotated, scaled, and translated
        90 degree arc.

        Other Parameters
        ----------------
        quadrant : {0, 1, 2, 3}, default: 0
            Uses 0-based indexing (0, 1, 2, or 3).
        cw : bool, default: True
            If True, the arc vertices are produced clockwise; counter-clockwise
            otherwise.
        radius : float, default: 1
            The radius of the arc.
        center : (float, float), default: (0, 0)
            (x, y) tuple of the arc's center.
        """
        # Note:  It would be possible to use matplotlib's transforms to rotate,
        # scale, and translate the arc, but since the angles are discrete,
        # it's just as easy and maybe more efficient to do it here.
        ARC_CODES = [Path.LINETO,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4]
        # Vertices of a cubic Bezier curve approximating a 90 deg arc
        # These can be determined by Path.arc(0, 90).
        ARC_VERTICES = np.array([[1.00000000e+00, 0.00000000e+00],
                                 [1.00000000e+00, 2.65114773e-01],
                                 [8.94571235e-01, 5.19642327e-01],
                                 [7.07106781e-01, 7.07106781e-01],
                                 [5.19642327e-01, 8.94571235e-01],
                                 [2.65114773e-01, 1.00000000e+00],
                                 # Insignificant
                                 # [6.12303177e-17, 1.00000000e+00]])
                                 [0.00000000e+00, 1.00000000e+00]])
        if quadrant in (0, 2):
            if cw:
                vertices = ARC_VERTICES
            else:
                vertices = ARC_VERTICES[:, ::-1]  # Swap x and y.
        else:  # 1, 3
            # Negate x.
            if cw:
                # Swap x and y.
                vertices = np.column_stack((-ARC_VERTICES[:, 1],
                                             ARC_VERTICES[:, 0]))
            else:
                vertices = np.column_stack((-ARC_VERTICES[:, 0],
                                             ARC_VERTICES[:, 1]))
        if quadrant > 1:
            radius = -radius  # Rotate 180 deg.
        return list(zip(ARC_CODES, radius * vertices +
                        np.tile(center, (ARC_VERTICES.shape[0], 1))))

    def _add_input(self, path, angle, flow, length):
        """
        Add an input to a path and return its tip and label locations.
        """
        if angle is None:
            return [0, 0], [0, 0]
        else:
            x, y = path[-1][1]  # Use the last point as a reference.
            dipdepth = (flow / 2) * self.pitch
            if angle == RIGHT:
                x -= length
                dip = [x + dipdepth, y + flow / 2.0]
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x + self.gap, y + flow])])
                label_location = [dip[0] - self.offset, dip[1]]
            else:  # Vertical
                x -= self.gap
                if angle == UP:
                    sign = 1
                else:
                    sign = -1

                dip = [x - flow / 2, y - sign * (length - dipdepth)]
                if angle == DOWN:
                    quadrant = 2
                else:
                    quadrant = 1

                # Inner arc isn't needed if inner radius is zero
                if self.radius:
                    path.extend(self._arc(quadrant=quadrant,
                                          cw=angle == UP,
                                          radius=self.radius,
                                          center=(x + self.radius,
                                                  y - sign * self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                path.extend([(Path.LINETO, [x, y - sign * length]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x - flow, y - sign * length])])
                path.extend(self._arc(quadrant=quadrant,
                                      cw=angle == DOWN,
                                      radius=flow + self.radius,
                                      center=(x + self.radius,
                                              y - sign * self.radius)))
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                label_location = [dip[0], dip[1] - sign * self.offset]

            return dip, label_location

    def _add_output(self, path, angle, flow, length):
        """
        Append an output to a path and return its tip and label locations.

        .. note:: *flow* is negative for an output.
        """
        if angle is None:
            return [0, 0], [0, 0]
        else:
            x, y = path[-1][1]  # Use the last point as a reference.
            tipheight = (self.shoulder - flow / 2) * self.pitch
            if angle == RIGHT:
                x += length
                tip = [x + tipheight, y + flow / 2.0]
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, [x, y + self.shoulder]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x, y - self.shoulder + flow]),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x - self.gap, y + flow])])
                label_location = [tip[0] + self.offset, tip[1]]
            else:  # Vertical
                x += self.gap
                if angle == UP:
                    sign, quadrant = 1, 3
                else:
                    sign, quadrant = -1, 0

                tip = [x - flow / 2.0, y + sign * (length + tipheight)]
                # Inner arc isn't needed if inner radius is zero
                if self.radius:
                    path.extend(self._arc(quadrant=quadrant,
                                          cw=angle == UP,
                                          radius=self.radius,
                                          center=(x - self.radius,
                                                  y + sign * self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                path.extend([(Path.LINETO, [x, y + sign * length]),
                             (Path.LINETO, [x - self.shoulder,
                                            y + sign * length]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x + self.shoulder - flow,
                                            y + sign * length]),
                             (Path.LINETO, [x - flow, y + sign * length])])
                path.extend(self._arc(quadrant=quadrant,
                                      cw=angle == DOWN,
                                      radius=self.radius - flow,
                                      center=(x - self.radius,
                                              y + sign * self.radius)))
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                label_location = [tip[0], tip[1] + sign * self.offset]
            return tip, label_location

    def _revert(self, path, first_action=Path.LINETO):
        """
        A path is not simply reversible by path[::-1] since the code
        specifies an action to take from the **previous** point.
        """
        reverse_path = []
        next_code = first_action
        for code, position in path[::-1]:
            reverse_path.append((next_code, position))
            next_code = code
        return reverse_path
        # This might be more efficient, but it fails because 'tuple' object
        # doesn't support item assignment:
        # path[1] = path[1][-1:0:-1]
        # path[1][0] = first_action
        # path[2] = path[2][::-1]
        # return path

    @_docstring.interpd
    def add(self, patchlabel='', flows=None, orientations=None, labels='',
            trunklength=1.0, pathlengths=0.25, prior=None, connect=(0, 0),
            rotation=0, **kwargs):
        """
        Add a simple Sankey diagram with flows at the same hierarchical level.

        Parameters
        ----------
        patchlabel : str
            Label to be placed at the center of the diagram.
            Note that *label* (not *patchlabel*) can be passed as keyword
            argument to create an entry in the legend.

        flows : list of float
            Array of flow values.  By convention, inputs are positive and
            outputs are negative.

            Flows are placed along the top of the diagram from the inside out
            in order of their index within *flows*.  They are placed along the
            sides of the diagram from the top down and along the bottom from
            the outside in.

            If the sum of the inputs and outputs is
            nonzero, the discrepancy will appear as a cubic Bézier curve along
            the top and bottom edges of the trunk.

        orientations : list of {-1, 0, 1}
            List of orientations of the flows (or a single orientation to be
            used for all flows).  Valid values are 0 (inputs from
            the left, outputs to the right), 1 (from and to the top) or -1
            (from and to the bottom).

        labels : list of (str or None)
            List of labels for the flows (or a single label to be used for all
            flows).  Each label may be *None* (no label), or a labeling string.
            If an entry is a (possibly empty) string, then the quantity for the
            corresponding flow will be shown below the string.  However, if
            the *unit* of the main diagram is None, then quantities are never
            shown, regardless of the value of this argument.

        trunklength : float
            Length between the bases of the input and output groups (in
            data-space units).

        pathlengths : list of float
            List of lengths of the vertical arrows before break-in or after
            break-away.  If a single value is given, then it will be applied to
            the first (inside) paths on the top and bottom, and the length of
            all other arrows will be justified accordingly.  The *pathlengths*
            are not applied to the horizontal inputs and outputs.

        prior : int
            Index of the prior diagram to which this diagram should be
            connected.

        connect : (int, int)
            A (prior, this) tuple indexing the flow of the prior diagram and
            the flow of this diagram which should be connected.  If this is the
            first diagram or *prior* is *None*, *connect* will be ignored.

        rotation : float
            Angle of rotation of the diagram in degrees.  The interpretation of
            the *orientations* argument will be rotated accordingly (e.g., if
            *rotation* == 90, an *orientations* entry of 1 means to/from the
            left).  *rotation* is ignored if this diagram is connected to an
            existing one (using *prior* and *connect*).

        Returns
        -------
        Sankey
            The current `.Sankey` instance.

        Other Parameters
        ----------------
        **kwargs
           Additional keyword arguments set `matplotlib.patches.PathPatch`
           properties, listed below.  For example, one may want to use
           ``fill=False`` or ``label="A legend entry"``.

        %(Patch:kwdoc)s

        See Also
        --------
        Sankey.finish
        """
        # Check and preprocess the arguments.
        flows = np.array([1.0, -1.0]) if flows is None else np.array(flows)
        n = flows.shape[0]  # Number of flows
        if rotation is None:
            rotation = 0
        else:
            # In the code below, angles are expressed in deg/90.
            rotation /= 90.0
        if orientations is None:
            orientations = 0
        try:
            orientations = np.broadcast_to(orientations, n)
        except ValueError:
            raise ValueError(
                f"The shapes of 'flows' {np.shape(flows)} and 'orientations' "
                f"{np.shape(orientations)} are incompatible"
            ) from None
        try:
            labels = np.broadcast_to(labels, n)
        except ValueError:
            raise ValueError(
                f"The shapes of 'flows' {np.shape(flows)} and 'labels' "
                f"{np.shape(labels)} are incompatible"
            ) from None
        if trunklength < 0:
            raise ValueError(
                "'trunklength' is negative, which is not allowed because it "
                "would cause poor layout")
        if abs(np.sum(flows)) > self.tolerance:
            _log.info("The sum of the flows is nonzero (%f; patchlabel=%r); "
                      "is the system not at steady state?",
                      np.sum(flows), patchlabel)
        scaled_flows = self.scale * flows
        gain = sum(max(flow, 0) for flow in scaled_flows)
        loss = sum(min(flow, 0) for flow in scaled_flows)
        if prior is not None:
            if prior < 0:
                raise ValueError("The index of the prior diagram is negative")
            if min(connect) < 0:
                raise ValueError(
                    "At least one of the connection indices is negative")
            if prior >= len(self.diagrams):
                raise ValueError(
                    f"The index of the prior diagram is {prior}, but there "
                    f"are only {len(self.diagrams)} other diagrams")
            if connect[0] >= len(self.diagrams[prior].flows):
                raise ValueError(
                    "The connection index to the source diagram is {}, but "
                    "that diagram has only {} flows".format(
                        connect[0], len(self.diagrams[prior].flows)))
            if connect[1] >= n:
                raise ValueError(
                    f"The connection index to this diagram is {connect[1]}, "
                    f"but this diagram has only {n} flows")
            if self.diagrams[prior].angles[connect[0]] is None:
                raise ValueError(
                    f"The connection cannot be made, which may occur if the "
                    f"magnitude of flow {connect[0]} of diagram {prior} is "
                    f"less than the specified tolerance")
            flow_error = (self.diagrams[prior].flows[connect[0]] +
                          flows[connect[1]])
            if abs(flow_error) >= self.tolerance:
                raise ValueError(
                    f"The scaled sum of the connected flows is {flow_error}, "
                    f"which is not within the tolerance ({self.tolerance})")

        # Determine if the flows are inputs.
        are_inputs = [None] * n
        for i, flow in enumerate(flows):
            if flow >= self.tolerance:
                are_inputs[i] = True
            elif flow <= -self.tolerance:
                are_inputs[i] = False
            else:
                _log.info(
                    "The magnitude of flow %d (%f) is below the tolerance "
                    "(%f).\nIt will not be shown, and it cannot be used in a "
                    "connection.", i, flow, self.tolerance)

        # Determine the angles of the arrows (before rotation).
        angles = [None] * n
        for i, (orient, is_input) in enumerate(zip(orientations, are_inputs)):
            if orient == 1:
                if is_input:
                    angles[i] = DOWN
                elif is_input is False:
                    # Be specific since is_input can be None.
                    angles[i] = UP
            elif orient == 0:
                if is_input is not None:
                    angles[i] = RIGHT
            else:
                if orient != -1:
                    raise ValueError(
                        f"The value of orientations[{i}] is {orient}, "
                        f"but it must be -1, 0, or 1")
                if is_input:
                    angles[i] = UP
                elif is_input is False:
                    angles[i] = DOWN

        # Justify the lengths of the paths.
        if np.iterable(pathlengths):
            if len(pathlengths) != n:
                raise ValueError(
                    f"The lengths of 'flows' ({n}) and 'pathlengths' "
                    f"({len(pathlengths)}) are incompatible")
        else:  # Make pathlengths into a list.
            urlength = pathlengths
            ullength = pathlengths
            lrlength = pathlengths
            lllength = pathlengths
            d = dict(RIGHT=pathlengths)
            pathlengths = [d.get(angle, 0) for angle in angles]
            # Determine the lengths of the top-side arrows
            # from the middle outwards.
            for i, (angle, is_input, flow) in enumerate(zip(angles, are_inputs,
                                                            scaled_flows)):
                if angle == DOWN and is_input:
                    pathlengths[i] = ullength
                    ullength += flow
                elif angle == UP and is_input is False:
                    pathlengths[i] = urlength
                    urlength -= flow  # Flow is negative for outputs.
            # Determine the lengths of the bottom-side arrows
            # from the middle outwards.
            for i, (angle, is_input, flow) in enumerate(reversed(list(zip(
                  angles, are_inputs, scaled_flows)))):
                if angle == UP and is_input:
                    pathlengths[n - i - 1] = lllength
                    lllength += flow
                elif angle == DOWN and is_input is False:
                    pathlengths[n - i - 1] = lrlength
                    lrlength -= flow
            # Determine the lengths of the left-side arrows
            # from the bottom upwards.
            has_left_input = False
            for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
                  angles, are_inputs, zip(scaled_flows, pathlengths))))):
                if angle == RIGHT:
                    if is_input:
                        if has_left_input:
                            pathlengths[n - i - 1] = 0
                        else:
                            has_left_input = True
            # Determine the lengths of the right-side arrows
            # from the top downwards.
            has_right_output = False
            for i, (angle, is_input, spec) in enumerate(zip(
                  angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
                if angle == RIGHT:
                    if is_input is False:
                        if has_right_output:
                            pathlengths[i] = 0
                        else:
                            has_right_output = True

        # Begin the subpaths, and smooth the transition if the sum of the flows
        # is nonzero.
        urpath = [(Path.MOVETO, [(self.gap - trunklength / 2.0),  # Upper right
                                 gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
                                 gain / 2.0]),
                  (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
                                 gain / 2.0]),
                  (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
                                 -loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
                                 -loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap),
                                 -loss / 2.0])]
        llpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),  # Lower left
                                 loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
                                 loss / 2.0]),
                  (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
                                 loss / 2.0]),
                  (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
                                 -gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
                                 -gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0),
                                 -gain / 2.0])]
        lrpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),  # Lower right
                                 loss / 2.0])]
        ulpath = [(Path.LINETO, [self.gap - trunklength / 2.0,  # Upper left
                                 gain / 2.0])]

        # Add the subpaths and assign the locations of the tips and labels.
        tips = np.zeros((n, 2))
        label_locations = np.zeros((n, 2))
        # Add the top-side inputs and outputs from the middle outwards.
        for i, (angle, is_input, spec) in enumerate(zip(
              angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
            if angle == DOWN and is_input:
                tips[i, :], label_locations[i, :] = self._add_input(
                    ulpath, angle, *spec)
            elif angle == UP and is_input is False:
                tips[i, :], label_locations[i, :] = self._add_output(
                    urpath, angle, *spec)
        # Add the bottom-side inputs and outputs from the middle outwards.
        for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
              angles, are_inputs, list(zip(scaled_flows, pathlengths)))))):
            if angle == UP and is_input:
                tip, label_location = self._add_input(llpath, angle, *spec)
                tips[n - i - 1, :] = tip
                label_locations[n - i - 1, :] = label_location
            elif angle == DOWN and is_input is False:
                tip, label_location = self._add_output(lrpath, angle, *spec)
                tips[n - i - 1, :] = tip
                label_locations[n - i - 1, :] = label_location
        # Add the left-side inputs from the bottom upwards.
        has_left_input = False
        for i, (angle, is_input, spec) in enumerate(reversed(list(zip(
              angles, are_inputs, list(zip(scaled_flows, pathlengths)))))):
            if angle == RIGHT and is_input:
                if not has_left_input:
                    # Make sure the lower path extends
                    # at least as far as the upper one.
                    if llpath[-1][1][0] > ulpath[-1][1][0]:
                        llpath.append((Path.LINETO, [ulpath[-1][1][0],
                                                     llpath[-1][1][1]]))
                    has_left_input = True
                tip, label_location = self._add_input(llpath, angle, *spec)
                tips[n - i - 1, :] = tip
                label_locations[n - i - 1, :] = label_location
        # Add the right-side outputs from the top downwards.
        has_right_output = False
        for i, (angle, is_input, spec) in enumerate(zip(
              angles, are_inputs, list(zip(scaled_flows, pathlengths)))):
            if angle == RIGHT and is_input is False:
                if not has_right_output:
                    # Make sure the upper path extends
                    # at least as far as the lower one.
                    if urpath[-1][1][0] < lrpath[-1][1][0]:
                        urpath.append((Path.LINETO, [lrpath[-1][1][0],
                                                     urpath[-1][1][1]]))
                    has_right_output = True
                tips[i, :], label_locations[i, :] = self._add_output(
                    urpath, angle, *spec)
        # Trim any hanging vertices.
        if not has_left_input:
            ulpath.pop()
            llpath.pop()
        if not has_right_output:
            lrpath.pop()
            urpath.pop()

        # Concatenate the subpaths in the correct order (clockwise from top).
        path = (urpath + self._revert(lrpath) + llpath + self._revert(ulpath) +
                [(Path.CLOSEPOLY, urpath[0][1])])

        # Create a patch with the Sankey outline.
        codes, vertices = zip(*path)
        vertices = np.array(vertices)

        def _get_angle(a, r):
            if a is None:
                return None
            else:
                return a + r

        if prior is None:
            if rotation != 0:  # By default, none of this is needed.
                angles = [_get_angle(angle, rotation) for angle in angles]
                rotate = Affine2D().rotate_deg(rotation * 90).transform_affine
                tips = rotate(tips)
                label_locations = rotate(label_locations)
                vertices = rotate(vertices)
            text = self.ax.text(0, 0, s=patchlabel, ha='center', va='center')
        else:
            rotation = (self.diagrams[prior].angles[connect[0]] -
                        angles[connect[1]])
            angles = [_get_angle(angle, rotation) for angle in angles]
            rotate = Affine2D().rotate_deg(rotation * 90).transform_affine
            tips = rotate(tips)
            offset = self.diagrams[prior].tips[connect[0]] - tips[connect[1]]
            translate = Affine2D().translate(*offset).transform_affine
            tips = translate(tips)
            label_locations = translate(rotate(label_locations))
            vertices = translate(rotate(vertices))
            kwds = dict(s=patchlabel, ha='center', va='center')
            text = self.ax.text(*offset, **kwds)
        if mpl.rcParams['_internal.classic_mode']:
            fc = kwargs.pop('fc', kwargs.pop('facecolor', '#bfd1d4'))
            lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
        else:
            fc = kwargs.pop('fc', kwargs.pop('facecolor', None))
            lw = kwargs.pop('lw', kwargs.pop('linewidth', None))
        if fc is None:
            fc = self.ax._get_patches_for_fill.get_next_color()
        patch = PathPatch(Path(vertices, codes), fc=fc, lw=lw, **kwargs)
        self.ax.add_patch(patch)

        # Add the path labels.
        texts = []
        for number, angle, label, location in zip(flows, angles, labels,
                                                  label_locations):
            if label is None or angle is None:
                label = ''
            elif self.unit is not None:
                if isinstance(self.format, str):
                    quantity = self.format % abs(number) + self.unit
                elif callable(self.format):
                    quantity = self.format(number)
                else:
                    raise TypeError(
                        'format must be callable or a format string')
                if label != '':
                    label += "\n"
                label += quantity
            texts.append(self.ax.text(x=location[0], y=location[1],
                                      s=label,
                                      ha='center', va='center'))
        # Text objects are placed even they are empty (as long as the magnitude
        # of the corresponding flow is larger than the tolerance) in case the
        # user wants to provide labels later.

        # Expand the size of the diagram if necessary.
        self.extent = (min(np.min(vertices[:, 0]),
                           np.min(label_locations[:, 0]),
                           self.extent[0]),
                       max(np.max(vertices[:, 0]),
                           np.max(label_locations[:, 0]),
                           self.extent[1]),
                       min(np.min(vertices[:, 1]),
                           np.min(label_locations[:, 1]),
                           self.extent[2]),
                       max(np.max(vertices[:, 1]),
                           np.max(label_locations[:, 1]),
                           self.extent[3]))
        # Include both vertices _and_ label locations in the extents; there are
        # where either could determine the margins (e.g., arrow shoulders).

        # Add this diagram as a subdiagram.
        self.diagrams.append(
            SimpleNamespace(patch=patch, flows=flows, angles=angles, tips=tips,
                            text=text, texts=texts))

        # Allow a daisy-chained call structure (see docstring for the class).
        return self

    def finish(self):
        """
        Adjust the Axes and return a list of information about the Sankey
        subdiagram(s).

        Returns a list of subdiagrams with the following fields:

        ========  =============================================================
        Field     Description
        ========  =============================================================
        *patch*   Sankey outline (a `~matplotlib.patches.PathPatch`).
        *flows*   Flow values (positive for input, negative for output).
        *angles*  List of angles of the arrows [deg/90].
                  For example, if the diagram has not been rotated,
                  an input to the top side has an angle of 3 (DOWN),
                  and an output from the top side has an angle of 1 (UP).
                  If a flow has been skipped (because its magnitude is less
                  than *tolerance*), then its angle will be *None*.
        *tips*    (N, 2)-array of the (x, y) positions of the tips (or "dips")
                  of the flow paths.
                  If the magnitude of a flow is less the *tolerance* of this
                  `Sankey` instance, the flow is skipped and its tip will be at
                  the center of the diagram.
        *text*    `.Text` instance for the diagram label.
        *texts*   List of `.Text` instances for the flow labels.
        ========  =============================================================

        See Also
        --------
        Sankey.add
        """
        self.ax.axis([self.extent[0] - self.margin,
                      self.extent[1] + self.margin,
                      self.extent[2] - self.margin,
                      self.extent[3] + self.margin])
        self.ax.set_aspect('equal', adjustable='datalim')
        return self.diagrams
<|endoftext|>"""
Support for plotting vector fields.

Presently this contains Quiver and Barb. Quiver plots an arrow in the
direction of the vector, with the size of the arrow related to the
magnitude of the vector.

Barbs are like quiver in that they point along a vector, but
the magnitude of the vector is given schematically by the presence of barbs
or flags on the barb.

This will also become a home for things such as standard
deviation ellipses, which can and will be derived very easily from
the Quiver code.
"""

import math

import numpy as np
from numpy import ma

from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms


_quiver_doc = """
Plot a 2D field of arrows.

Call signature::

  quiver([X, Y], U, V, [C], /, **kwargs)

*X*, *Y* define the arrow locations, *U*, *V* define the arrow directions, and
*C* optionally sets the color. The arguments *X*, *Y*, *U*, *V*, *C* are
positional-only.

**Arrow length**

The default settings auto-scales the length of the arrows to a reasonable size.
To change this behavior see the *scale* and *scale_units* parameters.

**Arrow shape**

The arrow shape is determined by *width*, *headwidth*, *headlength* and
*headaxislength*. See the notes below.

**Arrow styling**

Each arrow is internally represented by a filled polygon with a default edge
linewidth of 0. As a result, an arrow is rather a filled area, not a line with
a head, and `.PolyCollection` properties like *linewidth*, *edgecolor*,
*facecolor*, etc. act accordingly.


Parameters
----------
X, Y : 1D or 2D array-like, optional
    The x and y coordinates of the arrow locations.

    If not given, they will be generated as a uniform integer meshgrid based
    on the dimensions of *U* and *V*.

    If *X* and *Y* are 1D but *U*, *V* are 2D, *X*, *Y* are expanded to 2D
    using ``X, Y = np.meshgrid(X, Y)``. In this case ``len(X)`` and ``len(Y)``
    must match the column and row dimensions of *U* and *V*.

U, V : 1D or 2D array-like
    The x and y direction components of the arrow vectors. The interpretation
    of these components (in data or in screen space) depends on *angles*.

    *U* and *V* must have the same number of elements, matching the number of
    arrow locations in *X*, *Y*. *U* and *V* may be masked. Locations masked
    in any of *U*, *V*, and *C* will not be drawn.

C : 1D or 2D array-like, optional
    Numeric data that defines the arrow colors by colormapping via *norm* and
    *cmap*.

    This does not support explicit colors. If you want to set colors directly,
    use *color* instead.  The size of *C* must match the number of arrow
    locations.

angles : {'uv', 'xy'} or array-like, default: 'uv'
    Method for determining the angle of the arrows.

    - 'uv':  Arrow directions are based on
      :ref:`display coordinates <coordinate-systems>`; i.e. a 45° angle will
      always show up as diagonal on the screen, irrespective of figure or Axes
      aspect ratio or Axes data ranges. This is useful when the arrows represent
      a quantity whose direction is not tied to the x and y data coordinates.

      If *U* == *V* the orientation of the arrow on the plot is 45 degrees
      counter-clockwise from the horizontal axis (positive to the right).

    - 'xy': Arrow direction in data coordinates, i.e. the arrows point from
      (x, y) to (x+u, y+v). This is ideal for vector fields or gradient plots
      where the arrows should directly represent movements or gradients in the
      x and y directions.

    - Arbitrary angles may be specified explicitly as an array of values
      in degrees, counter-clockwise from the horizontal axis.

      In this case *U*, *V* is only used to determine the length of the
      arrows.

      For example, ``angles=[30, 60, 90]`` will orient the arrows at 30, 60, and 90
      degrees respectively, regardless of the *U* and *V* components.

    Note: inverting a data axis will correspondingly invert the
    arrows only with ``angles='xy'``.

pivot : {'tail', 'mid', 'middle', 'tip'}, default: 'tail'
    The part of the arrow that is anchored to the *X*, *Y* grid. The arrow
    rotates about this point.

    'mid' is a synonym for 'middle'.

scale : float, optional
    Scales the length of the arrow inversely.

    Number of data values represented by one unit of arrow length on the plot.
    For example, if the data represents velocity in meters per second (m/s), the
    scale parameter determines how many meters per second correspond to one unit of
    arrow length relative to the width of the plot.
    Smaller scale parameter makes the arrow longer.

    By default, an autoscaling algorithm is used to scale the arrow length to a
    reasonable size, which is based on the average vector length and the number of
    vectors.

    The arrow length unit is given by the *scale_units* parameter.

scale_units : {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}, default: 'width'

    The physical image unit, which is used for rendering the scaled arrow data *U*, *V*.

    The rendered arrow length is given by

        length in x direction = $\\frac{u}{\\mathrm{scale}} \\mathrm{scale_unit}$

        length in y direction = $\\frac{v}{\\mathrm{scale}} \\mathrm{scale_unit}$

    For example, ``(u, v) = (0.5, 0)`` with ``scale=10, scale_units="width"`` results
    in a horizontal arrow with a length of *0.5 / 10 * "width"*, i.e. 0.05 times the
    Axes width.

    Supported values are:

    - 'width' or 'height': The arrow length is scaled relative to the width or height
       of the Axes.
       For example, ``scale_units='width', scale=1.0``, will result in an arrow length
       of width of the Axes.

    - 'dots': The arrow length of the arrows is in measured in display dots (pixels).

    - 'inches': Arrow lengths are scaled based on the DPI (dots per inch) of the figure.
       This ensures that the arrows have a consistent physical size on the figure,
       in inches, regardless of data values or plot scaling.
       For example, ``(u, v) = (1, 0)`` with ``scale_units='inches', scale=2`` results
       in a 0.5 inch-long arrow.

    - 'x' or 'y': The arrow length is scaled relative to the x or y axis units.
       For example, ``(u, v) = (0, 1)`` with ``scale_units='x', scale=1`` results
       in a vertical arrow with the length of 1 x-axis unit.

    - 'xy': Arrow length will be same as 'x' or 'y' units.
       This is useful for creating vectors in the x-y plane where u and v have
       the same units as x and y. To plot vectors in the x-y plane with u and v having
       the same units as x and y, use ``angles='xy', scale_units='xy', scale=1``.

    Note: Setting *scale_units* without setting scale does not have any effect because
    the scale units only differ by a constant factor and that is rescaled through
    autoscaling.

units : {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}, default: 'width'
    Affects the arrow size (except for the length). In particular, the shaft
    *width* is measured in multiples of this unit.

    Supported values are:

    - 'width', 'height': The width or height of the Axes.
    - 'dots', 'inches': Pixels or inches based on the figure dpi.
    - 'x', 'y', 'xy': *X*, *Y* or :math:`\\sqrt{X^2 + Y^2}` in data units.

    The following table summarizes how these values affect the visible arrow
    size under zooming and figure size changes:

    =================  =================   ==================
    units              zoom                figure size change
    =================  =================   ==================
    'x', 'y', 'xy'     arrow size scales   —
    'width', 'height'  —                   arrow size scales
    'dots', 'inches'   —                   —
    =================  =================   ==================

width : float, optional
    Shaft width in arrow units. All head parameters are relative to *width*.

    The default depends on choice of *units* above, and number of vectors;
    a typical starting value is about 0.005 times the width of the plot.

headwidth : float, default: 3
    Head width as multiple of shaft *width*. See the notes below.

headlength : float, default: 5
    Head length as multiple of shaft *width*. See the notes below.

headaxislength : float, default: 4.5
    Head length at shaft intersection as multiple of shaft *width*.
    See the notes below.

minshaft : float, default: 1
    Length below which arrow scales, in units of head length. Do not
    set this to less than 1, or small arrows will look terrible!

minlength : float, default: 1
    Minimum length as a multiple of shaft width; if an arrow length
    is less than this, plot a dot (hexagon) of this diameter instead.

color : :mpltype:`color` or list :mpltype:`color`, optional
    Explicit color(s) for the arrows. If *C* has been set, *color* has no
    effect.

    This is a synonym for the `.PolyCollection` *facecolor* parameter.

Other Parameters
----------------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs : `~matplotlib.collections.PolyCollection` properties, optional
    All other keyword arguments are passed on to `.PolyCollection`:

    %(PolyCollection:kwdoc)s

Returns
-------
`~matplotlib.quiver.Quiver`

See Also
--------
.Axes.quiverkey : Add a key to a quiver plot.

Notes
-----

**Arrow shape**

The arrow is drawn as a polygon using the nodes as shown below. The values
*headwidth*, *headlength*, and *headaxislength* are in units of *width*.

.. image:: /_static/quiver_sizes.svg
   :width: 500px

The defaults give a slightly swept-back arrow. Here are some guidelines how to
get other head shapes:

- To make the head a triangle, make *headaxislength* the same as *headlength*.
- To make the arrow more pointed, reduce *headwidth* or increase *headlength*
  and *headaxislength*.
- To make the head smaller relative to the shaft, scale down all the head
  parameters proportionally.
- To remove the head completely, set all *head* parameters to 0.
- To get a diamond-shaped head, make *headaxislength* larger than *headlength*.
- Warning: For *headaxislength* < (*headlength* / *headwidth*), the "headaxis"
  nodes (i.e. the ones connecting the head with the shaft) will protrude out
  of the head in forward direction so that the arrow head looks broken.
""" % _docstring.interpd.params

_docstring.interpd.register(quiver_doc=_quiver_doc)


class QuiverKey(martist.Artist):
    """Labelled arrow for use as a quiver plot scale key."""
    halign = {'N': 'center', 'S': 'center', 'E': 'left', 'W': 'right'}
    valign = {'N': 'bottom', 'S': 'top', 'E': 'center', 'W': 'center'}
    pivot = {'N': 'middle', 'S': 'middle', 'E': 'tip', 'W': 'tail'}

    def __init__(self, Q, X, Y, U, label,
                 *, angle=0, coordinates='axes', color=None, labelsep=0.1,
                 labelpos='N', labelcolor=None, fontproperties=None,
                 zorder=None, **kwargs):
        """
        Add a key to a quiver plot.

        The positioning of the key depends on *X*, *Y*, *coordinates*, and
        *labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position of
        the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y* positions
        the head, and if *labelpos* is 'W', *X*, *Y* positions the tail; in
        either of these two cases, *X*, *Y* is somewhere in the middle of the
        arrow+label key object.

        Parameters
        ----------
        Q : `~matplotlib.quiver.Quiver`
            A `.Quiver` object as returned by a call to `~.Axes.quiver()`.
        X, Y : float
            The location of the key.
        U : float
            The length of the key.
        label : str
            The key label (e.g., length and units of the key).
        angle : float, default: 0
            The angle of the key arrow, in degrees anti-clockwise from the
            horizontal axis.
        coordinates : {'axes', 'figure', 'data', 'inches'}, default: 'axes'
            Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
            normalized coordinate systems with (0, 0) in the lower left and
            (1, 1) in the upper right; 'data' are the axes data coordinates
            (used for the locations of the vectors in the quiver plot itself);
            'inches' is position in the figure in inches, with (0, 0) at the
            lower left corner.
        color : :mpltype:`color`
            Overrides face and edge colors from *Q*.
        labelpos : {'N', 'S', 'E', 'W'}
            Position the label above, below, to the right, to the left of the
            arrow, respectively.
        labelsep : float, default: 0.1
            Distance in inches between the arrow and the label.
        labelcolor : :mpltype:`color`, default: :rc:`text.color`
            Label color.
        fontproperties : dict, optional
            A dictionary with keyword arguments accepted by the
            `~matplotlib.font_manager.FontProperties` initializer:
            *family*, *style*, *variant*, *size*, *weight*.
        zorder : float
            The zorder of the key. The default is 0.1 above *Q*.
        **kwargs
            Any additional keyword arguments are used to override vector
            properties taken from *Q*.
        """
        super().__init__()
        self.Q = Q
        self.X = X
        self.Y = Y
        self.U = U
        self.angle = angle
        self.coord = coordinates
        self.color = color
        self.label = label
        self._labelsep_inches = labelsep

        self.labelpos = labelpos
        self.labelcolor = labelcolor
        self.fontproperties = fontproperties or dict()
        self.kw = kwargs
        self.text = mtext.Text(
            text=label,
            horizontalalignment=self.halign[self.labelpos],
            verticalalignment=self.valign[self.labelpos],
            fontproperties=self.fontproperties)
        if self.labelcolor is not None:
            self.text.set_color(self.labelcolor)
        self._dpi_at_last_init = None
        self.zorder = zorder if zorder is not None else Q.zorder + 0.1

    @property
    def labelsep(self):
        return self._labelsep_inches * self.Q.axes.get_figure(root=True).dpi

    def _init(self):
        if True:  # self._dpi_at_last_init != self.axes.get_figure().dpi
            if self.Q._dpi_at_last_init != self.Q.axes.get_figure(root=True).dpi:
                self.Q._init()
            self._set_transform()
            with cbook._setattr_cm(self.Q, pivot=self.pivot[self.labelpos],
                                   # Hack: save and restore the Umask
                                   Umask=ma.nomask):
                u = self.U * np.cos(np.radians(self.angle))
                v = self.U * np.sin(np.radians(self.angle))
                self.verts = self.Q._make_verts([[0., 0.]],
                                                np.array([u]), np.array([v]), 'uv')
            kwargs = self.Q.polykw
            kwargs.update(self.kw)
            self.vector = mcollections.PolyCollection(
                self.verts,
                offsets=[(self.X, self.Y)],
                offset_transform=self.get_transform(),
                **kwargs)
            if self.color is not None:
                self.vector.set_color(self.color)
            self.vector.set_transform(self.Q.get_transform())
            self.vector.set_figure(self.get_figure())
            self._dpi_at_last_init = self.Q.axes.get_figure(root=True).dpi

    def _text_shift(self):
        return {
            "N": (0, +self.labelsep),
            "S": (0, -self.labelsep),
            "E": (+self.labelsep, 0),
            "W": (-self.labelsep, 0),
        }[self.labelpos]

    @martist.allow_rasterization
    def draw(self, renderer):
        self._init()
        self.vector.draw(renderer)
        pos = self.get_transform().transform((self.X, self.Y))
        self.text.set_position(pos + self._text_shift())
        self.text.draw(renderer)
        self.stale = False

    def _set_transform(self):
        fig = self.Q.axes.get_figure(root=False)
        self.set_transform(_api.check_getitem({
            "data": self.Q.axes.transData,
            "axes": self.Q.axes.transAxes,
            "figure": fig.transFigure,
            "inches": fig.dpi_scale_trans,
        }, coordinates=self.coord))

    def set_figure(self, fig):
        super().set_figure(fig)
        self.text.set_figure(fig)

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return False, {}
        # Maybe the dictionary should allow one to
        # distinguish between a text hit and a vector hit.
        if (self.text.contains(mouseevent)[0] or
                self.vector.contains(mouseevent)[0]):
            return True, {}
        return False, {}


def _parse_args(*args, caller_name='function'):
    """
    Helper function to parse positional parameters for colored vector plots.

    This is currently used for Quiver and Barbs.

    Parameters
    ----------
    *args : list
        list of 2-5 arguments. Depending on their number they are parsed to::

            U, V
            U, V, C
            X, Y, U, V
            X, Y, U, V, C

    caller_name : str
        Name of the calling method (used in error messages).
    """
    X = Y = C = None

    nargs = len(args)
    if nargs == 2:
        # The use of atleast_1d allows for handling scalar arguments while also
        # keeping masked arrays
        U, V = np.atleast_1d(*args)
    elif nargs == 3:
        U, V, C = np.atleast_1d(*args)
    elif nargs == 4:
        X, Y, U, V = np.atleast_1d(*args)
    elif nargs == 5:
        X, Y, U, V, C = np.atleast_1d(*args)
    else:
        raise _api.nargs_error(caller_name, takes="from 2 to 5", given=nargs)

    nr, nc = (1, U.shape[0]) if U.ndim == 1 else U.shape

    if X is not None:
        X = X.ravel()
        Y = Y.ravel()
        if len(X) == nc and len(Y) == nr:
            X, Y = (a.ravel() for a in np.meshgrid(X, Y))
        elif len(X) != len(Y):
            raise ValueError('X and Y must be the same size, but '
                             f'X.size is {X.size} and Y.size is {Y.size}.')
    else:
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
        X, Y = (np.ravel(a) for a in indexgrid)
    # Size validation for U, V, C is left to the set_UVC method.
    return X, Y, U, V, C


def _check_consistent_shapes(*arrays):
    all_shapes = {a.shape for a in arrays}
    if len(all_shapes) != 1:
        raise ValueError('The shapes of the passed in arrays do not match')


class Quiver(mcollections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """

    _PIVOT_VALS = ('tail', 'middle', 'tip')

    @_docstring.Substitution(_quiver_doc)
    def __init__(self, ax, *args,
                 scale=None, headwidth=3, headlength=5, headaxislength=4.5,
                 minshaft=1, minlength=1, units='width', scale_units=None,
                 angles='uv', width=None, color='k', pivot='tail', **kwargs):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %s
        """
        self._axes = ax  # The attr actually set by the Artist.axes property.
        X, Y, U, V, C = _parse_args(*args, caller_name='quiver')
        self.X = X
        self.Y = Y
        self.XY = np.column_stack((X, Y))
        self.N = len(X)
        self.scale = scale
        self.headwidth = headwidth
        self.headlength = float(headlength)
        self.headaxislength = headaxislength
        self.minshaft = minshaft
        self.minlength = minlength
        self.units = units
        self.scale_units = scale_units
        self.angles = angles
        self.width = width

        if pivot.lower() == 'mid':
            pivot = 'middle'
        self.pivot = pivot.lower()
        _api.check_in_list(self._PIVOT_VALS, pivot=self.pivot)

        self.transform = kwargs.pop('transform', ax.transData)
        kwargs.setdefault('facecolors', color)
        kwargs.setdefault('linewidths', (0,))
        super().__init__([], offsets=self.XY, offset_transform=self.transform,
                         closed=False, **kwargs)
        self.polykw = kwargs
        self.set_UVC(U, V, C)
        self._dpi_at_last_init = None

    def _init(self):
        """
        Initialization delayed until first draw;
        allow time for axes setup.
        """
        # It seems that there are not enough event notifications
        # available to have this work on an as-needed basis at present.
        if True:  # self._dpi_at_last_init != self.axes.figure.dpi
            trans = self._set_transform()
            self.span = trans.inverted().transform_bbox(self.axes.bbox).width
            if self.width is None:
                sn = np.clip(math.sqrt(self.N), 8, 25)
                self.width = 0.06 * self.span / sn

            # _make_verts sets self.scale if not already specified
            if (self._dpi_at_last_init != self.axes.get_figure(root=True).dpi
                    and self.scale is None):
                self._make_verts(self.XY, self.U, self.V, self.angles)

            self._dpi_at_last_init = self.axes.get_figure(root=True).dpi

    def get_datalim(self, transData):
        trans = self.get_transform()
        offset_trf = self.get_offset_transform()
        full_transform = (trans - transData) + (offset_trf - transData)
        XY = full_transform.transform(self.XY)
        bbox = transforms.Bbox.null()
        bbox.update_from_data_xy(XY, ignore=True)
        return bbox

    @martist.allow_rasterization
    def draw(self, renderer):
        self._init()
        verts = self._make_verts(self.XY, self.U, self.V, self.angles)
        self.set_verts(verts, closed=False)
        super().draw(renderer)
        self.stale = False

    def set_UVC(self, U, V, C=None):
        # We need to ensure we have a copy, not a reference
        # to an array that might change before draw().
        U = ma.masked_invalid(U, copy=True).ravel()
        V = ma.masked_invalid(V, copy=True).ravel()
        if C is not None:
            C = ma.masked_invalid(C, copy=True).ravel()
        for name, var in zip(('U', 'V', 'C'), (U, V, C)):
            if not (var is None or var.size == self.N or var.size == 1):
                raise ValueError(f'Argument {name} has a size {var.size}'
                                 f' which does not match {self.N},'
                                 ' the number of arrow positions')

        mask = ma.mask_or(U.mask, V.mask, copy=False, shrink=True)
        if C is not None:
            mask = ma.mask_or(mask, C.mask, copy=False, shrink=True)
            if mask is ma.nomask:
                C = C.filled()
            else:
                C = ma.array(C, mask=mask, copy=False)
        self.U = U.filled(1)
        self.V = V.filled(1)
        self.Umask = mask
        if C is not None:
            self.set_array(C)
        self.stale = True

    def _dots_per_unit(self, units):
        """Return a scale factor for converting from units to pixels."""
        bb = self.axes.bbox
        vl = self.axes.viewLim
        return _api.check_getitem({
            'x': bb.width / vl.width,
            'y': bb.height / vl.height,
            'xy': np.hypot(*bb.size) / np.hypot(*vl.size),
            'width': bb.width,
            'height': bb.height,
            'dots': 1.,
            'inches': self.axes.get_figure(root=True).dpi,
        }, units=units)

    def _set_transform(self):
        """
        Set the PolyCollection transform to go
        from arrow width units to pixels.
        """
        dx = self._dots_per_unit(self.units)
        self._trans_scale = dx  # pixels per arrow width unit
        trans = transforms.Affine2D().scale(dx)
        self.set_transform(trans)
        return trans

    # Calculate angles and lengths for segment between (x, y), (x+u, y+v)
    def _angles_lengths(self, XY, U, V, eps=1):
        xy = self.axes.transData.transform(XY)
        uv = np.column_stack((U, V))
        xyp = self.axes.transData.transform(XY + eps * uv)
        dxy = xyp - xy
        angles = np.arctan2(dxy[:, 1], dxy[:, 0])
        lengths = np.hypot(*dxy.T) / eps
        return angles, lengths

    # XY is stacked [X, Y].
    # See quiver() doc for meaning of X, Y, U, V, angles.
    def _make_verts(self, XY, U, V, angles):
        uv = (U + V * 1j)
        str_angles = angles if isinstance(angles, str) else ''
        if str_angles == 'xy' and self.scale_units == 'xy':
            # Here eps is 1 so that if we get U, V by diffing
            # the X, Y arrays, the vectors will connect the
            # points, regardless of the axis scaling (including log).
            angles, lengths = self._angles_lengths(XY, U, V, eps=1)
        elif str_angles == 'xy' or self.scale_units == 'xy':
            # Calculate eps based on the extents of the plot
            # so that we don't end up with roundoff error from
            # adding a small number to a large.
            eps = np.abs(self.axes.dataLim.extents).max() * 0.001
            angles, lengths = self._angles_lengths(XY, U, V, eps=eps)

        if str_angles and self.scale_units == 'xy':
            a = lengths
        else:
            a = np.abs(uv)

        if self.scale is None:
            sn = max(10, math.sqrt(self.N))
            if self.Umask is not ma.nomask:
                amean = a[~self.Umask].mean()
            else:
                amean = a.mean()
            # crude auto-scaling
            # scale is typical arrow length as a multiple of the arrow width
            scale = 1.8 * amean * sn / self.span

        if self.scale_units is None:
            if self.scale is None:
                self.scale = scale
            widthu_per_lenu = 1.0
        else:
            if self.scale_units == 'xy':
                dx = 1
            else:
                dx = self._dots_per_unit(self.scale_units)
            widthu_per_lenu = dx / self._trans_scale
            if self.scale is None:
                self.scale = scale * widthu_per_lenu
        length = a * (widthu_per_lenu / (self.scale * self.width))
        X, Y = self._h_arrows(length)
        if str_angles == 'xy':
            theta = angles
        elif str_angles == 'uv':
            theta = np.angle(uv)
        else:
            theta = ma.masked_invalid(np.deg2rad(angles)).filled(0)
        theta = theta.reshape((-1, 1))  # for broadcasting
        xy = (X + Y * 1j) * np.exp(1j * theta) * self.width
        XY = np.stack((xy.real, xy.imag), axis=2)
        if self.Umask is not ma.nomask:
            XY = ma.array(XY)
            XY[self.Umask] = ma.masked
            # This might be handled more efficiently with nans, given
            # that nans will end up in the paths anyway.

        return XY

    def _h_arrows(self, length):
        """Length is in arrow width units."""
        # It might be possible to streamline the code
        # and speed it up a bit by using complex (x, y)
        # instead of separate arrays; but any gain would be slight.
        minsh = self.minshaft * self.headlength
        N = len(length)
        length = length.reshape(N, 1)
        # This number is chosen based on when pixel values overflow in Agg
        # causing rendering errors
        # length = np.minimum(length, 2 ** 16)
        np.clip(length, 0, 2 ** 16, out=length)
        # x, y: normal horizontal arrow
        x = np.array([0, -self.headaxislength,
                      -self.headlength, 0],
                     np.float64)
        x = x + np.array([0, 1, 1, 1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis, :], N, axis=0)
        # x0, y0: arrow without shaft, for short vectors
        x0 = np.array([0, minsh - self.headaxislength,
                       minsh - self.headlength, minsh], np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        ii = [0, 1, 2, 3, 2, 1, 0, 0]
        X = x[:, ii]
        Y = y[:, ii]
        Y[:, 3:-1] *= -1
        X0 = x0[ii]
        Y0 = y0[ii]
        Y0[3:-1] *= -1
        shrink = length / minsh if minsh != 0. else 0.
        X0 = shrink * X0[np.newaxis, :]
        Y0 = shrink * Y0[np.newaxis, :]
        short = np.repeat(length < minsh, 8, axis=1)
        # Now select X0, Y0 if short, otherwise X, Y
        np.copyto(X, X0, where=short)
        np.copyto(Y, Y0, where=short)
        if self.pivot == 'middle':
            X -= 0.5 * X[:, 3, np.newaxis]
        elif self.pivot == 'tip':
            # numpy bug? using -= does not work here unless we multiply by a
            # float first, as with 'mid'.
            X = X - X[:, 3, np.newaxis]
        elif self.pivot != 'tail':
            _api.check_in_list(["middle", "tip", "tail"], pivot=self.pivot)

        tooshort = length < self.minlength
        if tooshort.any():
            # Use a heptagonal dot:
            th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
            x1 = np.cos(th) * self.minlength * 0.5
            y1 = np.sin(th) * self.minlength * 0.5
            X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
            Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
            tooshort = np.repeat(tooshort, 8, 1)
            np.copyto(X, X1, where=tooshort)
            np.copyto(Y, Y1, where=tooshort)
        # Mask handling is deferred to the caller, _make_verts.
        return X, Y


_barbs_doc = r"""
Plot a 2D field of wind barbs.

Call signature::

  barbs([X, Y], U, V, [C], /, **kwargs)

Where *X*, *Y* define the barb locations, *U*, *V* define the barb
directions, and *C* optionally sets the color.

The arguments *X*, *Y*, *U*, *V*, *C* are positional-only and may be
1D or 2D. *U*, *V*, *C* may be masked arrays, but masked *X*, *Y*
are not supported at present.

Barbs are traditionally used in meteorology as a way to plot the speed
and direction of wind observations, but can technically be used to
plot any two dimensional vector quantity.  As opposed to arrows, which
give vector magnitude by the length of the arrow, the barbs give more
quantitative information about the vector magnitude by putting slanted
lines or a triangle for various increments in magnitude, as show
schematically below::

  :                   /\    \
  :                  /  \    \
  :                 /    \    \    \
  :                /      \    \    \
  :               ------------------------------

The largest increment is given by a triangle (or "flag"). After those
come full lines (barbs). The smallest increment is a half line.  There
is only, of course, ever at most 1 half line.  If the magnitude is
small and only needs a single half-line and no full lines or
triangles, the half-line is offset from the end of the barb so that it
can be easily distinguished from barbs with a single full line.  The
magnitude for the barb shown above would nominally be 65, using the
standard increments of 50, 10, and 5.

See also https://en.wikipedia.org/wiki/Wind_barb.

Parameters
----------
X, Y : 1D or 2D array-like, optional
    The x and y coordinates of the barb locations. See *pivot* for how the
    barbs are drawn to the x, y positions.

    If not given, they will be generated as a uniform integer meshgrid based
    on the dimensions of *U* and *V*.

    If *X* and *Y* are 1D but *U*, *V* are 2D, *X*, *Y* are expanded to 2D
    using ``X, Y = np.meshgrid(X, Y)``. In this case ``len(X)`` and ``len(Y)``
    must match the column and row dimensions of *U* and *V*.

U, V : 1D or 2D array-like
    The x and y components of the barb shaft.

C : 1D or 2D array-like, optional
    Numeric data that defines the barb colors by colormapping via *norm* and
    *cmap*.

    This does not support explicit colors. If you want to set colors directly,
    use *barbcolor* instead.

length : float, default: 7
    Length of the barb in points; the other parts of the barb
    are scaled against this.

pivot : {'tip', 'middle'} or float, default: 'tip'
    The part of the arrow that is anchored to the *X*, *Y* grid. The barb
    rotates about this point. This can also be a number, which shifts the
    start of the barb that many points away from grid point.

barbcolor : :mpltype:`color` or color sequence
    The color of all parts of the barb except for the flags.  This parameter
    is analogous to the *edgecolor* parameter for polygons, which can be used
    instead. However this parameter will override facecolor.

flagcolor : :mpltype:`color` or color sequence
    The color of any flags on the barb.  This parameter is analogous to the
    *facecolor* parameter for polygons, which can be used instead. However,
    this parameter will override facecolor.  If this is not set (and *C* has
    not either) then *flagcolor* will be set to match *barbcolor* so that the
    barb has a uniform color. If *C* has been set, *flagcolor* has no effect.

sizes : dict, optional
    A dictionary of coefficients specifying the ratio of a given
    feature to the length of the barb. Only those values one wishes to
    override need to be included.  These features include:

    - 'spacing' - space between features (flags, full/half barbs)
    - 'height' - height (distance from shaft to top) of a flag or full barb
    - 'width' - width of a flag, twice the width of a full barb
    - 'emptybarb' - radius of the circle used for low magnitudes

fill_empty : bool, default: False
    Whether the empty barbs (circles) that are drawn should be filled with
    the flag color.  If they are not filled, the center is transparent.

rounding : bool, default: True
    Whether the vector magnitude should be rounded when allocating barb
    components.  If True, the magnitude is rounded to the nearest multiple
    of the half-barb increment.  If False, the magnitude is simply truncated
    to the next lowest multiple.

barb_increments : dict, optional
    A dictionary of increments specifying values to associate with
    different parts of the barb. Only those values one wishes to
    override need to be included.

    - 'half' - half barbs (Default is 5)
    - 'full' - full barbs (Default is 10)
    - 'flag' - flags (default is 50)

flip_barb : bool or array-like of bool, default: False
    Whether the lines and flags should point opposite to normal.
    Normal behavior is for the barbs and lines to point right (comes from wind
    barbs having these features point towards low pressure in the Northern
    Hemisphere).

    A single value is applied to all barbs. Individual barbs can be flipped by
    passing a bool array of the same size as *U* and *V*.

Returns
-------
barbs : `~matplotlib.quiver.Barbs`

Other Parameters
----------------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs
    The barbs can further be customized using `.PolyCollection` keyword
    arguments:

    %(PolyCollection:kwdoc)s
""" % _docstring.interpd.params

_docstring.interpd.register(barbs_doc=_barbs_doc)


class Barbs(mcollections.PolyCollection):
    """
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    """

    # This may be an abuse of polygons here to render what is essentially maybe
    # 1 triangle and a series of lines.  It works fine as far as I can tell
    # however.

    @_docstring.interpd
    def __init__(self, ax, *args,
                 pivot='tip', length=7, barbcolor=None, flagcolor=None,
                 sizes=None, fill_empty=False, barb_increments=None,
                 rounding=True, flip_barb=False, **kwargs):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %(barbs_doc)s
        """
        self.sizes = sizes or dict()
        self.fill_empty = fill_empty
        self.barb_increments = barb_increments or dict()
        self.rounding = rounding
        self.flip = np.atleast_1d(flip_barb)
        transform = kwargs.pop('transform', ax.transData)
        self._pivot = pivot
        self._length = length

        # Flagcolor and barbcolor provide convenience parameters for
        # setting the facecolor and edgecolor, respectively, of the barb
        # polygon.  We also work here to make the flag the same color as the
        # rest of the barb by default

        if None in (barbcolor, flagcolor):
            kwargs['edgecolors'] = 'face'
            if flagcolor:
                kwargs['facecolors'] = flagcolor
            elif barbcolor:
                kwargs['facecolors'] = barbcolor
            else:
                # Set to facecolor passed in or default to black
                kwargs.setdefault('facecolors', 'k')
        else:
            kwargs['edgecolors'] = barbcolor
            kwargs['facecolors'] = flagcolor

        # Explicitly set a line width if we're not given one, otherwise
        # polygons are not outlined and we get no barbs
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['linewidth'] = 1

        # Parse out the data arrays from the various configurations supported
        x, y, u, v, c = _parse_args(*args, caller_name='barbs')
        self.x = x
        self.y = y
        xy = np.column_stack((x, y))

        # Make a collection
        barb_size = self._length ** 2 / 4  # Empirically determined
        super().__init__(
            [], (barb_size,), offsets=xy, offset_transform=transform, **kwargs)
        self.set_transform(transforms.IdentityTransform())

        self.set_UVC(u, v, c)

    def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
        """
        Find how many of each of the tail pieces is necessary.

        Parameters
        ----------
        mag : `~numpy.ndarray`
            Vector magnitudes; must be non-negative (and an actual ndarray).
        rounding : bool, default: True
            Whether to round or to truncate to the nearest half-barb.
        half, full, flag : float, defaults: 5, 10, 50
            Increments for a half-barb, a barb, and a flag.

        Returns
        -------
        n_flags, n_barbs : int array
            For each entry in *mag*, the number of flags and barbs.
        half_flag : bool array
            For each entry in *mag*, whether a half-barb is needed.
        empty_flag : bool array
            For each entry in *mag*, whether nothing is drawn.
        """
        # If rounding, round to the nearest multiple of half, the smallest
        # increment
        if rounding:
            mag = half * np.around(mag / half)
        n_flags, mag = divmod(mag, flag)
        n_barb, mag = divmod(mag, full)
        half_flag = mag >= half
        empty_flag = ~(half_flag | (n_flags > 0) | (n_barb > 0))
        return n_flags.astype(int), n_barb.astype(int), half_flag, empty_flag

    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
                    pivot, sizes, fill_empty, flip):
        """
        Create the wind barbs.

        Parameters
        ----------
        u, v
            Components of the vector in the x and y directions, respectively.

        nflags, nbarbs, half_barb, empty_flag
            Respectively, the number of flags, number of barbs, flag for
            half a barb, and flag for empty barb, ostensibly obtained from
            :meth:`_find_tails`.

        length
            The length of the barb staff in points.

        pivot : {"tip", "middle"} or number
            The point on the barb around which the entire barb should be
            rotated.  If a number, the start of the barb is shifted by that
            many points from the origin.

        sizes : dict
            Coefficients specifying the ratio of a given feature to the length
            of the barb. These features include:

            - *spacing*: space between features (flags, full/half barbs).
            - *height*: distance from shaft of top of a flag or full barb.
            - *width*: width of a flag, twice the width of a full barb.
            - *emptybarb*: radius of the circle used for low magnitudes.

        fill_empty : bool
            Whether the circle representing an empty barb should be filled or
            not (this changes the drawing of the polygon).

        flip : list of bool
            Whether the features should be flipped to the other side of the
            barb (useful for winds in the southern hemisphere).

        Returns
        -------
        list of arrays of vertices
            Polygon vertices for each of the wind barbs.  These polygons have
            been rotated to properly align with the vector direction.
        """

        # These control the spacing and size of barb elements relative to the
        # length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)

        # Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length / 2.)

        endx = 0.0
        try:
            endy = float(pivot)
        except ValueError:
            endy = pivot_points[pivot.lower()]

        # Get the appropriate angle for the vector components.  The offset is
        # due to the way the barb is initially drawn, going down the y-axis.
        # This makes sense in a meteorological mode of thinking since there 0
        # degrees corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi / 2)

        # Used for low magnitude.  We just get the vertices, so if we make it
        # out here, it can be reused.  The center set here should put the
        # center of the circle at the location(offset), rather than at the
        # same point as the barb pivot; this seems more sensible.
        circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()
        if fill_empty:
            empty_barb = circ
        else:
            # If we don't want the empty one filled, we make a degenerate
            # polygon that wraps back over itself
            empty_barb = np.concatenate((circ, circ[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            # If the vector magnitude is too weak to draw anything, plot an
            # empty circle instead
            if empty_flag[index]:
                # We can skip the transform since the circle has no preferred
                # orientation
                barb_list.append(empty_barb)
                continue

            poly_verts = [(endx, endy)]
            offset = length

            # Handle if this barb should be flipped
            barb_height = -full_height if flip[index] else full_height

            # Add vertices for each flag
            for i in range(nflags[index]):
                # The spacing that works for the barbs is a little to much for
                # the flags, but this only occurs when we have more than 1
                # flag.
                if offset != length:
                    offset += spacing / 2.
                poly_verts.extend(
                    [[endx, endy + offset],
                     [endx + barb_height, endy - full_width / 2 + offset],
                     [endx, endy - full_width + offset]])

                offset -= full_width + spacing

            # Add vertices for each barb.  These really are lines, but works
            # great adding 3 vertices that basically pull the polygon out and
            # back down the line
            for i in range(nbarbs[index]):
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + barb_height, endy + offset + full_width / 2),
                     (endx, endy + offset)])

                offset -= spacing

            # Add the vertices for half a barb, if needed
            if half_barb[index]:
                # If the half barb is the first on the staff, traditionally it
                # is offset from the end to make it easy to distinguish from a
                # barb with a full one
                if offset == length:
                    poly_verts.append((endx, endy + offset))
                    offset -= 1.5 * spacing
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + barb_height / 2, endy + offset + full_width / 4),
                     (endx, endy + offset)])

            # Rotate the barb according the angle. Making the barb first and
            # then rotating it made the math for drawing the barb really easy.
            # Also, the transform framework makes doing the rotation simple.
            poly_verts = transforms.Affine2D().rotate(-angle).transform(
                poly_verts)
            barb_list.append(poly_verts)

        return barb_list

    def set_UVC(self, U, V, C=None):
        # We need to ensure we have a copy, not a reference to an array that
        # might change before draw().
        self.u = ma.masked_invalid(U, copy=True).ravel()
        self.v = ma.masked_invalid(V, copy=True).ravel()

        # Flip needs to have the same number of entries as everything else.
        # Use broadcast_to to avoid a bloated array of identical values.
        # (can't rely on actual broadcasting)
        if len(self.flip) == 1:
            flip = np.broadcast_to(self.flip, self.u.shape)
        else:
            flip = self.flip

        if C is not None:
            c = ma.masked_invalid(C, copy=True).ravel()
            x, y, u, v, c, flip = cbook.delete_masked_points(
                self.x.ravel(), self.y.ravel(), self.u, self.v, c,
                flip.ravel())
            _check_consistent_shapes(x, y, u, v, c, flip)
        else:
            x, y, u, v, flip = cbook.delete_masked_points(
                self.x.ravel(), self.y.ravel(), self.u, self.v, flip.ravel())
            _check_consistent_shapes(x, y, u, v, flip)

        magnitude = np.hypot(u, v)
        flags, barbs, halves, empty = self._find_tails(
            magnitude, self.rounding, **self.barb_increments)

        # Get the vertices for each of the barbs

        plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty,
                                      self._length, self._pivot, self.sizes,
                                      self.fill_empty, flip)
        self.set_verts(plot_barbs)

        # Set the color array
        if C is not None:
            self.set_array(c)

        # Update the offsets in case the masked data changed
        xy = np.column_stack((x, y))
        self._offsets = xy
        self.stale = True

    def set_offsets(self, xy):
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and masks them as appropriate for the existing U/V data.

        Parameters
        ----------
        xy : sequence of pairs of floats
        """
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        x, y, u, v = cbook.delete_masked_points(
            self.x.ravel(), self.y.ravel(), self.u, self.v)
        _check_consistent_shapes(x, y, u, v)
        xy = np.column_stack((x, y))
        super().set_offsets(xy)
        self.stale = True
<|endoftext|>"""
The legend module defines the Legend class, which is responsible for
drawing legends associated with Axes and/or figures.

.. important::

    It is unlikely that you would ever create a Legend instance manually.
    Most users would normally create a legend via the `~.Axes.legend`
    function. For more details on legends there is also a :ref:`legend guide
    <legend_guide>`.

The `Legend` class is a container of legend handles and legend texts.

The legend handler map specifies how to create legend handles from artists
(lines, patches, etc.) in the Axes or figures. Default legend handlers are
defined in the :mod:`~matplotlib.legend_handler` module. While not all artist
types are covered by the default legend handlers, custom legend handlers can be
defined to support arbitrary objects.

See the :ref`<legend_guide>` for more
information.
"""

import itertools
import logging
import numbers
import time

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring, cbook, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
                                StepPatch)
from matplotlib.collections import (
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection)
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
    AnchoredOffsetbox, DraggableOffsetBox,
    HPacker, VPacker,
    DrawingArea, TextArea,
)
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler


class DraggableLegend(DraggableOffsetBox):
    def __init__(self, legend, use_blit=False, update="loc"):
        """
        Wrapper around a `.Legend` to support mouse dragging.

        Parameters
        ----------
        legend : `.Legend`
            The `.Legend` instance to wrap.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            If "loc", update the *loc* parameter of the legend upon finalizing.
            If "bbox", update the *bbox_to_anchor* parameter.
        """
        self.legend = legend

        _api.check_in_list(["loc", "bbox"], update=update)
        self._update = update

        super().__init__(legend, legend._legend_box, use_blit=use_blit)

    def finalize_offset(self):
        if self._update == "loc":
            self._update_loc(self.get_loc_in_canvas())
        elif self._update == "bbox":
            self._update_bbox_to_anchor(self.get_loc_in_canvas())

    def _update_loc(self, loc_in_canvas):
        bbox = self.legend.get_bbox_to_anchor()
        # if bbox has zero width or height, the transformation is
        # ill-defined. Fall back to the default bbox_to_anchor.
        if bbox.width == 0 or bbox.height == 0:
            self.legend.set_bbox_to_anchor(None)
            bbox = self.legend.get_bbox_to_anchor()
        _bbox_transform = BboxTransformFrom(bbox)
        self.legend._loc = tuple(_bbox_transform.transform(loc_in_canvas))

    def _update_bbox_to_anchor(self, loc_in_canvas):
        loc_in_bbox = self.legend.axes.transAxes.transform(loc_in_canvas)
        self.legend.set_bbox_to_anchor(loc_in_bbox)


_legend_kw_doc_base = """
bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
    Box that is used to position the legend in conjunction with *loc*.
    Defaults to ``axes.bbox`` (if called as a method to `.Axes.legend`) or
    ``figure.bbox`` (if ``figure.legend``).  This argument allows arbitrary
    placement of the legend.

    Bbox coordinates are interpreted in the coordinate system given by
    *bbox_transform*, with the default transform
    Axes or Figure coordinates, depending on which ``legend`` is called.

    If a 4-tuple or `.BboxBase` is given, then it specifies the bbox
    ``(x, y, width, height)`` that the legend is placed in.
    To put the legend in the best location in the bottom right
    quadrant of the Axes (or figure)::

        loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)

    A 2-tuple ``(x, y)`` places the corner of the legend specified by *loc* at
    x, y.  For example, to put the legend's upper right-hand corner in the
    center of the Axes (or figure) the following keywords can be used::

        loc='upper right', bbox_to_anchor=(0.5, 0.5)

ncols : int, default: 1
    The number of columns that the legend has.

    For backward compatibility, the spelling *ncol* is also supported
    but it is discouraged. If both are given, *ncols* takes precedence.

prop : None or `~matplotlib.font_manager.FontProperties` or dict
    The font properties of the legend. If None (default), the current
    :data:`matplotlib.rcParams` will be used.

fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', \
'x-large', 'xx-large'}
    The font size of the legend. If the value is numeric the size will be the
    absolute font size in points. String values are relative to the current
    default font size. This argument is only used if *prop* is not specified.

labelcolor : str or list, default: :rc:`legend.labelcolor`
    The color of the text in the legend. Either a valid color string
    (for example, 'red'), or a list of color strings. The labelcolor can
    also be made to match the color of the line or marker using 'linecolor',
    'markerfacecolor' (or 'mfc'), or 'markeredgecolor' (or 'mec').

    Labelcolor can be set globally using :rc:`legend.labelcolor`. If None,
    use :rc:`text.color`.

numpoints : int, default: :rc:`legend.numpoints`
    The number of marker points in the legend when creating a legend
    entry for a `.Line2D` (line).

scatterpoints : int, default: :rc:`legend.scatterpoints`
    The number of marker points in the legend when creating
    a legend entry for a `.PathCollection` (scatter plot).

scatteryoffsets : iterable of floats, default: ``[0.375, 0.5, 0.3125]``
    The vertical offset (relative to the font size) for the markers
    created for a scatter plot legend entry. 0.0 is at the base the
    legend text, and 1.0 is at the top. To draw all markers at the
    same height, set to ``[0.5]``.

markerscale : float, default: :rc:`legend.markerscale`
    The relative size of legend markers compared to the originally drawn ones.

markerfirst : bool, default: True
    If *True*, legend marker is placed to the left of the legend label.
    If *False*, legend marker is placed to the right of the legend label.

reverse : bool, default: False
    If *True*, the legend labels are displayed in reverse order from the input.
    If *False*, the legend labels are displayed in the same order as the input.

    .. versionadded:: 3.7

frameon : bool, default: :rc:`legend.frameon`
    Whether the legend should be drawn on a patch (frame).

fancybox : bool, default: :rc:`legend.fancybox`
    Whether round edges should be enabled around the `.FancyBboxPatch` which
    makes up the legend's background.

shadow : None, bool or dict, default: :rc:`legend.shadow`
    Whether to draw a shadow behind the legend.
    The shadow can be configured using `.Patch` keywords.
    Customization via :rc:`legend.shadow` is currently not supported.

framealpha : float, default: :rc:`legend.framealpha`
    The alpha transparency of the legend's background.
    If *shadow* is activated and *framealpha* is ``None``, the default value is
    ignored.

facecolor : "inherit" or color, default: :rc:`legend.facecolor`
    The legend's background color.
    If ``"inherit"``, use :rc:`axes.facecolor`.

edgecolor : "inherit" or color, default: :rc:`legend.edgecolor`
    The legend's background patch edge color.
    If ``"inherit"``, use :rc:`axes.edgecolor`.

mode : {"expand", None}
    If *mode* is set to ``"expand"`` the legend will be horizontally
    expanded to fill the Axes area (or *bbox_to_anchor* if defines
    the legend's size).

bbox_transform : None or `~matplotlib.transforms.Transform`
    The transform for the bounding box (*bbox_to_anchor*). For a value
    of ``None`` (default) the Axes'
    :data:`~matplotlib.axes.Axes.transAxes` transform will be used.

title : str or None
    The legend's title. Default is no title (``None``).

title_fontproperties : None or `~matplotlib.font_manager.FontProperties` or dict
    The font properties of the legend's title. If None (default), the
    *title_fontsize* argument will be used if present; if *title_fontsize* is
    also None, the current :rc:`legend.title_fontsize` will be used.

title_fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', \
'x-large', 'xx-large'}, default: :rc:`legend.title_fontsize`
    The font size of the legend's title.
    Note: This cannot be combined with *title_fontproperties*. If you want
    to set the fontsize alongside other font properties, use the *size*
    parameter in *title_fontproperties*.

alignment : {'center', 'left', 'right'}, default: 'center'
    The alignment of the legend title and the box of entries. The entries
    are aligned as a single block, so that markers always lined up.

borderpad : float, default: :rc:`legend.borderpad`
    The fractional whitespace inside the legend border, in font-size units.

labelspacing : float, default: :rc:`legend.labelspacing`
    The vertical space between the legend entries, in font-size units.

handlelength : float, default: :rc:`legend.handlelength`
    The length of the legend handles, in font-size units.

handleheight : float, default: :rc:`legend.handleheight`
    The height of the legend handles, in font-size units.

handletextpad : float, default: :rc:`legend.handletextpad`
    The pad between the legend handle and text, in font-size units.

borderaxespad : float, default: :rc:`legend.borderaxespad`
    The pad between the Axes and legend border, in font-size units.

columnspacing : float, default: :rc:`legend.columnspacing`
    The spacing between columns, in font-size units.

handler_map : dict or None
    The custom dictionary mapping instances or types to a legend
    handler. This *handler_map* updates the default handler map
    found at `matplotlib.legend.Legend.get_legend_handler_map`.

draggable : bool, default: False
    Whether the legend can be dragged with the mouse.
"""

_loc_doc_base = """
loc : str or pair of floats, default: {default}
    The location of the legend.

    The strings ``'upper left'``, ``'upper right'``, ``'lower left'``,
    ``'lower right'`` place the legend at the corresponding corner of the
    {parent}.

    The strings ``'upper center'``, ``'lower center'``, ``'center left'``,
    ``'center right'`` place the legend at the center of the corresponding edge
    of the {parent}.

    The string ``'center'`` places the legend at the center of the {parent}.
{best}
    The location can also be a 2-tuple giving the coordinates of the lower-left
    corner of the legend in {parent} coordinates (in which case *bbox_to_anchor*
    will be ignored).

    For back-compatibility, ``'center right'`` (but no other location) can also
    be spelled ``'right'``, and each "string" location can also be given as a
    numeric value:

    ==================   =============
    Location String      Location Code
    ==================   =============
    'best' (Axes only)   0
    'upper right'        1
    'upper left'         2
    'lower left'         3
    'lower right'        4
    'right'              5
    'center left'        6
    'center right'       7
    'lower center'       8
    'upper center'       9
    'center'             10
    ==================   =============
    {outside}"""

_loc_doc_best = """
    The string ``'best'`` places the legend at the location, among the nine
    locations defined so far, with the minimum overlap with other drawn
    artists.  This option can be quite slow for plots with large amounts of
    data; your plotting speed may benefit from providing a specific location.
"""

_legend_kw_axes_st = (
    _loc_doc_base.format(parent='axes', default=':rc:`legend.loc`',
                         best=_loc_doc_best, outside='') +
    _legend_kw_doc_base)
_docstring.interpd.register(_legend_kw_axes=_legend_kw_axes_st)

_outside_doc = """
    If a figure is using the constrained layout manager, the string codes
    of the *loc* keyword argument can get better layout behaviour using the
    prefix 'outside'. There is ambiguity at the corners, so 'outside
    upper right' will make space for the legend above the rest of the
    axes in the layout, and 'outside right upper' will make space on the
    right side of the layout.  In addition to the values of *loc*
    listed above, we have 'outside right upper', 'outside right lower',
    'outside left upper', and 'outside left lower'.  See
    :ref:`legend_guide` for more details.
"""

_legend_kw_figure_st = (
    _loc_doc_base.format(parent='figure', default="'upper right'",
                         best='', outside=_outside_doc) +
    _legend_kw_doc_base)
_docstring.interpd.register(_legend_kw_figure=_legend_kw_figure_st)

_legend_kw_both_st = (
    _loc_doc_base.format(parent='axes/figure',
                         default=":rc:`legend.loc` for Axes, 'upper right' for Figure",
                         best=_loc_doc_best, outside=_outside_doc) +
    _legend_kw_doc_base)
_docstring.interpd.register(_legend_kw_doc=_legend_kw_both_st)

_legend_kw_set_loc_st = (
    _loc_doc_base.format(parent='axes/figure',
                         default=":rc:`legend.loc` for Axes, 'upper right' for Figure",
                         best=_loc_doc_best, outside=_outside_doc))
_docstring.interpd.register(_legend_kw_set_loc_doc=_legend_kw_set_loc_st)


class Legend(Artist):
    """
    Place a legend on the figure/axes.
    """

    # 'best' is only implemented for Axes legends
    codes = {'best': 0, **AnchoredOffsetbox.codes}
    zorder = 5

    def __str__(self):
        return "Legend"

    @_docstring.interpd
    def __init__(
        self, parent, handles, labels,
        *,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the Axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        """
        Parameters
        ----------
        parent : `~matplotlib.axes.Axes` or `.Figure`
            The artist that contains the legend.

        handles : list of (`.Artist` or tuple of `.Artist`)
            A list of Artists (lines, patches) to be added to the legend.

        labels : list of str
            A list of labels to show next to the artists. The length of handles
            and labels should be the same. If they are not, they are truncated
            to the length of the shorter list.

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        Attributes
        ----------
        legend_handles
            List of `.Artist` objects added as legend entries.

            .. versionadded:: 3.7
        """
        # local import only to avoid circularity
        from matplotlib.axes import Axes
        from matplotlib.figure import FigureBase

        super().__init__()

        if prop is None:
            self.prop = FontProperties(size=mpl._val_or_rc(fontsize, "legend.fontsize"))
        else:
            self.prop = FontProperties._from_any(prop)
            if isinstance(prop, dict) and "size" not in prop:
                self.prop.set_size(mpl.rcParams["legend.fontsize"])

        self._fontsize = self.prop.get_size_in_points()

        self.texts = []
        self.legend_handles = []
        self._legend_title_box = None

        #: A dictionary with the extra handler mappings for this Legend
        #: instance.
        self._custom_handler_map = handler_map

        self.numpoints = mpl._val_or_rc(numpoints, 'legend.numpoints')
        self.markerscale = mpl._val_or_rc(markerscale, 'legend.markerscale')
        self.scatterpoints = mpl._val_or_rc(scatterpoints, 'legend.scatterpoints')
        self.borderpad = mpl._val_or_rc(borderpad, 'legend.borderpad')
        self.labelspacing = mpl._val_or_rc(labelspacing, 'legend.labelspacing')
        self.handlelength = mpl._val_or_rc(handlelength, 'legend.handlelength')
        self.handleheight = mpl._val_or_rc(handleheight, 'legend.handleheight')
        self.handletextpad = mpl._val_or_rc(handletextpad, 'legend.handletextpad')
        self.borderaxespad = mpl._val_or_rc(borderaxespad, 'legend.borderaxespad')
        self.columnspacing = mpl._val_or_rc(columnspacing, 'legend.columnspacing')
        self.shadow = mpl._val_or_rc(shadow, 'legend.shadow')

        if reverse:
            labels = [*reversed(labels)]
            handles = [*reversed(handles)]

        handles = list(handles)
        if len(handles) < 2:
            ncols = 1
        self._ncols = ncols if ncols != 1 else ncol

        if self.numpoints <= 0:
            raise ValueError("numpoints must be > 0; it was %d" % numpoints)

        # introduce y-offset for handles of the scatter plot
        if scatteryoffsets is None:
            self._scatteryoffsets = np.array([3. / 8., 4. / 8., 2.5 / 8.])
        else:
            self._scatteryoffsets = np.asarray(scatteryoffsets)
        reps = self.scatterpoints // len(self._scatteryoffsets) + 1
        self._scatteryoffsets = np.tile(self._scatteryoffsets,
                                        reps)[:self.scatterpoints]

        # _legend_box is a VPacker instance that contains all
        # legend items and will be initialized from _init_legend_box()
        # method.
        self._legend_box = None

        if isinstance(parent, Axes):
            self.isaxes = True
            self.axes = parent
            self.set_figure(parent.get_figure(root=False))
        elif isinstance(parent, FigureBase):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError(
                "Legend needs either Axes or FigureBase as parent"
            )
        self.parent = parent

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        # Figure out if self.shadow is valid
        # If shadow was None, rcParams loads False
        # So it shouldn't be None here

        self._shadow_props = {'ox': 2, 'oy': -2}  # default location offsets
        if isinstance(self.shadow, dict):
            self._shadow_props.update(self.shadow)
            self.shadow = True
        elif self.shadow in (0, 1, True, False):
            self.shadow = bool(self.shadow)
        else:
            raise ValueError(
                'Legend shadow must be a dict or bool, not '
                f'{self.shadow!r} of type {type(self.shadow)}.'
            )

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.

        facecolor = mpl._val_or_rc(facecolor, "legend.facecolor")
        if facecolor == 'inherit':
            facecolor = mpl.rcParams["axes.facecolor"]

        edgecolor = mpl._val_or_rc(edgecolor, "legend.edgecolor")
        if edgecolor == 'inherit':
            edgecolor = mpl.rcParams["axes.edgecolor"]

        fancybox = mpl._val_or_rc(fancybox, "legend.fancybox")

        self.legendPatch = FancyBboxPatch(
            xy=(0, 0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            # If shadow is used, default to alpha=1 (#8943).
            alpha=(framealpha if framealpha is not None
                   else 1 if shadow
                   else mpl.rcParams["legend.framealpha"]),
            # The width and height of the legendPatch will be set (in draw())
            # to the length that includes the padding. Thus we set pad=0 here.
            boxstyle=("round,pad=0,rounding_size=0.2" if fancybox
                      else "square,pad=0"),
            mutation_scale=self._fontsize,
            snap=True,
            visible=mpl._val_or_rc(frameon, "legend.frameon")
        )
        self._set_artist_props(self.legendPatch)

        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        self._alignment = alignment

        # init with null renderer
        self._init_legend_box(handles, labels, markerfirst)

        # Set legend location
        self.set_loc(loc)

        # figure out title font properties:
        if title_fontsize is not None and title_fontproperties is not None:
            raise ValueError(
                "title_fontsize and title_fontproperties can't be specified "
                "at the same time. Only use one of them. ")
        title_prop_fp = FontProperties._from_any(title_fontproperties)
        if isinstance(title_fontproperties, dict):
            if "size" not in title_fontproperties:
                title_fontsize = mpl.rcParams["legend.title_fontsize"]
                title_prop_fp.set_size(title_fontsize)
        elif title_fontsize is not None:
            title_prop_fp.set_size(title_fontsize)
        elif not isinstance(title_fontproperties, FontProperties):
            title_fontsize = mpl.rcParams["legend.title_fontsize"]
            title_prop_fp.set_size(title_fontsize)

        self.set_title(title, prop=title_prop_fp)

        self._draggable = None
        self.set_draggable(state=draggable)

        # set the text color

        color_getters = {  # getter function depends on line or patch
            'linecolor':       ['get_color',           'get_facecolor'],
            'markerfacecolor': ['get_markerfacecolor', 'get_facecolor'],
            'mfc':             ['get_markerfacecolor', 'get_facecolor'],
            'markeredgecolor': ['get_markeredgecolor', 'get_edgecolor'],
            'mec':             ['get_markeredgecolor', 'get_edgecolor'],
        }
        labelcolor = mpl._val_or_rc(labelcolor, 'legend.labelcolor')
        if labelcolor is None:
            labelcolor = mpl.rcParams['text.color']
        if isinstance(labelcolor, str) and labelcolor in color_getters:
            getter_names = color_getters[labelcolor]
            for handle, text in zip(self.legend_handles, self.texts):
                try:
                    if handle.get_array() is not None:
                        continue
                except AttributeError:
                    pass
                for getter_name in getter_names:
                    try:
                        color = getattr(handle, getter_name)()
                        if isinstance(color, np.ndarray):
                            if (
                                    color.shape[0] == 1
                                    or np.isclose(color, color[0]).all()
                            ):
                                text.set_color(color[0])
                            else:
                                pass
                        else:
                            text.set_color(color)
                        break
                    except AttributeError:
                        pass
        elif cbook._str_equal(labelcolor, 'none'):
            for text in self.texts:
                text.set_color(labelcolor)
        elif np.iterable(labelcolor):
            for text, color in zip(self.texts,
                                   itertools.cycle(
                                       colors.to_rgba_array(labelcolor))):
                text.set_color(color)
        else:
            raise ValueError(f"Invalid labelcolor: {labelcolor!r}")

    def _set_artist_props(self, a):
        """
        Set the boilerplate props for artists added to Axes.
        """
        a.set_figure(self.get_figure(root=False))
        if self.isaxes:
            a.axes = self.axes

        a.set_transform(self.get_transform())

    @_docstring.interpd
    def set_loc(self, loc=None):
        """
        Set the location of the legend.

        .. versionadded:: 3.8

        Parameters
        ----------
        %(_legend_kw_set_loc_doc)s
        """
        loc0 = loc
        self._loc_used_default = loc is None
        if loc is None:
            loc = mpl.rcParams["legend.loc"]
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'

        type_err_message = ("loc must be string, coordinate tuple, or"
                            f" an integer 0-10, not {loc!r}")

        # handle outside legends:
        self._outside_loc = None
        if isinstance(loc, str):
            if loc.split()[0] == 'outside':
                # strip outside:
                loc = loc.split('outside ')[1]
                # strip "center" at the beginning
                self._outside_loc = loc.replace('center ', '')
                # strip first
                self._outside_loc = self._outside_loc.split()[0]
                locs = loc.split()
                if len(locs) > 1 and locs[0] in ('right', 'left'):
                    # locs doesn't accept "left upper", etc, so swap
                    if locs[0] != 'center':
                        locs = locs[::-1]
                    loc = locs[0] + ' ' + locs[1]
            # check that loc is in acceptable strings
            loc = _api.check_getitem(self.codes, loc=loc)
        elif np.iterable(loc):
            # coerce iterable into tuple
            loc = tuple(loc)
            # validate the tuple represents Real coordinates
            if len(loc) != 2 or not all(isinstance(e, numbers.Real) for e in loc):
                raise ValueError(type_err_message)
        elif isinstance(loc, int):
            # validate the integer represents a string numeric value
            if loc < 0 or loc > 10:
                raise ValueError(type_err_message)
        else:
            # all other cases are invalid values of loc
            raise ValueError(type_err_message)

        if self.isaxes and self._outside_loc:
            raise ValueError(
                f"'outside' option for loc='{loc0}' keyword argument only "
                "works for figure legends")

        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        tmp = self._loc_used_default
        self._set_loc(loc)
        self._loc_used_default = tmp  # ignore changes done by _set_loc

    def _set_loc(self, loc):
        # find_offset function will be provided to _legend_box and
        # _legend_box will draw itself at the location of the return
        # value of the find_offset.
        self._loc_used_default = False
        self._loc_real = loc
        self.stale = True
        self._legend_box.set_offset(self._findoffset)

    def set_ncols(self, ncols):
        """Set the number of columns."""
        self._ncols = ncols

    def _get_loc(self):
        return self._loc_real

    _loc = property(_get_loc, _set_loc)

    def _findoffset(self, width, height, xdescent, ydescent, renderer):
        """Helper function to locate the legend."""

        if self._loc == 0:  # "best".
            x, y = self._find_best_position(width, height, renderer)
        elif self._loc in Legend.codes.values():  # Fixed location.
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
        else:  # Axes or figure coordinates.
            fx, fy = self._loc
            bbox = self.get_bbox_to_anchor()
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

        return x + xdescent, y + ydescent

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return

        renderer.open_group('legend', gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # if mode == fill, set the width of the legend_box to the
        # width of the parent (minus pads)
        if self._mode in ["expand"]:
            pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
            self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

        # update the location and size of the legend. This needs to
        # be done in any case to clip the figure right.
        bbox = self._legend_box.get_window_extent(renderer)
        self.legendPatch.set_bounds(bbox.bounds)
        self.legendPatch.set_mutation_scale(fontsize)

        # self.shadow is validated in __init__
        # So by here it is a bool and self._shadow_props contains any configs

        if self.shadow:
            Shadow(self.legendPatch, **self._shadow_props).draw(renderer)

        self.legendPatch.draw(renderer)
        self._legend_box.draw(renderer)

        renderer.close_group('legend')
        self.stale = False

    # _default_handler_map defines the default mapping between plot
    # elements and the legend handlers.

    _default_handler_map = {
        StemContainer: legend_handler.HandlerStem(),
        ErrorbarContainer: legend_handler.HandlerErrorbar(),
        Line2D: legend_handler.HandlerLine2D(),
        Patch: legend_handler.HandlerPatch(),
        StepPatch: legend_handler.HandlerStepPatch(),
        LineCollection: legend_handler.HandlerLineCollection(),
        RegularPolyCollection: legend_handler.HandlerRegularPolyCollection(),
        CircleCollection: legend_handler.HandlerCircleCollection(),
        BarContainer: legend_handler.HandlerPatch(
            update_func=legend_handler.update_from_first_child),
        tuple: legend_handler.HandlerTuple(),
        PathCollection: legend_handler.HandlerPathCollection(),
        PolyCollection: legend_handler.HandlerPolyCollection()
        }

    # (get|set|update)_default_handler_maps are public interfaces to
    # modify the default handler map.

    @classmethod
    def get_default_handler_map(cls):
        """Return the global default handler map, shared by all legends."""
        return cls._default_handler_map

    @classmethod
    def set_default_handler_map(cls, handler_map):
        """Set the global default handler map, shared by all legends."""
        cls._default_handler_map = handler_map

    @classmethod
    def update_default_handler_map(cls, handler_map):
        """Update the global default handler map, shared by all legends."""
        cls._default_handler_map.update(handler_map)

    def get_legend_handler_map(self):
        """Return this legend instance's handler map."""
        default_handler_map = self.get_default_handler_map()
        return ({**default_handler_map, **self._custom_handler_map}
                if self._custom_handler_map else default_handler_map)

    @staticmethod
    def get_legend_handler(legend_handler_map, orig_handle):
        """
        Return a legend handler from *legend_handler_map* that
        corresponds to *orig_handler*.

        *legend_handler_map* should be a dictionary object (that is
        returned by the get_legend_handler_map method).

        It first checks if the *orig_handle* itself is a key in the
        *legend_handler_map* and return the associated value.
        Otherwise, it checks for each of the classes in its
        method-resolution-order. If no matching key is found, it
        returns ``None``.
        """
        try:
            return legend_handler_map[orig_handle]
        except (TypeError, KeyError):  # TypeError if unhashable.
            pass
        for handle_type in type(orig_handle).mro():
            try:
                return legend_handler_map[handle_type]
            except KeyError:
                pass
        return None

    def _init_legend_box(self, handles, labels, markerfirst=True):
        """
        Initialize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self._fontsize

        # legend_box is a HPacker, horizontally packed with columns.
        # Each column is a VPacker, vertically packed with legend items.
        # Each legend item is a HPacker packed with:
        # - handlebox: a DrawingArea which contains the legend handle.
        # - labelbox: a TextArea which contains the legend text.

        text_list = []  # the list of text instances
        handle_list = []  # the list of handle instances
        handles_and_labels = []

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.
        height = fontsize * self.handleheight - descent
        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their coordinates should
        # be given in the display coordinates.

        # The transformation of each handle will be automatically set
        # to self.get_transform(). If the artist does not use its
        # default transform (e.g., Collections), you need to
        # manually set their transform to the self.get_transform().
        legend_handler_map = self.get_legend_handler_map()

        for orig_handle, label in zip(handles, labels):
            handler = self.get_legend_handler(legend_handler_map, orig_handle)
            if handler is None:
                _api.warn_external(
                             "Legend does not support handles for "
                             f"{type(orig_handle).__name__} "
                             "instances.\nA proxy artist may be used "
                             "instead.\nSee: https://matplotlib.org/"
                             "stable/users/explain/axes/legend_guide.html"
                             "#controlling-the-legend-entries")
                # No handle for this artist, so we just defer to None.
                handle_list.append(None)
            else:
                textbox = TextArea(label, multilinebaseline=True,
                                   textprops=dict(
                                       verticalalignment='baseline',
                                       horizontalalignment='left',
                                       fontproperties=self.prop))
                handlebox = DrawingArea(width=self.handlelength * fontsize,
                                        height=height,
                                        xdescent=0., ydescent=descent)

                text_list.append(textbox._text)
                # Create the artist for the legend which represents the
                # original artist/handle.
                handle_list.append(handler.legend_artist(self, orig_handle,
                                                         fontsize, handlebox))
                handles_and_labels.append((handlebox, textbox))

        columnbox = []
        # array_split splits n handles_and_labels into ncols columns, with the
        # first n%ncols columns having an extra entry.  filter(len, ...)
        # handles the case where n < ncols: the last ncols-n columns are empty
        # and get filtered out.
        for handles_and_labels_column in filter(
                len, np.array_split(handles_and_labels, self._ncols)):
            # pack handlebox and labelbox into itembox
            itemboxes = [HPacker(pad=0,
                                 sep=self.handletextpad * fontsize,
                                 children=[h, t] if markerfirst else [t, h],
                                 align="baseline")
                         for h, t in handles_and_labels_column]
            # pack columnbox
            alignment = "baseline" if markerfirst else "right"
            columnbox.append(VPacker(pad=0,
                                     sep=self.labelspacing * fontsize,
                                     align=alignment,
                                     children=itemboxes))

        mode = "expand" if self._mode == "expand" else "fixed"
        sep = self.columnspacing * fontsize
        self._legend_handle_box = HPacker(pad=0,
                                          sep=sep, align="baseline",
                                          mode=mode,
                                          children=columnbox)
        self._legend_title_box = TextArea("")
        self._legend_box = VPacker(pad=self.borderpad * fontsize,
                                   sep=self.labelspacing * fontsize,
                                   align=self._alignment,
                                   children=[self._legend_title_box,
                                             self._legend_handle_box])
        self._legend_box.set_figure(self.get_figure(root=False))
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legend_handles = handle_list

    def _auto_legend_data(self, renderer):
        """
        Return display coordinates for hit testing for "best" positioning.

        Returns
        -------
        bboxes
            List of bounding boxes of all patches.
        lines
            List of `.Path` corresponding to each line.
        offsets
            List of (x, y) offsets of all collection.
        """
        assert self.isaxes  # always holds, as this is only called internally
        bboxes = []
        lines = []
        offsets = []
        for artist in self.parent._children:
            if isinstance(artist, Line2D):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Rectangle):
                bboxes.append(
                    artist.get_bbox().transformed(artist.get_data_transform()))
            elif isinstance(artist, Patch):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, PolyCollection):
                lines.extend(artist.get_transform().transform_path(path)
                             for path in artist.get_paths())
            elif isinstance(artist, Collection):
                transform, transOffset, hoffsets, _ = artist._prepare_points()
                if len(hoffsets):
                    offsets.extend(transOffset.transform(hoffsets))
            elif isinstance(artist, Text):
                bboxes.append(artist.get_window_extent(renderer))

        return bboxes, lines, offsets

    def get_children(self):
        # docstring inherited
        return [self._legend_box, self.get_frame()]

    def get_frame(self):
        """Return the `~.patches.Rectangle` used to frame the legend."""
        return self.legendPatch

    def get_lines(self):
        r"""Return the list of `~.lines.Line2D`\s in the legend."""
        return [h for h in self.legend_handles if isinstance(h, Line2D)]

    def get_patches(self):
        r"""Return the list of `~.patches.Patch`\s in the legend."""
        return silent_list('Patch',
                           [h for h in self.legend_handles
                            if isinstance(h, Patch)])

    def get_texts(self):
        r"""Return the list of `~.text.Text`\s in the legend."""
        return silent_list('Text', self.texts)

    def set_alignment(self, alignment):
        """
        Set the alignment of the legend title and the box of entries.

        The entries are aligned as a single block, so that markers always
        lined up.

        Parameters
        ----------
        alignment : {'center', 'left', 'right'}.

        """
        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        self._alignment = alignment
        self._legend_box.align = alignment

    def get_alignment(self):
        """Get the alignment value of the legend box"""
        return self._legend_box.align

    def set_title(self, title, prop=None):
        """
        Set legend title and title style.

        Parameters
        ----------
        title : str
            The legend title.

        prop : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            The font properties of the legend title.
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.

        """
        self._legend_title_box._text.set_text(title)
        if title:
            self._legend_title_box._text.set_visible(True)
            self._legend_title_box.set_visible(True)
        else:
            self._legend_title_box._text.set_visible(False)
            self._legend_title_box.set_visible(False)

        if prop is not None:
            self._legend_title_box._text.set_fontproperties(prop)

        self.stale = True

    def get_title(self):
        """Return the `.Text` instance for the legend title."""
        return self._legend_title_box._text

    def get_window_extent(self, renderer=None):
        # docstring inherited
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        return self._legend_box.get_window_extent(renderer=renderer)

    def get_tightbbox(self, renderer=None):
        # docstring inherited
        return self._legend_box.get_window_extent(renderer)

    def get_frame_on(self):
        """Get whether the legend box patch is drawn."""
        return self.legendPatch.get_visible()

    def set_frame_on(self, b):
        """
        Set whether the legend box patch is drawn.

        Parameters
        ----------
        b : bool
        """
        self.legendPatch.set_visible(b)
        self.stale = True

    draw_frame = set_frame_on  # Backcompat alias.

    def get_bbox_to_anchor(self):
        """Return the bbox that the legend will be anchored to."""
        if self._bbox_to_anchor is None:
            return self.parent.bbox
        else:
            return self._bbox_to_anchor

    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the legend will be anchored to.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or tuple
            The bounding box can be specified in the following ways:

            - A `.BboxBase` instance
            - A tuple of ``(left, bottom, width, height)`` in the given
              transform (normalized axes coordinate if None)
            - A tuple of ``(left, bottom)`` where the width and height will be
              assumed to be zero.
            - *None*, to remove the bbox anchoring, and use the parent bbox.

        transform : `~matplotlib.transforms.Transform`, optional
            A transform to apply to the bounding box. If not specified, this
            will use a transform to the bounding box of the parent.
        """
        if bbox is None:
            self._bbox_to_anchor = None
            return
        elif isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f"Invalid bbox: {bbox}") from err

            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        if transform is None:
            transform = BboxTransformTo(self.parent.bbox)

        self._bbox_to_anchor = TransformedBbox(self._bbox_to_anchor,
                                               transform)
        self.stale = True

    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x, y) coordinate of the bbox.

        Parameters
        ----------
        loc : int
            A location code in range(1, 11). This corresponds to the possible
            values for ``self._loc``, excluding "best".
        bbox : `~matplotlib.transforms.Bbox`
            bbox to be placed, in display coordinates.
        parentbbox : `~matplotlib.transforms.Bbox`
            A parent box which will contain the bbox, in display coordinates.
        """
        return offsetbox._get_anchored_bbox(
            loc, bbox, parentbbox,
            self.borderaxespad * renderer.points_to_pixels(self._fontsize))

    def _find_best_position(self, width, height, renderer):
        """Determine the best location to place the legend."""
        assert self.isaxes  # always holds, as this is only called internally

        start_time = time.perf_counter()

        bboxes, lines, offsets = self._auto_legend_data(renderer)

        bbox = Bbox.from_bounds(0, 0, width, height)

        candidates = []
        for idx in range(1, len(self.codes)):
            l, b = self._get_anchored_bbox(idx, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
            legendBox = Bbox.from_bounds(l, b, width, height)
            # XXX TODO: If markers are present, it would be good to take them
            # into account when checking vertex overlaps in the next line.
            badness = (sum(legendBox.count_contains(line.vertices)
                           for line in lines)
                       + legendBox.count_contains(offsets)
                       + legendBox.count_overlaps(bboxes)
                       + sum(line.intersects_bbox(legendBox, filled=False)
                             for line in lines))
            # Include the index to favor lower codes in case of a tie.
            candidates.append((badness, idx, (l, b)))
            if badness == 0:
                break

        _, _, (l, b) = min(candidates)

        if self._loc_used_default and time.perf_counter() - start_time > 1:
            _api.warn_external(
                'Creating legend with loc="best" can be slow with large '
                'amounts of data.')

        return l, b

    def contains(self, mouseevent):
        return self.legendPatch.contains(mouseevent)

    def set_draggable(self, state, use_blit=False, update='loc'):
        """
        Enable or disable mouse dragging support of the legend.

        Parameters
        ----------
        state : bool
            Whether mouse dragging is enabled.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            The legend parameter to be changed when dragged:

            - 'loc': update the *loc* parameter of the legend
            - 'bbox': update the *bbox_to_anchor* parameter of the legend

        Returns
        -------
        `.DraggableLegend` or *None*
            If *state* is ``True`` this returns the `.DraggableLegend` helper
            instance. Otherwise this returns *None*.
        """
        if state:
            if self._draggable is None:
                self._draggable = DraggableLegend(self,
                                                  use_blit,
                                                  update=update)
        else:
            if self._draggable is not None:
                self._draggable.disconnect()
            self._draggable = None
        return self._draggable

    def get_draggable(self):
        """Return ``True`` if the legend is draggable, ``False`` otherwise."""
        return self._draggable is not None


# Helper functions to parse legend arguments for both `figure.legend` and
# `axes.legend`:
def _get_legend_handles(axs, legend_handler_map=None):
    """Yield artists that can be used as handles in a legend."""
    handles_original = []
    for ax in axs:
        handles_original += [
            *(a for a in ax._children
              if isinstance(a, (Line2D, Patch, Collection, Text))),
            *ax.containers]
        # support parasite Axes:
        if hasattr(ax, 'parasites'):
            for axx in ax.parasites:
                handles_original += [
                    *(a for a in axx._children
                      if isinstance(a, (Line2D, Patch, Collection, Text))),
                    *axx.containers]

    handler_map = {**Legend.get_default_handler_map(),
                   **(legend_handler_map or {})}
    has_handler = Legend.get_legend_handler
    for handle in handles_original:
        label = handle.get_label()
        if label != '_nolegend_' and has_handler(handler_map, handle):
            yield handle
        elif (label and not label.startswith('_') and
                not has_handler(handler_map, handle)):
            _api.warn_external(
                             "Legend does not support handles for "
                             f"{type(handle).__name__} "
                             "instances.\nSee: https://matplotlib.org/stable/"
                             "tutorials/intermediate/legend_guide.html"
                             "#implementing-a-custom-legend-handler")
            continue


def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    handles = []
    labels = []
    for handle in _get_legend_handles(axs, legend_handler_map):
        label = handle.get_label()
        if label and not label.startswith('_'):
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _parse_legend_args(axs, *args, handles=None, labels=None, **kwargs):
    """
    Get the handles and labels from the calls to either ``figure.legend``
    or ``axes.legend``.

    The parser is a bit involved because we support::

        legend()
        legend(labels)
        legend(handles, labels)
        legend(labels=labels)
        legend(handles=handles)
        legend(handles=handles, labels=labels)

    The behavior for a mixture of positional and keyword handles and labels
    is undefined and issues a warning; it will be an error in the future.

    Parameters
    ----------
    axs : list of `.Axes`
        If handles are not given explicitly, the artists in these Axes are
        used as handles.
    *args : tuple
        Positional parameters passed to ``legend()``.
    handles
        The value of the keyword argument ``legend(handles=...)``, or *None*
        if that keyword argument was not used.
    labels
        The value of the keyword argument ``legend(labels=...)``, or *None*
        if that keyword argument was not used.
    **kwargs
        All other keyword arguments passed to ``legend()``.

    Returns
    -------
    handles : list of (`.Artist` or tuple of `.Artist`)
        The legend handles.
    labels : list of str
        The legend labels.
    kwargs : dict
        *kwargs* with keywords handles and labels removed.

    """
    log = logging.getLogger(__name__)

    handlers = kwargs.get('handler_map')

    if (handles is not None or labels is not None) and args:
        _api.warn_deprecated("3.9", message=(
            "You have mixed positional and keyword arguments, some input may "
            "be discarded.  This is deprecated since %(since)s and will "
            "become an error in %(removal)s."))

    if (hasattr(handles, "__len__") and
            hasattr(labels, "__len__") and
            len(handles) != len(labels)):
        _api.warn_external(f"Mismatched number of handles and labels: "
                           f"len(handles) = {len(handles)} "
                           f"len(labels) = {len(labels)}")
    # if got both handles and labels as kwargs, make same length
    if handles and labels:
        handles, labels = zip(*zip(handles, labels))

    elif handles is not None and labels is None:
        labels = [handle.get_label() for handle in handles]

    elif labels is not None and handles is None:
        # Get as many handles as there are labels.
        handles = [handle for handle, label
                   in zip(_get_legend_handles(axs, handlers), labels)]

    elif len(args) == 0:  # 0 args: automatically detect labels and handles.
        handles, labels = _get_legend_handles_labels(axs, handlers)
        if not handles:
            _api.warn_external(
                "No artists with labels found to put in legend.  Note that "
                "artists whose label start with an underscore are ignored "
                "when legend() is called with no argument.")

    elif len(args) == 1:  # 1 arg: user defined labels, automatic handle detection.
        labels, = args
        if any(isinstance(l, Artist) for l in labels):
            raise TypeError("A single argument passed to legend() must be a "
                            "list of labels, but found an Artist in there.")

        # Get as many handles as there are labels.
        handles = [handle for handle, label
                   in zip(_get_legend_handles(axs, handlers), labels)]

    elif len(args) == 2:  # 2 args: user defined handles and labels.
        handles, labels = args[:2]

    else:
        raise _api.nargs_error('legend', '0-2', len(args))

    return handles, labels, kwargs
<|endoftext|>"""
GUI neutral widgets
===================

Widgets that are designed to work for any of the GUI backends.
All of these widgets require you to predefine an `~.axes.Axes`
instance and pass that as the first parameter.  Matplotlib doesn't try to
be too smart with respect to layout -- you will have to figure out how
wide and tall you want your Axes to be to accommodate your widget.
"""

from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number

from cycler import cycler
import numpy as np

import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
               text as mtext, ticker, transforms)
from .lines import Line2D
from .patches import Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D


class LockDraw:
    """
    Some widgets, like the cursor, draw onto the canvas, and this is not
    desirable under all circumstances, like when the toolbar is in zoom-to-rect
    mode and drawing a rectangle.  To avoid this, a widget can acquire a
    canvas' lock with ``canvas.widgetlock(widget)`` before drawing on the
    canvas; this will prevent other widgets from doing so at the same time (if
    they also try to acquire the lock first).
    """

    def __init__(self):
        self._owner = None

    def __call__(self, o):
        """Reserve the lock for *o*."""
        if not self.available(o):
            raise ValueError('already locked')
        self._owner = o

    def release(self, o):
        """Release the lock from *o*."""
        if not self.available(o):
            raise ValueError('you do not own this lock')
        self._owner = None

    def available(self, o):
        """Return whether drawing is available to *o*."""
        return not self.locked() or self.isowner(o)

    def isowner(self, o):
        """Return whether *o* owns this lock."""
        return self._owner is o

    def locked(self):
        """Return whether the lock is currently held by an owner."""
        return self._owner is not None


class Widget:
    """
    Abstract base class for GUI neutral widgets.
    """
    drawon = True
    eventson = True
    _active = True

    def set_active(self, active):
        """Set whether the widget is active."""
        self._active = active

    def get_active(self):
        """Get whether the widget is active."""
        return self._active

    # set_active is overridden by SelectorWidgets.
    active = property(get_active, set_active, doc="Is the widget active?")

    def ignore(self, event):
        """
        Return whether *event* should be ignored.

        This method should be called at the beginning of any event callback.
        """
        return not self.active


class AxesWidget(Widget):
    """
    Widget connected to a single `~matplotlib.axes.Axes`.

    To guarantee that the widget remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the widget.  If there are no references to the widget
    object it may be garbage collected which will disconnect the callbacks.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    canvas : `~matplotlib.backend_bases.FigureCanvasBase`
        The parent figure canvas for the widget.
    active : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, ax):
        self.ax = ax
        self._cids = []

    canvas = property(lambda self: self.ax.get_figure(root=True).canvas)

    def connect_event(self, event, callback):
        """
        Connect a callback function with an event.

        This should be used in lieu of ``figure.canvas.mpl_connect`` since this
        function stores callback ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self._cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self._cids:
            self.canvas.mpl_disconnect(c)

    def _get_data_coords(self, event):
        """Return *event*'s data coordinates in this widget's Axes."""
        # This method handles the possibility that event.inaxes != self.ax (which may
        # occur if multiple Axes are overlaid), in which case event.xdata/.ydata will
        # be wrong.  Note that we still special-case the common case where
        # event.inaxes == self.ax and avoid re-running the inverse data transform,
        # because that can introduce floating point errors for synthetic events.
        return ((event.xdata, event.ydata) if event.inaxes is self.ax
                else self.ax.transData.inverted().transform((event.x, event.y)))


class Button(AxesWidget):
    """
    A GUI neutral button.

    For the button to remain responsive you must keep a reference to it.
    Call `.on_clicked` to connect to the button.

    Attributes
    ----------
    ax
        The `~.axes.Axes` the button renders into.
    label
        A `.Text` instance.
    color
        The color of the button when not hovering.
    hovercolor
        The color of the button when hovering.
    """

    def __init__(self, ax, label, image=None,
                 color='0.85', hovercolor='0.95', *, useblit=True):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            The button text.
        image : array-like or PIL Image
            The image to place in the button, if not *None*.  The parameter is
            directly forwarded to `~.axes.Axes.imshow`.
        color : :mpltype:`color`
            The color of the button when not activated.
        hovercolor : :mpltype:`color`
            The color of the button when the mouse is over it.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        if image is not None:
            ax.imshow(image)
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        self._useblit = useblit and self.canvas.supports_blit

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        ax.set_navigate(False)
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.color = color
        self.hovercolor = hovercolor

    def _click(self, event):
        if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0]:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)

    def _release(self, event):
        if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)
        if self.eventson and self.ax.contains(event)[0]:
            self._observers.process('clicked', event)

    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                if self._useblit:
                    self.ax.draw_artist(self.ax)
                    self.canvas.blit(self.ax.bbox)
                else:
                    self.canvas.draw()

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', lambda event: func(event))

    def disconnect(self, cid):
        """Remove the callback function with connection id *cid*."""
        self._observers.disconnect(cid)


class SliderBase(AxesWidget):
    """
    The base class for constructing Slider widgets. Not intended for direct
    usage.

    For the slider to remain responsive you must maintain a reference to it.
    """
    def __init__(self, ax, orientation, closedmin, closedmax,
                 valmin, valmax, valfmt, dragging, valstep):
        if ax.name == '3d':
            raise ValueError('Sliders cannot be added to 3D Axes')

        super().__init__(ax)
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)

        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        self.drag_active = False
        self.valfmt = valfmt

        if orientation == "vertical":
            ax.set_ylim((valmin, valmax))
            axis = ax.yaxis
        else:
            ax.set_xlim((valmin, valmax))
            axis = ax.xaxis

        self._fmt = axis.get_major_formatter()
        if not isinstance(self._fmt, ticker.ScalarFormatter):
            self._fmt = ticker.ScalarFormatter()
            self._fmt.set_axis(axis)
        self._fmt.set_useOffset(False)  # No additive offset.
        self._fmt.set_useMathText(True)  # x sign before multiplicative offset.

        ax.set_axis_off()
        ax.set_navigate(False)

        self.connect_event("button_press_event", self._update)
        self.connect_event("button_release_event", self._update)
        if dragging:
            self.connect_event("motion_notify_event", self._update)
        self._observers = cbook.CallbackRegistry(signals=["changed"])

    def _stepped_value(self, val):
        """Return *val* coerced to closest number in the ``valstep`` grid."""
        if isinstance(self.valstep, Number):
            val = (self.valmin
                   + round((val - self.valmin) / self.valstep) * self.valstep)
        elif self.valstep is not None:
            valstep = np.asanyarray(self.valstep)
            if valstep.ndim != 1:
                raise ValueError(
                    f"valstep must have 1 dimension but has {valstep.ndim}"
                )
            val = valstep[np.argmin(np.abs(valstep - val))]
        return val

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*.

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed.
        """
        self._observers.disconnect(cid)

    def reset(self):
        """Reset the slider to the initial value."""
        if np.any(self.val != self.valinit):
            self.set_val(self.valinit)


class Slider(SliderBase):
    """
    A slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in Axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : float
        Slider value.
    """

    def __init__(self, ax, label, valmin, valmax, *, valinit=0.5, valfmt=None,
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, valstep=None,
                 orientation='horizontal', initcolor='r',
                 track_color='lightgrey', handle_style=None, **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, default: 0.5
            The slider initial position.

        valfmt : str, default: None
            %-format string used to format the slider value.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        slidermin : Slider, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider *slidermin*.

        slidermax : Slider, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider *slidermax*.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float or array-like, default: None
            If a float, the slider will snap to multiples of *valstep*.
            If an array the slider will snap to the values in the array.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        initcolor : :mpltype:`color`, default: 'r'
            The color of the line at the *valinit* position. Set to ``'none'``
            for no line.

        track_color : :mpltype:`color`, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handle. Default values are

            ========= ===== ======= ========================================
            Key       Value Default Description
            ========= ===== ======= ========================================
            facecolor color 'white' The facecolor of the slider handle.
            edgecolor color '.75'   The edgecolor of the slider handle.
            size      int   10      The size of the slider handle in points.
            ========= ===== ======= ========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Rectangle` that draws the slider knob.  See the
        `.Rectangle` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax,
                         valmin, valmax, valfmt, dragging, valstep)

        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError(
                f"Argument slidermin ({type(slidermin)}) has no 'val'")
        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError(
                f"Argument slidermax ({type(slidermax)}) has no 'val'")
        self.slidermin = slidermin
        self.slidermax = slidermax
        valinit = self._value_in_bounds(valinit)
        if valinit is None:
            valinit = valmin
        self.val = valinit
        self.valinit = valinit

        defaults = {'facecolor': 'white', 'edgecolor': '.75', 'size': 10}
        handle_style = {} if handle_style is None else handle_style
        marker_props = {
            f'marker{k}': v for k, v in {**defaults, **handle_style}.items()
        }

        if orientation == 'vertical':
            self.track = Rectangle(
                (.25, 0), .5, 1,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axhspan(valmin, valinit, .25, .75, **kwargs)
            # Drawing a longer line and clipping it to the track avoids
            # pixelation-related asymmetries.
            self.hline = ax.axhline(valinit, 0, 1, color=initcolor, lw=1,
                                    clip_path=TransformedPatchPath(self.track))
            handleXY = [[0.5], [valinit]]
        else:
            self.track = Rectangle(
                (0, .25), 1, .5,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axvspan(valmin, valinit, .25, .75, **kwargs)
            self.vline = ax.axvline(valinit, 0, 1, color=initcolor, lw=1,
                                    clip_path=TransformedPatchPath(self.track))
            handleXY = [[valinit], [0.5]]
        self._handle, = ax.plot(
            *handleXY,
            "o",
            **marker_props,
            clip_on=False
        )

        if orientation == 'vertical':
            self.label = ax.text(0.5, 1.02, label, transform=ax.transAxes,
                                 verticalalignment='bottom',
                                 horizontalalignment='center')

            self.valtext = ax.text(0.5, -0.02, self._format(valinit),
                                   transform=ax.transAxes,
                                   verticalalignment='top',
                                   horizontalalignment='center')
        else:
            self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                                 verticalalignment='center',
                                 horizontalalignment='right')

            self.valtext = ax.text(1.02, 0.5, self._format(valinit),
                                   transform=ax.transAxes,
                                   verticalalignment='center',
                                   horizontalalignment='left')

        self.set_val(valinit)

    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
        val = self._stepped_value(val)

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == 'button_press_event' and self.ax.contains(event)[0]:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        if (event.name == 'button_release_event'
              or event.name == 'button_press_event' and not self.ax.contains(event)[0]):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        xdata, ydata = self._get_data_coords(event)
        val = self._value_in_bounds(
            xdata if self.orientation == 'horizontal' else ydata)
        if val not in [None, self.val]:
            self.set_val(val)

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return self.valfmt % val
        else:
            _, s, _ = self._fmt.format_ticks([self.valmin, val, self.valmax])
            # fmt.get_offset is actually the multiplicative factor, if any.
            return s + self._fmt.get_offset()

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : float
        """
        if self.orientation == 'vertical':
            self.poly.set_height(val - self.poly.get_y())
            self._handle.set_ydata([val])
        else:
            self.poly.set_width(val - self.poly.get_x())
            self._handle.set_xdata([val])
        self.valtext.set_text(self._format(val))
        if self.drawon:
            self.ax.get_figure(root=True).canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process('changed', val)

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))


class RangeSlider(SliderBase):
    """
    A slider representing a range of floating point values. Defines the min and
    max of the range via the *val* attribute as a tuple of (min, max).

    Create a slider that defines a range contained within [*valmin*, *valmax*]
    in Axes *ax*. For the slider to remain responsive you must maintain a
    reference to it. Call :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tuple of float
        Slider value.
    """

    def __init__(
        self,
        ax,
        label,
        valmin,
        valmax,
        *,
        valinit=None,
        valfmt=None,
        closedmin=True,
        closedmax=True,
        dragging=True,
        valstep=None,
        orientation="horizontal",
        track_color='lightgrey',
        handle_style=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : tuple of float or None, default: None
            The initial positions of the slider. If None the initial positions
            will be at the 25th and 75th percentiles of the range.

        valfmt : str, default: None
            %-format string used to format the slider values.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, default: None
            If given, the slider will snap to multiples of *valstep*.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        track_color : :mpltype:`color`, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handles. Default values are

            ========= ===== ======= =========================================
            Key       Value Default Description
            ========= ===== ======= =========================================
            facecolor color 'white' The facecolor of the slider handles.
            edgecolor color '.75'   The edgecolor of the slider handles.
            size      int   10      The size of the slider handles in points.
            ========= ===== ======= =========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Polygon` that draws the slider knob.  See the
        `.Polygon` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax,
                         valmin, valmax, valfmt, dragging, valstep)

        # Set a value to allow _value_in_bounds() to work.
        self.val = (valmin, valmax)
        if valinit is None:
            # Place at the 25th and 75th percentiles
            extent = valmax - valmin
            valinit = np.array([valmin + extent * 0.25,
                                valmin + extent * 0.75])
        else:
            valinit = self._value_in_bounds(valinit)
        self.val = valinit
        self.valinit = valinit

        defaults = {'facecolor': 'white', 'edgecolor': '.75', 'size': 10}
        handle_style = {} if handle_style is None else handle_style
        marker_props = {
            f'marker{k}': v for k, v in {**defaults, **handle_style}.items()
        }

        if orientation == "vertical":
            self.track = Rectangle(
                (.25, 0), .5, 2,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            poly_transform = self.ax.get_yaxis_transform(which="grid")
            handleXY_1 = [.5, valinit[0]]
            handleXY_2 = [.5, valinit[1]]
        else:
            self.track = Rectangle(
                (0, .25), 1, .5,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            poly_transform = self.ax.get_xaxis_transform(which="grid")
            handleXY_1 = [valinit[0], .5]
            handleXY_2 = [valinit[1], .5]
        self.poly = Polygon(np.zeros([5, 2]), **kwargs)
        self._update_selection_poly(*valinit)
        self.poly.set_transform(poly_transform)
        self.poly.get_path()._interpolation_steps = 100
        self.ax.add_patch(self.poly)
        self.ax._request_autoscale_view()
        self._handles = [
            ax.plot(
                *handleXY_1,
                "o",
                **marker_props,
                clip_on=False
            )[0],
            ax.plot(
                *handleXY_2,
                "o",
                **marker_props,
                clip_on=False
            )[0]
        ]

        if orientation == "vertical":
            self.label = ax.text(
                0.5,
                1.02,
                label,
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

            self.valtext = ax.text(
                0.5,
                -0.02,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
            )
        else:
            self.label = ax.text(
                -0.02,
                0.5,
                label,
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="right",
            )

            self.valtext = ax.text(
                1.02,
                0.5,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="left",
            )

        self._active_handle = None
        self.set_val(valinit)

    def _update_selection_poly(self, vmin, vmax):
        """
        Update the vertices of the *self.poly* slider in-place
        to cover the data range *vmin*, *vmax*.
        """
        # The vertices are positioned
        #  1 ------ 2
        #  |        |
        # 0, 4 ---- 3
        verts = self.poly.xy
        if self.orientation == "vertical":
            verts[0] = verts[4] = .25, vmin
            verts[1] = .25, vmax
            verts[2] = .75, vmax
            verts[3] = .75, vmin
        else:
            verts[0] = verts[4] = vmin, .25
            verts[1] = vmin, .75
            verts[2] = vmax, .75
            verts[3] = vmax, .25

    def _min_in_bounds(self, min):
        """Ensure the new min value is between valmin and self.val[1]."""
        if min <= self.valmin:
            if not self.closedmin:
                return self.val[0]
            min = self.valmin

        if min > self.val[1]:
            min = self.val[1]
        return self._stepped_value(min)

    def _max_in_bounds(self, max):
        """Ensure the new max value is between valmax and self.val[0]."""
        if max >= self.valmax:
            if not self.closedmax:
                return self.val[1]
            max = self.valmax

        if max <= self.val[0]:
            max = self.val[0]
        return self._stepped_value(max)

    def _value_in_bounds(self, vals):
        """Clip min, max values to the bounds."""
        return (self._min_in_bounds(vals[0]), self._max_in_bounds(vals[1]))

    def _update_val_from_pos(self, pos):
        """Update the slider value based on a given position."""
        idx = np.argmin(np.abs(self.val - pos))
        if idx == 0:
            val = self._min_in_bounds(pos)
            self.set_min(val)
        else:
            val = self._max_in_bounds(pos)
            self.set_max(val)
        if self._active_handle:
            if self.orientation == "vertical":
                self._active_handle.set_ydata([val])
            else:
                self._active_handle.set_xdata([val])

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == "button_press_event" and self.ax.contains(event)[0]:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        if (event.name == "button_release_event"
              or event.name == "button_press_event" and not self.ax.contains(event)[0]):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            self._active_handle = None
            return

        # determine which handle was grabbed
        xdata, ydata = self._get_data_coords(event)
        handle_index = np.argmin(np.abs(
            [h.get_xdata()[0] - xdata for h in self._handles]
            if self.orientation == "horizontal" else
            [h.get_ydata()[0] - ydata for h in self._handles]))
        handle = self._handles[handle_index]

        # these checks ensure smooth behavior if the handles swap which one
        # has a higher value. i.e. if one is dragged over and past the other.
        if handle is not self._active_handle:
            self._active_handle = handle

        self._update_val_from_pos(xdata if self.orientation == "horizontal" else ydata)

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return f"({self.valfmt % val[0]}, {self.valfmt % val[1]})"
        else:
            _, s1, s2, _ = self._fmt.format_ticks(
                [self.valmin, *val, self.valmax]
            )
            # fmt.get_offset is actually the multiplicative factor, if any.
            s1 += self._fmt.get_offset()
            s2 += self._fmt.get_offset()
            # Use f string to avoid issues with backslashes when cast to a str
            return f"({s1}, {s2})"

    def set_min(self, min):
        """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
        """
        self.set_val((min, self.val[1]))

    def set_max(self, max):
        """
        Set the lower value of the slider to *max*.

        Parameters
        ----------
        max : float
        """
        self.set_val((self.val[0], max))

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(val)
        _api.check_shape((2,), val=val)
        # Reset value to allow _value_in_bounds() to work.
        self.val = (self.valmin, self.valmax)
        vmin, vmax = self._value_in_bounds(val)
        self._update_selection_poly(vmin, vmax)
        if self.orientation == "vertical":
            self._handles[0].set_ydata([vmin])
            self._handles[1].set_ydata([vmax])
        else:
            self._handles[0].set_xdata([vmin])
            self._handles[1].set_xdata([vmax])

        self.valtext.set_text(self._format((vmin, vmax)))

        if self.drawon:
            self.ax.get_figure(root=True).canvas.draw_idle()
        self.val = (vmin, vmax)
        if self.eventson:
            self._observers.process("changed", (vmin, vmax))

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed. The function
            must accept a 2-tuple of floats as its argument.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))


def _expand_text_props(props):
    props = cbook.normalize_kwargs(props, mtext.Text)
    return cycler(**props)() if props else itertools.repeat({})


class CheckButtons(AxesWidget):
    r"""
    A GUI neutral set of check buttons.

    For the check buttons to remain responsive you must keep a
    reference to this object.

    Connect to the CheckButtons with the `~._Buttons.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    labels : list of `~matplotlib.text.Text`
        The text label objects of the check buttons.
    """

    def __init__(self, ax, labels, actives=None, *, useblit=True,
                 label_props=None, frame_props=None, check_props=None):
        """
        Add check buttons to `~.axes.Axes` instance *ax*.

        Parameters
        ----------
