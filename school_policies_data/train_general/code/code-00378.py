extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    Determines the ``%%(func)s``-coloring of values that are outside the
    *levels* range.

    If 'neither', values outside the *levels* range are not colored.  If 'min',
    'max' or 'both', color the values below, above or below and above the
    *levels* range.

    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
    under/over values of the `.Colormap`. Note that most colormaps do not have
    dedicated colors for these by default, so that the over and under values
    are the edge values of the colormap.  You may want to set these values
    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.

    .. note::

        An existing `.TriContourSet` does not get notified if properties of its
        colormap are changed. Therefore, an explicit call to
        `.ContourSet.changed()` is needed after modifying the colormap. The
        explicit call can be left out, if a colorbar is assigned to the
        `.TriContourSet` because it internally calls `.ContourSet.changed()`.

xunits, yunits : registered units, optional
    Override axis units by specifying an instance of a
    :class:`matplotlib.units.ConversionInterface`.

antialiased : bool, optional
    Enable antialiasing, overriding the defaults.  For
    filled contours, the default is *True*.  For line contours,
    it is taken from :rc:`lines.antialiased`.""" % _docstring.interpd.params)


@_docstring.Substitution(func='tricontour', type='lines')
@_docstring.interpd
def tricontour(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s

    linewidths : float or array-like, default: :rc:`contour.linewidth`
        The line width of the contour lines.

        If a number, all levels will be plotted with this linewidth.

        If a sequence, the levels in ascending order will be plotted with
        the linewidths in the order specified.

        If None, this falls back to :rc:`lines.linewidth`.

    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
        If *linestyles* is *None*, the default is 'solid' unless the lines are
        monochrome.  In that case, negative contours will take their linestyle
        from :rc:`contour.negative_linestyle` setting.

        *linestyles* can also be an iterable of the above strings specifying a
        set of linestyles to be used. If this iterable is shorter than the
        number of contour levels it will be repeated as necessary.
    """
    kwargs['filled'] = False
    return TriContourSet(ax, *args, **kwargs)


@_docstring.Substitution(func='tricontourf', type='regions')
@_docstring.interpd
def tricontourf(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s

    hatches : list[str], optional
        A list of crosshatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour.

    Notes
    -----
    `.tricontourf` fills intervals that are closed at the top; that is, for
    boundaries *z1* and *z2*, the filled region is::

        z1 < Z <= z2

    except for the lowest interval, which is closed on both sides (i.e. it
    includes the lowest value).
    """
    kwargs['filled'] = True
    return TriContourSet(ax, *args, **kwargs)
<|endoftext|>"""
Mesh refinement for triangular grids.
"""

import numpy as np

from matplotlib import _api
from matplotlib.tri._triangulation import Triangulation
import matplotlib.tri._triinterpolate


class TriRefiner:
    """
    Abstract base class for classes implementing mesh refinement.

    A TriRefiner encapsulates a Triangulation object and provides tools for
    mesh refinement and interpolation.

    Derived classes must implement:

    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
      the optional keyword arguments *kwargs* are defined in each
      TriRefiner concrete implementation, and which returns:

      - a refined triangulation,
      - optionally (depending on *return_tri_index*), for each
        point of the refined triangulation: the index of
        the initial triangulation triangle to which it belongs.

    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:

      - *z* array of field values (to refine) defined at the base
        triangulation nodes,
      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
      - the other optional keyword arguments *kwargs* are defined in
        each TriRefiner concrete implementation;

      and which returns (as a tuple) a refined triangular mesh and the
      interpolated values of the field at the refined triangulation nodes.
    """

    def __init__(self, triangulation):
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation


class UniformTriRefiner(TriRefiner):
    """
    Uniform mesh refinement by recursive subdivisions.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation (to be refined)
    """
#    See Also
#    --------
#    :class:`~matplotlib.tri.CubicTriInterpolator` and
#    :class:`~matplotlib.tri.TriAnalyzer`.
#    """
    def __init__(self, triangulation):
        super().__init__(triangulation)

    def refine_triangulation(self, return_tri_index=False, subdiv=3):
        """
        Compute a uniformly refined triangulation *refi_triangulation* of
        the encapsulated :attr:`triangulation`.

        This function refines the encapsulated triangulation by splitting each
        father triangle into 4 child sub-triangles built on the edges midside
        nodes, recursing *subdiv* times.  In the end, each triangle is hence
        divided into ``4**subdiv`` child triangles.

        Parameters
        ----------
        return_tri_index : bool, default: False
            Whether an index table indicating the father triangle index of each
            point is returned.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles;
            hence, the default results in 64 refined subtriangles for each
            triangle of the initial triangulation.

        Returns
        -------
        refi_triangulation : `~matplotlib.tri.Triangulation`
            The refined triangulation.
        found_index : int array
            Index of the initial triangulation containing triangle, for each
            point of *refi_triangulation*.
            Returned only if *return_tri_index* is set to True.
        """
        refi_triangulation = self._triangulation
        ntri = refi_triangulation.triangles.shape[0]

        # Computes the triangulation ancestors numbers in the reference
        # triangulation.
        ancestors = np.arange(ntri, dtype=np.int32)
        for _ in range(subdiv):
            refi_triangulation, ancestors = self._refine_triangulation_once(
                refi_triangulation, ancestors)
        refi_npts = refi_triangulation.x.shape[0]
        refi_triangles = refi_triangulation.triangles

        # Now we compute found_index table if needed
        if return_tri_index:
            # We have to initialize found_index with -1 because some nodes
            # may very well belong to no triangle at all, e.g., in case of
            # Delaunay Triangulation with DuplicatePointWarning.
            found_index = np.full(refi_npts, -1, dtype=np.int32)
            tri_mask = self._triangulation.mask
            if tri_mask is None:
                found_index[refi_triangles] = np.repeat(ancestors,
                                                        3).reshape(-1, 3)
            else:
                # There is a subtlety here: we want to avoid whenever possible
                # that refined points container is a masked triangle (which
                # would result in artifacts in plots).
                # So we impose the numbering from masked ancestors first,
                # then overwrite it with unmasked ancestor numbers.
                ancestor_mask = tri_mask[ancestors]
                found_index[refi_triangles[ancestor_mask, :]
                            ] = np.repeat(ancestors[ancestor_mask],
                                          3).reshape(-1, 3)
                found_index[refi_triangles[~ancestor_mask, :]
                            ] = np.repeat(ancestors[~ancestor_mask],
                                          3).reshape(-1, 3)
            return refi_triangulation, found_index
        else:
            return refi_triangulation

    def refine_field(self, z, triinterpolator=None, subdiv=3):
        """
        Refine a field defined on the encapsulated triangulation.

        Parameters
        ----------
        z : (npoints,) array-like
            Values of the field to refine, defined at the nodes of the
            encapsulated triangulation. (``n_points`` is the number of points
            in the initial triangulation)
        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
            Interpolator used for field interpolation. If not specified,
            a `~matplotlib.tri.CubicTriInterpolator` will be used.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles.

        Returns
        -------
        refi_tri : `~matplotlib.tri.Triangulation`
             The returned refined triangulation.
        refi_z : 1D array of length: *refi_tri* node count.
             The returned interpolated field (at *refi_tri* nodes).
        """
        if triinterpolator is None:
            interp = matplotlib.tri.CubicTriInterpolator(
                self._triangulation, z)
        else:
            _api.check_isinstance(matplotlib.tri.TriInterpolator,
                                  triinterpolator=triinterpolator)
            interp = triinterpolator

        refi_tri, found_index = self.refine_triangulation(
            subdiv=subdiv, return_tri_index=True)
        refi_z = interp._interpolate_multikeys(
            refi_tri.x, refi_tri.y, tri_index=found_index)[0]
        return refi_tri, refi_z

    @staticmethod
    def _refine_triangulation_once(triangulation, ancestors=None):
        """
        Refine a `.Triangulation` by splitting each triangle into 4
        child-masked_triangles built on the edges midside nodes.

        Masked triangles, if present, are also split, but their children
        returned masked.

        If *ancestors* is not provided, returns only a new triangulation:
        child_triangulation.

        If the array-like key table *ancestor* is given, it shall be of shape
        (ntri,) where ntri is the number of *triangulation* masked_triangles.
        In this case, the function returns
        (child_triangulation, child_ancestors)
        child_ancestors is defined so that the 4 child masked_triangles share
        the same index as their father: child_ancestors.shape = (4 * ntri,).
        """

        x = triangulation.x
        y = triangulation.y

        #    According to tri.triangulation doc:
        #         neighbors[i, j] is the triangle that is the neighbor
        #         to the edge from point index masked_triangles[i, j] to point
        #         index masked_triangles[i, (j+1)%3].
        neighbors = triangulation.neighbors
        triangles = triangulation.triangles
        npts = np.shape(x)[0]
        ntri = np.shape(triangles)[0]
        if ancestors is not None:
            ancestors = np.asarray(ancestors)
            if np.shape(ancestors) != (ntri,):
                raise ValueError(
                    "Incompatible shapes provide for "
                    "triangulation.masked_triangles and ancestors: "
                    f"{np.shape(triangles)} and {np.shape(ancestors)}")

        # Initiating tables refi_x and refi_y of the refined triangulation
        # points
        # hint: each apex is shared by 2 masked_triangles except the borders.
        borders = np.sum(neighbors == -1)
        added_pts = (3*ntri + borders) // 2
        refi_npts = npts + added_pts
        refi_x = np.zeros(refi_npts)
        refi_y = np.zeros(refi_npts)

        # First part of refi_x, refi_y is just the initial points
        refi_x[:npts] = x
        refi_y[:npts] = y

        # Second part contains the edge midside nodes.
        # Each edge belongs to 1 triangle (if border edge) or is shared by 2
        # masked_triangles (interior edge).
        # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
        # edge_apexes); we then extract only the masters to avoid overlaps.
        # The so-called 'master' is the triangle with biggest index
        # The 'slave' is the triangle with lower index
        # (can be -1 if border edge)
        # For slave and master we will identify the apex pointing to the edge
        # start
        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
        edge_neighbors = neighbors[edge_elems, edge_apexes]
        mask_masters = (edge_elems > edge_neighbors)

        # Identifying the "masters" and adding to refi_x, refi_y vec
        masters = edge_elems[mask_masters]
        apex_masters = edge_apexes[mask_masters]
        x_add = (x[triangles[masters, apex_masters]] +
                 x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
        y_add = (y[triangles[masters, apex_masters]] +
                 y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
        refi_x[npts:] = x_add
        refi_y[npts:] = y_add

        # Building the new masked_triangles; each old masked_triangles hosts
        # 4 new masked_triangles
        # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
        # 3 new_pt_midside
        new_pt_corner = triangles

        # What is the index in refi_x, refi_y of point at middle of apex iapex
        #  of elem ielem ?
        # If ielem is the apex master: simple count, given the way refi_x was
        #  built.
        # If ielem is the apex slave: yet we do not know; but we will soon
        # using the neighbors table.
        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
        cum_sum = npts
        for imid in range(3):
            mask_st_loc = (imid == apex_masters)
            n_masters_loc = np.sum(mask_st_loc)
            elem_masters_loc = masters[mask_st_loc]
            new_pt_midside[:, imid][elem_masters_loc] = np.arange(
                n_masters_loc, dtype=np.int32) + cum_sum
            cum_sum += n_masters_loc

        # Now dealing with slave elems.
        # for each slave element we identify the master and then the inode
        # once slave_masters is identified, slave_masters_apex is such that:
        # neighbors[slaves_masters, slave_masters_apex] == slaves
        mask_slaves = np.logical_not(mask_masters)
        slaves = edge_elems[mask_slaves]
        slaves_masters = edge_neighbors[mask_slaves]
        diff_table = np.abs(neighbors[slaves_masters, :] -
                            np.outer(slaves, np.ones(3, dtype=np.int32)))
        slave_masters_apex = np.argmin(diff_table, axis=1)
        slaves_apex = edge_apexes[mask_slaves]
        new_pt_midside[slaves, slaves_apex] = new_pt_midside[
            slaves_masters, slave_masters_apex]

        # Builds the 4 child masked_triangles
        child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
        child_triangles[0::4, :] = np.vstack([
            new_pt_corner[:, 0], new_pt_midside[:, 0],
            new_pt_midside[:, 2]]).T
        child_triangles[1::4, :] = np.vstack([
            new_pt_corner[:, 1], new_pt_midside[:, 1],
            new_pt_midside[:, 0]]).T
        child_triangles[2::4, :] = np.vstack([
            new_pt_corner[:, 2], new_pt_midside[:, 2],
            new_pt_midside[:, 1]]).T
        child_triangles[3::4, :] = np.vstack([
            new_pt_midside[:, 0], new_pt_midside[:, 1],
            new_pt_midside[:, 2]]).T
        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)

        # Builds the child mask
        if triangulation.mask is not None:
            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))

        if ancestors is None:
            return child_triangulation
        else:
            return child_triangulation, np.repeat(ancestors, 4)
<|endoftext|>import numpy as np

from matplotlib import _api, _docstring
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri._triangulation import Triangulation


@_docstring.interpd
def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
              vmax=None, shading='flat', facecolors=None, **kwargs):
    """
    Create a pseudocolor plot of an unstructured triangular grid.

    Call signatures::

      tripcolor(triangulation, c, *, ...)
      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)

    The triangular grid can be specified either by passing a `.Triangulation`
    object as the first parameter, or by passing the points *x*, *y* and
    optionally the *triangles* and a *mask*. See `.Triangulation` for an
    explanation of these parameters.

    It is possible to pass the triangles positionally, i.e.
    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
    For more clarity, pass *triangles* via keyword argument.

    If neither of *triangulation* or *triangles* are given, the triangulation
    is calculated on the fly. In this case, it does not make sense to provide
    colors at the triangle faces via *c* or *facecolors* because there are
    multiple possible triangulations for a group of points and you don't know
    which triangles will be constructed.

    Parameters
    ----------
    triangulation : `.Triangulation`
        An already created triangular grid.
    x, y, triangles, mask
        Parameters defining the triangular grid. See `.Triangulation`.
        This is mutually exclusive with specifying *triangulation*.
    c : array-like
        The color values, either for the points or for the triangles. Which one
        is automatically inferred from the length of *c*, i.e. does it match
        the number of points or the number of triangles. If there are the same
        number of points and triangles in the triangulation it is assumed that
        color values are defined at points; to force the use of color values at
        triangles use the keyword argument ``facecolors=c`` instead of just
        ``c``.
        This parameter is position-only.
    facecolors : array-like, optional
        Can be used alternatively to *c* to specify colors at the triangle
        faces. This parameter takes precedence over *c*.
    shading : {'flat', 'gouraud'}, default: 'flat'
        If  'flat' and the color values *c* are defined at points, the color
        values used for each triangle are from the mean c of the triangle's
        three points. If *shading* is 'gouraud' then color values must be
        defined at points.
    %(cmap_doc)s

    %(norm_doc)s

    %(vmin_vmax_doc)s

    %(colorizer_doc)s

    Returns
    -------
    `~matplotlib.collections.PolyCollection` or `~matplotlib.collections.TriMesh`
        The result depends on *shading*: For ``shading='flat'`` the result is a
        `.PolyCollection`, for ``shading='gouraud'`` the result is a `.TriMesh`.

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.collections.Collection` properties

        %(Collection:kwdoc)s
    """
    _api.check_in_list(['flat', 'gouraud'], shading=shading)

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    # Parse the color to be in one of (the other variable will be None):
    # - facecolors: if specified at the triangle faces
    # - point_colors: if specified at the points
    if facecolors is not None:
        if args:
            _api.warn_external(
                "Positional parameter c has no effect when the keyword "
                "facecolors is given")
        point_colors = None
        if len(facecolors) != len(tri.triangles):
            raise ValueError("The length of facecolors must match the number "
                             "of triangles")
    else:
        # Color from positional parameter c
        if not args:
            raise TypeError(
                "tripcolor() missing 1 required positional argument: 'c'; or "
                "1 required keyword-only argument: 'facecolors'")
        elif len(args) > 1:
            raise TypeError(f"Unexpected positional parameters: {args[1:]!r}")
        c = np.asarray(args[0])
        if len(c) == len(tri.x):
            # having this before the len(tri.triangles) comparison gives
            # precedence to nodes if there are as many nodes as triangles
            point_colors = c
            facecolors = None
        elif len(c) == len(tri.triangles):
            point_colors = None
            facecolors = c
        else:
            raise ValueError('The length of c must match either the number '
                             'of points or the number of triangles')

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)

    edgecolors = 'none'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', edgecolors)

    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    if 'antialiaseds' not in kwargs and ec.lower() == "none":
        kwargs['antialiaseds'] = False

    if shading == 'gouraud':
        if facecolors is not None:
            raise ValueError(
                "shading='gouraud' can only be used when the colors "
                "are specified at the points, not at the faces.")
        collection = TriMesh(tri, alpha=alpha, array=point_colors,
                             cmap=cmap, norm=norm, **kwargs)
    else:  # 'flat'
        # Vertices of triangles.
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # Color values.
        if facecolors is None:
            # One color per triangle, the mean of the 3 vertex color values.
            colors = point_colors[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # Remove color values of masked triangles.
            colors = facecolors[~tri.mask]
        else:
            colors = facecolors
        collection = PolyCollection(verts, alpha=alpha, array=colors,
                                    cmap=cmap, norm=norm, **kwargs)

    collection._scale_norm(norm, vmin, vmax)
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection
<|endoftext|>import numpy as np

from matplotlib import _api
from matplotlib.tri import Triangulation


class TriFinder:
    """
    Abstract base class for classes used to find the triangles of a
    Triangulation in which (x, y) points lie.

    Rather than instantiate an object of a class derived from TriFinder, it is
    usually better to use the function `.Triangulation.get_trifinder`.

    Derived classes implement __call__(x, y) where x and y are array-like point
    coordinates of the same shape.
    """

    def __init__(self, triangulation):
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

    def __call__(self, x, y):
        raise NotImplementedError


class TrapezoidMapTriFinder(TriFinder):
    """
    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
    map algorithm from the book "Computational Geometry, Algorithms and
    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
    and O. Schwarzkopf.

    The triangulation must be valid, i.e. it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.  The
    algorithm has some tolerance to triangles formed from colinear points, but
    this should not be relied upon.
    """

    def __init__(self, triangulation):
        from matplotlib import _tri
        super().__init__(triangulation)
        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
            triangulation.get_cpp_triangulation())
        self._initialize()

    def __call__(self, x, y):
        """
        Return an array containing the indices of the triangles in which the
        specified *x*, *y* points lie, or -1 for points that do not lie within
        a triangle.

        *x*, *y* are array-like x and y coordinates of the same shape and any
        number of dimensions.

        Returns integer array with the same shape and *x* and *y*.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must be array-like with the same shape")

        # C++ does the heavy lifting, and expects 1D arrays.
        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
                   .reshape(x.shape))
        return indices

    def _get_tree_stats(self):
        """
        Return a python list containing the statistics about the node tree:
            0: number of nodes (tree size)
            1: number of unique nodes
            2: number of trapezoids (tree leaf nodes)
            3: number of unique trapezoids
            4: maximum parent count (max number of times a node is repeated in
                   tree)
            5: maximum depth of tree (one more than the maximum number of
                   comparisons needed to search through the tree)
            6: mean of all trapezoid depths (one more than the average number
                   of comparisons needed to search through the tree)
        """
        return self._cpp_trifinder.get_tree_stats()

    def _initialize(self):
        """
        Initialize the underlying C++ object.  Can be called multiple times if,
        for example, the triangulation is modified.
        """
        self._cpp_trifinder.initialize()

    def _print_tree(self):
        """
        Print a text representation of the node tree, which is useful for
        debugging purposes.
        """
        self._cpp_trifinder.print_tree()
<|endoftext|>import math
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine


def _apply_theta_transforms_warn():
    _api.warn_deprecated(
                "3.9",
                message=(
                    "Passing `apply_theta_transforms=True` (the default) "
                    "is deprecated since Matplotlib %(since)s. "
                    "Support for this will be removed in Matplotlib in %(removal)s. "
                    "To prevent this warning, set `apply_theta_transforms=False`, "
                    "and make sure to shift theta values before being passed to "
                    "this transform."
                )
            )


class PolarTransform(mtransforms.Transform):
    r"""
    The base polar transform.

    This transform maps polar coordinates :math:`\theta, r` into Cartesian
    coordinates :math:`x, y = r \cos(\theta), r \sin(\theta)`
    (but does not fully transform into Axes coordinates or
    handle positioning in screen space).

    This transformation is designed to be applied to data after any scaling
    along the radial axis (e.g. log-scaling) has been applied to the input
    data.

    Path segments at a fixed radius are automatically transformed to circular
    arcs as long as ``path._interpolation_steps > 1``.
    """

    input_dims = output_dims = 2

    def __init__(self, axis=None, use_rmin=True, *,
                 apply_theta_transforms=True, scale_transform=None):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True``, subtract the minimum radial axis limit before
            transforming to Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        super().__init__()
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = apply_theta_transforms
        self._scale_transform = scale_transform
        if apply_theta_transforms:
            _apply_theta_transforms_warn()

    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        apply_theta_transforms="_apply_theta_transforms")

    def _get_rorigin(self):
        # Get lower r limit after being scaled by the radial scale transform
        return self._scale_transform.transform(
            (0, self._axis.get_rorigin()))[1]

    def transform_non_affine(self, values):
        # docstring inherited
        theta, r = np.transpose(values)
        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            theta *= self._axis.get_theta_direction()
            theta += self._axis.get_theta_offset()
        if self._use_rmin and self._axis is not None:
            r = (r - self._get_rorigin()) * self._axis.get_rsign()
        r = np.where(r >= 0, r, np.nan)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def transform_path_non_affine(self, path):
        # docstring inherited
        if not len(path) or path._interpolation_steps == 1:
            return Path(self.transform_non_affine(path.vertices), path.codes)
        xys = []
        codes = []
        last_t = last_r = None
        for trs, c in path.iter_segments():
            trs = trs.reshape((-1, 2))
            if c == Path.LINETO:
                (t, r), = trs
                if t == last_t:  # Same angle: draw a straight line.
                    xys.extend(self.transform_non_affine(trs))
                    codes.append(Path.LINETO)
                elif r == last_r:  # Same radius: draw an arc.
                    # The following is complicated by Path.arc() being
                    # "helpful" and unwrapping the angles, but we don't want
                    # that behavior here.
                    last_td, td = np.rad2deg([last_t, t])
                    if self._use_rmin and self._axis is not None:
                        r = ((r - self._get_rorigin())
                             * self._axis.get_rsign())
                    if last_td <= td:
                        while td - last_td > 360:
                            arc = Path.arc(last_td, last_td + 360)
                            xys.extend(arc.vertices[1:] * r)
                            codes.extend(arc.codes[1:])
                            last_td += 360
                        arc = Path.arc(last_td, td)
                        xys.extend(arc.vertices[1:] * r)
                        codes.extend(arc.codes[1:])
                    else:
                        # The reverse version also relies on the fact that all
                        # codes but the first one are the same.
                        while last_td - td > 360:
                            arc = Path.arc(last_td - 360, last_td)
                            xys.extend(arc.vertices[::-1][1:] * r)
                            codes.extend(arc.codes[1:])
                            last_td -= 360
                        arc = Path.arc(td, last_td)
                        xys.extend(arc.vertices[::-1][1:] * r)
                        codes.extend(arc.codes[1:])
                else:  # Interpolate.
                    trs = cbook.simple_linear_interpolation(
                        np.vstack([(last_t, last_r), trs]),
                        path._interpolation_steps)[1:]
                    xys.extend(self.transform_non_affine(trs))
                    codes.extend([Path.LINETO] * len(trs))
            else:  # Not a straight line.
                xys.extend(self.transform_non_affine(trs))
                codes.extend([c] * len(trs))
            last_t, last_r = trs[-1]
        return Path(xys, codes)

    def inverted(self):
        # docstring inherited
        return PolarAxes.InvertedPolarTransform(
            self._axis, self._use_rmin,
            apply_theta_transforms=self._apply_theta_transforms
        )


class PolarAffine(mtransforms.Affine2DBase):
    r"""
    The affine part of the polar projection.

    Scales the output so that maximum radius rests on the edge of the Axes
    circle and the origin is mapped to (0.5, 0.5). The transform applied is
    the same to x and y components and given by:

    .. math::

        x_{1} = 0.5 \left [ \frac{x_{0}}{(r_{\max} - r_{\min})} + 1 \right ]

    :math:`r_{\min}, r_{\max}` are the minimum and maximum radial limits after
    any scaling (e.g. log scaling) has been removed.
    """
    def __init__(self, scale_transform, limits):
        """
        Parameters
        ----------
        scale_transform : `~matplotlib.transforms.Transform`
            Scaling transform for the data. This is used to remove any scaling
            from the radial view limits.
        limits : `~matplotlib.transforms.BboxBase`
            View limits of the data. The only part of its bounds that is used
            is the y limits (for the radius limits).
        """
        super().__init__()
        self._scale_transform = scale_transform
        self._limits = limits
        self.set_children(scale_transform, limits)
        self._mtx = None

    __str__ = mtransforms._make_str_method("_scale_transform", "_limits")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            limits_scaled = self._limits.transformed(self._scale_transform)
            yscale = limits_scaled.ymax - limits_scaled.ymin
            affine = mtransforms.Affine2D() \
                .scale(0.5 / yscale) \
                .translate(0.5, 0.5)
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        return self._mtx


class InvertedPolarTransform(mtransforms.Transform):
    """
    The inverse of the polar transform, mapping Cartesian
    coordinate space *x* and *y* back to *theta* and *r*.
    """
    input_dims = output_dims = 2

    def __init__(self, axis=None, use_rmin=True,
                 *, apply_theta_transforms=True):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True``, add the minimum radial axis limit after
            transforming from Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        super().__init__()
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = apply_theta_transforms
        if apply_theta_transforms:
            _apply_theta_transforms_warn()

    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        apply_theta_transforms="_apply_theta_transforms")

    def transform_non_affine(self, values):
        # docstring inherited
        x, y = values.T
        r = np.hypot(x, y)
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            theta -= self._axis.get_theta_offset()
            theta *= self._axis.get_theta_direction()
            theta %= 2 * np.pi
        if self._use_rmin and self._axis is not None:
            r += self._axis.get_rorigin()
            r *= self._axis.get_rsign()
        return np.column_stack([theta, r])

    def inverted(self):
        # docstring inherited
        return PolarAxes.PolarTransform(
            self._axis, self._use_rmin,
            apply_theta_transforms=self._apply_theta_transforms
        )


class ThetaFormatter(mticker.Formatter):
    """
    Used to format the *theta* tick labels.  Converts the native
    unit of radians into degrees and adds a degree symbol.
    """

    def __call__(self, x, pos=None):
        vmin, vmax = self.axis.get_view_interval()
        d = np.rad2deg(abs(vmax - vmin))
        digits = max(-int(np.log10(d) - 1.5), 0)
        return f"{np.rad2deg(x):0.{digits}f}\N{DEGREE SIGN}"


class _AxisWrapper:
    def __init__(self, axis):
        self._axis = axis

    def get_view_interval(self):
        return np.rad2deg(self._axis.get_view_interval())

    def set_view_interval(self, vmin, vmax):
        self._axis.set_view_interval(*np.deg2rad((vmin, vmax)))

    def get_minpos(self):
        return np.rad2deg(self._axis.get_minpos())

    def get_data_interval(self):
        return np.rad2deg(self._axis.get_data_interval())

    def set_data_interval(self, vmin, vmax):
        self._axis.set_data_interval(*np.deg2rad((vmin, vmax)))

    def get_tick_space(self):
        return self._axis.get_tick_space()


class ThetaLocator(mticker.Locator):
    """
    Used to locate theta ticks.

    This will work the same as the base locator except in the case that the
    view spans the entire circle. In such cases, the previously used default
    locations of every 45 degrees are returned.
    """

    def __init__(self, base):
        self.base = base
        self.axis = self.base.axis = _AxisWrapper(self.base.axis)

    def set_axis(self, axis):
        self.axis = _AxisWrapper(axis)
        self.base.set_axis(self.axis)

    def __call__(self):
        lim = self.axis.get_view_interval()
        if _is_full_circle_deg(lim[0], lim[1]):
            return np.deg2rad(min(lim)) + np.arange(8) * 2 * np.pi / 8
        else:
            return np.deg2rad(self.base())

    def view_limits(self, vmin, vmax):
        vmin, vmax = np.rad2deg((vmin, vmax))
        return np.deg2rad(self.base.view_limits(vmin, vmax))


class ThetaTick(maxis.XTick):
    """
    A theta-axis tick.

    This subclass of `.XTick` provides angular ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    tick location. This results in ticks that are correctly perpendicular to
    the arc spine.

    When 'auto' rotation is enabled, labels are also rotated to be parallel to
    the spine. The label padding is also applied here since it's not possible
    to use a generic axes transform to produce tick-specific padding.
    """

    def __init__(self, axes, *args, **kwargs):
        self._text1_translate = mtransforms.ScaledTranslation(
            0, 0, axes.get_figure(root=False).dpi_scale_trans)
        self._text2_translate = mtransforms.ScaledTranslation(
            0, 0, axes.get_figure(root=False).dpi_scale_trans)
        super().__init__(axes, *args, **kwargs)
        self.label1.set(
            rotation_mode='anchor',
            transform=self.label1.get_transform() + self._text1_translate)
        self.label2.set(
            rotation_mode='anchor',
            transform=self.label2.get_transform() + self._text2_translate)

    def _apply_params(self, **kwargs):
        super()._apply_params(**kwargs)
        # Ensure transform is correct; sometimes this gets reset.
        trans = self.label1.get_transform()
        if not trans.contains_branch(self._text1_translate):
            self.label1.set_transform(trans + self._text1_translate)
        trans = self.label2.get_transform()
        if not trans.contains_branch(self._text2_translate):
            self.label2.set_transform(trans + self._text2_translate)

    def _update_padding(self, pad, angle):
        padx = pad * np.cos(angle) / 72
        pady = pad * np.sin(angle) / 72
        self._text1_translate._t = (padx, pady)
        self._text1_translate.invalidate()
        self._text2_translate._t = (-padx, -pady)
        self._text2_translate.invalidate()

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        angle = loc * axes.get_theta_direction() + axes.get_theta_offset()
        text_angle = np.rad2deg(angle) % 360 - 90
        angle -= np.pi / 2

        marker = self.tick1line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        marker = self.tick2line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans

        mode, user_angle = self._labelrotation
        if mode == 'default':
            text_angle = user_angle
        else:
            if text_angle > 90:
                text_angle -= 180
            elif text_angle < -90:
                text_angle += 180
            text_angle += user_angle
        self.label1.set_rotation(text_angle)
        self.label2.set_rotation(text_angle)

        # This extra padding helps preserve the look from previous releases but
        # is also needed because labels are anchored to their center.
        pad = self._pad + 7
        self._update_padding(pad,
                             self._loc * axes.get_theta_direction() +
                             axes.get_theta_offset())


class ThetaAxis(maxis.XAxis):
    """
    A theta Axis.

    This overrides certain properties of an `.XAxis` to provide special-casing
    for an angular axis.
    """
    __name__ = 'thetaaxis'
    axis_name = 'theta'  #: Read-only name identifying the axis.
    _tick_class = ThetaTick

    def _wrap_locator_formatter(self):
        self.set_major_locator(ThetaLocator(self.get_major_locator()))
        self.set_major_formatter(ThetaFormatter())
        self.isDefault_majloc = True
        self.isDefault_majfmt = True

    def clear(self):
        # docstring inherited
        super().clear()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        if value != 'linear':
            raise NotImplementedError(
                "The xscale cannot be set on a polar plot")
        super()._set_scale(value, **kwargs)
        # LinearScale.set_default_locators_and_formatters just set the major
        # locator to be an AutoLocator, so we customize it here to have ticks
        # at sensible degree multiples.
        self.get_major_locator().set_params(steps=[1, 1.5, 3, 4.5, 9, 10])
        self._wrap_locator_formatter()

    def _copy_tick_props(self, src, dest):
        """Copy the props from src tick to dest tick."""
        if src is None or dest is None:
            return
        super()._copy_tick_props(src, dest)

        # Ensure that tick transforms are independent so that padding works.
        trans = dest._get_text1_transform()[0]
        dest.label1.set_transform(trans + dest._text1_translate)
        trans = dest._get_text2_transform()[0]
        dest.label2.set_transform(trans + dest._text2_translate)


class RadialLocator(mticker.Locator):
    """
    Used to locate radius ticks.

    Ensures that all ticks are strictly positive.  For all other tasks, it
    delegates to the base `.Locator` (which may be different depending on the
    scale of the *r*-axis).
    """

    def __init__(self, base, axes=None):
        self.base = base
        self._axes = axes

    def set_axis(self, axis):
        self.base.set_axis(axis)

    def __call__(self):
        # Ensure previous behaviour with full circle non-annular views.
        if self._axes:
            if _is_full_circle_rad(*self._axes.viewLim.intervalx):
                rorigin = self._axes.get_rorigin() * self._axes.get_rsign()
                if self._axes.get_rmin() <= rorigin:
                    return [tick for tick in self.base() if tick > rorigin]
        return self.base()

    def _zero_in_bounds(self):
        """
        Return True if zero is within the valid values for the
        scale of the radial axis.
        """
        vmin, vmax = self._axes.yaxis._scale.limit_range_for_scale(0, 1, 1e-5)
        return vmin == 0

    def nonsingular(self, vmin, vmax):
        # docstring inherited
        if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
            # Initial view limits
            return (0, 1)
        else:
            return self.base.nonsingular(vmin, vmax)

    def view_limits(self, vmin, vmax):
        vmin, vmax = self.base.view_limits(vmin, vmax)
        if self._zero_in_bounds() and vmax > vmin:
            # this allows inverted r/y-lims
            vmin = min(0, vmin)
        return mtransforms.nonsingular(vmin, vmax)


class _ThetaShift(mtransforms.ScaledTranslation):
    """
    Apply a padding shift based on axes theta limits.

    This is used to create padding for radial ticks.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The owning Axes; used to determine limits.
    pad : float
        The padding to apply, in points.
    mode : {'min', 'max', 'rlabel'}
        Whether to shift away from the start (``'min'``) or the end (``'max'``)
        of the axes, or using the rlabel position (``'rlabel'``).
    """
    def __init__(self, axes, pad, mode):
        super().__init__(pad, pad, axes.get_figure(root=False).dpi_scale_trans)
        self.set_children(axes._realViewLim)
        self.axes = axes
        self.mode = mode
        self.pad = pad

    __str__ = mtransforms._make_str_method("axes", "pad", "mode")

    def get_matrix(self):
        if self._invalid:
            if self.mode == 'rlabel':
                angle = (
                    np.deg2rad(self.axes.get_rlabel_position()
                               * self.axes.get_theta_direction())
                    + self.axes.get_theta_offset()
                    - np.pi / 2
                )
            elif self.mode == 'min':
                angle = self.axes._realViewLim.xmin - np.pi / 2
            elif self.mode == 'max':
                angle = self.axes._realViewLim.xmax + np.pi / 2
            self._t = (self.pad * np.cos(angle) / 72, self.pad * np.sin(angle) / 72)
        return super().get_matrix()


class RadialTick(maxis.YTick):
    """
    A radial-axis tick.

    This subclass of `.YTick` provides radial ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    axes limits.  This results in ticks that are correctly perpendicular to
    the spine. Labels are also rotated to be perpendicular to the spine, when
    'auto' rotation is enabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label1.set_rotation_mode('anchor')
        self.label2.set_rotation_mode('anchor')

    def _determine_anchor(self, mode, angle, start):
        # Note: angle is the (spine angle - 90) because it's used for the tick
        # & text setup, so all numbers below are -90 from (normed) spine angle.
        if mode == 'auto':
            if start:
                if -90 <= angle <= 90:
                    return 'left', 'center'
                else:
                    return 'right', 'center'
            else:
                if -90 <= angle <= 90:
                    return 'right', 'center'
                else:
                    return 'left', 'center'
        else:
            if start:
                if angle < -68.5:
                    return 'center', 'top'
                elif angle < -23.5:
                    return 'left', 'top'
                elif angle < 22.5:
                    return 'left', 'center'
                elif angle < 67.5:
                    return 'left', 'bottom'
                elif angle < 112.5:
                    return 'center', 'bottom'
                elif angle < 157.5:
                    return 'right', 'bottom'
                elif angle < 202.5:
                    return 'right', 'center'
                elif angle < 247.5:
                    return 'right', 'top'
                else:
                    return 'center', 'top'
            else:
                if angle < -68.5:
                    return 'center', 'bottom'
                elif angle < -23.5:
                    return 'right', 'bottom'
                elif angle < 22.5:
                    return 'right', 'center'
                elif angle < 67.5:
                    return 'right', 'top'
                elif angle < 112.5:
                    return 'center', 'top'
                elif angle < 157.5:
                    return 'left', 'top'
                elif angle < 202.5:
                    return 'left', 'center'
                elif angle < 247.5:
                    return 'left', 'bottom'
                else:
                    return 'center', 'bottom'

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        thetamin = axes.get_thetamin()
        thetamax = axes.get_thetamax()
        direction = axes.get_theta_direction()
        offset_rad = axes.get_theta_offset()
        offset = np.rad2deg(offset_rad)
        full = _is_full_circle_deg(thetamin, thetamax)

        if full:
            angle = (axes.get_rlabel_position() * direction +
                     offset) % 360 - 90
            tick_angle = 0
        else:
            angle = (thetamin * direction + offset) % 360 - 90
            if direction > 0:
                tick_angle = np.deg2rad(angle)
            else:
                tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        if full:
            ha = self.label1.get_horizontalalignment()
            va = self.label1.get_verticalalignment()
        else:
            ha, va = self._determine_anchor(mode, angle, direction > 0)
        self.label1.set_horizontalalignment(ha)
        self.label1.set_verticalalignment(va)
        self.label1.set_rotation(text_angle)

        marker = self.tick1line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        if full:
            self.label2.set_visible(False)
            self.tick2line.set_visible(False)
        angle = (thetamax * direction + offset) % 360 - 90
        if direction > 0:
            tick_angle = np.deg2rad(angle)
        else:
            tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        ha, va = self._determine_anchor(mode, angle, direction < 0)
        self.label2.set_ha(ha)
        self.label2.set_va(va)
        self.label2.set_rotation(text_angle)

        marker = self.tick2line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans


class RadialAxis(maxis.YAxis):
    """
    A radial Axis.

    This overrides certain properties of a `.YAxis` to provide special-casing
    for a radial axis.
    """
    __name__ = 'radialaxis'
    axis_name = 'radius'  #: Read-only name identifying the axis.
    _tick_class = RadialTick

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sticky_edges.y.append(0)

    def _wrap_locator_formatter(self):
        self.set_major_locator(RadialLocator(self.get_major_locator(),
                                             self.axes))
        self.isDefault_majloc = True

    def clear(self):
        # docstring inherited
        super().clear()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        super()._set_scale(value, **kwargs)
        self._wrap_locator_formatter()


def _is_full_circle_deg(thetamin, thetamax):
    """
    Determine if a wedge (in degrees) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 360.0) < 1e-12


def _is_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 2 * np.pi) < 1.74e-14


class _WedgeBbox(mtransforms.Bbox):
    """
    Transform (theta, r) wedge Bbox into Axes bounding box.

    Parameters
    ----------
    center : (float, float)
        Center of the wedge
    viewLim : `~matplotlib.transforms.Bbox`
        Bbox determining the boundaries of the wedge
    originLim : `~matplotlib.transforms.Bbox`
        Bbox determining the origin for the wedge, if different from *viewLim*
    """
    def __init__(self, center, viewLim, originLim, **kwargs):
        super().__init__([[0, 0], [1, 1]], **kwargs)
        self._center = center
        self._viewLim = viewLim
        self._originLim = originLim
        self.set_children(viewLim, originLim)

    __str__ = mtransforms._make_str_method("_center", "_viewLim", "_originLim")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            points = self._viewLim.get_points().copy()
            # Scale angular limits to work with Wedge.
            points[:, 0] *= 180 / np.pi
            if points[0, 0] > points[1, 0]:
                points[:, 0] = points[::-1, 0]

            # Scale radial limits based on origin radius.
            points[:, 1] -= self._originLim.y0

            # Scale radial limits to match axes limits.
            rscale = 0.5 / points[1, 1]
            points[:, 1] *= rscale
            width = min(points[1, 1] - points[0, 1], 0.5)

            # Generate bounding box for wedge.
            wedge = mpatches.Wedge(self._center, points[1, 1],
                                   points[0, 0], points[1, 0],
                                   width=width)
            self.update_from_path(wedge.get_path())

            # Ensure equal aspect ratio.
            w, h = self._points[1] - self._points[0]
            deltah = max(w - h, 0) / 2
            deltaw = max(h - w, 0) / 2
            self._points += np.array([[-deltaw, -deltah], [deltaw, deltah]])

            self._invalid = 0

        return self._points


class PolarAxes(Axes):
    """
    A polar graph projection, where the input dimensions are *theta*, *r*.

    Theta starts pointing east and goes anti-clockwise.
    """
    name = 'polar'

    def __init__(self, *args,
                 theta_offset=0, theta_direction=1, rlabel_position=22.5,
                 **kwargs):
        # docstring inherited
        self._default_theta_offset = theta_offset
        self._default_theta_direction = theta_direction
        self._default_rlabel_position = np.deg2rad(rlabel_position)
        super().__init__(*args, **kwargs)
        self.use_sticky_edges = True
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.clear()

    def clear(self):
        # docstring inherited
        super().clear()

        self.title.set_y(1.05)

        start = self.spines.get('start', None)
        if start:
            start.set_visible(False)
        end = self.spines.get('end', None)
        if end:
            end.set_visible(False)
        self.set_xlim(0.0, 2 * np.pi)

        self.grid(mpl.rcParams['polaraxes.grid'])
        inner = self.spines.get('inner', None)
        if inner:
            inner.set_visible(False)

        self.set_rorigin(None)
        self.set_theta_offset(self._default_theta_offset)
        self.set_theta_direction(self._default_theta_direction)

    def _init_axis(self):
        # This is moved out of __init__ because non-separable axes don't use it
        self.xaxis = ThetaAxis(self, clear=False)
        self.yaxis = RadialAxis(self, clear=False)
        self.spines['polar'].register_axis(self.yaxis)
        inner_spine = self.spines.get('inner', None)
        if inner_spine is not None:
            # Subclasses may not have inner spine.
            inner_spine.register_axis(self.yaxis)

    def _set_lim_and_transforms(self):
        # A view limit where the minimum radius can be locked if the user
        # specifies an alternate origin.
        self._originViewLim = mtransforms.LockableBbox(self.viewLim)

        # Handle angular offset and direction.
        self._direction = mtransforms.Affine2D() \
            .scale(self._default_theta_direction, 1.0)
        self._theta_offset = mtransforms.Affine2D() \
            .translate(self._default_theta_offset, 0.0)
        self.transShift = self._direction + self._theta_offset
        # A view limit shifted to the correct location after accounting for
        # orientation and offset.
        self._realViewLim = mtransforms.TransformedBbox(self.viewLim,
                                                        self.transShift)

        # Transforms the x and y axis separately by a scale factor
        # It is assumed that this part will have non-linear components
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # Scale view limit into a bbox around the selected wedge. This may be
        # smaller than the usual unit axes rectangle if not plotting the full
        # circle.
        self.axesLim = _WedgeBbox((0.5, 0.5),
                                  self._realViewLim, self._originViewLim)

        # Scale the wedge to fill the axes.
        self.transWedge = mtransforms.BboxTransformFrom(self.axesLim)

        # Scale the axes to fill the figure.
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # A (possibly non-linear) projection on the (already scaled)
        # data.  This one is aware of rmin
        self.transProjection = self.PolarTransform(
            self,
            apply_theta_transforms=False,
            scale_transform=self.transScale
        )
        # Add dependency on rorigin.
        self.transProjection.set_children(self._originViewLim)

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transProjectionAffine = self.PolarAffine(self.transScale,
                                                      self._originViewLim)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        #
        # 1. Remove any radial axis scaling (e.g. log scaling)
        # 2. Shift data in the theta direction
        # 3. Project the data from polar to cartesian values
        #    (with the origin in the same place)
        # 4. Scale and translate the cartesian values to Axes coordinates
        #    (here the origin is moved to the lower left of the Axes)
        # 5. Move and scale to fill the Axes
        # 6. Convert from Axes coordinates to Figure coordinates
        self.transData = (
            self.transScale +
            self.transShift +
            self.transProjection +
            (
                self.transProjectionAffine +
                self.transWedge +
                self.transAxes
            )
        )

        # This is the transform for theta-axis ticks.  It is
        # equivalent to transData, except it always puts r == 0.0 and r == 1.0
        # at the edge of the axis circles.
        self._xaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.IdentityTransform(),
                mtransforms.BboxTransformTo(self.viewLim)) +
            self.transData)
        # The theta labels are flipped along the radius, so that text 1 is on
        # the outside by default. This should work the same as before.
        flipr_transform = mtransforms.Affine2D() \
            .translate(0.0, -0.5) \
            .scale(1.0, -1.0) \
            .translate(0.0, 0.5)
        self._xaxis_text_transform = flipr_transform + self._xaxis_transform

        # This is the transform for r-axis ticks.  It scales the theta
        # axis so the gridlines from 0.0 to 1.0, now go from thetamin to
        # thetamax.
        self._yaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.BboxTransformTo(self.viewLim),
                mtransforms.IdentityTransform()) +
            self.transData)
        # The r-axis labels are put at an angle and padded in the r-direction
        self._r_label_position = mtransforms.Affine2D() \
            .translate(self._default_rlabel_position, 0.0)
        self._yaxis_text_transform = mtransforms.TransformWrapper(
            self._r_label_position + self.transData)

    def get_xaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        if which in ('tick1', 'tick2'):
            return self._yaxis_text_transform
        elif which == 'grid':
            return self._yaxis_transform
        else:
            _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)

    def get_yaxis_text1_transform(self, pad):
        thetamin, thetamax = self._realViewLim.intervalx
        if _is_full_circle_rad(thetamin, thetamax):
            return self._yaxis_text_transform, 'bottom', 'left'
        elif self.get_theta_direction() > 0:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        else:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def get_yaxis_text2_transform(self, pad):
        if self.get_theta_direction() > 0:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        else:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def draw(self, renderer):
        self._unstale_viewLim()
        thetamin, thetamax = np.rad2deg(self._realViewLim.intervalx)
        if thetamin > thetamax:
            thetamin, thetamax = thetamax, thetamin
        rscale_tr = self.yaxis.get_transform()
        rmin, rmax = ((rscale_tr.transform(self._realViewLim.intervaly) -
                       rscale_tr.transform(self.get_rorigin())) *
                      self.get_rsign())
        if isinstance(self.patch, mpatches.Wedge):
            # Backwards-compatibility: Any subclassed Axes might override the
            # patch to not be the Wedge that PolarAxes uses.
            center = self.transWedge.transform((0.5, 0.5))
            self.patch.set_center(center)
            self.patch.set_theta1(thetamin)
            self.patch.set_theta2(thetamax)

            edge, _ = self.transWedge.transform((1, 0))
            radius = edge - center[0]
            width = min(radius * (rmax - rmin) / rmax, radius)
            self.patch.set_radius(radius)
            self.patch.set_width(width)

            inner_width = radius - width
            inner = self.spines.get('inner', None)
            if inner:
                inner.set_visible(inner_width != 0.0)

        visible = not _is_full_circle_deg(thetamin, thetamax)
        # For backwards compatibility, any subclassed Axes might override the
        # spines to not include start/end that PolarAxes uses.
        start = self.spines.get('start', None)
        end = self.spines.get('end', None)
        if start:
            start.set_visible(visible)
        if end:
            end.set_visible(visible)
        if visible:
            yaxis_text_transform = self._yaxis_transform
        else:
            yaxis_text_transform = self._r_label_position + self.transData
        if self._yaxis_text_transform != yaxis_text_transform:
            self._yaxis_text_transform.set(yaxis_text_transform)
            self.yaxis.reset_ticks()
            self.yaxis.set_clip_path(self.patch)

        super().draw(renderer)

    def _gen_axes_patch(self):
        return mpatches.Wedge((0.5, 0.5), 0.5, 0.0, 360.0)

    def _gen_axes_spines(self):
        spines = {
            'polar': Spine.arc_spine(self, 'top', (0.5, 0.5), 0.5, 0, 360),
            'start': Spine.linear_spine(self, 'left'),
            'end': Spine.linear_spine(self, 'right'),
            'inner': Spine.arc_spine(self, 'bottom', (0.5, 0.5), 0.0, 0, 360),
        }
        spines['polar'].set_transform(self.transWedge + self.transAxes)
        spines['inner'].set_transform(self.transWedge + self.transAxes)
        spines['start'].set_transform(self._yaxis_transform)
        spines['end'].set_transform(self._yaxis_transform)
        return spines

    def set_thetamax(self, thetamax):
        """Set the maximum theta limit in degrees."""
        self.viewLim.x1 = np.deg2rad(thetamax)

    def get_thetamax(self):
        """Return the maximum theta limit in degrees."""
        return np.rad2deg(self.viewLim.xmax)

    def set_thetamin(self, thetamin):
        """Set the minimum theta limit in degrees."""
        self.viewLim.x0 = np.deg2rad(thetamin)

    def get_thetamin(self):
        """Get the minimum theta limit in degrees."""
        return np.rad2deg(self.viewLim.xmin)

    def set_thetalim(self, *args, **kwargs):
        r"""
        Set the minimum and maximum theta values.

        Can take the following signatures:

        - ``set_thetalim(minval, maxval)``: Set the limits in radians.
        - ``set_thetalim(thetamin=minval, thetamax=maxval)``: Set the limits
          in degrees.

        where minval and maxval are the minimum and maximum limits. Values are
        wrapped in to the range :math:`[0, 2\pi]` (in radians), so for example
        it is possible to do ``set_thetalim(-np.pi / 2, np.pi / 2)`` to have
        an axis symmetric around 0. A ValueError is raised if the absolute
        angle difference is larger than a full circle.
        """
        orig_lim = self.get_xlim()  # in radians
        if 'thetamin' in kwargs:
            kwargs['xmin'] = np.deg2rad(kwargs.pop('thetamin'))
        if 'thetamax' in kwargs:
            kwargs['xmax'] = np.deg2rad(kwargs.pop('thetamax'))
        new_min, new_max = self.set_xlim(*args, **kwargs)
        # Parsing all permutations of *args, **kwargs is tricky; it is simpler
        # to let set_xlim() do it and then validate the limits.
        if abs(new_max - new_min) > 2 * np.pi:
            self.set_xlim(orig_lim)  # un-accept the change
            raise ValueError("The angle range must be less than a full circle")
        return tuple(np.rad2deg((new_min, new_max)))

    def set_theta_offset(self, offset):
        """
        Set the offset for the location of 0 in radians.
        """
        mtx = self._theta_offset.get_matrix()
        mtx[0, 2] = offset
        self._theta_offset.invalidate()

    def get_theta_offset(self):
        """
        Get the offset for the location of 0 in radians.
        """
        return self._theta_offset.get_matrix()[0, 2]

    def set_theta_zero_location(self, loc, offset=0.0):
        """
        Set the location of theta's zero.

        This simply calls `set_theta_offset` with the correct value in radians.

        Parameters
        ----------
        loc : str
            May be one of "N", "NW", "W", "SW", "S", "SE", "E", or "NE".
        offset : float, default: 0
            An offset in degrees to apply from the specified *loc*. **Note:**
            this offset is *always* applied counter-clockwise regardless of
            the direction setting.
        """
        mapping = {
            'N': np.pi * 0.5,
            'NW': np.pi * 0.75,
            'W': np.pi,
            'SW': np.pi * 1.25,
            'S': np.pi * 1.5,
            'SE': np.pi * 1.75,
            'E': 0,
            'NE': np.pi * 0.25}
        return self.set_theta_offset(mapping[loc] + np.deg2rad(offset))

    def set_theta_direction(self, direction):
        """
        Set the direction in which theta increases.

        clockwise, -1:
           Theta increases in the clockwise direction

        counterclockwise, anticlockwise, 1:
           Theta increases in the counterclockwise direction
        """
        mtx = self._direction.get_matrix()
        if direction in ('clockwise', -1):
            mtx[0, 0] = -1
        elif direction in ('counterclockwise', 'anticlockwise', 1):
            mtx[0, 0] = 1
        else:
            _api.check_in_list(
                [-1, 1, 'clockwise', 'counterclockwise', 'anticlockwise'],
                direction=direction)
        self._direction.invalidate()

    def get_theta_direction(self):
        """
        Get the direction in which theta increases.

        -1:
           Theta increases in the clockwise direction

        1:
           Theta increases in the counterclockwise direction
        """
        return self._direction.get_matrix()[0, 0]

    def set_rmax(self, rmax):
        """
        Set the outer radial limit.

        Parameters
        ----------
        rmax : float
        """
        self.viewLim.y1 = rmax

    def get_rmax(self):
        """
        Returns
        -------
        float
            Outer radial limit.
        """
        return self.viewLim.ymax

    def set_rmin(self, rmin):
        """
        Set the inner radial limit.

        Parameters
        ----------
        rmin : float
        """
        self.viewLim.y0 = rmin

    def get_rmin(self):
        """
        Returns
        -------
        float
            The inner radial limit.
        """
        return self.viewLim.ymin

    def set_rorigin(self, rorigin):
        """
        Update the radial origin.

        Parameters
        ----------
        rorigin : float
        """
        self._originViewLim.locked_y0 = rorigin

    def get_rorigin(self):
        """
        Returns
        -------
        float
        """
        return self._originViewLim.y0

    def get_rsign(self):
        return np.sign(self._originViewLim.y1 - self._originViewLim.y0)

    def set_rlim(self, bottom=None, top=None, *,
                 emit=True, auto=False, **kwargs):
        """
        Set the radial axis view limits.

        This function behaves like `.Axes.set_ylim`, but additionally supports
        *rmin* and *rmax* as aliases for *bottom* and *top*.

        See Also
        --------
        .Axes.set_ylim
        """
        if 'rmin' in kwargs:
            if bottom is None:
                bottom = kwargs.pop('rmin')
            else:
                raise ValueError('Cannot supply both positional "bottom"'
                                 'argument and kwarg "rmin"')
        if 'rmax' in kwargs:
            if top is None:
                top = kwargs.pop('rmax')
            else:
                raise ValueError('Cannot supply both positional "top"'
                                 'argument and kwarg "rmax"')
        return self.set_ylim(bottom=bottom, top=top, emit=emit, auto=auto,
                             **kwargs)

    def get_rlabel_position(self):
        """
        Returns
        -------
        float
            The theta position of the radius labels in degrees.
        """
        return np.rad2deg(self._r_label_position.get_matrix()[0, 2])

    def set_rlabel_position(self, value):
        """
        Update the theta position of the radius labels.

        Parameters
        ----------
        value : number
            The angular position of the radius labels in degrees.
        """
        self._r_label_position.clear().translate(np.deg2rad(value), 0.0)

    def set_yscale(self, *args, **kwargs):
        super().set_yscale(*args, **kwargs)
        self.yaxis.set_major_locator(
            self.RadialLocator(self.yaxis.get_major_locator(), self))

    def set_rscale(self, *args, **kwargs):
        return Axes.set_yscale(self, *args, **kwargs)

    def set_rticks(self, *args, **kwargs):
        return Axes.set_yticks(self, *args, **kwargs)

    def set_thetagrids(self, angles, labels=None, fmt=None, **kwargs):
        """
        Set the theta gridlines in a polar plot.

        Parameters
        ----------
        angles : tuple with floats, degrees
            The angles of the theta gridlines.

        labels : tuple with strings or None
            The labels to use at each theta gridline. The
            `.projections.polar.ThetaFormatter` will be used if None.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'. Note that the angle that is used is in
            radians.

        Returns
        -------
        lines : list of `.lines.Line2D`
            The theta gridlines.

        labels : list of `.text.Text`
            The tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `.Text` properties for the labels.

            .. warning::

                This only sets the properties of the current ticks.
                Ticks are not guaranteed to be persistent. Various operations
                can create, delete and modify the Tick instances. There is an
                imminent risk that these settings can get lost if you work on
                the figure further (including also panning/zooming on a
                displayed figure).

                Use `.set_tick_params` instead if possible.

        See Also
        --------
        .PolarAxes.set_rgrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """

        # Make sure we take into account unitized data
        angles = self.convert_yunits(angles)
        angles = np.deg2rad(angles)
        self.set_xticks(angles)
        if labels is not None:
            self.set_xticklabels(labels)
        elif fmt is not None:
            self.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        for t in self.xaxis.get_ticklabels():
            t._internal_update(kwargs)
        return self.xaxis.get_ticklines(), self.xaxis.get_ticklabels()

    def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kwargs):
        """
        Set the radial gridlines on a polar plot.

        Parameters
        ----------
        radii : tuple with floats
            The radii for the radial gridlines

        labels : tuple with strings or None
            The labels to use at each radial gridline. The
            `matplotlib.ticker.ScalarFormatter` will be used if None.

        angle : float
            The angular position of the radius labels in degrees.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'.

        Returns
        -------
        lines : list of `.lines.Line2D`
            The radial gridlines.

        labels : list of `.text.Text`
            The tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `.Text` properties for the labels.

            .. warning::

                This only sets the properties of the current ticks.
                Ticks are not guaranteed to be persistent. Various operations
                can create, delete and modify the Tick instances. There is an
                imminent risk that these settings can get lost if you work on
                the figure further (including also panning/zooming on a
                displayed figure).

                Use `.set_tick_params` instead if possible.

        See Also
        --------
        .PolarAxes.set_thetagrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """
        # Make sure we take into account unitized data
        radii = self.convert_xunits(radii)
        radii = np.asarray(radii)

        self.set_yticks(radii)
        if labels is not None:
            self.set_yticklabels(labels)
        elif fmt is not None:
            self.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        if angle is None:
            angle = self.get_rlabel_position()
        self.set_rlabel_position(angle)
        for t in self.yaxis.get_ticklabels():
            t._internal_update(kwargs)
        return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

    def format_coord(self, theta, r):
        # docstring inherited
        screen_xy = self.transData.transform((theta, r))
        screen_xys = screen_xy + np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape((2, -1)).T
        ts, rs = self.transData.inverted().transform(screen_xys).T
        delta_t = abs((ts - theta + np.pi) % (2 * np.pi) - np.pi).max()
        delta_t_halfturns = delta_t / np.pi
        delta_t_degrees = delta_t_halfturns * 180
        delta_r = abs(rs - r).max()
        if theta < 0:
            theta += 2 * np.pi
        theta_halfturns = theta / np.pi
        theta_degrees = theta_halfturns * 180

        # See ScalarFormatter.format_data_short.  For r, use #g-formatting
        # (as for linear axes), but for theta, use f-formatting as scientific
        # notation doesn't make sense and the trailing dot is ugly.
        def format_sig(value, delta, opt, fmt):
            # For "f", only count digits after decimal point.
            prec = (max(0, -math.floor(math.log10(delta))) if fmt == "f" else
                    cbook._g_sig_digits(value, delta))
            return f"{value:-{opt}.{prec}{fmt}}"

        # In case fmt_xdata was not specified, resort to default

        if self.fmt_ydata is None:
            r_label = format_sig(r, delta_r, "#", "g")
        else:
            r_label = self.format_ydata(r)

        if self.fmt_xdata is None:
            return ('\N{GREEK SMALL LETTER THETA}={}\N{GREEK SMALL LETTER PI} '
                    '({}\N{DEGREE SIGN}), r={}').format(
                    format_sig(theta_halfturns, delta_t_halfturns, "", "f"),
                    format_sig(theta_degrees, delta_t_degrees, "", "f"),
                    r_label
                )
        else:
            return '\N{GREEK SMALL LETTER THETA}={}, r={}'.format(
                        self.format_xdata(theta),
                        r_label
                        )

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.  For a polar plot,
        this should always be 1.0
        """
        return 1.0

    # # # Interactive panning

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        A polar Axes does not support zoom boxes.
        """
        return False

    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        For a polar Axes, this is slightly misleading. Both panning and
        zooming are performed by the same button. Panning is performed
        in azimuth while zooming is done along the radial.
        """
        return True

    def start_pan(self, x, y, button):
        angle = np.deg2rad(self.get_rlabel_position())
        mode = ''
        if button == 1:
            epsilon = np.pi / 45.0
            t, r = self.transData.inverted().transform((x, y))
            if angle - epsilon <= t <= angle + epsilon:
                mode = 'drag_r_labels'
        elif button == 3:
            mode = 'zoom'

        self._pan_start = types.SimpleNamespace(
            rmax=self.get_rmax(),
            trans=self.transData.frozen(),
            trans_inverse=self.transData.inverted().frozen(),
            r_label_angle=self.get_rlabel_position(),
            x=x,
            y=y,
            mode=mode)

    def end_pan(self):
        del self._pan_start

    def drag_pan(self, button, key, x, y):
        p = self._pan_start

        if p.mode == 'drag_r_labels':
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # Deal with theta
            dt = np.rad2deg(startt - t)
            self.set_rlabel_position(p.r_label_angle - dt)

            trans, vert1, horiz1 = self.get_yaxis_text1_transform(0.0)
            trans, vert2, horiz2 = self.get_yaxis_text2_transform(0.0)
            for t in self.yaxis.majorTicks + self.yaxis.minorTicks:
                t.label1.set_va(vert1)
                t.label1.set_ha(horiz1)
                t.label2.set_va(vert2)
                t.label2.set_ha(horiz2)

        elif p.mode == 'zoom':
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # Deal with r
            scale = r / startr
            self.set_rmax(p.rmax / scale)


# To keep things all self-contained, we can put aliases to the Polar classes
# defined above. This isn't strictly necessary, but it makes some of the
# code more readable, and provides a backwards compatible Polar API. In
# particular, this is used by the :doc:`/gallery/specialty_plots/radar_chart`
# example to override PolarTransform on a PolarAxes subclass, so make sure that
# that example is unaffected before changing this.
PolarAxes.PolarTransform = PolarTransform
PolarAxes.PolarAffine = PolarAffine
PolarAxes.InvertedPolarTransform = InvertedPolarTransform
PolarAxes.ThetaFormatter = ThetaFormatter
PolarAxes.RadialLocator = RadialLocator
PolarAxes.ThetaLocator = ThetaLocator
<|endoftext|>import numpy as np

import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
    Formatter, NullLocator, FixedLocator, NullFormatter)
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform


class GeoAxes(Axes):
    """An abstract base class for geographic projections."""

    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
            return f"{degrees:0.0f}\N{DEGREE SIGN}"

    RESOLUTION = 75

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self, clear=False)
        self.yaxis = maxis.YAxis(self, clear=False)
        self.spines['geo'].register_axis(self.yaxis)

    def clear(self):
        # docstring inherited
        super().clear()

        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
        self.yaxis.set_tick_params(label1On=True)
        # Why do we need to turn on yaxis tick labels, but
        # xaxis tick labels are already on?

        self.grid(mpl.rcParams['axes.grid'])

        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _set_lim_and_transforms(self):
        # A (possibly non-linear) projection on the (already scaled) data
        self.transProjection = self._get_core_transform(self.RESOLUTION)

        self.transAffine = self._get_affine_transform()

        self.transAxes = BboxTransformTo(self.bbox)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        # This is the transform for longitude ticks.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1, self._longitude_cap * 2) \
            .translate(0, -self._longitude_cap)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1, 0) + \
            self.transData + \
            Affine2D().translate(0, 4)
        self._xaxis_text2_transform = \
            Affine2D().scale(1, 0) + \
            self.transData + \
            Affine2D().translate(0, -4)

        # This is the transform for latitude ticks.
        yaxis_stretch = Affine2D().scale(np.pi * 2, 1).translate(-np.pi, 0)
        yaxis_space = Affine2D().scale(1, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space +
             self.transAffine +
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8, 0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8, 0)

    def _get_affine_transform(self):
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform((np.pi, 0))
        _, yscale = transform.transform((0, np.pi/2))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

    def get_xaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        return {'geo': mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    set_xscale = set_yscale

    def set_xlim(self, *args, **kwargs):
        """Not supported. Please consider using Cartopy."""
        raise TypeError("Changing axes limits of a geographic projection is "
                        "not supported.  Please consider using Cartopy.")

    set_ylim = set_xlim
    set_xbound = set_xlim
    set_ybound = set_ylim

    def invert_xaxis(self):
        """Not supported. Please consider using Cartopy."""
        raise TypeError("Changing axes limits of a geographic projection is "
                        "not supported.  Please consider using Cartopy.")

    invert_yaxis = invert_xaxis

    def format_coord(self, lon, lat):
        """Return a format string formatting the coordinate."""
        lon, lat = np.rad2deg([lon, lat])
        ns = 'N' if lat >= 0.0 else 'S'
        ew = 'E' if lon >= 0.0 else 'W'
        return ('%f\N{DEGREE SIGN}%s, %f\N{DEGREE SIGN}%s'
                % (abs(lat), ns, abs(lon), ew))

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        # Skip -180 and 180, which are the fixed limits.
        grid = np.arange(-180 + degrees, 180, degrees)
        self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each latitude grid.
        """
        # Skip -90 and 90, which are the fixed limits.
        grid = np.arange(-90 + degrees, 90, degrees)
        self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        self._longitude_cap = np.deg2rad(degrees)
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)

    def get_data_ratio(self):
        """Return the aspect ratio of the data itself."""
        return 1.0

    ### Interactive panning

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        This Axes object does not support interactive zoom box.
        """
        return False

    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        This Axes object does not support interactive pan/zoom.
        """
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass


class _GeoTransform(Transform):
    # Factoring out some common functionality.
    input_dims = output_dims = 2

    def __init__(self, resolution):
        """
        Create a new geographical transform.

        Resolution is the number of steps to interpolate between each input
        line segment to approximate its path in curved space.
        """
        super().__init__()
        self._resolution = resolution

    def __str__(self):
        return f"{type(self).__name__}({self._resolution})"

    def transform_path_non_affine(self, path):
        # docstring inherited
        ipath = path.interpolated(self._resolution)
        return Path(self.transform(ipath.vertices), ipath.codes)


class AitoffAxes(GeoAxes):
    name = 'aitoff'

    class AitoffTransform(_GeoTransform):
        """The base Aitoff transform."""

        def transform_non_affine(self, values):
            # docstring inherited
            longitude, latitude = values.T

            # Pre-compute some values
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)

            alpha = np.arccos(cos_latitude * np.cos(half_long))
            sinc_alpha = np.sinc(alpha / np.pi)  # np.sinc is sin(pi*x)/(pi*x).

            x = (cos_latitude * np.sin(half_long)) / sinc_alpha
            y = np.sin(latitude) / sinc_alpha
            return np.column_stack([x, y])

        def inverted(self):
            # docstring inherited
            return AitoffAxes.InvertedAitoffTransform(self._resolution)

    class InvertedAitoffTransform(_GeoTransform):

        def transform_non_affine(self, values):
            # docstring inherited
            # MGDTODO: Math is hard ;(
            return np.full_like(values, np.nan)

        def inverted(self):
            # docstring inherited
            return AitoffAxes.AitoffTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.AitoffTransform(resolution)


class HammerAxes(GeoAxes):
    name = 'hammer'

    class HammerTransform(_GeoTransform):
        """The base Hammer transform."""

        def transform_non_affine(self, values):
            # docstring inherited
            longitude, latitude = values.T
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)
            sqrt2 = np.sqrt(2.0)
            alpha = np.sqrt(1.0 + cos_latitude * np.cos(half_long))
            x = (2.0 * sqrt2) * (cos_latitude * np.sin(half_long)) / alpha
            y = (sqrt2 * np.sin(latitude)) / alpha
            return np.column_stack([x, y])

        def inverted(self):
            # docstring inherited
            return HammerAxes.InvertedHammerTransform(self._resolution)

    class InvertedHammerTransform(_GeoTransform):

        def transform_non_affine(self, values):
            # docstring inherited
            x, y = values.T
            z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
            longitude = 2 * np.arctan((z * x) / (2 * (2 * z ** 2 - 1)))
            latitude = np.arcsin(y*z)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            # docstring inherited
            return HammerAxes.HammerTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.HammerTransform(resolution)


class MollweideAxes(GeoAxes):
    name = 'mollweide'

    class MollweideTransform(_GeoTransform):
        """The base Mollweide transform."""

        def transform_non_affine(self, values):
            # docstring inherited
            def d(theta):
                delta = (-(theta + np.sin(theta) - pi_sin_l)
                         / (1 + np.cos(theta)))
                return delta, np.abs(delta) > 0.001

            longitude, latitude = values.T

            clat = np.pi/2 - np.abs(latitude)
            ihigh = clat < 0.087  # within 5 degrees of the poles
            ilow = ~ihigh
            aux = np.empty(latitude.shape, dtype=float)

            if ilow.any():  # Newton-Raphson iteration
                pi_sin_l = np.pi * np.sin(latitude[ilow])
                theta = 2.0 * latitude[ilow]
                delta, large_delta = d(theta)
                while np.any(large_delta):
                    theta[large_delta] += delta[large_delta]
                    delta, large_delta = d(theta)
                aux[ilow] = theta / 2

            if ihigh.any():  # Taylor series-based approx. solution
                e = clat[ihigh]
                d = 0.5 * (3 * np.pi * e**2) ** (1.0/3)
                aux[ihigh] = (np.pi/2 - d) * np.sign(latitude[ihigh])

            xy = np.empty(values.shape, dtype=float)
            xy[:, 0] = (2.0 * np.sqrt(2.0) / np.pi) * longitude * np.cos(aux)
            xy[:, 1] = np.sqrt(2.0) * np.sin(aux)

            return xy

        def inverted(self):
            # docstring inherited
            return MollweideAxes.InvertedMollweideTransform(self._resolution)

    class InvertedMollweideTransform(_GeoTransform):

        def transform_non_affine(self, values):
            # docstring inherited
            x, y = values.T
            # from Equations (7, 8) of
            # https://mathworld.wolfram.com/MollweideProjection.html
            theta = np.arcsin(y / np.sqrt(2))
            longitude = (np.pi / (2 * np.sqrt(2))) * x / np.cos(theta)
            latitude = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            # docstring inherited
            return MollweideAxes.MollweideTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.MollweideTransform(resolution)


class LambertAxes(GeoAxes):
    name = 'lambert'

    class LambertTransform(_GeoTransform):
        """The base Lambert transform."""

        def __init__(self, center_longitude, center_latitude, resolution):
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
            _GeoTransform.__init__(self, resolution)
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        def transform_non_affine(self, values):
            # docstring inherited
            longitude, latitude = values.T
            clong = self._center_longitude
            clat = self._center_latitude
            cos_lat = np.cos(latitude)
            sin_lat = np.sin(latitude)
            diff_long = longitude - clong
            cos_diff_long = np.cos(diff_long)

            inner_k = np.maximum(  # Prevent divide-by-zero problems
                1 + np.sin(clat)*sin_lat + np.cos(clat)*cos_lat*cos_diff_long,
                1e-15)
            k = np.sqrt(2 / inner_k)
            x = k * cos_lat*np.sin(diff_long)
            y = k * (np.cos(clat)*sin_lat - np.sin(clat)*cos_lat*cos_diff_long)

            return np.column_stack([x, y])

        def inverted(self):
            # docstring inherited
            return LambertAxes.InvertedLambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)

    class InvertedLambertTransform(_GeoTransform):

        def __init__(self, center_longitude, center_latitude, resolution):
            _GeoTransform.__init__(self, resolution)
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        def transform_non_affine(self, values):
            # docstring inherited
            x, y = values.T
            clong = self._center_longitude
            clat = self._center_latitude
            p = np.maximum(np.hypot(x, y), 1e-9)
            c = 2 * np.arcsin(0.5 * p)
            sin_c = np.sin(c)
            cos_c = np.cos(c)

            latitude = np.arcsin(cos_c*np.sin(clat) +
                                 ((y*sin_c*np.cos(clat)) / p))
            longitude = clong + np.arctan(
                (x*sin_c) / (p*np.cos(clat)*cos_c - y*np.sin(clat)*sin_c))

            return np.column_stack([longitude, latitude])

        def inverted(self):
            # docstring inherited
            return LambertAxes.LambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)

    def __init__(self, *args, center_longitude=0, center_latitude=0, **kwargs):
        self._longitude_cap = np.pi / 2
        self._center_longitude = center_longitude
        self._center_latitude = center_latitude
        super().__init__(*args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.clear()

    def clear(self):
        # docstring inherited
        super().clear()
        self.yaxis.set_major_formatter(NullFormatter())

    def _get_core_transform(self, resolution):
        return self.LambertTransform(
            self._center_longitude,
            self._center_latitude,
            resolution)

    def _get_affine_transform(self):
        return Affine2D() \
            .scale(0.25) \
            .translate(0.5, 0.5)
<|endoftext|>"""
Non-separable transforms that map from data space to screen space.

Projections are defined as `~.axes.Axes` subclasses.  They include the
following elements:

- A transformation from data coordinates into display coordinates.

- An inverse of that transformation.  This is used, for example, to convert
  mouse positions from screen space back into data space.

- Transformations for the gridlines, ticks and ticklabels.  Custom projections
  will often need to place these elements in special locations, and Matplotlib
  has a facility to help with doing so.

- Setting up default values (overriding `~.axes.Axes.cla`), since the defaults
  for a rectilinear Axes may not be appropriate.

- Defining the shape of the Axes, for example, an elliptical Axes, that will be
  used to draw the background of the plot and for clipping any data elements.

- Defining custom locators and formatters for the projection.  For example, in
  a geographic projection, it may be more convenient to display the grid in
  degrees, even if the data is in radians.

- Set up interactive panning and zooming.  This is left as an "advanced"
  feature left to the reader, but there is an example of this for polar plots
  in `matplotlib.projections.polar`.

- Any additional methods for additional convenience or features.

Once the projection Axes is defined, it can be used in one of two ways:

- By defining the class attribute ``name``, the projection Axes can be
  registered with `matplotlib.projections.register_projection` and subsequently
  simply invoked by name::

      fig.add_subplot(projection="my_proj_name")

- For more complex, parameterisable projections, a generic "projection" object
  may be defined which includes the method ``_as_mpl_axes``. ``_as_mpl_axes``
  should take no arguments and return the projection's Axes subclass and a
  dictionary of additional arguments to pass to the subclass' ``__init__``
  method.  Subsequently a parameterised projection can be initialised with::

      fig.add_subplot(projection=MyProjection(param1=param1_value))

  where MyProjection is an object which implements a ``_as_mpl_axes`` method.

A full-fledged and heavily annotated example is in
:doc:`/gallery/misc/custom_projection`.  The polar plot functionality in
`matplotlib.projections.polar` may also be of interest.
"""

from .. import axes, _docstring
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes

try:
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    import warnings
    warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "
                  "Matplotlib being installed (e.g. as a system package and as a pip "
                  "package). As a result, the 3D projection is not available.")
    Axes3D = None


class ProjectionRegistry:
    """A mapping of registered projection names to projection classes."""

    def __init__(self):
        self._all_projection_types = {}

    def register(self, *projections):
        """Register a new set of projections."""
        for projection in projections:
            name = projection.name
            self._all_projection_types[name] = projection

    def get_projection_class(self, name):
        """Get a projection class from its *name*."""
        return self._all_projection_types[name]

    def get_projection_names(self):
        """Return the names of all projections currently registered."""
        return sorted(self._all_projection_types)


projection_registry = ProjectionRegistry()
projection_registry.register(
    axes.Axes,
    PolarAxes,
    AitoffAxes,
    HammerAxes,
    LambertAxes,
    MollweideAxes,
)
if Axes3D is not None:
    projection_registry.register(Axes3D)
else:
    # remove from namespace if not importable
    del Axes3D


def register_projection(cls):
    projection_registry.register(cls)


def get_projection_class(projection=None):
    """
    Get a projection class from its name.

    If *projection* is None, a standard rectilinear projection is returned.
    """
    if projection is None:
        projection = 'rectilinear'

    try:
        return projection_registry.get_projection_class(projection)
    except KeyError as err:
        raise ValueError("Unknown projection %r" % projection) from err


get_projection_names = projection_registry.get_projection_names
_docstring.interpd.register(projection_names=get_projection_names())
<|endoftext|>"""
Custom roles for the Matplotlib documentation.

.. warning::

    These roles are considered semi-public. They are only intended to be used in
    the Matplotlib documentation.

However, it can happen that downstream packages end up pulling these roles into
their documentation, which will result in documentation build errors. The following
describes the exact mechanism and how to fix the errors.

There are two ways, Matplotlib docstrings can end up in downstream documentation.
You have to subclass a Matplotlib class and either use the ``:inherited-members:``
option in your autodoc configuration, or you have to override a method without
specifying a new docstring; the new method will inherit the original docstring and
still render in your autodoc. If the docstring contains one of the custom sphinx
roles, you'll see one of the following error messages:

.. code-block:: none

    Unknown interpreted text role "mpltype".
    Unknown interpreted text role "rc".

To fix this, you can add this module as extension to your sphinx :file:`conf.py`::

    extensions = [
        'matplotlib.sphinxext.roles',
        # Other extensions.
    ]

.. warning::

    Direct use of these roles in other packages is not officially supported. We
    reserve the right to modify or remove these roles without prior notification.
"""

from urllib.parse import urlsplit, urlunsplit

from docutils import nodes

import matplotlib
from matplotlib import rcParamsDefault


class _QueryReference(nodes.Inline, nodes.TextElement):
    """
    Wraps a reference or pending reference to add a query string.

    The query string is generated from the attributes added to this node.

    Also equivalent to a `~docutils.nodes.literal` node.
    """

    def to_query_string(self):
        """Generate query string from node attributes."""
        return '&'.join(f'{name}={value}' for name, value in self.attlist())


def _visit_query_reference_node(self, node):
    """
    Resolve *node* into query strings on its ``reference`` children.

    Then act as if this is a `~docutils.nodes.literal`.
    """
    query = node.to_query_string()
    for refnode in node.findall(nodes.reference):
        uri = urlsplit(refnode['refuri'])._replace(query=query)
        refnode['refuri'] = urlunsplit(uri)

    self.visit_literal(node)


def _depart_query_reference_node(self, node):
    """
    Act as if this is a `~docutils.nodes.literal`.
    """
    self.depart_literal(node)


def _rcparam_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Sphinx role ``:rc:`` to highlight and link ``rcParams`` entries.

    Usage: Give the desired ``rcParams`` key as parameter.

    :code:`:rc:`figure.dpi`` will render as: :rc:`figure.dpi`
    """
    # Generate a pending cross-reference so that Sphinx will ensure this link
    # isn't broken at some point in the future.
    title = f'rcParams["{text}"]'
    target = 'matplotlibrc-sample'
    ref_nodes, messages = inliner.interpreted(title, f'{title} <{target}>',
                                              'ref', lineno)

    qr = _QueryReference(rawtext, highlight=text)
    qr += ref_nodes
    node_list = [qr]

    # The default backend would be printed as "agg", but that's not correct (as
    # the default is actually determined by fallback).
    if text in rcParamsDefault and text != "backend":
        node_list.extend([
            nodes.Text(' (default: '),
            nodes.literal('', repr(rcParamsDefault[text])),
            nodes.Text(')'),
            ])

    return node_list, messages


def _mpltype_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Sphinx role ``:mpltype:`` for custom matplotlib types.

    In Matplotlib, there are a number of type-like concepts that do not have a
    direct type representation; example: color. This role allows to properly
    highlight them in the docs and link to their definition.

    Currently supported values:

    - :code:`:mpltype:`color`` will render as: :mpltype:`color`

    """
    mpltype = text
    type_to_link_target = {
        'color': 'colors_def',
    }
    if mpltype not in type_to_link_target:
        raise ValueError(f"Unknown mpltype: {mpltype!r}")

    node_list, messages = inliner.interpreted(
        mpltype, f'{mpltype} <{type_to_link_target[mpltype]}>', 'ref', lineno)
    return node_list, messages


def setup(app):
    app.add_role("rc", _rcparam_role)
    app.add_role("mpltype", _mpltype_role)
    app.add_node(
        _QueryReference,
        html=(_visit_query_reference_node, _depart_query_reference_node),
        latex=(_visit_query_reference_node, _depart_query_reference_node),
        text=(_visit_query_reference_node, _depart_query_reference_node),
    )
    return {"version": matplotlib.__version__,
            "parallel_read_safe": True, "parallel_write_safe": True}
<|endoftext|>r"""
A role and directive to display mathtext in Sphinx
==================================================

The ``mathmpl`` Sphinx extension creates a mathtext image in Matplotlib and
shows it in html output. Thus, it is a true and faithful representation of what
you will see if you pass a given LaTeX string to Matplotlib (see
:ref:`mathtext`).

.. warning::
    In most cases, you will likely want to use one of `Sphinx's builtin Math
    extensions
    <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`__
    instead of this one. The builtin Sphinx math directive uses MathJax to
    render mathematical expressions, and addresses accessibility concerns that
    ``mathmpl`` doesn't address.

Mathtext may be included in two ways:

1. Inline, using the role::

     This text uses inline math: :mathmpl:`\alpha > \beta`.

   which produces:

     This text uses inline math: :mathmpl:`\alpha > \beta`.

2. Standalone, using the directive::

     Here is some standalone math:

     .. mathmpl::

         \alpha > \beta

   which produces:

     Here is some standalone math:

     .. mathmpl::

         \alpha > \beta

Options
-------

The ``mathmpl`` role and directive both support the following options:

fontset : str, default: 'cm'
    The font set to use when displaying math. See :rc:`mathtext.fontset`.

fontsize : float
    The font size, in points. Defaults to the value from the extension
    configuration option defined below.

Configuration options
---------------------

The mathtext extension has the following configuration options:

mathmpl_fontsize : float, default: 10.0
    Default font size, in points.

mathmpl_srcset : list of str, default: []
    Additional image sizes to generate when embedding in HTML, to support
    `responsive resolution images
    <https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images>`__.
    The list should contain additional x-descriptors (``'1.5x'``, ``'2x'``,
    etc.) to generate (1x is the default and always included.)

"""

import hashlib
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError

import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None


# Define LaTeX math node:
class latex_math(nodes.General, nodes.Element):
    pass


def fontset_choice(arg):
    return directives.choice(arg, mathtext.MathTextParser._font_type_mapping)


def math_role(role, rawtext, text, lineno, inliner,
              options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    node['fontsize'] = options.get('fontsize',
                                   setup.app.config.mathmpl_fontsize)
    return [node], []
math_role.options = {'fontset': fontset_choice,
                     'fontsize': validate_float_or_None}


class MathDirective(Directive):
    """
    The ``.. mathmpl::`` directive, as documented in the module's docstring.
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {'fontset': fontset_choice,
                   'fontsize': validate_float_or_None}

    def run(self):
        latex = ''.join(self.content)
        node = latex_math(self.block_text)
        node['latex'] = latex
        node['fontset'] = self.options.get('fontset', 'cm')
        node['fontsize'] = self.options.get('fontsize',
                                            setup.app.config.mathmpl_fontsize)
        return [node]


# This uses mathtext to render the expression
def latex2png(latex, filename, fontset='cm', fontsize=10, dpi=100):
    with mpl.rc_context({'mathtext.fontset': fontset, 'font.size': fontsize}):
        try:
            depth = mathtext.math_to_image(
                f"${latex}$", filename, dpi=dpi, format="png")
        except Exception:
            _api.warn_external(f"Could not render math expression {latex}")
            depth = 0
    return depth


# LaTeX to HTML translation stuff:
def latex2html(node, source):
    inline = isinstance(node.parent, nodes.TextElement)
    latex = node['latex']
    fontset = node['fontset']
    fontsize = node['fontsize']
    name = 'math-{}'.format(
        hashlib.sha256(
            f'{latex}{fontset}{fontsize}'.encode(),
            usedforsecurity=False,
        ).hexdigest()[-10:])

    destdir = Path(setup.app.builder.outdir, '_images', 'mathmpl')
    destdir.mkdir(parents=True, exist_ok=True)

    dest = destdir / f'{name}.png'
    depth = latex2png(latex, dest, fontset, fontsize=fontsize)

    srcset = []
    for size in setup.app.config.mathmpl_srcset:
        filename = f'{name}-{size.replace(".", "_")}.png'
        latex2png(latex, destdir / filename, fontset, fontsize=fontsize,
                  dpi=100 * float(size[:-1]))
        srcset.append(
            f'{setup.app.builder.imgpath}/mathmpl/{filename} {size}')
    if srcset:
        srcset = (f'srcset="{setup.app.builder.imgpath}/mathmpl/{name}.png, ' +
                  ', '.join(srcset) + '" ')

    if inline:
        cls = ''
    else:
        cls = 'class="center" '
    if inline and depth != 0:
        style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
    else:
        style = ''

    return (f'<img src="{setup.app.builder.imgpath}/mathmpl/{name}.png"'
            f' {srcset}{cls}{style}/>')


def _config_inited(app, config):
    # Check for srcset hidpi images
    for i, size in enumerate(app.config.mathmpl_srcset):
        if size[-1] == 'x':  # "2x" = "2.0"
            try:
                float(size[:-1])
            except ValueError:
                raise ConfigError(
                    f'Invalid value for mathmpl_srcset parameter: {size!r}. '
                    'Must be a list of strings with the multiplicative '
                    'factor followed by an "x".  e.g. ["2.0x", "1.5x"]')
        else:
            raise ConfigError(
                f'Invalid value for mathmpl_srcset parameter: {size!r}. '
                'Must be a list of strings with the multiplicative '
                'factor followed by an "x".  e.g. ["2.0x", "1.5x"]')


def setup(app):
    setup.app = app
    app.add_config_value('mathmpl_fontsize', 10.0, True)
    app.add_config_value('mathmpl_srcset', [], True)
    try:
        app.connect('config-inited', _config_inited)  # Sphinx 1.8+
    except ExtensionError:
        app.connect('env-updated', lambda app, env: _config_inited(app, None))

    # Add visit/depart methods to HTML-Translator:
    def visit_latex_math_html(self, node):
        source = self.document.attributes['source']
        self.body.append(latex2html(node, source))

    def depart_latex_math_html(self, node):
        pass

    # Add visit/depart methods to LaTeX-Translator:
    def visit_latex_math_latex(self, node):
        inline = isinstance(node.parent, nodes.TextElement)
        if inline:
            self.body.append('$%s$' % node['latex'])
        else:
            self.body.extend(['\\begin{equation}',
                              node['latex'],
                              '\\end{equation}'])

    def depart_latex_math_latex(self, node):
        pass

    app.add_node(latex_math,
                 html=(visit_latex_math_html, depart_latex_math_html),
                 latex=(visit_latex_math_latex, depart_latex_math_latex))
    app.add_role('mathmpl', math_role)
    app.add_directive('mathmpl', MathDirective)
    if sphinx.version_info < (1, 8):
        app.add_role('math', math_role)
        app.add_directive('math', MathDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
<|endoftext|>"""
A directive for including a Matplotlib plot in a Sphinx document
================================================================

This is a Sphinx extension providing a reStructuredText directive
``.. plot::`` for including a plot in a Sphinx document.

In HTML output, ``.. plot::`` will include a .png file with a link
to a high-res .png and .pdf.  In LaTeX output, it will include a .pdf.

The plot content may be defined in one of three ways:

1. **A path to a source file** as the argument to the directive::

     .. plot:: path/to/plot.py

   When a path to a source file is given, the content of the
   directive may optionally contain a caption for the plot::

     .. plot:: path/to/plot.py

        The plot caption.

   Additionally, one may specify the name of a function to call (with
   no arguments) immediately after importing the module::

     .. plot:: path/to/plot.py plot_function1

2. Included as **inline content** to the directive::

     .. plot::

        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title("A plotting exammple")

3. Using **doctest** syntax::

     .. plot::

        A plotting example:
        >>> import matplotlib.pyplot as plt
        >>> plt.plot([1, 2, 3], [4, 5, 6])

Options
-------

The ``.. plot::`` directive supports the following options:

``:format:`` : {'python', 'doctest'}
    The format of the input.  If unset, the format is auto-detected.

``:include-source:`` : bool
    Whether to display the source code. The default can be changed using
    the ``plot_include_source`` variable in :file:`conf.py` (which itself
    defaults to False).

``:show-source-link:`` : bool
    Whether to show a link to the source in HTML. The default can be
    changed using the ``plot_html_show_source_link`` variable in
    :file:`conf.py` (which itself defaults to True).

``:context:`` : bool or str
    If provided, the code will be run in the context of all previous plot
    directives for which the ``:context:`` option was specified.  This only
    applies to inline code plot directives, not those run from files. If
    the ``:context: reset`` option is specified, the context is reset
    for this and future plots, and previous figures are closed prior to
    running the code. ``:context: close-figs`` keeps the context but closes
    previous figures before running the code.

``:nofigs:`` : bool
    If specified, the code block will be run, but no figures will be
    inserted.  This is usually useful with the ``:context:`` option.

``:caption:`` : str
    If specified, the option's argument will be used as a caption for the
    figure. This overwrites the caption given in the content, when the plot
    is generated from a file.

Additionally, this directive supports all the options of the `image directive
<https://docutils.sourceforge.io/docs/ref/rst/directives.html#image>`_,
except for ``:target:`` (since plot will add its own target).  These include
``:alt:``, ``:height:``, ``:width:``, ``:scale:``, ``:align:`` and ``:class:``.

Configuration options
---------------------

The plot directive has the following configuration options:

plot_include_source
    Default value for the include-source option (default: False).

plot_html_show_source_link
    Whether to show a link to the source in HTML (default: True).

plot_pre_code
    Code that should be executed before each plot. If None (the default),
    it will default to a string containing::

        import numpy as np
        from matplotlib import pyplot as plt

plot_basedir
    Base directory, to which ``plot::`` file names are relative to.
    If None or empty (the default), file names are relative to the
    directory where the file containing the directive is.

plot_formats
    File formats to generate (default: ['png', 'hires.png', 'pdf']).
    List of tuples or strings::

        [(suffix, dpi), suffix, ...]

    that determine the file format and the DPI. For entries whose
    DPI was omitted, sensible defaults are chosen. When passing from
    the command line through sphinx_build the list should be passed as
    suffix:dpi,suffix:dpi, ...

plot_html_show_formats
    Whether to show links to the files in HTML (default: True).

plot_rcparams
    A dictionary containing any non-standard rcParams that should
    be applied before each plot (default: {}).

plot_apply_rcparams
    By default, rcParams are applied when ``:context:`` option is not used
    in a plot directive.  If set, this configuration option overrides this
    behavior and applies rcParams before each plot.

plot_working_directory
    By default, the working directory will be changed to the directory of
    the example, so the code can get at its data files, if any.  Also its
    path will be added to `sys.path` so it can import any helper modules
    sitting beside it.  This configuration option can be used to specify
    a central directory (also added to `sys.path`) where data files and
    helper modules for all code are located.

plot_template
    Provide a customized template for preparing restructured text.

plot_srcset
    Allow the srcset image option for responsive image resolutions. List of
    strings with the multiplicative factors followed by an "x".
    e.g. ["2.0x", "1.5x"].  "2.0x" will create a png with the default "png"
    resolution from plot_formats, multiplied by 2. If plot_srcset is
    specified, the plot directive uses the
    :doc:`/api/sphinxext_figmpl_directive_api` (instead of the usual figure
    directive) in the intermediary rst file that is generated.
    The plot_srcset option is incompatible with *singlehtml* builds, and an
    error will be raised.

Notes on how it works
---------------------

The plot directive runs the code it is given, either in the source file or the
code under the directive. The figure created (if any) is saved in the sphinx
build directory under a subdirectory named ``plot_directive``.  It then creates
an intermediate rst file that calls a ``.. figure:`` directive (or
``.. figmpl::`` directive if ``plot_srcset`` is being used) and has links to
the ``*.png`` files in the ``plot_directive`` directory.  These translations can
be customized by changing the *plot_template*.  See the source of
:doc:`/api/sphinxext_plot_directive_api` for the templates defined in *TEMPLATE*
and *TEMPLATE_SRCSET*.
"""

import contextlib
import doctest
from io import StringIO
import itertools
import os
from os.path import relpath
from pathlib import Path
import re
import shutil
import sys
import textwrap
import traceback

from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.

from sphinx.errors import ExtensionError

import matplotlib
from matplotlib.backend_bases import FigureManagerBase
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, cbook

matplotlib.use("agg")

__version__ = 2


# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:
        raise ValueError(f'{arg!r} unknown boolean')


def _option_context(arg):
    if arg in [None, 'reset', 'close-figs']:
        return arg
    raise ValueError("Argument should be None or 'reset' or 'close-figs'")


def _option_format(arg):
    return directives.choice(arg, ('python', 'doctest'))


def mark_plot_labels(app, document):
    """
    To make plots referenceable, we need to move the reference from the
    "htmlonly" (or "latexonly") node to the actual figure node itself.
    """
    for name, explicit in document.nametypes.items():
        if not explicit:
            continue
        labelid = document.nameids[name]
        if labelid is None:
            continue
        node = document.ids[labelid]
        if node.tagname in ('html_only', 'latex_only'):
            for n in node:
                if n.tagname == 'figure':
                    sectname = name
                    for c in n:
                        if c.tagname == 'caption':
                            sectname = c.astext()
                            break

                    node['ids'].remove(labelid)
                    node['names'].remove(name)
                    n['ids'].append(labelid)
                    n['names'].append(name)
                    document.settings.env.labels[name] = \
                        document.settings.env.docname, labelid, sectname
                    break


class PlotDirective(Directive):
    """The ``.. plot::`` directive, as documented in the module's docstring."""

    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        'alt': directives.unchanged,
        'height': directives.length_or_unitless,
        'width': directives.length_or_percentage_or_unitless,
        'scale': directives.nonnegative_int,
        'align': Image.align,
        'class': directives.class_option,
        'include-source': _option_boolean,
        'show-source-link': _option_boolean,
        'format': _option_format,
        'context': _option_context,
        'nofigs': directives.flag,
        'caption': directives.unchanged,
        }

    def run(self):
        """Run the plot directive."""
        try:
            return run(self.arguments, self.content, self.options,
                       self.state_machine, self.state, self.lineno)
        except Exception as e:
            raise self.error(str(e))


def _copy_css_file(app, exc):
    if exc is None and app.builder.format == 'html':
        src = cbook._get_data_path('plot_directive/plot_directive.css')
        dst = app.outdir / Path('_static')
        dst.mkdir(exist_ok=True)
        # Use copyfile because we do not want to copy src's permissions.
        shutil.copyfile(src, dst / Path('plot_directive.css'))


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('plot', PlotDirective)
    app.add_config_value('plot_pre_code', None, True)
    app.add_config_value('plot_include_source', False, True)
    app.add_config_value('plot_html_show_source_link', True, True)
    app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)
    app.add_config_value('plot_basedir', None, True)
    app.add_config_value('plot_html_show_formats', True, True)
    app.add_config_value('plot_rcparams', {}, True)
    app.add_config_value('plot_apply_rcparams', False, True)
    app.add_config_value('plot_working_directory', None, True)
    app.add_config_value('plot_template', None, True)
    app.add_config_value('plot_srcset', [], True)
    app.connect('doctree-read', mark_plot_labels)
    app.add_css_file('plot_directive.css')
    app.connect('build-finished', _copy_css_file)
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True,
                'version': matplotlib.__version__}
    return metadata


# -----------------------------------------------------------------------------
# Doctest handling
# -----------------------------------------------------------------------------


def contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, '<string>', 'exec')
        return False
    except SyntaxError:
        pass
    r = re.compile(r'^\s*>>>', re.M)
    m = r.search(text)
    return bool(m)


def _split_code_at_show(text, function_name):
    """Split code at plt.show()."""

    is_doctest = contains_doctest(text)
    if function_name is None:
        parts = []
        part = []
        for line in text.split("\n"):
            if ((not is_doctest and line.startswith('plt.show(')) or
                   (is_doctest and line.strip() == '>>> plt.show()')):
                part.append(line)
                parts.append("\n".join(part))
                part = []
            else:
                part.append(line)
        if "\n".join(part).strip():
            parts.append("\n".join(part))
    else:
        parts = [text]
    return is_doctest, parts


# -----------------------------------------------------------------------------
# Template
# -----------------------------------------------------------------------------

_SOURCECODE = """
{{ source_code }}

.. only:: html

   {% if src_name or (html_show_formats and not multi_image) %}
   (
   {%- if src_name -%}
   :download:`Source code <{{ build_dir }}/{{ src_name }}>`
   {%- endif -%}
   {%- if html_show_formats and not multi_image -%}
     {%- for img in images -%}
       {%- for fmt in img.formats -%}
         {%- if src_name or not loop.first -%}, {% endif -%}
         :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
       {%- endfor -%}
     {%- endfor -%}
   {%- endif -%}
   )
   {% endif %}
"""

TEMPLATE_SRCSET = _SOURCECODE + """
   {% for img in images %}
   .. figure-mpl:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}
      {%- if caption -%}
      {{ caption }}  {# appropriate leading whitespace added beforehand #}
      {% endif -%}
      {%- if srcset -%}
        :srcset: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
        {%- for sr in srcset -%}
            , {{ build_dir }}/{{ img.basename }}.{{ sr }}.{{ default_fmt }} {{sr}}
        {%- endfor -%}
      {% endif %}

   {% if html_show_formats and multi_image %}
   (
    {%- for fmt in img.formats -%}
    {%- if not loop.first -%}, {% endif -%}
    :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
    {%- endfor -%}
   )
   {% endif %}


   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure-mpl:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

"""

TEMPLATE = _SOURCECODE + """

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}

      {% if html_show_formats and multi_image -%}
        (
        {%- for fmt in img.formats -%}
        {%- if not loop.first -%}, {% endif -%}
        :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
        {%- endfor -%}
        )
      {%- endif -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

"""

exception_template = """
.. only:: html

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# the context of the plot for all directives specified with the
# :context: option
plot_context = dict()


class ImageFile:
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, f"{self.basename}.{format}")

    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]


def out_of_date(original, derived, includes=None):
    """
    Return whether *derived* is out-of-date relative to *original* or any of
    the RST files included in it using the RST include directive (*includes*).
    *derived* and *original* are full paths, and *includes* is optionally a
    list of full paths which may have been included in the *original*.
    """
    if not os.path.exists(derived):
        return True

    if includes is None:
        includes = []
    files_to_check = [original, *includes]

    def out_of_date_one(original, derived_mtime):
        return (os.path.exists(original) and
                derived_mtime < os.stat(original).st_mtime)

    derived_mtime = os.stat(derived).st_mtime
    return any(out_of_date_one(f, derived_mtime) for f in files_to_check)


class PlotError(RuntimeError):
    pass


def _run_code(code, code_path, ns=None, function_name=None):
    """
    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """

    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.  Add its path to sys.path
    # so it can import any helper modules sitting beside it.
    pwd = os.getcwd()
    if setup.config.plot_working_directory is not None:
        try:
            os.chdir(setup.config.plot_working_directory)
        except OSError as err:
            raise OSError(f'{err}\n`plot_working_directory` option in '
                          f'Sphinx configuration file must be a valid '
                          f'directory path') from err
        except TypeError as err:
            raise TypeError(f'{err}\n`plot_working_directory` option in '
                            f'Sphinx configuration file must be a string or '
                            f'None') from err
    elif code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)

    with cbook._setattr_cm(
            sys, argv=[code_path], path=[os.getcwd(), *sys.path]), \
            contextlib.redirect_stdout(StringIO()):
        try:
            if ns is None:
                ns = {}
            if not ns:
                if setup.config.plot_pre_code is None:
                    exec('import numpy as np\n'
                         'from matplotlib import pyplot as plt\n', ns)
                else:
                    exec(str(setup.config.plot_pre_code), ns)
            if "__main__" in code:
                ns['__name__'] = '__main__'

            # Patch out non-interactive show() to avoid triggering a warning.
            with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
                exec(code, ns)
                if function_name is not None:
                    exec(function_name + "()", ns)

        except (Exception, SystemExit) as err:
            raise PlotError(traceback.format_exc()) from err
        finally:
            os.chdir(pwd)
    return ns


def clear_state(plot_rcparams, close=True):
    if close:
        plt.close('all')
    matplotlib.rc_file_defaults()
    matplotlib.rcParams.update(plot_rcparams)


def get_plot_formats(config):
    default_dpi = {'png': 80, 'hires.png': 200, 'pdf': 200}
    formats = []
    plot_formats = config.plot_formats
    for fmt in plot_formats:
        if isinstance(fmt, str):
            if ':' in fmt:
                suffix, dpi = fmt.split(':')
                formats.append((str(suffix), int(dpi)))
            else:
                formats.append((fmt, default_dpi.get(fmt, 80)))
        elif isinstance(fmt, (tuple, list)) and len(fmt) == 2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise PlotError('invalid image format "%r" in plot_formats' % fmt)
    return formats


def _parse_srcset(entries):
    """
    Parse srcset for multiples...
    """
    srcset = {}
    for entry in entries:
        entry = entry.strip()
        if len(entry) >= 2:
            mult = entry[:-1]
            srcset[float(mult)] = entry
        else:
            raise ExtensionError(f'srcset argument {entry!r} is invalid.')
    return srcset


def render_figures(code, code_path, output_dir, output_base, context,
                   function_name, config, context_reset=False,
                   close_figs=False,
                   code_includes=None):
    """
    Run a pyplot script and save the images in *output_dir*.

    Save the images under *output_dir* with file names derived from
    *output_base*
    """

    if function_name is not None:
        output_base = f'{output_base}_{function_name}'
    formats = get_plot_formats(config)

    # Try to determine if all images already exist

    is_doctest, code_pieces = _split_code_at_show(code, function_name)
    # Look for single-figure output files first
    img = ImageFile(output_base, output_dir)
    for format, dpi in formats:
        if context or out_of_date(code_path, img.filename(format),
                                  includes=code_includes):
            all_exists = False
            break
        img.formats.append(format)
    else:
        all_exists = True

    if all_exists:
        return [(code, [img])]

    # Then look for multi-figure output files
    results = []
    for i, code_piece in enumerate(code_pieces):
        images = []
        for j in itertools.count():
            if len(code_pieces) > 1:
                img = ImageFile('%s_%02d_%02d' % (output_base, i, j),
                                output_dir)
            else:
                img = ImageFile('%s_%02d' % (output_base, j), output_dir)
            for fmt, dpi in formats:
                if context or out_of_date(code_path, img.filename(fmt),
                                          includes=code_includes):
                    all_exists = False
                    break
                img.formats.append(fmt)

            # assume that if we have one, we have them all
            if not all_exists:
                all_exists = (j > 0)
                break
            images.append(img)
        if not all_exists:
            break
        results.append((code_piece, images))
    else:
        all_exists = True

    if all_exists:
        return results

    # We didn't find the files, so build them

    results = []
    ns = plot_context if context else {}

    if context_reset:
        clear_state(config.plot_rcparams)
        plot_context.clear()

    close_figs = not context or close_figs

    for i, code_piece in enumerate(code_pieces):

        if not context or config.plot_apply_rcparams:
            clear_state(config.plot_rcparams, close_figs)
        elif close_figs:
            plt.close('all')

        _run_code(doctest.script_from_examples(code_piece) if is_doctest
                  else code_piece,
                  code_path, ns, function_name)

        images = []
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        for j, figman in enumerate(fig_managers):
            if len(fig_managers) == 1 and len(code_pieces) == 1:
                img = ImageFile(output_base, output_dir)
            elif len(code_pieces) == 1:
                img = ImageFile("%s_%02d" % (output_base, j), output_dir)
            else:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j),
                                output_dir)
            images.append(img)

            for fmt, dpi in formats:
                try:
                    figman.canvas.figure.savefig(img.filename(fmt), dpi=dpi)
                    if fmt == formats[0][0] and config.plot_srcset:
                        # save a 2x, 3x etc version of the default...
                        srcset = _parse_srcset(config.plot_srcset)
                        for mult, suffix in srcset.items():
                            fm = f'{suffix}.{fmt}'
                            img.formats.append(fm)
                            figman.canvas.figure.savefig(img.filename(fm),
                                                         dpi=int(dpi * mult))
                except Exception as err:
                    raise PlotError(traceback.format_exc()) from err
                img.formats.append(fmt)

        results.append((code_piece, images))

    if not context or config.plot_apply_rcparams:
        clear_state(config.plot_rcparams, close=not context)

    return results


def run(arguments, content, options, state_machine, state, lineno):
    document = state_machine.document
    config = document.settings.env.config
    nofigs = 'nofigs' in options

    if config.plot_srcset and setup.app.builder.name == 'singlehtml':
        raise ExtensionError(
            'plot_srcset option not compatible with single HTML writer')

    formats = get_plot_formats(config)
    default_fmt = formats[0][0]

    options.setdefault('include-source', config.plot_include_source)
    options.setdefault('show-source-link', config.plot_html_show_source_link)

    if 'class' in options:
        # classes are parsed into a list of string, and output by simply
        # printing the list, abusing the fact that RST guarantees to strip
        # non-conforming characters
        options['class'] = ['plot-directive'] + options['class']
    else:
        options.setdefault('class', ['plot-directive'])
    keep_context = 'context' in options
    context_opt = None if not keep_context else options['context']

    rst_file = document.attributes['source']
    rst_dir = os.path.dirname(rst_file)

    if len(arguments):
        if not config.plot_basedir:
            source_file_name = os.path.join(setup.app.builder.srcdir,
                                            directives.uri(arguments[0]))
        else:
            source_file_name = os.path.join(setup.confdir, config.plot_basedir,
                                            directives.uri(arguments[0]))
        # If there is content, it will be passed as a caption.
        caption = '\n'.join(content)

        # Enforce unambiguous use of captions.
        if "caption" in options:
            if caption:
                raise ValueError(
                    'Caption specified in both content and options.'
                    ' Please remove ambiguity.'
                )
            # Use caption option
            caption = options["caption"]

        # If the optional function name is provided, use it
        if len(arguments) == 2:
            function_name = arguments[1]
        else:
            function_name = None

        code = Path(source_file_name).read_text(encoding='utf-8')
        output_base = os.path.basename(source_file_name)
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        output_base = '%s-%d.py' % (base, counter)
        function_name = None
        caption = options.get('caption', '')

    base, source_ext = os.path.splitext(output_base)
    if source_ext in ('.py', '.rst', '.txt'):
        output_base = base
    else:
        source_ext = ''

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace('.', '-')

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if 'format' in options:
        if options['format'] == 'python':
            is_doctest = False
        else:
            is_doctest = True

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = os.path.dirname(source_rel_name).lstrip(os.path.sep)

    # build_dir: where to place output files (temporarily)
    build_dir = os.path.join(os.path.dirname(setup.app.doctreedir),
                             'plot_directive',
                             source_rel_dir)
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # how to link to files from the RST file
    try:
        build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
    except ValueError:
        # on Windows, relpath raises ValueError when path and start are on
        # different mounts/drives
        build_dir_link = build_dir

    # get list of included rst files so that the output is updated when any
    # plots in the included files change. These attributes are modified by the
    # include directive (see the docutils.parsers.rst.directives.misc module).
    try:
        source_file_includes = [os.path.join(os.getcwd(), t[0])
                                for t in state.document.include_log]
    except AttributeError:
        # the document.include_log attribute only exists in docutils >=0.17,
        # before that we need to inspect the state machine
        possible_sources = {os.path.join(setup.confdir, t[0])
                            for t in state_machine.input_lines.items}
        source_file_includes = [f for f in possible_sources
                                if os.path.isfile(f)]
    # remove the source file itself from the includes
    try:
        source_file_includes.remove(source_file_name)
    except ValueError:
        pass

    # save script (if necessary)
    if options['show-source-link']:
        Path(build_dir, output_base + source_ext).write_text(
            doctest.script_from_examples(code)
            if source_file_name == rst_file and is_doctest
            else code,
            encoding='utf-8')

    # make figures
    try:
        results = render_figures(code=code,
                                 code_path=source_file_name,
                                 output_dir=build_dir,
                                 output_base=output_base,
                                 context=keep_context,
                                 function_name=function_name,
                                 config=config,
                                 context_reset=context_opt == 'reset',
                                 close_figs=context_opt == 'close-figs',
                                 code_includes=source_file_includes)
        errors = []
    except PlotError as err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2, "Exception occurred in plotting {}\n from {}:\n{}".format(
                output_base, source_file_name, err),
            line=lineno)
        results = [(code, [])]
        errors = [sm]

    # Properly indent the caption
    if caption and config.plot_srcset:
        caption = ':caption: ' + caption.replace('\n', ' ')
    elif caption:
        caption = '\n' + '\n'.join('      ' + line.strip()
                                   for line in caption.split('\n'))
    # generate output restructuredtext
    total_lines = []
    for j, (code_piece, images) in enumerate(results):
        if options['include-source']:
            if is_doctest:
                lines = ['', *code_piece.splitlines()]
            else:
                lines = ['.. code-block:: python', '',
                         *textwrap.indent(code_piece, '    ').splitlines()]
            source_code = "\n".join(lines)
        else:
            source_code = ""

        if nofigs:
            images = []

        if 'alt' in options:
            options['alt'] = options['alt'].replace('\n', ' ')

        opts = [
            f':{key}: {val}' for key, val in options.items()
            if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]

        # Not-None src_name signals the need for a source download in the
        # generated html
        if j == 0 and options['show-source-link']:
            src_name = output_base + source_ext
        else:
            src_name = None
        if config.plot_srcset:
            srcset = [*_parse_srcset(config.plot_srcset).values()]
            template = TEMPLATE_SRCSET
        else:
            srcset = None
            template = TEMPLATE

        result = jinja2.Template(config.plot_template or template).render(
            default_fmt=default_fmt,
            build_dir=build_dir_link,
            src_name=src_name,
            multi_image=len(images) > 1,
            options=opts,
            srcset=srcset,
            images=images,
            source_code=source_code,
            html_show_formats=config.plot_html_show_formats and len(images),
            caption=caption)
        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    return errors
<|endoftext|>"""
Add a ``figure-mpl`` directive that is a responsive version of ``figure``.

This implementation is very similar to ``.. figure::``, except it also allows a
``srcset=`` argument to be passed to the image tag, hence allowing responsive
resolution images.

There is no particular reason this could not be used standalone, but is meant
to be used with :doc:`/api/sphinxext_plot_directive_api`.

Note that the directory organization is a bit different than ``.. figure::``.
See the *FigureMpl* documentation below.

"""
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
from sphinx.errors import ExtensionError

import matplotlib


class figmplnode(nodes.General, nodes.Element):
    pass


class FigureMpl(Figure):
    """
    Implements a directive to allow an optional hidpi image.

    Meant to be used with the *plot_srcset* configuration option in conf.py,
    and gets set in the TEMPLATE of plot_directive.py

    e.g.::

        .. figure-mpl:: plot_directive/some_plots-1.png
            :alt: bar
            :srcset: plot_directive/some_plots-1.png,
                     plot_directive/some_plots-1.2x.png 2.00x
            :class: plot-directive

    The resulting html (at ``some_plots.html``) is::

        <img src="sphx_glr_bar_001_hidpi.png"
            srcset="_images/some_plot-1.png,
                    _images/some_plots-1.2x.png 2.00x",
            alt="bar"
            class="plot_directive" />

    Note that the handling of subdirectories is different than that used by the sphinx
    figure directive::

        .. figure-mpl:: plot_directive/nestedpage/index-1.png
            :alt: bar
            :srcset: plot_directive/nestedpage/index-1.png
                     plot_directive/nestedpage/index-1.2x.png 2.00x
            :class: plot_directive

    The resulting html (at ``nestedpage/index.html``)::

        <img src="../_images/nestedpage-index-1.png"
            srcset="../_images/nestedpage-index-1.png,
                    ../_images/_images/nestedpage-index-1.2x.png 2.00x",
            alt="bar"
            class="sphx-glr-single-img" />

    where the subdirectory is included in the image name for uniqueness.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        'alt': directives.unchanged,
        'height': directives.length_or_unitless,
        'width': directives.length_or_percentage_or_unitless,
        'scale': directives.nonnegative_int,
        'align': Image.align,
        'class': directives.class_option,
        'caption': directives.unchanged,
        'srcset': directives.unchanged,
    }

    def run(self):

        image_node = figmplnode()

        imagenm = self.arguments[0]
        image_node['alt'] = self.options.get('alt', '')
        image_node['align'] = self.options.get('align', None)
        image_node['class'] = self.options.get('class', None)
        image_node['width'] = self.options.get('width', None)
        image_node['height'] = self.options.get('height', None)
        image_node['scale'] = self.options.get('scale', None)
        image_node['caption'] = self.options.get('caption', None)

        # we would like uri to be the highest dpi version so that
        # latex etc will use that.  But for now, lets just make
        # imagenm... maybe pdf one day?

        image_node['uri'] = imagenm
        image_node['srcset'] = self.options.get('srcset', None)

        return [image_node]


def _parse_srcsetNodes(st):
    """
    parse srcset...
    """
    entries = st.split(',')
    srcset = {}
    for entry in entries:
        spl = entry.strip().split(' ')
        if len(spl) == 1:
            srcset[0] = spl[0]
        elif len(spl) == 2:
            mult = spl[1][:-1]
            srcset[float(mult)] = spl[0]
        else:
            raise ExtensionError(f'srcset argument "{entry}" is invalid.')
    return srcset


def _copy_images_figmpl(self, node):

    # these will be the temporary place the plot-directive put the images eg:
    # ../../../build/html/plot_directive/users/explain/artists/index-1.png
    if node['srcset']:
        srcset = _parse_srcsetNodes(node['srcset'])
    else:
        srcset = None

    # the rst file's location:  eg /Users/username/matplotlib/doc/users/explain/artists
    docsource = PurePath(self.document['source']).parent

    # get the relpath relative to root:
    srctop = self.builder.srcdir
    rel = relpath(docsource, srctop).replace('.', '').replace(os.sep, '-')
    if len(rel):
        rel += '-'
    # eg: users/explain/artists

    imagedir = PurePath(self.builder.outdir, self.builder.imagedir)
    # eg: /Users/username/matplotlib/doc/build/html/_images/users/explain/artists

    Path(imagedir).mkdir(parents=True, exist_ok=True)

    # copy all the sources to the imagedir:
    if srcset:
        for src in srcset.values():
            # the entries in srcset are relative to docsource's directory
            abspath = PurePath(docsource, src)
            name = rel + abspath.name
            shutil.copyfile(abspath, imagedir / name)
    else:
        abspath = PurePath(docsource, node['uri'])
        name = rel + abspath.name
        shutil.copyfile(abspath, imagedir / name)

    return imagedir, srcset, rel


def visit_figmpl_html(self, node):

    imagedir, srcset, rel = _copy_images_figmpl(self, node)

    # /doc/examples/subd/plot_1.rst
    docsource = PurePath(self.document['source'])
    # /doc/
    # make sure to add the trailing slash:
    srctop = PurePath(self.builder.srcdir, '')
    # examples/subd/plot_1.rst
    relsource = relpath(docsource, srctop)
    # /doc/build/html
    desttop = PurePath(self.builder.outdir, '')
    # /doc/build/html/examples/subd
    dest = desttop / relsource

    # ../../_images/ for dirhtml and ../_images/ for html
    imagerel = PurePath(relpath(imagedir, dest.parent)).as_posix()
    if self.builder.name == "dirhtml":
        imagerel = f'..{imagerel}'

    # make uri also be relative...
    nm = PurePath(node['uri'][1:]).name
    uri = f'{imagerel}/{rel}{nm}'
    img_attrs = {'src': uri, 'alt': node['alt']}

    # make srcset str.  Need to change all the prefixes!
    maxsrc = uri
    if srcset:
        maxmult = -1
        srcsetst = ''
        for mult, src in srcset.items():
            nm = PurePath(src[1:]).name
            # ../../_images/plot_1_2_0x.png
            path = f'{imagerel}/{rel}{nm}'
            srcsetst += path
            if mult == 0:
                srcsetst += ', '
            else:
                srcsetst += f' {mult:1.2f}x, '

            if mult > maxmult:
                maxmult = mult
                maxsrc = path

        # trim trailing comma and space...
        img_attrs['srcset'] = srcsetst[:-2]

    if node['class'] is not None:
        img_attrs['class'] = ' '.join(node['class'])
    for style in ['width', 'height', 'scale']:
        if node[style]:
            if 'style' not in img_attrs:
                img_attrs['style'] = f'{style}: {node[style]};'
            else:
                img_attrs['style'] += f'{style}: {node[style]};'

    # <figure class="align-default" id="id1">
    # <a class="reference internal image-reference" href="_images/index-1.2x.png">
    # <img alt="_images/index-1.2x.png"
    #  src="_images/index-1.2x.png" style="width: 53%;" />
    # </a>
    # <figcaption>
    # <p><span class="caption-text">Figure caption is here....</span>
    # <a class="headerlink" href="#id1" title="Permalink to this image">#</a></p>
    # </figcaption>
    # </figure>
    self.body.append(
        self.starttag(
            node, 'figure',
            CLASS=f'align-{node["align"]}' if node['align'] else 'align-center'))
    self.body.append(
        self.starttag(node, 'a', CLASS='reference internal image-reference',
                      href=maxsrc) +
        self.emptytag(node, 'img', **img_attrs) +
        '</a>\n')
    if node['caption']:
        self.body.append(self.starttag(node, 'figcaption'))
        self.body.append(self.starttag(node, 'p'))
        self.body.append(self.starttag(node, 'span', CLASS='caption-text'))
        self.body.append(node['caption'])
        self.body.append('</span></p></figcaption>\n')
    self.body.append('</figure>\n')


def visit_figmpl_latex(self, node):

    if node['srcset'] is not None:
        imagedir, srcset = _copy_images_figmpl(self, node)
        maxmult = -1
        # choose the highest res version for latex:
        maxmult = max(srcset, default=-1)
        node['uri'] = PurePath(srcset[maxmult]).name

    self.visit_figure(node)


def depart_figmpl_html(self, node):
    pass


def depart_figmpl_latex(self, node):
    self.depart_figure(node)


def figurempl_addnode(app):
    app.add_node(figmplnode,
                 html=(visit_figmpl_html, depart_figmpl_html),
                 latex=(visit_figmpl_latex, depart_figmpl_latex))


def setup(app):
    app.add_directive("figure-mpl", FigureMpl)
    figurempl_addnode(app)
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True,
                'version': matplotlib.__version__}
    return metadata
<|endoftext|>"""
Helper functions for deprecating parts of the Matplotlib API.

This documentation is only relevant for Matplotlib developers, not for users.

.. warning::

    This module is for internal use only.  Do not use it in your own code.
    We may change the API at any time with no warning.

"""

import contextlib
import functools
import inspect
import math
import warnings


class MatplotlibDeprecationWarning(DeprecationWarning):
    """A class for issuing deprecation warnings for Matplotlib users."""


def _generate_deprecation_warning(
        since, message='', name='', alternative='', pending=False, obj_type='',
        addendum='', *, removal=''):
    if pending:
        if removal:
            raise ValueError("A pending deprecation cannot have a scheduled removal")
    elif removal == '':
        macro, meso, *_ = since.split('.')
        removal = f'{macro}.{int(meso) + 2}'
    if not message:
        message = (
            ("The %(name)s %(obj_type)s" if obj_type else "%(name)s") +
            (" will be deprecated in a future version" if pending else
             (" was deprecated in Matplotlib %(since)s" +
              (" and will be removed in %(removal)s" if removal else ""))) +
            "." +
            (" Use %(alternative)s instead." if alternative else "") +
            (" %(addendum)s" if addendum else ""))
    warning_cls = PendingDeprecationWarning if pending else MatplotlibDeprecationWarning
    return warning_cls(message % dict(
        func=name, name=name, obj_type=obj_type, since=since, removal=removal,
        alternative=alternative, addendum=addendum))


def warn_deprecated(
        since, *, message='', name='', alternative='', pending=False,
        obj_type='', addendum='', removal=''):
    """
    Display a standardized deprecation.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.
    message : str, optional
        Override the default deprecation message.  The ``%(since)s``,
        ``%(name)s``, ``%(alternative)s``, ``%(obj_type)s``, ``%(addendum)s``,
        and ``%(removal)s`` format specifiers will be replaced by the values
        of the respective arguments passed to this function.
    name : str, optional
        The name of the deprecated object.
    alternative : str, optional
        An alternative API that the user may use in place of the deprecated
        API.  The deprecation warning will tell the user about this alternative
        if provided.
    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.  Cannot be used together with *removal*.
    obj_type : str, optional
        The object type being deprecated.
    addendum : str, optional
        Additional text appended directly to the final message.
    removal : str, optional
        The expected removal version.  With the default (an empty string), a
        removal version is automatically computed from *since*.  Set to other
        Falsy values to not schedule a removal date.  Cannot be used together
        with *pending*.

    Examples
    --------
    ::

        # To warn of the deprecation of "matplotlib.name_of_module"
        warn_deprecated('1.4.0', name='matplotlib.name_of_module',
                        obj_type='module')
    """
    warning = _generate_deprecation_warning(
        since, message, name, alternative, pending, obj_type, addendum,
        removal=removal)
    from . import warn_external
    warn_external(warning, category=MatplotlibDeprecationWarning)


def deprecated(since, *, message='', name='', alternative='', pending=False,
               obj_type=None, addendum='', removal=''):
    """
    Decorator to mark a function, a class, or a property as deprecated.

    When deprecating a classmethod, a staticmethod, or a property, the
    ``@deprecated`` decorator should go *under* ``@classmethod`` and
    ``@staticmethod`` (i.e., `deprecated` should directly decorate the
    underlying callable), but *over* ``@property``.

    When deprecating a class ``C`` intended to be used as a base class in a
    multiple inheritance hierarchy, ``C`` *must* define an ``__init__`` method
    (if ``C`` instead inherited its ``__init__`` from its own base class, then
    ``@deprecated`` would mess up ``__init__`` inheritance when installing its
    own (deprecation-emitting) ``C.__init__``).

    Parameters are the same as for `warn_deprecated`, except that *obj_type*
    defaults to 'class' if decorating a class, 'attribute' if decorating a
    property, and 'function' otherwise.

    Examples
    --------
    ::

        @deprecated('1.4.0')
        def the_function_to_deprecate():
            pass
    """

    def deprecate(obj, message=message, name=name, alternative=alternative,
                  pending=pending, obj_type=obj_type, addendum=addendum):
        from matplotlib._api import classproperty

        if isinstance(obj, type):
            if obj_type is None:
                obj_type = "class"
            func = obj.__init__
            name = name or obj.__name__
            old_doc = obj.__doc__

            def finalize(wrapper, new_doc):
                try:
                    obj.__doc__ = new_doc
                except AttributeError:  # Can't set on some extension objects.
                    pass
                obj.__init__ = functools.wraps(obj.__init__)(wrapper)
                return obj

        elif isinstance(obj, (property, classproperty)):
            if obj_type is None:
                obj_type = "attribute"
            func = None
            name = name or obj.fget.__name__
            old_doc = obj.__doc__

            class _deprecated_property(type(obj)):
                def __get__(self, instance, owner=None):
                    if instance is not None or owner is not None \
                            and isinstance(self, classproperty):
                        emit_warning()
                    return super().__get__(instance, owner)

                def __set__(self, instance, value):
                    if instance is not None:
                        emit_warning()
                    return super().__set__(instance, value)

                def __delete__(self, instance):
                    if instance is not None:
                        emit_warning()
                    return super().__delete__(instance)

                def __set_name__(self, owner, set_name):
                    nonlocal name
                    if name == "<lambda>":
                        name = set_name

            def finalize(_, new_doc):
                return _deprecated_property(
                    fget=obj.fget, fset=obj.fset, fdel=obj.fdel, doc=new_doc)

        else:
            if obj_type is None:
                obj_type = "function"
            func = obj
            name = name or obj.__name__
            old_doc = func.__doc__

            def finalize(wrapper, new_doc):
                wrapper = functools.wraps(func)(wrapper)
                wrapper.__doc__ = new_doc
                return wrapper

        def emit_warning():
            warn_deprecated(
                since, message=message, name=name, alternative=alternative,
                pending=pending, obj_type=obj_type, addendum=addendum,
                removal=removal)

        def wrapper(*args, **kwargs):
            emit_warning()
            return func(*args, **kwargs)

        old_doc = inspect.cleandoc(old_doc or '').strip('\n')

        notes_header = '\nNotes\n-----'
        second_arg = ' '.join([t.strip() for t in
                               (message, f"Use {alternative} instead."
                                if alternative else "", addendum) if t])
        new_doc = (f"[*Deprecated*] {old_doc}\n"
                   f"{notes_header if notes_header not in old_doc else ''}\n"
                   f".. deprecated:: {since}\n"
                   f"   {second_arg}")

        if not old_doc:
            # This is to prevent a spurious 'unexpected unindent' warning from
            # docutils when the original docstring was blank.
            new_doc += r'\ '

        return finalize(wrapper, new_doc)

    return deprecate


class deprecate_privatize_attribute:
    """
    Helper to deprecate public access to an attribute (or method).

    This helper should only be used at class scope, as follows::

        class Foo:
            attr = _deprecate_privatize_attribute(*args, **kwargs)

    where *all* parameters are forwarded to `deprecated`.  This form makes
    ``attr`` a property which forwards read and write access to ``self._attr``
    (same name but with a leading underscore), with a deprecation warning.
    Note that the attribute name is derived from *the name this helper is
    assigned to*.  This helper also works for deprecating methods.
    """

    def __init__(self, *args, **kwargs):
        self.deprecator = deprecated(*args, **kwargs)

    def __set_name__(self, owner, name):
        setattr(owner, name, self.deprecator(
            property(lambda self: getattr(self, f"_{name}"),
                     lambda self, value: setattr(self, f"_{name}", value)),
            name=name))


# Used by _copy_docstring_and_deprecators to redecorate pyplot wrappers and
# boilerplate.py to retrieve original signatures.  It may seem natural to store
# this information as an attribute on the wrapper, but if the wrapper gets
# itself functools.wraps()ed, then such attributes are silently propagated to
# the outer wrapper, which is not desired.
DECORATORS = {}


def rename_parameter(since, old, new, func=None):
    """
    Decorator indicating that parameter *old* of *func* is renamed to *new*.

    The actual implementation of *func* should use *new*, not *old*.  If *old*
    is passed to *func*, a DeprecationWarning is emitted, and its value is
    used, even if *new* is also passed by keyword (this is to simplify pyplot
    wrapper functions, which always pass *new* explicitly to the Axes method).
    If *new* is also passed but positionally, a TypeError will be raised by the
    underlying function during argument binding.

    Examples
    --------
    ::

        @_api.rename_parameter("3.1", "bad_name", "good_name")
        def func(good_name): ...
    """

    decorator = functools.partial(rename_parameter, since, old, new)

    if func is None:
        return decorator

    signature = inspect.signature(func)
    assert old not in signature.parameters, (
        f"Matplotlib internal error: {old!r} cannot be a parameter for "
        f"{func.__name__}()")
    assert new in signature.parameters, (
        f"Matplotlib internal error: {new!r} must be a parameter for "
        f"{func.__name__}()")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if old in kwargs:
            warn_deprecated(
                since, message=f"The {old!r} parameter of {func.__name__}() "
                f"has been renamed {new!r} since Matplotlib {since}; support "
                f"for the old name will be dropped in %(removal)s.")
            kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)

    # wrapper() must keep the same documented signature as func(): if we
    # instead made both *old* and *new* appear in wrapper()'s signature, they
    # would both show up in the pyplot function for an Axes method as well and
    # pyplot would explicitly pass both arguments to the Axes method.

    DECORATORS[wrapper] = decorator
    return wrapper


class _deprecated_parameter_class:
    def __repr__(self):
        return "<deprecated parameter>"


_deprecated_parameter = _deprecated_parameter_class()


def delete_parameter(since, name, func=None, **kwargs):
    """
    Decorator indicating that parameter *name* of *func* is being deprecated.

    The actual implementation of *func* should keep the *name* parameter in its
    signature, or accept a ``**kwargs`` argument (through which *name* would be
    passed).

    Parameters that come after the deprecated parameter effectively become
    keyword-only (as they cannot be passed positionally without triggering the
    DeprecationWarning on the deprecated parameter), and should be marked as
    such after the deprecation period has passed and the deprecated parameter
    is removed.

    Parameters other than *since*, *name*, and *func* are keyword-only and
    forwarded to `.warn_deprecated`.

    Examples
    --------
    ::

        @_api.delete_parameter("3.1", "unused")
        def func(used_arg, other_arg, unused, more_args): ...
    """

    decorator = functools.partial(delete_parameter, since, name, **kwargs)

    if func is None:
        return decorator

    signature = inspect.signature(func)
    # Name of `**kwargs` parameter of the decorated function, typically
    # "kwargs" if such a parameter exists, or None if the decorated function
    # doesn't accept `**kwargs`.
    kwargs_name = next((param.name for param in signature.parameters.values()
                        if param.kind == inspect.Parameter.VAR_KEYWORD), None)
    if name in signature.parameters:
        kind = signature.parameters[name].kind
        is_varargs = kind is inspect.Parameter.VAR_POSITIONAL
        is_varkwargs = kind is inspect.Parameter.VAR_KEYWORD
        if not is_varargs and not is_varkwargs:
            name_idx = (
                # Deprecated parameter can't be passed positionally.
                math.inf if kind is inspect.Parameter.KEYWORD_ONLY
                # If call site has no more than this number of parameters, the
                # deprecated parameter can't have been passed positionally.
                else [*signature.parameters].index(name))
            func.__signature__ = signature = signature.replace(parameters=[
                param.replace(default=_deprecated_parameter)
                if param.name == name else param
                for param in signature.parameters.values()])
        else:
            name_idx = -1  # Deprecated parameter can always have been passed.
    else:
        is_varargs = is_varkwargs = False
        # Deprecated parameter can't be passed positionally.
        name_idx = math.inf
        assert kwargs_name, (
            f"Matplotlib internal error: {name!r} must be a parameter for "
            f"{func.__name__}()")

    addendum = kwargs.pop('addendum', None)

    @functools.wraps(func)
    def wrapper(*inner_args, **inner_kwargs):
        if len(inner_args) <= name_idx and name not in inner_kwargs:
            # Early return in the simple, non-deprecated case (much faster than
            # calling bind()).
            return func(*inner_args, **inner_kwargs)
        arguments = signature.bind(*inner_args, **inner_kwargs).arguments
        if is_varargs and arguments.get(name):
            warn_deprecated(
                since, message=f"Additional positional arguments to "
                f"{func.__name__}() are deprecated since %(since)s and "
                f"support for them will be removed in %(removal)s.")
        elif is_varkwargs and arguments.get(name):
            warn_deprecated(
                since, message=f"Additional keyword arguments to "
                f"{func.__name__}() are deprecated since %(since)s and "
                f"support for them will be removed in %(removal)s.")
        # We cannot just check `name not in arguments` because the pyplot
        # wrappers always pass all arguments explicitly.
        elif any(name in d and d[name] != _deprecated_parameter
                 for d in [arguments, arguments.get(kwargs_name, {})]):
            deprecation_addendum = (
                f"If any parameter follows {name!r}, they should be passed as "
                f"keyword, not positionally.")
            warn_deprecated(
                since,
                name=repr(name),
                obj_type=f"parameter of {func.__name__}()",
                addendum=(addendum + " " + deprecation_addendum) if addendum
                         else deprecation_addendum,
                **kwargs)
        return func(*inner_args, **inner_kwargs)

    DECORATORS[wrapper] = decorator
    return wrapper


def make_keyword_only(since, name, func=None):
    """
    Decorator indicating that passing parameter *name* (or any of the following
    ones) positionally to *func* is being deprecated.

    When used on a method that has a pyplot wrapper, this should be the
    outermost decorator, so that :file:`boilerplate.py` can access the original
    signature.
    """

    decorator = functools.partial(make_keyword_only, since, name)

    if func is None:
        return decorator

    signature = inspect.signature(func)
    POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
    KWO = inspect.Parameter.KEYWORD_ONLY
    assert (name in signature.parameters
            and signature.parameters[name].kind == POK), (
        f"Matplotlib internal error: {name!r} must be a positional-or-keyword "
        f"parameter for {func.__name__}(). If this error happens on a function with a "
        f"pyplot wrapper, make sure make_keyword_only() is the outermost decorator.")
    names = [*signature.parameters]
    name_idx = names.index(name)
    kwonly = [name for name in names[name_idx:]
              if signature.parameters[name].kind == POK]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Don't use signature.bind here, as it would fail when stacked with
        # rename_parameter and an "old" argument name is passed in
        # (signature.bind would fail, but the actual call would succeed).
        if len(args) > name_idx:
            warn_deprecated(
                since, message="Passing the %(name)s %(obj_type)s "
                "positionally is deprecated since Matplotlib %(since)s; the "
                "parameter will become keyword-only in %(removal)s.",
                name=name, obj_type=f"parameter of {func.__name__}()")
        return func(*args, **kwargs)

    # Don't modify *func*'s signature, as boilerplate.py needs it.
    wrapper.__signature__ = signature.replace(parameters=[
        param.replace(kind=KWO) if param.name in kwonly else param
        for param in signature.parameters.values()])
    DECORATORS[wrapper] = decorator
    return wrapper


def deprecate_method_override(method, obj, *, allow_empty=False, **kwargs):
    """
    Return ``obj.method`` with a deprecation if it was overridden, else None.

    Parameters
    ----------
    method
        An unbound method, i.e. an expression of the form
        ``Class.method_name``.  Remember that within the body of a method, one
        can always use ``__class__`` to refer to the class that is currently
        being defined.
    obj
        Either an object of the class where *method* is defined, or a subclass
        of that class.
    allow_empty : bool, default: False
        Whether to allow overrides by "empty" methods without emitting a
        warning.
    **kwargs
        Additional parameters passed to `warn_deprecated` to generate the
        deprecation warning; must at least include the "since" key.
    """

    def empty(): pass
    def empty_with_docstring(): """doc"""

    name = method.__name__
    bound_child = getattr(obj, name)
    bound_base = (
        method  # If obj is a class, then we need to use unbound methods.
        if isinstance(bound_child, type(empty)) and isinstance(obj, type)
        else method.__get__(obj))
    if (bound_child != bound_base
            and (not allow_empty
                 or (getattr(getattr(bound_child, "__code__", None),
                             "co_code", None)
                     not in [empty.__code__.co_code,
                             empty_with_docstring.__code__.co_code]))):
        warn_deprecated(**{"name": name, "obj_type": "method", **kwargs})
        return bound_child
    return None


@contextlib.contextmanager
def suppress_matplotlib_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
        yield
<|endoftext|>"""
Helper functions for managing the Matplotlib API.

This documentation is only relevant for Matplotlib developers, not for users.

.. warning::

    This module and its submodules are for internal use only.  Do not use them
    in your own code.  We may change the API at any time with no warning.

"""

import functools
import itertools
import pathlib
import re
import sys
import warnings

from .deprecation import (  # noqa: F401
    deprecated, warn_deprecated,
    rename_parameter, delete_parameter, make_keyword_only,
    deprecate_method_override, deprecate_privatize_attribute,
    suppress_matplotlib_deprecation_warning,
    MatplotlibDeprecationWarning)


class classproperty:
    """
    Like `property`, but also triggers on access via the class, and it is the
    *class* that's passed as argument.

    Examples
    --------
    ::

        class C:
            @classproperty
            def foo(cls):
                return cls.__name__

        assert C.foo == "C"
    """

    def __init__(self, fget, fset=None, fdel=None, doc=None):
        self._fget = fget
        if fset is not None or fdel is not None:
            raise ValueError('classproperty only implements fget.')
        self.fset = fset
        self.fdel = fdel
        # docs are ignored for now
        self._doc = doc

    def __get__(self, instance, owner):
        return self._fget(owner)

    @property
    def fget(self):
        return self._fget


# In the following check_foo() functions, the first parameter is positional-only to make
# e.g. `_api.check_isinstance([...], types=foo)` work.

def check_isinstance(types, /, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is an instance
    of one of *types*; if not, raise an appropriate TypeError.

    As a special case, a ``None`` entry in *types* is treated as NoneType.

    Examples
    --------
    >>> _api.check_isinstance((SomeClass, None), arg=arg)
    """
    none_type = type(None)
    types = ((types,) if isinstance(types, type) else
             (none_type,) if types is None else
             tuple(none_type if tp is None else tp for tp in types))

    def type_name(tp):
        return ("None" if tp is none_type
                else tp.__qualname__ if tp.__module__ == "builtins"
                else f"{tp.__module__}.{tp.__qualname__}")

    for k, v in kwargs.items():
        if not isinstance(v, types):
            names = [*map(type_name, types)]
            if "None" in names:  # Move it to the end for better wording.
                names.remove("None")
                names.append("None")
            raise TypeError(
                "{!r} must be an instance of {}, not a {}".format(
                    k,
                    ", ".join(names[:-1]) + " or " + names[-1]
                    if len(names) > 1 else names[0],
                    type_name(type(v))))


def check_in_list(values, /, *, _print_supported_values=True, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *values*;
    if not, raise an appropriate ValueError.

    Parameters
    ----------
    values : iterable
        Sequence of values to check on.

        Note: All values must support == comparisons.
        This means in particular the entries must not be numpy arrays.
    _print_supported_values : bool, default: True
        Whether to print *values* when raising ValueError.
    **kwargs : dict
        *key, value* pairs as keyword arguments to find in *values*.

    Raises
    ------
    ValueError
        If any *value* in *kwargs* is not found in *values*.

    Examples
    --------
    >>> _api.check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    if not kwargs:
        raise TypeError("No argument to check!")
    for key, val in kwargs.items():
        try:
            exists = val in values
        except ValueError:
            # `in` internally uses `val == values[i]`. There are some objects
            # that do not support == to arbitrary other objects, in particular
            # numpy arrays.
            # Since such objects are not allowed in values, we can gracefully
            # handle the case that val (typically provided by users) is of such
            # type and directly state it's not in the list instead of letting
            # the individual `val == values[i]` ValueError surface.
            exists = False
        if not exists:
            msg = f"{val!r} is not a valid value for {key}"
            if _print_supported_values:
                msg += f"; supported values are {', '.join(map(repr, values))}"
            raise ValueError(msg)


def check_shape(shape, /, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* has the shape *shape*;
    if not, raise an appropriate ValueError.

    *None* in the shape is treated as a "free" size that can have any length.
    e.g. (None, 2) -> (N, 2)

    The values checked must be numpy arrays.

    Examples
    --------
    To check for (N, 2) shaped arrays

    >>> _api.check_shape((None, 2), arg=arg, other_arg=other_arg)
    """
    for k, v in kwargs.items():
        data_shape = v.shape

        if (len(data_shape) != len(shape)
                or any(s != t and t is not None for s, t in zip(data_shape, shape))):
            dim_labels = iter(itertools.chain(
                'NMLKJIH',
                (f"D{i}" for i in itertools.count())))
            text_shape = ", ".join([str(n) if n is not None else next(dim_labels)
                                    for n in shape[::-1]][::-1])
            if len(shape) == 1:
                text_shape += ","

            raise ValueError(
                f"{k!r} must be {len(shape)}D with shape ({text_shape}), "
                f"but your input has shape {v.shape}"
            )


def check_getitem(mapping, /, **kwargs):
    """
    *kwargs* must consist of a single *key, value* pair.  If *key* is in
    *mapping*, return ``mapping[value]``; else, raise an appropriate
    ValueError.

    Examples
    --------
    >>> _api.check_getitem({"foo": "bar"}, arg=arg)
    """
    if len(kwargs) != 1:
        raise ValueError("check_getitem takes a single keyword argument")
    (k, v), = kwargs.items()
    try:
        return mapping[v]
    except KeyError:
        raise ValueError(
            f"{v!r} is not a valid value for {k}; supported values are "
            f"{', '.join(map(repr, mapping))}") from None


def caching_module_getattr(cls):
    """
    Helper decorator for implementing module-level ``__getattr__`` as a class.

    This decorator must be used at the module toplevel as follows::

        @caching_module_getattr
        class __getattr__:  # The class *must* be named ``__getattr__``.
            @property  # Only properties are taken into account.
            def name(self): ...

    The ``__getattr__`` class will be replaced by a ``__getattr__``
    function such that trying to access ``name`` on the module will
    resolve the corresponding property (which may be decorated e.g. with
    ``_api.deprecated`` for deprecating module globals).  The properties are
    all implicitly cached.  Moreover, a suitable AttributeError is generated
    and raised if no property with the given name exists.
    """

    assert cls.__name__ == "__getattr__"
    # Don't accidentally export cls dunders.
    props = {name: prop for name, prop in vars(cls).items()
             if isinstance(prop, property)}
    instance = cls()

    @functools.cache
    def __getattr__(name):
        if name in props:
            return props[name].__get__(instance)
        raise AttributeError(
            f"module {cls.__module__!r} has no attribute {name!r}")

    return __getattr__


def define_aliases(alias_d, cls=None):
    """
    Class decorator for defining property aliases.

    Use as ::

        @_api.define_aliases({"property": ["alias", ...], ...})
        class C: ...

    For each property, if the corresponding ``get_property`` is defined in the
    class so far, an alias named ``get_alias`` will be defined; the same will
    be done for setters.  If neither the getter nor the setter exists, an
    exception will be raised.

    The alias map is stored as the ``_alias_map`` attribute on the class and
    can be used by `.normalize_kwargs` (which assumes that higher priority
    aliases come last).
    """
    if cls is None:  # Return the actual class decorator.
        return functools.partial(define_aliases, alias_d)

    def make_alias(name):  # Enforce a closure over *name*.
        @functools.wraps(getattr(cls, name))
        def method(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)
        return method

    for prop, aliases in alias_d.items():
        exists = False
        for prefix in ["get_", "set_"]:
            if prefix + prop in vars(cls):
                exists = True
                for alias in aliases:
                    method = make_alias(prefix + prop)
                    method.__name__ = prefix + alias
                    method.__doc__ = f"Alias for `{prefix + prop}`."
                    setattr(cls, prefix + alias, method)
        if not exists:
            raise ValueError(
                f"Neither getter nor setter exists for {prop!r}")

    def get_aliased_and_aliases(d):
        return {*d, *(alias for aliases in d.values() for alias in aliases)}

    preexisting_aliases = getattr(cls, "_alias_map", {})
    conflicting = (get_aliased_and_aliases(preexisting_aliases)
                   & get_aliased_and_aliases(alias_d))
    if conflicting:
        # Need to decide on conflict resolution policy.
        raise NotImplementedError(
            f"Parent class already defines conflicting aliases: {conflicting}")
    cls._alias_map = {**preexisting_aliases, **alias_d}
    return cls


def select_matching_signature(funcs, *args, **kwargs):
    """
    Select and call the function that accepts ``*args, **kwargs``.

    *funcs* is a list of functions which should not raise any exception (other
    than `TypeError` if the arguments passed do not match their signature).

    `select_matching_signature` tries to call each of the functions in *funcs*
    with ``*args, **kwargs`` (in the order in which they are given).  Calls
    that fail with a `TypeError` are silently skipped.  As soon as a call
    succeeds, `select_matching_signature` returns its return value.  If no
    function accepts ``*args, **kwargs``, then the `TypeError` raised by the
    last failing call is re-raised.

    Callers should normally make sure that any ``*args, **kwargs`` can only
    bind a single *func* (to avoid any ambiguity), although this is not checked
    by `select_matching_signature`.

    Notes
    -----
    `select_matching_signature` is intended to help implementing
    signature-overloaded functions.  In general, such functions should be
    avoided, except for back-compatibility concerns.  A typical use pattern is
    ::

        def my_func(*args, **kwargs):
            params = select_matching_signature(
                [lambda old1, old2: locals(), lambda new: locals()],
                *args, **kwargs)
            if "old1" in params:
                warn_deprecated(...)
                old1, old2 = params.values()  # note that locals() is ordered.
            else:
                new, = params.values()
            # do things with params

    which allows *my_func* to be called either with two parameters (*old1* and
    *old2*) or a single one (*new*).  Note that the new signature is given
    last, so that callers get a `TypeError` corresponding to the new signature
    if the arguments they passed in do not match any signature.
    """
    # Rather than relying on locals() ordering, one could have just used func's
    # signature (``bound = inspect.signature(func).bind(*args, **kwargs);
    # bound.apply_defaults(); return bound``) but that is significantly slower.
    for i, func in enumerate(funcs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            if i == len(funcs) - 1:
                raise


def nargs_error(name, takes, given):
    """Generate a TypeError to be raised by function calls with wrong arity."""
    return TypeError(f"{name}() takes {takes} positional arguments but "
                     f"{given} were given")


def kwarg_error(name, kw):
    """
    Generate a TypeError to be raised by function calls with wrong kwarg.

    Parameters
    ----------
    name : str
        The name of the calling function.
    kw : str or Iterable[str]
        Either the invalid keyword argument name, or an iterable yielding
        invalid keyword arguments (e.g., a ``kwargs`` dict).
    """
    if not isinstance(kw, str):
        kw = next(iter(kw))
    return TypeError(f"{name}() got an unexpected keyword argument '{kw}'")


def recursive_subclasses(cls):
    """Yield *cls* and direct and indirect subclasses of *cls*."""
    yield cls
    for subcls in cls.__subclasses__():
        yield from recursive_subclasses(subcls)


def warn_external(message, category=None):
    """
    `warnings.warn` wrapper that sets *stacklevel* to "outside Matplotlib".

    The original emitter of the warning can be obtained by patching this
    function back to `warnings.warn`, i.e. ``_api.warn_external =
    warnings.warn`` (or ``functools.partial(warnings.warn, stacklevel=2)``,
    etc.).
    """
    kwargs = {}
    if sys.version_info[:2] >= (3, 12):
        # Go to Python's `site-packages` or `lib` from an editable install.
        basedir = pathlib.Path(__file__).parents[2]
        kwargs['skip_file_prefixes'] = (str(basedir / 'matplotlib'),
                                        str(basedir / 'mpl_toolkits'))
    else:
        frame = sys._getframe()
        for stacklevel in itertools.count(1):
            if frame is None:
                # when called in embedded context may hit frame is None
                kwargs['stacklevel'] = stacklevel
                break
            if not re.match(r"\A(matplotlib|mpl_toolkits)(\Z|\.(?!tests\.))",
                            # Work around sphinx-gallery not setting __name__.
                            frame.f_globals.get("__name__", "")):
                kwargs['stacklevel'] = stacklevel
                break
            frame = frame.f_back
        # preemptively break reference cycle between locals and the frame
        del frame
    warnings.warn(message, category, **kwargs)
<|endoftext|>from enum import Enum
import importlib


class BackendFilter(Enum):
    """
    Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

    .. versionadded:: 3.9
    """
    INTERACTIVE = 0
    NON_INTERACTIVE = 1


class BackendRegistry:
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.

    All use of ``BackendRegistry`` should be via the singleton instance
    ``backend_registry`` which can be imported from ``matplotlib.backends``.

    Each backend has a name, a module name containing the backend code, and an
    optional GUI framework that must be running if the backend is interactive.
    There are three sources of backends: built-in (source code is within the
    Matplotlib repository), explicit ``module://some.backend`` syntax (backend is
    obtained by loading the module), or via an entry point (self-registering
    backend in an external package).

    .. versionadded:: 3.9
    """
    # Mapping of built-in backend name to GUI framework, or "headless" for no
    # GUI framework. Built-in backends are those which are included in the
    # Matplotlib repo. A backend with name 'name' is located in the module
    # f"matplotlib.backends.backend_{name.lower()}"
    _BUILTIN_BACKEND_TO_GUI_FRAMEWORK = {
        "gtk3agg": "gtk3",
        "gtk3cairo": "gtk3",
        "gtk4agg": "gtk4",
        "gtk4cairo": "gtk4",
        "macosx": "macosx",
        "nbagg": "nbagg",
        "notebook": "nbagg",
        "qtagg": "qt",
        "qtcairo": "qt",
        "qt5agg": "qt5",
        "qt5cairo": "qt5",
        "tkagg": "tk",
        "tkcairo": "tk",
        "webagg": "webagg",
        "wx": "wx",
        "wxagg": "wx",
        "wxcairo": "wx",
        "agg": "headless",
        "cairo": "headless",
        "pdf": "headless",
        "pgf": "headless",
        "ps": "headless",
        "svg": "headless",
        "template": "headless",
    }

    # Reverse mapping of gui framework to preferred built-in backend.
    _GUI_FRAMEWORK_TO_BACKEND = {
        "gtk3": "gtk3agg",
        "gtk4": "gtk4agg",
        "headless": "agg",
        "macosx": "macosx",
        "qt": "qtagg",
        "qt5": "qt5agg",
        "qt6": "qtagg",
        "tk": "tkagg",
        "wx": "wxagg",
    }

    def __init__(self):
        # Only load entry points when first needed.
        self._loaded_entry_points = False

        # Mapping of non-built-in backend to GUI framework, added dynamically from
        # entry points and from matplotlib.use("module://some.backend") format.
        # New entries have an "unknown" GUI framework that is determined when first
        # needed by calling _get_gui_framework_by_loading.
        self._backend_to_gui_framework = {}

        # Mapping of backend name to module name, where different from
        # f"matplotlib.backends.backend_{backend_name.lower()}". These are either
        # hardcoded for backward compatibility, or loaded from entry points or
        # "module://some.backend" syntax.
        self._name_to_module = {
            "notebook": "nbagg",
        }

    def _backend_module_name(self, backend):
        if backend.startswith("module://"):
            return backend[9:]

        # Return name of module containing the specified backend.
        # Does not check if the backend is valid, use is_valid_backend for that.
        backend = backend.lower()

        # Check if have specific name to module mapping.
        backend = self._name_to_module.get(backend, backend)

        return (backend[9:] if backend.startswith("module://")
                else f"matplotlib.backends.backend_{backend}")

    def _clear(self):
        # Clear all dynamically-added data, used for testing only.
        self.__init__()

    def _ensure_entry_points_loaded(self):
        # Load entry points, if they have not already been loaded.
        if not self._loaded_entry_points:
            entries = self._read_entry_points()
            self._validate_and_store_entry_points(entries)
            self._loaded_entry_points = True

    def _get_gui_framework_by_loading(self, backend):
        # Determine GUI framework for a backend by loading its module and reading the
        # FigureCanvas.required_interactive_framework attribute.
        # Returns "headless" if there is no GUI framework.
        module = self.load_backend_module(backend)
        canvas_class = module.FigureCanvas
        return canvas_class.required_interactive_framework or "headless"

    def _read_entry_points(self):
        # Read entry points of modules that self-advertise as Matplotlib backends.
