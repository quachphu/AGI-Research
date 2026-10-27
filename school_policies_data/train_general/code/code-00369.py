        the Axes instance. This allows the spine to clear any axis
        properties when needed.
        """
        self.axis = axis
        self.stale = True

    def clear(self):
        """Clear the current spine."""
        self._clear()
        if self.axis is not None:
            self.axis.clear()

    def _clear(self):
        """
        Clear things directly related to the spine.

        In this way it is possible to avoid clearing the Axis as well when calling
        from library code where it is known that the Axis is cleared separately.
        """
        self._position = None  # clear position

    def _adjust_location(self):
        """Automatically set spine bounds to the view interval."""

        if self.spine_type == 'circle':
            return

        if self._bounds is not None:
            low, high = self._bounds
        elif self.spine_type in ('left', 'right'):
            low, high = self.axes.viewLim.intervaly
        elif self.spine_type in ('top', 'bottom'):
            low, high = self.axes.viewLim.intervalx
        else:
            raise ValueError(f'unknown spine spine_type: {self.spine_type}')

        if self._patch_type == 'arc':
            if self.spine_type in ('bottom', 'top'):
                try:
                    direction = self.axes.get_theta_direction()
                except AttributeError:
                    direction = 1
                try:
                    offset = self.axes.get_theta_offset()
                except AttributeError:
                    offset = 0
                low = low * direction + offset
                high = high * direction + offset
                if low > high:
                    low, high = high, low

                self._path = mpath.Path.arc(np.rad2deg(low), np.rad2deg(high))

                if self.spine_type == 'bottom':
                    if self.axis is None:
                        tr = mtransforms.IdentityTransform()
                    else:
                        tr = self.axis.get_transform()
                    rmin, rmax = tr.transform(self.axes.viewLim.intervaly)
                    try:
                        rorigin = self.axes.get_rorigin()
                    except AttributeError:
                        rorigin = rmin
                    else:
                        rorigin = tr.transform(rorigin)
                    scaled_diameter = (rmin - rorigin) / (rmax - rorigin)
                    self._height = scaled_diameter
                    self._width = scaled_diameter

            else:
                raise ValueError('unable to set bounds for spine "%s"' %
                                 self.spine_type)
        else:
            v1 = self._path.vertices
            assert v1.shape == (2, 2), 'unexpected vertices shape'
            if self.spine_type in ['left', 'right']:
                v1[0, 1] = low
                v1[1, 1] = high
            elif self.spine_type in ['bottom', 'top']:
                v1[0, 0] = low
                v1[1, 0] = high
            else:
                raise ValueError('unable to set bounds for spine "%s"' %
                                 self.spine_type)

    @allow_rasterization
    def draw(self, renderer):
        self._adjust_location()
        ret = super().draw(renderer)
        self.stale = False
        return ret

    def set_position(self, position):
        """
        Set the position of the spine.

        Spine position is specified by a 2 tuple of (position type,
        amount). The position types are:

        * 'outward': place the spine out from the data area by the specified
          number of points. (Negative values place the spine inwards.)
        * 'axes': place the spine at the specified Axes coordinate (0 to 1).
        * 'data': place the spine at the specified data coordinate.

        Additionally, shorthand notations define a special positions:

        * 'center' -> ``('axes', 0.5)``
        * 'zero' -> ``('data', 0.0)``

        Examples
        --------
        :doc:`/gallery/spines/spine_placement_demo`
        """
        if position in ('center', 'zero'):  # special positions
            pass
        else:
            if len(position) != 2:
                raise ValueError("position should be 'center' or 2-tuple")
            if position[0] not in ['outward', 'axes', 'data']:
                raise ValueError("position[0] should be one of 'outward', "
                                 "'axes', or 'data' ")
        self._position = position
        self.set_transform(self.get_spine_transform())
        if self.axis is not None:
            self.axis.reset_ticks()
        self.stale = True

    def get_position(self):
        """Return the spine position."""
        self._ensure_position_is_set()
        return self._position

    def get_spine_transform(self):
        """Return the spine transform."""
        self._ensure_position_is_set()

        position = self._position
        if isinstance(position, str):
            if position == 'center':
                position = ('axes', 0.5)
            elif position == 'zero':
                position = ('data', 0)
        assert len(position) == 2, 'position should be 2-tuple'
        position_type, amount = position
        _api.check_in_list(['axes', 'outward', 'data'],
                           position_type=position_type)
        if self.spine_type in ['left', 'right']:
            base_transform = self.axes.get_yaxis_transform(which='grid')
        elif self.spine_type in ['top', 'bottom']:
            base_transform = self.axes.get_xaxis_transform(which='grid')
        else:
            raise ValueError(f'unknown spine spine_type: {self.spine_type!r}')

        if position_type == 'outward':
            if amount == 0:  # short circuit commonest case
                return base_transform
            else:
                offset_vec = {'left': (-1, 0), 'right': (1, 0),
                              'bottom': (0, -1), 'top': (0, 1),
                              }[self.spine_type]
                # calculate x and y offset in dots
                offset_dots = amount * np.array(offset_vec) / 72
                return (base_transform
                        + mtransforms.ScaledTranslation(
                            *offset_dots, self.get_figure(root=False).dpi_scale_trans))
        elif position_type == 'axes':
            if self.spine_type in ['left', 'right']:
                # keep y unchanged, fix x at amount
                return (mtransforms.Affine2D.from_values(0, 0, 0, 1, amount, 0)
                        + base_transform)
            elif self.spine_type in ['bottom', 'top']:
                # keep x unchanged, fix y at amount
                return (mtransforms.Affine2D.from_values(1, 0, 0, 0, 0, amount)
                        + base_transform)
        elif position_type == 'data':
            if self.spine_type in ('right', 'top'):
                # The right and top spines have a default position of 1 in
                # axes coordinates.  When specifying the position in data
                # coordinates, we need to calculate the position relative to 0.
                amount -= 1
            if self.spine_type in ('left', 'right'):
                return mtransforms.blended_transform_factory(
                    mtransforms.Affine2D().translate(amount, 0)
                    + self.axes.transData,
                    self.axes.transData)
            elif self.spine_type in ('bottom', 'top'):
                return mtransforms.blended_transform_factory(
                    self.axes.transData,
                    mtransforms.Affine2D().translate(0, amount)
                    + self.axes.transData)

    def set_bounds(self, low=None, high=None):
        """
        Set the spine bounds.

        Parameters
        ----------
        low : float or None, optional
            The lower spine bound. Passing *None* leaves the limit unchanged.

            The bounds may also be passed as the tuple (*low*, *high*) as the
            first positional argument.

            .. ACCEPTS: (low: float, high: float)

        high : float or None, optional
            The higher spine bound. Passing *None* leaves the limit unchanged.
        """
        if self.spine_type == 'circle':
            raise ValueError(
                'set_bounds() method incompatible with circular spines')
        if high is None and np.iterable(low):
            low, high = low
        old_low, old_high = self.get_bounds() or (None, None)
        if low is None:
            low = old_low
        if high is None:
            high = old_high
        self._bounds = (low, high)
        self.stale = True

    def get_bounds(self):
        """Get the bounds of the spine."""
        return self._bounds

    @classmethod
    def linear_spine(cls, axes, spine_type, **kwargs):
        """Create and return a linear `Spine`."""
        # all values of 0.999 get replaced upon call to set_bounds()
        if spine_type == 'left':
            path = mpath.Path([(0.0, 0.999), (0.0, 0.999)])
        elif spine_type == 'right':
            path = mpath.Path([(1.0, 0.999), (1.0, 0.999)])
        elif spine_type == 'bottom':
            path = mpath.Path([(0.999, 0.0), (0.999, 0.0)])
        elif spine_type == 'top':
            path = mpath.Path([(0.999, 1.0), (0.999, 1.0)])
        else:
            raise ValueError('unable to make path for spine "%s"' % spine_type)
        result = cls(axes, spine_type, path, **kwargs)
        result.set_visible(mpl.rcParams[f'axes.spines.{spine_type}'])

        return result

    @classmethod
    def arc_spine(cls, axes, spine_type, center, radius, theta1, theta2,
                  **kwargs):
        """Create and return an arc `Spine`."""
        path = mpath.Path.arc(theta1, theta2)
        result = cls(axes, spine_type, path, **kwargs)
        result.set_patch_arc(center, radius, theta1, theta2)
        return result

    @classmethod
    def circular_spine(cls, axes, center, radius, **kwargs):
        """Create and return a circular `Spine`."""
        path = mpath.Path.unit_circle()
        spine_type = 'circle'
        result = cls(axes, spine_type, path, **kwargs)
        result.set_patch_circle(center, radius)
        return result

    def set_color(self, c):
        """
        Set the edgecolor.

        Parameters
        ----------
        c : :mpltype:`color`

        Notes
        -----
        This method does not modify the facecolor (which defaults to "none"),
        unlike the `.Patch.set_color` method defined in the parent class.  Use
        `.Patch.set_facecolor` to set the facecolor.
        """
        self.set_edgecolor(c)
        self.stale = True


class SpinesProxy:
    """
    A proxy to broadcast ``set_*()`` and ``set()`` method calls to contained `.Spines`.

    The proxy cannot be used for any other operations on its members.

    The supported methods are determined dynamically based on the contained
    spines. If not all spines support a given method, it's executed only on
    the subset of spines that support it.
    """
    def __init__(self, spine_dict):
        self._spine_dict = spine_dict

    def __getattr__(self, name):
        broadcast_targets = [spine for spine in self._spine_dict.values()
                             if hasattr(spine, name)]
        if (name != 'set' and not name.startswith('set_')) or not broadcast_targets:
            raise AttributeError(
                f"'SpinesProxy' object has no attribute '{name}'")

        def x(_targets, _funcname, *args, **kwargs):
            for spine in _targets:
                getattr(spine, _funcname)(*args, **kwargs)
        x = functools.partial(x, broadcast_targets, name)
        x.__doc__ = broadcast_targets[0].__doc__
        return x

    def __dir__(self):
        names = []
        for spine in self._spine_dict.values():
            names.extend(name
                         for name in dir(spine) if name.startswith('set_'))
        return list(sorted(set(names)))


class Spines(MutableMapping):
    r"""
    The container of all `.Spine`\s in an Axes.

    The interface is dict-like mapping names (e.g. 'left') to `.Spine` objects.
    Additionally, it implements some pandas.Series-like features like accessing
    elements by attribute::

        spines['top'].set_visible(False)
        spines.top.set_visible(False)

    Multiple spines can be addressed simultaneously by passing a list::

        spines[['top', 'right']].set_visible(False)

    Use an open slice to address all spines::

        spines[:].set_visible(False)

    The latter two indexing methods will return a `SpinesProxy` that broadcasts all
    ``set_*()`` and ``set()`` calls to its members, but cannot be used for any other
    operation.
    """
    def __init__(self, **kwargs):
        self._dict = kwargs

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __getstate__(self):
        return self._dict

    def __setstate__(self, state):
        self.__init__(**state)

    def __getattr__(self, name):
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(
                f"'Spines' object does not contain a '{name}' spine")

    def __getitem__(self, key):
        if isinstance(key, list):
            unknown_keys = [k for k in key if k not in self._dict]
            if unknown_keys:
                raise KeyError(', '.join(unknown_keys))
            return SpinesProxy({k: v for k, v in self._dict.items()
                                if k in key})
        if isinstance(key, tuple):
            raise ValueError('Multiple spines must be passed as a single list')
        if isinstance(key, slice):
            if key.start is None and key.stop is None and key.step is None:
                return SpinesProxy(self._dict)
            else:
                raise ValueError(
                    'Spines does not support slicing except for the fully '
                    'open slice [:] to access all spines.')
        return self._dict[key]

    def __setitem__(self, key, value):
        # TODO: Do we want to deprecate adding spines?
        self._dict[key] = value

    def __delitem__(self, key):
        # TODO: Do we want to deprecate deleting spines?
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)
<|endoftext|>import inspect

from . import _api


def kwarg_doc(text):
    """
    Decorator for defining the kwdoc documentation of artist properties.

    This decorator can be applied to artist property setter methods.
    The given text is stored in a private attribute ``_kwarg_doc`` on
    the method.  It is used to overwrite auto-generated documentation
    in the *kwdoc list* for artists. The kwdoc list is used to document
    ``**kwargs`` when they are properties of an artist. See e.g. the
    ``**kwargs`` section in `.Axes.text`.

    The text should contain the supported types, as well as the default
    value if applicable, e.g.:

        @_docstring.kwarg_doc("bool, default: :rc:`text.usetex`")
        def set_usetex(self, usetex):

    See Also
    --------
    matplotlib.artist.kwdoc

    """
    def decorator(func):
        func._kwarg_doc = text
        return func
    return decorator


class Substitution:
    """
    A decorator that performs %-substitution on an object's docstring.

    This decorator should be robust even if ``obj.__doc__`` is None (for
    example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Substitution with a sequence or dictionary
    suitable for performing substitution; then decorate a suitable function
    with the constructed object, e.g.::

        sub_author_name = Substitution(author='Jason')

        @sub_author_name
        def some_function(x):
            "%(author)s wrote this function"

        # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments::

        sub_first_last_names = Substitution('Edgar Allen', 'Poe')

        @sub_first_last_names
        def some_function(x):
            "%s %s wrote the Raven"
    """
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError("Only positional or keyword args are allowed")
        self.params = args or kwargs

    def __call__(self, func):
        if func.__doc__:
            func.__doc__ = inspect.cleandoc(func.__doc__) % self.params
        return func


class _ArtistKwdocLoader(dict):
    def __missing__(self, key):
        if not key.endswith(":kwdoc"):
            raise KeyError(key)
        name = key[:-len(":kwdoc")]
        from matplotlib.artist import Artist, kwdoc
        try:
            cls, = (cls for cls in _api.recursive_subclasses(Artist)
                    if cls.__name__ == name)
        except ValueError as e:
            raise KeyError(key) from e
        return self.setdefault(key, kwdoc(cls))


class _ArtistPropertiesSubstitution:
    """
    A class to substitute formatted placeholders in docstrings.

    This is realized in a single instance ``_docstring.interpd``.

    Use `~._ArtistPropertiesSubstition.register` to define placeholders and
    their substitution, e.g. ``_docstring.interpd.register(name="some value")``.

    Use this as a decorator to apply the substitution::

        @_docstring.interpd
        def some_func():
            '''Replace %(name)s.'''

    Decorating a class triggers substitution both on the class docstring and
    on the class' ``__init__`` docstring (which is a commonly required
    pattern for Artist subclasses).

    Substitutions of the form ``%(classname:kwdoc)s`` (ending with the
    literal ":kwdoc" suffix) trigger lookup of an Artist subclass with the
    given *classname*, and are substituted with the `.kwdoc` of that class.
    """

    def __init__(self):
        self.params = _ArtistKwdocLoader()

    def register(self, **kwargs):
        """
        Register substitutions.

        ``_docstring.interpd.register(name="some value")`` makes "name" available
        as a named parameter that will be replaced by "some value".
        """
        self.params.update(**kwargs)

    def __call__(self, obj):
        if obj.__doc__:
            obj.__doc__ = inspect.cleandoc(obj.__doc__) % self.params
        if isinstance(obj, type) and obj.__init__ != object.__init__:
            self(obj.__init__)
        return obj


def copy(source):
    """Copy a docstring from another source function (if present)."""
    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return target
    return do_copy


# Create a decorator that will house the various docstring snippets reused
# throughout Matplotlib.
interpd = _ArtistPropertiesSubstitution()
<|endoftext|>r"""
:mod:`~matplotlib.gridspec` contains classes that help to layout multiple
`~.axes.Axes` in a grid-like pattern within a figure.

The `GridSpec` specifies the overall grid structure. Individual cells within
the grid are referenced by `SubplotSpec`\s.

Often, users need not access this module directly, and can use higher-level
methods like `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` and
`~.Figure.subfigures`. See the tutorial :ref:`arranging_axes` for a guide.
"""

import copy
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox

_log = logging.getLogger(__name__)


class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        if not isinstance(nrows, Integral) or nrows <= 0:
            raise ValueError(
                f"Number of rows must be a positive integer, not {nrows!r}")
        if not isinstance(ncols, Integral) or ncols <= 0:
            raise ValueError(
                f"Number of columns must be a positive integer, not {ncols!r}")
        self._nrows, self._ncols = nrows, ncols
        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)

    def __repr__(self):
        height_arg = (f', height_ratios={self._row_height_ratios!r}'
                      if len(set(self._row_height_ratios)) != 1 else '')
        width_arg = (f', width_ratios={self._col_width_ratios!r}'
                     if len(set(self._col_width_ratios)) != 1 else '')
        return '{clsname}({nrows}, {ncols}{optionals})'.format(
            clsname=self.__class__.__name__,
            nrows=self._nrows,
            ncols=self._ncols,
            optionals=height_arg + width_arg,
            )

    nrows = property(lambda self: self._nrows,
                     doc="The number of rows in the grid.")
    ncols = property(lambda self: self._ncols,
                     doc="The number of columns in the grid.")

    def get_geometry(self):
        """
        Return a tuple containing the number of rows and columns in the grid.
        """
        return self._nrows, self._ncols

    def get_subplot_params(self, figure=None):
        # Must be implemented in subclasses
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        Create and return a `.SubplotSpec` instance.

        Parameters
        ----------
        loc : (int, int)
            The position of the subplot in the grid as
            ``(row_index, column_index)``.
        rowspan, colspan : int, default: 1
            The number of rows and columns the subplot should span in the grid.
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        """
        Set the relative widths of the columns.

        *width_ratios* must be of length *ncols*. Each column gets a relative
        width of ``width_ratios[i] / sum(width_ratios)``.
        """
        if width_ratios is None:
            width_ratios = [1] * self._ncols
        elif len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to '
                             'match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        """
        Return the width ratios.

        This is *None* if no width ratios have been set explicitly.
        """
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
        if height_ratios is None:
            height_ratios = [1] * self._nrows
        elif len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to '
                             'match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        """
        Return the height ratios.

        This is *None* if no height ratios have been set explicitly.
        """
        return self._row_height_ratios

    def get_grid_positions(self, fig):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
        nrows, ncols = self.get_geometry()
        subplot_params = self.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
        tot_width = right - left
        tot_height = top - bottom

        # calculate accumulated heights of columns
        cell_h = tot_height / (nrows + hspace*(nrows-1))
        sep_h = hspace * cell_h
        norm = cell_h * nrows / sum(self._row_height_ratios)
        cell_heights = [r * norm for r in self._row_height_ratios]
        sep_heights = [0] + ([sep_h] * (nrows-1))
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)

        # calculate accumulated widths of rows
        cell_w = tot_width / (ncols + wspace*(ncols-1))
        sep_w = wspace * cell_w
        norm = cell_w * ncols / sum(self._col_width_ratios)
        cell_widths = [r * norm for r in self._col_width_ratios]
        sep_widths = [0] + ([sep_w] * (ncols-1))
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)

        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    @staticmethod
    def _check_gridspec_exists(figure, nrows, ncols):
        """
        Check if the figure already has a gridspec with these dimensions,
        or create a new one
        """
        for ax in figure.get_axes():
            gs = ax.get_gridspec()
            if gs is not None:
                if hasattr(gs, 'get_topmost_subplotspec'):
                    # This is needed for colorbar gridspec layouts.
                    # This is probably OK because this whole logic tree
                    # is for when the user is doing simple things with the
                    # add_subplot command.  For complicated layouts
                    # like subgridspecs the proper gridspec is passed in...
                    gs = gs.get_topmost_subplotspec().get_gridspec()
                if gs.get_geometry() == (nrows, ncols):
                    return gs
        # else gridspec not found:
        return GridSpec(nrows, ncols, figure=figure)

    def __getitem__(self, key):
        """Create and return a `.SubplotSpec` instance."""
        nrows, ncols = self.get_geometry()

        def _normalize(key, size, axis):  # Includes last index.
            orig_key = key
            if isinstance(key, slice):
                start, stop, _ = key.indices(size)
                if stop > start:
                    return start, stop - 1
                raise IndexError("GridSpec slice would result in no space "
                                 "allocated for subplot")
            else:
                if key < 0:
                    key = key + size
                if 0 <= key < size:
                    return key, key
                elif axis is not None:
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"axis {axis} with size {size}")
                else:  # flat index
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"GridSpec with size {size}")

        if isinstance(key, tuple):
            try:
                k1, k2 = key
            except ValueError as err:
                raise ValueError("Unrecognized subplot spec") from err
            num1, num2 = np.ravel_multi_index(
                [_normalize(k1, nrows, 0), _normalize(k2, ncols, 1)],
                (nrows, ncols))
        else:  # Single key
            num1, num2 = _normalize(key, nrows * ncols, None)

        return SubplotSpec(self, num1, num2)

    def subplots(self, *, sharex=False, sharey=False, squeeze=True,
                 subplot_kw=None):
        """
        Add all subplots specified by this `GridSpec` to its parent figure.

        See `.Figure.subplots` for detailed documentation.
        """

        figure = self.figure

        if figure is None:
            raise ValueError("GridSpec.subplots() only works for GridSpecs "
                             "created with a parent figure")

        if not isinstance(sharex, str):
            sharex = "all" if sharex else "none"
        if not isinstance(sharey, str):
            sharey = "all" if sharey else "none"

        _api.check_in_list(["all", "row", "col", "none", False, True],
                           sharex=sharex, sharey=sharey)
        if subplot_kw is None:
            subplot_kw = {}
        # don't mutate kwargs passed by user...
        subplot_kw = subplot_kw.copy()

        # Create array to hold all Axes.
        axarr = np.empty((self._nrows, self._ncols), dtype=object)
        for row in range(self._nrows):
            for col in range(self._ncols):
                shared_with = {"none": None, "all": axarr[0, 0],
                               "row": axarr[row, 0], "col": axarr[0, col]}
                subplot_kw["sharex"] = shared_with[sharex]
                subplot_kw["sharey"] = shared_with[sharey]
                axarr[row, col] = figure.add_subplot(
                    self[row, col], **subplot_kw)

        # turn off redundant tick labeling
        if sharex in ["col", "all"]:
            for ax in axarr.flat:
                ax._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for ax in axarr.flat:
                ax._label_outer_yaxis(skip_non_rectangular_axes=True)

        if squeeze:
            # Discarding unneeded dimensions that equal 1.  If we only have one
            # subplot, just return it instead of a 1-element array.
            return axarr.item() if axarr.size == 1 else axarr.squeeze()
        else:
            # Returned axis array will be always 2-d, even if nrows=ncols=1.
            return axarr


class GridSpec(GridSpecBase):
    """
    A grid layout to place subplots within a figure.

    The location of the grid cells is determined in a similar way to
    `.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
    and *hspace*.

    Indexing a GridSpec instance returns a `.SubplotSpec`.
    """
    def __init__(self, nrows, ncols, figure=None,
                 left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None,
                 width_ratios=None, height_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.

        figure : `.Figure`, optional
            Only used for constrained layout to create a proper layoutgrid.

        left, right, top, bottom : float, optional
            Extent of the subplots as a fraction of figure width or height.
            Left cannot be larger than right, and bottom cannot be larger than
            top. If not given, the values will be inferred from a figure or
            rcParams at draw time. See also `GridSpec.get_subplot_params`.

        wspace : float, optional
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        hspace : float, optional
            The amount of height reserved for space between subplots,
            expressed as a fraction of the average axis height.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.

        """
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.wspace = wspace
        self.hspace = hspace
        self.figure = figure

        super().__init__(nrows, ncols,
                         width_ratios=width_ratios,
                         height_ratios=height_ratios)

    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]

    def update(self, **kwargs):
        """
        Update the subplot parameters of the grid.

        Parameters that are not explicitly given are not changed. Setting a
        parameter to *None* resets it to :rc:`figure.subplot.*`.

        Parameters
        ----------
        left, right, top, bottom : float or None, optional
            Extent of the subplots as a fraction of figure width or height.
        wspace, hspace : float, optional
            Spacing between the subplots as a fraction of the average subplot
            width / height.
        """
        for k, v in kwargs.items():
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError(f"{k} is an unknown keyword")
        for figmanager in _pylab_helpers.Gcf.figs.values():
            for ax in figmanager.canvas.figure.axes:
                if ax.get_subplotspec() is not None:
                    ss = ax.get_subplotspec().get_topmost_subplotspec()
                    if ss.get_gridspec() == self:
                        fig = ax.get_figure(root=False)
                        ax._set_position(ax.get_subplotspec().get_position(fig))

    def get_subplot_params(self, figure=None):
        """
        Return the `.SubplotParams` for the GridSpec.

        In order of precedence the values are taken from

        - non-*None* attributes of the GridSpec
        - the provided *figure*
        - :rc:`figure.subplot.*`

        Note that the ``figure`` attribute of the GridSpec is always ignored.
        """
        if figure is None:
            kw = {k: mpl.rcParams["figure.subplot."+k]
                  for k in self._AllowedKeys}
            subplotpars = SubplotParams(**kw)
        else:
            subplotpars = copy.copy(figure.subplotpars)

        subplotpars.update(**{k: getattr(self, k) for k in self._AllowedKeys})

        return subplotpars

    def locally_modified_subplot_params(self):
        """
        Return a list of the names of the subplot parameters explicitly set
        in the GridSpec.

        This is a subset of the attributes of `.SubplotParams`.
        """
        return [k for k in self._AllowedKeys if getattr(self, k)]

    def tight_layout(self, figure, renderer=None,
                     pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust subplot parameters to give specified padding.

        Parameters
        ----------
        figure : `.Figure`
            The figure.
        renderer :  `.RendererBase` subclass, optional
            The renderer to be used.
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font-size.
        h_pad, w_pad : float, optional
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top), default: None
            (left, bottom, right, top) rectangle in normalized figure
            coordinates that the whole subplots area (including labels) will
            fit into. Default (None) is the whole figure.
        """
        if renderer is None:
            renderer = figure._get_renderer()
        kwargs = _tight_layout.get_tight_layout_figure(
            figure, figure.axes,
            _tight_layout.get_subplotspec_list(figure.axes, grid_spec=self),
            renderer, pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        if kwargs:
            self.update(**kwargs)


class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """
    def __init__(self, nrows, ncols,
                 subplot_spec,
                 wspace=None, hspace=None,
                 height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            Number of rows and number of columns of the grid.
        subplot_spec : SubplotSpec
            Spec from which the layout parameters are inherited.
        wspace, hspace : float, optional
            See `GridSpec` for more details. If not specified default values
            (from the figure or rcParams) are used.
        height_ratios : array-like of length *nrows*, optional
            See `GridSpecBase` for details.
        width_ratios : array-like of length *ncols*, optional
            See `GridSpecBase` for details.
        """
        self._wspace = wspace
        self._hspace = hspace
        if isinstance(subplot_spec, SubplotSpec):
            self._subplot_spec = subplot_spec
        else:
            raise TypeError(
                            "subplot_spec must be type SubplotSpec, "
                            "usually from GridSpec, or axes.get_subplotspec.")
        self.figure = self._subplot_spec.get_gridspec().figure
        super().__init__(nrows, ncols,
                         width_ratios=width_ratios,
                         height_ratios=height_ratios)

    def get_subplot_params(self, figure=None):
        """Return a dictionary of subplot layout parameters."""
        hspace = (self._hspace if self._hspace is not None
                  else figure.subplotpars.hspace if figure is not None
                  else mpl.rcParams["figure.subplot.hspace"])
        wspace = (self._wspace if self._wspace is not None
                  else figure.subplotpars.wspace if figure is not None
                  else mpl.rcParams["figure.subplot.wspace"])

        figbox = self._subplot_spec.get_position(figure)
        left, bottom, right, top = figbox.extents

        return SubplotParams(left=left, right=right,
                             bottom=bottom, top=top,
                             wspace=wspace, hspace=hspace)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `.SubplotSpec` instance associated with the subplot.
        """
        return self._subplot_spec.get_topmost_subplotspec()


class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """
    def __init__(self, gridspec, num1, num2=None):
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    def __repr__(self):
        return (f"{self.get_gridspec()}["
                f"{self.rowspan.start}:{self.rowspan.stop}, "
                f"{self.colspan.start}:{self.colspan.stop}]")

    @staticmethod
    def _from_subplot_args(figure, args):
        """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
        if len(args) == 1:
            arg, = args
            if isinstance(arg, SubplotSpec):
                return arg
            elif not isinstance(arg, Integral):
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}")
            try:
                rows, cols, num = map(int, str(arg))
            except ValueError:
                raise ValueError(
                    f"Single argument to subplot must be a three-digit "
                    f"integer, not {arg!r}") from None
        elif len(args) == 3:
            rows, cols, num = args
        else:
            raise _api.nargs_error("subplot", takes="1 or 3", given=len(args))

        gs = GridSpec._check_gridspec_exists(figure, rows, cols)
        if gs is None:
            gs = GridSpec(rows, cols, figure=figure)
        if isinstance(num, tuple) and len(num) == 2:
            if not all(isinstance(n, Integral) for n in num):
                raise ValueError(
                    f"Subplot specifier tuple must contain integers, not {num}"
                )
            i, j = num
        else:
            if not isinstance(num, Integral) or num < 1 or num > rows*cols:
                raise ValueError(
                    f"num must be an integer with 1 <= num <= {rows*cols}, "
                    f"not {num!r}"
                )
            i = j = num
        return gs[i-1:j]

    # num2 is a property only to handle the case where it is None and someone
    # mutates num1.

    @property
    def num2(self):
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        self._num2 = value

    def get_gridspec(self):
        return self._gridspec

    def get_geometry(self):
        """
        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

        The indices *start* and *stop* define the range of the subplot within
        the `GridSpec`. *stop* is inclusive (i.e. for a single cell
        ``start == stop``).
        """
        rows, cols = self.get_gridspec().get_geometry()
        return rows, cols, self.num1, self.num2

    @property
    def rowspan(self):
        """The rows spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        return range(self.num1 // ncols, self.num2 // ncols + 1)

    @property
    def colspan(self):
        """The columns spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        # We explicitly support num2 referring to a column on num1's *left*, so
        # we must sort the column indices here so that the range makes sense.
        c1, c2 = sorted([self.num1 % ncols, self.num2 % ncols])
        return range(c1, c2 + 1)

    def is_first_row(self):
        return self.rowspan.start == 0

    def is_last_row(self):
        return self.rowspan.stop == self.get_gridspec().nrows

    def is_first_col(self):
        return self.colspan.start == 0

    def is_last_col(self):
        return self.colspan.stop == self.get_gridspec().ncols

    def get_position(self, figure):
        """
        Update the subplot position from ``figure.subplotpars``.
        """
        gridspec = self.get_gridspec()
        nrows, ncols = gridspec.get_geometry()
        rows, cols = np.unravel_index([self.num1, self.num2], (nrows, ncols))
        fig_bottoms, fig_tops, fig_lefts, fig_rights = \
            gridspec.get_grid_positions(figure)

        fig_bottom = fig_bottoms[rows].min()
        fig_top = fig_tops[rows].max()
        fig_left = fig_lefts[cols].min()
        fig_right = fig_rights[cols].max()
        return Bbox.from_extents(fig_left, fig_bottom, fig_right, fig_top)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
        gridspec = self.get_gridspec()
        if hasattr(gridspec, "get_topmost_subplotspec"):
            return gridspec.get_topmost_subplotspec()
        else:
            return self

    def __eq__(self, other):
        """
        Two SubplotSpecs are considered equal if they refer to the same
        position(s) in the same `GridSpec`.
        """
        # other may not even have the attributes we are checking.
        return ((self._gridspec, self.num1, self.num2)
                == (getattr(other, "_gridspec", object()),
                    getattr(other, "num1", object()),
                    getattr(other, "num2", object())))

    def __hash__(self):
        return hash((self._gridspec, self.num1, self.num2))

    def subgridspec(self, nrows, ncols, **kwargs):
        """
        Create a GridSpec within this subplot.

        The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
        a parent.

        Parameters
        ----------
        nrows : int
            Number of rows in grid.

        ncols : int
            Number of columns in grid.

        Returns
        -------
        `.GridSpecFromSubplotSpec`

        Other Parameters
        ----------------
        **kwargs
            All other parameters are passed to `.GridSpecFromSubplotSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding three subplots in the space occupied by a single subplot::

            fig = plt.figure()
            gs0 = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs0[0])
            ax2 = fig.add_subplot(gs0[1])
            gssub = gs0[2].subgridspec(1, 3)
            for i in range(3):
                fig.add_subplot(gssub[0, i])
        """
        return GridSpecFromSubplotSpec(nrows, ncols, self, **kwargs)


class SubplotParams:
    """
    Parameters defining the positioning of a subplots grid in a figure.
    """

    def __init__(self, left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None):
        """
        Defaults are given by :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
        for key in ["left", "bottom", "right", "top", "wspace", "hspace"]:
            setattr(self, key, mpl.rcParams[f"figure.subplot.{key}"])
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self, left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
        if ((left if left is not None else self.left)
                >= (right if right is not None else self.right)):
            raise ValueError('left cannot be >= right')
        if ((bottom if bottom is not None else self.bottom)
                >= (top if top is not None else self.top)):
            raise ValueError('bottom cannot be >= top')
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if bottom is not None:
            self.bottom = bottom
        if top is not None:
            self.top = top
        if wspace is not None:
            self.wspace = wspace
        if hspace is not None:
            self.hspace = hspace
<|endoftext|>"""
Helper module for the *bbox_inches* parameter in `.Figure.savefig`.
"""

from matplotlib.transforms import Bbox, TransformedBbox, Affine2D


def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
    """
    Temporarily adjust the figure so that only the specified area
    (bbox_inches) is saved.

    It modifies fig.bbox, fig.bbox_inches,
    fig.transFigure._boxout, and fig.patch.  While the figure size
    changes, the scale of the original figure is conserved.  A
    function which restores the original values are returned.
    """
    origBbox = fig.bbox
    origBboxInches = fig.bbox_inches
    _boxout = fig.transFigure._boxout

    old_aspect = []
    locator_list = []
    sentinel = object()
    for ax in fig.axes:
        locator = ax.get_axes_locator()
        if locator is not None:
            ax.apply_aspect(locator(ax, None))
        locator_list.append(locator)
        current_pos = ax.get_position(original=False).frozen()
        ax.set_axes_locator(lambda a, r, _pos=current_pos: _pos)
        # override the method that enforces the aspect ratio on the Axes
        if 'apply_aspect' in ax.__dict__:
            old_aspect.append(ax.apply_aspect)
        else:
            old_aspect.append(sentinel)
        ax.apply_aspect = lambda pos=None: None

    def restore_bbox():
        for ax, loc, aspect in zip(fig.axes, locator_list, old_aspect):
            ax.set_axes_locator(loc)
            if aspect is sentinel:
                # delete our no-op function which un-hides the original method
                del ax.apply_aspect
            else:
                ax.apply_aspect = aspect

        fig.bbox = origBbox
        fig.bbox_inches = origBboxInches
        fig.transFigure._boxout = _boxout
        fig.transFigure.invalidate()
        fig.patch.set_bounds(0, 0, 1, 1)

    if fixed_dpi is None:
        fixed_dpi = fig.dpi
    tr = Affine2D().scale(fixed_dpi)
    dpi_scale = fixed_dpi / fig.dpi

    fig.bbox_inches = Bbox.from_bounds(0, 0, *bbox_inches.size)
    x0, y0 = tr.transform(bbox_inches.p0)
    w1, h1 = fig.bbox.size * dpi_scale
    fig.transFigure._boxout = Bbox.from_bounds(-x0, -y0, w1, h1)
    fig.transFigure.invalidate()

    fig.bbox = TransformedBbox(fig.bbox_inches, tr)

    fig.patch.set_bounds(x0 / w1, y0 / h1,
                         fig.bbox.width / w1, fig.bbox.height / h1)

    return restore_bbox


def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi=None):
    """
    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    """

    bbox_inches, restore_bbox = bbox_inches_restore
    restore_bbox()
    r = adjust_bbox(fig, bbox_inches, fixed_dpi)

    return bbox_inches, r
<|endoftext|>"""
Internal debugging utilities, that are not expected to be used in the rest of
the codebase.

WARNING: Code in this module may change without prior notice!
"""

from io import StringIO
from pathlib import Path
import subprocess

from matplotlib.transforms import TransformNode


def graphviz_dump_transform(transform, dest, *, highlight=None):
    """
    Generate a graphical representation of the transform tree for *transform*
    using the :program:`dot` program (which this function depends on).  The
    output format (png, dot, etc.) is determined from the suffix of *dest*.

    Parameters
    ----------
    transform : `~matplotlib.transform.Transform`
        The represented transform.
    dest : str
        Output filename.  The extension must be one of the formats supported
        by :program:`dot`, e.g. png, svg, dot, ...
        (see https://www.graphviz.org/doc/info/output.html).
    highlight : list of `~matplotlib.transform.Transform` or None
        The transforms in the tree to be drawn in bold.
        If *None*, *transform* is highlighted.
    """

    if highlight is None:
        highlight = [transform]
    seen = set()

    def recurse(root, buf):
        if id(root) in seen:
            return
        seen.add(id(root))
        props = {}
        label = type(root).__name__
        if root._invalid:
            label = f'[{label}]'
        if root in highlight:
            props['style'] = 'bold'
        props['shape'] = 'box'
        props['label'] = '"%s"' % label
        props = ' '.join(map('{0[0]}={0[1]}'.format, props.items()))
        buf.write(f'{id(root)} [{props}];\n')
        for key, val in vars(root).items():
            if isinstance(val, TransformNode) and id(root) in val._parents:
                buf.write(f'"{id(root)}" -> "{id(val)}" '
                          f'[label="{key}", fontsize=10];\n')
                recurse(val, buf)

    buf = StringIO()
    buf.write('digraph G {\n')
    recurse(transform, buf)
    buf.write('}\n')
    subprocess.run(
        ['dot', '-T', Path(dest).suffix[1:], '-o', dest],
        input=buf.getvalue().encode('utf-8'), check=True)
<|endoftext|>"""
Abstract base classes define the primitives that renderers and
graphics contexts must implement to serve as a Matplotlib backend.

`RendererBase`
    An abstract base class to handle drawing/rendering operations.

`FigureCanvasBase`
    The abstraction layer that separates the `.Figure` from the backend
    specific details like a user interface drawing area.

`GraphicsContextBase`
    An abstract base class that provides color, line styles, etc.

`Event`
    The base class for all of the Matplotlib event handling.  Derived classes
    such as `KeyEvent` and `MouseEvent` store the meta data like keys and
    buttons pressed, x and y locations in pixel and `~.axes.Axes` coordinates.

`ShowBase`
    The base class for the ``Show`` class of each interactive backend; the
    'show' callable is then set to ``Show.__call__``.

`ToolContainerBase`
    The base class for the Toolbar class of each interactive backend.
"""

from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import pathlib
import signal
import socket
import sys
import time
import weakref
from weakref import WeakKeyDictionary

import numpy as np

import matplotlib as mpl
from matplotlib import (
    _api, backend_tools as tools, cbook, colors, _docstring, text,
    _tight_bbox, transforms, widgets, is_interactive, rcParams)
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle


_log = logging.getLogger(__name__)
_default_filetypes = {
    'eps': 'Encapsulated Postscript',
    'jpg': 'Joint Photographic Experts Group',
    'jpeg': 'Joint Photographic Experts Group',
    'pdf': 'Portable Document Format',
    'pgf': 'PGF code for LaTeX',
    'png': 'Portable Network Graphics',
    'ps': 'Postscript',
    'raw': 'Raw RGBA bitmap',
    'rgba': 'Raw RGBA bitmap',
    'svg': 'Scalable Vector Graphics',
    'svgz': 'Scalable Vector Graphics',
    'tif': 'Tagged Image File Format',
    'tiff': 'Tagged Image File Format',
    'webp': 'WebP Image Format',
}
_default_backends = {
    'eps': 'matplotlib.backends.backend_ps',
    'jpg': 'matplotlib.backends.backend_agg',
    'jpeg': 'matplotlib.backends.backend_agg',
    'pdf': 'matplotlib.backends.backend_pdf',
    'pgf': 'matplotlib.backends.backend_pgf',
    'png': 'matplotlib.backends.backend_agg',
    'ps': 'matplotlib.backends.backend_ps',
    'raw': 'matplotlib.backends.backend_agg',
    'rgba': 'matplotlib.backends.backend_agg',
    'svg': 'matplotlib.backends.backend_svg',
    'svgz': 'matplotlib.backends.backend_svg',
    'tif': 'matplotlib.backends.backend_agg',
    'tiff': 'matplotlib.backends.backend_agg',
    'webp': 'matplotlib.backends.backend_agg',
}


def register_backend(format, backend, description=None):
    """
    Register a backend for saving to a given file format.

    Parameters
    ----------
    format : str
        File extension
    backend : module string or canvas class
        Backend for handling file output
    description : str, default: ""
        Description of the file type.
    """
    if description is None:
        description = ''
    _default_backends[format] = backend
    _default_filetypes[format] = description


def get_registered_canvas_class(format):
    """
    Return the registered default canvas for given file format.
    Handles deferred import of required backend.
    """
    if format not in _default_backends:
        return None
    backend_class = _default_backends[format]
    if isinstance(backend_class, str):
        backend_class = importlib.import_module(backend_class).FigureCanvas
        _default_backends[format] = backend_class
    return backend_class


class RendererBase:
    """
    An abstract base class to handle drawing/rendering operations.

    The following methods must be implemented in the backend for full
    functionality (though just implementing `draw_path` alone would give a
    highly capable backend):

    * `draw_path`
    * `draw_image`
    * `draw_gouraud_triangles`

    The following methods *should* be implemented in the backend for
    optimization reasons:

    * `draw_text`
    * `draw_markers`
    * `draw_path_collection`
    * `draw_quad_mesh`
    """
    def __init__(self):
        super().__init__()
        self._texmanager = None
        self._text2path = text.TextToPath()
        self._raster_depth = 0
        self._rasterizing = False

    def open_group(self, s, gid=None):
        """
        Open a grouping element with label *s* and *gid* (if set) as id.

        Only used by the SVG renderer.
        """

    def close_group(self, s):
        """
        Close a grouping element with label *s*.

        Only used by the SVG renderer.
        """

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Draw a `~.path.Path` instance using the given affine transform."""
        raise NotImplementedError

    def draw_markers(self, gc, marker_path, marker_trans, path,
                     trans, rgbFace=None):
        """
        Draw a marker at each of *path*'s vertices (excluding control points).

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this method in order to draw the marker
        only once and reuse it multiple times.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        marker_path : `~matplotlib.path.Path`
            The path for the marker.
        marker_trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the marker.
        path : `~matplotlib.path.Path`
            The locations to draw the markers.
        trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the path.
        rgbFace : :mpltype:`color`, optional
        """
        for vertices, codes in path.iter_segments(trans, simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                self.draw_path(gc, marker_path,
                               marker_trans +
                               transforms.Affine2D().translate(x, y),
                               rgbFace)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        """
        Draw a collection of *paths*.

        Each path is first transformed by the corresponding entry
        in *all_transforms* (a list of (3, 3) matrices) and then by
        *master_transform*.  They are then translated by the corresponding
        entry in *offsets*, which has been first transformed by *offset_trans*.

        *facecolors*, *edgecolors*, *linewidths*, *linestyles*, and
        *antialiased* are lists that set the corresponding properties.

        *offset_position* is unused now, but the argument is kept for
        backwards compatibility.

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this in order to render each set of
        path data only once, and then reference that path multiple times with
        the different offsets, colors, styles etc.  The generator methods
        `_iter_collection_raw_paths` and `_iter_collection` are provided to
        help with (and standardize) the implementation across backends.  It
        is highly recommended to use those generators, so that changes to the
        behavior of `draw_path_collection` can be made globally.
        """
        path_ids = self._iter_collection_raw_paths(master_transform,
                                                   paths, all_transforms)

        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, list(path_ids), offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            path, transform = path_id
            # Only apply another translation if we have an offset, else we
            # reuse the initial transform.
            if xo != 0 or yo != 0:
                # The transformation can be used by multiple paths. Since
                # translate is a inplace operation, we need to copy the
                # transformation by .frozen() before applying the translation.
                transform = transform.frozen()
                transform.translate(xo, yo)
            self.draw_path(gc0, path, transform, rgbFace)

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        """
        Draw a quadmesh.

        The base (fallback) implementation converts the quadmesh to paths and
        then calls `draw_path_collection`.
        """

        from matplotlib.collections import QuadMesh
        paths = QuadMesh._convert_mesh_to_paths(coordinates)

        if edgecolors is None:
            edgecolors = facecolors
        linewidths = np.array([gc.get_linewidth()], float)

        return self.draw_path_collection(
            gc, master_transform, paths, [], offsets, offsetTrans, facecolors,
            edgecolors, linewidths, [], [antialiased], [None], 'screen')

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        """
        Draw a series of Gouraud triangles.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        triangles_array : (N, 3, 2) array-like
            Array of *N* (x, y) points for the triangles.
        colors_array : (N, 3, 4) array-like
            Array of *N* RGBA colors for each point of the triangles.
        transform : `~matplotlib.transforms.Transform`
            An affine transform to apply to the points.
        """
        raise NotImplementedError

    def _iter_collection_raw_paths(self, master_transform, paths,
                                   all_transforms):
        """
        Helper method (along with `_iter_collection`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the base path/transform combinations, given a
        master transform, a list of paths and list of transforms.

        The arguments should be exactly what is passed in to
        `draw_path_collection`.

        The backend should take each yielded path and transform and create an
        object that can be referenced (reused) later.
        """
        Npaths = len(paths)
        Ntransforms = len(all_transforms)
        N = max(Npaths, Ntransforms)

        if Npaths == 0:
            return

        transform = transforms.IdentityTransform()
        for i in range(N):
            path = paths[i % Npaths]
            if Ntransforms:
                transform = Affine2D(all_transforms[i % Ntransforms])
            yield path, transform + master_transform

    def _iter_collection_uses_per_path(self, paths, all_transforms,
                                       offsets, facecolors, edgecolors):
        """
        Compute how many times each raw path object returned by
        `_iter_collection_raw_paths` would be used when calling
        `_iter_collection`. This is intended for the backend to decide
        on the tradeoff between using the paths in-line and storing
        them once and reusing. Rounds up in case the number of uses
        is not the same for every path.
        """
        Npaths = len(paths)
        if Npaths == 0 or len(facecolors) == len(edgecolors) == 0:
            return 0
        Npath_ids = max(Npaths, len(all_transforms))
        N = max(Npath_ids, len(offsets))
        return (N + Npath_ids - 1) // Npath_ids

    def _iter_collection(self, gc, path_ids, offsets, offset_trans, facecolors,
                         edgecolors, linewidths, linestyles,
                         antialiaseds, urls, offset_position):
        """
        Helper method (along with `_iter_collection_raw_paths`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the path, offset and graphics context
        combinations to draw the path collection.  The caller should already
        have looped over the results of `_iter_collection_raw_paths` to draw
        this collection.

        The arguments should be the same as that passed into
        `draw_path_collection`, with the exception of *path_ids*, which is a
        list of arbitrary objects that the backend will use to reference one of
        the paths created in the `_iter_collection_raw_paths` stage.

        Each yielded result is of the form::

           xo, yo, path_id, gc, rgbFace

        where *xo*, *yo* is an offset; *path_id* is one of the elements of
        *path_ids*; *gc* is a graphics context and *rgbFace* is a color to
        use for filling the path.
        """
        Npaths = len(path_ids)
        Noffsets = len(offsets)
        N = max(Npaths, Noffsets)
        Nfacecolors = len(facecolors)
        Nedgecolors = len(edgecolors)
        Nlinewidths = len(linewidths)
        Nlinestyles = len(linestyles)
        Nurls = len(urls)

        if (Nfacecolors == 0 and Nedgecolors == 0) or Npaths == 0:
            return

        gc0 = self.new_gc()
        gc0.copy_properties(gc)

        def cycle_or_default(seq, default=None):
            # Cycle over *seq* if it is not empty; else always yield *default*.
            return (itertools.cycle(seq) if len(seq)
                    else itertools.repeat(default))

        pathids = cycle_or_default(path_ids)
        toffsets = cycle_or_default(offset_trans.transform(offsets), (0, 0))
        fcs = cycle_or_default(facecolors)
        ecs = cycle_or_default(edgecolors)
        lws = cycle_or_default(linewidths)
        lss = cycle_or_default(linestyles)
        aas = cycle_or_default(antialiaseds)
        urls = cycle_or_default(urls)

        if Nedgecolors == 0:
            gc0.set_linewidth(0.0)

        for pathid, (xo, yo), fc, ec, lw, ls, aa, url in itertools.islice(
                zip(pathids, toffsets, fcs, ecs, lws, lss, aas, urls), N):
            if not (np.isfinite(xo) and np.isfinite(yo)):
                continue
            if Nedgecolors:
                if Nlinewidths:
                    gc0.set_linewidth(lw)
                if Nlinestyles:
                    gc0.set_dashes(*ls)
                if len(ec) == 4 and ec[3] == 0.0:
                    gc0.set_linewidth(0)
                else:
                    gc0.set_foreground(ec)
            if fc is not None and len(fc) == 4 and fc[3] == 0:
                fc = None
            gc0.set_antialiased(aa)
            if Nurls:
                gc0.set_url(url)
            yield xo, yo, pathid, gc0, fc
        gc0.restore()

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to `draw_image`.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return 1.0

    def draw_image(self, gc, x, y, im, transform=None):
        """
        Draw an RGBA image.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            A graphics context with clipping information.

        x : float
            The distance in physical units (i.e., dots or pixels) from the left
            hand side of the canvas.

        y : float
            The distance in physical units (i.e., dots or pixels) from the
            bottom side of the canvas.

        im : (N, M, 4) array of `numpy.uint8`
            An array of RGBA pixels.

        transform : `~matplotlib.transforms.Affine2DBase`
            If and only if the concrete backend is written such that
            `option_scale_image` returns ``True``, an affine transformation
            (i.e., an `.Affine2DBase`) *may* be passed to `draw_image`.  The
            translation vector of the transformation is given in physical units
            (i.e., dots or pixels). Note that the transformation does not
            override *x* and *y*, and has to be applied *before* translating
            the result by *x* and *y* (this can be accomplished by adding *x*
            and *y* to the translation vector defined by *transform*).
        """
        raise NotImplementedError

    def option_image_nocomposite(self):
        """
        Return whether image composition by Matplotlib should be skipped.

        Raster backends should usually return False (letting the C-level
        rasterizer take care of image composition); vector backends should
        usually return ``not rcParams["image.composite_image"]``.
        """
        return False

    def option_scale_image(self):
        """
        Return whether arbitrary affine transformations in `draw_image` are
        supported (True for most vector backends).
        """
        return False

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        """
        Draw a TeX instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        x : float
            The x location of the text in display coords.
        y : float
            The y location of the text baseline in display coords.
        s : str
            The TeX text string.
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties.
        angle : float
            The rotation angle in degrees anti-clockwise.
        mtext : `~matplotlib.text.Text`
            The original text object to be rendered.
        """
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath="TeX")

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """
        Draw a text instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        x : float
            The x location of the text in display coords.
        y : float
            The y location of the text baseline in display coords.
        s : str
            The text string.
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties.
        angle : float
            The rotation angle in degrees anti-clockwise.
        ismath : bool or "TeX"
            If True, use mathtext parser.
        mtext : `~matplotlib.text.Text`
            The original text object to be rendered.

        Notes
        -----
        **Notes for backend implementers:**

        `.RendererBase.draw_text` also supports passing "TeX" to the *ismath*
        parameter to use TeX rendering, but this is not required for actual
        rendering backends, and indeed many builtin backends do not support
        this.  Rather, TeX rendering is provided by `~.RendererBase.draw_tex`.
        """
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath)

    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
        """
        Draw the text by converting them to paths using `.TextToPath`.

        This private helper supports the same parameters as
        `~.RendererBase.draw_text`; setting *ismath* to "TeX" triggers TeX
        rendering.
        """
        text2path = self._text2path
        fontsize = self.points_to_pixels(prop.get_size_in_points())
        verts, codes = text2path.get_text_path(prop, s, ismath=ismath)
        path = Path(verts, codes)
        if self.flipy():
            width, height = self.get_canvas_width_height()
            transform = (Affine2D()
                         .scale(fontsize / text2path.FONT_SCALE)
                         .rotate_deg(angle)
                         .translate(x, height - y))
        else:
            transform = (Affine2D()
                         .scale(fontsize / text2path.FONT_SCALE)
                         .rotate_deg(angle)
                         .translate(x, y))
        color = gc.get_rgb()
        gc.set_linewidth(0.0)
        self.draw_path(gc, path, transform, rgbFace=color)

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        Get the width, height, and descent (offset from the bottom to the baseline), in
        display coords, of the string *s* with `.FontProperties` *prop*.

        Whitespace at the start and the end of *s* is included in the reported width.
        """
        fontsize = prop.get_size_in_points()

        if ismath == 'TeX':
            # todo: handle properties
            return self.get_texmanager().get_text_width_height_descent(
                s, fontsize, renderer=self)

        dpi = self.points_to_pixels(72)
        if ismath:
            dims = self._text2path.mathtext_parser.parse(s, dpi, prop)
            return dims[0:3]  # return width, height, descent

        flags = self._text2path._get_hinting_flag()
        font = self._text2path._get_font(prop)
        font.set_size(fontsize, dpi)
        # the width and height of unrotated string
        font.set_text(s, 0.0, flags=flags)
        w, h = font.get_width_height()
        d = font.get_descent()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d /= 64.0
        return w, h, d

    def flipy(self):
        """
        Return whether y values increase from top to bottom.

        Note that this only affects drawing of texts.
        """
        return True

    def get_canvas_width_height(self):
        """Return the canvas width and height in display coords."""
        return 1, 1

    def get_texmanager(self):
        """Return the `.TexManager` instance."""
        if self._texmanager is None:
            self._texmanager = TexManager()
        return self._texmanager

    def new_gc(self):
        """Return an instance of a `.GraphicsContextBase`."""
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        """
        Convert points to display units.

        You need to override this function (unless your backend
        doesn't have a dpi, e.g., postscript or svg).  Some imaging
        systems assume some value for pixels per inch::

            points to pixels = points * pixels_per_inch/72 * dpi/72

        Parameters
        ----------
        points : float or array-like

        Returns
        -------
        Points converted to pixels
        """
        return points

    def start_rasterizing(self):
        """
        Switch to the raster renderer.

        Used by `.MixedModeRenderer`.
        """

    def stop_rasterizing(self):
        """
        Switch back to the vector renderer and draw the contents of the raster
        renderer as an image on the vector renderer.

        Used by `.MixedModeRenderer`.
        """

    def start_filter(self):
        """
        Switch to a temporary renderer for image filtering effects.

        Currently only supported by the agg renderer.
        """

    def stop_filter(self, filter_func):
        """
        Switch back to the original renderer.  The contents of the temporary
        renderer is processed with the *filter_func* and is drawn on the
        original renderer as an image.

        Currently only supported by the agg renderer.
        """

    def _draw_disabled(self):
        """
        Context manager to temporary disable drawing.

        This is used for getting the drawn size of Artists.  This lets us
        run the draw process to update any Python state but does not pay the
        cost of the draw_XYZ calls on the canvas.
        """
        no_ops = {
            meth_name: lambda *args, **kwargs: None
            for meth_name in dir(RendererBase)
            if (meth_name.startswith("draw_")
                or meth_name in ["open_group", "close_group"])
        }

        return _setattr_cm(self, **no_ops)


class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc."""

    def __init__(self):
        self._alpha = 1.0
        self._forced_alpha = False  # if True, _alpha overrides A from RGBA
        self._antialiased = 1  # use 0, 1 not True, False for extension code
        self._capstyle = CapStyle('butt')
        self._cliprect = None
        self._clippath = None
        self._dashes = 0, None
        self._joinstyle = JoinStyle('round')
        self._linestyle = 'solid'
        self._linewidth = 1
        self._rgb = (0.0, 0.0, 0.0, 1.0)
        self._hatch = None
        self._hatch_color = colors.to_rgba(rcParams['hatch.color'])
        self._hatch_linewidth = rcParams['hatch.linewidth']
        self._url = None
        self._gid = None
        self._snap = None
        self._sketch = None

    def copy_properties(self, gc):
        """Copy properties from *gc* to self."""
        self._alpha = gc._alpha
        self._forced_alpha = gc._forced_alpha
        self._antialiased = gc._antialiased
        self._capstyle = gc._capstyle
        self._cliprect = gc._cliprect
        self._clippath = gc._clippath
        self._dashes = gc._dashes
        self._joinstyle = gc._joinstyle
        self._linestyle = gc._linestyle
        self._linewidth = gc._linewidth
        self._rgb = gc._rgb
        self._hatch = gc._hatch
        self._hatch_color = gc._hatch_color
        self._hatch_linewidth = gc._hatch_linewidth
        self._url = gc._url
        self._gid = gc._gid
        self._snap = gc._snap
        self._sketch = gc._sketch

    def restore(self):
        """
        Restore the graphics context from the stack - needed only
        for backends that save graphics contexts on a stack.
        """

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends.
        """
        return self._alpha

    def get_antialiased(self):
        """Return whether the object should try to do antialiased rendering."""
        return self._antialiased

    def get_capstyle(self):
        """Return the `.CapStyle`."""
        return self._capstyle.name

    def get_clip_rectangle(self):
        """
        Return the clip rectangle as a `~matplotlib.transforms.Bbox` instance.
        """
        return self._cliprect

    def get_clip_path(self):
        """
        Return the clip path in the form (path, transform), where path
        is a `~.path.Path` instance, and transform is
        an affine transform to apply to the path before clipping.
        """
        if self._clippath is not None:
            tpath, tr = self._clippath.get_transformed_path_and_affine()
            if np.all(np.isfinite(tpath.vertices)):
                return tpath, tr
            else:
                _log.warning("Ill-defined clip_path detected. Returning None.")
                return None, None
        return None, None

    def get_dashes(self):
        """
        Return the dash style as an (offset, dash-list) pair.

        See `.set_dashes` for details.

        Default value is (None, None).
        """
        return self._dashes

    def get_forced_alpha(self):
        """
        Return whether the value given by get_alpha() should be used to
        override any other alpha-channel values.
        """
        return self._forced_alpha

    def get_joinstyle(self):
        """Return the `.JoinStyle`."""
        return self._joinstyle.name

    def get_linewidth(self):
        """Return the line width in points."""
        return self._linewidth

    def get_rgb(self):
        """Return a tuple of three or four floats from 0-1."""
        return self._rgb

    def get_url(self):
        """Return a url if one is set, None otherwise."""
        return self._url

    def get_gid(self):
        """Return the object identifier if one is set, None otherwise."""
        return self._gid

    def get_snap(self):
        """
        Return the snap setting, which can be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
        return self._snap

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on all backends.

        If ``alpha=None`` (the default), the alpha components of the
        foreground and fill colors will be used to set their respective
        transparencies (where applicable); otherwise, ``alpha`` will override
        them.
        """
        if alpha is not None:
            self._alpha = alpha
            self._forced_alpha = True
        else:
            self._alpha = 1.0
            self._forced_alpha = False
        self.set_foreground(self._rgb, isRGBA=True)

    def set_antialiased(self, b):
        """Set whether object should be drawn with antialiased rendering."""
        # Use ints to make life easier on extension code trying to read the gc.
        self._antialiased = int(bool(b))

    @_docstring.interpd
    def set_capstyle(self, cs):
        """
        Set how to draw endpoints of lines.

        Parameters
        ----------
        cs : `.CapStyle` or %(CapStyle)s
        """
        self._capstyle = CapStyle(cs)

    def set_clip_rectangle(self, rectangle):
        """Set the clip rectangle to a `.Bbox` or None."""
        self._cliprect = rectangle

    def set_clip_path(self, path):
        """Set the clip path to a `.TransformedPath` or None."""
        _api.check_isinstance((transforms.TransformedPath, None), path=path)
        self._clippath = path

    def set_dashes(self, dash_offset, dash_list):
        """
        Set the dash style for the gc.

        Parameters
        ----------
        dash_offset : float
            Distance, in points, into the dash pattern at which to
            start the pattern. It is usually set to 0.
        dash_list : array-like or None
            The on-off sequence as points.  None specifies a solid line. All
            values must otherwise be non-negative (:math:`\\ge 0`).

        Notes
        -----
        See p. 666 of the PostScript
        `Language Reference
        <https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf>`_
        for more info.
        """
        if dash_list is not None:
            dl = np.asarray(dash_list)
            if np.any(dl < 0.0):
                raise ValueError(
                    "All values in the dash list must be non-negative")
            if dl.size and not np.any(dl > 0.0):
                raise ValueError(
                    'At least one value in the dash list must be positive')
        self._dashes = dash_offset, dash_list

    def set_foreground(self, fg, isRGBA=False):
        """
        Set the foreground color.

        Parameters
        ----------
        fg : :mpltype:`color`
        isRGBA : bool
            If *fg* is known to be an ``(r, g, b, a)`` tuple, *isRGBA* can be
            set to True to improve performance.
        """
        if self._forced_alpha and isRGBA:
            self._rgb = fg[:3] + (self._alpha,)
        elif self._forced_alpha:
            self._rgb = colors.to_rgba(fg, self._alpha)
        elif isRGBA:
            self._rgb = fg
        else:
            self._rgb = colors.to_rgba(fg)

    @_docstring.interpd
    def set_joinstyle(self, js):
        """
        Set how to draw connections between line segments.

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
        self._joinstyle = JoinStyle(js)

    def set_linewidth(self, w):
        """Set the linewidth in points."""
        self._linewidth = float(w)

    def set_url(self, url):
        """Set the url for links in compatible backends."""
        self._url = url

    def set_gid(self, id):
        """Set the id."""
        self._gid = id

    def set_snap(self, snap):
        """
        Set the snap setting which may be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
        self._snap = snap

    def set_hatch(self, hatch):
        """Set the hatch style (for fills)."""
        self._hatch = hatch

    def get_hatch(self):
        """Get the current hatch style."""
        return self._hatch

    def get_hatch_path(self, density=6.0):
        """Return a `.Path` for the current hatch."""
        hatch = self.get_hatch()
        if hatch is None:
            return None
        return Path.hatch(hatch, density)

    def get_hatch_color(self):
        """Get the hatch color."""
        return self._hatch_color

    def set_hatch_color(self, hatch_color):
        """Set the hatch color."""
        self._hatch_color = hatch_color

    def get_hatch_linewidth(self):
        """Get the hatch linewidth."""
        return self._hatch_linewidth

    def set_hatch_linewidth(self, hatch_linewidth):
        """Set the hatch linewidth."""
        self._hatch_linewidth = hatch_linewidth

    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or `None`

            A 3-tuple with the following elements:

            * ``scale``: The amplitude of the wiggle perpendicular to the
              source line.
            * ``length``: The length of the wiggle along the line.
            * ``randomness``: The scale factor by which the length is
              shrunken or expanded.

            May return `None` if no sketch parameters were set.
        """
        return self._sketch

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source line, in
            pixels.  If scale is `None`, or not provided, no sketch filter will
            be provided.
        length : float, default: 128
            The length of the wiggle along the line, in pixels.
        randomness : float, default: 16
            The scale factor by which the length is shrunken or expanded.
        """
        self._sketch = (
            None if scale is None
            else (scale, length or 128., randomness or 16.))


class TimerBase:
    """
    A base class for providing timer events, useful for things animations.
    Backends need to implement a few specific methods in order to use their
    own timing mechanisms so that the timer events are integrated into their
    event loops.

    Subclasses must override the following methods:

    - ``_timer_start``: Backend-specific code for starting the timer.
    - ``_timer_stop``: Backend-specific code for stopping the timer.

    Subclasses may additionally override the following methods:

    - ``_timer_set_single_shot``: Code for setting the timer to single shot
      operating mode, if supported by the timer object.  If not, the `Timer`
      class itself will store the flag and the ``_on_timer`` method should be
      overridden to support such behavior.

    - ``_timer_set_interval``: Code for setting the interval on the timer, if
      there is a method for doing so on the timer object.

    - ``_on_timer``: The internal function that any timer object should call,
      which will handle the task of running all callbacks that have been set.
    """

    def __init__(self, interval=None, callbacks=None):
        """
        Parameters
        ----------
        interval : int, default: 1000ms
            The time between timer events in milliseconds.  Will be stored as
            ``timer.interval``.
        callbacks : list[tuple[callable, tuple, dict]]
            List of (func, args, kwargs) tuples that will be called upon timer
            events.  This list is accessible as ``timer.callbacks`` and can be
            manipulated directly, or the functions `~.TimerBase.add_callback`
            and `~.TimerBase.remove_callback` can be used.
        """
        self.callbacks = [] if callbacks is None else callbacks.copy()
        # Set .interval and not ._interval to go through the property setter.
        self.interval = 1000 if interval is None else interval
        self.single_shot = False

    def __del__(self):
        """Need to stop timer and possibly disconnect timer."""
        self._timer_stop()

    @_api.delete_parameter("3.9", "interval", alternative="timer.interval")
    def start(self, interval=None):
        """
        Start the timer object.

        Parameters
        ----------
        interval : int, optional
            Timer interval in milliseconds; overrides a previously set interval
            if provided.
        """
        if interval is not None:
            self.interval = interval
        self._timer_start()

    def stop(self):
        """Stop the timer."""
        self._timer_stop()

    def _timer_start(self):
        pass

    def _timer_stop(self):
        pass

    @property
    def interval(self):
        """The time between timer events, in milliseconds."""
        return self._interval

    @interval.setter
    def interval(self, interval):
        # Force to int since none of the backends actually support fractional
        # milliseconds, and some error or give warnings.
        # Some backends also fail when interval == 0, so ensure >= 1 msec
        interval = max(int(interval), 1)
        self._interval = interval
        self._timer_set_interval()

    @property
    def single_shot(self):
        """Whether this timer should stop after a single run."""
        return self._single

    @single_shot.setter
    def single_shot(self, ss):
        self._single = ss
        self._timer_set_single_shot()

    def add_callback(self, func, *args, **kwargs):
        """
        Register *func* to be called by timer when the event fires. Any
        additional arguments provided will be passed to *func*.

        This function returns *func*, which makes it possible to use it as a
        decorator.
        """
        self.callbacks.append((func, args, kwargs))
        return func

    def remove_callback(self, func, *args, **kwargs):
        """
        Remove *func* from list of callbacks.

        *args* and *kwargs* are optional and used to distinguish between copies
        of the same function registered to be called with different arguments.
        This behavior is deprecated.  In the future, ``*args, **kwargs`` won't
        be considered anymore; to keep a specific callback removable by itself,
        pass it to `add_callback` as a `functools.partial` object.
        """
        if args or kwargs:
            _api.warn_deprecated(
                "3.1", message="In a future version, Timer.remove_callback "
                "will not take *args, **kwargs anymore, but remove all "
                "callbacks where the callable matches; to keep a specific "
                "callback removable by itself, pass it to add_callback as a "
                "functools.partial object.")
            self.callbacks.remove((func, args, kwargs))
        else:
            funcs = [c[0] for c in self.callbacks]
            if func in funcs:
                self.callbacks.pop(funcs.index(func))

    def _timer_set_interval(self):
        """Used to set interval on underlying timer object."""

    def _timer_set_single_shot(self):
        """Used to set single shot on underlying timer object."""

    def _on_timer(self):
        """
        Runs all function that have been registered as callbacks. Functions
        can return False (or 0) if they should not be called any more. If there
        are no callbacks, the timer is automatically stopped.
        """
        for func, args, kwargs in self.callbacks:
            ret = func(*args, **kwargs)
            # docstring above explains why we use `if ret == 0` here,
            # instead of `if not ret`.
            # This will also catch `ret == False` as `False == 0`
            # but does not annoy the linters
            # https://docs.python.org/3/library/stdtypes.html#boolean-values
            if ret == 0:
                self.callbacks.remove((func, args, kwargs))

        if len(self.callbacks) == 0:
            self.stop()


class Event:
    """
    A Matplotlib event.

    The following attributes are defined and shown with their default values.
    Subclasses may define additional attributes.

    Attributes
    ----------
    name : str
        The event name.
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance generating the event.
    guiEvent
        The GUI event that triggered the Matplotlib event.
    """

    def __init__(self, name, canvas, guiEvent=None):
        self.name = name
        self.canvas = canvas
        self.guiEvent = guiEvent

    def _process(self):
        """Process this event on ``self.canvas``, then unset ``guiEvent``."""
        self.canvas.callbacks.process(self.name, self)
        self.guiEvent = None


class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas.

    In most backends, callbacks subscribed to this event will be fired after
    the rendering is complete but before the screen is updated. Any extra
    artists drawn to the canvas's renderer will be reflected without an
    explicit call to ``blit``.

    .. warning::

       Calling ``canvas.draw`` and ``canvas.blit`` in these callbacks may
       not be safe with all backends and may cause infinite recursion.

    A DrawEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    renderer : `RendererBase`
        The renderer for the draw event.
    """
    def __init__(self, name, canvas, renderer):
        super().__init__(name, canvas)
        self.renderer = renderer


class ResizeEvent(Event):
    """
    An event triggered by a canvas resize.

    A ResizeEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    width : int
        Width of the canvas in pixels.
    height : int
        Height of the canvas in pixels.
    """

    def __init__(self, name, canvas):
        super().__init__(name, canvas)
        self.width, self.height = canvas.get_width_height()


class CloseEvent(Event):
    """An event triggered by a figure being closed."""


class LocationEvent(Event):
    """
    An event that has a screen location.

    A LocationEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    x, y : int or None
        Event location in pixels from bottom left of canvas.
    inaxes : `~matplotlib.axes.Axes` or None
        The `~.axes.Axes` instance over which the mouse is, if any.
    xdata, ydata : float or None
        Data coordinates of the mouse within *inaxes*, or *None* if the mouse
        is not over an Axes.
    modifiers : frozenset
        The keyboard modifiers currently being pressed (except for KeyEvent).
    """

    _last_axes_ref = None

    def __init__(self, name, canvas, x, y, guiEvent=None, *, modifiers=None):
        super().__init__(name, canvas, guiEvent=guiEvent)
        # x position - pixels from left of canvas
        self.x = int(x) if x is not None else x
        # y position - pixels from right of canvas
        self.y = int(y) if y is not None else y
        self.inaxes = None  # the Axes instance the mouse is over
        self.xdata = None   # x coord of mouse in data coords
        self.ydata = None   # y coord of mouse in data coords
        self.modifiers = frozenset(modifiers if modifiers is not None else [])

        if x is None or y is None:
            # cannot check if event was in Axes if no (x, y) info
            return

        self._set_inaxes(self.canvas.inaxes((x, y))
                         if self.canvas.mouse_grabber is None else
                         self.canvas.mouse_grabber,
                         (x, y))

    # Splitting _set_inaxes out is useful for the axes_leave_event handler: it
    # needs to generate synthetic LocationEvents with manually-set inaxes.  In
    # that latter case, xy has already been cast to int so it can directly be
    # read from self.x, self.y; in the normal case, however, it is more
    # accurate to pass the untruncated float x, y values passed to the ctor.

    def _set_inaxes(self, inaxes, xy=None):
        self.inaxes = inaxes
        if inaxes is not None:
            try:
                self.xdata, self.ydata = inaxes.transData.inverted().transform(
                    xy if xy is not None else (self.x, self.y))
            except ValueError:
                pass


class MouseButton(IntEnum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    BACK = 8
    FORWARD = 9


class MouseEvent(LocationEvent):
    """
    A mouse event ('button_press_event', 'button_release_event', \
'scroll_event', 'motion_notify_event').

    A MouseEvent has a number of special attributes in addition to those
    defined by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    button : None or `MouseButton` or {'up', 'down'}
        The button pressed. 'up' and 'down' are used for scroll events.

        Note that LEFT and RIGHT actually refer to the "primary" and
        "secondary" buttons, i.e. if the user inverts their left and right
        buttons ("left-handed setting") then the LEFT button will be the one
        physically on the right.

        If this is unset, *name* is "scroll_event", and *step* is nonzero, then
        this will be set to "up" or "down" depending on the sign of *step*.

    buttons : None or frozenset
        For 'motion_notify_event', the mouse buttons currently being pressed
        (a set of zero or more MouseButtons);
        for other events, None.

        .. note::
           For 'motion_notify_event', this attribute is more accurate than
           the ``button`` (singular) attribute, which is obtained from the last
           'button_press_event' or 'button_release_event' that occurred within
           the canvas (and thus 1. be wrong if the last change in mouse state
           occurred when the canvas did not have focus, and 2. cannot report
           when multiple buttons are pressed).

           This attribute is not set for 'button_press_event' and
           'button_release_event' because GUI toolkits are inconsistent as to
           whether they report the button state *before* or *after* the
           press/release occurred.

        .. warning::
           On macOS, the Tk backends only report a single button even if
           multiple buttons are pressed.

    key : None or str
        The key pressed when the mouse event triggered, e.g. 'shift'.
        See `KeyEvent`.

        .. warning::
           This key is currently obtained from the last 'key_press_event' or
           'key_release_event' that occurred within the canvas.  Thus, if the
           last change of keyboard state occurred while the canvas did not have
           focus, this attribute will be wrong.  On the other hand, the
           ``modifiers`` attribute should always be correct, but it can only
           report on modifier keys.

    step : float
        The number of scroll steps (positive for 'up', negative for 'down').
        This applies only to 'scroll_event' and defaults to 0 otherwise.

    dblclick : bool
        Whether the event is a double-click. This applies only to
        'button_press_event' and is False otherwise. In particular, it's
        not used in 'button_release_event'.

    Examples
    --------
    ::

        def on_press(event):
            print('you pressed', event.button, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('button_press_event', on_press)
    """

    def __init__(self, name, canvas, x, y, button=None, key=None,
                 step=0, dblclick=False, guiEvent=None, *,
                 buttons=None, modifiers=None):
        super().__init__(
            name, canvas, x, y, guiEvent=guiEvent, modifiers=modifiers)
        if button in MouseButton.__members__.values():
            button = MouseButton(button)
        if name == "scroll_event" and button is None:
            if step > 0:
                button = "up"
            elif step < 0:
                button = "down"
        self.button = button
        if name == "motion_notify_event":
            self.buttons = frozenset(buttons if buttons is not None else [])
        else:
            # We don't support 'buttons' for button_press/release_event because
            # toolkits are inconsistent as to whether they report the state
            # before or after the event.
            if buttons:
                raise ValueError(
                    "'buttons' is only supported for 'motion_notify_event'")
            self.buttons = None
        self.key = key
        self.step = step
        self.dblclick = dblclick

    def __str__(self):
        return (f"{self.name}: "
                f"xy=({self.x}, {self.y}) xydata=({self.xdata}, {self.ydata}) "
                f"button={self.button} dblclick={self.dblclick} "
                f"inaxes={self.inaxes}")


class PickEvent(Event):
    """
    A pick event.

    This event is fired when the user picks a location on the canvas
    sufficiently close to an artist that has been made pickable with
    `.Artist.set_picker`.

    A PickEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    mouseevent : `MouseEvent`
        The mouse event that generated the pick.
    artist : `~matplotlib.artist.Artist`
        The picked artist.  Note that artists are not pickable by default
        (see `.Artist.set_picker`).
    other
        Additional attributes may be present depending on the type of the
        picked object; e.g., a `.Line2D` pick may define different extra
        attributes than a `.PatchCollection` pick.

    Examples
    --------
    Bind a function ``on_pick()`` to pick events, that prints the coordinates
    of the picked data point::

        ax.plot(np.rand(100), 'o', picker=5)  # 5 points tolerance

        def on_pick(event):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            print(f'on pick line: {xdata[ind]:.3f}, {ydata[ind]:.3f}')

        cid = fig.canvas.mpl_connect('pick_event', on_pick)
    """

    def __init__(self, name, canvas, mouseevent, artist,
                 guiEvent=None, **kwargs):
        if guiEvent is None:
            guiEvent = mouseevent.guiEvent
        super().__init__(name, canvas, guiEvent)
        self.mouseevent = mouseevent
        self.artist = artist
        self.__dict__.update(kwargs)


class KeyEvent(LocationEvent):
    """
    A key event (key press, key release).

    A KeyEvent has a number of special attributes in addition to those defined
    by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    key : None or str
        The key(s) pressed. Could be *None*, a single case sensitive Unicode
        character ("g", "G", "#", etc.), a special key ("control", "shift",
        "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g",
        "ctrl+alt+G").

    Notes
    -----
    Modifier keys will be prefixed to the pressed key and will be in the order
    "ctrl", "alt", "super". The exception to this rule is when the pressed key
    is itself a modifier key, therefore "ctrl+alt" and "alt+control" can both
    be valid key values.

    Examples
    --------
    ::

        def on_key(event):
            print('you pressed', event.key, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
    """

    def __init__(self, name, canvas, key, x=0, y=0, guiEvent=None):
        super().__init__(name, canvas, x, y, guiEvent=guiEvent)
        self.key = key


# Default callback for key events.
def _key_handler(event):
    # Dead reckoning of key.
    if event.name == "key_press_event":
        event.canvas._key = event.key
    elif event.name == "key_release_event":
        event.canvas._key = None


# Default callback for mouse events.
def _mouse_handler(event):
    # Dead-reckoning of button and key.
    if event.name == "button_press_event":
        event.canvas._button = event.button
    elif event.name == "button_release_event":
        event.canvas._button = None
    elif event.name == "motion_notify_event" and event.button is None:
        event.button = event.canvas._button
    if event.key is None:
        event.key = event.canvas._key
    # Emit axes_enter/axes_leave.
    if event.name == "motion_notify_event":
        last_ref = LocationEvent._last_axes_ref
        last_axes = last_ref() if last_ref else None
        if last_axes != event.inaxes:
            if last_axes is not None:
                # Create a synthetic LocationEvent for the axes_leave_event.
                # Its inaxes attribute needs to be manually set (because the
                # cursor is actually *out* of that Axes at that point); this is
                # done with the internal _set_inaxes method which ensures that
                # the xdata and ydata attributes are also correct.
                try:
                    canvas = last_axes.get_figure(root=True).canvas
                    leave_event = LocationEvent(
                        "axes_leave_event", canvas,
                        event.x, event.y, event.guiEvent,
                        modifiers=event.modifiers)
                    leave_event._set_inaxes(last_axes)
                    canvas.callbacks.process("axes_leave_event", leave_event)
                except Exception:
                    pass  # The last canvas may already have been torn down.
            if event.inaxes is not None:
                event.canvas.callbacks.process("axes_enter_event", event)
        LocationEvent._last_axes_ref = (
            weakref.ref(event.inaxes) if event.inaxes else None)


def _get_renderer(figure, print_method=None):
    """
    Get the renderer that would be used to save a `.Figure`.

    If you need a renderer without any active draw methods use
    renderer._draw_disabled to temporary patch them out at your call site.
    """
    # This is implemented by triggering a draw, then immediately jumping out of
    # Figure.draw() by raising an exception.

    class Done(Exception):
        pass

    def _draw(renderer): raise Done(renderer)

    with cbook._setattr_cm(figure, draw=_draw), ExitStack() as stack:
        if print_method is None:
            fmt = figure.canvas.get_default_filetype()
            # Even for a canvas' default output type, a canvas switch may be
            # needed, e.g. for FigureCanvasBase.
            print_method = stack.enter_context(
                figure.canvas._switch_canvas_and_return_print_method(fmt))
        try:
            print_method(io.BytesIO())
        except Done as exc:
            renderer, = exc.args
            return renderer
        else:
            raise RuntimeError(f"{print_method} did not call Figure.draw, so "
                               f"no renderer is available")


def _no_output_draw(figure):
    # _no_output_draw was promoted to the figure level, but
    # keep this here in case someone was calling it...
    figure.draw_without_rendering()


def _is_non_interactive_terminal_ipython(ip):
    """
    Return whether we are in a terminal IPython, but non interactive.

    When in _terminal_ IPython, ip.parent will have and `interact` attribute,
    if this attribute is False we do not setup eventloop integration as the
    user will _not_ interact with IPython. In all other case (ZMQKernel, or is
    interactive), we do.
    """
    return (hasattr(ip, 'parent')
            and (ip.parent is not None)
            and getattr(ip.parent, 'interact', None) is False)


@contextmanager
def _allow_interrupt(prepare_notifier, handle_sigint):
    """
    A context manager that allows terminating a plot by sending a SIGINT.  It
    is necessary because the running backend prevents the Python interpreter
    from running and processing signals (i.e., to raise a KeyboardInterrupt).
    To solve this, one needs to somehow wake up the interpreter and make it
    close the plot window.  We do this by using the signal.set_wakeup_fd()
    function which organizes a write of the signal number into a socketpair.
    A backend-specific function, *prepare_notifier*, arranges to listen to
    the pair's read socket while the event loop is running.  (If it returns a
    notifier object, that object is kept alive while the context manager runs.)

    If SIGINT was indeed caught, after exiting the on_signal() function the
    interpreter reacts to the signal according to the handler function which
    had been set up by a signal.signal() call; here, we arrange to call the
    backend-specific *handle_sigint* function, passing the notifier object
    as returned by prepare_notifier().  Finally, we call the old SIGINT
    handler with the same arguments that were given to our custom handler.

    We do this only if the old handler for SIGINT was not None, which means
    that a non-python handler was installed, i.e. in Julia, and not SIG_IGN
    which means we should ignore the interrupts.

    Parameters
    ----------
    prepare_notifier : Callable[[socket.socket], object]
    handle_sigint : Callable[[object], object]
    """

    old_sigint_handler = signal.getsignal(signal.SIGINT)
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield
        return

    handler_args = None
    wsock, rsock = socket.socketpair()
    wsock.setblocking(False)
    rsock.setblocking(False)
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    notifier = prepare_notifier(rsock)

    def save_args_and_handle_sigint(*args):
        nonlocal handler_args, notifier
        handler_args = args
        handle_sigint(notifier)
        notifier = None

    signal.signal(signal.SIGINT, save_args_and_handle_sigint)
    try:
        yield
    finally:
        wsock.close()
        rsock.close()
        signal.set_wakeup_fd(old_wakeup_fd)
        signal.signal(signal.SIGINT, old_sigint_handler)
        if handler_args is not None:
            old_sigint_handler(*handler_args)


class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level figure instance.
    """

    # Set to one of {"qt", "gtk3", "gtk4", "wx", "tk", "macosx"} if an
    # interactive framework is required, or None otherwise.
    required_interactive_framework = None

    # The manager class instantiated by new_manager.
    # (This is defined as a classproperty because the manager class is
    # currently defined *after* the canvas class, but one could also assign
    # ``FigureCanvasBase.manager_class = FigureManagerBase``
    # after defining both classes.)
    manager_class = _api.classproperty(lambda cls: FigureManagerBase)

    events = [
        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',
        'figure_enter_event',
        'figure_leave_event',
        'axes_enter_event',
        'axes_leave_event',
        'close_event'
    ]

    fixed_dpi = None

    filetypes = _default_filetypes

    @_api.classproperty
    def supports_blit(cls):
        """If this Canvas sub-class supports blitting."""
        return (hasattr(cls, "copy_from_bbox")
                and hasattr(cls, "restore_region"))

    def __init__(self, figure=None):
        from matplotlib.figure import Figure
        self._fix_ipython_backend2gui()
        self._is_idle_drawing = True
        self._is_saving = False
        if figure is None:
            figure = Figure()
        figure.set_canvas(self)
        self.figure = figure
        self.manager = None
        self.widgetlock = widgets.LockDraw()
        self._button = None  # the button pressed
        self._key = None  # the key pressed
        self.mouse_grabber = None  # the Axes currently grabbing mouse
        self.toolbar = None  # NavigationToolbar2 will set me
        self._is_idle_drawing = False
        # We don't want to scale up the figure DPI more than once.
        figure._original_dpi = figure.dpi
        self._device_pixel_ratio = 1
        super().__init__()  # Typically the GUI widget init (if any).

    callbacks = property(lambda self: self.figure._canvas_callbacks)
    button_pick_id = property(lambda self: self.figure._button_pick_id)
    scroll_pick_id = property(lambda self: self.figure._scroll_pick_id)

    @classmethod
    @functools.cache
    def _fix_ipython_backend2gui(cls):
        # Fix hard-coded module -> toolkit mapping in IPython (used for
        # `ipython --auto`).  This cannot be done at import time due to
        # ordering issues, so we do it when creating a canvas, and should only
        # be done once per class (hence the `cache`).

        # This function will not be needed when Python 3.12, the latest version
        # supported by IPython < 8.24, reaches end-of-life in late 2028.
        # At that time this function can be made a no-op and deprecated.
        mod_ipython = sys.modules.get("IPython")
        if mod_ipython is None or mod_ipython.version_info[:2] >= (8, 24):
            # Use of backend2gui is not needed for IPython >= 8.24 as the
            # functionality has been moved to Matplotlib.
            return

        import IPython
        ip = IPython.get_ipython()
        if not ip:
            return
        from IPython.core import pylabtools as pt
        if (not hasattr(pt, "backend2gui")
                or not hasattr(ip, "enable_matplotlib")):
            # In case we ever move the patch to IPython and remove these APIs,
            # don't break on our side.
            return
        backend2gui_rif = {
            "qt": "qt",
            "gtk3": "gtk3",
            "gtk4": "gtk4",
            "wx": "wx",
            "macosx": "osx",
        }.get(cls.required_interactive_framework)
        if backend2gui_rif:
            if _is_non_interactive_terminal_ipython(ip):
                ip.enable_gui(backend2gui_rif)

    @classmethod
    def new_manager(cls, figure, num):
        """
        Create a new figure manager for *figure*, using this canvas class.

        Notes
        -----
        This method should not be reimplemented in subclasses.  If
        custom manager creation logic is needed, please reimplement
        ``FigureManager.create_with_canvas``.
        """
        return cls.manager_class.create_with_canvas(cls, figure, num)

    @contextmanager
    def _idle_draw_cntx(self):
        self._is_idle_drawing = True
        try:
            yield
        finally:
            self._is_idle_drawing = False

    def is_saving(self):
        """
        Return whether the renderer is in the process of saving
        to a file, rather than rendering for an on-screen buffer.
        """
        return self._is_saving

    def blit(self, bbox=None):
        """Blit the canvas in bbox (default entire canvas)."""

    def inaxes(self, xy):
        """
        Return the topmost visible `~.axes.Axes` containing the point *xy*.

        Parameters
        ----------
        xy : (float, float)
            (x, y) pixel positions from left/bottom of the canvas.

        Returns
        -------
        `~matplotlib.axes.Axes` or None
            The topmost visible Axes containing the point, or None if there
            is no Axes at the point.
        """
        axes_list = [a for a in self.figure.get_axes()
                     if a.patch.contains_point(xy) and a.get_visible()]
        if axes_list:
            axes = cbook._topmost_artist(axes_list)
        else:
            axes = None

        return axes

    def grab_mouse(self, ax):
        """
        Set the child `~.axes.Axes` which is grabbing the mouse events.

        Usually called by the widgets themselves. It is an error to call this
        if the mouse is already grabbed by another Axes.
        """
        if self.mouse_grabber not in (None, ax):
            raise RuntimeError("Another Axes already grabs mouse input")
        self.mouse_grabber = ax

    def release_mouse(self, ax):
        """
        Release the mouse grab held by the `~.axes.Axes` *ax*.

        Usually called by the widgets. It is ok to call this even if *ax*
        doesn't have the mouse grab currently.
        """
        if self.mouse_grabber is ax:
            self.mouse_grabber = None

    def set_cursor(self, cursor):
        """
        Set the current cursor.

        This may have no effect if the backend does not display anything.

        If required by the backend, this method should trigger an update in
        the backend event loop after the cursor is set, as this method may be
        called e.g. before a long-running task during which the GUI is not
        updated.

        Parameters
        ----------
        cursor : `.Cursors`
            The cursor to display over the canvas. Note: some backends may
            change the cursor for the entire window.
        """

    def draw(self, *args, **kwargs):
        """
        Render the `.Figure`.

        This method must walk the artist tree, even if no output is produced,
        because it triggers deferred work that users may want to access
        before saving output to disk. For example computing limits,
        auto-limits, and tick values.
        """

    def draw_idle(self, *args, **kwargs):
        """
        Request a widget redraw once control returns to the GUI event loop.

        Even if multiple calls to `draw_idle` occur before control returns
        to the GUI event loop, the figure will only be rendered once.

        Notes
        -----
        Backends may choose to override the method and implement their own
        strategy to prevent multiple renderings.

        """
        if not self._is_idle_drawing:
            with self._idle_draw_cntx():
                self.draw(*args, **kwargs)

    @property
    def device_pixel_ratio(self):
        """
        The ratio of physical to logical pixels used for the canvas on screen.

        By default, this is 1, meaning physical and logical pixels are the same
        size. Subclasses that support High DPI screens may set this property to
        indicate that said ratio is different. All Matplotlib interaction,
        unless working directly with the canvas, remains in logical pixels.

        """
        return self._device_pixel_ratio

    def _set_device_pixel_ratio(self, ratio):
        """
        Set the ratio of physical to logical pixels used for the canvas.

        Subclasses that support High DPI screens can set this property to
        indicate that said ratio is different. The canvas itself will be
        created at the physical size, while the client side will use the
        logical size. Thus the DPI of the Figure will change to be scaled by
        this ratio. Implementations that support High DPI screens should use
        physical pixels for events so that transforms back to Axes space are
        correct.

        By default, this is 1, meaning physical and logical pixels are the same
        size.

        Parameters
        ----------
        ratio : float
            The ratio of logical to physical pixels used for the canvas.

        Returns
        -------
        bool
            Whether the ratio has changed. Backends may interpret this as a
            signal to resize the window, repaint the canvas, or change any
            other relevant properties.
        """
        if self._device_pixel_ratio == ratio:
            return False
        # In cases with mixed resolution displays, we need to be careful if the
        # device pixel ratio changes - in this case we need to resize the
        # canvas accordingly. Some backends provide events that indicate a
        # change in DPI, but those that don't will update this before drawing.
        dpi = ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)
        self._device_pixel_ratio = ratio
        return True

    def get_width_height(self, *, physical=False):
        """
        Return the figure width and height in integral points or pixels.

        When the figure is used on High DPI screens (and the backend supports
        it), the truncation to integers occurs after scaling by the device
        pixel ratio.

        Parameters
        ----------
        physical : bool, default: False
            Whether to return true physical pixels or logical pixels. Physical
            pixels may be used by backends that support HiDPI, but still
            configure the canvas using its actual size.

        Returns
        -------
        width, height : int
            The size of the figure, in points or pixels, depending on the
            backend.
        """
        return tuple(int(size / (1 if physical else self.device_pixel_ratio))
                     for size in self.figure.bbox.max)

    @classmethod
    def get_supported_filetypes(cls):
        """Return dict of savefig file formats supported by this backend."""
        return cls.filetypes

    @classmethod
    def get_supported_filetypes_grouped(cls):
        """
        Return a dict of savefig file formats supported by this backend,
        where the keys are a file type name, such as 'Joint Photographic
        Experts Group', and the values are a list of filename extensions used
        for that filetype, such as ['jpg', 'jpeg'].
        """
        groupings = {}
        for ext, name in cls.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings

    @contextmanager
    def _switch_canvas_and_return_print_method(self, fmt, backend=None):
        """
        Context manager temporarily setting the canvas for saving the figure::

            with (canvas._switch_canvas_and_return_print_method(fmt, backend)
                  as print_method):
                # ``print_method`` is a suitable ``print_{fmt}`` method, and
                # the figure's canvas is temporarily switched to the method's
                # canvas within the with... block.  ``print_method`` is also
                # wrapped to suppress extra kwargs passed by ``print_figure``.

        Parameters
        ----------
        fmt : str
            If *backend* is None, then determine a suitable canvas class for
            saving to format *fmt* -- either the current canvas class, if it
            supports *fmt*, or whatever `get_registered_canvas_class` returns;
            switch the figure canvas to that canvas class.
        backend : str or None, default: None
            If not None, switch the figure canvas to the ``FigureCanvas`` class
            of the given backend.
        """
        canvas = None
        if backend is not None:
            # Return a specific canvas class, if requested.
            from .backends.registry import backend_registry
            canvas_class = backend_registry.load_backend_module(backend).FigureCanvas
            if not hasattr(canvas_class, f"print_{fmt}"):
                raise ValueError(
                    f"The {backend!r} backend does not support {fmt} output")
            canvas = canvas_class(self.figure)
        elif hasattr(self, f"print_{fmt}"):
            # Return the current canvas if it supports the requested format.
            canvas = self
        else:
            # Return a default canvas for the requested format, if it exists.
            canvas_class = get_registered_canvas_class(fmt)
            if canvas_class is None:
                raise ValueError(
                    "Format {!r} is not supported (supported formats: {})".format(
                        fmt, ", ".join(sorted(self.get_supported_filetypes()))))
            canvas = canvas_class(self.figure)
        canvas._is_saving = self._is_saving
        meth = getattr(canvas, f"print_{fmt}")
        mod = (meth.func.__module__
               if hasattr(meth, "func")  # partialmethod, e.g. backend_wx.
               else meth.__module__)
        if mod.startswith(("matplotlib.", "mpl_toolkits.")):
            optional_kws = {  # Passed by print_figure for other renderers.
                "dpi", "facecolor", "edgecolor", "orientation",
                "bbox_inches_restore"}
            skip = optional_kws - {*inspect.signature(meth).parameters}
            print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
                *args, **{k: v for k, v in kwargs.items() if k not in skip}))
        else:  # Let third-parties do as they see fit.
            print_method = meth
        try:
            yield print_method
        finally:
            self.figure.canvas = self

    def print_figure(
            self, filename, dpi=None, facecolor=None, edgecolor=None,
            orientation='portrait', format=None, *,
            bbox_inches=None, pad_inches=None, bbox_extra_artists=None,
            backend=None, **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        Parameters
        ----------
        filename : str or path-like or file-like
            The file where the figure is saved.

        dpi : float, default: :rc:`savefig.dpi`
            The dots per inch to save the figure in.

        facecolor : :mpltype:`color` or 'auto', default: :rc:`savefig.facecolor`
            The facecolor of the figure.  If 'auto', use the current figure
            facecolor.

        edgecolor : :mpltype:`color` or 'auto', default: :rc:`savefig.edgecolor`
            The edgecolor of the figure.  If 'auto', use the current figure
            edgecolor.

        orientation : {'landscape', 'portrait'}, default: 'portrait'
            Only currently applies to PostScript printing.

        format : str, optional
            Force a specific file format. If not given, the format is inferred
            from the *filename* extension, and if that fails from
            :rc:`savefig.format`.

        bbox_inches : 'tight' or `.Bbox`, default: :rc:`savefig.bbox`
            Bounding box in inches: only the given portion of the figure is
            saved.  If 'tight', try to figure out the tight bbox of the figure.

        pad_inches : float or 'layout', default: :rc:`savefig.pad_inches`
            Amount of padding in inches around the figure when bbox_inches is
            'tight'. If 'layout' use the padding from the constrained or
            compressed layout engine; ignored if one of those engines is not in
            use.

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        backend : str, optional
            Use a non-default backend to render the file, e.g. to render a
            png file with the "cairo" backend rather than the default "agg",
            or a pdf file with the "pgf" backend rather than the default
            "pdf".  Note that the default backend is normally sufficient.  See
            :ref:`the-builtin-backends` for a list of valid backends for each
            file format.  Custom backends can be referenced as "module://...".
        """
        if format is None:
            # get format from filename, or from backend's default filetype
            if isinstance(filename, os.PathLike):
                filename = os.fspath(filename)
            if isinstance(filename, str):
                format = os.path.splitext(filename)[1][1:]
            if format is None or format == '':
                format = self.get_default_filetype()
                if isinstance(filename, str):
                    filename = filename.rstrip('.') + '.' + format
        format = format.lower()

        if dpi is None:
            dpi = rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = getattr(self.figure, '_original_dpi', self.figure.dpi)

        # Remove the figure manager, if any, to avoid resizing the GUI widget.
        with (cbook._setattr_cm(self, manager=None),
              self._switch_canvas_and_return_print_method(format, backend)
                 as print_method,
              cbook._setattr_cm(self.figure, dpi=dpi),
              cbook._setattr_cm(self.figure.canvas, _device_pixel_ratio=1),
              cbook._setattr_cm(self.figure.canvas, _is_saving=True),
              ExitStack() as stack):

            for prop in ["facecolor", "edgecolor"]:
                color = locals()[prop]
                if color is None:
                    color = rcParams[f"savefig.{prop}"]
                if not cbook._str_equal(color, "auto"):
                    stack.enter_context(self.figure._cm_set(**{prop: color}))

            if bbox_inches is None:
                bbox_inches = rcParams['savefig.bbox']

            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None or bbox_inches == "tight":
                # we need to trigger a draw before printing to make sure
                # CL works.  "tight" also needs a draw to get the right
                # locations:
                renderer = _get_renderer(
                    self.figure,
                    functools.partial(
                        print_method, orientation=orientation)
                )
                # we do this instead of `self.figure.draw_without_rendering`
                # so that we can inject the orientation
                with getattr(renderer, "_draw_disabled", nullcontext)():
                    self.figure.draw(renderer)
            if bbox_inches:
                if bbox_inches == "tight":
                    bbox_inches = self.figure.get_tightbbox(
                        renderer, bbox_extra_artists=bbox_extra_artists)
                    if (isinstance(layout_engine, ConstrainedLayoutEngine) and
                            pad_inches == "layout"):
                        h_pad = layout_engine.get()["h_pad"]
                        w_pad = layout_engine.get()["w_pad"]
                    else:
                        if pad_inches in [None, "layout"]:
                            pad_inches = rcParams['savefig.pad_inches']
                        h_pad = w_pad = pad_inches
                    bbox_inches = bbox_inches.padded(w_pad, h_pad)

                # call adjust_bbox to save only the given area
                restore_bbox = _tight_bbox.adjust_bbox(
                    self.figure, bbox_inches, self.figure.canvas.fixed_dpi)

                _bbox_inches_restore = (bbox_inches, restore_bbox)
            else:
                _bbox_inches_restore = None

            # we have already done layout above, so turn it off:
            stack.enter_context(self.figure._cm_set(layout_engine='none'))
            try:
                # _get_renderer may change the figure dpi (as vector formats
                # force the figure dpi to 72), so we need to set it again here.
                with cbook._setattr_cm(self.figure, dpi=dpi):
                    result = print_method(
                        filename,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        orientation=orientation,
                        bbox_inches_restore=_bbox_inches_restore,
                        **kwargs)
            finally:
                if bbox_inches and restore_bbox:
                    restore_bbox()

            return result

    @classmethod
    def get_default_filetype(cls):
        """
        Return the default savefig file format as specified in
        :rc:`savefig.format`.

        The returned string does not include a period. This method is
        overridden in backends that only support a single file type.
        """
        return rcParams['savefig.format']

    def get_default_filename(self):
        """
        Return a suitable default filename, including the extension.
        """
        default_basename = (
            self.manager.get_window_title()
            if self.manager is not None
            else ''
        )
        default_basename = default_basename or 'image'
        # Characters to be avoided in a NT path:
        # https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx#naming_conventions
        # plus ' '
        removed_chars = '<>:"/\\|?*\0 '
        default_basename = default_basename.translate(
            {ord(c): "_" for c in removed_chars})
        default_filetype = self.get_default_filetype()
        return f'{default_basename}.{default_filetype}'

    def mpl_connect(self, s, func):
        """
        Bind function *func* to event *s*.

        Parameters
        ----------
        s : str
            One of the following events ids:

            - 'button_press_event'
            - 'button_release_event'
            - 'draw_event'
            - 'key_press_event'
            - 'key_release_event'
            - 'motion_notify_event'
            - 'pick_event'
            - 'resize_event'
            - 'scroll_event'
            - 'figure_enter_event',
            - 'figure_leave_event',
            - 'axes_enter_event',
            - 'axes_leave_event'
            - 'close_event'.

        func : callable
            The callback function to be executed, which must have the
            signature::

                def func(event: Event) -> Any

            For the location events (button and key press/release), if the
            mouse is over the Axes, the ``inaxes`` attribute of the event will
            be set to the `~matplotlib.axes.Axes` the event occurs is over, and
            additionally, the variables ``xdata`` and ``ydata`` attributes will
            be set to the mouse location in data coordinates.  See `.KeyEvent`
            and `.MouseEvent` for more info.

            .. note::

                If func is a method, this only stores a weak reference to the
                method. Thus, the figure does not influence the lifetime of
                the associated object. Usually, you want to make sure that the
                object is kept alive throughout the lifetime of the figure by
                holding a reference to it.

        Returns
        -------
        cid
            A connection id that can be used with
            `.FigureCanvasBase.mpl_disconnect`.

        Examples
        --------
        ::

            def on_press(event):
                print('you pressed', event.button, event.xdata, event.ydata)

            cid = canvas.mpl_connect('button_press_event', on_press)
        """

        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        Disconnect the callback with id *cid*.

        Examples
        --------
        ::

            cid = canvas.mpl_connect('button_press_event', on_press)
            # ... later
            canvas.mpl_disconnect(cid)
        """
        self.callbacks.disconnect(cid)

    # Internal subclasses can override _timer_cls instead of new_timer, though
    # this is not a public API for third-party subclasses.
    _timer_cls = TimerBase

    def new_timer(self, interval=None, callbacks=None):
        """
        Create a new backend-specific subclass of `.Timer`.

        This is useful for getting periodic events through the backend's native
        event loop.  Implemented only for backends with GUIs.

        Parameters
        ----------
        interval : int
            Timer interval in milliseconds.

        callbacks : list[tuple[callable, tuple, dict]]
            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
            will be executed by the timer every *interval*.

            Callbacks which return ``False`` or ``0`` will be removed from the
            timer.

        Examples
        --------
        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])
        """
        return self._timer_cls(interval=interval, callbacks=callbacks)

    def flush_events(self):
        """
        Flush the GUI events for the figure.

        Interactive backends need to reimplement this method.
        """

    def start_event_loop(self, timeout=0):
        """
        Start a blocking event loop.

        Such an event loop is used by interactive functions, such as
        `~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for
        events.

        The event loop blocks until a callback function triggers
        `stop_event_loop`, or *timeout* is reached.

        If *timeout* is 0 or negative, never timeout.

        Only interactive backends need to reimplement this method and it relies
        on `flush_events` being properly implemented.

        Interactive backends should implement this in a more native way.
        """
        if timeout <= 0:
            timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter * timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop(self):
        """
        Stop the current blocking event loop.

        Interactive backends need to reimplement this to match
        `start_event_loop`
        """
        self._looping = False


def key_press_handler(event, canvas=None, toolbar=None):
    """
    Implement the default Matplotlib key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`.

    Parameters
    ----------
    event : `KeyEvent`
        A key press/release event.
    canvas : `FigureCanvasBase`, default: ``event.canvas``
        The backend-specific canvas instance.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas``.
    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
        The navigation cursor toolbar.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas.toolbar``.
    """
    if event.key is None:
        return
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar

    # toggle fullscreen mode (default key 'f', 'ctrl + f')
    if event.key in rcParams['keymap.fullscreen']:
        try:
            canvas.manager.full_screen_toggle()
        except AttributeError:
            pass

    # quit the figure (default key 'ctrl+w')
    if event.key in rcParams['keymap.quit']:
        Gcf.destroy_fig(canvas.figure)
    if event.key in rcParams['keymap.quit_all']:
        Gcf.destroy_all()

    if toolbar is not None:
        # home or reset mnemonic  (default key 'h', 'home' and 'r')
        if event.key in rcParams['keymap.home']:
            toolbar.home()
        # forward / backward keys to enable left handed quick navigation
        # (default key for backward: 'left', 'backspace' and 'c')
        elif event.key in rcParams['keymap.back']:
            toolbar.back()
        # (default key for forward: 'right' and 'v')
        elif event.key in rcParams['keymap.forward']:
            toolbar.forward()
        # pan mnemonic (default key 'p')
        elif event.key in rcParams['keymap.pan']:
            toolbar.pan()
            toolbar._update_cursor(event)
        # zoom mnemonic (default key 'o')
        elif event.key in rcParams['keymap.zoom']:
            toolbar.zoom()
            toolbar._update_cursor(event)
        # saving current figure (default key 's')
        elif event.key in rcParams['keymap.save']:
            toolbar.save_figure()

    if event.inaxes is None:
        return

    # these bindings require the mouse to be over an Axes to trigger
    def _get_uniform_gridstate(ticks):
        # Return True/False if all grid lines are on or off, None if they are
        # not all in the same state.
        return (True if all(tick.gridline.get_visible() for tick in ticks) else
                False if not any(tick.gridline.get_visible() for tick in ticks) else
                None)

    ax = event.inaxes
    # toggle major grids in current Axes (default key 'g')
    # Both here and below (for 'G'), we do nothing if *any* grid (major or
    # minor, x or y) is not in a uniform state, to avoid messing up user
    # customization.
    if (event.key in rcParams['keymap.grid']
            # Exclude minor grids not in a uniform state.
            and None not in [_get_uniform_gridstate(ax.xaxis.minorTicks),
                             _get_uniform_gridstate(ax.yaxis.minorTicks)]):
        x_state = _get_uniform_gridstate(ax.xaxis.majorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.majorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # Exclude major grids not in a uniform state.
            pass
        else:
            # If turning major grids off, also turn minor grids off.
            ax.grid(x_state, which="major" if x_state else "both", axis="x")
            ax.grid(y_state, which="major" if y_state else "both", axis="y")
            canvas.draw_idle()
    # toggle major and minor grids in current Axes (default key 'G')
    if (event.key in rcParams['keymap.grid_minor']
            # Exclude major grids not in a uniform state.
            and None not in [_get_uniform_gridstate(ax.xaxis.majorTicks),
                             _get_uniform_gridstate(ax.yaxis.majorTicks)]):
        x_state = _get_uniform_gridstate(ax.xaxis.minorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.minorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # Exclude minor grids not in a uniform state.
            pass
        else:
            ax.grid(x_state, which="both", axis="x")
            ax.grid(y_state, which="both", axis="y")
            canvas.draw_idle()
    # toggle scaling of y-axes between 'log and 'linear' (default key 'l')
    elif event.key in rcParams['keymap.yscale']:
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.get_figure(root=True).canvas.draw_idle()
        elif scale == 'linear':
            try:
                ax.set_yscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_yscale('linear')
            ax.get_figure(root=True).canvas.draw_idle()
    # toggle scaling of x-axes between 'log and 'linear' (default key 'k')
    elif event.key in rcParams['keymap.xscale']:
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.get_figure(root=True).canvas.draw_idle()
        elif scalex == 'linear':
            try:
                ax.set_xscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_xscale('linear')
            ax.get_figure(root=True).canvas.draw_idle()


def button_press_handler(event, canvas=None, toolbar=None):
    """
    The default Matplotlib button actions for extra mouse buttons.

    Parameters are as for `key_press_handler`, except that *event* is a
    `MouseEvent`.
    """
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    if toolbar is not None:
        button_name = str(MouseButton(event.button))
        if button_name in rcParams['keymap.back']:
            toolbar.back()
        elif button_name in rcParams['keymap.forward']:
            toolbar.forward()


class NonGuiException(Exception):
    """Raised when trying show a figure in a non-GUI backend."""
    pass


class FigureManagerBase:
    """
    A backend-independent abstraction of a figure container and controller.

    The figure manager is used by pyplot to interact with the window in a
    backend-independent way. It's an adapter for the real (GUI) framework that
    represents the visual figure on screen.

    The figure manager is connected to a specific canvas instance, which in turn
    is connected to a specific figure instance. To access a figure manager for
    a given figure in user code, you typically use ``fig.canvas.manager``.

    GUI backends derive from this class to translate common operations such
    as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
    support these operations and can just use the base class.

    This following basic operations are accessible:

    **Window operations**

    - `~.FigureManagerBase.show`
    - `~.FigureManagerBase.destroy`
    - `~.FigureManagerBase.full_screen_toggle`
    - `~.FigureManagerBase.resize`
    - `~.FigureManagerBase.get_window_title`
    - `~.FigureManagerBase.set_window_title`

    **Key and mouse button press handling**

    The figure manager sets up default key and mouse button press handling by
    hooking up the `.key_press_handler` to the matplotlib event system. This
    ensures the same shortcuts and mouse actions across backends.

    **Other operations**

    Subclasses will have additional attributes and functions to access
    additional functionality. This is of course backend-specific. For example,
    most GUI backends have ``window`` and ``toolbar`` attributes that give
    access to the native GUI widgets of the respective framework.

    Attributes
    ----------
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance.

    num : int or str
        The figure number.

    key_press_handler_id : int
        The default key handler cid, when using the toolmanager.
        To disable the default key press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.key_press_handler_id)

    button_press_handler_id : int
        The default mouse button handler cid, when using the toolmanager.
        To disable the default button press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.button_press_handler_id)
    """

    _toolbar2_class = None
    _toolmanager_toolbar_class = None

    def __init__(self, canvas, num):
        self.canvas = canvas
        canvas.manager = self  # store a pointer to parent
        self.num = num
        self.set_window_title(f"Figure {num:d}")

        self.key_press_handler_id = None
        self.button_press_handler_id = None
        if rcParams['toolbar'] != 'toolmanager':
            self.key_press_handler_id = self.canvas.mpl_connect(
                'key_press_event', key_press_handler)
            self.button_press_handler_id = self.canvas.mpl_connect(
                'button_press_event', button_press_handler)

        self.toolmanager = (ToolManager(canvas.figure)
                            if mpl.rcParams['toolbar'] == 'toolmanager'
                            else None)
        if (mpl.rcParams["toolbar"] == "toolbar2"
                and self._toolbar2_class):
            self.toolbar = self._toolbar2_class(self.canvas)
        elif (mpl.rcParams["toolbar"] == "toolmanager"
                and self._toolmanager_toolbar_class):
            self.toolbar = self._toolmanager_toolbar_class(self.toolmanager)
        else:
            self.toolbar = None

        if self.toolmanager:
            tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                tools.add_tools_to_container(self.toolbar)

        @self.canvas.figure.add_axobserver
        def notify_axes_change(fig):
            # Called whenever the current Axes is changed.
            if self.toolmanager is None and self.toolbar is not None:
                self.toolbar.update()

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a manager for a given *figure* using a specific *canvas_class*.

        Backends should override this method if they have specific needs for
        setting up the canvas or the manager.
        """
        return cls(canvas_class(figure), num)

    @classmethod
    def start_main_loop(cls):
        """
        Start the main event loop.

        This method is called by `.FigureManagerBase.pyplot_show`, which is the
        implementation of `.pyplot.show`.  To customize the behavior of
        `.pyplot.show`, interactive backends should usually override
        `~.FigureManagerBase.start_main_loop`; if more customized logic is
        necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.
        """

    @classmethod
    def pyplot_show(cls, *, block=None):
        """
        Show all figures.  This method is the implementation of `.pyplot.show`.

        To customize the behavior of `.pyplot.show`, interactive backends
        should usually override `~.FigureManagerBase.start_main_loop`; if more
        customized logic is necessary, `~.FigureManagerBase.pyplot_show` can
        also be overridden.

        Parameters
        ----------
        block : bool, optional
            Whether to block by calling ``start_main_loop``.  The default,
            None, means to block if we are neither in IPython's ``%pylab`` mode
            nor in ``interactive`` mode.
        """
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return
        for manager in managers:
            try:
                manager.show()  # Emits a warning for non-interactive backend.
            except NonGuiException as exc:
                _api.warn_external(str(exc))
        if block is None:
            # Hack: Are we in IPython's %pylab mode?  In pylab mode, IPython
            # (>= 0.10) tacks a _needmain attribute onto pyplot.show (always
            # set to False).
            pyplot_show = getattr(sys.modules.get("matplotlib.pyplot"), "show", None)
            ipython_pylab = hasattr(pyplot_show, "_needmain")
            block = not ipython_pylab and not is_interactive()
        if block:
            cls.start_main_loop()

    def show(self):
        """
        For GUI backends, show the figure window and redraw.
        For non-GUI backends, raise an exception, unless running headless (i.e.
        on Linux with an unset DISPLAY); this exception is converted to a
        warning in `.Figure.show`.
        """
        # This should be overridden in GUI backends.
        if sys.platform == "linux" and not os.environ.get("DISPLAY"):
            # We cannot check _get_running_interactive_framework() ==
            # "headless" because that would also suppress the warning when
            # $DISPLAY exists but is invalid, which is more likely an error and
            # thus warrants a warning.
            return
        raise NonGuiException(
            f"{type(self.canvas).__name__} is non-interactive, and thus cannot be "
            f"shown")

    def destroy(self):
        pass

    def full_screen_toggle(self):
        pass

    def resize(self, w, h):
        """For GUI backends, resize the window (in physical pixels)."""

    def get_window_title(self):
        """Return the title text of the window containing the figure."""
        return self._window_title

    def set_window_title(self, title):
        """
        Set the title text of the window containing the figure.

        Examples
        --------
        >>> fig = plt.figure()
        >>> fig.canvas.manager.set_window_title('My figure')
        """
        # This attribute is not defined in __init__ (but __init__ calls this
        # setter), as derived classes (real GUI managers) will store this
        # information directly on the widget; only the base (non-GUI) manager
        # class needs a specific attribute for it (so that filename escaping
        # can be checked in the test suite).
        self._window_title = title


cursors = tools.cursors


class _Mode(str, Enum):
    NONE = ""
    PAN = "pan/zoom"
    ZOOM = "zoom rect"

    def __str__(self):
        return self.value

    @property
    def _navigate_mode(self):
        return self.name if self is not _Mode.NONE else None


class NavigationToolbar2:
    """
    Base class for the navigation cursor, version 2.

    Backends must implement a canvas that handles connections for
    'button_press_event' and 'button_release_event'.  See
    :meth:`FigureCanvasBase.mpl_connect` for more information.

    They must also define

    :meth:`save_figure`
        Save the current figure.

    :meth:`draw_rubberband` (optional)
        Draw the zoom to rect "rubberband" rectangle.

    :meth:`set_message` (optional)
        Display message.

    :meth:`set_history_buttons` (optional)
        You can change the history back / forward buttons to indicate disabled / enabled
        state.

    and override ``__init__`` to set up the toolbar -- without forgetting to
    call the base-class init.  Typically, ``__init__`` needs to set up toolbar
    buttons connected to the `home`, `back`, `forward`, `pan`, `zoom`, and
    `save_figure` methods and using standard icons in the "images" subdirectory
    of the data path.

    That's it, we'll do the rest!
    """

    # list of toolitems to add to the toolbar, format is:
    # (
    #   text, # the text of the button (often not visible to users)
    #   tooltip_text, # the tooltip shown on hover (where possible)
    #   image_file, # name of the image for the button (without the extension)
    #   name_of_method, # name of the method in NavigationToolbar2 to call
    # )
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan',
         'Left button pans, Right button zooms\n'
         'x/y fixes axis, CTRL fixes aspect',
         'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
      )

    UNKNOWN_SAVED_STATUS = object()

    def __init__(self, canvas):
        self.canvas = canvas
        canvas.toolbar = self
        self._nav_stack = cbook._Stack()
        # This cursor will be set after the initial draw.
        self._last_cursor = tools.Cursors.POINTER

        self._id_press = self.canvas.mpl_connect(
            'button_press_event', self._zoom_pan_handler)
        self._id_release = self.canvas.mpl_connect(
            'button_release_event', self._zoom_pan_handler)
        self._id_drag = self.canvas.mpl_connect(
            'motion_notify_event', self.mouse_move)
        self._pan_info = None
        self._zoom_info = None

        self.mode = _Mode.NONE  # a mode string for the status bar
        self.set_history_buttons()

    def set_message(self, s):
        """Display a message on toolbar or in status bar."""

    def draw_rubberband(self, event, x0, y0, x1, y1):
        """
        Draw a rectangle rubberband to indicate zoom limits.

        Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.
        """

    def remove_rubberband(self):
        """Remove the rubberband."""

    def home(self, *args):
        """
        Restore the original view.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.home()
        self.set_history_buttons()
        self._update_view()

    def back(self, *args):
        """
        Move back up the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.back()
        self.set_history_buttons()
        self._update_view()

    def forward(self, *args):
        """
        Move forward in the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        self._nav_stack.forward()
        self.set_history_buttons()
        self._update_view()

    def _update_cursor(self, event):
        """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
        if self.mode and event.inaxes and event.inaxes.get_navigate():
            if (self.mode == _Mode.ZOOM
                    and self._last_cursor != tools.Cursors.SELECT_REGION):
                self.canvas.set_cursor(tools.Cursors.SELECT_REGION)
                self._last_cursor = tools.Cursors.SELECT_REGION
            elif (self.mode == _Mode.PAN
                  and self._last_cursor != tools.Cursors.MOVE):
                self.canvas.set_cursor(tools.Cursors.MOVE)
                self._last_cursor = tools.Cursors.MOVE
        elif self._last_cursor != tools.Cursors.POINTER:
            self.canvas.set_cursor(tools.Cursors.POINTER)
            self._last_cursor = tools.Cursors.POINTER

    @contextmanager
    def _wait_cursor_for_draw_cm(self):
        """
        Set the cursor to a wait cursor when drawing the canvas.

        In order to avoid constantly changing the cursor when the canvas
        changes frequently, do nothing if this context was triggered during the
        last second.  (Optimally we'd prefer only setting the wait cursor if
        the *current* draw takes too long, but the current draw blocks the GUI
        thread).
        """
        self._draw_time, last_draw_time = (
            time.time(), getattr(self, "_draw_time", -np.inf))
        if self._draw_time - last_draw_time > 1:
            try:
                self.canvas.set_cursor(tools.Cursors.WAIT)
                yield
            finally:
                self.canvas.set_cursor(self._last_cursor)
        else:
            yield

    @staticmethod
    def _mouse_event_to_message(event):
        if event.inaxes and event.inaxes.get_navigate():
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                s = s.rstrip()
                artists = [a for a in event.inaxes._mouseover_set
                           if a.contains(event)[0] and a.get_visible()]
                if artists:
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            data_str = a.format_cursor_data(data).rstrip()
                            if data_str:
                                s = s + '\n' + data_str
                return s
        return ""

    def mouse_move(self, event):
        self._update_cursor(event)
        self.set_message(self._mouse_event_to_message(event))

    def _zoom_pan_handler(self, event):
        if self.mode == _Mode.PAN:
            if event.name == "button_press_event":
                self.press_pan(event)
            elif event.name == "button_release_event":
                self.release_pan(event)
        if self.mode == _Mode.ZOOM:
            if event.name == "button_press_event":
                self.press_zoom(event)
            elif event.name == "button_release_event":
                self.release_zoom(event)

    def _start_event_axes_interaction(self, event, *, method):

        def _ax_filter(ax):
            return (ax.in_axes(event) and
                    ax.get_navigate() and
                    getattr(ax, f"can_{method}")()
                    )

        def _capture_events(ax):
            f = ax.get_forward_navigation_events()
            if f == "auto":  # (capture = patch visibility)
                f = not ax.patch.get_visible()
            return not f

        # get all relevant axes for the event
        axes = list(filter(_ax_filter, self.canvas.figure.get_axes()))

        if len(axes) == 0:
            return []

        if self._nav_stack() is None:
            self.push_current()   # Set the home button to this view.

        # group axes by zorder (reverse to trigger later axes first)
        grps = dict()
        for ax in reversed(axes):
            grps.setdefault(ax.get_zorder(), []).append(ax)

        axes_to_trigger = []
        # go through zorders in reverse until we hit a capturing axes
        for zorder in sorted(grps, reverse=True):
            for ax in grps[zorder]:
                axes_to_trigger.append(ax)
                # NOTE: shared axes are automatically triggered, but twin-axes not!
                axes_to_trigger.extend(ax._twinned_axes.get_siblings(ax))

                if _capture_events(ax):
                    break  # break if we hit a capturing axes
            else:
                # If the inner loop finished without an explicit break,
                # (e.g. no capturing axes was found) continue the
                # outer loop to the next zorder.
                continue

            # If the inner loop was terminated with an explicit break,
            # terminate the outer loop as well.
            break

        # avoid duplicated triggers (but keep order of list)
        axes_to_trigger = list(dict.fromkeys(axes_to_trigger))

        return axes_to_trigger

    def pan(self, *args):
        """
        Toggle the pan/zoom tool.

        Pan with left button, zoom with right.
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message("pan unavailable")
            return
        if self.mode == _Mode.PAN:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.PAN
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

    _PanInfo = namedtuple("_PanInfo", "button axes cid")

    def press_pan(self, event):
        """Callback for mouse button press in pan/zoom mode."""
        if (event.button not in [MouseButton.LEFT, MouseButton.RIGHT]
                or event.x is None or event.y is None):
            return

        axes = self._start_event_axes_interaction(event, method="pan")
        if not axes:
            return

        # call "ax.start_pan(..)" on all relevant axes of an event
        for ax in axes:
            ax.start_pan(event.x, event.y, event.button)

        self.canvas.mpl_disconnect(self._id_drag)
        id_drag = self.canvas.mpl_connect("motion_notify_event", self.drag_pan)

        self._pan_info = self._PanInfo(
            button=event.button, axes=axes, cid=id_drag)

    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        for ax in self._pan_info.axes:
            # Using the recorded button at the press is safer than the current
            # button, as multiple buttons can get pressed during motion.
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        self.canvas.draw_idle()

    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        if self._pan_info is None:
            return
        self.canvas.mpl_disconnect(self._pan_info.cid)
        self._id_drag = self.canvas.mpl_connect(
            'motion_notify_event', self.mouse_move)
        for ax in self._pan_info.axes:
            ax.end_pan()
        self.canvas.draw_idle()
        self._pan_info = None
        self.push_current()

    def zoom(self, *args):
        if not self.canvas.widgetlock.available(self):
            self.set_message("zoom unavailable")
            return
        """Toggle zoom to rect mode."""
        if self.mode == _Mode.ZOOM:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.ZOOM
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

    _ZoomInfo = namedtuple("_ZoomInfo", "direction start_xy axes cid cbar")

    def press_zoom(self, event):
        """Callback for mouse button press in zoom to rect mode."""
        if (event.button not in [MouseButton.LEFT, MouseButton.RIGHT]
                or event.x is None or event.y is None):
            return

        axes = self._start_event_axes_interaction(event, method="zoom")
        if not axes:
            return

        id_zoom = self.canvas.mpl_connect(
            "motion_notify_event", self.drag_zoom)

        # A colorbar is one-dimensional, so we extend the zoom rectangle out
        # to the edge of the Axes bbox in the other dimension. To do that we
        # store the orientation of the colorbar for later.
        parent_ax = axes[0]
        if hasattr(parent_ax, "_colorbar"):
            cbar = parent_ax._colorbar.orientation
        else:
            cbar = None

        self._zoom_info = self._ZoomInfo(
            direction="in" if event.button == 1 else "out",
            start_xy=(event.x, event.y), axes=axes, cid=id_zoom, cbar=cbar)

    def drag_zoom(self, event):
        """Callback for dragging in zoom mode."""
        start_xy = self._zoom_info.start_xy
        ax = self._zoom_info.axes[0]
        (x1, y1), (x2, y2) = np.clip(
            [start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
        key = event.key
        # Force the key on colorbars to extend the short-axis bbox
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        if key == "x":
            y1, y2 = ax.bbox.intervaly
        elif key == "y":
            x1, x2 = ax.bbox.intervalx

        self.draw_rubberband(event, x1, y1, x2, y2)

    def release_zoom(self, event):
        """Callback for mouse button release in zoom to rect mode."""
        if self._zoom_info is None:
            return

        # We don't check the event button here, so that zooms can be cancelled
        # by (pressing and) releasing another mouse button.
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()

        start_x, start_y = self._zoom_info.start_xy
        key = event.key
        # Force the key on colorbars to ignore the zoom-cancel on the
        # short-axis side
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        # Ignore single clicks: 5 pixels is a threshold that allows the user to
        # "cancel" a zoom action by zooming by less than 5 pixels.
        if ((abs(event.x - start_x) < 5 and key != "y") or
                (abs(event.y - start_y) < 5 and key != "x")):
            self.canvas.draw_idle()
            self._zoom_info = None
            return

        for i, ax in enumerate(self._zoom_info.axes):
            # Detect whether this Axes is twinned with an earlier Axes in the
            # list of zoomed Axes, to avoid double zooming.
            twinx = any(ax.get_shared_x_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            twiny = any(ax.get_shared_y_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            ax._set_view_from_bbox(
                (start_x, start_y, event.x, event.y),
                self._zoom_info.direction, key, twinx, twiny)

        self.canvas.draw_idle()
        self._zoom_info = None
        self.push_current()

    def push_current(self):
        """Push the current view limits and position onto the stack."""
        self._nav_stack.push(
            WeakKeyDictionary(
                {ax: (ax._get_view(),
                      # Store both the original and modified positions.
                      (ax.get_position(True).frozen(),
                       ax.get_position().frozen()))
                 for ax in self.canvas.figure.axes}))
        self.set_history_buttons()

    def _update_view(self):
        """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
        nav_info = self._nav_stack()
        if nav_info is None:
            return
        # Retrieve all items at once to avoid any risk of GC deleting an Axes
        # while in the middle of the loop below.
        items = list(nav_info.items())
        for ax, (view, (pos_orig, pos_active)) in items:
            ax._set_view(view)
            # Restore both the original and modified positions
            ax._set_position(pos_orig, 'original')
            ax._set_position(pos_active, 'active')
        self.canvas.draw_idle()

    def configure_subplots(self, *args):
        if hasattr(self, "subplot_tool"):
            self.subplot_tool.figure.canvas.manager.show()
            return
        # This import needs to happen here due to circular imports.
        from matplotlib.figure import Figure
        with mpl.rc_context({"toolbar": "none"}):  # No navbar for the toolfig.
            manager = type(self.canvas).new_manager(Figure(figsize=(6, 3)), -1)
        manager.set_window_title("Subplot configuration tool")
        tool_fig = manager.canvas.figure
        tool_fig.subplots_adjust(top=0.9)
        self.subplot_tool = widgets.SubplotTool(self.canvas.figure, tool_fig)
        cid = self.canvas.mpl_connect(
            "close_event", lambda e: manager.destroy())

        def on_tool_fig_close(e):
            self.canvas.mpl_disconnect(cid)
            del self.subplot_tool

        tool_fig.canvas.mpl_connect("close_event", on_tool_fig_close)
        manager.show()
        return self.subplot_tool

    def save_figure(self, *args):
        """
        Save the current figure.

        Backend implementations may choose to return
        the absolute path of the saved file, if any, as
        a string.

        If no file is created then `None` is returned.

        If the backend does not implement this functionality
        then `NavigationToolbar2.UNKNOWN_SAVED_STATUS` is returned.

        Returns
        -------
        str or `NavigationToolbar2.UNKNOWN_SAVED_STATUS` or `None`
            The filepath of the saved figure.
            Returns `None` if figure is not saved.
            Returns `NavigationToolbar2.UNKNOWN_SAVED_STATUS` when
            the backend does not provide the information.
        """
        raise NotImplementedError

    def update(self):
        """Reset the Axes stack."""
        self._nav_stack.clear()
        self.set_history_buttons()

    def set_history_buttons(self):
        """Enable or disable the back/forward button."""


class ToolContainerBase:
    """
    Base class for all tool containers, e.g. toolbars.

    Attributes
    ----------
    toolmanager : `.ToolManager`
        The tools with which this `ToolContainer` wants to communicate.
    """

    _icon_extension = '.png'
    """
    Toolcontainer button icon image format extension

    **String**: Image extension
    """

    def __init__(self, toolmanager):
        self.toolmanager = toolmanager
        toolmanager.toolmanager_connect(
            'tool_message_event',
            lambda event: self.set_message(event.message))
        toolmanager.toolmanager_connect(
            'tool_removed_event',
            lambda event: self.remove_toolitem(event.tool.name))

    def _tool_toggled_cbk(self, event):
        """
        Capture the 'tool_trigger_[name]'

        This only gets used for toggled tools.
        """
        self.toggle_toolitem(event.tool.name, event.tool.toggled)

    def add_tool(self, tool, group, position=-1):
        """
        Add a tool to this container.

        Parameters
        ----------
        tool : tool_like
            The tool to add, see `.ToolManager.get_tool`.
        group : str
            The name of the group to add this tool to.
        position : int, default: -1
            The position within the group to place this tool.
        """
        tool = self.toolmanager.get_tool(tool)
        image = self._get_image_filename(tool)
        toggle = getattr(tool, 'toggled', None) is not None
        self.add_toolitem(tool.name, group, position,
                          image, tool.description, toggle)
        if toggle:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name,
                                                 self._tool_toggled_cbk)
            # If initially toggled
            if tool.toggled:
                self.toggle_toolitem(tool.name, True)

    def _get_image_filename(self, tool):
        """Resolve a tool icon's filename."""
        if not tool.image:
            return None
        if os.path.isabs(tool.image):
            filename = tool.image
        else:
            if "image" in getattr(tool, "__dict__", {}):
                raise ValueError("If 'tool.image' is an instance variable, "
                                 "it must be an absolute path")
            for cls in type(tool).__mro__:
                if "image" in vars(cls):
                    try:
                        src = inspect.getfile(cls)
                        break
                    except (OSError, TypeError):
                        raise ValueError("Failed to locate source file "
                                         "where 'tool.image' is defined") from None
            else:
                raise ValueError("Failed to find parent class defining 'tool.image'")
            filename = str(pathlib.Path(src).parent / tool.image)
        for filename in [filename, filename + self._icon_extension]:
            if os.path.isfile(filename):
                return os.path.abspath(filename)
        for fname in [  # Fallback; once deprecation elapses.
            tool.image,
            tool.image + self._icon_extension,
            cbook._get_data_path("images", tool.image),
            cbook._get_data_path("images", tool.image + self._icon_extension),
        ]:
            if os.path.isfile(fname):
                _api.warn_deprecated(
                    "3.9", message=f"Loading icon {tool.image!r} from the current "
                    "directory or from Matplotlib's image directory.  This behavior "
                    "is deprecated since %(since)s and will be removed in %(removal)s; "
                    "Tool.image should be set to a path relative to the Tool's source "
                    "file, or to an absolute path.")
                return os.path.abspath(fname)

    def trigger_tool(self, name):
        """
        Trigger the tool.

        Parameters
        ----------
        name : str
            Name (id) of the tool triggered from within the container.
        """
        self.toolmanager.trigger_tool(name, sender=self)

    def add_toolitem(self, name, group, position, image, description, toggle):
        """
        A hook to add a toolitem to the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to add an element to the toolbar.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolContainerBase.add_tool`.

        The callback associated with the button click event
        must be *exactly* ``self.trigger_tool(name)``.

        Parameters
        ----------
        name : str
            Name of the tool to add, this gets used as the tool's ID and as the
            default label of the buttons.
        group : str
            Name of the group that this tool belongs to.
        position : int
            Position of the tool within its group, if -1 it goes at the end.
        image : str
            Filename of the image for the button or `None`.
        description : str
            Description of the tool, used for the tooltips.
        toggle : bool
            * `True` : The button is a toggle (change the pressed/unpressed
              state between consecutive clicks).
            * `False` : The button is a normal button (returns to unpressed
              state after release).
        """
        raise NotImplementedError

    def toggle_toolitem(self, name, toggled):
        """
        A hook to toggle a toolitem without firing an event.

        This hook must be implemented in each backend and contains the
        backend-specific code to silently toggle a toolbar element.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.trigger_tool` or `.ToolContainerBase.trigger_tool`
            (which are equivalent).

        Parameters
        ----------
        name : str
            Id of the tool to toggle.
        toggled : bool
            Whether to set this tool as toggled or not.
        """
        raise NotImplementedError

    def remove_toolitem(self, name):
        """
        A hook to remove a toolitem from the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to remove an element from the toolbar; it is
        called when `.ToolManager` emits a ``tool_removed_event``.

        Because some tools are present only on the `.ToolManager` but not on
        the `ToolContainer`, this method must be a no-op when called on a tool
        absent from the container.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.remove_tool`.

        Parameters
        ----------
        name : str
            Name of the tool to remove.
        """
        raise NotImplementedError

    def set_message(self, s):
        """
        Display a message on the toolbar.

        Parameters
        ----------
        s : str
            Message text.
        """
        raise NotImplementedError


class _Backend:
    # A backend can be defined by using the following pattern:
    #
    # @_Backend.export
    # class FooBackend(_Backend):
    #     # override the attributes and methods documented below.

    # `backend_version` may be overridden by the subclass.
    backend_version = "unknown"

    # The `FigureCanvas` class must be defined.
    FigureCanvas = None

    # For interactive backends, the `FigureManager` class must be overridden.
    FigureManager = FigureManagerBase

    # For interactive backends, `mainloop` should be a function taking no
    # argument and starting the backend main loop.  It should be left as None
    # for non-interactive backends.
    mainloop = None

    # The following methods will be automatically defined and exported, but
    # can be overridden.

    @classmethod
    def new_figure_manager(cls, num, *args, **kwargs):
        """Create a new figure manager instance."""
        # This import needs to happen here due to circular imports.
        from matplotlib.figure import Figure
        fig_cls = kwargs.pop('FigureClass', Figure)
        fig = fig_cls(*args, **kwargs)
        return cls.new_figure_manager_given_figure(num, fig)

    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        """Create a new figure manager instance for the given figure."""
        return cls.FigureCanvas.new_manager(figure, num)

    @classmethod
    def draw_if_interactive(cls):
        manager_class = cls.FigureCanvas.manager_class
        # Interactive backends reimplement start_main_loop or pyplot_show.
        backend_is_interactive = (
            manager_class.start_main_loop != FigureManagerBase.start_main_loop
            or manager_class.pyplot_show != FigureManagerBase.pyplot_show)
        if backend_is_interactive and is_interactive():
            manager = Gcf.get_active()
            if manager:
                manager.canvas.draw_idle()

    @classmethod
    def show(cls, *, block=None):
        """
        Show all figures.

        `show` blocks by calling `mainloop` if *block* is ``True``, or if it is
        ``None`` and we are not in `interactive` mode and if IPython's
        ``%matplotlib`` integration has not been activated.
        """
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return
        for manager in managers:
            try:
                manager.show()  # Emits a warning for non-interactive backend.
            except NonGuiException as exc:
                _api.warn_external(str(exc))
        if cls.mainloop is None:
            return
        if block is None:
            # Hack: Is IPython's %matplotlib integration activated?  If so,
            # IPython's activate_matplotlib (>= 0.10) tacks a _needmain
            # attribute onto pyplot.show (always set to False).
            pyplot_show = getattr(sys.modules.get("matplotlib.pyplot"), "show", None)
            ipython_pylab = hasattr(pyplot_show, "_needmain")
            block = not ipython_pylab and not is_interactive()
        if block:
            cls.mainloop()

    # This method is the one actually exporting the required methods.

    @staticmethod
    def export(cls):
        for name in [
                "backend_version",
                "FigureCanvas",
                "FigureManager",
                "new_figure_manager",
                "new_figure_manager_given_figure",
                "draw_if_interactive",
                "show",
        ]:
            setattr(sys.modules[cls.__module__], name, getattr(cls, name))

        # For back-compatibility, generate a shim `Show` class.

        class Show(ShowBase):
            def mainloop(self):
                return cls.mainloop()

        setattr(sys.modules[cls.__module__], "Show", Show)
        return cls


class ShowBase(_Backend):
    """
    Simple base class to generate a ``show()`` function in backends.

    Subclass must override ``mainloop()`` method.
    """

    def __call__(self, block=None):
        return self.show(block=block)
<|endoftext|>r"""
Container classes for `.Artist`\s.

`OffsetBox`
    The base of all container artists defined in this module.

`AnchoredOffsetbox`, `AnchoredText`
    Anchor and align an arbitrary `.Artist` or a text relative to the parent
    axes or a specific anchor point.

`DrawingArea`
    A container with fixed width and height. Children have a fixed position
    inside the container and may be clipped.

`HPacker`, `VPacker`
    Containers for layouting their children vertically or horizontally.

`PaddedBox`
    A container to add a padding around an `.Artist`.

`TextArea`
    Contains a single `.Text` instance.
"""

import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, bbox_artist as mbbox_artist)
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox


DEBUG = False


def _compat_get_offset(meth):
    """
    Decorator for the get_offset method of OffsetBox and subclasses, that
    allows supporting both the new signature (self, bbox, renderer) and the old
    signature (self, width, height, xdescent, ydescent, renderer).
    """
    sigs = [lambda self, width, height, xdescent, ydescent, renderer: locals(),
            lambda self, bbox, renderer: locals()]

    @functools.wraps(meth)
    def get_offset(self, *args, **kwargs):
        params = _api.select_matching_signature(sigs, self, *args, **kwargs)
        bbox = (params["bbox"] if "bbox" in params else
                Bbox.from_bounds(-params["xdescent"], -params["ydescent"],
                                 params["width"], params["height"]))
        return meth(params["self"], bbox, params["renderer"])
    return get_offset


# for debugging use
def _bbox_artist(*args, **kwargs):
    if DEBUG:
        mbbox_artist(*args, **kwargs)


def _get_packed_offsets(widths, total, sep, mode="fixed"):
    r"""
    Pack boxes specified by their *widths*.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout, but the function works equally for a vertical layout.

    There are three packing *mode*\s:

    - 'fixed': The elements are packed tight to the left with a spacing of
      *sep* in between. If *total* is *None* the returned total will be the
      right edge of the last box. A non-*None* total will be passed unchecked
      to the output. In particular this means that right edge of the last
      box may be further to the right than the returned total.

    - 'expand': Distribute the boxes with equal spacing so that the left edge
      of the first box is at 0, and the right edge of the last box is at
      *total*. The parameter *sep* is ignored in this mode. A total of *None*
      is accepted and considered equal to 1. The total is returned unchanged
      (except for the conversion *None* to 1). If the total is smaller than
      the sum of the widths, the laid out boxes will overlap.

    - 'equal': If *total* is given, the total space is divided in N equal
      ranges and each box is left-aligned within its subspace.
      Otherwise (*total* is *None*), *sep* must be provided and each box is
      left-aligned in its subspace of width ``(max(widths) + sep)``. The
      total width is then calculated to be ``N * (max(widths) + sep)``.

    Parameters
    ----------
    widths : list of float
        Widths of boxes to be packed.
    total : float or None
        Intended total length. *None* if not used.
    sep : float or None
        Spacing between boxes.
    mode : {'fixed', 'expand', 'equal'}
        The packing mode.

    Returns
    -------
    total : float
        The total width needed to accommodate the laid out boxes.
    offsets : array of float
        The left offsets of the boxes.
    """
    _api.check_in_list(["fixed", "expand", "equal"], mode=mode)

    if mode == "fixed":
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        if total is None:
            total = offsets_[-1] - sep
        return total, offsets

    elif mode == "expand":
        # This is a bit of a hack to avoid a TypeError when *total*
        # is None and used in conjugation with tight layout.
        if total is None:
            total = 1
        if len(widths) > 1:
            sep = (total - sum(widths)) / (len(widths) - 1)
        else:
            sep = 0
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        return total, offsets

    elif mode == "equal":
        maxh = max(widths)
        if total is None:
            if sep is None:
                raise ValueError("total and sep cannot both be None when "
                                 "using layout mode 'equal'")
            total = (maxh + sep) * len(widths)
        else:
            sep = total / len(widths) - maxh
        offsets = (maxh + sep) * np.arange(len(widths))
        return total, offsets


def _get_aligned_offsets(yspans, height, align="baseline"):
    """
    Align boxes each specified by their ``(y0, y1)`` spans.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout (i.e., vertical alignment), but the function works
    equally for a vertical layout.

    Parameters
    ----------
    yspans
        List of (y0, y1) spans of boxes to be aligned.
    height : float or None
        Intended total height. If None, the maximum of the heights
        (``y1 - y0``) in *yspans* is used.
    align : {'baseline', 'left', 'top', 'right', 'bottom', 'center'}
        The alignment anchor of the boxes.

    Returns
    -------
    (y0, y1)
        y range spanned by the packing.  If a *height* was originally passed
        in, then for all alignments other than "baseline", a span of ``(0,
        height)`` is used without checking that it is actually large enough).
    descent
        The descent of the packing.
    offsets
        The bottom offsets of the boxes.
    """

    _api.check_in_list(
        ["baseline", "left", "top", "right", "bottom", "center"], align=align)
    if height is None:
        height = max(y1 - y0 for y0, y1 in yspans)

    if align == "baseline":
        yspan = (min(y0 for y0, y1 in yspans), max(y1 for y0, y1 in yspans))
        offsets = [0] * len(yspans)
    elif align in ["left", "bottom"]:
        yspan = (0, height)
        offsets = [-y0 for y0, y1 in yspans]
    elif align in ["right", "top"]:
        yspan = (0, height)
        offsets = [height - y1 for y0, y1 in yspans]
    elif align == "center":
        yspan = (0, height)
        offsets = [(height - (y1 - y0)) * .5 - y0 for y0, y1 in yspans]

    return yspan, offsets


class OffsetBox(martist.Artist):
    """
    A simple container artist.

    The child artists are meant to be drawn at a relative position to its
    parent.

    Being an artist itself, all parameters are passed on to `.Artist`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._internal_update(kwargs)
        # Clipping has not been implemented in the OffsetBox family, so
        # disable the clip flag for consistency. It can always be turned back
        # on to zero effect.
        self.set_clip_on(False)
        self._children = []
        self._offset = (0, 0)

    def set_figure(self, fig):
        """
        Set the `.Figure` for the `.OffsetBox` and all its children.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
        """
        super().set_figure(fig)
        for c in self.get_children():
            c.set_figure(fig)

    @martist.Artist.axes.setter
    def axes(self, ax):
        # TODO deal with this better
        martist.Artist.axes.fset(self, ax)
        for c in self.get_children():
            if c is not None:
                c.axes = ax

    def contains(self, mouseevent):
        """
        Delegate the mouse event contains-check to the children.

        As a container, the `.OffsetBox` does not respond itself to
        mouseevents.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            An artist-specific dictionary of details of the event context,
            such as which points are contained in the pick radius. See the
            individual Artist subclasses for details.

        See Also
        --------
        .Artist.contains
        """
        if self._different_canvas(mouseevent):
            return False, {}
        for c in self.get_children():
            a, b = c.contains(mouseevent)
            if a:
                return a, b
        return False, {}

    def set_offset(self, xy):
        """
        Set the offset.

        Parameters
        ----------
        xy : (float, float) or callable
            The (x, y) coordinates of the offset in display units. These can
            either be given explicitly as a tuple (x, y), or by providing a
            function that converts the extent into the offset. This function
            must have the signature::

                def offset(width, height, xdescent, ydescent, renderer) \
-> (float, float)
        """
        self._offset = xy
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        """
        Return the offset as a tuple (x, y).

        The extent parameters have to be provided to handle the case where the
        offset is dynamically determined by a callable (see
        `~.OffsetBox.set_offset`).

        Parameters
        ----------
        bbox : `.Bbox`
        renderer : `.RendererBase` subclass
        """
        return (
            self._offset(bbox.width, bbox.height, -bbox.x0, -bbox.y0, renderer)
            if callable(self._offset)
            else self._offset)

    def set_width(self, width):
        """
        Set the width of the box.

        Parameters
        ----------
        width : float
        """
        self.width = width
        self.stale = True

    def set_height(self, height):
        """
        Set the height of the box.

        Parameters
        ----------
        height : float
        """
        self.height = height
        self.stale = True

    def get_visible_children(self):
        r"""Return a list of the visible child `.Artist`\s."""
        return [c for c in self._children if c.get_visible()]

    def get_children(self):
        r"""Return a list of the child `.Artist`\s."""
        return self._children

    def _get_bbox_and_child_offsets(self, renderer):
        """
        Return the bbox of the offsetbox and the child offsets.

        The bbox should satisfy ``x0 <= x1 and y0 <= y1``.

        Parameters
        ----------
        renderer : `.RendererBase` subclass

        Returns
        -------
        bbox
        list of (xoffset, yoffset) pairs
        """
        raise NotImplementedError(
            "get_bbox_and_offsets must be overridden in derived classes")

    def get_bbox(self, renderer):
        """Return the bbox of the offsetbox, ignoring parent offsets."""
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        return bbox

    def get_window_extent(self, renderer=None):
        # docstring inherited
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        bbox = self.get_bbox(renderer)
        try:  # Some subclasses redefine get_offset to take no args.
            px, py = self.get_offset(bbox, renderer)
        except TypeError:
            px, py = self.get_offset()
        return bbox.translated(px, py)

    def draw(self, renderer):
        """
        Update the location of children if necessary and draw them
        to the given *renderer*.
        """
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        px, py = self.get_offset(bbox, renderer)
        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        self.stale = False


class PackerBase(OffsetBox):
    def __init__(self, pad=0., sep=0., width=None, height=None,
                 align="baseline", mode="fixed", children=None):
        """
        Parameters
        ----------
        pad : float, default: 0.0
            The boundary padding in points.

        sep : float, default: 0.0
            The spacing between items in points.

        width, height : float, optional
            Width and height of the container box in pixels, calculated if
            *None*.

        align : {'top', 'bottom', 'left', 'right', 'center', 'baseline'}, \
default: 'baseline'
            Alignment of boxes.

        mode : {'fixed', 'expand', 'equal'}, default: 'fixed'
            The packing mode.

            - 'fixed' packs the given `.Artist`\\s tight with *sep* spacing.
            - 'expand' uses the maximal available space to distribute the
              artists with equal spacing in between.
            - 'equal': Each artist an equal fraction of the available space
              and is left-aligned (or top-aligned) therein.

        children : list of `.Artist`
            The artists to pack.

        Notes
        -----
        *pad* and *sep* are in points and will be scaled with the renderer
        dpi, while *width* and *height* are in pixels.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.sep = sep
        self.pad = pad
        self.mode = mode
        self.align = align
        self._children = children


class VPacker(PackerBase):
    """
    VPacker packs its children vertically, automatically adjusting their
    relative positions at draw time.

    .. code-block:: none

       +---------+
       | Child 1 |
       | Child 2 |
       | Child 3 |
       +---------+
    """

