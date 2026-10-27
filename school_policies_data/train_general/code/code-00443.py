    - size : optional tuple of two floats, (width, height); default: None

    The per data series options and aesthetics are:
    There are none in the base series. See below for options for subclasses.

    Some data series support additional aesthetics or options:

    :class:`~.LineOver1DRangeSeries`, :class:`~.Parametric2DLineSeries`, and
    :class:`~.Parametric3DLineSeries` support the following:

    Aesthetics:

    - line_color : string, or float, or function, optional
        Specifies the color for the plot, which depends on the backend being
        used.

        For example, if ``MatplotlibBackend`` is being used, then
        Matplotlib string colors are acceptable (``"red"``, ``"r"``,
        ``"cyan"``, ``"c"``, ...).
        Alternatively, we can use a float number, 0 < color < 1, wrapped in a
        string (for example, ``line_color="0.5"``) to specify grayscale colors.
        Alternatively, We can specify a function returning a single
        float value: this will be used to apply a color-loop (for example,
        ``line_color=lambda x: math.cos(x)``).

        Note that by setting line_color, it would be applied simultaneously
        to all the series.

    Options:

    - label : str
    - steps : bool
    - integers_only : bool

    :class:`~.SurfaceOver2DRangeSeries` and :class:`~.ParametricSurfaceSeries`
    support the following:

    Aesthetics:

    - surface_color : function which returns a float.

    Notes
    =====

    How the plotting module works:

    1. Whenever a plotting function is called, the provided expressions are
       processed and a list of instances of the
       :class:`~sympy.plotting.series.BaseSeries` class is created, containing
       the necessary information to plot the expressions
       (e.g. the expression, ranges, series name, ...). Eventually, these
       objects will generate the numerical data to be plotted.
    2. A subclass of :class:`~.Plot` class is instantiaed (referred to as
       backend, from now on), which stores the list of series and the main
       attributes of the plot (e.g. axis labels, title, ...).
       The backend implements the logic to generate the actual figure with
       some plotting library.
    3. When the ``show`` command is executed, series are processed one by one
       to generate numerical data and add it to the figure. The backend is also
       going to set the axis labels, title, ..., according to the values stored
       in the Plot instance.

    The backend should check if it supports the data series that it is given
    (e.g. :class:`TextBackend` supports only
    :class:`~sympy.plotting.series.LineOver1DRangeSeries`).

    It is the backend responsibility to know how to use the class of data series
    that it's given. Note that the current implementation of the ``*Series``
    classes is "matplotlib-centric": the numerical data returned by the
    ``get_points`` and ``get_meshes`` methods is meant to be used directly by
    Matplotlib. Therefore, the new backend will have to pre-process the
    numerical data to make it compatible with the chosen plotting library.
    Keep in mind that future SymPy versions may improve the ``*Series`` classes
    in order to return numerical data "non-matplotlib-centric", hence if you code
    a new backend you have the responsibility to check if its working on each
    SymPy release.

    Please explore the :class:`MatplotlibBackend` source code to understand
    how a backend should be coded.

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods:

    * show(self): used to loop over the data series, generate the numerical
        data, plot it and set the axis labels, title, ...
    * save(self, path): used to save the current plot to the specified file
        path.
    * close(self): used to close the current plot backend (note: some plotting
        library does not support this functionality. In that case, just raise a
        warning).
    """

    def __init__(self, *args,
        title=None, xlabel=None, ylabel=None, zlabel=None, aspect_ratio='auto',
        xlim=None, ylim=None, axis_center='auto', axis=True,
        xscale='linear', yscale='linear', legend=False, autoscale=True,
        margin=0, annotations=None, markers=None, rectangles=None,
        fill=None, backend='default', size=None, **kwargs):

        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring of
        # Plot. They are based purely on convention, no checking is done.
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.aspect_ratio = aspect_ratio
        self.axis_center = axis_center
        self.axis = axis
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.autoscale = autoscale
        self.margin = margin
        self._annotations = annotations
        self._markers = markers
        self._rectangles = rectangles
        self._fill = fill

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)
        self._series.extend(_create_generic_data_series(
            annotations=annotations, markers=markers, rectangles=rectangles,
            fill=fill))

        is_real = \
            lambda lim: all(getattr(i, 'is_real', True) for i in lim)
        is_finite = \
            lambda lim: all(getattr(i, 'is_finite', True) for i in lim)

        # reduce code repetition
        def check_and_set(t_name, t):
            if t:
                if not is_real(t):
                    raise ValueError(
                    "All numbers from {}={} must be real".format(t_name, t))
                if not is_finite(t):
                    raise ValueError(
                    "All numbers from {}={} must be finite".format(t_name, t))
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", xlim)
        self.ylim = None
        check_and_set("ylim", ylim)
        self.size = None
        check_and_set("size", size)

    @property
    def _backend(self):
        return self

    @property
    def backend(self):
        return type(self)

    def __str__(self):
        series_strs = [('[%d]: ' % i) + str(s)
                       for i, s in enumerate(self._series)]
        return 'Plot object containing:\n' + '\n'.join(series_strs)

    def __getitem__(self, index):
        return self._series[index]

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args

    def __delitem__(self, index):
        del self._series[index]

    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot's first series object to the first, use the
        ``append`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x*x, show=False)
           >>> p2 = plot(x, show=False)
           >>> p1.append(p2[0])
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        extend

        """
        if isinstance(arg, BaseSeries):
            self._series.append(arg)
        else:
            raise TypeError('Must specify element of plot to append.')

    def extend(self, arg):
        """Adds all series from another plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot to the first, use the ``extend`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x**2, show=False)
           >>> p2 = plot(x, -x, show=False)
           >>> p1.extend(p2)
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           [2]: cartesian line: -x for x over (-10.0, 10.0)
           >>> p1.show()

        """
        if isinstance(arg, Plot):
            self._series.extend(arg._series)
        elif is_sequence(arg):
            self._series.extend(arg)
        else:
            raise TypeError('Expecting Plot or sequence of BaseSeries')

    def show(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    # deprecations

    @property
    def markers(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        return self._markers

    @markers.setter
    def markers(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        self._series.extend(_create_generic_data_series(markers=v))
        self._markers = v

    @property
    def annotations(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        return self._annotations

    @annotations.setter
    def annotations(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        self._series.extend(_create_generic_data_series(annotations=v))
        self._annotations = v

    @property
    def rectangles(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        return self._rectangles

    @rectangles.setter
    def rectangles(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        self._series.extend(_create_generic_data_series(rectangles=v))
        self._rectangles = v

    @property
    def fill(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        return self._fill

    @fill.setter
    def fill(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        self._series.extend(_create_generic_data_series(fill=v))
        self._fill = v
<|endoftext|>import sympy.plotting.backends.base_backend as base_backend
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.textplot import textplot


class TextBackend(base_backend.Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show(self):
        if not base_backend._show:
            return
        if len(self._series) != 1:
            raise ValueError(
                'The TextBackend supports only one graph per Plot.')
        elif not isinstance(self._series[0], LineOver1DRangeSeries):
            raise ValueError(
                'The TextBackend supports only expressions over a 1D range')
        else:
            ser = self._series[0]
            textplot(ser.expr, ser.start, ser.end)

    def close(self):
        pass
<|endoftext|>from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend
from sympy.printing.latex import latex


# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`


def _str_or_latex(label):
    if isinstance(label, Basic):
        return latex(label, mode='inline')
    return str(label)


def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start,
                          intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end,
                          intervaly.end, intervaly.start, None])
    else:
        #XXX Ugly hack. Matplotlib does not accept empty lists for ``fill``
        xlist.extend((None, None, None, None))
        ylist.extend((None, None, None, None))
    return xlist, ylist


# Don't have to check for the success of importing matplotlib in each case;
# we will only be using this backend if we can successfully import matploblib
class MatplotlibBackend(base_backend.Plot):
    """ This class implements the functionalities to use Matplotlib with SymPy
    plotting functions.
    """

    def __init__(self, *series, **kwargs):
        super().__init__(*series, **kwargs)
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.aspect = kwargs.get('aspect_ratio', 'auto')
        if self.aspect != 'auto':
            self.aspect = float(self.aspect[1]) / self.aspect[0]
        # PlotGrid can provide its figure and axes to be populated with
        # the data from the series.
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)

    def _create_figure(self):
        def set_spines(ax):
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if self._plotgrid_fig is not None:
            self.fig = self._plotgrid_fig
            self.ax = self._plotgrid_ax
            if not any(s.is_3D for s in self._series):
                set_spines(self.ax)
        else:
            self.fig = self.plt.figure(figsize=self.size)
            if any(s.is_3D for s in self._series):
                self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            else:
                self.ax = self.fig.add_subplot(1, 1, 1)
                set_spines(self.ax)

    @staticmethod
    def get_segments(x, y, z=None):
        """ Convert two list of coordinates to a list of segments to be used
        with Matplotlib's :external:class:`~matplotlib.collections.LineCollection`.

        Parameters
        ==========
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates

            z : list
                List of z-coordinates for a 3D line.
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _process_series(self, series, ax):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', import_kwargs={'fromlist': ['mplot3d']})

        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        for s in series:
            # Create the collections
            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                else:
                    x, y = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    segments = self.get_segments(x, y)
                    collection = self.LineCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    line, = ax.plot(x, y, label=lbl, color=s.line_color)
            elif s.is_contour:
                ax.contour(*s.get_data())
            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    art3d = mpl_toolkits.mplot3d.art3d
                    segments = self.get_segments(x, y, z)
                    collection = art3d.Line3DCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    ax.plot(x, y, z, label=lbl, color=s.line_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_3Dsurface:
                if s.is_parametric:
                    x, y, z, u, v = s.get_data()
                else:
                    x, y, z = s.get_data()
                collection = ax.plot_surface(x, y, z,
                    cmap=getattr(self.cm, 'viridis', self.cm.jet),
                    rstride=1, cstride=1, linewidth=0.1)
                if isinstance(s.surface_color, (float, int, Callable)):
                    color_array = s.get_color_array()
                    color_array = color_array.reshape(color_array.size)
                    collection.set_array(color_array)
                else:
                    collection.set_color(s.surface_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_implicit:
                points = s.get_data()
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    ax.fill(x, y, facecolor=s.line_color, edgecolor='None')
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    # XXX: ``contour`` plots multiple lines. Should be fixed.
                    ListedColormap = self.matplotlib.colors.ListedColormap
                    colormap = ListedColormap(["white", s.line_color])
                    xarray, yarray, zarray, plot_type = points
                    if plot_type == 'contour':
                        ax.contour(xarray, yarray, zarray, cmap=colormap)
                    else:
                        ax.contourf(xarray, yarray, zarray, cmap=colormap)
            elif s.is_generic:
                if s.type == "markers":
                    # s.rendering_kw["color"] = s.line_color
                    ax.plot(*s.args, **s.rendering_kw)
                elif s.type == "annotations":
                    ax.annotate(*s.args, **s.rendering_kw)
                elif s.type == "fill":
                    # s.rendering_kw["color"] = s.line_color
                    ax.fill_between(*s.args, **s.rendering_kw)
                elif s.type == "rectangles":
                    # s.rendering_kw["color"] = s.line_color
                    ax.add_patch(
                        self.matplotlib.patches.Rectangle(
                            *s.args, **s.rendering_kw))
            else:
                raise NotImplementedError(
                    '{} is not supported in the SymPy plotting module '
                    'with matplotlib backend. Please report this issue.'
                    .format(ax))

        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(ax, Axes3D):
            ax.autoscale_view(
                scalex=ax.get_autoscalex_on(),
                scaley=ax.get_autoscaley_on())
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
                ax.set_ylim(ylim)
            else:
                ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
                ax.set_zlim(zlim)
            else:
                ax.set_zlim([0, 1])

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(ax, Axes3D):
            ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(ax, Axes3D):
            ax.set_yscale(self.yscale)
        if not isinstance(ax, Axes3D) or self.matplotlib.__version__ >= '1.2.0':  # XXX in the distant future remove this check
            ax.set_autoscale_on(self.autoscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(ax, Axes3D):
                pass
            elif val == 'center':
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
            elif val == 'auto':
                xl, xh = ax.get_xlim()
                yl, yh = ax.get_ylim()
                pos_left = ('data', 0) if xl*xh <= 0 else 'center'
                pos_bottom = ('data', 0) if yl*yh <= 0 else 'center'
                ax.spines['left'].set_position(pos_left)
                ax.spines['bottom'].set_position(pos_bottom)
            else:
                ax.spines['left'].set_position(('data', val[0]))
                ax.spines['bottom'].set_position(('data', val[1]))
        if not self.axis:
            ax.set_axis_off()
        if self.legend:
            if ax.legend():
                ax.legend_.set_visible(self.legend)
        if self.margin:
            ax.set_xmargin(self.margin)
            ax.set_ymargin(self.margin)
        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            xlbl = _str_or_latex(self.xlabel)
            ax.set_xlabel(xlbl, position=(1, 0))
        if self.ylabel:
            ylbl = _str_or_latex(self.ylabel)
            ax.set_ylabel(ylbl, position=(0, 1))
        if isinstance(ax, Axes3D) and self.zlabel:
            zlbl = _str_or_latex(self.zlabel)
            ax.set_zlabel(zlbl, position=(0, 1))

        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        self.ax.set_aspect(self.aspect)


    def process_series(self):
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
        self._create_figure()
        self._process_series(self._series, self.ax)

    def show(self):
        self.process_series()
        #TODO after fixing https://github.com/ipython/ipython/issues/1255
        # you can uncomment the next line and remove the pyplot.show() call
        #self.fig.show()
        if base_backend._show:
            self.fig.tight_layout()
            self.plt.show()
        else:
            self.close()

    def save(self, path):
        self.process_series()
        self.fig.savefig(path)

    def close(self):
        self.plt.close(self.fig)
<|endoftext|>""" The module contains implemented functions for interval arithmetic."""
from functools import reduce

from sympy.plotting.intervalmath import interval
from sympy.external import import_module


def Abs(x):
    if isinstance(x, (int, float)):
        return interval(abs(x))
    elif isinstance(x, interval):
        if x.start < 0 and x.end > 0:
            return interval(0, max(abs(x.start), abs(x.end)), is_valid=x.is_valid)
        else:
            return interval(abs(x.start), abs(x.end))
    else:
        raise NotImplementedError

#Monotonic


def exp(x):
    """evaluates the exponential of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.exp(x), np.exp(x))
    elif isinstance(x, interval):
        return interval(np.exp(x.start), np.exp(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def log(x):
    """evaluates the natural logarithm of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)

        return interval(np.log(x.start), np.log(x.end))
    else:
        raise NotImplementedError


#Monotonic
def log10(x):
    """evaluates the logarithm to the base 10 of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log10(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)
        return interval(np.log10(x.start), np.log10(x.end))
    else:
        raise NotImplementedError


#Monotonic
def atan(x):
    """evaluates the tan inverse of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arctan(x))
    elif isinstance(x, interval):
        start = np.arctan(x.start)
        end = np.arctan(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#periodic
def sin(x):
    """evaluates the sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.sin(x.start), np.sin(x.end))
        end = max(np.sin(x.start), np.sin(x.end))
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na - 1) // 4 != (nb - 1) // 4:
                #sin has max
                end = 1
            if (na - 3) // 4 != (nb - 3) // 4:
                #sin has min
                start = -1
            return interval(start, end)
    else:
        raise NotImplementedError


#periodic
def cos(x):
    """Evaluates the cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not (np.isfinite(x.start) and np.isfinite(x.end)):
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.cos(x.start), np.cos(x.end))
        end = max(np.cos(x.start), np.cos(x.end))
        if nb - na > 4:
            #differ more than 2*pi
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            #in the same quadarant
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na) // 4 != (nb) // 4:
                #cos has max
                end = 1
            if (na - 2) // 4 != (nb - 2) // 4:
                #cos has min
                start = -1
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


def tan(x):
    """Evaluates the tan of an interval"""
    return sin(x) / cos(x)


#Monotonic
def sqrt(x):
    """Evaluates the square root of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x > 0:
            return interval(np.sqrt(x))
        else:
            return interval(-np.inf, np.inf, is_valid=False)
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 0:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < 0:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            return interval(np.sqrt(x.start), np.sqrt(x.end),
                    is_valid=x.is_valid)
    else:
        raise NotImplementedError


def imin(*args):
    """Evaluates the minimum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]
        return interval(min(start_array), min(end_array))


def imax(*args):
    """Evaluates the maximum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]

        return interval(max(start_array), max(end_array))


#Monotonic
def sinh(x):
    """Evaluates the hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sinh(x), np.sinh(x))
    elif isinstance(x, interval):
        return interval(np.sinh(x.start), np.sinh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def cosh(x):
    """Evaluates the hyperbolic cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.cosh(x), np.cosh(x))
    elif isinstance(x, interval):
        #both signs
        if x.start < 0 and x.end > 0:
            end = max(np.cosh(x.start), np.cosh(x.end))
            return interval(1, end, is_valid=x.is_valid)
        else:
            #Monotonic
            start = np.cosh(x.start)
            end = np.cosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def tanh(x):
    """Evaluates the hyperbolic tan of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.tanh(x), np.tanh(x))
    elif isinstance(x, interval):
        return interval(np.tanh(x.start), np.tanh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def asin(x):
    """Evaluates the inverse sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) > 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arcsin(x), np.arcsin(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arcsin(x.start)
            end = np.arcsin(x.end)
            return interval(start, end, is_valid=x.is_valid)


def acos(x):
    """Evaluates the inverse cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if abs(x) > 1:
            #Outside the domain
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccos(x), np.arccos(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccos(x.start)
            end = np.arccos(x.end)
            return interval(start, end, is_valid=x.is_valid)


def ceil(x):
    """Evaluates the ceiling of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.ceil(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.ceil(x.start)
            end = np.ceil(x.end)
            #Continuous over the interval
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #Not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def floor(x):
    """Evaluates the floor of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.floor(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.floor(x.start)
            end = np.floor(x.end)
            #continuous over the argument
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def acosh(x):
    """Evaluates the inverse hyperbolic cosine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if x < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccosh(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partly outside the domain
        elif x.start < 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccosh(x.start)
            end = np.arccosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Monotonic
def asinh(x):
    """Evaluates the inverse hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arcsinh(x))
    elif isinstance(x, interval):
        start = np.arcsinh(x.start)
        end = np.arcsinh(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


def atanh(x):
    """Evaluates the inverse hyperbolic tangent of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) >= 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arctanh(x))
    elif isinstance(x, interval):
        #outside the domain
        if x.is_valid is False or x.start >= 1 or x.end <= -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #partly outside the domain
        elif x.start <= -1 or x.end >= 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arctanh(x.start)
            end = np.arctanh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Three valued logic for interval plotting.

def And(*args):
    """Defines the three valued ``And`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_and(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is False or cmp_intervalb[0] is False:
            first = False
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = True
        if cmp_intervala[1] is False or cmp_intervalb[1] is False:
            second = False
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = True
        return (first, second)
    return reduce(reduce_and, args)


def Or(*args):
    """Defines the three valued ``Or`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_or(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is True or cmp_intervalb[0] is True:
            first = True
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = False

        if cmp_intervala[1] is True or cmp_intervalb[1] is True:
            second = True
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = False
        return (first, second)
    return reduce(reduce_or, args)
<|endoftext|>"""
Interval Arithmetic for plotting.
This module does not implement interval arithmetic accurately and
hence cannot be used for purposes other than plotting. If you want
to use interval arithmetic, use mpmath's interval arithmetic.

The module implements interval arithmetic using numpy and
python floating points. The rounding up and down is not handled
and hence this is not an accurate implementation of interval
arithmetic.

The module uses numpy for speed which cannot be achieved with mpmath.
"""

# Q: Why use numpy? Why not simply use mpmath's interval arithmetic?
# A: mpmath's interval arithmetic simulates a floating point unit
# and hence is slow, while numpy evaluations are orders of magnitude
# faster.

# Q: Why create a separate class for intervals? Why not use SymPy's
# Interval Sets?
# A: The functionalities that will be required for plotting is quite
# different from what Interval Sets implement.

# Q: Why is rounding up and down according to IEEE754 not handled?
# A: It is not possible to do it in both numpy and python. An external
# library has to used, which defeats the whole purpose i.e., speed. Also
# rounding is handled for very few functions in those libraries.

# Q Will my plots be affected?
# A It will not affect most of the plots. The interval arithmetic
# module based suffers the same problems as that of floating point
# arithmetic.

from sympy.core.numbers import int_valued
from sympy.core.logic import fuzzy_and
from sympy.simplify.simplify import nsimplify

from .interval_membership import intervalMembership


class interval:
    """ Represents an interval containing floating points as start and
    end of the interval
    The is_valid variable tracks whether the interval obtained as the
    result of the function is in the domain and is continuous.
    - True: Represents the interval result of a function is continuous and
            in the domain of the function.
    - False: The interval argument of the function was not in the domain of
             the function, hence the is_valid of the result interval is False
    - None: The function was not continuous over the interval or
            the function's argument interval is partly in the domain of the
            function

    A comparison between an interval and a real number, or a
    comparison between two intervals may return ``intervalMembership``
    of two 3-valued logic values.
    """

    def __init__(self, *args, is_valid=True, **kwargs):
        self.is_valid = is_valid
        if len(args) == 1:
            if isinstance(args[0], interval):
                self.start, self.end = args[0].start, args[0].end
            else:
                self.start = float(args[0])
                self.end = float(args[0])
        elif len(args) == 2:
            if args[0] < args[1]:
                self.start = float(args[0])
                self.end = float(args[1])
            else:
                self.start = float(args[1])
                self.end = float(args[0])

        else:
            raise ValueError("interval takes a maximum of two float values "
                            "as arguments")

    @property
    def mid(self):
        return (self.start + self.end) / 2.0

    @property
    def width(self):
        return self.end - self.start

    def __repr__(self):
        return "interval(%f, %f)" % (self.start, self.end)

    def __str__(self):
        return "[%f, %f]" % (self.start, self.end)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            if self.end < other:
                return intervalMembership(True, self.is_valid)
            elif self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        elif isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end < other. start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            if self.start > other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__lt__(self)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(True, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(False, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(True, valid)
            elif self.__lt__(other)[0] is not None:
                return intervalMembership(False, valid)
            else:
                return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(False, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(True, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(False, valid)
            if not self.__lt__(other)[0] is None:
                return intervalMembership(True, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            if self.end <= other:
                return intervalMembership(True, self.is_valid)
            if self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end <= other.start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            if self.start >= other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__le__(self)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.is_valid:
                return interval(self.start + other, self.end + other)
            else:
                start = self.start + other
                end = self.end + other
                return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start + other.start
            end = self.end + other.end
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            start = self.start - other
            end = self.end - other
            return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start - other.end
            end = self.end - other.start
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            start = other - self.end
            end = other - self.start
            return interval(start, end, is_valid=self.is_valid)
        elif isinstance(other, interval):
            return other.__sub__(self)
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_valid:
            return interval(-self.end, -self.start)
        else:
            return interval(-self.end, -self.start, is_valid=self.is_valid)

    def __mul__(self, other):
        if isinstance(other, interval):
            if self.is_valid is False or other.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif self.is_valid is None or other.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
                inters = []
                inters.append(self.start * other.start)
                inters.append(self.end * other.start)
                inters.append(self.start * other.end)
                inters.append(self.end * other.end)
                start = min(inters)
                end = max(inters)
                return interval(start, end)
        elif isinstance(other, (int, float)):
            return interval(self.start*other, self.end*other, is_valid=self.is_valid)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __contains__(self, other):
        if isinstance(other, (int, float)):
            return self.start <= other and self.end >= other
        else:
            return self.start <= other.start and other.end <= self.end

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = interval(other)
            return other.__truediv__(self)
        elif isinstance(other, interval):
            return other.__truediv__(self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        # Both None and False are handled
        if not self.is_valid:
            # Don't divide as the value is not valid
            return interval(-float('inf'), float('inf'), is_valid=self.is_valid)
        if isinstance(other, (int, float)):
            if other == 0:
                # Divide by zero encountered. valid nowhere
                return interval(-float('inf'), float('inf'), is_valid=False)
            else:
                return interval(self.start / other, self.end / other)

        elif isinstance(other, interval):
            if other.is_valid is False or self.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif other.is_valid is None or self.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
               # denominator contains both signs, i.e. being divided by zero
               # return the whole real line with is_valid = None
                if 0 in other:
                    return interval(-float('inf'), float('inf'), is_valid=None)

                # denominator negative
                this = self
                if other.end < 0:
                    this = -this
                    other = -other

                # denominator positive
                inters = []
                inters.append(this.start / other.start)
                inters.append(this.end / other.start)
                inters.append(this.start / other.end)
                inters.append(this.end / other.end)
                start = max(inters)
                end = min(inters)
                return interval(start, end)
        else:
            return NotImplemented

    def __pow__(self, other):
        # Implements only power to an integer.
        from .lib_interval import exp, log
        if not self.is_valid:
            return self
        if isinstance(other, interval):
            return exp(other * log(self))
        elif isinstance(other, (float, int)):
            if other < 0:
                return 1 / self.__pow__(abs(other))
            else:
                if int_valued(other):
                    return _pow_int(self, other)
                else:
                    return _pow_float(self, other)
        else:
            return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            if not self.is_valid:
                #Don't do anything
                return self
            elif other < 0:
                if self.width > 0:
                    return interval(-float('inf'), float('inf'), is_valid=False)
                else:
                    power_rational = nsimplify(self.start)
                    num, denom = power_rational.as_numer_denom()
                    if denom % 2 == 0:
                        return interval(-float('inf'), float('inf'),
                                        is_valid=False)
                    else:
                        start = -abs(other)**self.start
                        end = start
                        return interval(start, end)
            else:
                return interval(other**self.start, other**self.end)
        elif isinstance(other, interval):
            return other.__pow__(self)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.is_valid, self.start, self.end))


def _pow_float(inter, power):
    """Evaluates an interval raised to a floating point."""
    power_rational = nsimplify(power)
    num, denom = power_rational.as_numer_denom()
    if num % 2 == 0:
        start = abs(inter.start)**power
        end = abs(inter.end)**power
        if start < 0:
            ret = interval(0, max(start, end))
        else:
            ret = interval(start, end)
        return ret
    elif denom % 2 == 0:
        if inter.end < 0:
            return interval(-float('inf'), float('inf'), is_valid=False)
        elif inter.start < 0:
            return interval(0, inter.end**power, is_valid=None)
        else:
            return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0:
            start = -abs(inter.start)**power
        else:
            start = inter.start**power

        if inter.end < 0:
            end = -abs(inter.end)**power
        else:
            end = inter.end**power

        return interval(start, end, is_valid=inter.is_valid)


def _pow_int(inter, power):
    """Evaluates an interval raised to an integer power"""
    power = int(power)
    if power & 1:
        return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0 and inter.end > 0:
            start = 0
            end = max(inter.start**power, inter.end**power)
            return interval(start, end)
        else:
            return interval(inter.start**power, inter.end**power)
<|endoftext|>from sympy.core.logic import fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_xor


class intervalMembership:
    """Represents a boolean expression returned by the comparison of
    the interval object.

    Parameters
    ==========

    (a, b) : (bool, bool)
        The first value determines the comparison as follows:
        - True: If the comparison is True throughout the intervals.
        - False: If the comparison is False throughout the intervals.
        - None: If the comparison is True for some part of the intervals.

        The second value is determined as follows:
        - True: If both the intervals in comparison are valid.
        - False: If at least one of the intervals is False, else
        - None
    """
    def __init__(self, a, b):
        self._wrapped = (a, b)

    def __getitem__(self, i):
        try:
            return self._wrapped[i]
        except IndexError:
            raise IndexError(
                "{} must be a valid indexing for the 2-tuple."
                .format(i))

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self._wrapped)

    def __str__(self):
        return "intervalMembership({}, {})".format(*self)
    __repr__ = __str__

    def __and__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_and([a1, a2]), fuzzy_and([b1, b2]))

    def __or__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_or([a1, a2]), fuzzy_and([b1, b2]))

    def __invert__(self):
        a, b = self
        return intervalMembership(fuzzy_not(a), b)

    def __xor__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_xor([a1, a2]), fuzzy_and([b1, b2]))

    def __eq__(self, other):
        return self._wrapped == other

    def __ne__(self, other):
        return self._wrapped != other
<|endoftext|>"""
The Schur number S(k) is the largest integer n for which the interval [1,n]
can be partitioned into k sum-free sets.(https://mathworld.wolfram.com/SchurNumber.html)
"""
import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer


class SchurNumber(Function):
    r"""
    This function creates a SchurNumber object
    which is evaluated for `k \le 5` otherwise only
    the lower bound information can be retrieved.

    Examples
    ========

    >>> from sympy.combinatorics.schur_number import SchurNumber

    Since S(3) = 13, hence the output is a number
    >>> SchurNumber(3)
    13

    We do not know the Schur number for values greater than 5, hence
    only the object is returned
    >>> SchurNumber(6)
    SchurNumber(6)

    Now, the lower bound information can be retrieved using lower_bound()
    method
    >>> SchurNumber(6).lower_bound()
    536

    """

    @classmethod
    def eval(cls, k):
        if k.is_Number:
            if k is S.Infinity:
                return S.Infinity
            if k.is_zero:
                return S.Zero
            if not k.is_integer or k.is_negative:
                raise ValueError("k should be a positive integer")
            first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
            if k <= 5:
                return Integer(first_known_schur_numbers[k])

    def lower_bound(self):
        f_ = self.args[0]
        # Improved lower bounds known for S(6) and S(7)
        if f_ == 6:
            return Integer(536)
        if f_ == 7:
            return Integer(1680)
        # For other cases, use general expression
        if f_.is_Integer:
            return 3*self.func(f_ - 1).lower_bound() - 1
        return (3**f_ - 1)/2


def _schur_subsets_number(n):

    if n is S.Infinity:
        raise ValueError("Input must be finite")
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    elif n <= 3:
        min_k = 1
    else:
        min_k = math.ceil(math.log(2*n + 1, 3))

    return Integer(min_k)


def schur_partition(n):
    """

    This function returns the partition in the minimum number of sum-free subsets
    according to the lower bound given by the Schur Number.

    Parameters
    ==========

    n: a number
        n is the upper limit of the range [1, n] for which we need to find and
        return the minimum number of free subsets according to the lower bound
        of schur number

    Returns
    =======

    List of lists
        List of the minimum number of sum-free subsets

    Notes
    =====

    It is possible for some n to make the partition into less
    subsets since the only known Schur numbers are:
    S(1) = 1, S(2) = 4, S(3) = 13, S(4) = 44.
    e.g for n = 44 the lower bound from the function above is 5 subsets but it has been proven
    that can be done with 4 subsets.

    Examples
    ========

    For n = 1, 2, 3 the answer is the set itself

    >>> from sympy.combinatorics.schur_number import schur_partition
    >>> schur_partition(2)
    [[1, 2]]

    For n > 3, the answer is the minimum number of sum-free subsets:

    >>> schur_partition(5)
    [[3, 2], [5], [1, 4]]

    >>> schur_partition(8)
    [[3, 2], [6, 5, 8], [1, 4, 7]]
    """

    if isinstance(n, Basic) and not n.is_Number:
        raise ValueError("Input value must be a number")

    number_of_subsets = _schur_subsets_number(n)
    if n == 1:
        sum_free_subsets = [[1]]
    elif n == 2:
        sum_free_subsets = [[1, 2]]
    elif n == 3:
        sum_free_subsets = [[1, 2, 3]]
    else:
        sum_free_subsets = [[1, 4], [2, 3]]

    while len(sum_free_subsets) < number_of_subsets:
        sum_free_subsets = _generate_next_list(sum_free_subsets, n)
        missed_elements = [3*k + 1 for k in range(len(sum_free_subsets), (n-1)//3 + 1)]
        sum_free_subsets[-1] += missed_elements

    return sum_free_subsets


def _generate_next_list(current_list, n):
    new_list = []

    for item in current_list:
        temp_1 = [number*3 for number in item if number*3 <= n]
        temp_2 = [number*3 - 1 for number in item if number*3 - 1 <= n]
        new_item = temp_1 + temp_2
        new_list.append(new_item)

    last_list = [3*k + 1 for k in range(len(current_list)+1) if 3*k + 1 <= n]
    new_list.append(last_list)
    current_list = new_list

    return current_list
<|endoftext|>from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

_af_new = Permutation._af_new


def AbelianGroup(*cyclic_orders):
    """
    Returns the direct product of cyclic groups with the given orders.

    Explanation
    ===========

    According to the structure theorem for finite abelian groups ([1]),
    every finite abelian group can be written as the direct product of
    finitely many cyclic groups.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AbelianGroup
    >>> AbelianGroup(3, 4)
    PermutationGroup([
            (6)(0 1 2),
            (3 4 5 6)])
    >>> _.is_group
    True

    See Also
    ========

    DirectProduct

    References
    ==========

    .. [1] https://groupprops.subwiki.org/wiki/Structure_theorem_for_finitely_generated_abelian_groups

    """
    groups = []
    degree = 0
    order = 1
    for size in cyclic_orders:
        degree += size
        order *= size
        groups.append(CyclicGroup(size))
    G = DirectProduct(*groups)
    G._is_abelian = True
    G._degree = degree
    G._order = order

    return G


def AlternatingGroup(n):
    """
    Generates the alternating group on ``n`` elements as a permutation group.

    Explanation
    ===========

    For ``n > 2``, the generators taken are ``(0 1 2), (0 1 2 ... n-1)`` for
    ``n`` odd
    and ``(0 1 2), (1 2 ... n-1)`` for ``n`` even (See [1], p.31, ex.6.9.).
    After the group is generated, some of its basic properties are set.
    The cases ``n = 1, 2`` are handled separately.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> G = AlternatingGroup(4)
    >>> G.is_group
    True
    >>> a = list(G.generate_dimino())
    >>> len(a)
    12
    >>> all(perm.is_even for perm in a)
    True

    See Also
    ========

    SymmetricGroup, CyclicGroup, DihedralGroup

    References
    ==========

    .. [1] Armstrong, M. "Groups and Symmetry"

    """
    # small cases are special
    if n in (1, 2):
        return PermutationGroup([Permutation([0])])

    a = list(range(n))
    a[0], a[1], a[2] = a[1], a[2], a[0]
    gen1 = a
    if n % 2:
        a = list(range(1, n))
        a.append(0)
        gen2 = a
    else:
        a = list(range(2, n))
        a.append(1)
        a.insert(0, 0)
        gen2 = a
    gens = [gen1, gen2]
    if gen1 == gen2:
        gens = gens[:1]
    G = PermutationGroup([_af_new(a) for a in gens], dups=False)

    set_alternating_group_properties(G, n, n)
    G._is_alt = True
    return G


def set_alternating_group_properties(G, n, degree):
    """Set known properties of an alternating group. """
    if n < 4:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        G._is_abelian = False
        G._is_nilpotent = False
    if n < 5:
        G._is_solvable = True
    else:
        G._is_solvable = False
    G._degree = degree
    G._is_transitive = True
    G._is_dihedral = False


def CyclicGroup(n):
    """
    Generates the cyclic group of order ``n`` as a permutation group.

    Explanation
    ===========

    The generator taken is the ``n``-cycle ``(0 1 2 ... n-1)``
    (in cycle notation). After the group is generated, some of its basic
    properties are set.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> G = CyclicGroup(6)
    >>> G.is_group
    True
    >>> G.order()
    6
    >>> list(G.generate_schreier_sims(af=True))
    [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 0, 1],
    [3, 4, 5, 0, 1, 2], [4, 5, 0, 1, 2, 3], [5, 0, 1, 2, 3, 4]]

    See Also
    ========

    SymmetricGroup, DihedralGroup, AlternatingGroup

    """
    a = list(range(1, n))
    a.append(0)
    gen = _af_new(a)
    G = PermutationGroup([gen])

    G._is_abelian = True
    G._is_nilpotent = True
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = n
    G._is_dihedral = (n == 2)
    return G


def DihedralGroup(n):
    r"""
    Generates the dihedral group `D_n` as a permutation group.

    Explanation
    ===========

    The dihedral group `D_n` is the group of symmetries of the regular
    ``n``-gon. The generators taken are the ``n``-cycle ``a = (0 1 2 ... n-1)``
    (a rotation of the ``n``-gon) and ``b = (0 n-1)(1 n-2)...``
    (a reflection of the ``n``-gon) in cycle rotation. It is easy to see that
    these satisfy ``a**n = b**2 = 1`` and ``bab = ~a`` so they indeed generate
    `D_n` (See [1]). After the group is generated, some of its basic properties
    are set.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> G = DihedralGroup(5)
    >>> G.is_group
    True
    >>> a = list(G.generate_dimino())
    >>> [perm.cyclic_form for perm in a]
    [[], [[0, 1, 2, 3, 4]], [[0, 2, 4, 1, 3]],
    [[0, 3, 1, 4, 2]], [[0, 4, 3, 2, 1]], [[0, 4], [1, 3]],
    [[1, 4], [2, 3]], [[0, 1], [2, 4]], [[0, 2], [3, 4]],
    [[0, 3], [1, 2]]]

    See Also
    ========

    SymmetricGroup, CyclicGroup, AlternatingGroup

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dihedral_group

    """
    # small cases are special
    if n == 1:
        return PermutationGroup([Permutation([1, 0])])
    if n == 2:
        return PermutationGroup([Permutation([1, 0, 3, 2]),
               Permutation([2, 3, 0, 1]), Permutation([3, 2, 1, 0])])

    a = list(range(1, n))
    a.append(0)
    gen1 = _af_new(a)
    a = list(range(n))
    a.reverse()
    gen2 = _af_new(a)
    G = PermutationGroup([gen1, gen2])
    # if n is a power of 2, group is nilpotent
    if n & (n-1) == 0:
        G._is_nilpotent = True
    else:
        G._is_nilpotent = False
    G._is_dihedral = True
    G._is_abelian = False
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = 2*n
    return G


def SymmetricGroup(n):
    """
    Generates the symmetric group on ``n`` elements as a permutation group.

    Explanation
    ===========

    The generators taken are the ``n``-cycle
    ``(0 1 2 ... n-1)`` and the transposition ``(0 1)`` (in cycle notation).
    (See [1]). After the group is generated, some of its basic properties
    are set.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import SymmetricGroup
    >>> G = SymmetricGroup(4)
    >>> G.is_group
    True
    >>> G.order()
    24
    >>> list(G.generate_schreier_sims(af=True))
    [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 1, 2, 0], [0, 2, 3, 1],
    [1, 3, 0, 2], [2, 0, 1, 3], [3, 2, 0, 1], [0, 3, 1, 2], [1, 0, 2, 3],
    [2, 1, 3, 0], [3, 0, 1, 2], [0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0],
    [3, 1, 0, 2], [0, 2, 1, 3], [1, 3, 2, 0], [2, 0, 3, 1], [3, 2, 1, 0],
    [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 0, 2, 1]]

    See Also
    ========

    CyclicGroup, DihedralGroup, AlternatingGroup

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_group#Generators_and_relations

    """
    if n == 1:
        G = PermutationGroup([Permutation([0])])
    elif n == 2:
        G = PermutationGroup([Permutation([1, 0])])
    else:
        a = list(range(1, n))
        a.append(0)
        gen1 = _af_new(a)
        a = list(range(n))
        a[0], a[1] = a[1], a[0]
        gen2 = _af_new(a)
        G = PermutationGroup([gen1, gen2])
    set_symmetric_group_properties(G, n, n)
    G._is_sym = True
    return G


def set_symmetric_group_properties(G, n, degree):
    """Set known properties of a symmetric group. """
    if n < 3:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        G._is_abelian = False
        G._is_nilpotent = False
    if n < 5:
        G._is_solvable = True
    else:
        G._is_solvable = False
    G._degree = degree
    G._is_transitive = True
    G._is_dihedral = (n in [2, 3])  # cf Landau's func and Stirling's approx


def RubikGroup(n):
    """Return a group of Rubik's cube generators

    >>> from sympy.combinatorics.named_groups import RubikGroup
    >>> RubikGroup(2).is_group
    True
    """
    from sympy.combinatorics.generators import rubik
    if n <= 1:
        raise ValueError("Invalid cube. n has to be greater than 1")
    return PermutationGroup(rubik(n))
<|endoftext|>r"""
Construct transitive subgroups of symmetric groups, useful in Galois theory.

Besides constructing instances of the :py:class:`~.PermutationGroup` class to
represent the transitive subgroups of $S_n$ for small $n$, this module provides
*names* for these groups.

In some applications, it may be preferable to know the name of a group,
rather than receive an instance of the :py:class:`~.PermutationGroup`
class, and then have to do extra work to determine which group it is, by
checking various properties.

Names are instances of ``Enum`` classes defined in this module. With a name in
hand, the name's ``get_perm_group`` method can then be used to retrieve a
:py:class:`~.PermutationGroup`.

The names used for groups in this module are taken from [1].

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
from enum import Enum
import itertools

from sympy.combinatorics.named_groups import (
    SymmetricGroup, AlternatingGroup, CyclicGroup, DihedralGroup,
    set_symmetric_group_properties, set_alternating_group_properties,
)
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation


class S1TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S1.
    """
    S1 = "S1"

    def get_perm_group(self):
        return SymmetricGroup(1)


class S2TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S2.
    """
    S2 = "S2"

    def get_perm_group(self):
        return SymmetricGroup(2)


class S3TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S3.
    """
    A3 = "A3"
    S3 = "S3"

    def get_perm_group(self):
        if self == S3TransitiveSubgroups.A3:
            return AlternatingGroup(3)
        elif self == S3TransitiveSubgroups.S3:
            return SymmetricGroup(3)


class S4TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S4.
    """
    C4 = "C4"
    V = "V"
    D4 = "D4"
    A4 = "A4"
    S4 = "S4"

    def get_perm_group(self):
        if self == S4TransitiveSubgroups.C4:
            return CyclicGroup(4)
        elif self == S4TransitiveSubgroups.V:
            return four_group()
        elif self == S4TransitiveSubgroups.D4:
            return DihedralGroup(4)
        elif self == S4TransitiveSubgroups.A4:
            return AlternatingGroup(4)
        elif self == S4TransitiveSubgroups.S4:
            return SymmetricGroup(4)


class S5TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S5.
    """
    C5 = "C5"
    D5 = "D5"
    M20 = "M20"
    A5 = "A5"
    S5 = "S5"

    def get_perm_group(self):
        if self == S5TransitiveSubgroups.C5:
            return CyclicGroup(5)
        elif self == S5TransitiveSubgroups.D5:
            return DihedralGroup(5)
        elif self == S5TransitiveSubgroups.M20:
            return M20()
        elif self == S5TransitiveSubgroups.A5:
            return AlternatingGroup(5)
        elif self == S5TransitiveSubgroups.S5:
            return SymmetricGroup(5)


class S6TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S6.
    """
    C6 = "C6"
    S3 = "S3"
    D6 = "D6"
    A4 = "A4"
    G18 = "G18"
    A4xC2 = "A4 x C2"
    S4m = "S4-"
    S4p = "S4+"
    G36m = "G36-"
    G36p = "G36+"
    S4xC2 = "S4 x C2"
    PSL2F5 = "PSL2(F5)"
    G72 = "G72"
    PGL2F5 = "PGL2(F5)"
    A6 = "A6"
    S6 = "S6"

    def get_perm_group(self):
        if self == S6TransitiveSubgroups.C6:
            return CyclicGroup(6)
        elif self == S6TransitiveSubgroups.S3:
            return S3_in_S6()
        elif self == S6TransitiveSubgroups.D6:
            return DihedralGroup(6)
        elif self == S6TransitiveSubgroups.A4:
            return A4_in_S6()
        elif self == S6TransitiveSubgroups.G18:
            return G18()
        elif self == S6TransitiveSubgroups.A4xC2:
            return A4xC2()
        elif self == S6TransitiveSubgroups.S4m:
            return S4m()
        elif self == S6TransitiveSubgroups.S4p:
            return S4p()
        elif self == S6TransitiveSubgroups.G36m:
            return G36m()
        elif self == S6TransitiveSubgroups.G36p:
            return G36p()
        elif self == S6TransitiveSubgroups.S4xC2:
            return S4xC2()
        elif self == S6TransitiveSubgroups.PSL2F5:
            return PSL2F5()
        elif self == S6TransitiveSubgroups.G72:
            return G72()
        elif self == S6TransitiveSubgroups.PGL2F5:
            return PGL2F5()
        elif self == S6TransitiveSubgroups.A6:
            return AlternatingGroup(6)
        elif self == S6TransitiveSubgroups.S6:
            return SymmetricGroup(6)


def four_group():
    """
    Return a representation of the Klein four-group as a transitive subgroup
    of S4.
    """
    return PermutationGroup(
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3)
    )


def M20():
    """
    Return a representation of the metacyclic group M20, a transitive subgroup
    of S5 that is one of the possible Galois groups for polys of degree 5.

    Notes
    =====

    See [1], Page 323.

    """
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(1, 2, 4, 3))
    G._degree = 5
    G._order = 20
    G._is_transitive = True
    G._is_sym = False
    G._is_alt = False
    G._is_cyclic = False
    G._is_dihedral = False
    return G


def S3_in_S6():
    """
    Return a representation of S3 as a transitive subgroup of S6.

    Notes
    =====

    The representation is found by viewing the group as the symmetries of a
    triangular prism.

    """
    G = PermutationGroup(Permutation(0, 1, 2)(3, 4, 5), Permutation(0, 3)(2, 4)(1, 5))
    set_symmetric_group_properties(G, 3, 6)
    return G


def A4_in_S6():
    """
    Return a representation of A4 as a transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4))
    set_alternating_group_properties(G, 4, 6)
    return G


def S4m():
    """
    Return a representation of the S4- transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3))
    set_symmetric_group_properties(G, 4, 6)
    return G


def S4p():
    """
    Return a representation of the S4+ transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(0, 2, 4, 1)(3, 5), Permutation(0, 3)(4, 5))
    set_symmetric_group_properties(G, 4, 6)
    return G


def A4xC2():
    """
    Return a representation of the (A4 x C2) transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4),
        Permutation(5)(2, 4))


def S4xC2():
    """
    Return a representation of the (S4 x C2) transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3),
        Permutation(1, 4)(3, 5))


def G18():
    """
    Return a representation of the group G18, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C2.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 4)(1, 5)(2, 3))


def G36m():
    """
    Return a representation of the group G36-, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C2^2.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(1, 2)(3, 5), Permutation(0, 4)(1, 5)(2, 3))


def G36p():
    """
    Return a representation of the group G36+, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C4.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 5, 2, 3)(1, 4))


def G72():
    """
    Return a representation of the group G72, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with D4.

    Notes
    =====

    See [1], Page 325.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2),
        Permutation(0, 4, 1, 3)(2, 5), Permutation(0, 3)(1, 4)(2, 5))


def PSL2F5():
    r"""
    Return a representation of the group $PSL_2(\mathbb{F}_5)$, as a transitive
    subgroup of S6, isomorphic to $A_5$.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 4, 3, 1, 5))
    set_alternating_group_properties(G, 5, 6)
    return G


def PGL2F5():
    r"""
    Return a representation of the group $PGL_2(\mathbb{F}_5)$, as a transitive
    subgroup of S6, isomorphic to $S_5$.

    Notes
    =====

    See [1], Page 325.

    """
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4), Permutation(0, 5)(1, 2)(3, 4))
    set_symmetric_group_properties(G, 5, 6)
    return G


def find_transitive_subgroups_of_S6(*targets, print_report=False):
    r"""
    Search for certain transitive subgroups of $S_6$.

    The symmetric group $S_6$ has 16 different transitive subgroups, up to
    conjugacy. Some are more easily constructed than others. For example, the
    dihedral group $D_6$ is immediately found, but it is not at all obvious how
    to realize $S_4$ or $S_5$ *transitively* within $S_6$.

    In some cases there are well-known constructions that can be used. For
    example, $S_5$ is isomorphic to $PGL_2(\mathbb{F}_5)$, which acts in a
    natural way on the projective line $P^1(\mathbb{F}_5)$, a set of order 6.

    In absence of such special constructions however, we can simply search for
    generators. For example, transitive instances of $A_4$ and $S_4$ can be
    found within $S_6$ in this way.

    Once we are engaged in such searches, it may then be easier (if less
    elegant) to find even those groups like $S_5$ that do have special
    constructions, by mere search.

    This function locates generators for transitive instances in $S_6$ of the
    following subgroups:

    * $A_4$
    * $S_4^-$ ($S_4$ not contained within $A_6$)
    * $S_4^+$ ($S_4$ contained within $A_6$)
    * $A_4 \times C_2$
    * $S_4 \times C_2$
    * $G_{18}   = C_3^2 \rtimes C_2$
    * $G_{36}^- = C_3^2 \rtimes C_2^2$
    * $G_{36}^+ = C_3^2 \rtimes C_4$
    * $G_{72}   = C_3^2 \rtimes D_4$
    * $A_5$
    * $S_5$

    Note: Each of these groups also has a dedicated function in this module
    that returns the group immediately, using generators that were found by
    this search procedure.

    The search procedure serves as a record of how these generators were
    found. Also, due to randomness in the generation of the elements of
    permutation groups, it can be called again, in order to (probably) get
    different generators for the same groups.

    Parameters
    ==========

    targets : list of :py:class:`~.S6TransitiveSubgroups` values
        The groups you want to find.

    print_report : bool (default False)
        If True, print to stdout the generators found for each group.

    Returns
    =======

    dict
        mapping each name in *targets* to the :py:class:`~.PermutationGroup`
        that was found

    References
    ==========

    .. [2] https://en.wikipedia.org/wiki/Projective_linear_group#Exceptional_isomorphisms
    .. [3] https://en.wikipedia.org/wiki/Automorphisms_of_the_symmetric_and_alternating_groups#PGL%282,5%29

    """
    def elts_by_order(G):
        """Sort the elements of a group by their order. """
        elts = defaultdict(list)
        for g in G.elements:
            elts[g.order()].append(g)
        return elts

    def order_profile(G, name=None):
        """Determine how many elements a group has, of each order. """
        elts = elts_by_order(G)
        profile = {o:len(e) for o, e in elts.items()}
        if name:
            print(f'{name}: ' + ' '.join(f'{len(profile[r])}@{r}' for r in sorted(profile.keys())))
        return profile

    S6 = SymmetricGroup(6)
    A6 = AlternatingGroup(6)
    S6_by_order = elts_by_order(S6)

    def search(existing_gens, needed_gen_orders, order, alt=None, profile=None, anti_profile=None):
        """
        Find a transitive subgroup of S6.

        Parameters
        ==========

        existing_gens : list of Permutation
            Optionally empty list of generators that must be in the group.

        needed_gen_orders : list of positive int
            Nonempty list of the orders of the additional generators that are
            to be found.

        order: int
            The order of the group being sought.

        alt: bool, None
            If True, require the group to be contained in A6.
            If False, require the group not to be contained in A6.

        profile : dict
            If given, the group's order profile must equal this.

        anti_profile : dict
            If given, the group's order profile must *not* equal this.

        """
        for gens in itertools.product(*[S6_by_order[n] for n in needed_gen_orders]):
            if len(set(gens)) < len(gens):
                continue
            G = PermutationGroup(existing_gens + list(gens))
            if G.order() == order and G.is_transitive():
                if alt is not None and G.is_subgroup(A6) != alt:
                    continue
                if profile and order_profile(G) != profile:
                    continue
                if anti_profile and order_profile(G) == anti_profile:
                    continue
                return G

    def match_known_group(G, alt=None):
        needed = [g.order() for g in G.generators]
        return search([], needed, G.order(), alt=alt, profile=order_profile(G))

    found = {}

    def finish_up(name, G):
        found[name] = G
        if print_report:
            print("=" * 40)
            print(f"{name}:")
            print(G.generators)

    if S6TransitiveSubgroups.A4 in targets or S6TransitiveSubgroups.A4xC2 in targets:
        A4_in_S6 = match_known_group(AlternatingGroup(4))
        finish_up(S6TransitiveSubgroups.A4, A4_in_S6)

    if S6TransitiveSubgroups.S4m in targets or S6TransitiveSubgroups.S4xC2 in targets:
        S4m_in_S6 = match_known_group(SymmetricGroup(4), alt=False)
        finish_up(S6TransitiveSubgroups.S4m, S4m_in_S6)

    if S6TransitiveSubgroups.S4p in targets:
        S4p_in_S6 = match_known_group(SymmetricGroup(4), alt=True)
        finish_up(S6TransitiveSubgroups.S4p, S4p_in_S6)

    if S6TransitiveSubgroups.A4xC2 in targets:
        A4xC2_in_S6 = search(A4_in_S6.generators, [2], 24, anti_profile=order_profile(SymmetricGroup(4)))
        finish_up(S6TransitiveSubgroups.A4xC2, A4xC2_in_S6)

    if S6TransitiveSubgroups.S4xC2 in targets:
        S4xC2_in_S6 = search(S4m_in_S6.generators, [2], 48)
        finish_up(S6TransitiveSubgroups.S4xC2, S4xC2_in_S6)

    # For the normal factor N = C3^2 in any of the G_n subgroups, we take one
    # obvious instance of C3^2 in S6:
    N_gens = [Permutation(5)(0, 1, 2), Permutation(5)(3, 4, 5)]

    if S6TransitiveSubgroups.G18 in targets:
        G18_in_S6 = search(N_gens, [2], 18)
        finish_up(S6TransitiveSubgroups.G18, G18_in_S6)

    if S6TransitiveSubgroups.G36m in targets:
        G36m_in_S6 = search(N_gens, [2, 2], 36, alt=False)
        finish_up(S6TransitiveSubgroups.G36m, G36m_in_S6)

    if S6TransitiveSubgroups.G36p in targets:
        G36p_in_S6 = search(N_gens, [4], 36, alt=True)
        finish_up(S6TransitiveSubgroups.G36p, G36p_in_S6)

    if S6TransitiveSubgroups.G72 in targets:
        G72_in_S6 = search(N_gens, [4, 2], 72)
        finish_up(S6TransitiveSubgroups.G72, G72_in_S6)

    # The PSL2(F5) and PGL2(F5) subgroups are isomorphic to A5 and S5, resp.

    if S6TransitiveSubgroups.PSL2F5 in targets:
        PSL2F5_in_S6 = match_known_group(AlternatingGroup(5))
        finish_up(S6TransitiveSubgroups.PSL2F5, PSL2F5_in_S6)

    if S6TransitiveSubgroups.PGL2F5 in targets:
        PGL2F5_in_S6 = match_known_group(SymmetricGroup(5))
        finish_up(S6TransitiveSubgroups.PGL2F5, PGL2F5_in_S6)

    # There is little need to "search" for any of the groups C6, S3, D6, A6,
    # or S6, since they all have obvious realizations within S6. However, we
    # support them here just in case a random representation is desired.

    if S6TransitiveSubgroups.C6 in targets:
        C6 = match_known_group(CyclicGroup(6))
        finish_up(S6TransitiveSubgroups.C6, C6)

    if S6TransitiveSubgroups.S3 in targets:
        S3 = match_known_group(SymmetricGroup(3))
        finish_up(S6TransitiveSubgroups.S3, S3)

    if S6TransitiveSubgroups.D6 in targets:
        D6 = match_known_group(DihedralGroup(6))
        finish_up(S6TransitiveSubgroups.D6, D6)

    if S6TransitiveSubgroups.A6 in targets:
        A6 = match_known_group(A6)
        finish_up(S6TransitiveSubgroups.A6, A6)

    if S6TransitiveSubgroups.S6 in targets:
        S6 = match_known_group(S6)
        finish_up(S6TransitiveSubgroups.S6, S6)

    return found
<|endoftext|>from itertools import chain, combinations

from sympy.external.gmpy import gcd
from sympy.ntheory.factor_ import factorint
from sympy.utilities.misc import as_int


def _is_nilpotent_number(factors: dict) -> bool:
    """ Check whether `n` is a nilpotent number.
    Note that ``factors`` is a prime factorization of `n`.

    This is a low-level helper for ``is_nilpotent_number``, for internal use.
    """
    for p in factors.keys():
        for q, e in factors.items():
            # We want to calculate
            # any(pow(q, k, p) == 1 for k in range(1, e + 1))
            m = 1
            for _ in range(e):
                m = m*q % p
                if m == 1:
                    return False
    return True


def is_nilpotent_number(n) -> bool:
    """
    Check whether `n` is a nilpotent number. A number `n` is said to be
    nilpotent if and only if every finite group of order `n` is nilpotent.
    For more information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_nilpotent_number
    >>> from sympy import randprime
    >>> is_nilpotent_number(21)
    False
    >>> is_nilpotent_number(randprime(1, 30)**12)
    True

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A056867

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    return _is_nilpotent_number(factorint(n))


def is_abelian_number(n) -> bool:
    """
    Check whether `n` is an abelian number. A number `n` is said to be abelian
    if and only if every finite group of order `n` is abelian. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_abelian_number
    >>> from sympy import randprime
    >>> is_abelian_number(4)
    True
    >>> is_abelian_number(randprime(1, 2000)**2)
    True
    >>> is_abelian_number(60)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A051532

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    return all(e < 3 for e in factors.values()) and _is_nilpotent_number(factors)


def is_cyclic_number(n) -> bool:
    """
    Check whether `n` is a cyclic number. A number `n` is said to be cyclic
    if and only if every finite group of order `n` is cyclic. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_cyclic_number
    >>> from sympy import randprime
    >>> is_cyclic_number(15)
    True
    >>> is_cyclic_number(randprime(1, 2000)**2)
    False
    >>> is_cyclic_number(4)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A003277

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    return all(e == 1 for e in factors.values()) and _is_nilpotent_number(factors)


def _holder_formula(prime_factors):
    r""" Number of groups of order `n`.
    where `n` is squarefree and its prime factors are ``prime_factors``.
    i.e., ``n == math.prod(prime_factors)``

    Explanation
    ===========

    When `n` is squarefree, the number of groups of order `n` is expressed by

    .. math ::
        \sum_{d \mid n} \prod_p \frac{p^{c(p, d)} - 1}{p - 1}

    where `n=de`, `p` is the prime factor of `e`,
    and `c(p, d)` is the number of prime factors `q` of `d` such that `q \equiv 1 \pmod{p}` [2]_.

    The formula is elegant, but can be improved when implemented as an algorithm.
    Since `n` is assumed to be squarefree, the divisor `d` of `n` can be identified with the power set of prime factors.
    We let `N` be the set of prime factors of `n`.
    `F = \{p \in N : \forall q \in N, q \not\equiv 1 \pmod{p} \}, M = N \setminus F`, we have the following.

    .. math ::
        \sum_{d \in 2^{M}} \prod_{p \in M \setminus d} \frac{p^{c(p, F \cup d)} - 1}{p - 1}

    Practically, many prime factors are expected to be members of `F`, thus reducing computation time.

    Parameters
    ==========

    prime_factors : set
        The set of prime factors of ``n``. where `n` is squarefree.

    Returns
    =======

    int : Number of groups of order ``n``

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import _holder_formula
    >>> _holder_formula({2}) # n = 2
    1
    >>> _holder_formula({2, 3}) # n = 2*3 = 6
    2

    See Also
    ========

    groups_count

    References
    ==========

    .. [1] Otto Holder, Die Gruppen der Ordnungen p^3, pq^2, pqr, p^4,
           Math. Ann. 43 pp. 301-412 (1893).
           http://dx.doi.org/10.1007/BF01443651
    .. [2] John H. Conway, Heiko Dietrich and E.A. O'Brien,
           Counting groups: gnus, moas and other exotica
           The Mathematical Intelligencer 30, 6-15 (2008)
           https://doi.org/10.1007/BF02985731

    """
    F = {p for p in prime_factors if all(q % p != 1 for q in prime_factors)}
    M = prime_factors - F

    s = 0
    powerset = chain.from_iterable(combinations(M, r) for r in range(len(M)+1))
    for ps in powerset:
        ps = set(ps)
        prod = 1
        for p in M - ps:
            c = len([q for q in F | ps if q % p == 1])
            prod *= (p**c - 1) // (p - 1)
            if not prod:
                break
        s += prod
    return s


def groups_count(n):
    r""" Number of groups of order `n`.
    In [1]_, ``gnu(n)`` is given, so we follow this notation here as well.

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer

    Returns
    =======

    int : ``gnu(n)``

    Raises
    ======

    ValueError
        Number of groups of order ``n`` is unknown or not implemented.
        For example, gnu(`2^{11}`) is not yet known.
        On the other hand, gnu(99) is known to be 2,
        but this has not yet been implemented in this function.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import groups_count
    >>> groups_count(3) # There is only one cyclic group of order 3
    1
    >>> # There are two groups of order 10: the cyclic group and the dihedral group
    >>> groups_count(10)
    2

    See Also
    ========

    is_cyclic_number
        `n` is cyclic iff gnu(n) = 1

    References
    ==========

    .. [1] John H. Conway, Heiko Dietrich and E.A. O'Brien,
           Counting groups: gnus, moas and other exotica
           The Mathematical Intelligencer 30, 6-15 (2008)
           https://doi.org/10.1007/BF02985731
    .. [2] https://oeis.org/A000001

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    if len(factors) == 1:
        (p, e) = list(factors.items())[0]
        if p == 2:
            A000679 = [1, 1, 2, 5, 14, 51, 267, 2328, 56092, 10494213, 49487367289]
            if e < len(A000679):
                return A000679[e]
        if p == 3:
            A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
            if e < len(A090091):
                return A090091[e]
        if e <= 2: # gnu(p) = 1, gnu(p**2) = 2
            return e
        if e == 3: # gnu(p**3) = 5
            return 5
        if e == 4: # if p is an odd prime, gnu(p**4) = 15
            return 15
        if e == 5: # if p >= 5, gnu(p**5) is expressed by the following equation
            return 61 + 2*p + 2*gcd(p-1, 3) + gcd(p-1, 4)
        if e == 6: # if p >= 6, gnu(p**6) is expressed by the following equation
            return 3*p**2 + 39*p + 344 +\
                  24*gcd(p-1, 3) + 11*gcd(p-1, 4) + 2*gcd(p-1, 5)
        if e == 7: # if p >= 7, gnu(p**7) is expressed by the following equation
            if p == 5:
                return 34297
            return 3*p**5 + 12*p**4 + 44*p**3 + 170*p**2 + 707*p + 2455 +\
                  (4*p**2 + 44*p + 291)*gcd(p-1, 3) + (p**2 + 19*p + 135)*gcd(p-1, 4) + \
                  (3*p + 31)*gcd(p-1, 5) + 4*gcd(p-1, 7) + 5*gcd(p-1, 8) + gcd(p-1, 9)
    if any(e > 1 for e in factors.values()): # n is not squarefree
        # some known values for small n that have more than 1 factor and are not square free (https://oeis.org/A000001)
        small = {12: 5, 18: 5, 20: 5, 24: 15, 28: 4, 36: 14, 40: 14, 44: 4, 45: 2, 48: 52,
                50: 5, 52: 5, 54: 15, 56: 13, 60: 13, 63: 4, 68: 5, 72: 50, 75: 3, 76: 4,
                80: 52, 84: 15, 88: 12, 90: 10, 92: 4}
        if n in small:
            return small[n]
        raise ValueError("Number of groups of order n is unknown or not implemented")
    if len(factors) == 2: # n is squarefree semiprime
        p, q = sorted(factors.keys())
        return 2 if q % p == 1 else 1
    return _holder_formula(set(factors.keys()))
<|endoftext|>from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting

from itertools import chain, product
from bisect import bisect_left


###############################################################################
#                           COSET TABLE                                       #
###############################################################################

class CosetTable(DefaultPrinting):
    # coset_table: Mathematically a coset table
    #               represented using a list of lists
    # alpha: Mathematically a coset (precisely, a live coset)
    #       represented by an integer between i with 1 <= i <= n
    #       alpha in c
    # x: Mathematically an element of "A" (set of generators and
    #   their inverses), represented using "FpGroupElement"
    # fp_grp: Finitely Presented Group with < X|R > as presentation.
    # H: subgroup of fp_grp.
    # NOTE: We start with H as being only a list of words in generators
    #       of "fp_grp". Since `.subgroup` method has not been implemented.

    r"""

    Properties
    ==========

    [1] `0 \in \Omega` and `\tau(1) = \epsilon`
    [2] `\alpha^x = \beta \Leftrightarrow \beta^{x^{-1}} = \alpha`
    [3] If `\alpha^x = \beta`, then `H \tau(\alpha)x = H \tau(\beta)`
    [4] `\forall \alpha \in \Omega, 1^{\tau(\alpha)} = \alpha`

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
           Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
           "Implementation and Analysis of the Todd-Coxeter Algorithm"

    """
    # default limit for the number of cosets allowed in a
    # coset enumeration.
    coset_table_max_limit = 4096000
    # limit for the current instance
    coset_table_limit = None
    # maximum size of deduction stack above or equal to
    # which it is emptied
    max_stack_size = 100

    def __init__(self, fp_grp, subgroup, max_cosets=None):
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        self.fp_group = fp_grp
        self.subgroup = subgroup
        self.coset_table_limit = max_cosets
        # "p" is setup independent of Omega and n
        self.p = [0]
        # a list of the form `[gen_1, gen_1^{-1}, ... , gen_k, gen_k^{-1}]`
        self.A = list(chain.from_iterable((gen, gen**-1) \
                for gen in self.fp_group.generators))
        #P[alpha, x] Only defined when alpha^x is defined.
        self.P = [[None]*len(self.A)]
        # the mathematical coset table which is a list of lists
        self.table = [[None]*len(self.A)]
        self.A_dict = {x: self.A.index(x) for x in self.A}
        self.A_dict_inv = {}
        for x, index in self.A_dict.items():
            if index % 2 == 0:
                self.A_dict_inv[x] = self.A_dict[x] + 1
            else:
                self.A_dict_inv[x] = self.A_dict[x] - 1
        # used in the coset-table based method of coset enumeration. Each of
        # the element is called a "deduction" which is the form (alpha, x) whenever
        # a value is assigned to alpha^x during a definition or "deduction process"
        self.deduction_stack = []
        # Attributes for modified methods.
        H = self.subgroup
        self._grp = free_group(', ' .join(["a_%d" % i for i in range(len(H))]))[0]
        self.P = [[None]*len(self.A)]
        self.p_p = {}

    @property
    def omega(self):
        """Set of live cosets. """
        return [coset for coset in range(len(self.p)) if self.p[coset] == coset]

    def copy(self):
        """
        Return a shallow copy of Coset Table instance ``self``.

        """
        self_copy = self.__class__(self.fp_group, self.subgroup)
        self_copy.table = [list(perm_rep) for perm_rep in self.table]
        self_copy.p = list(self.p)
        self_copy.deduction_stack = list(self.deduction_stack)
        return self_copy

    def __str__(self):
        return "Coset Table on %s with %s as subgroup generators" \
                % (self.fp_group, self.subgroup)

    __repr__ = __str__

    @property
    def n(self):
        """The number `n` represents the length of the sublist containing the
        live cosets.

        """
        if not self.table:
            return 0
        return max(self.omega) + 1

    # Pg. 152 [1]
    def is_complete(self):
        r"""
        The coset table is called complete if it has no undefined entries
        on the live cosets; that is, `\alpha^x` is defined for all
        `\alpha \in \Omega` and `x \in A`.

        """
        return not any(None in self.table[coset] for coset in self.omega)

    # Pg. 153 [1]
    def define(self, alpha, x, modified=False):
        r"""
        This routine is used in the relator-based strategy of Todd-Coxeter
        algorithm if some `\alpha^x` is undefined. We check whether there is
        space available for defining a new coset. If there is enough space
        then we remedy this by adjoining a new coset `\beta` to `\Omega`
        (i.e to set of live cosets) and put that equal to `\alpha^x`, then
        make an assignment satisfying Property[1]. If there is not enough space
        then we halt the Coset Table creation. The maximum amount of space that
        can be used by Coset Table can be manipulated using the class variable
        ``CosetTable.coset_table_max_limit``.

        See Also
        ========

        define_c

        """
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            # abort the further generation of cosets
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        table.append([None]*len(A))
        self.P.append([None]*len(self.A))
        # beta is the new coset generated
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        # P[alpha][x] = epsilon, P[beta][x**-1] = epsilon
        if modified:
            self.P[alpha][self.A_dict[x]] = self._grp.identity
            self.P[beta][self.A_dict_inv[x]] = self._grp.identity
            self.p_p[beta] = self._grp.identity

    def define_c(self, alpha, x):
        r"""
        A variation of ``define`` routine, described on Pg. 165 [1], used in
        the coset table-based strategy of Todd-Coxeter algorithm. It differs
        from ``define`` routine in that for each definition it also adds the
        tuple `(\alpha, x)` to the deduction stack.

        See Also
        ========

        define

        """
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            # abort the further generation of cosets
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        table.append([None]*len(A))
        # beta is the new coset generated
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        # append to deduction stack
        self.deduction_stack.append((alpha, x))

    def scan_c(self, alpha, word):
        """
        A variation of ``scan`` routine, described on pg. 165 of [1], which
        puts at tuple, whenever a deduction occurs, to deduction stack.

        See Also
        ========

        scan, scan_check, scan_and_fill, scan_and_fill_c

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        # list of union of generators and their inverses
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            if f != b:
                self.coincidence_c(f, b)
            return
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            # we have an incorrect completed scan with coincidence f ~ b
            # run the "coincidence" routine
            self.coincidence_c(f, b)
        elif j == i:
            # deduction process
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
            self.deduction_stack.append((f, word[i]))
        # otherwise scan is incomplete and yields no information

    # alpha, beta coincide, i.e. alpha, beta represent the pair of cosets where
    # coincidence occurs
    def coincidence_c(self, alpha, beta):
        """
        A variation of ``coincidence`` routine used in the coset-table based
        method of coset enumeration. The only difference being on addition of
        a new coset in coset table(i.e new coset introduction), then it is
        appended to ``deduction_stack``.

        See Also
        ========

        coincidence

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        # behaves as a queue
        q = []
        self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    # only line of difference from ``coincidence`` routine
                    self.deduction_stack.append((delta, x**-1))
                    mu = self.rep(gamma)
                    nu = self.rep(delta)
                    if table[mu][A_dict[x]] is not None:
                        self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu

    def scan(self, alpha, word, y=None, fill=False, modified=False):
        r"""
        ``scan`` performs a scanning process on the input ``word``.
        It first locates the largest prefix ``s`` of ``word`` for which
        `\alpha^s` is defined (i.e is not ``None``), ``s`` may be empty. Let
        ``word=sv``, let ``t`` be the longest suffix of ``v`` for which
        `\alpha^{t^{-1}}` is defined, and let ``v=ut``. Then three
        possibilities are there:

        1. If ``t=v``, then we say that the scan completes, and if, in addition
        `\alpha^s = \alpha^{t^{-1}}`, then we say that the scan completes
        correctly.

        2. It can also happen that scan does not complete, but `|u|=1`; that
        is, the word ``u`` consists of a single generator `x \in A`. In that
        case, if `\alpha^s = \beta` and `\alpha^{t^{-1}} = \gamma`, then we can
        set `\beta^x = \gamma` and `\gamma^{x^{-1}} = \beta`. These assignments
        are known as deductions and enable the scan to complete correctly.

        3. See ``coicidence`` routine for explanation of third condition.

        Notes
        =====

        The code for the procedure of scanning `\alpha \in \Omega`
        under `w \in A*` is defined on pg. 155 [1]

        See Also
        ========

        scan_c, scan_check, scan_and_fill, scan_and_fill_c

        Scan and Fill
        =============

        Performed when the default argument fill=True.

        Modified Scan
        =============

        Performed when the default argument modified=True

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        b_p = y
        if modified:
            f_p = self._grp.identity
        flag = 0
        while fill or flag == 0:
            flag = 1
            while i <= j and table[f][A_dict[word[i]]] is not None:
                if modified:
                    f_p = f_p*self.P[f][A_dict[word[i]]]
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    if modified:
                        self.modified_coincidence(f, b, f_p**-1*y)
                    else:
                        self.coincidence(f, b)
                return
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                if modified:
                    b_p = b_p*self.P[b][self.A_dict_inv[word[j]]]
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                # we have an incorrect completed scan with coincidence f ~ b
                # run the "coincidence" routine
                if modified:
                    self.modified_coincidence(f, b, f_p**-1*b_p)
                else:
                    self.coincidence(f, b)
            elif j == i:
                # deduction process
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                if modified:
                    self.P[f][self.A_dict[word[i]]] = f_p**-1*b_p
                    self.P[b][self.A_dict_inv[word[i]]] = b_p**-1*f_p
                return
            elif fill:
                self.define(f, word[i], modified=modified)
            # otherwise scan is incomplete and yields no information

    # used in the low-index subgroups algorithm
    def scan_check(self, alpha, word):
        r"""
        Another version of ``scan`` routine, described on, it checks whether
        `\alpha` scans correctly under `word`, it is a straightforward
        modification of ``scan``. ``scan_check`` returns ``False`` (rather than
        calling ``coincidence``) if the scan completes incorrectly; otherwise
        it returns ``True``.

        See Also
        ========

        scan, scan_c, scan_and_fill, scan_and_fill_c

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            return f == b
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            # we have an incorrect completed scan with coincidence f ~ b
            # return False, instead of calling coincidence routine
            return False
        elif j == i:
            # deduction process
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
        return True

    def merge(self, k, lamda, q, w=None, modified=False):
        """
        Merge two classes with representatives ``k`` and ``lamda``, described
        on Pg. 157 [1] (for pseudocode), start by putting ``p[k] = lamda``.
        It is more efficient to choose the new representative from the larger
        of the two classes being merged, i.e larger among ``k`` and ``lamda``.
        procedure ``merge`` performs the merging operation, adds the deleted
        class representative to the queue ``q``.

        Parameters
        ==========

        'k', 'lamda' being the two class representatives to be merged.

        Notes
        =====

        Pg. 86-87 [1] contains a description of this method.

        See Also
        ========

        coincidence, rep

        """
        p = self.p
        rep = self.rep
        phi = rep(k, modified=modified)
        psi = rep(lamda, modified=modified)
        if phi != psi:
            mu = min(phi, psi)
            v = max(phi, psi)
            p[v] = mu
            if modified:
                if v == phi:
                    self.p_p[phi] = self.p_p[k]**-1*w*self.p_p[lamda]
                else:
                    self.p_p[psi] = self.p_p[lamda]**-1*w**-1*self.p_p[k]
            q.append(v)

    def rep(self, k, modified=False):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`, as for ``self`` only array ``p`` is used

        Returns
        =======

        Representative of the class containing ``k``.

        Returns the representative of `\sim` class containing ``k``, it also
        makes some modification to array ``p`` of ``self`` to ease further
        computations, described on Pg. 157 [1].

        The information on classes under `\sim` is stored in array `p` of
        ``self`` argument, which will always satisfy the property:

        `p[\alpha] \sim \alpha` and `p[\alpha]=\alpha \iff \alpha=rep(\alpha)`
        `\forall \in [0 \ldots n-1]`.

        So, for `\alpha \in [0 \ldots n-1]`, we find `rep(self, \alpha)` by
        continually replacing `\alpha` by `p[\alpha]` until it becomes
        constant (i.e satisfies `p[\alpha] = \alpha`):w

        To increase the efficiency of later ``rep`` calculations, whenever we
        find `rep(self, \alpha)=\beta`, we set
        `p[\gamma] = \beta \forall \gamma \in p-chain` from `\alpha` to `\beta`

        Notes
        =====

        ``rep`` routine is also described on Pg. 85-87 [1] in Atkinson's
        algorithm, this results from the fact that ``coincidence`` routine
        introduces functionality similar to that introduced by the
        ``minimal_block`` routine on Pg. 85-87 [1].

        See Also
        ========

        coincidence, merge

        """
        p = self.p
        lamda = k
        rho = p[lamda]
        if modified:
            s = p[:]
        while rho != lamda:
            if modified:
                s[rho] = lamda
            lamda = rho
            rho = p[lamda]
        if modified:
            rho = s[lamda]
            while rho != k:
                mu = rho
                rho = s[mu]
                p[rho] = lamda
                self.p_p[rho] = self.p_p[rho]*self.p_p[mu]
        else:
            mu = k
            rho = p[mu]
            while rho != lamda:
                p[mu] = lamda
                mu = rho
                rho = p[mu]
        return lamda

    # alpha, beta coincide, i.e. alpha, beta represent the pair of cosets
    # where coincidence occurs
    def coincidence(self, alpha, beta, w=None, modified=False):
        r"""
        The third situation described in ``scan`` routine is handled by this
        routine, described on Pg. 156-161 [1].

        The unfortunate situation when the scan completes but not correctly,
        then ``coincidence`` routine is run. i.e when for some `i` with
        `1 \le i \le r+1`, we have `w=st` with `s = x_1 x_2 \dots x_{i-1}`,
        `t = x_i x_{i+1} \dots x_r`, and `\beta = \alpha^s` and
        `\gamma = \alpha^{t-1}` are defined but unequal. This means that
        `\beta` and `\gamma` represent the same coset of `H` in `G`. Described
        on Pg. 156 [1]. ``rep``

        See Also
        ========

        scan

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        # behaves as a queue
        q = []
        if modified:
            self.modified_merge(alpha, beta, w, q)
        else:
            self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    mu = self.rep(gamma, modified=modified)
                    nu = self.rep(delta, modified=modified)
                    if table[mu][A_dict[x]] is not None:
                        if modified:
                            v = self.p_p[delta]**-1*self.P[gamma][self.A_dict[x]]**-1
                            v = v*self.p_p[gamma]*self.P[mu][self.A_dict[x]]
                            self.modified_merge(nu, table[mu][self.A_dict[x]], v, q)
                        else:
                            self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        if modified:
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]
                            v = v*self.p_p[delta]*self.P[mu][self.A_dict_inv[x]]
                            self.modified_merge(mu, table[nu][self.A_dict_inv[x]], v, q)
                        else:
                            self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu
                        if modified:
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]*self.p_p[delta]
                            self.P[mu][self.A_dict[x]] = v
                            self.P[nu][self.A_dict_inv[x]] = v**-1

    # method used in the HLT strategy
    def scan_and_fill(self, alpha, word):
        """
        A modified version of ``scan`` routine used in the relator-based
        method of coset enumeration, described on pg. 162-163 [1], which
        follows the idea that whenever the procedure is called and the scan
        is incomplete then it makes new definitions to enable the scan to
        complete; i.e it fills in the gaps in the scan of the relator or
        subgroup generator.

        """
        self.scan(alpha, word, fill=True)

    def scan_and_fill_c(self, alpha, word):
        """
        A modified version of ``scan`` routine, described on Pg. 165 second
        para. [1], with modification similar to that of ``scan_anf_fill`` the
        only difference being it calls the coincidence procedure used in the
        coset-table based method i.e. the routine ``coincidence_c`` is used.

        See Also
        ========

        scan, scan_and_fill

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        r = len(word)
        f = alpha
        i = 0
        b = alpha
        j = r - 1
        # loop until it has filled the alpha row in the table.
        while True:
            # do the forward scanning
            while i <= j and table[f][A_dict[word[i]]] is not None:
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    self.coincidence_c(f, b)
                return
            # forward scan was incomplete, scan backwards
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                self.coincidence_c(f, b)
            elif j == i:
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                self.deduction_stack.append((f, word[i]))
            else:
                self.define_c(f, word[i])

    # method used in the HLT strategy
    def look_ahead(self):
        """
        When combined with the HLT method this is known as HLT+Lookahead
        method of coset enumeration, described on pg. 164 [1]. Whenever
        ``define`` aborts due to lack of space available this procedure is
        executed. This routine helps in recovering space resulting from
        "coincidence" of cosets.

        """
        R = self.fp_group.relators
        p = self.p
        # complete scan all relators under all cosets(obviously live)
        # without making new definitions
        for beta in self.omega:
            for w in R:
                self.scan(beta, w)
                if p[beta] < beta:
                    break

    # Pg. 166
    def process_deductions(self, R_c_x, R_c_x_inv):
        """
        Processes the deductions that have been pushed onto ``deduction_stack``,
        described on Pg. 166 [1] and is used in coset-table based enumeration.

        See Also
        ========

        deduction_stack

        """
        p = self.p
        table = self.table
        while len(self.deduction_stack) > 0:
            if len(self.deduction_stack) >= CosetTable.max_stack_size:
                self.look_ahead()
                del self.deduction_stack[:]
                continue
            else:
                alpha, x = self.deduction_stack.pop()
                if p[alpha] == alpha:
                    for w in R_c_x:
                        self.scan_c(alpha, w)
                        if p[alpha] < alpha:
                            break
            beta = table[alpha][self.A_dict[x]]
            if beta is not None and p[beta] == beta:
                for w in R_c_x_inv:
                    self.scan_c(beta, w)
                    if p[beta] < beta:
                        break

    def process_deductions_check(self, R_c_x, R_c_x_inv):
        """
        A variation of ``process_deductions``, this calls ``scan_check``
        wherever ``process_deductions`` calls ``scan``, described on Pg. [1].

        See Also
        ========

        process_deductions

        """
        table = self.table
        while len(self.deduction_stack) > 0:
            alpha, x = self.deduction_stack.pop()
            if not all(self.scan_check(alpha, w) for w in R_c_x):
                return False
            beta = table[alpha][self.A_dict[x]]
            if beta is not None:
                if not all(self.scan_check(beta, w) for w in R_c_x_inv):
                    return False
        return True

    def switch(self, beta, gamma):
        r"""Switch the elements `\beta, \gamma \in \Omega` of ``self``, used
        by the ``standardize`` procedure, described on Pg. 167 [1].

        See Also
        ========

        standardize

        """
        A = self.A
        A_dict = self.A_dict
        table = self.table
        for x in A:
            z = table[gamma][A_dict[x]]
            table[gamma][A_dict[x]] = table[beta][A_dict[x]]
            table[beta][A_dict[x]] = z
            for alpha in range(len(self.p)):
                if self.p[alpha] == alpha:
                    if table[alpha][A_dict[x]] == beta:
                        table[alpha][A_dict[x]] = gamma
                    elif table[alpha][A_dict[x]] == gamma:
                        table[alpha][A_dict[x]] = beta

    def standardize(self):
        r"""
        A coset table is standardized if when running through the cosets and
        within each coset through the generator images (ignoring generator
        inverses), the cosets appear in order of the integers
        `0, 1, \dots, n`. "Standardize" reorders the elements of `\Omega`
        such that, if we scan the coset table first by elements of `\Omega`
        and then by elements of A, then the cosets occur in ascending order.
        ``standardize()`` is used at the end of an enumeration to permute the
        cosets so that they occur in some sort of standard order.

        Notes
        =====

        procedure is described on pg. 167-168 [1], it also makes use of the
        ``switch`` routine to replace by smaller integer value.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")

        # Example 5.3 from [1]
        >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
        >>> C = coset_enumeration_r(f, [])
        >>> C.compress()
        >>> C.table
        [[1, 3, 1, 3], [2, 0, 2, 0], [3, 1, 3, 1], [0, 2, 0, 2]]
        >>> C.standardize()
        >>> C.table
        [[1, 2, 1, 2], [3, 0, 3, 0], [0, 3, 0, 3], [2, 1, 2, 1]]

        """
        A = self.A
        A_dict = self.A_dict
        gamma = 1
        for alpha, x in product(range(self.n), A):
            beta = self.table[alpha][A_dict[x]]
            if beta >= gamma:
                if beta > gamma:
                    self.switch(gamma, beta)
                gamma += 1
                if gamma == self.n:
                    return

    # Compression of a Coset Table
    def compress(self):
        """Removes the non-live cosets from the coset table, described on
        pg. 167 [1].

        """
        gamma = -1
        A = self.A
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        chi = tuple([i for i in range(len(self.p)) if self.p[i] != i])
        for alpha in self.omega:
            gamma += 1
            if gamma != alpha:
                # replace alpha by gamma in coset table
                for x in A:
                    beta = table[alpha][A_dict[x]]
                    table[gamma][A_dict[x]] = beta
                    # XXX: The line below uses == rather than = which means
                    # that it has no effect. It is not clear though if it is
                    # correct simply to delete the line or to change it to
                    # use =. Changing it causes some tests to fail.
                    #
                    # https://github.com/sympy/sympy/issues/27633
                    table[beta][A_dict_inv[x]] == gamma # noqa: B015
        # all the cosets in the table are live cosets
        self.p = list(range(gamma + 1))
        # delete the useless columns
        del table[len(self.p):]
        # re-define values
        for row in table:
            for j in range(len(self.A)):
                row[j] -= bisect_left(chi, row[j])

    def conjugates(self, R):
        R_c = list(chain.from_iterable((rel.cyclic_conjugates(), \
                (rel**-1).cyclic_conjugates()) for rel in R))
        R_set = set()
        for conjugate in R_c:
            R_set = R_set.union(conjugate)
        R_c_list = []
        for x in self.A:
            r = {word for word in R_set if word[0] == x}
            R_c_list.append(r)
            R_set.difference_update(r)
        return R_c_list

    def coset_representative(self, coset):
        '''
        Compute the coset representative of a given coset.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
        >>> C = coset_enumeration_r(f, [x])
        >>> C.compress()
        >>> C.table
        [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
        >>> C.coset_representative(0)
        <identity>
        >>> C.coset_representative(1)
        y
        >>> C.coset_representative(2)
        y**-1

        '''
        for x in self.A:
            gamma = self.table[coset][self.A_dict[x]]
            if coset == 0:
                return self.fp_group.identity
            if gamma < coset:
                return self.coset_representative(gamma)*x**-1

    ##############################
    #      Modified Methods      #
    ##############################

    def modified_define(self, alpha, x):
        r"""
        Define a function p_p from from [1..n] to A* as
        an additional component of the modified coset table.

        Parameters
        ==========

        \alpha \in \Omega
        x \in A*

        See Also
        ========

        define

        """
        self.define(alpha, x, modified=True)

    def modified_scan(self, alpha, w, y, fill=False):
        r"""
        Parameters
        ==========
        \alpha \in \Omega
        w \in A*
        y \in (YUY^-1)
        fill -- `modified_scan_and_fill` when set to True.

        See Also
        ========

        scan
        """
        self.scan(alpha, w, y=y, fill=fill, modified=True)

    def modified_scan_and_fill(self, alpha, w, y):
        self.modified_scan(alpha, w, y, fill=True)

    def modified_merge(self, k, lamda, w, q):
        r"""
        Parameters
        ==========

        'k', 'lamda' -- the two class representatives to be merged.
        q -- queue of length l of elements to be deleted from `\Omega` *.
        w -- Word in (YUY^-1)

        See Also
        ========

        merge
        """
        self.merge(k, lamda, q, w=w, modified=True)

    def modified_rep(self, k):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`

        See Also
        ========

        rep
        """
        self.rep(k, modified=True)

    def modified_coincidence(self, alpha, beta, w):
        r"""
        Parameters
        ==========

        A coincident pair `\alpha, \beta \in \Omega, w \in Y \cup Y^{-1}`

        See Also
        ========

        coincidence

        """
        self.coincidence(alpha, beta, w=w, modified=True)

###############################################################################
#                           COSET ENUMERATION                                 #
###############################################################################

# relator-based method
def coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                                    incomplete=False, modified=False):
    """
    This is easier of the two implemented methods of coset enumeration.
    and is often called the HLT method, after Hazelgrove, Leech, Trotter
    The idea is that we make use of ``scan_and_fill`` makes new definitions
    whenever the scan is incomplete to enable the scan to complete; this way
    we fill in the gaps in the scan of the relator or subgroup generator,
    that's why the name relator-based method.

    An instance of `CosetTable` for `fp_grp` can be passed as the keyword
    argument `draft` in which case the coset enumeration will start with
    that instance and attempt to complete it.

    When `incomplete` is `True` and the function is unable to complete for
    some reason, the partially complete table will be returned.

    # TODO: complete the docstring

    See Also
    ========

    scan_and_fill,

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
    >>> F, x, y = free_group("x, y")

    # Example 5.1 from [1]
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_r(f, [x])
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 1, 2]
    [1, 1, 2, 0]
    [2, 2, 0, 1]
    >>> C.p
    [0, 1, 2, 1, 1]

    # Example from exercises Q2 [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> C = coset_enumeration_r(f, [])
    >>> C.compress(); C.standardize()
    >>> C.table
    [[1, 2, 3, 4],
    [5, 0, 6, 7],
    [0, 5, 7, 6],
    [7, 6, 5, 0],
    [6, 7, 0, 5],
    [2, 1, 4, 3],
    [3, 4, 2, 1],
    [4, 3, 1, 2]]

    # Example 5.2
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**3])
    >>> Y = [x*y]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 1, 2, 1]
    [0, 0, 0, 2]
    [3, 3, 1, 0]
    [2, 2, 3, 3]

    # Example 5.3
    >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
    >>> Y = []
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 3, 1, 3]
    [2, 0, 2, 0]
    [3, 1, 3, 1]
    [0, 2, 0, 2]

    # Example 5.4
    >>> F, a, b, c, d, e = free_group("a, b, c, d, e")
    >>> f = FpGroup(F, [a*b*c**-1, b*c*d**-1, c*d*e**-1, d*e*a**-1, e*a*b**-1])
    >>> Y = [a]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # example of "compress" method
    >>> C.compress()
    >>> C.table
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Exercises Pg. 161, Q2.
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> Y = []
    >>> C = coset_enumeration_r(f, Y)
    >>> C.compress()
    >>> C.standardize()
    >>> C.table
    [[1, 2, 3, 4],
    [5, 0, 6, 7],
    [0, 5, 7, 6],
    [7, 6, 5, 0],
    [6, 7, 0, 5],
    [2, 1, 4, 3],
    [3, 4, 2, 1],
    [4, 3, 1, 2]]

    # John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
    # Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490
    # from 1973chwd.pdf
    # Table 1. Ex. 1
    >>> F, r, s, t = free_group("r, s, t")
    >>> E1 = FpGroup(F, [t**-1*r*t*r**-2, r**-1*s*r*s**-2, s**-1*t*s*t**-2])
    >>> C = coset_enumeration_r(E1, [r])
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 0, 0, 0, 0]

    Ex. 2
    >>> F, a, b = free_group("a, b")
    >>> Cox = FpGroup(F, [a**6, b**6, (a*b)**2, (a**2*b**2)**2, (a**3*b**3)**5])
    >>> C = coset_enumeration_r(Cox, [a])
    >>> index = 0
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         index += 1
    >>> index
    500

    # Ex. 3
    >>> F, a, b = free_group("a, b")
    >>> B_2_4 = FpGroup(F, [a**4, b**4, (a*b)**4, (a**-1*b)**4, (a**2*b)**4, \
            (a*b**2)**4, (a**2*b**2)**4, (a**-1*b*a*b)**4, (a*b**-1*a*b)**4])
    >>> C = coset_enumeration_r(B_2_4, [a])
    >>> index = 0
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         index += 1
    >>> index
    1024

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    # 1. Initialize a coset table C for < X|R >
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    # Define coset table methods.
    if modified:
        _scan_and_fill = C.modified_scan_and_fill
        _define = C.modified_define
    else:
        _scan_and_fill = C.scan_and_fill
        _define = C.define
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
    R = fp_grp.relators
    A_dict = C.A_dict
    p = C.p
    for i in range(len(Y)):
        if modified:
            _scan_and_fill(0, Y[i], C._grp.generators[i])
        else:
            _scan_and_fill(0, Y[i])
    alpha = 0
    while alpha < C.n:
        if p[alpha] == alpha:
            try:
                for w in R:
                    if modified:
                        _scan_and_fill(alpha, w, C._grp.identity)
                    else:
                        _scan_and_fill(alpha, w)
                    # if alpha was eliminated during the scan then break
                    if p[alpha] < alpha:
                        break
                if p[alpha] == alpha:
                    for x in A_dict:
                        if C.table[alpha][A_dict[x]] is None:
                            _define(alpha, x)
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C

def modified_coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                                    incomplete=False):
    r"""
    Introduce a new set of symbols y \in Y that correspond to the
    generators of the subgroup. Store the elements of Y as a
    word P[\alpha, x] and compute the coset table similar to that of
    the regular coset enumeration methods.

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.coset_table import modified_coset_enumeration_r
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = modified_coset_enumeration_r(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [None, 1, None, None], [1, 3, None, None]]

    See Also
    ========

    coset_enumertation_r

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.,
           "Handbook of Computational Group Theory",
           Section 5.3.2
    """
    return coset_enumeration_r(fp_grp, Y, max_cosets=max_cosets, draft=draft,
                             incomplete=incomplete, modified=True)

# Pg. 166
# coset-table based method
def coset_enumeration_c(fp_grp, Y, max_cosets=None, draft=None,
                                                incomplete=False):
    """
    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_c
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_c(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]

    """
    # Initialize a coset table C for < X|R >
    X = fp_grp.generators
    R = fp_grp.relators
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
        C.deduction_stack = draft.deduction_stack
        for alpha, x in product(range(len(C.table)), X):
            if C.table[alpha][C.A_dict[x]] is not None:
                C.deduction_stack.append((alpha, x))
    A = C.A
    # replace all the elements by cyclic reductions
    R_cyc_red = [rel.identity_cyclic_reduction() for rel in R]
    R_c = list(chain.from_iterable((rel.cyclic_conjugates(), (rel**-1).cyclic_conjugates()) \
            for rel in R_cyc_red))
    R_set = set()
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    # a list of subsets of R_c whose words start with "x".
    R_c_list = []
    for x in C.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    for w in Y:
        C.scan_and_fill_c(0, w)
    for x in A:
        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
    alpha = 0
    while alpha < len(C.table):
        if C.p[alpha] == alpha:
            try:
                for x in C.A:
                    if C.p[alpha] != alpha:
                        break
                    if C.table[alpha][C.A_dict[x]] is None:
                        C.define_c(alpha, x)
                        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C
<|endoftext|>from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int

rmul = Perm.rmul


class Polyhedron(Basic):
    """
    Represents the polyhedral symmetry group (PSG).

    Explanation
    ===========

    The PSG is one of the symmetry groups of the Platonic solids.
    There are three polyhedral groups: the tetrahedral group
    of order 12, the octahedral group of order 24, and the
    icosahedral group of order 60.

    All doctests have been given in the docstring of the
    constructor of the object.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PolyhedralGroup.html

    """
    _edges = None

    def __new__(cls, corners, faces=(), pgroup=()):
        """
        The constructor of the Polyhedron group object.

        Explanation
        ===========

        It takes up to three parameters: the corners, faces, and
        allowed transformations.

        The corners/vertices are entered as a list of arbitrary
        expressions that are used to identify each vertex.

        The faces are entered as a list of tuples of indices; a tuple
        of indices identifies the vertices which define the face. They
        should be entered in a cw or ccw order; they will be standardized
        by reversal and rotation to be give the lowest lexical ordering.
        If no faces are given then no edges will be computed.

            >>> from sympy.combinatorics.polyhedron import Polyhedron
            >>> Polyhedron(list('abc'), [(1, 2, 0)]).faces
            {(0, 1, 2)}
            >>> Polyhedron(list('abc'), [(1, 0, 2)]).faces
            {(0, 1, 2)}

        The allowed transformations are entered as allowable permutations
        of the vertices for the polyhedron. Instance of Permutations
        (as with faces) should refer to the supplied vertices by index.
        These permutation are stored as a PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy import init_printing
        >>> from sympy.abc import w, x, y, z
        >>> init_printing(pretty_print=False, perm_cyclic=False)

        Here we construct the Polyhedron object for a tetrahedron.

        >>> corners = [w, x, y, z]
        >>> faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]

        Next, allowed transformations of the polyhedron must be given. This
        is given as permutations of vertices.

        Although the vertices of a tetrahedron can be numbered in 24 (4!)
        different ways, there are only 12 different orientations for a
        physical tetrahedron. The following permutations, applied once or
        twice, will generate all 12 of the orientations. (The identity
        permutation, Permutation(range(4)), is not included since it does
        not change the orientation of the vertices.)

        >>> pgroup = [Permutation([[0, 1, 2], [3]]), \
                      Permutation([[0, 1, 3], [2]]), \
                      Permutation([[0, 2, 3], [1]]), \
                      Permutation([[1, 2, 3], [0]]), \
                      Permutation([[0, 1], [2, 3]]), \
                      Permutation([[0, 2], [1, 3]]), \
                      Permutation([[0, 3], [1, 2]])]

        The Polyhedron is now constructed and demonstrated:

        >>> tetra = Polyhedron(corners, faces, pgroup)
        >>> tetra.size
        4
        >>> tetra.edges
        {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        >>> tetra.corners
        (w, x, y, z)

        It can be rotated with an arbitrary permutation of vertices, e.g.
        the following permutation is not in the pgroup:

        >>> tetra.rotate(Permutation([0, 1, 3, 2]))
        >>> tetra.corners
        (w, x, z, y)

        An allowed permutation of the vertices can be constructed by
        repeatedly applying permutations from the pgroup to the vertices.
        Here is a demonstration that applying p and p**2 for every p in
        pgroup generates all the orientations of a tetrahedron and no others:

        >>> all = ( (w, x, y, z), \
                    (x, y, w, z), \
                    (y, w, x, z), \
                    (w, z, x, y), \
                    (z, w, y, x), \
                    (w, y, z, x), \
                    (y, z, w, x), \
                    (x, z, y, w), \
                    (z, y, x, w), \
                    (y, x, z, w), \
                    (x, w, z, y), \
                    (z, x, w, y) )

        >>> got = []
        >>> for p in (pgroup + [p**2 for p in pgroup]):
        ...     h = Polyhedron(corners)
        ...     h.rotate(p)
        ...     got.append(h.corners)
        ...
        >>> set(got) == set(all)
        True

        The make_perm method of a PermutationGroup will randomly pick
        permutations, multiply them together, and return the permutation that
        can be applied to the polyhedron to give the orientation produced
        by those individual permutations.

        Here, 3 permutations are used:

        >>> tetra.pgroup.make_perm(3) # doctest: +SKIP
        Permutation([0, 3, 1, 2])

        To select the permutations that should be used, supply a list
        of indices to the permutations in pgroup in the order they should
        be applied:

        >>> use = [0, 0, 2]
        >>> p002 = tetra.pgroup.make_perm(3, use)
        >>> p002
        Permutation([1, 0, 3, 2])


        Apply them one at a time:

        >>> tetra.reset()
        >>> for i in use:
        ...     tetra.rotate(pgroup[i])
        ...
        >>> tetra.vertices
        (x, w, z, y)
        >>> sequentially = tetra.vertices

        Apply the composite permutation:

        >>> tetra.reset()
        >>> tetra.rotate(p002)
        >>> tetra.corners
        (x, w, z, y)
        >>> tetra.corners in all and tetra.corners == sequentially
        True

        Notes
        =====

        Defining permutation groups
        ---------------------------

        It is not necessary to enter any permutations, nor is necessary to
        enter a complete set of transformations. In fact, for a polyhedron,
        all configurations can be constructed from just two permutations.
        For example, the orientations of a tetrahedron can be generated from
        an axis passing through a vertex and face and another axis passing
        through a different vertex or from an axis passing through the
        midpoints of two edges opposite of each other.

        For simplicity of presentation, consider a square --
        not a cube -- with vertices 1, 2, 3, and 4:

        1-----2  We could think of axes of rotation being:
        |     |  1) through the face
        |     |  2) from midpoint 1-2 to 3-4 or 1-3 to 2-4
        3-----4  3) lines 1-4 or 2-3


        To determine how to write the permutations, imagine 4 cameras,
        one at each corner, labeled A-D:

        A       B          A       B
         1-----2            1-----3             vertex index:
         |     |            |     |                 1   0
         |     |            |     |                 2   1
         3-----4            2-----4                 3   2
        C       D          C       D                4   3

        original           after rotation
                           along 1-4

        A diagonal and a face axis will be chosen for the "permutation group"
        from which any orientation can be constructed.

        >>> pgroup = []

        Imagine a clockwise rotation when viewing 1-4 from camera A. The new
        orientation is (in camera-order): 1, 3, 2, 4 so the permutation is
        given using the *indices* of the vertices as:

        >>> pgroup.append(Permutation((0, 2, 1, 3)))

        Now imagine rotating clockwise when looking down an axis entering the
        center of the square as viewed. The new camera-order would be
        3, 1, 4, 2 so the permutation is (using indices):

        >>> pgroup.append(Permutation((2, 0, 3, 1)))

        The square can now be constructed:
            ** use real-world labels for the vertices, entering them in
               camera order
            ** for the faces we use zero-based indices of the vertices
               in *edge-order* as the face is traversed; neither the
               direction nor the starting point matter -- the faces are
               only used to define edges (if so desired).

        >>> square = Polyhedron((1, 2, 3, 4), [(0, 1, 3, 2)], pgroup)

        To rotate the square with a single permutation we can do:

        >>> square.rotate(square.pgroup[0])
        >>> square.corners
        (1, 3, 2, 4)

        To use more than one permutation (or to use one permutation more
        than once) it is more convenient to use the make_perm method:

        >>> p011 = square.pgroup.make_perm([0, 1, 1]) # diag flip + 2 rotations
        >>> square.reset() # return to initial orientation
        >>> square.rotate(p011)
        >>> square.corners
        (4, 2, 3, 1)

        Thinking outside the box
        ------------------------

        Although the Polyhedron object has a direct physical meaning, it
        actually has broader application. In the most general sense it is
        just a decorated PermutationGroup, allowing one to connect the
        permutations to something physical. For example, a Rubik's cube is
        not a proper polyhedron, but the Polyhedron class can be used to
        represent it in a way that helps to visualize the Rubik's cube.

        >>> from sympy import flatten, unflatten, symbols
        >>> from sympy.combinatorics import RubikGroup
        >>> facelets = flatten([symbols(s+'1:5') for s in 'UFRBLD'])
        >>> def show():
        ...     pairs = unflatten(r2.corners, 2)
        ...     print(pairs[::2])
        ...     print(pairs[1::2])
        ...
        >>> r2 = Polyhedron(facelets, pgroup=RubikGroup(2))
        >>> show()
        [(U1, U2), (F1, F2), (R1, R2), (B1, B2), (L1, L2), (D1, D2)]
        [(U3, U4), (F3, F4), (R3, R4), (B3, B4), (L3, L4), (D3, D4)]
        >>> r2.rotate(0) # cw rotation of F
        >>> show()
        [(U1, U2), (F3, F1), (U3, R2), (B1, B2), (L1, D1), (R3, R1)]
        [(L4, L2), (F4, F2), (U4, R4), (B3, B4), (L3, D2), (D3, D4)]

        Predefined Polyhedra
        ====================

        For convenience, the vertices and faces are defined for the following
        standard solids along with a permutation group for transformations.
        When the polyhedron is oriented as indicated below, the vertices in
        a given horizontal plane are numbered in ccw direction, starting from
        the vertex that will give the lowest indices in a given face. (In the
        net of the vertices, indices preceded by "-" indicate replication of
        the lhs index in the net.)

        tetrahedron, tetrahedron_faces
        ------------------------------

            4 vertices (vertex up) net:

                 0 0-0
                1 2 3-1

            4 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 1) (1, 2, 3)

        cube, cube_faces
        ----------------

            8 vertices (face up) net:

                0 1 2 3-0
                4 5 6 7-4

            6 faces:

            (0, 1, 2, 3)
            (0, 1, 5, 4) (1, 2, 6, 5) (2, 3, 7, 6) (0, 3, 7, 4)
            (4, 5, 6, 7)

        octahedron, octahedron_faces
        ----------------------------

            6 vertices (vertex up) net:

                 0 0 0-0
                1 2 3 4-1
                 5 5 5-5

            8 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 4) (0, 1, 4)
            (1, 2, 5) (2, 3, 5) (3, 4, 5) (1, 4, 5)

        dodecahedron, dodecahedron_faces
        --------------------------------

            20 vertices (vertex up) net:

                  0  1  2  3  4 -0
                  5  6  7  8  9 -5
                14 10 11 12 13-14
                15 16 17 18 19-15

            12 faces:

            (0, 1, 2, 3, 4) (0, 1, 6, 10, 5) (1, 2, 7, 11, 6)
            (2, 3, 8, 12, 7) (3, 4, 9, 13, 8) (0, 4, 9, 14, 5)
            (5, 10, 16, 15, 14) (6, 10, 16, 17, 11) (7, 11, 17, 18, 12)
            (8, 12, 18, 19, 13) (9, 13, 19, 15, 14)(15, 16, 17, 18, 19)

        icosahedron, icosahedron_faces
        ------------------------------

            12 vertices (face up) net:

                 0  0  0  0 -0
                1  2  3  4  5 -1
                 6  7  8  9  10 -6
                  11 11 11 11 -11

            20 faces:

            (0, 1, 2) (0, 2, 3) (0, 3, 4)
            (0, 4, 5) (0, 1, 5) (1, 2, 6)
            (2, 3, 7) (3, 4, 8) (4, 5, 9)
            (1, 5, 10) (2, 6, 7) (3, 7, 8)
            (4, 8, 9) (5, 9, 10) (1, 6, 10)
            (6, 7, 11) (7, 8, 11) (8, 9, 11)
            (9, 10, 11) (6, 10, 11)

        >>> from sympy.combinatorics.polyhedron import cube
        >>> cube.edges
        {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)}

        If you want to use letters or other names for the corners you
        can still use the pre-calculated faces:

        >>> corners = list('abcdefgh')
        >>> Polyhedron(corners, cube.faces).corners
        (a, b, c, d, e, f, g, h)

        References
        ==========

        .. [1] www.ocf.berkeley.edu/~wwu/articles/platonicsolids.pdf

        """
        faces = [minlex(f, directed=False, key=default_sort_key) for f in faces]
        corners, faces, pgroup = args = \
            [Tuple(*a) for a in (corners, faces, pgroup)]
        obj = Basic.__new__(cls, *args)
        obj._corners = tuple(corners)  # in order given
        obj._faces = FiniteSet(*faces)
        if pgroup and pgroup[0].size != len(corners):
            raise ValueError("Permutation size unequal to number of corners.")
        # use the identity permutation if none are given
        obj._pgroup = PermutationGroup(
            pgroup or [Perm(range(len(corners)))] )
        return obj

    @property
    def corners(self):
        """
        Get the corners of the Polyhedron.

        The method ``vertices`` is an alias for ``corners``.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron
        >>> from sympy.abc import a, b, c, d
        >>> p = Polyhedron(list('abcd'))
        >>> p.corners == p.vertices == (a, b, c, d)
        True

        See Also
        ========

        array_form, cyclic_form
        """
        return self._corners
    vertices = corners

    @property
    def array_form(self):
        """Return the indices of the corners.

        The indices are given relative to the original position of corners.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron
        >>> tetrahedron = tetrahedron.copy()
        >>> tetrahedron.array_form
        [0, 1, 2, 3]

        >>> tetrahedron.rotate(0)
        >>> tetrahedron.array_form
        [0, 2, 3, 1]
        >>> tetrahedron.pgroup[0].array_form
        [0, 2, 3, 1]

        See Also
        ========

        corners, cyclic_form
        """
        corners = list(self.args[0])
        return [corners.index(c) for c in self.corners]

    @property
    def cyclic_form(self):
        """Return the indices of the corners in cyclic notation.

        The indices are given relative to the original position of corners.

        See Also
        ========

        corners, array_form
        """
        return Perm._af_new(self.array_form).cyclic_form

    @property
    def size(self):
        """
        Get the number of corners of the Polyhedron.
        """
        return len(self._corners)

    @property
    def faces(self):
        """
        Get the faces of the Polyhedron.
        """
        return self._faces

    @property
    def pgroup(self):
        """
        Get the permutations of the Polyhedron.
        """
        return self._pgroup

    @property
    def edges(self):
        """
        Given the faces of the polyhedra we can get the edges.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron
        >>> from sympy.abc import a, b, c
        >>> corners = (a, b, c)
        >>> faces = [(0, 1, 2)]
        >>> Polyhedron(corners, faces).edges
        {(0, 1), (0, 2), (1, 2)}

        """
        if self._edges is None:
            output = set()
            for face in self.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    output.add(edge)
            self._edges = FiniteSet(*output)
        return self._edges

    def rotate(self, perm):
        """
        Apply a permutation to the polyhedron *in place*. The permutation
        may be given as a Permutation instance or an integer indicating
        which permutation from pgroup of the Polyhedron should be
        applied.

        This is an operation that is analogous to rotation about
        an axis by a fixed increment.

        Notes
        =====

        When a Permutation is applied, no check is done to see if that
        is a valid permutation for the Polyhedron. For example, a cube
        could be given a permutation which effectively swaps only 2
        vertices. A valid permutation (that rotates the object in a
        physical way) will be obtained if one only uses
        permutations from the ``pgroup`` of the Polyhedron. On the other
        hand, allowing arbitrary rotations (applications of permutations)
        gives a way to follow named elements rather than indices since
        Polyhedron allows vertices to be named while Permutation works
        only with indices.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron, Permutation
        >>> from sympy.combinatorics.polyhedron import cube
        >>> cube = cube.copy()
        >>> cube.corners
        (0, 1, 2, 3, 4, 5, 6, 7)
        >>> cube.rotate(0)
        >>> cube.corners
        (1, 2, 3, 0, 5, 6, 7, 4)

        A non-physical "rotation" that is not prohibited by this method:

        >>> cube.reset()
        >>> cube.rotate(Permutation([[1, 2]], size=8))
        >>> cube.corners
        (0, 2, 1, 3, 4, 5, 6, 7)

        Polyhedron can be used to follow elements of set that are
        identified by letters instead of integers:

        >>> shadow = h5 = Polyhedron(list('abcde'))
        >>> p = Permutation([3, 0, 1, 2, 4])
        >>> h5.rotate(p)
        >>> h5.corners
        (d, a, b, c, e)
        >>> _ == shadow.corners
        True
        >>> copy = h5.copy()
        >>> h5.rotate(p)
        >>> h5.corners == copy.corners
        False
        """
        if not isinstance(perm, Perm):
            perm = self.pgroup[perm]
            # and we know it's valid
        else:
            if perm.size != self.size:
                raise ValueError('Polyhedron and Permutation sizes differ.')
        a = perm.array_form
        corners = [self.corners[a[i]] for i in range(len(self.corners))]
        self._corners = tuple(corners)

    def reset(self):
        """Return corners to their original positions.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron as T
        >>> T = T.copy()
        >>> T.corners
        (0, 1, 2, 3)
        >>> T.rotate(0)
        >>> T.corners
        (0, 2, 3, 1)
        >>> T.reset()
        >>> T.corners
        (0, 1, 2, 3)
        """
        self._corners = self.args[0]


def _pgroup_calcs():
    """Return the permutation groups for each of the polyhedra and the face
    definitions: tetrahedron, cube, octahedron, dodecahedron, icosahedron,
    tetrahedron_faces, cube_faces, octahedron_faces, dodecahedron_faces,
    icosahedron_faces

    Explanation
    ===========

    (This author did not find and did not know of a better way to do it though
    there likely is such a way.)

    Although only 2 permutations are needed for a polyhedron in order to
    generate all the possible orientations, a group of permutations is
    provided instead. A set of permutations is called a "group" if::

    a*b = c (for any pair of permutations in the group, a and b, their
    product, c, is in the group)

    a*(b*c) = (a*b)*c (for any 3 permutations in the group associativity holds)

    there is an identity permutation, I, such that I*a = a*I for all elements
    in the group

    a*b = I (the inverse of each permutation is also in the group)

    None of the polyhedron groups defined follow these definitions of a group.
    Instead, they are selected to contain those permutations whose powers
    alone will construct all orientations of the polyhedron, i.e. for
    permutations ``a``, ``b``, etc... in the group, ``a, a**2, ..., a**o_a``,
    ``b, b**2, ..., b**o_b``, etc... (where ``o_i`` is the order of
    permutation ``i``) generate all permutations of the polyhedron instead of
    mixed products like ``a*b``, ``a*b**2``, etc....

    Note that for a polyhedron with n vertices, the valid permutations of the
    vertices exclude those that do not maintain its faces. e.g. the
    permutation BCDE of a square's four corners, ABCD, is a valid
    permutation while CBDE is not (because this would twist the square).

    Examples
    ========

    The is_group checks for: closure, the presence of the Identity permutation,
    and the presence of the inverse for each of the elements in the group. This
    confirms that none of the polyhedra are true groups:

    >>> from sympy.combinatorics.polyhedron import (
    ... tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    ...
    >>> polyhedra = (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    >>> [h.pgroup.is_group for h in polyhedra]
    ...
    [True, True, True, True, True]

    Although tests in polyhedron's test suite check that powers of the
    permutations in the groups generate all permutations of the vertices
    of the polyhedron, here we also demonstrate the powers of the given
    permutations create a complete group for the tetrahedron:

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> for h in polyhedra[:1]:
    ...     G = h.pgroup
    ...     perms = set()
    ...     for g in G:
    ...         for e in range(g.order()):
    ...             p = tuple((g**e).array_form)
    ...             perms.add(p)
    ...
    ...     perms = [Permutation(p) for p in perms]
    ...     assert PermutationGroup(perms).is_group

    In addition to doing the above, the tests in the suite confirm that the
    faces are all present after the application of each permutation.

    References
    ==========

    .. [1] https://dogschool.tripod.com/trianglegroup.html

    """
    def _pgroup_of_double(polyh, ordered_faces, pgroup):
        n = len(ordered_faces[0])
        # the vertices of the double which sits inside a give polyhedron
        # can be found by tracking the faces of the outer polyhedron.
        # A map between face and the vertex of the double is made so that
        # after rotation the position of the vertices can be located
        fmap = dict(zip(ordered_faces,
                        range(len(ordered_faces))))
        flat_faces = flatten(ordered_faces)
        new_pgroup = []
        for p in pgroup:
            h = polyh.copy()
            h.rotate(p)
            c = h.corners
            # reorder corners in the order they should appear when
            # enumerating the faces
            reorder = unflatten([c[j] for j in flat_faces], n)
            # make them canonical
            reorder = [tuple(map(as_int,
                       minlex(f, directed=False)))
                       for f in reorder]
            # map face to vertex: the resulting list of vertices are the
            # permutation that we seek for the double
            new_pgroup.append(Perm([fmap[f] for f in reorder]))
        return new_pgroup

    tetrahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 1),  # upper 3
        (1, 2, 3),  # bottom
    ]

    # cw from top
    #
    _t_pgroup = [
        Perm([[1, 2, 3], [0]]),  # cw from top
        Perm([[0, 1, 2], [3]]),  # cw from front face
        Perm([[0, 3, 2], [1]]),  # cw from back right face
        Perm([[0, 3, 1], [2]]),  # cw from back left face
        Perm([[0, 1], [2, 3]]),  # through front left edge
        Perm([[0, 2], [1, 3]]),  # through front right edge
        Perm([[0, 3], [1, 2]]),  # through back edge
    ]

    tetrahedron = Polyhedron(
        range(4),
        tetrahedron_faces,
        _t_pgroup)

    cube_faces = [
        (0, 1, 2, 3),  # upper
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 3, 7, 4),  # middle 4
        (4, 5, 6, 7),  # lower
    ]

    # U, D, F, B, L, R = up, down, front, back, left, right
    _c_pgroup = [Perm(p) for p in
        [
        [1, 2, 3, 0, 5, 6, 7, 4],  # cw from top, U
        [4, 0, 3, 7, 5, 1, 2, 6],  # cw from F face
        [4, 5, 1, 0, 7, 6, 2, 3],  # cw from R face

        [1, 0, 4, 5, 2, 3, 7, 6],  # cw through UF edge
        [6, 2, 1, 5, 7, 3, 0, 4],  # cw through UR edge
        [6, 7, 3, 2, 5, 4, 0, 1],  # cw through UB edge
        [3, 7, 4, 0, 2, 6, 5, 1],  # cw through UL edge
        [4, 7, 6, 5, 0, 3, 2, 1],  # cw through FL edge
        [6, 5, 4, 7, 2, 1, 0, 3],  # cw through FR edge

        [0, 3, 7, 4, 1, 2, 6, 5],  # cw through UFL vertex
        [5, 1, 0, 4, 6, 2, 3, 7],  # cw through UFR vertex
        [5, 6, 2, 1, 4, 7, 3, 0],  # cw through UBR vertex
        [7, 4, 0, 3, 6, 5, 1, 2],  # cw through UBL
        ]]

    cube = Polyhedron(
        range(8),
        cube_faces,
        _c_pgroup)

    octahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),  # top 4
        (1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5),  # bottom 4
    ]

    octahedron = Polyhedron(
        range(6),
        octahedron_faces,
        _pgroup_of_double(cube, cube_faces, _c_pgroup))

    dodecahedron_faces = [
        (0, 1, 2, 3, 4),  # top
        (0, 1, 6, 10, 5), (1, 2, 7, 11, 6), (2, 3, 8, 12, 7),  # upper 5
        (3, 4, 9, 13, 8), (0, 4, 9, 14, 5),
        (5, 10, 16, 15, 14), (6, 10, 16, 17, 11), (7, 11, 17, 18,
          12),  # lower 5
        (8, 12, 18, 19, 13), (9, 13, 19, 15, 14),
        (15, 16, 17, 18, 19)  # bottom
    ]

    def _string_to_perm(s):
        rv = [Perm(range(20))]
        p = None
        for si in s:
            if si not in '01':
                count = int(si) - 1
            else:
                count = 1
                if si == '0':
                    p = _f0
                elif si == '1':
                    p = _f1
            rv.extend([p]*count)
        return Perm.rmul(*rv)

    # top face cw
    _f0 = Perm([
        1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11,
        12, 13, 14, 10, 16, 17, 18, 19, 15])
    # front face cw
    _f1 = Perm([
        5, 0, 4, 9, 14, 10, 1, 3, 13, 15,
        6, 2, 8, 19, 16, 17, 11, 7, 12, 18])
    # the strings below, like 0104 are shorthand for F0*F1*F0**4 and are
    # the remaining 4 face rotations, 15 edge permutations, and the
    # 10 vertex rotations.
    _dodeca_pgroup = [_f0, _f1] + [_string_to_perm(s) for s in '''
    0104 140 014 0410
    010 1403 03104 04103 102
    120 1304 01303 021302 03130
    0412041 041204103 04120410 041204104 041204102
    10 01 1402 0140 04102 0412 1204 1302 0130 03120'''.strip().split()]

    dodecahedron = Polyhedron(
        range(20),
        dodecahedron_faces,
        _dodeca_pgroup)

    icosahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
        (1, 6, 7), (1, 2, 7), (2, 7, 8), (2, 3, 8), (3, 8, 9),
        (3, 4, 9), (4, 9, 10), (4, 5, 10), (5, 6, 10), (1, 5, 6),
        (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]

    icosahedron = Polyhedron(
        range(12),
        icosahedron_faces,
        _pgroup_of_double(
            dodecahedron, dodecahedron_faces, _dodeca_pgroup))

    return (tetrahedron, cube, octahedron, dodecahedron, icosahedron,
        tetrahedron_faces, cube_faces, octahedron_faces,
        dodecahedron_faces, icosahedron_faces)

# -----------------------------------------------------------------------
#   Standard Polyhedron groups
#
#   These are generated using _pgroup_calcs() above. However to save
#   import time we encode them explicitly here.
# -----------------------------------------------------------------------

tetrahedron = Polyhedron(
    Tuple(0, 1, 2, 3),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 1, 3),
        Tuple(1, 2, 3)),
    Tuple(
        Perm(1, 2, 3),
        Perm(3)(0, 1, 2),
        Perm(0, 3, 2),
        Perm(0, 3, 1),
        Perm(0, 1)(2, 3),
        Perm(0, 2)(1, 3),
        Perm(0, 3)(1, 2)
    ))

cube = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7),
    Tuple(
        Tuple(0, 1, 2, 3),
        Tuple(0, 1, 5, 4),
        Tuple(1, 2, 6, 5),
        Tuple(2, 3, 7, 6),
        Tuple(0, 3, 7, 4),
        Tuple(4, 5, 6, 7)),
    Tuple(
        Perm(0, 1, 2, 3)(4, 5, 6, 7),
        Perm(0, 4, 5, 1)(2, 3, 7, 6),
        Perm(0, 4, 7, 3)(1, 5, 6, 2),
        Perm(0, 1)(2, 4)(3, 5)(6, 7),
        Perm(0, 6)(1, 2)(3, 5)(4, 7),
        Perm(0, 6)(1, 7)(2, 3)(4, 5),
        Perm(0, 3)(1, 7)(2, 4)(5, 6),
        Perm(0, 4)(1, 7)(2, 6)(3, 5),
        Perm(0, 6)(1, 5)(2, 4)(3, 7),
        Perm(1, 3, 4)(2, 7, 5),
        Perm(7)(0, 5, 2)(3, 4, 6),
        Perm(0, 5, 7)(1, 6, 3),
        Perm(0, 7, 2)(1, 4, 6)))

octahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 1, 4),
        Tuple(1, 2, 5),
        Tuple(2, 3, 5),
        Tuple(3, 4, 5),
        Tuple(1, 4, 5)),
    Tuple(
        Perm(5)(1, 2, 3, 4),
        Perm(0, 4, 5, 2),
        Perm(0, 1, 5, 3),
        Perm(0, 1)(2, 4)(3, 5),
        Perm(0, 2)(1, 3)(4, 5),
        Perm(0, 3)(1, 5)(2, 4),
        Perm(0, 4)(1, 3)(2, 5),
        Perm(0, 5)(1, 4)(2, 3),
        Perm(0, 5)(1, 2)(3, 4),
        Perm(0, 4, 1)(2, 3, 5),
        Perm(0, 1, 2)(3, 4, 5),
        Perm(0, 2, 3)(1, 5, 4),
        Perm(0, 4, 3)(1, 5, 2)))

dodecahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    Tuple(
        Tuple(0, 1, 2, 3, 4),
        Tuple(0, 1, 6, 10, 5),
        Tuple(1, 2, 7, 11, 6),
        Tuple(2, 3, 8, 12, 7),
        Tuple(3, 4, 9, 13, 8),
        Tuple(0, 4, 9, 14, 5),
        Tuple(5, 10, 16, 15, 14),
        Tuple(6, 10, 16, 17, 11),
        Tuple(7, 11, 17, 18, 12),
        Tuple(8, 12, 18, 19, 13),
        Tuple(9, 13, 19, 15, 14),
        Tuple(15, 16, 17, 18, 19)),
    Tuple(
        Perm(0, 1, 2, 3, 4)(5, 6, 7, 8, 9)(10, 11, 12, 13, 14)(15, 16, 17, 18, 19),
        Perm(0, 5, 10, 6, 1)(2, 4, 14, 16, 11)(3, 9, 15, 17, 7)(8, 13, 19, 18, 12),
        Perm(0, 10, 17, 12, 3)(1, 6, 11, 7, 2)(4, 5, 16, 18, 8)(9, 14, 15, 19, 13),
        Perm(0, 6, 17, 19, 9)(1, 11, 18, 13, 4)(2, 7, 12, 8, 3)(5, 10, 16, 15, 14),
        Perm(0, 2, 12, 19, 14)(1, 7, 18, 15, 5)(3, 8, 13, 9, 4)(6, 11, 17, 16, 10),
        Perm(0, 4, 9, 14, 5)(1, 3, 13, 15, 10)(2, 8, 19, 16, 6)(7, 12, 18, 17, 11),
        Perm(0, 1)(2, 5)(3, 10)(4, 6)(7, 14)(8, 16)(9, 11)(12, 15)(13, 17)(18, 19),
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 12)(8, 10)(9, 17)(13, 16)(14, 18)(15, 19),
        Perm(0, 12)(1, 8)(2, 3)(4, 7)(5, 18)(6, 13)(9, 11)(10, 19)(14, 17)(15, 16),
        Perm(0, 8)(1, 13)(2, 9)(3, 4)(5, 12)(6, 19)(7, 14)(10, 18)(11, 15)(16, 17),
        Perm(0, 4)(1, 9)(2, 14)(3, 5)(6, 13)(7, 15)(8, 10)(11, 19)(12, 16)(17, 18),
        Perm(0, 5)(1, 14)(2, 15)(3, 16)(4, 10)(6, 9)(7, 19)(8, 17)(11, 13)(12, 18),
        Perm(0, 11)(1, 6)(2, 10)(3, 16)(4, 17)(5, 7)(8, 15)(9, 18)(12, 14)(13, 19),
        Perm(0, 18)(1, 12)(2, 7)(3, 11)(4, 17)(5, 19)(6, 8)(9, 16)(10, 13)(14, 15),
        Perm(0, 18)(1, 19)(2, 13)(3, 8)(4, 12)(5, 17)(6, 15)(7, 9)(10, 16)(11, 14),
        Perm(0, 13)(1, 19)(2, 15)(3, 14)(4, 9)(5, 8)(6, 18)(7, 16)(10, 12)(11, 17),
        Perm(0, 16)(1, 15)(2, 19)(3, 18)(4, 17)(5, 10)(6, 14)(7, 13)(8, 12)(9, 11),
        Perm(0, 18)(1, 17)(2, 16)(3, 15)(4, 19)(5, 12)(6, 11)(7, 10)(8, 14)(9, 13),
        Perm(0, 15)(1, 19)(2, 18)(3, 17)(4, 16)(5, 14)(6, 13)(7, 12)(8, 11)(9, 10),
        Perm(0, 17)(1, 16)(2, 15)(3, 19)(4, 18)(5, 11)(6, 10)(7, 14)(8, 13)(9, 12),
        Perm(0, 19)(1, 18)(2, 17)(3, 16)(4, 15)(5, 13)(6, 12)(7, 11)(8, 10)(9, 14),
        Perm(1, 4, 5)(2, 9, 10)(3, 14, 6)(7, 13, 16)(8, 15, 11)(12, 19, 17),
        Perm(19)(0, 6, 2)(3, 5, 11)(4, 10, 7)(8, 14, 17)(9, 16, 12)(13, 15, 18),
        Perm(0, 11, 8)(1, 7, 3)(4, 6, 12)(5, 17, 13)(9, 10, 18)(14, 16, 19),
        Perm(0, 7, 13)(1, 12, 9)(2, 8, 4)(5, 11, 19)(6, 18, 14)(10, 17, 15),
        Perm(0, 3, 9)(1, 8, 14)(2, 13, 5)(6, 12, 15)(7, 19, 10)(11, 18, 16),
        Perm(0, 14, 10)(1, 9, 16)(2, 13, 17)(3, 19, 11)(4, 15, 6)(7, 8, 18),
        Perm(0, 16, 7)(1, 10, 11)(2, 5, 17)(3, 14, 18)(4, 15, 12)(8, 9, 19),
        Perm(0, 16, 13)(1, 17, 8)(2, 11, 12)(3, 6, 18)(4, 10, 19)(5, 15, 9),
        Perm(0, 11, 15)(1, 17, 14)(2, 18, 9)(3, 12, 13)(4, 7, 19)(5, 6, 16),
        Perm(0, 8, 15)(1, 12, 16)(2, 18, 10)(3, 19, 5)(4, 13, 14)(6, 7, 17)))

icosahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 4, 5),
        Tuple(0, 1, 5),
        Tuple(1, 6, 7),
        Tuple(1, 2, 7),
        Tuple(2, 7, 8),
        Tuple(2, 3, 8),
        Tuple(3, 8, 9),
        Tuple(3, 4, 9),
        Tuple(4, 9, 10),
        Tuple(4, 5, 10),
        Tuple(5, 6, 10),
        Tuple(1, 5, 6),
        Tuple(6, 7, 11),
        Tuple(7, 8, 11),
        Tuple(8, 9, 11),
        Tuple(9, 10, 11),
        Tuple(6, 10, 11)),
    Tuple(
        Perm(11)(1, 2, 3, 4, 5)(6, 7, 8, 9, 10),
        Perm(0, 5, 6, 7, 2)(3, 4, 10, 11, 8),
        Perm(0, 1, 7, 8, 3)(4, 5, 6, 11, 9),
        Perm(0, 2, 8, 9, 4)(1, 7, 11, 10, 5),
        Perm(0, 3, 9, 10, 5)(1, 2, 8, 11, 6),
        Perm(0, 4, 10, 6, 1)(2, 3, 9, 11, 7),
        Perm(0, 1)(2, 5)(3, 6)(4, 7)(8, 10)(9, 11),
        Perm(0, 2)(1, 3)(4, 7)(5, 8)(6, 9)(10, 11),
        Perm(0, 3)(1, 9)(2, 4)(5, 8)(6, 11)(7, 10),
        Perm(0, 4)(1, 9)(2, 10)(3, 5)(6, 8)(7, 11),
        Perm(0, 5)(1, 4)(2, 10)(3, 6)(7, 9)(8, 11),
        Perm(0, 6)(1, 5)(2, 10)(3, 11)(4, 7)(8, 9),
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 8)(9, 10),
        Perm(0, 8)(1, 9)(2, 3)(4, 7)(5, 11)(6, 10),
        Perm(0, 9)(1, 11)(2, 10)(3, 4)(5, 8)(6, 7),
        Perm(0, 10)(1, 9)(2, 11)(3, 6)(4, 5)(7, 8),
        Perm(0, 11)(1, 6)(2, 10)(3, 9)(4, 8)(5, 7),
        Perm(0, 11)(1, 8)(2, 7)(3, 6)(4, 10)(5, 9),
        Perm(0, 11)(1, 10)(2, 9)(3, 8)(4, 7)(5, 6),
        Perm(0, 11)(1, 7)(2, 6)(3, 10)(4, 9)(5, 8),
        Perm(0, 11)(1, 9)(2, 8)(3, 7)(4, 6)(5, 10),
        Perm(0, 5, 1)(2, 4, 6)(3, 10, 7)(8, 9, 11),
        Perm(0, 1, 2)(3, 5, 7)(4, 6, 8)(9, 10, 11),
        Perm(0, 2, 3)(1, 8, 4)(5, 7, 9)(6, 11, 10),
        Perm(0, 3, 4)(1, 8, 10)(2, 9, 5)(6, 7, 11),
        Perm(0, 4, 5)(1, 3, 10)(2, 9, 6)(7, 8, 11),
        Perm(0, 10, 7)(1, 5, 6)(2, 4, 11)(3, 9, 8),
        Perm(0, 6, 8)(1, 7, 2)(3, 5, 11)(4, 10, 9),
        Perm(0, 7, 9)(1, 11, 4)(2, 8, 3)(5, 6, 10),
        Perm(0, 8, 10)(1, 7, 6)(2, 11, 5)(3, 9, 4),
        Perm(0, 9, 6)(1, 3, 11)(2, 8, 7)(4, 10, 5)))

tetrahedron_faces = [tuple(arg) for arg in tetrahedron.faces]

cube_faces = [tuple(arg) for arg in cube.faces]

octahedron_faces = [tuple(arg) for arg in octahedron.faces]

dodecahedron_faces = [tuple(arg) for arg in dodecahedron.faces]

icosahedron_faces = [tuple(arg) for arg in icosahedron.faces]
<|endoftext|>from itertools import combinations

from sympy.combinatorics.graycode import GrayCode


class Subset():
    """
    Represents a basic subset object.

    Explanation
    ===========

    We generate subsets using essentially two techniques,
    binary enumeration and lexicographic enumeration.
    The Subset class takes two arguments, the first one
    describes the initial subset to consider and the second
    describes the superset.

    Examples
    ========

    >>> from sympy.combinatorics import Subset
    >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    >>> a.next_binary().subset
    ['b']
    >>> a.prev_binary().subset
    ['c']
    """

    _rank_binary = None
    _rank_lex = None
    _rank_graycode = None
    _subset = None
    _superset = None

    def __new__(cls, subset, superset):
        """
        Default constructor.

        It takes the ``subset`` and its ``superset`` as its parameters.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']
        >>> a.superset
        ['a', 'b', 'c', 'd']
        >>> a.size
        2
        """
        if len(subset) > len(superset):
            raise ValueError('Invalid arguments have been provided. The '
                             'superset must be larger than the subset.')
        for elem in subset:
            if elem not in superset:
                raise ValueError('The superset provided is invalid as it does '
                                 'not contain the element {}'.format(elem))
        obj = object.__new__(cls)
        obj._subset = subset
        obj._superset = superset
        return obj

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        whether both objects are of the class Subset and if the values
        of the subset and superset attributes are the same.
        """
        if not isinstance(other, Subset):
            return NotImplemented
        return self.subset == other.subset and self.superset == other.superset

    def iterate_binary(self, k):
        """
        This is a helper function. It iterates over the
        binary subsets by ``k`` steps. This variable can be
        both positive or negative.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(-2).subset
        ['d']
        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(2).subset
        []

        See Also
        ========

        next_binary, prev_binary
        """
        bin_list = Subset.bitlist_from_subset(self.subset, self.superset)
        n = (int(''.join(bin_list), 2) + k) % 2**self.superset_size
        bits = bin(n)[2:].rjust(self.superset_size, '0')
        return Subset.subset_from_bitlist(self.superset, bits)

    def next_binary(self):
        """
        Generates the next binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        ['b']
        >>> a = Subset(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        []

        See Also
        ========

        prev_binary, iterate_binary
        """
        return self.iterate_binary(1)

    def prev_binary(self):
        """
        Generates the previous binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['a', 'b', 'c', 'd']
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['c']

        See Also
        ========

        next_binary, iterate_binary
        """
        return self.iterate_binary(-1)

    def next_lexicographic(self):
        """
        Generates the next lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        ['d']
        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        []

        See Also
        ========

        prev_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        if i in indices:
            if i - 1 in indices:
                indices.remove(i - 1)
            else:
                indices.remove(i)
                i = i - 1
                while i >= 0 and i not in indices:
                    i = i - 1
                if i >= 0:
                    indices.remove(i)
                    indices.append(i+1)
        else:
            while i not in indices and i >= 0:
                i = i - 1
            indices.append(i + 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def prev_lexicographic(self):
        """
        Generates the previous lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['d']
        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['c']

        See Also
        ========

        next_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        while i >= 0 and i not in indices:
            i = i - 1

        if i == 0 or i - 1 in indices:
            indices.remove(i)
        else:
            if i >= 0:
                indices.remove(i)
                indices.append(i - 1)
            indices.append(self.superset_size - 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def iterate_graycode(self, k):
        """
        Helper function used for prev_gray and next_gray.
        It performs ``k`` step overs to get the respective Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.iterate_graycode(3).subset
        [1, 4]
        >>> a.iterate_graycode(-2).subset
        [1, 2, 4]

        See Also
        ========

        next_gray, prev_gray
        """
        unranked_code = GrayCode.unrank(self.superset_size,
                                       (self.rank_gray + k) % self.cardinality)
        return Subset.subset_from_bitlist(self.superset,
                                          unranked_code)

    def next_gray(self):
        """
        Generates the next Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.next_gray().subset
        [1, 3]

        See Also
        ========

        iterate_graycode, prev_gray
        """
        return self.iterate_graycode(1)

    def prev_gray(self):
        """
        Generates the previous Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([2, 3, 4], [1, 2, 3, 4, 5])
        >>> a.prev_gray().subset
        [2, 3, 4, 5]

        See Also
        ========

        iterate_graycode, next_gray
        """
        return self.iterate_graycode(-1)

    @property
    def rank_binary(self):
        """
        Computes the binary ordered rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a','b','c','d'])
        >>> a.rank_binary
        0
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_binary
        3

        See Also
        ========

        iterate_binary, unrank_binary
        """
        if self._rank_binary is None:
            self._rank_binary = int("".join(
                Subset.bitlist_from_subset(self.subset,
                                           self.superset)), 2)
        return self._rank_binary

    @property
    def rank_lexicographic(self):
        """
        Computes the lexicographic ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_lexicographic
        14
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_lexicographic
        43
        """
        if self._rank_lex is None:
            def _ranklex(self, subset_index, i, n):
                if subset_index == [] or i > n:
                    return 0
                if i in subset_index:
                    subset_index.remove(i)
                    return 1 + _ranklex(self, subset_index, i + 1, n)
                return 2**(n - i - 1) + _ranklex(self, subset_index, i + 1, n)
            indices = Subset.subset_indices(self.subset, self.superset)
            self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
        return self._rank_lex

    @property
    def rank_gray(self):
        """
        Computes the Gray code ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c','d'], ['a','b','c','d'])
        >>> a.rank_gray
        2
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_gray
        27

        See Also
        ========

        iterate_graycode, unrank_gray
        """
        if self._rank_graycode is None:
            bits = Subset.bitlist_from_subset(self.subset, self.superset)
            self._rank_graycode = GrayCode(len(bits), start=bits).rank
        return self._rank_graycode

    @property
    def subset(self):
        """
        Gets the subset represented by the current instance.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']

        See Also
        ========

        superset, size, superset_size, cardinality
        """
        return self._subset

    @property
    def size(self):
        """
        Gets the size of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.size
        2

        See Also
        ========

        subset, superset, superset_size, cardinality
        """
        return len(self.subset)

    @property
    def superset(self):
        """
        Gets the superset of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset
        ['a', 'b', 'c', 'd']

        See Also
        ========

        subset, size, superset_size, cardinality
        """
        return self._superset

    @property
    def superset_size(self):
        """
        Returns the size of the superset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset_size
        4

        See Also
        ========

        subset, superset, size, cardinality
        """
        return len(self.superset)

    @property
    def cardinality(self):
        """
        Returns the number of all possible subsets.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.cardinality
        16

        See Also
        ========

        subset, superset, size, superset_size
        """
        return 2**(self.superset_size)

    @classmethod
    def subset_from_bitlist(self, super_set, bitlist):
        """
        Gets the subset defined by the bitlist.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset
        ['c', 'd']

        See Also
        ========

        bitlist_from_subset
        """
        if len(super_set) != len(bitlist):
            raise ValueError("The sizes of the lists are not equal")
        ret_set = []
        for i in range(len(bitlist)):
            if bitlist[i] == '1':
                ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    @classmethod
    def bitlist_from_subset(self, subset, superset):
        """
        Gets the bitlist corresponding to a subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.bitlist_from_subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        '0011'

        See Also
        ========

        subset_from_bitlist
        """
        bitlist = ['0'] * len(superset)
        if isinstance(subset, Subset):
            subset = subset.subset
        for i in Subset.subset_indices(subset, superset):
            bitlist[i] = '1'
        return ''.join(bitlist)

    @classmethod
    def unrank_binary(self, rank, superset):
        """
        Gets the binary ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset
        ['b']

        See Also
        ========

        iterate_binary, rank_binary
        """
        bits = bin(rank)[2:].rjust(len(superset), '0')
        return Subset.subset_from_bitlist(superset, bits)

    @classmethod
    def unrank_gray(self, rank, superset):
        """
        Gets the Gray code ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset
        ['a', 'b']
        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset
        []

        See Also
        ========

        iterate_graycode, rank_gray
        """
        graycode_bitlist = GrayCode.unrank(len(superset), rank)
        return Subset.subset_from_bitlist(superset, graycode_bitlist)

    @classmethod
    def subset_indices(self, subset, superset):
        """Return indices of subset in superset in a list; the list is empty
        if all elements of ``subset`` are not in ``superset``.

        Examples
        ========

            >>> from sympy.combinatorics import Subset
            >>> superset = [1, 3, 2, 5, 4]
            >>> Subset.subset_indices([3, 2, 1], superset)
            [1, 2, 0]
            >>> Subset.subset_indices([1, 6], superset)
            []
            >>> Subset.subset_indices([], superset)
            []

        """
        a, b = superset, subset
        sb = set(b)
        d = {}
        for i, ai in enumerate(a):
            if ai in sb:
                d[ai] = i
                sb.remove(ai)
                if not sb:
                    break
        else:
            return []
        return [d[bi] for bi in b]


def ksubsets(superset, k):
    """
    Finds the subsets of size ``k`` in lexicographic order.

    This uses the itertools generator.

    Examples
    ========

    >>> from sympy.combinatorics.subsets import ksubsets
    >>> list(ksubsets([1, 2, 3], 2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(ksubsets([1, 2, 3, 4, 5], 2))
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), \
    (2, 5), (3, 4), (3, 5), (4, 5)]

    See Also
    ========

    Subset
    """
    return combinations(superset, k)
<|endoftext|>"""Finitely Presented Groups and its algorithms. """

from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
                                                free_group)
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
                                             coset_enumeration_r,
                                             coset_enumeration_c)
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute

from itertools import product


@public
def fp_group(fr_grp, relators=()):
    _fp_group = FpGroup(fr_grp, relators)
    return (_fp_group,) + tuple(_fp_group._generators)

@public
def xfp_group(fr_grp, relators=()):
    _fp_group = FpGroup(fr_grp, relators)
    return (_fp_group, _fp_group._generators)

# Does not work. Both symbols and pollute are undefined. Never tested.
@public
def vfp_group(fr_grpm, relators):
    _fp_group = FpGroup(symbols, relators)
    pollute([sym.name for sym in _fp_group.symbols], _fp_group.generators)
    return _fp_group


def _parse_relators(rels):
    """Parse the passed relators."""
    return rels


###############################################################################
#                           FINITELY PRESENTED GROUPS                         #
###############################################################################


class FpGroup(DefaultPrinting):
    """
    The FpGroup would take a FreeGroup and a list/tuple of relators, the
    relators would be specified in such a way that each of them be equal to the
    identity of the provided free group.

    """
    is_group = True
    is_FpGroup = True
    is_PermutationGroup = False

    def __init__(self, fr_grp, relators):
        relators = _parse_relators(relators)
        self.free_group = fr_grp
        self.relators = relators
        self.generators = self._generators()
        self.dtype = type("FpGroupElement", (FpGroupElement,), {"group": self})

        # CosetTable instance on identity subgroup
        self._coset_table = None
        # returns whether coset table on identity subgroup
        # has been standardized
        self._is_standardized = False

        self._order = None
        self._center = None

        self._rewriting_system = RewritingSystem(self)
        self._perm_isomorphism = None
        return

    def _generators(self):
        return self.free_group.generators

    def make_confluent(self):
        '''
        Try to make the group's rewriting system confluent

        '''
        self._rewriting_system.make_confluent()
        return

    def reduce(self, word):
        '''
        Return the reduced form of `word` in `self` according to the group's
        rewriting system. If it's confluent, the reduced form is the unique normal
        form of the word in the group.

        '''
        return self._rewriting_system.reduce(word)

    def equals(self, word1, word2):
        '''
        Compare `word1` and `word2` for equality in the group
        using the group's rewriting system. If the system is
        confluent, the returned answer is necessarily correct.
        (If it is not, `False` could be returned in some cases
        where in fact `word1 == word2`)

        '''
        if self.reduce(word1*word2**-1) == self.identity:
            return True
        elif self._rewriting_system.is_confluent:
            return False
        return None

    @property
    def identity(self):
        return self.free_group.identity

    def __contains__(self, g):
        return g in self.free_group

    def subgroup(self, gens, C=None, homomorphism=False):
        '''
        Return the subgroup generated by `gens` using the
        Reidemeister-Schreier algorithm
        homomorphism -- When set to True, return a dictionary containing the images
                     of the presentation generators in the original group.

        Examples
        ========

        >>> from sympy.combinatorics.fp_groups import FpGroup
        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**3, y**5, (x*y)**2])
        >>> H = [x*y, x**-1*y**-1*x*y*x]
        >>> K, T = f.subgroup(H, homomorphism=True)
        >>> T(K.generators)
        [x*y, x**-1*y**2*x**-1]

        '''

        if not all(isinstance(g, FreeGroupElement) for g in gens):
            raise ValueError("Generators must be `FreeGroupElement`s")
        if not all(g.group == self.free_group for g in gens):
                raise ValueError("Given generators are not members of the group")
        if homomorphism:
            g, rels, _gens = reidemeister_presentation(self, gens, C=C, homomorphism=True)
        else:
            g, rels = reidemeister_presentation(self, gens, C=C)
        if g:
            g = FpGroup(g[0].group, rels)
        else:
            g = FpGroup(free_group('')[0], [])
        if homomorphism:
            from sympy.combinatorics.homomorphisms import homomorphism
            return g, homomorphism(g, self, g.generators, _gens, check=False)
        return g

    def coset_enumeration(self, H, strategy="relator_based", max_cosets=None,
                                                        draft=None, incomplete=False):
        """
        Return an instance of ``coset table``, when Todd-Coxeter algorithm is
        run over the ``self`` with ``H`` as subgroup, using ``strategy``
        argument as strategy. The returned coset table is compressed but not
        standardized.

        An instance of `CosetTable` for `fp_grp` can be passed as the keyword
        argument `draft` in which case the coset enumeration will start with
        that instance and attempt to complete it.

        When `incomplete` is `True` and the function is unable to complete for
        some reason, the partially complete table will be returned.

        """
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        if strategy == 'relator_based':
            C = coset_enumeration_r(self, H, max_cosets=max_cosets,
                                                    draft=draft, incomplete=incomplete)
        else:
            C = coset_enumeration_c(self, H, max_cosets=max_cosets,
                                                    draft=draft, incomplete=incomplete)
        if C.is_complete():
            C.compress()
        return C

    def standardize_coset_table(self):
        """
        Standardized the coset table ``self`` and makes the internal variable
        ``_is_standardized`` equal to ``True``.

        """
        self._coset_table.standardize()
        self._is_standardized = True

    def coset_table(self, H, strategy="relator_based", max_cosets=None,
                                                 draft=None, incomplete=False):
        """
        Return the mathematical coset table of ``self`` in ``H``.

        """
        if not H:
            if self._coset_table is not None:
                if not self._is_standardized:
                    self.standardize_coset_table()
            else:
                C = self.coset_enumeration([], strategy, max_cosets=max_cosets,
                                            draft=draft, incomplete=incomplete)
                self._coset_table = C
                self.standardize_coset_table()
            return self._coset_table.table
        else:
            C = self.coset_enumeration(H, strategy, max_cosets=max_cosets,
                                            draft=draft, incomplete=incomplete)
            C.standardize()
