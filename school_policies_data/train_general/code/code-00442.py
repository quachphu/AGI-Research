        # verify that all symbols are known (they either represent plotting
        # ranges or parameters)
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                # ranges can't depend on each other, for example this are
                # not allowed:
                # (x, 0, y), (y, 0, 3)
                # (x, 0, y), (y, x + 2, 3)
                raise ValueError("Range symbols can't be included into "
                    "minimum and maximum of a range. "
                    "Received range: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unknown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)

    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.

        Notes
        =====
        The old sympy.plotting used experimental_lambdify. It created one
        lambda function each time an evaluation was requested. If that failed,
        it went on to create a different lambda function and evaluated it,
        and so on.

        This new module changes strategy: it creates right away the default
        lambda function as well as the backup one. The reason is that the
        series could be interactive, hence the numerical function will be
        evaluated multiple times. So, let's create the functions just once.

        This approach works fine for the majority of cases, in which the
        symbolic expression is relatively short, hence the lambdification
        is fast. If the expression is very long, this approach takes twice
        the time to create the lambda functions. Be aware of that!
        """
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if not any(callable(e) for e in exprs):
            fs = _get_free_symbols(exprs)
            self._signature = sorted(fs, key=lambda t: t.name)

            # Generate a list of lambda functions, two for each expression:
            # 1. the default one.
            # 2. the backup one, in case of failures with the default one.
            self._functions = []
            for e in exprs:
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                self._functions.append([
                    lambdify(self._signature, e, modules=self.modules),
                    lambdify(self._signature, e, modules="sympy", dummify=True),
                ])
        else:
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            self._functions = [(e, None) for e in exprs]

        # deal with symbolic color_func
        if isinstance(self.color_func, Expr):
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        """If the value of a plotting range is a symbolic expression,
        substitute the parameters in order to get a numerical value.
        """
        if not self._interactive_ranges:
            return complex(t)
        return complex(t.subs(self.params))

    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization
        discr_symbols = []
        discretizations = []

        # create a 1D discretization
        for i, r in enumerate(self.ranges):
            discr_symbols.append(r[0])
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.only_integers or (r[0] in self._needs_to_be_int)
            d = BaseSeries._discretize(start, end, self.n[i],
                scale=self.scales[i],
                only_integers=needs_integer_discr)

            if ((not self._force_real_eval) and (not needs_integer_discr) and
                (d.dtype != "complex")):
                d = d + 1j * c_start.imag

            if needs_integer_discr:
                d = d.astype(int)

            discretizations.append(d)

        # create 2D or 3D
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
        np = import_module('numpy')

        # discretization suitable for 2D line plots, 3D surface plots,
        # contours plots, vector plots
        # NOTE: why indexing='ij'? Because it produces consistent results with
        # np.mgrid. This is important as Mayavi requires this indexing
        # to correctly compute 3D streamlines. While VTK is able to compute
        # streamlines regardless of the indexing, with indexing='xy' it
        # produces "strange" results with "voids" into the
        # discretization volume. indexing='ij' solves the problem.
        # Also note that matplotlib 2D streamlines requires indexing='xy'.
        indexing = "xy"
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = "ij"
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = dict(zip(discr_symbols, meshes))

    def _evaluate(self, cast_to_real=True):
        """Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
        np = import_module('numpy')

        # create lambda functions
        if not self._functions:
            self._create_lambda_func()
        # create (or update) the discretized domain
        if (not self._discretized_domain) or self._interactive_ranges:
            self._create_discretized_domain()
        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.ranges]

        args = self._aggregate_args()

        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr[0])
            # sometime the evaluation is performed over arrays of type object.
            # hence, `result` might be of type object, which don't work well
            # with numpy real and imag functions.
            r = r.astype(complex)
            results.append(r)

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        """Create a list of arguments to be passed to the lambda function,
        sorted according to self._signature.
        """
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(
                    int(self._params[s]) if s in self._needs_to_be_int else
                    self._params[s] if self._force_real_eval
                    else complex(self._params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    @property
    def expr(self):
        """Return the expression (or expressions) of the series."""
        return self._expr

    @expr.setter
    def expr(self, e):
        """Set the expression (or expressions) of the series."""
        is_iter = hasattr(e, "__iter__")
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        if is_callable:
            self._expr = e
        else:
            self._expr = sympify(e) if not is_iter else Tuple(*e)

            # look for the upper bound of summations and products
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)

            # list of sympy functions that when lambdified, the corresponding
            # numpy functions don't like complex-type arguments
            pf = [ceiling, floor, atan2, frac, zeta]
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                if self._force_real_eval and ((self.modules is None) or
                    (isinstance(self.modules, str) and "numpy" in self.modules)):
                    funcs = [f for f, c in zip(pf, check_res) if c]
                    warnings.warn("NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            if self._functions:
                # update lambda functions
                self._create_lambda_func()

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        self._line_surface_color("_line_color", val)

    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points.
        """
        return self._n

    @n.setter
    def n(self, v):
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        The following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
        if not hasattr(v, "__iter__"):
            self._n[0] = v
        else:
            self._n[:len(v)] = v
        if self._discretized_domain:
            # update the discretized domain
            self._create_discretized_domain()

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p

    def _post_init(self):
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following in order to avoid
        # surprises on the backend
        if any(callable(e) for e in exprs):
            if self._label == str(self.expr):
                self.label = ""

        self._check_fs()

        if hasattr(self, "adaptive") and self.adaptive and self.params:
            warnings.warn("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "adaptive evaluation. Automatically switched to "
                "adaptive=False.")
            self.adaptive = False

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, v):
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        self._line_surface_color("_surface_color", val)

    @property
    def rendering_kw(self):
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            self._rendering_kw = {}
            if kwargs is not None:
                warnings.warn(
                    "`rendering_kw` must be a dictionary, instead an "
                    "object of type %s was received. " % type(kwargs) +
                    "Automatically setting `rendering_kw` to an empty "
                    "dictionary")

    @staticmethod
    def _discretize(start, end, N, scale="linear", only_integers=False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
        np = import_module('numpy')

        if only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1

        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_shape(a, b):
        """Convert ``a`` to a np.ndarray of the same shape of ``b``.

        Parameters
        ==========

        a : int, float, complex, np.ndarray
            Usually, this is the result of a numerical evaluation of a
            symbolic expression. Even if a discretized domain was used to
            evaluate the function, the result can be a scalar (int, float,
            complex). Think for example to ``expr = Float(2)`` and
            ``f = lambdify(x, expr)``. No matter the shape of the numerical
            array representing x, the result of the evaluation will be
            a single value.

        b : np.ndarray
            It represents the correct shape that ``a`` should have.

        Returns
        =======
        new_a : np.ndarray
            An array with the correct shape.
        """
        np = import_module('numpy')

        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if a.shape != b.shape:
            if a.shape == ():
                a = a * np.ones_like(b)
            else:
                a = a.reshape(b.shape)
        return a

    def eval_color_func(self, *args):
        """Evaluate the color function.

        Parameters
        ==========

        args : tuple
            Arguments to be passed to the coloring function. Can be coordinates
            or parameters or both.

        Notes
        =====

        The backend will request the data series to generate the numerical
        data. Depending on the data series, either the data series itself or
        the backend will eventually execute this function to generate the
        appropriate coloring value.
        """
        np = import_module('numpy')
        if self.color_func is None:
            # NOTE: with the line_color and surface_color attributes
            # (back-compatibility with the old sympy.plotting module) it is
            # possible to create a plot with a callable line_color (or
            # surface_color). For example:
            # p = plot(sin(x), line_color=lambda x, y: -y)
            # This creates a ColoredLineOver1DRangeSeries with line_color=None
            # and color_func=lambda x, y: -y, which effectively is a
            # parametric series. Later we could change it to a string value:
            # p[0].line_color = "red"
            # However, this sets ine_color="red" and color_func=None, but the
            # series is still ColoredLineOver1DRangeSeries (a parametric
            # series), which will render using a color_func...
            warnings.warn("This is likely not the result you were "
                "looking for. Please, re-execute the plot command, this time "
                "with the appropriate an appropriate value to line_color "
                "or surface_color.")
            return np.ones_like(args[0])

        if self._eval_color_func_with_signature:
            args = self._aggregate_args()
            color = self.color_func(*args)
            _re, _im = np.real(color), np.imag(color)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re

        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    # ColoredLineOver1DRangeSeries
                    return self._correct_shape(self.color_func(args[0]), args[0])
                # Parametric2DLineSeries
                return self._correct_shape(self.color_func(args[2]), args[2])
            elif self.is_3Dline and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            return self._correct_shape(self.color_func(args[0]), args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(*args[3:]), args[3])
            return self._correct_shape(self.color_func(*args[:2]), args[0])
        return self._correct_shape(self.color_func(*args[:nargs]), args[0])

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        return wrapper % label

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self.expr):
            # when the backend requests a latex label and user didn't provide
            # any label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

    @property
    def label(self):
        return self.get_label()

    @label.setter
    def label(self, val):
        """Set the labels associated to this series."""
        # NOTE: the init method of any series requires a label. If the user do
        # not provide it, the preprocessing function will set label=None, which
        # informs the series to initialize two attributes:
        # _label contains the string representation of the expression.
        # _latex_label contains the latex representation of the expression.
        self._label = self._latex_label = val

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        t = lambda x, transform: x if transform is None else transform(x)
        x, y, z = None, None, None
        if len(args) == 2:
            x, y = args
            return t(x, self._tx), t(y, self._ty)
        elif (len(args) == 3) and isinstance(self, Parametric2DLineSeries):
            x, y, u = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        elif (len(args) == 4) and isinstance(self, Parametric3DLineSeries):
            x, y, z, u = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        elif (len(args) == 5) and isinstance(self, ParametricSurfaceSeries):
            x, y, z, u, v = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        elif (len(args) == 6) and self.is_3Dvector: # 3D vector plot
            x, y, z, u, v, w = args
            return (
                t(x, self._tx), t(y, self._ty), t(z, self._tz),
                t(u, self._tx), t(v, self._ty), t(w, self._tz)
            )
        elif len(args) == 6: # complex plot
            x, y, _abs, _arg, img, colors = args
            return (
                x, y, t(_abs, self._tz), _arg, img, colors)
        return args

    def _str_helper(self, s):
        pre, post = "", ""
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        return pre + s + post


def _detect_poles_numerical_helper(x, y, eps=0.01, expr=None, symb=None, symbolic=False):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN and record the
    corresponding x-location for further processing.

    Returns
    =======
    x : np.ndarray
        Unchanged x-data.
    yy : np.ndarray
        Modified y-data with NaN values.
    """
    np = import_module('numpy')

    yy = y.copy()
    threshold = np.pi / 2 - eps
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = abs(y[i + 1] - y[i])
        angle = np.arctan(dy / dx)
        if abs(angle) >= threshold:
            yy[i + 1] = np.nan

    return x, yy

def _detect_poles_symbolic_helper(expr, symb, start, end):
    """Attempts to compute symbolic discontinuities.

    Returns
    =======
    pole : list
        List of symbolic poles, possibly empty.
    """
    poles = []
    interval = Interval(nsimplify(start), nsimplify(end))
    res = continuous_domain(expr, symb, interval)
    res = res.simplify()
    if res == interval:
        pass
    elif (isinstance(res, Union) and
        all(isinstance(t, Interval) for t in res.args)):
        poles = []
        for s in res.args:
            if s.left_open:
                poles.append(s.left)
            if s.right_open:
                poles.append(s.right)
        poles = list(set(poles))
    else:
        raise ValueError(
            f"Could not parse the following object: {res} .\n"
            "Please, submit this as a bug. Consider also to set "
            "`detect_poles=True`."
        )
    return poles


### 2D lines
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """

    is_2Dline = True
    _dim = 2
    _N = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get("steps", False)
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.adaptive = kwargs.get("adaptive", False)
        self.depth = kwargs.get('depth', 12)
        self.use_cm = kwargs.get("use_cm", False)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.unwrap = kwargs.get("unwrap", False)
        # when detect_poles="symbolic", stores the location of poles so that
        # they can be appropriately rendered
        self.poles_locations = []
        exclude = kwargs.get("exclude", [])
        if isinstance(exclude, Set):
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, "__iter__"):
            exclude = [exclude]
        exclude = [float(e) for e in exclude]
        self.exclude = sorted(exclude)

    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
        np = import_module('numpy')
        points = self._get_data_helper()

        if (isinstance(self, LineOver1DRangeSeries) and
            (self.detect_poles == "symbolic")):
            poles = _detect_poles_symbolic_helper(
                self.expr.subs(self.params), *self.ranges[0])
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = t(np.array(poles), self._tx)

        # postprocessing
        points = self._apply_transform(*points)

        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                x, y = _detect_poles_numerical_helper(
                    x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)

        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            if self.is_2Dline:
                if len(points) == 2:
                    x, y = points
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    x, y, p = points
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)

        if self.steps is True:
            if self.is_2Dline:
                x, y = points[0], points[1]
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)

        if len(self.exclude) > 0:
            points = self._insert_exclusions(points)
        return points

    def get_segments(self):
        sympy_deprecation_warning(
            """
            The Line2DBaseSeries.get_segments() method is deprecated.

            Instead, use the MatplotlibBackend.get_segments() method, or use
            The get_points() or get_data() methods.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-get-segments")

        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exclusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
        np = import_module("numpy")
        points = list(points)
        n = len(points)
        # index of the x-coordinate (for 2d plots) or parameter (for 2d/3d
        # parametric plots)
        k = n - 1
        if n == 2:
            k = 0
        # indices of the other coordinates
        j_indeces = sorted(set(range(n)).difference([k]))
        # TODO: for now, I assume that numpy functions are going to succeed
        funcs = [f[0] for f in self._functions]

        for e in self.exclude:
            res = points[k] - e >= 0
            # if res contains both True and False, ie, if e is found
            if any(res) and any(~res):
                idx = np.nanargmax(res)
                # select the previous point with respect to e
                idx -= 1
                # TODO: what if points[k][idx]==e or points[k][idx+1]==e?

                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])
                    delta_post = abs(e - points[k][idx + 1])
                    delta = min(delta_prev, delta_post) / 100
                    prev = e - delta
                    post = e + delta

                    # add points to the x-coord or the parameter
                    points[k] = np.concatenate(
                        (points[k][:idx], [prev, e, post], points[k][idx+1:]))

                    # add points to the other coordinates
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))
                        c += 1
                        points[j] = np.concatenate(
                            (points[j][:idx], [values[0], np.nan, values[1]], points[j][idx+1:]))
        return points

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def xscale(self):
        return self._scales[0]

    @xscale.setter
    def xscale(self, v):
        self.scales = v

    def get_color_array(self):
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:  # only if the line is 3D (otherwise raises an error)
                    return f(*variables)
        else:
            return c*np.ones(self.nb_of_points)


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)
        np = import_module('numpy')
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(list_x)) +
                "and len(list_y) = {}".format(len(list_y))
            )
        self._block_lambda_functions(list_x, list_y)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)
            self.list_y = np.array(list_y, dtype=np.float64)

        self._expr = (self.list_x, self.list_y)
        if not any(isinstance(t, np.ndarray) for t in [self.list_x, self.list_y]):
            self._check_fs()
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.label = label
        self.rendering_kw = kwargs.get("rendering_kw", {})
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError(
                    "%s don't support symbolic " % self.__class__.__name__ +
                    "expression for `color_func`.")

    def __str__(self):
        return "2D list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._eval_color_func_and_return(lx, ly)

    def _eval_color_func_and_return(self, *data):
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]
        return data


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        self._cast = complex
        # for complex-related data series, this determines what data to return
        # on the y-axis
        self._return = kwargs.get("return", None)
        self._post_init()

        if not self._interactive_ranges:
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            start, end = [complex(t) for t in self.ranges[0][1:]]
            if im(start) != im(end):
                raise ValueError(
                    "%s requires the imaginary " % self.__class__.__name__ +
                    "part of the start and end values of the range "
                    "to be the same.")

        if self.adaptive and self._return:
            warnings.warn("The adaptive algorithm is unable to deal with "
                "complex numbers. Automatically switching to uniform meshing.")
            self.adaptive = False

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v

    def __str__(self):
        def f(t):
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = "interactive " if self.is_interactive else ""
        post = ""
        if self.is_interactive:
            post = " and parameters " + str(tuple(self.params.keys()))
        wrapper = _get_wrapper_for_expr(self._return)
        return pre + "cartesian line: %s for %s over %s" % (
            wrapper % self.expr,
            str(self.var),
            str((f(self.start), f(self.end))),
        ) + post

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates
        """
        return self._get_data_helper()

    def _adaptive_sampling(self):
        try:
            if callable(self.expr):
                f = self.expr
            else:
                f = lambdify([self.var], self.expr, self.modules)
            x, y = self._adaptive_sampling_helper(f)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f = lambdify([self.var], self.expr, "sympy")
            x, y = self._adaptive_sampling_helper(f)
        return x, y

    def _adaptive_sampling_helper(self, f):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
        np = import_module('numpy')

        x_coords = []
        y_coords = []
        def sample(p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            if self.xscale == 'log':
                xnew = 10**(np.log10(p[0]) + random * (np.log10(q[0]) -
                                                        np.log10(p[0])))
            else:
                xnew = p[0] + random * (q[0] - p[0])
            ynew = _adaptive_eval(f, xnew)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])

            # Sample to depth of 6 (whether the line is flat or not)
            # without using linspace (to avoid aliasing).
            elif depth < 6:
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif p[1] is None and q[1] is None:
                if self.xscale == 'log':
                    xarray = np.logspace(p[0], q[0], 10)
                else:
                    xarray = np.linspace(p[0], q[0], 10)
                yarray = list(map(f, xarray))
                if not all(y is None for y in yarray):
                    for i in range(len(yarray) - 1):
                        if not (yarray[i] is None and yarray[i + 1] is None):
                            sample([xarray[i], yarray[i]],
                                [xarray[i + 1], yarray[i + 1]], depth + 1)

            # Sample further if one of the end points in None (i.e. a
            # complex value) or the three points are not almost collinear.
            elif (p[1] is None or q[1] is None or new_point[1] is None
                    or not flat(p, new_point, q)):
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])

        f_start = _adaptive_eval(f, self.start.real)
        f_end = _adaptive_eval(f, self.end.real)
        x_coords.append(self.start.real)
        y_coords.append(f_start)
        sample(np.array([self.start.real, f_start]),
                np.array([self.end.real, f_end]), 0)

        return (x_coords, y_coords)

    def _uniform_sampling(self):
        np = import_module('numpy')

        x, result = self._evaluate()
        _re, _im = np.real(result), np.imag(result)
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        return x, _re, _im

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """
        np = import_module('numpy')
        if self.adaptive and (not self.only_integers):
            x, y = self._adaptive_sampling()
            return [np.array(t) for t in [x, y]]

        x, _re, _im = self._uniform_sampling()

        if self._return is None:
            # The evaluation could produce complex numbers. Set real elements
            # to NaN where there are non-zero imaginary elements
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == "real":
            pass
        elif self._return == "imag":
            _re = _im
        elif self._return == "abs":
            _re = np.sqrt(_re**2 + _im**2)
        elif self._return == "arg":
            _re = np.arctan2(_im, _re)
        else:
            raise ValueError("`_return` not recognized. "
                "Received: %s" % self._return)

        return x, _re


class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label = str(self.var) if label is None else label
        self._latex_label = latex(self.var) if label is None else label
        if (self.use_cm is False) and (self._label == str(self.var)):
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid surprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            if self._label == str(self.expr):
                self._label = ""

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            np = import_module("numpy")
            coords = self._adaptive_sampling()
            coords = [np.array(t) for t in coords]
        else:
            coords = self._uniform_sampling()

        if self.is_2Dline and self.is_polar:
            # when plot_polar is executed with polar_axis=True
            np = import_module('numpy')
            x, y, _ = coords
            r = np.sqrt(x**2 + y**2)
            t = np.arctan2(y, x)
            coords = [t, r, coords[-1]]

        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)

        return coords

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return [*results[1:], results[0]]

    def get_parameter_points(self):
        return self.get_data()[-1]

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates
            y : list
                List of y-coordinates
            z : list
                List of z-coordinates, only for 3D parametric line plot.
        """
        return self._get_data_helper()[:-1]

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v


class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""

    is_2Dline = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end))
        ))

    def _adaptive_sampling(self):
        try:
            if callable(self.expr_x) and callable(self.expr_y):
                f_x = self.expr_x
                f_y = self.expr_y
            else:
                f_x = lambdify([self.var], self.expr_x)
                f_y = lambdify([self.var], self.expr_y)
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f_x = lambdify([self.var], self.expr_x, "sympy")
            f_y = lambdify([self.var], self.expr_y, "sympy")
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        return x, y, p

    def _adaptive_sampling_helper(self, f_x, f_y):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
            Luiz Henrique de Figueiredo.
        """
        x_coords = []
        y_coords = []
        param = []

        def sample(param_p, param_q, p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            np = import_module('numpy')
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = _adaptive_eval(f_x, param_new)
            ynew = _adaptive_eval(f_y, param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < 6:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif ((p[0] is None and q[1] is None) or
                    (p[1] is None and q[1] is None)):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = [_adaptive_eval(f_x, t) for t in param_array]
                y_array = [_adaptive_eval(f_y, t) for t in param_array]
                if not all(x is None and y is None
                           for x, y in zip(x_array, y_array)):
                    for i in range(len(y_array) - 1):
                        if ((x_array[i] is not None and y_array[i] is not None) or
                                (x_array[i + 1] is not None and y_array[i + 1] is not None)):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(param_array[i], param_array[i], point_a,
                                   point_b, depth + 1)

            # Sample further if one of the end points in None (i.e. a complex
            # value) or the three points are not almost collinear.
            elif (p[0] is None or p[1] is None
                    or q[1] is None or q[0] is None
                    or not flat(p, new_point, q)):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

        f_start_x = _adaptive_eval(f_x, self.start)
        f_start_y = _adaptive_eval(f_y, self.start)
        start = [f_start_x, f_start_y]
        f_end_x = _adaptive_eval(f_x, self.end)
        f_end_y = _adaptive_eval(f_y, self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(self.start)
        sample(self.start, self.end, start, end, 0)

        return x_coords, y_coords, param


### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False
    is_3Dline = True
    _dim = 3

    def __init__(self):
        super().__init__()


class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""

    is_2Dline = False
    is_3Dline = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._cast = float
        self.adaptive = False
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))

    def get_data(self):
        # TODO: remove this
        np = import_module("numpy")
        x, y, z, p = super().get_data()
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        return x, y, z, p


### Surfaces
class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.use_cm = kwargs.get("use_cm", False)
        # NOTE: why should SurfaceOver2DRangeSeries support is polar?
        # After all, the same result can be achieve with
        # ParametricSurfaceSeries. For example:
        # sin(r) for (r, 0, 2 * pi) and (theta, 0, pi/2) can be parameterized
        # as (r * cos(theta), r * sin(theta), sin(t)) for (r, 0, 2 * pi) and
        # (theta, 0, pi/2).
        # Because it is faster to evaluate (important for interactive plots).
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        exprs = self.expr
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # surprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self._label = ""
                self._latex_label = ""

    def get_color_array(self):
        np = import_module('numpy')
        c = self.surface_color
        if isinstance(c, Callable):
            f = np.vectorize(c)
            nargs = arity(c)
            if self.is_parametric:
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            variables = list(map(centers_of_faces, self.get_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        else:
            if isinstance(self, SurfaceOver2DRangeSeries):
                return c*np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
            else:
                return c*np.ones(min(self.nb_of_points_u, self.nb_of_points_v))


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._set_surface_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    @property
    def var_x(self):
        return self.ranges[0][0]

    @property
    def var_y(self):
        return self.ranges[1][0]

    @property
    def start_x(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_x(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_y(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_y(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_x(self):
        return self.n[0]

    @nb_of_points_x.setter
    def nb_of_points_x(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_y(self):
        return self.n[1]

    @nb_of_points_y.setter
    def nb_of_points_y(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        ))

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray
            Discretized x-domain.
        mesh_y : np.ndarray
            Discretized y-domain.
        mesh_z : np.ndarray
            Results of the evaluation.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        x, y, z = results
        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)

        # TODO: remove this
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(x, y, z)


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric SymPy
    expressions and a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_u(self):
        return self.ranges[0][0]

    @property
    def var_v(self):
        return self.ranges[1][0]

    @property
    def start_u(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_u(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_v(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_v(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_u(self):
        return self.n[0]

    @nb_of_points_u.setter
    def nb_of_points_u(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_v(self):
        return self.n[1]

    @nb_of_points_v.setter
    def nb_of_points_v(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        ))

    def get_parameter_meshes(self):
        return self.get_data()[3:]

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()[:3]

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        # TODO: remove this
        x, y, z = results[2:]
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(*results[2:], *results[:2])


### Contours
class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.show_clabels = kwargs.get("clabels", True)

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", {}))


class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readability. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    """
    is_generic = True

    def __init__(self, tp, *args, **kwargs):
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        return self.args


class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""

    is_implicit = True
    use_cm = False
    _N = 100

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.adaptive = kwargs.get("adaptive", False)
        self.expr = expr
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.ranges = [var_start_end_x, var_start_end_y]
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self._color = kwargs.get("color", kwargs.get("line_color", None))

        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
        self._post_init()

    @property
    def expr(self):
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    @expr.setter
    def expr(self, expr):
        self._block_lambda_functions(expr)
        # these are needed for adaptive evaluation
        expr, has_equality = self._has_equality(sympify(expr))
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self._label = str(expr)
        self._latex_label = latex(expr)

        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            self.adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(
                "The provided expression " + msg
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            # these are needed for uniform meshing evaluation
            expr, is_equality = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

    @property
    def line_color(self):
        return self._color

    @line_color.setter
    def line_color(self, v):
        self._color = v

    color = line_color

    def _has_equality(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan
        # or LessThan
        has_equality = False

        def arg_expand(bool_expr):
            """Recursively expands the arguments of an Boolean Function"""
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # Check whether there is an equality in the expression provided.
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

    def __str__(self):
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            "Implicit expression: %s for %s over %s and %s over %s") % (
            str(self._adaptive_expr),
            str(self.var_x),
            str((f(self.start_x), f(self.end_x))),
            str(self.var_y),
            str((f(self.start_y), f(self.end_y))),
        )

    def get_data(self):
        """Returns numerical data.

        Returns
        =======

        If the series is evaluated with the `adaptive=True` it returns:

        interval_list : list
            List of bounding rectangular intervals to be postprocessed and
            eventually used with Matplotlib's ``fill`` command.
        dummy : str
            A string containing ``"fill"``.

        Otherwise, it returns 2D numpy arrays to be used with Matplotlib's
        ``contour`` or ``contourf`` commands:

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            A string specifying which plot command to use, ``"contour"``
            or ``"contourf"``.
        """
        if self.adaptive:
            data = self._adaptive_eval()
            if data is not None:
                return data

        return self._get_meshes_grid()

    def _adaptive_eval(self):
        """
        References
        ==========

        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
        Mathematical Formulae with Two Free Variables.

        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
        Arithmetic. Master's thesis. University of Toronto, 1996
        """
        import sympy.plotting.intervalmath.lib_interval as li

        user_functions = {}
        printer = IntervalMathPrinter({
            'fully_qualified_modules': False, 'inline': True,
            'allow_unknown_functions': True,
            'user_functions': user_functions})

        keys = [t for t in dir(li) if ("__" not in t) and (t not in ["import_module", "interval"])]
        vals = [getattr(li, k) for k in keys]
        d = dict(zip(keys, vals))
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)
        data = None

        try:
            data = self._get_raster_interval(func)
        except NameError as err:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, as some functions are not yet implemented"
                " in the interval math module:\n\n"
                "NameError: %s\n\n" % err +
                "Proceeding with uniform meshing."
                )
            self.adaptive = False
        except TypeError:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression. Using uniform meshing.")
            self.adaptive = False

        return data

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self.depth
        interval_list = []
        sx, sy = [float(t) for t in [self.start_x, self.start_y]]
        ex, ey = [float(t) for t in [self.end_x, self.end_y]]
        # Create initial 32 divisions
        xsample = np.linspace(sx, ex, 33)
        ysample = np.linspace(sy, ey, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (ex - sx)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (ey - sy)
            / 2 ** 20
        )
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        # recursive call refinepixels which subdivides the intervals which are
        # neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                # Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                # The expression is valid in the interval. Change the contour
                # array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    # Subdivide
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return temp_interval_list, plot_list

        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        # Check whether the expression represents an equality
        # If it represents an equality, then none of the intervals
        # would have satisfied the expression due to floating point
        # differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, "fill"

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        np = import_module('numpy')

        xarray, yarray, z_grid = self._evaluate()
        _re, _im = np.real(z_grid), np.imag(z_grid)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        if self._is_equality:
            return xarray, yarray, _re, 'contour'
        return xarray, yarray, _re, 'contourf'

    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Whether the original expression was an Equality or not.
        """
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        elif isinstance(expr, Relational):
            expr = expr.gts - expr.lts
        elif not adaptive:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

def centers_of_segments(array):
    np = import_module('numpy')
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    np = import_module('numpy')
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
    np = import_module('numpy')
    # Workaround plotting piecewise (#8577)
    vector_a = (x - y).astype(float)
    vector_b = (z - y).astype(float)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments ``n, n1, n2`` to
    specify the number of discretization points in one and two
    directions, while keeping back-compatibility with older keyword arguments
    like, ``nb_of_points, nb_of_points_*, points``.

    Parameters
    ==========

    kwargs : dict
        Dictionary of keyword arguments passed into a plotting function.
    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.
    """
    replace_old_keywords = {
        "nb_of_points": "n",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in replace_old_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)

    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
        Parametric3DLineSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n1"] = kwargs["n"][0]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries,
        ParametricSurfaceSeries, ImplicitSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    return kwargs
<|endoftext|>from .plot import plot_backends
from .plot_implicit import plot_implicit
from .textplot import textplot
from .pygletplot import PygletPlot
from .plot import PlotGrid
from .plot import (plot, plot_parametric, plot3d, plot3d_parametric_surface,
                  plot3d_parametric_line, plot_contour)

__all__ = [
    'plot_backends',

    'plot_implicit',

    'textplot',

    'PygletPlot',

    'PlotGrid',

    'plot', 'plot_parametric', 'plot3d', 'plot3d_parametric_surface',
    'plot3d_parametric_line', 'plot_contour'
]
<|endoftext|>from sympy.core.numbers import Float
from sympy.core.symbol import Dummy
from sympy.utilities.lambdify import lambdify

import math


def is_valid(x):
    """Check if a floating point number is valid"""
    if x is None:
        return False
    if isinstance(x, complex):
        return False
    return not math.isinf(x) and not math.isnan(x)


def rescale(y, W, H, mi, ma):
    """Rescale the given array `y` to fit into the integer values
    between `0` and `H-1` for the values between ``mi`` and ``ma``.
    """
    y_new = []

    norm = ma - mi
    offset = (ma + mi) / 2

    for x in range(W):
        if is_valid(y[x]):
            normalized = (y[x] - offset) / norm
            if not is_valid(normalized):
                y_new.append(None)
            else:
                rescaled = Float((normalized*H + H/2) * (H-1)/H).round()
                rescaled = int(rescaled)
                y_new.append(rescaled)
        else:
            y_new.append(None)
    return y_new


def linspace(start, stop, num):
    return [start + (stop - start) * x / (num-1) for x in range(num)]


def textplot_str(expr, a, b, W=55, H=21):
    """Generator for the lines of the plot"""
    free = expr.free_symbols
    if len(free) > 1:
        raise ValueError(
            "The expression must have a single variable. (Got {})"
            .format(free))
    x = free.pop() if free else Dummy()
    f = lambdify([x], expr)
    if isinstance(a, complex):
        if a.imag == 0:
            a = a.real
    if isinstance(b, complex):
        if b.imag == 0:
            b = b.real
    a = float(a)
    b = float(b)

    # Calculate function values
    x = linspace(a, b, W)
    y = []
    for val in x:
        try:
            y.append(f(val))
        # Not sure what exceptions to catch here or why...
        except (ValueError, TypeError, ZeroDivisionError):
            y.append(None)

    # Normalize height to screen space
    y_valid = list(filter(is_valid, y))
    if y_valid:
        ma = max(y_valid)
        mi = min(y_valid)
        if ma == mi:
            if ma:
                mi, ma = sorted([0, 2*ma])
            else:
                mi, ma = -1, 1
    else:
        mi, ma = -1, 1
    y_range = ma - mi
    precision = math.floor(math.log10(y_range)) - 1
    precision *= -1
    mi = round(mi, precision)
    ma = round(ma, precision)
    y = rescale(y, W, H, mi, ma)

    y_bins = linspace(mi, ma, H)

    # Draw plot
    margin = 7
    for h in range(H - 1, -1, -1):
        s = [' '] * W
        for i in range(W):
            if y[i] == h:
                if (i == 0 or y[i - 1] == h - 1) and (i == W - 1 or y[i + 1] == h + 1):
                    s[i] = '/'
                elif (i == 0 or y[i - 1] == h + 1) and (i == W - 1 or y[i + 1] == h - 1):
                    s[i] = '\\'
                else:
                    s[i] = '.'

        if h == 0:
            for i in range(W):
                s[i] = '_'

        # Print y values
        if h in (0, H//2, H - 1):
            prefix = ("%g" % y_bins[h]).rjust(margin)[:margin]
        else:
            prefix = " "*margin
        s = "".join(s)
        if h == H//2:
            s = s.replace(" ", "-")
        yield prefix + " |" + s

    # Print x values
    bottom = " " * (margin + 2)
    bottom += ("%g" % x[0]).ljust(W//2)
    if W % 2 == 1:
        bottom += ("%g" % x[W//2]).ljust(W//2)
    else:
        bottom += ("%g" % x[W//2]).ljust(W//2-1)
    bottom += "%g" % x[-1]
    yield bottom


def textplot(expr, a, b, W=55, H=21):
    r"""
    Print a crude ASCII art plot of the SymPy expression 'expr' (which
    should contain a single symbol, e.g. x or something else) over the
    interval [a, b].

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.plotting import textplot
    >>> t = Symbol('t')
    >>> textplot(sin(t)*t, 0, 15)
     14 |                                                  ...
        |                                                     .
        |                                                 .
        |                                                      .
        |                                                .
        |                            ...
        |                           /   .               .
        |                          /
        |                         /      .
        |                        .        .            .
    1.5 |----.......--------------------------------------------
        |....       \           .          .
        |            \         /                      .
        |             ..      /             .
        |               \    /                       .
        |                ....
        |                                    .
        |                                     .     .
        |
        |                                      .   .
    -11 |_______________________________________________________
         0                          7.5                        15
    """
    for line in textplot_str(expr, a, b, W, H):
        print(line)
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift


class ColorGradient:
    colors = [0.4, 0.4, 0.4], [0.9, 0.9, 0.9]
    intervals = 0.0, 1.0

    def __init__(self, *args):
        if len(args) == 2:
            self.colors = list(args)
            self.intervals = [0.0, 1.0]
        elif len(args) > 0:
            if len(args) % 2 != 0:
                raise ValueError("len(args) should be even")
            self.colors = [args[i] for i in range(1, len(args), 2)]
            self.intervals = [args[i] for i in range(0, len(args), 2)]
        assert len(self.colors) == len(self.intervals)

    def copy(self):
        c = ColorGradient()
        c.colors = [e[::] for e in self.colors]
        c.intervals = self.intervals[::]
        return c

    def _find_interval(self, v):
        m = len(self.intervals)
        i = 0
        while i < m - 1 and self.intervals[i] <= v:
            i += 1
        return i

    def _interpolate_axis(self, axis, v):
        i = self._find_interval(v)
        v = rinterpolate(self.intervals[i - 1], self.intervals[i], v)
        return interpolate(self.colors[i - 1][axis], self.colors[i][axis], v)

    def __call__(self, r, g, b):
        c = self._interpolate_axis
        return c(0, r), c(1, g), c(2, b)

default_color_schemes = {}  # defined at the bottom of this file


class ColorScheme:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.f, self.gradient = None, ColorGradient()

        if len(args) == 1 and not isinstance(args[0], Basic) and callable(args[0]):
            self.f = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            if args[0] in default_color_schemes:
                cs = default_color_schemes[args[0]]
                self.f, self.gradient = cs.f, cs.gradient.copy()
            else:
                self.f = lambdify('x,y,z,u,v', args[0])
        else:
            self.f, self.gradient = self._interpret_args(args)
        self._test_color_function()
        if not isinstance(self.gradient, ColorGradient):
            raise ValueError("Color gradient not properly initialized. "
                             "(Not a ColorGradient instance.)")

    def _interpret_args(self, args):
        f, gradient = None, self.gradient
        atoms, lists = self._sort_args(args)
        s = self._pop_symbol_list(lists)
        s = self._fill_in_vars(s)

        # prepare the error message for lambdification failure
        f_str = ', '.join(str(fa) for fa in atoms)
        s_str = (str(sa) for sa in s)
        s_str = ', '.join(sa for sa in s_str if sa.find('unbound') < 0)
        f_error = ValueError("Could not interpret arguments "
                             "%s as functions of %s." % (f_str, s_str))

        # try to lambdify args
        if len(atoms) == 1:
            fv = atoms[0]
            try:
                f = lambdify(s, [fv, fv, fv])
            except TypeError:
                raise f_error

        elif len(atoms) == 3:
            fr, fg, fb = atoms
            try:
                f = lambdify(s, [fr, fg, fb])
            except TypeError:
                raise f_error

        else:
            raise ValueError("A ColorScheme must provide 1 or 3 "
                             "functions in x, y, z, u, and/or v.")

        # try to intrepret any given color information
        if len(lists) == 0:
            gargs = []

        elif len(lists) == 1:
            gargs = lists[0]

        elif len(lists) == 2:
            try:
                (r1, g1, b1), (r2, g2, b2) = lists
            except TypeError:
                raise ValueError("If two color arguments are given, "
                                 "they must be given in the format "
                                 "(r1, g1, b1), (r2, g2, b2).")
            gargs = lists

        elif len(lists) == 3:
            try:
                (r1, r2), (g1, g2), (b1, b2) = lists
            except Exception:
                raise ValueError("If three color arguments are given, "
                                 "they must be given in the format "
                                 "(r1, r2), (g1, g2), (b1, b2). To create "
                                 "a multi-step gradient, use the syntax "
                                 "[0, colorStart, step1, color1, ..., 1, "
                                 "colorEnd].")
            gargs = [[r1, g1, b1], [r2, g2, b2]]

        else:
            raise ValueError("Don't know what to do with collection "
                             "arguments %s." % (', '.join(str(l) for l in lists)))

        if gargs:
            try:
                gradient = ColorGradient(*gargs)
            except Exception as ex:
                raise ValueError(("Could not initialize a gradient "
                                  "with arguments %s. Inner "
                                  "exception: %s") % (gargs, str(ex)))

        return f, gradient

    def _pop_symbol_list(self, lists):
        symbol_lists = []
        for l in lists:
            mark = True
            for s in l:
                if s is not None and not isinstance(s, Symbol):
                    mark = False
                    break
            if mark:
                lists.remove(l)
                symbol_lists.append(l)
        if len(symbol_lists) == 1:
            return symbol_lists[0]
        elif len(symbol_lists) == 0:
            return []
        else:
            raise ValueError("Only one list of Symbols "
                             "can be given for a color scheme.")

    def _fill_in_vars(self, args):
        defaults = symbols('x,y,z,u,v')
        v_error = ValueError("Could not find what to plot.")
        if len(args) == 0:
            return defaults
        if not isinstance(args, (tuple, list)):
            raise v_error
        if len(args) == 0:
            return defaults
        for s in args:
            if s is not None and not isinstance(s, Symbol):
                raise v_error
        # when vars are given explicitly, any vars
        # not given are marked 'unbound' as to not
        # be accidentally used in an expression
        vars = [Symbol('unbound%i' % (i)) for i in range(1, 6)]
        # interpret as t
        if len(args) == 1:
            vars[3] = args[0]
        # interpret as u,v
        elif len(args) == 2:
            if args[0] is not None:
                vars[3] = args[0]
            if args[1] is not None:
                vars[4] = args[1]
        # interpret as x,y,z
        elif len(args) >= 3:
            # allow some of x,y,z to be
            # left unbound if not given
            if args[0] is not None:
                vars[0] = args[0]
            if args[1] is not None:
                vars[1] = args[1]
            if args[2] is not None:
                vars[2] = args[2]
            # interpret the rest as t
            if len(args) >= 4:
                vars[3] = args[3]
                # ...or u,v
                if len(args) >= 5:
                    vars[4] = args[4]
        return vars

    def _sort_args(self, args):
        lists, atoms = sift(args,
            lambda a: isinstance(a, (tuple, list)), binary=True)
        return atoms, lists

    def _test_color_function(self):
        if not callable(self.f):
            raise ValueError("Color function is not callable.")
        try:
            result = self.f(0, 0, 0, 0, 0)
            if len(result) != 3:
                raise ValueError("length should be equal to 3")
        except TypeError:
            raise ValueError("Color function needs to accept x,y,z,u,v, "
                             "as arguments even if it doesn't use all of them.")
        except AssertionError:
            raise ValueError("Color function needs to return 3-tuple r,g,b.")
        except Exception:
            pass  # color function probably not valid at 0,0,0,0,0

    def __call__(self, x, y, z, u, v):
        try:
            return self.f(x, y, z, u, v)
        except Exception:
            return None

    def apply_to_curve(self, verts, u_set, set_len=None, inc_pos=None):
        """
        Apply this color scheme to a
        set of vertices over a single
        independent variable u.
        """
        bounds = create_bounds()
        cverts = []
        if callable(set_len):
            set_len(len(u_set)*2)
        # calculate f() = r,g,b for each vert
        # and find the min and max for r,g,b
        for _u in range(len(u_set)):
            if verts[_u] is None:
                cverts.append(None)
            else:
                x, y, z = verts[_u]
                u, v = u_set[_u], None
                c = self(x, y, z, u, v)
                if c is not None:
                    c = list(c)
                    update_bounds(bounds, c)
                cverts.append(c)
            if callable(inc_pos):
                inc_pos()
        # scale and apply gradient
        for _u in range(len(u_set)):
            if cverts[_u] is not None:
                for _c in range(3):
                    # scale from [f_min, f_max] to [0,1]
                    cverts[_u][_c] = rinterpolate(bounds[_c][0], bounds[_c][1],
                                                  cverts[_u][_c])
                # apply gradient
                cverts[_u] = self.gradient(*cverts[_u])
            if callable(inc_pos):
                inc_pos()
        return cverts

    def apply_to_surface(self, verts, u_set, v_set, set_len=None, inc_pos=None):
        """
        Apply this color scheme to a
        set of vertices over two
        independent variables u and v.
        """
        bounds = create_bounds()
        cverts = []
        if callable(set_len):
            set_len(len(u_set)*len(v_set)*2)
        # calculate f() = r,g,b for each vert
        # and find the min and max for r,g,b
        for _u in range(len(u_set)):
            column = []
            for _v in range(len(v_set)):
                if verts[_u][_v] is None:
                    column.append(None)
                else:
                    x, y, z = verts[_u][_v]
                    u, v = u_set[_u], v_set[_v]
                    c = self(x, y, z, u, v)
                    if c is not None:
                        c = list(c)
                        update_bounds(bounds, c)
                    column.append(c)
                if callable(inc_pos):
                    inc_pos()
            cverts.append(column)
        # scale and apply gradient
        for _u in range(len(u_set)):
            for _v in range(len(v_set)):
                if cverts[_u][_v] is not None:
                    # scale from [f_min, f_max] to [0,1]
                    for _c in range(3):
                        cverts[_u][_v][_c] = rinterpolate(bounds[_c][0],
                                             bounds[_c][1], cverts[_u][_v][_c])
                    # apply gradient
                    cverts[_u][_v] = self.gradient(*cverts[_u][_v])
                if callable(inc_pos):
                    inc_pos()
        return cverts

    def str_base(self):
        return ", ".join(str(a) for a in self.args)

    def __repr__(self):
        return "%s" % (self.str_base())


x, y, z, t, u, v = symbols('x,y,z,t,u,v')

default_color_schemes['rainbow'] = ColorScheme(z, y, x)
default_color_schemes['zfade'] = ColorScheme(z, (0.4, 0.4, 0.97),
                                             (0.97, 0.4, 0.4), (None, None, z))
default_color_schemes['zfade3'] = ColorScheme(z, (None, None, z),
                                              [0.00, (0.2, 0.2, 1.0),
                                               0.35, (0.2, 0.8, 0.4),
                                               0.50, (0.3, 0.9, 0.3),
                                               0.65, (0.4, 0.8, 0.2),
                                               1.00, (1.0, 0.2, 0.2)])

default_color_schemes['zfade4'] = ColorScheme(z, (None, None, z),
                                              [0.0, (0.3, 0.3, 1.0),
                                               0.30, (0.3, 1.0, 0.3),
                                               0.55, (0.95, 1.0, 0.2),
                                               0.65, (1.0, 0.95, 0.2),
                                               0.85, (1.0, 0.7, 0.2),
                                               1.0, (1.0, 0.3, 0.2)])
<|endoftext|>class PlotObject:
    """
    Base class for objects which can be displayed in
    a Plot.
    """
    visible = True

    def _draw(self):
        if self.visible:
            self.draw()

    def draw(self):
        """
        OpenGL rendering code for the plot object.
        Override in base class.
        """
        pass
<|endoftext|>from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface

from math import sin as p_sin
from math import cos as p_cos


def float_vec3(f):
    def inner(*args):
        v = f(*args)
        return float(v[0]), float(v[1]), float(v[2])
    return inner


class Cartesian2D(PlotCurve):
    i_vars, d_vars = 'x', 'y'
    intervals = [[-5, 5, 100]]
    aliases = ['cartesian']
    is_default = True

    def _get_sympy_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v

        @float_vec3
        def e(_x):
            return (_x, fy.subs(x, _x), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v
        return lambdify([x], [x, fy, 0.0])


class Cartesian3D(PlotSurface):
    i_vars, d_vars = 'xy', 'z'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['cartesian', 'monge']
    is_default = True

    def _get_sympy_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v

        @float_vec3
        def e(_x, _y):
            return (_x, _y, fz.subs(x, _x).subs(y, _y))
        return e

    def _get_lambda_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v
        return lambdify([x, y], [x, y, fz])


class ParametricCurve2D(PlotCurve):
    i_vars, d_vars = 't', 'xy'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy = self.d_vars
        t = self.t_interval.v

        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fx, fy = self.d_vars
        t = self.t_interval.v
        return lambdify([t], [fx, fy, 0.0])


class ParametricCurve3D(PlotCurve):
    i_vars, d_vars = 't', 'xyz'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v

        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), fz.subs(t, _t))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v
        return lambdify([t], [fx, fy, fz])


class ParametricSurface(PlotSurface):
    i_vars, d_vars = 'uv', 'xyz'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v

        @float_vec3
        def e(_u, _v):
            return (fx.subs(u, _u).subs(v, _v),
                    fy.subs(u, _u).subs(v, _v),
                    fz.subs(u, _u).subs(v, _v))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v
        return lambdify([u, v], [fx, fy, fz])


class Polar(PlotCurve):
    i_vars, d_vars = 't', 'r'
    intervals = [[0, 2*pi, 100]]
    aliases = ['polar']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.t_interval.v

        def e(_t):
            _r = float(fr.subs(t, _t))
            return (_r*p_cos(_t), _r*p_sin(_t), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.t_interval.v
        fx, fy = fr*cos(t), fr*sin(t)
        return lambdify([t], [fx, fy, 0.0])


class Cylindrical(PlotSurface):
    i_vars, d_vars = 'th', 'r'
    intervals = [[0, 2*pi, 40], [-1, 1, 20]]
    aliases = ['cylindrical', 'polar']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v

        def e(_t, _h):
            _r = float(fr.subs(t, _t).subs(h, _h))
            return (_r*p_cos(_t), _r*p_sin(_t), _h)
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v
        fx, fy = fr*cos(t), fr*sin(t)
        return lambdify([t, h], [fx, fy, h])


class Spherical(PlotSurface):
    i_vars, d_vars = 'tp', 'r'
    intervals = [[0, 2*pi, 40], [0, pi, 20]]
    aliases = ['spherical']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v

        def e(_t, _p):
            _r = float(fr.subs(t, _t).subs(p, _p))
            return (_r*p_cos(_t)*p_sin(_p),
                    _r*p_sin(_t)*p_sin(_p),
                    _r*p_cos(_p))
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v
        fx = fr * cos(t) * sin(p)
        fy = fr * sin(t) * sin(p)
        fz = fr * cos(p)
        return lambdify([t, p], [fx, fy, fz])

Cartesian2D._register()
Cartesian3D._register()
ParametricCurve2D._register()
ParametricCurve3D._register()
ParametricSurface._register()
Polar._register()
Cylindrical._register()
Spherical._register()
<|endoftext|>import pyglet.gl as pgl

from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase


class PlotSurface(PlotModeBase):

    default_rot_preset = 'perspective'

    def _on_calculate_verts(self):
        self.u_interval = self.intervals[0]
        self.u_set = list(self.u_interval.frange())
        self.v_interval = self.intervals[1]
        self.v_set = list(self.v_interval.frange())
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        evaluate = self._get_evaluator()

        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = float(
            self.u_interval.v_len*self.v_interval.v_len)

        verts = []
        b = self.bounds
        for u in self.u_set:
            column = []
            for v in self.v_set:
                try:
                    _e = evaluate(u, v)  # calculate vertex
                except ZeroDivisionError:
                    _e = None
                if _e is not None:  # update bounding box
                    for axis in range(3):
                        b[axis][0] = min([b[axis][0], _e[axis]])
                        b[axis][1] = max([b[axis][1], _e[axis]])
                column.append(_e)
                self._calculating_verts_pos += 1.0

            verts.append(column)
        for axis in range(3):
            b[axis][2] = b[axis][1] - b[axis][0]
            if b[axis][2] == 0.0:
                b[axis][2] = 1.0

        self.verts = verts
        self.push_wireframe(self.draw_verts(False, False))
        self.push_solid(self.draw_verts(False, True))

    def _on_calculate_cverts(self):
        if not self.verts or not self.color:
            return

        def set_work_len(n):
            self._calculating_cverts_len = float(n)

        def inc_work_pos():
            self._calculating_cverts_pos += 1.0
        set_work_len(1)
        self._calculating_cverts_pos = 0
        self.cverts = self.color.apply_to_surface(self.verts,
                                                  self.u_set,
                                                  self.v_set,
                                                  set_len=set_work_len,
                                                  inc_pos=inc_work_pos)
        self.push_solid(self.draw_verts(True, True))

    def calculate_one_cvert(self, u, v):
        vert = self.verts[u][v]
        return self.color(vert[0], vert[1], vert[2],
                          self.u_set[u], self.v_set[v])

    def draw_verts(self, use_cverts, use_solid_color):
        def f():
            for u in range(1, len(self.u_set)):
                pgl.glBegin(pgl.GL_QUAD_STRIP)
                for v in range(len(self.v_set)):
                    pa = self.verts[u - 1][v]
                    pb = self.verts[u][v]
                    if pa is None or pb is None:
                        pgl.glEnd()
                        pgl.glBegin(pgl.GL_QUAD_STRIP)
                        continue
                    if use_cverts:
                        ca = self.cverts[u - 1][v]
                        cb = self.cverts[u][v]
                        if ca is None:
                            ca = (0, 0, 0)
                        if cb is None:
                            cb = (0, 0, 0)
                    else:
                        if use_solid_color:
                            ca = cb = self.default_solid_color
                        else:
                            ca = cb = self.default_wireframe_color
                    pgl.glColor3f(*ca)
                    pgl.glVertex3f(*pa)
                    pgl.glColor3f(*cb)
                    pgl.glVertex3f(*pb)
                pgl.glEnd()
        return f
<|endoftext|>from pyglet.window import Window
from pyglet.clock import Clock

from threading import Thread, Lock

gl_lock = Lock()


class ManagedWindow(Window):
    """
    A pyglet window with an event loop which executes automatically
    in a separate thread. Behavior is added by creating a subclass
    which overrides setup, update, and/or draw.
    """
    fps_limit = 30
    default_win_args = {"width": 600,
                            "height": 500,
                            "vsync": False,
                            "resizable": True}

    def __init__(self, **win_args):
        """
        It is best not to override this function in the child
        class, unless you need to take additional arguments.
        Do any OpenGL initialization calls in setup().
        """

        # check if this is run from the doctester
        if win_args.get('runfromdoctester', False):
            return

        self.win_args = dict(self.default_win_args, **win_args)
        self.Thread = Thread(target=self.__event_loop__)
        self.Thread.start()

    def __event_loop__(self, **win_args):
        """
        The event loop thread function. Do not override or call
        directly (it is called by __init__).
        """
        gl_lock.acquire()
        try:
            try:
                super().__init__(**self.win_args)
                self.switch_to()
                self.setup()
            except Exception as e:
                print("Window initialization failed: %s" % (str(e)))
                self.has_exit = True
        finally:
            gl_lock.release()

        clock = Clock()
        clock.fps_limit = self.fps_limit
        while not self.has_exit:
            dt = clock.tick()
            gl_lock.acquire()
            try:
                try:
                    self.switch_to()
                    self.dispatch_events()
                    self.clear()
                    self.update(dt)
                    self.draw()
                    self.flip()
                except Exception as e:
                    print("Uncaught exception in event loop: %s" % str(e))
                    self.has_exit = True
            finally:
                gl_lock.release()
        super().close()

    def close(self):
        """
        Closes the window.
        """
        self.has_exit = True

    def setup(self):
        """
        Called once before the event loop begins.
        Override this method in a child class. This
        is the best place to put things like OpenGL
        initialization calls.
        """
        pass

    def update(self, dt):
        """
        Called before draw during each iteration of
        the event loop. dt is the elapsed time in
        seconds since the last update. OpenGL rendering
        calls are best put in draw() rather than here.
        """
        pass

    def draw(self):
        """
        Called after update during each iteration of
        the event loop. Put OpenGL rendering calls
        here.
        """
        pass

if __name__ == '__main__':
    ManagedWindow()
<|endoftext|>try:
    from ctypes import c_float, c_int, c_double
except ImportError:
    pass

import pyglet.gl as pgl
from sympy.core import S


def get_model_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    m = (array_type*16)()
    glGetMethod(pgl.GL_MODELVIEW_MATRIX, m)
    return m


def get_projection_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    m = (array_type*16)()
    glGetMethod(pgl.GL_PROJECTION_MATRIX, m)
    return m


def get_viewport():
    """
    Returns the current viewport.
    """
    m = (c_int*4)()
    pgl.glGetIntegerv(pgl.GL_VIEWPORT, m)
    return m


def get_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[4], m[8]),
            (m[1], m[5], m[9]),
            (m[2], m[6], m[10]))


def get_view_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[1], m[2]),
            (m[4], m[5], m[6]),
            (m[8], m[9], m[10]))


def get_basis_vectors():
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def screen_to_model(x, y, z):
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    w = get_viewport()
    mx, my, mz = c_double(), c_double(), c_double()
    pgl.gluUnProject(x, y, z, m, p, w, mx, my, mz)
    return float(mx.value), float(my.value), float(mz.value)


def model_to_screen(x, y, z):
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    w = get_viewport()
    mx, my, mz = c_double(), c_double(), c_double()
    pgl.gluProject(x, y, z, m, p, w, mx, my, mz)
    return float(mx.value), float(my.value), float(mz.value)


def vec_subs(a, b):
    return tuple(a[i] - b[i] for i in range(len(a)))


def billboard_matrix():
    """
    Removes rotational components of
    current matrix so that primitives
    are always drawn facing the viewer.

    |1|0|0|x|
    |0|1|0|x|
    |0|0|1|x| (x means left unchanged)
    |x|x|x|x|
    """
    m = get_model_matrix()
    # XXX: for i in range(11): m[i] = i ?
    m[0] = 1
    m[1] = 0
    m[2] = 0
    m[4] = 0
    m[5] = 1
    m[6] = 0
    m[8] = 0
    m[9] = 0
    m[10] = 1
    pgl.glLoadMatrixf(m)


def create_bounds():
    return [[S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0]]


def update_bounds(b, v):
    if v is None:
        return
    for axis in range(3):
        b[axis][0] = min([b[axis][0], v[axis]])
        b[axis][1] = max([b[axis][1], v[axis]])


def interpolate(a_min, a_max, a_ratio):
    return a_min + a_ratio * (a_max - a_min)


def rinterpolate(a_min, a_max, a_value):
    a_range = a_max - a_min
    if a_max == a_min:
        a_range = 1.0
    return (a_value - a_min) / float(a_range)


def interpolate_color(color1, color2, ratio):
    return tuple(interpolate(color1[i], color2[i], ratio) for i in range(3))


def scale_value(v, v_min, v_len):
    return (v - v_min) / v_len


def scale_value_list(flist):
    v_min, v_max = min(flist), max(flist)
    v_len = v_max - v_min
    return [scale_value(f, v_min, v_len) for f in flist]


def strided_range(r_min, r_max, stride, max_steps=50):
    o_min, o_max = r_min, r_max
    if abs(r_min - r_max) < 0.001:
        return []
    try:
        range(int(r_min - r_max))
    except (TypeError, OverflowError):
        return []
    if r_min > r_max:
        raise ValueError("r_min cannot be greater than r_max")
    r_min_s = (r_min % stride)
    r_max_s = stride - (r_max % stride)
    if abs(r_max_s - stride) < 0.001:
        r_max_s = 0.0
    r_min -= r_min_s
    r_max += r_max_s
    r_steps = int((r_max - r_min)/stride)
    if max_steps and r_steps > max_steps:
        return strided_range(o_min, o_max, stride*2)
    return [r_min] + [r_min + e*stride for e in range(1, r_steps + 1)] + [r_max]


def parse_option_string(s):
    if not isinstance(s, str):
        return None
    options = {}
    for token in s.split(';'):
        pieces = token.split('=')
        if len(pieces) == 1:
            option, value = pieces[0], ""
        elif len(pieces) == 2:
            option, value = pieces
        else:
            raise ValueError("Plot option string '%s' is malformed." % (s))
        options[option.strip()] = value.strip()
    return options


def dot_product(v1, v2):
    return sum(v1[i]*v2[i] for i in range(3))


def vec_sub(v1, v2):
    return tuple(v1[i] - v2[i] for i in range(3))


def vec_mag(v):
    return sum(v[i]**2 for i in range(3))**(0.5)
<|endoftext|>import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
                                            screen_to_model, vec_subs


class PlotCamera:

    min_dist = 0.05
    max_dist = 500.0

    min_ortho_dist = 100.0
    max_ortho_dist = 10000.0

    _default_dist = 6.0
    _default_ortho_dist = 600.0

    rot_presets = {
        'xy': (0, 0, 0),
        'xz': (-90, 0, 0),
        'yz': (0, 90, 0),
        'perspective': (-45, 0, -45)
    }

    def __init__(self, window, ortho=False):
        self.window = window
        self.axes = self.window.plot.axes
        self.ortho = ortho
        self.reset()

    def init_rot_matrix(self):
        pgl.glPushMatrix()
        pgl.glLoadIdentity()
        self._rot = get_model_matrix()
        pgl.glPopMatrix()

    def set_rot_preset(self, preset_name):
        self.init_rot_matrix()
        if preset_name not in self.rot_presets:
            raise ValueError(
                "%s is not a valid rotation preset." % preset_name)
        r = self.rot_presets[preset_name]
        self.euler_rotate(r[0], 1, 0, 0)
        self.euler_rotate(r[1], 0, 1, 0)
        self.euler_rotate(r[2], 0, 0, 1)

    def reset(self):
        self._dist = 0.0
        self._x, self._y = 0.0, 0.0
        self._rot = None
        if self.ortho:
            self._dist = self._default_ortho_dist
        else:
            self._dist = self._default_dist
        self.init_rot_matrix()

    def mult_rot_matrix(self, rot):
        pgl.glPushMatrix()
        pgl.glLoadMatrixf(rot)
        pgl.glMultMatrixf(self._rot)
        self._rot = get_model_matrix()
        pgl.glPopMatrix()

    def setup_projection(self):
        pgl.glMatrixMode(pgl.GL_PROJECTION)
        pgl.glLoadIdentity()
        if self.ortho:
            # yep, this is pseudo ortho (don't tell anyone)
            pgl.gluPerspective(
                0.3, float(self.window.width)/float(self.window.height),
                self.min_ortho_dist - 0.01, self.max_ortho_dist + 0.01)
        else:
            pgl.gluPerspective(
                30.0, float(self.window.width)/float(self.window.height),
                self.min_dist - 0.01, self.max_dist + 0.01)
        pgl.glMatrixMode(pgl.GL_MODELVIEW)

    def _get_scale(self):
        return 1.0, 1.0, 1.0

    def apply_transformation(self):
        pgl.glLoadIdentity()
        pgl.glTranslatef(self._x, self._y, -self._dist)
        if self._rot is not None:
            pgl.glMultMatrixf(self._rot)
        pgl.glScalef(*self._get_scale())

    def spherical_rotate(self, p1, p2, sensitivity=1.0):
        mat = get_spherical_rotatation(p1, p2, self.window.width,
                                       self.window.height, sensitivity)
        if mat is not None:
            self.mult_rot_matrix(mat)

    def euler_rotate(self, angle, x, y, z):
        pgl.glPushMatrix()
        pgl.glLoadMatrixf(self._rot)
        pgl.glRotatef(angle, x, y, z)
        self._rot = get_model_matrix()
        pgl.glPopMatrix()

    def zoom_relative(self, clicks, sensitivity):

        if self.ortho:
            dist_d = clicks * sensitivity * 50.0
            min_dist = self.min_ortho_dist
            max_dist = self.max_ortho_dist
        else:
            dist_d = clicks * sensitivity
            min_dist = self.min_dist
            max_dist = self.max_dist

        new_dist = (self._dist - dist_d)
        if (clicks < 0 and new_dist < max_dist) or new_dist > min_dist:
            self._dist = new_dist

    def mouse_translate(self, x, y, dx, dy):
        pgl.glPushMatrix()
        pgl.glLoadIdentity()
        pgl.glTranslatef(0, 0, -self._dist)
        z = model_to_screen(0, 0, 0)[2]
        d = vec_subs(screen_to_model(x, y, z), screen_to_model(x - dx, y - dy, z))
        pgl.glPopMatrix()
        self._x += d[0]
        self._y += d[1]
<|endoftext|>import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase


class PlotCurve(PlotModeBase):

    style_override = 'wireframe'

    def _on_calculate_verts(self):
        self.t_interval = self.intervals[0]
        self.t_set = list(self.t_interval.frange())
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        evaluate = self._get_evaluator()

        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = float(self.t_interval.v_len)

        self.verts = []
        b = self.bounds
        for t in self.t_set:
            try:
                _e = evaluate(t)    # calculate vertex
            except (NameError, ZeroDivisionError):
                _e = None
            if _e is not None:      # update bounding box
                for axis in range(3):
                    b[axis][0] = min([b[axis][0], _e[axis]])
                    b[axis][1] = max([b[axis][1], _e[axis]])
            self.verts.append(_e)
            self._calculating_verts_pos += 1.0

        for axis in range(3):
            b[axis][2] = b[axis][1] - b[axis][0]
            if b[axis][2] == 0.0:
                b[axis][2] = 1.0

        self.push_wireframe(self.draw_verts(False))

    def _on_calculate_cverts(self):
        if not self.verts or not self.color:
            return

        def set_work_len(n):
            self._calculating_cverts_len = float(n)

        def inc_work_pos():
            self._calculating_cverts_pos += 1.0
        set_work_len(1)
        self._calculating_cverts_pos = 0
        self.cverts = self.color.apply_to_curve(self.verts,
                                                self.t_set,
                                                set_len=set_work_len,
                                                inc_pos=inc_work_pos)
        self.push_wireframe(self.draw_verts(True))

    def calculate_one_cvert(self, t):
        vert = self.verts[t]
        return self.color(vert[0], vert[1], vert[2],
                          self.t_set[t], None)

    def draw_verts(self, use_cverts):
        def f():
            pgl.glBegin(pgl.GL_LINE_STRIP)
            for t in range(len(self.t_set)):
                p = self.verts[t]
                if p is None:
                    pgl.glEnd()
                    pgl.glBegin(pgl.GL_LINE_STRIP)
                    continue
                if use_cverts:
                    c = self.cverts[t]
                    if c is None:
                        c = (0, 0, 0)
                    pgl.glColor3f(*c)
                else:
                    pgl.glColor3f(*self.default_wireframe_color)
                pgl.glVertex3f(*p)
            pgl.glEnd()
        return f
<|endoftext|>from threading import RLock

# it is sufficient to import "pyglet" here once
try:
    import pyglet.gl as pgl
except ImportError:
    raise ImportError("pyglet is required for plotting.\n "
                      "visit https://pyglet.org/")

from sympy.core.numbers import Integer
from sympy.external.gmpy import SYMPY_INTS
from sympy.geometry.entity import GeometryEntity
from sympy.plotting.pygletplot.plot_axes import PlotAxes
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.plot_window import PlotWindow
from sympy.plotting.pygletplot.util import parse_option_string
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import is_sequence

from time import sleep
from os import getcwd, listdir

import ctypes

@doctest_depends_on(modules=('pyglet',))
class PygletPlot:
    """
    Plot Examples
    =============

    See examples/advanced/pyglet_plotting.py for many more examples.

    >>> from sympy.plotting.pygletplot import PygletPlot as Plot
    >>> from sympy.abc import x, y, z

    >>> Plot(x*y**3-y*x**3)
    [0]: -x**3*y + x*y**3, 'mode=cartesian'

    >>> p = Plot()
    >>> p[1] = x*y
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)

    >>> p = Plot()
    >>> p[1] =  x**2+y**2
    >>> p[2] = -x**2-y**2


    Variable Intervals
    ==================

    The basic format is [var, min, max, steps], but the
    syntax is flexible and arguments left out are taken
    from the defaults for the current coordinate mode:

    >>> Plot(x**2) # implies [x,-5,5,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [], []) # [x,-1,1,40], [y,-1,1,40]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2-y**2, [100], [100]) # [x,-1,1,100], [y,-1,1,100]
    [0]: x**2 - y**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13,100])
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [-13,13]) # [x,-13,13,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13]) # [x,-13,13,10]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(1*x, [], [x], mode='cylindrical')
    ... # [unbound_theta,0,2*Pi,40], [x,-1,1,20]
    [0]: x, 'mode=cartesian'


    Coordinate Modes
    ================

    Plot supports several curvilinear coordinate modes, and
    they independent for each plotted function. You can specify
    a coordinate mode explicitly with the 'mode' named argument,
    but it can be automatically determined for Cartesian or
    parametric plots, and therefore must only be specified for
    polar, cylindrical, and spherical modes.

    Specifically, Plot(function arguments) and Plot[n] =
    (function arguments) will interpret your arguments as a
    Cartesian plot if you provide one function and a parametric
    plot if you provide two or three functions. Similarly, the
    arguments will be interpreted as a curve if one variable is
    used, and a surface if two are used.

    Supported mode names by number of variables:

    1: parametric, cartesian, polar
    2: parametric, cartesian, cylindrical = polar, spherical

    >>> Plot(1, mode='spherical')


    Calculator-like Interface
    =========================

    >>> p = Plot(visible=False)
    >>> f = x**2
    >>> p[1] = f
    >>> p[2] = f.diff(x)
    >>> p[3] = f.diff(x).diff(x)
    >>> p
    [1]: x**2, 'mode=cartesian'
    [2]: 2*x, 'mode=cartesian'
    [3]: 2, 'mode=cartesian'
    >>> p.show()
    >>> p.clear()
    >>> p
    <blank plot>
    >>> p[1] =  x**2+y**2
    >>> p[1].style = 'solid'
    >>> p[2] = -x**2-y**2
    >>> p[2].style = 'wireframe'
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)
    >>> p[1].style = 'both'
    >>> p[2].style = 'both'
    >>> p.close()


    Plot Window Keyboard Controls
    =============================

    Screen Rotation:
        X,Y axis      Arrow Keys, A,S,D,W, Numpad 4,6,8,2
        Z axis        Q,E, Numpad 7,9

    Model Rotation:
        Z axis        Z,C, Numpad 1,3

    Zoom:             R,F, PgUp,PgDn, Numpad +,-

    Reset Camera:     X, Numpad 5

    Camera Presets:
        XY            F1
        XZ            F2
        YZ            F3
        Perspective   F4

    Sensitivity Modifier: SHIFT

    Axes Toggle:
        Visible       F5
        Colors        F6

    Close Window:     ESCAPE

    =============================

    """

    @doctest_depends_on(modules=('pyglet',))
    def __init__(self, *fargs, **win_args):
        """
        Positional Arguments
        ====================

        Any given positional arguments are used to
        initialize a plot function at index 1. In
        other words...

        >>> from sympy.plotting.pygletplot import PygletPlot as Plot
        >>> from sympy.abc import x
        >>> p = Plot(x**2, visible=False)

        ...is equivalent to...

        >>> p = Plot(visible=False)
        >>> p[1] = x**2

        Note that in earlier versions of the plotting
        module, you were able to specify multiple
        functions in the initializer. This functionality
        has been dropped in favor of better automatic
        plot plot_mode detection.


        Named Arguments
        ===============

        axes
            An option string of the form
            "key1=value1; key2 = value2" which
            can use the following options:

            style = ordinate
                none OR frame OR box OR ordinate

            stride = 0.25
                val OR (val_x, val_y, val_z)

            overlay = True (draw on top of plot)
                True OR False

            colored = False (False uses Black,
                             True uses colors
                             R,G,B = X,Y,Z)
                True OR False

            label_axes = False (display axis names
                                at endpoints)
                True OR False

        visible = True (show immediately
            True OR False


        The following named arguments are passed as
        arguments to window initialization:

        antialiasing = True
            True OR False

        ortho = False
            True OR False

        invert_mouse_zoom = False
            True OR False

        """
        # Register the plot modes
        from . import plot_modes # noqa

        self._win_args = win_args
        self._window = None

        self._render_lock = RLock()

        self._functions = {}
        self._pobjects = []
        self._screenshot = ScreenShot(self)

        axe_options = parse_option_string(win_args.pop('axes', ''))
        self.axes = PlotAxes(**axe_options)
        self._pobjects.append(self.axes)

        self[0] = fargs
        if win_args.get('visible', True):
            self.show()

    ## Window Interfaces

    def show(self):
        """
        Creates and displays a plot window, or activates it
        (gives it focus) if it has already been created.
        """
        if self._window and not self._window.has_exit:
            self._window.activate()
        else:
            self._win_args['visible'] = True
            self.axes.reset_resources()

            #if hasattr(self, '_doctest_depends_on'):
            #    self._win_args['runfromdoctester'] = True

            self._window = PlotWindow(self, **self._win_args)

    def close(self):
        """
        Closes the plot window.
        """
        if self._window:
            self._window.close()

    def saveimage(self, outfile=None, format='', size=(600, 500)):
        """
        Saves a screen capture of the plot window to an
        image file.

        If outfile is given, it can either be a path
        or a file object. Otherwise a png image will
        be saved to the current working directory.
        If the format is omitted, it is determined from
        the filename extension.
        """
        self._screenshot.save(outfile, format, size)

    ## Function List Interfaces

    def clear(self):
        """
        Clears the function list of this plot.
        """
        self._render_lock.acquire()
        self._functions = {}
        self.adjust_all_bounds()
        self._render_lock.release()

    def __getitem__(self, i):
        """
        Returns the function at position i in the
        function list.
        """
        return self._functions[i]

    def __setitem__(self, i, args):
        """
        Parses and adds a PlotMode to the function
        list.
        """
        if not (isinstance(i, (SYMPY_INTS, Integer)) and i >= 0):
            raise ValueError("Function index must "
                             "be an integer >= 0.")

        if isinstance(args, PlotObject):
            f = args
        else:
            if (not is_sequence(args)) or isinstance(args, GeometryEntity):
                args = [args]
            if len(args) == 0:
                return  # no arguments given
            kwargs = {"bounds_callback": self.adjust_all_bounds}
            f = PlotMode(*args, **kwargs)

        if f:
            self._render_lock.acquire()
            self._functions[i] = f
            self._render_lock.release()
        else:
            raise ValueError("Failed to parse '%s'."
                    % ', '.join(str(a) for a in args))

    def __delitem__(self, i):
        """
        Removes the function in the function list at
        position i.
        """
        self._render_lock.acquire()
        del self._functions[i]
        self.adjust_all_bounds()
        self._render_lock.release()

    def firstavailableindex(self):
        """
        Returns the first unused index in the function list.
        """
        i = 0
        self._render_lock.acquire()
        while i in self._functions:
            i += 1
        self._render_lock.release()
        return i

    def append(self, *args):
        """
        Parses and adds a PlotMode to the function
        list at the first available index.
        """
        self.__setitem__(self.firstavailableindex(), args)

    def __len__(self):
        """
        Returns the number of functions in the function list.
        """
        return len(self._functions)

    def __iter__(self):
        """
        Allows iteration of the function list.
        """
        return self._functions.itervalues()

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        Returns a string containing a new-line separated
        list of the functions in the function list.
        """
        s = ""
        if len(self._functions) == 0:
            s += "<blank plot>"
        else:
            self._render_lock.acquire()
            s += "\n".join(["%s[%i]: %s" % ("", i, str(self._functions[i]))
                            for i in self._functions])
            self._render_lock.release()
        return s

    def adjust_all_bounds(self):
        self._render_lock.acquire()
        self.axes.reset_bounding_box()
        for f in self._functions:
            self.axes.adjust_bounds(self._functions[f].bounds)
        self._render_lock.release()

    def wait_for_calculations(self):
        sleep(0)
        self._render_lock.acquire()
        for f in self._functions:
            a = self._functions[f]._get_calculating_verts
            b = self._functions[f]._get_calculating_cverts
            while a() or b():
                sleep(0)
        self._render_lock.release()

class ScreenShot:
    def __init__(self, plot):
        self._plot = plot
        self.screenshot_requested = False
        self.outfile = None
        self.format = ''
        self.invisibleMode = False
        self.flag = 0

    def __bool__(self):
        return self.screenshot_requested

    def _execute_saving(self):
        if self.flag < 3:
            self.flag += 1
            return

        size_x, size_y = self._plot._window.get_size()
        size = size_x*size_y*4*ctypes.sizeof(ctypes.c_ubyte)
        image = ctypes.create_string_buffer(size)
        pgl.glReadPixels(0, 0, size_x, size_y, pgl.GL_RGBA, pgl.GL_UNSIGNED_BYTE, image)
        from PIL import Image
        im = Image.frombuffer('RGBA', (size_x, size_y),
                              image.raw, 'raw', 'RGBA', 0, 1)
        im.transpose(Image.FLIP_TOP_BOTTOM).save(self.outfile, self.format)

        self.flag = 0
        self.screenshot_requested = False
        if self.invisibleMode:
            self._plot._window.close()

    def save(self, outfile=None, format='', size=(600, 500)):
        self.outfile = outfile
        self.format = format
        self.size = size
        self.screenshot_requested = True

        if not self._plot._window or self._plot._window.has_exit:
            self._plot._win_args['visible'] = False

            self._plot._win_args['width'] = size[0]
            self._plot._win_args['height'] = size[1]

            self._plot.axes.reset_resources()
            self._plot._window = PlotWindow(self._plot, **self._plot._win_args)
            self.invisibleMode = True

        if self.outfile is None:
            self.outfile = self._create_unique_path()
            print(self.outfile)

    def _create_unique_path(self):
        cwd = getcwd()
        l = listdir(cwd)
        path = ''
        i = 0
        while True:
            if not 'plot_%s.png' % i in l:
                path = cwd + '/plot_%s.png' % i
                break
            i += 1
        return path
<|endoftext|>import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings


class PlotModeBase(PlotMode):
    """
    Intended parent class for plotting
    modes. Provides base functionality
    in conjunction with its parent,
    PlotMode.
    """

    ##
    ## Class-Level Attributes
    ##

    """
    The following attributes are meant
    to be set at the class level, and serve
    as parameters to the plot mode registry
    (in PlotMode). See plot_modes.py for
    concrete examples.
    """

    """
    i_vars
        'x' for Cartesian2D
        'xy' for Cartesian3D
        etc.

    d_vars
        'y' for Cartesian2D
        'r' for Polar
        etc.
    """
    i_vars, d_vars = '', ''

    """
    intervals
        Default intervals for each i_var, and in the
        same order. Specified [min, max, steps].
        No variable can be given (it is bound later).
    """
    intervals = []

    """
    aliases
        A list of strings which can be used to
        access this mode.
        'cartesian' for Cartesian2D and Cartesian3D
        'polar' for Polar
        'cylindrical', 'polar' for Cylindrical

        Note that _init_mode chooses the first alias
        in the list as the mode's primary_alias, which
        will be displayed to the end user in certain
        contexts.
    """
    aliases = []

    """
    is_default
        Whether to set this mode as the default
        for arguments passed to PlotMode() containing
        the same number of d_vars as this mode and
        at most the same number of i_vars.
    """
    is_default = False

    """
    All of the above attributes are defined in PlotMode.
    The following ones are specific to PlotModeBase.
    """

    """
    A list of the render styles. Do not modify.
    """
    styles = {'wireframe': 1, 'solid': 2, 'both': 3}

    """
    style_override
        Always use this style if not blank.
    """
    style_override = ''

    """
    default_wireframe_color
    default_solid_color
        Can be used when color is None or being calculated.
        Used by PlotCurve and PlotSurface, but not anywhere
        in PlotModeBase.
    """

    default_wireframe_color = (0.85, 0.85, 0.85)
    default_solid_color = (0.6, 0.6, 0.9)
    default_rot_preset = 'xy'

    ##
    ## Instance-Level Attributes
    ##

    ## 'Abstract' member functions
    def _get_evaluator(self):
        if self.use_lambda_eval:
            try:
                e = self._get_lambda_evaluator()
                return e
            except Exception:
                warnings.warn("\nWarning: creating lambda evaluator failed. "
                       "Falling back on SymPy subs evaluator.")
        return self._get_sympy_evaluator()

    def _get_sympy_evaluator(self):
        raise NotImplementedError()

    def _get_lambda_evaluator(self):
        raise NotImplementedError()

    def _on_calculate_verts(self):
        raise NotImplementedError()

    def _on_calculate_cverts(self):
        raise NotImplementedError()

    ## Base member functions
    def __init__(self, *args, bounds_callback=None, **kwargs):
        self.verts = []
        self.cverts = []
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        self.cbounds = [[S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0]]

        self._draw_lock = RLock()

        self._calculating_verts = Event()
        self._calculating_cverts = Event()
        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = 0.0
        self._calculating_cverts_pos = 0.0
        self._calculating_cverts_len = 0.0

        self._max_render_stack_size = 3
        self._draw_wireframe = [-1]
        self._draw_solid = [-1]

        self._style = None
        self._color = None

        self.predraw = []
        self.postdraw = []

        self.use_lambda_eval = self.options.pop('use_sympy_eval', None) is None
        self.style = self.options.pop('style', '')
        self.color = self.options.pop('color', 'rainbow')
        self.bounds_callback = bounds_callback

        self._on_calculate()

    def synchronized(f):
        def w(self, *args, **kwargs):
            self._draw_lock.acquire()
            try:
                r = f(self, *args, **kwargs)
                return r
            finally:
                self._draw_lock.release()
        return w

    @synchronized
    def push_wireframe(self, function):
        """
        Push a function which performs gl commands
        used to build a display list. (The list is
        built outside of the function)
        """
        assert callable(function)
        self._draw_wireframe.append(function)
        if len(self._draw_wireframe) > self._max_render_stack_size:
            del self._draw_wireframe[1]  # leave marker element

    @synchronized
    def push_solid(self, function):
        """
        Push a function which performs gl commands
        used to build a display list. (The list is
        built outside of the function)
        """
        assert callable(function)
        self._draw_solid.append(function)
        if len(self._draw_solid) > self._max_render_stack_size:
            del self._draw_solid[1]  # leave marker element

    def _create_display_list(self, function):
        dl = pgl.glGenLists(1)
        pgl.glNewList(dl, pgl.GL_COMPILE)
        function()
        pgl.glEndList()
        return dl

    def _render_stack_top(self, render_stack):
        top = render_stack[-1]
        if top == -1:
            return -1  # nothing to display
        elif callable(top):
            dl = self._create_display_list(top)
            render_stack[-1] = (dl, top)
            return dl  # display newly added list
        elif len(top) == 2:
            if pgl.GL_TRUE == pgl.glIsList(top[0]):
                return top[0]  # display stored list
            dl = self._create_display_list(top[1])
            render_stack[-1] = (dl, top[1])
            return dl  # display regenerated list

    def _draw_solid_display_list(self, dl):
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_FILL)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    def _draw_wireframe_display_list(self, dl):
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_LINE)
        pgl.glEnable(pgl.GL_POLYGON_OFFSET_LINE)
        pgl.glPolygonOffset(-0.005, -50.0)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    @synchronized
    def draw(self):
        for f in self.predraw:
            if callable(f):
                f()
        if self.style_override:
            style = self.styles[self.style_override]
        else:
            style = self.styles[self._style]
        # Draw solid component if style includes solid
        if style & 2:
            dl = self._render_stack_top(self._draw_solid)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_solid_display_list(dl)
        # Draw wireframe component if style includes wireframe
        if style & 1:
            dl = self._render_stack_top(self._draw_wireframe)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_wireframe_display_list(dl)
        for f in self.postdraw:
            if callable(f):
                f()

    def _on_change_color(self, color):
        Thread(target=self._calculate_cverts).start()

    def _on_calculate(self):
        Thread(target=self._calculate_all).start()

    def _calculate_all(self):
        self._calculate_verts()
        self._calculate_cverts()

    def _calculate_verts(self):
        if self._calculating_verts.is_set():
            return
        self._calculating_verts.set()
        try:
            self._on_calculate_verts()
        finally:
            self._calculating_verts.clear()
        if callable(self.bounds_callback):
            self.bounds_callback()

    def _calculate_cverts(self):
        if self._calculating_verts.is_set():
            return
        while self._calculating_cverts.is_set():
            sleep(0)  # wait for previous calculation
        self._calculating_cverts.set()
        try:
            self._on_calculate_cverts()
        finally:
            self._calculating_cverts.clear()

    def _get_calculating_verts(self):
        return self._calculating_verts.is_set()

    def _get_calculating_verts_pos(self):
        return self._calculating_verts_pos

    def _get_calculating_verts_len(self):
        return self._calculating_verts_len

    def _get_calculating_cverts(self):
        return self._calculating_cverts.is_set()

    def _get_calculating_cverts_pos(self):
        return self._calculating_cverts_pos

    def _get_calculating_cverts_len(self):
        return self._calculating_cverts_len

    ## Property handlers
    def _get_style(self):
        return self._style

    @synchronized
    def _set_style(self, v):
        if v is None:
            return
        if v == '':
            step_max = 0
            for i in self.intervals:
                if i.v_steps is None:
                    continue
                step_max = max([step_max, int(i.v_steps)])
            v = ['both', 'solid'][step_max > 40]
        if v not in self.styles:
            raise ValueError("v should be there in self.styles")
        if v == self._style:
            return
        self._style = v

    def _get_color(self):
        return self._color

    @synchronized
    def _set_color(self, v):
        try:
            if v is not None:
                if is_sequence(v):
                    v = ColorScheme(*v)
                else:
                    v = ColorScheme(v)
            if repr(v) == repr(self._color):
                return
            self._on_change_color(v)
            self._color = v
        except Exception as e:
            raise RuntimeError("Color change failed. "
                                "Reason: %s" % (str(e)))

    style = property(_get_style, _set_style)
    color = property(_get_color, _set_color)

    calculating_verts = property(_get_calculating_verts)
    calculating_verts_pos = property(_get_calculating_verts_pos)
    calculating_verts_len = property(_get_calculating_verts_len)

    calculating_cverts = property(_get_calculating_cverts)
    calculating_cverts_pos = property(_get_calculating_cverts_pos)
    calculating_cverts_len = property(_get_calculating_cverts_len)

    ## String representations

    def __str__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        o = "'mode=%s'" % (self.primary_alias)
        return ", ".join([f, o])

    def __repr__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        i = ", ".join(str(i) for i in self.intervals)
        d = [('mode', self.primary_alias),
             ('color', str(self.color)),
             ('style', str(self.style))]

        o = "'%s'" % ("; ".join("%s=%s" % (k, v)
                                for k, v in d if v != 'None'))
        return ", ".join([f, i, o])
<|endoftext|>"""Plotting module that can plot 2D and 3D functions
"""

from sympy.utilities.decorator import doctest_depends_on

@doctest_depends_on(modules=('pyglet',))
def PygletPlot(*args, **kwargs):
    """

    Plot Examples
    =============

    See examples/advanced/pyglet_plotting.py for many more examples.

    >>> from sympy.plotting.pygletplot import PygletPlot as Plot
    >>> from sympy.abc import x, y, z

    >>> Plot(x*y**3-y*x**3)
    [0]: -x**3*y + x*y**3, 'mode=cartesian'

    >>> p = Plot()
    >>> p[1] = x*y
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)

    >>> p = Plot()
    >>> p[1] =  x**2+y**2
    >>> p[2] = -x**2-y**2


    Variable Intervals
    ==================

    The basic format is [var, min, max, steps], but the
    syntax is flexible and arguments left out are taken
    from the defaults for the current coordinate mode:

    >>> Plot(x**2) # implies [x,-5,5,100]
    [0]: x**2, 'mode=cartesian'

    >>> Plot(x**2, [], []) # [x,-1,1,40], [y,-1,1,40]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2-y**2, [100], [100]) # [x,-1,1,100], [y,-1,1,100]
    [0]: x**2 - y**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13,100])
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [-13,13]) # [x,-13,13,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13]) # [x,-13,13,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(1*x, [], [x], mode='cylindrical')
    ... # [unbound_theta,0,2*Pi,40], [x,-1,1,20]
    [0]: x, 'mode=cartesian'


    Coordinate Modes
    ================

    Plot supports several curvilinear coordinate modes, and
    they independent for each plotted function. You can specify
    a coordinate mode explicitly with the 'mode' named argument,
    but it can be automatically determined for Cartesian or
    parametric plots, and therefore must only be specified for
    polar, cylindrical, and spherical modes.

    Specifically, Plot(function arguments) and Plot[n] =
    (function arguments) will interpret your arguments as a
    Cartesian plot if you provide one function and a parametric
    plot if you provide two or three functions. Similarly, the
    arguments will be interpreted as a curve if one variable is
    used, and a surface if two are used.

    Supported mode names by number of variables:

    1: parametric, cartesian, polar
    2: parametric, cartesian, cylindrical = polar, spherical

    >>> Plot(1, mode='spherical')


    Calculator-like Interface
    =========================

    >>> p = Plot(visible=False)
    >>> f = x**2
    >>> p[1] = f
    >>> p[2] = f.diff(x)
    >>> p[3] = f.diff(x).diff(x)
    >>> p
    [1]: x**2, 'mode=cartesian'
    [2]: 2*x, 'mode=cartesian'
    [3]: 2, 'mode=cartesian'
    >>> p.show()
    >>> p.clear()
    >>> p
    <blank plot>
    >>> p[1] =  x**2+y**2
    >>> p[1].style = 'solid'
    >>> p[2] = -x**2-y**2
    >>> p[2].style = 'wireframe'
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)
    >>> p[1].style = 'both'
    >>> p[2].style = 'both'
    >>> p.close()


    Plot Window Keyboard Controls
    =============================

    Screen Rotation:
        X,Y axis      Arrow Keys, A,S,D,W, Numpad 4,6,8,2
        Z axis        Q,E, Numpad 7,9

    Model Rotation:
        Z axis        Z,C, Numpad 1,3

    Zoom:             R,F, PgUp,PgDn, Numpad +,-

    Reset Camera:     X, Numpad 5

    Camera Presets:
        XY            F1
        XZ            F2
        YZ            F3
        Perspective   F4

    Sensitivity Modifier: SHIFT

    Axes Toggle:
        Visible       F5
        Colors        F6

    Close Window:     ESCAPE

    =============================
    """

    from sympy.plotting.pygletplot.plot import PygletPlot
    return PygletPlot(*args, **kwargs)
<|endoftext|>try:
    from ctypes import c_float
except ImportError:
    pass

import pyglet.gl as pgl
from math import sqrt as _sqrt, acos as _acos, pi


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def mag(a):
    return _sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def norm(a):
    m = mag(a)
    return (a[0] / m, a[1] / m, a[2] / m)


def get_sphere_mapping(x, y, width, height):
    x = min([max([x, 0]), width])
    y = min([max([y, 0]), height])

    sr = _sqrt((width/2)**2 + (height/2)**2)
    sx = ((x - width / 2) / sr)
    sy = ((y - height / 2) / sr)

    sz = 1.0 - sx**2 - sy**2

    if sz > 0.0:
        sz = _sqrt(sz)
        return (sx, sy, sz)
    else:
        sz = 0
        return norm((sx, sy, sz))

rad2deg = 180.0 / pi


def get_spherical_rotatation(p1, p2, width, height, theta_multiplier):
    v1 = get_sphere_mapping(p1[0], p1[1], width, height)
    v2 = get_sphere_mapping(p2[0], p2[1], width, height)

    d = min(max([dot(v1, v2), -1]), 1)

    if abs(d - 1.0) < 0.000001:
        return None

    raxis = norm( cross(v1, v2) )
    rtheta = theta_multiplier * rad2deg * _acos(d)

    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    pgl.glRotatef(rtheta, *raxis)
    mat = (c_float*16)()
    pgl.glGetFloatv(pgl.GL_MODELVIEW_MATRIX, mat)
    pgl.glPopMatrix()

    return mat
<|endoftext|>import pyglet.gl as pgl
from pyglet import font

from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
        get_direction_vectors, strided_range, vec_mag, vec_sub
from sympy.utilities.iterables import is_sequence


class PlotAxes(PlotObject):

    def __init__(self, *args,
            style='', none=None, frame=None, box=None, ordinate=None,
            stride=0.25,
            visible='', overlay='', colored='', label_axes='', label_ticks='',
            tick_length=0.1,
            font_face='Arial', font_size=28,
            **kwargs):
        # initialize style parameter
        style = style.lower()

        # allow alias kwargs to override style kwarg
        if none is not None:
            style = 'none'
        if frame is not None:
            style = 'frame'
        if box is not None:
            style = 'box'
        if ordinate is not None:
            style = 'ordinate'

        if style in ['', 'ordinate']:
            self._render_object = PlotAxesOrdinate(self)
        elif style in ['frame', 'box']:
            self._render_object = PlotAxesFrame(self)
        elif style in ['none']:
            self._render_object = None
        else:
            raise ValueError(("Unrecognized axes style %s.") % (style))

        # initialize stride parameter
        try:
            stride = eval(stride)
        except TypeError:
            pass
        if is_sequence(stride):
            if len(stride) != 3:
                raise ValueError("length should be equal to 3")
            self._stride = stride
        else:
            self._stride = [stride, stride, stride]
        self._tick_length = float(tick_length)

        # setup bounding box and ticks
        self._origin = [0, 0, 0]
        self.reset_bounding_box()

        def flexible_boolean(input, default):
            if input in [True, False]:
                return input
            if input in ('f', 'F', 'false', 'False'):
                return False
            if input in ('t', 'T', 'true', 'True'):
                return True
            return default

        # initialize remaining parameters
        self.visible = flexible_boolean(kwargs, True)
        self._overlay = flexible_boolean(overlay, True)
        self._colored = flexible_boolean(colored, False)
        self._label_axes = flexible_boolean(label_axes, False)
        self._label_ticks = flexible_boolean(label_ticks, True)

        # setup label font
        self.font_face = font_face
        self.font_size = font_size

        # this is also used to reinit the
        # font on window close/reopen
        self.reset_resources()

    def reset_resources(self):
        self.label_font = None

    def reset_bounding_box(self):
        self._bounding_box = [[None, None], [None, None], [None, None]]
        self._axis_ticks = [[], [], []]

    def draw(self):
        if self._render_object:
            pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT | pgl.GL_DEPTH_BUFFER_BIT)
            if self._overlay:
                pgl.glDisable(pgl.GL_DEPTH_TEST)
            self._render_object.draw()
            pgl.glPopAttrib()

    def adjust_bounds(self, child_bounds):
        b = self._bounding_box
        c = child_bounds
        for i in range(3):
            if abs(c[i][0]) is S.Infinity or abs(c[i][1]) is S.Infinity:
                continue
            b[i][0] = c[i][0] if b[i][0] is None else min([b[i][0], c[i][0]])
            b[i][1] = c[i][1] if b[i][1] is None else max([b[i][1], c[i][1]])
            self._bounding_box = b
            self._recalculate_axis_ticks(i)

    def _recalculate_axis_ticks(self, axis):
        b = self._bounding_box
        if b[axis][0] is None or b[axis][1] is None:
            self._axis_ticks[axis] = []
        else:
            self._axis_ticks[axis] = strided_range(b[axis][0], b[axis][1],
                                                   self._stride[axis])

    def toggle_visible(self):
        self.visible = not self.visible

    def toggle_colors(self):
        self._colored = not self._colored


class PlotAxesBase(PlotObject):

    def __init__(self, parent_axes):
        self._p = parent_axes

    def draw(self):
        color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]),
                 ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])][self._p._colored]
        self.draw_background(color)
        self.draw_axis(2, color[2])
        self.draw_axis(1, color[1])
        self.draw_axis(0, color[0])

    def draw_background(self, color):
        pass  # optional

    def draw_axis(self, axis, color):
        raise NotImplementedError()

    def draw_text(self, text, position, color, scale=1.0):
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)

        if self._p.label_font is None:
            self._p.label_font = font.load(self._p.font_face,
                                           self._p.font_size,
                                           bold=True, italic=False)

        label = font.Text(self._p.label_font, text,
                          color=color,
                          valign=font.Text.BASELINE,
                          halign=font.Text.CENTER)

        pgl.glPushMatrix()
        pgl.glTranslatef(*position)
        billboard_matrix()
        scale_factor = 0.005 * scale
        pgl.glScalef(scale_factor, scale_factor, scale_factor)
        pgl.glColor4f(0, 0, 0, 0)
        label.draw()
        pgl.glPopMatrix()

    def draw_line(self, v, color):
        o = self._p._origin
        pgl.glBegin(pgl.GL_LINES)
        pgl.glColor3f(*color)
        pgl.glVertex3f(v[0][0] + o[0], v[0][1] + o[1], v[0][2] + o[2])
        pgl.glVertex3f(v[1][0] + o[0], v[1][1] + o[1], v[1][2] + o[2])
        pgl.glEnd()


class PlotAxesOrdinate(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_axis(self, axis, color):
        ticks = self._p._axis_ticks[axis]
        radius = self._p._tick_length / 2.0
        if len(ticks) < 2:
            return

        # calculate the vector for this axis
        axis_lines = [[0, 0, 0], [0, 0, 0]]
        axis_lines[0][axis], axis_lines[1][axis] = ticks[0], ticks[-1]
        axis_vector = vec_sub(axis_lines[1], axis_lines[0])

        # calculate angle to the z direction vector
        pos_z = get_direction_vectors()[2]
        d = abs(dot_product(axis_vector, pos_z))
        d = d / vec_mag(axis_vector)

        # don't draw labels if we're looking down the axis
        labels_visible = abs(d - 1.0) > 0.02

        # draw the ticks and labels
        for tick in ticks:
            self.draw_tick_line(axis, color, radius, tick, labels_visible)

        # draw the axis line and labels
        self.draw_axis_line(axis, color, ticks[0], ticks[-1], labels_visible)

    def draw_axis_line(self, axis, color, a_min, a_max, labels_visible):
        axis_line = [[0, 0, 0], [0, 0, 0]]
        axis_line[0][axis], axis_line[1][axis] = a_min, a_max
        self.draw_line(axis_line, color)
        if labels_visible:
            self.draw_axis_line_labels(axis, color, axis_line)

    def draw_axis_line_labels(self, axis, color, axis_line):
        if not self._p._label_axes:
            return
        axis_labels = [axis_line[0][::], axis_line[1][::]]
        axis_labels[0][axis] -= 0.3
        axis_labels[1][axis] += 0.3
        a_str = ['X', 'Y', 'Z'][axis]
        self.draw_text("-" + a_str, axis_labels[0], color)
        self.draw_text("+" + a_str, axis_labels[1], color)

    def draw_tick_line(self, axis, color, radius, tick, labels_visible):
        tick_axis = {0: 1, 1: 0, 2: 1}[axis]
        tick_line = [[0, 0, 0], [0, 0, 0]]
        tick_line[0][axis] = tick_line[1][axis] = tick
        tick_line[0][tick_axis], tick_line[1][tick_axis] = -radius, radius
        self.draw_line(tick_line, color)
        if labels_visible:
            self.draw_tick_line_label(axis, color, radius, tick)

    def draw_tick_line_label(self, axis, color, radius, tick):
        if not self._p._label_axes:
            return
        tick_label_vector = [0, 0, 0]
        tick_label_vector[axis] = tick
        tick_label_vector[{0: 1, 1: 0, 2: 1}[axis]] = [-1, 1, 1][
            axis] * radius * 3.5
        self.draw_text(str(tick), tick_label_vector, color, scale=0.5)


class PlotAxesFrame(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_background(self, color):
        pass

    def draw_axis(self, axis, color):
        raise NotImplementedError()
<|endoftext|>from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer


class PlotInterval:
    """
    """
    _v, _v_min, _v_max, _v_steps = None, None, None, None

    def require_all_args(f):
        def check(self, *args, **kwargs):
            for g in [self._v, self._v_min, self._v_max, self._v_steps]:
                if g is None:
                    raise ValueError("PlotInterval is incomplete.")
            return f(self, *args, **kwargs)
        return check

    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], PlotInterval):
                self.fill_from(args[0])
                return
            elif isinstance(args[0], str):
                try:
                    args = eval(args[0])
                except TypeError:
                    s_eval_error = "Could not interpret string %s."
                    raise ValueError(s_eval_error % (args[0]))
            elif isinstance(args[0], (tuple, list)):
                args = args[0]
            else:
                raise ValueError("Not an interval.")
        if not isinstance(args, (tuple, list)) or len(args) > 4:
            f_error = "PlotInterval must be a tuple or list of length 4 or less."
            raise ValueError(f_error)

        args = list(args)
        if len(args) > 0 and (args[0] is None or isinstance(args[0], Symbol)):
            self.v = args.pop(0)
        if len(args) in [2, 3]:
            self.v_min = args.pop(0)
            self.v_max = args.pop(0)
            if len(args) == 1:
                self.v_steps = args.pop(0)
        elif len(args) == 1:
            self.v_steps = args.pop(0)

    def get_v(self):
        return self._v

    def set_v(self, v):
        if v is None:
            self._v = None
            return
        if not isinstance(v, Symbol):
            raise ValueError("v must be a SymPy Symbol.")
        self._v = v

    def get_v_min(self):
        return self._v_min

    def set_v_min(self, v_min):
        if v_min is None:
            self._v_min = None
            return
        try:
            self._v_min = sympify(v_min)
            float(self._v_min.evalf())
        except TypeError:
            raise ValueError("v_min could not be interpreted as a number.")

    def get_v_max(self):
        return self._v_max

    def set_v_max(self, v_max):
        if v_max is None:
            self._v_max = None
            return
        try:
            self._v_max = sympify(v_max)
            float(self._v_max.evalf())
        except TypeError:
            raise ValueError("v_max could not be interpreted as a number.")

    def get_v_steps(self):
        return self._v_steps

    def set_v_steps(self, v_steps):
        if v_steps is None:
            self._v_steps = None
            return
        if isinstance(v_steps, int):
            v_steps = Integer(v_steps)
        elif not isinstance(v_steps, Integer):
            raise ValueError("v_steps must be an int or SymPy Integer.")
        if v_steps <= S.Zero:
            raise ValueError("v_steps must be positive.")
        self._v_steps = v_steps

    @require_all_args
    def get_v_len(self):
        return self.v_steps + 1

    v = property(get_v, set_v)
    v_min = property(get_v_min, set_v_min)
    v_max = property(get_v_max, set_v_max)
    v_steps = property(get_v_steps, set_v_steps)
    v_len = property(get_v_len)

    def fill_from(self, b):
        if b.v is not None:
            self.v = b.v
        if b.v_min is not None:
            self.v_min = b.v_min
        if b.v_max is not None:
            self.v_max = b.v_max
        if b.v_steps is not None:
            self.v_steps = b.v_steps

    @staticmethod
    def try_parse(*args):
        """
        Returns a PlotInterval if args can be interpreted
        as such, otherwise None.
        """
        if len(args) == 1 and isinstance(args[0], PlotInterval):
            return args[0]
        try:
            return PlotInterval(*args)
        except ValueError:
            return None

    def _str_base(self):
        return ",".join([str(self.v), str(self.v_min),
                         str(self.v_max), str(self.v_steps)])

    def __repr__(self):
        """
        A string representing the interval in class constructor form.
        """
        return "PlotInterval(%s)" % (self._str_base())

    def __str__(self):
        """
        A string representing the interval in list form.
        """
        return "[%s]" % (self._str_base())

    @require_all_args
    def assert_complete(self):
        pass

    @require_all_args
    def vrange(self):
        """
        Yields v_steps+1 SymPy numbers ranging from
        v_min to v_max.
        """
        d = (self.v_max - self.v_min) / self.v_steps
        for i in range(self.v_steps + 1):
            a = self.v_min + (d * Integer(i))
            yield a

    @require_all_args
    def vrange2(self):
        """
        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
        d = (self.v_max - self.v_min) / self.v_steps
        a = self.v_min + (d * S.Zero)
        for i in range(self.v_steps):
            b = self.v_min + (d * Integer(i + 1))
            yield a, b
            a = b

    def frange(self):
        for i in self.vrange():
            yield float(i.evalf())
<|endoftext|>from time import perf_counter


import pyglet.gl as pgl

from sympy.plotting.pygletplot.managed_window import ManagedWindow
from sympy.plotting.pygletplot.plot_camera import PlotCamera
from sympy.plotting.pygletplot.plot_controller import PlotController


class PlotWindow(ManagedWindow):

    def __init__(self, plot, antialiasing=True, ortho=False,
                 invert_mouse_zoom=False, linewidth=1.5, caption="SymPy Plot",
                 **kwargs):
        """
        Named Arguments
        ===============

        antialiasing = True
            True OR False
        ortho = False
            True OR False
        invert_mouse_zoom = False
            True OR False
        """
        self.plot = plot

        self.camera = None
        self._calculating = False

        self.antialiasing = antialiasing
        self.ortho = ortho
        self.invert_mouse_zoom = invert_mouse_zoom
        self.linewidth = linewidth
        self.title = caption
        self.last_caption_update = 0
        self.caption_update_interval = 0.2
        self.drawing_first_object = True

        super().__init__(**kwargs)

    def setup(self):
        self.camera = PlotCamera(self, ortho=self.ortho)
        self.controller = PlotController(self,
                invert_mouse_zoom=self.invert_mouse_zoom)
        self.push_handlers(self.controller)

        pgl.glClearColor(1.0, 1.0, 1.0, 0.0)
        pgl.glClearDepth(1.0)

        pgl.glDepthFunc(pgl.GL_LESS)
        pgl.glEnable(pgl.GL_DEPTH_TEST)

        pgl.glEnable(pgl.GL_LINE_SMOOTH)
        pgl.glShadeModel(pgl.GL_SMOOTH)
        pgl.glLineWidth(self.linewidth)

        pgl.glEnable(pgl.GL_BLEND)
        pgl.glBlendFunc(pgl.GL_SRC_ALPHA, pgl.GL_ONE_MINUS_SRC_ALPHA)

        if self.antialiasing:
            pgl.glHint(pgl.GL_LINE_SMOOTH_HINT, pgl.GL_NICEST)
            pgl.glHint(pgl.GL_POLYGON_SMOOTH_HINT, pgl.GL_NICEST)

        self.camera.setup_projection()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        if self.camera is not None:
            self.camera.setup_projection()

    def update(self, dt):
        self.controller.update(dt)

    def draw(self):
        self.plot._render_lock.acquire()
        self.camera.apply_transformation()

        calc_verts_pos, calc_verts_len = 0, 0
        calc_cverts_pos, calc_cverts_len = 0, 0

        should_update_caption = (perf_counter() - self.last_caption_update >
                                 self.caption_update_interval)

        if len(self.plot._functions.values()) == 0:
            self.drawing_first_object = True

        iterfunctions = iter(self.plot._functions.values())

        for r in iterfunctions:
            if self.drawing_first_object:
                self.camera.set_rot_preset(r.default_rot_preset)
                self.drawing_first_object = False

            pgl.glPushMatrix()
            r._draw()
            pgl.glPopMatrix()

            # might as well do this while we are
            # iterating and have the lock rather
            # than locking and iterating twice
            # per frame:

            if should_update_caption:
                try:
                    if r.calculating_verts:
                        calc_verts_pos += r.calculating_verts_pos
                        calc_verts_len += r.calculating_verts_len
                    if r.calculating_cverts:
                        calc_cverts_pos += r.calculating_cverts_pos
                        calc_cverts_len += r.calculating_cverts_len
                except ValueError:
                    pass

        for r in self.plot._pobjects:
            pgl.glPushMatrix()
            r._draw()
            pgl.glPopMatrix()

        if should_update_caption:
            self.update_caption(calc_verts_pos, calc_verts_len,
                                calc_cverts_pos, calc_cverts_len)
            self.last_caption_update = perf_counter()

        if self.plot._screenshot:
            self.plot._screenshot._execute_saving()

        self.plot._render_lock.release()

    def update_caption(self, calc_verts_pos, calc_verts_len,
            calc_cverts_pos, calc_cverts_len):
        caption = self.title
        if calc_verts_len or calc_cverts_len:
            caption += " (calculating"
            if calc_verts_len > 0:
                p = (calc_verts_pos / calc_verts_len) * 100
                caption += " vertices %i%%" % (p)
            if calc_cverts_len > 0:
                p = (calc_cverts_pos / calc_cverts_len) * 100
                caption += " colors %i%%" % (p)
            caption += ")"
        if self.caption != caption:
            self.set_caption(caption)
<|endoftext|>from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors


class PlotController:

    normal_mouse_sensitivity = 4.0
    modified_mouse_sensitivity = 1.0

    normal_key_sensitivity = 160.0
    modified_key_sensitivity = 40.0

    keymap = {
        key.LEFT: 'left',
        key.A: 'left',
        key.NUM_4: 'left',

        key.RIGHT: 'right',
        key.D: 'right',
        key.NUM_6: 'right',

        key.UP: 'up',
        key.W: 'up',
        key.NUM_8: 'up',

        key.DOWN: 'down',
        key.S: 'down',
        key.NUM_2: 'down',

        key.Z: 'rotate_z_neg',
        key.NUM_1: 'rotate_z_neg',

        key.C: 'rotate_z_pos',
        key.NUM_3: 'rotate_z_pos',

        key.Q: 'spin_left',
        key.NUM_7: 'spin_left',
        key.E: 'spin_right',
        key.NUM_9: 'spin_right',

        key.X: 'reset_camera',
        key.NUM_5: 'reset_camera',

        key.NUM_ADD: 'zoom_in',
        key.PAGEUP: 'zoom_in',
        key.R: 'zoom_in',

        key.NUM_SUBTRACT: 'zoom_out',
        key.PAGEDOWN: 'zoom_out',
        key.F: 'zoom_out',

        key.RSHIFT: 'modify_sensitivity',
        key.LSHIFT: 'modify_sensitivity',

        key.F1: 'rot_preset_xy',
        key.F2: 'rot_preset_xz',
        key.F3: 'rot_preset_yz',
        key.F4: 'rot_preset_perspective',

        key.F5: 'toggle_axes',
        key.F6: 'toggle_axe_colors',

        key.F8: 'save_image'
    }

    def __init__(self, window, *, invert_mouse_zoom=False, **kwargs):
        self.invert_mouse_zoom = invert_mouse_zoom
        self.window = window
        self.camera = window.camera
        self.action = {
            # Rotation around the view Y (up) vector
            'left': False,
            'right': False,
            # Rotation around the view X vector
            'up': False,
            'down': False,
            # Rotation around the view Z vector
            'spin_left': False,
            'spin_right': False,
            # Rotation around the model Z vector
            'rotate_z_neg': False,
            'rotate_z_pos': False,
            # Reset to the default rotation
            'reset_camera': False,
            # Performs camera z-translation
            'zoom_in': False,
            'zoom_out': False,
            # Use alternative sensitivity (speed)
            'modify_sensitivity': False,
            # Rotation presets
            'rot_preset_xy': False,
            'rot_preset_xz': False,
            'rot_preset_yz': False,
            'rot_preset_perspective': False,
            # axes
            'toggle_axes': False,
            'toggle_axe_colors': False,
            # screenshot
            'save_image': False
        }

    def update(self, dt):
        z = 0
        if self.action['zoom_out']:
            z -= 1
        if self.action['zoom_in']:
            z += 1
        if z != 0:
            self.camera.zoom_relative(z/10.0, self.get_key_sensitivity()/10.0)

        dx, dy, dz = 0, 0, 0
        if self.action['left']:
            dx -= 1
        if self.action['right']:
            dx += 1
        if self.action['up']:
            dy -= 1
        if self.action['down']:
            dy += 1
        if self.action['spin_left']:
            dz += 1
        if self.action['spin_right']:
            dz -= 1

        if not self.is_2D():
            if dx != 0:
                self.camera.euler_rotate(dx*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[1]))
            if dy != 0:
                self.camera.euler_rotate(dy*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[0]))
            if dz != 0:
                self.camera.euler_rotate(dz*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[2]))
        else:
            self.camera.mouse_translate(0, 0, dx*dt*self.get_key_sensitivity(),
                                        -dy*dt*self.get_key_sensitivity())

        rz = 0
        if self.action['rotate_z_neg'] and not self.is_2D():
            rz -= 1
        if self.action['rotate_z_pos'] and not self.is_2D():
            rz += 1

        if rz != 0:
            self.camera.euler_rotate(rz*dt*self.get_key_sensitivity(),
                                     *(get_basis_vectors()[2]))

        if self.action['reset_camera']:
            self.camera.reset()

        if self.action['rot_preset_xy']:
            self.camera.set_rot_preset('xy')
        if self.action['rot_preset_xz']:
            self.camera.set_rot_preset('xz')
        if self.action['rot_preset_yz']:
            self.camera.set_rot_preset('yz')
        if self.action['rot_preset_perspective']:
            self.camera.set_rot_preset('perspective')

        if self.action['toggle_axes']:
            self.action['toggle_axes'] = False
            self.camera.axes.toggle_visible()

        if self.action['toggle_axe_colors']:
            self.action['toggle_axe_colors'] = False
            self.camera.axes.toggle_colors()

        if self.action['save_image']:
            self.action['save_image'] = False
            self.window.plot.saveimage()

        return True

    def get_mouse_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_mouse_sensitivity
        else:
            return self.normal_mouse_sensitivity

    def get_key_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_key_sensitivity
        else:
            return self.normal_key_sensitivity

    def on_key_press(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = True

    def on_key_release(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & LEFT:
            if self.is_2D():
                self.camera.mouse_translate(x, y, dx, dy)
            else:
                self.camera.spherical_rotate((x - dx, y - dy), (x, y),
                                             self.get_mouse_sensitivity())
        if buttons & MIDDLE:
            self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                      self.get_mouse_sensitivity()/20.0)
        if buttons & RIGHT:
            self.camera.mouse_translate(x, y, dx, dy)

    def on_mouse_scroll(self, x, y, dx, dy):
        self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                  self.get_mouse_sensitivity())

    def is_2D(self):
        functions = self.window.plot._functions
        for i in functions:
            if len(functions[i].i_vars) > 1 or len(functions[i].d_vars) > 2:
                return False
        return True
<|endoftext|>from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence


class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """

    ## Class-level attributes
    ## used to register and lookup
    ## plot modes. See PlotModeBase
    ## for descriptions and usage.

    i_vars, d_vars = '', ''
    intervals = []
    aliases = []
    is_default = False

    ## Draw is the only method here which
    ## is meant to be overridden in child
    ## classes, and PlotModeBase provides
    ## a base implementation.
    def draw(self):
        raise NotImplementedError()

    ## Everything else in this file has to
    ## do with registration and retrieval
    ## of plot modes. This is where I've
    ## hidden much of the ugliness of automatic
    ## plot mode divination...

    ## Plot mode registry data structures
    _mode_alias_list = []
    _mode_map = {
        1: {1: {}, 2: {}},
        2: {1: {}, 2: {}},
        3: {1: {}, 2: {}},
    }  # [d][i][alias_str]: class
    _mode_default_map = {
        1: {},
        2: {},
        3: {},
    }  # [d][i]: class
    _i_var_max, _d_var_max = 2, 3

    def __new__(cls, *args, **kwargs):
        """
        This is the function which interprets
        arguments given to Plot.__init__ and
        Plot.__setattr__. Returns an initialized
        instance of the appropriate child class.
        """

        newargs, newkwargs = PlotMode._extract_options(args, kwargs)
        mode_arg = newkwargs.get('mode', '')

        # Interpret the arguments
        d_vars, intervals = PlotMode._interpret_args(newargs)
        i_vars = PlotMode._find_i_vars(d_vars, intervals)
        i, d = max([len(i_vars), len(intervals)]), len(d_vars)

        # Find the appropriate mode
        subcls = PlotMode._get_mode(mode_arg, i, d)

        # Create the object
        o = object.__new__(subcls)

        # Do some setup for the mode instance
        o.d_vars = d_vars
        o._fill_i_vars(i_vars)
        o._fill_intervals(intervals)
        o.options = newkwargs

        return o

    @staticmethod
    def _get_mode(mode_arg, i_var_count, d_var_count):
        """
        Tries to return an appropriate mode class.
        Intended to be called only by __new__.

        mode_arg
            Can be a string or a class. If it is a
            PlotMode subclass, it is simply returned.
            If it is a string, it can an alias for
            a mode or an empty string. In the latter
            case, we try to find a default mode for
            the i_var_count and d_var_count.

        i_var_count
            The number of independent variables
            needed to evaluate the d_vars.

        d_var_count
            The number of dependent variables;
            usually the number of functions to
            be evaluated in plotting.

        For example, a Cartesian function y = f(x) has
        one i_var (x) and one d_var (y). A parametric
        form x,y,z = f(u,v), f(u,v), f(u,v) has two
        two i_vars (u,v) and three d_vars (x,y,z).
        """
        # if the mode_arg is simply a PlotMode class,
        # check that the mode supports the numbers
        # of independent and dependent vars, then
        # return it
        try:
            m = None
            if issubclass(mode_arg, PlotMode):
                m = mode_arg
        except TypeError:
            pass
        if m:
            if not m._was_initialized:
                raise ValueError(("To use unregistered plot mode %s "
                                  "you must first call %s._init_mode().")
                                 % (m.__name__, m.__name__))
            if d_var_count != m.d_var_count:
                raise ValueError(("%s can only plot functions "
                                  "with %i dependent variables.")
                                 % (m.__name__,
                                     m.d_var_count))
            if i_var_count > m.i_var_count:
                raise ValueError(("%s cannot plot functions "
                                  "with more than %i independent "
                                  "variables.")
                                 % (m.__name__,
                                     m.i_var_count))
            return m
        # If it is a string, there are two possibilities.
        if isinstance(mode_arg, str):
            i, d = i_var_count, d_var_count
            if i > PlotMode._i_var_max:
                raise ValueError(var_count_error(True, True))
            if d > PlotMode._d_var_max:
                raise ValueError(var_count_error(False, True))
            # If the string is '', try to find a suitable
            # default mode
            if not mode_arg:
                return PlotMode._get_default_mode(i, d)
            # Otherwise, interpret the string as a mode
            # alias (e.g. 'cartesian', 'parametric', etc)
            else:
                return PlotMode._get_aliased_mode(mode_arg, i, d)
        else:
            raise ValueError("PlotMode argument must be "
                             "a class or a string")

    @staticmethod
    def _get_default_mode(i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        try:
            return PlotMode._mode_default_map[d][i]
        except KeyError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count until we
            # reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_default_mode(i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a default mode "
                                  "for %i independent and %i "
                                  "dependent variables.") % (i_vars, d))

    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        if alias not in PlotMode._mode_alias_list:
            raise ValueError(("Couldn't find a mode called"
                              " %s. Known modes: %s.")
                             % (alias, ", ".join(PlotMode._mode_alias_list)))
        try:
            return PlotMode._mode_map[d][i][alias]
        except TypeError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count and alias
            # until we reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a %s mode "
                                  "for %i independent and %i "
                                  "dependent variables.")
                                 % (alias, i_vars, d))

    @classmethod
    def _register(cls):
        """
        Called once for each user-usable plot mode.
        For Cartesian2D, it is invoked after the
        class definition: Cartesian2D._register()
        """
        name = cls.__name__
        cls._init_mode()

        try:
            i, d = cls.i_var_count, cls.d_var_count
            # Add the mode to _mode_map under all
            # given aliases
            for a in cls.aliases:
                if a not in PlotMode._mode_alias_list:
                    # Also track valid aliases, so
                    # we can quickly know when given
                    # an invalid one in _get_mode.
                    PlotMode._mode_alias_list.append(a)
                PlotMode._mode_map[d][i][a] = cls
            if cls.is_default:
                # If this mode was marked as the
                # default for this d,i combination,
                # also set that.
                PlotMode._mode_default_map[d][i] = cls

        except Exception as e:
            raise RuntimeError(("Failed to register "
                              "plot mode %s. Reason: %s")
                               % (name, (str(e))))

    @classmethod
    def _init_mode(cls):
        """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """
        def symbols_list(symbol_str):
            return [Symbol(s) for s in symbol_str]

        # Convert the vars strs into
        # lists of symbols.
        cls.i_vars = symbols_list(cls.i_vars)
        cls.d_vars = symbols_list(cls.d_vars)

        # Var count is used often, calculate
        # it once here
        cls.i_var_count = len(cls.i_vars)
        cls.d_var_count = len(cls.d_vars)

        if cls.i_var_count > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, False))
        if cls.d_var_count > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, False))

        # Try to use first alias as primary_alias
        if len(cls.aliases) > 0:
            cls.primary_alias = cls.aliases[0]
        else:
            cls.primary_alias = cls.__name__

        di = cls.intervals
        if len(di) != cls.i_var_count:
            raise ValueError("Plot mode must provide a "
                             "default interval for each i_var.")
        for i in range(cls.i_var_count):
            # default intervals must be given [min,max,steps]
            # (no var, but they must be in the same order as i_vars)
            if len(di[i]) != 3:
                raise ValueError("length should be equal to 3")

            # Initialize an incomplete interval,
            # to later be filled with a var when
            # the mode is instantiated.
            di[i] = PlotInterval(None, *di[i])

        # To prevent people from using modes
        # without these required fields set up.
        cls._was_initialized = True

    _was_initialized = False

    ## Initializer Helper Methods

    @staticmethod
    def _find_i_vars(functions, intervals):
        i_vars = []

        # First, collect i_vars in the
        # order they are given in any
        # intervals.
        for i in intervals:
            if i.v is None:
                continue
            elif i.v in i_vars:
                raise ValueError(("Multiple intervals given "
                                  "for %s.") % (str(i.v)))
            i_vars.append(i.v)

        # Then, find any remaining
        # i_vars in given functions
        # (aka d_vars)
        for f in functions:
            for a in f.free_symbols:
                if a not in i_vars:
                    i_vars.append(a)

        return i_vars

    def _fill_i_vars(self, i_vars):
        # copy default i_vars
        self.i_vars = [Symbol(str(i)) for i in self.i_vars]
        # replace with given i_vars
        for i in range(len(i_vars)):
            self.i_vars[i] = i_vars[i]

    def _fill_intervals(self, intervals):
        # copy default intervals
        self.intervals = [PlotInterval(i) for i in self.intervals]
        # track i_vars used so far
        v_used = []
        # fill copy of default
        # intervals with given info
        for i in range(len(intervals)):
            self.intervals[i].fill_from(intervals[i])
            if self.intervals[i].v is not None:
                v_used.append(self.intervals[i].v)
        # Find any orphan intervals and
        # assign them i_vars
        for i in range(len(self.intervals)):
            if self.intervals[i].v is None:
                u = [v for v in self.i_vars if v not in v_used]
                if len(u) == 0:
                    raise ValueError("length should not be equal to 0")
                self.intervals[i].v = u[0]
                v_used.append(u[0])

    @staticmethod
    def _interpret_args(args):
        interval_wrong_order = "PlotInterval %s was given before any function(s)."
        interpret_error = "Could not interpret %s as a function or interval."

        functions, intervals = [], []
        if isinstance(args[0], GeometryEntity):
            for coords in list(args[0].arbitrary_point()):
                functions.append(coords)
            intervals.append(PlotInterval.try_parse(args[0].plot_interval()))
        else:
            for a in args:
                i = PlotInterval.try_parse(a)
                if i is not None:
                    if len(functions) == 0:
                        raise ValueError(interval_wrong_order % (str(i)))
                    else:
                        intervals.append(i)
                else:
                    if is_sequence(a, include=str):
                        raise ValueError(interpret_error % (str(a)))
                    try:
                        f = sympify(a)
                        functions.append(f)
                    except TypeError:
                        raise ValueError(interpret_error % str(a))

        return functions, intervals

    @staticmethod
    def _extract_options(args, kwargs):
        newkwargs, newargs = {}, []
        for a in args:
            if isinstance(a, str):
                newkwargs = dict(newkwargs, **parse_option_string(a))
            else:
                newargs.append(a)
        newkwargs = dict(newkwargs, **kwargs)
        return newargs, newkwargs


def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """
    if is_plotting:
        v = "Plotting"
    else:
        v = "Registering plot modes"
    if is_independent:
        n, s = PlotMode._i_var_max, "independent"
    else:
        n, s = PlotMode._d_var_max, "dependent"
    return ("%s with more than %i %s variables "
            "is not supported.") % (v, n, s)
<|endoftext|>from sympy.plotting.series import BaseSeries, GenericDataSeries
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {
    ('Plot.append', 'Plot.extend'): ['matplotlib'],
}


# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True

def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False


def _deprecation_msg_m_a_r_f(attr):
    sympy_deprecation_warning(
        f"The `{attr}` property is deprecated. The `{attr}` keyword "
        "argument should be passed to a plotting function, which generates "
        "the appropriate data series. If needed, index the plot object to "
        "retrieve a specific data series.",
        deprecated_since_version="1.13",
        active_deprecations_target="deprecated-markers-annotations-fill-rectangles",
        stacklevel=4)


def _create_generic_data_series(**kwargs):
    keywords = ["annotations", "markers", "fill", "rectangles"]
    series = []
    for kw in keywords:
        dictionaries = kwargs.pop(kw, [])
        if dictionaries is None:
            dictionaries = []
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        for d in dictionaries:
            args = d.pop("args", [])
            series.append(GenericDataSeries(kw, *args, **d))
    return series


class Plot:
    """Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    For interactive work the function :func:`plot` is better suited.

    This class permits the plotting of SymPy expressions using numerous
    backends (:external:mod:`matplotlib`, textplot, the old pyglet module for SymPy, Google
    charts api, etc).

    The figure can contain an arbitrary number of plots of SymPy expressions,
    lists of coordinates of points, etc. Plot has a private attribute _series that
    contains all data series to be plotted (expressions for lines or surfaces,
    lists of points, etc (all subclasses of BaseSeries)). Those data series are
    instances of classes not imported by ``from sympy import *``.

    The customization of the figure is on two levels. Global options that
    concern the figure as a whole (e.g. title, xlabel, scale, etc) and
    per-data series options (e.g. name) and aesthetics (e.g. color, point shape,
    line type, etc.).

    The difference between options and aesthetics is that an aesthetic can be
    a function of the coordinates (or parameters in a parametric plot). The
    supported values for an aesthetic are:

    - None (the backend uses default values)
    - a constant
    - a function of one variable (the first coordinate or parameter)
    - a function of two variables (the first and second coordinate or parameters)
    - a function of three variables (only in nonparametric 3D plots)

    Their implementation depends on the backend so they may not work in some
    backends.

    If the plot is parametric and the arity of the aesthetic function permits
    it the aesthetic is calculated over parameters and not over coordinates.
    If the arity does not permit calculation over parameters the calculation is
    done over coordinates.

    Only cartesian coordinates are supported for the moment, but you can use
    the parametric plots to plot in polar, spherical and cylindrical
    coordinates.

    The arguments for the constructor Plot must be subclasses of BaseSeries.

    Any global option can be specified as a keyword argument.

    The global options for a figure are:

    - title : str
    - xlabel : str or Symbol
    - ylabel : str or Symbol
    - zlabel : str or Symbol
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - axis : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - aspect_ratio : tuple of two floats or {'auto'}
    - autoscale : bool
    - margin : float in [0, 1]
    - backend : {'default', 'matplotlib', 'text'} or a subclass of BaseBackend
