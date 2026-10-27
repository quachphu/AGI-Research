        Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
        """
        A = np.asanyarray(A)
        self.halfrange = max(self._vcenter-A.min(),
                             A.max()-self._vcenter)

    def autoscale_None(self, A):
        """Set *vmin* and *vmax*."""
        A = np.asanyarray(A)
        if self.halfrange is None and A.size:
            self.autoscale(A)

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmin:
            self._vmin = value
            self._vmax = 2*self.vcenter - value
            self._changed()

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmax:
            self._vmax = value
            self._vmin = 2*self.vcenter - value
            self._changed()

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, vcenter):
        if vcenter != self._vcenter:
            self._vcenter = vcenter
            # Trigger an update of the vmin/vmax values through the setter
            self.halfrange = self.halfrange
            self._changed()

    @property
    def halfrange(self):
        if self.vmin is None or self.vmax is None:
            return None
        return (self.vmax - self.vmin) / 2

    @halfrange.setter
    def halfrange(self, halfrange):
        if halfrange is None:
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = self.vcenter - abs(halfrange)
            self.vmax = self.vcenter + abs(halfrange)


def make_norm_from_scale(scale_cls, base_norm_cls=None, *, init=None):
    """
    Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`
    subclass.

    After ::

        @make_norm_from_scale(scale_cls)
        class norm_cls(Normalize):
            ...

    *norm_cls* is filled with methods so that normalization computations are
    forwarded to *scale_cls* (i.e., *scale_cls* is the scale that would be used
    for the colorbar of a mappable normalized with *norm_cls*).

    If *init* is not passed, then the constructor signature of *norm_cls*
    will be ``norm_cls(vmin=None, vmax=None, clip=False)``; these three
    parameters will be forwarded to the base class (``Normalize.__init__``),
    and a *scale_cls* object will be initialized with no arguments (other than
    a dummy axis).

    If the *scale_cls* constructor takes additional parameters, then *init*
    should be passed to `make_norm_from_scale`.  It is a callable which is
    *only* used for its signature.  First, this signature will become the
    signature of *norm_cls*.  Second, the *norm_cls* constructor will bind the
    parameters passed to it using this signature, extract the bound *vmin*,
    *vmax*, and *clip* values, pass those to ``Normalize.__init__``, and
    forward the remaining bound values (including any defaults defined by the
    signature) to the *scale_cls* constructor.
    """

    if base_norm_cls is None:
        return functools.partial(make_norm_from_scale, scale_cls, init=init)

    if isinstance(scale_cls, functools.partial):
        scale_args = scale_cls.args
        scale_kwargs_items = tuple(scale_cls.keywords.items())
        scale_cls = scale_cls.func
    else:
        scale_args = scale_kwargs_items = ()

    if init is None:
        def init(vmin=None, vmax=None, clip=False): pass

    return _make_norm_from_scale(
        scale_cls, scale_args, scale_kwargs_items,
        base_norm_cls, inspect.signature(init))


@functools.cache
def _make_norm_from_scale(
    scale_cls, scale_args, scale_kwargs_items,
    base_norm_cls, bound_init_signature,
):
    """
    Helper for `make_norm_from_scale`.

    This function is split out to enable caching (in particular so that
    different unpickles reuse the same class).  In order to do so,

    - ``functools.partial`` *scale_cls* is expanded into ``func, args, kwargs``
      to allow memoizing returned norms (partial instances always compare
      unequal, but we can check identity based on ``func, args, kwargs``;
    - *init* is replaced by *init_signature*, as signatures are picklable,
      unlike to arbitrary lambdas.
    """

    class Norm(base_norm_cls):
        def __reduce__(self):
            cls = type(self)
            # If the class is toplevel-accessible, it is possible to directly
            # pickle it "by name".  This is required to support norm classes
            # defined at a module's toplevel, as the inner base_norm_cls is
            # otherwise unpicklable (as it gets shadowed by the generated norm
            # class).  If either import or attribute access fails, fall back to
            # the general path.
            try:
                if cls is getattr(importlib.import_module(cls.__module__),
                                  cls.__qualname__):
                    return (_create_empty_object_of_class, (cls,), vars(self))
            except (ImportError, AttributeError):
                pass
            return (_picklable_norm_constructor,
                    (scale_cls, scale_args, scale_kwargs_items,
                     base_norm_cls, bound_init_signature),
                    vars(self))

        def __init__(self, *args, **kwargs):
            ba = bound_init_signature.bind(*args, **kwargs)
            ba.apply_defaults()
            super().__init__(
                **{k: ba.arguments.pop(k) for k in ["vmin", "vmax", "clip"]})
            self._scale = functools.partial(
                scale_cls, *scale_args, **dict(scale_kwargs_items))(
                    axis=None, **ba.arguments)
            self._trf = self._scale.get_transform()

        __init__.__signature__ = bound_init_signature.replace(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *bound_init_signature.parameters.values()])

        def __call__(self, value, clip=None):
            value, is_scalar = self.process_value(value)
            if self.vmin is None or self.vmax is None:
                self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            t_value = self._trf.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self._trf.transform([self.vmin, self.vmax])
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

        def inverse(self, value):
            if not self.scaled():
                raise ValueError("Not invertible until scaled")
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            t_vmin, t_vmax = self._trf.transform([self.vmin, self.vmax])
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            value, is_scalar = self.process_value(value)
            rescaled = value * (t_vmax - t_vmin)
            rescaled += t_vmin
            value = (self._trf
                     .inverted()
                     .transform(rescaled)
                     .reshape(np.shape(value)))
            return value[0] if is_scalar else value

        def autoscale_None(self, A):
            # i.e. A[np.isfinite(...)], but also for non-array A's
            in_trf_domain = np.extract(np.isfinite(self._trf.transform(A)), A)
            if in_trf_domain.size == 0:
                in_trf_domain = np.ma.masked
            return super().autoscale_None(in_trf_domain)

    if base_norm_cls is Normalize:
        Norm.__name__ = f"{scale_cls.__name__}Norm"
        Norm.__qualname__ = f"{scale_cls.__qualname__}Norm"
    else:
        Norm.__name__ = base_norm_cls.__name__
        Norm.__qualname__ = base_norm_cls.__qualname__
    Norm.__module__ = base_norm_cls.__module__
    Norm.__doc__ = base_norm_cls.__doc__

    return Norm


def _create_empty_object_of_class(cls):
    return cls.__new__(cls)


def _picklable_norm_constructor(*args):
    return _create_empty_object_of_class(_make_norm_from_scale(*args))


@make_norm_from_scale(
    scale.FuncScale,
    init=lambda functions, vmin=None, vmax=None, clip=False: None)
class FuncNorm(Normalize):
    """
    Arbitrary normalization using functions for the forward and inverse.

    Parameters
    ----------
    functions : (callable, callable)
        two-tuple of the forward and inverse functions for the normalization.
        The forward function must be monotonic.

        Both functions must have the signature ::

           def forward(values: array-like) -> array-like

    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.

    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values outside the range ``[vmin, vmax]`` are also
        transformed by the function, resulting in values outside ``[0, 1]``.
        This behavior is usually desirable, as colormaps can mark these *under*
        and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.
    """


LogNorm = make_norm_from_scale(
    functools.partial(scale.LogScale, nonpositive="mask"))(Normalize)
LogNorm.__name__ = LogNorm.__qualname__ = "LogNorm"
LogNorm.__doc__ = "Normalize a given value to the 0-1 range on a log scale."


@make_norm_from_scale(
    scale.SymmetricalLogScale,
    init=lambda linthresh, linscale=1., vmin=None, vmax=None, clip=False, *,
                base=10: None)
class SymLogNorm(Normalize):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    Parameters
    ----------
    linthresh : float
        The range within which the plot is linear (to avoid having the plot
        go to infinity around zero).
    linscale : float, default: 1
        This allows the linear range (-*linthresh* to *linthresh*) to be
        stretched relative to the logarithmic range. Its value is the
        number of decades to use for each half of the linear range. For
        example, when *linscale* == 1.0 (the default), the space used for
        the positive and negative halves of the linear range will be equal
        to one decade in the logarithmic range.
    base : float, default: 10
    """

    @property
    def linthresh(self):
        return self._scale.linthresh

    @linthresh.setter
    def linthresh(self, value):
        self._scale.linthresh = value


@make_norm_from_scale(
    scale.AsinhScale,
    init=lambda linear_width=1, vmin=None, vmax=None, clip=False: None)
class AsinhNorm(Normalize):
    """
    The inverse hyperbolic sine scale is approximately linear near
    the origin, but becomes logarithmic for larger positive
    or negative values. Unlike the `SymLogNorm`, the transition between
    these linear and logarithmic regions is smooth, which may reduce
    the risk of visual artifacts.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.

    Parameters
    ----------
    linear_width : float, default: 1
        The effective width of the linear region, beyond which
        the transformation becomes asymptotically logarithmic
    """

    @property
    def linear_width(self):
        return self._scale.linear_width

    @linear_width.setter
    def linear_width(self, value):
        self._scale.linear_width = value


class PowerNorm(Normalize):
    r"""
    Linearly map a given value to the 0-1 range and then apply
    a power-law normalization over that range.

    Parameters
    ----------
    gamma : float
        Power law exponent.
    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.
    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values above *vmax* are transformed by the power
        function, resulting in values above 1, and values below *vmin* are linearly
        transformed resulting in values below 0. This behavior is usually desirable, as
        colormaps can mark these *under* and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.

    Notes
    -----
    The normalization formula is

    .. math::

        \left ( \frac{x - v_{min}}{v_{max}  - v_{min}} \right )^{\gamma}

    For input values below *vmin*, gamma is set to one.
    """
    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            resdat = result.data
            resdat -= vmin
            resdat /= (vmax - vmin)
            resdat[resdat > 0] = np.power(resdat[resdat > 0], gamma)

            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")

        result, is_scalar = self.process_value(value)

        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax

        resdat = result.data
        resdat[resdat > 0] = np.power(resdat[resdat > 0], 1 / gamma)
        resdat *= (vmax - vmin)
        resdat += vmin

        result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result


class BoundaryNorm(Normalize):
    """
    Generate a colormap index based on discrete intervals.

    Unlike `Normalize` or `LogNorm`, `BoundaryNorm` maps values to integers
    instead of to the interval 0-1.
    """

    # Mapping to the 0-1 interval could have been done via piece-wise linear
    # interpolation, but using integers seems simpler, and reduces the number
    # of conversions back and forth between int and float.

    def __init__(self, boundaries, ncolors, clip=False, *, extend='neither'):
        """
        Parameters
        ----------
        boundaries : array-like
            Monotonically increasing sequence of at least 2 bin edges:  data
            falling in the n-th bin will be mapped to the n-th color.

        ncolors : int
            Number of colors in the colormap to be used.

        clip : bool, optional
            If clip is ``True``, out of range values are mapped to 0 if they
            are below ``boundaries[0]`` or mapped to ``ncolors - 1`` if they
            are above ``boundaries[-1]``.

            If clip is ``False``, out of range values are mapped to -1 if
            they are below ``boundaries[0]`` or mapped to *ncolors* if they are
            above ``boundaries[-1]``. These are then converted to valid indices
            by `Colormap.__call__`.

        extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
            Extend the number of bins to include one or both of the
            regions beyond the boundaries.  For example, if ``extend``
            is 'min', then the color to which the region between the first
            pair of boundaries is mapped will be distinct from the first
            color in the colormap, and by default a
            `~matplotlib.colorbar.Colorbar` will be drawn with
            the triangle extension on the left or lower end.

        Notes
        -----
        If there are fewer bins (including extensions) than colors, then the
        color index is chosen by linearly interpolating the ``[0, nbins - 1]``
        range onto the ``[0, ncolors - 1]`` range, effectively skipping some
        colors in the middle of the colormap.
        """
        if clip and extend != 'neither':
            raise ValueError("'clip=True' is not compatible with 'extend'")
        super().__init__(vmin=boundaries[0], vmax=boundaries[-1], clip=clip)
        self.boundaries = np.asarray(boundaries)
        self.N = len(self.boundaries)
        if self.N < 2:
            raise ValueError("You must provide at least 2 boundaries "
                             f"(1 region) but you passed in {boundaries!r}")
        self.Ncmap = ncolors
        self.extend = extend

        self._scale = None  # don't use the default scale.

        self._n_regions = self.N - 1  # number of colors needed
        self._offset = 0
        if extend in ('min', 'both'):
            self._n_regions += 1
            self._offset = 1
        if extend in ('max', 'both'):
            self._n_regions += 1
        if self._n_regions > self.Ncmap:
            raise ValueError(f"There are {self._n_regions} color bins "
                             "including extensions, but ncolors = "
                             f"{ncolors}; ncolors must equal or exceed the "
                             "number of bins")

    def __call__(self, value, clip=None):
        """
        This method behaves similarly to `.Normalize.__call__`, except that it
        returns integers or arrays of int16.
        """
        if clip is None:
            clip = self.clip

        xx, is_scalar = self.process_value(value)
        mask = np.ma.getmaskarray(xx)
        # Fill masked values a value above the upper boundary
        xx = np.atleast_1d(xx.filled(self.vmax + 1))
        if clip:
            np.clip(xx, self.vmin, self.vmax, out=xx)
            max_col = self.Ncmap - 1
        else:
            max_col = self.Ncmap
        # this gives us the bins in the lookup table in the range
        # [0, _n_regions - 1]  (the offset is set in the init)
        iret = np.digitize(xx, self.boundaries) - 1 + self._offset
        # if we have more colors than regions, stretch the region
        # index computed above to full range of the color bins.  This
        # will make use of the full range (but skip some of the colors
        # in the middle) such that the first region is mapped to the
        # first color and the last region is mapped to the last color.
        if self.Ncmap > self._n_regions:
            if self._n_regions == 1:
                # special case the 1 region case, pick the middle color
                iret[iret == 0] = (self.Ncmap - 1) // 2
            else:
                # otherwise linearly remap the values from the region index
                # to the color index spaces
                iret = (self.Ncmap - 1) / (self._n_regions - 1) * iret
        # cast to 16bit integers in all cases
        iret = iret.astype(np.int16)
        iret[xx < self.vmin] = -1
        iret[xx >= self.vmax] = max_col
        ret = np.ma.array(iret, mask=mask)
        if is_scalar:
            ret = int(ret[0])  # assume python scalar
        return ret

    def inverse(self, value):
        """
        Raises
        ------
        ValueError
            BoundaryNorm is not invertible, so calling this method will always
            raise an error
        """
        raise ValueError("BoundaryNorm is not invertible")


class NoNorm(Normalize):
    """
    Dummy replacement for `Normalize`, for the case where we want to use
    indices directly in a `~matplotlib.cm.ScalarMappable`.
    """
    def __call__(self, value, clip=None):
        if np.iterable(value):
            return np.ma.array(value)
        return value

    def inverse(self, value):
        if np.iterable(value):
            return np.ma.array(value)
        return value


def rgb_to_hsv(arr):
    """
    Convert an array of float RGB values (in the range [0, 1]) to HSV values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to HSV values in range [0, 1]
    """
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {arr.shape} was found.")

    in_shape = arr.shape
    arr = np.array(
        arr, copy=False,
        dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    # Check if input is in the expected range
    if np.any(arr_max > 1):
        raise ValueError(
            "Input array must be in the range [0, 1]. "
            f"Found a maximum value of {arr_max.max()}"
        )

    if arr.min() < 0:
        raise ValueError(
            "Input array must be in the range [0, 1]. "
            f"Found a minimum value of {arr.min()}"
        )

    ipos = arr_max > 0
    delta = np.ptp(arr, -1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)


def hsv_to_rgb(hsv):
    """
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {hsv.shape} was found.")

    in_shape = hsv.shape
    hsv = np.array(
        hsv, copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


def _vector_magnitude(arr):
    # things that don't work here:
    #  * np.linalg.norm: drops mask from ma.array
    #  * np.sum: drops mask from ma.array unless entire vector is masked
    sum_sq = 0
    for i in range(arr.shape[-1]):
        sum_sq += arr[..., i, np.newaxis] ** 2
    return np.sqrt(sum_sq)


class LightSource:
    """
    Create a light source coming from the specified azimuth and elevation.
    Angles are in degrees, with the azimuth measured
    clockwise from north and elevation up from the zero plane of the surface.

    `shade` is used to produce "shaded" RGB values for a data array.
    `shade_rgb` can be used to combine an RGB image with an elevation map.
    `hillshade` produces an illumination map of a surface.
    """

    def __init__(self, azdeg=315, altdeg=45, hsv_min_val=0, hsv_max_val=1,
                 hsv_min_sat=1, hsv_max_sat=0):
        """
        Specify the azimuth (measured clockwise from south) and altitude
        (measured up from the plane of the surface) of the light source
        in degrees.

        Parameters
        ----------
        azdeg : float, default: 315 degrees (from the northwest)
            The azimuth (0-360, degrees clockwise from North) of the light
            source.
        altdeg : float, default: 45 degrees
            The altitude (0-90, degrees up from horizontal) of the light
            source.
        hsv_min_val : number, default: 0
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_max_val : number, default: 1
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_min_sat : number, default: 1
            The minimum saturation value that the *intensity* map can shift the output
            image to.
        hsv_max_sat : number, default: 0
            The maximum saturation value that the *intensity* map can shift the output
            image to.

        Notes
        -----
        For backwards compatibility, the parameters *hsv_min_val*,
        *hsv_max_val*, *hsv_min_sat*, and *hsv_max_sat* may be supplied at
        initialization as well.  However, these parameters will only be used if
        "blend_mode='hsv'" is passed into `shade` or `shade_rgb`.
        See the documentation for `blend_hsv` for more details.
        """
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.hsv_min_val = hsv_min_val
        self.hsv_max_val = hsv_max_val
        self.hsv_min_sat = hsv_min_sat
        self.hsv_max_sat = hsv_max_sat

    @property
    def direction(self):
        """The unit vector direction towards the light source."""
        # Azimuth is in degrees clockwise from North. Convert to radians
        # counterclockwise from East (mathematical notation).
        az = np.radians(90 - self.azdeg)
        alt = np.radians(self.altdeg)
        return np.array([
            np.cos(az) * np.cos(alt),
            np.sin(az) * np.cos(alt),
            np.sin(alt)
        ])

    def hillshade(self, elevation, vert_exag=1, dx=1, dy=1, fraction=1.):
        """
        Calculate the illumination intensity for a surface using the defined
        azimuth and elevation for the light source.

        This computes the normal vectors for the surface, and then passes them
        on to `shade_normals`

        Parameters
        ----------
        elevation : 2D array-like
            The height values used to generate an illumination map
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topographic effects.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """

        # Because most image and raster GIS data has the first row in the array
        # as the "top" of the image, dy is implicitly negative.  This is
        # consistent to what `imshow` assumes, as well.
        dy = -dy

        # compute the normal vectors from the partial derivatives
        e_dy, e_dx = np.gradient(vert_exag * elevation, dy, dx)

        # .view is to keep subclasses
        normal = np.empty(elevation.shape + (3,)).view(type(elevation))
        normal[..., 0] = -e_dx
        normal[..., 1] = -e_dy
        normal[..., 2] = 1
        normal /= _vector_magnitude(normal)

        return self.shade_normals(normal, fraction)

    def shade_normals(self, normals, fraction=1.):
        """
        Calculate the illumination intensity for the normal vectors of a
        surface using the defined azimuth and elevation for the light source.

        Imagine an artificial sun placed at infinity in some azimuth and
        elevation position illuminating our surface. The parts of the surface
        that slope toward the sun should brighten while those sides facing away
        should become darker.

        Parameters
        ----------
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """

        intensity = normals.dot(self.direction)

        # Apply contrast stretch
        imin, imax = intensity.min(), intensity.max()
        intensity *= fraction

        # Rescale to 0-1, keeping range before contrast stretch
        # If constant slope, keep relative scaling (i.e. flat should be 0.5,
        # fully occluded 0, etc.)
        if (imax - imin) > 1e-6:
            # Strictly speaking, this is incorrect. Negative values should be
            # clipped to 0 because they're fully occluded. However, rescaling
            # in this manner is consistent with the previous implementation and
            # visually appears better than a "hard" clip.
            intensity -= imin
            intensity /= (imax - imin)
        intensity = np.clip(intensity, 0, 1)

        return intensity

    def shade(self, data, cmap, norm=None, blend_mode='overlay', vmin=None,
              vmax=None, vert_exag=1, dx=1, dy=1, fraction=1, **kwargs):
        """
        Combine colormapped data values with an illumination intensity map
        (a.k.a.  "hillshade") of the values.

        Parameters
        ----------
        data : 2D array-like
            The height values used to generate a shaded map.
        cmap : `~matplotlib.colors.Colormap`
            The colormap used to color the *data* array. Note that this must be
            a `~matplotlib.colors.Colormap` instance.  For example, rather than
            passing in ``cmap='gist_earth'``, use
            ``cmap=plt.get_cmap('gist_earth')`` instead.
        norm : `~matplotlib.colors.Normalize` instance, optional
            The normalization used to scale values before colormapping. If
            None, the input will be linearly scaled between its min and max.
        blend_mode : {'hsv', 'overlay', 'soft'} or callable, optional
            The type of blending used to combine the colormapped data
            values with the illumination intensity.  Default is
            "overlay".  Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to
            combine an (M, N, 3) RGB array of floats (ranging 0 to 1) with
            an (M, N, 1) hillshade array (also 0 to 1).  (Call signature
            ``func(rgb, illum, **kwargs)``) Additional kwargs supplied
            to this function will be passed on to the *blend_mode*
            function.
        vmin : float or None, optional
            The minimum value used in colormapping *data*. If *None* the
            minimum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vmax : float or None, optional
            The maximum value used in colormapping *data*. If *None* the
            maximum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        **kwargs
            Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 4) array of floats ranging between 0-1.
        """
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        rgb0 = cmap(norm(data))
        rgb1 = self.shade_rgb(rgb0, elevation=data, blend_mode=blend_mode,
                              vert_exag=vert_exag, dx=dx, dy=dy,
                              fraction=fraction, **kwargs)
        # Don't overwrite the alpha channel, if present.
        rgb0[..., :3] = rgb1[..., :3]
        return rgb0

    def shade_rgb(self, rgb, elevation, fraction=1., blend_mode='hsv',
                  vert_exag=1, dx=1, dy=1, **kwargs):
        """
        Use this light source to adjust the colors of the *rgb* input array to
        give the impression of a shaded relief map with the given *elevation*.

        Parameters
        ----------
        rgb : array-like
            An (M, N, 3) RGB array, assumed to be in the range of 0 to 1.
        elevation : array-like
            An (M, N) array of the height values used to generate a shaded map.
        fraction : number
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        blend_mode : {'hsv', 'overlay', 'soft'} or callable, optional
            The type of blending used to combine the colormapped data values
            with the illumination intensity.  For backwards compatibility, this
            defaults to "hsv". Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to combine an
            (M, N, 3) RGB array of floats (ranging 0 to 1) with an (M, N, 1)
            hillshade array (also 0 to 1).  (Call signature
            ``func(rgb, illum, **kwargs)``)
            Additional kwargs supplied to this function will be passed on to
            the *blend_mode* function.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        **kwargs
            Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        `~numpy.ndarray`
            An (m, n, 3) array of floats ranging between 0-1.
        """
        # Calculate the "hillshade" intensity.
        intensity = self.hillshade(elevation, vert_exag, dx, dy, fraction)
        intensity = intensity[..., np.newaxis]

        # Blend the hillshade and rgb data using the specified mode
        lookup = {
                'hsv': self.blend_hsv,
                'soft': self.blend_soft_light,
                'overlay': self.blend_overlay,
                }
        if blend_mode in lookup:
            blend = lookup[blend_mode](rgb, intensity, **kwargs)
        else:
            try:
                blend = blend_mode(rgb, intensity, **kwargs)
            except TypeError as err:
                raise ValueError('"blend_mode" must be callable or one of '
                                 f'{lookup.keys}') from err

        # Only apply result where hillshade intensity isn't masked
        if np.ma.is_masked(intensity):
            mask = intensity.mask[..., 0]
            for i in range(3):
                blend[..., i][mask] = rgb[..., i][mask]

        return blend

    def blend_hsv(self, rgb, intensity, hsv_max_sat=None, hsv_max_val=None,
                  hsv_min_val=None, hsv_min_sat=None):
        """
        Take the input data array, convert to HSV values in the given colormap,
        then adjust those color values to give the impression of a shaded
        relief map with a specified light source.  RGBA values are returned,
        which can then be used to plot the shaded image with imshow.

        The color of the resulting image will be darkened by moving the (s, v)
        values (in HSV colorspace) toward (hsv_min_sat, hsv_min_val) in the
        shaded regions, or lightened by sliding (s, v) toward (hsv_max_sat,
        hsv_max_val) in regions that are illuminated.  The default extremes are
        chose so that completely shaded points are nearly black (s = 1, v = 0)
        and completely illuminated points are nearly white (s = 0, v = 1).

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).
        hsv_max_sat : number, optional
            The maximum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_min_sat : number, optional
            The minimum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_max_val : number, optional
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.
        hsv_min_val : number, optional
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
        # Backward compatibility...
        if hsv_max_sat is None:
            hsv_max_sat = self.hsv_max_sat
        if hsv_max_val is None:
            hsv_max_val = self.hsv_max_val
        if hsv_min_sat is None:
            hsv_min_sat = self.hsv_min_sat
        if hsv_min_val is None:
            hsv_min_val = self.hsv_min_val

        # Expects a 2D intensity array scaled between -1 to 1...
        intensity = intensity[..., 0]
        intensity = 2 * intensity - 1

        # Convert to rgb, then rgb to hsv
        hsv = rgb_to_hsv(rgb[:, :, 0:3])
        hue, sat, val = np.moveaxis(hsv, -1, 0)

        # Modify hsv values (in place) to simulate illumination.
        # putmask(A, mask, B) <=> A[mask] = B[mask]
        np.putmask(sat, (np.abs(sat) > 1.e-10) & (intensity > 0),
                   (1 - intensity) * sat + intensity * hsv_max_sat)
        np.putmask(sat, (np.abs(sat) > 1.e-10) & (intensity < 0),
                   (1 + intensity) * sat - intensity * hsv_min_sat)
        np.putmask(val, intensity > 0,
                   (1 - intensity) * val + intensity * hsv_max_val)
        np.putmask(val, intensity < 0,
                   (1 + intensity) * val - intensity * hsv_min_val)
        np.clip(hsv[:, :, 1:], 0, 1, out=hsv[:, :, 1:])

        # Convert modified hsv back to rgb.
        return hsv_to_rgb(hsv)

    def blend_soft_light(self, rgb, intensity):
        """
        Combine an RGB image with an intensity map using "soft light" blending,
        using the "pegtop" formula.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
        return 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2

    def blend_overlay(self, rgb, intensity):
        """
        Combine an RGB image with an intensity map using "overlay" blending.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        ndarray
            An (M, N, 3) RGB array representing the combined images.
        """
        low = 2 * intensity * rgb
        high = 1 - 2 * (1 - intensity) * (1 - rgb)
        return np.where(rgb <= 0.5, low, high)


def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the `BoundaryNorm`.
        Value ``v`` is quantized to level ``i`` if ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If *extend* is "neither" there
        must be ``n_level - 1`` colors. For an *extend* of "min" or "max" add
        one extra color, and for an *extend* of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See `~.Axes.contourf` for details.

    Returns
    -------
    cmap : `~matplotlib.colors.Colormap`
    norm : `~matplotlib.colors.Normalize`
    """
    slice_map = {
        'both': slice(1, -1),
        'min': slice(1, None),
        'max': slice(0, -1),
        'neither': slice(0, None),
    }
    _api.check_in_list(slice_map, extend=extend)
    color_slice = slice_map[extend]

    n_data_colors = len(levels) - 1
    n_expected = n_data_colors + color_slice.start - (color_slice.stop or 0)
    if len(colors) != n_expected:
        raise ValueError(
            f'With extend == {extend!r} and {len(levels)} levels, '
            f'expected {n_expected} colors, but got {len(colors)}')

    cmap = ListedColormap(colors[color_slice], N=n_data_colors)

    if extend in ['min', 'both']:
        cmap.set_under(colors[0])
    else:
        cmap.set_under('none')

    if extend in ['max', 'both']:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over('none')

    cmap.colorbar_extend = extend

    norm = BoundaryNorm(levels, ncolors=n_data_colors)
    return cmap, norm
<|endoftext|>"""
Enums representing sets of strings that Matplotlib uses as input parameters.

Matplotlib often uses simple data types like strings or tuples to define a
concept; e.g. the line capstyle can be specified as one of 'butt', 'round',
or 'projecting'. The classes in this module are used internally and serve to
document these concepts formally.

As an end-user you will not use these classes directly, but only the values
they define.
"""

from enum import Enum
from matplotlib import _docstring


class JoinStyle(str, Enum):
    """
    Define how the connection between two line segments is drawn.

    For a visual impression of each *JoinStyle*, `view these docs online
    <JoinStyle>`, or run `JoinStyle.demo`.

    Lines in Matplotlib are typically defined by a 1D `~.path.Path` and a
    finite ``linewidth``, where the underlying 1D `~.path.Path` represents the
    center of the stroked line.

    By default, `~.backend_bases.GraphicsContextBase` defines the boundaries of
    a stroked line to simply be every point within some radius,
    ``linewidth/2``, away from any point of the center line. However, this
    results in corners appearing "rounded", which may not be the desired
    behavior if you are drawing, for example, a polygon or pointed star.

    **Supported values:**

    .. rst-class:: value-list

        'miter'
            the "arrow-tip" style. Each boundary of the filled-in area will
            extend in a straight line parallel to the tangent vector of the
            centerline at the point it meets the corner, until they meet in a
            sharp point.
        'round'
            stokes every point within a radius of ``linewidth/2`` of the center
            lines.
        'bevel'
            the "squared-off" style. It can be thought of as a rounded corner
            where the "circular" part of the corner has been cut off.

    .. note::

        Very long miter tips are cut off (to form a *bevel*) after a
        backend-dependent limit called the "miter limit", which specifies the
        maximum allowed ratio of miter length to line width. For example, the
        PDF backend uses the default value of 10 specified by the PDF standard,
        while the SVG backend does not even specify the miter limit, resulting
        in a default value of 4 per the SVG specification. Matplotlib does not
        currently allow the user to adjust this parameter.

        A more detailed description of the effect of a miter limit can be found
        in the `Mozilla Developer Docs
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-miterlimit>`_

    .. plot::
        :alt: Demo of possible JoinStyle's

        from matplotlib._enums import JoinStyle
        JoinStyle.demo()

    """

    miter = "miter"
    round = "round"
    bevel = "bevel"

    @staticmethod
    def demo():
        """Demonstrate how each JoinStyle looks for various join angles."""
        import numpy as np
        import matplotlib.pyplot as plt

        def plot_angle(ax, x, y, angle, style):
            phi = np.radians(angle)
            xx = [x + .5, x, x + .5*np.cos(phi)]
            yy = [y, y, y + .5*np.sin(phi)]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)

        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.set_title('Join style')
        for x, style in enumerate(['miter', 'round', 'bevel']):
            ax.text(x, 5, style)
            for y, angle in enumerate([20, 45, 60, 90, 120]):
                plot_angle(ax, x, y, angle, style)
                if x == 0:
                    ax.text(-1.3, y, f'{angle} degrees')
        ax.set_xlim(-1.5, 2.75)
        ax.set_ylim(-.5, 5.5)
        ax.set_axis_off()
        fig.show()


JoinStyle.input_description = "{" \
        + ", ".join([f"'{js.name}'" for js in JoinStyle]) \
        + "}"


class CapStyle(str, Enum):
    r"""
    Define how the two endpoints (caps) of an unclosed line are drawn.

    How to draw the start and end points of lines that represent a closed curve
    (i.e. that end in a `~.path.Path.CLOSEPOLY`) is controlled by the line's
    `JoinStyle`. For all other lines, how the start and end points are drawn is
    controlled by the *CapStyle*.

    For a visual impression of each *CapStyle*, `view these docs online
    <CapStyle>` or run `CapStyle.demo`.

    By default, `~.backend_bases.GraphicsContextBase` draws a stroked line as
    squared off at its endpoints.

    **Supported values:**

    .. rst-class:: value-list

        'butt'
            the line is squared off at its endpoint.
        'projecting'
            the line is squared off as in *butt*, but the filled in area
            extends beyond the endpoint a distance of ``linewidth/2``.
        'round'
            like *butt*, but a semicircular cap is added to the end of the
            line, of radius ``linewidth/2``.

    .. plot::
        :alt: Demo of possible CapStyle's

        from matplotlib._enums import CapStyle
        CapStyle.demo()

    """
    butt = "butt"
    projecting = "projecting"
    round = "round"

    @staticmethod
    def demo():
        """Demonstrate how each CapStyle looks for a thick line segment."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 1.2))
        ax = fig.add_axes([0, 0, 1, 0.8])
        ax.set_title('Cap style')

        for x, style in enumerate(['butt', 'round', 'projecting']):
            ax.text(x+0.25, 0.85, style, ha='center')
            xx = [x, x+0.5]
            yy = [0, 0]
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_capstyle=style)
            ax.plot(xx, yy, lw=1, color='black')
            ax.plot(xx, yy, 'o', color='tab:red', markersize=3)

        ax.set_ylim(-.5, 1.5)
        ax.set_axis_off()
        fig.show()


CapStyle.input_description = "{" \
        + ", ".join([f"'{cs.name}'" for cs in CapStyle]) \
        + "}"

_docstring.interpd.register(
    JoinStyle=JoinStyle.input_description,
    CapStyle=CapStyle.input_description,
)
<|endoftext|>"""
A module for parsing and generating `fontconfig patterns`_.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
"""

# This class logically belongs in `matplotlib.font_manager`, but placing it
# there would have created cyclical dependency problems, because it also needs
# to be available from `matplotlib.rcsetup` (for parsing matplotlibrc files).

from functools import lru_cache, partial
import re

from pyparsing import (
    Group, Optional, ParseException, Regex, StringEnd, Suppress, ZeroOrMore, one_of)


_family_punc = r'\\\-:,'
_family_unescape = partial(re.compile(r'\\(?=[%s])' % _family_punc).sub, '')
_family_escape = partial(re.compile(r'(?=[%s])' % _family_punc).sub, r'\\')
_value_punc = r'\\=_:,'
_value_unescape = partial(re.compile(r'\\(?=[%s])' % _value_punc).sub, '')
_value_escape = partial(re.compile(r'(?=[%s])' % _value_punc).sub, r'\\')


_CONSTANTS = {
    'thin':           ('weight', 'light'),
    'extralight':     ('weight', 'light'),
    'ultralight':     ('weight', 'light'),
    'light':          ('weight', 'light'),
    'book':           ('weight', 'book'),
    'regular':        ('weight', 'regular'),
    'normal':         ('weight', 'normal'),
    'medium':         ('weight', 'medium'),
    'demibold':       ('weight', 'demibold'),
    'semibold':       ('weight', 'semibold'),
    'bold':           ('weight', 'bold'),
    'extrabold':      ('weight', 'extra bold'),
    'black':          ('weight', 'black'),
    'heavy':          ('weight', 'heavy'),
    'roman':          ('slant', 'normal'),
    'italic':         ('slant', 'italic'),
    'oblique':        ('slant', 'oblique'),
    'ultracondensed': ('width', 'ultra-condensed'),
    'extracondensed': ('width', 'extra-condensed'),
    'condensed':      ('width', 'condensed'),
    'semicondensed':  ('width', 'semi-condensed'),
    'expanded':       ('width', 'expanded'),
    'extraexpanded':  ('width', 'extra-expanded'),
    'ultraexpanded':  ('width', 'ultra-expanded'),
}


@lru_cache  # The parser instance is a singleton.
def _make_fontconfig_parser():
    def comma_separated(elem):
        return elem + ZeroOrMore(Suppress(",") + elem)

    family = Regex(fr"([^{_family_punc}]|(\\[{_family_punc}]))*")
    size = Regex(r"([0-9]+\.?[0-9]*|\.[0-9]+)")
    name = Regex(r"[a-z]+")
    value = Regex(fr"([^{_value_punc}]|(\\[{_value_punc}]))*")
    prop = Group((name + Suppress("=") + comma_separated(value)) | one_of(_CONSTANTS))
    return (
        Optional(comma_separated(family)("families"))
        + Optional("-" + comma_separated(size)("sizes"))
        + ZeroOrMore(":" + prop("properties*"))
        + StringEnd()
    )


# `parse_fontconfig_pattern` is a bottleneck during the tests because it is
# repeatedly called when the rcParams are reset (to validate the default
# fonts).  In practice, the cache size doesn't grow beyond a few dozen entries
# during the test suite.
@lru_cache
def parse_fontconfig_pattern(pattern):
    """
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    """
    parser = _make_fontconfig_parser()
    try:
        parse = parser.parse_string(pattern)
    except ParseException as err:
        # explain becomes a plain method on pyparsing 3 (err.explain(0)).
        raise ValueError("\n" + ParseException.explain(err, 0)) from None
    parser.reset_cache()
    props = {}
    if "families" in parse:
        props["family"] = [*map(_family_unescape, parse["families"])]
    if "sizes" in parse:
        props["size"] = [*parse["sizes"]]
    for prop in parse.get("properties", []):
        if len(prop) == 1:
            prop = _CONSTANTS[prop[0]]
        k, *v = prop
        props.setdefault(k, []).extend(map(_value_unescape, v))
    return props


def generate_fontconfig_pattern(d):
    """Convert a `.FontProperties` to a fontconfig pattern string."""
    kvs = [(k, getattr(d, f"get_{k}")())
           for k in ["style", "variant", "weight", "stretch", "file", "size"]]
    # Families is given first without a leading keyword.  Other entries (which
    # are necessarily scalar) are given as key=value, skipping Nones.
    return (",".join(_family_escape(f) for f in d.get_family())
            + "".join(f":{k}={_value_escape(str(v))}"
                      for k, v in kvs if v is not None))
<|endoftext|>"""
Low-level text helper utilities.
"""

from __future__ import annotations

import dataclasses

from . import _api
from .ft2font import FT2Font, Kerning, LoadFlags


@dataclasses.dataclass(frozen=True)
class LayoutItem:
    ft_object: FT2Font
    char: str
    glyph_idx: int
    x: float
    prev_kern: float


def warn_on_missing_glyph(codepoint, fontnames):
    _api.warn_external(
        f"Glyph {codepoint} "
        f"({chr(codepoint).encode('ascii', 'namereplace').decode('ascii')}) "
        f"missing from font(s) {fontnames}.")

    block = ("Hebrew" if 0x0590 <= codepoint <= 0x05ff else
             "Arabic" if 0x0600 <= codepoint <= 0x06ff else
             "Devanagari" if 0x0900 <= codepoint <= 0x097f else
             "Bengali" if 0x0980 <= codepoint <= 0x09ff else
             "Gurmukhi" if 0x0a00 <= codepoint <= 0x0a7f else
             "Gujarati" if 0x0a80 <= codepoint <= 0x0aff else
             "Oriya" if 0x0b00 <= codepoint <= 0x0b7f else
             "Tamil" if 0x0b80 <= codepoint <= 0x0bff else
             "Telugu" if 0x0c00 <= codepoint <= 0x0c7f else
             "Kannada" if 0x0c80 <= codepoint <= 0x0cff else
             "Malayalam" if 0x0d00 <= codepoint <= 0x0d7f else
             "Sinhala" if 0x0d80 <= codepoint <= 0x0dff else
             None)
    if block:
        _api.warn_external(
            f"Matplotlib currently does not support {block} natively.")


def layout(string, font, *, kern_mode=Kerning.DEFAULT):
    """
    Render *string* with *font*.

    For each character in *string*, yield a LayoutItem instance. When such an instance
    is yielded, the font's glyph is set to the corresponding character.

    Parameters
    ----------
    string : str
        The string to be rendered.
    font : FT2Font
        The font.
    kern_mode : Kerning
        A FreeType kerning mode.

    Yields
    ------
    LayoutItem
    """
    x = 0
    prev_glyph_idx = None
    char_to_font = font._get_fontmap(string)
    base_font = font
    for char in string:
        # This has done the fallback logic
        font = char_to_font.get(char, base_font)
        glyph_idx = font.get_char_index(ord(char))
        kern = (
            base_font.get_kerning(prev_glyph_idx, glyph_idx, kern_mode) / 64
            if prev_glyph_idx is not None else 0.
        )
        x += kern
        glyph = font.load_glyph(glyph_idx, flags=LoadFlags.NO_HINTING)
        yield LayoutItem(font, char, glyph_idx, x, kern)
        x += glyph.linearHoriAdvance / 65536
        prev_glyph_idx = glyph_idx
<|endoftext|>"""Contains classes for generating hatch patterns."""

import numpy as np

from matplotlib import _api
from matplotlib.path import Path


class HatchPatternBase:
    """The base class for a hatch pattern."""
    pass


class HorizontalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('-') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = 0.0
        vertices[0::2, 1] = steps
        vertices[1::2, 0] = 1.0
        vertices[1::2, 1] = steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class VerticalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('|') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = steps
        vertices[0::2, 1] = 0.0
        vertices[1::2, 0] = steps
        vertices[1::2, 1] = 1.0
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class NorthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('/') + hatch.count('x') + hatch.count('X')) * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class SouthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('\\') + hatch.count('x') + hatch.count('X'))
            * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class Shapes(HatchPatternBase):
    filled = False

    def __init__(self, hatch, density):
        if self.num_rows == 0:
            self.num_shapes = 0
            self.num_vertices = 0
        else:
            self.num_shapes = ((self.num_rows // 2 + 1) * (self.num_rows + 1) +
                               (self.num_rows // 2) * self.num_rows)
            self.num_vertices = (self.num_shapes *
                                 len(self.shape_vertices) *
                                 (1 if self.filled else 2))

    def set_vertices_and_codes(self, vertices, codes):
        offset = 1.0 / self.num_rows
        shape_vertices = self.shape_vertices * offset * self.size
        shape_codes = self.shape_codes
        if not self.filled:
            shape_vertices = np.concatenate(  # Forward, then backward.
                [shape_vertices, shape_vertices[::-1] * 0.9])
            shape_codes = np.concatenate([shape_codes, shape_codes])
        vertices_parts = []
        codes_parts = []
        for row in range(self.num_rows + 1):
            if row % 2 == 0:
                cols = np.linspace(0, 1, self.num_rows + 1)
            else:
                cols = np.linspace(offset / 2, 1 - offset / 2, self.num_rows)
            row_pos = row * offset
            for col_pos in cols:
                vertices_parts.append(shape_vertices + [col_pos, row_pos])
                codes_parts.append(shape_codes)
        np.concatenate(vertices_parts, out=vertices)
        np.concatenate(codes_parts, out=codes)


class Circles(Shapes):
    def __init__(self, hatch, density):
        path = Path.unit_circle()
        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        super().__init__(hatch, density)


class SmallCircles(Circles):
    size = 0.2

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('o')) * density
        super().__init__(hatch, density)


class LargeCircles(Circles):
    size = 0.35

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('O')) * density
        super().__init__(hatch, density)


class SmallFilledCircles(Circles):
    size = 0.1
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('.')) * density
        super().__init__(hatch, density)


class Stars(Shapes):
    size = 1.0 / 3.0
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('*')) * density
        path = Path.unit_regular_star(5)
        self.shape_vertices = path.vertices
        self.shape_codes = np.full(len(self.shape_vertices), Path.LINETO,
                                   dtype=Path.code_type)
        self.shape_codes[0] = Path.MOVETO
        super().__init__(hatch, density)

_hatch_types = [
    HorizontalHatch,
    VerticalHatch,
    NorthEastHatch,
    SouthEastHatch,
    SmallCircles,
    LargeCircles,
    SmallFilledCircles,
    Stars
    ]


def _validate_hatch_pattern(hatch):
    valid_hatch_patterns = set(r'-+|/\xXoO.*')
    if hatch is not None:
        invalids = set(hatch).difference(valid_hatch_patterns)
        if invalids:
            valid = ''.join(sorted(valid_hatch_patterns))
            invalids = ''.join(sorted(invalids))
            _api.warn_deprecated(
                '3.4',
                removal='3.11',  # one release after custom hatches (#20690)
                message=f'hatch must consist of a string of "{valid}" or '
                        'None, but found the following invalid values '
                        f'"{invalids}". Passing invalid values is deprecated '
                        'since %(since)s and will become an error in %(removal)s.'
            )


def get_path(hatchpattern, density=6):
    """
    Given a hatch specifier, *hatchpattern*, generates Path to render
    the hatch in a unit square.  *density* is the number of lines per
    unit square.
    """
    density = int(density)

    patterns = [hatch_type(hatchpattern, density)
                for hatch_type in _hatch_types]
    num_vertices = sum([pattern.num_vertices for pattern in patterns])

    if num_vertices == 0:
        return Path(np.empty((0, 2)))

    vertices = np.empty((num_vertices, 2))
    codes = np.empty(num_vertices, Path.code_type)

    cursor = 0
    for pattern in patterns:
        if pattern.num_vertices != 0:
            vertices_chunk = vertices[cursor:cursor + pattern.num_vertices]
            codes_chunk = codes[cursor:cursor + pattern.num_vertices]
            pattern.set_vertices_and_codes(vertices_chunk, codes_chunk)
            cursor += pattern.num_vertices

    return Path(vertices, codes)
<|endoftext|>"""
A layoutgrid is a nrows by ncols set of boxes, meant to be used by
`._constrained_layout`, each box is analogous to a subplotspec element of
a gridspec.

Each box is defined by left[ncols], right[ncols], bottom[nrows] and top[nrows],
and by two editable margins for each side.  The main margin gets its value
set by the size of ticklabels, titles, etc on each Axes that is in the figure.
The outer margin is the padding around the Axes, and space for any
colorbars.

The "inner" widths and heights of these boxes are then constrained to be the
same (relative the values of `width_ratios[ncols]` and `height_ratios[nrows]`).

The layoutgrid is then constrained to be contained within a parent layoutgrid,
its column(s) and row(s) specified when it is created.
"""

import itertools
import kiwisolver as kiwi
import logging
import numpy as np

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

_log = logging.getLogger(__name__)


class LayoutGrid:
    """
    Analogous to a gridspec, and contained in another LayoutGrid.
    """

    def __init__(self, parent=None, parent_pos=(0, 0),
                 parent_inner=False, name='', ncols=1, nrows=1,
                 h_pad=None, w_pad=None, width_ratios=None,
                 height_ratios=None):
        Variable = kiwi.Variable
        self.parent_pos = parent_pos
        self.parent_inner = parent_inner
        self.name = name + seq_id()
        if isinstance(parent, LayoutGrid):
            self.name = f'{parent.name}.{self.name}'
        self.nrows = nrows
        self.ncols = ncols
        self.height_ratios = np.atleast_1d(height_ratios)
        if height_ratios is None:
            self.height_ratios = np.ones(nrows)
        self.width_ratios = np.atleast_1d(width_ratios)
        if width_ratios is None:
            self.width_ratios = np.ones(ncols)

        sn = self.name + '_'
        if not isinstance(parent, LayoutGrid):
            # parent can be a rect if not a LayoutGrid
            # allows specifying a rectangle to contain the layout.
            self.solver = kiwi.Solver()
        else:
            parent.add_child(self, *parent_pos)
            self.solver = parent.solver
        # keep track of artist associated w/ this layout.  Can be none
        self.artists = np.empty((nrows, ncols), dtype=object)
        self.children = np.empty((nrows, ncols), dtype=object)

        self.margins = {}
        self.margin_vals = {}
        # all the boxes in each column share the same left/right margins:
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            # track the value so we can change only if a margin is larger
            # than the current value
            self.margin_vals[todo] = np.zeros(ncols)

        sol = self.solver

        self.lefts = [Variable(f'{sn}lefts[{i}]') for i in range(ncols)]
        self.rights = [Variable(f'{sn}rights[{i}]') for i in range(ncols)]
        for todo in ['left', 'right', 'leftcb', 'rightcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(ncols)]
            for i in range(ncols):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = np.empty((nrows), dtype=object)
            self.margin_vals[todo] = np.zeros(nrows)

        self.bottoms = [Variable(f'{sn}bottoms[{i}]') for i in range(nrows)]
        self.tops = [Variable(f'{sn}tops[{i}]') for i in range(nrows)]
        for todo in ['bottom', 'top', 'bottomcb', 'topcb']:
            self.margins[todo] = [Variable(f'{sn}margins[{todo}][{i}]')
                                  for i in range(nrows)]
            for i in range(nrows):
                sol.addEditVariable(self.margins[todo][i], 'strong')

        # set these margins to zero by default. They will be edited as
        # children are filled.
        self.reset_margins()
        self.add_constraints(parent)

        self.h_pad = h_pad
        self.w_pad = w_pad

    def __repr__(self):
        str = f'LayoutBox: {self.name:25s} {self.nrows}x{self.ncols},\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                str += f'{i}, {j}: '\
                       f'L{self.lefts[j].value():1.3f}, ' \
                       f'B{self.bottoms[i].value():1.3f}, ' \
                       f'R{self.rights[j].value():1.3f}, ' \
                       f'T{self.tops[i].value():1.3f}, ' \
                       f'ML{self.margins["left"][j].value():1.3f}, ' \
                       f'MR{self.margins["right"][j].value():1.3f}, ' \
                       f'MB{self.margins["bottom"][i].value():1.3f}, ' \
                       f'MT{self.margins["top"][i].value():1.3f}, \n'
        return str

    def reset_margins(self):
        """
        Reset all the margins to zero.  Must do this after changing
        figure size, for instance, because the relative size of the
        axes labels etc changes.
        """
        for todo in ['left', 'right', 'bottom', 'top',
                     'leftcb', 'rightcb', 'bottomcb', 'topcb']:
            self.edit_margins(todo, 0.0)

    def add_constraints(self, parent):
        # define self-consistent constraints
        self.hard_constraints()
        # define relationship with parent layoutgrid:
        self.parent_constraints(parent)
        # define relative widths of the grid cells to each other
        # and stack horizontally and vertically.
        self.grid_constraints()

    def hard_constraints(self):
        """
        These are the redundant constraints, plus ones that make the
        rest of the code easier.
        """
        for i in range(self.ncols):
            hc = [self.rights[i] >= self.lefts[i],
                  (self.rights[i] - self.margins['right'][i] -
                    self.margins['rightcb'][i] >=
                    self.lefts[i] - self.margins['left'][i] -
                    self.margins['leftcb'][i])
                  ]
            for c in hc:
                self.solver.addConstraint(c | 'required')

        for i in range(self.nrows):
            hc = [self.tops[i] >= self.bottoms[i],
                  (self.tops[i] - self.margins['top'][i] -
                    self.margins['topcb'][i] >=
                    self.bottoms[i] - self.margins['bottom'][i] -
                    self.margins['bottomcb'][i])
                  ]
            for c in hc:
                self.solver.addConstraint(c | 'required')

    def add_child(self, child, i=0, j=0):
        # np.ix_ returns the cross product of i and j indices
        self.children[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] = child

    def parent_constraints(self, parent):
        # constraints that are due to the parent...
        # i.e. the first column's left is equal to the
        # parent's left, the last column right equal to the
        # parent's right...
        if not isinstance(parent, LayoutGrid):
            # specify a rectangle in figure coordinates
            hc = [self.lefts[0] == parent[0],
                  self.rights[-1] == parent[0] + parent[2],
                  # top and bottom reversed order...
                  self.tops[0] == parent[1] + parent[3],
                  self.bottoms[-1] == parent[1]]
        else:
            rows, cols = self.parent_pos
            rows = np.atleast_1d(rows)
            cols = np.atleast_1d(cols)

            left = parent.lefts[cols[0]]
            right = parent.rights[cols[-1]]
            top = parent.tops[rows[0]]
            bottom = parent.bottoms[rows[-1]]
            if self.parent_inner:
                # the layout grid is contained inside the inner
                # grid of the parent.
                left += parent.margins['left'][cols[0]]
                left += parent.margins['leftcb'][cols[0]]
                right -= parent.margins['right'][cols[-1]]
                right -= parent.margins['rightcb'][cols[-1]]
                top -= parent.margins['top'][rows[0]]
                top -= parent.margins['topcb'][rows[0]]
                bottom += parent.margins['bottom'][rows[-1]]
                bottom += parent.margins['bottomcb'][rows[-1]]
            hc = [self.lefts[0] == left,
                  self.rights[-1] == right,
                  # from top to bottom
                  self.tops[0] == top,
                  self.bottoms[-1] == bottom]
        for c in hc:
            self.solver.addConstraint(c | 'required')

    def grid_constraints(self):
        # constrain the ratio of the inner part of the grids
        # to be the same (relative to width_ratios)

        # constrain widths:
        w = (self.rights[0] - self.margins['right'][0] -
             self.margins['rightcb'][0])
        w = (w - self.lefts[0] - self.margins['left'][0] -
             self.margins['leftcb'][0])
        w0 = w / self.width_ratios[0]
        # from left to right
        for i in range(1, self.ncols):
            w = (self.rights[i] - self.margins['right'][i] -
                 self.margins['rightcb'][i])
            w = (w - self.lefts[i] - self.margins['left'][i] -
                 self.margins['leftcb'][i])
            c = (w == w0 * self.width_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly next to each other.
            c = (self.rights[i - 1] == self.lefts[i])
            self.solver.addConstraint(c | 'strong')

        # constrain heights:
        h = self.tops[0] - self.margins['top'][0] - self.margins['topcb'][0]
        h = (h - self.bottoms[0] - self.margins['bottom'][0] -
             self.margins['bottomcb'][0])
        h0 = h / self.height_ratios[0]
        # from top to bottom:
        for i in range(1, self.nrows):
            h = (self.tops[i] - self.margins['top'][i] -
                 self.margins['topcb'][i])
            h = (h - self.bottoms[i] - self.margins['bottom'][i] -
                 self.margins['bottomcb'][i])
            c = (h == h0 * self.height_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly above each other.
            c = (self.bottoms[i - 1] == self.tops[i])
            self.solver.addConstraint(c | 'strong')

    # Margin editing:  The margins are variable and meant to
    # contain things of a fixed size like axes labels, tick labels, titles
    # etc
    def edit_margin(self, todo, size, cell):
        """
        Change the size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        size : float
            Size of the margin.  If it is larger than the existing minimum it
            updates the margin size. Fraction of figure size.

        cell : int
            Cell column or row to edit.
        """
        self.solver.suggestValue(self.margins[todo][cell], size)
        self.margin_vals[todo][cell] = size

    def edit_margin_min(self, todo, size, cell=0):
        """
        Change the minimum size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        size : float
            Minimum size of the margin .  If it is larger than the
            existing minimum it updates the margin size. Fraction of
            figure size.

        cell : int
            Cell column or row to edit.
        """

        if size > self.margin_vals[todo][cell]:
            self.edit_margin(todo, size, cell)

    def edit_margins(self, todo, size):
        """
        Change the size of all the margin of all the cells in the layout grid.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        size : float
            Size to set the margins.  Fraction of figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            self.edit_margin(todo, size, i)

    def edit_all_margins_min(self, todo, size):
        """
        Change the minimum size of all the margin of all
        the cells in the layout grid.

        Parameters
        ----------
        todo : {'left', 'right', 'bottom', 'top'}
            The margin to alter.

        size : float
            Minimum size of the margin.  If it is larger than the
            existing minimum it updates the margin size. Fraction of
            figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            self.edit_margin_min(todo, size, i)

    def edit_outer_margin_mins(self, margin, ss):
        """
        Edit all four margin minimums in one statement.

        Parameters
        ----------
        margin : dict
            size of margins in a dict with keys 'left', 'right', 'bottom',
            'top'

        ss : SubplotSpec
            defines the subplotspec these margins should be applied to
        """

        self.edit_margin_min('left', margin['left'], ss.colspan.start)
        self.edit_margin_min('leftcb', margin['leftcb'], ss.colspan.start)
        self.edit_margin_min('right', margin['right'], ss.colspan.stop - 1)
        self.edit_margin_min('rightcb', margin['rightcb'], ss.colspan.stop - 1)
        # rows are from the top down:
        self.edit_margin_min('top', margin['top'], ss.rowspan.start)
        self.edit_margin_min('topcb', margin['topcb'], ss.rowspan.start)
        self.edit_margin_min('bottom', margin['bottom'], ss.rowspan.stop - 1)
        self.edit_margin_min('bottomcb', margin['bottomcb'],
                             ss.rowspan.stop - 1)

    def get_margins(self, todo, col):
        """Return the margin at this position"""
        return self.margin_vals[todo][col]

    def get_outer_bbox(self, rows=0, cols=0):
        """
        Return the outer bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            self.lefts[cols[0]].value(),
            self.bottoms[rows[-1]].value(),
            self.rights[cols[-1]].value(),
            self.tops[rows[0]].value())
        return bbox

    def get_inner_bbox(self, rows=0, cols=0):
        """
        Return the inner bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['left'][cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value() +
                self.margins['bottomcb'][rows[-1]].value()),
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            (self.tops[rows[0]].value() -
                self.margins['top'][rows[0]].value() -
                self.margins['topcb'][rows[0]].value())
        )
        return bbox

    def get_bbox_for_cb(self, rows=0, cols=0):
        """
        Return the bounding box that includes the
        decorations but, *not* the colorbar...
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),
            (self.bottoms[rows[-1]].value() +
                self.margins['bottomcb'][rows[-1]].value()),
            (self.rights[cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value())
        )
        return bbox

    def get_left_margin_bbox(self, rows=0, cols=0):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value()),
            (self.bottoms[rows[-1]].value()),
            (self.lefts[cols[0]].value() +
                self.margins['leftcb'][cols[0]].value() +
                self.margins['left'][cols[0]].value()),
            (self.tops[rows[0]].value()))
        return bbox

    def get_bottom_margin_bbox(self, rows=0, cols=0):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value()),
            (self.bottoms[rows[-1]].value() +
             self.margins['bottomcb'][rows[-1]].value()),
            (self.rights[cols[-1]].value()),
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value() +
             self.margins['bottomcb'][rows[-1]].value()
             ))
        return bbox

    def get_right_margin_bbox(self, rows=0, cols=0):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            (self.bottoms[rows[-1]].value()),
            (self.rights[cols[-1]].value() -
                self.margins['rightcb'][cols[-1]].value()),
            (self.tops[rows[0]].value()))
        return bbox

    def get_top_margin_bbox(self, rows=0, cols=0):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value()),
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value()),
            (self.rights[cols[-1]].value()),
            (self.tops[rows[0]].value() -
                self.margins['topcb'][rows[0]].value() -
                self.margins['top'][rows[0]].value()))
        return bbox

    def update_variables(self):
        """
        Update the variables for the solver attached to this layoutgrid.
        """
        self.solver.updateVariables()

_layoutboxobjnum = itertools.count()


def seq_id():
    """Generate a short sequential id for layoutbox objects."""
    return '%06d' % next(_layoutboxobjnum)


def plot_children(fig, lg=None, level=0):
    """Simple plotting to show where boxes are."""
    if lg is None:
        _layoutgrids = fig.get_layout_engine().execute(fig)
        lg = _layoutgrids[fig]
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    col = colors[level]
    for i in range(lg.nrows):
        for j in range(lg.ncols):
            bb = lg.get_outer_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=1,
                                   edgecolor='0.7', facecolor='0.7',
                                   alpha=0.2, transform=fig.transFigure,
                                   zorder=-3))
            bbi = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=2,
                                   edgecolor=col, facecolor='none',
                                   transform=fig.transFigure, zorder=-2))

            bbi = lg.get_left_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.7, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_right_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.5, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_bottom_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.5, 0.7],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_top_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.2, 0.7],
                                   transform=fig.transFigure, zorder=-2))
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level+1)
<|endoftext|>"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which internally use `.Colorbar` together with
   `.make_axes_gridspec` (for `.GridSpec`-positioned Axes) or `.make_axes` (for
   non-`.GridSpec`-positioned Axes).

   End-users most likely won't need to directly use this module's API.
"""

import logging

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring

_log = logging.getLogger(__name__)

_docstring.interpd.register(
    _make_axes_kw_doc="""
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent Axes, where the colorbar Axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.

orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.

fraction : float, default: 0.15
    Fraction of original Axes to use for colorbar.

shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.

aspect : float, default: 20
    Ratio of long to short dimensions.

pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original Axes between colorbar and new image Axes.

anchor : (float, float), optional
    The anchor point of the colorbar Axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.

panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent Axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.""",
    _colormap_kw_doc="""
extend : {'neither', 'both', 'min', 'max'}
    Make pointed end(s) for out-of-range values (unless 'neither').  These are
    set for a given colormap using the colormap set_under and set_over methods.

extendfrac : {*None*, 'auto', length, lengths}
    If set to *None*, both the minimum and maximum triangular colorbar
    extensions will have a length of 5% of the interior colorbar length (this
    is the default setting).

    If set to 'auto', makes the triangular colorbar extensions the same lengths
    as the interior boxes (when *spacing* is set to 'uniform') or the same
    lengths as the respective adjacent interior boxes (when *spacing* is set to
    'proportional').

    If a scalar, indicates the length of both the minimum and maximum
    triangular colorbar extensions as a fraction of the interior colorbar
    length.  A two-element sequence of fractions may also be given, indicating
    the lengths of the minimum and maximum colorbar extensions respectively as
    a fraction of the interior colorbar length.

extendrect : bool
    If *False* the minimum and maximum colorbar extensions will be triangular
    (the default).  If *True* the extensions will be rectangular.

ticks : None or list of ticks or Locator
    If None, ticks are determined automatically from the input.

format : None or str or Formatter
    If None, `~.ticker.ScalarFormatter` is used.
    Format strings, e.g., ``"%4.2e"`` or ``"{x:.2e}"``, are supported.
    An alternative `~.ticker.Formatter` may be given instead.

drawedges : bool
    Whether to draw lines at color boundaries.

label : str
    The label on the colorbar's long axis.

boundaries, values : None or a sequence
    If unset, the colormap will be displayed on a 0-1 scale.
    If sequences, *values* must have a length 1 less than *boundaries*.  For
    each region delimited by adjacent entries in *boundaries*, the color mapped
    to the corresponding value in *values* will be used.  The size of each
    region is determined by the *spacing* parameter.
    Normally only useful for indexed colors (i.e. ``norm=NoNorm()``) or other
    unusual circumstances.

spacing : {'uniform', 'proportional'}
    For discrete colorbars (`.BoundaryNorm` or contours), 'uniform' gives each
    color the same space; 'proportional' makes the space proportional to the
    data interval.""")


def _set_ticks_on_axis_warn(*args, **kwargs):
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by Colorbar.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")


class _ColorbarSpine(mspines.Spine):
    def __init__(self, axes):
        self._ax = axes
        super().__init__(axes, 'colorbar', mpath.Path(np.empty((0, 2))))
        mpatches.Patch.set_transform(self, axes.transAxes)

    def get_window_extent(self, renderer=None):
        # This Spine has no Axis associated with it, and doesn't need to adjust
        # its location, so we can directly get the window extent from the
        # super-super-class.
        return mpatches.Patch.get_window_extent(self, renderer=renderer)

    def set_xy(self, xy):
        self._path = mpath.Path(xy, closed=True)
        self._xy = xy
        self.stale = True

    def draw(self, renderer):
        ret = mpatches.Patch.draw(self, renderer)
        self.stale = False
        return ret


class _ColorbarAxesLocator:
    """
    Shrink the Axes if there are triangular or rectangular extends.
    """
    def __init__(self, cbar):
        self._cbar = cbar
        self._orig_locator = cbar.ax._axes_locator

    def __call__(self, ax, renderer):
        if self._orig_locator is not None:
            pos = self._orig_locator(ax, renderer)
        else:
            pos = ax.get_position(original=True)
        if self._cbar.extend == 'neither':
            return pos

        y, extendlen = self._cbar._proportional_y()
        if not self._cbar._extend_lower():
            extendlen[0] = 0
        if not self._cbar._extend_upper():
            extendlen[1] = 0
        len = sum(extendlen) + 1
        shrink = 1 / len
        offset = extendlen[0] / len
        # we need to reset the aspect ratio of the axes to account
        # of the extends...
        if hasattr(ax, '_colorbar_info'):
            aspect = ax._colorbar_info['aspect']
        else:
            aspect = False
        # now shrink and/or offset to take into account the
        # extend tri/rectangles.
        if self._cbar.orientation == 'vertical':
            if aspect:
                self._cbar.ax.set_box_aspect(aspect*shrink)
            pos = pos.shrunk(1, shrink).translated(0, offset * pos.height)
        else:
            if aspect:
                self._cbar.ax.set_box_aspect(1/(aspect * shrink))
            pos = pos.shrunk(shrink, 1).translated(offset * pos.width, 0)
        return pos

    def get_subplotspec(self):
        # make tight_layout happy..
        return (
            self._cbar.ax.get_subplotspec()
            or getattr(self._orig_locator, "get_subplotspec", lambda: None)())


@_docstring.interpd
class Colorbar:
    r"""
    Draw a colorbar in an existing Axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize

    def __init__(
        self, ax, mappable=None, *,
        alpha=None,
        location=None,
        extend=None,
        extendfrac=None,
        extendrect=False,
        ticks=None,
        format=None,
        values=None,
        boundaries=None,
        spacing='uniform',
        drawedges=False,
        label='',
        cmap=None, norm=None,  # redundant with *mappable*
        orientation=None, ticklocation='auto',  # redundant with *location*
    ):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance in which the colorbar is drawn.

        mappable : `.ScalarMappable`
            The mappable whose colormap and norm will be used.

            To show the colors versus index instead of on a 0-1 scale, set the
            mappable's norm to ``colors.NoNorm()``.

        alpha : float
            The colorbar transparency between 0 (transparent) and 1 (opaque).

        location : None or {'left', 'right', 'top', 'bottom'}
            Set the colorbar's *orientation* and *ticklocation*. Colorbars on
            the left and right are vertical, colorbars at the top and bottom
            are horizontal. The *ticklocation* is the same as *location*, so if
            *location* is 'top', the ticks are on the top. *orientation* and/or
            *ticklocation* can be provided as well and overrides the value set by
            *location*, but there will be an error for incompatible combinations.

            .. versionadded:: 3.7

        %(_colormap_kw_doc)s

        Other Parameters
        ----------------
        cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The colormap to use.  This parameter is ignored, unless *mappable* is
            None.

        norm : `~matplotlib.colors.Normalize`
            The normalization to use.  This parameter is ignored, unless *mappable*
            is None.

        orientation : None or {'vertical', 'horizontal'}
            If None, use the value determined by *location*. If both
            *orientation* and *location* are None then defaults to 'vertical'.

        ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}
            The location of the colorbar ticks. The *ticklocation* must match
            *orientation*. For example, a horizontal colorbar can only have ticks
            at the top or the bottom. If 'auto', the ticks will be the same as
            *location*, so a colorbar to the left will have ticks to the left. If
            *location* is None, the ticks will be at the bottom for a horizontal
            colorbar and at the right for a vertical.
        """
        if mappable is None:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        self.mappable = mappable
        cmap = mappable.cmap
        norm = mappable.norm

        filled = True
        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            alpha = cs.get_alpha()
            boundaries = cs._levels
            values = cs.cvalues
            extend = cs.extend
            filled = cs.filled
            if ticks is None:
                ticks = ticker.FixedLocator(cs.levels, nbins=10)
        elif isinstance(mappable, martist.Artist):
            alpha = mappable.get_alpha()

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

        location_orientation = _get_orientation_from_location(location)

        _api.check_in_list(
            [None, 'vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        if location_orientation is not None and orientation is not None:
            if location_orientation != orientation:
                raise TypeError(
                    "location and orientation are mutually exclusive")
        else:
            orientation = orientation or location_orientation or "vertical"

        self.ax = ax
        self.ax._axes_locator = _ColorbarAxesLocator(self)

        if extend is None:
            if (not isinstance(mappable, contour.ContourSet)
                    and getattr(cmap, 'colorbar_extend', False) is not False):
                extend = cmap.colorbar_extend
            elif hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = None
        # Call set_alpha to handle array-like alphas properly
        self.set_alpha(alpha)
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self._filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self._extend_patches = []
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']],
            clip_on=False)
        self.ax.add_collection(self.dividers)

        self._locator = None
        self._minorlocator = None
        self._formatter = None
        self._minorformatter = None

        if ticklocation == 'auto':
            ticklocation = _get_ticklocation_from_orientation(
                orientation) if location is None else location
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self._locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self._locator = ticks

        if isinstance(format, str):
            # Check format between FormatStrFormatter and StrMethodFormatter
            try:
                self._formatter = ticker.FormatStrFormatter(format)
                _ = self._formatter(0)
            except (TypeError, ValueError):
                self._formatter = ticker.StrMethodFormatter(format)
        else:
            self._formatter = format  # Assume it is a Formatter or None
        self._draw_all()

        if isinstance(mappable, contour.ContourSet) and not mappable.filled:
            self.add_lines(mappable)

        # Link the Axes and Colorbar for interactive use
        self.ax._colorbar = self
        # Don't navigate on any of these types of mappables
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm)) or
                isinstance(self.mappable, contour.ContourSet)):
            self.ax.set_navigate(False)

        # These are the functions that set up interactivity on this colorbar
        self._interactive_funcs = ["_get_view", "_set_view",
                                   "_set_view_from_bbox", "drag_pan"]
        for x in self._interactive_funcs:
            setattr(self.ax, x, getattr(self, x))
        # Set the cla function to the cbar's method to override it
        self.ax.cla = self._cbar_cla
        # Callbacks for the extend calculations to handle inverting the axis
        self._extend_cid1 = self.ax.callbacks.connect(
            "xlim_changed", self._do_extends)
        self._extend_cid2 = self.ax.callbacks.connect(
            "ylim_changed", self._do_extends)

    @property
    def long_axis(self):
        """Axis that has decorations (ticks, etc) on it."""
        if self.orientation == 'vertical':
            return self.ax.yaxis
        return self.ax.xaxis

    @property
    def locator(self):
        """Major tick `.Locator` for the colorbar."""
        return self.long_axis.get_major_locator()

    @locator.setter
    def locator(self, loc):
        self.long_axis.set_major_locator(loc)
        self._locator = loc

    @property
    def minorlocator(self):
        """Minor tick `.Locator` for the colorbar."""
        return self.long_axis.get_minor_locator()

    @minorlocator.setter
    def minorlocator(self, loc):
        self.long_axis.set_minor_locator(loc)
        self._minorlocator = loc

    @property
    def formatter(self):
        """Major tick label `.Formatter` for the colorbar."""
        return self.long_axis.get_major_formatter()

    @formatter.setter
    def formatter(self, fmt):
        self.long_axis.set_major_formatter(fmt)
        self._formatter = fmt

    @property
    def minorformatter(self):
        """Minor tick `.Formatter` for the colorbar."""
        return self.long_axis.get_minor_formatter()

    @minorformatter.setter
    def minorformatter(self, fmt):
        self.long_axis.set_minor_formatter(fmt)
        self._minorformatter = fmt

    def _cbar_cla(self):
        """Function to clear the interactive colorbar state."""
        for x in self._interactive_funcs:
            delattr(self.ax, x)
        # We now restore the old cla() back and can call it directly
        del self.ax.cla
        self.ax.cla()

    def update_normal(self, mappable=None):
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
        if mappable:
            # The mappable keyword argument exists because
            # ScalarMappable.changed() emits self.callbacks.process('changed', self)
            # in contrast, ColorizingArtist (and Colorizer) does not use this keyword.
            # [ColorizingArtist.changed() emits self.callbacks.process('changed')]
            # Also, there is no test where self.mappable == mappable is not True
            # and possibly no use case.
            # Therefore, the mappable keyword can be deprecated if cm.ScalarMappable
            # is removed.
            self.mappable = mappable
        _log.debug('colorbar update normal %r %r', self.mappable.norm, self.norm)
        self.set_alpha(self.mappable.get_alpha())
        self.cmap = self.mappable.cmap
        if self.mappable.norm != self.norm:
            self.norm = self.mappable.norm
            self._reset_locator_formatter_scale()

        self._draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
        self.stale = True

    def _draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        if self.orientation == 'vertical':
            if mpl.rcParams['ytick.minor.visible']:
                self.minorticks_on()
        else:
            if mpl.rcParams['xtick.minor.visible']:
                self.minorticks_on()
        self.long_axis.set(label_position=self.ticklocation,
                              ticks_position=self.ticklocation)
        self._short_axis().set_ticks([])
        self._short_axis().set_ticks([], minor=True)

        # Set self._boundaries and self._values, including extensions.
        # self._boundaries are the edges of each square of color, and
        # self._values are the value to map into the norm to get the
        # color:
        self._process_values()
        # Set self.vmin and self.vmax to first and last boundary, excluding
        # extensions:
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
        # Compute the X/Y mesh.
        X, Y = self._mesh()
        # draw the extend triangles, and shrink the inner Axes to accommodate.
        # also adds the outline path to self.outline spine:
        self._do_extends()
        lower, upper = self.vmin, self.vmax
        if self.long_axis.get_inverted():
            # If the axis is inverted, we need to swap the vmin/vmax
            lower, upper = upper, lower
        if self.orientation == 'vertical':
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(lower, upper)

        # set up the tick locators and formatters.  A bit complicated because
        # boundary norms + uniform spacing requires a manual locator.
        self.update_ticks()

        if self._filled:
            ind = np.arange(len(self._values))
            if self._extend_lower():
                ind = ind[1:]
            if self._extend_upper():
                ind = ind[:-1]
            self._add_solids(X, Y, self._values[ind, np.newaxis])

    def _add_solids(self, X, Y, C):
        """Draw the colors; optionally add separators."""
        # Cleanup previously set artists.
        if self.solids is not None:
            self.solids.remove()
        for solid in self.solids_patches:
            solid.remove()
        # Add new artist(s), based on mappable type.  Use individual patches if
        # hatching is needed, pcolormesh otherwise.
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            self._add_solids_patches(X, Y, C, mappable)
        else:
            self.solids = self.ax.pcolormesh(
                X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha,
                edgecolors='none', shading='flat')
            if not self.drawedges:
                if len(self._y) >= self.n_rasterize:
                    self.solids.set_rasterized(True)
        self._update_dividers()

    def _update_dividers(self):
        if not self.drawedges:
            self.dividers.set_segments([])
            return
        # Place all *internal* dividers.
        if self.orientation == 'vertical':
            lims = self.ax.get_ylim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        else:
            lims = self.ax.get_xlim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        y = self._y[bounds]
        # And then add outer dividers if extensions are on.
        if self._extend_lower():
            y = np.insert(y, 0, lims[0])
        if self._extend_upper():
            y = np.append(y, lims[1])
        X, Y = np.meshgrid([0, 1], y)
        if self.orientation == 'vertical':
            segments = np.dstack([X, Y])
        else:
            segments = np.dstack([Y, X])
        self.dividers.set_segments(segments)

    def _add_solids_patches(self, X, Y, C, mappable):
        hatches = mappable.hatches * (len(C) + 1)  # Have enough hatches.
        if self._extend_lower():
            # remove first hatch that goes into the extend patch
            hatches = hatches[1:]
        patches = []
        for i in range(len(X) - 1):
            xy = np.array([[X[i, 0], Y[i, 1]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        self.solids_patches = patches

    def _do_extends(self, ax=None):
        """
        Add the extend tri/rectangles on the outside of the Axes.

        ax is unused, but required due to the callbacks on xlim/ylim changed
        """
        # Clean up any previous extend patches
        for patch in self._extend_patches:
            patch.remove()
        self._extend_patches = []
        # extend lengths are fraction of the *inner* part of colorbar,
        # not the total colorbar:
        _, extendlen = self._proportional_y()
        bot = 0 - (extendlen[0] if self._extend_lower() else 0)
        top = 1 + (extendlen[1] if self._extend_upper() else 0)

        # xyout is the outline of the colorbar including the extend patches:
        if not self.extendrect:
            # triangle:
            xyout = np.array([[0, 0], [0.5, bot], [1, 0],
                              [1, 1], [0.5, top], [0, 1], [0, 0]])
        else:
            # rectangle:
            xyout = np.array([[0, 0], [0, bot], [1, bot], [1, 0],
                              [1, 1], [1, top], [0, top], [0, 1],
                              [0, 0]])

        if self.orientation == 'horizontal':
            xyout = xyout[:, ::-1]

        # xyout is the path for the spine:
        self.outline.set_xy(xyout)
        if not self._filled:
            return

        # Make extend triangles or rectangles filled patches.  These are
        # defined in the outer parent axes' coordinates:
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            hatches = mappable.hatches * (len(self._y) + 1)
        else:
            hatches = [None] * (len(self._y) + 1)

        if self._extend_lower():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 0], [0.5, bot], [1, 0]])
            else:
                # rectangle
                xy = np.array([[0, 0], [0, bot], [1., bot], [1, 0]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            val = -1 if self.long_axis.get_inverted() else 0
            color = self.cmap(self.norm(self._values[val]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, alpha=self.alpha,
                linewidth=0, antialiased=False,
                transform=self.ax.transAxes,
                hatch=hatches[0], clip_on=False,
                # Place it right behind the standard patches, which is
                # needed if we updated the extends
                zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
            self.ax.add_patch(patch)
            self._extend_patches.append(patch)
            # remove first hatch that goes into the extend patch
            hatches = hatches[1:]
        if self._extend_upper():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 1], [0.5, top], [1, 1]])
            else:
                # rectangle
                xy = np.array([[0, 1], [0, top], [1, top], [1, 1]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            val = 0 if self.long_axis.get_inverted() else -1
            color = self.cmap(self.norm(self._values[val]))
            hatch_idx = len(self._y) - 1
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, alpha=self.alpha,
                linewidth=0, antialiased=False,
                transform=self.ax.transAxes, hatch=hatches[hatch_idx],
                clip_on=False,
                # Place it right behind the standard patches, which is
                # needed if we updated the extends
                zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
            self.ax.add_patch(patch)
            self._extend_patches.append(patch)

        self._update_dividers()

    def add_lines(self, *args, **kwargs):
        """
        Draw lines on the colorbar.

        The lines are appended to the list :attr:`lines`.

        Parameters
        ----------
        levels : array-like
            The positions of the lines.
        colors : :mpltype:`color` or list of :mpltype:`color`
            Either a single color applying to all lines or one color value for
            each line.
        linewidths : float or array-like
            Either a single linewidth applying to all lines or one linewidth
            for each line.
        erase : bool, default: True
            Whether to remove any previously added lines.

        Notes
        -----
        Alternatively, this method can also be called with the signature
        ``colorbar.add_lines(contour_set, erase=True)``, in which case
        *levels*, *colors*, and *linewidths* are taken from *contour_set*.
        """
        params = _api.select_matching_signature(
            [lambda self, CS, erase=True: locals(),
             lambda self, levels, colors, linewidths, erase=True: locals()],
            self, *args, **kwargs)
        if "CS" in params:
            self, cs, erase = params.values()
            if not isinstance(cs, contour.ContourSet) or cs.filled:
                raise ValueError("If a single artist is passed to add_lines, "
                                 "it must be a ContourSet of lines")
            # TODO: Make colorbar lines auto-follow changes in contour lines.
            return self.add_lines(
                cs.levels,
                cs.to_rgba(cs.cvalues, cs.alpha),
                cs.get_linewidths(),
                erase=erase)
        else:
            self, levels, colors, linewidths, erase = params.values()

        y = self._locate(levels)
        rtol = (self._y[-1] - self._y[0]) * 1e-10
        igood = (y < self._y[-1] + rtol) & (y > self._y[0] - rtol)
        y = y[igood]
        if np.iterable(colors):
            colors = np.asarray(colors)[igood]
        if np.iterable(linewidths):
            linewidths = np.asarray(linewidths)[igood]
        X, Y = np.meshgrid([0, 1], y)
        if self.orientation == 'vertical':
            xy = np.stack([X, Y], axis=-1)
        else:
            xy = np.stack([Y, X], axis=-1)
        col = collections.LineCollection(xy, linewidths=linewidths,
                                         colors=colors)

        if erase and self.lines:
            for lc in self.lines:
                lc.remove()
            self.lines = []
        self.lines.append(col)

        # make a clip path that is just a linewidth bigger than the Axes...
        fac = np.max(linewidths) / 72
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        inches = self.ax.get_figure().dpi_scale_trans
        # do in inches:
        xy = inches.inverted().transform(self.ax.transAxes.transform(xy))
        xy[[0, 1, 4], 1] -= fac
        xy[[2, 3], 1] += fac
        # back to axes units...
        xy = self.ax.transAxes.inverted().transform(inches.transform(xy))
        col.set_clip_path(mpath.Path(xy, closed=True),
                          self.ax.transAxes)
        self.ax.add_collection(col)
        self.stale = True

    def update_ticks(self):
        """
        Set up the ticks and ticklabels. This should not be needed by users.
        """
        # Get the locator and formatter; defaults to self._locator if not None.
        self._get_ticker_locator_formatter()
        self.long_axis.set_major_locator(self._locator)
        self.long_axis.set_minor_locator(self._minorlocator)
        self.long_axis.set_major_formatter(self._formatter)

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
        locator = self._locator
        formatter = self._formatter
        minorlocator = self._minorlocator
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
            if minorlocator is None:
                minorlocator = ticker.FixedLocator(b)
        elif isinstance(self.norm, colors.NoNorm):
            if locator is None:
                # put ticks on integers between the boundaries of NoNorm
                nv = len(self._values)
                base = 1 + int(nv / 10)
                locator = ticker.IndexLocator(base=base, offset=.5)
        elif self.boundaries is not None:
            b = self._boundaries[self._inside]
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        else:  # most cases:
            if locator is None:
                # we haven't set the locator explicitly, so use the default
                # for this axis:
                locator = self.long_axis.get_major_locator()
            if minorlocator is None:
                minorlocator = self.long_axis.get_minor_locator()

        if minorlocator is None:
            minorlocator = ticker.NullLocator()

        if formatter is None:
            formatter = self.long_axis.get_major_formatter()

        self._locator = locator
        self._formatter = formatter
        self._minorlocator = minorlocator
        _log.debug('locator: %r', locator)

    def set_ticks(self, ticks, *, labels=None, minor=False, **kwargs):
        """
        Set tick locations.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.
        """
        if np.iterable(ticks):
            self.long_axis.set_ticks(ticks, labels=labels, minor=minor,
                                        **kwargs)
            self._locator = self.long_axis.get_major_locator()
        else:
            self._locator = ticks
            self.long_axis.set_major_locator(self._locator)
        self.stale = True

    def get_ticks(self, minor=False):
        """
        Return the ticks as a list of locations.

        Parameters
        ----------
        minor : boolean, default: False
            if True return the minor ticks.
        """
        if minor:
            return self.long_axis.get_minorticklocs()
        else:
            return self.long_axis.get_majorticklocs()

    def set_ticklabels(self, ticklabels, *, minor=False, **kwargs):
        """
        [*Discouraged*] Set tick labels.

        .. admonition:: Discouraged

            The use of this method is discouraged, because of the dependency
            on tick positions. In most cases, you'll want to use
            ``set_ticks(positions, labels=labels)`` instead.

            If you are using this method, you should always fix the tick
            positions before, e.g. by using `.Colorbar.set_ticks` or by
            explicitly setting a `~.ticker.FixedLocator` on the long axis
            of the colorbar. Otherwise, ticks are free to move and the
            labels may end up in unexpected positions.

        Parameters
        ----------
        ticklabels : sequence of str or of `.Text`
            Texts for labeling each tick location in the sequence set by
            `.Colorbar.set_ticks`; the number of labels must match the number
            of locations.

        update_ticks : bool, default: True
            This keyword argument is ignored and will be removed.
            Deprecated

        minor : bool
            If True, set minor ticks instead of major ticks.

        **kwargs
            `.Text` properties for the labels.
        """
        self.long_axis.set_ticklabels(ticklabels, minor=minor, **kwargs)

    def minorticks_on(self):
        """
        Turn on colorbar minor ticks.
        """
        self.ax.minorticks_on()
        self._short_axis().set_minor_locator(ticker.NullLocator())

    def minorticks_off(self):
        """Turn the minor ticks of the colorbar off."""
        self._minorlocator = ticker.NullLocator()
        self.long_axis.set_minor_locator(self._minorlocator)

    def set_label(self, label, *, loc=None, **kwargs):
        """
        Add a label to the long axis of the colorbar.

        Parameters
        ----------
        label : str
            The label text.
        loc : str, optional
            The location of the label.

            - For horizontal orientation one of {'left', 'center', 'right'}
            - For vertical orientation one of {'bottom', 'center', 'top'}

            Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
            depending on the orientation.
        **kwargs
            Keyword arguments are passed to `~.Axes.set_xlabel` /
            `~.Axes.set_ylabel`.
            Supported keywords are *labelpad* and `.Text` properties.
        """
        if self.orientation == "vertical":
            self.ax.set_ylabel(label, loc=loc, **kwargs)
        else:
            self.ax.set_xlabel(label, loc=loc, **kwargs)
        self.stale = True

    def set_alpha(self, alpha):
        """
        Set the transparency between 0 (transparent) and 1 (opaque).

        If an array is provided, *alpha* will be set to None to use the
        transparency values associated with the colormap.
        """
        self.alpha = None if isinstance(alpha, np.ndarray) else alpha

    def _set_scale(self, scale, **kwargs):
        """
        Set the colorbar long axis scale.

        Parameters
        ----------
        scale : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
            The axis scale type to apply.

        **kwargs
            Different keyword arguments are accepted, depending on the scale.
            See the respective class keyword arguments:

            - `matplotlib.scale.LinearScale`
            - `matplotlib.scale.LogScale`
            - `matplotlib.scale.SymmetricalLogScale`
            - `matplotlib.scale.LogitScale`
            - `matplotlib.scale.FuncScale`
            - `matplotlib.scale.AsinhScale`

        Notes
        -----
        By default, Matplotlib supports the above-mentioned scales.
        Additionally, custom scales may be registered using
        `matplotlib.scale.register_scale`. These scales can then also
        be used here.
        """
        self.long_axis._set_axes_scale(scale, **kwargs)

    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        if hasattr(self.ax, '_colorbar_info'):
            parents = self.ax._colorbar_info['parents']
            for a in parents:
                if self.ax in a._colorbars:
                    a._colorbars.remove(self.ax)

        self.ax.remove()

        self.mappable.callbacks.disconnect(self.mappable.colorbar_cid)
        self.mappable.colorbar = None
        self.mappable.colorbar_cid = None
        # Remove the extension callbacks
        self.ax.callbacks.disconnect(self._extend_cid1)
        self.ax.callbacks.disconnect(self._extend_cid2)

        try:
            ax = self.mappable.axes
        except AttributeError:
            return
        try:
            subplotspec = self.ax.get_subplotspec().get_gridspec()._subplot_spec
        except AttributeError:  # use_gridspec was False
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:  # use_gridspec was True
            ax.set_subplotspec(subplotspec)

    def _process_values(self):
        """
        Set `_boundaries` and `_values` based on the self.boundaries and
        self.values if not None, or based on the size of the colormap and
        the vmin/vmax of the norm.
        """
        if self.values is not None:
            # set self._boundaries from the values...
            self._values = np.array(self.values)
            if self.boundaries is None:
                # bracket values by 1/2 dv:
                b = np.zeros(len(self.values) + 1)
                b[1:-1] = 0.5 * (self._values[:-1] + self._values[1:])
                b[0] = 2.0 * b[1] - b[2]
                b[-1] = 2.0 * b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = np.array(self.boundaries)
            return

        # otherwise values are set from the boundaries
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
        elif isinstance(self.norm, colors.NoNorm):
            # NoNorm has N blocks, so N+1 boundaries, centered on integers:
            b = np.arange(self.cmap.N + 1) - .5
        elif self.boundaries is not None:
            b = self.boundaries
        else:
            # otherwise make the boundaries from the size of the cmap:
            N = self.cmap.N + 1
            b, _ = self._uniform_y(N)
        # add extra boundaries if needed:
        if self._extend_lower():
            b = np.hstack((b[0] - 1, b))
        if self._extend_upper():
            b = np.hstack((b, b[-1] + 1))

        # transform from 0-1 to vmin-vmax:
        if self.mappable.get_array() is not None:
            self.mappable.autoscale_None()
        if not self.norm.scaled():
            # If we still aren't scaled after autoscaling, use 0, 1 as default
            self.norm.vmin = 0
            self.norm.vmax = 1
        self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(
            self.norm.vmin, self.norm.vmax, expander=0.1)
        if (not isinstance(self.norm, colors.BoundaryNorm) and
                (self.boundaries is None)):
            b = self.norm.inverse(b)

        self._boundaries = np.asarray(b, dtype=float)
        self._values = 0.5 * (self._boundaries[:-1] + self._boundaries[1:])
        if isinstance(self.norm, colors.NoNorm):
            self._values = (self._values + 0.00001).astype(np.int16)

    def _mesh(self):
        """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
        y, _ = self._proportional_y()
        # Use the vmin and vmax of the colorbar, which may not be the same
        # as the norm. There are situations where the colormap has a
        # narrower range than the colorbar and we want to accommodate the
        # extra contours.
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm))
                or self.boundaries is not None):
            # not using a norm.
            y = y * (self.vmax - self.vmin) + self.vmin
        else:
            # Update the norm values in a context manager as it is only
            # a temporary change and we don't want to propagate any signals
            # attached to the norm (callbacks.blocked).
            with (self.norm.callbacks.blocked(),
                  cbook._setattr_cm(self.norm, vmin=self.vmin, vmax=self.vmax)):
                y = self.norm.inverse(y)
        self._y = y
        X, Y = np.meshgrid([0., 1.], y)
        if self.orientation == 'vertical':
            return (X, Y)
        else:
            return (Y, X)

    def _forward_boundaries(self, x):
        # map boundaries equally between 0 and 1...
        b = self._boundaries
        y = np.interp(x, b, np.linspace(0, 1, len(b)))
        # the following avoids ticks in the extends:
        eps = (b[-1] - b[0]) * 1e-6
        # map these _well_ out of bounds to keep any ticks out
        # of the extends region...
        y[x < b[0]-eps] = -1
        y[x > b[-1]+eps] = 2
        return y

    def _inverse_boundaries(self, x):
        # invert the above...
        b = self._boundaries
        return np.interp(x, np.linspace(0, 1, len(b)), b)

    def _reset_locator_formatter_scale(self):
        """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
        self._process_values()
        self._locator = None
        self._minorlocator = None
        self._formatter = None
        self._minorformatter = None
        if (isinstance(self.mappable, contour.ContourSet) and
                isinstance(self.norm, colors.LogNorm)):
            # if contours have lognorm, give them a log scale...
            self._set_scale('log')
        elif (self.boundaries is not None or
                isinstance(self.norm, colors.BoundaryNorm)):
            if self.spacing == 'uniform':
                funcs = (self._forward_boundaries, self._inverse_boundaries)
                self._set_scale('function', functions=funcs)
            elif self.spacing == 'proportional':
                self._set_scale('linear')
        elif getattr(self.norm, '_scale', None):
            # use the norm's scale (if it exists and is not None):
            self._set_scale(self.norm._scale)
        elif type(self.norm) is colors.Normalize:
            # plain Normalize:
            self._set_scale('linear')
        else:
            # norm._scale is None or not an attr: derive the scale from
            # the Norm:
            funcs = (self.norm, self.norm.inverse)
            self._set_scale('function', functions=funcs)

    def _locate(self, x):
        """
        Given a set of color data values, return their
        corresponding colorbar data coordinates.
        """
        if isinstance(self.norm, (colors.NoNorm, colors.BoundaryNorm)):
            b = self._boundaries
            xn = x
        else:
            # Do calculations using normalized coordinates so
            # as to make the interpolation more accurate.
            b = self.norm(self._boundaries, clip=False).filled()
            xn = self.norm(x, clip=False).filled()

        bunique = b[self._inside]
        yunique = self._y

        z = np.interp(xn, bunique, yunique)
        return z

    # trivial helpers

    def _uniform_y(self, N):
        """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus extension lengths if required.
        """
        automin = automax = 1. / (N - 1.)
        extendlength = self._get_extension_lengths(self.extendfrac,
                                                   automin, automax,
                                                   default=0.05)
        y = np.linspace(0, 1, N)
        return y, extendlength

    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar, plus extension lengths if required:
        """
        if (isinstance(self.norm, colors.BoundaryNorm) or
                self.boundaries is not None):
            y = (self._boundaries - self._boundaries[self._inside][0])
            y = y / (self._boundaries[self._inside][-1] -
                     self._boundaries[self._inside][0])
            # need yscaled the same as the axes scale to get
            # the extend lengths.
            if self.spacing == 'uniform':
                yscaled = self._forward_boundaries(self._boundaries)
            else:
                yscaled = y
        else:
            y = self.norm(self._boundaries.copy())
            y = np.ma.filled(y, np.nan)
            # the norm and the scale should be the same...
            yscaled = y
        y = y[self._inside]
        yscaled = yscaled[self._inside]
        # normalize from 0..1:
        norm = colors.Normalize(y[0], y[-1])
        y = np.ma.filled(norm(y), np.nan)
        norm = colors.Normalize(yscaled[0], yscaled[-1])
        yscaled = np.ma.filled(norm(yscaled), np.nan)
        # make the lower and upper extend lengths proportional to the lengths
        # of the first and last boundary spacing (if extendfrac='auto'):
        automin = yscaled[1] - yscaled[0]
        automax = yscaled[-1] - yscaled[-2]
        extendlength = [0, 0]
        if self._extend_lower() or self._extend_upper():
            extendlength = self._get_extension_lengths(
                    self.extendfrac, automin, automax, default=0.05)
        return y, extendlength

    def _get_extension_lengths(self, frac, automin, automax, default=0.05):
        """
        Return the lengths of colorbar extensions.

        This is a helper method for _uniform_y and _proportional_y.
        """
        # Set the default value.
        extendlength = np.array([default, default])
        if isinstance(frac, str):
            _api.check_in_list(['auto'], extendfrac=frac.lower())
            # Use the provided values when 'auto' is required.
            extendlength[:] = [automin, automax]
        elif frac is not None:
            try:
                # Try to set min and max extension fractions directly.
                extendlength[:] = frac
                # If frac is a sequence containing None then NaN may
                # be encountered. This is an error.
                if np.isnan(extendlength).any():
                    raise ValueError()
            except (TypeError, ValueError) as err:
                # Raise an error on encountering an invalid value for frac.
                raise ValueError('invalid value for extendfrac') from err
        return extendlength

    def _extend_lower(self):
        """Return whether the lower limit is open ended."""
        minmax = "max" if self.long_axis.get_inverted() else "min"
        return self.extend in ('both', minmax)

    def _extend_upper(self):
        """Return whether the upper limit is open ended."""
        minmax = "min" if self.long_axis.get_inverted() else "max"
        return self.extend in ('both', minmax)

    def _short_axis(self):
        """Return the short axis"""
        if self.orientation == 'vertical':
            return self.ax.xaxis
        return self.ax.yaxis

    def _get_view(self):
        # docstring inherited
        # An interactive view for a colorbar is the norm's vmin/vmax
        return self.norm.vmin, self.norm.vmax

    def _set_view(self, view):
        # docstring inherited
        # An interactive view for a colorbar is the norm's vmin/vmax
        self.norm.vmin, self.norm.vmax = view

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        # docstring inherited
        # For colorbars, we use the zoom bbox to scale the norm's vmin/vmax
        new_xbound, new_ybound = self.ax._prepare_view_from_bbox(
            bbox, direction=direction, mode=mode, twinx=twinx, twiny=twiny)
        if self.orientation == 'horizontal':
            self.norm.vmin, self.norm.vmax = new_xbound
        elif self.orientation == 'vertical':
            self.norm.vmin, self.norm.vmax = new_ybound

    def drag_pan(self, button, key, x, y):
        # docstring inherited
        points = self.ax._get_pan_points(button, key, x, y)
        if points is not None:
            if self.orientation == 'horizontal':
                self.norm.vmin, self.norm.vmax = points[:, 0]
            elif self.orientation == 'vertical':
                self.norm.vmin, self.norm.vmax = points[:, 1]


ColorbarBase = Colorbar  # Backcompat API


def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _get_ticklocation_from_orientation(orientation)
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "anchor": (1.0, 0.5),
                   "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "anchor": (0.0, 0.5),
                   "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "anchor": (0.5, 0.0),
                   "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "anchor": (0.5, 1.0),
                   "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    loc_settings["orientation"] = _get_orientation_from_location(location)
    if orientation is not None and orientation != loc_settings["orientation"]:
        # Allow the user to pass both if they are consistent.
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings


def _get_orientation_from_location(location):
    return _api.check_getitem(
        {None: None, "left": "vertical", "right": "vertical",
         "top": "horizontal", "bottom": "horizontal"}, location=location)


def _get_ticklocation_from_orientation(orientation):
    return _api.check_getitem(
        {None: "right", "vertical": "right", "horizontal": "bottom"},
        orientation=orientation)


@_docstring.interpd
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The Axes is placed in the figure of the *parents* Axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
    loc_settings = _normalize_location_orientation(location, orientation)
    # put appropriate values into the kwargs dict for passing back to
    # the Colorbar class
    kwargs['orientation'] = loc_settings['orientation']
    location = kwargs['ticklocation'] = loc_settings['location']

    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    aspect0 = aspect
    # turn parents into a list if it is not already.  Note we cannot
    # use .flatten or .ravel as these copy the references rather than
    # reuse them, leading to a memory leak
    if isinstance(parents, np.ndarray):
        parents = list(parents.flat)
    elif np.iterable(parents):
        parents = list(parents)
    else:
        parents = [parents]

    fig = parents[0].get_figure()

    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kwargs.pop('pad', pad0)

    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar Axes as not all '
                         'parents share the same figure.')

    # take a bounding box around all of the given Axes
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # define the aspect ratio in terms of y's per x rather than x's per y
        aspect = 1.0 / aspect

    # define a transform which takes us from old axes coordinates to
    # new axes coordinates
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)

    # transform each of the Axes in parents using the new transform
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if panchor is not False:
            ax.set_anchor(panchor)

    cax = fig.add_axes(pbcb, label="<colorbar>")
    for a in parents:
        a._colorbars.append(cax)  # tell the parent it has a colorbar
    cax._colorbar_info = dict(
        parents=parents,
        location=location,
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect0,
        pad=pad)
    # and we need to set the aspect ratio by hand...
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')

    return cax, kwargs


@_docstring.interpd
def make_axes_gridspec(parent, *, location=None, orientation=None,
                       fraction=0.15, shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The Axes is placed in the figure of the *parent* Axes, by resizing and
    repositioning *parent*.

    This function is similar to `.make_axes` and mostly compatible with it.
    Primary differences are

    - `.make_axes_gridspec` requires the *parent* to have a subplotspec.
    - `.make_axes` positions the Axes in figure coordinates;
      `.make_axes_gridspec` positions it using a subplotspec.
    - `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
      replaces the parent gridspec with a new one.

    Parameters
    ----------
    parent : `~matplotlib.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """

    loc_settings = _normalize_location_orientation(location, orientation)
    kwargs['orientation'] = loc_settings['orientation']
    location = kwargs['ticklocation'] = loc_settings['location']

    aspect0 = aspect
    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    pad = kwargs.pop('pad', loc_settings["pad"])
    wh_space = 2 * pad / (1 - pad)

    if location in ('left', 'right'):
        gs = parent.get_subplotspec().subgridspec(
            3, 2, wspace=wh_space, hspace=0,
            height_ratios=[(1-anchor[1])*(1-shrink), shrink, anchor[1]*(1-shrink)])
        if location == 'left':
            gs.set_width_ratios([fraction, 1 - fraction - pad])
            ss_main = gs[:, 1]
            ss_cb = gs[1, 0]
        else:
            gs.set_width_ratios([1 - fraction - pad, fraction])
            ss_main = gs[:, 0]
            ss_cb = gs[1, 1]
    else:
        gs = parent.get_subplotspec().subgridspec(
            2, 3, hspace=wh_space, wspace=0,
            width_ratios=[anchor[0]*(1-shrink), shrink, (1-anchor[0])*(1-shrink)])
        if location == 'top':
            gs.set_height_ratios([fraction, 1 - fraction - pad])
            ss_main = gs[1, :]
            ss_cb = gs[0, 1]
        else:
            gs.set_height_ratios([1 - fraction - pad, fraction])
            ss_main = gs[0, :]
            ss_cb = gs[1, 1]
        aspect = 1 / aspect

    parent.set_subplotspec(ss_main)
    if panchor is not False:
        parent.set_anchor(panchor)

    fig = parent.get_figure()
    cax = fig.add_subplot(ss_cb, label="<colorbar>")
    parent._colorbars.append(cax)  # tell the parent it has a colorbar
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')
    cax._colorbar_info = dict(
        location=location,
        parents=[parent],
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect0,
        pad=pad)

    return cax, kwargs
<|endoftext|>r"""
Support for embedded TeX expressions in Matplotlib.

Requirements:

* LaTeX.
* \*Agg backends: dvipng>=1.6.
* PS backend: PSfrag, dvips, and Ghostscript>=9.0.
* PDF and SVG backends: if LuaTeX is present, it will be used to speed up some
  post-processing steps, but note that it is not used to parse the TeX string
  itself (only LaTeX is supported).

To enable TeX rendering of all text in your Matplotlib figure, set
:rc:`text.usetex` to True.

TeX and dvipng/dvips processing results are cached
in ~/.matplotlib/tex.cache for reuse between sessions.

`TexManager.get_rgba` can also be used to directly obtain raster output as RGBA
NumPy arrays.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

import matplotlib as mpl
from matplotlib import cbook, dviread

_log = logging.getLogger(__name__)


def _usepackage_if_not_loaded(package, *, option=None):
    """
    Output LaTeX code that loads a package (possibly with an option) if it
    hasn't been loaded yet.

    LaTeX cannot load twice a package with different options, so this helper
    can be used to protect against users loading arbitrary packages/options in
    their custom preamble.
    """
    option = f"[{option}]" if option is not None else ""
    return (
        r"\makeatletter"
        r"\@ifpackageloaded{%(package)s}{}{\usepackage%(option)s{%(package)s}}"
        r"\makeatother"
    ) % {"package": package, "option": option}


class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a directory.

    The cache directory is called ``tex.cache`` and is located in the directory
    returned by `.get_cachedir`.

    Repeated calls to this constructor always return the same instance.
    """

    _texcache = os.path.join(mpl.get_cachedir(), 'tex.cache')
    _grey_arrayd = {}

    _font_families = ('serif', 'sans-serif', 'cursive', 'monospace')
    _font_preambles = {
        'new century schoolbook': r'\renewcommand{\rmdefault}{pnc}',
        'bookman': r'\renewcommand{\rmdefault}{pbk}',
        'times': r'\usepackage{mathptmx}',
        'palatino': r'\usepackage{mathpazo}',
        'zapf chancery': r'\usepackage{chancery}',
        'cursive': r'\usepackage{chancery}',
        'charter': r'\usepackage{charter}',
        'serif': '',
        'sans-serif': '',
        'helvetica': r'\usepackage{helvet}',
        'avant garde': r'\usepackage{avant}',
        'courier': r'\usepackage{courier}',
        # Loading the type1ec package ensures that cm-super is installed, which
        # is necessary for Unicode computer modern.  (It also allows the use of
        # computer modern at arbitrary sizes, but that's just a side effect.)
        'monospace': r'\usepackage{type1ec}',
        'computer modern roman': r'\usepackage{type1ec}',
        'computer modern sans serif': r'\usepackage{type1ec}',
        'computer modern typewriter': r'\usepackage{type1ec}',
    }
    _font_types = {
        'new century schoolbook': 'serif',
        'bookman': 'serif',
        'times': 'serif',
        'palatino': 'serif',
        'zapf chancery': 'cursive',
        'charter': 'serif',
        'helvetica': 'sans-serif',
        'avant garde': 'sans-serif',
        'courier': 'monospace',
        'computer modern roman': 'serif',
        'computer modern sans serif': 'sans-serif',
        'computer modern typewriter': 'monospace',
    }

    @functools.lru_cache  # Always return the same instance.
    def __new__(cls):
        Path(cls._texcache).mkdir(parents=True, exist_ok=True)
        return object.__new__(cls)

    @classmethod
    def _get_font_family_and_reduced(cls):
        """Return the font family name and whether the font is reduced."""
        ff = mpl.rcParams['font.family']
        ff_val = ff[0].lower() if len(ff) == 1 else None
        if len(ff) == 1 and ff_val in cls._font_families:
            return ff_val, False
        elif len(ff) == 1 and ff_val in cls._font_preambles:
            return cls._font_types[ff_val], True
        else:
            _log.info('font.family must be one of (%s) when text.usetex is '
                      'True. serif will be used by default.',
                      ', '.join(cls._font_families))
            return 'serif', False

    @classmethod
    def _get_font_preamble_and_command(cls):
        requested_family, is_reduced_font = cls._get_font_family_and_reduced()

        preambles = {}
        for font_family in cls._font_families:
            if is_reduced_font and font_family == requested_family:
                preambles[font_family] = cls._font_preambles[
                    mpl.rcParams['font.family'][0].lower()]
            else:
                rcfonts = mpl.rcParams[f"font.{font_family}"]
                for i, font in enumerate(map(str.lower, rcfonts)):
                    if font in cls._font_preambles:
                        preambles[font_family] = cls._font_preambles[font]
                        _log.debug(
                            'family: %s, package: %s, font: %s, skipped: %s',
                            font_family, cls._font_preambles[font], rcfonts[i],
                            ', '.join(rcfonts[:i]),
                        )
                        break
                else:
                    _log.info('No LaTeX-compatible font found for the %s font'
                              'family in rcParams. Using default.',
                              font_family)
                    preambles[font_family] = cls._font_preambles[font_family]

        # The following packages and commands need to be included in the latex
        # file's preamble:
        cmd = {preambles[family]
               for family in ['serif', 'sans-serif', 'monospace']}
        if requested_family == 'cursive':
            cmd.add(preambles['cursive'])
        cmd.add(r'\usepackage{type1cm}')
        preamble = '\n'.join(sorted(cmd))
        fontcmd = (r'\sffamily' if requested_family == 'sans-serif' else
                   r'\ttfamily' if requested_family == 'monospace' else
                   r'\rmfamily')
        return preamble, fontcmd

    @classmethod
    def get_basefile(cls, tex, fontsize, dpi=None):
        """
        Return a filename based on a hash of the string, fontsize, and dpi.
        """
        src = cls._get_tex_source(tex, fontsize) + str(dpi)
        filehash = hashlib.sha256(
            src.encode('utf-8'),
            usedforsecurity=False
        ).hexdigest()
        filepath = Path(cls._texcache)

        num_letters, num_levels = 2, 2
        for i in range(0, num_letters*num_levels, num_letters):
            filepath = filepath / Path(filehash[i:i+2])

        filepath.mkdir(parents=True, exist_ok=True)
        return os.path.join(filepath, filehash)

    @classmethod
    def get_font_preamble(cls):
        """
        Return a string containing font configuration for the tex preamble.
        """
        font_preamble, command = cls._get_font_preamble_and_command()
        return font_preamble

    @classmethod
    def get_custom_preamble(cls):
        """Return a string containing user additions to the tex preamble."""
        return mpl.rcParams['text.latex.preamble']

    @classmethod
    def _get_tex_source(cls, tex, fontsize):
        """Return the complete TeX source for processing a TeX string."""
        font_preamble, fontcmd = cls._get_font_preamble_and_command()
        baselineskip = 1.25 * fontsize
        return "\n".join([
            r"\documentclass{article}",
            r"% Pass-through \mathdefault, which is used in non-usetex mode",
            r"% to use the default text font but was historically suppressed",
            r"% in usetex mode.",
            r"\newcommand{\mathdefault}[1]{#1}",
            font_preamble,
            r"\usepackage[utf8]{inputenc}",
            r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}",
            r"% geometry is loaded before the custom preamble as ",
            r"% convert_psfrags relies on a custom preamble to change the ",
            r"% geometry.",
            r"\usepackage[papersize=72in, margin=1in]{geometry}",
            cls.get_custom_preamble(),
            r"% Use `underscore` package to take care of underscores in text.",
            r"% The [strings] option allows to use underscores in file names.",
            _usepackage_if_not_loaded("underscore", option="strings"),
            r"% Custom packages (e.g. newtxtext) may already have loaded ",
            r"% textcomp with different options.",
            _usepackage_if_not_loaded("textcomp"),
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"% The empty hbox ensures that a page is printed even for empty",
            r"% inputs, except when using psfrag which gets confused by it.",
            r"% matplotlibbaselinemarker is used by dviread to detect the",
            r"% last line's baseline.",
            rf"\fontsize{{{fontsize}}}{{{baselineskip}}}%",
            r"\ifdefined\psfrag\else\hbox{}\fi%",
            rf"{{{fontcmd} {tex}}}%",
            r"\end{document}",
        ])

    @classmethod
    def make_tex(cls, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        Return the file name.
        """
        texfile = cls.get_basefile(tex, fontsize) + ".tex"
        Path(texfile).write_text(cls._get_tex_source(tex, fontsize),
                                 encoding='utf-8')
        return texfile

    @classmethod
    def _run_checked_subprocess(cls, command, tex, *, cwd=None):
        _log.debug(cbook._pformat_subprocess(command))
        try:
            report = subprocess.check_output(
                command, cwd=cwd if cwd is not None else cls._texcache,
                stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f'Failed to process string with tex because {command[0]} '
                'could not be found') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                '{prog} was not able to process the following string:\n'
                '{tex!r}\n\n'
                'Here is the full command invocation and its output:\n\n'
                '{format_command}\n\n'
                '{exc}\n\n'.format(
                    prog=command[0],
                    format_command=cbook._pformat_subprocess(command),
                    tex=tex.encode('unicode_escape'),
                    exc=exc.output.decode('utf-8', 'backslashreplace'))
                ) from None
        _log.debug(report)
        return report

    @classmethod
    def make_dvi(cls, tex, fontsize):
        """
        Generate a dvi file containing latex's layout of tex string.

        Return the file name.
        """
        dvifile = Path(cls.get_basefile(tex, fontsize)).with_suffix(".dvi")
        if not dvifile.exists():
            # Generate the tex and dvi in a temporary directory to avoid race
            # conditions e.g. if multiple processes try to process the same tex
            # string at the same time.  Having tmpdir be a subdirectory of the
            # final output dir ensures that they are on the same filesystem,
            # and thus replace() works atomically.  It also allows referring to
            # the texfile with a relative path (for pathological MPLCONFIGDIRs,
            # the absolute path may contain characters (e.g. ~) that TeX does
            # not support; n.b. relative paths cannot traverse parents, or it
            # will be blocked when `openin_any = p` in texmf.cnf).
            with TemporaryDirectory(dir=dvifile.parent) as tmpdir:
                Path(tmpdir, "file.tex").write_text(
                    cls._get_tex_source(tex, fontsize), encoding='utf-8')
                cls._run_checked_subprocess(
                    ["latex", "-interaction=nonstopmode", "-halt-on-error",
                     "-no-shell-escape", "file.tex"], tex, cwd=tmpdir)
                Path(tmpdir, "file.dvi").replace(dvifile)
                # Also move the tex source to the main cache directory, but
                # only for backcompat.
                Path(tmpdir, "file.tex").replace(dvifile.with_suffix(".tex"))
        return str(dvifile)

    @classmethod
    def make_png(cls, tex, fontsize, dpi):
        """
        Generate a png file containing latex's rendering of tex string.

        Return the file name.
        """
        pngfile = Path(cls.get_basefile(tex, fontsize, dpi)).with_suffix(".png")
        # see get_rgba for a discussion of the background
        if not pngfile.exists():
            dvifile = cls.make_dvi(tex, fontsize)
            with TemporaryDirectory(dir=pngfile.parent) as tmpdir:
                cmd = ["dvipng", "-bg", "Transparent", "-D", str(dpi),
                       "-T", "tight", "-o", "file.png", dvifile]
                # When testing, disable FreeType rendering for reproducibility;
                # but dvipng 1.16 has a bug (fixed in f3ff241) that breaks
                # --freetype0 mode, so for it we keep FreeType enabled; the
                # image will be slightly off.
                if (getattr(mpl, "_called_from_pytest", False) and
                        mpl._get_executable_info("dvipng").raw_version != "1.16"):
                    cmd.insert(1, "--freetype0")
                cls._run_checked_subprocess(cmd, tex, cwd=tmpdir)
                Path(tmpdir, "file.png").replace(pngfile)
        return str(pngfile)

    @classmethod
    def get_grey(cls, tex, fontsize=None, dpi=None):
        """Return the alpha channel."""
        if not fontsize:
            fontsize = mpl.rcParams['font.size']
        if not dpi:
            dpi = mpl.rcParams['savefig.dpi']
        key = cls._get_tex_source(tex, fontsize), dpi
        alpha = cls._grey_arrayd.get(key)
        if alpha is None:
            pngfile = cls.make_png(tex, fontsize, dpi)
            rgba = mpl.image.imread(os.path.join(cls._texcache, pngfile))
            cls._grey_arrayd[key] = alpha = rgba[:, :, -1]
        return alpha

    @classmethod
    def get_rgba(cls, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
        r"""
        Return latex's rendering of the tex string as an RGBA array.

        Examples
        --------
        >>> texmanager = TexManager()
        >>> s = r"\TeX\ is $\displaystyle\sum_n\frac{-e^{i\pi}}{2^n}$!"
        >>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))
        """
        alpha = cls.get_grey(tex, fontsize, dpi)
        rgba = np.empty((*alpha.shape, 4))
        rgba[..., :3] = mpl.colors.to_rgb(rgb)
        rgba[..., -1] = alpha
        return rgba

    @classmethod
    def get_text_width_height_descent(cls, tex, fontsize, renderer=None):
        """Return width, height and descent of the text."""
        if tex.strip() == '':
            return 0, 0, 0
        dvifile = cls.make_dvi(tex, fontsize)
        dpi_fraction = renderer.points_to_pixels(1.) if renderer else 1
        with dviread.Dvi(dvifile, 72 * dpi_fraction) as dvi:
            page, = dvi
        # A total height (including the descent) needs to be returned.
        return page.width, page.height + page.descent, page.descent
<|endoftext|># Original code by:
#    John Gill <jng@europe.renre.com>
#    Copyright 2004 John Gill and John Hunter
#
# Subsequent changes:
#    The Matplotlib development team
#    Copyright The Matplotlib development team

"""
Tables drawing.

.. note::
    The table implementation in Matplotlib is lightly maintained. For a more
    featureful table implementation, you may wish to try `blume
    <https://github.com/swfiua/blume>`_.

Use the factory function `~matplotlib.table.table` to create a ready-made
table from texts. If you need more control, use the `.Table` class and its
methods.

The table consists of a grid of cells, which are indexed by (row, column).
The cell (0, 0) is positioned at the top left.

Thanks to John Gill for providing the class and table.
"""

import numpy as np

from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path

from .cbook import _is_pandas_dataframe


class Cell(Rectangle):
    """
    A cell is a `.Rectangle` with some associated `.Text`.

    As a user, you'll most likely not creates cells yourself. Instead, you
    should use either the `~matplotlib.table.table` factory function or
    `.Table.add_cell`.
    """

    PAD = 0.1
    """Padding between text and rectangle."""

    _edges = 'BRTL'
    _edge_aliases = {'open':         '',
                     'closed':       _edges,  # default
                     'horizontal':   'BT',
                     'vertical':     'RL'
                     }

    def __init__(self, xy, width, height, *,
                 edgecolor='k', facecolor='w',
                 fill=True,
                 text='',
                 loc='right',
                 fontproperties=None,
                 visible_edges='closed',
                 ):
        """
        Parameters
        ----------
        xy : 2-tuple
            The position of the bottom left corner of the cell.
        width : float
            The cell width.
        height : float
            The cell height.
        edgecolor : :mpltype:`color`, default: 'k'
            The color of the cell border.
        facecolor : :mpltype:`color`, default: 'w'
            The cell facecolor.
        fill : bool, default: True
            Whether the cell background is filled.
        text : str, optional
            The cell text.
        loc : {'right', 'center', 'left'}
            The alignment of the text within the cell.
        fontproperties : dict, optional
            A dict defining the font properties of the text. Supported keys and
            values are the keyword arguments accepted by `.FontProperties`.
        visible_edges : {'closed', 'open', 'horizontal', 'vertical'} or \
substring of 'BRTL'
            The cell edges to be drawn with a line: a substring of 'BRTL'
            (bottom, right, top, left), or one of 'open' (no edges drawn),
            'closed' (all edges drawn), 'horizontal' (bottom and top),
            'vertical' (right and left).
        """

        # Call base
        super().__init__(xy, width=width, height=height, fill=fill,
                         edgecolor=edgecolor, facecolor=facecolor)
        self.set_clip_on(False)
        self.visible_edges = visible_edges

        # Create text object
        self._loc = loc
        self._text = Text(x=xy[0], y=xy[1], clip_on=False,
                          text=text, fontproperties=fontproperties,
                          horizontalalignment=loc, verticalalignment='center')

    def set_transform(self, t):
        super().set_transform(t)
        # the text does not get the transform!
        self.stale = True

    def set_figure(self, fig):
        super().set_figure(fig)
        self._text.set_figure(fig)

    def get_text(self):
        """Return the cell `.Text` instance."""
        return self._text

    def set_fontsize(self, size):
        """Set the text fontsize."""
        self._text.set_fontsize(size)
        self.stale = True

    def get_fontsize(self):
        """Return the cell fontsize."""
        return self._text.get_fontsize()

    def auto_set_font_size(self, renderer):
        """Shrink font size until the text fits into the cell width."""
        fontsize = self.get_fontsize()
        required = self.get_required_width(renderer)
        while fontsize > 1 and required > self.get_width():
            fontsize -= 1
            self.set_fontsize(fontsize)
            required = self.get_required_width(renderer)

        return fontsize

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        # draw the rectangle
        super().draw(renderer)
        # position the text
        self._set_text_position(renderer)
        self._text.draw(renderer)
        self.stale = False

    def _set_text_position(self, renderer):
        """Set text up so it is drawn in the right place."""
        bbox = self.get_window_extent(renderer)
        # center vertically
        y = bbox.y0 + bbox.height / 2
        # position horizontally
        loc = self._text.get_horizontalalignment()
        if loc == 'center':
            x = bbox.x0 + bbox.width / 2
        elif loc == 'left':
            x = bbox.x0 + bbox.width * self.PAD
        else:  # right.
            x = bbox.x0 + bbox.width * (1 - self.PAD)
        self._text.set_position((x, y))

    def get_text_bounds(self, renderer):
        """
        Return the text bounds as *(x, y, width, height)* in table coordinates.
        """
        return (self._text.get_window_extent(renderer)
                .transformed(self.get_data_transform().inverted())
                .bounds)

    def get_required_width(self, renderer):
        """Return the minimal required width for the cell."""
        l, b, w, h = self.get_text_bounds(renderer)
        return w * (1.0 + (2.0 * self.PAD))

    @_docstring.interpd
    def set_text_props(self, **kwargs):
        """
        Update the text properties.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
        self._text._internal_update(kwargs)
        self.stale = True

    @property
    def visible_edges(self):
        """
        The cell edges to be drawn with a line.

        Reading this property returns a substring of 'BRTL' (bottom, right,
        top, left').

        When setting this property, you can use a substring of 'BRTL' or one
        of {'open', 'closed', 'horizontal', 'vertical'}.
        """
        return self._visible_edges

    @visible_edges.setter
    def visible_edges(self, value):
        if value is None:
            self._visible_edges = self._edges
        elif value in self._edge_aliases:
            self._visible_edges = self._edge_aliases[value]
        else:
            if any(edge not in self._edges for edge in value):
                raise ValueError('Invalid edge param {}, must only be one of '
                                 '{} or string of {}'.format(
                                     value,
                                     ", ".join(self._edge_aliases),
                                     ", ".join(self._edges)))
            self._visible_edges = value
        self.stale = True

    def get_path(self):
        """Return a `.Path` for the `.visible_edges`."""
        codes = [Path.MOVETO]
        codes.extend(
            Path.LINETO if edge in self._visible_edges else Path.MOVETO
            for edge in self._edges)
        if Path.MOVETO not in codes[1:]:  # All sides are visible
            codes[-1] = Path.CLOSEPOLY
        return Path(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            codes,
            readonly=True
            )


CustomCell = Cell  # Backcompat. alias.


class Table(Artist):
    """
    A table of cells.

    The table consists of a grid of cells, which are indexed by (row, column).

    For a simple table, you'll have a full grid of cells with indices from
    (0, 0) to (num_rows-1, num_cols-1), in which the cell (0, 0) is positioned
    at the top left. However, you can also add cells with negative indices.
    You don't have to add a cell to every grid position, so you can create
    tables that have holes.

    *Note*: You'll usually not create an empty table from scratch. Instead use
    `~matplotlib.table.table` to create a table from data.
    """
    codes = {'best': 0,
             'upper right':  1,  # default
             'upper left':   2,
             'lower left':   3,
             'lower right':  4,
             'center left':  5,
             'center right': 6,
             'lower center': 7,
             'upper center': 8,
             'center':       9,
             'top right':    10,
             'top left':     11,
             'bottom left':  12,
             'bottom right': 13,
             'right':        14,
             'left':         15,
             'top':          16,
             'bottom':       17,
             }
    """Possible values where to place the table relative to the Axes."""

    FONTSIZE = 10

    AXESPAD = 0.02
    """The border between the Axes and the table edge in Axes units."""

    def __init__(self, ax, loc=None, bbox=None, **kwargs):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` to plot the table into.
        loc : str, optional
            The position of the cell with respect to *ax*. This must be one of
            the `~.Table.codes`.
        bbox : `.Bbox` or [xmin, ymin, width, height], optional
            A bounding box to draw the table into. If this is not *None*, this
            overrides *loc*.

        Other Parameters
        ----------------
        **kwargs
            `.Artist` properties.
        """

        super().__init__()

        if isinstance(loc, str):
            if loc not in self.codes:
                raise ValueError(
                    "Unrecognized location {!r}. Valid locations are\n\t{}"
                    .format(loc, '\n\t'.join(self.codes)))
            loc = self.codes[loc]
        self.set_figure(ax.get_figure(root=False))
        self._axes = ax
        self._loc = loc
        self._bbox = bbox

        # use axes coords
        ax._unstale_viewLim()
        self.set_transform(ax.transAxes)

        self._cells = {}
        self._edges = None
        self._autoColumns = []
        self._autoFontsize = True
        self._internal_update(kwargs)

        self.set_clip_on(False)

    def add_cell(self, row, col, *args, **kwargs):
        """
        Create a cell and add it to the table.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        *args, **kwargs
            All other parameters are passed on to `Cell`.

        Returns
        -------
        `.Cell`
            The created cell.

        """
        xy = (0, 0)
        cell = Cell(xy, visible_edges=self.edges, *args, **kwargs)
        self[row, col] = cell
        return cell

    def __setitem__(self, position, cell):
        """
        Set a custom cell in a given position.
        """
        _api.check_isinstance(Cell, cell=cell)
        try:
            row, col = position[0], position[1]
        except Exception as err:
            raise KeyError('Only tuples length 2 are accepted as '
                           'coordinates') from err
        cell.set_figure(self.get_figure(root=False))
        cell.set_transform(self.get_transform())
        cell.set_clip_on(False)
        self._cells[row, col] = cell
        self.stale = True

    def __getitem__(self, position):
        """Retrieve a custom cell from a given position."""
        return self._cells[position]

    @property
    def edges(self):
        """
        The default value of `~.Cell.visible_edges` for newly added
        cells using `.add_cell`.

        Notes
        -----
        This setting does currently only affect newly created cells using
        `.add_cell`.

        To change existing cells, you have to set their edges explicitly::

            for c in tab.get_celld().values():
                c.visible_edges = 'horizontal'

        """
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value
        self.stale = True

    def _approx_text_height(self):
        return (self.FONTSIZE / 72.0 * self.get_figure(root=True).dpi /
                self._axes.bbox.height * 1.2)

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        # Need a renderer to do hit tests on mouseevent; assume the last one
        # will do
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        if renderer is None:
            raise RuntimeError('No renderer defined')

        if not self.get_visible():
            return
        renderer.open_group('table', gid=self.get_gid())
        self._update_positions(renderer)

        for key in sorted(self._cells):
            self._cells[key].draw(renderer)

        renderer.close_group('table')
        self.stale = False

    def _get_grid_bbox(self, renderer):
        """
        Get a bbox, in axes coordinates for the cells.

        Only include those in the range (0, 0) to (maxRow, maxCol).
        """
        boxes = [cell.get_window_extent(renderer)
                 for (row, col), cell in self._cells.items()
                 if row >= 0 and col >= 0]
        bbox = Bbox.union(boxes)
        return bbox.transformed(self.get_transform().inverted())

    def contains(self, mouseevent):
        # docstring inherited
        if self._different_canvas(mouseevent):
            return False, {}
        # TODO: Return index of the cell containing the cursor so that the user
        # doesn't have to bind to each one individually.
        renderer = self.get_figure(root=True)._get_renderer()
        if renderer is not None:
            boxes = [cell.get_window_extent(renderer)
                     for (row, col), cell in self._cells.items()
                     if row >= 0 and col >= 0]
            bbox = Bbox.union(boxes)
            return bbox.contains(mouseevent.x, mouseevent.y), {}
        else:
            return False, {}

    def get_children(self):
        """Return the Artists contained by the table."""
        return list(self._cells.values())

    def get_window_extent(self, renderer=None):
        # docstring inherited
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        self._update_positions(renderer)
        boxes = [cell.get_window_extent(renderer)
                 for cell in self._cells.values()]
        return Bbox.union(boxes)

    def _do_cell_alignment(self):
        """
        Calculate row heights and column widths; position cells accordingly.
        """
        # Calculate row/column widths
        widths = {}
        heights = {}
        for (row, col), cell in self._cells.items():
            height = heights.setdefault(row, 0.0)
            heights[row] = max(height, cell.get_height())
            width = widths.setdefault(col, 0.0)
            widths[col] = max(width, cell.get_width())

        # work out left position for each column
        xpos = 0
        lefts = {}
        for col in sorted(widths):
            lefts[col] = xpos
            xpos += widths[col]

        ypos = 0
        bottoms = {}
        for row in sorted(heights, reverse=True):
            bottoms[row] = ypos
            ypos += heights[row]

        # set cell positions
        for (row, col), cell in self._cells.items():
            cell.set_x(lefts[col])
            cell.set_y(bottoms[row])

    def auto_set_column_width(self, col):
        """
        Automatically set the widths of given columns to optimal sizes.

        Parameters
        ----------
        col : int or sequence of ints
            The indices of the columns to auto-scale.
        """
        col1d = np.atleast_1d(col)
        if not np.issubdtype(col1d.dtype, np.integer):
            raise TypeError("col must be an int or sequence of ints.")
        for cell in col1d:
            self._autoColumns.append(cell)

        self.stale = True

    def _auto_set_column_width(self, col, renderer):
        """Automatically set width for column."""
        cells = [cell for key, cell in self._cells.items() if key[1] == col]
        max_width = max((cell.get_required_width(renderer) for cell in cells),
                        default=0)
        for cell in cells:
            cell.set_width(max_width)

    def auto_set_font_size(self, value=True):
        """Automatically set font size."""
        self._autoFontsize = value
        self.stale = True

    def _auto_set_font_size(self, renderer):

        if len(self._cells) == 0:
            return
        fontsize = next(iter(self._cells.values())).get_fontsize()
        cells = []
        for key, cell in self._cells.items():
            # ignore auto-sized columns
            if key[1] in self._autoColumns:
                continue
            size = cell.auto_set_font_size(renderer)
            fontsize = min(fontsize, size)
            cells.append(cell)

        # now set all fontsizes equal
        for cell in self._cells.values():
            cell.set_fontsize(fontsize)

    def scale(self, xscale, yscale):
        """Scale column widths by *xscale* and row heights by *yscale*."""
        for c in self._cells.values():
            c.set_width(c.get_width() * xscale)
            c.set_height(c.get_height() * yscale)

    def set_fontsize(self, size):
        """
        Set the font size, in points, of the cell text.

        Parameters
        ----------
        size : float

        Notes
        -----
        As long as auto font size has not been disabled, the value will be
        clipped such that the text fits horizontally into the cell.

        You can disable this behavior using `.auto_set_font_size`.

        >>> the_table.auto_set_font_size(False)
        >>> the_table.set_fontsize(20)

        However, there is no automatic scaling of the row height so that the
        text may exceed the cell boundary.
        """
        for cell in self._cells.values():
            cell.set_fontsize(size)
        self.stale = True

    def _offset(self, ox, oy):
        """Move all the artists by ox, oy (axes coords)."""
        for c in self._cells.values():
            x, y = c.get_x(), c.get_y()
            c.set_x(x + ox)
            c.set_y(y + oy)

    def _update_positions(self, renderer):
        # called from renderer to allow more precise estimates of
        # widths and heights with get_window_extent

        # Do any auto width setting
        for col in self._autoColumns:
            self._auto_set_column_width(col, renderer)

        if self._autoFontsize:
            self._auto_set_font_size(renderer)

        # Align all the cells
        self._do_cell_alignment()

        bbox = self._get_grid_bbox(renderer)
        l, b, w, h = bbox.bounds

        if self._bbox is not None:
            # Position according to bbox
            if isinstance(self._bbox, Bbox):
                rl, rb, rw, rh = self._bbox.bounds
            else:
                rl, rb, rw, rh = self._bbox
            self.scale(rw / w, rh / h)
            ox = rl - l
            oy = rb - b
            self._do_cell_alignment()
        else:
            # Position using loc
            (BEST, UR, UL, LL, LR, CL, CR, LC, UC, C,
             TR, TL, BL, BR, R, L, T, B) = range(len(self.codes))
            # defaults for center
            ox = (0.5 - w / 2) - l
            oy = (0.5 - h / 2) - b
            if self._loc in (UL, LL, CL):   # left
                ox = self.AXESPAD - l
            if self._loc in (BEST, UR, LR, R, CR):  # right
                ox = 1 - (l + w + self.AXESPAD)
            if self._loc in (BEST, UR, UL, UC):     # upper
                oy = 1 - (b + h + self.AXESPAD)
            if self._loc in (LL, LR, LC):           # lower
                oy = self.AXESPAD - b
            if self._loc in (LC, UC, C):            # center x
                ox = (0.5 - w / 2) - l
            if self._loc in (CL, CR, C):            # center y
                oy = (0.5 - h / 2) - b

            if self._loc in (TL, BL, L):            # out left
                ox = - (l + w)
            if self._loc in (TR, BR, R):            # out right
                ox = 1.0 - l
            if self._loc in (TR, TL, T):            # out top
                oy = 1.0 - b
            if self._loc in (BL, BR, B):           # out bottom
                oy = - (b + h)

        self._offset(ox, oy)

    def get_celld(self):
        r"""
        Return a dict of cells in the table mapping *(row, column)* to
        `.Cell`\s.

        Notes
        -----
        You can also directly index into the Table object to access individual
        cells::

            cell = table[row, col]

        """
        return self._cells


@_docstring.interpd
def table(ax,
          cellText=None, cellColours=None,
          cellLoc='right', colWidths=None,
          rowLabels=None, rowColours=None, rowLoc='left',
          colLabels=None, colColours=None, colLoc='center',
          loc='bottom', bbox=None, edges='closed',
          **kwargs):
    """
    Add a table to an `~.axes.Axes`.

    At least one of *cellText* or *cellColours* must be specified. These
    parameters must be 2D lists, in which the outer lists define the rows and
    the inner list define the column values per row. Each row must have the
    same number of elements.

    The table can optionally have row and column headers, which are configured
    using *rowLabels*, *rowColours*, *rowLoc* and *colLabels*, *colColours*,
    *colLoc* respectively.

    For finer grained control over tables, use the `.Table` class and add it to
    the Axes with `.Axes.add_table`.

    Parameters
    ----------
    cellText : 2D list of str or pandas.DataFrame, optional
        The texts to place into the table cells.

        *Note*: Line breaks in the strings are currently not accounted for and
        will result in the text exceeding the cell boundaries.

    cellColours : 2D list of :mpltype:`color`, optional
        The background colors of the cells.

    cellLoc : {'right', 'center', 'left'}
        The alignment of the text within the cells.

    colWidths : list of float, optional
        The column widths in units of the axes. If not given, all columns will
        have a width of *1 / ncols*.

    rowLabels : list of str, optional
        The text of the row header cells.

    rowColours : list of :mpltype:`color`, optional
        The colors of the row header cells.

    rowLoc : {'left', 'center', 'right'}
        The text alignment of the row header cells.

    colLabels : list of str, optional
        The text of the column header cells.

    colColours : list of :mpltype:`color`, optional
        The colors of the column header cells.

    colLoc : {'center', 'left', 'right'}
        The text alignment of the column header cells.

    loc : str, default: 'bottom'
        The position of the cell with respect to *ax*. This must be one of
        the `~.Table.codes`.

    bbox : `.Bbox` or [xmin, ymin, width, height], optional
        A bounding box to draw the table into. If this is not *None*, this
        overrides *loc*.

    edges : {'closed', 'open', 'horizontal', 'vertical'} or substring of 'BRTL'
        The cell edges to be drawn with a line. See also
        `~.Cell.visible_edges`.

    Returns
    -------
    `~matplotlib.table.Table`
        The created table.

    Other Parameters
    ----------------
    **kwargs
        `.Table` properties.

    %(Table:kwdoc)s
    """

    if cellColours is None and cellText is None:
        raise ValueError('At least one argument from "cellColours" or '
                         '"cellText" must be provided to create a table.')

    # Check we have some cellText
    if cellText is None:
        # assume just colours are needed
        rows = len(cellColours)
        cols = len(cellColours[0])
        cellText = [[''] * cols] * rows

    # Check if we have a Pandas DataFrame
    if _is_pandas_dataframe(cellText):
        # if rowLabels/colLabels are empty, use DataFrame entries.
        # Otherwise, throw an error.
        if rowLabels is None:
            rowLabels = cellText.index
        else:
            raise ValueError("rowLabels cannot be used alongside Pandas DataFrame")
        if colLabels is None:
            colLabels = cellText.columns
        else:
            raise ValueError("colLabels cannot be used alongside Pandas DataFrame")
        # Update cellText with only values
        cellText = cellText.values

    rows = len(cellText)
    cols = len(cellText[0])
    for row in cellText:
        if len(row) != cols:
            raise ValueError(f"Each row in 'cellText' must have {cols} "
                             "columns")

    if cellColours is not None:
        if len(cellColours) != rows:
            raise ValueError(f"'cellColours' must have {rows} rows")
        for row in cellColours:
            if len(row) != cols:
                raise ValueError("Each row in 'cellColours' must have "
                                 f"{cols} columns")
    else:
        cellColours = ['w' * cols] * rows

    # Set colwidths if not given
    if colWidths is None:
        colWidths = [1.0 / cols] * cols

    # Fill in missing information for column
    # and row labels
    rowLabelWidth = 0
    if rowLabels is None:
        if rowColours is not None:
            rowLabels = [''] * rows
            rowLabelWidth = colWidths[0]
    elif rowColours is None:
        rowColours = 'w' * rows

    if rowLabels is not None:
        if len(rowLabels) != rows:
            raise ValueError(f"'rowLabels' must be of length {rows}")

    # If we have column labels, need to shift
    # the text and colour arrays down 1 row
    offset = 1
    if colLabels is None:
        if colColours is not None:
            colLabels = [''] * cols
        else:
            offset = 0
    elif colColours is None:
        colColours = 'w' * cols

    # Set up cell colours if not given
    if cellColours is None:
        cellColours = ['w' * cols] * rows

    # Now create the table
    table = Table(ax, loc, bbox, **kwargs)
    table.edges = edges
    height = table._approx_text_height()

    # Add the cells
    for row in range(rows):
        for col in range(cols):
            table.add_cell(row + offset, col,
                           width=colWidths[col], height=height,
                           text=cellText[row][col],
                           facecolor=cellColours[row][col],
                           loc=cellLoc)
    # Do column labels
    if colLabels is not None:
        for col in range(cols):
            table.add_cell(0, col,
                           width=colWidths[col], height=height,
                           text=colLabels[col], facecolor=colColours[col],
                           loc=colLoc)

    # Do row labels
    if rowLabels is not None:
        for row in range(rows):
            table.add_cell(row + offset, -1,
                           width=rowLabelWidth or 1e-15, height=height,
                           text=rowLabels[row], facecolor=rowColours[row],
                           loc=rowLoc)
        if rowLabelWidth == 0:
            table.auto_set_column_width(-1)

    # set_fontsize is only effective after cells are added
    if "fontsize" in kwargs:
        table.set_fontsize(kwargs["fontsize"])

    ax.add_table(table)
    return table
<|endoftext|>from collections.abc import MutableMapping
import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath


class Spine(mpatches.Patch):
    """
    An axis spine -- the line noting the data area boundaries.

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions. See `~.Spine.set_position` for more information.

    The default position is ``('outward', 0)``.

    Spines are subclasses of `.Patch`, and inherit much of their behavior.

    Spines draw a line, a circle, or an arc depending on if
    `~.Spine.set_patch_line`, `~.Spine.set_patch_circle`, or
    `~.Spine.set_patch_arc` has been called. Line-like is the default.

    For examples see :ref:`spines_examples`.
    """
    def __str__(self):
        return "Spine"

    @_docstring.interpd
    def __init__(self, axes, spine_type, path, **kwargs):
        """
        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance containing the spine.
        spine_type : str
            The spine type.
        path : `~matplotlib.path.Path`
            The `.Path` instance used to draw the spine.

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are:

            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        self.axes = axes
        self.set_figure(self.axes.get_figure(root=False))
        self.spine_type = spine_type
        self.set_facecolor('none')
        self.set_edgecolor(mpl.rcParams['axes.edgecolor'])
        self.set_linewidth(mpl.rcParams['axes.linewidth'])
        self.set_capstyle('projecting')
        self.axis = None

        self.set_zorder(2.5)
        self.set_transform(self.axes.transData)  # default transform

        self._bounds = None  # default bounds

        # Defer initial position determination. (Not much support for
        # non-rectangular axes is currently implemented, and this lets
        # them pass through the spines machinery without errors.)
        self._position = None
        _api.check_isinstance(mpath.Path, path=path)
        self._path = path

        # To support drawing both linear and circular spines, this
        # class implements Patch behavior three ways. If
        # self._patch_type == 'line', behave like a mpatches.PathPatch
        # instance. If self._patch_type == 'circle', behave like a
        # mpatches.Ellipse instance. If self._patch_type == 'arc', behave like
        # a mpatches.Arc instance.
        self._patch_type = 'line'

        # Behavior copied from mpatches.Ellipse:
        # Note: This cannot be calculated until this is added to an Axes
        self._patch_transform = mtransforms.IdentityTransform()

    def set_patch_arc(self, center, radius, theta1, theta2):
        """Set the spine to be arc-like."""
        self._patch_type = 'arc'
        self._center = center
        self._width = radius * 2
        self._height = radius * 2
        self._theta1 = theta1
        self._theta2 = theta2
        self._path = mpath.Path.arc(theta1, theta2)
        # arc drawn on axes transform
        self.set_transform(self.axes.transAxes)
        self.stale = True

    def set_patch_circle(self, center, radius):
        """Set the spine to be circular."""
        self._patch_type = 'circle'
        self._center = center
        self._width = radius * 2
        self._height = radius * 2
        # circle drawn on axes transform
        self.set_transform(self.axes.transAxes)
        self.stale = True

    def set_patch_line(self):
        """Set the spine to be linear."""
        self._patch_type = 'line'
        self.stale = True

    # Behavior copied from mpatches.Ellipse:
    def _recompute_transform(self):
        """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
        assert self._patch_type in ('arc', 'circle')
        center = (self.convert_xunits(self._center[0]),
                  self.convert_yunits(self._center[1]))
        width = self.convert_xunits(self._width)
        height = self.convert_yunits(self._height)
        self._patch_transform = mtransforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .translate(*center)

    def get_patch_transform(self):
        if self._patch_type in ('arc', 'circle'):
            self._recompute_transform()
            return self._patch_transform
        else:
            return super().get_patch_transform()

    def get_window_extent(self, renderer=None):
        """
        Return the window extent of the spines in display space, including
        padding for ticks (but not their labels)

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axes.Axes.get_window_extent
        """
        # make sure the location is updated so that transforms etc are correct:
        self._adjust_location()
        bb = super().get_window_extent(renderer=renderer)
        if self.axis is None or not self.axis.get_visible():
            return bb
        bboxes = [bb]
        drawn_ticks = self.axis._update_ticks()

        major_tick = next(iter({*drawn_ticks} & {*self.axis.majorTicks}), None)
        minor_tick = next(iter({*drawn_ticks} & {*self.axis.minorTicks}), None)
        for tick in [major_tick, minor_tick]:
            if tick is None:
                continue
            bb0 = bb.frozen()
            tickl = tick._size
            tickdir = tick._tickdir
            if tickdir == 'out':
                padout = 1
                padin = 0
            elif tickdir == 'in':
                padout = 0
                padin = 1
            else:
                padout = 0.5
                padin = 0.5
            dpi = self.get_figure(root=True).dpi
            padout = padout * tickl / 72 * dpi
            padin = padin * tickl / 72 * dpi

            if tick.tick1line.get_visible():
                if self.spine_type == 'left':
                    bb0.x0 = bb0.x0 - padout
                    bb0.x1 = bb0.x1 + padin
                elif self.spine_type == 'bottom':
                    bb0.y0 = bb0.y0 - padout
                    bb0.y1 = bb0.y1 + padin

            if tick.tick2line.get_visible():
                if self.spine_type == 'right':
                    bb0.x1 = bb0.x1 + padout
                    bb0.x0 = bb0.x0 - padin
                elif self.spine_type == 'top':
                    bb0.y1 = bb0.y1 + padout
                    bb0.y0 = bb0.y0 - padout
            bboxes.append(bb0)

        return mtransforms.Bbox.union(bboxes)

    def get_path(self):
        return self._path

    def _ensure_position_is_set(self):
        if self._position is None:
            # default position
            self._position = ('outward', 0.0)  # in points
            self.set_position(self._position)

    def register_axis(self, axis):
        """
        Register an axis.

        An axis should be registered with its corresponding spine from
