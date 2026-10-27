        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pdf
    logpdf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    fit
    fit_loc_scale
    nnlf
    support

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    the survival function ``_sf`` is computed as ``1 - _cdf`` which can
    result in loss of precision if ``_cdf(x)`` is close to one.

    **Methods that can be overwritten by subclasses**
    ::

      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support

    There are additional (internal and private) generic methods that can
    be useful for cross-checking and for debugging, but might work in all
    cases when directly called.

    A note on ``shapes``: subclasses need not specify them explicitly. In this
    case, `shapes` will be automatically deduced from the signatures of the
    overridden methods (`pdf`, `cdf` etc).
    If, for some reason, you prefer to avoid relying on introspection, you can
    specify ``shapes`` explicitly as an argument to the instance constructor.


    **Frozen Distributions**

    Normally, you must provide shape parameters (and, optionally, location and
    scale parameters to each call of a method of a distribution.

    Alternatively, the object may be called (as a function) to fix the shape,
    location, and scale parameters returning a "frozen" continuous RV object:

    rv = generic(<shape(s)>, loc=0, scale=1)
        `rv_frozen` object with the same methods but holding the given shape,
        location, and scale fixed

    **Statistics**

    Statistics are computed using numerical integration by default.
    For speed you can redefine this using ``_stats``:

     - take shape parameters and return mu, mu2, g1, g2
     - If you can't compute one of these, return it as None
     - Can also be defined with a keyword argument ``moments``, which is a
       string composed of "m", "v", "s", and/or "k".
       Only the components appearing in string should be computed and
       returned in the order "m", "v", "s", or "k"  with missing values
       returned as None.

    Alternatively, you can override ``_munp``, which takes ``n`` and shape
    parameters and returns the n-th non-central moment of the distribution.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    To create a new Gaussian distribution, we would do the following:

    >>> from scipy.stats import rv_continuous
    >>> class gaussian_gen(rv_continuous):
    ...     "Gaussian distribution"
    ...     def _pdf(self, x):
    ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    >>> gaussian = gaussian_gen(name='gaussian')

    ``scipy.stats`` distributions are *instances*, so here we subclass
    `rv_continuous` and create an instance. With this, we now have
    a fully functional distribution with all relevant methods automagically
    generated by the framework.

    Note that above we defined a standard normal distribution, with zero mean
    and unit variance. Shifting and scaling of the distribution can be done
    by using ``loc`` and ``scale`` parameters: ``gaussian.pdf(x, loc, scale)``
    essentially computes ``y = (x - loc) / scale`` and
    ``gaussian._pdf(y) / scale``.

    """

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, seed=None):

        super().__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            momtype=momtype, a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, seed=seed)

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes

        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')
        self._attach_methods()

        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            docdict=docdict,
                                            discrete='continuous')
            else:
                dct = dict(distcont)
                self._construct_doc(docdict, dct.get(self.name))

    def __getstate__(self):
        dct = self.__dict__.copy()

        # these methods will be remade in __setstate__
        # _random_state attribute is taken care of by rv_generic
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "vecentropy", "generic_moment"]
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """
        Attaches dynamically created methods to the rv_continuous instance.
        """
        # _attach_methods is responsible for calling _attach_argparser_methods
        self._attach_argparser_methods()

        # nin correction
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x, )+args)-q

    def _ppf_single(self, q, *args):
        factor = 10.
        left, right = self._get_support(*args)

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    # moment from definition
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b,
                              args=(m,)+args)[0]

    # moment calculated using ppf
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args))**m

    def _mom1_sc(self, m, *args):
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

    def _pdf(self, x, *args):
        return _derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    # Could also define any of these
    def _logpdf(self, x, *args):
        p = self._pdf(x, *args)
        with np.errstate(divide='ignore'):
            return log(p)

    def _logpxf(self, x, *args):
        # continuous distributions have PDF, discrete have PMF, but sometimes
        # the distinction doesn't matter. This lets us use `_logpxf` for both
        # discrete and continuous distributions.
        return self._logpdf(x, *args)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    def _logcdf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide='ignore'):
            return _lazywhere(x < median, (x,) + args,
                              f=lambda x, *args: np.log(self._cdf(x, *args)),
                              f2=lambda x, *args: np.log1p(-self._sf(x, *args)))

    def _logsf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide='ignore'):
            return _lazywhere(x > median, (x,) + args,
                              f=lambda x, *args: np.log(self._sf(x, *args)),
                              f2=lambda x, *args: np.log1p(-self._cdf(x, *args)))

    # generic _argcheck, _sf, _ppf, _isf, _rvs are defined
    # in rv_generic

    def pdf(self, x, *args, **kwds):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        """Log of the probability density function at x of the given RV.

        This uses a more numerically accurate calculation if available.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1-cond0)*(cond1 == cond1)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        """Survival function (1 - `cdf`) at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        sf : array_like
            Survival function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as (1 - `cdf`),
        evaluated at `x`.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `x`.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            upper tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : ndarray or scalar
            Quantile corresponding to the upper tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def _nnlf_and_penalty(self, x, args):
        """
        Compute the penalized negative log-likelihood for the
        "standardized" data (i.e. already shifted by loc and
        scaled by scale) for the shape parameters in `args`.

        `x` can be a 1D numpy array or a CensoredData instance.
        """
        if isinstance(x, CensoredData):
            # Filter out the data that is not in the support.
            xs = x._supported(*self._get_support(*args))
            n_bad = len(x) - len(xs)
            i1, i2 = xs._interval.T
            terms = [
                # logpdf of the noncensored data.
                self._logpdf(xs._uncensored, *args),
                # logcdf of the left-censored data.
                self._logcdf(xs._left, *args),
                # logsf of the right-censored data.
                self._logsf(xs._right, *args),
                # log of probability of the interval-censored data.
                np.log(self._delta_cdf(i1, i2, *args)),
            ]
        else:
            cond0 = ~self._support_mask(x, *args)
            n_bad = np.count_nonzero(cond0)
            if n_bad > 0:
                x = argsreduce(~cond0, x)[0]
            terms = [self._logpdf(x, *args)]

        totals, bad_counts = zip(*[_sum_finite(term) for term in terms])
        total = sum(totals)
        n_bad += sum(bad_counts)

        return -total + n_bad * _LOGXMAX * 100

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.

        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        if isinstance(x, CensoredData):
            x = (x - loc) / scale
            n_log_scale = (len(x) - x.num_censored()) * log(scale)
        else:
            x = (x - loc) / scale
            n_log_scale = len(x) * log(scale)

        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (1.0,)*self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        # Convert fixed shape parameters to the standard numeric form: e.g. for
        # stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        # for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        # into the first (numeric) form.
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        methods = {"mle", "mm"}
        method = kwds.pop('method', "mle").lower()
        if method == "mm":
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = (np.arange(1, n_params+1))[:, np.newaxis]
            data_moments = np.sum(data[None, :]**exponents/len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)

        elif method == "mle":
            objective = self._penalized_nnlf
        else:
            raise ValueError(f"Method '{method}' not available; "
                             f"must be one of {methods}")

        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)

        return x0, func, restore, args

    def _moment_error(self, theta, x, data_moments):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf

        dist_moments = np.array([self.moment(i+1, *args, loc=loc, scale=scale)
                                 for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite "
                             "distribution moment and cannot continue. "
                             "Consider trying method='MLE'.")

        return (((data_moments - dist_moments) /
                 np.maximum(np.abs(data_moments), 1e-8))**2).sum()

    def fit(self, data, *args, **kwds):
        r"""
        Return estimates of shape (if applicable), location, and scale
        parameters from data. The default estimation method is Maximum
        Likelihood Estimation (MLE), but Method of Moments (MM)
        is also available.

        Starting estimates for the fit are given by input arguments;
        for any arguments not provided with starting estimates,
        ``self._fitstart(data)`` is called to generate such.

        One can hold some parameters fixed to specific values by passing in
        keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
        and ``floc`` and ``fscale`` (for location and scale parameters,
        respectively).

        Parameters
        ----------
        data : array_like or `CensoredData` instance
            Data to use in estimating the distribution parameters.
        arg1, arg2, arg3,... : floats, optional
            Starting value(s) for any shape-characterizing arguments (those not
            provided will be determined by a call to ``_fitstart(data)``).
            No default value.
        **kwds : floats, optional
            - `loc`: initial guess of the distribution's location parameter.
            - `scale`: initial guess of the distribution's scale parameter.

            Special keyword arguments are recognized as holding certain
            parameters fixed:

            - f0...fn : hold respective shape parameters fixed.
              Alternatively, shape parameters to fix can be specified by name.
              For example, if ``self.shapes == "a, b"``, ``fa`` and ``fix_a``
              are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
              equivalent to ``f1``.

            - floc : hold location parameter fixed to specified value.

            - fscale : hold scale parameter fixed to specified value.

            - optimizer : The optimizer to use.  The optimizer must take
              ``func`` and starting position as the first two arguments,
              plus ``args`` (for extra arguments to pass to the
              function to be optimized) and ``disp``.
              The ``fit`` method calls the optimizer with ``disp=0`` to suppress output.
              The optimizer must return the estimated parameters.

            - method : The method to use. The default is "MLE" (Maximum
              Likelihood Estimate); "MM" (Method of Moments)
              is also available.

        Raises
        ------
        TypeError, ValueError
            If an input is invalid
        `~scipy.stats.FitError`
            If fitting fails or the fit produced would be invalid

        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).

        Notes
        -----
        With ``method="MLE"`` (default), the fit is computed by minimizing
        the negative log-likelihood function. A large, finite penalty
        (rather than infinite negative log-likelihood) is applied for
        observations beyond the support of the distribution.

        With ``method="MM"``, the fit is computed by minimizing the L2 norm
        of the relative errors between the first *k* raw (about zero) data
        moments and the corresponding distribution moments, where *k* is the
        number of non-fixed parameters.
        More precisely, the objective function is::

            (((data_moments - dist_moments)
              / np.maximum(np.abs(data_moments), 1e-8))**2).sum()

        where the constant ``1e-8`` avoids division by zero in case of
        vanishing data moments. Typically, this error norm can be reduced to
        zero.
        Note that the standard method of moments can produce parameters for
        which some data are outside the support of the fitted distribution;
        this implementation does nothing to prevent this.

        For either method,
        the returned answer is not guaranteed to be globally optimal; it
        may only be locally optimal, or the optimization may fail altogether.
        If the data contain any of ``np.nan``, ``np.inf``, or ``-np.inf``,
        the `fit` method will raise a ``RuntimeError``.

        When passing a ``CensoredData`` instance to ``data``, the log-likelihood
        function is defined as:

        .. math::

            l(\pmb{\theta}; k) & = \sum
                                    \log(f(k_u; \pmb{\theta}))
                                + \sum
                                    \log(F(k_l; \pmb{\theta})) \\
                                & + \sum
                                    \log(1 - F(k_r; \pmb{\theta})) \\
                                & + \sum
                                    \log(F(k_{\text{high}, i}; \pmb{\theta})
                                    - F(k_{\text{low}, i}; \pmb{\theta}))

        where :math:`f` and :math:`F` are the pdf and cdf, respectively, of the
        function being fitted, :math:`\pmb{\theta}` is the parameter vector,
        :math:`u` are the indices of uncensored observations,
        :math:`l` are the indices of left-censored observations,
        :math:`r` are the indices of right-censored observations,
        subscripts "low"/"high" denote endpoints of interval-censored observations, and
        :math:`i` are the indices of interval-censored observations.

        Examples
        --------

        Generate some data to fit: draw random variates from the `beta`
        distribution

        >>> import numpy as np
        >>> from scipy.stats import beta
        >>> a, b = 1., 2.
        >>> rng = np.random.default_rng(172786373191770012695001057628748821561)
        >>> x = beta.rvs(a, b, size=1000, random_state=rng)

        Now we can fit all four parameters (``a``, ``b``, ``loc`` and
        ``scale``):

        >>> a1, b1, loc1, scale1 = beta.fit(x)
        >>> a1, b1, loc1, scale1
        (1.0198945204435628, 1.9484708982737828, 4.372241314917588e-05, 0.9979078845964814)

        The fit can be done also using a custom optimizer:

        >>> from scipy.optimize import minimize
        >>> def custom_optimizer(func, x0, args=(), disp=0):
        ...     res = minimize(func, x0, args, method="slsqp", options={"disp": disp})
        ...     if res.success:
        ...         return res.x
        ...     raise RuntimeError('optimization routine failed')
        >>> a1, b1, loc1, scale1 = beta.fit(x, method="MLE", optimizer=custom_optimizer)
        >>> a1, b1, loc1, scale1
        (1.0198821087258905, 1.948484145914738, 4.3705304486881485e-05, 0.9979104663953395)

        We can also use some prior knowledge about the dataset: let's keep
        ``loc`` and ``scale`` fixed:

        >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
        >>> loc1, scale1
        (0, 1)

        We can also keep shape parameters fixed by using ``f``-keywords. To
        keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
        equivalently, ``fa=1``:

        >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
        >>> a1
        1

        Not all distributions return estimates for the shape parameters.
        ``norm`` for example just returns estimates for location and scale:

        >>> from scipy.stats import norm
        >>> x = norm.rvs(a, b, size=1000, random_state=123)
        >>> loc1, scale1 = norm.fit(x)
        >>> loc1, scale1
        (0.92087172783841631, 2.0015750750324668)
        """ # noqa: E501
        method = kwds.get('method', "mle").lower()

        censored = isinstance(data, CensoredData)
        if censored:
            if method != 'mle':
                raise ValueError('For censored data, the method must'
                                 ' be "MLE".')
            if data.num_censored() == 0:
                # There are no censored values in data, so replace the
                # CensoredData instance with a regular array.
                data = data._uncensored
                censored = False

        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        # Check the finiteness of data only if data is not an instance of
        # CensoredData.  The arrays in a CensoredData instance have already
        # been validated.
        if not censored:
            # Note: `ravel()` is called for backwards compatibility.
            data = np.asarray(data).ravel()
            if not np.isfinite(data).all():
                raise ValueError("The data contains non-finite values.")

        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        optimizer = _fit_determine_optimizer(optimizer)
        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError(f"Unknown arguments: {kwds}.")

        # In some cases, method of moments can be done with fsolve/root
        # instead of an optimizer, but sometimes no solution exists,
        # especially when the user fixes parameters. Minimizing the sum
        # of squares of the error generalizes to these cases.
        vals = optimizer(func, x0, args=(data,), disp=0)
        obj = func(vals, data)

        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)

        loc, scale, shapes = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise FitError("Optimization converged to parameters that are "
                           "outside the range allowed by the distribution.")

        if method == 'mm':
            if not np.isfinite(obj):
                raise FitError("Optimization failed: either a data moment "
                               "or fitted distribution moment is "
                               "non-finite.")

        return vals

    def _fit_loc_scale_support(self, data, *args):
        """Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        if isinstance(data, CensoredData):
            # For this estimate, "uncensor" the data by taking the
            # given endpoints as the data for the left- or right-censored
            # data, and the mean for the interval-censored data.
            data = data._uncensor()
        else:
            data = np.asarray(data)

        # Estimate location and scale according to the method of moments.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Compute the support according to the shape parameters.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support is empty then return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support according to the loc and scale
        # estimates.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use the moment-based estimates if they are compatible with the data.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise find other estimates that are compatible with the data.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, both the location and scale
        # should have interesting values.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, use only an interesting location parameter.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        with np.errstate(invalid='ignore'):
            Lhat = muhat - Shat*mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        return Lhat, Shat

    def _entropy(self, *args):
        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)

        # upper limit is often inf, so suppress warnings when integrating
        _a, _b = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = integrate.quad(integ, _a, _b)[0]

        if not np.isnan(h):
            return h
        else:
            # try with different limits if integration problems
            low, upp = self.ppf([1e-10, 1. - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None,
               conditional=False, **kwds):
        """Calculate expected value of a function with respect to the
        distribution by numerical integration.

        The expected value of a function ``f(x)`` with respect to a
        distribution ``dist`` is defined as::

                    ub
            E[f(x)] = Integral(f(x) * dist.pdf(x)),
                    lb

        where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)``
        distribution. If the bounds ``lb`` and ``ub`` correspond to the
        support of the distribution, e.g. ``[-inf, inf]`` in the default
        case, then the integral is the unrestricted expectation of ``f(x)``.
        Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0``
        outside a finite interval in which case the expectation is
        calculated within the finite range ``[lb, ub]``.

        Parameters
        ----------
        func : callable, optional
            Function for which integral is calculated. Takes only one argument.
            The default is the identity mapping f(x) = x.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter (default=0).
        scale : float, optional
            Scale parameter (default=1).
        lb, ub : scalar, optional
            Lower and upper bound for integration. Default is set to the
            support of the distribution.
        conditional : bool, optional
            If True, the integral is corrected by the conditional probability
            of the integration interval.  The return value is the expectation
            of the function, conditional on being in the given interval.
            Default is False.

        Additional keyword arguments are passed to the integration routine.

        Returns
        -------
        expect : float
            The calculated expected value.

        Notes
        -----
        The integration behavior of this function is inherited from
        `scipy.integrate.quad`. Neither this function nor
        `scipy.integrate.quad` can verify whether the integral exists or is
        finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and
        ``cauchy(0).expect()`` returns ``0.0``.

        Likewise, the accuracy of results is not verified by the function.
        `scipy.integrate.quad` is typically reliable for integrals that are
        numerically favorable, but it is not guaranteed to converge
        to a correct value for all possible intervals and integrands. This
        function is provided for convenience; for critical applications,
        check results against other integration methods.

        The function is not vectorized.

        Examples
        --------

        To understand the effect of the bounds of integration consider

        >>> from scipy.stats import expon
        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)
        0.6321205588285578

        This is close to

        >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)
        0.6321205588285577

        If ``conditional=True``

        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)
        1.0000000000000002

        The slight deviation from 1 is due to numerical integration.

        The integrand can be treated as a complex-valued function
        by passing ``complex_func=True`` to `scipy.integrate.quad` .

        >>> import numpy as np
        >>> from scipy.stats import vonmises
        >>> res = vonmises(loc=2, kappa=1).expect(lambda x: np.exp(1j*x),
        ...                                       complex_func=True)
        >>> res
        (-0.18576377217422957+0.40590124735052263j)

        >>> np.angle(res)  # location of the (circular) distribution
        2.0

        """
        lockwds = {'loc': loc,
                   'scale': scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:
            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)
        else:
            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale

        cdf_bounds = self.cdf([lb, ub], *args, **lockwds)
        invfac = cdf_bounds[1] - cdf_bounds[0]

        kwds['args'] = args

        # split interval to help integrator w/ infinite support; see gh-8928
        alpha = 0.05  # split body from tails at probability mass `alpha`
        inner_bounds = np.array([alpha, 1-alpha])
        cdf_inner_bounds = cdf_bounds[0] + invfac * inner_bounds
        c, d = loc + self._ppf(cdf_inner_bounds, *args) * scale

        # Do not silence warnings from integration.
        lbc = integrate.quad(fun, lb, c, **kwds)[0]
        cd = integrate.quad(fun, c, d, **kwds)[0]
        dub = integrate.quad(fun, d, ub, **kwds)[0]
        vals = (lbc + cd + dub)

        if conditional:
            vals /= invfac
        return np.array(vals)[()]  # make it a numpy scalar like other methods

    def _param_info(self):
        shape_info = self._shape_info()
        loc_info = _ShapeInfo("loc", False, (-np.inf, np.inf), (False, False))
        scale_info = _ShapeInfo("scale", False, (0, np.inf), (False, False))
        param_info = shape_info + [loc_info, scale_info]
        return param_info

    # For now, _delta_cdf is a private method.
    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        """
        Compute CDF(x2) - CDF(x1).

        Where x1 is greater than the median, compute SF(x1) - SF(x2),
        otherwise compute CDF(x2) - CDF(x1).

        This function is only useful if `dist.sf(x, ...)` has an implementation
        that is numerically more accurate than `1 - dist.cdf(x, ...)`.
        """
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        # Possible optimizations (needs investigation-these might not be
        # better):
        # * Use _lazywhere instead of np.where
        # * Instead of cdf1 > 0.5, compare x1 to the median.
        result = np.where(cdf1 > 0.5,
                          (self.sf(x1, *args, loc=loc, scale=scale)
                           - self.sf(x2, *args, loc=loc, scale=scale)),
                          self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
        if result.ndim == 0:
            result = result[()]
        return result


# Helpers for the discrete distributions
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""
    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)

    _a, _b = self._get_support(*args)
    return _expect(fun, _a, _b, self._ppf(0.5, *args), self.inc)


def _drv2_ppfsingle(self, q, *args):  # Use basic bisection algorithm
    _a, _b = self._get_support(*args)
    b = _b
    a = _a

    step = 10
    if isinf(b):            # Be sure ending point is > q
        b = float(max(100*q, 10))
        while 1:
            if b >= _b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if (qb < q):
                b += step
                step *= 2
            else:
                break
    else:
        qb = 1.0

    step = 10
    if isinf(a):    # be sure starting point < q
        a = float(min(-100*q, -10))
        while 1:
            if a <= _a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if (qa > q):
                a -= step
                step *= 2
            else:
                break
    else:
        qa = self._cdf(a, *args)

    if np.isinf(a) or np.isinf(b):
        message = "Arguments that bracket the requested quantile could not be found."
        raise RuntimeError(message)

    # maximum number of bisections within the normal float64s
    # maxiter = int(np.log2(finfo.max) - np.log2(finfo.smallest_normal))
    maxiter = 2046
    for i in range(maxiter):
        if (qa == q):
            return a
        if (qb == q):
            return b
        if b <= a+1:
            if qa > q:
                return a
            else:
                return b
        c = int((a+b)/2.0)
        qc = self._cdf(c, *args)
        if (qc < q):
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif (qc > q):
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c


# Must over-ride one of _pmf or _cdf or pass in
#  x_k, p(x_k) lists in initialization


class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing.

    `rv_discrete` is a base class to construct specific distribution classes
    and instances for discrete random variables. It can also be used
    to construct an arbitrary distribution defined by a list of support
    points and corresponding probabilities.

    Parameters
    ----------
    a : float, optional
        Lower bound of the support of the distribution, default: 0
    b : float, optional
        Upper bound of the support of the distribution, default: plus infinity
    moment_tol : float, optional
        The tolerance for the generic calculation of moments.
    values : tuple of two array_like, optional
        ``(xk, pk)`` where ``xk`` are integers and ``pk`` are the non-zero
        probabilities between 0 and 1 with ``sum(pk) = 1``. ``xk``
        and ``pk`` must have the same shape, and ``xk`` must be unique.
    inc : integer, optional
        Increment for the support of the distribution.
        Default is 1. (other values have not been tested)
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example "m, n" for a distribution
        that takes two integers as the two shape arguments for all its methods
        If not provided, shape parameters will be inferred from
        the signatures of the private methods, ``_pmf`` and ``_cdf`` of
        the instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pmf
    logpmf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    support

    Notes
    -----
    This class is similar to `rv_continuous`. Whether a shape parameter is
    valid is decided by an ``_argcheck`` method (which defaults to checking
    that its arguments are strictly positive.)
    The main differences are as follows.

    - The support of the distribution is a set of integers.
    - Instead of the probability density function, ``pdf`` (and the
      corresponding private ``_pdf``), this class defines the
      *probability mass function*, `pmf` (and the corresponding
      private ``_pmf``.)
    - There is no ``scale`` parameter.
    - The default implementations of methods (e.g. ``_cdf``) are not designed
      for distributions with support that is unbounded below (i.e.
      ``a=-np.inf``), so they must be overridden.

    To create a new discrete distribution, we would do the following:

    >>> from scipy.stats import rv_discrete
    >>> class poisson_gen(rv_discrete):
    ...     "Poisson distribution"
    ...     def _pmf(self, k, mu):
    ...         return exp(-mu) * mu**k / factorial(k)

    and create an instance::

    >>> poisson = poisson_gen(name="poisson")

    Note that above we defined the Poisson distribution in the standard form.
    Shifting the distribution can be done by providing the ``loc`` parameter
    to the methods of the instance. For example, ``poisson.pmf(x, mu, loc)``
    delegates the work to ``poisson._pmf(x-loc, mu)``.

    **Discrete distributions from a list of probabilities**

    Alternatively, you can construct an arbitrary discrete rv defined
    on a finite set of values ``xk`` with ``Prob{X=xk} = pk`` by using the
    ``values`` keyword argument to the `rv_discrete` constructor.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    Custom made discrete distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>> xk = np.arange(7)
    >>> pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    >>> custm = stats.rv_discrete(name='custm', values=(xk, pk))
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    >>> ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    >>> plt.show()

    Random number generation:

    >>> R = custm.rvs(size=100)

    """
    def __new__(cls, a=0, b=inf, name=None, badvalue=None,
                moment_tol=1e-8, values=None, inc=1, longname=None,
                shapes=None, seed=None):

        if values is not None:
            # dispatch to a subclass
            return super().__new__(rv_sample)
        else:
            # business as usual
            return super().__new__(cls)

    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, seed=None):

        super().__init__(seed)

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes

        if values is not None:
            raise ValueError("rv_discrete.__init__(..., values != None, ...)")

        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname)

    def __getstate__(self):
        dct = self.__dict__.copy()
        # these methods will be remade in __setstate__
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "generic_moment"]
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_discrete instance."""
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self.vecentropy = vectorize(self._entropy)

        # _attach_methods is responsible for calling _attach_argparser_methods
        self._attach_argparser_methods()

        # nin correction needs to be after we know numargs
        # correct nin for generic moment vectorization
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = types.MethodType(_vec_generic_moment, self)

        # correct nin for ppf vectorization
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = types.MethodType(_vppf, self)

        # now that self.numargs is defined, we can adjust nin
        self._cdfvec.nin = self.numargs + 1

    def _construct_docstrings(self, name, longname):
        if name is None:
            name = 'Distribution'
        self.name = name

        # generate docstring for subclass instances
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            docdict=docdict_discrete,
                                            discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))

            # discrete RV do not have the scale parameter, remove it
            self.__doc__ = self.__doc__.replace(
                '\n    scale : array_like, '
                'optional\n        scale parameter (default=1)', '')

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k-1, *args)

    def _logpmf(self, k, *args):
        with np.errstate(divide='ignore'):
            return log(self._pmf(k, *args))

    def _logpxf(self, k, *args):
        # continuous distributions have PDF, discrete have PMF, but sometimes
        # the distinction doesn't matter. This lets us use `_logpxf` for both
        # discrete and continuous distributions.
        return self._logpmf(k, *args)

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-1]
            scale = 1
            args = tuple(theta[:-1])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def _cdf_single(self, k, *args):
        _a, _b = self._get_support(*args)
        m = arange(int(_a), k+1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x).astype(np.float64)
        return self._cdfvec(k, *args)

    # generic _logcdf, _sf, _logsf, _ppf, _isf, _rvs defined in rv_generic

    def rvs(self, *args, **kwargs):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        size : int or tuple of ints, optional
            Defining number of random variates (Default is 1). Note that `size`
            has to be given as keyword, not as positional argument.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `random_state` is None (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance, that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        kwargs['discrete'] = True
        return super().rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond3 = np.isneginf(k)
        cond = cond0 & cond1 & np.isfinite(k)

        output = zeros(shape(cond), 'd')
        place(output, cond2*(cond0 == cond0), 1.0)
        place(output, cond3*(cond0 == cond0), 0.0)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 0.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = ((k < _a) | np.isneginf(k)) & cond0
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), _a-1 + loc)
        place(output, cond2, _b + loc)
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond3 = (q == 0) & cond0
        cond = cond0 & cond1

        # same problem as with ppf; copied from ppf and changed
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        # output type 'd' to handle nin and inf
        lower_bound = _a - 1 + loc
        upper_bound = _b + loc
        place(output, cond2*(cond == cond), lower_bound)
        place(output, cond3*(cond == cond), upper_bound)

        # call place only if at least 1 valid argument
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # PB same as ticket 766
            place(output, cond, self._isf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        if hasattr(self, 'pk'):
            return stats.entropy(self.pk)
        else:
            _a, _b = self._get_support(*args)
            return _expect(lambda x: entr(self._pmf(x, *args)),
                           _a, _b, self._ppf(0.5, *args), self.inc)

    def expect(self, func=None, args=(), loc=0, lb=None, ub=None,
               conditional=False, maxcount=1000, tolerance=1e-10, chunksize=32):
        """
        Calculate expected value of a function with respect to the distribution
        for discrete distribution by numerical summation.

        Parameters
        ----------
        func : callable, optional
            Function for which the expectation value is calculated.
            Takes only one argument.
            The default is the identity mapping f(k) = k.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter.
            Default is 0.
        lb, ub : int, optional
            Lower and upper bound for the summation, default is set to the
            support of the distribution, inclusive (``lb <= k <= ub``).
        conditional : bool, optional
            If true then the expectation is corrected by the conditional
            probability of the summation interval. The return value is the
            expectation of the function, `func`, conditional on being in
            the given interval (k such that ``lb <= k <= ub``).
            Default is False.
        maxcount : int, optional
            Maximal number of terms to evaluate (to avoid an endless loop for
            an infinite sum). Default is 1000.
        tolerance : float, optional
            Absolute tolerance for the summation. Default is 1e-10.
        chunksize : int, optional
            Iterate over the support of a distributions in chunks of this size.
            Default is 32.

        Returns
        -------
        expect : float
            Expected value.

        Notes
        -----
        For heavy-tailed distributions, the expected value may or
        may not exist,
        depending on the function, `func`. If it does exist, but the
        sum converges
        slowly, the accuracy of the result may be rather low. For instance, for
        ``zipf(4)``, accuracy for mean, variance in example is only 1e-5.
        increasing `maxcount` and/or `chunksize` may improve the result,
        but may also make zipf very slow.

        The function is not vectorized.

        """
        # Although `args` is just the shape parameters, `poisson_binom` needs this
        # to split the vector-valued shape into a tuple of separate shapes
        args, _, _ = self._parse_args(*args)

        if func is None:
            def fun(x):
                # loc and args from outer scope
                return (x+loc)*self._pmf(x, *args)
        else:
            def fun(x):
                # loc and args from outer scope
                return func(x+loc)*self._pmf(x, *args)
        # used pmf because _pmf does not check support in randint and there
        # might be problems(?) with correct self.a, self.b at this stage maybe
        # not anymore, seems to work now with _pmf

        _a, _b = self._get_support(*args)
        if lb is None:
            lb = _a
        else:
            lb = lb - loc   # convert bound for standardized distribution
        if ub is None:
            ub = _b
        else:
            ub = ub - loc   # convert bound for standardized distribution
        if conditional:
            invfac = self.sf(lb-1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0

        if isinstance(self, rv_sample):
            res = self._expect(fun, lb, ub)
            return res / invfac

        # iterate over the support, starting from the median
        x0 = self._ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac

    def _param_info(self):
        shape_info = self._shape_info()
        loc_info = _ShapeInfo("loc", True, (-np.inf, np.inf), (False, False))
        param_info = shape_info + [loc_info]
        return param_info


def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10,
            chunksize=32):
    """Helper for computing the expectation value of `fun`."""
    # short-circuit if the support size is small enough
    if (ub - lb) <= chunksize:
        supp = np.arange(lb, ub+1, inc)
        vals = fun(supp)
        return np.sum(vals)

    # otherwise, iterate starting from x0
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub

    count, tot = 0, 0.
    # iterate over [x0, ub] inclusive
    for x in _iter_chunked(x0, ub+1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge',
                          RuntimeWarning, stacklevel=3)
            return tot

    # iterate over [lb, x0)
    for x in _iter_chunked(x0-1, lb-1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge',
                          RuntimeWarning, stacklevel=3)
            break

    return tot


def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """Iterate from x0 to x1 in chunks of chunksize and steps inc.

    x0 must be finite, x1 need not be. In the latter case, the iterator is
    infinite.
    Handles both x0 < x1 and x0 > x1. In the latter case, iterates downwards
    (make sure to set inc < 0.)

    >>> from scipy.stats._distn_infrastructure import _iter_chunked
    >>> [x for x in _iter_chunked(2, 5, inc=2)]
    [array([2, 4])]
    >>> [x for x in _iter_chunked(2, 11, inc=2)]
    [array([2, 4, 6, 8]), array([10])]
    >>> [x for x in _iter_chunked(2, -5, inc=-2)]
    [array([ 2,  0, -2, -4])]
    >>> [x for x in _iter_chunked(2, -9, inc=-2)]
    [array([ 2,  0, -2, -4]), array([-6, -8])]

    """
    if inc == 0:
        raise ValueError('Cannot increment by zero.')
    if chunksize <= 0:
        raise ValueError(f'Chunk size must be positive; got {chunksize}.')

    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)

    x = np.copy(x0)
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp


class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

    The ctor ignores most of the arguments, only needs the `values` argument.
    """

    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, seed=None):

        super(rv_discrete, self).__init__(seed)

        if values is None:
            raise ValueError("rv_sample.__init__(..., values=None,...)")

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy

        xk, pk = values

        if np.shape(xk) != np.shape(pk):
            raise ValueError("xk and pk must have the same shape.")
        if np.less(pk, 0.0).any():
            raise ValueError("All elements of pk must be non-negative.")
        if not np.allclose(np.sum(pk), 1):
            raise ValueError("The sum of provided pk is not 1.")
        if not len(set(np.ravel(xk))) == np.size(xk):
            raise ValueError("xk may not contain duplicate values.")

        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]

        self.qvals = np.cumsum(self.pk, axis=0)

        self.shapes = ' '   # bypass inspection

        self._construct_argparser(meths_to_inspect=[self._pmf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')

        self._attach_methods()

        self._construct_docstrings(name, longname)

    def __getstate__(self):
        dct = self.__dict__.copy()

        # these methods will be remade in rv_generic.__setstate__,
        # which calls rv_generic._attach_methods
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs"]
        [dct.pop(attr, None) for attr in attrs]

        return dct

    def _attach_methods(self):
        """Attaches dynamically created argparser methods."""
        self._attach_argparser_methods()

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support.
        """
        return self.a, self.b

    def _pmf(self, x):
        return np.select([x == k for k in self.xk],
                         [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self, size=None, random_state=None):
        # Need to define it explicitly, otherwise .rvs() with size=None
        # fails due to explicit broadcasting in _ppf
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def _entropy(self):
        return stats.entropy(self.pk)

    def generic_moment(self, n):
        n = asarray(n)
        return np.sum(self.xk**n[np.newaxis, ...] * self.pk, axis=0)

    def _expect(self, fun, lb, ub, *args, **kwds):
        # ignore all args, just do a brute force summation
        supp = self.xk[(lb <= self.xk) & (self.xk <= ub)]
        vals = fun(supp)
        return np.sum(vals)


def _check_shape(argshape, size):
    """
    This is a utility function used by `_rvs()` in the class geninvgauss_gen.
    It compares the tuple argshape to the tuple size.

    Parameters
    ----------
    argshape : tuple of integers
        Shape of the arguments.
    size : tuple of integers or integer
        Size argument of rvs().

    Returns
    -------
    The function returns two tuples, scalar_shape and bc.

    scalar_shape : tuple
        Shape to which the 1-d array of random variates returned by
        _rvs_scalar() is converted when it is copied into the
        output array of _rvs().

    bc : tuple of booleans
        bc is an tuple the same length as size. bc[j] is True if the data
        associated with that index is generated in one call of _rvs_scalar().

    """
    scalar_shape = []
    bc = []
    for argdim, sizedim in zip_longest(argshape[::-1], size[::-1],
                                       fillvalue=1):
        if sizedim > argdim or (argdim == sizedim == 1):
            scalar_shape.append(sizedim)
            bc.append(True)
        else:
            bc.append(False)
    return tuple(scalar_shape[::-1]), tuple(bc[::-1])


def get_distribution_names(namespace_pairs, rv_base_class):
    """Collect names of statistical distributions and their generators.

    Parameters
    ----------
    namespace_pairs : sequence
        A snapshot of (name, value) pairs in the namespace of a module.
    rv_base_class : class
        The base class of random variable generator classes in a module.

    Returns
    -------
    distn_names : list of strings
        Names of the statistical distributions.
    distn_gen_names : list of strings
        Names of the generators of the statistical distributions.
        Note that these are not simply the names of the statistical
        distributions, with a _gen suffix added.

    """
    distn_names = []
    distn_gen_names = []
    for name, value in namespace_pairs:
        if name.startswith('_'):
            continue
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    return distn_names, distn_gen_names
<|endoftext|>"""
===================================================================
Statistical functions for masked arrays (:mod:`scipy.stats.mstats`)
===================================================================

.. currentmodule:: scipy.stats.mstats

This module contains a large number of statistical functions that can
be used with masked arrays.

Most of these functions are similar to those in `scipy.stats` but might
have small differences in the API or in the algorithm used. Since this
is a relatively new package, some API changes are still possible.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe
   gmean
   hmean
   kurtosis
   mode
   mquantiles
   hdmedian
   hdquantiles
   hdquantiles_sd
   idealfourths
   plotting_positions
   meppf
   moment
   skew
   tmean
   tvar
   tmin
   tmax
   tsem
   variation
   find_repeats
   sem
   trimmed_mean
   trimmed_mean_ci
   trimmed_std
   trimmed_var

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   scoreatpercentile

Correlation functions
=====================

.. autosummary::
   :toctree: generated/

   f_oneway
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   kendalltau_seasonal
   linregress
   siegelslopes
   theilslopes
   sen_seasonal_slopes

Statistical tests
=================

.. autosummary::
   :toctree: generated/

   ttest_1samp
   ttest_onesamp
   ttest_ind
   ttest_rel
   chisquare
   kstest
   ks_2samp
   ks_1samp
   ks_twosamp
   mannwhitneyu
   rankdata
   kruskal
   kruskalwallis
   friedmanchisquare
   brunnermunzel
   skewtest
   kurtosistest
   normaltest

Transformations
===============

.. autosummary::
   :toctree: generated/

   obrientransform
   trim
   trima
   trimmed_stde
   trimr
   trimtail
   trimboth
   winsorize
   zmap
   zscore

Other
=====

.. autosummary::
   :toctree: generated/

   argstoarray
   count_tied_groups
   msign
   compare_medians_ms
   median_cihs
   mjci
   mquantiles_cimj
   rsh

"""
from . import _mstats_basic
from . import _mstats_extras
from ._mstats_basic import *  # noqa: F403
from ._mstats_extras import *  # noqa: F403
# Functions that support masked array input in stats but need to be kept in the
# mstats namespace for backwards compatibility:
from scipy.stats import gmean, hmean, zmap, zscore, chisquare

__all__ = _mstats_basic.__all__ + _mstats_extras.__all__
__all__ += ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare']
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'compare_medians_ms',
    'hdquantiles', 'hdmedian', 'hdquantiles_sd',
    'idealfourths',
    'median_cihs','mjci','mquantiles_cimj',
    'rsh',
    'trimmed_mean_ci',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mstats_extras",
                                   private_modules=["_mstats_extras"], all=__all__,
                                   attribute=name, correct_module="mstats")
<|endoftext|>#-------------------------------------------------------------------------------
#
#  Define classes for (uni/multi)-variate kernel density estimation.
#
#  Currently, only Gaussian kernels are implemented.
#
#  Written by: Robert Kern
#
#  Date: 2004-08-09
#
#  Modified: 2005-02-10 by Robert Kern.
#              Contributed to SciPy
#            2005-10-07 by Robert Kern.
#              Some fixes to match the new scipy_core
#
#  Copyright 2004-2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

# Standard library imports.
import threading
import warnings

# SciPy imports.
from scipy import linalg, special
from scipy._lib._util import check_random_state

from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)
import numpy as np

# Local imports.
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log


__all__ = ['gaussian_kde']

MVN_LOCK = threading.Lock()


class gaussian_kde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.

        .. versionadded:: 1.2.0
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`. The square
        of `kde.factor` multiplies the covariance matrix of the data in the kde
        estimation.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
    __call__
    integrate_gaussian
    integrate_box_1d
    integrate_box
    integrate_kde
    pdf
    logpdf
    resample
    set_bandwidth
    covariance_factor

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),

    with ``neff`` the effective number of datapoints.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)

    as detailed in [5]_.

    `gaussian_kde` does not currently support data that lies in a
    lower-dimensional subspace of the space in which it is expressed. For such
    data, consider performing principal component analysis / dimensionality
    reduction and using `gaussian_kde` with the transformed data.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272

    Examples
    --------
    Generate some random two-dimensional data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> def measure(n):
    ...     "Measurement model, return two coupled measurements."
    ...     m1 = np.random.normal(size=n)
    ...     m2 = np.random.normal(scale=0.5, size=n)
    ...     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """
    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = atleast_2d(asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        # This can be converted to a warning once gh-10205 is resolved
        if self.d > self.n:
            msg = ("Number of dimensions is greater than number of samples. "
                   "This results in a singular data covariance matrix, which "
                   "cannot be treated using the algorithms implemented in "
                   "`gaussian_kde`. Note that `gaussian_kde` interprets each "
                   "*column* of `dataset` to be a point; consider transposing "
                   "the input to `dataset`.")
            raise ValueError(msg)

        try:
            self.set_bandwidth(bw_method=bw_method)
        except linalg.LinAlgError as e:
            msg = ("The data appears to lie in a lower-dimensional subspace "
                   "of the space in which it is expressed. This has resulted "
                   "in a singular data covariance matrix, which cannot be "
                   "treated using the algorithms implemented in "
                   "`gaussian_kde`. Consider performing principal component "
                   "analysis / dimensionality reduction and using "
                   "`gaussian_kde` with the transformed data.")
            raise linalg.LinAlgError(msg) from e

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = (f"points have dimension {d}, "
                       f"dataset has dimension {self.d}")
                raise ValueError(msg)

        output_dtype, spec = _get_output_dtype(self.covariance, points)
        result = gaussian_kernel_estimate[spec](
            self.dataset.T, self.weights[:, None],
            points.T, self.cho_cov, output_dtype)

        return result[:, 0]

    __call__ = evaluate

    def integrate_gaussian(self, mean, cov):
        """
        Multiply estimated density by a multivariate Gaussian and integrate
        over the whole space.

        Parameters
        ----------
        mean : aray_like
            A 1-D array, specifying the mean of the Gaussian.
        cov : array_like
            A 2-D array, specifying the covariance matrix of the Gaussian.

        Returns
        -------
        result : scalar
            The value of the integral.

        Raises
        ------
        ValueError
            If the mean or covariance of the input Gaussian differs from
            the KDE's dimensionality.

        """
        mean = atleast_1d(squeeze(mean))
        cov = atleast_2d(cov)

        if mean.shape != (self.d,):
            raise ValueError(f"mean does not have dimension {self.d}")
        if cov.shape != (self.d, self.d):
            raise ValueError(f"covariance does not have dimension {self.d}")

        # make mean a column vector
        mean = mean[:, newaxis]

        sum_cov = self.covariance + cov

        # This will raise LinAlgError if the new cov matrix is not s.p.d
        # cho_factor returns (ndarray, bool) where bool is a flag for whether
        # or not ndarray is upper or lower triangular
        sum_cov_chol = linalg.cho_factor(sum_cov)

        diff = self.dataset - mean
        tdiff = linalg.cho_solve(sum_cov_chol, diff)

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        energies = sum(diff * tdiff, axis=0) / 2.0
        result = sum(exp(-energies)*self.weights, axis=0) / norm_const

        return result

    def integrate_box_1d(self, low, high):
        """
        Computes the integral of a 1D pdf between two bounds.

        Parameters
        ----------
        low : scalar
            Lower bound of integration.
        high : scalar
            Upper bound of integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDE is over more than one dimension.

        """
        if self.d != 1:
            raise ValueError("integrate_box_1d() only handles 1D pdfs")

        stdev = ravel(sqrt(self.covariance))[0]

        normalized_low = ravel((low - self.dataset) / stdev)
        normalized_high = ravel((high - self.dataset) / stdev)

        value = np.sum(self.weights*(
                        special.ndtr(normalized_high) -
                        special.ndtr(normalized_low)))
        return value

    def integrate_box(self, low_bounds, high_bounds, maxpts=None):
        """Computes the integral of a pdf over a rectangular interval.

        Parameters
        ----------
        low_bounds : array_like
            A 1-D array containing the lower bounds of integration.
        high_bounds : array_like
            A 1-D array containing the upper bounds of integration.
        maxpts : int, optional
            The maximum number of points to use for integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        """
        if maxpts is not None:
            extra_kwds = {'maxpts': maxpts}
        else:
            extra_kwds = {}

        with MVN_LOCK:
            value, inform = _mvn.mvnun_weighted(low_bounds, high_bounds,
                                                self.dataset, self.weights,
                                                self.covariance, **extra_kwds)
        if inform:
            msg = f'An integral in _mvn.mvnun requires more points than {self.d * 1000}'
            warnings.warn(msg, stacklevel=2)

        return value

    def integrate_kde(self, other):
        """
        Computes the integral of the product of this  kernel density estimate
        with another.

        Parameters
        ----------
        other : gaussian_kde instance
            The other kde.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDEs have different dimensionality.

        """
        if other.d != self.d:
            raise ValueError("KDEs are not the same dimensionality")

        # we want to iterate over the smallest number of points
        if other.n < self.n:
            small = other
            large = self
        else:
            small = self
            large = other

        sum_cov = small.covariance + large.covariance
        sum_cov_chol = linalg.cho_factor(sum_cov)
        result = 0.0
        for i in range(small.n):
            mean = small.dataset[:, i, newaxis]
            diff = large.dataset - mean
            tdiff = linalg.cho_solve(sum_cov_chol, diff)

            energies = sum(diff * tdiff, axis=0) / 2.0
            result += sum(exp(-energies)*large.weights, axis=0)*small.weights[i]

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        result /= norm_const

        return result

    def resample(self, size=None, seed=None):
        """Randomly sample a dataset from the estimated pdf.

        Parameters
        ----------
        size : int, optional
            The number of samples to draw.  If not provided, then the size is
            the same as the effective number of samples in the underlying
            dataset.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.

        Returns
        -------
        resample : (self.d, `size`) ndarray
            The sampled dataset.

        """ # numpy/numpydoc#87  # noqa: E501
        if size is None:
            size = int(self.neff)

        random_state = check_random_state(seed)
        norm = transpose(random_state.multivariate_normal(
            zeros((self.d,), float), self.covariance, size=size
        ))
        indices = random_state.choice(self.n, size=size, p=self.weights)
        means = self.dataset[:, indices]

        return means + norm

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s : float
            Scott's factor.
        """
        return power(self.neff, -1./(self.d+4))

    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        return power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = """Computes the coefficient (`kde.factor`) that
        multiplies the data covariance matrix to obtain the kernel covariance
        matrix. The default is `scotts_factor`.  A subclass can overwrite this
        method to provide a different method, or set it through a call to
        `kde.set_bandwidth`."""

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_cho_cov = linalg.cholesky(self._data_covariance,
                                                 lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*pi))).sum()

    @property
    def inv_cov(self):
        # Re-compute from scratch each time because I'm not sure how this is
        # used in the wild. (Perhaps users change the `dataset`, since it's
        # not a private attribute?) `_compute_covariance` used to recalculate
        # all these, so we'll recalculate everything now that this is a
        # a property.
        self.factor = self.covariance_factor()
        self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                           bias=False, aweights=self.weights))
        return linalg.inv(self._data_covariance) / self.factor**2

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        return self.evaluate(x)

    def logpdf(self, x):
        """
        Evaluate the log of the estimated pdf on a provided set of points.
        """
        points = atleast_2d(x)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = (f"points have dimension {d}, "
                       f"dataset has dimension {self.d}")
                raise ValueError(msg)

        output_dtype, spec = _get_output_dtype(self.covariance, points)
        result = gaussian_kernel_estimate_log[spec](
            self.dataset.T, self.weights[:, None],
            points.T, self.cho_cov, output_dtype)

        return result[:, 0]

    def marginal(self, dimensions):
        """Return a marginal KDE distribution

        Parameters
        ----------
        dimensions : int or 1-d array_like
            The dimensions of the multivariate distribution corresponding
            with the marginal variables, that is, the indices of the dimensions
            that are being retained. The other dimensions are marginalized out.

        Returns
        -------
        marginal_kde : gaussian_kde
            An object representing the marginal distribution.

        Notes
        -----
        .. versionadded:: 1.10.0

        """

        dims = np.atleast_1d(dimensions)

        if not np.issubdtype(dims.dtype, np.integer):
            msg = ("Elements of `dimensions` must be integers - the indices "
                   "of the marginal variables being retained.")
            raise ValueError(msg)

        n = len(self.dataset)  # number of dimensions
        original_dims = dims.copy()

        dims[dims < 0] = n + dims[dims < 0]

        if len(np.unique(dims)) != len(dims):
            msg = ("All elements of `dimensions` must be unique.")
            raise ValueError(msg)

        i_invalid = (dims < 0) | (dims >= n)
        if np.any(i_invalid):
            msg = (f"Dimensions {original_dims[i_invalid]} are invalid "
                   f"for a distribution in {n} dimensions.")
            raise ValueError(msg)

        dataset = self.dataset[dims]
        weights = self.weights

        return gaussian_kde(dataset, bw_method=self.covariance_factor(),
                            weights=weights)

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = ones(self.n)/self.n
            return self._weights

    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff


def _get_output_dtype(covariance, points):
    """
    Calculates the output dtype and the "spec" (=C type name).

    This was necessary in order to deal with the fused types in the Cython
    routine `gaussian_kernel_estimate`. See gh-10824 for details.
    """
    output_dtype = np.common_type(covariance, points)
    itemsize = np.dtype(output_dtype).itemsize
    if itemsize == 4:
        spec = 'float'
    elif itemsize == 8:
        spec = 'double'
    elif itemsize in (12, 16):
        spec = 'long double'
    else:
        raise ValueError(
                f"{output_dtype} has unexpected item size: {itemsize}"
            )

    return output_dtype, spec
<|endoftext|>from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval


class BinomTestResult:
    """
    Result of `scipy.stats.binomtest`.

    Attributes
    ----------
    k : int
        The number of successes (copied from `binomtest` input).
    n : int
        The number of trials (copied from `binomtest` input).
    alternative : str
        Indicates the alternative hypothesis specified in the input
        to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
        or ``'less'``.
    statistic: float
        The estimate of the proportion of successes.
    pvalue : float
        The p-value of the hypothesis test.

    """
    def __init__(self, k, n, alternative, statistic, pvalue):
        self.k = k
        self.n = n
        self.alternative = alternative
        self.statistic = statistic
        self.pvalue = pvalue

        # add alias for backward compatibility
        self.proportion_estimate = statistic

    def __repr__(self):
        s = ("BinomTestResult("
             f"k={self.k}, "
             f"n={self.n}, "
             f"alternative={self.alternative!r}, "
             f"statistic={self.statistic}, "
             f"pvalue={self.pvalue})")
        return s

    def proportion_ci(self, confidence_level=0.95, method='exact'):
        """
        Compute the confidence interval for ``statistic``.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is 0.95.
        method : {'exact', 'wilson', 'wilsoncc'}, optional
            Selects the method used to compute the confidence interval
            for the estimate of the proportion:

            'exact' :
                Use the Clopper-Pearson exact method [1]_.
            'wilson' :
                Wilson's method, without continuity correction ([2]_, [3]_).
            'wilsoncc' :
                Wilson's method, with continuity correction ([2]_, [3]_).

            Default is ``'exact'``.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        References
        ----------
        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or
               fiducial limits illustrated in the case of the binomial,
               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).
        .. [2] E. B. Wilson, Probable inference, the law of succession, and
               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212
               (1927).
        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the
               single proportion: comparison of seven methods, Statistics
               in Medicine, 17, pp 857-872 (1998).

        Examples
        --------
        >>> from scipy.stats import binomtest
        >>> result = binomtest(k=7, n=50, p=0.1)
        >>> result.statistic
        0.14
        >>> result.proportion_ci()
        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)
        """
        if method not in ('exact', 'wilson', 'wilsoncc'):
            raise ValueError(f"method ('{method}') must be one of 'exact', "
                             "'wilson' or 'wilsoncc'.")
        if not (0 <= confidence_level <= 1):
            raise ValueError(f'confidence_level ({confidence_level}) must be in '
                             'the interval [0, 1].')
        if method == 'exact':
            low, high = _binom_exact_conf_int(self.k, self.n,
                                              confidence_level,
                                              self.alternative)
        else:
            # method is 'wilson' or 'wilsoncc'
            low, high = _binom_wilson_conf_int(self.k, self.n,
                                               confidence_level,
                                               self.alternative,
                                               correction=method == 'wilsoncc')
        return ConfidenceInterval(low=low, high=high)


def _findp(func):
    try:
        p = brentq(func, 0, 1)
    except RuntimeError:
        raise RuntimeError('numerical solver failed to converge when '
                           'computing the confidence limits') from None
    except ValueError as exc:
        raise ValueError('brentq raised a ValueError; report this to the '
                         'SciPy developers') from exc
    return p


def _binom_exact_conf_int(k, n, confidence_level, alternative):
    """
    Compute the estimate and confidence interval for the binomial test.

    Returns proportion, prop_low, prop_high
    """
    if alternative == 'two-sided':
        alpha = (1 - confidence_level) / 2
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'less':
        alpha = 1 - confidence_level
        plow = 0.0
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'greater':
        alpha = 1 - confidence_level
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        phigh = 1.0
    return plow, phigh


def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    # This function assumes that the arguments have already been validated.
    # In particular, `alternative` must be one of 'two-sided', 'less' or
    # 'greater'.
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5*confidence_level)
    else:
        z = ndtri(confidence_level)

    # For reference, the formulas implemented here are from
    # Newcombe (1998) (ref. [3] in the proportion_ci docstring).
    denom = 2*(n + z**2)
    center = (2*n*p + z**2)/denom
    q = 1 - p
    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (1 + z*sqrt(z**2 - 2 - 1/n + 4*p*(n*q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (1 + z*sqrt(z**2 + 2 - 1/n + 4*p*(n*q - 1))) / denom
            hi = center + dhi
    else:
        delta = z/denom * sqrt(4*n*p*q + z**2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta

    return lo, hi


def binomtest(k, n, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    The binomial test [1]_ is a test of the null hypothesis that the
    probability of success in a Bernoulli experiment is `p`.

    Details of the test can be found in many texts on statistics, such
    as section 24.5 of [2]_.

    Parameters
    ----------
    k : int
        The number of successes.
    n : int
        The number of trials.
    p : float, optional
        The hypothesized probability of success, i.e. the expected
        proportion of successes.  The value must be in the interval
        ``0 <= p <= 1``. The default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.

    Returns
    -------
    result : `~scipy.stats._result_classes.BinomTestResult` instance
        The return value is an object with the following attributes:

        k : int
            The number of successes (copied from `binomtest` input).
        n : int
            The number of trials (copied from `binomtest` input).
        alternative : str
            Indicates the alternative hypothesis specified in the input
            to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
            or ``'less'``.
        statistic : float
            The estimate of the proportion of successes.
        pvalue : float
            The p-value of the hypothesis test.

        The object has the following methods:

        proportion_ci(confidence_level=0.95, method='exact') :
            Compute the confidence interval for ``statistic``.

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Binomial test, https://en.wikipedia.org/wiki/Binomial_test
    .. [2] Jerrold H. Zar, Biostatistical Analysis (fifth edition),
           Prentice Hall, Upper Saddle River, New Jersey USA (2010)

    Examples
    --------
    >>> from scipy.stats import binomtest

    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:

    >>> result = binomtest(3, n=15, p=0.1, alternative='greater')
    >>> result.pvalue
    0.18406106910639114

    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.

    The test statistic is equal to the estimated proportion, which is simply
    ``3/15``:

    >>> result.statistic
    0.2

    We can use the `proportion_ci()` method of the result to compute the
    confidence interval of the estimate:

    >>> result.proportion_ci(confidence_level=0.95)
    ConfidenceInterval(low=0.05684686759024681, high=1.0)

    """
    k = _validate_int(k, 'k', minimum=0)
    n = _validate_int(n, 'n', minimum=1)
    if k > n:
        raise ValueError(f'k ({k}) must not be greater than n ({n}).')

    if not (0 <= p <= 1):
        raise ValueError(f"p ({p}) must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError(f"alternative ('{alternative}') not recognized; \n"
                         "must be 'two-sided', 'less' or 'greater'")
    if alternative == 'less':
        pval = binom.cdf(k, n, p)
    elif alternative == 'greater':
        pval = binom.sf(k-1, n, p)
    else:
        # alternative is 'two-sided'
        d = binom.pmf(k, n, p)
        rerr = 1 + 1e-7
        if k == p * n:
            # special case as shortcut, would also be handled by `else` below
            pval = 1.
        elif k < p * n:
            ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p),
                                              -d*rerr, np.ceil(p * n), n)
            # y is the number of terms between mode and n that are <= d*rerr.
            # ix gave us the first term where a(ix) <= d*rerr < a(ix-1)
            # if the first equality doesn't hold, y=n-ix. Otherwise, we
            # need to include ix as well as the equality holds. Note that
            # the equality will hold in very very rare situations due to rerr.
            y = n - ix + int(d*rerr == binom.pmf(ix, n, p))
            pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
        else:
            ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p),
                                              d*rerr, 0, np.floor(p * n))
            # y is the number of terms between 0 and mode that are <= d*rerr.
            # we need to add a 1 to account for the 0 index.
            # For comparing this with old behavior, see
            # tst_binary_srch_for_binom_tst method in test_morestats.
            y = ix + 1
            pval = binom.cdf(y-1, n, p) + binom.sf(k-1, n, p)

        pval = min(1.0, pval)

    result = BinomTestResult(k=k, n=n, alternative=alternative,
                             statistic=k/n, pvalue=pval)
    return result


def _binary_search_for_binom_tst(a, d, lo, hi):
    """
    Conducts an implicit binary search on a function specified by `a`.

    Meant to be used on the binomial PMF for the case of two-sided tests
    to obtain the value on the other side of the mode where the tail
    probability should be computed. The values on either side of
    the mode are always in order, meaning binary search is applicable.

    Parameters
    ----------
    a : callable
      The function over which to perform binary search. Its values
      for inputs lo and hi should be in ascending order.
    d : float
      The value to search.
    lo : int
      The lower end of range to search.
    hi : int
      The higher end of the range to search.

    Returns
    -------
    int
      The index, i between lo and hi
      such that a(i)<=d<a(i+1)
    """
    while lo < hi:
        mid = lo + (hi-lo)//2
        midval = a(mid)
        if midval < d:
            lo = mid+1
        elif midval > d:
            hi = mid-1
        else:
            return mid
    if a(lo) <= d:
        return lo
    else:
        return lo-1
<|endoftext|>"""
Created on Fri Apr  2 09:06:05 2021

@author: matth
"""

import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
from scipy._lib._array_api import array_namespace, xp_moveaxis_to_end

__all__ = ['entropy', 'differential_entropy']


@_axis_nan_policy_factory(
    lambda x: x,
    n_samples=lambda kwgs: (
        2 if ("qk" in kwgs and kwgs["qk"] is not None)
        else 1
    ),
    n_outputs=1, result_to_tuple=lambda x: (x,), paired=True,
    too_small=-1  # entropy doesn't have too small inputs
)
def entropy(pk: np.typing.ArrayLike,
            qk: np.typing.ArrayLike | None = None,
            base: float | None = None,
            axis: int = 0
            ) -> np.number | np.ndarray:
    """
    Calculate the Shannon entropy/relative entropy of given distribution(s).

    If only probabilities `pk` are given, the Shannon entropy is calculated as
    ``H = -sum(pk * log(pk))``.

    If `qk` is not None, then compute the relative entropy
    ``D = sum(pk * log(pk / qk))``. This quantity is also known
    as the Kullback-Leibler divergence.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : array_like
        Defines the (discrete) distribution. Along each axis-slice of ``pk``,
        element ``i`` is the  (possibly unnormalized) probability of event
        ``i``.
    qk : array_like, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis : int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : {float, array_like}
        The calculated entropy.

    Notes
    -----
    Informally, the Shannon entropy quantifies the expected uncertainty
    inherent in the possible outcomes of a discrete random variable.
    For example,
    if messages consisting of sequences of symbols from a set are to be
    encoded and transmitted over a noiseless channel, then the Shannon entropy
    ``H(pk)`` gives a tight lower bound for the average number of units of
    information needed per symbol if the symbols occur with frequencies
    governed by the discrete distribution `pk` [1]_. The choice of base
    determines the choice of units; e.g., ``e`` for nats, ``2`` for bits, etc.

    The relative entropy, ``D(pk|qk)``, quantifies the increase in the average
    number of units of information needed per symbol if the encoding is
    optimized for the probability distribution `qk` instead of the true
    distribution `pk`. Informally, the relative entropy quantifies the expected
    excess in surprise experienced if one believes the true distribution is
    `qk` when it is actually `pk`.

    A related quantity, the cross entropy ``CE(pk, qk)``, satisfies the
    equation ``CE(pk, qk) = H(pk) + D(pk|qk)`` and can also be calculated with
    the formula ``CE = -sum(pk * log(qk))``. It gives the average
    number of units of information needed per symbol if an encoding is
    optimized for the probability distribution `qk` when the true distribution
    is `pk`. It is not computed directly by `entropy`, but it can be computed
    using two calls to the function (see Examples).

    See [2]_ for more information.

    References
    ----------
    .. [1] Shannon, C.E. (1948), A Mathematical Theory of Communication.
           Bell System Technical Journal, 27: 379-423.
           https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
    .. [2] Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information
           Theory (Wiley Series in Telecommunications and Signal Processing).
           Wiley-Interscience, USA.


    Examples
    --------
    The outcome of a fair coin is the most uncertain:

    >>> import numpy as np
    >>> from scipy.stats import entropy
    >>> base = 2  # work in units of bits
    >>> pk = np.array([1/2, 1/2])  # fair coin
    >>> H = entropy(pk, base=base)
    >>> H
    1.0
    >>> H == -np.sum(pk * np.log(pk)) / np.log(base)
    True

    The outcome of a biased coin is less uncertain:

    >>> qk = np.array([9/10, 1/10])  # biased coin
    >>> entropy(qk, base=base)
    0.46899559358928117

    The relative entropy between the fair coin and biased coin is calculated
    as:

    >>> D = entropy(pk, qk, base=base)
    >>> D
    0.7369655941662062
    >>> np.isclose(D, np.sum(pk * np.log(pk/qk)) / np.log(base), rtol=4e-16, atol=0)
    True

    The cross entropy can be calculated as the sum of the entropy and
    relative entropy`:

    >>> CE = entropy(pk, base=base) + entropy(pk, qk, base=base)
    >>> CE
    1.736965594166206
    >>> CE == -np.sum(pk * np.log(qk)) / np.log(base)
    True

    """
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    xp = array_namespace(pk) if qk is None else array_namespace(pk, qk)

    pk = xp.asarray(pk)
    with np.errstate(invalid='ignore'):
        pk = 1.0*pk / xp.sum(pk, axis=axis, keepdims=True)  # type: ignore[operator]
    if qk is None:
        vec = special.entr(pk)
    else:
        qk = xp.asarray(qk)
        pk, qk = _broadcast_arrays((pk, qk), axis=None, xp=xp)  # don't ignore any axes
        sum_kwargs = dict(axis=axis, keepdims=True)
        qk = 1.0*qk / xp.sum(qk, **sum_kwargs)  # type: ignore[operator, call-overload]
        vec = special.rel_entr(pk, qk)
    S = xp.sum(vec, axis=axis)
    if base is not None:
        S /= math.log(base)
    return S


def _differential_entropy_is_too_small(samples, kwargs, axis=-1):
    values = samples[0]
    n = values.shape[axis]
    window_length = kwargs.get("window_length",
                               math.floor(math.sqrt(n) + 0.5))
    if not 2 <= 2 * window_length < n:
        return True
    return False


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,),
    too_small=_differential_entropy_is_too_small
)
def differential_entropy(
    values: np.typing.ArrayLike,
    *,
    window_length: int | None = None,
    base: float | None = None,
    axis: int = 0,
    method: str = "auto",
) -> np.number | np.ndarray:
    r"""Given a sample of a distribution, estimate the differential entropy.

    Several estimation methods are available using the `method` parameter. By
    default, a method is selected based the size of the sample.

    Parameters
    ----------
    values : sequence
        Sample from a continuous distribution.
    window_length : int, optional
        Window length for computing Vasicek estimate. Must be an integer
        between 1 and half of the sample size. If ``None`` (the default), it
        uses the heuristic value

        .. math::
            \left \lfloor \sqrt{n} + 0.5 \right \rfloor

        where :math:`n` is the sample size. This heuristic was originally
        proposed in [2]_ and has become common in the literature.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis : int, optional
        The axis along which the differential entropy is calculated.
        Default is 0.
    method : {'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'}, optional
        The method used to estimate the differential entropy from the sample.
        Default is ``'auto'``.  See Notes for more information.

    Returns
    -------
    entropy : float
        The calculated differential entropy.

    Notes
    -----
    This function will converge to the true differential entropy in the limit

    .. math::
        n \to \infty, \quad m \to \infty, \quad \frac{m}{n} \to 0

    The optimal choice of ``window_length`` for a given sample size depends on
    the (unknown) distribution. Typically, the smoother the density of the
    distribution, the larger the optimal value of ``window_length`` [1]_.

    The following options are available for the `method` parameter.

    * ``'vasicek'`` uses the estimator presented in [1]_. This is
      one of the first and most influential estimators of differential entropy.
    * ``'van es'`` uses the bias-corrected estimator presented in [3]_, which
      is not only consistent but, under some conditions, asymptotically normal.
    * ``'ebrahimi'`` uses an estimator presented in [4]_, which was shown
      in simulation to have smaller bias and mean squared error than
      the Vasicek estimator.
    * ``'correa'`` uses the estimator presented in [5]_ based on local linear
      regression. In a simulation study, it had consistently smaller mean
      square error than the Vasiceck estimator, but it is more expensive to
      compute.
    * ``'auto'`` selects the method automatically (default). Currently,
      this selects ``'van es'`` for very small samples (<10), ``'ebrahimi'``
      for moderate sample sizes (11-1000), and ``'vasicek'`` for larger
      samples, but this behavior is subject to change in future versions.

    All estimators are implemented as described in [6]_.

    References
    ----------
    .. [1] Vasicek, O. (1976). A test for normality based on sample entropy.
           Journal of the Royal Statistical Society:
           Series B (Methodological), 38(1), 54-59.
    .. [2] Crzcgorzewski, P., & Wirczorkowski, R. (1999). Entropy-based
           goodness-of-fit test for exponentiality. Communications in
           Statistics-Theory and Methods, 28(5), 1183-1202.
    .. [3] Van Es, B. (1992). Estimating functionals related to a density by a
           class of statistics based on spacings. Scandinavian Journal of
           Statistics, 61-72.
    .. [4] Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). Two measures
           of sample entropy. Statistics & Probability Letters, 20(3), 225-234.
    .. [5] Correa, J. C. (1995). A new estimator of entropy. Communications
           in Statistics-Theory and Methods, 24(10), 2439-2449.
    .. [6] Noughabi, H. A. (2015). Entropy Estimation Using Numerical Methods.
           Annals of Data Science, 2(2), 231-241.
           https://link.springer.com/article/10.1007/s40745-015-0045-9

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import differential_entropy, norm

    Entropy of a standard normal distribution:

    >>> rng = np.random.default_rng()
    >>> values = rng.standard_normal(100)
    >>> differential_entropy(values)
    1.3407817436640392

    Compare with the true entropy:

    >>> float(norm.entropy())
    1.4189385332046727

    For several sample sizes between 5 and 1000, compare the accuracy of
    the ``'vasicek'``, ``'van es'``, and ``'ebrahimi'`` methods. Specifically,
    compare the root mean squared error (over 1000 trials) between the estimate
    and the true differential entropy of the distribution.

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>>
    >>>
    >>> def rmse(res, expected):
    ...     '''Root mean squared error'''
    ...     return np.sqrt(np.mean((res - expected)**2))
    >>>
    >>>
    >>> a, b = np.log10(5), np.log10(1000)
    >>> ns = np.round(np.logspace(a, b, 10)).astype(int)
    >>> reps = 1000  # number of repetitions for each sample size
    >>> expected = stats.expon.entropy()
    >>>
    >>> method_errors = {'vasicek': [], 'van es': [], 'ebrahimi': []}
    >>> for method in method_errors:
    ...     for n in ns:
    ...        rvs = stats.expon.rvs(size=(reps, n), random_state=rng)
    ...        res = stats.differential_entropy(rvs, method=method, axis=-1)
    ...        error = rmse(res, expected)
    ...        method_errors[method].append(error)
    >>>
    >>> for method, errors in method_errors.items():
    ...     plt.loglog(ns, errors, label=method)
    >>>
    >>> plt.legend()
    >>> plt.xlabel('sample size')
    >>> plt.ylabel('RMSE (1000 trials)')
    >>> plt.title('Entropy Estimator Error (Exponential Distribution)')

    """
    xp = array_namespace(values)
    values = xp.asarray(values)
    if xp.isdtype(values.dtype, "integral"):  # type: ignore[union-attr]
        values = xp.astype(values, xp.asarray(1.).dtype)
    values = xp_moveaxis_to_end(values, axis, xp=xp)
    n = values.shape[-1]  # type: ignore[union-attr]

    if window_length is None:
        window_length = math.floor(math.sqrt(n) + 0.5)

    if not 2 <= 2 * window_length < n:
        raise ValueError(
            f"Window length ({window_length}) must be positive and less "
            f"than half the sample size ({n}).",
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    sorted_data = xp.sort(values, axis=-1)

    methods = {"vasicek": _vasicek_entropy,
               "van es": _van_es_entropy,
               "correa": _correa_entropy,
               "ebrahimi": _ebrahimi_entropy,
               "auto": _vasicek_entropy}
    method = method.lower()
    if method not in methods:
        message = f"`method` must be one of {set(methods)}"
        raise ValueError(message)

    if method == "auto":
        if n <= 10:
            method = 'van es'
        elif n <= 1000:
            method = 'ebrahimi'
        else:
            method = 'vasicek'

    res = methods[method](sorted_data, window_length, xp=xp)

    if base is not None:
        res /= math.log(base)

    # avoid dtype changes due to data-apis/array-api-compat#152
    # can be removed when data-apis/array-api-compat#152 is resolved
    return xp.astype(res, values.dtype)  # type: ignore[union-attr]


def _pad_along_last_axis(X, m, *, xp):
    """Pad the data for computing the rolling window difference."""
    # scales a  bit better than method in _vasicek_like_entropy
    shape = X.shape[:-1] + (m,)
    Xl = xp.broadcast_to(X[..., :1], shape)  # :1 vs 0 to maintain shape
    Xr = xp.broadcast_to(X[..., -1:], shape)
    return xp.concat((Xl, X, Xr), axis=-1)


def _vasicek_entropy(X, m, *, xp):
    """Compute the Vasicek estimator as described in [6] Eq. 1.3."""
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m, xp=xp)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    logs = xp.log(n/(2*m) * differences)
    return xp.mean(logs, axis=-1)


def _van_es_entropy(X, m, *, xp):
    """Compute the van Es estimator as described in [6]."""
    # No equation number, but referred to as HVE_mn.
    # Typo: there should be a log within the summation.
    n = X.shape[-1]
    difference = X[..., m:] - X[..., :-m]
    term1 = 1/(n-m) * xp.sum(xp.log((n+1)/m * difference), axis=-1)
    k = xp.arange(m, n+1, dtype=term1.dtype)
    return term1 + xp.sum(1/k) + math.log(m) - math.log(n+1)


def _ebrahimi_entropy(X, m, *, xp):
    """Compute the Ebrahimi estimator as described in [6]."""
    # No equation number, but referred to as HE_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m, xp=xp)

    differences = X[..., 2 * m:] - X[..., : -2 * m:]

    i = xp.arange(1, n+1, dtype=X.dtype)
    ci = xp.ones_like(i)*2
    ci[i <= m] = 1 + (i[i <= m] - 1)/m
    ci[i >= n - m + 1] = 1 + (n - i[i >= n-m+1])/m

    logs = xp.log(n * differences / (ci * m))
    return xp.mean(logs, axis=-1)


def _correa_entropy(X, m, *, xp):
    """Compute the Correa estimator as described in [6]."""
    # No equation number, but referred to as HC_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m, xp=xp)

    i = xp.arange(1, n+1)
    dj = xp.arange(-m, m+1)[:, None]
    j = i + dj
    j0 = j + m - 1  # 0-indexed version of j

    Xibar = xp.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = xp.sum(difference*dj, axis=-2)  # dj is d-i
    den = n*xp.sum(difference**2, axis=-2)
    return -xp.mean(xp.log(num/den), axis=-1)
<|endoftext|>import numpy as np


def _validate_1d(a, name, allow_inf=False):
    if np.ndim(a) != 1:
        raise ValueError(f'`{name}` must be a one-dimensional sequence.')
    if np.isnan(a).any():
        raise ValueError(f'`{name}` must not contain nan.')
    if not allow_inf and np.isinf(a).any():
        raise ValueError(f'`{name}` must contain only finite values.')


def _validate_interval(interval):
    interval = np.asarray(interval)
    if interval.shape == (0,):
        # The input was a sequence with length 0.
        interval = interval.reshape((0, 2))
    if interval.ndim != 2 or interval.shape[-1] != 2:
        raise ValueError('`interval` must be a two-dimensional array with '
                         'shape (m, 2), where m is the number of '
                         'interval-censored values, but got shape '
                         f'{interval.shape}')

    if np.isnan(interval).any():
        raise ValueError('`interval` must not contain nan.')
    if np.isinf(interval).all(axis=1).any():
        raise ValueError('In each row in `interval`, both values must not'
                         ' be infinite.')
    if (interval[:, 0] > interval[:, 1]).any():
        raise ValueError('In each row of `interval`, the left value must not'
                         ' exceed the right value.')

    uncensored_mask = interval[:, 0] == interval[:, 1]
    left_mask = np.isinf(interval[:, 0])
    right_mask = np.isinf(interval[:, 1])
    interval_mask = np.isfinite(interval).all(axis=1) & ~uncensored_mask

    uncensored2 = interval[uncensored_mask, 0]
    left2 = interval[left_mask, 1]
    right2 = interval[right_mask, 0]
    interval2 = interval[interval_mask]

    return uncensored2, left2, right2, interval2


def _validate_x_censored(x, censored):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('`x` must be one-dimensional.')
    censored = np.asarray(censored)
    if censored.ndim != 1:
        raise ValueError('`censored` must be one-dimensional.')
    if (~np.isfinite(x)).any():
        raise ValueError('`x` must not contain nan or inf.')
    if censored.size != x.size:
        raise ValueError('`x` and `censored` must have the same length.')
    return x, censored.astype(bool)


class CensoredData:
    """
    Instances of this class represent censored data.

    Instances may be passed to the ``fit`` method of continuous
    univariate SciPy distributions for maximum likelihood estimation.
    The *only* method of the univariate continuous distributions that
    understands `CensoredData` is the ``fit`` method.  An instance of
    `CensoredData` can not be passed to methods such as ``pdf`` and
    ``cdf``.

    An observation is said to be *censored* when the precise value is unknown,
    but it has a known upper and/or lower bound.  The conventional terminology
    is:

    * left-censored: an observation is below a certain value but it is
      unknown by how much.
    * right-censored: an observation is above a certain value but it is
      unknown by how much.
    * interval-censored: an observation lies somewhere on an interval between
      two values.

    Left-, right-, and interval-censored data can be represented by
    `CensoredData`.

    For convenience, the class methods ``left_censored`` and
    ``right_censored`` are provided to create a `CensoredData`
    instance from a single one-dimensional array of measurements
    and a corresponding boolean array to indicate which measurements
    are censored.  The class method ``interval_censored`` accepts two
    one-dimensional arrays that hold the lower and upper bounds of the
    intervals.

    Parameters
    ----------
    uncensored : array_like, 1D
        Uncensored observations.
    left : array_like, 1D
        Left-censored observations.
    right : array_like, 1D
        Right-censored observations.
    interval : array_like, 2D, with shape (m, 2)
        Interval-censored observations.  Each row ``interval[k, :]``
        represents the interval for the kth interval-censored observation.

    Notes
    -----
    In the input array `interval`, the lower bound of the interval may
    be ``-inf``, and the upper bound may be ``inf``, but at least one must be
    finite. When the lower bound is ``-inf``, the row represents a left-
    censored observation, and when the upper bound is ``inf``, the row
    represents a right-censored observation.  If the length of an interval
    is 0 (i.e. ``interval[k, 0] == interval[k, 1]``, the observation is
    treated as uncensored.  So one can represent all the types of censored
    and uncensored data in ``interval``, but it is generally more convenient
    to use `uncensored`, `left` and `right` for uncensored, left-censored and
    right-censored observations, respectively.

    Examples
    --------
    In the most general case, a censored data set may contain values that
    are left-censored, right-censored, interval-censored, and uncensored.
    For example, here we create a data set with five observations.  Two
    are uncensored (values 1 and 1.5), one is a left-censored observation
    of 0, one is a right-censored observation of 10 and one is
    interval-censored in the interval [2, 3].

    >>> import numpy as np
    >>> from scipy.stats import CensoredData
    >>> data = CensoredData(uncensored=[1, 1.5], left=[0], right=[10],
    ...                     interval=[[2, 3]])
    >>> print(data)
    CensoredData(5 values: 2 not censored, 1 left-censored,
    1 right-censored, 1 interval-censored)

    Equivalently,

    >>> data = CensoredData(interval=[[1, 1],
    ...                               [1.5, 1.5],
    ...                               [-np.inf, 0],
    ...                               [10, np.inf],
    ...                               [2, 3]])
    >>> print(data)
    CensoredData(5 values: 2 not censored, 1 left-censored,
    1 right-censored, 1 interval-censored)

    A common case is to have a mix of uncensored observations and censored
    observations that are all right-censored (or all left-censored). For
    example, consider an experiment in which six devices are started at
    various times and left running until they fail.  Assume that time is
    measured in hours, and the experiment is stopped after 30 hours, even
    if all the devices have not failed by that time.  We might end up with
    data such as this::

        Device  Start-time  Fail-time  Time-to-failure
           1         0         13           13
           2         2         24           22
           3         5         22           17
           4         8         23           15
           5        10        ***          >20
           6        12        ***          >18

    Two of the devices had not failed when the experiment was stopped;
    the observations of the time-to-failure for these two devices are
    right-censored.  We can represent this data with

    >>> data = CensoredData(uncensored=[13, 22, 17, 15], right=[20, 18])
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    Alternatively, we can use the method `CensoredData.right_censored` to
    create a representation of this data.  The time-to-failure observations
    are put the list ``ttf``.  The ``censored`` list indicates which values
    in ``ttf`` are censored.

    >>> ttf = [13, 22, 17, 15, 20, 18]
    >>> censored = [False, False, False, False, True, True]

    Pass these lists to `CensoredData.right_censored` to create an
    instance of `CensoredData`.

    >>> data = CensoredData.right_censored(ttf, censored)
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    If the input data is interval censored and already stored in two
    arrays, one holding the low end of the intervals and another
    holding the high ends, the class method ``interval_censored`` can
    be used to create the `CensoredData` instance.

    This example creates an instance with four interval-censored values.
    The intervals are [10, 11], [0.5, 1], [2, 3], and [12.5, 13.5].

    >>> a = [10, 0.5, 2, 12.5]  # Low ends of the intervals
    >>> b = [11, 1.0, 3, 13.5]  # High ends of the intervals
    >>> data = CensoredData.interval_censored(low=a, high=b)
    >>> print(data)
    CensoredData(4 values: 0 not censored, 4 interval-censored)

    Finally, we create and censor some data from the `weibull_min`
    distribution, and then fit `weibull_min` to that data. We'll assume
    that the location parameter is known to be 0.

    >>> from scipy.stats import weibull_min
    >>> rng = np.random.default_rng()

    Create the random data set.

    >>> x = weibull_min.rvs(2.5, loc=0, scale=30, size=250, random_state=rng)
    >>> x[x > 40] = 40  # Right-censor values greater or equal to 40.

    Create the `CensoredData` instance with the `right_censored` method.
    The censored values are those where the value is 40.

    >>> data = CensoredData.right_censored(x, x == 40)
    >>> print(data)
    CensoredData(250 values: 215 not censored, 35 right-censored)

    35 values have been right-censored.

    Fit `weibull_min` to the censored data.  We expect to shape and scale
    to be approximately 2.5 and 30, respectively.

    >>> weibull_min.fit(data, floc=0)
    (2.3575922823897315, 0, 30.40650074451254)

    """

    def __init__(self, uncensored=None, *, left=None, right=None,
                 interval=None):
        if uncensored is None:
            uncensored = []
        if left is None:
            left = []
        if right is None:
            right = []
        if interval is None:
            interval = np.empty((0, 2))

        _validate_1d(uncensored, 'uncensored')
        _validate_1d(left, 'left')
        _validate_1d(right, 'right')
        uncensored2, left2, right2, interval2 = _validate_interval(interval)

        self._uncensored = np.concatenate((uncensored, uncensored2))
        self._left = np.concatenate((left, left2))
        self._right = np.concatenate((right, right2))
        # Note that by construction, the private attribute _interval
        # will be a 2D array that contains only finite values representing
        # intervals with nonzero but finite length.
        self._interval = interval2

    def __repr__(self):
        uncensored_str = " ".join(np.array_repr(self._uncensored).split())
        left_str = " ".join(np.array_repr(self._left).split())
        right_str = " ".join(np.array_repr(self._right).split())
        interval_str = " ".join(np.array_repr(self._interval).split())
        return (f"CensoredData(uncensored={uncensored_str}, left={left_str}, "
                f"right={right_str}, interval={interval_str})")

    def __str__(self):
        num_nc = len(self._uncensored)
        num_lc = len(self._left)
        num_rc = len(self._right)
        num_ic = len(self._interval)
        n = num_nc + num_lc + num_rc + num_ic
        parts = [f'{num_nc} not censored']
        if num_lc > 0:
            parts.append(f'{num_lc} left-censored')
        if num_rc > 0:
            parts.append(f'{num_rc} right-censored')
        if num_ic > 0:
            parts.append(f'{num_ic} interval-censored')
        return f'CensoredData({n} values: ' + ', '.join(parts) + ')'

    # This is not a complete implementation of the arithmetic operators.
    # All we need is subtracting a scalar and dividing by a scalar.

    def __sub__(self, other):
        return CensoredData(uncensored=self._uncensored - other,
                            left=self._left - other,
                            right=self._right - other,
                            interval=self._interval - other)

    def __truediv__(self, other):
        return CensoredData(uncensored=self._uncensored / other,
                            left=self._left / other,
                            right=self._right / other,
                            interval=self._interval / other)

    def __len__(self):
        """
        The number of values (censored and not censored).
        """
        return (len(self._uncensored) + len(self._left) + len(self._right)
                + len(self._interval))

    def num_censored(self):
        """
        Number of censored values.
        """
        return len(self._left) + len(self._right) + len(self._interval)

    @classmethod
    def right_censored(cls, x, censored):
        """
        Create a `CensoredData` instance of right-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is right-censored.  That is, the value ``x[k]``
            is the lower bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and right-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (4 and 10) and two right-censored values
        (24 and 25).

        >>> data = CensoredData.right_censored([4, 10, 24, 25],
        ...                                    [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([ 4., 10.]),
        left=array([], dtype=float64), right=array([24., 25.]),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 right-censored)
        """
        x, censored = _validate_x_censored(x, censored)
        return cls(uncensored=x[~censored], right=x[censored])

    @classmethod
    def left_censored(cls, x, censored):
        """
        Create a `CensoredData` instance of left-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is left-censored.  That is, the value ``x[k]``
            is the upper bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and left-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (0.12 and 0.033) and two left-censored values
        (both 1e-3).

        >>> data = CensoredData.left_censored([0.12, 0.033, 1e-3, 1e-3],
        ...                                   [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([0.12 , 0.033]),
        left=array([0.001, 0.001]), right=array([], dtype=float64),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 left-censored)
        """
        x, censored = _validate_x_censored(x, censored)
        return cls(uncensored=x[~censored], left=x[censored])

    @classmethod
    def interval_censored(cls, low, high):
        """
        Create a `CensoredData` instance of interval-censored data.

        This method is useful when all the data is interval-censored, and
        the low and high ends of the intervals are already stored in
        separate one-dimensional arrays.

        Parameters
        ----------
        low : array_like
            The one-dimensional array containing the low ends of the
            intervals.
        high : array_like
            The one-dimensional array containing the high ends of the
            intervals.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of censored values.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import CensoredData

        ``a`` and ``b`` are the low and high ends of a collection of
        interval-censored values.

        >>> a = [0.5, 2.0, 3.0, 5.5]
        >>> b = [1.0, 2.5, 3.5, 7.0]
        >>> data = CensoredData.interval_censored(low=a, high=b)
        >>> print(data)
        CensoredData(4 values: 0 not censored, 4 interval-censored)
        """
        _validate_1d(low, 'low', allow_inf=True)
        _validate_1d(high, 'high', allow_inf=True)
        if len(low) != len(high):
            raise ValueError('`low` and `high` must have the same length.')
        interval = np.column_stack((low, high))
        uncensored, left, right, interval = _validate_interval(interval)
        return cls(uncensored=uncensored, left=left, right=right,
                   interval=interval)

    def _uncensor(self):
        """
        This function is used when a non-censored version of the data
        is needed to create a rough estimate of the parameters of a
        distribution via the method of moments or some similar method.
        The data is "uncensored" by taking the given endpoints as the
        data for the left- or right-censored data, and the mean for the
        interval-censored data.
        """
        data = np.concatenate((self._uncensored, self._left, self._right,
                               self._interval.mean(axis=1)))
        return data

    def _supported(self, a, b):
        """
        Return a subset of self containing the values that are in
        (or overlap with) the interval (a, b).
        """
        uncensored = self._uncensored
        uncensored = uncensored[(a < uncensored) & (uncensored < b)]
        left = self._left
        left = left[a < left]
        right = self._right
        right = right[right < b]
        interval = self._interval
        interval = interval[(a < interval[:, 1]) & (interval[:, 0] < b)]
        return CensoredData(uncensored, left=left, right=right,
                            interval=interval)
<|endoftext|># Integration of multivariate normal and t distributions.

# Adapted from the MATLAB original implementations by Dr. Alan Genz.

#     http://www.math.wsu.edu/faculty/genz/software/software.html

# Copyright (C) 2013, Alan Genz,  All rights reserved.
# Python implementation is copyright (C) 2022, Robert Kern,  All rights
# reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided the following conditions are met:
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#   3. The contributor name(s) may not be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to


phi = ndtr
phinv = ndtri


def _factorize_int(n):
    """Return a sorted list of the unique prime factors of a positive integer.
    """
    # NOTE: There are lots faster ways to do this, but this isn't terrible.
    factors = set()
    for p in primes_from_2_to(int(np.sqrt(n)) + 1):
        while not (n % p):
            factors.add(p)
            n //= p
        if n == 1:
            break
    if n != 1:
        factors.add(n)
    return sorted(factors)


def _primitive_root(p):
    """Compute a primitive root of the prime number `p`.

    Used in the CBC lattice construction.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    """
    # p is prime
    pm = p - 1
    factors = _factorize_int(pm)
    n = len(factors)
    r = 2
    k = 0
    while k < n:
        d = pm // factors[k]
        # pow() doesn't like numpy scalar types.
        rd = pow(int(r), int(d), int(p))
        if rd == 1:
            r += 1
            k = 0
        else:
            k += 1
    return r


def _cbc_lattice(n_dim, n_qmc_samples):
    """Compute a QMC lattice generator using a Fast CBC construction.

    Parameters
    ----------
    n_dim : int > 0
        The number of dimensions for the lattice.
    n_qmc_samples : int > 0
        The desired number of QMC samples. This will be rounded down to the
        nearest prime to enable the CBC construction.

    Returns
    -------
    q : float array : shape=(n_dim,)
        The lattice generator vector. All values are in the open interval
        ``(0, 1)``.
    actual_n_qmc_samples : int
        The prime number of QMC samples that must be used with this lattice,
        no more, no less.

    References
    ----------
    .. [1] Nuyens, D. and Cools, R. "Fast Component-by-Component Construction,
           a Reprise for Different Kernels", In H. Niederreiter and D. Talay,
           editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
           Springer-Verlag, 2006, 371-385.
    """
    # Round down to the nearest prime number.
    primes = primes_from_2_to(n_qmc_samples + 1)
    n_qmc_samples = primes[-1]

    bt = np.ones(n_dim)
    gm = np.hstack([1.0, 0.8 ** np.arange(n_dim - 1)])
    q = 1
    w = 0
    z = np.arange(1, n_dim + 1)
    m = (n_qmc_samples - 1) // 2
    g = _primitive_root(n_qmc_samples)
    # Slightly faster way to compute perm[j] = pow(g, j, n_qmc_samples)
    # Shame that we don't have modulo pow() implemented as a ufunc.
    perm = np.ones(m, dtype=int)
    for j in range(m - 1):
        perm[j + 1] = (g * perm[j]) % n_qmc_samples
    perm = np.minimum(n_qmc_samples - perm, perm)
    pn = perm / n_qmc_samples
    c = pn * pn - pn + 1.0 / 6
    fc = fft(c)
    for s in range(1, n_dim):
        reordered = np.hstack([
            c[:w+1][::-1],
            c[w+1:m][::-1],
        ])
        q = q * (bt[s-1] + gm[s-1] * reordered)
        w = ifft(fc * fft(q)).real.argmin()
        z[s] = perm[w]
    q = z / n_qmc_samples
    return q, n_qmc_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the development of a parameter for users
# to set the desired CDF accuracy, but must be reviewed and tested before use.
def _qauto(func, covar, low, high, rng, error=1e-3, limit=10_000, **kwds):
    """Automatically rerun the integration to get the required error bound.

    Parameters
    ----------
    func : callable
        Either :func:`_qmvn` or :func:`_qmvt`.
    covar, low, high : array
        As specified in :func:`_qmvn` and :func:`_qmvt`.
    rng : Generator, optional
        default_rng(), yada, yada
    error : float > 0
        The desired error bound.
    limit : int > 0:
        The rough limit of the number of integration points to consider. The
        integration will stop looping once this limit has been *exceeded*.
    **kwds :
        Other keyword arguments to pass to `func`. When using :func:`_qmvt`, be
        sure to include ``nu=`` as one of these.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    n_samples : int
        The number of integration points actually used.
    """
    n = len(covar)
    n_samples = 0
    if n == 1:
        prob = phi(high) - phi(low)
        # More or less
        est_error = 1e-15
    else:
        mi = min(limit, n * 1000)
        prob = 0.0
        est_error = 1.0
        ei = 0.0
        while est_error > error and n_samples < limit:
            mi = round(np.sqrt(2) * mi)
            pi, ei, ni = func(mi, covar, low, high, rng=rng, **kwds)
            n_samples += ni
            wt = 1.0 / (1 + (ei / est_error)**2)
            prob += wt * (pi - prob)
            est_error = np.sqrt(wt) * ei
    return prob, est_error, n_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _qmvn(m, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate normal integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n - 1, max(m // n_batches, 1))
    y = np.zeros((n - 1, n_qmc_samples))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i - 1] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _mvn_qmc_integrand(covar, low, high, use_tent=False):
    """Transform the multivariate normal integration into a QMC integrand over
    a unit hypercube.

    The dimensionality of the resulting hypercube integration domain is one
    less than the dimensionality of the original integrand. Note that this
    transformation subsumes the integration bounds in order to account for
    infinite bounds. The QMC integration one does with the returned integrand
    should be on the unit hypercube.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    use_tent : bool, optional
        If True, then use tent periodization. Only helpful for lattice rules.

    Returns
    -------
    integrand : Callable[[NDArray], NDArray]
        The QMC-integrable integrand. It takes an
        ``(n_qmc_samples, ndim_integrand)`` array of QMC samples in the unit
        hypercube and returns the ``(n_qmc_samples,)`` evaluations of at these
        QMC points.
    ndim_integrand : int
        The dimensionality of the integrand. Equal to ``n-1``.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ndim_integrand = n - 1
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci

    def integrand(*zs):
        ndim_qmc = len(zs)
        n_qmc_samples = len(np.atleast_1d(zs[0]))
        assert ndim_qmc == ndim_integrand
        y = np.zeros((ndim_qmc, n_qmc_samples))
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            if use_tent:
                # Tent periodization transform.
                x = abs(2 * zs[i-1] - 1)
            else:
                x = zs[i-1]
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        return pv

    return integrand, ndim_integrand


def _qmvt(m, nu, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate t integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    nu : float >= 0
        The shape parameter of the multivariate t distribution.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    n_samples : int
        The number of samples actually used.
    """
    sn = max(1.0, np.sqrt(nu))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    cho, lo, hi = _permuted_cholesky(covar, low / sn, high / sn)
    n = cho.shape[0]
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n, max(m // n_batches, 1))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        pv = np.ones(n_qmc_samples)
        s = np.zeros((n, n_qmc_samples))
        for i in range(n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            # FIXME: Lift the i==0 case out of the loop to make the logic
            # easier to follow.
            if i == 0:
                # We'll use one of the QR variates to pull out the
                # t-distribution scaling.
                if nu > 0:
                    r = np.sqrt(2 * gammaincinv(nu / 2, x))
                else:
                    r = np.ones_like(x)
            else:
                y = phinv(c + x * dc)  # noqa: F821
                with np.errstate(invalid='ignore'):
                    s[i:, :] += cho[i:, i - 1][:, np.newaxis] * y
            si = s[i, :]

            c = np.ones(n_qmc_samples)
            d = np.ones(n_qmc_samples)
            with np.errstate(invalid='ignore'):
                lois = lo[i] * r - si
                hiis = hi[i] * r - si
            c[lois < -9] = 0.0
            d[hiis < -9] = 0.0
            lo_mask = abs(lois) < 9
            hi_mask = abs(hiis) < 9
            c[lo_mask] = phi(lois[lo_mask])
            d[hi_mask] = phi(hiis[hi_mask])

            dc = d - c
            pv *= dc

        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples


def _permuted_cholesky(covar, low, high, tol=1e-10):
    """Compute a scaled, permuted Cholesky factor, with integration bounds.

    The scaling and permuting of the dimensions accomplishes part of the
    transformation of the original integration problem into a more numerically
    tractable form. The lower-triangular Cholesky factor will then be used in
    the subsequent integration. The integration bounds will be scaled and
    permuted as well.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    tol : float, optional
        The singularity tolerance.

    Returns
    -------
    cho : (n, n) float array
        Lower Cholesky factor, scaled and permuted.
    new_low, new_high : (n,) float array
        The scaled and permuted low and high integration bounds.
    """
    # Make copies for outputting.
    cho = np.array(covar, dtype=np.float64)
    new_lo = np.array(low, dtype=np.float64)
    new_hi = np.array(high, dtype=np.float64)
    n = cho.shape[0]
    if cho.shape != (n, n):
        raise ValueError("expected a square symmetric array")
    if new_lo.shape != (n,) or new_hi.shape != (n,):
        raise ValueError(
            "expected integration boundaries the same dimensions "
            "as the covariance matrix"
        )
    # Scale by the sqrt of the diagonal.
    dc = np.sqrt(np.maximum(np.diag(cho), 0.0))
    # But don't divide by 0.
    dc[dc == 0.0] = 1.0
    new_lo /= dc
    new_hi /= dc
    cho /= dc
    cho /= dc[:, np.newaxis]

    y = np.zeros(n)
    sqtp = np.sqrt(2 * np.pi)
    for k in range(n):
        epk = (k + 1) * tol
        im = k
        ck = 0.0
        dem = 1.0
        s = 0.0
        lo_m = 0.0
        hi_m = 0.0
        for i in range(k, n):
            if cho[i, i] > tol:
                ci = np.sqrt(cho[i, i])
                if i > 0:
                    s = cho[i, :k] @ y[:k]
                lo_i = (new_lo[i] - s) / ci
                hi_i = (new_hi[i] - s) / ci
                de = phi(hi_i) - phi(lo_i)
                if de <= dem:
                    ck = ci
                    dem = de
                    lo_m = lo_i
                    hi_m = hi_i
                    im = i
        if im > k:
            # Swap im and k
            cho[im, im] = cho[k, k]
            _swap_slices(cho, np.s_[im, :k], np.s_[k, :k])
            _swap_slices(cho, np.s_[im + 1:, im], np.s_[im + 1:, k])
            _swap_slices(cho, np.s_[k + 1:im, k], np.s_[im, k + 1:im])
            _swap_slices(new_lo, k, im)
            _swap_slices(new_hi, k, im)
        if ck > epk:
            cho[k, k] = ck
            cho[k, k + 1:] = 0.0
            for i in range(k + 1, n):
                cho[i, k] /= ck
                cho[i, k + 1:i + 1] -= cho[i, k] * cho[k + 1:i + 1, k]
            if abs(dem) > tol:
                y[k] = ((np.exp(-lo_m * lo_m / 2) - np.exp(-hi_m * hi_m / 2)) /
                        (sqtp * dem))
            else:
                y[k] = (lo_m + hi_m) / 2
                if lo_m < -10:
                    y[k] = hi_m
                elif hi_m > 10:
                    y[k] = lo_m
            cho[k, :k + 1] /= ck
            new_lo[k] /= ck
            new_hi[k] /= ck
        else:
            cho[k:, k] = 0.0
            y[k] = (new_lo[k] + new_hi[k]) / 2
    return cho, new_lo, new_hi


def _swap_slices(x, slc1, slc2):
    t = x[slc1].copy()
    x[slc1] = x[slc2].copy()
    x[slc2] = t
<|endoftext|># Compute the two-sided one-sample Kolmogorov-Smirnov Prob(Dn <= d) where:
#    D_n = sup_x{|F_n(x) - F(x)|},
#    F_n(x) is the empirical CDF for a sample of size n {x_i: i=1,...,n},
#    F(x) is the CDF of a probability distribution.
#
# Exact methods:
# Prob(D_n >= d) can be computed via a matrix algorithm of Durbin[1]
#   or a recursion algorithm due to Pomeranz[2].
# Marsaglia, Tsang & Wang[3] gave a computation-efficient way to perform
#   the Durbin algorithm.
#   D_n >= d <==>  D_n+ >= d or D_n- >= d (the one-sided K-S statistics), hence
#   Prob(D_n >= d) = 2*Prob(D_n+ >= d) - Prob(D_n+ >= d and D_n- >= d).
#   For d > 0.5, the latter intersection probability is 0.
#
# Approximate methods:
# For d close to 0.5, ignoring that intersection term may still give a
#   reasonable approximation.
# Li-Chien[4] and Korolyuk[5] gave an asymptotic formula extending
# Kolmogorov's initial asymptotic, suitable for large d. (See
#   scipy.special.kolmogorov for that asymptotic)
# Pelz-Good[6] used the functional equation for Jacobi theta functions to
#   transform the Li-Chien/Korolyuk formula produce a computational formula
#   suitable for small d.
#
# Simard and L'Ecuyer[7] provided an algorithm to decide when to use each of
#   the above approaches and it is that which is used here.
#
# Other approaches:
# Carvalho[8] optimizes Durbin's matrix algorithm for large values of d.
# Moscovich and Nadler[9] use FFTs to compute the convolutions.

# References:
# [1] Durbin J (1968).
#     "The Probability that the Sample Distribution Function Lies Between Two
#     Parallel Straight Lines."
#     Annals of Mathematical Statistics, 39, 398-411.
# [2] Pomeranz J (1974).
#     "Exact Cumulative Distribution of the Kolmogorov-Smirnov Statistic for
#     Small Samples (Algorithm 487)."
#     Communications of the ACM, 17(12), 703-704.
# [3] Marsaglia G, Tsang WW, Wang J (2003).
#     "Evaluating Kolmogorov's Distribution."
#     Journal of Statistical Software, 8(18), 1-4.
# [4] LI-CHIEN, C. (1956).
#     "On the exact distribution of the statistics of A. N. Kolmogorov and
#     their asymptotic expansion."
#     Acta Matematica Sinica, 6, 55-81.
# [5] KOROLYUK, V. S. (1960).
#     "Asymptotic analysis of the distribution of the maximum deviation in
#     the Bernoulli scheme."
#     Theor. Probability Appl., 4, 339-366.
# [6] Pelz W, Good IJ (1976).
#     "Approximating the Lower Tail-areas of the Kolmogorov-Smirnov One-sample
#     Statistic."
#     Journal of the Royal Statistical Society, Series B, 38(2), 152-156.
#  [7] Simard, R., L'Ecuyer, P. (2011)
# 	  "Computing the Two-Sided Kolmogorov-Smirnov Distribution",
# 	  Journal of Statistical Software, Vol 39, 11, 1-18.
#  [8] Carvalho, Luis (2015)
#     "An Improved Evaluation of Kolmogorov's Distribution"
#     Journal of Statistical Software, Code Snippets; Vol 65(3), 1-8.
#  [9] Amit Moscovich, Boaz Nadler (2017)
#     "Fast calculation of boundary crossing probabilities for Poisson
#     processes",
#     Statistics & Probability Letters, Vol 123, 177-182.


import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative

_E128 = 128
_EP128 = np.ldexp(np.longdouble(1), _E128)
_EM128 = np.ldexp(np.longdouble(1), -_E128)

_SQRT2PI = np.sqrt(2 * np.pi)
_LOG_2PI = np.log(2 * np.pi)
_MIN_LOG = -708
_SQRT3 = np.sqrt(3)
_PI_SQUARED = np.pi ** 2
_PI_FOUR = np.pi ** 4
_PI_SIX = np.pi ** 6

# [Lifted from _loggamma.pxd.] If B_m are the Bernoulli numbers,
# then Stirling coeffs are B_{2j}/(2j)/(2j-1) for j=8,...1.
_STIRLING_COEFFS = [-2.955065359477124183e-2, 6.4102564102564102564e-3,
                    -1.9175269175269175269e-3, 8.4175084175084175084e-4,
                    -5.952380952380952381e-4, 7.9365079365079365079e-4,
                    -2.7777777777777777778e-3, 8.3333333333333333333e-2]


def _log_nfactorial_div_n_pow_n(n):
    # Computes n! / n**n
    #    = (n-1)! / n**(n-1)
    # Uses Stirling's approximation, but removes n*log(n) up-front to
    # avoid subtractive cancellation.
    #    = log(n)/2 - n + log(sqrt(2pi)) + sum B_{2j}/(2j)/(2j-1)/n**(2j-1)
    rn = 1.0/n
    return np.log(n)/2 - n + _LOG_2PI/2 + rn * np.polyval(_STIRLING_COEFFS, rn/n)


def _clip_prob(p):
    """clips a probability to range 0<=p<=1."""
    return np.clip(p, 0.0, 1.0)


def _select_and_clip_prob(cdfprob, sfprob, cdf=True):
    """Selects either the CDF or SF, and then clips to range 0<=p<=1."""
    p = np.where(cdf, cdfprob, sfprob)
    return _clip_prob(p)


def _kolmogn_DMTW(n, d, cdf=True):
    r"""Computes the Kolmogorov CDF:  Pr(D_n <= d) using the MTW approach to
    the Durbin matrix algorithm.

    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].
    """
    # Write d = (k-h)/n, where k is positive integer and 0 <= h < 1
    # Generate initial matrix H of size m*m where m=(2k-1)
    # Compute k-th row of (n!/n^n) * H^n, scaling intermediate results.
    # Requires memory O(m^2) and computation O(m^2 log(n)).
    # Most suitable for small m.

    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(np.ceil(nd))
    h = k - nd
    m = 2 * k - 1

    H = np.zeros([m, m])

    # Initialize: v is first column (and last row) of H
    #  v[j] = (1-h^(j+1)/(j+1)!  (except for v[-1])
    #  w[j] = 1/(j)!
    # q = k-th row of H (actually i!/n^i*H^i)
    intm = np.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = np.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j  # This might underflow.  Isn't a problem.
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0)**m - 2*h**m
    v[-1] = (1.0 + tt) * fac

    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = np.flip(v, axis=0)

    Hpwr = np.eye(np.shape(H)[0])  # Holds intermediate powers of H
    nn = n
    expnt = 0  # Scaling of Hpwr
    Hexpnt = 0  # Scaling of H
    while nn > 0:
        if nn % 2:
            Hpwr = np.matmul(Hpwr, H)
            expnt += Hexpnt
        H = np.matmul(H, H)
        Hexpnt *= 2
        # Scale as needed.
        if np.abs(H[k - 1, k - 1]) > _EP128:
            H /= _EP128
            Hexpnt += _E128
        nn = nn // 2

    p = Hpwr[k - 1, k - 1]

    # Multiply by n!/n^n
    for i in range(1, n + 1):
        p = i * p / n
        if np.abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128

    # unscale
    if expnt != 0:
        p = np.ldexp(p, expnt)

    return _select_and_clip_prob(p, 1.0-p, cdf)


def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    """Compute the endpoints of the interval for row i."""
    if i == 0:
        j1, j2 = -ll - ceilf - 1, ll + ceilf - 1
    else:
        # i + 1 = 2*ip1div2 + ip1mod2
        ip1div2, ip1mod2 = divmod(i + 1, 2)
        if ip1mod2 == 0:  # i is odd
            if ip1div2 == n + 1:
                j1, j2 = n - ll - ceilf - 1, n + ll + ceilf - 1
            else:
                j1, j2 = ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1
        else:
            j1, j2 = ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1

    return max(j1 + 2, 0), min(j2, n)


def _kolmogn_Pomeranz(n, x, cdf=True):
    r"""Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.

    Pomeranz (1974) [2]
    """

    # V is n*(2n+2) matrix.
    # Each row is convolution of the previous row and probabilities from a
    #  Poisson distribution.
    # Desired CDF probability is n! V[n-1, 2n+1]  (final entry in final row).
    # Only two rows are needed at any given stage:
    #  - Call them V0 and V1.
    #  - Swap each iteration
    # Only a few (contiguous) entries in each row can be non-zero.
    #  - Keep track of start and end (j1 and j2 below)
    #  - V0s and V1s track the start in the two rows
    # Scale intermediate results as needed.
    # Only a few different Poisson distributions can occur
    t = n * x
    ll = int(np.floor(t))
    f = 1.0 * (t - ll)  # fractional part of t
    g = min(f, 1.0 - f)
    ceilf = (1 if f > 0 else 0)
    roundf = (1 if f > 0.5 else 0)
    npwrs = 2 * (ll + 1)    # Maximum number of powers needed in convolutions
    gpower = np.empty(npwrs)  # gpower = (g/n)^m/m!
    twogpower = np.empty(npwrs)  # twogpower = (2g/n)^m/m!
    onem2gpower = np.empty(npwrs)  # onem2gpower = ((1-2g)/n)^m/m!
    # gpower etc are *almost* Poisson probs, just missing normalizing factor.

    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n, two_g_over_n, one_minus_two_g_over_n = g/n, 2*g/n, (1 - 2*g)/n
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m

    V0 = np.zeros([npwrs])
    V1 = np.zeros([npwrs])
    V1[0] = 1  # first row
    V0s, V1s = 0, 0  # start indices of the two rows

    j1, j2 = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        # Preserve j1, V1, V1s, V0s from last iteration
        k1 = j1
        V0, V1 = V1, V0
        V0s, V1s = V1s, V0s
        V1.fill(0.0)
        j1, j2 = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = (twogpower if i % 2 else onem2gpower)
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1  # First index to use from conv
            conv_len = j2 - j1 + 1  # Number of entries to use from conv
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]
            # Scale to avoid underflow.
            if 0 < np.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1

    # multiply by n!
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if np.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m

    # Undo any intermediate scaling
    if expnt != 0:
        ans = np.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans


def _kolmogn_PelzGood(n, x, cdf=True):
    """Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.

    Start with Li-Chien, Korolyuk approximation:
        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5
    where z = x*sqrt(n).
    Transform each K_(z) using Jacobi theta functions into a form suitable
    for small z.
    Pelz-Good (1976). [6]
    """
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)

    z = np.sqrt(n) * x
    zsquared, zthree, zfour, zsix = z**2, z**3, z**4, z**6

    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:  # z ~ 0.041743441416853426
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)

    q = np.exp(qlog)

    # Coefficients of terms in the sums for K1, K2 and K3
    k1a = -zsquared
    k1b = _PI_SQUARED / 4

    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16

    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z**8

    K0to3 = np.zeros(4)
    # Use a Horner scheme to evaluate sum c_i q^(i^2)
    # Reduces to a sum over odd integers.
    maxk = int(np.ceil(16 * z / np.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msquared, mfour, msix = m**2, m**4, m**6
        qpower = np.power(q, 8 * k)
        coeffs = np.array([1.0,
                           k1a + k1b*msquared,
                           k2a + k2b*msquared + k2c*mfour,
                           k3a + k3b*msquared + k3c*mfour + k3d*msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    # z**10 > 0 as z > 0.04
    K0to3 /= np.array([z, 6 * zfour, 72 * z**7, 6480 * z**10])

    # Now do the other sum over the other terms, all integers k
    # K_2:  (pi^2 k^2) q^(k^2),
    # K_3:  (3pi^2 k^2 z^2 - pi^4 k^4)*q^(k^2)
    # Don't expect much subtractive cancellation so use direct calculation
    q = np.exp(-_PI_SQUARED / 2 / zsquared)
    ks = np.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = np.pi * ks
    qpwers = q ** ksquared
    k2extra = np.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI/(-36 * zthree)
    K0to3[2] += k2extra
    k3extra = np.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI/(216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = np.power(n * 1.0, np.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n

    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1

    Ksum = sum(K0to3)
    return Ksum


def _kolmogn(n, x, cdf=True):
    """Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.

    Simard & L'Ecuyer (2011) [7].
    """
    if np.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    t = n * x
    if t <= 1.0:  # Ruben-Gambino: 1/2n <= x <= 1/n
        if t <= 0.5:
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
        if n <= 140:
            prob = np.prod(np.arange(1, n+1) * (1.0/n) * (2*t - 1))
        else:
            prob = np.exp(_log_nfactorial_div_n_pow_n(n) + n * np.log(2*t-1))
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
    if t >= n - 1:  # Ruben-Gambino
        prob = 2 * (1.0 - x)**n
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)
