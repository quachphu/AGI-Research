    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    This won't call ``BackendBC`` because it doesn't support ``TypeA``.

    We can also use leave out the ``ua.Dispatchable`` if we specify the
    default ``dispatch_type`` for the ``dispatchables`` argument.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [a, b], dispatch_type="mark", domain="ua_examples"
    ...     ):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    """
    if "dispatch_type" in kwargs:
        disp_type = kwargs.pop("dispatch_type")
        dispatchables = tuple(
            d if isinstance(d, Dispatchable) else Dispatchable(d, disp_type)
            for d in dispatchables
        )
    else:
        dispatchables = tuple(dispatchables)
        if not all(isinstance(d, Dispatchable) for d in dispatchables):
            raise TypeError("dispatchables must be instances of uarray.Dispatchable")

    if len(kwargs) != 0:
        raise TypeError(f"Received unexpected keyword arguments: {kwargs}")

    backend = _uarray.determine_backend(domain, dispatchables, coerce)

    return set_backend(backend, coerce=coerce, only=only)
<|endoftext|>""" Collection of Model instances for use with the odrpack fitting package.
"""
import numpy as np
from scipy.odr._odrpack import Model

__all__ = ['Model', 'exponential', 'multilinear', 'unilinear', 'quadratic',
           'polynomial']


def _lin_fcn(B, x):
    a, b = B[0], B[1:]
    b.shape = (b.shape[0], 1)

    return a + (x*b).sum(axis=0)


def _lin_fjb(B, x):
    a = np.ones(x.shape[-1], float)
    res = np.concatenate((a, x.ravel()))
    res.shape = (B.shape[-1], x.shape[-1])
    return res


def _lin_fjd(B, x):
    b = B[1:]
    b = np.repeat(b, (x.shape[-1],)*b.shape[-1], axis=0)
    b.shape = x.shape
    return b


def _lin_est(data):
    # Eh. The answer is analytical, so just return all ones.
    # Don't return zeros since that will interfere with
    # ODRPACK's auto-scaling procedures.

    if len(data.x.shape) == 2:
        m = data.x.shape[0]
    else:
        m = 1

    return np.ones((m + 1,), float)


def _poly_fcn(B, x, powers):
    a, b = B[0], B[1:]
    b.shape = (b.shape[0], 1)

    return a + np.sum(b * np.power(x, powers), axis=0)


def _poly_fjacb(B, x, powers):
    res = np.concatenate((np.ones(x.shape[-1], float),
                          np.power(x, powers).flat))
    res.shape = (B.shape[-1], x.shape[-1])
    return res


def _poly_fjacd(B, x, powers):
    b = B[1:]
    b.shape = (b.shape[0], 1)

    b = b * powers

    return np.sum(b * np.power(x, powers-1), axis=0)


def _exp_fcn(B, x):
    return B[0] + np.exp(B[1] * x)


def _exp_fjd(B, x):
    return B[1] * np.exp(B[1] * x)


def _exp_fjb(B, x):
    res = np.concatenate((np.ones(x.shape[-1], float), x * np.exp(B[1] * x)))
    res.shape = (2, x.shape[-1])
    return res


def _exp_est(data):
    # Eh.
    return np.array([1., 1.])


class _MultilinearModel(Model):
    r"""
    Arbitrary-dimensional linear model

    This model is defined by :math:`y=\beta_0 + \sum_{i=1}^m \beta_i x_i`

    Examples
    --------
    We can calculate orthogonal distance regression with an arbitrary
    dimensional linear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 10.0 + 5.0 * x
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.multilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [10.  5.]

    """

    def __init__(self):
        super().__init__(
            _lin_fcn, fjacb=_lin_fjb, fjacd=_lin_fjd, estimate=_lin_est,
            meta={'name': 'Arbitrary-dimensional Linear',
                  'equ': 'y = B_0 + Sum[i=1..m, B_i * x_i]',
                  'TeXequ': r'$y=\beta_0 + \sum_{i=1}^m \beta_i x_i$'})


multilinear = _MultilinearModel()


def polynomial(order):
    """
    Factory function for a general polynomial model.

    Parameters
    ----------
    order : int or sequence
        If an integer, it becomes the order of the polynomial to fit. If
        a sequence of numbers, then these are the explicit powers in the
        polynomial.
        A constant term (power 0) is always included, so don't include 0.
        Thus, polynomial(n) is equivalent to polynomial(range(1, n+1)).

    Returns
    -------
    polynomial : Model instance
        Model instance.

    Examples
    --------
    We can fit an input data using orthogonal distance regression (ODR) with
    a polynomial model:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import odr
    >>> x = np.linspace(0.0, 5.0)
    >>> y = np.sin(x)
    >>> poly_model = odr.polynomial(3)  # using third order polynomial model
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, poly_model)
    >>> output = odr_obj.run()  # running ODR fitting
    >>> poly = np.poly1d(output.beta[::-1])
    >>> poly_y = poly(x)
    >>> plt.plot(x, y, label="input data")
    >>> plt.plot(x, poly_y, label="polynomial ODR")
    >>> plt.legend()
    >>> plt.show()

    """

    powers = np.asarray(order)
    if powers.shape == ():
        # Scalar.
        powers = np.arange(1, powers + 1)

    powers.shape = (len(powers), 1)
    len_beta = len(powers) + 1

    def _poly_est(data, len_beta=len_beta):
        # Eh. Ignore data and return all ones.
        return np.ones((len_beta,), float)

    return Model(_poly_fcn, fjacd=_poly_fjacd, fjacb=_poly_fjacb,
                 estimate=_poly_est, extra_args=(powers,),
                 meta={'name': 'Sorta-general Polynomial',
                 'equ': 'y = B_0 + Sum[i=1..%s, B_i * (x**i)]' % (len_beta-1),
                 'TeXequ': r'$y=\beta_0 + \sum_{i=1}^{%s} \beta_i x^i$' %
                        (len_beta-1)})


class _ExponentialModel(Model):
    r"""
    Exponential model

    This model is defined by :math:`y=\beta_0 + e^{\beta_1 x}`

    Examples
    --------
    We can calculate orthogonal distance regression with an exponential model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = -10.0 + np.exp(0.5*x)
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.exponential)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [-10.    0.5]

    """

    def __init__(self):
        super().__init__(_exp_fcn, fjacd=_exp_fjd, fjacb=_exp_fjb,
                         estimate=_exp_est,
                         meta={'name': 'Exponential',
                               'equ': 'y= B_0 + exp(B_1 * x)',
                               'TeXequ': r'$y=\beta_0 + e^{\beta_1 x}$'})


exponential = _ExponentialModel()


def _unilin(B, x):
    return x*B[0] + B[1]


def _unilin_fjd(B, x):
    return np.ones(x.shape, float) * B[0]


def _unilin_fjb(B, x):
    _ret = np.concatenate((x, np.ones(x.shape, float)))
    _ret.shape = (2,) + x.shape

    return _ret


def _unilin_est(data):
    return (1., 1.)


def _quadratic(B, x):
    return x*(x*B[0] + B[1]) + B[2]


def _quad_fjd(B, x):
    return 2*x*B[0] + B[1]


def _quad_fjb(B, x):
    _ret = np.concatenate((x*x, x, np.ones(x.shape, float)))
    _ret.shape = (3,) + x.shape

    return _ret


def _quad_est(data):
    return (1.,1.,1.)


class _UnilinearModel(Model):
    r"""
    Univariate linear model

    This model is defined by :math:`y = \beta_0 x + \beta_1`

    Examples
    --------
    We can calculate orthogonal distance regression with an unilinear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x + 2.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.unilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2.]

    """

    def __init__(self):
        super().__init__(_unilin, fjacd=_unilin_fjd, fjacb=_unilin_fjb,
                         estimate=_unilin_est,
                         meta={'name': 'Univariate Linear',
                               'equ': 'y = B_0 * x + B_1',
                               'TeXequ': '$y = \\beta_0 x + \\beta_1$'})


unilinear = _UnilinearModel()


class _QuadraticModel(Model):
    r"""
    Quadratic model

    This model is defined by :math:`y = \beta_0 x^2 + \beta_1 x + \beta_2`

    Examples
    --------
    We can calculate orthogonal distance regression with a quadratic model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x ** 2 + 2.0 * x + 3.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.quadratic)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2. 3.]

    """

    def __init__(self):
        super().__init__(
            _quadratic, fjacd=_quad_fjd, fjacb=_quad_fjb, estimate=_quad_est,
            meta={'name': 'Quadratic',
                  'equ': 'y = B_0*x**2 + B_1*x + B_2',
                  'TeXequ': '$y = \\beta_0 x^2 + \\beta_1 x + \\beta_2'})


quadratic = _QuadraticModel()
<|endoftext|>"""
Python wrappers for Orthogonal Distance Regression (ODRPACK).

Notes
=====

* Array formats -- FORTRAN stores its arrays in memory column first, i.e., an
  array element A(i, j, k) will be next to A(i+1, j, k). In C and, consequently,
  NumPy, arrays are stored row first: A[i, j, k] is next to A[i, j, k+1]. For
  efficiency and convenience, the input and output arrays of the fitting
  function (and its Jacobians) are passed to FORTRAN without transposition.
  Therefore, where the ODRPACK documentation says that the X array is of shape
  (N, M), it will be passed to the Python function as an array of shape (M, N).
  If M==1, the 1-D case, then nothing matters; if M>1, then your
  Python functions will be dealing with arrays that are indexed in reverse of
  the ODRPACK documentation. No real issue, but watch out for your indexing of
  the Jacobians: the i,jth elements (@f_i/@x_j) evaluated at the nth
  observation will be returned as jacd[j, i, n]. Except for the Jacobians, it
  really is easier to deal with x[0] and x[1] than x[:,0] and x[:,1]. Of course,
  you can always use the transpose() function from SciPy explicitly.

* Examples -- See the accompanying file test/test.py for examples of how to set
  up fits of your own. Some are taken from the User's Guide; some are from
  other sources.

* Models -- Some common models are instantiated in the accompanying module
  models.py . Contributions are welcome.

Credits
=======

* Thanks to Arnold Moene and Gerard Vermeulen for fixing some killer bugs.

Robert Kern
robert.kern@gmail.com

"""
import os
from threading import Lock

import numpy as np
from warnings import warn
from scipy.odr import __odrpack

__all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop',
           'Data', 'RealData', 'Model', 'Output', 'ODR',
           'odr_error', 'odr_stop']

odr = __odrpack.odr
ODR_LOCK = Lock()


class OdrWarning(UserWarning):
    """
    Warning indicating that the data passed into
    ODR will cause problems when passed into 'odr'
    that the user should be aware of.
    """
    pass


class OdrError(Exception):
    """
    Exception indicating an error in fitting.

    This is raised by `~scipy.odr.odr` if an error occurs during fitting.
    """
    pass


class OdrStop(Exception):
    """
    Exception stopping fitting.

    You can raise this exception in your objective function to tell
    `~scipy.odr.odr` to stop fitting.
    """
    pass


# Backwards compatibility
odr_error = OdrError
odr_stop = OdrStop

__odrpack._set_exceptions(OdrError, OdrStop)


def _conv(obj, dtype=None):
    """ Convert an object to the preferred form for input to the odr routine.
    """

    if obj is None:
        return obj
    else:
        if dtype is None:
            obj = np.asarray(obj)
        else:
            obj = np.asarray(obj, dtype)
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)
        else:
            return obj


def _report_error(info):
    """ Interprets the return code of the odr routine.

    Parameters
    ----------
    info : int
        The return code of the odr routine.

    Returns
    -------
    problems : list(str)
        A list of messages about why the odr() routine stopped.
    """

    stopreason = ('Blank',
                  'Sum of squares convergence',
                  'Parameter convergence',
                  'Both sum of squares and parameter convergence',
                  'Iteration limit reached')[info % 5]

    if info >= 5:
        # questionable results or fatal error

        I = (info//10000 % 10,
             info//1000 % 10,
             info//100 % 10,
             info//10 % 10,
             info % 10)
        problems = []

        if I[0] == 0:
            if I[1] != 0:
                problems.append('Derivatives possibly not correct')
            if I[2] != 0:
                problems.append('Error occurred in callback')
            if I[3] != 0:
                problems.append('Problem is not full rank at solution')
            problems.append(stopreason)
        elif I[0] == 1:
            if I[1] != 0:
                problems.append('N < 1')
            if I[2] != 0:
                problems.append('M < 1')
            if I[3] != 0:
                problems.append('NP < 1 or NP > N')
            if I[4] != 0:
                problems.append('NQ < 1')
        elif I[0] == 2:
            if I[1] != 0:
                problems.append('LDY and/or LDX incorrect')
            if I[2] != 0:
                problems.append('LDWE, LD2WE, LDWD, and/or LD2WD incorrect')
            if I[3] != 0:
                problems.append('LDIFX, LDSTPD, and/or LDSCLD incorrect')
            if I[4] != 0:
                problems.append('LWORK and/or LIWORK too small')
        elif I[0] == 3:
            if I[1] != 0:
                problems.append('STPB and/or STPD incorrect')
            if I[2] != 0:
                problems.append('SCLB and/or SCLD incorrect')
            if I[3] != 0:
                problems.append('WE incorrect')
            if I[4] != 0:
                problems.append('WD incorrect')
        elif I[0] == 4:
            problems.append('Error in derivatives')
        elif I[0] == 5:
            problems.append('Error occurred in callback')
        elif I[0] == 6:
            problems.append('Numerical error detected')

        return problems

    else:
        return [stopreason]


class Data:
    """
    The data to fit.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    we : array_like, optional
        If `we` is a scalar, then that value is used for all data points (and
        all dimensions of the response variable).
        If `we` is a rank-1 array of length q (the dimensionality of the
        response variable), then this vector is the diagonal of the covariant
        weighting matrix for all data points.
        If `we` is a rank-1 array of length n (the number of data points), then
        the i'th element is the weight for the i'th response variable
        observation (single-dimensional only).
        If `we` is a rank-2 array of shape (q, q), then this is the full
        covariant weighting matrix broadcast to each observation.
        If `we` is a rank-2 array of shape (q, n), then `we[:,i]` is the
        diagonal of the covariant weighting matrix for the i'th observation.
        If `we` is a rank-3 array of shape (q, q, n), then `we[:,:,i]` is the
        full specification of the covariant weighting matrix for each
        observation.
        If the fit is implicit, then only a positive scalar value is used.
    wd : array_like, optional
        If `wd` is a scalar, then that value is used for all data points
        (and all dimensions of the input variable). If `wd` = 0, then the
        covariant weighting matrix for each observation is set to the identity
        matrix (so each dimension of each observation has the same weight).
        If `wd` is a rank-1 array of length m (the dimensionality of the input
        variable), then this vector is the diagonal of the covariant weighting
        matrix for all data points.
        If `wd` is a rank-1 array of length n (the number of data points), then
        the i'th element is the weight for the ith input variable observation
        (single-dimensional only).
        If `wd` is a rank-2 array of shape (m, m), then this is the full
        covariant weighting matrix broadcast to each observation.
        If `wd` is a rank-2 array of shape (m, n), then `wd[:,i]` is the
        diagonal of the covariant weighting matrix for the ith observation.
        If `wd` is a rank-3 array of shape (m, m, n), then `wd[:,:,i]` is the
        full specification of the covariant weighting matrix for each
        observation.
    fix : array_like of ints, optional
        The `fix` argument is the same as ifixx in the class ODR. It is an
        array of integers with the same shape as data.x that determines which
        input observations are treated as fixed. One can use a sequence of
        length m (the dimensionality of the input observations) to fix some
        dimensions for all observations. A value of 0 fixes the observation,
        a value > 0 makes it free.
    meta : dict, optional
        Free-form dictionary for metadata.

    Notes
    -----
    Each argument is attached to the member of the instance of the same name.
    The structures of `x` and `y` are described in the Model class docstring.
    If `y` is an integer, then the Data instance can only be used to fit with
    implicit models where the dimensionality of the response is equal to the
    specified value of `y`.

    The `we` argument weights the effect a deviation in the response variable
    has on the fit. The `wd` argument weights the effect a deviation in the
    input variable has on the fit. To handle multidimensional inputs and
    responses easily, the structure of these arguments has the n'th
    dimensional axis first. These arguments heavily use the structured
    arguments feature of ODRPACK to conveniently and flexibly support all
    options. See the ODRPACK User's Guide for a full explanation of how these
    weights are used in the algorithm. Basically, a higher value of the weight
    for a particular data point makes a deviation at that point more
    detrimental to the fit.

    """

    def __init__(self, x, y=None, we=None, wd=None, fix=None, meta=None):
        self.x = _conv(x)

        if not isinstance(self.x, np.ndarray):
            raise ValueError("Expected an 'ndarray' of data for 'x', "
                             f"but instead got data of type '{type(self.x).__name__}'")

        self.y = _conv(y)
        self.we = _conv(we)
        self.wd = _conv(wd)
        self.fix = _conv(fix)
        self.meta = {} if meta is None else meta

    def set_meta(self, **kwds):
        """ Update the metadata dictionary with the keywords and data provided
        by keywords.

        Examples
        --------
        ::

            data.set_meta(lab="Ph 7; Lab 26", title="Ag110 + Ag108 Decay")
        """

        self.meta.update(kwds)

    def __getattr__(self, attr):
        """ Dispatch attribute access to the metadata dictionary.
        """
        if attr != "meta" and attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError(f"'{attr}' not in metadata")


class RealData(Data):
    """
    The data, with weightings as actual standard deviations and/or
    covariances.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    sx : array_like, optional
        Standard deviations of `x`.
        `sx` are standard deviations of `x` and are converted to weights by
        dividing 1.0 by their squares.
    sy : array_like, optional
        Standard deviations of `y`.
        `sy` are standard deviations of `y` and are converted to weights by
        dividing 1.0 by their squares.
    covx : array_like, optional
        Covariance of `x`
        `covx` is an array of covariance matrices of `x` and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    covy : array_like, optional
        Covariance of `y`
        `covy` is an array of covariance matrices and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    fix : array_like, optional
        The argument and member fix is the same as Data.fix and ODR.ifixx:
        It is an array of integers with the same shape as `x` that
        determines which input observations are treated as fixed. One can
        use a sequence of length m (the dimensionality of the input
        observations) to fix some dimensions for all observations. A value
        of 0 fixes the observation, a value > 0 makes it free.
    meta : dict, optional
        Free-form dictionary for metadata.

    Notes
    -----
    The weights `wd` and `we` are computed from provided values as follows:

    `sx` and `sy` are converted to weights by dividing 1.0 by their squares.
    For example, ``wd = 1./np.power(`sx`, 2)``.

    `covx` and `covy` are arrays of covariance matrices and are converted to
    weights by performing a matrix inversion on each observation's covariance
    matrix. For example, ``we[i] = np.linalg.inv(covy[i])``.

    These arguments follow the same structured argument conventions as wd and
    we only restricted by their natures: `sx` and `sy` can't be rank-3, but
    `covx` and `covy` can be.

    Only set *either* `sx` or `covx` (not both). Setting both will raise an
    exception. Same with `sy` and `covy`.

    """

    def __init__(self, x, y=None, sx=None, sy=None, covx=None, covy=None,
                 fix=None, meta=None):
        if (sx is not None) and (covx is not None):
            raise ValueError("cannot set both sx and covx")
        if (sy is not None) and (covy is not None):
            raise ValueError("cannot set both sy and covy")

        # Set flags for __getattr__
        self._ga_flags = {}
        if sx is not None:
            self._ga_flags['wd'] = 'sx'
        else:
            self._ga_flags['wd'] = 'covx'
        if sy is not None:
            self._ga_flags['we'] = 'sy'
        else:
            self._ga_flags['we'] = 'covy'

        self.x = _conv(x)

        if not isinstance(self.x, np.ndarray):
            raise ValueError("Expected an 'ndarray' of data for 'x', "
                              f"but instead got data of type '{type(self.x).__name__}'")

        self.y = _conv(y)
        self.sx = _conv(sx)
        self.sy = _conv(sy)
        self.covx = _conv(covx)
        self.covy = _conv(covy)
        self.fix = _conv(fix)
        self.meta = {} if meta is None else meta

    def _sd2wt(self, sd):
        """ Convert standard deviation to weights.
        """

        return 1./np.power(sd, 2)

    def _cov2wt(self, cov):
        """ Convert covariance matrix(-ices) to weights.
        """

        from scipy.linalg import inv

        if len(cov.shape) == 2:
            return inv(cov)
        else:
            weights = np.zeros(cov.shape, float)

            for i in range(cov.shape[-1]):  # n
                weights[:,:,i] = inv(cov[:,:,i])

            return weights

    def __getattr__(self, attr):

        if attr not in ('wd', 'we'):
            if attr != "meta" and attr in self.meta:
                return self.meta[attr]
            else:
                raise AttributeError(f"'{attr}' not in metadata")
        else:
            lookup_tbl = {('wd', 'sx'): (self._sd2wt, self.sx),
                      ('wd', 'covx'): (self._cov2wt, self.covx),
                      ('we', 'sy'): (self._sd2wt, self.sy),
                      ('we', 'covy'): (self._cov2wt, self.covy)}

            func, arg = lookup_tbl[(attr, self._ga_flags[attr])]

            if arg is not None:
                return func(*(arg,))
            else:
                return None


class Model:
    """
    The Model class stores information about the function you wish to fit.

    It stores the function itself, at the least, and optionally stores
    functions which compute the Jacobians used during fitting. Also, one
    can provide a function that will provide reasonable starting values
    for the fit parameters possibly given the set of data.

    Parameters
    ----------
    fcn : function
          fcn(beta, x) --> y
    fjacb : function
          Jacobian of fcn wrt the fit parameters beta.

          fjacb(beta, x) --> @f_i(x,B)/@B_j
    fjacd : function
          Jacobian of fcn wrt the (possibly multidimensional) input
          variable.

          fjacd(beta, x) --> @f_i(x,B)/@x_j
    extra_args : tuple, optional
          If specified, `extra_args` should be a tuple of extra
          arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called
          by `apply(fcn, (beta, x) + extra_args)`
    estimate : array_like of rank-1
          Provides estimates of the fit parameters from the data

          estimate(data) --> estbeta
    implicit : boolean
          If TRUE, specifies that the model
          is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit
          against
    meta : dict, optional
          freeform dictionary of metadata for the model

    Notes
    -----
    Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and
    return a NumPy array. The `estimate` object takes an instance of the
    Data class.

    Here are the rules for the shapes of the argument and return
    arrays of the callback functions:

    `x`
        if the input data is single-dimensional, then `x` is rank-1
        array; i.e., ``x = array([1, 2, 3, ...]); x.shape = (n,)``
        If the input data is multi-dimensional, then `x` is a rank-2 array;
        i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.
        In all cases, it has the same shape as the input data array passed to
        `~scipy.odr.odr`. `m` is the dimensionality of the input data,
        `n` is the number of observations.
    `y`
        if the response variable is single-dimensional, then `y` is a
        rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.
        If the response variable is multi-dimensional, then `y` is a rank-2
        array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =
        (q, n)`` where `q` is the dimensionality of the response variable.
    `beta`
        rank-1 array of length `p` where `p` is the number of parameters;
        i.e. ``beta = array([B_1, B_2, ..., B_p])``
    `fjacb`
        if the response variable is multi-dimensional, then the
        return array's shape is ``(q, p, n)`` such that ``fjacb(x,beta)[l,k,i] =
        d f_l(X,B)/d B_k`` evaluated at the ith data point.  If ``q == 1``, then
        the return array is only rank-2 and with shape ``(p, n)``.
    `fjacd`
        as with fjacb, only the return array's shape is ``(q, m, n)``
        such that ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` at the ith data
        point.  If ``q == 1``, then the return array's shape is ``(m, n)``. If
        ``m == 1``, the shape is (q, n). If `m == q == 1`, the shape is ``(n,)``.

    """

    def __init__(self, fcn, fjacb=None, fjacd=None,
                 extra_args=None, estimate=None, implicit=0, meta=None):

        self.fcn = fcn
        self.fjacb = fjacb
        self.fjacd = fjacd

        if extra_args is not None:
            extra_args = tuple(extra_args)

        self.extra_args = extra_args
        self.estimate = estimate
        self.implicit = implicit
        self.meta = meta if meta is not None else {}

    def set_meta(self, **kwds):
        """ Update the metadata dictionary with the keywords and data provided
        here.

        Examples
        --------
        set_meta(name="Exponential", equation="y = a exp(b x) + c")
        """

        self.meta.update(kwds)

    def __getattr__(self, attr):
        """ Dispatch attribute access to the metadata.
        """

        if attr != "meta" and attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError(f"'{attr}' not in metadata")


class Output:
    """
    The Output class stores the output of an ODR run.

    Attributes
    ----------
    beta : ndarray
        Estimated parameter values, of shape (q,).
    sd_beta : ndarray
        Standard deviations of the estimated parameters, of shape (p,).
    cov_beta : ndarray
        Covariance matrix of the estimated parameters, of shape (p,p).
        Note that this `cov_beta` is not scaled by the residual variance
        `res_var`, whereas `sd_beta` is. This means
        ``np.sqrt(np.diag(output.cov_beta * output.res_var))`` is the same
        result as `output.sd_beta`.
    delta : ndarray, optional
        Array of estimated errors in input variables, of same shape as `x`.
    eps : ndarray, optional
        Array of estimated errors in response variables, of same shape as `y`.
    xplus : ndarray, optional
        Array of ``x + delta``.
    y : ndarray, optional
        Array ``y = fcn(x + delta)``.
    res_var : float, optional
        Residual variance.
    sum_square : float, optional
        Sum of squares error.
    sum_square_delta : float, optional
        Sum of squares of delta error.
    sum_square_eps : float, optional
        Sum of squares of eps error.
    inv_condnum : float, optional
        Inverse condition number (cf. ODRPACK UG p. 77).
    rel_error : float, optional
        Relative error in function values computed within fcn.
    work : ndarray, optional
        Final work array.
    work_ind : dict, optional
        Indices into work for drawing out values (cf. ODRPACK UG p. 83).
    info : int, optional
        Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
    stopreason : list of str, optional
        `info` interpreted into English.

    Notes
    -----
    Takes one argument for initialization, the return value from the
    function `~scipy.odr.odr`. The attributes listed as "optional" above are
    only present if `~scipy.odr.odr` was run with ``full_output=1``.

    """

    def __init__(self, output):
        self.beta = output[0]
        self.sd_beta = output[1]
        self.cov_beta = output[2]

        if len(output) == 4:
            # full output
            self.__dict__.update(output[3])
            self.stopreason = _report_error(self.info)

    def pprint(self):
        """ Pretty-print important results.
        """

        print('Beta:', self.beta)
        print('Beta Std Error:', self.sd_beta)
        print('Beta Covariance:', self.cov_beta)
        if hasattr(self, 'info'):
            print('Residual Variance:',self.res_var)
            print('Inverse Condition #:', self.inv_condnum)
            print('Reason(s) for Halting:')
            for r in self.stopreason:
                print(f'  {r}')


class ODR:
    """
    The ODR class gathers all information and coordinates the running of the
    main fitting routine.

    Members of instances of the ODR class have the same names as the arguments
    to the initialization routine.

    Parameters
    ----------
    data : Data class instance
        instance of the Data class
    model : Model class instance
        instance of the Model class

    Other Parameters
    ----------------
    beta0 : array_like of rank-1
        a rank-1 sequence of initial parameter values. Optional if
        model provides an "estimate" function to estimate these values.
    delta0 : array_like of floats of rank-1, optional
        a (double-precision) float array to hold the initial values of
        the errors in the input variables. Must be same shape as data.x
    ifixb : array_like of ints of rank-1, optional
        sequence of integers with the same length as beta0 that determines
        which parameters are held fixed. A value of 0 fixes the parameter,
        a value > 0 makes the parameter free.
    ifixx : array_like of ints with same shape as data.x, optional
        an array of integers with the same shape as data.x that determines
        which input observations are treated as fixed. One can use a sequence
        of length m (the dimensionality of the input observations) to fix some
        dimensions for all observations. A value of 0 fixes the observation,
        a value > 0 makes it free.
    job : int, optional
        an integer telling ODRPACK what tasks to perform. See p. 31 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_job post-initialization for a more readable interface.
    iprint : int, optional
        an integer telling ODRPACK what to print. See pp. 33-34 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_iprint post-initialization for a more readable interface.
    errfile : str, optional
        string with the filename to print ODRPACK errors to. If the file already
        exists, an error will be thrown. The `overwrite` argument can be used to
        prevent this. *Do Not Open This File Yourself!*
    rptfile : str, optional
        string with the filename to print ODRPACK summaries to. If the file
        already exists, an error will be thrown. The `overwrite` argument can be
        used to prevent this. *Do Not Open This File Yourself!*
    ndigit : int, optional
        integer specifying the number of reliable digits in the computation
        of the function.
    taufac : float, optional
        float specifying the initial trust region. The default value is 1.
        The initial trust region is equal to taufac times the length of the
        first computed Gauss-Newton step. taufac must be less than 1.
    sstol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the sum-of-squares. The default value is eps**(1/2) where eps
        is the smallest value such that 1 + eps > 1 for double precision
        computation on the machine. sstol must be less than 1.
    partol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the estimated parameters. The default value is eps**(2/3) for
        explicit models and ``eps**(1/3)`` for implicit models. partol must be less
        than 1.
    maxit : int, optional
        integer specifying the maximum number of iterations to perform. For
        first runs, maxit is the total number of iterations performed and
        defaults to 50. For restarts, maxit is the number of additional
        iterations to perform and defaults to 10.
    stpb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
        finite difference derivatives wrt the parameters.
    stpd : optional
        array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
        step sizes to compute finite difference derivatives wrt the input
        variable errors. If stpd is a rank-1 array with length m (the
        dimensionality of the input variable), then the values are broadcast to
        all observations.
    sclb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of scaling factors for the
        parameters. The purpose of these scaling factors are to scale all of
        the parameters to around unity. Normally appropriate scaling factors
        are computed if this argument is not specified. Specify them yourself
        if the automatic procedure goes awry.
    scld : array_like, optional
        array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
        factors for the *errors* in the input variables. Again, these factors
        are automatically computed if you do not provide them. If scld.shape ==
        (m,), then the scaling factors are broadcast to all observations.
    work : ndarray, optional
        array to hold the double-valued working data for ODRPACK. When
        restarting, takes the value of self.output.work.
    iwork : ndarray, optional
        array to hold the integer-valued working data for ODRPACK. When
        restarting, takes the value of self.output.iwork.
    overwrite : bool, optional
        If it is True, output files defined by `errfile` and `rptfile` are
        overwritten. The default is False.

    Attributes
    ----------
    data : Data
        The data for this fit
    model : Model
        The model used in fit
    output : Output
        An instance if the Output class containing all of the returned
        data from an invocation of ODR.run() or ODR.restart()

    """

    def __init__(self, data, model, beta0=None, delta0=None, ifixb=None,
        ifixx=None, job=None, iprint=None, errfile=None, rptfile=None,
        ndigit=None, taufac=None, sstol=None, partol=None, maxit=None,
        stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None,
        overwrite=False):

        self.data = data
        self.model = model

        if beta0 is None:
            if self.model.estimate is not None:
                self.beta0 = _conv(self.model.estimate(self.data))
            else:
                raise ValueError(
                  "must specify beta0 or provide an estimator with the model"
                )
        else:
            self.beta0 = _conv(beta0)

        if ifixx is None and data.fix is not None:
            ifixx = data.fix

        if overwrite:
            # remove output files for overwriting.
            if rptfile is not None and os.path.exists(rptfile):
                os.remove(rptfile)
            if errfile is not None and os.path.exists(errfile):
                os.remove(errfile)

        self.delta0 = _conv(delta0)
        # These really are 32-bit integers in FORTRAN (gfortran), even on 64-bit
        # platforms.
        # XXX: some other FORTRAN compilers may not agree.
        self.ifixx = _conv(ifixx, dtype=np.int32)
        self.ifixb = _conv(ifixb, dtype=np.int32)
        self.job = job
        self.iprint = iprint
        self.errfile = errfile
        self.rptfile = rptfile
        self.ndigit = ndigit
        self.taufac = taufac
        self.sstol = sstol
        self.partol = partol
        self.maxit = maxit
        self.stpb = _conv(stpb)
        self.stpd = _conv(stpd)
        self.sclb = _conv(sclb)
        self.scld = _conv(scld)
        self.work = _conv(work)
        self.iwork = _conv(iwork)

        self.output = None

        self._check()

    def _check(self):
        """ Check the inputs for consistency, but don't bother checking things
        that the builtin function odr will check.
        """

        x_s = list(self.data.x.shape)

        if isinstance(self.data.y, np.ndarray):
            y_s = list(self.data.y.shape)
            if self.model.implicit:
                raise OdrError("an implicit model cannot use response data")
        else:
            # implicit model with q == self.data.y
            y_s = [self.data.y, x_s[-1]]
            if not self.model.implicit:
                raise OdrError("an explicit model needs response data")
            self.set_job(fit_type=1)

        if x_s[-1] != y_s[-1]:
            raise OdrError("number of observations do not match")

        n = x_s[-1]

        if len(x_s) == 2:
            m = x_s[0]
        else:
            m = 1
        if len(y_s) == 2:
            q = y_s[0]
        else:
            q = 1

        p = len(self.beta0)

        # permissible output array shapes

        fcn_perms = [(q, n)]
        fjacd_perms = [(q, m, n)]
        fjacb_perms = [(q, p, n)]

        if q == 1:
            fcn_perms.append((n,))
            fjacd_perms.append((m, n))
            fjacb_perms.append((p, n))
        if m == 1:
            fjacd_perms.append((q, n))
        if p == 1:
            fjacb_perms.append((q, n))
        if m == q == 1:
            fjacd_perms.append((n,))
        if p == q == 1:
            fjacb_perms.append((n,))

        # try evaluating the supplied functions to make sure they provide
        # sensible outputs

        arglist = (self.beta0, self.data.x)
        if self.model.extra_args is not None:
            arglist = arglist + self.model.extra_args
        res = self.model.fcn(*arglist)

        if res.shape not in fcn_perms:
            print(res.shape)
            print(fcn_perms)
            raise OdrError(f"fcn does not output {y_s}-shaped array")

        if self.model.fjacd is not None:
            res = self.model.fjacd(*arglist)
            if res.shape not in fjacd_perms:
                raise OdrError(
                    f"fjacd does not output {repr((q, m, n))}-shaped array")
        if self.model.fjacb is not None:
            res = self.model.fjacb(*arglist)
            if res.shape not in fjacb_perms:
                raise OdrError(
                    f"fjacb does not output {repr((q, p, n))}-shaped array")

        # check shape of delta0

        if self.delta0 is not None and self.delta0.shape != self.data.x.shape:
            raise OdrError(
                f"delta0 is not a {repr(self.data.x.shape)}-shaped array")

        if self.data.x.size == 0:
            warn("Empty data detected for ODR instance. "
                 "Do not expect any fitting to occur",
                 OdrWarning, stacklevel=3)

    def _gen_work(self):
        """ Generate a suitable work array if one does not already exist.
        """

        n = self.data.x.shape[-1]
        p = self.beta0.shape[0]

        if len(self.data.x.shape) == 2:
            m = self.data.x.shape[0]
        else:
            m = 1

        if self.model.implicit:
            q = self.data.y
        elif len(self.data.y.shape) == 2:
            q = self.data.y.shape[0]
        else:
            q = 1

        if self.data.we is None:
            ldwe = ld2we = 1
        elif len(self.data.we.shape) == 3:
            ld2we, ldwe = self.data.we.shape[1:]
        else:
            we = self.data.we
            ldwe = 1
            ld2we = 1
            if we.ndim == 1 and q == 1:
                ldwe = n
            elif we.ndim == 2:
                if we.shape == (q, q):
                    ld2we = q
                elif we.shape == (q, n):
                    ldwe = n

        if self.job % 10 < 2:
            # ODR not OLS
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 6*n*m + 2*n*q*p +
                     2*n*q*m + q*q + 5*q + q*(p+m) + ldwe*ld2we*q)
        else:
            # OLS not ODR
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 2*n*m + 2*n*q*p +
                     5*q + q*(p+m) + ldwe*ld2we*q)

        if isinstance(self.work, np.ndarray) and self.work.shape == (lwork,)\
                and self.work.dtype.str.endswith('f8'):
            # the existing array is fine
            return
        else:
            self.work = np.zeros((lwork,), float)

    def set_job(self, fit_type=None, deriv=None, var_calc=None,
        del_init=None, restart=None):
        """
        Sets the "job" parameter is a hopefully comprehensible way.

        If an argument is not specified, then the value is left as is. The
        default value from class initialization is for all of these options set
        to 0.

        Parameters
        ----------
        fit_type : {0, 1, 2} int
            0 -> explicit ODR

            1 -> implicit ODR

            2 -> ordinary least-squares
        deriv : {0, 1, 2, 3} int
            0 -> forward finite differences

            1 -> central finite differences

            2 -> user-supplied derivatives (Jacobians) with results
              checked by ODRPACK

            3 -> user-supplied derivatives, no checking
        var_calc : {0, 1, 2} int
            0 -> calculate asymptotic covariance matrix and fit
                 parameter uncertainties (V_B, s_B) using derivatives
                 recomputed at the final solution

            1 -> calculate V_B and s_B using derivatives from last iteration

            2 -> do not calculate V_B and s_B
        del_init : {0, 1} int
            0 -> initial input variable offsets set to 0

            1 -> initial offsets provided by user in variable "work"
        restart : {0, 1} int
            0 -> fit is not a restart

            1 -> fit is a restart

        Notes
        -----
        The permissible values are different from those given on pg. 31 of the
        ODRPACK User's Guide only in that one cannot specify numbers greater than
        the last value for each variable.

        If one does not supply functions to compute the Jacobians, the fitting
        procedure will change deriv to 0, finite differences, as a default. To
        initialize the input variable offsets by yourself, set del_init to 1 and
        put the offsets into the "work" variable correctly.

        """

        if self.job is None:
            job_l = [0, 0, 0, 0, 0]
        else:
            job_l = [self.job // 10000 % 10,
                     self.job // 1000 % 10,
                     self.job // 100 % 10,
                     self.job // 10 % 10,
                     self.job % 10]

        if fit_type in (0, 1, 2):
            job_l[4] = fit_type
        if deriv in (0, 1, 2, 3):
            job_l[3] = deriv
        if var_calc in (0, 1, 2):
            job_l[2] = var_calc
        if del_init in (0, 1):
            job_l[1] = del_init
        if restart in (0, 1):
            job_l[0] = restart

        self.job = (job_l[0]*10000 + job_l[1]*1000 +
                    job_l[2]*100 + job_l[3]*10 + job_l[4])

    def set_iprint(self, init=None, so_init=None,
        iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
        """ Set the iprint parameter for the printing of computation reports.

        If any of the arguments are specified here, then they are set in the
        iprint member. If iprint is not set manually or with this method, then
        ODRPACK defaults to no printing. If no filename is specified with the
        member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
        print to stdout in addition to the specified filename by setting the
        so_* arguments to this function, but one cannot specify to print to
        stdout but not a file since one can do that by not specifying a rptfile
        filename.

        There are three reports: initialization, iteration, and final reports.
        They are represented by the arguments init, iter, and final
        respectively.  The permissible values are 0, 1, and 2 representing "no
        report", "short report", and "long report" respectively.

        The argument iter_step (0 <= iter_step <= 9) specifies how often to make
        the iteration report; the report will be made for every iter_step'th
        iteration starting with iteration one. If iter_step == 0, then no
        iteration report is made, regardless of the other arguments.

        If the rptfile is None, then any so_* arguments supplied will raise an
        exception.
        """
        if self.iprint is None:
            self.iprint = 0

        ip = [self.iprint // 1000 % 10,
              self.iprint // 100 % 10,
              self.iprint // 10 % 10,
              self.iprint % 10]

        # make a list to convert iprint digits to/from argument inputs
        #                   rptfile, stdout
        ip2arg = [[0, 0],  # none,  none
                  [1, 0],  # short, none
                  [2, 0],  # long,  none
                  [1, 1],  # short, short
                  [2, 1],  # long,  short
                  [1, 2],  # short, long
                  [2, 2]]  # long,  long

        if (self.rptfile is None and
            (so_init is not None or
             so_iter is not None or
             so_final is not None)):
            raise OdrError(
                "no rptfile specified, cannot output to stdout twice")

        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]

        if init is not None:
            iprint_l[0] = init
        if so_init is not None:
            iprint_l[1] = so_init
        if iter is not None:
            iprint_l[2] = iter
        if so_iter is not None:
            iprint_l[3] = so_iter
        if final is not None:
            iprint_l[4] = final
        if so_final is not None:
            iprint_l[5] = so_final

        if iter_step in range(10):
            # 0..9
            ip[2] = iter_step

        ip[0] = ip2arg.index(iprint_l[0:2])
        ip[1] = ip2arg.index(iprint_l[2:4])
        ip[3] = ip2arg.index(iprint_l[4:6])

        self.iprint = ip[0]*1000 + ip[1]*100 + ip[2]*10 + ip[3]

    def run(self):
        """ Run the fitting routine with all of the information given and with ``full_output=1``.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """  # noqa: E501

        args = (self.model.fcn, self.beta0, self.data.y, self.data.x)
        kwds = {'full_output': 1}
        kwd_l = ['ifixx', 'ifixb', 'job', 'iprint', 'errfile', 'rptfile',
                 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb',
                 'stpd', 'sclb', 'scld', 'work', 'iwork']

        if self.delta0 is not None and (self.job // 10000) % 10 == 0:
            # delta0 provided and fit is not a restart
            self._gen_work()

            d0 = np.ravel(self.delta0)

            self.work[:len(d0)] = d0

        # set the kwds from other objects explicitly
        if self.model.fjacb is not None:
            kwds['fjacb'] = self.model.fjacb
        if self.model.fjacd is not None:
            kwds['fjacd'] = self.model.fjacd
        if self.data.we is not None:
            kwds['we'] = self.data.we
        if self.data.wd is not None:
            kwds['wd'] = self.data.wd
        if self.model.extra_args is not None:
            kwds['extra_args'] = self.model.extra_args

        # implicitly set kwds from self's members
        for attr in kwd_l:
            obj = getattr(self, attr)
            if obj is not None:
                kwds[attr] = obj

        with ODR_LOCK:
            self.output = Output(odr(*args, **kwds))

        return self.output

    def restart(self, iter=None):
        """ Restarts the run with iter more iterations.

        Parameters
        ----------
        iter : int, optional
            ODRPACK's default for the number of new iterations is 10.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """

        if self.output is None:
            raise OdrError("cannot restart: run() has not been called before")

        self.set_job(restart=1)
        self.work = self.output.work
        self.iwork = self.output.iwork

        self.maxit = iter

        return self.run()
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.odr` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'Model', 'exponential', 'multilinear', 'unilinear',
    'quadratic', 'polynomial'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="odr", module="models",
                                   private_modules=["_models"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
=================================================
Orthogonal distance regression (:mod:`scipy.odr`)
=================================================

.. currentmodule:: scipy.odr

Package Content
===============

.. autosummary::
   :toctree: generated/

   Data          -- The data to fit.
   RealData      -- Data with weights as actual std. dev.s and/or covariances.
   Model         -- Stores information about the function to be fit.
   ODR           -- Gathers all info & manages the main fitting routine.
   Output        -- Result from the fit.
   odr           -- Low-level function for ODR.

   OdrWarning    -- Warning about potential problems when running ODR.
   OdrError      -- Error exception.
   OdrStop       -- Stop exception.

   polynomial    -- Factory function for a general polynomial model.
   exponential   -- Exponential model
   multilinear   -- Arbitrary-dimensional linear model
   unilinear     -- Univariate linear model
   quadratic     -- Quadratic model

Usage information
=================

Introduction
------------

Why Orthogonal Distance Regression (ODR)? Sometimes one has
measurement errors in the explanatory (a.k.a., "independent")
variable(s), not just the response (a.k.a., "dependent") variable(s).
Ordinary Least Squares (OLS) fitting procedures treat the data for
explanatory variables as fixed, i.e., not subject to error of any kind.
Furthermore, OLS procedures require that the response variables be an
explicit function of the explanatory variables; sometimes making the
equation explicit is impractical and/or introduces errors.  ODR can
handle both of these cases with ease, and can even reduce to the OLS
case if that is sufficient for the problem.

ODRPACK is a FORTRAN-77 library for performing ODR with possibly
non-linear fitting functions. It uses a modified trust-region
Levenberg-Marquardt-type algorithm [1]_ to estimate the function
parameters.  The fitting functions are provided by Python functions
operating on NumPy arrays. The required derivatives may be provided
by Python functions as well, or may be estimated numerically. ODRPACK
can do explicit or implicit ODR fits, or it can do OLS. Input and
output variables may be multidimensional. Weights can be provided to
account for different variances of the observations, and even
covariances between dimensions of the variables.

The `scipy.odr` package offers an object-oriented interface to
ODRPACK, in addition to the low-level `odr` function.

Additional background information about ODRPACK can be found in the
`ODRPACK User's Guide
<https://docs.scipy.org/doc/external/odrpack_guide.pdf>`_, reading
which is recommended.

Basic usage
-----------

1. Define the function you want to fit against.::

       def f(B, x):
           '''Linear function y = m*x + b'''
           # B is a vector of the parameters.
           # x is an array of the current x values.
           # x is in the same format as the x passed to Data or RealData.
           #
           # Return an array in the same format as y passed to Data or RealData.
           return B[0]*x + B[1]

2. Create a Model.::

       linear = Model(f)

3. Create a Data or RealData instance.::

       mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))

   or, when the actual covariances are known::

       mydata = RealData(x, y, sx=sx, sy=sy)

4. Instantiate ODR with your data, model and initial parameter estimate.::

       myodr = ODR(mydata, linear, beta0=[1., 2.])

5. Run the fit.::

       myoutput = myodr.run()

6. Examine output.::

       myoutput.pprint()


References
----------
.. [1] P. T. Boggs and J. E. Rogers, "Orthogonal Distance Regression,"
   in "Statistical analysis of measurement error models and
   applications: proceedings of the AMS-IMS-SIAM joint summer research
   conference held June 10-16, 1989," Contemporary Mathematics,
   vol. 112, pg. 186, 1990.

"""
# version: 0.7
# author: Robert Kern <robert.kern@gmail.com>
# date: 2006-09-21

from ._odrpack import *
from ._models import *
from . import _add_newdocs

# Deprecated namespaces, to be removed in v2.0.0
from . import models, odrpack

__all__ = [s for s in dir()
           if not (s.startswith('_') or s in ('odr_stop', 'odr_error'))]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.odr` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'odr', 'OdrWarning', 'OdrError', 'OdrStop',
    'Data', 'RealData', 'Model', 'Output', 'ODR',
    'odr_error', 'odr_stop'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="odr", module="odrpack",
                                   private_modules=["_odrpack"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
K-means clustering and vector quantization (:mod:`scipy.cluster.vq`)
====================================================================

Provides routines for k-means clustering, generating code books
from k-means models and quantizing vectors by comparing them with
centroids in a code book.

.. autosummary::
   :toctree: generated/

   whiten -- Normalize a group of observations so each feature has unit variance
   vq -- Calculate code book membership of a set of observation vectors
   kmeans -- Perform k-means on a set of observation vectors forming k clusters
   kmeans2 -- A different implementation of k-means with more methods
           -- for initializing centroids

Background information
----------------------
The k-means algorithm takes as input the number of clusters to
generate, k, and a set of observation vectors to cluster. It
returns a set of centroids, one for each of the k clusters. An
observation vector is classified with the cluster number or
centroid index of the centroid closest to it.

A vector v belongs to cluster i if it is closer to centroid i than
any other centroid. If v belongs to i, we say centroid i is the
dominating centroid of v. The k-means algorithm tries to
minimize distortion, which is defined as the sum of the squared distances
between each observation vector and its dominating centroid.
The minimization is achieved by iteratively reclassifying
the observations into clusters and recalculating the centroids until
a configuration is reached in which the centroids are stable. One can
also define a maximum number of iterations.

Since vector quantization is a natural application for k-means,
information theory terminology is often used. The centroid index
or cluster index is also referred to as a "code" and the table
mapping codes to centroids and, vice versa, is often referred to as a
"code book". The result of k-means, a set of centroids, can be
used to quantize vectors. Quantization aims to find an encoding of
vectors that reduces the expected distortion.

All routines expect obs to be an M by N array, where the rows are
the observation vectors. The codebook is a k by N array, where the
ith row is the centroid of code word i. The observation vectors
and centroids have the same feature dimension.

As an example, suppose we wish to compress a 24-bit color image
(each pixel is represented by one byte for red, one for blue, and
one for green) before sending it over the web. By using a smaller
8-bit encoding, we can reduce the amount of data by two
thirds. Ideally, the colors for each of the 256 possible 8-bit
encoding values should be chosen to minimize distortion of the
color. Running k-means with k=256 generates a code book of 256
codes, which fills up all possible 8-bit sequences. Instead of
sending a 3-byte value for each pixel, the 8-bit centroid index
(or code word) of the dominating centroid is transmitted. The code
book is also sent over the wire so each 8-bit code can be
translated back to a 24-bit pixel value representation. If the
image of interest was of an ocean, we would expect many 24-bit
blues to be represented by 8-bit codes. If it was an image of a
human face, more flesh-tone colors would be represented in the
code book.

"""
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
    _asarray, array_namespace, xp_size, xp_copy
)
from scipy._lib._util import (check_random_state, rng_integers,
                              _transition_to_rng)
from scipy._lib import array_api_extra as xpx
from scipy.spatial.distance import cdist

from . import _vq

__docformat__ = 'restructuredtext'

__all__ = ['whiten', 'vq', 'kmeans', 'kmeans2']


class ClusterError(Exception):
    pass


def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.

    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set by its standard deviation (i.e. "whiten"
    it - as in "white noise" where each frequency has equal power).
    Each feature is divided by its standard deviation across all observations
    to give it unit variance.

    Parameters
    ----------
    obs : ndarray
        Each row of the array is an observation.  The
        columns are the features seen during each observation.

        >>> #         f0    f1    f2
        >>> obs = [[  1.,   1.,   1.],  #o0
        ...        [  2.,   2.,   2.],  #o1
        ...        [  3.,   3.,   3.],  #o2
        ...        [  4.,   4.,   4.]]  #o3

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation
        of each column.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import whiten
    >>> features  = np.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7,]])
    >>> whiten(features)
    array([[ 4.17944278,  2.69811351,  7.21248917],
           [ 3.29956009,  2.93273208,  9.33380951],
           [ 1.75976538,  0.7038557 ,  7.21248917]])

    """
    xp = array_namespace(obs)
    obs = _asarray(obs, check_finite=check_finite, xp=xp)
    std_dev = xp.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if xp.any(zero_std_mask):
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning, stacklevel=2)
    return obs / std_dev


def vq(obs, code_book, check_finite=True):
    """
    Assign codes from a code book to observations.

    Assigns a code from a code book to each observation. Each
    observation vector in the 'M' by 'N' `obs` array is compared with the
    centroids in the code book and assigned the code of the closest
    centroid.

    The features in `obs` should have unit variance, which can be
    achieved by passing them through the whiten function. The code
    book can be created with the k-means algorithm or a different
    encoding algorithm.

    Parameters
    ----------
    obs : ndarray
        Each row of the 'M' x 'N' array is an observation. The columns are
        the "features" seen during each observation. The features must be
        whitened first using the whiten function or something equivalent.
    code_book : ndarray
        The code book is usually generated using the k-means algorithm.
        Each row of the array holds a different code, and the columns are
        the features of the code.

         >>> #              f0    f1    f2   f3
         >>> code_book = [
         ...             [  1.,   2.,   3.,   4.],  #c0
         ...             [  1.,   2.,   3.,   4.],  #c1
         ...             [  1.,   2.,   3.,   4.]]  #c2

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        A length M array holding the code book index for each observation.
    dist : ndarray
        The distortion (distance) between the observation and its nearest
        code.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import vq
    >>> code_book = np.array([[1., 1., 1.],
    ...                       [2., 2., 2.]])
    >>> features  = np.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7]])
    >>> vq(features, code_book)
    (array([1, 1, 0], dtype=int32), array([0.43588989, 0.73484692, 0.83066239]))

    """
    xp = array_namespace(obs, code_book)
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    code_book = _asarray(code_book, xp=xp, check_finite=check_finite)
    ct = xp.result_type(obs, code_book)

    c_obs = xp.astype(obs, ct, copy=False)
    c_code_book = xp.astype(code_book, ct, copy=False)

    if xp.isdtype(ct, kind='real floating'):
        c_obs = np.asarray(c_obs)
        c_code_book = np.asarray(c_code_book)
        result = _vq.vq(c_obs, c_code_book)
        return xp.asarray(result[0]), xp.asarray(result[1])
    return py_vq(obs, code_book, check_finite=False)


def py_vq(obs, code_book, check_finite=True):
    """ Python version of vq algorithm.

    The algorithm computes the Euclidean distance between each
    observation and every frame in the code_book.

    Parameters
    ----------
    obs : ndarray
        Expects a rank 2 array. Each row is one observation.
    code_book : ndarray
        Code book to use. Same format than obs. Should have same number of
        features (e.g., columns) than obs.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        code[i] gives the label of the ith obversation; its code is
        code_book[code[i]].
    mind_dist : ndarray
        min_dist[i] gives the distance between the ith observation and its
        corresponding code.

    Notes
    -----
    This function is slower than the C version but works for
    all input types. If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.

    """
    xp = array_namespace(obs, code_book)
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    code_book = _asarray(code_book, xp=xp, check_finite=check_finite)

    if obs.ndim != code_book.ndim:
        raise ValueError("Observation and code_book should have the same rank")

    if obs.ndim == 1:
        obs = obs[:, xp.newaxis]
        code_book = code_book[:, xp.newaxis]

    # Once `cdist` has array API support, this `xp.asarray` call can be removed
    dist = xp.asarray(cdist(obs, code_book))
    code = xp.argmin(dist, axis=1)
    min_dist = xp.min(dist, axis=1)
    return code, min_dist


def _kmeans(obs, guess, thresh=1e-5, xp=None):
    """ "raw" version of k-means.

    Returns
    -------
    code_book
        The lowest distortion codebook found.
    avg_dist
        The average distance a observation is from a code in the book.
        Lower means the code_book matches the data better.

    See Also
    --------
    kmeans : wrapper around k-means

    Examples
    --------
    Note: not whitened in this example.

    >>> import numpy as np
    >>> from scipy.cluster.vq import _kmeans
    >>> features  = np.array([[ 1.9,2.3],
    ...                       [ 1.5,2.5],
    ...                       [ 0.8,0.6],
    ...                       [ 0.4,1.8],
    ...                       [ 1.0,1.0]])
    >>> book = np.array((features[0],features[2]))
    >>> _kmeans(features,book)
    (array([[ 1.7       ,  2.4       ],
           [ 0.73333333,  1.13333333]]), 0.40563916697728591)

    """
    xp = np if xp is None else xp
    code_book = guess
    diff = xp.inf
    prev_avg_dists = deque([diff], maxlen=2)
    while diff > thresh:
        # compute membership and distances between obs and code_book
        obs_code, distort = vq(obs, code_book, check_finite=False)
        prev_avg_dists.append(xp.mean(distort, axis=-1))
        # recalc code_book as centroids of associated obs
        obs = np.asarray(obs)
        obs_code = np.asarray(obs_code)
        code_book, has_members = _vq.update_cluster_means(obs, obs_code,
                                                          code_book.shape[0])
        obs = xp.asarray(obs)
        obs_code = xp.asarray(obs_code)
        code_book = xp.asarray(code_book)
        has_members = xp.asarray(has_members)
        code_book = code_book[has_members]
        diff = xp.abs(prev_avg_dists[0] - prev_avg_dists[1])

    return code_book, prev_avg_dists[1]


@_transition_to_rng("seed")
def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True,
           *, rng=None):
    """
    Performs k-means on a set of observation vectors forming k clusters.

    The k-means algorithm adjusts the classification of the observations
    into clusters and updates the cluster centroids until the position of
    the centroids is stable over successive iterations. In this
    implementation of the algorithm, the stability of the centroids is
    determined by comparing the absolute value of the change in the average
    Euclidean distance between the observations and their corresponding
    centroids against a threshold. This yields
    a code book mapping centroids to codes and vice versa.

    Parameters
    ----------
    obs : ndarray
       Each row of the M by N array is an observation vector. The
       columns are the features seen during each observation.
       The features must be whitened first with the `whiten` function.

    k_or_guess : int or ndarray
       The number of centroids to generate. A code is assigned to
       each centroid, which is also the row index of the centroid
       in the code_book matrix generated.

       The initial k centroids are chosen by randomly selecting
       observations from the observation matrix. Alternatively,
       passing a k by N array specifies the initial k centroids.

    iter : int, optional
       The number of times to run k-means, returning the codebook
       with the lowest distortion. This argument is ignored if
       initial centroids are specified with an array for the
       ``k_or_guess`` parameter. This parameter does not represent the
       number of iterations of the k-means algorithm.

    thresh : float, optional
       Terminates the k-means algorithm if the change in
       distortion since the last k-means iteration is less than
       or equal to threshold.

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    codebook : ndarray
       A k by N array of k centroids. The ith centroid
       codebook[i] is represented with the code i. The centroids
       and codes generated represent the lowest distortion seen,
       not necessarily the globally minimal distortion.
       Note that the number of centroids is not necessarily the same as the
       ``k_or_guess`` parameter, because centroids assigned to no observations
       are removed during iterations.

    distortion : float
       The mean (non-squared) Euclidean distance between the observations
       passed and the centroids generated. Note the difference to the standard
       definition of distortion in the context of the k-means algorithm, which
       is the sum of the squared distances.

    See Also
    --------
    kmeans2 : a different implementation of k-means clustering
       with more methods for generating initial centroids but without
       using a distortion change threshold as a stopping criterion.

    whiten : must be called prior to passing an observation matrix
       to kmeans.

    Notes
    -----
    For more functionalities or optimal performance, you can use
    `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    `This <https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations>`_
    is a benchmark result of several implementations.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import vq, kmeans, whiten
    >>> import matplotlib.pyplot as plt
    >>> features  = np.array([[ 1.9,2.3],
    ...                       [ 1.5,2.5],
    ...                       [ 0.8,0.6],
    ...                       [ 0.4,1.8],
    ...                       [ 0.1,0.1],
    ...                       [ 0.2,1.8],
    ...                       [ 2.0,0.5],
    ...                       [ 0.3,1.5],
    ...                       [ 1.0,1.0]])
    >>> whitened = whiten(features)
    >>> book = np.array((whitened[0],whitened[2]))
    >>> kmeans(whitened,book)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)

    >>> codes = 3
    >>> kmeans(whitened,codes)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 1.32544402,  0.65607529],
           [ 0.40782893,  2.02786907]]), 0.5196582527686241)

    >>> # Create 50 datapoints in two clusters a and b
    >>> pts = 50
    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
    >>> b = rng.multivariate_normal([30, 10],
    ...                             [[10, 2], [2, 1]],
    ...                             size=pts)
    >>> features = np.concatenate((a, b))
    >>> # Whiten data
    >>> whitened = whiten(features)
    >>> # Find 2 clusters in the data
    >>> codebook, distortion = kmeans(whitened, 2)
    >>> # Plot whitened data and cluster centers in red
    >>> plt.scatter(whitened[:, 0], whitened[:, 1])
    >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    >>> plt.show()

    """
    if isinstance(k_or_guess, int):
        xp = array_namespace(obs)
    else:
        xp = array_namespace(obs, k_or_guess)
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    guess = _asarray(k_or_guess, xp=xp, check_finite=check_finite)
    if iter < 1:
        raise ValueError(f"iter must be at least 1, got {iter}")

    # Determine whether a count (scalar) or an initial guess (array) was passed.
    if xp_size(guess) != 1:
        if xp_size(guess) < 1:
            raise ValueError(f"Asked for 0 clusters. Initial book was {guess}")
        return _kmeans(obs, guess, thresh=thresh, xp=xp)

    # k_or_guess is a scalar, now verify that it's an integer
    k = int(guess)
    if k != guess:
        raise ValueError("If k_or_guess is a scalar, it must be an integer.")
    if k < 1:
        raise ValueError("Asked for %d clusters." % k)

    rng = check_random_state(rng)

    # initialize best distance value to a large value
    best_dist = xp.inf
    for i in range(iter):
        # the initial code book is randomly selected from observations
        guess = _kpoints(obs, k, rng, xp)
        book, dist = _kmeans(obs, guess, thresh=thresh, xp=xp)
        if dist < best_dist:
            best_book = book
            best_dist = dist
    return best_book, best_dist


def _kpoints(data, k, rng, xp):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    idx = rng.choice(data.shape[0], size=int(k), replace=False)
    # convert to array with default integer dtype (avoids numpy#25607)
    idx = xp.asarray(idx, dtype=xp.asarray([1]).dtype)
    return xp.take(data, idx, axis=0)


def _krandinit(data, k, rng, xp):
    """Returns k samples of a random variable whose parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable whose mean and covariances are the ones estimated from the data.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    mu = xp.mean(data, axis=0)
    k = np.asarray(k)

    if data.ndim == 1:
        _cov = xpx.cov(data, xp=xp)
        x = rng.standard_normal(size=k)
        x = xp.asarray(x)
        x *= xp.sqrt(_cov)
    elif data.shape[1] > data.shape[0]:
        # initialize when the covariance matrix is rank deficient
        _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
        x = rng.standard_normal(size=(k, xp_size(s)))
        x = xp.asarray(x)
        sVh = s[:, None] * vh / xp.sqrt(data.shape[0] - xp.asarray(1.))
        x = x @ sVh
    else:
        _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)

        # k rows, d cols (one row = one obs)
        # Generate k sample of a random variable ~ Gaussian(mu, cov)
        x = rng.standard_normal(size=(k, xp_size(mu)))
        x = xp.asarray(x)
        x = x @ xp.linalg.cholesky(_cov).T

    x += mu
    return x


def _kpp(data, k, rng, xp):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    """

    ndim = len(data.shape)
    if ndim == 1:
        data = data[:, None]

    dims = data.shape[1]

    init = xp.empty((int(k), dims))

    for i in range(k):
        if i == 0:
            init[i, :] = data[rng_integers(rng, data.shape[0]), :]

        else:
            D2 = cdist(init[:i,:], data, metric='sqeuclidean').min(axis=0)
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            r = rng.uniform()
            cumprobs = np.asarray(cumprobs)
            init[i, :] = data[np.searchsorted(cumprobs, r), :]

    if ndim == 1:
        init = init[:, 0]
    return init


_valid_init_meth = {'random': _krandinit, 'points': _kpoints, '++': _kpp}


def _missing_warn():
    """Print a warning when called."""
    warnings.warn("One of the clusters is empty. "
                  "Re-run kmeans with a different initialization.",
                  stacklevel=3)


def _missing_raise():
    """Raise a ClusterError when called."""
    raise ClusterError("One of the clusters is empty. "
                       "Re-run kmeans with a different initialization.")


_valid_miss_meth = {'warn': _missing_warn, 'raise': _missing_raise}


@_transition_to_rng("seed")
def kmeans2(data, k, iter=10, thresh=1e-5, minit='random',
            missing='warn', check_finite=True, *, rng=None):
    """
    Classify a set of observations into k clusters using the k-means algorithm.

    The algorithm attempts to minimize the Euclidean distance between
    observations and centroids. Several initialization methods are
    included.

    Parameters
    ----------
    data : ndarray
        A 'M' by 'N' array of 'M' observations in 'N' dimensions or a length
        'M' array of 'M' 1-D observations.
    k : int or ndarray
        The number of clusters to form as well as the number of
        centroids to generate. If `minit` initialization string is
        'matrix', or if a ndarray is given instead, it is
        interpreted as initial cluster to use instead.
    iter : int, optional
        Number of iterations of the k-means algorithm to run. Note
        that this differs in meaning from the iters parameter to
        the kmeans function.
    thresh : float, optional
        (not used yet)
    minit : str, optional
        Method for initialization. Available methods are 'random',
        'points', '++' and 'matrix':

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        'points': choose k observations (rows) at random from data for
        the initial centroids.

        '++': choose k observations accordingly to the kmeans++ method
        (careful seeding)

        'matrix': interpret the k parameter as a k by M (or length k
        array for 1-D data) array of initial centroids.
    missing : str, optional
        Method to deal with empty clusters. Available methods are
        'warn' and 'raise':

        'warn': give a warning and continue.

        'raise': raise an ClusterError and terminate the algorithm.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    centroid : ndarray
        A 'k' by 'N' array of centroids found at the last iteration of
        k-means.
    label : ndarray
        label[i] is the code or index of the centroid the
        ith observation is closest to.

    See Also
    --------
    kmeans

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.

    Examples
    --------
    >>> from scipy.cluster.vq import kmeans2
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    Create z, an array with shape (100, 2) containing a mixture of samples
    from three multivariate normal distributions.

    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 6], [[2, 1], [1, 1.5]], size=45)
    >>> b = rng.multivariate_normal([2, 0], [[1, -1], [-1, 3]], size=30)
    >>> c = rng.multivariate_normal([6, 4], [[5, 0], [0, 1.2]], size=25)
    >>> z = np.concatenate((a, b, c))
    >>> rng.shuffle(z)

    Compute three clusters.

    >>> centroid, label = kmeans2(z, 3, minit='points')
    >>> centroid
    array([[ 2.22274463, -0.61666946],  # may vary
           [ 0.54069047,  5.86541444],
           [ 6.73846769,  4.01991898]])

    How many points are in each cluster?

    >>> counts = np.bincount(label)
    >>> counts
    array([29, 51, 20])  # may vary

    Plot the clusters.

    >>> w0 = z[label == 0]
    >>> w1 = z[label == 1]
    >>> w2 = z[label == 2]
    >>> plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
    >>> plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
    >>> plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
    >>> plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
    >>> plt.axis('equal')
    >>> plt.legend(shadow=True)
    >>> plt.show()

    """
    if int(iter) < 1:
        raise ValueError(f"Invalid iter ({iter}), must be a positive integer.")
    try:
        miss_meth = _valid_miss_meth[missing]
    except KeyError as e:
        raise ValueError(f"Unknown missing method {missing!r}") from e

    if isinstance(k, int):
        xp = array_namespace(data)
    else:
        xp = array_namespace(data, k)
    data = _asarray(data, xp=xp, check_finite=check_finite)
    code_book = xp_copy(k, xp=xp)
    if data.ndim == 1:
        d = 1
    elif data.ndim == 2:
        d = data.shape[1]
    else:
        raise ValueError("Input of rank > 2 is not supported.")

    if xp_size(data) < 1 or xp_size(code_book) < 1:
        raise ValueError("Empty input is not supported.")

    # If k is not a single value, it should be compatible with data's shape
    if minit == 'matrix' or xp_size(code_book) > 1:
        if data.ndim != code_book.ndim:
            raise ValueError("k array doesn't match data rank")
        nc = code_book.shape[0]
        if data.ndim > 1 and code_book.shape[1] != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        nc = int(code_book)

        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters"
                             " (k was %s)" % (nc, code_book))
        elif nc != code_book:
            warnings.warn("k was not an integer, was converted.", stacklevel=2)

        try:
            init_meth = _valid_init_meth[minit]
        except KeyError as e:
            raise ValueError(f"Unknown init method {minit!r}") from e
        else:
            rng = check_random_state(rng)
            code_book = init_meth(data, code_book, rng, xp)

    data = np.asarray(data)
    code_book = np.asarray(code_book)
    for i in range(iter):
        # Compute the nearest neighbor for each obs using the current code book
        label = vq(data, code_book, check_finite=check_finite)[0]
        # Update the code book by computing centroids
        new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
        if not has_members.all():
            miss_meth()
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return xp.asarray(code_book), xp.asarray(label)
<|endoftext|>"""
Hierarchical clustering (:mod:`scipy.cluster.hierarchy`)
========================================================

.. currentmodule:: scipy.cluster.hierarchy

These functions cut hierarchical clusterings into flat clusterings
or find the roots of the forest formed by a cut by providing the flat
cluster ids of each observation.

.. autosummary::
   :toctree: generated/

   fcluster
   fclusterdata
   leaders

These are routines for agglomerative clustering.

.. autosummary::
   :toctree: generated/

   linkage
   single
   complete
   average
   weighted
   centroid
   median
   ward

These routines compute statistics on hierarchies.

.. autosummary::
   :toctree: generated/

   cophenet
   from_mlab_linkage
   inconsistent
   maxinconsts
   maxdists
   maxRstat
   to_mlab_linkage

Routines for visualizing flat clusters.

.. autosummary::
   :toctree: generated/

   dendrogram

These are data structures and routines for representing hierarchies as
tree objects.

.. autosummary::
   :toctree: generated/

   ClusterNode
   leaves_list
   to_tree
   cut_tree
   optimal_leaf_ordering

These are predicates for checking the validity of linkage and
inconsistency matrices as well as for checking isomorphism of two
flat cluster assignments.

.. autosummary::
   :toctree: generated/

   is_valid_im
   is_valid_linkage
   is_isomorphic
   is_monotonic
   correspond
   num_obs_linkage

Utility routines for plotting:

.. autosummary::
   :toctree: generated/

   set_link_color_palette

Utility classes:

.. autosummary::
   :toctree: generated/

   DisjointSet -- data structure for incremental connectivity queries

"""
# Copyright (C) Damian Eads, 2007-2008. New BSD License.

# hierarchy.py (derived from cluster.py, http://scipy-cluster.googlecode.com)
#
# Author: Damian Eads
# Date:   September 22, 2007
#
# Copyright (c) 2007, 2008, Damian Eads
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import bisect
from collections import deque

import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, _asarray, xp_copy, is_jax
from scipy._lib._disjoint_set import DisjointSet


_LINKAGE_METHODS = {'single': 0, 'complete': 1, 'average': 2, 'centroid': 3,
                    'median': 4, 'ward': 5, 'weighted': 6}
_EUCLIDEAN_METHODS = ('centroid', 'median', 'ward')

__all__ = ['ClusterNode', 'DisjointSet', 'average', 'centroid', 'complete',
           'cophenet', 'correspond', 'cut_tree', 'dendrogram', 'fcluster',
           'fclusterdata', 'from_mlab_linkage', 'inconsistent',
           'is_isomorphic', 'is_monotonic', 'is_valid_im', 'is_valid_linkage',
           'leaders', 'leaves_list', 'linkage', 'maxRstat', 'maxdists',
           'maxinconsts', 'median', 'num_obs_linkage', 'optimal_leaf_ordering',
           'set_link_color_palette', 'single', 'to_mlab_linkage', 'to_tree',
           'ward', 'weighted']


class ClusterWarning(UserWarning):
    pass


def _warning(s):
    warnings.warn(f'scipy.cluster: {s}', ClusterWarning, stacklevel=3)


def int_floor(arr, xp):
    # array_api_strict is strict about not allowing `int()` on a float array.
    # That's typically not needed, here it is - so explicitly convert
    return int(xp.astype(xp.asarray(arr), xp.int64))


def single(y):
    """
    Perform single/min/nearest linkage on the condensed distance matrix ``y``.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        The linkage matrix.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import single, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = single(y)
    >>> Z
    array([[ 0.,  1.,  1.,  2.],
           [ 2., 12.,  1.,  3.],
           [ 3.,  4.,  1.,  2.],
           [ 5., 14.,  1.,  3.],
           [ 6.,  7.,  1.,  2.],
           [ 8., 16.,  1.,  3.],
           [ 9., 10.,  1.,  2.],
           [11., 18.,  1.,  3.],
           [13., 15.,  2.,  6.],
           [17., 20.,  2.,  9.],
           [19., 21.,  2., 12.]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  4,  5,  6,  1,  2,  3], dtype=int32)
    >>> fcluster(Z, 1, criterion='distance')
    array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)
    >>> fcluster(Z, 2, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.
    """
    return linkage(y, method='single', metric='euclidean')


def complete(y):
    """
    Perform complete/max/farthest point linkage on a condensed distance matrix.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        the `linkage` function documentation for more information
        on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import complete, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = complete(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.41421356,  3.        ],
           [ 5.        , 13.        ,  1.41421356,  3.        ],
           [ 8.        , 14.        ,  1.41421356,  3.        ],
           [11.        , 15.        ,  1.41421356,  3.        ],
           [16.        , 17.        ,  4.12310563,  6.        ],
           [18.        , 19.        ,  4.12310563,  6.        ],
           [20.        , 21.        ,  5.65685425, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    >>> fcluster(Z, 4.5, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.
    """
    return linkage(y, method='complete', metric='euclidean')


def average(y):
    """
    Perform average/UPGMA linkage on a condensed distance matrix.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        `linkage` for more information on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import average, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = average(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.20710678,  3.        ],
           [ 5.        , 13.        ,  1.20710678,  3.        ],
           [ 8.        , 14.        ,  1.20710678,  3.        ],
           [11.        , 15.        ,  1.20710678,  3.        ],
           [16.        , 17.        ,  3.39675184,  6.        ],
           [18.        , 19.        ,  3.39675184,  6.        ],
           [20.        , 21.        ,  4.09206523, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='average', metric='euclidean')


def weighted(y):
    """
    Perform weighted/WPGMA linkage on the condensed distance matrix.

    See `linkage` for more information on the return
    structure and algorithm.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        `linkage` for more information on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import weighted, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = weighted(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 11.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.20710678,  3.        ],
           [ 8.        , 13.        ,  1.20710678,  3.        ],
           [ 5.        , 14.        ,  1.20710678,  3.        ],
           [10.        , 15.        ,  1.20710678,  3.        ],
           [18.        , 19.        ,  3.05595762,  6.        ],
           [16.        , 17.        ,  3.32379407,  6.        ],
           [20.        , 21.        ,  4.06357713, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9,  1,  2,  3, 10, 11, 12,  4,  6,  5], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([3, 3, 3, 1, 1, 1, 4, 4, 4, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    array([2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='weighted', metric='euclidean')


def centroid(y):
    """
    Perform centroid/UPGMC linkage.

    See `linkage` for more information on the input matrix,
    return structure, and algorithm.

    The following are common calling conventions:

    1. ``Z = centroid(y)``

       Performs centroid/UPGMC linkage on the condensed distance
       matrix ``y``.

    2. ``Z = centroid(X)``

       Performs centroid/UPGMC linkage on the observation matrix ``X``
       using Euclidean distance as the distance metric.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns. Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        the `linkage` function documentation for more information
        on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import centroid, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = centroid(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.33333333,  6.        ],
           [16.        , 17.        ,  3.33333333,  6.        ],
           [20.        , 21.        ,  3.33333333, 12.        ]]) # may vary

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32) # may vary
    >>> fcluster(Z, 1.1, criterion='distance')
    array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32) # may vary
    >>> fcluster(Z, 2, criterion='distance')
    array([3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2], dtype=int32) # may vary
    >>> fcluster(Z, 4, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='centroid', metric='euclidean')


def median(y):
    """
    Perform median/WPGMC linkage.

    See `linkage` for more information on the return structure
    and algorithm.

     The following are common calling conventions:

     1. ``Z = median(y)``

        Performs median/WPGMC linkage on the condensed distance matrix
        ``y``.  See ``linkage`` for more information on the return
        structure and algorithm.

     2. ``Z = median(X)``

        Performs median/WPGMC linkage on the observation matrix ``X``
        using Euclidean distance as the distance metric. See `linkage`
        for more information on the return structure and algorithm.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns.  Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = median(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32)
    >>> fcluster(Z, 1.1, criterion='distance')
    array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32)
    >>> fcluster(Z, 2, criterion='distance')
    array([3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='median', metric='euclidean')


def ward(y):
    """
    Perform Ward's linkage on a condensed distance matrix.

    See `linkage` for more information on the return structure
    and algorithm.

    The following are common calling conventions:

    1. ``Z = ward(y)``
       Performs Ward's linkage on the condensed distance matrix ``y``.

    2. ``Z = ward(X)``
       Performs Ward's linkage on the observation matrix ``X`` using
       Euclidean distance as the distance metric.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns.  Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix. See
        `linkage` for more information on the return structure and
        algorithm.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = ward(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.1, criterion='distance')
    array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)
    >>> fcluster(Z, 3, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    >>> fcluster(Z, 9, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='ward', metric='euclidean')


def linkage(y, method='single', metric='euclidean', optimal_ordering=False):
    """
    Perform hierarchical/agglomerative clustering.

    The input y may be either a 1-D condensed distance matrix
    or a 2-D array of observation vectors.

    If y is a 1-D condensed distance matrix,
    then y must be a :math:`\\binom{n}{2}` sized
    vector, where n is the number of original observations paired
    in the distance matrix. The behavior of this function is very
    similar to the MATLAB linkage function.

    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.

    The following linkage methods are used to compute the distance
    :math:`d(s, t)` between two clusters :math:`s` and
    :math:`t`. The algorithm begins with a forest of clusters that
    have yet to be used in the hierarchy being formed. When two
    clusters :math:`s` and :math:`t` from this forest are combined
    into a single cluster :math:`u`, :math:`s` and :math:`t` are
    removed from the forest, and :math:`u` is added to the
    forest. When only one cluster remains in the forest, the algorithm
    stops, and this cluster becomes the root.

    A distance matrix is maintained at each iteration. The ``d[i,j]``
    entry corresponds to the distance between cluster :math:`i` and
    :math:`j` in the original forest.

    At each iteration, the algorithm must update the distance matrix
    to reflect the distance of the newly formed cluster u with the
    remaining clusters in the forest.

    Suppose there are :math:`|u|` original observations
    :math:`u[0], \\ldots, u[|u|-1]` in cluster :math:`u` and
    :math:`|v|` original objects :math:`v[0], \\ldots, v[|v|-1]` in
    cluster :math:`v`. Recall, :math:`s` and :math:`t` are
    combined to form cluster :math:`u`. Let :math:`v` be any
    remaining cluster in the forest that is not :math:`u`.

    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

      * method='single' assigns

        .. math::
           d(u,v) = \\min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.

      * method='complete' assigns

        .. math::
           d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

      * method='average' assigns

        .. math::
           d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}
                                   {(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.

      * method='weighted' assigns

        .. math::
           d(u,v) = (dist(s,v) + dist(t,v))/2

        where cluster u was formed with cluster s and t and v
        is a remaining cluster in the forest (also called WPGMA).

      * method='centroid' assigns

        .. math::
           dist(s,t) = ||c_s-c_t||_2

        where :math:`c_s` and :math:`c_t` are the centroids of
        clusters :math:`s` and :math:`t`, respectively. When two
        clusters :math:`s` and :math:`t` are combined into a new
        cluster :math:`u`, the new centroid is computed over all the
        original objects in clusters :math:`s` and :math:`t`. The
        distance then becomes the Euclidean distance between the
        centroid of :math:`u` and the centroid of a remaining cluster
        :math:`v` in the forest. This is also known as the UPGMC
        algorithm.

      * method='median' assigns :math:`d(s,t)` like the ``centroid``
        method. When two clusters :math:`s` and :math:`t` are combined
        into a new cluster :math:`u`, the average of centroids s and t
        give the new centroid :math:`u`. This is also known as the
        WPGMC algorithm.

      * method='ward' uses the Ward variance minimization algorithm.
        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    implementation may choose a different minimum than the MATLAB
    version.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed distance matrix
        is a flat array containing the upper triangular of the distance matrix.
        This is the form that ``pdist`` returns. Alternatively, a collection of
        :math:`m` observation vectors in :math:`n` dimensions may be passed as
        an :math:`m` by :math:`n` array. All elements of the condensed distance
        matrix must be finite, i.e., no NaNs or infs.
    method : str, optional
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions.
    metric : str or function, optional
        The distance metric to use in the case that y is a collection of
        observation vectors; ignored otherwise. See the ``pdist``
        function for a list of valid distance metrics. A custom distance
        function can also be used.
    optimal_ordering : bool, optional
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal. This results in a more intuitive
        tree structure when the data are visualized. defaults to False, because
        this algorithm can be slow, particularly on large datasets [2]_. See
        also the `optimal_leaf_ordering` function.

        .. versionadded:: 1.0.0

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix.

    Notes
    -----
    1. For method 'single', an optimized algorithm based on minimum spanning
       tree is implemented. It has time complexity :math:`O(n^2)`.
       For methods 'complete', 'average', 'weighted' and 'ward', an algorithm
       called nearest-neighbors chain is implemented. It also has time
       complexity :math:`O(n^2)`.
       For other methods, a naive algorithm is implemented with :math:`O(n^3)`
       time complexity.
       All algorithms use :math:`O(n^2)` memory.
       Refer to [1]_ for details about the algorithms.
    2. Methods 'centroid', 'median', and 'ward' are correctly defined only if
       Euclidean pairwise metric is used. If `y` is passed as precomputed
       pairwise distances, then it is the user's responsibility to assure that
       these distances are in fact Euclidean, otherwise the produced result
       will be incorrect.

    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
           algorithms", :arXiv:`1109.2378v1`.
    .. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal
           leaf ordering for hierarchical clustering", 2001. Bioinformatics
           :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    Examples
    --------
    >>> from scipy.cluster.hierarchy import dendrogram, linkage
    >>> from matplotlib import pyplot as plt
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

    >>> Z = linkage(X, 'ward')
    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)

    >>> Z = linkage(X, 'single')
    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)
    >>> plt.show()
    """
    xp = array_namespace(y)
    y = _asarray(y, order='C', dtype=xp.float64, xp=xp)

    if method not in _LINKAGE_METHODS:
        raise ValueError(f"Invalid method: {method}")

    if method in _EUCLIDEAN_METHODS and metric != 'euclidean' and y.ndim == 2:
        msg = f"`method={method}` requires the distance metric to be Euclidean"
        raise ValueError(msg)

    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
    elif y.ndim == 2:
        if (y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and
                xp.all(y >= 0) and np.allclose(y, y.T)):
            warnings.warn('The symmetric non-negative hollow observation '
                          'matrix looks suspiciously like an uncondensed '
                          'distance matrix',
                          ClusterWarning, stacklevel=2)
        y = distance.pdist(y, metric)
        y = xp.asarray(y)
    else:
        raise ValueError("`y` must be 1 or 2 dimensional.")

    if not xp.all(xp.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")

    n = int(distance.num_obs_y(y))
    method_code = _LINKAGE_METHODS[method]

    y = np.asarray(y)
    if method == 'single':
        result = _hierarchy.mst_single_linkage(y, n)
    elif method in ['complete', 'average', 'weighted', 'ward']:
        result = _hierarchy.nn_chain(y, n, method_code)
    else:
        result = _hierarchy.fast_linkage(y, n, method_code)
    result = xp.asarray(result)

    if optimal_ordering:
        y = xp.asarray(y)
        return optimal_leaf_ordering(result, y)
    else:
        return result


class ClusterNode:
    """
    A tree node class for representing a cluster.

    Leaf nodes correspond to original observations, while non-leaf nodes
    correspond to non-singleton clusters.

    The `to_tree` function converts a matrix returned by the linkage
    function into an easy-to-use tree representation.

    All parameter names are also attributes.

    Parameters
    ----------
    id : int
        The node id.
    left : ClusterNode instance, optional
        The left child tree node.
    right : ClusterNode instance, optional
        The right child tree node.
    dist : float, optional
        Distance for this cluster in the linkage matrix.
    count : int, optional
        The number of samples in this cluster.

    See Also
    --------
    to_tree : for converting a linkage matrix ``Z`` into a tree object.

    """

    def __init__(self, id, left=None, right=None, dist=0.0, count=1):
        if id < 0:
            raise ValueError('The id must be non-negative.')
        if dist < 0:
            raise ValueError('The distance must be non-negative.')
        if (left is None and right is not None) or \
           (left is not None and right is None):
            raise ValueError('Only full or proper binary trees are permitted.'
                             '  This node has one child.')
        if count < 1:
            raise ValueError('A cluster must contain at least one original '
                             'observation.')
        self.id = id
        self.left = left
        self.right = right
        self.dist = dist
        if self.left is None:
            self.count = count
        else:
            self.count = left.count + right.count

    def __lt__(self, node):
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist < node.dist

    def __gt__(self, node):
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist > node.dist

    def __eq__(self, node):
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist == node.dist

    def get_id(self):
        """
        The identifier of the target node.

        For ``0 <= i < n``, `i` corresponds to original observation i.
        For ``n <= i < 2n-1``, `i` corresponds to non-singleton cluster formed
        at iteration ``i-n``.

        Returns
        -------
        id : int
            The identifier of the target node.

        """
        return self.id

    def get_count(self):
        """
        The number of leaf nodes (original observations) belonging to
        the cluster node nd. If the target node is a leaf, 1 is
        returned.

        Returns
        -------
        get_count : int
            The number of leaf nodes below the target node.

        """
        return self.count

    def get_left(self):
        """
        Return a reference to the left child tree object.

        Returns
        -------
        left : ClusterNode
            The left child of the target node. If the node is a leaf,
            None is returned.

        """
        return self.left

    def get_right(self):
        """
        Return a reference to the right child tree object.

        Returns
        -------
        right : ClusterNode
            The left child of the target node. If the node is a leaf,
            None is returned.

        """
        return self.right

    def is_leaf(self):
        """
        Return True if the target node is a leaf.

        Returns
        -------
        leafness : bool
            True if the target node is a leaf node.

        """
        return self.left is None

    def pre_order(self, func=(lambda x: x.id)):
        """
        Perform pre-order traversal without recursive function calls.

        When a leaf node is first encountered, ``func`` is called with
        the leaf node as its argument, and its result is appended to
        the list.

        For example, the statement::

           ids = root.pre_order(lambda x: x.id)

        returns a list of the node ids corresponding to the leaf nodes
        of the tree as they appear from left to right.

        Parameters
        ----------
        func : function
            Applied to each leaf ClusterNode object in the pre-order traversal.
            Given the ``i``-th leaf node in the pre-order traversal ``n[i]``,
            the result of ``func(n[i])`` is stored in ``L[i]``. If not
            provided, the index of the original observation to which the node
            corresponds is used.

        Returns
        -------
        L : list
            The pre-order traversal.

        """
        # Do a preorder traversal, caching the result. To avoid having to do
        # recursion, we'll store the previous index we've visited in a vector.
        n = self.count

        curNode = [None] * (2 * n)
        lvisited = set()
        rvisited = set()
        curNode[0] = self
        k = 0
        preorder = []
        while k >= 0:
            nd = curNode[k]
            ndid = nd.id
            if nd.is_leaf():
                preorder.append(func(nd))
                k = k - 1
            else:
                if ndid not in lvisited:
                    curNode[k + 1] = nd.left
                    lvisited.add(ndid)
                    k = k + 1
                elif ndid not in rvisited:
                    curNode[k + 1] = nd.right
                    rvisited.add(ndid)
                    k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
                else:
                    k = k - 1

        return preorder


_cnode_bare = ClusterNode(0)
_cnode_type = type(ClusterNode)


def _order_cluster_tree(Z):
    """
    Return clustering nodes in bottom-up order by distance.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.

    Returns
    -------
    nodes : list
        A list of ClusterNode objects.
    """
    q = deque()
    tree = to_tree(Z)
    q.append(tree)
    nodes = []

    while q:
        node = q.popleft()
        if not node.is_leaf():
            bisect.insort_left(nodes, node)
            q.append(node.get_right())
            q.append(node.get_left())
    return nodes


def cut_tree(Z, n_clusters=None, height=None):
    """
    Given a linkage matrix Z, return the cut tree.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.
    n_clusters : array_like, optional
        Number of clusters in the tree at the cut point.
    height : array_like, optional
        The height at which to cut the tree. Only possible for ultrametric
        trees.

    Returns
    -------
    cutree : array
        An array indicating group membership at each agglomeration step. I.e.,
        for a full cut tree, in the first column each data point is in its own
        cluster. At the next step, two nodes are merged. Finally, all
        singleton and non-singleton clusters are in one group. If `n_clusters`
        or `height` are given, the columns correspond to the columns of
        `n_clusters` or `height`.

    Examples
    --------
    >>> from scipy import cluster
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> X = rng.random((50, 4))
    >>> Z = cluster.hierarchy.ward(X)
    >>> cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
    >>> cutree[:10]
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3],
           [3, 4],
           [2, 2],
           [0, 0],
           [1, 5],
           [3, 6],
           [4, 7]])  # random

    """
    xp = array_namespace(Z)
    nobs = num_obs_linkage(Z)
    nodes = _order_cluster_tree(Z)

    if height is not None and n_clusters is not None:
        raise ValueError("At least one of either height or n_clusters "
                         "must be None")
    elif height is None and n_clusters is None:  # return the full cut tree
        cols_idx = xp.arange(nobs)
    elif height is not None:
        height = xp.asarray(height)
        heights = xp.asarray([x.dist for x in nodes])
        cols_idx = xp.searchsorted(heights, height)
    else:
        n_clusters = xp.asarray(n_clusters)
        cols_idx = nobs - xp.searchsorted(xp.arange(nobs), n_clusters)

    try:
        n_cols = len(cols_idx)
    except TypeError:  # scalar
        n_cols = 1
        cols_idx = xp.asarray([cols_idx])

    groups = xp.zeros((n_cols, nobs), dtype=xp.int64)
    last_group = xp.arange(nobs)
    if 0 in cols_idx:
        groups[0] = last_group

    for i, node in enumerate(nodes):
        idx = node.pre_order()
        this_group = xp_copy(last_group, xp=xp)
        # TODO ARRAY_API complex indexing not supported
        this_group[idx] = xp.min(last_group[idx])
        this_group[this_group > xp.max(last_group[idx])] -= 1
        if i + 1 in cols_idx:
            groups[np.nonzero(i + 1 == cols_idx)[0]] = this_group
        last_group = this_group

    return groups.T


def to_tree(Z, rd=False):
    """
    Convert a linkage matrix into an easy-to-use tree object.

    The reference to the root `ClusterNode` object is returned (by default).

    Each `ClusterNode` object has a ``left``, ``right``, ``dist``, ``id``,
    and ``count`` attribute. The left and right attributes point to
    ClusterNode objects that were combined to generate the cluster.
    If both are None then the `ClusterNode` object is a leaf node, its count
    must be 1, and its distance is meaningless but set to 0.

    *Note: This function is provided for the convenience of the library
    user. ClusterNodes are not used as input to any of the functions in this
    library.*

    Parameters
    ----------
    Z : ndarray
        The linkage matrix in proper form (see the `linkage`
        function documentation).
    rd : bool, optional
        When False (default), a reference to the root `ClusterNode` object is
        returned.  Otherwise, a tuple ``(r, d)`` is returned. ``r`` is a
        reference to the root node while ``d`` is a list of `ClusterNode`
        objects - one per original entry in the linkage matrix plus entries
        for all clustering steps. If a cluster id is
        less than the number of samples ``n`` in the data that the linkage
        matrix describes, then it corresponds to a singleton cluster (leaf
        node).
        See `linkage` for more information on the assignment of cluster ids
        to clusters.

    Returns
    -------
    tree : ClusterNode or tuple (ClusterNode, list of ClusterNode)
        If ``rd`` is False, a `ClusterNode`.
        If ``rd`` is True, a list of length ``2*n - 1``, with ``n`` the number
        of samples.  See the description of `rd` above for more details.

    See Also
    --------
    linkage, is_valid_linkage, ClusterNode

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> x = rng.random((5, 2))
    >>> Z = hierarchy.linkage(x)
    >>> hierarchy.to_tree(Z)
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
    >>> rootnode
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> len(nodelist)
    9

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='c', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    # Number of original objects is equal to the number of rows plus 1.
    n = Z.shape[0] + 1

    # Create a list full of None's to store the node objects
    d = [None] * (n * 2 - 1)

    # Create the nodes corresponding to the n original objects.
    for i in range(0, n):
        d[i] = ClusterNode(i)

    nd = None

    for i in range(Z.shape[0]):
        row = Z[i, :]

        fi = int_floor(row[0], xp)
        fj = int_floor(row[1], xp)
        if fi > i + n:
            raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                              'is used before it is formed. See row %d, '
                              'column 0') % fi)
        if fj > i + n:
            raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                              'is used before it is formed. See row %d, '
                              'column 1') % fj)

        nd = ClusterNode(i + n, d[fi], d[fj], row[2])
        #                ^ id   ^ left ^ right ^ dist
        if row[3] != nd.count:
            raise ValueError(('Corrupt matrix Z. The count Z[%d,3] is '
                              'incorrect.') % i)
        d[n + i] = nd

    if rd:
        return (nd, d)
    else:
        return nd


def optimal_leaf_ordering(Z, y, metric='euclidean'):
    """
    Given a linkage matrix Z and distance, reorder the cut tree.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix. See
        `linkage` for more information on the return structure and
        algorithm.
    y : ndarray
        The condensed distance matrix from which Z was generated.
        Alternatively, a collection of m observation vectors in n
        dimensions may be passed as an m by n array.
    metric : str or function, optional
        The distance metric to use in the case that y is a collection of
        observation vectors; ignored otherwise. See the ``pdist``
        function for a list of valid distance metrics. A custom distance
        function can also be used.

    Returns
    -------
    Z_ordered : ndarray
        A copy of the linkage matrix Z, reordered to minimize the distance
        between adjacent leaves.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> X = rng.standard_normal((10, 10))
    >>> Z = hierarchy.ward(X)
    >>> hierarchy.leaves_list(Z)
    array([0, 3, 1, 9, 2, 5, 7, 4, 6, 8], dtype=int32)
    >>> hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
    array([3, 0, 2, 5, 7, 4, 8, 6, 9, 1], dtype=int32)

    """
    xp = array_namespace(Z, y)
    Z = _asarray(Z, order='C', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    y = _asarray(y, order='C', dtype=xp.float64, xp=xp)

    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
    elif y.ndim == 2:
        if (y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and
                np.all(y >= 0) and np.allclose(y, y.T)):
            warnings.warn('The symmetric non-negative hollow observation '
                          'matrix looks suspiciously like an uncondensed '
                          'distance matrix',
                          ClusterWarning, stacklevel=2)
        y = distance.pdist(y, metric)
        y = xp.asarray(y)
    else:
        raise ValueError("`y` must be 1 or 2 dimensional.")

    if not xp.all(xp.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")

    Z = np.asarray(Z)
    y = np.asarray(y)
    return xp.asarray(_optimal_leaf_ordering.optimal_leaf_ordering(Z, y))


def cophenet(Z, Y=None):
    """
    Calculate the cophenetic distances between each observation in
    the hierarchical clustering defined by the linkage ``Z``.

    Suppose ``p`` and ``q`` are original observations in
    disjoint clusters ``s`` and ``t``, respectively and
    ``s`` and ``t`` are joined by a direct parent cluster
    ``u``. The cophenetic distance between observations
    ``i`` and ``j`` is simply the distance between
    clusters ``s`` and ``t``.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as an array
        (see `linkage` function).
    Y : ndarray (optional)
        Calculates the cophenetic correlation coefficient ``c`` of a
        hierarchical clustering defined by the linkage matrix `Z`
        of a set of :math:`n` observations in :math:`m`
        dimensions. `Y` is the condensed distance matrix from which
        `Z` was generated.

    Returns
    -------
    c : ndarray
        The cophentic correlation distance (if ``Y`` is passed).
    d : ndarray
        The cophenetic distance matrix in condensed form. The
        :math:`ij` th entry is the cophenetic distance between
        original observations :math:`i` and :math:`j`.

    See Also
    --------
    linkage :
        for a description of what a linkage matrix is.
    scipy.spatial.distance.squareform :
        transforming condensed matrices into square ones.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import single, cophenet
    >>> from scipy.spatial.distance import pdist, squareform

    Given a dataset ``X`` and a linkage matrix ``Z``, the cophenetic distance
    between two points of ``X`` is the distance between the largest two
    distinct clusters that each of the points:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    ``X`` corresponds to this dataset ::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> Z = single(pdist(X))
    >>> Z
    array([[ 0.,  1.,  1.,  2.],
           [ 2., 12.,  1.,  3.],
           [ 3.,  4.,  1.,  2.],
           [ 5., 14.,  1.,  3.],
           [ 6.,  7.,  1.,  2.],
           [ 8., 16.,  1.,  3.],
           [ 9., 10.,  1.,  2.],
           [11., 18.,  1.,  3.],
           [13., 15.,  2.,  6.],
           [17., 20.,  2.,  9.],
           [19., 21.,  2., 12.]])
    >>> cophenet(Z)
    array([1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2.,
           2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 2., 2.,
           2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
           1., 1., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 1., 1., 1.])

    The output of the `scipy.cluster.hierarchy.cophenet` method is
    represented in condensed form. We can use
    `scipy.spatial.distance.squareform` to see the output as a
    regular matrix (where each element ``ij`` denotes the cophenetic distance
    between each ``i``, ``j`` pair of points in ``X``):

    >>> squareform(cophenet(Z))
    array([[0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 0., 1., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 0., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 1., 0., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 0., 1., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 0., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 1., 0., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0.]])

    In this example, the cophenetic distance between points on ``X`` that are
    very close (i.e., in the same corner) is 1. For other pairs of points is 2,
    because the points will be located in clusters at different
    corners - thus, the distance between these clusters will be larger.

    """
    xp = array_namespace(Z, Y)
    # Ensure float64 C-contiguous array. Cython code doesn't deal with striding.
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    zz = np.zeros((n * (n-1)) // 2, dtype=np.float64)

    Z = np.asarray(Z)
    _hierarchy.cophenetic_distances(Z, zz, int(n))
    zz = xp.asarray(zz)
    if Y is None:
        return zz

    Y = _asarray(Y, order='C', xp=xp)
    distance.is_valid_y(Y, throw=True, name='Y')

    z = xp.mean(zz)
    y = xp.mean(Y)
    Yy = Y - y
    Zz = zz - z
    numerator = (Yy * Zz)
    denomA = Yy**2
    denomB = Zz**2
    c = xp.sum(numerator) / xp.sqrt(xp.sum(denomA) * xp.sum(denomB))
    return (c, zz)


def inconsistent(Z, d=2):
    r"""
    Calculate inconsistency statistics on a linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The :math:`(n-1)` by 4 matrix encoding the linkage (hierarchical
        clustering).  See `linkage` documentation for more information on its
        form.
    d : int, optional
        The number of links up to `d` levels below each non-singleton cluster.

    Returns
    -------
    R : ndarray
        A :math:`(n-1)` by 4 matrix where the ``i``'th row contains the link
        statistics for the non-singleton cluster ``i``. The link statistics are
        computed over the link heights for links :math:`d` levels below the
        cluster ``i``. ``R[i,0]`` and ``R[i,1]`` are the mean and standard
        deviation of the link heights, respectively; ``R[i,2]`` is the number
        of links included in the calculation; and ``R[i,3]`` is the
        inconsistency coefficient,

        .. math:: \frac{\mathtt{Z[i,2]} - \mathtt{R[i,0]}} {R[i,1]}

    Notes
    -----
    This function behaves similarly to the MATLAB(TM) ``inconsistent``
    function.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import inconsistent, linkage
    >>> from matplotlib import pyplot as plt
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
    >>> Z = linkage(X, 'ward')
    >>> print(Z)
    [[ 5.          6.          0.          2.        ]
     [ 2.          7.          0.          2.        ]
     [ 0.          4.          1.          2.        ]
     [ 1.          8.          1.15470054  3.        ]
     [ 9.         10.          2.12132034  4.        ]
     [ 3.         12.          4.11096096  5.        ]
     [11.         13.         14.07183949  8.        ]]
    >>> inconsistent(Z)
    array([[ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 1.        ,  0.        ,  1.        ,  0.        ],
           [ 0.57735027,  0.81649658,  2.        ,  0.70710678],
           [ 1.04044011,  1.06123822,  3.        ,  1.01850858],
           [ 3.11614065,  1.40688837,  2.        ,  0.70710678],
           [ 6.44583366,  6.76770586,  3.        ,  1.12682288]])

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    if (not d == np.floor(d)) or d < 0:
        raise ValueError('The second argument d must be a nonnegative '
                         'integer value.')

    n = Z.shape[0] + 1
    R = np.zeros((n - 1, 4), dtype=np.float64)

    Z = np.asarray(Z)
    _hierarchy.inconsistent(Z, R, int(n), int(d))
    R = xp.asarray(R)
    return R


def from_mlab_linkage(Z):
    """
    Convert a linkage matrix generated by MATLAB(TM) to a new
    linkage matrix compatible with this module.

    The conversion does two things:

     * the indices are converted from ``1..N`` to ``0..(N-1)`` form,
       and

     * a fourth column ``Z[:,3]`` is added where ``Z[i,3]`` represents the
       number of original observations (leaves) in the non-singleton
       cluster ``i``.

    This function is useful when loading in linkages from legacy data
    files generated by MATLAB.

    Parameters
    ----------
    Z : ndarray
        A linkage matrix generated by MATLAB(TM).

    Returns
    -------
    ZS : ndarray
        A linkage matrix compatible with ``scipy.cluster.hierarchy``.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    to_mlab_linkage : transform from SciPy to MATLAB format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.hierarchy import ward, from_mlab_linkage

    Given a linkage matrix in MATLAB format ``mZ``, we can use
    `scipy.cluster.hierarchy.from_mlab_linkage` to import
    it into SciPy format:

    >>> mZ = np.array([[1, 2, 1], [4, 5, 1], [7, 8, 1],
    ...                [10, 11, 1], [3, 13, 1.29099445],
    ...                [6, 14, 1.29099445],
    ...                [9, 15, 1.29099445],
    ...                [12, 16, 1.29099445],
    ...                [17, 18, 5.77350269],
    ...                [19, 20, 5.77350269],
    ...                [21, 22,  8.16496581]])

    >>> Z = from_mlab_linkage(mZ)
    >>> Z
    array([[  0.        ,   1.        ,   1.        ,   2.        ],
           [  3.        ,   4.        ,   1.        ,   2.        ],
           [  6.        ,   7.        ,   1.        ,   2.        ],
           [  9.        ,  10.        ,   1.        ,   2.        ],
           [  2.        ,  12.        ,   1.29099445,   3.        ],
           [  5.        ,  13.        ,   1.29099445,   3.        ],
           [  8.        ,  14.        ,   1.29099445,   3.        ],
           [ 11.        ,  15.        ,   1.29099445,   3.        ],
           [ 16.        ,  17.        ,   5.77350269,   6.        ],
           [ 18.        ,  19.        ,   5.77350269,   6.        ],
           [ 20.        ,  21.        ,   8.16496581,  12.        ]])

    As expected, the linkage matrix ``Z`` returned includes an
    additional column counting the number of original samples in
    each cluster. Also, all cluster indices are reduced by 1
    (MATLAB format uses 1-indexing, whereas SciPy uses 0-indexing).

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, dtype=xp.float64, order='C', xp=xp)
    Zs = Z.shape

    # If it's empty, return it.
    if len(Zs) == 0 or (len(Zs) == 1 and Zs[0] == 0):
        return xp_copy(Z, xp=xp)

    if len(Zs) != 2:
        raise ValueError("The linkage array must be rectangular.")

    # If it contains no rows, return it.
    if Zs[0] == 0:
        return xp_copy(Z, xp=xp)

    if xp.min(Z[:, 0:2]) != 1.0 and xp.max(Z[:, 0:2]) != 2 * Zs[0]:
        raise ValueError('The format of the indices is not 1..N')

    Zpart = xp.concat((Z[:, 0:2] - 1.0, Z[:, 2:]), axis=1)
    CS = np.zeros((Zs[0],), dtype=np.float64)
    if is_jax(xp):
        # calculate_cluster_sizes doesn't accept read-only arrays
        Zpart = np.array(Zpart, copy=True)
    else:
        Zpart = np.asarray(Zpart)
    _hierarchy.calculate_cluster_sizes(Zpart, CS, int(Zs[0]) + 1)
    res = np.hstack([Zpart, CS.reshape(Zs[0], 1)])
    return xp.asarray(res)


def to_mlab_linkage(Z):
    """
    Convert a linkage matrix to a MATLAB(TM) compatible one.

    Converts a linkage matrix ``Z`` generated by the linkage function
    of this module to a MATLAB(TM) compatible one. The return linkage
    matrix has the last column removed and the cluster indices are
    converted to ``1..N`` indexing.

    Parameters
    ----------
    Z : ndarray
        A linkage matrix generated by ``scipy.cluster.hierarchy``.

    Returns
    -------
    to_mlab_linkage : ndarray
        A linkage matrix compatible with MATLAB(TM)'s hierarchical
        clustering functions.

        The return linkage matrix has the last column removed
        and the cluster indices are converted to ``1..N`` indexing.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    from_mlab_linkage : transform from Matlab to SciPy format.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, to_mlab_linkage
    >>> from scipy.spatial.distance import pdist

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    After a linkage matrix ``Z`` has been created, we can use
    `scipy.cluster.hierarchy.to_mlab_linkage` to convert it
    into MATLAB format:

    >>> mZ = to_mlab_linkage(Z)
    >>> mZ
    array([[  1.        ,   2.        ,   1.        ],
           [  4.        ,   5.        ,   1.        ],
           [  7.        ,   8.        ,   1.        ],
           [ 10.        ,  11.        ,   1.        ],
           [  3.        ,  13.        ,   1.29099445],
           [  6.        ,  14.        ,   1.29099445],
           [  9.        ,  15.        ,   1.29099445],
           [ 12.        ,  16.        ,   1.29099445],
           [ 17.        ,  18.        ,   5.77350269],
           [ 19.        ,  20.        ,   5.77350269],
           [ 21.        ,  22.        ,   8.16496581]])

    The new linkage matrix ``mZ`` uses 1-indexing for all the
    clusters (instead of 0-indexing). Also, the last column of
    the original linkage matrix has been dropped.

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    Zs = Z.shape
    if len(Zs) == 0 or (len(Zs) == 1 and Zs[0] == 0):
        return xp_copy(Z, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    return xp.concat((Z[:, :2] + 1.0, Z[:, 2:3]), axis=1)


def is_monotonic(Z):
    """
    Return True if the linkage passed is monotonic.

    The linkage is monotonic if for every cluster :math:`s` and :math:`t`
    joined, the distance between them is no less than the distance
    between any previously joined clusters.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix to check for monotonicity.

    Returns
    -------
    b : bool
        A boolean indicating whether the linkage is monotonic.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, ward, is_monotonic
    >>> from scipy.spatial.distance import pdist

    By definition, some hierarchical clustering algorithms - such as
    `scipy.cluster.hierarchy.ward` - produce monotonic assignments of
    samples to clusters; however, this is not always true for other
    hierarchical methods - e.g. `scipy.cluster.hierarchy.median`.

    Given a linkage matrix ``Z`` (as the result of a hierarchical clustering
    method) we can test programmatically whether it has the monotonicity
    property or not, using `scipy.cluster.hierarchy.is_monotonic`:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> is_monotonic(Z)
    True

    >>> Z = median(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
    >>> is_monotonic(Z)
    False

    Note that this method is equivalent to just verifying that the distances
    in the third column of the linkage matrix appear in a monotonically
    increasing order.

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='c', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    # We expect the i'th value to be greater than its successor.
    return xp.all(Z[1:, 2] >= Z[:-1, 2])


def is_valid_im(R, warning=False, throw=False, name=None):
    """Return True if the inconsistency matrix passed is valid.

    It must be a :math:`n` by 4 array of doubles. The standard
    deviations ``R[:,1]`` must be nonnegative. The link counts
    ``R[:,2]`` must be positive and no greater than :math:`n-1`.

    Parameters
    ----------
    R : ndarray
        The inconsistency matrix to check for validity.
    warning : bool, optional
        When True, issues a Python warning if the linkage
        matrix passed is invalid.
    throw : bool, optional
        When True, throws a Python exception if the linkage
        matrix passed is invalid.
    name : str, optional
        This string refers to the variable name of the invalid
        linkage matrix.

    Returns
    -------
    b : bool
        True if the inconsistency matrix is valid.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, inconsistent, is_valid_im
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> R = inconsistent(Z)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [6.57065706, 1.38071187, 3.        , 1.15470054]])

    Now we can use `scipy.cluster.hierarchy.is_valid_im` to verify that
    ``R`` is correct:

    >>> is_valid_im(R)
    True

    However, if ``R`` is wrongly constructed (e.g., one of the standard
    deviations is set to a negative value), then the check will fail:

    >>> R[-1,1] = R[-1,1] * -1
    >>> is_valid_im(R)
    False

    """
    xp = array_namespace(R)
    R = _asarray(R, order='c', xp=xp)
    valid = True
    name_str = f"{name!r} " if name else ''
    try:
        if R.dtype != xp.float64:
            raise TypeError(f'Inconsistency matrix {name_str}must contain doubles '
                            '(double).')
        if len(R.shape) != 2:
            raise ValueError(f'Inconsistency matrix {name_str}must have shape=2 (i.e. '
                             'be two-dimensional).')
        if R.shape[1] != 4:
            raise ValueError(f'Inconsistency matrix {name_str}'
                             'must have 4 columns.')
        if R.shape[0] < 1:
            raise ValueError(f'Inconsistency matrix {name_str}'
                             'must have at least one row.')
        if xp.any(R[:, 0] < 0):
            raise ValueError(f'Inconsistency matrix {name_str}'
                             'contains negative link height means.')
        if xp.any(R[:, 1] < 0):
            raise ValueError(f'Inconsistency matrix {name_str}'
                             'contains negative link height standard deviations.')
        if xp.any(R[:, 2] < 0):
            raise ValueError(f'Inconsistency matrix {name_str}'
                             'contains negative link counts.')
    except Exception as e:
        if throw:
            raise
        if warning:
            _warning(str(e))
        valid = False

    return valid


def is_valid_linkage(Z, warning=False, throw=False, name=None):
    """
    Check the validity of a linkage matrix.

    A linkage matrix is valid if it is a 2-D array (type double)
    with :math:`n` rows and 4 columns. The first two columns must contain
    indices between 0 and :math:`2n-1`. For a given row ``i``, the following
    two expressions have to hold:

    .. math::

        0 \\leq \\mathtt{Z[i,0]} \\leq i+n-1
        0 \\leq Z[i,1] \\leq i+n-1

    I.e., a cluster cannot join another cluster unless the cluster being joined
    has been generated.

    The fourth column of `Z` represents the number of original observations
    in a cluster, so a valid ``Z[i, 3]`` value may not exceed the number of
    original observations.

    Parameters
    ----------
    Z : array_like
        Linkage matrix.
    warning : bool, optional
        When True, issues a Python warning if the linkage
        matrix passed is invalid.
    throw : bool, optional
        When True, throws a Python exception if the linkage
        matrix passed is invalid.
    name : str, optional
        This string refers to the variable name of the invalid
        linkage matrix.

    Returns
    -------
    b : bool
        True if the inconsistency matrix is valid.

    See Also
    --------
    linkage: for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, is_valid_linkage
    >>> from scipy.spatial.distance import pdist

    All linkage matrices generated by the clustering methods in this module
    will be valid (i.e., they will have the appropriate dimensions and the two
    required expressions will hold for all the rows).

    We can check this using `scipy.cluster.hierarchy.is_valid_linkage`:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> is_valid_linkage(Z)
    True

    However, if we create a linkage matrix in a wrong way - or if we modify
    a valid one in a way that any of the required expressions don't hold
    anymore, then the check will fail:

    >>> Z[3][1] = 20    # the cluster number 20 is not defined at this point
    >>> is_valid_linkage(Z)
    False

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='c', xp=xp)
    valid = True
    name_str = f"{name!r} " if name else ''
    try:
        if Z.dtype != xp.float64:
            raise TypeError(f'Linkage matrix {name_str}must contain doubles.')
        if len(Z.shape) != 2:
            raise ValueError(f'Linkage matrix {name_str}must have shape=2 (i.e. be'
                             ' two-dimensional).')
        if Z.shape[1] != 4:
            raise ValueError(f'Linkage matrix {name_str}must have 4 columns.')
        if Z.shape[0] == 0:
            raise ValueError('Linkage must be computed on at least two '
                             'observations.')
        n = Z.shape[0]
        if n > 1:
            if (xp.any(Z[:, 0] < 0) or xp.any(Z[:, 1] < 0)):
                raise ValueError(f'Linkage {name_str}contains negative indices.')
            if xp.any(Z[:, 2] < 0):
                raise ValueError(f'Linkage {name_str}contains negative distances.')
            if xp.any(Z[:, 3] < 0):
                raise ValueError(f'Linkage {name_str}contains negative counts.')
            if xp.any(Z[:, 3] > (Z.shape[0] + 1)):
                raise ValueError('Linkage matrix contains excessive observations'
                                 'in a cluster')
        if _check_hierarchy_uses_cluster_before_formed(Z):
            raise ValueError(f'Linkage {name_str}uses non-singleton cluster before'
                             ' it is formed.')
        if _check_hierarchy_uses_cluster_more_than_once(Z):
            raise ValueError(f'Linkage {name_str}uses the same cluster more than once.')
    except Exception as e:
        if throw:
            raise
        if warning:
            _warning(str(e))
        valid = False

    return valid


def _check_hierarchy_uses_cluster_before_formed(Z):
    n = Z.shape[0] + 1
    for i in range(0, n - 1):
        if Z[i, 0] >= n + i or Z[i, 1] >= n + i:
            return True
    return False


def _check_hierarchy_uses_cluster_more_than_once(Z):
    n = Z.shape[0] + 1
    chosen = set()
    for i in range(0, n - 1):
        used_more_than_once = (
            (float(Z[i, 0]) in chosen)
            or (float(Z[i, 1]) in chosen)
            or Z[i, 0] == Z[i, 1]
        )
        if used_more_than_once:
            return True
        chosen.add(float(Z[i, 0]))
        chosen.add(float(Z[i, 1]))
    return False


def _check_hierarchy_not_all_clusters_used(Z):
    n = Z.shape[0] + 1
    chosen = set()
    for i in range(0, n - 1):
        chosen.add(int(Z[i, 0]))
        chosen.add(int(Z[i, 1]))
    must_chosen = set(range(0, 2 * n - 2))
    return len(must_chosen.difference(chosen)) > 0


def num_obs_linkage(Z):
    """
    Return the number of original observations of the linkage matrix passed.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix on which to perform the operation.

    Returns
    -------
    n : int
        The number of original observations in the linkage.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, num_obs_linkage
    >>> from scipy.spatial.distance import pdist

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    ``Z`` is a linkage matrix obtained after using the Ward clustering method
    with ``X``, a dataset with 12 data points.

    >>> num_obs_linkage(Z)
    12

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='c', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    return (Z.shape[0] + 1)


def correspond(Z, Y):
    """
    Check for correspondence between linkage and condensed distance matrices.

    They must have the same number of original observations for
    the check to succeed.

    This function is useful as a sanity check in algorithms that make
    extensive use of linkage and distance matrices that must
    correspond to the same set of original observations.

    Parameters
    ----------
    Z : array_like
        The linkage matrix to check for correspondence.
    Y : array_like
        The condensed distance matrix to check for correspondence.

    Returns
    -------
    b : bool
        A boolean indicating whether the linkage matrix and distance
        matrix could possibly correspond to one another.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, correspond
    >>> from scipy.spatial.distance import pdist

    This method can be used to check if a given linkage matrix ``Z`` has been
    obtained from the application of a cluster method over a dataset ``X``:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]
    >>> X_condensed = pdist(X)
    >>> Z = ward(X_condensed)

    Here, we can compare ``Z`` and ``X`` (in condensed form):

    >>> correspond(Z, X_condensed)
    True

    """
    is_valid_linkage(Z, throw=True)
    distance.is_valid_y(Y, throw=True)
    xp = array_namespace(Z, Y)
    Z = _asarray(Z, order='c', xp=xp)
    Y = _asarray(Y, order='c', xp=xp)
    return distance.num_obs_y(Y) == num_obs_linkage(Z)


def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    """
    Form flat clusters from the hierarchical clustering defined by
    the given linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded with the matrix returned
        by the `linkage` function.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested.
    criterion : str, optional
        The criterion to use in forming flat clusters. This can
        be any of the following values:

          ``inconsistent`` :
              If a cluster node and all its
              descendants have an inconsistent value less than or equal
              to `t`, then all its leaf descendants belong to the
              same flat cluster. When no non-singleton cluster meets
              this criterion, every node is assigned to its own
              cluster. (Default)

          ``distance`` :
              Forms flat clusters so that the original
              observations in each flat cluster have no greater a
              cophenetic distance than `t`.

          ``maxclust`` :
              Finds a minimum threshold ``r`` so that
              the cophenetic distance between any two original
              observations in the same flat cluster is no more than
              ``r`` and no more than `t` flat clusters are formed.

          ``monocrit`` :
              Forms a flat cluster from a cluster node c
              with index i when ``monocrit[j] <= t``.

              For example, to threshold on the maximum mean distance
              as computed in the inconsistency matrix R with a
              threshold of 0.8 do::

                  MR = maxRstat(Z, R, 3)
                  fcluster(Z, t=0.8, criterion='monocrit', monocrit=MR)

          ``maxclust_monocrit`` :
              Forms a flat cluster from a
              non-singleton cluster node ``c`` when ``monocrit[i] <=
              r`` for all cluster indices ``i`` below and including
              ``c``. ``r`` is minimized such that no more than ``t``
              flat clusters are formed. monocrit must be
              monotonic. For example, to minimize the threshold t on
              maximum inconsistency values so that no more than 3 flat
              clusters are formed, do::

                  MI = maxinconsts(Z, R)
                  fcluster(Z, t=3, criterion='maxclust_monocrit', monocrit=MI)
    depth : int, optional
        The maximum depth to perform the inconsistency calculation.
        It has no meaning for the other criteria. Default is 2.
    R : ndarray, optional
        The inconsistency matrix to use for the ``'inconsistent'``
        criterion. This matrix is computed if not provided.
    monocrit : ndarray, optional
        An array of length n-1. `monocrit[i]` is the
        statistics upon which non-singleton i is thresholded. The
        monocrit vector must be monotonic, i.e., given a node c with
        index i, for all node indices j corresponding to nodes
        below c, ``monocrit[i] >= monocrit[j]``.

    Returns
    -------
    fcluster : ndarray
        An array of length ``n``. ``T[i]`` is the flat cluster number to
        which original observation ``i`` belongs.

    See Also
    --------
    linkage : for information about hierarchical clustering methods work.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster
    >>> from scipy.spatial.distance import pdist

    All cluster linkage methods - e.g., `scipy.cluster.hierarchy.ward`
    generate a linkage matrix ``Z`` as their output:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    This matrix represents a dendrogram, where the first and second elements
    are the two clusters merged at each step, the third element is the
    distance between these clusters, and the fourth element is the size of
    the new cluster - the number of original data points included.

    `scipy.cluster.hierarchy.fcluster` can be used to flatten the
    dendrogram, obtaining as a result an assignation of the original data
    points to single clusters.

    This assignation mostly depends on a distance threshold ``t`` - the maximum
    inter-cluster distance allowed:

    >>> fcluster(Z, t=0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)

    >>> fcluster(Z, t=1.1, criterion='distance')
    array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)

    >>> fcluster(Z, t=3, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    >>> fcluster(Z, t=9, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    In the first case, the threshold ``t`` is too small to allow any two
    samples in the data to form a cluster, so 12 different clusters are
    returned.

    In the second case, the threshold is large enough to allow the first
    4 points to be merged with their nearest neighbors. So, here, only 8
    clusters are returned.

    The third case, with a much higher threshold, allows for up to 8 data
    points to be connected - so 4 clusters are returned here.

    Lastly, the threshold of the fourth case is large enough to allow for
    all data points to be merged together - so a single cluster is returned.

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')

    n = Z.shape[0] + 1
    T = np.zeros((n,), dtype='i')

    if monocrit is not None:
        monocrit = np.asarray(monocrit, order='C', dtype=np.float64)

    Z = np.asarray(Z)
    monocrit = np.asarray(monocrit)
    if criterion == 'inconsistent':
        if R is None:
            R = inconsistent(Z, depth)
        else:
            R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
            is_valid_im(R, throw=True, name='R')
            # Since the C code does not support striding using strides.
            # The dimensions are used instead.
            R = np.asarray(R)
        _hierarchy.cluster_in(Z, R, T, float(t), int(n))
    elif criterion == 'distance':
        _hierarchy.cluster_dist(Z, T, float(t), int(n))
    elif criterion == 'maxclust':
        _hierarchy.cluster_maxclust_dist(Z, T, int(n), t)
    elif criterion == 'monocrit':
        _hierarchy.cluster_monocrit(Z, monocrit, T, float(t), int(n))
    elif criterion == 'maxclust_monocrit':
        _hierarchy.cluster_maxclust_monocrit(Z, monocrit, T, int(n), int(t))
    else:
        raise ValueError(f'Invalid cluster formation criterion: {str(criterion)}')
    return xp.asarray(T)


def fclusterdata(X, t, criterion='inconsistent',
                 metric='euclidean', depth=2, method='single', R=None):
    """
    Cluster observation data using a given metric.

    Clusters the original observations in the n-by-m data
    matrix X (n observations in m dimensions), using the euclidean
    distance metric to calculate distances between original observations,
    performs hierarchical clustering using the single linkage algorithm,
    and forms flat clusters using the inconsistency method with `t` as the
    cut-off threshold.

    A 1-D array ``T`` of length ``n`` is returned. ``T[i]`` is
    the index of the flat cluster to which the original observation ``i``
    belongs.

    Parameters
    ----------
    X : (N, M) ndarray
        N by M data matrix with N observations in M dimensions.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested.
    criterion : str, optional
        Specifies the criterion for forming flat clusters. Valid
        values are 'inconsistent' (default), 'distance', or 'maxclust'
        cluster formation algorithms. See `fcluster` for descriptions.
    metric : str or function, optional
        The distance metric for calculating pairwise distances. See
        ``distance.pdist`` for descriptions and linkage to verify
        compatibility with the linkage method.
    depth : int, optional
        The maximum depth for the inconsistency calculation. See
        `inconsistent` for more information.
    method : str, optional
        The linkage method to use (single, complete, average,
        weighted, median centroid, ward). See `linkage` for more
        information. Default is "single".
    R : ndarray, optional
        The inconsistency matrix. It will be computed if necessary
        if it is not passed.

    Returns
    -------
    fclusterdata : ndarray
        A vector of length n. T[i] is the flat cluster number to
        which original observation i belongs.

    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    Notes
    -----
    This function is similar to the MATLAB function ``clusterdata``.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import fclusterdata

    This is a convenience method that abstracts all the steps to perform in a
    typical SciPy's hierarchical clustering workflow.

    * Transform the input data into a condensed matrix with
      `scipy.spatial.distance.pdist`.

    * Apply a clustering method.

    * Obtain flat clusters at a user defined distance threshold ``t`` using
      `scipy.cluster.hierarchy.fcluster`.

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> fclusterdata(X, t=1)
    array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)

    The output here (for the dataset ``X``, distance threshold ``t``, and the
    default settings) is four clusters with three data points each.

    """
    xp = array_namespace(X)
    X = _asarray(X, order='C', dtype=xp.float64, xp=xp)

    if X.ndim != 2:
        raise TypeError('The observation matrix X must be an n by m '
                        'array.')

    Y = distance.pdist(X, metric=metric)
    Y = xp.asarray(Y)
    Z = linkage(Y, method=method)
    if R is None:
        R = inconsistent(Z, d=depth)
    else:
        R = _asarray(R, order='c', xp=xp)
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    return T


def leaves_list(Z):
    """
    Return a list of leaf node ids.

    The return corresponds to the observation vector index as it appears
    in the tree from left to right. Z is a linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix.  `Z` is
        a linkage matrix.  See `linkage` for more information.

    Returns
    -------
    leaves_list : ndarray
        The list of leaf node ids.

    See Also
    --------
    dendrogram : for information about dendrogram structure.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
    >>> from scipy.spatial.distance import pdist
    >>> from matplotlib import pyplot as plt

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    The linkage matrix ``Z`` represents a dendrogram, that is, a tree that
    encodes the structure of the clustering performed.
    `scipy.cluster.hierarchy.leaves_list` shows the mapping between
    indices in the ``X`` dataset and leaves in the dendrogram:

    >>> leaves_list(Z)
    array([ 2,  0,  1,  5,  3,  4,  8,  6,  7, 11,  9, 10], dtype=int32)

    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)
    >>> plt.show()

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    ML = np.zeros((n,), dtype='i')
    Z = np.asarray(Z)
    _hierarchy.prelist(Z, ML, n)
    return xp.asarray(ML)


# Maps number of leaves to text size.
#
# p <= 20, size="12"
# 20 < p <= 30, size="10"
# 30 < p <= 50, size="8"
# 50 < p <= np.inf, size="6"

_dtextsizes = {20: 12, 30: 10, 50: 8, 85: 6, np.inf: 5}
_drotation = {20: 0, 40: 45, np.inf: 90}
_dtextsortedkeys = list(_dtextsizes.keys())
_dtextsortedkeys.sort()
_drotationsortedkeys = list(_drotation.keys())
_drotationsortedkeys.sort()


def _remove_dups(L):
    """
    Remove duplicates AND preserve the original order of the elements.

    The set class is not guaranteed to do this.
    """
    seen_before = set()
    L2 = []
    for i in L:
        if i not in seen_before:
            seen_before.add(i)
            L2.append(i)
    return L2


def _get_tick_text_size(p):
    for k in _dtextsortedkeys:
        if p <= k:
            return _dtextsizes[k]


def _get_tick_rotation(p):
    for k in _drotationsortedkeys:
        if p <= k:
            return _drotation[k]


def _plot_dendrogram(icoords, dcoords, ivl, p, n, mh, orientation,
                     no_labels, color_list, leaf_font_size=None,
                     leaf_rotation=None, contraction_marks=None,
                     ax=None, above_threshold_color='C0'):
    # Import matplotlib here so that it's not imported unless dendrograms
    # are plotted. Raise an informative error if importing fails.
    try:
        # if an axis is provided, don't use pylab at all
        if ax is None:
            import matplotlib.pylab
        import matplotlib.patches
        import matplotlib.collections
    except ImportError as e:
        raise ImportError("You must install the matplotlib library to plot "
                          "the dendrogram. Use no_plot=True to calculate the "
                          "dendrogram without plotting.") from e

    if ax is None:
        ax = matplotlib.pylab.gca()
        # if we're using pylab, we want to trigger a draw at the end
        trigger_redraw = True
    else:
        trigger_redraw = False

    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = mh + mh * 0.05

    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)
    if orientation in ('top', 'bottom'):
        if orientation == 'top':
            ax.set_ylim([0, dvw])
            ax.set_xlim([0, ivw])
        else:
            ax.set_ylim([dvw, 0])
            ax.set_xlim([0, ivw])

        xlines = icoords
        ylines = dcoords
        if no_labels:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(iv_ticks)

            if orientation == 'top':
                ax.xaxis.set_ticks_position('bottom')
            else:
                ax.xaxis.set_ticks_position('top')

            # Make the tick marks invisible because they cover up the links
            for line in ax.get_xticklines():
                line.set_visible(False)

            leaf_rot = (float(_get_tick_rotation(len(ivl)))
                        if (leaf_rotation is None) else leaf_rotation)
            leaf_font = (float(_get_tick_text_size(len(ivl)))
                         if (leaf_font_size is None) else leaf_font_size)
            ax.set_xticklabels(ivl, rotation=leaf_rot, size=leaf_font)

    elif orientation in ('left', 'right'):
        if orientation == 'left':
            ax.set_xlim([dvw, 0])
            ax.set_ylim([0, ivw])
        else:
            ax.set_xlim([0, dvw])
            ax.set_ylim([0, ivw])

        xlines = dcoords
        ylines = icoords
        if no_labels:
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_yticks(iv_ticks)

            if orientation == 'left':
                ax.yaxis.set_ticks_position('right')
            else:
                ax.yaxis.set_ticks_position('left')

            # Make the tick marks invisible because they cover up the links
            for line in ax.get_yticklines():
                line.set_visible(False)

            leaf_font = (float(_get_tick_text_size(len(ivl)))
                         if (leaf_font_size is None) else leaf_font_size)

            if leaf_rotation is not None:
                ax.set_yticklabels(ivl, rotation=leaf_rotation, size=leaf_font)
            else:
                ax.set_yticklabels(ivl, size=leaf_font)

    # Let's use collections instead. This way there is a separate legend item
    # for each tree grouping, rather than stupidly one for each line segment.
    colors_used = _remove_dups(color_list)
    color_to_lines = {}
    for color in colors_used:
        color_to_lines[color] = []
    for (xline, yline, color) in zip(xlines, ylines, color_list):
        color_to_lines[color].append(list(zip(xline, yline)))

    colors_to_collections = {}
    # Construct the collections.
    for color in colors_used:
        coll = matplotlib.collections.LineCollection(color_to_lines[color],
                                                     colors=(color,))
        colors_to_collections[color] = coll

    # Add all the groupings below the color threshold.
    for color in colors_used:
        if color != above_threshold_color:
            ax.add_collection(colors_to_collections[color])
    # If there's a grouping of links above the color threshold, it goes last.
    if above_threshold_color in colors_to_collections:
        ax.add_collection(colors_to_collections[above_threshold_color])

    if contraction_marks is not None:
        Ellipse = matplotlib.patches.Ellipse
        for (x, y) in contraction_marks:
            if orientation in ('left', 'right'):
                e = Ellipse((y, x), width=dvw / 100, height=1.0)
            else:
                e = Ellipse((x, y), width=1.0, height=dvw / 100)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor('k')

    if trigger_redraw:
        matplotlib.pylab.draw_if_interactive()


# C0  is used for above threshold color
_link_line_colors_default = ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9')
_link_line_colors = list(_link_line_colors_default)


def set_link_color_palette(palette):
    """
    Set list of matplotlib color codes for use by dendrogram.

    Note that this palette is global (i.e., setting it once changes the colors
    for all subsequent calls to `dendrogram`) and that it affects only the
    the colors below ``color_threshold``.

    Note that `dendrogram` also accepts a custom coloring function through its
    ``link_color_func`` keyword, which is more flexible and non-global.

    Parameters
    ----------
    palette : list of str or None
        A list of matplotlib color codes.  The order of the color codes is the
        order in which the colors are cycled through when color thresholding in
        the dendrogram.

        If ``None``, resets the palette to its default (which are matplotlib
        default colors C1 to C9).

    Returns
    -------
    None

    See Also
    --------
    dendrogram

    Notes
    -----
    Ability to reset the palette with ``None`` added in SciPy 0.17.0.

    Thread safety: using this function in a multi-threaded fashion may
    result in `dendrogram` producing plots with unexpected colors.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    ...                    400., 754., 564., 138., 219., 869., 669.])
    >>> Z = hierarchy.linkage(ytdist, 'single')
    >>> dn = hierarchy.dendrogram(Z, no_plot=True)
    >>> dn['color_list']
    ['C1', 'C0', 'C0', 'C0', 'C0']
    >>> hierarchy.set_link_color_palette(['c', 'm', 'y', 'k'])
    >>> dn = hierarchy.dendrogram(Z, no_plot=True, above_threshold_color='b')
    >>> dn['color_list']
    ['c', 'b', 'b', 'b', 'b']
    >>> dn = hierarchy.dendrogram(Z, no_plot=True, color_threshold=267,
    ...                           above_threshold_color='k')
    >>> dn['color_list']
    ['c', 'm', 'm', 'k', 'k']

    Now, reset the color palette to its default:

    >>> hierarchy.set_link_color_palette(None)

    """
    if palette is None:
        # reset to its default
        palette = _link_line_colors_default
    elif not isinstance(palette, (list, tuple)):
        raise TypeError("palette must be a list or tuple")
    _ptypes = [isinstance(p, str) for p in palette]

    if False in _ptypes:
        raise TypeError("all palette list elements must be color strings")

    global _link_line_colors
    _link_line_colors = palette


def dendrogram(Z, p=30, truncate_mode=None, color_threshold=None,
               get_leaves=True, orientation='top', labels=None,
               count_sort=False, distance_sort=False, show_leaf_counts=True,
               no_plot=False, no_labels=False, leaf_font_size=None,
               leaf_rotation=None, leaf_label_func=None,
               show_contracted=False, link_color_func=None, ax=None,
               above_threshold_color='C0'):
    """
    Plot the hierarchical clustering as a dendrogram.

    The dendrogram illustrates how each cluster is
    composed by drawing a U-shaped link between a non-singleton
    cluster and its children. The top of the U-link indicates a
    cluster merge. The two legs of the U-link indicate which clusters
    were merged. The length of the two legs of the U-link represents
    the distance between the child clusters. It is also the
    cophenetic distance between original observations in the two
    children clusters.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix encoding the hierarchical clustering to
        render as a dendrogram. See the ``linkage`` function for more
        information on the format of ``Z``.
    p : int, optional
        The ``p`` parameter for ``truncate_mode``.
    truncate_mode : str, optional
        The dendrogram can be hard to read when the original
        observation matrix from which the linkage is derived is
        large. Truncation is used to condense the dendrogram. There
        are several modes:

        ``None``
          No truncation is performed (default).
          Note: ``'none'`` is an alias for ``None`` that's kept for
          backward compatibility.

        ``'lastp'``
          The last ``p`` non-singleton clusters formed in the linkage are the
          only non-leaf nodes in the linkage; they correspond to rows
          ``Z[n-p-2:end]`` in ``Z``. All other non-singleton clusters are
          contracted into leaf nodes.

        ``'level'``
          No more than ``p`` levels of the dendrogram tree are displayed.
          A "level" includes all nodes with ``p`` merges from the final merge.

          Note: ``'mtica'`` is an alias for ``'level'`` that's kept for
          backward compatibility.

    color_threshold : double, optional
        For brevity, let :math:`t` be the ``color_threshold``.
        Colors all the descendent links below a cluster node
        :math:`k` the same color if :math:`k` is the first node below
        the cut threshold :math:`t`. All links connecting nodes with
        distances greater than or equal to the threshold are colored
        with de default matplotlib color ``'C0'``. If :math:`t` is less
        than or equal to zero, all nodes are colored ``'C0'``.
        If ``color_threshold`` is None or 'default',
        corresponding with MATLAB(TM) behavior, the threshold is set to
        ``0.7*max(Z[:,2])``.

    get_leaves : bool, optional
        Includes a list ``R['leaves']=H`` in the result
        dictionary. For each :math:`i`, ``H[i] == j``, cluster node
        ``j`` appears in position ``i`` in the left-to-right traversal
        of the leaves, where :math:`j < 2n-1` and :math:`i < n`.
    orientation : str, optional
        The direction to plot the dendrogram, which can be any
        of the following strings:

        ``'top'``
          Plots the root at the top, and plot descendent links going downwards.
          (default).

        ``'bottom'``
          Plots the root at the bottom, and plot descendent links going
          upwards.

        ``'left'``
          Plots the root at the left, and plot descendent links going right.

        ``'right'``
          Plots the root at the right, and plot descendent links going left.

    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
    count_sort : str or bool, optional
        For each node n, the order (visually, from left-to-right) n's
        two descendent links are plotted is determined by this
        parameter, which can be any of the following values:

        ``False``
          Nothing is done.

        ``'ascending'`` or ``True``
          The child with the minimum number of original objects in its cluster
          is plotted first.

        ``'descending'``
          The child with the maximum number of original objects in its cluster
          is plotted first.

        Note, ``distance_sort`` and ``count_sort`` cannot both be True.
    distance_sort : str or bool, optional
        For each node n, the order (visually, from left-to-right) n's
        two descendent links are plotted is determined by this
        parameter, which can be any of the following values:

        ``False``
          Nothing is done.

        ``'ascending'`` or ``True``
          The child with the minimum distance between its direct descendents is
          plotted first.

        ``'descending'``
          The child with the maximum distance between its direct descendents is
          plotted first.

        Note ``distance_sort`` and ``count_sort`` cannot both be True.
    show_leaf_counts : bool, optional
         When True, leaf nodes representing :math:`k>1` original
         observation are labeled with the number of observations they
         contain in parentheses.
    no_plot : bool, optional
        When True, the final rendering is not performed. This is
        useful if only the data structures computed for the rendering
        are needed or if matplotlib is not available.
    no_labels : bool, optional
        When True, no labels appear next to the leaf nodes in the
        rendering of the dendrogram.
    leaf_rotation : double, optional
        Specifies the angle (in degrees) to rotate the leaf
        labels. When unspecified, the rotation is based on the number of
        nodes in the dendrogram (default is 0).
    leaf_font_size : int, optional
        Specifies the font size (in points) of the leaf labels. When
        unspecified, the size based on the number of nodes in the
        dendrogram.
    leaf_label_func : lambda or function, optional
        When ``leaf_label_func`` is a callable function, for each
        leaf with cluster index :math:`k < 2n-1`. The function
        is expected to return a string with the label for the
        leaf.

        Indices :math:`k < n` correspond to original observations
        while indices :math:`k \\geq n` correspond to non-singleton
        clusters.

        For example, to label singletons with their node id and
        non-singletons with their id, count, and inconsistency
        coefficient, simply do::

            # First define the leaf label function.
            def llf(id):
                if id < n:
                    return str(id)
                else:
                    return '[%d %d %1.2f]' % (id, count, R[n-id,3])

            # The text for the leaf nodes is going to be big so force
            # a rotation of 90 degrees.
            dendrogram(Z, leaf_label_func=llf, leaf_rotation=90)

            # leaf_label_func can also be used together with ``truncate_mode``,
            # in which case you will get your leaves labeled after truncation:
            dendrogram(Z, leaf_label_func=llf, leaf_rotation=90,
                       truncate_mode='level', p=2)

    show_contracted : bool, optional
        When True the heights of non-singleton nodes contracted
        into a leaf node are plotted as crosses along the link
        connecting that leaf node.  This really is only useful when
        truncation is used (see ``truncate_mode`` parameter).
    link_color_func : callable, optional
        If given, `link_color_function` is called with each non-singleton id
        corresponding to each U-shaped link it will paint. The function is
        expected to return the color to paint the link, encoded as a matplotlib
        color string code. For example::

            dendrogram(Z, link_color_func=lambda k: colors[k])

