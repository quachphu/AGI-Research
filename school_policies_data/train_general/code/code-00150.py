        then a good s-value should be found in the range
        ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).
    eps : float, optional
        A threshold for determining the effective rank of an
        over-determined linear system of equations (0 < eps < 1).
        `eps` is not likely to need changing.
    tx, ty : ndarray, optional
        Rank-1 arrays of the knots of the spline for task=-1
    full_output : int, optional
        Non-zero to return optional outputs.
    nxest, nyest : int, optional
        Over-estimates of the total number of knots. If None then
        ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
        ``nyest = max(ky+sqrt(m/2),2*ky+3)``.
    quiet : int, optional
        Non-zero to suppress printing of messages.

    Returns
    -------
    tck : array_like
        A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and
        coefficients (c) of the bivariate B-spline representation of the
        surface along with the degree of the spline.
    fp : ndarray
        The weighted sum of squared residuals of the spline approximation.
    ier : int
        An integer flag about splrep success. Success is indicated if
        ier<=0. If ier in [1,2,3] an error occurred but was not raised.
        Otherwise an error is raised.
    msg : str
        A message corresponding to the integer flag, ier.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
    See `bisplev` to evaluate the value of the B-spline given its tck
    representation.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolation.

    References
    ----------
    .. [1] Dierckx P.:An algorithm for surface fitting with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P.:An algorithm for surface fitting with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

    """
    x, y, z = map(ravel, [x, y, z])  # ensure 1-d arrays.
    m = len(x)
    if not (m == len(y) == len(z)):
        raise TypeError('len(x)==len(y)==len(z) must hold.')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
    if xb is None:
        xb = x.min()
    if xe is None:
        xe = x.max()
    if yb is None:
        yb = y.min()
    if ye is None:
        ye = y.max()
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if s is None:
        s = m - sqrt(2*m)
    if tx is None and task == -1:
        raise TypeError('Knots_x must be given for task=-1')
    if tx is not None:
        _surfit_cache['tx'] = atleast_1d(tx)
    nx = len(_surfit_cache['tx'])
    if ty is None and task == -1:
        raise TypeError('Knots_y must be given for task=-1')
    if ty is not None:
        _surfit_cache['ty'] = atleast_1d(ty)
    ny = len(_surfit_cache['ty'])
    if task == -1 and nx < 2*kx+2:
        raise TypeError('There must be at least 2*kx+2 knots_x for task=-1')
    if task == -1 and ny < 2*ky+2:
        raise TypeError('There must be at least 2*ky+2 knots_x for task=-1')
    if not ((1 <= kx <= 5) and (1 <= ky <= 5)):
        raise TypeError('Given degree of the spline (kx,ky=%d,%d) is not '
                        'supported. (1<=k<=5)' % (kx, ky))
    if m < (kx + 1)*(ky + 1):
        raise TypeError('m >= (kx+1)(ky+1) must hold')
    if nxest is None:
        nxest = int(kx + sqrt(m/2))
    if nyest is None:
        nyest = int(ky + sqrt(m/2))
    nxest, nyest = max(nxest, 2*kx + 3), max(nyest, 2*ky + 3)
    if task >= 0 and s == 0:
        nxest = int(kx + sqrt(3*m))
        nyest = int(ky + sqrt(3*m))
    if task == -1:
        _surfit_cache['tx'] = atleast_1d(tx)
        _surfit_cache['ty'] = atleast_1d(ty)
    tx, ty = _surfit_cache['tx'], _surfit_cache['ty']
    wrk = _surfit_cache['wrk']
    u = nxest - kx - 1
    v = nyest - ky - 1
    km = max(kx, ky) + 1
    ne = max(nxest, nyest)
    bx, by = kx*v + ky + 1, ky*u + kx + 1
    b1, b2 = bx, bx + v - ky
    if bx > by:
        b1, b2 = by, by + u - kx
    msg = "Too many data points to interpolate"
    lwrk1 = _int_overflow(u*v*(2 + b1 + b2) +
                          2*(u + v + km*(m + ne) + ne - kx - ky) + b2 + 1,
                          OverflowError,
                          msg=msg)
    lwrk2 = _int_overflow(u*v*(b2 + 1) + b2, OverflowError, msg=msg)
    tx, ty, c, o = _fitpack._surfit(x, y, z, w, xb, xe, yb, ye, kx, ky,
                                    task, s, eps, tx, ty, nxest, nyest,
                                    wrk, lwrk1, lwrk2)
    _curfit_cache['tx'] = tx
    _curfit_cache['ty'] = ty
    _curfit_cache['wrk'] = o['wrk']
    ier, fp = o['ier'], o['fp']
    tck = [tx, ty, c, kx, ky]

    ierm = min(11, max(-3, ier))
    if ierm <= 0 and not quiet:
        _mess = (_iermess2[ierm][0] +
                 "\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                 (kx, ky, len(tx), len(ty), m, fp, s))
        warnings.warn(RuntimeWarning(_mess), stacklevel=2)
    if ierm > 0 and not full_output:
        if ier in [1, 2, 3, 4, 5]:
            _mess = ("\n\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                     (kx, ky, len(tx), len(ty), m, fp, s))
            warnings.warn(RuntimeWarning(_iermess2[ierm][0] + _mess), stacklevel=2)
        else:
            try:
                raise _iermess2[ierm][1](_iermess2[ierm][0])
            except KeyError as e:
                raise _iermess2['unknown'][1](_iermess2['unknown'][0]) from e
    if full_output:
        try:
            return tck, fp, ier, _iermess2[ierm][0]
        except KeyError:
            return tck, fp, ier, _iermess2['unknown'][0]
    else:
        return tck


def bisplev(x, y, tck, dx=0, dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.

    Return a rank-2 array of spline function values (or spline derivative
    values) at points given by the cross-product of the rank-1 arrays `x` and
    `y`.  In special cases, return an array or just a float if either `x` or
    `y` or both are floats.  Based on BISPEV and PARDER from FITPACK.

    Parameters
    ----------
    x, y : ndarray
        Rank-1 arrays specifying the domain over which to evaluate the
        spline or its derivative.
    tck : tuple
        A sequence of length 5 returned by `bisplrep` containing the knot
        locations, the coefficients, and the degree of the spline:
        [tx, ty, c, kx, ky].
    dx, dy : int, optional
        The orders of the partial derivatives in `x` and `y` respectively.

    Returns
    -------
    vals : ndarray
        The B-spline or its derivative evaluated over the set formed by
        the cross-product of `x` and `y`.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
        See `bisplrep` to generate the `tck` representation.

    References
    ----------
    .. [1] Dierckx P. : An algorithm for surface fitting
       with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P. : An algorithm for surface fitting
       with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P. : Curve and surface fitting with splines,
       Monographs on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

    """
    tx, ty, c, kx, ky = tck
    if not (0 <= dx < kx):
        raise ValueError("0 <= dx = %d < kx = %d must hold" % (dx, kx))
    if not (0 <= dy < ky):
        raise ValueError("0 <= dy = %d < ky = %d must hold" % (dy, ky))
    x, y = map(atleast_1d, [x, y])
    if (len(x.shape) != 1) or (len(y.shape) != 1):
        raise ValueError("First two entries should be rank-1 arrays.")

    msg = "Too many data points to interpolate."

    _int_overflow(x.size * y.size, MemoryError, msg=msg)

    if dx != 0 or dy != 0:
        _int_overflow((tx.size - kx - 1)*(ty.size - ky - 1),
                      MemoryError, msg=msg)
        z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
    else:
        z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)

    if ier == 10:
        raise ValueError("Invalid input data")
    if ier:
        raise TypeError("An error occurred")
    z.shape = len(x), len(y)
    if len(z) > 1:
        return z
    if len(z[0]) > 1:
        return z[0]
    return z[0][0]


def dblint(xa, xb, ya, yb, tck):
    """Evaluate the integral of a spline over area [xa,xb] x [ya,yb].

    Parameters
    ----------
    xa, xb : float
        The end-points of the x integration interval.
    ya, yb : float
        The end-points of the y integration interval.
    tck : list [tx, ty, c, kx, ky]
        A sequence of length 5 returned by bisplrep containing the knot
        locations tx, ty, the coefficients c, and the degrees kx, ky
        of the spline.

    Returns
    -------
    integ : float
        The value of the resulting integral.
    """
    tx, ty, c, kx, ky = tck
    return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)


def insert(x, tck, m=1, per=0):
    # see the docstring of `_fitpack_py/insert`
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        cc = []
        for c_vals in c:
            tt, cc_val, kk = insert(x, [t, c_vals, k], m)
            cc.append(cc_val)
        return (tt, cc, kk)
    else:
        tt, cc, ier = _fitpack._insert(per, t, c, k, x, m)
        if ier == 10:
            raise ValueError("Invalid input data")
        if ier:
            raise TypeError("An error occurred")
        return (tt, cc, k)


def splder(tck, n=1):
    # see the docstring of `_fitpack_py/splder`
    if n < 0:
        return splantider(tck, -n)

    t, c, k = tck

    if n > k:
        raise ValueError(f"Order of derivative (n = {n!r}) must be <= "
                         f"order of spline (k = {tck[2]!r})")

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + ((None,)*len(c.shape[1:]))

    with np.errstate(invalid='raise', divide='raise'):
        try:
            for j in range(n):
                # See e.g. Schumaker, Spline Functions: Basic Theory, Chapter 5

                # Compute the denominator in the differentiation formula.
                # (and append trailing dims, if necessary)
                dt = t[k+1:-1] - t[1:-k-1]
                dt = dt[sh]
                # Compute the new coefficients
                c = (c[1:-1-k] - c[:-2-k]) * k / dt
                # Pad coefficient array to same size as knots (FITPACK
                # convention)
                c = np.r_[c, np.zeros((k,) + c.shape[1:])]
                # Adjust knots
                t = t[1:-1]
                k -= 1
        except FloatingPointError as e:
            raise ValueError(("The spline has internal repeated knots "
                              "and is not differentiable %d times") % n) from e

    return t, c, k


def splantider(tck, n=1):
    # see the docstring of `_fitpack_py/splantider`
    if n < 0:
        return splder(tck, -n)

    t, c, k = tck

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + (None,)*len(c.shape[1:])

    for j in range(n):
        # This is the inverse set of operations to splder.

        # Compute the multiplier in the antiderivative formula.
        dt = t[k+1:] - t[:-k-1]
        dt = dt[sh]
        # Compute the new coefficients
        c = np.cumsum(c[:-k-1] * dt, axis=0) / (k + 1)
        c = np.r_[np.zeros((1,) + c.shape[1:]),
                  c,
                  [c[-1]] * (k+2)]
        # New knots
        t = np.r_[t[0], t, t[-1]]
        k += 1

    return t, c, k
<|endoftext|>__all__ = ['interp1d', 'interp2d', 'lagrange', 'PPoly', 'BPoly', 'NdPPoly']

from math import prod

import numpy as np
from numpy import array, asarray, intp, poly1d, searchsorted

import scipy.special as spec
from scipy._lib._util import copy_if_needed
from scipy.special import comb

from . import _fitpack_py
from ._polyint import _Interpolator1D
from . import _ppoly
from ._interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline


def lagrange(x, w):
    r"""
    Return a Lagrange interpolating polynomial.

    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.

    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.

    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e., f(`x`).

    Returns
    -------
    lagrange : `numpy.poly1d` instance
        The Lagrange interpolating polynomial.

    Examples
    --------
    Interpolate :math:`f(x) = x^3` by 3 points.

    >>> import numpy as np
    >>> from scipy.interpolate import lagrange
    >>> x = np.array([0, 1, 2])
    >>> y = x**3
    >>> poly = lagrange(x, y)

    Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
    it is given by

    .. math::

        \begin{aligned}
            L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
                 &= x (-2 + 3x)
        \end{aligned}

    >>> from numpy.polynomial.polynomial import Polynomial
    >>> Polynomial(poly.coef[::-1]).coef
    array([ 0., -2.,  3.])

    >>> import matplotlib.pyplot as plt
    >>> x_new = np.arange(0, 2.1, 0.1)
    >>> plt.scatter(x, y, label='data')
    >>> plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Polynomial')
    >>> plt.plot(x_new, 3*x_new**2 - 2*x_new + 0*x_new,
    ...          label=r"$3 x^2 - 2 x$", linestyle='-.')
    >>> plt.legend()
    >>> plt.show()

    """

    M = len(x)
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(w[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0, -x[k]])/fac
        p += pt
    return p


# !! Need to find argument for keeping initialize. If it isn't
# !! found, get rid of it!


err_mesg = """\
`interp2d` has been removed in SciPy 1.14.0.

For legacy code, nearly bug-for-bug compatible replacements are
`RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
scattered 2D data.

In new code, for regular grids use `RegularGridInterpolator` instead.
For scattered data, prefer `LinearNDInterpolator` or
`CloughTocher2DInterpolator`.

For more details see
https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
"""

class interp2d:
    """
    interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
             fill_value=None)

    .. versionremoved:: 1.14.0

        `interp2d` has been removed in SciPy 1.14.0.

        For legacy code, nearly bug-for-bug compatible replacements are
        `RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
        scattered 2D data.

        In new code, for regular grids use `RegularGridInterpolator` instead.
        For scattered data, prefer `LinearNDInterpolator` or
        `CloughTocher2DInterpolator`.

        For more details see :ref:`interp-transition-guide`.
    """
    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=None):
        raise NotImplementedError(err_mesg)


def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError(f'{name} argument must be able to broadcast up '
                     f'to shape {shape_to} but had shape {shape_from}')


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    return (isinstance(fill_value, str) and
            fill_value == 'extrapolate')


class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    .. legacy:: class

        For a guide to the intended replacements for `interp1d` see
        :ref:`tutorial-interpolate_1Dsection`.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (npoints, ) array_like
        A 1-D array of real values.
    y : (..., npoints, ...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`. Use the ``axis`` parameter
        to select correct axis. Unlike other interpolators, the default
        interpolation axis is the last axis of `y`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Unlike
        other interpolators, defaults to ``axis=-1``.
    copy : bool, optional
        If ``True``, the class makes internal copies of x and y. If ``False``,
        references to ``x`` and ``y`` are used if possible. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``. Using a two-element tuple
          or ndarray requires ``bounds_error=False``.

          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.

          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Attributes
    ----------
    fill_value

    Methods
    -------
    __call__

    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation

    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.

    If the values in `x` are not unique, the resulting behavior is
    undefined and specific to the choice of `kind`, i.e., changing
    `kind` will change the behavior for duplicates.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)

    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=np.nan,
                 assume_sorted=False):
        """ Initialize a 1-D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)

        self.bounds_error = bounds_error  # used by fill_value setter

        # `copy` keyword semantics changed in NumPy 2.0, once that is
        # the minimum version this can use `copy=None`.
        self.copy = copy
        if not copy:
            self.copy = copy_if_needed

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1,
                     'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest', 'nearest-up', 'previous',
                          'next'):
            raise NotImplementedError(f"{kind} is unsupported: Use fitpack "
                                      "routines for other types.")
        x = array(x, copy=self.copy)
        y = array(y, copy=self.copy)

        if not assume_sorted:
            ind = np.argsort(x, kind="mergesort")
            x = x[ind]
            y = np.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float64)

        # Backward compatibility
        self.axis = axis % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x  # clean up namespace to prevent misuse; use attributes
        self._kind = kind

        # Adjust to interpolation kind; store reference to *unbound*
        # interpolation methods, in order to avoid circular references to self
        # stored in the bound instance methods, and therefore delayed garbage
        # collection.  See: https://docs.python.org/reference/datamodel.html
        if kind in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            minval = 1
            if kind == 'nearest':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'left'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'nearest-up':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'right'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'previous':
                # Side for np.searchsorted and index for clipping
                self._side = 'left'
                self._ind = 0
                # Move x by one floating point value to the left
                self._x_shift = np.nextafter(self.x, -np.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    # assume y is sorted by x ascending order here.
                    fill_value = (np.nan, np.take(self.y, -1, axis))
            elif kind == 'next':
                self._side = 'right'
                self._ind = 1
                # Move x by one floating point value to the right
                self._x_shift = np.nextafter(self.x, np.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    # assume y is sorted by x ascending order here.
                    fill_value = (np.take(self.y, 0, axis), np.nan)
            else:
                # Check if we can delegate to numpy.interp (2x-10x faster).
                np_dtypes = (np.dtype(np.float64), np.dtype(int))
                cond = self.x.dtype in np_dtypes and self.y.dtype in np_dtypes
                cond = cond and self.y.ndim == 1
                cond = cond and not _do_extrapolate(fill_value)

                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1

            rewrite_nan = False
            xx, yy = self.x, self._y
            if order > 1:
                # Quadratic or cubic spline. If input contains even a single
                # nan, then the output is all nans. We cannot just feed data
                # with nans to make_interp_spline because it calls LAPACK.
                # So, we make up a bogus x and y with no nans and use it
                # to get the correct shape of the output, which we then fill
                # with nans.
                # For slinear or zero order spline, we just pass nans through.
                mask = np.isnan(self.x)
                if mask.any():
                    sx = self.x[~mask]
                    if sx.size == 0:
                        raise ValueError("`x` array is all-nan")
                    xx = np.linspace(np.nanmin(self.x),
                                     np.nanmax(self.x),
                                     len(self.x))
                    rewrite_nan = True
                if np.isnan(self._y).any():
                    yy = np.ones_like(self._y)
                    rewrite_nan = True

            self._spline = make_interp_spline(xx, yy, k=order,
                                              check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline

        if len(self.x) < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)

        self.fill_value = fill_value  # calls the setter, can modify bounds_err

    @property
    def fill_value(self):
        """The fill value."""
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        # extrapolation only works for nearest neighbor and linear methods
        if _do_extrapolate(fill_value):
            self._check_and_update_bounds_error_for_extrapolation()
            self._extrapolate = True
        else:
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # it's either a pair (_below_range, _above_range) or a single value
            # for both above and below range
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]),
                               np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def _check_and_update_bounds_error_for_extrapolation(self):
        if self.bounds_error:
            raise ValueError("Cannot extrapolate and raise "
                             "at the same time.")
        self.bounds_error = False

    def _call_linear_np(self, x_new):
        # Note that out-of-bounds values are taken care of in self._evaluate
        return np.interp(x_new, self.x, self.y)

    def _call_linear(self, x_new):
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbor interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbor
        x_new_indices = searchsorted(self.x_bds, x_new, side=self._side)

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices]

        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbor of x_new, y_new = f(x_new)."""

        # 1. Get index of left/right value
        x_new_indices = searchsorted(self._x_shift, x_new, side=self._side)

        # 2. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(1-self._ind,
                                           len(self.x)-self._ind).astype(intp)

        # 3. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices+self._ind-1]

        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x. Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x. Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        if self.bounds_error and below_bounds.any():
            below_bounds_value = x_new[np.argmax(below_bounds)]
            raise ValueError(f"A value ({below_bounds_value}) in x_new is below "
                             f"the interpolation range's minimum value ({self.x[0]}).")
        if self.bounds_error and above_bounds.any():
            above_bounds_value = x_new[np.argmax(above_bounds)]
            raise ValueError(f"A value ({above_bounds_value}) in x_new is above "
                             f"the interpolation range's maximum value ({self.x[-1]}).")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds


class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = np.asarray(c)
        self.x = np.ascontiguousarray(x, dtype=np.float64)

        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError(f"axis={axis} must be between 0 and {self.c.ndim-1}")

        self.axis = axis
        if axis != 0:
            # move the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = np.moveaxis(self.c, axis+1, 0)
            self.c = np.moveaxis(self.c, axis+1, 0)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = np.diff(self.x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex128
        else:
            return np.float64

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.

        Notes
        -----
        This method is not thread safe and must not be executed concurrently
        with other methods available in this class. Doing so may cause
        unexpected errors or numerical output mismatches.
        """

        c = np.asarray(c)
        x = np.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError(f"Shapes of x {x.shape} and c {c.shape} are incompatible")
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError(
                f"Shapes of c {c.shape} and self.c {self.c.shape} are incompatible"
            )

        if c.size == 0:
            return

        dx = np.diff(x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                "from `self.x`.")
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                "from `self.x`.")

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                    dtype=dtype)

        if action == 'append':
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]
        elif action == 'prepend':
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        out = np.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out


class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    solve
    roots
    extend
    from_spline
    from_bernstein_basis
    construct_fast

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.
    """

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly.evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,)*(c2.ndim-1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        c = np.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
                     dtype=self.c.dtype)
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # fix continuity of added degrees of freedom
        self._ensure_c_contiguous()
        _ppoly.fix_continuity(c.reshape(c.shape[0], c.shape[1], -1),
                              self.x, nu - 1)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        range_int = np.empty((prod(self.c.shape[2:]),), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        # Compute the integral.
        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            remainder_int = np.empty_like(range_int)
            if b <= xe:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int

                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            _ppoly.integrate(
                self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                self.x, a, b, bool(extrapolate), out=range_int)

        # Return
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0., discontinuity=True, extrapolate=None):
        """
        Find real solutions of the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.

        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.

        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        Examples
        --------

        Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
        ``[-2, 1], [1, 2]``:

        >>> import numpy as np
        >>> from scipy.interpolate import PPoly
        >>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
        >>> pp.solve()
        array([-1.,  1.])
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        self._ensure_c_contiguous()

        if np.issubdtype(self.c.dtype, np.complexfloating):
            raise ValueError("Root finding is only for "
                             "real-valued polynomials")

        y = float(y)
        r = _ppoly.real_roots(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                              self.x, y, bool(discontinuity),
                              bool(extrapolate))
        if self.c.ndim == 2:
            return r[0]
        else:
            r2 = np.empty(prod(self.c.shape[2:]), dtype=object)
            # this for-loop is equivalent to ``r2[...] = r``, but that's broken
            # in NumPy 1.6.0
            for ii, root in enumerate(r):
                r2[ii] = root

            return r2.reshape(self.c.shape[2:])

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        See Also
        --------
        PPoly.solve
        """
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as returned by `splrep` or a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Examples
        --------
        Construct an interpolating spline and convert it to a `PPoly` instance

        >>> import numpy as np
        >>> from scipy.interpolate import splrep, PPoly
        >>> x = np.linspace(0, 1, 11)
        >>> y = np.sin(2*np.pi*x)
        >>> tck = splrep(x, y, s=0)
        >>> p = PPoly.from_spline(tck)
        >>> isinstance(p, PPoly)
        True

        Note that this function only supports 1D splines out of the box.

        If the ``tck`` object represents a parametric spline (e.g. constructed
        by `splprep` or a `BSpline` with ``c.ndim > 1``), you will need to loop
        over the dimensions manually.

        >>> from scipy.interpolate import splprep, splev
        >>> t = np.linspace(0, 1, 11)
        >>> x = np.sin(2*np.pi*t)
        >>> y = np.cos(2*np.pi*t)
        >>> (t, c, k), u = splprep([x, y], s=0)

        Note that ``c`` is a list of two arrays of length 11.

        >>> unew = np.arange(0, 1.01, 0.01)
        >>> out = splev(unew, (t, c, k))

        To convert this spline to the power basis, we convert each
        component of the list of b-spline coefficients, ``c``, into the
        corresponding cubic polynomial.

        >>> polys = [PPoly.from_spline((t, cj, k)) for cj in c]
        >>> polys[0].c.shape
        (4, 14)

        Note that the coefficients of the polynomials `polys` are in the
        power basis and their dimensions reflect just that: here 4 is the order
        (degree+1), and 14 is the number of intervals---which is nothing but
        the length of the knot array of the original `tck` minus one.

        Optionally, we can stack the components into a single `PPoly` along
        the third dimension:

        >>> cc = np.dstack([p.c for p in polys])    # has shape = (4, 14, 2)
        >>> poly = PPoly(cc, polys[0].x)
        >>> np.allclose(poly(unew).T,     # note the transpose to match `splev`
        ...             out, atol=1e-15)
        True

        """
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck

        cvals = np.empty((k + 1, len(t)-1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = _fitpack_py.splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y/spec.gamma(m+1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(bp, BPoly):
            raise TypeError(f".from_bernstein_basis only accepts BPoly instances. "
                            f"Got {type(bp)} instead.")

        dx = np.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,)*(bp.c.ndim-2)

        c = np.zeros_like(bp.c)
        for a in range(k+1):
            factor = (-1)**a * comb(k, a) * bp.c[a]
            for s in range(a, k+1):
                val = comb(k-a, s-a) * (-1)**s
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)


class BPoly(_PPolyBase):
    """Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    extend
    derivative
    antiderivative
    integrate
    construct_fast
    from_power_basis
    from_derivatives

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial

    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf

    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546, :doi:`10.1155/2011/829543`.

    Examples
    --------
    >>> from scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) + 3
               \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2

    """  # noqa: E501

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly.evaluate_bernstein(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k - nu representing the derivative of
            this polynomial.

        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            # For a polynomial
            #    B(x) = \sum_{a=0}^{k} c_a b_{a, k}(x),
            # we use the fact that
            #   b'_{a, k} = k ( b_{a-1, k-1} - b_{a, k-1} ),
            # which leads to
            #   B'(x) = \sum_{a=0}^{k-1} (c_{a+1} - c_a) b_{a, k-1}
            #
            # finally, for an interval [y, y + dy] with dy != 1,
            # we need to correct for an extra power of dy

            rest = (None,)*(self.c.ndim-2)

            k = self.c.shape[0] - 1
            dx = np.diff(self.x)[(None, slice(None))+rest]
            c2 = k * np.diff(self.c, axis=0) / dx

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k + nu representing the
            antiderivative of this polynomial.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp

        # Construct the indefinite integrals on individual intervals
        c, x = self.c, self.x
        k = c.shape[0]
        c2 = np.zeros((k+1,) + c.shape[1:], dtype=c.dtype)

        c2[1:, ...] = np.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,)*(c.ndim-2)]

        # Now fix continuity: on the very first interval, take the integration
        # constant to be zero; on an interval [x_j, x_{j+1}) with j>0,
        # the integration constant is then equal to the jump of the `bp` at x_j.
        # The latter is given by the coefficient of B_{n+1, n+1}
        # *on the previous interval* (other B. polynomials are zero at the
        # breakpoint). Finally, use the fact that BPs form a partition of unity.
        c2[:,1:] += np.cumsum(c2[k, :], axis=0)[:-1]

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, x, extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs. If 'periodic', periodic
            extrapolation is used. If None (default), use `self.extrapolate`.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]

        """
        # XXX: can probably use instead the fact that
        # \int_0^{1} B_{j, n}(x) \dx = 1/(n+1)
        ib = self.antiderivative()
        if extrapolate is None:
            extrapolate = self.extrapolate

        # ib.extrapolate shouldn't be 'periodic', it is converted to
        # False for 'periodic. in antiderivative() call.
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            # For simplicity and clarity convert to a <= b case.
            if a <= b:
                sign = 1
            else:
                a, b = b, a
                sign = -1

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))

            # Map a and b to [xs, xe].
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)

            return sign * res
        else:
            return ib(b) - ib(a)

    def extend(self, c, x):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(pp, PPoly):
            raise TypeError(f".from_power_basis only accepts PPoly instances. "
                            f"Got {type(pp)} instead.")

        dx = np.diff(pp.x)
        k = pp.c.shape[0] - 1   # polynomial order

        rest = (None,)*(pp.c.ndim-2)

        c = np.zeros_like(pp.c)
        for a in range(k+1):
            factor = pp.c[a] / comb(k, k-a) * dx[(slice(None),)+rest]**(k-a)
            for j in range(k-a, k+1):
                c[j] += factor * comb(j, k-a)

        if extrapolate is None:
            extrapolate = pp.extrapolate

        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        """Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1-D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j``\\ th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------

        >>> from scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on ``[0, 1]``
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on ``[0, 1]`` and 1 on ``[1, 2]``.
        Notice that no restriction is imposed on the derivatives at
        ``x = 1`` and ``x = 2``.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2

        """
        xi = np.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError("xi and yi need to have the same length")
        if np.any(xi[1:] - xi[:1] <= 0):
            raise ValueError("x coordinates are not in increasing order")

        # number of intervals
        m = len(xi) - 1

        # global poly order is k-1, local orders are <=k and can vary
        try:
            k = max(len(yi[i]) + len(yi[i+1]) for i in range(m))
        except TypeError as e:
            raise ValueError(
                "Using a 1-D array for y? Please .reshape(-1, 1)."
            ) from e

        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (int, np.integer)):
                orders = [orders] * m
            k = max(k, max(orders))

            if any(o <= 0 for o in orders):
                raise ValueError("Orders must be positive.")

        c = []
        for i in range(m):
            y1, y2 = yi[i], yi[i+1]
            if orders[i] is None:
                n1, n2 = len(y1), len(y2)
            else:
                n = orders[i]+1
                n1 = min(n//2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1+n2 != n:
                    mesg = ("Point %g has %d derivatives, point %g"
                            " has %d derivatives, but order %d requested" % (
                               xi[i], len(y1), xi[i+1], len(y2), orders[i]))
                    raise ValueError(mesg)

                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError("`order` input incompatible with"
                                     " length y1 or y2.")

            b = BPoly._construct_from_derivatives(xi[i], xi[i+1],
                                                  y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)

        c = np.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        r"""Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on ``[xa, xb]`` and having the values and derivatives at the
        endpoints `xa` and `xb` as specified by `ya` and `yb`.
        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of `ya` and `yb` are `na` and `nb`, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at `xa`. ``ya[0]`` is the value of the function, and
            ``ya[i]`` for ``i > 0`` is the value of the ``i``\ th derivative.
        yb : array_like
            Derivatives at `xb`.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

          .. math:: Q_a = \sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At ``x = xb`` it's the same with ``a = n - q``.

        """
        ya, yb = np.asarray(ya), np.asarray(yb)
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError(
                f"Shapes of ya {ya.shape} and yb {yb.shape} are incompatible"
            )

        dta, dtb = ya.dtype, yb.dtype
        if (np.issubdtype(dta, np.complexfloating) or
               np.issubdtype(dtb, np.complexfloating)):
            dt = np.complex128
        else:
            dt = np.float64

        na, nb = len(ya), len(yb)
        n = na + nb

        c = np.empty((na+nb,) + ya.shape[1:], dtype=dt)

        # compute coefficients of a polynomial degree na+nb-1
        # walk left-to-right
        for q in range(0, na):
            c[q] = ya[q] / spec.poch(n - q, q) * (xb - xa)**q
            for j in range(0, q):
                c[q] -= (-1)**(j+q) * comb(q, j) * c[j]

        # now walk right-to-left
        for q in range(0, nb):
            c[-q-1] = yb[q] / spec.poch(n - q, q) * (-1)**q * (xb - xa)**q
            for j in range(0, q):
                c[-q-1] -= (-1)**(j+1) * comb(q, j+1) * c[-q+j]

        return c

    @staticmethod
    def _raise_degree(c, d):
        r"""Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \sum_{j=0}^{d} b_{a+j, k+d} \
                                 comb(d, j) / comb(k+d, a+j)

        """
        if d == 0:
            return c

        k = c.shape[0] - 1
        out = np.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)

        for a in range(c.shape[0]):
            f = c[a] * comb(k, a)
            for j in range(d+1):
                out[a+j] += f * comb(d, j) / comb(k+d, a+j)
        return out


class NdPPoly:
    """
    Piecewise tensor product polynomial

    The value at point ``xp = (x', y', z', ...)`` is evaluated by first
    computing the interval indices `i` such that::

        x[0][i[0]] <= x' < x[0][i[0]+1]
        x[1][i[1]] <= y' < x[1][i[1]+1]
        ...

    and then computing::

        S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
                * (xp[0] - x[0][i[0]])**m0
                * ...
                * (xp[n] - x[n][i[n]])**mn
                for m0 in range(k[0]+1)
                ...
                for mn in range(k[n]+1))

    where ``k[j]`` is the degree of the polynomial in dimension j. This
    representation is the piecewise multivariate power basis.

    Parameters
    ----------
    c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
        Polynomial coefficients, with polynomial order `kj` and
        `mj+1` intervals for each dimension `j`.
    x : ndim-tuple of ndarrays, shapes (mj+1,)
        Polynomial breakpoints for each dimension. These must be
        sorted in increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    x : tuple of ndarrays
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    integrate_1d
    construct_fast

    See also
    --------
    PPoly : piecewise polynomials in 1D

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable.

    """

    def __init__(self, c, x, extrapolate=None):
        self.x = tuple(np.ascontiguousarray(v, dtype=np.float64) for v in x)
        self.c = np.asarray(c)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        ndim = len(self.x)
        if any(v.ndim != 1 for v in self.x):
            raise ValueError("x arrays must all be 1-dimensional")
        if any(v.size < 2 for v in self.x):
            raise ValueError("x arrays must all contain at least 2 points")
        if c.ndim < 2*ndim:
            raise ValueError("c must have at least 2*len(x) dimensions")
        if any(np.any(v[1:] - v[:-1] < 0) for v in self.x):
            raise ValueError("x-coordinates are not in increasing order")
        if any(a != b.size - 1 for a, b in zip(c.shape[ndim:2*ndim], self.x)):
            raise ValueError("x and c do not agree on the number of intervals")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type.  The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.

        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex128
        else:
            return np.float64

    def _ensure_c_contiguous(self):
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()
        if not isinstance(self.x, tuple):
            self.x = tuple(self.x)

    def __call__(self, x, nu=None, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nu : tuple, optional
            Orders of derivatives to evaluate. Each must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array-like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        ndim = len(self.x)

        x = _ndim_coords_from_arrays(x)
        x_shape = x.shape
        x = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float64)

        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError("invalid number of derivative orders nu")

        dim1 = prod(self.c.shape[:ndim])
        dim2 = prod(self.c.shape[ndim:2*ndim])
        dim3 = prod(self.c.shape[2*ndim:])
        ks = np.array(self.c.shape[:ndim], dtype=np.intc)

        out = np.empty((x.shape[0], dim3), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        _ppoly.evaluate_nd(self.c.reshape(dim1, dim2, dim3),
                           self.x,
                           ks,
                           x,
                           nu,
                           bool(extrapolate),
                           out)

        return out.reshape(x_shape[:-1] + self.c.shape[2*ndim:])

    def _derivative_inplace(self, nu, axis):
        """
        Compute 1-D derivative along a selected dimension in-place
        May result to non-contiguous c array.
        """
        if nu < 0:
            return self._antiderivative_inplace(-nu, axis)

        ndim = len(self.x)
        axis = axis % ndim

        # reduce order
        if nu == 0:
            # noop
            return
        else:
            sl = [slice(None)]*ndim
            sl[axis] = slice(None, -nu, None)
            c2 = self.c[tuple(sl)]

        if c2.shape[axis] == 0:
            # derivative of order 0 is zero
            shp = list(c2.shape)
            shp[axis] = 1
            c2 = np.zeros(shp, dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[axis], 0, -1), nu)
        sl = [None]*c2.ndim
        sl[axis] = slice(None)
        c2 *= factor[tuple(sl)]

        self.c = c2

    def _antiderivative_inplace(self, nu, axis):
        """
        Compute 1-D antiderivative along a selected dimension
        May result to non-contiguous c array.
        """
        if nu <= 0:
            return self._derivative_inplace(-nu, axis)

        ndim = len(self.x)
        axis = axis % ndim

        perm = list(range(ndim))
        perm[0], perm[axis] = perm[axis], perm[0]
        perm = perm + list(range(ndim, self.c.ndim))

        c = self.c.transpose(perm)

        c2 = np.zeros((c.shape[0] + nu,) + c.shape[1:],
                     dtype=c.dtype)
        c2[:-nu] = c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(c.shape[0], 0, -1), nu)
        c2[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # fix continuity of added degrees of freedom
        perm2 = list(range(c2.ndim))
        perm2[1], perm2[ndim+axis] = perm2[ndim+axis], perm2[1]

        c2 = c2.transpose(perm2)
        c2 = c2.copy()
        _ppoly.fix_continuity(c2.reshape(c2.shape[0], c2.shape[1], -1),
                              self.x[axis], nu-1)

        c2 = c2.transpose(perm2)
        c2 = c2.transpose(perm)

        # Done
        self.c = c2

    def derivative(self, nu):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the antiderivative is returned.

        Returns
        -------
        pp : NdPPoly
            Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])
            representing the derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals in each dimension are
        considered half-open, ``[a, b)``, except for the last interval
        which is closed ``[a, b]``.

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        for axis, n in enumerate(nu):
            p._derivative_inplace(n, axis)

        p._ensure_c_contiguous()
        return p

    def antiderivative(self, nu):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        for axis, n in enumerate(nu):
            p._antiderivative_inplace(n, axis)

        p._ensure_c_contiguous()
        return p

    def integrate_1d(self, a, b, axis, extrapolate=None):
        r"""
        Compute NdPPoly representation for one dimensional definite integral

        The result is a piecewise polynomial representing the integral:

        .. math::

           p(y, z, ...) = \int_a^b dx\, p(x, y, z, ...)

        where the dimension integrated over is specified with the
        `axis` parameter.

        Parameters
        ----------
        a, b : float
            Lower and upper bound for integration.
        axis : int
            Dimension over which to compute the 1-D integrals
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : NdPPoly or array-like
            Definite integral of the piecewise polynomial over [a, b].
            If the polynomial was 1D, an array is returned,
            otherwise, an NdPPoly object.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        ndim = len(self.x)
        axis = int(axis) % ndim

        # reuse 1-D integration routines
        c = self.c
        swap = list(range(c.ndim))
        swap.insert(0, swap[axis])
        del swap[axis + 1]
        swap.insert(1, swap[ndim + axis])
        del swap[ndim + axis + 1]

        c = c.transpose(swap)
        p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1),
                                 self.x[axis],
                                 extrapolate=extrapolate)
        out = p.integrate(a, b, extrapolate=extrapolate)

        # Construct result
        if ndim == 1:
            return out.reshape(c.shape[2:])
        else:
            c = out.reshape(c.shape[2:])
            x = self.x[:axis] + self.x[axis+1:]
            return self.construct_fast(c, x, extrapolate=extrapolate)

    def integrate(self, ranges, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        ranges : ndim-tuple of 2-tuples float
            Sequence of lower and upper bounds for each dimension,
            ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over
            [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]

        """

        ndim = len(self.x)

        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        if not hasattr(ranges, '__len__') or len(ranges) != ndim:
            raise ValueError("Range not a sequence of correct length")

        self._ensure_c_contiguous()

        # Reuse 1D integration routine
        c = self.c
        for n, (a, b) in enumerate(ranges):
            swap = list(range(c.ndim))
            swap.insert(1, swap[ndim - n])
            del swap[ndim - n + 1]

            c = c.transpose(swap)

            p = PPoly.construct_fast(c, self.x[n], extrapolate=extrapolate)
            out = p.integrate(a, b, extrapolate=extrapolate)
            c = out.reshape(c.shape[2:])

        return c
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'CloughTocher2DInterpolator',
    'GradientEstimationWarning',
    'LinearNDInterpolator',
    'NDInterpolatorBase',
    'estimate_gradients_2d_global',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="interpnd",
                                   private_modules=["_interpnd"], all=__all__,
                                   attribute=name)

<|endoftext|>import itertools
import functools
import operator
import numpy as np

from math import prod

from . import _bspl   # type: ignore[attr-defined]

import scipy.sparse.linalg as ssl
from scipy.sparse import csr_array

from ._bsplines import _not_a_knot

__all__ = ["NdBSpline"]


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64


class NdBSpline:
    """Tensor product spline object.

    The value at point ``xp = (x1, x2, ..., xN)`` is evaluated as a linear
    combination of products of one-dimensional b-splines in each of the ``N``
    dimensions::

       c[i1, i2, ..., iN] * B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)


    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Parameters
    ----------
    t : tuple of 1D ndarrays
        knot vectors in directions 1, 2, ... N,
        ``len(t[i]) == n[i] + k + 1``
    c : ndarray, shape (n1, n2, ..., nN, ...)
        b-spline coefficients
    k : int or length-d tuple of integers
        spline degrees.
        A single integer is interpreted as having this degree for
        all dimensions.
    extrapolate : bool, optional
        Whether to extrapolate out-of-bounds inputs, or return `nan`.
        Default is to extrapolate.

    Attributes
    ----------
    t : tuple of ndarrays
        Knots vectors.
    c : ndarray
        Coefficients of the tensor-product spline.
    k : tuple of integers
        Degrees for each dimension.
    extrapolate : bool, optional
        Whether to extrapolate or return nans for out-of-bounds inputs.
        Defaults to true.

    Methods
    -------
    __call__
    design_matrix

    See Also
    --------
    BSpline : a one-dimensional B-spline object
    NdPPoly : an N-dimensional piecewise tensor product polynomial

    """
    def __init__(self, t, c, k, *, extrapolate=None):
        self._k, self._indices_k1d, (self._t, self._len_t) = _preprocess_inputs(k, t)

        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        self.c = np.asarray(c)

        ndim = self._t.shape[0]   # == len(self.t)
        if self.c.ndim < ndim:
            raise ValueError(f"Coefficients must be at least {ndim}-dimensional.")

        for d in range(ndim):
            td = self.t[d]
            kd = self.k[d]
            n = td.shape[0] - kd - 1

            if self.c.shape[d] != n:
                raise ValueError(f"Knots, coefficients and degree in dimension"
                                 f" {d} are inconsistent:"
                                 f" got {self.c.shape[d]} coefficients for"
                                 f" {len(td)} knots, need at least {n} for"
                                 f" k={k}.")

        dt = _get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    @property
    def k(self):
        return tuple(self._k)

    @property
    def t(self):
        # repack the knots into a tuple
        return tuple(self._t[d, :self._len_t[d]] for d in range(self._t.shape[0]))

    def __call__(self, xi, *, nu=None, extrapolate=None):
        """Evaluate the tensor product b-spline at ``xi``.

        Parameters
        ----------
        xi : array_like, shape(..., ndim)
            The coordinates to evaluate the interpolator at.
            This can be a list or tuple of ndim-dimensional points
            or an array with the shape (num_points, ndim).
        nu : array_like, optional, shape (ndim,)
            Orders of derivatives to evaluate. Each must be non-negative.
            Defaults to the zeroth derivivative.
        extrapolate : bool, optional
            Whether to exrapolate based on first and last intervals in each
            dimension, or return `nan`. Default is to ``self.extrapolate``.

        Returns
        -------
        values : ndarray, shape ``xi.shape[:-1] + self.c.shape[ndim:]``
            Interpolated values at ``xi``
        """
        ndim = self._t.shape[0]  # == len(self.t)

        if extrapolate is None:
            extrapolate = self.extrapolate
        extrapolate = bool(extrapolate)

        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError(
                    f"invalid number of derivative orders {nu = } for "
                    f"ndim = {len(self.t)}.")
            if any(nu < 0):
                raise ValueError(f"derivatives must be positive, got {nu = }")

        # prepare xi : shape (..., m1, ..., md) -> (1, m1, ..., md)
        xi = np.asarray(xi, dtype=float)
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.ascontiguousarray(xi)

        if xi_shape[-1] != ndim:
            raise ValueError(f"Shapes: xi.shape={xi_shape} and ndim={ndim}")

        # complex -> double
        was_complex = self.c.dtype.kind == 'c'
        cc = self.c
        if was_complex and self.c.ndim == ndim:
            # make sure that core dimensions are intact, and complex->float
            # size doubling only adds a trailing dimension
            cc = self.c[..., None]
        cc = cc.view(float)

        # prepare the coefficients: flatten the trailing dimensions
        c1 = cc.reshape(cc.shape[:ndim] + (-1,))
        c1r = c1.ravel()

        # replacement for np.ravel_multi_index for indexing of `c1`:
        _strides_c1 = np.asarray([s // c1.dtype.itemsize
                                  for s in c1.strides], dtype=np.intp)

        num_c_tr = c1.shape[-1]  # # of trailing coefficients
        out = np.empty(xi.shape[:-1] + (num_c_tr,), dtype=c1.dtype)

        _bspl.evaluate_ndbspline(xi,
                                 self._t,
                                 self._len_t,
                                 self._k,
                                 nu,
                                 extrapolate,
                                 c1r,
                                 num_c_tr,
                                 _strides_c1,
                                 self._indices_k1d,
                                 out,)
        out = out.view(self.c.dtype)
        return out.reshape(xi_shape[:-1] + self.c.shape[ndim:])

    @classmethod
    def design_matrix(cls, xvals, t, k, extrapolate=True):
        """Construct the design matrix as a CSR format sparse array.

        Parameters
        ----------
        xvals :  ndarray, shape(npts, ndim)
            Data points. ``xvals[j, :]`` gives the ``j``-th data point as an
            ``ndim``-dimensional array.
        t : tuple of 1D ndarrays, length-ndim
            Knot vectors in directions 1, 2, ... ndim,
        k : int
            B-spline degree.
        extrapolate : bool, optional
            Whether to extrapolate out-of-bounds values of raise a `ValueError`

        Returns
        -------
        design_matrix : a CSR array
            Each row of the design matrix corresponds to a value in `xvals` and
            contains values of b-spline basis elements which are non-zero
            at this value.

        """
        xvals = np.asarray(xvals, dtype=float)
        ndim = xvals.shape[-1]
        if len(t) != ndim:
            raise ValueError(
                f"Data and knots are inconsistent: len(t) = {len(t)} for "
                f" {ndim = }."
            )

        # tabulate the flat indices for iterating over the (k+1)**ndim subarray
        k, _indices_k1d, (_t, len_t) = _preprocess_inputs(k, t)

        # Precompute the shape and strides of the 'coefficients array'.
        # This would have been the NdBSpline coefficients; in the present context
        # this is a helper to compute the indices into the colocation matrix.
        c_shape = tuple(len_t[d] - k[d] - 1 for d in range(ndim))

        # The strides of the coeffs array: the computation is equivalent to
        # >>> cstrides = [s // 8 for s in np.empty(c_shape).strides]
        cs = c_shape[1:] + (1,)
        cstrides = np.cumprod(cs[::-1], dtype=np.intp)[::-1].copy()

        # heavy lifting happens here
        data, indices, indptr = _bspl._colloc_nd(xvals,
                                                _t,
                                                len_t,
                                                k,
                                                _indices_k1d,
                                                cstrides)
        return csr_array((data, indices, indptr))


def _preprocess_inputs(k, t_tpl):
    """Helpers: validate and preprocess NdBSpline inputs.

       Parameters
       ----------
       k : int or tuple
          Spline orders
       t_tpl : tuple or array-likes
          Knots.
    """
    # 1. Make sure t_tpl is a tuple
    if not isinstance(t_tpl, tuple):
        raise ValueError(f"Expect `t` to be a tuple of array-likes. "
                         f"Got {t_tpl} instead."
        )

    # 2. Make ``k`` a tuple of integers
    ndim = len(t_tpl)
    try:
        len(k)
    except TypeError:
        # make k a tuple
        k = (k,)*ndim

    k = np.asarray([operator.index(ki) for ki in k], dtype=np.int32)

    if len(k) != ndim:
        raise ValueError(f"len(t) = {len(t_tpl)} != {len(k) = }.")

    # 3. Validate inputs
    ndim = len(t_tpl)
    for d in range(ndim):
        td = np.asarray(t_tpl[d])
        kd = k[d]
        n = td.shape[0] - kd - 1
        if kd < 0:
            raise ValueError(f"Spline degree in dimension {d} cannot be"
                             f" negative.")
        if td.ndim != 1:
            raise ValueError(f"Knot vector in dimension {d} must be"
                             f" one-dimensional.")
        if n < kd + 1:
            raise ValueError(f"Need at least {2*kd + 2} knots for degree"
                             f" {kd} in dimension {d}.")
        if (np.diff(td) < 0).any():
            raise ValueError(f"Knots in dimension {d} must be in a"
                             f" non-decreasing order.")
        if len(np.unique(td[kd:n + 1])) < 2:
            raise ValueError(f"Need at least two internal knots in"
                             f" dimension {d}.")
        if not np.isfinite(td).all():
            raise ValueError(f"Knots in dimension {d} should not have"
                             f" nans or infs.")

    # 4. tabulate the flat indices for iterating over the (k+1)**ndim subarray
    # non-zero b-spline elements
    shape = tuple(kd + 1 for kd in k)
    indices = np.unravel_index(np.arange(prod(shape)), shape)
    _indices_k1d = np.asarray(indices, dtype=np.intp).T.copy()

    # 5. pack the knots into a single array:
    #    ([1, 2, 3, 4], [5, 6], (7, 8, 9)) -->
    #    array([[1, 2, 3, 4],
    #           [5, 6, nan, nan],
    #           [7, 8, 9, nan]])
    ndim = len(t_tpl)
    len_t = [len(ti) for ti in t_tpl]
    _t = np.empty((ndim, max(len_t)), dtype=float)
    _t.fill(np.nan)
    for d in range(ndim):
        _t[d, :len(t_tpl[d])] = t_tpl[d]
    len_t = np.asarray(len_t, dtype=np.int32)

    return k, _indices_k1d, (_t, len_t)


def _iter_solve(a, b, solver=ssl.gcrotmk, **solver_args):
    # work around iterative solvers not accepting multiple r.h.s.

    # also work around a.dtype == float64 and b.dtype == complex128
    # cf https://github.com/scipy/scipy/issues/19644
    if np.issubdtype(b.dtype, np.complexfloating):
        real = _iter_solve(a, b.real, solver, **solver_args)
        imag = _iter_solve(a, b.imag, solver, **solver_args)
        return real + 1j*imag

    if b.ndim == 2 and b.shape[1] !=1:
        res = np.empty_like(b)
        for j in range(b.shape[1]):
            res[:, j], info = solver(a, b[:, j], **solver_args)
            if info != 0:
                raise ValueError(f"{solver = } returns {info =} for column {j}.")
        return res
    else:
        res, info = solver(a, b, **solver_args)
        if info != 0:
            raise ValueError(f"{solver = } returns {info = }.")
        return res


def make_ndbspl(points, values, k=3, *, solver=ssl.gcrotmk, **solver_args):
    """Construct an interpolating NdBspline.

    Parameters
    ----------
    points : tuple of ndarrays of float, with shapes (m1,), ... (mN,)
        The points defining the regular grid in N dimensions. The points in
        each dimension (i.e. every element of the `points` tuple) must be
        strictly ascending or descending.      
    values : ndarray of float, shape (m1, ..., mN, ...)
        The data on the regular grid in n dimensions.
    k : int, optional
        The spline degree. Must be odd. Default is cubic, k=3
    solver : a `scipy.sparse.linalg` solver (iterative or direct), optional.
        An iterative solver from `scipy.sparse.linalg` or a direct one,
        `sparse.sparse.linalg.spsolve`.
        Used to solve the sparse linear system
        ``design_matrix @ coefficients = rhs`` for the coefficients.
        Default is `scipy.sparse.linalg.gcrotmk`
    solver_args : dict, optional
        Additional arguments for the solver. The call signature is
        ``solver(csr_array, rhs_vector, **solver_args)``

    Returns
    -------
    spl : NdBSpline object

    Notes
    -----
    Boundary conditions are not-a-knot in all dimensions.
    """
    ndim = len(points)
    xi_shape = tuple(len(x) for x in points)

    try:
        len(k)
    except TypeError:
        # make k a tuple
        k = (k,)*ndim

    for d, point in enumerate(points):
        numpts = len(np.atleast_1d(point))
        if numpts <= k[d]:
            raise ValueError(f"There are {numpts} points in dimension {d},"
                             f" but order {k[d]} requires at least "
                             f" {k[d]+1} points per dimension.")

    t = tuple(_not_a_knot(np.asarray(points[d], dtype=float), k[d])
              for d in range(ndim))
    xvals = np.asarray([xv for xv in itertools.product(*points)], dtype=float)

    # construct the colocation matrix
    matr = NdBSpline.design_matrix(xvals, t, k)

    # Solve for the coefficients given `values`.
    # Trailing dimensions: first ndim dimensions are data, the rest are batch
    # dimensions, so stack `values` into a 2D array for `spsolve` to undestand.
    v_shape = values.shape
    vals_shape = (prod(v_shape[:ndim]), prod(v_shape[ndim:]))
    vals = values.reshape(vals_shape)

    if solver != ssl.spsolve:
        solver = functools.partial(_iter_solve, solver=solver)
        if "atol" not in solver_args:
            # avoid a DeprecationWarning, grumble grumble
            solver_args["atol"] = 1e-6

    coef = solver(matr, vals, **solver_args)
    coef = coef.reshape(xi_shape + v_shape[ndim:])
    return NdBSpline(t, coef, k)

<|endoftext|>"""Module for RBF interpolation."""
import warnings
from itertools import combinations_with_replacement

import numpy as np
from numpy.linalg import LinAlgError
from scipy.spatial import KDTree
from scipy.special import comb
from scipy.linalg.lapack import dgesv  # type: ignore[attr-defined]

from ._rbfinterp_pythran import (_build_system,
                                 _build_evaluation_coefficients,
                                 _polynomial_matrix)


__all__ = ["RBFInterpolator"]


# These RBFs are implemented.
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
    }


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
    }


def _monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=np.dtype("long"))
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    lhs, rhs, shift, scale = _build_system(
        y, d, smoothing, kernel, epsilon, powers
        )
    _, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
    if info < 0:
        raise ValueError(f"The {-info}-th argument had an illegal value.")
    elif info > 0:
        msg = "Singular matrix."
        nmonos = powers.shape[0]
        if nmonos > 0:
            pmat = _polynomial_matrix((y - shift)/scale, powers)
            rank = np.linalg.matrix_rank(pmat)
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                    )

        raise LinAlgError(msg)

    return shift, scale, coeffs


class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions.

    Parameters
    ----------
    y : (npoints, ndims) array_like
        2-D array of data point coordinates.
    d : (npoints, ...) array_like
        N-D array of data values at `y`. The length of `d` along the first
        axis must be equal to the length of `y`. Unlike some interpolators, the
        interpolation axis cannot be changed.
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    smoothing : float or (npoints, ) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

        Default is 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of data values :math:`d`, which are from
    locations :math:`y`, is a linear combination of RBFs centered at :math:`y`
    plus a polynomial with a specified degree. The RBF interpolant is written
    as

    .. math::
        f(x) = K(x, y) a + P(x) b,

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    evaluated at the points :math:`x`, and :math:`P(x)` is a matrix of
    monomials, which span polynomials with the specified degree, evaluated at
    :math:`x`. The coefficients :math:`a` and :math:`b` are the solution to the
    linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a non-negative smoothing parameter that controls
    how well we want to fit the data. The data are fit exactly when the
    smoothing parameter is 0.

    The above system is uniquely solvable if the following requirements are
    met:

        - :math:`P(y)` must have full column rank. :math:`P(y)` always has full
          column rank when `degree` is -1 or 0. When `degree` is 1,
          :math:`P(y)` has full column rank if the data point locations are not
          all collinear (N=2), coplanar (N=3), etc.
        - If `kernel` is 'multiquadric', 'linear', 'thin_plate_spline',
          'cubic', or 'quintic', then `degree` must not be lower than the
          minimum value listed above.
        - If `smoothing` is 0, then each data point location must be distinct.

    When using an RBF that is not scale invariant ('multiquadric',
    'inverse_multiquadric', 'inverse_quadratic', or 'gaussian'), an appropriate
    shape parameter must be chosen (e.g., through cross validation). Smaller
    values for the shape parameter correspond to wider RBFs. The problem can
    become ill-conditioned or singular when the shape parameter is too small.

    The memory required to solve for the RBF interpolation coefficients
    increases quadratically with the number of data points, which can become
    impractical when interpolating more than about a thousand data points.
    To overcome memory limitations for large interpolation problems, the
    `neighbors` argument can be specified to compute an RBF interpolant for
    each evaluation point using only the nearest data points.

    .. versionadded:: 1.7.0

    See Also
    --------
    NearestNDInterpolator
    LinearNDInterpolator
    CloughTocher2DInterpolator

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    Examples
    --------
    Demonstrate interpolating scattered data to a grid in 2-D.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import RBFInterpolator
    >>> from scipy.stats.qmc import Halton

    >>> rng = np.random.default_rng()
    >>> xobs = 2*Halton(2, seed=rng).random(100) - 1
    >>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    >>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    >>> xflat = xgrid.reshape(2, -1).T
    >>> yflat = RBFInterpolator(xobs, yobs)(xflat)
    >>> ygrid = yflat.reshape(50, 50)

    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    >>> p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    >>> fig.colorbar(p)
    >>> plt.show()

    """

    def __init__(self, y, d,
                 neighbors=None,
                 smoothing=0.0,
                 kernel="thin_plate_spline",
                 epsilon=None,
                 degree=None):
        y = np.asarray(y, dtype=float, order="C")
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")

        ny, ndim = y.shape

        d_dtype = complex if np.iscomplexobj(d) else float
        d = np.asarray(d, dtype=d_dtype, order="C")
        if d.shape[0] != ny:
            raise ValueError(
                f"Expected the first axis of `d` to have length {ny}."
                )

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        d = d.view(float)

        if np.isscalar(smoothing):
            smoothing = np.full(ny, smoothing, dtype=float)
        else:
            smoothing = np.asarray(smoothing, dtype=float, order="C")
            if smoothing.shape != (ny,):
                raise ValueError(
                    "Expected `smoothing` to be a scalar or have shape "
                    f"({ny},)."
                    )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{_SCALE_INVARIANT}."
                    )
        else:
            epsilon = float(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif -1 < degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} except -1 "
                    f"when `kernel` is '{kernel}'."
                    f"The interpolant may not be uniquely "
                    f"solvable, and the smoothing parameter may have an "
                    f"unintuitive effect.",
                    UserWarning, stacklevel=2
                )

        if neighbors is None:
            nobs = ny
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, ny))
            nobs = neighbors

        powers = _monomial_powers(ndim, degree)
        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}."
                )

        if neighbors is None:
            shift, scale, coeffs = _build_and_solve_system(
                y, d, smoothing, kernel, epsilon, powers
                )

            # Make these attributes private since they do not always exist.
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs

        else:
            self._tree = KDTree(y)

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.powers = powers

    def _chunk_evaluator(
            self,
            x,
            y,
            shift,
            scale,
            coeffs,
            memory_budget=1000000
    ):
        """
        Evaluate the interpolation while controlling memory consumption.
        We chunk the input if we need more memory than specified.

        Parameters
        ----------
        x : (Q, N) float ndarray
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget: int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) float ndarray
        Interpolated array
        """
        nx, ndim = x.shape
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors
        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1
        if chunksize <= nx:
            out = np.empty((nx, self.d.shape[1]), dtype=float)
            for i in range(0, nx, chunksize):
                vec = _build_evaluation_coefficients(
                    x[i:i + chunksize, :],
                    y,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale)
                out[i:i + chunksize, :] = np.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(
                x,
                y,
                self.kernel,
                self.epsilon,
                self.powers,
                shift,
                scale)
            out = np.dot(vec, coeffs)
        return out

    def __call__(self, x):
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """
        x = np.asarray(x, dtype=float, order="C")
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError("Expected the second axis of `x` to have length "
                             f"{self.y.shape[1]}.")

        # Our memory budget for storing RBF coefficients is
        # based on how many floats in memory we already occupy
        # If this number is below 1e6 we just use 1e6
        # This memory budget is used to decide how we chunk
        # the inputs
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

        if self.neighbors is None:
            out = self._chunk_evaluator(
                x,
                self.y,
                self._shift,
                self._scale,
                self._coeffs,
                memory_budget=memory_budget)
        else:
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            _, yindices = self._tree.query(x, self.neighbors)
            if self.neighbors == 1:
                # `KDTree` squeezes the output when neighbors=1.
                yindices = yindices[:, None]

            # Multiple evaluation points may have the same neighborhood of
            # observation points. Make the neighborhoods unique so that we only
            # compute the interpolation coefficients once for each
            # neighborhood.
            yindices = np.sort(yindices, axis=1)
            yindices, inv = np.unique(yindices, return_inverse=True, axis=0)
            inv = np.reshape(inv, (-1,))  # flatten, we need 1-D indices
            # `inv` tells us which neighborhood will be used by each evaluation
            # point. Now we find which evaluation points will be using each
            # neighborhood.
            xindices = [[] for _ in range(len(yindices))]
            for i, j in enumerate(inv):
                xindices[j].append(i)

            out = np.empty((nx, self.d.shape[1]), dtype=float)
            for xidx, yidx in zip(xindices, yindices):
                # `yidx` are the indices of the observations in this
                # neighborhood. `xidx` are the indices of the evaluation points
                # that are using this neighborhood.
                xnbr = x[xidx]
                ynbr = self.y[yidx]
                dnbr = self.d[yidx]
                snbr = self.smoothing[yidx]
                shift, scale, coeffs = _build_and_solve_system(
                    ynbr,
                    dnbr,
                    snbr,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                )
                out[xidx] = self._chunk_evaluator(
                    xnbr,
                    ynbr,
                    shift,
                    scale,
                    coeffs,
                    memory_budget=memory_budget)

        out = out.view(self.d_dtype)
        out = out.reshape((nx, ) + self.d_shape)
        return out
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BivariateSpline',
    'InterpolatedUnivariateSpline',
    'LSQBivariateSpline',
    'LSQSphereBivariateSpline',
    'LSQUnivariateSpline',
    'RectBivariateSpline',
    'RectSphereBivariateSpline',
    'SmoothBivariateSpline',
    'SmoothSphereBivariateSpline',
    'UnivariateSpline',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="fitpack2",
                                   private_modules=["_fitpack2"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'CloughTocher2DInterpolator',
    'LinearNDInterpolator',
    'NearestNDInterpolator',
    'griddata',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="ndgriddata",
                                   private_modules=["_ndgriddata"], all=__all__,
                                   attribute=name)
<|endoftext|>__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
           'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']


import numpy as np

# These are in the API for fitpack even if not used in fitpack.py itself.
from ._fitpack_impl import bisplrep, bisplev, dblint  # noqa: F401
from . import _fitpack_impl as _impl
from ._bsplines import BSpline


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    """
    Find the B-spline representation of an N-D curve.

    .. legacy:: function

        Specifically, we recommend using `make_splprep` in new code.

    Given a list of N rank-1 arrays, `x`, which represent a curve in
    N-dimensional space parametrized by `u`, find a smooth approximating
    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

    Parameters
    ----------
    x : array_like
        A list of sample vector arrays representing the curve.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x[0]`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `x` values have standard-deviation given by
        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
    u : array_like, optional
        An array of parameter values. If not given, these values are
        calculated automatically as ``M = len(x[0])``, where

            v[0] = 0

            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

            u[i] = v[i] / v[M-1]

    ub, ue : int, optional
        The end-points of the parameters interval.  Defaults to
        u[0] and u[-1].
    k : int, optional
        Degree of the spline. Cubic splines are recommended.
        Even values of `k` should be avoided especially with a small s-value.
        ``1 <= k <= 5``, default is 3.
    task : int, optional
        If task==0 (default), find t and c for a given smoothing factor, s.
        If task==1, find t and c for another value of the smoothing factor, s.
        There must have been a previous call with task=0 or task=1
        for the same set of data.
        If task=-1 find the weighted least square spline for a given set of
        knots, t.
    s : float, optional
        A smoothing condition.  The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
        where g(x) is the smoothed interpolation of (x,y).  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit.  Larger `s` means more smoothing while smaller values of `s`
        indicate less smoothing. Recommended values of `s` depend on the
        weights, w.  If the weights represent the inverse of the
        standard-deviation of y, then a good `s` value should be found in
        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
        data points in x, y, and w.
    t : array, optional
        The knots needed for ``task=-1``.
        There must be at least ``2*k+2`` knots.
    full_output : int, optional
        If non-zero, then return optional outputs.
    nest : int, optional
        An over-estimate of the total number of knots of the spline to
        help in determining the storage space.  By default nest=m/2.
        Always large enough is nest=m+k+1.
    per : int, optional
       If non-zero, data points are considered periodic with period
       ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
       returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
    quiet : int, optional
         Non-zero to suppress messages.

    Returns
    -------
    tck : tuple
        A tuple, ``(t,c,k)`` containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.
    u : array
        An array of the values of the parameter.
    fp : float
        The weighted sum of squared residuals of the spline approximation.
    ier : int
        An integer flag about splrep success.  Success is indicated
        if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
        Otherwise an error is raised.
    msg : str
        A message corresponding to the integer flag, ier.

    See Also
    --------
    splrep, splev, sproot, spalde, splint,
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline
    BSpline
    make_interp_spline

    Notes
    -----
    See `splev` for evaluation of the spline and its derivatives.
    The number of dimensions N must be smaller than 11.

    The number of coefficients in the `c` array is ``k+1`` less than the number
    of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads
    the array of coefficients to have the same length as the array of knots.
    These additional coefficients are ignored by evaluation routines, `splev`
    and `BSpline`.

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines", report tw55, Dept. Computer Science,
        K.U.Leuven, 1981.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Generate a discretization of a limacon curve in the polar coordinates:

    >>> import numpy as np
    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.5 + np.cos(phi)         # polar coords
    >>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian

    And interpolate:

    >>> from scipy.interpolate import splprep, splev
    >>> tck, u = splprep([x, y], s=0)
    >>> new_points = splev(u, tck)

    Notice that (i) we force interpolation by using ``s=0``,
    (ii) the parameterization, ``u``, is generated automatically.
    Now plot the result:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, 'ro')
    >>> ax.plot(new_points[0], new_points[1], 'r-')
    >>> plt.show()

    """

    res = _impl.splprep(x, w, u, ub, ue, k, task, s, t, full_output, nest, per,
                        quiet)
    return res


def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):
    """
    Find the B-spline representation of a 1-D curve.

    .. legacy:: function

        Specifically, we recommend using `make_splrep` in new code.


    Given the set of data points ``(x[i], y[i])`` determine a smooth spline
    approximation of degree k on the interval ``xb <= x <= xe``.

    Parameters
    ----------
    x, y : array_like
        The data points defining a curve ``y = f(x)``.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x` and `y`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `y` values have standard-deviation given by the
        vector ``d``, then `w` should be ``1/d``. Default is ``ones(len(x))``.
    xb, xe : float, optional
        The interval to fit.  If None, these default to ``x[0]`` and ``x[-1]``
        respectively.
    k : int, optional
        The degree of the spline fit. It is recommended to use cubic splines.
        Even values of `k` should be avoided especially with small `s` values.
        ``1 <= k <= 5``.
    task : {1, 0, -1}, optional
        If ``task==0``, find ``t`` and ``c`` for a given smoothing factor, `s`.

        If ``task==1`` find ``t`` and ``c`` for another value of the smoothing factor,
        `s`. There must have been a previous call with ``task=0`` or ``task=1`` for
        the same set of data (``t`` will be stored an used internally)

        If ``task=-1`` find the weighted least square spline for a given set of
        knots, ``t``. These should be interior knots as knots on the ends will be
        added automatically.
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s`` where ``g(x)``
        is the smoothed interpolation of ``(x,y)``. The user can use `s` to control
        the tradeoff between closeness and smoothness of fit. Larger `s` means
        more smoothing while smaller values of `s` indicate less smoothing.
        Recommended values of `s` depend on the weights, `w`. If the weights
        represent the inverse of the standard-deviation of `y`, then a good `s`
        value should be found in the range ``(m-sqrt(2*m),m+sqrt(2*m))`` where ``m`` is
        the number of datapoints in `x`, `y`, and `w`. default : ``s=m-sqrt(2*m)`` if
        weights are supplied. ``s = 0.0`` (interpolating) if no weights are
        supplied.
    t : array_like, optional
        The knots needed for ``task=-1``. If given then task is automatically set
        to ``-1``.
    full_output : bool, optional
        If non-zero, then return optional outputs.
    per : bool, optional
        If non-zero, data points are considered periodic with period ``x[m-1]`` -
        ``x[0]`` and a smooth periodic spline approximation is returned. Values of
        ``y[m-1]`` and ``w[m-1]`` are not used.
        The default is zero, corresponding to boundary condition 'not-a-knot'.
    quiet : bool, optional
        Non-zero to suppress messages.

    Returns
    -------
    tck : tuple
        A tuple ``(t,c,k)`` containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.
    fp : array, optional
        The weighted sum of squared residuals of the spline approximation.
    ier : int, optional
        An integer flag about splrep success. Success is indicated if ``ier<=0``.
        If ``ier in [1,2,3]``, an error occurred but was not raised. Otherwise an
        error is raised.
    msg : str, optional
        A message corresponding to the integer flag, `ier`.

    See Also
    --------
    UnivariateSpline, BivariateSpline
    splprep, splev, sproot, spalde, splint
    bisplrep, bisplev
    BSpline
    make_interp_spline

    Notes
    -----
    See `splev` for evaluation of the spline and its derivatives. Uses the
    FORTRAN routine ``curfit`` from FITPACK.

    The user is responsible for assuring that the values of `x` are unique.
    Otherwise, `splrep` will not return sensible results.

    If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    This routine zero-pads the coefficients array ``c`` to have the same length
    as the array of knots ``t`` (the trailing ``k + 1`` coefficients are ignored
    by the evaluation routines, `splev` and `BSpline`.) This is in contrast with
    `splprep`, which does not zero-pad the coefficients.

    The default boundary condition is 'not-a-knot', i.e. the first and second
    segment at a curve end are the same polynomial. More boundary conditions are
    available in `CubicSpline`.

    References
    ----------
    Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:

    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
       integration of experimental data using spline functions",
       J.Comp.Appl.Maths 1 (1975) 165-184.
    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
       1286-1304.
    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    You can interpolate 1-D points with a B-spline curve.
    Further examples are given in
    :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import splev, splrep
    >>> x = np.linspace(0, 10, 10)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y)
    >>> x2 = np.linspace(0, 10, 200)
    >>> y2 = splev(x2, spl)
    >>> plt.plot(x, y, 'o', x2, y2)
    >>> plt.show()

    """
    res = _impl.splrep(x, y, w, xb, xe, k, task, s, t, full_output, per, quiet)
    return res


def splev(x, tck, der=0, ext=0):
    """
    Evaluate a B-spline or its derivatives.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using
        its ``__call__`` method.

    Given the knots and coefficients of a B-spline representation, evaluate
    the value of the smoothing polynomial and its derivatives. This is a
    wrapper around the FORTRAN routines splev and splder of FITPACK.

    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : BSpline instance or tuple
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    ext : int, optional
        Controls the value returned for elements of ``x`` not in the
        interval defined by the knot sequence.

        * if ext=0, return the extrapolated value.
        * if ext=1, return 0
        * if ext=2, raise a ValueError
        * if ext=3, return the boundary value.

        The default value is 0.

    Returns
    -------
    y : ndarray or list of ndarrays
        An array of values representing the spline function evaluated at
        the points in `x`.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in an N-D space.

    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev
    BSpline

    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using `BSpline` objects.

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    A comparison between `splev`, `splder` and `spalde` to compute the derivatives of a 
    B-spline can be found in the `spalde` examples section.

    """
    if isinstance(tck, BSpline):
        if tck.c.ndim > 1:
            mesg = ("Calling splev() with BSpline objects with c.ndim > 1 is "
                    "not allowed. Use BSpline.__call__(x) instead.")
            raise ValueError(mesg)

        # remap the out-of-bounds behavior
        try:
            extrapolate = {0: True, }[ext]
        except KeyError as e:
            raise ValueError(f"Extrapolation mode {ext} is not supported "
                             "by BSpline.") from e

        return tck(x, der, extrapolate=extrapolate)
    else:
        return _impl.splev(x, tck, der, ext)


def splint(a, b, tck, full_output=0):
    """
    Evaluate the definite integral of a B-spline between two given points.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using its
        ``integrate`` method.

    Parameters
    ----------
    a, b : float
        The end-points of the integration interval.
    tck : tuple or a BSpline instance
        If a tuple, then it should be a sequence of length 3, containing the
        vector of knots, the B-spline coefficients, and the degree of the
        spline (see `splev`).
    full_output : int, optional
        Non-zero to return optional output.

    Returns
    -------
    integral : float
        The resulting integral.
    wrk : ndarray
        An array containing the integrals of the normalized B-splines
        defined on the set of knots.
        (Only returned if `full_output` is non-zero)

    See Also
    --------
    splprep, splrep, sproot, spalde, splev
    bisplrep, bisplev
    BSpline

    Notes
    -----
    `splint` silently assumes that the spline function is zero outside the data
    interval (`a`, `b`).

    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using the `BSpline` objects.

    References
    ----------
    .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",
        J. Inst. Maths Applics, 17, p.37-41, 1976.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    """
    if isinstance(tck, BSpline):
        if tck.c.ndim > 1:
            mesg = ("Calling splint() with BSpline objects with c.ndim > 1 is "
                    "not allowed. Use BSpline.integrate() instead.")
            raise ValueError(mesg)

        if full_output != 0:
            mesg = (f"full_output = {full_output} is not supported. Proceeding as if "
                    "full_output = 0")

        return tck.integrate(a, b, extrapolate=False)
    else:
        return _impl.splint(a, b, tck, full_output)


def sproot(tck, mest=10):
    """
    Find the roots of a cubic B-spline.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using the
        following pattern: `PPoly.from_spline(spl).roots()`.

    Given the knots (>=8) and coefficients of a cubic B-spline return the
    roots of the spline.

    Parameters
    ----------
    tck : tuple or a BSpline object
        If a tuple, then it should be a sequence of length 3, containing the
        vector of knots, the B-spline coefficients, and the degree of the
        spline.
        The number of knots must be >= 8, and the degree must be 3.
        The knots must be a montonically increasing sequence.
    mest : int, optional
        An estimate of the number of zeros (Default is 10).

    Returns
    -------
    zeros : ndarray
        An array giving the roots of the spline.

    See Also
    --------
    splprep, splrep, splint, spalde, splev
    bisplrep, bisplev
    BSpline

    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using the `BSpline` objects.

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------

    For some data, this method may miss a root. This happens when one of
    the spline knots (which FITPACK places automatically) happens to
    coincide with the true root. A workaround is to convert to `PPoly`,
    which uses a different root-finding algorithm.

    For example,

    >>> x = [1.96, 1.97, 1.98, 1.99, 2.00, 2.01, 2.02, 2.03, 2.04, 2.05]
    >>> y = [-6.365470e-03, -4.790580e-03, -3.204320e-03, -1.607270e-03,
    ...      4.440892e-16,  1.616930e-03,  3.243000e-03,  4.877670e-03,
    ...      6.520430e-03,  8.170770e-03]
    >>> from scipy.interpolate import splrep, sproot, PPoly
    >>> tck = splrep(x, y, s=0)
    >>> sproot(tck)
    array([], dtype=float64)

    Converting to a PPoly object does find the roots at ``x=2``:

    >>> ppoly = PPoly.from_spline(tck)
    >>> ppoly.roots(extrapolate=False)
    array([2.])


    Further examples are given :ref:`in the tutorial
    <tutorial-interpolate_splXXX>`.

    """
    if isinstance(tck, BSpline):
        if tck.c.ndim > 1:
            mesg = ("Calling sproot() with BSpline objects with c.ndim > 1 is "
                    "not allowed.")
            raise ValueError(mesg)

        t, c, k = tck.tck

        # _impl.sproot expects the interpolation axis to be last, so roll it.
        # NB: This transpose is a no-op if c is 1D.
        sh = tuple(range(c.ndim))
        c = c.transpose(sh[1:] + (0,))
        return _impl.sproot((t, c, k), mest)
    else:
        return _impl.sproot(tck, mest)


def spalde(x, tck):
    """
    Evaluate a B-spline and all its derivatives at one point (or set of points) up
    to order k (the degree of the spline), being 0 the spline itself.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and evaluate
        its derivative in a loop or a list comprehension.

    Parameters
    ----------
    x : array_like
        A point or a set of points at which to evaluate the derivatives.
        Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
    tck : tuple
        A tuple (t,c,k) containing the vector of knots,
        the B-spline coefficients, and the degree of the spline whose 
        derivatives to compute.

    Returns
    -------
    results : {ndarray, list of ndarrays}
        An array (or a list of arrays) containing all derivatives
        up to order k inclusive for each point `x`, being the first element the 
        spline itself.

    See Also
    --------
    splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
    UnivariateSpline, BivariateSpline

    References
    ----------
    .. [1] de Boor C : On calculating with b-splines, J. Approximation Theory
       6 (1972) 50-62.
    .. [2] Cox M.G. : The numerical evaluation of b-splines, J. Inst. Maths
       applics 10 (1972) 134-149.
    .. [3] Dierckx P. : Curve and surface fitting with splines, Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    To calculate the derivatives of a B-spline there are several aproaches. 
    In this example, we will demonstrate that `spalde` is equivalent to
    calling `splev` and `splder`.
    
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import BSpline, spalde, splder, splev
    
    >>> # Store characteristic parameters of a B-spline
    >>> tck = ((-2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2),  # knots
    ...        (0, 0, 0, 6, 0, 0, 0),  # coefficients
    ...        3)  # degree (cubic)
    >>> # Instance a B-spline object
    >>> # `BSpline` objects are preferred, except for spalde()
    >>> bspl = BSpline(tck[0], tck[1], tck[2])
    >>> # Generate extra points to get a smooth curve
    >>> x = np.linspace(min(tck[0]), max(tck[0]), 100)
    
    Evaluate the curve and all derivatives
    
    >>> # The order of derivative must be less or equal to k, the degree of the spline
    >>> # Method 1: spalde()
    >>> f1_y_bsplin = [spalde(i, tck)[0] for i in x ]  # The B-spline itself
    >>> f1_y_deriv1 = [spalde(i, tck)[1] for i in x ]  # 1st derivative
    >>> f1_y_deriv2 = [spalde(i, tck)[2] for i in x ]  # 2nd derivative
    >>> f1_y_deriv3 = [spalde(i, tck)[3] for i in x ]  # 3rd derivative
    >>> # You can reach the same result by using `splev`and `splder`
    >>> f2_y_deriv3 = splev(x, bspl, der=3)
    >>> f3_y_deriv3 = splder(bspl, n=3)(x)
    
    >>> # Generate a figure with three axes for graphic comparison
    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    >>> suptitle = fig.suptitle(f'Evaluate a B-spline and all derivatives')
    >>> # Plot B-spline and all derivatives using the three methods
    >>> orders = range(4)
    >>> linetypes = ['-', '--', '-.', ':']
    >>> labels = ['B-Spline', '1st deriv.', '2nd deriv.', '3rd deriv.']
    >>> functions = ['splev()', 'splder()', 'spalde()']
    >>> for order, linetype, label in zip(orders, linetypes, labels):
    ...     ax1.plot(x, splev(x, bspl, der=order), linetype, label=label)
    ...     ax2.plot(x, splder(bspl, n=order)(x), linetype, label=label)
    ...     ax3.plot(x, [spalde(i, tck)[order] for i in x], linetype, label=label)
    >>> for ax, function in zip((ax1, ax2, ax3), functions):
    ...     ax.set_title(function)
    ...     ax.legend()
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if isinstance(tck, BSpline):
        raise TypeError("spalde does not accept BSpline instances.")
    else:
        return _impl.spalde(x, tck)


def insert(x, tck, m=1, per=0):
    """
    Insert knots into a B-spline.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using
        its ``insert_knot`` method.

    Given the knots and coefficients of a B-spline representation, create a
    new B-spline with a knot inserted `m` times at point `x`.
    This is a wrapper around the FORTRAN routine insert of FITPACK.

    Parameters
    ----------
    x (u) : float
        A knot value at which to insert a new knot.  If `tck` was returned
        from ``splprep``, then the parameter values, u should be given.
    tck : a `BSpline` instance or a tuple
        If tuple, then it is expected to be a tuple (t,c,k) containing
        the vector of knots, the B-spline coefficients, and the degree of
        the spline.
    m : int, optional
        The number of times to insert the given knot (its multiplicity).
        Default is 1.
    per : int, optional
        If non-zero, the input spline is considered periodic.

    Returns
    -------
    BSpline instance or a tuple
        A new B-spline with knots t, coefficients c, and degree k.
        ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
        In case of a periodic spline (``per != 0``) there must be
        either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
        or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
        A tuple is returned iff the input argument `tck` is a tuple, otherwise
        a BSpline object is constructed and returned.

    Notes
    -----
    Based on algorithms from [1]_ and [2]_.

    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using the `BSpline` objects, in particular `BSpline.insert_knot`
    method.

    See Also
    --------
    BSpline.insert_knot

    References
    ----------
    .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
        Computer Aided Design, 12, p.199-201, 1980.
    .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
        Numerical Analysis", Oxford University Press, 1993.

    Examples
    --------
    You can insert knots into a B-spline.

    >>> from scipy.interpolate import splrep, insert
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 5)
    >>> y = np.sin(x)
    >>> tck = splrep(x, y)
    >>> tck[0]
    array([ 0.,  0.,  0.,  0.,  5., 10., 10., 10., 10.])

    A knot is inserted:

    >>> tck_inserted = insert(3, tck)
    >>> tck_inserted[0]
    array([ 0.,  0.,  0.,  0.,  3.,  5., 10., 10., 10., 10.])

    Some knots are inserted:

    >>> tck_inserted2 = insert(8, tck, m=3)
    >>> tck_inserted2[0]
    array([ 0.,  0.,  0.,  0.,  5.,  8.,  8.,  8., 10., 10., 10., 10.])

    """
    if isinstance(tck, BSpline):

        t, c, k = tck.tck

        # FITPACK expects the interpolation axis to be last, so roll it over
        # NB: if c array is 1D, transposes are no-ops
        sh = tuple(range(c.ndim))
        c = c.transpose(sh[1:] + (0,))
        t_, c_, k_ = _impl.insert(x, (t, c, k), m, per)

        # and roll the last axis back
        c_ = np.asarray(c_)
        c_ = c_.transpose((sh[-1],) + sh[:-1])
        return BSpline(t_, c_, k_)
    else:
        return _impl.insert(x, tck, m, per)


def splder(tck, n=1):
    """
    Compute the spline representation of the derivative of a given spline

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using its
        ``derivative`` method.

    Parameters
    ----------
    tck : BSpline instance or tuple
        BSpline instance or a tuple (t,c,k) containing the vector of knots,
        the B-spline coefficients, and the degree of the spline whose 
        derivative to compute
    n : int, optional
        Order of derivative to evaluate. Default: 1

    Returns
    -------
    `BSpline` instance or tuple
        Spline of order k2=k-n representing the derivative
        of the input spline.
        A tuple is returned if the input argument `tck` is a tuple, otherwise
        a BSpline object is constructed and returned.

    See Also
    --------
    splantider, splev, spalde
    BSpline

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    This can be used for finding maxima of a curve:

    >>> from scipy.interpolate import splrep, splder, sproot
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 70)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y, k=4)

    Now, differentiate the spline and find the zeros of the
    derivative. (NB: `sproot` only works for order 3 splines, so we
    fit an order 4 spline):

    >>> dspl = splder(spl)
    >>> sproot(dspl) / np.pi
    array([ 0.50000001,  1.5       ,  2.49999998])

    This agrees well with roots :math:`\\pi/2 + n\\pi` of
    :math:`\\cos(x) = \\sin'(x)`.

    A comparison between `splev`, `splder` and `spalde` to compute the derivatives of a 
    B-spline can be found in the `spalde` examples section.

    """
    if isinstance(tck, BSpline):
        return tck.derivative(n)
    else:
        return _impl.splder(tck, n)


def splantider(tck, n=1):
    """
    Compute the spline for the antiderivative (integral) of a given spline.

    .. legacy:: function

        Specifically, we recommend constructing a `BSpline` object and using its
        ``antiderivative`` method.

    Parameters
    ----------
    tck : BSpline instance or a tuple of (t, c, k)
        Spline whose antiderivative to compute
    n : int, optional
        Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    BSpline instance or a tuple of (t2, c2, k2)
        Spline of order k2=k+n representing the antiderivative of the input
        spline.
        A tuple is returned iff the input argument `tck` is a tuple, otherwise
        a BSpline object is constructed and returned.

    See Also
    --------
    splder, splev, spalde
    BSpline

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    .. versionadded:: 0.13.0

    Examples
    --------
    >>> from scipy.interpolate import splrep, splder, splantider, splev
    >>> import numpy as np
    >>> x = np.linspace(0, np.pi/2, 70)
    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
    >>> spl = splrep(x, y)

    The derivative is the inverse operation of the antiderivative,
    although some floating point error accumulates:

    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
    (array(2.1565429877197317), array(2.1565429877201865))

    Antiderivative can be used to evaluate definite integrals:

    >>> ispl = splantider(spl)
    >>> splev(np.pi/2, ispl) - splev(0, ispl)
    2.2572053588768486

    This is indeed an approximation to the complete elliptic integral
    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

    >>> from scipy.special import ellipk
    >>> ellipk(0.8)
    2.2572053268208538

    """
    if isinstance(tck, BSpline):
        return tck.antiderivative(n)
    else:
        return _impl.splantider(tck, n)

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BPoly',
    'BSpline',
    'NdPPoly',
    'PPoly',
    'RectBivariateSpline',
    'RegularGridInterpolator',
    'interp1d',
    'interp2d',
    'interpn',
    'lagrange',
    'make_interp_spline',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="interpolate",
                                   private_modules=["_interpolate", "fitpack2", "_rgi"],
                                   all=__all__, attribute=name)
<|endoftext|>from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg

__all__ = ["pade"]

def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.

    Parameters
    ----------
    an : (N,) array_like
        Taylor series coefficients.
    m : int
        The order of the returned approximating polynomial `q`.
    n : int, optional
        The order of the returned approximating polynomial `p`. By default,
        the order is ``len(an)-1-m``.

    Returns
    -------
    p, q : Polynomial class
        The Pade approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import pade
    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
    >>> p, q = pade(e_exp, 2)

    >>> e_exp.reverse()
    >>> e_poly = np.poly1d(e_exp)

    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

    >>> e_poly(1)
    2.7166666666666668

    >>> p(1)/q(1)
    2.7179487179487181

    """
    an = asarray(an)
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    if n < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + n
    if N > len(an)-1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    an = an[:N+1]
    Akj = eye(N+1, n+1, dtype=an.dtype)
    Bkj = zeros((N+1, m), dtype=an.dtype)
    for row in range(1, m+1):
        Bkj[row,:row] = -(an[:row])[::-1]
    for row in range(m+1, N+1):
        Bkj[row,:] = -(an[row-m:row])[::-1]
    C = hstack((Akj, Bkj))
    pq = linalg.solve(C, an)
    p = pq[:n+1]
    q = r_[1.0, pq[n+1:]]
    return poly1d(p[::-1]), poly1d(q[::-1])

<|endoftext|>"""Interpolation algorithms using piecewise cubic polynomials."""

from typing import Literal

import numpy as np

from scipy.linalg import solve, solve_banded

from . import PPoly
from ._polyint import _isscalar

__all__ = ["CubicHermiteSpline", "PchipInterpolator", "pchip_interpolate",
           "Akima1DInterpolator", "CubicSpline"]


def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.

    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).
    """

    x, y = map(np.asarray, (x, y))
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float

    if dydx is not None:
        dydx = np.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if np.issubdtype(dydx.dtype, np.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError(f"The length of `y` along `axis`={axis} doesn't "
                         "match the length of `x`")

    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not np.all(np.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = np.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = np.moveaxis(dydx, axis, 0)

    return x, dx, y, axis, dydx


class CubicHermiteSpline(PPoly):
    """Piecewise-cubic interpolator matching values and first derivatives.

    The result is represented as a `PPoly` instance.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    dydx : array_like
        Array containing derivatives of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), it is set to True.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-D, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    If you want to create a higher-order spline matching higher-order
    derivatives, use `BPoly.from_derivatives`.

    References
    ----------
    .. [1] `Cubic Hermite spline
            <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_
            on Wikipedia.
    """

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = True

        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

        c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - dydx[:-1]) / dxr - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]

        super().__init__(c, x, extrapolate=extrapolate)
        self.axis = axis


class PchipInterpolator(CubicHermiteSpline):
    r"""PCHIP 1-D monotonic cubic interpolation.

    ``x`` and ``y`` are arrays of values used to approximate some function f,
    with ``y = f(x)``. The interpolant uses monotonic cubic splines
    to find the value of new points. (PCHIP stands for Piecewise Cubic
    Hermite Interpolating Polynomial).

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
    y : ndarray, shape (..., npoints, ...)
        A N-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. Use the ``axis``
        parameter to select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.

    Methods
    -------
    __call__
    derivative
    antiderivative
    roots

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    Akima1DInterpolator : Akima 1D interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.

    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at :math:`x_k`.

    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
    by using PCHIP algorithm [1]_.

    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
    are the slopes at internal points :math:`x_k`.
    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
    weighted harmonic mean

    .. math::

        \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

    The end slopes are set using a one-sided scheme [2]_.


    References
    ----------
    .. [1] F. N. Fritsch and J. Butland,
           A method for constructing local
           monotone piecewise cubic interpolants,
           SIAM J. Sci. Comput., 5(2), 300-304 (1984).
           :doi:`10.1137/0905021`.
    .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
           :doi:`10.1137/1.9780898717952`

    """

    def __init__(self, x, y, axis=0, extrapolate=None):
        x, _, y, axis, _ = prepare_input(x, y, axis)
        if np.iscomplexobj(y):
            msg = ("`PchipInterpolator` only works with real values for `y`. "
                   "If you are trying to use the real components of the passed array, "
                   "use `np.real` on the array before passing to `PchipInterpolator`.")
            raise ValueError(msg)
        xp = x.reshape((x.shape[0],) + (1,)*(y.ndim-1))
        dk = self._find_derivatives(xp, y)
        super().__init__(x, y, dk, axis=0, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _edge_case(h0, h1, m0, m1):
        # one-sided three-point estimate for the derivative
        d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)

        # try to preserve shape
        mask = np.sign(d) != np.sign(m0)
        mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.*np.abs(m0))
        mmm = (~mask) & mask2

        d[mask] = 0.
        d[mmm] = 3.*m0[mmm]

        return d

    @staticmethod
    def _find_derivatives(x, y):
        # Determine the derivatives at the points y_k, d_k, by using
        #  PCHIP algorithm is:
        # We choose the derivatives at the point x_k by
        # Let m_k be the slope of the kth segment (between k and k+1)
        # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
        # else use weighted harmonic mean:
        #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
        #   where h_k is the spacing between x_k and x_{k+1}
        y_shape = y.shape
        if y.ndim == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            y = y[:, None]

        hk = x[1:] - x[:-1]
        mk = (y[1:] - y[:-1]) / hk

        if y.shape[0] == 2:
            # edge case: only have two points, use linear interpolation
            dk = np.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return dk.reshape(y_shape)

        smk = np.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)

        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]

        # values where division by zero occurs will be excluded
        # by 'condition' afterwards
        with np.errstate(divide='ignore', invalid='ignore'):
            whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1 + w2)

        dk = np.zeros_like(y)
        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean[~condition]

        # special case endpoints, as suggested in
        # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
        dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return dk.reshape(y_shape)


def pchip_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for pchip interpolation.

    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``. The interpolant uses monotonic cubic splines
    to find the value of new points x and the derivatives there.

    See `scipy.interpolate.PchipInterpolator` for details.

    Parameters
    ----------
    xi : array_like
        A sorted list of x-coordinates, of length N.
    yi : array_like
        A 1-D array of real values. `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.

        .. deprecated:: 1.13.0
            Complex data is deprecated and will raise an error in
            SciPy 1.15.0. If you are trying to use the real components of
            the passed array, use ``np.real`` on `yi`.

    x : scalar or array_like
        Of length M.
    der : int or list, optional
        Derivatives to extract. The 0th derivative can be included to
        return the function value.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Returns
    -------
    y : scalar or array_like
        The result, of length R or length M or M by R.

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.

    Examples
    --------
    We can interpolate 2D observed data using pchip interpolation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import pchip_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = pchip_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="pchip interpolation")
    >>> plt.legend()
    >>> plt.show()

    """
    P = PchipInterpolator(xi, yi, axis=axis)

    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(der)(x)
    else:
        return [P.derivative(nu)(x) for nu in der]


class Akima1DInterpolator(CubicHermiteSpline):
    r"""
    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural.

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (..., npoints, ...)
        N-D array of real values. The length of ``y`` along the interpolation axis
        must be equal to the length of ``x``. Use the ``axis`` parameter to
        select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    method : {'akima', 'makima'}, optional
        If ``"makima"``, use the modified Akima interpolation [2]_.
        Defaults to ``"akima"``, use the Akima interpolation [1]_.

        .. versionadded:: 1.13.0

    extrapolate : {bool, None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points 
        based on first and last intervals, or to return NaNs. If None, 
        ``extrapolate`` is set to False.
        
    Methods
    -------
    __call__
    derivative
    antiderivative
    roots

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    .. versionadded:: 0.14

    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    Let :math:`\delta_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)` be the slopes of
    the interval :math:`\left[x_i, x_{i+1}\right)`. Akima's derivative at
    :math:`x_i` is defined as:

    .. math::

        d_i = \frac{w_1}{w_1 + w_2}\delta_{i-1} + \frac{w_2}{w_1 + w_2}\delta_i

    In the Akima interpolation [1]_ (``method="akima"``), the weights are:

    .. math::

        \begin{aligned}
        w_1 &= |\delta_{i+1} - \delta_i| \\
        w_2 &= |\delta_{i-1} - \delta_{i-2}|
        \end{aligned}

    In the modified Akima interpolation [2]_ (``method="makima"``),
    to eliminate overshoot and avoid edge cases of both numerator and
    denominator being equal to 0, the weights are modified as follows:

    .. math::

        \begin{align*}
        w_1 &= |\delta_{i+1} - \delta_i| + |\delta_{i+1} + \delta_i| / 2 \\
        w_2 &= |\delta_{i-1} - \delta_{i-2}| + |\delta_{i-1} + \delta_{i-2}| / 2
        \end{align*}

    Examples
    --------
    Comparison of ``method="akima"`` and ``method="makima"``:

    >>> import numpy as np
    >>> from scipy.interpolate import Akima1DInterpolator
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(1, 7, 7)
    >>> y = np.array([-1, -1, -1, 0, 1, 1, 1])
    >>> xs = np.linspace(min(x), max(x), num=100)
    >>> y_akima = Akima1DInterpolator(x, y, method="akima")(xs)
    >>> y_makima = Akima1DInterpolator(x, y, method="makima")(xs)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, "o", label="data")
    >>> ax.plot(xs, y_akima, label="akima")
    >>> ax.plot(xs, y_makima, label="makima")
    >>> ax.legend()
    >>> fig.show()

    The overshoot that occurred in ``"akima"`` has been avoided in ``"makima"``.

    References
    ----------
    .. [1] A new method of interpolation and smooth curve fitting based
           on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
           589-602. :doi:`10.1145/321607.321609`
    .. [2] Makima Piecewise Cubic Interpolation. Cleve Moler and Cosmin Ionita, 2019.
           https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/

    """

    def __init__(self, x, y, axis=0, *, method: Literal["akima", "makima"]="akima", 
                 extrapolate:bool | None = None):
        if method not in {"akima", "makima"}:
            raise NotImplementedError(f"`method`={method} is unsupported.")
        # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
        # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
        x, dx, y, axis, _ = prepare_input(x, y, axis)

        if np.iscomplexobj(y):
            msg = ("`Akima1DInterpolator` only works with real values for `y`. "
                   "If you are trying to use the real components of the passed array, "
                   "use `np.real` on the array before passing to "
                   "`Akima1DInterpolator`.")
            raise ValueError(msg)

        # Akima extrapolation historically False; parent class defaults to True.
        extrapolate = False if extrapolate is None else extrapolate

        # determine slopes between breakpoints
        m = np.empty((x.size + 3, ) + y.shape[1:])
        dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
        m[2:-2] = np.diff(y, axis=0) / dx

        # add two additional points on the left ...
        m[1] = 2. * m[2] - m[3]
        m[0] = 2. * m[1] - m[2]
        # ... and on the right
        m[-2] = 2. * m[-3] - m[-4]
        m[-1] = 2. * m[-2] - m[-3]

        # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
        # defined. This is the fill value:
        t = .5 * (m[3:] + m[:-3])
        # get the denominator of the slope t
        dm = np.abs(np.diff(m, axis=0))
        if method == "makima":
            pm = np.abs(m[1:] + m[:-1])
            f1 = dm[2:] + 0.5 * pm[2:]
            f2 = dm[:-2] + 0.5 * pm[:-2]
        else:
            f1 = dm[2:]
            f2 = dm[:-2]
        f12 = f1 + f2
        # These are the mask of where the slope at breakpoint is defined:
        ind = np.nonzero(f12 > 1e-9 * np.max(f12, initial=-np.inf))
        x_ind, y_ind = ind[0], ind[1:]
        # Set the slope at breakpoint
        t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
                  f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]

        super().__init__(x, y, t, axis=0, extrapolate=extrapolate)
        self.axis = axis

    def extend(self, c, x, right=True):
        raise NotImplementedError("Extending a 1-D Akima interpolator is not "
                                  "yet implemented")

    # These are inherited from PPoly, but they do not produce an Akima
    # interpolator. Hence stub them out.
    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")


class CubicSpline(CubicHermiteSpline):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple ``(order, deriv_values)`` allowing to specify arbitrary
        derivatives at curve ends:

        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array_like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1-D, then `deriv_value` must be a scalar. If `y` is 3-D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2-D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    Parameters `bc_type` and ``extrapolate`` work independently, i.e. the
    former controls only construction of a spline, and the latter only
    evaluation.

    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.

    When 'not-a-knot' boundary conditions is applied to both ends, the
    resulting spline will be the same as returned by `splrep` (with ``s=0``)
    and `InterpolatedUnivariateSpline`, but these two methods use a
    representation in B-spline basis.

    .. versionadded:: 0.18.0

    Examples
    --------
    In this example the cubic spline is used to interpolate a sampled sinusoid.
    You can see that the spline continuity property holds for the first and
    second derivatives and violates only for the third derivative.

    >>> import numpy as np
    >>> from scipy.interpolate import CubicSpline
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> cs = CubicSpline(x, y)
    >>> xs = np.arange(-0.5, 9.6, 0.1)
    >>> fig, ax = plt.subplots(figsize=(6.5, 4))
    >>> ax.plot(x, y, 'o', label='data')
    >>> ax.plot(xs, np.sin(xs), label='true')
    >>> ax.plot(xs, cs(xs), label="S")
    >>> ax.plot(xs, cs(xs, 1), label="S'")
    >>> ax.plot(xs, cs(xs, 2), label="S''")
    >>> ax.plot(xs, cs(xs, 3), label="S'''")
    >>> ax.set_xlim(-0.5, 9.5)
    >>> ax.legend(loc='lower left', ncol=2)
    >>> plt.show()

    In the second example, the unit circle is interpolated with a spline. A
    periodic boundary condition is used. You can see that the first derivative
    values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly
    computed. Note that a circle cannot be exactly represented by a cubic
    spline. To increase precision, more breakpoints would be required.

    >>> theta = 2 * np.pi * np.linspace(0, 1, 5)
    >>> y = np.c_[np.cos(theta), np.sin(theta)]
    >>> cs = CubicSpline(theta, y, bc_type='periodic')
    >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
    ds/dx=0.0 ds/dy=1.0
    >>> xs = 2 * np.pi * np.linspace(0, 1, 100)
    >>> fig, ax = plt.subplots(figsize=(6.5, 4))
    >>> ax.plot(y[:, 0], y[:, 1], 'o', label='data')
    >>> ax.plot(np.cos(xs), np.sin(xs), label='true')
    >>> ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    >>> ax.axes.set_aspect('equal')
    >>> ax.legend(loc='center')
    >>> plt.show()

    The third example is the interpolation of a polynomial y = x**3 on the
    interval 0 <= x<= 1. A cubic spline can represent this function exactly.
    To achieve that we need to specify values and first derivatives at
    endpoints of the interval. Note that y' = 3 * x**2 and thus y'(0) = 0 and
    y'(1) = 3.

    >>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
    >>> x = np.linspace(0, 1)
    >>> np.allclose(x**3, cs(x))
    True

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """

    def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        n = len(x)

        bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)

        if extrapolate is None:
            if bc[0] == 'periodic':
                extrapolate = 'periodic'
            else:
                extrapolate = True

        if y.size == 0:
            # bail out early for zero-sized arrays
            s = np.zeros_like(y)
        else:
            dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
            slope = np.diff(y, axis=0) / dxr

            # If bc is 'not-a-knot' this change is just a convention.
            # If bc is 'periodic' then we already checked that y[0] == y[-1],
            # and the spline is just a constant, we handle this case in the
            # same way by setting the first derivatives to slope, which is 0.
            if n == 2:
                if bc[0] in ['not-a-knot', 'periodic']:
                    bc[0] = (1, slope[0])
                if bc[1] in ['not-a-knot', 'periodic']:
                    bc[1] = (1, slope[0])

            # This is a special case, when both conditions are 'not-a-knot'
            # and n == 3. In this case 'not-a-knot' can't be handled regularly
            # as the both conditions are identical. We handle this case by
            # constructing a parabola passing through given points.
            if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
                A = np.zeros((3, 3))  # This is a standard matrix.
                b = np.empty((3,) + y.shape[1:], dtype=y.dtype)

                A[0, 0] = 1
                A[0, 1] = 1
                A[1, 0] = dx[1]
                A[1, 1] = 2 * (dx[0] + dx[1])
                A[1, 2] = dx[0]
                A[2, 1] = 1
                A[2, 2] = 1

                b[0] = 2 * slope[0]
                b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
                b[2] = 2 * slope[1]

                s = solve(A, b, overwrite_a=True, overwrite_b=True,
                          check_finite=False)
            elif n == 3 and bc[0] == 'periodic':
                # In case when number of points is 3 we compute the derivatives
                # manually
                t = (slope / dxr).sum(0) / (1. / dxr).sum(0)
                s = np.broadcast_to(t, (n,) + y.shape[1:])
            else:
                # Find derivative values at each x[i] by solving a tridiagonal
                # system.
                A = np.zeros((3, n))  # This is a banded matrix representation.
                b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

                # Filling the system for i=1..n-2
                #                         (x[i-1] - x[i]) * s[i-1] +\
                # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
                #                         (x[i] - x[i-1]) * s[i+1] =\
                #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
                #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

                A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
                A[0, 2:] = dx[:-1]                   # The upper diagonal
                A[-1, :-2] = dx[1:]                  # The lower diagonal

                b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

                bc_start, bc_end = bc

                if bc_start == 'periodic':
                    # Due to the periodicity, and because y[-1] = y[0], the
                    # linear system has (n-1) unknowns/equations instead of n:
                    A = A[:, 0:-1]
                    A[1, 0] = 2 * (dx[-1] + dx[0])
                    A[0, 1] = dx[-1]

                    b = b[:-1]

                    # Also, due to the periodicity, the system is not tri-diagonal.
                    # We need to compute a "condensed" matrix of shape (n-2, n-2).
                    # See https://web.archive.org/web/20151220180652/http://www.cfm.brown.edu/people/gk/chap6/node14.html
                    # for more explanations.
                    # The condensed matrix is obtained by removing the last column
                    # and last row of the (n-1, n-1) system matrix. The removed
                    # values are saved in scalar variables with the (n-1, n-1)
                    # system matrix indices forming their names:
                    a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
                    a_m1_m2 = dx[-1]
                    a_m1_m1 = 2 * (dx[-1] + dx[-2])
                    a_m2_m1 = dx[-3]
                    a_0_m1 = dx[0]

                    b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
                    b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

                    Ac = A[:, :-1]
                    b1 = b[:-1]
                    b2 = np.zeros_like(b1)
                    b2[0] = -a_0_m1
                    b2[-1] = -a_m2_m1

                    # s1 and s2 are the solutions of (n-2, n-2) system
                    s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
                                      overwrite_b=False, check_finite=False)

                    s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
                                      overwrite_b=False, check_finite=False)

                    # computing the s[n-2] solution:
                    s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
                            (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))

                    # s is the solution of the (n, n) system:
                    s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                    s[:-2] = s1 + s_m1 * s2
                    s[-2] = s_m1
                    s[-1] = s[0]
                else:
                    if bc_start == 'not-a-knot':
                        A[1, 0] = dx[1]
                        A[0, 1] = x[2] - x[0]
                        d = x[2] - x[0]
                        b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                                dxr[0]**2 * slope[1]) / d
                    elif bc_start[0] == 1:
                        A[1, 0] = 1
                        A[0, 1] = 0
                        b[0] = bc_start[1]
                    elif bc_start[0] == 2:
                        A[1, 0] = 2 * dx[0]
                        A[0, 1] = dx[0]
                        b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])

                    if bc_end == 'not-a-knot':
                        A[1, -1] = dx[-2]
                        A[-1, -2] = x[-1] - x[-3]
                        d = x[-1] - x[-3]
                        b[-1] = ((dxr[-1]**2*slope[-2] +
                                 (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
                    elif bc_end[0] == 1:
                        A[1, -1] = 1
                        A[-1, -2] = 0
                        b[-1] = bc_end[1]
                    elif bc_end[0] == 2:
                        A[1, -1] = 2 * dx[-1]
                        A[-1, -2] = dx[-1]
                        b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])

                    s = solve_banded((1, 1), A, b, overwrite_ab=True,
                                     overwrite_b=True, check_finite=False)

        super().__init__(x, y, s, axis=0, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _validate_bc(bc_type, y, expected_deriv_shape, axis):
        """Validate and prepare boundary conditions.

        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
        if isinstance(bc_type, str):
            if bc_type == 'periodic':
                if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                    raise ValueError(
                        f"The first and last `y` point along axis {axis} must "
                        "be identical (within machine precision) when "
                        "bc_type='periodic'.")

            bc_type = (bc_type, bc_type)

        else:
            if len(bc_type) != 2:
                raise ValueError("`bc_type` must contain 2 elements to "
                                 "specify start and end conditions.")

            if 'periodic' in bc_type:
                raise ValueError("'periodic' `bc_type` is defined for both "
                                 "curve ends and cannot be used with other "
                                 "boundary conditions.")

        validated_bc = []
        for bc in bc_type:
            if isinstance(bc, str):
                if bc == 'clamped':
                    validated_bc.append((1, np.zeros(expected_deriv_shape)))
                elif bc == 'natural':
