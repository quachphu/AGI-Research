       "Two Efficient Algorithms with Guaranteed Convergence for Finding a Zero
       of a Function", ACM Transactions on Mathematical Software, Vol. 1, Issue
       4, Dec. 1975, pp. 330-345. Section 3: "Algorithm M".
       :doi:`10.1145/355656.355659`

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brenth(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brenth(f, 0, 2)
    >>> root
    1.0

    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError(f"xtol too small ({xtol:g} <= 0)")
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "brenth")


################################
# TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions", by
#  Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
#  See [1]


def _notclose(fs, rtol=_rtol, atol=_xtol):
    # Ensure not None, not 0, all finite, and not very close to each other
    notclosefvals = (
            all(fs) and all(np.isfinite(fs)) and
            not any(any(np.isclose(_f, fs[i + 1:], rtol=rtol, atol=atol))
                    for i, _f in enumerate(fs[:-1])))
    return notclosefvals


def _secant(xvals, fvals):
    """Perform a secant step, taking a little care"""
    # Secant has many "mathematically" equivalent formulations
    # x2 = x0 - (x1 - x0)/(f1 - f0) * f0
    #    = x1 - (x1 - x0)/(f1 - f0) * f1
    #    = (-x1 * f0 + x0 * f1) / (f1 - f0)
    #    = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    #    = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    x0, x1 = xvals[:2]
    f0, f1 = fvals[:2]
    if f0 == f1:
        return np.nan
    if np.abs(f1) > np.abs(f0):
        x2 = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    else:
        x2 = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    return x2


def _update_bracket(ab, fab, c, fc):
    """Update a bracket given (c, fc), return the discarded endpoints."""
    fa, fb = fab
    idx = (0 if np.sign(fa) * np.sign(fc) > 0 else 1)
    rx, rfx = ab[idx], fab[idx]
    fab[idx] = fc
    ab[idx] = c
    return rx, rfx


def _compute_divided_differences(xvals, fvals, N=None, full=True,
                                 forward=True):
    """Return a matrix of divided differences for the xvals, fvals pairs

    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i

    If full is False, just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = (np.diff(DD[i - 1:, i - 1]) /
                         (xvals[i:] - xvals[:M - i]))
        return DD

    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = (0 if forward else -1)
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i:i + len(row) - 1] - xvals[:len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd


def _interpolated_poly(xvals, fvals, x):
    """Compute p(x) for the polynomial passing through the specified locations.

    Use Neville's algorithm to compute p(x) where p is the minimal degree
    polynomial passing through the points xvals, fvals"""
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    for k in range(1, N):
        alpha = D[k:, k - 1] - Q[k - 1:N - 1, k - 1]
        diffik = xvals[0:N - k] - xvals[k:N]
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        D[k:, k] = (xvals[:N - k] - x) / diffik * alpha
    # Expect Q[-1, 1:] to be small relative to Q[-1, 0] as x approaches a root
    return np.sum(Q[-1, 1:]) + Q[-1, 0]


def _inverse_poly_zero(a, b, c, d, fa, fb, fc, fd):
    """Inverse cubic interpolation f-values -> x-values

    Given four points (fa, a), (fb, b), (fc, c), (fd, d) with
    fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points
    and compute x=IP(0).
    """
    return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)


def _newton_quadratic(ab, fab, d, fd, k):
    """Apply Newton-Raphson like steps, using divided differences to approximate f'

    ab is a real interval [a, b] containing a root,
    fab holds the real values of f(a), f(b)
    d is a real number outside [ab, b]
    k is the number of steps to apply
    """
    a, b = ab
    fa, fb = fab
    _, B, A = _compute_divided_differences([a, b, d], [fa, fb, fd],
                                           forward=True, full=False)

    # _P  is the quadratic polynomial through the 3 points
    def _P(x):
        # Horner evaluation of fa + B * (x - a) + A * (x - a) * (x - b)
        return (A * (x - b) + B) * (x - a) + fa

    if A == 0:
        r = a - fa / B
    else:
        r = (a if np.sign(A) * np.sign(fa) > 0 else b)
        # Apply k Newton-Raphson steps to _P(x), starting from x=r
        for i in range(k):
            r1 = r - _P(r) / (B + A * (2 * r - a - b))
            if not (ab[0] < r1 < ab[1]):
                if (ab[0] < r < ab[1]):
                    return r
                r = sum(ab) / 2.0
                break
            r = r1

    return r


class TOMS748Solver:
    """Solve f(x, *args) == 0 using Algorithm748 of Alefeld, Potro & Shi.
    """
    _MU = 0.5
    _K_MIN = 1
    _K_MAX = 100  # A very high value for real usage. Expect 1, 2, maybe 3.

    def __init__(self):
        self.f = None
        self.args = None
        self.function_calls = 0
        self.iterations = 0
        self.k = 2
        # ab=[a,b] is a global interval containing a root
        self.ab = [np.nan, np.nan]
        # fab is function values at a, b
        self.fab = [np.nan, np.nan]
        self.d = None
        self.fd = None
        self.e = None
        self.fe = None
        self.disp = False
        self.xtol = _xtol
        self.rtol = _rtol
        self.maxiter = _iter

    def configure(self, xtol, rtol, maxiter, disp, k):
        self.disp = disp
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter
        # Silently replace a low value of k with 1
        self.k = max(k, self._K_MIN)
        # Noisily replace a high value of k with self._K_MAX
        if self.k > self._K_MAX:
            msg = "toms748: Overriding k: ->%d" % self._K_MAX
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            self.k = self._K_MAX

    def _callf(self, x, error=True):
        """Call the user-supplied function, update book-keeping"""
        fx = self.f(x, *self.args)
        self.function_calls += 1
        if not np.isfinite(fx) and error:
            raise ValueError(f"Invalid function value: f({x:f}) -> {fx} ")
        return fx

    def get_result(self, x, flag=_ECONVERGED):
        r"""Package the result and statistics into a tuple."""
        return (x, self.function_calls, self.iterations, flag)

    def _update_bracket(self, c, fc):
        return _update_bracket(self.ab, self.fab, c, fc)

    def start(self, f, a, b, args=()):
        r"""Prepare for the iterations."""
        self.function_calls = 0
        self.iterations = 0

        self.f = f
        self.args = args
        self.ab[:] = [a, b]
        if not np.isfinite(a) or np.imag(a) != 0:
            raise ValueError(f"Invalid x value: {a} ")
        if not np.isfinite(b) or np.imag(b) != 0:
            raise ValueError(f"Invalid x value: {b} ")

        fa = self._callf(a)
        if not np.isfinite(fa) or np.imag(fa) != 0:
            raise ValueError(f"Invalid function value: f({a:f}) -> {fa} ")
        if fa == 0:
            return _ECONVERGED, a
        fb = self._callf(b)
        if not np.isfinite(fb) or np.imag(fb) != 0:
            raise ValueError(f"Invalid function value: f({b:f}) -> {fb} ")
        if fb == 0:
            return _ECONVERGED, b

        if np.sign(fb) * np.sign(fa) > 0:
            raise ValueError("f(a) and f(b) must have different signs, but "
                             f"f({a:e})={fa:e}, f({b:e})={fb:e} ")
        self.fab[:] = [fa, fb]

        return _EINPROGRESS, sum(self.ab) / 2.0

    def get_status(self):
        """Determine the current status."""
        a, b = self.ab[:2]
        if np.isclose(a, b, rtol=self.rtol, atol=self.xtol):
            return _ECONVERGED, sum(self.ab) / 2.0
        if self.iterations >= self.maxiter:
            return _ECONVERR, sum(self.ab) / 2.0
        return _EINPROGRESS, sum(self.ab) / 2.0

    def iterate(self):
        """Perform one step in the algorithm.

        Implements Algorithm 4.1(k=1) or 4.2(k=2) in [APS1995]
        """
        self.iterations += 1
        eps = np.finfo(float).eps
        d, fd, e, fe = self.d, self.fd, self.e, self.fe
        ab_width = self.ab[1] - self.ab[0]  # Need the start width below
        c = None

        for nsteps in range(2, self.k+2):
            # If the f-values are sufficiently separated, perform an inverse
            # polynomial interpolation step. Otherwise, nsteps repeats of
            # an approximate Newton-Raphson step.
            if _notclose(self.fab + [fd, fe], rtol=0, atol=32*eps):
                c0 = _inverse_poly_zero(self.ab[0], self.ab[1], d, e,
                                        self.fab[0], self.fab[1], fd, fe)
                if self.ab[0] < c0 < self.ab[1]:
                    c = c0
            if c is None:
                c = _newton_quadratic(self.ab, self.fab, d, fd, nsteps)

            fc = self._callf(c)
            if fc == 0:
                return _ECONVERGED, c

            # re-bracket
            e, fe = d, fd
            d, fd = self._update_bracket(c, fc)

        # u is the endpoint with the smallest f-value
        uix = (0 if np.abs(self.fab[0]) < np.abs(self.fab[1]) else 1)
        u, fu = self.ab[uix], self.fab[uix]

        _, A = _compute_divided_differences(self.ab, self.fab,
                                            forward=(uix == 0), full=False)
        c = u - 2 * fu / A
        if np.abs(c - u) > 0.5 * (self.ab[1] - self.ab[0]):
            c = sum(self.ab) / 2.0
        else:
            if np.isclose(c, u, rtol=eps, atol=0):
                # c didn't change (much).
                # Either because the f-values at the endpoints have vastly
                # differing magnitudes, or because the root is very close to
                # that endpoint
                frs = np.frexp(self.fab)[1]
                if frs[uix] < frs[1 - uix] - 50:  # Differ by more than 2**50
                    c = (31 * self.ab[uix] + self.ab[1 - uix]) / 32
                else:
                    # Make a bigger adjustment, about the
                    # size of the requested tolerance.
                    mm = (1 if uix == 0 else -1)
                    adj = mm * np.abs(c) * self.rtol + mm * self.xtol
                    c = u + adj
                if not self.ab[0] < c < self.ab[1]:
                    c = sum(self.ab) / 2.0

        fc = self._callf(c)
        if fc == 0:
            return _ECONVERGED, c

        e, fe = d, fd
        d, fd = self._update_bracket(c, fc)

        # If the width of the new interval did not decrease enough, bisect
        if self.ab[1] - self.ab[0] > self._MU * ab_width:
            e, fe = d, fd
            z = sum(self.ab) / 2.0
            fz = self._callf(z)
            if fz == 0:
                return _ECONVERGED, z
            d, fd = self._update_bracket(z, fz)

        # Record d and e for next iteration
        self.d, self.fd = d, fd
        self.e, self.fe = e, fe

        status, xn = self.get_status()
        return status, xn

    def solve(self, f, a, b, args=(),
              xtol=_xtol, rtol=_rtol, k=2, maxiter=_iter, disp=True):
        r"""Solve f(x) = 0 given an interval containing a root."""
        self.configure(xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp, k=k)
        status, xn = self.start(f, a, b, args)
        if status == _ECONVERGED:
            return self.get_result(xn)

        # The first step only has two x-values.
        c = _secant(self.ab, self.fab)
        if not self.ab[0] < c < self.ab[1]:
            c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return self.get_result(c)

        self.d, self.fd = self._update_bracket(c, fc)
        self.e, self.fe = None, None
        self.iterations += 1

        while True:
            status, xn = self.iterate()
            if status == _ECONVERGED:
                return self.get_result(xn)
            if status == _ECONVERR:
                fmt = "Failed to converge after %d iterations, bracket is %s"
                if disp:
                    msg = fmt % (self.iterations + 1, self.ab)
                    raise RuntimeError(msg)
                return self.get_result(xn, _ECONVERR)


def toms748(f, a, b, args=(), k=1,
            xtol=_xtol, rtol=_rtol, maxiter=_iter,
            full_output=False, disp=True):
    """
    Find a root using TOMS Algorithm 748 method.

    Implements the Algorithm 748 method of Alefeld, Potro and Shi to find a
    root of the function `f` on the interval ``[a , b]``, where ``f(a)`` and
    `f(b)` must have opposite signs.

    It uses a mixture of inverse cubic interpolation and
    "Newton-quadratic" steps. [APS1995].

    Parameters
    ----------
    f : function
        Python function returning a scalar. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)`
        have opposite signs.
    a : scalar,
        lower boundary of the search interval
    b : scalar,
        upper boundary of the search interval
    args : tuple, optional
        containing extra arguments for the function `f`.
        `f` is called by ``f(x, *args)``.
    k : int, optional
        The number of Newton quadratic steps to perform each
        iteration. ``k>=1``.
    xtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in the `RootResults`
        return object.

    Returns
    -------
    root : float
        Approximate root of `f`
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, ridder, bisect, newton
    fsolve : find roots in N dimensions.

    Notes
    -----
    `f` must be continuous.
    Algorithm 748 with ``k=2`` is asymptotically the most efficient
    algorithm known for finding roots of a four times continuously
    differentiable function.
    In contrast with Brent's algorithm, which may only decrease the length of
    the enclosing bracket on the last step, Algorithm 748 decreases it each
    iteration with the same asymptotic efficiency as it finds the root.

    For easy statement of efficiency indices, assume that `f` has 4
    continuous deriviatives.
    For ``k=1``, the convergence order is at least 2.7, and with about
    asymptotically 2 function evaluations per iteration, the efficiency
    index is approximately 1.65.
    For ``k=2``, the order is about 4.6 with asymptotically 3 function
    evaluations per iteration, and the efficiency index 1.66.
    For higher values of `k`, the efficiency index approaches
    the kth root of ``(3k-2)``, hence ``k=1`` or ``k=2`` are
    usually appropriate.

    References
    ----------
    .. [APS1995]
       Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
       *Algorithm 748: Enclosing Zeros of Continuous Functions*,
       ACM Trans. Math. Softw. Volume 221(1995)
       doi = {10.1145/210089.210111}

    Examples
    --------
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> from scipy import optimize
    >>> root, results = optimize.toms748(f, 0, 2, full_output=True)
    >>> root
    1.0
    >>> results
          converged: True
               flag: converged
     function_calls: 11
         iterations: 5
               root: 1.0
             method: toms748
    """
    if xtol <= 0:
        raise ValueError(f"xtol too small ({xtol:g} <= 0)")
    if rtol < _rtol / 4:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol/4:g})")
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if not np.isfinite(a):
        raise ValueError(f"a is not finite {a}")
    if not np.isfinite(b):
        raise ValueError(f"b is not finite {b}")
    if a >= b:
        raise ValueError(f"a and b are not an interval [{a}, {b}]")
    if not k >= 1:
        raise ValueError(f"k too small ({k} < 1)")

    if not isinstance(args, tuple):
        args = (args,)
    f = _wrap_nan_raise(f)
    solver = TOMS748Solver()
    result = solver.solve(f, a, b, args=args, k=k, xtol=xtol, rtol=rtol,
                          maxiter=maxiter, disp=disp)
    x, function_calls, iterations, flag = result
    return _results_select(full_output, (x, function_calls, iterations, flag),
                           "toms748")
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = ["line_search"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="linesearch",
                                   private_modules=["_linesearch"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Interior-point method for linear programming

The *interior-point* method uses the primal-dual path following algorithm
outlined in [1]_. This algorithm supports sparse constraint matrices and
is typically faster than the simplex methods, especially for large, sparse
problems. Note, however, that the solution returned may be slightly less
accurate than those of the simplex methods and will not, in general,
correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

References
----------
.. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""
# Author: Matt Haberland

import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
has_umfpack = True
has_cholmod = True
try:
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod  # noqa: F401
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack  # test whether to use factorized  # noqa: F401
except ImportError:
    has_umfpack = False


def _get_solver(M, sparse=False, lstsq=False, sym_pos=True,
                cholesky=True, permc_spec='MMD_AT_PLUS_A'):
    """
    Given solver options, return a handle to the appropriate linear system
    solver.

    Parameters
    ----------
    M : 2-D array
        As defined in [4] Equation 8.31
    sparse : bool (default = False)
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool (default = False)
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool (default = True)
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool (default = True)
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        Sparsity preservation strategy used by SuperLU. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        See SuperLU documentation.

    Returns
    -------
    solve : function
        Handle to the appropriate solver function

    """
    try:
        if sparse:
            if lstsq:
                def solve(r, sym_pos=False):
                    return sps.linalg.lsqr(M, r)[0]
            elif cholesky:
                try:
                    # Will raise an exception in the first call,
                    # or when the matrix changes due to a new problem
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                except Exception:
                    _get_solver.cholmod_factor = cholmod_analyze(M)
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                solve = _get_solver.cholmod_factor
            else:
                if has_umfpack and sym_pos:
                    solve = sps.linalg.factorized(M)
                else:  # factorized doesn't pass permc_spec
                    solve = sps.linalg.splu(M, permc_spec=permc_spec).solve

        else:
            if lstsq:  # sometimes necessary as solution is approached
                def solve(r):
                    return sp.linalg.lstsq(M, r)[0]
            elif cholesky:
                L = sp.linalg.cho_factor(M)

                def solve(r):
                    return sp.linalg.cho_solve(L, r)
            else:
                # this seems to cache the matrix factorization, so solving
                # with multiple right hand sides is much faster
                def solve(r, sym_pos=sym_pos):
                    if sym_pos:
                        return sp.linalg.solve(M, r, assume_a="pos")
                    else:
                        return sp.linalg.solve(M, r)
    # There are many things that can go wrong here, and it's hard to say
    # what all of them are. It doesn't really matter: if the matrix can't be
    # factorized, return None. get_solver will be called again with different
    # inputs, and a new routine will try to factorize the matrix.
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    return solve


def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse=False,
               lstsq=False, sym_pos=True, cholesky=True, pc=True, ip=False,
               permc_spec='MMD_AT_PLUS_A'):
    """
    Given standard form problem defined by ``A``, ``b``, and ``c``;
    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;
    algorithmic parameters ``gamma and ``eta;
    and options ``sparse``, ``lstsq``, ``sym_pos``, ``cholesky``, ``pc``
    (predictor-corrector), and ``ip`` (initial point improvement),
    get the search direction for increments to the variable estimates.

    Parameters
    ----------
    As defined in [4], except:
    sparse : bool
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    pc : bool
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial. Even though it requires
        the solution of an additional linear system, the factorization
        is typically (implicitly) reused so solution is efficient, and the
        number of algorithm iterations is typically reduced.
    ip : bool
        True if the improved initial point suggestion due to [4] section 4.3
        is desired. It's unclear whether this is beneficial.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.

    Returns
    -------
    Search directions as defined in [4]

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    if A.shape[0] == 0:
        # If there are no constraints, some solvers fail (understandably)
        # rather than returning empty solution. This gets the job done.
        sparse, lstsq, sym_pos, cholesky = False, False, True, False
    n_x = len(x)

    # [4] Equation 8.8
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [4] Equation 8.31
    Dinv = x / z

    if sparse:
        M = A.dot(sps.diags(Dinv, 0, format="csc").dot(A.T))
    else:
        M = A.dot(Dinv.reshape(-1, 1) * A.T)
    solve = _get_solver(M, sparse, lstsq, sym_pos, cholesky, permc_spec)

    # pc: "predictor-corrector" [4] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    n_corrections = 1 if pc else 0

    i = 0
    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    while i <= n_corrections:
        # Reference [4] Eq. 8.6
        rhatp = eta(gamma) * r_P
        rhatd = eta(gamma) * r_D
        rhatg = eta(gamma) * r_G

        # Reference [4] Eq. 8.7
        rhatxs = gamma * mu - x * z
        rhattk = gamma * mu - tau * kappa

        if i == 1:
            if ip:  # if the correction is to get "initial point"
                # Reference [4] Eq. 8.23
                rhatxs = ((1 - alpha) * gamma * mu -
                          x * z - alpha**2 * d_x * d_z)
                rhattk = ((1 - alpha) * gamma * mu -
                    tau * kappa -
                    alpha**2 * d_tau * d_kappa)
            else:  # if the correction is for "predictor-corrector"
                # Reference [4] Eq. 8.13
                rhatxs -= d_x * d_z
                rhattk -= d_tau * d_kappa

        # sometimes numerical difficulties arise as the solution is approached
        # this loop tries to solve the equations using a sequence of functions
        # for solve. For dense systems, the order is:
        # 1. scipy.linalg.cho_factor/scipy.linalg.cho_solve,
        # 2. scipy.linalg.solve w/ sym_pos = True,
        # 3. scipy.linalg.solve w/ sym_pos = False, and if all else fails
        # 4. scipy.linalg.lstsq
        # For sparse systems, the order is:
        # 1. sksparse.cholmod.cholesky (if available)
        # 2. scipy.sparse.linalg.factorized (if umfpack available)
        # 3. scipy.sparse.linalg.splu
        # 4. scipy.sparse.linalg.lsqr
        solved = False
        while not solved:
            try:
                # [4] Equation 8.28
                p, q = _sym_solve(Dinv, A, c, b, solve)
                # [4] Equation 8.29
                u, v = _sym_solve(Dinv, A, rhatd -
                                  (1 / x) * rhatxs, rhatp, solve)
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    raise LinAlgError
                solved = True
            except (LinAlgError, ValueError, TypeError) as e:
                # Usually this doesn't happen. If it does, it happens when
                # there are redundant constraints or when approaching the
                # solution. If so, change solver.
                if cholesky:
                    cholesky = False
                    warn(
                        "Solving system with option 'cholesky':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'cholesky' to False.",
                        OptimizeWarning, stacklevel=5)
                elif sym_pos:
                    sym_pos = False
                    warn(
                        "Solving system with option 'sym_pos':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'sym_pos' to False.",
                        OptimizeWarning, stacklevel=5)
                elif not lstsq:
                    lstsq = True
                    warn(
                        "Solving system with option 'sym_pos':False "
                        "failed. This may happen occasionally, "
                        "especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "your problem may be numerically challenging. "
                        "If you cannot improve the formulation, consider "
                        "setting 'lstsq' to True. Consider also setting "
                        "`presolve` to True, if it is not already.",
                        OptimizeWarning, stacklevel=5)
                else:
                    raise e
                solve = _get_solver(M, sparse, lstsq, sym_pos,
                                    cholesky, permc_spec)
        # [4] Results after 8.29
        d_tau = ((rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) /
                 (1 / tau * kappa + (-c.dot(p) + b.dot(q))))
        d_x = u + p * d_tau
        d_y = v + q * d_tau

        # [4] Relations between  after 8.25 and 8.26
        d_z = (1 / x) * (rhatxs - z * d_x)
        d_kappa = 1 / tau * (rhattk - kappa * d_tau)

        # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
        if ip:  # initial point - see [4] 4.4
            gamma = 10
        else:  # predictor-corrector, [4] definition after 8.12
            beta1 = 0.1  # [4] pg. 220 (Table 8.1)
            gamma = (1 - alpha)**2 * min(beta1, (1 - alpha))
        i += 1

    return d_x, d_y, d_z, d_tau, d_kappa


def _sym_solve(Dinv, A, r1, r2, solve):
    """
    An implementation of [4] equation 8.31 and 8.32

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [4] 8.31
    r = r2 + A.dot(Dinv * r1)
    v = solve(r)
    # [4] 8.32
    u = Dinv * (A.T.dot(v) - r1)
    return u, v


def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    """
    An implementation of [4] equation 8.21

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha


def _get_message(status):
    """
    Given problem status code, return a more detailed message.

    Parameters
    ----------
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    Returns
    -------
    message : str
        A string descriptor of the exit status of the optimization.

    """
    messages = (
        ["Optimization terminated successfully.",
         "The iteration limit was reached before the algorithm converged.",
         "The algorithm terminated successfully and determined that the "
         "problem is infeasible.",
         "The algorithm terminated successfully and determined that the "
         "problem is unbounded.",
         "Numerical difficulties were encountered before the problem "
         "converged. Please check your problem formulation for errors, "
         "independence of linear equality constraints, and reasonable "
         "scaling and matrix condition numbers. If you continue to "
         "encounter this error, please submit a bug report."
         ])
    return messages[status]


def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    """
    An implementation of [4] Equation 8.9

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa


def _get_blind_start(shape):
    """
    Return the starting point from [4] 4.4

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    m, n = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, z, kappa):
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)

    obj = c.dot(x / tau) + c0

    def norm(a):
        return np.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj


def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    """
    Print indicators of optimization status to the console.

    Parameters
    ----------
    rho_p : float
        The (normalized) primal feasibility, see [4] 4.5
    rho_d : float
        The (normalized) dual feasibility, see [4] 4.5
    rho_g : float
        The (normalized) duality gap, see [4] 4.5
    alpha : float
        The step size, see [4] 4.3
    rho_mu : float
        The (normalized) path parameter, see [4] 4.5
    obj : float
        The objective function value of the current iterate
    header : bool
        True if a header is to be printed

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    if header:
        print("Primal Feasibility ",
              "Dual Feasibility   ",
              "Duality Gap        ",
              "Step            ",
              "Path Parameter     ",
              "Objective          ")

    # no clue why this works
    fmt = '{0:<20.13}{1:<20.13}{2:<20.13}{3:<17.13}{4:<20.13}{5:<20.13}'
    print(fmt.format(
        float(rho_p),
        float(rho_d),
        float(rho_g),
        alpha if isinstance(alpha, str) else float(alpha),
        float(rho_mu),
        float(obj)))


def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq,
            sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args):
    r"""
    Solve a linear programming problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    using the interior point method of [4].

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A`` (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    alpha0 : float
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_3`of [4] Table 8.1
    beta : float
        The desired reduction of the path parameter :math:`\mu` (see  [6]_)
    maxiter : int
        The maximum number of iterations of the algorithm.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    tol : float
        Termination tolerance; see [4]_ Section 4.5.
    sparse : bool
        Set to ``True`` if the problem is to be treated as sparse. However,
        the inputs ``A_eq`` and ``A_ub`` should nonetheless be provided as
        (dense) arrays rather than sparse matrices.
    lstsq : bool
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left as ``False`` unless severe
        numerical difficulties are frequently encountered, and a better option
        would be to improve the formulation of the problem.
    sym_pos : bool
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix (almost always).
    cholesky : bool
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for moderate, dense problems
        that are numerically well-behaved.
    pc : bool
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. It's unclear whether this is beneficial.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed. This is always
                1 for the interior-point method because it has only one phase.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Returns
    -------
    x_hat : float
        Solution vector (for standard form problem).
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf

    """

    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    ip = ip if pc else False

    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol  # we might get lucky : )

    if disp:
        _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)
    if callback is not None:
        x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration, 'phase': 1,
                              'complete': False, 'status': 0,
                              'message': "", 'success': False})
        callback(res)

    status = 0
    message = "Optimization terminated successfully."

    if sparse:
        A = sps.csc_matrix(A)

    while go:

        iteration += 1

        if ip:  # initial point
            # [4] Section 4.4
            gamma = 1

            def eta(g):
                return 1
        else:
            # gamma = 0 in predictor step according to [4] 4.1
            # if predictor/corrector is off, use mean of complementarity [6]
            # 5.1 / [4] Below Figure 10-4
            gamma = 0 if pc else beta * np.mean(z * x)
            # [4] Section 4.1

            def eta(g=gamma):
                return 1 - g

        try:
            # Solve [4] 8.6 and 8.7/8.13/8.23
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
                A, b, c, x, y, z, tau, kappa, gamma, eta,
                sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec)

            if ip:  # initial point
                # [4] 4.4
                # Formula after 8.23 takes a full step regardless if this will
                # take it negative
                alpha = 1.0
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y,
                    d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False  # done with initial point
            else:
                # [4] Section 4.3
                alpha = _get_step(x, d_x, z, d_z, tau,
                                  d_tau, kappa, d_kappa, alpha0)
                # [4] Equation 8.9
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        except (LinAlgError, FloatingPointError,
                ValueError, ZeroDivisionError):
            # this can happen when sparse solver is used and presolve
            # is turned off. Also observed ValueError in AppVeyor Python 3.6
            # Win32 build (PR #8676). I've never seen it otherwise.
            status = 4
            message = _get_message(status)
            break

        # [4] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
            A, b, c, c0, x, y, z, tau, kappa)
        go = rho_p > tol or rho_d > tol or rho_A > tol

        if disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)
        if callback is not None:
            x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                                  'con': con, 'nit': iteration, 'phase': 1,
                                  'complete': False, 'status': 0,
                                  'message': "", 'success': False})
            callback(res)

        # [4] 4.5
        inf1 = (rho_p < tol and rho_d < tol and rho_g < tol and tau < tol *
                max(1, kappa))
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            # [4] Lemma 8.4 / Theorem 8.3
            if b.transpose().dot(y) > tol:
                status = 2
            else:  # elif c.T.dot(x) < tol: ? Probably not necessary.
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break

    x_hat = x / tau
    # [4] Statement after Theorem 8.2
    return x_hat, status, message, iteration


def _linprog_ip(c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-8,
                disp=False, alpha0=.99995, beta=0.1, sparse=False, lstsq=False,
                sym_pos=True, cholesky=None, pc=True, ip=False,
                permc_spec='MMD_AT_PLUS_A', **unknown_options):
    r"""
    Minimize a linear objective function subject to linear
    equality and non-negativity constraints using the interior point method
    of [4]_. Linear programming is intended to solve problems
    of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        Callback function to be executed once per iteration.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int (default = 1000)
        The maximum number of iterations of the algorithm.
    tol : float (default = 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    disp : bool (default = False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    alpha0 : float (default = 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_{3}` of [4]_ Table 8.1.
    beta : float (default = 0.1)
        The desired reduction of the path parameter :math:`\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    sparse : bool (default = False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default = False)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default = True)
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : bool (default = True)
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for problems
        that are numerically well-behaved.
    pc : bool (default = True)
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool (default = False)
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``, and no SuiteSparse.)
        A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    x : 1-D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.

    Notes
    -----
    This method implements the algorithm outlined in [4]_ with ideas from [8]_
    and a structure inspired by the simpler methods of [6]_.

    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.

    With default options, the solver used to perform the factorization depends
    on third-party software availability and the conditioning of the problem.

    For dense problems, solvers are tried in the following order:

    1. ``scipy.linalg.cho_factor``

    2. ``scipy.linalg.solve`` with option ``sym_pos=True``

    3. ``scipy.linalg.solve`` with option ``sym_pos=False``

    4. ``scipy.linalg.lstsq``

    For sparse problems:

    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are installed)

    2. ``scipy.sparse.linalg.factorized``
        (if scikit-umfpack and SuiteSparse are installed)

    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)

    4. ``scipy.sparse.linalg.lsqr``

    If the solver fails for any reason, successively more robust (but slower)
    solvers are attempted in the order indicated. Attempting, failing, and
    re-starting factorization can be time consuming, so if the problem is
    numerically challenging, options can be set to  bypass solvers that are
    failing. Setting ``cholesky=False`` skips to solver 2,
    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips
    to solver 4 for both sparse and dense problems.

    Potential improvements for combating issues associated with dense
    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and
    [10]_ Section 4.1-4.2; the latter also discusses the alleviation of
    accuracy issues associated with the substitution approach to free
    variables.

    After calculating the search direction, the maximum possible step size
    that does not activate the non-negativity constraints is calculated, and
    the smaller of this step size and unity is applied (as in [4]_ Section
    4.1.) [4]_ Section 4.3 suggests improvements for choosing the step size.

    The new point is tested according to the termination conditions of [4]_
    Section 4.5. The same tolerance, which can be set using the ``tol`` option,
    is used for all checks. (A potential improvement would be to expose
    the different tolerances to be set independently.) If optimality,
    unboundedness, or infeasibility is detected, the solve procedure
    terminates; otherwise it repeats.

    The expected problem formulation differs between the top level ``linprog``
    module and the method specific solvers. The method specific solvers expect a
    problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Whereas the top level ``linprog`` module expects a problem of form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    The original problem contains equality, upper-bound and variable constraints
    whereas the method specific solver requires equality constraints and
    variable non-negativity.

    ``linprog`` module converts the original problem to standard form by
    converting the simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.


    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point methods
            for large scale linear programming. HEC/Universite de Geneve, 1996.

    """

    _check_unknown_options(unknown_options)

    # These should be warnings, not errors
    if (cholesky or cholesky is None) and sparse and not has_cholmod:
        if cholesky:
            warn("Sparse cholesky is only available with scikit-sparse. "
                 "Setting `cholesky = False`",
                 OptimizeWarning, stacklevel=3)
        cholesky = False

    if sparse and lstsq:
        warn("Option combination 'sparse':True and 'lstsq':True "
             "is not recommended.",
             OptimizeWarning, stacklevel=3)

    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True "
             "and 'cholesky':True; option 'cholesky' has no effect when "
             "'lstsq' is set True.",
             OptimizeWarning, stacklevel=3)

    valid_permc_spec = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD')
    if permc_spec.upper() not in valid_permc_spec:
        warn("Invalid permc_spec option: '" + str(permc_spec) + "'. "
             "Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', "
             "and 'COLAMD'. Reverting to default.",
             OptimizeWarning, stacklevel=3)
        permc_spec = 'MMD_AT_PLUS_A'

    # This can be an error
    if not sym_pos and cholesky:
        raise ValueError(
            "Invalid option combination 'sym_pos':False "
            "and 'cholesky':True: Cholesky decomposition is only possible "
            "for symmetric positive definite matrices.")

    cholesky = cholesky or (cholesky is None and sym_pos and not lstsq)

    x, status, message, iteration = _ip_hsd(A, b, c, c0, alpha0, beta,
                                            maxiter, disp, tol, sparse,
                                            lstsq, sym_pos, cholesky,
                                            pc, ip, permc_spec, callback,
                                            postsolve_args)

    return x, status, message, iteration
<|endoftext|>"""
Spectral Algorithm for Nonlinear Equations
"""
import collections

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options
from ._linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng

class _NoConvergence(Exception):
    pass


def _root_df_sane(func, x0, args=(), ftol=1e-8, fatol=1e-300, maxfev=1000,
                  fnorm=None, callback=None, disp=False, M=10, eta_strategy=None,
                  sigma_eps=1e-10, sigma_0=1.0, line_search='cruz', **unknown_options):
    r"""
    Solve nonlinear equation with the DF-SANE method

    Options
    -------
    ftol : float, optional
        Relative norm tolerance.
    fatol : float, optional
        Absolute norm tolerance.
        Algorithm terminates when ``||func(x)|| < fatol + ftol ||func(x_0)||``.
    fnorm : callable, optional
        Norm to use in the convergence check. If None, 2-norm is used.
    maxfev : int, optional
        Maximum number of function evaluations.
    disp : bool, optional
        Whether to print convergence process to stdout.
    eta_strategy : callable, optional
        Choice of the ``eta_k`` parameter, which gives slack for growth
        of ``||F||**2``.  Called as ``eta_k = eta_strategy(k, x, F)`` with
        `k` the iteration number, `x` the current iterate and `F` the current
        residual. Should satisfy ``eta_k > 0`` and ``sum(eta, k=0..inf) < inf``.
        Default: ``||F||**2 / (1 + k)**2``.
    sigma_eps : float, optional
        The spectral coefficient is constrained to ``sigma_eps < sigma < 1/sigma_eps``.
        Default: 1e-10
    sigma_0 : float, optional
        Initial spectral coefficient.
        Default: 1.0
    M : int, optional
        Number of iterates to include in the nonmonotonic line search.
        Default: 10
    line_search : {'cruz', 'cheng'}
        Type of line search to employ. 'cruz' is the original one defined in
        [Martinez & Raydan. Math. Comp. 75, 1429 (2006)], 'cheng' is
        a modified search defined in [Cheng & Li. IMA J. Numer. Anal. 29, 814 (2009)].
        Default: 'cruz'

    References
    ----------
    .. [1] "Spectral residual method without gradient information for solving
           large-scale nonlinear systems of equations." W. La Cruz,
           J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
    .. [2] W. La Cruz, Opt. Meth. Software, 29, 24 (2014).
    .. [3] W. Cheng, D.-H. Li. IMA J. Numer. Anal. **29**, 814 (2009).

    """
    _check_unknown_options(unknown_options)

    if line_search not in ('cheng', 'cruz'):
        raise ValueError(f"Invalid value {line_search!r} for 'line_search'")

    nexp = 2

    if eta_strategy is None:
        # Different choice from [1], as their eta is not invariant
        # vs. scaling of F.
        def eta_strategy(k, x, F):
            # Obtain squared 2-norm of the initial residual from the outer scope
            return f_0 / (1 + k)**2

    if fnorm is None:
        def fnorm(F):
            # Obtain squared 2-norm of the current residual from the outer scope
            return f_k**(1.0/nexp)

    def fmerit(F):
        return np.linalg.norm(F)**nexp

    nfev = [0]
    f, x_k, x_shape, f_k, F_k, is_complex = _wrap_func(func, x0, fmerit,
                                                       nfev, maxfev, args)

    k = 0
    f_0 = f_k
    sigma_k = sigma_0

    F_0_norm = fnorm(F_k)

    # For the 'cruz' line search
    prev_fs = collections.deque([f_k], M)

    # For the 'cheng' line search
    Q = 1.0
    C = f_0

    converged = False
    message = "too many function evaluations required"

    while True:
        F_k_norm = fnorm(F_k)

        if disp:
            print("iter %d: ||F|| = %g, sigma = %g" % (k, F_k_norm, sigma_k))

        if callback is not None:
            callback(x_k, F_k)

        if F_k_norm < ftol * F_0_norm + fatol:
            # Converged!
            message = "successful convergence"
            converged = True
            break

        # Control spectral parameter, from [2]
        if abs(sigma_k) > 1/sigma_eps:
            sigma_k = 1/sigma_eps * np.sign(sigma_k)
        elif abs(sigma_k) < sigma_eps:
            sigma_k = sigma_eps

        # Line search direction
        d = -sigma_k * F_k

        # Nonmonotone line search
        eta = eta_strategy(k, x_k, F_k)
        try:
            if line_search == 'cruz':
                alpha, xp, fp, Fp = _nonmonotone_line_search_cruz(f, x_k, d, prev_fs,
                                                                  eta=eta)
            elif line_search == 'cheng':
                alpha, xp, fp, Fp, C, Q = _nonmonotone_line_search_cheng(f, x_k, d, f_k,
                                                                         C, Q, eta=eta)
        except _NoConvergence:
            break

        # Update spectral parameter
        s_k = xp - x_k
        y_k = Fp - F_k
        sigma_k = np.vdot(s_k, s_k) / np.vdot(s_k, y_k)

        # Take step
        x_k = xp
        F_k = Fp
        f_k = fp

        # Store function value
        if line_search == 'cruz':
            prev_fs.append(fp)

        k += 1

    x = _wrap_result(x_k, is_complex, shape=x_shape)
    F = _wrap_result(F_k, is_complex)

    result = OptimizeResult(x=x, success=converged,
                            message=message,
                            fun=F, nfev=nfev[0], nit=k, method="df-sane")

    return result


def _wrap_func(func, x0, fmerit, nfev_list, maxfev, args=()):
    """
    Wrap a function and an initial value so that (i) complex values
    are wrapped to reals, and (ii) value for a merit function
    fmerit(x, f) is computed at the same time, (iii) iteration count
    is maintained and an exception is raised if it is exceeded.

    Parameters
    ----------
    func : callable
        Function to wrap
    x0 : ndarray
        Initial value
    fmerit : callable
        Merit function fmerit(f) for computing merit value from residual.
    nfev_list : list
        List to store number of evaluations in. Should be [0] in the beginning.
    maxfev : int
        Maximum number of evaluations before _NoConvergence is raised.
    args : tuple
        Extra arguments to func

    Returns
    -------
    wrap_func : callable
        Wrapped function, to be called as
        ``F, fp = wrap_func(x0)``
    x0_wrap : ndarray of float
        Wrapped initial value; raveled to 1-D and complex
        values mapped to reals.
    x0_shape : tuple
        Shape of the initial value array
    f : float
        Merit function at F
    F : ndarray of float
        Residual at x0_wrap
    is_complex : bool
        Whether complex values were mapped to reals

    """
    x0 = np.asarray(x0)
    x0_shape = x0.shape
    F = np.asarray(func(x0, *args)).ravel()
    is_complex = np.iscomplexobj(x0) or np.iscomplexobj(F)
    x0 = x0.ravel()

    nfev_list[0] = 1

    if is_complex:
        def wrap_func(x):
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            nfev_list[0] += 1
            z = _real2complex(x).reshape(x0_shape)
            v = np.asarray(func(z, *args)).ravel()
            F = _complex2real(v)
            f = fmerit(F)
            return f, F

        x0 = _complex2real(x0)
        F = _complex2real(F)
    else:
        def wrap_func(x):
            if nfev_list[0] >= maxfev:
                raise _NoConvergence()
            nfev_list[0] += 1
            x = x.reshape(x0_shape)
            F = np.asarray(func(x, *args)).ravel()
            f = fmerit(F)
            return f, F

    return wrap_func, x0, x0_shape, fmerit(F), F, is_complex


def _wrap_result(result, is_complex, shape=None):
    """
    Convert from real to complex and reshape result arrays.
    """
    if is_complex:
        z = _real2complex(result)
    else:
        z = result
    if shape is not None:
        z = z.reshape(shape)
    return z


def _real2complex(x):
    return np.ascontiguousarray(x, dtype=float).view(np.complex128)


def _complex2real(z):
    return np.ascontiguousarray(z, dtype=complex).view(np.float64)
<|endoftext|>"""Nearly exact trust-region optimization subproblem."""
import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
                          cho_solve)
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = ['_minimize_trustregion_exact',
           'estimate_smallest_singular_value',
           'singular_leading_submatrix',
           'IterativeSubproblem']


def _minimize_trustregion_exact(fun, x0, args=(), jac=None, hess=None,
                                **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.

    Options
    -------
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region '
                         'exact minimization.')
    if not callable(hess):
        raise ValueError('Hessian matrix is required for trust region '
                         'exact minimization.')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=IterativeSubproblem,
                                  **trust_region_options)


def estimate_smallest_singular_value(U):
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    Parameters
    ----------
    U : ndarray
        Square upper triangular matrix.

    Returns
    -------
    s_min : float
        Estimated smallest singular value of the provided matrix.
    z_min : ndarray
        Estimated right singular vector.

    Notes
    -----
    The procedure is based on [1]_ and is done in two steps. First, it finds
    a vector ``e`` with components selected from {+1, -1} such that the
    solution ``w`` from the system ``U.T w = e`` is as large as possible.
    Next it estimate ``U v = w``. The smallest singular value is close
    to ``norm(w)/norm(v)`` and the right singular vector is close
    to ``v/norm(v)``.

    The estimation will be better more ill-conditioned is the matrix.

    References
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """

    U = np.atleast_2d(U)
    m, n = U.shape

    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    # A vector `e` with components selected from {+1, -1}
    # is selected so that the solution `w` to the system
    # `U.T w = e` is as large as possible. Implementation
    # based on algorithm 3.5.1, p. 142, from reference [2]
    # adapted for lower triangular matrix.

    p = np.zeros(n)
    w = np.empty(n)

    # Implemented according to:  Golub, G. H., Van Loan, C. F. (2013).
    # "Matrix computations". Forth Edition. JHU press. pp. 140-142.
    for k in range(n):
        wp = (1-p[k]) / U.T[k, k]
        wm = (-1-p[k]) / U.T[k, k]
        pp = p[k+1:] + U.T[k+1:, k]*wp
        pm = p[k+1:] + U.T[k+1:, k]*wm

        if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
            w[k] = wp
            p[k+1:] = pp
        else:
            w[k] = wm
            p[k+1:] = pm

    # The system `U v = w` is solved using backward substitution.
    v = solve_triangular(U, w)

    v_norm = norm(v)
    w_norm = norm(w)

    # Smallest singular value
    s_min = w_norm / v_norm

    # Associated vector
    z_min = v / v_norm

    return s_min, z_min


def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    Defined ref. [1].

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """

    H_diag = np.diag(H)
    H_diag_abs = np.abs(H_diag)
    H_row_sums = np.sum(np.abs(H), axis=1)
    lb = np.min(H_diag + H_diag_abs - H_row_sums)
    ub = np.max(H_diag - H_diag_abs + H_row_sums)

    return lb, ub


def singular_leading_submatrix(A, U, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix that is not positive definite.
    U : ndarray
        Upper triangular matrix resulting of an incomplete
        Cholesky decomposition of matrix ``A``.
    k : int
        Positive integer such that the leading k by k submatrix from
        `A` is the first non-positive definite leading submatrix.

    Returns
    -------
    delta : float
        Amount that should be added to the element (k, k) of the
        leading k by k submatrix of ``A`` to make it singular.
    v : ndarray
        A vector such that ``v.T B v = 0``. Where B is the matrix A after
        ``delta`` is added to its element (k, k).
    """

    # Compute delta
    delta = np.sum(U[:k-1, k-1]**2) - A[k-1, k-1]

    n = len(A)

    # Initialize v
    v = np.zeros(n)
    v[k-1] = 1

    # Compute the remaining values of v by solving a triangular system.
    if k != 1:
        v[:k-1] = solve_triangular(U[:k-1, :k-1], -U[:k-1, k-1])

    return delta, v


class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method.

    Notes
    -----
    This subproblem solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.
    """

    # UPDATE_COEFF appears in reference [1]_
    # in formula 7.3.14 (p. 190) named as "theta".
    # As recommended there it value is fixed in 0.01.
    UPDATE_COEFF = 0.01

    EPS = np.finfo(float).eps

    def __init__(self, x, fun, jac, hess, hessp=None,
                 k_easy=0.1, k_hard=0.2):

        super().__init__(x, fun, jac, hess)

        # When the trust-region shrinks in two consecutive
        # calculations (``tr_radius < previous_tr_radius``)
        # the lower bound ``lambda_lb`` may be reused,
        # facilitating  the convergence. To indicate no
        # previous value is known at first ``previous_tr_radius``
        # is set to -1  and ``lambda_lb`` to None.
        self.previous_tr_radius = -1
        self.lambda_lb = None

        self.niter = 0

        # ``k_easy`` and ``k_hard`` are parameters used
        # to determine the stop criteria to the iterative
        # subproblem solver. Take a look at pp. 194-197
        # from reference _[1] for a more detailed description.
        self.k_easy = k_easy
        self.k_hard = k_hard

        # Get Lapack function for cholesky decomposition.
        # The implemented SciPy wrapper does not return
        # the incomplete factorization needed by the method.
        self.cholesky, = get_lapack_funcs(('potrf',), (self.hess,))

        # Get info about Hessian
        self.dimension = len(self.hess)
        self.hess_gershgorin_lb,\
            self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, np.inf)
        self.hess_fro = norm(self.hess, 'fro')

        # A constant such that for vectors smaller than that
        # backward substitution is not reliable. It was established
        # based on Golub, G. H., Van Loan, C. F. (2013).
        # "Matrix computations". Forth Edition. JHU press., p.165.
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """

        # Upper bound for the damping factor
        lambda_ub = max(0, self.jac_mag/tr_radius + min(-self.hess_gershgorin_lb,
                                                        self.hess_fro,
                                                        self.hess_inf))

        # Lower bound for the damping factor
        lambda_lb = max(0, -min(self.hess.diagonal()),
                        self.jac_mag/tr_radius - min(self.hess_gershgorin_ub,
                                                     self.hess_fro,
                                                     self.hess_inf))

        # Improve bounds with previous info
        if tr_radius < self.previous_tr_radius:
            lambda_lb = max(self.lambda_lb, lambda_lb)

        # Initial guess for the damping factor
        if lambda_lb == 0:
            lambda_initial = 0
        else:
            lambda_initial = max(np.sqrt(lambda_lb * lambda_ub),
                                 lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve quadratic subproblem"""

        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        self.niter = 0

        while True:

            # Compute Cholesky factorization
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess+lambda_current*np.eye(n)
                U, info = self.cholesky(H, lower=False,
                                        overwrite_a=False,
                                        clean=True)

            self.niter += 1

            # Check if factorization succeeded
            if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
                # Successful factorization

                # Solve `U.T U p = s`
                p = cho_solve((U, False), -self.jac)

                p_norm = norm(p)

                # Check for interior convergence
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break

                # Solve `U.T w = p`
                w = solve_triangular(U, p, trans='T')

                w_norm = norm(w)

                # Compute Newton step accordingly to
                # formula (4.44) p.87 from ref [2]_.
                delta_lambda = (p_norm/w_norm)**2 * (p_norm-tr_radius)/tr_radius
                lambda_new = lambda_current + delta_lambda

                if p_norm < tr_radius:  # Inside boundary
                    s_min, z_min = estimate_smallest_singular_value(U)

                    ta, tb = self.get_boundaries_intersections(p, z_min,
                                                               tr_radius)

                    # Choose `step_len` with the smallest magnitude.
                    # The reason for this choice is explained at
                    # ref [3]_, p. 6 (Immediately before the formula
                    # for `tau`).
                    step_len = min([ta, tb], key=abs)

                    # Compute the quadratic term  (p.T*H*p)
                    quadratic_term = np.dot(p, np.dot(H, p))

                    # Check stop criteria
                    relative_error = ((step_len**2 * s_min**2)
                                      / (quadratic_term + lambda_current*tr_radius**2))
                    if relative_error <= self.k_hard:
                        p += step_len * z_min
                        break

                    # Update uncertainty bounds
                    lambda_ub = lambda_current
                    lambda_lb = max(lambda_lb, lambda_current - s_min**2)

                    # Compute Cholesky factorization
                    H = self.hess + lambda_new*np.eye(n)
                    c, info = self.cholesky(H, lower=False,
                                            overwrite_a=False,
                                            clean=True)

                    # Check if the factorization have succeeded
                    #
                    if info == 0:  # Successful factorization
                        # Update damping factor
                        lambda_current = lambda_new
                        already_factorized = True
                    else:  # Unsuccessful factorization
                        # Update uncertainty bounds
                        lambda_lb = max(lambda_lb, lambda_new)

                        # Update damping factor
                        lambda_current = max(
                            np.sqrt(lambda_lb * lambda_ub),
                            lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb)
                        )

                else:  # Outside boundary
                    # Check stop criteria
                    relative_error = abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break

                    # Update uncertainty bounds
                    lambda_lb = lambda_current

                    # Update damping factor
                    lambda_current = lambda_new

            elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
                # jac_mag very close to zero

                # Check for interior convergence
                if lambda_current == 0:
                    p = np.zeros(n)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value(U)
                step_len = tr_radius

                # Check stop criteria
                if (step_len**2 * s_min**2
                    <= self.k_hard * lambda_current * tr_radius**2):
                    p = step_len * z_min
                    break

                # Update uncertainty bounds
                lambda_ub = lambda_current
                lambda_lb = max(lambda_lb, lambda_current - s_min**2)

                # Update damping factor
                lambda_current = max(
                    np.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb)
                )

            else:  # Unsuccessful factorization

                # Compute auxiliary terms
                delta, v = singular_leading_submatrix(H, U, info)
                v_norm = norm(v)

                # Update uncertainty interval
                lambda_lb = max(lambda_lb, lambda_current + delta/v_norm**2)

                # Update damping factor
                lambda_current = max(
                    np.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb)
                )

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius

        return p, hits_boundary
<|endoftext|>from scipy.optimize._bracket import _bracket_root, _bracket_minimum
from scipy.optimize._chandrupatla import _chandrupatla, _chandrupatla_minimize
from scipy._lib._util import _RichResult


def find_root(f, init, /, *, args=(), tolerances=None, maxiter=None, callback=None):
    """Find the root of a monotonic, real-valued function of a real variable.

    For each element of the output of `f`, `find_root` seeks the scalar
    root that makes the element 0. This function currently uses Chandrupatla's
    bracketing algorithm [1]_ and therefore requires argument `init` to
    provide a bracket around the root: the function values at the two endpoints
    must have opposite signs.

    Provided a valid bracket, `find_root` is guaranteed to converge to a solution
    that satisfies the provided `tolerances` if the function is continuous within
    the bracket.

    This function works elementwise when `init` and `args` contain (broadcastable)
    arrays.

    Parameters
    ----------
    f : callable
        The function whose root is desired. The signature must be::

            f(x: array, *args) -> array

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.

        `f` must be an elementwise function: each element ``f(x)[i]``
        must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
        array ``x`` or the arrays in ``args``.

        `find_root` seeks an array ``x`` such that ``f(x)`` is an array of zeros.
    init : 2-tuple of float array_like
        The lower and upper endpoints of a bracket surrounding the desired root.
        A bracket is valid if arrays ``xl, xr = init`` satisfy ``xl < xr`` and
        ``sign(f(xl)) == -sign(f(xr))`` elementwise. Arrays be broadcastable with
        one another and `args`.
    args : tuple of array_like, optional
        Additional positional array arguments to be passed to `f`. Arrays
        must be broadcastable with one another and the arrays of `init`.
        If the callable for which the root is desired requires arguments that are
        not broadcastable with `x`, wrap that callable with `f` such that `f`
        accepts only `x` and broadcastable ``*args``.
    tolerances : dictionary of floats, optional
        Absolute and relative tolerances on the root and function value.
        Valid keys of the dictionary are:

        - ``xatol`` - absolute tolerance on the root
        - ``xrtol`` - relative tolerance on the root
        - ``fatol`` - absolute tolerance on the function value
        - ``frtol`` - relative tolerance on the function value

        See Notes for default values and explicit termination conditions.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
        The default is the maximum possible number of bisections within
        the (normal) floating point numbers of the relevant dtype.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `find_root` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `find_root` will return a result. `callback` must not mutate
        `res` or its attributes.

    Returns
    -------
    res : _RichResult
        An object similar to an instance of `scipy.optimize.OptimizeResult` with the
        following attributes. The descriptions are written as though the values will
        be scalars; however, if `f` returns an array, the outputs will be
        arrays of the same shape.

        success : bool array
            ``True`` where the algorithm terminated successfully (status ``0``);
            ``False`` otherwise.
        status : int array
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm converged to the specified tolerances.
            - ``-1`` : The initial bracket was invalid.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).

        x : float array
            The root of the function, if the algorithm terminated successfully.
        f_x : float array
            The value of `f` evaluated at `x`.
        nfev : int array
            The number of abscissae at which `f` was evaluated to find the root.
            This is distinct from the number of times `f` is *called* because the
            the function may evaluated at multiple points in a single call.
        nit : int array
            The number of iterations of the algorithm that were performed.
        bracket : tuple of float arrays
            The lower and upper endpoints of the final bracket.
        f_bracket : tuple of float arrays
            The value of `f` evaluated at the lower and upper endpoints of the
            bracket.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    Let:

    -  ``a, b = init`` be the left and right endpoints of the initial bracket,
    - ``xl`` and ``xr`` be the left and right endpoints of the final bracket,
    - ``xmin = xl if abs(f(xl)) <= abs(f(xr)) else xr`` be the final bracket
      endpoint with the smaller function value, and
    - ``fmin0 = min(f(a), f(b))`` be the minimum of the two values of the
      function evaluated at the initial bracket endpoints.

    Then the algorithm is considered to have converged when

    - ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or
    - ``fun(xmin) <= fatol + abs(fmin0) * frtol``.

    This is equivalent to the termination condition described in [1]_ with
    ``xrtol = 4e-10``, ``xatol = 1e-5``, and ``fatol = frtol = 0``.
    However, the default values of the `tolerances` dictionary are
    ``xatol = 4*tiny``, ``xrtol = 4*eps``, ``frtol = 0``, and ``fatol = tiny``,
    where ``eps`` and ``tiny`` are the precision and smallest normal number
    of the result ``dtype`` of function inputs and outputs.

    References
    ----------

    .. [1] Chandrupatla, Tirupathi R.
        "A new hybrid quadratic/bisection algorithm for finding the zero of a
        nonlinear function without using derivatives".
        Advances in Engineering Software, 28(3), 145-149.
        https://doi.org/10.1016/s0965-9978(96)00051-8

    See Also
    --------
    bracket_root

    Examples
    --------
    Suppose we wish to find the root of the following function.

    >>> def f(x, c=5):
    ...     return x**3 - 2*x - c

    First, we must find a valid bracket. The function is not monotonic,
    but `bracket_root` may be able to provide a bracket.

    >>> from scipy.optimize import elementwise
    >>> res_bracket = elementwise.bracket_root(f, 0)
    >>> res_bracket.success
    True
    >>> res_bracket.bracket
    (2.0, 4.0)

    Indeed, the values of the function at the bracket endpoints have
    opposite signs.

    >>> res_bracket.f_bracket
    (-1.0, 51.0)

    Once we have a valid bracket, `find_root` can be used to provide
    a precise root.

    >>> res_root = elementwise.find_root(f, res_bracket.bracket)
    >>> res_root.x
    2.0945514815423265

    The final bracket is only a few ULPs wide, so the error between
    this value and the true root cannot be much smaller within values
    that are representable in double precision arithmetic.

    >>> import numpy as np
    >>> xl, xr = res_root.bracket
    >>> (xr - xl) / np.spacing(xl)
    2.0
    >>> res_root.f_bracket
    (-8.881784197001252e-16, 9.769962616701378e-15)

    `bracket_root` and `find_root` accept arrays for most arguments.
    For instance, to find the root for a few values of the parameter ``c``
    at once:

    >>> c = np.asarray([3, 4, 5])
    >>> res_bracket = elementwise.bracket_root(f, 0, args=(c,))
    >>> res_bracket.bracket
    (array([1., 1., 2.]), array([2., 2., 4.]))
    >>> res_root = elementwise.find_root(f, res_bracket.bracket, args=(c,))
    >>> res_root.x
    array([1.8932892 , 2.        , 2.09455148])

    """

    def reformat_result(res_in):
        res_out = _RichResult()
        res_out.status = res_in.status
        res_out.success = res_in.success
        res_out.x = res_in.x
        res_out.f_x = res_in.fun
        res_out.nfev = res_in.nfev
        res_out.nit = res_in.nit
        res_out.bracket = (res_in.xl, res_in.xr)
        res_out.f_bracket = (res_in.fl, res_in.fr)
        res_out._order_keys = ['success', 'status', 'x', 'f_x',
                               'nfev', 'nit', 'bracket', 'f_bracket']
        return res_out

    xl, xr = init
    default_tolerances = dict(xatol=None, xrtol=None, fatol=None, frtol=0)
    tolerances = {} if tolerances is None else tolerances
    default_tolerances.update(tolerances)
    tolerances = default_tolerances

    if callable(callback):
        def _callback(res):
            return callback(reformat_result(res))
    else:
        _callback = callback

    res = _chandrupatla(f, xl, xr, args=args, **tolerances,
                        maxiter=maxiter, callback=_callback)
    return reformat_result(res)


def find_minimum(f, init, /, *, args=(), tolerances=None, maxiter=100, callback=None):
    """Find the minimum of an unimodal, real-valued function of a real variable.

    For each element of the output of `f`, `find_minimum` seeks the scalar minimizer
    that minimizes the element. This function currently uses Chandrupatla's
    bracketing minimization algorithm [1]_ and therefore requires argument `init`
    to provide a three-point minimization bracket: ``x1 < x2 < x3`` such that
    ``func(x1) >= func(x2) <= func(x3)``, where one of the inequalities is strict.

    Provided a valid bracket, `find_minimum` is guaranteed to converge to a local
    minimum that satisfies the provided `tolerances` if the function is continuous
    within the bracket.

    This function works elementwise when `init` and `args` contain (broadcastable)
    arrays.

    Parameters
    ----------
    f : callable
        The function whose minimizer is desired. The signature must be::

            f(x: array, *args) -> array

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.

        `f` must be an elementwise function: each element ``f(x)[i]``
        must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
        array ``x`` or the arrays in ``args``.

        `find_minimum` seeks an array ``x`` such that ``f(x)`` is an array of
        local minima.
    init : 3-tuple of float array_like
        The abscissae of a standard scalar minimization bracket. A bracket is
        valid if arrays ``x1, x2, x3 = init`` satisfy ``x1 < x2 < x3`` and
        ``func(x1) >= func(x2) <= func(x3)``, where one of the inequalities
        is strict. Arrays must be broadcastable with one another and the arrays
        of `args`.
    args : tuple of array_like, optional
        Additional positional array arguments to be passed to `f`. Arrays
        must be broadcastable with one another and the arrays of `init`.
        If the callable for which the root is desired requires arguments that are
        not broadcastable with `x`, wrap that callable with `f` such that `f`
        accepts only `x` and broadcastable ``*args``.
    tolerances : dictionary of floats, optional
        Absolute and relative tolerances on the root and function value.
        Valid keys of the dictionary are:

        - ``xatol`` - absolute tolerance on the root
        - ``xrtol`` - relative tolerance on the root
        - ``fatol`` - absolute tolerance on the function value
        - ``frtol`` - relative tolerance on the function value

        See Notes for default values and explicit termination conditions.
    maxiter : int, default: 100
        The maximum number of iterations of the algorithm to perform.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `find_minimum` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `find_root` will return a result. `callback` must not mutate
        `res` or its attributes.

    Returns
    -------
    res : _RichResult
        An object similar to an instance of `scipy.optimize.OptimizeResult` with the
        following attributes. The descriptions are written as though the values will
        be scalars; however, if `f` returns an array, the outputs will be
        arrays of the same shape.

        success : bool array
            ``True`` where the algorithm terminated successfully (status ``0``);
            ``False`` otherwise.
        status : int array
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm converged to the specified tolerances.
            - ``-1`` : The algorithm encountered an invalid bracket.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).

        x : float array
            The minimizer of the function, if the algorithm terminated successfully.
        f_x : float array
            The value of `f` evaluated at `x`.
        nfev : int array
            The number of abscissae at which `f` was evaluated to find the root.
            This is distinct from the number of times `f` is *called* because the
            the function may evaluated at multiple points in a single call.
        nit : int array
            The number of iterations of the algorithm that were performed.
        bracket : tuple of float arrays
            The final three-point bracket.
        f_bracket : tuple of float arrays
            The value of `f` evaluated at the bracket points.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``xl < xm < xr`` are the points of the bracket and ``fl >= fm <= fr``
    (where one of the inequalities is strict) are the values of `f` evaluated
    at those points, then the algorithm is considered to have converged when:

    - ``xr - xl <= abs(xm)*xrtol + xatol`` or
    - ``(fl - 2*fm + fr)/2 <= abs(fm)*frtol + fatol``.

    Note that first of these differs from the termination conditions described
    in [1]_.

    The default value of `xrtol` is the square root of the precision of the
    appropriate dtype, and ``xatol = fatol = frtol`` is the smallest normal
    number of the appropriate dtype.

    References
    ----------

    .. [1] Chandrupatla, Tirupathi R. (1998).
        "An efficient quadratic fit-sectioning algorithm for minimization
        without derivatives".
        Computer Methods in Applied Mechanics and Engineering, 152 (1-2),
        211-217. https://doi.org/10.1016/S0045-7825(97)00190-4

    See Also
    --------
    bracket_minimum

    Examples
    --------
    Suppose we wish to minimize the following function.

    >>> def f(x, c=1):
    ...     return (x - c)**2 + 2

    First, we must find a valid bracket. The function is unimodal,
    so `bracket_minium` will easily find a bracket.

    >>> from scipy.optimize import elementwise
    >>> res_bracket = elementwise.bracket_minimum(f, 0)
    >>> res_bracket.success
    True
    >>> res_bracket.bracket
    (0.0, 0.5, 1.5)

    Indeed, the bracket points are ordered and the function value
    at the middle bracket point is less than at the surrounding
    points.

    >>> xl, xm, xr = res_bracket.bracket
    >>> fl, fm, fr = res_bracket.f_bracket
    >>> (xl < xm < xr) and (fl > fm <= fr)
    True

    Once we have a valid bracket, `find_minimum` can be used to provide
    an estimate of the minimizer.

    >>> res_minimum = elementwise.find_minimum(f, res_bracket.bracket)
    >>> res_minimum.x
    1.0000000149011612

    The function value changes by only a few ULPs within the bracket, so
    the minimizer cannot be determined much more precisely by evaluating
    the function alone (i.e. we would need its derivative to do better).

    >>> import numpy as np
    >>> fl, fm, fr = res_minimum.f_bracket
    >>> (fl - fm) / np.spacing(fm), (fr - fm) / np.spacing(fm)
    (0.0, 2.0)

    Therefore, a precise minimum of the function is given by:

    >>> res_minimum.f_x
    2.0

    `bracket_minimum` and `find_minimum` accept arrays for most arguments.
    For instance, to find the minimizers and minima for a few values of the
    parameter ``c`` at once:

    >>> c = np.asarray([1, 1.5, 2])
    >>> res_bracket = elementwise.bracket_minimum(f, 0, args=(c,))
    >>> res_bracket.bracket
    (array([0. , 0.5, 0.5]), array([0.5, 1.5, 1.5]), array([1.5, 2.5, 2.5]))
    >>> res_minimum = elementwise.find_minimum(f, res_bracket.bracket, args=(c,))
    >>> res_minimum.x
    array([1.00000001, 1.5       , 2.        ])
    >>> res_minimum.f_x
    array([2., 2., 2.])

    """

    def reformat_result(res_in):
        res_out = _RichResult()
        res_out.status = res_in.status
        res_out.success = res_in.success
        res_out.x = res_in.x
        res_out.f_x = res_in.fun
        res_out.nfev = res_in.nfev
        res_out.nit = res_in.nit
        res_out.bracket = (res_in.xl, res_in.xm, res_in.xr)
        res_out.f_bracket = (res_in.fl, res_in.fm, res_in.fr)
        res_out._order_keys = ['success', 'status', 'x', 'f_x',
                               'nfev', 'nit', 'bracket', 'f_bracket']
        return res_out

    xl, xm, xr = init
    default_tolerances = dict(xatol=None, xrtol=None, fatol=None, frtol=None)
    tolerances = {} if tolerances is None else tolerances
    default_tolerances.update(tolerances)
    tolerances = default_tolerances

    if callable(callback):
        def _callback(res):
            return callback(reformat_result(res))
    else:
        _callback = callback

    res = _chandrupatla_minimize(f, xl, xm, xr, args=args, **tolerances,
                                 maxiter=maxiter, callback=_callback)
    return reformat_result(res)


def bracket_root(f, xl0, xr0=None, *, xmin=None, xmax=None, factor=None, args=(),
                 maxiter=1000):
    """Bracket the root of a monotonic, real-valued function of a real variable.

    For each element of the output of `f`, `bracket_root` seeks the scalar
    bracket endpoints ``xl`` and ``xr`` such that ``sign(f(xl)) == -sign(f(xr))``
    elementwise.

    The function is guaranteed to find a valid bracket if the function is monotonic,
    but it may find a bracket under other conditions.

    This function works elementwise when `xl0`, `xr0`, `xmin`, `xmax`, `factor`, and
    the elements of `args` are (mutually broadcastable) arrays.

    Parameters
    ----------
    f : callable
        The function for which the root is to be bracketed. The signature must be::

            f(x: array, *args) -> array

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.

        `f` must be an elementwise function: each element ``f(x)[i]``
        must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
        array ``x`` or the arrays in ``args``.
    xl0, xr0: float array_like
        Starting guess of bracket, which need not contain a root. If `xr0` is
        not provided, ``xr0 = xl0 + 1``. Must be broadcastable with all other
        array inputs.
    xmin, xmax : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with all other array inputs.
    factor : float array_like, default: 2
        The factor used to grow the bracket. See Notes.
    args : tuple of array_like, optional
        Additional positional array arguments to be passed to `f`.
        If the callable for which the root is desired requires arguments that are
        not broadcastable with `x`, wrap that callable with `f` such that `f`
        accepts only `x` and broadcastable ``*args``.
    maxiter : int, default: 1000
        The maximum number of iterations of the algorithm to perform.

    Returns
    -------
    res : _RichResult
        An object similar to an instance of `scipy.optimize.OptimizeResult` with the
        following attributes. The descriptions are written as though the values will
        be scalars; however, if `f` returns an array, the outputs will be
        arrays of the same shape.

        success : bool array
            ``True`` where the algorithm terminated successfully (status ``0``);
            ``False`` otherwise.
        status : int array
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits without success.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``-5``: The initial bracket does not satisfy`xmin <= xl0 < xr0 < xmax`.
            
        bracket : 2-tuple of float arrays
            The lower and upper endpoints of the bracket, if the algorithm
            terminated successfully.
        f_bracket : 2-tuple of float arrays
            The values of `f` evaluated at the endpoints of ``res.bracket``,
            respectively.
        nfev : int array
            The number of abscissae at which `f` was evaluated to find the root.
            This is distinct from the number of times `f` is *called* because the
            the function may evaluated at multiple points in a single call.
        nit : int array
            The number of iterations of the algorithm that were performed.

    Notes
    -----
    This function generalizes an algorithm found in pieces throughout the
    `scipy.stats` codebase. The strategy is to iteratively grow the bracket `(l, r)`
    until ``f(l) < 0 < f(r)`` or ``f(r) < 0 < f(l)``. The bracket grows to the left
    as follows.

    - If `xmin` is not provided, the distance between `xl0` and `l` is iteratively
      increased by `factor`.
    - If `xmin` is provided, the distance between `xmin` and `l` is iteratively
      decreased by `factor`. Note that this also *increases* the bracket size.

    Growth of the bracket to the right is analogous.

    Growth of the bracket in one direction stops when the endpoint is no longer
    finite, the function value at the endpoint is no longer finite, or the
    endpoint reaches its limiting value (`xmin` or `xmax`). Iteration terminates
    when the bracket stops growing in both directions, the bracket surrounds
    the root, or a root is found (by chance).

    If two brackets are found - that is, a bracket is found on both sides in
    the same iteration, the smaller of the two is returned.
    
    If roots of the function are found, both `xl` and `xr` are set to the
    leftmost root.
    
    See Also
    --------
    find_root

    Examples
    --------
    Suppose we wish to find the root of the following function.

    >>> def f(x, c=5):
    ...     return x**3 - 2*x - c

    First, we must find a valid bracket. The function is not monotonic,
    but `bracket_root` may be able to provide a bracket.

    >>> from scipy.optimize import elementwise
    >>> res_bracket = elementwise.bracket_root(f, 0)
    >>> res_bracket.success
    True
    >>> res_bracket.bracket
    (2.0, 4.0)

    Indeed, the values of the function at the bracket endpoints have
    opposite signs.

    >>> res_bracket.f_bracket
    (-1.0, 51.0)

    Once we have a valid bracket, `find_root` can be used to provide
    a precise root.

    >>> res_root = elementwise.find_root(f, res_bracket.bracket)
    >>> res_root.x
    2.0945514815423265

    `bracket_root` and `find_root` accept arrays for most arguments.
    For instance, to find the root for a few values of the parameter ``c``
    at once:

    >>> import numpy as np
    >>> c = np.asarray([3, 4, 5])
    >>> res_bracket = elementwise.bracket_root(f, 0, args=(c,))
    >>> res_bracket.bracket
    (array([1., 1., 2.]), array([2., 2., 4.]))
    >>> res_root = elementwise.find_root(f, res_bracket.bracket, args=(c,))
    >>> res_root.x
    array([1.8932892 , 2.        , 2.09455148])

    """  # noqa: E501

    res = _bracket_root(f, xl0, xr0=xr0, xmin=xmin, xmax=xmax, factor=factor,
                        args=args, maxiter=maxiter)
    res.bracket = res.xl, res.xr
    res.f_bracket = res.fl, res.fr
    del res.xl
    del res.xr
    del res.fl
    del res.fr
    return res


def bracket_minimum(f, xm0, *, xl0=None, xr0=None, xmin=None, xmax=None,
                     factor=None, args=(), maxiter=1000):
    """Bracket the minimum of a unimodal, real-valued function of a real variable.

    For each element of the output of `f`, `bracket_minimum` seeks the scalar
    bracket points ``xl < xm < xr`` such that ``fl >= fm <= fr`` where one of the
    inequalities is strict.

    The function is guaranteed to find a valid bracket if the function is
    strongly unimodal, but it may find a bracket under other conditions.

    This function works elementwise when `xm0`, `xl0`, `xr0`, `xmin`, `xmax`, `factor`,
    and the elements of `args` are (mutually broadcastable) arrays.

    Parameters
    ----------
    f : callable
        The function for which the root is to be bracketed. The signature must be::

            f(x: array, *args) -> array

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.

        `f` must be an elementwise function: each element ``f(x)[i]``
        must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
        array ``x`` or the arrays in ``args``.
    xm0: float array_like
        Starting guess for middle point of bracket.
    xl0, xr0: float array_like, optional
        Starting guesses for left and right endpoints of the bracket. Must
        be broadcastable with all other array inputs.
    xmin, xmax : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with all other array inputs.
    factor : float array_like, default: 2
        The factor used to grow the bracket. See Notes.
    args : tuple of array_like, optional
        Additional positional array arguments to be passed to `f`.
        If the callable for which the root is desired requires arguments that are
        not broadcastable with `x`, wrap that callable with `f` such that `f`
        accepts only `x` and broadcastable ``*args``.
    maxiter : int, default: 1000
        The maximum number of iterations of the algorithm to perform.

    Returns
    -------
    res : _RichResult
        An object similar to an instance of `scipy.optimize.OptimizeResult` with the
        following attributes. The descriptions are written as though the values will
        be scalars; however, if `f` returns an array, the outputs will be
        arrays of the same shape.

        success : bool array
            ``True`` where the algorithm terminated successfully (status ``0``);
            ``False`` otherwise.
        status : int array
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits. Assuming
              unimodality, this implies the endpoint at the limit is a minimizer.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : ``None`` shall pass.
            - ``-5`` : The initial bracket does not satisfy
              `xmin <= xl0 < xm0 < xr0 <= xmax`.

        bracket : 3-tuple of float arrays
            The left, middle, and right points of the bracket, if the algorithm
            terminated successfully.
        f_bracket : 3-tuple of float arrays
            The function value at the left, middle, and right points of the bracket.
        nfev : int array
            The number of abscissae at which `f` was evaluated to find the root.
            This is distinct from the number of times `f` is *called* because the
            the function may evaluated at multiple points in a single call.
        nit : int array
            The number of iterations of the algorithm that were performed.

    Notes
    -----
    Similar to `scipy.optimize.bracket`, this function seeks to find real
    points ``xl < xm < xr`` such that ``f(xl) >= f(xm)`` and ``f(xr) >= f(xm)``,
    where at least one of the inequalities is strict. Unlike `scipy.optimize.bracket`,
    this function can operate in a vectorized manner on array input, so long as
    the input arrays are broadcastable with each other. Also unlike
    `scipy.optimize.bracket`, users may specify minimum and maximum endpoints
    for the desired bracket.

    Given an initial trio of points ``xl = xl0``, ``xm = xm0``, ``xr = xr0``,
    the algorithm checks if these points already give a valid bracket. If not,
    a new endpoint, ``w`` is chosen in the "downhill" direction, ``xm`` becomes the new
    opposite endpoint, and either `xl` or `xr` becomes the new middle point,
    depending on which direction is downhill. The algorithm repeats from here.

    The new endpoint `w` is chosen differently depending on whether or not a
    boundary `xmin` or `xmax` has been set in the downhill direction. Without
    loss of generality, suppose the downhill direction is to the right, so that
    ``f(xl) > f(xm) > f(xr)``. If there is no boundary to the right, then `w`
    is chosen to be ``xr + factor * (xr - xm)`` where `factor` is controlled by
    the user (defaults to 2.0) so that step sizes increase in geometric proportion.
    If there is a boundary, `xmax` in this case, then `w` is chosen to be
    ``xmax - (xmax - xr)/factor``, with steps slowing to a stop at
    `xmax`. This cautious approach ensures that a minimum near but distinct from
    the boundary isn't missed while also detecting whether or not the `xmax` is
    a minimizer when `xmax` is reached after a finite number of steps.

    See Also
    --------
    scipy.optimize.bracket
    scipy.optimize.elementwise.find_minimum

    Examples
    --------
    Suppose we wish to minimize the following function.

    >>> def f(x, c=1):
    ...     return (x - c)**2 + 2

    First, we must find a valid bracket. The function is unimodal,
    so `bracket_minium` will easily find a bracket.

    >>> from scipy.optimize import elementwise
    >>> res_bracket = elementwise.bracket_minimum(f, 0)
    >>> res_bracket.success
    True
    >>> res_bracket.bracket
    (0.0, 0.5, 1.5)

    Indeed, the bracket points are ordered and the function value
    at the middle bracket point is less than at the surrounding
    points.

    >>> xl, xm, xr = res_bracket.bracket
    >>> fl, fm, fr = res_bracket.f_bracket
    >>> (xl < xm < xr) and (fl > fm <= fr)
    True

    Once we have a valid bracket, `find_minimum` can be used to provide
    an estimate of the minimizer.

    >>> res_minimum = elementwise.find_minimum(f, res_bracket.bracket)
    >>> res_minimum.x
    1.0000000149011612

    `bracket_minimum` and `find_minimum` accept arrays for most arguments.
    For instance, to find the minimizers and minima for a few values of the
    parameter ``c`` at once:

    >>> import numpy as np
    >>> c = np.asarray([1, 1.5, 2])
    >>> res_bracket = elementwise.bracket_minimum(f, 0, args=(c,))
    >>> res_bracket.bracket
    (array([0. , 0.5, 0.5]), array([0.5, 1.5, 1.5]), array([1.5, 2.5, 2.5]))
    >>> res_minimum = elementwise.find_minimum(f, res_bracket.bracket, args=(c,))
    >>> res_minimum.x
    array([1.00000001, 1.5       , 2.        ])
    >>> res_minimum.f_x
    array([2., 2., 2.])

    """  # noqa: E501

    res = _bracket_minimum(f, xm0, xl0=xl0, xr0=xr0, xmin=xmin, xmax=xmax,
                           factor=factor, args=args, maxiter=maxiter)
    res.bracket = res.xl, res.xm, res.xr
    res.f_bracket = res.fl, res.fm, res.fr
    del res.xl
    del res.xm
    del res.xr
    del res.fl
    del res.fm
    del res.fr
    return res
<|endoftext|>"""Hessian update strategies for quasi-Newton optimization methods."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs, issymmetric
from warnings import warn


__all__ = ['HessianUpdateStrategy', 'BFGS', 'SR1']


class HessianUpdateStrategy:
    """Interface for implementing Hessian update strategies.

    Many optimization methods make use of Hessian (or inverse Hessian)
    approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
    Some of these  approximations, however, do not actually need to store
    the entire matrix or can compute the internal matrix product with a
    given vector in a very efficiently manner. This class serves as an
    abstract interface between the optimization algorithm and the
    quasi-Newton update strategies, giving freedom of implementation
    to store and update the internal matrix as efficiently as possible.
    Different choices of initialization and update procedure will result
    in different quasi-Newton strategies.

    Four methods should be implemented in derived classes: ``initialize``,
    ``update``, ``dot`` and ``get_matrix``. The matrix multiplication
    operator ``@`` is also defined to call the ``dot`` method.

    Notes
    -----
    Any instance of a class that implements this interface,
    can be accepted by the method ``minimize`` and used by
    the compatible solvers to approximate the Hessian (or
    inverse Hessian) used by the optimization algorithms.
    """

    def initialize(self, n, approx_type):
        """Initialize internal matrix.

        Allocate internal memory for storing and updating
        the Hessian or its inverse.

        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        raise NotImplementedError("The method ``initialize(n, approx_type)``"
                                  " is not implemented.")

    def update(self, delta_x, delta_grad):
        """Update internal matrix.

        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.

        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        raise NotImplementedError("The method ``update(delta_x, delta_grad)``"
                                  " is not implemented.")

    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.

        Parameters
        ----------
        p : array_like
            1-D array representing a vector.

        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        raise NotImplementedError("The method ``dot(p)``"
                                  " is not implemented.")

    def get_matrix(self):
        """Return current internal matrix.

        Returns
        -------
        H : ndarray, shape (n, n)
            Dense matrix containing either the Hessian
            or its inverse (depending on how 'approx_type'
            is defined).
        """
        raise NotImplementedError("The method ``get_matrix(p)``"
                                  " is not implemented.")

    def __matmul__(self, p):
        return self.dot(p)


class FullHessianUpdateStrategy(HessianUpdateStrategy):
    """Hessian update strategy with full dimensional internal representation.
    """
    _syr = get_blas_funcs('syr', dtype='d')  # Symmetric rank 1 update
    _syr2 = get_blas_funcs('syr2', dtype='d')  # Symmetric rank 2 update
    # Symmetric matrix-vector product
    _symv = get_blas_funcs('symv', dtype='d')

    def __init__(self, init_scale='auto'):
        self.init_scale = init_scale
        # Until initialize is called we can't really use the class,
        # so it makes sense to set everything to None.
        self.first_iteration = None
        self.approx_type = None
        self.B = None
        self.H = None

    def initialize(self, n, approx_type):
        """Initialize internal matrix.

        Allocate internal memory for storing and updating
        the Hessian or its inverse.

        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type not in ('hess', 'inv_hess'):
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        # Create matrix
        if self.approx_type == 'hess':
            self.B = np.eye(n, dtype=float)
        else:
            self.H = np.eye(n, dtype=float)

    def _auto_scale(self, delta_x, delta_grad):
        # Heuristic to scale matrix at first iteration.
        # Described in Nocedal and Wright "Numerical Optimization"
        # p.143 formula (6.20).
        s_norm2 = np.dot(delta_x, delta_x)
        y_norm2 = np.dot(delta_grad, delta_grad)
        ys = np.abs(np.dot(delta_grad, delta_x))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            return 1
        if self.approx_type == 'hess':
            return y_norm2 / ys
        else:
            return ys / y_norm2

    def _update_implementation(self, delta_x, delta_grad):
        raise NotImplementedError("The method ``_update_implementation``"
                                  " is not implemented.")

    def update(self, delta_x, delta_grad):
        """Update internal matrix.

        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.

        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            warn('delta_grad == 0.0. Check if the approximated '
                 'function is linear. If the function is linear '
                 'better results can be obtained by defining the '
                 'Hessian as zero instead of using quasi-Newton '
                 'approximations.',
                 UserWarning, stacklevel=2)
            return
        if self.first_iteration:
            # Get user specific scale
            if isinstance(self.init_scale, str) and self.init_scale == "auto":
                scale = self._auto_scale(delta_x, delta_grad)
            else:
                scale = self.init_scale

            # Check for complex: numpy will silently cast a complex array to
            # a real one but not so for scalar as it raises a TypeError.
            # Checking here brings a consistent behavior.
            replace = False
            if np.size(scale) == 1:
                # to account for the legacy behavior having the exact same cast
                scale = float(scale)
            elif np.iscomplexobj(scale):
                raise TypeError("init_scale contains complex elements, "
                                "must be real.")
            else:  # test explicitly for allowed shapes and values
                replace = True
                if self.approx_type == 'hess':
                    shape = np.shape(self.B)
                    dtype = self.B.dtype
                else:
                    shape = np.shape(self.H)
                    dtype = self.H.dtype
                # copy, will replace the original
                scale = np.array(scale, dtype=dtype, copy=True)

                # it has to match the shape of the matrix for the multiplication,
                # no implicit broadcasting is allowed
                if shape != (init_shape := np.shape(scale)):
                    raise ValueError("If init_scale is an array, it must have the "
                                     f"dimensions of the hess/inv_hess: {shape}."
                                     f" Got {init_shape}.")
                if not issymmetric(scale):
                    raise ValueError("If init_scale is an array, it must be"
                                     " symmetric (passing scipy.linalg.issymmetric)"
                                     " to be an approximation of a hess/inv_hess.")

            # Scale initial matrix with ``scale * np.eye(n)`` or replace
            # This is not ideal, we could assign the scale directly in
            # initialize, but we would need to
            if self.approx_type == 'hess':
                if replace:
                    self.B = scale
                else:
                    self.B *= scale
            else:
                if replace:
                    self.H = scale
                else:
                    self.H *= scale
            self.first_iteration = False
        self._update_implementation(delta_x, delta_grad)

    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.

        Parameters
        ----------
        p : array_like
            1-D array representing a vector.

        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        if self.approx_type == 'hess':
            return self._symv(1, self.B, p)
        else:
            return self._symv(1, self.H, p)

    def get_matrix(self):
        """Return the current internal matrix.

        Returns
        -------
        M : ndarray, shape (n, n)
            Dense matrix containing either the Hessian or its inverse
            (depending on how `approx_type` was defined).
        """
        if self.approx_type == 'hess':
            M = np.copy(self.B)
        else:
            M = np.copy(self.H)
        li = np.tril_indices_from(M, k=-1)
        M[li] = M.T[li]
        return M


class BFGS(FullHessianUpdateStrategy):
    """Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.

    Parameters
    ----------
    exception_strategy : {'skip_update', 'damp_update'}, optional
        Define how to proceed when the curvature condition is violated.
        Set it to 'skip_update' to just skip the update. Or, alternatively,
        set it to 'damp_update' to interpolate between the actual BFGS
        result and the unmodified matrix. Both exceptions strategies
        are explained  in [1]_, p.536-537.
    min_curvature : float
        This number, scaled by a normalization factor, defines the
        minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
        unaffected by the exception strategy. By default is equal to
        1e-8 when ``exception_strategy = 'skip_update'`` and equal
        to 0.2 when ``exception_strategy = 'damp_update'``.
    init_scale : {float, np.array, 'auto'}
        This parameter can be used to initialize the Hessian or its
        inverse. When a float is given, the relevant array is initialized
        to ``np.eye(n) * init_scale``, where ``n`` is the problem dimension.
        Alternatively, if a precisely ``(n, n)`` shaped, symmetric array is given,
        this array will be used. Otherwise an error is generated.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        The default is 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.140.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, exception_strategy='skip_update', min_curvature=None,
                 init_scale='auto'):
        if exception_strategy == 'skip_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-8
        elif exception_strategy == 'damp_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            raise ValueError("`exception_strategy` must be 'skip_update' "
                             "or 'damp_update'.")

        super().__init__(init_scale)
        self.exception_strategy = exception_strategy

    def _update_inverse_hessian(self, ys, Hy, yHy, s):
        """Update the inverse Hessian matrix.

        BFGS update using the formula:

            ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
                     - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``

        where ``s = delta_x`` and ``y = delta_grad``. This formula is
        equivalent to (6.17) in [1]_ written in a more efficient way
        for implementation.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        self.H = self._syr2(-1.0 / ys, s, Hy, a=self.H)
        self.H = self._syr((ys + yHy) / ys ** 2, s, a=self.H)

    def _update_hessian(self, ys, Bs, sBs, y):
        """Update the Hessian matrix.

        BFGS update using the formula:

            ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``

        where ``s`` is short for ``delta_x`` and ``y`` is short
        for ``delta_grad``. Formula (6.19) in [1]_.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        self.B = self._syr(1.0 / ys, y, a=self.B)
        self.B = self._syr(-1.0 / sBs, Bs, a=self.B)

    def _update_implementation(self, delta_x, delta_grad):
        # Auxiliary variables w and z
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        # Do some common operations
        wz = np.dot(w, z)
        Mw = self @ w
        wMw = Mw.dot(w)
        # Guarantee that wMw > 0 by reinitializing matrix.
        # While this is always true in exact arithmetic,
        # indefinite matrix may appear due to roundoff errors.
        if wMw <= 0.0:
            scale = self._auto_scale(delta_x, delta_grad)
            # Reinitialize matrix
            if self.approx_type == 'hess':
                self.B = scale * np.eye(self.n, dtype=float)
            else:
                self.H = scale * np.eye(self.n, dtype=float)
            # Do common operations for new matrix
            Mw = self @ w
            wMw = Mw.dot(w)
        # Check if curvature condition is violated
        if wz <= self.min_curvature * wMw:
            # If the option 'skip_update' is set
            # we just skip the update when the condition
            # is violated.
            if self.exception_strategy == 'skip_update':
                return
            # If the option 'damp_update' is set we
            # interpolate between the actual BFGS
            # result and the unmodified matrix.
            elif self.exception_strategy == 'damp_update':
                update_factor = (1-self.min_curvature) / (1 - wz/wMw)
                z = update_factor*z + (1-update_factor)*Mw
                wz = np.dot(w, z)
        # Update matrix
        if self.approx_type == 'hess':
            self._update_hessian(wz, Mw, wMw, z)
        else:
            self._update_inverse_hessian(wz, Mw, wMw, z)


class SR1(FullHessianUpdateStrategy):
    """Symmetric-rank-1 Hessian update strategy.

    Parameters
    ----------
    min_denominator : float
        This number, scaled by a normalization factor,
        defines the minimum denominator magnitude allowed
        in the update. When the condition is violated we skip
        the update. By default uses ``1e-8``.
    init_scale : {float, np.array, 'auto'}, optional
        This parameter can be used to initialize the Hessian or its
        inverse. When a float is given, the relevant array is initialized
        to ``np.eye(n) * init_scale``, where ``n`` is the problem dimension.
        Alternatively, if a precisely ``(n, n)`` shaped, symmetric array is given,
        this array will be used. Otherwise an error is generated.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        The default is 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.144-146.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, min_denominator=1e-8, init_scale='auto'):
        self.min_denominator = min_denominator
        super().__init__(init_scale)

    def _update_implementation(self, delta_x, delta_grad):
        # Auxiliary variables w and z
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        # Do some common operations
        Mw = self @ w
        z_minus_Mw = z - Mw
        denominator = np.dot(w, z_minus_Mw)
        # If the denominator is too small
        # we just skip the update.
        if np.abs(denominator) <= self.min_denominator*norm(w)*norm(z_minus_Mw):
            return
        # Update matrix
        if self.approx_type == 'hess':
            self.B = self._syr(1/denominator, z_minus_Mw, a=self.B)
        else:
            self.H = self._syr(1/denominator, z_minus_Mw, a=self.H)
<|endoftext|>import numpy as np
import operator
import warnings
import numbers
from . import (linear_sum_assignment, OptimizeResult)
from ._optimize import _check_unknown_options

from scipy._lib._util import check_random_state
import itertools

QUADRATIC_ASSIGNMENT_METHODS = ['faq', '2opt']


def quadratic_assignment(A, B, method="faq", options=None):
    r"""
    Approximates solution to the quadratic assignment problem and
    the graph matching problem.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.


    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.

    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.

    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem.
        :ref:`'faq' <optimize.qap-faq>` (default) and
        :ref:`'2opt' <optimize.qap-2opt>` are available.

    options : dict, optional
        A dictionary of solver options. All solvers support the following:

        maximize : bool (default: False)
            Maximizes the objective function if ``True``.

        partial_match : 2-D array of integers, optional (default: None)
            Fixes part of the matching. Also known as a "seed" [2]_.

            Each row of `partial_match` specifies a pair of matched nodes:
            node ``partial_match[i, 0]`` of `A` is matched to node
            ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,
            where ``m`` is not greater than the number of nodes, :math:`n`.

        rng : `numpy.random.Generator`, optional
            Pseudorandom number generator state. When `rng` is None, a new
            `numpy.random.Generator` is created using entropy from the
            operating system. Types other than `numpy.random.Generator` are
            passed to `numpy.random.default_rng` to instantiate a ``Generator``.

            .. versionchanged:: 1.15.0
                As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
                transition from use of `numpy.random.RandomState` to
                `numpy.random.Generator` is occurring. Supplying
                `np.random.RandomState` to this function will now emit a
                `DeprecationWarning`. In SciPy 1.17 its use will raise an exception.
                In addition relying on global state using `np.random.seed`
                will emit a `FutureWarning`. In SciPy 1.17 the global random number
                generator will no longer be used.
                Use of an int-like seed will raise a `FutureWarning`, in SciPy 1.17 it
                will be normalized via `np.random.default_rng` rather than
                `np.random.RandomState`.

        For method-specific options, see
        :func:`show_options('quadratic_assignment') <show_options>`.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : 1-D array
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of iterations performed during optimization.

    Notes
    -----
    The default method :ref:`'faq' <optimize.qap-faq>` uses the Fast
    Approximate QAP algorithm [1]_; it typically offers the best combination of
    speed and accuracy.
    Method :ref:`'2opt' <optimize.qap-2opt>` can be computationally expensive,
    but may be a useful alternative, or it can be used to refine the solution
    returned by another method.

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`

    .. [3] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import quadratic_assignment
    >>> rng = np.random.default_rng()
    >>> A = np.array([[0, 80, 150, 170], [80, 0, 130, 100],
    ...               [150, 130, 0, 120], [170, 100, 120, 0]])
    >>> B = np.array([[0, 5, 2, 7], [0, 0, 3, 8],
    ...               [0, 0, 0, 3], [0, 0, 0, 0]])
    >>> res = quadratic_assignment(A, B, options={'rng': rng})
    >>> print(res)
         fun: 3260
     col_ind: [0 3 2 1]
         nit: 9

    The see the relationship between the returned ``col_ind`` and ``fun``,
    use ``col_ind`` to form the best permutation matrix found, then evaluate
    the objective function :math:`f(P) = trace(A^T P B P^T )`.

    >>> perm = res['col_ind']
    >>> P = np.eye(len(A), dtype=int)[perm]
    >>> fun = np.trace(A.T @ P @ B @ P.T)
    >>> print(fun)
    3260

    Alternatively, to avoid constructing the permutation matrix explicitly,
    directly permute the rows and columns of the distance matrix.

    >>> fun = np.trace(A.T @ B[perm][:, perm])
    >>> print(fun)
    3260

    Although not guaranteed in general, ``quadratic_assignment`` happens to
    have found the globally optimal solution.

    >>> from itertools import permutations
    >>> perm_opt, fun_opt = None, np.inf
    >>> for perm in permutations([0, 1, 2, 3]):
    ...     perm = np.array(perm)
    ...     fun = np.trace(A.T @ B[perm][:, perm])
    ...     if fun < fun_opt:
    ...         fun_opt, perm_opt = fun, perm
    >>> print(np.array_equal(perm_opt, res['col_ind']))
    True

    Here is an example for which the default method,
    :ref:`'faq' <optimize.qap-faq>`, does not find the global optimum.

    >>> A = np.array([[0, 5, 8, 6], [5, 0, 5, 1],
    ...               [8, 5, 0, 2], [6, 1, 2, 0]])
    >>> B = np.array([[0, 1, 8, 4], [1, 0, 5, 2],
    ...               [8, 5, 0, 5], [4, 2, 5, 0]])
    >>> res = quadratic_assignment(A, B, options={'rng': rng})
    >>> print(res)
         fun: 178
     col_ind: [1 0 3 2]
         nit: 13

    If accuracy is important, consider using  :ref:`'2opt' <optimize.qap-2opt>`
    to refine the solution.

    >>> guess = np.array([np.arange(len(A)), res.col_ind]).T
    >>> res = quadratic_assignment(A, B, method="2opt",
    ...     options = {'rng': rng, 'partial_guess': guess})
    >>> print(res)
         fun: 176
     col_ind: [1 2 3 0]
         nit: 17

    """

    if options is None:
        options = {}

    method = method.lower()
    methods = {"faq": _quadratic_assignment_faq,
               "2opt": _quadratic_assignment_2opt}
    if method not in methods:
        raise ValueError(f"method {method} must be in {methods}.")

    _spec007_transition(options.get("rng", None))
    res = methods[method](A, B, **options)
    return res


def _spec007_transition(rng):
    if isinstance(rng, np.random.RandomState):
        warnings.warn(
            "Use of `RandomState` with `quadratic_assignment` is deprecated"
            " and will result in an exception in SciPy 1.17",
            DeprecationWarning,
            stacklevel=2
        )
    if ((rng is None or rng is np.random) and
            np.random.mtrand._rand._bit_generator._seed_seq is None):
        warnings.warn(
            "The NumPy global RNG was seeded by calling `np.random.seed`."
            " From SciPy 1.17, this function will no longer use the global RNG.",
            FutureWarning,
            stacklevel=2
        )
    if isinstance(rng, numbers.Integral | np.integer):
        warnings.warn(
            "The behavior when the rng option is an integer is changing: the value"
            " will be normalized using np.random.default_rng beginning in SciPy 1.17,"
            " and the resulting Generator will be used to generate random numbers.",
            FutureWarning,
            stacklevel=2
        )


def _calc_score(A, B, perm):
    # equivalent to objective function but avoids matmul
    return np.sum(A * B[perm][:, perm])


def _common_input_validation(A, B, partial_match):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match).astype(int)

    msg = None
    if A.shape[0] != A.shape[1]:
        msg = "`A` must be square"
    elif B.shape[0] != B.shape[1]:
        msg = "`B` must be square"
    elif A.ndim != 2 or B.ndim != 2:
        msg = "`A` and `B` must have exactly two dimensions"
    elif A.shape != B.shape:
        msg = "`A` and `B` matrices must be of equal size"
    elif partial_match.shape[0] > A.shape[0]:
        msg = "`partial_match` can have only as many seeds as there are nodes"
    elif partial_match.shape[1] != 2:
        msg = "`partial_match` must have two columns"
    elif partial_match.ndim != 2:
        msg = "`partial_match` must have exactly two dimensions"
    elif (partial_match < 0).any():
        msg = "`partial_match` must contain only positive indices"
    elif (partial_match >= len(A)).any():
        msg = "`partial_match` entries must be less than number of nodes"
    elif (not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or
          not len(set(partial_match[:, 1])) == len(partial_match[:, 1])):
        msg = "`partial_match` column entries must be unique"

    if msg is not None:
        raise ValueError(msg)

    return A, B, partial_match


def _quadratic_assignment_faq(A, B,
                              maximize=False, partial_match=None, rng=None,
                              P0="barycenter", shuffle_input=False, maxiter=30,
                              tol=0.03, **unknown_options):
    r"""Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the Fast Approximate QAP Algorithm
    (FAQ) [1]_.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.

    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.
    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for 'faq'.
        :ref:`'2opt' <optimize.qap-2opt>` is also available.

    Options
    -------
    maximize : bool (default: False)
        Maximizes the objective function if ``True``.
    partial_match : 2-D array of integers, optional (default: None)
        Fixes part of the matching. Also known as a "seed" [2]_.

        Each row of `partial_match` specifies a pair of matched nodes:
        node ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``, where
        ``m`` is not greater than the number of nodes, :math:`n`.

    rng : {None, int, `numpy.random.Generator`}, optional
        Pseudorandom number generator state. See `quadratic_assignment` for details.
    P0 : 2-D array, "barycenter", or "randomized" (default: "barycenter")
        Initial position. Must be a doubly-stochastic matrix [3]_.

        If the initial position is an array, it must be a doubly stochastic
        matrix of size :math:`m' \times m'` where :math:`m' = n - m`.

        If ``"barycenter"`` (default), the initial position is the barycenter
        of the Birkhoff polytope (the space of doubly stochastic matrices).
        This is a :math:`m' \times m'` matrix with all entries equal to
        :math:`1 / m'`.

        If ``"randomized"`` the initial search position is
        :math:`P_0 = (J + K) / 2`, where :math:`J` is the barycenter and
        :math:`K` is a random doubly stochastic matrix.
    shuffle_input : bool (default: False)
        Set to `True` to resolve degenerate gradients randomly. For
        non-degenerate gradients this option has no effect.
    maxiter : int, positive (default: 30)
        Integer specifying the max number of Frank-Wolfe iterations performed.
    tol : float (default: 0.03)
        Tolerance for termination. Frank-Wolfe iteration terminates when
        :math:`\frac{||P_{i}-P_{i+1}||_F}{\sqrt{m'}} \leq tol`,
        where :math:`i` is the iteration number.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : 1-D array
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of Frank-Wolfe iterations performed.

    Notes
    -----
    The algorithm may be sensitive to the initial permutation matrix (or
    search "position") due to the possibility of several local minima
    within the feasible region. A barycenter initialization is more likely to
    result in a better solution than a single random initialization. However,
    calling ``quadratic_assignment`` several times with different random
    initializations may result in a better optimum at the cost of longer
    total execution time.

    Examples
    --------
    As mentioned above, a barycenter initialization often results in a better
    solution than a single random initialization.

    >>> from scipy.optimize import quadratic_assignment
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> n = 15
    >>> A = rng.random((n, n))
    >>> B = rng.random((n, n))
    >>> options = {"rng": rng}
    >>> res = quadratic_assignment(A, B, options=options)  # FAQ is default method
    >>> print(res.fun)
    47.797048706380636  # may vary

    >>> options = {"rng": rng, "P0": "randomized"}  # use randomized initialization
    >>> res = quadratic_assignment(A, B, options=options)
    >>> print(res.fun)
    47.37287069769966 # may vary

    However, consider running from several randomized initializations and
    keeping the best result.

    >>> res = min([quadratic_assignment(A, B, options=options)
    ...            for i in range(30)], key=lambda x: x.fun)
    >>> print(res.fun)
    46.55974835248574 # may vary

    The '2-opt' method can be used to attempt to refine the results.

    >>> options = {"partial_guess": np.array([np.arange(n), res.col_ind]).T, "rng": rng}
    >>> res = quadratic_assignment(A, B, method="2opt", options=options)
    >>> print(res.fun)
    46.55974835248574 # may vary

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`

    .. [3] "Doubly stochastic Matrix," Wikipedia.
           https://en.wikipedia.org/wiki/Doubly_stochastic_matrix

    """

    _check_unknown_options(unknown_options)

    maxiter = operator.index(maxiter)

    # ValueError check
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = "Invalid 'P0' parameter string"
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    elif tol <= 0:
        msg = "'tol' must be a positive float"
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = len(A)  # number of vertices in graphs
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # [1] Algorithm 1 Line 1 - choose initialization
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = "`P0` matrix must have shape m' x m', where m'=n-m"
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = "`P0` matrix must be doubly stochastic"
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # generate a nxn matrix where each entry is a random number [0, 1]
        # would use rand, but Generators don't have it
        # would use random, but old mtrand.RandomStates don't have it
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2

    # check trivial cases
    if n == 0 or n_seeds == n:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    # definitions according to Seeded Graph Matching [2].
    A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    const_sum = A21 @ B21.T + A12.T @ B12

    P = P0
    # [1] Algorithm 1 Line 2 - loop while stopping criteria not met
    for n_iter in range(1, maxiter+1):
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        grad_fp = (const_sum + A22 @ P @ B22.T + A22.T @ P @ B22)
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
        Q = np.eye(n_unseed)[cols]

        # [1] Algorithm 1 Line 5 - compute the step size
        # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
        # terms as ax**2 + bx + c. c does not affect location of minimum
        # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
        # apply where possible for efficiency.
        R = P - Q
        b21 = ((R.T @ A21) * B21).sum()
        b12 = ((R.T @ A12.T) * B12.T).sum()
        AR22 = A22.T @ R
        BR22 = B22 @ R.T
        b22a = (AR22 * B22.T[cols]).sum()
        b22b = (A22 * BR22[cols]).sum()
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b
        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        if a*obj_func_scalar > 0 and 0 <= -b/(2*a) <= 1:
            alpha = -b/(2*a)
        else:
            alpha = np.argmin([0, (b + a)*obj_func_scalar])

        # [1] Algorithm 1 Line 6 - Update P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    # [1] Algorithm 1 Line 7 - end main loop

    # [1] Algorithm 1 Line 8 - project onto the set of permutation matrices
    _, col = linear_sum_assignment(P, maximize=True)
    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]

    score = _calc_score(A, B, unshuffled_perm)
    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter}
    return OptimizeResult(res)


def _split_matrix(X, n):
    # definitions according to Seeded Graph Matching [2].
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _doubly_stochastic(P, tol=1e-3):
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if ((np.abs(P_eps.sum(axis=1) - 1) < tol).all() and
                (np.abs(P_eps.sum(axis=0) - 1) < tol).all()):
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _quadratic_assignment_2opt(A, B, maximize=False, rng=None,
                               partial_match=None,
                               partial_guess=None,
                               **unknown_options):
    r"""Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the 2-opt algorithm [1]_.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.

    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.
    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for '2opt'.
        :ref:`'faq' <optimize.qap-faq>` is also available.

    Options
    -------
    maximize : bool (default: False)
        Maximizes the objective function if ``True``.
    rng : {None, int, `numpy.random.Generator`}, optional
        Pseudorandom number generator state. See `quadratic_assignment` for details.
    partial_match : 2-D array of integers, optional (default: None)
        Fixes part of the matching. Also known as a "seed" [2]_.

        Each row of `partial_match` specifies a pair of matched nodes: node
        ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,
        where ``m`` is not greater than the number of nodes, :math:`n`.

        .. note::
             `partial_match` must be sorted by the first column.

    partial_guess : 2-D array of integers, optional (default: None)
        A guess for the matching between the two matrices. Unlike
        `partial_match`, `partial_guess` does not fix the indices; they are
        still free to be optimized.

        Each row of `partial_guess` specifies a pair of matched nodes: node
        ``partial_guess[i, 0]`` of `A` is matched to node
        ``partial_guess[i, 1]`` of `B`. The array has shape ``(m, 2)``,
        where ``m`` is not greater than the number of nodes, :math:`n`.

        .. note::
                `partial_guess` must be sorted by the first column.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : 1-D array
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of iterations performed during optimization.

    Notes
    -----
    This is a greedy algorithm that works similarly to bubble sort: beginning
    with an initial permutation, it iteratively swaps pairs of indices to
    improve the objective function until no such improvements are possible.

    References
    ----------
    .. [1] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, https://doi.org/10.1016/j.patcog.2018.09.014

    """
    _check_unknown_options(unknown_options)
    rng = check_random_state(rng)
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    N = len(A)
    # check trivial cases
    if N == 0 or partial_match.shape[0] == N:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    if partial_guess is None:
        partial_guess = np.array([[], []]).T
    partial_guess = np.atleast_2d(partial_guess).astype(int)

    msg = None
    if partial_guess.shape[0] > A.shape[0]:
        msg = ("`partial_guess` can have only as "
               "many entries as there are nodes")
    elif partial_guess.shape[1] != 2:
        msg = "`partial_guess` must have two columns"
    elif partial_guess.ndim != 2:
        msg = "`partial_guess` must have exactly two dimensions"
    elif (partial_guess < 0).any():
        msg = "`partial_guess` must contain only positive indices"
    elif (partial_guess >= len(A)).any():
        msg = "`partial_guess` entries must be less than number of nodes"
    elif (not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or
          not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1])):
        msg = "`partial_guess` column entries must be unique"
    if msg is not None:
        raise ValueError(msg)

    fixed_rows = None
    if partial_match.size or partial_guess.size:
        # use partial_match and partial_guess for initial permutation,
        # but randomly permute the rest.
        guess_rows = np.zeros(N, dtype=bool)
        guess_cols = np.zeros(N, dtype=bool)
        fixed_rows = np.zeros(N, dtype=bool)
        fixed_cols = np.zeros(N, dtype=bool)
        perm = np.zeros(N, dtype=int)

        rg, cg = partial_guess.T
        guess_rows[rg] = True
        guess_cols[cg] = True
        perm[guess_rows] = cg

        # match overrides guess
        rf, cf = partial_match.T
        fixed_rows[rf] = True
        fixed_cols[cf] = True
        perm[fixed_rows] = cf

        random_rows = ~fixed_rows & ~guess_rows
        random_cols = ~fixed_cols & ~guess_cols
        perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
    else:
        perm = rng.permutation(np.arange(N))

    best_score = _calc_score(A, B, perm)

    i_free = np.arange(N)
    if fixed_rows is not None:
        i_free = i_free[~fixed_rows]

    better = operator.gt if maximize else operator.lt
    n_iter = 0
    done = False
    while not done:
        # equivalent to nested for loops i in range(N), j in range(i, N)
        for i, j in itertools.combinations_with_replacement(i_free, 2):
            n_iter += 1
            perm[i], perm[j] = perm[j], perm[i]
            score = _calc_score(A, B, perm)
            if better(score, best_score):
                best_score = score
                break
            # faster to swap back than to create a new list every time
            perm[i], perm[j] = perm[j], perm[i]
        else:  # no swaps made
            done = True

    res = {"col_ind": perm, "fun": best_score, "nit": n_iter}
    return OptimizeResult(res)
<|endoftext|>"""HiGHS Linear Optimization Methods

Interface to HiGHS linear optimization software.
https://highs.dev/

.. versionadded:: 1.5.0

References
----------
.. [1] Q. Huangfu and J.A.J. Hall. "Parallelizing the dual revised simplex
           method." Mathematical Programming Computation, 10 (1), 119-142,
           2018. DOI: 10.1007/s12532-017-0130-5

"""

import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highspy._highs_wrapper import _highs_wrapper
from ._highspy._core import(
    kHighsInf,
    HighsDebugLevel,
    ObjSense,
    HighsModelStatus,
    simplex_constants as s_c,  # [1]
)
from scipy.sparse import csc_matrix, vstack, issparse

# [1]: Directly importing from "._highspy._core.simplex_constants"
# causes problems when reloading.
# See https://github.com/scipy/scipy/pull/22869 for details.

def _highs_to_scipy_status_message(highs_status, highs_message):
    """Converts HiGHS status number/message to SciPy status number/message"""

    scipy_statuses_messages = {
        None: (4, "HiGHS did not provide a status code. "),
        HighsModelStatus.kNotset: (4, ""),
        HighsModelStatus.kLoadError: (4, ""),
        HighsModelStatus.kModelError: (2, ""),
        HighsModelStatus.kPresolveError: (4, ""),
        HighsModelStatus.kSolveError: (4, ""),
        HighsModelStatus.kPostsolveError: (4, ""),
        HighsModelStatus.kModelEmpty: (4, ""),
        HighsModelStatus.kObjectiveBound: (4, ""),
        HighsModelStatus.kObjectiveTarget: (4, ""),
        HighsModelStatus.kOptimal: (0, "Optimization terminated successfully. "),
        HighsModelStatus.kTimeLimit: (1, "Time limit reached. "),
        HighsModelStatus.kIterationLimit: (1, "Iteration limit reached. "),
        HighsModelStatus.kInfeasible: (2, "The problem is infeasible. "),
        HighsModelStatus.kUnbounded: (3, "The problem is unbounded. "),
        HighsModelStatus.kUnboundedOrInfeasible: (4, "The problem is unbounded "
                                                  "or infeasible. ")}
    unrecognized = (4, "The HiGHS status code was not recognized. ")
    scipy_status, scipy_message = (
        scipy_statuses_messages.get(highs_status, unrecognized))
    hstat = int(highs_status) if highs_status is not None else None
    scipy_message = (f"{scipy_message}"
                     f"(HiGHS Status {hstat}: {highs_message})")
    return scipy_status, scipy_message


def _replace_inf(x):
    # Replace `np.inf` with kHighsInf
    infs = np.isinf(x)
    with np.errstate(invalid="ignore"):
        x[infs] = np.sign(x[infs])*kHighsInf
    return x


def _convert_to_highs_enum(option, option_str, choices):
    # If option is in the choices we can look it up, if not use
    # the default value taken from function signature and warn:
    try:
        return choices[option.lower()]
    except AttributeError:
        return choices[option]
    except KeyError:
        sig = inspect.signature(_linprog_highs)
        default_str = sig.parameters[option_str].default
        warn(f"Option {option_str} is {option}, but only values in "
             f"{set(choices.keys())} are allowed. Using default: "
             f"{default_str}.",
             OptimizeWarning, stacklevel=3)
        return choices[default_str]


def _linprog_highs(lp, solver, time_limit=None, presolve=True,
                   disp=False, maxiter=None,
                   dual_feasibility_tolerance=None,
                   primal_feasibility_tolerance=None,
                   ipm_optimality_tolerance=None,
                   simplex_dual_edge_weight_strategy=None,
                   mip_rel_gap=None,
                   mip_max_nodes=None,
                   **unknown_options):
    r"""
    Solve the following linear programming problem using one of the HiGHS
    solvers:

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    lp :  _LPProblem
        A ``scipy.optimize._linprog_util._LPProblem`` ``namedtuple``.
    solver : "ipm" or "simplex" or None
        Which HiGHS solver to use.  If ``None``, "simplex" will be used.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase. For
        ``solver='ipm'``, this does not include the number of crossover
        iterations.  Default is the largest possible value for an ``int``
        on the platform.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration; default ``False``.
    time_limit : float
        The maximum time in seconds allotted to solve the problem; default is
        the largest possible value for a ``double`` on the platform.
    presolve : bool
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if presolve is
        to be disabled.
    dual_feasibility_tolerance : double
        Dual feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    primal_feasibility_tolerance : double
        Primal feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    ipm_optimality_tolerance : double
        Optimality tolerance for ``solver='ipm'``.  Default is 1e-08.
        Minimum possible value is 1e-12 and must be smaller than the largest
        possible value for a ``double`` on the platform.
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.

        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Currently, using ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.

    mip_max_nodes : int
        The maximum number of nodes allotted to solve the problem; default is
        the largest possible value for a ``HighsInt`` on the platform.
        Ignored if not using the MIP solver.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing all
        unused options.

    Returns
    -------
    sol : dict
        A dictionary consisting of the fields:

            x : 1D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
            fun : float
                The optimal value of the objective function ``c @ x``.
            slack : 1D array
                The (nominally positive) values of the slack,
                ``b_ub - A_ub @ x``.
            con : 1D array
                The (nominally zero) residuals of the equality constraints,
                ``b_eq - A_eq @ x``.
            success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
            status : int
                An integer representing the exit status of the algorithm.

                ``0`` : Optimization terminated successfully.

                ``1`` : Iteration or time limit reached.

                ``2`` : Problem appears to be infeasible.

                ``3`` : Problem appears to be unbounded.

                ``4`` : The HiGHS solver ran into a problem.

            message : str
                A string descriptor of the exit status of the algorithm.
            nit : int
                The total number of iterations performed.
                For ``solver='simplex'``, this includes iterations in all
                phases. For ``solver='ipm'``, this does not include
                crossover iterations.
            crossover_nit : int
                The number of primal/dual pushes performed during the
                crossover routine for ``solver='ipm'``.  This is ``0``
                for ``solver='simplex'``.
            ineqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                inequality constraints, `b_ub`. A dictionary consisting of the
                fields:

                residual : np.ndnarray
                    The (nominally positive) values of the slack variables,
                    ``b_ub - A_ub @ x``.  This quantity is also commonly
                    referred to as "slack".

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    inequality constraints, `b_ub`.

            eqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                equality constraints, `b_eq`.  A dictionary consisting of the
                fields:

                residual : np.ndarray
                    The (nominally zero) residuals of the equality constraints,
                    ``b_eq - A_eq @ x``.

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    equality constraints, `b_eq`.

            lower, upper : OptimizeResult
                Solution and sensitivity information corresponding to the
                lower and upper bounds on decision variables, `bounds`.

                residual : np.ndarray
                    The (nominally positive) values of the quantity
                    ``x - lb`` (lower) or ``ub - x`` (upper).

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the lower and upper
                    `bounds`.

            mip_node_count : int
                The number of subproblems or "nodes" solved by the MILP
                solver. Only present when `integrality` is not `None`.

            mip_dual_bound : float
                The MILP solver's final estimate of the lower bound on the
                optimal solution. Only present when `integrality` is not
                `None`.

            mip_gap : float
                The difference between the final objective function value
                and the final dual bound, scaled by the final objective
                function value. Only present when `integrality` is not
                `None`.

    Notes
    -----
    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
    if unknown_options:
        message = (f"Unrecognized options detected: {unknown_options}. "
                   "These will be passed to HiGHS verbatim.")
        warn(message, OptimizeWarning, stacklevel=3)

    # Map options to HiGHS enum values
    simplex_dual_edge_weight_strategy_enum = _convert_to_highs_enum(
        simplex_dual_edge_weight_strategy,
        'simplex_dual_edge_weight_strategy',
        choices={'dantzig': \
                 s_c.SimplexEdgeWeightStrategy.kSimplexEdgeWeightStrategyDantzig,
                 'devex': \
                 s_c.SimplexEdgeWeightStrategy.kSimplexEdgeWeightStrategyDevex,
                 'steepest-devex': \
                 s_c.SimplexEdgeWeightStrategy.kSimplexEdgeWeightStrategyChoose,
                 'steepest': \
                 s_c.SimplexEdgeWeightStrategy.kSimplexEdgeWeightStrategySteepestEdge,
                 None: None})

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp

    lb, ub = bounds.T.copy()  # separate bounds, copy->C-cntgs
    # highs_wrapper solves LHS <= A*x <= RHS, not equality constraints
    with np.errstate(invalid="ignore"):
        lhs_ub = -np.ones_like(b_ub)*np.inf  # LHS of UB constraints is -inf
    rhs_ub = b_ub  # RHS of UB constraints is b_ub
    lhs_eq = b_eq  # Equality constraint is inequality
    rhs_eq = b_eq  # constraint with LHS=RHS
    lhs = np.concatenate((lhs_ub, lhs_eq))
    rhs = np.concatenate((rhs_ub, rhs_eq))

    if issparse(A_ub) or issparse(A_eq):
        A = vstack((A_ub, A_eq))
    else:
        A = np.vstack((A_ub, A_eq))
    A = csc_matrix(A)

    options = {
        'presolve': presolve,
        'sense': ObjSense.kMinimize,
        'solver': solver,
        'time_limit': time_limit,
        'highs_debug_level': HighsDebugLevel.kHighsDebugLevelNone,
        'dual_feasibility_tolerance': dual_feasibility_tolerance,
        'ipm_optimality_tolerance': ipm_optimality_tolerance,
        'log_to_console': disp,
        'mip_max_nodes': mip_max_nodes,
        'output_flag': disp,
        'primal_feasibility_tolerance': primal_feasibility_tolerance,
        'simplex_dual_edge_weight_strategy':
            simplex_dual_edge_weight_strategy_enum,
        'simplex_strategy': s_c.SimplexStrategy.kSimplexStrategyDual,
        'ipm_iteration_limit': maxiter,
        'simplex_iteration_limit': maxiter,
        'mip_rel_gap': mip_rel_gap,
    }
    options.update(unknown_options)

    # np.inf doesn't work; use very large constant
    rhs = _replace_inf(rhs)
    lhs = _replace_inf(lhs)
    lb = _replace_inf(lb)
    ub = _replace_inf(ub)

    if integrality is None or np.sum(integrality) == 0:
        integrality = np.empty(0)
    else:
        integrality = np.array(integrality)

    res = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs,
                         lb, ub, integrality.astype(np.uint8), options)

    # HiGHS represents constraints as lhs/rhs, so
    # Ax + s = b => Ax = b - s
    # and we need to split up s by A_ub and A_eq
    if 'slack' in res:
        slack = res['slack']
        con = np.array(slack[len(b_ub):])
        slack = np.array(slack[:len(b_ub)])
    else:
        slack, con = None, None

    # lagrange multipliers for equalities/inequalities and upper/lower bounds
    if 'lambda' in res:
        lamda = res['lambda']
        marg_ineqlin = np.array(lamda[:len(b_ub)])
        marg_eqlin = np.array(lamda[len(b_ub):])
        marg_upper = np.array(res['marg_bnds'][1, :])
        marg_lower = np.array(res['marg_bnds'][0, :])
    else:
        marg_ineqlin, marg_eqlin = None, None
        marg_upper, marg_lower = None, None

    # this needs to be updated if we start choosing the solver intelligently

    # Convert to scipy-style status and message
    highs_status = res.get('status', None)
    highs_message = res.get('message', None)
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)

    x = res['x'] # is None if not set
    sol = {'x': x,
           'slack': slack,
           'con': con,
           'ineqlin': OptimizeResult({
               'residual': slack,
               'marginals': marg_ineqlin,
           }),
           'eqlin': OptimizeResult({
               'residual': con,
               'marginals': marg_eqlin,
           }),
           'lower': OptimizeResult({
               'residual': None if x is None else x - lb,
               'marginals': marg_lower,
           }),
           'upper': OptimizeResult({
               'residual': None if x is None else ub - x,
               'marginals': marg_upper
            }),
           'fun': res.get('fun'),
           'status': status,
           'success': res['status'] == HighsModelStatus.kOptimal,
           'message': message,
           'nit': res.get('simplex_nit', 0) or res.get('ipm_nit', 0),
           'crossover_nit': res.get('crossover_nit'),
           }

    if np.any(x) and integrality is not None:
        sol.update({
            'mip_node_count': res.get('mip_node_count', 0),
            'mip_dual_bound': res.get('mip_dual_bound', 0.0),
            'mip_gap': res.get('mip_gap', 0.0),
        })

    return sol
<|endoftext|>"""
basinhopping: The basinhopping global optimization algorithm
"""
import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state, _transition_to_rng

__all__ = ['basinhopping']


_params = (inspect.Parameter('res_new', kind=inspect.Parameter.KEYWORD_ONLY),
           inspect.Parameter('res_old', kind=inspect.Parameter.KEYWORD_ONLY))
_new_accept_test_signature = inspect.Signature(parameters=_params)


class Storage:
    """
    Class used to store the lowest energy structure
    """
    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        if minres.success and (minres.fun < self.minres.fun
                               or not self.minres.success):
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        return self.minres


class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether or not to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """
    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp

        self.nstep = 0

        # initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.minimization_failures = 0

        # do initial minimization
        minres = minimizer(self.x)
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        self.incumbent_minres = minres  # best minimize result found so far
        if self.disp:
            print("basinhopping step %d: f %g" % (self.nstep, self.energy))

        # initialize storage class
        self.storage = Storage(minres)

        if hasattr(minres, "nfev"):
            self.res.nfev = minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev = minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        """Do one Monte Carlo iteration

        Randomly displace the coordinates, minimize, and decide whether
        or not to accept the new coordinates.
        """
        # Take a random step.  Make a copy of x because the step_taking
        # algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        # do a local minimization
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")
        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        # accept the move based on self.accept_tests. If any test is False,
        # then reject the step.  If any test returns the special string
        # 'force accept', then accept the step regardless. This can be used
        # to forcefully escape from a local minimum if normal basin hopping
        # steps are not sufficient.
        accept = True
        for test in self.accept_tests:
            if inspect.signature(test) == _new_accept_test_signature:
                testres = test(res_new=minres, res_old=self.incumbent_minres)
            else:
                testres = test(f_new=energy_after_quench, x_new=x_after_quench,
                               f_old=self.energy, x_old=self.x)

            if testres == 'force accept':
                accept = True
                break
            elif testres is None:
                raise ValueError("accept_tests must return True, False, or "
                                 "'force accept'")
            elif not testres:
                accept = False

        # Report the result of the acceptance test to the take step class.
        # This is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, f_new=energy_after_quench,
                                    x_new=x_after_quench, f_old=self.energy,
                                    x_old=self.x)

        return accept, minres

    def one_cycle(self):
        """Do one cycle of the basinhopping algorithm
        """
        self.nstep += 1
        new_global_min = False

        accept, minres = self._monte_carlo_step()

        if accept:
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            self.incumbent_minres = minres  # best minimize result found so far
            new_global_min = self.storage.update(minres)

        # print some information
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                print("found new global minimum on step %d with function"
                      " value %g" % (self.nstep, self.energy))

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = minres.x
        self.energy_trial = minres.fun
        self.accept = accept

        return new_global_min

    def print_report(self, energy_trial, accept):
        """print a status update"""
        minres = self.storage.get_lowest()
        print("basinhopping step %d: f %g trial_f %g accepted %d "
              " lowest_f %g" % (self.nstep, self.energy, energy_trial,
                                accept, minres.fun))


class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
                 verbose=True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            # We're accepting too many steps. This generally means we're
            # trapped in a basin. Take bigger steps.
            self.takestep.stepsize /= self.factor
        else:
            # We're not accepting enough steps. Take smaller steps.
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print(f"adaptive stepsize: acceptance rate {accept_rate:f} target "
                  f"{self.target_accept_rate:f} new stepsize "
                  f"{self.takestep.stepsize:g} old stepsize {old_stepsize:g}")

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        "called by basinhopping to report the result of the step"
        if accept:
            self.naccept += 1


class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator
    """

    def __init__(self, stepsize=0.5, rng=None):
        self.stepsize = stepsize
        self.rng = check_random_state(rng)

    def __call__(self, x):
        x += self.rng.uniform(-self.stepsize, self.stepsize,
                              np.shape(x))
        return x


class MinimizerWrapper:
    """
    wrap a minimizer function as a minimizer class
    """
    def __init__(self, minimizer, func=None, **kwargs):
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)


class Metropolis:
    """Metropolis acceptance criterion.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator used for acceptance test.

    """

    def __init__(self, T, rng=None):
        # Avoid ZeroDivisionError since "MBH can be regarded as a special case
        # of the BH framework with the Metropolis criterion, where temperature
        # T = 0." (Reject all steps that increase energy.)
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.rng = check_random_state(rng)

    def accept_reject(self, res_new, res_old):
        """
        Assuming the local search underlying res_new was successful:
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
        with np.errstate(invalid='ignore'):
            # The energy values being fed to Metropolis are 1-length arrays, and if
            # they are equal, their difference is 0, which gets multiplied by beta,
            # which is inf, and array([0]) * float('inf') causes
            #
            # RuntimeWarning: invalid value encountered in multiply
            #
            # Ignore this warning so when the algorithm is on a flat plane, it always
            # accepts the step, to try to move off the plane.
            prod = -(res_new.fun - res_old.fun) * self.beta
            w = math.exp(min(0, prod))

        rand = self.rng.uniform()
        return w >= rand and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(res_new, res_old))


@_transition_to_rng("seed", position_num=12, replace_doc=True)
def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 rng=None, *, target_accept_rate=0.5, stepwise_factor=0.9):
    """Find the global minimum of a function using the basin-hopping algorithm.

    Basin-hopping is a two-phase method that combines a global stepping
    algorithm with local minimization at each step. Designed to mimic
    the natural process of energy minimization of clusters of atoms, it works
    well for similar problems with "funnel-like, but rugged" energy landscapes
    [5]_.

    As the step-taking, step acceptance, and minimization methods are all
    customizable, this function can also be used to implement other two-phase
    methods.

    Parameters
    ----------
    func : callable ``f(x, *args)``
        Function to be optimized.  ``args`` can be passed as an optional item
        in the dict `minimizer_kwargs`
    x0 : array_like
        Initial guess.
    niter : integer, optional
        The number of basin-hopping iterations. There will be a total of
        ``niter + 1`` runs of the local minimizer.
    T : float, optional
        The "temperature" parameter for the acceptance or rejection criterion.
