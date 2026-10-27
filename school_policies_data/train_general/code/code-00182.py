import inspect
import sys
import warnings

import numpy as np
from numpy import asarray, dot, vdot

from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
from scipy._lib._util import copy_if_needed
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo


__all__ = [
    'broyden1', 'broyden2', 'anderson', 'linearmixing',
    'diagbroyden', 'excitingmixing', 'newton_krylov',
    'BroydenFirst', 'KrylovJacobian', 'InverseJacobian', 'NoConvergence']

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------


class NoConvergence(Exception):
    """Exception raised when nonlinear solver fails to converge within the specified
    `maxiter`."""
    pass


def maxnorm(x):
    return np.absolute(x).max()


def _as_inexact(x):
    """Return `x` as an array, of either floats or complex floats"""
    x = asarray(x)
    if not np.issubdtype(x.dtype, np.inexact):
        return asarray(x, dtype=np.float64)
    return x


def _array_like(x, x0):
    """Return ndarray `x` as same array subclass and shape as `x0`"""
    x = np.reshape(x, np.shape(x0))
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)


def _safe_norm(v):
    if not np.isfinite(v).all():
        return np.array(np.inf)
    return norm(v)

#------------------------------------------------------------------------------
# Generic nonlinear solver machinery
#------------------------------------------------------------------------------


_doc_parts = dict(
    params_basic="""
    F : function(x) -> f
        Function whose root to find; should take and return an array-like
        object.
    xin : array_like
        Initial guess for the solution
    """.strip(),
    params_extra="""
    iter : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    verbose : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make. If more are needed to
        meet convergence, `NoConvergence` is raised.
    f_tol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    f_rtol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    x_tol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    x_rtol : float, optional
        Relative minimum step size. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in the
        direction given by the Jacobian approximation. Defaults to 'armijo'.
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual.

    Returns
    -------
    sol : ndarray
        An array (of similar array type as `x0`) containing the final solution.

    Raises
    ------
    NoConvergence
        When a solution was not found.

    """.strip()
)


def _set_doc(obj):
    if obj.__doc__:
        obj.__doc__ = obj.__doc__ % _doc_parts


def nonlin_solve(F, x0, jacobian='krylov', iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search='armijo', callback=None,
                 full_output=False, raise_exception=True):
    """
    Find a root of a function, in a way suitable for large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    jacobian : Jacobian
        A Jacobian approximation: `Jacobian` object or something that
        `asjacobian` can transform to one. Alternatively, a string specifying
        which of the builtin Jacobian approximations to use:

            krylov, broyden1, broyden2, anderson
            diagbroyden, linearmixing, excitingmixing

    %(params_extra)s
    full_output : bool
        If true, returns a dictionary `info` containing convergence
        information.
    raise_exception : bool
        If True, a `NoConvergence` exception is raise if no solution is found.

    See Also
    --------
    asjacobian, Jacobian

    Notes
    -----
    This algorithm implements the inexact Newton method, with
    backtracking or full line searches. Several Jacobian
    approximations are available, including Krylov and Quasi-Newton
    methods.

    References
    ----------
    .. [KIM] C. T. Kelley, \"Iterative Methods for Linear and Nonlinear
       Equations\". Society for Industrial and Applied Mathematics. (1995)
       https://archive.siam.org/books/kelley/fr16/

    """
    # Can't use default parameters because it's being explicitly passed as None
    # from the calling function, so we need to set it here.
    tol_norm = maxnorm if tol_norm is None else tol_norm
    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    x0 = _as_inexact(x0)
    def func(z):
        return _as_inexact(F(_array_like(z, x0))).flatten()
    x = x0.flatten()

    dx = np.full_like(x, np.inf)
    Fx = func(x)
    Fx_norm = norm(Fx)

    jacobian = asjacobian(jacobian)
    jacobian.setup(x.copy(), Fx, func)

    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    if line_search is True:
        line_search = 'armijo'
    elif line_search is False:
        line_search = None

    if line_search not in (None, 'armijo', 'wolfe'):
        raise ValueError("Invalid line search")

    # Solver tolerance selection
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3

    for n in range(maxiter):
        status = condition.check(Fx, x, dx)
        if status:
            break

        # The tolerance, as computed for scipy.sparse.linalg.* routines
        tol = min(eta, eta*Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        if norm(dx) == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        # Line search, or Newton step
        if line_search:
            s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx,
                                                        line_search)
        else:
            s = 1.0
            x = x + dx
            Fx = func(x)
            Fx_norm_new = norm(Fx)

        jacobian.update(x.copy(), Fx)

        if callback:
            callback(x, Fx)

        # Adjust forcing parameters for inexact methods
        eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma*eta**2))

        Fx_norm = Fx_norm_new

        # Print status
        if verbose:
            sys.stdout.write("%d:  |F(x)| = %g; step %g\n" % (
                n, tol_norm(Fx), s))
            sys.stdout.flush()
    else:
        if raise_exception:
            raise NoConvergence(_array_like(x, x0))
        else:
            status = 2

    if full_output:
        info = {'nit': condition.iteration,
                'fun': Fx,
                'status': status,
                'success': status == 1,
                'message': {1: 'A solution was found at the specified '
                               'tolerance.',
                            2: 'The maximum number of iterations allowed '
                               'has been reached.'
                            }[status]
                }
        return _array_like(x, x0), info
    else:
        return _array_like(x, x0)


_set_doc(nonlin_solve)


def _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-8,
                        smin=1e-2):
    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [norm(Fx)**2]
    s_norm = norm(x) / norm(dx)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s*dx
        v = func(xt)
        p = _safe_norm(v)**2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_Fx[0] = v
        return p

    def derphi(s):
        ds = (abs(s) + s_norm + 1) * rdiff
        return (phi(s+ds, store=False) - phi(s)) / ds

    if search_type == 'wolfe':
        s, phi1, phi0 = scalar_search_wolfe1(phi, derphi, tmp_phi[0],
                                             xtol=1e-2, amin=smin)
    elif search_type == 'armijo':
        s, phi1 = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0],
                                       amin=smin)

    if s is None:
        # XXX: No suitable step length found. Take the full Newton step,
        #      and hope for the best.
        s = 1.0

    x = x + s*dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = norm(Fx)

    return s, x, Fx, Fx_norm


class TerminationCondition:
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """
    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 iter=None, norm=maxnorm):

        if f_tol is None:
            f_tol = np.finfo(np.float64).eps ** (1./3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf

        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol

        self.norm = norm

        self.iter = iter

        self.f0_norm = None
        self.iteration = 0

    def check(self, f, x, dx):
        self.iteration += 1
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)

        if self.f0_norm is None:
            self.f0_norm = f_norm

        if f_norm == 0:
            return 1

        if self.iter is not None:
            # backwards compatibility with SciPy 0.6.0
            return 2 * (self.iteration > self.iter)

        # NB: condition must succeed for rtol=inf even if norm == 0
        return int((f_norm <= self.f_tol
                    and f_norm/self.f_rtol <= self.f0_norm)
                   and (dx_norm <= self.x_tol
                        and dx_norm/self.x_rtol <= x_norm))


#------------------------------------------------------------------------------
# Generic Jacobian approximation
#------------------------------------------------------------------------------

class Jacobian:
    """
    Common interface for Jacobians or Jacobian approximations.

    The optional methods come useful when implementing trust region
    etc., algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.

    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    def __init__(self, **kw):
        names = ["solve", "update", "matvec", "rmatvec", "rsolve",
                 "matmat", "todense", "shape", "dtype"]
        for name, value in kw.items():
            if name not in names:
                raise ValueError(f"Unknown keyword argument {name}")
            if value is not None:
                setattr(self, name, kw[name])


        if hasattr(self, "todense"):
            def __array__(self, dtype=None, copy=None):
                if dtype is not None:
                    raise ValueError(f"`dtype` must be None, was {dtype}")
                return self.todense()

    def aspreconditioner(self):
        return InverseJacobian(self)

    def solve(self, v, tol=0):
        raise NotImplementedError

    def update(self, x, F):
        pass

    def setup(self, x, F, func):
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        if self.__class__.setup is Jacobian.setup:
            # Call on the first point unless overridden
            self.update(x, F)


class InverseJacobian:
    """
    A simple wrapper that inverts the Jacobian using the `solve` method.

    .. legacy:: class

        See the newer, more consistent interfaces in :mod:`scipy.optimize`.

    Parameters
    ----------
    jacobian : Jacobian
        The Jacobian to invert.
    
    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.

    """
    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.matvec = jacobian.solve
        self.update = jacobian.update
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    @property
    def shape(self):
        return self.jacobian.shape

    @property
    def dtype(self):
        return self.jacobian.dtype


def asjacobian(J):
    """
    Convert given object to one suitable for use as a Jacobian.
    """
    spsolve = scipy.sparse.linalg.spsolve
    if isinstance(J, Jacobian):
        return J
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    elif isinstance(J, np.ndarray):
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        J = np.atleast_2d(np.asarray(J))
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')

        return Jacobian(matvec=lambda v: dot(J, v),
                        rmatvec=lambda v: dot(J.conj().T, v),
                        solve=lambda v, tol=0: solve(J, v),
                        rsolve=lambda v, tol=0: solve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif scipy.sparse.issparse(J):
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        return Jacobian(matvec=lambda v: J @ v,
                        rmatvec=lambda v: J.conj().T @ v,
                        solve=lambda v, tol=0: spsolve(J, v),
                        rsolve=lambda v, tol=0: spsolve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        return Jacobian(matvec=getattr(J, 'matvec'),
                        rmatvec=getattr(J, 'rmatvec'),
                        solve=J.solve,
                        rsolve=getattr(J, 'rsolve'),
                        update=getattr(J, 'update'),
                        setup=getattr(J, 'setup'),
                        dtype=J.dtype,
                        shape=J.shape)
    elif callable(J):
        # Assume it's a function J(x) that returns the Jacobian
        class Jac(Jacobian):
            def update(self, x, F):
                self.x = x

            def solve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m, v)
                else:
                    raise ValueError("Unknown matrix type")

            def matvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.issparse(m):
                    return m @ v
                else:
                    raise ValueError("Unknown matrix type")

            def rsolve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError("Unknown matrix type")

            def rmatvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return m.conj().T @ v
                else:
                    raise ValueError("Unknown matrix type")
        return Jac()
    elif isinstance(J, str):
        return dict(broyden1=BroydenFirst,
                    broyden2=BroydenSecond,
                    anderson=Anderson,
                    diagbroyden=DiagBroyden,
                    linearmixing=LinearMixing,
                    excitingmixing=ExcitingMixing,
                    krylov=KrylovJacobian)[J]()
    else:
        raise TypeError('Cannot convert object to a Jacobian')


#------------------------------------------------------------------------------
# Broyden
#------------------------------------------------------------------------------

class GenericBroyden(Jacobian):
    def setup(self, x0, f0, func):
        Jacobian.setup(self, x0, f0, func)
        self.last_f = f0
        self.last_x = x0

        if hasattr(self, 'alpha') and self.alpha is None:
            # Autoscale the initial Jacobian parameter
            # unless we have already guessed the solution.
            normf0 = norm(f0)
            if normf0:
                self.alpha = 0.5*max(norm(x0), 1) / normf0
            else:
                self.alpha = 1.0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        raise NotImplementedError

    def update(self, x, f):
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        self.last_f = f
        self.last_x = x


class LowRankMatrix:
    r"""
    A matrix represented as

    .. math:: \alpha I + \sum_{n=0}^{n=M} c_n d_n^\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """

    def __init__(self, alpha, n, dtype):
        self.alpha = alpha
        self.cs = []
        self.ds = []
        self.n = n
        self.dtype = dtype
        self.collapsed = None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        axpy, scal, dotc = get_blas_funcs(['axpy', 'scal', 'dotc'],
                                          cs[:1] + [v])
        w = alpha * v
        for c, d in zip(cs, ds):
            a = dotc(d, v)
            w = axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        """Evaluate w = M^-1 v"""
        if len(cs) == 0:
            return v/alpha

        # (B + C D^H)^-1 = B^-1 - B^-1 C (I + D^H B^-1 C)^-1 D^H B^-1

        axpy, dotc = get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])

        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)
        for i, d in enumerate(ds):
            for j, c in enumerate(cs):
                A[i,j] += dotc(d, c)

        q = np.zeros(len(cs), dtype=c0.dtype)
        for j, d in enumerate(ds):
            q[j] = dotc(d, v)
        q /= alpha
        q = solve(A, q)

        w = v/alpha
        for c, qc in zip(cs, q):
            w = axpy(c, w, w.size, -qc)

        return w

    def matvec(self, v):
        """Evaluate w = M v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v):
        """Evaluate w = M^H v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)

    def solve(self, v, tol=0):
        """Evaluate w = M^-1 v"""
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v, tol=0):
        """Evaluate w = M^-H v"""
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    def append(self, c, d):
        if self.collapsed is not None:
            self.collapsed += c[:,None] * d[None,:].conj()
            return

        self.cs.append(c)
        self.ds.append(d)

        if len(self.cs) > c.size:
            self.collapse()

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            warnings.warn("LowRankMatrix is scipy-internal code, `dtype` "
                          f"should only be None but was {dtype} (not handled)",
                          stacklevel=3)
        if copy is not None:
            warnings.warn("LowRankMatrix is scipy-internal code, `copy` "
                          f"should only be None but was {copy} (not handled)",
                          stacklevel=3)
        if self.collapsed is not None:
            return self.collapsed

        Gm = self.alpha*np.identity(self.n, dtype=self.dtype)
        for c, d in zip(self.cs, self.ds):
            Gm += c[:,None]*d[None,:].conj()
        return Gm

    def collapse(self):
        """Collapse the low-rank matrix to a full-rank one."""
        self.collapsed = np.array(self, copy=copy_if_needed)
        self.cs = None
        self.ds = None
        self.alpha = None

    def restart_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping all vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    def simple_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]

    def svd_reduce(self, max_rank, to_retain=None):
        """
        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the \"Broyden Rank Reduction Inverse\"
        algorithm described in [1]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [1] B.A. van der Rotten, PhD thesis,
           \"A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations\". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).

           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        """
        if self.collapsed is not None:
            return

        p = max_rank
        if to_retain is not None:
            q = to_retain
        else:
            q = p - 2

        if self.cs:
            p = min(p, len(self.cs[0]))
        q = max(0, min(q, p-1))

        m = len(self.cs)
        if m < p:
            # nothing to do
            return

        C = np.array(self.cs).T
        D = np.array(self.ds).T

        D, R = qr(D, mode='economic')
        C = dot(C, R.T.conj())

        U, S, WH = svd(C, full_matrices=False)

        C = dot(C, inv(WH))
        D = dot(D, WH.T.conj())

        for k in range(q):
            self.cs[k] = C[:,k].copy()
            self.ds[k] = D[:,k].copy()

        del self.cs[q:]
        del self.ds[q:]


_doc_parts['broyden_params'] = """
    alpha : float, optional
        Initial guess for the Jacobian is ``(-1/alpha)``.
    reduction_method : str or tuple, optional
        Method used in ensuring that the rank of the Broyden matrix
        stays low. Can either be a string giving the name of the method,
        or a tuple of the form ``(method, param1, param2, ...)``
        that gives the name of the method and values for additional parameters.

        Methods available:

        - ``restart``: drop all matrix columns. Has no extra parameters.
        - ``simple``: drop oldest matrix column. Has no extra parameters.
        - ``svd``: keep only the most significant SVD components.
          Takes an extra parameter, ``to_retain``, which determines the
          number of SVD components to retain when rank reduction is done.
          Default is ``max_rank - 2``.

    max_rank : int, optional
        Maximum rank for the Broyden matrix.
        Default is infinity (i.e., no rank reduction).
    """.strip()


class BroydenFirst(GenericBroyden):
    """
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as "Broyden's good method".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden1'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) dx^\\dagger H / ( dx^\\dagger H df)

    which corresponds to Broyden's first Jacobian update

    .. math:: J_+ = J + (df - J dx) dx^\\dagger / dx^\\dagger dx


    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       "A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).
       https://math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden1(fun, [0, 0])
    >>> sol
    array([0.84116396, 0.15883641])

    """

    def __init__(self, alpha=None, reduction_method='restart', max_rank=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.Gm = None

        if max_rank is None:
            max_rank = np.inf
        self.max_rank = max_rank

        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        reduce_params = (max_rank - 1,) + reduce_params

        if reduction_method == 'svd':
            self._reduce = lambda: self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda: self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda: self.Gm.restart_reduce(*reduce_params)
        else:
            raise ValueError(f"Unknown rank reduction method '{reduction_method}'")

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        return inv(self.Gm)

    def solve(self, f, tol=0):
        r = self.Gm.matvec(f)
        if not np.isfinite(r).all():
            # singular; reset the Jacobian approximation
            self.setup(self.last_x, self.last_f, self.func)
            return self.Gm.matvec(f)
        return r

    def matvec(self, f):
        return self.Gm.solve(f)

    def rsolve(self, f, tol=0):
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()  # reduce first to preserve secant condition

        v = self.Gm.rmatvec(dx)
        c = dx - self.Gm.matvec(df)
        d = v / vdot(df, v)

        self.Gm.append(c, d)


class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden2'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) df^\\dagger / ( df^\\dagger df)

    corresponding to Broyden's second method.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden2(fun, [0, 0])
    >>> sol
    array([0.84116365, 0.15883529])

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()  # reduce first to preserve secant condition

        v = df
        c = dx - self.Gm.matvec(df)
        d = v / df_norm**2
        self.Gm.append(c, d)


#------------------------------------------------------------------------------
# Broyden-like (restricted memory)
#------------------------------------------------------------------------------

class Anderson(GenericBroyden):
    """
    Find a root of a function, using (extended) Anderson mixing.

    The Jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversions and MxN multiplications are required. [Ey]_

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter for numerical stability.
        Compared to unity, good values of the order of 0.01.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='anderson'`` in particular.

    References
    ----------
    .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.anderson(fun, [0, 0])
    >>> sol
    array([0.84116588, 0.15883789])

    """

    # Note:
    #
    # Anderson method maintains a rank M approximation of the inverse Jacobian,
    #
    #     J^-1 v ~ -v*alpha + (dX + alpha dF) A^-1 dF^H v
    #     A      = W + dF^H dF
    #     W      = w0^2 diag(dF^H dF)
    #
    # so that for w0 = 0 the secant condition applies for last M iterates, i.e.,
    #
    #     J^-1 df_j = dx_j
    #
    # for all j = 0 ... M-1.
    #
    # Moreover, (from Sherman-Morrison-Woodbury formula)
    #
    #    J v ~ [ b I - b^2 C (I + b dF^H A^-1 C)^-1 dF^H ] v
    #    C   = (dX + alpha dF) A^-1
    #    b   = -1/alpha
    #
    # and after simplification
    #
    #    J v ~ -v/alpha + (dX/alpha + dF) (dF^H dX - alpha W)^-1 dF^H v
    #

    def __init__(self, alpha=None, w0=0.01, M=5):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f, tol=0):
        dx = -self.alpha*f

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        try:
            gamma = solve(self.a, df_f)
        except LinAlgError:
            # singular; reset the Jacobian approximation
            del self.dx[:]
            del self.df[:]
            return dx

        for m in range(n):
            dx += gamma[m]*(self.dx[m] + self.alpha*self.df[m])
        return dx

    def matvec(self, f):
        dx = -f/self.alpha

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        b = np.empty((n, n), dtype=f.dtype)
        for i in range(n):
            for j in range(n):
                b[i,j] = vdot(self.df[i], self.dx[j])
                if i == j and self.w0 != 0:
                    b[i,j] -= vdot(self.df[i], self.df[i])*self.w0**2*self.alpha
        gamma = solve(b, df_f)

        for m in range(n):
            dx += gamma[m]*(self.df[m] + self.dx[m]/self.alpha)
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if self.M == 0:
            return

        self.dx.append(dx)
        self.df.append(df)

        while len(self.dx) > self.M:
            self.dx.pop(0)
            self.df.pop(0)

        n = len(self.dx)
        a = np.zeros((n, n), dtype=f.dtype)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1+wd)*vdot(self.df[i], self.df[j])

        a += np.triu(a, 1).T.conj()
        self.a = a

#------------------------------------------------------------------------------
# Simple iterations
#------------------------------------------------------------------------------


class DiagBroyden(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.

    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='diagbroyden'`` in particular.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.diagbroyden(fun, [0, 0])
    >>> sol
    array([0.84116403, 0.15883384])

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.d = np.full((self.shape[0],), 1 / self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f / self.d

    def matvec(self, f):
        return -f * self.d

    def rsolve(self, f, tol=0):
        return -f / self.d.conj()

    def rmatvec(self, f):
        return -f * self.d.conj()

    def todense(self):
        return np.diag(-self.d)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d*dx)*dx/dx_norm**2


class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='linearmixing'`` in particular.

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def solve(self, f, tol=0):
        return -f*self.alpha

    def matvec(self, f):
        return -f/self.alpha

    def rsolve(self, f, tol=0):
        return -f*np.conj(self.alpha)

    def rmatvec(self, f):
        return -f/np.conj(self.alpha)

    def todense(self):
        return np.diag(np.full(self.shape[0], -1/self.alpha))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass


class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='excitingmixing'`` in particular.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, alpha=None, alphamax=1.0):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.beta = np.full((self.shape[0],), self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f*self.beta

    def matvec(self, f):
        return -f/self.beta

    def rsolve(self, f, tol=0):
        return -f*self.beta.conj()

    def rmatvec(self, f):
        return -f/self.beta.conj()

    def todense(self):
        return np.diag(-1/self.beta)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        incr = f*self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)


#------------------------------------------------------------------------------
# Iterative/Krylov approximated Jacobians
#------------------------------------------------------------------------------

class KrylovJacobian(Jacobian):
    """
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : str or callable, optional
        Krylov method to use to approximate the Jacobian.  Can be a string,
        or a function implementing the same interface as the iterative
        solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
        ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
        ``'tfqmr'``.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_maxiter : int, optional
        Parameter to pass to the "inner" Krylov solver: maximum number of
        iterations. Iteration will stop after maxiter steps even if the
        specified tolerance has not been achieved.
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> from scipy.optimize import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    inner_kwargs : kwargs
        Keyword parameters for the "inner" Krylov solver
        (defined with `method`). Parameter names must start with
        the `inner_` prefix which will be stripped before passing on
        the inner method. See, e.g., `scipy.sparse.linalg.gmres` for details.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='krylov'`` in particular.
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by a finite difference:

    .. math:: J v \\approx (f(x + \\omega*v/|v|) - f(x)) / \\omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    References
    ----------
    .. [1] C. T. Kelley, Solving Nonlinear Equations with Newton's Method,
           SIAM, pp.57-83, 2003.
           :doi:`10.1137/1.9780898718898.ch3`
    .. [2] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [3] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.newton_krylov(fun, [0, 0])
    >>> sol
    array([0.66731771, 0.66536458])

    """

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=10, **kw):
        self.preconditioner = inner_M
        self.rdiff = rdiff
        # Note that this retrieves one of the named functions, or otherwise
        # uses `method` as is (i.e., for a user-provided callable).
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            tfqmr=scipy.sparse.linalg.tfqmr,
            ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

        if self.method is scipy.sparse.linalg.gmres:
            # Replace GMRES's outer iteration with Newton steps
            self.method_kw['restart'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method in (scipy.sparse.linalg.gcrotmk,
                             scipy.sparse.linalg.bicgstab,
                             scipy.sparse.linalg.cgs):
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            # Replace LGMRES's outer iteration with Newton steps
            self.method_kw['maxiter'] = 1
            # Carry LGMRES's `outer_v` vectors across nonlinear iterations
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            # But don't carry the corresponding Jacobian*v products, in case
            # the Jacobian changes a lot in the nonlinear step
            #
            # XXX: some trust-region inspired ideas might be more efficient...
            #      See e.g., Brown & Saad. But needs to be implemented separately
            #      since it's not an inexact Newton method.
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)

        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError(f"Unknown parameter {key}")
            self.method_kw[key[6:]] = value

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if 'rtol' in self.method_kw:
            sol, info = self.method(self.op, rhs, **self.method_kw)
        else:
            sol, info = self.method(self.op, rhs, rtol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()

        # Update also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)

        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1./2)

        self._update_diff_step()

        # Setup also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)


#------------------------------------------------------------------------------
# Wrapper functions
#------------------------------------------------------------------------------

def _nonlin_wrapper(name, jac):
    """
    Construct a solver wrapper with given name and Jacobian approx.

    It inspects the keyword arguments of ``jac.__init__``, and allows to
    use the same arguments in the wrapper function, in addition to the
    keyword arguments of `nonlin_solve`

    """
    signature = _getfullargspec(jac.__init__)
    args, varargs, varkw, defaults, kwonlyargs, kwdefaults, _ = signature
    kwargs = list(zip(args[-len(defaults):], defaults))
    kw_str = ", ".join([f"{k}={v!r}" for k, v in kwargs])
    if kw_str:
        kw_str = ", " + kw_str
    kwkw_str = ", ".join([f"{k}={k}" for k, v in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ", "
    if kwonlyargs:
        raise ValueError(f'Unexpected signature {signature}')

    # Construct the wrapper function so that its keyword arguments
    # are visible in pydoc.help etc.
    wrapper = """
def %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None,
             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
             tol_norm=None, line_search='armijo', callback=None, **kw):
    jac = %(jac)s(%(kwkw)s **kw)
    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,
                        callback)
"""

    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__,
                             kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    exec(wrapper, ns)
    func = ns[name]
    func.__doc__ = jac.__doc__
    _set_doc(func)
    return func


broyden1 = _nonlin_wrapper('broyden1', BroydenFirst)
broyden2 = _nonlin_wrapper('broyden2', BroydenSecond)
anderson = _nonlin_wrapper('anderson', Anderson)
linearmixing = _nonlin_wrapper('linearmixing', LinearMixing)
diagbroyden = _nonlin_wrapper('diagbroyden', DiagBroyden)
excitingmixing = _nonlin_wrapper('excitingmixing', ExcitingMixing)
newton_krylov = _nonlin_wrapper('newton_krylov', KrylovJacobian)
<|endoftext|>"""Dog-leg trust-region optimization."""
import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = []


def _minimize_dogleg(fun, x0, args=(), jac=None, hess=None,
                     **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the dog-leg trust-region algorithm.

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
        Gradient norm must be less than `gtol` before successful
        termination.

    """
    if jac is None:
        raise ValueError('Jacobian is required for dogleg minimization')
    if not callable(hess):
        raise ValueError('Hessian is required for dogleg minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=DoglegSubproblem,
                                  **trust_region_options)


class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""

    def cauchy_point(self):
        """
        The Cauchy point is minimal along the direction of steepest descent.
        """
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
        return self._cauchy_point

    def newton_point(self):
        """
        The Newton point is a global minimum of the approximate function.
        """
        if self._newton_point is None:
            g = self.jac
            B = self.hess
            cho_info = scipy.linalg.cho_factor(B)
            self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
        return self._newton_point

    def solve(self, trust_radius):
        """
        Minimize a function using the dog-leg trust-region algorithm.

        This algorithm requires function values and first and second derivatives.
        It also performs a costly Hessian decomposition for most iterations,
        and the Hessian is required to be positive definite.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        The Hessian is required to be positive definite.

        References
        ----------
        .. [1] Jorge Nocedal and Stephen Wright,
               Numerical Optimization, second edition,
               Springer-Verlag, 2006, page 73.
        """

        # Compute the Newton point.
        # This is the optimum for the quadratic model function.
        # If it is inside the trust radius then return this point.
        p_best = self.newton_point()
        if scipy.linalg.norm(p_best) < trust_radius:
            hits_boundary = False
            return p_best, hits_boundary

        # Compute the Cauchy point.
        # This is the predicted optimum along the direction of steepest descent.
        p_u = self.cauchy_point()

        # If the Cauchy point is outside the trust region,
        # then return the point where the path intersects the boundary.
        p_u_norm = scipy.linalg.norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return p_boundary, hits_boundary

        # Compute the intersection of the trust region boundary
        # and the line segment connecting the Cauchy and Newton points.
        # This requires solving a quadratic equation.
        # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
        # Solve this for positive time t using the quadratic formula.
        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u,
                                                  trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return p_boundary, hits_boundary
<|endoftext|>"""
Functions
---------
.. autosummary::
   :toctree: generated/

    fmin_l_bfgs_b

"""

## License for the Python wrapper
## ==============================

## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.

## Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy

import numpy as np
from numpy import array, asarray, float64, zeros
from . import _lbfgsb
from ._optimize import (MemoizeJac, OptimizeResult, _call_callback_maybe_halt,
                        _wrap_callback, _check_unknown_options,
                        _prepare_scalar_function)
from ._constraints import old_bound_to_new

from scipy.sparse.linalg import LinearOperator

__all__ = ['fmin_l_bfgs_b', 'LbfgsInvHessProduct']


status_messages = {
    0 : "START",
    1 : "NEW_X",
    2 : "RESTART",
    3 : "FG",
    4 : "CONVERGENCE",
    5 : "STOP",
    6 : "WARNING",
    7 : "ERROR",
    8 : "ABNORMAL"
}


task_messages = {
    0 : "",
    301 : "",
    302 : "",
    401 : "NORM OF PROJECTED GRADIENT <= PGTOL",
    402 : "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    501 : "CPU EXCEEDING THE TIME LIMIT",
    502 : "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    503 : "PROJECTED GRADIENT IS SUFFICIENTLY SMALL",
    504 : "TOTAL NO. OF ITERATIONS REACHED LIMIT",
    505 : "CALLBACK REQUESTED HALT",
    601 : "ROUNDING ERRORS PREVENT PROGRESS",
    602 : "STP = STPMAX",
    603 : "STP = STPMIN",
    604 : "XTOL TEST SATISFIED",
    701 : "NO FEASIBLE SOLUTION",
    702 : "FACTR < 0",
    703 : "FTOL < 0",
    704 : "GTOL < 0",
    705 : "XTOL < 0",
    706 : "STP < STPMIN",
    707 : "STP > STPMAX",
    708 : "STPMIN < 0",
    709 : "STPMAX < STPMIN",
    710 : "INITIAL G >= 0",
    711 : "M <= 0",
    712 : "N <= 0",
    713 : "INVALID NBD",
}

def fmin_l_bfgs_b(func, x0, fprime=None, args=(),
                  approx_grad=0,
                  bounds=None, m=10, factr=1e7, pgtol=1e-5,
                  epsilon=1e-8,
                  iprint=-1, maxfun=15000, maxiter=15000, disp=None,
                  callback=None, maxls=20):
    """
    Minimize a function func using the L-BFGS-B algorithm.

    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimize.
    x0 : ndarray
        Initial guess.
    fprime : callable fprime(x,*args), optional
        The gradient of `func`. If None, then `func` returns the function
        value and the gradient (``f, g = func(x, *args)``), unless
        `approx_grad` is True in which case `func` returns only ``f``.
    args : sequence, optional
        Arguments to pass to `func` and `fprime`.
    approx_grad : bool, optional
        Whether to approximate the gradient numerically (in which case
        `func` returns only the function value).
    bounds : list, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None or +-inf for one of ``min`` or
        ``max`` when there is no bound in that direction.
    m : int, optional
        The maximum number of variable metric corrections
        used to define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms in an
        approximation to it.)
    factr : float, optional
        The iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
        where ``eps`` is the machine precision, which is automatically
        generated by the code. Typical values for `factr` are: 1e12 for
        low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
        high accuracy. See Notes for relationship to `ftol`, which is exposed
        (instead of `factr`) by the `scipy.optimize.minimize` interface to
        L-BFGS-B.
    pgtol : float, optional
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n} <= pgtol``
        where ``proj g_i`` is the i-th component of the projected gradient.
    epsilon : float, optional
        Step size used when `approx_grad` is True, for numerically
        calculating the gradient
    iprint : int, optional
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.17.0.

    disp : int, optional
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.17.0.

    maxfun : int, optional
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
    maxiter : int, optional
        Maximum number of iterations.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.

    Returns
    -------
    x : array_like
        Estimated position of the minimum.
    f : float
        Value of `func` at the minimum.
    d : dict
        Information dictionary.

        * d['warnflag'] is

          - 0 if converged,
          - 1 if too many function evaluations or too many iterations,
          - 2 if stopped for another reason, given in d['task']

        * d['grad'] is the gradient at the minimum (should be 0 ish)
        * d['funcalls'] is the number of function calls made.
        * d['nit'] is the number of iterations.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'L-BFGS-B' `method` in particular. Note that the
        `ftol` option is made available via that interface, while `factr` is
        provided via this interface, where `factr` is the factor multiplying
        the default machine floating-point precision to arrive at `ftol`:
        ``ftol = factr * numpy.finfo(float).eps``.

    Notes
    -----
    SciPy uses a C-translated and modified version of the Fortran code,
    L-BFGS-B v3.0 (released April 25, 2011, BSD-3 licensed). Original Fortran
    version was written by Ciyou Zhu, Richard Byrd, Jorge Nocedal and,
    Jose Luis Morales.

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.

    Examples
    --------
    Solve a linear regression problem via `fmin_l_bfgs_b`. To do this, first we
    define an objective function ``f(m, b) = (y - y_model)**2``, where `y`
    describes the observations and `y_model` the prediction of the linear model
    as ``y_model = m*x + b``. The bounds for the parameters, ``m`` and ``b``,
    are arbitrarily chosen as ``(0,5)`` and ``(5,10)`` for this example.

    >>> import numpy as np
    >>> from scipy.optimize import fmin_l_bfgs_b
    >>> X = np.arange(0, 10, 1)
    >>> M = 2
    >>> B = 3
    >>> Y = M * X + B
    >>> def func(parameters, *args):
    ...     x = args[0]
    ...     y = args[1]
    ...     m, b = parameters
    ...     y_model = m*x + b
    ...     error = sum(np.power((y - y_model), 2))
    ...     return error

    >>> initial_values = np.array([0.0, 1.0])

    >>> x_opt, f_opt, info = fmin_l_bfgs_b(func, x0=initial_values, args=(X, Y),
    ...                                    approx_grad=True)
    >>> x_opt, f_opt
    array([1.99999999, 3.00000006]), 1.7746231151323805e-14  # may vary

    The optimized parameters in ``x_opt`` agree with the ground truth parameters
    ``m`` and ``b``. Next, let us perform a bound constrained optimization using
    the `bounds` parameter.

    >>> bounds = [(0, 5), (5, 10)]
    >>> x_opt, f_op, info = fmin_l_bfgs_b(func, x0=initial_values, args=(X, Y),
    ...                                   approx_grad=True, bounds=bounds)
    >>> x_opt, f_opt
    array([1.65990508, 5.31649385]), 15.721334516453945  # may vary
    """
    # handle fprime/approx_grad
    if approx_grad:
        fun = func
        jac = None
    elif fprime is None:
        fun = MemoizeJac(func)
        jac = fun.derivative
    else:
        fun = func
        jac = fprime

    # build options
    callback = _wrap_callback(callback)
    opts = {'maxcor': m,
            'ftol': factr * np.finfo(float).eps,
            'gtol': pgtol,
            'eps': epsilon,
            'maxfun': maxfun,
            'maxiter': maxiter,
            'callback': callback,
            'maxls': maxls}

    res = _minimize_lbfgsb(fun, x0, args=args, jac=jac, bounds=bounds,
                           **opts)
    d = {'grad': res['jac'],
         'task': res['message'],
         'funcalls': res['nfev'],
         'nit': res['nit'],
         'warnflag': res['status']}
    f = res['fun']
    x = res['x']

    return x, f, d


def _minimize_lbfgsb(fun, x0, args=(), jac=None, bounds=None,
                     disp=None, maxcor=10, ftol=2.2204460492503131e-09,
                     gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
                     iprint=-1, callback=None, maxls=20,
                     finite_diff_rel_step=None, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.

    Options
    -------
    disp : None or int
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.17.0.

    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``proj g_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Deprecated option that previously controlled the text printed on the
        screen during the problem solution. Now the code does not emit any
        output and this keyword has no function.

        .. deprecated:: 1.15.0
            This keyword is deprecated and will be removed from SciPy 1.17.0.

    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.

    """
    _check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps

    x0 = asarray(x0).ravel()
    n, = x0.shape

    # historically old-style bounds were/are expected by lbfgsb.
    # That's still the case but we'll deal with new-style from here on,
    # it's easier
    if bounds is None:
        pass
    elif len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    else:
        bounds = np.array(old_bound_to_new(bounds))

        # check bounds
        if (bounds[0] > bounds[1]).any():
            raise ValueError(
                "LBFGSB - one of the lower bounds is greater than an upper bound."
            )

        # initial vector must lie within the bounds. Otherwise ScalarFunction and
        # approx_derivative will cause problems
        x0 = np.clip(x0, bounds[0], bounds[1])

    # _prepare_scalar_function can use bounds=None to represent no bounds
    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  bounds=bounds,
                                  finite_diff_rel_step=finite_diff_rel_step)

    func_and_grad = sf.fun_and_grad

    nbd = zeros(n, np.int32)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(-np.inf, np.inf): 0,
                  (1, np.inf): 1,
                  (1, 1): 2,
                  (-np.inf, 1): 3}

    if bounds is not None:
        for i in range(0, n):
            L, U = bounds[0, i], bounds[1, i]
            if not np.isinf(L):
                low_bnd[i] = L
                L = 1
            if not np.isinf(U):
                upper_bnd[i] = U
                U = 1
            nbd[i] = bounds_map[L, U]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    x = array(x0, dtype=np.float64)
    f = array(0.0, dtype=np.int32)
    g = zeros((n,), dtype=np.int32)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
    iwa = zeros(3*n, dtype=np.int32)
    task = zeros(2, dtype=np.int32)
    ln_task = zeros(2, dtype=np.int32)
    lsave = zeros(4, dtype=np.int32)
    isave = zeros(44, dtype=np.int32)
    dsave = zeros(29, dtype=float64)

    n_iterations = 0

    while True:
        # g may become float32 if a user provides a function that calculates
        # the Jacobian in float32 (see gh-18730). The underlying code expects
        # float64, so upcast it
        g = g.astype(np.float64)
        # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa,
                       iwa, task, lsave, isave, dsave, maxls, ln_task)

        if task[0] == 3:
            # The minimization routine wants f and g at the current x.
            # Note that interruptions due to maxfun are postponed
            # until the completion of the current minimization iteration.
            # Overwrite f and g:
            f, g = func_and_grad(x)
        elif task[0] == 1:
            # new iteration
            n_iterations += 1

            intermediate_result = OptimizeResult(x=x, fun=f)
            if _call_callback_maybe_halt(callback, intermediate_result):
                task[0] = 5
                task[1] = 505
            if n_iterations >= maxiter:
                task[0] = 5
                task[1] = 504
            elif sf.nfev > maxfun:
                task[0] = 5
                task[1] = 502
        else:
            break

    if task[0] == 4:
        warnflag = 0
    elif sf.nfev > maxfun or n_iterations >= maxiter:
        warnflag = 1
    else:
        warnflag = 2

    # These two portions of the workspace are described in the mainlb
    # function docstring in "__lbfgsb.c", ws and wy arguments.
    s = wa[0: m*n].reshape(m, n)
    y = wa[m*n: 2*m*n].reshape(m, n)

    # isave(31) = the total number of BFGS updates prior the current iteration.
    n_bfgs_updates = isave[30]

    n_corrs = min(n_bfgs_updates, maxcor)
    hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

    msg = status_messages[task[0]] + ": " + task_messages[task[1]]

    return OptimizeResult(fun=f, jac=g, nfev=sf.nfev,
                          njev=sf.ngev,
                          nit=n_iterations, status=warnflag, message=msg,
                          x=x, success=(warnflag == 0), hess_inv=hess_inv)


class LbfgsInvHessProduct(LinearOperator):
    """Linear operator for the L-BFGS approximate inverse Hessian.

    This operator computes the product of a vector with the approximate inverse
    of the Hessian of the objective function, using the L-BFGS limited
    memory approximation to the inverse Hessian, accumulated during the
    optimization.

    Objects of this class implement the ``scipy.sparse.linalg.LinearOperator``
    interface.

    Parameters
    ----------
    sk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the solution vector.
        (See [1]).
    yk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the gradient. (See [1]).

    References
    ----------
    .. [1] Nocedal, Jorge. "Updating quasi-Newton matrices with limited
       storage." Mathematics of computation 35.151 (1980): 773-782.

    """

    def __init__(self, sk, yk):
        """Construct the operator."""
        if sk.shape != yk.shape or sk.ndim != 2:
            raise ValueError('sk and yk must have matching shape, (n_corrs, n)')
        n_corrs, n = sk.shape

        super().__init__(dtype=np.float64, shape=(n, n))

        self.sk = sk
        self.yk = yk
        self.n_corrs = n_corrs
        self.rho = 1 / np.einsum('ij,ij->i', sk, yk)

    def _matvec(self, x):
        """Efficient matrix-vector multiply with the BFGS matrices.

        This calculation is described in Section (4) of [1].

        Parameters
        ----------
        x : ndarray
            An array with shape (n,) or (n,1).

        Returns
        -------
        y : ndarray
            The matrix-vector product

        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        q = np.array(x, dtype=self.dtype, copy=True)
        if q.ndim == 2 and q.shape[1] == 1:
            q = q.reshape(-1)

        alpha = np.empty(n_corrs)

        for i in range(n_corrs-1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q = q - alpha[i]*y[i]

        r = q
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)

        return r

    def todense(self):
        """Return a dense array representation of this operator.

        Returns
        -------
        arr : ndarray, shape=(n, n)
            An array with the same shape and containing
            the same data represented by this `LinearOperator`.

        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        I_arr = np.eye(*self.shape, dtype=self.dtype)
        Hk = I_arr

        for i in range(n_corrs):
            A1 = I_arr - s[i][:, np.newaxis] * y[i][np.newaxis, :] * rho[i]
            A2 = I_arr - y[i][:, np.newaxis] * s[i][np.newaxis, :] * rho[i]

            Hk = np.dot(A1, np.dot(Hk, A2)) + (rho[i] * s[i][:, np.newaxis] *
                                                        s[i][np.newaxis, :])
        return Hk
<|endoftext|>import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx


FD_METHODS = ('2-point', '3-point', 'cs')


def _wrapper_fun(fun, args=()):
    ncalls = [0]

    def wrapped(x):
        ncalls[0] += 1
        # Send a copy because the user may overwrite it.
        # Overwriting results in undefined behaviour because
        # fun(self.x) will change self.x, with the two no longer linked.
        fx = fun(np.copy(x), *args)
        # Make sure the function returns a true scalar
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function "
                    "must return a scalar value."
                ) from e
        return fx
    return wrapped, ncalls


def _wrapper_grad(grad, fun=None, args=(), finite_diff_options=None):
    ncalls = [0]

    if callable(grad):
        def wrapped(x, **kwds):
            # kwds present to give function same signature as numdiff variant
            ncalls[0] += 1
            return np.atleast_1d(grad(np.copy(x), *args))
        return wrapped, ncalls

    elif grad in FD_METHODS:
        def wrapped1(x, f0=None):
            ncalls[0] += 1
            return approx_derivative(
                fun, x, f0=f0, **finite_diff_options
            )

        return wrapped1, ncalls


def _wrapper_hess(hess, grad=None, x0=None, args=(), finite_diff_options=None):
    if callable(hess):
        H = hess(np.copy(x0), *args)
        ncalls = [1]

        if sps.issparse(H):
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return sps.csr_matrix(hess(np.copy(x), *args))

            H = sps.csr_matrix(H)

        elif isinstance(H, LinearOperator):
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return hess(np.copy(x), *args)

        else:  # dense
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return np.atleast_2d(np.asarray(hess(np.copy(x), *args)))

            H = np.atleast_2d(np.asarray(H))

        return wrapped, ncalls, H
    elif hess in FD_METHODS:
        ncalls = [0]

        def wrapped1(x, f0=None):
            return approx_derivative(
                grad, x, f0=f0, **finite_diff_options
            )

        return wrapped1, ncalls, None


class ScalarFunction:
    """Scalar function and its derivatives.

    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.

    Parameters
    ----------
    fun : callable
        evaluates the scalar function. Must be of the form ``fun(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args`` is
        a tuple of any additional fixed parameters needed to completely specify
        the function. Should return a scalar.
    x0 : array-like
        Provides an initial set of variables for evaluating fun. Array of real
        elements of size (n,), where 'n' is the number of independent
        variables.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the scalar
        function.
    grad : {callable, '2-point', '3-point', 'cs'}
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``grad(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}
        Method for computing the Hessian matrix. If it is callable, it should
        return the  Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
        select a finite difference scheme for numerical estimation. Or, objects
        implementing `HessianUpdateStrategy` interface can be used to
        approximate the Hessian.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.
    finite_diff_rel_step : None or array_like
        Relative step size to use. The absolute step size is computed as
        ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
        adjusted to fit into the bounds. For ``method='3-point'`` the sign
        of `h` is ignored. If None then finite_diff_rel_step is selected
        automatically,
    finite_diff_bounds : tuple of array_like
        Lower and upper bounds on independent variables. Defaults to no bounds,
        (-np.inf, np.inf). Each bound must match the size of `x0` or be a
        scalar, in the latter case the bound will be the same for all
        variables. Use it to limit the range of function evaluation.
    epsilon : None or array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `epsilon` is ignored. By default
        relative steps are used, only if ``epsilon is not None`` are absolute
        steps used.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `grad` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step,
                 finite_diff_bounds, epsilon=None):
        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(
                f"`grad` must be either callable or one of {FD_METHODS}."
            )

        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError(
                f"`hess` must be either callable, HessianUpdateStrategy"
                f" or one of {FD_METHODS}."
            )

        if grad in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the gradient is estimated via "
                             "finite-differences, we require the Hessian "
                             "to be estimated using one of the "
                             "quasi-Newton strategies.")

        self.xp = xp = array_namespace(x0)
        _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # original arguments
        self._wrapped_fun, self._nfev = _wrapper_fun(fun, args=args)
        self._orig_fun = fun
        self._orig_grad = grad
        self._orig_hess = hess
        self._args = args

        # promotes to floating
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype
        self.n = self.x.size
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        self._lowest_x = None
        self._lowest_f = np.inf

        finite_diff_options = {}
        if grad in FD_METHODS:
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["as_linear_operator"] = True

        # Initial function evaluation
        self._update_fun()

        # Initial gradient evaluation
        self._wrapped_grad, self._ngev = _wrapper_grad(
            grad,
            fun=self._wrapped_fun,
            args=args,
            finite_diff_options=finite_diff_options
        )
        self._update_grad()

        # Hessian evaluation
        if callable(hess):
            self._wrapped_hess, self._nhev, self.H = _wrapper_hess(
                hess, x0=x0, args=args
            )
            self.H_updated = True
        elif hess in FD_METHODS:
            self._wrapped_hess, self._nhev, self.H = _wrapper_hess(
                hess,
                grad=self._wrapped_grad,
                x0=x0,
                finite_diff_options=finite_diff_options
            )
            self._update_grad()
            self.H = self._wrapped_hess(self.x, f0=self.g)
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.g_prev = None
            self._nhev = [0]

    @property
    def nfev(self):
        return self._nfev[0]

    @property
    def ngev(self):
        return self._ngev[0]

    @property
    def nhev(self):
        return self._nhev[0]

    def _update_x(self, x):
        if isinstance(self._orig_hess, HessianUpdateStrategy):
            self._update_grad()
            self.x_prev = self.x
            self.g_prev = self.g
            # ensure that self.x is a copy of x. Don't store a reference
            # otherwise the memoization doesn't work properly.

            _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
            self._update_hess()
        else:
            # ensure that self.x is a copy of x. Don't store a reference
            # otherwise the memoization doesn't work properly.
            _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False

    def _update_fun(self):
        if not self.f_updated:
            fx = self._wrapped_fun(self.x)
            if fx < self._lowest_f:
                self._lowest_x = self.x
                self._lowest_f = fx

            self.f = fx
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            if self._orig_grad in FD_METHODS:
                self._update_fun()
            self.g = self._wrapped_grad(self.x, f0=self.f)
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            if self._orig_hess in FD_METHODS:
                self._update_grad()
                self.H = self._wrapped_hess(self.x, f0=self.g)
            elif isinstance(self._orig_hess, HessianUpdateStrategy):
                self._update_grad()
                self.H.update(self.x - self.x_prev, self.g - self.g_prev)
            else:       # should be callable(hess)
                self.H = self._wrapped_hess(self.x)

            self.H_updated = True

    def fun(self, x):
        if not np.array_equal(x, self.x):
            self._update_x(x)
        self._update_fun()
        return self.f

    def grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x(x)
        self._update_grad()
        return self.g

    def hess(self, x):
        if not np.array_equal(x, self.x):
            self._update_x(x)
        self._update_hess()
        return self.H

    def fun_and_grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g


class VectorFunction:
    """Vector function and its derivatives.

    This class defines a vector function F: R^n->R^m and methods for
    computing or approximating its first and second derivatives.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `jac`, hess` and corresponding attributes `f`, `J` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `jac` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def __init__(self, fun, x0, jac, hess,
                 finite_diff_rel_step, finite_diff_jac_sparsity,
                 finite_diff_bounds, sparse_jacobian):
        if not callable(jac) and jac not in FD_METHODS:
            raise ValueError(f"`jac` must be either callable or one of {FD_METHODS}.")

        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError("`hess` must be either callable,"
                             f"HessianUpdateStrategy or one of {FD_METHODS}.")

        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the Jacobian is estimated via "
                             "finite-differences, we require the Hessian to "
                             "be estimated using one of the quasi-Newton "
                             "strategies.")

        self.xp = xp = array_namespace(x0)
        _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype

        self.n = self.x.size
        self.nfev = 0
        self.njev = 0
        self.nhev = 0
        self.f_updated = False
        self.J_updated = False
        self.H_updated = False

        finite_diff_options = {}
        if jac in FD_METHODS:
            finite_diff_options["method"] = jac
            finite_diff_options["rel_step"] = finite_diff_rel_step
            if finite_diff_jac_sparsity is not None:
                sparsity_groups = group_columns(finite_diff_jac_sparsity)
                finite_diff_options["sparsity"] = (finite_diff_jac_sparsity,
                                                   sparsity_groups)
            finite_diff_options["bounds"] = finite_diff_bounds
            self.x_diff = np.copy(self.x)
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["as_linear_operator"] = True
            self.x_diff = np.copy(self.x)
        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the Jacobian is estimated via "
                             "finite-differences, we require the Hessian to "
                             "be estimated using one of the quasi-Newton "
                             "strategies.")

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            return np.atleast_1d(fun(x))

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun
        update_fun()

        self.v = np.zeros_like(self.f)
        self.m = self.v.size

        # Jacobian Evaluation
        if callable(jac):
            self.J = jac(self.x)
            self.J_updated = True
            self.njev += 1

            if (sparse_jacobian or
                    sparse_jacobian is None and sps.issparse(self.J)):
                def jac_wrapped(x):
                    self.njev += 1
                    return sps.csr_matrix(jac(x))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True

            elif sps.issparse(self.J):
                def jac_wrapped(x):
                    self.njev += 1
                    return jac(x).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False

            else:
                def jac_wrapped(x):
                    self.njev += 1
                    return np.atleast_2d(jac(x))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False

            def update_jac():
                self.J = jac_wrapped(self.x)

        elif jac in FD_METHODS:
            self.J = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                       **finite_diff_options)
            self.J_updated = True

            if (sparse_jacobian or
                    sparse_jacobian is None and sps.issparse(self.J)):
                def update_jac():
                    self._update_fun()
                    self.J = sps.csr_matrix(
                        approx_derivative(fun_wrapped, self.x, f0=self.f,
                                          **finite_diff_options))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True

            elif sps.issparse(self.J):
                def update_jac():
                    self._update_fun()
                    self.J = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                               **finite_diff_options).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False

            else:
                def update_jac():
                    self._update_fun()
                    self.J = np.atleast_2d(
                        approx_derivative(fun_wrapped, self.x, f0=self.f,
                                          **finite_diff_options))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False

        self._update_jac_impl = update_jac

        # Define Hessian
        if callable(hess):
            self.H = hess(self.x, self.v)
            self.H_updated = True
            self.nhev += 1

            if sps.issparse(self.H):
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return sps.csr_matrix(hess(x, v))
                self.H = sps.csr_matrix(self.H)

            elif isinstance(self.H, LinearOperator):
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return hess(x, v)

            else:
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return np.atleast_2d(np.asarray(hess(x, v)))
                self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x, self.v)
        elif hess in FD_METHODS:
            def jac_dot_v(x, v):
                return jac_wrapped(x).T.dot(v)

            def update_hess():
                self._update_jac()
                self.H = approx_derivative(jac_dot_v, self.x,
                                           f0=self.J.T.dot(self.v),
                                           args=(self.v,),
                                           **finite_diff_options)
            update_hess()
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.J_prev = None

            def update_hess():
                self._update_jac()
                # When v is updated before x was updated, then x_prev and
                # J_prev are None and we need this check.
                if self.x_prev is not None and self.J_prev is not None:
                    delta_x = self.x - self.x_prev
                    delta_g = self.J.T.dot(self.v) - self.J_prev.T.dot(self.v)
                    self.H.update(delta_x, delta_g)

        self._update_hess_impl = update_hess

        if isinstance(hess, HessianUpdateStrategy):
            def update_x(x):
                self._update_jac()
                self.x_prev = self.x
                self.J_prev = self.J
                _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False
                self._update_hess()
        else:
            def update_x(x):
                _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False

        self._update_x_impl = update_x

    def _update_v(self, v):
        if not np.array_equal(v, self.v):
            self.v = v
            self.H_updated = False

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_jac(self):
        if not self.J_updated:
            self._update_jac_impl()
            self.J_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        self._update_x(x)
        self._update_fun()
        return self.f

    def jac(self, x):
        self._update_x(x)
        self._update_jac()
        return self.J

    def hess(self, x, v):
        # v should be updated before x.
        self._update_v(v)
        self._update_x(x)
        self._update_hess()
        return self.H


class LinearVectorFunction:
    """Linear vector function and its derivatives.

    Defines a linear function F = A x, where x is N-D vector and
    A is m-by-n matrix. The Jacobian is constant and equals to A. The Hessian
    is identically zero and it is returned as a csr matrix.
    """
    def __init__(self, A, x0, sparse_jacobian):
        if sparse_jacobian or sparse_jacobian is None and sps.issparse(A):
            self.J = sps.csr_matrix(A)
            self.sparse_jacobian = True
        elif sps.issparse(A):
            self.J = A.toarray()
            self.sparse_jacobian = False
        else:
            # np.asarray makes sure A is ndarray and not matrix
            self.J = np.atleast_2d(np.asarray(A))
            self.sparse_jacobian = False

        self.m, self.n = self.J.shape

        self.xp = xp = array_namespace(x0)
        _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype

        self.f = self.J.dot(self.x)
        self.f_updated = True

        self.v = np.zeros(self.m, dtype=float)
        self.H = sps.csr_matrix((self.n, self.n))

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False

    def fun(self, x):
        self._update_x(x)
        if not self.f_updated:
            self.f = self.J.dot(x)
            self.f_updated = True
        return self.f

    def jac(self, x):
        self._update_x(x)
        return self.J

    def hess(self, x, v):
        self._update_x(x)
        self.v = v
        return self.H


class IdentityVectorFunction(LinearVectorFunction):
    """Identity vector function and its derivatives.

    The Jacobian is the identity matrix, returned as a dense array when
    `sparse_jacobian=False` and as a csr matrix otherwise. The Hessian is
    identically zero and it is returned as a csr matrix.
    """
    def __init__(self, x0, sparse_jacobian):
        n = len(x0)
        if sparse_jacobian or sparse_jacobian is None:
            A = sps.eye(n, format='csr')
            sparse_jacobian = True
        else:
            A = np.eye(n)
            sparse_jacobian = False
        super().__init__(A, x0, sparse_jacobian)
<|endoftext|>from ._trustregion import (_minimize_trust_region)
from ._trlib import (get_trlib_quadratic_subproblem)

__all__ = ['_minimize_trust_krylov']

def _minimize_trust_krylov(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           inexact=True, **trust_region_options):
    """
    Minimization of a scalar function of one or more variables using
    a nearly exact trust-region algorithm that only requires matrix
    vector products with the hessian matrix.

    .. versionadded:: 1.0.0

    Options
    -------
    inexact : bool, optional
        Accuracy to solve subproblems. If True requires less nonlinear
        iterations, but more vector products.
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region ',
                         'exact minimization.')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Krylov trust-region minimization')

    # tol_rel specifies the termination tolerance relative to the initial
    # gradient norm in the Krylov subspace iteration.

    # - tol_rel_i specifies the tolerance for interior convergence.
    # - tol_rel_b specifies the tolerance for boundary convergence.
    #   in nonlinear programming applications it is not necessary to solve
    #   the boundary case as exact as the interior case.

    # - setting tol_rel_i=-2 leads to a forcing sequence in the Krylov
    #   subspace iteration leading to quadratic convergence if eventually
    #   the trust region stays inactive.
    # - setting tol_rel_b=-3 leads to a forcing sequence in the Krylov
    #   subspace iteration leading to superlinear convergence as long
    #   as the iterates hit the trust region boundary.

    # For details consult the documentation of trlib_krylov_min
    # in _trlib/trlib_krylov.h
    #
    # Optimality of this choice of parameters among a range of possibilities
    # has been tested on the unconstrained subset of the CUTEst library.

    if inexact:
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=-2.0, tol_rel_b=-3.0,
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
    else:
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=1e-8, tol_rel_b=1e-6,
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
<|endoftext|>"""Routines for numerical differentiation."""
import functools
import numpy as np
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx


def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
    """Adjust final difference scheme to the presence of bounds.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Point at which we wish to estimate derivative.
    h : ndarray, shape (n,)
        Desired absolute finite difference steps.
    num_steps : int
        Number of `h` steps in one direction required to implement finite
        difference scheme. For example, 2 means that we need to evaluate
        f(x0 + 2 * h) or f(x0 - 2 * h)
    scheme : {'1-sided', '2-sided'}
        Whether steps in one or both directions are required. In other
        words '1-sided' applies to forward and backward schemes, '2-sided'
        applies to center schemes.
    lb : ndarray, shape (n,)
        Lower bounds on independent variables.
    ub : ndarray, shape (n,)
        Upper bounds on independent variables.

    Returns
    -------
    h_adjusted : ndarray, shape (n,)
        Adjusted absolute step sizes. Step size decreases only if a sign flip
        or switching to one-sided scheme doesn't allow to take a full step.
    use_one_sided : ndarray of bool, shape (n,)
        Whether to switch to one-sided scheme. Informative only for
        ``scheme='2-sided'``.
    """
    if scheme == '1-sided':
        use_one_sided = np.ones_like(h, dtype=bool)
    elif scheme == '2-sided':
        h = np.abs(h)
        use_one_sided = np.zeros_like(h, dtype=bool)
    else:
        raise ValueError("`scheme` must be '1-sided' or '2-sided'.")

    if np.all((lb == -np.inf) & (ub == np.inf)):
        return h, use_one_sided

    h_total = h * num_steps
    h_adjusted = h.copy()

    lower_dist = x0 - lb
    upper_dist = ub - x0

    if scheme == '1-sided':
        x = x0 + h_total
        violated = (x < lb) | (x > ub)
        fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
        h_adjusted[violated & fitting] *= -1

        forward = (upper_dist >= lower_dist) & ~fitting
        h_adjusted[forward] = upper_dist[forward] / num_steps
        backward = (upper_dist < lower_dist) & ~fitting
        h_adjusted[backward] = -lower_dist[backward] / num_steps
    elif scheme == '2-sided':
        central = (lower_dist >= h_total) & (upper_dist >= h_total)

        forward = (upper_dist >= lower_dist) & ~central
        h_adjusted[forward] = np.minimum(
            h[forward], 0.5 * upper_dist[forward] / num_steps)
        use_one_sided[forward] = True

        backward = (upper_dist < lower_dist) & ~central
        h_adjusted[backward] = -np.minimum(
            h[backward], 0.5 * lower_dist[backward] / num_steps)
        use_one_sided[backward] = True

        min_dist = np.minimum(upper_dist, lower_dist) / num_steps
        adjusted_central = (~central & (np.abs(h_adjusted) <= min_dist))
        h_adjusted[adjusted_central] = min_dist[adjusted_central]
        use_one_sided[adjusted_central] = False

    return h_adjusted, use_one_sided


@functools.lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """
    Calculates relative EPS step to use for a given data type
    and numdiff step method.

    Progressively smaller steps are used for larger floating point types.

    Parameters
    ----------
    f0_dtype: np.dtype
        dtype of function evaluation

    x0_dtype: np.dtype
        dtype of parameter vector

    method: {'2-point', '3-point', 'cs'}

    Returns
    -------
    EPS: float
        relative step size. May be np.float16, np.float32, np.float64

    Notes
    -----
    The default relative step will be np.float64. However, if x0 or f0 are
    smaller floating point types (np.float16, np.float32), then the smallest
    floating point type is chosen.
    """
    # the default EPS value
    EPS = np.finfo(np.float64).eps

    x0_is_fp = False
    if np.issubdtype(x0_dtype, np.inexact):
        # if you're a floating point type then over-ride the default EPS
        EPS = np.finfo(x0_dtype).eps
        x0_itemsize = np.dtype(x0_dtype).itemsize
        x0_is_fp = True

    if np.issubdtype(f0_dtype, np.inexact):
        f0_itemsize = np.dtype(f0_dtype).itemsize
        # choose the smallest itemsize between x0 and f0
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = np.finfo(f0_dtype).eps

    if method in ["2-point", "cs"]:
        return EPS**0.5
    elif method in ["3-point"]:
        return EPS**(1/3)
    else:
        raise RuntimeError("Unknown step method, should be one of "
                           "{'2-point', '3-point', 'cs'}")


def _compute_absolute_step(rel_step, x0, f0, method):
    """
    Computes an absolute step from a relative step for finite difference
    calculation.

    Parameters
    ----------
    rel_step: None or array-like
        Relative step for the finite difference calculation
    x0 : np.ndarray
        Parameter vector
    f0 : np.ndarray or scalar
    method : {'2-point', '3-point', 'cs'}

    Returns
    -------
    h : float
        The absolute step size

    Notes
    -----
    `h` will always be np.float64. However, if `x0` or `f0` are
    smaller floating point dtypes (e.g. np.float32), then the absolute
    step size will be calculated from the smallest floating point size.
    """
    # this is used instead of np.sign(x0) because we need
    # sign_x0 to be 1 when x0 == 0.
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1

    rstep = _eps_for_method(x0.dtype, f0.dtype, method)

    if rel_step is None:
        abs_step = rstep * sign_x0 * np.maximum(1.0, np.abs(x0))
    else:
        # User has requested specific relative steps.
        # Don't multiply by max(1, abs(x0) because if x0 < 1 then their
        # requested step is not used.
        abs_step = rel_step * sign_x0 * np.abs(x0)

        # however we don't want an abs_step of 0, which can happen if
        # rel_step is 0, or x0 is 0. Instead, substitute a realistic step
        dx = ((x0 + abs_step) - x0)
        abs_step = np.where(dx == 0,
                            rstep * sign_x0 * np.maximum(1.0, np.abs(x0)),
                            abs_step)

    return abs_step


def _prepare_bounds(bounds, x0):
    """
    Prepares new-style bounds from a two-tuple specifying the lower and upper
    limits for values in x0. If a value is not bound then the lower/upper bound
    will be expected to be -np.inf/np.inf.

    Examples
    --------
    >>> _prepare_bounds([(0, 1, 2), (1, 2, np.inf)], [0.5, 1.5, 2.5])
    (array([0., 1., 2.]), array([ 1.,  2., inf]))
    """
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, x0.shape)

    if ub.ndim == 0:
        ub = np.resize(ub, x0.shape)

    return lb, ub


def group_columns(A, order=0):
    """Group columns of a 2-D matrix for sparse finite differencing [1]_.

    Two columns are in the same group if in each row at least one of them
    has zero. A greedy sequential algorithm is used to construct groups.

    Parameters
    ----------
    A : array_like or sparse matrix, shape (m, n)
        Matrix of which to group columns.
    order : int, iterable of int with shape (n,) or None
        Permutation array which defines the order of columns enumeration.
        If int or None, a random permutation is used with `order` used as
        a random seed. Default is 0, that is use a random permutation but
        guarantee repeatability.

    Returns
    -------
    groups : ndarray of int, shape (n,)
        Contains values from 0 to n_groups-1, where n_groups is the number
        of found groups. Each value ``groups[i]`` is an index of a group to
        which ith column assigned. The procedure was helpful only if
        n_groups is significantly less than n.

    References
    ----------
    .. [1] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13 (1974), pp. 117-120.
    """
    if issparse(A):
        A = csc_matrix(A)
    else:
        A = np.atleast_2d(A)
        A = (A != 0).astype(np.int32)

    if A.ndim != 2:
        raise ValueError("`A` must be 2-dimensional.")

    m, n = A.shape

    if order is None or np.isscalar(order):
        rng = np.random.RandomState(order)
        order = rng.permutation(n)
    else:
        order = np.asarray(order)
        if order.shape != (n,):
            raise ValueError("`order` has incorrect shape.")

    A = A[:, order]

    if issparse(A):
        groups = group_sparse(m, n, A.indices, A.indptr)
    else:
        groups = group_dense(m, n, A)

    groups[order] = groups.copy()

    return groups


def approx_derivative(fun, x0, method='3-point', rel_step=None, abs_step=None,
                      f0=None, bounds=(-np.inf, np.inf), sparsity=None,
                      as_linear_operator=False, args=(), kwargs=None):
    """Compute finite difference approximation of the derivatives of a
    vector-valued function.

    If a function maps from R^n to R^m, its derivatives form m-by-n matrix
    called the Jacobian, where an element (i, j) is a partial derivative of
    f[i] with respect to x[j].

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is ndarray of shape (n,) (never a scalar
        even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    x0 : array_like of shape (n,) or float
        Point at which to estimate the derivatives. Float will be converted
        to a 1-D array.
    method : {'3-point', '2-point', 'cs'}, optional
        Finite difference method to use:
            - '2-point' - use the first order accuracy forward or backward
                          difference.
            - '3-point' - use central difference in interior points and the
                          second order accuracy forward or backward difference
                          near the boundary.
            - 'cs' - use a complex-step finite difference scheme. This assumes
                     that the user function is real-valued and can be
                     analytically continued to the complex plane. Otherwise,
                     produces bogus results.
    rel_step : None or array_like, optional
        Relative step size to use. If None (default) the absolute step size is
        computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
        `rel_step` being selected automatically, see Notes. Otherwise
        ``h = rel_step * sign(x0) * abs(x0)``. For ``method='3-point'`` the
        sign of `h` is ignored. The calculated step size is possibly adjusted
        to fit into the bounds.
    abs_step : array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `abs_step` is ignored. By default
        relative steps are used, only if ``abs_step is not None`` are absolute
        steps used.
    f0 : None or array_like, optional
        If not None it is assumed to be equal to ``fun(x0)``, in this case
        the ``fun(x0)`` is not called. Default is None.
    bounds : tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each bound must match the size of `x0` or be a scalar, in the latter
        case the bound will be the same for all variables. Use it to limit the
        range of function evaluation. Bounds checking is not implemented
        when `as_linear_operator` is True.
    sparsity : {None, array_like, sparse matrix, 2-tuple}, optional
        Defines a sparsity structure of the Jacobian matrix. If the Jacobian
        matrix is known to have only few non-zero elements in each row, then
        it's possible to estimate its several columns by a single function
        evaluation [3]_. To perform such economic computations two ingredients
        are required:

        * structure : array_like or sparse matrix of shape (m, n). A zero
          element means that a corresponding element of the Jacobian
          identically equals to zero.
        * groups : array_like of shape (n,). A column grouping for a given
          sparsity structure, use `group_columns` to obtain it.

        A single array or a sparse matrix is interpreted as a sparsity
        structure, and groups are computed inside the function. A tuple is
        interpreted as (structure, groups). If None (default), a standard
        dense differencing will be used.

        Note, that sparse differencing makes sense only for large Jacobian
        matrices where each row contains few non-zero elements.
    as_linear_operator : bool, optional
        When True the function returns an `scipy.sparse.linalg.LinearOperator`.
        Otherwise it returns a dense array or a sparse matrix depending on
        `sparsity`. The linear operator provides an efficient way of computing
        ``J.dot(p)`` for any vector ``p`` of shape (n,), but does not allow
        direct access to individual elements of the matrix. By default
        `as_linear_operator` is False.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------
    J : {ndarray, sparse matrix, LinearOperator}
        Finite difference approximation of the Jacobian matrix.
        If `as_linear_operator` is True returns a LinearOperator
        with shape (m, n). Otherwise it returns a dense array or sparse
        matrix depending on how `sparsity` is defined. If `sparsity`
        is None then a ndarray with shape (m, n) is returned. If
        `sparsity` is not None returns a csr_matrix with shape (m, n).
        For sparse matrices and linear operators it is always returned as
        a 2-D structure, for ndarrays, if m=1 it is returned
        as a 1-D gradient array with shape (n,).

    See Also
    --------
    check_derivative : Check correctness of a function computing derivatives.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point dtype of `x0` or `fun(x0)`,
    ``np.finfo(x0.dtype).eps``, s=2 for '2-point' method and
    s=3 for '3-point' method. Such relative step approximately minimizes a sum
    of truncation and round-off errors, see [1]_. Relative steps are used by
    default. However, absolute steps are used when ``abs_step is not None``.
    If any of the absolute or relative steps produces an indistinguishable
    difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
    automatic step size is substituted for that particular entry.

    A finite difference scheme for '3-point' method is selected automatically.
    The well-known central difference scheme is used for points sufficiently
    far from the boundary, and 3-point forward or backward scheme is used for
    points near the boundary. Both schemes have the second-order accuracy in
    terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point
    forward and backward difference schemes.

    For dense differencing when m=1 Jacobian is returned with a shape (n,),
    on the other hand when n=1 Jacobian is returned with a shape (m, 1).
    Our motivation is the following: a) It handles a case of gradient
    computation (m=1) in a conventional way. b) It clearly separates these two
    different cases. b) In all cases np.atleast_2d can be called to get 2-D
    Jacobian with correct dimensions.

    References
    ----------
    .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific
           Computing. 3rd edition", sec. 5.7.

    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13 (1974), pp. 117-120.

    .. [3] B. Fornberg, "Generation of Finite Difference Formulas on
           Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize._numdiff import approx_derivative
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * np.sin(c1 * x[1]),
    ...                      x[0] * np.cos(c2 * x[1])])
    ...
    >>> x0 = np.array([1.0, 0.5 * np.pi])
    >>> approx_derivative(f, x0, args=(1, 2))
    array([[ 1.,  0.],
           [-1.,  0.]])

    Bounds can be used to limit the region of function evaluation.
    In the example below we compute left and right derivative at point 1.0.

    >>> def g(x):
    ...     return x**2 if x >= 1 else x
    ...
    >>> x0 = 1.0
    >>> approx_derivative(g, x0, bounds=(-np.inf, 1.0))
    array([ 1.])
    >>> approx_derivative(g, x0, bounds=(1.0, np.inf))
    array([ 2.])
    """
    if method not in ['2-point', '3-point', 'cs']:
        raise ValueError(f"Unknown method '{method}'. ")

    xp = array_namespace(x0)
    _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
    _dtype = xp.float64
    if xp.isdtype(_x.dtype, "real floating"):
        _dtype = _x.dtype

    # promotes to floating
    x0 = xp.astype(_x, _dtype)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = _prepare_bounds(bounds, x0)

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    if as_linear_operator and not (np.all(np.isinf(lb))
                                   and np.all(np.isinf(ub))):
        raise ValueError("Bounds not supported when "
                         "`as_linear_operator` is True.")

    if kwargs is None:
        kwargs = {}

    def fun_wrapped(x):
        # send user function same fp type as x0. (but only if cs is not being
        # used
        if xp.isdtype(x.dtype, "real floating"):
            x = xp.astype(x, x0.dtype)

        f = np.atleast_1d(fun(x, *args, **kwargs))
        if f.ndim > 1:
            raise RuntimeError("`fun` return value has "
                               "more than 1 dimension.")
        return f

    if f0 is None:
        f0 = fun_wrapped(x0)
    else:
        f0 = np.atleast_1d(f0)
        if f0.ndim > 1:
            raise ValueError("`f0` passed has more than 1 dimension.")

    if np.any((x0 < lb) | (x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    if as_linear_operator:
        if rel_step is None:
            rel_step = _eps_for_method(x0.dtype, f0.dtype, method)

        return _linear_operator_difference(fun_wrapped, x0,
                                           f0, rel_step, method)
    else:
        # by default we use rel_step
        if abs_step is None:
            h = _compute_absolute_step(rel_step, x0, f0, method)
        else:
            # user specifies an absolute step
            sign_x0 = (x0 >= 0).astype(float) * 2 - 1
            h = abs_step

            # cannot have a zero step. This might happen if x0 is very large
            # or small. In which case fall back to relative step.
            dx = ((x0 + h) - x0)
            h = np.where(dx == 0,
                         _eps_for_method(x0.dtype, f0.dtype, method) *
                         sign_x0 * np.maximum(1.0, np.abs(x0)),
                         h)

        if method == '2-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '1-sided', lb, ub)
        elif method == '3-point':
            h, use_one_sided = _adjust_scheme_to_bounds(
                x0, h, 1, '2-sided', lb, ub)
        elif method == 'cs':
            use_one_sided = False

        if sparsity is None:
            return _dense_difference(fun_wrapped, x0, f0, h,
                                     use_one_sided, method)
        else:
            if not issparse(sparsity) and len(sparsity) == 2:
                structure, groups = sparsity
            else:
                structure = sparsity
                groups = group_columns(sparsity)

            if issparse(structure):
                structure = csc_matrix(structure)
            else:
                structure = np.atleast_2d(structure)

            groups = np.atleast_1d(groups)
            return _sparse_difference(fun_wrapped, x0, f0, h,
                                      use_one_sided, structure,
                                      groups, method)


def _linear_operator_difference(fun, x0, f0, h, method):
    m = f0.size
    n = x0.size

    if method == '2-point':
        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = h / norm(p)
            x = x0 + dx*p
            df = fun(x) - f0
            return df / dx

    elif method == '3-point':
        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = 2*h / norm(p)
            x1 = x0 - (dx/2)*p
            x2 = x0 + (dx/2)*p
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
            return df / dx

    elif method == 'cs':
        def matvec(p):
            if np.array_equal(p, np.zeros_like(p)):
                return np.zeros(m)
            dx = h / norm(p)
            x = x0 + dx*p*1.j
            f1 = fun(x)
            df = f1.imag
            return df / dx

    else:
        raise RuntimeError("Never be here.")

    return LinearOperator((m, n), matvec)


def _dense_difference(fun, x0, f0, h, use_one_sided, method):
    m = f0.size
    n = x0.size
    J_transposed = np.empty((n, m))
    x1 = x0.copy()
    x2 = x0.copy()
    xc = x0.astype(complex, copy=True)

    for i in range(h.size):
        if method == '2-point':
            x1[i] += h[i]
            dx = x1[i] - x0[i]  # Recompute dx as exactly representable number.
            df = fun(x1) - f0
        elif method == '3-point' and use_one_sided[i]:
            x1[i] += h[i]
            x2[i] += 2 * h[i]
            dx = x2[i] - x0[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = -3.0 * f0 + 4 * f1 - f2
        elif method == '3-point' and not use_one_sided[i]:
            x1[i] -= h[i]
            x2[i] += h[i]
            dx = x2[i] - x1[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
        elif method == 'cs':
            xc[i] += h[i] * 1.j
            f1 = fun(xc)
            df = f1.imag
            dx = h[i]
        else:
            raise RuntimeError("Never be here.")

        J_transposed[i] = df / dx
        x1[i] = x2[i] = xc[i] = x0[i]

    if m == 1:
        J_transposed = np.ravel(J_transposed)

    return J_transposed.T


def _sparse_difference(fun, x0, f0, h, use_one_sided,
                       structure, groups, method):
    m = f0.size
    n = x0.size
    row_indices = []
    col_indices = []
    fractions = []

    n_groups = np.max(groups) + 1
    for group in range(n_groups):
        # Perturb variables which are in the same group simultaneously.
        e = np.equal(group, groups)
        h_vec = h * e
        if method == '2-point':
            x = x0 + h_vec
            dx = x - x0
            df = fun(x) - f0
            # The result is  written to columns which correspond to perturbed
            # variables.
            cols, = np.nonzero(e)
            # Find all non-zero elements in selected columns of Jacobian.
            i, j, _ = find(structure[:, cols])
            # Restore column indices in the full array.
            j = cols[j]
        elif method == '3-point':
            # Here we do conceptually the same but separate one-sided
            # and two-sided schemes.
            x1 = x0.copy()
            x2 = x0.copy()

            mask_1 = use_one_sided & e
            x1[mask_1] += h_vec[mask_1]
            x2[mask_1] += 2 * h_vec[mask_1]

            mask_2 = ~use_one_sided & e
            x1[mask_2] -= h_vec[mask_2]
            x2[mask_2] += h_vec[mask_2]

            dx = np.zeros(n)
            dx[mask_1] = x2[mask_1] - x0[mask_1]
            dx[mask_2] = x2[mask_2] - x1[mask_2]

            f1 = fun(x1)
            f2 = fun(x2)

            cols, = np.nonzero(e)
            i, j, _ = find(structure[:, cols])
            j = cols[j]

            mask = use_one_sided[j]
            df = np.empty(m)

            rows = i[mask]
            df[rows] = -3 * f0[rows] + 4 * f1[rows] - f2[rows]

            rows = i[~mask]
            df[rows] = f2[rows] - f1[rows]
        elif method == 'cs':
            f1 = fun(x0 + h_vec*1.j)
            df = f1.imag
            dx = h_vec
            cols, = np.nonzero(e)
            i, j, _ = find(structure[:, cols])
            j = cols[j]
        else:
            raise ValueError("Never be here.")

        # All that's left is to compute the fraction. We store i, j and
        # fractions as separate arrays and later construct coo_matrix.
        row_indices.append(i)
        col_indices.append(j)
        fractions.append(df[i] / dx[j])

    row_indices = np.hstack(row_indices)
    col_indices = np.hstack(col_indices)
    fractions = np.hstack(fractions)
    J = coo_matrix((fractions, (row_indices, col_indices)), shape=(m, n))
    return csr_matrix(J)


def check_derivative(fun, jac, x0, bounds=(-np.inf, np.inf), args=(),
                     kwargs=None):
    """Check correctness of a function computing derivatives (Jacobian or
    gradient) by comparison with a finite difference approximation.

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is ndarray of shape (n,) (never a scalar
        even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    jac : callable
        Function which computes Jacobian matrix of `fun`. It must work with
        argument x the same way as `fun`. The return value must be array_like
        or sparse matrix with an appropriate shape.
    x0 : array_like of shape (n,) or float
        Point at which to estimate the derivatives. Float will be converted
        to 1-D array.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each bound must match the size of `x0` or be a scalar, in the latter
        case the bound will be the same for all variables. Use it to limit the
        range of function evaluation.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same
        for `jac`.

    Returns
    -------
    accuracy : float
        The maximum among all relative errors for elements with absolute values
        higher than 1 and absolute errors for elements with absolute values
        less or equal than 1. If `accuracy` is on the order of 1e-6 or lower,
        then it is likely that your `jac` implementation is correct.

    See Also
    --------
    approx_derivative : Compute finite difference approximation of derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize._numdiff import check_derivative
    >>>
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * np.sin(c1 * x[1]),
    ...                      x[0] * np.cos(c2 * x[1])])
    ...
    >>> def jac(x, c1, c2):
    ...     return np.array([
    ...         [np.sin(c1 * x[1]),  c1 * x[0] * np.cos(c1 * x[1])],
    ...         [np.cos(c2 * x[1]), -c2 * x[0] * np.sin(c2 * x[1])]
    ...     ])
    ...
    >>>
    >>> x0 = np.array([1.0, 0.5 * np.pi])
    >>> check_derivative(f, jac, x0, args=(1, 2))
    2.4492935982947064e-16
    """
    if kwargs is None:
        kwargs = {}
    J_to_test = jac(x0, *args, **kwargs)
    if issparse(J_to_test):
        J_diff = approx_derivative(fun, x0, bounds=bounds, sparsity=J_to_test,
                                   args=args, kwargs=kwargs)
        J_to_test = csr_matrix(J_to_test)
        abs_err = J_to_test - J_diff
        i, j, abs_err_data = find(abs_err)
        J_diff_data = np.asarray(J_diff[i, j]).ravel()
        return np.max(np.abs(abs_err_data) /
                      np.maximum(1, np.abs(J_diff_data)))
    else:
        J_diff = approx_derivative(fun, x0, bounds=bounds,
                                   args=args, kwargs=kwargs)
        abs_err = np.abs(J_to_test - J_diff)
        return np.max(abs_err / np.maximum(1, np.abs(J_diff)))
<|endoftext|>"""
Functions
---------
.. autosummary::
   :toctree: generated/

    line_search_armijo
    line_search_wolfe1
    line_search_wolfe2
    scalar_search_wolfe1
    scalar_search_wolfe2

"""
from warnings import warn

from ._dcsrch import DCSRCH
import numpy as np

__all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2',
           'scalar_search_wolfe1', 'scalar_search_wolfe2',
           'line_search_armijo']

class LineSearchWarning(RuntimeWarning):
    pass


def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


#------------------------------------------------------------------------------
# Minpack's Wolfe line and scalar searches
#------------------------------------------------------------------------------

def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    """
    if gfk is None:
        gfk = fprime(xk, *args)

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.

    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`

    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1`` as described in [1]_.

    References
    ----------
    
    .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
       In Springer Series in Operations Research and Financial Engineering.
       (Springer Series in Operations Research and Financial Engineering).
       Springer Nature.

    """
    _check_c1_c2(c1, c2)

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    maxiter = 100

    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    stp, phi1, phi0, task = dcsrch(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    return stp, phi1, phi0


line_search = line_search_wolfe1


#------------------------------------------------------------------------------
# Pure-Python Wolfe line and scalar searches
#------------------------------------------------------------------------------

# Note: `line_search_wolfe2` is the public `scipy.optimize.line_search`

def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction. The search direction must be a descent direction
        for the algorithm to converge.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    The search direction `pk` must be a descent direction (e.g.
    ``-myfprime(xk)``) to find a step length that satisfies the strong Wolfe
    conditions. If the search direction is not a descent direction (e.g.
    ``myfprime(xk)``), then `alpha`, `new_fval`, and `new_slope` will be None.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import line_search

    A objective function and its gradient are defined.

    >>> def obj_func(x):
    ...     return (x[0])**2+(x[1])**2
    >>> def obj_grad(x):
    ...     return [2*x[0], 2*x[1]]

    We can find alpha that satisfies strong Wolfe conditions.

    >>> start_point = np.array([1.8, 1.7])
    >>> search_gradient = np.array([-1.0, -1.0])
    >>> line_search(obj_func, obj_grad, start_point, search_gradient)
    (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])

    """
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)

    fprime = myfprime

    def derphi(alpha):
        gc[0] += 1
        gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
        gval_alpha[0] = alpha
        return np.dot(gval[0], pk)

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    if extra_condition is not None:
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
            extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalar_search_wolfe2(phi, derphi, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None,
                         extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    phi0 : float
        phi at 0.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    """
    _check_c1_c2(c1, c2)

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    #derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        def extra_condition(alpha, phi):
            return True

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 > amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = "The line search algorithm could not find a solution " + \
                      f"less than or equal to amax: {amax}"

            warn(msg, LineSearchWarning, stacklevel=2)
            break

        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)

    return alpha_star, phi_star, phi0, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found, return None.

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


#------------------------------------------------------------------------------
# Armijo line and scalar searches
#------------------------------------------------------------------------------

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    gfk : array_like
        Gradient of `f` at point `xk`.
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
    c1 : float, optional
        Value to control stopping criterion.
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.

    Returns
    -------
    alpha
    f_count
    f_val_at_alpha

    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1*pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    derphi0 = np.dot(gfk, pk)
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1


def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """
    Compatibility wrapper for `line_search_armijo`
    """
    r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1,
                           alpha0=alpha0)
    return r[0], r[1], 0, r[2]


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
    phi1

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1


#------------------------------------------------------------------------------
# Non-monotone line search for DF-SANE
#------------------------------------------------------------------------------

def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta,
                                  gamma=1e-4, tau_min=0.1, tau_max=0.5):
    """
    Nonmonotone backtracking line search as described in [1]_

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    prev_fs : float
        List of previous merit function values. Should have ``len(prev_fs) <= M``
        where ``M`` is the nonmonotonicity window parameter.
    eta : float
        Allowed merit function increase, see [1]_
    gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position

    References
    ----------
    [1] "Spectral residual method without gradient information for solving
        large-scale nonlinear systems of equations." W. La Cruz,
        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).

    """
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)

    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    return alpha, xp, fp, Fp


def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta,
                                   gamma=1e-4, tau_min=0.1, tau_max=0.5,
                                   nu=0.85):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    f_k : float
        Initial merit function value.
    C, Q : float
        Control parameters. On the first iteration, give values
        Q=1.0, C=f_k
    eta : float
        Allowed merit function increase, see [1]_
    nu, gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position
    C : float
        New value for the control parameter C
    Q : float
        New value for the control parameter Q

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    """
    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    # Update C and Q
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next

    return alpha, xp, fp, Fp, C, Q
<|endoftext|>"""
Method agnostic utility functions for linear programming
"""

import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
    _remove_redundancy_svd, _remove_redundancy_pivot_sparse,
    _remove_redundancy_pivot_dense, _remove_redundancy_id
    )
from collections import namedtuple

_LPProblem = namedtuple('_LPProblem',
                        'c A_ub b_ub A_eq b_eq bounds x0 integrality')
_LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg
_LPProblem.__doc__ = \
    """ Represents a linear-programming problem.

    Attributes
    ----------
    c : 1D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : various valid formats, optional
        The bounds of ``x``, as ``min`` and ``max`` pairs.
        If bounds are specified for all N variables separately, valid formats
        are:
        * a 2D array (N x 2);
        * a sequence of N sequences, each with 2 values.
        If all variables have the same bounds, the bounds can be specified as
        a 1-D or 2-D array or sequence with 2 scalar values.
        If all variables have a lower bound of 0 and no upper bound, the bounds
        parameter can be omitted (or given as None).
        Absent lower and/or upper bounds can be specified as -numpy.inf (no
        lower bound), numpy.inf (no upper bound) or None (both).
    x0 : 1D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.
    integrality : 1-D array or int, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous.

        For mixed integrality constraints, supply an array of shape `c.shape`.
        To infer a constraint on each decision variable from shorter inputs,
        the argument will be broadcast to `c.shape` using `np.broadcast_to`.

        This argument is currently used only by the ``'highs'`` method and
        ignored otherwise.

    Notes
    -----
    This namedtuple supports 2 ways of initialization:
    >>> lp1 = _LPProblem(c=[-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4])
    >>> lp2 = _LPProblem([-1, 4], [[-3, 1], [1, 2]], [6, 4])

    Note that only ``c`` is a required argument here, whereas all other arguments
    ``A_ub``, ``b_ub``, ``A_eq``, ``b_eq``, ``bounds``, ``x0`` are optional with
    default values of None.
    For example, ``A_eq`` and ``b_eq`` can be set without ``A_ub`` or ``b_ub``:
    >>> lp3 = _LPProblem(c=[-1, 4], A_eq=[[2, 1]], b_eq=[10])
    """


def _check_sparse_inputs(options, meth, A_ub, A_eq):
    """
    Check the provided ``A_ub`` and ``A_eq`` matrices conform to the specified
    optional sparsity variables.

    Parameters
    ----------
    A_ub : 2-D array, optional
        2-D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    A_eq : 2-D array, optional
        2-D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.
    method : str, optional
        The algorithm used to solve the standard form problem.

    Returns
    -------
    A_ub : 2-D array, optional
        2-D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    A_eq : 2-D array, optional
        2-D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.
    """
    # This is an undocumented option for unit testing sparse presolve
    _sparse_presolve = options.pop('_sparse_presolve', False)
    if _sparse_presolve and A_eq is not None:
        A_eq = sps.coo_matrix(A_eq)
    if _sparse_presolve and A_ub is not None:
        A_ub = sps.coo_matrix(A_ub)

    sparse_constraint = sps.issparse(A_eq) or sps.issparse(A_ub)

    preferred_methods = {"highs", "highs-ds", "highs-ipm"}
    dense_methods = {"simplex", "revised simplex"}
    if meth in dense_methods and sparse_constraint:
        raise ValueError(f"Method '{meth}' does not support sparse "
                         "constraint matrices. Please consider using one of "
                         f"{preferred_methods}.")

    sparse = options.get('sparse', False)
    if not sparse and sparse_constraint and meth == 'interior-point':
        options['sparse'] = True
        warn("Sparse constraint matrix detected; setting 'sparse':True.",
             OptimizeWarning, stacklevel=4)
    return options, A_ub, A_eq


def _format_A_constraints(A, n_x, sparse_lhs=False):
    """Format the left hand side of the constraints to a 2-D array

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.
    n_x : int
        The number of variables in the linear programming problem.
    sparse_lhs : bool
        Whether either of `A_ub` or `A_eq` are sparse. If true return a
        coo_matrix instead of a numpy array.

    Returns
    -------
    np.ndarray or sparse.coo_matrix
        2-D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.

    """
    if sparse_lhs:
        return sps.coo_matrix(
            (0, n_x) if A is None else A, dtype=float, copy=True
        )
    elif A is None:
        return np.zeros((0, n_x), dtype=float)
    else:
        return np.array(A, dtype=float, copy=True)


def _format_b_constraints(b):
    """Format the upper bounds of the constraints to a 1-D array

    Parameters
    ----------
    b : 1-D array
        1-D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.

    Returns
    -------
    1-D np.array
        1-D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.

    """
    if b is None:
        return np.array([], dtype=float)
    b = np.array(b, dtype=float, copy=True).squeeze()
    return b if b.size != 1 else b.reshape(-1)


def _clean_inputs(lp):
    """
    Given user inputs for a linear programming problem, return the
    objective vector, upper bound constraints, equality constraints,
    and simple bounds in a preferred format.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : various valid formats, optional
            The bounds of ``x``, as ``min`` and ``max`` pairs.
            If bounds are specified for all N variables separately, valid formats are:
            * a 2D array (2 x N or N x 2);
            * a sequence of N sequences, each with 2 values.
            If all variables have the same bounds, a single pair of values can
            be specified. Valid formats are:
            * a sequence with 2 scalar values;
            * a sequence with a single element containing 2 scalar values.
            If all variables have a lower bound of 0 and no upper bound, the bounds
            parameter can be omitted (or given as None).
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    Returns
    -------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N
            elements of ``x``. The N x 2 array contains lower bounds in the first
            column and upper bounds in the 2nd. Unbounded variables have lower
            bound -np.inf and/or upper bound np.inf.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp

    if c is None:
        raise TypeError

    try:
        c = np.array(c, dtype=np.float64, copy=True).squeeze()
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: c must be a 1-D array of numerical "
            "coefficients") from e
    else:
        # If c is a single value, convert it to a 1-D array.
        if c.size == 1:
            c = c.reshape(-1)

        n_x = len(c)
        if n_x == 0 or len(c.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: c must be a 1-D array and must "
                "not have more than one non-singleton dimension")
        if not np.isfinite(c).all():
            raise ValueError(
                "Invalid input for linprog: c must not contain values "
                "inf, nan, or None")

    sparse_lhs = sps.issparse(A_eq) or sps.issparse(A_ub)
    try:
        A_ub = _format_A_constraints(A_ub, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: A_ub must be a 2-D array "
            "of numerical values") from e
    else:
        n_ub = A_ub.shape[0]
        if len(A_ub.shape) != 2 or A_ub.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_ub must have exactly two "
                "dimensions, and the number of columns in A_ub must be "
                "equal to the size of c")
        if (sps.issparse(A_ub) and not np.isfinite(A_ub.data).all()
                or not sps.issparse(A_ub) and not np.isfinite(A_ub).all()):
            raise ValueError(
                "Invalid input for linprog: A_ub must not contain values "
                "inf, nan, or None")

    try:
        b_ub = _format_b_constraints(b_ub)
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: b_ub must be a 1-D array of "
            "numerical values, each representing the upper bound of an "
            "inequality constraint (row) in A_ub") from e
    else:
        if b_ub.shape != (n_ub,):
            raise ValueError(
                "Invalid input for linprog: b_ub must be a 1-D array; b_ub "
                "must not have more than one non-singleton dimension and "
                "the number of rows in A_ub must equal the number of values "
                "in b_ub")
        if not np.isfinite(b_ub).all():
            raise ValueError(
                "Invalid input for linprog: b_ub must not contain values "
                "inf, nan, or None")

    try:
        A_eq = _format_A_constraints(A_eq, n_x, sparse_lhs=sparse_lhs)
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: A_eq must be a 2-D array "
            "of numerical values") from e
    else:
        n_eq = A_eq.shape[0]
        if len(A_eq.shape) != 2 or A_eq.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_eq must have exactly two "
                "dimensions, and the number of columns in A_eq must be "
                "equal to the size of c")

        if (sps.issparse(A_eq) and not np.isfinite(A_eq.data).all()
                or not sps.issparse(A_eq) and not np.isfinite(A_eq).all()):
            raise ValueError(
                "Invalid input for linprog: A_eq must not contain values "
                "inf, nan, or None")

    try:
        b_eq = _format_b_constraints(b_eq)
    except ValueError as e:
        raise TypeError(
            "Invalid input for linprog: b_eq must be a dense, 1-D array of "
            "numerical values, each representing the right hand side of an "
            "equality constraint (row) in A_eq") from e
    else:
        if b_eq.shape != (n_eq,):
            raise ValueError(
                "Invalid input for linprog: b_eq must be a 1-D array; b_eq "
                "must not have more than one non-singleton dimension and "
                "the number of rows in A_eq must equal the number of values "
                "in b_eq")
        if not np.isfinite(b_eq).all():
            raise ValueError(
                "Invalid input for linprog: b_eq must not contain values "
                "inf, nan, or None")

    # x0 gives a (optional) starting solution to the solver. If x0 is None,
    # skip the checks. Initial solution will be generated automatically.
    if x0 is not None:
        try:
            x0 = np.array(x0, dtype=float, copy=True).squeeze()
        except ValueError as e:
            raise TypeError(
                "Invalid input for linprog: x0 must be a 1-D array of "
                "numerical coefficients") from e
        if x0.ndim == 0:
            x0 = x0.reshape(-1)
        if len(x0) == 0 or x0.ndim != 1:
            raise ValueError(
                "Invalid input for linprog: x0 should be a 1-D array; it "
                "must not have more than one non-singleton dimension")
        if not x0.size == c.size:
            raise ValueError(
                "Invalid input for linprog: x0 and c should contain the "
                "same number of elements")
        if not np.isfinite(x0).all():
            raise ValueError(
                "Invalid input for linprog: x0 must not contain values "
                "inf, nan, or None")

    # Bounds can be one of these formats:
    # (1) a 2-D array or sequence, with shape N x 2
    # (2) a 1-D or 2-D sequence or array with 2 scalars
    # (3) None (or an empty sequence or array)
    # Unspecified bounds can be represented by None or (-)np.inf.
    # All formats are converted into a N x 2 np.array with (-)np.inf where
    # bounds are unspecified.

    # Prepare clean bounds array
    bounds_clean = np.zeros((n_x, 2), dtype=float)

    # Convert to a numpy array.
    # np.array(..,dtype=float) raises an error if dimensions are inconsistent
    # or if there are invalid data types in bounds. Just add a linprog prefix
    # to the error and re-raise.
    # Creating at least a 2-D array simplifies the cases to distinguish below.
    if bounds is None or np.array_equal(bounds, []) or np.array_equal(bounds, [[]]):
        bounds = (0, np.inf)
    try:
        bounds_conv = np.atleast_2d(np.array(bounds, dtype=float))
    except ValueError as e:
        raise ValueError(
            "Invalid input for linprog: unable to interpret bounds, "
            "check values and dimensions: " + e.args[0]) from e
    except TypeError as e:
        raise TypeError(
            "Invalid input for linprog: unable to interpret bounds, "
            "check values and dimensions: " + e.args[0]) from e

    # Check bounds options
    bsh = bounds_conv.shape
    if len(bsh) > 2:
        # Do not try to handle multidimensional bounds input
        raise ValueError(
            "Invalid input for linprog: provide a 2-D array for bounds, "
            f"not a {len(bsh):d}-D array.")
    elif np.all(bsh == (n_x, 2)):
        # Regular N x 2 array
        bounds_clean = bounds_conv
    elif (np.all(bsh == (2, 1)) or np.all(bsh == (1, 2))):
        # 2 values: interpret as overall lower and upper bound
        bounds_flat = bounds_conv.flatten()
        bounds_clean[:, 0] = bounds_flat[0]
        bounds_clean[:, 1] = bounds_flat[1]
    elif np.all(bsh == (2, n_x)):
        # Reject a 2 x N array
        raise ValueError(
            f"Invalid input for linprog: provide a {n_x:d} x 2 array for bounds, "
            f"not a 2 x {n_x:d} array.")
    else:
        raise ValueError(
            "Invalid input for linprog: unable to interpret bounds with this "
            f"dimension tuple: {bsh}.")

    # The process above creates nan-s where the input specified None
    # Convert the nan-s in the 1st column to -np.inf and in the 2nd column
    # to np.inf
    i_none = np.isnan(bounds_clean[:, 0])
    bounds_clean[i_none, 0] = -np.inf
    i_none = np.isnan(bounds_clean[:, 1])
    bounds_clean[i_none, 1] = np.inf

    return _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds_clean, x0, integrality)


def _presolve(lp, rr, rr_method, tol=1e-9):
    """
    Given inputs for a linear programming problem in preferred format,
    presolve the problem: identify trivial infeasibilities, redundancies,
    and unboundedness, tighten bounds where possible, and eliminate fixed
    variables.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N
            elements of ``x``. The N x 2 array contains lower bounds in the first
            column and upper bounds in the 2nd. Unbounded variables have lower
            bound -np.inf and/or upper bound np.inf.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    rr : bool
        If ``True`` attempts to eliminate any redundant rows in ``A_eq``.
        Set False if ``A_eq`` is known to be of full row rank, or if you are
        looking for a potential speedup (at the expense of reliability).
    rr_method : string
        Method used to identify and remove redundant rows from the
        equality constraint matrix after presolve.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.

    Returns
    -------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, possibly tightened.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : 1D array
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x : 1D array
        Solution vector (when the solution is trivial and can be determined
        in presolve)
    revstack: list of functions
        the functions in the list reverse the operations of _presolve()
        the function signature is x_org = f(x_mod), where x_mod is the result
        of a presolve step and x_org the value at the start of the step
        (currently, the revstack contains only one function)
    complete: bool
        Whether the solution is complete (solved or determined to be infeasible
        or unbounded in presolve)
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.

    """
    # ideas from Reference [5] by Andersen and Andersen
    # however, unlike the reference, this is performed before converting
    # problem to standard form
    # There are a few advantages:
    #  * artificial variables have not been added, so matrices are smaller
    #  * bounds have not been converted to constraints yet. (It is better to
    #    do that after presolve because presolve may adjust the simple bounds.)
    # There are many improvements that can be made, namely:
    #  * implement remaining checks from [5]
    #  * loop presolve until no additional changes are made
    #  * implement additional efficiency improvements in redundancy removal [2]

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, _ = lp

    revstack = []               # record of variables eliminated from problem
    # constant term in cost function may be added if variables are eliminated
    c0 = 0
    complete = False        # complete is True if detected infeasible/unbounded
    x = np.zeros(c.shape)   # this is solution vector if completed in presolve

    status = 0              # all OK unless determined otherwise
    message = ""

    # Lower and upper bounds. Copy to prevent feedback.
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()

    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape

    if (rr_method is not None
            and rr_method.lower() not in {"svd", "pivot", "id"}):
        message = ("'" + str(rr_method) + "' is not a valid option "
                   "for redundancy removal. Valid options are 'SVD', "
                   "'pivot', and 'ID'.")
        raise ValueError(message)

    if sps.issparse(A_eq):
        A_eq = A_eq.tocsr()
        A_ub = A_ub.tocsr()

        def where(A):
            return A.nonzero()

        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack

    # upper bounds > lower bounds
    if np.any(ub < lb) or np.any(lb == np.inf) or np.any(ub == -np.inf):
        status = 2
        message = ("The problem is (trivially) infeasible since one "
                   "or more upper bounds are smaller than the corresponding "
                   "lower bounds, a lower bound is np.inf or an upper bound "
                   "is -np.inf.")
        complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # zero row in equality constraints
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(
            np.logical_and(
                zero_row,
                np.abs(b_eq) > tol)):  # test_zero_row_1
            # infeasible if RHS is not zero
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding constraint value.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # if RHS is zero, we can eliminate this equation entirely
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]

    # zero row in inequality constraints
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
            # infeasible if RHS is less than zero (because LHS is zero)
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding  constraint value.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # if LHS is >= 0, we can eliminate this constraint entirely
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]

    # zero column in (both) constraints
    # this indicates that a variable isn't constrained and can be removed
    A = vstack((A_eq, A_ub))
    if A.shape[0] > 0:
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        # variable will be at upper or lower bound, depending on objective
        x[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):  # if an unconstrained variable has no bound
            status = 3
            message = ("If feasible, the problem is (trivially) unbounded "
                       "due  to a zero column in the constraint matrices. If "
                       "you wish to check whether the problem is infeasible, "
                       "turn presolve off.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        # variables will equal upper/lower bounds will be removed later
        lb[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]

    # row singleton in equality constraints
    # this fixes a variable and removes the constraint
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = where(singleton_row)[0]
    cols = where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                # infeasible if fixed value is not within bounds
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the equality constraints is "
                           "inconsistent with the bounds.")
                complete = True
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
            else:
                # sets upper and lower bounds at that fixed value - variable
                # will be removed later
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]

    # row singleton in inequality constraints
    # this indicates a simple bound and the constraint can be removed
    # simple bounds may be adjusted here
    # After all of the simple bound information is combined here, get_Abc will
    # turn the simple bounds into constraints
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = where(A_ub[singleton_row, :])[1]
    rows = where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:  # upper bound
                if val < lb[col] - tol:  # infeasible
                    complete = True
                elif val < ub[col]:  # new upper bound
                    ub[col] = val
            else:  # lower bound
                if val > ub[col] + tol:  # infeasible
                    complete = True
                elif val > lb[col]:  # new lower bound
                    lb[col] = val
            if complete:
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the upper bound constraints is "
                           "inconsistent with the bounds.")
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]

    # identical bounds indicate that variable can be removed
    i_f = np.abs(lb - ub) < tol   # indices of "fixed" variables
    i_nf = np.logical_not(i_f)  # indices of "not fixed" variables

    # test_bounds_equal_but_infeasible
    if np.all(i_f):  # if bounds define solution, check for consistency
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        if ((A_ub.size > 0 and np.any(slack < 0)) or
                (A_eq.size > 0 and not np.allclose(residual, 0))):
            status = 2
            message = ("The problem is (trivially) infeasible because the "
                       "bounds fix all variables to values inconsistent with "
                       "the constraints")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)

    ub_mod = ub
    lb_mod = lb
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x_undo = lb[i_f]  # not x[i_f], x is just zeroes
        x = x[i_nf]
        # user guess x0 stays separate from presolve solution x
        if x0 is not None:
            x0 = x0[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        # modify bounds
        lb_mod = lb[i_nf]
        ub_mod = ub[i_nf]

        def rev(x_mod):
            # Function to restore x: insert x_undo into x_mod.
            # When elements have been removed at positions k1, k2, k3, ...
            # then these must be replaced at (after) positions k1-1, k2-2,
            # k3-3, ... in the modified array to recreate the original
            i = np.flatnonzero(i_f)
            # Number of variables to restore
            N = len(i)
            index_offset = np.arange(N)
            # Create insert indices
            insert_indices = i - index_offset
            x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
            return x_rev

        # Use revstack as a list of functions, currently just this one.
        revstack.append(rev)

    # no constraints indicates that problem is trivial
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        # test_empty_constraint_1
        if c.size == 0:
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        elif (np.any(np.logical_and(c < 0, ub_mod == np.inf)) or
              np.any(np.logical_and(c > 0, lb_mod == -np.inf))):
            # test_no_constraints()
            # test_unbounded_no_nontrivial_constraints_1
            # test_unbounded_no_nontrivial_constraints_2
            status = 3
            message = ("The problem is (trivially) unbounded "
                       "because there are no non-trivial constraints and "
                       "a) at least one decision variable is unbounded "
                       "above and its corresponding cost is negative, or "
                       "b) at least one decision variable is unbounded below "
                       "and its corresponding cost is positive. ")
        else:  # test_empty_constraint_2
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        complete = True
        x[c < 0] = ub_mod[c < 0]
        x[c > 0] = lb_mod[c > 0]
        # where c is zero, set x to a finite bound or zero
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
        # if this is not the last step of presolve, should convert bounds back
        # to array and return here

    # Convert modified lb and ub back into N x 2 bounds
    bounds = np.hstack((lb_mod[:, np.newaxis], ub_mod[:, np.newaxis]))

    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")
    if (sps.issparse(A_eq)):
        if rr and A_eq.size > 0:  # TODO: Fast sparse rank check?
            rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
            A_eq, b_eq, status, message = rr_res
            if A_eq.shape[0] < n_rows_A:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)
            if status != 0:
                complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # This is a wild guess for which redundancy removal algorithm will be
    # faster. More testing would be good.
    small_nullspace = 5
    if rr and A_eq.size > 0:
        try:  # TODO: use results of first SVD in _remove_redundancy_svd
            rank = np.linalg.matrix_rank(A_eq)
        # oh well, we'll have to go with _remove_redundancy_pivot_dense
        except Exception:
            rank = 0
    if rr and A_eq.size > 0 and rank < A_eq.shape[0]:
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0]-rank
        if rr_method is None:
            if dim_row_nullspace <= small_nullspace:
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            if dim_row_nullspace > small_nullspace or status == 4:
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res

        else:
            rr_method = rr_method.lower()
            if rr_method == "svd":
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "pivot":
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "id":
                rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                A_eq, b_eq, status, message = rr_res
            else:  # shouldn't get here; option validity checked above
                pass
        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4
        if status != 0:
            complete = True
    return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
