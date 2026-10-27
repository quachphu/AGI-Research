if zvode.runner is not None:
    IntegratorBase.integrator_classes.append(zvode)


class dopri5(IntegratorBase):
    runner = getattr(_dop, 'dopri5', None)
    name = 'dopri5'
    supports_solout = True

    messages = {1: 'computation successful',
                2: 'computation successful (interrupted by solout)',
                -1: 'input is not consistent',
                -2: 'larger nsteps is needed',
                -3: 'step size becomes too small',
                -4: 'problem is probably stiff (interrupted)',
                }

    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  # determined by solver
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 beta=0.0,
                 method=None,
                 verbosity=-1,  # no messages if negative
                 ):
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.max_step = max_step
        self.first_step = first_step
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        self.beta = beta
        self.verbosity = verbosity
        self.success = 1
        self.set_solout(None)

    def set_solout(self, solout, complex=False):
        self.solout = solout
        self.solout_cmplx = complex
        if solout is None:
            self.iout = 0
        else:
            self.iout = 1

    def reset(self, n, has_jac):
        work = zeros((8 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]
        self.success = 1

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        x, y, iwork, istate = self.runner(*((f, t0, y0, t1) +
                                          tuple(self.call_args) + (f_params,)))
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn(f'{self.__class__.__name__:s}: '
                          f'{self.messages.get(istate, unexpected_istate_msg):s}',
                          stacklevel=2)
            self.success = 0
        return y, x

    def _solout(self, nr, xold, x, y, nd, icomp, con):
        if self.solout is not None:
            if self.solout_cmplx:
                y = y[::2] + 1j * y[1::2]
            return self.solout(x, y)
        else:
            return 1


if dopri5.runner is not None:
    IntegratorBase.integrator_classes.append(dopri5)


class dop853(dopri5):
    runner = getattr(_dop, 'dop853', None)
    name = 'dop853'

    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  # determined by solver
                 safety=0.9,
                 ifactor=6.0,
                 dfactor=0.3,
                 beta=0.0,
                 method=None,
                 verbosity=-1,  # no messages if negative
                 ):
        super().__init__(rtol, atol, nsteps, max_step, first_step, safety,
                         ifactor, dfactor, beta, method, verbosity)

    def reset(self, n, has_jac):
        work = zeros((11 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]
        self.success = 1


if dop853.runner is not None:
    IntegratorBase.integrator_classes.append(dop853)


class lsoda(IntegratorBase):
    runner = getattr(_lsoda, 'lsoda', None)
    active_global_handle = 0

    messages = {
        2: "Integration successful.",
        -1: "Excess work done on this call (perhaps wrong Dfun type).",
        -2: "Excess accuracy requested (tolerances too small).",
        -3: "Illegal input detected (internal error).",
        -4: "Repeated error test failures (internal error).",
        -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
        -6: "Error weight became zero during problem.",
        -7: "Internal workspace insufficient to finish (internal error)."
    }

    def __init__(self,
                 with_jacobian=False,
                 rtol=1e-6, atol=1e-12,
                 lband=None, uband=None,
                 nsteps=500,
                 max_step=0.0,  # corresponds to infinite
                 min_step=0.0,
                 first_step=0.0,  # determined by solver
                 ixpr=0,
                 max_hnil=0,
                 max_order_ns=12,
                 max_order_s=5,
                 method=None
                 ):

        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband

        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.ixpr = ixpr
        self.max_hnil = max_hnil
        self.success = 1

        self.initialized = False

    def reset(self, n, has_jac):
        # Calculate parameters for Fortran subroutine dvode.
        if has_jac:
            if self.mu is None and self.ml is None:
                jt = 1
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 4
        else:
            if self.mu is None and self.ml is None:
                jt = 2
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 5
        lrn = 20 + (self.max_order_ns + 4) * n
        if jt in [1, 2]:
            lrs = 22 + (self.max_order_s + 4) * n + n * n
        elif jt in [4, 5]:
            lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
        else:
            raise ValueError(f'Unexpected jt={jt}')
        lrw = max(lrn, lrs)
        liw = 20 + n
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _lsoda_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.ixpr
        iwork[5] = self.nsteps
        iwork[6] = self.max_hnil
        iwork[7] = self.max_order_ns
        iwork[8] = self.max_order_s
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.rwork, self.iwork, jt]
        self.success = 1
        self.initialized = False

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()
        args = [f, y0, t0, t1] + self.call_args[:-1] + \
               [jac, self.call_args[-1], f_params, 0, jac_params]

        with LSODA_LOCK:
            y1, t, istate = self.runner(*args)

        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn(f'{self.__class__.__name__:s}: '
                          f'{self.messages.get(istate, unexpected_istate_msg):s}',
                          stacklevel=2)
            self.success = 0
        else:
            self.call_args[3] = 2  # upgrade istate from 1 to 2
            self.istate = 2
        return y1, t

    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r


if lsoda.runner:
    IntegratorBase.integrator_classes.append(lsoda)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="vode",
                                   private_modules=["_vode"], all=__all__,
                                   attribute=name)
<|endoftext|>import math
import heapq
import itertools

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, TypeAlias

from scipy._lib._array_api import (
    array_namespace,
    xp_size,
    xp_copy,
    xp_broadcast_promote
)
from scipy._lib._util import MapWrapper

from scipy.integrate._rules import (
    ProductNestedFixed,
    GaussKronrodQuadrature,
    GenzMalikCubature,
)
from scipy.integrate._rules._base import _split_subregion

__all__ = ['cubature']

Array: TypeAlias = Any  # To be changed to an array-api-typing Protocol later


@dataclass
class CubatureRegion:
    estimate: Array
    error: Array
    a: Array
    b: Array
    _xp: ModuleType = field(repr=False)

    def __lt__(self, other):
        # Consider regions with higher error estimates as being "less than" regions with
        # lower order estimates, so that regions with high error estimates are placed at
        # the top of the heap.

        this_err = self._xp.max(self._xp.abs(self.error))
        other_err = self._xp.max(self._xp.abs(other.error))

        return this_err > other_err


@dataclass
class CubatureResult:
    estimate: Array
    error: Array
    status: str
    regions: list[CubatureRegion]
    subdivisions: int
    atol: float
    rtol: float


def cubature(f, a, b, *, rule="gk21", rtol=1e-8, atol=0, max_subdivisions=10000,
             args=(), workers=1, points=None):
    r"""
    Adaptive cubature of multidimensional array-valued function.

    Given an arbitrary integration rule, this function returns an estimate of the
    integral to the requested tolerance over the region defined by the arrays `a` and
    `b` specifying the corners of a hypercube.

    Convergence is not guaranteed for all integrals.

    Parameters
    ----------
    f : callable
        Function to integrate. `f` must have the signature::

            f(x : ndarray, *args) -> ndarray

        `f` should accept arrays ``x`` of shape::

            (npoints, ndim)

        and output arrays of shape::

            (npoints, output_dim_1, ..., output_dim_n)

        In this case, `cubature` will return arrays of shape::

            (output_dim_1, ..., output_dim_n)
    a, b : array_like
        Lower and upper limits of integration as 1D arrays specifying the left and right
        endpoints of the intervals being integrated over. Limits can be infinite.
    rule : str, optional
        Rule used to estimate the integral. If passing a string, the options are
        "gauss-kronrod" (21 node), or "genz-malik" (degree 7). If a rule like
        "gauss-kronrod" is specified for an ``n``-dim integrand, the corresponding
        Cartesian product rule is used. "gk21", "gk15" are also supported for
        compatibility with `quad_vec`. See Notes.
    rtol, atol : float, optional
        Relative and absolute tolerances. Iterations are performed until the error is
        estimated to be less than ``atol + rtol * abs(est)``. Here `rtol` controls
        relative accuracy (number of correct digits), while `atol` controls absolute
        accuracy (number of correct decimal places). To achieve the desired `rtol`, set
        `atol` to be smaller than the smallest value that can be expected from
        ``rtol * abs(y)`` so that `rtol` dominates the allowable error. If `atol` is
        larger than ``rtol * abs(y)`` the number of correct digits is not guaranteed.
        Conversely, to achieve the desired `atol`, set `rtol` such that
        ``rtol * abs(y)`` is always smaller than `atol`. Default values are 1e-8 for
        `rtol` and 0 for `atol`.
    max_subdivisions : int, optional
        Upper bound on the number of subdivisions to perform. Default is 10,000.
    args : tuple, optional
        Additional positional args passed to `f`, if any.
    workers : int or map-like callable, optional
        If `workers` is an integer, part of the computation is done in parallel
        subdivided to this many tasks (using :class:`python:multiprocessing.pool.Pool`).
        Supply `-1` to use all cores available to the Process. Alternatively, supply a
        map-like callable, such as :meth:`python:multiprocessing.pool.Pool.map` for
        evaluating the population in parallel. This evaluation is carried out as
        ``workers(func, iterable)``.
    points : list of array_like, optional
        List of points to avoid evaluating `f` at, under the condition that the rule
        being used does not evaluate `f` on the boundary of a region (which is the
        case for all Genz-Malik and Gauss-Kronrod rules). This can be useful if `f` has
        a singularity at the specified point. This should be a list of array-likes where
        each element has length ``ndim``. Default is empty. See Examples.

    Returns
    -------
    res : object
        Object containing the results of the estimation. It has the following
        attributes:

        estimate : ndarray
            Estimate of the value of the integral over the overall region specified.
        error : ndarray
            Estimate of the error of the approximation over the overall region
            specified.
        status : str
            Whether the estimation was successful. Can be either: "converged",
            "not_converged".
        subdivisions : int
            Number of subdivisions performed.
        atol, rtol : float
            Requested tolerances for the approximation.
        regions: list of object
            List of objects containing the estimates of the integral over smaller
            regions of the domain.

        Each object in ``regions`` has the following attributes:

        a, b : ndarray
            Points describing the corners of the region. If the original integral
            contained infinite limits or was over a region described by `region`,
            then `a` and `b` are in the transformed coordinates.
        estimate : ndarray
            Estimate of the value of the integral over this region.
        error : ndarray
            Estimate of the error of the approximation over this region.

    Notes
    -----
    The algorithm uses a similar algorithm to `quad_vec`, which itself is based on the
    implementation of QUADPACK's DQAG* algorithms, implementing global error control and
    adaptive subdivision.

    The source of the nodes and weights used for Gauss-Kronrod quadrature can be found
    in [1]_, and the algorithm for calculating the nodes and weights in Genz-Malik
    cubature can be found in [2]_.

    The rules currently supported via the `rule` argument are:

    - ``"gauss-kronrod"``, 21-node Gauss-Kronrod
    - ``"genz-malik"``, n-node Genz-Malik

    If using Gauss-Kronrod for an ``n``-dim integrand where ``n > 2``, then the
    corresponding Cartesian product rule will be found by taking the Cartesian product
    of the nodes in the 1D case. This means that the number of nodes scales
    exponentially as ``21^n`` in the Gauss-Kronrod case, which may be problematic in a
    moderate number of dimensions.

    Genz-Malik is typically less accurate than Gauss-Kronrod but has much fewer nodes,
    so in this situation using "genz-malik" might be preferable.

    Infinite limits are handled with an appropriate variable transformation. Assuming
    ``a = [a_1, ..., a_n]`` and ``b = [b_1, ..., b_n]``:

    If :math:`a_i = -\infty` and :math:`b_i = \infty`, the i-th integration variable
    will use the transformation :math:`x = \frac{1-|t|}{t}` and :math:`t \in (-1, 1)`.

    If :math:`a_i \ne \pm\infty` and :math:`b_i = \infty`, the i-th integration variable
    will use the transformation :math:`x = a_i + \frac{1-t}{t}` and
    :math:`t \in (0, 1)`.

    If :math:`a_i = -\infty` and :math:`b_i \ne \pm\infty`, the i-th integration
    variable will use the transformation :math:`x = b_i - \frac{1-t}{t}` and
    :math:`t \in (0, 1)`.

    References
    ----------
    .. [1] R. Piessens, E. de Doncker, Quadpack: A Subroutine Package for Automatic
        Integration, files: dqk21.f, dqk15.f (1983).

    .. [2] A.C. Genz, A.A. Malik, Remarks on algorithm 006: An adaptive algorithm for
        numerical integration over an N-dimensional rectangular region, Journal of
        Computational and Applied Mathematics, Volume 6, Issue 4, 1980, Pages 295-302,
        ISSN 0377-0427
        :doi:`10.1016/0771-050X(80)90039-X`

    Examples
    --------
    **1D integral with vector output**:

    .. math::

        \int^1_0 \mathbf f(x) \text dx

    Where ``f(x) = x^n`` and ``n = np.arange(10)`` is a vector. Since no rule is
    specified, the default "gk21" is used, which corresponds to Gauss-Kronrod
    integration with 21 nodes.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> def f(x, n):
    ...    # Make sure x and n are broadcastable
    ...    return x[:, np.newaxis]**n[np.newaxis, :]
    >>> res = cubature(
    ...     f,
    ...     a=[0],
    ...     b=[1],
    ...     args=(np.arange(10),),
    ... )
    >>> res.estimate
     array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ,
            0.16666667, 0.14285714, 0.125     , 0.11111111, 0.1       ])

    **7D integral with arbitrary-shaped array output**::

        f(x) = cos(2*pi*r + alphas @ x)

    for some ``r`` and ``alphas``, and the integral is performed over the unit
    hybercube, :math:`[0, 1]^7`. Since the integral is in a moderate number of
    dimensions, "genz-malik" is used rather than the default "gauss-kronrod" to
    avoid constructing a product rule with :math:`21^7 \approx 2 \times 10^9` nodes.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> def f(x, r, alphas):
    ...     # f(x) = cos(2*pi*r + alphas @ x)
    ...     # Need to allow r and alphas to be arbitrary shape
    ...     npoints, ndim = x.shape[0], x.shape[-1]
    ...     alphas = alphas[np.newaxis, ...]
    ...     x = x.reshape(npoints, *([1]*(len(alphas.shape) - 1)), ndim)
    ...     return np.cos(2*np.pi*r + np.sum(alphas * x, axis=-1))
    >>> rng = np.random.default_rng()
    >>> r, alphas = rng.random((2, 3)), rng.random((2, 3, 7))
    >>> res = cubature(
    ...     f=f,
    ...     a=np.array([0, 0, 0, 0, 0, 0, 0]),
    ...     b=np.array([1, 1, 1, 1, 1, 1, 1]),
    ...     rtol=1e-5,
    ...     rule="genz-malik",
    ...     args=(r, alphas),
    ... )
    >>> res.estimate
     array([[-0.79812452,  0.35246913, -0.52273628],
            [ 0.88392779,  0.59139899,  0.41895111]])

    **Parallel computation with** `workers`:

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> with ThreadPoolExecutor() as executor:
    ...     res = cubature(
    ...         f=f,
    ...         a=np.array([0, 0, 0, 0, 0, 0, 0]),
    ...         b=np.array([1, 1, 1, 1, 1, 1, 1]),
    ...         rtol=1e-5,
    ...         rule="genz-malik",
    ...         args=(r, alphas),
    ...         workers=executor.map,
    ...      )
    >>> res.estimate
     array([[-0.79812452,  0.35246913, -0.52273628],
            [ 0.88392779,  0.59139899,  0.41895111]])

    **2D integral with infinite limits**:

    .. math::

        \int^{ \infty }_{ -\infty }
        \int^{ \infty }_{ -\infty }
            e^{-x^2-y^2}
        \text dy
        \text dx

    >>> def gaussian(x):
    ...     return np.exp(-np.sum(x**2, axis=-1))
    >>> res = cubature(gaussian, [-np.inf, -np.inf], [np.inf, np.inf])
    >>> res.estimate
     3.1415926

    **1D integral with singularities avoided using** `points`:

    .. math::

        \int^{ 1 }_{ -1 }
          \frac{\sin(x)}{x}
        \text dx

    It is necessary to use the `points` parameter to avoid evaluating `f` at the origin.

    >>> def sinc(x):
    ...     return np.sin(x)/x
    >>> res = cubature(sinc, [-1], [1], points=[[0]])
    >>> res.estimate
     1.8921661
    """

    # It is also possible to use a custom rule, but this is not yet part of the public
    # API. An example of this can be found in the class scipy.integrate._rules.Rule.

    xp = array_namespace(a, b)
    max_subdivisions = float("inf") if max_subdivisions is None else max_subdivisions
    points = [] if points is None else points

    # Convert a and b to arrays and convert each point in points to an array, promoting
    # each to a common floating dtype.
    a, b, *points = xp_broadcast_promote(a, b, *points, force_floating=True)
    result_dtype = a.dtype

    if xp_size(a) == 0 or xp_size(b) == 0:
        raise ValueError("`a` and `b` must be nonempty")

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("`a` and `b` must be 1D arrays")

    # If the rule is a string, convert to a corresponding product rule
    if isinstance(rule, str):
        ndim = xp_size(a)

        if rule == "genz-malik":
            rule = GenzMalikCubature(ndim, xp=xp)
        else:
            quadratues = {
                "gauss-kronrod": GaussKronrodQuadrature(21, xp=xp),

                # Also allow names quad_vec uses:
                "gk21": GaussKronrodQuadrature(21, xp=xp),
                "gk15": GaussKronrodQuadrature(15, xp=xp),
            }

            base_rule = quadratues.get(rule)

            if base_rule is None:
                raise ValueError(f"unknown rule {rule}")

            rule = ProductNestedFixed([base_rule] * ndim)

    # If any of limits are the wrong way around (a > b), flip them and keep track of
    # the sign.
    sign = (-1) ** xp.sum(xp.astype(a > b, xp.int8), dtype=result_dtype)

    a_flipped = xp.min(xp.stack([a, b]), axis=0)
    b_flipped = xp.max(xp.stack([a, b]), axis=0)

    a, b = a_flipped, b_flipped

    # If any of the limits are infinite, apply a transformation
    if xp.any(xp.isinf(a)) or xp.any(xp.isinf(b)):
        f = _InfiniteLimitsTransform(f, a, b, xp=xp)
        a, b = f.transformed_limits

        # Map points from the original coordinates to the new transformed coordinates.
        #
        # `points` is a list of arrays of shape (ndim,), but transformations are applied
        # to arrays of shape (npoints, ndim).
        #
        # It is not possible to combine all the points into one array and then apply
        # f.inv to all of them at once since `points` needs to remain iterable.
        # Instead, each point is reshaped to an array of shape (1, ndim), `f.inv` is
        # applied, and then each is reshaped back to (ndim,).
        points = [xp.reshape(point, (1, -1)) for point in points]
        points = [f.inv(point) for point in points]
        points = [xp.reshape(point, (-1,)) for point in points]

        # Include any problematic points introduced by the transformation
        points.extend(f.points)

    # If any problematic points are specified, divide the initial region so that these
    # points lie on the edge of a subregion.
    #
    # This means ``f`` won't be evaluated there if the rule being used has no evaluation
    # points on the boundary.
    if len(points) == 0:
        initial_regions = [(a, b)]
    else:
        initial_regions = _split_region_at_points(a, b, points, xp)

    regions = []
    est = 0.0
    err = 0.0

    for a_k, b_k in initial_regions:
        est_k = rule.estimate(f, a_k, b_k, args)
        err_k = rule.estimate_error(f, a_k, b_k, args)
        regions.append(CubatureRegion(est_k, err_k, a_k, b_k, xp))

        est += est_k
        err += err_k

    subdivisions = 0
    success = True

    with MapWrapper(workers) as mapwrapper:
        while xp.any(err > atol + rtol * xp.abs(est)):
            # region_k is the region with highest estimated error
            region_k = heapq.heappop(regions)

            est_k = region_k.estimate
            err_k = region_k.error

            a_k, b_k = region_k.a, region_k.b

            # Subtract the estimate of the integral and its error over this region from
            # the current global estimates, since these will be refined in the loop over
            # all subregions.
            est -= est_k
            err -= err_k

            # Find all 2^ndim subregions formed by splitting region_k along each axis,
            # e.g. for 1D integrals this splits an estimate over an interval into an
            # estimate over two subintervals, for 3D integrals this splits an estimate
            # over a cube into 8 subcubes.
            #
            # For each of the new subregions, calculate an estimate for the integral and
            # the error there, and push these regions onto the heap for potential
            # further subdividing.

            executor_args = zip(
                itertools.repeat(f),
                itertools.repeat(rule),
                itertools.repeat(args),
                _split_subregion(a_k, b_k, xp),
            )

            for subdivision_result in mapwrapper(_process_subregion, executor_args):
                a_k_sub, b_k_sub, est_sub, err_sub = subdivision_result

                est += est_sub
                err += err_sub

                new_region = CubatureRegion(est_sub, err_sub, a_k_sub, b_k_sub, xp)

                heapq.heappush(regions, new_region)

            subdivisions += 1

            if subdivisions >= max_subdivisions:
                success = False
                break

        status = "converged" if success else "not_converged"

        # Apply sign change to handle any limits which were initially flipped.
        est = sign * est

        return CubatureResult(
            estimate=est,
            error=err,
            status=status,
            subdivisions=subdivisions,
            regions=regions,
            atol=atol,
            rtol=rtol,
        )


def _process_subregion(data):
    f, rule, args, coord = data
    a_k_sub, b_k_sub = coord

    est_sub = rule.estimate(f, a_k_sub, b_k_sub, args)
    err_sub = rule.estimate_error(f, a_k_sub, b_k_sub, args)

    return a_k_sub, b_k_sub, est_sub, err_sub


def _is_strictly_in_region(a, b, point, xp):
    if xp.all(point == a) or xp.all(point == b):
        return False

    return xp.all(a <= point) and xp.all(point <= b)


def _split_region_at_points(a, b, points, xp):
    """
    Given the integration limits `a` and `b` describing a rectangular region and a list
    of `points`, find the list of ``[(a_1, b_1), ..., (a_l, b_l)]`` which breaks up the
    initial region into smaller subregion such that no `points` lie strictly inside
    any of the subregions.
    """

    regions = [(a, b)]

    for point in points:
        if xp.any(xp.isinf(point)):
            # If a point is specified at infinity, ignore.
            #
            # This case occurs when points are given by the user to avoid, but after
            # applying a transformation, they are removed.
            continue

        new_subregions = []

        for a_k, b_k in regions:
            if _is_strictly_in_region(a_k, b_k, point, xp):
                subregions = _split_subregion(a_k, b_k, xp, point)

                for left, right in subregions:
                    # Skip any zero-width regions.
                    if xp.any(left == right):
                        continue
                    else:
                        new_subregions.append((left, right))

                new_subregions.extend(subregions)

            else:
                new_subregions.append((a_k, b_k))

        regions = new_subregions

    return regions


class _VariableTransform:
    """
    A transformation that can be applied to an integral.
    """

    @property
    def transformed_limits(self):
        """
        New limits of integration after applying the transformation.
        """

        raise NotImplementedError

    @property
    def points(self):
        """
        Any problematic points introduced by the transformation.

        These should be specified as points where ``_VariableTransform(f)(self, point)``
        would be problematic.

        For example, if the transformation ``x = 1/((1-t)(1+t))`` is applied to a
        univariate integral, then points should return ``[ [1], [-1] ]``.
        """

        return []

    def inv(self, x):
        """
        Map points ``x`` to ``t`` such that if ``f`` is the original function and ``g``
        is the function after the transformation is applied, then::

            f(x) = g(self.inv(x))
        """

        raise NotImplementedError

    def __call__(self, t, *args, **kwargs):
        """
        Apply the transformation to ``f`` and multiply by the Jacobian determinant.
        This should be the new integrand after the transformation has been applied so
        that the following is satisfied::

            f_transformed = _VariableTransform(f)

            cubature(f, a, b) == cubature(
                f_transformed,
                *f_transformed.transformed_limits(a, b),
            )
        """

        raise NotImplementedError


class _InfiniteLimitsTransform(_VariableTransform):
    r"""
    Transformation for handling infinite limits.

    Assuming ``a = [a_1, ..., a_n]`` and ``b = [b_1, ..., b_n]``:

    If :math:`a_i = -\infty` and :math:`b_i = \infty`, the i-th integration variable
    will use the transformation :math:`x = \frac{1-|t|}{t}` and :math:`t \in (-1, 1)`.

    If :math:`a_i \ne \pm\infty` and :math:`b_i = \infty`, the i-th integration variable
    will use the transformation :math:`x = a_i + \frac{1-t}{t}` and
    :math:`t \in (0, 1)`.

    If :math:`a_i = -\infty` and :math:`b_i \ne \pm\infty`, the i-th integration
    variable will use the transformation :math:`x = b_i - \frac{1-t}{t}` and
    :math:`t \in (0, 1)`.
    """

    def __init__(self, f, a, b, xp):
        self._xp = xp

        self._f = f
        self._orig_a = a
        self._orig_b = b

        # (-oo, oo) will be mapped to (-1, 1).
        self._double_inf_pos = (a == -math.inf) & (b == math.inf)

        # (start, oo) will be mapped to (0, 1).
        start_inf_mask = (a != -math.inf) & (b == math.inf)

        # (-oo, end) will be mapped to (0, 1).
        inf_end_mask = (a == -math.inf) & (b != math.inf)

        # This is handled by making the transformation t = -x and reducing it to
        # the other semi-infinite case.
        self._semi_inf_pos = start_inf_mask | inf_end_mask

        # Since we flip the limits, we don't need to separately multiply the
        # integrand by -1.
        self._orig_a[inf_end_mask] = -b[inf_end_mask]
        self._orig_b[inf_end_mask] = -a[inf_end_mask]

        self._num_inf = self._xp.sum(
            self._xp.astype(self._double_inf_pos | self._semi_inf_pos, self._xp.int64),
        ).__int__()

    @property
    def transformed_limits(self):
        a = xp_copy(self._orig_a)
        b = xp_copy(self._orig_b)

        a[self._double_inf_pos] = -1
        b[self._double_inf_pos] = 1

        a[self._semi_inf_pos] = 0
        b[self._semi_inf_pos] = 1

        return a, b

    @property
    def points(self):
        # If there are infinite limits, then the origin becomes a problematic point
        # due to a division by zero there.

        # If the function using this class only wraps f when a and b contain infinite
        # limits, this condition will always be met (as is the case with cubature).
        #
        # If a and b do not contain infinite limits but f is still wrapped with this
        # class, then without this condition the initial region of integration will
        # be split around the origin unnecessarily.
        if self._num_inf != 0:
            return [self._xp.zeros(self._orig_a.shape)]
        else:
            return []

    def inv(self, x):
        t = xp_copy(x)
        npoints = x.shape[0]

        double_inf_mask = self._xp.tile(
            self._double_inf_pos[self._xp.newaxis, :],
            (npoints, 1),
        )

        semi_inf_mask = self._xp.tile(
            self._semi_inf_pos[self._xp.newaxis, :],
            (npoints, 1),
        )

        # If any components of x are 0, then this component will be mapped to infinity
        # under the transformation used for doubly-infinite limits.
        #
        # Handle the zero values and non-zero values separately to avoid division by
        # zero.
        zero_mask = x[double_inf_mask] == 0
        non_zero_mask = double_inf_mask & ~zero_mask
        t[zero_mask] = math.inf
        t[non_zero_mask] = 1/(x[non_zero_mask] + self._xp.sign(x[non_zero_mask]))

        start = self._xp.tile(self._orig_a[self._semi_inf_pos], (npoints,))
        t[semi_inf_mask] = 1/(x[semi_inf_mask] - start + 1)

        return t

    def __call__(self, t, *args, **kwargs):
        x = xp_copy(t)
        npoints = t.shape[0]

        double_inf_mask = self._xp.tile(
            self._double_inf_pos[self._xp.newaxis, :],
            (npoints, 1),
        )

        semi_inf_mask = self._xp.tile(
            self._semi_inf_pos[self._xp.newaxis, :],
            (npoints, 1),
        )

        # For (-oo, oo) -> (-1, 1), use the transformation x = (1-|t|)/t.
        x[double_inf_mask] = (
            (1 - self._xp.abs(t[double_inf_mask])) / t[double_inf_mask]
        )

        start = self._xp.tile(self._orig_a[self._semi_inf_pos], (npoints,))

        # For (start, oo) -> (0, 1), use the transformation x = start + (1-t)/t.
        x[semi_inf_mask] = start + (1 - t[semi_inf_mask]) / t[semi_inf_mask]

        jacobian_det = 1/self._xp.prod(
            self._xp.reshape(
                t[semi_inf_mask | double_inf_mask]**2,
                (-1, self._num_inf),
            ),
            axis=-1,
        )

        f_x = self._f(x, *args, **kwargs)
        jacobian_det = self._xp.reshape(jacobian_det, (-1, *([1]*(len(f_x.shape) - 1))))

        return f_x * jacobian_det
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.integrate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    "quad",
    "dblquad",
    "tplquad",
    "nquad",
    "IntegrationWarning",
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="quadpack",
                                   private_modules=["_quadpack_py"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ['lsoda']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="lsoda",
                                   private_modules=["_lsoda"], all=__all__,
                                   attribute=name)
<|endoftext|>from scipy._lib._array_api import np_compat, array_namespace

from functools import cached_property

from ._base import NestedFixedRule
from ._gauss_legendre import GaussLegendreQuadrature


class GaussKronrodQuadrature(NestedFixedRule):
    """
    Gauss-Kronrod quadrature.

    Gauss-Kronrod rules consist of two quadrature rules, one higher-order and one
    lower-order. The higher-order rule is used as the estimate of the integral and the
    difference between them is used as an estimate for the error.

    Gauss-Kronrod is a 1D rule. To use it for multidimensional integrals, it will be
    necessary to use ProductNestedFixed and multiple Gauss-Kronrod rules. See Examples.

    For n-node Gauss-Kronrod, the lower-order rule has ``n//2`` nodes, which are the
    ordinary Gauss-Legendre nodes with corresponding weights. The higher-order rule has
    ``n`` nodes, ``n//2`` of which are the same as the lower-order rule and the
    remaining nodes are the Kronrod extension of those nodes.

    Parameters
    ----------
    npoints : int
        Number of nodes for the higher-order rule.

    xp : array_namespace, optional
        The namespace for the node and weight arrays. Default is None, where NumPy is
        used.

    Attributes
    ----------
    lower : Rule
        Lower-order rule.

    References
    ----------
    .. [1] R. Piessens, E. de Doncker, Quadpack: A Subroutine Package for Automatic
        Integration, files: dqk21.f, dqk15.f (1983).

    Examples
    --------
    Evaluate a 1D integral. Note in this example that ``f`` returns an array, so the
    estimates will also be arrays, despite the fact that this is a 1D problem.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import GaussKronrodQuadrature
    >>> def f(x):
    ...     return np.cos(x)
    >>> rule = GaussKronrodQuadrature(21) # Use 21-point GaussKronrod
    >>> a, b = np.array([0]), np.array([1])
    >>> rule.estimate(f, a, b) # True value sin(1), approximately 0.84147
     array([0.84147098])
    >>> rule.estimate_error(f, a, b)
     array([1.11022302e-16])

    Evaluate a 2D integral. Note that in this example ``f`` returns a float, so the
    estimates will also be floats.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import (
    ...     ProductNestedFixed, GaussKronrodQuadrature
    ... )
    >>> def f(x):
    ...     # f(x) = cos(x_1) + cos(x_2)
    ...     return np.sum(np.cos(x), axis=-1)
    >>> rule = ProductNestedFixed(
    ...     [GaussKronrodQuadrature(15), GaussKronrodQuadrature(15)]
    ... ) # Use 15-point Gauss-Kronrod
    >>> a, b = np.array([0, 0]), np.array([1, 1])
    >>> rule.estimate(f, a, b) # True value 2*sin(1), approximately 1.6829
     np.float64(1.682941969615793)
    >>> rule.estimate_error(f, a, b)
     np.float64(2.220446049250313e-16)
    """

    def __init__(self, npoints, xp=None):
        # TODO: nodes and weights are currently hard-coded for values 15 and 21, but in
        # the future it would be best to compute the Kronrod extension of the lower rule
        if npoints != 15 and npoints != 21:
            raise NotImplementedError("Gauss-Kronrod quadrature is currently only"
                                      "supported for 15 or 21 nodes")

        self.npoints = npoints

        if xp is None:
            xp = np_compat

        self.xp = array_namespace(xp.empty(0))

        self.gauss = GaussLegendreQuadrature(npoints//2, xp=self.xp)

    @cached_property
    def nodes_and_weights(self):
        # These values are from QUADPACK's `dqk21.f` and `dqk15.f` (1983).
        if self.npoints == 21:
            nodes = self.xp.asarray(
                [
                    0.995657163025808080735527280689003,
                    0.973906528517171720077964012084452,
                    0.930157491355708226001207180059508,
                    0.865063366688984510732096688423493,
                    0.780817726586416897063717578345042,
                    0.679409568299024406234327365114874,
                    0.562757134668604683339000099272694,
                    0.433395394129247190799265943165784,
                    0.294392862701460198131126603103866,
                    0.148874338981631210884826001129720,
                    0,
                    -0.148874338981631210884826001129720,
                    -0.294392862701460198131126603103866,
                    -0.433395394129247190799265943165784,
                    -0.562757134668604683339000099272694,
                    -0.679409568299024406234327365114874,
                    -0.780817726586416897063717578345042,
                    -0.865063366688984510732096688423493,
                    -0.930157491355708226001207180059508,
                    -0.973906528517171720077964012084452,
                    -0.995657163025808080735527280689003,
                ],
                dtype=self.xp.float64,
            )

            weights = self.xp.asarray(
                [
                    0.011694638867371874278064396062192,
                    0.032558162307964727478818972459390,
                    0.054755896574351996031381300244580,
                    0.075039674810919952767043140916190,
                    0.093125454583697605535065465083366,
                    0.109387158802297641899210590325805,
                    0.123491976262065851077958109831074,
                    0.134709217311473325928054001771707,
                    0.142775938577060080797094273138717,
                    0.147739104901338491374841515972068,
                    0.149445554002916905664936468389821,
                    0.147739104901338491374841515972068,
                    0.142775938577060080797094273138717,
                    0.134709217311473325928054001771707,
                    0.123491976262065851077958109831074,
                    0.109387158802297641899210590325805,
                    0.093125454583697605535065465083366,
                    0.075039674810919952767043140916190,
                    0.054755896574351996031381300244580,
                    0.032558162307964727478818972459390,
                    0.011694638867371874278064396062192,
                ],
                dtype=self.xp.float64,
            )
        elif self.npoints == 15:
            nodes = self.xp.asarray(
                [
                    0.991455371120812639206854697526329,
                    0.949107912342758524526189684047851,
                    0.864864423359769072789712788640926,
                    0.741531185599394439863864773280788,
                    0.586087235467691130294144838258730,
                    0.405845151377397166906606412076961,
                    0.207784955007898467600689403773245,
                    0.000000000000000000000000000000000,
                    -0.207784955007898467600689403773245,
                    -0.405845151377397166906606412076961,
                    -0.586087235467691130294144838258730,
                    -0.741531185599394439863864773280788,
                    -0.864864423359769072789712788640926,
                    -0.949107912342758524526189684047851,
                    -0.991455371120812639206854697526329,
                ],
                dtype=self.xp.float64,
            )

            weights = self.xp.asarray(
                [
                    0.022935322010529224963732008058970,
                    0.063092092629978553290700663189204,
                    0.104790010322250183839876322541518,
                    0.140653259715525918745189590510238,
                    0.169004726639267902826583426598550,
                    0.190350578064785409913256402421014,
                    0.204432940075298892414161999234649,
                    0.209482141084727828012999174891714,
                    0.204432940075298892414161999234649,
                    0.190350578064785409913256402421014,
                    0.169004726639267902826583426598550,
                    0.140653259715525918745189590510238,
                    0.104790010322250183839876322541518,
                    0.063092092629978553290700663189204,
                    0.022935322010529224963732008058970,
                ],
                dtype=self.xp.float64,
            )

        return nodes, weights

    @property
    def lower_nodes_and_weights(self):
        return self.gauss.nodes_and_weights
<|endoftext|>from scipy._lib._array_api import array_namespace, np_compat

from functools import cached_property

from scipy.special import roots_legendre

from ._base import FixedRule


class GaussLegendreQuadrature(FixedRule):
    """
    Gauss-Legendre quadrature.

    Parameters
    ----------
    npoints : int
        Number of nodes for the higher-order rule.

    xp : array_namespace, optional
        The namespace for the node and weight arrays. Default is None, where NumPy is
        used.

    Examples
    --------
    Evaluate a 1D integral. Note in this example that ``f`` returns an array, so the
    estimates will also be arrays.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import GaussLegendreQuadrature
    >>> def f(x):
    ...     return np.cos(x)
    >>> rule = GaussLegendreQuadrature(21) # Use 21-point GaussLegendre
    >>> a, b = np.array([0]), np.array([1])
    >>> rule.estimate(f, a, b) # True value sin(1), approximately 0.84147
     array([0.84147098])
    >>> rule.estimate_error(f, a, b)
     array([1.11022302e-16])
    """

    def __init__(self, npoints, xp=None):
        if npoints < 2:
            raise ValueError(
                "At least 2 nodes required for Gauss-Legendre cubature"
            )

        self.npoints = npoints

        if xp is None:
            xp = np_compat

        self.xp = array_namespace(xp.empty(0))

    @cached_property
    def nodes_and_weights(self):
        # TODO: current converting to/from numpy
        nodes, weights = roots_legendre(self.npoints)

        return (
            self.xp.asarray(nodes, dtype=self.xp.float64),
            self.xp.asarray(weights, dtype=self.xp.float64)
        )
<|endoftext|>import math
import itertools

from functools import cached_property

from scipy._lib._array_api import array_namespace, np_compat

from scipy.integrate._rules import NestedFixedRule


class GenzMalikCubature(NestedFixedRule):
    """
    Genz-Malik cubature.

    Genz-Malik is only defined for integrals of dimension >= 2.

    Parameters
    ----------
    ndim : int
        The spatial dimension of the integrand.

    xp : array_namespace, optional
        The namespace for the node and weight arrays. Default is None, where NumPy is
        used.

    Attributes
    ----------
    higher : Cubature
        Higher-order rule.

    lower : Cubature
        Lower-order rule.

    References
    ----------
    .. [1] A.C. Genz, A.A. Malik, Remarks on algorithm 006: An adaptive algorithm for
        numerical integration over an N-dimensional rectangular region, Journal of
        Computational and Applied Mathematics, Volume 6, Issue 4, 1980, Pages 295-302,
        ISSN 0377-0427, https://doi.org/10.1016/0771-050X(80)90039-X.

    Examples
    --------
    Evaluate a 3D integral:

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import GenzMalikCubature
    >>> def f(x):
    ...     # f(x) = cos(x_1) + cos(x_2) + cos(x_3)
    ...     return np.sum(np.cos(x), axis=-1)
    >>> rule = GenzMalikCubature(3) # Use 3D Genz-Malik
    >>> a, b = np.array([0, 0, 0]), np.array([1, 1, 1])
    >>> rule.estimate(f, a, b) # True value 3*sin(1), approximately 2.5244
     np.float64(2.5244129547230862)
    >>> rule.estimate_error(f, a, b)
     np.float64(1.378269656626685e-06)
    """

    def __init__(self, ndim, degree=7, lower_degree=5, xp=None):
        if ndim < 2:
            raise ValueError("Genz-Malik cubature is only defined for ndim >= 2")

        if degree != 7 or lower_degree != 5:
            raise NotImplementedError("Genz-Malik cubature is currently only supported"
                                      "for degree=7, lower_degree=5")

        self.ndim = ndim
        self.degree = degree
        self.lower_degree = lower_degree

        if xp is None:
            xp = np_compat

        self.xp = array_namespace(xp.empty(0))

    @cached_property
    def nodes_and_weights(self):
        # TODO: Currently only support for degree 7 Genz-Malik cubature, should aim to
        # support arbitrary degree
        l_2 = math.sqrt(9/70)
        l_3 = math.sqrt(9/10)
        l_4 = math.sqrt(9/10)
        l_5 = math.sqrt(9/19)

        its = itertools.chain(
            [(0,) * self.ndim],
            _distinct_permutations((l_2,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((-l_2,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((l_3,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((-l_3,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((l_4, l_4) + (0,) * (self.ndim - 2)),
            _distinct_permutations((l_4, -l_4) + (0,) * (self.ndim - 2)),
            _distinct_permutations((-l_4, -l_4) + (0,) * (self.ndim - 2)),
            itertools.product((l_5, -l_5), repeat=self.ndim),
        )

        nodes_size = 1 + (2 * (self.ndim + 1) * self.ndim) + 2**self.ndim

        nodes = self.xp.asarray(
            list(zip(*its)),
            dtype=self.xp.float64,
        )

        nodes = self.xp.reshape(nodes, (self.ndim, nodes_size))

        # It's convenient to generate the nodes as a sequence of evaluation points
        # as an array of shape (npoints, ndim), but nodes needs to have shape
        # (ndim, npoints)
        nodes = nodes.T

        w_1 = (
            (2**self.ndim) * (12824 - 9120*self.ndim + (400 * self.ndim**2)) / 19683
        )
        w_2 = (2**self.ndim) * 980/6561
        w_3 = (2**self.ndim) * (1820 - 400 * self.ndim) / 19683
        w_4 = (2**self.ndim) * (200 / 19683)
        w_5 = 6859 / 19683

        weights = self.xp.concat([
            self.xp.asarray([w_1] * 1, dtype=self.xp.float64),
            self.xp.asarray([w_2] * (2 * self.ndim), dtype=self.xp.float64),
            self.xp.asarray([w_3] * (2 * self.ndim), dtype=self.xp.float64),
            self.xp.asarray(
                [w_4] * (2 * (self.ndim - 1) * self.ndim),
                dtype=self.xp.float64,
            ),
            self.xp.asarray([w_5] * (2**self.ndim), dtype=self.xp.float64),
        ])

        return nodes, weights

    @cached_property
    def lower_nodes_and_weights(self):
        # TODO: Currently only support for the degree 5 lower rule, in the future it
        # would be worth supporting arbitrary degree

        # Nodes are almost the same as the full rule, but there are no nodes
        # corresponding to l_5.
        l_2 = math.sqrt(9/70)
        l_3 = math.sqrt(9/10)
        l_4 = math.sqrt(9/10)

        its = itertools.chain(
            [(0,) * self.ndim],
            _distinct_permutations((l_2,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((-l_2,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((l_3,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((-l_3,) + (0,) * (self.ndim - 1)),
            _distinct_permutations((l_4, l_4) + (0,) * (self.ndim - 2)),
            _distinct_permutations((l_4, -l_4) + (0,) * (self.ndim - 2)),
            _distinct_permutations((-l_4, -l_4) + (0,) * (self.ndim - 2)),
        )

        nodes_size = 1 + (2 * (self.ndim + 1) * self.ndim)

        nodes = self.xp.asarray(list(zip(*its)), dtype=self.xp.float64)
        nodes = self.xp.reshape(nodes, (self.ndim, nodes_size))
        nodes = nodes.T

        # Weights are different from those in the full rule.
        w_1 = (2**self.ndim) * (729 - 950*self.ndim + 50*self.ndim**2) / 729
        w_2 = (2**self.ndim) * (245 / 486)
        w_3 = (2**self.ndim) * (265 - 100*self.ndim) / 1458
        w_4 = (2**self.ndim) * (25 / 729)

        weights = self.xp.concat([
            self.xp.asarray([w_1] * 1, dtype=self.xp.float64),
            self.xp.asarray([w_2] * (2 * self.ndim), dtype=self.xp.float64),
            self.xp.asarray([w_3] * (2 * self.ndim), dtype=self.xp.float64),
            self.xp.asarray(
                [w_4] * (2 * (self.ndim - 1) * self.ndim),
                dtype=self.xp.float64,
            ),
        ])

        return nodes, weights


def _distinct_permutations(iterable):
    """
    Find the number of distinct permutations of elements of `iterable`.
    """

    # Algorithm: https://w.wiki/Qai

    items = sorted(iterable)
    size = len(items)

    while True:
        # Yield the permutation we have
        yield tuple(items)

        # Find the largest index i such that A[i] < A[i + 1]
        for i in range(size - 2, -1, -1):
            if items[i] < items[i + 1]:
                break

        #  If no such index exists, this permutation is the last one
        else:
            return

        # Find the largest index j greater than j such that A[i] < A[j]
        for j in range(size - 1, i, -1):
            if items[i] < items[j]:
                break

        # Swap the value of A[i] with that of A[j], then reverse the
        # sequence from A[i + 1] to form the new permutation
        items[i], items[j] = items[j], items[i]
        items[i+1:] = items[:i-size:-1]  # A[i + 1:][::-1]
<|endoftext|>"""Numerical cubature algorithms"""

from ._base import (
    Rule, FixedRule,
    NestedFixedRule,
    ProductNestedFixed,
)
from ._genz_malik import GenzMalikCubature
from ._gauss_kronrod import GaussKronrodQuadrature
from ._gauss_legendre import GaussLegendreQuadrature

__all__ = [s for s in dir() if not s.startswith('_')]
<|endoftext|>from scipy._lib._array_api import array_namespace, xp_size

from functools import cached_property


class Rule:
    """
    Base class for numerical integration algorithms (cubatures).

    Finds an estimate for the integral of ``f`` over the region described by two arrays
    ``a`` and ``b`` via `estimate`, and find an estimate for the error of this
    approximation via `estimate_error`.

    If a subclass does not implement its own `estimate_error`, then it will use a
    default error estimate based on the difference between the estimate over the whole
    region and the sum of estimates over that region divided into ``2^ndim`` subregions.

    See Also
    --------
    FixedRule

    Examples
    --------
    In the following, a custom rule is created which uses 3D Genz-Malik cubature for
    the estimate of the integral, and the difference between this estimate and a less
    accurate estimate using 5-node Gauss-Legendre quadrature as an estimate for the
    error.

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import (
    ...     Rule, ProductNestedFixed, GenzMalikCubature, GaussLegendreQuadrature
    ... )
    >>> def f(x, r, alphas):
    ...     # f(x) = cos(2*pi*r + alpha @ x)
    ...     # Need to allow r and alphas to be arbitrary shape
    ...     npoints, ndim = x.shape[0], x.shape[-1]
    ...     alphas_reshaped = alphas[np.newaxis, :]
    ...     x_reshaped = x.reshape(npoints, *([1]*(len(alphas.shape) - 1)), ndim)
    ...     return np.cos(2*np.pi*r + np.sum(alphas_reshaped * x_reshaped, axis=-1))
    >>> genz = GenzMalikCubature(ndim=3)
    >>> gauss = GaussKronrodQuadrature(npoints=21)
    >>> # Gauss-Kronrod is 1D, so we find the 3D product rule:
    >>> gauss_3d = ProductNestedFixed([gauss, gauss, gauss])
    >>> class CustomRule(Rule):
    ...     def estimate(self, f, a, b, args=()):
    ...         return genz.estimate(f, a, b, args)
    ...     def estimate_error(self, f, a, b, args=()):
    ...         return np.abs(
    ...             genz.estimate(f, a, b, args)
    ...             - gauss_3d.estimate(f, a, b, args)
    ...         )
    >>> rng = np.random.default_rng()
    >>> res = cubature(
    ...     f=f,
    ...     a=np.array([0, 0, 0]),
    ...     b=np.array([1, 1, 1]),
    ...     rule=CustomRule(),
    ...     args=(rng.random((2,)), rng.random((3, 2, 3)))
    ... )
    >>> res.estimate
     array([[-0.95179502,  0.12444608],
            [-0.96247411,  0.60866385],
            [-0.97360014,  0.25515587]])
    """

    def estimate(self, f, a, b, args=()):
        r"""
        Calculate estimate of integral of `f` in rectangular region described by
        corners `a` and ``b``.

        Parameters
        ----------
        f : callable
            Function to integrate. `f` must have the signature::
                f(x : ndarray, \*args) -> ndarray

            `f` should accept arrays ``x`` of shape::
                (npoints, ndim)

            and output arrays of shape::
                (npoints, output_dim_1, ..., output_dim_n)

            In this case, `estimate` will return arrays of shape::
                (output_dim_1, ..., output_dim_n)
        a, b : ndarray
            Lower and upper limits of integration as rank-1 arrays specifying the left
            and right endpoints of the intervals being integrated over. Infinite limits
            are currently not supported.
        args : tuple, optional
            Additional positional args passed to ``f``, if any.

        Returns
        -------
        est : ndarray
            Result of estimation. If `f` returns arrays of shape ``(npoints,
            output_dim_1, ..., output_dim_n)``, then `est` will be of shape
            ``(output_dim_1, ..., output_dim_n)``.
        """
        raise NotImplementedError

    def estimate_error(self, f, a, b, args=()):
        r"""
        Estimate the error of the approximation for the integral of `f` in rectangular
        region described by corners `a` and `b`.

        If a subclass does not override this method, then a default error estimator is
        used. This estimates the error as ``|est - refined_est|`` where ``est`` is
        ``estimate(f, a, b)`` and ``refined_est`` is the sum of
        ``estimate(f, a_k, b_k)`` where ``a_k, b_k`` are the coordinates of each
        subregion of the region described by ``a`` and ``b``. In the 1D case, this
        is equivalent to comparing the integral over an entire interval ``[a, b]`` to
        the sum of the integrals over the left and right subintervals, ``[a, (a+b)/2]``
        and ``[(a+b)/2, b]``.

        Parameters
        ----------
        f : callable
            Function to estimate error for. `f` must have the signature::
                f(x : ndarray, \*args) -> ndarray

            `f` should accept arrays `x` of shape::
                (npoints, ndim)

            and output arrays of shape::
                (npoints, output_dim_1, ..., output_dim_n)

            In this case, `estimate` will return arrays of shape::
                (output_dim_1, ..., output_dim_n)
        a, b : ndarray
            Lower and upper limits of integration as rank-1 arrays specifying the left
            and right endpoints of the intervals being integrated over. Infinite limits
            are currently not supported.
        args : tuple, optional
            Additional positional args passed to `f`, if any.

        Returns
        -------
        err_est : ndarray
            Result of error estimation. If `f` returns arrays of shape
            ``(npoints, output_dim_1, ..., output_dim_n)``, then `est` will be
            of shape ``(output_dim_1, ..., output_dim_n)``.
        """

        est = self.estimate(f, a, b, args)
        refined_est = 0

        for a_k, b_k in _split_subregion(a, b):
            refined_est += self.estimate(f, a_k, b_k, args)

        return self.xp.abs(est - refined_est)


class FixedRule(Rule):
    """
    A rule implemented as the weighted sum of function evaluations at fixed nodes.

    Attributes
    ----------
    nodes_and_weights : (ndarray, ndarray)
        A tuple ``(nodes, weights)`` of nodes at which to evaluate ``f`` and the
        corresponding weights. ``nodes`` should be of shape ``(num_nodes,)`` for 1D
        cubature rules (quadratures) and more generally for N-D cubature rules, it
        should be of shape ``(num_nodes, ndim)``. ``weights`` should be of shape
        ``(num_nodes,)``. The nodes and weights should be for integrals over
        :math:`[-1, 1]^n`.

    See Also
    --------
    GaussLegendreQuadrature, GaussKronrodQuadrature, GenzMalikCubature

    Examples
    --------

    Implementing Simpson's 1/3 rule:

    >>> import numpy as np
    >>> from scipy.integrate._rules import FixedRule
    >>> class SimpsonsQuad(FixedRule):
    ...     @property
    ...     def nodes_and_weights(self):
    ...         nodes = np.array([-1, 0, 1])
    ...         weights = np.array([1/3, 4/3, 1/3])
    ...         return (nodes, weights)
    >>> rule = SimpsonsQuad()
    >>> rule.estimate(
    ...     f=lambda x: x**2,
    ...     a=np.array([0]),
    ...     b=np.array([1]),
    ... )
     [0.3333333]
    """

    def __init__(self):
        self.xp = None

    @property
    def nodes_and_weights(self):
        raise NotImplementedError

    def estimate(self, f, a, b, args=()):
        r"""
        Calculate estimate of integral of `f` in rectangular region described by
        corners `a` and `b` as ``sum(weights * f(nodes))``.

        Nodes and weights will automatically be adjusted from calculating integrals over
        :math:`[-1, 1]^n` to :math:`[a, b]^n`.

        Parameters
        ----------
        f : callable
            Function to integrate. `f` must have the signature::
                f(x : ndarray, \*args) -> ndarray

            `f` should accept arrays `x` of shape::
                (npoints, ndim)

            and output arrays of shape::
                (npoints, output_dim_1, ..., output_dim_n)

            In this case, `estimate` will return arrays of shape::
                (output_dim_1, ..., output_dim_n)
        a, b : ndarray
            Lower and upper limits of integration as rank-1 arrays specifying the left
            and right endpoints of the intervals being integrated over. Infinite limits
            are currently not supported.
        args : tuple, optional
            Additional positional args passed to `f`, if any.

        Returns
        -------
        est : ndarray
            Result of estimation. If `f` returns arrays of shape ``(npoints,
            output_dim_1, ..., output_dim_n)``, then `est` will be of shape
            ``(output_dim_1, ..., output_dim_n)``.
        """
        nodes, weights = self.nodes_and_weights

        if self.xp is None:
            self.xp = array_namespace(nodes)

        return _apply_fixed_rule(f, a, b, nodes, weights, args, self.xp)


class NestedFixedRule(FixedRule):
    r"""
    A cubature rule with error estimate given by the difference between two underlying
    fixed rules.

    If constructed as ``NestedFixedRule(higher, lower)``, this will use::

        estimate(f, a, b) := higher.estimate(f, a, b)
        estimate_error(f, a, b) := \|higher.estimate(f, a, b) - lower.estimate(f, a, b)|

    (where the absolute value is taken elementwise).

    Attributes
    ----------
    higher : Rule
        Higher accuracy rule.

    lower : Rule
        Lower accuracy rule.

    See Also
    --------
    GaussKronrodQuadrature

    Examples
    --------

    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import (
    ...     GaussLegendreQuadrature, NestedFixedRule, ProductNestedFixed
    ... )
    >>> higher = GaussLegendreQuadrature(10)
    >>> lower = GaussLegendreQuadrature(5)
    >>> rule = NestedFixedRule(
    ...     higher,
    ...     lower
    ... )
    >>> rule_2d = ProductNestedFixed([rule, rule])
    """

    def __init__(self, higher, lower):
        self.higher = higher
        self.lower = lower
        self.xp = None

    @property
    def nodes_and_weights(self):
        if self.higher is not None:
            return self.higher.nodes_and_weights
        else:
            raise NotImplementedError

    @property
    def lower_nodes_and_weights(self):
        if self.lower is not None:
            return self.lower.nodes_and_weights
        else:
            raise NotImplementedError

    def estimate_error(self, f, a, b, args=()):
        r"""
        Estimate the error of the approximation for the integral of `f` in rectangular
        region described by corners `a` and `b`.

        Parameters
        ----------
        f : callable
            Function to estimate error for. `f` must have the signature::
                f(x : ndarray, \*args) -> ndarray

            `f` should accept arrays `x` of shape::
                (npoints, ndim)

            and output arrays of shape::
                (npoints, output_dim_1, ..., output_dim_n)

            In this case, `estimate` will return arrays of shape::
                (output_dim_1, ..., output_dim_n)
        a, b : ndarray
            Lower and upper limits of integration as rank-1 arrays specifying the left
            and right endpoints of the intervals being integrated over. Infinite limits
            are currently not supported.
        args : tuple, optional
            Additional positional args passed to `f`, if any.

        Returns
        -------
        err_est : ndarray
            Result of error estimation. If `f` returns arrays of shape
            ``(npoints, output_dim_1, ..., output_dim_n)``, then `est` will be
            of shape ``(output_dim_1, ..., output_dim_n)``.
        """

        nodes, weights = self.nodes_and_weights
        lower_nodes, lower_weights = self.lower_nodes_and_weights

        if self.xp is None:
            self.xp = array_namespace(nodes)

        error_nodes = self.xp.concat([nodes, lower_nodes], axis=0)
        error_weights = self.xp.concat([weights, -lower_weights], axis=0)

        return self.xp.abs(
            _apply_fixed_rule(f, a, b, error_nodes, error_weights, args, self.xp)
        )


class ProductNestedFixed(NestedFixedRule):
    """
    Find the n-dimensional cubature rule constructed from the Cartesian product of 1-D
    `NestedFixedRule` quadrature rules.

    Given a list of N 1-dimensional quadrature rules which support error estimation
    using NestedFixedRule, this will find the N-dimensional cubature rule obtained by
    taking the Cartesian product of their nodes, and estimating the error by taking the
    difference with a lower-accuracy N-dimensional cubature rule obtained using the
    ``.lower_nodes_and_weights`` rule in each of the base 1-dimensional rules.

    Parameters
    ----------
    base_rules : list of NestedFixedRule
        List of base 1-dimensional `NestedFixedRule` quadrature rules.

    Attributes
    ----------
    base_rules : list of NestedFixedRule
        List of base 1-dimensional `NestedFixedRule` qudarature rules.

    Examples
    --------

    Evaluate a 2D integral by taking the product of two 1D rules:

    >>> import numpy as np
    >>> from scipy.integrate import cubature
    >>> from scipy.integrate._rules import (
    ...  ProductNestedFixed, GaussKronrodQuadrature
    ... )
    >>> def f(x):
    ...     # f(x) = cos(x_1) + cos(x_2)
    ...     return np.sum(np.cos(x), axis=-1)
    >>> rule = ProductNestedFixed(
    ...     [GaussKronrodQuadrature(15), GaussKronrodQuadrature(15)]
    ... ) # Use 15-point Gauss-Kronrod, which implements NestedFixedRule
    >>> a, b = np.array([0, 0]), np.array([1, 1])
    >>> rule.estimate(f, a, b) # True value 2*sin(1), approximately 1.6829
     np.float64(1.682941969615793)
    >>> rule.estimate_error(f, a, b)
     np.float64(2.220446049250313e-16)
    """

    def __init__(self, base_rules):
        for rule in base_rules:
            if not isinstance(rule, NestedFixedRule):
                raise ValueError("base rules for product need to be instance of"
                                 "NestedFixedRule")

        self.base_rules = base_rules
        self.xp = None

    @cached_property
    def nodes_and_weights(self):
        nodes = _cartesian_product(
            [rule.nodes_and_weights[0] for rule in self.base_rules]
        )

        if self.xp is None:
            self.xp = array_namespace(nodes)

        weights = self.xp.prod(
            _cartesian_product(
                [rule.nodes_and_weights[1] for rule in self.base_rules]
            ),
            axis=-1,
        )

        return nodes, weights

    @cached_property
    def lower_nodes_and_weights(self):
        nodes = _cartesian_product(
            [cubature.lower_nodes_and_weights[0] for cubature in self.base_rules]
        )

        if self.xp is None:
            self.xp = array_namespace(nodes)

        weights = self.xp.prod(
            _cartesian_product(
                [cubature.lower_nodes_and_weights[1] for cubature in self.base_rules]
            ),
            axis=-1,
        )

        return nodes, weights


def _cartesian_product(arrays):
    xp = array_namespace(*arrays)

    arrays_ix = xp.meshgrid(*arrays, indexing='ij')
    result = xp.reshape(xp.stack(arrays_ix, axis=-1), (-1, len(arrays)))

    return result


def _split_subregion(a, b, xp, split_at=None):
    """
    Given the coordinates of a region like a=[0, 0] and b=[1, 1], yield the coordinates
    of all subregions, which in this case would be::

        ([0, 0], [1/2, 1/2]),
        ([0, 1/2], [1/2, 1]),
        ([1/2, 0], [1, 1/2]),
        ([1/2, 1/2], [1, 1])
    """
    xp = array_namespace(a, b)

    if split_at is None:
        split_at = (a + b) / 2

    left = [xp.asarray([a[i], split_at[i]]) for i in range(a.shape[0])]
    right = [xp.asarray([split_at[i], b[i]]) for i in range(b.shape[0])]

    a_sub = _cartesian_product(left)
    b_sub = _cartesian_product(right)

    for i in range(a_sub.shape[0]):
        yield a_sub[i, ...], b_sub[i, ...]


def _apply_fixed_rule(f, a, b, orig_nodes, orig_weights, args, xp):
    # Downcast nodes and weights to common dtype of a and b
    result_dtype = a.dtype
    orig_nodes = xp.astype(orig_nodes, result_dtype)
    orig_weights = xp.astype(orig_weights, result_dtype)

    # Ensure orig_nodes are at least 2D, since 1D cubature methods can return arrays of
    # shape (npoints,) rather than (npoints, 1)
    if orig_nodes.ndim == 1:
        orig_nodes = orig_nodes[:, None]

    rule_ndim = orig_nodes.shape[-1]

    a_ndim = xp_size(a)
    b_ndim = xp_size(b)

    if rule_ndim != a_ndim or rule_ndim != b_ndim:
        raise ValueError(f"rule and function are of incompatible dimension, nodes have"
                         f"ndim {rule_ndim}, while limit of integration has ndim"
                         f"a_ndim={a_ndim}, b_ndim={b_ndim}")

    lengths = b - a

    # The underlying rule is for the hypercube [-1, 1]^n.
    #
    # To handle arbitrary regions of integration, it's necessary to apply a linear
    # change of coordinates to map each interval [a[i], b[i]] to [-1, 1].
    nodes = (orig_nodes + 1) * (lengths * 0.5) + a

    # Also need to multiply the weights by a scale factor equal to the determinant
    # of the Jacobian for this coordinate change.
    weight_scale_factor = xp.prod(lengths, dtype=result_dtype) / 2**rule_ndim
    weights = orig_weights * weight_scale_factor

    f_nodes = f(nodes, *args)
    weights_reshaped = xp.reshape(weights, (-1, *([1] * (f_nodes.ndim - 1))))

    # f(nodes) will have shape (num_nodes, output_dim_1, ..., output_dim_n)
    # Summing along the first axis means estimate will shape (output_dim_1, ...,
    # output_dim_n)
    est = xp.sum(weights_reshaped * f_nodes, axis=0, dtype=result_dtype)

    return est
<|endoftext|>import inspect
import numpy as np
from .bdf import BDF
from .radau import Radau
from .rk import RK23, RK45, DOP853
from .lsoda import LSODA
from scipy.optimize import OptimizeResult
from .common import EPS, OdeSolution
from .base import OdeSolver


METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}


MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class OdeResult(OptimizeResult):
    pass


def prepare_events(events):
    """Standardize event functions and extract attributes."""
    if callable(events):
        events = (events,)

    max_events = np.empty(len(events))
    direction = np.empty(len(events))
    for i, event in enumerate(events):
        terminal = getattr(event, 'terminal', None)
        direction[i] = getattr(event, 'direction', 0)

        message = ('The `terminal` attribute of each event '
                   'must be a boolean or positive integer.')
        if terminal is None or terminal == 0:
            max_events[i] = np.inf
        elif int(terminal) == terminal and terminal > 0:
            max_events[i] = terminal
        else:
            raise ValueError(message)

    return events, max_events, direction


def solve_event_equation(event, sol, t_old, t):
    """Solve an equation corresponding to an ODE event.

    The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an
    ODE solver using some sort of interpolation. It is solved by
    `scipy.optimize.brentq` with xtol=atol=4*EPS.

    Parameters
    ----------
    event : callable
        Function ``event(t, y)``.
    sol : callable
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    t_old, t : float
        Previous and new values of time. They will be used as a bracketing
        interval.

    Returns
    -------
    root : float
        Found solution.
    """
    from scipy.optimize import brentq
    return brentq(lambda t: event(t, sol(t)), t_old, t,
                  xtol=4 * EPS, rtol=4 * EPS)


def handle_events(sol, events, active_events, event_count, max_events,
                  t_old, t):
    """Helper function to handle events.

    Parameters
    ----------
    sol : DenseOutput
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    events : list of callables, length n_events
        Event functions with signatures ``event(t, y)``.
    active_events : ndarray
        Indices of events which occurred.
    event_count : ndarray
        Current number of occurrences for each event.
    max_events : ndarray, shape (n_events,)
        Number of occurrences allowed for each event before integration
        termination is issued.
    t_old, t : float
        Previous and new values of time.

    Returns
    -------
    root_indices : ndarray
        Indices of events which take zero between `t_old` and `t` and before
        a possible termination.
    roots : ndarray
        Values of t at which events occurred.
    terminate : bool
        Whether a terminal event occurred.
    """
    roots = [solve_event_equation(events[event_index], sol, t_old, t)
             for event_index in active_events]

    roots = np.asarray(roots)

    if np.any(event_count[active_events] >= max_events[active_events]):
        if t > t_old:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = np.nonzero(event_count[active_events]
                       >= max_events[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate


def find_active_events(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    g, g_new = np.asarray(g), np.asarray(g_new)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))

    return np.nonzero(mask)[0]


def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    """Solve an initial value problem for a system of ODEs.

    This function numerically integrates a system of ordinary differential
    equations given an initial value::

        dy / dt = f(t, y)
        y(t0) = y0

    Here t is a 1-D independent variable (time), y(t) is an
    N-D vector-valued function (state), and an N-D
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.

    Some of the solvers support integration in the complex domain, but note
    that for stiff ODE solvers, the right-hand side must be
    complex-differentiable (satisfy Cauchy-Riemann equations [11]_).
    To solve a problem in the complex domain, pass y0 with a complex data type.
    Another option always available is to rewrite your problem for real and
    imaginary parts separately.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. Additional
        arguments need to be passed if ``args`` is used (see documentation of
        ``args`` argument). ``fun`` must return an array of the same shape as
        ``y``. See `vectorized` for more information.
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    method : string or `OdeSolver`, optional
        Integration method to use:

            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [2]_. Can be applied in
              the complex domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
              Python implementation of the "DOP853" algorithm originally
              written in Fortran [14]_. A 7-th order interpolation polynomial
              accurate to 7-th order is used for the dense output.
              Can be applied in the complex domain.
            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
              order 5 [4]_. The error is controlled with a third-order accurate
              embedded formula. A cubic polynomial which satisfies the
              collocation conditions is used for the dense output.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the
              complex domain.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
              switching [7]_, [8]_. This is a wrapper of the Fortran solver
              from ODEPACK.

        Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
        for non-stiff problems and implicit methods ('Radau', 'BDF') for
        stiff problems [9]_. Among Runge-Kutta methods, 'DOP853' is recommended
        for solving with high precision (low values of `rtol` and `atol`).

        If not sure, first try to run 'RK45'. If it makes unusually many
        iterations, diverges, or fails, your problem is likely to be stiff and
        you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
        choice, but it might be somewhat less convenient to work with as it
        wraps old Fortran code.

        You can also pass an arbitrary class derived from `OdeSolver` which
        implements the solver.
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool, optional
        Whether to compute a continuous solution. Default is False.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` where
        additional argument have to be passed if ``args`` is used (see
        documentation of ``args`` argument). Each function must return a
        float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default,
        all zeros will be found. The solver looks for a sign change over
        each step, so if multiple zero crossings occur within one step,
        events may be missed. Additionally each `event` function might
        have the following attributes:

            terminal: bool or int, optional
                When boolean, whether to terminate integration if this event occurs.
                When integral, termination occurs after the specified the number of
                occurrences of this event.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods and for 'Radau' and
        'BDF' in some circumstances (e.g. small ``len(y0)``).
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    **options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : array_like, sparse_matrix, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect
        to y, required by the 'Radau', 'BDF' and 'LSODA' method. The
        Jacobian matrix has shape (n, n) and its element (i, j) is equal to
        ``d f_i / d y_j``.  There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Not supported by 'LSODA'.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)``, as necessary.
              Additional arguments have to be passed if ``args`` is
              used (see documentation of ``args`` argument).
              For 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : array_like, sparse matrix or None, optional
        Defines a sparsity structure of the Jacobian matrix for a finite-
        difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few
        non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations [10]_. A zero entry means that
        a corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
        Not supported by 'LSODA', see `lband` and `uband` instead.
    lband, uband : int or None, optional
        Parameters defining the bandwidth of the Jacobian for the 'LSODA'
        method, i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``.
        Default is None. Setting these requires your jac routine to return the
        Jacobian in the packed format: the returned array must have ``n``
        columns and ``uband + lband + 1`` rows in which Jacobian diagonals are
        written. Specifically ``jac_packed[uband + i - j , j] = jac[i, j]``.
        The same format is used in `scipy.linalg.solve_banded` (check for an
        illustration).  These parameters can be also used with ``jac=None`` to
        reduce the number of Jacobian elements estimated by finite differences.
    min_step : float, optional
        The minimum allowed step size for 'LSODA' method.
        By default `min_step` is zero.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.

    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [11] `Cauchy-Riemann equations
             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
             Wikipedia.
    .. [12] `Lotka-Volterra equations
            <https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations>`_
            on Wikipedia.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

    Examples
    --------
    Basic exponential decay showing automatically chosen time points.

    >>> import numpy as np
    >>> from scipy.integrate import solve_ivp
    >>> def exponential_decay(t, y): return -0.5 * y
    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
    >>> print(sol.t)
    [ 0.          0.11487653  1.26364188  3.06061781  4.81611105  6.57445806
      8.33328988 10.        ]
    >>> print(sol.y)
    [[2.         1.88836035 1.06327177 0.43319312 0.18017253 0.07483045
      0.03107158 0.01350781]
     [4.         3.7767207  2.12654355 0.86638624 0.36034507 0.14966091
      0.06214316 0.02701561]
     [8.         7.5534414  4.25308709 1.73277247 0.72069014 0.29932181
      0.12428631 0.05403123]]

    Specifying points where the solution is desired.

    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
    ...                 t_eval=[0, 1, 2, 4, 10])
    >>> print(sol.t)
    [ 0  1  2  4 10]
    >>> print(sol.y)
    [[2.         1.21305369 0.73534021 0.27066736 0.01350938]
     [4.         2.42610739 1.47068043 0.54133472 0.02701876]
     [8.         4.85221478 2.94136085 1.08266944 0.05403753]]

    Cannon fired upward with terminal event upon impact. The ``terminal`` and
    ``direction`` fields of an event are applied by monkey patching a function.
    Here ``y[0]`` is position and ``y[1]`` is velocity. The projectile starts
    at position 0 with velocity +10. Note that the integration never reaches
    t=100 because the event is terminal.

    >>> def upward_cannon(t, y): return [y[1], -0.5]
    >>> def hit_ground(t, y): return y[0]
    >>> hit_ground.terminal = True
    >>> hit_ground.direction = -1
    >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)
    >>> print(sol.t_events)
    [array([40.])]
    >>> print(sol.t)
    [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
     1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]

    Use `dense_output` and `events` to find position, which is 100, at the apex
    of the cannonball's trajectory. Apex is not defined as terminal, so both
    apex and hit_ground are found. There is no information at t=20, so the sol
    attribute is used to evaluate the solution. The sol attribute is returned
    by setting ``dense_output=True``. Alternatively, the `y_events` attribute
    can be used to access the solution at the time of the event.

    >>> def apex(t, y): return y[1]
    >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10],
    ...                 events=(hit_ground, apex), dense_output=True)
    >>> print(sol.t_events)
    [array([40.]), array([20.])]
    >>> print(sol.t)
    [0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
     1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]
    >>> print(sol.sol(sol.t_events[1][0]))
    [100.   0.]
    >>> print(sol.y_events)
    [array([[-5.68434189e-14, -1.00000000e+01]]),
     array([[1.00000000e+02, 1.77635684e-15]])]

    As an example of a system with additional parameters, we'll implement
    the Lotka-Volterra equations [12]_.

    >>> def lotkavolterra(t, z, a, b, c, d):
    ...     x, y = z
    ...     return [a*x - b*x*y, -c*y + d*x*y]
    ...

    We pass in the parameter values a=1.5, b=1, c=3 and d=1 with the `args`
    argument.

    >>> sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
    ...                 dense_output=True)

    Compute a dense solution and plot it.

    >>> t = np.linspace(0, 15, 300)
    >>> z = sol.sol(t)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, z.T)
    >>> plt.xlabel('t')
    >>> plt.legend(['x', 'y'], shadow=True)
    >>> plt.title('Lotka-Volterra System')
    >>> plt.show()

    A couple examples of using solve_ivp to solve the differential
    equation ``y' = Ay`` with complex matrix ``A``.

    >>> A = np.array([[-0.25 + 0.14j, 0, 0.33 + 0.44j],
    ...               [0.25 + 0.58j, -0.2 + 0.14j, 0],
    ...               [0, 0.2 + 0.4j, -0.1 + 0.97j]])

    Solving an IVP with ``A`` from above and ``y`` as 3x1 vector:

    >>> def deriv_vec(t, y):
    ...     return A @ y
    >>> result = solve_ivp(deriv_vec, [0, 25],
    ...                    np.array([10 + 0j, 20 + 0j, 30 + 0j]),
    ...                    t_eval=np.linspace(0, 25, 101))
    >>> print(result.y[:, 0])
    [10.+0.j 20.+0.j 30.+0.j]
    >>> print(result.y[:, -1])
    [18.46291039+45.25653651j 10.01569306+36.23293216j
     -4.98662741+80.07360388j]

    Solving an IVP with ``A`` from above with ``y`` as 3x3 matrix :

    >>> def deriv_mat(t, y):
    ...     return (A @ y.reshape(3, 3)).flatten()
    >>> y0 = np.array([[2 + 0j, 3 + 0j, 4 + 0j],
    ...                [5 + 0j, 6 + 0j, 7 + 0j],
    ...                [9 + 0j, 34 + 0j, 78 + 0j]])

    >>> result = solve_ivp(deriv_mat, [0, 25], y0.flatten(),
    ...                    t_eval=np.linspace(0, 25, 101))
    >>> print(result.y[:, 0].reshape(3, 3))
    [[ 2.+0.j  3.+0.j  4.+0.j]
     [ 5.+0.j  6.+0.j  7.+0.j]
     [ 9.+0.j 34.+0.j 78.+0.j]]
    >>> print(result.y[:, -1].reshape(3, 3))
    [[  5.67451179 +12.07938445j  17.2888073  +31.03278837j
        37.83405768 +63.25138759j]
     [  3.39949503 +11.82123994j  21.32530996 +44.88668871j
        53.17531184+103.80400411j]
     [ -2.26105874 +22.19277664j -15.1255713  +70.19616341j
       -38.34616845+153.29039931j]]


    """
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError(f"`method` must be one of {METHODS} or OdeSolver class.")

    t0, tf = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        def fun(t, x, fun=fun):
            return fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []

    if events is not None:
        events, max_events, event_dir = prepare_events(events)
        event_count = np.zeros(len(events))
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                event_count[active_events] += 1
                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, event_count, max_events,
                    t_old, t)

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            donot_append = (len(ts) > 1 and
                            ts[-1] == t and
                            dense_output)
            if not donot_append:
                ts.append(t)
                ys.append(y)
            else:
                if len(interpolants) > 0:
                    interpolants.pop()
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(
                ts, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
        else:
            sol = OdeSolution(
                ti, interpolants, alt_segment=True if method in [BDF, LSODA] else False
            )
    else:
        sol = None

    return OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)
<|endoftext|>import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, num_jac, EPS, warn_extraneous,
                     validate_first_step)
from .base import OdeSolver, DenseOutput

S6 = 6 ** 0.5

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))

# These are transformation matrices.
T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
# These linear combinations are used in the algorithm.
TI_REAL = TI[0]
TI_COMPLEX = TI[1] + 1j * TI[2]

# Interpolator coefficients.
P = np.array([
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu):
    """Solve the collocation system.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    h : float
        Step to try.
    Z0 : ndarray, shape (3, n)
        Initial guess for the solution. It determines new values of `y` at
        ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
    scale : ndarray, shape (n)
        Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
    tol : float
        Tolerance to which solve the system. This value is compared with
        the normalized by `scale` error.
    LU_real, LU_complex
        LU decompositions of the system Jacobians.
    solve_lu : callable
        Callable which solves a linear system given a LU decomposition. The
        signature is ``solve_lu(LU, b)``.

    Returns
    -------
    converged : bool
        Whether iterations converged.
    n_iter : int
        Number of completed iterations.
    Z : ndarray, shape (3, n)
        Found solution.
    rate : float
        The rate of convergence.
    """
    n = y.shape[0]
    M_real = MU_REAL / h
    M_complex = MU_COMPLEX / h

    W = TI.dot(Z0)
    Z = Z0

    F = np.empty((3, n))
    ch = h * C

    dW_norm_old = None
    dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(3):
            F[i] = fun(t + ch[i], y + Z[i])

        if not np.all(np.isfinite(F)):
            break

        f_real = F.T.dot(TI_REAL) - M_real * W[0]
        f_complex = F.T.dot(TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])

        dW_real = solve_lu(LU_real, f_real)
        dW_complex = solve_lu(LU_complex, f_complex)

        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        dW_norm = norm(dW / scale)
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
            break

        W += dW
        Z = T.dot(W)

        if (dW_norm == 0 or
                rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        dW_norm_old = dW_norm

    return converged, k + 1, Z, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25

    return factor


class Radau(OdeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [1]_. The error is controlled with a
    third-order accurate embedded formula. A cubic polynomial which satisfies
    the collocation conditions is used for the dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    """
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        # Select initial step assuming the same order which is used to control
        # the error.
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, t_bound, max_step, self.f, self.direction,
                3, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format='csc')
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, f):
                self.njev += 1
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0, self.f)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev = 1
            if issparse(J):
                J = csc_matrix(J)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=float)

            else:
                J = np.asarray(J, dtype=float)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError(f"`jac` is expected to have shape {(self.n, self.n)},"
                                 f" but actually has {J.shape}.")
        else:
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                J = np.asarray(jac, dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError(f"`jac` is expected to have shape {(self.n, self.n)},"
                                 f" but actually has {J.shape}.")
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.f

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * C).T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(MU_REAL / h * self.I - J)
                    LU_complex = self.lu(MU_COMPLEX / h * self.I - J)

                converged, n_iter, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU_real, LU_complex, self.solve_lu)

                if not converged:
                    if current_jac:
                        break

                    J = self.jac(t, y, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = y + Z[-1]
            ZE = Z.T.dot(E) / h
            error = self.solve_lu(LU_real, f + ZE)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = norm(error / scale)
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            if rejected and error_norm > 1:
                error = self.solve_lu(LU_real, self.fun(t, y + error) + ZE)
                error_norm = norm(error / scale)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old,
                                        error_norm, error_norm_old)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message

    def _compute_dense_output(self):
        Q = np.dot(self.Z.T, P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake.
        y = np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y
<|endoftext|>from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix


EPS = np.finfo(float).eps


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn("The following arguments have no effect for a chosen solver: "
             f"{', '.join(f'`{x}`' for x in extraneous)}.",
             stacklevel=3)


def validate_tol(rtol, atol, n):
    """Validate tolerance values."""

    if np.any(rtol < 100 * EPS):
        warn("At least one element of `rtol` is too small. "
             f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.",
             stacklevel=3)
        rtol = np.maximum(rtol, 100 * EPS)

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5


def select_initial_step(fun, t0, y0, t_bound,
                        max_step, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    t_bound : float
        End-point of integration interval; used to ensure that t0+step<=tbound
        and that fun is only evaluated in the interval [t0,tbound]
    max_step : float
        Maximum allowable step size.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return np.inf

    interval_length = abs(t_bound - t0)
    if interval_length == 0.0:
        return 0.0

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    # Check t0+h0*direction doesn't take us beyond t_bound
    h0 = min(h0, interval_length)
    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1, interval_length, max_step)


class OdeSolution:
    """Continuous ODE solution.

    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.

    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below). Evaluation outside this interval is not forbidden, but
    the accuracy is not guaranteed.

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of DenseOutput with n_segments elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    alt_segment : boolean
        Requests the alternative interpolant segment selection scheme. At each
        solver integration point, two interpolant segments are available. The
        default (False) and alternative (True) behaviours select the segment
        for which the requested time corresponded to ``t`` and ``t_old``,
        respectively. This functionality is only relevant for testing the
        interpolants' accuracy: different integrators use different
        construction strategies.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, ts, interpolants, alt_segment=False):
        ts = np.asarray(ts)
        d = np.diff(ts)
        # The first case covers integration on zero segment.
        if not ((ts.size == 2 and ts[0] == ts[-1])
                or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.side = "right" if alt_segment else "left"
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.side = "left" if alt_segment else "right"
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # if alt_segment=False, then we prioritize a segment with a lower
        # index.
        ind = np.searchsorted(self.ts_sorted, t, side=self.side)

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-D array.
        """
        t = np.asarray(t)

        if t.ndim == 0:
            return self._call_single(t)

        order = np.argsort(t)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.shape[0])
        t_sorted = t[order]

        # See comment in self._call_single.
        segments = np.searchsorted(self.ts_sorted, t_sorted, side=self.side)
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            group_start = group_end

        ys = np.hstack(ys)
        ys = ys[:, reverse]

        return ys


NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1


def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
    """Finite differences Jacobian approximation tailored for ODE solvers.

    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e., the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * np.maximum(np.abs(y), threshold)``. Typically, the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return np.empty((0, 0)), factor

    if factor is None:
        factor = np.full(n, EPS ** 0.5)
    else:
        factor = factor.copy()

    # Direct the step as ODE dictates, hoping that such a step won't lead to
    # a problematic region. For complex ODEs it makes sense to use the real
    # part of f as we use steps along real axis.
    f_sign = 2 * (np.real(f) >= 0).astype(float) - 1
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y

    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in np.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

    if sparsity is None:
        return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
    else:
        structure, groups = sparsity
        return _sparse_num_jac(fun, t, y, f, h, factor, y_scale,
                               structure, groups)


def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    n = y.shape[0]
    h_vecs = np.diag(h)
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    max_ind = np.argmax(np.abs(diff), axis=0)
    r = np.arange(n)
    max_diff = np.abs(diff[max_ind, r])
    scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_vecs[ind, ind] = h_new
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        diff_new = f_new - f[:, None]
        max_ind = np.argmax(np.abs(diff_new), axis=0)
        r = np.arange(ind.shape[0])
        max_diff_new = np.abs(diff_new[max_ind, r])
        scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff /= h

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor


def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
    n = y.shape[0]
    n_groups = np.max(groups) + 1
    h_vecs = np.empty((n_groups, n))
    for group in range(n_groups):
        e = np.equal(group, groups)
        h_vecs[group] = h * e
    h_vecs = h_vecs.T

    f_new = fun(t, y[:, None] + h_vecs)
    df = f_new - f[:, None]

    i, j, _ = find(structure)
    diff = coo_matrix((df[i, groups[j]], (i, j)), shape=(n, n)).tocsc()
    max_ind = np.array(abs(diff).argmax(axis=0)).ravel()
    r = np.arange(n)
    max_diff = np.asarray(np.abs(diff[max_ind, r])).ravel()
    scale = np.maximum(np.abs(f[max_ind]),
                       np.abs(f_new[max_ind, groups[r]]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_new_all = np.zeros(n)
        h_new_all[ind] = h_new

        groups_unique = np.unique(groups[ind])
        groups_map = np.empty(n_groups, dtype=int)
        h_vecs = np.empty((groups_unique.shape[0], n))
        for k, group in enumerate(groups_unique):
            e = np.equal(group, groups)
            h_vecs[k] = h_new_all * e
            groups_map[group] = k
        h_vecs = h_vecs.T

        f_new = fun(t, y[:, None] + h_vecs)
        df = f_new - f[:, None]
        i, j, _ = find(structure[:, ind])
        diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]],
                               (i, j)), shape=(n, ind.shape[0])).tocsc()

        max_ind_new = np.array(abs(diff_new).argmax(axis=0)).ravel()
        r = np.arange(ind.shape[0])
        max_diff_new = np.asarray(np.abs(diff_new[max_ind_new, r])).ravel()
        scale_new = np.maximum(
            np.abs(f[max_ind_new]),
            np.abs(f_new[max_ind_new, groups_map[groups[ind]]]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff.data /= np.repeat(h, np.diff(diff.indptr))

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor
<|endoftext|>import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from .base import OdeSolver, DenseOutput


MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def compute_R(order, factor):
    """Compute the matrix for changing the differences array."""
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return np.cumprod(M, axis=0)


def change_D(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R(order, factor)
    U = compute_R(order, 1)
    RU = R.dot(U)
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y)
        if not np.all(np.isfinite(f)):
            break

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        y += dy
        d += dy

        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, d


class BDF(OdeSolver):
    """Implicit method based on backward-differentiation formulas.

    This is a variable order method with the order varying automatically from
    1 to 5. The general framework of the BDF algorithm is described in [1]_.
    This class implements a quasi-constant step size as explained in [2]_.
    The error estimation strategy for the constant-step BDF is derived in [3]_.
    An accuracy enhancement using modified formulas (NDF) [2]_ is also implemented.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to y,
        required by this method. The Jacobian matrix has shape (n, n) and its
        element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [4]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
           Solution of Ordinary Differential Equations", ACM Transactions on
           Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
    .. [2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
           Nonstiff Problems", Sec. III.2.
    .. [4] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    """
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(self.fun, self.t, self.y, 
                                             t_bound, max_step, f,
                                             self.direction, 1,
                                             self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format='csc', dtype=self.y.dtype)
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n, dtype=self.y.dtype)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        kappa = np.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
        self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
        self.alpha = (1 - kappa) * self.gamma
        self.error_const = kappa * self.gamma + 1 / np.arange(1, MAX_ORDER + 2)

        D = np.empty((MAX_ORDER + 3, self.n), dtype=self.y.dtype)
        D[0] = self.y
        D[1] = f * self.h_abs * self.direction
        self.D = D

        self.order = 1
        self.n_equal_steps = 0
        self.LU = None

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y):
                self.njev += 1
                f = self.fun_single(t, y)
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev += 1
            if issparse(J):
                J = csc_matrix(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=y0.dtype)
            else:
                J = np.asarray(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=y0.dtype)

            if J.shape != (self.n, self.n):
                raise ValueError(f"`jac` is expected to have shape {(self.n, self.n)},"
                                 f" but actually has {J.shape}.")
        else:
            if issparse(jac):
                J = csc_matrix(jac, dtype=y0.dtype)
            else:
                J = np.asarray(jac, dtype=y0.dtype)

            if J.shape != (self.n, self.n):
                raise ValueError(f"`jac` is expected to have shape {(self.n, self.n)},"
                                 f" but actually has {J.shape}.")
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[:order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]

            converged = False
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = self.lu(self.I - c * J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
                    scale, self.newton_tol)

                if not converged:
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

            if not converged:
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)

            if error_norm > 1:
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
            else:
                step_accepted = True

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None

    def _dense_output_impl(self):
        return BdfDenseOutput(self.t_old, self.t, self.h_abs * self.direction,
                              self.order, self.D[:self.order + 1].copy())


class BdfDenseOutput(DenseOutput):
    def __init__(self, t_old, t, h, order, D):
        super().__init__(t_old, t)
        self.order = order
        self.t_shift = self.t - h * np.arange(self.order)
        self.denom = h * (1 + np.arange(self.order))
        self.D = D

    def _call_impl(self, t):
        if t.ndim == 0:
            x = (t - self.t_shift) / self.denom
            p = np.cumprod(x)
        else:
            x = (t - self.t_shift[:, None]) / self.denom[:, None]
            p = np.cumprod(x, axis=0)

        y = np.dot(self.D[1:].T, p)
        if y.ndim == 1:
            y += self.D[0]
        else:
            y += self.D[0, :, None]

        return y
<|endoftext|>"""Suite of ODE solvers implemented in Python."""
from .ivp import solve_ivp
from .rk import RK23, RK45, DOP853
from .radau import Radau
from .bdf import BDF
from .lsoda import LSODA
from .common import OdeSolution
from .base import DenseOutput, OdeSolver
<|endoftext|>import numpy as np
from .base import OdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)
from . import dop853_coefficients

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


def rk_step(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new

    return y_new, f_new


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, t_bound, max_step, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class RK23(RungeKutta):
    """Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` may be called in a vectorized fashion. False (default)
        is recommended for this solver.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for this solver.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
        Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1/2, 3/4])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = np.array([2/9, 1/3, 4/9])
    E = np.array([5/72, -1/12, -1/9, 1/8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])


class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
        Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


class DOP853(RungeKutta):
    """Explicit Runge-Kutta method of order 8.

    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [1]_, [2]_. Note that this is not a literal translation, but
    the algorithmic core and coefficients are the same.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here, ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver
        as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """
    n_stages = dop853_coefficients.N_STAGES
    order = 8
    error_estimator_order = 7
    A = dop853_coefficients.A[:n_stages, :n_stages]
    B = dop853_coefficients.B
    C = dop853_coefficients.C[:n_stages]
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5
    D = dop853_coefficients.D

    A_EXTRA = dop853_coefficients.A[n_stages + 1:]
    C_EXTRA = dop853_coefficients.C[n_stages + 1:]

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol,
                         vectorized, first_step, **extraneous)
        self.K_extended = np.empty((dop853_coefficients.N_STAGES_EXTENDED,
                                    self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages + 1]

    def _estimate_error(self, K, h):  # Left for testing purposes.
        err5 = np.dot(K.T, self.E5)
        err3 = np.dot(K.T, self.E3)
        denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        correction_factor = np.ones_like(err5)
        mask = denom > 0
        correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        return h * err5 * correction_factor

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale
        err5_norm_2 = np.linalg.norm(err5)**2
        err3_norm_2 = np.linalg.norm(err3)**2
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))

    def _dense_output_impl(self):
        K = self.K_extended
        h = self.h_previous
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA),
                                   start=self.n_stages + 1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n),
                     dtype=self.y_old.dtype)

        f_old = K[0]
        delta_y = self.y - self.y_old

        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)


class RkDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        y = self.h * np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y


class Dop853DenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, F):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.F = F
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h

        if t.ndim == 0:
            y = np.zeros_like(self.y_old)
        else:
            x = x[:, None]
            y = np.zeros((len(x), len(self.y_old)), dtype=self.y_old.dtype)

        for i, f in enumerate(reversed(self.F)):
            y += f
            if i % 2 == 0:
                y *= x
            else:
                y *= 1 - x
        y += self.y_old

        return y.T
<|endoftext|>import numpy as np


def check_arguments(fun, y0, support_complex):
    """Helper function for checking arguments common to all solvers."""
    y0 = np.asarray(y0)
    if np.issubdtype(y0.dtype, np.complexfloating):
        if not support_complex:
            raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")
        dtype = complex
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    if not np.isfinite(y0).all():
        raise ValueError("All components of the initial state `y0` must be finite.")

    def fun_wrapped(t, y):
        return np.asarray(fun(t, y), dtype=dtype)

    return fun_wrapped, y0


class OdeSolver:
    """Base class for ODE solvers.

    In order to implement a new solver you need to follow the guidelines:

        1. A constructor must accept parameters presented in the base class
           (listed below) along with any other parameters specific to a solver.
        2. A constructor must accept arbitrary extraneous arguments
           ``**extraneous``, but warn that these arguments are irrelevant
           using `common.warn_extraneous` function. Do not pass these
           arguments to the base class.
        3. A solver must implement a private method `_step_impl(self)` which
           propagates a solver one step further. It must return tuple
           ``(success, message)``, where ``success`` is a boolean indicating
           whether a step was successful, and ``message`` is a string
           containing description of a failure if a step failed or None
           otherwise.
        4. A solver must implement a private method `_dense_output_impl(self)`,
           which returns a `DenseOutput` object covering the last successful
           step.
        5. A solver must have attributes listed below in Attributes section.
           Note that ``t_old`` and ``step_size`` are updated automatically.
        6. Use `fun(self, t, y)` method for the system rhs evaluation, this
           way the number of function evaluations (`nfev`) will be tracked
           automatically.
        7. For convenience, a base class provides `fun_single(self, t, y)` and
           `fun_vectorized(self, t, y)` for evaluating the rhs in
           non-vectorized and vectorized fashions respectively (regardless of
           how `fun` from the constructor is implemented). These calls don't
           increment `nfev`.
        8. If a solver uses a Jacobian matrix and LU decompositions, it should
           track the number of Jacobian evaluations (`njev`) and the number of
           LU decompositions (`nlu`).
        9. By convention, the function evaluations used to compute a finite
           difference approximation of the Jacobian should not be counted in
           `nfev`, thus use `fun_single(self, t, y)` or
           `fun_vectorized(self, t, y)` when computing a finite difference
           approximation of the Jacobian.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.
    vectorized : bool
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods. It can also
        result in slower overall execution for 'Radau' and 'BDF' in some
        circumstances (e.g. small ``len(y0)``).
    support_complex : bool, optional
        Whether integration in a complex domain should be supported.
        Generally determined by a derived solver class capabilities.
        Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, t_bound, vectorized,
                 support_complex=False):
        self.t_old = None
        self.t = t0
        self._fun, self.y = check_arguments(fun, y0, support_complex)
        self.t_bound = t_bound
        self.vectorized = vectorized

        if vectorized:
            def fun_single(t, y):
                return self._fun(t, y[:, None]).ravel()
            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            def fun_vectorized(t, y):
                f = np.empty_like(y)
                for i, yi in enumerate(y.T):
                    f[:, i] = self._fun(t, yi)
                return f

        def fun(t, y):
            self.nfev += 1
            return self.fun_single(t, y)

        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        self.status = 'running'

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = 'finished'

        return message

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        if self.t_old is None:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")

        if self.n == 0 or self.t == self.t_old:
            # Handle corner cases of empty solver and no integration.
            return ConstantDenseOutput(self.t_old, self.t, self.y)
        else:
            return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    def _dense_output_impl(self):
        raise NotImplementedError


class DenseOutput:
    """Base class for local interpolant over step made by an ODE solver.

    It interpolates between `t_min` and `t_max` (see Attributes below).
    Evaluation outside this interval is not forbidden, but the accuracy is not
    guaranteed.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, t_old, t):
        self.t_old = t_old
        self.t = t
        self.t_min = min(t, t_old)
        self.t_max = max(t, t_old)

    def __call__(self, t):
        """Evaluate the interpolant.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate the solution at.

        Returns
        -------
        y : ndarray, shape (n,) or (n, n_points)
            Computed values. Shape depends on whether `t` was a scalar or a
            1-D array.
        """
        t = np.asarray(t)
        if t.ndim > 1:
            raise ValueError("`t` must be a float or a 1-D array.")
        return self._call_impl(t)

    def _call_impl(self, t):
        raise NotImplementedError


class ConstantDenseOutput(DenseOutput):
    """Constant value interpolator.

    This class used for degenerate integration cases: equal integration limits
    or a system with 0 equations.
    """
    def __init__(self, t_old, t, value):
        super().__init__(t_old, t)
        self.value = value

    def _call_impl(self, t):
        if t.ndim == 0:
            return self.value
        else:
            ret = np.empty((self.value.shape[0], t.shape[0]))
            ret[:] = self.value[:, None]
            return ret
<|endoftext|>import numpy as np
from scipy.integrate import ode
from .common import validate_tol, validate_first_step, warn_extraneous
from .base import OdeSolver, DenseOutput


class LSODA(OdeSolver):
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches
    automatically between the nonstiff Adams method and the stiff BDF method.
    The method was originally detailed in [2]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    min_step : float, optional
        Minimum allowed step size. Default is 0.0, i.e., the step size is not
        bounded and determined solely by the solver.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : None or callable, optional
        Jacobian matrix of the right-hand side of the system with respect to
        ``y``. The Jacobian matrix has shape (n, n) and its element (i, j) is
        equal to ``d f_i / d y_j``. The function will be called as
        ``jac(t, y)``. If None (default), the Jacobian will be
        approximated by finite differences. It is generally recommended to
        provide the Jacobian rather than relying on a finite-difference
        approximation.
    lband, uband : int or None
        Parameters defining the bandwidth of the Jacobian,
        i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting
        these requires your jac routine to return the Jacobian in the packed format:
        the returned array must have ``n`` columns and ``uband + lband + 1``
        rows in which Jacobian diagonals are written. Specifically
        ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used
        in `scipy.linalg.solve_banded` (check for an illustration).
        These parameters can be also used with ``jac=None`` to reduce the
        number of Jacobian elements estimated by finite differences.
    vectorized : bool, optional
        Whether `fun` may be called in a vectorized fashion. False (default)
        is recommended for this solver.

        If ``vectorized`` is False, `fun` will always be called with ``y`` of
        shape ``(n,)``, where ``n = len(y0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` of shape
        ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
        such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
        the returned array is the time derivative of the state corresponding
        with a column of ``y``).

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for this solver.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.

    References
    ----------
    .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [2] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    """
    def __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, jac=None, lband=None,
                 uband=None, vectorized=False, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)

        if first_step is None:
            first_step = 0  # LSODA value for automatic selection.
        else:
            first_step = validate_first_step(first_step, t0, t_bound)

        first_step *= self.direction

        if max_step == np.inf:
            max_step = 0  # LSODA value for infinity.
        elif max_step <= 0:
            raise ValueError("`max_step` must be positive.")

        if min_step < 0:
            raise ValueError("`min_step` must be nonnegative.")

        rtol, atol = validate_tol(rtol, atol, self.n)

        solver = ode(self.fun, jac)
        solver.set_integrator('lsoda', rtol=rtol, atol=atol, max_step=max_step,
                              min_step=min_step, first_step=first_step,
                              lband=lband, uband=uband)
        solver.set_initial_value(y0, t0)

        # Inject t_bound into rwork array as needed for itask=5.
        solver._integrator.rwork[0] = self.t_bound
        solver._integrator.call_args[4] = solver._integrator.rwork

        self._lsoda_solver = solver

    def _step_impl(self):
        solver = self._lsoda_solver
        integrator = solver._integrator

        # From lsoda.step and lsoda.integrate itask=5 means take a single
        # step and do not go past t_bound.
        itask = integrator.call_args[2]
        integrator.call_args[2] = 5
        solver._y, solver.t = integrator.run(
            solver.f, solver.jac or (lambda: None), solver._y, solver.t,
            self.t_bound, solver.f_params, solver.jac_params)
        integrator.call_args[2] = itask

        if solver.successful():
            self.t = solver.t
            self.y = solver._y
            # From LSODA Fortran source njev is equal to nlu.
            self.njev = integrator.iwork[12]
            self.nlu = integrator.iwork[12]
            return True, None
        else:
            return False, 'Unexpected istate in LSODA.'

    def _dense_output_impl(self):
        iwork = self._lsoda_solver._integrator.iwork
        rwork = self._lsoda_solver._integrator.rwork

        # We want to produce the Nordsieck history array, yh, up to the order
        # used in the last successful iteration. The step size is unimportant
        # because it will be scaled out in LsodaDenseOutput. Some additional
        # work may be required because ODEPACK's LSODA implementation produces
        # the Nordsieck history in the state needed for the next iteration.

        # iwork[13] contains order from last successful iteration, while
        # iwork[14] contains order to be attempted next.
        order = iwork[13]

        # rwork[11] contains the step size to be attempted next, while
        # rwork[10] contains step size from last successful iteration.
        h = rwork[11]

        # rwork[20:20 + (iwork[14] + 1) * self.n] contains entries of the
        # Nordsieck array in state needed for next iteration. We want
        # the entries up to order for the last successful step so use the 
        # following.
        yh = np.reshape(rwork[20:20 + (order + 1) * self.n],
                        (self.n, order + 1), order='F').copy()
        if iwork[14] < order:
            # If the order is set to decrease then the final column of yh
            # has not been updated within ODEPACK's LSODA
            # implementation because this column will not be used in the
            # next iteration. We must rescale this column to make the
            # associated step size consistent with the other columns.
            yh[:, -1] *= (h / rwork[10]) ** order

        return LsodaDenseOutput(self.t_old, self.t, h, order, yh)


class LsodaDenseOutput(DenseOutput):
    def __init__(self, t_old, t, h, order, yh):
        super().__init__(t_old, t)
        self.h = h
        self.yh = yh
        self.p = np.arange(order + 1)

    def _call_impl(self, t):
        if t.ndim == 0:
            x = ((t - self.t) / self.h) ** self.p
        else:
            x = ((t - self.t) / self.h) ** self.p[:, None]

        return np.dot(self.yh, x)
<|endoftext|># Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

