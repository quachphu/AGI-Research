    ...                       jac_sparsity=sparsity_broyden(n))
    >>> res_3.cost
    4.5687069299604613e-23
    >>> res_3.optimality
    1.1650454296851518e-11

    Let's also solve a curve fitting problem using robust loss function to
    take care of outliers in the data. Define the model function as
    ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an
    observation and a, b, c are parameters to estimate.

    First, define the function which generates the data with noise and
    outliers, define the model parameters, and generate data:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    ...     rng = default_rng(seed)
    ...
    ...     y = a + b * np.exp(t * c)
    ...
    ...     error = noise * rng.standard_normal(t.size)
    ...     outliers = rng.integers(0, t.size, n_outliers)
    ...     error[outliers] *= 10
    ...
    ...     return y + error
    ...
    >>> a = 0.5
    >>> b = 2.0
    >>> c = -1
    >>> t_min = 0
    >>> t_max = 10
    >>> n_points = 15
    ...
    >>> t_train = np.linspace(t_min, t_max, n_points)
    >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)

    Define function for computing residuals and initial estimate of
    parameters.

    >>> def fun(x, t, y):
    ...     return x[0] + x[1] * np.exp(x[2] * t) - y
    ...
    >>> x0 = np.array([1.0, 1.0, 0.0])

    Compute a standard least-squares solution:

    >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))

    Now compute two solutions with two different robust loss functions. The
    parameter `f_scale` is set to 0.1, meaning that inlier residuals should
    not significantly exceed 0.1 (the noise level used).

    >>> res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
    ...                             args=(t_train, y_train))
    >>> res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
    ...                         args=(t_train, y_train))

    And, finally, plot all the curves. We see that by selecting an appropriate
    `loss`  we can get estimates close to optimal even in the presence of
    strong outliers. But keep in mind that generally it is recommended to try
    'soft_l1' or 'huber' losses first (if at all necessary) as the other two
    options may cause difficulties in optimization process.

    >>> t_test = np.linspace(t_min, t_max, n_points * 10)
    >>> y_true = gen_data(t_test, a, b, c)
    >>> y_lsq = gen_data(t_test, *res_lsq.x)
    >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
    >>> y_log = gen_data(t_test, *res_log.x)
    ...
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t_train, y_train, 'o')
    >>> plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    >>> plt.plot(t_test, y_lsq, label='linear loss')
    >>> plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
    >>> plt.plot(t_test, y_log, label='cauchy loss')
    >>> plt.xlabel("t")
    >>> plt.ylabel("y")
    >>> plt.legend()
    >>> plt.show()

    In the next example, we show how complex-valued residual functions of
    complex variables can be optimized with ``least_squares()``. Consider the
    following function:

    >>> def f(z):
    ...     return z - (0.5 + 0.5j)

    We wrap it into a function of real variables that returns real residuals
    by simply handling the real and imaginary parts as independent variables:

    >>> def f_wrap(x):
    ...     fx = f(x[0] + 1j*x[1])
    ...     return np.array([fx.real, fx.imag])

    Thus, instead of the original m-D complex function of n complex
    variables we optimize a 2m-D real function of 2n real variables:

    >>> from scipy.optimize import least_squares
    >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))
    >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j
    >>> z
    (0.49999999999925893+0.49999999999925893j)

    """
    if method not in ['trf', 'dogbox', 'lm']:
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")

    if jac not in ['2-point', '3-point', 'cs'] and not callable(jac):
        raise ValueError("`jac` must be '2-point', '3-point', 'cs' or "
                         "callable.")

    if tr_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")

    if loss not in IMPLEMENTED_LOSSES and not callable(loss):
        raise ValueError(f"`loss` must be one of {IMPLEMENTED_LOSSES.keys()}"
                         " or a callable.")

    if method == 'lm' and loss != 'linear':
        raise ValueError("method='lm' supports only 'linear' loss function.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    if np.iscomplexobj(x0):
        raise ValueError("`x0` must be real.")

    x0 = np.atleast_1d(x0).astype(float)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    if isinstance(bounds, Bounds):
        lb, ub = bounds.lb, bounds.ub
        bounds = (lb, ub)
    else:
        if len(bounds) == 2:
            lb, ub = prepare_bounds(bounds, x0.shape[0])
        else:
            raise ValueError("`bounds` must contain 2 elements.")

    if method == 'lm' and not np.all((lb == -np.inf) & (ub == np.inf)):
        raise ValueError("Method 'lm' doesn't support bounds.")

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    if not in_bounds(x0, lb, ub):
        raise ValueError("Initial guess is outside of provided bounds")

    x_scale = check_x_scale(x_scale, x0)

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)

    if method == 'trf':
        x0 = make_strictly_feasible(x0, lb, ub)

    if kwargs is None:
        kwargs = {}
    if tr_options is None:
        tr_options = {}

    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args, **kwargs))

    f0 = fun_wrapped(x0)

    if f0.ndim != 1:
        raise ValueError("`fun` must return at most 1-d array_like. "
                         f"f0.shape: {f0.shape}")

    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")

    n = x0.size
    m = f0.size

    if method == 'lm' and m < n:
        raise ValueError("Method 'lm' doesn't work when the number of "
                         "residuals is less than the number of variables.")

    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong "
                             "shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)

    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        if issparse(J0):
            J0 = J0.tocsr()

            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs).tocsr()

        elif isinstance(J0, LinearOperator):
            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # Estimate Jacobian by finite differences.
        if method == 'lm':
            if jac_sparsity is not None:
                raise ValueError("method='lm' does not support "
                                 "`jac_sparsity`.")

            if jac != '2-point':
                warn(f"jac='{jac}' works equivalently to '2-point' for method='lm'.",
                     stacklevel=2)

            J0 = jac_wrapped = None
        else:
            if jac_sparsity is not None and tr_solver == 'exact':
                raise ValueError("tr_solver='exact' is incompatible "
                                 "with `jac_sparsity`.")

            jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

            def jac_wrapped(x, f):
                J = approx_derivative(fun, x, rel_step=diff_step, method=jac,
                                      f0=f, bounds=bounds, args=args,
                                      kwargs=kwargs, sparsity=jac_sparsity)
                if J.ndim != 2:  # J is guaranteed not sparse.
                    J = np.atleast_2d(J)

                return J

            J0 = jac_wrapped(x0, f0)

    if J0 is not None:
        if J0.shape != (m, n):
            raise ValueError(
                f"The return value of `jac` has wrong shape: expected {(m, n)}, "
                f"actual {J0.shape}."
            )

        if not isinstance(J0, np.ndarray):
            if method == 'lm':
                raise ValueError("method='lm' works only with dense "
                                 "Jacobian matrices.")

            if tr_solver == 'exact':
                raise ValueError(
                    "tr_solver='exact' works only with dense "
                    "Jacobian matrices.")

        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError("x_scale='jac' can't be used when `jac` "
                             "returns LinearOperator.")

        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = 'exact'
            else:
                tr_solver = 'lsmr'

    if method == 'lm':
        result = call_minpack(fun_wrapped, x0, jac_wrapped, ftol, xtol, gtol,
                              max_nfev, x_scale, diff_step)

    elif method == 'trf':
        result = trf(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol,
                     gtol, max_nfev, x_scale, loss_function, tr_solver,
                     tr_options.copy(), verbose)

    elif method == 'dogbox':
        if tr_solver == 'lsmr' and 'regularize' in tr_options:
            warn("The keyword 'regularize' in `tr_options` is not relevant "
                 "for 'dogbox' method.",
                 stacklevel=2)
            tr_options = tr_options.copy()
            del tr_options['regularize']

        result = dogbox(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol,
                        xtol, gtol, max_nfev, x_scale, loss_function,
                        tr_solver, tr_options, verbose)

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    if verbose >= 1:
        print(result.message)
        print(f"Function evaluations {result.nfev}, initial cost {initial_cost:.4e}, "
              f"final cost {result.cost:.4e}, "
              f"first-order optimality {result.optimality:.2e}.")

    return result
<|endoftext|>"""
Dogleg algorithm with rectangular trust regions for least-squares minimization.

The description of the algorithm can be found in [Voglis]_. The algorithm does
trust-region iterations, but the shape of trust regions is rectangular as
opposed to conventional elliptical. The intersection of a trust region and
an initial feasible region is again some rectangle. Thus, on each iteration a
bound-constrained quadratic optimization problem is solved.

A quadratic problem is solved by well-known dogleg approach, where the
function is minimized along piecewise-linear "dogleg" path [NumOpt]_,
Chapter 4. If Jacobian is not rank-deficient then the function is decreasing
along this path, and optimization amounts to simply following along this
path as long as a point stays within the bounds. A constrained Cauchy step
(along the anti-gradient) is considered for safety in rank deficient cases,
in this situations the convergence might be slow.

If during iterations some variable hit the initial bound and the component
of anti-gradient points outside the feasible region, then a next dogleg step
won't make any progress. At this state such variables satisfy first-order
optimality conditions and they are excluded before computing a next dogleg
step.

Gauss-Newton step can be computed exactly by `numpy.linalg.lstsq` (for dense
Jacobian matrices) or by iterative procedure `scipy.sparse.linalg.lsmr` (for
dense and sparse matrices, or Jacobian being LinearOperator). The second
option allows to solve very large problems (up to couple of millions of
residuals on a regular PC), provided the Jacobian matrix is sufficiently
sparse. But note that dogbox is not very good for solving problems with
large number of constraints, because of variables exclusion-inclusion on each
iteration (a required number of function evaluations might be high or accuracy
of a solution will be poor), thus its large-scale usage is probably limited
to unconstrained problems.

References
----------
.. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
            Approach for Unconstrained and Bound Constrained Nonlinear
            Optimization", WSEAS International Conference on Applied
            Mathematics, Corfu, Greece, 2004.
.. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition".
"""
import numpy as np
from numpy.linalg import lstsq, norm

from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult

from .common import (
    step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic,
    build_quadratic_1d, minimize_quadratic_1d, compute_grad,
    compute_jac_scale, check_termination, scale_for_robust_loss_function,
    print_header_nonlinear, print_iteration_nonlinear)


def lsmr_operator(Jop, d, active_set):
    """Compute LinearOperator to use in LSMR by dogbox algorithm.

    `active_set` mask is used to excluded active variables from computations
    of matrix-vector products.
    """
    m, n = Jop.shape

    def matvec(x):
        x_free = x.ravel().copy()
        x_free[active_set] = 0
        return Jop.matvec(x * d)

    def rmatvec(x):
        r = d * Jop.rmatvec(x)
        r[active_set] = 0
        return r

    return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)


def find_intersection(x, tr_bounds, lb, ub):
    """Find intersection of trust-region bounds and initial bounds.

    Returns
    -------
    lb_total, ub_total : ndarray with shape of x
        Lower and upper bounds of the intersection region.
    orig_l, orig_u : ndarray of bool with shape of x
        True means that an original bound is taken as a corresponding bound
        in the intersection region.
    tr_l, tr_u : ndarray of bool with shape of x
        True means that a trust-region bound is taken as a corresponding bound
        in the intersection region.
    """
    lb_centered = lb - x
    ub_centered = ub - x

    lb_total = np.maximum(lb_centered, -tr_bounds)
    ub_total = np.minimum(ub_centered, tr_bounds)

    orig_l = np.equal(lb_total, lb_centered)
    orig_u = np.equal(ub_total, ub_centered)

    tr_l = np.equal(lb_total, -tr_bounds)
    tr_u = np.equal(ub_total, tr_bounds)

    return lb_total, ub_total, orig_l, orig_u, tr_l, tr_u


def dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub):
    """Find dogleg step in a rectangular region.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    bound_hits : ndarray of int, shape (n,)
        Each component shows whether a corresponding variable hits the
        initial bound after the step is taken:
            *  0 - a variable doesn't hit the bound.
            * -1 - lower bound is hit.
            *  1 - upper bound is hit.
    tr_hit : bool
        Whether the step hit the boundary of the trust-region.
    """
    lb_total, ub_total, orig_l, orig_u, tr_l, tr_u = find_intersection(
        x, tr_bounds, lb, ub
    )
    bound_hits = np.zeros_like(x, dtype=int)

    if in_bounds(newton_step, lb_total, ub_total):
        return newton_step, bound_hits, False

    to_bounds, _ = step_size_to_bound(np.zeros_like(x), -g, lb_total, ub_total)

    # The classical dogleg algorithm would check if Cauchy step fits into
    # the bounds, and just return it constrained version if not. But in a
    # rectangular trust region it makes sense to try to improve constrained
    # Cauchy step too. Thus, we don't distinguish these two cases.

    cauchy_step = -minimize_quadratic_1d(a, b, 0, to_bounds)[0] * g

    step_diff = newton_step - cauchy_step
    step_size, hits = step_size_to_bound(cauchy_step, step_diff,
                                         lb_total, ub_total)
    bound_hits[(hits < 0) & orig_l] = -1
    bound_hits[(hits > 0) & orig_u] = 1
    tr_hit = np.any((hits < 0) & tr_l | (hits > 0) & tr_u)

    return cauchy_step + step_size * step_diff, bound_hits, tr_hit


def dogbox(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
           loss_function, tr_solver, tr_options, verbose):
    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv, ord=np.inf)
    if Delta == 0:
        Delta = 1.0

    on_bound = np.zeros_like(x0, dtype=int)
    on_bound[np.equal(x0, lb)] = -1
    on_bound[np.equal(x0, ub)] = 1

    x = x0
    step = np.empty_like(x0)

    if max_nfev is None:
        max_nfev = x0.size * 100

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        active_set = on_bound * g < 0
        free_set = ~active_set

        g_free = g[free_set]
        g_full = g.copy()
        g[active_set] = 0

        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        x_free = x[free_set]
        lb_free = lb[free_set]
        ub_free = ub[free_set]
        scale_free = scale[free_set]

        # Compute (Gauss-)Newton and build quadratic model for Cauchy step.
        if tr_solver == 'exact':
            J_free = J[:, free_set]
            newton_step = lstsq(J_free, -f, rcond=-1)[0]

            # Coefficients for the quadratic model along the anti-gradient.
            a, b = build_quadratic_1d(J_free, g_free, -g_free)
        elif tr_solver == 'lsmr':
            Jop = aslinearoperator(J)

            # We compute lsmr step in scaled variables and then
            # transform back to normal variables, if lsmr would give exact lsq
            # solution, this would be equivalent to not doing any
            # transformations, but from experience it's better this way.

            # We pass active_set to make computations as if we selected
            # the free subset of J columns, but without actually doing any
            # slicing, which is expensive for sparse matrices and impossible
            # for LinearOperator.

            lsmr_op = lsmr_operator(Jop, scale, active_set)
            newton_step = -lsmr(lsmr_op, f, **tr_options)[0][free_set]
            newton_step *= scale_free

            # Components of g for active variables were zeroed, so this call
            # is correct and equivalent to using J_free and g_free.
            a, b = build_quadratic_1d(Jop, g, -g)

        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            tr_bounds = Delta * scale_free

            step_free, on_bound_free, tr_hit = dogleg_step(
                x_free, newton_step, g_free, a, b, tr_bounds, lb_free, ub_free)

            step.fill(0.0)
            step[free_set] = step_free

            if tr_solver == 'exact':
                predicted_reduction = -evaluate_quadratic(J_free, g_free,
                                                          step_free)
            elif tr_solver == 'lsmr':
                predicted_reduction = -evaluate_quadratic(Jop, g, step)

            # gh11403 ensure that solution is fully within bounds.
            x_new = np.clip(x + step, lb, ub)

            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step * scale_inv, ord=np.inf)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, tr_hit
            )

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)

            if termination_status is not None:
                break

        if actual_reduction > 0:
            on_bound[free_set] = on_bound_free

            x = x_new
            # Set variables exactly at the boundary.
            mask = on_bound == -1
            x[mask] = lb[mask]
            mask = on_bound == 1
            x[mask] = ub[mask]

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g_full, optimality=g_norm,
        active_mask=on_bound, nfev=nfev, njev=njev, status=termination_status)
<|endoftext|>"""Bounded-variable least-squares algorithm."""
import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize import OptimizeResult

from .common import print_header_linear, print_iteration_linear


def compute_kkt_optimality(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = np.abs(g[free_set])
    return np.max(g_kkt)


def bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose, rcond=None):
    m, n = A.shape

    x = x_lsq.copy()
    on_bound = np.zeros(n)

    mask = x <= lb
    x[mask] = lb[mask]
    on_bound[mask] = -1

    mask = x >= ub
    x[mask] = ub[mask]
    on_bound[mask] = 1

    free_set = on_bound == 0
    active_set = ~free_set
    free_set, = np.nonzero(free_set)

    r = A.dot(x) - b
    cost = 0.5 * np.dot(r, r)
    initial_cost = cost
    g = A.T.dot(r)

    cost_change = None
    step_norm = None
    iteration = 0

    if verbose == 2:
        print_header_linear()

    # This is the initialization loop. The requirement is that the
    # least-squares solution on free variables is feasible before BVLS starts.
    # One possible initialization is to set all variables to lower or upper
    # bounds, but many iterations may be required from this state later on.
    # The implemented ad-hoc procedure which intuitively should give a better
    # initial state: find the least-squares solution on current free variables,
    # if its feasible then stop, otherwise, set violating variables to
    # corresponding bounds and continue on the reduced set of free variables.

    while free_set.size > 0:
        if verbose == 2:
            optimality = compute_kkt_optimality(g, on_bound)
            print_iteration_linear(iteration, cost, cost_change, step_norm,
                                   optimality)

        iteration += 1
        x_free_old = x[free_set].copy()

        A_free = A[:, free_set]
        b_free = b - A.dot(x * active_set)
        z = lstsq(A_free, b_free, rcond=rcond)[0]

        lbv = z < lb[free_set]
        ubv = z > ub[free_set]
        v = lbv | ubv

        if np.any(lbv):
            ind = free_set[lbv]
            x[ind] = lb[ind]
            active_set[ind] = True
            on_bound[ind] = -1

        if np.any(ubv):
            ind = free_set[ubv]
            x[ind] = ub[ind]
            active_set[ind] = True
            on_bound[ind] = 1

        ind = free_set[~v]
        x[ind] = z[~v]

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new
        cost = cost_new
        g = A.T.dot(r)
        step_norm = norm(x[free_set] - x_free_old)

        if np.any(v):
            free_set = free_set[~v]
        else:
            break

    if max_iter is None:
        max_iter = n
    max_iter += iteration

    termination_status = None

    # Main BVLS loop.

    optimality = compute_kkt_optimality(g, on_bound)
    for iteration in range(iteration, max_iter):  # BVLS Loop A
        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, optimality)

        if optimality < tol:
            termination_status = 1

        if termination_status is not None:
            break

        move_to_free = np.argmax(g * on_bound)
        on_bound[move_to_free] = 0
        
        while True:   # BVLS Loop B

            free_set = on_bound == 0
            active_set = ~free_set
            free_set, = np.nonzero(free_set)
    
            x_free = x[free_set]
            x_free_old = x_free.copy()
            lb_free = lb[free_set]
            ub_free = ub[free_set]

            A_free = A[:, free_set]
            b_free = b - A.dot(x * active_set)
            z = lstsq(A_free, b_free, rcond=rcond)[0]

            lbv, = np.nonzero(z < lb_free)
            ubv, = np.nonzero(z > ub_free)
            v = np.hstack((lbv, ubv))

            if v.size > 0:
                alphas = np.hstack((
                    lb_free[lbv] - x_free[lbv],
                    ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])

                i = np.argmin(alphas)
                i_free = v[i]
                alpha = alphas[i]

                x_free *= 1 - alpha
                x_free += alpha * z
                x[free_set] = x_free

                if i < lbv.size:
                    on_bound[free_set[i_free]] = -1
                else:
                    on_bound[free_set[i_free]] = 1
            else:
                x_free = z
                x[free_set] = x_free
                break

        step_norm = norm(x_free - x_free_old)

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new

        if cost_change < tol * cost:
            termination_status = 2
        cost = cost_new

        g = A.T.dot(r)
        optimality = compute_kkt_optimality(g, on_bound)

    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=optimality, active_mask=on_bound,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
<|endoftext|>"""Trust Region Reflective algorithm for least-squares optimization.

The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v:

           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise

where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as

    D^2 g(x) = 0.

Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.

Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies

    (D^2 H + diag(g) Jv) p = -D^2 g

where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying

    B_h p_h = -g_h,

where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables:

    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta

In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is

    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta

Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).

The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).

The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
three: a constrained trust-region step, a reflected step and a constrained
Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
space).

Another feature is that a trust-region radius control strategy is modified to
account for appearance of the diagonal C matrix (called diag_h in the code).

Note that all described peculiarities are completely gone as we consider
problems without bounds (the algorithm becomes a standard trust-region type
algorithm very similar to ones implemented in MINPACK).

The implementation supports two methods of solving the trust-region problem.
The first, called 'exact', applies SVD on Jacobian and then solves the problem
very accurately using the algorithm described in [JJMore]_. It is not
applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
approach (sometimes called "indefinite dogleg"), where the problem is solved
in a subspace spanned by the gradient and the approximate Gauss-Newton step
found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
reformulated as a 4th order algebraic equation and solved very accurately by
``numpy.roots``. The subspace approach allows to solve very large problems
(up to couple of millions of residuals on a regular PC), provided the Jacobian
matrix is sufficiently sparse.

References
----------
.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
      and Conjugate Gradient Method for Large-Scale Bound-Constrained
      Minimization Problems," SIAM Journal on Scientific Computing,
      Vol. 21, Number 1, pp 1-23, 1999.
.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
"""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult

from .common import (
    step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, intersect_trust_region, solve_lsq_trust_region,
    solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
    evaluate_quadratic, right_multiplied_operator, regularized_lsq_operator,
    CL_scaling_vector, compute_grad, compute_jac_scale, check_termination,
    update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear,
    print_iteration_nonlinear)


def trf(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
        loss_function, tr_solver, tr_options, verbose):
    # For efficiency, it makes sense to run the simplified version of the
    # algorithm when no bounds are imposed. We decided to write the two
    # separate functions. It violates the DRY principle, but the individual
    # functions are kept the most readable.
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        return trf_no_bounds(
            fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale,
            loss_function, tr_solver, tr_options, verbose)
    else:
        return trf_bounds(
            fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
            loss_function, tr_solver, tr_options, verbose)


def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict trust-region step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or trust
    # region boundary.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Find lower and upper bounds on a step size along the reflected
    # direction, considering the strict feasibility requirement. There is no
    # single correct way to do that, the chosen approach seems to work best
    # on test problems.
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    # Check if reflection step is available.
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

    ag_h = -g_h
    ag = d * ag_h

    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr

    a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


def trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev,
               x_scale, loss_function, tr_solver, tr_options, verbose):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    f_augmented = np.zeros(m + n)
    if tr_solver == 'exact':
        J_augmented = np.empty((m + n, n))
    elif tr_solver == 'lsmr':
        reg_term = 0.0
        regularize = tr_options.pop('regularize', True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        # Now compute variables in "hat" space. Here, we also account for
        # scaling introduced by `x_scale` parameter. This part is a bit tricky,
        # you have to write down the formulas and see how the trust-region
        # problem is formulated when the two types of scaling are applied.
        # The idea is that first we apply `x_scale` and then apply Coleman-Li
        # approach in the new variables.

        # v is recomputed in the variables after applying `x_scale`, note that
        # components which were identically 1 not affected.
        v[dv != 0] *= scale_inv[dv != 0]

        # Here, we apply two types of scaling.
        d = v**0.5 * scale

        # C = diag(g * scale) Jv
        diag_h = g * dv * scale

        # After all this has been done, we continue normally.

        # "hat" gradient.
        g_h = d * g

        f_augmented[:m] = f
        if tr_solver == 'exact':
            J_augmented[:m] = J * d
            J_h = J_augmented[:m]  # Memory view.
            J_augmented[m:] = np.diag(diag_h**0.5)
            U, s, V = svd(J_augmented, full_matrices=False)
            V = V.T
            uf = U.T.dot(f_augmented)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h, diag=diag_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            lsmr_op = regularized_lsq_operator(J_h, (diag_h + reg_term)**0.5)
            gn_h = lsmr(lsmr_op, f_augmented, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode='economic')
            JS = J_h.dot(S)  # LinearOperator does dot too.
            B_S = np.dot(JS.T, JS) + np.dot(S.T * diag_h, S)
            g_S = S.T.dot(g_h)

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                p_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                p_h = S.dot(p_S)

            p = d * p_h  # Trust-region solution in the original space.
            step, step_h, predicted_reduction = select_step(
                x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)


def trf_no_bounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev,
                  x_scale, loss_function, tr_solver, tr_options, verbose):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    if tr_solver == 'lsmr':
        reg_term = 0
        damp = tr_options.pop('damp', 0.0)
        regularize = tr_options.pop('regularize', True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        d = scale
        g_h = d * g

        if tr_solver == 'exact':
            J_h = J * d
            U, s, V = svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.dot(f)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            damp_full = (damp**2 + reg_term)**0.5
            gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode='economic')
            JS = J_h.dot(S)
            B_S = np.dot(JS.T, JS)
            g_S = S.T.dot(g_h)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                step_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                step_h = S.dot(p_S)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = np.zeros_like(x)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)
<|endoftext|>"""
================================
Datasets (:mod:`scipy.datasets`)
================================

.. currentmodule:: scipy.datasets

Dataset Methods
===============

.. autosummary::
   :toctree: generated/

   ascent
   face
   electrocardiogram

Utility Methods
===============

.. autosummary::
   :toctree: generated/

   download_all    -- Download all the dataset files to specified path.
   clear_cache     -- Clear cached dataset directory.


Usage of Datasets
=================

SciPy dataset methods can be simply called as follows: ``'<dataset-name>()'``
This downloads the dataset files over the network once, and saves the cache,
before returning a `numpy.ndarray` object representing the dataset.

Note that the return data structure and data type might be different for
different dataset methods. For a more detailed example on usage, please look
into the particular dataset method documentation above.


How dataset retrieval and storage works
=======================================

SciPy dataset files are stored within individual GitHub repositories under the
SciPy GitHub organization, following a naming convention as
``'dataset-<name>'``, for example `scipy.datasets.face` files live at
https://github.com/scipy/dataset-face.  The `scipy.datasets` submodule utilizes
and depends on `Pooch <https://www.fatiando.org/pooch/latest/>`_, a Python
package built to simplify fetching data files. Pooch uses these repos to
retrieve the respective dataset files when calling the dataset function.

A registry of all the datasets, essentially a mapping of filenames with their
SHA256 hash and repo urls are maintained, which Pooch uses to handle and verify
the downloads on function call. After downloading the dataset once, the files
are saved in the system cache directory under ``'scipy-data'``.

Dataset cache locations may vary on different platforms.

For macOS::

    '~/Library/Caches/scipy-data'

For Linux and other Unix-like platforms::

    '~/.cache/scipy-data'  # or the value of the XDG_CACHE_HOME env var, if defined

For Windows::

    'C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\scipy-data\\Cache'


In environments with constrained network connectivity for various security
reasons or on systems without continuous internet connections, one may manually
load the cache of the datasets by placing the contents of the dataset repo in
the above mentioned cache directory to avoid fetching dataset errors without
the internet connectivity.

"""


from ._fetchers import face, ascent, electrocardiogram
from ._download_all import download_all
from ._utils import clear_cache

__all__ = ['ascent', 'electrocardiogram', 'face',
           'download_all', 'clear_cache']


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import os
import shutil
from ._registry import method_files_map

try:
    import platformdirs
except ImportError:
    platformdirs = None  # type: ignore[assignment]


def _clear_cache(datasets, cache_dir=None, method_map=None):
    if method_map is None:
        # Use SciPy Datasets method map
        method_map = method_files_map
    if cache_dir is None:
        # Use default cache_dir path
        if platformdirs is None:
            # platformdirs is pooch dependency
            raise ImportError("Missing optional dependency 'pooch' required "
                              "for scipy.datasets module. Please use pip or "
                              "conda to install 'pooch'.")
        cache_dir = platformdirs.user_cache_dir("scipy-data")

    if not os.path.exists(cache_dir):
        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
        return

    if datasets is None:
        print(f"Cleaning the cache directory {cache_dir}!")
        shutil.rmtree(cache_dir)
    else:
        if not isinstance(datasets, (list, tuple)):
            # single dataset method passed should be converted to list
            datasets = [datasets, ]
        for dataset in datasets:
            assert callable(dataset)
            dataset_name = dataset.__name__  # Name of the dataset method
            if dataset_name not in method_map:
                raise ValueError(f"Dataset method {dataset_name} doesn't "
                                 "exist. Please check if the passed dataset "
                                 "is a subset of the following dataset "
                                 f"methods: {list(method_map.keys())}")

            data_files = method_map[dataset_name]
            data_filepaths = [os.path.join(cache_dir, file)
                              for file in data_files]
            for data_filepath in data_filepaths:
                if os.path.exists(data_filepath):
                    print("Cleaning the file "
                          f"{os.path.split(data_filepath)[1]} "
                          f"for dataset {dataset_name}")
                    os.remove(data_filepath)
                else:
                    print(f"Path {data_filepath} doesn't exist. "
                          "Nothing to clear.")


def clear_cache(datasets=None):
    """
    Cleans the scipy datasets cache directory.

    If a scipy.datasets method or a list/tuple of the same is
    provided, then clear_cache removes all the data files
    associated to the passed dataset method callable(s).

    By default, it removes all the cached data files.

    Parameters
    ----------
    datasets : callable or list/tuple of callable or None

    Examples
    --------
    >>> from scipy import datasets
    >>> ascent_array = datasets.ascent()
    >>> ascent_array.shape
    (512, 512)
    >>> datasets.clear_cache([datasets.ascent])
    Cleaning the file ascent.dat for dataset ascent
    """
    _clear_cache(datasets)
<|endoftext|>"""
Platform independent script to download all the
`scipy.datasets` module data files.
This doesn't require a full scipy build.

Run: python _download_all.py <download_dir>
"""

import argparse
try:
    import pooch
except ImportError:
    pooch = None


if __package__ is None or __package__ == '':
    # Running as python script, use absolute import
    import _registry  # type: ignore
else:
    # Running as python module, use relative import
    from . import _registry


def download_all(path=None):
    """
    Utility method to download all the dataset files
    for `scipy.datasets` module.

    Parameters
    ----------
    path : str, optional
        Directory path to download all the dataset files.
        If None, default to the system cache_dir detected by pooch.
    """
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    if path is None:
        path = pooch.os_cache('scipy-data')
    for dataset_name, dataset_hash in _registry.registry.items():
        pooch.retrieve(url=_registry.registry_urls[dataset_name],
                       known_hash=dataset_hash,
                       fname=dataset_name, path=path)


def main():
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)


if __name__ == "__main__":
    main()
<|endoftext|>from numpy import array, frombuffer, load
from ._registry import registry, registry_urls

try:
    import pooch
except ImportError:
    pooch = None
    data_fetcher = None
else:
    data_fetcher = pooch.create(
        # Use the default cache folder for the operating system
        # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
        # select an appropriate directory for the cache on each platform.
        path=pooch.os_cache("scipy-data"),

        # The remote data is on Github
        # base_url is a required param, even though we override this
        # using individual urls in the registry.
        base_url="https://github.com/scipy/",
        registry=registry,
        urls=registry_urls
    )


def fetch_data(dataset_name, data_fetcher=data_fetcher):
    if data_fetcher is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    # The "fetch" method returns the full path to the downloaded data file.
    return data_fetcher.fetch(dataset_name)


def ascent():
    """
    Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy
    use in demos.

    The image is derived from
    https://pixnio.com/people/accent-to-the-top

    Parameters
    ----------
    None

    Returns
    -------
    ascent : ndarray
       convenient image to use for testing and demonstration

    Examples
    --------
    >>> import scipy.datasets
    >>> ascent = scipy.datasets.ascent()
    >>> ascent.shape
    (512, 512)
    >>> ascent.max()
    np.uint8(255)

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(ascent)
    >>> plt.show()

    """
    import pickle

    # The file will be downloaded automatically the first time this is run,
    # returning the path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = fetch_data("ascent.dat")
    # Now we just need to load it with our standard Python tools.
    with open(fname, 'rb') as f:
        ascent = array(pickle.load(f))
    return ascent


def electrocardiogram():
    """
    Load an electrocardiogram as an example for a 1-D signal.

    The returned signal is a 5 minute long electrocardiogram (ECG), a medical
    recording of the heart's electrical activity, sampled at 360 Hz.

    Returns
    -------
    ecg : ndarray
        The electrocardiogram in millivolt (mV) sampled at 360 Hz.

    Notes
    -----
    The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
    (lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
    PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
    heartbeats as well as pathological changes.

    .. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
           IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
           (PMID: 11446209); :doi:`10.13026/C2F305`
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
           PhysioToolkit, and PhysioNet: Components of a New Research Resource
           for Complex Physiologic Signals. Circulation 101(23):e215-e220;
           :doi:`10.1161/01.CIR.101.23.e215`

    Examples
    --------
    >>> from scipy.datasets import electrocardiogram
    >>> ecg = electrocardiogram()
    >>> ecg
    array([-0.245, -0.215, -0.185, ..., -0.405, -0.395, -0.385], shape=(108000,))
    >>> ecg.shape, ecg.mean(), ecg.std()
    ((108000,), -0.16510875, 0.5992473991177294)

    As stated the signal features several areas with a different morphology.
    E.g., the first few seconds show the electrical activity of a heart in
    normal sinus rhythm as seen below.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fs = 360
    >>> time = np.arange(ecg.size) / fs
    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(9, 10.2)
    >>> plt.ylim(-1, 1.5)
    >>> plt.show()

    After second 16, however, the first premature ventricular contractions,
    also called extrasystoles, appear. These have a different morphology
    compared to typical heartbeats. The difference can easily be observed
    in the following plot.

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(46.5, 50)
    >>> plt.ylim(-2, 1.5)
    >>> plt.show()

    At several points large artifacts disturb the recording, e.g.:

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(207, 215)
    >>> plt.ylim(-2, 3.5)
    >>> plt.show()

    Finally, examining the power spectrum reveals that most of the biosignal is
    made up of lower frequencies. At 60 Hz the noise induced by the mains
    electricity can be clearly observed.

    >>> from scipy.signal import welch
    >>> f, Pxx = welch(ecg, fs=fs, nperseg=2048, scaling="spectrum")
    >>> plt.semilogy(f, Pxx)
    >>> plt.xlabel("Frequency in Hz")
    >>> plt.ylabel("Power spectrum of the ECG in mV**2")
    >>> plt.xlim(f[[0, -1]])
    >>> plt.show()
    """
    fname = fetch_data("ecg.dat")
    with load(fname) as file:
        ecg = file["ecg"].astype(int)  # np.uint16 -> int
    # Convert raw output of ADC to mV: (ecg - adc_zero) / adc_gain
    ecg = (ecg - 1024) / 200.0
    return ecg


def face(gray=False):
    """
    Get a 1024 x 768, color image of a raccoon face.

    The image is derived from
    https://pixnio.com/fauna-animals/raccoons/raccoon-procyon-lotor

    Parameters
    ----------
    gray : bool, optional
        If True return 8-bit grey-scale image, otherwise return a color image

    Returns
    -------
    face : ndarray
        image of a raccoon face

    Examples
    --------
    >>> import scipy.datasets
    >>> face = scipy.datasets.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    np.uint8(255)

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    """
    import bz2
    fname = fetch_data("face.dat")
    with open(fname, 'rb') as f:
        rawdata = f.read()
    face_data = bz2.decompress(rawdata)
    face = frombuffer(face_data, dtype='uint8')
    face.shape = (768, 1024, 3)
    if gray is True:
        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
                0.07 * face[:, :, 2]).astype('uint8')
    return face
<|endoftext|># mypy: disable-error-code="attr-defined"
import warnings
import numpy as np
import scipy._lib._elementwise_iterative_method as eim
from scipy._lib._util import _RichResult
from scipy._lib._array_api import array_namespace, xp_sign, xp_copy, xp_take_along_axis

_EERRORINCREASE = -1  # used in derivative

def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
                   step_factor, step_direction, preserve_shape, callback):
    # Input validation for `derivative`
    xp = array_namespace(x)

    if not callable(f):
        raise ValueError('`f` must be callable.')

    if not np.iterable(args):
        args = (args,)

    tolerances = {} if tolerances is None else tolerances
    atol = tolerances.get('atol', None)
    rtol = tolerances.get('rtol', None)

    # tolerances are floats, not arrays; OK to use NumPy
    message = 'Tolerances and step parameters must be non-negative scalars.'
    tols = np.asarray([atol if atol is not None else 1,
                       rtol if rtol is not None else 1,
                       step_factor])
    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
            or np.any(np.isnan(tols)) or tols.shape != (3,)):
        raise ValueError(message)
    step_factor = float(tols[2])

    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter <= 0:
        raise ValueError('`maxiter` must be a positive integer.')

    order_int = int(order)
    if order_int != order or order <= 0:
        raise ValueError('`order` must be a positive integer.')

    step_direction = xp.asarray(step_direction)
    initial_step = xp.asarray(initial_step)
    temp = xp.broadcast_arrays(x, step_direction, initial_step)
    x, step_direction, initial_step = temp

    message = '`preserve_shape` must be True or False.'
    if preserve_shape not in {True, False}:
        raise ValueError(message)

    if callback is not None and not callable(callback):
        raise ValueError('`callback` must be callable.')

    return (f, x, args, atol, rtol, maxiter_int, order_int, initial_step,
            step_factor, step_direction, preserve_shape, callback)


def derivative(f, x, *, args=(), tolerances=None, maxiter=10,
               order=8, initial_step=0.5, step_factor=2.0,
               step_direction=0, preserve_shape=False, callback=None):
    """Evaluate the derivative of a elementwise, real scalar function numerically.

    For each element of the output of `f`, `derivative` approximates the first
    derivative of `f` at the corresponding element of `x` using finite difference
    differentiation.

    This function works elementwise when `x`, `step_direction`, and `args` contain
    (broadcastable) arrays.

    Parameters
    ----------
    f : callable
        The function whose derivative is desired. The signature must be::

            f(xi: ndarray, *argsi) -> ndarray

        where each element of ``xi`` is a finite real number and ``argsi`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable with
        ``xi``. `f` must be an elementwise function: each scalar element ``f(xi)[j]``
        must equal ``f(xi[j])`` for valid indices ``j``. It must not mutate the array
        ``xi`` or the arrays in ``argsi``.
    x : float array_like
        Abscissae at which to evaluate the derivative. Must be broadcastable with
        `args` and `step_direction`.
    args : tuple of array_like, optional
        Additional positional array arguments to be passed to `f`. Arrays
        must be broadcastable with one another and the arrays of `init`.
        If the callable for which the root is desired requires arguments that are
        not broadcastable with `x`, wrap that callable with `f` such that `f`
        accepts only `x` and broadcastable ``*args``.
    tolerances : dictionary of floats, optional
        Absolute and relative tolerances. Valid keys of the dictionary are:

        - ``atol`` - absolute tolerance on the derivative
        - ``rtol`` - relative tolerance on the derivative

        Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
        `atol` is the smallest normal number of the appropriate dtype, and
        the default `rtol` is the square root of the precision of the
        appropriate dtype.
    order : int, default: 8
        The (positive integer) order of the finite difference formula to be
        used. Odd integers will be rounded up to the next even integer.
    initial_step : float array_like, default: 0.5
        The (absolute) initial step size for the finite difference derivative
        approximation.
    step_factor : float, default: 2.0
        The factor by which the step size is *reduced* in each iteration; i.e.
        the step size in iteration 1 is ``initial_step/step_factor``. If
        ``step_factor < 1``, subsequent steps will be greater than the initial
        step; this may be useful if steps smaller than some threshold are
        undesirable (e.g. due to subtractive cancellation error).
    maxiter : int, default: 10
        The maximum number of iterations of the algorithm to perform. See
        Notes.
    step_direction : integer array_like
        An array representing the direction of the finite difference steps (for
        use when `x` lies near to the boundary of the domain of the function.)
        Must be broadcastable with `x` and all `args`.
        Where 0 (default), central differences are used; where negative (e.g.
        -1), steps are non-positive; and where positive (e.g. 1), all steps are
        non-negative.
    preserve_shape : bool, default: False
        In the following, "arguments of `f`" refers to the array ``xi`` and
        any arrays within ``argsi``. Let ``shape`` be the broadcasted shape
        of `x` and all elements of `args` (which is conceptually
        distinct from ``xi` and ``argsi`` passed into `f`).

        - When ``preserve_shape=False`` (default), `f` must accept arguments
          of *any* broadcastable shapes.

        - When ``preserve_shape=True``, `f` must accept arguments of shape
          ``shape`` *or* ``shape + (n,)``, where ``(n,)`` is the number of
          abscissae at which the function is being evaluated.

        In either case, for each scalar element ``xi[j]`` within ``xi``, the array
        returned by `f` must include the scalar ``f(xi[j])`` at the same index.
        Consequently, the shape of the output is always the shape of the input
        ``xi``.

        See Examples.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `derivative` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `derivative` will return a result. `callback` must not mutate
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
            - ``-1`` : The error estimate increased, so iteration was terminated.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).

        df : float array
            The derivative of `f` at `x`, if the algorithm terminated
            successfully.
        error : float array
            An estimate of the error: the magnitude of the difference between
            the current estimate of the derivative and the estimate in the
            previous iteration.
        nit : int array
            The number of iterations of the algorithm that were performed.
        nfev : int array
            The number of points at which `f` was evaluated.
        x : float array
            The value at which the derivative of `f` was evaluated
            (after broadcasting with `args` and `step_direction`).

    See Also
    --------
    jacobian, hessian

    Notes
    -----
    The implementation was inspired by jacobi [1]_, numdifftools [2]_, and
    DERIVEST [3]_, but the implementation follows the theory of Taylor series
    more straightforwardly (and arguably naively so).
    In the first iteration, the derivative is estimated using a finite
    difference formula of order `order` with maximum step size `initial_step`.
    Each subsequent iteration, the maximum step size is reduced by
    `step_factor`, and the derivative is estimated again until a termination
    condition is reached. The error estimate is the magnitude of the difference
    between the current derivative approximation and that of the previous
    iteration.

    The stencils of the finite difference formulae are designed such that
    abscissae are "nested": after `f` is evaluated at ``order + 1``
    points in the first iteration, `f` is evaluated at only two new points
    in each subsequent iteration; ``order - 1`` previously evaluated function
    values required by the finite difference formula are reused, and two
    function values (evaluations at the points furthest from `x`) are unused.

    Step sizes are absolute. When the step size is small relative to the
    magnitude of `x`, precision is lost; for example, if `x` is ``1e20``, the
    default initial step size of ``0.5`` cannot be resolved. Accordingly,
    consider using larger initial step sizes for large magnitudes of `x`.

    The default tolerances are challenging to satisfy at points where the
    true derivative is exactly zero. If the derivative may be exactly zero,
    consider specifying an absolute tolerance (e.g. ``atol=1e-12``) to
    improve convergence.

    References
    ----------
    .. [1] Hans Dembinski (@HDembinski). jacobi.
           https://github.com/HDembinski/jacobi
    .. [2] Per A. Brodtkorb and John D'Errico. numdifftools.
           https://numdifftools.readthedocs.io/en/latest/
    .. [3] John D'Errico. DERIVEST: Adaptive Robust Numerical Differentiation.
           https://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation
    .. [4] Numerical Differentition. Wikipedia.
           https://en.wikipedia.org/wiki/Numerical_differentiation

    Examples
    --------
    Evaluate the derivative of ``np.exp`` at several points ``x``.

    >>> import numpy as np
    >>> from scipy.differentiate import derivative
    >>> f = np.exp
    >>> df = np.exp  # true derivative
    >>> x = np.linspace(1, 2, 5)
    >>> res = derivative(f, x)
    >>> res.df  # approximation of the derivative
    array([2.71828183, 3.49034296, 4.48168907, 5.75460268, 7.3890561 ])
    >>> res.error  # estimate of the error
    array([7.13740178e-12, 9.16600129e-12, 1.17594823e-11, 1.51061386e-11,
           1.94262384e-11])
    >>> abs(res.df - df(x))  # true error
    array([2.53130850e-14, 3.55271368e-14, 5.77315973e-14, 5.59552404e-14,
           6.92779167e-14])

    Show the convergence of the approximation as the step size is reduced.
    Each iteration, the step size is reduced by `step_factor`, so for
    sufficiently small initial step, each iteration reduces the error by a
    factor of ``1/step_factor**order`` until finite precision arithmetic
    inhibits further improvement.

    >>> import matplotlib.pyplot as plt
    >>> iter = list(range(1, 12))  # maximum iterations
    >>> hfac = 2  # step size reduction per iteration
    >>> hdir = [-1, 0, 1]  # compare left-, central-, and right- steps
    >>> order = 4  # order of differentiation formula
    >>> x = 1
    >>> ref = df(x)
    >>> errors = []  # true error
    >>> for i in iter:
    ...     res = derivative(f, x, maxiter=i, step_factor=hfac,
    ...                      step_direction=hdir, order=order,
    ...                      # prevent early termination
    ...                      tolerances=dict(atol=0, rtol=0))
    ...     errors.append(abs(res.df - ref))
    >>> errors = np.array(errors)
    >>> plt.semilogy(iter, errors[:, 0], label='left differences')
    >>> plt.semilogy(iter, errors[:, 1], label='central differences')
    >>> plt.semilogy(iter, errors[:, 2], label='right differences')
    >>> plt.xlabel('iteration')
    >>> plt.ylabel('error')
    >>> plt.legend()
    >>> plt.show()
    >>> (errors[1, 1] / errors[0, 1], 1 / hfac**order)
    (0.06215223140159822, 0.0625)

    The implementation is vectorized over `x`, `step_direction`, and `args`.
    The function is evaluated once before the first iteration to perform input
    validation and standardization, and once per iteration thereafter.

    >>> def f(x, p):
    ...     f.nit += 1
    ...     return x**p
    >>> f.nit = 0
    >>> def df(x, p):
    ...     return p*x**(p-1)
    >>> x = np.arange(1, 5)
    >>> p = np.arange(1, 6).reshape((-1, 1))
    >>> hdir = np.arange(-1, 2).reshape((-1, 1, 1))
    >>> res = derivative(f, x, args=(p,), step_direction=hdir, maxiter=1)
    >>> np.allclose(res.df, df(x, p))
    True
    >>> res.df.shape
    (3, 5, 4)
    >>> f.nit
    2

    By default, `preserve_shape` is False, and therefore the callable
    `f` may be called with arrays of any broadcastable shapes.
    For example:

    >>> shapes = []
    >>> def f(x, c):
    ...    shape = np.broadcast_shapes(x.shape, c.shape)
    ...    shapes.append(shape)
    ...    return np.sin(c*x)
    >>>
    >>> c = [1, 5, 10, 20]
    >>> res = derivative(f, 0, args=(c,))
    >>> shapes
    [(4,), (4, 8), (4, 2), (3, 2), (2, 2), (1, 2)]

    To understand where these shapes are coming from - and to better
    understand how `derivative` computes accurate results - note that
    higher values of ``c`` correspond with higher frequency sinusoids.
    The higher frequency sinusoids make the function's derivative change
    faster, so more function evaluations are required to achieve the target
    accuracy:

    >>> res.nfev
    array([11, 13, 15, 17], dtype=int32)

    The initial ``shape``, ``(4,)``, corresponds with evaluating the
    function at a single abscissa and all four frequencies; this is used
    for input validation and to determine the size and dtype of the arrays
    that store results. The next shape corresponds with evaluating the
    function at an initial grid of abscissae and all four frequencies.
    Successive calls to the function evaluate the function at two more
    abscissae, increasing the effective order of the approximation by two.
    However, in later function evaluations, the function is evaluated at
    fewer frequencies because the corresponding derivative has already
    converged to the required tolerance. This saves function evaluations to
    improve performance, but it requires the function to accept arguments of
    any shape.

    "Vector-valued" functions are unlikely to satisfy this requirement.
    For example, consider

    >>> def f(x):
    ...    return [x, np.sin(3*x), x+np.sin(10*x), np.sin(20*x)*(x-1)**2]

    This integrand is not compatible with `derivative` as written; for instance,
    the shape of the output will not be the same as the shape of ``x``. Such a
    function *could* be converted to a compatible form with the introduction of
    additional parameters, but this would be inconvenient. In such cases,
    a simpler solution would be to use `preserve_shape`.

    >>> shapes = []
    >>> def f(x):
    ...     shapes.append(x.shape)
    ...     x0, x1, x2, x3 = x
    ...     return [x0, np.sin(3*x1), x2+np.sin(10*x2), np.sin(20*x3)*(x3-1)**2]
    >>>
    >>> x = np.zeros(4)
    >>> res = derivative(f, x, preserve_shape=True)
    >>> shapes
    [(4,), (4, 8), (4, 2), (4, 2), (4, 2), (4, 2)]

    Here, the shape of ``x`` is ``(4,)``. With ``preserve_shape=True``, the
    function may be called with argument ``x`` of shape ``(4,)`` or ``(4, n)``,
    and this is what we observe.

    """
    # TODO (followup):
    #  - investigate behavior at saddle points
    #  - multivariate functions?
    #  - relative steps?
    #  - show example of `np.vectorize`

    res = _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
                            step_factor, step_direction, preserve_shape, callback)
    (func, x, args, atol, rtol, maxiter, order,
     h0, fac, hdir, preserve_shape, callback) = res

    # Initialization
    # Since f(x) (no step) is not needed for central differences, it may be
    # possible to eliminate this function evaluation. However, it's useful for
    # input validation and standardization, and everything else is designed to
    # reduce function calls, so let's keep it simple.
    temp = eim._initialize(func, (x,), args, preserve_shape=preserve_shape)
    func, xs, fs, args, shape, dtype, xp = temp

    finfo = xp.finfo(dtype)
    atol = finfo.smallest_normal if atol is None else atol
    rtol = finfo.eps**0.5 if rtol is None else rtol  # keep same as `hessian`

    x, f = xs[0], fs[0]
    df = xp.full_like(f, xp.nan)

    # Ideally we'd broadcast the shape of `hdir` in `_elementwise_algo_init`, but
    # it's simpler to do it here than to generalize `_elementwise_algo_init` further.
    # `hdir` and `x` are already broadcasted in `_derivative_iv`, so we know
    # that `hdir` can be broadcasted to the final shape. Same with `h0`.
    hdir = xp.broadcast_to(hdir, shape)
    hdir = xp.reshape(hdir, (-1,))
    hdir = xp.astype(xp_sign(hdir), dtype)
    h0 = xp.broadcast_to(h0, shape)
    h0 = xp.reshape(h0, (-1,))
    h0 = xp.astype(h0, dtype)
    h0[h0 <= 0] = xp.asarray(xp.nan, dtype=dtype)

    status = xp.full_like(x, eim._EINPROGRESS, dtype=xp.int32)  # in progress
    nit, nfev = 0, 1  # one function evaluations performed above
    # Boolean indices of left, central, right, and (all) one-sided steps
    il = hdir < 0
    ic = hdir == 0
    ir = hdir > 0
    io = il | ir

    # Most of these attributes are reasonably obvious, but:
    # - `fs` holds all the function values of all active `x`. The zeroth
    #   axis corresponds with active points `x`, the first axis corresponds
    #   with the different steps (in the order described in
    #   `_derivative_weights`).
    # - `terms` (which could probably use a better name) is half the `order`,
    #   which is always even.
    work = _RichResult(x=x, df=df, fs=f[:, xp.newaxis], error=xp.nan, h=h0,
                       df_last=xp.nan, error_last=xp.nan, fac=fac,
                       atol=atol, rtol=rtol, nit=nit, nfev=nfev,
                       status=status, dtype=dtype, terms=(order+1)//2,
                       hdir=hdir, il=il, ic=ic, ir=ir, io=io,
                       # Store the weights in an object so they can't get compressed
                       # Using RichResult to allow dot notation, but a dict would work
                       diff_state=_RichResult(central=[], right=[], fac=None))

    # This is the correspondence between terms in the `work` object and the
    # final result. In this case, the mapping is trivial. Note that `success`
    # is prepended automatically.
    res_work_pairs = [('status', 'status'), ('df', 'df'), ('error', 'error'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('x', 'x')]

    def pre_func_eval(work):
        """Determine the abscissae at which the function needs to be evaluated.

        See `_derivative_weights` for a description of the stencil (pattern
        of the abscissae).

        In the first iteration, there is only one stored function value in
        `work.fs`, `f(x)`, so we need to evaluate at `order` new points. In
        subsequent iterations, we evaluate at two new points. Note that
        `work.x` is always flattened into a 1D array after broadcasting with
        all `args`, so we add a new axis at the end and evaluate all point
        in one call to the function.

        For improvement:
        - Consider measuring the step size actually taken, since ``(x + h) - x``
          is not identically equal to `h` with floating point arithmetic.
        - Adjust the step size automatically if `x` is too big to resolve the
          step.
        - We could probably save some work if there are no central difference
          steps or no one-sided steps.
        """
        n = work.terms  # half the order
        h = work.h[:, xp.newaxis]  # step size
        c = work.fac  # step reduction factor
        d = c**0.5  # square root of step reduction factor (one-sided stencil)
        # Note - no need to be careful about dtypes until we allocate `x_eval`

        if work.nit == 0:
            hc = h / c**xp.arange(n, dtype=work.dtype)
            hc = xp.concat((-xp.flip(hc, axis=-1), hc), axis=-1)
        else:
            hc = xp.concat((-h, h), axis=-1) / c**(n-1)

        if work.nit == 0:
            hr = h / d**xp.arange(2*n, dtype=work.dtype)
        else:
            hr = xp.concat((h, h/d), axis=-1) / c**(n-1)

        n_new = 2*n if work.nit == 0 else 2  # number of new abscissae
        x_eval = xp.zeros((work.hdir.shape[0], n_new), dtype=work.dtype)
        il, ic, ir = work.il, work.ic, work.ir
        x_eval[ir] = work.x[ir][:, xp.newaxis] + hr[ir]
        x_eval[ic] = work.x[ic][:, xp.newaxis] + hc[ic]
        x_eval[il] = work.x[il][:, xp.newaxis] - hr[il]
        return x_eval

    def post_func_eval(x, f, work):
        """ Estimate the derivative and error from the function evaluations

        As in `pre_func_eval`: in the first iteration, there is only one stored
        function value in `work.fs`, `f(x)`, so we need to add the `order` new
        points. In subsequent iterations, we add two new points. The tricky
        part is getting the order to match that of the weights, which is
        described in `_derivative_weights`.

        For improvement:
        - Change the order of the weights (and steps in `pre_func_eval`) to
          simplify `work_fc` concatenation and eliminate `fc` concatenation.
        - It would be simple to do one-step Richardson extrapolation with `df`
          and `df_last` to increase the order of the estimate and/or improve
          the error estimate.
        - Process the function evaluations in a more numerically favorable
          way. For instance, combining the pairs of central difference evals
          into a second-order approximation and using Richardson extrapolation
          to produce a higher order approximation seemed to retain accuracy up
          to very high order.
        - Alternatively, we could use `polyfit` like Jacobi. An advantage of
          fitting polynomial to more points than necessary is improved noise
          tolerance.
        """
        n = work.terms
        n_new = n if work.nit == 0 else 1
        il, ic, io = work.il, work.ic, work.io

        # Central difference
        # `work_fc` is *all* the points at which the function has been evaluated
        # `fc` is the points we're using *this iteration* to produce the estimate
        work_fc = (f[ic][:, :n_new], work.fs[ic], f[ic][:, -n_new:])
        work_fc = xp.concat(work_fc, axis=-1)
        if work.nit == 0:
            fc = work_fc
        else:
            fc = (work_fc[:, :n], work_fc[:, n:n+1], work_fc[:, -n:])
            fc = xp.concat(fc, axis=-1)

        # One-sided difference
        work_fo = xp.concat((work.fs[io], f[io]), axis=-1)
        if work.nit == 0:
            fo = work_fo
        else:
            fo = xp.concat((work_fo[:, 0:1], work_fo[:, -2*n:]), axis=-1)

        work.fs = xp.zeros((ic.shape[0], work.fs.shape[-1] + 2*n_new), dtype=work.dtype)
        work.fs[ic] = work_fc
        work.fs[io] = work_fo

        wc, wo = _derivative_weights(work, n, xp)
        work.df_last = xp.asarray(work.df, copy=True)
        work.df[ic] = fc @ wc / work.h[ic]
        work.df[io] = fo @ wo / work.h[io]
        work.df[il] *= -1

        work.h /= work.fac
        work.error_last = work.error
        # Simple error estimate - the difference in derivative estimates between
        # this iteration and the last. This is typically conservative because if
        # convergence has begin, the true error is much closer to the difference
        # between the current estimate and the *next* error estimate. However,
        # we could use Richarson extrapolation to produce an error estimate that
        # is one order higher, and take the difference between that and
        # `work.df` (which would just be constant factor that depends on `fac`.)
        work.error = xp.abs(work.df - work.df_last)

    def check_termination(work):
        """Terminate due to convergence, non-finite values, or error increase"""
        stop = xp.astype(xp.zeros_like(work.df), xp.bool)

        i = work.error < work.atol + work.rtol*abs(work.df)
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        if work.nit > 0:
            i = ~((xp.isfinite(work.x) & xp.isfinite(work.df)) | stop)
            work.df[i], work.status[i] = xp.nan, eim._EVALUEERR
            stop[i] = True

        # With infinite precision, there is a step size below which
        # all smaller step sizes will reduce the error. But in floating point
        # arithmetic, catastrophic cancellation will begin to cause the error
        # to increase again. This heuristic tries to avoid step sizes that are
        # too small. There may be more theoretically sound approaches for
        # detecting a step size that minimizes the total error, but this
        # heuristic seems simple and effective.
        i = (work.error > work.error_last*10) & ~stop
        work.status[i] = _EERRORINCREASE
        stop[i] = True

        return stop

    def post_termination_check(work):
        return

    def customize_result(res, shape):
        return shape

    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp, preserve_shape)


def _derivative_weights(work, n, xp):
    # This produces the weights of the finite difference formula for a given
    # stencil. In experiments, use of a second-order central difference formula
    # with Richardson extrapolation was more accurate numerically, but it was
    # more complicated, and it would have become even more complicated when
    # adding support for one-sided differences. However, now that all the
    # function evaluation values are stored, they can be processed in whatever
    # way is desired to produce the derivative estimate. We leave alternative
    # approaches to future work. To be more self-contained, here is the theory
    # for deriving the weights below.
    #
    # Recall that the Taylor expansion of a univariate, scalar-values function
    # about a point `x` may be expressed as:
    #      f(x + h)  =     f(x) + f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    # Suppose we evaluate f(x), f(x+h), and f(x-h).  We have:
    #      f(x)      =     f(x)
    #      f(x + h)  =     f(x) + f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    #      f(x - h)  =     f(x) - f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    # We can solve for weights `wi` such that:
    #   w1*f(x)      = w1*(f(x))
    # + w2*f(x + h)  = w2*(f(x) + f'(x)*h + f''(x)/2!*h**2) + O(h**3)
    # + w3*f(x - h)  = w3*(f(x) - f'(x)*h + f''(x)/2!*h**2) + O(h**3)
    #                =     0    + f'(x)*h + 0               + O(h**3)
    # Then
    #     f'(x) ~ (w1*f(x) + w2*f(x+h) + w3*f(x-h))/h
    # is a finite difference derivative approximation with error O(h**2),
    # and so it is said to be a "second-order" approximation. Under certain
    # conditions (e.g. well-behaved function, `h` sufficiently small), the
    # error in the approximation will decrease with h**2; that is, if `h` is
    # reduced by a factor of 2, the error is reduced by a factor of 4.
    #
    # By default, we use eighth-order formulae. Our central-difference formula
    # uses abscissae:
    #   x-h/c**3, x-h/c**2, x-h/c, x-h, x, x+h, x+h/c, x+h/c**2, x+h/c**3
    # where `c` is the step factor. (Typically, the step factor is greater than
    # one, so the outermost points - as written above - are actually closest to
    # `x`.) This "stencil" is chosen so that each iteration, the step can be
    # reduced by the factor `c`, and most of the function evaluations can be
    # reused with the new step size. For example, in the next iteration, we
    # will have:
    #   x-h/c**4, x-h/c**3, x-h/c**2, x-h/c, x, x+h/c, x+h/c**2, x+h/c**3, x+h/c**4
    # We do not reuse `x-h` and `x+h` for the new derivative estimate.
    # While this would increase the order of the formula and thus the
    # theoretical convergence rate, it is also less stable numerically.
    # (As noted above, there are other ways of processing the values that are
    # more stable. Thus, even now we store `f(x-h)` and `f(x+h)` in `work.fs`
    # to simplify future development of this sort of improvement.)
    #
    # The (right) one-sided formula is produced similarly using abscissae
    #   x, x+h, x+h/d, x+h/d**2, ..., x+h/d**6, x+h/d**7, x+h/d**7
    # where `d` is the square root of `c`. (The left one-sided formula simply
    # uses -h.) When the step size is reduced by factor `c = d**2`, we have
    # abscissae:
    #   x, x+h/d**2, x+h/d**3..., x+h/d**8, x+h/d**9, x+h/d**9
    # `d` is chosen as the square root of `c` so that the rate of the step-size
    # reduction is the same per iteration as in the central difference case.
    # Note that because the central difference formulas are inherently of even
    # order, for simplicity, we use only even-order formulas for one-sided
    # differences, too.

    # It's possible for the user to specify `fac` in, say, double precision but
    # `x` and `args` in single precision. `fac` gets converted to single
    # precision, but we should always use double precision for the intermediate
    # calculations here to avoid additional error in the weights.
    fac = float(work.fac)

    # Note that if the user switches back to floating point precision with
    # `x` and `args`, then `fac` will not necessarily equal the (lower
    # precision) cached `_derivative_weights.fac`, and the weights will
    # need to be recalculated. This could be fixed, but it's late, and of
    # low consequence.

    diff_state = work.diff_state
    if fac != diff_state.fac:
        diff_state.central = []
        diff_state.right = []
        diff_state.fac = fac

    if len(diff_state.central) != 2*n + 1:
        # Central difference weights. Consider refactoring this; it could
        # probably be more compact.
        # Note: Using NumPy here is OK; we convert to xp-type at the end
        i = np.arange(-n, n + 1)
        p = np.abs(i) - 1.  # center point has power `p` -1, but sign `s` is 0
        s = np.sign(i)

        h = s / fac ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2*n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        # Enforce identities to improve accuracy
        weights[n] = 0
        for i in range(n):
            weights[-i-1] = -weights[i]

        # Cache the weights. We only need to calculate them once unless
        # the step factor changes.
        diff_state.central = weights

        # One-sided difference weights. The left one-sided weights (with
        # negative steps) are simply the negative of the right one-sided
        # weights, so no need to compute them separately.
        i = np.arange(2*n + 1)
        p = i - 1.
        s = np.sign(i)

        h = s / np.sqrt(fac) ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        diff_state.right = weights

    return (xp.asarray(diff_state.central, dtype=work.dtype),
            xp.asarray(diff_state.right, dtype=work.dtype))


def jacobian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5,
             step_factor=2.0, step_direction=0):
    r"""Evaluate the Jacobian of a function numerically.

    Parameters
    ----------
    f : callable
        The function whose Jacobian is desired. The signature must be::

            f(xi: ndarray) -> ndarray

        where each element of ``xi`` is a finite real. If the function to be
        differentiated accepts additional arguments, wrap it (e.g. using
        `functools.partial` or ``lambda``) and pass the wrapped callable
        into `jacobian`. `f` must not mutate the array ``xi``. See Notes
        regarding vectorization and the dimensionality of the input and output.
    x : float array_like
        Points at which to evaluate the Jacobian. Must have at least one dimension.
        See Notes regarding the dimensionality and vectorization.
    tolerances : dictionary of floats, optional
        Absolute and relative tolerances. Valid keys of the dictionary are:

        - ``atol`` - absolute tolerance on the derivative
        - ``rtol`` - relative tolerance on the derivative

        Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
        `atol` is the smallest normal number of the appropriate dtype, and
        the default `rtol` is the square root of the precision of the
        appropriate dtype.
    maxiter : int, default: 10
        The maximum number of iterations of the algorithm to perform. See
        Notes.
    order : int, default: 8
        The (positive integer) order of the finite difference formula to be
        used. Odd integers will be rounded up to the next even integer.
    initial_step : float array_like, default: 0.5
        The (absolute) initial step size for the finite difference derivative
        approximation. Must be broadcastable with `x` and `step_direction`.
    step_factor : float, default: 2.0
        The factor by which the step size is *reduced* in each iteration; i.e.
        the step size in iteration 1 is ``initial_step/step_factor``. If
        ``step_factor < 1``, subsequent steps will be greater than the initial
        step; this may be useful if steps smaller than some threshold are
        undesirable (e.g. due to subtractive cancellation error).
    step_direction : integer array_like
        An array representing the direction of the finite difference steps (e.g.
        for use when `x` lies near to the boundary of the domain of the function.)
        Must be broadcastable with `x` and `initial_step`.
        Where 0 (default), central differences are used; where negative (e.g.
        -1), steps are non-positive; and where positive (e.g. 1), all steps are
        non-negative.

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
            - ``-1`` : The error estimate increased, so iteration was terminated.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.

        df : float array
            The Jacobian of `f` at `x`, if the algorithm terminated
            successfully.
        error : float array
            An estimate of the error: the magnitude of the difference between
            the current estimate of the Jacobian and the estimate in the
            previous iteration.
        nit : int array
            The number of iterations of the algorithm that were performed.
        nfev : int array
            The number of points at which `f` was evaluated.

        Each element of an attribute is associated with the corresponding
        element of `df`. For instance, element ``i`` of `nfev` is the
        number of points at which `f` was evaluated for the sake of
        computing element ``i`` of `df`.

    See Also
    --------
    derivative, hessian

    Notes
    -----
    Suppose we wish to evaluate the Jacobian of a function
    :math:`f: \mathbf{R}^m \rightarrow \mathbf{R}^n`. Assign to variables
    ``m`` and ``n`` the positive integer values of :math:`m` and :math:`n`,
    respectively, and let ``...`` represent an arbitrary tuple of integers.
    If we wish to evaluate the Jacobian at a single point, then:

    - argument `x` must be an array of shape ``(m,)``
    - argument `f` must be vectorized to accept an array of shape ``(m, ...)``.
      The first axis represents the :math:`m` inputs of :math:`f`; the remainder
      are for evaluating the function at multiple points in a single call.
    - argument `f` must return an array of shape ``(n, ...)``. The first
      axis represents the :math:`n` outputs of :math:`f`; the remainder
      are for the result of evaluating the function at multiple points.
    - attribute ``df`` of the result object will be an array of shape ``(n, m)``,
      the Jacobian.

    This function is also vectorized in the sense that the Jacobian can be
    evaluated at ``k`` points in a single call. In this case, `x` would be an
    array of shape ``(m, k)``, `f` would accept an array of shape
    ``(m, k, ...)`` and return an array of shape ``(n, k, ...)``, and the ``df``
    attribute of the result would have shape ``(n, m, k)``.

    Suppose the desired callable ``f_not_vectorized`` is not vectorized; it can
    only accept an array of shape ``(m,)``. A simple solution to satisfy the required
    interface is to wrap ``f_not_vectorized`` as follows::

        def f(x):
            return np.apply_along_axis(f_not_vectorized, axis=0, arr=x)

    Alternatively, suppose the desired callable ``f_vec_q`` is vectorized, but
    only for 2-D arrays of shape ``(m, q)``. To satisfy the required interface,
    consider::

        def f(x):
            m, batch = x.shape[0], x.shape[1:]  # x.shape is (m, ...)
            x = np.reshape(x, (m, -1))  # `-1` is short for q = prod(batch)
            res = f_vec_q(x)  # pass shape (m, q) to function
            n = res.shape[0]
            return np.reshape(res, (n,) + batch)  # return shape (n, ...)

    Then pass the wrapped callable ``f`` as the first argument of `jacobian`.

    References
    ----------
    .. [1] Jacobian matrix and determinant, *Wikipedia*,
           https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

    Examples
    --------
    The Rosenbrock function maps from :math:`\mathbf{R}^m \rightarrow \mathbf{R}`;
    the SciPy implementation `scipy.optimize.rosen` is vectorized to accept an
    array of shape ``(m, p)`` and return an array of shape ``p``. Suppose we wish
    to evaluate the Jacobian (AKA the gradient because the function returns a scalar)
    at ``[0.5, 0.5, 0.5]``.

    >>> import numpy as np
    >>> from scipy.differentiate import jacobian
    >>> from scipy.optimize import rosen, rosen_der
    >>> m = 3
    >>> x = np.full(m, 0.5)
    >>> res = jacobian(rosen, x)
    >>> ref = rosen_der(x)  # reference value of the gradient
    >>> res.df, ref
    (array([-51.,  -1.,  50.]), array([-51.,  -1.,  50.]))

    As an example of a function with multiple outputs, consider Example 4
    from [1]_.

    >>> def f(x):
    ...     x1, x2, x3 = x
    ...     return [x1, 5*x3, 4*x2**2 - 2*x3, x3*np.sin(x1)]

    The true Jacobian is given by:

    >>> def df(x):
    ...         x1, x2, x3 = x
    ...         one = np.ones_like(x1)
    ...         return [[one, 0*one, 0*one],
    ...                 [0*one, 0*one, 5*one],
    ...                 [0*one, 8*x2, -2*one],
    ...                 [x3*np.cos(x1), 0*one, np.sin(x1)]]

    Evaluate the Jacobian at an arbitrary point.

    >>> rng = np.random.default_rng(389252938452)
    >>> x = rng.random(size=3)
    >>> res = jacobian(f, x)
    >>> ref = df(x)
    >>> res.df.shape == (4, 3)
    True
    >>> np.allclose(res.df, ref)
    True

    Evaluate the Jacobian at 10 arbitrary points in a single call.

    >>> x = rng.random(size=(3, 10))
    >>> res = jacobian(f, x)
    >>> ref = df(x)
    >>> res.df.shape == (4, 3, 10)
    True
    >>> np.allclose(res.df, ref)
    True

    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    int_dtype = xp.isdtype(x.dtype, 'integral')
    x0 = xp.asarray(x, dtype=xp.asarray(1.0).dtype) if int_dtype else x

    if x0.ndim < 1:
        message = "Argument `x` must be at least 1-D."
        raise ValueError(message)

    m = x0.shape[0]
    i = xp.arange(m)

    def wrapped(x):
        p = () if x.ndim == x0.ndim else (x.shape[-1],)  # number of abscissae

        new_shape = (m, m) + x0.shape[1:] + p
        xph = xp.expand_dims(x0, axis=1)
        if x.ndim != x0.ndim:
            xph = xp.expand_dims(xph, axis=-1)
        xph = xp_copy(xp.broadcast_to(xph, new_shape), xp=xp)
        xph[i, i] = x
        return f(xph)

    res = derivative(wrapped, x, tolerances=tolerances,
                     maxiter=maxiter, order=order, initial_step=initial_step,
                     step_factor=step_factor, preserve_shape=True,
                     step_direction=step_direction)

    del res.x  # the user knows `x`, and the way it gets broadcasted is meaningless here
    return res


def hessian(f, x, *, tolerances=None, maxiter=10,
            order=8, initial_step=0.5, step_factor=2.0):
    r"""Evaluate the Hessian of a function numerically.

    Parameters
    ----------
    f : callable
        The function whose Hessian is desired. The signature must be::

            f(xi: ndarray) -> ndarray

        where each element of ``xi`` is a finite real. If the function to be
        differentiated accepts additional arguments, wrap it (e.g. using
        `functools.partial` or ``lambda``) and pass the wrapped callable
        into `hessian`. `f` must not mutate the array ``xi``. See Notes
        regarding vectorization and the dimensionality of the input and output.
    x : float array_like
        Points at which to evaluate the Hessian. Must have at least one dimension.
        See Notes regarding the dimensionality and vectorization.
    tolerances : dictionary of floats, optional
        Absolute and relative tolerances. Valid keys of the dictionary are:

        - ``atol`` - absolute tolerance on the derivative
        - ``rtol`` - relative tolerance on the derivative

        Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
        `atol` is the smallest normal number of the appropriate dtype, and
        the default `rtol` is the square root of the precision of the
        appropriate dtype.
    order : int, default: 8
        The (positive integer) order of the finite difference formula to be
        used. Odd integers will be rounded up to the next even integer.
    initial_step : float, default: 0.5
        The (absolute) initial step size for the finite difference derivative
        approximation.
    step_factor : float, default: 2.0
        The factor by which the step size is *reduced* in each iteration; i.e.
        the step size in iteration 1 is ``initial_step/step_factor``. If
        ``step_factor < 1``, subsequent steps will be greater than the initial
        step; this may be useful if steps smaller than some threshold are
        undesirable (e.g. due to subtractive cancellation error).
    maxiter : int, default: 10
        The maximum number of iterations of the algorithm to perform. See
        Notes.

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
            - ``-1`` : The error estimate increased, so iteration was terminated.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.

        ddf : float array
            The Hessian of `f` at `x`, if the algorithm terminated
            successfully.
        error : float array
            An estimate of the error: the magnitude of the difference between
            the current estimate of the Hessian and the estimate in the
            previous iteration.
        nfev : int array
            The number of points at which `f` was evaluated.

        Each element of an attribute is associated with the corresponding
        element of `ddf`. For instance, element ``[i, j]`` of `nfev` is the
        number of points at which `f` was evaluated for the sake of
        computing element ``[i, j]`` of `ddf`.

    See Also
    --------
    derivative, jacobian

    Notes
    -----
    Suppose we wish to evaluate the Hessian of a function
    :math:`f: \mathbf{R}^m \rightarrow \mathbf{R}`, and we assign to variable
    ``m`` the positive integer value of :math:`m`. If we wish to evaluate
    the Hessian at a single point, then:

    - argument `x` must be an array of shape ``(m,)``
    - argument `f` must be vectorized to accept an array of shape
      ``(m, ...)``. The first axis represents the :math:`m` inputs of
      :math:`f`; the remaining axes indicated by ellipses are for evaluating
      the function at several abscissae in a single call.
    - argument `f` must return an array of shape ``(...)``.
    - attribute ``dff`` of the result object will be an array of shape ``(m, m)``,
      the Hessian.

    This function is also vectorized in the sense that the Hessian can be
    evaluated at ``k`` points in a single call. In this case, `x` would be an
    array of shape ``(m, k)``, `f` would accept an array of shape
    ``(m, ...)`` and return an array of shape ``(...)``, and the ``ddf``
    attribute of the result would have shape ``(m, m, k)``. Note that the
    axis associated with the ``k`` points is included within the axes
    denoted by ``(...)``.

    Currently, `hessian` is implemented by nesting calls to `jacobian`.
    All options passed to `hessian` are used for both the inner and outer
    calls with one exception: the `rtol` used in the inner `jacobian` call
    is tightened by a factor of 100 with the expectation that the inner
    error can be ignored. A consequence is that `rtol` should not be set
    less than 100 times the precision of the dtype of `x`; a warning is
    emitted otherwise.

    References
    ----------
    .. [1] Hessian matrix, *Wikipedia*,
           https://en.wikipedia.org/wiki/Hessian_matrix

    Examples
    --------
    The Rosenbrock function maps from :math:`\mathbf{R}^m \rightarrow \mathbf{R}`;
    the SciPy implementation `scipy.optimize.rosen` is vectorized to accept an
    array of shape ``(m, ...)`` and return an array of shape ``...``. Suppose we
    wish to evaluate the Hessian at ``[0.5, 0.5, 0.5]``.

    >>> import numpy as np
    >>> from scipy.differentiate import hessian
    >>> from scipy.optimize import rosen, rosen_hess
    >>> m = 3
    >>> x = np.full(m, 0.5)
    >>> res = hessian(rosen, x)
    >>> ref = rosen_hess(x)  # reference value of the Hessian
    >>> np.allclose(res.ddf, ref)
    True

    `hessian` is vectorized to evaluate the Hessian at multiple points
    in a single call.

    >>> rng = np.random.default_rng(4589245925010)
    >>> x = rng.random((m, 10))
    >>> res = hessian(rosen, x)
    >>> ref = [rosen_hess(xi) for xi in x.T]
    >>> ref = np.moveaxis(ref, 0, -1)
    >>> np.allclose(res.ddf, ref)
    True

    """
    # todo:
    # - add ability to vectorize over additional parameters (*args?)
    # - error estimate stack with inner jacobian (or use legit 2D stencil)

    kwargs = dict(maxiter=maxiter, order=order, initial_step=initial_step,
                  step_factor=step_factor)
    tolerances = {} if tolerances is None else tolerances
    atol = tolerances.get('atol', None)
    rtol = tolerances.get('rtol', None)

    xp = array_namespace(x)
    x = xp.asarray(x)
    dtype = x.dtype if not xp.isdtype(x.dtype, 'integral') else xp.asarray(1.).dtype
    finfo = xp.finfo(dtype)
    rtol = finfo.eps**0.5 if rtol is None else rtol  # keep same as `derivative`

    # tighten the inner tolerance to make the inner error negligible
    rtol_min = finfo.eps * 100
    message = (f"The specified `{rtol=}`, but error estimates are likely to be "
               f"unreliable when `rtol < {rtol_min}`.")
    if 0 < rtol < rtol_min:  # rtol <= 0 is an error
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        rtol = rtol_min

    def df(x):
        tolerances = dict(rtol=rtol/100, atol=atol)
        temp = jacobian(f, x, tolerances=tolerances, **kwargs)
        nfev.append(temp.nfev if len(nfev) == 0 else temp.nfev.sum(axis=-1))
        return temp.df

    nfev = []  # track inner function evaluations
    res = jacobian(df, x, tolerances=tolerances, **kwargs)  # jacobian of jacobian

    nfev = xp.cumulative_sum(xp.stack(nfev), axis=0)
    res_nit = xp.astype(res.nit[xp.newaxis, ...], xp.int64)  # appease torch
    res.nfev = xp_take_along_axis(nfev, res_nit, axis=0)[0]
    res.ddf = res.df
    del res.df  # this is renamed to ddf
    del res.nit  # this is only the outer-jacobian nit

    return res
<|endoftext|>#
# Author: Pearu Peterson, March 2002
#
# additions by Travis Oliphant, March 2002
# additions by Eric Jones,      June 2002
# additions by Johannes Loehnert, June 2006
# additions by Bart Vandereycken, June 2006
# additions by Andrew D Straw, May 2007
# additions by Tiziano Zito, November 2008
#
# April 2010: Functions for LU, QR, SVD, Schur, and Cholesky decompositions
# were moved to their own files. Still in this file are functions for
# eigenstuff and for the Hessenberg form.

__all__ = ['eig', 'eigvals', 'eigh', 'eigvalsh',
           'eig_banded', 'eigvals_banded',
           'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf']

import numpy as np
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
                   flatnonzero, conj, asarray, argsort, empty,
                   iscomplex, zeros, einsum, eye, inf)
# Local imports
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork


_I = np.array(1j, dtype='F')


def _make_complex_eigvecs(w, vin, dtype):
    """
    Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output
    """
    # - see LAPACK man page DGGEV at ALPHAI
    v = np.array(vin, dtype=dtype)
    m = (w.imag > 0)
    m[:-1] |= (w.imag[1:] < 0)  # workaround for LAPACK bug, cf. ticket #709
    for i in flatnonzero(m):
        v.imag[:, i] = vin[:, i+1]
        conj(v[:, i], v[:, i+1])
    return v


def _make_eigvals(alpha, beta, homogeneous_eigvals):
    if homogeneous_eigvals:
        if beta is None:
            return np.vstack((alpha, np.ones_like(alpha)))
        else:
            return np.vstack((alpha, beta))
    else:
        if beta is None:
            return alpha
        else:
            w = np.empty_like(alpha)
            alpha_zero = (alpha == 0)
            beta_zero = (beta == 0)
            beta_nonzero = ~beta_zero
            w[beta_nonzero] = alpha[beta_nonzero]/beta[beta_nonzero]
            # Use np.inf for complex values too since
            # 1/np.inf = 0, i.e., it correctly behaves as projective
            # infinity.
            w[~alpha_zero & beta_zero] = np.inf
            if np.all(alpha.imag == 0):
                w[alpha_zero & beta_zero] = np.nan
            else:
                w[alpha_zero & beta_zero] = complex(np.nan, np.nan)
            return w


def _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
            homogeneous_eigvals):
    ggev, = get_lapack_funcs(('ggev',), (a1, b1))
    cvl, cvr = left, right
    res = ggev(a1, b1, lwork=-1)
    lwork = res[-2][0].real.astype(np.int_)
    if ggev.typecode in 'cz':
        alpha, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr, lwork,
                                               overwrite_a, overwrite_b)
        w = _make_eigvals(alpha, beta, homogeneous_eigvals)
    else:
        alphar, alphai, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr,
                                                        lwork, overwrite_a,
                                                        overwrite_b)
        alpha = alphar + _I * alphai
        w = _make_eigvals(alpha, beta, homogeneous_eigvals)
    _check_info(info, 'generalized eig algorithm (ggev)')

    only_real = np.all(w.imag == 0.0)
    if not (ggev.typecode in 'cz' or only_real):
        t = w.dtype.char
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        if right:
            vr = _make_complex_eigvecs(w, vr, t)

    # the eigenvectors returned by the lapack function are NOT normalized
    for i in range(vr.shape[0]):
        if right:
            vr[:, i] /= norm(vr[:, i])
        if left:
            vl[:, i] /= norm(vl[:, i])

    if not (left or right):
        return w
    if left:
        if right:
            return w, vl, vr
        return w, vl
    return w, vr


def eig(a, b=None, left=False, right=True, overwrite_a=False,
        overwrite_b=False, check_finite=True, homogeneous_eigvals=False):
    """
    Solve an ordinary or generalized eigenvalue problem of a square matrix.

    Find eigenvalues w and right or left eigenvectors of a general matrix::

        a   vr[:,i] = w[i]        b   vr[:,i]
        a.H vl[:,i] = w[i].conj() b.H vl[:,i]

    where ``.H`` is the Hermitian conjugation.

    Parameters
    ----------
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        Default is None, identity matrix is assumed.
    left : bool, optional
        Whether to calculate and return left eigenvectors.  Default is False.
    right : bool, optional
        Whether to calculate and return right eigenvectors.  Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.  Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `b`; may improve performance.  Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case ``w`` is a (2, M) array so that::

            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its
        multiplicity. The shape is (M,) unless
        ``homogeneous_eigvals=True``.
    vl : (M, M) double or complex ndarray
        The left eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column ``vl[:,i]``. Only returned if ``left=True``.
        The left eigenvector is not normalized.
    vr : (M, M) double or complex ndarray
        The normalized right eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of general arrays
    eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a)
    array([0.+1.j, 0.-1.j])

    >>> b = np.array([[0., 1.], [1., 1.]])
    >>> linalg.eigvals(a, b)
    array([ 1.+0.j, -1.+0.j])

    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
    >>> linalg.eigvals(a, homogeneous_eigvals=True)
    array([[3.+0.j, 8.+0.j, 7.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j]])

    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a) == linalg.eig(a)[0]
    array([ True,  True])
    >>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector
    array([[-0.70710678+0.j        , -0.70710678-0.j        ],
           [-0.        +0.70710678j, -0.        -0.70710678j]])
    >>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])



    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    # accommodate square empty matrices
    if a1.size == 0:
        w_n, vr_n = eig(np.eye(2, dtype=a1.dtype))
        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        w = _make_eigvals(w, None, homogeneous_eigvals)
        vl = np.empty_like(a1, shape=(0, 0), dtype=vr_n.dtype)
        vr = np.empty_like(a1, shape=(0, 0), dtype=vr_n.dtype)
        if not (left or right):
            return w
        if left:
            if right:
                return w, vl, vr
            return w, vl
        return w, vr

    overwrite_a = overwrite_a or (_datacopied(a1, a))
    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square matrix')
        if b1.shape != a1.shape:
            raise ValueError('a and b must have the same shape')
        return _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
                       homogeneous_eigvals)

    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a1,))
    compute_vl, compute_vr = left, right

    lwork = _compute_lwork(geev_lwork, a1.shape[0],
                           compute_vl=compute_vl,
                           compute_vr=compute_vr)

    if geev.typecode in 'cz':
        w, vl, vr, info = geev(a1, lwork=lwork,
                               compute_vl=compute_vl,
                               compute_vr=compute_vr,
                               overwrite_a=overwrite_a)
        w = _make_eigvals(w, None, homogeneous_eigvals)
    else:
        wr, wi, vl, vr, info = geev(a1, lwork=lwork,
                                    compute_vl=compute_vl,
                                    compute_vr=compute_vr,
                                    overwrite_a=overwrite_a)
        w = wr + _I * wi
        w = _make_eigvals(w, None, homogeneous_eigvals)

    _check_info(info, 'eig algorithm (geev)',
                positive='did not converge (only eigenvalues '
                         'with order >= %d have converged)')

    only_real = np.all(w.imag == 0.0)
    if not (geev.typecode in 'cz' or only_real):
        t = w.dtype.char
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        if right:
            vr = _make_complex_eigvecs(w, vr, t)
    if not (left or right):
        return w
    if left:
        if right:
            return w, vl, vr
        return w, vl
    return w, vr


def eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, type=1, check_finite=True, subset_by_index=None,
         subset_by_value=None, driver=None):
    """
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues array ``w`` and optionally eigenvectors array ``v`` of
    array ``a``, where ``b`` is positive definite such that for every
    eigenvalue λ (i-th entry of w) and its eigenvector ``vi`` (i-th column of
    ``v``) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, ``b`` is assumed to be the identity matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric definite positive matrix in.
        If omitted, identity matrix is assumed.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of ``a`` and, if applicable, ``b``. (Default: lower)
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    subset_by_index : iterable, optional
        If provided, this two-element iterable defines the start and the end
        indices of the desired eigenvalues (ascending order and 0-indexed).
        To return only the second smallest to fifth smallest eigenvalues,
        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only
        available with "evr", "evx", and "gvx" drivers. The entries are
        directly converted to integers via ``int()``.
    subset_by_value : iterable, optional
        If provided, this two-element iterable defines the half-open interval
        ``(a, b]`` that, if any, only the eigenvalues between these values
        are returned. Only available with "evr", "evx", and "gvx" drivers. Use
        ``np.inf`` for the unconstrained ends.
    driver : str, optional
        Defines which LAPACK driver should be used. Valid options are "ev",
        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for
        generalized (where b is not None) problems. See the Notes section.
        The default for standard problems is "evr". For generalized problems,
        "gvd" is used for full set, and "gvx" for subset requested cases.
    type : int, optional
        For the generalized problems, this keyword specifies the problem type
        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
        inputs)::

            1 =>     a @ v = w @ b @ v
            2 => a @ b @ v = w @ v
            3 => b @ a @ v = w @ v

        This keyword is ignored for standard problems.
    overwrite_a : bool, optional
        Whether to overwrite data in ``a`` (may improve performance). Default
        is False.
    overwrite_b : bool, optional
        Whether to overwrite data in ``b`` (may improve performance). Default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    w : (N,) ndarray
        The N (N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.
    v : (M, N) ndarray
        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
        the column ``v[:,i]``. Only returned if ``eigvals_only=False``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge, an error occurred, or
        b matrix is not definite positive. Note that if input matrices are
        not symmetric or Hermitian, no error will be reported but results will
        be wrong.

    See Also
    --------
    eigvalsh : eigenvalues of symmetric or Hermitian arrays
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Notes
    -----
    This function does not check the input array for being Hermitian/symmetric
    in order to allow for representing arrays with only their upper/lower
    triangular parts. Also, note that even though not taken into account,
    finiteness check applies to the whole array and unaffected by "lower"
    keyword.

    This function uses LAPACK drivers for computations in all possible keyword
    combinations, prefixed with ``sy`` if arrays are real and ``he`` if
    complex, e.g., a float array with "evr" driver is solved via
    "syevr", complex arrays with "gvx" driver problem is solved via "hegvx"
    etc.

    As a brief summary, the slowest and the most robust driver is the
    classical ``<sy/he>ev`` which uses symmetric QR. ``<sy/he>evr`` is seen as
    the optimal choice for the most general cases. However, there are certain
    occasions that ``<sy/he>evd`` computes faster at the expense of more
    memory usage. ``<sy/he>evx``, while still being faster than ``<sy/he>ev``,
    often performs worse than the rest except when very few eigenvalues are
    requested for large arrays though there is still no performance guarantee.

    Note that the underlying LAPACK algorithms are different depending on whether
    `eigvals_only` is True or False --- thus the eigenvalues may differ
    depending on whether eigenvectors are requested or not. The difference is
    generally of the order of machine epsilon times the largest eigenvalue,
    so is likely only visible for zero or nearly zero eigenvalues.

    For the generalized problem, normalization with respect to the given
    type argument::

            type 1 and 3 :      v.conj().T @ a @ v = w
            type 2       : inv(v).conj().T @ a @ inv(v) = w

            type 1 or 2  :      v.conj().T @ b @ v  = I
            type 3       : v.conj().T @ inv(b) @ v  = I


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigh
    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
    >>> w, v = eigh(A)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True

    Request only the eigenvalues

    >>> w = eigh(A, eigvals_only=True)

    Request eigenvalues that are less than 10.

    >>> A = np.array([[34, -4, -10, -7, 2],
    ...               [-4, 7, 2, 12, 0],
    ...               [-10, 2, 44, 2, -19],
    ...               [-7, 12, 2, 79, -34],
    ...               [2, 0, -19, -34, 29]])
    >>> eigh(A, eigvals_only=True, subset_by_value=[-np.inf, 10])
    array([6.69199443e-07, 9.11938152e+00])

    Request the second smallest eigenvalue and its eigenvector

    >>> w, v = eigh(A, subset_by_index=[1, 1])
    >>> w
    array([9.11938152])
    >>> v.shape  # only a single column is returned
    (5, 1)

    """
    # set lower
    uplo = 'L' if lower else 'U'
    # Set job for Fortran routines
    _job = 'N' if eigvals_only else 'V'

    drv_str = [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]
    if driver not in drv_str:
        raise ValueError('"{}" is unknown. Possible values are "None", "{}".'
                         ''.format(driver, '", "'.join(drv_str[1:])))

    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')

    # accommodate square empty matrices
    if a1.size == 0:
        w_n, v_n = eigh(np.eye(2, dtype=a1.dtype))

        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        v = np.empty_like(a1, shape=(0, 0), dtype=v_n.dtype)
        if eigvals_only:
            return w
        else:
            return w, v

    overwrite_a = overwrite_a or (_datacopied(a1, a))
    cplx = True if iscomplexobj(a1) else False
    n = a1.shape[0]
    drv_args = {'overwrite_a': overwrite_a}

    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')

        if b1.shape != a1.shape:
            raise ValueError(f"wrong b dimensions {b1.shape}, should be {a1.shape}")

        if type not in [1, 2, 3]:
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')

        cplx = True if iscomplexobj(b1) else (cplx or False)
        drv_args.update({'overwrite_b': overwrite_b, 'itype': type})

    subset = (subset_by_index is not None) or (subset_by_value is not None)

    # Both subsets can't be given
    if subset_by_index and subset_by_value:
        raise ValueError('Either index or value subset can be requested.')

    # Check indices if given
    if subset_by_index:
        lo, hi = (int(x) for x in subset_by_index)
        if not (0 <= lo <= hi < n):
            raise ValueError('Requested eigenvalue indices are not valid. '
                             f'Valid range is [0, {n-1}] and start <= end, but '
                             f'start={lo}, end={hi} is given')
        # fortran is 1-indexed
        drv_args.update({'range': 'I', 'il': lo + 1, 'iu': hi + 1})

    if subset_by_value:
        lo, hi = subset_by_value
        if not (-inf <= lo < hi <= inf):
            raise ValueError('Requested eigenvalue bounds are not valid. '
                             'Valid range is (-inf, inf) and low < high, but '
                             f'low={lo}, high={hi} is given')

        drv_args.update({'range': 'V', 'vl': lo, 'vu': hi})

    # fix prefix for lapack routines
    pfx = 'he' if cplx else 'sy'

    # decide on the driver if not given
    # first early exit on incompatible choice
    if driver:
        if b is None and (driver in ["gv", "gvd", "gvx"]):
            raise ValueError(f'{driver} requires input b array to be supplied '
                             'for generalized eigenvalue problems.')
        if (b is not None) and (driver in ['ev', 'evd', 'evr', 'evx']):
            raise ValueError(f'"{driver}" does not accept input b array '
                             'for standard eigenvalue problems.')
        if subset and (driver in ["ev", "evd", "gv", "gvd"]):
            raise ValueError(f'"{driver}" cannot compute subsets of eigenvalues')

    # Default driver is evr and gvd
    else:
        driver = "evr" if b is None else ("gvx" if subset else "gvd")

    lwork_spec = {
                  'syevd': ['lwork', 'liwork'],
                  'syevr': ['lwork', 'liwork'],
                  'heevd': ['lwork', 'liwork', 'lrwork'],
                  'heevr': ['lwork', 'lrwork', 'liwork'],
                  }

    if b is None:  # Standard problem
        drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                      [a1])
        clw_args = {'n': n, 'lower': lower}
        if driver == 'evd':
            clw_args.update({'compute_v': 0 if _job == "N" else 1})

        lw = _compute_lwork(drvlw, **clw_args)
        # Multiple lwork vars
        if isinstance(lw, tuple):
            lwork_args = dict(zip(lwork_spec[pfx+driver], lw))
        else:
            lwork_args = {'lwork': lw}

        drv_args.update({'lower': lower, 'compute_v': 0 if _job == "N" else 1})
        w, v, *other_args, info = drv(a=a1, **drv_args, **lwork_args)

    else:  # Generalized problem
        # 'gvd' doesn't have lwork query
        if driver == "gvd":
            drv = get_lapack_funcs(pfx + "gvd", [a1, b1])
            lwork_args = {}
        else:
            drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                          [a1, b1])
            # generalized drivers use uplo instead of lower
            lw = _compute_lwork(drvlw, n, uplo=uplo)
            lwork_args = {'lwork': lw}

        drv_args.update({'uplo': uplo, 'jobz': _job})

        w, v, *other_args, info = drv(a=a1, b=b1, **drv_args, **lwork_args)

    # m is always the first extra argument
    w = w[:other_args[0]] if subset else w
    v = v[:, :other_args[0]] if (subset and not eigvals_only) else v

    # Check if we had a  successful exit
    if info == 0:
        if eigvals_only:
            return w
        else:
            return w, v
    else:
        if info < -1:
            raise LinAlgError(f'Illegal value in argument {-info} of internal '
                              f'{drv.typecode + pfx + driver}')
        elif info > n:
            raise LinAlgError(f'The leading minor of order {info-n} of B is not '
                              'positive definite. The factorization of B '
                              'could not be completed and no eigenvalues '
                              'or eigenvectors were computed.')
        else:
            drv_err = {'ev': 'The algorithm failed to converge; {} '
                             'off-diagonal elements of an intermediate '
                             'tridiagonal form did not converge to zero.',
                       'evx': '{} eigenvectors failed to converge.',
                       'evd': 'The algorithm failed to compute an eigenvalue '
                              'while working on the submatrix lying in rows '
                              'and columns {0}/{1} through mod({0},{1}).',
                       'evr': 'Internal Error.'
                       }
            if driver in ['ev', 'gv']:
                msg = drv_err['ev'].format(info)
            elif driver in ['evx', 'gvx']:
                msg = drv_err['evx'].format(info)
            elif driver in ['evd', 'gvd']:
                if eigvals_only:
                    msg = drv_err['ev'].format(info)
                else:
                    msg = drv_err['evd'].format(info, n+1)
            else:
                msg = drv_err['evr']

            raise LinAlgError(msg)


_conv_dict = {0: 0, 1: 1, 2: 2,
              'all': 0, 'value': 1, 'index': 2,
              'a': 0, 'v': 1, 'i': 2}


def _check_select(select, select_range, max_ev, max_len):
    """Check that select is valid, convert to Fortran style."""
    if isinstance(select, str):
        select = select.lower()
    try:
        select = _conv_dict[select]
    except KeyError as e:
        raise ValueError('invalid argument for select') from e
    vl, vu = 0., 1.
    il = iu = 1
    if select != 0:  # (non-all)
        sr = asarray(select_range)
        if sr.ndim != 1 or sr.size != 2 or sr[1] < sr[0]:
            raise ValueError('select_range must be a 2-element array-like '
                             'in nondecreasing order')
        if select == 1:  # (value)
            vl, vu = sr
            if max_ev == 0:
                max_ev = max_len
        else:  # 2 (index)
            if sr.dtype.char.lower() not in 'hilqp':
                raise ValueError(
                    f'when using select="i", select_range must '
                    f'contain integers, got dtype {sr.dtype} ({sr.dtype.char})'
                )
            # translate Python (0 ... N-1) into Fortran (1 ... N) with + 1
            il, iu = sr + 1
            if min(il, iu) < 1 or max(il, iu) > max_len:
                raise ValueError('select_range out of bounds')
            max_ev = iu - il + 1
    return select, vl, vu, il, iu, max_ev


def eig_banded(a_band, lower=False, eigvals_only=False, overwrite_a_band=False,
               select='a', select_range=None, max_ev=0, check_finite=True):
    """
    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

    Find eigenvalues w and optionally right eigenvectors v of a::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    The matrix a is stored in a_band either in lower diagonal or upper
    diagonal ordered form:

        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

    where u is the number of bands above the diagonal.

    Example of a_band (shape of a is (6,6), u=2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Cells marked with * are not used.

    Parameters
    ----------
    a_band : (u+1, M) array_like
        The bands of the M by M matrix a.
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    eigvals_only : bool, optional
        Compute only the eigenvalues and no eigenvectors.
        (Default: calculate also eigenvectors)
    overwrite_a_band : bool, optional
        Discard data in a_band (may enhance performance)
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    max_ev : int, optional
        For select=='v', maximum number of eigenvalues expected.
        For other values of select, has no meaning.

        In doubt, leave this parameter untouched.

    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.
    v : (M, M) float or complex ndarray
        The normalized eigenvector corresponding to the eigenvalue w[i] is
        the column v[:,i]. Only returned if ``eigvals_only=False``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
    eig : eigenvalues and right eigenvectors of general arrays.
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eigh_tridiagonal : eigenvalues and right eigenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eig_banded
    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
    >>> w, v = eig_banded(Ab, lower=True)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True
    >>> w = eig_banded(Ab, lower=True, eigvals_only=True)
    >>> w
    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])

    Request only the eigenvalues between ``[-3, 4]``

    >>> w, v = eig_banded(Ab, lower=True, select='v', select_range=[-3, 4])
    >>> w
    array([-2.22987175,  3.95222349])

    """
    if eigvals_only or overwrite_a_band:
        a1 = _asarray_validated(a_band, check_finite=check_finite)
        overwrite_a_band = overwrite_a_band or (_datacopied(a1, a_band))
    else:
        a1 = array(a_band)
        if issubclass(a1.dtype.type, inexact) and not isfinite(a1).all():
            raise ValueError("array must not contain infs or NaNs")
        overwrite_a_band = 1

    if len(a1.shape) != 2:
        raise ValueError('expected a 2-D array')

    # accommodate square empty matrices
    if a1.size == 0:
        w_n, v_n = eig_banded(np.array([[0, 0], [1, 1]], dtype=a1.dtype))

        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        v = np.empty_like(a1, shape=(0, 0), dtype=v_n.dtype)
        if eigvals_only:
            return w
        else:
            return w, v

    select, vl, vu, il, iu, max_ev = _check_select(
        select, select_range, max_ev, a1.shape[1])

    del select_range
    if select == 0:
        if a1.dtype.char in 'GFD':
            # FIXME: implement this somewhen, for now go with builtin values
            # FIXME: calc optimal lwork by calling ?hbevd(lwork=-1)
            #        or by using calc_lwork.f ???
            # lwork = calc_lwork.hbevd(bevd.typecode, a1.shape[0], lower)
            internal_name = 'hbevd'
        else:  # a1.dtype.char in 'fd':
            # FIXME: implement this somewhen, for now go with builtin values
            #         see above
            # lwork = calc_lwork.sbevd(bevd.typecode, a1.shape[0], lower)
            internal_name = 'sbevd'
        bevd, = get_lapack_funcs((internal_name,), (a1,))
        w, v, info = bevd(a1, compute_v=not eigvals_only,
                          lower=lower, overwrite_ab=overwrite_a_band)
    else:  # select in [1, 2]
        if eigvals_only:
            max_ev = 1
        # calculate optimal abstol for dsbevx (see manpage)
        if a1.dtype.char in 'fF':  # single precision
            lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='f'),))
        else:
            lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='d'),))
        abstol = 2 * lamch('s')
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevx'
        else:  # a1.dtype.char in 'gfd'
            internal_name = 'sbevx'
        bevx, = get_lapack_funcs((internal_name,), (a1,))
        w, v, m, ifail, info = bevx(
            a1, vl, vu, il, iu, compute_v=not eigvals_only, mmax=max_ev,
            range=select, lower=lower, overwrite_ab=overwrite_a_band,
            abstol=abstol)
        # crop off w and v
        w = w[:m]
        if not eigvals_only:
            v = v[:, :m]
    _check_info(info, internal_name)

    if eigvals_only:
        return w
    return w, v


def eigvals(a, b=None, overwrite_a=False, check_finite=True,
            homogeneous_eigvals=False):
    """
    Compute eigenvalues from an ordinary or generalized eigenvalue problem.

    Find eigenvalues of a general matrix::

        a   vr[:,i] = w[i]        b   vr[:,i]

    Parameters
    ----------
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        If omitted, identity matrix is assumed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities
        or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case ``w`` is a (2, M) array so that::

            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its multiplicity
        but not in any specific order. The shape is (M,) unless
        ``homogeneous_eigvals=True``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays.
    eigvalsh : eigenvalues of symmetric or Hermitian arrays
    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a)
    array([0.+1.j, 0.-1.j])

    >>> b = np.array([[0., 1.], [1., 1.]])
    >>> linalg.eigvals(a, b)
    array([ 1.+0.j, -1.+0.j])

    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
    >>> linalg.eigvals(a, homogeneous_eigvals=True)
    array([[3.+0.j, 8.+0.j, 7.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j]])

    """
    return eig(a, b=b, left=0, right=0, overwrite_a=overwrite_a,
               check_finite=check_finite,
               homogeneous_eigvals=homogeneous_eigvals)


def eigvalsh(a, b=None, *, lower=True, overwrite_a=False,
             overwrite_b=False, type=1, check_finite=True, subset_by_index=None,
             subset_by_value=None, driver=None):
    """
    Solves a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues array ``w`` of array ``a``, where ``b`` is positive
    definite such that for every eigenvalue λ (i-th entry of w) and its
    eigenvector vi (i-th column of v) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, b is assumed to be the identity matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues will
        be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric definite positive matrix in.
        If omitted, identity matrix is assumed.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of ``a`` and, if applicable, ``b``. (Default: lower)
    overwrite_a : bool, optional
        Whether to overwrite data in ``a`` (may improve performance). Default
        is False.
    overwrite_b : bool, optional
        Whether to overwrite data in ``b`` (may improve performance). Default
        is False.
    type : int, optional
        For the generalized problems, this keyword specifies the problem type
        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
        inputs)::

            1 =>     a @ v = w @ b @ v
            2 => a @ b @ v = w @ v
            3 => b @ a @ v = w @ v

        This keyword is ignored for standard problems.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    subset_by_index : iterable, optional
        If provided, this two-element iterable defines the start and the end
        indices of the desired eigenvalues (ascending order and 0-indexed).
        To return only the second smallest to fifth smallest eigenvalues,
        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only
        available with "evr", "evx", and "gvx" drivers. The entries are
        directly converted to integers via ``int()``.
    subset_by_value : iterable, optional
        If provided, this two-element iterable defines the half-open interval
        ``(a, b]`` that, if any, only the eigenvalues between these values
        are returned. Only available with "evr", "evx", and "gvx" drivers. Use
        ``np.inf`` for the unconstrained ends.
    driver : str, optional
        Defines which LAPACK driver should be used. Valid options are "ev",
        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for
        generalized (where b is not None) problems. See the Notes section of
        `scipy.linalg.eigh`.

    Returns
    -------
    w : (N,) ndarray
        The N (N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge, an error occurred, or
        b matrix is not definite positive. Note that if input matrices are
        not symmetric or Hermitian, no error will be reported but results will
        be wrong.

    See Also
    --------
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eigvals : eigenvalues of general arrays
    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices

    Notes
    -----
    This function does not check the input array for being Hermitian/symmetric
    in order to allow for representing arrays with only their upper/lower
    triangular parts.

    This function serves as a one-liner shorthand for `scipy.linalg.eigh` with
    the option ``eigvals_only=True`` to get the eigenvalues and not the
    eigenvectors. Here it is kept as a legacy convenience. It might be
    beneficial to use the main function to have full control and to be a bit
    more pythonic.

    Examples
    --------
    For more examples see `scipy.linalg.eigh`.

    >>> import numpy as np
    >>> from scipy.linalg import eigvalsh
    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
    >>> w = eigvalsh(A)
    >>> w
    array([-3.74637491, -0.76263923,  6.08502336, 12.42399079])

    """
    return eigh(a, b=b, lower=lower, eigvals_only=True, overwrite_a=overwrite_a,
                overwrite_b=overwrite_b, type=type, check_finite=check_finite,
                subset_by_index=subset_by_index, subset_by_value=subset_by_value,
                driver=driver)


def eigvals_banded(a_band, lower=False, overwrite_a_band=False,
                   select='a', select_range=None, check_finite=True):
    """
    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

    Find eigenvalues w of a::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    The matrix a is stored in a_band either in lower diagonal or upper
    diagonal ordered form:

        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

    where u is the number of bands above the diagonal.

    Example of a_band (shape of a is (6,6), u=2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Cells marked with * are not used.

    Parameters
    ----------
    a_band : (u+1, M) array_like
        The bands of the M by M matrix a.
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    overwrite_a_band : bool, optional
        Discard data in a_band (may enhance performance)
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices
    eigvals : eigenvalues of general arrays
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eig : eigenvalues and right eigenvectors for non-symmetric arrays

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigvals_banded
    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
    >>> w = eigvals_banded(Ab, lower=True)
    >>> w
    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])
    """
    return eig_banded(a_band, lower=lower, eigvals_only=1,
                      overwrite_a_band=overwrite_a_band, select=select,
                      select_range=select_range, check_finite=check_finite)


def eigvalsh_tridiagonal(d, e, select='a', select_range=None,
                         check_finite=True, tol=0., lapack_driver='auto'):
    """
    Solve eigenvalue problem for a real symmetric tridiagonal matrix.

    Find eigenvalues `w` of ``a``::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    For a real symmetric matrix ``a`` with diagonal elements `d` and
    off-diagonal elements `e`.

    Parameters
    ----------
    d : ndarray, shape (ndim,)
        The diagonal elements of the array.
    e : ndarray, shape (ndim-1,)
        The off-diagonal elements of the array.
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each eigenvalue is required
        (only used when ``lapack_driver='stebz'``).
        An eigenvalue (or cluster) is considered to have converged if it
        lies in an interval of this width. If <= 0. (default),
        the value ``eps*|a|`` is used where eps is the machine precision,
        and ``|a|`` is the 1-norm of the matrix ``a``.
    lapack_driver : str
        LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
        and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
        ``select='a'``.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
    >>> d = 3*np.ones(4)
    >>> e = -1*np.ones(3)
    >>> w = eigvalsh_tridiagonal(d, e)
    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    >>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines
    >>> np.allclose(w - w2, np.zeros(4))
    True
    """
    return eigh_tridiagonal(
        d, e, eigvals_only=True, select=select, select_range=select_range,
        check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)


def eigh_tridiagonal(d, e, eigvals_only=False, select='a', select_range=None,
                     check_finite=True, tol=0., lapack_driver='auto'):
    """
    Solve eigenvalue problem for a real symmetric tridiagonal matrix.

    Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    For a real symmetric matrix ``a`` with diagonal elements `d` and
    off-diagonal elements `e`.

    Parameters
    ----------
    d : ndarray, shape (ndim,)
        The diagonal elements of the array.
    e : ndarray, shape (ndim-1,)
        The off-diagonal elements of the array.
    eigvals_only : bool, optional
        Compute only the eigenvalues and no eigenvectors.
        (Default: calculate also eigenvectors)
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each eigenvalue is required
        (only used when 'stebz' is the `lapack_driver`).
        An eigenvalue (or cluster) is considered to have converged if it
        lies in an interval of this width. If <= 0. (default),
        the value ``eps*|a|`` is used where eps is the machine precision,
        and ``|a|`` is the 1-norm of the matrix ``a``.
    lapack_driver : str
        LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',
        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
        and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and
        ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is
        used to find the corresponding eigenvectors. 'sterf' can only be
        used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only
        be used when ``select='a'``.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.
    v : (M, M) ndarray
        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
        the column ``v[:,i]``. Only returned if ``eigvals_only=False``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
        matrices
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices

    Notes
    -----
    This function makes use of LAPACK ``S/DSTEMR`` routines.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigh_tridiagonal
    >>> d = 3*np.ones(4)
    >>> e = -1*np.ones(3)
    >>> w, v = eigh_tridiagonal(d, e)
    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True
    """
    d = _asarray_validated(d, check_finite=check_finite)
    e = _asarray_validated(e, check_finite=check_finite)
    for check in (d, e):
        if check.ndim != 1:
            raise ValueError('expected a 1-D array')
        if check.dtype.char in 'GFD':  # complex
            raise TypeError('Only real arrays currently supported')
    if d.size != e.size + 1:
        raise ValueError(f'd ({d.size}) must have one more element than e ({e.size})')
    select, vl, vu, il, iu, _ = _check_select(
        select, select_range, 0, d.size)
    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be str')
    drivers = ('auto', 'stemr', 'sterf', 'stebz', 'stev')
    if lapack_driver not in drivers:
        raise ValueError(f'lapack_driver must be one of {drivers}, '
                         f'got {lapack_driver}')
    if lapack_driver == 'auto':
        lapack_driver = 'stemr' if select == 0 else 'stebz'

    # Quick exit for 1x1 case
    if len(d) == 1:
        if select == 1 and (not (vl < d[0] <= vu)):  # request by value
            w = array([])
            v = empty([1, 0], dtype=d.dtype)
        else:  # all and request by index
            w = array([d[0]], dtype=d.dtype)
            v = array([[1.]], dtype=d.dtype)

        if eigvals_only:
            return w
        else:
            return w, v

    func, = get_lapack_funcs((lapack_driver,), (d, e))
    compute_v = not eigvals_only
    if lapack_driver == 'sterf':
        if select != 0:
            raise ValueError('sterf can only be used when select == "a"')
        if not eigvals_only:
            raise ValueError('sterf can only be used when eigvals_only is '
                             'True')
        w, info = func(d, e)
        m = len(w)
    elif lapack_driver == 'stev':
        if select != 0:
            raise ValueError('stev can only be used when select == "a"')
        w, v, info = func(d, e, compute_v=compute_v)
        m = len(w)
    elif lapack_driver == 'stebz':
        tol = float(tol)
        internal_name = 'stebz'
        stebz, = get_lapack_funcs((internal_name,), (d, e))
        # If getting eigenvectors, needs to be block-ordered (B) instead of
        # matrix-ordered (E), and we will reorder later
        order = 'E' if eigvals_only else 'B'
        m, w, iblock, isplit, info = stebz(d, e, select, vl, vu, il, iu, tol,
                                           order)
    else:   # 'stemr'
        # ?STEMR annoyingly requires size N instead of N-1
        e_ = empty(e.size+1, e.dtype)
        e_[:-1] = e
        stemr_lwork, = get_lapack_funcs(('stemr_lwork',), (d, e))
        lwork, liwork, info = stemr_lwork(d, e_, select, vl, vu, il, iu,
                                          compute_v=compute_v)
        _check_info(info, 'stemr_lwork')
        m, w, v, info = func(d, e_, select, vl, vu, il, iu,
                             compute_v=compute_v, lwork=lwork, liwork=liwork)
    _check_info(info, lapack_driver + ' (eigh_tridiagonal)')
    w = w[:m]
    if eigvals_only:
        return w
    else:
        # Do we still need to compute the eigenvalues?
        if lapack_driver == 'stebz':
            func, = get_lapack_funcs(('stein',), (d, e))
            v, info = func(d, e, w, iblock, isplit)
            _check_info(info, 'stein (eigh_tridiagonal)',
                        positive='%d eigenvectors failed to converge')
            # Convert block-order to matrix-order
            order = argsort(w)
            w, v = w[order], v[:, order]
        else:
            v = v[:, :m]
        return w, v


def _check_info(info, driver, positive='did not converge (LAPACK info=%d)'):
    """Check info return value."""
    if info < 0:
        raise ValueError('illegal value in argument %d of internal %s'
                         % (-info, driver))
    if info > 0 and positive:
        raise LinAlgError(("%s " + positive) % (driver, info,))


def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    """
    Compute Hessenberg form of a matrix.

    The Hessenberg decomposition is::

        A = Q H Q^H

    where `Q` is unitary/orthogonal and `H` has only zero elements below
    the first sub-diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to bring into Hessenberg form.
    calc_q : bool, optional
        Whether to compute the transformation matrix.  Default is False.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    H : (M, M) ndarray
        Hessenberg form of `a`.
    Q : (M, M) ndarray
        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
        Only returned if ``calc_q=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import hessenberg
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> H, Q = hessenberg(A, calc_q=True)
    >>> H
    array([[  2.        , -11.65843866,   1.42005301,   0.25349066],
           [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],
           [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],
           [  0.        ,   0.        ,  -3.83189513,   1.07494686]])
    >>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))
    True
    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if a1.size == 0:
        h3 = hessenberg(np.eye(3, dtype=a1.dtype))
        h = np.empty(a1.shape, dtype=h3.dtype)
        if not calc_q:
            return h
        else:
            h3, q3 = hessenberg(np.eye(3, dtype=a1.dtype), calc_q=True)
            q = np.empty(a1.shape, dtype=q3.dtype)
            h = np.empty(a1.shape, dtype=h3.dtype)
            return h, q

    # if 2x2 or smaller: already in Hessenberg
    if a1.shape[0] <= 2:
        if calc_q:
            return a1, eye(a1.shape[0])
        return a1

    gehrd, gebal, gehrd_lwork = get_lapack_funcs(('gehrd', 'gebal',
                                                  'gehrd_lwork'), (a1,))
    ba, lo, hi, pivscale, info = gebal(a1, permute=0, overwrite_a=overwrite_a)
    _check_info(info, 'gebal (hessenberg)', positive=False)
    n = len(a1)

    lwork = _compute_lwork(gehrd_lwork, ba.shape[0], lo=lo, hi=hi)

    hq, tau, info = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'gehrd (hessenberg)', positive=False)
    h = np.triu(hq, -1)
    if not calc_q:
        return h

    # use orghr/unghr to compute q
    orghr, orghr_lwork = get_lapack_funcs(('orghr', 'orghr_lwork'), (a1,))
    lwork = _compute_lwork(orghr_lwork, n, lo=lo, hi=hi)

    q, info = orghr(a=hq, tau=tau, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'orghr (hessenberg)', positive=False)
    return h, q


def cdf2rdf(w, v):
    """
    Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real
    eigenvalues in a block diagonal form ``wr`` and the associated real
    eigenvectors ``vr``, such that::

        vr @ wr = X @ vr

    continues to hold, where ``X`` is the original array for which ``w`` and
    ``v`` are the eigenvalues and eigenvectors.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    w : (..., M) array_like
        Complex or real eigenvalues, an array or stack of arrays

        Conjugate pairs must not be interleaved, else the wrong result
        will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result,
        but ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.

    v : (..., M, M) array_like
        Complex or real eigenvectors, a square array or stack of square arrays.

    Returns
    -------
    wr : (..., M, M) ndarray
        Real diagonal block form of eigenvalues
    vr : (..., M, M) ndarray
        Real eigenvectors associated with ``wr``

    See Also
    --------
    eig : Eigenvalues and right eigenvectors for non-symmetric arrays
    rsf2csf : Convert real Schur form to complex Schur form

    Notes
    -----
    ``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.
    For example, obtained by ``w, v = scipy.linalg.eig(X)`` or
    ``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent
    stacked arrays.

    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
    >>> X
    array([[ 1,  2,  3],
           [ 0,  4,  5],
           [ 0, -5,  4]])

    >>> from scipy import linalg
    >>> w, v = linalg.eig(X)
    >>> w
    array([ 1.+0.j,  4.+5.j,  4.-5.j])
    >>> v
    array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],
           [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],
           [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])

    >>> wr, vr = linalg.cdf2rdf(w, v)
    >>> wr
    array([[ 1.,  0.,  0.],
           [ 0.,  4.,  5.],
           [ 0., -5.,  4.]])
    >>> vr
    array([[ 1.     ,  0.40016, -0.01906],
           [ 0.     ,  0.64788,  0.     ],
           [ 0.     ,  0.     ,  0.64788]])

    >>> vr @ wr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    >>> X @ vr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    """
    w, v = _asarray_validated(w), _asarray_validated(v)

    # check dimensions
    if w.ndim < 1:
        raise ValueError('expected w to be at least 1D')
    if v.ndim < 2:
        raise ValueError('expected v to be at least 2D')
    if v.ndim != w.ndim + 1:
        raise ValueError('expected eigenvectors array to have exactly one '
                         'dimension more than eigenvalues array')

    # check shapes
    n = w.shape[-1]
    M = w.shape[:-1]
    if v.shape[-2] != v.shape[-1]:
        raise ValueError('expected v to be a square matrix or stacked square '
                         'matrices: v.shape[-2] = v.shape[-1]')
    if v.shape[-1] != n:
        raise ValueError('expected the same number of eigenvalues as '
                         'eigenvectors')

    # get indices for each first pair of complex eigenvalues
    complex_mask = iscomplex(w)
    n_complex = complex_mask.sum(axis=-1)

    # check if all complex eigenvalues have conjugate pairs
    if not (n_complex % 2 == 0).all():
        raise ValueError('expected complex-conjugate pairs of eigenvalues')

    # find complex indices
    idx = nonzero(complex_mask)
    idx_stack = idx[:-1]
    idx_elem = idx[-1]

    # filter them to conjugate indices, assuming pairs are not interleaved
    j = idx_elem[0::2]
    k = idx_elem[1::2]
    stack_ind = ()
    for i in idx_stack:
        # should never happen, assuming nonzero orders by the last axis
        assert (i[0::2] == i[1::2]).all(), \
                "Conjugate pair spanned different arrays!"
        stack_ind += (i[0::2],)

    # all eigenvalues to diagonal form
    wr = zeros(M + (n, n), dtype=w.real.dtype)
    di = range(n)
    wr[..., di, di] = w.real

    # complex eigenvalues to real block diagonal form
    wr[stack_ind + (j, k)] = w[stack_ind + (j,)].imag
    wr[stack_ind + (k, j)] = w[stack_ind + (k,)].imag

    # compute real eigenvectors associated with real block diagonal eigenvalues
    u = zeros(M + (n, n), dtype=np.cdouble)
    u[..., di, di] = 1.0
    u[stack_ind + (j, j)] = 0.5j
    u[stack_ind + (j, k)] = 0.5
    u[stack_ind + (k, j)] = -0.5j
    u[stack_ind + (k, k)] = 0.5

    # multiply matrices v and u (equivalent to v @ u)
    vr = einsum('...ij,...jk->...ik', v, u).real

    return wr, vr
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'schur', 'rsf2csf', 'norm', 'LinAlgError', 'get_lapack_funcs', 'eigvals',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_schur",
                                   private_modules=["_decomp_schur"], all=__all__,
                                   attribute=name)

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'toeplitz', 'circulant', 'hankel',
    'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
    'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft',
    'fiedler', 'fiedler_companion', 'convolution_matrix'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="special_matrices",
                                   private_modules=["_special_matrices"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space',
    'LinAlgError', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_svd",
                                   private_modules=["_decomp_svd"], all=__all__,
                                   attribute=name)
<|endoftext|>import numpy as np
from scipy.linalg import svd


__all__ = ['polar']


def polar(a, side="right"):
    """
    Compute the polar decomposition.

    Returns the factors of the polar decomposition [1]_ `u` and `p` such
    that ``a = up`` (if `side` is "right") or ``a = pu`` (if `side` is
    "left"), where `p` is positive semidefinite. Depending on the shape
    of `a`, either the rows or columns of `u` are orthonormal. When `a`
    is a square array, `u` is a square unitary array. When `a` is not
    square, the "canonical polar decomposition" [2]_ is computed.

    Parameters
    ----------
    a : (m, n) array_like
        The array to be factored.
    side : {'left', 'right'}, optional
        Determines whether a right or left polar decomposition is computed.
        If `side` is "right", then ``a = up``.  If `side` is "left",  then
        ``a = pu``.  The default is "right".

    Returns
    -------
    u : (m, n) ndarray
        If `a` is square, then `u` is unitary. If m > n, then the columns
        of `a` are orthonormal, and if m < n, then the rows of `u` are
        orthonormal.
    p : ndarray
        `p` is Hermitian positive semidefinite. If `a` is nonsingular, `p`
        is positive definite. The shape of `p` is (n, n) or (m, m), depending
        on whether `side` is "right" or "left", respectively.

    References
    ----------
    .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge
           University Press, 1985.
    .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",
           SIAM, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import polar
    >>> a = np.array([[1, -1], [2, 4]])
    >>> u, p = polar(a)
    >>> u
    array([[ 0.85749293, -0.51449576],
           [ 0.51449576,  0.85749293]])
    >>> p
    array([[ 1.88648444,  1.2004901 ],
           [ 1.2004901 ,  3.94446746]])

    A non-square example, with m < n:

    >>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])
    >>> u, p = polar(b)
    >>> u
    array([[-0.21196618, -0.42393237,  0.88054056],
           [ 0.39378971,  0.78757942,  0.4739708 ]])
    >>> p
    array([[ 0.48470147,  0.96940295,  1.15122648],
           [ 0.96940295,  1.9388059 ,  2.30245295],
           [ 1.15122648,  2.30245295,  3.65696431]])
    >>> u.dot(p)   # Verify the decomposition.
    array([[ 0.5,  1. ,  2. ],
           [ 1.5,  3. ,  4. ]])
    >>> u.dot(u.T)   # The rows of u are orthonormal.
    array([[  1.00000000e+00,  -2.07353665e-17],
           [ -2.07353665e-17,   1.00000000e+00]])

    Another non-square example, with m > n:

    >>> c = b.T
    >>> u, p = polar(c)
    >>> u
    array([[-0.21196618,  0.39378971],
           [-0.42393237,  0.78757942],
           [ 0.88054056,  0.4739708 ]])
    >>> p
    array([[ 1.23116567,  1.93241587],
           [ 1.93241587,  4.84930602]])
    >>> u.dot(p)   # Verify the decomposition.
    array([[ 0.5,  1.5],
           [ 1. ,  3. ],
           [ 2. ,  4. ]])
    >>> u.T.dot(u)  # The columns of u are orthonormal.
    array([[  1.00000000e+00,  -1.26363763e-16],
           [ -1.26363763e-16,   1.00000000e+00]])

    """
    if side not in ['right', 'left']:
        raise ValueError("`side` must be either 'right' or 'left'")
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")

    w, s, vh = svd(a, full_matrices=False)
    u = w.dot(vh)
    if side == 'right':
        # a = up
        p = (vh.T.conj() * s).dot(vh)
    else:
        # a = pu
        p = (w * s).dot(w.T.conj())
    return u, p
<|endoftext|>from warnings import warn

import numpy as np
from numpy import (atleast_2d, arange, zeros_like, imag, diag,
                   iscomplexobj, tril, triu, argsort, empty_like)
from scipy._lib._util import ComplexWarning
from ._decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork

__all__ = ['ldl']


def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    """ Computes the LDLt or Bunch-Kaufman factorization of a symmetric/
    hermitian matrix.

    This function returns a block diagonal matrix D consisting blocks of size
    at most 2x2 and also a possibly permuted unit lower triangular matrix
    ``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``
    holds. If `lower` is False then (again possibly permuted) upper
    triangular matrices are returned as outer factors.

    The permutation array can be used to triangularize the outer factors
    simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower
    triangular matrix. This is also equivalent to multiplication with a
    permutation matrix ``P.dot(lu)``, where ``P`` is a column-permuted
    identity matrix ``I[:, perm]``.

    Depending on the value of the boolean `lower`, only upper or lower
    triangular part of the input array is referenced. Hence, a triangular
    matrix on entry would give the same result as if the full matrix is
    supplied.

    Parameters
    ----------
    A : array_like
        Square input array
    lower : bool, optional
        This switches between the lower and upper triangular outer factors of
        the factorization. Lower triangular (``lower=True``) is the default.
    hermitian : bool, optional
        For complex-valued arrays, this defines whether ``A = A.conj().T`` or
        ``A = A.T`` is assumed. For real-valued arrays, this switch has no
        effect.
    overwrite_a : bool, optional
        Allow overwriting data in `A` (may enhance performance). The default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : ndarray
        The (possibly) permuted upper/lower triangular outer factor of the
        factorization.
    d : ndarray
        The block diagonal multiplier of the factorization.
    perm : ndarray
        The row-permutation index array that brings lu into triangular form.

    Raises
    ------
    ValueError
        If input array is not square.
    ComplexWarning
        If a complex-valued array with nonzero imaginary parts on the
        diagonal is given and hermitian is set to True.

    See Also
    --------
    cholesky, lu

    Notes
    -----
    This function uses ``?SYTRF`` routines for symmetric matrices and
    ``?HETRF`` routines for Hermitian matrices from LAPACK. See [1]_ for
    the algorithm details.

    Depending on the `lower` keyword value, only lower or upper triangular
    part of the input array is referenced. Moreover, this keyword also defines
    the structure of the outer factors of the factorization.

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating
       inertia and solving symmetric linear systems, Math. Comput. Vol.31,
       1977. :doi:`10.2307/2005787`

    Examples
    --------
    Given an upper triangular array ``a`` that represents the full symmetric
    array with its entries, obtain ``l``, 'd' and the permutation vector `perm`:

    >>> import numpy as np
    >>> from scipy.linalg import ldl
    >>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])
    >>> lu, d, perm = ldl(a, lower=0) # Use the upper part
    >>> lu
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 1. ,  1. ,  1.5]])
    >>> d
    array([[-5. ,  0. ,  0. ],
           [ 0. ,  1.5,  0. ],
           [ 0. ,  0. ,  2. ]])
    >>> perm
    array([2, 1, 0])
    >>> lu[perm, :]
    array([[ 1. ,  1. ,  1.5],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1. ]])
    >>> lu.dot(d).dot(lu.T)
    array([[ 2., -1.,  3.],
           [-1.,  2.,  0.],
           [ 3.,  0.,  1.]])

    """
    a = atleast_2d(_asarray_validated(A, check_finite=check_finite))
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    # Return empty arrays for empty square input
    if a.size == 0:
        return empty_like(a), empty_like(a), np.array([], dtype=int)

    n = a.shape[0]
    r_or_c = complex if iscomplexobj(a) else float

    # Get the LAPACK routine
    if r_or_c is complex and hermitian:
        s, sl = 'hetrf', 'hetrf_lwork'
        if np.any(imag(diag(a))):
            warn('scipy.linalg.ldl():\nThe imaginary parts of the diagonal'
                 'are ignored. Use "hermitian=False" for factorization of'
                 'complex symmetric arrays.', ComplexWarning, stacklevel=2)
    else:
        s, sl = 'sytrf', 'sytrf_lwork'

    solver, solver_lwork = get_lapack_funcs((s, sl), (a,))
    lwork = _compute_lwork(solver_lwork, n, lower=lower)
    ldu, piv, info = solver(a, lwork=lwork, lower=lower,
                            overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError(f'{s.upper()} exited with the internal error "illegal value '
                         f'in argument number {-info}". See LAPACK documentation '
                         'for the error codes.')

    swap_arr, pivot_arr = _ldl_sanitize_ipiv(piv, lower=lower)
    d, lu = _ldl_get_d_and_l(ldu, pivot_arr, lower=lower, hermitian=hermitian)
    lu, perm = _ldl_construct_tri_factor(lu, swap_arr, pivot_arr, lower=lower)

    return lu, d, perm


def _ldl_sanitize_ipiv(a, lower=True):
    """
    This helper function takes the rather strangely encoded permutation array
    returned by the LAPACK routines ?(HE/SY)TRF and converts it into
    regularized permutation and diagonal pivot size format.

    Since FORTRAN uses 1-indexing and LAPACK uses different start points for
    upper and lower formats there are certain offsets in the indices used
    below.

    Let's assume a result where the matrix is 6x6 and there are two 2x2
    and two 1x1 blocks reported by the routine. To ease the coding efforts,
    we still populate a 6-sized array and fill zeros as the following ::

        pivots = [2, 0, 2, 0, 1, 1]

    This denotes a diagonal matrix of the form ::

        [x x        ]
        [x x        ]
        [    x x    ]
        [    x x    ]
        [        x  ]
        [          x]

    In other words, we write 2 when the 2x2 block is first encountered and
    automatically write 0 to the next entry and skip the next spin of the
    loop. Thus, a separate counter or array appends to keep track of block
    sizes are avoided. If needed, zeros can be filtered out later without
    losing the block structure.

    Parameters
    ----------
    a : ndarray
        The permutation array ipiv returned by LAPACK
    lower : bool, optional
        The switch to select whether upper or lower triangle is chosen in
        the LAPACK call.

    Returns
    -------
    swap_ : ndarray
        The array that defines the row/column swap operations. For example,
        if row two is swapped with row four, the result is [0, 3, 2, 3].
    pivots : ndarray
        The array that defines the block diagonal structure as given above.

    """
    n = a.size
    swap_ = arange(n)
    pivots = zeros_like(swap_, dtype=int)
    skip_2x2 = False

    # Some upper/lower dependent offset values
    # range (s)tart, r(e)nd, r(i)ncrement
    x, y, rs, re, ri = (1, 0, 0, n, 1) if lower else (-1, -1, n-1, -1, -1)

    for ind in range(rs, re, ri):
        # If previous spin belonged already to a 2x2 block
        if skip_2x2:
            skip_2x2 = False
            continue

        cur_val = a[ind]
        # do we have a 1x1 block or not?
        if cur_val > 0:
            if cur_val != ind+1:
                # Index value != array value --> permutation required
                swap_[ind] = swap_[cur_val-1]
            pivots[ind] = 1
        # Not.
        elif cur_val < 0 and cur_val == a[ind+x]:
            # first neg entry of 2x2 block identifier
            if -cur_val != ind+2:
                # Index value != array value --> permutation required
                swap_[ind+x] = swap_[-cur_val-1]
            pivots[ind+y] = 2
            skip_2x2 = True
        else:  # Doesn't make sense, give up
            raise ValueError('While parsing the permutation array '
                             'in "scipy.linalg.ldl", invalid entries '
                             'found. The array syntax is invalid.')
    return swap_, pivots


def _ldl_get_d_and_l(ldu, pivs, lower=True, hermitian=True):
    """
    Helper function to extract the diagonal and triangular matrices for
    LDL.T factorization.

    Parameters
    ----------
    ldu : ndarray
        The compact output returned by the LAPACK routing
    pivs : ndarray
        The sanitized array of {0, 1, 2} denoting the sizes of the pivots. For
        every 2 there is a succeeding 0.
    lower : bool, optional
        If set to False, upper triangular part is considered.
    hermitian : bool, optional
        If set to False a symmetric complex array is assumed.

    Returns
    -------
    d : ndarray
        The block diagonal matrix.
    lu : ndarray
        The upper/lower triangular matrix
    """
    is_c = iscomplexobj(ldu)
    d = diag(diag(ldu))
    n = d.shape[0]
    blk_i = 0  # block index

    # row/column offsets for selecting sub-, super-diagonal
    x, y = (1, 0) if lower else (0, 1)

    lu = tril(ldu, -1) if lower else triu(ldu, 1)
    diag_inds = arange(n)
    lu[diag_inds, diag_inds] = 1

    for blk in pivs[pivs != 0]:
        # increment the block index and check for 2s
        # if 2 then copy the off diagonals depending on uplo
        inc = blk_i + blk

        if blk == 2:
            d[blk_i+x, blk_i+y] = ldu[blk_i+x, blk_i+y]
            # If Hermitian matrix is factorized, the cross-offdiagonal element
            # should be conjugated.
            if is_c and hermitian:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y].conj()
            else:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y]

            lu[blk_i+x, blk_i+y] = 0.
        blk_i = inc

    return d, lu


def _ldl_construct_tri_factor(lu, swap_vec, pivs, lower=True):
    """
    Helper function to construct explicit outer factors of LDL factorization.

    If lower is True the permuted factors are multiplied as L(1)*L(2)*...*L(k).
    Otherwise, the permuted factors are multiplied as L(k)*...*L(2)*L(1). See
    LAPACK documentation for more details.

    Parameters
    ----------
    lu : ndarray
        The triangular array that is extracted from LAPACK routine call with
        ones on the diagonals.
    swap_vec : ndarray
        The array that defines the row swapping indices. If the kth entry is m
        then rows k,m are swapped. Notice that the mth entry is not necessarily
        k to avoid undoing the swapping.
    pivs : ndarray
        The array that defines the block diagonal structure returned by
        _ldl_sanitize_ipiv().
    lower : bool, optional
        The boolean to switch between lower and upper triangular structure.

    Returns
    -------
    lu : ndarray
        The square outer factor which satisfies the L * D * L.T = A
    perm : ndarray
        The permutation vector that brings the lu to the triangular form

    Notes
    -----
    Note that the original argument "lu" is overwritten.

    """
    n = lu.shape[0]
    perm = arange(n)
    # Setup the reading order of the permutation matrix for upper/lower
    rs, re, ri = (n-1, -1, -1) if lower else (0, n, 1)

    for ind in range(rs, re, ri):
        s_ind = swap_vec[ind]
        if s_ind != ind:
            # Column start and end positions
            col_s = ind if lower else 0
            col_e = n if lower else ind+1

            # If we stumble upon a 2x2 block include both cols in the perm.
            if pivs[ind] == (0 if lower else 2):
                col_s += -1 if lower else 0
                col_e += 0 if lower else 1
            lu[[s_ind, ind], col_s:col_e] = lu[[ind, s_ind], col_s:col_e]
            perm[[s_ind, ind]] = perm[[ind, s_ind]]

    return lu, argsort(perm)
<|endoftext|>"""
Solve the orthogonal Procrustes problem.

"""
import numpy as np
from ._decomp_svd import svd


__all__ = ['orthogonal_procrustes']


def orthogonal_procrustes(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal (or unitary) Procrustes problem.

    Given matrices `A` and `B` of the same shape, find an orthogonal (or unitary in
    the case of complex input) matrix `R` that most closely maps `A` to `B` using the
    algorithm given in [1]_.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to be mapped.
    B : (M, N) array_like
        Target matrix.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : (N, N) ndarray
        The matrix solution of the orthogonal Procrustes problem.
        Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
        ``R.conj().T @ R = I``.
    scale : float
        Sum of the singular values of ``A.conj().T @ B``.

    Raises
    ------
    ValueError
        If the input array shapes don't match or if check_finite is True and
        the arrays contain Inf or NaN.

    Notes
    -----
    Note that unlike higher level Procrustes analyses of spatial data, this
    function only uses orthogonal transformations like rotations and
    reflections, and it does not use scaling or translation.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1966.
           :doi:`10.1007/BF02289451`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orthogonal_procrustes
    >>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

    Flip the order of columns and check for the anti-diagonal mapping

    >>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
    >>> R
    array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
    >>> sca
    9.0

    As an example of the unitary Procrustes problem, generate a
    random complex matrix ``A``, a random unitary matrix ``Q``,
    and their product ``B``.

    >>> shape = (4, 4)
    >>> rng = np.random.default_rng(589234981235)
    >>> A = rng.random(shape) + rng.random(shape)*1j
    >>> Q = rng.random(shape) + rng.random(shape)*1j
    >>> Q, _ = np.linalg.qr(Q)
    >>> B = A @ Q

    `orthogonal_procrustes` recovers the unitary matrix ``Q``
    from ``A`` and ``B``.

    >>> R, _ = orthogonal_procrustes(A, B)
    >>> np.allclose(R, Q)
    True

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    if A.ndim != 2:
        raise ValueError(f'expected ndim to be 2, but observed {A.ndim}')
    if A.shape != B.shape:
        raise ValueError(f'the shapes of A and B differ ({A.shape} vs {B.shape})')
    # Be clever with transposes, with the intention to save memory.
    # The conjugate has no effect for real inputs, but gives the correct solution
    # for complex inputs.
    u, w, vt = svd((B.T @ np.conjugate(A)).T)
    R = u @ vt
    scale = w.sum()
    return R, scale
<|endoftext|>"""QR decomposition functions."""
import numpy as np

# Local imports
from .lapack import get_lapack_funcs
from ._misc import _datacopied

__all__ = ['qr', 'qr_multiply', 'rq']


def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    lwork = kwargs.get("lwork", None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(np.int_)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError("illegal value in %dth argument of internal %s"
                         % (-ret[-1], name))
    return ret[:-2]


def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False,
       check_finite=True):
    """
    Compute QR decomposition of a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in `a` is overwritten (may improve performance if
        `overwrite_a` is set to True by reusing the existing input data
        structure rather than creating a new one.)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic', 'raw'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes). The final option 'raw'
        (added in SciPy 0.11) makes the function return two matrices
        (Q, TAU) in the internal format used by LAPACK.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition. If pivoting, compute the decomposition
        ``A[:, P] = Q @ R`` as above, but where P is chosen such that the
        diagonal of R is non-increasing. Equivalently, albeit less efficiently,
        an explicit P matrix may be formed explicitly by permuting the rows or columns
        (depending on the side of the equation on which it is to be used) of
        an identity matrix. See Examples.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    Q : float or complex ndarray
        Of shape (M, M), or (M, K) for ``mode='economic'``. Not returned
        if ``mode='r'``. Replaced by tuple ``(Q, TAU)`` if ``mode='raw'``.
    R : float or complex ndarray
        Of shape (M, N), or (K, N) for ``mode in ['economic', 'raw']``.
        ``K = min(M, N)``.
    P : int ndarray
        Of shape (N,) for ``pivoting=True``. Not returned if
        ``pivoting=False``.

    Raises
    ------
    LinAlgError
        Raised if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, zungqr, dgeqp3, and zgeqp3.

    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
    of (M,M) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))

    >>> q, r = linalg.qr(a)
    >>> np.allclose(a, np.dot(q, r))
    True
    >>> q.shape, r.shape
    ((9, 9), (9, 6))

    >>> r2 = linalg.qr(a, mode='r')
    >>> np.allclose(r, r2)
    True

    >>> q3, r3 = linalg.qr(a, mode='economic')
    >>> q3.shape, r3.shape
    ((9, 6), (6, 6))

    >>> q4, r4, p4 = linalg.qr(a, pivoting=True)
    >>> d = np.abs(np.diag(r4))
    >>> np.all(d[1:] <= d[:-1])
    True
    >>> np.allclose(a[:, p4], np.dot(q4, r4))
    True
    >>> P = np.eye(p4.size)[p4]
