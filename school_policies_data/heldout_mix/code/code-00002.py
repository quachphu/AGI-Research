        (Optional output) Optimum function value.
    funcalls : int
        (Optional output) Number of objective function evaluations made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Golden' `method` in particular.

    Notes
    -----
    Uses analog of bisection method to decrease the bracketed
    interval.

    Examples
    --------
    We illustrate the behaviour of the function when `brack` is of
    size 2 and 3, respectively. In the case where `brack` is of the
    form (xa,xb), we can see for the given values, the output need
    not necessarily lie in the range ``(xa, xb)``.

    >>> def f(x):
    ...     return (x-1)**2

    >>> from scipy import optimize

    >>> minimizer = optimize.golden(f, brack=(1, 2))
    >>> minimizer
    1
    >>> res = optimize.golden(f, brack=(-1, 0.5, 2), full_output=True)
    >>> xmin, fval, funcalls = res
    >>> f(xmin), fval
    (9.925165290385052e-18, 9.925165290385052e-18)

    """
    options = {'xtol': tol, 'maxiter': maxiter}
    res = _minimize_scalar_golden(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_golden(func, brack=None, args=(),
                            xtol=_epsilon, maxiter=5000, disp=0,
                            **unknown_options):
    """
    Options
    -------
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.
    """
    _check_unknown_options(unknown_options)
    tol = xtol
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                   xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        if (xa > xc):  # swap so xa < xc can be assumed
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not"
                " fulfill this requirement: (xa < xb) and (xb < xc)"
            )
        fa = func(*((xa,) + args))
        fb = func(*((xb,) + args))
        fc = func(*((xc,) + args))
        if not ((fb < fa) and (fb < fc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not fulfill"
                " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
            )
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    _gR = 0.61803399  # golden ratio conjugate: 2.0/(1.0+sqrt(5.0))
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if (np.abs(xc - xb) > np.abs(xb - xa)):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = func(*((x1,) + args))
    f2 = func(*((x2,) + args))
    funcalls += 2
    nit = 0

    if disp > 2:
        print(" ")
        print(f"{'Func-count':^12} {'x':^12} {'f(x)': ^12}")

    for i in range(maxiter):
        if np.abs(x3 - x0) <= tol * (np.abs(x1) + np.abs(x2)):
            break
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,) + args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,) + args))
        funcalls += 1
        if disp > 2:
            if (f1 < f2):
                xmin, fval = x1, f1
            else:
                xmin, fval = x2, f2
            print(f"{funcalls:^12g} {xmin:^12.6g} {fval:^12.6g}")

        nit += 1
    # end of iteration loop

    if (f1 < f2):
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2

    success = nit < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    if success:
        message = ("\nOptimization terminated successfully;\n"
                   "The returned value satisfies the termination criteria\n"
                   f"(using xtol = {xtol} )")
    else:
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if np.isnan(xmin) or np.isnan(fval):
            message = f"{_status_message['nan']}"

    if disp:
        _print_success_message_or_warn(not success, message)

    return OptimizeResult(fun=fval, nfev=funcalls, x=xmin, nit=nit,
                          success=success, message=message)


def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of a function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    three points that bracket the minimum of the function.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Initial points. Defaults `xa` to 0.0, and `xb` to 1.0.
        A local minimum need not be contained within this interval.
    args : tuple, optional
        Additional arguments (if present), passed to `func`.
    grow_limit : float, optional
        Maximum grow limit.  Defaults to 110.0
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.

    Returns
    -------
    xa, xb, xc : float
        Final points of the bracket.
    fa, fb, fc : float
        Objective function values at the bracket points.
    funcalls : int
        Number of function evaluations made.

    Raises
    ------
    BracketError
        If no valid bracket is found before the algorithm terminates.
        See notes for conditions of a valid bracket.

    Notes
    -----
    The algorithm attempts to find three strictly ordered points (i.e.
    :math:`x_a < x_b < x_c` or :math:`x_c < x_b < x_a`) satisfying
    :math:`f(x_b) ≤ f(x_a)` and :math:`f(x_b) ≤ f(x_c)`, where one of the
    inequalities must be satisfied strictly and all :math:`x_i` must be
    finite.

    Examples
    --------
    This function can find a downward convex region of a function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import bracket
    >>> def f(x):
    ...     return 10*x**2 + 3*x + 5
    >>> x = np.linspace(-2, 2)
    >>> y = f(x)
    >>> init_xa, init_xb = 0.1, 1
    >>> xa, xb, xc, fa, fb, fc, funcalls = bracket(f, xa=init_xa, xb=init_xb)
    >>> plt.axvline(x=init_xa, color="k", linestyle="--")
    >>> plt.axvline(x=init_xb, color="k", linestyle="--")
    >>> plt.plot(x, y, "-k")
    >>> plt.plot(xa, fa, "bx")
    >>> plt.plot(xb, fb, "rx")
    >>> plt.plot(xc, fc, "bx")
    >>> plt.show()

    Note that both initial points were to the right of the minimum, and the
    third point was found in the "downhill" direction: the direction
    in which the function appeared to be decreasing (to the left).
    The final points are strictly ordered, and the function value
    at the middle point is less than the function values at the endpoints;
    it follows that a minimum must lie within the bracket.

    """
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    # convert to numpy floats if not already
    xa, xb = np.asarray([xa, xb])
    fa = func(*(xa,) + args)
    fb = func(*(xb,) + args)
    if (fa < fb):                      # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func(*((xc,) + args))
    funcalls = 3
    iter = 0
    while (fc < fb):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        msg = ("No valid bracket was found before the iteration limit was "
               "reached. Consider trying different initial points or "
               "increasing `maxiter`.")
        if iter > maxiter:
            raise RuntimeError(msg)
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif (fw > fb):
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(xc - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,) + args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # three conditions for a valid bracket
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc or xc < xb < xa)
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    msg = ("The algorithm terminated without finding a valid bracket. "
           "Consider trying different initial points.")
    if not (cond1 and cond2 and cond3):
        e = BracketError(msg)
        e.data = (xa, xb, xc, fa, fb, fc, funcalls)
        raise e

    return xa, xb, xc, fa, fb, fc, funcalls


class BracketError(RuntimeError):
    pass


def _recover_from_bracket_error(solver, fun, bracket, args, **options):
    # `bracket` was originally written without checking whether the resulting
    # bracket is valid. `brent` and `golden` built on top of it without
    # checking the returned bracket for validity, and their output can be
    # incorrect without warning/error if the original bracket is invalid.
    # gh-14858 noticed the problem, and the following is the desired
    # behavior:
    # - `scipy.optimize.bracket`, `scipy.optimize.brent`, and
    #   `scipy.optimize.golden` should raise an error if the bracket is
    #   invalid, as opposed to silently returning garbage
    # - `scipy.optimize.minimize_scalar` should return with `success=False`
    #   and other information
    # The changes that would be required to achieve this the traditional
    # way (`return`ing all the required information from bracket all the way
    # up to `minimizer_scalar`) are extensive and invasive. (See a6aa40d.)
    # We can achieve the same thing by raising the error in `bracket`, but
    # storing the information needed by `minimize_scalar` in the error object,
    # and intercepting it here.
    try:
        res = solver(fun, bracket, args, **options)
    except BracketError as e:
        msg = str(e)
        xa, xb, xc, fa, fb, fc, funcalls = e.data
        xs, fs = [xa, xb, xc], [fa, fb, fc]
        if np.any(np.isnan([xs, fs])):
            x, fun = np.nan, np.nan
        else:
            imin = np.argmin(fs)
            x, fun = xs[imin], fs[imin]
        return OptimizeResult(fun=fun, nfev=funcalls, x=x,
                              nit=0, success=False, message=msg)
    return res


def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return fret, p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return res.fun, p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return res.fun, p + xi, xi


def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func.
    xtol : float, optional
        Line-search error tolerance.
    ftol : float, optional
        Relative error in ``func(xopt)`` acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        If True, ``fopt``, ``xi``, ``direc``, ``iter``, ``funcalls``, and
        ``warnflag`` are returned.
    disp : bool, optional
        If True, print convergence messages.
    retall : bool, optional
        If True, return a list of the solution at each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each
        iteration.  Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    direc : ndarray, optional
        Initial fitting step and parameter order set as an (N, N) array, where N
        is the number of fitting parameters in `x0`. Defaults to step size 1.0
        fitting all parameters simultaneously (``np.eye((N, N))``). To
        prevent initial consideration of values in a step or to change initial
        step size, set to 0 or desired step size in the Jth position in the Mth
        block, where J is the position in `x0` and M is the desired evaluation
        step, with steps being evaluated in index order. Step size and ordering
        will change freely as minimization proceeds.

    Returns
    -------
    xopt : ndarray
        Parameter which minimizes `func`.
    fopt : number
        Value of function at minimum: ``fopt = func(xopt)``.
    direc : ndarray
        Current direction set.
    iter : int
        Number of iterations.
    funcalls : int
        Number of function calls made.
    warnflag : int
        Integer warning flag:
            1 : Maximum number of function evaluations.
            2 : Maximum number of iterations.
            3 : NaN result encountered.
            4 : The result is out of the provided bounds.
    allvecs : list
        List of solutions at each iteration.

    See also
    --------
    minimize: Interface to unconstrained minimization algorithms for
        multivariate functions. See the 'Powell' method in particular.

    Notes
    -----
    Uses a modification of Powell's method to find the minimum of
    a function of N variables. Powell's method is a conjugate
    direction method.

    The algorithm has two loops. The outer loop merely iterates over the inner
    loop. The inner loop minimizes over each current direction in the direction
    set. At the end of the inner loop, if certain conditions are met, the
    direction that gave the largest decrease is dropped and replaced with the
    difference between the current estimated x and the estimated x from the
    beginning of the inner-loop.

    The technical conditions for replacing the direction of greatest
    increase amount to checking that

    1. No further gain can be made along the direction of greatest increase
       from that iteration.
    2. The direction of greatest increase accounted for a large sufficient
       fraction of the decrease in the function value from that iteration of
       the inner loop.

    References
    ----------
    Powell M.J.D. (1964) An efficient method for finding the minimum of a
    function of several variables without calculating derivatives,
    Computer Journal, 7 (2):155-162.

    Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
    Numerical Recipes (any edition), Cambridge University Press

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin_powell(f, -1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 2
             Function evaluations: 16
    >>> minimum
    array(0.0)

    """
    opts = {'xtol': xtol,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'direc': direc,
            'return_all': retall}

    callback = _wrap_callback(callback)
    res = _minimize_powell(func, x0, args, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['direc'], res['nit'],
                   res['nfev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized::

            fun(x, *args) -> float

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        The present documentation is specific to ``method='powell'``, but other
        options are available. See documentation for `scipy.optimize.minimize`.
    bounds : sequence or `Bounds`, optional
        Bounds on decision variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
           is used to specify no bound.

        If bounds are not provided, then an unbounded line search will be used.
        If bounds are provided and the initial guess is within the bounds, then
        every function evaluation throughout the minimization procedure will be
        within the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (or left to default), then some
        function evaluations during the first iteration may be outside the
        bounds, but every function evaluation after the first iteration will be
        within the bounds. If `direc` is not full rank, then some parameters
        may not be optimized and the solution is not guaranteed to be within
        the bounds.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

        maxiter : int
            Maximum number of iterations to perform. Depending on the
            method each iteration may use several function evaluations.
        disp : bool
            Set to True to print convergence messages.

        See method-specific options for ``method='powell'`` below.
    callback : callable, optional
        Called after each iteration. The signature is::

            callback(xk)

        where ``xk`` is the current parameter vector.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all

    x = asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    # we need to use a mutable object here that we can update in the
    # wrapper function
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    if direc is None:
        direc = eye(N, dtype=float)
    else:
        direc = asarray(direc, dtype=float)
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn("direc input is not full rank, some parameters may "
                          "not be optimized",
                          OptimizeWarning, stacklevel=3)

    if bounds is None:
        # don't make these arrays of all +/- inf. because
        # _linesearch_powell will do an unnecessary check of all the elements.
        # just keep them None, _linesearch_powell will not have to check
        # all the elements.
        lower_bound, upper_bound = None, None
    else:
        # bounds is standardized in _minimize.py.
        lower_bound, upper_bound = bounds.lb, bounds.ub
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, stacklevel=3)

    fval = func(x)
    x1 = x.copy()
    iter = 0
    while True:
        try:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in range(N):
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i
            iter += 1
            if retall:
                allvecs.append(x)
            intermediate_result = OptimizeResult(x=x, fun=fval)
            if _call_callback_maybe_halt(callback, intermediate_result):
                break
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break
            if fcalls[0] >= maxfun:
                break
            if iter >= maxiter:
                break
            if np.isnan(fx) and np.isnan(fval):
                # Ended up in a nan-region: bail out
                break

            # Construct the extrapolated point
            direc1 = x - x1
            x1 = x.copy()
            # make sure that we don't go outside the bounds when extrapolating
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1) * direc1
            fx2 = func(x2)

            if (fx > fx2):
                t = 2.0*(fx + fx2 - 2.0*fval)
                temp = (fx - fval - delta)
                t *= temp*temp
                temp = fx - fx2
                t -= delta*temp*temp
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(
                        func, x, direc1,
                        tol=xtol * 100,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        fval=fval
                    )
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _MaxFuncCallError:
            break

    warnflag = 0
    msg = _status_message['success']
    # out of bounds is more urgent than exceeding function evals or iters,
    # but I don't want to cause inconsistencies by changing the
    # established warning flags for maxfev and maxiter, so the out of bounds
    # warning flag becomes 3, but is checked for first.
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']

    if disp:
        _print_success_message_or_warn(warnflag, msg, RuntimeWarning)
        print(f"         Current function value: {fval:f}")
        print("         Iterations: %d" % iter)
        print("         Function evaluations: %d" % fcalls[0])
    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x)
    if retall:
        result['allvecs'] = allvecs
    return result


def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        if disp > 1:
            print("\nOptimization terminated successfully;\n"
                  "The returned value satisfies the termination criteria\n"
                  "(using xtol = ", xtol, ")")
        return

    if flag == 1:
        msg = ("\nMaximum number of function evaluations exceeded --- "
               "increase maxfun argument.\n")
    elif flag == 2:
        msg = f"\n{_status_message['nan']}"

    _print_success_message_or_warn(flag, msg)
    return


def brute(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
          disp=False, workers=1):
    """Minimize a function over a given range by brute force.

    Uses the "brute force" method, i.e., computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.

    The function is evaluated everywhere in the range with the datatype of the
    first call to the function, as enforced by the ``vectorize`` NumPy
    function. The value and type of the function evaluation returned when
    ``full_output=True`` are affected in addition by the ``finish`` argument
    (see Notes).

    The brute force approach is inefficient because the number of grid points
    increases exponentially - the number of grid points to evaluate is
    ``Ns ** len(x)``. Consequently, even with coarse grid spacing, even
    moderately sized problems can take a long time to run, and/or run into
    memory limitations.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be either a
        "slice object" or a range tuple of the form ``(low, high)``.
        The program uses these to create the grid of points on which
        the objective function will be computed. See `Note 2` for
        more detail.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified. See `Note2`.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
    finish : callable, optional
        An optimization function that is called with the result of brute force
        minimization as initial guess. `finish` should take `func` and
        the initial guess as positional arguments, and take `args` as
        keyword arguments. It may additionally take `full_output`
        and/or `disp` as keyword arguments. Use None if no "polishing"
        function is to be used. See Notes for more details.
    disp : bool, optional
        Set to True to print convergence messages from the `finish` callable.
    workers : int or map-like callable, optional
        If `workers` is an int the grid is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the grid in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.3.0

    Returns
    -------
    x0 : ndarray
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value. (See `Note 1` for
        which point is returned.)
    fval : float
        Function value at the point `x0`. (Returned when `full_output` is
        True.)
    grid : tuple
        Representation of the evaluation grid. It has the same
        length as `x0`. (Returned when `full_output` is True.)
    Jout : ndarray
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``. (Returned
        when `full_output` is True.)

    See Also
    --------
    basinhopping, differential_evolution

    Notes
    -----
    *Note 1*: The program finds the gridpoint at which the lowest value
    of the objective function occurs. If `finish` is None, that is the
    point returned. When the global minimum occurs within (or not very far
    outside) the grid's boundaries, and the grid is fine enough, that
    point will be in the neighborhood of the global minimum.

    However, users often employ some other optimization program to
    "polish" the gridpoint values, i.e., to seek a more precise
    (local) minimum near `brute's` best gridpoint.
    The `brute` function's `finish` option provides a convenient way to do
    that. Any polishing program used must take `brute's` output as its
    initial guess as a positional argument, and take `brute's` input values
    for `args` as keyword arguments, otherwise an error will be raised.
    It may additionally take `full_output` and/or `disp` as keyword arguments.

    `brute` assumes that the `finish` function returns either an
    `OptimizeResult` object or a tuple in the form:
    ``(xmin, Jmin, ... , statuscode)``, where ``xmin`` is the minimizing
    value of the argument, ``Jmin`` is the minimum value of the objective
    function, "..." may be some other returned values (which are not used
    by `brute`), and ``statuscode`` is the status code of the `finish` program.

    Note that when `finish` is not None, the values returned are those
    of the `finish` program, *not* the gridpoint ones. Consequently,
    while `brute` confines its search to the input grid points,
    the `finish` program's results usually will not coincide with any
    gridpoint, and may fall outside the grid's boundary. Thus, if a
    minimum only needs to be found over the provided grid points, make
    sure to pass in ``finish=None``.

    *Note 2*: The grid of points is a `numpy.mgrid` object.
    For `brute` the `ranges` and `Ns` inputs have the following effect.
    Each component of the `ranges` tuple can be either a slice object or a
    two-tuple giving a range of values, such as (0, 5). If the component is a
    slice object, `brute` uses it directly. If the component is a two-tuple
    range, `brute` internally converts it to a slice object that interpolates
    `Ns` points from its low-value to its high-value, inclusive.

    Examples
    --------
    We illustrate the use of `brute` to seek the global minimum of a function
    of two variables that is given as the sum of a positive-definite
    quadratic and two deep "Gaussian-shaped" craters. Specifically, define
    the objective function `f` as the sum of three other functions,
    ``f = f1 + f2 + f3``. We suppose each of these has a signature
    ``(z, *params)``, where ``z = (x, y)``,  and ``params`` and the functions
    are as defined below.

    >>> import numpy as np
    >>> params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
    >>> def f1(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

    >>> def f2(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

    >>> def f3(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

    >>> def f(z, *params):
    ...     return f1(z, *params) + f2(z, *params) + f3(z, *params)

    Thus, the objective function may have local minima near the minimum
    of each of the three functions of which it is composed. To
    use `fmin` to polish its gridpoint result, we may then continue as
    follows:

    >>> rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
    >>> from scipy import optimize
    >>> resbrute = optimize.brute(f, rranges, args=params, full_output=True,
    ...                           finish=optimize.fmin)
    >>> resbrute[0]  # global minimum
    array([-1.05665192,  1.80834843])
    >>> resbrute[1]  # function value at global minimum
    -3.4085818767

    Note that if `finish` had been set to None, we would have gotten the
    gridpoint [-1.0 1.75] where the rounded function value is -2.892.

    """
    N = len(ranges)
    if N > 40:
        raise ValueError("Brute Force not possible with more "
                         "than 40 variables.")
    lrange = list(ranges)
    for k in range(N):
        if not isinstance(lrange[k], slice):
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            lrange[k] = slice(*lrange[k])
    if (N == 1):
        lrange = lrange[0]

    grid = np.mgrid[lrange]

    # obtain an array of parameters that is iterable by a map-like callable
    inpt_shape = grid.shape
    if (N > 1):
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    if not np.iterable(args):
        args = (args,)

    wrapped_func = _Brute_Wrapper(func, args)

    # iterate over input arrays, possibly in parallel
    with MapWrapper(pool=workers) as mapper:
        Jout = np.array(list(mapper(wrapped_func, grid)))
        if (N == 1):
            grid = (grid,)
            Jout = np.squeeze(Jout)
        elif (N > 1):
            Jout = np.reshape(Jout, inpt_shape[1:])
            grid = np.reshape(grid.T, inpt_shape)

    Nshape = shape(Jout)

    indx = argmin(Jout.ravel(), axis=-1)
    Nindx = np.empty(N, int)
    xmin = np.empty(N, float)
    for k in range(N - 1, -1, -1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx // thisN
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]

    Jmin = Jout[tuple(Nindx)]
    if (N == 1):
        grid = grid[0]
        xmin = xmin[0]

    if callable(finish):
        # set up kwargs for `finish` function
        finish_args = _getfullargspec(finish).args
        finish_kwargs = dict()
        if 'full_output' in finish_args:
            finish_kwargs['full_output'] = 1
        if 'disp' in finish_args:
            finish_kwargs['disp'] = disp
        elif 'options' in finish_args:
            # pass 'disp' as `options`
            # (e.g., if `finish` is `minimize`)
            finish_kwargs['options'] = {'disp': disp}

        # run minimizer
        res = finish(func, xmin, args=args, **finish_kwargs)

        if isinstance(res, OptimizeResult):
            xmin = res.x
            Jmin = res.fun
            success = res.success
        else:
            xmin = res[0]
            Jmin = res[1]
            success = res[-1] == 0
        if not success:
            if disp:
                warnings.warn("Either final optimization did not succeed or `finish` "
                              "does not return `statuscode` as its last argument.",
                              RuntimeWarning, stacklevel=2)

    if full_output:
        return xmin, Jmin, grid, Jout
    else:
        return xmin


class _Brute_Wrapper:
    """
    Object to wrap user cost function for optimize.brute, allowing picklability
    """

    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        # flatten needed for one dimensional case.
        return self.f(np.asarray(x).flatten(), *self.args)


def show_options(solver=None, method=None, disp=True):
    """
    Show documentation for additional options of optimization solvers.

    These are method-specific options that can be supplied through the
    ``options`` dict.

    Parameters
    ----------
    solver : str
        Type of optimization solver. One of 'minimize', 'minimize_scalar',
        'root', 'root_scalar', 'linprog', or 'quadratic_assignment'.
    method : str, optional
        If not given, shows all methods of the specified solver. Otherwise,
        show only the options for the specified method. Valid values
        corresponds to methods' names of respective solver (e.g., 'BFGS' for
        'minimize').
    disp : bool, optional
        Whether to print the result rather than returning it.

    Returns
    -------
    text
        Either None (for disp=True) or the text string (disp=False)

    Notes
    -----
    The solver-specific methods are:

    `scipy.optimize.minimize`

    - :ref:`Nelder-Mead <optimize.minimize-neldermead>`
    - :ref:`Powell      <optimize.minimize-powell>`
    - :ref:`CG          <optimize.minimize-cg>`
    - :ref:`BFGS        <optimize.minimize-bfgs>`
    - :ref:`Newton-CG   <optimize.minimize-newtoncg>`
    - :ref:`L-BFGS-B    <optimize.minimize-lbfgsb>`
    - :ref:`TNC         <optimize.minimize-tnc>`
    - :ref:`COBYLA      <optimize.minimize-cobyla>`
    - :ref:`COBYQA      <optimize.minimize-cobyqa>`
    - :ref:`SLSQP       <optimize.minimize-slsqp>`
    - :ref:`dogleg      <optimize.minimize-dogleg>`
    - :ref:`trust-ncg   <optimize.minimize-trustncg>`

    `scipy.optimize.root`

    - :ref:`hybr              <optimize.root-hybr>`
    - :ref:`lm                <optimize.root-lm>`
    - :ref:`broyden1          <optimize.root-broyden1>`
    - :ref:`broyden2          <optimize.root-broyden2>`
    - :ref:`anderson          <optimize.root-anderson>`
    - :ref:`linearmixing      <optimize.root-linearmixing>`
    - :ref:`diagbroyden       <optimize.root-diagbroyden>`
    - :ref:`excitingmixing    <optimize.root-excitingmixing>`
    - :ref:`krylov            <optimize.root-krylov>`
    - :ref:`df-sane           <optimize.root-dfsane>`

    `scipy.optimize.minimize_scalar`

    - :ref:`brent       <optimize.minimize_scalar-brent>`
    - :ref:`golden      <optimize.minimize_scalar-golden>`
    - :ref:`bounded     <optimize.minimize_scalar-bounded>`

    `scipy.optimize.root_scalar`

    - :ref:`bisect  <optimize.root_scalar-bisect>`
    - :ref:`brentq  <optimize.root_scalar-brentq>`
    - :ref:`brenth  <optimize.root_scalar-brenth>`
    - :ref:`ridder  <optimize.root_scalar-ridder>`
    - :ref:`toms748 <optimize.root_scalar-toms748>`
    - :ref:`newton  <optimize.root_scalar-newton>`
    - :ref:`secant  <optimize.root_scalar-secant>`
    - :ref:`halley  <optimize.root_scalar-halley>`

    `scipy.optimize.linprog`

    - :ref:`simplex           <optimize.linprog-simplex>`
    - :ref:`interior-point    <optimize.linprog-interior-point>`
    - :ref:`revised simplex   <optimize.linprog-revised_simplex>`
    - :ref:`highs             <optimize.linprog-highs>`
    - :ref:`highs-ds          <optimize.linprog-highs-ds>`
    - :ref:`highs-ipm         <optimize.linprog-highs-ipm>`

    `scipy.optimize.quadratic_assignment`

    - :ref:`faq             <optimize.qap-faq>`
    - :ref:`2opt            <optimize.qap-2opt>`

    Examples
    --------
    We can print documentations of a solver in stdout:

    >>> from scipy.optimize import show_options
    >>> show_options(solver="minimize")
    ...

    Specifying a method is possible:

    >>> show_options(solver="minimize", method="Nelder-Mead")
    ...

    We can also get the documentations as a string:

    >>> show_options(solver="minimize", method="Nelder-Mead", disp=False)
    Minimization of scalar function of one or more variables using the ...

    """
    import textwrap

    doc_routines = {
        'minimize': (
            ('bfgs', 'scipy.optimize._optimize._minimize_bfgs'),
            ('cg', 'scipy.optimize._optimize._minimize_cg'),
            ('cobyla', 'scipy.optimize._cobyla_py._minimize_cobyla'),
            ('cobyqa', 'scipy.optimize._cobyqa_py._minimize_cobyqa'),
            ('dogleg', 'scipy.optimize._trustregion_dogleg._minimize_dogleg'),
            ('l-bfgs-b', 'scipy.optimize._lbfgsb_py._minimize_lbfgsb'),
            ('nelder-mead', 'scipy.optimize._optimize._minimize_neldermead'),
            ('newton-cg', 'scipy.optimize._optimize._minimize_newtoncg'),
            ('powell', 'scipy.optimize._optimize._minimize_powell'),
            ('slsqp', 'scipy.optimize._slsqp_py._minimize_slsqp'),
            ('tnc', 'scipy.optimize._tnc._minimize_tnc'),
            ('trust-ncg',
             'scipy.optimize._trustregion_ncg._minimize_trust_ncg'),
            ('trust-constr',
             'scipy.optimize._trustregion_constr.'
             '_minimize_trustregion_constr'),
            ('trust-exact',
             'scipy.optimize._trustregion_exact._minimize_trustregion_exact'),
            ('trust-krylov',
             'scipy.optimize._trustregion_krylov._minimize_trust_krylov'),
        ),
        'root': (
            ('hybr', 'scipy.optimize._minpack_py._root_hybr'),
            ('lm', 'scipy.optimize._root._root_leastsq'),
            ('broyden1', 'scipy.optimize._root._root_broyden1_doc'),
            ('broyden2', 'scipy.optimize._root._root_broyden2_doc'),
            ('anderson', 'scipy.optimize._root._root_anderson_doc'),
            ('diagbroyden', 'scipy.optimize._root._root_diagbroyden_doc'),
            ('excitingmixing', 'scipy.optimize._root._root_excitingmixing_doc'),
            ('linearmixing', 'scipy.optimize._root._root_linearmixing_doc'),
            ('krylov', 'scipy.optimize._root._root_krylov_doc'),
            ('df-sane', 'scipy.optimize._spectral._root_df_sane'),
        ),
        'root_scalar': (
            ('bisect', 'scipy.optimize._root_scalar._root_scalar_bisect_doc'),
            ('brentq', 'scipy.optimize._root_scalar._root_scalar_brentq_doc'),
            ('brenth', 'scipy.optimize._root_scalar._root_scalar_brenth_doc'),
            ('ridder', 'scipy.optimize._root_scalar._root_scalar_ridder_doc'),
            ('toms748', 'scipy.optimize._root_scalar._root_scalar_toms748_doc'),
            ('secant', 'scipy.optimize._root_scalar._root_scalar_secant_doc'),
            ('newton', 'scipy.optimize._root_scalar._root_scalar_newton_doc'),
            ('halley', 'scipy.optimize._root_scalar._root_scalar_halley_doc'),
        ),
        'linprog': (
            ('simplex', 'scipy.optimize._linprog._linprog_simplex_doc'),
            ('interior-point', 'scipy.optimize._linprog._linprog_ip_doc'),
            ('revised simplex', 'scipy.optimize._linprog._linprog_rs_doc'),
            ('highs-ipm', 'scipy.optimize._linprog._linprog_highs_ipm_doc'),
            ('highs-ds', 'scipy.optimize._linprog._linprog_highs_ds_doc'),
            ('highs', 'scipy.optimize._linprog._linprog_highs_doc'),
        ),
        'quadratic_assignment': (
            ('faq', 'scipy.optimize._qap._quadratic_assignment_faq'),
            ('2opt', 'scipy.optimize._qap._quadratic_assignment_2opt'),
        ),
        'minimize_scalar': (
            ('brent', 'scipy.optimize._optimize._minimize_scalar_brent'),
            ('bounded', 'scipy.optimize._optimize._minimize_scalar_bounded'),
            ('golden', 'scipy.optimize._optimize._minimize_scalar_golden'),
        ),
    }

    if solver is None:
        text = ["\n\n\n========\n", "minimize\n", "========\n"]
        text.append(show_options('minimize', disp=False))
        text.extend(["\n\n===============\n", "minimize_scalar\n",
                     "===============\n"])
        text.append(show_options('minimize_scalar', disp=False))
        text.extend(["\n\n\n====\n", "root\n",
                     "====\n"])
        text.append(show_options('root', disp=False))
        text.extend(['\n\n\n=======\n', 'linprog\n',
                     '=======\n'])
        text.append(show_options('linprog', disp=False))
        text = "".join(text)
    else:
        solver = solver.lower()
        if solver not in doc_routines:
            raise ValueError(f'Unknown solver {solver!r}')

        if method is None:
            text = []
            for name, _ in doc_routines[solver]:
                text.extend(["\n\n" + name, "\n" + "="*len(name) + "\n\n"])
                text.append(show_options(solver, name, disp=False))
            text = "".join(text)
        else:
            method = method.lower()
            methods = dict(doc_routines[solver])
            if method not in methods:
                raise ValueError(f"Unknown method {method!r}")
            name = methods[method]

            # Import function object
            parts = name.split('.')
            mod_name = ".".join(parts[:-1])
            __import__(mod_name)
            obj = getattr(sys.modules[mod_name], parts[-1])

            # Get doc
            doc = obj.__doc__
            if doc is not None:
                text = textwrap.dedent(doc).strip()
            else:
                text = ""

    if disp:
        print(text)
        return
    else:
        return text
<|endoftext|>from typing import TYPE_CHECKING

import numpy as np

from ._optimize import OptimizeResult
from ._pava_pybind import pava

if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = ["isotonic_regression"]


def isotonic_regression(
    y: "npt.ArrayLike",
    *,
    weights: "npt.ArrayLike | None" = None,
    increasing: bool = True,
) -> OptimizeResult:
    r"""Nonparametric isotonic regression.

    A (not strictly) monotonically increasing array `x` with the same length
    as `y` is calculated by the pool adjacent violators algorithm (PAVA), see
    [1]_. See the Notes section for more details.

    Parameters
    ----------
    y : (N,) array_like
        Response variable.
    weights : (N,) array_like or None
        Case weights.
    increasing : bool
        If True, fit monotonic increasing, i.e. isotonic, regression.
        If False, fit a monotonic decreasing, i.e. antitonic, regression.
        Default is True.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are:

        - ``x``: The isotonic regression solution, i.e. an increasing (or
          decreasing) array of the same length than y, with elements in the
          range from min(y) to max(y).
        - ``weights`` : Array with the sum of case weights for each block
          (or pool) B.
        - ``blocks``: Array of length B+1 with the indices of the start
          positions of each block (or pool) B. The j-th block is given by
          ``x[blocks[j]:blocks[j+1]]`` for which all values are the same.

    Notes
    -----
    Given data :math:`y` and case weights :math:`w`, the isotonic regression
    solves the following optimization problem:

    .. math::

        \operatorname{argmin}_{x_i} \sum_i w_i (y_i - x_i)^2 \quad
        \text{subject to } x_i \leq x_j \text{ whenever } i \leq j \,.

    For every input value :math:`y_i`, it generates a value :math:`x_i` such
    that :math:`x` is increasing (but not strictly), i.e.
    :math:`x_i \leq x_{i+1}`. This is accomplished by the PAVA.
    The solution consists of pools or blocks, i.e. neighboring elements of
    :math:`x`, e.g. :math:`x_i` and :math:`x_{i+1}`, that all have the same
    value.

    Most interestingly, the solution stays the same if the squared loss is
    replaced by the wide class of Bregman functions which are the unique
    class of strictly consistent scoring functions for the mean, see [2]_
    and references therein.

    The implemented version of PAVA according to [1]_ has a computational
    complexity of O(N) with input size N.

    References
    ----------
    .. [1] Busing, F. M. T. A. (2022).
           Monotone Regression: A Simple and Fast O(n) PAVA Implementation.
           Journal of Statistical Software, Code Snippets, 102(1), 1-25.
           :doi:`10.18637/jss.v102.c01`
    .. [2] Jordan, A.I., Mühlemann, A. & Ziegel, J.F.
           Characterizing the optimal solutions to the isotonic regression
           problem for identifiable functionals.
           Ann Inst Stat Math 74, 489-514 (2022).
           :doi:`10.1007/s10463-021-00808-0`

    Examples
    --------
    This example demonstrates that ``isotonic_regression`` really solves a
    constrained optimization problem.

    >>> import numpy as np
    >>> from scipy.optimize import isotonic_regression, minimize
    >>> y = [1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0]
    >>> def objective(yhat, y):
    ...     return np.sum((yhat - y)**2)
    >>> def constraint(yhat, y):
    ...     # This is for a monotonically increasing regression.
    ...     return np.diff(yhat)
    >>> result = minimize(objective, x0=y, args=(y,),
    ...                   constraints=[{'type': 'ineq',
    ...                                 'fun': lambda x: constraint(x, y)}])
    >>> result.x
    array([1.25      , 1.25      , 4.        , 5.56666667, 5.56666667,
           5.56666667, 7.8       , 8.25      , 8.25      , 9.25      ,
           9.25      ])
    >>> result = isotonic_regression(y)
    >>> result.x
    array([1.25      , 1.25      , 4.        , 5.56666667, 5.56666667,
           5.56666667, 7.8       , 8.25      , 8.25      , 9.25      ,
           9.25      ])

    The big advantage of ``isotonic_regression`` compared to calling
    ``minimize`` is that it is more user friendly, i.e. one does not need to
    define objective and constraint functions, and that it is orders of
    magnitudes faster. On commodity hardware (in 2023), for normal distributed
    input y of length 1000, the minimizer takes about 4 seconds, while
    ``isotonic_regression`` takes about 200 microseconds.
    """
    yarr = np.atleast_1d(y)  # Check yarr.ndim == 1 is implicit (pybind11) in pava.
    order = slice(None) if increasing else slice(None, None, -1)
    x = np.array(yarr[order], order="C", dtype=np.float64, copy=True)
    if weights is None:
        wx = np.ones_like(yarr, dtype=np.float64)
    else:
        warr = np.atleast_1d(weights)

        if not (yarr.ndim == warr.ndim == 1 and yarr.shape[0] == warr.shape[0]):
            raise ValueError(
                "Input arrays y and w must have one dimension of equal length."
            )
        if np.any(warr <= 0):
            raise ValueError("Weights w must be strictly positive.")

        wx = np.array(warr[order], order="C", dtype=np.float64, copy=True)
    n = x.shape[0]
    r = np.full(shape=n + 1, fill_value=-1, dtype=np.intp)
    x, wx, r, b = pava(x, wx, r)
    # Now that we know the number of blocks b, we only keep the relevant part
    # of r and wx.
    # As information: Due to the pava implementation, after the last block
    # index, there might be smaller numbers appended to r, e.g.
    # r = [0, 10, 8, 7] which in the end should be r = [0, 10].
    r = r[:b + 1]  # type: ignore[assignment]
    wx = wx[:b]
    if not increasing:
        x = x[::-1]
        wx = wx[::-1]
        r = r[-1] - r[::-1]
    return OptimizeResult(
        x=x,
        weights=wx,
        blocks=r,
    )
<|endoftext|># mypy: allow-untyped-defs
import collections
import functools
import warnings
from typing import Any

import torch
from torch.types import _dtype


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
]


def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return torch._C._is_autocast_available(device_type)


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = (  # type: ignore[attr-defined]
        "@autocast() decorator is not supported in script mode"
    )
    return decorate_autocast


class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...

            @torch.autocast(device_type="cuda")
            def forward(self, input): ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.
    CUDA Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with torch.autocast(device_type="cuda"):
            # torch.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    CPU Training Example::

        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()


    CPU Inference Example::

        # Creates model in default precision
        model = Net().eval()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for input in data:
                # Runs the forward pass with autocasting.
                output = model(input)

    CPU Inference Example with Jit Trace::

        class TestModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, num_classes)

            def forward(self, x):
                return self.fc1(x)


        input_size = 2
        num_classes = 2
        model = TestModel(input_size, num_classes).eval()

        # For now, we suggest to disable the Jit Autocast Pass,
        # As the issue: https://github.com/pytorch/pytorch/issues/75956
        torch._C._jit_set_autocast_mode(False)

        with torch.cpu.amp.autocast(cache_enabled=False):
            model = torch.jit.trace(model, torch.randn(1, input_size))
        model = torch.jit.freeze(model)
        # Models Run
        for _ in range(3):
            model(torch.randn(1, input_size))

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with torch.autocast(device_type="cuda"):
            e_float16 = torch.mm(a_float32, b_float32)
            with torch.autocast(device_type="cuda", enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(str, required):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and 'hpu'.
                                     The type is the same as the `type` attribute of a :class:`torch.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(torch_dtype, optional):  Data type for ops run in autocast. It uses the default value
            (``torch.float16`` for CUDA and ``torch.bfloat16`` for CPU), given by
            :func:`~torch.get_autocast_dtype`, if :attr:`dtype` is ``None``.
            Default: ``None``
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: _dtype | None = None,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ):
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        self.fast_dtype = (
            torch.get_autocast_dtype(device_type) if dtype is None else dtype
        )
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            if self.fast_dtype is None:
                raise AssertionError("fast_dtype must not be None in scripting mode")
            return
        self.device = device_type
        if not is_autocast_available(self.device):
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )

        device_supported_dtypes = [torch.bfloat16, torch.float16]

        self.custom_backend_name = torch._C._get_privateuse1_backend_name()
        if self.device == self.custom_backend_name:
            necessary_funcs = [
                "get_amp_supported_dtype",
            ]
            message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
            message += "registered a module or the module misses some necessary funcs. The backend should register "
            message += "a module by `torch._register_device_module`, and the module must have these funcs: \n"
            message += "`get_amp_supported_dtype() -> List[torch.dtype]`. \n"

            if not hasattr(torch, self.custom_backend_name):
                raise AssertionError(message)
            self.custom_device_mod = getattr(torch, self.custom_backend_name)
            for func in necessary_funcs:
                if not hasattr(self.custom_device_mod, func):
                    raise AssertionError(
                        message + f"But the func `{func}` is missing. \n"
                    )
            device_supported_dtypes = self.custom_device_mod.get_amp_supported_dtype()

        self._cache_enabled = (
            torch.is_autocast_cache_enabled()
            if cache_enabled is None
            else cache_enabled
        )

        device_name = (
            self.device
            if self.device == self.custom_backend_name
            else self.device.upper()
        )
        if enabled:
            # Special case for CUDA AMP and bfloat16 support
            if self.device == "cuda":
                if torch.cuda.amp.common.amp_definitely_not_available():
                    warnings.warn(
                        "CUDA is not available or torch_xla is imported. Disabling autocast.",
                        stacklevel=2,
                    )
                    enabled = False
                elif (
                    self.fast_dtype == torch.bfloat16
                    and not torch.cuda.is_bf16_supported()
                ):
                    raise RuntimeError(
                        "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                    )
            elif self.fast_dtype not in device_supported_dtypes:
                error_message = (
                    f"In {device_name} autocast, but the target dtype is not supported. Disabling autocast.\n"
                    f"{device_name} Autocast only supports dtypes of "
                    + ", ".join(map(str, device_supported_dtypes))
                    + " currently."
                )
                warnings.warn(error_message, stacklevel=2)
                enabled = False
        self._enabled = enabled

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            if self.fast_dtype is None:
                raise AssertionError("fast_dtype must not be None in scripting mode")
            return self

        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        self.prev = torch.is_autocast_enabled(self.device)
        self.prev_fastdtype = torch.get_autocast_dtype(self.device)
        torch.set_autocast_enabled(self.device, self._enabled)
        torch.set_autocast_dtype(self.device, self.fast_dtype)  # type: ignore[arg-type]
        torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

        # only dispatch to PreDispatchTorchFunctionMode to avoid exposing this
        # API to other functional modes. We only expose to PreDispatchTorchFunctionMode
        # for preserving autocast in torch.export.export.
        if torch._C._is_torch_function_mode_enabled():
            stacks = torch.overrides._get_current_function_mode_stack()
            for mode in stacks:
                if isinstance(
                    mode,
                    torch.fx.experimental.proxy_tensor.PreDispatchTorchFunctionMode,
                ):
                    args = (
                        self.device,
                        self.fast_dtype,
                        self._enabled,
                        self._cache_enabled,
                    )
                    mode.__torch_function__(torch.amp._enter_autocast, (), args)
                    return self

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return

        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.device, self.prev)
        torch.set_autocast_dtype(self.device, self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)

        # only dispatch to PreDispatchTorchFunctionMode to avoid exposing this
        # API to other functional modes. We only expose to PreDispatchTorchFunctionMode
        # for preserving autocast in torch.export.export.
        if torch._C._is_torch_function_mode_enabled():
            stacks = torch.overrides._get_current_function_mode_stack()
            for mode in stacks:
                if isinstance(
                    mode,
                    torch.fx.experimental.proxy_tensor.PreDispatchTorchFunctionMode,
                ):
                    mode.__torch_function__(torch.amp._exit_autocast, (), ())
                    # This is very important because the above line actually doesn't
                    # run exit code so it end up swallowing exceptions.
                    return False
        return False

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        if not callable(func):
            raise TypeError(
                f"autocast()(func) requires a callable, but got {type(func).__name__}. "
                f"Did you mean to use autocast as a context manager? For example:\n"
                f"    with torch.autocast(device_type=...):\n"
                f"        output = model(input)"
            )
        return autocast_decorator(self, func)


# Subclass to distinguish autocast variables created by _enter_autocast (and not managed by a with statement)
class _UnmanagedAutocast(autocast):
    pass


# These functions aren't meant for public usage.
# They are what we trace into a graph during pre_dispatch tracing
# when we encounter an autocast context manager.
def _enter_autocast(*vals):
    # For pre-dispatch tracing, if a TorchFunction mode is active, we'll want to trace this into a graph.
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(
            torch.amp._enter_autocast, [], *vals
        )
    mode = _UnmanagedAutocast(*vals)
    mode.__enter__()
    return mode


def _exit_autocast(mode):
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(torch.amp._exit_autocast, [], mode)
    mode.__exit__(None, None, None)


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, device_type: str, dtype: _dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(
        value,
        # pyrefly: ignore [missing-attribute]
        np.ndarray,
    ):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def custom_fwd(
    fwd=None,
    *,
    device_type: str,
    cast_inputs: _dtype | None = None,
):
    """
    Create a helper decorator for ``forward`` methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = torch.get_autocast_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)  # pyrefly: ignore [not-callable]
        else:
            autocast_context = torch.is_autocast_enabled(device_type)
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(device_type=device_type, enabled=False):
                    return fwd(  # pyrefly: ignore  # not-callable
                        *_cast(args, device_type, cast_inputs),
                        **_cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return fwd(*args, **kwargs)  # pyrefly: ignore [not-callable]

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """

    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if bwd is None:
        return functools.partial(custom_bwd, device_type=device_type)

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,
            dtype=args[0]._dtype,
        ):
            return bwd(*args, **kwargs)  # pyrefly: ignore [not-callable]

    return decorate_bwd
<|endoftext|># mypy: allow-untyped-defs
from typing_extensions import deprecated as _deprecated


@_deprecated(
    "Usage of `backward_compatibility.worker_init_fn` is deprecated "
    "as `DataLoader` automatically applies sharding in every worker",
    category=FutureWarning,
)
def worker_init_fn(worker_id) -> None:
    pass
<|endoftext|># mypy: allow-untyped-defs
"""Utilities for manipulating the torch.Graph object and the torchscript."""

from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import registration
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


_ATTR_PATTERN = re.compile("^(.+)_(([ifstgz])|(ty))$")
_SKIP_NODE_ATTRIBUTES = {"inplace", "aten"}


@dataclasses.dataclass
class GraphContext:
    """Extra context for symbolic functions with all methods from torch.Graph.

    NOTE: This class is not meant for external consumption. Please do not depend on
    it outside of torch.onnx as the interface may evolve.

    Attributes:
        graph: The _C.Graph being constructed.
        block: The current _C.Block being constructed.
        opset: The opset version.
        original_node: Current node that is being converted from.
        params_dict: Mapping from graph initializer name to IValue.
        env: Mapping from Torch domain graph Value to ONNX domain graph Value.
        values_in_env: Set of all values in env, for constant-time lookups.
        new_nodes: List that tracks all new nodes that are added (used to make
            sure metadata is propagated to all new nodes).
    """

    graph: _C.Graph
    block: _C.Block
    opset: int
    original_node: _C.Node
    params_dict: dict[str, _C.IValue]
    env: dict[_C.Value, _C.Value]
    values_in_env: set[_C.Value]
    new_nodes: list[_C.Node] = dataclasses.field(default_factory=list)

    # Relay methods from _C.Graph for compatibility with symbolic functions that expect
    # a _C.Graph
    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    def op(
        self,
        opname: str,
        *raw_args: torch.Tensor | _C.Value,
        outputs: int = 1,
        **kwargs,
    ):
        """Creates an ONNX operator "opname", taking "raw_args" as inputs and "kwargs" as attributes.

        The set of operators and the inputs/attributes they take
        is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

        Args:
            opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
                with a namespace, e.g., `aten::add`.
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this function returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)

    def aten_op(self, operator: str, *args, overload_name: str = "", **kwargs):
        """Generates an ONNX ATen op node.

        This function is for backward compatibility with the old symbolic functions.
        """
        return self.op(
            "aten::ATen",
            *args,
            operator_s=operator,
            overload_name_s=overload_name,
            **kwargs,
        )

    # NOTE: For backward compatibility with the old symbolic functions.
    # We are probably going to remove this only after the fx exporter is established.
    at = aten_op

    def onnxscript_op(
        self,
        onnx_fn,
        *raw_args: torch.Tensor | _C.Value,
        outputs: int = 1,
        **kwargs,
    ):
        """Creates an ONNX operator from onnx-script function, taking "raw_args" as inputs and "kwargs" as attributes.

        onnx-script repository: https://github.com/microsoft/onnx-script

        Args:
            onnx_fn: ONNXFunction from onnx-script; An example can be found at
                https://github.com/microsoft/onnx-script#example
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this function returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        # NOTE(titaiwang): This is using class attributes, and it needs to be updated
        # if onnx-script makes any change on these.
        symbolic_name = f"{onnx_fn.opset.domain}::{onnx_fn.name}"
        opset_version = onnx_fn.opset.version

        registration.custom_onnx_symbolic(symbolic_name, opset_version)(onnx_fn)

        return _add_op(self, symbolic_name, *raw_args, outputs=outputs, **kwargs)


def add_op_with_blocks(
    graph_context: GraphContext,
    opname: str,
    *inputs: _C.Value,
    outputs: int = 1,
    n_blocks: int = 1,
    **attributes,
) -> tuple[Any, tuple[GraphContext, ...], _C.Node]:
    """Creates an ONNX operator "opname", taking inputs and attributes.

    Args:
        graph_context: The context for the current graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        inputs: The inputs to the operator.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this function returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        n_blocks: The number of sub-blocks to create in the node.
        attributes: The attributes of the ONNX operator.

    Returns:
        A tuple of (output_values, new_contexts, node) where:
            output_values: One or more output value of this operator
                (see the `outputs` keyword argument for multi-return nodes).
            new_contexts: A tuple of new graph contexts for each sub-block.
            node: The node representing the operator.
    """

    output_values = graph_context.op(opname, *inputs, outputs=outputs, **attributes)
    if isinstance(output_values, Sequence):
        node = output_values[0].node()
    else:
        node = output_values.node()

    new_contexts = []
    for _ in range(n_blocks):
        new_block = node.addBlock()
        # Create shallow copy of the graph context and update the block
        new_context = dataclasses.replace(graph_context, block=new_block)
        new_contexts.append(new_context)

    return output_values, tuple(new_contexts), node


def _add_op(
    graph_context: GraphContext,
    opname: str,
    *args: torch.Tensor | _C.Value,
    outputs: int = 1,
    **kwargs,
):
    """Creates an ONNX operator "opname", taking "args" as inputs and attributes "kwargs".

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Args:
        graph_context: The Torch Graph or Block.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this function returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        kwargs: The attributes of the ONNX operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).

    Returns:
        (Union[_C.Value, Tuple[_C.Value, ...]])
        The value representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    inputs = [_const_if_tensor(graph_context, arg) for arg in args]
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in kwargs.items() if v is not None}

    if "::" not in opname:
        opname = "onnx::" + opname

    node = _create_node(
        graph_context.block,
        opname,
        inputs,
        attributes,
        params_dict=graph_context.params_dict,
        opset_version=graph_context.opset,
        n_outputs=outputs,
        shape_inference=GLOBALS.onnx_shape_inference,
    )
    graph_context.new_nodes.append(node)

    if outputs == 1:
        return node.output()
    return tuple(node.outputs())


def _const_if_tensor(graph_context: GraphContext, arg):
    if arg is None:
        return arg
    if isinstance(arg, _C.Value):
        return arg

    return _add_op(graph_context, "onnx::Constant", value_z=arg)


def _create_node(
    graph_or_block: _C.Graph | _C.Block,
    domain_op: str,
    inputs: Sequence,
    attributes: dict,
    params_dict: dict,
    opset_version: int,
    n_outputs: int,
    shape_inference: bool = True,
) -> _C.Node:
    """Creates a node 'domain_op', taking inputs and attributes."""
    if isinstance(graph_or_block, _C.Graph):
        graph = graph_or_block
        node = graph.create(domain_op, inputs, n_outputs)
        node = graph.insertNode(node)
    elif isinstance(graph_or_block, _C.Block):
        block = graph_or_block
        node = block.addNode(domain_op, inputs)

        # Block does not have create defined, so we need to add outputs manually
        if n_outputs > 1:
            for _ in range(1, n_outputs):
                node.addOutput()

    node_outputs = tuple(node.outputs())  # type: ignore[possibly-undefined]
    if len(node_outputs) != n_outputs:
        raise AssertionError(
            f"len(node_outputs)={len(node_outputs)} != n_outputs={n_outputs}"
        )

    aten = domain_op.startswith("aten::")

    # Add all attributes
    for key, value in sorted(attributes.items()):
        if key in _SKIP_NODE_ATTRIBUTES:
            continue
        # pyrefly: ignore [unbound-name]
        _add_attribute(node, key, value, aten=aten)
    if shape_inference:
        # pyrefly: ignore [unbound-name]
        _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    # pyrefly: ignore [unbound-name]
    return node


def _is_onnx_list(value):
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, torch.Tensor)
    )


def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    if x.numel() != 1:
        raise AssertionError(f"Expected numel() == 1, got {x.numel()}")
    return x[0]


def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    m = _ATTR_PATTERN.match(key)
    if m is None:
        raise ValueError(
            f"Invalid attribute specifier '{key}' names "
            "must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
        )
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"

    return getattr(node, f"{kind}_")(name, value)


def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


def get_device_from_value(value: _C.Value) -> torch.device | None:
    if not _is_tensor(value):
        return None
    tensor_type = typing.cast(_C.TensorType, value.type())
    return tensor_type.device()


def parse_node_kind(kind: str) -> tuple[str, str]:
    """Parse node kind into domain and Op name."""
    if "::" not in kind:
        raise ValueError(f"Node kind: {kind} is invalid. '::' is not in node kind.")
    domain, opname = kind.split("::", 1)
    if "::" in opname:
        raise ValueError(f"Node kind: {kind} is invalid. '::' should only appear once.")
    return domain, opname


def is_aten(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "aten"


def is_prim(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "prim"


def is_onnx(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "onnx"
<|endoftext|># mypy: allow-untyped-defs
"""
This file includes private common utilities for FSDP.
"""

import dataclasses
import logging
import traceback
import warnings
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator
from enum import auto, Enum
from functools import partial
from itertools import chain
from typing import Any, cast, no_type_check, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch

from .api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    OptimStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

    from ._flat_param import FlatParamHandle


_MAX_TRAVERSE_DEPTH = 128


def _is_namedtuple(obj: Any) -> bool:
    # Mirrors torch.nn.parallel.scatter_gather._is_namedtuple
    fields = getattr(type(obj), "_fields", None)
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and isinstance(fields, tuple)
        and all(isinstance(f, str) for f in fields)
    )


def collect_grad_tensors(output: Any) -> tuple[torch.Tensor, ...]:
    """
    Recursively collect tensors that require gradients from a nested structure.

    Traverses dict, list, tuple, NamedTuple, and dataclass containers.
    Sets and other iterables are *not* traversed (consistent with
    ``tree_flatten``).  Uses the same traversal order as
    :func:`replace_grad_tensors`.
    """
    tensors_list: list[torch.Tensor] = []
    _collect_grad_tensors(output, tensors_list)
    return tuple(tensors_list)


def _collect_grad_tensors(
    output: Any, out: list[torch.Tensor], _depth: int = 0
) -> None:
    """Collect grad-requiring tensors in the same order as _replace_grad_tensors."""
    if _depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"collect_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH}), "
            "likely due to a circular reference in the output structure"
        )
    # Branch order must mirror _replace_grad_tensors exactly.
    # Only dict, list, tuple, NamedTuple, and dataclass are traversed;
    # set and other iterables are intentionally skipped (matching tree_flatten).
    if torch.is_tensor(output) and output.requires_grad:
        out.append(output)
    elif _is_namedtuple(output):
        # NamedTuple before dataclass to match _replace_grad_tensors ordering.
        for item in output:
            _collect_grad_tensors(item, out, _depth + 1)
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        for field in dataclasses.fields(output):
            _collect_grad_tensors(getattr(output, field.name), out, _depth + 1)
    elif isinstance(output, dict):
        for v in output.values():
            _collect_grad_tensors(v, out, _depth + 1)
    elif isinstance(output, (list, tuple)):
        for item in output:
            _collect_grad_tensors(item, out, _depth + 1)


def replace_grad_tensors(output: Any, tensor_iter: Iterator[torch.Tensor]) -> Any:
    """
    Replace grad-requiring tensors in a nested structure using replacements
    from tensor_iter.

    Tensors are consumed from tensor_iter in the same traversal order as
    :func:`collect_grad_tensors`. Traverses dict, list, tuple, NamedTuple,
    and dataclass containers; sets and other iterables are *not* traversed
    (consistent with ``tree_flatten``).

    Note: dataclass reconstruction uses ``dataclasses.replace()``, which calls
    ``__init__``. Dataclasses with custom ``__init__`` validation,
    ``__post_init__`` side effects, or non-standard dict subclass constructors
    may not be compatible. In practice, FSDP module outputs are expected to be
    shallowly nested, so recursion depth is not a concern.
    """
    result = _replace_grad_tensors(output, tensor_iter)
    sentinel = object()
    leftover = next(tensor_iter, sentinel)
    if leftover is not sentinel:
        # Count remaining without holding references to all of them
        n = 1 + sum(1 for _ in tensor_iter)
        raise RuntimeError(
            f"{n} replacement tensors were not consumed while processing "
            f"{type(output).__qualname__}"
        )
    return result


def _replace_grad_tensors(
    output: Any, tensor_iter: Iterator[torch.Tensor], _depth: int = 0
) -> Any:
    # Branch order must mirror _collect_grad_tensors exactly.
    if _depth >= _MAX_TRAVERSE_DEPTH:
        raise RuntimeError(
            f"replace_grad_tensors exceeded max depth ({_MAX_TRAVERSE_DEPTH}), "
            "likely due to a circular reference in the output structure"
        )
    if torch.is_tensor(output) and output.requires_grad:
        return next(tensor_iter)
    elif _is_namedtuple(output):
        # NamedTuple before dataclass: a NamedTuple that is also a dataclass
        # should be reconstructed via positional args, not dataclasses.replace.
        new_items = []
        any_changed = False
        for item in output:
            new_item = _replace_grad_tensors(item, tensor_iter, _depth + 1)
            new_items.append(new_item)
            if new_item is not item:
                any_changed = True
        if any_changed:
            return type(output)(*new_items)
        return output
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        changes = {}
        for field in dataclasses.fields(output):
            old_val = getattr(output, field.name)
            new_val = _replace_grad_tensors(old_val, tensor_iter, _depth + 1)
            if new_val is not old_val:
                changes[field.name] = new_val
        if changes:
            try:
                return dataclasses.replace(output, **changes)
            except TypeError as e:
                raise TypeError(
                    f"Failed to reconstruct dataclass {type(output).__qualname__} "
                    f"via dataclasses.replace(). Dataclasses used as FSDP module "
                    f"inputs/outputs must support dataclasses.replace(): {e}"
                ) from None
        return output
    elif isinstance(output, dict):
        new_dict = {}
        any_changed = False
        for k, v in output.items():
            new_v = _replace_grad_tensors(v, tensor_iter, _depth + 1)
            new_dict[k] = new_v
            if new_v is not v:
                any_changed = True
        if any_changed:
            return new_dict if type(output) is dict else type(output)(new_dict)
        return output
    elif isinstance(output, (list, tuple)):
        new_items = []
        any_changed = False
        for item in output:
            new_item = _replace_grad_tensors(item, tensor_iter, _depth + 1)
            new_items.append(new_item)
            if new_item is not item:
                any_changed = True
        if any_changed:
            typ = type(output)
            try:
                return typ(new_items)
            except TypeError:
                # Fall back to base type for subclasses with custom __init__
                return list(new_items) if isinstance(output, list) else tuple(new_items)
        return output
    else:
        return output


FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "."
FSDP_FLATTENED = "_fsdp_flattened"

# Save a global mapping from module to its input tensor dtype to be populated
# during the forward pre-hook and consumed in the forward post-hook when
# overriding a module's mixed precision
# NOTE: We currently take the last input tensor's dtype in the case of multiple
# floating-point input tensors, which may be incorrect. However, since there is
# not a 1:1 correspondence between input and output tensors, we must use *some*
# heuristic like this to predict the desired output dtype.
_MODULE_TO_INP_DTYPE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class _FSDPDeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any = None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError as exc:
                raise AttributeError(
                    f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'."
                ) from exc
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> "_FSDPDeviceHandle":
        """
        Return a device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
        if device.type == "cuda":
            return cast(_FSDPDeviceHandle, torch.cuda)
        elif device.type == "mtia":
            return cast(_FSDPDeviceHandle, torch.mtia)
        return cls(device)

    def __getattr__(self, name: str, /) -> Any:
        try:
            return getattr(self.__backend, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{name}'"
            ) from exc


class _UninitializedDeviceHandle(_FSDPDeviceHandle):
    def __init__(self) -> None:
        pass

    def __getattribute__(self, name: str, /) -> Any:
        raise RuntimeError("Trying to use an uninitialized device handle.")


class _FSDPState(_State):
    def __init__(self) -> None:
        # TODO: Move all the attributes to this class to enable typing for
        # FSDP/fully_shard.
        self._ignored_modules: set[nn.Module] = set()
        self._ignored_params: set[nn.Parameter] = set()
        # Buffer names are cleaned (without wrapper prefixes)
        self._ignored_buffer_names: set[str] = set()
        self.process_group: dist.ProcessGroup | None = None
        self.rank: int = -1
        self.world_size: int = -1
        self._device_mesh: DeviceMesh | None = None
        self.sharding_strategy = ShardingStrategy.FULL_SHARD
        self._use_orig_params: bool = False
        self.training_state = TrainingState.IDLE
        self._unshard_params_ctx: dict[nn.Module, Generator] = {}
        self._state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._is_root: bool | None = None
        self._handle: flat_param_file.FlatParamHandle | None = None
        self._fully_sharded_module_to_handle: dict[
            nn.Module, flat_param_file.FlatParamHandle | None
        ] = {}
        self.compute_device: torch.device | None = None
        self._gradient_predivide_factor: int = 0
        self._gradient_postdivide_factor: int = 0
        self._comm_hook: Callable | None = None
        self._comm_hook_state: Any | None = None
        self._unshard_event: torch.Event | None = None
        # Abstract device handle for fsdp compute device. For now,
        # the compute device must implement cuda semantics used by fsdp
        self._device_handle: _FSDPDeviceHandle = _UninitializedDeviceHandle()
        # All following attributes should only be used for root states:
        # Save these static lists to avoid the repeated tree traversals
        self._all_fsdp_states: list[_FSDPState] = []
        self._all_handles: list[flat_param_file.FlatParamHandle] = []
        self._fsdp_extension: FSDPExtensions | None = None


def _get_module_fsdp_state(module: nn.Module) -> _FSDPState | None:
    state = _get_module_state(module)
    if state is None or not isinstance(state, _FSDPState):
        return None
    return state


def _get_module_fsdp_state_if_fully_sharded_module(
    module: nn.Module,
) -> _FSDPState | None:
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state == module:  # FullyShardedDataParallel module case.
        return state
    if module in state._fully_sharded_module_to_handle:  # fully_shard case.
        return state
    return None


class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


def _is_composable(state: _FSDPState):
    # TODO: This is a temporary hack for differentiate between code paths.
    return not isinstance(state, nn.Module)


@no_type_check
def _module_handle(state: _FSDPState, module: nn.Module) -> Optional["FlatParamHandle"]:
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
    if _is_composable(state):
        # A valid FSDP state may have no managed parameters and hence no
        # handles, meaning no entry in `_fully_sharded_module_to_handles`
        if state._handle is None:
            return None
        if module not in state._fully_sharded_module_to_handle:
            raise AssertionError(
                f"Expects a fully sharded module but got {module} on rank {state.rank}"
            )
        return state._fully_sharded_module_to_handle[module]
    else:
        # NOTE: This assumes `module` is a `FullyShardedDataParallel` instance.
        return module._handle


@no_type_check
def _has_fsdp_params(state: _FSDPState, module: nn.Module) -> bool:
    """Returns if ``module`` has parameters managed by FSDP."""
    return _module_handle(state, module) is not None


def _get_sharding_strategy(handle):
    """
    Returns the sharding strategy of the handle.
    """
    return handle._sharding_strategy if handle else None


def clean_tensor_name(tensor_name: str) -> str:
    """
    Cleans the parameter or buffer name by removing any module wrapper
    prefixes.
    """
    tensor_name = tensor_name.replace(FSDP_PREFIX, "")
    # TODO: Explicitly replacing the checkpoint wrapper prefix is not ideal as
    # it couples `CheckpointWrapper` and FSDP and also does not scale for more
    # module wrappers.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX, "")
    return tensor_name


def _set_fsdp_flattened(tensor: torch.Tensor) -> None:
    """
    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to
    avoid re-flattening it during nested construction.
    """
    setattr(tensor, FSDP_FLATTENED, True)


def _is_fsdp_flattened(tensor: torch.Tensor) -> bool:
    """Returns if ``tensor`` has been marked as flattened by FSDP."""
    return getattr(tensor, FSDP_FLATTENED, False)


def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> list[tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    if "remove_duplicate" in kwargs:
        raise AssertionError(
            "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
        )
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret


def _get_param_to_fqns(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> dict[nn.Parameter, list[str]]:
    """
    Constructs a mapping from parameter to a list of its \"canonical\" FQNs. Here,
    we use canonical to mean the fully-qualified name assigned to the parameter
    based on its position in the original nn.Module hierarchy before any wrapper
    or parallelism has been applied to it. This is in contrast to FQNs that may be
    generated after parallelisms or wrappers have been applied to the model.

    Each normal parameter maps to a singleton list containing its FQN, while each
    ``FlatParameter`` maps to a list of its original parameter FQNs, which may
    have length greater than one.  All FQNs are prefixed starting from ``model``.

    In the case where FSDP was applied with ``use_orig_params=True``, there should be no
    ``FlatParameter`` s registered to the model's modules and this mapping will only
    contain mappings from ``nn.Parameter`` s to singleton FQN lists.

    It is only in the case where FSDP was applied with ``use_orig_params=False`` where
    a ``FlatParameter`` will be registered in place of the original parameters and there
    will be mappings from each ``FlatParameter`` to lists of FQNs corresponding to the
    original parameters.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): For shared parameters, if ``True``, only
            includes the FQNs corresponding to the first encounter of the
            shared parameter in the module traversal; if ``False``, then
            includes the FQNs across all encounters. (Default: ``True``)
    """

    def module_fn(module, prefix, tree_level, param_to_fqns):
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            local_fqns = (
                param._fqns
                if isinstance(param, flat_param_file.FlatParameter)
                else [param_name]
            )  # prefixed from `module`
            global_fqns = [
                clean_tensor_name(prefix + name) for name in local_fqns
            ]  # prefixed from the top level `model` (i.e. including `prefix`)
            is_shared_param = param in param_to_fqns
            if not is_shared_param:
                param_to_fqns[param] = global_fqns
            else:
                if isinstance(param, flat_param_file.FlatParameter):
                    # DMP overwrites `named_parameters` and skip (advance to
                    # the next child module) the wrapped_module (e.g.,
                    # _dmp_wrapped_module and _fsdp_wrapped_module). When a user
                    # calls `named_child` to traverse the module recursively and
                    # calls `named_parameters` with `recurse=False`, parameters
                    # will be traversed more than once.
                    # This hack is specified designed for DMP + FSDP. We
                    # overwrite the flat_parameters traversal result to only obtain
                    # the last one, which happens to be the correct one.
                    #
                    # TODO: Remove this hack once DMP + FSDP is not supported.
                    warnings.warn(
                        "FlatParameter is being traversed more than once. "
                        "This case should only happen when using "
                        "DistributedModelParallel with FullyShardedDataParallel.",
                        stacklevel=2,
                    )
                    param_to_fqns[param] = global_fqns
                elif not dedup_shared_params:
                    param_to_fqns[param].extend(global_fqns)

    def return_fn(param_to_fqns):
        return param_to_fqns

    param_to_unflat_param_names: dict[torch.nn.Parameter, list[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in _named_parameters_with_duplicates(model)],
        param_to_unflat_param_names,
    )


@no_type_check
def _log_post_backward_hook(
    state: _FSDPState, handle: "FlatParamHandle", logger: logging.Logger
) -> None:
    # Under TORCH_DISTRIBUTED_DEBUG=INFO, log the module names this hook fires for.
    # Below logging of module names this post-bwd hook fires for can help debug certain
    # cases where hooks don't fire, such as under certain activation checkpoint configs.
    if state._use_orig_params and handle._debug_level == dist.DebugLevel.INFO:
        param_fqns = _get_handle_fqns_from_root(state, handle)
        logger.warning("FSDP firing post-backward hooks for parameters %s", param_fqns)


@no_type_check
def _get_handle_fqns_from_root(
    state: _FSDPState, handle: "FlatParamHandle"
) -> list[str] | None:
    if handle is None:
        return None
    param_to_fqn = state._exec_order_data.param_to_fqn
    handle_params = handle.flat_param._params  # only populated for use_orig_params
    param_fqns = [*chain.from_iterable(param_to_fqn[p] for p in handle_params)]
    return param_fqns


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    filter_fqns: list[str] | None = None,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.

    ``filter_fqns`` is used because some module may have its own prefix similar
    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten
    to remove the prefix.
    """

    # Precompute the set of all prefixes from filter_fqns so that the
    # "any FQN starts with new_prefix?" check is O(1) instead of O(N).
    filter_prefixes: set[str] | None = None
    if filter_fqns is not None:
        filter_prefixes = set()
        for fqn in filter_fqns:
            i = fqn.find(".")
            while i != -1:
                filter_prefixes.add(fqn[: i + 1])
                i = fqn.find(".", i + 1)

    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is None:
                continue
            new_prefix = prefix + submodule_name + "."
            new_tree_level = tree_level + 1
            if filter_prefixes is not None:
                if new_prefix not in filter_prefixes:
                    # DMP's named_parameter() will mess up the traversal with
                    # ``named_children`` + `named_parameter(recurse=False)``.
                    # This hack is a must to make the traversal work.
                    # TODO: Remove this hack once DMP + FSDP is not supported.
                    # It turns out that recursive wrapping may trigger this as
                    # well.
                    if (
                        submodule_name == "_fsdp_wrapped_module"
                        or submodule_name == "_dmp_wrapped_module"
                    ):
                        new_prefix = prefix
                    elif submodule_name == "module":
                        new_prefix = prefix
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)

    f(root_module, "", 0, *args, **kwargs)
    return return_fn(*args, **kwargs)


@no_type_check
def _assert_in_training_states(
    state: _FSDPState,
    training_states: list[TrainingState],
) -> None:
    """Asserts that FSDP is in the states ``_training_states``."""
    # Raise a `ValueError` instead of using `assert` to ensure that these
    # logical assertions run even if `assert`s are disabled
    if state.training_state not in training_states:
        msg = (
            f"expected to be in states {training_states} but current state is "
            f"{state.training_state}"
        )
        # Print the error on rank 0 in case this is called in the backward pass
        if state.rank == 0:
            if isinstance(state, nn.Module):
                print(f"Asserting FSDP instance is: {state}")
            print(f"ERROR: {msg}")
            traceback.print_stack()
        raise ValueError(msg)


def _get_root_modules(modules: set[nn.Module]) -> set[nn.Module]:
    """
    Returns:
        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.
        parent-less) with respect to the modules in the set itself. In other
        words, these are the modules in ``modules`` that are not the child of
        any other module in ``modules``.
    """
    root_modules: set[nn.Module] = set()
    module_to_submodules = {module: set(module.modules()) for module in modules}
    for candidate_module in modules:
        is_root_module = True
        for module, submodules in module_to_submodules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in submodules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.add(candidate_module)
    return root_modules


def _override_module_mixed_precision(
    root: torch.nn.Module,
    module_classes_to_override: Iterable[type[nn.Module]],
    wrap_override_dict: dict[str, Any] = {"mixed_precision": None},  # noqa: B006
) -> set[type[nn.Module]]:
    module_classes_to_override = tuple(set(module_classes_to_override))
    # Return a set of the actually overridden module classes
    overridden_module_classes: set[type[nn.Module]] = set()
    for mod in root.modules():
        if isinstance(mod, module_classes_to_override):
            overridden_module_classes.add(type(mod))
            mod._wrap_overrides = wrap_override_dict  # type: ignore[assignment]
            # TODO: We need to run this mixed precision ignored module in fp32,
            # but ensure subsequent modules, that may possibly be running with
            # mixed precision, still receive the appropriate precision inputs
            # without user having to adjust mixed precision config too much.
            # As a result, we attach pre and post forward hooks to up / down
            # cast. We should revisit this design.

            def cast_fn(
                dtype: torch.dtype, module: nn.Module, x: torch.Tensor
            ) -> torch.Tensor:
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                _MODULE_TO_INP_DTYPE[module] = x.dtype
                return x.to(dtype)

            def forward_pre_hook(module, args):
                return _apply_to_tensors(partial(cast_fn, torch.float32, module), args)

            def forward_post_hook(module, args, output):
                # NOTE: If the forward did not have any floating-point tensors,
                # then the dtype will not be set for this module, and we do not
                # upcast the dtype.
                if module in _MODULE_TO_INP_DTYPE:
                    old_dtype = _MODULE_TO_INP_DTYPE[module]
                    return _apply_to_tensors(
                        partial(cast_fn, old_dtype, module), output
                    )

            # We intentionally append both of these hooks so that they run after
            # all other hooks.
            mod.register_forward_pre_hook(forward_pre_hook, prepend=False)
            mod.register_forward_hook(forward_post_hook, prepend=False)
    return overridden_module_classes


def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    # FIXME record_stream doesn't work with non-cuda/mtia/xpu tensors
    if tensor.device.type not in [
        "cuda",
        "mtia",
        "xpu",
        torch._C._get_privateuse1_backend_name(),
    ]:
        return

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        return
        # from @ezyang:
        # The no_dispatch was added in https://github.com/pytorch/pytorch/pull/88014 cc @fegin
        # Looking over the PR, it looks like this is because we don't actually support Stream arguments
        # in torch dispatch, so it just chokes.
        # If Dynamo is able to answer "are there any torch dispatch modes" active (it should answer False),
        # a better version of this would just be to check if there are any modes before disabling dispatch.
        # TODO(voz): Extend a dynamo util to answer the above, unify the codepaths here.
        tensor.record_stream(stream)
    else:
        with no_dispatch():
            tensor.record_stream(stream)
<|endoftext|>"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""

import abc
import ast
import contextlib
import dataclasses
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import os
import pickle
import platform
import shutil
import sys
import types
from collections.abc import Callable, Generator, Iterator
from contextlib import nullcontext
from typing import Any, NewType, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import torch
from torch._dynamo.exc import PackageError
from torch._dynamo.graph_utils import _graph_device_type
from torch.utils.weak import WeakIdKeyDictionary

from .bytecode_transformation import (
    COMPILED_FN_PREFIX,
    get_code_keys,
    is_compiled_fn_name,
)
from .utils import CleanupHook, counters, dynamo_timed, increment_frame


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper, GuardsState


_CODE_CACHE = WeakIdKeyDictionary()


def _code_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    def _(
        cls: type[Any], code: Union["SerializedCode", types.CodeType]
    ) -> Union["SerializedCode", types.CodeType]:
        if code in _CODE_CACHE:
            return _CODE_CACHE[code]
        res = fn(cls, code)
        _CODE_CACHE[code] = res
        return res

    return _


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = None
    co_qualname: str | None = None
    co_exceptiontable: bytes | None = None
    co_lnotab: str | None = None

    @classmethod
    @_code_cache
    def from_code_object(cls, code: types.CodeType) -> "SerializedCode":
        kwargs = {key: getattr(code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.from_code_object(c) if isinstance(c, types.CodeType) else c
            for c in kwargs["co_consts"]
        )
        return cls(**kwargs)

    @classmethod
    @_code_cache
    def to_code_object(cls, serialized_code: "SerializedCode") -> types.CodeType:
        kwargs = {key: getattr(serialized_code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.to_code_object(c) if isinstance(c, SerializedCode) else c
            for c in kwargs["co_consts"]
        )
        return types.CodeType(
            *kwargs.values(),
        )


@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    """
    Contains the serializable information associated with a single compilation in dynamo.
    To restore an execution of compiled code, we will need to serialize the following data:
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """

    guards_state: bytes
    dynamo_code: SerializedCode


def load_guards_state(guards_state: bytes) -> Any:
    try:
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard

        ctx = _fully_shard.disable_fsdp_module_new_init()
    except ImportError:
        ctx = nullcontext()  # type: ignore[assignment]
    with ctx:
        return pickle.loads(guards_state)


def load_guard_manager(
    guards_state: "GuardsState",
    target_code: types.CodeType,
    runtime_global_scope: Any,
) -> "GuardManagerWrapper":
    from .output_graph import OutputGraphCommon

    return torch._dynamo.guards.CheckFunctionManager(
        target_code,
        OutputGraphCommon(guards_state.output_graph),
        shape_code_parts=guards_state.shape_code_parts,
        runtime_global_scope=runtime_global_scope,
        guard_build_local_state=getattr(guards_state, "local_state", None),
    ).guard_manager


_BackendId = NewType("_BackendId", str)  # __compiled_fn
_FunctionId = NewType("_FunctionId", str)  # __resume_at


def _backend_ids_from_code(code: types.CodeType) -> Iterator[_BackendId]:
    for name in code.co_names:
        if is_compiled_fn_name(name):
            yield _BackendId(name)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _backend_ids_from_code(const)


@dataclasses.dataclass(frozen=True)
class InlinedSource:
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str
    content: str


@functools.cache
def _get_module_content(module: types.ModuleType) -> str:
    return inspect.getsource(module)


@dataclasses.dataclass
class SourceInfo:
    inlined_sources: set[InlinedSource]

    def add_code(self, code: types.CodeType) -> None:
        module = inspect.getmodule(code)
        if module is None:
            return
        sourcelines, firstlineno = inspect.getsourcelines(code)
        lastlineno = firstlineno + len(sourcelines)
        source = "".join(sourcelines)
        if source != "".join(_get_sourcelines(module, firstlineno, lastlineno)):
            raise AssertionError(
                f"Source mismatch for {module.__name__} "
                f"(line {firstlineno}-{lastlineno})"
            )
        self.inlined_sources.add(
            InlinedSource(
                module=module.__name__,
                firstlineno=firstlineno,
                lastlineno=lastlineno,
                checksum=_hash_source(source),
                content=_get_module_content(module),
            )
        )


@dataclasses.dataclass
class _DynamoCodeCacheEntry:
    """
    Contains the serializable information associated with a single code object
    in dynamo. To restore an execution of compiled code, we will need the following
    ingredients:
      1. The "original" code object, which serves as the entry point for eager
         execution, i.e. the code only executed when there's no cache entry hit.
      2. The python module name this code object belongs to, for identifying the
         enclosing global scope to inject compiled and resume functions.
      3. A list of function names that pointing to this code object. There could be
         multiple function objects pointing to the same code such as recursive functions.
      4. A list of guarded code that eval frame dispatches to.
      5. A list of imported module objects unioned from all compiled branches.
      6. A list of "backends" (compiled fx graph) unioned from all compiled branches.
      7. A string path used to access the original code object users defined.
         A code object can be accessed by "{python_module}.{function_name}.{code_source}" .
      8. A boolean flag indicating whether the function is installed to global scope.
      9. A boolean flag indicating whether the function has a compile id.
      10. Whether or not this code entry was bypassed
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]
    code_source: str | None
    install_to_global: bool
    has_compile_id: bool = False
    bypassed: bool = False


def _lookup_code(entry: _DynamoCodeCacheEntry) -> types.CodeType:
    if len(entry.function_names) != 1:
        raise AssertionError(
            f"Expected exactly one function name, got {len(entry.function_names)}"
        )
    fn: Any = sys.modules[entry.python_module]
    parts = entry.function_names[0].split(".")
    for part in parts:
        fn = getattr(fn, part)
    if entry.code_source:
        parts = entry.code_source.split(".")
        for part in parts:
            if part.endswith("]"):
                index_begin = part.rfind("[")
                if not isinstance(index_begin, int):
                    raise AssertionError(
                        f"Expected int for index_begin, got {type(index_begin)}"
                    )
                if index_begin < 0:
                    raise AssertionError(
                        f"Expected non-negative index_begin, got {index_begin}"
                    )
                attr = getattr(fn, part[:index_begin], None)
                if attr is None:
                    raise PackageError(f"Cannot find source for code entry {entry}")
                fn = attr[ast.literal_eval(part[index_begin + 1 : -1])]
            else:
                fn = getattr(fn, part)
    else:
        raise PackageError(f"Cannot find source for code entry {entry}")
    if not isinstance(fn, types.CodeType):
        raise AssertionError(
            f"Expected CodeType, got {type(fn)} for code entry {entry}"
        )
    return fn


def _raise_resolution_error(code: types.CodeType, scope: Any) -> Never:
    raise PackageError(
        f"Cannot resolve a fully qualified name for {code}. Lookup scope: {scope}"
    )


def _get_code_source(code: types.CodeType) -> tuple[str, str]:
    """
    Given a code object, return a fully qualified name which will be used as
    a serialized handle to access the code object from the new process.
    This is normally a straightforward process, but there are some corner cases:
    1. When a function is defined with decorator, then this function will be captured
       inside a closure with the wrapper object.
    2. When a function is defined as a nested function, then the code object will be
       stored on the co_consts field of the parent code object by Python compiler.
    This function handles all of the corner cases above.
    """

    module = inspect.getmodule(code)
    if module is None:
        raise PackageError(f"Cannot find module for code {code}")

    toplevel: Any = module
    if sys.version_info >= (3, 11):
        parts = code.co_qualname.split(".")

        for part in parts:
            if not hasattr(toplevel, part):
                _raise_resolution_error(code, toplevel)
            toplevel = getattr(toplevel, part)
            if inspect.isfunction(toplevel) or inspect.ismethod(toplevel):
                break
    seen = set()

    def _find_code_source(obj: Any) -> str | None:
        nonlocal toplevel
        nonlocal seen
        if obj in seen:
            return None

        seen.add(obj)

        if inspect.iscode(obj):
            if obj is code:
                return ""

            for i, const in enumerate(obj.co_consts):
                if (res := _find_code_source(const)) is not None:
                    return f".co_consts[{i}]{res}"

        if inspect.ismethod(obj):
            if (res := _find_code_source(obj.__func__)) is not None:
                toplevel = obj
                return f".__func__{res}"

        if inspect.isfunction(obj):
            if (res := _find_code_source(obj.__code__)) is not None:
                toplevel = obj
                return f".__code__{res}"
            if obj.__closure__ is not None:
                for i, cell in enumerate(obj.__closure__):
                    try:
                        cell_contents = cell.cell_contents
                    except ValueError:
                        continue
                    if not (
                        inspect.isfunction(cell_contents)
                        or inspect.iscode(cell_contents)
                        or inspect.ismethod(cell_contents)
                    ):
                        continue
                    if (res := _find_code_source(cell_contents)) is not None:
                        toplevel = obj
                        return f".__closure__[{i}].cell_contents{res}"

        if sys.version_info < (3, 11):
            if inspect.ismodule(obj):
                for value in obj.__dict__.values():
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        return res

            if inspect.isclass(obj):
                for name in itertools.chain(obj.__dict__.keys(), dir(obj)):
                    try:
                        value = getattr(obj, name)
                    except AttributeError:
                        continue
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        if value.__name__ != name:
                            _raise_resolution_error(code, toplevel)
                        return res
        return None

    code_source = _find_code_source(toplevel)
    if code_source is None:
        _raise_resolution_error(code, toplevel)
    return toplevel.__qualname__, code_source.strip(".")


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """
    System information including Python, PyTorch, and GPU details.
    This information is used to ensure compiled artifacts can only be loaded
    with compatible system configurations.
    """

    python_version: str
    torch_version: str
    toolkit_version: str | None
    triton_version: tuple[int, int] | None
    gpu_name: str | None
    CHECK_GPUS = ("cuda", "xpu")

    @classmethod
    def current(cls) -> "SystemInfo":
        """Create a SystemInfo instance with current system information."""
        # Get GPU name if CUDA or XPU is available
        gpu_name = None
        from torch.utils._triton import get_triton_version

        gpu_name, toolkit_version = None, None
        for device_type in cls.CHECK_GPUS:
            if getattr(torch, device_type).is_available():
                try:
                    gpu_name = getattr(torch, device_type).get_device_name()
                    toolkit_version = getattr(torch.version, device_type)
                    break
                except Exception:
                    pass

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            toolkit_version=toolkit_version,
            triton_version=get_triton_version((0, 0)),
            gpu_name=gpu_name,
        )

    def check_compatibility(
        self, other: "SystemInfo", device_type: str = "cpu"
    ) -> None:
        """
        Check if this SystemInfo is compatible with another SystemInfo.
        Raises RuntimeError if incompatible.
        """
        if self.python_version != other.python_version:
            raise RuntimeError(
                f"Compile package was created with a different Python version: {self.python_version}"
            )

        if self.torch_version != other.torch_version:
            raise RuntimeError(
                f"Compile package was created with a different PyTorch version: {self.torch_version}"
            )
        if device_type in self.CHECK_GPUS:
            if not getattr(torch, device_type).is_available():
                raise RuntimeError(f"{device_type} is not available")

            if self.toolkit_version != other.toolkit_version:
                raise RuntimeError(
                    f"Compile package was created with a different toolkit version: {self.toolkit_version}"
                )

            if (
                other.triton_version != (0, 0)
                and self.triton_version != other.triton_version
            ):
                raise RuntimeError(
                    f"Compile package was created with a different Triton version: {self.triton_version}"
                )

            # Check GPU name if CUDA/XPU was used
            if other.gpu_name is not None and self.gpu_name != other.gpu_name:
                raise RuntimeError(
                    f"Compile package was created with different GPU: "
                    f"cached={self.gpu_name}, current={other.gpu_name}"
                )


@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
    source_info: SourceInfo
    device_type: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)
    fn_name: str | None = None
    fn_first_lineno: str | None = None

    @property
    def backend_ids(self) -> set[_BackendId]:
        return {backend_id for code in self.codes for backend_id in code.backend_ids}

    def check_versions(self) -> None:
        """Check if the current system is compatible with the system used to create this cache entry."""
        current_system_info = SystemInfo.current()
        self.system_info.check_compatibility(current_system_info, self.device_type)

    def debug_info(self) -> dict[str, Any]:
        if len(self.codes) == 0:
            raise AssertionError("Expected at least one code entry")
        return {
            "num_codes": str(len(self.codes)),
            "fn_name": self.fn_name,
            "fn_first_lineno": self.fn_first_lineno,
            "device_type": self.device_type,
            "backend_ids": list(self.backend_ids),
        }


from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)


@CacheArtifactFactory.register
class PrecompileCacheArtifact(CacheArtifact):
    def populate_cache(self) -> None:
        DynamoCache._write_to_local_cache(self.content, self.key)

    @staticmethod
    def type() -> str:
        return "precompile"


@dataclasses.dataclass
class PrecompileCacheEntry:
    """
    A full cache entry for caching precompile, for a toplevel torch.compile.
    Consists of a _DynamoCacheEntry, which contains all the dynamo related contents,
    and a set of backends content. In general, the backend content here will always
    be of type precompile_context.BackendCacheArtifact
    """

    dynamo: _DynamoCacheEntry
    backends: dict[_BackendId, Any]

    @staticmethod
    def from_cache_entry(
        cache_entry: _DynamoCacheEntry, backends: dict[_BackendId, Any]
    ) -> Optional["PrecompileCacheEntry"]:
        backend_content: dict[_BackendId, Any] = {}

        for code in cache_entry.codes:
            for backend_id in code.backend_ids:
                if backend_id not in backends:
                    logger.warning("Backend not found")
                    debug_str = json.dumps(
                        {
                            "entry": cache_entry.debug_info(),
                            "missing_backend": backend_id,
                        }
                    )
                    torch._logging.trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "dynamo_cache_bypass",
                            "encoding": "json",
                        },
                        payload_fn=lambda: debug_str,
                        expect_trace_id=False,
                    )
                    code.bypassed = True
                    break
                else:
                    backend_content[backend_id] = backends[backend_id]

        return PrecompileCacheEntry(dynamo=cache_entry, backends=backend_content)


def _hash_source(source: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(source.encode())
    return sha256_hash.hexdigest()


def _get_sourcelines(
    m: types.ModuleType, firstlineno: int, lastlineno: int
) -> list[str]:
    return inspect.getsourcelines(m)[0][firstlineno - 1 : lastlineno - 1]


def _hash_sourcelines(m: types.ModuleType, firstlineno: int, lastlineno: int) -> str:
    return _hash_source("".join(_get_sourcelines(m, firstlineno, lastlineno)))


def _compile_frame_context(
    code: types.CodeType,
) -> contextlib.AbstractContextManager[None]:
    from torch._dynamo.convert_frame import get_compile_id, log_dynamo_start
    from torch._guards import compile_context, CompileContext

    # Each code represents a new compile frame
    # recompiles on the same frame are all saved
    # under the same cache entry, so we don't have recompile ids
    # i.e. If cold start had 0/0, 0/1, 1/0, 1/1, these would be
    # collapsed into 0/0, 1/0 on warm.
    # pyrefly: ignore [deprecated]
    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        increment_frame()
        compile_id = get_compile_id(frame_state={})
        with (
            compile_context(CompileContext(compile_id)),
            dynamo_timed(
                "_compile.compile_inner",
                phase_name="entire_frame_compile",
                dynamo_compile_column_us="dynamo_cumulative_compile_time_us",
                # TODO: save all relevant compilation metrics
                metadata={
                    "frame_key": str(torch._dynamo.utils.curr_frame),
                    "co_name": code.co_name,
                    "co_filename": code.co_filename,
                    "co_firstlineno": code.co_firstlineno,
                },
            ),
        ):
            log_dynamo_start(code)
            yield

    return _ctx()


class CompilePackage:
    """
    CompilePackage is considered a low level component and should not be directly exposed to
    end users. It has the following interface:

    1. `CompilePackage.__init__()` which optionally takes previously serialized dynamo states.
        a. when `dynamo` argument is None, it will construct a brand new CompilePackage object.
        b. when `dynamo` argument is not None, it will load a pre-compiled dynamo state.
    2. `package.save()` which dumps the dynamo and backend states to a DynamoCacheEntry object.
    3. `package.install(backends) which will handle all the side-effectful global scope
        updates with compiled functions and resume functions.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}

        self._current_entry: _DynamoCodeCacheEntry | None = None
        self._installed_globals: dict[types.ModuleType, list[str]] = {}
        # device_type that model compiled with.
        self._device_type = "cpu"

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}
        self._source_info: SourceInfo = SourceInfo(inlined_sources=set())
        self._resume_codes: set[types.CodeType] = set()
        self._initialized = False
        if fn is not None:
            self.initialize(fn, dynamo, ignore_inlined_sources)
            self.uninstall()
            self.validate()

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        fn: Any,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        from .eval_frame import innermost_fn

        if self._initialized:
            raise AssertionError("CompilePackage is already initialized")
        self._source_info = SourceInfo(inlined_sources=set())
        self._innermost_fn = innermost_fn(fn)  # type: ignore[assignment]
        if self._innermost_fn is None:
            raise AssertionError("innermost_fn returned None")
        if dynamo is not None:
            if not isinstance(dynamo, _DynamoCacheEntry):
                raise AssertionError(f"Expected _DynamoCacheEntry, got {type(dynamo)}")
            dynamo.check_versions()
            if not ignore_inlined_sources:
                for code in dynamo.source_info.inlined_sources:
                    m = importlib.import_module(code.module)
                    checksum = _hash_sourcelines(m, code.firstlineno, code.lastlineno)
                    if checksum != code.checksum:
                        raise RuntimeError(
                            f"Source code changes detected for {code.module} (line {code.firstlineno} - line {code.lastlineno})"
                        )

                self._source_info = dynamo.source_info

            main, *codes = dynamo.codes
            self._codes = {self._innermost_fn.__code__: main}
            for code in codes:
                self._codes[SerializedCode.to_code_object(code.python_code)] = code
        else:
            self._add_function(
                self._innermost_fn.__code__, self._innermost_fn.__module__
            )
        self._initialized = True

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: _FunctionId | None = None,
        code_source: str | None = None,
        install_to_global: bool = False,
    ) -> None:
        if python_code not in self._codes:
            code = _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(python_code),
                python_module=python_module,
                function_names=[],
                guarded_codes=[],
                import_sources={},
                backend_ids=[],
                code_source=code_source,
                install_to_global=install_to_global,
            )
            self._codes[python_code] = code
        else:
            code = self._codes[python_code]
            if code.python_module != python_module:
                raise AssertionError(
                    f"python_module mismatch: {code.python_module} != {python_module}"
                )
            if code.install_to_global != install_to_global:
                raise AssertionError(
                    f"install_to_global mismatch: {code.install_to_global} != {install_to_global}"
                )
            if code.code_source != code_source:
                raise AssertionError(
                    f"code_source mismatch: {code.code_source} != {code_source}"
                )

        if function_name is not None:
            code.function_names.append(function_name)

    @property
    def cached_backends(self) -> dict[_BackendId, Any]:
        return self._cached_backends

    @functools.cached_property
    def source_id(self) -> str:
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set")
        return CompilePackage.source_id_from_fn(self._innermost_fn)

    def _add_user_function(self, code: types.CodeType) -> None:
        function_name, code_source = _get_code_source(code)
        module = inspect.getmodule(code)
        if module is None:
            raise PackageError(f"Cannot find module for code {code}")
        self._add_function(
            code,
            module.__name__,
            function_name=_FunctionId(function_name),
            code_source=code_source,
        )

    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None, None, None]:
        if self._current_entry is not None:
            raise AssertionError("_current_entry is already set in code_context")

        # Sometimes user code cannot be inlined in dynamo resulting in extra user code
        # being compiled. We should record these as when they are actually invoked.
        if code not in self._codes:
            self._add_user_function(code)

        entry = self._codes[code]
        self._current_entry = entry
        try:
            yield
        finally:
            entry.has_compile_id = True
            self._current_entry = None

    def add_guarded_code(
        self,
        guards_state: bytes,
        dynamo_code: types.CodeType,
    ) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_guarded_code")
        if self._current_entry.bypassed:
            return
        guarded_code_entry = _GuardedCodeCacheEntry(
            guards_state=guards_state,
            dynamo_code=SerializedCode.from_code_object(dynamo_code),
        )
        self._current_entry.guarded_codes.append(guarded_code_entry)
        for backend_id in _backend_ids_from_code(dynamo_code):
            self._add_backend_id(backend_id)

    def add_inlined_source(self, sources: list[types.CodeType]) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_inlined_source")
        if self._current_entry.bypassed:
            return
        for code in sources:
            if code in self._resume_codes:
                continue
            self._source_info.add_code(code)

    def update_device_type(self, graph: torch.fx.Graph | None) -> None:
        self._device_type = _graph_device_type(graph)

    def bypass_current_entry(self) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in bypass_current_entry")
        self._current_entry.bypassed = True

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: str,
    ) -> None:
        self._add_function(
            python_code,
            python_module,
            function_name=_FunctionId(function_name),
            install_to_global=True,
        )
        self._resume_codes.add(python_code)

    def add_import_source(self, alias: str, module_name: str) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_import_source")
        self._current_entry.import_sources[alias] = module_name

    def _add_backend_id(
        self, backend_id: _BackendId, backend: Any | None = None
    ) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_backend_id")
        if backend_id not in self._current_entry.backend_ids:
            self._current_entry.backend_ids.append(backend_id)
        if backend is not None:
            self._cached_backends[backend_id] = backend

    def add_backend_id(self, backend_id: str, backend: Any | None = None) -> None:
        if not backend_id.startswith(f"{COMPILED_FN_PREFIX}_"):
            raise AssertionError(
                f"backend_id must start with '{COMPILED_FN_PREFIX}_', got '{backend_id}'"
            )
        self._add_backend_id(_BackendId(backend_id), backend)

    def validate(self) -> None:
        if self._current_entry is not None:
            raise AssertionError("_current_entry should be None during validate")
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set during validate")
        if not self._initialized:
            raise AssertionError("CompilePackage is not initialized during validate")
        if next(iter(self._codes)) is not self._innermost_fn.__code__:
            raise AssertionError(
                "First code entry does not match _innermost_fn.__code__"
            )

    def _install_global(self, module: types.ModuleType, name: str, value: Any) -> None:
        # A pre-reset compile in this process may still own `name` via a
        # CleanupHook that hasn't fired yet. We're taking over the binding now,
        # so that hook must not delete it once its code object is collected.
        CleanupHook.disown(module.__dict__, name)
        module.__dict__[name] = value
        self._installed_globals.setdefault(module, []).append(name)

    def uninstall(self) -> None:
        from torch._C._dynamo.eval_frame import _reset_precompile_entries

        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in uninstall")
        for module, names in self._installed_globals.items():
            for name in names:
                module.__dict__.pop(name, None)

        self._installed_globals = {}

        _reset_precompile_entries(self._innermost_fn.__code__)

    def install(self, backends: dict[_BackendId, Any]) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        from .convert_frame import input_codes
        from .output_graph import get_builtins_dict

        self.uninstall()
        for code, entry in self._codes.items():
            context = (
                _compile_frame_context(code)
                if entry.has_compile_id
                else contextlib.nullcontext()
            )
            with context:
                module = sys.modules[entry.python_module]
                for alias, module_name in entry.import_sources.items():
                    self._install_global(
                        module, alias, importlib.import_module(module_name)
                    )
                target_code = code
                if entry.install_to_global:
                    for function_name in entry.function_names:
                        if code.co_freevars:
                            # Resume functions with freevars need a factory
                            # that takes a closure tuple, matching
                            # install_resume_function_global in output_graph.py.
                            f_globals = module.__dict__
                            fn_name = function_name

                            def _make_fn(
                                closure: tuple[types.CellType, ...],
                                _code: types.CodeType = code,
                                _globals: dict[str, Any] = f_globals,
                                _name: str = fn_name,
                            ) -> types.FunctionType:
                                return types.FunctionType(
                                    _code, _globals, _name, None, closure
                                )

                            self._install_global(module, function_name, _make_fn)
                        else:
                            fn = types.FunctionType(
                                code, module.__dict__, function_name
                            )
                            self._install_global(module, function_name, fn)
                if entry.code_source:
                    target_code = _lookup_code(entry)

                if entry.bypassed:
                    # If the entry is bypassed, do not install backends
                    # or guarded codes.
                    continue

                input_codes.add(target_code)
                for backend_id in entry.backend_ids:
                    if backend_id not in backends:
                        raise RuntimeError(
                            f"Backend {backend_id} is not found in the given backends"
                        )
                    with dynamo_timed(
                        "after_deserialization", phase_name="backend_compile"
                    ):
                        backend = backends[backend_id].after_deserialization()
                        self._install_global(
                            module,
                            backend_id,
                            torch._dynamo.disable(backend),
                        )

                if len(entry.guarded_codes) == 0:
                    # Dynamo generates empty graph for trivial functions, should just skip them
                    # in these cases.
                    torch._dynamo.eval_frame.skip_code(target_code)

                for guarded_code in entry.guarded_codes:
                    with dynamo_timed("precompile_load_guards"):
                        guards_state = load_guards_state(guarded_code.guards_state)
                    runtime_global_scope = sys.modules[entry.python_module].__dict__
                    # The installed builtins dict might be absent from the runtime
                    # while loading guards. Populate it if it's missing.
                    if (
                        builtin_dict_name
                        := guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                    ):
                        # A pre-reset compile's CleanupHook may still own this
                        # name even when we're about to leave its value alone
                        # below (same dict object every compile in this
                        # module), so it must not delete it once collected.
                        CleanupHook.disown(runtime_global_scope, builtin_dict_name)
                        builtins_dict = get_builtins_dict(runtime_global_scope)
                        if builtin_dict_name in runtime_global_scope:
                            if (
                                runtime_global_scope[builtin_dict_name]
                                is not builtins_dict
                            ):
                                raise AssertionError(
                                    f"Builtins dict mismatch for key '{builtin_dict_name}'"
                                )
                        else:
                            runtime_global_scope[builtin_dict_name] = builtins_dict
                    if not isinstance(guards_state, torch._dynamo.guards.GuardsState):
                        raise AssertionError(
                            f"Expected GuardsState, got {type(guards_state)}"
                        )
                    with dynamo_timed("precompile_build_guards"):
                        guard_manager = load_guard_manager(
                            guards_state, target_code, runtime_global_scope
                        )
                    _load_precompile_entry(
                        target_code,
                        guard_manager,
                        SerializedCode.to_code_object(guarded_code.dynamo_code),
                    )

    def cache_entry(self) -> _DynamoCacheEntry:
        self.validate()
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in cache_entry")
        return _DynamoCacheEntry(
            codes=list(self._codes.values()),
            source_info=self._source_info,
            device_type=self._device_type,
            fn_name=self._innermost_fn.__qualname__,
            fn_first_lineno=self._innermost_fn.__code__.co_firstlineno,
        )

    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str:
        from .eval_frame import innermost_fn

        innermost_fn_ = innermost_fn(fn)

        sha256_hash = hashlib.sha256()
        sha256_hash.update(innermost_fn_.__qualname__.encode())
        sha256_hash.update(str(innermost_fn_.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()


_Backends = dict[_BackendId, Any]


class DynamoStore(abc.ABC):
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.

    This is an abstract base class for different storage implementations.
    """

    def record_package(self, package: CompilePackage) -> None:
        """
        Records a package to PrecompileContext, so that it can be serialized later.
        """
        from torch._dynamo.precompile_context import PrecompileContext

        cache_entry = package.cache_entry()
        PrecompileContext.record_dynamo_cache_entry(
            cache_entry=cache_entry, key=package.source_id
        )

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """
        Records eager fx graphs to PrecompileContext for testing purposes.
        """
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        result = EagerCacheArtifact(key=backend_id, content=backend)
        PrecompileContext.record_artifact(result)

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Abstract method to write dynamo cache entry and backends to storage.

        Args:
            dynamo: The dynamo cache entry to write
            backends: Dictionary of backend content to write
            path: Path or key to identify where to write the data
        """
        ...

    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        backend_content: _Backends = {}
        for backend_id in cache_entry.backend_ids:
            serialized_backend = PrecompileContext.serialize_artifact_by_key(backend_id)
            if serialized_backend is None:
                raise RuntimeError(
                    f"Backend {backend_id} is not found in the given backends"
                )
            if not isinstance(serialized_backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(serialized_backend)}"
                )
            backend_content[backend_id] = serialized_backend

        entry = PrecompileCacheEntry(cache_entry, backend_content)

        self.write(entry, key)

    def save_package(self, package: CompilePackage, key: str) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        self.record_package(package)
        cache_entry = package.cache_entry()
        self.save_cache_entry(cache_entry, key)

    @abc.abstractmethod
    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Abstract method to read dynamo cache entry and backends from storage.

        Args:
            path: Path or key to identify where to read the data from

        Returns:
            A tuple containing (dynamo_cache_entry, backend_content)
        """
        ...

    def load_cache_entry(self, key: str) -> PrecompileCacheEntry:
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        precompile_entry = self.read(key)
        for backend in precompile_entry.backends.values():
            if not isinstance(backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(backend)}"
                )
            PrecompileContext.record_artifact(backend)

        return precompile_entry

    def load_package(
        self, fn: Any, key: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        entry = self.load_cache_entry(key)
        package = CompilePackage(fn, entry.dynamo)
        return package, entry.backends


class InMemoryDynamoStore(DynamoStore):
    """
    A DynamoStore implementation that keeps state about CompilePackages in memory.
    """

    def __init__(self) -> None:
        self.packages: dict[str, PrecompileCacheEntry] = {}

    def clear(self) -> None:
        self.packages.clear()

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Store the dynamo cache entry and backends in memory instead of writing to disk.
        """
        self.packages[path] = cache_entry

    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Read dynamo cache entry and backends from memory.
        """
        if path not in self.packages:
            raise RuntimeError(f"No package found with key {path}")

        return self.packages[path]


class DiskDynamoStore(DynamoStore):
    """
    A DynamoStore implementation that keeps state about CompilePackages on disk.
    """

    def __init__(self, path_prefix: str = "") -> None:
        """
        Initialize a DiskDynamoStore with a path prefix.

        Args:
            path_prefix: Prefix directory for where to put CompilePackages on disk
        """
        self._path_prefix = path_prefix

    def path_prefix(self) -> str:
        return self._path_prefix

    def clear(self) -> None:
        """
        Clear all CompilePackages from disk.
        """
        if self.path_prefix():
            shutil.rmtree(self.path_prefix(), ignore_errors=True)

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Write dynamo cache entry and backends to disk.
        """
        try:
            pickled_content: bytes = pickle.dumps(cache_entry)
            CacheArtifactRecorder(PrecompileCacheArtifact.type(), path).record(
                pickled_content
            )
            self._write_to_local_cache(pickled_content, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def _write_to_local_cache(self, pickled_content: bytes, path: str) -> None:
        from torch._inductor.codecache import write_atomic

        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            os.makedirs(path, exist_ok=True)
            write_atomic(os.path.join(path, "entry"), pickled_content)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Read dynamo cache entry and backends from disk.
        """
        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            with open(os.path.join(path, "entry"), "rb") as f:
                pickled_content = f.read()
                entry = pickle.loads(pickled_content)
                return entry
        except Exception as e:
            raise RuntimeError(f"Failed to load package from path {path}: {e}") from e


class DiskDynamoCache(DiskDynamoStore):
    """
    Special DiskDynamoStore which adds some helper functions for automatically
    tracking paths of packages
    """

    def save(self, package: CompilePackage) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        key = package.source_id
        logger.info("Saving CompilePackage for %s", package.source_id)
        super().save_package(package, key)

    def load(self, fn: Callable[..., Any]) -> PrecompileCacheEntry | None:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        key = CompilePackage.source_id_from_fn(fn)
        logger.info("Loading CompilePackage for %s", key)
        path = os.path.join(self.path_prefix(), key)
        if os.path.exists(path):
            try:
                result = super().load_cache_entry(key)
                counters["dynamo_cache"]["dynamo_cache_hit"] += 1
                return result
            except Exception:
                counters["dynamo_cache"]["dynamo_cache_error"] += 1
                logger.warning("Failed to load package from path %s", exc_info=True)
                return None
        logger.info("No package found for %s", key)
        counters["dynamo_cache"]["dynamo_cache_miss"] += 1
        return None

    def load_and_install_package(self, fn: Callable[..., Any]) -> CompilePackage | None:
        """
        Load directly into a package and install backends
        """
        results = self.load(fn)
        if results is None:
            return None
        else:
            package = CompilePackage(fn, results.dynamo)
            package.install(results.backends)
            return package

    def path_prefix(self) -> str:
        return os.path.join(cache_dir(), "dynamo")


def cache_dir() -> str:
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return cache_dir()


DynamoCache = DiskDynamoCache(os.path.join(cache_dir(), "dynamo"))
<|endoftext|># mypy: allow-untyped-defs
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

# Executed in the order they're registered
INTERMEDIATE_HOOKS: list[Callable[[str, "torch.Tensor"], None]] = []


@contextlib.contextmanager
def intermediate_hook(fn):
    INTERMEDIATE_HOOKS.append(fn)
    try:
        yield
    finally:
        INTERMEDIATE_HOOKS.pop()


def run_intermediate_hooks(name, val):
    global INTERMEDIATE_HOOKS
    hooks = INTERMEDIATE_HOOKS
    INTERMEDIATE_HOOKS = []
    try:
        for hook in hooks:
            hook(name, val)
    finally:
        INTERMEDIATE_HOOKS = hooks
<|endoftext|>from sympy.concrete.summations import (Sum, summation)
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.polys.polytools import poly
from sympy.series.series import series

from sympy.polys.polyerrors import PolynomialError
from sympy.stats.crv import reduce_rational_inequalities_wrap
from sympy.stats.rv import (NamedArgsMixin, SinglePSpace, SingleDomain,
                            random_symbols, PSpace, ConditionalDomain, RandomDomain,
                            ProductDomain, Distribution)
from sympy.stats.symbolic_probability import Probability
from sympy.sets.fancysets import Range, FiniteSet
from sympy.sets.sets import Union
from sympy.sets.contains import Contains
from sympy.utilities import filldedent
from sympy.core.sympify import _sympify


class DiscreteDistribution(Distribution):
    def __call__(self, *args):
        return self.pdf(*args)


class SingleDiscreteDistribution(DiscreteDistribution, NamedArgsMixin):
    """ Discrete distribution of a single variable.

    Serves as superclass for PoissonDistribution etc....

    Provides methods for pdf, cdf, and sampling

    See Also:
        sympy.stats.crv_types.*
    """

    set = S.Integers

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x = symbols('x', integer=True, cls=Dummy)
        z = symbols('z', real=True, cls=Dummy)
        left_bound = self.set.inf

        # CDF is integral of PDF from left bound to z
        pdf = self.pdf(x)
        cdf = summation(pdf, (x, left_bound, floor(z)), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        return None

    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if not kwargs:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = summation(exp(I*t*x)*pdf, (x, self.set.inf, self.set.sup))
        return Lambda(t, cf)

    def _characteristic_function(self, t):
        return None

    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if not kwargs:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)

    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        t = Dummy('t', real=True)
        x = Dummy('x', integer=True)
        pdf = self.pdf(x)
        mgf = summation(exp(t*x)*pdf, (x, self.set.inf, self.set.sup))
        return Lambda(t, mgf)

    def _moment_generating_function(self, t):
        return None

    def moment_generating_function(self, t, **kwargs):
        if not kwargs:
            mgf = self._moment_generating_function(t)
            if mgf is not None:
                return mgf
        return self.compute_moment_generating_function(**kwargs)(t)

    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        x = Dummy('x', integer=True)
        p = Dummy('p', real=True)
        left_bound = self.set.inf
        pdf = self.pdf(x)
        cdf = summation(pdf, (x, left_bound, x), **kwargs)
        set = ((x, p <= cdf), )
        return Lambda(p, Piecewise(*set))

    def _quantile(self, x):
        return None

    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        if not kwargs:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        return self.compute_quantile(**kwargs)(x)

    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        # TODO: support discrete sets with non integer stepsizes

        if evaluate:
            try:
                p = poly(expr, var)

                t = Dummy('t', real=True)

                mgf = self.moment_generating_function(t)
                deg = p.degree()
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                for k in range(deg+1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)

                return result

            except PolynomialError:
                return summation(expr * self.pdf(var),
                                 (var, self.set.inf, self.set.sup), **kwargs)

        else:
            return Sum(expr * self.pdf(var),
                         (var, self.set.inf, self.set.sup), **kwargs)

    def __call__(self, *args):
        return self.pdf(*args)


class DiscreteDomain(RandomDomain):
    """
    A domain with discrete support with step size one.
    Represented using symbols and Range.
    """
    is_Discrete = True

class SingleDiscreteDomain(DiscreteDomain, SingleDomain):
    def as_boolean(self):
        return Contains(self.symbol, self.set)


class ConditionalDiscreteDomain(DiscreteDomain, ConditionalDomain):
    """
    Domain with discrete support of step size one, that is restricted by
    some condition.
    """
    @property
    def set(self):
        rv = self.symbols
        if len(self.symbols) > 1:
            raise NotImplementedError(filldedent('''
                Multivariate conditional domains are not yet implemented.'''))
        rv = list(rv)[0]
        return reduce_rational_inequalities_wrap(self.condition,
            rv).intersect(self.fulldomain.set)


class DiscretePSpace(PSpace):
    is_real = True
    is_Discrete = True

    @property
    def pdf(self):
        return self.density(*self.symbols)

    def where(self, condition):
        rvs = random_symbols(condition)
        assert all(r.symbol in self.symbols for r in rvs)
        if len(rvs) > 1:
            raise NotImplementedError(filldedent('''Multivariate discrete
            random variables are not yet supported.'''))
        conditional_domain = reduce_rational_inequalities_wrap(condition,
            rvs[0])
        conditional_domain = conditional_domain.intersect(self.domain.set)
        return SingleDiscreteDomain(rvs[0].symbol, conditional_domain)

    def probability(self, condition):
        complement = isinstance(condition, Ne)
        if complement:
            condition = Eq(condition.args[0], condition.args[1])
        try:
            _domain = self.where(condition).set
            if condition == False or _domain is S.EmptySet:
                return S.Zero
            if condition == True or _domain == self.domain.set:
                return S.One
            prob = self.eval_prob(_domain)
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            dens = density(expr)
            if not isinstance(dens, DiscreteDistribution):
                from sympy.stats.drv_types import DiscreteDistributionHandmade
                dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleDiscretePSpace(z, dens)
            prob = space.probability(condition.__class__(space.value, 0))
        if prob is None:
            prob = Probability(condition)
        return prob if not complement else S.One - prob

    def eval_prob(self, _domain):
        sym = list(self.symbols)[0]
        if isinstance(_domain, Range):
            n = symbols('n', integer=True)
            inf, sup, step = (r for r in _domain.args)
            summand = ((self.pdf).replace(
              sym, n*step))
            rv = summation(summand,
                (n, inf/step, (sup)/step - 1)).doit()
            return rv
        elif isinstance(_domain, FiniteSet):
            pdf = Lambda(sym, self.pdf)
            rv = sum(pdf(x) for x in _domain)
            return rv
        elif isinstance(_domain, Union):
            rv = sum(self.eval_prob(x) for x in _domain.args)
            return rv

    def conditional_space(self, condition):
        # XXX: Converting from set to tuple. The order matters to Lambda
        # though so we should be starting with a set...
        density = Lambda(tuple(self.symbols), self.pdf/self.probability(condition))
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalDiscreteDomain(self.domain, condition)
        return DiscretePSpace(domain, density)

class ProductDiscreteDomain(ProductDomain, DiscreteDomain):
    def as_boolean(self):
        return And(*[domain.as_boolean for domain in self.domains])

class SingleDiscretePSpace(DiscretePSpace, SinglePSpace):
    """ Discrete probability space over a single univariate variable """
    is_real = True

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return SingleDiscreteDomain(self.symbol, self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=True, **kwargs):
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr

        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        x = self.value.symbol
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate,
                    **kwargs)
        except NotImplementedError:
            return Sum(expr * self.pdf, (x, self.set.inf, self.set.sup),
                    **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if expr == self.value:
            x = Dummy("x", real=True)
            return Lambda(x, self.distribution.cdf(x, **kwargs))
        else:
            raise NotImplementedError()

    def compute_density(self, expr, **kwargs):
        if expr == self.value:
            return self.distribution
        raise NotImplementedError()

    def compute_characteristic_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            raise NotImplementedError()

    def compute_moment_generating_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            raise NotImplementedError()

    def compute_quantile(self, expr, **kwargs):
        if expr == self.value:
            p = Dummy("p", real=True)
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            raise NotImplementedError()
<|endoftext|>r"""
Galois resolvents

Each of the functions in ``sympy.polys.numberfields.galoisgroups`` that
computes Galois groups for a particular degree $n$ uses resolvents. Given the
polynomial $T$ whose Galois group is to be computed, a resolvent is a
polynomial $R$ whose roots are defined as functions of the roots of $T$.

One way to compute the coefficients of $R$ is by approximating the roots of $T$
to sufficient precision. This module defines a :py:class:`~.Resolvent` class
that handles this job, determining the necessary precision, and computing $R$.

In some cases, the coefficients of $R$ are symmetric in the roots of $T$,
meaning they are equal to fixed functions of the coefficients of $T$. Therefore
another approach is to compute these functions once and for all, and record
them in a lookup table. This module defines code that can compute such tables.
The tables for polynomials $T$ of degrees 4 through 6, produced by this code,
are recorded in the resolvent_lookup.py module.

"""

from sympy.core.evalf import (
    evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath,
)
from sympy.core.symbol import symbols, Dummy
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ
from sympy.polys.orderings import lex
from sympy.polys.polyroots import preprocess_roots
from sympy.polys.polytools import Poly
from sympy.polys.rings import xring
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.lambdify import lambdify

from mpmath import MPContext
from mpmath.libmp.libmpf import prec_to_dps


class GaloisGroupException(Exception):
    ...


class ResolventException(GaloisGroupException):
    ...


class Resolvent:
    r"""
    If $G$ is a subgroup of the symmetric group $S_n$,
    $F$ a multivariate polynomial in $\mathbb{Z}[X_1, \ldots, X_n]$,
    $H$ the stabilizer of $F$ in $G$ (i.e. the permutations $\sigma$ such that
    $F(X_{\sigma(1)}, \ldots, X_{\sigma(n)}) = F(X_1, \ldots, X_n)$), and $s$
    a set of left coset representatives of $H$ in $G$, then the resolvent
    polynomial $R(Y)$ is the product over $\sigma \in s$ of
    $Y - F(X_{\sigma(1)}, \ldots, X_{\sigma(n)})$.

    For example, consider the resolvent for the form
    $$F = X_0 X_2 + X_1 X_3$$
    and the group $G = S_4$. In this case, the stabilizer $H$ is the dihedral
    group $D4 = < (0123), (02) >$, and a set of representatives of $G/H$ is
    $\{I, (01), (03)\}$. The resolvent can be constructed as follows:

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.core.symbol import symbols
    >>> from sympy.polys.numberfields.galoisgroups import Resolvent
    >>> X = symbols('X0 X1 X2 X3')
    >>> F = X[0]*X[2] + X[1]*X[3]
    >>> s = [Permutation([0, 1, 2, 3]), Permutation([1, 0, 2, 3]),
    ... Permutation([3, 1, 2, 0])]
    >>> R = Resolvent(F, X, s)

    This resolvent has three roots, which are the conjugates of ``F`` under the
    three permutations in ``s``:

    >>> R.root_lambdas[0](*X)
    X0*X2 + X1*X3
    >>> R.root_lambdas[1](*X)
    X0*X3 + X1*X2
    >>> R.root_lambdas[2](*X)
    X0*X1 + X2*X3

    Resolvents are useful for computing Galois groups. Given a polynomial $T$
    of degree $n$, we will use a resolvent $R$ where $Gal(T) \leq G \leq S_n$.
    We will then want to substitute the roots of $T$ for the variables $X_i$
    in $R$, and study things like the discriminant of $R$, and the way $R$
    factors over $\mathbb{Q}$.

    From the symmetry in $R$'s construction, and since $Gal(T) \leq G$, we know
    from Galois theory that the coefficients of $R$ must lie in $\mathbb{Z}$.
    This allows us to compute the coefficients of $R$ by approximating the
    roots of $T$ to sufficient precision, plugging these values in for the
    variables $X_i$ in the coefficient expressions of $R$, and then simply
    rounding to the nearest integer.

    In order to determine a sufficient precision for the roots of $T$, this
    ``Resolvent`` class imposes certain requirements on the form ``F``. It
    could be possible to design a different ``Resolvent`` class, that made
    different precision estimates, and different assumptions about ``F``.

    ``F`` must be homogeneous, and all terms must have unit coefficient.
    Furthermore, if $r$ is the number of terms in ``F``, and $t$ the total
    degree, and if $m$ is the number of conjugates of ``F``, i.e. the number
    of permutations in ``s``, then we require that $m < r 2^t$. Again, it is
    not impossible to work with forms ``F`` that violate these assumptions, but
    this ``Resolvent`` class requires them.

    Since determining the integer coefficients of the resolvent for a given
    polynomial $T$ is one of the main problems this class solves, we take some
    time to explain the precision bounds it uses.

    The general problem is:
    Given a multivariate polynomial $P \in \mathbb{Z}[X_1, \ldots, X_n]$, and a
    bound $M \in \mathbb{R}_+$, compute an $\varepsilon > 0$ such that for any
    complex numbers $a_1, \ldots, a_n$ with $|a_i| < M$, if the $a_i$ are
    approximated to within an accuracy of $\varepsilon$ by $b_i$, that is,
    $|a_i - b_i| < \varepsilon$ for $i = 1, \ldots, n$, then
    $|P(a_1, \ldots, a_n) - P(b_1, \ldots, b_n)| < 1/2$. In other words, if it
    is known that $P(a_1, \ldots, a_n) = c$ for some $c \in \mathbb{Z}$, then
    $P(b_1, \ldots, b_n)$ can be rounded to the nearest integer in order to
    determine $c$.

    To derive our error bound, consider the monomial $xyz$. Defining
    $d_i = b_i - a_i$, our error is
    $|(a_1 + d_1)(a_2 + d_2)(a_3 + d_3) - a_1 a_2 a_3|$, which is bounded
    above by $|(M + \varepsilon)^3 - M^3|$. Passing to a general monomial of
    total degree $t$, this expression is bounded by
    $M^{t-1}\varepsilon(t + 2^t\varepsilon/M)$ provided $\varepsilon < M$,
    and by $(t+1)M^{t-1}\varepsilon$ provided $\varepsilon < M/2^t$.
    But since our goal is to make the error less than $1/2$, we will choose
    $\varepsilon < 1/(2(t+1)M^{t-1})$, which implies the condition that
    $\varepsilon < M/2^t$, as long as $M \geq 2$.

    Passing from the general monomial to the general polynomial is easy, by
    scaling and summing error bounds.

    In our specific case, we are given a homogeneous polynomial $F$ of
    $r$ terms and total degree $t$, all of whose coefficients are $\pm 1$. We
    are given the $m$ permutations that make the conjugates of $F$, and
    we want to bound the error in the coefficients of the monic polynomial
    $R(Y)$ having $F$ and its conjugates as roots (i.e. the resolvent).

    For $j$ from $1$ to $m$, the coefficient of $Y^{m-j}$ in $R(Y)$ is the
    $j$th elementary symmetric polynomial in the conjugates of $F$. This sums
    the products of these conjugates, taken $j$ at a time, in all possible
    combinations. There are $\binom{m}{j}$ such combinations, and each product
    of $j$ conjugates of $F$ expands to a sum of $r^j$ terms, each of unit
    coefficient, and total degree $jt$. An error bound for the $j$th coeff of
    $R$ is therefore
    $$\binom{m}{j} r^j (jt + 1) M^{jt - 1} \varepsilon$$
    When our goal is to evaluate all the coefficients of $R$, we will want to
    use the maximum of these error bounds. It is clear that this bound is
    strictly increasing for $j$ up to the ceiling of $m/2$. After that point,
    the first factor $\binom{m}{j}$ begins to decrease, while the others
    continue to increase. However, the binomial coefficient never falls by more
    than a factor of $1/m$ at a time, so our assumptions that $M \geq 2$ and
    $m < r 2^t$ are enough to tell us that the constant coefficient of $R$,
    i.e. that where $j = m$, has the largest error bound. Therefore we can use
    $$r^m (mt + 1) M^{mt - 1} \varepsilon$$
    as our error bound for all the coefficients.

    Note that this bound is also (more than) adequate to determine whether any
    of the roots of $R$ is an integer. Each of these roots is a single
    conjugate of $F$, which contains less error than the trace, i.e. the
    coefficient of $Y^{m - 1}$. By rounding the roots of $R$ to the nearest
    integers, we therefore get all the candidates for integer roots of $R$. By
    plugging these candidates into $R$, we can check whether any of them
    actually is a root.

    Note: We take the definition of resolvent from Cohen, but the error bound
    is ours.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
       (Def 6.3.2)

    """

    def __init__(self, F, X, s):
        r"""
        Parameters
        ==========

        F : :py:class:`~.Expr`
            polynomial in the symbols in *X*
        X : list of :py:class:`~.Symbol`
        s : list of :py:class:`~.Permutation`
            representing the cosets of the stabilizer of *F* in
            some subgroup $G$ of $S_n$, where $n$ is the length of *X*.
        """
        self.F = F
        self.X = X
        self.s = s

        # Number of conjugates:
        self.m = len(s)
        # Total degree of F (computed below):
        self.t = None
        # Number of terms in F (computed below):
        self.r = 0

        for monom, coeff in Poly(F).terms():
            if abs(coeff) != 1:
                raise ResolventException('Resolvent class expects forms with unit coeffs')
            t = sum(monom)
            if t != self.t and self.t is not None:
                raise ResolventException('Resolvent class expects homogeneous forms')
            self.t = t
            self.r += 1

        m, t, r = self.m, self.t, self.r
        if not m < r * 2**t:
            raise ResolventException('Resolvent class expects m < r*2^t')
        M = symbols('M')
        # Precision sufficient for computing the coeffs of the resolvent:
        self.coeff_prec_func = Poly(r**m*(m*t + 1)*M**(m*t - 1))
        # Precision sufficient for checking whether any of the roots of the
        # resolvent are integers:
        self.root_prec_func = Poly(r*(t + 1)*M**(t - 1))

        # The conjugates of F are the roots of the resolvent.
        # For evaluating these to required numerical precisions, we need
        # lambdified versions.
        # Note: for a given permutation sigma, the conjugate (sigma F) is
        # equivalent to lambda [sigma^(-1) X]: F.
        self.root_lambdas = [
            lambdify((~s[j])(X), F)
            for j in range(self.m)
        ]

        # For evaluating the coeffs, we'll also need lambdified versions of
        # the elementary symmetric functions for degree m.
        Y = symbols('Y')
        R = symbols(' '.join(f'R{i}' for i in range(m)))
        f = 1
        for r in R:
            f *= (Y - r)
        C = Poly(f, Y).coeffs()
        self.esf_lambdas = [lambdify(R, c) for c in C]

    def get_prec(self, M, target='coeffs'):
        r"""
        For a given upper bound *M* on the magnitude of the complex numbers to
        be plugged in for this resolvent's symbols, compute a sufficient
        precision for evaluating those complex numbers, such that the
        coefficients, or the integer roots, of the resolvent can be determined.

        Parameters
        ==========

        M : real number
            Upper bound on magnitude of the complex numbers to be plugged in.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Name the task for which a sufficient precision is desired.
            This is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        int $m$
            such that $2^{-m}$ is a sufficient upper bound on the
            error in approximating the complex numbers to be plugged in.

        """
        # As explained in the docstring for this class, our precision estimates
        # require that M be at least 2.
        M = max(M, 2)
        f = self.coeff_prec_func if target == 'coeffs' else self.root_prec_func
        r, _, _, _ = evalf(2*f(M), 1, {})
        return fastlog(r) + 1

    def approximate_roots_of_poly(self, T, target='coeffs'):
        """
        Approximate the roots of a given polynomial *T* to sufficient precision
        in order to evaluate this resolvent's coefficients, or determine
        whether the resolvent has an integer root.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        target : str, 'coeffs' or 'roots', default='coeffs'
            Set the approximation precision to be sufficient for the desired
            task, which is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        list of elements of :ref:`CC`

        """
        ctx = MPContext()
        # Because sympy.polys.polyroots._integer_basis() is called when a CRootOf
        # is formed, we proactively extract the integer basis now. This means that
        # when we call T.all_roots(), every root will be a CRootOf, not a Mul
        # of Integer*CRootOf.
        coeff, T = preprocess_roots(T)
        coeff = ctx.mpf(str(coeff))

        scaled_roots = T.all_roots(radicals=False)

        # Since we're going to be approximating the roots of T anyway, we can
        # get a good upper bound on the magnitude of the roots by starting with
        # a very low precision approx.
        approx0 = [coeff * quad_to_mpmath(_evalf_with_bounded_error(r, m=0)) for r in scaled_roots]
        # Here we add 1 to account for the possible error in our initial approximation.
        M = max(abs(b) for b in approx0) + 1
        m = self.get_prec(M, target=target)
        n = fastlog(M._mpf_) + 1
        p = m + n + 1
        ctx.prec = p
        d = prec_to_dps(p)

        approx1 = [r.eval_approx(d, return_mpmath=True) for r in scaled_roots]
        approx1 = [coeff*ctx.mpc(r) for r in approx1]

        return approx1

    @staticmethod
    def round_mpf(a):
        if isinstance(a, int):
            return a
        # If we use python's built-in `round()`, we lose precision.
        # If we use `ZZ` directly, we may add or subtract 1.
        #
