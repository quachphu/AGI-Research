        args : sequence, optional
            Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.

    tol : float, optional
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods except `TNC` accept the
        following generic options:

        maxiter : int
            Maximum number of iterations to perform. Depending on the
            method each iteration may use several function evaluations.

            For `TNC` use `maxfun` instead of `maxiter`.
        disp : bool
            Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.
    callback : callable, optional
        A callable called after each iteration.

        All methods except TNC, SLSQP, and COBYLA support a callable with
        the signature::

            callback(intermediate_result: OptimizeResult)

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the present values
        of the parameter vector and objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`. These methods will also terminate if
        the callback raises ``StopIteration``.

        All methods except trust-constr (also) support a signature like::

            callback(xk)

        where ``xk`` is the current parameter vector.

        Introspection is used to determine which of the signatures above to
        invoke.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize_scalar : Interface to minimization algorithms for scalar
        univariate functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *BFGS*.

    **Unconstrained minimization**

    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
    gradient algorithm by Polak and Ribiere, a variant of the
    Fletcher-Reeves method described in [5]_ pp.120-122. Only the
    first derivatives are used.

    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
    pp. 136. It uses the first derivatives only. BFGS has proven good
    performance even for non-smooth optimizations. This method also
    returns an approximation of the Hessian inverse, stored as
    `hess_inv` in the OptimizeResult object.

    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
    Newton method). It uses a CG method to the compute the search
    direction. See also *TNC* method for a box-constrained
    minimization with a similar algorithm. Suitable for large-scale
    problems.

    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
    trust-region algorithm [5]_ for unconstrained minimization. This
    algorithm requires the gradient and Hessian; furthermore the
    Hessian is required to be positive definite.

    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
    Newton conjugate gradient trust-region algorithm [5]_ for
    unconstrained minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.

    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
    minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.
    On indefinite problems it requires usually less iterations than the
    `trust-ncg` method and is recommended for medium and large-scale problems.

    Method :ref:`trust-exact <optimize.minimize-trustexact>`
    is a trust-region method for unconstrained minimization in which
    quadratic subproblems are solved almost exactly [13]_. This
    algorithm requires the gradient and the Hessian (which is
    *not* required to be positive definite). It is, in many
    situations, the Newton method to converge in fewer iterations
    and the most recommended for small and medium-size problems.

    **Bound-Constrained minimization**

    Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the
    Simplex algorithm [1]_, [2]_. This algorithm is robust in many
    applications. However, if numerical computation of derivative can be
    trusted, other algorithms using the first and/or second derivatives
    information might be preferred for their better performance in
    general.

    Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B
    algorithm [6]_, [7]_ for bound constrained minimization.

    Method :ref:`Powell <optimize.minimize-powell>` is a modification
    of Powell's method [3]_, [4]_ which is a conjugate direction
    method. It performs sequential one-dimensional minimizations along
    each vector of the directions set (`direc` field in `options` and
    `info`), which is updated at each iteration of the main
    minimization loop. The function need not be differentiable, and no
    derivatives are taken. If bounds are not provided, then an
    unbounded line search will be used. If bounds are provided and
    the initial guess is within the bounds, then every function
    evaluation throughout the minimization procedure will be within
    the bounds. If bounds are provided, the initial guess is outside
    the bounds, and `direc` is full rank (default has full rank), then
    some function evaluations during the first iteration may be
    outside the bounds, but every function evaluation after the first
    iteration will be within the bounds. If `direc` is not full rank,
    then some parameters may not be optimized and the solution is not
    guaranteed to be within the bounds.

    Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton
    algorithm [5]_, [8]_ to minimize a function with variables subject
    to bounds. This algorithm uses gradient information; it is also
    called Newton Conjugate-Gradient. It differs from the *Newton-CG*
    method described above as it wraps a C implementation and allows
    each variable to be given upper and lower bounds.

    **Constrained Minimization**

    Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the
    Constrained Optimization BY Linear Approximation (COBYLA) method
    [9]_, [10]_, [11]_. The algorithm is based on linear
    approximations to the objective function and each constraint. The
    method wraps a FORTRAN implementation of the algorithm. The
    constraints functions 'fun' may return either a single number
    or an array or list of numbers.

    Method :ref:`COBYQA <optimize.minimize-cobyqa>` uses the Constrained
    Optimization BY Quadratic Approximations (COBYQA) method [18]_. The
    algorithm is a derivative-free trust-region SQP method based on quadratic
    approximations to the objective function and each nonlinear constraint. The
    bounds are treated as unrelaxable constraints, in the sense that the
    algorithm always respects them throughout the optimization process.

    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
    Least SQuares Programming to minimize a function of several
    variables with any combination of bounds, equality and inequality
    constraints. The method wraps the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft [12]_. Note that the
    wrapper handles infinite values in bounds by converting them into
    large floating values.

    Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a
    trust-region algorithm for constrained optimization. It switches
    between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When
    inequality constraints are imposed as well, it switches to the trust-region
    interior point method described in [16]_. This interior point algorithm,
    in turn, solves inequality constraints by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    **Finite-Difference Options**

    For Method :ref:`trust-constr <optimize.minimize-trustconstr>`
    the gradient and the Hessian may be approximated using
    three finite-difference schemes: {'2-point', '3-point', 'cs'}.
    The scheme 'cs' is, potentially, the most accurate but it
    requires the function to correctly handle complex inputs and to
    be differentiable in the complex plane. The scheme '3-point' is more
    accurate than '2-point' but requires twice as many operations. If the
    gradient is estimated via finite-differences the Hessian must be
    estimated using one of the quasi-Newton strategies.

    **Method specific options for the** `hess` **keyword**

    +--------------+------+----------+-------------------------+-----+
    | method/Hess  | None | callable | '2-point/'3-point'/'cs' | HUS |
    +==============+======+==========+=========================+=====+
    | Newton-CG    | x    | (n, n)   | x                       | x   |
    |              |      | LO       |                         |     |
    +--------------+------+----------+-------------------------+-----+
    | dogleg       |      | (n, n)   |                         |     |
    +--------------+------+----------+-------------------------+-----+
    | trust-ncg    |      | (n, n)   | x                       | x   |
    +--------------+------+----------+-------------------------+-----+
    | trust-krylov |      | (n, n)   | x                       | x   |
    +--------------+------+----------+-------------------------+-----+
    | trust-exact  |      | (n, n)   |                         |     |
    +--------------+------+----------+-------------------------+-----+
    | trust-constr | x    | (n, n)   |  x                      | x   |
    |              |      | LO       |                         |     |
    |              |      | sp       |                         |     |
    +--------------+------+----------+-------------------------+-----+

    where LO=LinearOperator, sp=Sparse matrix, HUS=HessianUpdateStrategy

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using a frontend to this method such as `scipy.optimize.basinhopping`
    or a different library.  You can simply pass a callable as the ``method``
    parameter.

    The callable is called as ``method(fun, x0, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `callback`, `hess`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  Also, if
    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
    `fun` returns just the function values and `jac` is converted to a function
    returning the Jacobian.  The method shall return an `OptimizeResult`
    object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method.  You can find an example in the scipy.optimize tutorial.

    References
    ----------
    .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function
        Minimization. The Computer Journal 7: 308-13.
    .. [2] Wright M H. 1996. Direct search methods: Once scorned, now
        respectable, in Numerical Analysis 1995: Proceedings of the 1995
        Dundee Biennial Conference in Numerical Analysis (Eds. D F
        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.
        191-208.
    .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of
       a function of several variables without calculating derivatives. The
       Computer Journal 7: 155-162.
    .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
       Numerical Recipes (any edition), Cambridge University Press.
    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
       Springer New York.
    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
       Algorithm for Bound Constrained Optimization. SIAM Journal on
       Scientific and Statistical Computing 16 (5): 1190-1208.
    .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm
       778: L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization. ACM Transactions on Mathematical Software 23 (4):
       550-560.
    .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.
       1984. SIAM Journal of Numerical Analysis 21: 770-778.
    .. [9] Powell, M J D. A direct search optimization method that models
       the objective and constraint functions by linear interpolation.
       1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez
       and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.
    .. [10] Powell M J D. Direct search algorithms for optimization
       calculations. 1998. Acta Numerica 7: 287-336.
    .. [11] Powell M J D. A view of algorithms for optimization without
       derivatives. 2007.Cambridge University Technical Report DAMTP
       2007/NA03
    .. [12] Kraft, D. A software package for sequential quadratic
       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace
       Center -- Institute for Flight Mechanics, Koln, Germany.
    .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.
       Trust region methods. 2000. Siam. pp. 169-200.
    .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free
       implementation of the GLTR method for iterative solution of
       the trust region problem", :arxiv:`1611.04718`
    .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the
       Trust-Region Subproblem using the Lanczos Method",
       SIAM J. Optim., 9(2), 504--525, (1999).
    .. [16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.
        An interior point algorithm for large-scale nonlinear  programming.
        SIAM Journal on Optimization 9.4: 877-900.
    .. [17] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. 1998. On the
        implementation of an algorithm for large-scale equality constrained
        optimization. SIAM Journal on Optimization 8.3: 682-706.
    .. [18] Ragonneau, T. M. *Model-Based Derivative-Free Optimization Methods
        and Software*. PhD thesis, Department of Applied Mathematics, The Hong
        Kong Polytechnic University, Hong Kong, China, 2022. URL:
        https://theses.lib.polyu.edu.hk/handle/200/12294.

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function (and its respective derivatives) is implemented in `rosen`
    (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.

    >>> from scipy.optimize import minimize, rosen, rosen_der

    A simple application of the *Nelder-Mead* method is:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    Now using the *BFGS* algorithm, using the first derivative and a few
    options:

    >>> res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
    ...                options={'gtol': 1e-6, 'disp': True})
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 26
             Function evaluations: 31
             Gradient evaluations: 31
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> print(res.message)
    Optimization terminated successfully.
    >>> res.hess_inv
    array([
        [ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary
        [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],
        [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],
        [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],
        [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]
    ])


    Next, consider a minimization problem with several constraints (namely
    Example 16.4 from [5]_). The objective function is:

    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

    There are three constraints defined as:

    >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

    And variables must be positive, hence the following bounds:

    >>> bnds = ((0, None), (0, None))

    The optimization problem is solved using the SLSQP method as:

    >>> res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
    ...                constraints=cons)

    It should converge to the theoretical solution (1.4 ,1.7).

    """
    x0 = np.atleast_1d(np.asarray(x0))

    if x0.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")

    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        # Select automatically
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()

    if options is None:
        options = {}
    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla', 'cobyqa') and bool(jac):
        warn(f'Method {method} does not use gradient information (jac).',
             RuntimeWarning, stacklevel=2)
    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn(f'Method {method} does not use Hessian information (hess).',
             RuntimeWarning, stacklevel=2)
    # - hessp
    if meth not in ('newton-cg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
       and hessp is not None:
        warn(f'Method {method} does not use Hessian-vector product'
             ' information (hessp).',
             RuntimeWarning, stacklevel=2)
    # - constraints or bounds
    if (meth not in ('cobyla', 'cobyqa', 'slsqp', 'trust-constr', '_custom') and
            np.any(constraints)):
        warn(f'Method {method} cannot handle constraints.',
             RuntimeWarning, stacklevel=2)
    if meth not in (
            'nelder-mead', 'powell', 'l-bfgs-b', 'cobyla', 'cobyqa', 'slsqp',
            'tnc', 'trust-constr', '_custom') and bounds is not None:
        warn(f'Method {method} cannot handle bounds.',
             RuntimeWarning, stacklevel=2)
    # - return_all
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp') and
            options.get('return_all', False)):
        warn(f'Method {method} does not support the return_all option.',
             RuntimeWarning, stacklevel=2)

    # check gradient vector
    if callable(jac):
        pass
    elif jac is True:
        # fun returns func and grad
        fun = MemoizeJac(fun)
        jac = fun.derivative
    elif (jac in FD_METHODS and
          meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']):
        # finite differences with relative step
        pass
    elif meth in ['trust-constr']:
        # default jac calculation for this method
        jac = '2-point'
    elif jac is None or bool(jac) is False:
        # this will cause e.g. LBFGS to use forward difference, absolute step
        jac = None
    else:
        # default if jac option is not understood
        jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'cobyqa':
            options.setdefault('final_tr_radius', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)

    if meth == '_custom':
        # custom method called before bounds and constraints are 'standardised'
        # custom method should be able to accept whatever bounds/constraints
        # are provided to it.
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    constraints = standardize_constraints(constraints, x0, meth)

    remove_vars = False
    if bounds is not None:
        # convert to new-style bounds so we only have to consider one case
        bounds = standardize_bounds(bounds, x0, 'new')
        bounds = _validate_bounds(bounds, x0, meth)

        if meth in {"tnc", "slsqp", "l-bfgs-b"}:
            # These methods can't take the finite-difference derivatives they
            # need when a variable is fixed by the bounds. To avoid this issue,
            # remove fixed variables from the problem.
            # NOTE: if this list is expanded, then be sure to update the
            # accompanying tests and test_optimize.eb_data. Consider also if
            # default OptimizeResult will need updating.

            # determine whether any variables are fixed
            i_fixed = (bounds.lb == bounds.ub)

            if np.all(i_fixed):
                # all the parameters are fixed, a minimizer is not able to do
                # anything
                return _optimize_result_for_equal_bounds(
                    fun, bounds, meth, args=args, constraints=constraints
                )

            # determine whether finite differences are needed for any grad/jac
            fd_needed = (not callable(jac))
            for con in constraints:
                if not callable(con.get('jac', None)):
                    fd_needed = True

            # If finite differences are ever used, remove all fixed variables
            # Always remove fixed variables for TNC; see gh-14565
            remove_vars = i_fixed.any() and (fd_needed or meth == "tnc")
            if remove_vars:
                x_fixed = (bounds.lb)[i_fixed]
                x0 = x0[~i_fixed]
                bounds = _remove_from_bounds(bounds, i_fixed)
                fun = _remove_from_func(fun, i_fixed, x_fixed)
                if callable(callback):
                    callback = _remove_from_func(callback, i_fixed, x_fixed)
                if callable(jac):
                    jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)

                # make a copy of the constraints so the user's version doesn't
                # get changed. (Shallow copy is ok)
                constraints = [con.copy() for con in constraints]
                for con in constraints:  # yes, guaranteed to be a list
                    con['fun'] = _remove_from_func(con['fun'], i_fixed,
                                                   x_fixed, min_dim=1,
                                                   remove=0)
                    if callable(con.get('jac', None)):
                        con['jac'] = _remove_from_func(con['jac'], i_fixed,
                                                       x_fixed, min_dim=2,
                                                       remove=1)
        bounds = standardize_bounds(bounds, x0, meth)

    callback = _wrap_callback(callback, meth)

    if meth == 'nelder-mead':
        res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
                                   **options)
    elif meth == 'powell':
        res = _minimize_powell(fun, x0, args, callback, bounds, **options)
    elif meth == 'cg':
        res = _minimize_cg(fun, x0, args, jac, callback, **options)
    elif meth == 'bfgs':
        res = _minimize_bfgs(fun, x0, args, jac, callback, **options)
    elif meth == 'newton-cg':
        res = _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
                                 **options)
    elif meth == 'l-bfgs-b':
        res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
                               callback=callback, **options)
    elif meth == 'tnc':
        res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
                            **options)
    elif meth == 'cobyla':
        res = _minimize_cobyla(fun, x0, args, constraints, callback=callback,
                               bounds=bounds, **options)
    elif meth == 'cobyqa':
        res = _minimize_cobyqa(fun, x0, args, bounds, constraints, callback,
                               **options)
    elif meth == 'slsqp':
        res = _minimize_slsqp(fun, x0, args, jac, bounds,
                              constraints, callback=callback, **options)
    elif meth == 'trust-constr':
        res = _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp,
                                           bounds, constraints,
                                           callback=callback, **options)
    elif meth == 'dogleg':
        res = _minimize_dogleg(fun, x0, args, jac, hess,
                               callback=callback, **options)
    elif meth == 'trust-ncg':
        res = _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
                                  callback=callback, **options)
    elif meth == 'trust-krylov':
        res = _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
                                     callback=callback, **options)
    elif meth == 'trust-exact':
        res = _minimize_trustregion_exact(fun, x0, args, jac, hess,
                                          callback=callback, **options)
    else:
        raise ValueError(f'Unknown solver {method}')

    if remove_vars:
        res.x = _add_to_array(res.x, i_fixed, x_fixed)
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        if "hess_inv" in res:
            res.hess_inv = None  # unknown

    if getattr(callback, 'stop_iteration', False):
        res.success = False
        res.status = 99
        res.message = "`callback` raised `StopIteration`."

    return res


def minimize_scalar(fun, bracket=None, bounds=None, args=(),
                    method=None, tol=None, options=None):
    """Local minimization of scalar function of one variable.

    Parameters
    ----------
    fun : callable
        Objective function.
        Scalar function, must return a scalar.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.

    bracket : sequence, optional
        For methods 'brent' and 'golden', `bracket` defines the bracketing
        interval and is required.
        Either a triple ``(xa, xb, xc)`` satisfying ``xa < xb < xc`` and
        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair
        ``(xa, xb)`` to be used as initial points for a downhill bracket search
        (see `scipy.optimize.bracket`).
        The minimizer ``res.x`` will not necessarily satisfy
        ``xa <= res.x <= xb``.
    bounds : sequence, optional
        For method 'bounded', `bounds` is mandatory and must have two finite
        items corresponding to the optimization bounds.
    args : tuple, optional
        Extra arguments passed to the objective function.
    method : str or callable, optional
        Type of solver.  Should be one of:

        - :ref:`Brent <optimize.minimize_scalar-brent>`
        - :ref:`Bounded <optimize.minimize_scalar-bounded>`
        - :ref:`Golden <optimize.minimize_scalar-golden>`
        - custom - a callable object (added in version 0.14.0), see below

        Default is "Bounded" if bounds are provided and "Brent" otherwise.
        See the 'Notes' section for details of each solver.

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options.

        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            Set to True to print convergence messages.

        See :func:`show_options()` for solver-specific options.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize : Interface to minimization algorithms for scalar multivariate
        functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is the ``"Bounded"`` Brent method if
    `bounds` are passed and unbounded ``"Brent"`` otherwise.

    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
    algorithm [1]_ to find a local minimum.  The algorithm uses inverse
    parabolic interpolation when possible to speed up convergence of
    the golden section method.

    Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the
    golden section search technique [1]_. It uses analog of the bisection
    method to decrease the bracketed interval. It is usually
    preferable to use the *Brent* method.

    Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can
    perform bounded minimization [2]_ [3]_. It uses the Brent method to find a
    local minimum in the interval x1 < xopt < x2.

    Note that the Brent and Golden methods do not guarantee success unless a
    valid ``bracket`` triple is provided. If a three-point bracket cannot be
    found, consider `scipy.optimize.minimize`. Also, all methods are intended
    only for local minimization. When the function of interest has more than
    one local minimum, consider :ref:`global_optimization`.

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using some library frontend to minimize_scalar. You can simply
    pass a callable as the ``method`` parameter.

    The callable is called as ``method(fun, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `bracket`, `tol`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  The method
    shall return an `OptimizeResult` object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method. You can find an example in the scipy.optimize tutorial.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Press, W., S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery.
           Numerical Recipes in C. Cambridge University Press.
    .. [2] Forsythe, G.E., M. A. Malcolm, and C. B. Moler. "Computer Methods
           for Mathematical Computations." Prentice-Hall Series in Automatic
           Computation 259 (1977).
    .. [3] Brent, Richard P. Algorithms for Minimization Without Derivatives.
           Courier Corporation, 2013.

    Examples
    --------
    Consider the problem of minimizing the following function.

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    Using the *Brent* method, we find the local minimum as:

    >>> from scipy.optimize import minimize_scalar
    >>> res = minimize_scalar(f)
    >>> res.fun
    -9.9149495908

    The minimizer is:

    >>> res.x
    1.28077640403

    Using the *Bounded* method, we find a local minimum with specified
    bounds as:

    >>> res = minimize_scalar(f, bounds=(-3, -1), method='bounded')
    >>> res.fun  # minimum
    3.28365179850e-13
    >>> res.x  # minimizer
    -2.0000002026

    """
    if not isinstance(args, tuple):
        args = (args,)

    if callable(method):
        meth = "_custom"
    elif method is None:
        meth = 'brent' if bounds is None else 'bounded'
    else:
        meth = method.lower()
    if options is None:
        options = {}

    if bounds is not None and meth in {'brent', 'golden'}:
        message = f"Use of `bounds` is incompatible with 'method={method}'."
        raise ValueError(message)

    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; "
                 "defaulting to absolute tolerance.",
                 RuntimeWarning, stacklevel=2)
            options['xatol'] = tol
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            options.setdefault('xtol', tol)

    # replace boolean "disp" option, if specified, by an integer value.
    disp = options.get('disp')
    if isinstance(disp, bool):
        options['disp'] = 2 * int(disp)

    if meth == '_custom':
        res = method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        res = _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        res = _recover_from_bracket_error(_minimize_scalar_golden,
                                          fun, bracket, args, **options)
    else:
        raise ValueError(f'Unknown solver {method}')

    # gh-16196 reported inconsistencies in the output shape of `res.x`. While
    # fixing this, future-proof it for when the function is vectorized:
    # the shape of `res.x` should match that of `res.fun`.
    res.fun = np.asarray(res.fun)[()]
    res.x = np.reshape(res.x, res.fun.shape)[()]
    return res


def _remove_from_bounds(bounds, i_fixed):
    """Removes fixed variables from a `Bounds` instance"""
    lb = bounds.lb[~i_fixed]
    ub = bounds.ub[~i_fixed]
    return Bounds(lb, ub)  # don't mutate original Bounds object


def _remove_from_func(fun_in, i_fixed, x_fixed, min_dim=None, remove=0):
    """Wraps a function such that fixed variables need not be passed in"""
    def fun_out(x_in, *args, **kwargs):
        x_out = np.zeros_like(i_fixed, dtype=x_in.dtype)
        x_out[i_fixed] = x_fixed
        x_out[~i_fixed] = x_in
        y_out = fun_in(x_out, *args, **kwargs)
        y_out = np.array(y_out)

        if min_dim == 1:
            y_out = np.atleast_1d(y_out)
        elif min_dim == 2:
            y_out = np.atleast_2d(y_out)

        if remove == 1:
            y_out = y_out[..., ~i_fixed]
        elif remove == 2:
            y_out = y_out[~i_fixed, ~i_fixed]

        return y_out
    return fun_out


def _add_to_array(x_in, i_fixed, x_fixed):
    """Adds fixed variables back to an array"""
    i_free = ~i_fixed
    if x_in.ndim == 2:
        i_free = i_free[:, None] @ i_free[None, :]
    x_out = np.zeros_like(i_free, dtype=x_in.dtype)
    x_out[~i_free] = x_fixed
    x_out[i_free] = x_in.ravel()
    return x_out


def _validate_bounds(bounds, x0, meth):
    """Check that bounds are valid."""

    msg = "An upper bound is less than the corresponding lower bound."
    if np.any(bounds.ub < bounds.lb):
        raise ValueError(msg)

    msg = "The number of bounds is not compatible with the length of `x0`."
    try:
        bounds.lb = np.broadcast_to(bounds.lb, x0.shape)
        bounds.ub = np.broadcast_to(bounds.ub, x0.shape)
    except Exception as e:
        raise ValueError(msg) from e

    return bounds

def standardize_bounds(bounds, x0, meth):
    """Converts bounds to the form required by the solver."""
    if meth in {'trust-constr', 'powell', 'nelder-mead', 'cobyla', 'cobyqa',
                'new'}:
        if not isinstance(bounds, Bounds):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
    elif meth in ('l-bfgs-b', 'tnc', 'slsqp', 'old'):
        if isinstance(bounds, Bounds):
            bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
    return bounds


def standardize_constraints(constraints, x0, meth):
    """Converts constraints to the form required by the solver."""
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    new_constraint_types = all_constraint_types[:-1]
    if constraints is None:
        constraints = []
    elif isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    else:
        constraints = list(constraints)  # ensure it's a mutable sequence

    if meth in ['trust-constr', 'cobyqa', 'new']:
        for i, con in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)
    else:
        # iterate over copy, changing original
        for i, con in enumerate(list(constraints)):
            if isinstance(con, new_constraint_types):
                old_constraints = new_constraint_to_old(con, x0)
                constraints[i] = old_constraints[0]
                constraints.extend(old_constraints[1:])  # appends 1 if present

    return constraints


def _optimize_result_for_equal_bounds(
        fun, bounds, method, args=(), constraints=()
):
    """
    Provides a default OptimizeResult for when a bounded minimization method
    has (lb == ub).all().

    Parameters
    ----------
    fun: callable
    bounds: Bounds
    method: str
    constraints: Constraint
    """
    success = True
    message = 'All independent variables were fixed by bounds.'

    # bounds is new-style
    x0 = bounds.lb

    if constraints:
        message = ("All independent variables were fixed by bounds at values"
                   " that satisfy the constraints.")
        constraints = standardize_constraints(constraints, x0, 'new')

    maxcv = 0
    for c in constraints:
        pc = PreparedConstraint(c, x0)
        violation = pc.violation(x0)
        if np.sum(violation):
            maxcv = max(maxcv, np.max(violation))
            success = False
            message = (f"All independent variables were fixed by bounds, but "
                       f"the independent variables do not satisfy the "
                       f"constraints exactly. (Maximum violation: {maxcv}).")

    return OptimizeResult(
        x=x0, fun=fun(x0, *args), success=success, message=message, nfev=1,
        njev=0, nhev=0,
    )
<|endoftext|>"""
Routines for removing redundant (linearly dependent) equations from linear
programming equality constraints.
"""
# Author: Matt Haberland

import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm


def _row_count(A):
    """
    Counts the number of nonzeros in each row of input array A.
    Nonzeros are defined as any element with absolute value greater than
    tol = 1e-13. This value should probably be an input to the function.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix

    Returns
    -------
    rowcount : 1-D array
        Number of nonzeros in each row of A

    """
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()


def _get_densest(A, eligibleRows):
    """
    Returns the index of the densest row of A. Ignores rows that are not
    eligible for consideration.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    eligibleRows : 1-D logical array
        Values indicate whether the corresponding row of A is eligible
        to be considered

    Returns
    -------
    i_densest : int
        Index of the densest row in A eligible for consideration

    """
    rowCounts = _row_count(A)
    return np.argmax(rowCounts * eligibleRows)


def _remove_zero_rows(A, b):
    """
    Eliminates trivial equations from system of equations defined by Ax = b
   and identifies trivial infeasibilities

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the removal operation
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """
    status = 0
    message = ""
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    if not np.allclose(b[i_zero], 0):
        status = 2
        message = "There is a zero row in A_eq with a nonzero corresponding " \
                  "entry in b_eq. The problem is infeasible."
    b = b[np.logical_not(i_zero)]
    return A, b, status, message


def bg_update_dense(plu, perm_r, v, j):
    LU, p = plu

    vperm = v[perm_r]
    u = dtrsm(1, LU, vperm, lower=1, diag=1)
    LU[:j+1, j] = u[:j+1]
    l = u[j+1:]
    piv = LU[j, j]
    LU[j+1:, j] += (l/piv)
    return LU, p


def _remove_redundancy_pivot_dense(A, rhs, true_rank=None):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """
    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    d = []                  # Indices of dependent rows
    perm_r = None

    A_orig = A
    A = np.zeros((m, m + n), order='F')
    np.fill_diagonal(A, 1)
    A[:, m:] = A_orig
    e = np.zeros(m)

    js_candidates = np.arange(m, m+n, dtype=int)  # candidate columns for basis
    # manual masking was faster than masked array
    js_mask = np.ones(js_candidates.shape, dtype=bool)

    # Implements basic algorithm from [2]
    # Uses some of the suggested improvements (removing zero rows and
    # Bartels-Golub update idea).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis are only really useful if
    # the matrix is sparse.

    lu = np.eye(m, order='F'), np.arange(m)  # initial LU is trivial
    perm_r = lu[1]
    for i in v:

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        try:  # fails for i==0 and any time it gets ill-conditioned
            j = b[i-1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
        except Exception:
            lu = scipy.linalg.lu_factor(A[:, b])
            LU, p = lu
            perm_r = list(range(m))
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        js = js_candidates[js_mask]
        batch = 50

        # This is a tiny bit faster than looping over columns individually,
        # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
        for j_index in range(0, len(js), batch):
            j_indices = js[j_index: min(j_index+batch, len(js))]

            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]  # very independent column
                b[i] = j
                js_mask[j-m] = False
                break
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar)/(1+bnorm) > tolprimal:  # inconsistent
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # dependent
                d.append(i)
                if true_rank is not None and len(d) == m - true_rank:
                    break   # found all redundancies

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], rhs[keep], status, message


def _remove_redundancy_pivot_sparse(A, rhs):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """

    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    k = set(range(m, m+n))  # Structural column indices.
    d = []                  # Indices of dependent rows

    A_orig = A
    A = scipy.sparse.hstack((scipy.sparse.eye(m), A)).tocsc()
    e = np.zeros(m)

    # Implements basic algorithm from [2]
    # Uses only one of the suggested improvements (removing zero rows).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis sound useful, but the
    # description of the procedure seems to assume a lot of familiarity with
    # the subject; it is not very explicit. I already went through enough
    # trouble getting the basic algorithm working, so I was not interested in
    # trying to decipher this, too. (Overall, the paper is fraught with
    # mistakes and ambiguities - which is strange, because the rest of
    # Andersen's papers are quite good.)
    # I tried and tried and tried to improve performance using the
    # Bartels-Golub update. It works, but it's only practical if the LU
    # factorization can be specialized as described, and that is not possible
    # until the SciPy SuperLU interface permits control over column
    # permutation - see issue #7700.

    for i in v:
        B = A[:, b]

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        pi = scipy.sparse.linalg.spsolve(B.transpose(), e).reshape(-1, 1)

        js = list(k-set(b))  # not efficient, but this is not the time sink...

        # Due to overhead, it tends to be faster (for problems tested) to
        # compute the full matrix-vector product rather than individual
        # vector-vector products (with the chance of terminating as soon
        # as any are nonzero). For very large matrices, it might be worth
        # it to compute, say, 100 or 1000 at a time and stop when a nonzero
        # is found.

        c = (np.abs(A[:, js].transpose().dot(pi)) > tolapiv).nonzero()[0]
        if len(c) > 0:  # independent
            j = js[c[0]]
            # in a previous commit, the previous line was changed to choose
            # index j corresponding with the maximum dot product.
            # While this avoided issues with almost
            # singular matrices, it slowed the routine in most NETLIB tests.
            # I think this is because these columns were denser than the
            # first column with nonzero dot product (c[0]).
            # It would be nice to have a heuristic that balances sparsity with
            # high dot product, but I don't think it's worth the time to
            # develop one right now. Bartels-Golub update is a much higher
            # priority.
            b[i] = j  # replace artificial column
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar)/(1 + bnorm) > tolprimal:
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # dependent
                d.append(i)

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], rhs[keep], status, message


def _remove_redundancy_svd(A, b):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """

    A, b, status, message = _remove_zero_rows(A, b)

    if status != 0:
        return A, b, status, message

    U, s, Vh = svd(A)
    eps = np.finfo(float).eps
    tol = s.max() * max(A.shape) * eps

    m, n = A.shape
    s_min = s[-1] if m <= n else 0

    # this algorithm is faster than that of [2] when the nullspace is small
    # but it could probably be improvement by randomized algorithms and with
    # a sparse implementation.
    # it relies on repeated singular value decomposition to find linearly
    # dependent rows (as identified by columns of U that correspond with zero
    # singular values). Unfortunately, only one row can be removed per
    # decomposition (I tried otherwise; doing so can cause problems.)
    # It would be nice if we could do truncated SVD like sp.sparse.linalg.svds
    # but that function is unreliable at finding singular values near zero.
    # Finding max eigenvalue L of A A^T, then largest eigenvalue (and
    # associated eigenvector) of -A A^T + L I (I is identity) via power
    # iteration would also work in theory, but is only efficient if the
    # smallest nonzero eigenvalue of A A^T is close to the largest nonzero
    # eigenvalue.

    while abs(s_min) < tol:
        v = U[:, -1]  # TODO: return these so user can eliminate from problem?
        # rows need to be represented in significant amount
        eligibleRows = np.abs(v) > tol * 10e6
        if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
            status = 4
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            break
        if np.any(np.abs(v.dot(b)) > tol * 100):  # factor of 100 to fix 10038 and 10349
            status = 2
            message = ("There is a linear combination of rows of A_eq that "
                       "results in zero, suggesting a redundant constraint. "
                       "However the same linear combination of b_eq is "
                       "nonzero, suggesting that the constraints conflict "
                       "and the problem is infeasible.")
            break

        i_remove = _get_densest(A, eligibleRows)
        A = np.delete(A, i_remove, axis=0)
        b = np.delete(b, i_remove)
        U, s, Vh = svd(A)
        m, n = A.shape
        s_min = s[-1] if m <= n else 0

    return A, b, status, message


def _remove_redundancy_id(A, rhs, rank=None, randomized=True):
    """Eliminates redundant equations from a system of equations.

    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    rank : int, optional
        The rank of A
    randomized: bool, optional
        True for randomized interpolative decomposition

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """

    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")

    A, rhs, status, message = _remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    k = rank
    if rank is None:
        k = np.linalg.matrix_rank(A)

    idx, proj = interp_decomp(A.T, k, rand=randomized)

    # first k entries in idx are indices of the independent rows
    # remaining entries are the indices of the m-k dependent rows
    # proj provides a linear combinations of rows of A2 that form the
    # remaining m-k (dependent) rows. The same linear combination of entries
    # in rhs2 must give the remaining m-k entries. If not, the system is
    # inconsistent, and the problem is infeasible.
    if not np.allclose(rhs[idx[:k]] @ proj, rhs[idx[k:]]):
        status = 2
        message = inconsistent

    # sort indices because the other redundancy removal routines leave rows
    # in original order and tests were written with that in mind
    idx = sorted(idx[:k])
    A2 = A[idx, :]
    rhs2 = rhs[idx]
    return A2, rhs2, status, message
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="minpack2",
                                   private_modules=["_minpack2"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'fmin_cobyla',
]

def __dir__():
    return __all__

def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="cobyla",
                                   private_modules=["_cobyla_py"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Interface to Constrained Optimization By Linear Approximation

Functions
---------
.. autosummary::
   :toctree: generated/

    fmin_cobyla

"""

import functools
from threading import RLock

import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,
    _prepare_scalar_function)
try:
    from itertools import izip
except ImportError:
    izip = zip

__all__ = ['fmin_cobyla']

# Workaround as _cobyla.minimize is not threadsafe
# due to an unknown f2py bug and can segfault,
# see gh-9658.
_module_lock = RLock()
def synchronized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _module_lock:
            return func(*args, **kwargs)
    return wrapper

@synchronized
def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
                rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4,
                *, callback=None):
    """
    Minimize a function using the Constrained Optimization By Linear
    Approximation (COBYLA) method. This method wraps a FORTRAN
    implementation of the algorithm.

    Parameters
    ----------
    func : callable
        Function to minimize. In the form func(x, \\*args).
    x0 : ndarray
        Initial guess.
    cons : sequence
        Constraint functions; must all be ``>=0`` (a single function
        if only 1 constraint). Each function takes the parameters `x`
        as its first argument, and it can return either a single number or
        an array or list of numbers.
    args : tuple, optional
        Extra arguments to pass to function.
    consargs : tuple, optional
        Extra arguments to pass to constraint functions (default of None means
        use same extra arguments as those passed to func).
        Use ``()`` for no extra arguments.
    rhobeg : float, optional
        Reasonable initial changes to the variables.
    rhoend : float, optional
        Final accuracy in the optimization (not precisely guaranteed). This
        is a lower bound on the size of the trust region.
    disp : {0, 1, 2, 3}, optional
        Controls the frequency of output; 0 implies no output.
    maxfun : int, optional
        Maximum number of function evaluations.
    catol : float, optional
        Absolute tolerance for constraint violations.
    callback : callable, optional
        Called after each iteration, as ``callback(x)``, where ``x`` is the
        current parameter vector.

    Returns
    -------
    x : ndarray
        The argument that minimises `f`.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'COBYLA' `method` in particular.

    Notes
    -----
    This algorithm is based on linear approximations to the objective
    function and each constraint. We briefly describe the algorithm.

    Suppose the function is being minimized over k variables. At the
    jth iteration the algorithm has k+1 points v_1, ..., v_(k+1),
    an approximate solution x_j, and a radius RHO_j.
    (i.e., linear plus a constant) approximations to the objective
    function and constraint functions such that their function values
    agree with the linear approximation on the k+1 points v_1,.., v_(k+1).
    This gives a linear program to solve (where the linear approximations
    of the constraint functions are constrained to be non-negative).

    However, the linear approximations are likely only good
    approximations near the current simplex, so the linear program is
    given the further requirement that the solution, which
    will become x_(j+1), must be within RHO_j from x_j. RHO_j only
    decreases, never increases. The initial RHO_j is rhobeg and the
    final RHO_j is rhoend. In this way COBYLA's iterations behave
    like a trust region algorithm.

    Additionally, the linear program may be inconsistent, or the
    approximation may give poor improvement. For details about
    how these issues are resolved, as well as how the points v_i are
    updated, refer to the source code or the references below.


    References
    ----------
    Powell M.J.D. (1994), "A direct search optimization method that models
    the objective and constraint functions by linear interpolation.", in
    Advances in Optimization and Numerical Analysis, eds. S. Gomez and
    J-P Hennart, Kluwer Academic (Dordrecht), pp. 51-67

    Powell M.J.D. (1998), "Direct search algorithms for optimization
    calculations", Acta Numerica 7, 287-336

    Powell M.J.D. (2007), "A view of algorithms for optimization without
    derivatives", Cambridge University Technical Report DAMTP 2007/NA03


    Examples
    --------
    Minimize the objective function f(x,y) = x*y subject
    to the constraints x**2 + y**2 < 1 and y > 0::

        >>> def objective(x):
        ...     return x[0]*x[1]
        ...
        >>> def constr1(x):
        ...     return 1 - (x[0]**2 + x[1]**2)
        ...
        >>> def constr2(x):
        ...     return x[1]
        ...
        >>> from scipy.optimize import fmin_cobyla
        >>> fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7)
        array([-0.70710685,  0.70710671])

    The exact solution is (-sqrt(2)/2, sqrt(2)/2).



    """
    err = "cons must be a sequence of callable functions or a single"\
          " callable function."
    try:
        len(cons)
    except TypeError as e:
        if callable(cons):
            cons = [cons]
        else:
            raise TypeError(err) from e
    else:
        for thisfunc in cons:
            if not callable(thisfunc):
                raise TypeError(err)

    if consargs is None:
        consargs = args

    # build constraints
    con = tuple({'type': 'ineq', 'fun': c, 'args': consargs} for c in cons)

    # options
    opts = {'rhobeg': rhobeg,
            'tol': rhoend,
            'disp': disp,
            'maxiter': maxfun,
            'catol': catol,
            'callback': callback}

    sol = _minimize_cobyla(func, x0, args, constraints=con,
                           **opts)
    if disp and not sol['success']:
        print(f"COBYLA failed to find a solution: {sol.message}")
    return sol['x']


@synchronized
def _minimize_cobyla(fun, x0, args=(), constraints=(),
                     rhobeg=1.0, tol=1e-4, maxiter=1000,
                     disp=False, catol=2e-4, callback=None, bounds=None,
                     **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.

    Options
    -------
    rhobeg : float
        Reasonable initial changes to the variables.
    tol : float
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored as set to 0.
    maxiter : int
        Maximum number of function evaluations.
    catol : float
        Tolerance (absolute) for constraint violations

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    rhoend = tol
    iprint = int(bool(disp))

    # check constraints
    if isinstance(constraints, dict):
        constraints = (constraints, )

    if bounds:
        i_lb = np.isfinite(bounds.lb)
        if np.any(i_lb):
            def lb_constraint(x, *args, **kwargs):
                return x[i_lb] - bounds.lb[i_lb]

            constraints.append({'type': 'ineq', 'fun': lb_constraint})

        i_ub = np.isfinite(bounds.ub)
        if np.any(i_ub):
            def ub_constraint(x):
                return bounds.ub[i_ub] - x[i_ub]

            constraints.append({'type': 'ineq', 'fun': ub_constraint})

    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype != 'ineq':
                raise ValueError(f"Constraints of type '{con['type']}' not handled by "
                                 "COBYLA.")

        # check function
        if 'fun' not in con:
            raise KeyError('Constraint %d has no function defined.' % ic)

        # check extra arguments
        if 'args' not in con:
            con['args'] = ()

    # m is the total number of constraint values
    # it takes into account that some constraints may be vector-valued
    cons_lengths = []
    for c in constraints:
        f = c['fun'](x0, *c['args'])
        try:
            cons_length = len(f)
        except TypeError:
            cons_length = 1
        cons_lengths.append(cons_length)
    m = sum(cons_lengths)

    # create the ScalarFunction, cobyla doesn't require derivative function
    def _jac(x, *args):
        return None

    sf = _prepare_scalar_function(fun, x0, args=args, jac=_jac)

    def calcfc(x, con):
        f = sf.fun(x)
        i = 0
        for size, c in izip(cons_lengths, constraints):
            con[i: i + size] = c['fun'](x, *c['args'])
            i += size
        return f

    def wrapped_callback(x):
        if callback is not None:
            callback(np.copy(x))

    info = np.zeros(4, np.float64)
    xopt, info = cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg,
                                  rhoend=rhoend, iprint=iprint, maxfun=maxfun,
                                  dinfo=info, callback=wrapped_callback)

    if info[3] > catol:
        # Check constraint violation
        info[0] = 4

    return OptimizeResult(x=xopt,
                          status=int(info[0]),
                          success=info[0] == 1,
                          message={1: 'Optimization terminated successfully.',
                                   2: 'Maximum number of function evaluations '
                                      'has been exceeded.',
                                   3: 'Rounding errors are becoming damaging '
                                      'in COBYLA subroutine.',
                                   4: 'Did not converge to a solution '
                                      'satisfying the constraints. See '
                                      '`maxcv` for magnitude of violation.',
                                   5: 'NaN result encountered.'
                                   }.get(info[0], 'Unknown exit status.'),
                          nfev=int(info[1]),
                          fun=info[2],
                          maxcv=info[3])
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'OptimizeWarning',
    'curve_fit',
    'fixed_point',
    'fsolve',
    'least_squares',
    'leastsq',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="minpack",
                                   private_modules=["_minpack_py"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
A top-level linear programming interface.

.. versionadded:: 0.15.0

Functions
---------
.. autosummary::
   :toctree: generated/

    linprog
    linprog_verbose_callback
    linprog_terse_callback

"""

import numpy as np

from ._optimize import OptimizeResult, OptimizeWarning
from warnings import warn
from ._linprog_highs import _linprog_highs
from ._linprog_ip import _linprog_ip
from ._linprog_simplex import _linprog_simplex
from ._linprog_rs import _linprog_rs
from ._linprog_doc import (_linprog_highs_doc, _linprog_ip_doc,  # noqa: F401
                           _linprog_rs_doc, _linprog_simplex_doc,
                           _linprog_highs_ipm_doc, _linprog_highs_ds_doc)
from ._linprog_util import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)
from copy import deepcopy

__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']

__docformat__ = "restructuredtext en"

LINPROG_METHODS = [
    'simplex', 'revised simplex', 'interior-point', 'highs', 'highs-ds', 'highs-ipm'
]


def linprog_verbose_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization:

            ``0`` : Optimization terminated successfully

            ``1`` : Iteration limit reached

            ``2`` : Problem appears to be infeasible

            ``3`` : Problem appears to be unbounded

            ``4`` : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    x = res['x']
    fun = res['fun']
    phase = res['phase']
    status = res['status']
    nit = res['nit']
    message = res['message']
    complete = res['complete']

    saved_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: f"{x: 12.4f}"})
    if status:
        print('--------- Simplex Early Exit -------\n')
        print(f'The simplex method exited early with status {status:d}')
        print(message)
    elif complete:
        print('--------- Simplex Complete --------\n')
        print(f'Iterations required: {nit}')
    else:
        print(f'--------- Iteration {nit:d}  ---------\n')

    if nit > 0:
        if phase == 1:
            print('Current Pseudo-Objective Value:')
        else:
            print('Current Objective Value:')
        print('f = ', fun)
        print()
        print('Current Solution Vector:')
        print('x = ', x)
        print()

    np.set_printoptions(**saved_printoptions)


def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization:

            ``0`` : Optimization terminated successfully

            ``1`` : Iteration limit reached

            ``2`` : Problem appears to be infeasible

            ``3`` : Problem appears to be unbounded

            ``4`` : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    nit = res['nit']
    x = res['x']

    if nit == 0:
        print("Iter:   X:")
    print(f"{nit: <5d}   ", end="")
    print(x)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=(0, None), method='highs', callback=None,
            options=None, x0=None, integrality=None):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints.

    Linear programming solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

    - minimize ::

        c @ x

    - such that ::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None``. Other bounds can be
    specified with ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable.
        If a single tuple ``(min, max)`` is provided, then ``min`` and ``max``
        will serve as bounds for all decision variables.
        Use ``None`` to indicate that there is no bound. For instance, the
        default bound ``(0, None)`` means that all decision variables are
        non-negative, and the pair ``(None, None)`` means no bounds at all,
        i.e. all variables are allowed to be any real.
    method : str, optional
        The algorithm used to solve the standard form problem.
        The following are supported.

        - :ref:`'highs' <optimize.linprog-highs>` (default)
        - :ref:`'highs-ds' <optimize.linprog-highs-ds>`
        - :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
        - :ref:`'interior-point' <optimize.linprog-interior-point>` (legacy)
        - :ref:`'revised simplex' <optimize.linprog-revised_simplex>` (legacy)
        - :ref:`'simplex' <optimize.linprog-simplex>` (legacy)

        The legacy methods are deprecated and will be removed in SciPy 1.11.0.
    callback : callable, optional
        If a callback function is provided, it will be called at least once per
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The current solution vector.
        fun : float
            The current value of the objective function ``c @ x``.
        success : bool
            ``True`` when the algorithm has completed successfully.
        slack : 1-D array
            The (nominally positive) values of the slack,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        phase : int
            The phase of the algorithm being executed.
        status : int
            An integer representing the status of the algorithm.

            ``0`` : Optimization proceeding nominally.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The current iteration number.
        message : str
            A string descriptor of the algorithm status.

        Callback functions are not currently supported by the HiGHS methods.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        options:

        maxiter : int
            Maximum number of iterations to perform.
            Default: see method-specific documentation.
        disp : bool
            Set to ``True`` to print convergence messages.
            Default: ``False``.
        presolve : bool
            Set to ``False`` to disable automatic presolve.
            Default: ``True``.

        All methods except the HiGHS solvers also accept:

        tol : float
            A tolerance which determines when a residual is "close enough" to
            zero to be considered exactly zero.
        autoscale : bool
            Set to ``True`` to automatically perform equilibration.
            Consider using this option if the numerical values in the
            constraints are separated by several orders of magnitude.
            Default: ``False``.
        rr : bool
            Set to ``False`` to disable automatic redundancy removal.
            Default: ``True``.
        rr_method : string
            Method used to identify and remove redundant rows from the
            equality constraint matrix after presolve. For problems with
            dense input, the available methods for redundancy removal are:

            ``SVD``:
                Repeatedly performs singular value decomposition on
                the matrix, detecting redundant rows based on nonzeros
                in the left singular vectors that correspond with
                zero singular values. May be fast when the matrix is
                nearly full rank.
            ``pivot``:
                Uses the algorithm presented in [5]_ to identify
                redundant rows.
            ``ID``:
                Uses a randomized interpolative decomposition.
                Identifies columns of the matrix transpose not used in
                a full-rank interpolative decomposition of the matrix.
            ``None``:
                Uses ``svd`` if the matrix is nearly full rank, that is,
                the difference between the matrix rank and the number
                of rows is less than five. If not, uses ``pivot``. The
                behavior of this default is subject to change without
                prior notice.

            Default: None.
            For problems with sparse input, this option is ignored, and the
            pivot-based algorithm presented in [5]_ is used.

        For method-specific options, see
        :func:`show_options('linprog') <show_options>`.

    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>` method,
        and can only be used if `x0` represents a basic feasible solution.

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

        For mixed integrality constraints, supply an array of shape ``c.shape``.
        To infer a constraint on each decision variable from shorter inputs,
        the argument will be broadcast to ``c.shape`` using `numpy.broadcast_to`.

        This argument is currently used only by the
        :ref:`'highs' <optimize.linprog-highs>` method and is ignored otherwise.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields
        below. Note that the return types of the fields may depend on whether
        the optimization was successful, therefore it is recommended to check
        `OptimizeResult.status` before relying on the other fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.

    See Also
    --------
    show_options : Additional options accepted by the solvers.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    :ref:`'highs-ds' <optimize.linprog-highs-ds>`, and
    :ref:`'highs-ipm' <optimize.linprog-highs-ipm>` are interfaces to the
    HiGHS simplex and interior-point method solvers [13]_, respectively.
    :ref:`'highs' <optimize.linprog-highs>` (default) chooses between
    the two automatically. These are the fastest linear
    programming solvers in SciPy, especially for large, sparse problems;
    which of these two is faster is problem-dependent.
    The other solvers are legacy methods and will be removed when `callback` is
    supported by the HiGHS methods.

    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>`, is a wrapper of the C++ high
    performance dual revised simplex implementation (HSOL) [13]_, [14]_.
    Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>` is a wrapper of a C++
    implementation of an **i**\ nterior-\ **p**\ oint **m**\ ethod [13]_; it
    features a crossover routine, so it is as accurate as a simplex solver.
    Method :ref:`'highs' <optimize.linprog-highs>` chooses between the two
    automatically.
    For new code involving `linprog`, we recommend explicitly choosing one of
    these three method values.

    .. versionadded:: 1.6.0

    Method :ref:`'interior-point' <optimize.linprog-interior-point>`
    uses the primal-dual path following algorithm
    as outlined in [4]_. This algorithm supports sparse constraint matrices and
    is typically faster than the simplex methods, especially for large, sparse
    problems. Note, however, that the solution returned may be slightly less
    accurate than those of the simplex methods and will not, in general,
    correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

    Method :ref:`'revised simplex' <optimize.linprog-revised_simplex>`
    uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    .. versionadded:: 1.3.0

    Method :ref:`'simplex' <optimize.linprog-simplex>` uses a traditional,
    full-tableau implementation of
    Dantzig's simplex algorithm [1]_, [2]_ (*not* the
    Nelder-Mead simplex). This algorithm is included for backwards
    compatibility and educational purposes.

    .. versionadded:: 0.15.0

    Before applying :ref:`'interior-point' <optimize.linprog-interior-point>`,
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, or
    :ref:`'simplex' <optimize.linprog-simplex>`,
    a presolve procedure based on [8]_ attempts
    to identify trivial infeasibilities, trivial unboundedness, and potential
    problem simplifications. Specifically, it checks for:

    - rows of zeros in ``A_eq`` or ``A_ub``, representing trivial constraints;
    - columns of zeros in ``A_eq`` `and` ``A_ub``, representing unconstrained
      variables;
    - column singletons in ``A_eq``, representing fixed variables; and
    - column singletons in ``A_ub``, representing simple bounds.

    If presolve reveals that the problem is unbounded (e.g. an unconstrained
    and unbounded variable has negative cost) or infeasible (e.g., a row of
    zeros in ``A_eq`` corresponds with a nonzero in ``b_eq``), the solver
    terminates with the appropriate status code. Note that presolve terminates
    as soon as any sign of unboundedness is detected; consequently, a problem
    may be reported as unbounded when in reality the problem is infeasible
    (but infeasibility has not been detected yet). Therefore, if it is
    important to know whether the problem is actually infeasible, solve the
    problem again with option ``presolve=False``.

    If neither infeasibility nor unboundedness are detected in a single pass
    of the presolve, bounds are tightened where possible and fixed
    variables are removed from the problem. Then, linearly dependent rows
    of the ``A_eq`` matrix are removed, (unless they represent an
    infeasibility) to avoid numerical difficulties in the primary solve
    routine. Note that rows that are nearly linearly dependent (within a
    prescribed tolerance) may also be removed, which can change the optimal
    solution in rare cases. If this is a concern, eliminate redundancy from
    your problem formulation and run with option ``rr=False`` or
    ``presolve=False``.

    Several potential improvements can be made here: additional presolve
    checks outlined in [8]_ should be implemented, the presolve routine should
    be run multiple times (until no further simplifications can be made), and
    more of the efficiency improvements from [5]_ should be implemented in the
    redundancy removal routines.

    After presolve, the problem is transformed to standard form by converting
    the (tightened) simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.
    Optionally, the problem is automatically scaled via equilibration [12]_.
    The selected algorithm solves the standard form problem, and a
    postprocessing routine converts the result to a solution to the original
    problem.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point
            methods for large scale linear programming. HEC/Universite de
            Geneve, 1996.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    .. [12] Tomlin, J. A. "On scaling linear programming problems."
            Mathematical Programming Study 4 (1975): 146-166.
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
            "HiGHS - high performance software for linear optimization."
            https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
            simplex method." Mathematical Programming Computation, 10 (1),
            119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the following problem:

    .. math::

        \min_{x_0, x_1} \ -x_0 + 4x_1 & \\
        \mbox{such that} \ -3x_0 + x_1 & \leq 6,\\
        -x_0 - 2x_1 & \geq -4,\\
        x_1 & \geq -3.

    The problem is not presented in the form accepted by `linprog`. This is
    easily remedied by converting the "greater than" inequality
    constraint to a "less than" inequality constraint by
    multiplying both sides by a factor of :math:`-1`. Note also that the last
    constraint is really the simple bound :math:`-3 \leq x_1 \leq \infty`.
    Finally, since there are no bounds on :math:`x_0`, we must explicitly
    specify the bounds :math:`-\infty \leq x_0 \leq \infty`, as the
    default is for variables to be non-negative. After collecting coeffecients
    into arrays and tuples, the input for this problem is:

    >>> from scipy.optimize import linprog
    >>> c = [-1, 4]
    >>> A = [[-3, 1], [1, 2]]
    >>> b = [6, 4]
    >>> x0_bounds = (None, None)
    >>> x1_bounds = (-3, None)
    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
    >>> res.fun
    -22.0
    >>> res.x
    array([10., -3.])
    >>> res.message
    'Optimization terminated successfully. (HiGHS Status 7: Optimal)'

    The marginals (AKA dual values / shadow prices / Lagrange multipliers)
    and residuals (slacks) are also available.

    >>> res.ineqlin
      residual: [ 3.900e+01  0.000e+00]
     marginals: [-0.000e+00 -1.000e+00]

    For example, because the marginal associated with the second inequality
    constraint is -1, we expect the optimal value of the objective function
    to decrease by ``eps`` if we add a small amount ``eps`` to the right hand
    side of the second inequality constraint:

    >>> eps = 0.05
    >>> b[1] += eps
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.05

    Also, because the residual on the first inequality constraint is 39, we
    can decrease the right hand side of the first constraint by 39 without
    affecting the optimal solution.

    >>> b = [6, 4]  # reset to original values
    >>> b[0] -= 39
    >>> linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
    -22.0

    """

    meth = method.lower()
    methods = {"highs", "highs-ds", "highs-ipm",
               "simplex", "revised simplex", "interior-point"}

    if meth not in methods:
        raise ValueError(f"Unknown solver '{method}'")

    if x0 is not None and meth != "revised simplex":
        warning_message = "x0 is used only when method is 'revised simplex'. "
        warn(warning_message, OptimizeWarning, stacklevel=2)

    if np.any(integrality) and not meth == "highs":
        integrality = None
        warning_message = ("Only `method='highs'` supports integer "
                           "constraints. Ignoring `integrality`.")
        warn(warning_message, OptimizeWarning, stacklevel=2)
    elif np.any(integrality):
        integrality = np.broadcast_to(integrality, np.shape(c))
    else:
        integrality = None

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)
    lp, solver_options = _parse_linprog(lp, options, meth)
    tol = solver_options.get('tol', 1e-9)

    # Give unmodified problem to HiGHS
    if meth.startswith('highs'):
        if callback is not None:
            raise NotImplementedError("HiGHS solvers do not support the "
                                      "callback interface.")
        highs_solvers = {'highs-ipm': 'ipm', 'highs-ds': 'simplex',
                         'highs': None}

        sol = _linprog_highs(lp, solver=highs_solvers[meth],
                             **solver_options)
        sol['status'], sol['message'] = (
            _check_result(sol['x'], sol['fun'], sol['status'], sol['slack'],
                          sol['con'], lp.bounds, tol, sol['message'],
                          integrality))
        sol['success'] = sol['status'] == 0
        return OptimizeResult(sol)

    warn(f"`method='{meth}'` is deprecated and will be removed in SciPy "
         "1.11.0. Please use one of the HiGHS solvers (e.g. "
         "`method='highs'`) in new code.", DeprecationWarning, stacklevel=2)

    iteration = 0
    complete = False  # will become True if solved in presolve
    undo = []

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    lp_o = deepcopy(lp)

    # Solve trivial problem, eliminate variables, tighten bounds, etc.
    rr_method = solver_options.pop('rr_method', None)  # need to pop these;
    rr = solver_options.pop('rr', True)  # they're not passed to methods
    c0 = 0  # we might get a constant term in the objective
    if solver_options.pop('presolve', True):
        (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr,
                                                                 rr_method,
                                                                 tol)

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    if not complete:
        A, b, c, c0, x0 = _get_Abc(lp, c0)
        if solver_options.pop('autoscale', False):
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            postsolve_args = postsolve_args[:-2] + (C, b_scale)

        if meth == 'simplex':
            x, status, message, iteration = _linprog_simplex(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'interior-point':
            x, status, message, iteration = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'revised simplex':
            x, status, message, iteration = _linprog_rs(
                c, c0=c0, A=A, b=b, x0=x0, callback=callback,
                postsolve_args=postsolve_args, **solver_options)

    # Eliminate artificial variables, re-introduce presolved variables, etc.
    disp = solver_options.get('disp', False)

    x, fun, slack, con = _postsolve(x, postsolve_args, complete)

    status, message = _check_result(x, fun, status, slack, con, lp_o.bounds,
                                    tol, message, integrality)

    if disp:
        _display_summary(message, status, fun, iteration)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0}

    return OptimizeResult(sol)
<|endoftext|>import numpy as np
from ._cython_nnls import _nnls


__all__ = ['nnls']


def nnls(A, b, maxiter=None, *, atol=None):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    This problem, often called as NonNegative Least Squares, is a convex
    optimization problem with convex constraints. It typically arises when
    the ``x`` models quantities for which only nonnegative values are
    attainable; weight of ingredients, component costs and so on.

    Parameters
    ----------
    A : (m, n) ndarray
        Coefficient array
    b : (m,) ndarray, float
        Right-hand side vector.
    maxiter: int, optional
        Maximum number of iterations, optional. Default value is ``3 * n``.
    atol: float
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(A.T @ (A x - b)`` entries. Increasing this
        value relaxes the solution constraints. A typical relaxation value can
        be selected as ``max(m, n) * np.linalg.norm(a, 1) * np.spacing(1.)``.
        This value is not set as default since the norm operation becomes
        expensive for large problems hence can be used only when necessary.

    Returns
    -------
    x : ndarray
        Solution vector.
    rnorm : float
        The 2-norm of the residual, ``|| Ax-b ||_2``.

    See Also
    --------
    lsq_linear : Linear least squares with bounds on the variables

    Notes
    -----
    The code is based on [2]_ which is an improved version of the classical
    algorithm of [1]_. It utilizes an active set method and solves the KKT
    (Karush-Kuhn-Tucker) conditions for the non-negative least squares problem.

    References
    ----------
    .. [1] : Lawson C., Hanson R.J., "Solving Least Squares Problems", SIAM,
       1995, :doi:`10.1137/1.9781611971217`
    .. [2] : Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-
       Constrained Least Squares Algorithm", Journal Of Chemometrics, 1997,
       :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`

     Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import nnls
    ...
    >>> A = np.array([[1, 0], [1, 0], [0, 1]])
    >>> b = np.array([2, 1, 1])
    >>> nnls(A, b)
    (array([1.5, 1. ]), 0.7071067811865475)

    >>> b = np.array([-1, -1, -1])
    >>> nnls(A, b)
    (array([0., 0.]), 1.7320508075688772)

    """

    A = np.asarray_chkfinite(A, dtype=np.float64, order='C')
    b = np.asarray_chkfinite(b, dtype=np.float64)

    if len(A.shape) != 2:
        raise ValueError("Expected a two-dimensional array (matrix)" +
                         f", but the shape of A is {A.shape}")
    if len(b.shape) != 1:
        raise ValueError("Expected a one-dimensional array (vector)" +
                         f", but the shape of b is {b.shape}")

    m, n = A.shape

    if m != b.shape[0]:
        raise ValueError(
                "Incompatible dimensions. The first dimension of " +
                f"A is {m}, while the shape of b is {(b.shape[0], )}")

    if not maxiter:
        maxiter = 3*n
    x, rnorm, info = _nnls(A, b, maxiter)
    if info == -1:
        raise RuntimeError("Maximum number of iterations reached.")

    return x, rnorm
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'LbfgsInvHessProduct',
    'OptimizeResult',
    'fmin_l_bfgs_b',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="lbfgsb",
                                   private_modules=["_lbfgsb_py"], all=__all__,
                                   attribute=name)
<|endoftext|># Dual Annealing implementation.
# Copyright (c) 2018 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, Yang Xiang, PMP S.A.

"""
A Dual Annealing global optimization algorithm
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state, _transition_to_rng
from scipy.optimize._constraints import new_bounds_to_old

__all__ = ['dual_annealing']


class VisitingDistribution:
    """
    Class used to generate new coordinates based on the distorted
    Cauchy-Lorentz distribution. Depending on the steps within the strategy
    chain, the class implements the strategy for generating new location
    changes.

    Parameters
    ----------
    lb : array_like
        A 1-D NumPy ndarray containing lower bounds of the generated
        components. Neither NaN or inf are allowed.
    ub : array_like
        A 1-D NumPy ndarray containing upper bounds for the generated
        components. Neither NaN or inf are allowed.
    visiting_param : float
        Parameter for visiting distribution. Default value is 2.62.
        Higher values give the visiting distribution a heavier tail, this
        makes the algorithm jump to a more distant region.
        The value range is (1, 3]. Its value is fixed for the life of the
        object.
    rng_gen : {`~numpy.random.Generator`}
        A `~numpy.random.Generator` object for generating new locations.
        (can be a `~numpy.random.RandomState` object until SPEC007 transition
         is fully complete).

    """
    TAIL_LIMIT = 1.e8
    MIN_VISIT_BOUND = 1.e-10

    def __init__(self, lb, ub, visiting_param, rng_gen):
        # if you wish to make _visiting_param adjustable during the life of
        # the object then _factor2, _factor3, _factor5, _d1, _factor6 will
        # have to be dynamically calculated in `visit_fn`. They're factored
        # out here so they don't need to be recalculated all the time.
        self._visiting_param = visiting_param
        self.rng_gen = rng_gen
        self.lower = lb
        self.upper = ub
        self.bound_range = ub - lb

        # these are invariant numbers unless visiting_param changes
        self._factor2 = np.exp((4.0 - self._visiting_param) * np.log(
            self._visiting_param - 1.0))
        self._factor3 = np.exp((2.0 - self._visiting_param) * np.log(2.0)
                               / (self._visiting_param - 1.0))
        self._factor4_p = np.sqrt(np.pi) * self._factor2 / (self._factor3 * (
            3.0 - self._visiting_param))

        self._factor5 = 1.0 / (self._visiting_param - 1.0) - 0.5
        self._d1 = 2.0 - self._factor5
        self._factor6 = np.pi * (1.0 - self._factor5) / np.sin(
            np.pi * (1.0 - self._factor5)) / np.exp(gammaln(self._d1))

    def visiting(self, x, step, temperature):
        """ Based on the step in the strategy chain, new coordinates are
        generated by changing all components is the same time or only
        one of them, the new values are computed with visit_fn method
        """
        dim = x.size
        if step < dim:
            # Changing all coordinates with a new visiting value
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rng_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            x_visit = visits + x
            a = x_visit - self.lower
            b = np.fmod(a, self.bound_range) + self.bound_range
            x_visit = np.fmod(b, self.bound_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
        else:
            # Changing only one coordinate at a time based on strategy
            # chain step
            x_visit = np.copy(x)
            visit = self.visit_fn(temperature, 1)[0]
            if visit > self.TAIL_LIMIT:
                visit = self.TAIL_LIMIT * self.rng_gen.uniform()
            elif visit < -self.TAIL_LIMIT:
                visit = -self.TAIL_LIMIT * self.rng_gen.uniform()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit[index] - self.lower[index]
            b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
            x_visit[index] = np.fmod(b, self.bound_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                    index]) < self.MIN_VISIT_BOUND:
                x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit

    def visit_fn(self, temperature, dim):
        """ Formula Visita from p. 405 of reference [2] """
        x, y = self.rng_gen.normal(size=(dim, 2)).T

        factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
        factor4 = self._factor4_p * factor1

        # sigmax
        x *= np.exp(-(self._visiting_param - 1.0) * np.log(
            self._factor6 / factor4) / (3.0 - self._visiting_param))

        den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) /
                     (3.0 - self._visiting_param))

        return x / den


class EnergyState:
    """
    Class used to record the energy state. At any time, it knows what is the
    currently used coordinates and the most recent best location.

    Parameters
    ----------
    lower : array_like
        A 1-D NumPy ndarray containing lower bounds for generating an initial
        random components in the `reset` method.
    upper : array_like
        A 1-D NumPy ndarray containing upper bounds for generating an initial
        random components in the `reset` method
        components. Neither NaN or inf are allowed.
    callback : callable, ``callback(x, f, context)``, optional
        A callback function which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and `context` has value in [0, 1, 2]
    """
    # Maximum number of trials for generating a valid starting point
    MAX_REINIT_COUNT = 1000

    def __init__(self, lower, upper, callback=None):
        self.ebest = None
        self.current_energy = None
        self.current_location = None
        self.xbest = None
        self.lower = lower
        self.upper = upper
        self.callback = callback

    def reset(self, func_wrapper, rng_gen, x0=None):
        """
        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        """
        if x0 is None:
            self.current_location = rng_gen.uniform(self.lower, self.upper,
                                                    size=len(self.lower))
        else:
            self.current_location = np.copy(x0)
        init_error = True
        reinit_counter = 0
        while init_error:
            self.current_energy = func_wrapper.fun(self.current_location)
            if self.current_energy is None:
                raise ValueError('Objective function is returning None')
            if not np.isfinite(self.current_energy):
                if reinit_counter >= EnergyState.MAX_REINIT_COUNT:
                    init_error = False
                    message = (
                        'Stopping algorithm because function '
                        'create NaN or (+/-) infinity values even with '
                        'trying new random parameters'
                    )
                    raise ValueError(message)
                self.current_location = rng_gen.uniform(self.lower,
                                                        self.upper,
                                                        size=self.lower.size)
                reinit_counter += 1
            else:
                init_error = False
            # If first time reset, initialize ebest and xbest
            if self.ebest is None and self.xbest is None:
                self.ebest = self.current_energy
                self.xbest = np.copy(self.current_location)
            # Otherwise, we keep them in case of reannealing reset

    def update_best(self, e, x, context):
        self.ebest = e
        self.xbest = np.copy(x)
        if self.callback is not None:
            val = self.callback(x, e, context)
            if val is not None:
                if val:
                    return ('Callback function requested to stop early by '
                           'returning True')

    def update_current(self, e, x):
        self.current_energy = e
        self.current_location = np.copy(x)


class StrategyChain:
    """
    Class that implements within a Markov chain the strategy for location
    acceptance and local search decision making.

    Parameters
    ----------
    acceptance_param : float
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    visit_dist : VisitingDistribution
        Instance of `VisitingDistribution` class.
    func_wrapper : ObjectiveFunWrapper
        Instance of `ObjectiveFunWrapper` class.
    minimizer_wrapper: LocalSearchWrapper
        Instance of `LocalSearchWrapper` class.
    rand_gen : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    energy_state: EnergyState
        Instance of `EnergyState` class.

    """

    def __init__(self, acceptance_param, visit_dist, func_wrapper,
                 minimizer_wrapper, rand_gen, energy_state):
        # Local strategy chain minimum energy and location
        self.emin = energy_state.current_energy
        self.xmin = np.array(energy_state.current_location)
        # Global optimizer state
        self.energy_state = energy_state
        # Acceptance parameter
        self.acceptance_param = acceptance_param
        # Visiting distribution instance
        self.visit_dist = visit_dist
        # Wrapper to objective function
        self.func_wrapper = func_wrapper
        # Wrapper to the local minimizer
        self.minimizer_wrapper = minimizer_wrapper
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self._rand_gen = rand_gen
        self.temperature_step = 0
        self.K = 100 * len(energy_state.current_location)

    def accept_reject(self, j, e, x_visit):
        r = self._rand_gen.uniform()
        pqv_temp = 1.0 - ((1.0 - self.acceptance_param) *
            (e - self.energy_state.current_energy) / self.temperature_step)
        if pqv_temp <= 0.:
            pqv = 0.
        else:
            pqv = np.exp(np.log(pqv_temp) / (
                1. - self.acceptance_param))

        if r <= pqv:
            # We accept the new location and update state
            self.energy_state.update_current(e, x_visit)
            self.xmin = np.copy(self.energy_state.current_location)

        # No improvement for a long time
        if self.not_improved_idx >= self.not_improved_max_idx:
            if j == 0 or self.energy_state.current_energy < self.emin:
                self.emin = self.energy_state.current_energy
                self.xmin = np.copy(self.energy_state.current_location)

    def run(self, step, temperature):
        self.temperature_step = temperature / float(step + 1)
        self.not_improved_idx += 1
        for j in range(self.energy_state.current_location.size * 2):
            if j == 0:
                if step == 0:
                    self.energy_state_improved = True
                else:
                    self.energy_state_improved = False
            x_visit = self.visit_dist.visiting(
                self.energy_state.current_location, j, temperature)
            # Calling the objective function
            e = self.func_wrapper.fun(x_visit)
            if e < self.energy_state.current_energy:
                # We have got a better energy value
                self.energy_state.update_current(e, x_visit)
                if e < self.energy_state.ebest:
                    val = self.energy_state.update_best(e, x_visit, 0)
                    if val is not None:
                        if val:
                            return val
                    self.energy_state_improved = True
                    self.not_improved_idx = 0
            else:
                # We have not improved but do we accept the new location?
                self.accept_reject(j, e, x_visit)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during annealing')
        # End of StrategyChain loop

    def local_search(self):
        # Decision making for performing a local search
        # based on strategy chain results
        # If energy has been improved or no improvement since too long,
        # performing a local search with the best strategy chain location
        if self.energy_state_improved:
            # Global energy has improved, let's see if LS improves further
            e, x = self.minimizer_wrapper.local_search(self.energy_state.xbest,
                                                       self.energy_state.ebest)
            if e < self.energy_state.ebest:
                self.not_improved_idx = 0
                val = self.energy_state.update_best(e, x, 1)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during local search')
        # Check probability of a need to perform a LS even if no improvement
        do_ls = False
        if self.K < 90 * len(self.energy_state.current_location):
            pls = np.exp(self.K * (
                self.energy_state.ebest - self.energy_state.current_energy) /
                self.temperature_step)
            if pls >= self._rand_gen.uniform():
                do_ls = True
        # Global energy not improved, let's see what LS gives
        # on the best strategy chain location
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self.minimizer_wrapper.local_search(self.xmin, self.emin)
            self.xmin = np.copy(x)
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.energy_state.current_location.size
            if e < self.energy_state.ebest:
                val = self.energy_state.update_best(
                    self.emin, self.xmin, 2)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during dual annealing')


class ObjectiveFunWrapper:

    def __init__(self, func, maxfun=1e7, *args):
        self.func = func
        self.args = args
        # Number of objective function evaluations
        self.nfev = 0
        # Number of gradient function evaluation if used
        self.ngev = 0
        # Number of hessian of the objective function if used
        self.nhev = 0
        self.maxfun = maxfun

    def fun(self, x):
        self.nfev += 1
        return self.func(x, *self.args)


class LocalSearchWrapper:
    """
    Class used to wrap around the minimizer used for local search
    Default local minimizer is SciPy minimizer L-BFGS-B
    """

    LS_MAXITER_RATIO = 6
    LS_MAXITER_MIN = 100
    LS_MAXITER_MAX = 1000

    def __init__(self, search_bounds, func_wrapper, *args, **kwargs):
        self.func_wrapper = func_wrapper
        self.kwargs = kwargs
        self.jac = self.kwargs.get('jac', None)
        self.hess = self.kwargs.get('hess', None)
        self.hessp = self.kwargs.get('hessp', None)
        self.kwargs.pop("args", None)
        self.minimizer = minimize
        bounds_list = list(zip(*search_bounds))
        self.lower = np.array(bounds_list[0])
        self.upper = np.array(bounds_list[1])

        # If no minimizer specified, use SciPy minimize with 'L-BFGS-B' method
        if not self.kwargs:
            n = len(self.lower)
            ls_max_iter = min(max(n * self.LS_MAXITER_RATIO,
                                  self.LS_MAXITER_MIN),
                              self.LS_MAXITER_MAX)
            self.kwargs['method'] = 'L-BFGS-B'
            self.kwargs['options'] = {
                'maxiter': ls_max_iter,
            }
            self.kwargs['bounds'] = list(zip(self.lower, self.upper))
        else:
            if callable(self.jac):
                def wrapped_jac(x):
                    return self.jac(x, *args)
                self.kwargs['jac'] = wrapped_jac
            if callable(self.hess):
                def wrapped_hess(x):
                    return self.hess(x, *args)
                self.kwargs['hess'] = wrapped_hess
            if callable(self.hessp):
                def wrapped_hessp(x, p):
                    return self.hessp(x, p, *args)
                self.kwargs['hessp'] = wrapped_hessp

    def local_search(self, x, e):
        # Run local search from the given x location where energy value is e
        x_tmp = np.copy(x)
        mres = self.minimizer(self.func_wrapper.fun, x, **self.kwargs)
        if 'njev' in mres:
            self.func_wrapper.ngev += mres.njev
        if 'nhev' in mres:
            self.func_wrapper.nhev += mres.nhev
        # Check if is valid value
        is_finite = np.all(np.isfinite(mres.x)) and np.isfinite(mres.fun)
        in_bounds = np.all(mres.x >= self.lower) and np.all(
            mres.x <= self.upper)
        is_valid = is_finite and in_bounds

        # Use the new point only if it is valid and return a better results
        if is_valid and mres.fun < e:
            return mres.fun, mres.x
        else:
            return e, x_tmp


@_transition_to_rng("seed", position_num=10)
def dual_annealing(func, bounds, args=(), maxiter=1000,
                   minimizer_kwargs=None, initial_temp=5230.,
                   restart_temp_ratio=2.e-5, visit=2.62, accept=-5.0,
                   maxfun=1e7, rng=None, no_local_search=False,
                   callback=None, x0=None):
    """
    Find the global minimum of a function using Dual Annealing.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    maxiter : int, optional
        The maximum number of global search iterations. Default value is 1000.
    minimizer_kwargs : dict, optional
        Keyword arguments to be passed to the local minimizer
        (`minimize`). An important option could be ``method`` for the minimizer
        method to use.
        If no keyword arguments are provided, the local minimizer defaults to
        'L-BFGS-B' and uses the already supplied bounds. If `minimizer_kwargs`
        is specified, then the dict must contain all parameters required to
        control the local minimization. `args` is ignored in this dict, as it is
        passed automatically. `bounds` is not automatically passed on to the
        local minimizer as the method may not support them.
    initial_temp : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing dual_annealing to escape
        local minima that it is trapped in. Default value is 5230. Range is
        (0.01, 5.e4].
    restart_temp_ratio : float, optional
        During the annealing process, temperature is decreasing, when it
        reaches ``initial_temp * restart_temp_ratio``, the reannealing process
        is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62. Higher
        values give the visiting distribution a heavier tail, this makes
        the algorithm jump to a more distant region. The value range is (1, 3].
    accept : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a `Generator`.

        Specify `rng` for repeatable minimizations. The random numbers
        generated only affect the visiting distribution function
        and new coordinates generation.
    no_local_search : bool, optional
        If `no_local_search` is set to True, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
    callback : callable, optional
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has one of the following
        values:

        - ``0``: minimum detected in the annealing process.
        - ``1``: detection occurred in the local search process.
        - ``2``: detection done in the dual annealing process.

        If the callback implementation returns True, the algorithm will stop.
    x0 : ndarray, shape(n,), optional
        Coordinates of a single N-D starting point.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See `OptimizeResult` for a description of other attributes.

    Notes
    -----
    This function implements the Dual Annealing optimization. This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`

    .. math::

        g_{q_{v}}(\\Delta x(t)) \\propto \\frac{ \\
        \\left[T_{q_{v}}(t) \\right]^{-\\frac{D}{3-q_{v}}}}{ \\
        \\left[{1+(q_{v}-1)\\frac{(\\Delta x(t))^{2}} { \\
        \\left[T_{q_{v}}(t)\\right]^{\\frac{2}{3-q_{v}}}}}\\right]^{ \\
        \\frac{1}{q_{v}-1}+\\frac{D-1}{2}}}

    Where :math:`t` is the artificial time. This visiting distribution is used
    to generate a trial jump distance :math:`\\Delta x(t)` of variable
    :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.

    From the starting point, after calling the visiting distribution
    function, the acceptance probability is computed as follows:

    .. math::

        p_{q_{a}} = \\min{\\{1,\\left[1-(1-q_{a}) \\beta \\Delta E \\right]^{ \\
        \\frac{1}{1-q_{a}}}\\}}

    Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
    acceptance probability is assigned to the cases where

    .. math::

        [1-(1-q_{a}) \\beta \\Delta E] < 0

    The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to

    .. math::

        T_{q_{v}}(t) = T_{q_{v}}(1) \\frac{2^{q_{v}-1}-1}{\\left( \\
        1 + t\\right)^{q_{v}-1}-1}

    Where :math:`q_{v}` is the visiting parameter.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Tsallis C. Possible generalization of Boltzmann-Gibbs
        statistics. Journal of Statistical Physics, 52, 479-487 (1998).
    .. [2] Tsallis C, Stariolo DA. Generalized Simulated Annealing.
        Physica A, 233, 395-406 (1996).
    .. [3] Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model.
        Physics Letters A, 233, 216-220 (1997).
    .. [4] Xiang Y, Gong XG. Efficiency of Generalized Simulated
        Annealing. Physical Review E, 62, 4473 (2000).
    .. [5] Xiang Y, Gubian S, Suomela B, Hoeng J. Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R. The R Journal, Volume 5/1 (2013).
    .. [6] Mullen, K. Continuous Global Optimization in R. Journal of
        Statistical Software, 60(6), 1 - 45, (2014).
        :doi:`10.18637/jss.v060.i06`

    Examples
    --------
    The following example is a 10-D problem, with many local minima.
    The function involved is called Rastrigin
    (https://en.wikipedia.org/wiki/Rastrigin_function)

    >>> import numpy as np
    >>> from scipy.optimize import dual_annealing
    >>> func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    >>> lw = [-5.12] * 10
    >>> up = [5.12] * 10
    >>> ret = dual_annealing(func, bounds=list(zip(lw, up)))
    >>> ret.x
    array([-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.97165720e-09,
           -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09,
           -6.05775280e-09, -5.00668935e-09]) # random
    >>> ret.fun
    0.000000

    """

    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))

    if x0 is not None and not len(x0) == len(bounds):
        raise ValueError('Bounds size does not match x0')

    lu = list(zip(*bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    # Check that restart temperature ratio is correct
    if restart_temp_ratio <= 0. or restart_temp_ratio >= 1.:
        raise ValueError('Restart temperature ratio has to be in range (0, 1)')
    # Checking bounds are valid
    if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
            np.isnan(lower)) or np.any(np.isnan(upper))):
        raise ValueError('Some bounds values are inf values or nan values')
    # Checking that bounds are consistent
    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent min < max')
    # Checking that bounds are the same length
    if not len(lower) == len(upper):
        raise ValueError('Bounds do not have the same dimensions')

    # Wrapper for the objective function
    func_wrapper = ObjectiveFunWrapper(func, maxfun, *args)

    # minimizer_kwargs has to be a dict, not None
    minimizer_kwargs = minimizer_kwargs or {}

    minimizer_wrapper = LocalSearchWrapper(
        bounds, func_wrapper, *args, **minimizer_kwargs)

    # Initialization of random Generator for reproducible runs if rng provided
    rng_gen = check_random_state(rng)
    # Initialization of the energy state
    energy_state = EnergyState(lower, upper, callback)
    energy_state.reset(func_wrapper, rng_gen, x0)
    # Minimum value of annealing temperature reached to perform
    # re-annealing
    temperature_restart = initial_temp * restart_temp_ratio
    # VisitingDistribution instance
    visit_dist = VisitingDistribution(lower, upper, visit, rng_gen)
    # Strategy chain instance
    strategy_chain = StrategyChain(accept, visit_dist, func_wrapper,
                                   minimizer_wrapper, rng_gen, energy_state)
    need_to_stop = False
    iteration = 0
    message = []
    # OptimizeResult object to be returned
    optimize_res = OptimizeResult()
    optimize_res.success = True
    optimize_res.status = 0

    t1 = np.exp((visit - 1) * np.log(2.0)) - 1.0
    # Run the search loop
    while not need_to_stop:
        for i in range(maxiter):
            # Compute temperature for this step
            s = float(i) + 2.0
            t2 = np.exp((visit - 1) * np.log(s)) - 1.0
            temperature = initial_temp * t1 / t2
            if iteration >= maxiter:
                message.append("Maximum number of iteration reached")
                need_to_stop = True
                break
            # Need a re-annealing process?
            if temperature < temperature_restart:
                energy_state.reset(func_wrapper, rng_gen)
                break
            # starting strategy chain
            val = strategy_chain.run(i, temperature)
            if val is not None:
                message.append(val)
                need_to_stop = True
                optimize_res.success = False
                break
            # Possible local search at the end of the strategy chain
            if not no_local_search:
                val = strategy_chain.local_search()
                if val is not None:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
            iteration += 1

    # Setting the OptimizeResult values
    optimize_res.x = energy_state.xbest
    optimize_res.fun = energy_state.ebest
    optimize_res.nit = iteration
    optimize_res.nfev = func_wrapper.nfev
    optimize_res.njev = func_wrapper.ngev
    optimize_res.nhev = func_wrapper.nhev
    optimize_res.message = message
    return optimize_res
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'OptimizeWarning',
    'approx_fprime',
    'bracket',
    'brent',
    'brute',
    'check_grad',
    'fmin',
    'fmin_bfgs',
    'fmin_cg',
    'fmin_ncg',
    'fmin_powell',
    'fminbound',
    'golden',
    'line_search',
    'rosen',
    'rosen_der',
    'rosen_hess',
    'rosen_hess_prod',
    'show_options',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="optimize",
                                   private_modules=["_optimize"], all=__all__,
                                   attribute=name)
<|endoftext|>"""shgo: The simplicial homology global optimisation algorithm."""
from collections import namedtuple
import time
import logging
import warnings
import sys

import numpy as np

from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper

from scipy.optimize._shgo_lib._complex import Complex

__all__ = ['shgo']


def shgo(
    func, bounds, args=(), constraints=None, n=100, iters=1, callback=None,
    minimizer_kwargs=None, options=None, sampling_method='simplicial', *,
    workers=1
):
    """
    Finds the global minimum of a function using SHG optimization.

    SHGO stands for "simplicial homology global optimization".

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.
        See the tutorial [5]_ for further details on specifying constraints.

        .. note::

           Only COBYLA, COBYQA, SLSQP, and trust-constr local minimize methods
           currently support constraint arguments. If the ``constraints``
           sequence used in the local optimization problem is not defined in
           ``minimizer_kwargs`` and a constrained method is used then the
           global ``constraints`` will be used.
           (Defining a ``constraints`` sequence in ``minimizer_kwargs``
           means that ``constraints`` will not be added so if equality
           constraints and so forth need to be added then the inequality
           functions in ``constraints`` need to be added to
           ``minimizer_kwargs`` too).
           COBYLA only supports inequality constraints.

        .. versionchanged:: 1.11.0

           ``constraints`` accepts `NonlinearConstraint`, `LinearConstraint`.

    n : int, optional
        Number of sampling points used in the construction of the simplicial
        complex. For the default ``simplicial`` sampling method 2**dim + 1
        sampling points are generated instead of the default ``n=100``. For all
        other specified values `n` sampling points are generated. For
        ``sobol``, ``halton`` and other arbitrary `sampling_methods` ``n=100`` or
        another specified number of sampling points are generated.
    iters : int, optional
        Number of iterations used in the construction of the simplicial
        complex. Default is 1.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize``. Some important options could be:

        method : str
            The minimization method. If not given, chosen to be one of
            BFGS, L-BFGS-B, SLSQP, depending on whether or not the
            problem has constraints or bounds.
        args : tuple
            Extra arguments passed to the objective function (``func``) and
            its derivatives (Jacobian, Hessian).
        options : dict, optional
            Note that by default the tolerance is specified as
            ``{ftol: 1e-12}``

    options : dict, optional
        A dictionary of solver options. Many of the options specified for the
        global routine are also passed to the ``scipy.optimize.minimize``
        routine. The options that are also passed to the local routine are
        marked with "(L)".

        Stopping criteria, the algorithm will terminate if any of the specified
        criteria are met. However, the default algorithm does not require any
        to be specified:

        maxfev : int (L)
            Maximum number of function evaluations in the feasible domain.
            (Note only methods that support this option will terminate
            the routine at precisely exact specified value. Otherwise the
            criterion will only terminate during a global iteration)
        f_min : float
            Specify the minimum objective function value, if it is known.
        f_tol : float
            Precision goal for the value of f in the stopping
            criterion. Note that the global routine will also
            terminate if a sampling point in the global routine is
            within this tolerance.
        maxiter : int
            Maximum number of iterations to perform.
        maxev : int
            Maximum number of sampling evaluations to perform (includes
            searching in infeasible points).
        maxtime : float
            Maximum processing runtime allowed
        minhgrd : int
            Minimum homology group rank differential. The homology group of the
            objective function is calculated (approximately) during every
            iteration. The rank of this group has a one-to-one correspondence
            with the number of locally convex subdomains in the objective
            function (after adequate sampling points each of these subdomains
            contain a unique global minimum). If the difference in the hgr is 0
            between iterations for ``maxhgrd`` specified iterations the
            algorithm will terminate.

        Objective function knowledge:

        symmetry : list or bool
            Specify if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by up to
            O(n!) times in the fully symmetric case. If `True` is specified
            then all variables will be set symmetric to the first variable.
            Default
            is set to False.

            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
            x_6 are symmetric to x_4, this can be specified to the solver as::

                symmetry = [0,  # Variable 1
                            0,  # symmetric to variable 1
                            0,  # symmetric to variable 1
                            3,  # Variable 4
                            3,  # symmetric to variable 4
                            3,  # symmetric to variable 4
                            ]

        jac : bool or callable, optional
            Jacobian (gradient) of objective function. Only for CG, BFGS,
            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If ``jac`` is a
            boolean and is True, ``fun`` is assumed to return the gradient
            along with the objective function. If False, the gradient will be
            estimated numerically. ``jac`` can also be a callable returning the
            gradient of the objective. In this case, it must accept the same
            arguments as ``fun``. (Passed to `scipy.optimize.minimize`
            automatically)

        hess, hessp : callable, optional
            Hessian (matrix of second-order derivatives) of objective function
            or Hessian of objective function times an arbitrary vector p.
            Only for Newton-CG, dogleg, trust-ncg. Only one of ``hessp`` or
            ``hess`` needs to be given. If ``hess`` is provided, then
            ``hessp`` will be ignored. If neither ``hess`` nor ``hessp`` is
            provided, then the Hessian product will be approximated using
            finite differences on ``jac``. ``hessp`` must compute the Hessian
            times an arbitrary vector. (Passed to `scipy.optimize.minimize`
            automatically)

        Algorithm settings:

        minimize_every_iter : bool
            If True then promising global sampling points will be passed to a
            local minimization routine every iteration. If True then only the
            final minimizer pool will be run. Defaults to True.

        local_iter : int
            Only evaluate a few of the best minimizer pool candidates every
            iteration. If False all potential points are passed to the local
            minimization routine.

        infty_constraints : bool
            If True then any sampling points generated which are outside will
            the feasible domain will be saved and given an objective function
            value of ``inf``. If False then these points will be discarded.
            Using this functionality could lead to higher performance with
            respect to function evaluations before the global minimum is found,
            specifying False will use less memory at the cost of a slight
            decrease in performance. Defaults to True.

        Feedback:

        disp : bool (L)
            Set to True to print convergence messages.

    sampling_method : str or function, optional
        Current built in sampling method options are ``halton``, ``sobol`` and
        ``simplicial``. The default ``simplicial`` provides
        the theoretical guarantee of convergence to the global minimum in
        finite time. ``halton`` and ``sobol`` method are faster in terms of
        sampling point generation at the cost of the loss of
        guaranteed convergence. It is more appropriate for most "easier"
        problems where the convergence is relatively fast.
        User defined sampling functions must accept two arguments of ``n``
        sampling points of dimension ``dim`` per call and output an array of
        sampling points with shape `n x dim`.

    workers : int or map-like callable, optional
        Sample and run the local serial minimizations in parallel.
        Supply -1 to use all available CPU cores, or an int to use
        that many Processes (uses `multiprocessing.Pool <multiprocessing>`).

        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for parallel evaluation.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.11.0

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are:
        ``x`` the solution array corresponding to the global minimum,
        ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions,
        ``funl`` the function output at the corresponding local solutions,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully,
        ``message`` which describes the cause of the termination,
        ``nfev`` the total number of objective function evaluations including
        the sampling calls,
        ``nlfev`` the total number of objective function evaluations
        culminating from all local search optimizations,
        ``nit`` number of iterations performed by the global routine.

    Notes
    -----
    Global optimization using simplicial homology global optimization [1]_.
    Appropriate for solving general purpose NLP and blackbox optimization
    problems to global optimality (low-dimensional problems).

    In general, the optimization problems are of the form::

        minimize f(x) subject to

        g_i(x) >= 0,  i = 1,...,m
        h_j(x)  = 0,  j = 1,...,p

    where x is a vector of one or more variables. ``f(x)`` is the objective
    function ``R^n -> R``, ``g_i(x)`` are the inequality constraints, and
    ``h_j(x)`` are the equality constraints.

    Optionally, the lower and upper bounds for each element in x can also be
    specified using the `bounds` argument.

    While most of the theoretical advantages of SHGO are only proven for when
    ``f(x)`` is a Lipschitz smooth function, the algorithm is also proven to
    converge to the global optimum for the more general case where ``f(x)`` is
    non-continuous, non-convex and non-smooth, if the default sampling method
    is used [1]_.

    The local search method may be specified using the ``minimizer_kwargs``
    parameter which is passed on to ``scipy.optimize.minimize``. By default,
    the ``SLSQP`` method is used. In general, it is recommended to use the
    ``SLSQP``, ``COBYLA``, or ``COBYQA`` local minimization if inequality
    constraints are defined for the problem since the other methods do not use
    constraints.

    The ``halton`` and ``sobol`` method points are generated using
    `scipy.stats.qmc`. Any other QMC method could be used.

    References
    ----------
    .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology
           algorithm for lipschitz optimisation", Journal of Global
           Optimization.
    .. [2] Joe, SW and Kuo, FY (2008) "Constructing Sobol' sequences with
           better  two-dimensional projections", SIAM J. Sci. Comput. 30,
           2635-2654.
    .. [3] Hock, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes", Lecture Notes in Economics and Mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    .. [4] Wales, DJ (2015) "Perspective: Insight into reaction coordinates and
           dynamics from the potential energy landscape",
           Journal of Chemical Physics, 142(13), 2015.
    .. [5] https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

    Examples
    --------
    First consider the problem of minimizing the Rosenbrock function, `rosen`:

    >>> from scipy.optimize import rosen, shgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = shgo(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 2.920392374190081e-18)

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like ``np.inf`` which will be
    converted to large float numbers.

    >>> bounds = [(None, None), ]*4
    >>> result = shgo(rosen, bounds)
    >>> result.x
    array([0.99999851, 0.99999704, 0.99999411, 0.9999882 ])

    Next, we consider the Eggholder function, a problem with several local
    minima and one global minimum. We will demonstrate the use of arguments and
    the capabilities of `shgo`.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> import numpy as np
    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]

    `shgo` has built-in low discrepancy sampling sequences. First, we will
    input 64 initial sampling points of the *Sobol'* sequence:

    >>> result = shgo(eggholder, bounds, n=64, sampling_method='sobol')
    >>> result.x, result.fun
    (array([512.        , 404.23180824]), -959.6406627208397)

    `shgo` also has a return for any other local minima that was found, these
    can be called using:

    >>> result.xl
    array([[ 512.        ,  404.23180824],
           [ 283.0759062 , -487.12565635],
           [-294.66820039, -462.01964031],
           [-105.87688911,  423.15323845],
           [-242.97926   ,  274.38030925],
           [-506.25823477,    6.3131022 ],
           [-408.71980731, -156.10116949],
           [ 150.23207937,  301.31376595],
           [  91.00920901, -391.283763  ],
           [ 202.89662724, -269.38043241],
           [ 361.66623976, -106.96493868],
           [-219.40612786, -244.06020508]])

    >>> result.funl
    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097,
           -559.78685655, -557.36868733, -507.87385942, -493.9605115 ,
           -426.48799655, -421.15571437, -419.31194957, -410.98477763])

    These results are useful in applications where there are many global minima
    and the values of other global minima are desired or where the local minima
    can provide insight into the system (for example morphologies
    in physical chemistry [4]_).

    If we want to find a larger number of local minima, we can increase the
    number of sampling points or the number of iterations. We'll increase the
    number of sampling points to 64 and the number of iterations from the
    default of 1 to 3. Using ``simplicial`` this would have given us
    64 x 3 = 192 initial sampling points.

    >>> result_2 = shgo(eggholder,
    ...                 bounds, n=64, iters=3, sampling_method='sobol')
    >>> len(result.xl), len(result_2.xl)
    (12, 23)

    Note the difference between, e.g., ``n=192, iters=1`` and ``n=64,
    iters=3``.
    In the first case the promising points contained in the minimiser pool
    are processed only once. In the latter case it is processed every 64
    sampling points for a total of 3 times.

    To demonstrate solving problems with non-linear constraints consider the
    following example from Hock and Schittkowski problem 73 (cattle-feed)
    [3]_::

        minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4

        subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5    >= 0,

                    12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21
                        -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +
                                      20.5 * x_3**2 + 0.62 * x_4**2)      >= 0,

                    x_1 + x_2 + x_3 + x_4 - 1                             == 0,

                    1 >= x_i >= 0 for all i

    The approximate answer given in [3]_ is::

        f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) = 29.894378

    >>> def f(x):  # (cattle-feed)
    ...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]
    ...
    >>> def g1(x):
    ...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0
    ...
    >>> def g2(x):
    ...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
    ...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2
    ...                             + 20.5*x[2]**2 + 0.62*x[3]**2)
    ...             ) # >=0
    ...
    >>> def h1(x):
    ...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0
    ...
    >>> cons = ({'type': 'ineq', 'fun': g1},
    ...         {'type': 'ineq', 'fun': g2},
    ...         {'type': 'eq', 'fun': h1})
    >>> bounds = [(0, 1.0),]*4
    >>> res = shgo(f, bounds, n=150, constraints=cons)
    >>> res
     message: Optimization terminated successfully.
     success: True
         fun: 29.894378159142136
        funl: [ 2.989e+01]
           x: [ 6.355e-01  1.137e-13  3.127e-01  5.178e-02] # may vary
          xl: [[ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]] # may vary
         nit: 1
        nfev: 142 # may vary
       nlfev: 35 # may vary
       nljev: 5
       nlhev: 0

    >>> g1(res.x), g2(res.x), h1(res.x)
    (-5.062616992290714e-14, -2.9594104944408173e-12, 0.0)

    """
    # if necessary, convert bounds class to old bounds
    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))

    # Initiate SHGO class
    # use in context manager to make sure that any parallelization
    # resources are freed.
    with SHGO(func, bounds, args=args, constraints=constraints, n=n,
               iters=iters, callback=callback,
               minimizer_kwargs=minimizer_kwargs,
               options=options, sampling_method=sampling_method,
               workers=workers) as shc:
        # Run the algorithm, process results and test success
        shc.iterate_all()

    if not shc.break_routine:
        if shc.disp:
            logging.info("Successfully completed construction of complex.")

    # Test post iterations success
    if len(shc.LMC.xl_maps) == 0:
        # If sampling failed to find pool, return lowest sampled point
        # with a warning
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.fail_routine(mes="Failed to find a feasible minimizer point. "
                             f"Lowest sampling point = {shc.f_lowest}")
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn
        shc.res.tnev = shc.n_sampled
    else:
        # Test that the optimal solutions do not violate any constraints
        pass  # TODO

    # Confirm the routine ran successfully
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True

    # Return the final results
    return shc.res


class SHGO:
    def __init__(self, func, bounds, args=(), constraints=None, n=None,
                 iters=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='simplicial', workers=1):
        from scipy.stats import qmc
        # Input checks
        methods = ['halton', 'sobol', 'simplicial']
        if isinstance(sampling_method, str) and sampling_method not in methods:
            raise ValueError(("Unknown sampling_method specified."
                              " Valid methods: {}").format(', '.join(methods)))

        # Split obj func if given with Jac
        try:
            if ((minimizer_kwargs['jac'] is True) and
                    (not callable(minimizer_kwargs['jac']))):
                self.func = MemoizeJac(func)
                jac = self.func.derivative
                minimizer_kwargs['jac'] = jac
                func = self.func  # .fun
            else:
                self.func = func  # Normal definition of objective function
        except (TypeError, KeyError):
            self.func = func  # Normal definition of objective function

        # Initiate class
        self.func = _FunctionWrapper(func, args)
        self.bounds = bounds
        self.args = args
        self.callback = callback

        # Bounds
        abound = np.array(bounds, float)
        self.dim = np.shape(abound)[0]  # Dimensionality of problem

        # Set none finite values to large floats
        infind = ~np.isfinite(abound)
        abound[infind[:, 0], 0] = -1e50
        abound[infind[:, 1], 1] = 1e50

        # Check if bounds are correctly specified
        bnderr = abound[:, 0] > abound[:, 1]
        if bnderr.any():
            raise ValueError("Error: lb > ub in bounds "
                             f"{', '.join(str(b) for b in bnderr)}.")

        self.bounds = abound

        # Constraints
        # Process constraint dict sequence:
        self.constraints = constraints
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []

            # shgo internals deals with old-style constraints
            # self.constraints is used to create Complex, so need
            # to be stored internally in old-style.
            # `minimize` takes care of normalising these constraints
            # for slsqp/cobyla/cobyqa/trust-constr.
            self.constraints = standardize_constraints(
                constraints,
                np.empty(self.dim, float),
                'old'
            )
            for cons in self.constraints:
                if cons['type'] in ('ineq'):
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None

        # Define local minimization keyword arguments
        # Start with defaults
        self.minimizer_kwargs = {'method': 'SLSQP',
                                 'bounds': self.bounds,
                                 'options': {},
                                 'callback': self.callback
                                 }
        if minimizer_kwargs is not None:
            # Overwrite with supplied values
            self.minimizer_kwargs.update(minimizer_kwargs)

        else:
            self.minimizer_kwargs['options'] = {'ftol': 1e-12}

        if (
            self.minimizer_kwargs['method'].lower() in ('slsqp', 'cobyla',
                                                        'cobyqa',
                                                        'trust-constr')
            and (
                minimizer_kwargs is not None and
                'constraints' not in minimizer_kwargs and
                constraints is not None
            ) or
            (self.g_cons is not None)
        ):
            self.minimizer_kwargs['constraints'] = self.min_cons

        # Process options dict
        if options is not None:
            self.init_options(options)
        else:  # Default settings:
            self.f_min_true = None
            self.minimize_every_iter = True

            # Algorithm limits
            self.maxiter = None
            self.maxfev = None
            self.maxev = None
            self.maxtime = None
            self.f_min_true = None
            self.minhgrd = None

            # Objective function knowledge
            self.symmetry = None

            # Algorithm functionality
            self.infty_cons_sampl = True
            self.local_iter = False

            # Feedback
            self.disp = False

        # Remove unknown arguments in self.minimizer_kwargs
        # Start with arguments all the solvers have in common
        self.min_solver_args = ['fun', 'x0', 'args',
                                'callback', 'options', 'method']
        # then add the ones unique to specific solvers
        solver_args = {
            '_custom': ['jac', 'hess', 'hessp', 'bounds', 'constraints'],
            'nelder-mead': [],
            'powell': [],
            'cg': ['jac'],
            'bfgs': ['jac'],
            'newton-cg': ['jac', 'hess', 'hessp'],
            'l-bfgs-b': ['jac', 'bounds'],
            'tnc': ['jac', 'bounds'],
            'cobyla': ['constraints', 'catol'],
            'cobyqa': ['bounds', 'constraints', 'feasibility_tol'],
            'slsqp': ['jac', 'bounds', 'constraints'],
            'dogleg': ['jac', 'hess'],
            'trust-ncg': ['jac', 'hess', 'hessp'],
            'trust-krylov': ['jac', 'hess', 'hessp'],
            'trust-exact': ['jac', 'hess'],
            'trust-constr': ['jac', 'hess', 'hessp', 'constraints'],
        }
        method = self.minimizer_kwargs['method']
        self.min_solver_args += solver_args[method.lower()]

        # Only retain the known arguments
        def _restrict_to_keys(dictionary, goodkeys):
            """Remove keys from dictionary if not in goodkeys - inplace"""
            existingkeys = set(dictionary)
            for key in existingkeys - set(goodkeys):
                dictionary.pop(key, None)

        _restrict_to_keys(self.minimizer_kwargs, self.min_solver_args)
        _restrict_to_keys(self.minimizer_kwargs['options'],
                          self.min_solver_args + ['ftol'])

        # Algorithm controls
        # Global controls
        self.stop_global = False  # Used in the stopping_criteria method
        self.break_routine = False  # Break the algorithm globally
        self.iters = iters  # Iterations to be ran
        self.iters_done = 0  # Iterations completed
        self.n = n  # Sampling points per iteration
        self.nc = 0  # n  # Sampling points to sample in current iteration
        self.n_prc = 0  # Processed points (used to track Delaunay iters)
        self.n_sampled = 0  # To track no. of sampling points already generated
        self.fn = 0  # Number of feasible sampling points evaluations performed
        self.hgr = 0  # Homology group rank
        # Initially attempt to build the triangulation incrementally:
        self.qhull_incremental = True

        # Default settings if no sampling criteria.
        if (self.n is None) and (self.iters is None) \
                and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1
            self.nc = 0  # self.n
        if self.iters is None:
            self.iters = 1
        if (self.n is None) and not (sampling_method == 'simplicial'):
            self.n = self.n = 100
            self.nc = 0  # self.n
        if (self.n == 100) and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1

        if not ((self.maxiter is None) and (self.maxfev is None) and (
                self.maxev is None)
                and (self.minhgrd is None) and (self.f_min_true is None)):
            self.iters = None

        # Set complex construction mode based on a provided stopping criteria:
        # Initialise sampling Complex and function cache
        # Note that sfield_args=() since args are already wrapped in self.func
        # using the_FunctionWrapper class.
        self.HC = Complex(dim=self.dim, domain=self.bounds,
                          sfield=self.func, sfield_args=(),
                          symmetry=self.symmetry,
                          constraints=self.constraints,
                          workers=workers)

        # Choose complex constructor
        if sampling_method == 'simplicial':
            self.iterate_complex = self.iterate_hypercube
            self.sampling_method = sampling_method

        elif sampling_method in ['halton', 'sobol'] or \
                not isinstance(sampling_method, str):
            self.iterate_complex = self.iterate_delaunay
            # Sampling method used
            if sampling_method in ['halton', 'sobol']:
                if sampling_method == 'sobol':
                    self.n = int(2 ** np.ceil(np.log2(self.n)))
                    # self.n #TODO: Should always be self.n, this is
                    # unacceptable for shgo, check that nfev behaves as
                    # expected.
                    self.nc = 0
                    self.sampling_method = 'sobol'
                    self.qmc_engine = qmc.Sobol(d=self.dim, scramble=False,
                                                seed=0)
                else:
                    self.sampling_method = 'halton'
                    self.qmc_engine = qmc.Halton(d=self.dim, scramble=True,
                                                 seed=0)

                def sampling_method(n, d):
                    return self.qmc_engine.random(n)

            else:
                # A user defined sampling method:
                self.sampling_method = 'custom'

            self.sampling = self.sampling_custom
            self.sampling_function = sampling_method  # F(n, d)

        # Local controls
        self.stop_l_iter = False  # Local minimisation iterations
        self.stop_complex_iter = False  # Sampling iterations

        # Initiate storage objects used in algorithm classes
        self.minimizer_pool = []

        # Cache of local minimizers mapped
        self.LMC = LMapCache()

        # Initialize return object
        self.res = OptimizeResult()  # scipy.optimize.OptimizeResult object
        self.res.nfev = 0  # Includes each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local Jacobian evals for all minimisers
        self.res.nlhev = 0  # Local Hessian evals for all minimisers

    # Initiation aids
    def init_options(self, options):
        """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None

        """
        # Update 'options' dict passed to optimize.minimize
        # Do this first so we don't mutate `options` below.
        self.minimizer_kwargs['options'].update(options)

        # Ensure that 'jac', 'hess', and 'hessp' are passed directly to
        # `minimize` as keywords, not as part of its 'options' dictionary.
        for opt in ['jac', 'hess', 'hessp']:
            if opt in self.minimizer_kwargs['options']:
                self.minimizer_kwargs[opt] = (
                    self.minimizer_kwargs['options'].pop(opt))

        # Default settings:
        self.minimize_every_iter = options.get('minimize_every_iter', True)

        # Algorithm limits
        # Maximum number of iterations to perform.
        self.maxiter = options.get('maxiter', None)
        # Maximum number of function evaluations in the feasible domain
        self.maxfev = options.get('maxfev', None)
        # Maximum number of sampling evaluations (includes searching in
        # infeasible points
        self.maxev = options.get('maxev', None)
        # Maximum processing runtime allowed
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)
        if 'f_min' in options:
            # Specify the minimum objective function value, if it is known.
            self.f_min_true = options['f_min']
            self.f_tol = options.get('f_tol', 1e-4)
        else:
            self.f_min_true = None

        self.minhgrd = options.get('minhgrd', None)

        # Objective function knowledge
        self.symmetry = options.get('symmetry', False)
        if self.symmetry:
            self.symmetry = [0, ]*len(self.bounds)
        else:
            self.symmetry = None
        # Algorithm functionality
        # Only evaluate a few of the best candidates
        self.local_iter = options.get('local_iter', False)
        self.infty_cons_sampl = options.get('infty_constraints', True)

        # Feedback
        self.disp = options.get('disp', False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.HC.V._mapwrapper.__exit__(*args)

    # Iteration properties
    # Main construction loop:
    def iterate_all(self):
        """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
        if self.disp:
            logging.info('Splitting first generation')

        while not self.stop_global:
            if self.break_routine:
                break
            # Iterate complex, process minimisers
            self.iterate()
            self.stopping_criteria()

        # Build minimiser pool
        # Final iteration only needed if pools weren't minimised every
        # iteration
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()

        self.res.nit = self.iters_done  # + 1
        self.fn = self.HC.V.nfev

    def find_minima(self):
        """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
        if self.disp:
            logging.info('Searching for minimizer pool...')

        self.minimizers()

        if len(self.X_min) != 0:
            # Minimize the pool of minimizers with local minimization methods
            # Note that if Options['local_iter'] is an `int` instead of default
            # value False then only that number of candidates will be minimized
            self.minimise_pool(self.local_iter)
            # Sort results and build the global return object
            self.sort_result()

            # Lowest values used to report in case of failures
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            self.find_lowest_vertex()

        if self.disp:
            logging.info(f"Minimiser pool = SHGO.X_min = {self.X_min}")

    def find_lowest_vertex(self):
        # Find the lowest objective function value on one of
        # the vertices of the simplicial complex
        self.f_lowest = np.inf
        for x in self.HC.V.cache:
            if self.HC.V[x].f < self.f_lowest:
                if self.disp:
                    logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
                self.f_lowest = self.HC.V[x].f
                self.x_lowest = self.HC.V[x].x_a
        for lmc in self.LMC.cache:
            if self.LMC[lmc].f_min < self.f_lowest:
                self.f_lowest = self.LMC[lmc].f_min
                self.x_lowest = self.LMC[lmc].x_l

        if self.f_lowest == np.inf:  # no feasible point
            self.f_lowest = None
            self.x_lowest = None

    # Stopping criteria functions:
    def finite_iterations(self):
        mi = min(x for x in [self.iters, self.maxiter] if x is not None)
        if self.disp:
            logging.info(f'Iterations done = {self.iters_done} / {mi}')
        if self.iters is not None:
            if self.iters_done >= (self.iters):
                self.stop_global = True

        if self.maxiter is not None:  # Stop for infeasible sampling
            if self.iters_done >= (self.maxiter):
                self.stop_global = True
        return self.stop_global

    def finite_fev(self):
        # Finite function evals in the feasible domain
        if self.disp:
            logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
        if self.fn >= self.maxfev:
            self.stop_global = True
        return self.stop_global

    def finite_ev(self):
        # Finite evaluations including infeasible sampling points
        if self.disp:
            logging.info(f'Sampling evaluations done = {self.n_sampled} '
                         f'/ {self.maxev}')
        if self.n_sampled >= self.maxev:
            self.stop_global = True

    def finite_time(self):
        if self.disp:
            logging.info(f'Time elapsed = {time.time() - self.init} '
                         f'/ {self.maxtime}')
        if (time.time() - self.init) >= self.maxtime:
            self.stop_global = True

    def finite_precision(self):
        """
        Stop the algorithm if the final function value is known

        Specify in options (with ``self.f_min_true = options['f_min']``)
        and the tolerance with ``f_tol = options['f_tol']``
        """
        # If no minimizer has been found use the lowest sampling value
        self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Lowest function evaluation = {self.f_lowest}')
            logging.info(f'Specified minimum = {self.f_min_true}')
        # If no feasible point was return from test
        if self.f_lowest is None:
            return self.stop_global

        # Function to stop algorithm at specified percentage error:
        if self.f_min_true == 0.0:
            if self.f_lowest <= self.f_tol:
                self.stop_global = True
        else:
            pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
            if self.f_lowest <= self.f_min_true:
                self.stop_global = True
                # 2if (pe - self.f_tol) <= abs(1.0 / abs(self.f_min_true)):
                if abs(pe) >= 2 * self.f_tol:
                    warnings.warn(
                        f"A much lower value than expected f* = {self.f_min_true} "
                        f"was found f_lowest = {self.f_lowest}",
                        stacklevel=3
                    )
            if pe <= self.f_tol:
                self.stop_global = True

        return self.stop_global

    def finite_homology_growth(self):
        """
        Stop the algorithm if homology group rank did not grow in iteration.
        """
        if self.LMC.size == 0:
            return  # pass on no reason to stop yet.
        self.hgrd = self.LMC.size - self.hgr

        self.hgr = self.LMC.size
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        if self.disp:
            logging.info(f'Current homology growth = {self.hgrd} '
                         f' (minimum growth = {self.minhgrd})')
        return self.stop_global

    def stopping_criteria(self):
        """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
        if self.maxiter is not None:
            self.finite_iterations()
        if self.iters is not None:
            self.finite_iterations()
        if self.maxfev is not None:
            self.finite_fev()
        if self.maxev is not None:
            self.finite_ev()
        if self.maxtime is not None:
            self.finite_time()
        if self.f_min_true is not None:
            self.finite_precision()
        if self.minhgrd is not None:
            self.finite_homology_growth()
        return self.stop_global

    def iterate(self):
        self.iterate_complex()

        # Build minimizer pool
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()  # Process minimizer pool

        # Algorithm updates
        self.iters_done += 1

    def iterate_hypercube(self):
        """
        Iterate a subdivision of the complex

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        # Iterate the complex
        if self.disp:
            logging.info('Constructing and refining simplicial complex graph '
                         'structure')
        if self.n is None:
            self.HC.refine_all()
            self.n_sampled = self.HC.V.size()  # nevs counted
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n

        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints '
                         'and objective function values.')

        # Re-add minimisers to complex
        if len(self.LMC.xl_maps) > 0:
            for xl in self.LMC.cache:
                v = self.HC.V[xl]
                v_near = v.star()
                for v in v.nn:
                    v_near = v_near.union(v.nn)
                # Reconnect vertices to complex
                # if self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l),
                #                                   near=v_near):
                #    continue
                # else:
                    # If failure to find in v_near, then search all vertices
                    # (very expensive operation:
                #    self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l)
                #                                    )

        # Evaluate all constraints and functions
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')

        # feasible sampling points counted by the triangulation.py routines
        self.fn = self.HC.V.nfev
        return

    def iterate_delaunay(self):
        """
        Build a complex of Delaunay triangulated points

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        self.nc += self.n
        self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)

        # Add sampled points to a triangulation, construct self.Tri
        if self.disp:
            logging.info(f'self.n = {self.n}')
            logging.info(f'self.nc = {self.nc}')
            logging.info('Constructing and refining simplicial complex graph '
                         'structure from sampling points.')

        if self.dim < 2:
            self.Ind_sorted = np.argsort(self.C, axis=0)
            self.Ind_sorted = self.Ind_sorted.flatten()
            tris = []
            for ind, ind_s in enumerate(self.Ind_sorted):
                if ind > 0:
                    tris.append(self.Ind_sorted[ind - 1:ind + 1])

            tris = np.array(tris)
            # Store 1D triangulation:
            self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
            self.points = {}
        else:
            if self.C.shape[0] > self.dim + 1:  # Ensure a simplex can be built
                self.delaunay_triangulation(n_prc=self.n_prc)
            self.n_prc = self.C.shape[0]

        if self.disp:
            logging.info('Triangulation completed, evaluating all '
                         'constraints and objective function values.')

        if hasattr(self, 'Tri'):
            self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)

        # Process all pools
        # Evaluate all constraints and functions
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints '
                         'and objective function values.')

        # Evaluate all constraints and functions
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')

        # feasible sampling points counted by the triangulation.py routines
        self.fn = self.HC.V.nfev
        self.n_sampled = self.nc  # nevs counted in triangulation
        return

    # Hypercube minimizers
    def minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        # Note: Can implement parallelization here
        for x in self.HC.V.cache:
            in_LMC = False
            if len(self.LMC.xl_maps) > 0:
                for xlmi in self.LMC.xl_maps:
                    if np.all(np.array(x) == np.array(xlmi)):
                        in_LMC = True
            if in_LMC:
                continue

            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                    logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                    logging.info('=' * 30)

                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])

                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info(f'x = {vn.x} || f = {vn.f}')

                    logging.info('=' * 60)
        self.minimizer_pool_F = []
        self.X_min = []
        # normalized tuple in the Vertex cache
        self.X_min_cache = {}  # Cache used in hypercube sampling

        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            self.X_min_cache[tuple(v.x_a)] = v.x

        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        self.X_min = np.array(self.X_min)

        # TODO: Only do this if global mode
        self.sort_min_pool()

        return self.X_min

    # Local minimisation
    # Minimiser pool processing
    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be specified
                     globally or locally)

        """
        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])

        # Trim minimized point from current minimizer set
        self.trim_min_pool(0)

        while not self.stop_l_iter:
            # Global stopping criteria:
            self.stopping_criteria()

            # Note first iteration is outside loop:
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break

            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break

            # Construct topograph from current minimizer set
            # (NOTE: This is a very small topograph using only the minizer pool
            #        , it might be worth using some graph theory tools instead.
            self.g_topograph(lres_f_min.x, self.X_min)

            # Find local minimum at the miniser with the greatest Euclidean
            # distance from the current solution
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])

            # Trim minimised point from current minimizer set
            self.trim_min_pool(ind_xmin_l)

        # Reset controls
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[
            self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
        x_min = np.array([x_min])
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        # Find sorted indexes of spatial distances:
        self.Z = np.argsort(self.Y, axis=-1)

        self.Ss = X_min[self.Z][0]
        self.minimizer_pool = self.minimizer_pool[self.Z]
        self.minimizer_pool = self.minimizer_pool[0]
        return self.Ss

    # Local bound functions
    def construct_lcb_simplicial(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.

        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        # Loop over all bounds
        for vn in v_min.nn:
            for i, x_i in enumerate(vn.x_a):
                # Lower bound
                if (x_i < v_min.x_a[i]) and (x_i > cbounds[i][0]):
                    cbounds[i][0] = x_i

                # Upper bound
                if (x_i > v_min.x_a[i]) and (x_i < cbounds[i][1]):
                    cbounds[i][1] = x_i

        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a}')
            logging.info(f'cbounds = {cbounds}')

        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.
        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]

        return cbounds

    # Minimize a starting point locally
    def minimize(self, x_min, ind=None):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimize.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        # Use minima maps if vertex was already run
        if self.disp:
            logging.info(f'Vertex minimiser maps = {self.LMC.v_maps}')

        if self.LMC[x_min].lres is not None:
            logging.info(f'Found self.LMC[x_min].lres = '
                         f'{self.LMC[x_min].lres}')
            return self.LMC[x_min].lres

        if self.callback is not None:
            logging.info(f'Callback for minimizer starting at {x_min}:')

        if self.disp:
            logging.info(f'Starting minimization at {x_min}...')

        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            # Find the normalized tuple in the Vertex cache:
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]
            x_min_t_norm = tuple(x_min_t_norm)
            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        if self.disp and 'bounds' in self.minimizer_kwargs:
            logging.info('bounds in kwarg:')
            logging.info(self.minimizer_kwargs['bounds'])

        # Local minimization using scipy.optimize.minimize:
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)

        if self.disp:
            logging.info(f'lres = {lres}')

        # Local function evals for all minimizers
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev

        try:  # Needed because of the brain dead 1x1 NumPy arrays
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun

        # Append minima maps
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)

        return lres

    # Post local minimization processing
    def sort_result(self):
        """
        Sort results and build the global return object
        """
        # Sort results in local minima cache
        results = self.LMC.sort_cache_result()
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']

        # Add local func evals to sampling func evals
        # Count the number of feasible vertices and add to local func evals:
        self.res.nfev = self.fn + self.res.nlfev
        return self.res

    # Algorithm controls
    def fail_routine(self, mes=("Failed to converge")):
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective
        function evaluations at the computational costs of more Delaunay
        triangulation points.

        If ``infty_cons_sampl`` is False, then the infeasible points are
        discarded and only a subspace of the sampled points are used. This
        comes at the cost of the loss of guaranteed convergence and usually
        requires more objective function evaluations.
        """
        # Generate sampling points
        if self.disp:
            logging.info('Generating sampling points')
        self.sampling(self.nc, self.dim)
        if len(self.LMC.xl_maps) > 0:
            self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
        if not infty_cons_sampl:
            # Find subspace of feasible points
            if self.g_cons is not None:
                self.sampling_subspace()

        # Sort remaining samples
        self.sorted_samples()

        # Find objective function references
        self.n_sampled = self.nc

    def sampling_custom(self, n, dim):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        # Generate uniform sample points in [0, 1]^m \subset R^m
        if self.n_sampled == 0:
            self.C = self.sampling_function(n, dim)
        else:
            self.C = self.sampling_function(n, dim)
        # Distribute over bounds
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])
        return self.C

    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # Subspace of feasible points.
        for ind, g in enumerate(self.g_cons):
            # C.shape = (Z, dim) where Z is the number of sampling points to
            # evaluate and dim is the dimensionality of the problem.
            # the constraint function may not be vectorised so have to step
            # through each sampling point sequentially.
            feasible = np.array(
                [np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C],
                dtype=bool
            )
            self.C = self.C[feasible]

            if self.C.size == 0:
                self.res.message = ('No sampling point found within the '
                                    + 'feasible set. Increasing sampling '
                                    + 'size.')
                # sampling correctly for both 1-D and >1-D cases
                if self.disp:
                    logging.info(self.res.message)

    def sorted_samples(self):  # Validated
        """Find indexes of the sorted sampling points"""
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return self.Ind_sorted, self.Xs

    def delaunay_triangulation(self, n_prc=0):
        if hasattr(self, 'Tri') and self.qhull_incremental:
            # TODO: Uncertain if n_prc needs to add len(self.LMC.xl_maps)
            # in self.sampled_surface
            self.Tri.add_points(self.C[n_prc:, :])
        else:
            try:
                self.Tri = spatial.Delaunay(self.C,
                                            incremental=self.qhull_incremental,
                                            )
            except spatial.QhullError:
                if str(sys.exc_info()[1])[:6] == 'QH6239':
                    logging.warning('QH6239 Qhull precision error detected, '
                                    'this usually occurs when no bounds are '
                                    'specified, Qhull can only run with '
                                    'handling cocircular/cospherical points'
                                    ' and in this case incremental mode is '
                                    'switched off. The performance of shgo '
                                    'will be reduced in this mode.')
                    self.qhull_incremental = False
                    self.Tri = spatial.Delaunay(self.C,
                                                incremental=
                                                self.qhull_incremental)
                else:
                    raise

        return self.Tri


class LMap:
    def __init__(self, v):
        self.v = v
        self.x_l = None
        self.lres = None
        self.f_min = None
        self.lbounds = []


class LMapCache:
    def __init__(self):
        self.cache = {}

        # Lists for search queries
        self.v_maps = []
        self.xl_maps = []
        self.xl_maps_set = set()
        self.f_maps = []
        self.lbound_maps = []
        self.size = 0

    def __getitem__(self, v):
        try:
            v = np.ndarray.tolist(v)
        except TypeError:
            pass
        v = tuple(v)
        try:
            return self.cache[v]
        except KeyError:
            xval = LMap(v)
